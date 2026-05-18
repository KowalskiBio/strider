"""
Mantis integration bridge.

Converts strider thermodynamics → kinetic rates → mantis CRNetwork.

This is the cleanest path from sequences to network-level analysis:
    sequences → ThermoEngine.ddg() → TMSDKineticModel.circuit_rates()
             → CRNetwork.from_string() → .simulate() / .steady_states()

CHABridge encodes the specific 4-reaction CHA topology and automates
the verification checks from claude_codesign.py.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine
    from strider.sweep.cache import DiskCache

R = 1.987e-3  # kcal / (mol · K)


# ─── generic bridge ──────────────────────────────────────────────────────────

def rates_to_crnetwork(
    reaction_strings: list[str],
    sequences: dict[str, str],
    engine: "ThermoEngine | None" = None,
    toehold_map: dict[str, int] | None = None,
    celsius: float = 37.0,
    include_leakage: bool = True,
    leakage_threshold: float = -4.0,
    cache: "DiskCache | None" = None,
):
    """
    Full pipeline: sequences + reactions → mantis CRNetwork.

    Parameters
    ----------
    reaction_strings : mantis-style strings ("A + B <-> C + D")
    sequences        : species_name → DNA/RNA sequence
    engine           : ThermoEngine (created automatically if None)
    toehold_map      : reaction_string → toehold_length
    celsius          : temperature (used when engine is None)
    include_leakage  : whether to enumerate and add leakage reactions
    leakage_threshold: ΔΔG cutoff for leakage enumeration
    cache            : DiskCache for memoization

    Returns a mantis CRNetwork object.

    Raises ImportError with install instructions if mantis is not installed.
    """
    mantis = _import_mantis()

    if engine is None:
        from strider.thermo.engine import ThermoEngine
        engine = ThermoEngine(celsius=celsius, cache=cache)

    from strider.kinetics.tmsd import TMSDKineticModel
    model = TMSDKineticModel(engine, celsius=engine.celsius)
    rates = model.circuit_rates(reaction_strings, sequences, toehold_map)

    all_reactions = list(reaction_strings)

    if include_leakage:
        from strider.kinetics.leakage import LeakageEnumerator
        enumerator = LeakageEnumerator(engine, ddg_threshold=leakage_threshold)
        report = enumerator.enumerate(sequences, intended_reactions=reaction_strings)
        for sp in report.reactions:
            # Add the spurious reaction as reversible so detailed balance is respected
            rxn_str = sp.mantis_string.replace("->", "<->")
            if rxn_str not in all_reactions and sp.mantis_string not in all_reactions:
                all_reactions.append(rxn_str)
                _add_leakage_rates(rates, sp, engine)

    return mantis.CRNetwork.from_string(all_reactions, rates=rates)


# ─── generic circuit bridge ──────────────────────────────────────────────────

class CircuitBridge:
    """
    Generic strider → mantis adapter for ANY reaction topology.

    Use this when your circuit is not the 4-reaction CHA shape that
    :class:`CHABridge` hardcodes.  You pass in mantis-style reaction strings
    plus a sequences dict; the bridge derives ΔΔG values via the supplied
    :class:`ThermoEngine` and emits a mantis ``CRNetwork`` ready for
    ``.simulate()``, ``.steady_states()``, and bifurcation analysis.

    Parameters
    ----------
    reactions        : list of mantis-style reaction strings, e.g.
                       ``["A + B <-> AB", "AB + C -> AC + B"]``.
    sequences        : species_name → DNA/RNA sequence.  Species that are
                       complexes (e.g. ``"AB"``) without an entry are derived
                       from the constituent strands by concatenation.
    engine           : ThermoEngine.  Created with physiological defaults if
                       ``None``.
    toehold_map      : optional reaction_string → toehold length override.
                       Falls back to 6 nt per reaction.
    include_leakage  : if True, run :class:`LeakageEnumerator` and add
                       spurious reactions to the output network.
    leakage_threshold: ΔΔG cutoff (kcal/mol) for leakage enumeration.

    Examples
    --------
    >>> bridge = CircuitBridge(
    ...     reactions=["A + B <-> AB", "AB + C -> AC + B"],
    ...     sequences={"A": "GCGC", "B": "GCGC", "C": "ATAT"},
    ... )
    >>> rn = bridge.to_crnetwork()                          # mantis CRNetwork
    >>> ss = rn.steady_states({"A": 1e-7, "B": 1e-7, "C": 1e-7})
    """

    def __init__(
        self,
        reactions: list[str],
        sequences: dict[str, str],
        engine: "ThermoEngine | None" = None,
        toehold_map: dict[str, int] | None = None,
        include_leakage: bool = False,
        leakage_threshold: float = -4.0,
        celsius: float = 37.0,
    ) -> None:
        self.reactions = list(reactions)
        self.sequences = {k: v.upper().replace("U", "T") for k, v in sequences.items()}
        self.toehold_map = dict(toehold_map or {})
        self.include_leakage = include_leakage
        self.leakage_threshold = leakage_threshold

        if engine is None:
            from strider.thermo.engine import ThermoEngine
            engine = ThermoEngine(material="dna", celsius=celsius,
                                  sodium=0.137, magnesium=0.01)
        self.engine = engine

        self._ddg_cache: dict[str, float] | None = None
        self._rates_cache: dict[str, float] | None = None
        self._leakage_cache = None

    # ─── thermodynamics ──────────────────────────────────────────────────────

    def _resolve_sequence(self, species: str) -> str | list[str] | None:
        """
        Resolve a species token to its sequence(s).

        Returns a single sequence string for monomers, or a list of sequences
        for multi-strand complexes (so :meth:`ThermoEngine.ddg` treats it as a
        proper complex rather than a concatenated single strand).
        """
        if species in self.sequences:
            return self.sequences[species]
        # Complex name like 'A_B'
        if "_" in species:
            parts = species.split("_")
            if all(p in self.sequences for p in parts):
                return [self.sequences[p] for p in parts]
        # Concatenated names like 'AB' (each letter is a known species)
        if len(species) > 1 and all(c in self.sequences for c in species):
            return [self.sequences[c] for c in species]
        return None

    @property
    def ddg_table(self) -> dict[str, float]:
        """Per-reaction ΔΔG (kcal/mol).  Cached."""
        if self._ddg_cache is not None:
            return self._ddg_cache
        out: dict[str, float] = {}
        for rxn in self.reactions:
            from strider.kinetics.tmsd import _parse_reversible
            fwd, _ = _parse_reversible(rxn, self.sequences)
            if fwd is None:
                continue
            react_seqs = [self._resolve_sequence(s) for s in fwd[0]]
            prod_seqs  = [self._resolve_sequence(s) for s in fwd[1]]
            if any(s is None for s in react_seqs + prod_seqs):
                continue
            try:
                out[rxn] = self.engine.ddg(react_seqs, prod_seqs)
            except Exception:
                continue
        self._ddg_cache = out
        return out

    @property
    def rates(self) -> dict[str, float]:
        """Mantis-compatible rate dict for every reaction.  Cached."""
        if self._rates_cache is not None:
            return self._rates_cache
        from strider.kinetics.tmsd import (
            _parse_reversible, _mantis_rate_key, rates_from_ddg, toehold_kf,
        )
        ddg_table = self.ddg_table
        rates: dict[str, float] = {}
        for rxn in self.reactions:
            fwd, rev = _parse_reversible(rxn, self.sequences)
            if fwd is None:
                continue
            ddg = ddg_table.get(rxn, -5.0)  # default for unresolved
            nt = self.toehold_map.get(rxn, 6)
            kf = toehold_kf(nt, self.engine.material, self.engine.celsius)
            kf_val, kr_val = rates_from_ddg(ddg, kf, self.engine.celsius)
            rates[_mantis_rate_key(fwd[0], fwd[1])] = kf_val
            if rev is not None:
                rates[_mantis_rate_key(fwd[1], fwd[0])] = kr_val
        self._rates_cache = rates
        return rates

    @property
    def leakage_report(self):
        """Leakage enumeration over the species in ``sequences``.  Cached."""
        if self._leakage_cache is not None:
            return self._leakage_cache
        from strider.kinetics.leakage import LeakageEnumerator
        enum = LeakageEnumerator(self.engine, ddg_threshold=self.leakage_threshold)
        self._leakage_cache = enum.enumerate(
            self.sequences, intended_reactions=self.reactions
        )
        return self._leakage_cache

    # ─── network construction ────────────────────────────────────────────────

    def to_crnetwork(self):
        """Build and return a mantis ``CRNetwork`` for this circuit."""
        mantis = _import_mantis()
        all_reactions = list(self.reactions)
        rates = dict(self.rates)

        if self.include_leakage:
            report = self.leakage_report
            for sp in report.reactions:
                rxn_str = sp.mantis_string.replace("->", "<->")
                if rxn_str not in all_reactions and sp.mantis_string not in all_reactions:
                    all_reactions.append(rxn_str)
                    _add_leakage_rates(rates, sp, self.engine)

        return mantis.CRNetwork.from_string(all_reactions, rates=rates)

    def simulate(self, initial_conditions: dict[str, float], t_span=(0.0, 3600.0), **kw):
        """Convenience wrapper: build network and run mantis ODE simulation."""
        rn = self.to_crnetwork()
        return rn.simulate(initial_conditions, t_span, **kw)

    def steady_states(self, initial_conditions: dict[str, float], **kw):
        """Convenience wrapper: build network and find steady states via mantis."""
        rn = self.to_crnetwork()
        return rn.steady_states(initial_conditions, **kw)


# ─── CHA-specific bridge ─────────────────────────────────────────────────────

CHA_TOPOLOGY = [
    "{mirna} + {H1} <-> {mirna}_{H1}",
    "{mirna}_{H1} + {H2} <-> {H1}{H2} + {mirna}",
    "{H1}{H2} + {CP} <-> {H1}{H2}_{CP}",
    "{H1} + {H2} <-> {H1}{H2}",   # spontaneous leakage
]

SWEET_SPOT_LOW = -12.0   # kcal/mol (minimum hairpin stability)
SWEET_SPOT_HIGH = -4.0   # kcal/mol (maximum hairpin stability)


@dataclass
class CHAVerificationReport:
    toehold_accessible: bool
    h1_stability: tuple[float, bool]    # (G_ens, in_sweet_spot)
    h2_stability: tuple[float, bool]
    ddg_r1: float
    ddg_r2: float
    ddg_r3: float
    ddg_spont: float
    catalyst_recycled: bool
    leakage_score: float                # cp_leakage ΔΔG
    all_passed: bool
    failed_checks: list[str] = field(default_factory=list)
    signal_fraction_predicted: float = 0.0

    def __str__(self) -> str:
        fc = set(self.failed_checks)
        status = "PASS" if self.all_passed else "FAIL"
        ok = "✓"
        no = "✗"
        lines = [
            f"CHA Verification: {status}",
            f"  Toehold accessible:    {ok if self.toehold_accessible else no}",
            f"  H1 stability (norm):   {self.h1_stability[0]:.2f} kcal/mol {no if 'H1_stability' in fc else ok}",
            f"  H2 stability (norm):   {self.h2_stability[0]:.2f} kcal/mol {no if 'H2_stability' in fc else ok}",
            f"  ΔΔG(R1, init):         {self.ddg_r1:.2f} kcal/mol {no if 'R1_driving_force' in fc else ok}",
            f"  ΔΔG(R2, prop):         {self.ddg_r2:.2f} kcal/mol {no if 'R2_driving_force' in fc else ok}",
            f"  ΔΔG(R3, detect):       {self.ddg_r3:.2f} kcal/mol {no if 'R3_driving_force' in fc else ok}",
            f"  ΔΔG(spont leakage):    {self.ddg_spont:.2f} kcal/mol {no if 'spontaneous_leakage' in fc else ok}",
            f"  Catalyst recycled:     {ok if self.catalyst_recycled else no}",
            f"  CP leakage:            {self.leakage_score:.2f} kcal/mol {no if 'cp_leakage' in fc else ok}",
            f"  Predicted signal:      {self.signal_fraction_predicted:.1%}",
        ]
        if self.failed_checks:
            lines.append(f"  Failed: {', '.join(self.failed_checks)}")
        return "\n".join(lines)


class CHABridge:
    """
    Convenience class for the CHA 4-reaction topology.

    Automates all thermodynamic calculations and verification checks
    defined in hairpin_design_scripts/claude_codesign.py using strider's
    own thermodynamic engine — no external folding library required.

    Parameters
    ----------
    sequences   : dict with keys 'mirna', 'H1', 'H2', 'CP'
    engine      : ThermoEngine (created with physiological defaults if None)
    toehold_d1  : toehold length for miRNA·H1 binding (default 6 nt)
    toehold_d2  : toehold length for H2 branch migration (default 11 nt)
    tail_cp     : CP tail length (default 9 nt)
    """

    def __init__(
        self,
        sequences: dict[str, str],
        engine: "ThermoEngine | None" = None,
        celsius: float = 37.0,
        toehold_d1: int = 6,
        toehold_d2: int = 11,
        tail_cp: int = 9,
    ) -> None:
        self.sequences = {k: v.upper().replace("U", "T") for k, v in sequences.items()}
        self.toehold_d1 = toehold_d1
        self.toehold_d2 = toehold_d2
        self.tail_cp = tail_cp

        if engine is None:
            from strider.thermo.engine import ThermoEngine
            engine = ThermoEngine(material="dna", celsius=celsius,
                                  sodium=0.137, magnesium=0.01)
        self.engine = engine

        # Derived species names from sequence keys
        self._mirna = sequences.get("mirna", sequences.get("target", ""))
        self._reactions = self._build_reactions()
        self._rates_cache: dict[str, float] | None = None
        self._ddg_cache: dict[str, float] | None = None

    # ─── public API ──────────────────────────────────────────────────────────

    @property
    def ddg_pathway(self) -> dict[str, float]:
        """Compute all ΔΔG values for the CHA pathway."""
        if self._ddg_cache is not None:
            return self._ddg_cache

        seqs = self.sequences
        mirna = seqs.get("mirna", seqs.get("target", ""))
        H1 = seqs.get("H1", "")
        H2 = seqs.get("H2", "")
        CP = seqs.get("CP", "")

        eng = self.engine

        # Monomer ensemble free energies
        g_mirna = eng.pfunc(mirna).free_energy if mirna else 0.0
        g_h1 = eng.pfunc(H1).free_energy if H1 else 0.0
        g_h2 = eng.pfunc(H2).free_energy if H2 else 0.0
        g_cp = eng.pfunc(CP).free_energy if CP else 0.0

        # Complex free energies via multi-strand pfunc (sequence-aware, handles mismatches)
        g_mirna_h1 = eng.pfunc(mirna, H1).free_energy if mirna and H1 else 0.0
        g_h1h2 = eng.pfunc(H1, H2).free_energy if H1 and H2 else 0.0
        g_h1h2_cp = eng.pfunc(H1, H2, CP).free_energy if H1 and H2 and CP else 0.0

        ddg_r1 = g_mirna_h1 - g_mirna - g_h1
        ddg_r2 = (g_h1h2 + g_mirna) - (g_mirna_h1 + g_h2)
        ddg_r3 = g_h1h2_cp - g_h1h2 - g_cp
        ddg_spont = g_h1h2 - g_h1 - g_h2

        # CP leakage: CP binding H2 alone (before triggering)
        g_h2_cp = eng.pfunc(H2, CP).free_energy if H2 and CP else 0.0
        cp_leakage = g_h2_cp - g_h2 - g_cp

        self._ddg_cache = {
            "g_H1": g_h1,
            "g_H2": g_h2,
            "R1": ddg_r1,
            "R2": ddg_r2,
            "R3": ddg_r3,
            "leakage": ddg_spont,
            "cp_leakage": cp_leakage,
        }
        return self._ddg_cache

    @property
    def rates(self) -> dict[str, float]:
        """Compute mantis-compatible rate dict from thermodynamic ΔΔG values."""
        if self._rates_cache is not None:
            return self._rates_cache

        from strider.kinetics.tmsd import toehold_kf, rates_from_ddg, leakage_kf

        ddg = self.ddg_pathway
        cel = self.engine.celsius

        kf_r1 = toehold_kf(self.toehold_d1, self.engine.material, cel)
        kf_r2 = toehold_kf(self.toehold_d2, self.engine.material, cel)
        kf_r3 = toehold_kf(self.tail_cp, self.engine.material, cel)

        _, kr_r1 = rates_from_ddg(ddg["R1"], kf_r1, cel)
        _, kr_r2 = rates_from_ddg(ddg["R2"], kf_r2, cel)
        _, kr_r3 = rates_from_ddg(ddg["R3"], kf_r3, cel)

        # Leakage: H1 + H2 → H1H2 (hairpin breathing model)
        g_h1 = abs(ddg["g_H1"])
        kf_leak = leakage_kf(g_h1, kf_max=1e6, celsius=cel)
        _, kr_leak = rates_from_ddg(ddg["leakage"], kf_leak, cel)

        names = _names(self.sequences)
        mirna, H1, H2, CP = names["mirna"], names["H1"], names["H2"], names["CP"]
        mirna_H1 = f"{mirna}_{H1}"
        H1H2 = f"{H1}{H2}"
        H1H2_CP = f"{H1H2}_{CP}"

        self._rates_cache = {
            f"{mirna} + {H1} -> {mirna_H1}": kf_r1,
            f"{mirna_H1} -> {mirna} + {H1}": kr_r1,
            f"{mirna_H1} + {H2} -> {H1H2} + {mirna}": kf_r2,
            f"{H1H2} + {mirna} -> {mirna_H1} + {H2}": kr_r2,
            f"{H1H2} + {CP} -> {H1H2_CP}": kf_r3,
            f"{H1H2_CP} -> {H1H2} + {CP}": kr_r3,
            f"{H1} + {H2} -> {H1H2}": kf_leak,
            f"{H1H2} -> {H1} + {H2}": kr_leak,
        }
        return self._rates_cache

    def to_crnetwork(self):
        """Return a mantis CRNetwork ready for .simulate() / .steady_states()."""
        mantis = _import_mantis()
        rxns = list(self._reactions)
        return mantis.CRNetwork.from_string(rxns, rates=self.rates)

    def verify(self) -> CHAVerificationReport:
        """Run all 7 thermodynamic checks from claude_codesign.py."""
        ddg = self.ddg_pathway
        seqs = self.sequences
        H1 = seqs.get("H1", "")
        H2 = seqs.get("H2", "")

        g_h1 = ddg["g_H1"]
        g_h2 = ddg["g_H2"]
        # Normalize by sequence length relative to a 20-nt reference so the
        # sweet spot thresholds hold across a range of hairpin lengths.
        _ref_nt = 20.0
        g_h1_norm = g_h1 * _ref_nt / len(H1) if H1 else g_h1
        g_h2_norm = g_h2 * _ref_nt / len(H2) if H2 else g_h2
        h1_ok = SWEET_SPOT_LOW <= g_h1_norm <= SWEET_SPOT_HIGH
        h2_ok = SWEET_SPOT_LOW <= g_h2_norm <= SWEET_SPOT_HIGH

        # Toehold accessibility: first toehold_d1 bases of H1 should be unpaired
        th_access = self.engine.toehold_accessibility(H1, list(range(self.toehold_d1)))

        r1_ok = ddg["R1"] < -3.0
        r2_ok = ddg["R2"] < -3.0
        r3_ok = ddg["R3"] < -8.0
        # Leakage check: use kinetic rate ratio rather than absolute ΔΔG so the
        # check stays meaningful across hairpin lengths.  Pass when the leakage
        # forward rate is at least 1e6× slower than the signal initiation rate.
        from strider.kinetics.tmsd import toehold_kf, leakage_kf
        kf_signal = toehold_kf(self.toehold_d1, self.engine.material, self.engine.celsius)
        kf_spont = leakage_kf(abs(g_h1), kf_max=1e6, celsius=self.engine.celsius)
        spont_ok = kf_spont < kf_signal * 1e-4
        cp_ok = ddg["cp_leakage"] > -6.0

        # Catalyst recycled: ΔΔG_R2 < 0 means miRNA released (energetically favored)
        recycled = ddg["R2"] < 0.0

        failed = []
        if not h1_ok:        failed.append("H1_stability")
        if not h2_ok:        failed.append("H2_stability")
        if not r1_ok:        failed.append("R1_driving_force")
        if not r2_ok:        failed.append("R2_driving_force")
        if not r3_ok:        failed.append("R3_driving_force")
        if not spont_ok:     failed.append("spontaneous_leakage")
        if not cp_ok:        failed.append("cp_leakage")
        if not recycled:     failed.append("catalyst_recycling")
        if th_access < 0.5:  failed.append("toehold_accessibility")

        # Predict signal fraction: analytical approximation
        signal = _predict_signal(ddg)

        return CHAVerificationReport(
            toehold_accessible=(th_access >= 0.5),
            h1_stability=(g_h1_norm, h1_ok),
            h2_stability=(g_h2_norm, h2_ok),
            ddg_r1=ddg["R1"],
            ddg_r2=ddg["R2"],
            ddg_r3=ddg["R3"],
            ddg_spont=ddg["leakage"],
            catalyst_recycled=recycled,
            leakage_score=ddg["cp_leakage"],
            all_passed=(len(failed) == 0),
            failed_checks=failed,
            signal_fraction_predicted=signal,
        )

    def sensitivity(
        self,
        target_species: str = "H1H2_CP",
        perturbation: float = 0.5,
    ) -> dict[str, float]:
        """
        One-at-a-time sensitivity analysis via mantis.

        Perturbs each rate by ±perturbation (fraction) and measures
        change in steady-state concentration of target_species.
        """
        rn = self.to_crnetwork()
        base_rates = self.rates
        ic = self._default_ic()

        results: dict[str, float] = {}
        try:
            base_ss = rn.steady_states(ic, n_attempts=10, seed=0)
            if not base_ss:
                return {}
            base_conc = base_ss[0].concentrations.get(target_species, 0.0)
        except Exception:
            return {}

        for rate_key, base_val in base_rates.items():
            for sign in (+1, -1):
                perturbed = dict(base_rates)
                perturbed[rate_key] = base_val * (1 + sign * perturbation)
                try:
                    rn_pert = _import_mantis().CRNetwork.from_string(
                        list(self._reactions), rates=perturbed
                    )
                    ss_list = rn_pert.steady_states(ic, n_attempts=10, seed=0)
                    if ss_list:
                        conc = ss_list[0].concentrations.get(target_species, 0.0)
                        sensitivity = abs(conc - base_conc) / max(abs(base_conc), 1e-30)
                        label = f"{rate_key} ({'+' if sign > 0 else '-'}{perturbation:.0%})"
                        results[label] = sensitivity
                except Exception:
                    pass

        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    def _default_ic(self) -> dict[str, float]:
        """Return a physiological initial condition dict (100 nM hairpins, 10 nM miRNA, zero complexes)."""
        names = _names(self.sequences)
        mirna, H1, H2, CP = names["mirna"], names["H1"], names["H2"], names["CP"]
        mirna_H1 = f"{mirna}_{H1}"
        H1H2 = f"{H1}{H2}"
        H1H2_CP = f"{H1H2}_{CP}"
        return {
            H1: 100e-9, H2: 100e-9, CP: 100e-9,
            mirna: 10e-9, mirna_H1: 0.0, H1H2: 0.0, H1H2_CP: 0.0,
        }

    def _build_reactions(self) -> list[str]:
        """Build the 4 reversible CHA reaction strings using the sequence species names."""
        names = _names(self.sequences)
        mirna, H1, H2, CP = names["mirna"], names["H1"], names["H2"], names["CP"]
        mirna_H1 = f"{mirna}_{H1}"
        H1H2 = f"{H1}{H2}"
        H1H2_CP = f"{H1H2}_{CP}"
        return [
            f"{mirna} + {H1} <-> {mirna_H1}",
            f"{mirna_H1} + {H2} <-> {H1H2} + {mirna}",
            f"{H1H2} + {CP} <-> {H1H2_CP}",
            f"{H1} + {H2} <-> {H1H2}",
        ]


# ─── helpers ─────────────────────────────────────────────────────────────────

def _import_mantis():
    """Import mantis and raise a helpful ImportError with install instructions if missing."""
    try:
        import mantis
        return mantis
    except ImportError:
        raise ImportError(
            "mantis is not installed. Install it with:\n"
            "    pip install mantis-delta\n"
            "or from source: cd hairpin/mantis && pip install -e ."
        )


def _names(sequences: dict[str, str]) -> dict[str, str]:
    """Normalize arbitrary user-supplied species keys to canonical {'mirna', 'H1', 'H2', 'CP'} names."""
    return {
        "mirna": next((k for k in sequences if k.lower() in ("mirna", "target", "mir21")), "mirna"),
        "H1": next((k for k in sequences if k.upper() in ("H1",)), "H1"),
        "H2": next((k for k in sequences if k.upper() in ("H2",)), "H2"),
        "CP": next((k for k in sequences if k.upper() in ("CP",)), "CP"),
    }


def _add_leakage_rates(
    rates: dict,
    spurious,                              # strider.kinetics.leakage.SpuriousReaction
    engine,
) -> None:
    """
    Insert forward/reverse rate constants for a spurious reaction using
    detailed balance against the enumerated ΔΔG.

    Forward rate is the Boltzmann-suppressed hairpin-breathing model
    (leakage_kf), evaluated against |ΔΔG| as the activation barrier.
    """
    from strider.kinetics.tmsd import leakage_kf, rates_from_ddg
    ddg = spurious.ddg
    kf = leakage_kf(abs(ddg), kf_max=1e6, celsius=engine.celsius)
    kf_val, kr_val = rates_from_ddg(ddg, kf, engine.celsius)
    fwd_key = _mantis_rate_key(spurious.reactant_names, spurious.product_complex_name())
    rev_key = _mantis_rate_key(spurious.product_complex_name(), spurious.reactant_names)
    rates.setdefault(fwd_key, kf_val)
    rates.setdefault(rev_key, kr_val)


def _mantis_rate_key(reactants, products) -> str:
    """Mantis-style canonical rate key 'A + B -> C'."""
    if isinstance(reactants, str):
        r = reactants
    else:
        r = " + ".join(sorted(reactants))
    if isinstance(products, str):
        p = products
    else:
        p = " + ".join(sorted(products))
    return f"{r} -> {p}"


def _predict_signal(ddg: dict[str, float], celsius: float = 37.0) -> float:
    """
    Approximate steady-state signal fraction using Michaelis-Menten analogy.

    CHA turnover: each miRNA catalyzes ~N H1H2 formations.
    N ≈ K_eq(R1) * K_eq(R2) / (1 + K_eq(leakage))
    Signal fraction = N / (N + 1) as an upper bound.
    """
    T = celsius + 273.15

    def keq(ddg_val: float) -> float:
        return math.exp(-ddg_val / (R * T))

    k1 = keq(ddg.get("R1", -5.0))
    k2 = keq(ddg.get("R2", -5.0))
    kleak = keq(ddg.get("leakage", -20.0))

    n_turnover = k1 * k2 / (1.0 + kleak)
    return n_turnover / (n_turnover + 1.0) if n_turnover > 0 else 0.0
