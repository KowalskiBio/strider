"""
Catalytic Hairpin Assembly (CHA) as a circuit template.

Same biophysics as :class:`strider.bridge.mantis_bridge.CHABridge`, but
implemented as a :class:`~strider.circuits.base.CircuitTemplate` so it
plugs into the generic verification framework.  Recommended for new code;
``CHABridge`` is retained as a thin compatibility shim.

References
----------
Yin P. et al. (2008) Nature 451:318-322.
"""

from __future__ import annotations

from dataclasses import dataclass

from strider.circuits.base import CircuitTemplate


@dataclass
class CHA(CircuitTemplate):
    """
    Catalytic Hairpin Assembly (4-reaction topology).

    Parameters
    ----------
    sequences   : dict with keys ``{"mirna", "H1", "H2", "CP"}``.
                  Renaming the trigger from ``mirna`` to anything else is
                  fine — pass it as ``mirna``.
    toehold_d1  : initiation toehold length (miRNA·H1), nt.
    toehold_d2  : branch-migration toehold length (H1·H2), nt.
    tail_cp     : capture-probe tail length (H1H2·CP), nt.
    """
    toehold_d1: int = 6
    toehold_d2: int = 11
    tail_cp:    int = 9

    # default synthesizability forbidden motifs (loop + handle domains)
    DEFAULT_FORBIDDEN = (
        "AAAA", "CCCC", "GGGG", "TTTT",
        "ATATAT", "TATATA", "CGCGCG", "GCGCGC",
    )

    def __post_init__(self):
        if self.name == "circuit":
            self.name = "CHA"
        if not self.reactions:
            self.reactions = [
                "mirna + H1 <-> mirna_H1",
                "mirna_H1 + H2 <-> H1_H2 + mirna",
                "H1_H2 + CP <-> H1_H2_CP",
                "H1 + H2 <-> H1_H2",
            ]
        if not self.toehold_map:
            self.toehold_map = {
                "mirna + H1 <-> mirna_H1":         self.toehold_d1,
                "mirna_H1 + H2 <-> H1_H2 + mirna": self.toehold_d2,
                "H1_H2 + CP <-> H1_H2_CP":         self.tail_cp,
            }

    def _default_checks(self):
        from strider.circuits.checks import (
            CheckRegistry, toehold_accessible, stability_in_range,
            reaction_driving_force, no_spurious_dimer, leakage_below_signal,
        )
        from strider.kinetics.tmsd import toehold_kf

        n_th = self.toehold_d1
        h1_seq = self.sequences.get("H1", "")
        access_positions = list(range(min(n_th, len(h1_seq))))
        # Compute signal forward rate once so the leakage check can compare.
        material = "rna" if any("U" in s for s in self.sequences.values()) else "dna"
        signal_kf = toehold_kf(self.toehold_d1, material, 37.0)

        return (CheckRegistry()
            .add(toehold_accessible("H1", access_positions, min_prob=0.5,
                                    name="toehold_accessible"))
            .add(stability_in_range("H1", -12, -4, name="H1_stability"))
            .add(stability_in_range("H2", -12, -4, name="H2_stability"))
            .add(reaction_driving_force(["mirna", "H1"], [["mirna", "H1"]],
                                        max_ddg=-3.0, name="R1_driving_force"))
            .add(reaction_driving_force([["mirna", "H1"], "H2"],
                                        [["H1", "H2"], "mirna"],
                                        max_ddg=-3.0, name="R2_driving_force"))
            .add(reaction_driving_force([["H1", "H2"], "CP"], [["H1", "H2", "CP"]],
                                        max_ddg=-8.0, name="R3_driving_force"))
            .add(no_spurious_dimer("H2", "CP", min_ddg=-6.0,
                                   name="CP_leakage"))
            .add(leakage_below_signal(signal_kf=signal_kf, hairpin="H1",
                                      ratio=1e-4, name="spontaneous_leakage"))
        )

    # ─── generator: target → assembled CHA ─────────────────────────────────────

    @classmethod
    def from_target(cls, target: str, *, d1_len: int, d2_len: int, loop: str,
                    capture: str = "", engine=None) -> "CHA":
        """
        Assemble a CHA from a target sequence and a chosen domain split.

        The target's 3′ ``d1_len`` nt become the initiation toehold D1, the next
        ``d2_len`` nt the branch-migration stem D2, ``loop`` the hairpin loop, and
        ``capture`` (if given) the orthogonal capture handle K on H1 (CP = K*).
        Returns a fully-populated :class:`CHA` ready for :meth:`verify` /
        :meth:`to_crnetwork`.  This is the inverse of the checker: it *builds* the
        strands a :class:`CHA` would otherwise be handed.
        """
        from strider.circuits import cha_architecture as arch

        dom = arch.build_domains(target, d1_len=d1_len, d2_len=d2_len,
                                 loop_h1=loop, loop_h2=loop, capture=capture)
        seqs = arch.assemble(dom)                 # {mirna, H1, H2, CP}
        if not seqs.get("CP"):
            seqs.pop("CP", None)                  # cascade-only: no probe yet
        cha = cls(sequences=seqs, engine=engine,
                  toehold_d1=d1_len, toehold_d2=d2_len,
                  tail_cp=len(capture) or cls.tail_cp)
        cha.domains = dom
        return cha

    @classmethod
    def design(cls, target: str, *, engine=None,
               d1_grid: tuple[int, ...] = (6, 7, 8),
               d2_grid: tuple[int, ...] = (11, 13, 15),
               loops: list[str] | None = None,
               loop_len: int = 12, capture_len: int = 20,
               r1_max: float = -3.0, r2_max: float = -3.0,
               toehold_min_prob: float = 0.5,
               stability_low: float = -12.0, stability_high: float = -4.0,
               orthogonality_thr: float = -3.0,
               w_orthogonality: float = 1.5, w_toehold: float = 1.0,
               w_tail_gc: float = 0.3, w_capture_r2: float = 2.0,
               gc_min: float = 0.35, gc_max: float = 0.65, max_run: int = 3,
               forbidden: tuple[str, ...] | None = None,
               rerank_top_n: int = 3, n_trials: int = 3,
               max_iterations: int = 120, seed: int = 0,
               verbose: bool = False) -> "CHA":
        """
        Goal-oriented CHA design from a target sequence.

        Two phases (ported from urotrace's design pipeline):

        1. **Split + loop selection.**  Scan ``d1_grid × d2_grid × loops`` and
           rank each (d1, d2, loop) combo by a cascade-gate penalty on the *bare*
           hairpins (R1/R2 driving forces, toehold accessibility, hairpin-
           stability sweet-spot).
        2. **Capture-handle design + post-capture re-rank.**  For the top
           ``rerank_top_n`` combos, design an orthogonal handle K with
           :class:`SequenceDesigner` (K/CP must not bind H2/target/H1-core) under
           a :meth:`DesignObjective.reaction_driving_force` term that *preserves*
           the with-handle R2, then re-rank the combos by the post-capture gate
           (R1/R2 measured with K attached) via
           :func:`strider.design.design_with_rerank` — the value validation
           actually gates on, not the bare proxy.

        Returns a populated :class:`CHA` (``.verify()`` / ``.to_crnetwork()``
        ready); design diagnostics are attached as ``.design_info``.
        """
        from strider.thermo.engine import ThermoEngine
        from strider.thermo.nn_dna import reverse_complement
        from strider.circuits import cha_architecture as arch
        from strider.design.objective import DesignObjective
        from strider.design.constraints import HardConstraint
        from strider.design.optimizer import SequenceDesigner, DomainSpec
        from strider.design.rerank import design_with_rerank

        engine = engine or ThermoEngine(material="dna", celsius=37.0,
                                        sodium=0.137, magnesium=0.01)
        target = arch.normalize(target)
        forbidden = forbidden if forbidden is not None else cls.DEFAULT_FORBIDDEN
        loops = loops or arch.loop_candidates(loop_len, forbidden, max_run)
        g_mir = engine.pfunc(target).free_energy

        def _norm_stab(g: float, length: int, ref: int = 20) -> float:
            return g * ref / length if length else g

        def _score_combo(d1: int, d2: int, loop: str):
            dom = arch.build_domains(target, d1_len=d1, d2_len=d2,
                                     loop_h1=loop, loop_h2=loop, capture="")
            s = arch.assemble(dom)
            H1, H2 = s["H1"], s["H2"]
            g_h1 = engine.pfunc(H1).free_energy
            g_h2 = engine.pfunc(H2).free_energy
            g_mir_h1 = engine.pfunc(target, H1).free_energy
            g_h1h2 = engine.pfunc(H1, H2).free_energy
            r1 = g_mir_h1 - g_mir - g_h1
            r2 = (g_h1h2 + g_mir) - (g_mir_h1 + g_h2)
            toeacc = engine.toehold_accessibility(H1, list(range(d1)))
            gh1n, gh2n = _norm_stab(g_h1, len(H1)), _norm_stab(g_h2, len(H2))
            pen = max(0.0, r1 - r1_max) ** 2 + 3.0 * max(0.0, r2 - r2_max) ** 2
            pen += w_toehold * max(0.0, toehold_min_prob - toeacc) ** 2 * 10
            for gn in (gh1n, gh2n):
                if gn < stability_low:
                    pen += (stability_low - gn) ** 2 * 0.2
                elif gn > stability_high:
                    pen += (gn - stability_high) ** 2 * 0.2
            return pen, {"R1": r1, "R2": r2, "toehold_acc": toeacc}

        combos = []
        for d1 in d1_grid:
            for d2 in d2_grid:
                if d1 + d2 > len(target):
                    continue
                for loop in loops:
                    pen, mt = _score_combo(d1, d2, loop)
                    combos.append(((d1, d2, loop), pen, mt))
        if not combos:
            raise ValueError(
                f"no feasible (d1,d2) split for target length {len(target)}")
        combos.sort(key=lambda t: t[1])
        contexts = [c[0] for c in combos]

        # cascade-only (no capture handle requested): return the best combo.
        if capture_len <= 0:
            d1, d2, loop = contexts[0]
            cha = cls.from_target(target, d1_len=d1, d2_len=d2, loop=loop,
                                  engine=engine)
            cha.design_info = {"context": (d1, d2, loop),
                               "split_metrics": combos[0][2]}
            return cha

        def _bind(a: str, b: str) -> float:
            return (engine.pfunc(a, b).free_energy
                    - engine.pfunc(a).free_energy - engine.pfunc(b).free_energy)

        def _build_problem(ctx):
            d1, d2, loop = ctx

            def _assemble_fn(seqs: dict[str, str]) -> dict[str, str]:
                dom = arch.build_domains(target, d1_len=d1, d2_len=d2,
                                         loop_h1=loop, loop_h2=loop,
                                         capture=seqs["K"])
                return arch.assemble(dom)

            def _orthogonality(seqs: dict[str, str]) -> float:
                K = seqs["K"]
                dom = arch.build_domains(target, d1_len=d1, d2_len=d2,
                                         loop_h1=loop, loop_h2=loop, capture=K)
                s = arch.assemble(dom)
                H2, core = s["H2"], arch.h1_core(dom)
                CP = reverse_complement(K)
                pen = 0.0
                for a, b in ((K, H2), (CP, H2), (K, target), (CP, target),
                             (K, core), (CP, core)):
                    pen += max(0.0, orthogonality_thr - _bind(a, b)) ** 2
                gc = sum(c in "GC" for c in K) / len(K) if K else 0.0
                return w_orthogonality * pen + w_tail_gc * (gc - 0.5) ** 2

            objective = DesignObjective.from_callable(
                _orthogonality, label="capture_orthogonality")
            if w_capture_r2 > 0.0:
                objective = objective + DesignObjective.reaction_driving_force(
                    engine, [["mirna", "H1"], "H2"], [["H1", "H2"], "mirna"],
                    max_ddg=r2_max, assemble_fn=_assemble_fn,
                    weight=w_capture_r2, label="R2_preserve")
            constraints = [
                HardConstraint.no_repeats(list(forbidden)),
                HardConstraint.gc_content(min_gc=gc_min, max_gc=gc_max),
                HardConstraint.max_run(max_run_length=max_run),
            ]
            return {"domains": {"K": DomainSpec(length=capture_len, material="dna")},
                    "objective": objective, "hard_constraints": constraints}

        def _post_capture_penalty(ctx, result) -> float:
            d1, d2, loop = ctx
            dom = arch.build_domains(target, d1_len=d1, d2_len=d2,
                                     loop_h1=loop, loop_h2=loop,
                                     capture=result.sequences["K"])
            s = arch.assemble(dom)
            g_h1 = engine.pfunc(s["H1"]).free_energy
            g_h2 = engine.pfunc(s["H2"]).free_energy
            g_mir_h1 = engine.pfunc(target, s["H1"]).free_energy
            g_h1h2 = engine.pfunc(s["H1"], s["H2"]).free_energy
            r1 = g_mir_h1 - g_mir - g_h1
            r2 = (g_h1h2 + g_mir) - (g_mir_h1 + g_h2)
            return (3.0 * max(0.0, r2 - r2_max) ** 2
                    + max(0.0, r1 - r1_max) ** 2 + 5.0 * max(0.0, r2) ** 2)

        designer = SequenceDesigner(engine=engine, seed=seed)
        rr = design_with_rerank(
            designer, contexts, _build_problem, _post_capture_penalty,
            top_n=rerank_top_n, n_trials=n_trials,
            max_iterations=max_iterations, verbose=verbose)

        d1, d2, loop = rr.context
        cha = cls.from_target(target, d1_len=d1, d2_len=d2, loop=loop,
                              capture=rr.result.sequences["K"], engine=engine)
        cha.design_info = {"context": rr.context,
                           "post_capture_penalty": rr.score,
                           "all_scores": rr.all_scores,
                           "objective_value": rr.result.objective_value}
        return cha
