"""
CircuitChecks — pluggable verification framework for DNA/RNA circuits.

Every circuit template exposes a ``verify()`` method that runs a list of
named checks against the current sequences and reports pass/fail.  This
module defines the check protocol and a handful of widely-useful checks
(toehold accessibility, hairpin stability sweet-spot, leakage thresholds,
off-target binding, custom user predicates).

Each check is a callable
``(ctx: CheckContext) → CheckResult``
where ``ctx`` exposes the engine, the sequences dict, and a structured
view of the circuit being verified.  The result carries a name, a pass /
fail flag, a numeric value (whatever the check measured), and a message.

This replaces the ad-hoc 7-criterion logic that was previously hardcoded
in :class:`strider.bridge.mantis_bridge.CHABridge.verify`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine


@dataclass
class CheckContext:
    """Shared state passed to every check function."""
    engine: "ThermoEngine"
    sequences: dict[str, str]
    metadata: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    """Outcome of a single named check."""
    name: str
    passed: bool
    value: float | None = None
    message: str = ""
    unit: str = ""

    def __str__(self) -> str:
        mark = "✓" if self.passed else "✗"
        val = f" {self.value:.3f} {self.unit}".rstrip() if self.value is not None else ""
        msg = f" — {self.message}" if self.message else ""
        return f"  {mark} {self.name}:{val}{msg}"


@dataclass
class CircuitReport:
    """Aggregated verification report for a circuit."""
    name: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_checks(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{self.name}: {status}"]
        for r in self.results:
            lines.append(str(r))
        if not self.passed:
            lines.append(f"  Failed: {', '.join(self.failed_checks)}")
        return "\n".join(lines)


# A check is just a callable.  Defining it as a Protocol lets us register
# bound methods, lambdas, or free functions interchangeably.
Check = Callable[[CheckContext], CheckResult]


class CheckRegistry:
    """
    Holds an ordered list of named checks.  Used by circuit templates to
    declare their default verification suite, but users can also build a
    registry by hand and call :meth:`run` against any circuit.

    Example
    -------
    >>> registry = CheckRegistry()
    >>> registry.add(toehold_accessible("input_strand", positions=range(6)))
    >>> registry.add(stability_in_range("H1", min_dg=-12, max_dg=-4))
    >>> report = registry.run(engine, {"input_strand": "...", "H1": "..."}, name="my_circuit")
    """

    def __init__(self) -> None:
        self._checks: list[Check] = []

    def add(self, check: Check) -> "CheckRegistry":
        self._checks.append(check)
        return self

    def extend(self, checks: list[Check]) -> "CheckRegistry":
        self._checks.extend(checks)
        return self

    def run(
        self,
        engine: "ThermoEngine",
        sequences: dict[str, str],
        *,
        name: str = "circuit",
        metadata: dict | None = None,
    ) -> CircuitReport:
        ctx = CheckContext(engine=engine, sequences=sequences,
                           metadata=metadata or {})
        report = CircuitReport(name=name)
        for check in self._checks:
            try:
                report.results.append(check(ctx))
            except Exception as e:
                report.results.append(CheckResult(
                    name=getattr(check, "__check_name__", check.__name__),
                    passed=False,
                    message=f"check raised {type(e).__name__}: {e}",
                ))
        return report


# ─── built-in checks ─────────────────────────────────────────────────────────


def toehold_accessible(
    strand: str,
    positions,
    min_prob: float = 0.5,
    name: str | None = None,
) -> Check:
    """Strand's toehold positions are unpaired ≥ ``min_prob`` of the ensemble."""
    label = name or f"toehold_accessible({strand})"
    pos_list = list(positions)

    def check(ctx: CheckContext) -> CheckResult:
        seq = ctx.sequences.get(strand)
        if seq is None:
            return CheckResult(label, False, None, f"strand {strand!r} missing")
        prob = ctx.engine.toehold_accessibility(seq, pos_list)
        return CheckResult(label, prob >= min_prob, prob,
                          f"unpaired probability {prob:.2f} (≥ {min_prob:.2f})",
                          unit="(prob)")

    check.__check_name__ = label
    return check


def stability_in_range(
    strand: str,
    min_dg: float = -12.0,
    max_dg: float = -4.0,
    reference_length: int = 20,
    name: str | None = None,
) -> Check:
    """
    Hairpin ensemble ΔG (normalized to a reference length) falls within the
    [min_dg, max_dg] sweet spot, e.g. [-12, -4] kcal/mol — stable enough to
    suppress leakage but not so stable that toeholds get buried.
    """
    label = name or f"stability_in_range({strand})"

    def check(ctx: CheckContext) -> CheckResult:
        seq = ctx.sequences.get(strand)
        if seq is None:
            return CheckResult(label, False, None, f"strand {strand!r} missing")
        dg = ctx.engine.pfunc(seq).free_energy
        dg_norm = dg * reference_length / len(seq) if seq else dg
        ok = min_dg <= dg_norm <= max_dg
        return CheckResult(label, ok, dg_norm,
                          f"normalised ΔG {dg_norm:.2f} kcal/mol "
                          f"({'in' if ok else 'OUT of'} [{min_dg}, {max_dg}])",
                          unit="kcal/mol")

    check.__check_name__ = label
    return check


def reaction_driving_force(
    reactants: list[str],
    products: list[str],
    max_ddg: float = -3.0,
    name: str | None = None,
) -> Check:
    """ΔΔG(reactants → products) is at least ``max_ddg`` favorable."""
    label = name or f"ddG({'+'.join(reactants)} → {'+'.join(products)})"

    def check(ctx: CheckContext) -> CheckResult:
        try:
            r = [_resolve(ctx.sequences, n) for n in reactants]
            p = [_resolve(ctx.sequences, n) for n in products]
        except KeyError as e:
            return CheckResult(label, False, None, f"missing strand {e}")
        ddg = ctx.engine.ddg(r, p)
        return CheckResult(label, ddg <= max_ddg, ddg,
                          f"ΔΔG {ddg:.2f} kcal/mol (≤ {max_ddg})",
                          unit="kcal/mol")

    check.__check_name__ = label
    return check


def no_spurious_dimer(
    a: str,
    b: str,
    min_ddg: float = -6.0,
    name: str | None = None,
) -> Check:
    """The (a, b) dimer must NOT form too strongly (ΔΔG above ``min_ddg``)."""
    label = name or f"no_spurious_dimer({a}, {b})"

    def check(ctx: CheckContext) -> CheckResult:
        sa = ctx.sequences.get(a)
        sb = ctx.sequences.get(b)
        if sa is None or sb is None:
            return CheckResult(label, False, None, f"missing strand(s) for {a}/{b}")
        ddg = ctx.engine.ddg([sa, sb], [[sa, sb]])
        return CheckResult(label, ddg >= min_ddg, ddg,
                          f"ΔΔG {ddg:.2f} kcal/mol (≥ {min_ddg})",
                          unit="kcal/mol")

    check.__check_name__ = label
    return check


def leakage_below_signal(
    signal_kf: float,
    leakage_kf_max: float = 1e6,
    hairpin: str | None = None,
    ratio: float = 1e-4,
    name: str | None = None,
) -> Check:
    """
    Spontaneous hairpin-breathing leakage is at least ``ratio`` times slower
    than the intended (toehold-mediated) signal forward rate.
    """
    label = name or "leakage_below_signal"

    def check(ctx: CheckContext) -> CheckResult:
        from strider.kinetics.tmsd import leakage_kf as kf_fn
        if hairpin is None:
            return CheckResult(label, False, None, "no hairpin given")
        seq = ctx.sequences.get(hairpin)
        if seq is None:
            return CheckResult(label, False, None, f"hairpin {hairpin!r} missing")
        g = abs(ctx.engine.pfunc(seq).free_energy)
        kf_leak = kf_fn(g, kf_max=leakage_kf_max, celsius=ctx.engine.celsius)
        rel = kf_leak / signal_kf if signal_kf > 0 else float("inf")
        ok = kf_leak <= signal_kf * ratio
        return CheckResult(label, ok, rel,
                          f"leak/signal = {rel:.2e} (≤ {ratio:.0e})",
                          unit="ratio")

    check.__check_name__ = label
    return check


def custom(
    name: str,
    fn: Callable[[CheckContext], tuple[bool, float | None, str]],
) -> Check:
    """Wrap an arbitrary callable as a check (returns ``(passed, value, msg)``)."""
    def check(ctx: CheckContext) -> CheckResult:
        passed, value, msg = fn(ctx)
        return CheckResult(name, passed, value, msg)
    check.__check_name__ = name
    return check


# ─── helpers ──────────────────────────────────────────────────────────────────

def _resolve(seqs: dict[str, str], token) -> str | list[str]:
    """Look up a token in seqs.

    ``token`` may be a single strand name (``"H1"``), an explicit list of
    strand names (``["H1", "H2"]``) for a multi-strand complex, or an
    underscore-joined complex name (``"H1_H2"``).
    """
    if isinstance(token, (list, tuple)):
        return [_resolve(seqs, t) for t in token]
    if token in seqs:
        return seqs[token]
    if "_" in token:
        parts = token.split("_")
        if all(p in seqs for p in parts):
            return [seqs[p] for p in parts]
    raise KeyError(token)
