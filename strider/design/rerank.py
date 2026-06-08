"""
Re-rank design helper: pick the candidate context that wins on the *gate*.

A common design pattern: a quantity is optimized over candidate *contexts*
(e.g. domain-length splits, loop choices) using a cheap pre-rank proxy, but the
property you are actually judged on is only measurable *after* a downstream
domain is designed into that context.  Optimizing the proxy then trusting it can
pick a context that collapses once the downstream piece is attached.

:func:`design_with_rerank` closes that gap: it runs the (expensive) designer for
the top-N pre-ranked contexts and re-ranks them by a ``score_fn`` evaluated on
the *finished* design — the gate value, measured with the downstream domain in
place.  This generalizes urotrace's post-capture re-rank, where a split with a
healthy bare-hairpin R2 (−3.29) collapsed to +0.75 once the capture handle was
designed onto H1; the re-rank rejected it for one that holds.

Reuses :meth:`strider.design.optimizer.SequenceDesigner.design`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from strider.design.optimizer import SequenceDesigner, DesignResult


@dataclass
class RerankResult:
    """Winner of a re-ranked design sweep, plus the full score table."""
    context: Any                       # the chosen candidate context
    result: "DesignResult"             # its SequenceDesigner output
    score: float                       # downstream gate score (lower = better)
    all_scores: list[tuple[Any, float]] = field(default_factory=list)


def design_with_rerank(
    designer: "SequenceDesigner",
    contexts: list,
    build_problem: Callable[[Any], dict],
    score_fn: Callable[[Any, "DesignResult"], float],
    *,
    top_n: int = 3,
    verbose: bool = False,
    **design_kw,
) -> RerankResult:
    """Design the top-N contexts and return the one best on the downstream gate.

    Parameters
    ----------
    designer      : a configured :class:`SequenceDesigner`.
    contexts      : candidate contexts, **pre-ranked best-first** (only the first
                    ``top_n`` are designed).  A context is any opaque object the
                    callbacks understand (e.g. a ``(d1, d2, loop)`` tuple).
    build_problem : ``context -> dict`` with keys ``domains`` (``{name:
                    DomainSpec}``), ``objective`` (a :class:`DesignObjective`),
                    and optional ``hard_constraints`` — the design problem for
                    that context.
    score_fn      : ``(context, DesignResult) -> float``; lower is better.  This
                    is where the downstream gate is measured on the finished
                    design (e.g. the with-handle R2).
    top_n         : number of pre-ranked contexts to actually design.
    design_kw     : forwarded to :meth:`SequenceDesigner.design`
                    (``n_trials``, ``max_iterations``, …).

    Returns a :class:`RerankResult`.
    """
    chosen = list(contexts)[: max(1, top_n)]
    if not chosen:
        raise ValueError("design_with_rerank: no candidate contexts")

    best: RerankResult | None = None
    scored: list[tuple[Any, float]] = []
    for ctx in chosen:
        problem = build_problem(ctx)
        result = designer.design(
            domains=problem["domains"],
            objective=problem["objective"],
            hard_constraints=problem.get("hard_constraints"),
            verbose=verbose,
            **design_kw,
        )
        s = float(score_fn(ctx, result))
        scored.append((ctx, s))
        if verbose and len(chosen) > 1:
            print(f"  [rerank] context={ctx!r}  gate_score={s:.4f}")
        if best is None or s < best.score:
            best = RerankResult(context=ctx, result=result, score=s)

    assert best is not None
    best.all_scores = scored
    return best
