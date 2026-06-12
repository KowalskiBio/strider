"""
Gradient-based sequence designer with a hybrid simulated-annealing hand-off.

Where :class:`~strider.design.optimizer.SequenceDesigner` explores discrete
sequence space by simulated annealing, :class:`DifferentiableDesigner` optimizes
a *continuous relaxation*: each free position is a point on the 4-simplex over
(A, C, G, U/T), parametrized by logits, and the whole sequence is folded by the
differentiable McCaskill engine so a :class:`~strider.thermo.diff_design.DiffObjective`
can be minimized by Adam.  A temperature schedule sharpens the per-position
distributions toward one-hot as optimization proceeds, after which the design is
rounded to a discrete sequence.

The two designers are complementary, so the default flow is **hybrid**: gradient
descent quickly finds a near-optimal continuous sequence (cheap, batched,
many restarts at once), it is rounded, and the existing SA ``SequenceDesigner``
runs a short *polish* warm-started from that rounding to clean up the
discretization gap and honour hard constraints exactly.  The return value is the
shared :class:`~strider.design.optimizer.DesignResult`, so this slots into the
existing design API.

Domains are concatenated (in dict order) into a single complex whose length must
match the objective's target structure; ``DomainSpec.sequence`` marks a fixed
domain (clamped to one-hot, never moved), and ``DomainSpec.length`` a free one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from strider.design.optimizer import (
    DomainSpec,
    DesignResult,
    SequenceDesigner,
)
from strider.thermo.differentiable import ThermoParameters, BatchedPartitionFunction
from strider.thermo.diff_design import DiffObjective

if TYPE_CHECKING:
    from strider.design.objective import DesignObjective
    from strider.thermo.engine import ThermoEngine

_RNA_BASES = "ACGU"
_DNA_BASES = "ACGT"
_BASE_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


@dataclass
class _Layout:
    """Concatenation layout of the designed complex."""
    order: list[str]
    bounds: dict[str, tuple[int, int]]   # name -> (start, stop) in the concat
    total: int
    material: str


class DifferentiableDesigner:
    """
    Continuous-relaxation sequence designer (Adam on simplex logits) with an
    optional SA polish hand-off.

    Parameters
    ----------
    material : 'rna' or 'dna' (used to build default ``ThermoParameters`` and the
               output alphabet).
    params   : optional pre-built / trained ``ThermoParameters`` (e.g. tables fit
               by ``thermo/train.py``) — designing against learned energies is
               something a closed engine cannot do.
    engine   : optional discrete ``ThermoEngine`` for the SA polish + final
               scoring; if ``None`` the polish is skipped and ranking uses the
               differentiable engine on the rounded (one-hot) sequences.
    seed     : RNG seed for reproducible initialization / polish.
    device   : torch device string (e.g. 'cuda').
    """

    def __init__(self, material: str = "rna",
                 params: ThermoParameters | None = None,
                 engine: "ThermoEngine | None" = None,
                 seed: int | None = None,
                 device: str | None = None) -> None:
        self.material = material
        self.params = params or ThermoParameters(material=material)
        if device is not None:
            self.params = self.params.to(device)
        self.model = BatchedPartitionFunction(self.params)
        self.engine = engine
        self.device = self.params.stack_table.device
        self.seed = seed

    # ── public API ───────────────────────────────────────────────────────────

    def design(
        self,
        domains: dict[str, DomainSpec],
        objective: DiffObjective,
        n_restarts: int = 8,
        n_steps: int = 300,
        lr: float = 0.2,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        sa_polish: bool = True,
        sa_objective: "DesignObjective | None" = None,
        sa_iterations: int = 200,
        sa_trials: int = 1,
        nicks: list[int] | None = None,
        verbose: bool = False,
    ) -> DesignResult:
        """
        Run gradient design and return the best :class:`DesignResult`.

        ``n_restarts`` independent designs are optimized in parallel as one batch
        (rows of the soft tensor); the temperature is annealed geometrically from
        ``tau_start`` to ``tau_end`` so distributions sharpen toward one-hot.
        After rounding, every restart is scored discretely and, when
        ``sa_polish`` and an ``engine`` are available, the best is refined by a
        short warm-started SA run against ``sa_objective`` (a discrete
        :class:`~strider.design.objective.DesignObjective`).

        ``nicks`` (cumulative strand lengths in the concatenated design) switches
        on multi-strand folding for the differentiable objective — design a
        complex (e.g. a duplex or a toehold-mediated displacement substrate) by
        passing the positions where one strand ends and the next begins.
        """
        gen = torch.Generator(device="cpu")
        if self.seed is not None:
            gen.manual_seed(self.seed)

        layout = self._layout(domains)
        free_mask, fixed_onehot = self._masks(domains, layout)

        logits = self._init_logits(n_restarts, layout, gen)
        opt = torch.optim.Adam([logits], lr=lr)

        for step in range(n_steps):
            tau = tau_start * (tau_end / tau_start) ** (step / max(n_steps - 1, 1))
            probs = self._to_probs(logits, free_mask, fixed_onehot, tau)
            loss = objective(probs, self.model, nicks=nicks).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if verbose and (step % max(n_steps // 10, 1) == 0 or step == n_steps - 1):
                print(f"  step {step:4d}  tau={tau:.3f}  loss={loss.item():.4f}")

        # Round each restart to a discrete sequence and score it.
        with torch.no_grad():
            final_probs = self._to_probs(logits, free_mask, fixed_onehot, tau_end)
        seqs_per_restart = self._round(final_probs, layout)
        best_seqs, best_score = self._select_best(seqs_per_restart, objective, layout, nicks)

        result_seqs = self._slice_domains(best_seqs, layout)
        breakdown = self._diff_breakdown(best_seqs, objective, nicks)
        n_iter = n_steps

        result = DesignResult(
            sequences=result_seqs,
            objective_value=best_score,
            objective_breakdown=breakdown,
            n_iterations=n_iter,
            converged=(best_score < 1e-2),
        )

        # Hybrid hand-off: short SA polish warm-started from the rounded design.
        if sa_polish and self.engine is not None and sa_objective is not None:
            polished = self._polish(
                domains, sa_objective, result_seqs, sa_iterations, sa_trials, verbose
            )
            if polished.objective_value <= sa_objective(result_seqs):
                result = polished
        elif sa_polish and (self.engine is None or sa_objective is None) and verbose:
            print("  [skip SA polish: needs both engine and sa_objective]")

        # The SA polish stores sequences T-form; present them in the design
        # material's alphabet (U for RNA) regardless of which path produced them.
        result.sequences = {n: self._to_alphabet(s) for n, s in result.sequences.items()}
        return result

    def _to_alphabet(self, seq: str) -> str:
        s = seq.upper()
        return s.replace("T", "U") if self.material == "rna" else s.replace("U", "T")

    # ── layout / masks ───────────────────────────────────────────────────────

    def _layout(self, domains: dict[str, DomainSpec]) -> _Layout:
        order, bounds, pos = [], {}, 0
        material = self.material
        for name, spec in domains.items():
            length = spec.length if spec.length else (len(spec.sequence) if spec.sequence else 0)
            if length == 0:
                raise ValueError(f"domain {name!r} has no length")
            bounds[name] = (pos, pos + length)
            pos += length
            order.append(name)
            material = spec.material
        return _Layout(order=order, bounds=bounds, total=pos, material=material)

    def _masks(self, domains: dict[str, DomainSpec], layout: _Layout):
        """Per-position free mask (1=free) and fixed one-hot tensor (N, 4)."""
        free = torch.ones(layout.total, 1, dtype=torch.float64, device=self.device)
        fixed = torch.zeros(layout.total, 4, dtype=torch.float64, device=self.device)
        for name, spec in domains.items():
            start, stop = layout.bounds[name]
            if spec.sequence is not None:
                free[start:stop, 0] = 0.0
                for k, ch in enumerate(spec.sequence.upper()):
                    fixed[start + k, _BASE_IDX.get(ch, 0)] = 1.0
        return free, fixed

    def _init_logits(self, n_restarts: int, layout: _Layout,
                     gen: torch.Generator) -> torch.Tensor:
        logits = 0.1 * torch.randn(n_restarts, layout.total, 4,
                                   dtype=torch.float64, generator=gen)
        return logits.to(self.device).requires_grad_(True)

    def _to_probs(self, logits: torch.Tensor, free_mask: torch.Tensor,
                  fixed_onehot: torch.Tensor, tau: float) -> torch.Tensor:
        """Softmax(logits/tau) on free positions, clamped one-hot on fixed ones."""
        soft = torch.softmax(logits / tau, dim=-1)
        return free_mask * soft + (1.0 - free_mask) * fixed_onehot

    # ── rounding / selection ─────────────────────────────────────────────────

    def _round(self, probs: torch.Tensor, layout: _Layout) -> list[str]:
        alphabet = _RNA_BASES if layout.material == "rna" else _DNA_BASES
        idx = probs.argmax(dim=-1)  # (B, N)
        return ["".join(alphabet[i] for i in row.tolist()) for row in idx]

    def _select_best(self, seqs: list[str], objective: DiffObjective,
                     layout: _Layout, nicks: list[int] | None) -> tuple[str, float]:
        """Score each rounded restart on the one-hot fold; return the best."""
        from strider.thermo.differentiable import seq_to_probs
        probs = seq_to_probs(seqs, material=layout.material, device=str(self.device))
        with torch.no_grad():
            scores = objective(probs, self.model, nicks=nicks)
        best = int(torch.argmin(scores))
        return seqs[best], float(scores[best])

    def _slice_domains(self, concat: str, layout: _Layout) -> dict[str, str]:
        out = {}
        for name in layout.order:
            start, stop = layout.bounds[name]
            out[name] = concat[start:stop]
        return out

    def _diff_breakdown(self, concat: str, objective: DiffObjective,
                        nicks: list[int] | None) -> dict[str, float]:
        from strider.thermo.differentiable import seq_to_probs
        probs = seq_to_probs([concat], material=self.material, device=str(self.device))
        return objective.evaluate_breakdown(probs, self.model, nicks=nicks)

    # ── SA polish hand-off ───────────────────────────────────────────────────

    def _polish(self, domains: dict[str, DomainSpec],
                sa_objective: "DesignObjective", warm: dict[str, str],
                iterations: int, trials: int, verbose: bool) -> DesignResult:
        """Short SA refinement warm-started from the rounded gradient design."""
        designer = SequenceDesigner(engine=self.engine, seed=self.seed)
        return designer.design(
            domains=domains,
            objective=sa_objective,
            n_trials=trials,
            max_iterations=iterations,
            initial_sequences=warm,
            verbose=verbose,
        )
