"""
Base class for DSD circuit templates.

A circuit template combines:
  * a strand set (sequences keyed by species name)
  * a reaction topology (mantis-style reaction strings)
  * an optional toehold-length map
  * a default check registry for ``verify()``

Templates produce a :class:`~strider.bridge.mantis_bridge.CircuitBridge`
through ``to_bridge()`` so they integrate with mantis exactly like the
:class:`CHABridge` does, but the topology is configurable.

This is the parent class for :class:`HCR`, :class:`SeesawGate`,
:class:`Translator`, and (after the refactor) :class:`CHABridge`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strider.thermo.engine import ThermoEngine
    from strider.bridge.mantis_bridge import CircuitBridge
    from strider.circuits.checks import CheckRegistry, CircuitReport


@dataclass
class CircuitTemplate:
    """
    Shared base class for all built-in circuit templates.

    Subclasses set :attr:`name`, populate :attr:`sequences`, :attr:`reactions`,
    optional :attr:`toehold_map`, and override :meth:`_default_checks` to
    declare their verification suite.
    """
    name: str = "circuit"
    sequences: dict[str, str] = field(default_factory=dict)
    reactions: list[str] = field(default_factory=list)
    toehold_map: dict[str, int] = field(default_factory=dict)
    engine: "ThermoEngine | None" = None

    def to_bridge(
        self,
        include_leakage: bool = False,
        leakage_threshold: float = -4.0,
    ) -> "CircuitBridge":
        """Return a :class:`CircuitBridge` consuming this template's data."""
        from strider.bridge.mantis_bridge import CircuitBridge
        return CircuitBridge(
            reactions=list(self.reactions),
            sequences=dict(self.sequences),
            engine=self.engine,
            toehold_map=dict(self.toehold_map),
            include_leakage=include_leakage,
            leakage_threshold=leakage_threshold,
        )

    def to_crnetwork(self, **kw):
        """Shortcut: build the bridge and emit a mantis ``CRNetwork``."""
        return self.to_bridge(**kw).to_crnetwork()

    def simulate(self, initial_conditions, t_span, **kw):
        return self.to_bridge().simulate(initial_conditions, t_span, **kw)

    def steady_states(self, initial_conditions, **kw):
        return self.to_bridge().steady_states(initial_conditions, **kw)

    # ─── verification ────────────────────────────────────────────────────────

    def verify(self, registry: "CheckRegistry | None" = None) -> "CircuitReport":
        """Run the circuit's default checks (or a user-supplied registry)."""
        from strider.thermo.engine import ThermoEngine
        eng = self.engine or ThermoEngine(material="dna", celsius=37.0,
                                         sodium=0.137, magnesium=0.01)
        reg = registry if registry is not None else self._default_checks()
        return reg.run(eng, self.sequences, name=self.name)

    def _default_checks(self) -> "CheckRegistry":
        """Override to declare a template-specific check suite."""
        from strider.circuits.checks import CheckRegistry
        return CheckRegistry()
