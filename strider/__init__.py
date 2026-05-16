"""
strider — Nucleic Acid Thermodynamics, Kinetics, and Circuit Design

The biophysics layer below mantis-delta: sequences → thermodynamics →
TMSD kinetics → CRNetwork.

Quick start:
    from strider import ThermoEngine, CHABridge, LeakageEnumerator

    engine = ThermoEngine(material='dna', celsius=37, sodium=0.137, magnesium=0.01)

    bridge = CHABridge(
        sequences={'mirna': MIR21, 'H1': h1_seq, 'H2': h2_seq, 'CP': cp_seq},
        engine=engine,
    )
    report = bridge.verify()
    print(report)

    rn = bridge.to_crnetwork()   # → mantis CRNetwork
    rn.simulate(bridge._default_ic(), (0, 7200))
"""

from strider.thermo.engine import ThermoEngine, MFEResult, PFuncResult
from strider.thermo.nn_dna import (
    duplex_dg,
    melting_temperature,
    reverse_complement,
    is_self_complementary,
)
from strider.thermo.salt import owczarzy_tm_correction
from strider.thermo.modified import ModificationSite, apply_modifications

from strider.structure.mfe import fold_mfe
from strider.structure.dot_bracket import parse_pairs, to_dot_bracket, validate
from strider.structure.pseudoknot import fold_pseudoknot
from strider.structure.mountain import mountain_vector, compare_structures

from strider.kinetics.tmsd import (
    toehold_kf,
    displacement_kf,
    leakage_kf,
    rates_from_ddg,
    TMSDRateSet,
    TMSDKineticModel,
)
from strider.kinetics.arrhenius import (
    arrhenius,
    detailed_balance_kr,
    k_eq_from_ddg,
    ddg_from_k_eq,
)
from strider.kinetics.leakage import LeakageEnumerator, LeakageReport, SpuriousReaction

from strider.design.objective import DesignObjective
from strider.design.constraints import HardConstraint
from strider.design.optimizer import SequenceDesigner, DomainSpec, DesignResult
from strider.design.mutation import MutationAnalyzer, MutationProfile

from strider.screen.offtarget import OffTargetScreener, ScreeningReport

from strider.bridge.mantis_bridge import CHABridge, CHAVerificationReport, rates_to_crnetwork

from strider.sweep.cache import DiskCache
from strider.sweep.batch import ParameterSweep, SweepResult

from strider.export.formats import to_vienna, to_ct, to_bpseq, to_fasta, to_oxdna, write

__version__ = "0.1.0"
__author__ = "Emilio Venegas"
__license__ = "MIT"

__all__ = [
    # Core engine
    "ThermoEngine", "MFEResult", "PFuncResult",
    # Thermodynamics
    "duplex_dg", "melting_temperature", "reverse_complement", "is_self_complementary",
    "owczarzy_tm_correction",
    "ModificationSite", "apply_modifications",
    # Structure
    "fold_mfe", "fold_pseudoknot",
    "parse_pairs", "to_dot_bracket", "validate",
    "mountain_vector", "compare_structures",
    # Kinetics
    "toehold_kf", "displacement_kf", "leakage_kf", "rates_from_ddg",
    "TMSDRateSet", "TMSDKineticModel",
    "arrhenius", "detailed_balance_kr", "k_eq_from_ddg", "ddg_from_k_eq",
    "LeakageEnumerator", "LeakageReport", "SpuriousReaction",
    # Design
    "DesignObjective", "HardConstraint",
    "SequenceDesigner", "DomainSpec", "DesignResult",
    "MutationAnalyzer", "MutationProfile",
    # Screening
    "OffTargetScreener", "ScreeningReport",
    # Bridge to mantis
    "CHABridge", "CHAVerificationReport", "rates_to_crnetwork",
    # Sweep & cache
    "DiskCache", "ParameterSweep", "SweepResult",
    # Export
    "to_vienna", "to_ct", "to_bpseq", "to_fasta", "to_oxdna", "write",
]
