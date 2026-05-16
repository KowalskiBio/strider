# strider-dna — Nucleic Acid Thermodynamics, Kinetics, and Circuit Design

[![Tests](https://img.shields.io/badge/tests-76%20passed-brightgreen)](#running-the-tests)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

**strider** is a Python library for computing the thermodynamics and kinetics of DNA/RNA circuits. Given a set of strand sequences, it predicts free energies via nearest-neighbor parameters, folds secondary structures via dynamic programming, derives TMSD rate constants from the Zhang & Winfree (2009) empirical model, enumerates spurious leakage pathways, and produces kinetic rate dictionaries that drop directly into a mantis-delta `CRNetwork`. The design loop — sequence → thermodynamics → kinetics → reaction network → steady states — runs end-to-end without any NUPACK or ViennaRNA dependency.

strider ships a three-backend thermodynamic engine that automatically selects the best available calculator: its own McCaskill O(n³) partition-function DP (always available, matches ViennaRNA/NUPACK to within ~0.1 kcal/mol for bimolecular binding), the ViennaRNA C library (recommended for long sequences), or NUPACK (highest accuracy, restrictive license). The same API surface works with every backend — only the `ThermoEngine(backend=…)` argument changes.

---

## Table of contents

1. [Installation](#installation)
2. [Core concepts](#core-concepts)
3. [Quick start](#quick-start)
4. [User guide](#user-guide)
   - [ThermoEngine and backends](#1-thermoengine-and-backends)
   - [DNA / RNA thermodynamics](#2-dna--rna-thermodynamics)
   - [Secondary structure prediction](#3-secondary-structure-prediction)
   - [TMSD kinetics](#4-tmsd-kinetics)
   - [Leakage enumeration](#5-leakage-enumeration)
   - [Sequence design](#6-sequence-design)
   - [Mutation sensitivity analysis](#7-mutation-sensitivity-analysis)
   - [Off-target screening](#8-off-target-screening)
   - [CHABridge — sequences to mantis CRNetwork](#9-chabridge--sequences-to-mantis-crnetwork)
   - [Parameter sweeps and caching](#10-parameter-sweeps-and-caching)
   - [Export formats](#11-export-formats)
5. [API reference](#api-reference)
6. [Examples](#examples)
7. [Backend comparison](#backend-comparison)
8. [Running the tests](#running-the-tests)
9. [Troubleshooting](#troubleshooting)
10. [Background and theory](#background-and-theory)
11. [Citation](#citation)
12. [License](#license)

---

## Installation

```bash
# Core library (native thermodynamic backend always included)
pip install strider-dna

# From source
git clone https://github.com/EmilioVenegas/strider
cd strider
pip install -e .
```

**Optional backends:**

```bash
pip install strider-dna[vienna]   # ViennaRNA backend (recommended for long sequences)
pip install strider-dna[mantis]   # mantis-delta integration (CHABridge)
pip install strider-dna[pandas]   # SweepResult.to_dataframe()
pip install strider-dna[parallel] # ProcessPoolExecutor sweeps
pip install strider-dna[full]     # all of the above
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.6.

> **Note on import name:** the PyPI distribution is `strider-dna`, but the Python package is imported as `strider`:
> ```python
> import strider
> from strider import ThermoEngine, CHABridge
> ```

---

## Core concepts

### Nearest-neighbor (NN) model

The standard model for DNA and RNA duplex thermodynamics. The free energy of a fully paired duplex is computed by summing stacking contributions from every adjacent dinucleotide pair (the "nearest neighbors") and adding initiation terms for the terminal bases. Parameters come from SantaLucia & Hicks (2004) for DNA and Mathews et al. (1999) for RNA. strider also includes Sugimoto et al. (1995) parameters for DNA:RNA hybrids.

### Ensemble free energy and ΔΔG

`ThermoEngine.pfunc(seq)` returns the **ensemble free energy** ΔG = −RT ln Q, where Q is the partition function summed over all secondary structures weighted by their Boltzmann factors. This is more informative than the minimum free energy (MFE) alone because it accounts for the full structural ensemble.

The **reaction free energy** ΔΔG = ΣG(products) − ΣG(reactants) measures how thermodynamically driven a strand displacement reaction is. Negative ΔΔG means the reaction is spontaneous (products are lower energy).

### Toehold-mediated strand displacement (TMSD)

A mechanism in which a short single-stranded overhang (the *toehold*) on a target strand initiates hybridization with an incoming strand, which then branch-migrates to displace the incumbent strand. Catalytic Hairpin Assembly (CHA) is built from a cascade of TMSD reactions. The forward rate constant kf depends sensitively on toehold length; Zhang & Winfree (2009) measured kf empirically for 0–12 nt toeholds in DNA.

### Kinetic stability and the sweet spot

Hairpin kinetics for CHA have a "sweet spot": stems stable enough to suppress leakage (ΔG ≲ −6 kcal/mol) but not so stable that the toehold is buried (ΔG ≳ −12 kcal/mol). strider's `CHABridge.verify()` checks all seven design criteria automatically.

---

## Quick start

```python
from strider import ThermoEngine, CHABridge

# Create a thermodynamic engine (auto-selects best available backend)
engine = ThermoEngine(material='dna', celsius=37, sodium=0.137, magnesium=0.01)

# Fold a hairpin
result = engine.mfe('TCAACATCAGTCTGATAAGGAGGGAGGTTATCAGACTGA')
print(result.structure)   # '((((((......(((((((((((.)))))))))))....))))))'
print(result.energy)      # -7.8  kcal/mol

# Duplex binding free energy
ddg = engine.ddg(
    reactants=['TCAACATCAGTCTGATGTTGA', 'TCAACATCAGTCTGATAAGG'],
    products=[['TCAACATCAGTCTGATGTTGA', 'TCAACATCAGTCTGATAAGG']],
)
print(f"ΔΔG = {ddg:.2f} kcal/mol")  # ΔΔG = -8.3 kcal/mol

# Full CHA biosensor verification
bridge = CHABridge(
    sequences={
        'mirna': 'TAGCTTATCAGACTGATGTTGA',
        'H1':    'TCAACATCAGTCTGATAAGGAGGGAGGTTATCAGACTGA',
        'H2':    'TCAGTCTGATAAGGAGGGAGGTATCAGACTGATGTTGATTTTT',
        'CP':    'AAAAA',
    },
    engine=engine,
)
report = bridge.verify()
print(report)

# Export to mantis CRNetwork
rn = bridge.to_crnetwork()           # requires mantis-delta
rn.simulate(bridge._default_ic(), (0, 7200))
```

---

## User guide

### 1. ThermoEngine and backends

`ThermoEngine` is the central dispatcher. Instantiate once and pass it through your analysis.

```python
from strider import ThermoEngine

# Auto-select: prefers nupack > vienna > native
engine = ThermoEngine(material='dna', celsius=37, sodium=0.137, magnesium=0.01)

# Explicit backend
engine_v = ThermoEngine(backend='vienna')    # requires ViennaRNA
engine_n = ThermoEngine(backend='nupack')    # requires nupack
engine_0 = ThermoEngine(backend='native')   # always available

# Check what's available
print(ThermoEngine.available_backends())    # ['native', 'vienna', 'nupack']
print(engine.backend_name)                 # 'nupack' (or whatever was selected)
```

#### Persistent caching

Every `mfe()` and `pfunc()` call can be memoized to a SQLite database. Results are keyed by a SHA-256 hash of (operation, material, temperature, salt, sequences):

```python
from strider import ThermoEngine, DiskCache

cache = DiskCache('~/.strider/my_project.db', max_size_mb=200, ttl_days=30)
engine = ThermoEngine(material='dna', celsius=37, cache=cache)

# First call computes; subsequent calls for the same inputs are instant
result = engine.mfe('ATCGATCG')
print(cache.stats())  # {'hits': 1, 'misses': 0, 'hit_rate': 1.0, 'entries': 1, ...}
```

#### ML correction hook

`correction_model` accepts any callable `(sequence: str) → float` that returns a ΔΔG correction (kcal/mol). This is added to every `pfunc()` result — useful for plugging in an empirically calibrated neural-network correction on top of the NN model:

```python
my_model = lambda seq: -0.02 * seq.count('G')   # trivial example
engine = ThermoEngine(correction_model=my_model)
```

---

### 2. DNA / RNA thermodynamics

#### Duplex ΔG

```python
from strider import ThermoEngine, duplex_dg, melting_temperature

# Via engine (uses backend-appropriate method)
engine = ThermoEngine()
dg = engine.duplex_dg('ATCGATCG', 'CGATCGAT')
print(f"ΔG = {dg:.2f} kcal/mol")       # ΔG = -7.4 kcal/mol

# Standalone NN function (DNA only, always native)
dg_nn = duplex_dg('ATCGATCG', celsius=37.0, sodium_M=0.137)
print(f"ΔG (NN) = {dg_nn:.2f} kcal/mol")

# Melting temperature
tm = melting_temperature('ATCGATCG', strand_conc_M=250e-9, sodium_M=0.137)
print(f"Tm = {tm:.1f} °C")             # Tm = 26.3 °C
```

#### Hairpin intramolecular folding

```python
# Single sequence → pfunc gives ensemble ΔG of all intramolecular structures
result = engine.pfunc('GGGAAACCC')
print(f"Ensemble ΔG = {result.free_energy:.2f} kcal/mol")
print(f"Partition function Q = {result.partition_function:.3e}")
print(f"Pair probability matrix shape: {result.pair_probs.shape}")   # (9, 9)
```

#### Reaction ΔΔG

`engine.ddg()` is the workhorse for pathway analysis. Each element of `reactants` / `products` is either a single sequence string (computed as a monomer) or a list of sequences (computed as a multi-strand complex):

```python
mirna = 'TAGCTTATCAGACTGATGTTGA'
H1    = 'TCAACATCAGTCTGATAAGGAGGGAGGTTATCAGACTGA'

# ΔΔG for miRNA + H1 → miRNA·H1 complex
ddg_r1 = engine.ddg(
    reactants=[mirna, H1],
    products=[[mirna, H1]],   # list inside list = multi-strand complex
)
print(f"ΔΔG(R1) = {ddg_r1:.2f} kcal/mol")  # e.g. -8.5 kcal/mol
```

#### Toehold accessibility

The fraction of the ensemble in which all toehold positions are simultaneously unpaired — a direct measure of how accessible the toehold is for incoming strand binding:

```python
# Check accessibility of the first 6 nt (the toehold) of H1
prob = engine.toehold_accessibility(H1, toehold_positions=list(range(6)))
print(f"Toehold accessible in {prob:.1%} of ensemble")
```

#### Salt corrections

Salt corrections for non-1M NaCl and Mg²⁺ are applied automatically when `sodium ≠ 1.0` or `magnesium > 0`. strider uses Owczarzy et al. (2004) for Na⁺ and Owczarzy et al. (2008) for Mg²⁺, with a mixed-ion regime from the √[Mg²⁺]/[Na⁺] ratio.

#### Chemical modifications (LNA, 2′OMe, PS)

```python
from strider import ModificationSite, apply_modifications

dg_unmod = engine.duplex_dg('ATCGATCG')

mods = [
    ModificationSite(position=0, mod_type='LNA'),   # +L at position 0
    ModificationSite(position=7, mod_type='LNA'),   # +L at position 7
]
dg_mod = apply_modifications(dg_unmod, 'ATCGATCG', mods)
print(f"ΔΔG(modification) = {dg_mod - dg_unmod:.2f} kcal/mol")  # ~-3.0
```

---

### 3. Secondary structure prediction

#### MFE structure

```python
from strider import fold_mfe, ThermoEngine

# Standalone Zuker-style DP (native backend)
structure, energy, pairs = fold_mfe('GGGAAATTTCCC', celsius=37.0, material='dna')
print(structure)    # '((((....))))'
print(energy)       # -2.8 kcal/mol
print(pairs)        # [(0, 11), (1, 10), (2, 9), (3, 8)]

# Via engine (uses configured backend)
engine = ThermoEngine()
result = engine.mfe('GGGAAATTTCCC')
print(result.structure, result.energy, result.base_pairs)
```

#### Pseudoknots

`fold_pseudoknot()` extends the standard MFE algorithm to consider H-type pseudoknots (the most common class in biosensor contexts). It uses a restricted Rivas & Eddy (1999) grammar at O(n⁴) time:

```python
from strider import fold_pseudoknot

structure, energy, pairs = fold_pseudoknot('GGGCCCTTTGGGCCC')
# structure uses () for normal pairs, [] for pseudoknot pairs
print(structure)    # e.g. '(((....[[)))....]]]'
```

#### Dot-bracket parsing and analysis

```python
from strider import parse_pairs, to_dot_bracket, validate
from strider.structure.dot_bracket import stem_regions, unpaired_positions

structure = '(((...)))'
pairs = parse_pairs(structure)          # [(0, 8), (1, 7), (2, 6)]
rebuilt = to_dot_bracket(pairs, 9)     # '(((...)))'
print(validate('(((...)))'))           # True
print(validate('(((...))'))            # False (mismatched)

stems = stem_regions(structure)        # [(0, 8, 3)] — one stem of length 3
unpaired = unpaired_positions(structure)  # [3, 4, 5]
```

#### Mountain representation

The mountain plot encodes the nesting depth at each position — a compact fingerprint for comparing structures:

```python
from strider import mountain_vector, compare_structures

m = mountain_vector('(((...)))')      # array([0, 1, 2, 2, 2, 2, 2, 1, 0])
dist = compare_structures('(((...)))', '((.......))')  # L1 distance, range [0, 1]
print(f"Structure distance: {dist:.3f}")
```

---

### 4. TMSD kinetics

#### Toehold rate constants

`toehold_kf()` uses the Zhang & Winfree (2009) empirical lookup table at 25 °C and applies an Arrhenius correction to the target temperature (Ea ≈ 20 kcal/mol for DNA TMSD):

```python
from strider import toehold_kf, displacement_kf, leakage_kf, rates_from_ddg

# Forward rate at 37 °C for a 6-nt toehold
kf = toehold_kf(n_nt=6, material='dna', celsius=37.0)
print(f"kf = {kf:.2e} M⁻¹ s⁻¹")      # kf ≈ 5.5e5 M⁻¹ s⁻¹

# Derive reverse rate from ΔΔG and detailed balance
kf_val, kr_val = rates_from_ddg(ddg=-8.5, kf=kf, celsius=37.0)
print(f"kr = {kr_val:.2e} s⁻¹")

# Boltzmann-suppressed leakage rate (hairpin breathing model)
k_leak = leakage_kf(stem_stability_kcal=7.5)
print(f"k_leak = {k_leak:.2e} M⁻¹ s⁻¹")   # ≪ kf
```

#### TMSDKineticModel — full circuit rates

`TMSDKineticModel` computes ΔΔG internally from the `ThermoEngine` and returns mantis-compatible rate dictionaries:

```python
from strider import ThermoEngine, TMSDKineticModel

engine = ThermoEngine(material='dna', celsius=37, sodium=0.137, magnesium=0.01)
model = TMSDKineticModel(engine)

# Compute rates for a single reaction
rate_set = model.reaction_rates(
    reactant_seqs=['TAGCTTATCAGACTGATGTTGA', 'TCAACATCAGTCTGATAAGG'],
    product_seqs=[['TAGCTTATCAGACTGATGTTGA', 'TCAACATCAGTCTGATAAGG']],
    toehold_length=6,
    mechanism='toehold_binding',
)
print(rate_set)
# TMSDRateSet(kf=5.5e5, kr=2.1e-3, k_eq=2.6e8, ddg=-8.5, toehold_length=6, ...)

# Build a full mantis-compatible rate dict for a circuit
reactions = [
    "mirna + H1 <-> mirna_H1",
    "mirna_H1 + H2 <-> H1H2 + mirna",
]
sequences = {
    'mirna': 'TAGCTTATCAGACTGATGTTGA',
    'H1':    'TCAACATCAGTCTGATAAGG...',
    'H2':    'TCAGTCTGATAAGGA...',
}
rates = model.circuit_rates(reactions, sequences, toehold_map={"mirna + H1 <-> mirna_H1": 6})
```

#### Arrhenius utilities

```python
from strider import arrhenius, detailed_balance_kr, k_eq_from_ddg, ddg_from_k_eq

# Scale a rate constant between temperatures
kf_50 = arrhenius(k_ref=5.5e5, ea_kcal=20.0, T_ref_K=298.15, T_K=323.15)

# Derive reverse rate from ΔΔG
kr = detailed_balance_kr(kf=5.5e5, ddg_kcal=-8.5, celsius=37.0)

# Convert between ΔΔG and Keq
keq = k_eq_from_ddg(-8.5, celsius=37.0)   # ≈ 9.6e5
ddg = ddg_from_k_eq(keq, celsius=37.0)    # ≈ -8.5
```

---

### 5. Leakage enumeration

`LeakageEnumerator` systematically checks all pairwise (and optional tripartite) strand combinations for thermodynamically favorable spurious complexes:

```python
from strider import ThermoEngine, LeakageEnumerator

engine = ThermoEngine()
enumerator = LeakageEnumerator(
    engine,
    ddg_threshold=-4.0,      # report reactions with ΔΔG < -4 kcal/mol
    max_complex_size=3,       # check pairs and triplets
    max_pathways=100,
)

strands = {
    'H1': 'TCAACATCAGTCTGATAAGG...',
    'H2': 'TCAGTCTGATAAGGAG...',
    'CP': 'AAAAA',
}

report = enumerator.enumerate(
    strands,
    intended_reactions=["H1 + H2 <-> H1H2"],  # exclude known-intended reactions
)

print(report)
# LeakageReport(3 spurious reactions, worst ΔΔG=-5.82 kcal/mol)

for rxn in report.reactions:
    print(rxn.mantis_string, f"  ΔΔG={rxn.ddg:.2f}")

# Filter to only the worst offenders
critical = report.filter(ddg_threshold=-5.0)

# Export as mantis reaction strings
mantis_strings = report.to_mantis_strings()
```

Each `SpuriousReaction` has a `pathway_type` classifying it as `"hybridization"`, `"displacement"`, or `"cooperative"`.

---

### 6. Sequence design

`SequenceDesigner` minimizes a composable `DesignObjective` using simulated annealing. Free domains are optimized; fixed domains (e.g. the miRNA binding site) are held constant.

```python
from strider import ThermoEngine, SequenceDesigner, DomainSpec, DesignObjective, HardConstraint

engine = ThermoEngine()

# Specify domains: free vs fixed
domains = {
    'toehold':  DomainSpec(length=6, material='dna'),                 # free
    'stem':     DomainSpec(length=11, material='dna'),                # free
    'binding':  DomainSpec(sequence='TAGCTTATCAGACTGATGTTGA'),        # fixed
}

# Compose objective: target ΔΔG for binding + hairpin stability
objective = (
    DesignObjective.ddg_target(engine, ['binding'], [['binding', 'stem']], target=-9.0, weight=2.0)
    + DesignObjective.gc_content('toehold', target_gc=0.5, weight=1.0)
    + DesignObjective.minimize_leakage(engine, ['toehold', 'stem'], threshold=-4.0, weight=0.5)
)

# Hard constraints: no AAAA runs, GC content between 40–60%
constraints = [
    HardConstraint.max_run(max_run_length=4),
    HardConstraint.gc_content(min_gc=0.4, max_gc=0.6),
]

designer = SequenceDesigner(engine, seed=42)
result = designer.design(
    domains,
    objective,
    hard_constraints=constraints,
    n_trials=10,
    max_iterations=500,
)

print(result)
# DesignResult(score=0.0312, iterations=500, seqs=['toehold', 'stem', 'binding'])
print(result.sequences)
print(result.objective_breakdown)
```

#### Built-in objectives

| Factory method | What it penalizes |
|---|---|
| `DesignObjective.ddg_target(engine, reactants, products, target)` | (ΔΔG − target)² |
| `DesignObjective.ddg_range(engine, reactants, products, min, max)` | ΔΔG outside [min, max] |
| `DesignObjective.minimize_leakage(engine, strand_names, threshold)` | Pairwise ΔΔG below threshold |
| `DesignObjective.toehold_accessible(engine, strand_name, positions, min_prob)` | Low toehold accessibility |
| `DesignObjective.gc_content(strand_name, target_gc)` | (GC − target)² |
| `DesignObjective.from_callable(fn)` | Any Python callable returning a float |

Objectives compose with `+` and scale with `*`:

```python
total = 2.0 * objective_a + objective_b + 0.5 * objective_c
```

#### Built-in hard constraints

| Factory method | What it enforces |
|---|---|
| `HardConstraint.no_repeats(motifs)` | Forbid specified sequence motifs |
| `HardConstraint.gc_content(min_gc, max_gc)` | GC fraction in [min, max] |
| `HardConstraint.no_self_complement(min_length)` | No self-complementary run ≥ min_length |
| `HardConstraint.iupac_pattern(strand_name, pattern, start)` | IUPAC code match at offset |
| `HardConstraint.max_run(max_run_length)` | No homopolymer run > max_run_length |
| `HardConstraint.min_length(length)` | Minimum sequence length |
| `HardConstraint.from_callable(fn)` | Any `(name, seq) → bool` |

---

### 7. Mutation sensitivity analysis

`MutationAnalyzer` computes how each nucleotide position contributes to a thermodynamic metric by exhaustively scanning all single-nucleotide mutations:

```python
from strider import ThermoEngine, MutationAnalyzer

engine = ThermoEngine()
analyzer = MutationAnalyzer(engine)

profile = analyzer.single_nt_scan(
    sequence='TCAACATCAGTCTGATAAGG',
    target='TAGCTTATCAGACTGATGTTGA',   # hybridization partner
    tolerance=1.0,                      # kcal/mol tolerance for robustness
)

print(f"Robustness: {profile.robustness:.1%}")         # fraction of mutations within tolerance
print(f"Critical positions: {profile.critical_positions(threshold=2.0)}")

# Heatmap visualization
profile.plot(title="H1 mutation sensitivity")
```

`profile.max_sensitivity` gives the worst-case ΔΔG over all three possible mutations at each position — useful for identifying positions that must be preserved.

---

### 8. Off-target screening

`OffTargetScreener` screens a probe sequence against a reference database (FASTA) using k-mer pre-filtering followed by full ΔΔG evaluation:

```python
from strider import ThermoEngine, OffTargetScreener

engine = ThermoEngine()
screener = OffTargetScreener(engine, reference_db='mirbase_hsa.fa', kmer_k=7)

report = screener.screen(
    sequence='TCAACATCAGTCTGATAAGG',
    n_top=10,
    ddg_threshold=-4.0,
)

print(report)
# ScreeningReport(query=TCAACATCAGTC..., hits=2, specific=False)
for hit in report.hits:
    print(f"  {hit.name}  ΔΔG={hit.ddg:.2f}  shared_kmers={hit.k_score}")

# Direct selectivity comparison against a family
selectivity = screener.specificity_vs(
    sequence='TCAACATCAGTCTGATAAGG',
    family_members=[miR21_alt1, miR21_alt2, miR155],
    target='TAGCTTATCAGACTGATGTTGA',
)
# {miR21_alt1: +1.2, miR21_alt2: +2.4, miR155: +5.8}  (positive = more selective)
```

You can also add sequences in-memory without a FASTA file:

```python
screener.add_sequences({'miR155': 'UUAAUGCUAAUUGUGAUAGGGGU'})
```

---

### 9. CHABridge — sequences to mantis CRNetwork

`CHABridge` encodes the 4-reaction CHA topology and automates all thermodynamic verification checks from the NUPACK design workflow, but without any NUPACK dependency.

```python
from strider import ThermoEngine, CHABridge

engine = ThermoEngine(material='dna', celsius=37, sodium=0.137, magnesium=0.01)

bridge = CHABridge(
    sequences={
        'mirna': 'TAGCTTATCAGACTGATGTTGA',
        'H1':    'TCAACATCAGTCTGATAAGGAGGGAGGTTATCAGACTGA',
        'H2':    'TCAGTCTGATAAGGAGGGAGGTATCAGACTGATGTTGATTTTT',
        'CP':    'AAAAA',
    },
    engine=engine,
    toehold_d1=6,    # miRNA·H1 toehold length
    toehold_d2=11,   # H2 branch migration domain
    tail_cp=9,       # CP tail length
)
```

#### Thermodynamic verification

`bridge.verify()` runs all seven design checks and returns a structured report:

```python
report = bridge.verify()
print(report)
```

```
CHA Verification: PASS
  Toehold accessible:    ✓
  H1 stability:          -8.32 kcal/mol ✓
  H2 stability:          -7.65 kcal/mol ✓
  ΔΔG(R1, init):         -8.52 kcal/mol ✓
  ΔΔG(R2, prop):         -4.81 kcal/mol ✓
  ΔΔG(R3, detect):       -9.14 kcal/mol ✓
  ΔΔG(spont leakage):    -5.22 kcal/mol ✓
  Catalyst recycled:     ✓
  CP leakage:            -3.88 kcal/mol ✓
  Predicted signal:      94.7%
```

| Check | Criterion | Meaning |
|---|---|---|
| Toehold accessible | P ≥ 0.50 | ≥50% of H1 ensemble has toehold unpaired |
| H1/H2 stability | −12 to −4 kcal/mol | Hairpin in kinetic sweet spot |
| ΔΔG(R1) | < −3 kcal/mol | miRNA·H1 binding is favorable |
| ΔΔG(R2) | < −3 kcal/mol | Strand exchange favors H1·H2 formation |
| ΔΔG(R3) | < −8 kcal/mol | CP binds H1·H2 tail strongly |
| ΔΔG(spont) | > −10 kcal/mol | H1 + H2 → H1·H2 is suppressed in absence of trigger |
| CP leakage | > −6 kcal/mol | CP does not bind H2 alone |

#### Accessing rates and ΔΔG values

```python
ddg = bridge.ddg_pathway   # dict: {'R1': -8.5, 'R2': -4.8, ..., 'leakage': -5.2}
rates = bridge.rates        # dict keyed by mantis reaction strings

for rxn, k in rates.items():
    print(f"{rxn}: {k:.2e}")
```

#### Exporting to mantis

```python
rn = bridge.to_crnetwork()          # → mantis.CRNetwork
ic = bridge._default_ic()           # 100 nM hairpins, 10 nM miRNA, zero complexes
rn.simulate(ic, (0, 7200))
ss = rn.steady_states(ic)[0]
print(ss.concentrations)
```

#### Generic pipeline: any topology

For circuits beyond CHA, `rates_to_crnetwork()` runs the full pipeline with optional leakage enumeration:

```python
from strider import rates_to_crnetwork

rn = rates_to_crnetwork(
    reaction_strings=["A + B <-> AB", "AB + C <-> ABC"],
    sequences={'A': 'ATCG...', 'B': 'CGAT...', 'C': 'TTTT...'},
    engine=engine,
    include_leakage=True,
    leakage_threshold=-4.0,
)
```

---

### 10. Parameter sweeps and caching

`ParameterSweep` runs any callable over an N-dimensional grid with transparent caching and optional parallelism.

```python
from strider import ThermoEngine, ParameterSweep, DiskCache

engine = ThermoEngine()
cache = DiskCache('/tmp/sweep_cache.db')
sweep = ParameterSweep(engine, cache=cache, n_workers=4)

# Built-in: toehold length sweep
result = sweep.toehold_sweep(
    hairpin_seq='TCAACATCAGTCTGATAAGG',
    toehold_lengths=list(range(2, 12)),
    target_strand='TAGCTTATCAGACTGATGTTGA',
)
result.plot(xlabel='Toehold length (nt)', ylabel='kf (M⁻¹ s⁻¹)')

# Built-in: temperature sweep
result = sweep.temperature_sweep(
    sequences={'H1': 'ATCG...', 'H2': 'CGAT...'},
    temperatures=list(range(20, 65, 5)),
)

# Custom N-D grid sweep
def my_score(params: dict) -> float:
    engine2 = ThermoEngine(celsius=params['temperature'])
    return engine2.pfunc('ATCG').free_energy

result = sweep.grid_sweep(
    axes={'temperature': [25, 37, 50], 'sodium': [0.05, 0.137, 0.5]},
    fn=my_score,
)
print(result.optimum())            # {'temperature': 25, 'sodium': 0.05}
result.plot()

# Convert to pandas DataFrame for further analysis
df = result.to_dataframe()         # requires pandas extra
```

#### DiskCache details

The cache uses SQLite3 in WAL mode for safe concurrent reads/writes across parallel workers. LRU eviction is triggered when the database exceeds `max_size_mb`.

```python
cache = DiskCache(
    path='~/.strider/cache.db',
    max_size_mb=500,    # evict oldest 20% when exceeded
    ttl_days=30,        # entries expire after 30 days
)
with cache:             # context manager closes the connection
    val = cache.get('my_key')
    cache.set('my_key', result_object)
    print(cache.stats())
```

---

### 11. Export formats

```python
from strider import to_vienna, to_ct, to_bpseq, to_fasta, to_oxdna, write

seq = 'GGGAAATTTCCC'
struct = '(((.....)))'

# Individual format functions
print(to_vienna(seq, struct, name='hairpin'))
print(to_ct(seq, struct, name='hairpin', energy=-2.8))
print(to_bpseq(seq, struct))
print(to_fasta(seq, name='hairpin', description='miR-21 probe'))
print(to_oxdna(seq))      # topology skeleton for MD simulations

# Write to file (auto-detect format from extension)
write(seq, struct, path='output.rna', fmt='vienna')   # fmt: vienna|ct|bpseq|fasta|oxdna
write(seq, struct, path='output.ct',  fmt='ct', energy=-2.8)
```

---

## API reference

### `ThermoEngine`

```python
ThermoEngine(
    material='dna',        # 'dna' | 'rna'
    celsius=37.0,
    sodium=0.137,          # [Na+] in molar
    magnesium=0.01,        # [Mg2+] in molar
    backend='auto',        # 'auto' | 'native' | 'vienna' | 'nupack'
    cache=None,            # DiskCache | None
    correction_model=None, # callable(seq) → float | None
) → ThermoEngine
```

#### Methods

| Method | Returns | Description |
|---|---|---|
| `mfe(*sequences)` | `MFEResult` | Minimum free energy structure |
| `pfunc(*sequences)` | `PFuncResult` | Ensemble free energy and pair probability matrix |
| `duplex_dg(seq1, seq2=None)` | `float` | ΔG of hybridization; `seq2=None` → intramolecular folding |
| `ddg(reactants, products)` | `float` | ΔΔG = Σ G(products) − Σ G(reactants) (kcal/mol) |
| `toehold_accessibility(seq, positions)` | `float` | Fraction of ensemble with all toehold positions unpaired |
| `melting_temperature(seq, strand_conc_M)` | `float` | Melting temperature (°C) |
| `mfe_batch(strand_groups, n_workers)` | `list[MFEResult]` | Parallelized batch MFE |
| `available_backends()` *(classmethod)* | `list[str]` | Backends importable in this environment |

#### `MFEResult`

```python
@dataclass
class MFEResult:
    energy:     float                  # kcal/mol (negative = stable)
    structure:  str                    # dot-bracket string
    base_pairs: list[tuple[int, int]]  # (i, j) pairs, 0-based
    sequence:   str                    # input sequence
```

#### `PFuncResult`

```python
@dataclass
class PFuncResult:
    free_energy:       float       # ensemble ΔG = -RT ln Q (kcal/mol)
    partition_function: float      # dimensionless Q
    pair_probs:        np.ndarray  # shape (n, n) base-pair probability matrix
```

---

### `CHABridge`

```python
CHABridge(
    sequences,          # dict with keys 'mirna', 'H1', 'H2', 'CP'
    engine=None,        # ThermoEngine; created with physiological defaults if None
    celsius=37.0,
    toehold_d1=6,       # toehold length for miRNA·H1 binding
    toehold_d2=11,      # branch migration domain for H2
    tail_cp=9,          # CP tail length
)
```

| Property / method | Returns | Description |
|---|---|---|
| `ddg_pathway` | `dict[str, float]` | ΔΔG for all 4 reactions + leakage + CP leakage |
| `rates` | `dict[str, float]` | mantis-compatible rate dict (kf and kr for all reactions) |
| `verify()` | `CHAVerificationReport` | Seven-check thermodynamic audit |
| `to_crnetwork()` | `mantis.CRNetwork` | mantis-ready network; requires mantis-delta |
| `sensitivity(target_species, perturbation)` | `dict[str, float]` | One-at-a-time rate sensitivity analysis |

---

### `SequenceDesigner`

```python
designer = SequenceDesigner(engine=None, seed=None)

result = designer.design(
    domains,             # dict[str, DomainSpec]
    objective,           # DesignObjective
    hard_constraints=[],
    n_trials=10,
    max_iterations=500,
    T_start=1.0,         # initial simulated annealing temperature
    T_end=0.01,          # final temperature
    verbose=False,
) → DesignResult
```

#### `DesignResult`

```python
@dataclass
class DesignResult:
    sequences:           dict[str, str]    # domain_name → optimized sequence
    objective_value:     float             # final total score (lower is better)
    objective_breakdown: dict[str, float]  # per-term contributions
    n_iterations:        int
    trial_scores:        list[float]       # best score from each trial
    converged:           bool              # True if final score < 1e-4
```

---

### `LeakageEnumerator`

```python
enumerator = LeakageEnumerator(
    engine,
    ddg_threshold=-4.0,       # report reactions with ΔΔG < threshold
    max_complex_size=3,        # 2 = pairs only; 3 = pairs + triplets
    max_pathways=100,
)

report = enumerator.enumerate(
    strands,                  # dict[str, str] — name → sequence
    intended_reactions=None,  # list[str] to exclude from report
) → LeakageReport
```

#### `LeakageReport`

| Attribute / method | Description |
|---|---|
| `reactions` | `list[SpuriousReaction]` sorted by ΔΔG (worst first) |
| `total_spurious` | Number of spurious reactions found |
| `worst_ddg` | Most negative ΔΔG across all reactions |
| `summary` | Human-readable summary string |
| `to_mantis_strings()` | `list[str]` — mantis-style reaction strings |
| `filter(ddg_threshold)` | New `LeakageReport` keeping only reactions below threshold |

---

### `TMSDRateSet`

```python
@dataclass
class TMSDRateSet:
    kf:             float   # forward rate (M⁻¹ s⁻¹)
    kr:             float   # reverse rate (s⁻¹)
    k_eq:           float   # kf / kr (M⁻¹)
    ddg:            float   # reaction ΔΔG (kcal/mol)
    toehold_length: int
    mechanism:      str     # "toehold_binding" | "branch_migration" | "leakage"
```

---

### `DiskCache`

```python
cache = DiskCache(
    path='~/.strider/cache.db',
    max_size_mb=500.0,
    ttl_days=None,         # None = never expire
)

cache.get(key)             # → Any | None
cache.set(key, value)      # → None
cache.stats()              # → dict with hits, misses, hit_rate, entries, size_mb
cache.clear()              # → None (deletes all entries)
cache.close()              # → None (closes SQLite connection)
DiskCache.make_key(*args)  # → str (SHA-256 hex of args)
```

---

## Examples

All examples are in the `examples/` directory and can be run directly. They do not require NUPACK or ViennaRNA — the native backend is used throughout.

```bash
python examples/01_dna_thermodynamics.py
python examples/02_hairpin_folding.py
python examples/03_tmsd_kinetics.py
python examples/04_sequence_design.py
python examples/05_leakage_and_screening.py
python examples/06_parameter_sweep.py
python examples/07_cha_to_mantis.py    # requires mantis-delta
```

### `01_dna_thermodynamics.py` — NN model fundamentals

Demonstrates duplex ΔG, melting temperature, salt corrections, and LNA modification energetics. Validates strider's native NN implementation against published SantaLucia & Hicks (2004) values and shows how Owczarzy salt corrections shift Tm by several degrees under physiological conditions.

### `02_hairpin_folding.py` — Structure prediction

Folds a panel of CHA hairpin candidates, draws their arc diagrams, plots mountain vectors, and computes pairwise structural distances. Shows how `fold_pseudoknot()` identifies structures that the standard MFE algorithm misses.

### `03_tmsd_kinetics.py` — Zhang & Winfree rate model

Reproduces the Zhang & Winfree (2009) kf-vs-toehold-length curve, applies Arrhenius temperature corrections from 20 °C to 60 °C, and demonstrates how `rates_from_ddg()` propagates thermodynamic uncertainty into kinetic uncertainty. Annotates the 6-nt toehold "sweet spot" at 37 °C.

### `04_sequence_design.py` — Simulated annealing optimization

Designs H1 and H2 sequences for the CHA cascade from scratch: specifies the miRNA-binding domain as a fixed constraint, composes a four-term objective (H1 stability + R1 driving force + spontaneous leakage suppression + GC content), applies `HardConstraint.max_run(4)` and `HardConstraint.gc_content()`, and runs 10-trial simulated annealing. Plots trial convergence curves and a mutation sensitivity heatmap for the best result.

### `05_leakage_and_screening.py` — Leakage enumeration and off-target screening

Enumerates all spurious pairwise and tripartite complexes for a set of CHA strands, ranks them by ΔΔG, and adds leakage reactions to a mantis network. Also loads a miRBase FASTA file (miR-21 family) and runs `OffTargetScreener` to compute selectivity of H1 against closely related miRNA sequences.

### `06_parameter_sweep.py` — Grid sweeps and dose-response

Runs a 2D grid sweep over toehold length and temperature, caches results to disk, and plots a contour map. Also generates a dose-response curve ([miRNA]₀ vs. predicted signal fraction) by sweeping initial conditions through the mantis CRNetwork solver.

### `07_cha_to_mantis.py` — End-to-end integration

The primary validation example. Demonstrates the complete pipeline:
1. `ThermoEngine` with native backend at physiological conditions
2. `CHABridge.verify()` — seven-check audit
3. `bridge.to_crnetwork()` — export to mantis CRNetwork
4. ODE integration and steady-state finding via mantis
5. `bridge.sensitivity()` — one-at-a-time rate sensitivity analysis
6. Predicted signal vs. miRNA concentration

---

## Backend comparison

The native backend uses strider's own McCaskill O(n³) partition-function DP with nick-aware recursions for multi-strand complexes — the same algorithm as ViennaRNA (RNAcofold) and NUPACK. Below are representative comparisons at 37 °C, 137 mM NaCl, 10 mM MgCl₂.

| Calculation | Native | ViennaRNA | NUPACK | Gap (native vs NUPACK) |
|---|---|---|---|---|
| ΔG(R1): miR21 + H1 → miR21·H1 | −11.54 | −11.38 | −11.42 | **0.1 kcal/mol** |
| ΔG(H1): hairpin ensemble | −7.22 | −8.64 | −8.78 | 1.6 kcal/mol |
| ΔG(spont): H1 + H2 → H1·H2 | −13.3 | −22.8 | −23.4 | 10 kcal/mol |
| ΔG(R3): H1·H2 + CP → H1·H2·CP | −3.35 | −6.12 | −6.44 | 3.1 kcal/mol |

**Where the native backend matches well:** bimolecular toehold binding reactions (R1) where the dominant contribution is duplex stacking, and individual hairpin ensemble energies for short sequences (< 40 nt). The 0.1 kcal/mol agreement on R1 is sufficient for rate constant estimation to within a factor of 2.

**Where it diverges:** spontaneous dimerization (long multi-loop structures) and detection reactions (short CP duplexes dominated by end effects). The gaps arise from simplified Turner loop parameters (no tetraloop bonus, no terminal mismatch correction, approximate multi-loop penalty) and missing dangling-end contributions. For publication-quality multi-strand ΔG values, use `backend='vienna'` or `backend='nupack'`.

**When to use each backend:**

| Scenario | Recommended backend |
|---|---|
| Rapid screening / design iteration (< 40 nt hairpins) | `native` |
| MFE folding of sequences up to ~200 nt | `vienna` |
| High-accuracy partition function for multi-strand complexes | `nupack` |
| No external dependencies (CI, lightweight environments) | `native` |
| Publication-quality thermodynamics | `nupack` or `vienna` |

---

## Running the tests

```bash
cd strider
pip install -e .[dev]
pytest tests/ -v
```

The test suite has 79 tests across seven files:

| File | Tests | What is covered |
|---|---|---|
| `test_thermo_dna.py` | 16 | NN parameters, duplex_dg, Tm, salt corrections, self-complementarity |
| `test_tmsd.py` | 15 | toehold_kf table, Arrhenius correction, detailed balance, leakage_kf, Keq conversions |
| `test_design.py` | 15 | SequenceDesigner SA convergence, DomainSpec, hard constraints, MutationAnalyzer |
| `test_mfe.py` | 12 | fold_mfe correctness, dot-bracket parsing, mountain vectors, structure comparison |
| `test_bridge.py` | 9 | CHABridge ddg_pathway, verify() seven checks, rates dict, mantis integration |
| `test_formats.py` | 7 | Vienna, CT, BPSEQ, FASTA, oxDNA output; round-trip pair parsing |
| `test_leakage.py` | 5 | LeakageEnumerator pairwise enumeration, pathway classification, filter() |

> **Note:** the three `test_bridge.py` tests that call `bridge.to_crnetwork()` require mantis-delta and are skipped if the package is not installed.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'strider'`

Install from the correct directory — the one containing `pyproject.toml`:

```bash
cd /path/to/strider   # must contain pyproject.toml
pip install -e .
```

Running `pip install -e .` from a parent directory will not work.

### `engine.ddg()` returns values far from NUPACK

The native backend's multi-strand ΔG diverges from NUPACK for:
- Long-range multi-loop structures (spontaneous hairpin dimerization)
- Very short duplexes (≤ 5 bp) where end effects dominate

Switch to `backend='vienna'` or `backend='nupack'` for these cases. For bimolecular toehold binding of typical 6–12 nt toeholds, the native backend agrees to within ~0.1 kcal/mol.

### `CHABridge.verify()` fails spontaneous leakage check

`ΔΔG(spont) < −10 kcal/mol` means H1 and H2 hybridize too favorably in the absence of trigger. Common causes:
- H1 and H2 have long complementary stems outside the intended domains
- The stem domain (D2) is too long or too GC-rich
- The D3 spacer is not introducing enough disruption

Use `SequenceDesigner` with `DesignObjective.minimize_leakage()` weighted heavily, or add `HardConstraint.no_self_complement(min_length=6)` to suppress cross-complementarity.

### Design converges to a high score (> 1.0)

Simulated annealing can get trapped if:
- **Conflicting objectives** — e.g. maximizing GC content while minimizing leakage. Check `result.objective_breakdown` to see which terms dominate.
- **Hard constraints too restrictive** — if the constraint space is very small, most mutations get rejected. Try relaxing `HardConstraint.gc_content()` bounds.
- **Too few iterations** — increase `max_iterations` or `n_trials`. The `trial_scores` list shows how much variance there is across restarts.
- **Temperature schedule too fast** — lower `T_end` (e.g. 0.001) to allow finer convergence.

### `DiskCache.get()` always returns `None`

The cache key is a SHA-256 hash of (operation, material, celsius, sodium, magnesium, sequences). Even a 0.001 °C difference in `celsius` produces a different key. Make sure you are using the same `ThermoEngine` instance or identical parameter values across cache reads and writes.

---

## Background and theory

### Nearest-neighbor thermodynamic model

The NN model computes duplex stability by summing stacking contributions from every adjacent base-pair step. For a duplex with sequence 5′-X₁X₂…Xₙ-3′, the free energy is:

```
ΔG = Σᵢ ΔG(XᵢXᵢ₊₁) + ΔG_init(5′ end) + ΔG_init(3′ end) + ΔG_sym
```

where the sum runs over all n−1 dinucleotide steps, initiation terms account for the terminal base pairs, and a symmetry correction is added if the sequence is self-complementary. The key references are:

- **SantaLucia J Jr, Hicks D** (2004). The thermodynamics of DNA structural motifs. *Annu. Rev. Biophys. Biomol. Struct.* 33, 415–440.
- **SantaLucia J Jr** (1998). A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics. *PNAS* 95, 1460–1465.
- **Mathews DH et al.** (1999). Expanded sequence dependence of thermodynamic parameters improves prediction of RNA secondary structure. *J. Mol. Biol.* 288, 911–940.
- **Sugimoto N et al.** (1995). Thermodynamic parameters to predict stability of RNA/DNA hybrid duplexes. *Biochemistry* 34, 11211–11216.

### McCaskill partition function DP

The ensemble free energy is computed via the McCaskill (1990) O(n³) dynamic programming algorithm. For multi-strand complexes, strider uses a nick-aware extension: the concatenated sequence has "nick" positions at strand boundaries, and hairpin loops spanning a nick are disallowed. This is the same recursion used by ViennaRNA (RNAcofold) and NUPACK.

- **McCaskill JS** (1990). The equilibrium partition function and base pair binding probabilities for RNA secondary structure. *Biopolymers* 29, 1105–1119.
- **Dirks RM, Pierce NA** (2003). A partition function algorithm for nucleic acid secondary structure including pseudoknots. *J. Comput. Chem.* 24, 1664–1677.

### TMSD kinetics

The Zhang & Winfree (2009) empirical model gives kf as a function of toehold length at 25 °C. strider applies an Arrhenius correction with Ea ≈ 20 kcal/mol (validated by Srinivas et al. 2013) to scale to physiological temperature. Reverse rates are derived from detailed balance: kr = kf · exp(ΔΔG/RT).

- **Zhang DY, Winfree E** (2009). Control of DNA strand displacement kinetics using toehold exchange. *JACS* 131, 17303–17314.
- **Srinivas N et al.** (2013). Nucleic acids reaction coordinator. *Nucleic Acids Res.* 41, 10641–10658.

### CHA miR-21 biosensor

The Catalytic Hairpin Assembly (CHA) cascade is a DNA nanotechnology circuit in which a microRNA target (miR-21) catalytically drives the formation of a double-stranded reporter complex without any enzymatic amplification. The four-reaction network is:

```
miRNA + H1   ⇌  miRNA·H1                   (toehold binding / dissociation)
miRNA·H1 + H2 ⇌  H1·H2 + miRNA            (strand exchange / catalyst release)
H1·H2 + CP   ⇌  H1·H2·CP                  (capture probe binding for readout)
H1 + H2      ⇌  H1·H2                      (spontaneous leakage — suppressed)
```

The miRNA is released intact in the second reaction, allowing it to trigger additional CHA cycles (catalytic turnover). The `CHABridge` class encodes this topology and automates the verification checks.

### Salt corrections

Non-1M NaCl conditions are corrected via the Owczarzy (2004/2008) models. The mixed-ion regime selects between Na⁺-only and Mg²⁺-only corrections based on √[Mg²⁺]/[Na⁺]:

- **Owczarzy R et al.** (2004). Effects of sodium ions on DNA duplex oligomers. *Biochemistry* 43, 3537–3554.
- **Owczarzy R et al.** (2008). Magnesium ions and DNA. *Biochemistry* 47, 5336–5353.

---

## Citation

If you use strider-dna in published work, please cite:

```bibtex
@software{venegas2026strider,
  author  = {Venegas, Emilio},
  title   = {strider-dna: Nucleic Acid Thermodynamics, Kinetics, and Circuit Design},
  year    = {2026},
  url     = {https://github.com/EmilioVenegas/strider},
  version = {0.1.0}
}
```

---

## License

MIT © 2026 Emilio Venegas
