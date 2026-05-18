# strider-dna — Nucleic Acid Thermodynamics, Kinetics, and Circuit Design

[![Tests](https://img.shields.io/badge/tests-193%20passed-brightgreen)](#running-the-tests)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

**strider** is a Python library for computing the thermodynamics and kinetics of DNA/RNA circuits. Given a set of strand sequences, it predicts free energies via nearest-neighbor parameters, folds secondary structures via dynamic programming, derives TMSD rate constants from the Zhang & Winfree (2009) empirical model, enumerates spurious leakage pathways, and produces kinetic rate dictionaries that drop directly into a mantis-delta `CRNetwork`. The full pipeline — sequence → thermodynamics → kinetics → reaction network → steady states / stochastic trajectories / bifurcation — runs end-to-end without any NUPACK or ViennaRNA dependency.

strider ships a three-backend thermodynamic engine that automatically selects the best available calculator: its own McCaskill O(n³) partition-function DP (always available, matches ViennaRNA/NUPACK to within ~0.1 kcal/mol for bimolecular binding), the ViennaRNA C library (recommended for long sequences), or NUPACK (highest accuracy, restrictive license). The same API surface works with every backend — only the `ThermoEngine(backend=…)` argument changes. NUPACK pfunc values are auto-shifted to a 1 M standard state so engine output is backend-agnostic.

Beyond thermodynamics, strider provides a **circuit catalog** of ready-made DSD templates (CHA, HCR, seesaw gates, translators) wrapped around a generic verification framework, a **pure-thermo concentration solver** that matches NUPACK's `tube_analysis` to ~1 %, **Boltzmann sampling** and **suboptimal-structure enumeration** on top of the partition function, an **`Assay` / `AssayPanel`** design abstraction for ensemble-defect minimization, a lightweight **`DSDCompiler`** for domain-level sequence assembly, and — via the companion **mantis** library — Gillespie SSA stochastic simulation in addition to deterministic ODE integration.

---

## Table of contents

1. [Installation](#installation)
2. [Core concepts](#core-concepts)
3. [Quick start](#quick-start)
4. [Command-line interface](#command-line-interface)
5. [User guide](#user-guide)
   - [ThermoEngine and backends](#1-thermoengine-and-backends)
   - [DNA / RNA thermodynamics](#2-dna--rna-thermodynamics)
   - [Secondary structure prediction](#3-secondary-structure-prediction)
   - [Boltzmann sampling and subopt enumeration](#4-boltzmann-sampling-and-subopt-enumeration)
   - [TMSD kinetics](#5-tmsd-kinetics)
   - [Leakage enumeration](#6-leakage-enumeration)
   - [Equilibrium concentration solver](#7-equilibrium-concentration-solver)
   - [Sequence design and the Assay abstraction](#8-sequence-design-and-the-assay-abstraction)
   - [Mutation sensitivity analysis](#9-mutation-sensitivity-analysis)
   - [Off-target screening](#10-off-target-screening)
   - [Circuit catalog and the mantis bridge](#11-circuit-catalog-and-the-mantis-bridge)
   - [DSDCompiler — domain-level circuit assembly](#12-dsdcompiler--domain-level-circuit-assembly)
   - [Stochastic simulation via mantis](#13-stochastic-simulation-via-mantis)
   - [Parameter sweeps and caching](#14-parameter-sweeps-and-caching)
   - [Export formats](#15-export-formats)
6. [API reference](#api-reference)
7. [Examples](#examples)
8. [Backend comparison](#backend-comparison)
9. [Running the tests](#running-the-tests)
10. [Troubleshooting](#troubleshooting)
11. [Background and theory](#background-and-theory)
12. [Citation](#citation)
13. [License](#license)

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
pip install strider-dna[mantis]   # mantis-delta integration (circuit templates, CRNetwork)
pip install strider-dna[pandas]   # SweepResult.to_dataframe()
pip install strider-dna[parallel] # ProcessPoolExecutor sweeps
pip install strider-dna[full]     # all of the above
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.6.

> **Note on import name:** the PyPI distribution is `strider-dna`, but the Python package is imported as `strider`:
> ```python
> import strider
> from strider import ThermoEngine, CHA, HCR, SeesawGate, Translator
> from strider import Assay, AssayPanel, Assembly, DSDCompiler
> from strider import solve_equilibrium, sample_structures, subopt_structures
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

Hairpin kinetics for CHA have a "sweet spot": stems stable enough to suppress leakage (ΔG ≲ −6 kcal/mol) but not so stable that the toehold is buried (ΔG ≳ −12 kcal/mol). The `CHA` circuit template's `verify()` method checks this and several other design criteria via the generic `CheckRegistry` framework — easy to extend with custom checks for non-CHA topologies (`HCR`, `SeesawGate`, etc.).

---

## Quick start

```python
from strider import ThermoEngine, CHA, solve_equilibrium

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

# Equilibrium concentrations of a 2-strand mix
res = solve_equilibrium(
    complexes={
        'A':  (['A'], 0.0),
        'B':  (['B'], 0.0),
        'AB': (['A', 'B'], -10.0),
    },
    totals={'A': 1e-7, 'B': 1e-7},
    celsius=37.0,
)
print(f"[AB] = {res.concentrations['AB']:.2e} M")  # ~4.0e-8

# Full CHA biosensor verification + mantis export
cha = CHA(
    sequences={
        'mirna': 'TAGCTTATCAGACTGATGTTGA',
        'H1':    'TCAACATCAGTCTGATAAGGAGGGAGGTTATCAGACTGA',
        'H2':    'TCAGTCTGATAAGGAGGGAGGTATCAGACTGATGTTGATTTTT',
        'CP':    'AAAAA',
    },
    engine=engine,
)
print(cha.verify())                  # pretty-printed CircuitReport
rn = cha.to_crnetwork()              # requires mantis-delta
sim = cha.simulate(                  # deterministic ODE
    {'mirna': 10e-9, 'H1': 100e-9, 'H2': 100e-9, 'CP': 100e-9}, (0, 7200),
)
```

Beyond CHA: replace `CHA(...)` with `HCR(...)`, `SeesawGate(logic='AND', ...)`, `Translator(...)`, or roll your own subclass of `CircuitTemplate`. Every template has the same `.verify()`, `.to_crnetwork()`, `.simulate()`, `.steady_states()` surface.

---

## Command-line interface

Installing strider registers a `strider` console script. Every subcommand takes `--json` for machine-readable output, and any sequence argument accepts `-` for stdin or `@path` for a file.

```bash
# MFE structure
$ strider fold GCGCAAAAGCGC
GCGCAAAAGCGC
((((....))))
ΔG = -2.350 kcal/mol  (4 bp)

# Ensemble ΔG (single- or multi-strand)
$ strider pfunc GCGCAAAAGCGC
ΔG_ens = -3.127 kcal/mol  (Z = 159.7, backend=native)

$ strider pfunc GCGCATGC GCATGCGC --backend nupack

# Duplex ΔG and melting temperature
$ strider duplex GCGCATGC                     # auto-uses reverse complement
$ strider duplex GCGCATGC GCATGCGC --sodium 0.05

# Tm only
$ strider melt GCGCATGCATGC --strand-conc 1e-7

# Co-transcriptional folding trajectory
$ strider cotx GGGAAACCCAAAGGG --min-length 5 --material rna
   5  .....              ΔG=+0.000
   ...
  15  (((...)))(((...))) ΔG=-2.240

# CHA / circuit verification from a JSON sequence spec
$ strider verify cha_spec.json
```

All commands accept `--celsius`, `--material {dna,rna}`, `--sodium`, `--magnesium`, and `--backend` where relevant. Run `strider <cmd> --help` for full options.

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

Salt corrections for non-1M NaCl and Mg²⁺ are applied automatically when `sodium ≠ 1.0` or `magnesium > 0`. Two distinct corrections are wired in:

- **Duplex / melting temperature** — Owczarzy et al. (2004) for Na⁺ and Owczarzy et al. (2008) for Mg²⁺, with a mixed-ion regime from the √[Mg²⁺]/[Na⁺] ratio.
- **Partition function / ensemble ΔG** — per-base-pair correction ``ΔG_per_bp = −0.114·ln([Na⁺] + 3.4·√[Mg²⁺])`` kcal/mol, applied to each closed pair inside the McCaskill DP so it is automatically ensemble-weighted by the pair probability. This is an empirical fit to NUPACK pfunc (±0.005 kcal/mol/bp over Na⁺ ∈ [0.05, 1.0] M, Mg²⁺ ∈ [0, 0.1] M); see `strider.thermo.salt.dg_per_bp_salt`.

The two formulas serve different purposes (Tm uses the original Owczarzy Tm-shift form; pfunc needs a per-pair ΔG that integrates over the structural ensemble). Both reduce to zero at 1 M Na⁺ / 0 Mg²⁺, the SantaLucia/Turner reference state.

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

#### Co-transcriptional folding

`fold_cotranscriptional()` sweeps prefix lengths and folds each one, returning the trajectory of structures the strand passes through while being transcribed. This matters for riboswitches, aptamers, and any RNA whose biology depends on a kinetic intermediate rather than the final MFE:

```python
from strider import fold_cotranscriptional

traj = fold_cotranscriptional('GGGAAACCCAAAGGG', material='rna', min_length=5)
for p in traj.prefixes:
    print(f'{p.length:>3}  {p.structure}  ΔG={p.energy:+.2f}')

# Detect where existing pairs broke as 3' sequence arrived
print(traj.rearrangements())   # e.g. [(9, 12)] — refold between length 9 and 12
print(traj.final().structure)  # fully-transcribed MFE
```

`step=N` subsamples every Nth prefix for long sequences. The full-length prefix is always included regardless of step.

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

### 4. Boltzmann sampling and subopt enumeration

When the MFE structure alone misrepresents the ensemble (e.g. competing folds within a few kcal/mol of the optimum), two routines on top of the partition function help inspect what's really happening.

#### Boltzmann sampling

Draw `n` structures distributed according to their equilibrium probabilities (Ding & Lawrence 2003, stochastic traceback over the Qb/Q/QM matrices):

```python
from strider import sample_structures
from collections import Counter

samples = sample_structures('GCGCGCAAAAGCGCGC', n_samples=100, seed=0)
counts = Counter(db for db, _ in samples)
for db, n in counts.most_common(5):
    print(f"{n:3d}  {db}")
# 78  ((((((....))))))      ← MFE dominates for a strong hairpin
#  9  ............
#  5  (((((......))))).
#  ...
```

#### Suboptimal-structure enumeration

Enumerate *all* structures within `gap` kcal/mol of the MFE (Wuchty-style worklist over the V/W matrices, energy-pruned):

```python
from strider import subopt_structures

for db, e, _ in subopt_structures('GCGCAAAAGCGC', gap=3.0, max_structures=20):
    print(f"{e:7.3f}  {db}")
# -2.350  ((((....))))
# -0.110  .(((....))).
# -0.010  (((......)))
#  0.000  ............
```

Both procedures are also exposed as engine methods (`engine.sample(seq, n)` and `engine.subopt(seq, gap)`) for use inside design objectives.

---

### 5. TMSD kinetics

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

### 6. Leakage enumeration

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

### 7. Equilibrium concentration solver

`solve_equilibrium()` returns the equilibrium concentration of every complex in a multi-strand mixture given total strand concentrations and per-complex partition functions. It solves the convex dual of the standard mass-action problem (Dirks et al. 2007) via damped Newton iteration on chemical potentials, matching NUPACK's `tube_analysis` to within ~1 % on the validated test cases.

```python
from strider import solve_equilibrium

res = solve_equilibrium(
    complexes={
        'A':  (['A'],      0.0),
        'B':  (['B'],      0.0),
        'AB': (['A', 'B'], -10.0),     # ΔG of the dimer (kcal/mol, 1 M standard)
    },
    totals={'A': 1e-7, 'B': 1e-7},
    celsius=37.0,
)
print(res.converged, res.iterations, res.residual)
print(res.concentrations)              # {'A': 6.0e-8, 'B': 6.0e-8, 'AB': 4.0e-8}
print(res.strand_free)                 # {'A': 6.0e-8, 'B': 6.0e-8}
```

For a NUPACK round-trip — where NUPACK reports ΔG at the water-molarity (~55 M) standard state instead of 1 M — pass `standard_state_M=water_molarity(celsius)` so the solver applies the corresponding shift:

```python
from strider import solve_equilibrium, water_molarity
res = solve_equilibrium(complexes, totals, standard_state_M=water_molarity(37.0))
```

#### Auto-enumeration from a ThermoEngine

`equilibrium_from_engine` enumerates all complexes up to a chosen strand count, computes each pfunc with the active backend, then solves:

```python
from strider import equilibrium_from_engine, ThermoEngine

engine = ThermoEngine(material='dna', celsius=37, sodium=0.137, magnesium=0.01)
res = equilibrium_from_engine(
    engine,
    strands={'A': 'GCGCGCAAAA', 'B': 'TTTTGCGCGC', 'C': 'GCATATGC'},
    totals={'A': 1e-7, 'B': 1e-7, 'C': 1e-7},
    max_size=3,                         # enumerate monomers, dimers, trimers
)
for name, c in sorted(res.concentrations.items(), key=lambda kv: -kv[1])[:5]:
    print(f"{name:6s} {c:.2e} M")
```

#### Rotational symmetry

`cyclic_symmetry(strand_list)` returns the cyclic-symmetry number σ used to correct homomeric multi-strand pfunc values. Strider's native backend applies this correction automatically inside `ThermoEngine.pfunc` so that *every* backend reports species-level (not ordered-complex) ΔG. The same applies when feeding NUPACK pfunc values into `solve_equilibrium`: NUPACK's reported Q already includes σ, so no double correction is needed.

---

### 8. Sequence design and the Assay abstraction

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
| `DesignObjective.ensemble_defect(engine, strand_names, target_structure)` | NUPACK-style expected mispaired nucleotides vs target dot-bracket |
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

#### Assay / AssayPanel — high-level design intent

For circuit-level design, `Assay` bundles a set of on-target assemblies (each with its dot-bracket target and expected concentration) plus off-target assemblies that must *not* form. It compiles into a `DesignObjective` that the designer minimizes:

```python
from strider import Assay, AssayPanel, Assembly, SequenceDesigner, DomainSpec

assay = Assay(
    name='hairpin_sensor',
    on_targets=[
        Assembly('H', ['H'], '((((....))))', concentration=1e-7),
    ],
    off_targets=[
        Assembly('H_H', ['H', 'H']),       # forbid homodimer
    ],
    off_target_ddg_threshold=-4.0,         # penalize binding stronger than this
)

designer = SequenceDesigner(engine, seed=0)
result = designer.design(
    domains={'H': DomainSpec(length=12)},
    objective=assay.to_objective(engine),
    n_trials=4,
    max_iterations=200,
)
```

An `AssayPanel` sums multiple `Assay` objectives so you can design across several test tubes at once:

```python
panel = AssayPanel(assays=[assay_low_temp, assay_high_temp])
objective = panel.to_objective(engine)
```

---

### 9. Mutation sensitivity analysis

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

### 10. Off-target screening

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

### 11. Circuit catalog and the mantis bridge

strider ships a catalog of DSD circuit templates under `strider.circuits`. All templates share the same API: a strand set, a reaction topology, a toehold map, and a default check suite. Each emits a `CircuitBridge` for use with the mantis simulator.

```python
from strider import CHA, HCR, SeesawGate, Translator
```

#### CHA — Catalytic Hairpin Assembly

```python
cha = CHA(
    sequences={
        'mirna': 'TAGCTTATCAGACTGATGTTGA',
        'H1':    'TCAACATCAGTCTGATAAGGAGGGAGGTTATCAGACTGA',
        'H2':    'TCAGTCTGATAAGGAGGGAGGTATCAGACTGATGTTGATTTTT',
        'CP':    'AAAAA',
    },
    toehold_d1=6,    # miRNA·H1 toehold length
    toehold_d2=11,   # H2 branch migration domain
    tail_cp=9,       # CP tail length
)
```

#### HCR — Hybridization Chain Reaction

```python
hcr = HCR(
    sequences={'I': '...', 'H1': '...', 'H2': '...'},
    toehold_initiator=6,
    toehold_branch=6,
)
```

#### SeesawGate — Qian-Winfree compute primitive (YES / AND / OR / NOT)

```python
gate = SeesawGate(
    logic='AND',
    sequences={
        'Input1': '...', 'Input2': '...',
        'Gate': '...', 'Threshold_Input1': '...',
        'Threshold_Input2': '...', 'Fuel': '...', 'Output': '...',
    },
    toehold=6,
)
```

#### Translator — input strand X triggers release of output strand Y

```python
tr = Translator(
    sequences={'X': '...', 'Y': '...', 'Gate': '...'},
    toehold_x=6,
)
```

#### Verification via CheckRegistry

Every template has a `verify()` method that runs its default check suite and returns a structured `CircuitReport`:

```python
report = cha.verify()
print(report)
```

```
CHA: PASS
  ✓ toehold_accessible: 0.996 (prob) — unpaired probability 1.00 (≥ 0.50)
  ✓ H1_stability: -5.07 kcal/mol — normalised ΔG -5.07 kcal/mol (in [-12, -4])
  ✓ H2_stability: -4.51 kcal/mol — normalised ΔG -4.51 kcal/mol (in [-12, -4])
  ✓ R1_driving_force: -10.54 kcal/mol — ΔΔG -10.54 kcal/mol (≤ -3.0)
  ✓ R2_driving_force: -12.89 kcal/mol — ΔΔG -12.89 kcal/mol (≤ -3.0)
  ✓ R3_driving_force: -9.56 kcal/mol — ΔΔG -9.56 kcal/mol (≤ -8.0)
  ✓ CP_leakage: -5.14 kcal/mol — ΔΔG -5.14 kcal/mol (≥ -6.0)
  ✓ spontaneous_leakage: 1.49e-07 ratio — leak/signal = 1.49e-07 (≤ 1e-04)
```

Build your own checks with `CheckRegistry`:

```python
from strider import CheckRegistry, stability_in_range, no_spurious_dimer

custom = (CheckRegistry()
    .add(stability_in_range('H1', min_dg=-10, max_dg=-5))
    .add(no_spurious_dimer('H1', 'CP', min_ddg=-4.0))
)
report = cha.verify(registry=custom)
```

Built-in checks: `toehold_accessible`, `stability_in_range`, `reaction_driving_force`, `no_spurious_dimer`, `leakage_below_signal`, and `custom(name, fn)` for arbitrary user predicates.

#### Exporting to mantis

Every template has the same downstream methods:

```python
rn = cha.to_crnetwork()                # → mantis.CRNetwork
sim = cha.simulate(initial_conditions, (0, 7200))
ss  = cha.steady_states(initial_conditions)
```

#### Defining your own circuit

Subclass `CircuitTemplate` to add a new topology — declare reactions and a default check registry, get the full pipeline for free:

```python
from dataclasses import dataclass
from strider import CircuitTemplate, CheckRegistry, reaction_driving_force

@dataclass
class MyAmplifier(CircuitTemplate):
    def __post_init__(self):
        if self.name == 'circuit':
            self.name = 'MyAmplifier'
        self.reactions = ['A + B <-> AB', 'AB + C -> AC + B']
        self.toehold_map = {'A + B <-> AB': 6}

    def _default_checks(self):
        return (CheckRegistry()
            .add(reaction_driving_force(['A', 'B'], [['A', 'B']], max_ddg=-3.0)))
```

#### Generic CircuitBridge

For ad-hoc circuits without a dedicated template, `CircuitBridge` accepts any list of reaction strings:

```python
from strider import CircuitBridge

bridge = CircuitBridge(
    reactions=['A + B <-> AB', 'AB + C <-> ABC'],
    sequences={'A': '...', 'B': '...', 'C': '...'},
    include_leakage=True,
    leakage_threshold=-4.0,
)
rn = bridge.to_crnetwork()
```

> **Compatibility note:** `CHABridge` from the prior API is still available and unchanged, but new code should prefer `strider.circuits.CHA`, which uses the generic check registry and composes with other templates.

---

### 12. DSDCompiler — domain-level circuit assembly

`DSDCompiler` lets you describe a circuit in *domain space* — registered domains plus strands defined as ordered domain lists — and resolves to nucleotide sequences automatically, including reverse-complement (`a*`) generation.

```python
from strider import DSDCompiler

dsd = DSDCompiler(domains={
    't': 'GCATGC',            # toehold
    'a': 'ATGCATATGC',         # branch migration region
    'b': 'TTGCATGCAA',         # extension
})
dsd.add_strand('S1', ['t', 'a', 'b'])
dsd.add_strand('S2', ['b*', 'a*', 't*'])         # auto-derived complements
dsd.add_reaction('S1 + S2 <-> S1_S2', toehold='t')

print(dsd)                                        # pretty-printed circuit
bridge = dsd.to_bridge()                          # CircuitBridge
rn = bridge.to_crnetwork()
```

The compiler intentionally does *not* infer reactions from strand topology — you still write them explicitly. The job is to keep the symbolic layer (domains, strands) in sync with the sequence layer.

---

### 13. Stochastic simulation via mantis

For low-copy-number regimes where deterministic ODE breaks down (e.g. single-cell concentrations, stochastic switching in bistable circuits), mantis provides a Gillespie SSA direct-method simulator:

```python
rn = cha.to_crnetwork()

# 100 µL = 1e-4 L  →  10 nM mirna ≈ 600 molecules
result = rn.stochastic_simulate(
    initial_conditions={'mirna': 10e-9, 'H1': 100e-9, 'H2': 100e-9, 'CP': 100e-9},
    t_span=(0.0, 60.0),
    volume_L=1e-4,
    seed=0,
)
print(result.n_events, result.success)
print(result.final())               # {'mirna': ..., 'H1': ..., ...}
print(result.counts['H1_H2'][-1])   # integer molecule count
```

`StochasticResult` carries both `.counts` (integer arrays) and `.concentrations` (M). For cellular volumes use `volume_L ≈ 1e-15` and `initial_as='count'` to specify molecule counts directly. The deterministic `simulate()` and the stochastic `stochastic_simulate()` should agree in the high-count limit; for `< ~10³` molecules they typically diverge.

---

### 14. Parameter sweeps and caching

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

### 15. Export formats

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
| `pfunc(*sequences)` | `PFuncResult` | Ensemble free energy and pair probability matrix (σ-corrected for homomeric multi-strand) |
| `pairs(*sequences)` | `np.ndarray` | Pair-probability matrix only |
| `ensemble_defect(seqs, target_structure, normalize=True)` | `float` | Expected mispaired nucleotides vs a target dot-bracket |
| `sample(seq, n_samples, seed=None)` | `list[(str, list)]` | Boltzmann-sampled structures |
| `subopt(seq, gap=1.0, max_structures=200)` | `list[(str, float, list)]` | Suboptimal structures within `gap` of MFE |
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

### Circuit templates

All templates subclass `CircuitTemplate` and share the same downstream surface.

| Class | Required keys | Key parameters |
|---|---|---|
| `CHA(sequences, ...)` | `mirna`, `H1`, `H2`, `CP` | `toehold_d1=6, toehold_d2=11, tail_cp=9` |
| `HCR(sequences, ...)` | `I`, `H1`, `H2` | `toehold_initiator=6, toehold_branch=6` |
| `Translator(sequences, ...)` | `X`, `Y`, `Gate` | `toehold_x=6` |
| `SeesawGate(sequences, ...)` | `Input1`, [`Input2`,] `Gate`, `Threshold[_InputN]`, `Fuel`, `Output` | `logic='YES'\|'AND'\|'OR'\|'NOT'`, `toehold=6` |

Shared methods:

| Method | Returns | Description |
|---|---|---|
| `to_bridge(include_leakage=False, leakage_threshold=-4.0)` | `CircuitBridge` | Build the generic mantis bridge |
| `to_crnetwork(**kw)` | `mantis.CRNetwork` | Shortcut: bridge → network |
| `simulate(ic, t_span, **kw)` | `SimulationResult` | Deterministic ODE |
| `steady_states(ic, **kw)` | `list[SteadyState]` | mantis steady-state finder |
| `verify(registry=None)` | `CircuitReport` | Run default (or user) check suite |

### `CircuitBridge` and `CHABridge`

`CircuitBridge(reactions, sequences, engine=None, toehold_map=None, include_leakage=False, leakage_threshold=-4.0)` — generic, accepts any reaction topology. Returned by every template's `to_bridge()`.

`CHABridge(sequences, ...)` is retained for backwards compatibility — same parameters and API as in the original 0.1.0 release. New code should prefer `circuits.CHA`.

### `CheckRegistry`

`CheckRegistry().add(check).add(check)...` → use `.run(engine, sequences, name=...)` to produce a `CircuitReport`.

| Built-in check | Signature |
|---|---|
| `toehold_accessible(strand, positions, min_prob=0.5)` | Strand's toehold positions are unpaired ≥ `min_prob` of the ensemble |
| `stability_in_range(strand, min_dg, max_dg, reference_length=20)` | Normalized hairpin ΔG falls in the sweet spot |
| `reaction_driving_force(reactants, products, max_ddg=-3.0)` | ΔΔG of the reaction is sufficiently favorable |
| `no_spurious_dimer(a, b, min_ddg=-6.0)` | Pairwise dimer is NOT too stable |
| `leakage_below_signal(signal_kf, hairpin, ratio=1e-4)` | Spontaneous breathing rate is ≥ `ratio`× slower than signal |
| `custom(name, fn)` | Wrap any `(ctx) → (passed, value, msg)` function |

### `solve_equilibrium`

```python
solve_equilibrium(
    complexes,                # {name: ([strand_names], dG_kcal_per_mol)}
    totals,                   # {strand_name: total_concentration_M}
    celsius=37.0,
    max_iter=200,
    tol=1e-9,
    standard_state_M=1.0,     # use water_molarity(celsius) for NUPACK input
) → EquilibriumResult
```

Companions: `equilibrium_from_engine(engine, strands, totals, max_size=2)` for auto-enumeration, `cyclic_symmetry(strand_list)` and `water_molarity(celsius)` helpers.

### `Assay` / `AssayPanel` / `Assembly`

```python
Assembly(name, strands, structure=None, concentration=1e-6)
Assay(name, on_targets=[Assembly, ...], off_targets=[Assembly, ...],
      off_target_ddg_threshold=-4.0, off_target_penalty_weight=1.0)
AssayPanel(assays=[Assay, ...])
```

Methods: `defect(sequences, engine) → float`, `to_objective(engine, weight=1.0) → DesignObjective`.

### `DSDCompiler`

```python
DSDCompiler(domains={name: sequence}).add_strand(name, [domain, ...])
                                     .add_reaction(rxn_str, toehold=...)
                                     .to_bridge() → CircuitBridge
```

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
2. `CHABridge.verify()` — seven-check audit (the 0.1.0 API; the new `circuits.CHA().verify()` runs the same checks via the generic `CheckRegistry`)
3. `bridge.to_crnetwork()` — export to mantis CRNetwork
4. ODE integration and steady-state finding via mantis
5. `bridge.sensitivity()` — one-at-a-time rate sensitivity analysis
6. Predicted signal vs. miRNA concentration

> For non-CHA topologies, swap `CHABridge` for `HCR(...)`, `SeesawGate(logic='AND', ...)`, `Translator(...)`, or any custom `CircuitTemplate` subclass — the rest of the pipeline is unchanged.

---

## Backend comparison

The native backend uses strider's own McCaskill O(n³) partition-function DP with nick-aware recursions for multi-strand complexes — the same family of algorithm as ViennaRNA (RNAcofold) and NUPACK. NUPACK output is auto-shifted to the 1 M standard state (matching the SantaLucia / Turner convention) and the multi-strand pfunc applies the σ rotational correction internally, so engine output is consistent across backends.

All numbers below at physiological salt (Na⁺=0.137 M, Mg²⁺=0.01 M, the engine default).

| Calculation | Native | NUPACK | Gap |
|---|---|---|---|
| Single hairpin ΔG (12 nt) | −3.13 | −3.12 | **0.01 kcal/mol** |
| Single hairpin ΔG (16 nt GC stem) | −7.36 | −7.35 | **0.01 kcal/mol** |
| Bimolecular short duplex (20 nt total) | −14.80 | −13.36 | 1.4 kcal/mol |
| Bimolecular exact-complement (24 nt total) | −14.13 | −12.67 | 1.5 kcal/mol |
| Bimolecular partial-complement (67 nt total) | −21.88 | −22.23 | 0.4 kcal/mol |
| ΔG(spont): H1 + H2 → H1·H2 (106 nt complex) | −54.00 | −52.38 | 1.6 kcal/mol |

**Where the native backend matches well:** **single hairpins under physiological salt** (mean bias −0.002 kcal/mol, max 0.03 kcal/mol on the cases pinned by `tests/test_native_vs_nupack_accuracy.py`). Concentration-solver round-trips agree with NUPACK `tube_analysis` to ~1 %.

**Where it diverges:** multi-strand complexes, where native over-stabilizes by ~0.4–1.6 kcal/mol depending on the topology. The residual is small enough for design-iteration use but matters for absolute affinity predictions — prefer `nupack` (or `vienna` for single-strand MFE) for publication numbers.

**History:** an earlier 0.15–0.50 kcal/mol systematic over-stabilization on *single hairpins* came from a missing per-base-pair salt correction in the McCaskill DP, now wired in via `strider.thermo.salt.dg_per_bp_salt` (see [Salt corrections](#salt-corrections)). A previously documented 10 kcal/mol multi-loop gap was also a pre-fix artifact and no longer applies.

**When to use each backend:**

| Scenario | Recommended backend |
|---|---|
| Rapid screening / design iteration (< 40 nt hairpins) | `native` |
| Concentration solver / equilibrium analysis | `native` (matches NUPACK to ~1 %) |
| MFE folding of sequences up to ~200 nt | `vienna` |
| High-accuracy partition function for long multi-strand complexes | `nupack` |
| No external dependencies (CI, lightweight environments) | `native` |
| Publication-quality thermodynamics | `nupack` or `vienna` |

---

## Running the tests

```bash
cd strider
pip install -e .[dev]
pytest tests/ -v
```

The test suite has **216 tests** (228 in `nupack_env` where NUPACK round-trip tests light up):

| File | Tests | What is covered |
|---|---|---|
| `test_thermo_dna.py` | 16 | NN parameters, duplex_dg, Tm, salt corrections, self-complementarity |
| `test_tmsd.py` | 15 | toehold_kf table, Arrhenius correction, detailed balance, leakage_kf, Keq conversions |
| `test_design.py` | 19 | SequenceDesigner SA convergence, DomainSpec, hard constraints, ensemble defect, MutationAnalyzer |
| `test_mfe.py` | 12 | fold_mfe correctness, dot-bracket parsing, mountain vectors, structure comparison |
| `test_sampling.py` | 11 | Boltzmann sampling distribution, subopt enumeration, energy gap correctness |
| `test_equilibrium.py` | 17 | concentration solver convergence, σ correction, water-molarity standard state |
| `test_circuits.py` | 20 | CheckRegistry, CHA/HCR/Translator/SeesawGate templates, custom-registry composition |
| `test_bridge.py` | 15 | CHABridge ddg_pathway, verify() checks, CircuitBridge generic topology, mantis integration |
| `test_dsd.py` | 15 | DSDCompiler domain resolution, strand assembly, bridge integration |
| `test_assay.py` | 8 | Assay/AssayPanel defect, off-target penalty, designer integration |
| `test_formats.py` | 7 | Vienna, CT, BPSEQ, FASTA, oxDNA output; round-trip pair parsing |
| `test_leakage.py` | 5 | LeakageEnumerator pairwise enumeration, pathway classification, filter() |
| `test_screener.py` | 6 | Off-target k-mer screening |
| `test_cotranscriptional.py` | 9 | Prefix-by-prefix folding trajectory, rearrangement detection |
| `test_cli.py` | 14 | `strider fold/pfunc/duplex/melt/cotx`, JSON output, stdin / @file sequence input |
| `test_stack_vs_nupack.py` | 5 | (nupack_env only) — solver vs NUPACK `tube_analysis` round-trip, kinetics → equilibrium |
| `test_native_vs_nupack_accuracy.py` | 6 | (nupack_env only) — native pfunc within 0.05 kcal/mol of NUPACK at physiological salt |

> **Note:** tests requiring `mantis-delta` are skipped if it is not installed (install via `pip install -e ../mantis` for editable mode).  Tests requiring `nupack` only run when invoked from a Python that has NUPACK importable.

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

### `CHA().verify()` fails spontaneous leakage check

A failing `spontaneous_leakage` check means H1 and H2 hybridize too favorably in the absence of trigger. Common causes:
- H1 and H2 have long complementary stems outside the intended domains
- The stem domain (D2) is too long or too GC-rich
- The D3 spacer is not introducing enough disruption

Use `SequenceDesigner` with `DesignObjective.minimize_leakage()` weighted heavily, or add `HardConstraint.no_self_complement(min_length=6)` to suppress cross-complementarity.  The same applies to any `CircuitTemplate` that includes a `leakage_below_signal` check.

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
