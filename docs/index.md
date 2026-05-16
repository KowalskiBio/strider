# strider-dna

**Nucleic acid thermodynamics, kinetics, and circuit design — pure-Python, no restrictive licenses.**

[![Tests](https://img.shields.io/badge/tests-76%20passed-brightgreen)](https://github.com/EmilioVenegas/strider/actions)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://pypi.org/project/strider-dna)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/EmilioVenegas/strider/blob/main/LICENSE)

strider is an open-source Python library for nucleic acid biophysics built on published open science: SantaLucia & Hicks 2004 nearest-neighbour parameters, Zhang & Winfree 2009 TMSD kinetics, and McCaskill's 1990 partition-function dynamic program. It runs without NUPACK or ViennaRNA (though both can be used as optional backends), making it suitable for open research and reproducible pipelines.

## Key features

- **Nearest-neighbour thermodynamics** — ΔH/ΔS/Tm for DNA, RNA, and DNA:RNA hybrids with Owczarzy salt corrections
- **Ensemble free energies** — McCaskill O(n³) partition-function DP; matches ViennaRNA to ~0.1 kcal/mol
- **Structure prediction** — MFE folding with H-type pseudoknot detection
- **TMSD kinetics** — Zhang & Winfree 2009 toehold-exchange rate model with Arrhenius temperature correction
- **Sequence design** — Simulated-annealing optimizer with composable, weighted `DesignObjective` functions
- **mantis bridge** — `CHABridge(sequences).to_crnetwork()` returns a ready-to-simulate `mantis.CRNetwork`
- **Parameter sweeps** — N-D grid sweeps with SQLite3 disk cache and multiprocessing
- **Export** — Vienna, CT, BPSEQ, FASTA, and oxDNA formats

## Quick start

```python
import strider

eng = strider.ThermoEngine()
dg = eng.duplex_dg("GCATGC", complement=True)
print(f"ΔG = {dg:.2f} kcal/mol")   # ΔG = -8.3 kcal/mol
```

See the [Installation](installation.md) page to get started.

## Design philosophy

NUPACK has a restrictive academic license that limits reproducibility in open research. strider is built entirely on published algorithms and parameters, carries an MIT license, and can be installed from PyPI with a single command. Optional ViennaRNA and NUPACK backends are supported for cross-validation but are never required.
