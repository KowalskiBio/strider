# Changelog

All notable changes to strider-dna are documented here.
This file is generated from the git history by [git-cliff](https://git-cliff.org).
The format follows [Keep a Changelog](https://keepachangelog.com) and the project
uses [Semantic Versioning](https://semver.org).

## [0.3.2] - 2026-06-11

### Features

- Add DNA loop ΔH tables for temperature-resolved energetics
- Structure-resolved ΔG/ΔH for hairpin Tm; unify hairpin API

## [0.3.1] - 2026-06-11

### Features

- Add hairpin melting temperature calculation and thermodynamics module
- Add hairpin thermodynamic functions and update test coverage badge

## [0.3.0] - 2026-06-11

### Features

- Implement StochasticSurfaceModel to estimate shot-noise-limited LOD using Currie statistics and mantis SSA
- Implement differentiable soft_forward for sequence-to-energy gradient optimization

### Refactor

- Simplify Tm calculation and add project branding assets to README

## [0.2.0] - 2026-06-08

### Bug Fixes

- Correct K_CURRIE constant to match the canonical Currie limit of 2.71 counts

### Documentation

- Refresh README for Workstream A–D APIs and new design optimiser
- Surface benchmark receipts in README + refresh test counts
- Add closed-loop example + document new objectives

### Features

- Loadable Turner-style ParameterSet (Workstream A)
- Full Zuker MFE sharing energy code with pfunc (Workstream B)
- Strand / Complex / Tube / ComplexSet API (Workstream C)
- Defect-based design optimizer (Workstream D)
- Thread ParameterSet into the energy DP (Workstream A follow-up)
- Head-to-head accuracy + timing receipts
- Expand ParameterSet schema to every advanced sub-table
- Add closed-loop dynamical objective factories
- Implement differentiable PyTorch thermodynamics engine for trainable McCaskill partition functions and add associated tests
- Add surface transducer model for signal prediction and implement associated structure sampling logic
- Template-free domain-level reaction enumerator (Peppercorn paradigm)
- Align differentiable McCaskill + batched/GPU backend (native speed)
- Implement G-quadruplex folding thermodynamics and motif identification

### Testing

- Cover dynamical objective factories

## [0.1.0] - 2026-05-18

### Features

- Initialize Strider DNA/RNA design and thermodynamic simulation framework
- Add equilibrium concentration solver, differentiable thermo parameters, and accuracy benchmarking tools
- Implement circuit design framework with CHA and Seesaw templates and add associated verification tools and CLI.

### Refactor

- Restructure documentation by consolidating guides into index and generating a centralized API reference file
- Update thermo energy calculations to use multi-strand partition functions and improve off-target screening index and logic.


