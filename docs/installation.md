# Installation

## From PyPI (recommended)

```bash
pip install strider-dna
```

## From source

```bash
git clone https://github.com/EmilioVenegas/strider.git
cd strider
pip install -e .
```

## Optional backends

```bash
# ViennaRNA Python bindings (GPL)
pip install ViennaRNA

# NUPACK (separate academic license required — see nupack.org)
# Install per NUPACK documentation, then:
pip install strider-dna[nupack]
```

## Verify the installation

```python
import strider

eng = strider.ThermoEngine()
print(eng.backend_name)          # "native", "vienna", or "nupack"
print(eng.available_backends())  # list of all usable backends
```

## Requirements

- Python ≥ 3.10
- NumPy (pulled in automatically)
- Optional: ViennaRNA ≥ 2.5, NUPACK ≥ 4.0
