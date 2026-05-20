"""
Example 08 — Multi-strand tube analysis.

Builds a two-strand "test tube" with monomers, homodimers, and heterodimers
allowed to form, then computes the equilibrium concentrations of every
species at 37 °C.

The :class:`strider.Tube` API is the high-level wrapper around
:func:`strider.solve_equilibrium`: enumerate complexes, evaluate each one's
ensemble free energy with a :class:`ThermoEngine`, hand the table to the
mass-balance solver.  Pair probabilities and ensemble defect are then
available on demand for any complex in the result.

Run::

    python examples/08_tube_analysis.py
"""

from strider import (
    ComplexSet,
    SetSpec,
    Strand,
    ThermoEngine,
    Tube,
    tube_analysis,
)


def main() -> None:
    # Two short DNA strands with imperfect complementarity.
    H1 = Strand("H1", "GCAGTGAGACGAGCTGCT", material="dna")
    H2 = Strand("H2", "AGCAGCTCGTCTCACTGC", material="dna")

    engine = ThermoEngine(
        material="dna",
        celsius=37.0,
        sodium=0.137,
        magnesium=0.01,
    )

    # Build two tubes that differ only in total concentration so we can see
    # how dimer occupancy scales.
    tubes = [
        Tube(
            name="dilute_100nM",
            strand_totals={H1: 1e-7, H2: 1e-7},
            complexes=ComplexSet([H1, H2], SetSpec(max_size=2)),
        ),
        Tube(
            name="dense_10uM",
            strand_totals={H1: 1e-5, H2: 1e-5},
            complexes=ComplexSet([H1, H2], SetSpec(max_size=2)),
        ),
    ]

    results = tube_analysis(tubes, engine)

    for name, res in results.items():
        print(f"\n=== Tube: {name} (converged={res.converged}) ===")
        print(f"{'species':10s}  {'ΔG (kcal/mol)':>14s}  {'[X] (M)':>14s}")
        print("-" * 44)
        # Sort by concentration descending for readability.
        ranked = sorted(res.concentrations.items(), key=lambda kv: kv[1], reverse=True)
        for species, conc in ranked:
            dG = res.free_energies.get(species, 0.0)
            print(f"{species:10s}  {dG:>14.2f}  {conc:>14.3e}")

    # Pair-probability matrix for the most interesting species — the H1·H2
    # heterodimer in the denser tube.
    res = results["dense_10uM"]
    if "H1_H2" in res.concentrations:
        P = res.pair_probabilities("H1_H2")
        print(f"\nH1·H2 pair-probability matrix shape: {P.shape}")
        print(f"  max P(i,j) = {P.max():.3f}")
        print(f"  sum_ij P  ≈ {P.sum() / 2:.1f} expected base pairs")


if __name__ == "__main__":
    main()
