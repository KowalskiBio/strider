"""
Example 10: Template-free reaction enumeration — strands → CRN, no hand-written reactions

`DSDCompiler` keeps a circuit's sequence layer in sync but still needs you to *write*
the reactions.  `DomainReactionEnumerator` derives them: it reads the strand topology
and enumerates the reachable complexes plus the bind / branch-migration / open
transitions between them (the Visual DSD / Peppercorn job), assigns detailed-balance
rate constants from the active ThermoEngine, and hands you a simulable mantis CRNetwork.

Here we enumerate a textbook toehold-mediated strand displacement (TMSD):

    Invader (t b) invades the Output·Base substrate, displaces Output via branch
    migration on domain b, and is released as the Invader·Base product.
"""

from strider import ThermoEngine, DomainReactionEnumerator

engine = ThermoEngine(material="dna", celsius=37.0, sodium=0.137, magnesium=0.01)

# ── 1. Domains (t = 4-nt toehold, b = 12-nt branch-migration domain) ──────────
enum = DomainReactionEnumerator(
    domains={"t": "CCCT", "b": "ACGTACGTACGT"},
    engine=engine,
)

# ── 2. Strands in domain space; Output·Base starts pre-hybridised ─────────────
result = enum.enumerate(
    strands={
        "Invader": ["t", "b"],     # toehold + branch
        "Output":  ["b"],          # incumbent output strand
        "Base":    ["b*", "t*"],   # substrate bottom strand
    },
    initial_complexes=[["Output", "Base"]],
)

# ── 3. The derived network ────────────────────────────────────────────────────
print(result.summary())

# ── 4. Straight to mantis and simulate ────────────────────────────────────────
crn = result.to_crnetwork()
print(f"\nCRNetwork: {crn.n_species} species, {crn.n_reactions} reactions, "
      f"deficiency {crn.deficiency}")

substrate = next(s for s in crn.species if s.startswith("Base_Output"))
ic = {s: 0.0 for s in crn.species}
ic["Invader"] = 1e-7      # 100 nM
ic[substrate] = 1e-7      # 100 nM

traj = crn.simulate(ic, (0, 3600))
print(f"\nOutput released after 1 h: {traj.final()['Output'] * 1e9:.1f} nM "
      f"(of 100 nM)")
