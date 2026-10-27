# Amber Lipid21 for GROMACS

GROMACS port of the Amber **Lipid21** lipid force field

> C. J. Dickson, R. C. Walker, I. R. Gould, *Lipid21: Complex Lipid Membrane Simulations with AMBER*,
> J. Chem. Theory Comput. **18**, 1726–1736 (2022), doi:10.1021/acs.jctc.1c01217

converted from `lipid21.dat` and `lipid21.lib` as shipped with AmberTools26 (byte-identical to
<https://github.com/callumjd/lipid21>, cloned in `lipid21/`), and validated term by term and by
energies/forces against tleap, sander and pmemd (see [validation/REPORT.md](validation/REPORT.md)).

## Contents

| path | |
|---|---|
| `lipid21.ff/forcefield.itp` | standalone entry point: Amber `[ defaults ]` + Lipid21 + TIP3P/ion atom types |
| `lipid21.ff/lipid21_params.itp` | add-on entry point: Lipid21 atom types + bonded parameters, to include after `amber19sb.ff` or `amber14sb.ff` |
| `lipid21.ff/ffnonbonded.itp`, `ffbonded.itp` | the converted `lipid21.dat` (20 atom types, bonds, angles, torsions, impropers) |
| `lipid21.ff/<LIPID>.itp` | 43 lipid molecule types (list below) |
| `lipid21.ff/tip3p.itp`, `ions_tip3p.itp`, `ffwater_ions.itp` | TIP3P and Na+, K+, Cl-, Mg2+, Ca2+ for the standalone force field |
| `mdp/` | energy minimisation and NPT production settings for bilayers |
| `tools/lipid21_gmx.py` | the converter (writes `lipid21.ff`, or one `.itp` for any residue combination) |
| `tools/lipid21_pdb2gmx.py` | Amber/Lipid21-named membrane PDB -> `.gro` + `.top` |
| `tools/compare_prmtop.py`, `validate_*.py`, `pme_precision_test.py`, `md_summary.py` | validation against AmberTools, MD analysis |
| `validation/` | validation results and report |

## Lipids

Each lipid is one molecule type made of the Lipid21 residues, tail–head–tail, with Amber residue
and atom names and atom order (as in tleap, packmol-memgen and `charmmlipid2amber.py` output).

| head group (Lipid21 residue) | lipids (`lipid21.ff/<name>.itp`) |
|---|---|
| PA (`PH-`) | DLPA, DMPA, DPPA, DSPA, DOPA, DAPA, POPA, SDPA |
| PC (`PC`) | DLPC, DMPC, DPPC, DSPC, DOPC, DAPC, POPC, SDPC |
| PE (`PE`) | DLPE, DMPE, DPPE, DSPE, DOPE, DAPE, POPE, SDPE |
| PG (`PGR`) | DLPG, DMPG, DPPG, DSPG, DOPG, DAPG, POPG, SDPG |
| PS (`PS`) | DLPS, DMPS, DPPS, DSPS, DOPS, DAPS, POPS, SDPS |
| sphingomyelin (acyl + `SPM` + `SA`) | PSM, SSM |
| sterol | CHL (cholesterol) |

Tails: L = `LAL` 12:0, M = `MY` 14:0, P = `PA` 16:0, S = `ST` 18:0, O = `OL` 18:1, A = `AR` 20:4,
D = `DHA` 22:6. These are the Lipid21 species of `charmmlipid2amber.csv` (the PL* lipids there need a
linoleoyl residue that Lipid21 does not contain).

Any other combination of Lipid21 residues (e.g. mixed chains, the S-glycerol PG head `PGS`, other
sphingomyelin acyl chains) is generated with

```bash
python3 tools/lipid21_gmx.py itp LOPG LAL PGS OL -o LOPG.itp
```

## Usage

grompp must find `lipid21.ff`: copy or symlink it into the run directory, or put its parent
directory into `GMXLIB`. Tested with GROMACS 2026.3 and AmberTools26.

**Membrane, water and ions** (standalone `lipid21.ff`, no protein force field needed):

```
#include "lipid21.ff/forcefield.itp"
#include "lipid21.ff/POPC.itp"
#include "lipid21.ff/CHL.itp"
#include "lipid21.ff/tip3p.itp"
#include "lipid21.ff/ions_tip3p.itp"

[ system ]
POPC/cholesterol

[ molecules ]
POPC   100
CHL     28
SOL   5000
K       10
CL      10
```

**With a protein** (GROMACS 2026, `amber19sb.ff` or `amber14sb.ff`, as Amber's
`leaprc.protein.ff19SB` + `leaprc.lipid21`): build the protein with `gmx pdb2gmx -ff amber19sb`,
then add the Lipid21 parameters right after the force-field include, before any molecule type:

```
#include "amber19sb.ff/forcefield.itp"
#include "lipid21.ff/lipid21_params.itp"
#include "protein.itp"
#include "lipid21.ff/POPC.itp"
#include "amber19sb.ff/tip3p.itp"
#include "amber19sb.ff/ions_tip3p.itp"
```

Lipid21 atom type names (cA, hL, oP, ...) do not clash with the protein types, and grompp's
`_FF_AMBER_LEAP_ATOM_REORDERING` (defined by these force fields) leaves the Lipid21 impropers unchanged.

**From a membrane PDB** with Lipid21 names (packmol-memgen, the Lipid21 repository, or CHARMM-GUI
after `charmmlipid2amber.py`):

```bash
python3 tools/lipid21_pdb2gmx.py membrane.pdb -o conf.gro -p topol.top
```

```bash
gmx grompp -f mdp/em.mdp -c conf.gro -p topol.top -o em.tpr
```

The tool reorders atoms to topology order, renames water (`O H1 H2` -> `OW HW1 HW2`) and ions, and
writes `.itp` files for non-standard lipids. Add `--ff amber19sb` for the add-on route. Proteins
are not handled; prepare them with `gmx pdb2gmx` and merge the topologies.

`gmx pdb2gmx` cannot be used for the lipids themselves: there is no `.rtp`, on purpose (see
"1-4 interactions" below).

### Run parameters

`mdp/md_npt.mdp` follows the Lipid21 Amber protocol (`lipid21/prod.in`): 1.0 nm cut-offs
(plain LJ truncation; GROMACS' potential shift changes only the reported energy), PME, constrained
X–H bonds with a 2 fs step, Langevin dynamics with γ = 1 ps⁻¹ (`integrator = sd`), and tensionless
semi-isotropic pressure coupling at 1 bar. **No long-range dispersion correction is used for bilayers**
(`DispCorr = no`). pmemd's default (`vdwmeth = 1`) does add one, so the Amber `prod.in` runs differ
from this in that respect. Set `ref-t` above the main phase transition of your lipids.

## Conversion details

- **Units.** σ = 2·(R*/2)/2^(1/6)/10 nm, ε = 4.184·ε kJ/mol; bonds k = 2·K·418.4 kJ mol⁻¹ nm⁻²,
  angles k = 2·K·4.184 kJ mol⁻¹ rad⁻²; torsions k = 4.184·PK/IDIVF kJ/mol as Fourier series
  (function 9, all terms kept, including zero-amplitude ones); impropers function 4. Reversed
  duplicate entries in `lipid21.dat` (identical values) are written once.
- **1-4 interactions.** SCEE = 1.2 everywhere (`fudgeQQ = 1/1.2`). SCNB = 2.0 (`fudgeLJ = 0.5`)
  everywhere **except the cD-cD-cD-cD torsion, which has SCNB = 6.0** in `lipid21.dat`, and tleap
  applies it. The carbon pairs 1-4 across that torsion in the acyl chains and the cholesterol side
  chain (e.g. 21 pairs in POPC) therefore have explicit `[ pairs ]` parameters σ_ij, ε_ij/6.
  A single global `fudgeLJ` cannot represent this, which is why pdb2gmx (`.rtp`) is not supported.
- **Impropers** are applied where tleap applies them, in tleap's atom order (central atom third,
  the other three sorted by atom type, ties by atom index), so energies match exactly.
- **Missing parameters.** tleap reports no missing torsion parameters for any combination of
  Lipid21 residues; the converter would drop such a torsion and its 1-4 pair exactly as tleap does.
- **Charges** are those of `lipid21.lib`, with one exception. The sphingomyelin residues (`SA`
  −0.069718 and `SPM` +0.069721) leave each SM with +0.000003 e. From 34 SM molecules on (net
  charge > 10⁻⁴ e) grompp stops with a net-charge warning for Ewald, so the excess is removed from SPM P31
  (1.324762 → 1.324759), noted in the `.itp` header. Use `--keep-lib-charges` for the original values.
- **Equilibrium angles and torsion phases** keep the values of `lipid21.dat`. tleap converts
  degrees with π ≈ 3.141594, i.e. its values are larger by 4.3×10⁻⁷ (relative). This is negligible
  and below the numerical noise of the comparisons.
- **Coulomb constant.** GROMACS uses 138.935458 kJ mol⁻¹ nm e⁻², Amber 332.0522173 kcal mol⁻¹ Å e⁻²,
  i.e. GROMACS electrostatic energies are 3.5×10⁻⁵ larger for identical charges. This is inherent to
  GROMACS and was not compensated.
- **Water and ions** (standalone files) come from the files `leaprc.water.tip3p` loads: TIP3P with
  Amber's rigid geometry (O–H 0.9572 Å, H–H 1.5136 Å; GROMACS' own TIP3P files use H–H 1.5139 Å),
  Joung–Cheatham Na+, K+, Cl- and Li/Merz 12-6 Mg2+, Ca2+. Atom types and values are identical to
  `amber19sb.ff`/`amber14sb.ff` in GROMACS 2026. The Lipid21 reference bilayers in `lipid21/*/build.leap`
  instead load `frcmod.ionsff99_tip3p` (Åqvist K+).

## Validation

Details and all numbers: [validation/REPORT.md](validation/REPORT.md).

1. **Topology vs. tleap**: 302 molecules (all 294 tail–head–tail combinations, 7 sphingomyelins,
   cholesterol) are identical term by term to the tleap prmtops: 37,001 atoms, 36,703 bonds,
   68,879 angles, 150,013 torsion terms, 2,537 impropers in identical atom order, and 94,332 1-4
   pairs with their scale factors (4,750 with SCNB = 6). Charges, masses, LJ and exclusions match too.
2. **What grompp makes of it**: for the same 302 molecules the interactions and parameters in the
   `.tpr` (gmx dump) agree with the prmtops, for the standalone and for the amber19sb add-on route.
3. **Single molecules, vacuum, no cut-off, 3,322 off-equilibrium conformations** (450 K MD):
   every energy term agrees with sander to ≤ 0.014 kJ/mol (terms of a few hundred kJ/mol), and
   forces to 0.15 kJ/mol/nm RMS (|F| ≈ 1,700). Both are below GROMACS' own mixed-precision noise,
   measured by evaluating rigidly translated copies of the same frames.
4. **The 20 Lipid21 reference bilayers** (PME, identical discretisation in both programs, no
   dispersion correction): bonded, 1-4 and LJ energies agree to ≤ 4×10⁻⁵ relative and forces to
   0.07–0.10 kJ/mol/nm RMS on lipid atoms (~1×10⁻⁴ relative).
5. **PME energies** differ by 20–70 kJ/mol (~1.5×10⁻⁴ relative) in these large systems. This is a
   property of mixed-precision GROMACS, not of the topology. It depends on the Ewald splitting,
   while the double-precision Amber result does not, and it is also seen against an exact Ewald sum
   for small water boxes (`tools/pme_precision_test.py`). Forces are not affected.
6. **An independent mixed membrane** (POPC/POPE/POPS/CHL/PSM, K+/Cl-) built with packmol-memgen
   and converted with `tools/lipid21_pdb2gmx.py` agrees with its own tleap topology to the same level.
7. **MD with `mdp/md_npt.mdp`** (no dispersion correction): 5 ns of the 128-POPC reference bilayer
   started from the Lipid21 300 ns structure. The area per lipid goes from 63.3 Å² (instantaneous,
   Amber end point) to block averages of 64.6–65.4 Å², with no drift after the first nanosecond;
   density 1002 kg/m³, no constraint warnings. The converted mixed membrane also ran cleanly.

## Reproducing

Python ≥ 3.8 only for the converter and the PDB tool. The validation needs AmberTools (tleap,
sander, pmemd, pysander and numpy via `$AMBERHOME/miniconda/bin/python`) and GROMACS 2026.

```bash
python3 tools/lipid21_gmx.py ff -o lipid21.ff
```

```bash
source $AMBERHOME/amber.sh
python3 tools/compare_prmtop.py --build /tmp/lipid21_tleap
$AMBERHOME/miniconda/bin/python tools/validate_molecules.py -o validation/molecules
$AMBERHOME/miniconda/bin/python tools/validate_bilayers.py -o validation/bilayers
$AMBERHOME/miniconda/bin/python tools/pme_precision_test.py -o /tmp/pme_test
python3 tools/summarize_validation.py
```

## Citation

If you use this port, cite Lipid21 (Dickson, Walker & Gould, JCTC 2022) and the work it builds on:
Lipid14 (Dickson et al., JCTC 10, 865 (2014)) and the cholesterol parameters (Madej, Gould &
Walker, J. Phys. Chem. B 119, 12424 (2015)). For TIP3P and ions see the headers of `tip3p.itp`
and `ions_tip3p.itp`.
