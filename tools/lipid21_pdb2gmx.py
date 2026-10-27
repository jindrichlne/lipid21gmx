#!/usr/bin/env python3
"""Convert an Amber/Lipid21-named membrane PDB into GROMACS input (.gro + .top).

The input is a PDB with Lipid21 residue and atom names, as written by
packmol-memgen, by charmmlipid2amber.py (CHARMM-GUI) or found in the Lipid21
repository: every lipid is a run of residues TAIL HEAD TAIL (e.g. PA PC OL),
ACYL SPM SA for sphingomyelin, or CHL; water is WAT/HOH/TIP3/SOL, ions are
Na+/K+/Cl-/Mg2+/Ca2+ (several common spellings accepted).  Hydrogens must be
present and named as in lipid21.lib; atoms are reordered to topology order.

The topology uses the shipped lipid21.ff/<NAME>.itp for the standard lipids
and writes an .itp next to the output for any other residue combination.

    lipid21_pdb2gmx.py membrane.pdb -o conf.gro -p topol.top [--ff standalone|amber19sb|amber14sb]

Water: TIP3P (rigid).  Ions: Joung-Cheatham (Na+, K+, Cl-) and Li/Merz 12-6
(Mg2+, Ca2+) for TIP3P, as loaded by Amber's leaprc.water.tip3p.
Pure standard-library Python.
"""
import argparse
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lipid21_gmx as L  # noqa: E402

WATER_NAMES = {'WAT', 'HOH', 'TIP3', 'TP3', 'SOL', 'T3P'}
WATER_ATOMS = {'O': 'OW', 'OW': 'OW', 'OH2': 'OW', 'H1': 'HW1', 'HW1': 'HW1', 'H2': 'HW2', 'HW2': 'HW2'}
ION_NAMES = {  # PDB residue name -> GROMACS molecule (and atom) name in ions_tip3p.itp
    'NA+': 'NA', 'NA': 'NA', 'SOD': 'NA', 'K+': 'K', 'K': 'K', 'POT': 'K',
    'CL-': 'CL', 'CL': 'CL', 'CLA': 'CL', 'MG': 'MG', 'MG2+': 'MG', 'CA': 'CA', 'CA2+': 'CA', 'CAL': 'CA'}
HEADS = {'PC', 'PE', 'PS', 'PGR', 'PGS', 'PH-'}


def read_pdb(path):
    residues = []     # list of dict(name, atoms=[(name, x, y, z)])
    box = None
    key = None
    for line in open(path):
        rec = line[:6].strip()
        if rec == 'CRYST1':
            box = [float(line[6:15]), float(line[15:24]), float(line[24:33])]
            angles = [float(line[33:40]), float(line[40:47]), float(line[47:54])]
            if any(abs(a - 90.0) > 1e-3 for a in angles):
                raise SystemExit('only rectangular boxes are supported')
        elif rec == 'TER':
            key = None
        elif rec in ('ATOM', 'HETATM'):
            an = line[12:16].strip()
            rn = line[17:21].strip()
            k = (line[21], line[22:27], rn)
            if k != key:
                residues.append({'name': rn, 'atoms': []})
                key = k
            residues[-1]['atoms'].append((an, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return residues, box


def assemble(residues, ff):
    """Group residues into molecules: list of (kind, resnames, [residue dicts])."""
    tails = {n for n, r in ff.residues.items() if len(r.connect) == 2 and r.connect[0] == r.connect[1] >= 0}
    mols = []
    i = 0
    while i < len(residues):
        rn = residues[i]['name']
        up = rn.upper()
        if up in WATER_NAMES:
            mols.append(('water', ['SOL'], [residues[i]]))
            i += 1
        elif up in ION_NAMES and len(residues[i]['atoms']) == 1:
            mols.append(('ion', [ION_NAMES[up]], [residues[i]]))
            i += 1
        elif rn == 'CHL':
            mols.append(('lipid', ['CHL'], [residues[i]]))
            i += 1
        elif rn in tails and i + 2 < len(residues) and residues[i + 1]['name'] in HEADS | {'SPM'} \
                and residues[i + 2]['name'] in tails:
            names = [residues[i + k]['name'] for k in range(3)]
            if names[1] == 'SPM' and names[2] != 'SA':
                raise SystemExit('sphingomyelin must be ACYL SPM SA, found %s' % names)
            mols.append(('lipid', names, residues[i:i + 3]))
            i += 3
        else:
            raise SystemExit('cannot assign residue %d (%s): not a Lipid21 lipid, water or ion '
                             '(build proteins etc. with gmx pdb2gmx and merge the topologies)' % (i + 1, rn))
    return mols


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdb')
    ap.add_argument('-o', '--gro', default='conf.gro')
    ap.add_argument('-p', '--top', default='topol.top')
    ap.add_argument('--ff', choices=['standalone', 'amber19sb', 'amber14sb'], default='standalone',
                    help='standalone lipid21.ff (default) or Lipid21 on top of a GROMACS Amber protein force field')
    ap.add_argument('--ffdir', default='lipid21.ff', help='path of lipid21.ff as written in #include (default: lipid21.ff)')
    ap.add_argument('--box', nargs=3, type=float, help='box in nm (default: CRYST1 record)')
    args = ap.parse_args(argv)

    ff = L.ForceField(os.path.join(L.DEFAULT_SRC, 'lipid21.lib'), os.path.join(L.DEFAULT_SRC, 'lipid21.dat'))
    residues, box = read_pdb(args.pdb)
    box_nm = args.box or ([b / 10.0 for b in box] if box else None)
    if box_nm is None:
        raise SystemExit('no CRYST1 record; give --box')
    mols = assemble(residues, ff)
    std = {tuple(v): k for k, v in L.standard_lipids().items()}
    shipped_dir = os.path.join(os.path.dirname(HERE), 'lipid21.ff')
    outdir = os.path.dirname(os.path.abspath(args.top))

    itps = OrderedDict()   # molecule name -> include path
    gro = []
    order = []             # molecule names in order
    resnr = 0
    for kind, names, resl in mols:
        if kind == 'water':
            pos = {WATER_ATOMS.get(a[0], a[0]): a for a in resl[0]['atoms']}
            if sorted(pos) != ['HW1', 'HW2', 'OW']:
                raise SystemExit('water residue with atoms %s' % [a[0] for a in resl[0]['atoms']])
            resnr += 1
            for an in ('OW', 'HW1', 'HW2'):
                gro.append((resnr, 'SOL', an, pos[an][1:]))
            order.append('SOL')
            continue
        if kind == 'ion':
            resnr += 1
            gro.append((resnr, names[0], names[0], resl[0]['atoms'][0][1:]))
            order.append(names[0])
            continue
        molname = std.get(tuple(names))
        if molname is None:
            molname = '_'.join(names).replace('-', 'm')
        if molname not in itps:
            if os.path.exists(os.path.join(shipped_dir, molname + '.itp')) and tuple(names) in std:
                itps[molname] = '%s/%s.itp' % (args.ffdir, molname)
            else:
                m = L.Molecule(ff, names, molname)
                path = os.path.join(outdir, molname + '.itp')
                open(path, 'w').write(m.itp(['AMBER Lipid21 -- GROMACS port, generated by tools/lipid21_pdb2gmx.py',
                                             'Lipid21 residues: ' + ' '.join(names)] ))
                itps[molname] = molname + '.itp'
                print('wrote', path, file=sys.stderr)
        for rn, res in zip(names, resl):
            tmpl = ff.residues[rn]
            have = OrderedDict()
            for a in res['atoms']:
                if a[0] in have:
                    raise SystemExit('duplicate atom %s in residue %s' % (a[0], rn))
                have[a[0]] = a
            want = [a['name'] for a in tmpl.atoms]
            missing = [n for n in want if n not in have]
            extra = [n for n in have if n not in want]
            if missing or extra:
                raise SystemExit('residue %s: missing atoms %s, unknown atoms %s' % (rn, missing, extra))
            resnr += 1
            for n in want:
                gro.append((resnr, rn, n, have[n][1:]))
        order.append(molname)

    with open(args.gro, 'w') as out:
        out.write('Lipid21 system from %s\n%d\n' % (os.path.basename(args.pdb), len(gro)))
        for i, (rnr, rn, an, x) in enumerate(gro):
            out.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n' % (rnr % 100000, rn[:5], an[:5], (i + 1) % 100000,
                                                        x[0] / 10, x[1] / 10, x[2] / 10))
        out.write('%10.5f%10.5f%10.5f\n' % tuple(box_nm))

    counts = []
    for m in order:
        if counts and counts[-1][0] == m:
            counts[-1][1] += 1
        else:
            counts.append([m, 1])
    if args.ff == 'standalone':
        head = ['#include "%s/forcefield.itp"' % args.ffdir]
        water = ['#include "%s/tip3p.itp"' % args.ffdir, '#include "%s/ions_tip3p.itp"' % args.ffdir]
    else:
        head = ['#include "%s.ff/forcefield.itp"' % args.ff, '#include "%s/lipid21_params.itp"' % args.ffdir]
        water = ['#include "%s.ff/tip3p.itp"' % args.ff, '#include "%s.ff/ions_tip3p.itp"' % args.ff]
    top = ['; topology written by tools/lipid21_pdb2gmx.py from %s' % os.path.basename(args.pdb)] + head + ['']
    top += ['#include "%s"' % p for p in itps.values()] + water
    top += ['', '[ system ]', 'Lipid21 membrane from %s' % os.path.basename(args.pdb), '', '[ molecules ]']
    top += ['%-10s %6d' % (m, c) for m, c in counts]
    open(args.top, 'w').write('\n'.join(top) + '\n')
    print('wrote %s (%d atoms) and %s: %s' % (args.gro, len(gro), args.top,
                                             ', '.join('%s %d' % (m, c) for m, c in counts)), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
