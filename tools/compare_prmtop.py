#!/usr/bin/env python3
"""Term-by-term comparison of tools/lipid21_gmx.py topologies with tleap prmtops.

usage: compare_prmtop.py PRMTOP RES1 [RES2 ...]      (one molecule)
       compare_prmtop.py --dir DIR                   (all DIR/<R1>_<R2>_<R3>.prmtop)
       compare_prmtop.py --build DIR                 (run tleap for all 302 residue
                                                      combinations into DIR, then compare)

Everything is compared in Amber units against the numbers stored in the prmtop:
atom names/types/residues, charges, masses, LJ A/B coefficients, bonds,
angles, every proper torsion term, impropers (including atom order), 1-4 pairs
with their SCEE/SCNB factors, and the exclusion list.
"""
import glob
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lipid21_gmx as L  # noqa: E402
from prmtop import Prmtop  # noqa: E402

TOL_Q = 6e-9       # prmtop charges carry 9 significant digits (q*18.2223, E16.8)
TOL_REL = 2e-8     # relative tolerance for force constants etc. (E16.8 storage)
LEAP_DEG = 3.141594 / math.pi   # leap's degree->radian factor relative to the exact one


def close(a, b, rel=TOL_REL, abs_=1e-10):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def compare(ff, prm_path, resnames):
    errs = []
    mol = L.Molecule(ff, resnames)
    p = Prmtop(prm_path)
    n = len(mol.atoms)
    if p.natom != n:
        return ['natom %d vs %d' % (n, p.natom)]
    for i, a in enumerate(mol.atoms):
        if (a['name'], a['type'], a['resname']) != (p.names[i], p.types[i], p.resname[i]):
            errs.append('atom %d: %s vs %s' % (i, (a['name'], a['type'], a['resname']),
                                               (p.names[i], p.types[i], p.resname[i])))
        # the only intended charge change: neutralisation of SM (documented in the .itp)
        if abs(a['charge'] - mol.charge_adjust.get(i, 0.0) - p.charges[i]) > TOL_Q:
            errs.append('charge %d %s: %.9f vs %.9f' % (i, a['name'], a['charge'], p.charges[i]))
        if not close(a['mass'], p.masses[i]):
            errs.append('mass %d' % i)
    # LJ: compare A, B for all type pairs present
    types = sorted(set(range(n)), key=lambda i: p.type_index[i])
    rep = {}
    for i in range(n):
        rep.setdefault(p.type_index[i], i)
    for ti, i in rep.items():
        for tj, j in rep.items():
            A, B = p.lj_ab(i, j)
            ri, ei = ff.ps.lj[mol.t(i)]
            rj, ej = ff.ps.lj[mol.t(j)]
            R, eps = ri + rj, math.sqrt(ei * ej)
            if not (close(A, eps * R ** 12, 1e-7) and close(B, 2 * eps * R ** 6, 1e-7)):
                errs.append('LJ %s-%s' % (mol.t(i), mol.t(j)))
    # bonds
    pb = {(min(i, j), max(i, j)): (k, r) for i, j, k, r in p.bonds}
    mb = {(min(i, j), max(i, j)): prm for (i, j), prm in zip(mol.bonds, mol.bond_params)}
    if set(pb) != set(mb):
        errs.append('bond sets differ: %s' % (set(pb) ^ set(mb)))
    for key in set(pb) & set(mb):
        if not (close(pb[key][0], mb[key][0]) and close(pb[key][1], mb[key][1])):
            errs.append('bond %s: %s vs %s' % (key, mb[key], pb[key]))
    # angles
    pa = {(min(i, k), j, max(i, k)): (kk, th) for i, j, k, kk, th in p.angles}
    ma = {(min(i, k), j, max(i, k)): prm for i, j, k, prm in mol.angles}
    if set(pa) != set(ma):
        errs.append('angle sets differ: %s' % (set(pa) ^ set(ma)))
    for key in set(pa) & set(ma):
        # tleap converts degrees to radians with pi ~ 3.141594 (relative error 4.3e-7),
        # so prmtop equilibrium angles are 4.3e-7 (relative) larger than in lipid21.dat
        if not (close(pa[key][0], ma[key][0]) and abs(pa[key][1] / ma[key][1] - LEAP_DEG) < 1e-8):
            errs.append('angle %s: %s vs %s' % (key, ma[key], pa[key]))
    # proper torsions: compare the multiset of terms on each quartet
    pp = defaultdict(list)
    pimp = []
    for (i, j, k, l, pk, ph, pn, imp, ign, scee, scnb) in p.dihedrals:
        if imp:
            pimp.append(((i, j, k, l), (pk, ph, pn)))
            continue
        key = (i, j, k, l) if i < l else (l, k, j, i)
        pp[key].append((round(pk, 6), round(ph % 360.0, 3), int(round(pn))))
    mp = {}
    for key, tkey, terms in mol.propers:
        mp[key] = [(round(t['pk'] / t['idivf'], 6), round(t['phase'] % 360.0, 3), t['pn']) for t in terms]
    if set(pp) != set(mp):
        only_m = set(mp) - set(pp)
        only_p = set(pp) - set(mp)
        # tleap drops torsion terms whose force constants are all zero?  Report explicitly.
        errs.append('torsion quartets differ: only builder %d, only prmtop %d' % (len(only_m), len(only_p)))
        for key in sorted(only_m)[:10]:
            errs.append('   only builder: %s %s %s' % (key, '-'.join(mol.t(x) for x in key), mp[key]))
        for key in sorted(only_p)[:10]:
            errs.append('   only prmtop : %s %s %s' % (key, '-'.join(mol.t(x) for x in key), pp[key]))
    for key in set(pp) & set(mp):
        a, b = sorted(pp[key]), sorted(mp[key])
        if len(a) != len(b) or any(not (abs(x[0] - y[0]) < 1e-6 and abs(x[1] - y[1]) < 1e-3 and x[2] == y[2])
                                   for x, y in zip(a, b)):
            errs.append('torsion %s %s: builder %s vs prmtop %s' % (key, '-'.join(mol.t(x) for x in key), b, a))
    # impropers: exact atom order.  A prmtop cannot hold atom 0 in the 3rd/4th
    # position of a dihedral (the sign of those indices carries flags), so tleap
    # writes such dihedrals reversed; l-k-j-i is the same angle as i-j-k-l.
    pimp = [((q[3], q[2], q[1], q[0]), prm) if 0 in (q[0], q[1]) and q[3] != 0 and q[2] != 0
            and (q[3], q[2], q[1], q[0]) in {m[0] for m in mol.impropers} else (q, prm) for q, prm in pimp]
    mimp = [(quad, (pk, ph, pn)) for quad, ikey, (pk, ph, pn) in mol.impropers]
    if sorted(q for q, _ in pimp) != sorted(q for q, _ in mimp):
        errs.append('improper quartets/order differ:\n   builder %s\n   prmtop  %s'
                    % (sorted(q for q, _ in mimp), sorted(q for q, _ in pimp)))
    else:
        pd_ = dict(pimp)
        for q, (pk, ph, pn) in mimp:
            a = pd_[q]
            if not (close(a[0], pk) and abs(a[1] % 360 - ph % 360) < 1e-3 and int(round(a[2])) == pn):
                errs.append('improper %s params %s vs %s' % (q, (pk, ph, pn), a))
    # 1-4 pairs
    p14 = p.pairs14()
    m14 = {k: (v[0], v[1]) for k, v in mol.pairs.items()}
    if set(p14) != set(m14):
        errs.append('1-4 pair sets differ: %s' % (set(p14) ^ set(m14)))
    for key in set(p14) & set(m14):
        if not (close(p14[key][0], m14[key][0]) and close(p14[key][1], m14[key][1])):
            errs.append('1-4 scaling %s: %s vs %s' % (key, m14[key], p14[key]))
    # exclusions (prmtop: 1-2, 1-3, 1-4 neighbours)
    mex = set(mol.dist)
    if p.exclusions != mex:
        errs.append('exclusions differ: %s' % (p.exclusions ^ mex))
    stats = dict(atoms=n, bonds=len(mb), angles=len(ma), torsions=len(mp),
                 torsion_terms=sum(len(v) for v in mp.values()), impropers=len(mimp),
                 pairs=len(m14), pairs_scnb6=sum(1 for v in m14.values() if v[1] != 2.0),
                 charge_adjusted_atoms=len(mol.charge_adjust))
    return errs, stats


TAILS = ['LAL', 'MY', 'PA', 'ST', 'OL', 'AR', 'DHA']
HEADS = ['PC', 'PE', 'PS', 'PGR', 'PGS', 'PH-']


def build_all(outdir):
    """tleap: every TAIL-HEAD-TAIL combination, every ACYL-SPM-SA and CHL."""
    import subprocess
    os.makedirs(outdir, exist_ok=True)
    seqs = [[t1, h, t2] for t1 in TAILS for h in HEADS for t2 in TAILS] + [[t, 'SPM', 'SA'] for t in TAILS]
    cmds = ['source leaprc.lipid21']
    for s in seqs:
        cmds += ['m = sequence { %s }' % ' '.join(s), 'saveamberparm m %s.prmtop %s.inpcrd' % ('_'.join(s), '_'.join(s))]
    cmds += ['m = copy CHL', 'saveamberparm m CHL.prmtop CHL.inpcrd', 'quit']
    open(os.path.join(outdir, 'all.leap'), 'w').write('\n'.join(cmds) + '\n')
    r = subprocess.run(['tleap', '-f', 'all.leap'], cwd=outdir, capture_output=True, text=True)
    bad = [l for l in r.stdout.splitlines() if 'No torsion' in l or 'Could not' in l]
    print('tleap: %d prmtops, %d missing-parameter messages' % (len(seqs) + 1, len(bad)))


def main():
    ff = L.ForceField(os.path.join(L.DEFAULT_SRC, 'lipid21.lib'), os.path.join(L.DEFAULT_SRC, 'lipid21.dat'))
    if sys.argv[1] == '--build':
        build_all(sys.argv[2])
        sys.argv[1] = '--dir'
    if sys.argv[1] == '--dir':
        files = sorted(glob.glob(os.path.join(sys.argv[2], '*.prmtop')))
        bad = 0
        tot = defaultdict(int)
        for f in files:
            res = os.path.basename(f)[:-7].split('_')
            errs, st = compare(ff, f, res)
            for k, v in st.items():
                tot[k] += v
            if errs:
                bad += 1
                print('FAIL', os.path.basename(f))
                for e in errs[:20]:
                    print('   ', e)
        print('%d molecules compared, %d with differences' % (len(files), bad))
        print('totals:', dict(tot))
        return 1 if bad else 0
    errs, st = compare(ff, sys.argv[1], sys.argv[2:])
    print('\n'.join(errs) if errs else 'IDENTICAL', st)
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
