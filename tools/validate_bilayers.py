#!/usr/bin/env python
"""Validate the GROMACS Lipid21 port on the Lipid21 reference bilayers (PME).

For every bilayer of the Lipid21 repository (128 lipids, TIP3P, K+ where
needed; relaxed 300 ns structure in start.rst):
  * the Amber topology is built with the repository's build.leap
    (ff19SB + TIP3P + Lipid21 + frcmod.ionsff99_tip3p),
  * the GROMACS topology is assembled from the lipid21.ff lipid .itp files,
    plus the Aqvist K+ of frcmod.ionsff99_tip3p and, for these single points
    only, a TIP3P with the three bonds of an Amber prmtop (O-H, O-H, H-H; no
    angle) so that Amber's ntf=1 water bond terms have a GROMACS counterpart,
  * energies (pmemd) and forces (sander via pysander) are compared with a
    GROMACS rerun using the *same* smooth-PME discretisation as Amber's
    defaults (cut 10 A, dsum_tol 1e-5 -> ewald-rtol 1e-4, order 4, identical
    FFT grid), no potential modifiers and no dispersion correction
    (pmemd vdwmeth=0, GROMACS DispCorr=no).

Run with the AmberTools Python:
    $AMBERHOME/miniconda/bin/python tools/validate_bilayers.py -o validation/bilayers
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gmxio  # noqa: E402
import lipid21_gmx as L  # noqa: E402
from prmtop import Prmtop  # noqa: E402
from validate_molecules import parse_forces  # noqa: E402

REPO = os.path.join(ROOT, 'lipid21')
FFDIR = os.path.join(ROOT, 'lipid21.ff')
KCAL = 4.184
K_AMBER = 18.2223 ** 2 * KCAL / 10.0
K_GMX = 138.935457833
SYSTEMS = ['DAPC', 'DLPC', 'DLPG', 'DMPC', 'DMPG', 'DOPC', 'DOPG', 'DOPS', 'DPPC', 'DPPG', 'DSPC', 'DSPG',
           'POPA', 'POPC', 'POPE', 'POPG', 'POPS', 'PSM', 'SDPC', 'SSM']

AMBER_IN = """single point
&cntrl
 imin=1, maxcyc=1, ncyc=1, ntb=1, cut=10.0, ntc=1, ntf=1, ntpr=1, ntxo=1,
/
&ewald
 dsum_tol=1.0e-5, order=4, nfft1=%d, nfft2=%d, nfft3=%d, vdwmeth=0,
/
"""

MDP = """integrator = md
nsteps = 0
continuation = yes
constraints = none
cutoff-scheme = Verlet
pbc = xyz
verlet-buffer-tolerance = -1
rlist = 1.0
coulombtype = PME
coulomb-modifier = None
rcoulomb = 1.0
ewald-rtol = 1e-4          ; = Amber dsum_tol 1e-5 at cut 10 A: erfc(beta*rc) = 1e-4
pme-order = 4
fourier-nx = %d
fourier-ny = %d
fourier-nz = %d
vdwtype = Cut-off
vdw-modifier = None
rvdw = 1.0
DispCorr = no
nstcalcenergy = 1
nstenergy = 1
nstfout = 1
"""

# Aqvist K+ for TIP3P (frcmod.ionsff99_tip3p), as loaded by the reference build.leap
K_FF99 = """
; K+ of frcmod.ionsff99_tip3p (Aqvist, adapted), as used for the Lipid21 reference bilayers
[ atomtypes ]
  K+_ff99   19   39.10   0.0000   A   %.12f   %.12f

[ moleculetype ]
K+        1

[ atoms ]
    1  K+_ff99   1   K+   K+   1   1.000000   39.10
""" % (2 * 2.658 * 2 ** (-1 / 6) / 10, 0.000328 * KCAL)

# TIP3P with the bonded terms of an Amber prmtop (validation of single points only)
TIP3P_AMBER_BONDS = """
; TIP3P as in an Amber prmtop: 3 bonds (incl. H-H), no angle -- single-point validation only
[ moleculetype ]
SOL        2

[ atoms ]
    1  OW_tip3p   1  SOL  OW   1  -0.834  16.0
    2  HW_tip3p   1  SOL  HW1  1   0.417  1.008
    3  HW_tip3p   1  SOL  HW2  1   0.417  1.008

[ bonds ]
  1  2  1  0.09572  462750.4
  1  3  1  0.09572  462750.4
  2  3  1  0.15136  462750.4
"""


def fft_size(n):
    """Smallest number >= n with prime factors 2, 3, 5 only."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('%s failed:\n%s\n%s' % (cmd, r.stdout[-2000:], r.stderr[-2000:]))
    return r.stdout + r.stderr


def read_rst7_box(path, natom):
    lines = open(path).read().splitlines()
    vals = []
    for l in lines[2:]:
        vals += [float(l[i:i + 12]) for i in range(0, len(l), 12) if l[i:i + 12].strip()]
    x = np.array(vals[:3 * natom]).reshape(natom, 3)
    box = vals[-6:-3]
    return x, box


def parse_amber_out(path):
    txt = open(path).read()
    blk = txt[txt.index('NSTEP'):]
    out = {}
    for key, name in (('BOND', 'bond'), ('ANGLE', 'angle'), ('DIHED', 'dihed'), ('VDWAALS', 'vdw'),
                      ('EEL', 'eel'), ('1-4 VDW', 'vdw14'), ('1-4 NB', 'vdw14'), ('1-4 EEL', 'eel14')):
        m = re.search(r'(?<![-\w])%s\s*=\s*([-0-9.]+)' % re.escape(key), blk)
        if m:
            out[name] = float(m.group(1))
    m = re.search(r'NSTEP\s+ENERGY.*\n\s+\d+\s+([-0-9.E+]+)', blk)
    out['total'] = float(m.group(1))
    m = re.search(r'Ewald Coefficient =\s*([0-9.]+)', txt)
    out['ewald_coefficient'] = float(m.group(1))
    return out


def molecules(prm):
    """Connected components -> list of molecule names in order."""
    n = prm.natom
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j, k, r in prm.bonds:
        parent[find(i)] = find(j)
    std = {tuple(v): k for k, v in L.standard_lipids().items()}
    out = []
    start = 0
    for a in range(1, n + 1):
        if a == n or find(a) != find(start):
            names = []
            for b in range(start, a):
                if not names or prm.resnum[b] != names[-1][0]:
                    names.append((prm.resnum[b], prm.resname[b]))
            names = tuple(r for _, r in names)
            if names == ('WAT',):
                out.append('SOL')
            elif names == ('K+',):
                out.append('K+')
            elif names in std:
                out.append(std[names])
            else:
                raise ValueError('unknown molecule %s' % (names,))
            start = a
    return out


def validate(system, outroot):
    wd = os.path.join(outroot, 'work', system)
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    res = {'system': system}
    try:
        src = os.path.join(REPO, system)
        leap = open(os.path.join(src, 'build.leap')).read()
        leap = leap.replace('input_%s_128.pdb' % system, os.path.join(src, 'input_%s_128.pdb' % system))
        leap = leap.replace('%s_128.prmtop %s_128.inpcrd' % (system, system), 'sys.prmtop sys.inpcrd')
        open(os.path.join(wd, 'build.leap'), 'w').write(leap)
        out = run(['tleap', '-f', 'build.leap'], wd)
        res['tleap_warnings'] = sorted(set(l.strip() for l in out.splitlines()
                                           if 'No torsion' in l or 'Could not' in l or 'FATAL' in l.upper()))
        prm = Prmtop(os.path.join(wd, 'sys.prmtop'))
        x, box = read_rst7_box(os.path.join(src, 'start.rst'), prm.natom)
        n = prm.natom
        res['natom'], res['box_A'] = n, box
        nfft = [fft_size(int(math.ceil(b / 1.0))) for b in box]   # Amber default spacing ~1 A
        res['nfft'] = nfft
        # ---------------- Amber energies (pmemd) ----------------
        shutil.copy(os.path.join(src, 'start.rst'), os.path.join(wd, 'start.rst'))
        open(os.path.join(wd, 'sp.in'), 'w').write(AMBER_IN % tuple(nfft))
        run(['pmemd', '-O', '-i', 'sp.in', '-p', 'sys.prmtop', '-c', 'start.rst', '-o', 'sp.out', '-r', 'sp.rst7',
             '-inf', 'sp.mdinfo'], wd)
        amber = parse_amber_out(os.path.join(wd, 'sp.out'))
        res['amber'] = amber
        # ---------------- Amber forces (sander via pysander, same PME defaults) ----------------
        import sander
        inp = sander.pme_input()
        inp.cut = 10.0
        inp.ntc = 1
        inp.ntf = 1
        with sander.setup(os.path.join(wd, 'sys.prmtop'), x, list(box) + [90.0, 90.0, 90.0], inp):
            e, f = sander.energy_forces()
        famb = np.array(f).reshape(n, 3) * KCAL * 10.0
        res['sander_eel_kcal'] = e.elec
        # ---------------- GROMACS ----------------
        mols = molecules(prm)
        shutil.copytree(FFDIR, os.path.join(wd, 'lipid21.ff'))
        counts = []
        for name in mols:
            if counts and counts[-1][0] == name:
                counts[-1][1] += 1
            else:
                counts.append([name, 1])
        lipids = sorted(set(m for m in mols if m not in ('SOL', 'K+')))
        top = ['#include "lipid21.ff/forcefield.itp"', K_FF99]
        top += ['#include "lipid21.ff/%s.itp"' % l for l in lipids]
        top += [TIP3P_AMBER_BONDS, '[ system ]', system, '', '[ molecules ]']
        top += ['%-6s %d' % (m, c) for m, c in counts]
        open(os.path.join(wd, 'topol.top'), 'w').write('\n'.join(top) + '\n')
        res['gmx_molecules'] = counts
        atoms = []
        for i in range(n):
            an = prm.names[i]
            if prm.resname[i] == 'WAT':
                an = {'O': 'OW', 'H1': 'HW1', 'H2': 'HW2'}[an]
            atoms.append((prm.resnum[i] + 1, prm.resname[i], an))
        gmxio.write_g96(os.path.join(wd, 'conf.g96'), atoms, [x / 10.0], [b / 10.0 for b in box])
        open(os.path.join(wd, 'sp.mdp'), 'w').write(MDP % tuple(nfft))
        log = gmxio.gmx(['grompp', '-f', 'sp.mdp', '-c', 'conf.g96', '-p', 'topol.top', '-o', 'sp.tpr', '-maxwarn', '0'], wd)
        open(os.path.join(wd, 'grompp.log'), 'w').write(log)
        gmxio.gmx(['mdrun', '-s', 'sp.tpr', '-rerun', 'conf.g96', '-deffnm', 'sp', '-ntmpi', '1'], wd)
        mdlog = open(os.path.join(wd, 'sp.log')).read()
        res['gmx_beta_per_A'] = 0.1 / float(re.search(r'width \(1/beta\) of ([0-9.]+)', mdlog).group(1))
        terms = ['Bond', 'Angle', 'Proper-Dih.', 'Per.-Imp.-Dih.', 'LJ-14', 'Coulomb-14', 'LJ-(SR)',
                 'Coulomb-(SR)', 'Coul.-recip.', 'Potential']
        gmxio.gmx(['energy', '-f', 'sp.edr', '-o', 'sp.xvg'], wd, stdin='\n'.join(terms) + '\n\n')
        leg, rows = gmxio.read_xvg(os.path.join(wd, 'sp.xvg'))
        g = dict(zip(leg, rows[0][1:]))
        res['gmx'] = g
        fg = parse_forces(gmxio.gmx(['dump', '-f', 'sp.trr'], wd), n)[0]
        # ---------------- comparison (kJ/mol) ----------------
        a = amber
        kq = K_GMX / K_AMBER
        cmp = {
            'Bond': (g['Bond'], a['bond'] * KCAL),
            'Angle': (g['Angle'], a['angle'] * KCAL),
            'Dihedral': (g['Proper Dih.'] + g['Per. Imp. Dih.'], a['dihed'] * KCAL),
            'LJ-14': (g['LJ-14'], a['vdw14'] * KCAL),
            'Coulomb-14': (g['Coulomb-14'], a['eel14'] * KCAL * kq),
            'LJ (r<1 nm)': (g['LJ (SR)'], a['vdw'] * KCAL),
            'Coulomb (PME)': (g['Coulomb (SR)'] + g['Coul. recip.'], a['eel'] * KCAL * kq),
            'Potential': (g['Potential'], (a['total'] + (kq - 1) * (a['eel'] + a['eel14'])) * KCAL),
        }
        res['compare'] = {k: {'gmx': gv, 'amber': av, 'diff': gv - av} for k, (gv, av) in cmp.items()}
        d = np.linalg.norm(fg - famb, axis=1)
        fn = np.linalg.norm(famb, axis=1)
        wat = np.array([r == 'WAT' for r in prm.resname])
        ion = np.array([r == 'K+' for r in prm.resname])
        lip = ~wat & ~ion
        res['forces'] = {}
        for lab, mask in (('all', np.ones(n, bool)), ('lipid', lip), ('water', wat), ('K+', ion)):
            if mask.any():
                res['forces'][lab] = {'rms_F': float(np.sqrt((fn[mask] ** 2).mean())),
                                      'rms_dF': float(np.sqrt((d[mask] ** 2).mean())),
                                      'max_dF': float(d[mask].max())}
        res['ok'] = True
    except Exception:
        res['ok'] = False
        res['error'] = traceback.format_exc()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--outdir', default=os.path.join(ROOT, 'validation', 'bilayers'))
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('-j', '--jobs', type=int, default=6)
    args = ap.parse_args()
    systems = args.only or SYSTEMS
    os.makedirs(args.outdir, exist_ok=True)
    results = []
    # one fresh process per system: pysander can be set up only once per process
    import multiprocessing as mp
    with mp.Pool(args.jobs, maxtasksperchild=1) as pool:
        computed = pool.starmap(validate, [(s, args.outdir) for s in systems], chunksize=1)
    for s, r in zip(systems, computed):
        results.append(r)
        if r['ok']:
            print('%-5s natom %6d |' % (s, r['natom']), '  '.join('%s %+.3f' % (k, v['diff']) for k, v in r['compare'].items()),
                  '| F rms dF %.3f (lipid %.3f) max %.3f kJ/mol/nm' % (r['forces']['all']['rms_dF'], r['forces']['lipid']['rms_dF'],
                                                                   r['forces']['all']['max_dF']), flush=True)
            if not args.keep:
                shutil.rmtree(os.path.join(args.outdir, 'work', s), ignore_errors=True)
        else:
            print(s, 'FAILED', r['error'][-2000:], flush=True)
        json.dump(results, open(os.path.join(args.outdir, 'results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
