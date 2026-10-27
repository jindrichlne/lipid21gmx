#!/usr/bin/env python
"""Compare a GROMACS topology of a Lipid21 system with its Amber prmtop (PME).

    validate_system.py --prmtop sys.prmtop --inpcrd sys.inpcrd --top topol.top [-o workdir]

The GROMACS topology (e.g. written by lipid21_pdb2gmx.py) must list the atoms in
prmtop order.  The Amber coordinates are used for both programs.  Energies
(pmemd) and forces (sander via pysander) are compared using the same smooth-PME
discretisation (Amber defaults: cut 10 A, dsum_tol 1e-5 <-> ewald-rtol 1e-4,
order 4, identical grid), no potential modifiers and no dispersion correction.
For the single point, TIP3P is given the three bonds it has in a prmtop
(validation only).  Run with the AmberTools Python.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gmxio  # noqa: E402
from prmtop import Prmtop  # noqa: E402
from validate_molecules import parse_forces  # noqa: E402
from validate_bilayers import (AMBER_IN, MDP, TIP3P_AMBER_BONDS, KCAL, K_AMBER, K_GMX,  # noqa: E402
                               fft_size, parse_amber_out, read_rst7_box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prmtop', required=True)
    ap.add_argument('--inpcrd', required=True)
    ap.add_argument('--top', required=True)
    ap.add_argument('-o', '--workdir', default='validate_system')
    args = ap.parse_args()
    wd = os.path.abspath(args.workdir)
    os.makedirs(wd, exist_ok=True)
    prm = Prmtop(args.prmtop)
    n = prm.natom
    x, box = read_rst7_box(args.inpcrd, n)
    nfft = [fft_size(int(math.ceil(b / 1.0))) for b in box]
    res = {'natom': n, 'box_A': box, 'nfft': nfft}

    # Amber energies (pmemd)
    shutil.copy(args.prmtop, os.path.join(wd, 'sys.prmtop'))
    shutil.copy(args.inpcrd, os.path.join(wd, 'sys.rst7'))
    open(os.path.join(wd, 'sp.in'), 'w').write(AMBER_IN % tuple(nfft))
    subprocess.run(['pmemd', '-O', '-i', 'sp.in', '-p', 'sys.prmtop', '-c', 'sys.rst7', '-o', 'sp.out',
                    '-r', 'sp.r', '-inf', 'sp.info'], cwd=wd, check=True, capture_output=True)
    a = parse_amber_out(os.path.join(wd, 'sp.out'))
    # Amber forces (sander)
    import sander
    inp = sander.pme_input()
    inp.cut, inp.ntc, inp.ntf = 10.0, 1, 1
    with sander.setup(os.path.join(wd, 'sys.prmtop'), x, list(box) + [90.0, 90.0, 90.0], inp):
        e, f = sander.energy_forces()
    famb = np.array(f).reshape(n, 3) * KCAL * 10.0

    # GROMACS: same topology, TIP3P with prmtop bonds, Amber coordinates
    topdir = os.path.dirname(os.path.abspath(args.top))
    top = open(args.top).read()
    top = re.sub(r'#include "([^"]*/)?tip3p\.itp"', lambda m: TIP3P_AMBER_BONDS, top)
    top = re.sub(r'#include "([^"/]+\.itp)"', lambda m: '#include "%s"' % os.path.join(topdir, m.group(1)), top)
    if not os.path.exists(os.path.join(wd, 'lipid21.ff')):
        os.symlink(os.path.join(os.path.dirname(HERE), 'lipid21.ff'), os.path.join(wd, 'lipid21.ff'))
    open(os.path.join(wd, 'topol.top'), 'w').write(top)
    # atom names as in the GROMACS topology (water and ions renamed)
    atoms = []
    for i in range(n):
        an, rn = prm.names[i], prm.resname[i]
        if rn == 'WAT':
            an = {'O': 'OW', 'H1': 'HW1', 'H2': 'HW2'}[an]
        elif rn in ('Na+', 'K+', 'Cl-', 'MG', 'CA'):
            an = {'Na+': 'NA', 'K+': 'K', 'Cl-': 'CL'}.get(rn, rn)
        atoms.append((prm.resnum[i] + 1, rn, an))
    gmxio.write_g96(os.path.join(wd, 'conf.g96'), atoms, [x / 10.0], [b / 10.0 for b in box])
    open(os.path.join(wd, 'sp.mdp'), 'w').write(MDP % tuple(nfft))
    log = gmxio.gmx(['grompp', '-f', 'sp.mdp', '-c', 'conf.g96', '-p', 'topol.top', '-o', 'sp.tpr', '-maxwarn', '0'], wd)
    open(os.path.join(wd, 'grompp.log'), 'w').write(log)
    gmxio.gmx(['mdrun', '-s', 'sp.tpr', '-rerun', 'conf.g96', '-deffnm', 'sp', '-ntmpi', '1'], wd)
    terms = ['Bond', 'Angle', 'Proper-Dih.', 'Per.-Imp.-Dih.', 'LJ-14', 'Coulomb-14', 'LJ-(SR)',
             'Coulomb-(SR)', 'Coul.-recip.', 'Potential']
    gmxio.gmx(['energy', '-f', 'sp.edr', '-o', 'sp.xvg'], wd, stdin='\n'.join(terms) + '\n\n')
    leg, rows = gmxio.read_xvg(os.path.join(wd, 'sp.xvg'))
    g = dict(zip(leg, rows[0][1:]))
    fg = parse_forces(gmxio.gmx(['dump', '-f', 'sp.trr'], wd), n)[0]

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
    groups = {'all': np.ones(n, bool), 'water': np.array([r == 'WAT' for r in prm.resname]),
              'ions': np.array([r in ('Na+', 'K+', 'Cl-', 'MG', 'CA') for r in prm.resname])}
    groups['lipids'] = ~groups['water'] & ~groups['ions']
    res['forces'] = {k: {'rms_F': float(np.sqrt((fn[m] ** 2).mean())), 'rms_dF': float(np.sqrt((d[m] ** 2).mean())),
                         'max_dF': float(d[m].max())} for k, m in groups.items() if m.any()}
    json.dump(res, open(os.path.join(wd, 'results.json'), 'w'), indent=1)
    for k, v in res['compare'].items():
        print('%-15s GROMACS %14.3f  Amber %14.3f  diff %+9.3f kJ/mol' % (k, v['gmx'], v['amber'], v['diff']))
    for k, v in res['forces'].items():
        print('forces %-7s RMS|F| %8.1f  RMS|dF| %.4f  max|dF| %.3f kJ/mol/nm' % (k, v['rms_F'], v['rms_dF'], v['max_dF']))


if __name__ == '__main__':
    main()
