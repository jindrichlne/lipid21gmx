#!/usr/bin/env python
"""Where does the remaining PME *energy* difference between GROMACS and Amber come from?

Part 1: exact Ewald sum (numpy, double precision, no cut-off approximations)
        for 1 and 8 rigid TIP3P waters in a 2.4 nm box, compared with sander
        and GROMACS at several Ewald splitting parameters.
Part 2: total electrostatic energy of the POPC reference bilayer: Amber at
        several tolerances (all converged) vs GROMACS at ewald-rtol 1e-5,
        1e-6, 1e-7 with converged grids.

Run with the AmberTools Python:  $AMBERHOME/miniconda/bin/python tools/pme_precision_test.py [-o workdir]
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from scipy.special import erf, erfc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gmxio  # noqa: E402
import validate_bilayers as VB  # noqa: E402
from prmtop import Prmtop  # noqa: E402

K_AMBER = 18.2223 ** 2 * 4.184 / 10.0
K_GMX = 138.935457833


def water_boxes(nw, L, seed=7):
    roh, rhh = 0.09572, 0.15136
    half = math.asin(rhh / 2 / roh)
    w = np.array([[0, 0, 0], [roh * math.sin(half), roh * math.cos(half), 0],
                  [-roh * math.sin(half), roh * math.cos(half), 0]])
    rng = np.random.default_rng(seed)
    X = []
    for k in range(nw):
        q_, r_ = np.linalg.qr(rng.normal(size=(3, 3)))
        c = rng.uniform(0.3, L - 0.3, 3) if nw > 1 else np.array([L / 2] * 3)
        X.append(w @ q_.T + c)
    return np.vstack(X), np.tile([-0.834, 0.417, 0.417], nw), np.repeat(np.arange(nw), 3)


def exact_ewald(X, Q, mol, L, beta, kcoul, shells=3, kmax=30):
    N = len(Q)
    Ed = 0.0
    rr = range(-shells, shells + 1)
    for i in range(N):
        for j in range(N):
            for nx in rr:
                for ny in rr:
                    for nz in rr:
                        if nx == ny == nz == 0 and (i == j or mol[i] == mol[j]):
                            continue
                        r = np.linalg.norm(X[j] - X[i] + np.array([nx, ny, nz]) * L)
                        Ed += 0.5 * Q[i] * Q[j] * erfc(beta * r) / r
    m = np.arange(-kmax, kmax + 1)
    kx, ky, kz = np.meshgrid(m, m, m, indexing='ij')
    K = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], 1) * 2 * np.pi / L
    K = K[np.any(K != 0, axis=1)]
    k2 = (K ** 2).sum(1)
    S = (Q[None, :] * np.exp(1j * (K @ X.T))).sum(1)
    Er = (2 * np.pi / L ** 3) * np.sum(np.exp(-k2 / (4 * beta ** 2)) / k2 * np.abs(S) ** 2)
    Es = beta / np.sqrt(np.pi) * np.sum(Q ** 2)
    Ex = sum(Q[a] * Q[b] * erf(beta * np.linalg.norm(X[a] - X[b])) / np.linalg.norm(X[a] - X[b])
             for a in range(N) for b in range(a + 1, N) if mol[a] == mol[b])
    return kcoul * (Ed + Er - Es - Ex)


def part1(wd):
    L = 2.4
    print('Part 1: exact Ewald sum of rigid TIP3P water in a %.1f nm box (kJ/mol)' % L)
    for nw in (1, 8):
        X, Q, mol = water_boxes(nw, L)
        N = len(Q)
        ex_g = exact_ewald(X, Q, mol, L, 3.0, K_GMX)
        ex_a = exact_ewald(X, Q, mol, L, 3.0, K_AMBER)
        print('  %d water(s): exact %.6f (GROMACS constant), %.6f (Amber constant)' % (nw, ex_g, ex_a))
        d = os.path.join(wd, 'water%d' % nw)
        os.makedirs(d, exist_ok=True)
        # GROMACS
        open(os.path.join(d, 'w.top'), 'w').write(
            '#include "%s/lipid21.ff/forcefield.itp"\n#include "%s/lipid21.ff/tip3p.itp"\n'
            '[ system ]\nw\n[ molecules ]\nSOL %d\n' % (ROOT, ROOT, nw))
        atoms = [(k // 3 + 1, 'SOL', ['OW', 'HW1', 'HW2'][k % 3]) for k in range(N)]
        gmxio.write_g96(os.path.join(d, 'w.g96'), atoms, [X], (L, L, L))
        for rtol in (1e-4, 1e-5, 1e-7):
            open(os.path.join(d, 'w.mdp'), 'w').write(
                'integrator=md\nnsteps=0\ncontinuation=yes\ncutoff-scheme=Verlet\nverlet-buffer-tolerance=-1\n'
                'rlist=1.0\ncoulombtype=PME\ncoulomb-modifier=None\nrcoulomb=1.0\newald-rtol=%g\npme-order=8\n'
                'fourier-spacing=0.03\nvdwtype=Cut-off\nvdw-modifier=None\nrvdw=1.0\nnstcalcenergy=1\nnstenergy=1\n' % rtol)
            gmxio.gmx(['grompp', '-f', 'w.mdp', '-c', 'w.g96', '-p', 'w.top', '-o', 'w.tpr', '-maxwarn', '2'], d)
            gmxio.gmx(['mdrun', '-s', 'w.tpr', '-deffnm', 'w', '-ntmpi', '1'], d)
            beta = 1.0 / float(re.search(r'width \(1/beta\) of ([0-9.]+)', open(os.path.join(d, 'w.log')).read()).group(1))
            gmxio.gmx(['energy', '-f', 'w.edr', '-o', 'w.xvg'], d, stdin='Coulomb-(SR)\nCoul.-recip.\n\n')
            leg, rows = gmxio.read_xvg(os.path.join(d, 'w.xvg'))
            g = rows[0][1] + rows[0][2]
            exb = exact_ewald(X, Q, mol, L, beta, K_GMX)
            print('    GROMACS ewald-rtol %.0e (beta %.3f/nm): %.6f  vs exact at this beta %+.6f'
                  % (rtol, beta, g, g - exb))
        # sander
        with open(os.path.join(d, 'w.pdb'), 'w') as f:
            for k in range(N):
                xx = X[k] * 10
                f.write('ATOM  %5d  %-3s WAT  %4d    %8.3f%8.3f%8.3f\n' % (k + 1, ['O', 'H1', 'H2'][k % 3], k // 3 + 1, *xx))
                if k % 3 == 2:
                    f.write('TER\n')
        open(os.path.join(d, 'w.leap'), 'w').write(
            'source leaprc.water.tip3p\nm = loadpdb w.pdb\nset m box {%g %g %g}\nsaveamberparm m w.prmtop w.inpcrd\nquit\n'
            % (L * 10, L * 10, L * 10))
        subprocess.run(['tleap', '-f', 'w.leap'], cwd=d, capture_output=True, check=True)
        xyz = (X * 10).ravel()
        with open(os.path.join(d, 'w.rst7'), 'w') as f:
            f.write('w\n%6d\n' % N)
            for i in range(0, len(xyz), 6):
                f.write(''.join('%12.7f' % v for v in xyz[i:i + 6]) + '\n')
            f.write('%12.7f%12.7f%12.7f%12.7f%12.7f%12.7f\n' % (L * 10, L * 10, L * 10, 90, 90, 90))
        for tol in (1e-5, 1e-7, 1e-9):
            open(os.path.join(d, 'w.in'), 'w').write(
                'sp\n&cntrl\n imin=1, maxcyc=1, ncyc=1, ntb=1, cut=10.0, ntc=1, ntf=1, ntpr=1, ntxo=1,\n/\n'
                '&ewald\n dsum_tol=%g, order=8, nfft1=80, nfft2=80, nfft3=80, vdwmeth=0,\n/\n' % tol)
            subprocess.run(['sander', '-O', '-i', 'w.in', '-p', 'w.prmtop', '-c', 'w.rst7', '-o', 'w.out', '-r', 'w.r'],
                           cwd=d, check=True, capture_output=True)
            t = open(os.path.join(d, 'w.out')).read()
            beta = float(re.search(r'Ewald Coefficient =\s*([0-9.]+)', t).group(1)) * 10
            eel = float(re.search(r'EEL     =\s*([-0-9.]+)', t[t.index('NSTEP'):]).group(1)) * 4.184
            print('    sander  dsum_tol   %.0e (beta %.3f/nm): %.6f  vs exact            %+.6f  (EEL printed to 1e-4 kcal)'
                  % (tol, beta, eel, eel - ex_a))


def part2(wd):
    print('\nPart 2: total electrostatic energy of the POPC reference bilayer (kJ/mol, GROMACS Coulomb constant)')
    d = os.path.join(wd, 'popc')
    os.makedirs(d, exist_ok=True)
    src = os.path.join(VB.REPO, 'POPC')
    leap = open(os.path.join(src, 'build.leap')).read()
    leap = leap.replace('input_POPC_128.pdb', os.path.join(src, 'input_POPC_128.pdb'))
    leap = leap.replace('POPC_128.prmtop POPC_128.inpcrd', 'sys.prmtop sys.inpcrd')
    open(os.path.join(d, 'build.leap'), 'w').write(leap)
    subprocess.run(['tleap', '-f', 'build.leap'], cwd=d, capture_output=True, check=True)
    shutil.copy(os.path.join(src, 'start.rst'), os.path.join(d, 'start.rst'))
    prm = Prmtop(os.path.join(d, 'sys.prmtop'))
    x, box = VB.read_rst7_box(os.path.join(d, 'start.rst'), prm.natom)
    kq = K_GMX / K_AMBER
    for tol, sp, order, cut in ((1e-7, 0.6, 6, 10.0), (1e-9, 0.45, 8, 12.0), (1e-11, 0.35, 8, 14.0)):
        nfft = [VB.fft_size(int(math.ceil(b / sp))) for b in box]
        open(os.path.join(d, 'a.in'), 'w').write(
            'sp\n&cntrl\n imin=1, maxcyc=1, ncyc=1, ntb=1, cut=%g, ntc=1, ntf=1, ntpr=1, ntxo=1,\n/\n'
            '&ewald\n dsum_tol=%g, order=%d, nfft1=%d, nfft2=%d, nfft3=%d, vdwmeth=0,\n/\n' % (cut, tol, order, *nfft))
        subprocess.run(['pmemd', '-O', '-i', 'a.in', '-p', 'sys.prmtop', '-c', 'start.rst', '-o', 'a.out', '-r', 'a.r',
                        '-inf', 'a.info'], cwd=d, check=True, capture_output=True)
        a = VB.parse_amber_out(os.path.join(d, 'a.out'))
        print('  pmemd   dsum_tol %.0e, grid %.2f A, order %d, cut %4.1f A (beta %.3f/nm): %.2f'
              % (tol, sp, order, cut, a['ewald_coefficient'] * 10, a['eel'] * 4.184 * kq))
    # GROMACS topology
    if not os.path.exists(os.path.join(d, 'lipid21.ff')):
        os.symlink(os.path.join(ROOT, 'lipid21.ff'), os.path.join(d, 'lipid21.ff'))
    open(os.path.join(d, 'topol.top'), 'w').write(
        '#include "lipid21.ff/forcefield.itp"\n#include "lipid21.ff/POPC.itp"\n#include "lipid21.ff/tip3p.itp"\n'
        '[ system ]\nPOPC\n[ molecules ]\nPOPC 128\nSOL 5120\n')
    atoms = [(prm.resnum[i] + 1, prm.resname[i], {'O': 'OW', 'H1': 'HW1', 'H2': 'HW2'}.get(prm.names[i], prm.names[i])
              if prm.resname[i] == 'WAT' else prm.names[i]) for i in range(prm.natom)]
    gmxio.write_g96(os.path.join(d, 'conf.g96'), atoms, [x / 10.0], [b / 10.0 for b in box])
    for rtol in (1e-5, 1e-6, 1e-7):
        open(os.path.join(d, 'g.mdp'), 'w').write(
            'integrator=md\nnsteps=0\ncontinuation=yes\ncutoff-scheme=Verlet\nverlet-buffer-tolerance=-1\nrlist=1.0\n'
            'coulombtype=PME\ncoulomb-modifier=None\nrcoulomb=1.0\newald-rtol=%g\npme-order=8\nfourier-spacing=0.04\n'
            'vdwtype=Cut-off\nvdw-modifier=None\nrvdw=1.0\nDispCorr=no\nnstcalcenergy=1\nnstenergy=1\n' % rtol)
        gmxio.gmx(['grompp', '-f', 'g.mdp', '-c', 'conf.g96', '-p', 'topol.top', '-o', 'g.tpr'], d)
        gmxio.gmx(['mdrun', '-s', 'g.tpr', '-deffnm', 'g', '-ntmpi', '1'], d)
        beta = 1.0 / float(re.search(r'width \(1/beta\) of ([0-9.]+)', open(os.path.join(d, 'g.log')).read()).group(1))
        gmxio.gmx(['energy', '-f', 'g.edr', '-o', 'g.xvg'], d, stdin='Coulomb-(SR)\nCoul.-recip.\n\n')
        leg, rows = gmxio.read_xvg(os.path.join(d, 'g.xvg'))
        print('  GROMACS ewald-rtol %.0e, spacing 0.04 nm, order 8, rc 1.0 nm (beta %.3f/nm): %.2f'
              % (rtol, beta, rows[0][1] + rows[0][2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--workdir', default='pme_precision_test')
    args = ap.parse_args()
    wd = os.path.abspath(args.workdir)
    os.makedirs(wd, exist_ok=True)
    part1(wd)
    part2(wd)


if __name__ == '__main__':
    main()
