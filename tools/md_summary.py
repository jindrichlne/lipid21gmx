#!/usr/bin/env python3
"""Block averages of a bilayer NPT run from a GROMACS .edr file.

    md_summary.py ener.edr NLIPIDS [--blocks 5] [--label text]

Prints area per lipid (box x*y / (NLIPIDS/2)), box height, density, temperature
and pressure for consecutive time blocks.
"""
import argparse
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmxio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('edr')
    ap.add_argument('nlipids', type=int)
    ap.add_argument('--blocks', type=int, default=5)
    ap.add_argument('--label', default='')
    args = ap.parse_args()
    xvg = args.edr[:-4] + '_summary.xvg'
    r = subprocess.run(['gmx', '-quiet', 'energy', '-f', args.edr, '-o', xvg],
                       input='Box-X\nBox-Y\nBox-Z\nDensity\nTemperature\nPressure\n\n',
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])
    leg, rows = gmxio.read_xvg(xvg)
    col = {k: i + 1 for i, k in enumerate(leg)}
    t = [row[0] for row in rows]
    apl = [row[col['Box-X']] * row[col['Box-Y']] / (args.nlipids / 2) * 100 for row in rows]   # A^2
    out = ['%s%d frames, %.1f-%.1f ns' % (args.label + ': ' if args.label else '', len(rows), t[0] / 1000, t[-1] / 1000),
           '   time (ns)        APL (A^2)       box z (nm)   density (kg/m3)   T (K)    P (bar)']
    n = len(rows)
    edges = [round(k * n / args.blocks) for k in range(args.blocks + 1)]
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        sl = slice(a, b)
        f = lambda key: statistics.mean(row[col[key]] for row in rows[sl])
        out.append('  %5.2f-%5.2f   %7.2f +- %4.2f    %7.3f       %8.1f       %6.1f  %7.1f'
                   % (t[a] / 1000, t[b - 1] / 1000, statistics.mean(apl[sl]),
                      statistics.pstdev(apl[sl]), f('Box-Z'), f('Density'), f('Temperature'), f('Pressure')))
    out.append('   start APL %.2f A^2, final-block mean %.2f A^2' % (apl[0], statistics.mean(apl[edges[-2]:])))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
