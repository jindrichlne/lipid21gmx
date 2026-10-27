"""Small helpers for writing GROMACS input (g96 coordinates) and running gmx."""
import os
import subprocess


def write_g96(path, atoms, frames, box_nm, title='frames'):
    """atoms: list of (resnr, resname, atomname); frames: list of coordinate lists (nm).

    One frame is written as POSITION (usable by grompp -c), several frames as a
    trajectory of TIMESTEP/POSITIONRED/BOX blocks (usable by mdrun -rerun).
    """
    with open(path, 'w') as out:
        out.write('TITLE\n%s\nEND\n' % title)
        if len(frames) == 1:
            out.write('POSITION\n')
            for i, ((rnr, rn, an), x) in enumerate(zip(atoms, frames[0])):
                out.write('%5d %-5s %-5s%7d%15.9f%15.9f%15.9f\n' % (rnr % 100000, rn, an, (i + 1) % 10000000, x[0], x[1], x[2]))
            out.write('END\nBOX\n%15.9f%15.9f%15.9f\nEND\n' % tuple(box_nm))
        else:
            for n, xyz in enumerate(frames):
                out.write('TIMESTEP\n%15d%15.6f\nEND\nPOSITIONRED\n' % (n, float(n)))
                for x in xyz:
                    out.write('%15.9f%15.9f%15.9f\n' % (x[0], x[1], x[2]))
                out.write('END\nBOX\n%15.9f%15.9f%15.9f\nEND\n' % tuple(box_nm))


def gmx(args, cwd, stdin=None, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(['gmx', '-quiet', '-nocopyright'] + args, cwd=cwd, input=stdin, capture_output=True, text=True, env=e)
    if r.returncode != 0:
        raise RuntimeError('gmx %s failed:\n%s\n%s' % (' '.join(args), r.stdout[-3000:], r.stderr[-6000:]))
    return r.stdout + r.stderr


def read_xvg(path):
    legends, rows = [], []
    for l in open(path):
        if l.startswith('@ s') and 'legend' in l:
            legends.append(l.split('"')[1])
        elif l[0] not in '#@':
            rows.append([float(x) for x in l.split()])
    return legends, rows
