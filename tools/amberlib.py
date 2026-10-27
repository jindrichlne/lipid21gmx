"""Minimal readers for Amber OFF library (.lib) and parameter (.dat/.frcmod) files.

Only the subset of the formats used by Lipid21 (and the Amber water/ion files)
is implemented.  Pure standard-library Python.
"""
import re
from collections import OrderedDict


# --------------------------------------------------------------------------
# OFF library
# --------------------------------------------------------------------------
class OffResidue:
    def __init__(self, name):
        self.name = name
        self.atoms = []        # list of dicts: name, type, charge, element
        self.bonds = []        # list of (i, j) 0-based
        self.connect = []      # [head, tail] 0-based (-1 = none)
        self.positions = []    # Angstrom
        self.residueconnect = []

    @property
    def charge(self):
        return sum(a['charge'] for a in self.atoms)

    def index(self, atomname):
        for i, a in enumerate(self.atoms):
            if a['name'] == atomname:
                return i
        raise KeyError(atomname)


def _tok(line):
    # tokens separated by whitespace, double-quoted strings may hold spaces
    return [t[1:-1] if t.startswith('"') else t for t in re.findall(r'"[^"]*"|\S+', line)]


def read_off(path):
    lines = open(path).read().splitlines()
    residues = OrderedDict()
    i = 0
    if not lines[0].startswith('!!index'):
        raise ValueError('not an OFF file')
    i = 1
    while i < len(lines) and not lines[i].startswith('!'):
        name = _tok(lines[i])[0]
        residues[name] = OffResidue(name)
        i += 1
    section = None
    while i < len(lines):
        line = lines[i]
        if line.startswith('!entry.'):
            m = re.match(r'!entry\.(.+?)\.unit\.(\S+)', line)
            resname, section = m.group(1), m.group(2)
            res = residues[resname]
            i += 1
            continue
        t = _tok(line)
        if section == 'atoms':
            res.atoms.append(dict(name=t[0], type=t[1], element=int(t[6]), charge=float(t[7]), charge_str=t[7]))
        elif section == 'connectivity':
            res.bonds.append((int(t[0]) - 1, int(t[1]) - 1))
        elif section == 'connect':
            res.connect.append(int(t[0]) - 1)
        elif section == 'positions':
            res.positions.append(tuple(float(x) for x in t[:3]))
        elif section == 'residueconnect':
            res.residueconnect = [int(x) - 1 for x in t]
        i += 1
    return residues


# --------------------------------------------------------------------------
# Parameter files (.dat main parameter file; frcmod)
# --------------------------------------------------------------------------
def _types(field, n):
    """Split an Amber type field like 'cA-cB-cC-cD' into n stripped types."""
    parts = field.split('-')
    if len(parts) != n:
        raise ValueError(field)
    return tuple(p.strip() for p in parts)


class ParmSet:
    def __init__(self):
        self.masses = OrderedDict()     # type -> mass
        self.bonds = OrderedDict()      # (t1,t2) -> (k, r0)        Amber units
        self.angles = OrderedDict()     # (t1,t2,t3) -> (k, theta0)
        self.dihedrals = OrderedDict()  # (t1..t4) -> list of dict(idivf, pk, phase, pn, scee, scnb)
        self.impropers = OrderedDict()  # (t1..t4) -> (pk, phase, pn)
        self.lj = OrderedDict()         # type -> (rmin_half, eps)
        self.equiv = []                 # list of lists (type equivalencing for LJ)


def _scale(comment, key, default):
    m = re.search(key + r'\s*=\s*([-+0-9.eE]+)', comment)
    return float(m.group(1)) if m else default


def read_dat(path, ps=None):
    """Read an Amber main parameter file (parm.dat style).

    Fixed-column layout as described in the Amber manual (file format section);
    values are parsed with whitespace splitting after the atom-type columns.
    """
    ps = ps or ParmSet()
    lines = open(path).read().splitlines()
    i = 1  # skip title
    # 1) masses
    while lines[i].strip():
        t = lines[i][:2].strip()
        ps.masses[t] = float(lines[i][2:].split()[0])
        i += 1
    i += 1
    # 2) hydrophilic atom types line
    # (Lipid21.dat has none: the next block starts directly with bonds -- detect)
    if '-' not in lines[i][:5]:
        i += 1
    # 3) bonds
    while lines[i].strip():
        l = lines[i]
        key = _types(l[:5], 2)
        k, r0 = (float(x) for x in l[5:].split()[:2])
        ps.bonds[key] = (k, r0)
        i += 1
    i += 1
    # 4) angles
    while lines[i].strip():
        l = lines[i]
        key = _types(l[:8], 3)
        k, th = (float(x) for x in l[8:].split()[:2])
        ps.angles[key] = (k, th)
        i += 1
    i += 1
    # 5) proper dihedrals
    while lines[i].strip():
        l = lines[i]
        key = _types(l[:11], 4)
        f = l[11:].split()
        idivf, pk, phase, pn = int(float(f[0])), float(f[1]), float(f[2]), float(f[3])
        comment = l[11:]
        ps.dihedrals.setdefault(key, [])
        term = dict(idivf=idivf, pk=pk, phase=phase, pn=abs(int(round(pn))),
                    scee=_scale(comment, 'SCEE', 1.2), scnb=_scale(comment, 'SCNB', 2.0),
                    last=(pn > 0))
        ps.dihedrals[key].append(term)
        i += 1
    i += 1
    # 6) impropers
    while lines[i].strip():
        l = lines[i]
        key = _types(l[:11], 4)
        f = l[11:].split()
        pk, phase, pn = float(f[0]), float(f[1]), float(f[2])
        ps.impropers[key] = (pk, phase, abs(int(round(pn))))
        i += 1
    i += 1
    # 7) 10-12 H-bond terms (ignored, must be zero)
    while lines[i].strip():
        f = lines[i].split()
        if float(f[2]) != 0.0 or float(f[3]) != 0.0:
            raise ValueError('non-zero 10-12 term: ' + lines[i])
        i += 1
    i += 1
    # 8) equivalencing for LJ
    while lines[i].strip():
        ps.equiv.append(lines[i].split())
        i += 1
    i += 1
    # 9) LJ
    label, kind = lines[i].split()[:2]
    if kind != 'RE':
        raise ValueError('only RE LJ format supported')
    i += 1
    while i < len(lines) and lines[i].strip() and not lines[i].startswith('END'):
        f = lines[i].split()
        ps.lj[f[0]] = (float(f[1]), float(f[2]))
        i += 1
    # apply equivalences
    for eq in ps.equiv:
        for t in eq[1:]:
            if t not in ps.lj and eq[0] in ps.lj:
                ps.lj[t] = ps.lj[eq[0]]
    return ps


def read_frcmod(path, ps=None):
    """Read an frcmod file (MASS/BOND/ANGL/DIHE/IMPR/NONB sections)."""
    ps = ps or ParmSet()
    sec = None
    for l in open(path).read().splitlines()[1:]:
        s = l.strip()
        head = s.split()[0] if s else ''
        if head in ('MASS', 'BOND', 'ANGL', 'ANGLE', 'DIHE', 'IMPR', 'IMPROPER', 'NONB', 'NONBON', 'HBON', 'CMAP', 'LJEDIT'):
            sec = head[:4]
            continue
        if not s or s.startswith('END'):
            continue
        if sec == 'MASS':
            # ion types may be longer than the canonical 2 columns ("Mg2+")
            f = l.split() if l[:2].strip() and len(l.split()[0]) > 2 else [l[:2].strip()] + l[2:].split()
            ps.masses[f[0]] = float(f[1])
        elif sec == 'BOND':
            ps.bonds[_types(l[:5], 2)] = tuple(float(x) for x in l[5:].split()[:2])
        elif sec == 'ANGL':
            ps.angles[_types(l[:8], 3)] = tuple(float(x) for x in l[8:].split()[:2])
        elif sec == 'NONB':
            f = l.split()
            ps.lj[f[0]] = (float(f[1]), float(f[2]))
        elif sec in ('DIHE', 'IMPR', 'LJED', 'HBON', 'CMAP'):
            raise NotImplementedError('frcmod section %s not needed here' % sec)
    return ps
