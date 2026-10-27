"""Minimal, dependency-free reader for Amber prmtop files (used for validation).

Returns raw arrays plus convenience lists of bonded terms with their
parameters, in Amber units, exactly as stored in the file.
"""
import math
import re

AMBER_CHARGE_SCALE = 18.2223


def _parse_format(fmt):
    m = re.match(r'\((\d+)([aAiIeEfF])(\d+)(?:\.\d+)?\)', fmt.strip())
    return int(m.group(1)), m.group(2).lower(), int(m.group(3))


def read_prmtop(path):
    flags = {}
    name = fmt = None
    data = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('%FLAG'):
                if name is not None:
                    flags[name] = (fmt, data)
                name = line.split()[1]
                data = []
            elif line.startswith('%FORMAT'):
                fmt = _parse_format(line[7:])
            elif line.startswith('%VERSION') or line.startswith('%COMMENT'):
                continue
            elif name is not None:
                per, kind, width = fmt
                for i in range(0, len(line), width):
                    tok = line[i:i + width]
                    if not tok.strip() and kind != 'a':
                        continue
                    if kind == 'a':
                        data.append(tok.strip())
                    elif kind == 'i':
                        data.append(int(tok))
                    else:
                        data.append(float(tok))
        if name is not None:
            flags[name] = (fmt, data)
    return {k: v[1] for k, v in flags.items()}


class Prmtop:
    def __init__(self, path):
        f = self.raw = read_prmtop(path)
        ptr = f['POINTERS']
        self.natom = ptr[0]
        self.ntypes = ptr[1]
        self.names = f['ATOM_NAME'][:self.natom]
        self.types = f['AMBER_ATOM_TYPE'][:self.natom]
        self.charges = [q / AMBER_CHARGE_SCALE for q in f['CHARGE']]
        self.masses = f['MASS']
        self.type_index = [t - 1 for t in f['ATOM_TYPE_INDEX']]
        reslab = f['RESIDUE_LABEL']
        resptr = [p - 1 for p in f['RESIDUE_POINTER']] + [self.natom]
        self.resname = [None] * self.natom
        self.resnum = [None] * self.natom
        for r in range(len(reslab)):
            for a in range(resptr[r], resptr[r + 1]):
                self.resname[a] = reslab[r]
                self.resnum[a] = r
        # bonds
        kb, rb = f['BOND_FORCE_CONSTANT'], f['BOND_EQUIL_VALUE']
        self.bonds = []
        for key in ('BONDS_INC_HYDROGEN', 'BONDS_WITHOUT_HYDROGEN'):
            arr = f.get(key, [])
            for n in range(0, len(arr), 3):
                i, j, t = arr[n] // 3, arr[n + 1] // 3, arr[n + 2] - 1
                self.bonds.append((i, j, kb[t], rb[t]))
        ka, ta = f['ANGLE_FORCE_CONSTANT'], f['ANGLE_EQUIL_VALUE']
        self.angles = []
        for key in ('ANGLES_INC_HYDROGEN', 'ANGLES_WITHOUT_HYDROGEN'):
            arr = f.get(key, [])
            for n in range(0, len(arr), 4):
                i, j, k, t = arr[n] // 3, arr[n + 1] // 3, arr[n + 2] // 3, arr[n + 3] - 1
                self.angles.append((i, j, k, ka[t], math.degrees(ta[t])))
        kd, pd, phd = f['DIHEDRAL_FORCE_CONSTANT'], f['DIHEDRAL_PERIODICITY'], f['DIHEDRAL_PHASE']
        scee = f.get('SCEE_SCALE_FACTOR', [1.2] * len(kd))
        scnb = f.get('SCNB_SCALE_FACTOR', [2.0] * len(kd))
        self.dihedrals = []   # (i,j,k,l, pk, phase_deg, pn, improper, ignore14, scee, scnb)
        for key in ('DIHEDRALS_INC_HYDROGEN', 'DIHEDRALS_WITHOUT_HYDROGEN'):
            arr = f.get(key, [])
            for n in range(0, len(arr), 5):
                i, j, k, l, t = arr[n:n + 5]
                ignore14 = k < 0
                improper = l < 0
                i, j, k, l = abs(i) // 3, abs(j) // 3, abs(k) // 3, abs(l) // 3
                t -= 1
                self.dihedrals.append((i, j, k, l, kd[t], math.degrees(phd[t]), pd[t],
                                       improper, ignore14, scee[t], scnb[t]))
        # exclusions
        nex = f['NUMBER_EXCLUDED_ATOMS']
        exl = f['EXCLUDED_ATOMS_LIST']
        self.exclusions = set()
        p = 0
        for i in range(self.natom):
            for e in exl[p:p + nex[i]]:
                if e > 0:
                    self.exclusions.add((min(i, e - 1), max(i, e - 1)))
            p += nex[i]
        # LJ
        self.acoef = f['LENNARD_JONES_ACOEF']
        self.bcoef = f['LENNARD_JONES_BCOEF']
        self.nbidx = f['NONBONDED_PARM_INDEX']
        self.box = f.get('BOX_DIMENSIONS')

    def lj_ab(self, i, j):
        ti, tj = self.type_index[i], self.type_index[j]
        idx = self.nbidx[self.ntypes * ti + tj] - 1
        return self.acoef[idx], self.bcoef[idx]

    def pairs14(self):
        """1-4 pairs as evaluated by sander/pmemd: {(i,l): (scee, scnb)}."""
        out = {}
        for (i, j, k, l, pk, ph, pn, imp, ign, scee, scnb) in self.dihedrals:
            if imp or ign:
                continue
            key = (min(i, l), max(i, l))
            if key in out:
                raise ValueError('1-4 pair counted twice: %s' % (key,))
            out[key] = (scee, scnb)
        return out
