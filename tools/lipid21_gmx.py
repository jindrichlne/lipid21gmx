#!/usr/bin/env python3
"""Convert the Amber Lipid21 force field (lipid21.dat + lipid21.lib) to GROMACS.

The topology of a lipid is assembled from Lipid21 residues exactly as tleap does
it for ``sequence { TAIL1 HEAD TAIL2 }``:

* bonds from the residue templates plus tail(i)--head(i+1) inter-residue bonds,
* every angle and every proper torsion of the bond graph,
* torsion parameters by explicit type match (Lipid21 has no wildcard torsions);
  a torsion without parameters is dropped, and so is its 1-4 pair (tleap does
  the same),
* impropers on every atom with three bonded neighbours: the outer atoms are
  sorted by atom type, ties by atom index, and the central atom goes third
  (tleap convention); only impropers present in lipid21.dat are applied,
* 1-4 pairs are the end atoms of parameterised proper torsions that are not
  already 1-2 or 1-3 neighbours.  The 1-4 scaling factors are taken from the
  torsion (SCEE/SCNB keywords in lipid21.dat): SCEE=1.2 everywhere, SCNB=2.0
  everywhere except cD-cD-cD-cD, which has SCNB=6.0.  GROMACS has a single
  global fudgeLJ, so pairs with SCNB != 2.0 get explicit LJ-14 parameters.
* charges are those of lipid21.lib; a non-integral net charge from rounding
  (sphingomyelin: +0.000003 e) is removed from the atom with the largest
  |charge| (SPM P31) unless --keep-lib-charges is given.

Usage:
    lipid21_gmx.py ff  [--lib F] [--dat F] -o lipid21.ff     # whole force field
    lipid21_gmx.py itp NAME RES1 [RES2 ...] [-o NAME.itp]    # one molecule

Pure standard-library Python (3.8+).
"""
import argparse
import math
import os
import sys
from collections import OrderedDict
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amberlib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(os.path.dirname(HERE), 'lipid21', 'lipid21')
KCAL = 4.184                   # kJ/kcal (thermochemical calorie, as used by Amber)
RMIN_TO_SIGMA = 2.0 ** (-1.0 / 6.0)
ELEMENT = {'c': 6, 'h': 1, 'n': 7, 'o': 8, 'p': 15}
SCEE_DEFAULT = 1.2
SCNB_DEFAULT = 2.0
FUDGE_QQ = 1.0 / SCEE_DEFAULT
FUDGE_LJ = 1.0 / SCNB_DEFAULT

# Lipid species of charmmlipid2amber.csv (Lipid21 v1.0) that can be built from
# lipid21.lib.  The PL* species need the linoleoyl residue (LEO), which is not
# part of Lipid21 and are therefore not listed.
TAILS = OrderedDict([('LA', 'LAL'), ('M', 'MY'), ('P', 'PA'), ('S', 'ST'),
                     ('O', 'OL'), ('A', 'AR'), ('D', 'DHA')])
HEADS = OrderedDict([('PA', 'PH-'), ('PC', 'PC'), ('PE', 'PE'), ('PG', 'PGR'), ('PS', 'PS')])


def standard_lipids():
    """name -> residue list, for the Lipid21 species of charmmlipid2amber.csv."""
    out = OrderedDict()
    for h, hr in HEADS.items():
        for sym, t in (('DL', 'LAL'), ('DM', 'MY'), ('DP', 'PA'), ('DS', 'ST'),
                       ('DO', 'OL'), ('DA', 'AR')):
            out[sym + h] = [t, hr, t]
        out['PO' + h] = ['PA', hr, 'OL']
        out['SD' + h] = ['ST', hr, 'DHA']
    out['PSM'] = ['PA', 'SPM', 'SA']
    out['SSM'] = ['ST', 'SPM', 'SA']
    out['CHL'] = ['CHL']
    return out


def fmt(x, sig=10):
    """Shortest decimal representation with up to `sig` significant digits."""
    if x == 0:
        return '0.0'
    s = '%.*g' % (sig, x)
    if 'e' in s or 'E' in s:
        s = '%.*f' % (sig, x)
        s = s.rstrip('0')
    if '.' not in s:
        s += '.0'
    return s


# --------------------------------------------------------------------------
class ForceField:
    def __init__(self, lib, dat):
        self.lib_path, self.dat_path = lib, dat
        self.residues = amberlib.read_off(lib)
        self.ps = amberlib.read_dat(dat)
        self._check()

    def _check(self):
        ps = self.ps
        for key, terms in ps.dihedrals.items():
            if not terms[-1]['last']:
                raise ValueError('unterminated torsion block %s' % (key,))
            if any(t['idivf'] != 1 for t in terms):
                pass  # handled by dividing pk / idivf
            sc = {(t['scee'], t['scnb']) for t in terms}
            if len(sc) != 1:
                raise ValueError('mixed 1-4 scaling inside torsion %s' % (key,))
        for t in ps.masses:
            if t not in ps.lj:
                raise ValueError('no LJ parameters for type %s' % t)

    # -- parameter lookup (tleap semantics) ---------------------------------
    def bond(self, a, b):
        return self.ps.bonds.get((a, b)) or self.ps.bonds.get((b, a))

    def angle(self, a, b, c):
        return self.ps.angles.get((a, b, c)) or self.ps.angles.get((c, b, a))

    def torsion(self, a, b, c, d):
        ps = self.ps.dihedrals
        for key in ((a, b, c, d), (d, c, b, a), ('X', b, c, 'X'), ('X', c, b, 'X')):
            if key in ps:
                return key, ps[key]
        return None, None

    def improper(self, a, b, c, d):
        ps = self.ps.impropers
        for key in ((a, b, c, d), ('X', b, c, d), ('X', 'X', c, d)):
            if key in ps:
                return key, ps[key]
        return None, None

    def sigma_eps(self, t):
        rmin_half, eps = self.ps.lj[t]
        return 2.0 * rmin_half * RMIN_TO_SIGMA / 10.0, eps * KCAL


class Molecule:
    """A molecule assembled from Lipid21 residues the way tleap's `sequence` does."""

    def __init__(self, ff, resnames, name=None, neutralize=True):
        self.ff = ff
        self.name = name or '_'.join(resnames)
        self.resnames = list(resnames)
        self.atoms = []
        self.bonds = []
        self.warnings = []
        self.notes = []
        self.charge_adjust = {}   # atom index -> charge change made by _neutralize()
        offsets = []
        for r, rn in enumerate(resnames):
            res = ff.residues[rn]
            offsets.append(len(self.atoms))
            for a in res.atoms:
                self.atoms.append(dict(name=a['name'], type=a['type'], charge=a['charge'],
                                       qdec=Decimal(a['charge_str']), resnr=r + 1, resname=rn))
            for i, j in res.bonds:
                self.bonds.append((offsets[r] + i, offsets[r] + j))
        if neutralize:
            self._neutralize()
        for r in range(len(resnames) - 1):
            tail = ff.residues[resnames[r]].connect[1]
            head = ff.residues[resnames[r + 1]].connect[0]
            if tail < 0 or head < 0:
                raise ValueError('residues %s and %s cannot be connected' % (resnames[r], resnames[r + 1]))
            self.bonds.append((offsets[r] + tail, offsets[r + 1] + head))
        n = len(self.atoms)
        self.nbr = [[] for _ in range(n)]
        for i, j in self.bonds:
            self.nbr[i].append(j)
            self.nbr[j].append(i)
        for a in self.atoms:
            a['mass'] = ff.ps.masses[a['type']]
        self._build()

    def t(self, i):
        return self.atoms[i]['type']

    def _neutralize(self):
        """Remove a non-integral net charge left by rounding in lipid21.lib.

        Sphingomyelin (SA -0.069718 + SPM +0.069721) carries +0.000003 e, which
        makes grompp warn about Ewald electrostatics with a net charge for
        larger SM contents.  The excess is removed from the atom with the
        largest |charge| (SPM P31: 1.324762 -> 1.324759).  Exact decimal sums.
        """
        tot = sum(a['qdec'] for a in self.atoms)
        excess = tot - tot.to_integral_value()
        if excess == 0:
            return
        if abs(excess) > Decimal('0.0001'):
            raise ValueError('%s: net charge %s is not a rounding artefact' % (self.name, tot))
        i = max(range(len(self.atoms)), key=lambda k: abs(self.atoms[k]['qdec']))
        a = self.atoms[i]
        new = a['qdec'] - excess
        self.notes.append('charge of %s %s changed from %s to %s (lipid21.lib net charge %s; made integral)'
                          % (a['resname'], a['name'], a['qdec'], new, tot))
        a['qdec'], a['charge'] = new, float(new)
        self.charge_adjust[i] = float(-excess)

    def _build(self):
        ff = self.ff
        n = len(self.atoms)
        # bonds
        self.bond_params = []
        for i, j in self.bonds:
            p = ff.bond(self.t(i), self.t(j))
            if p is None:
                raise ValueError('no bond parameter %s-%s' % (self.t(i), self.t(j)))
            self.bond_params.append(p)
        # angles: every i-j-k with i<k
        self.angles = []
        for j in range(n):
            nb = sorted(self.nbr[j])
            for a in range(len(nb)):
                for b in range(a + 1, len(nb)):
                    i, k = nb[a], nb[b]
                    p = ff.angle(self.t(i), self.t(j), self.t(k))
                    if p is None:
                        raise ValueError('no angle parameter %s-%s-%s' % (self.t(i), self.t(j), self.t(k)))
                    self.angles.append((i, j, k, p))
        # proper torsions: every path i-j-k-l, each counted once
        self.propers = []
        self.missing_torsions = []
        seen = set()
        for j, k in self.bonds:
            for i in sorted(self.nbr[j]):
                if i == k:
                    continue
                for l in sorted(self.nbr[k]):
                    if l == j or l == i:
                        continue
                    key = (i, j, k, l) if i < l else (l, k, j, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    tkey, terms = ff.torsion(*(self.t(x) for x in key))
                    if terms is None:
                        self.missing_torsions.append(key)
                        continue
                    self.propers.append((key, tkey, terms))
        # impropers: atoms with exactly three neighbours
        self.impropers = []
        for c in range(n):
            if len(self.nbr[c]) != 3:
                continue
            o = sorted(self.nbr[c], key=lambda x: (self.t(x), x))
            quad = (o[0], o[1], c, o[2])
            ikey, p = ff.improper(*(self.t(x) for x in quad))
            if p is not None:
                self.impropers.append((quad, ikey, p))
        # topological distances (up to 3 bonds)
        self.dist = {}
        for s in range(n):
            front, d = {s}, {s: 0}
            for step in (1, 2, 3):
                nxt = set()
                for x in front:
                    for y in self.nbr[x]:
                        if y not in d:
                            d[y] = step
                            nxt.add(y)
                front = nxt
            for y, dy in d.items():
                if y > s:
                    self.dist[(s, y)] = dy
        # 1-4 pairs from parameterised torsions
        self.pairs = OrderedDict()
        for key, tkey, terms in self.propers:
            i, l = key[0], key[3]
            pk = (min(i, l), max(i, l))
            if self.dist.get(pk) != 3 or pk in self.pairs:
                continue
            self.pairs[pk] = (terms[0]['scee'], terms[0]['scnb'], tkey)
        # every 1-4 neighbour pair should have a 1-4 interaction
        for pk, d in self.dist.items():
            if d == 3 and pk not in self.pairs:
                self.warnings.append('1-4 pair %d-%d has no torsion parameters (no 1-4 interaction, as in tleap)' % pk)
        for key in self.missing_torsions:
            self.warnings.append('no torsion parameters for %s (%s), torsion omitted as in tleap'
                                 % ('-'.join(self.atoms[x]['name'] for x in key),
                                    '-'.join(self.t(x) for x in key)))

    @property
    def charge(self):
        return sum(a['charge'] for a in self.atoms)

    # ------------------------------------------------------------------
    def itp(self, header=None):
        ff = self.ff
        L = []
        if header:
            L += ['; ' + h for h in header]
        L += ['; NOTE: ' + n for n in self.notes]
        if header or self.notes:
            L.append('')
        L += ['[ moleculetype ]', '; name  nrexcl', '%-8s 3' % self.name, '']
        L += ['[ atoms ]',
              ';   nr  type  resnr  residue  atom   cgnr      charge       mass']
        for i, a in enumerate(self.atoms):
            L.append('%6d  %-4s %6d  %-6s  %-5s %6d  %10.6f  %9s'
                     % (i + 1, a['type'], a['resnr'], a['resname'], a['name'], i + 1, a['charge'], fmt(a['mass'])))
        L.append('; total charge %s' % sum(a['qdec'] for a in self.atoms))
        L += ['', '[ bonds ]', ';  ai    aj  funct']
        for i, j in self.bonds:
            L.append('%6d %6d  1' % (i + 1, j + 1))
        L += ['', '[ pairs ]', ';  ai    aj  funct   (sigma epsilon: explicit LJ-14 where SCNB != %.1f)' % SCNB_DEFAULT]
        for (i, l), (scee, scnb, tkey) in self.pairs.items():
            if abs(scee - SCEE_DEFAULT) > 1e-12:
                raise NotImplementedError('SCEE != 1.2 would need pair function 2')
            if abs(scnb - SCNB_DEFAULT) < 1e-12:
                L.append('%6d %6d  1' % (i + 1, l + 1))
            else:
                si, ei = ff.sigma_eps(self.t(i))
                sl, el = ff.sigma_eps(self.t(l))
                L.append('%6d %6d  1  %.12f  %.12f  ; SCNB=%g (%s)'
                         % (i + 1, l + 1, 0.5 * (si + sl), math.sqrt(ei * el) / scnb, scnb, '-'.join(tkey)))
        L += ['', '[ angles ]', ';  ai    aj    ak  funct']
        for i, j, k, p in self.angles:
            L.append('%6d %6d %6d  1' % (i + 1, j + 1, k + 1))
        L += ['', '[ dihedrals ]', '; propers (Fourier series, parameters from ffbonded.itp)',
              ';  ai    aj    ak    al  funct']
        for key, tkey, terms in self.propers:
            L.append('%6d %6d %6d %6d  9' % tuple(x + 1 for x in key))
        L += ['', '[ dihedrals ]', '; impropers (Amber order: central atom third)',
              ';  ai    aj    ak    al  funct']
        for quad, ikey, p in self.impropers:
            L.append('%6d %6d %6d %6d  4' % tuple(x + 1 for x in quad))
        L.append('')
        return '\n'.join(L)


# --------------------------------------------------------------------------
def write_ffnonbonded(ff, path):
    L = ['; Lipid21 atom types (from %s)' % os.path.basename(ff.dat_path),
         '; sigma = 2*Rmin/2 / 2^(1/6), epsilon = eps * 4.184 kJ/mol',
         '', '[ atomtypes ]',
         '; name  at.num      mass   charge  ptype           sigma         epsilon']
    for t, m in ff.ps.masses.items():
        s, e = ff.sigma_eps(t)
        L.append('  %-4s  %6d  %8s   0.0000   A    %.12f  %.12f' % (t, ELEMENT[t[0]], fmt(m), s, e))
    open(path, 'w').write('\n'.join(L) + '\n')


def write_ffbonded(ff, path):
    ps = ff.ps
    L = ['; Lipid21 bonded parameters (from %s)' % os.path.basename(ff.dat_path),
         '; bonds:     kb = 2*K*4.184*100 kJ/mol/nm^2, b0 = r0/10 nm',
         '; angles:    kth = 2*K*4.184 kJ/mol/rad^2',
         '; dihedrals: k = (PK/IDIVF)*4.184 kJ/mol; function 9 = Amber Fourier series,',
         ';            function 4 = Amber improper (central atom third)',
         '', '[ bondtypes ]', '; i    j   func      b0          kb']
    done = set()
    for (a, b), (k, r0) in ps.bonds.items():
        if (b, a) in done:
            continue  # reversed duplicate (checked identical)
        done.add((a, b))
        L.append('  %-4s %-4s  1   %-10s  %s' % (a, b, fmt(r0 / 10.0), fmt(2 * k * KCAL * 100.0)))
    L += ['', '[ angletypes ]', '; i    j    k   func    th0        cth']
    done = set()
    for (a, b, c), (k, th) in ps.angles.items():
        if (c, b, a) in done:
            continue  # reversed duplicate (checked identical)
        done.add((a, b, c))
        L.append('  %-4s %-4s %-4s  1   %-9s  %s' % (a, b, c, fmt(th), fmt(2 * k * KCAL)))
    L += ['', '[ dihedraltypes ]', '; proper torsions', '; i    j    k    l   func   phase      kd        pn']
    done = set()
    for key, terms in ps.dihedrals.items():
        canon = min(key, key[::-1])
        if canon in done:
            continue  # reversed duplicate (checked identical when building the force field)
        done.add(canon)
        scnb = terms[0]['scnb']
        for t in terms:
            L.append('  %-4s %-4s %-4s %-4s  9   %8.3f   %-12s  %d%s'
                     % (key + (t['phase'], fmt(t['pk'] / t['idivf'] * KCAL), t['pn'],
                               '   ; SCNB=%g (explicit [ pairs ] parameters in the molecule topologies)' % scnb
                               if scnb != SCNB_DEFAULT and t is terms[0] else '')))
    L += ['', '[ dihedraltypes ]', '; improper torsions', '; i    j    k    l   func   phase      kd        pn']
    for key, (pk, phase, pn) in ps.impropers.items():
        L.append('  %-4s %-4s %-4s %-4s  4   %8.3f   %-12s  %d' % (key + (phase, fmt(pk * KCAL), pn)))
    open(path, 'w').write('\n'.join(L) + '\n')


# --------------------------------------------------------------------------
# Water and ions for the standalone force field, converted from the files that
# Amber's leaprc.water.tip3p loads (later files override earlier ones, as in tleap)
WATER_ION_FRCMODS = ['frcmod.tip3p', 'frcmod.ions1lm_126_tip3p', 'frcmod.ionsjc_tip3p',
                     'frcmod.ions234lm_126_tip3p']
# GROMACS molecule name, Amber type, element number, charge (same naming as amber14sb.ff/amber19sb.ff)
IONS = [('NA', 'Na+', 11, 1.0), ('K', 'K+', 19, 1.0), ('CL', 'Cl-', 17, -1.0),
        ('MG', 'Mg2+', 12, 2.0), ('CA', 'Ca2+', 20, 2.0)]


def find_amberhome():
    if os.environ.get('AMBERHOME'):
        return os.environ['AMBERHOME']
    for d in os.environ.get('PATH', '').split(os.pathsep):
        if os.path.exists(os.path.join(d, 'tleap')):
            return os.path.dirname(os.path.realpath(d))
    return None


def water_ion_params(amberhome):
    ps = amberlib.ParmSet()
    used = []
    for f in WATER_ION_FRCMODS:
        path = os.path.join(amberhome, 'dat', 'leap', 'parm', f)
        amberlib.read_frcmod(path, ps)
        used.append(f)
    solv = amberlib.read_off(os.path.join(amberhome, 'dat', 'leap', 'lib', 'solvents.lib'))['TP3']
    q = {a['name']: a['charge'] for a in solv.atoms}
    return ps, q, used


def write_water_ions(outdir, amberhome):
    ps, qw, used = water_ion_params(amberhome)
    sig = lambda t: 2.0 * ps.lj[t][0] * RMIN_TO_SIGMA / 10.0
    eps = lambda t: ps.lj[t][1] * KCAL
    src = '; converted from $AMBERHOME/dat/leap/parm/{%s} (load order of leaprc.water.tip3p)' % ','.join(used)
    L = ['; Atom types of TIP3P water and ions for the standalone Lipid21 force field', src,
         '; (same type names and values as amber14sb.ff/amber19sb.ff in GROMACS 2026)', '',
         '[ atomtypes ]', '; name        at.num      mass   charge  ptype           sigma         epsilon',
         '  %-11s  %6d  %8s   0.0000   A    %.12f  %.12f' % ('OW_tip3p', 8, fmt(ps.masses['OW']), sig('OW'), eps('OW')),
         '  %-11s  %6d  %8s   0.0000   A    %.12f  %.12f' % ('HW_tip3p', 1, fmt(ps.masses['HW']), 0.0, 0.0)]
    for mol, t, z, q in IONS:
        L.append('  %-11s  %6d  %8s   0.0000   A    %.12f  %.12f' % (t + '_tip3p', z, fmt(ps.masses[t]), sig(t), eps(t)))
    open(os.path.join(outdir, 'ffwater_ions.itp'), 'w').write('\n'.join(L) + '\n')

    roh, rhh = ps.bonds[('OW', 'HW')][1], ps.bonds[('HW', 'HW')][1]
    koh = ps.bonds[('OW', 'HW')][0]
    theta = 2.0 * math.degrees(math.asin(rhh / 2.0 / roh))
    W = ['; TIP3P water (Jorgensen et al. 1983) as used by Amber', src,
         '; Rigid geometry as in Amber (SETTLE on O-H %.4f A and H-H %.4f A, i.e. HOH %.2f deg);' % (roh, rhh, theta),
         '; note that the TIP3P files shipped with GROMACS use H-H 1.5139 A.',
         '; With -DFLEXIBLE: O-H bonds as in frcmod.tip3p and HOH angle 104.52 deg, 100 kcal/mol/rad^2.',
         '', '[ moleculetype ]', '; molname  nrexcl', 'SOL        2', '',
         '[ atoms ]', ';  id  at type    res nr  res name  at name  cg nr  charge     mass',
         '    1  OW_tip3p   1       SOL       OW       1      %.6f  %s' % (qw['O'], fmt(ps.masses['OW'])),
         '    2  HW_tip3p   1       SOL       HW1      1       %.6f  %s' % (qw['H1'], fmt(ps.masses['HW'])),
         '    3  HW_tip3p   1       SOL       HW2      1       %.6f  %s' % (qw['H2'], fmt(ps.masses['HW'])),
         '', '#ifndef FLEXIBLE', '', '[ settles ]', '; OW  funct  doh      dhh',
         '  1   1      %s  %s' % (fmt(roh / 10.0), fmt(rhh / 10.0)), '',
         '[ exclusions ]', '1  2  3', '2  1  3', '3  1  2', '', '#else', '',
         '[ bonds ]', '; i  j  funct  length   force_constant',
         '  1  2  1      %s  %s' % (fmt(roh / 10.0), fmt(2 * koh * KCAL * 100)),
         '  1  3  1      %s  %s' % (fmt(roh / 10.0), fmt(2 * koh * KCAL * 100)), '',
         '[ angles ]', '; i  j  k  funct  angle   force_constant',
         '  2  1  3  1      104.52  %s' % fmt(2 * 100.0 * KCAL), '', '#endif', '']
    open(os.path.join(outdir, 'tip3p.itp'), 'w').write('\n'.join(W))

    I = ['; Ions for TIP3P water as loaded by Amber leaprc.water.tip3p', src,
         '; Na+, K+, Cl-: Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008)',
         '; Mg2+, Ca2+:  Li, Roberts, Chakravorty & Merz, JCTC 9, 2733 (2013), 12-6 CM set', '']
    for mol, t, z, q in IONS:
        I += ['[ moleculetype ]', '; molname  nrexcl', '%-6s     1' % mol, '',
              '[ atoms ]', ';  id  at type     res nr  res name  at name  cg nr  charge     mass',
              '    1  %-11s  1       %-6s    %-6s   1      %+.6f  %s' % (t + '_tip3p', mol, mol, q, fmt(ps.masses[t])), '']
    open(os.path.join(outdir, 'ions_tip3p.itp'), 'w').write('\n'.join(I))


FORCEFIELD_ITP = """\
; AMBER Lipid21 force field for GROMACS -- standalone entry point
;
; Lipid21: Dickson, Walker & Gould, J. Chem. Theory Comput. 18, 1726 (2022)
;          doi:10.1021/acs.jctc.1c01217
; converted from lipid21.dat / lipid21.lib (AmberTools) by tools/lipid21_gmx.py
;
; Use this file for lipid/water/ion systems without a protein force field.
; To combine Lipid21 with a protein force field (amber14sb.ff or amber19sb.ff
; shipped with GROMACS), include that force field's forcefield.itp and then
; "lipid21.ff/lipid21_params.itp" instead of this file.

#define _FF_AMBER
#define _FF_AMBER_LIPID21

[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
  1       2          yes        0.5      0.83333333333333333

#include "ffnonbonded.itp"
#include "ffbonded.itp"
#include "ffwater_ions.itp"
"""

PARAMS_ITP = """\
; AMBER Lipid21 atom types and bonded parameters -- add-on entry point
;
; Include this file right after the forcefield.itp of an Amber protein force
; field that uses the standard Amber [ defaults ] (1 2 yes 0.5 0.8333), e.g.
;     #include "amber19sb.ff/forcefield.itp"
;     #include "lipid21.ff/lipid21_params.itp"
; Lipid21 atom types (cA, hL, oP, ...) do not clash with the protein types.

#define _FF_AMBER_LIPID21

#include "ffnonbonded.itp"
#include "ffbonded.itp"
"""

FORCEFIELD_DOC = 'AMBER Lipid21 lipids (Dickson, Walker & Gould, JCTC 2022) + TIP3P/ions; molecule .itp files, no pdb2gmx support\n'


def check_reversed_duplicates(ff):
    ps = ff.ps
    for table in (ps.bonds, ps.angles):
        for key, val in table.items():
            rev = key[::-1]
            if rev != key and rev in table and table[rev] != val:
                raise ValueError('%s and its reverse have different parameters' % (key,))
    for key, terms in ps.dihedrals.items():
        rev = key[::-1]
        if rev != key and rev in ps.dihedrals:
            a = sorted((t['pk'], t['phase'], t['pn'], t['idivf'], t['scee'], t['scnb']) for t in terms)
            b = sorted((t['pk'], t['phase'], t['pn'], t['idivf'], t['scee'], t['scnb']) for t in ps.dihedrals[rev])
            if a != b:
                raise ValueError('torsion %s and its reverse have different parameters' % (key,))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lib', default=os.path.join(DEFAULT_SRC, 'lipid21.lib'))
    ap.add_argument('--dat', default=os.path.join(DEFAULT_SRC, 'lipid21.dat'))
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_ff = sub.add_parser('ff', help='write the force-field parameter files and all standard lipids')
    p_ff.add_argument('-o', '--outdir', default='lipid21.ff')
    p_ff.add_argument('--amberhome', help='AmberTools installation (for TIP3P/ion parameters); '
                                          'default: $AMBERHOME or the location of tleap in $PATH')
    p_itp = sub.add_parser('itp', help='write one molecule topology')
    p_itp.add_argument('name')
    p_itp.add_argument('residues', nargs='+')
    p_itp.add_argument('-o', '--output')
    p_itp.add_argument('--keep-lib-charges', action='store_true',
                       help='do not remove non-integral net charges (sphingomyelin: +0.000003 e)')
    args = ap.parse_args(argv)

    ff = ForceField(args.lib, args.dat)
    check_reversed_duplicates(ff)
    stamp = ['AMBER Lipid21 (Dickson, Walker & Gould, J. Chem. Theory Comput. 18, 1726 (2022)) -- GROMACS port',
             'generated by tools/lipid21_gmx.py from %s and %s' % (os.path.basename(args.dat), os.path.basename(args.lib)),
             'use with lipid21.ff/forcefield.itp, or after amber14sb.ff/amber19sb.ff + lipid21.ff/lipid21_params.itp']

    if args.cmd == 'itp':
        mol = Molecule(ff, args.residues, args.name, neutralize=not args.keep_lib_charges)
        for w in mol.warnings:
            print('WARNING:', w, file=sys.stderr)
        out = args.output or args.name + '.itp'
        open(out, 'w').write(mol.itp(stamp + ['Lipid21 residues: ' + ' '.join(args.residues)]))
        print('wrote %s (%d atoms, charge %s)' % (out, len(mol.atoms), sum(a['qdec'] for a in mol.atoms)))
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    write_ffnonbonded(ff, os.path.join(args.outdir, 'ffnonbonded.itp'))
    write_ffbonded(ff, os.path.join(args.outdir, 'ffbonded.itp'))
    open(os.path.join(args.outdir, 'forcefield.itp'), 'w').write(FORCEFIELD_ITP)
    open(os.path.join(args.outdir, 'lipid21_params.itp'), 'w').write(PARAMS_ITP)
    open(os.path.join(args.outdir, 'forcefield.doc'), 'w').write(FORCEFIELD_DOC)
    amberhome = args.amberhome or find_amberhome()
    if amberhome is None:
        raise SystemExit('AmberTools not found (set AMBERHOME or use --amberhome): needed for water/ions')
    write_water_ions(args.outdir, amberhome)
    for name, resnames in standard_lipids().items():
        mol = Molecule(ff, resnames, name)
        if mol.warnings:
            raise RuntimeError('%s: %s' % (name, mol.warnings))
        open(os.path.join(args.outdir, name + '.itp'), 'w').write(
            mol.itp(stamp + ['Lipid21 residues: ' + ' '.join(resnames)]))
    print('wrote force field to', args.outdir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
