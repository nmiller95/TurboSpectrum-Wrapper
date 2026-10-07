# Regular Teff-logg-[Fe/H] grid for the pseudo-continuum polynomials (cont_suppress_param.txt in mstesci1).
# NJM / 30.09.2026, revised 07.10.2026
#
# The first pseudo-grid (Olander+25, Sect. 3.2.2) was 3000-4150 K x 50 K, logg 4.41-4.91 x 0.1, [Fe/H] -0.8..+0.5 x 0.1
# This script makes the same spacing over the validity range of the new ANN (make_random_grid_input.py):
# Teff 2800-4500 K, logg 4.5-5.5 (<= 5.0 above 3900 K, the same L-shape), [Fe/H] -1.7..+0.6.
#
# Abundances - match ANN (v2) grid, whose input columns are  Teff logg Vturb FeH C O Mg Al Si K Ca Ti Fe H:
#   * every element the ANN grid leaves out (N, Na, ...) stays at its SOLAR abundance
#   * listed elements are scaled with [Fe/H] (Magg+22, the ANN grid's solar mixture).
#   * listed alpha elements (O, Mg, Si, Ca, Ti) also follow the MARCS standard alpha law
#     (Gustafsson et al. 2008, A&A 486, 951): [a/Fe] = +0.4 for [Fe/H] <= -1, -0.4 [Fe/H] for -1..0, 0 above;
#     as per the Olander+25 pseudo-grid. --alpha solar sets [X/Fe] = 0 for all of them instead.
#   * Vturb is fixed (default 1.0 km/s, the MARCS dwarf value and the middle of the ANN range).
#
# --exclude FILE drops nodes (teff-logg-feh combos) that are known to fail in Turbospectrum: 'pseudo_grid_exclude.txt'
#
# The output has the same format as random_grid_*.txt, so the wrapper, ts_array.sh and audit_grid.py work unchanged.
# Usage (from turbowrapper/source, on slurm):
#   python make_pseudo_grid_input.py --exclude pseudo_grid_exclude.txt --outdir ../input --base-config ../input/config.txt
#   sbatch --export=ALL,CFG_PREFIX=config_pseudo_grid --array=0-13 ../slurm/ts_array.sh
# IMPORTANT: use the SAME config.txt (line lists, MARCS models, lam_start/lam_end, lam_step) as the ANN grid.
import argparse
import os

import numpy as np

from make_random_grid_input import SpectrumGridGenerator

SOLAR = SpectrumGridGenerator({'Teff': (0, 0)}).solar_abund  # Magg+22, identical to the ANN grid
ANN_ELEMENTS = ['C', 'O', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ti']  # ANN v2 grid columns (Fe is always written)
MARCS_ALPHA = {'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti'}  # Gustafsson+08


def marcs_alpha(feh):
    """MARCS 'standard' composition: [alpha/Fe] = +0.4 (<= -1), -0.4*[Fe/H] (-1..0), 0 (>= 0)."""
    return float(np.clip(-0.4 * feh, 0.0, 0.4))


def axis(lo, hi, step):
    n = int(round((hi - lo) / step))
    return np.round(lo + step * np.arange(n + 1), 6)


def node(t, g, m):
    return (int(round(t)), round(float(g), 2), round(float(m), 2))


def read_exclude(path):
    if not path:
        return set()
    return {node(*r) for r in np.atleast_2d(np.loadtxt(path, usecols=(0, 1, 2)))}


def build_grid(args):
    teffs, loggs, fehs = axis(*args.teff), axis(*args.logg), axis(*args.feh)
    exclude = read_exclude(args.exclude)
    rows, n_excl = [], 0
    for t in teffs:
        for g in loggs:
            if not args.rectangle and t > args.lshape_teff + 1e-6 and g > args.lshape_logg + 1e-6:
                continue  # outside the ANN's L-shaped Teff-logg domain
            for m in fehs:
                if node(t, g, m) in exclude:
                    n_excl += 1
                    continue
                afe = marcs_alpha(m) if args.alpha == 'marcs' else 0.0
                abund = {el: SOLAR[el] + m + (afe if el in MARCS_ALPHA else 0.0) for el in args.elements}
                abund['Fe'] = SOLAR['Fe'] + m
                rows.append((t, g, args.vturb, m, abund))
    if exclude and n_excl != len(exclude):
        print(f"Note: {len(exclude) - n_excl} of {len(exclude)} excluded nodes were not in the grid anyway")
    return rows, n_excl


def write_rows(rows, elements, path):
    cols = list(elements) + ['Fe']
    with open(path, 'w') as f:
        f.write(" ".join(["Teff", "logg", "Vturb", "FeH"] + cols + ["H"]) + "\n")
        for t, g, v, m, ab in rows:
            f.write(" ".join([f"{t:.0f}", f"{g:.2f}", f"{v:.2f}", f"{m:.3f}"]
                             + [f"{ab[c]:.4f}" for c in cols] + ["12.0"]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regular pseudo-continuum grid (see header comments)")
    ap.add_argument("--teff", type=float, nargs=3, default=(2800, 4500, 50), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--logg", type=float, nargs=3, default=(4.5, 5.5, 0.1), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--feh", type=float, nargs=3, default=(-1.7, 0.6, 0.1), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--vturb", type=float, default=1.0)
    ap.add_argument("--elements", nargs="+", default=ANN_ELEMENTS,
                    help="elements written besides Fe; must be the ANN grid's set (default %(default)s)")
    ap.add_argument("--alpha", choices=["marcs", "solar"], default="marcs")
    ap.add_argument("--exclude", default=None, help="file with teff logg feh of nodes to leave out")
    ap.add_argument("--rectangle", action="store_true",
                    help="fill the full Teff-logg box. DON'T for MARCS: there are no MARCS models at logg > 5.0 above "
                         "3900 K, which is why the grid is L-shaped. The convex-hull check would drop part of the "
                         "notch and interpolate atmospheres across the gap for the rest")
    ap.add_argument("--lshape-teff", type=float, default=3900)
    ap.add_argument("--lshape-logg", type=float, default=5.0)
    ap.add_argument("--batches", type=int, default=14)
    ap.add_argument("--outdir", default="../input")
    ap.add_argument("--stem", default="pseudo_grid")
    ap.add_argument("--base-config", default=None)
    args = ap.parse_args()
    bad = [e for e in args.elements if e not in SOLAR or e == 'Fe']
    if bad:
        raise SystemExit(f"Unknown or invalid elements {bad}; Fe is always written")

    rows, n_excl = build_grid(args)
    os.makedirs(args.outdir, exist_ok=True)
    write_rows(rows, args.elements, os.path.join(args.outdir, f"{args.stem}_all.txt"))
    # Interleaved split: every batch spans the whole grid, so batches take similar time
    for k in range(args.batches):
        idx = list(range(k, len(rows), args.batches))
        tag = f"b{k:02d}"
        write_rows([rows[i] for i in idx], args.elements, os.path.join(args.outdir, f"{args.stem}_{tag}.txt"))
        with open(os.path.join(args.outdir, f"{args.stem}_{tag}.idx"), "w") as f:
            f.write("\n".join(map(str, idx)) + "\n")
        if args.base_config:
            SpectrumGridGenerator._write_batch_config(
                args.base_config, os.path.join(args.outdir, f"config_{args.stem}_{tag}.txt"),
                f"{args.stem}_{tag}.txt", f"_{args.stem}_{tag}")
    print(f"{len(rows)} models ({len(set((r[0], r[1]) for r in rows))} Teff-logg nodes x "
          f"{len(set(r[3] for r in rows))} [Fe/H]; {n_excl} excluded) in {args.batches} batches of "
          f"~{len(rows) // args.batches} -> {args.outdir}/{args.stem}_*.txt")
