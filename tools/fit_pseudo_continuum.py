"""
Fit the pseudo-continuum polynomials for mstesci1 (cont_suppress_param.txt) from a regular Turbospectrum grid,
one batch folder at a time, then merge the batches into the final file.
NJM / 30.09.2026, batch mode 07.10.2026

Method (Olander+25, Sect. 3.2.2): for every (Teff, logg, [Fe/H]) model, find the highest flux peaks of the
continuum-normalised synthetic spectrum, discard peaks more than 3 sigma from the mean of the peaks, and fit a
second-degree polynomial in wavelength (Å) to what is left. mstesci1 multiplies the (FGK-)normalised observed
spectrum by this polynomial, i.e. the polynomial is the model's pseudo-continuum level (~0.8 at 3000 K in the H band).

Implementation choices (the original fitting script is not in the repo):
  * spectra are first broadened to APOGEE resolution (R = 22500), as for the ANN training set. Tested on spec_29
    (3422 K, 4.80, -0.27): with broadening the fit is within ~0.01 of the current grid, without it ~0.04 too high,
    so the original grid was evidently fitted to broadened spectra;
  * "highest peaks" = the maximum flux in each window of --window Å (default 10 Å, ~200 peaks over 15000-17000 Å);
  * 3-sigma clipping about the mean is iterated until nothing more is rejected (--clip-iter 1 = single pass, as
    literally described in the paper);
  * the fit range defaults to 15000-17000 Å (prep_training_data.py defaults, covering the ANN wavelength array).

Output: tab-separated, np.savetxt default precision ('%.18e'), columns
    Teff  logg  [Fe/H]  c2  c1  c0          (np.poly1d order, as read by mstesci1 spectroscopy.py)
i.e. identical in format to the current cont_suppress_param.txt, plus a .qa.csv with per-model diagnostics
(peaks found/used, rms about the fit, polynomial at 15000/16000/17000 Å, source file).

Usage (numpy only; run from turbowrapper/tools):

  1. Fit each batch folder separately (each writes parts/part_<folder>.txt + .qa.csv [+ .failed.txt]):
       python fit_pseudo_continuum.py fit ROOT --pattern "spectra*_pseudo*" --index 3      # 4th folder (sorted)
       python fit_pseudo_continuum.py fit ROOT --pattern "spectra*_pseudo*" --batch b03    # folder name contains b03
       python fit_pseudo_continuum.py fit ROOT --pattern "spectra*_pseudo*"                # all folders, one by one
       python fit_pseudo_continuum.py list ROOT --pattern "spectra*_pseudo*"               # show index -> folder
     On Pelle as a job array (one task per folder): see ../slurm/pseudo_fit_array.sh
     Folders that already have a part file are skipped unless --overwrite.

  2. Merge the parts into the final file (seconds; fine on a login node):
       python fit_pseudo_continuum.py merge parts --out cont_suppress_param_v2.txt --grid ../input/pseudo_grid_all.txt

  3. Optional diagnostic plots: plot_pseudo_continuum.py
"""
import argparse
import glob
import os
import re
from multiprocessing import Pool

import numpy as np

HDR = re.compile(r"^#\s*(teff|logg|vturb|feh)\s*=\s*([-+\d.eE]+)")
QA_NAMES = ["n_peaks", "n_used", "rms", "poly_15000", "poly_16000", "poly_17000", "vturb"]
DEFAULTS = dict(lmin=15000.0, lmax=17000.0, resolution=22500.0, window=10.0, nsigma=3.0, clip_iter=10)


# ----------------------------------------------------------------------------------------------- the method
def read_spectrum(path):
    params = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = HDR.match(line)
            if m:
                params[m.group(1)] = float(m.group(2))
    wave, flux = np.loadtxt(path, comments="#", usecols=(0, 1), unpack=True)
    return params, wave, flux


def read_header(path):
    params = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = HDR.match(line)
            if m:
                params[m.group(1)] = float(m.group(2))
    return params


def broaden(wave, flux, resolution):
    """Gaussian instrumental broadening at constant R (convolution on a uniform ln-lambda grid)."""
    if not resolution:
        return wave, flux
    dlnl = np.median(np.diff(wave)) / np.median(wave)
    lnl = np.arange(np.log(wave[0]), np.log(wave[-1]), dlnl)
    f = np.interp(lnl, np.log(wave), flux)
    sigma = 1.0 / (resolution * 2.0 * np.sqrt(2.0 * np.log(2.0))) / dlnl  # in pixels
    half = int(np.ceil(5 * sigma))
    x = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (x / sigma) ** 2)
    kern /= kern.sum()
    fpad = np.concatenate([np.full(half, f[0]), f, np.full(half, f[-1])])
    fc = np.convolve(fpad, kern, mode="valid")
    return np.exp(lnl), fc


def highest_peaks(wave, flux, window):
    edges = np.arange(wave[0], wave[-1] + window, window)
    idx = np.digitize(wave, edges)
    pw, pf = [], []
    for k in np.unique(idx):
        sel = np.where(idx == k)[0]
        if len(sel) < 3:
            continue
        j = sel[np.argmax(flux[sel])]
        pw.append(wave[j])
        pf.append(flux[j])
    return np.array(pw), np.array(pf)


def fit_spectrum(wave, flux, cfg):
    """Return everything needed for fitting and plotting: broadened spectrum, peaks, clip mask, coefficients."""
    m = (wave >= cfg["lmin"] - 5) & (wave <= cfg["lmax"] + 5)  # small pad for the convolution edges
    if m.sum() < 100 or np.any(~np.isfinite(flux[m])):
        raise ValueError("too few points or NaN flux")
    w, f = broaden(wave[m], flux[m], cfg["resolution"])
    keep = (w >= cfg["lmin"]) & (w <= cfg["lmax"])
    w, f = w[keep], f[keep]
    pw, pf = highest_peaks(w, f, cfg["window"])
    ok = np.ones(len(pf), bool)
    for _ in range(cfg["clip_iter"]):
        mu, sd = pf[ok].mean(), pf[ok].std()
        new = np.abs(pf - mu) <= cfg["nsigma"] * sd
        if np.array_equal(new, ok):
            break
        ok = new
    coef = np.polyfit(pw[ok], pf[ok], 2)
    return dict(wave=w, flux=f, peak_wave=pw, peak_flux=pf, used=ok, coef=coef)


def node_key(teff, logg, feh):
    """Grid node as written in the input file (Teff %.0f, logg %.2f, [Fe/H] %.3f) - avoids float noise."""
    return (float(round(teff)), round(float(logg), 2), round(float(feh), 3))


def fit_one(job):
    path, cfg = job
    try:
        p, wave, flux = read_spectrum(path)
        if not all(k in p for k in ("teff", "logg", "feh")):
            return path, None, "header lacks teff/logg/feh"
        r = fit_spectrum(wave, flux, cfg)
    except (OSError, ValueError, np.linalg.LinAlgError) as e:
        return path, None, str(e)
    coef = r["coef"]
    resid = r["peak_flux"][r["used"]] - np.polyval(coef, r["peak_wave"][r["used"]])
    qa = [len(r["peak_flux"]), int(r["used"].sum()), float(resid.std()),
          *(float(np.polyval(coef, l)) for l in (15000.0, 16000.0, 17000.0)), p.get("vturb", np.nan)]
    return path, (*node_key(p["teff"], p["logg"], p["feh"]), *coef), qa


# ----------------------------------------------------------------------------------------------- I/O helpers
def batch_folders(root, pattern):
    return sorted(d for d in glob.glob(os.path.join(root, pattern)) if os.path.isdir(d))


def spectra_in(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*"))
                  if os.path.isfile(f) and not os.path.basename(f).startswith("."))


def write_table(rows, path):
    """rows: dict node -> (teff, logg, feh, c2, c1, c0). Same format as cont_suppress_param.txt."""
    table = np.array([rows[k] for k in sorted(rows)])
    np.savetxt(path, table, delimiter="\t")
    return len(table)


def write_qa(qa_rows, path):
    with open(path, "w") as f:
        f.write(",".join(["teff", "logg", "feh"] + QA_NAMES + ["file"]) + "\n")
        for r in sorted(qa_rows):
            f.write(",".join(str(x) for x in r) + "\n")


def read_qa(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            v = line.rstrip("\n").split(",")
            out.append((*node_key(*map(float, v[:3])), *v[3:]))
    return out


# ----------------------------------------------------------------------------------------------- commands
def fit_folder(folder, root, a, cfg):
    name = os.path.basename(os.path.normpath(folder))
    stem = os.path.join(a.outdir, f"part_{name}")
    if os.path.exists(stem + ".txt") and not a.overwrite:
        print(f"{name}: {stem}.txt exists, skipping (use --overwrite)")
        return
    files = spectra_in(folder)
    if not files:
        print(f"{name}: no spectra, skipping")
        return
    print(f"{name}: fitting {len(files)} spectra on {a.ncpu} CPUs", flush=True)
    with Pool(a.ncpu) as pool:
        res = pool.map(fit_one, [(f, cfg) for f in files], chunksize=4)

    rows, qa_rows, bad = {}, [], []
    for path, row, qa in res:
        if row is None:
            bad.append((path, qa))
            continue
        key = row[:3]
        if key in rows:
            print(f"  WARNING: duplicate node {key}: {path} (keeping the first)")
            continue
        rows[key] = row
        qa_rows.append((*key, *qa, os.path.relpath(path, root)))
    if not rows:
        print(f"{name}: nothing fitted; no part file written")
    else:
        # write the .txt last, so its existence means the part is complete
        write_qa(qa_rows, stem + ".qa.csv")
        n = write_table(rows, stem + ".txt")
        print(f"{name}: wrote {n} polynomials to {stem}.txt")
    if bad:
        with open(stem + ".failed.txt", "w") as f:
            f.writelines(f"{p}\t{why}\n" for p, why in bad)
        print(f"{name}: {len(bad)} spectra failed -> {stem}.failed.txt")


def cmd_fit(a):
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    folders = batch_folders(a.root, a.pattern)
    if not folders:
        raise SystemExit(f"No folders match {os.path.join(a.root, a.pattern)}")
    if a.index is not None:
        if not 0 <= a.index < len(folders):
            raise SystemExit(f"--index {a.index} out of range: {len(folders)} folders (0-{len(folders) - 1})")
        folders = [folders[a.index]]
    elif a.batch:
        folders = [d for d in folders if re.search(rf"(?<![A-Za-z0-9]){re.escape(a.batch)}(?![0-9])",
                                                   os.path.basename(d))]
        if not folders:
            raise SystemExit(f"--batch {a.batch} matched no folder")
        if len(folders) > 1:  # e.g. a batch re-run on another day -> spectra-<date>_pseudo_grid_b03 twice
            print(f"--batch {a.batch} matched {len(folders)} folders; fitting each (merge removes duplicates)")
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "fit_settings.txt"), "w") as f:  # provenance, overwritten each run
        f.write("".join(f"{k} = {v}\n" for k, v in cfg.items()))
    for d in folders:
        fit_folder(d, a.root, a, cfg)


def cmd_list(a):
    folders = batch_folders(a.root, a.pattern)
    for i, d in enumerate(folders):
        print(f"{i:3d}  {os.path.basename(d)}  ({len(spectra_in(d))} files)")
    print(f"{len(folders)} folders -> sbatch --array=0-{len(folders) - 1} ...")


def cmd_merge(a):
    parts = sorted(glob.glob(os.path.join(a.parts, "part_*.txt")))
    parts = [p for p in parts if not p.endswith((".failed.txt",))]
    if not parts:
        raise SystemExit(f"No part_*.txt in {a.parts}")
    rows, qa_rows, n_failed = {}, [], 0
    for p in parts:
        t = np.atleast_2d(np.loadtxt(p))
        for r in t:
            key = node_key(*r[:3])
            if key in rows:
                print(f"WARNING: node {key} appears in more than one part ({p}); keeping the first")
                continue
            rows[key] = (*key, *r[3:])
        qa = os.path.splitext(p)[0] + ".qa.csv"
        if os.path.exists(qa):
            qa_rows += read_qa(qa)
        fl = os.path.splitext(p)[0] + ".failed.txt"
        if os.path.exists(fl):
            n_failed += sum(1 for _ in open(fl))
    n = write_table(rows, a.out)
    write_qa(qa_rows, os.path.splitext(a.out)[0] + ".qa.csv")
    print(f"Merged {len(parts)} parts: {n} polynomials -> {a.out} ({n_failed} spectra failed to fit)")

    if qa_rows:
        rms = np.array([float(r[5]) for r in qa_rows])
        used = np.array([float(r[4]) / max(float(r[3]), 1) for r in qa_rows])
        print(f"rms about fit: median {np.median(rms):.4f}, max {rms.max():.4f}; "
              f"fraction of peaks kept: median {np.median(used):.3f}, min {used.min():.3f}")

    if a.grid:
        g = np.loadtxt(a.grid, skiprows=1, usecols=(0, 1, 3))
        want = {node_key(*r) for r in g}
        missing = sorted(want - set(rows))
        extra = sorted(set(rows) - want)
        print(f"{len(missing)} of {len(want)} requested nodes missing" + (f"; {len(extra)} not in grid" if extra else ""))
        if missing:
            path = os.path.splitext(a.out)[0] + ".missing.txt"
            np.savetxt(path, np.array(missing), fmt="%.0f %.2f %.3f", header="teff logg feh")
            print(f"  e.g. {missing[:5]} -> {path}; mstesci1 falls back to the nearest node")


def main():
    ap = argparse.ArgumentParser(description="Fit pseudo-continuum polynomials (cont_suppress_param format)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit one batch folder (or all, one after another)")
    f.add_argument("root", help="TS wrapper output/ directory containing the batch folders")
    f.add_argument("--pattern", default="spectra*_pseudo*", help="glob for batch folders under root")
    sel = f.add_mutually_exclusive_group()
    sel.add_argument("--index", type=int, default=None, help="fit only folder number N (0-based, sorted; see 'list')")
    sel.add_argument("--batch", default=None, help="fit only the folder whose name contains this tag, e.g. b03")
    f.add_argument("--outdir", default="parts", help="where the part files go")
    f.add_argument("--overwrite", action="store_true")
    f.add_argument("--lmin", type=float, default=DEFAULTS["lmin"])
    f.add_argument("--lmax", type=float, default=DEFAULTS["lmax"])
    f.add_argument("--resolution", type=float, default=DEFAULTS["resolution"], help="0 = no broadening")
    f.add_argument("--window", type=float, default=DEFAULTS["window"], help="Å per peak window")
    f.add_argument("--nsigma", type=float, default=DEFAULTS["nsigma"])
    f.add_argument("--clip-iter", type=int, default=DEFAULTS["clip_iter"])
    f.add_argument("--ncpu", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    f.set_defaults(func=cmd_fit)

    l = sub.add_parser("list", help="list the batch folders with their index")
    l.add_argument("root")
    l.add_argument("--pattern", default="spectra*_pseudo*")
    l.set_defaults(func=cmd_list)

    m = sub.add_parser("merge", help="combine part files into the final cont_suppress_param file")
    m.add_argument("parts", nargs="?", default="parts")
    m.add_argument("--out", default="cont_suppress_param_v2.txt")
    m.add_argument("--grid", default=None, help="pseudo_grid_all.txt, to report nodes that are missing")
    m.set_defaults(func=cmd_merge)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
