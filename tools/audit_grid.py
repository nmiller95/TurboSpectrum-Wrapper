"""
Audit a batched TurboSpectrum-Wrapper grid for silent non-uniformity.

    python audit_grid.py ../output --pattern "spectra-*_big" --png audit.png

Checks, per batch folder:
  1. How many of the requested points (rows in random_grid.txt) produced a spectrum.
     Missing ones were dropped silently: outside the MARCS hull, or babsma/bsyn failed.
  2. Whether the parameters written in each spectrum's header match the row in that
     folder's random_grid.txt (catches a random_grid.txt copied from the wrong batch).
  3. Wavelength sampling of the synthetic spectra (is it fine enough to convolve later?).
  4. Wall-clock span of each batch from file mtimes, and whether batches overlapped in time.
     Overlapping batches with the same jobID share temp dirs and opacity file names.

Then, over the combined grid:
  5. 1-D histograms of every label against the uniform expectation (chi^2 per label).
  6. Occupancy of each Teff x logg tile, and where the dropped points sit.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HDR_PARAM = re.compile(r"^#\s*(teff|logg|vturb|feh)\s*=\s*([-+\d.eE]+)")
HDR_ABUND = re.compile(r"^#\s*A\((\w+)\)\s*=\s*([-+\d.eE]+)")
SPEC_NAME = re.compile(r"^spec_(\d+)_(N?LTE)$", re.IGNORECASE)


def read_header(path, max_lines=80):
    """Return (params dict, first two wavelength values) from a TS-wrapper spectrum file."""
    params, waves = {}, []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                m = HDR_PARAM.match(line)
                if m:
                    params[m.group(1).lower()] = float(m.group(2))
                    continue
                m = HDR_ABUND.match(line)
                if m:
                    params[m.group(1)] = float(m.group(2))
                continue
            parts = line.split()
            if parts:
                waves.append(float(parts[0]))
            if len(waves) >= 3 or i > max_lines + 3:
                break
    return params, waves


def audit_folder(folder, grid_file):
    grid = pd.read_csv(os.path.join(folder, grid_file), sep=r"\s+")
    colmap = {"Teff": "teff", "logg": "logg", "Vturb": "vturb", "FeH": "feh"}
    files = {}
    for fn in os.listdir(folder):
        m = SPEC_NAME.match(fn)
        if m:
            files[int(m.group(1))] = os.path.join(folder, fn)

    mismatches, steps, mtimes = [], [], []
    for idx, path in files.items():
        mtimes.append(os.path.getmtime(path))
        if idx >= len(grid):
            mismatches.append((idx, "index beyond random_grid.txt"))
            continue
        hdr, waves = read_header(path)
        if len(waves) >= 2:
            steps.append(waves[1] - waves[0])
        row = grid.iloc[idx]
        for gcol, hkey in colmap.items():
            if hkey in hdr and gcol in row and abs(hdr[hkey] - row[gcol]) > 1e-3 * max(1, abs(row[gcol])):
                mismatches.append((idx, f"{gcol}: header {hdr[hkey]} vs grid {row[gcol]}"))
                break
        else:
            for el in [c for c in grid.columns if c not in colmap and c != "H"]:
                if el in hdr and abs(hdr[el] - row[el]) > 2e-3:
                    mismatches.append((idx, f"A({el}): header {hdr[el]} vs grid {row[el]}"))
                    break

    present = np.zeros(len(grid), bool)
    present[[i for i in files if i < len(grid)]] = True
    return {
        "folder": os.path.basename(folder.rstrip("/")),
        "grid": grid,
        "present": present,
        "n_requested": len(grid),
        "n_spectra": int(present.sum()),
        "mismatches": mismatches,
        "step": float(np.median(steps)) if steps else np.nan,
        "t0": min(mtimes) if mtimes else np.nan,
        "t1": max(mtimes) if mtimes else np.nan,
    }


def uniform_chi2(values, lo, hi, nbins=10):
    counts, _ = np.histogram(values, bins=nbins, range=(lo, hi))
    expected = len(values) / nbins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return chi2, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="output/ directory containing the spectra-* batch folders")
    ap.add_argument("--pattern", default="spectra-*", help="glob for batch folders (default: spectra-*)")
    ap.add_argument("--grid-file", default="random_grid.txt")
    ap.add_argument("--png", default=None, help="write diagnostic plots to this PNG")
    ap.add_argument("--dropped-csv", default=None, help="write the dropped (missing) points to this CSV")
    args = ap.parse_args()

    folders = sorted(f for f in glob.glob(os.path.join(args.root, args.pattern)) if os.path.isdir(f))
    folders = [f for f in folders if os.path.isfile(os.path.join(f, args.grid_file))]
    if not folders:
        sys.exit(f"No folders matching {args.pattern} with a {args.grid_file} under {args.root}")

    results = [audit_folder(f, args.grid_file) for f in folders]

    print(f"\n{'folder':45s} {'req':>5s} {'got':>5s} {'drop%':>6s} {'mismatch':>8s} {'dλ[Å]':>7s} {'hours':>6s}")
    for r in results:
        drop = 100 * (1 - r["n_spectra"] / r["n_requested"])
        hrs = (r["t1"] - r["t0"]) / 3600 if np.isfinite(r["t0"]) else np.nan
        print(f"{r['folder']:45s} {r['n_requested']:5d} {r['n_spectra']:5d} {drop:6.1f} "
              f"{len(r['mismatches']):8d} {r['step']:7.4f} {hrs:6.1f}")
        for idx, msg in r["mismatches"][:3]:
            print(f"    spec_{idx}: {msg}")

    # --- time overlap between batches ------------------------------------------------
    spans = sorted((r["t0"], r["t1"], r["folder"]) for r in results if np.isfinite(r["t0"]))
    overlaps = [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    print("\nBatches whose file-write windows overlap in time (possible shared temp/opacity files):")
    print("  none" if not overlaps else "\n".join(f"  {a}  <->  {b}" for a, b in overlaps))

    # --- combined grid ---------------------------------------------------------------
    allgrid = pd.concat([r["grid"].assign(_present=r["present"], _folder=r["folder"]) for r in results],
                        ignore_index=True)
    kept = allgrid[allgrid["_present"]]
    dropped = allgrid[~allgrid["_present"]]
    print(f"\nTotal requested {len(allgrid)}, spectra present {len(kept)}, dropped {len(dropped)} "
          f"({100 * len(dropped) / len(allgrid):.1f}%)")
    print(f"Median wavelength step across batches: {np.nanmedian([r['step'] for r in results]):.4f} Å "
          f"(R=22 500 FWHM at 16 000 Å is 0.71 Å; you want ≲ FWHM/5 before convolving)")

    labels = [c for c in allgrid.columns if not c.startswith("_") and c != "H"]
    # express abundances as [X/Fe] so their expected distribution is uniform
    solar = {'C': 8.56, 'N': 7.98, 'O': 8.77, 'Mg': 7.55, 'Al': 6.43, 'Si': 7.59, 'K': 5.14,
             'Ca': 6.37, 'Ti': 4.94, 'Fe': 7.50, 'Na': 6.29, 'V': 3.95, 'Cr': 5.74, 'Mn': 5.52, 'Ni': 6.24}
    xfe = {}
    for el in labels:
        if el in solar:
            xfe[el] = kept[el] - solar[el] - kept["FeH"]

    print("\n1-D uniformity of the spectra you actually have (10 bins, chi^2 with 9 dof; >21.7 is p<0.01):")
    print("  NB Teff and logg are only expected to be uniform within the rectangular part of the domain.")
    for lab in labels:
        vals = xfe[lab] if lab in xfe else kept[lab]
        if lab == "Fe":
            spread = float(np.ptp(vals))
            print(f"  {'[Fe/Fe]':8s} spread {spread:.4f} dex  -> Fe is tied to [Fe/H], not a free label")
            continue
        lo, hi = float(allgrid[lab].min()), float(allgrid[lab].max())
        if lab in xfe:
            lo, hi = float(np.floor(vals.min() * 10) / 10), float(np.ceil(vals.max() * 10) / 10)
        chi2, counts = uniform_chi2(vals, lo, hi)
        name = f"[{lab}/Fe]" if lab in xfe else lab
        flag = "  <-- non-uniform" if chi2 > 21.7 else ""
        print(f"  {name:8s} [{lo:8.3f},{hi:8.3f}] chi2={chi2:7.1f} counts={counts.tolist()}{flag}")

    if len(dropped):
        print("\nWhere the dropped points are (median and range of each label, dropped vs kept):")
        for lab in ["Teff", "logg", "FeH", "Vturb"]:
            if lab in allgrid:
                print(f"  {lab:6s} dropped median {dropped[lab].median():8.3f}  kept median {kept[lab].median():8.3f}"
                      f"   dropped range [{dropped[lab].min():.3f}, {dropped[lab].max():.3f}]")
        if args.dropped_csv:
            dropped.to_csv(args.dropped_csv, index=False)
            print(f"  written to {args.dropped_csv}")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(14, 8))
        ax[0, 0].scatter(kept["Teff"], kept["logg"], s=0.3, c="k", label="kept")
        if len(dropped):
            ax[0, 0].scatter(dropped["Teff"], dropped["logg"], s=2, c="r", label="dropped")
        ax[0, 0].set(xlabel="Teff", ylabel="logg", title="Teff–logg (red = no spectrum)")
        ax[0, 0].invert_xaxis(); ax[0, 0].invert_yaxis(); ax[0, 0].legend(markerscale=5)
        ax[0, 1].scatter(kept["FeH"], kept["Vturb"], s=0.3, c="k")
        if len(dropped):
            ax[0, 1].scatter(dropped["FeH"], dropped["Vturb"], s=2, c="r")
        ax[0, 1].set(xlabel="[Fe/H]", ylabel="Vturb", title="[Fe/H]–Vturb")
        for a, lab in zip([ax[0, 2], ax[1, 0], ax[1, 1]], ["Teff", "FeH", "Vturb"]):
            a.hist(kept[lab], bins=40, histtype="step", color="k", label="kept")
            a.hist(allgrid[lab], bins=40, histtype="step", color="0.6", ls="--", label="requested")
            a.set(xlabel=lab, title=f"{lab} histogram"); a.legend()
        for el, v in xfe.items():
            if el != "Fe":
                ax[1, 2].hist(v, bins=30, histtype="step", label=f"[{el}/Fe]")
        ax[1, 2].set(xlabel="[X/Fe]", title="abundance ratios (should be flat)"); ax[1, 2].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)
        print(f"\nPlots written to {args.png}")


if __name__ == "__main__":
    main()
