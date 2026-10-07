"""
Diagnostic plots for the pseudo-continuum polynomials made by fit_pseudo_continuum.py.
NJM / 07.10.2026

Makes two figures:
  <prefix>_spectra.png   a representative sample of grid nodes (default 6 Teff columns x 5 [Fe/H] rows, solar
                         included, at logg ~4.8; --n-teff / --n-feh to change):
                         broadened spectrum, peaks used / clipped, the polynomial re-fitted now, the polynomial
                         stored in the file (they should lie on top of each other) and, where the node is inside
                         the old grid, the current mstesci1 polynomial;
  <prefix>_overview.png  the stored polynomial at 15000 / 16000 / 17000 Å vs Teff, one line per [Fe/H], for a few
                         logg values, plus the rms about the fit. [Fe/H] = 0 (--compare-feh) is drawn bold in
                         black, with the old grid at the same [Fe/H] in red on top for direct comparison. Needs only the parameter file: steps, kinks or
                         outliers here are what to look for before swapping the file into mstesci1.

Usage (from turbowrapper/tools, after `fit_pseudo_continuum.py merge`):
    python plot_pseudo_continuum.py ROOT cont_suppress_param_v2.txt \
        --old ../../mstesci1/mstesci1_m/Input_data/spectroscopy_model_data/linemasks_continuum/cont_suppress_param.txt
  Choose the sample yourself instead:
    python plot_pseudo_continuum.py ROOT cont_suppress_param_v2.txt --nodes 3000,5.0,-1.0 3500,4.8,0.0 4200,4.6,0.3
  Works on a single part file too (parts/part_<folder>.txt), e.g. to check one batch before the others finish.
ROOT is the wrapper output/ directory; spectra are located through the .qa.csv written next to the parameter file
(falls back to scanning the headers under ROOT/--pattern). Use the same fit options as for the fit itself if you
changed any (--window, --resolution, ...); the defaults are the same.
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fit_pseudo_continuum as F  # noqa: E402

C_FIT, C_FILE, C_OLD, C_CLIP = "#1f5fa8", "#e08a1e", "#b02a2a", "#b02a2a"


def load_params(path):
    t = np.atleast_2d(np.loadtxt(path))
    return {F.node_key(*r[:3]): r[3:6] for r in t}


def locate_spectra(param_path, root, pattern):
    qa = os.path.splitext(param_path)[0] + ".qa.csv"
    where = {}
    if os.path.exists(qa):
        for r in F.read_qa(qa):
            where[r[:3]] = os.path.join(root, r[-1])
    if not where or not all(os.path.exists(p) for p in list(where.values())[:5]):
        print("Scanning spectrum headers under", os.path.join(root, pattern))
        for d in F.batch_folders(root, pattern):
            for f in F.spectra_in(d):
                h = F.read_header(f)
                if all(k in h for k in ("teff", "logg", "feh")):
                    where.setdefault(F.node_key(h["teff"], h["logg"], h["feh"]), f)
    return where


def nearest(values, target):
    values = np.asarray(sorted(set(values)))
    return float(values[np.argmin(np.abs(values - target))])


def representative_nodes(nodes, n_teff, n_feh, logg_target):
    """Teff x [Fe/H], each spread evenly over its range (solar always included), at the logg closest to logg_target.
    Returns the nodes row by row ([Fe/H]) and the number of columns (Teff)."""
    teffs = sorted({k[0] for k in nodes})
    fehs = sorted({k[2] for k in nodes})
    t_sel = sorted({nearest(teffs, t) for t in np.linspace(teffs[0], teffs[-1], n_teff)})
    m_sel = sorted({nearest(fehs, m) for m in np.linspace(fehs[0], fehs[-1], n_feh)} | {nearest(fehs, 0.0)})
    out = []
    for m in m_sel:
        for t in t_sel:
            cand = [k for k in nodes if k[0] == t and k[2] == m]
            if not cand:  # node failed in TS: take the nearest existing one in this Teff column
                cand = [k for k in nodes if k[0] == t] or list(nodes)
                cand = [min(cand, key=lambda k: abs(k[2] - m))]
            out.append(min(cand, key=lambda k: abs(k[1] - logg_target)))
    return out, len(t_sel)


def old_poly(old, key, tol=(25, 0.06, 0.05)):
    """Old-grid polynomial at the nearest old node, if that node is within half a step (else None)."""
    if not old:
        return None, None
    k = min(old, key=lambda o: abs(o[0] - key[0]) / 50 + abs(o[1] - key[1]) / 0.1 + abs(o[2] - key[2]) / 0.1)
    if all(abs(a - b) <= t for a, b, t in zip(k, key, tol)):
        return old[k], k
    return None, None


def plot_spectra(sample, ncol, params, where, old, cfg, out):
    n = len(sample)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.7 * nrow), sharex=True, squeeze=False)
    lam = np.linspace(cfg["lmin"], cfg["lmax"], 300)
    for ax, key in zip(axes.flat, sample):
        title = f"{key[0]:.0f} K   logg {key[1]:.2f}   [Fe/H] {key[2]:+.2f}"
        path = where.get(key)
        if path is None or not os.path.exists(path):
            ax.text(0.5, 0.5, "spectrum not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue
        _, wave, flux = F.read_spectrum(path)
        r = F.fit_spectrum(wave, flux, cfg)
        u = r["used"]
        rms = np.std(r["peak_flux"][u] - np.polyval(r["coef"], r["peak_wave"][u]))
        ax.plot(r["wave"], r["flux"], lw=0.3, c="0.72", zorder=1)
        ax.plot(r["peak_wave"][u], r["peak_flux"][u], ".", c="k", ms=3, zorder=3)
        if (~u).any():
            ax.plot(r["peak_wave"][~u], r["peak_flux"][~u], "x", c=C_CLIP, ms=5, mew=1.2, zorder=4)
        ax.plot(lam, np.polyval(r["coef"], lam), c=C_FIT, lw=2, zorder=5)
        ax.plot(lam, np.polyval(params[key], lam), c=C_FILE, lw=1.4, ls="--", zorder=6)
        oc, ok = old_poly(old, key)
        if oc is not None:
            ax.plot(lam, np.polyval(oc, lam), c=C_OLD, lw=1.6, ls=":", zorder=6)
        top = np.percentile(r["peak_flux"][u], 99)
        ax.set_ylim(top - 0.35, min(top + 0.06, 1.08))
        ax.set_title(title, fontsize=9)
        ax.text(0.02, 0.04, f"rms {rms:.4f}   kept {u.sum()}/{len(u)}"
                + (f"\nold node {ok[0]:.0f}/{ok[1]:.2f}/{ok[2]:+.1f}" if oc is not None else ""),
                transform=ax.transAxes, fontsize=7, color="0.3", va="bottom")
        ax.tick_params(labelsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength (Å)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalised flux", fontsize=9)
    handles = [plt.Line2D([], [], c="0.72", lw=1, label=f"spectrum, R = {cfg['resolution']:.0f}"),
               plt.Line2D([], [], c="k", marker=".", ls="", label=f"highest peak per {cfg['window']:g} Å"),
               plt.Line2D([], [], c=C_CLIP, marker="x", ls="", label=f"clipped (> {cfg['nsigma']:g}σ)"),
               plt.Line2D([], [], c=C_FIT, lw=2, label="fit (recomputed)"),
               plt.Line2D([], [], c=C_FILE, lw=1.4, ls="--", label="stored in file"),
               plt.Line2D([], [], c=C_OLD, lw=1.6, ls=":", label="current mstesci1 grid")]
    fig.legend(handles=handles if old else handles[:-1], loc="upper center", ncol=6, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


C_REF = "#111111"


def plot_overview(params, qa_path, old, logg_targets, compare_feh, out):
    nodes = list(params)
    loggs = sorted({nearest([k[1] for k in nodes], g) for g in logg_targets})
    fehs = sorted({k[2] for k in nodes})
    ref = nearest(fehs, compare_feh)  # this [Fe/H] is drawn bold, and the old grid at the same [Fe/H]
    cmap = plt.get_cmap("Blues")
    colour = {m: cmap(0.25 + 0.6 * i / max(len(fehs) - 1, 1)) for i, m in enumerate(fehs)}
    rms = {}
    if os.path.exists(qa_path):
        for r in F.read_qa(qa_path):
            rms[r[:3]] = float(r[5])
    lams = (15000.0, 16000.0, 17000.0)
    nrow = len(lams) + (1 if rms else 0)
    fig, axes = plt.subplots(nrow, len(loggs), figsize=(4.0 * len(loggs), 2.4 * nrow), sharex=True,
                             sharey="row", squeeze=False)
    fig.subplots_adjust(top=1 - 0.75 / fig.get_figheight())  # room for the legend; before the colour bar
    for j, g in enumerate(loggs):
        for m in fehs:
            ks = sorted(k for k in nodes if k[1] == g and k[2] == m)
            if not ks:
                continue
            t = [k[0] for k in ks]
            sty = (dict(c=C_REF, lw=2.2, marker="o", ms=3, zorder=6) if m == ref else
                   dict(c=colour[m], lw=0.8, marker=".", ms=2, alpha=0.8, zorder=2))
            for i, lam in enumerate(lams):
                axes[i, j].plot(t, [np.polyval(params[k], lam) for k in ks], **sty)
            if rms:
                axes[-1, j].plot(t, [rms.get(k, np.nan) for k in ks], **sty)
        if old:  # old grid at the nearest old logg and the same [Fe/H], for direct comparison with the bold line
            og = nearest([k[1] for k in old], g)
            om = nearest([k[2] for k in old], ref)
            ks = sorted(k for k in old if k[1] == og and k[2] == om)
            if ks and abs(og - g) <= 0.06 and abs(om - ref) < 0.05:
                for i, lam in enumerate(lams):
                    axes[i, j].plot([k[0] for k in ks], [np.polyval(old[k], lam) for k in ks],
                                    c=C_OLD, lw=2.2, ls=(0, (1, 1.2)), zorder=7)
            else:
                axes[0, j].text(0.97, 0.05, "outside old grid", transform=axes[0, j].transAxes, fontsize=7,
                                color="0.45", ha="right")
        axes[0, j].set_title(f"logg {g:.2f}", fontsize=10)
        axes[-1, j].set_xlabel("Teff (K)", fontsize=9)
    for i, lam in enumerate(lams):
        axes[i, 0].set_ylabel(f"poly at {lam:.0f} Å", fontsize=9)
    if rms:
        axes[-1, 0].set_ylabel("rms of peaks\nabout fit", fontsize=9)
    for ax in axes.flat:
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap([colour[m] for m in fehs]),
                               norm=plt.Normalize(fehs[0] - 0.05, fehs[-1] + 0.05))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("[Fe/H]", fontsize=9)
    handles = [plt.Line2D([], [], c=C_REF, lw=2.2, marker="o", ms=3, label=f"new grid, [Fe/H] = {ref:+.1f}"),
               plt.Line2D([], [], c=colour[fehs[len(fehs) // 2]], lw=0.8, label="new grid, other [Fe/H] (colour bar)")]
    if old:
        handles.append(plt.Line2D([], [], c=C_OLD, lw=2.2, ls=(0, (1, 1.2)),
                                  label=f"current mstesci1 grid, [Fe/H] = {ref:+.1f}, logg + 0.01"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=9, frameon=False)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description="Diagnostic plots for the pseudo-continuum polynomials",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("root", help="TS wrapper output/ directory")
    ap.add_argument("params", help="merged cont_suppress_param file or a parts/part_*.txt")
    ap.add_argument("--pattern", default="spectra*_pseudo*")
    ap.add_argument("--old", default=None, help="current cont_suppress_param.txt, overplotted where it overlaps")
    ap.add_argument("--nodes", nargs="+", default=None, metavar="TEFF,LOGG,FEH",
                    help="plot these nodes instead of the automatic sample")
    ap.add_argument("--n-teff", type=int, default=6, help="Teff values (columns) in the automatic sample")
    ap.add_argument("--n-feh", type=int, default=5, help="[Fe/H] values (rows) in the automatic sample")
    ap.add_argument("--ncol", type=int, default=6, help="panels per row when using --nodes")
    ap.add_argument("--compare-feh", type=float, default=0.0,
                    help="[Fe/H] drawn bold in the overview and compared with the old grid")
    ap.add_argument("--logg", type=float, default=4.8, help="logg of the automatic sample")
    ap.add_argument("--overview-logg", type=float, nargs="+", default=[4.5, 4.8, 5.2, 5.5])
    ap.add_argument("--prefix", default=None, help="output file prefix (default: params file name)")
    for k, v in F.DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in F.DEFAULTS}

    params = load_params(a.params)
    old = load_params(a.old) if a.old else {}
    prefix = a.prefix or os.path.splitext(a.params)[0]

    if a.nodes:
        sample = []
        for s in a.nodes:
            want = tuple(float(x) for x in s.split(","))
            k = min(params, key=lambda n: abs(n[0] - want[0]) / 50 + abs(n[1] - want[1]) / 0.1
                    + abs(n[2] - want[2]) / 0.1)
            if k != F.node_key(*want):
                print(f"{want} not in file; using nearest node {k}")
            sample.append(k)
    else:
        sample, a.ncol = representative_nodes(list(params), a.n_teff, a.n_feh, a.logg)
    where = locate_spectra(a.params, a.root, a.pattern)
    plot_spectra(sample, a.ncol, params, where, old, cfg, prefix + "_spectra.png")
    plot_overview(params, os.path.splitext(a.params)[0] + ".qa.csv", old, a.overview_logg, a.compare_feh,
                  prefix + "_overview.png")


if __name__ == "__main__":
    main()
