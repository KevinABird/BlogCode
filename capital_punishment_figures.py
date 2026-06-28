"""
Figures for "Capital Punishment Is Not a Breeding Program"
Sequences and Consequences, 2026

Model: R = h² × S × r × (1-f) × n
  h² = narrow-sense heritability
  S  = selection differential (0.049, F&H upper estimate)
  r  = selection accuracy (ρ/h, where ρ is Robertson accuracy)
  f  = fraction of reproduction completed before execution
  n  = number of generations (10)

Observed decline: 0.56 SD threshold shift over 10 generations (F&H)
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 200,
})

# Colors
C_BLUE = "#2166ac"
C_RED = "#b2182b"
C_GREEN = "#1b7837"
C_TEAL = "#01665e"
C_ORANGE = "#e08214"
C_GREY = "#636363"
C_PURPLE = "#762a83"

# Constants
S_FH = 0.049       # F&H upper selection differential (2% execution rate)
N_GEN = 10          # generations, 1500-1750
OBS_SHIFT = 0.56    # observed threshold shift in SD


def S_remove(p):
    """Selection differential from removing proportion p from upper tail."""
    z = stats.norm.ppf(1 - p)
    return stats.norm.pdf(z) / stats.norm.cdf(z)


def i_keep(p):
    """Selection intensity from keeping proportion p from lower tail."""
    z = stats.norm.ppf(1 - p)
    return stats.norm.pdf(z) / p


# ── Figure 1: Culling vs. selective breeding ─────────────────────────────

def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.linspace(-4, 4, 500)
    pdf = stats.norm.pdf(x)

    S_cull = S_remove(0.01)
    S_breed = i_keep(0.10)

    # Left: remove worst 1%
    ax1.plot(x, pdf, color=C_BLUE, linewidth=2)
    z_cut = stats.norm.ppf(0.99)
    ax1.fill_between(x[x < z_cut], pdf[x < z_cut], color=C_BLUE, alpha=0.08)
    ax1.fill_between(x[x >= z_cut], pdf[x >= z_cut], color=C_RED, alpha=0.5)
    ax1.axvline(0, color="#bdbdbd", linewidth=0.8, linestyle="--", alpha=0.4)
    ax1.axvline(-S_cull, color=C_BLUE, linewidth=2, alpha=0.7)
    ax1.annotate("", xy=(-S_cull, 0.32), xytext=(0, 0.32),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=2))
    ax1.text(-S_cull / 2, 0.335, f"|S| = {S_cull:.4f}σ per gen",
            ha="center", fontsize=11, color=C_RED, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    ax1.text(z_cut + 0.15, 0.06, "1%\nremoved", fontsize=10, color=C_RED,
            fontweight="bold", ha="left")
    ax1.text(-2.5, 0.38, "99% survive\nand reproduce", fontsize=10,
            color=C_BLUE, alpha=0.8)
    ax1.set_title("Remove the worst 1%\n(the execution model)",
                  fontsize=13, fontweight="bold", color=C_RED)
    ax1.set_xlabel("Aggression liability (σ)")
    ax1.set_ylabel("Density")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(0, 0.44)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: keep tamest 10%
    ax2.plot(x, pdf, color=C_GREEN, linewidth=2)
    z_keep = stats.norm.ppf(0.10)
    ax2.fill_between(x[x > z_keep], pdf[x > z_keep], color=C_RED, alpha=0.20)
    ax2.fill_between(x[x <= z_keep], pdf[x <= z_keep], color=C_GREEN, alpha=0.5)
    ax2.axvline(0, color="#bdbdbd", linewidth=0.8, linestyle="--", alpha=0.4)
    ax2.axvline(-S_breed, color=C_GREEN, linewidth=2, alpha=0.7)
    ax2.annotate("", xy=(-S_breed, 0.32), xytext=(0, 0.32),
                arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=2))
    ax2.text(-S_breed / 2, 0.335, f"|S| = {S_breed:.3f}σ per gen",
            ha="center", fontsize=11, color=C_GREEN, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
    ax2.text(-3.2, 0.02, "10% kept", fontsize=10, color=C_GREEN,
            fontweight="bold", ha="center")
    ax2.text(1.5, 0.38, "90% eliminated", fontsize=10, color=C_RED,
            alpha=0.8, ha="center")
    ax2.set_title("Keep the tamest 10%\n(a selective breeding program)",
                  fontsize=13, fontweight="bold", color=C_GREEN)
    ax2.set_xlabel("Aggression liability (σ)")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(0, 0.44)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ratio = S_breed / S_cull
    fig.suptitle("Culling vs. selective breeding: the selection intensity gap",
                fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 1: S_cull={S_cull:.5f}, S_breed={S_breed:.4f}, ratio={ratio:.0f}×")


# ── Figure 2: Parameter effects ──────────────────────────────────────────

def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel A: heritability
    ax = axes[0]
    h2r = np.linspace(0, 0.75, 200)
    frac = h2r * S_FH * N_GEN / OBS_SHIFT * 100
    ax.plot(h2r, frac, color=C_BLUE, linewidth=2.5)
    for h2, label, color, xoff, yoff in [
        (0.077, "SNP h² antisocial\n(Tielbeek et al. 2022)\nh²=0.077 → 6.7%",
         C_TEAL, 0.08, 6),
        (0.40, "Twin H²\n(diff. raters)\nh²=0.4 → 35.0%",
         C_ORANGE, -0.14, 5),
        (0.69, "F&H's value\n(same-rater twin H²)\nh²=0.69 → 60.4%",
         C_RED, -0.28, -8),
    ]:
        fr = h2 * S_FH * N_GEN / OBS_SHIFT * 100
        ax.plot(h2, fr, "o", color=color, markersize=8, zorder=5)
        ax.annotate(label, xy=(h2, fr), xytext=(h2 + xoff, fr + yoff),
                   fontsize=7.5, color=color, fontweight="bold",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
                   arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
    ax.set_xlabel("Narrow-sense h²")
    ax.set_ylabel("% of decline explained\n(r = 1, f = 0)")
    ax.set_title("A. Heritability\nR = h² × S × n", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 0.78)
    ax.set_ylim(0, 70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: selection accuracy
    ax = axes[1]
    r_range = np.linspace(0, 1.0, 200)
    for h2, label, color, ls in [
        (0.69, "h² = 0.69 (F&H)", "#969696", "--"),
        (0.077, "h² = 0.077 (SNP)", C_BLUE, "-"),
    ]:
        fr = h2 * S_FH * r_range * N_GEN / OBS_SHIFT * 100
        ax.plot(r_range, fr, color=color, linewidth=2, linestyle=ls, label=label)
    ax.set_xlabel("r (selection accuracy)")
    ax.set_ylabel("% of decline explained\n(f = 0)")
    ax.set_title("B. Selection accuracy\nR × r attenuation", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 65)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: reproductive timing
    ax = axes[2]
    f_range = np.linspace(0, 1, 200)
    for h2, r, label, color, ls in [
        (0.69, 1.0, "F&H (h²=0.69, r=1)", "#969696", "--"),
        (0.077, 1.0, "h²=0.077, r=1", C_BLUE, "-."),
        (0.077, 0.35, "h²=0.077, r=0.35", C_TEAL, "-"),
    ]:
        fr = h2 * S_FH * r * (1 - f_range) * N_GEN / OBS_SHIFT * 100
        ax.plot(f_range, fr, color=color, linewidth=2, linestyle=ls, label=label)
    ax.set_xlabel("f (fraction already reproduced)")
    ax.set_ylabel("% of decline explained")
    ax.set_title("C. Reproductive timing\nR × (1 − f)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "How each parameter attenuates the response to selection\n"
        "R = h² × S × r × (1 − f) × n,  with S = 0.049, n = 10",
        fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_parameter_effects.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 2: done")


# ── Figure 3: Parameter sweep + specification curve ──────────────────────

def fig3():
    np.random.seed(2024)
    N = 500_000
    N_sca = 50_000

    def sweep(n):
        h2 = np.random.uniform(0.03, 0.69, n)
        r = np.random.uniform(0.10, 1.00, n)
        f = np.random.uniform(0.00, 1.00, n)
        frac = h2 * S_FH * (1 - f) * r * N_GEN / OBS_SHIFT
        return h2, r, f, frac

    _, _, _, frac = sweep(N)
    h2s, rs, fs, fracs = sweep(N_sca)
    fh = 0.69 * S_FH * 1.0 * 1.0 * N_GEN / OBS_SHIFT

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.4], hspace=0.25, wspace=0.3)
    ax_hist = fig.add_subplot(gs[:, 0])
    ax_sca = fig.add_subplot(gs[0, 1])
    ax_p1 = fig.add_subplot(gs[1, 1], sharex=ax_sca)
    ax_p2 = fig.add_subplot(gs[2, 1], sharex=ax_sca)
    ax_p3 = fig.add_subplot(gs[3, 1], sharex=ax_sca)

    # Histogram
    bins = np.linspace(0, 0.65, 200)
    ax_hist.hist(frac, bins=bins, density=True, color=C_BLUE, alpha=0.7,
                 edgecolor="none")
    for pct, ls in [(50, "-"), (90, "--"), (95, ":"), (99, "-.")]:
        v = np.percentile(frac, pct)
        ax_hist.axvline(v, color=C_RED, linewidth=1.2, linestyle=ls, alpha=0.7)
        ym = {50: 16, 90: 13, 95: 10, 99: 7}
        ax_hist.text(v + 0.008, ym[pct], f"P{pct} = {v * 100:.1f}%",
                    fontsize=8.5, color=C_RED)
    ax_hist.axvline(fh, color="#969696", linewidth=2.5, alpha=0.8)
    ax_hist.text(fh - 0.015, 14, f"F&H\n({fh * 100:.0f}%)", fontsize=10,
                color="#969696", ha="right", fontweight="bold")
    ax_hist.set_xlabel("Fraction of decline explained", fontsize=12)
    ax_hist.set_ylabel("Density", fontsize=12)
    ax_hist.set_title("Distribution across\nparameter space",
                      fontsize=12, fontweight="bold")
    ax_hist.set_xlim(0, 0.65)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    # Specification curve
    order = np.argsort(fracs)
    fs_ = fracs[order] * 100
    h2_ = h2s[order]
    r_ = rs[order]
    f_ = fs[order]
    x = np.arange(N_sca)

    ax_sca.fill_between(x, 0, fs_, color=C_BLUE, alpha=0.5, linewidth=0)
    ax_sca.axhline(20, color=C_RED, linewidth=1.5, linestyle="--", alpha=0.7)
    ax_sca.axhline(10, color=C_ORANGE, linewidth=1, linestyle=":", alpha=0.5)
    fa20 = (fracs * 100 >= 20).mean() * 100
    fa10 = (fracs * 100 >= 10).mean() * 100
    ax_sca.text(N_sca * 0.60, 22, f"{fa20:.1f}% of space above 20%",
               fontsize=9, color=C_RED, fontweight="bold")
    ax_sca.text(N_sca * 0.50, 12, f"{fa10:.1f}% above 10%",
               fontsize=9, color=C_ORANGE)
    ax_sca.set_ylabel("% decline\nexplained", fontsize=10)
    ax_sca.set_title("Specification curve: all combinations ranked",
                     fontsize=11, fontweight="bold")
    ax_sca.set_ylim(0, min(fs_[-1] * 1.1, 65))
    ax_sca.spines["top"].set_visible(False)
    ax_sca.spines["right"].set_visible(False)

    # Parameter strips
    nb = 400
    bs = N_sca // nb
    for ax, data, lab, col, bds in [
        (ax_p1, h2_, "h²", C_BLUE, [0.03, 0.69]),
        (ax_p2, r_, "r", C_GREEN, [0.1, 1.0]),
        (ax_p3, f_, "f", C_ORANGE, [0.0, 1.0]),
    ]:
        binned = [data[i * bs:(i + 1) * bs].mean() for i in range(nb)]
        ax.bar(np.linspace(0, N_sca, nb), binned, width=N_sca / nb,
              color=col, alpha=0.7, edgecolor="none")
        ax.set_ylabel(lab, fontsize=10, rotation=0, ha="right", va="center")
        ax.set_ylim(bds)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_p3.set_xlabel("Parameter combinations (ranked lowest → highest)", fontsize=10)

    fig.suptitle(
        "Parameter space sweep: three-parameter model\n"
        "R = h² × S × r × (1−f) × n;    "
        "h² ∈ [0.03, 0.69],  r ∈ [0.1, 1.0],  f ∈ [0, 1.0]",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sweep_sca.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Fig 3: median={np.median(frac) * 100:.1f}%, "
          f"P90={np.percentile(frac, 90) * 100:.1f}%, "
          f"P(≥20%)={( frac * 100 >= 20).mean() * 100:.1f}%")


# ── Figure 4: Belyaev comparison (lollipop) ──────────────────────────────

def fig4():
    # Belyaev: h²=0.44 (Morrill et al. 2022, human sociability in dogs),
    # i=1.755 (keep best 10%), r=1 (direct measurement), f=0
    R_bel = 0.44 * i_keep(0.10)
    R_fh = 0.69 * S_FH          # F&H idealized (r=1, f=0)
    R_med = 0.00306              # median from sweep above

    fig, ax = plt.subplots(figsize=(11, 4))

    scenarios = [
        ("Parameter ensemble\nmedian", R_med, C_BLUE,
         f"{R_med:.3f} σ/gen  ({R_med * N_GEN / OBS_SHIFT * 100:.1f}% of decline)"),
        ("F&H idealized\n(h²=0.69, r=1, f=0)", R_fh, C_GREY,
         f"{R_fh:.3f} σ/gen  ({R_fh * N_GEN / OBS_SHIFT * 100:.0f}% of decline)"),
        ("Belyaev foxes\n(h²≈0.44, keep best 10%)", R_bel, C_GREEN,
         f"{R_bel:.2f} σ/gen"),
    ]

    y = np.arange(len(scenarios))[::-1]
    for yi, (label, val, color, note) in zip(y, scenarios):
        ax.plot([0, val], [yi, yi], color=color, linewidth=2, alpha=0.5)
        ax.plot(val, yi, "o", color=color, markersize=12,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        ax.text(val + 0.012, yi, note, va="center", fontsize=10,
               color=color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in scenarios], fontsize=11)
    ax.set_xlabel("Per-generation selection response (σ$_P$)", fontsize=12)
    ax.set_xlim(-0.01, R_bel + 0.22)
    ax.set_ylim(-0.5, 2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_belyaev.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Fig 4: R_bel={R_bel:.4f}, R_fh={R_fh:.4f}, R_med={R_med:.5f}, "
          f"Bel/F&H={R_bel / R_fh:.0f}×, Bel/med={R_bel / R_med:.0f}×")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"\nAll figures saved to {OUT}/")
