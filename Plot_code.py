"""
Rare Variant Heritability Under Purifying Selection
Simons et al. (2018, 2025) · Zhu et al. (2026)

Equations (matching blog appendix numbering):
  Eq.1: g(q|s) ∝ exp(-γq) / [q(1-q)],  γ = 4Ne·s
  Eq.2: VG(s) ∝ (1 - exp(-2Ne·s)) / (2Ne)
  Eq.3: f_rare(s, q*) = [1 - exp(-γ q*)] / [1 - exp(-γ/2)]
  Eq.4: F_rare(q*) = ∫ f_rare · VG · f(s) ds / ∫ VG · f(s) ds

Ne = 15,000 (harmonic-mean; Tenesa et al. 2007).
Equilibrium approximation; Simons 2025 note ~20% corrections
under non-equilibrium British demography.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.linewidth":     0.7,
    "axes.grid":          True,
    "grid.linewidth":     0.3,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "0.8",
    "legend.fontsize":    8.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

# ── Constants ─────────────────────────────────────────────────────────────────
NE   = 15_000
NGRD = 600
LS   = np.linspace(-8, -1, NGRD)   # log10(s) grid
SS   = 10.0 ** LS


# ── Core maths ────────────────────────────────────────────────────────────────

def f_rare_scalar(s, q_star, ne=NE):
    """Per-class rare fraction [Eq.3]."""
    gamma = 4.0 * ne * s
    if gamma < 1e-10:
        return min(2.0 * q_star, 1.0)
    num = 1.0 - np.exp(-min(gamma * q_star, 700.0))
    den = 1.0 - np.exp(-min(gamma * 0.5,   700.0))
    return min(num / den, 1.0) if den > 1e-20 else min(2.0 * q_star, 1.0)


def f_rare_vec(s_arr, q_star, ne=NE):
    return np.array([f_rare_scalar(s, q_star, ne) for s in s_arr])


def vg_shape(s_arr, ne=NE):
    """Normalised per-site heritability VG(s) [Eq.2]."""
    g  = 4.0 * ne * np.asarray(s_arr)
    vg = (1.0 - np.exp(-np.minimum(g * 0.5, 700.0))) / (2.0 * ne)
    return vg / vg.max()


def lognorm_fs(log_s, mode, sigma):
    """Unnormalised log-normal f(s) on log10 scale."""
    z = (log_s - mode) / sigma
    p = np.exp(-0.5 * z * z)
    return p / p.max()


def F_rare_integrated(q_star, mode, sigma, ne=NE, n=600):
    """Trait-level F_rare(q*) [Eq.4], numerically integrated."""
    ls  = np.linspace(-8, -1, n)
    s   = 10.0 ** ls
    dl  = ls[1] - ls[0]
    z   = (ls - mode) / sigma
    fs  = np.exp(-0.5 * z * z)
    g   = 4.0 * ne * s
    vg  = (1.0 - np.exp(-np.minimum(g * 0.5, 700.0))) / (2.0 * ne)
    w   = fs * vg
    wt  = np.trapezoid(w, dx=dl)
    if wt < 1e-30:
        return 0.0
    fr  = np.array([f_rare_scalar(si, q_star, ne) for si in s])
    return float(np.trapezoid(w * fr, dx=dl) / wt)


# ── Distribution parameters ───────────────────────────────────────────────────
SIM_MODES  = [-5.5, -4.75, -4.0]
SIM_SIG    = 1.3
SIM_COLS   = ["#78C9A8", "#0F6E56", "#3B6D11"]
SIM_LS     = ["--",      "-",       "--"]
SIM_LW     = [1.4,       2.2,       1.4]
SIM_LABELS = ["Simons low", "Simons central", "Simons high"]

ZHU_MODE  = -4.20   # Fig 4C peak
ZHU_SIG   = 0.75    # Fig 4C FWHM ~1.7 → σ ≈ 0.72
ZHU_COL   = "#A32D2D"
ZHU_LABEL = "Brain/CNS-related traits (Zhu et al. 2026)"

QS_LOG  = np.linspace(-4, -1, 80)
QS_VALS = 10.0 ** QS_LOG

# ── Pre-compute Panel C curves ────────────────────────────────────────────────
print("Computing F_rare integrals...")
FR_sim = [np.array([F_rare_integrated(q, m, SIM_SIG) for q in QS_VALS])
          for m in SIM_MODES]
FR_zhu  = np.array([F_rare_integrated(q, ZHU_MODE, ZHU_SIG) for q in QS_VALS])
print("  done.\n")

S_TICKS  = list(range(-8, 0))
S_LABELS = [f"$10^{{{k}}}$" for k in S_TICKS]


def style_s_axis(ax):
    ax.set_xlim(-8, -1)
    ax.set_xticks(S_TICKS)
    ax.set_xticklabels(S_LABELS, fontsize=8)
    ax.set_xlabel(r"Selection coefficient $s$", fontsize=10)


def style_q_axis(ax):
    q_ticks = [-4, -3.5, -3, -2.5, -2, -1.5, -1]
    ax.set_xlim(QS_LOG[0], QS_LOG[-1])
    ax.set_xticks(q_ticks)
    labels = []
    for t in q_ticks:
        pct = 100 * 10**t
        s = f"{pct:.3g}%" if pct >= 0.1 else f"{pct:.4g}%"
        labels.append(f"$10^{{{t}}}$\n({s})")
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel(r"MAF threshold $q^*$", fontsize=10)


# ── Panel drawing functions ───────────────────────────────────────────────────

def draw_panel_a(ax):
    fs_lo = lognorm_fs(LS, SIM_MODES[0], SIM_SIG)
    fs_hi = lognorm_fs(LS, SIM_MODES[2], SIM_SIG)
    ax.fill_between(LS, fs_lo, fs_hi, color="#0F6E56", alpha=0.12,
                    label="Simons 2025 uncertainty band")
    for mode, col, ls, lw, lab in zip(
            SIM_MODES, SIM_COLS, SIM_LS, SIM_LW, SIM_LABELS):
        ax.plot(LS, lognorm_fs(LS, mode, SIM_SIG),
                color=col, ls=ls, lw=lw, label=lab)
    ax.plot(LS, lognorm_fs(LS, ZHU_MODE, ZHU_SIG),
            color=ZHU_COL, lw=2.2, ls=":", label=ZHU_LABEL)
    ax.axvline(-3, color="gray", lw=0.9, ls=":", alpha=0.6,
               label=r"$s=10^{-3}$  (Simons 'substantial tail' boundary)")
    ax.set_ylabel("$f(s)$  (normalised, peak = 1)", fontsize=10)
    ax.set_ylim(0, 1.6)
    style_s_axis(ax)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("A — Inferred $f(s)$ distributions", fontsize=11, pad=8)


def draw_panel_b(ax):
    THRESH = 0.65
    fr_05  = f_rare_vec(SS, 0.005)
    ax.plot(LS, fr_05, color="#185FA5", lw=1.5, ls="--",
            label=r"$q^*{=}0.5\%$")
    ax.axhspan(0.50, 0.80, color="#C62828", alpha=0.07, zorder=0)
    ax.axhline(THRESH, color="#C62828", lw=1.3, ls="--",
               label="'Missing heritability' range")
    ax.axvspan(-5.5, -4.0, color="#0F6E56", alpha=0.10, zorder=0)
    ax.axvspan(ZHU_MODE - ZHU_SIG, ZHU_MODE + ZHU_SIG,
               color=ZHU_COL, alpha=0.10, zorder=0)
    ax.axvline(-4.75, color="#0F6E56", lw=1.2, ls="-.", alpha=0.8,
               label="Simons et al., 2025")
    ax.axvline(ZHU_MODE, color=ZHU_COL, lw=1.2, ls="-.", alpha=0.8,
               label=ZHU_LABEL)
    # s_crit for central threshold at q*=0.5%
    sc = -np.log(1 - THRESH) / (4.0 * NE * 0.005)
    lsc = np.log10(sc)
    if -8 < lsc < -1:
        ax.axvline(lsc, color="#C62828", lw=0.9, ls="--", alpha=0.45)
        ax.text(lsc + 0.07, 0.01,
                f"$s_{{crit}}$={sc:.1e} ($q^*$=0.5%)",
                fontsize=6.5, color="#C62828")
    # Callouts at Simons central and Zhu median
    for xr, col, vo in [(-4.75, "#0F6E56", 0.09),
                         (ZHU_MODE, ZHU_COL, 0.09)]:
        idx = np.argmin(np.abs(LS - xr))
        yv  = fr_05[idx]
        ax.annotate(f"{yv*100:.2f}%",
                    xy=(xr, yv), xytext=(xr + 0.55, yv + vo),
                    fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    ax.set_ylabel(r"$f_\mathrm{rare}(s,\;q^*)$", fontsize=10)
    ax.set_ylim(0, 1.05)
    style_s_axis(ax)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        r"B — Per-class $f_\mathrm{rare}(s,\;q^*)$, $N_e=15{,}000$",
        fontsize=11, pad=8)


def draw_panel_c(ax):
    ax.fill_between(QS_LOG, FR_sim[0], FR_sim[2],
                    color="#0F6E56", alpha=0.13)
    for FR, col, ls, lw, lab in zip(
            FR_sim, SIM_COLS, SIM_LS, SIM_LW, ["", "Simons et al. 2025", ""]):
        ax.plot(QS_LOG, FR, color=col, ls=ls, lw=lw, label=lab)
    ax.plot(QS_LOG, FR_zhu, color=ZHU_COL, lw=2.5, ls=":", label=ZHU_LABEL)
    ax.axhspan(0.50, 0.80, color="#C62828", alpha=0.07, zorder=0)
    ax.axhline(0.65, color="#C62828", lw=1.3, ls="--",
               label="'Missing Heritability' range")
    ax.axvline(np.log10(0.005), color="#888888", lw=0.9, ls=":",
               label="Array detection limit (~0.5%)")
    # Callouts at array limit for Simons central and Zhu
    qi   = np.argmin(np.abs(QS_VALS - 0.005))
    xpos = np.log10(0.005)
    for FR, col, lab, yo in [(FR_sim[1], "#0F6E56", "S.cent.", 0.06),
                              (FR_zhu,    ZHU_COL,  "Brain\nTraits", -0.06)]:
        yv = FR[qi]
        ha = "left" if yv < 0.15 else "right"
        ax.annotate(f"{yv*100:.1f}%\n({lab})",
                    xy=(xpos, yv), xytext=(xpos - 0.28, yv + yo),
                    fontsize=7.5, color=col, ha=ha,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
    ax.set_ylabel(r"$F_\mathrm{rare}(q^*)$ — trait-level fraction", fontsize=10)
    ax.set_ylim(0, 1.05)
    style_q_axis(ax)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        r"C — Integrated $F_\mathrm{rare}(q^*)$, $N_e=15{,}000$  [Eq. 4]",
        fontsize=11, pad=8)


# ── Combined figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.suptitle(
    r"Rare Variant Heritability Under Purifying Selection   ($N_e = 15{,}000$)"
    "\nSimons 2025 model · Zhu 2026 brain/CNS)",
    fontsize=11, y=1.02)
draw_panel_a(axes[0])
draw_panel_b(axes[1])
draw_panel_c(axes[2])
fig.tight_layout(w_pad=3.0)
out_dir = Path(__file__).parent
fig.savefig(out_dir / "rvh_combined.png", dpi=180, bbox_inches="tight")
print("Saved: rvh_combined.png")
plt.close(fig)

# ── Individual panels ─────────────────────────────────────────────────────────
for fname, draw_fn, title in [
        ("rvh_a_fs_distributions",
         draw_panel_a, r"$f(s)$ Distributions — Simons 2025 · Zhu 2026"),
        ("rvh_b_per_class_f_rare",
         draw_panel_b, r"Per-class $f_\mathrm{rare}(s,\,q^*)$  ($N_e=15{,}000$)"),
        ("rvh_c_integrated_F_rare",
         draw_panel_c, r"Integrated $F_\mathrm{rare}(q^*)$  ($N_e=15{,}000$)"),
]:
    fig_i, ax_i = plt.subplots(figsize=(6.5, 5.2))
    fig_i.suptitle(title, fontsize=11, y=1.01)
    draw_fn(ax_i)
    fig_i.tight_layout()
    fig_i.savefig(out_dir / f"{fname}.png", dpi=180, bbox_inches="tight")
    print(f"Saved: {fname}.png")
    plt.close(fig_i)

# ── Numerical summary ─────────────────────────────────────────────────────────
print("\n" + "="*72)
print("F_rare(q*)  —  Ne=15,000")
print("="*72)
print(f"{'q*':>9}  {'S.low':>8}  {'S.cent':>8}  {'S.high':>8}  {'Zhu brain':>10}")
for q_ref in [0.0001, 0.001, 0.005, 0.01, 0.05]:
    qi = np.argmin(np.abs(QS_VALS - q_ref))
    print(f"{q_ref*100:>8.4g}%  "
          f"{FR_sim[0][qi]*100:>7.2f}%  "
          f"{FR_sim[1][qi]*100:>7.2f}%  "
          f"{FR_sim[2][qi]*100:>7.2f}%  "
          f"{FR_zhu[qi]*100:>9.2f}%")

print("\nPer-class f_rare at q*=0.5% (array detection limit):")
for lsr in [-5.5, -4.75, -4.20, -4.0, -3.0]:
    v = f_rare_scalar(10**lsr, 0.005)
    print(f"  s = 10^{{{lsr:.2f}}}:  {v*100:.4f}%")

print("\nAnalytic check  s=1e-3, q*=0.5%, Ne=15,000:")
g  = 4 * NE * 1e-3
ex = (1 - np.exp(-g * 0.005)) / (1 - np.exp(-g * 0.5))
ap = min(4 * NE * 1e-3 * 0.005, 1.0)
print(f"  Exact [Eq.3]: {ex*100:.4f}%")
print(f"  Linear approx (γ q*): {ap*100:.4f}%")
