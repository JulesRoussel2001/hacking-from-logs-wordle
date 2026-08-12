"""
Departure-plane scatter (ESS x %floor) -- the lead figure of the post.

Every number here traces to the OPE table in key_numbers_table.md (S13):
  D    ESS 0.64  floor 0.05   benign (drift)
  H6   ESS 0.48  floor 0.02   benign (proxy-pressured, non-hacked)
  IMP  ESS 0.03  floor 0.01   benign (improver)
  BON1 ESS 0.12  floor 0.12   bad (degraded, Gate 2 FAIL)
  HL5  ESS 0.00  floor 0.15   hacked (solve-improving)
  HF   ESS 0.00  floor 0.15   hacked (truth-destroying)
  HF-lite ESS 0.00 floor 0.16 hacked (half-dose)  [weights hollow; row from farmer_lite-era OPE log]

X axis is symlog so the 0.00 cluster and 0.64 are both readable.
Output: figures/departure_plane.png (1600 px wide, white background).

Run from the repo root:  python make_departure_plane.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({"font.size": 14, "figure.facecolor": "white",
                     "axes.facecolor": "white"})

# ---- data (name, ESS, floor%, category) --------------------------------
POLICIES = [
    ("D",       0.64, 0.05, "benign"),
    ("H6",      0.48, 0.02, "benign"),
    ("IMP",     0.03, 0.01, "benign"),
    ("BON1",    0.12, 0.12, "bad"),
    # HL5 and HF share exact coordinates (0.00, 0.15); +/-0.0015 vertical
    # jitter is DISPLAY-ONLY so both markers are visible. True values in caption.
    ("HL5",     0.00, 0.1485, "hacked"),
    ("HF",      0.00, 0.1515, "hacked"),
    ("HF-lite", 0.00, 0.16, "hacked"),
]
COLORS  = {"benign": "#2e7d32", "bad": "#e69f00", "hacked": "#c62828"}
MARKERS = {"benign": "o",       "bad": "s",       "hacked": "^"}

# per-label text offsets (points) so nothing overlaps, incl. the 0.00 cluster
OFFSETS = {"D": (8, 6), "H6": (8, 6), "IMP": (8, 6), "BON1": (8, 6),
           "HL5": (10, -16), "HF": (10, 2), "HF-lite": (10, 14)}

fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

# symlog X: linear below linthresh, log above -- keeps 0.00 on the axis
ax.set_xscale("symlog", linthresh=0.02, linscale=0.6)
ax.set_xlim(-0.004, 1.0)
ax.set_ylim(0, 0.20)

# ---- category bands on the %floor axis (the three-way ordering) --------
ax.axhspan(0.00, 0.055, color=COLORS["benign"], alpha=0.06)
ax.axhspan(0.10, 0.135, color=COLORS["bad"],    alpha=0.06)
ax.axhspan(0.14, 0.175, color=COLORS["hacked"], alpha=0.06)
ax.text(0.965, 0.028, "benign \u2264 0.05",  ha="right", va="center",
        fontsize=11, color=COLORS["benign"], style="italic")
ax.text(0.965, 0.118, "bad 0.12",            ha="right", va="center",
        fontsize=11, color=COLORS["bad"],    style="italic")
ax.text(0.965, 0.157, "hacked 0.15\u20130.16", ha="right", va="center",
        fontsize=11, color=COLORS["hacked"], style="italic")

# ---- the ESS threshold that failed as a classifier ---------------------
ax.axvline(0.05, ls="--", color="grey", lw=1.4)
ax.text(0.05, 0.192, " ESS reliability threshold (0.05)\n cuts between IMP and BON1",
        fontsize=10.5, color="dimgrey", va="top")

# ---- the two unconstructed counterexample regions ----------------------
# (a) in-support hacker: hacked but %floor in the benign band, far from A
r1 = mpatches.FancyBboxPatch((0.0005, 0.005), 0.028, 0.045,
        boxstyle="round,pad=0.002", lw=1.4, ls=":",
        edgecolor=COLORS["hacked"], facecolor="none")
ax.add_patch(r1)
ax.text(0.0022, 0.058, "unbuilt: in-support hacker\nwould land here",
        fontsize=10, color=COLORS["hacked"], va="bottom")
# (b) systematic-alien-benign: benign but %floor in the hacked band
r2 = mpatches.FancyBboxPatch((0.10, 0.142), 0.55, 0.028,
        boxstyle="round,pad=0.002", lw=1.4, ls=":",
        edgecolor=COLORS["benign"], facecolor="none")
ax.add_patch(r2)
ax.text(0.145, 0.176, "unbuilt: alien-but-benign policy would land here",
        fontsize=10, color=COLORS["benign"], va="bottom")

# ---- points ------------------------------------------------------------
for name, ess, fl, cat in POLICIES:
    ax.scatter(ess, fl, s=140, c=COLORS[cat], marker=MARKERS[cat],
               edgecolors="black", linewidths=0.8, zorder=5)
    dx, dy = OFFSETS[name]
    ax.annotate(name, (ess, fl), textcoords="offset points",
                xytext=(dx, dy), fontsize=12, fontweight="bold",
                color=COLORS[cat])

# ---- axes, legend, title ----------------------------------------------
ax.set_xlabel("ESS  (effective sample size fraction, symlog scale)")
ax.set_ylabel("%floor  (fraction of actions at A's exploration floor)")
ax.set_title("Six departure profiles on A's logs: ESS grades distance;\n"
             "%floor separates benign / bad / hacked \u2014 single seed",
             fontsize=14)
handles = [plt.Line2D([], [], marker=MARKERS[c], color="w",
                      markerfacecolor=COLORS[c], markeredgecolor="black",
                      markersize=11, label=lbl)
           for c, lbl in (("benign", "benign (D, H6, IMP)"),
                          ("bad", "bad (BON1)"),
                          ("hacked", "hacked (HF, HF-lite, HL5)"))]
ax.legend(handles=handles, loc="center right", frameon=True, fontsize=11)
ax.set_xticks([0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ax.set_xticklabels(["0", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
ax.grid(alpha=0.25)

fig.tight_layout()
import os
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/departure_plane.png", bbox_inches="tight")
print("wrote figures/departure_plane.png")
