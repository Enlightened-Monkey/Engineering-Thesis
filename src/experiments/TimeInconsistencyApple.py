from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "exp": "#0b84a5",
    "qh": "#ca472f",
    "accent": "#f6c85f",
}


def set_plot_style():
    """Applies a clean, publication-ready style."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("deep")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.facecolor": "#f7f9fc",
        "figure.facecolor": "#f7f9fc",
        "axes.edgecolor": "#d0d7e2",
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.color": "#dbe3ef",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.25,
        "xtick.color": "#2f3440",
        "ytick.color": "#2f3440",
        "axes.labelcolor": "#2f3440",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.4,
        "legend.frameon": True,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#d0d7e2",
        "legend.framealpha": 0.95,
    })


set_plot_style()


def qh_value(reward, delay, alpha, beta):
    """Quasi-hyperbolic wartość nagrody."""
    if delay == 0:
        return reward
    return alpha * (beta ** delay) * reward


def exp_value(reward, delay, beta):
    """Zwykły wykładniczy discounting (bez biasu teraźniejszości)."""
    return reward if delay == 0 else (beta ** delay) * reward


# --- Parametry eksperymentu ---
alpha = 0.5   # bias teraźniejszości
beta = 0.99  # cierpliwość

reward_small = 1  # 1 jabłko
reward_large = 2  # 2 jabłka

# Dwa scenariusze: (1) dzisiaj vs jutro, (2) 50 vs 51 dni
scenarios = [
    {"name": "Blisko: dziś vs jutro", "delay_small": 0, "delay_large": 1},
    {"name": "Daleko: 50 vs 51 dni", "delay_small": 50, "delay_large": 51},
]


def evaluate_scenario(model_value_fn, delay_small, delay_large):
    v_small = model_value_fn(reward_small, delay_small)
    v_large = model_value_fn(reward_large, delay_large)
    choice = "2 jabłka" if v_large > v_small else "1 jabłko"
    return v_small, v_large, choice


# --- Obliczenia tabelaryczne ---
rows = []
for scen in scenarios:
    ds, dl = scen["delay_small"], scen["delay_large"]
    v_s_qh, v_l_qh, ch_qh = evaluate_scenario(lambda r, d: qh_value(r, d, alpha, beta), ds, dl)
    v_s_exp, v_l_exp, ch_exp = evaluate_scenario(lambda r, d: exp_value(r, d, beta), ds, dl)
    rows.append({
        "scenario": scen["name"],
        "ds": ds,
        "dl": dl,
        "qh_small": v_s_qh,
        "qh_large": v_l_qh,
        "qh_choice": ch_qh,
        "exp_small": v_s_exp,
        "exp_large": v_l_exp,
        "exp_choice": ch_exp,
    })

print(f"{'Scenariusz':<28} | {'ds':<3} {'dl':<3} | {'QH 1j':<10} {'QH 2j':<10} {'QH wybór':<10} | {'EXP 1j':<10} {'EXP 2j':<10} {'EXP wybór':<10}")
print("-" * 120)
for r in rows:
    print(f"{r['scenario']:<28} | {r['ds']:<3} {r['dl']:<3} | {r['qh_small']:<10.4f} {r['qh_large']:<10.4f} {r['qh_choice']:<10} | {r['exp_small']:<10.4f} {r['exp_large']:<10.4f} {r['exp_choice']:<10}")


# --- Wykresy decay (discounting curves) ---
T = np.arange(0, 61)
value_qh = [qh_value(1.0, t, alpha, beta) for t in T]
value_exp = [exp_value(1.0, t, beta) for t in T]

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(T, value_exp, label='Wykładnicze (β)', color=PALETTE["exp"], linewidth=2.6, linestyle='--', alpha=0.9)
ax.plot(T, value_qh, label='Quasi-hiperboliczne (α, β)', color=PALETTE["qh"], linewidth=2.6, linestyle='-', alpha=0.9)
ax.set_title('Krzywe dyskontowania: wykładnicze vs quasi-hiperboliczne', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Opóźnienie (dni)', fontweight='semibold')
ax.set_ylabel('Zdyskontowana wartość nagrody 1', fontweight='semibold')
ax.set_ylim(0, 1.05)
ax.set_facecolor('#ffffff')
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#cccccc')
ax.legend(framealpha=0.95, edgecolor='gray')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'discount_curves.png', dpi=200, bbox_inches='tight')


# --- Wykres słupkowy: wygrane opcje ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
bar_width = 0.35

for ax, model in zip(axes, ["QH", "EXP"]):
    for i, r in enumerate(rows):
        if model == "QH":
            vals = [r["qh_small"], r["qh_large"]]
            choice = r["qh_choice"]
        else:
            vals = [r["exp_small"], r["exp_large"]]
            choice = r["exp_choice"]

        x0 = i * 2
        colors = [PALETTE["accent"], PALETTE["exp"] if model == "EXP" else PALETTE["qh"]]
        ax.bar([x0 - bar_width/2, x0 + bar_width/2], vals, width=bar_width, color=colors, 
               alpha=0.85, edgecolor='#ffffff', linewidth=1.5)
        ax.text(x0 - bar_width/2, vals[0] + 0.02, f"{vals[0]:.2f}", ha='center', va='bottom', 
                fontsize=9, fontweight='semibold')
        ax.text(x0 + bar_width/2, vals[1] + 0.02, f"{vals[1]:.2f}", ha='center', va='bottom', 
                fontsize=9, fontweight='semibold')
        ax.text(x0, max(vals) + 0.12, f"Wygrywa: {choice}", ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='#2f3440')
        ax.set_xticks([i * 2 for i in range(len(rows))])
        ax.set_xticklabels([r['scenario'] for r in rows], rotation=12, fontsize=10)
    ax.set_title(f"Model: {model}", fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Postrzegana wartość', fontweight='semibold')
    ax.set_ylim(0, max(max(r['qh_small'], r['qh_large'], r['exp_small'], r['exp_large']) for r in rows) + 0.5)
    ax.set_facecolor('#ffffff')
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.8, color='#cccccc')

fig.suptitle('Porównanie wyborów: QH vs Wykładniczy', fontsize=14, fontweight='bold')
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(OUTPUT_DIR / 'choices_bar.png', dpi=200, bbox_inches='tight')