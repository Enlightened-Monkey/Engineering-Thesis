"""
Wizualizacja quasi-hiperbolicznego dyskontowania dla różnych parametrów alfa.

Porównuje funkcje dyskontowania d(t) dla różnych wartości α:
- d(0) = 1
- d(t) = α * β^t dla t ≥ 1
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Stałe do ponownego użycia w całym module
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "docs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#0b84a5", "#f6c85f", "#9dd866", "#ca472f", "#6f4e7c"]


def set_plot_style():
    """Konfiguruje spójny, czytelny styl wykresów."""
    sns.set_theme(context="notebook", style="ticks", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "axes.facecolor": "#f7f9fc",
        "figure.facecolor": "#f7f9fc",
        "axes.edgecolor": "#d0d7e2",
        "axes.linewidth": 1.1,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#dbe3ef",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "xtick.color": "#3a3f4b",
        "ytick.color": "#3a3f4b",
        "axes.labelcolor": "#2f3440",
        "font.size": 12,
        "legend.frameon": True,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#d0d7e2",
        "legend.framealpha": 0.9,
    })


def save_figure(fig, filename: str) -> None:
    """Zapisuje figurę w katalogu data/plots z podwyższoną rozdzielczością."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=400, bbox_inches="tight")
    print(f"Saved: {path}")


# Ustaw styl globalnie po zdefiniowaniu helperów
set_plot_style()


def qh_discount(t, alpha, beta):
    """
    Oblicza współczynnik quasi-hiperbolicznego dyskontowania.
    
    Parameters:
    -----------
    t : int or array
        Krok czasowy
    alpha : float
        Parametr uprzedzenia teraźniejszości (0 < alpha ≤ 1)
    beta : float
        Współczynnik dyskontowania wykładniczego (0 ≤ beta < 1)
    
    Returns:
    --------
    float or array
        Współczynnik dyskontowania d(t)
    """
    t = np.asarray(t)
    discount = np.where(t == 0, 1.0, alpha * (beta ** t))
    return discount


def exponential_discount(t, gamma):
    """
    Oblicza współczynnik wykładniczego dyskontowania.
    
    Parameters:
    -----------
    t : int or array
        Krok czasowy
    gamma : float
        Współczynnik dyskontowania (0 ≤ gamma < 1)
    
    Returns:
    --------
    float or array
        Współczynnik dyskontowania γ^t
    """
    return gamma ** t


def plot_qh_comparison():
    """Porównanie QH dyskontowania dla różnych wartości alfa."""
    
    # Parametry
    beta = 0.95
    alpha_values = [1.0, 0.9, 0.8, 0.7, 0.5]
    t_max = 40
    t = np.arange(0, t_max + 1)
    
    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = PALETTE[: len(alpha_values)]

    for i, alpha in enumerate(alpha_values):
        curve = qh_discount(t, alpha, beta)
        label = f'α = {alpha:.1f}' + (' (wykładnicze)' if alpha == 1.0 else '')
        linestyle = "--" if alpha == 1.0 else "-"
        linewidth = 2.6 if alpha == 1.0 else 2.3
        ax.plot(
            t,
            curve,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            color=colors[i],
            alpha=0.9,
        )

    ax.set_xlabel('Krok czasowy (t)', fontsize=13)
    ax.set_ylabel('Współczynnik dyskontowania d(t)', fontsize=13)
    ax.set_title(
        f'Quasi-hiperboliczne dyskontowanie dla różnych α (β = {beta})',
        fontsize=15,
    )
    ax.legend(fontsize=11, loc='upper right', title='Parametry', title_fontsize=11)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.05)
    ax.margins(x=0)

    plt.tight_layout()
    save_figure(fig, 'qh_discounting_comparison.png')
    plt.close(fig)


def plot_qh_log_scale():
    """Wykres QH dyskontowania w skali logarytmicznej."""
    
    beta = 0.95
    alpha_values = [0.5, 0.7, 0.9, 1.0]
    t_max = 100
    t = np.arange(0, t_max + 1)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = PALETTE[: len(alpha_values)]

    for i, alpha in enumerate(alpha_values):
        discount = qh_discount(t, alpha, beta)
        label = f'α = {alpha:.1f}' + (' (wykładnicze)' if alpha == 1.0 else '')
        linestyle = "--" if alpha == 1.0 else "-"
        linewidth = 2.6 if alpha == 1.0 else 2.2
        ax.semilogy(
            t,
            discount,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            color=colors[i],
            alpha=0.9,
        )

    ax.set_xlabel('Krok czasowy (t)', fontsize=13)
    ax.set_ylabel('Współczynnik dyskontowania d(t) [skala log]', fontsize=13)
    ax.set_title(f'QH dyskontowanie w skali logarytmicznej (β = {beta})', fontsize=15)
    ax.legend(fontsize=11, loc='upper right', title='Parametry', title_fontsize=11)
    ax.grid(True, alpha=0.35, which='both')
    ax.set_xlim(0, t_max)
    ax.set_ylim(1e-4, 1.1)

    plt.tight_layout()
    save_figure(fig, 'qh_discounting_log_scale.png')
    plt.close(fig)


def plot_qh_beta_comparison():
    """Porównanie QH dyskontowania dla różnych wartości beta."""
    
    alpha = 0.8
    beta_values = [0.85, 0.90, 0.95, 0.99]
    t_max = 50
    t = np.arange(0, t_max + 1)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("crest", len(beta_values))

    for i, beta in enumerate(beta_values):
        discount = qh_discount(t, alpha, beta)
        ax.plot(
            t,
            discount,
            label=f'β = {beta}',
            linewidth=2.4,
            color=colors[i],
            alpha=0.9,
        )

    ax.set_xlabel('Krok czasowy (t)', fontsize=13)
    ax.set_ylabel('Współczynnik dyskontowania d(t)', fontsize=13)
    ax.set_title(f'QH dyskontowanie dla różnych β (α = {alpha})', fontsize=15)
    ax.legend(fontsize=11, loc='upper right', title='Parametry', title_fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_figure(fig, 'qh_discounting_beta_comparison.png')
    plt.close(fig)


def plot_qh_heatmap():
    """Mapa ciepła pokazująca wpływ α i β na dyskontowanie w czasie t=10."""
    
    alpha_range = np.linspace(0.5, 1.0, 50)
    beta_range = np.linspace(0.85, 0.99, 50)
    t = 10
    
    discount_values = np.zeros((len(beta_range), len(alpha_range)))
    
    for i, beta in enumerate(beta_range):
        for j, alpha in enumerate(alpha_range):
            discount_values[i, j] = qh_discount(t, alpha, beta)
    
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        discount_values,
        aspect='auto',
        origin='lower',
        extent=[alpha_range[0], alpha_range[-1], beta_range[0], beta_range[-1]],
        cmap='rocket_r',
        interpolation='bilinear',
    )

    ax.set_xlabel('α (parametr uprzedzenia teraźniejszości)', fontsize=13)
    ax.set_ylabel('β (współczynnik dyskontowania)', fontsize=13)
    ax.set_title(f'Współczynnik dyskontowania d({t}) dla różnych α i β', fontsize=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label('d(t)', fontsize=12)

    contours = ax.contour(
        alpha_range,
        beta_range,
        discount_values,
        levels=12,
        colors='black',
        alpha=0.35,
        linewidths=0.7,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    plt.tight_layout()
    save_figure(fig, 'qh_discounting_heatmap.png')
    plt.close(fig)


def plot_present_bias_effect():
    """Wizualizacja efektu uprzedzenia teraźniejszości."""
    
    beta = 0.95
    t_max = 20
    t = np.arange(0, t_max + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel 1: Porównanie α = 0.7 vs α = 1.0
    alpha_low = 0.7
    alpha_high = 1.0
    
    discount_qh = qh_discount(t, alpha_low, beta)
    discount_exp = qh_discount(t, alpha_high, beta)
    
    ax1.plot(
        t,
        discount_exp,
        '--',
        linewidth=2.8,
        label=f'α = {alpha_high} (wykładnicze)',
        alpha=0.9,
        color=PALETTE[0],
    )
    ax1.plot(
        t,
        discount_qh,
        '-',
        linewidth=2.8,
        label=f'α = {alpha_low} (QH)',
        alpha=0.9,
        color=PALETTE[3],
    )
    ax1.fill_between(
        t,
        discount_exp,
        discount_qh,
        alpha=0.12,
        color='#9aa6b2',
        label='Efekt uprzedzenia',
    )
    
    ax1.set_xlabel('Krok czasowy (t)', fontsize=13)
    ax1.set_ylabel('Współczynnik dyskontowania d(t)', fontsize=13)
    ax1.set_title('Efekt uprzedzenia teraźniejszości', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.35)
    ax1.set_xlim(0, t_max)
    
    # Panel 2: Stosunek dyskontowania QH do wykładniczego
    alpha_values = [0.5, 0.7, 0.9]
    colors = PALETTE[: len(alpha_values)]

    for i, alpha in enumerate(alpha_values):
        discount_qh = qh_discount(t, alpha, beta)
        discount_exp = exponential_discount(t, beta)
        ratio = discount_qh / discount_exp
        ratio[0] = 1.0

        ax2.plot(
            t[1:],
            ratio[1:],
            linewidth=2.4,
            label=f'α = {alpha}',
            color=colors[i],
            alpha=0.9,
        )

    ax2.axhline(y=1.0, color='#4a4f5c', linestyle='--', linewidth=1.1, alpha=0.6)
    ax2.set_xlabel('Krok czasowy (t)', fontsize=13)
    ax2.set_ylabel('Stosunek: d_QH(t) / d_exp(t)', fontsize=13)
    ax2.set_title('Relatywne uprzedzenie teraźniejszości', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.35)
    ax2.set_xlim(1, t_max)
    
    plt.tight_layout()
    save_figure(fig, 'qh_present_bias_effect.png')
    plt.close(fig)


def main():
    """Uruchamia wszystkie wizualizacje."""
    
    print("Generowanie wykresów quasi-hiperbolicznego dyskontowania...")
    print("=" * 60)
    
    print("\n1. Porównanie dla różnych wartości α...")
    plot_qh_comparison()
    
    print("\n" + "=" * 60)
    print("Wygenerowano wybrany wykres.")


if __name__ == "__main__":
    main()
