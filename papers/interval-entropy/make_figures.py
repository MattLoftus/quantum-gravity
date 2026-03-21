"""
Generate figures for the interval entropy paper.
Uses matplotlib (saves to PDF for LaTeX inclusion).
Falls back to text-based output if matplotlib is unavailable.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — generating text summaries instead")


def load_data():
    with open('/Users/Loftus/workspace/quantum-gravity/paper/data.json') as f:
        return json.load(f)


def fig1_transition(data):
    """Figure 1: S and H vs beta for N=50, eps=0.12"""
    primary = data['primary']
    betas = [r['beta'] for r in primary]
    S_vals = [r['S_mean'] for r in primary]
    S_errs = [r['S_err'] for r in primary]
    H_vals = [r['H_mean'] for r in primary]
    H_errs = [r['H_err'] for r in primary]

    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 5), sharex=True)

        ax1.errorbar(betas, S_vals, yerr=S_errs, fmt='o-', markersize=3,
                      color='#1f77b4', linewidth=1, capsize=2)
        ax1.axvline(x=9.22, color='gray', linestyle='--', alpha=0.5, label=r'$\beta_c^{\mathrm{pred}}$')
        ax1.set_ylabel(r'$\langle S/\hbar \rangle$')
        ax1.legend(fontsize=8)
        ax1.set_title(r'$N=50$, $\epsilon=0.12$', fontsize=10)

        ax2.errorbar(betas, H_vals, yerr=H_errs, fmt='s-', markersize=3,
                      color='#d62728', linewidth=1, capsize=2)
        ax2.axvline(x=9.22, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel(r'$\beta$')
        ax2.set_ylabel(r'Interval entropy $H$')
        ax2.set_ylim(-0.2, 3.0)

        plt.tight_layout()
        plt.savefig('/Users/Loftus/workspace/quantum-gravity/paper/fig1_transition.pdf',
                     dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved fig1_transition.pdf")
    else:
        print("\nFigure 1 data (S and H vs beta):")
        print(f"  {'beta':>7} {'S':>8} {'H':>8}")
        for b, s, h in zip(betas, S_vals, H_vals):
            print(f"  {b:>7.2f} {s:>8.3f} {h:>8.3f}")


def fig2_susceptibility(data):
    """Figure 2: Susceptibility chi_S and chi_H vs beta"""
    primary = data['primary']
    betas = [r['beta'] for r in primary]
    N = primary[0]['N']
    chi_S = [r['S_var'] * N for r in primary]
    chi_H = [r['H_var'] * N for r in primary]

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(betas, chi_S, 'o-', markersize=3, color='#1f77b4', label=r'$\chi_S$')
        ax.plot(betas, chi_H, 's-', markersize=3, color='#d62728', label=r'$\chi_H$')
        ax.axvline(x=9.22, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'Susceptibility')
        ax.legend()
        ax.set_title(r'$N=50$, $\epsilon=0.12$', fontsize=10)
        plt.tight_layout()
        plt.savefig('/Users/Loftus/workspace/quantum-gravity/paper/fig2_susceptibility.pdf',
                     dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved fig2_susceptibility.pdf")
    else:
        print("\nFigure 2 data (susceptibilities):")
        print(f"  {'beta':>7} {'chi_S':>8} {'chi_H':>8}")
        for b, cs, ch in zip(betas, chi_S, chi_H):
            print(f"  {b:>7.2f} {cs:>8.2f} {ch:>8.2f}")


def fig3_fss(data):
    """Figure 3: Finite-size scaling of chi_S_max"""
    if 'fss_012' not in data:
        print("FSS data not available yet")
        return

    fss = data['fss_012']
    Ns = sorted([int(k) for k in fss.keys()])
    chi_maxs = []
    beta_cs = []

    for N in Ns:
        results = fss[str(N)]
        chi_S = [r['S_var'] * N for r in results]
        pk = np.argmax(chi_S)
        chi_maxs.append(chi_S[pk])
        beta_cs.append(results[pk]['beta'])

    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

        ax1.plot(Ns, chi_maxs, 'ko-', markersize=5)
        ax1.set_xlabel('$N$')
        ax1.set_ylabel(r'$\chi_S^{\max}$')
        ax1.set_title('Susceptibility peak', fontsize=10)

        ax2.plot(Ns, beta_cs, 'ko-', markersize=5, label='Measured')
        bc_pred = [6.64 / (N * 0.12 ** 2) for N in Ns]
        ax2.plot(Ns, bc_pred, 'r--', label='Predicted')
        ax2.set_xlabel('$N$')
        ax2.set_ylabel(r'$\beta_c$')
        ax2.legend(fontsize=8)
        ax2.set_title('Critical coupling', fontsize=10)

        plt.tight_layout()
        plt.savefig('/Users/Loftus/workspace/quantum-gravity/paper/fig3_fss.pdf',
                     dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved fig3_fss.pdf")
    else:
        print("\nFigure 3 data (FSS):")
        for N, chi, bc in zip(Ns, chi_maxs, beta_cs):
            print(f"  N={N}: chi_max={chi:.1f}, beta_c={bc:.2f}")


def fig4_random_control(data):
    """Figure 4: Random graph control"""
    if 'random_graph' not in data:
        print("Random graph data not available yet")
        return

    rg = data['random_graph']

    if HAS_MPL:
        labels = []
        H_bd = []
        H_rand = []

        for N in [30, 50, 70]:
            cont_key = f"N{N}_continuum"
            cryst_key = f"N{N}_crystalline"
            if cont_key in rg and cryst_key in rg:
                # For the crystalline phase, we need BD H values from FSS data
                labels.append(f'N={N}')
                H_rand.append(rg[cryst_key]['H_mean'])

        fig, ax = plt.subplots(figsize=(4.5, 3))
        x = range(len(labels))
        ax.bar([i - 0.15 for i in x], [0.3] * len(labels), 0.3,
                label='BD crystalline', color='#d62728', alpha=0.8)
        ax.bar([i + 0.15 for i in x], H_rand, 0.3,
                label='Random DAG (same $L/N$)', color='#7f7f7f', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Interval entropy $H$')
        ax.legend(fontsize=8)
        ax.set_title('Random graph control', fontsize=10)
        plt.tight_layout()
        plt.savefig('/Users/Loftus/workspace/quantum-gravity/paper/fig4_control.pdf',
                     dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved fig4_control.pdf")
    else:
        print("\nFigure 4 data (random graph control):")
        for key, val in rg.items():
            print(f"  {key}: H={val['H_mean']:.3f}±{val['H_std']:.3f}")


if __name__ == '__main__':
    try:
        data = load_data()
    except FileNotFoundError:
        print("data.json not found — run exp30_paper_data.py first")
        sys.exit(1)

    fig1_transition(data)
    fig2_susceptibility(data)
    fig3_fss(data)
    fig4_random_control(data)

    print("\nAll figures generated.")
