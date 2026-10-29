import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 110, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
C = {'C0': '#4d4d4d', 'C1': '#1f77b4', 'C2': '#d95f02'}
RC = {'P0': '#4d4d4d', 'P1': '#1b7837', 'P2': '#762a83'}
L1 = json.load(open('results/layer1.json')); L2 = json.load(open('results/layer2.json'))
NAMES = {'C0': 'C0 classical', 'C1': 'C1 hybrid', 'C2': 'C2 PQC-only'}
RN = {'P0': 'P0 no intervention', 'P1': 'P1 mandate', 'P2': 'P2 subsidy'}
POOLN = {'psp': 'PSP', 'npci': 'NPCI', 'bankL': 'Large banks', 'bankS': 'Small banks'}

# ---------- Fig 1: proxy architecture
fig, ax = plt.subplots(figsize=(7.6, 3.6)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(-1.3, 4.8)
boxes = [('Payer\nhandset app', 0.2), ('Payer PSP\n(TPAP / bank)', 2.2), ('NPCI\nswitch', 4.2), ('Remitter\nbank', 6.2), ('Beneficiary\nbank', 8.2)]
for t, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.4), 1.6, 1.2, boxstyle='round,pad=0.03', fc='#f2f2f2', ec='#333'))
    ax.text(x + 0.8, 2.0, t, ha='center', va='center', fontsize=8.5)
for x0, x1, lab in [(1.8, 2.2, '1'), (3.8, 4.2, '2'), (5.8, 6.2, '3')]:
    ax.add_patch(FancyArrowPatch((x0, 2.0), (x1, 2.0), arrowstyle='-|>', mutation_scale=9, color='#1f77b4'))
    ax.text((x0 + x1) / 2, 2.12, lab, ha='center', fontsize=7.5, color='#1f77b4')
ax.add_patch(FancyArrowPatch((5.0, 2.65), (9.0, 2.65), arrowstyle='-|>', mutation_scale=9, color='#1b7837', lw=0.9, connectionstyle='arc3,rad=-0.35'))
ax.text(7.0, 3.55, 'credit request via NPCI (legs 4-5)', ha='center', fontsize=7.5, color='#1b7837')
ax.add_patch(FancyArrowPatch((9.0, 1.35), (1.0, 1.35), arrowstyle='-|>', mutation_scale=9, color='#888', lw=0.9, connectionstyle='arc3,rad=0.25'))
ax.text(5.0, -0.55, 'confirmation legs 6-8 return to the payer (via NPCI and PSP)', ha='center', fontsize=7.5, color='#666')
ax.text(5.0, 4.35, 'Signed message on every leg; PIN/credential key material on legs 1-3 (KEM or ECDH).\nPrivate-key operations queue at HSM pools (PSP, NPCI, large banks, small banks).', ha='center', fontsize=8)
fig.savefig('figs/fig1_architecture.png'); plt.close(fig)

# ---------- Fig 2: message size and capacity multiple
cap = L1['capacity']
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
cfgs = ['C0', 'C1', 'C2']
vals = [cap[c]['msg_bytes_cred'] for c in cfgs]
b = ax[0].bar([NAMES[c] for c in cfgs], vals, color=[C[c] for c in cfgs])
for r, v in zip(b, vals): ax[0].text(r.get_x() + r.get_width() / 2, v + 60, f'{v:,}', ha='center', fontsize=8)
ax[0].set_ylabel('Bytes per credential-bearing message'); ax[0].set_title('(a) Message size', fontsize=9)
ax[0].tick_params(axis='x', labelsize=7.5)
x = np.arange(4); wd = 0.26
for i, c in enumerate(['C1', 'C2']):
    r = [cap[c]['ratio'][p] for p in ['psp', 'npci', 'bankL', 'bankS']]
    bb = ax[1].bar(x + (i - 0.5) * wd * 1.2, r, wd * 1.15, color=C[c], label=NAMES[c])
    for xi, v in zip(x + (i - 0.5) * wd * 1.2, r): ax[1].text(xi, v + 0.1, f'{v:.1f}', ha='center', fontsize=7.5)
ax[1].axhline(1, color='#4d4d4d', lw=1, ls='--'); ax[1].text(3.45, 1.1, 'C0 = 1', fontsize=7.5, ha='right')
ax[1].set_xticks(x); ax[1].set_xticklabels([POOLN[p] for p in ['psp', 'npci', 'bankL', 'bankS']], fontsize=8)
ax[1].set_ylabel('HSM units needed, relative to C0'); ax[1].set_title('(b) Capacity multiple to hold design utilisation', fontsize=9)
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('figs/fig2_size_capacity.png'); plt.close(fig)

# ---------- Fig 3: latency and capacity cliff
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0))
E = L1['E2_upgraded']
for j, ln in enumerate(['normal', 'peak']):
    for i, c in enumerate(cfgs):
        for k, (q, ls) in enumerate([('p50', 'o'), ('p99', 's')]):
            m = E[f'{c}_{ln}'][q]['mean'] * 1e3; ci = E[f'{c}_{ln}'][q]['ci'] * 1e3
            ax[0].errorbar(i + (j - 0.5) * 0.28 + (k - 0.5) * 0.09, m, yerr=ci, fmt=ls, color=C[c], mfc=('white' if j == 0 else C[c]), ms=5, capsize=2)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([NAMES[c] for c in cfgs], fontsize=7.5)
ax[0].set_ylabel('End-to-end latency (ms)'); ax[0].set_title('(a) Capacity restored\n(circle p50, square p99; open = busy hour, filled = festival peak)', fontsize=8)
phis = sorted(L1['E2_cliff'].keys(), key=float, reverse=True)
x = [float(p) for p in phis]
ax[1].errorbar(x, [L1['E2_cliff'][p]['p99']['mean'] * 1e3 if L1['E2_cliff'][p]['p99']['mean'] is not None else np.nan for p in phis],
               yerr=[(L1['E2_cliff'][p]['p99']['ci'] or 0) * 1e3 for p in phis], fmt='s-', color=C['C1'], capsize=2)
ax[1].set_xlabel('Provisioned capacity / capacity required for hybrid'); ax[1].set_ylabel('p99 latency (ms)'); ax[1].invert_xaxis()
ax[1].set_title('(b) Hybrid, festival peak: p99', fontsize=8)
ax[2].plot(x, [100 * L1['E2_cliff'][p]['fail_rate']['mean'] for p in phis], 'o-', color='#b2182b')
ax[2].set_xlabel('Provisioned capacity / capacity required for hybrid'); ax[2].set_ylabel('Timeout / failure rate (%)'); ax[2].invert_xaxis()
ax[2].set_title('(c) Hybrid, festival peak: failures', fontsize=8)
fig.tight_layout(); fig.savefig('figs/fig3_latency_cliff.png'); plt.close(fig)

# ---------- Fig 4: S-curves by tier (central scenario)
T = 100; yrs = np.arange(T) / 4 + 2026.75
fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.9), sharey=True)
for j, cl in enumerate(['Large banks', 'Small banks', 'PSPs']):
    for r in ['P0', 'P1', 'P2']:
        cv = np.array(L2['main']['central'][r]['curves'])[:, j]
        ax[j].plot(yrs, cv, color=RC[r], label=RN[r], lw=1.8)
    ax[j].axvline(2026.75 + 12, color='#999', ls=':', lw=1); ax[j].set_title(cl, fontsize=9); ax[j].set_xlim(2026.75, 2046)
    ax[j].set_xlabel('Year')
ax[0].set_ylabel('Volume-weighted share PQC-ready'); ax[0].legend(frameon=False, fontsize=7.5, loc='lower right')
ax[2].text(2038.9, 0.05, 'median T_q\n(central)', fontsize=7, color='#666', ha='right')
fig.tight_layout(); fig.savefig('figs/fig4_scurves.png'); plt.close(fig)

# ---------- Fig 5: social cost decomposition by regime and scenario
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
for j, sc in enumerate(['early', 'central', 'late']):
    for i, r in enumerate(['P0', 'P1', 'P2']):
        d = L2['main'][sc][r]
        ax[j].bar(i, d['cost'], color='#9ecae1', label='Migration cost' if (j == 0 and i == 0) else None)
        ax[j].bar(i, d['loss'], bottom=d['cost'], color='#fc9272', label='Expected loss' if (j == 0 and i == 0) else None)
        ax[j].bar(i, 0.2 * d['fiscal'], bottom=d['cost'] + d['loss'], color='#bcbddc', label='Shadow cost of subsidy' if (j == 0 and i == 0) else None)
        ax[j].errorbar(i, d['social'], yerr=1.96 * d['social_se'], color='k', capsize=3, lw=1)
        ax[j].text(i, d['social'] + 600, f"{d['social']:,.0f}", ha='center', fontsize=7.5)
    ax[j].set_xticks(range(3)); ax[j].set_xticklabels(['P0', 'P1', 'P2']); ax[j].set_title({'early': 'Early T_q (median 7 yr)', 'central': 'Central T_q (12 yr)', 'late': 'Late T_q (20 yr)'}[sc], fontsize=9)
ax[0].set_ylabel('Present-value social cost (INR crore)'); ax[0].legend(frameon=False, fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig('figs/fig5_social_cost.png'); plt.close(fig)

# ---------- Fig 6: policy sweeps
fig, ax = plt.subplots(1, 4, figsize=(12.2, 2.9))
SC = ['early', 'central', 'late']; SCC = {'early': '#b2182b', 'central': '#4d4d4d', 'late': '#2166ac'}
sw = L2['sweeps']
for sc in SC:
    ks = list(sw['deadline'].keys())
    ax[0].plot([2026.75 + int(k) / 4 for k in ks], [sw['deadline'][k][sc]['P1']['d_social_vs_P0']['mean'] for k in ks], 'o-', color=SCC[sc], label=sc)
    ks = list(sw['subsidy'].keys())
    ax[1].plot([float(k) for k in ks], [sw['subsidy'][k][sc]['P2']['d_social_vs_P0']['mean'] for k in ks], 'o-', color=SCC[sc], label=sc)
    ks = list(sw['theta'].keys())
    ax[2].plot([float(k) for k in ks], [sw['theta'][k][sc]['P0']['t90_median'] if sw['theta'][k][sc]['P0']['t90_median'] else np.nan for k in ks], 'o-', color=SCC[sc])
    ks = list(sw['loss_scale'].keys())
    ax[3].plot([float(k) for k in ks], [sw['loss_scale'][k][sc]['P1']['d_social_vs_P0']['mean'] for k in ks], 'o-', color=SCC[sc])
for a in (ax[0], ax[1], ax[3]): a.axhline(0, color="k", lw=0.8)
ax[0].set_xlabel('Mandate deadline (year)'); ax[0].set_ylabel('Social cost vs P0 (INR crore)'); ax[0].set_title('(a) P1: deadline', fontsize=9); ax[0].legend(frameon=False, fontsize=7.5, title='T_q scenario', title_fontsize=7.5, loc='center right')
ax[1].set_xlabel('Subsidy rate s'); ax[1].set_title('(b) P2: subsidy rate', fontsize=9)
ax[2].set_xlabel('Systemic internalisation theta'); ax[2].set_ylabel('Median years to 90% (P0)'); ax[2].set_title('(c) P0: externality', fontsize=9)
ax[3].set_xlabel('Loss scale (x baseline)'); ax[3].set_xscale('log'); ax[3].set_title('(d) P1 vs P0: loss-to-cost ratio', fontsize=9)
fig.tight_layout(); fig.savefig('figs/fig6_sweeps.png'); plt.close(fig)

# ---------- Fig 7: tornado
ob = L2['oat_base']; oat = L2['oat']
def tornado(ax, key, title, xlabel):
    rows = []
    for n, v in oat.items():
        lo, hi = v['low'][key], v['high'][key]
        if lo is None or hi is None: continue
        rows.append((n, lo - ob[key], hi - ob[key]))
    rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])))
    rows = rows[-10:]
    for i, (n, lo, hi) in enumerate(rows):
        ax.barh(i, lo, color='#2166ac', height=0.6); ax.barh(i, hi, color='#b2182b', height=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.axvline(0, color='k', lw=0.8); ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
tornado(ax[0], 't90_P0', '(a) Median years to 90 % (P0)', 'Change vs baseline (years)')
tornado(ax[1], 'd_P1', '(b) Social cost, P1 minus P0', 'Change vs baseline (INR crore)')
tornado(ax[2], 'd_P2', '(c) Social cost, P2 minus P0', 'Change vs baseline (INR crore)')
fig.tight_layout(); fig.savefig('figs/fig7_tornado.png'); plt.close(fig)
print('figs done')
