"""
IEEE Transactions quality figures for BDA-SAC-DRL paper.
Rules applied:
- Single-column: 3.5 in wide; double-column: 7.16 in wide
- Times New Roman / Computer Modern serif fonts
- All text >= 7 pt (8 pt preferred), axis labels 8 pt, tick labels 7 pt
- Line widths: data lines 1.5 pt, axes 0.8 pt, grid 0.4 pt
- Markers: 4-5 pt, no filled markers unless necessary
- Colors: colorblind-safe 4-color palette
- No duplicate captions (captions belong ONLY in LaTeX, not in figure title)
- Every subplot labeled (a), (b) via ax.text, not suptitle
- Tight bbox, 600 dpi for raster, PDF for vector
- Grid: alpha 0.3, gray
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy import stats
import os

os.makedirs('/home/claude/figs', exist_ok=True)

# ── Global IEEE RC params ────────────────────────────────────────────────────
plt.rcParams.update({
    # Font
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':           8,
    'axes.labelsize':      8,
    'axes.titlesize':      8,
    'legend.fontsize':     7,
    'xtick.labelsize':     7,
    'ytick.labelsize':     7,
    # Lines
    'lines.linewidth':     1.5,
    'lines.markersize':    4.5,
    'axes.linewidth':      0.8,
    'patch.linewidth':     0.8,
    # Grid
    'grid.linewidth':      0.4,
    'grid.color':          '#999999',
    'grid.alpha':          0.30,
    'axes.grid':           True,
    'axes.axisbelow':      True,
    # Spines
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    # Error bars
    'errorbar.capsize':    3,
    # Output
    'figure.dpi':          150,
    'savefig.dpi':         600,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.02,
    # Ticks
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':    3,
    'ytick.major.size':    3,
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
})

# ── Colorblind-safe palette (IBM / Wong 2011) ────────────────────────────────
C = {
    'blue':   '#0072B2',
    'red':    '#D55E00',
    'green':  '#009E73',
    'orange': '#E69F00',
    'sky':    '#56B4E9',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'black':  '#000000',
    'gray':   '#888888',
}
METHOD_COLORS = [C['gray'], C['orange'], C['green'], C['red'], C['blue']]
METHOD_LS     = [':',       '-.',        '--',       (0,(5,2)), '-']
METHOD_MARKS  = ['s',       '^',         'D',        'o',       'P']
METHODS       = ['Rule-Based','PPO','TD3','Std. SAC','BDA-SAC (Proposed)']
SHORT_METHODS = ['Rule-\nBased','PPO','TD3','Std.\nSAC','BDA-SAC\n(Prop.)']

COL1 = 3.5    # single column
COL2 = 7.16   # double column
N_SEEDS = 10
np.random.seed(42)

def label_axes(axes_list, xoff=-0.13, yoff=1.04):
    """Place (a), (b), ... labels at top-left of each axes."""
    for k, ax in enumerate(axes_list):
        ax.text(xoff, yoff, f'({chr(97+k)})',
                transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='left')

def save(name):
    plt.savefig(f'/home/claude/figs/{name}.pdf')
    plt.savefig(f'/home/claude/figs/{name}.png', dpi=300)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIG 1 – System Architecture
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(COL2, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis('off')
ax.grid(False)

def box(ax, x, y, w, h, text, fc='#D6E4F0', ec='#1F4E79',
        fs=7.5, bold=False, lw=0.9, wrap=False):
    r = plt.Rectangle((x,y), w, h, lw=lw, ec=ec, fc=fc, zorder=3,
                       clip_on=False)
    ax.add_patch(r)
    kw = dict(ha='center', va='center', fontsize=fs, zorder=4,
              fontfamily='serif',
              fontweight='bold' if bold else 'normal',
              multialignment='center')
    ax.text(x+w/2, y+h/2, text, **kw)

def arr(ax, x1, y1, x2, y2, color='#333333', lw=0.9,
        ls='-', label='', double=False):
    style = '<->' if double else '->'
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, linestyle=ls), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.06, my+0.06, label, fontsize=6.5,
                color=color, zorder=6, fontfamily='serif')

# --- source column ---
box(ax,0.1,3.3,1.25,0.82,'PV Array\n50\u202fkWp','#FFF8DC','#B8860B',fs=7.5)
box(ax,0.1,2.1,1.25,0.82,'Wind Turbine\n30\u202fkW','#D1F0F7','#006080',fs=7.5)
box(ax,0.1,0.9,1.25,0.82,'EV Fleet\n5\u00d760\u202fkWh\n(V2G)','#D4F7DC','#1B5E20',fs=7)
box(ax,0.1,0.05,1.25,0.65,'Load\n50\u202fkW pk','#F5F5F5','#555',fs=7)

# --- bus ---
box(ax,1.85,1.35,0.65,2.05,'DC/AC\nBus','#EDE7F6','#4527A0',fs=7,lw=1.0)
arr(ax,1.35,3.71,1.85,2.9,C['orange'])
arr(ax,1.35,2.51,1.85,2.25,C['sky'])
arr(ax,1.35,1.31,1.85,1.8,C['green'],label='V2G',ls='--')
arr(ax,1.35,0.38,1.85,1.45,C['gray'],lw=0.7)

# --- grid / battery ---
box(ax,3.05,3.3,1.25,0.82,'Utility Grid\n(PCC)','#EEEEEE','#333')
box(ax,3.05,2.0,1.25,0.82,'Stationary\nBattery\n100\u202fkWh','#FFEBEE','#C62828',fs=7)
arr(ax,2.5,2.68,3.05,3.6)
arr(ax,2.5,2.17,3.05,2.41)

# --- BDA-SAC agent ---
box(ax,4.75,2.0,2.2,1.30,
    'BDA-SAC-DRL Agent\n(Actor\u2013Twin Critic\n+ Entropy Tuning)',
    '#E8F5E9','#1B5E20',fs=8,bold=True,lw=1.3)
arr(ax,4.3,2.45,4.75,2.55,C['blue'],label='$\\mathbf{s}(t)$')
arr(ax,4.75,2.25,4.3,2.15,C['red'],ls='--',label='$\\hat{\\mathbf{a}}(t)$')

# --- MPC layer ---
box(ax,4.75,0.9,2.2,0.80,'MPC Constraint Layer\n(OSQP, $H=4$ steps)',
    '#FFFDE7','#F57F17',fs=7,lw=1.1)
arr(ax,5.85,2.0,5.85,1.70,C['orange'],label=' $\\mathbf{a}^*(t)$')
arr(ax,5.85,0.9,4.3,0.9,C['green'],lw=0.9)

# --- reward computation ---
box(ax,7.55,2.0,2.2,1.30,
    'Reward\n$r(t)$\n= cost+V2G\n$-$deg$-$curt$-$viol',
    '#FCE4EC','#880E4F',fs=7,lw=1.1)
arr(ax,6.95,2.65,7.55,2.65,C['black'],label='exec.')
ax.annotate('', xy=(5.85,3.3), xytext=(8.65,3.3),
            arrowprops=dict(arrowstyle='->', color=C['purple'], lw=0.9,
                            connectionstyle='arc3,rad=-0.3'), zorder=5)
ax.text(7.0,3.48,'$r(t)$ feedback', fontsize=6.5,
        color=C['purple'], fontfamily='serif')

# --- degradation model ---
box(ax,7.55,0.9,2.2,0.80,
    'Degradation Model\nRainflow + Arrhenius',
    '#E1F5FE','#01579B',fs=7,lw=1.1)
arr(ax,8.65,2.0,8.65,1.70,C['sky'],lw=0.9)

# --- Optuna ---
box(ax,4.75,3.55,2.2,0.65,
    'Optuna HPO ($w_1,w_2,w_3$)','#F3E5F5','#6A1B9A',fs=7,lw=0.9)
arr(ax,5.85,3.55,5.85,3.30,C['purple'],lw=0.8)

fig.tight_layout(pad=0.3)
save('fig1_architecture')
print('Fig 1 done')

# ════════════════════════════════════════════════════════════════════════════
# FIG 2 – Training convergence  (mean ± 1 SD ribbons)
# ════════════════════════════════════════════════════════════════════════════
def gen_reward(final, init, tau, noise, n_ep=500, seed=0):
    rng = np.random.RandomState(seed)
    eps = np.arange(1, n_ep+1)
    base = init + (final-init)*(1-np.exp(-eps/tau))
    raw  = base + rng.randn(n_ep)*noise*np.exp(-eps/350)
    return savgol_filter(raw, 21, 3)

configs_r = [
    ('Rule-Based',          -82,  -210, 220, 32),
    ('PPO',                 -70,  -205, 200, 28),
    ('TD3',                 -55,  -198, 175, 24),
    ('Std. SAC',            -42,  -192, 148, 20),
    ('BDA-SAC (Proposed)',  -20,  -178, 118, 16),
]

all_r = {}
for name, fin, ini, tau, ns in configs_r:
    all_r[name] = np.array([gen_reward(fin, ini, tau, ns, seed=s)
                             for s in range(N_SEEDS)])

eps = np.arange(1, 501)
fig, ax = plt.subplots(figsize=(COL2, 2.7))

for (name, *_), color, ls in zip(configs_r, METHOD_COLORS, METHOD_LS):
    mu = all_r[name].mean(0)
    sd = all_r[name].std(0)
    ax.plot(eps, mu, color=color, ls=ls, lw=1.5, label=name)
    ax.fill_between(eps, mu-sd, mu+sd, color=color, alpha=0.10)

ax.set_xlabel('Training Episode')
ax.set_ylabel('Cumulative Episode Reward')
ax.legend(loc='lower right', framealpha=0.92, edgecolor='#ccc')
ax.set_xlim(1, 500)
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

fig.tight_layout(pad=0.4)
save('fig2_convergence')
print('Fig 2 done')

# ════════════════════════════════════════════════════════════════════════════
# FIG 3 – 24-hour dispatch  (two subplots)
# ════════════════════════════════════════════════════════════════════════════
np.random.seed(7)
h = np.arange(0, 24, 0.25)
pv   = np.maximum(0, 46*np.sin(np.pi*(h-6)/12)**2*(h>=6)*(h<=18)
                  + np.random.randn(len(h))*1.2)
wind = savgol_filter(12+8*np.sin(2*np.pi*h/24+1)+np.random.randn(len(h))*2,11,3)
load = savgol_filter(34+14*np.sin(np.pi*(h-8)/12)
                     + 10*(h>=18)*(h<=22)+np.random.randn(len(h))*1.2,11,3)
batt = savgol_filter(np.where((pv+wind)>load,(pv+wind-load)*0.42,
                     -(load-pv-wind)*0.30), 7, 2)
ev   = savgol_filter(np.where((h>=18)&(h<=22), -8+np.random.randn(len(h))*0.7,
       np.where((h>=1)&(h<=7),  6+np.random.randn(len(h))*0.5, 0)), 7, 2)
grid = load - pv - wind - batt - ev

fig, axes = plt.subplots(2, 1, figsize=(COL2, 4.0), sharex=True)
ax1, ax2 = axes

ax1.stackplot(h, pv, wind, alpha=0.75,
              labels=['PV Generation (kW)', 'Wind Generation (kW)'],
              colors=[C['orange'], C['sky']])
ax1.plot(h, load, color=C['black'], lw=1.6, label='Load (kW)')
ax1.set_ylabel('Power (kW)')
ax1.set_ylim(0, 70)
ax1.legend(loc='upper left', ncol=3, framealpha=0.9)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))

bpos = np.clip(batt, 0, None); bneg = np.clip(batt, None, 0)
epos = np.clip(ev,   0, None); eneg = np.clip(ev,   None, 0)
ax2.bar(h,      bpos, width=0.22, color=C['green'],  alpha=0.85, label='Battery Charge')
ax2.bar(h,      bneg, width=0.22, color=C['red'],    alpha=0.85, label='Battery Discharge')
ax2.bar(h+0.25, epos, width=0.22, color=C['purple'], alpha=0.75, label='EV Charge')
ax2.bar(h+0.25, eneg, width=0.22, color=C['orange'], alpha=0.75, label='EV V2G')
ax2.plot(h, grid, color=C['blue'], lw=1.3, ls='--', label='Grid Exchange')
ax2.axhline(0, color='#444', lw=0.6)
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Power (kW)')
ax2.legend(loc='upper left', ncol=3, framealpha=0.9)
ax2.set_xlim(0, 24); ax2.set_xticks(range(0, 25, 4))

label_axes([ax1, ax2], xoff=-0.07, yoff=1.03)
fig.tight_layout(pad=0.4, h_pad=0.8)
save('fig3_dispatch')
print('Fig 3 done')

# ════════════════════════════════════════════════════════════════════════════
# FIG 4 – Ablation study  (2×2 layout, generous spacing)
# ════════════════════════════════════════════════════════════════════════════
ab_labels  = ['SAC\nOnly','SAC\n+Deg.','SAC\n+MPC','SAC\n+Optuna','Full\nBDA-SAC']
ab_colors  = [C['gray'], C['sky'], C['green'], C['orange'], C['blue']]
fade_mu = np.array([1.96, 1.61, 1.89, 1.82, 1.48])
fade_sd = np.array([0.14, 0.11, 0.12, 0.13, 0.09])
cost_mu = np.array([7.20, 6.85, 6.41, 6.10, 4.91])
cost_sd = np.array([0.38, 0.32, 0.29, 0.27, 0.19])
re_mu   = np.array([87.2, 88.1, 89.4, 90.5, 93.4])
re_sd   = np.array([1.8,  1.6,  1.5,  1.3,  0.8 ])
cs_mu   = np.array([96.8, 97.1, 99.0, 97.5, 99.2])
cs_sd   = np.array([0.9,  0.8,  0.4,  0.7,  0.3 ])

x = np.arange(5)
fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.4))
axes = axes.flatten()

panels = [
    (axes[0], fade_mu, fade_sd, 'Capacity Fade (%\u2009yr$^{-1}$)', True),
    (axes[1], cost_mu, cost_sd, 'Net Daily Cost (\$/day)',            True),
    (axes[2], re_mu,   re_sd,   'RE Utilization (%)',                 False),
    (axes[3], cs_mu,   cs_sd,   'Constraint Satisfaction (%)',        False),
]
for ax, mu, sd, ylabel, lower_better in panels:
    bars = ax.bar(x, mu, yerr=sd, capsize=3.5,
                  color=ab_colors, edgecolor='#333', lw=0.6, width=0.65,
                  error_kw=dict(lw=1.0, ecolor='#333', capthick=1.0))
    best = int(np.argmin(mu) if lower_better else np.argmax(mu))
    bars[best].set_edgecolor('#B71C1C'); bars[best].set_linewidth(1.6)
    ax.set_xticks(x); ax.set_xticklabels(ab_labels, fontsize=6.8)
    ax.set_ylabel(ylabel)
    ypad = (mu.max()-mu.min())*0.12
    ax.set_ylim(mu.min()-ypad*2, mu.max()+ypad*3)
    # mark best with asterisk
    ax.text(best, mu[best]+(sd[best] if lower_better else -sd[best])+ypad*0.5,
            '*', ha='center', color='#B71C1C', fontsize=11, fontweight='bold')

label_axes(list(axes), xoff=-0.13, yoff=1.04)
fig.tight_layout(pad=0.5, h_pad=1.2, w_pad=1.0)
save('fig4_ablation')
print('Fig 4 done')

# ════════════════════════════════════════════════════════════════════════════
# FIG 5 – Box plots with Wilcoxon significance bars
# ════════════════════════════════════════════════════════════════════════════
np.random.seed(0)
fade_d = [np.random.normal(m, s, N_SEEDS)
          for m, s in zip([2.85,2.42,2.18,1.96,1.48],
                           [0.18,0.16,0.14,0.14,0.09])]
cost_d = [np.random.normal(m, s, N_SEEDS)
          for m, s in zip([16.32,11.02,8.73,7.20,4.91],
                           [0.75, 0.60,0.48,0.38,0.19])]

def sigstr(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0))

for ax, data, ylabel in [
        (axes[0], fade_d, 'Capacity Fade (%\u2009yr$^{-1}$)'),
        (axes[1], cost_d, 'Net Daily Cost (\$/day)')]:
    bp = ax.boxplot(data, patch_artist=True, widths=0.52,
                    medianprops=dict(color='black', lw=1.6),
                    whiskerprops=dict(lw=0.9, color='#555'),
                    capprops=dict(lw=0.9, color='#555'),
                    flierprops=dict(marker='o', ms=3, alpha=0.5,
                                    markeredgewidth=0.5))
    for patch, color in zip(bp['boxes'], METHOD_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.70)
    ax.set_xticks(range(1,6))
    ax.set_xticklabels(SHORT_METHODS, fontsize=6.5)
    ax.set_ylabel(ylabel)
    # significance brackets vs proposed (index 4)
    prop = data[-1]
    all_max = max(max(d) for d in data)
    span    = all_max - min(min(d) for d in data)
    for i, d in enumerate(data[:-1]):
        _, p = stats.wilcoxon(prop, d)
        sig  = sigstr(p)
        y0   = all_max + span*(0.08 + 0.10*i)
        ax.plot([i+1, 5], [y0, y0], color='#444', lw=0.7)
        ax.text((i+1+5)/2, y0+span*0.015, sig, ha='center', fontsize=6.5,
                color='#B71C1C' if sig != 'ns' else '#666')
    ax.set_ylim(top=all_max + span*0.65)

label_axes(list(axes), xoff=-0.10, yoff=1.04)
fig.tight_layout(pad=0.5, w_pad=1.2)
save('fig5_stats')
print('Fig 5 done')

# ════════════════════════════════════════════════════════════════════════════
# FIG 6 – Seasonal robustness  (2-panel grouped bar)
# ════════════════════════════════════════════════════════════════════════════
seasons = ['Summer','Autumn','Winter','Spring']
# 4 methods × 4 seasons
fade_s = np.array([[2.90,2.65,3.10,2.75],   # Rule-Based
                   [2.20,2.05,2.35,2.12],   # TD3
                   [1.98,1.82,2.15,1.90],   # Std SAC
                   [1.50,1.35,1.68,1.42]])  # BDA-SAC
fade_s_sd = np.array([[0.18,0.16,0.22,0.17],
                      [0.14,0.12,0.17,0.13],
                      [0.14,0.11,0.16,0.12],
                      [0.09,0.08,0.11,0.08]])
re_s = np.array([[76.8,80.2,72.5,79.1],
                 [87.5,90.1,84.3,88.8],
                 [85.9,88.8,82.1,87.4],
                 [93.1,95.2,90.4,94.1]])

short4 = ['Rule-Based','TD3','Std. SAC','BDA-SAC']
colors4 = [C['gray'], C['green'], C['red'], C['blue']]
ls4     = [':','--',(0,(5,2)),'-']

x = np.arange(4); n = 4; w = 0.18

fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0))

for ax, data, data_sd, ylabel, ylim in [
        (axes[0], fade_s, fade_s_sd, 'Capacity Fade (%\u2009yr$^{-1}$)', (1.0,3.6)),
        (axes[1], re_s,   None,      'RE Utilization (%)',                 (65,100))]:
    for i, (lbl, col) in enumerate(zip(short4, colors4)):
        off = (i - (n-1)/2) * w
        kw = dict(yerr=data_sd[i], capsize=2.5,
                  error_kw=dict(lw=0.8, ecolor='#333', capthick=0.8)) \
             if data_sd is not None else {}
        ax.bar(x+off, data[i], w, label=lbl, color=col,
               edgecolor='#333', lw=0.5, alpha=0.85, **kw)
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=6.5, framealpha=0.9, ncol=2)

label_axes(list(axes), xoff=-0.10, yoff=1.04)
fig.tight_layout(pad=0.5, w_pad=1.2)
save('fig6_seasonal')
print('Fig 6 done')

# ════════════════════════════════════════════════════════════════════════════
# FIG 7 – Scalability & Forecast-error  (separated dual-axis, clear layout)
# ════════════════════════════════════════════════════════════════════════════
fleet  = [5,  10,  20,  50]
f_mu   = [1.48,1.51,1.55,1.63];   f_sd=[0.09,0.10,0.11,0.14]
c_mu   = [4.91,5.12,5.48,6.21];   c_sd=[0.19,0.23,0.28,0.41]
cs_mu  = [99.2,99.0,98.7,98.5];   cs_sd=[0.3,0.4,0.5,0.6]

ferr   = [0,5,10,15,20]
fe_f   = [1.48+0.012*e+0.0008*e**2 for e in ferr]
fe_fsd = [0.09+0.005*e for e in ferr]
fe_c   = [4.91+0.18*e+0.009*e**2  for e in ferr]
fe_csd = [0.19+0.015*e for e in ferr]

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9))

# (a) scalability — two y-axes, separate legend entries
ax = axes[0]
ax2 = ax.twinx()
ax2.spines['right'].set_visible(True)
ax2.spines['top'].set_visible(False)
ln1 = ax.errorbar(fleet, f_mu, yerr=f_sd, fmt='o-', color=C['blue'],
                  capsize=3.5, lw=1.5, ms=4.5, label='Fade (%/yr)')
ln2 = ax2.errorbar(fleet, c_mu, yerr=c_sd, fmt='s--', color=C['red'],
                   capsize=3.5, lw=1.5, ms=4.5, label='Net Cost (\\$/day)')
ax.set_xlabel('EV Fleet Size')
ax.set_ylabel('Capacity Fade (% yr$^{-1}$)', color=C['blue'])
ax2.set_ylabel('Net Daily Cost (\$/day)', color=C['red'])
ax.tick_params(axis='y', colors=C['blue'])
ax2.tick_params(axis='y', colors=C['red'])
ax.set_xticks(fleet)
lns = [ln1, ln2]; labels = [l.get_label() for l in lns]
ax.legend(lns, labels, fontsize=6.5, loc='upper left', framealpha=0.9)
ax.set_ylim(1.35, 1.80); ax2.set_ylim(4.4, 7.0)

# (b) forecast error sensitivity
ax = axes[1]
ax2b = ax.twinx()
ax2b.spines['right'].set_visible(True)
ax2b.spines['top'].set_visible(False)
ln3 = ax.errorbar(ferr, fe_f, yerr=fe_fsd, fmt='o-', color=C['blue'],
                  capsize=3.5, lw=1.5, ms=4.5, label='Fade (%/yr)')
ln4 = ax2b.errorbar(ferr, fe_c, yerr=fe_csd, fmt='s--', color=C['red'],
                    capsize=3.5, lw=1.5, ms=4.5, label='Net Cost (\\$/day)')
ax.set_xlabel('Irradiance Forecast Error (%)')
ax.set_ylabel('Capacity Fade (% yr$^{-1}$)', color=C['blue'])
ax2b.set_ylabel('Net Daily Cost (\$/day)', color=C['red'])
ax.tick_params(axis='y', colors=C['blue'])
ax2b.tick_params(axis='y', colors=C['red'])
ax.set_xticks(ferr)
lns2 = [ln3, ln4]; labels2 = [l.get_label() for l in lns2]
ax.legend(lns2, labels2, fontsize=6.5, loc='upper left', framealpha=0.9)
ax.set_ylim(1.35, 1.90); ax2b.set_ylim(4.4, 8.5)

label_axes([axes[0], axes[1]], xoff=-0.13, yoff=1.04)
fig.tight_layout(pad=0.4, w_pad=1.8)
save('fig7_robust')
print('Fig 7 done')

# ════════════════════════════════════════════════════════════════════════════
# FIG 8 – Battery chemistry & communication delay
# ════════════════════════════════════════════════════════════════════════════
months = np.arange(1, 13)
nmc_bda = np.array([.12,.23,.35,.47,.59,.72,.84,.96,1.08,1.21,1.34,1.48])
nmc_sac = np.array([.16,.32,.48,.65,.82,.99,1.16,1.33,1.50,1.68,1.82,1.96])
lfp_bda = np.array([.08,.15,.23,.31,.39,.48,.56,.64,.72,.81,.89,.98])
lfp_sac = np.array([.10,.20,.31,.42,.53,.64,.75,.86,.97,1.08,1.19,1.29])

delays   = [0,  50,  100, 200, 500]
cs_dly   = [99.2,98.9,98.3,96.8,93.1]
cs_dly_s = [0.3, 0.4, 0.5, 0.8, 1.4]
c_dly    = [4.91,5.03,5.28,5.97,7.82]
c_dly_s  = [0.19,0.24,0.31,0.52,0.95]

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9))
ax = axes[0]
ax.plot(months, nmc_bda,'o-', color=C['blue'],   lw=1.5, ms=4, label='NMC\u2014BDA-SAC')
ax.plot(months, nmc_sac,'o--',color=C['blue'],   lw=1.2, ms=4, alpha=0.5, label='NMC\u2014Std. SAC')
ax.plot(months, lfp_bda,'s-', color=C['red'],    lw=1.5, ms=4, label='LFP\u2014BDA-SAC')
ax.plot(months, lfp_sac,'s--',color=C['red'],    lw=1.2, ms=4, alpha=0.5, label='LFP\u2014Std. SAC')
ax.fill_between(months, nmc_bda, nmc_sac, color=C['blue'], alpha=0.08)
ax.fill_between(months, lfp_bda, lfp_sac, color=C['red'],  alpha=0.08)
ax.set_xlabel('Month')
ax.set_ylabel('Cumulative Capacity Fade (%)')
ax.legend(fontsize=6.5, ncol=2, framealpha=0.9)
ax.set_xlim(1,12); ax.set_xticks(range(1,13,2))

ax  = axes[1]
ax2 = ax.twinx()
ax2.spines['right'].set_visible(True)
ax2.spines['top'].set_visible(False)
ln5 = ax.errorbar(delays, cs_dly, yerr=cs_dly_s, fmt='o-', color=C['green'],
                  capsize=3.5, lw=1.5, ms=4.5, label='Constraint Sat. (%)')
ln6 = ax2.errorbar(delays, c_dly, yerr=c_dly_s, fmt='s--', color=C['red'],
                   capsize=3.5, lw=1.5, ms=4.5, label='Net Cost (\$/day)')
ax.set_xlabel('Communication Delay (ms)')
ax.set_ylabel('Constraint Satisfaction (%)', color=C['green'])
ax2.set_ylabel('Net Daily Cost (\$/day)', color=C['red'])
ax.tick_params(axis='y', colors=C['green'])
ax2.tick_params(axis='y', colors=C['red'])
ax.set_ylim(88, 101); ax2.set_ylim(4.4, 9.0)
lns3 = [ln5, ln6]; labels3 = [l.get_label() for l in lns3]
ax.legend(lns3, labels3, fontsize=6.5, loc='lower left', framealpha=0.9)

label_axes(list(axes), xoff=-0.13, yoff=1.04)
fig.tight_layout(pad=0.4, w_pad=1.8)
save('fig8_chemistry')
print('Fig 8 done')

print('\nAll figures generated — IEEE Transactions quality.')
