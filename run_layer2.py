"""Run Layer 2 experiments (E3 policy comparison, sweeps, E4 sensitivity, limiting cases) -> results/layer2.json"""
import json, math, sys, time
import numpy as np
from dataclasses import asdict, replace
from layer2_abm import *
import layer1_sim as l1

N_MAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 600
N_SW = int(sys.argv[2]) if len(sys.argv) > 2 else 300
OUT = 'results/layer2.json'

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if (math.isinf(o) or math.isnan(o)) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o

def paired(res, a, b, key='social'):
    d = np.array([x[key] for x in res[a]]) - np.array([x[key] for x in res[b]])
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(len(d))))

def summarise_cell(res, with_curves=False):
    out = {}
    for r, lst in res.items():
        t90 = np.array([x['t90'] for x in lst])
        o = dict(t90_median=float(np.median(t90)) if np.isfinite(np.median(t90)) else None,
                 t90_p10=float(np.percentile(t90, 10)), t90_p90=float(np.percentile(t90, 90)),
                 censored=float((~np.isfinite(t90)).mean()),
                 cm_before=float(np.mean([x['cm_before'] for x in lst])),
                 m_at_D=float(np.mean([x['m_at_D'] for x in lst])))
        for k in ('cost', 'loss', 'social', 'fiscal'):
            v = np.array([x[k] for x in lst]); o[k] = float(v.mean()); o[k + '_se'] = float(v.std(ddof=1) / math.sqrt(len(v)))
        if with_curves: o['curves'] = np.mean([x['curves'] for x in lst], axis=0)
        out[r] = o
    if 'P0' in res:
        for r in res:
            if r != 'P0': out[r]['d_social_vs_P0'] = paired(res, r, 'P0')
    return out

def l1_inputs_from(Pm):
    tab = l1.capacity_table(Pm)['C1']['extra']
    return dict(extra_bankL=tab['bankL'], extra_bankS=tab['bankS'], extra_psp=tab['psp'], extra_npci=tab['npci'])

base_in = layer1_inputs()
p0 = L2(**base_in)
R = {'params': asdict(p0), 'layer1_inputs': base_in}
t0 = time.time()
def save(): json.dump(clean(R), open(OUT, 'w'), indent=1)

# ---------------- main policy comparison
R['main'] = {}
for sc in SCEN:
    res = run_mc(p0, scenario=sc, N=N_MAIN, seed=7)
    R['main'][sc] = summarise_cell(res, with_curves=True)
    save(); print('main', sc, time.time() - t0, flush=True)

# ---------------- policy-parameter sweeps (all three scenarios)
def sweep(name, values, mod, regimes, reg_over_fn=None):
    R.setdefault('sweeps', {})[name] = {}
    for v in values:
        pv = mod(p0, v)
        cell = {}
        for sc in SCEN:
            ro = reg_over_fn(v) if reg_over_fn else None
            cell[sc] = summarise_cell(run_mc(pv, regimes=regimes, scenario=sc, N=N_SW, seed=7, reg_over=ro))
        R['sweeps'][name][str(v)] = cell
    save(); print('sweep', name, time.time() - t0, flush=True)

sweep('deadline', [20, 24, 28, 32, 36, 40], lambda p, v: p, ('P0', 'P1'), lambda v: {'P1': {'D': v}})
sweep('subsidy', [0.1, 0.2, 0.4, 0.6, 0.8], lambda p, v: p, ('P0', 'P2'), lambda v: {'P2': {'s': v}})
sweep('theta', [0.0, 0.15, 0.3, 0.5, 0.8], lambda p, v: replace(p, theta=v), ('P0', 'P1', 'P2'))
sweep('loss_scale', [0.25, 0.5, 1.0, 2.0, 4.0], lambda p, v: replace(p, loss_scale=v), ('P0', 'P1', 'P2'))
sweep('npci_ready', [8, 16, 24, 32], lambda p, v: replace(p, tN=v), ('P0', 'P1', 'P2'))

# (The one-at-a-time sensitivity is in run_oat.py -> results/oat_v2.json)

def scale_fixed(p, f): return replace(p, fixed_L=p.fixed_L * f, fixed_S=p.fixed_S * f, fixed_P=p.fixed_P * f)

# ---------------- limiting-case verification
lim = {}
def lc(name, p, regimes=('P0',), sc='central', n=100, ro=None):
    res = run_mc(p, regimes=regimes, scenario=sc, N=n, seed=11, reg_over=ro)
    s = summarise_cell(res)
    lim[name] = {r: dict(t90_median=s[r]['t90_median'], censored=s[r]['censored'], social=s[r]['social'],
                         m_end=float(np.mean([x['m_end'] for x in res[r]]))) for r in res}
lc('No loss (loss_scale=0): nobody should migrate under P0', replace(p0, loss_scale=0.0))
lc('Huge loss (loss_scale=1000): everybody migrates at once', replace(p0, loss_scale=1000.0))
lc('No migration cost (fixed=0, units free): everybody migrates at once', replace(scale_fixed(p0, 0.0), unit_cost=0.0))
lc('theta=0 (no externality internalised)', replace(p0, theta=0.0))
lc('theta=1 (full internalisation)', replace(p0, theta=1.0))
lc('omega=1, NPCI ready at t=0 (protection hinges on NPCI only)', replace(p0, omega=1.0, tN=0))
lc('Full subsidy (s=1) under P2', p0, regimes=('P0', 'P2'), n=100, ro={'P2': {'s': 1.0}})
R['limiting'] = lim
save(); print('ALL DONE', time.time() - t0, flush=True)
