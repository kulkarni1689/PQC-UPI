"""Run Layer 2 experiments (E3 policy comparison, sweeps, E4 sensitivity, limiting cases) -> results/layer2.json"""
import json, math, sys, time
import numpy as np
from dataclasses import asdict, replace
from layer2_abm import *
import layer1_sim as l1

N_OAT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
OUT = 'results/oat_v2.json'

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

R = {'N_OAT': N_OAT, 'params': asdict(p0)} if False else None
# ---------------- E4 one-at-a-time sensitivity (central scenario)
def scale_fixed(p, f): return replace(p, fixed_L=p.fixed_L * f, fixed_S=p.fixed_S * f, fixed_P=p.fixed_P * f)
OAT = {
 'Loss scale (x0.5 / x2)':              (lambda p: replace(p, loss_scale=0.5), lambda p: replace(p, loss_scale=2.0)),
 'Systemic internalisation theta (0.1 / 0.6)': (lambda p: replace(p, theta=0.1), lambda p: replace(p, theta=0.6)),
 'Own-side protection a (0.15 / 0.5)':  (lambda p: replace(p, a_own=0.15), lambda p: replace(p, a_own=0.5)),
 'NPCI weight omega (0.3 / 0.7)':       (lambda p: replace(p, omega=0.3), lambda p: replace(p, omega=0.7)),
 'HNDL ratio (0.2 / 1.0)':              (lambda p: replace(p, hndl_ratio=0.2), lambda p: replace(p, hndl_ratio=1.0)),
 'Fixed integration cost (x0.5 / x2)':  (lambda p: scale_fixed(p, 0.5), lambda p: scale_fixed(p, 2.0)),
 'HSM unit cost (INR 0.5 / 3.0 cr)':    (lambda p: replace(p, unit_cost=0.5), lambda p: replace(p, unit_cost=3.0)),
 'HSM PQC penalty, Layer 1 (1.0 / 3.0)': (lambda p: replace(p, **l1_inputs_from(replace(l1.Params(), pqc_penalty=1.0))),
                                          lambda p: replace(p, **l1_inputs_from(replace(l1.Params(), pqc_penalty=3.0)))),
 'ML-DSA sign time, Layer 1 (80 / 200 us)': (lambda p: replace(p, **l1_inputs_from(replace(l1.Params(), t_mldsa_sign=80.0))),
                                             lambda p: replace(p, **l1_inputs_from(replace(l1.Params(), t_mldsa_sign=200.0)))),
 'Latency cost (INR 1e-5 / 1e-3 per ms)': (lambda p: replace(p, perf_cost_inr_per_ms=1e-5), lambda p: replace(p, perf_cost_inr_per_ms=1e-3)),
 'Capex decline (0 / 10 % per year)':   (lambda p: replace(p, cost_decline=0.0), lambda p: replace(p, cost_decline=0.10)),
 'Lead time (x0.5 / x1.5)':             (lambda p: replace(p, lead_scale=0.5), lambda p: replace(p, lead_scale=1.5)),
 'Review probability (0.1 / 0.5)':      (lambda p: replace(p, p_eval=0.1), lambda p: replace(p, p_eval=0.5)),
 'Belief heterogeneity (0 / 0.5)':      (lambda p: replace(p, belief_sigma=0.0), lambda p: replace(p, belief_sigma=0.5)),
 'NPCI ready quarter (8 / 28)':         (lambda p: replace(p, tN=8), lambda p: replace(p, tN=28)),
 'Discount rate (3 % / 8 %)':           (lambda p: replace(p, disc=0.03), lambda p: replace(p, disc=0.08)),
 'Weibull shape of T_q (1.5 / 4.0)':    (lambda p: replace(p, k_shape=1.5), lambda p: replace(p, k_shape=4.0)),
 'Mandate penalty multiple (0.5 / 2.0)': (lambda p: replace(p, penalty_mult=0.5), lambda p: replace(p, penalty_mult=2.0)),
 'Subsidy rate (0.2 / 0.6)':            (lambda p: replace(p, subsidy=0.2), lambda p: replace(p, subsidy=0.6)),
 'Mandate deadline (Q4 2031 / Q4 2035)': (lambda p: replace(p, D_mand=20), lambda p: replace(p, D_mand=36)),
}

def metrics(p, sc='central'):
    res = run_mc(p, scenario=sc, N=N_OAT, seed=7)
    s = summarise_cell(res)
    out = dict(social_P0=s['P0']['social'], t90_P0=s['P0']['t90_median'], cm_P0=s['P0']['cm_before'])
    for r in ('P1', 'P2'):
        d = s[r]['d_social_vs_P0']; out['d_'+r] = d['mean']; out['se_'+r] = d['se']
    return out
t0 = time.time()
import os
if os.path.exists(OUT) and json.load(open(OUT)).get('N_OAT') == N_OAT:
    R = json.load(open(OUT))
else:
    R = {'N_OAT': N_OAT, 'seed': 7, 'scenario': 'central'}
    R['oat_base'] = metrics(p0); R['oat'] = {}
    json.dump(clean(R), open(OUT, 'w'), indent=1)
print('base', R['oat_base'], flush=True)
for name, (lo, hi) in OAT.items():
    if name in R['oat']: continue
    R['oat'][name] = dict(low=metrics(lo(p0)), high=metrics(hi(p0)))
    json.dump(clean(R), open(OUT, 'w'), indent=1)
    print('oat', name, time.time() - t0, flush=True)
print('ALL DONE', time.time()-t0, flush=True)
