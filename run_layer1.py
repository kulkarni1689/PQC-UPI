"""Run all Layer 1 experiments (E1-E3) and write results/layer1.json"""
import json, math, time, sys
import numpy as np
from scipy import stats
from dataclasses import asdict, replace
from layer1_sim import *

P = Params()
OUT = 'results/layer1.json'
R = {'params': asdict(P)}

def save(): json.dump(R, open(OUT, 'w'), indent=1)

def agg(runs):
    keys = ['p50', 'p95', 'p99', 'mean', 'fail_rate', 'slow_share', 'full_hybrid_share']
    d = {}
    for k in keys:
        v = np.array([r[k] for r in runs], float)
        if np.all(np.isnan(v)): d[k] = dict(mean=None, ci=None); continue
        m = float(np.nanmean(v)); n = np.sum(~np.isnan(v))
        ci = float(stats.t.ppf(0.975, n - 1) * np.nanstd(v, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        d[k] = dict(mean=m, ci=ci)
    d['util'] = {p: float(np.mean([r['util'][p] for r in runs])) for p in POOLS}
    d['wait_ms'] = {p: float(np.mean([r['wait_mean'][p] for r in runs]) * 1e3) for p in POOLS}
    d['reps'] = len(runs)
    return d

def reps(cfg, load, units, n, ready=None, seed0=1000, PP=None):
    PP = PP or P
    return agg([run_sim(PP, cfg, load, units, seed0 + i, ready) for i in range(n)])

t0 = time.time()
# ---- E1: validation
R['E1_mmc'] = validate_mmc(n=300000); save()
R['E1_pooling'] = pooling_bias(P); save()
print('E1 done', time.time() - t0, flush=True)

# ---- capacity table (analytic) 
R['capacity'] = capacity_table(P); save()
u0 = baseline_units(P)
uC = {c: R['capacity'][c]['units'] for c in ('C0', 'C1', 'C2')}
loads = {'normal': P.mult_normal, 'peak': P.mult_peak}

# ---- E2: upgraded capacity (utilisation restored to design level)
R['E2_upgraded'] = {}
for cfg in ('C0', 'C1', 'C2'):
    for ln, lm in loads.items():
        R['E2_upgraded'][f'{cfg}_{ln}'] = reps(cfg, lm, uC[cfg], 5); save()
        print('E2 up', cfg, ln, round(R['E2_upgraded'][f'{cfg}_{ln}']['p99']['mean'], 4), time.time() - t0, flush=True)

# ---- E2: capacity cliff (hybrid, peak): provisioned capacity as a fraction of required
R['E2_cliff'] = {}
for phi in (1.0, 0.9, 0.8, 0.7, 0.6):
    u = {k: max(1, int(math.ceil(phi * uC['C1'][k]))) for k in POOLS}
    R['E2_cliff'][str(phi)] = reps('C1', P.mult_peak, u, 3); save()
    print('cliff', phi, R['E2_cliff'][str(phi)]['fail_rate']['mean'], time.time() - t0, flush=True)

# ---- E2: no capacity upgrade (baseline units) 
R['E2_noupgrade'] = {}
for cfg in ('C1', 'C2'):
    for ln, lm in loads.items():
        R['E2_noupgrade'][f'{cfg}_{ln}'] = reps(cfg, lm, u0, 2); save()
        print('noupg', cfg, ln, R['E2_noupgrade'][f'{cfg}_{ln}']['fail_rate']['mean'], time.time() - t0, flush=True)

# ---- E2: mixed-state ecosystem (NPCI ready; fraction f of PSP/banks ready), capacity matched to expected work
R['E2_mixed'] = {}
w0, w1 = work_per_txn(P, 'C0'), work_per_txn(P, 'C1')
for f in (0.25, 0.5, 0.75):
    lam = P.avg_tps * P.mult_peak; r0 = rho0(P)
    u = {k: int(math.ceil(((1 - f) * w0[k] + f * w1[k]) * lam / r0[k])) for k in POOLS}
    R['E2_mixed'][str(f)] = reps('MIX', P.mult_peak, u, 3, ready=f); R['E2_mixed'][str(f)]['units'] = u; save()
    print('mixed', f, time.time() - t0, flush=True)

# ---- OAT sensitivity of Layer 1 (hybrid, upgraded capacity, festival peak): p99 and capacity ratio at NPCI
variants = {
 'ML-DSA-44 sign time (80 / 200 us)': [('t_mldsa_sign', 80.0), ('t_mldsa_sign', 200.0)],
 'HSM PQC penalty (1.0 / 3.0)':       [('pqc_penalty', 1.0), ('pqc_penalty', 3.0)],
 'HSM slowdown factor (8 / 30)':      [('hsm_factor', 8.0), ('hsm_factor', 30.0)],
 'Handset factor (3 / 10)':           [('dev_factor', 3.0), ('dev_factor', 10.0)],
 'Mobile uplink (20 / 2 Mbps)':       [('mobile_up_bps', 20e6), ('mobile_up_bps', 2e6)],
 'Sender key in every message':       [('pk_in_msg', False), ('pk_in_msg', True)],
 'Festival multiple (2 / 4->6)':      [('mult_peak', 2.0), ('mult_peak', 6.0)],
}
R['L1_oat'] = {}
for name, pair in variants.items():
    row = []
    for (k, v) in pair:
        PP = replace(P, **{k: v})
        ub = baseline_units(PP); u1 = units_full(PP, work_per_txn(PP, 'C1'))
        lm = PP.mult_peak
        r = reps('C1', lm, u1, 3, PP=PP)
        row.append(dict(param=k, value=v, p99=r['p99']['mean'], p50=r['p50']['mean'],
                        ratio_npci=u1['npci'] / ub['npci'], ratio_psp=u1['psp'] / ub['psp'],
                        ratio_bankS=u1['bankS'] / ub['bankS'], extra_total=sum(u1.values()) - sum(ub.values())))
    R['L1_oat'][name] = row; save(); print('oat', name, time.time() - t0, flush=True)
print('ALL DONE', time.time() - t0, flush=True)
