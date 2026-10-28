"""
Layer 2: agent-based model of institution-level migration timing under uncertain
quantum-attacker arrival (T_q), harvest-now-decrypt-later (HNDL) exposure and network
externalities.  Layer 1 supplies the capacity (extra HSM units) and latency cost inputs.
Monetary values are in INR crore and are ILLUSTRATIVE (see paper, Section 5 and 9).
"""
import math, json, os
from dataclasses import dataclass, replace, asdict
import numpy as np

QUARTERS_PER_YEAR = 4
CLS = ['Large banks', 'Small banks', 'PSPs']

@dataclass
class L2:
    T: int = 100                       # quarters (25 years from Q4 2026)
    nL: int = 10; nS: int = 60; nP: int = 8
    share_bank_L: float = 0.65         # share of bank-side traffic handled by large banks
    w_bank: float = 0.5                # weight of the bank side in loss exposure (rest = PSP side)
    annual_txn: float = 2.4162e11
    # attacker arrival: Weibull(shape k, scale from median years)
    k_shape: float = 2.5
    belief_median: float = 12.0        # agents' common central belief (years)
    belief_sigma: float = 0.25         # idiosyncratic multiplicative belief noise (log-sd)
    truth_median: float = 12.0         # set by scenario
    # loss scale (INR crore, illustrative)
    loss_scale: float = 1.0
    D_tot: float = 8000.0              # direct loss if the whole sector were unprotected at T_q
    S_tot: float = 16000.0             # systemic (spill-over) loss, same condition
    hndl_ratio: float = 0.5            # HNDL loss relative to direct loss
    a_own: float = 0.3                 # own-side protection that does not need counterparties
    omega: float = 0.5                 # weight of NPCI hop-by-hop readiness in protection
    theta: float = 0.3                 # share of systemic loss internalised by agents
    # costs (INR crore)
    unit_cost: float = 1.5             # per extra HSM unit, fully loaded (HA pair, licences, rack, integration)
    fixed_L: float = 120.0; fixed_S: float = 12.0; fixed_P: float = 80.0
    fixed_sigma: float = 0.3
    opex_rate: float = 0.10; opex_years: float = 5.0
    cost_decline: float = 0.05         # annual decline of capex
    perf_cost_inr_per_ms: float = 1e-4 # INR per millisecond of extra latency per transaction
    npci_fixed: float = 150.0
    disc: float = 0.05
    lam_fiscal: float = 0.20           # shadow cost of public funds
    # behaviour
    p_eval: float = 0.25               # prob. an agent reviews its decision in a quarter
    lead_scale: float = 1.0
    tN: int = 16                       # quarter in which NPCI becomes ready (exogenous)
    # Layer 1 inputs (filled from Layer 1)
    extra_bankL: float = 0.0; extra_bankS: float = 0.0; extra_psp: float = 0.0; extra_npci: float = 0.0
    dlat_ms: float = 12.0
    # policy
    q_ann: int = 2
    D_mand: int = 28                   # mandate deadline (quarter index; 28 = Q4 2033)
    penalty_mult: float = 1.0          # penalty = penalty_mult * CAPEX at announcement
    enforce: float = 0.7
    subsidy: float = 0.4

SCEN = {'early': 7.0, 'central': 12.0, 'late': 20.0}

def layer1_inputs(path='results/layer1.json'):
    from layer1_sim import Params, capacity_table
    tab = capacity_table(Params())['C1']['extra']
    dl = 12.0
    if os.path.exists(path):
        R = json.load(open(path))
        try:
            dl = 1e3 * (R['E2_upgraded']['C1_normal']['mean']['mean'] - R['E2_upgraded']['C0_normal']['mean']['mean'])
        except Exception: pass
    return dict(extra_bankL=tab['bankL'], extra_bankS=tab['bankS'], extra_psp=tab['psp'], extra_npci=tab['npci'], dlat_ms=dl)

# ------------------------------------------------------------------ population
def draw_population(rng, p):
    n = p.nL + p.nS + p.nP
    cls = np.array([0] * p.nL + [1] * p.nS + [2] * p.nP)
    wL = rng.lognormal(0, 0.5, p.nL); wL = wL / wL.sum() * p.share_bank_L
    wS = rng.lognormal(0, 0.8, p.nS); wS = wS / wS.sum() * (1 - p.share_bank_L)
    z = 1.0 / np.arange(1, p.nP + 1) ** 1.5; wP = z / z.sum()
    v = np.concatenate([wL, wS, wP])                         # traffic share (banks: of bank side; PSPs: of PSP side)
    w = np.where(cls < 2, p.w_bank * v, (1 - p.w_bank) * v)  # exposure weight (sums to 1)
    D = p.D_tot * p.loss_scale * w; S = p.S_tot * p.loss_scale * w
    extra = np.where(cls == 0, p.extra_bankL * v / p.share_bank_L,
             np.where(cls == 1, p.extra_bankS * v / (1 - p.share_bank_L), p.extra_psp * v))
    fixed = np.where(cls == 0, p.fixed_L, np.where(cls == 1, p.fixed_S, p.fixed_P)) * rng.lognormal(0, p.fixed_sigma, n)
    capex0 = extra * p.unit_cost + fixed
    lead = np.where(cls == 0, rng.integers(6, 11, n), np.where(cls == 1, rng.integers(8, 17, n), rng.integers(4, 9, n)))
    lead = np.maximum(1, np.round(lead * p.lead_scale)).astype(int)
    tau = np.where(cls < 2, 7.0, 5.0) * rng.lognormal(0, 0.3, n)
    bscale = p.belief_median / (math.log(2) ** (1 / p.k_shape)) * rng.lognormal(0, p.belief_sigma, n)
    ntx = p.annual_txn * v * np.where(cls < 2, 2.0, 1.0)
    perf = ntx * p.dlat_ms * p.perf_cost_inr_per_ms / 1e7    # INR crore per year
    return dict(n=n, cls=cls, v=v, w=w, D=D, S=S, capex0=capex0, lead=lead, tau=tau, bscale=bscale, perf=perf)

# ------------------------------------------------------------------ one run
def simulate(pop, U, p, reg):
    n, T = pop['n'], p.T
    kind = reg.get('kind', 'P0')
    s_sub = reg.get('s', p.subsidy) if kind == 'P2' else 0.0
    Dq = reg.get('D', p.D_mand)
    qready = np.full(n, 10 ** 6); committed = np.zeros(n, bool)
    capex_commit = np.zeros(n); subsidy_paid = np.zeros(n); qcommit = np.full(n, -1)
    k = p.k_shape; w = pop['w']; L = pop['lead']
    for q in range(T):
        rdy = qready <= q
        m = float((w * rdy).sum())
        cand = (~committed) & (U[q] < p.p_eval)
        if not cand.any(): continue
        ty = q / QUARTERS_PER_YEAR
        dt = L / QUARTERS_PER_YEAR + pop['tau']
        sb = pop['bscale']
        phi = 1.0 - np.exp(-(((ty + dt) / sb) ** k) + ((ty / sb) ** k))
        Nf = (q + L >= p.tN).astype(float)
        rho = p.omega * Nf + (1 - p.omega) * m
        g = p.a_own + (1 - p.a_own) * rho
        B = phi * (pop['D'] * (1 + p.hndl_ratio) * g + p.theta * pop['S'] * (1 - rho))
        capex_now = pop['capex0'] * math.exp(-p.cost_decline * ty)
        s_act = s_sub if (kind == 'P2' and q >= p.q_ann) else 0.0
        cost = (1 - s_act) * capex_now + p.opex_years * (p.opex_rate * capex_now + pop['perf'])
        if kind == 'P1' and q >= p.q_ann:
            frac = np.clip(1.0 - np.maximum(0, q + L - Dq) / 8.0, 0.0, 1.0)
            B = B + p.enforce * p.penalty_mult * pop['capex0'] * frac
        com = cand & (B >= cost)
        if com.any():
            qready[com] = q + L[com]; committed |= com; qcommit[com] = q
            capex_commit[com] = capex_now[com]; subsidy_paid[com] = s_act * capex_now[com]
    return dict(qready=qready, qcommit=qcommit, capex=capex_commit, subsidy=subsidy_paid)

def outcomes(pop, sim, Tq_years, p):
    n, T = pop['n'], p.T
    qs = np.arange(T)
    ready = sim['qready'][None, :] <= qs[:, None]                     # T x n
    m = (ready * pop['w'][None, :]).sum(1)                          # sector readiness over time
    Nq = (qs >= p.tN).astype(float)
    rho = p.omega * Nq + (1 - p.omega) * m
    G = ready * (p.a_own + (1 - p.a_own) * rho[:, None])
    cls = pop['cls']
    curves = np.stack([(ready[:, cls == c] * pop['v'][cls == c]).sum(1) / pop['v'][cls == c].sum() for c in range(3)], 1)
    hit = np.where(m >= 0.9)[0]
    t90 = (hit[0] / QUARTERS_PER_YEAR) if len(hit) else np.inf
    # costs (present value at t=0)
    DF = (1 + p.disc) ** (-qs / QUARTERS_PER_YEAR)
    cumDF = np.concatenate([[0.0], np.cumsum(DF)])
    done = sim['qcommit'] >= 0
    pv_capex = float((sim['capex'] * DF[np.maximum(sim['qcommit'], 0)] * done).sum())
    pv_subsidy = float((sim['subsidy'] * DF[np.maximum(sim['qcommit'], 0)] * done).sum())
    run = (p.opex_rate * sim['capex'] + pop['perf']) / QUARTERS_PER_YEAR
    qr = np.minimum(sim['qready'], T)
    pv_run = float((run * (cumDF[T] - cumDF[qr]) * done).sum())
    npci = (p.extra_npci * p.unit_cost + p.npci_fixed) * DF[min(p.tN, T - 1)]
    # losses
    qt = int(Tq_years * QUARTERS_PER_YEAR)
    if qt >= T:
        loss = 0.0; cm_before = np.isfinite(t90)
    else:
        direct = float((pop['D'] * (1 - G[qt])).sum())
        sysl = float(pop['S'].sum() * (1 - rho[qt]))
        hn = 0.0
        for i in range(n):
            lo = max(0, qt - int(math.ceil(pop['tau'][i] * QUARTERS_PER_YEAR)))
            hn += p.hndl_ratio * pop['D'][i] * float((1 - G[lo:qt + 1, i]).mean())
        loss = (direct + sysl + hn) * DF[qt]
        cm_before = bool(t90 * QUARTERS_PER_YEAR < qt)
    cost_real = pv_capex + pv_run + npci
    social = cost_real + loss + p.lam_fiscal * pv_subsidy
    return dict(t90=t90, cm_before=cm_before, cost=cost_real, loss=loss, social=social,
                fiscal=pv_subsidy, curves=curves, m_at_D=float(m[min(T - 1, p.D_mand)]), m_end=float(m[-1]))

REGIMES = {'P0': dict(kind='P0'), 'P1': dict(kind='P1'), 'P2': dict(kind='P2')}

def run_mc(p, regimes=('P0', 'P1', 'P2'), scenario='central', N=300, seed=7, reg_over=None):
    p = replace(p, truth_median=SCEN[scenario])
    out = {r: [] for r in regimes}
    scale_t = p.truth_median / (math.log(2) ** (1 / p.k_shape))
    for i in range(N):
        rng = np.random.default_rng(seed + i)
        pop = draw_population(rng, p)
        U = rng.random((p.T, pop['n']))
        Tq = scale_t * rng.weibull(p.k_shape)
        for r in regimes:
            reg = dict(REGIMES[r]); 
            if reg_over and r in reg_over: reg.update(reg_over[r])
            sim = simulate(pop, U, p, reg)
            out[r].append(outcomes(pop, sim, Tq, p))
    return out

def summarise(res, key, q=None):
    v = np.array([r[key] for r in res], float)
    return v

def stat(v, ci=True):
    v = np.asarray(v, float); fin = v[np.isfinite(v)]
    return dict(mean=float(fin.mean()) if len(fin) else float('nan'),
                se=float(fin.std(ddof=1) / math.sqrt(len(fin))) if len(fin) > 1 else 0.0,
                median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                censored=float((~np.isfinite(v)).mean()))

if __name__ == '__main__':
    import time
    p = L2(**layer1_inputs())
    print(asdict(p)['extra_bankL'], p.dlat_ms)
    t = time.time()
    for sc in SCEN:
        res = run_mc(p, scenario=sc, N=60)
        for r in res:
            print(sc, r, 't90 med %.1f' % np.median([x['t90'] for x in res[r]]),
                  'cens %.2f' % np.mean([not np.isfinite(x['t90']) for x in res[r]]),
                  'cmB %.2f' % np.mean([x['cm_before'] for x in res[r]]),
                  'cost %.0f loss %.0f soc %.0f fisc %.0f' % tuple(np.mean([x[k] for x in res[r]]) for k in ('cost', 'loss', 'social', 'fiscal')),
                  'mD %.2f mEnd %.2f' % (np.mean([x['m_at_D'] for x in res[r]]), np.mean([x['m_end'] for x in res[r]])))
    print('secs', time.time() - t)
