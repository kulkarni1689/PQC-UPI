"""
Layer 1: discrete-event simulation of a UPI-inspired transaction path under
classical (C0), hybrid (C1) and PQC-only (C2) cryptography.

Python 3.12, SimPy 4.x, NumPy.  All timings are REPRESENTATIVE values (see
PARAMETERS) and must be replaced by cited liboqs/SUPERCOP/HSM-vendor figures.
"""
import math, json, time, sys
from dataclasses import dataclass, asdict, replace
import numpy as np
import simpy

# ------------------------------------------------------------------ sizes (bytes)
# FIPS 203 / FIPS 204 and standard curve encodings
SIZE = dict(ecdsa_sig=64, ecdsa_pk=64, ecdh_pt=65,
            mlkem_pk=1184, mlkem_ct=1088, mldsa_pk=1312, mldsa_sig=2420)

# ------------------------------------------------------------------ parameters
@dataclass
class Params:
    # workload
    annual_txn: float = 2.4162e11             # FY2025-26 volume (24,162 crore)
    mult_normal: float = 1.5                  # busy-hour multiple of annual average
    mult_peak: float = 4.0                    # festival peak multiple (assumption 2-4x)
    # software timings, microseconds, one core (representative; see text)
    t_ecdsa_sign: float = 30.0
    t_ecdsa_verify: float = 80.0
    t_ecdh: float = 45.0                      # one scalar multiplication
    t_mldsa_sign: float = 120.0
    t_mldsa_verify: float = 35.0
    t_mlkem_enc: float = 25.0
    t_mlkem_dec: float = 30.0
    # device / HSM scaling
    dev_factor: float = 5.0                   # handset vs server core
    hsm_factor: float = 15.0                  # HSM private-key op vs software
    pqc_penalty: float = 1.5                  # extra HSM slowdown for PQC firmware
    # message sizes
    payload: int = 1200
    pk_in_msg: bool = False                   # include sender public key in each message
    # network
    mobile_lat_ms: float = 35.0
    inter_lat_ms: float = 8.0
    mobile_up_bps: float = 8e6
    mobile_down_bps: float = 20e6
    inter_bps: float = 1e9
    core_rem_ms: float = 120.0                # remitter core-banking processing (median)
    core_ben_ms: float = 80.0
    # service-level
    t_out: float = 3.0                        # s, abandon threshold
    slo: float = 1.0                          # s, slow-transaction threshold
    # tier shares (bank-side traffic) and design utilisation at festival peak
    share_bankL: float = 0.65
    rho0_npci: float = 0.60
    rho0_bankL: float = 0.60
    rho0_bankS: float = 0.80
    rho0_psp: float = 0.65
    # simulation
    scale_k: int = 10
    warm: float = 2.0
    meas: float = 10.0
    svc_sigma: float = 0.10                   # lognormal sigma of service times

    @property
    def avg_tps(self): return self.annual_txn / (365 * 86400)

POOLS = ['psp', 'npci', 'bankL', 'bankS']

# ------------------------------------------------------------------ op costs
def sw_us(P, cfg, op):
    """software time in microseconds split into (classical_part, pqc_part)"""
    if op == 'sign':       c, q = P.t_ecdsa_sign, P.t_mldsa_sign
    elif op == 'verify':   c, q = P.t_ecdsa_verify, P.t_mldsa_verify
    elif op == 'kex_init': c, q = 2 * P.t_ecdh, P.t_mlkem_enc
    elif op == 'kex_resp': c, q = P.t_ecdh, P.t_mlkem_dec
    else: raise ValueError(op)
    return {'C0': (c, 0.0), 'C1': (c, q), 'C2': (0.0, q)}[cfg]

def hsm_s(P, cfg, op):
    c, q = sw_us(P, cfg, op)
    return P.hsm_factor * (c + P.pqc_penalty * q) * 1e-6

def dev_s(P, cfg, op):
    c, q = sw_us(P, cfg, op)
    return P.dev_factor * (c + q) * 1e-6

def swsrv_s(P, cfg, op):
    c, q = sw_us(P, cfg, op)
    return (c + q) * 1e-6

def msg_bytes(P, cfg, cred):
    sig = {'C0': SIZE['ecdsa_sig'], 'C1': SIZE['ecdsa_sig'] + SIZE['mldsa_sig'], 'C2': SIZE['mldsa_sig']}[cfg]
    kex = {'C0': SIZE['ecdh_pt'], 'C1': SIZE['ecdh_pt'] + SIZE['mlkem_ct'], 'C2': SIZE['mlkem_ct']}[cfg] if cred else 0
    pk = ({'C0': SIZE['ecdsa_pk'], 'C1': SIZE['ecdsa_pk'] + SIZE['mldsa_pk'], 'C2': SIZE['mldsa_pk']}[cfg]
          if P.pk_in_msg else 0)
    return P.payload + sig + kex + pk

# legs: (from, to, link, carries credential/KEM element)
LEGS = [('dev', 'psp', 'mup', True), ('psp', 'npci', 'int', True), ('npci', 'rem', 'int', True),
        ('rem', 'npci', 'int', False), ('npci', 'ben', 'int', False), ('ben', 'npci', 'int', False),
        ('npci', 'psp', 'int', False), ('psp', 'dev', 'mdn', False)]

def work_per_txn(P, cfg):
    """expected HSM seconds per transaction in each pool (all legs in mode cfg)"""
    pL, pS = P.share_bankL, 1 - P.share_bankL
    s, d = hsm_s(P, cfg, 'sign'), hsm_s(P, cfg, 'kex_resp')
    return {'psp': 2 * s, 'npci': d + 3 * s, 'bankL': pL * (d + s) + pL * s, 'bankS': pS * (d + s) + pS * s}

def rho0(P): return {'psp': P.rho0_psp, 'npci': P.rho0_npci, 'bankL': P.rho0_bankL, 'bankS': P.rho0_bankS}

def units_full(P, cfg_work, lam_design=None):
    lam = (P.avg_tps * P.mult_peak) if lam_design is None else lam_design
    r = rho0(P)
    return {k: int(math.ceil(cfg_work[k] * lam / r[k])) for k in POOLS}

def baseline_units(P): return units_full(P, work_per_txn(P, 'C0'))

def bytes_npci(P, cfg):
    """bytes crossing NPCI-facing legs per transaction (legs 1..6)"""
    return sum(msg_bytes(P, cfg, LEGS[i][3]) for i in range(1, 7))

# ------------------------------------------------------------------ simulation
def _lognorm_mean(rng, mean, sigma):
    return mean * math.exp(sigma * rng.standard_normal() - 0.5 * sigma * sigma)

def run_sim(P, cfg, load_mult, units, seed, ready_frac=None):
    """
    cfg: 'C0','C1','C2' for all legs, or 'MIX' with ready_frac (legs hybrid iff both ends ready).
    units: dict pool -> FULL-SCALE unit count (scaled down by P.scale_k internally).
    """
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    k = P.scale_k
    pools, svc_scale, cap = {}, {}, {}
    for name in POOLS:
        c_full = units[name]
        c_sim = max(1, int(math.ceil(c_full / k)))
        cap[name] = c_sim
        svc_scale[name] = c_sim * k / c_full          # keeps total capacity = c_full / k
        pools[name] = simpy.PriorityResource(env, capacity=c_sim)
    busy = {n: 0.0 for n in POOLS}
    waits = {n: [] for n in POOLS}
    lat, fails, slow = [], [0], [0]
    n_meas = [0]
    full_hyb = [0]
    t0w, t1w = P.warm, P.warm + P.meas
    lam = P.avg_tps * load_mult / k

    def mode_list():
        if cfg != 'MIX': return [cfg] * 8
        f = ready_frac
        psp, rem, ben = (rng.random() < f), (rng.random() < f), (rng.random() < f)
        m = lambda r: 'C1' if r else 'C0'
        return [m(psp), m(psp), m(rem), m(rem), m(ben), m(ben), m(psp), m(psp)]

    def hsm(pool, work, deadline, measured, stage=0):
        pl = pools[pool]
        t_req = env.now
        req = pl.request(priority=-stage)   # in-flight (later-stage) requests served first
        if len(pl.users) < pl.capacity and not pl.queue:
            yield req
        else:
            res = yield req | env.timeout(max(0.0, deadline - env.now))
            if req not in res:
                req.cancel()
                return False
        w = env.now - t_req
        st = work * svc_scale[pool] * math.exp(P.svc_sigma * rng.standard_normal() - 0.5 * P.svc_sigma ** 2)
        if measured:
            waits[pool].append(w); busy[pool] += st
        yield env.timeout(st)
        pl.release(req)
        return True

    def net(i, mode, deadline):
        link = LEGS[i][2]
        size = msg_bytes(P, mode, LEGS[i][3])
        if link == 'mup': lt, bw = P.mobile_lat_ms, P.mobile_up_bps
        elif link == 'mdn': lt, bw = P.mobile_lat_ms, P.mobile_down_bps
        else: lt, bw = P.inter_lat_ms, P.inter_bps
        d = _lognorm_mean(rng, lt * 1e-3, 0.35 if link != 'int' else 0.2) + size * 8 / bw
        yield env.timeout(d)

    def txn():
        t_arr = env.now
        measured = t0w <= t_arr < t1w
        if measured: n_meas[0] += 1
        deadline = t_arr + P.t_out
        m = mode_list()
        tierL = rng.random() < P.share_bankL
        rem_pool, ben_pool = ('bankL' if tierL else 'bankS'), ('bankL' if rng.random() < P.share_bankL else 'bankS')
        ok = True
        def fail():
            if measured: fails[0] += 1
        # node0 device
        yield env.timeout(dev_s(P, m[0], 'sign') + dev_s(P, m[0], 'kex_init'))
        yield from net(0, m[0], deadline)
        # node1 psp
        yield env.timeout(swsrv_s(P, m[0], 'verify'))
        if not (yield from hsm('psp', hsm_s(P, m[1], 'sign'), deadline, measured, 1)): return fail()
        yield from net(1, m[1], deadline)
        # node2 npci
        yield env.timeout(swsrv_s(P, m[1], 'verify') + swsrv_s(P, m[2], 'kex_init'))
        if not (yield from hsm('npci', hsm_s(P, m[1], 'kex_resp') + hsm_s(P, m[2], 'sign'), deadline, measured, 2)): return fail()
        yield from net(2, m[2], deadline)
        if env.now > deadline: return fail()
        # node3 remitter
        yield env.timeout(swsrv_s(P, m[2], 'verify'))
        if not (yield from hsm(rem_pool, hsm_s(P, m[2], 'kex_resp'), deadline, measured, 3)): return fail()
        yield env.timeout(_lognorm_mean(rng, P.core_rem_ms * 1e-3, 0.4))
        if not (yield from hsm(rem_pool, hsm_s(P, m[3], 'sign'), deadline, measured, 4)): return fail()
        yield from net(3, m[3], deadline)
        # node4 npci
        yield env.timeout(swsrv_s(P, m[3], 'verify'))
        if not (yield from hsm('npci', hsm_s(P, m[4], 'sign'), deadline, measured, 5)): return fail()
        yield from net(4, m[4], deadline)
        if env.now > deadline: return fail()
        # node5 beneficiary
        yield env.timeout(swsrv_s(P, m[4], 'verify'))
        yield env.timeout(_lognorm_mean(rng, P.core_ben_ms * 1e-3, 0.4))
        if not (yield from hsm(ben_pool, hsm_s(P, m[5], 'sign'), deadline, measured, 6)): return fail()
        yield from net(5, m[5], deadline)
        # node6 npci
        yield env.timeout(swsrv_s(P, m[5], 'verify'))
        if not (yield from hsm('npci', hsm_s(P, m[6], 'sign'), deadline, measured, 7)): return fail()
        yield from net(6, m[6], deadline)
        # node7 psp
        yield env.timeout(swsrv_s(P, m[6], 'verify'))
        if not (yield from hsm('psp', hsm_s(P, m[7], 'sign'), deadline, measured, 8)): return fail()
        yield from net(7, m[7], deadline)
        # node8 device
        yield env.timeout(dev_s(P, m[7], 'verify'))
        L = env.now - t_arr
        if L > P.t_out: return fail()
        if measured:
            lat.append(L)
            if L > P.slo: slow[0] += 1
            if all(x == 'C1' for x in m) or cfg in ('C1', 'C2'): full_hyb[0] += 1

    def arrivals():
        while env.now < t1w:
            yield env.timeout(rng.exponential(1.0 / lam))
            env.process(txn())

    env.process(arrivals())
    env.run(until=t1w + P.t_out + 0.5)
    a = np.array(lat) if lat else np.array([np.nan])
    util = {n: busy[n] / (cap[n] * P.meas) for n in POOLS}
    return dict(n=n_meas[0], completed=len(lat), fail_rate=fails[0] / max(1, n_meas[0]),
                p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                mean=float(a.mean()), slow_share=slow[0] / max(1, n_meas[0]),
                full_hybrid_share=full_hyb[0] / max(1, n_meas[0]),
                util=util, wait_mean={n: float(np.mean(waits[n])) if waits[n] else 0.0 for n in POOLS},
                c_sim=cap)

# ------------------------------------------------------------------ validation
def erlang_c_wq(lam, mu, c):
    a = lam / mu
    rho = a / c
    s = sum(a ** n / math.factorial(n) for n in range(c))
    top = a ** c / (math.factorial(c) * (1 - rho))
    pw = top / (s + top)
    return pw / (c * mu - lam), pw

def validate_mmc(c=5, mu=1.0, rhos=(0.5, 0.7, 0.85, 0.92), n=150000, seed=1):
    out = []
    for rho in rhos:
        lam = rho * c * mu
        rng = np.random.default_rng(seed)
        env = simpy.Environment(); res = simpy.Resource(env, c); w = []
        def cust():
            t = env.now
            with res.request() as r:
                yield r
                w.append(env.now - t)
                yield env.timeout(rng.exponential(1 / mu))
        def gen():
            for _ in range(n):
                yield env.timeout(rng.exponential(1 / lam)); env.process(cust())
        env.process(gen()); env.run()
        w = np.array(w[n // 10:])
        wq, pw = erlang_c_wq(lam, mu, c)
        # batch-means 95% CI
        b = np.array_split(w, 20); bm = np.array([x.mean() for x in b])
        out.append(dict(rho=rho, sim_wq=float(w.mean()), ci=float(1.96 * bm.std(ddof=1) / math.sqrt(20)),
                        analytic_wq=wq, rel_err=float((w.mean() - wq) / wq), p_wait_analytic=pw,
                        p_wait_sim=float((w > 1e-12).mean())))
    return out

def pooling_bias(P):
    """Erlang-C waiting probability at full-scale c vs simulated (scaled) c at the same utilisation."""
    out = []
    u0 = baseline_units(P)
    for pool in POOLS:
        c_full = u0[pool]; c_sim = max(1, math.ceil(c_full / P.scale_k))
        for rho in (0.6, 0.8):
            _, pf = erlang_c_wq(rho * c_full, 1.0, c_full) if c_full < 170 else (0, float('nan'))
            _, ps = erlang_c_wq(rho * c_sim, 1.0, c_sim)
            out.append(dict(pool=pool, c_full=c_full, c_sim=c_sim, rho=rho, pwait_full=pf, pwait_sim=ps))
    return out

# ------------------------------------------------------------------ analytic capacity table
def capacity_table(P):
    u0 = baseline_units(P)
    tab = {}
    for cfg in ('C0', 'C1', 'C2'):
        u = units_full(P, work_per_txn(P, cfg))
        tab[cfg] = dict(units=u, extra={k: u[k] - u0[k] for k in POOLS},
                        ratio={k: u[k] / u0[k] for k in POOLS},
                        work_ms={k: work_per_txn(P, cfg)[k] * 1e3 for k in POOLS},
                        bytes_npci=bytes_npci(P, cfg),
                        gbps_peak=bytes_npci(P, cfg) * 8 * P.avg_tps * P.mult_peak / 1e9,
                        msg_bytes_cred=msg_bytes(P, cfg, True), msg_bytes_plain=msg_bytes(P, cfg, False))
    return tab

if __name__ == '__main__':
    P = Params()
    t = time.time()
    r = run_sim(P, 'C0', P.mult_peak, baseline_units(P), seed=1)
    print(json.dumps(r, indent=1)); print('secs', time.time() - t)
