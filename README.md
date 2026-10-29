# UPI quantum-safe migration: simulation code

Python 3.12; `pip install simpy numpy scipy matplotlib`.

| File | Purpose |
| --- | --- |
| `layer1_sim.py` | Layer 1: SimPy discrete-event model of the UPI-inspired transaction path (C0 classical, C1 hybrid, C2 PQC-only); capacity table; M/M/c validation |
| `run_layer1.py` | Layer 1 experiments E1-E3 -> `results/layer1.json` (about 14 min on one core) |
| `layer2_abm.py` | Layer 2: agent-based migration-timing model (NumPy), policy regimes P0-P2 |
| `run_layer2.py` | Layer 2 experiments: policy comparison, sweeps, one-at-a-time sensitivity, limiting cases -> `results/layer2.json` (`python run_layer2.py 1000 300`, about 6.5 min) |
| `make_figures.py` | Figures 1-7 |

Run order: `python run_layer1.py && python run_layer2.py 1000 300 && python make_figures.py`.
All random seeds are fixed in the scripts. Timings, costs and losses are representative / illustrative
assumptions (see the manuscript, Sections 4, 5 and 9); replace them with cited values before relying on any number.
