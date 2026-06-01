# Battery-Degradation-Aware SAC-DRL for Optimal V2G Scheduling

**Official code repository for the IEEE Transactions on Transportation Electrification submission:**

> *"Battery-Degradation-Aware Soft Actor-Critic Deep Reinforcement Learning for Optimal Vehicle-to-Grid Scheduling in Hybrid Renewable Microgrids"*
> — Subrahmanyam Tanala, Member IEEE, ANITS Visakhapatnam

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org/)
[![IEEE TTE](https://img.shields.io/badge/Target-IEEE%20TTE-green.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6687316)

---

## Overview

BDA-SAC-DRL is a hybrid deep reinforcement learning + model predictive control framework that simultaneously optimises:

- 🔋 **Battery longevity** — via an online semi-empirical degradation cost (rainflow cycle counting + Arrhenius calendar aging) embedded in the SAC reward
- 💰 **Energy economics** — minimises grid import cost, maximises V2G revenue
- ♻️ **Renewable utilisation** — reduces curtailment through degradation-aware dispatch
- 🔒 **Constraint satisfaction** — MPC projection guarantees SoC bounds and power limits

Validated against NASA PCoE 18650 battery aging data (RMSE = 0.40 %, R² = 0.996) and confirmed on a **Typhoon HIL 604+ Controller Hardware-in-the-Loop** testbench.

---

## Key Results

| Method | Fade (%/yr) | Net Cost ($/day) | RE Util. (%) | CS (%) |
|---|---|---|---|---|
| Rule-Based | 2.85 ± 0.18 | 16.32 ± 0.75 | 78.5 | 94.1 |
| MPC only | 2.10 ± 0.13 | 7.85 ± 0.42 | 88.0 | 98.5 |
| Standard SAC | 1.96 ± 0.14 | 7.20 ± 0.38 | 87.2 | 96.8 |
| **BDA-SAC-DRL** | **1.48 ± 0.09** | **4.91 ± 0.19** | **93.4** | **99.2** |

*n = 10 seeds; Wilcoxon p < 0.001; Cliff's δ = 0.91 vs standard SAC*

---

## System Architecture

```
PV Array (50 kWp) ─┐
Wind Turbine (30 kW)─┤──► DC/AC Bus ──► Utility Grid
EV Fleet (5×60 kWh) ─┤         │
Stationary BESS ─────┘         ▼
                        BDA-SAC-DRL Agent
                        ┌──────────────────┐
                        │  Actor-Critic     │
                        │  + Degradation    │
                        │    Reward         │
                        │  + MPC Layer      │
                        │  + Optuna HPO     │
                        └──────────────────┘
```

---

## Repository Structure

```
bda-sac-drl-v2g/
├── paper/
│   ├── paper_v19.tex          # LaTeX source (IEEEtran)
│   └── paper_v19.pdf          # Final manuscript (9 pages)
├── figures/
│   ├── fig1_architecture.pdf  # System architecture
│   ├── fig2_convergence.pdf   # Training convergence (mean ± SD, n=10)
│   ├── fig3_dispatch.pdf      # 24-hour dispatch profile
│   ├── fig4_ablation.pdf      # Ablation study
│   ├── fig5_stats.pdf         # Statistical comparison + effect sizes
│   ├── fig6_seasonal.pdf      # Seasonal robustness
│   ├── fig7_robust.pdf        # Scalability & forecast sensitivity
│   ├── fig9_hil_arch.pdf      # C-HIL test bench architecture
│   ├── fig10_hil_tracking.pdf # Power tracking (5 test cases)
│   ├── fig11_hil_latency.pdf  # Dispatch latency + SoC violations
│   ├── fig12_hil_comparison.pdf # Sim-to-HIL parity
│   └── fig16_deg_validation.pdf # Degradation model validation
├── scripts/
│   ├── gen_figures.py         # Main simulation figures
│   ├── gen_hil_figs.py        # HIL validation figures
│   └── gen_deg_v2.py          # Degradation model validation figure
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/subrahmanyamtanala-rgb/bda-sac-drl-v2g.git
cd bda-sac-drl-v2g
pip install -r requirements.txt
```

---

## Reproducing Figures

```bash
# Generate all simulation figures
python scripts/gen_figures.py

# Generate HIL validation figures
python scripts/gen_hil_figs.py

# Generate degradation model validation (NASA PCoE data)
python scripts/gen_deg_v2.py
```

All figures are saved to `figures/` as both `.pdf` (vector) and `.png` (300 dpi).

---

## Compile the Paper

Requires a LaTeX distribution with IEEEtran class:

```bash
cd paper
pdflatex paper_v19.tex
pdflatex paper_v19.tex   # twice for cross-references
pdflatex paper_v19.tex
```

---

## Dependencies

```
numpy>=1.24
matplotlib>=3.7
scipy>=1.11
torch>=2.0
gymnasium>=0.29      # OpenAI Gym successor
osqp>=0.6
optuna>=3.4
```

See `requirements.txt` for exact pinned versions.

---

## Datasets

| Dataset | Source | Usage |
|---|---|---|
| Solar irradiance | [NREL NSRDB](https://nsrdb.nrel.gov) | PV generation model |
| Battery aging | [NASA PCoE](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) | Degradation model validation |
| Load profile | IEEE RTS-96 | Demand simulation |
| EV mobility | NextGen NTHS | Stochastic arrival/departure |
| Temperature | ERA5 reanalysis | Calendar aging model |

---

## Degradation Model Validation

The Schmalstieg et al. (2014) semi-empirical model is validated against NASA PCoE 18650 cells (No. 5, 6, 7, 18) under 1C cycling at 24 °C:

- **RMSE = 0.40 %** remaining capacity
- **R² = 0.996**
- Residuals within ±2 % band
- No re-fitting to V2G application (held-out validation)

---

## C-HIL Validation Summary

| Test | Scenario | RMSE (kW) | SoC Viols. | Result |
|---|---|---|---|---|
| TC-1 | PV cloud transient | 0.22 | 0 | ✅ Pass |
| TC-2 | EV surge arrival | 0.31 | 1 | ✅ Pass |
| TC-3 | RTP price spike | 0.17 | 0 | ✅ Pass |
| TC-4 | SoC boundary test | 0.19 | 0 | ✅ Pass |
| TC-5 | 200 ms delay | 0.28 | 1 | ✅ Pass |

Mean dispatch latency: **4.2 ± 0.8 ms** (deadline: 15 ms, ISO 15118-20)

---

## Citation

If you use this code or manuscript, please cite:

```bibtex
@article{tanala2026bdasac,
  author  = {Tanala, Subrahmanyam},
  title   = {Battery-Degradation-Aware Soft Actor-Critic Deep Reinforcement
             Learning for Optimal Vehicle-to-Grid Scheduling in Hybrid
             Renewable Microgrids},
  journal = {IEEE Transactions on Transportation Electrification},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contact

**Subrahmanyam Tanala**, Member IEEE
Department of Electrical and Electronics Engineering
Anil Neerukonda Institute of Technology and Sciences (ANITS), Visakhapatnam 530003, India
✉️ subrahmanyam.eee@anits.edu.in

---

## License

MIT License — see [LICENSE](LICENSE) for details.
