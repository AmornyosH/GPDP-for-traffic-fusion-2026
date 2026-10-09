# GPDP for Traffic Fusion

**Gaussian Process Diffusion Policy (GPDP) for Autonomous Vehicles in Mixed Traffic**

A PhD research implementation of the **Stochastic Gaussian Process Diffusion Policy (SGDP)** — an offline reinforcement learning algorithm that combines Gaussian Processes with Diffusion Models to control a single autonomous vehicle (AV) in a mixed-traffic ring road, improving overall traffic flow in congested scenarios.

Simulations are built on [SUMO (Simulation of Urban Mobility)](https://sumo.dlr.de/).

---

## Algorithm Overview

SGDP learns from offline expert demonstrations (no live environment interaction during training). The policy uses:

- **Diffusion model** (VP schedule, 50 forward + 50 denoising steps) to generate actions.
- **Gaussian Process** (Exact GP or Sparse Variational GP) to model uncertainty and shape the policy.
- **MLP value functions** (q₁, q₂) trained with an offline RL objective.

At test time, the AV observes a 10-dimensional state (own speed, lane, plus leader/follower/neighbor vehicle kinematics) and outputs a 2-dimensional action (longitudinal acceleration ∈ [−1.5, 1.0] m/s², lateral lane-change command).

---

## Repository Structure

```
GPDP-for-traffic-fusion-2026/
├── main.py                        # Entry point — dataset / training / testing / plotting
├── agents/
│   ├── AV.py                      # AV agent using SGDP controller
│   ├── DerivedAV.py               # Rule-based expert agent
│   ├── IDM.py                     # Intelligent Driver Model baseline
│   ├── RandomAV.py                # Random-action baseline
│   ├── GPs/
│   │   ├── exactGP.py             # Exact GP with RBF kernel
│   │   └── SVGP.py                # Sparse Variational GP
│   └── RLs/
│       ├── SGDP.py                # Main algorithm: SGDP
│       └── SGDQ.py                # Q-learning variant
├── environments/
│   └── double_ring/
│       ├── double_ring.py         # SUMO environment wrapper
│       ├── double_ring.sumocfg.xml
│       ├── double_ring.net.xml
│       ├── double_ring.rou.xml
│       └── double_ring.add.xml
└── utilities/
    ├── dataset_creation.py        # Generate offline RL datasets
    ├── training.py                # Offline and online training loops
    ├── evaluation.py              # Policy evaluation in SUMO
    ├── MLP.py                     # 3-layer MLP (policy & value networks)
    ├── miscellaneous.py           # Path helpers, scaling, encoding
    ├── plot_results.py            # Speed vs. density result plots
    └── plot_actions.py            # Action distribution visualisations
```

---

## Requirements

### Python packages

| Package | Notes |
|---|---|
| `torch` | PyTorch ≥ 1.12 (GPU recommended for GP training) |
| `numpy` | Array operations and dataset storage |
| `matplotlib` | Result and action-distribution plots |
| `traci` | SUMO Python API (bundled with SUMO install) |
| `sumolib` | SUMO utility library (bundled with SUMO install) |

Install Python dependencies:

```bash
pip install torch numpy matplotlib
```

### SUMO

SUMO ≥ 1.19.0 is required. Install from [https://sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php) and ensure the `SUMO_HOME` environment variable is set:

```bash
# Linux / macOS
export SUMO_HOME=/usr/share/sumo      # adjust to your install path

# Windows (PowerShell)
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

---

## Data & Output Paths

The project stores datasets, training records, and evaluation results outside the repository. Default paths (configured in `utilities/miscellaneous.py`) are:

| Resource | Default path |
|---|---|
| Datasets | `C:/Amornyos/PhD/resources/T-ITS/datasets/` |
| Training records | `C:/Amornyos/PhD/resources/T-ITS/training_records/` |
| Evaluation results | `C:/Amornyos/PhD/resources/T-ITS/training_records/` |

Adjust `getPaths()` in [utilities/miscellaneous.py](utilities/miscellaneous.py) if your directory layout differs.

---

## Usage

All tasks are run through `main.py`. Global seed is fixed at `2203`.

### 1. Create an offline dataset

```bash
python main.py --task dataset --env double_ring --dataset_type expert
```

- `--dataset_type`: `expert` (DerivedAV), `human` (IDM), or `human-expert` (mixed).
- You will be prompted to enter the number of data points (e.g. `50000`).
- Saves `<dataset_path>_<type>.npz` containing `observations`, `actions`, `rewards`, `next_observations`, `truncated`, `terminations`.

### 2. Train SGDP

```bash
python main.py --task training --alg SGDP --dataset_type expert --gradient_step 200000
```

When prompted, enter `offline` (or `online` for continued environment interaction).

| Argument | Options | Default |
|---|---|---|
| `--alg` | `SGDP`, `SGDQ` | `SGDP` |
| `--gradient_step` | integer | `200000` |
| `--load` | flag — resume from checkpoint | off |

Training saves a PyTorch checkpoint (`.pth`) to `<training_record_path>`.

### 3. Evaluate a policy

```bash
python main.py --task testing --alg SGDP --dataset_type expert --eval_mode standard --load
```

Evaluates across vehicle densities `[40, 42, 44, 46, 48]` with 10 runs each (1 000 timesteps per run). Use `--render` to open the SUMO GUI.

| `--eval_mode` | Description |
|---|---|
| `standard` | Normal traffic scenario |
| `shifted` | Robustness test — includes "imposter" vehicles with shifted dynamics |

Results are saved to `<evaluation_path>_<dataset_type>_<eval_mode>.npz`.

### 4. Plot speed results

```bash
python main.py --task plot --dataset_type expert --eval_mode standard
```

Produces a **mean vehicle speed vs. traffic density** plot comparing IDM, DerivedAV (Expert), TRPO, and SGDP (Ours).

### 5. Plot action distributions

```bash
python main.py --task plot_action --dataset_type expert --eval_mode standard
```

Produces scatter plots of longitudinal vs. lateral actions, comparing dataset actions, TRPO, and SGDP.

---

## Environment: Double-Ring Road

| Property | Value |
|---|---|
| Topology | 2-lane circular road, 250 m circumference |
| Vehicle count | 40 – 48 (configurable density) |
| Simulation step | 0.1 s |
| AV observation dim | 10 (own speed, lane; leader, follower, lane-neighbour kinematics) |
| Action dim | 2 (acceleration m/s², lane-change) |
| Reward | Speed-based (global average of 5 observed vehicles) |

---

## Agents & Baselines

| Agent | Description |
|---|---|
| **SGDP** (Ours) | Stochastic GP Diffusion Policy — proposed algorithm |
| **DerivedAV** (Expert) | Rule-based expert: accelerate/decelerate towards target speed |
| **IDM** | Intelligent Driver Model — SUMO default car-following model |
| **RandomAV** | Uniformly random actions — weakest baseline |
| **TRPO** | Trust Region Policy Optimisation — online RL baseline (pre-computed results loaded for plotting) |

---

## Results

Evaluation metric: **mean vehicle speed (km/h)** across traffic densities 80 – 96 veh/km (corresponding to 40 – 48 vehicles on a 500 m double-ring road).

Key findings:
- SGDP trained on expert offline data achieves speeds comparable to or exceeding the rule-based expert (DerivedAV) across all tested densities.
- SGDP outperforms online RL (TRPO) in the shifted (distribution-shifted) evaluation mode, demonstrating improved robustness via GP-based uncertainty modelling.
- The IDM baseline degrades significantly at high density, while SGDP maintains smoother traffic flow.

Output files after evaluation:

```
<evaluation_path>_expert_standard.npz   # standard scenario results
<evaluation_path>_expert_shifted.npz    # robustness (shifted) results
```

Each `.npz` contains: `num_veh`, `accum_reward`, `avg_av_speed`, `avg_all_speed`, `std_av_speed`, `std_all_speed`.

---

## Reproducibility

- Global seed: `2203`
- Evaluation seeds: `2203 + i × 100` for the *i*-th evaluation run
- Dataset creation and training both respect the global seed set in `main.py`
