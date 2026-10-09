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

Datasets, training records, and evaluation results are stored **outside** the repository.

### Recommended: edit `config.json` once

Fill in `config.json` at the project root **once** — every subsequent command picks it up automatically:

```json
{
    "dataset_path": "/your/data/datasets",
    "train_path":   "/your/data/training_records",
    "eval_path":    "/your/data/evaluation_results"
}
```

After that, all commands work with no extra flags:

```bash
python main.py --task dataset  --dataset_type expert
python main.py --task training --alg SGDP
python main.py --task testing  --alg SGDP --load
```

### Alternative: CLI flags (one-off override)

Pass the flags directly to override `config.json` for a single run:

```bash
python main.py --task dataset \
  --dataset_path /your/data/datasets \
  --train_path   /your/data/training_records \
  --eval_path    /your/data/evaluation_results
```

### Priority order

```
CLI flags  >  config.json  >  hardcoded defaults (miscellaneous.py)
```

The algorithm and environment names are appended automatically, e.g. `train_path: /data` → `/data/SGDP_double_ring_training_records.pth`.

---

## Dataset Collection

The offline dataset **D = {o_n, a_n}^{N−1}_{n=0}** consists of MDP transitions collected by running the **DerivedAV (expert)** behaviour policy in SUMO, following the procedure in Section V of the paper.

### Collection procedure

1. **Warmup** — after all vehicles are deployed, the simulation runs for **500 / δ = 5 000 steps** (δ = 0.1 s) under IDM control so that stop-and-go congestion forms naturally before any data is recorded.

2. **Trajectory rollout** — the DerivedAV then runs for **H = 1 000 / δ = 10 000 steps**, and every transition (o_t, a_t, r_{t+1}, o_{t+1}) is stored.

3. **Optimal + sub-optimal mixing** — to expose the policy to the full action distribution, the dataset is split equally:
   - **First half** — DerivedAV departs and stays in the **outer lane (lane 0)**, which yields higher average speed → *optimal* expert actions.
   - **Second half** — DerivedAV departs and stays in the **inner lane (lane 1)**, which yields lower average speed → *sub-optimal* actions.

   This mirrors the paper statement: *"We also mix the data in D with sub-optimal actions by running the derived policy in the inner lane of the system, which provides a lower speed than running the AV in the outer lane."*

4. **Density coverage** — vehicle count cycles through **[40, 42, 44, 46, 48]** across trajectories so that the dataset covers the full evaluation density range.

5. **Target size** — the paper uses **N = 10⁶** transitions. You can specify any size when prompted.

### MDP observation vector (dim = 10)

```
o = [v_AV, l_AV, v_L, d_L, v_F, d_F, v_nL, d_nL, v_nF, d_nF]
```

| Index | Symbol | Description |
|---|---|---|
| 0 | v_AV | AV speed (m/s) |
| 1 | l_AV | AV lane index (0 = outer, 1 = inner) |
| 2 | v_L | Leader speed (m/s) |
| 3 | d_L | Distance to leader (m) |
| 4 | v_F | Follower speed (m/s) |
| 5 | d_F | Distance to follower (m) |
| 6 | v_nL | Neighbour-lane leader speed (m/s) |
| 7 | d_nL | Distance to neighbour-lane leader (m) |
| 8 | v_nF | Neighbour-lane follower speed (m/s) |
| 9 | d_nF | Distance to neighbour-lane follower (m) |

### Action encoding (dim = 2, diffusion space [−1, 1])

The raw SUMO actions are encoded into the diffusion output space before storage:

| Index | Raw space | Encoded space | Conversion |
|---|---|---|---|
| 0 | a_accel ∈ [−1.5, 1.0] m/s² | [−1, 1] | linear mapping |
| 1 | a_lane ∈ {0, 1} | outer → +0.5, inner → −0.5 | Gaussian centred at ±0.5, σ = 0.005 |

### Reward

r_{t+1} = (v_AV + v_L + v_F + v_nL + v_nF) / (5 × v_max)

Global average speed of the 5 observed vehicles, normalised by maximum speed (10 m/s).

### Dataset types

| `--dataset_type` | Behaviour policy | Notes |
|---|---|---|
| `expert` | DerivedAV (outer lane → inner lane) | Used in paper |
| `human` | IDM only | Human-driving baseline |
| `human-expert` | DerivedAV (first half) + IDM (second half) | Mixed |
| `human-random` | IDM (first half) + RandomAV (second half) | Ablation |

### Saved format

`.npz` file with arrays:

| Key | Shape | Description |
|---|---|---|
| `observations` | (N, 10) | o_t |
| `actions` | (N, 2) | a_t (encoded) |
| `rewards` | (N, 1) | r_{t+1} |
| `next_observations` | (N, 10) | o_{t+1} |
| `truncrated` | (N,) | trajectory boundary flag |
| `terminations` | (N,) | episode termination flag |

---

## Usage

All tasks are run through `main.py`. Global seed is fixed at `2203`.

### 1. Create an offline dataset

```bash
python main.py --task dataset --env double_ring --dataset_type expert
```

- `--dataset_type`: `expert` (paper default), `human`, `human-expert`, or `human-random`.
- You will be prompted to enter the number of data points (paper uses `1000000`).
- Saves `<dataset_path>_<type>.npz`.

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
