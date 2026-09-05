# Code for DQN-VSSI-EWMA

This repository contains the Python/Jupyter Notebook implementation accompanying the paper:

**_A Deep Reinforcement Learning Framework for Adaptive Control Charts_**

The code implements the simulation-based framework proposed in the paper for learning an adaptive variable sample size and variable sampling interval (VSSI) policy for an EWMA control chart using a Deep Q-Network (DQN).

## Contents

### 1. `General_Case.ipynb`

This notebook contains the main implementation for the general DQN-VSSI-EWMA framework. The general action space contains 70 possible sampling pairs formed by 10 candidate sample sizes and 7 candidate sampling intervals.

The notebook is organized as follows:

#### Cell 1 — General 70-action DQN-VSSI-EWMA implementation

This is the main computational cell for the general DQN model. It includes:

- specification of the EWMA monitoring model and design parameters;
- construction of the 70-action sampling space;
- Monte Carlo calibration of the EWMA control-limit coefficient;
- estimation of the in-control mean risk score used in the structural reward;
- construction of the DQN training environment and state representation;
- implementation of the reward function;
- definition of the DQN architecture and replay buffer;
- two-stage DQN training;
- checkpoint-based in-control validation;
- out-of-control validation of feasible checkpoints;
- selection of the final feasible checkpoint;
- final in-control and out-of-control Monte Carlo performance evaluation; and
- analysis of the learned adaptive sampling behavior.

The main outputs from this cell are stored in the `outputs` object and are used by the subsequent cells.

#### Cell 2 — Checkpoint-validation figure

This cell uses the checkpoint history produced in Cell 1 to create **Figure 3** in the paper.

It summarizes the checkpoint-based policy-selection process, including:

- checkpoint episode;
- in-control feasibility;
- mean validation out-of-control ATS;
- average in-control sampling interval;
- average in-control sample size; and
- the checkpoint selected as the final policy.

Cell 1 must be run before this cell.

#### Cell 3 — Illustrative online monitoring examples

This cell uses the trained general DQN policy from Cell 1 to generate illustrative monitoring examples.

It applies the greedy trained policy sequentially and records the evolution of the EWMA statistic together with the sample size and sampling interval selected by the DQN. The out-of-control shift size can be changed through the `delta_oc` setting to examine illustrative monitoring paths under different process shifts.

Cell 1 must be run before this cell.

---

### 2. `4_Level_cases.ipynb`

This notebook contains the restricted four-action DQN implementation and the corresponding traditional four-level VSSI-EWMA benchmark.

The restricted DQN and traditional chart use the same four sampling pairs:

- `(n = 2, h = 1.50)`
- `(n = 3, h = 1.25)`
- `(n = 4, h = 0.75)`
- `(n = 5, h = 0.50)`

This allows a controlled comparison between a learned state-dependent sampling rule and a fixed region-to-action rule.

The notebook is organized as follows:

#### Cell 1.1 — Restricted four-action DQN-VSSI-EWMA implementation

This cell contains the complete restricted DQN computational pipeline. It includes:

- specification of the restricted four-action sampling space;
- EWMA model and design settings;
- Monte Carlo calibration of the control-limit coefficient;
- construction of the restricted DQN environment;
- restricted-action state normalization;
- reward-function specification for the restricted action space;
- two-stage DQN training;
- checkpoint-based in-control validation;
- out-of-control validation of feasible checkpoints;
- final checkpoint selection;
- final in-control and out-of-control Monte Carlo evaluation; and
- analysis of the learned restricted sampling policy.

The main results are stored in the `outputs` object for use in later cells.

#### Cell 1.2 — Checkpoint-validation figure

This cell uses the checkpoint history from Cell 1.1 to create **Figure 4** in the paper.

It summarizes the checkpoint-validation trajectory of the restricted four-action DQN and identifies the checkpoint selected as the final policy.

Cell 1.1 must be run before this cell.

#### Cell 2 — Traditional four-level VSSI-EWMA benchmark

This cell implements the fixed four-level traditional VSSI-EWMA chart used as the benchmark in the paper.

It includes:

- the same four admissible sampling pairs used by the restricted DQN;
- Monte Carlo calibration of the EWMA control-limit coefficient under stationary initialization;
- construction of the four fixed decision regions;
- assignment of the four sampling pairs to those regions;
- in-control Monte Carlo evaluation;
- out-of-control Monte Carlo evaluation over the reported shift sizes; and
- calculation of the traditional chart's performance and resource-use measures.

The results are stored in `outputs_traditional` for use in the comparison cell.

#### Cell 3 — Restricted DQN versus traditional-chart comparison

This cell compares the results produced by Cell 1.1 and Cell 2.

It compares the:

- restricted four-action DQN-VSSI-EWMA chart; and
- fixed four-level traditional VSSI-EWMA chart.

The comparison includes their in-control performance, out-of-control ARL and ATS results, sampling-resource behavior, and associated numerical summaries.

Cells 1.1 and 2 must be run before this cell.

## Requirements

The notebooks require a Python environment with the packages used in the notebooks, including:

- Python 3.x
- NumPy
- pandas
- PyTorch
- SciPy
- matplotlib
- joblib

A minimal installation can be created with:

```bash
pip install numpy pandas torch scipy matplotlib joblib jupyter
```

The notebooks use only these third-party packages together with Python standard-library modules.

## How to run

1. Install Python 3.x and the required packages.
2. Open the desired notebook in Jupyter Notebook, JupyterLab, or another compatible environment.
3. Run the cells in order from the beginning of the notebook.
4. Allow the Monte Carlo calibration, DQN training, checkpoint validation, and final evaluation stages to complete.

The notebooks are simulation based; no external empirical dataset is required. The process observations used for training and evaluation are generated within the simulation environment defined in the notebooks. Thus, the simulation generator itself provides the example data needed to execute and verify the implementation.
The default parameter and hyperparameter values used in the paper are retained in the notebooks. Users may modify these values to investigate alternative design settings and rerun the corresponding calibration, training, and evaluation procedures.

### Parallel computation and exact reproducibility

The notebooks set `N_JOBS = min(8, max(1, os.cpu_count() - 1))`. Monte Carlo replications are divided among these workers, with deterministic worker-specific random seeds. Therefore, machines with different CPU counts may use different random-number streams and produce slightly different Monte Carlo estimates. For exact replication of the reported computational streams, set `N_JOBS = 8`, which is the setting used for the reported runs.


## Offline design and online monitoring (Phase I / Phase II interpretation)

The numerical study assumes that the in-control process parameters are known. Therefore, it does not include a separate empirical Phase I parameter-estimation dataset. Instead, the code contains the complete **offline design stage** required before deployment:

1. specify the in-control model, EWMA parameters, admissible sampling actions, and resource targets;
2. calibrate the EWMA control-limit coefficient to the target in-control ARL;
3. construct the DQN environment and state representation;
4. train the DQN in two stages;
5. evaluate candidate checkpoints under the in-control resource constraints; and
6. select the feasible checkpoint with the best validation ATS performance.

After this offline stage, the trained network is fixed. During **Phase II-style online monitoring**, the DQN does not retrain. At each sampling epoch, the current monitoring state is formed, the greedy trained policy selects the next sample size and sampling interval, the EWMA statistic is updated from the new sample, and the calibrated EWMA control limit determines whether a signal occurs.

`General_Case.ipynb`, Cell 3, provides illustrative sequential monitoring examples of this deployment stage.

## Adapting the implementation

The notebooks can be modified to examine alternative monitoring designs by changing the relevant configuration values before rerunning calibration and training. Examples include:

- the EWMA smoothing parameter;
- the target in-control ARL;
- the target average sample size or sampling interval;
- the admissible `(n, h)` action space;
- the shift sizes used for DQN training or checkpoint validation; and
- Monte Carlo replication sizes.

For an empirical application, the present notebooks should be viewed as a reference implementation rather than a plug-and-play data-import interface. The current simulation generates the standardized sample statistic internally. To use observed process data, the data-generation portion would need to be replaced by the corresponding standardized statistic computed from the user's samples under an appropriate in-control reference model. The trained DQN can then use the resulting EWMA state to select the next sampling action.

Because the learned policy is configuration-specific, changes to the process model, EWMA smoothing parameter, control-chart target, admissible action space, or resource targets generally require offline recalibration and retraining before deployment. If the in-control mean and standard deviation are unknown, they must first be estimated from suitable Phase I reference data before applying the present framework.

## Main design settings

The principal implementation uses:

- EWMA smoothing parameter: `lambda = 0.2`
- target in-control ARL: `ARL0 = 370.4`
- target average sample size: `3.5`
- target average sampling interval: `1.0`
- 100,000 Monte Carlo replications for final performance evaluation
- checkpoint validation every 1,000 training episodes
- two DQN training stages: 20,000 initial-training episodes followed by 10,000 fine-tuning episodes

The detailed settings and implementation are contained directly in the notebooks.

## Interpretation of the code

The DQN does not replace the EWMA statistic or the control-chart signaling rule. It learns the sampling policy: at each sampling epoch it selects the next sample size and sampling interval. The EWMA statistic and calibrated control limit continue to determine whether a signal occurs.

The final policy is selected using checkpoint-based statistical validation rather than simply taking the policy from the last training episode. In-control resource constraints are checked during checkpoint selection, and out-of-control performance is evaluated for feasible checkpoints.

## Scope

The numerical study is a proof-of-concept under the simulation assumptions described in the paper, including independent normal observations and known in-control process parameters. The code is intended to provide a transparent implementation of the proposed methodology, reproduce the principal computational pipeline and numerical evaluation reported in the paper, and provide a basis for further experimentation and extension. The repository is not intended to reproduce every sensitivity or ablation experiment in the manuscript as a separate notebook.

## Citation

When using or extending this code, please cite the associated paper.
