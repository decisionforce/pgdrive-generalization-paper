
**Status:** Archive - code is provided as-is, no updates expected.

# Improving the Generalization of End-to-End Driving through Procedural Generation

This is the official material of the paper: "Improving the Generalization of End-to-End Driving through Procedural Generation".

Please visit the following links to learn more on our PGDrive simulator:

**[ [Website](https://decisionforce.github.io/pgdrive/) | [Github Repo](https://github.com/decisionforce/pgdrive) | [Documentation](https://pgdrive.readthedocs.io/) | [Paper](#) ]**

## Setup the environment

```bash
# Clone this repo to local
git clone https://github.com/decisionforce/pgdrive-generalization-paper.git
cd pgdrive-generalization-paper

# Install dependencies
pip install -r requirements.txt
```


## Draw the main experiments' results

```bash
# Generate result to results/ppo-main-result-up.pdf and results/ppo-main-result-down.pdf
python draw_ppo_results.py

# Generate result to results/sac-main-result-up.pdf and results/sac-main-result-down.pdf
python draw_sac_results.py
```


## Draw the other experiments' results

```bash
# Draw change density experiment result to results/change-friction-result.pdf
python eval_density.py

# Draw change friction experiment result to results/change-friction-result.pdf
python eval_friction.py
```


## Reproduce the experiments

```bash
# PPO main experiments
python train_ppo.py --exp-name main_ppo --num-gpus 0

# SAC Main experiments
python train_ppo.py --exp-name main_ppo --num-gpus 0

# Change friction experiments
python train_ppo_change_friction.py --exp-name change_friction --num-gpus 0

# Change density experiments
python train_ppo_change_density.py --exp-name change_density --num-gpus 0

# For the change density / friction experiments, you need to uncomment the script to call
# get_result function, then call
python eval_density.py
python eval_friction.py
```


## Cite this work

Coming Soon!
