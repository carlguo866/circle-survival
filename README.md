# Survival of the Fittest Representation
This repository contains the code for the paper "Survival of the Fittest Representation: A Case Study with Modular Addition."

## Installation
All required packages are listed in `requirements.txt`. To install, run:
```
conda create -n circle-survival --file requirements.txt
```

## Reproducing Our Results

### Train models

Collect data first using `training/train_model.py`, such as the following to train a model with fixed embedding and randomized MLPs and dataset orderings:
```
for init_seed in $(seq 0 1 9)
do  
    python training/train_model.py \
        --init_seed $init_seed \
        --start_seed 0 \
        --end_seed 50 \
        --fix_embedding True \
        --data_path data \
        --steps 30000
done
```
Similarly, you can train the model with specific interventions with `perturbation.py`, `ablation.py`, and `manual_construction.py`. 

### Analyze and plot results
Then, run various analyses on the collected data with scripts in `analysis`. Run all analyses in the root directory of the repository. 

Reproduce figures in Section 2 (setup) with `plot_signal_spectrum.py`, Section 3.1 (varying dimensionality) `plot_loss_embed_freeze.py` and `plot_num_circles.py`, Section 3.2.1 (initial signal) with `plot_init_signal_*.py`, Section 3.2.2 (initial gradient) with `plot_init_gradient.py`, Section 4.1 (circle collaborations) with `plot_loss_collaboration.py` and Section 4.3 (ODE analysis) with scripts in `plot_scripts/ode`.

## Contact
If you have any questions about the paper or reproducing results, feel free to email [carlguo@mit.edu](mailto:carlguo@mit.edu).

## Citation

