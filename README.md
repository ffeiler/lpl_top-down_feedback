# Extend Latent Predictive Learning with Top-Down Feedback

This repository is a fork of [LPL](https://github.com/fmi-basel/latent-predictive-learning), a framework for biologically plausible self-supervised learning. Please refer there for citations, basic setups, usage instructions, and licenses.

## Usage

To train a deep net with layer-local LPL, simply run

```
python lpl_main.py
```

in the virtual environment you just created. Several useful command-line arguments are provided in `lpl_main.py` and `models\modules.py`. A few are listed below:
- `--train_with_supervision` trains the same network with supervision.
- `--use_negative_samples` trains the network with a cosine-distance-based contrastive loss.
- `--topdown` trains the network with top-down feeback
Multiple implementations exist, yet the error correction based on Rao & Ballard (1999) yields the best results:
```python
python lpl_main.py --topdown --distance_top_down 1 --error_correction --error_nb_updates 1 --alpha_error 2.0
´´´
