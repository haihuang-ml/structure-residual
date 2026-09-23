# Structure-Residual Inductive Biases For Language Model Pretraining

This is the official implementation of the paper *"Structure-Residual Inductive Biases For Language Model Pretraining"*.

# Setup

```bash
pip install -r requirements.txt
```

# Usage

### Dataset

First, you need to download the dataset. To recreate the 1B dataset used throughout the paper,
run the following command:

```bash
python preprocess.py \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --split train \
  --tokenizer_name gpt2 \
  --output_dir ./data/fineweb_edu_gpt2 \
  --max_tokens 1200000000 \
  --val_tokens 10000000 \
  --add_eos true \
  --dtype uint16 \
  --streaming true
```

### Training

To reproduce the 1B training results for `gpt2-medium` run the following commands:

For `vanilla` training run:

```bash
python train.py \
  --data_dir ./data/fineweb_edu_gpt2 \
  --model_size gpt2-medium \
  --output_dir ./runs/gpt2-medium-1B \
  --seq_len 1024 \
  --batch_size 8 \
  --grad_accum_steps 8 \
  --target_tokens 1000000000 \
  --lr_decay_tokens 1000000000 \
  --learning_rate 6e-4 \
  --lr_scheduler cosine \
  --warmup_tokens 5000000 \
  --weight_decay 0.1 \
  --torch_compile true \
  --seed 82 \
  --wandb true \
  --bf16 true \
  --eval_tokens_final 10000000 \
```

For `bbc` training add the following arguments:

```commandline
  --bottleneck_mode boundary \
  --bottleneck_concept_k 16 \
  --bottleneck_layers 2
```

For `bbc+cc` training add the following arguments:

```commandline
  --bottleneck_mode boundary \
  --bottleneck_concept_k 16 \
  --bottleneck_layers 2 \
  --continuous_concept_mode mse \
  --continuous_concept_lambda 0.00125 \
  --continuous_concept_layer 22
```

To run in multi-GPU mode, prefix the command with `torchrun --nproc_per_node=<num_gpus>` and adjust the batch size accordingly.

For the full list of arguments run `python train.py --help`.

### Boundary Bottleneck Concept

There are three arguments that control **BBC** behavior:

- `--bottleneck_concept_k <int>`: size of the bottleneck segment.
- `--bottleneck_layers <int|str>`: layer in which to apply BBC. Can be either index of layer or one of ("all", "mid", "last").
- `--bottleneck_mode <mode>`: mode of BBC.

Where `<mode>` can be one of:
- `none`: ordinary causal GPT-2 attention. This is the default.
- `boundary`: each token attends to same-segment causal history plus previous segment boundary tokens.
- `sliding_window`: each token gets the same attention budget as boundary mode, but assigned to the most recent causal tokens.
- `random_boundary`: same as boundary mode, but each segment's summary token is a deterministic random position.
- `random_sparse`: each token gets the same attention budget as boundary mode, but assigned to deterministic random causal positions.

### Continuous Concept

There are six arguments that control **CC** behavior:

- `--continuous_concept_lambda <float>`: CC loss coefficient (named $\beta$ in the paper).
- `--continuous_concept_layer <int|str>`: layer in between which to apply CC. Can be either index of layer or one of ("all", "mid", "last").
- `--continuous_concept_dropout <float>`: probability of dropping CC loss component at each training step.
- `--continuous_concept_adaptive_dropout <int>`: $p$-th percentile of the preceding step history – CC component is dropped if it falls below it.
- `--continuous_concept_adaptive_dropout_history <int>`: the number of steps in the history used for adaptive dropout.
- `--continuous_concept_mode <mode>`: mode of CC.

Where `<mode>` can be one of:
- `none`: no CC loss applied
- `mse`: MSE between adjecent hidden states (named *CC* in the paper)
- `cosine`: 1 - *cosine distance* between adjecent hidden states (named *Cosine* in the paper)
- `l2`: L2 regularization on each hidden state (named *Scale* in the paper)
- `norm`: MSE normalized by the L2 norms sum of adjecent hidden states (named *Normalized* in the paper)
- `mse2`: a two-predecessor Laplacian-residual penalty (named *CC-2* in the paper)

### Representation Diagnostics

These evaluation-only diagnostics compare matched checkpoints on deterministic validation batches; they do not train probes or modify checkpoint weights. Run their synthetic correctness checks first:

```bash
python representation_diagnostics.py --run_tests
```

For the BBC boundary-state diagnostic, omit `--layer` only when the BBC checkpoint used exactly one bottleneck layer. Otherwise, pass the BBC block index (for example `--layer 6`):

```bash
python representation_diagnostics.py bbc \
  --vanilla_checkpoint ./runs/vanilla_100m/final/checkpoint.pt \
  --bbc_checkpoint ./runs/bbc_100m/final/checkpoint.pt \
  --data_dir ./data/fineweb_edu_gpt2 \
  --output_dir ./runs/diagnostics/bbc_100m \
  --layer 6
```

The BBC output compares full evaluation, replacing one boundary state per completed segment, and a
matched control that replaces one seeded-random non-boundary state per completed segment. The same
sampled positions are used for vanilla and BBC. `--mode zero` (default) writes zero to the selected
state; `--mode zero:k` instead selects the token `k` positions before each boundary for the interior
control. `--mode mean` writes the causal mean of the preceding `K-1` states, while `--mode mean:k`
writes the mean of the preceding `k` states (`mean:1` is the prior `last` intervention). The first
`--warmup_segments` segments (default `1`) are excluded from both intervention and NLL scoring.
The console reports each ablation delta as mean $\pm$ sample standard deviation, followed by a
two-sided paired t-test of the BBC-versus-vanilla boundary-minus-interior contrast.

For CC spectral residual compression, `--layer` defaults to the CC checkpoint's `continuous_concept_layer` and uses the same selected hidden-state index for both models:

```bash
python representation_diagnostics.py cc \
  --vanilla_checkpoint ./runs/vanilla_100m/final/checkpoint.pt \
  --cc_checkpoint ./runs/cc_100m/final/checkpoint.pt \
  --data_dir ./data/fineweb_edu_gpt2 \
  --output_dir ./runs/diagnostics/cc_100m \
  --layer mid
```

Each diagnostic saves one CSV row per example plus a JSON mean/standard-error summary. The CC command also writes `cc_spectral_residual.pdf`.

To regenerate only the CC figure from downloaded results, without loading checkpoints or data:

```bash
python representation_diagnostics.py \
  --plot_only \
  --summary_json ./cc_spectral_residual_summary.json \
  --sequence_length 1024 \
  --output_pdf ./cc_spectral_residual.pdf
```