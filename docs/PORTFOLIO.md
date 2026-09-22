# From Sequence to Spatial: Mamba S6 From Scratch

A collaborative MVA Deep Learning project by **Clement Marie and Ying JIN**. This fork preserves [clmrie/Mamba-S6-From-Scratch](https://github.com/clmrie/Mamba-S6-From-Scratch) and its source history. The [project poster](../poster/poster.pdf) records both authors.

## Project scope

The repository implements selective state-space computations (S6) in PyTorch, including a JIT-scripted scan, and studies a bidirectional image-model variant. The experiments use MNIST pixel-sequence classification and TinyShakespeare language modeling, with an RNN and causal/bidirectional Mamba comparisons.

Mamba and selective state-space models are existing methods. This is a student implementation and experimental study, not an invention claim or a production inference library. The project is coauthored; the reviewed poster does not assign every implementation or experiment to one individual.

## Setup and entry points

Install dependencies in an isolated environment and run from the repository root:

```bash
python -m pip install torch torchvision pandas matplotlib requests
python train_mnist.py
python train_mnist_causal.py
python train_rnn_baseline.py
```

The current upstream/fork uses `train_mnist.py`. An older local coursework snapshot used a different filename; the commands here match the current fork. No requirements file is present in this fork, so the command above lists dependencies found in source imports, without claiming version compatibility has been tested. The MNIST scripts automatically select CUDA or CPU and download MNIST into `data/`. The causal and bidirectional scripts use five epochs, with settings defined in their source. They write CSV logs under `results_mnist/`; the bidirectional script also writes checkpoints under `checkpoints_mnist/`. Repeating a run can overwrite its logs.

```bash
python train_shakespeare.py
```

The language-model script downloads TinyShakespeare to `shakespeare.txt` and uses its source-defined training configuration (10 epochs, context length 128, batch size 32). These commands start training; they are not quick validation tests.

## Results and validation

The original repository and poster contain the recorded experiment results and figures. Compare models only under the stated dataset, architecture, training budget and metric. An MNIST accuracy difference does not establish a general language-model quality improvement or a hardware-independent speedup.

Entry-point names, data download behavior and configuration were inspected in the freshly cloned upstream-derived fork on 22 September 2026. Training was **not rerun**, and no new numerical result is claimed by this README update.

## Attribution and reuse

Retain the upstream history and both authors' credit. No explicit LICENSE file was found in the current fork; this documentation update does not add or assume one. Cite the methods and the original project when discussing the work.
