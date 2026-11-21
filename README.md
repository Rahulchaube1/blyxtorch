<img src="assets/logo.png" width="88" height="88" alt="BlyxTorch">

# BlyxTorch

Train a small dense neural network, inspect its gradients, and ship it as a standalone CPU executable. BlyxTorch specializes the network shape ahead of time. Its differentiation rules, layer generator, losses, optimizers, checkpoint handling and command-line trainer are written in Blyx.

**Version 0.1.0** runs on Windows x64, Linux x64 and Apple Silicon macOS. The native downloads include a ready-to-train binary classifier. Python and PyTorch are only used by the optional independent verification tools. [Download the release](https://github.com/Rahulchaube1/blyxtorch/releases/tag/v0.1.0) · [Handbook](https://doc.blyx-lang.space/guide/blyxtorch) · [Browser example](https://www.blyx-lang.space/play?example=package-blyxtorch)

## Install with a package manager

On Apple Silicon macOS or x86-64 Linux, install the tested Homebrew formula.

```sh
brew install rahulchaube1/blyx/blyxtorch
```

On Windows x64, add the project bucket and install the command from PowerShell.

```powershell
scoop bucket add blyx https://github.com/Rahulchaube1/scoop-blyx
scoop install blyx/blyxtorch
```

Run `blyxtorch --help` to see the default network's input schema. With a package-manager install the command is on PATH. The examples below use the executable and fixtures supplied in the release archive. [Homebrew formula](https://github.com/Rahulchaube1/homebrew-blyx/blob/main/Formula/blyxtorch.rb) · [Scoop manifest](https://github.com/Rahulchaube1/scoop-blyx/blob/main/bucket/blyxtorch.json)

## Train your first network

Extract the release archive for your machine. These commands train a two-input network on the four XOR examples, save the weights and load them in a separate process for prediction.

```sh
./blyxtorch train fixtures/xor.csv 250 0.03 4 adam 7 > xor.bt
./blyxtorch predict xor.bt fixtures/xor-features.csv
./blyxtorch score xor.bt fixtures/xor.csv
```

In Windows PowerShell, use ASCII output for the checkpoint so it also works with Windows PowerShell 5.1.

```powershell
.\blyxtorch.exe train fixtures/xor.csv 250 0.03 4 adam 7 | Out-File -Encoding ascii xor.bt
.\blyxtorch.exe predict xor.bt fixtures/xor-features.csv
.\blyxtorch.exe score xor.bt fixtures/xor.csv
```

| Inputs | Target | Probability after training |
| --- | ---: | ---: |
| 0, 0 | 0 | 0.00365 |
| 0, 1 | 1 | 0.99391 |
| 1, 0 | 1 | 0.99308 |
| 1, 1 | 0 | 0.00846 |

The mean binary cross-entropy on these four training rows is approximately 0.00630. XOR demonstrates nonlinear learning and checkpoint round-tripping. Use separate held-out data to evaluate a model for a real application.

## Choose an architecture

Supply the widths from input to output, one objective and one hidden activation. The example builds a network for three classes with two input features and two hidden layers. Install Blyx 0.2.1, Rust 1.85 or newer, and a host linker on the build machine.

```sh
./blyxtorch emit 2,8,4,3 softmax relu > classifier.blyx
blyxc build classifier.blyx -o classifier
./classifier --help
```

In PowerShell, pipe `.\blyxtorch.exe emit 2,8,4,3 softmax relu` to `Out-File -Encoding ascii classifier.blyx`. Compile to `classifier.exe`. The generated file contains the full model and trainer. Once compiled, the executable runs separately from the generator, templates, Python and Rust. Keep the two supplied Blyx template files beside the generator when using `emit`.

| Objective | Training columns after `x0,x1,...` | Prediction |
| --- | --- | --- |
| `mse` | One real-valued `y0,y1,...` per output | Raw output values. Loss averages squared error over outputs. |
| `binary` | One `y0` from 0 through 1 | Probability from a single output logit. Training uses stable binary cross-entropy with logits. |
| `softmax` | One integer `class` from 0 through output width minus one | A probability per class. Training uses log-sum-exp categorical cross-entropy. |

Hidden activations are `tanh`, `relu` and `sigmoid`. A model with just input and output widths is an affine model. The release accepts up to eight affine layers, widths from 1 through 64 and up to 1,024 trainable parameters in total. Shapes are fixed at generation. Computation uses CPU float64. GPU execution, arbitrary-rank tensors, convolution and transformer layers are outside this release.

## Commands and data

| Command | Result |
| --- | --- |
| `init [seed]` | Write deterministic initial weights to a checkpoint on standard output. |
| `train DATA [epochs] [rate] [batch] [adam or sgd] [seed]` | Train from initialization. Defaults are 100 epochs, learning rate 0.03, batch size 32, Adam and seed 7. |
| `predict CHECKPOINT FEATURES.csv` | Emit a CSV of predictions in input order. |
| `score CHECKPOINT LABELLED.csv` | Return a JSON row count and mean loss. |
| `gradient CHECKPOINT LABELLED.csv` | Return the loss and every parameter derivative separately for each observation. |

Use UTF-8 unquoted numerical CSV. Feature columns are `x0,x1,...` in order, followed by the target columns for training, scoring and gradients. Prediction takes feature columns alone. Ask the compiled model for its exact header with `--help`. The trainer loads the file into memory and reuses its row order across epochs. A final short mini-batch divides by its actual size.

SGD applies the mean mini-batch gradient. Adam uses first and second moment decay of 0.9 and 0.999, bias correction per update, and epsilon of 1e-8. Initial weights use a seeded symmetric fan-in/fan-out scale and zero biases. A checkpoint contains a model identity line and the weights in round-trippable decimal form. The loader checks the architecture, objective, activation, parameter count and finiteness. Checkpoints contain inference weights only.

## Use automatic differentiation from Blyx

Import `blyxtorch.blyx` for portable forward-mode differentiation. A dual value contains a value and a tangent. Seed a parameter's tangent with 1 and a constant's with 0. Reuse variables freely in the expression.

```blyx
use blyxtorch;

fn main() {
    let w = bt_dual(1.0, 1.0);
    let x = bt_dual(2.0, 0.0);
    let target = bt_dual(5.0, 0.0);
    let loss = bt_square(bt_sub(bt_mul(w, x), target));
    println!(at(loss, 0));
    println!(at(loss, 1));
}
```

This prints 9 and −12. The library supplies addition, subtraction, multiplication, division, square, ReLU, tanh, sigmoid, softplus and binary cross-entropy with logits. Domain or shape errors return an empty vector. ReLU's chosen derivative at zero is zero. [Complete example](example.blyx)

A generated model module uses reverse-mode differentiation for all trainable parameters in one backward pass. From a neighboring file, import it with `use classifier;` and call the functions below.

| Function | Return |
| --- | --- |
| `bt_inputs(), bt_outputs(), bt_targets(), bt_parameters()` | Static dimensions |
| `bt_init(seed)` | Initial parameter vector |
| `bt_predict(weights, features)` | Prediction vector |
| `bt_sample_loss(weights, features, target)` | One-element loss vector |
| `bt_loss_grad(weights, features, target)` | Loss followed by all weight and bias derivatives |

Within each layer, parameters are the output-by-input weight matrix in row-major order followed by the bias vector. Layers appear in forward order. Predictions and gradients return an empty vector for invalid input. The CLI exits with a diagnostic for invalid data.

## Verification and performance

The independent oracle compares predictions, per-row losses, parameter derivatives and entire multi-epoch SGD and Adam updates against CPU PyTorch in float64. Five networks cover the three objectives, all hidden activations, two hidden layers and a partial final mini-batch. The verifier also checks selected derivatives by central differences. [Verification source](verification/pytorch_reference.py) · [Recorded results](verification/parity.json) · [CI runs](https://github.com/Rahulchaube1/blyxtorch/actions)

On one Windows x64 CPU, scoring 4,096 CSV rows took 59.3 ms for a new BlyxTorch process, 3.52 s for a new Python process loading PyTorch, and 6.8 ms with PyTorch already loaded. These are medians of seven runs in rotating order. All three read the checkpoint and CSV, score the same small model and produce the same mean loss. The warm PyTorch path uses vectorized batches. GPU, larger models and training throughput need separate measurements. [Raw timings and environment](verification/score-benchmark.json) · [Benchmark harness](verification/bench_score.py)

## Build and contribute

```sh
blyxc build main.blyx -o blyxtorch
blyxc native-run tests.blyx
blyxc native-run example.blyx
python verification/test_cli.py --exe ./blyxtorch
```

On Windows compile with `-o blyxtorch.exe`. The shipped default module can be reproduced from the repository root with `blyxc native-run generate.blyx -- 2,4,1 binary tanh`. CLI tests check it against the committed source. Developers with CPU PyTorch installed can run `python verification/pytorch_reference.py --exe ./blyxtorch`. Please include the model widths, input, expected output and command when reporting an issue. [Issues](https://github.com/Rahulchaube1/blyxtorch/issues) · [Blyx community](https://www.blyx-lang.space/community)

BlyxTorch is an independent project by Rahul Chaube and contributors. It is distributed under your choice of the [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) license.
