# Train a model on fixed features

BlyxTorch 0.4 fits two kinds of models at runtime, a softmax classifier and a multi-output linear regressor. They accept numerical features from a sensor, a spreadsheet or a frozen encoder. Save the feature order, units and preprocessing alongside your checkpoint. Predictions depend on all three.

## Run a small classifier

Extract the native archive for your platform. The commands below use the six labelled rows in the archive. Each row has two features and one of three class IDs.

```sh
./blyxtorch probe-init 2 3 > initial.bt
./blyxtorch probe-fit initial.bt fixtures/probe-routing.csv 40 0.2 4 > routing.bt
./blyxtorch probe-predict routing.bt 1,0
./blyxtorch probe-score routing.bt fixtures/probe-routing.csv
```

Prediction returns a JSON object containing `class_index` and the three ordered probabilities. The score on this file describes the training examples. Pass a separate labelled file to measure generalization. On Windows use `.\blyxtorch.exe`. In Windows PowerShell 5.1 use `| Out-File -Encoding ascii initial.bt` in place of shell redirection for text checkpoints.

For ticket routing with a frozen embedding model, each ticket becomes a row of numbers. Keep the model version constant across training and prediction, assign your own label to each training ticket and reserve a set of newer tickets for evaluation. Pass a wide embedding as a file containing a single comma-separated row without a header.

```sh
./blyxtorch probe-predict-file routing.bt one-row.csv
```

For regression, create a checkpoint with `probe-init FEATURES OUTPUTS regression`. Feature columns must be named `x0,x1,...` followed by real-valued `y0,y1,...`. Classification uses a final `class` column with a zero-based decimal integer. All values must be finite. `probe-score` reports pooled RMSE for regression or correct count and accuracy for classification.

## Move from a CPU to a device

The native CPU trainer uses f64 and needs only the executable. The optional device route uses f32. Blyx emits the training graph, including its derivatives and SGD update. The packaged Python host reads one batch at a time, selects a runtime, compiles each encountered shape once per process and retains the weights on the device between updates. The last batch uses its actual row count. A new process can resume training from the resulting weights with either route. Supply a distinct output filename when resuming.

```sh
python -m pip install numpy iree-base-compiler==3.14.0 iree-base-runtime==3.14.0
./blyxtorch probe-device-fit vulkan initial.bt fixtures/probe-routing.csv 40 0.2 4 > gpu.bt
./blyxtorch probe-predict gpu.bt 1,0
```

Set `BLYXTORCH_PYTHON` to the absolute path of the interpreter with the optional runtime installed. Keep the shipped `device_fit.py` alongside the executable.

| Selection | Execution path | Verification for 0.4 |
| --- | --- | --- |
| `probe-fit` | Native f64 CPU | End-to-end classification and regression |
| `probe-device-fit cpu` | IREE LLVM CPU | Training, checkpoint reload and layer-gradient comparisons |
| `probe-device-fit vulkan` | IREE Vulkan f32 | Intel UHD GPU, including held-out accuracy and regression |
| `probe-device-fit xla-cpu` | JAX/PJRT CPU | Reference gradients and minibatch training |
| `probe-device-fit tpu` | JAX/PJRT, one TPU device | Requires a TPU-enabled JAX installation and hardware verification |

The device selector requires the requested backend. A missing runtime or device exits with an error. The OpenXLA path was exercised with JAX 0.12.1. Distributed training, CUDA-specific kernels and dense trillion-parameter training are outside this release's verified capabilities. [IREE Vulkan setup](https://iree.dev/guides/deployment-configurations/gpu-vulkan/) · [JAX installation](https://docs.jax.dev/en/latest/installation.html)

## Export the computation

```sh
./blyxtorch probe-graph train 32 64 10 > update.mlir
./blyxtorch probe-graph predict 1 64 10 > predict.mlir
./blyxtorch layer-graph conv2d-backward 2 5 6 2 3 3 2 2 1 > conv-grad.mlir
./blyxtorch layer-graph attention 5 2 3 causal > attention.mlir
```

`probe-graph` accepts `train`, `predict`, `regress-train` and `regress-predict`, followed by batch size, features and outputs. A training graph takes features, weights, bias, targets and a scalar learning rate. Classification device targets are one-hot f32. Regression uses half the sum of squared output errors per sample, averaged across the batch. The output is the updated weight matrix and bias. The predict graph returns probabilities or raw regression values.

`layer-graph conv2d` and `conv2d-backward` take batch size, input height, input width, input channels, output channels, kernel height, kernel width, stride and symmetric padding. Forward takes NHWC input, OHWI weights and a bias vector. Backward takes input, weights and the upstream gradient, and returns input, weight and bias gradients. Attention takes token count, head count, head width and either `causal` or `full`. Supply Q, K and V as [heads, tokens, head width]. Backward additionally takes the upstream gradient and returns gradients for all three inputs. These exports are individual operations that a caller can compose. [StableHLO specification](https://openxla.org/stablehlo/spec)

## Use a checkpoint from an AI application

`probe-tool openai` produces a Responses API function definition. `probe-tool claude` produces an Anthropic client tool definition. The optional Node module in the native archive runs the selected local classifier and returns its JSON result to your application. It passes arguments directly to the executable and handles vectors longer than a Windows command line.

```js
import { localClassifier, openAIClassification, claudeClassification } from "./provider-tools.mjs";
const classifier = localClassifier("./blyxtorch", "./routing.bt");
const local = await classifier.call("classify_features", { features: [1, 0] });
console.log(local.class_index);
// With your initialized SDK client and an available tool-capable model:
// await openAIClassification(openaiClient, modelID, prompt, classifier);
// await claudeClassification(anthropicClient, modelID, prompt, classifier);
```

Your application decides which executable and checkpoint the tool may use. Supply features from the same pipeline as training. A generated list of numbers is a poor substitute for a real embedding or a measured signal. The bridge was checked against local protocol fixtures. Calling a hosted API requires your own account and credentials. The checkpoint trained here belongs to your local classifier. [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling) · [Anthropic client tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)

[Read the experiment and see the measurements](https://www.blyx-lang.space/articles/blyxtorch-portable-training) · [Reproduce the checks](https://github.com/Rahulchaube1/blyxtorch-verification)
