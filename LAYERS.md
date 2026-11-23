# Native layers in BlyxTorch 0.3

The native CPU API trains shared filters, multi-head transformer blocks and sparse embedding rows. Each function accepts flat f64 vectors and explicit integer shapes. The download contains a compiled CLI. Building the layer sources requires the Blyx compiler revision with `std::buffer`. [Compiler source](https://github.com/Rahulchaube1/blyxxxx/commit/120d6b6ec6df09ff018d7c18752fb3297b5e629d)

```sh
blyxtorch layer-demo
blyxc native-run sequence_example.blyx
```

The second command learns a running mean. Each token contains a sensor value and its position. A causal mask lets each prediction use the current reading and earlier readings. The example trains on 64 synthetic streams for 1,200 SGD steps. On 16 held-out streams the mean squared error went from 0.907813 to 0.005154. You can change the signal generator, inspect the targets and reproduce the split. [Example source](sequence_example.blyx)

## Convolution

`bt_conv1d` accepts `[time, input_channels]`. `bt_conv2d` accepts `[height, width, input_channels]`. They use cross-correlation with a shared kernel, configurable stride and symmetric zero padding. The 2D kernel can be rectangular. Output channels are contiguous. The parameter vector contains the kernel in `[output_channel, kernel_y, kernel_x, input_channel]` order followed by one bias per output channel. For a 1D kernel, omit the height dimension. Each spatial output extent is `floor((input + 2*padding - kernel)/stride) + 1`. [Blyx implementation](convolution.blyx) · [Keras convolution convention](https://keras.io/api/layers/convolution_layers/convolution2d/)

```blyx
use convolution;
use tensor_ops;

fn main() {
    let signal = [2,3,5,7];
    let kernel = [1,0,-1,0];
    let y = bt_conv1d(signal,4,1,1,3,1,0,kernel);
    println!(y); // [-3, -4]
    let grad = bt_conv1d_backward(signal,4,1,1,3,1,0,kernel,[1,1]);
    println!(bt_slice(grad,0,4)); // input gradient [1, 1, -1, -1]
    println!(bt_slice(grad,4,4)); // weight and bias gradient [5, 8, 12, 2]
}
```

Use `bt_conv_count(kh,kw,inputs,outputs)` to size the parameter vector. Both backward functions take the gradient arriving from the next layer. They return the input gradient followed by all parameter gradients. Reusing a kernel accumulates contributions from every output position.

## Transformers

A block consumes `[tokens, width]` and returns the same shape. It applies per-token layer normalization, learned query/key/value projections, scaled dot-product attention, an output projection and a residual connection. A second normalization feeds two affine layers with a ReLU between them and another residual connection. Set `causal` to `true` for prefix-only attention or `false` for bidirectional attention. The number of heads must divide the width. Add position information to the input when a task depends on order. [Block implementation](transformer.blyx) · [Attention and normalization](attention.blyx) · [Vaswani et al.](https://arxiv.org/abs/1706.03762)

| Function | Result |
| --- | --- |
| `bt_transformer_count(width, hidden)` | Exact parameter count for one block |
| `bt_transformer_init(width, hidden, seed)` | Seeded Xavier-uniform matrices, zero biases, unit normalization gains |
| `bt_transformer(x, tokens, width, heads, hidden, causal, p)` | Per-token output |
| `bt_transformer_backward(x, tokens, width, heads, hidden, causal, p, dy)` | Input gradient followed by every parameter gradient, in initialization order |
| `bt_sgd_step(p, dp, rate)` | Updated parameter vector |

For a block of width `D` and feed-forward width `F`, the parameter count is `4D² + 2DF + 9D + F`. The parameter order is Q, K, V, attention output, feed-forward expansion, feed-forward contraction, first normalization and second normalization. Within each affine layer weights use `[output,input]` order followed by biases. Each normalization stores gain then bias and uses epsilon 0.00001. Stack blocks by passing one block's output into the next and propagating input gradients in reverse order.

Attention subtracts each row's largest score before taking exponentials. Backward recomputes one row of attention probabilities at a time. Attention arithmetic still grows quadratically with sequence length. This release uses single-process CPU f64 and full in-memory dense weights. GPU kernels, mixed precision and distributed dense training remain future work. [Independent PyTorch comparison](https://github.com/Rahulchaube1/blyxtorch-verification/blob/2307476b713b4ba0cdde4e8d59845b83ede0585f/layer_reference.py)

## Sparse embeddings and checkpoint files

An embedding is a table of trainable vectors indexed by integer ID. Unseen rows start at zero. A lookup or SGD update only needs the active rows, and the checkpoint stores each active row once. The API accepts exact signed 64-bit nonnegative row IDs and checks the product of rows and width for overflow. Internal row keys use two integer limbs so adjacent IDs above 2⁵³ remain distinct. Lookup scans materialized rows, and loading a checkpoint reads it into memory. This representation suits trials with a small active set. [Sparse embedding implementation](embedding.blyx)

The following POSIX shell session updates four scalar weights in a table with 4,000,000,000,004 logical parameters. Send each new checkpoint to a different path so its input remains available during the update. On Windows, the same commands work in `cmd.exe` with `blyxtorch.exe`.

```sh
blyxtorch embedding-init 1000000000001 4 > empty.bt
blyxtorch embedding-step empty.bt 1000000000000 0.5 2,4,6,8 > trained.bt
blyxtorch embedding-get trained.bt 1000000000000
# [-1.0, -2.0, -3.0, -4.0]
```

The checkpoint test addresses **36,028,797,018,963,980** possible scalar weights. It updates two neighboring rows, reopens the file in a fresh process, checks all eight learned values and verifies an untouched row. The Windows fixture occupied 86 bytes including its header. [Checkpoint regression test](https://github.com/Rahulchaube1/blyxtorch-verification/blob/2307476b713b4ba0cdde4e8d59845b83ede0585f/test_embeddings.py)

## Dense capacity and verification

`blyxtorch transformer-plan LAYERS WIDTH HEADS FEED_FORWARD VOCAB` calculates the parameter count for stacked blocks, one tied token embedding/output matrix and a final layer normalization. The plan assumes external position features. It allocates no model weights. For example, `transformer-plan 128 32768 128 131072 128000` reports 1,653,516,337,152 parameters and 13,228,130,697,216 bytes for f64 weights alone. Gradients, optimizer state, activations and temporary buffers require additional memory. Published dense trillion-parameter training systems partition both the computation and the training state across accelerators. [Narayanan et al.](https://arxiv.org/abs/2104.04473) · [Rajbhandari et al.](https://arxiv.org/abs/1910.02054)

The dense training test instantiates **41,728 trainable parameters** and takes a full backward and SGD step. Smaller reference fixtures compare all outputs and input/parameter gradients against CPU PyTorch for strided, padded 1D and 2D convolutions and for both attention masks. Across 520 scalar comparisons the largest absolute difference was 1.56 × 10⁻¹⁵ or less. The tests also check future-token isolation, large attention logits, row ID collisions and checkpoint errors. These are correctness measurements. [Blyx tests](layer_tests.blyx) · [Recorded parity results](verification/layer-parity.json)
