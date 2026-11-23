# BlyxTorch

Train a classifier, fit a multi-output regression or run a convolution and transformer demo. The extension bundles BlyxTorch 0.4.0 for Windows x64, Linux x64 and macOS arm64. Native CPU training works offline. Optional device training uses an installed IREE or OpenXLA runtime. The Blyx Language dependency provides syntax highlighting and **Blyx: Run File**. [Portable training examples](https://github.com/Rahulchaube1/blyxtorch/blob/v0.4.0/PORTABLE.md) · [CPU and GPU experiments](https://www.blyx-lang.space/articles/blyxtorch-portable-training)

## Run a project in the editor

1. Create an empty file named `sensor.blyx`.
2. Type `bt-sensor` and accept the **BlyxTorch: Fit a desk sensor** completion. You can also use **Snippets: Fill File with Snippet** from the command palette.
3. Choose **Blyx: Run File**. The output panel shows 18.625 and 22.375 for two readings held out of training. Edit one input and run it again.

Use `bt-queue` in another empty Blyx file to classify four print jobs using the trained network. It prints short, long, short and long waits.

## Train on CSV files

Download the starter, extract it and open `BlyxTorch.code-workspace`. In a trusted workspace, use **Terminal → Run Task**. **BlyxTorch: Evaluate print queue** fits a model on 16 synthetic rows and scores six separate rows. **BlyxTorch: Predict new print jobs** trains and prints four probabilities. **BlyxTorch: Use saved queue weights** runs inference without retraining. [Download the starter projects](https://github.com/Rahulchaube1/blyxtorch/releases/download/v0.4.0/blyxtorch-projects-0.4.0.zip)

The native tasks support Windows x64, Linux x64 and macOS arm64. The portable single-file examples also run in the Blyx website playground. To train from your own terminal, use the native package for your platform. The native CPU workflow runs without Python or PyTorch. [Framework and installation guide](https://www.blyx-lang.space/packages/blyxtorch) · [Training handbook](https://doc.blyx-lang.space/guide/blyxtorch)

## Install without a marketplace connection

Download the Blyx Language VSIX and BlyxTorch VSIX on a connected machine. On the target desktop choose **Extensions: Install from VSIX** for Blyx Language first, followed by BlyxTorch. The trainer runs locally. [Blyx Language VSIX](https://github.com/Rahulchaube1/blyxxxx/releases/download/editor-v1.2.4/blyx-language-1.2.4.vsix) · [BlyxTorch release downloads](https://github.com/Rahulchaube1/blyxtorch/releases/tag/v0.4.0)
