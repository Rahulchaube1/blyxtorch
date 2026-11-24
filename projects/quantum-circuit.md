# Differentiate a quantum circuit

Each layer rotates every qubit about Y, then applies an ascending chain of nearest-neighbour CNOT gates. Angles are in radians and qubit zero is the least significant bit. The readout is the expectation of Pauli Z on one selected qubit. An adjoint sweep recovers intermediate states by reversing the gates and computes one derivative per angle. Training minimizes half the squared difference between the readout and a target in [−1, 1].

An independent dense-matrix and parameter-shift reference checked 18 expectations and 116 partial derivatives on circuits of one through five qubits. The maximum absolute gradient discrepancy was 1.34 × 10⁻¹⁵ or less. The two-qubit training example approached −0.24999977 after 40 steps.

[Native commands and compiler installation](https://github.com/Rahulchaube1/blyxtorch/blob/v0.5.0/SCALE.md)

Open the workspace and choose **Terminal → Run Task → BlyxTorch: Fit a quantum circuit**. The four outputs are updated rotation angles in radians.
