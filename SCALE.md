# Classical simulation of trainable circuits

Each layer rotates every qubit about Y, then applies an ascending chain of nearest-neighbour CNOT gates. Angles are in radians and qubit zero is the least significant bit. The readout is the expectation of Pauli Z on one selected qubit. An adjoint sweep recovers intermediate states by reversing the gates and computes one derivative per angle. Training minimizes half the squared difference between the readout and a target in [−1, 1].

## Try the native command

Extract the archive for your platform from the [BlyxTorch 0.5.0 release](https://github.com/Rahulchaube1/blyxtorch/releases/tag/v0.5.0). Add its directory to PATH or use the executable’s full path. Run the fixture commands from the extracted directory.

~~~sh
blyxtorch q-expect 2 1 0.3,-0.7,0.2,0.6
blyxtorch q-grad 2 1 0.3,-0.7,0.2,0.6
blyxtorch q-fit 2 1 0.3,-0.7,0.2,0.6 -0.25 0.2 40
blyxtorch q-qasm 2 1 0.3,-0.7,0.2,0.6 > circuit.qasm
quantum-study adjoint 10 3 4
~~~

## Scope and input contract

The CPU simulator accepts 1 through 16 qubits, positive depth and finite angles up to one million radians in magnitude. Each state vector contains 2^q real doubles. OpenQASM 3 export emits gates and a final measurement for an external backend. Physical hardware, noise models and arbitrary gate sets fall outside this implementation.

## Import in Blyx

The executable includes its runtime. To compile the source example, install the tested compiler revision and a host linker, then run these commands from the tagged source tree.

~~~sh
cargo install --locked --git https://github.com/Rahulchaube1/blyxxxx --rev 120d6b6ec6df09ff018d7c18752fb3297b5e629d blyxc
blyxc native-run scale_example.blyx
~~~

~~~blyx
use quantum;

fn main() {
    let angles = [0.3, -0.7, 0.2, 0.6];
    let mut learned = angles;
    let mut step = 0;
    while step < 40 { learned = bt_q_step(learned,2,1,-0.25,0.2); step += 1; }
    let z = bt_q_expect(learned,2,1);
    println!(z > -0.26 && z < -0.24);
}
~~~

## Verification

An independent dense-matrix and parameter-shift reference checked 18 expectations and 116 partial derivatives on circuits of one through five qubits. The maximum absolute gradient discrepancy was 1.34 × 10⁻¹⁵ or less. The two-qubit training example approached −0.24999977 after 40 steps. [Study, raw timings and reproduction instructions](https://www.blyx-lang.space/articles/training-in-factors)
