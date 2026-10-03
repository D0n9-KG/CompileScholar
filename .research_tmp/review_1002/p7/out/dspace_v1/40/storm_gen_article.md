## Related Work

**ZKP Protocol Paradigms**
Prior research on ZKP acceleration has largely focused on protocols requiring a trusted setup for each application, such as Groth16 and Pinocchio, which are accelerated by systems like SZKP [12]. While these protocols offer efficient verification, the requirement for per-application trusted setups limits their applicability in decentralized environments. In contrast, recent efforts have shifted toward universal setup protocols that support small proof sizes, such as HyperPlonk, which is utilized in systems like VeriZexe for decentralized private computation [14]. This paper aligns with the latter paradigm, targeting HyperPlonk to overcome the trade-offs inherent in trusted setup protocols while maintaining small proof sizes suitable for public verification.

**Acceleration Scope**
A significant portion of prior work focuses on accelerating single primitives, such as Multi-Scalar Multiplications (MSMs) or Fast Fourier Transforms (FFTs), rather than the entire proof generation pipeline. For instance, several studies propose specialized hardware or software optimizations exclusively for MSMs, including PriorMSM [1], multi-GPU systems [2], cuZK [3], Myosotis [5], MSMAC [6], Elastic MSM [7], and high-performance NTT/MSM accelerators [8]. Other approaches address scalability and memory efficiency through chiplet-based techniques [9] or hardware-algorithm co-design [11]. While some systems accelerate the full protocol, such as BatchZK [10], SZKP [12], and PipeZK [13], or offer reconfigurable architectures like ReZK [4], this work distinguishes itself by developing a dedicated accelerator for the entire HyperPlonk protocol, specifically optimizing both SumCheck and MSMs to achieve holistic end-to-end speedups.

**Hardware Implementation**
Hardware implementations for ZKP acceleration vary widely, ranging from general-purpose GPU acceleration [2, 3, 7, 10] to custom ASICs designed for specific primitives [1, 5, 8]. Some works explore reconfigurable ASICs [4], chiplet-based architectures for scalability [9], or pipelined designs [13], while others focus on hardware-algorithm co-design [11]. SZKP represents a closest prior effort in this dimension, employing a full-chip custom ASIC architecture to accelerate zero-knowledge proofs on-chip [12]. This paper follows a similar trajectory by implementing a full-chip custom ASIC with 366.46 mm² area and 2 TB/s bandwidth, but it is specifically tailored to the HyperPlonk protocol to maximize performance for its unique computational requirements.

**Target Application Context**
The target application context significantly influences design choices in ZKP accelerators. Prior work such as SZKP is primarily oriented toward private or point-to-point verification scenarios [12]. In contrast, systems like VeriZexe are designed for publicly verifiable, consensus-based systems, such as blockchain networks, where numerous verifiers require fast verification times and small proof sizes [14]. This paper targets the latter context, optimizing the accelerator for publicly verifiable, consensus-based systems to ensure that the small proof sizes and efficient verification capabilities of HyperPlonk are fully leveraged in high-concurrency blockchain environments.

## References

[1] PriorMSM: An Efficient Acceleration Architecture for Multi-Scalar Multiplication
[2] Accelerating Multi-Scalar Multiplication for Efficient Zero Knowledge Proofs with Multi-GPU Systems
[3] cuZK: Accelerating Zero-Knowledge Proof with A Faster Parallel Multi-Scalar Multiplication Algorithm on GPUs
[4] ReZK: A Highly Reconfigurable Accelerator for Zero-Knowledge Proof
[5] Myosotis: An Efficiently Pipelined and Parameterized Multi-Scalar Multiplication Architecture via Data Sharing
[6] MSMAC: Accelerating Multi-Scalar Multiplication for Zero-Knowledge Proof
[7] Elastic {MSM}: A Fast, Elastic and Modular Preprocessing Technique for Multi-Scalar Multiplication Algorithm on {GPUs}
[8] A High-performance NTT/MSM Accelerator for Zero-knowledge Proof Using Load-balanced Fully-pipelined Montgomery Multiplier
[9] Chiplet-Based Techniques for Scalable and Memory-Aware Multi-Scalar Multiplication
[10] {BatchZK}: A Fully Pipelined {GPU}-Accelerated System for Batch Generation of Zero-Knowledge Proofs
[11] Accelerating Zero-Knowledge Proofs Through Hardware-Algorithm Co-Design
[12] SZKP: A Scalable Accelerator Architecture for Zero-Knowledge Proofs
[13] PipeZK: Accelerating Zero-Knowledge Proof with a Pipelined Architecture
[14] VeriZexe: Decentralized Private Computation with Universal Setup
[15] Poseidon: A New Hash Function for Zero-Knowledge Proof Systems