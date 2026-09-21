# Quantum state preparation, tomography, and entanglement of mechanical oscillators

E. Alex Wollack, $^{*}$ Agnetta Y. Cleland, $^{*}$ Rachel G. Gruenke, Zhaoyou

Wang, Patricio Arrangoiz-Arriola, and Amir H. Safavi-Naeini $^{†}$

Department of Applied Physics and Ginzton Laboratory, Stanford University

348 Via Pueblo Mall, Stanford, California 94305, USA

(Dated: October 15, 2021)

Precisely engineered mechanical oscillators keep time, filter signals, and sense motion, making them an indispensable part of today's technological landscape. These unique capabilities motivate bringing mechanical devices into the quantum domain by interfacing them with engineered quantum circuits. Proposals to combine microwave-frequency mechanical resonators with superconducting devices suggest the possibility of powerful quantum acoustic processors $^{1-3}$ . Meanwhile, experiments in several mechanical systems have demonstrated quantum state control and readout $^{4,5}$ , phonon number resolution $^{6,7}$ , and phonon-mediated qubit-qubit interactions $^{8,9}$ . Currently, these acoustic platforms lack processors capable of controlling multiple mechanical oscillators' quantum states with a single qubit, and the rapid quantum non-demolition measurements of mechanical states needed for error correction. Here we use a superconducting qubit to control and read out the quantum state of a pair of nanomechanical resonators. Our device is capable of fast qubit-mechanics swap operations, which we use to deterministically manipulate the mechanical states. By placing the qubit into the strong dispersive regime with both mechanical resonators simultaneously, we determine the resonators' phonon number distributions via Ramsey measurements. Finally, we present quantum tomography of the prepared nonclassical and entangled mechanical states. Our result represents a concrete step toward feedback-based operation of a quantum acoustic processor.

The burgeoning field of quantum acoustics combines the established tools and infrastructure of circuit quantum electrodynamics (cQED) with the many benefits of nanomechanical oscillators. This creates a rich platform for explorations of fundamental quantum physics $^{4-7,10-12}$ , with promising applications toward scalable quantum computation $^{1,2,13}$ . Over a small footprint, mechanical systems have the potential to provide access to a large number of highly coherent microwave-frequency modes which can act as high-precision sensors of force and motion, store long-lived quantum memories with minimal crosstalk, and form interconnects with optical systems. Furthermore, it is possible to generate nonclassical $^{4,5}$ and entangled states of motion $^{14-19}$ in mechanical oscillators, making them a compelling system for storing and processing quantum information. By placing these acoustic systems in the strong dispersive coupling limit $^{20,21}$ , both non-Gaussian and non-demolition measurements can be made via phonon-number resolved detection.

Access to this regime is enabled by our device design and heterogeneously integrated material platform. We leverage the small mode volume and strong piezoelectricity of a phononic crystal resonator in thin-film lithium niobate (LN), combined with a high coherence aluminum transmon qubit, to achieve large coupling rates between a superconducting qubit processor and two nanomechanical resonators. Our approach allows for strong coupling while suppressing the phonon radiation loss channels that arise in piezoelectric materials. In phononic crystal devices, the density of states for acoustic radiation loss can be eliminated over a wide frequency range by choosing a periodic geometry that produces a full phononic bandgap $^{11}$ . This approach localizes the gigahertz frequency mechanical mode to a wavelength-scale volume $^{22}$ , and has produced resonators with extremely long mechanical lifetimes $^{23}$ . With improved fabrication processes (see methods), we have extended both the qubit and mechanical resonators' coherence times, $T_{1}$ and $T_{1,m}$ , which limited experimental capabilities in prior work $^{6}$ .

Our hybrid device is composed of two chips integrated in a flip-chip architecture $^{24}$ (Fig. 1a). We fabricate a frequency-tunable transmon qubit $^{25}$ with microwave control lines and a coplanar waveguide readout resonator (Fig. 1b) on a $6\mathrm{mm} \times 9\mathrm{mm}$ silicon chip. The qubit is capacitively coupled through a small vacuum gap to two phononic crystal resonators fabricated on a separate $2\mathrm{mm} \times 4\mathrm{mm}$ top chip (Fig. 1c). These cavities are patterned by argon ion milling a thin film of $\mathrm{LN}^{26}$ , which is then released from the chip's silicon handle by a xenon difluoride dry etch $^{6,27}$ . Each mechanical eigenmode is confined to a small defect site suspended on either side by a one-dimensional phononic crystal mirror $^{11,22}$ . Utilizing the piezoelectric effect of LN, the qubit couples to the mechanical modes via aluminum electrodes patterned on each resonator. These electrodes extend to a metallized pad which forms the top half of a cross-chip coupling capacitor, with a matching pad on the qubit island. The capacitor gap is defined by the flip-chip separation distance of $1\mu \mathrm{m}$ (see methods for flip-chip procedure).

The Hamiltonian for the resulting device includes two mechanical oscillators with frequencies $\omega_{m_{i}}$ and lowering operators $\hat{b}_{i}$ , in addition to a qubit with transition frequency $\omega_{ge}$ and Pauli operators $\hat{\sigma}$ : $\hat{H}_{0} = \omega_{m_{1}}\hat{b}_{1}^{\dagger}\hat{b}_{1} + \omega_{m_{2}}\hat{b}_{2}^{\dagger}\hat{b}_{2} + \frac{1}{2}\omega_{ge}\hat{\sigma}_{z}$ . A direct piezoelectric coupling be-

![](images/20179c84f9541695ef8a9d1e75b38132ddcf65e4853578684e2fd52221924cdf.jpg)

<details>
<summary>text_image</summary>

a
mechanics
ωm1 ωm2
g1 g2
e ωge
qubit
top chip
bottom chip
</details>

![](images/256098662a9cbc0c1f03c735eb6c45a747eb2ab52b2123e3f1dc47b00eac7994.jpg)

<details>
<summary>text_image</summary>

b
500 µm
c
LiNbO₃
Al
1 µm
</details>

FIG. 1. Device description. a, Schematic of the modes and flip-chip device. A frequency-tunable qubit on the bottom chip (blue) is capacitively coupled through a small vacuum gap to two mechanical modes on the top chip (orange). The mechanical modes are represented as Butterworth-van Dyke equivalent circuits. b, Optical micrograph of the bottom (qubit) chip, with inset showing the qubit's SQUID and adjacent flux-line, used for frequency control. The rightmost arm of the transmon island extends to form the bottom pad of the coupling capacitor. c, False-color scanning electron micrograph of the top (mechanics) chip, showing two phononic crystal resonators (red). Aluminum electrodes (orange) are galvanically connected both to the top chip's coupling capacitor pad and ground plane, as shown in the inset.

tween the qubit and mechanics leads to an interaction Hamiltonian $\hat{H}_{\mathrm{int}} = \sum_{i} g_{i} (\hat{b}_{i} + \hat{b}_{i}^{\dagger}) \hat{\sigma}_{x}$ , with coupling rates $g_{i}$ . In the limit of large detuning, the interaction is best described by an effective dispersive Hamiltonian $^{28}$

$$
\hat {H} _ {\mathrm{eff}} = \hat {H} _ {0} + (\chi_ {1} \hat {b} _ {1} ^ {\dagger} \hat {b} _ {1} + \chi_ {2} \hat {b} _ {2} ^ {\dagger} \hat {b} _ {2}) \hat {\sigma} _ {\mathrm{z}}.
$$

In this regime, each mechanical mode imparts a frequency shift of $2\chi_{i}$ per phonon on the qubit. This dispersive coupling rate $\chi_{i}$ is related to the qubit anharmonicity $\alpha_{q}$ , each mechanical mode's coupling rate $g_{i}$ , and the detunings $\Delta_{i} = \omega_{ge} - \omega_{m_{i}}$ by $^{28}$

$$
\chi_ {i} = - \frac {g _ {i} ^ {2}}{\Delta_ {i}} \frac {\alpha_ {q}}{\Delta_ {i} - \alpha_ {q}}.
$$

The time required to resolve these phonon-induced frequency shifts is roughly $\pi/\chi_{i}$ , making it important for $\chi_{i}$ to exceed the decoherence rates of both the mechanical resonators and the qubit. A system that satisfies this condition, while maintaining the detuning requirement $\Delta_{i} \gg g_{i}$ for the effective Hamiltonian to hold, is said to be in the strong dispersive coupling regime $^{21}$ , which has only recently been demonstrated for circuit quantum acoustic devices $^{6,7}$ . A useful figure of merit for devices in this regime is the dispersive cooperativity $C = 4\chi^{2}T_{1}T_{1,m}$ , which our device improves to C = 490 compared with C = 170 in previous work in quantum acoustics $^{6}$ .

For this experiment, we leverage established techniques in cQED to perform state preparation and readout of the qubit (see methods), allowing characterization of the mechanical resonators using the qubit as a probe. We control the qubit frequency by flowing current through an on-chip flux-line shown in Fig 1b. Tuning the qubit yields avoided crossings in the qubit spectrum at both the lower and upper mechanical frequencies, $\omega_{\mathrm{m_1}} / 2\pi = 2.053\mathrm{GHz}$ and $\omega_{\mathrm{m_2}} / 2\pi = 2.339\mathrm{GHz}$ (Fig. 2a). From these avoided crossings, we determine the qubit-mechanics coupling strengths to be $g_{1} / 2\pi = (9.5\pm 0.1)\mathrm{MHz}$ and $g_{2} / 2\pi = (10.5\pm 0.1)\mathrm{MHz}$ .

Although the static capacitive coupling between the qubit and mechanics is fixed, the qubit-mechanics interaction is controlled on nanosecond timescales by rapidly tuning the frequency of the qubit between the off-resonant $\left(\left|\omega_{\mathrm{ge}}-\omega_{\mathrm{m}_{i}}\right|\gg g_{i}\right)$ and on-resonant $\left(\omega_{\mathrm{ge}}=\omega_{\mathrm{m}_{i}}\right)$ regimes via current pulses sent through the flux-line. To characterize and calibrate swap operations, we bias the qubit frequency to $\omega_{ge}/2\pi=2.26GHz$ , far from the mechanical resonances. Using the pulse sequence of Fig. 2b, we perform Rabi-swap experiments using a single initial excitation in Fig. 2d. At the correct detunings, the excitation is exchanged between the qubit and one mechanical resonator, enabling transfer of the qubit state to the mechanics. We perform an iSWAP operation in a time of $\pi/2g_{i}\simeq24-26ns$ , and estimate a fidelity of $0.95\pm0.01$ from the fringe visibility.

Access to a fast, high-fidelity swap operation allows us to extend our control of the qubit to the mechanical devices. We perform single-phonon characterization of both resonators using the pulse sequence in Fig. 2c. In these experiments, we use the qubit to prepare a quantum state of the resonator, then wait a delay time t before swapping the mechanical state back into the qubit for measurement. By choosing to initially rotate the qubit into the state $|e\rangle$ or $|g\rangle + |e\rangle$ , we characterize either the mechanical energy decay time $T_{1,m}$ or mechanical dephasing time $T_{2,m}$ .

We observe that both resonators exhibit energy relaxation dynamics that are best described as the sum of three decaying exponentials (Fig. 2e). The fastest decay is observed to be $T_{1,m_{1}} = (1.23 \pm 0.08) \mu s$ and $T_{1,m_{2}} = (0.99 \pm 0.03) \mu s$ for mechanical resonators $M_{1}$ and $M_{2}$ . In contrast, the other decay times are on the order of 10 and 90 $\mu s$ for both resonators. The observed multi-exponential response may be explained by resonant decay into saturable and rapidly dephasing two-level systems (TLS) in the device $^{29,30}$ , but a more detailed study is required.

The results of a similar Ramsey experiment are shown in Fig. 2f and used to extract the mechanical dephasing times $T_{2,\mathrm{m}_1} = (0.87 \pm 0.02) \mu \mathrm{s}$ and $T_{2,\mathrm{m}_2} = (1.71 \pm 0.03) \mu \mathrm{s}$ . For a harmonic oscillator under the presence of amplitude damping, we expect each mechanical resonator's $T_{2,\mathrm{m}}$ to be twice its $T_{1,\mathrm{m}}$ ; however, both

![](images/d9b0a5ddbbe723d90c08e42ad66b28851803104196d2d06dd9dae30fa5cb7bb2.jpg)

<details>
<summary>line</summary>

| Detuning (MHz) | Frequency (GHz) | Amplitude (mV) |
| -------------- | --------------- | -------------- |
| -20            | 2.32            | 0              |
| 0              | 2.34            | 0              |
| 20             | 2.36            | 30             |
</details>

![](images/07d11fc67e3a19b22f93ee01ddb86dcd6557f0bbd0532bceeb8e3d4af1af4be8.jpg)

![](images/a886dd74dbf4fd7c30a07b2b91324cbdc580b7885b6818af7d8b64084d615874.jpg)

![](images/0c94f340d7001f716b40a6563e1aa60e32437d832174ebe2f8cb378823f227c8.jpg)

<details>
<summary>heatmap</summary>

| Qubit bias frequency, (ωge+Δ)/2π (GHz) | Interaction time, τ (ns) | Amplitude (mV) |
| ------------------------------------- | ------------------------ | -------------- |
| 2.05                                  | ~200                     | ~35            |
| 2.15                                  | ~100                     | ~35            |
| 2.25                                  | ~50                      | ~35            |
| 2.35                                  | ~200                     | ~35            |
</details>

![](images/cdf7866328ac266a597c906f2eeb74d83b02975b851c79d46c86a672f8942048.jpg)

<details>
<summary>scatter</summary>

| Swap pulse delay, t (μs) | Amplitude (mV) for M₁ | Amplitude (mV) for M₂ |
| ------------------------ | --------------------- | --------------------- |
| 0                        | ~10¹                  | ~10¹                  |
| 50                       | ~10⁰.⁵                | ~10⁰.⁵                |
| 100                      | ~10⁰                  | ~10⁰                  |
</details>

![](images/746b69c5f03281c1ed8074a3fa2c1d732ac2ecb172304cbbd2e69f06953b1f47.jpg)

<details>
<summary>line</summary>

| Swap pulse delay, t (μs) | Amplitude (mV) - X_π/2 | Amplitude (mV) - Y_π/2 |
| ------------------------ | ---------------------- | ---------------------- |
| 0                        | ~20                    | ~20                    |
| 0.5                      | ~20                    | ~20                    |
| 1.0                      | ~20                    | ~20                    |
| 1.5                      | ~20                    | ~20                    |
| 2.0                      | ~20                    | ~20                    |
</details>

FIG. 2. Characterization of the mechanical modes. a, Qubit spectroscopy near the mechanical mode $M_2$ , with $\omega_{\mathrm{ge}}$ detuned relative to $\omega_{\mathrm{m}_2}$ . b, Pulse sequence for Rabi-swap experiment. The qubit is excited to $|e\rangle$ using a $\hat{X}_{\pi}$ pulse, then flux-detuned by frequency $\Delta$ for an interaction time $\tau$ before qubit measurement. c, Pulse sequence for single-phonon $T_{1,\mathrm{m}}$ and $T_{2,\mathrm{m}}$ experiments. The qubit is prepared using either a $\hat{X}_{\pi}$ or $\hat{X}_{\pi /2}$ rotation, then swapped to one of mechanical modes $M_1$ or $M_2$ . After waiting a variable delay time $t$ , the qubit and mechanics are swapped again, followed by an optional qubit tomography rotation $R$ and measurement. d, Qubit response as a function of bias frequency $\omega_{\mathrm{ge}} + \Delta$ and interaction time $\tau$ of the applied flux pulse in b. At the start of the experiment, the qubit is held at $\omega_{\mathrm{ge}} / 2\pi = 2.26\mathrm{GHz}$ (rightmost blue line, where $\Delta = 0$ ) before being frequency-detuned to interact with the mechanical modes (red lines). e, Single phonon $T_{1,\mathrm{m}}$ measurement for each mechanical mode (blue: $M_1$ , red: $M_2$ ), using the pulse sequence in c with the qubit prepared in $|e\rangle$ and the identity operation for R. f, Single phonon $T_{2,\mathrm{m}}$ measurements of the mechanical modes (top: $M_1$ , bottom: $M_2$ ). Here, the qubit is initially prepared in the superposition $|g\rangle + |e\rangle$ , and we use tomography rotations $R = \hat{X}_{\pi /2}$ (blue) or $\hat{Y}_{\pi /2}$ (red) in c.

modes seem to suffer from an additional, non-negligible source of phase decoherence, with inferred pure dephasing times $T_{\phi,m_{1}} = 1.4 \mu s$ and $T_{\phi,m_{2}} = 13 \mu s$ . This may be also due to the presence of TLS, and a more complete analysis of decoherence in these devices will be the subject of future studies.

After characterizing the device, we use the qubit to perform full quantum state tomography of the upper mechanical resonator. Our goal is to obtain the density matrix $\hat{\rho}$ describing a single resonator's state. Previously, this has been achieved through dynamics where the qubit and mechanics directly exchange excitations $^{4,5}$ . Here, we use the strong dispersive interaction to impart a phonon-number dependent frequency shift on the qubit, which is then read out by a Ramsey measurement $^{20,21,31-34}$ that yields the phonon number distribution $P_0(n)$ . This provides us with the diagonal elements of the density matrix $\langle n|\hat{\rho}|n\rangle$ , but does not fully determine the state. To gain information about $\hat{\rho}$ 's off-diagonal elements, we perform a calibrated displacement operation $\hat{D}_{\alpha}$ on the mechanical resonator before the Ramsey measurement to find $P_{\alpha}(n) \equiv \langle n|\hat{D}_{\alpha}\hat{\rho}\hat{D}_{\alpha}^{\dagger}|n\rangle$ .

We begin the tomography protocol by using the qubit to prepare phonon states $|0\rangle$ , $|1\rangle$ , or $|0\rangle + |1\rangle$ in the upper mechanical mode. For this experiment, the qubit is initially biased to $\omega_{ge}/2\pi = 2.26$ GHz to ensure sufficient detuning for a dispersive interaction, $|\Delta_{2}|/g_{2} \simeq 8$ . We synthesize these states by first rotating the qubit to the desired state with an $\hat{X}_{\pi}$ or $\hat{X}_{\pi/2}$ pulse, then swapping it into the resonator, as shown in Fig. 3a. Next, we displace the resonator state ( $\hat{D}_{\alpha}$ ) with a microwave pulse at the mechanical frequency, applied to the qubit's XY line $^{6}$ . We then perform a Ramsey measurement to resolve the dispersive shifts on the qubit resulting from each populated Fock level in the displaced mechanical state. The resulting signal takes the form of a sum of oscillating terms with an exponentially decaying envelope,

$$
S (t) = \sum_ {n = 0} A _ {n} e ^ {- \kappa t / 2} \cos [ (\omega_ {0} + 2 \chi n) t + \varphi_ {n} ]. \tag {1}
$$

We fit the data to Eq. 1 to learn the weight $A_{n}$ and frequency of each spectral component (see methods) and measure a dispersive shift $\chi/2\pi = (-718 \pm 7)$ kHz. This fit allows us to extract the population $P_{\alpha}(n)$ in each Fock level n by normalizing the spectral amplitudes: $P_{\alpha}(n) = A_{n}/\Sigma_{n}A_{n}$ . A representative time-domain fit and extracted distribution are shown in Fig. 3c,d.

Finally, we estimate the most likely state $\hat{\rho}$ of the mechanical resonator using convex optimization $^{35}$ . In this procedure, we perform Ramsey measurements (Fig. 3c) to find $P_{\alpha}(n)$ for 36 different complex values of $\alpha$ (Fig. 3b). We then infer the most likely state $\hat{\rho}$ by minimizing the distance between the experimentally obtained $P_{\alpha}(n)$ and $\langle n|\hat{D}_{\alpha}\hat{\rho}\hat{D}_{\alpha}^{\dagger}|n\rangle$ over all measured $\alpha$ . The reconstructed $\hat{\rho}$ are shown in Fig. 3e, and have state fidelities $\mathcal{F}_s = \langle \psi |\hat{\rho} |\psi \rangle = 0.913\pm 0.003$ , $0.600\pm 0.002$ , and $0.811\pm 0.002$ for the target phonon states $|\psi \rangle = |0\rangle$ , $|1\rangle$ , and $|0\rangle +|1\rangle$ , respectively. These reconstructed $\hat{\rho}$ are then used to compute the Wigner functions $W(\alpha)$ in Fig. 3f, where the negative values in $W(\alpha)$ for $|1\rangle$ demonstrate the quantum nature of the phonon state. We note that the phonon parity can also simply be extracted from the measured $P_{\alpha}(n)$ , from which we directly observe negative parity values in $|1\rangle$ (see methods and Fig. S5).

We attribute the imperfect overlaps $\mathcal{F}_s$ between the target and measured states mostly to mechanical decay

![](images/1795eba12278f93efcd35dcbf5bf061f4345993845a053f76f60d68590edb7c2.jpg)

![](images/a50f3ea7457d22f461f1708310db7ba0b19456ac47bc7a1be58a4d80aa9b32a1.jpg)

f   
![](images/a0d1125e5ffbbeb3a03d479a7682ea74db2ef6a307c286d4efb661b6e3115afd.jpg)

<details>
<summary>heatmap</summary>

| Re(α) \ Im(α) | -1    | 0     | 1     |
|---------------|-------|-------|-------|
| -1            | 0.5   | 0.5   | 0.5   |
| 0             | 0.5   | 0.5   | 0.5   |
| 1             | 0.5   | 0.5   | 0.5   |
</details>

FIG. 3. Single-mode tomography. a, Pulse sequence showing state preparation, displacement, and Ramsey measurement. First, we use the qubit (blue) to prepare $|1\rangle$ in the upper mechanical mode, $M_2$ . $M_2$ is then displaced by a microwave pulse $\hat{D}_{\alpha}$ with variable amplitude and phase. Finally, we perform a Ramsey sequence on the qubit. For $|0\rangle$ , state preparation (left) is omitted, and for $|0\rangle + |1\rangle$ the $\hat{X}_{\pi}$ pulse is replaced with $\hat{X}_{\pi / 2}$ . b, Complex-valued amplitudes $\alpha$ of the displacements $\hat{D}_{\alpha}$ (red points), with a few corresponding measurement results (highlighted points). c, Representative Ramsey measurement result and d, extracted phonon number distribution. The data (dark blue points) are fit to Eq. 1 (light blue line) with the grey dashes showing the fitted decay envelope. e, Reconstructed density matrices and f, Wigner functions for each prepared state, extracted by convex optimization.

during the experiment. The Ramsey measurement duration is constrained by the time required to resolve the dispersive shifts, $\pi/\chi_{2} \simeq 700$ ns, which is comparable to the decay $T_{1,m_{2}}$ . Quantum master equation simulations of the tomography protocol agree with the measured fidelities when we include the observed mechanical decoherences $T_{1,m_{2}}$ and $T_{2,m_{2}}$ (see methods).

By developing fast gates for multiple mechanical oscillators and extending our tomography protocol to bipartite states, we realize a small quantum acoustic processor that can generate and characterize entangled states of mechanical systems. As in Fig. 4a, our entangling gate consists of multiple sub-operations to create a mechanical Bell-state, $|\psi_{\mathrm{Bell}}\rangle = |01\rangle + e^{i\phi} |10\rangle$ . After exciting the qubit, a $\sqrt{iSWAP}$ operation is performed between the qubit and upper mechanical mode to maximally entangle the two. The qubit state is then fully swapped to the lower mechanical mode, which translates the entanglement to be between the two mechanical systems.

Next, we perform tomography on the joint mechanical system by extending our Ramsey measurement approach. We position the qubit frequency such that the mechanical dispersive shifts, $\chi_{1}/2\pi = (-517 \pm 6)$ kHz and $\chi_{2}/2\pi = (-799 \pm 5)$ kHz, are distinguishable from each other, with a typical Ramsey measurement signal shown in Fig. 4b. In fitting these two-mode experiments' interference patterns, the model of Eq. 1 is extended to accommodate both resonators by replacing $A_{n} \rightarrow A_{mn}$ and $2\chi n \rightarrow 2\chi_{1}m + 2\chi_{2}n$ for Fock indices m and n of the lower and upper mechanics, respectively. Normalization of the signal amplitudes $A_{mn}$ gives the joint phonon number distribution $P_{\alpha\beta}(m,n)$ (Fig. 4c). In order to reconstruct the joint state $\hat{\rho}$ of the two mechanical systems, we repeat the experiment for 25 different combinations of displacements $\hat{D}_{\alpha} \otimes \hat{D}_{\beta}$ on the resonators, each time extracting the associated $P_{\alpha\beta}(m,n)$ . We then estimate $\hat{\rho}$ from the set of $P_{\alpha\beta}(m,n)$ using convex optimization (see methods), resulting in the reconstructed state shown in Fig. 4d. The overlap of the inferred state with the target Bell-state is $\mathcal{F}_{\mathrm{Bell}} = \langle \psi_{\mathrm{Bell}} | \hat{\rho} | \psi_{\mathrm{Bell}} \rangle = 0.57 \pm 0.02$ , with a quantum state purity $\operatorname{tr}(\hat{\rho}^{2}) = 0.46 \pm 0.02$ . Numerical simulations of the mechanical system, shown in Fig. 4d, are in good agreement with the measured $\hat{\rho}$ , allowing us to attribute the dominant source of loss in fidelity $F_{Bell}$ to mechanical $T_{1,m}$ and $T_{2,m}$ decay during the Ramsey measurement (see methods).

In conclusion, we demonstrate deterministic quantum control over a pair of nanomechanical resonators and characterize their joint quantum state using a dispersive, non-demolition measurement. In future work, mitigation of TLS-induced decoherence in lithium niobate phononic crystal resonators should allow for longer mechanical coherence times $^{23,29}$ , which presently limit the observed state fidelities in our device. This experiment's flip-chip architecture is well-suited for separate optimization of the qubit and mechanical systems by enabling a

modular approach to engineering hybrid quantum systems. Our hardware approach has enabled deterministic manipulation of quantum entanglement between macroscopic mechanical objects, and can be extended to architectures including quantum random access memories and biased-error cat qubits $^{1-3}$ .

![](images/d87f113e1db301f6df0d01e88346214c0dde7b4f913aa0bd3dbdf0c91251d830.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    M1 --> X_π
    X_π --> Q_ge
    Q_ge --> X_π
    X_π --> D_α
    D_α --> X_ξ
    X_ξ <-->|t→| X_ξ
    X_ξ --> D_β
    style M1 fill:#f9f,stroke:#333
    style M2 fill:#bbf,stroke:#333
    style Q_ge fill:#dfd,stroke:#333
    style D_α fill:#ffd,stroke:#333
    style D_β fill:#ffd,stroke:#333
```
</details>

![](images/ee4dbf1dec19069afce4a27702a33429532daebc1f2389fa6ba1979a6b0bef5e.jpg)

<details>
<summary>bar</summary>

| Fock index, n | Population, P(m,n) |
| ------------- | ------------------ |
| 0             | .38                |
| 1             | -                  |
| 2             | -                  |
| 3             | -                  |
</details>

![](images/5d4e2ac673737e96a44e76eb8b54cb3bb75449e8e262bee515e0be749a41f9ae.jpg)

<details>
<summary>line</summary>

| Time, t (ns) | Amplitude (mV) |
| ------------ | -------------- |
| 0            | -10            |
| 200          | 5              |
| 400          | 0              |
| 600          | -5             |
| 800          | 0              |
| 1000         | 5              |
</details>

![](images/3f1b6f030ea3c526dbef6bc54b02b018275d5a2827a3d152546fc192ea993999.jpg)  
FIG. 4. Joint tomography of a mechanical Bell-state. a, Pulse sequence for mechanical Bell-state preparation (left) and two-mode tomography (right). The qubit is first excited to $|e\rangle$ , followed by a $\sqrt{i\text{SWAP}}$ and $i\text{SWAP}$ operation to the mechanical resonators $M_2$ and $M_1$ , respectively. Two-mode tomography is performed as in Fig. 3a, with the modification that a displacement $\hat{D}_{\alpha} \otimes \hat{D}_{\beta}$ is now applied simultaneously to each resonator. b, Representative Ramsey interference pattern (points) and fit (line) for the two-mode measurements. The qubit frequency is positioned between the mechanical resonators to achieve discernibly different dispersive shifts $\chi_1$ and $\chi_2$ from each mode (inset). c, Extracted joint phonon number distribution $P_{\alpha\beta}(m,n)$ from the fit of b, with the $P_{\alpha\beta}(0,0) \simeq 0.38$ element truncated for visual clarity. d, Reconstructed density matrix $\hat{\rho}$ for the mechanical Bell-state. 25 different Ramsey measurements (a-c) are combined to obtain the most likely quantum state (blue bars), in good agreement with numerical simulations (red) that take into account mechanical decay during the Ramsey measurement of an initial ideal Bell-state (grey).

# METHODS

Fabrication. Our device fabrication closely follows previous methods $^{6}$ , with the important difference that the processes for qubits and nanomechanical structures are now performed on separate dies. We have moved to a slightly different material platform for the mechanics chip, in which the thin-film LN has been doped with magnesium oxide (MgO) to improve the mechanical properties of the crystal $^{29}$ . Additionally, we thermally anneal the mechanics chip (8 hours at 500 C) before patterning the device. On the qubit chip, we have added aluminum crossovers across the qubit control lines and readout transmission line. We have also developed an oxygen plasma descum process to remove polymer residues from inter-metallic layers, reducing TLS-induced microwave loss.

The flip-chip bonding procedure is the final step in our fabrication process. We use a submicron die bonder (Finetech) to align the two chips by positioning the two pads of the coupling capacitor on top of each other. The two chips' active surfaces are brought to a separation distance of $1 \mu m$ , as allowed by 500 nm aluminum spacer ridges patterned on each chip. Finally, an adhesive polymer (9:1 ethanol/GE varnish) is manually applied to the outer edges of the top chip to secure it in place. Images of the final integrated flip-chip device are shown in Fig. S1.

Mechanics design. For our experiment, it is important to carefully choose the frequency arrangement such that all modes are sufficiently protected from decoherence channels, while the qubit-mechanics interaction remains in the dispersive regime. Using finite-element simulations, we choose a phononic crystal geometry with a bandgap extending from approximately 1.90 to 2.50 GHz and a pitch of $a = 900 \, nm$ . The mechanical frequencies, controlled by adjusting the width of the defect site, are designed to be approximately 150 MHz away from the bandgap edges, ensuring that the modes are protected from clamping losses.

Qubit design and control. Our device utilizes a transmon-style qubit with microwave control lines and a dispersively coupled microwave resonator for readout. An on-chip flux line positioned near the qubit's SQUID loop provides capability for both static (DC) and rapid (pulsed) frequency tuning of the qubit via externally applied magnetic flux. In Fig. S2a, we measure the qubit's frequency tuning curve, with the maximum qubit frequency at $\omega_{ge}^{max}/2\pi = 2.443$ GHz. Our device has charging energy $E_{C}/h = \alpha_{q}/2\pi = 126$ MHz and Josephson energy $E_{J}/h = 6.550$ GHz, ensuring that it operates well into the transmon regime $^{28}$ , $E_{J}/E_{C} \gg 1$ .

For tomography experiments, the qubit operating frequency is chosen to be in between the two mechanical frequencies. This ensures that both the primary qubit transition $\omega_{ge}$ and the next higher transition $\omega_{ef} = \omega_{ge} - \alpha_{q}$ are sufficiently distant from the mechanical modes that the qubit is effectively decoupled, allowing us to perform rotations of the qubit state with high fidelity. Placing the qubit frequency in this region also gives strong dispersive coupling to both mechanical modes in order to perform joint tomography of the mechanical systems.

We perform gates on the qubit state by applying mi-

crowave pulses with variable amplitude, phase, and duration to the qubit's $XY$ line. For these experiments, we use 20 ns DRAG pulses with approximately Gaussian envelopes $^{36,37}$ . Utilizing randomized benchmarking techniques $^{38,39}$ , we observe a single qubit gate fidelity of 0.996 at the operating frequency used for tomography ( $\omega_{\text{ge}} / 2\pi = 2.26 \text{ GHz}$ ) as shown in Fig. S2c. To measure the qubit state, we use a standard cQED approach of dispersive readout through an off-resonantly coupled coplanar waveguide resonator. To infer the qubit excited state probability, we apply a microwave pulse to the readout resonator's transmission line and measure the scattered response, which allows us to detect shifts in its resonant frequency induced by the qubit's state.

In Fig. S2b, we measure the qubit energy decay time $T_{1}$ over a large frequency range and plot the results. For each horizontal slice, we statically bias the qubit to the indicated frequency and perform a standard ringdown measurement to study its $T_{1}$ energy decay. The qubit excited state probability, indicated by the color bar, is plotted as a function of time. The white points show the resulting $T_{1}$ values, extracted by fitting each data slice to an exponential decay function. From this data set, we find the average qubit decay time to be $T_{1,\mathrm{avg}} = (4.9 \pm 2.3) \mu\mathrm{s}$ .

We also characterize the thermal population of the qubit with a thermometry experiment, shown in Fig. S2d. For this measurement, we use a Rabi population method $^{4,40}$ to quantify the residual qubit population in $|e\rangle$ . This is done by driving rotations of the qubit state between the $|e\rangle$ and $|f\rangle$ levels with varying rotation angle. A final $X_{\pi}$ pulse exchanges the $|g\rangle$ and $|e\rangle$ populations before we measure the qubit state. We perform this measurement both with and without an optional $X_{\pi}$ pulse at the beginning of the sequence, which exchanges the steady-state $|g\rangle$ and $|e\rangle$ populations in the qubit. These measurements produce two Rabi-like oscillation patterns whose amplitudes $A_{g}$ and $A_{e}$ contain information about the thermal $|e\rangle$ population, $P_{e,\mathrm{th}} = A_e / (A_g + A_e)$ . By this method, we find $P_{e,\mathrm{th}} = 0.057$ with the qubit biased to $\omega_{\mathrm{ge}} / 2\pi = 1.798\mathrm{GHz}$ . We perform this measurement far detuned from both mechanical modes to ensure high fidelity rotations of the qubit states. At this operating point, the reported value is likely an upper bound on the qubit thermal population relevant for our primary experiments.

Master equation simulations. To model the quantum dynamics of our device and obtain estimates of the mechanical state fidelities, we perform time-domain master equation simulations using the QuTiP package $^{41}$ . First, we simulate the qubit-mechanics dynamics during Rabi-swap experiments to illustrate how nanosecond-timescale flux pulses can be used to manipulate the device. Here, the qubit is modeled as a 3-level nonlinear resonator $\hat{a}$ with time-dependent frequency $\omega_{\mathrm{ge}}(t)$ and anharmonicity $\alpha_{q}$ , subject to $T_{1}$ decay and pure dephasing $T_{\phi}$ . The mechanical resonators are taken to be 3-level harmonic oscillators $\hat{b}_{i}$ , also with $T_{1,m}$ and $T_{\phi,m}$ decoherence channels. For this experiment, the total Hamiltonian $\hat{H} = \hat{H}_{0} + \hat{H}_{int} + \hat{H}_{d}$ has contributions

$$
\hat {H} _ {0} = \omega_ {\mathrm{ge}} (t) \hat {a} ^ {\dagger} \hat {a} - \frac {\alpha_ {q}}{2} \hat {a} ^ {\dagger} \hat {a} ^ {\dagger} \hat {a} \hat {a} + \sum_ {i} \omega_ {\mathrm{m} _ {i}} \hat {b} _ {i} ^ {\dagger} \hat {b} _ {i},
$$

$$
\hat {H} _ {\mathrm{int}} = \sum_ {i} g _ {i} (\hat {a} + \hat {a} ^ {\dagger}) (\hat {b} _ {i} + \hat {b} _ {i} ^ {\dagger}),
$$

$$
\hat {H} _ {\mathrm{d}} = \frac {1}{2} \left(\Omega (t) \hat {a} + \Omega^ {*} (t) \hat {a} ^ {\dagger}\right).
$$

A drive term $\hat{H}_{d}$ allows for population of either the qubit or the mechanical resonators, depending on the chosen modulation frequency of the applied drive, $\Omega(t)$ . We also allow applied flux pulses to add time-dependent frequency control of the qubit, $\omega_{\mathrm{ge}}(t)=\omega_{\mathrm{ge}}+\tilde{\omega}_{\mathrm{ge}}(t)$ , where $\omega_{ge}$ is the static qubit frequency and $\tilde{\omega}_{\mathrm{ge}}(t)$ represents the transient frequency control.

In simulating the time evolution of the total quantum system $\hat{\rho}$ , we numerically integrate the Lindblad master equation,

$$
\frac {d \hat {\rho}}{d t} = - i [ \hat {H}, \hat {\rho} ] + \sum_ {k} \left(\hat {c} _ {k} \hat {\rho} \hat {c} _ {k} ^ {\dagger} - \frac {1}{2} \{\hat {c} _ {k} ^ {\dagger} \hat {c} _ {k}, \hat {\rho} \}\right),
$$

with collapse operators $\hat{c}_{k} = \hat{b}_{1}/\sqrt{T_{1,m_{1}}}$ , $\hat{b}_{2}/\sqrt{T_{1,m_{2}}}$ , $\hat{b}_{1}^{\dagger}\hat{b}_{1}/\sqrt{2T_{\phi,m_{1}}}$ , $\hat{b}_{2}^{\dagger}\hat{b}_{2}/\sqrt{2T_{\phi,m_{2}}}$ , $\hat{a}/\sqrt{T_{1}}$ , and $\hat{a}^{\dagger}\hat{a}/\sqrt{2T_{\phi}}$ . Note that all model parameters are experimentally determined from standard qubit measurements, with a subtlety in our choice of the bare frequencies $\omega_{ge}$ and $\omega_{m_{i}}$ . In order to obtain consistent results between experiment and theory, $\hat{H}$ is first defined in the bare basis, then diagonalized to find the dressed basis eigenstates and eigenvalues. The bare frequencies of $\hat{H}$ are then chosen such that the dressed frequencies of the qubit and mechanics best match the experimentally measured values at steady state. These dressed states can then be used to evaluate final state probabilities and expectation values.

Using this framework, we aim to reproduce the asymmetry present in the qubit-mechanics chevrons of Fig. 2d in the main text. These simulations show excellent agreement with experimental Rabi-swap results, as shown in Fig. S3a. The limited visibility of the fringes near the static qubit bias $\omega_{\mathrm{ge}}$ is due to the details of the dressed state in a system where the coupling $g_{i}$ is always present. In the case of small detuning $\tilde{\omega}_{\mathrm{ge}}(t)$ , the instantaneous dressed bases do not change appreciably, and so the qubit-like mode roughly remains in the same dressed eigenstate throughout the operation. This is in contrast to systems where the $g_{i}$ can be turned off during single qubit operations, thereby avoiding dressing of the qubit state except when the coupling is desired.

We perform additional master equation simulations to compare with the observed mechanical state fidelities $F_{s}$ and $F_{Bell}$ reported in the main text. For these simulations, we assume the initial mechanical state $\hat{\rho}(0)$ is the ideal target state, then let $\hat{\rho}(t)$ freely evolve during the Ramsey measurement, subject only to mechanical collapse operators $\hat{c}_{k} = \hat{b}_{i} / \sqrt{T_{1,m_{i}}}$ and $\hat{b}_{i}^{\dagger}\hat{b}_{i} / \sqrt{2T_{\phi,m_{i}}}$ . Note

that we now ignore any dynamics of the qubit, as well as the details of the state preparation. For the single-mode states $|1\rangle$ and $|0\rangle + |1\rangle$ , we obtain fidelities 0.566 and 0.809 from simulation, similar to the measured $F_{s}$ of $0.600 \pm 0.002$ and $0.811 \pm 0.002$ . For the mechanical Bell-state, master equation simulations give a fidelity of 0.587, compared to the observed $F_{Bell} = 0.57 \pm 0.02$ . The good agreement between simulation and our measured fidelities suggest that $F_{s}$ and $F_{Bell}$ are likely limited by $T_{1,m}$ and $T_{2,m}$ decoherence mechanisms in the mechanical system.

Swap characterization. To demonstrate control of the qubit-mechanics swap operation, we perform quantum state tomography on the qubit during resonant Rabi-swap experiments. Using the pulse sequence of Fig. 2b, we bring the qubit into resonance with the upper mechanical mode $M_2$ for an interaction time $\tau$ before applying tomography gates $X_\theta$ or $Y_\theta$ , chosen from $\theta = \{0, \pm\pi/2, \pm\pi\}$ . Combining the results of this set of measurements allows for the reconstruction of the qubit Bloch vector $\langle \vec{\sigma} \rangle$ , shown in Fig. S3b,c. Note that since our experiment does not have single-shot qubit state readout, we calibrate the observed qubit response using the measured qubit thermal population $P_{e,\text{th}}$ in order to estimate the Bloch vector. In Fig. S3b, the qubit is prepared in $|e\rangle$ before resonantly interacting with $M_2$ ; the resulting data show the excitation periodically returning to the qubit, with the $\sigma_X$ and $\sigma_Y$ components largely unaffected. A similar experiment is performed in Fig. S3c, with the qubit now starting in the superposition $|g\rangle + |e\rangle$ . As expected, the qubit's superposition is recovered from the mechanical resonator at even multiples of the swap time.

Time domain data analysis. The Ramsey measurements used for state tomography contain information about the phonon number distribution of the dispersively coupled mechanical state. The distinct spectral components in the number-split qubit spectrum create an interference pattern which depends strongly on the mechanical occupation, as shown in Fig. 3b. We fit the data to a function of the form (Eq. 1 in main text)

$$
S (t) = \sum_ {n = 0} A _ {n} e ^ {- \kappa t / 2} \cos [ (\omega_ {0} + 2 \chi n) t + \varphi_ {n} ],
$$

where $\chi$ , $\kappa$ , and $A_{n}$ are model fit parameters. In $S(t)$ , the component corresponding to the nth Fock level's occupation is given an amplitude $A_{n}$ and frequency $\omega_{0} + 2\chi n$ . Here, the dispersive shift $\chi$ is constrained to be the same for all n, and the frequency $\omega_{0}/2\pi = 25 MHz$ is the programmed frame detuning of the Ramsey sequence's second $\hat{X}_{\pi/2}$ pulse. The phases $\varphi_{n} = 2\chi n(2\tilde{\tau})$ account for qubit phase accumulation during the two $\hat{X}_{\pi/2}$ pulses of the Ramsey sequence, each with a pulse duration $\tilde{\tau} = 20 ns$ . Since we define $S(t)$ in terms of the elapsed time t between the Ramsey pulses, there is a small phonon-state dependent phase accumulation at t = 0 due to the finite operation time of our single qubit gates. The exponential decay term $e^{-\kappa t/2}$ represents an effective dephasing time which is dominated by the qubit $T_{2}$ , but also includes a contribution from the mechanical state. This fit allows us to extract the Fock populations $P_{\alpha}(n)$ by normalizing the spectral amplitudes: $P_{\alpha}(n) = A_{n}/\Sigma_{n}A_{n}$ .

To extend the model to the two-mode case, we replace $A_{n} \rightarrow A_{mn}$ and $2\chi n \rightarrow 2\chi_{1}m + 2\chi_{2}n$ for Fock indices m and n of the resonators. We also need to adjust the zero-delay qubit phase $\varphi_{n} \rightarrow \varphi_{mn} = (2\chi_{1}m + 2\chi_{2}n)(2\tilde{\tau})$ to account for both resonators shifting the qubit frame during the Ramsey $\hat{X}_{\pi/2}$ pulses. This yields an adjusted time domain model

$$
S (t) = \sum_ {m, n} A _ {m n} e ^ {- \kappa t / 2} \cos [ (\omega_ {0} + 2 \chi_ {1} m + 2 \chi_ {2} n) t + \varphi_ {m n} ].
$$

We use this function to fit the Ramsey measurements of Fig.4b in the main text, and thereby determine the two-mode phonon number distribution $P_{\alpha \beta}(m,n) = A_{mn} / \Sigma_{m,n}A_{mn}$ .

State reconstruction. For both the single- and two-mode tomography demonstrated in this experiment, we can use similar protocols for state reconstruction. We perform tomography of an unknown two-mode mechanical state $\hat{\rho}$ by applying displacements $\hat{D}_{\alpha\beta} \equiv \hat{D}_{\alpha} \otimes \hat{D}_{\beta}$ on $\hat{\rho}$ , then measuring the diagonal elements $P_{\alpha\beta}(m,n)$ of the resulting joint state $\hat{\rho}(\alpha,\beta) = \hat{D}_{\alpha\beta} \hat{\rho} \hat{D}_{\alpha\beta}^{\dagger}^{35}$ . More specifically, the set of measurement data can be represented as $P_{\alpha\beta}(m,n) = \langle m,n | \hat{D}_{\alpha\beta} \hat{\rho} \hat{D}_{\alpha\beta}^{\dagger} | m,n \rangle$ , where $|m,n\rangle = |m\rangle \otimes |n\rangle$ is the joint Fock basis of the resonators. In our experiment, we choose from combinations of complex displacements with amplitudes ( $|\alpha|, |\beta|$ ) $\leq (0.6, 0.7)$ , and fit the resulting $P_{\alpha\beta}(m,n)$ up to maximum Fock indices ( $m_{max}, n_{max}$ ) = (3,3). The unknown density matrix $\hat{\rho}$ can then be reconstructed by minimizing the loss function

$$
\mathcal {L} (\hat {\rho}) = \sum_ {\alpha , \beta} \sum_ {m, n} \left| \langle m, n | \hat {D} _ {\alpha \beta} \hat {\rho} \hat {D} _ {\alpha \beta} ^ {\dagger} | m, n \rangle - P _ {\alpha \beta} (m, n) \right| ^ {2},
$$

a convex problem that can be solved efficiently using the CVX package $^{42}$ . The single-mode mechanical state tomography follows a similar method by setting either $\alpha$ or $\beta$ to zero and ignoring the corresponding mode in the analysis. For the single-mode tomography experiments, we use $|\alpha| \leq 1.25$ and $n_{max} = 8$ .

Direct parity calculation. In our single-mode tomography protocol, we choose to reconstruct the Wigner functions by fitting over candidate $\hat{\rho}$ for experimental efficiency. However, the mechanical resonator's Wigner function $W(\alpha)$ can also be directly computed from the parity $\Pi (\alpha) = \Sigma_n(-1)^n P_\alpha (n) = \frac{\pi}{2} W(\alpha)$ of each Ramsey measurement. In a separate data set (Fig. S5), we measure $\Pi (0) = -0.36$ for the $|\psi \rangle = |1\rangle$ target state, which confirms the quantum nature of this prepared state by direct measurement.

Displacement calibration. Mechanical state reconstruction relies on knowing the amplitudes $|\alpha|$ of the displacements $\hat{D}_{\alpha}$ that we apply during each tomography

pulse sequence. This requires a calibration relating the applied microwave pulse's voltage amplitude to the resulting mechanical displacement amplitude $|\alpha|$ , shown in Fig. S4a. Here, we displace the upper mechanical mode by applying a microwave pulse to the qubit $XY$ line at the mechanical frequency, then perform Ramsey interferometry on the resulting state to extract the phonon number distribution $P_{\alpha}(n)$ . Next, we perform a least squares fit to find the coherent state $|\alpha\rangle$ whose coefficients $|\langle n|\alpha\rangle|^2$ most closely match the measured $P_{\alpha}(n)$ to obtain the inferred displacement amplitude $|\alpha_{\mathrm{inf}}|$ . We find that the relation between the applied voltage amplitude $V$ and the inferred displacement amplitude $|\alpha_{\mathrm{inf}}|$ follows a “hockey-stick” curve $|\alpha_{\mathrm{inf}}| = ((c_1V)^2 + c_2)^{1/2}$ , where the second fit parameter $c_2$ accounts for the thermal population of the mechanical mode. During state reconstruction, we only attribute the displacement amplitude to the voltage we apply, namely $|\alpha| = c_1V$ . For our system, we find $c_1 = 1.664 \pm 0.005$ and $c_2 = 0.091 \pm 0.001$ .

We also perform a simulation of this displacement calibration procedure, and the results (Fig. S4b) show good agreement with the experimental data. In these simulations, a small thermal state $\hat{\rho}_{\mathrm{th}}$ with population $P_{\mathrm{th}} = 0.10$ is coherently displaced $\hat{D}_{\alpha}\hat{\rho}_{\mathrm{th}}\hat{D}_{\alpha}^{\dagger}$ with a programmed amplitude $|\alpha|$ . The phonon number distribution of the resulting state is fit to the nearest coherent state, from which we infer the effective displacement amplitude $|\alpha_{\mathrm{inf}}|$ . This operation yields a similar hockey-stick behavior as the experimental data, with a y-intercept $|\alpha_{\mathrm{inf}}| = 0.305$ . This corresponds to an average phonon number $n_{\mathrm{avg}} = 0.093 \simeq P_{\mathrm{th}}$ . Fitting this simulated data to the hockey-stick model yields a scale factor between the inferred $|\alpha|$ and the input $|\alpha|$ of $c_{1} = 0.984 \pm 0.002 \simeq 1$ , as we expect.

Error analysis of state reconstruction. We use Monte Carlo error propagation to determine the robustness of the mechanical state reconstruction previously described. In fitting the $P_{\alpha}(n)$ or $P_{\alpha \beta}(m,n)$ from $S(t)$ , we obtain an estimate for the model parameters' covariance matrix during the nonlinear least squares regression. For error propagation testing, we then randomly resample the $P_{\alpha}(n)$ or $P_{\alpha \beta}(m,n)$ and displacement calibrations, using their respective statistical uncertainties computed from the model's covariance matrix. The resampled parameters are then fed into the convex optimization routine that minimizes $\mathcal{L}(\hat{\rho})$ to obtain the reconstructed state $\hat{\rho}$ . This resampling process is repeated $3\times 10^{3}$ times to obtain the resulting variations in the reconstructed density matrix fidelities shown in Fig. S4c,d. We use the standard deviation of these reconstructed fidelities to obtain the error estimates for the state fidelities $F_{s}$ and $F_{Bell}$ reported in the main text.

# ACKNOWLEDGMENTS

The authors would like to thank M. Kang, T.P. McKenna, W. Jiang, Y.P.Zhong, K.K.S. Multani, N.R. Lee, M.M. Fejer, and P.J. Stas for discussions. We acknowledge the support of the David and Lucille Packard, and Sloan Fellowships. This work was funded by the U.S. government through the Office of Naval Research (ONR) under grant No. N00014-20-1-2422, the U.S. Department of Energy through Grant No. DE-SC0019174, and the National Science Foundation CAREER award No. ECCS-1941826. E.A.W. was supported by the Department of Defense through the National Defense & Engineering Graduate Fellowship. A.Y.C. was supported by the Army Research Office through the Quantum Computing Graduate Research Fellowship as well as the Stanford Graduate Fellowship. Device fabrication was performed at the Stanford Nano Shared Facilities (SNSF), supported by the National Science Foundation under award ECCS-2026822, and the Stanford Nanofabrication Facility (SNF). The authors wish to thank NTT Research for their financial and technical support.

# AUTHOR CONTRIBUTIONS

E.A.W. and A.Y.C. designed and fabricated the device. E.A.W., A.Y.C., R.G.G., and P.A.A. developed the fabrication process. Z.W., R.G.G., and A.H.S.-N. provided experimental and theoretical support. E.A.W. and A.Y.C. performed the experiments and analyzed the data. E.A.W., A.Y.C. and A.H.S.-N. wrote the manuscript, with all others assisting. A.H.S.-N. supervised all efforts.

# ADDITIONAL INFORMATION

P. Arrangoiz-Arriola is currently a research scientist at Amazon, and A. H. Safavi-Naeini is an Amazon Scholar. The other authors declare no competing financial interests. Correspondence and requests for materials should be addressed to A. H. Safavi-Naeini (safavi@stanford.edu)

hdanowicz, S. T. Flammia, A. Keller, G. Refael, J. Preskill, L. Jiang, A. H. Safavi-Naeini, O. Painter, and F. G. Brandão, arXiv:2012.04108 (2020).   
$^{4}$ K. J. Satzinger, Y. P. Zhong, H.-S. Chang, G. A. Peairs, A. Bienfait, M.-H. Chou, A. Y. Cleland, C. R. Conner, É. Dumur, J. Grebel, I. Gutierrez, B. H. November, R. G. Povey, S. J. Whiteley, D. D. Awschalom, D. I. Schuster, and A. N. Cleland, Nature 563, 661 (2018).   
$^{5}$ Y. Chu, P. Kharel, T. Yoon, L. Frunzio, P. T. Rakich, and R. J. Schoelkopf, Nature 563, 666 (2018).   
$^{6}$ P. Arrangoiz-Arriola, E. A. Wollack, Z. Wang, M. Pechal, W. Jiang, T. P. McKenna, J. D. Witmer, R. Van Laer, and A. H. Safavi-Naeini, Nature 571, 537 (2019).   
$^{7}$ L. R. Sletten, B. A. Moores, J. J. Viennot, and K. W. Lehnert, Phys. Rev. X 9, 021056 (2019).   
$^{8}$ A. Bienfait, K. J. Satzinger, Y. P. Zhong, H.-S. Chang, M.-H. Chou, C. R. Conner, É. Dumur, J. Grebel, G. A. Peairs, R. G. Povey, and A. N. Cleland, Science 364, 368 (2019).   
$^{9}$ A. Bienfait, Y. Zhong, H.-S. Chang, M.-H. Chou, C. Conner, É. Dumur, J. Grebel, G. Peairs, R. Povey, K. Satzinger, and A. Cleland, Phys. Rev. X 10, 021055 (2020).   
$^{10}$ A. D. O'Connell, M. Hofheinz, M. Ansmann, R. C. Bialczak, M. Lenander, E. Lucero, M. Neeley, D. Sank, H. Wang, M. Weides, J. Wenner, J. M. Martinis, and A. N. Cleland, Nature 464, 697 (2010).   
$^{11}$ P. Arrangoiz-Arriola and A. H. Safavi-Naeini, Phys. Rev. A 94, 063864 (2016).   
$^{12}$ Y. Chu, P. Kharel, W. H. Renninger, L. D. Burkhart, L. Frunzio, P. T. Rakich, and R. J. Schoelkopf, Science 358, 199 (2017).   
$^{13}$ Y. Chu and S. Gröblacher, Applied Physics Letters 117, 150503 (2020).   
$^{14}$ J. D. Jost, J. P. Home, J. M. Amini, D. Hanneke, R. Ozeri, C. Langer, J. J. Bollinger, D. Leibfried, and D. J. Wineland, Nature 459, 683–685 (2009).   
$^{15}$ C. F. Ockelen-Korppi, E. Damskägg, J.-M. Pirkkalainen, M. Asjad, A. A. Clerk, F. Massel, M. J. Woolley, and M. A. Sillanpää, Nature 556, 478–482 (2018).   
$^{16}$ R. Riedinger, A. Wallucks, I. Marinković, C. Löschnauer, M. Aspelmeyer, S. Hong, and S. Gröblacher, Nature 556, 473–477 (2018).   
$^{17}$ S. Barzanjeh, E. S. Redchenko, M. Peruzzo, M. Wulf, D. P. Lewis, G. Arnold, and J. M. Fink, Nature 570, 480–483 (2019).   
$^{18}$ L. M. de Lépinay, C. F. Ockelen-Korppi, M. J. Woolley, and M. A. Sillanpää, Science 372, 625 (2021).   
$^{19}$ S. Kotler, G. A. Peterson, E. Shojaee, F. Lecocq, K. Cicak, A. Kwiatkowski, S. Geller, S. Glancy, E. Knill, R. W. Simmonds, J. Aumentado, and J. D. Teufel, Science 372, 622 (2021).   
$^{20}$ P. Bertet, A. Auffeves, P. Maioli, S. Osnaghi, T. Meunier, M. Brune, J. M. Raimond, and S. Haroche, Physical Review Letters 89, 200402 (2002).   
$^{21}$ D. I. Schuster, A. A. Houck, J. A. Schreier, A. Wallraff, J. M. Gambetta, A. Blais, L. Frunzio, J. Majer, B. Johnson, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, Nature 445, 515 (2007).   
$^{22}$ P. Arrangoiz-Arriola, E. A. Wollack, M. Pechal, J. D. Witmer, J. T. Hill, and A. H. Safavi-Naeini, Physical Review X 8, 031007 (2018).   
$^{23}$ G. S. MacCabe, H. Ren, J. Luo, J. D. Cohen, H. Zhou, A. Sipahigil, M. Mirhosseini, and O. Painter, Science 370,

840 (2020).   
$^{24}$ K. J. Satzinger, C. R. Conner, A. Bienfait, H.-S. Chang, M.-H. Chou, A. Y. Cleland, É. Dumur, J. Grebel, G. A. Peairs, R. G. Povey, S. J. Whiteley, Y. P. Zhong, D. D. Awschalom, D. I. Schuster, and A. N. Cleland, Applied Physics Letters 114, 173501 (2019).   
$^{25}$ J. Kelly, Fault-tolerant superconducting qubits, Thesis, University of California, Santa Barbara (2015).   
$^{26}$ C. Wang, M. J. Burek, Z. Lin, H. A. Atikian, V. Venkataraman, I.-C. Huang, P. Stark, and M. Lončar, Optics Express 22, 30924 (2014).   
$^{27}$ G. Vidal-Álvarez, A. Kochhar, and G. Piazza, 2017 IEEE International Ultrasonics Symposium (IUS), 1 (2017).   
$^{28}$ J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer, A. Blais, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, Physical Review A 76, 042319 (2007).   
$^{29}$ E. A. Wollack, A. Y. Cleland, P. Arrangoiz-Arriola, T. P. McKenna, R. G. Gruenke, R. N. Patel, W. Jiang, C. J. Sarabalis, and A. H. Safavi-Naeini, Applied Physics Letters 118, 123501 (2021).   
$^{30}$ P. Heidler, C. M. F. Schneider, K. Kustura, C. Gonzalez-Ballestero, O. Romero-Isart, and G. Kirchmair, Phys. Rev. Applied 16, 034024 (2021).   
$^{31}$ D. Lachance-Quirion, S. Wolski, Y. Tabuchi, S. Kono, K. Usami, and Y. Nakamura, Science 367, 425 (2020).   
$^{32}$ J. Gambetta, A. Blais, D. I. Schuster, A. Wallraff, L. Frunzio, J. Majer, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, Physical Review A 74, 042318 (2006).   
$^{33}$ M. Brune, S. Haroche, V. Lefevre, J. M. Raimond, and N. Zagury, Physical Review Letters 65, 976 (1990).   
$^{34}$ M. Brune, P. Nussenzveig, F. Schmidt-Kaler, F. Bernardot, A. Maali, J. M. Raimond, and S. Haroche, Physical Review Letters 72, 3339 (1994).   
$^{35}$ Z. Wang, M. Pechal, E. A. Wollack, P. Arrangoiz-Arriola, M. Gao, N. R. Lee, and A. H. Safavi-Naeini, Phys. Rev. X 9, 021049 (2019).   
$^{36}$ F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm, Phys. Rev. Lett. 103, 110501 (2009).   
$^{37}$ Z. Chen, J. Kelly, C. Quintana, R. Barends, B. Campbell, Y. Chen, B. Chiaro, A. Dunsworth, A. G. Fowler, E. Lucero, E. Jeffrey, A. Megrant, J. Mutus, M. Neeley, C. Neill, P. J. J. O'Malley, P. Roushan, D. Sank, A. Vainsencher, J. Wenner, T. C. White, A. N. Korotkov, and J. M. Martinis, Phys. Rev. Lett. 116, 020501 (2016).   
$^{38}$ E. Magesan, J. M. Gambetta, and J. Emerson, Phys. Rev. Lett. 106, 180504 (2011).   
$^{39}$ A. D. Córcoles, J. M. Gambetta, J. M. Chow, J. A. Smolin, M. Ware, J. Strand, B. L. T. Plourde, and M. Steffen, Phys. Rev. A 87, 030301 (2013).   
$^{40}$ K. Geerlings, Z. Leghtas, I. M. Pop, S. Shankar, L. Frunzio, R. J. Schoelkopf, M. Mirrahimi, , and M. H. Devoret, Phys. Rev. Lett. 110, 120501 (2013).   
$^{41}$ J. Johansson, P. Nation, and F. Nori, Computer Physics Communications 184, 1234 (2013).   
$^{42}$ M. Grant and S. Boyd, “CVX: Matlab software for disciplined convex programming, version 2.1,” http://cvxr.com/cvx (2014).

![](images/b49b995d798cb30dafa53991156c0d269d6b4da8d857c4d95b54b698223cec7a.jpg)

<details>
<summary>natural_image</summary>

Close-up of a microchip mounted on a yellow circuit board with multiple gold connectors (no visible text or symbols)
</details>

![](images/3fa9486fef20f729dd7de8ce88ec9acf04e8536946b77dc0932d965b8efe72c1.jpg)

<details>
<summary>natural_image</summary>

Close-up of a small electronic component mounted on a perforated circuit board (no visible text or symbols)
</details>

FIG. S1. Device images. a, Angled top view and b, angled side view photographs of the fully packaged device. The top (mechanics) chip is secured face-down to the bottom (qubit) chip by an adhesive polymer (9:1 ethanol to GE varnish) applied manually to the sides of the chip.   
![](images/aeba3658c034d0249cd29cbec1831e7f8209b75a66c273fd4664905b594d45d4.jpg)

<details>
<summary>line</summary>

| Applied magnetic flux, Φe / Φ0 | Frequency (GHz) |
| ------------------------------ | --------------- |
| 0.0                            | 2.4             |
| 0.1                            | 2.3             |
| 0.2                            | 2.1             |
| 0.3                            | 1.7             |
</details>

![](images/dd6b5a4043c6410f642ddd6311aa982d7ae88b1f52d8b12bacf5c0140fca4402.jpg)

<details>
<summary>heatmap</summary>

| Qubit Frequency (GHz) | Time (μs) | Amplitude (mV) |
| --------------------- | --------- | -------------- |
| 2.4                   | 0         | 35             |
| 2.3                   | 1         | 30             |
| 2.2                   | 2         | 25             |
| 2.1                   | 3         | 20             |
| 2.0                   | 4         | 15             |
| 1.9                   | 5         | 10             |
| 1.8                   | 6         | 5              |
| 1.7                   | 7         | 0              |
</details>

![](images/219c19efb80f71189fc1a7d3a38c0f223759c14a6b5744ba48a0d48c32b37c27.jpg)

<details>
<summary>line</summary>

| Number of Cliffords | Amplitude (mV) |
| ------------------- | -------------- |
| 0                   | 40             |
| 50                  | 30             |
| 100                 | 25             |
| 150                 | 22             |
| 200                 | 21             |
| 250                 | 20             |
| 300                 | 20             |
| 350                 | 20             |
</details>

![](images/40c2f9918b497ab4eb169448f02987761604d206d1bb2b46218d31a714b68b9d.jpg)

<details>
<summary>line</summary>

| e-f Drive Amplitude (V) | Amplitude (mV) |
| ----------------------- | -------------- |
| -400                    | 25             |
| -300                    | 0              |
| -200                    | 25             |
| -100                    | 0              |
| 0                       | 25             |
| 100                     | 0              |
| 200                     | 25             |
| 300                     | 0              |
| 400                     | 25             |
| 500                     | 0              |
</details>

FIG. S2. Qubit characterization. a, Qubit spectrum as a function of the externally applied magnetic flux $\Phi_{e}$ , in units of the magnetic flux quantum $\Phi_{0}$ . The qubit is tuned from its maximum frequency $\omega_{ge}^{max}/2\pi = 2.443GHz$ and shows avoided crossings at $\omega_{m_{1}}/2\pi = 2.053GHz$ and $\omega_{m_{2}}/2\pi = 2.339GHz$ , corresponding to the mechanical modes $M_{1}$ and $M_{2}$ . b, Qubit $T_{1}$ as a function of qubit frequency. Each horizontal slice represents a separate ringdown measurement, with qubit excited state probability indicated by the color bar. The fitted $T_{1}$ values for each slice are plotted as white points. c, Randomized benchmarking results. Here, the qubit response is measured after applying a random sequence of Clifford gates, with the final Clifford always chosen to map the sequence's the cumulative effect to $|e\rangle$ . d, Qubit thermometry measurement. The dark blue points (dark red points) show the measurement result with (without) an initial $X_{\pi}$ pulse to exchange the steady-state $|g\rangle$ and $|e\rangle$ populations. The light blue (light red) lines show the fits for these Rabi-like oscillations, with amplitudes $A_{g}$ ( $A_{e}$ ). From these amplitudes, we estimate a qubit thermal population in $|e\rangle$ of $P_{e,th} = 0.057$ .

![](images/51b9fb39bb9b6131e15fb2d4e9ffd713e15a0adfe05ec31fdf534ef978603242.jpg)

![](images/09ac9f4780166315604ed73acc879679a288ef81fd4b948779719e63dbedde38.jpg)

<details>
<summary>line</summary>

| Interaction time, τ (ns) | Bloch vector |
| ------------------------ | ------------ |
| 0                        | 1.0          |
| 50                       | -1.0         |
| 100                      | 1.0          |
| 150                      | -1.0         |
</details>

![](images/b3374aa9f1878da3231d56b69f2f4373110907ed2534b260d7bedc067bf5c167.jpg)

<details>
<summary>line</summary>

| Interaction time, τ (ns) | Bloch vector (black) | Bloch vector (blue) | Bloch vector (red) |
| ------------------------ | -------------------- | ------------------- | ------------------ |
| 0                        | 0.0                  | 0.0                 | 0.0                |
| 50                       | -1.0                 | 0.5                 | 0.3                |
| 100                      | 0.0                  | 0.0                 | 0.0                |
</details>

FIG. S3. Swap characterization. a, Experimental results (top) and simulated qubit excited state probability $P_{e}$ (bottom) for the Rabi-swap experiment described in Fig. 2b,d of the main text. The asymmetry in the chevrons of the qubit-mechanics interaction is replicated by master equation simulations. b, Qubit state tomography results for the X (blue), Y (red), and Z (black) components of the qubit state Bloch vector during a resonant Rabi-swap experiment. The qubit is initially prepared in $|e\rangle$ , then swapped to the upper mechanical mode $M_{2}$ before performing tomography on the qubit. c, Qubit state tomography results for a resonant Rabi-swap experiment, similar to b, where the qubit now starts in $|g\rangle + |e\rangle$ .

![](images/768db776efffb84ab05a517288ea3d40978e3770526c50c18eea890d9ee380ff.jpg)

![](images/d6831c087811f352d1d56af030fad5060083ce9afaee180c316ba8b264e20317.jpg)

![](images/dcd643de47a718c975776491f6395d6bcd9f04e19a5c8a479046315fb93cbef3.jpg)

<details>
<summary>histogram</summary>

| Reconstructed Fidelity | Counts |
| ---------------------- | ------ |
| 0.805                  | 0      |
| 0.81                   | 300    |
| 0.815                  | 0      |
</details>

![](images/b27d4afd12ee6884b7ab9fe1526bd956c5d9858b8953b5823cbd1c1d2c5a5163.jpg)

<details>
<summary>histogram</summary>

| Reconstructed Fidelity | Counts |
| ---------------------- | ------ |
| 0.50                   | 10     |
| 0.51                   | 20     |
| 0.52                   | 40     |
| 0.53                   | 70     |
| 0.54                   | 100    |
| 0.55                   | 150    |
| 0.56                   | 200    |
| 0.57                   | 250    |
| 0.58                   | 300    |
| 0.59                   | 280    |
| 0.60                   | 220    |
| 0.61                   | 150    |
| 0.62                   | 80     |
| 0.63                   | 40     |
| 0.64                   | 20     |
| 0.65                   | 10     |
</details>

FIG. S4. State reconstruction. a, Experimental results for the mechanical displacement calibration, showing the inferred displacement amplitudes $|\alpha_{\mathrm{inf}}|$ (points) corresponding to each applied pulse's voltage amplitude, and a fit to the hockey-stick model (line). b, Simulation and fit of the displacement calibration performed experimentally in a. A small thermal state $(P_{\mathrm{th}} = 0.10)$ is displaced with a programmed amplitude (input $|\alpha|$ , x-axis), from which we determine the inferred displacement amplitude $(|\alpha_{\mathrm{inf}}|, \mathrm{y - axis})$ . The simulation data are plotted in dark blue points, with a fit to the hockey-stick model plotted in light blue. In the linear portion of the graph, we find the ratio of these values, $c_1 = (\text{inferred } |\alpha|) / (\text{input } |\alpha|) \simeq 1$ , as expected. c, Results of error propagation for the reconstructed fidelity $\mathcal{F}_s = 0.811 \pm 0.002$ in single-mode tomography of the $|0\rangle + |1\rangle$ state. d, Results of error propagation for the joint tomography Bell-state fidelity $\mathcal{F}_{\mathrm{Bell}} = 0.57 \pm 0.02$ .

![](images/794493e824e4063158c4f3185f93ef3523c0e4ac80cbd2bad61a77d2db78f3ae.jpg)

<details>
<summary>heatmap</summary>

| Re(V) [mV] | Im(V) [mV] |
|---|---|
| 0 | 600 |
| 200 | 400 |
| 400 | 200 |
| 600 | 0 |
</details>

![](images/cf45a7ead75059125d7560ec57a5a3bb9d4434f8e96aa6ca8017af84f978572f.jpg)

<details>
<summary>heatmap</summary>

| Re(α) \ Im(α) | 0    | 0.5  | 1    |
| ------------- | ---- | ---- | ---- |
| 0.5           | Red  | White| Blue |
| 1.0           | Blue | Blue | Blue |
| 1.5           | Blue | Blue | Blue |
</details>

![](images/7986126b4556226a2c289ad1e7a68bbe062c21cfea26e129835b99b5e30c71d2.jpg)

<details>
<summary>heatmap</summary>

| Re(α) | Im(α) | Parity |
|-------|-------|--------|
| 0.0   | 0.0   | -1.0   |
| 0.5   | 0.5   | 0.0    |
| 1.0   | 1.0   | 0.5    |
| 1.5   | 1.5   | 1.0    |
</details>

FIG. S5. Direct parity measurement. a, Parity of the displaced mechanical state $|1\rangle$ prepared in the upper mechanical mode. The state is prepared and characterized using the same pulse sequence as in Fig. 3a. Here, we use 16 displacements in the upper-right quadrant of the complex plane and perform the Ramsey measurement for a total time $t_{max} = 4.0 \mu s$ . For each displacement $\hat{D}(\alpha)$ , parity is computed from the extracted phonon number distribution as $\Pi(\alpha) = \Sigma_{n=0}(-1)^{n} P_{\alpha}(n)$ . The axes correspond to the complex voltage amplitudes of the applied microwave pulses that generate these displacements. This yields a minimum observed parity $\Pi(0) \simeq -0.36$ at the origin. b, The same parity measurement, plotted in terms of the inferred displacement amplitudes. We extrapolate these $\alpha$ values using the calibration scheme described in Fig. S4. c, Upper-right quadrant of the reconstructed Wigner function shown in Fig. 3f, reproduced here for comparison.

<table><tr><td>Parameter</td><td>Value(s)</td></tr><tr><td> $\omega_{\text{ge}}^{\text{max}}/2\pi$ </td><td>2.443 GHz</td></tr><tr><td> $\alpha_q/2\pi$ </td><td>126 MHz</td></tr><tr><td> $T_1$ </td><td>(4.9 ± 2.3) μs</td></tr><tr><td> $T_2$  (flux sweet spot)</td><td>1.4 μs</td></tr><tr><td> $T_2$  ( $\omega_{\text{ge}}/2\pi = 2.26$  GHz)</td><td>0.8 - 1.2 μs</td></tr><tr><td> $\omega_{\text{m}_i}/2\pi$ </td><td>2.053, 2.339 GHz</td></tr><tr><td> $T_{1,\text{m}}$ </td><td>1.23, 0.99 μs</td></tr><tr><td> $T_{2,\text{m}}$ </td><td>0.87, 1.71 μs</td></tr><tr><td> $g_i/2\pi$ </td><td>9.5, 10.5 MHz</td></tr><tr><td> $\omega_{\text{r}}^{\text{max}}/2\pi$ </td><td>2.872 GHz</td></tr><tr><td> $\kappa_r/2\pi$ </td><td>1.29 MHz</td></tr></table>

TABLE S1. Device parameters. Parameters of the qubit, mechanical modes, and readout resonator.