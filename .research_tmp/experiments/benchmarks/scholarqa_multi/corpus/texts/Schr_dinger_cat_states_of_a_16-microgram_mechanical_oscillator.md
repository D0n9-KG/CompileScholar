# Schrödinger cat states of a 16-microgram mechanical oscillator

Marius Bild $^{1,2,\dagger}$ , Matteo Fadel $^{1,2,\dagger,*}$ , Yu Yang $^{1,2,\dagger}$ , Uwe von Lüpke $^{1,2}$ , Phillip Martin $^{1,2}$ , Alessandro Bruno $^{1,2}$ , and Yiwen Chu $^{1,2,**}$

$^{1}$ Department of Physics, ETH Zürich, 8093 Zürich, Switzerland

$^{2}$ Quantum Center, ETH Zürich, 8093 Zürich, Switzerland

$^{*}$ fadelm@phys.ethz.ch

\*\*yiwen.chu@phys.ethz.ch

$^{\dagger}$ these authors contributed equally to this work

The superposition principle is one of the most fundamental principles of quantum mechanics. According to the Schrödinger equation, a physical system can be in any linear combination of its possible states. While the validity of this principle is routinely validated for microscopic systems, it is still unclear why we do not observe macroscopic objects to be in superpositions of states that can be distinguished by some classical property. Here we demonstrate the preparation of a mechanical resonator with an effective mass of 16.2 micrograms in Schrödinger cat states of motion, where the constituent atoms are in a superposition of oscillating with two opposite phases. We show control over the size and phase of the superposition and investigate the decoherence dynamics of these states. Apart from shedding light at the boundary between the quantum and the classical world, our results are of interest for quantum technologies, as they pave the way towards continuous-variable quantum information processing and quantum metrology with mechanical resonators.

Quantum mechanics is one of the most successful scientific theories ever formulated. However, from the early days of quantum mechanics until now, it has been unclear why quantum phenomena, such as state superpositions, are never observed in the macroscopic world. In his 1935 work $^{1}$ , Erwin Schrödinger imagined a device able to poison a cat as a consequence of a radioactive decay, concluding that the superposition of an atom being “decayed” and “not decayed” could be mapped onto a superposition of the cat being simultaneously “dead” and “alive”. There are two aspects of this hypothetical scenario that make it seem absurd and counter intuitive: First, a cat is a macroscopic, everyday object, and second, ”dead” and ”alive” are states with properties that are clearly distinguishable within our classical experience.

Many explanations have been proposed as to why we may never encounter a cat in such an unfortunate situation. Macroscopic objects may simply be too complex and subject to too many sources of decoherence to sustain a superposition of classically distinct states. Other theories introduce additional effects beyond standard quantum mechanics, such as wavefunction collapse due to intrinsic stochastic noise or gravitational decoherence $^{2}$ . In keeping with the spirit of Schrödinger's cat, these effects are typically expected to scale with the mass of the system and the distinctness of the states that are superposed. Therefore, observing state superpositions in massive objects is of key importance for exploring the validity range of quantum mechanics as we know it. Beyond its fundamental interest, preparing and detecting Schrödinger's cat states is essential for applications in quantum technologies. Main examples include Heisenberg-limited parameter estimation protocols $^{3,4}$ and error-protected quantum information processing $^{5,6}$ .

There have been many experimental demonstrations of Schrödinger cat states (which we will call “cat states” from here on). These include superpositions of internal and motional degrees of freedom in trapped ions $^{7,8}$ , phase-space superpositions of electromagnetic waves in both the optical $^{9,10}$ and microwave domains $^{11-13}$ , Greenberger–Horne–Zeilinger states $^{14,15}$ , current superpositions in SQUIDs $^{16}$ , and spatial superpositions of large molecules $^{17}$ . In this work, we experimentally demonstrate the preparation of cat states in the motional degree of freedom of a solid state mechanical resonator. Given the variety of definitions found in previous works, here we define a cat state of a harmonic oscillator as a coherent superposition of two or more states with well-separated phase space distributions.

Our mechanical resonator is a high-overtone bulk acoustic-wave resonator (HBAR), which we couple to a superconducting transmon qubit. The latter allows us to create, control, and read out phonon states in the HBAR. Qubit and HBAR are fabricated on separate sapphire chips, which are subsequently flip chip bonded into the final device (Fig. 1a). The acoustic free spectral range is approximately 12 MHz, and frequency tuning the qubit allows us to address several longitudinal phononic modes. The acoustic lattice oscillations are localized within a Gaussian mode with waist $w_{0} = 27 \mu m$ and length $L = 435 \mu m$ , giving a mode volume of $\pi w_{0}^{2}L \approx 0.001 mm^{3}$ (see supplementary materials $^{18}$ , section B for details). More details about this circuit quantum acoustodynamics (cQAD) system $^{19}$ and the device $^{20}$ can be found in previous works.

In the classical picture, one can imagine a coherent state $|\alpha\rangle$ in the phonon mode as a coherent displacement of the atomic lattice with an amplitude proportional to $\alpha$ . In the quantum picture,

![](images/d438eed41da2646bf929cb72ddaf1656441c7f873acabf860eaa2c71dfd71f63.jpg)

<details>
<summary>natural_image</summary>

Diagram of a layered crystal structure with a highlighted inset showing a pink sphere and a black rectangular element (no text or symbols present)
</details>

![](images/0cfa6cfe6657ef9eeec726bb02bb27f4b3fa925932eb9767b1877b6551d8ccab.jpg)

![](images/14ff1d5ccf1e47cf73404b3e02d140511d041dc0c8d2b801ce87016a5647c441.jpg)

<details>
<summary>text_image</summary>

C
|Φ₊⟩ Im(β)
t_c
|α⟩ interference
Re(β)
t_R
|Φ₋⟩
</details>

Figure 1: Illustration of the $\hbar$ BAR device and system evolution. (a) Schematics of the $\hbar$ BAR device. The HBAR chip (top) has a layer of piezoelectric aluminum nitride (orange) and supports standing acoustic waves (pink). The transmon qubit on the lower chip has a circular antenna to couple with the HBAR. The inset shows the superposition of two opposite-phase oscillations of atoms in the crystal lattice. (b) Simulated evolution of the qubit $|e\rangle$ state population $P_{|e\rangle}$ and purity $\gamma$ under the JC interaction when the qubit is initialized in $|-Z\rangle$ and the phonon in a coherent state. (c) Illustration of the evolution of an initial phonon coherent state (red circle on the left) in phase space. The blue (yellow) crescent shapes indicate the state $|\Phi_+ \rangle (|\Phi_- \rangle)$ , which is the phonon state when the qubit is initialized in $|+X\rangle (|-X\rangle)$ . Interference fringes appear around time $t_C$ when the qubit is prepared in a superposition of $|+X\rangle$ and $|-X\rangle$ . Around the revival time $t_R$ , the two phonon states again overlap (purple).

an example of a cat state is a quantum superposition of two coherent states with opposite displacement amplitudes, leading to the physical interpretation of such a state as the superposition of two oscillations of the atomic lattice with the same frequency $\omega_{p}$ and relative phase $\pi$ . Considering a snapshot in time where both oscillations are at their displacement maximum, Schrödinger's cat being in a superposition of dead and alive is analogous to a superposition of atoms in the HBAR being in two distinct positions in space, as illustrated in the inset of Fig.1a. Note that here, we define the positions as distinct when their separation is larger than the fluctuations due to quantum, thermal, or other sources of noise.

To realize a cat state in our system, we use the Jaynes-Cummings (JC) interaction with the qubit and phonon on resonance $^{11,21}$ . The interaction Hamiltonian is

$$
H / \hbar = g _ {0} (\sigma^ {+} a + \sigma^ {-} a ^ {\dagger})  , \tag {1}
$$

where $g_{0}$ is the coupling strength between qubit and phonon mode, $\sigma^{+}$ is the raising operator for the qubit, and $a^{\dagger}$ the raising operator for the phonon mode. This results in Rabi oscillations between the states $|e, n-1\rangle$ and $|g, n\rangle$ at a rate $g_{0}\sqrt{n}$ , where $|g\rangle(|e\rangle)$ is the qubit ground (excited) state and $|n\rangle$ is the Fock state of n phonons. As a consequence of this $\sqrt{n}$ scaling, if the phonon mode is prepared in a coherent state with large enough amplitude and the qubit is prepared in $|g\rangle$ or $|e\rangle$ , their coherent interaction rapidly dephases. Hence, the oscillations of the qubit population “collapses” (see Fig. 1b) with a decaying amplitude proportional to $^{22,23}\exp(-(t/t_{\text{collapse}})^{2})$ , where $t_{collapse} = \sqrt{2}/g_{0}$ is the collapse time in the limit of $\alpha \gg 1$ . At this time, the qubit and phonon states are entangled. This can be seen in Fig.1b as a minimum in the qubit state purity $\gamma(t) = \text{Tr}(\rho_{\text{q}}(t)^{2})$ around $t_{collapse}$ , where $\rho_{\text{q}}(t)$ is the reduced density matrix of the qubit. Strikingly, due

to the quantized phonon energy and the consequent discrete oscillation frequency spectrum, the oscillations revive in finite time $^{21}$ . For $\alpha \gg 1$ , this revival occurs at $t_{R} = 2\pi\alpha/g_{0}^{24}$ . Between the collapse and revival, at time $t_{R}/2$ , the qubit and phonon disentangle from each other. The state being separable at $t_{R}/2$ coincides with the occurrence of a superposition of two distinct states in phase space, realizing a cat state in the phonon mode $^{11,18,21,25}$ .

A more intuitive explanation for the origin of the cat state comes from the time evolution of the reduced phonon state in phase space. As illustrated in Fig. 1c and shown in the supplementary materials $^{18}$ , section A, if the qubit is initialized in the state $|\pm X\rangle \equiv (|e\rangle \pm |g\rangle)/\sqrt{2}$ and in the limit of large $\alpha$ , the evolution leads to a rotation in phase space with an angular velocity $\mp |g_{0}/2\alpha|$ and a distortion of the coherent states. We call the resulting states $|\Phi_{\pm}(t)\rangle$ , whose full expressions are given in the supplementary materials $^{18}$ (Eq. S15). Initializing the qubit in the state $|\pm Z\rangle \equiv (|+X\rangle \pm |-X\rangle)/\sqrt{2}$ , the phonon state will evolve into $|\Phi_{+}(t)\rangle \pm |\Phi_{-}(t)\rangle$ as shown in Fig. 1c. At time $t_{R}/2$ , the two state components $|\Phi_{\pm}\rangle$ have covered a rotation angle of $\mp\pi/2$ around a circle of radius $\alpha$ , maximizing their separation in phase space and forming a cat state $^{21}$ . Finally, at the revival time $t_{R}$ , the two phonon state components $|\Phi_{\pm}\rangle$ have both rotated by a phase of $\pi$ and approximately recombine in phase space.

In the following, we experimentally confirm both the predicted collapse and revival of Rabi oscillations and the creation of mechanical cat states in the phonon mode. The basic sequence used in the experimental demonstration of the JC dynamics described above can be seen in Fig. 2a. We displace the phonon mode with a resonant drive of amplitude A to a coherent state with amplitude

a   
![](images/a207a297386a55ea6e57198a725ff87d92e41a9c0fa66597cac520cf7bce33fa.jpg)

<details>
<summary>line</summary>

| Time Segment       | Waveform Description |
| ------------------ | --------------------- |
| D(α)               | qubit                 |
| SWAP               | phonon                |
| qubit init.        | qubit                 |
| resonant interaction | quantified value     |
| measurement         | measurement           |
</details>

b   
![](images/7471034d0c0a6828cec387f8f5f264f2da1adcba72ea87f08c50ec5f7af5950f.jpg)

<details>
<summary>line</summary>

| interaction time t (μs) | P_e) | γ(t) | P_l e) |
| ----------------------- | ---- | ---- | ------ |
| 0                       | 1.0  | 0.8  | 0.7    |
| 2                       | 0.3  | 0.6  | 0.5    |
| 4                       | 0.5  | 0.7  | 0.6    |
| 6                       | 0.2  | 0.5  | 0.4    |
| 8                       | 0.1  | 0.4  | 0.3    |
| 10                      | 0.1  | 0.3  | 0.2    |
</details>

C   
![](images/34abef3e9097ba7bdb954456759c08f39baa72aba05d7ecf9e4ccc3315dfe59d.jpg)  
Figure 2: Collapse and revival dynamics. (a) Experimental sequence for observing collapse and revivals dynamics and for preparing cat states (details in the main text). (b) Measured qubit population and state purity. The solid and dashed black lines are the simulation results of the qubit population and purity, respectively. Three time points of particular interest are highlighted (dashed lines): initial state time, cat state time ( $t_C$ ), and revival state time ( $t_R$ ). (c) Measured Wigner function of the phonon state at the three time points. Axes are the real and imaginary parts of the complex displacement amplitude $\beta$ used during Wigner tomography $^{20}$ . The black crosses indicate the positions of the two coherent states composing the fitted CSS state Eq. (2). (d) Corresponding simulated Wigner functions.

α. To mitigate any effect of the drive on the qubit state, we then cool the qubit with an ancillary phonon mode $^{19,20}$ . The qubit is subsequently prepared in its initial state by applying a drive pulse with variable phase and amplitude. In order to induce the resonant interaction, we tune the qubit to the phonon mode frequency for a variable interaction time t. Depending on which of the subsystems we want to characterize, we choose a measurement sequence that implements the appropriate measurement operator. First, we simply measure the qubit excited state population. The resulting data is shown in Fig. 2b for A = 0.35. Here the value for A is a scaling factor for the amplitude of a microwave drive, which we calibrate to find a corresponding initial coherent state size of $\alpha = 1.75$ (see supplementary materials $^{18}$ , section D). As expected, we observe oscillations that collapse after a time $t_{collapse} \approx 0.9 \mu s$ and revive at $t_{R} \approx 6.7 \mu s$ . This revival indicates the coherent exchange of energy quanta between the qubit and phonon mode during the resonant interaction. By performing full qubit tomography after the resonant interaction, we can also reconstruct the reduced density matrix of the qubit subsystem $\rho_{q}$ and calculate the purity of the qubit state $\gamma(t)$ . We confirm a local minimum of $\gamma(t)$ around the predicted collapse time $t_{collapse}$ , followed by a local maximum around $t_{R}/2$ (Fig. 2b).

We now focus on the time evolution of the phonon subsystem by performing full Wigner tomography of the phonon state after the resonant interaction times t = 0, 2.9 and 7.0 $\mu$ s. To this end, we use the parity measurement technique established in a previous work $^{20}$ . To compensate for the effect of qubit dephasing during the parity measurement, we normalize all measured parity values to that of the Fock $|0\rangle$ phonon state (see supplementary materials $^{18}$ , section C). The measured Wigner functions are shown in Fig. 2c, where axes in phase space are normalized by measuring

the distribution of populations in the phonon Fock states for coherent states created with different drive amplitudes $^{19}$ (see supplementary materials $^{18}$ , section D).

From the measured data, we confirm the evolution of the initial coherent state (Fig. 2c left) into a cat state at $t_C = 2.9\mu \mathrm{s}$ (Fig. 2c center), showing two state components clearly distinct in phase space and interference fringes located between them. We choose this value of $t_C$ because it corresponds to the measured maximum in the qubit state purity. It deviates somewhat from the value predicted using the large $\alpha$ limit, which is $t_R / 2 \approx 3.3\mu \mathrm{s}$ . For the evolution time $t = 7.0\mu \mathrm{s}$ , the predicted refocusing into a crescent shaped overlap between the counter-rotating state components can be observed (Fig. 2c right).

In order to benchmark the cat state and obtain an estimate of its size, we implement a maximum-likelihood reconstruction $^{26}$ of the phonon state $\rho_{p}$ from the measured state with A = 0.35 and $t_{C} = 2.9 \mu s$ . We then fit the reconstructed state to an analytical expression of the expected phonon state $\rho(t_{C})$ in the absence of decoherence and after tracing out the qubit (see supplementary materials $^{18}$ , section A). Fixing the interaction time to $t_{C}$ from the experiment, the fit maximizes the fidelity between $\rho_{p}$ and $\rho(t_{C})$ by varying the initial coherent state size $\alpha_{\mathrm{fit}}$ of $\rho(t_{C})$ . The result yields a fidelity of $F \approx 76\%$ to an analytical state with initial coherent state size $\alpha_{\mathrm{fit}} = 1.62$ , which is smaller than the initial displacement $\alpha = 1.75$ because the expression for $\rho(t_{C})$ does not include phonon losses. We attribute the infidelity to a combination of decoherence and measurement imperfections that lead to additional artifacts in the Wigner function $^{20}$ . To further confirm that the phonon state behaves as expected, Fig. 2d shows the results of a master equation simulation

of the full experimental protocol with independently measured system parameters, showing good agreement with the measurements in Fig. 2c.

The state we obtained resembles the two-component coherent state superpositions (CSS)

$$
| C \rangle = \mathcal {N} \left(| \alpha_ {1} \rangle + e ^ {i \vartheta} | \alpha_ {2} \rangle\right), \tag {2}
$$

a type of cat state that is often invoked in quantum information $^{5,6}$ and parameter estimation protocols $^{3,4}$ . Here $|\alpha_{1,2}\rangle$ are two coherent states and $\mathcal{N}$ is the appropriate normalization constant. We can fit our reconstructed state to Eq. (2) by optimizing $\alpha_1$ , $\alpha_2$ , and $\vartheta$ for maximum fidelity $\mathcal{F}(\rho_{\mathrm{p}}, |C\rangle\langle C|)$ . Since our state is not centered around the origin in phase space, we use half the phase space distance between the coherent state components $D = |\alpha_1 - \alpha_2|/2$ as a measure of the cat state size. This choice is motivated by considering a coherent state superposition centered around the origin in phase space, such that $\alpha_1 = -\alpha_2$ . Then $D = |\alpha_{1,2}| = \sqrt{\bar{n}}$ , where $\bar{n}$ is the average phonon population of the state created. For the state in Fig. 2c, we obtain a state size $D = 1.61$ , corresponding to $\bar{n} = D^2 = 2.60$ , with a fidelity of $\mathcal{F} \approx 66\%$ . The smaller state size $D$ compared to the initial coherent displacement is a combination of decoherence and the choice of interaction time $t_C < t_R/2$ , resulting in the two counter-rotating state components not reaching their maximum separation in phase space. The fidelity is lower compared to the fitted analytical state $\rho(t_C)$ , since $\rho(t_C)$ itself has a finite infidelity to the CSS state $|C\rangle$ .

We can now translate the parameters of the measured cat state into physical properties of the phonon mode, such as the spatial separation between atoms. A state size of D = 1.61 corresponds to a maximal delocalization of $7.0 \cdot x_{ZPF}$ , where $x_{ZPF}$ is the zero point motion of an equivalent

1D quantum harmonic oscillator. Since we are not considering a center-of-mass mode, there is some freedom in choosing $x_{ZPF}$ , which is then associated with an effective oscillating mass of the mode. If we choose the root-mean-square (rms) value of the atomic displacements, we find an effective mass of $M_{\mathrm{eff}}^{\mathrm{(rms)}} = 16.2 \mu\mathrm{g}$ , corresponding to $\sim 10^{17}$ atoms, delocalized over a distance of $2.1 \cdot 10^{-18}$ m (see supplementary materials $^{18}$ , section B).

In applications such as bosonic encodings of a qubit state, full control over the phase and amplitude of the created cat state is required $^{5,6,27}$ . In the following, we demonstrate this level of control in our experiment. By varying the amplitude A of the phonon displacement drive, we can control the amplitude of the initial coherent state and the size of the resulting cat state. For displacement amplitudes A = 0.25 and 0.30, we create cat states with D = 1.09 and 1.43, respectively (Fig. 3a). The fidelities of reconstructions of the measured states to both a CSS state and $\rho(t_{C})$ are given in Fig. 3b. The best fit $\rho(t_{C})$ are plotted in the lower row of Fig. 3a, showing good qualitative agreement with the data. As before, the finite fidelities and lower contrast fringes of the measured states compared to the best fit $\rho(t_{C})$ arise mainly from decoherence of the state during measurement.

In the two-component cat state encoding of a qubit, the six cardinal points of the Bloch sphere are given by two coherent states $|\alpha_{1}\rangle$ and $|\alpha_{2}\rangle$ , along with their four superpositions with $\pi/2$ difference in the phase $\vartheta$ (see Eq.(2)). We can prepare similar states by initializing the transmon qubit state in all six cardinal points $|\pm X\rangle$ , $|\pm Y\rangle$ , $|\pm Z\rangle$ of its Bloch sphere before performing the cat generation protocol. The preparation of $|\pm X\rangle$ and $|\pm Y\rangle$ is calibrated using collapse and

![](images/bfb25d1a163861d04be8e5391b83e2e0d06bf8c4af418c97cb989f04010a3f02.jpg)

<details>
<summary>scatter</summary>

| Re(β) | Im(β) | Label |
|-------|-------|-------|
| -1.0  | 1.0   | A=0.25 |
| -1.0  | -1.0  | A=0.25 |
| -1.0  | 0.0   | A=0.25 |
| -1.0  | -1.0  | A=0.25 |
| -1.0  | 0.0   | A=0.30 |
| -1.0  | 1.0   | A=0.30 |
| -1.0  | -1.0  | A=0.30 |
| -1.0  | 0.0   | A=0.30 |
| -1.0  | -1.0  | A=0.30 |
| -1.0  | 0.0   | A=0.30 |
| -1.0  | -1.0  | A=0.30 |
| -1.0  | 0.0   | A=0.30 |
| -1.0  | -2.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -2.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -2.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -3.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -3.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -3.0  | A=0.30 |
| -1.0  | 3.0   | A=0.30 |
| -1.0  | -3.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -3.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -4.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -4.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -4.0  | A=0.30 |
| -1.0  | 3.0   | A=0.30 |
| -1.0  | -3.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -3.0  | A=0.30 |
| -1.0  | 3.0   | A=0.30 |
| -1.0  | -4.0  | A=0.30 |
| -1.0  | 2.0   | A=0.30 |
| -1.0  | -4.0  | A=0.30 |
| -1.5  | 1.5   | A=0.25 |
| -1.5  | -1.5  | A=0.25 |
| -1.5  | 1.5   | A=0.25 |
| -1.5  | -1.5  | A=0.25 |
| -1.5  | 1.5   | A=0.25 |
| -1.5  | -1.5  | A=0.25 |
| -1.5  | 1.5   | A=0.26 |
| -1.5  | -1.5  | A=0.26 |
| -1.5  | 1.5   | A=0.26 |
| -1.5  | -1.5  | A=0.26 |
| -1.5  | 2.5   | A=0.26 |
| -1.5  | -2.5  | A=0.26 |
| -1.5  | 2.5   | A=0.26 |
| -1.5  | -2.5  | A=0.26 |
| -1.5  | 2.5   | A=0.26 |
| -1.5  | -2.5  | A=0.26 |
| -2.5  | 1.5   | A=0.25 |
| -2.5  | -1.5  | A=0.25 |
| -2.5  | 1.5   | A=0.25 |
| -2.5  | -1.5  | A=0.25 |
| -2.5  | 1.5   | A=0.26 |
| -2.5  | -1.5  | A=0.26 |
| -2.5  | 1.5   | A=0.26 |
| -2.5  | -1.5  | A=0.26 |
| -2.5  | 2.5   | A=0,26 |
| -2.5  | -2.5  | A=0,26 |
| -2.5  | 2.5   | A=0,26 |
| -2.5  | -3.5  | A=0,26 |
| -2.5  | 2.5   | A=0,26 |
| -2.5  | -4 .    | A=0,26 |
| -2 .    | 2 .    | A=0,26 |
| -2 .    | -2 .   | A=0,26 |
| -2 .    | 2 .    | A=1,26 |
| -2 .    | -2 .   | A=1,26 |
| -2 .    | 2 .    | A=1,26 |
| -2 .    | -4 .   | A=1,26 |
| -2 .    | 4 .    | A=1,26 |
| -2 .    | -4 .   | A[ ]
</details>

![](images/de4b68a51e1a4f15f3bd4c129fb73b2d867ffe18c0e206b7a55c2143ce01676b.jpg)

<details>
<summary>scatter</summary>

| displacement amplitude A | CSS cat size D | analytical cat size D | CSS α_fit | analytical α_fit |
| ------------------------- | -------------- | --------------------- | --------- | ---------------- |
| 0.25                      | 1.0            | 1.0                   | 82%       | 72%              |
| 0.30                      | 1.5            | 1.5                   | 78%       | 68%              |
| 0.35                      | 1.8            | 1.8                   | 76%       | 66%              |
</details>

![](images/93a6065176ab9a7f2d7fe5059837d234adb85c08c69a67037772383d4a785c14.jpg)

<details>
<summary>scatter</summary>

| Re(β) | Im(β) | Party | Value |
|-------|-------|-------|-------|
| -2    | 0     | +X    | 1     |
| -1    | 0     | +Y    | 1     |
| 0     | 0     | +Z    | 1     |
| 1     | 0     | -X    | 1     |
| -2    | 0     | -Y    | 1     |
| -1    | 0     | -Z    | 1     |
| 0     | 0     | +X    | 1     |
| 1     | 0     | +Y    | 1     |
| -2    | 0     | +Z    | 1     |
| -1    | 0     | -X    | 1     |
| 0     | 0     | -Y    | 1     |
| 1     | 0     | -Z    | 1     |
</details>

Figure 3: Cat state amplitude and phase control. (a) Cat states prepared with different displacement pulse amplitudes A. Top row: measured Wigner functions, Bottom row: Analytical state $\rho(t_{C})$ which best fits the data. The fitted CSS states, with coherent state positions indicated by black crosses, have D = 1.09 (1.43) for A = 0.25 (0.30). (b) Cat state sizes as a function of the displacement amplitude A, obtained from fitting the data in a to analytical states $\rho(t_{C})$ and CSS states $|C\rangle$ . Numbers are the fidelity with respect to the fitted state. Error bars show cat sizes resulting in 1% deviation in the fidelity (see supplementary materials ${}^{18}$ , section F) (c) Cat states resulting from the initial qubit states indicated by the respective labels.

revival measurements as a function of the qubit drive phase (see supplementary materials $^{18}$ , section A). The results for A = 0.35 and $t_{C} = 2.10 \mu s$ are shown in Fig. 3c. We observe a distorted coherent state located on the upper (lower) half of phase space for the qubit initially in $|\pm X\rangle$ . This separation in phase space is expected from the opposite rotation directions between the phonon states when the qubit is prepared in $|\pm X\rangle$ (see Fig.1c). The $|\pm Y\rangle$ , $|\pm Z\rangle$ states then give rise to four cat states that differ in phase by $\pi/2$ , as can be observed in the phases of the interference fringes in Fig. 3c. Note that the initial energy of the qubit, and thus of the total system, is not the same for all six scenarios, resulting in slightly different sizes for the cat states. In the limit of large cat size, this difference becomes negligible, and the phonon subspace maps onto that of the cat state encoding.

Superposition states are non-classical states that are notoriously prone to decoherence. We now investigate the quantum to classical transition of different sized cat states by letting them evolve freely for a varying wait time $\tau$ before performing Wigner tomography. In particular, we focus on a slice through the Wigner function's interference fringes at $\mathrm{Im}(\beta) = 0$ , which highlights the non-classical features of the superposition. Fig. 4a shows the time evolution of this slice for the $D = 1.43$ cat state. We observe that the negative features disappear on a time scale much faster than $T_{1}^{\mathrm{ph}} \approx 84\mu s$ , the energy relaxation time of the phonon mode.

As a measure for the non-classicality of the state, we extract the time dependent negativity $^{28}$ , defined as $\delta(t) \equiv \int (|W(\beta, t)| - W(\beta, t)) d\beta$ . Here, $W(\beta, t)$ is the measured Wigner function of the cat state at time t, and the integration is over the 1D slice in phase space parametrized by the

![](images/6b38475c6a77f8334b978f8097903117ee041919beae70aa636d65930924ef66.jpg)

<details>
<summary>heatmap</summary>

| Re(β) | Wait time τ (μs) | parity |
|-------|------------------|--------|
| -1.5  | 0                | -1     |
| -1.5  | 10               | -1     |
| -1.5  | 20               | -1     |
| -1.5  | 30               | -1     |
| -1.5  | 40               | -1     |
| -0.5  | 0                | -1     |
| -0.5  | 10               | -1     |
| -0.5  | 20               | -1     |
| -0.5  | 30               | -1     |
| -0.5  | 40               | -1     |
| 0.5   | 0                | -1     |
| 0.5   | 10               | -1     |
| 0.5   | 20               | -1     |
| 0.5   | 30               | -1     |
| 0.5   | 40               | -1     |
</details>

![](images/b386797de9cd891cc84593c8bdebb7f6ad5728d15a18233b093d9648d1b68ee1.jpg)

<details>
<summary>line</summary>

| wait time τ (μs) | D = 1.09 | D = 1.43 | D = 1.62 |
| ---------------- | -------- | -------- | -------- |
| 0                | 1.0      | 1.0      | 1.0      |
| 5                | 0.8      | 0.75     | 0.85     |
| 10               | 0.6      | 0.55     | 0.7      |
| 15               | 0.4      | 0.35     | 0.5      |
| 20               | 0.2      | 0.15     | 0.3      |
| 25               | 0.1      | 0.05     | 0.15     |
| 30               | 0.05     | 0.02     | 0.1      |
| 35               | 0.02     | 0.01     | 0.05     |
| 40               | 0.01     | 0.005    | 0.02     |
</details>

![](images/ed57963bf8a66136cd51cdd8e245be45af48e080bf7d6ad501b7311dc8d255cf.jpg)

<details>
<summary>scatter</summary>

| cat state size D | τ_cat (μs) |
| ---------------- | ---------- |
| 1.0              | 12.3       |
| 1.3              | 10.3       |
| 1.6              | 9.5        |
</details>

Figure 4: Decoherence of cat states. (a) Measured 1D cuts through the interference fringes of the D = 1.43 cat state for a range of wait times between state creation and measurement. (b): Extracted negativities (squares) from each cut vs. wait times for three cat state sizes, together with fitted exponential decays (solid lines). Both data and fitted curves are normalized to the fitted value at $\tau = 0$ . (c) Characteristic decay times $\tau_{cat}$ extracted from the fits in (b) for all three cat state sizes. Errorbars are uncertainties extracted from the fit.

complex displacement amplitude $\beta$ . Fig. 4b shows the resulting $\delta(t)$ for the three different cat state sizes of Fig. 3b. We fit each dataset to an exponential decay plus a constant offset. The offset in the measured Wigner values arises from the fact that our Wigner tomography is not performed in the ideal dispersive limit $^{20}$ . The extracted decay timescales $\tau_{cat}$ are plotted in Fig. 4c. We show in section G of the supplementary materials $^{18}$ that, in the limit of large $|\alpha|$ , $\delta(\tau)$ decays exponentially with a time constant $\tau_{\mathrm{cat}} = T_{1}^{\mathrm{ph}} / (2 |\alpha|^{2})$ . However, for small $|\alpha|$ , $\tau_{cat}$ deviates from this expression and is in fact dependent on properties of the exact state, such as the phase of the superposition. The data in Fig. 4c shows the expected qualitative behavior of faster decaying negativity for larger sized cat states, and we present a more detailed quantitative analysis in the supplementary materials $^{18}$ .

Our results show the generation of cat states in a microgram-mass solid-state mechanical mode using the tools of cQAD. We observed the collapse and revival dynamics of the qubit population under the resonant JC interaction and used it to create cat states of different sizes and superposition phases. As a proof-of-principle demonstration of how such macroscopic superpositions can be used to investigate mechanisms of quantum decoherence, we measured the decay of negativity in the Wigner function of different sized cat states. Future tests of wavefunction collapse models using such systems $^{29}$ would benefit from larger sized cats and longer phonon lifetimes. We note that the HBAR mode is a standing wave with longitudinal mode number $\sim 500$ , and a half-wavelength section approximates a center-of-mass mode. The mass of this section is on the order of 30 nanograms, by far the most massive object that has been placed into a cat state.

The maximum size of the cat state we can prepare is currently limited by our device parameters, including both the qubit and phonon decoherence rates. The latter is especially important given that, in general, the decoherence rate of the cat state is proportional to the square of the cat state size D. Furthermore, additional improvements to the properties of qubit and phonon resonator would enable alternative cat state generation protocols that can in principle lead to states with a higher fidelity to, for example, a CSS state $^{12,13}$ . We point out, however, that while CSS states represent a useful benchmark because they have been extensively studied for applications such as quantum information and quantum metrology, many of their salient features are present already in the states we have demonstrated. These include the phase-space separation of state components, important for error protection of encoded qubits $^{27,30,31}$ , and the presence of interference fringes with high Fisher information, useful for quantum-enhanced sensing ${}^{4,32,33}$ .

# Acknowledgements

We thank Oriol Romero-Isart, Alexander Grimm, and Ines C. Rodrigues for useful discussions, and Arianne Brooks for help with figure making. Fabrication of devices was performed at the FIRST cleanroom of ETH Zürich and the BRNC cleanroom of IBM Zürich. M.B. was supported by the QuantERA II Program that has received funding from the European Union's Horizon 2020 research and innovation program under Grant Agreement No 101017733, and with the Swiss National Science Foundation. M.F. was supported by The Branco Weiss Fellowship – Society in Science, administered by the ETH Zürich.

# Author contributions statement

U.v.L fabricated the device. M.B., M.F., Y.Y., and U.v.L performed the experiments and analyzed the data. All authors performed theoretical calculations and simulations. Y.C. conceived of the project and supervised the work. M.B., M.F., Y.Y., U.v.L, and Y.C. wrote the manuscript.

# Additional information

Competing interests The authors declare no competing interests.

Data and code availability Raw data, analysis code and QuTiP simulations are available from the corresponding author on reasonable request. Requests for materials should be addressed to M.F or Y.C.

# Supplementary information for “Schrödinger cat states of a 16-microgram mechanical oscillator”

Marius Bild $^{1,2,\dagger}$ , Matteo Fadel $^{1,2,\dagger,*}$ , Yu Yang $^{1,2,\dagger}$ , Uwe von Lüpke $^{1,2}$ , Phillip Martin $^{1,2}$ , Alessandro Bruno $^{1,2}$ , Yiwen Chu $^{1,2,**}$

$^{1}$ Department of Physics, ETH Zürich, 8093 Zürich, Switzerland   
$^{2}$ Quantum Center, ETH Zürich, 8093 Zürich, Switzerland   
$^{\dagger}$ these authors contributed equally to this work   
\* fadelm@phys.ethz.ch   
\*\* yiwen.chu@phys.ethz.ch

# Contents

A State evolution S3   
A.1 Evolution in the $x$ -basis S6   
B Lattice displacement and effective mass . . . . . . . . . . . . . . . . . . . . . . . . . . . . S11   
C Parity measurement calibration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . S14   
D Resonant phonon number measurement ..... S15   
E Post-processing and fitting of Wigner functions . . . . . . . . . . . . . . . . . . . . . . S17

F State reconstruction and fidelity ..... S19

G Cat state decoherence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . S23

G.1 Decoherence Measurements ..... S26

G.2 Decoherence simulations for analytical states . . . . . . . . . . . . . . . . S28

# A State evolution

For a resonant interaction between the qubit and the phonon, we can consider the time evolution given by the Jaynes–Cummings Hamiltonian

$$
H _ {\mathrm{JC}} = g _ {0} \left(\sigma^ {+} a + \sigma^ {-} a ^ {\dagger}\right). \tag {S1}
$$

Consider the initial product state

$$
\left| \Psi_ {0} \right\rangle = \left(c _ {g} | g \rangle + c _ {e} | e \rangle\right) \otimes | \alpha \rangle , \tag {S2}
$$

where

$$
| \alpha \rangle = e ^ {- | \alpha | ^ {2} / 2} \sum_ {n = 0} ^ {\infty} \frac {\alpha^ {n}}{\sqrt {n !}} | n \rangle \equiv \sum_ {n} c _ {n} | n \rangle \tag {S3}
$$

is a coherent state for the field. In the following, we will consider $\alpha \in \mathbb{R}$ for simplicity.

Defining the time-evolution operator $U_{JC}(t) = e^{-iH_{\mathrm{JC}}t}$ we get

$$
U _ {J C} (t) | g, n \rangle = \cos \left(g _ {0} \sqrt {n} t\right) | g, n \rangle - i \sin \left(g _ {0} \sqrt {n} t\right) | e, n - 1 \rangle \tag {S4}
$$

$$
U _ {J C} (t) | e, n \rangle = \cos \left(g _ {0} \sqrt {n + 1} t\right) | e, n \rangle - i \sin \left(g _ {0} \sqrt {n + 1} t\right) | g, n + 1 \rangle , \tag {S5}
$$

which gives

$$
\begin{array}{l} \left| \Psi (t) \right\rangle = U _ {J C} (t) \left| \Psi_ {0} \right\rangle (S6) \\ = \sum_ {n = 0} ^ {\infty} c _ {n} \left[ c _ {g} (\cos (g _ {0} \sqrt {n} t) | g, n \rangle - i \sin (g _ {0} \sqrt {n} t) | e, n - 1 \rangle\right) (S7) \\ \left. + c _ {e} \left(\cos \left(g _ {0} \sqrt {n + 1} t\right) | e, n \rangle - i \sin \left(g _ {0} \sqrt {n + 1} t\right) | g, n + 1 \rangle\right) \right] \\ = \sum_ {n = 0} ^ {\infty} \left[ \left(c _ {n} c _ {g} \cos \left(g _ {0} \sqrt {n} t\right) - i c _ {n - 1} c _ {e} \sin \left(g _ {0} \sqrt {n} t\right)\right) | g \rangle \right. (S8) \\ \left. \right.\left. \right. + \left( \right.c _ {n} c _ {e} \cos (g _ {0} \sqrt {n + 1} t) - i c _ {n + 1} c _ {g} \sin (g _ {0} \sqrt {n + 1} t)) | e \rangle \left. \right] | n \rangle . \\ \end{array}
$$

Tracing out the qubit state, we are left with the phonon state density matrix

$$
\rho (t) = \operatorname{Tr} _ {\mathrm{q}} \left(| \Psi (t) \rangle \langle \Psi (t) |\right) = \sum_ {i \in \{e, g \}} \langle i | \Psi (t) \rangle \langle \Psi (t) | i \rangle . \tag {S9}
$$

This is the “analytical state” that we then use to fit to the reconstructed measured states.

Inserting $c_{n+1} = c_n \alpha / \sqrt{n + 1}$ , the probability to measure the qubit in the excited state is

$$
\begin{array}{l} P _ {e} (t) = \sum_ {n = 0} ^ {\infty} | c _ {n} | ^ {2} \left(| c _ {e} | ^ {2} \cos (g _ {0} \sqrt {n + 1} t) ^ {2} + | c _ {g} | ^ {2} \frac {| \alpha | ^ {2}}{n + 1} \sin (g _ {0} \sqrt {n + 1} t) ^ {2}\right) \\ + 2 \cos (g _ {0} \sqrt {n + 1} t) \sin (g _ {0} \sqrt {n + 1} t) \Im \mathfrak {m} [ c _ {n + 1} c _ {n} ^ {*} c _ {g} c _ {e} ^ {*} ]. \tag {S10} \\ \end{array}
$$

Note that, for $\alpha = 0$ , only the term $n = 0$ matters, and one recovers the expression for the vacuum Rabi oscillations $P_{e}(t) = |c_{e}|^{2}\cos (g_{0}t)^{2}$ . In the limit of $\alpha \gg 1$ and $t\ll \alpha^2 /g_0$ , the sum in Eq. (S10) can be well approximated by a closed form. To see this, note that for $\alpha \gg 1$ the distribution of $c_{n}$ is sharply peaked around $n\approx \alpha^{2}$ . This allows us to expand $g_{0}t\sqrt{n + 1}\approx g_{0}t\sqrt{n}$ as

$$
\begin{array}{l} g _ {0} t \sqrt {n} = g _ {0} t \sqrt {\alpha^ {2} + (n - \alpha^ {2})} \\ \approx g _ {0} t \left(\alpha + \frac {1}{2} \frac {(n - \alpha^ {2})}{\alpha} - \frac {1}{8} \frac {(n - \alpha^ {2}) ^ {2}}{\alpha^ {3}}\right), \tag {S11} \\ \end{array}
$$

where terms of even higher order in n can be neglected for $g_{0}t \ll 8\alpha^{2}$ , since the standard deviation of $(n - \alpha^{2})$ is equal to $\alpha$ for a Poisson distribution.

If we consider only the term $\alpha$ in the expansion Eq. (S11), which is valid for short times $g_0t \ll 2$ or for the interaction with a classical field, we obtain the expression for undamped Rabi oscillations $P_e(t) = |c_e|^2 \cos(g_0\alpha t)^2 + |c_g|^2 \sin(g_0\alpha t)^2 + \sin(2g_0\alpha t)\mathfrak{Im}[c_g c_e^*]$ .

If we include also the second term in the expansion Eq. (S11), the probability Eq. (S10) can be calculated in the continuous limit by replacing $c_{n}$ by a Gaussian distribution and the summation over $n$ by an integral. We obtain

$$
P _ {e} (t) \approx \frac {1}{2} \left(1 + e ^ {- 2 (g _ {0} t / 2) ^ {2}} \left(\left(| c _ {e} | ^ {2} - | c _ {g} | ^ {2}\right) \cos (2 g _ {0} \alpha t) + 2 \Im \mathfrak {m} [ c _ {g} c _ {e} ^ {*} ] \sin (2 g _ {0} \alpha t)\right)\right). \tag {S12}
$$

From this result, we notice that the amplitude of the population oscillations decays as $e^{-(t/t_{\mathrm{collapse}})^{2}}$ , where $t_{collapse} = \sqrt{2}/g_{0}$ is the collapse time. The origin of this collapse is due to the fact that an atom interacting with a field in a Fock state $|n\rangle$ undergoes Rabi oscillations at a frequency $\sim\sqrt{n}$ , and that a field in a coherent state corresponds to a Poissonian distribution of Fock states. The destructive interference between oscillations with different frequency components that dephase results in a decaying signal analogous to the one originating from an inhomogeneously broadened ensemble of atoms.

Equation (S12) describes the collapse of Rabi oscillations, but it does not predict revivals. In fact, it is the discrete spectrum of oscillation frequencies that leads to revivals. Intuitively, this can be seen from the fact that oscillating terms $\cos(g_{0}\sqrt{nt})^{2} \approx \cos(g_{0}nt/\alpha + g_{0}\alpha t)$ interfere constructively when $g_{0}t/\alpha = 2\pi q$ , with $q = 1, 2, \ldots$ . This motivates the definition of the revival time as $t_{R} = 2\pi\alpha/g_{0}$ . To see this effect concretely, one can for example compute Eq. (S10) numerically by truncating the sum to a suitably large $n_{max}$ .

Here we note that the revivals in the Rabi oscillations are a genuinely quantum feature arising from the discrete spectrum, since if the field were classical, the frequencies would have a continuous spectrum, and such re-phasing could never occur within a finite time. Interestingly, this revival

does not tend to a coherent state in the limit of large $\alpha$ . Or, in other words, the amplitude of the Rabi oscillations at the revival points never reaches unity. To see this, we can use the results of Ref. $^{24}$ , where a bound for the revival contrast is presented (Eq. (6) there). Evaluating this expression at $t_{R}$ results in a contrast that approaches $B(t_{R}) = (1 + \pi^{2})^{-1/4} \simeq 0.55$ in the limit of large $\alpha$ .

Of particular interest is the phonon state at half the revival time, $t_{C} \equiv t_{R}/2 = \pi\alpha/g_{0}$ , since it takes the form of a cat state. To see this, first note that for large $\alpha$ the $c_{n}$ coefficients vary slowly with n (as they represent a distribution with standard deviation $\sim \alpha$ ), and therefore one can approximate $c_{n-1}$ and $c_{n+1}$ with $c_{n}$ in Eq. (S8). Moreover, around $t_{C}$ we can write $g_{0}t\sqrt{n+1} \simeq g_{0}t\sqrt{n} + \pi/2^{25}$ . With these approximations, the state Eq. (S8) simplifies to

$$
| \Psi (t) \rangle \simeq (| g \rangle - i | e \rangle) \sum_ {n = 0} ^ {\infty} \left[ c _ {g} \cos (g _ {0} \sqrt {n} t) - i c _ {e} \sin (g _ {0} \sqrt {n} t) \right] c _ {n} | n \rangle . \tag {S13}
$$

This is a product state, meaning that at $t \approx t_{C}$ the qubit and the bosonic mode disentangle. Moreover, note that the qubit state at this time does not depend on its initial state at time t = 0.

# A.1 Evolution in the x-basis

The previous description of the atom-field dynamics in the qubit's $Z$ -basis is especially suited for explaining collapse and revivals of the Rabi oscillations. However, to build a better intuition of what happens to the phonon state, it is more instructive to look at the state dynamics in a different basis. For the phonon mode initially in a coherent state $|\alpha\rangle$ with $\alpha \in \mathbb{R}$ , we consider the qubit's $X$ -

basis $\left|\pm X\right\rangle = \left(\left|e\right\rangle \pm \left|g\right\rangle\right)/\sqrt{2}$ . This choice is motivated by the fact that, in the semi-classical limit where the substitution $a \rightarrow \alpha$ is legitimate, the states $\left|\pm X\right\rangle$ are eigenstates of $g_{0}\alpha(\sigma^{+} + \sigma^{-})$ , and thus do not evolve in time. In a full quantum description, however, the $\left|\pm X\right\rangle$ states also evolve in time according to a rather complex dynamics. The latter can be significantly simplified for $\alpha \gg 1$ , where it is possible to write the time evolution of $\left|\pm X\right\rangle$ under $H_{JC}$ as $^{34}$

$$
U _ {\mathrm{JC}} (t) | \pm X \rangle | \alpha \rangle \simeq \frac {1}{\sqrt {2}} \left(e ^ {\mp i g _ {0} t / 2 \alpha} | e \rangle \pm | g \rangle\right) | \Phi_ {\pm} (t) \rangle , \tag {S14}
$$

with the phonon states

$$
| \Phi_ {\pm} (t) \rangle \equiv e ^ {- | \alpha | ^ {2} / 2} \sum_ {n} \frac {\alpha^ {n}}{\sqrt {n !}} e ^ {\mp i g _ {0} t \sqrt {n}} | n \rangle = \sum_ {n} c _ {n} e ^ {\mp i g _ {0} t \sqrt {n}} | n \rangle . \tag {S15}
$$

Eq. (S14) can be used for any finite time t, and it holds in the sense that the difference between the left and the right hand sides is a state vector whose norm vanishes in the limit $\alpha \rightarrow \infty$ .

As we have seen in Eq. (S11), for large $\alpha$ and $t \ll \alpha^2 / g_0$ , it is possible to approximate the (nonlinear) term $\sqrt{n}$ by a second-order series expansion. When $t \ll \alpha / g_0$ , the term quadratic in $n$ can also be neglected, and we can write

$$
\left| \Phi_ {\pm} (t) \right\rangle \simeq e ^ {\mp i g _ {0} \alpha t / 2} \left| \alpha e ^ {\mp i g _ {0} t / 2 \alpha} \right\rangle , \tag {S16}
$$

which shows that for an atom in $|\pm X\rangle$ , the evolution of the phonon state is simply a rotation in phase-space with angular velocity $\mp |g_0 / 2\alpha|$ (see Fig. 1 of the main text). Note that the cat and revival times, $t_C$ and $t_R$ , can be directly inferred by setting $g_0t / 2\alpha = \pi /2$ or $\pi$ , respectively. However, when $t\sim \alpha /g_0$ , the term quadratic in $n$ appearing in Eq. (S11) cannot be neglected,

because the Poissonian distribution of $c_{n}$ results in a standard deviation of $(n - \alpha^{2})$ of order $\alpha$ . This implies that for such timescales the nonlinearity of $\sqrt{n}$ plays a significant role in the phase evolution of Eq. (S15), which results in $|\Phi_{\pm}(t)\rangle$ becoming distorted (and squeezed) coherent states at the cat time $t_{C}$ .

From Eq. (S14) it is straightforward to derive the system evolution for any qubit initial state $|\psi_{q}\rangle = (c_{+}| + X\rangle + c_{-}| - X\rangle)$ . In particular, for $t_{C} = \pi\alpha / g_{0}$ we obtain (cfr. Eq. (S13))

$$
U _ {\mathrm{JC}} (t _ {C}) | \psi_ {q} \rangle | \alpha \rangle \simeq \frac {1}{\sqrt {2}} \left(| g \rangle - i | e \rangle\right) \left(c _ {+} | \Phi_ {+} (t _ {C}) \rangle + c _ {-} | \Phi_ {-} (t _ {C}) \rangle\right). \tag {S17}
$$

From this expression it is easy to see that the qubit state at $t = t_{C}$ is always $|-Y\rangle = (|g\rangle - i|e\rangle)/\sqrt{2}$ , irrespectively of its initial state $|\psi_{q}\rangle$ . The latter, in fact, gets fully mapped into the phonon state that takes the form of a superposition of $|\Phi_{\pm}(t)\rangle$ with relative complex amplitudes equal to $c_{\pm}$ .

If for simplicity we consider the approximation Eq. (S16) and assume $\alpha^{2}$ to be an even integer, we obtain (omitting a global phase)

$$
U _ {\mathrm{JC}} (t _ {C}) | \psi_ {q} \rangle | \alpha \rangle \simeq \frac {1}{\sqrt {2}} \left(| g \rangle - i | e \rangle\right) (c _ {+} | - i \alpha \rangle + c _ {-} | i \alpha \rangle). \tag {S18}
$$

While for $|\psi_{q}\rangle = |\pm X\rangle$ the field is simply in a coherent state, for $|\psi_{q}\rangle$ on the YZ-plane ( $c_{+} = 1/\sqrt{2}$ , $c_{-} = e^{i\vartheta}/\sqrt{2}$ ) we obtain for the field the cat state ( $|-i\alpha\rangle + e^{i\vartheta}|i\alpha\rangle$ ). This shows how the phase of the cat state can be controlled through the qubit initial state.

To demonstrate that the qubit state at $t_{C}$ is in fact $|-Y\rangle$ , regardless of the initial state of the qubit, we perform qubit state tomography during the collapse and revival measurement shown in Figure 2b. We repeat this measurement for initial qubit states $(|g\rangle + e^{i\varphi}|e\rangle)/\sqrt{2}$ , varying $\varphi$ . The

a   
![](images/d5e3122a8a0df969b1da7e3533b6e1ea0e1715905fa4e7f05165d6f7c2731780.jpg)

<details>
<summary>heatmap</summary>

| initial qubit state | 2.9 | 7.0 |
| ------------------- | --- | --- |
| -Y                  | .5  | .5  |
| -X                  | .5  | .5  |
| Y                   | .5  | .5  |
| X                   | .5  | .5  |
| -Y                  | .5  | .5  |
</details>

![](images/a46b5490bae47dc1b6fe52e50eb62a482123c3f8a99115ce81fbdf820e90ec0e.jpg)

<details>
<summary>heatmap</summary>

| x    | y    | arctan(⟨Y⟩/⟨X⟩) |
| ---- | ---- | --------------- |
| 2.9  | π    | -π              |
| 7.0  | π    | -π              |
</details>

b   
![](images/342a52ed0b28750e2b9f9b72732ace1599f6121c95a7ca8b7035c6af455c5daa.jpg)

<details>
<summary>heatmap</summary>

| interaction time (μs) | initial qubit state | ⟨Z⟩    |
| --------------------- | ------------------- | ------ |
| 2.9                   | -Y                  | ~0.5   |
| 2.9                   | X                   | ~0.5   |
| 2.9                   | Y                   | ~0.5   |
| 7.0                   | -Y                  | ~0.5   |
| 7.0                   | X                   | ~0.5   |
| 7.0                   | Y                   | ~0.5   |
</details>

![](images/f359f2fba287fd869d4dc024acb052a7c935d77a82c13036ae553b7d1d3d4a62.jpg)

<details>
<summary>heatmap</summary>

| interaction time (μs) | arctan(⟨Y⟩/⟨X⟩) |
| --------------------- | ---------------- |
| 2.9                   | -π               |
| 7.0                   | -π               |
</details>

Figure S1: Collapse and revival for different initial qubit phase. (a) Qubit state tomography during collapse and revival when starting with the qubit in $(|g\rangle + e^{i\varphi}|e\rangle)/\sqrt{2}$ , sweeping $\varphi$ on the y-axis. Left: $\langle Z\rangle$ . When the qubit starts in $|\pm X\rangle$ , no collapse and revival is observed. On the other hand, when the qubit starts in $|\pm Y\rangle$ , it is observed. Right: Angle between $\langle Y\rangle$ and $\langle X\rangle$ . At $t = 2.9 \mu s$ , the qubit state is $|-Y\rangle$ for all $\varphi$ . At $t = 7 \mu s$ , we observe a revival of the qubit state, best visible in $\langle Z\rangle$ . (b) Corresponding simulations.

resulting expectation values $\langle Z\rangle$ , as well as the resulting qubit phase $\arctan[\langle Y\rangle/\langle X\rangle]$ , are shown in Fig.S1a, from which we make two observations: First, we see collapse and revival of the qubit population for different initial qubit states, except when the qubit is initially prepared in $|\pm X\rangle$ , where the qubit population is stationary. Second, at $t = t_{c} = 2.9 \mu s$ , where we expect a cat state to form in the phonon mode, the qubit state is found to be in $|-Y\rangle$ (blue), regardless of $\varphi$ . Fig.S1b shows the simulated dynamics of the experiment with the same initial state size and disregarding dissipation, which exhibits good agreement with the measured data.

# B Lattice displacement and effective mass

Analogously to the electromagnetic field in an optical cavity, the strain field in our device can be described by a Laguerre-Gaussian (LG) mode. Considering a beam waist of $w_{0} = 27 \mu m$ (radius for 1/e of the amplitude) and a wavelength of $\lambda = 1.7 \mu m$ , we obtain a Rayleigh length $z_{R} = \pi w_{0}^{2}/\lambda \approx 1.4 \, mm$ , which is about three times larger than the length $L = 435 \mu m$ of the mode itself. For this reason, we approximate the transverse profile of the LG mode as a constant function, which will considerably simplify the following calculations. This function takes the form

$$
L G _ {p l} (r, \phi) = \sqrt {\frac {2 p !}{\pi (p + | l |) !}} \left(\frac {r \sqrt {2}}{w _ {0}}\right) ^ {| l |} e ^ {- (r / w _ {0}) ^ {2}} L _ {p} ^ {| l |} \left(\frac {2 r ^ {2}}{w _ {0} ^ {2}}\right) e ^ {- i l \phi}, \tag {S19}
$$

where the integer $p \geq 0$ is the radial index, $l \in Z$ is the azimuthal index, and $L_{p}^{l}$ are the generalized Laguerre polynomials. These modes are normalized so that $\int_{0}^{2\pi} d\phi \int_{0}^{\infty} dr r |LG_{pl}(r, \phi)|^{2} = w_{0}^{2}$ .

The strain field can now be written as

$$
s _ {p l m} (r, \phi , z) = S L G _ {p l} (r, \phi) \sin \left(\frac {m \pi}{L} z\right), \tag {S20}
$$

where S is a normalization constant, and $m\pi/L = k = 2\pi/\lambda$ , with the integer $m \geq 1$ being the longitudinal index. The normalization is such that

$$
U = \frac {c _ {3 3}}{2} \int_ {V} d V \left| s _ {p l m} (r, \phi , z) \right| ^ {2} \tag {S21}
$$

$$
= \frac {c _ {3 3}}{2} S ^ {2} w _ {0} ^ {2} \frac {L}{2} \tag {S22}
$$

is the potential energy stored in the deformation of the crystal, with $c_{33}$ the material stiffness tensor component. In this sense, $S$ depends on the system's state. If we set $U$ to be the energy of a phonon $\hbar \omega_{p}$ , Eq. (S22) defines $S = S_{0} \equiv \sqrt{4\hbar\omega_{p}/(Lw_{0}^{2}c_{33})}$ as the strain per phonon.

To find the effective mass of the mechanical mode of interest, it is convenient to compare it to a one-dimensional harmonic oscillator of mass $M_{eff}$ and potential energy

$$
U = \frac {1}{2} M _ {\text {eff}} \omega_ {p} ^ {2} x _ {\text {eff}} ^ {2}, \tag {S23}
$$

where $x_{eff}$ is the effective oscillation amplitude. This expression alone is not sufficient to define $M_{eff}$ and $x_{eff}$ unambiguously, as it constraints only their product. Equating Eq. (S23) and Eq. (S22), we get

$$
M _ {\text {eff}} = \frac {c _ {3 3}}{2} \frac {S ^ {2} w _ {0} ^ {2}}{2 \omega_ {p} ^ {2} x _ {\text {eff}} ^ {2}} L \tag {S24}
$$

$$
= \left(\frac {S ^ {2} L ^ {2}}{2 \pi^ {3} m ^ {2} x _ {\text {eff}} ^ {2}}\right) \rho \pi w _ {0} ^ {2} L \tag {S25}
$$

where in going to the last line we used the relations $\omega_{p}=2\pi c/\lambda,\lambda=2L/m$ and $c=\sqrt{c_{33}/\rho}$ .

The term in parenthesis of Eq. (S25) is a rescaling factor of the mass $M_{0} \equiv \rho\pi w_{0}^{2}L = 4.0 \mu g$ . Computing it requires an estimate of $x_{eff}$ . Note here that $M_{0}$ is the mass of a cylinder with radius $w_{0}$ and the length L and density $\rho$ , which is the naive approximation for the volume of the phonon mode. To obtain an estimate of $x_{eff}$ , note that Eq. (S20) describes the strain in the material, which we assumed to be mostly in the longitudinal (i.e. z) direction. This is defined as $s \equiv s_{zz} = \partial u_{z}(x,y,z)/\partial z$ , where $u_{z}(x,y,z)$ is the z component of the displacement field $\vec{u}(x,y,z) = (0,0,u_{z}(x,y,z))$ . Therefore, $u_{z}$ can be found by a straightforward integration of Eq. (S20) to be

$$
u _ {z} (r, \phi , z) = - \frac {L}{m \pi} S L G _ {p l} (r, \phi) \cos \left(\frac {m \pi z}{L}\right). \tag {S26}
$$

The maximum displacement can be easily found for $l = 0$ , for which the maximum of

Eq. (S19) is $LG_{p0}(0,0)=\sqrt{2/\pi}\approx0.8$ . This gives $u_{z}^{\max}=\frac{L}{m\pi}\sqrt{\frac{2}{\pi}}S$ . On the other hand, a root-mean-square (RMS) value can be found by defining an integration area $A_{k}=\pi R^{2}$ , such that for $R=2w_{0}$ and the values p=0,1 of typical interest we find

$$
\sqrt {A _ {k} ^ {- 1} \int_ {0} ^ {2 \pi} d \phi \int_ {0} ^ {R} d r r \left| L G _ {p 0} (r , \phi) \right| ^ {2}} \approx 0. 2 8. \tag {S27}
$$

Together with the factor $1 / \sqrt{2}$ coming from the RMS of the cosine, we have $u_{z}^{\mathrm{RMS}} = \frac{L}{m\pi}\frac{0.28}{\sqrt{2}} S$ .

Choosing $x_{\mathrm{eff}} = u_z^{\mathrm{max}}$ results in $M_{\mathrm{eff}} = M_0 / 4 \approx 1.0 \, \mu \mathrm{g}$ , while for $x_{\mathrm{eff}} = u_z^{\mathrm{RMS}}$ , we obtain $M_{\mathrm{eff}} \approx 4.1M_0 \approx 16.2 \, \mu \mathrm{g}$ .

To find the displacement of the lattice for a coherent state, we use Eq. (S23) and the relation $U = \hbar \omega_{p}(\alpha^{2} + 1 / 2)$ to write

$$
x _ {\text {eff}} (\alpha) = \sqrt {1 + 2 \alpha^ {2}} \sqrt {\frac {\hbar}{M _ {\text {eff}} \omega_ {p}}} \tag {S28}
$$

$$
= \sqrt {2 (1 + 2 \alpha^ {2})} x _ {\mathrm{ZPF}}, \tag {S29}
$$

where $x_{ZPF} \equiv \sqrt{\hbar/2M_{eff}\omega_{p}}$ is the zero point fluctuation. Here, $x_{eff}^{RMS}$ or $x_{eff}^{max}$ is obtained by choosing the corresponding $M_{eff}$ . For example, a cat state with $\alpha = 1.61$ would correspond to a mass of $M_{eff} = 1.0 \mu g$ delocalized over $2x_{\mathrm{eff}}^{\mathrm{max}}(1.61) = 8.4 \cdot 10^{-18} \, \mathrm{m}$ , or a mass of $M_{eff} = 16.2 \mu g$ delocalized over $2x_{\mathrm{eff}}^{\mathrm{RMS}}(1.61) = 2.1 \cdot 10^{-18} \, \mathrm{m}$ .

![](images/f0cd7883306b1a4b2a77fdff8f7a7021b5f10dc2cfb82db2572335b8225a0987.jpg)

<details>
<summary>line</summary>

| 2nd π/2 pulse phase (rad) | qubit population |
| ------------------------- | ---------------- |
| 0                         | 0.75             |
| 1/2                       | -1.0             |
| 2π                        | 0.75             |
| 3π/4                      | -1.0             |
| 4π                        | 0.75             |
</details>

Figure S2: Parity measurement calibration curve. The two dashed black lines indicate values of maxima and minima of the parity values for a Fock $|0\rangle$ state. These two values are then used as the $\pm1$ parity values to normalize subsequent Wigner tomography measurements.

# C Parity measurement calibration

To perform the phonon parity measurement needed for Wigner tomography, we follow the procedure developed in earlier work $^{20}$ . This consists of a Ramsey-type qubit experiment, where we let the qubit dispersively interact with the phonon mode for a carefully calibrated interaction time. After this interaction, the phonon state parity has mapped onto the qubit state due to the phonon-state-dependent dispersive frequency shift of the qubit. Ideally, even (odd) phonon numbers map to the $|e\rangle(|g\rangle)$ state of the qubit. Measuring the qubit state then corresponds to a measurement of the phonon parity.

However, because of finite qubit coherence, the contrast of the parity measurement is reduced. For example, the parity measurement of an undisplaced Fock state $|0\rangle$ results in a probability $P(e|0) < 1$ , which translates to a parity value $\langle \hat{\Pi} \rangle < 1$ . To correct for this measurement error,

we normalize the measured parity values using the contrast of the Ramsey parity measurement with the phonon in $|0\rangle$ , obtained by sweeping the phase of the second $\pi/2$ pulse. An example for an interaction time of $6.7\ \mu s$ can be seen in Fig.S2. $P(e|0)$ oscillates between the maximum and minimum values possible at the given interaction time. Fitting this oscillation to a cosine model, we extract the oscillation amplitude and use it to normalize all parity measurements so that the maximum and minimum readout values correspond to a parity of +1 and -1, respectively.

# D Resonant phonon number measurement

Due to limitations in our setup, the magnitude $|\beta|$ of the coherent state we generate may not scale linearly with the drive amplitude A. We find that $|\beta|$ saturates for larger A, when the drive strength on the qubit is close to the detuning between qubit and phonon. Therefore, we calibrate the generated $|\beta|$ by measuring the phonon Fock state populations for different A, using the method described in earlier works $^{19}$ . Figure S3a is an example time trace showing the qubit population as it interacts resonantly with the phonon initialized in a coherent state. We first perform a fit of this data to extract the phonon Fock state populations and then fit these populations with a Poissonian distribution (Fig. S3b). In Fig.S3c, we show the extracted $|\beta|$ as a function of A for three different pulse shapes, which we fit to a phenomenological function of the form $|\beta| = C(\exp(A/B) - 1)$ . The pulse shapes are all Gaussian-square pulses, and the three examples have different lengths and rise times, with the amplitude A as an overall scaling factor. When performing Wigner tomography, we use a linear grid of drive amplitudes for convenience. In post-processing, we then use the fitted dependence of $|\beta|$ on A to plot the parity values as a function of the actual coherent displacement amplitudes.

![](images/60107c5613efe53184545e7298e2d83968b09a9a82d714cdd1d700f86234e783.jpg)

<details>
<summary>line</summary>

| time (μs) | qubit population |
| --------- | ---------------- |
| 0.0       | 0.0              |
| 0.5       | 0.75             |
| 1.0       | 0.45             |
| 2.0       | 0.48             |
| 3.0       | 0.49             |
| 4.0       | 0.47             |
| 5.0       | 0.58             |
| 6.0       | 0.35             |
| 7.0       | 0.52             |
| 8.0       | 0.38             |
| 9.0       | 0.49             |
| 10.0      | 0.47             |
| 11.0      | 0.46             |
| 12.0      | 0.48             |
| 13.0      | 0.45             |
| 14.0      | 0.43             |
| 15.0      | 0.42             |
</details>

![](images/8c14cf71f22ccade79f204099aaef45aa5f88f731f612a64d3bf782e2e375f67.jpg)

<details>
<summary>bar</summary>

| phonon Fock state n | population |
| ------------------- | ---------- |
| 0                   | 0.05       |
| 1                   | 0.15       |
| 2                   | 0.22       |
| 3                   | 0.22       |
| 4                   | 0.17       |
| 5                   | 0.10       |
| 6                   | 0.05       |
| 7                   | 0.03       |
| 8                   | 0.01       |
| 9                   | 0.01       |
| 10                  | 0.00       |
| 11                  | 0.00       |
| 12                  | 0.00       |
| 13                  | 0.00       |
| 14                  | 0.00       |
| 15                  | 0.00       |
| 16                  | 0.01       |
| 17                  | 0.00       |
| 18                  | 0.00       |
| 19                  | 0.00       |
| 20                  | 0.01       |
</details>

![](images/96d51afb379ed09dacadf0581d05c372894feb4801b2dd2b37f1031f6814c73b.jpg)

<details>
<summary>line</summary>

| amplitude A (a.u.) | pulse shape 1 | pulse shape 2 | pulse shape 3 |
| ------------------ | ------------- | ------------- | ------------- |
| 0.0                | 0.0           | 0.0           | 0.0           |
| 0.5                | 2.5           | 1.5           | 0.8           |
| 1.0                | -             | 2.2           | 1.4           |
| 1.5                | -             | 2.6           | 1.8           |
| 2.0                | -             | -             | 2.1           |
</details>

Figure S3: Phonon displacement drive calibration. (a) An example of the qubit dynamics during resonant interaction with the phonon mode. Orange dots are experiment data, and the blue curve is the fitting result. (b) Distribution of Fock state populations extracted from the measurement of a. The black boxes are the Poisson distribution of the fitted coherent state. (c) Relation between the drive amplitude and generated coherent state $|\beta|$ for three different pulse shapes.

# E Post-processing and fitting of Wigner functions

In Fig.S4, we show an overview of how the measured Wigner functions are post-processed and fitted to different cat state models. We start with the raw data in the first row, which is plotted as a function of the complex drive amplitude. We then normalize the measured parity values using the procedure described in section C and perform a nonlinear scaling of the axes as described in section D. The resulting post-processed data is shown in the second row of Fig.S4, where the Wigner function is now plotted as a function of the actual complex displacement amplitude. Rows three to five then show the result of maximum-likelihood reconstructions of the physical states from the post-processed data, as well as fits to the analytical and CSS cat state models, as described in section F and in the main text.

![](images/8abc371cc2a2b549002b857d9d8f76ee06bfd460970171600f90b6c6d4ba3c6b.jpg)  
Figure S4: Post-processing and analysis of Wigner functions. Each column corresponds to one of the three initial displacement amplitudes A used to prepare the cat states shown in the main text. The axes for the first row are the I and Q quadrature amplitudes of the complex drives used in the Wigner tomography sequence $^{20,35}$ . These are then converted into the real and imaginary parts of the actual displacement amplitude using the procedure in section D.

# F State reconstruction and fidelity

Reconstruction of the density matrix from the measured Wigner function is performed using a Maximum Likelihood Estimate, as described in $^{26}$ . As an example, Fig. S5 shows the absolute value of the displaced reconstructed density matrix of the A=0.35 cat state from the main text. The additional displacement shifts one of the coherent state components to the center of phase space. The two peaks on the diagonal represent the coherent state components. In turn, the features along the coordinate axes indicate interference between the coherent state components, corresponding to fringes of alternating parity in phase space.

We then calculate the fidelity of the reconstructed state with respect to two different target states. Here we define the fidelity between two states $\rho$ and $\sigma$ as

$$
\mathcal {F} (\rho , \sigma) = \operatorname{Tr} \left(\sqrt {\sqrt {\rho} \sigma \sqrt {\rho}}\right). \tag {S30}
$$

The first target state we consider is the phonon state resulting from the JC interaction between the qubit and phonon modes, as described by Eq. S9 (up to rotations). To most accurately model the physical state produced in our system, we fix the initial qubit state (described by $c_{g}$ and $c_{e}$ ) and JC evolution time $t = t_{C}$ of each fitted state to match those of the corresponding experiment. We also allow for a rotation $\theta$ in phase space, which could result from a slight detuning of the phonon drive frequency. This leaves a target state of the form

$$
\rho (t _ {C}) = R (\theta) \rho^ {\prime} (t _ {C}) R ^ {\dagger} (\theta), \tag {S31}
$$

![](images/d2a55c8bb53673ff111f244372ca1855c673877bd90fce0d081a9aa96d697cc5.jpg)

<details>
<summary>bar</summary>

| x-axis label | y-axis label | value  |
| ------------ | ------------ | ------ |
| 0>           | 0>           | 0.08   |
| 0>           | 5>           | 0.07   |
| 0>           | 10>          | 0.06   |
| 0>           | 15>          | 0.05   |
| 0>           | 5>           | 0.04   |
| 0>           | 10>          | 0.03   |
| 0>           | 15>          | 0.02   |
| 0>           | 5>           | 0.01   |
| 0>           | 10>          | 0.00   |
| 5>           | 0>           | 0.08   |
| 5>           | 5>           | 0.07   |
| 5>           | 10>          | 0.06   |
| 5>           | 15>          | 0.05   |
| 5>           | 5>           | 0.04   |
| 5>           | 10>          | 0.03   |
| 5>           | 15>          | 0.02   |
| 5>           | 5>           | 0.01   |
| 5>           | 10>          | 0.00   |
| 10>          | 0>           | 0.08   |
| 10>          | 5>           | 0.07   |
| 10>          | 10>          | 0.06   |
| 10>          | 15>          | 0.05   |
| 10>          | 5>           | 0.04   |
| 10>          | 10>          | 0.03   |
| 10>          | 15>          | 0.02   |
| 10>          | 5>           | 0.01   |
| 15>          | 0>           | 0.08   |
| 15>          | 5>           | 0.07   |
| 15>          | 10>          | 0.06   |
| 15>          | 15>          | 0.05   |
| 15>          | 5>           | 0.04   |
| 15>          | 10>          | 0.03   |
| 15>          | 15>          | 0.02   |
| 15>          | 5>           | 0.01   |
| 15>          | 10>          | 0.00   |
| 20>          | 0>           | 0.08   |
| 20>          | 5>           | 0.07   |
| 20>          | 10>          | 0.06   |
| 20>          | 15>          | 0.05   |
| 20>          | 5>           | 0.04   |
| 20>          | 10>          | 0.03   |
| 20>          | 15>          | 0.02   |
| 20>          | 5>           | 0.01   |
| 25>          | 0>           | 0.08   |
| 25>          | 5>           | 0.07   |
| 25>          | 10>          | 0.06   |
| 25>          | 15>          | 0.05   |
| 25>          | 5>           | 0.04   |
| 25>          | 10>          | 0.03   |
| 25>          | 15>          | 0.02   |
| 25>          | 5>           | 0.01   |
| 25>          | 10>          | 0.00   |
| ... (additional bars) are not provided in the image; they are estimated based on the visual representation of the 'values' in the table.
</details>

Figure S5: Reconstructed density matrix for the cat state with A=0.35. The height and color of the bars both indicate the absolute value of the corresponding density matrix elements.

where $\rho'(t_C)$ is given by Eq. S9 with $t = t_C$ and $R(\theta) = \exp(-i\theta a^\dagger a)$ . The parameters $\alpha$ and $\theta$ are optimized numerically in order to maximize the fitted state's fidelity to the reconstructed experimental state. We quote the fitted $\alpha$ values as $\alpha_{\text{fit}}$ in the main text.

The second target state is a coherent state superposition of the form

$$
| \mathbf {C} \rangle = \mathcal {N} \left(| \alpha_ {1} \rangle + e ^ {i \vartheta} | \alpha_ {2} \rangle\right), \tag {S32}
$$

where N is an appropriate normalization constant. Again, the parameters $|\alpha_{1,2}\rangle$ and $\vartheta$ are optimized numerically in order to maximize the fidelity. From these, we obtain the CSS cat state size as half of the distance between the two coherent states, $D = |\alpha_{1} - \alpha_{2}|^{36}$ .

To determine how sensitive the fidelities resulting from our fitting procedure are to different values of $\alpha_{fit}$ and D, we repeat the fits, but rather than optimizing $\alpha_{fit}$ or D, we keep them fixed in the fit and sweep across a range of values. Fig. S6 shows an example for $\alpha_{fit}$ , from which we then calculate the alpha value where the fidelity is 1% lower than the highest fidelity and use this range as the error bar in Fig. 3b.

![](images/59209e40b462b65c45340fe8639541885ed37cbf944762bf3afabe466c233326.jpg)

<details>
<summary>line</summary>

| α_fit | Fidelity |
|-------|----------|
| 1.0   | 0.64     |
| 1.5   | 0.75     |
| 2.0   | 0.61     |
</details>

Figure S6: Fidelity sensitivity to $\alpha_{fit}$ . Fitted fidelity of the analytical state to the cat state generated with A = 0.35. In the fit, we constrain the initial coherent state size $\alpha_{fit}$ . The two purple dashed lines are the positions where the fitted fidelity is 1% lower than the highest fidelity.

# G Cat state decoherence

We consider here a bosonic mode subject to relaxation at a rate $\kappa$ . For an initial coherent state $|\alpha\rangle$ , this relaxation process results in the time-dependent state $|\alpha(t)\rangle = |\alpha e^{-\kappa t/2}\rangle$ . In Ref. 37,38, it is shown that the Wigner function of a CSS state evolves in time as

$$
W (\beta) = \frac {1}{\pi (1 + e ^ {- | \alpha | ^ {2}})} \left(e ^ {- 2 | \beta - \alpha \epsilon | ^ {2}} + e ^ {- 2 | \beta + \alpha \epsilon | ^ {2}} + 2 e ^ {- 2 | \beta | ^ {2}} e ^ {- 2 | \alpha | ^ {2} (1 - \epsilon^ {2})} \cos (4 \mathrm{Im} (\beta) \alpha \epsilon)\right), \tag {S33}
$$

where $\alpha$ is the amplitude of the coherent states in the superposition, and $\epsilon = e^{-\kappa t/2}$ parameterizes the relaxation process.

The two exponential terms $e^{-2|\beta\pm\alpha\epsilon|^{2}}$ represent the two coherent state components relaxing towards the origin with a timescale $2/\kappa = 2T_{1}^{ph}$ . On the other hand, the amplitude of the cosine term, representing the coherence of the quantum superposition, decays much more rapidly as it includes the factor

$$
\xi \equiv e ^ {- 2 | \alpha | ^ {2} (1 - \epsilon^ {2})}. \tag {S34}
$$

To see this effect, it is necessary to access a quantity which is able to reflect this decay of coherence. Here, we show that the Wigner function negativity can be taken as a faithful quantifier for the coherence. The negativity is defined as $^{28}$

$$
\delta \equiv \int d ^ {2} \beta \left(| W (\beta) | - W (\beta)\right), \tag {S35}
$$

and it is an indicator of non-classicality that has been related to a number of quantum information tasks and entanglement measures ${}^{39-42}$ .

For the CSS state in Eq. (S33), the negativity can be computed analytically for large $\alpha$ by

taking into account that the overlap between the three exponential terms is negligible. These correspond to the two coherent states and to the interference fringes in between them, which for large $\alpha$ are well separated in phase space. This allows us to approximate $|W(\beta)|$ by the sum of the moduli of each exponential term, giving (using $\int d^{2}\beta W(\beta)=1$ )

$$
\begin{array}{l} \delta_ {\mathrm{cat}} (\epsilon) \approx \frac {1}{\pi (1 + e ^ {- | \alpha | ^ {2}})} \int d ^ {2} \beta \left(| e ^ {- 2 | \beta - \alpha \epsilon | ^ {2}} | + | e ^ {- 2 | \beta + \alpha \epsilon | ^ {2}} | + | 2 e ^ {- 2 | \beta | ^ {2}} e ^ {- 2 | \alpha | ^ {2} (1 - \epsilon^ {2})} \cos (4 \mathbf {I m} (\beta) \alpha \epsilon) |\right) - 1 \\ = \frac {1}{2} (1 + \tanh (\alpha^ {2})) + I _ {\text {decay}} (\epsilon) - 1. \tag {S36} \\ \end{array}
$$

Here we introduced the term

$$
I _ {\text { decay }} (\epsilon) = \frac {\sqrt {2} e ^ {2 \alpha^ {2} \epsilon^ {2}}}{\sqrt {\pi} (1 + e ^ {2 | \alpha | ^ {2}})} \int d \beta_ {i} e ^ {- 2 \beta_ {i} ^ {2}} | \cos (4 \beta_ {i} \alpha \epsilon) |, \tag {S37}
$$

with $\beta_{i}$ as the imaginary part of $\beta$ , which is responsible for the negativity decay. The time constant for this process can be estimated by considering the lower bound $\cos(x)^{2} \leq |\cos(x)|$ , which allows us to approximate the integral in $I_{decay}$ with

$$
\int d \beta_ {i} e ^ {- 2 \beta_ {i} ^ {2}} \cos (4 \beta_ {i} \alpha \epsilon) ^ {2} = \sqrt {\frac {\pi}{8}} \left(1 + e ^ {- 8 \alpha^ {2} \epsilon^ {2}}\right). \tag {S38}
$$

For large $\alpha$ and short times, this expression can be used to find

$$
I _ {\mathrm{decay}} (\epsilon) \approx \frac {1}{4} (1 + \tanh (\alpha^ {2})) e ^ {- 2 t \alpha^ {2} \kappa}, \tag {S39}
$$

which shows that the cat state negativity decays at a rate $2\alpha^2\kappa$ , corresponding to a time constant $\tau_{\mathrm{cat}} = T_1^{ph} / (2\alpha^2)$ . We note that, by looking at Eq. (S37), it's clear that the same decay time constant should apply to the negativity of a 1D slice of the Wigner function perpendicular to the fringes at $\operatorname{Im}(\beta) = 0$ if $\alpha$ is purely imaginary, which corresponds to the measured data in Figure 4 of the main text.

![](images/c8428f0732079e4a22b575160f555329f51f50c58217cd8d4ae3ec3ecb4839b1.jpg)

<details>
<summary>line</summary>

| α    | θ = 0  | θ = ±π/2 | θ = π  | T₁^ph/2α² | simulated decay | measured decay |
|------|--------|----------|--------|-----------|-----------------|----------------|
| 0.5  | ~5     | ~8       | ~23    | ~35       | -               | -              |
| 1.0  | ~18    | ~21      | ~24    | ~26       | ~26             | ~12            |
| 1.5  | ~19    | ~20      | ~23    | ~18       | ~10             | ~10            |
| 2.0  | ~17    | ~18      | ~20    | ~14       | -               | -              |
| 3.0  | ~8     | ~10      | ~10    | ~8        | -               | -              |
| 4.0  | ~5     | ~6       | ~6     | ~5        | -               | -              |
| 5.0  | ~3     | ~4       | ~4     | ~3        | -               | -              |
</details>

Figure S7: Cat decay rate predictions and measurements. Wigner negativity decay rate for CSS with different phases $\vartheta$ in Eq. (2) (blue, orange, green lines), compared to the analytical decay rate obtained in the limit of large $\alpha$ (red line). Purple crosses are the decay times obtained from a simulation of the full experiment as described in section G.2, black squares are the decay times of the measured states shown in Fig.4c of the main text.

In deriving Eq. (S39), we assumed the coherent state amplitude $\alpha$ to be large. In order to check the validity range of this approximation, we compare $\tau_{cat}$ to the decay rate obtained by numerically computing the negativity of Eq. (S33) for varying $\alpha$ . The results are shown in Fig. S7, where we can see good agreement for $\alpha \gtrsim 2$ . However, for smaller values of $\alpha$ we see a significant discrepancy, as we expect from the fact that the approximation used to obtain Eq. (S36) becomes inaccurate. Moreover, when $\alpha$ is small, we observe a decay speed that is also dependent on the phase $\vartheta$ of the coherent states superposition Eq. (2).

# G.1 Decoherence Measurements

In order to characterize the decay of quantum features in the created cat states, we extract the state negativity (see Eq. (S35)) after different wait times $\tau$ between state creation and tomography. To illustrate the decaying state in phase space, we perform full Wigner tomography on the cat state of size D = 1.43 at the delay times $\tau = 0, 10, 40 \mu s$ . The resulting data is presented in Fig. S8a. While the negative parity regions of the interference fringes are decaying at a fast rate of $\approx 10 \mu s$ , the decay of the coherent state components towards the vacuum state happen at the slower timescale of $2T_{1}^{ph}$ , as shown in the previous section.

Extracting the negativity of a state with high enough accuracy requires increasing the number of averages by a factor of 5 compared to the presented states in Fig. 2 and 3 of the main text. Additionally, the number of delay times $\tau$ between state creation and tomography needs to be large enough to extract fitting parameters with reasonable errorbars. In order to fulfill these requirements efficiently, we choose to measure the decaying states along a 1D crosscut perpendicular to the interference fringes, around the location of largest contrast between positive and negative parity fringes (see Fig. S8a).

We plot these crosscuts of the Wigner function for all three cat state sizes and $\tau$ ranging from from 0 to 40 $\mu$ s in Fig. S8b. For each crosscut, the negativity is calculated according to Eq. (S35). One can clearly observe the faster decay rate with increasing state size, confirmed by the analysis presented in Fig. 4 of the main text.

![](images/865cb4ecc89f8930d69e5449b78caf77d3e1fb3d55583c0d23616c5ad5450a55.jpg)

![](images/20b6ab98d570536374a8e8698d926cec7bdfd604919b8b194bf057f07434821e.jpg)

<details>
<summary>heatmap</summary>

| Re(β) | Wait time τ (μs) |
|-------|------------------|
| -1    | 0                |
| 0     | 10               |
| 1     | 40               |
</details>

![](images/51d077ab695b5262a966f16e361e24c98bf258530041ef5193b2678577b7a779.jpg)

<details>
<summary>heatmap</summary>

| Re(β) Range | Density |
|-------------|---------|
| -1 to 0     | High    |
| 0 to 1      | Low     |
</details>

![](images/8bc673e4ac31a71931a5e54195b3c05951c83ec7e4f703e6f151384e879009e1.jpg)

<details>
<summary>heatmap</summary>

| Re(β) | parity |
|-------|--------|
| -1    | -1     |
| 0     | 1      |
| 1     | -1     |
</details>

Figure S8: Wigner negativity decay measurements. (a) Wigner tomographies of the cat state with size D = 1.43 for three different wait times. (b) 1D crosscuts vs. wait time for all three measured cat states with size D as indicated in the plot titles. Black dashed lines in (a) and (b) indicate corresponding crosscuts.

# G.2 Decoherence simulations for analytical states

As we have seen from Fig. S7, for small values of $\alpha$ we expect a discrepancy between the theoretical and the measured $\tau_{cat}$ . For this reason, in order to obtain a more accurate prediction of the cat decay time, we perform a Master equation simulation of the full experiment and compare it to the decay of the CSS states shown in Fig. S7.

In order to reproduce the state decay as accurately as possible, we first run a full master equation simulation of the cat state preparation, starting from an initial coherent state with same amplitude as in the experiment. After having verified that the simulated states agree well with the corresponding measured cat states at $t_C$ , the simulated states are evolved freely for a variable time $\tau_{\text{sim}}$ under Eq. (1) with the qubit far detuned from the phonon mode. Finally, the states' Wigner negativity is extracted for each $\tau_{\text{sim}}$ following the same procedure as described in section G1. These simulated negativities are fitted to the same decaying exponential model that we also apply to the measured data, allowing us to extract the desired time constant.

The results obtained for three cat states of the same size as the one we measured can be seen as purple crosses in Fig.S7. The simulated decay rates are 24.68, 12.34, and $10.52\mu \mathrm{s}$ , for the state sizes $D = 1.09, 1.43$ , and 1.61 respectively. The data shown in Fig. 4c of the main text is also included for comparison. The deviation of the measured value for the smallest state size from the simulated value can be attributed to a slight positive background offset of the measured Wigner function, which results in the negativity values being set to zero in the tail of the decay.

1. Schrödinger, E. Die gegenwärtige Situation in der Quantenmechanik. Naturwissenschaften 23, 807–812 (1935). URL https://doi.org/10.1007/BF01491891.   
2. Bassi, A., Lochan, K., Satin, S., Singh, T. P. & Ulbricht, H. Models of wave-function collapse, underlying theories, and experimental tests. Rev. Mod. Phys. 85, 471–527 (2013). URL https://link.aps.org/doi/10.1103/RevModPhys.85.471.   
3. Pezzè, L. & Smerzi, A. Quantum theory of phase estimation. In Tino, G. M. & Kasevich, M. A. (eds.) Atom Interferometry, Proceedings of the International School of Physics “Enrico Fermi”, Varenna (IOS Press, Amsterdam, 2014). URL https://ebooks.iospress.nl/volumearticle/38107.   
4. Munro, W. J., Nemoto, K., Milburn, G. J. & Braunstein, S. L. Weak-force detection with superposed coherent states. Physical Review A 66, 023819 (2002). URL https://link.aps.org/doi/10.1103/PhysRevA.66.023819.   
5. Cochrane, P. T., Milburn, G. J. & Munro, W. J. Macroscopically distinct quantum-superposition states as a bosonic code for amplitude damping. Phys. Rev. A 59, 2631–2634 (1999). URL https://link.aps.org/doi/10.1103/PhysRevA.59.2631.   
6. Mirrahimi, M. et al. Dynamically protected cat-qubits: a new paradigm for universal quantum computation. New Journal of Physics 16, 045014 (2014). URL https://doi.org/10.1088%2F1367-2630%2F16%2F4%2F045014.

7. Monroe, C., Meekhof, D. M., King, B. E. & Wineland, D. J. A Schrödinger cat; superposition state of an atom. Science 272, 1131–1136 (1996). URL https://www.science.org/doi/abs/10.1126/science.272.5265.1131.   
8. Lo, H.-Y. et al. Spin-motion entanglement and state diagnosis with squeezed oscillator wavepackets. Nature 521, 336–339 (2015). URL http://www.nature.com/articles/nature14458.   
9. Ourjoumtsev, A., Jeong, H., Tualle-Brouri, R. & Grangier, P. Generation of optical ‘Schrödinger cats’ from photon number states. Nature 448, 784–786 (2007). URL https://www.nature.com/articles/nature06054. Number: 7155 Publisher: Nature Publishing Group.   
10. Huang, K. et al. Optical Synthesis of Large-Amplitude Squeezed Coherent-State Superpositions with Minimal Resources. Physical Review Letters 115, 023602 (2015). URL https://link.aps.org/doi/10.1103/PhysRevLett.115.023602.   
11. Auffeves, A. et al. Entanglement of a Mesoscopic Field with an Atom Induced by Photon Graininess in a Cavity (2003). URL https://link.aps.org/doi/10.1103/PhysRevLett.91.230405.   
12. Deléglise, S. et al. Reconstruction of non-classical cavity field states with snapshots of their decoherence. Nature 455, 510–514 (2008). URL https://doi.org/10.1038/nature07288.

13. Vlastakis, B. et al. Deterministically encoding quantum information using 100-photon Schrodinger cat states. Science 342, 607–610 (2013). URL https://www.science.org/doi/abs/10.1126/science.1243289.   
14. Leibfried, D. et al. Creation of a six-atom ‘Schrödinger cat’ state. Nature 438, 639–642 (2005). URL https://www.nature.com/articles/nature04251. Number: 7068 Publisher: Nature Publishing Group.   
15. Gao, W.-B. et al. Experimental demonstration of a hyper-entangled ten-qubit Schrödinger cat state. Nature Physics 6, 331–335 (2010). URL https://www.nature.com/articles/nphys1603. Number: 5 Publisher: Nature Publishing Group.   
16. Friedman, J. R., Patel, V., Chen, W., Tolpygo, S. K. & Lukens, J. E. Quantum superposition of distinct macroscopic states. Nature 406, 43–46 (2000). URL https://doi.org/10.1038/35017505.   
17. Gerlich, S. et al. Quantum interference of large organic molecules. Nature Communications 2, 263 (2011). URL https://www.nature.com/articles/ncomms1263. Number: 1 Publisher: Nature Publishing Group.   
18. See supplementary materials.   
19. Chu, Y. et al. Creation and control of multi-phonon Fock states in a bulk acoustic-wave resonator. Nature 563, 666–670 (2018). URL https://doi.org/10.1038/s41586-018-0717-7.

20. von Lüpke, U. et al. Parity measurement in the strong dispersive regime of circuit quantum acoustodynamics. Nature Physics 18, 794–799 (2022). URL https://doi.org/10.1038/s41567-022-01591-2.   
21. Bužek, V., Moya-Cessa, H., Knight, P. L. & Phoenix, S. J. D. Schrödinger-cat states in the resonant jaynes-cummings model: Collapse and revival of oscillations of the photon-number distribution. Phys. Rev. A 45, 8190–8203 (1992). URL https://link.aps.org/doi/10.1103/PhysRevA.45.8190.   
22. Cummings, F. W. Stimulated emission of radiation in a single mode. Phys. Rev. 140, A1051–A1056 (1965). URL https://link.aps.org/doi/10.1103/PhysRev.140.A1051.   
23. Rempe, G., Walther, H. & Klein, N. Observation of quantum collapse and revival in a one-atom maser. Phys. Rev. Lett. 58, 353–356 (1987). URL https://link.aps.org/doi/10.1103/PhysRevLett.58.353.   
24. Eberly, J. H., Narozhny, N. B. & Sanchez-Mondragon, J. J. Periodic spontaneous collapse and revival in a simple quantum model. Phys. Rev. Lett. 44, 1323–1326 (1980). URL https://link.aps.org/doi/10.1103/PhysRevLett.44.1323.   
25. Gea-Banacloche, J. Collapse and revival of the state vector in the jaynes-cummings model: An example of state preparation by a quantum apparatus. Phys. Rev. Lett. 65, 3385–3388 (1990). URL https://link.aps.org/doi/10.1103/PhysRevLett.65.3385.

26. Chou, K. S. et al. Deterministic teleportation of a quantum gate between two logical qubits. Nature 561, 368–373 (2018). URL https://doi.org/10.1038/s41586-018-0470-y.   
27. Grimm, A. et al. Stabilization and operation of a Kerr-cat qubit. Nature 584, 205–209 (2020). URL https://www.nature.com/articles/s41586-020-2587-z.   
28. Kenfack, A. & Zyczkowski, K. Negativity of the wigner function as an indicator of non-classicality. Journal of Optics B: Quantum and Semiclassical Optics 6, 396–404 (2004). URL https://doi.org/10.1088/1464-4266/6/10/003.   
29. Schrinski, B. et al. Macroscopic quantum test with bulk acoustic wave resonators (2022). URL https://arxiv.org/abs/2209.06635. ArXiv:2209.06635 [quant-ph].   
30. Ofek, N. et al. Extending the lifetime of a quantum bit with error correction in superconducting circuits. Nature 536, 441–445 (2016). URL https://doi.org/10.1038/nature18949.   
31. Flühmann, C. et al. Encoding a qubit in a trapped-ion mechanical oscillator. Nature 566, 513–517 (2019). URL https://doi.org/10.1038/s41586-019-0960-6.   
32. Joo, J., Munro, W. J. & Spiller, T. P. Quantum metrology with entangled coherent states.
Phys. Rev. Lett. 107, 083601 (2011). URL https://link.aps.org/doi/10.1103/PhysRevLett.107.083601.   
33. Facon, A. et al. A sensitive electrometer based on a rydberg atom in a schrödinger-cat state.
Nature 535, 262–265 (2016). URL https://doi.org/10.1038/nature18327.

34. Gea-Banacloche, J. Atom- and field-state evolution in the jaynes-cummings model for large initial fields. Phys. Rev. A 44, 5913–5931 (1991). URL https://link.aps.org/doi/10.1103/PhysRevA.44.5913.   
35. Chu, Y. & Gröblacher, S. A perspective on hybrid quantum opto- and electromechanical systems. Applied Physics Letters 117 (2020). URL https://aip.scitation.org/doi/10.1063/5.0021088.   
36. Brune, M. et al. Observing the progressive decoherence of the “meter” in a quantum measurement. Physical Review Letters 77, 4887 (1996). URL https://link.aps.org/doi/10.1103/PhysRevLett.77.4887.   
37. Brune, M., Haroche, S., Raimond, J. M., Davidovich, L. & Zagury, N. Manipulation of photons in a cavity by dispersive atom-field coupling: Quantum-nondemolition measurements and generation of “schrödinger cat” states. Phys. Rev. A 45, 5193–5214 (1992). URL https://link.aps.org/doi/10.1103/PhysRevA.45.5193.   
38. Haroche, S. & Raimond, J. M. Exploring the Quantum: Atoms, Cavities, and Photons (Oxford Univ. Press, Oxford, 2006). URL https://cds.cern.ch/record/993568.   
39. Mari, A. & Eisert, J. Positive wigner functions render classical simulation of quantum computation efficient. Phys. Rev. Lett. 109, 230503 (2012). URL https://link.aps.org/doi/10.1103/PhysRevLett.109.230503.

40. Walschaers, M., Fabre, C., Parigi, V. & Treps, N. Entanglement and wigner function negativity of multimode non-gaussian states. Phys. Rev. Lett. 119, 183601 (2017). URL https://link.aps.org/doi/10.1103/PhysRevLett.119.183601.   
41. Albarelli, F., Genoni, M. G., Paris, M. G. A. & Ferraro, A. Resource theory of quantum non-gaussianity and wigner negativity. Phys. Rev. A 98, 052350 (2018). URL https://link.aps.org/doi/10.1103/PhysRevA.98.052350.   
42. Tan, K. C., Choi, S. & Jeong, H. Negativity of quasiprobability distributions as a measure of nonclassicality. Phys. Rev. Lett. 124, 110404 (2020). URL https://link.aps.org/doi/10.1103/PhysRevLett.124.110404.