This is the accepted manuscript made available via CHORUS. The article has been published as:

# Proposal for Heralded Generation and Detection of Entangled Microwave-Optical-Photon Pairs

Changchun Zhong, Zhixin Wang, Changling Zou, Mengzhen Zhang, Xu Han, Wei Fu, Mingrui Xu, S. Shankar, Michel H. Devoret, Hong X. Tang, and Liang Jiang

Phys. Rev. Lett. 124, 010511 — Published 10 January 2020

DOI: 10.1103/PhysRevLett.124.010511

# Proposal for Heralded Generation and Detection of Entangled Microwave–Optical Photon Pairs

Changchun Zhong, $^{1,2,*}$ Zhixin Wang, $^{1,2}$ Changling Zou, $^{3}$ Mengzhen Zhang, $^{1,2}$ Xu Han, $^{2,4}$ Wei Fu, $^{2,4}$ Mingrui Xu, $^{2,4}$ S. Shankar, $^{1,2}$ Michel H. Devoret, $^{1,2}$ Hong X. Tang, $^{1,2,4}$ and Liang Jiang $^{1,2,\dagger}$

$^{1}$ Department of Applied Physics, Yale University, New Haven, CT 06520, USA

$^{2}$ Yale Quantum Institute, Yale University, New Haven, CT 06520, USA

$^{3}$ Key Laboratory of Quantum Information, CAS,

University of Science and Technology of China, Hefei, Anhui 230026, China

$^{4}$ Department of Electrical Engineering, Yale University, New Haven, CT 06520, USA

(Dated: December 2, 2019)

Quantum state transfer between microwave and optical frequencies is essential for connecting superconducting quantum circuits to optical systems and extending microwave quantum networks over long distances. However, establishing such a quantum interface is extremely challenging because the standard direct quantum transduction requires both high coupling efficiency and small added noise. We propose an entanglement-based scheme—generating microwave-optical entanglement and using it to transfer quantum states via quantum teleportation—which can bypass the stringent requirements in direct quantum transduction and is robust against loss errors. In addition, we propose and analyze a counter-intuitive design—suppress the added noise by placing the device at a higher temperature environment—which can improve both the device quality factor and power handling capability. We systematically analyze the generation and verification of entangled microwave-optical photon pairs. The parameter for entanglement verification favors the regime of cooperativity mismatch and can tolerate certain thermal noises. Our scheme is feasible given the latest advances on electro-optomechanics, and can be generalized to various physical systems.

The modular quantum architecture—moderate-sized quantum registers connected by efficient communication channels—is a competitive approach toward a scalable quantum network $[1-3]$ . Physically, it is comprised of natural or artificial “atoms”—the nodes—and flying photons—the interconnects. As engineerable mesoscopic “atoms,” superconducting qubits $[4, 5]$ can strongly interact with microwave photons in cavities or waveguide resonators, known as the circuit quantum electrodynamics (cQED) architecture $[6, 7]$ . Microwave photons have been employed to entangle remote transmon qubits $[8-10]$ and cavity memories $[11, 12]$ . However, the high loss in microwave cables at room temperature prevents the transmission of quantum signals over long distances $[13]$ . In contrast, optical photons stand out as quantum information carriers at large spatial scales—entanglement and teleportation have been demonstrated over kilometers through telecommunication fibers $[14, 15]$ and free space $[16]$ . Therefore, high-fidelity quantum state transfer between superconducting circuits and optical photons will greatly expand the quantum computing network, as well as bridging superconducting qubits with different quantum modules $[17-22]$ .

However, superconducting circuits do not have an optical transition. A quantum transducer is needed to interface microwave and optical photons in a quantum coherent manner. So far, most investigations of quantum transducers are based on direct quantum transduction, which linearly converts photons between different frequencies. Proposed direct transducers involve cold alkali atoms $[23–25]$ , rare-earth-doped crystals $[26, 27]$ , magnons [28], electro-optics [29–31] and nanomechanics [32–44]. Recent experiments on cavity electro-optomechanics are very encouraging [32, 34, 41–46], but it is still challenging to achieve both high conversion efficiency and small added noise. In these setups, microwave and mechanical resonators are coupled by either electrostatic or piezoelectric forces. Nanomembranes for electrostatic coupling usually vibrate at MHz frequencies [32, 41, 45, 47], resulting in a narrow conversion bandwidth and high added thermal noise. On the other hand, piezoelectric oscillators at GHz frequencies [34, 46, 48–51]—couple to much lower thermal noise—can be routinely fabricated with high cooperativity [39, 46]. However, there are still challenges from the coupling loss at both microwave and optical interfaces that limit the conversion efficiency [42, 51, 52], and the undesired optical heating of the device [53, 54] which requires new design to improve the power handling capability (PHC). In this letter, we propose to boost the PHC by immersing the unit in a 1 K environment, and the GHz oscillator can have a favorably low intrinsic mechanical loss and be further cooled close to its ground state by coupling to another 10 mK bath.

A direct photon converter is capable of transferring quantum states only if the conversion efficiency $\eta > 1/2$ [56]. In principle, we can bypass this stringent requirement by introducing two-way classical communication that first heralds successful entanglement generation [8, 57–63] then completes the quantum state transfer by quantum teleportation [64–66]. Thus, we present such a heralded microwave-optical (M–O) entangled photon

pair generation and detection scheme, which is the first step of realizing this entanglement-based quantum transduction. We analyze its implementation in a generic cavity electro-optomechanical system, and identify the preferable parameter ranges for manifesting M–O entanglement. The entanglement analysis can be generalized to transducers based on various physical platforms which potentially inspires new M–O entanglement related applications $[67]$ .

Entanglement generation.—Without loss of generality, our discussions are based on the model depicted in Fig. 1. The thickness mode of a mechanical oscillator is on one side linearly coupled to a microwave resonator by piezoelectric force [39, 49, 51], and on the other side parametrically coupled to an optical cavity by radiation pressure [68]. The frequencies of the optical, the mechanical, and the microwave resonators are $\omega_{\mathrm{o}}$ , $\omega_{\mathrm{m}}$ , and $\omega_{\mathrm{e}}$ , respectively. A laser at frequency $\omega_{\mathrm{p}} = \omega_{\mathrm{o}} + \Delta_{\mathrm{p}}$ pumps the optical cavity and populates it by $\bar{n}_{\mathrm{o}}$ photons on average. In the rotating frame of the pump, we write the linearized Hamiltonian of the system

$$
\begin{array}{l} \hat {H} / \hbar = - \Delta_ {\mathrm{p}} \hat {a} ^ {\dagger} \hat {a} + \omega_ {\mathrm{m}} \hat {b} ^ {\dagger} \hat {b} + \omega_ {\mathrm{e}} \hat {c} ^ {\dagger} \hat {c} - g _ {\mathrm{em}} (\hat {b} ^ {\dagger} \hat {c} + \hat {b} \hat {c} ^ {\dagger}) \tag {1} \\ - g _ {\mathrm{om}, 0} \sqrt {\bar {n} _ {\mathrm{o}}} (\hat {a} ^ {\dagger} + \hat {a}) (\hat {b} ^ {\dagger} + \hat {b}), \\ \end{array}
$$

where $\hat{a}$ , $\hat{b}$ , and $\hat{c}$ represent the optical, mechanical, and microwave modes; $g_{em}$ and $g_{om,0}$ are the piezoelectric and the single-photon optomechanical coupling rates.

To entangle microwave and optical photons, as shown in the inset (a) of Fig. 1, we generate entangled phonon-photon modes by driving an optomechanical parametric down-conversion (PDC) process using a blue-sideband pump with $\Delta_{\mathrm{p}} = \omega_{\mathrm{m}}$ [60, 63, 69]. Meanwhile, mechanical excitations are swapped into the microwave resonator through the piezoelectric interaction. The M-O mode is approximated in a two-mode squeezed vacuum $|\psi_{\mathrm{sq}}(\lambda)\rangle_{\mathrm{oe}} \simeq \sum_{N=0}^{\infty} \frac{\lambda^{N}}{N!} (\hat{a}^{\dagger})^{N} (\hat{c}^{\dagger})^{N} |0\rangle_{\mathrm{o}} |0\rangle_{\mathrm{e}}$ , where subscripts "o" and "e" represent the optical and microwave modes. $\lambda$ is the effective squeezing factor [70]. For a weak pump ( $\lambda \ll 1$ ), a M-O photon pair can be obtained with probability $|\lambda|^2$ .

To use flying photons as information carriers and demonstrate non-classical correlations, we can encode qubits into polarizations of multiple modes $[71, 72]$ , momenta $[73]$ , time or frequency bins $[74, 75]$ . As an example, we present the scheme of generating time-bin entanglement in our proposed setup $^{1}$ . As shown in Fig. 1, the optical cavity is pumped by two blue-sideband pulses separated by time $\Delta t$ , which is within the coherence time of the pump laser and superconducting qubits. Denoting $\hat{a}_{\mathrm{out,c}}^{(1,2)}$ and $\hat{c}_{\mathrm{out,c}}^{(1,2)}$ as the optical and microwave output mode operators in time-bin 1 and 2, respectively, the M–O output modes is

$$
\begin{array}{l} \left| \Psi_ {\mathrm{tb}} (\lambda) \right\rangle_ {\mathrm{oe}} \simeq \left| 0, 0 \right\rangle_ {\mathrm{o}} \left| 0, 0 \right\rangle_ {\mathrm{e}} + \lambda \hat {a} _ {\text { out,c }} ^ {(1) \dagger} \hat {c} _ {\text { out,c }} ^ {(1) \dagger} \left| 0, 0 \right\rangle_ {\mathrm{o}} \left| 0, 0 \right\rangle_ {\mathrm{e}} \\ + \lambda \hat {a} _ {\text { out,c }} ^ {(2) \dagger} \hat {c} _ {\text { out,c }} ^ {(2) \dagger} | 0, 0 \rangle_ {\mathrm{o}} | 0, 0 \rangle_ {\mathrm{e}} + O (\lambda^ {2}), \tag {2} \\ \end{array}
$$

where for either mode, $|0,0\rangle$ denotes the state with zero photon in the first and second time bin. Neglecting the $O(\lambda^{2})$ terms and discarding the zero-photon events by postselection, we obtain a time-bin Bell state $\frac{\sqrt{2}}{2}(|1,0\rangle_{\mathrm{o}}|1,0\rangle_{\mathrm{e}}+|0,1\rangle_{\mathrm{o}}|0,1\rangle_{\mathrm{e}})$ with probability $|\lambda|^{2}$ , which is to be verified in the rest of the discussion.

Flying photon measurement.—Optical time-bin qubits can be detected with an unbalanced Mach–Zehnder interferometer $[76]$ . As depicted on the left of Fig. 1, given the relative time delay between the long and short arms precisely matching $\Delta t$ , the photons in the first (second) time-bin passing through the long (short) arm will interfere at the second beam splitter. A click at $D_{1,2}$ projects the optical qubit on $|\varphi_{o}^{\pm}\rangle_{o} = \frac{1}{\sqrt{2}}(|1,0\rangle_{o} \pm e^{i\varphi_{o}}|0,1\rangle_{o})$ , where $\varphi_{o}$ is an adjustable phase. Note the unconditional maximum efficiency of this projective readout is 0.5, because half of the clicks correspond to the photons in the first (second) time-bin passing through the short (long) arm produce no interference $[77]$ . Maximum visibility of unity can be recovered by discarding these early and late counts through postselection.

On the microwave side (right panel of Fig. 1), flying photons in two time-bins are first converted to excitations of two transmon qubits in a cQED system through stimulated two-photon Raman absorption [9], after which the transmons and the optical modes are entangled as $\frac{1}{\sqrt{2}}(|1,0\rangle_{\mathrm{o}}|eg\rangle + |0,1\rangle_{\mathrm{o}}|ge\rangle)$ , where $g(e)$ denotes the ground (first excited) state of a transmon. After a joint parity measurement (to be explained in the next paragraph), a CNOT gate [78] followed by a high-fidelity single-shot qubit readout [79] projects the two transmons on $\frac{1}{\sqrt{2}}(|eg\rangle \pm e^{-i\varphi_{\mathrm{e}}} |ge\rangle)$ , where $\varphi_{e}$ is a tunable phase. We thus effectively project the microwave photons on $|\varphi_{\mathrm{e}}^{\pm}\rangle_{\mathrm{e}} = \frac{1}{\sqrt{2}}(|1,0\rangle_{\mathrm{e}} \pm e^{-i\varphi_{\mathrm{e}}} |0,1\rangle_{\mathrm{e}})$ .

During the experiment, heralded signals indicating successful entanglement generation are produced by measuring the output photons in both microwave and optical domains—a M–O Bell state contains only one photon pair. Specifically, on the microwave side, instead of counting photons, we can measure the parity of the two-qubit state immediately after the stimulated Raman absorption and postselect on the condition that only one transmon is excited $[80, 81]$ . Excluded by heralding are null events $|0,0\rangle_{e}|0,0\rangle_{o}$ , undecayed higher-order generation events, and the cases where the flying photons are lost. Although not increasing the generation rate, this

TABLE I. Feasible parameters. Unless specified otherwise, these parameters apply to all evaluations in the text. 

<table><tr><td> $g_{\text{em}}/\text{MHz}$ </td><td> $g_{\text{om},0}/\text{kHz}$ </td><td> $\kappa_{\text{e,i}}/\text{kHz}$ </td><td> $\kappa_{\text{o,i}}/(\text{GHz})$ </td><td> $\kappa_{\text{e,c}}$ </td><td> $\kappa_{\text{o,c}}$ </td><td> $\kappa_{\text{m}}/\text{kHz}$ </td><td> $\bar{n}_{\text{ba}}(\text{T}=1\ \text{K})$ </td><td> $\omega_{\text{m}}/\text{GHz}$ </td><td> $\omega_{\text{e}}/\text{GHz}$ </td><td> $\omega_{\text{o}}/\text{THz}$ </td></tr><tr><td> $2\pi \times 2.0$ </td><td> $2\pi \times 5.5$ </td><td> $2\pi \times 100$ </td><td> $2\pi \times 0.24$ </td><td> $(\kappa_{\text{e,i}}, 10^{3}\kappa_{\text{e,i}})$ </td><td> $\kappa_{\text{o,i}}$ </td><td> $2\pi \times 20$ </td><td> $\sim 1.6$ </td><td> $2\pi \times 10$ </td><td> $2\pi \times 10$ </td><td> $2\pi \times 195$ </td></tr></table>

![](images/56e6bc9eb87c59711674bcfeaa9393f3667f5f05dbd58235a68652419161c162.jpg)

<details>
<summary>flowchart</summary>

```mermaid
```mermaid
graph LR
    subgraph "time delay = Δt"
        D1 --> D2
        D2 --> φo
        φo --> laser
        laser --> beam splitters
        beam splitters --> ωp
        beam splitters --> ωm
        beam splitters --> ωe
    end

    subgraph "T_E=1 K"
        T_e["TE=1 K"] --> ωp
        T_e --> ωm
        T_e --> ωe
        ωp --> laser
        ωm --> laser
        laser --> beam splitters
        beam splitters --> ωp
        beam splitters --> ωm
        beam splitters --> ωe
    end

    subgraph "T_E=10 mK"
        T_e --> cQED
        cQED --> qubit_control_&_readout
        qubit_control_&_readout --> QLA
        qubit_control_&_readout --> Q1
        qubit_control_&_readout --> Q2
        qubit_control_&_readout --> Q1
        qubit_control_&_readout --> Q2
        qubit_control_&_readout --> Q1
        qubit_control_&_readout --> Q2
    end

    subgraph "(a)" and "(b)"
        subgraph "(a) ω_e = ω_m ω_o ω_p"
            direction TB
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction TB
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
            direction LR
    
    %% Note: The diagram depicts a schematic of an optical or signal processing system with labeled components and parameters. The structure is designed to be a multi-component optical system with multiple input/output channels and outputs. The output is structured as a single line from the left to right, with arrows indicating waveforms or transitions. The output is then connected to a common output node via a feedback loop.
```
</details>

FIG. 1. Schematic setup. The thickness mode of a piezoelectric resonator (yellow) is simultaneously coupled to a microwave LC resonator (green) and an optical cavity. In each round, two blue-detuned pulses (purple) separated by $\Delta t$ pump the optical cavity. Inset (a) gives the frequency landscape. Optical photons (red) are analyzed by an unbalanced Mach–Zehnder interferometer: on the long arm, photons are delayed by time $\Delta t$ ; on the short arm, photon phase is shifted by $\varphi_{o}$ . The outputs of the second beam splitter are sent into two single-photon detectors. On the microwave side, through stimulated Raman absorption, photons (green) are converted into qubit excitations in a cQED system consisting of two transmon qubits (blue and brown) with matched dispersive shifts to a cavity mode (black). Qubit readout is performed with the aid of a quantum-limited amplifier (QLA). Inset (b) shows a joint parity measurement, a CNOT gate, and a single-qubit readout. The black line represents the readout cavity initialized in a coherent state $|\alpha\rangle$ . A controlled-phase gate ( $C_{\theta}$ ) followed by a meter represents the cQED dispersive readout. $R_{\varphi_{e}}(\pi/2)$ represents a $\pi/2$ rotation about the (sin $\varphi_{e}$ , cos $\varphi_{e}$ , 0) axis on the Bloch sphere. The mechanical/microwave mode, with environment $T_{E} = 1~K$ , can be cooled to the quantum regime by radiatively coupled to a 10 mK cold bath. At every stage of generation, manipulation, and detection of microwave photons, the average noise occupation is sufficiently small with an effective temperature lower than $\hbar\omega_{e}$ .

heralded scheme significantly improves the fidelities of Bell pairs and enables the entanglement verification given non-negligible transmission loss and limited detection efficiencies.

Gaussian output with dissipation and noise.—Coupling to the thermal environment will degrade the output to a two-mode squeezed Gaussian state. Besides thermal noise from finite fridge temperature, an extra heating to the device is unavoidable from the optical pump $[53]$ . To ensure that the device is not overheated, we propose a counter-intuitive design—operating the device in a (higher temperature) 1 K environment (Fig. 1), on one hand to increase the PHC, on the other hand to achieve an optimal mechanical quality factor $^{2}$ . 1 K is the environment temperature of the mechanical/microwave mode, while the thermal mode occupations can be further reduced to the quantum regime by over-coupling to a 10 mK bath $[55]$ . This is a crucial part of the design, which is distinct from previous proposals. Also, one can further enhance the PHC by immersing the unit in liquid with high thermal conductivity and low fluid damping, e.g., super fluid He II is a possibility [83].

We label the optical, microwave, and mechanical dissipation rates by $\kappa_{o} = \kappa_{o,i} + \kappa_{o,c}$ , $\kappa_{e} = \kappa_{e,i} + \kappa_{e,c}$ , and $\kappa_{m}$ (the subscript ‘i’ for internal loss, ‘c’ for external coupling), and assume the thermal phonon (photon) population of the mechanical (microwave) dissipation bath is $\bar{n}_{\mathrm{ba}} = (e^{\hbar\omega_{\mathrm{m(e)}}/k_{\mathrm{B}}T} - 1)^{-1}$ on average, while the optical resonator and the optical and microwave coupling ports are purely subject to vacuum fluctuations. Combining the Heisenberg-Langevin equations with the input-output theory, we derive the output two-mode Gaussian state described by a covariance matrix [84]. Denoting the M–O state quadratures as $x_{oe}^{out} = \{\hat{q}_{o}, \hat{p}_{o}, \hat{q}_{e}, \hat{p}_{e}\}$ , we characterize the output M–O state by showing its power spectrum densities $u(\omega) = \langle |\hat{q}_{o}(\omega)|^{2} \rangle$ (optical mode) and $v(\omega) = \langle |\hat{q}_{e}(\omega)|^{2} \rangle$ (microwave mode) in Fig. 2(d). The microwave mode has higher spectrum density due to its intrinsic coupling to the thermal bath. Also, we numerically calculate the entanglement of formation $E_{F}$ [84] as a function of the optomechanical cooperativity $C_{om} = 4g_{om}^{2}/\kappa_{o}\kappa_{m}$ and the microwave readout ratio $\kappa_{e,c}/\kappa_{e,i}$ , which is tunable in experiments via the pump power and the position of the microwave readout probe, respectively. Using feasible parameters in Tab. I, $E_{F}$ with resonant

![](images/8ecd69ef8556fdddadf5ffe682cdcda709ca0f498bbb596fe24df1f0958da249.jpg)

<details>
<summary>heatmap</summary>

| K_e,c / K_e,i | C_om | E_F (ebit) |
| ------------- | ---- | ---------- |
| 10^2          | 10^1 | 1.25       |
| 10^3          | 10^0 | 0.50       |
| 10^4          | 10^-1| 0.00       |
</details>

![](images/cd6028fe5891ebde0a34f7a0da497f873e442f6238f31c032941a10a86bf39c4.jpg)

<details>
<summary>heatmap</summary>

| C_om | K_e,c / K_e,i | μ_P |
|------|---------------|-----|
| 10^0 | 10^2          | unstable |
| 10^0 | 10^3          | stable |
| 10^1 | 10^2          | unstable |
| 10^1 | 10^4          | stable |
</details>

![](images/6cc6bcd0ef4a9c7dcd063dc1761a1aef9f67f4258247e7e881ff21e2359e07ac.jpg)

<details>
<summary>line</summary>

| K_e,c / K_e,i | E_F (r̄_ba = 0) | E_F (r̄_ba = 1.6) | E_F (r̄_ba = 5) | E_F (r̄_ba = 10) | μ_P (r̄_ba = 0) | μ_P (r̄_ba = 1.6) | μ_P (r̄_ba = 5) | μ_P (r̄_ba = 10) |
| ------------- | --------------- | ----------------- | --------------- | ---------------- | -------------- | ----------------- | -------------- | ---------------- |
| 10^2          | ~1.0            | ~1.0              | ~0.8            | ~0.8             | ~0.1           | ~0.1              | ~0.1           | ~0.1             |
| 10^3          | ~0.6            | ~0.7              | ~0.5            | ~0.4             | ~0.2           | ~0.2              | ~0.2           | ~0.2             |
| 10^4          | ~0.2            | ~0.3              | ~0.2            | ~0.1             | ~0.05          | ~0.05             | ~0.05          | ~0.05            |
</details>

![](images/c88a000b66eea9d7a6bae32514e8975aa17d412455d5e4da227bd8efc8797759.jpg)

<details>
<summary>line</summary>

| ω (2π×10⁶ Hz) | u(ω) | v(ω) |
| ------------- | ---- | ---- |
| -4            | 1.0  | 1.0  |
| -2            | 1.0  | 1.0  |
| 0             | 1.1  | 1.4  |
| 2             | 1.0  | 1.0  |
| 4             | 1.0  | 1.0  |
</details>

FIG. 2. (a) Entanglement of formation $E_{F}$ and (b) state purity $\mu_{P}$ versus $C_{om}$ and $\kappa_{e,c}/\kappa_{e,i}$ . The cyan lines are given by $C_{om} = C_{em} + 1$ , separating the stable from the unstable regimes. (c) $E_{F}$ and $\mu_{P}$ plotted with varied thermal noises and fixed $C_{om} = 1$ —the black dashed lines in (a) and (b). (d) The optical (red) and microwave (blue) output power spectrum densities where $C_{om} = 1, \kappa_{e,c}/\kappa_{e,i} = 150$ . Parameters in Tab.I apply to all figures.

frequency is plotted in Fig. 2(a). $E_{F} > 0$ indicates the continuous-variable entanglement under any finite squeezing and $E_{F}$ reaches its maximum when the electromechanical cooperativity $C_{em} = 4g_{em}^{2}/\kappa_{e}\kappa_{m} \simeq C_{om} - 1$ , where strong PDC dominates and highly entangled state with extremely low purity $\mu_{P}$ is generated, as shown in Fig. 2(b) (see [85] for purity definition). In fact, by checking the Routh-Hurwitz stability criterion [86–88], the system becomes unstable when the optical pump is too strong $C_{om} \gg C_{em} + 1$ , as indicated in Fig. 2(a,b). In Fig. 2(c), the $E_{F}$ and $\mu_{P}$ are shown with $C_{om} = 1$ and different thermal nosies. As expected, higher thermal noise degrades the entanglement and purity. It is worth pointing out that in these plots, the parameter regime $g_{em} < \kappa_{e}$ is chosen in order to avoid electromechanical strong coupling where the mode splitting can potentially complicate the photon detection. Nevertheless, entanglement in the strong coupling regime would be interesting for further investigations.

Entanglement verification.—A lower bound of the M–O output state fidelity with respect to an ideal Bell state is given by $[18, 62]$

$$
F_{\mathrm{lb}} = \sum_{\substack{\nu = \pm \\ \varphi = 0,\frac{\pi}{2}}}\frac{p^{\nu\nu}_{\varphi\varphi}}{2} -\frac{p^{\frac{+ - }{\frac{\pi}{2}\frac{\pi}{2}}}}{2} -\frac{p^{\frac{- + }{\frac{\pi}{2}\frac{\pi}{2}}}}{2} -\sqrt{p_{00}^{+ - }p_{00}^{- + }},\qquad (3)
$$

in which $p_{\varphi_{o}\varphi_{e}}^{\mu\nu}$ is the normalized probability of detecting $|\varphi_{o}^{\mu}\rangle_{o}$ and $|\varphi_{e}^{\nu}\rangle_{e}$ on the optical and microwave sides.

![](images/c062a225aa43e766a0b94cf28e8a220bf3054de65498d9856137d448bffc31ec.jpg)

![](images/873a63761dd22b3d5e490ab18d012d18bb5118da70fdc76d1228f41253c38ffa.jpg)

<details>
<summary>line</summary>

| φ_e   | S (π̄_ba = 0) | S (π̄_ba = -1.6) | S (π̄_ba = 5) | S (π̄_ba = 10) |
|-------|--------------|-----------------|--------------|---------------|
| π/4   | ~2√2         | ~2√2            | ~2√2         | ~2√2          |
| 3π/4  | ~0           | ~0              | ~0           | ~0            |
| 5π/4  | ~-2√2        | ~-2√2           | ~-2√2        | ~-2√2         |
| 7π/4  | ~2√2         | ~2√2            | ~2√2         | ~2√2          |
</details>

FIG. 3. (a) “Phase diagram” of Bell state fidelity, Bell inequality violation for $\bar{n}_{ba} = 1.6$ , and intrinsic direct conversion efficiency. The regions with $|S|_{max} > 2$ , $F_{lb} > 0.5$ (sufficient condition for entanglement) are shaded with orange, light yellow and $\eta_{in} > 0.5$ (necessary condition for direct quantum transduction) with light blue, respectively. (b) $S(0, \varphi_{\mathrm{e}}; \pi/2, \varphi_{\mathrm{e}} + \pi/2)$ for varied thermal noises, given $C_{om} = 1$ and $\kappa_{e,c}/\kappa_{e,i} = 150$ . In these calculations, the measurement efficiencies are assumed to be 1/2 on both the microwave and optical sides.

$F_{lb} > 1/2$ indicates the M–O entanglement. Such regions are delineated in Fig. 3(a) showing that a better fidelity should avoid the strong PDC parameter regime. Although $E_{F}$ is large at these regimes, the highly-mixed-entangled state (shown in Fig. 2(b)) is not suitable for current entanglement verification. This is consistent with the previous discussions that weakly squeezed M–O output states better approximate the two-mode squeezed vacuum and thus the time-bin Bell state.

A stronger entanglement manifestation that excludes local hidden-variables is the violation of CHSH inequality [89], which can be tested in our proposed setup by measuring

$$
\begin{array}{c} S (\varphi_ {\mathrm{o}}, \varphi_ {\mathrm{e}}; \varphi_ {\mathrm{o}} ^ {\prime}, \varphi_ {\mathrm{e}} ^ {\prime}) = \\ E (\varphi_ {\mathrm{o}}, \varphi_ {\mathrm{e}}) + E (\varphi_ {\mathrm{o}} ^ {\prime}, \varphi_ {\mathrm{e}} ^ {\prime}) + E (\varphi_ {\mathrm{o}} ^ {\prime}, \varphi_ {\mathrm{e}}) - E (\varphi_ {\mathrm{o}}, \varphi_ {\mathrm{e}} ^ {\prime}). \end{array} \tag {4}
$$

$E(\varphi_{\mathrm{o}},\varphi_{\mathrm{e}})\equiv p_{\varphi_{\mathrm{o}}\varphi_{\mathrm{e}}}^{++}+p_{\varphi_{\mathrm{o}}\varphi_{\mathrm{e}}}^{--}-p_{\varphi_{\mathrm{o}}\varphi_{\mathrm{e}}}^{+-}-p_{\varphi_{\mathrm{o}}\varphi_{\mathrm{e}}}^{-+}$ is acquired by detecting photonic and superconducting qubits on the $\{|\varphi_{\mathrm{o}}^{\pm}\rangle_{\mathrm{o}},|\varphi_{\mathrm{e}}^{\pm}\rangle_{\mathrm{q}}\}$ basis. Choosing $\varphi_{\mathrm{o}}^{\prime}-\varphi_{\mathrm{o}}=\varphi_{\mathrm{e}}^{\prime}-\varphi_{\mathrm{e}}=\pi/2$ , we simulate the typical curves of S in Fig. 3(b) for varied $\bar{n}_{\mathrm{ba}}$ . We obtain familiar sinusoidal curves which reach the maximum when $\varphi_{\mathrm{e}}-\varphi_{\mathrm{o}}=\pi/4+k\pi$ ( $k\in N$ ). For low thermal noises, clear violation ( $|S|_{max}>2$ ) is observed. The $|S|_{max}>2$ regions are delineated in Fig. 3(a), indicating that the threshold for CHSH violation is more stringent than $F_{1b}>0.5$ .

The entanglement generation, thus the entanglement-based transduction, is feasible giving the latest experimental developments. In comparison, the direct quantum transduction favors the cooperativity-matched regime, shown by the blue shaded area in Fig. 3(a), which encloses the regime with intrinsic conversion efficiency $\eta_{\mathrm{in}} > 0.5$ —a necessary condition for direct quantum con

vesion $^{3}$ . First, this regime is generally hard to reach. Second, to demonstrate quantum transduction, it is necessary to maintain big coupling ratios $(\kappa_{\mathrm{e,c}}/\kappa_{\mathrm{e}}, \kappa_{\mathrm{o,c}}/\kappa_{\mathrm{o}})$ and low thermal noises, which are extremely challenging to realize simultaneously, e.g., no quantum information can be directly transferred with the same coupling ratios and thermal bath as in Tab. I since the overall conversion efficiency $\eta < 0.5$ .

Discussion.—Experimentally, entanglement is revealed by coincidentally counting M–O photons. The coincidence counting rate $R_{c}$ is an essential quantity, which contains two parts: 1) the accidental coincidence rate $R_{ac}$ —random M–O photons detected simultaneously; 2) the correlated coincidence rate $R_{cc}$ —parametrically-down-converted M–O photons detected simultaneously. Ideally, all output M–O photons will be detected. However, practical imperfections could degrade the detection scheme, e.g., photon transmission losses, dark counts and detector inefficiencies. To estimate the practical coincidence rate, we use $T_{o}$ , $D_{o}$ , $\eta_{o}$ ( $T_{e}$ , $D_{e}$ , $\eta_{e}$ ) to model the optical (microwave) photon transmissivity, dark count rate and detector efficiency, respectively (see [84] for details). With feasible parameters, we show the coincidence counting rate $R_{c}$ is on the order of 10 Hz.

To have a good entangled M–O source, we must require the correlated coincidence rate to be larger than the accidental coincidence rate $R_{cc} > R_{ac}$ . This condition leads to $g^{(2)} > 2 + \xi_{o} + \xi_{e} + \xi_{o}\xi_{e}$ , where $g^{(2)}$ is the Glauber correlation function [90] and $\xi_{o} (\xi_{e})$ is the ratio of the optical (microwave) dark count rate to the generated photon detection rate. This inequality has a clear physical meaning that the quantum correlation—indicated by $g^{(2)}$ function should outperform all noises induced by the dark count, detector inefficiency and transmission loss.

Our proposal is compatible with recent experiments on cavity electro-optomechanics. Given a relatively smaller $C_{om}$ [39, 42, 46], the system can be well prepared in the cooperativity mismatched regime, leading to efficient entanglement generation and detection even with a few thermal photons. Furthermore, although our analysis is on electro-optomechanics, the scheme can be generalized to other types of M–O converters based on parametric interactions—for instance, the electro-optic converters [31, 91]—and thus shed light on M–O quantum transduction in various physical systems.

We acknowledge insightful discussions with Chiao-Hsuan Wang, Yiwen Chu, Vijay Jain. This research is mainly supported by the ARO CQTS (W911NF-18-1-0020). We also acknowledge support from the ARL-CDQI (W911NF-15-2-0067, W911NF-18-2-0237), ARO HIPS (W911NF-18-1-0212), ARO MURI (W911NF-16-1-0349), AFOSR MURI (FA9550-14-1-0052, FA9550-15-1-0015), DOE (DE-SC0019406), NSF (EFMA-1640959), and the Packard Foundation (2013-39273).

\* zhong.changchun@yale.edu   
$^{\dagger}$ liang.jiang@yale.edu   
[1] J. I. Cirac, P. Zoller, H. J. Kimble, and H. Mabuchi, Phys. Rev. Lett. 78, 3221 (1997); H. J. Kimble, Nature 453, 1023 (2008).   
[2] L. Jiang, J. M. Taylor, A. S. Sørensen, and M. D. Lukin, Phys. Rev. A 76, 062323 (2007).   
[3] C. Monroe, R. Raussendorf, A. Ruthven, K. R. Brown, P. Maunz, L.-M. Duan, and J. Kim, Phys. Rev. A 89, 022317 (2014).   
[4] D. Esteve, J.-M. Raimond, and J. Dalibard, eds., Quantum Entanglement and Information Processing (Les Houches Summer School, Session LXXIX) (Elsevier, New York, 2004).   
[5] J. Clarke and F. K. Wilhelm, Nature 453, 1031 (2008).   
[6] A. Blais, R.-S. Huang, A. Wallraff, S. M. Girvin, and R. J. Schoelkopf, Phys. Rev. A 69, 062320 (2004).   
[7] S. M. Girvin, in Quantum Machines: Measurement and Control of Engineered Quantum Systems (Les Houches Summer School, Session XCVI), edited by M. Devoret, B. Huard, R. Schoelkopf, and L. F. Cugliandolo (Oxford University Press, Oxford, 2014) pp. 113–255.   
[8] A. Narla, S. Shankar, M. Hatridge, Z. Leghtas, K. M. Sliwa, E. Zalys-Geller, S. O. Mundhada, W. Pfaff, L. Frunzio, R. J. Schoelkopf, and M. H. Devoret, Phys. Rev. X 6, 031036 (2016).   
[9] P. Campagne-Ibarcq, E. Zalys-Geller, A. Narla, S. Shankar, P. Reinhold, L. Burkhart, C. Axline, W. Pfaff, L. Frunzio, R. J. Schoelkopf, and M. H. Devoret, Phys. Rev. Lett. 120, 200501 (2018).   
[10] P. Kurpiers, P. Magnard, T. Walter, B. Royer, M. Pechal, J. Heinsoo, Y. Salathé, A. Akin, S. Storz, J.-C. Besse, S. Gasparinetti, A. Blais, and A. Wallraff, Nature 558, 264 (2018).   
[11] C. J. Axline, L. D. Burkhart, W. Pfaff, M. Zhang, K. Chou, P. Campagne-Ibarcq, P. Reinhold, L. Frunzio, L. Jiang, M. H. Devoret, and R. J. Schoelkopf, Nat. Phys. 14, 705 (2018).   
[12] K. S. Chou, J. Z. Blumoff, C. S. Wang, P. C. Reinhold, C. J. Axline, Y. Y. Gao, L. Frunzio, M. H. Devoret, L. Jiang, and R. J. Schoelkopf, Nature 561, 368 (2018).   
[13] Recent measurements show that below 4 K, the attenuation constants of Nb and NbTi superconducting coaxial cables are $\sim10^{-2}$ dB/m at gigahertz frequencies. In principle, they can transmit single microwave photons within laboratory distances. See P. Kurpiers, T. Walter, P. Magnard, Y. Salathe, and A. Wallraff, EPJ Quant. Technol. 4, 8 (2017).   
[14] W. Tittel, J. Brendel, H. Zbinden, and N. Gisin, Phys. Rev. Lett. 81, 3563 (1998).   
[15] R. Valivarthi, M. l. G. Puigibert, Q. Zhou, G. H. Aguilar, V. B. Verma, F. Marsili, M. D. Shaw, S. W. Nam, D. Oblak, and W. Tittel, Nat. Photon. 10, 676 (2016).   
[16] J. Yin, Y. Cao, Y.-H. Li, S.-K. Liao, L. Zhang, J.-G. Ren, W.-Q. Cai, W.-Y. Liu, B. Li, H. Dai, et al., Science 356,

1140 (2017); J.-G. Ren, P. Xu, H.-L. Yong, L. Zhang, S.-K. Liao, J. Yin, W.-Y. Liu, W.-Q. Cai, M. Yang, L. Li, et al., Nature 549, 70 (2017).   
[17] D. N. Matsukevich and A. Kuzmich, Science 306, 663 (2004).   
[18] B. B. Blinov, D. L. Moehring, L.-M. Duan, and C. Monroe, Nature 428, 153 (2004).   
[19] E. Togan, Y. Chu, A. S. Trifonov, L. Jiang, J. Maze, L. Childress, M. V. G. Dutt, A. S. Sørensen, P. R. Hemmer, A. S. Zibrov, and M. D. Lukin, Nature 466, 730 (2010).   
[20] W. F. Koehl, B. B. Buckley, F. J. Heremans, G. Calusine, and D. D. Awschalom, Nature 479, 84 (2011).   
[21] W. B. Gao, P. Fallahi, E. Togan, J. Miguel-Sanchez, and A. Imamoglu, Nature 491, 426 (2012).   
[22] K. De Greve, L. Yu, P. L. McMahon, J. S. Pelc, C. M. Natarajan, N. Y. Kim, E. Abe, S. Maier, C. Schneider, M. Kamp, S. Höfling, R. H. Hadfield, A. Forchel, M. M. Fejer, and Y. Yamamoto, Nature 491, 421 (2012).   
[23] M. Hafezi, Z. Kim, S. L. Rolston, L. A. Orozco, B. L. Lev, and J. M. Taylor, Phys. Rev. A 85, 020302(R) (2012).   
[24] M. Kiffner, A. Feizpour, K. T. Kaczmarek, D. Jaksch, and J. Nunn, New J. Phys. 18, 093030 (2016).   
[25] B. T. Gard, K. Jacobs, R. McDermott, and M. Saffman, Phys. Rev. A 96, 013833 (2017).   
[26] L. A. Williamson, Y.-H. Chen, and J. J. Longdell, Phys. Rev. Lett. 113, 203601 (2014).   
[27] C. O'Brien, N. Lauk, S. Blum, G. Morigi, and M. Fleischhauer, Phys. Rev. Lett. 113, 063603 (2014).   
[28] R. Hisatomi, A. Osada, Y. Tabuchi, T. Ishikawa, A. Noguchi, R. Yamazaki, K. Usami, and Y. Nakamura, Phys. Rev. B 93, 174427 (2016).   
[29] M. Tsang, Phys. Rev. A 81, 063837 (2010); Phys. Rev. A 84, 043845 (2011).   
[30] C. Javerzac-Galy, K. Plekhanov, N. R. Bernier, L. D. Toth, A. K. Feofanov, and T. J. Kippenberg, Phys. Rev. A 94, 053815 (2016).   
[31] L. Fan, C.-L. Zou, R. Cheng, X. Guo, X. Han, Z. Gong, S. Wang, and H. X. Tang, Sci. Adv. Vol. 4, no. 8, eaar4994 (2018).   
[32] R. W. Andrews, R. W. Peterson, T. P. Purdy, K. Cicak, R. W. Simmonds, C. A. Regal, and K. W. Lehnert, Nat. Phys. 10, 321 (2014).   
[33] C. A. Regal and K. W. Lehnert, J. Phys.: Conf. Ser. 264, 012025 (2011).   
[34] J. Bochmann, A. Vainsencher, D. D. Awschalom, and A. N. Cleland, Nat. Phys. 9, 712 (2013).   
[35] J. M. Taylor, A. S. Sørensen, C. M. Marcus, and E. S. Polzik, Phys. Rev. Lett. 107, 273601 (2011).   
[36] S. Barzanjeh, D. Vitali, P. Tombesi, and G. J. Milburn, Phys. Rev. A 84, 042342 (2011).   
[37] Y.-D. Wang and A. A. Clerk, Phys. Rev. Lett. 108, 153603 (2012).   
[38] L. Tian and H. Wang, Phys. Rev. A 82, 053806 (2010); L. Tian, Phys. Rev. Lett. 108, 153604 (2012); Ann. Phys. 527, 1 (2014).   
[39] C.-L. Zou, X. Han, L. Jiang, and H. X. Tang, Phys. Rev. A 94, 013812 (2016).   
[40] L. Midolo, A. Schliesser, and A. Fiore, Nat. Nanotech. 13, 11 (2018).   
[41] T. Bagci, A. Simonsen, S. Schmid, L. G. Villanueva, E. Zeuthen, J. Appel, J. M. Taylor, A. Sørensen, K. Usami, A. Schliesser, and E. S. Polzik, Nature 507, 81 (2014).

[42] A. Vainsencher, K. J. Satzinger, G. A. Peairs, and A. N. Cleland, Appl. Phys. Lett. 109, 033107 (2016).   
[43] M. Winger, T. D. Blasius, T. P. M. Alegre, A. H. Safavi-Naeini, S. Meenehan, J. Cohen, S. Stobbe, and O. Painter, Opt. Express 19, 24905 (2011).   
[44] A. Pitanti, J. M. Fink, A. H. Safavi-Naeini, J. T. Hill, C. U. Lei, A. Tredicucci, and O. Painter, Opt. Express 23, 3196 (2015).   
[45] A. P. Higginbotham, P. S. Burns, M. D. Urmey, R. W. Peterson, N. S. Kampel, B. M. Brubaker, G. Smith, K. W. Lehnert, and C. A. Regal, Nat. Phys. 14, 1038-1042 (2018).   
[46] X. Han, C. Xiong, K. Y. Fong, X. Zhang, and H. X. Tang, New J. Phys. 16, 063060 (2014).   
[47] J. D. Teufel, D. Li, M. S. Allman, K. Cicak, A. J. Sirois, J. D. Whittaker, and R. W. Simmonds, Nature 471, 204 (2011); T. A. Palomaki, J. W. Harlow, J. D. Teufel, R. W. Simmonds, and K. W. Lehnert, Nature 495, 210 (2013); A. P. Reed, K. H. Mayer, J. D. Teufel, L. D. Burkhart, W. Pfaff, M. Reagor, L. Sletten, X. Ma, R. J. Schoelkopf, E. Knill, and K. W. Lehnert, Nat. Phys. 13, 1163 (2017).   
[48] A. D. O'Connell, M. Hofheinz, M. Ansmann, R. C. Bialczak, M. Lenander, E. Lucero, M. Neeley, D. Sank, H. Wang, M. Weides, J. Wenner, J. M. Martinis, and A. N. Cleland, Nature 464, 697 (2010).   
[49] Y. Chu, P. Kharel, W. H. Renninger, L. D. Burkhart, L. Frunzio, P. T. Rakich, and R. J. Schoelkopf, Science 358, 199 (2017); Y. Chu, P. Kharel, T. Yoon, L. Frunzio, P. T. Rakich, and R. J. Schoelkopf, arXiv:1804.07426.   
[50] X. Han, K. Y. Fong, and H. X. Tang, Appl. Phys. Lett. 106, 161108 (2015).   
[51] X. Han, C.-L. Zou, and H. X. Tang, Phys. Rev. Lett. 117, 123603 (2016).   
[52] M. Forsch, R. Stockill, A. Wallucks, I. Marinkovic, C. Gärtner, R. A. Norte, F. van Otten, A. Fiore, K. Srinivasan, and S. Gröblacher, arXiv preprint arXiv:1812.07588 (2018).   
[53] J. T. Hill, A. H. Safavi-Naeini, J. Chan, and O. Painter, Nature communications 3, 1196 (2012).   
[54] K. Fang, M. H. Matheny, X. Luan, and O. Painter, Nature Photonics 10, 489 (2016).   
[55] M. Xu, X. Han, C.-L. Zou, W. Fu, Y. Xu, C. Zhong, L. Jiang, and X. T. Hong, arXiv preprint arXiv:1910.01203 (2019); Z. Wang, M. Xu, X. Han, W. Fu, S. Puri, S. Girvin, H. X. Tang, S. Shankar, and M. Devoret, arXiv preprint arXiv:1909.12295 (2019).   
[56] A quantum loss channel has finite quantum capacity only if $\eta > 1/2$ . M. M. Wolf, D. Pérez-García, and G. Giedke, Phys. Rev. Lett. 98, 130501 (2007); L. Gyongyosi, S. Imre, H. V. Nguyen, IEEE Commun. Surv. Tutor. 20, 1149 (2018).   
[57] L.-M. Duan, M. D. Lukin, J. I. Cirac, and P. Zoller, Nature 414, 413 (2001).   
[58] C. W. Chou, H. de Riedmatten, D. Felinto, S. V. Polyakov, S. J. van Enk, and H. J. Kimble, Nature 438, 828 (2005).   
[59] D. L. Moehring, P. Maunz, S. Olmschenk, K. C. Younge, D. N. Matsukevich, L.-M. Duan, and C. Monroe, Nature 449, 68 (2007).   
[60] K. C. Lee, M. R. Sprague, B. J. Sussman, J. Nunn, N. K. Langford, X.-M. Jin, T. Champion, P. Michelberger, K. F. Reim, D. England, D. Jaksch, and I. A. Walmsley, Science 334, 1253 (2011).

[61] J. Hofmann, M. Krug, N. Ortegel, L. Gérard, M. Weber, W. Rosenfeld, and H. Weinfurter, Science 337, 72 (2012).   
[62] H. Bernien, B. Hensen, W. Pfaff, G. Koolstra, M. S. Blok, L. Robledo, T. H. Taminiau, M. Markham, D. J. Twitchen, L. Childress, and R. Hanson, Nature 497, 86 (2013).   
[63] R. Riedinger, A. Wallucks, I. Marinkovic, C. Löschnauer, M. Aspelmeyer, S. Hong, and S. Gröblacher, Nature 556, 473 (2018).   
[64] C. H. Bennett, G. Brassard, C. Crépeau, R. Jozsa, A. Peres, and W. K. Wootters, Phys. Rev. Lett. 70, 1895 (1993); S. Pirandola, J. Eisert, C. Weedbrook, A. Furusawa, and S. L. Braunstein, Nature Photonics 9, 641 (2015); A. Furusawa, J. L. Sørensen, S. L. Braunstein, C. A. Fuchs, H. J. Kimble, and E. S. Polzik, Science 282, 706 (1998).   
[65] S. Barzanjeh, M. Abdi, G. J. Milburn, P. Tombesi, and D. Vitali, Phys. Rev. Lett. 109, 130503 (2012).   
[66] R. Valivarthi, I. Lucio-Martinez, A. Rubenok, P. Chan, F. Marsili, V. B. Verma, M. D. Shaw, J. Stern, J. A. Slater, D. Oblak, et al., Optics express 22, 24497 (2014).   
[67] S. Barzanjeh, S. Guha, C. Weedbrook, D. Vitali, J. H. Shapiro, and S. Pirandola, Phys. Rev. Lett. 114, 080503 (2015).   
[68] M. Aspelmeyer, T. J. Kippenberg, and F. Marquardt, Rev. Mod. Phys. 86, 1391 (2014).   
[69] R. Riedinger, S. Hong, R. A. Norte, J. A. Slater, J. Shang, A. G. Krause, V. Anant, M. Aspelmeyer, and S. Gröblacher, Nature 530, 313 (2016); S. Hong, R. Riedinger, I. Marinković, A. Wallucks, S. G. Hofer, R. A. Norte, M. Aspelmeyer, and S. Gröblacher, Science 358, 203 (2017).   
[70] In the over-coupled regime with low thermal noise, the squeezing parameter is $\lambda = \frac{1}{2}\ln \frac{1 + (\sqrt{C_{\mathrm{om}}} + \sqrt{C_{\mathrm{em}}})^2}{1 + (\sqrt{C_{\mathrm{om}}} - \sqrt{C_{\mathrm{em}}})^2}$ . See details from the discussion of entanglement of formation in the supplemental material.   
[71] S. J. Freedman and J. F. Clauser, Phys. Rev. Lett. 28, 938 (1972).   
[72] A. Aspect, P. Grangier, and G. Roger, Phys. Rev. Lett. 47, 460 (1981); Phys. Rev. Lett. 49, 91 (1982).   
[73] J. G. Rarity and P. R. Tapster, Phys. Rev. Lett. 64, 2495 (1990).   
[74] I. Marcikic, H. de Riedmatten, W. Tittel, V. Scarani, H. Zbinden, and N. Gisin, Phys. Rev. A 66, 062308 (2002); P. Kurpiers, M. Pechal, B. Royer, P. Magnard, T. Walter, J. Heinsoo, Y. Salathé, A. Akin, S. Storz, J.-C. Besse, et al., arXiv preprint arXiv:1811.07604 (2018).

[75] S. Ramelow, L. Ratschbacher, A. Fedrizzi, N. K. Langford, and A. Zeilinger, Phys. Rev. Lett. 103, 253601 (2009); L. Olislager, J. Cussey, A. T. Nguyen, P. Emplit, S. Massar, J.-M. Merolla, and K. P. Huy, Phys. Rev. A 82, 013804 (2010).   
[76] J. D. Franson, Phys. Rev. Lett. 62, 2205 (1989).   
[77] W. Tittel, J. Brendel, H. Zbinden, and N. Gisin, Phys. Rev. Lett. 84, 4737 (2000).   
[78] H. Paik, A. Mezzacapo, M. Sandberg, D. T. McClure, B. Abdo, A. D. Córcoles, O. Dial, D. F. Bogorin, B. L. T. Plourde, M. Steffen, A. W. Cross, J. M. Gambetta, and J. M. Chow, Phys. Rev. Lett. 117, 250502 (2016).   
[79] M. Hatridge, S. Shankar, M. Mirrahimi, F. Schackert, K. Geerlings, T. Brecht, K. M. Sliwa, B. Abdo, L. Frunzio, S. M. Girvin, R. J. Schoelkopf, and M. H. Devoret, Science 339, 178 (2013).   
[80] D. Ristè, M. Dukalski, C. A. Watson, G. de Lange, M. J. Tiggelman, Y. M. Blanter, K. W. Lehnert, R. N. Schouten, and L. DiCarlo, Nature 502, 350 (2013).   
[81] S. Shankar, M. Hatridge, Z. Leghtas, K. M. Sliwa, A. Narla, U. Vool, S. M. Girvin, L. Frunzio, M. Mirrahimi, and M. H. Devoret, Nature 504, 419 (2013); Y. Liu, S. Shankar, N. Ofek, M. Hatridge, A. Narla, K. M. Sliwa, L. Frunzio, R. J. Schoelkopf, and M. H. Devoret, Phys. Rev. X 6, 011022 (2016).   
[82] J. Lisenfeld, G. J. Grabovskij, C. Müller, J. H. Cole, G. Weiss, and A. V. Ustinov, Nature communications 6, 6182 (2015).   
[83] C. E. Chase, Phys. Rev. 127, 361 (1962).   
[84] See supplemental material for the derivations of the covariance matrix $\mathbf{V}_{\mathrm{eo}}^{\mathrm{out}}$ , the entanglement of formation $E_{\mathrm{F}}$ , the lower bound of Bell state fidelity $F_{\mathrm{lb}}$ , and the $S(\varphi_{\mathrm{o}},\varphi_{\mathrm{e}};\varphi_{\mathrm{o}}^{\prime},\varphi_{\mathrm{e}}^{\prime})$ in the CHSH equality.   
[85] For a given state $\hat{\rho}$ , its purity is defined as $\mu_{\mathrm{P}} = \mathrm{tr}(\hat{\rho}^2)$ . The purity equals to one for a pure output state.   
[86] E. X. DeJesus and C. Kaufman, Phys. Rev. A 35, 5288 (1987).   
[87] Y.-D. Wang, S. Chesi, and A. A. Clerk, Phys. Rev. A 91, 013807 (2015).   
[88] L. Tian, Phys. Rev. Lett. 110, 233602 (2013).   
[89] J. F. Clauser, M. A. Horne, A. Shimony, and R. A. Holt, Phys. Rev. Lett. 23, 880 (1969).   
[90] R. J. Glauber, Phys. Rev. 130, 2529 (1963); Phys. Rev. 131, 2766 (1963).   
[91] A. Rueda, F. Sedlmeir, M. C. Collodo, U. Vogl, B. Stiller, G. Schunk, D. V. Strekalov, C. Marquardt, J. M. Fink, O. Painter, G. Leuchs, and H. G. L. Schwefel, Optica 3, 597 (2016).