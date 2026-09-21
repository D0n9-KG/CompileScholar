# Quantum Acoustics with Superconducting Qubits in the Multimode Transition-Coupling Regime

Li Li, $^{1,2,*}$ Xinhui Ruan, $^{1,3,4,*}$ Si-Lu Zhao, $^{1,2}$ Bing-Jie Chen, $^{1,2}$ Gui-Han Liang, $^{1,2}$ Yu Liu, $^{1,2}$ Cheng-Lin Deng, $^{1,2}$ Wei-Ping Yuan, $^{1,2}$ Jia-Cheng Song, $^{1,2}$ Zheng-He Liu, $^{1,2}$ Tian-Ming Li, $^{1,2}$ Yun-Hao Shi, $^{1}$ He Zhang, $^{1,2}$ Ming Han, $^{1,2}$ Jin-Ming Guo, $^{1,2}$ Xue-Yi Guo, $^{5}$ Qianchuan Zhao, $^{3}$ Jing Zhang, $^{6,7}$ Pengtao Song, $^{6,7}$ Xiaohui Song, $^{1,8}$ Kai Xu, $^{1,5,8}$ Heng Fan, $^{1,2,5,8}$ Yu-Xi Liu, $^{9}$ Zhihui Peng, $^{4,8,\dagger}$ Zhongcheng Xiang, $^{1,8,\ddagger}$ and Dongning Zheng $^{1,2,8,\S}$

$^{1}$ Beijing National Laboratory for Condensed Matter Physics, Institute of Physics, Chinese Academy of Sciences, Beijing 100190, China $^{2}$ School of Physical Sciences, University of Chinese Academy of Sciences, Beijing 100049, China $^{3}$ Department of Automation, Tsinghua University, Beijing 100084, P. R.

China $^{4}$ Key Laboratory of Low-Dimensional Quantum Structures and Quantum Control of Ministry of Education, Key Laboratory for Matter Microstructure and Function of Hunan Province, Department of Physics and Synergetic Innovation Center for Quantum Effects and Applications, Hunan Normal University, Changsha 410081, People's Republic of China $^{5}$ Beijing Academy of Quantum Information Sciences, Beijing 100193, China $^{6}$ School of Automation Science and Engineering, Xi'an Jiaotong University, Xi'an 710049, China $^{7}$ MOE Key Lab for Intelligent Networks and Network Security, Xi'an

Jiaotong University, Xi'an 710049, China $^{8}$ Hefei National Laboratory, Hefei 230088, China $^{9}$ School of Integrated Circuits, Tsinghua University, Beijing 100084, China (Dated: May 9, 2025)

Hybrid mechanical-superconducting systems for quantum information processing have attracted significant attention due to their potential applications. In such systems, the weak coupling regime, dominated by dissipation, has been extensively studied. The strong coupling regime, where coherent energy exchange exceeds losses, has also been widely explored. However, the transition-coupling regime, which lies between the above two and exhibits rich, unique physics, remains underexplored.

In this study, we fabricate a tunable coupling device to investigate the coupling of a superconducting transmon qubit to a seven-mode surface acoustic wave resonator (SAWR), with a particular focus on the transition-coupling regime. Through a series of phonon oscillation experiments and studies in the dispersive regime, we systematically characterize the performance of the SAWR. We then explore the complex dynamics of energy exchange between the qubit and the mechanical modes, highlighting the interplay between dissipation and coherence.

Finally, we propose a protocol for qubit readout and fast reset with a multimode mechanical cavity using one mode for readout and another mode for reset. We have demonstrated in simulation that the qubit achieves both fast reset and high coherence performance when the qubit is coupled to the reset mode in the transition-coupling regime.

# I. INTRODUCTION

Quantum acoustics, the study of quantum phenomena in phononic systems, is rapidly advancing with applications in quantum information processing [1-3]. Phononic systems offer advantages including longer coherence times [4-7] and higher mode density [8]. These properties enable diverse applications, such as microwave-optical conversion [9-13], quantum registers [14], microwave amplifiers [15], and feedback networks [16], highlighting the potential of phononic systems in next-generation quantum technologies. Recently, various hybrid quantum acoustic systems have been explored, including mechanical oscillator [17-19], surface acoustic

wave resonator [20-22], and bulk acoustic wave resonator [23-25]. These systems have been coupled to superconducting circuits [26], optomechanical setups [27], and spin systems [28], providing versatile platforms for studying quantum behaviors through phononic interactions. Among them, circuit quantum acoustodynamics (cQAD), which involves coupling superconducting qubits with acoustic resonators, is an important platform in the field. This approach combines the coherence of phononic systems with the high coherence [29], controllability [30], and scalability [31] of superconducting qubits, making cQAD systems promising candidates for quantum state transfer [32, 33] and quantum random access memories [34, 35].

In cQAD, the coupling between superconducting qubits and mechanical resonators has allowed researchers to explore the quantum mechanical properties of the phononic system. Earlier studies concentrated predominantly on the weak-coupling or strong-coupling regimes, where the dynamics is well-established and well-

arXiv:2505.05127v1 [quant-ph] 8 May 2025

* These authors contributed equally to this work.

† zhihui.peng@hunnu.edu.cn

‡ zcxiang@iphy.ac.cn

$\S$ dzheng@iphy.ac.cn

understood. The weak-coupling regime enables the study of phonon-mediated decoherence [27, 36, 37]. In contrast, the strong-coupling regime enables coherent energy exchange between qubits and mechanical resonators, thus allowing the control of quantum acoustic states [23, 26] and qubit-driven phonon interactions [25, 38]. However, the intermediate regime, also known as transition-coupling regime in cQAD, which bridges the weak- and strong-coupling limits, has remained largely unexplored.

This regime is particularly intriguing, as it enables the exploration of rich and complex dynamics, where both dissipative effects and coherent interactions come into play. For example, a quantum emitter in the transition-coupling regime will exhibit an additional bandwidth modulation degree of freedom [39]. On the other hand, by utilizing the properties of transition-coupling regime to balance coherent coupling and dissipation, the sensitivity of quantum sensing can be enhanced [40, 41].

Furthermore, the physics of exceptional points (EPs) based on transition-coupling regime continues to be applied in various domains of non-Hermitian physics [42-46]. Researches on the transition-coupling regime and its implementation in cQAD are highly meaningful.

In this study, we employed flip-chip technologies to fabricate a superconducting transmon qubit and a seven-mode SAWR with a tunable coupling between them. We began with detailed microstructural measurements by phonons oscillation experiment to verify the integrity and alignment of the bonded device, ensuring acoustic coupling between the qubit and the SAWR. Subsequently, we examined the AC Stark effect within the dispersive regime to characterize phononic interactions and probed the decay rate of individual mechanical modes.

Finally, we extended our investigation to the transition-coupling regime, where resonant evolution allowed us to explore the dynamics of energy exchange between the qubit and phononic modes. We observe a transition from underdamped behavior with coherent oscillations to overdamped behavior characterized by decay without oscillations, depending on the mode number and the coupler bias. We demonstrated that a multimode cavity enables both dispersive readout and fast reset of a qubit.

When the fast reset mode is in the transition-coupling regime with the qubit, high reset efficiency is achieved while the induced qubit decoherence remains negligible. This approach not only highlights the practical utility of the transition-coupling regime but also provides new insights for optical readout, contributing to the design of future large-scale superconducting quantum chips.

By systematically studying the dynamics that depend on both the modes and the coupler, our work provides insights into the interplay between dissipative and coherent interactions in quantum acoustics, thereby advancing the development of robust and tunable quantum systems.

# II. THE DEVICE

In
this experiment, we designed a multi-mode SAWR and coupled it to a transmon qubit through flip-chip technology. The transmon qubit and its control lines are fabricated on the sapphire substrate, shown in Fig. 1(a). The SAWR is a 2D Fabry-Perot cavity formed by IDTs and Bragg mirrors and fabricated on the $128^{\circ}\mathrm{Y}-$ X $\mathrm{LiNbO_3}$ substrate, as shown in Figs. 1(b)-(c) and (e)-(f). In order to directly control the surface acoustic wave (SAW) modes, we additionally integrated two input/output IDTs (IDT1 and IDT3 in Fig. 2(a)) into the SAWR. The middle IDT (IDT2 in Fig.

2(a)) is used to perform energy conversion between the SAW and the qubit. After alignment and flip-chip process, there is an inter-chip mutual inductance caused by the overlapping between the inductive line of the IDT2 and the corresponding part of the bottom chip. We use a gmon as the coupler to tune the coupling strength between SAWR and the qubit by applied voltage bias through gmon bias line [47]. The fabrication processes of this device are the same as our former work [48].

The period $p$ of the IDTs is $864\mathrm{nm}$ , corresponding to the centre frequency of the SAWR around $4.6\mathrm{GHz}$ . The length and thick of all the electrodes are $75\mu \mathrm{m}$ and $30\mathrm{nm}$ , respectively. The width of IDT1, 3 and Bragg mirrors is $p / 4$ , while that of IDT2 is $p / 6$ and each cell of IDT2 has three fingers (Fig. 1(f)). This design aims to minimize the reflection effect of IDT2 on the acoustic wave. The cell numbers of the three IDTs are 5, 10 and 5, respectively, while each mirror has 400 periodic stripe electrodes. The distance between the mirrors is $50\times p = 50\mu \mathrm{m}$ . Unlike single-IDT resonator, the modes in multi-IDT resonator are more complex.

The transmon qubit has a DC SQUID and its transition frequency can be tuned through changing the external flux applied to the SQUID. The transition frequency of the qubit under different Z bias is shown in Fig. 1(d).

# III. PHONONIC RESPONSE OF SAWR

The schematic of the SAWR is shown in Fig. 2(a). We conducted measurements in a commercial dilution refrigerator, with a base temperature of around $12\mathrm{mK}$ . By performing transmission spectrum measurements across the two IDT ports of the SAWR, we can obtain its resonant frequencies, as illustrated in Fig. 2(b). The $S_{21}$ spectrum reveals seven distinct peaks. By fitting each peak, the corresponding resonant frequencies, $f_{m}$ , are determined as summarized in Tab. I. The non-ideal IDTs caused asymmetric reflection of SAW [49] leading to irregularities in certain peak profiles. Additionally, impedance mismatch and transmission line response contribute to the asymmetry observed in some peak shapes.

Assuming the device is symmetric, we can calculate the SAWs propagation speed on the $128^{\circ}\mathrm{Y - X}$ $\mathrm{LiNbO_3}$ substrate as $v_{e} = pf_{\mathrm{m4}}\approx 3938\mathrm{m / s}$ , where $f_{\mathrm{m4}} = 4.5576\mathrm{GHz}$

2

![](dt=2026-06-04/ht=19/a38d5b01df4e4c328e66112417a206c028ea3aa344beb4bbd0ef7aa99f217b5a.jpg)

![](dt=2026-06-04/ht=19/1a56c986625edf1e08fcdfe285f89fd449f84bb9f47ffa09b9adfe43870dad8d.jpg)

![](dt=2026-06-04/ht=19/e811b07c19cfb79563ddc5cb3cdcaad53283151f3db8c1869b34a8b8dcf08453.jpg)

![](dt=2026-06-04/ht=19/ba29c41b0dd2ef7914c9b739dc0d365b6caf2f2a067af98ab74b038d9973709e.jpg)

![](dt=2026-06-04/ht=19/adc610f2ddf3d4e6cceb09d94078920a3ebf5da12acef5a69b224cef0bb6625b.jpg)

![](dt=2026-06-04/ht=19/10809f62ad79de9f7f4e56c01a2bffd2d3c7e4fe10b1df5c839938fe0ce98660.jpg)

is the frequency of the central mechanical mode. To obtain the structural parameters of the SAWR, we conducted a phonon oscillation experiment [20] using the central mechanical mode. A short pulse was applied at Port1 ( $f = f_{\mathrm{m4}}$ ), and the time-dependent response of the electrical signal was monitored at Port2. The pulse duration satisfies $t \leq 2L_{c} / v_{e}$ , where $L_{c}$ is the effective cavity length of the SAWR. The experimental results are shown in Fig. 2(c). Due to the propagation and the reflection by the Bragg grating, the SAW signal captured by the IDT at Port2 exhibits oscillatory decay.

To improve the efficiency of microwave-acoustic conversion, Gaussian pulses were employed in the experiment (Appendix B). Pulses of varying lengths were used to obtain the most accurate structural parameters, as indicated by the color transition from blue to brown in Fig. 2(c). When the pulse length was reduced from 30 ns to 12 ns, a distinct downward oscillation emerged at the first rising edge of the curve. This oscillation was caused by SAW propagating through IDT2, reflecting off the Bragg grating on the right, and passing through IDT2 again, as illustrated in the inset of Fig. 2(c).

The ADC sampling rate employed in our experiments

was $1\mathrm{GSa / s}$ , achieving a time resolution of $1\mathrm{ns}$ . The time between the first rising peak and the subsequent falling peak named $\Delta t_{1}$ was measured to be $3\sim 4\mathrm{ns}$ , allowing the calculation of $d_0 = d_1 + L_p = \Delta t_1v_e / 2 = 5.91\sim 7.88\mu \mathrm{m}$ where $L_{p}$ represents the penetration depth and $d_{1}$ is the distance between IDT2 and the Bragg grating, measured to be $2.2\mu \mathrm{m}$ .

Consequently, $L_{p}$ is determined to be within the range of $3.71\sim 5.68\mu \mathrm{m}$ , corresponding to a single-electrode reflectivity of $r_s = p / (4L_p)$ , which falls between 0.038 and 0.058. These values align well with the design parameters of $L_{c} = 4.65\mu \mathrm{m}$ and $r_s = 0.0473$ . Furthermore, the spacing between oscillation peaks corresponds to twice of the effective cavity length, such that $L_{c} = \Delta t_{2}v_{e} / 2 = 53.2\mu \mathrm{m}$ matches the design value of $53.3\mu \mathrm{m}$ .

Experimental data for various modes, pulse shapes, and power levels in phonon oscillation experiments are provided in the Appendix B.

3

TABLE I. The parameters of mechanical modes.

![](dt=2026-06-04/ht=19/22dfa11a50aab483e3e94842f73dc1d4ca71f12e28b01ef8fabe5ba79a7e034b.jpg)

<table><tr><td>Quantities</td><td>Symbols</td><td>Units</td><td colspan="7">Values</td></tr><tr><td>Mode</td><td>m</td><td></td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>Resonant Frequency</td><td>fm</td><td>GHz</td><td>4.4413</td><td>4.4855</td><td>4.5247</td><td>4.5576</td><td>4.5874</td><td>4.6243</td><td>4.6660</td></tr><tr><td>Phonon lifetime</td><td>T1m</td><td>ns</td><td>73.5</td><td>98.8</td><td>58.3</td><td>127.6</td><td>233.1</td><td>204.2</td><td>105.8</td></tr><tr><td>Decay rate</td><td>κm/2π</td><td>MHz</td><td>2.17</td><td>1.61</td><td>2.73</td><td>1.25</td><td>0.68</td><td>0.78</td><td>1.50</td></tr><tr><td>Quality factor</td><td>Qm</td><td></td><td>2050</td><td>2783</td><td>1657</td><td>3654</td><td>6716</td><td>5933</td><td>3102</td></tr></table>

![](dt=2026-06-04/ht=19/25855bf1dca78c2508b2401e2aac0d886b0ffc3f2aba32fb544b9ecd00a64467.jpg)

![](dt=2026-06-04/ht=19/7107b88ab1f1de66466d6eaf67179a4987d95d6bbe0a125598d5cab8aee28cbe.jpg)

![](image)
7c83f24b8461fac29bf4f0ab2839db4.jpg)

# IV. DISPERSIVE ACOUSTIC COUPLING: THE ACOUSTIC AC-STARK EFFECT

In this section, we measure the AC Stark effect of the qubit in the dispersive regime. Initially, the qubit is biased to its idle point via the Z control line, with a transi

tion frequency $f_{01}^{\mathrm{idle}} = 4.7685\mathrm{GHz}$ . Subsequently, the frequencies of the applied pulses for both the qubit and the SAW are scanned simultaneously. When the pump frequency sweeps across the resonant frequencies of the mechanical modes, the qubit, which is dispersively coupled to the resonator, induces an AC Stark shift. This results in a series of Lorentzian profiles in the two-dimensional scan, as depicted in Fig. 3(a).

In the case of multi-mode cavity, the AC Stark shift of the qubit includes contributions from individual modes and cross terms arising from qubit-induced virtual phonon exchanges between nondegenerate modes [50]. However, under conditions of low phonon numbers and large detuning, the effects of these cross interactions can be neglected [8, 20, 51]. At the same time, due to the monochromaticity of the pulse frequency, the pump excites phonons only in a single mode during the scan.

Drawing an analogy from cQED [52], the AC Stark shift induced by $n$ -phonons in m-th mode is given by $\delta \omega_{q} \equiv 2\chi_{\mathrm{m}}\langle n\rangle$ where

$$
\chi_ {\mathrm {m}} = - \frac {g ^ {2} E _ {c} / \hbar}{\Delta_ {\mathrm {m}} (\Delta_ {\mathrm {m}} - E _ {c} / \hbar)}, (1)
$$

$\Delta_{\mathrm{m}} / 2\pi = f_{01} - f_{m}, E_{c} / 2\pi \hbar \equiv f_{01} - f_{12} = 171\mathrm{MHz}$ and $\langle n\rangle$ is the average phonon number in m-th SAW mode. When $\Delta_{\mathrm{m}} > 0$ and $\Delta_{\mathrm{m}} - E_{c} < 0$ , specifically when $f_{12} < f_{m} < f_{01}$ , the system is operated in the straddling dispersive regime, resulting in a positive frequency shift [52-54], as illustrated in Fig. 3(b).

By setting the SAW driving frequency to each of the seven resonant frequencies in turn, we varied the driving power and observed a linear relationship between the AC Stark shift and the driving power, as shown in Fig. 3(c). This relationship enables us to determine the range of driving powers that satisfy the low phonon number condition [55] required for the dispersive regime.

Using the qubit as a probe, we can monitor variations in the phonon number within the resonator by adjusting the delay between the XY pulse and the SAW drive. These variations are reflected in changes to the qubit transition frequency during the SAW drive, as illustrated in Fig. 3(d). The variation in phonon number occurs in three distinct stages: during the charging phase, the coherent drive rapidly increases the phonon number, which then stabilizes at a steady value due to the balance between pumping and dissipation. Once the drive is removed, the number of phonons undergoes rapid decay.

4

![](dt=2026-06-04/ht=19/46cf6f71707cb68dcc7137e02a56bba2382d9ba615610a0964ea36db6cddf4c2.jpg)

![](dt=2026-06-04/ht=19/f1e4aad322853e7281304752a69051b8d5ec0970c2bdc09634a5f738e4d116aa.jpg)

![](dt=2026-06-04/ht=19/138324d29fea535059b2fa1918b6a279bb51a225ac2a40da2bc74af436649ae5.jpg)

![](dt=2026-06-04/ht=19/650b0eadbad950eed24ee6bd1ab22a723605b9da51ff4c19732dc8730051fa82.jpg)

![](dt=2026-06-04/ht=19/7cfe910804680c62a644dea41fb7484655bd52bf4b4568beafe10c8cb6aa429f.jpg)

The duration of the drive pulse is $t_{\mathrm{d}} = 2\mu \mathrm{s}$ corresponding to the start time of decay. Data collected beyond $t_{\mathrm{d}}$ are normalized as $\delta f_{\mathrm{q}} / \delta f_{\mathrm{q}}^{\mathrm{max}}$ for ease of comparison, giving Fig. 3(e). By fitting these decay curves, we extract the phonon lifetimes, which are summarized in Tab. I.

# V. REASONTE ACOUSTIC COUPLING: DYNAMICS IN TRANSITION-COUPLING REGIME

In this section, we examine the qubit dynamics when it is resonant with one of the modes in the multimode cavity. The schematic of the model is shown in Fig. 4(a). The qubit transition frequency is given by $\omega_{q} = 2\pi f_{q}$ with an intrinsic dissipation rate $\gamma$ while the resonant mode frequency is $\omega_{r} = 2\pi f_{r}$ , with a dissipation rate $\kappa$ . As schematically illustrated in Fig. 4(b), in the weak coupling regime ( $g \ll \gamma, \kappa$ ), the population of the qubit's excited state undergoes a non-oscillatory decay over time.

In the strong coupling regime ( $g \gg \gamma, \kappa$ ), pronounced Rabi oscillations emerge, indicating periodic energy exchange between the qubit and the mechanical mode. As the system enters the ultrastrong coupling regime ( $g \gtrsim$

$0.1\omega_{q / r})$ , the breakdown of the rotating-wave approximation leads to an increased oscillation frequency and an asymmetric waveform. In the deep strong coupling regime $(g\gtrsim \omega_{q / r})$ , the oscillations become irregular, the system rapidly reaches a steady state, and the steadystate population of the qubit remains nonzero [56].

Fig. 4(c) shows the variation of the qubit transition frequency as a function of the gmon bias. By adjusting the bias of the gmon, both the coupling strength between the qubit and the mechanical modes and the qubit transition frequency are modulated [57]. We use an in-situ calibration to prevent the qubit frequency from being affected by the gmon bias as described in Appendix C. All AC Stark experiments shown in Fig. 3 were performed with the gmon bias set to $V_{g} = -3.0$ . Fig. 4(d) shows how the energy relaxation time $T_{1}$ of the qubit at the idle point varies as a function of the gmon bias.

We model the dissipation of the qubit at the idle point as being influenced by the Purcell effect from seven mechanical modes. The qubit's decay rate is given by [52]

$$
\gamma_ {q} ^ {\mathrm {i d l e}} = \sum_ {m = 1} ^ {7} \left(\frac {g _ {m}}{\Delta_ {m}}\right) ^ {2} \kappa_ {m} + \gamma_ {0} ^ {\mathrm {i d l e}}, \tag {2}
$$

5

where $g_{m}$ is the coupling strength between the qubit and the mechanical mode- $m$ , $\Delta_{m} / 2\pi = f_{m} - f_{q}$ and $\gamma_0^{\mathrm{idle}}$ is the intrinsic decay rate of the qubit at the idle point. Fitting the qubit decay data results in the solid black line shown in Fig. 4(d). As demonstrated in Eq. (2), the qubit's decoherence rate is maximized (minimized) when the effective coupling strength between the qubit and the mechanical mode is at its maximum (minimum).

This approach allows us to identify both the off point $(V_{g} = -0.9)$ and the maximum coupling point $(V_{g} = 0.75)$ of the gmon coupler, even within the dispersive regime. After decoupling from the mechanical modes, the qubit's $T_{1}$ at the idle point increases to approximately $15\mu s$ . Additionally, the $T_{1}$ of the qubit improves significantly across the entire frequency range after decoupling, as shown in Appendix D.

Finally, we conducted resonant evolution measurements with $V_{g} = -3.0$ , $V_{g} = -0.9$ and $V_{g} = 0.75$ , respectively. A $\pi$ -pulse was applied to excite the qubit to the $|e\rangle$ state, followed by a fast Z-pulse to align the qubit's frequency with a specific mode for the evolution during a delay time. The gmon bias was maintained at $V_{g}$ through its fast Z-control, as illustrated in Fig. 4(e). The obtained population dynamics of the qubit during resonant evolution with mode-6 are shown in Fig. 4(f). Figure 4(g) presents the resonant evolution of the qubit with seven different modes, where the blue heatmap corresponds to $V_{g} = 0.75$ and the orange heatmap corresponds to $V_{
g} = -3.0$ .

To analyze the data in Figs. 4(f)-(g), it is essential to accurately understand how the dissipation of the qubit and multiple cavity modes influences the dynamics of the resonant evolution. During this evolution, the behavior of the entire system can be described by the Lindblad master equation

$$
\frac {d \rho}{d t} = - \frac {i}{\hbar} [ H, \rho ] + \sum_ {k = q, m} (2 L _ {k} \rho L _ {k} ^ {\dagger} - \{L _ {k} ^ {\dagger} L _ {k}, \rho \}), \tag {3}
$$

$$
\begin{array}{l} \frac {H}{\hbar} = 2 \pi f _ {q} b ^ {\dagger} b - \frac {E _ {c}}{2} b ^ {\dagger} b ^ {\dagger} b b \\ + \sum_ {m} \left[ 2 \pi f _ {m} a _ {m} ^ {\dagger} a _ {m} + g _ {m} \left(b a _ {m} ^ {\dagger} + b ^ {\dagger} a _ {m}\right) \right], \tag {4} \\ \end{array}
$$

$$
L _ {q} = \sqrt {\gamma_ {q}} b, L _ {m} = \sqrt {\kappa_ {m}} a _ {m}, \tag {5}
$$

where $\gamma_{q}$ is a variable related to the qubit transition frequency. In our experiment, whenever the qubit is resonant with one of the modes, it remains in a dispersive regime with the other six modes. Therefore, the influence of these six detuned modes can be treated as an additive Purcell effect, modeled using a standard dispersive transformation [58]. Using this insight and truncating Eq. (4) to the two lowest levels of the transmon, the master equa

tion can be equivalently rewritten as:

$$
\begin{array}{l} \frac {d \rho}{d t} = - \frac {i}{\hbar} [ H _ {\mathrm {d i s p}}, \rho ] + (2 L _ {q} \rho L _ {q} ^ {\dagger} - \{L _ {q} ^ {\dagger} L _ {q}, \rho \}) \\ + \left(2 L _ {m} \rho L _ {q} ^ {\dagger} - \left\{L _ {m} ^ {\dagger} L _ {m}, \rho \right\}\right), \tag {6} \\ \end{array}
$$

$$
\begin{array}{l} \frac {H _ {\mathrm {d i s p}}}{\hbar} = - \frac {2 \pi f _ {q}}{2} \sigma_ {z} + 2 \pi f _ {m} a _ {m} ^ {\dagger} a _ {m} + g _ {m} (\sigma_ {+} a _ {m} ^ {\dagger} + \sigma_ {-} a _ {m}) \\ + \sum_ {n \neq m} \left(2 \pi f _ {n} a _ {n} ^ {\dagger} a _ {n} + \chi_ {n} \sigma_ {z} a _ {n} ^ {\dagger} a _ {n}\right), \tag {7} \\ \end{array}
$$

$$
L _ {q} = \sqrt {\gamma_ {q} ^ {\prime}} \sigma_ {-}, L _ {m} = \sqrt {\kappa_ {m}} a _ {m}, \tag {8}
$$

where $\chi_{n}$ is the dispersive shift of the qubit caused by the $n$ -th mode and $\gamma_q^\prime$ includes the Purcell effect from the remaining six modes:

$$
\gamma_ {q} ^ {\prime} = \gamma_ {q} + \sum_ {n \neq m} \left(\frac {g _ {n}}{\Delta_ {n}}\right) ^ {2} \kappa_ {n}. \tag {9}
$$

Through the steps outlined above, we have incorporated the influence of the non-resonant modes. Next, we will consider the dynamics of resonant evolution by considering the frequency resonance conditions $f_{q} = f_{m}$ . This is done by performing a rotating frame transformation on Eq. (7) and applying the rotating wave approximation, yielding:

$$
\frac {H _ {\text {d i s p}} ^ {\mathrm {R W A}}}{\hbar} = g _ {m} \left(\sigma_ {+} a _ {m} ^ {\dagger} + \sigma_ {-} a _ {m}\right), \tag {10}
$$

which gives the standard Jaynes-Cummings interaction Hamiltonian. Focusing only on the two lowest energy levels of the mode, the Hamiltonian in the single-excitation subspace is

$$
\frac {H ^ {\prime}}{\hbar} = g _ {m} \left(\left| e, 0 \right\rangle_ {m} \left\langle g, 1 \right| _ {m} + \left| g, 1 \right\rangle_ {m} \left\langle e, 0 \right| _ {m}\right), \tag {11}
$$

where $|e\rangle (|g\rangle)$ denotes the first excited (ground) state of the qubit, and $|n\rangle_{m}$ denotes the $n$ photon Fock state of the mode- $m$ . By substituting this Hamiltonian into the master equation in the interaction picture, we can derive the expression for the density matrix $\rho_{S}$ . The probability of the qubit being in the $|e\rangle$ state can be written as:

$$
\begin{array}{l} P _ {e} (t) = \operatorname {T r} \left[ \rho_ {S} (t) | e \rangle \langle e | \right] = f (t) \cdot \cos \left(g _ {m} t\right) ^ {2}, \\ f (t) = \exp \left[ - t \left(\frac {\gamma_ {q} ^ {\prime} + \kappa_ {m}}{2}\right) + \left(\frac {\kappa_ {m} - \gamma_ {q} ^ {\prime}}{4 g _ {m}}\right) \sin \left(2 g _ {m} t\right) \right]. \tag {12} \\ \end{array}
$$

From Eq. (12), as $g_{m}$ approaches zero in the weak coupling regime, the exponential term simplifies to $-\gamma_q' t$ . In the strong coupling regime, when $g_{m} \gg \gamma_q'$ , $\kappa_{m}$ , the effects of the oscillatory term in the exponential can be neglected. However, when $g_{m}$ is comparable to $|\gamma_q' - \kappa_m|$ , i.e., in the transition-coupling regime, the system exhibits distinct dynamics.

Using the insight of Eq. (12) and $\gamma_{q}$ obtained from

6

![](dt=2026-06-04/ht=19/e8e030ab01599c8c700e2b449707173c63a777f6fca3f9710f5713a08035e51d.jpg)

![](dt=2026-06-04/ht=19/f04ab2a22a4fce19629710f95f445f53d6343f2d3597c0beeee37e13bb527553.jpg)

![](dt=2026-06-04/ht=19/ae667333b06a19ebd854cf2c82c633952bbf1c799bdee934b008e026a6067377.jpg)

![](dt=2026-06-04/ht=19/1f6837676cb5711a2be58730c8d50070f444898dade12d588105b0dcfec2bdc0.jpg)

![](dt=2026-06-04/ht=19/8ff34aba446609dd886c84d9da96aa76e63ba8e28c1423bb6589282e9262384f.jpg)

![](dt=2026-06-04/ht=19/2a463f553de4e23f7944c8c265a2007600245e4a6e50c9e6899c75128521fdfb.jpg)

![](dt=2026-06-04/ht=19/26e644b3edbc809bc9ab495a8a2ecea8de10a25d44cee3f3e6f6ca1b5f66461f.jpg)

![](dt=2026-06-04/ht=19/d16cacd4b8c1e4dbc100621cd28a3af9ee0d5b3628280fa7ba421a30601e30a4.jpg)

TABLE II. The parameters of the qubit under different gmon biases. The $\gamma_{q}$ $(\gamma_q^{\prime})$ represents the intrinsic (total) decay rate of the qubit at the transition frequency $f_{m}$ . The $g_{m}$ represents the coupling strength between the qubit and the $m$ -th mechanical mode.

![](dt=2026-06-04/ht=19/afcc46db4f5c2a021babbf31c4c264f2f78c73d7de7e23174e1539ec70b99942.jpg)

<table><tr><td>Gmon bias</td><td>Quantities</td><td>Symbols</td><td>Units</td><td>Mode1</td><td>Mode2</td><td>Mode3</td><td>Mode4</td><td>Mode5</td><td>Mode6</td><td>Mode7</td></tr><tr><td>Vg=-0.9</td><td>Intrinsic decay rate</td><td>γq/2π</td><td>kHz</td><td>19.6</td><td>21.8</td><td>31.8</td><td>12.4</td><td>12.2</td><td>12.2</td><td>17.6</td></tr><tr><td rowspan="2">Vg=-3.0</td><td>Coupling Strength</td><td>gm/2π</td><td>MHz</td><td>-0.18</td><td>-0.21</td><td>0.18</td><td>0.23</td><td>-0.18</td><td>-0.28</td><td>0.09</td></tr><tr><td>Total decay rate</td><td>γq&#x27;/2π</td><td>kHz</td><td>19.7</td><td>21.9</td><td>31.9</td><td>12.5</td><td>12.4</td><td>12.3</td><td>17.7</td></tr><tr><td rowspan="2">Vg=0.75</td><td>Coupling Strength</td><td>gm/2π</td><td>MHz</td><td>0.59</td><td>1.51</td><td>-0.75</td><td>-1.52</td><td>0.82</td><td>1.67</td><td>-0.47</td></tr><tr><td>Total decay rate</td><td>γq&#x27;/2π</td><td>kHz</td><td>22.0</td><td>23.9</td><td>37.3</td><td>15.6</td><td>17.9</td><td>13.7</td><td>19.3</td></tr></table>

the coupling-off evolution, we fit the resonant evolution data for the qubit interacting with each mechanical mode under different gmon bias. The results show good agreement with our model. It is worth noting that the fitted $\kappa_{m}$ is larger than the value listed in Tab. II. This discrep

ancy arises because the phonon population in the SAWR is higher during the AC Stark measurement, leading to an increased cavity quality factor $Q_{m}$ [4]. The coupling strengths are shown in Fi
g 4(h). It shows the mode-dependent coupling strengths of the qubit at the position

7

of $L_{c} / 2$ , following a sinusoidal variation pattern [8]:

$$
g _ {m} = g _ {0} \sin \left(\frac {\pi}{2} m + \phi_ {q}\right), \tag {13}
$$

where $g_{0}$ represents an overall coupling strength. Eq. (13) exhibits sinusoidal modulation with a period corresponding to seven modes, and $\phi_{q}$ denotes the overall phase shift resulting from a slight deviation of the qubit position from $L_{c} / 2$ .

It is important to note that the coupling strength obtained through the fitting of the resonant evolution only provides its absolute value, $|g|$ . The chosen value of the gmon bias determines the relative signs of the coupling strengths at $V_{g} = -3.0$ and $V_{g} = 0.75$ .

# VI. INTRINSIC RESET PROTOCOL IN TRANSITION-COUPLING REGIME

Fast reset of qubits—one of the DiVincenzo criteria [59]—has seen a series of advances in recent years. One major approach involves resonantly coupling the qubit level to be reset with an external dissipation source (such as the qubit's readout cavity or a Purcell filter) via a microwave field, achieving rapid population transfer and dissipation to the ground state [60-62]. However, this method requires precise calibration of multiple parameters, increasing the calibration overhead on large-scale integrated chips.

In this section, we propose a method that leverages the multimode characteristics of the mechanical cavity. One mode facilitates fast qubit reset, while another enables dispersive readout, leading to a more compact and scalable architecture for superconducting quantum chips. Through numerical simulations, we demonstrate that the coupling between the mechanical mode and the qubit achieves optimal performance in the transition-coupling regime.

Label the two modes of the SAWR as the dispersive readout mode and the reset mode, with frequencies $\omega_{d}$ and $\omega_{r}$ , dissipation rates $\kappa_{d}$ and $\kappa_{r}$ , and coupling strengths with the qubit $g_{d}$ and $g_{r}$ , respectively. As shown in Fig. 5(a), the reset process is implemented by applying a fast Z-bias to bring the qubit into resonance with the reset mode. Coherent evolution then proceeds until the remaining excitation reaches a predefined threshold. This duration is recorded as the reset time. According to the conclusions in Sec.

V, the dynamics of the qubit and mechanical mode in the resonant regime is governed by the interplay between the coupling strength and dissipation rate. A weak dissipation rate leads to a slow relaxation to the ground state, while excessively strong coupling induces pronounced oscillations, allowing photons to cycle back into the qubit and ultimately hindering the reset efficiency.

We use QuTiP [63, 64] for numerical simulations, and input experimentally relevant parameters: $\gamma /2\pi = 0.2\mathrm{MHz}$ and $\kappa_r / 2\pi = 2.5\mathrm{MHz}$ . The qubit is initialized

![](dt=2026-06-04/ht=19/e15c110d7e418c6f5370d4877d92865ad1e4cf5adf037b09c484ac170d2fe952.jpg)

![](dt=2026-06-04/ht=19/602d3587e1b2bbf5dc20c829541a1fce938af8a89554412e163e62637988dfbf.jpg)

![](dt=2026-06-04/ht=19/6c7b5fffacf83830404581ac5e271aace6bc55fa723a247ac5bb3d36dfa3ce47.jpg)

in the $|e\rangle$ state, and we vary the ratio of the coupling strength $g_{d}$ to the reset mode dissipation rate $\kappa_{r}$ over the range of $10^{-2}$ to $10^{2}$ to study the resonant evolution. Define the reset threshold as the population of the $|g\rangle$ state. The reset time is defined as the time after which

8

the population of the $|g\rangle$ state always remains above the threshold. Fig. 5(b) presents the simulation results, illustrating the reset time for different fidelity thresholds. In the weak coupling regime, the dissipation of the mechanical mode dominates the reset process, requiring a duration of at least $10 / \kappa$ to achieve the desired threshold. As the coupling strength increases, the reset time gradually decreases, and coherent oscillations begin to emerge. After reaching a minimum, the reset time stabilizes, with the minimum occurring around $g_{m}\sim \kappa_{m}$ . These results indicate that to ensure an efficient reset rate, the coupling strength must at least reach the transition-coupling regime.

On the other hand, we need to examine the impact of the reset mode on the qubit at the idle point, which is primarily manifested through the Purcell effect, $\gamma' = \gamma + \gamma_p$ , $\gamma_p = g_r^2 \kappa_r / \Delta_r^2$ . We calculate the dependence of $\gamma_p / \gamma$ on $g_r$ , set $\Delta_r / 2\pi = 300\mathrm{MHz}$ , with the results shown in Fig. 5(c). In the weak and transition-coupling regimes, the Purcell effect contributes negligibly to qubit decoherence.

As the system enters the strong coupling regime, the influence of the Purcell effect becomes significant, reaching approximately $10\%$ at the strong coupling boundary ( $g_r \sim 10\kappa_r$ ). Considering both the dissipation rate and the Purcell effect, the reset mode should be coupled to the qubit in the transition-coupling regime, where the reset process is maximized in speed while the decoherence impact on the qubit remains negligible.

For the dispersive readout mode, an estimate based on the optimal condition for dispersive readout, $2\chi = \kappa_{d}$ , suggests that achieving an obvious dispersive shift requires operation in the strong coupling regime with the qubit [52]. Due to this constraint, the intrinsic readout and reset experiment proposed in this section was not implemented in the multimode SWAR used in this work.

In future designs, we can leverage the sinusoidal dependence of coupling strength on mode index to engineer the system such that the dispersive readout mode operates in the strong coupling regime, while the reset mode remains in the transition-coupling regime, as illustrated in Fig. 5(a). Another solution is to design the mechanical mode frequencies below the qubit's idle point, utilizing the straddling dispersive coupling regime to improve qubit readout [65, 66].

Using a mechanical cavity for readout not only leverages its high mode density and quality factor but also allows integration with optical fibers for optical readout. Recent works [67, 68] have demonstrated that optical readout can simplify cryogenic systems by eliminating components such as circulators and isolators. It can also reduce the thermal load on a single optical fiber by three orders of magnitude, offering the potential for modular scalability and parallel readout.

# VII. CONCLUSION

In this work, we have systematically explored the integration of a superconducting transmon qubit with a seven-mode SAWR on a chip which is fabricated with flip-chip technology, establishing a versatile platform for studying cQAD. We investigated the rich physics governing the interplay between coherent and dissipative processes in hybrid quantum systems. To this end, we combined three complementary approaches. First, we performed structural verification through phononic oscillation experiments. Second, we characterized phonon-qubit interactions spectroscopically in the dispersive regime.

Third, we analyzed the dynamics of energy exchange in the transition-coupling regime. The observation of mode-dependent transitions between underdamped coherent oscillations and overdamped decay—controlled by coupler bias or mode number—reveals the important role of tunable coupling in mediating energy flow and dissipation. Our demonstration of a multimode SAWR as a dual-functional resource—enabling both dispersive readout and fast qubit reset—highlights the practical advantages of the transition-coupling regime.

Future work will focus on further improvi
ng the quality factor of mechanical modes and leveraging the properties of mode-number-dependent coupling modulation to engineer a multimode SAWR with alternating strong and transition-coupling regimes.

Our findings not only underscore the practical utility of the transition-coupling regime but also offer new insights for optical readout schemes with piezo-optomechanical transducers. Such insights are important for the future design of large-scale superconducting quantum chips, laying a solid foundation for the engineering of robust and controllable quantum devices.

Note added. While preparing this manuscript, we became aware of an independent study [69]. The authors observe the coherent evolution between the qubit and the modes of a multimode mechanical resonator. Our work complements this study by defining the transition-coupling regime, while additionally demonstrating tunable coupling and enabling the observation of phononic responses.

# ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (Grant Nos. 92365209, 12204528, 92265207, T2121001), the Innovation Program for Quantum Science and Technology (Grant No. 2021ZD0301800) and the Micro/nano Fabrication Laboratory of Synergetic Extreme Condition User Facility (SECUF). Devices were made at the Nanofabrication Facilities at the Institute of Physics, CAS in Beijing.

9

The experimental setup, illustrated in Fig. 6, consists of several modules arranged sequentially from left to right: the qubit Z control, the gmon Z control, the qubit XY control, the SAW pump, and the qubit readout (including input and output). The circuitry incorporates a series of attenuators, filters, circulators, and amplifiers to ensure proper signal handling. The qubit Z control line delivers long Z square pulses and fast pulses to tune the qubit's transition frequency.

Similarly, the gmon Z control line provides long Z square pulses and fast pulses to modulate the coupling strength between the qubit and the SAW. The XY control line applies microwave pulses for manipulating the qubit state. The SAW pump line drives the interdigitated transducer (IDT) with microwave pulses, enabling the conversion of microwave photons into phonons. For qubit readout, the qubit is capacitively coupled to a quarter-wave coplanar waveguide resonator with a frequency of $6.7005\mathrm{GHz}$ (readout cavity).

The opposite end of the readout resonator is inductively coupled to a transmission line. A microwave pulse at the resonator's bare frequency is injected into the transmission line and subsequently amplified by a high-electron-mobility transistor (HEMT) amplifier and a room-temperature microwave amplifier. Finally, the output signal is demodulated using an IQ mixer and digitized by an analog-to-digital converter (ADC).

# Appendix B: Phonon Oscillation

Figure 7 presents the experimental results of phonon oscillations under different power levels and pulse envelopes, with all pulses having a duration of $12\mathrm{ns}$ . It can be observed that, compared to the Gaussian envelope, the oscillations induced by the rectangular envelope do not reach zero at the minima, and the oscillation peaks also differ. This discrepancy is attributed to spectral impurity in the phonon signals generated by the IDT, likely caused by pulse broadening. Therefore, Gaussian pulses were used in all experiments discussed in the main text. Figure 8 shows the phonon oscillation results for the excitation of seven modes using Gaussian pulses with a length of $12\mathrm{ns}$ .

# Appendix C: Frequency Calibration of Qubit

In the experiment to determine the coupling point, it is necessary to compensate for the shift in qubit frequency at each point on-site, ensuring that the qubit frequency remains stationary at the idle point. Experimentally, this can be achieved by scanning the zpa offset and gmon bias, as depicted in Fig. 9. The relationship between the offset Z-pulse amplitude and the gmon bias can be determined by fitting the curve.

We measured the qubit lifetime $T_{1}$ over the frequency range from 4.2 GHz to 5.3 GHz for $V_{g} = -3.0, V_{g} = 0.75$ , and $V_{g} = -0.9$ , respectively. The results are plotted in Fig. 10. The average lifetime of the qubit reaches a maximum of approximately 18.2 $\mu s$ when the coupling is turned off. At maximum coupling, the average lifetime decreases to 1.2 $\mu s$ .

To determine the intrinsic dissipation rate of the qubit when it is resonant with the seven mechanical modes, we measured the lifetime of the qubit at the corresponding resonance frequencies with the coupling turned off. The data are presented in Fig. 11. It can be observed that the intrinsic dissipation rates of the qubit at these seven positions are very similar, indicating the absence of non-Markovian dissipation at these points. This ensures that our resonant evolution model can successfully describe the qubit and multimode SAW resonator composite system.

10

Appendix A: Measurement Setup

Appendix D: Decoherence of qubit

![](dt=2026-06-04/ht=19/836ab344a5b1edc15fd840c03b7ae66df6cad7f7f3b96002629d75bbc795afbf.jpg)

11

![](dt=2026-06-04/ht=19/83fe659b22a18703cfc1793e7782f51d4946189befabe61a37130a713f3ce5f4.jpg)

![](dt=2026-06-04/ht=19/5e30daab16344d3383d3cc400dc750189fb92d36a536c32fcc1cbcc22b711670.jpg)

![](dt=2026-06-04/ht=19/9aa7bcc94fe7b9c1ba101245297e81cf6f9445a49f5b638ca4f589730c5ea143.jpg)

![](dt=2026-06-04/ht=19/d3bb6a6324b38b872ecdc5e9c1a8986c3eb15ade6c84810baebcada1c8d99b31.jpg)

![](dt=2026-06-04/ht=19/c828ea2c52257a94e45eba72892e0e238477e6e6daa1050ead28a40b16c7e2a6.jpg)

12

13

14