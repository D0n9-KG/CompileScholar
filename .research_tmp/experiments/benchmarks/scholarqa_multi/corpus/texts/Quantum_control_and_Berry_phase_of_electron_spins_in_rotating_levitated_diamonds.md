# Quantum control and Berry phase of electron spins in rotating levitated diamonds in high vacuum

Received: 13 September 2023

Accepted: 23 May 2024

Published online: 13 June 2024

![](images/d54a23372268d9cbdce37e98401fad3e5aa10fc0ee623b4443ba01b72beb3878.jpg)

Check for updates

Yuanbin Jin $^{1,7}$ , Kunhong Shen $^{1,7}$ , Peng Ju $^{1}$ , Xingyu Gao $^{1}$ , Chong Zu $^{2}$ , Alejandro J. Grine $^{3}$ & Tongcang Li $^{1,4,5,6}$

Levitated diamond particles in high vacuum with internal spin qubits have been proposed for exploring macroscopic quantum mechanics, quantum gravity, and precision measurements. The coupling between spins and particle rotation can be utilized to study quantum geometric phase, create gyroscopes and rotational matter-wave interferometers. However, previous efforts in levitated diamonds struggled with vacuum level or spin state readouts. To address these gaps, we fabricate an integrated surface ion trap with multiple stabilization electrodes. This facilitates on-chip levitation and, for the first time, optically detected magnetic resonance measurements of a nanodiamond levitated in high vacuum. The internal temperature of our levitated nanodiamond remains moderate at pressures below $10^{-5}$ Torr. We have driven a nanodiamond to rotate up to 20 MHz ( $1.2 \times 10^{9}$ rpm), surpassing typical nitrogen-vacancy (NV) center electron spin dephasing rates. Using these NV spins, we observe the effect of the Berry phase arising from particle rotation. In addition, we demonstrate quantum control of spins in a rotating nanodiamond. These results mark an important development in interfacing mechanical rotation with spin qubits, expanding our capacity to study quantum phenomena.

Levitated nanoparticles and microparticles in high vacuum $^{1-3}$ offer a remarkable degree of isolation from environmental noises, rendering them exceptionally suitable for studying fundamental physics $^{4-6}$ and executing precision measurements $^{7-11}$ . Recently, the center-of-mass (CoM) motion of levitated nanoparticles in high vacuum has been cooled to the quantum regime $^{12-14}$ . Unlike tethered oscillators, levitated particles can also exhibit rotation $^{15-22}$ , which is intrinsically nonlinear $^{23,24}$ . Beyond rigid-body motion, levitated particles can host embedded spin qubits to provide more functionalities $^{25,26}$ . Notably, levitated nanodiamonds with embedded NV center spin qubits have been proposed for creating massive quantum superpositions $^{25,26}$ to test the limit of quantum mechanics and quantum gravity $^{27,28}$ . The embedded spin qubits can also sense the pseudo-magnetic field $^{29-31}$ , related to the Barnett effect $^{32,33}$ , and the quantum geometric phase $^{34,35}$ associated with particle rotation. The coupling between spin and mechanical rotation can be utilized for building sensitive gyroscopes $^{36,37}$ and rotational matter-wave interferometers $^{38,39}$ . These innovative proposals necessitate levitating diamond particles in high vacuum, well below $10^{-3}$ Torr. However, prior experiments with levitated diamonds struggled with vacuum level or spin state readouts.

Optical levitation of nanodiamonds has been experimentally achieved, but it was restricted to pressures above 1 Torr due to

$^{1}$ Department of Physics and Astronomy, Purdue University, West Lafayette, IN 47907, USA. $^{2}$ Department of Physics, Washington University, St. Louis, MO 63130, USA. $^{3}$ Sandia National Laboratories, Albuquerque, NM 87185, USA. $^{4}$ Elmore Family School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN 47907, USA. $^{5}$ Purdue Quantum Science and Engineering Institute, Purdue University, West Lafayette, IN 47907, USA. $^{6}$ Birck Nanotechnology Center, Purdue University, West Lafayette, IN 47907, USA. $^{7}$ These authors contributed equally: Yuanbin Jin, Kunhong Shen e-mail: tcli@purdue.edu

laser-induced heating $^{40-42}$ . Earlier studies using Paul traps, or ion traps, have demonstrated spin cooling $^{43}$ and angle locking $^{44}$ of levitated diamonds. However, they encountered a similar issue: diamond particles were lost when the air pressure was reduced to about 0.01 Torr $^{43-47}$ . This phenomenon could be due to the nonideal design of the Paul traps used in those experiments or the heating effects of detection lasers. While nanodiamonds can be levitated in a magneto-gravitational trap $^{48,49}$ , reading out the spin state within this setup remains elusive.

In this article, we design and fabricate an integrated surface ion trap (Fig. 1a) that incorporates an $\Omega$ -shaped stripline to deliver both a low-frequency high voltage for trapping and a microwave for NV spin control. Additionally, it comprises multiple electrodes to stabilize the trap and drive a levitated diamond to rotate. With this advanced Paul trap, we have performed optically detected magnetic resonance (ODMR) measurements of a levitated nanodiamond in high vacuum for the first time. Using NV spins, we measure the internal temperature of the levitated nanodiamond, which remains stable at approximately $350\mathrm{K}$ under pressures below $10^{-5}$ Torr. This suggests prospects for levitation in ultra-high vacuum. With a rotating electric field, we have been able to drive a levitated nanodiamond to rotate at high speeds up to $20\mathrm{MHz}$ ( $1.2\times 10^{9}\mathrm{rpm}$ ), which is about three orders of magnitudes faster than previous achievements using diamonds mounted on motor spindles[29,30]. Notably, this rotation speed surpasses the typical dephasing rate of NV spins in the diamond. With embedded NV electron spins in the levitated nanodiamond, we observe the effect of the Berry phase generated by the mechanical rotation, which also improves the ODMR spectrum in an external magnetic field. Moreover, we achieve quantum control of NV centers in a rotating levitated nanodiamond. Our work represents a pivotal advancement in interfacing mechanical rotation with spin qubits.

# Results

# Levitation of a nanodiamond in high vacuum

In the experiment, we levitate a nanodiamond in vacuum using a surface ion trap (Fig. 1a). The surface ion trap is fabricated on a sapphire wafer, which has high transmittance for visible and near-infrared lasers. To achieve levitation of nanodiamonds and quantum control of NV spins simultaneously, we apply both an AC high voltage and a microwave on a $\Omega$ -shaped circuit. The AC high voltage has a frequency of about $20\mathrm{kHz}$ and an amplitude of about $200\mathrm{V}$ . The microwave has a frequency of a few GHz. They are combined together with a homemade bias tee. The center ring electrode is grounded to generate a trapping center above the chip surface. The four electrodes at the corners are used to compensate the static electric fields from surface charges to minimize the micro-motion of a levitated nanodiamond. Figure 1c shows a simulated distribution of the electric field of the trap. The trapping center is $253\mu \mathrm{m}$ away from the chip surface.

The trapping potential depends on the charge-to-mass ratio $(Q/m)$ of a levitated particle. Thus, it is necessary to increase the charge number of particles for stable levitation in an ion trap. In our experiment, nanodiamonds are charged and sprayed out by electrospray and delivered to the surface ion trap with an extra linear Paul trap. The charge of the sprayed nanodiamond is typically larger than 1000 e, where e is the elementary charge, enabling a large trapping depth of more than 100 eV (see Supplementary Note 1 for more details). A 532 nm laser is used to excite diamond NV centers and a 1064 nm laser is

(a)   
![](images/beae34e012b158c56044c57f074d3b831fb47d5453318f0a56bbe46b34c9fc85.jpg)

<details>
<summary>text_image</summary>

a)
z
N
θ
C
V
C
C
ω
</details>

(b)   
![](images/f2aaf1feceec042428a2e863548c2e1524878d07cd4afa3300a46acc60ff7b6f.jpg)

<details>
<summary>chemical</summary>

Energy level diagram showing 532 nm and D transitions between ³E and ³A₂ orbitals, with ±1 and 0 states labeled
</details>

(c)   
![](images/8391ae193f9bd4d1c0a45aecb9bfdb3bf5bd2003028012f8bf5688a60117c52a.jpg)

<details>
<summary>heatmap</summary>

| x (mm) | y (mm) | z (mm) | Electric field (×10⁵ V/m) |
|--------|--------|--------|---------------------------|
| -1     | 0      | 0      | 0.5                       |
| -0.5   | 0.5    | 0.3    | 1.0                       |
| 0      | 1      | 0.6    | 1.5                       |
| 0.5    | 0.5    | 0.3    | 1.0                       |
| 1      | 0      | 0      | 0.5                       |
</details>

(d)   
![](images/613d9e5777d1bf396c898ed57ae51294e4f09ba8a7094484b04a4546a13eb9a1.jpg)

<details>
<summary>line</summary>

| Frequency (kHz) | PSD (V²/Hz) for 1.0 × 10⁻¹ Torr | PSD (V²/Hz) for 9.8 × 10⁻⁶ Torr |
| --------------- | ------------------------------- | ------------------------------- |
| ~1.0            | ~10⁻⁸                           | ~10⁻⁷                           |
| ~1.6            | ~10⁻⁹                           | ~10⁻⁸                           |
</details>

(e)   
![](images/869b9ea2133a86751e11557473e18c249849a2360cf13ef1ecfc3627b7d9d5ac.jpg)

<details>
<summary>line</summary>

| MW frequency (GHz) | 1.0 × 10^1 Torr | 6.9 × 10^-6 Torr |
| ------------------ | --------------- | ---------------- |
| 2.82               | ~0              | ~0               |
| 2.84               | ~-0.5           | ~-0.5            |
| 2.86               | ~-1.5           | ~-1.5            |
| 2.88               | ~-1.5           | ~-1.5            |
| 2.90               | ~-0.5           | ~-0.5            |
| 2.92               | ~0              | ~0               |
</details>

(f)   
![](images/5d20057effd22bdbde125e5497d5208dd0dd39fb39c76ca6ce69b838f88af09c.jpg)

<details>
<summary>line</summary>

| Pressure (Torr) | Temperature (K) |
| --------------- | --------------- |
| 1e-6            | 348             |
| 2e-6            | 347             |
| 5e-6            | 345             |
| 1e-5            | 340             |
| 2e-5            | 325             |
| 5e-5            | 310             |
| 1e-4            | 300             |
| 2e-4            | 295             |
| 5e-4            | 298             |
| 1e-3            | 300             |
</details>

Fig. 1 | Stable levitation of a nanodiamond in high vacuum. a Schematic of a levitated nanodiamond in a surface ion trap. The center ring electrode is grounded (GND). It has a hole at its center for sending a 1064 nm laser to monitor the nanodiamond's motion. A combination of a low-frequency high voltage (HV) and a high-frequency microwave (MW) is applied to the Ω-shaped circuit to trap the nanodiamond and control the NV centers. b Energy level diagram of a diamond NV center. A 532 nm laser (green arrow) excites the NV center. The red solid arrows and gray dashed arrows represent radiative decays and nonradiative decays, respectively. c Simulation of the electric field of the ion trap in the $xy$ -plane (top) and in the $xz$ -plane (bottom) when a voltage of 200 V is applied to the Ω-shaped circuit. The   
trap center is 253 $\mu$ m away from the chip surface. d Power spectrum densities (PSDs) of the center-of-mass (CoM) motion of the levitated nanodiamond at the pressure of 0.1 Torr (blue) and $9.8 \times 10^{-6}$ Torr (red). e Optically detected magnetic resonances (ODMRs) of the levitated nanodiamond measured at 10 Torr (blue circles) and $6.9 \times 10^{-6}$ Torr (red squares). The blue and red dashed lines are the corresponding zero-field splittings. The intensities of the 532 nm laser and the 1064 nm laser are 0.030 W/mm $^{2}$ and 0.520 W/mm $^{2}$ , respectively. f Internal temperature of the levitated nanodiamond as a function of pressure with the same laser intensities as shown in (e). Error bars represent the standard deviation of temperature among three measurements.

(a)   
![](images/d83fab85410226534228593050d2d2456d07708058f68f6f3c221d1bb69fa1d4.jpg)

<details>
<summary>text_image</summary>

A sin(ωt)
+DC1
-HA cos(ωt)
+DC2
HV+MW
GND1
GND3
GND2
+DC3
A cos(ωt)
+DC4
-A sin(ωt)
</details>

(b)   
![](images/f4a2f9e36bab323f5e80536b047cb6ff961b065f0f6fffa5132adddf5170fce7.jpg)

<details>
<summary>heatmap</summary>

| x (mm) | y (mm) | Potential (V) |
| ------ | ------ | ------------- |
| -1.0   | 1.0    | -4            |
| -0.5   | 0.5    | -2            |
| 0.0    | 0.0    | 0             |
| 0.5    | -0.5   | 2             |
| 1.0    | -1.0   | 4             |
</details>

(c)   
![](images/1c455455437128dc72d57c782dd8fe5314c912ca949d7e61d74ae1d469ed628f.jpg)

<details>
<summary>bar</summary>

| Frequency (MHz) | PSD (V²/Hz) |
| --------------- | ----------- |
| 0               | ~10⁻¹²      |
| 5               | ~10⁻¹¹      |
| 10              | ~10⁻¹⁰      |
| 15              | ~10⁻¹⁰      |
| 20              | ~10⁻¹⁰      |
| 25              | ~10⁻¹⁰      |
| 30              | ~10⁻¹⁰      |
| 35              | ~10⁻¹⁰      |
| 40              | ~10⁻¹⁰      |
</details>

Fig. 2 | Fast rotation of a levitated nanodiamond. a Optical image of the surface ion trap. AC voltage signals $(A \sin(\omega t + \varphi))$ with the same frequency $(\omega)$ and amplitude $(A)$ but different phases $(\varphi)$ are applied to the four corner electrodes to generate a rotating electric field. The phase is different by $\pi/2$ between neighboring electrodes. DC1, DC2, DC3 and DC4 are compensation voltages that minimize the   
micromotion to stabilize the trap. b Simulation of the electric potential in the $z = 253 \mu m$ plane at t = 0. The amplitude is A = 10 V. (c) PSDs of the rotational motion of the levitated nanodiamond at the rotation frequencies from 0.1 MHz to 20 MHz. The pressure is $1.0 \times 10^{-4}$ Torr.

applied to monitor the nanodiamond's motion. More details of our experimental setup are shown in Supplementary Fig. 1.

A main result of our experiment is that we can levitate a nanodiamond with the surface ion trap in high vacuum, which is a breakthrough as levitated diamond particles were lost around 0.01 Torr in previous studies using ion traps $^{43-47}$ . The red curve in Fig. 1d shows the power spectrum density (PSD) of the center-of-mass (CoM) motion of a levitated nanodiamond at $9.8 \times 10^{-6}$ Torr. The radius of the levitated nanodiamond is estimated to be about 264 nm based on its PSDs at 0.01 Torr (Supplementary Fig. 2). Our surface ion trap is remarkably stable in high vacuum. We can levitate a nanodiamond in high vacuum continuously for several weeks.

The internal temperature of a levitated nanodiamond is important as it will affect the spin coherence time and trapping stability. We measure the internal temperature using NV centers. The energy levels of an NV center is shown in Fig. 1b. We use a 532 nm laser to excite the NV centers and a single photon counting module to detect their PL. Then we sweep the frequency of a microwave to perform the ODMR measurement of a levitated nanodiamond in the absence of an external magnetic field. Figure 1e shows the ODMRs measured at 10 Torr (blue circles) and $6.9 \times 10^{-6}$ Torr (red squares). Based on the fitting of the ODMRs, the corresponding zero-field splittings (blue and red dashed lines) are 2.8694 GHz and 2.8650 GHz, respectively. The internal temperature of the levitated nanodiamond can be obtained from the zero-field splitting (see Methods and Supplementary Note 2 for details). The measured internal temperature at different pressures are shown in Fig. 1f. The internal temperature is close to the room temperature at pressures above 0.1 Torr, and increases when we reduce the pressure from 0.1 Torr to $10^{-4}$ Torr. Finally, it remains stable at approximately 350 K at pressures below $5 \times 10^{-5}$ Torr. This temperature is low enough to maintain quantum coherence of NV spins for quantum control $^{50}$ .

The observed phenomena (Fig. 1f) arise from the balance between laser-induced heating (Supplementary Fig. 3) and the cooling effects of air molecules and black-body radiation on the internal temperature of a levitated nanodiamond $^{51,52}$ . When the air pressure is high, the cooling rate due to surrounding air molecules is large and the internal temperature of the levitated nanodiamond is close to the room temperature. However, as air pressure decreases, cooling from air molecules diminishes, leading to a rise in internal temperature. When the pressure is below $5 \times 10^{-5}$ Torr, the temperature stabilizes as the cooling is dominated by the black body radiation, which is independent of the air pressure.

# Fast rotation and Berry phase

After a nanodiamond is levitated in high vacuum, we use a rotating electric field to drive the levitated nanodiamond to rotate at high speeds, which also stabilizes the orientation of the levitated nanodiamond. The four electrodes at the corners are applied with AC voltage signals $(A\sin(\omega t+\varphi))$ with the same frequency $(\omega)$ and amplitude $(A)$ but different phases $(\varphi)$ to generate a rotating electric field (Fig. 2a). The phases of neighboring signals are different by $\pi/2$ . Figure 2b shows the simulation of the electric potential in the xy-plane at t=0. More information can be found in Supplementary Note 3, Supplementary Fig. 4, and Supplementary Fig. 5. A levitated charged object naturally has an electric dipole moment due to inhomogeneous distribution of charges. In a rotating electric field, the levitated charged particle will rotate due to the torque produced by the interaction between the rotating electric field and the electric dipole of the particle. Figure 2c depicts the PSDs of the rotation at different driving frequencies (0.1–20 MHz). The maximum rotation frequency is 20 MHz in the experiment, which is limited by our phase shifters used to generate phase delays between signals on the four electrodes. This is about 3 orders of magnitudes faster than previous achievements using diamonds mounted on electric motor spindles $^{29,30}$ . When the rotation frequency is 100 kHz, the linewidth of the PSD of the rotational signal is fitted to be about $9.9\times10^{-5}$ Hz (Supplementary Fig. 5d), which is limited by the measurement time. This shows that the rotation is extremely stable and is locked to the driving electric signal. With easy control and ultra-stability, this driving scheme enables us to adjust and lock the rotation of the levitated nanodiamond over a large range of frequencies (see Supplementary Note 3 for more details).

The fast-rotating diamond with embedded NV spins allows us to observe the effects of the Berry phase due to mechanical rotation. The Berry phase, also known as the geometric phase, is a fundamental aspect of quantum mechanics with applications in multiple fields, including the topological phase of matter and the quantum hall effect $^{53-57}$ . The Berry phase in the laboratory frame is equivalent to the pseudo-magnetic field (called the Barnett field in ref. 57) in the rotating frame: $B_{\omega} = \omega_{r}/\gamma$ , where $\gamma$ is the spin gyromagnetic ratio. In this work, the microwave source is fixed in the laboratory frame. Only the levitated diamond is rotating. So, we can observe the effect of the Berry phase due to rotation $^{57}$ .

In a rotating diamond, the embedded NV centers also follow the rotation with an angular frequency of $\omega_{r}$ (Fig. 3). The levitated nanodiamond in our experiment contains ensembles of NV centers with four groups of orientations. Figure 3d shows an NV center embedded in a nanodiamond rotating around the z-axis in the presence of an external magnetic field. The direction of the magnetic field is along the z-axis. The angle between the NV axis and z-axis is $\theta$ , and the azimuth is $\phi(t)$ relative to the x-axis. The Hamiltonian of the rotating NV electron spin in the laboratory frame, neglecting strain effects, can be

(a)   
![](images/c4df1bb567dd70b5bb7dc80c4902aad6afc101eb53baadc0fe0528cca4a9d970.jpg)

<details>
<summary>text_image</summary>

z
ωr
BMW
NV axis
θ
φ
x
y
</details>

(b)   
![](images/0b4898d72fe98985fe74cfa639b7084a395890513100e440951b88b78642ac19.jpg)

<details>
<summary>line</summary>

| MW frequency (GHz) | Contrast (%) - ω_r / 2π = 0.1 MHz | Contrast (%) - ω_r / 2π = 14 MHz |
| ------------------ | --------------------------------- | --------------------------------- |
| 2.82               | ~0.0                              | ~0.0                              |
| 2.84               | ~-0.5                             | ~-0.7                             |
| 2.86               | ~-1.5                             | ~-1.6                             |
| 2.88               | ~-1.0                             | ~-1.3                             |
| 2.90               | ~-0.5                             | ~-0.7                             |
| 2.92               | ~0.0                              | ~0.0                              |
</details>

(c)   
![](images/2da83cc25165608004ad1d25208239df9ff87913b49e03214404efdfbfe00018.jpg)

<details>
<summary>line</summary>

| - ω_r / 2π (MHz) | Experiment | Theory: θ = 0 | Theory: θ = 20° |
| ---------------- | ---------- | ------------- | --------------- |
| 0                | 28         | 30            | 30              |
| 2                | 31         | 32            | 31              |
| 4                | 35         | 36            | 35              |
| 6                | 37         | 39            | 38              |
| 8                | 40         | 42            | 41              |
| 10               | 44         | 46            | 45              |
| 12               | 48         | 49            | 48              |
| 14               | 52         | 53            | 52              |
| 16               | 56         | 57            | 56              |
| 18               | 60         | 61            | 60              |
| 20               | 64         | 65            | 64              |
</details>

(d)   
![](images/03781866753082bed147a796e72e9f5bae46c40360237fcf155411d1bfd3feb7.jpg)

<details>
<summary>text_image</summary>

B
z
ωr
BMW
NV axis
θ
xi
φ
x
y
</details>

(e)   
![](images/4bd96b5a85c878efd693e4401be87c210212d8fdd71167121c94842f89cbecbd.jpg)

<details>
<summary>line</summary>

| MW frequency (GHz) | Contrast (%) - ω_r / 2π = 0.1 MHz | Contrast (%) - ω_r / 2π = 0 |
| ------------------ | --------------------------------- | --------------------------- |
| 2.6                | ~0                                | ~0                          |
| 2.8                | ~-1                               | ~-2                         |
| 3.0                | ~-1                               | ~-3                         |
| 3.2                | ~0                                | ~0                          |
</details>

(f)   
![](images/cd19f27b23e6c2e396cc0e41919f042f47baf57b47a294e57a291acd7a6bc501.jpg)

<details>
<summary>line</summary>

| - ω_r / 2π (MHz) | Experiment | Theory: θ = 20.7° | Theory: θ = 24.0° | Fitting |
| ---------------- | ---------- | ----------------- | ----------------- | ------- |
| 0                | 3.12       | 3.12              | 3.115             | 3.12    |
| 5                | 3.125      | 3.125             | 3.12              | 3.125   |
| 10               | 3.13       | 3.13              | 3.125             | 3.13    |
</details>

Fig. 3 | Effects of the Berry phase generated by a rotating nanodiamond.   
a Schematic of an NV center in the nanodiamond rotating around the z-axis in the absence of an external magnetic field. The small angle between $B_{MW}$ and the z-axis is due to the asymmetric design of the waveguide. b ODMRs of the levitated nanodiamond at rotation frequencies of 0.1 MHz (blue circles) and 14 MHz (red squares). c Experimentally measured FWHM of the ODMR spectrum as a function of rotation frequency (blue circles). Error bars show the standard deviations among three measurements. The red solid curve and orange dashed curve are theoretically calculated FWHMs at $\theta = 0^{\circ}$ and $\theta = 20^{\circ}$ , respectively. d Schematic of an NV center in the nanodiamond rotating around the z-axis in an external magnetic field. The magnetic field is along the z-axis and is about 100 G. e The upper panel (red   
squares) shows the ODMR of the levitated nanodiamond at a rotation frequency of 0.1 MHz and a pressure of $1.0 \times 10^{-4}$ Torr. The bottom panel (gray circles) shows the ODMR of a nanodiamond without a stable rotation at the pressure of 10 Torr. The corresponding solid curves are the fittings with eight Lorentzian dips.   
f Experimentally measured frequency of the right-most dip of the ODMR spectrum of an NV center as a function of rotation frequency (blue circles). The green solid curve and violet dashed curve are theoretical calculations at $\theta=20.7^{\circ}$ and $\theta=24.0^{\circ}$ , respectively. The magenta dashed curve is a linear fitting of the resonance frequency. The error bars represent standard deviations among three measurements.

written as $^{34}$ :

$$
\begin{array}{l} H _ {l a b} = H _ {0, l a b} + g \mu_ {B} B S _ {z} = \frac {1}{\hbar} R (t) D S _ {z} ^ {2} R ^ {\dagger} (t) + g \mu_ {B} B S _ {z} \\ = D \hbar \left( \begin{array}{c c c} \cos^ {2} \theta + \frac {\sin^ {2} \theta}{2} + \frac {g \mu_ {B} B}{D} & \frac {e ^ {- i \phi} \cos \theta \sin \theta}{\sqrt {2}} & \frac {e ^ {- 2 i \phi} \sin^ {2} \theta}{2} \\ \frac {e ^ {i \phi} \cos \theta \sin \theta}{\sqrt {2}} & \sin^ {2} \theta & - \frac {e ^ {- i \phi} \cos \theta \sin \theta}{\sqrt {2}} \\ \frac {e ^ {2 i \phi} \sin^ {2} \theta}{2} & - \frac {e ^ {i \phi} \cos \theta \sin \theta}{\sqrt {2}} & \cos^ {2} \theta + \frac {\sin^ {2} \theta}{2} - \frac {g \mu_ {B} B}{D} \end{array} \right), \tag {1} \\ \end{array}
$$

where D is the zero-field splitting, $R(t)=R_{z}(\phi(t))R_{y}(\theta)$ is the rotation transformation, and $R_{j}(\theta)=\exp(-i\theta\mathbf{n}\cdot\mathbf{S})$ for the rotation angle $\theta$ around the n direction, j=y, z, and S are the spin operators. The Stark shift for NV centers induced by the electric field is negligible and hence is not included in the equation. The Hamiltonian possesses three eigenstates $|m_{s},t\rangle_{lab}$ ( $m_{s}=0,\pm1$ ). The detailed expressions can be found in the Supplementary Note 4. Based on its definition, the Berry phase can be calculated as $^{57}$

$$
\gamma_ {m _ {s}} = i \int_ {0} ^ {t} l a b \left\langle m _ {s}, t ^ {\prime} \right| \frac {\partial}{\partial t ^ {\prime}} \left| m _ {s}, t ^ {\prime} \right\rangle_ {l a b} d t ^ {\prime} = m _ {s} \omega_ {r} t \cos \theta . \tag {2}
$$

Here the Berry phase is calculated for an open-path and is hence gauge-dependent. The spin state of the NV center is observed through the interaction with a microwave magnetic field. In our experiment, the direction of the microwave is in the yz-plane and has a small angle $\theta' = 8.5^{\circ}$ relative to the z axis, resulting from the asymmetric design of the waveguide. However, the dominant transition probability arises from the longitudinal (z) component. The expected value of the transition probability of the spin states interacting with the microwave can be expressed as

$$
\begin{array}{l} l a b \langle \pm 1, t | e ^ {i H _ {l a b} t / \hbar} e ^ {- i \gamma_ {\pm 1}} H _ {M W, z, l a b} e ^ {i \gamma_ {0}} e ^ {- i H _ {l a b} t / \hbar} | 0, t \rangle_ {l a b} \\ = \frac {1}{2} g \mu_ {B} B _ {M W} \cos \theta^ {\prime} e ^ {i (- \omega_ {M W} + D \pm g \mu_ {B} B \cos \theta \mp \omega_ {r} \cos \theta) t} _ {l a b} \langle \pm 1, 0 | e ^ {i \theta S _ {y}} S _ {z} e ^ {- i \theta S _ {y}} | 0, 0 \rangle_ {l a b}. \tag {3} \\ \end{array}
$$

According to Eq. (3), the transition of spin states from $|m_{s}=0\rangle_{lab}$ to $|m_{s}=\pm1\rangle_{lab}$ can be driven by a microwave at the resonance frequency of $D\pm g\mu_{B}B\cos\theta\mp\omega_{r}\cos\theta$ , where the frequency shift $\mp\omega_{r}\cos\theta$ is due to the Berry phase induced by the mechanical rotation.

We first investigate the effect of the Berry phase in the absence of an external magnetic field. Figure 3a shows the diagram of an NV center rotating around the z-axis without an external magnetic field. To observe the frequency shift due to fast rotation, ODMR measurements of the levitated nanodiamond are carried out at different rotation frequencies. Figure 3b displays ODMRs at the rotation frequencies of 0.1 MHz (bule circles) and 14 MHz (red squares). The full width at half maximum (FWHM) of the ODMR at $\omega_{r}=2\pi\times14$ MHz is clearly larger than that at $\omega_{r}=2\pi\times0.1$ MHz, which is caused by the Berry phase due to rotation. The FWHM of the ODMR at different rotation frequencies is shown in Fig. 3c. The blue circles are the experimental results. The red and orange curves are theoretical results for $\theta=0^{\circ}$ and $\theta=20^{\circ}$ , respectively. Experimentally, the NV ensemble contains NV centers with four orientations. Based on Eq. (3), the broadening of the ODMR spectrum is mainly determined by NV centers with the smallest $\theta$ , which have the largest frequency shift induced by the Berry phase(Supplementary Fig. 6). The frequency shift of $\mp\omega_{r}\cos\theta$ is

insensitive to the angle $\theta$ for small $\theta$ . This explains why the theoretical results for $\theta=0^{\circ}$ and $\theta=20^{\circ}$ are similar, and both agree well with the experimental results. All data shown in Fig. 3b, c are taken from one levitated diamond.

To determine the frequency shift as a function of the rotational frequency unambiguously, an external magnetic field can be applied to separate the energy levels of NV centers along four different orientations. Here we apply a static magnetic field of about 100 G along the z-axis to separate energy levels (Fig. 3d). Data shown in Fig. 3e, f are taken from one levitated diamond, which is different from the one used for Fig. 3b, c. In Fig. 3e, the red squares show the ODMR spectrum measured at a rotation frequency of 0.1 MHz. The linewidths of ODMR dips for levitated diamond NV centers are broader than those for fixed NV centers due to the continuous change of NV orientations relative to the magnetic field. Compared with the ODMR spectrum of a levitated nanodiamond without stable rotation (gray circles), the linewidth of each dip for a diamond rotating at 0.1 MHz is narrower. This clearly demonstrates that fast rotation can stabilize the orientation of the levitated nanodiamond. Now we consider the NV centers with the smallest $\theta$ (largest Zeeman shift) and the transition between the state $|m_{s}=0\rangle$ and the state $|m_{s}=+1\rangle$ as an example. The electron spin resonance frequency is 3.120 GHz at $\omega_{r}=2\pi\times0.1$ MHz for this transition. The corresponding angle between the NV-axis and the rotation axis is $\theta=20.7^{\circ}$ , which is calculated based on the transition frequency and the magnitude of the external magnetic field. We then measure the resonance frequency at different clockwise (unless otherwise specified, all are viewed from the positive z direction) rotation frequencies, as shown in Fig. 3f. The resonance frequency increases following the increase of the rotation frequency. The experimental data points are in between the theoretically calculated curves for $\theta=20.7^{\circ}$ (green solid line) and $\theta=24.0^{\circ}$ (violet dashed line), indicating the orientation of the NV axes changes slightly when the rotation frequency increases. This is because the electric dipole moment of the levitated nanodiamond is not exactly perpendicular to the axis of the largest or the smallest moment of inertia. Once the rotation frequency increases, the nanodiamond tends to rotate along its stable axis and the driving torque is not large enough to keep its former orientation. The magenta dashed curve is a linear fitting of the resonance frequency. The orientation of the NV center can be calculated by the resonance frequency at the various rotation frequencies. The angle $\theta$ changes by approximately $3.3^{\circ}$ at $\omega_{r}=2\pi\times10$ MHz, compared with that at $\omega_{r}=2\pi\times0.1$ MHz. A rotating diamond can also serve as a gyroscope $^{58,59}$ .

The effect of the Berry phase in a levitated nanodiamond rotating at the counterclockwise direction is shown in Supplementary Fig. 6. The resonance frequency between the state $|m_{s}=0\rangle$ and the state $|m_{s}=+1\rangle$ decreases as the rotation frequency increases for counterclockwise rotation (Supplementary Fig. 6c), which is different from that of the levitated nanodiamond rotate clockwise (Fig. 3f).

# Quantum control of fast-rotating NV centers

Quantum control of spins is important for creating superposition states $^{25,26,39}$ and performing advanced quantum sensing protocols $^{60}$ . Here, we apply a resonant microwave pulse to demonstrate quantum state control of fast-rotating NV centers. The spin state can be read out by measuring the emission PL. Because a weak 532 nm laser is used to avoid significant heating, the initialization time should be long enough to prepare the NV spins to the $|m_{s}=0\rangle$ state. When the laser intensity is 0.113 W/mm $^{2}$ , the initialization time is 1.05 ms (Supplementary Fig. 7a). This is shorter than the spin relaxation time ( $T_{1}\sim3.6$ ms) of this levitated nanodiamond (Supplementary Fig. 7b). We also measure Rabi oscillation of a nanodiamond fixed on a glass cover slip with the same 532 nm laser intensity for comparison. We get similar results for both high and low intensities of the 532 nm laser (Supplementary Fig. 8). Due to the $\Omega$ -shape of the microwave antenna, the orientation of the magnetic field of the microwave is located in the $yz$ -plane and slightly different from the $z$ -axis with an angle of about $\theta' = 8.5^\circ$ (Fig. 4a). So, $\mathbf{n}_{MW} = (-\sin \theta', 0, \cos \theta')$ . The effective microwave magnetic field acting on NV spins, with the orientation of $\mathbf{n}_{NV} = (\cos \phi(t) \sin \theta, \sin \phi(t) \sin \theta, \cos \theta)$ , changes as a function of the rotation phase $\phi(t)$ of the levitated nanodiamond. The Rabi frequency $\Omega_{Rabi}$ can be written as $^{61,62}$ :

$$
\Omega_ {R a b i} \propto \sqrt {1 - \left(\cos \theta \cos \theta^ {\prime} - \sin \phi (t) \sin \theta \sin \theta^ {\prime}\right) ^ {2}}. \tag {4}
$$

Therefore, it is necessary to synchronize the microwave pulse and the rotation phase of the levitated nanodiamond. Figure 4b shows the pulse sequence of the Rabi oscillation measurement. The time gap between the initialization and the readout laser pulses is twice of the rotation period, which allows us to apply the microwave pulse at an arbitrary rotation phase between 0 and 2π.

The measured Rabi oscillations between the state $|m_{s}=0\rangle$ and the state $|m_{s}=+1\rangle$ of NV centers are shown in Fig. 4d, e. All these measurements are carried out at a rotation frequency of 100 kHz. The rotation period is 10 $\mu$ s which is much longer than the microwave pulse. For NV centers with different orientations, the Rabi frequencies are different. The measured Rabi frequencies are 7.10 MHz, 6.57 MHz and 2.80 MHz when the applied microwave frequencies are 2.936 GHz (dip 1), 3.009 GHz (dip 2), and 3.129 GHz (dip 3), respectively (Fig. 4d). Figure 4e shows Rabi oscillations of the NV centers with $\theta=22^{\circ}$ at different rotation phases. The blue circles and black squares are measured at the rotation phase of $\phi=\pi/2$ and $\phi=\pi$ , respectively. The corresponding Rabi frequencies are 2.72 MHz and 2.23 MHz due to the different projections of the microwave magnetic field along the NV-axis. We also apply microwave pulse at other rotation phases to explore how it affects the Rabi frequency. Figure 4f shows the Rabi frequency $\Omega_{Rabi}$ for NV centers with $\theta=22^{\circ}$ (blue circles) as a function of the rotation phase. The Rabi frequency is smallest at $\phi=3\pi/2$ . The red curve is the theoretical prediction, which agrees well with our experimental results.

# Feedback cooling of the center-of-mass motion

To study quantum spin-mechanics and use a levitated diamond for precision measurements, it will be crucial to reduce the energy of the CoM motion of a levitated diamond. Our integrated Paul trap has multiple electrodes, which can be used for feedback cooling. Because of the high quality factor of the CoM in high vacuum and the low frequency of the CoM motion, we add $\pi /2$ phase delays to the position signals of the levitated diamond to obtain its velocity signals. We then apply electric forces on the charged diamond proportional to the velocities but with opposite signs to cool the CoM motion. The schematic diagram is shown in Fig. 5a. The feedback loop is implemented through an FPGA (Field Programmable Gate Array). The motion signals are read out, followed by band-pass filters, amplifiers and phase delayers, and then fed back to the four electrodes at the corners of the ion trap. Figure 5b-d shows the PSDs of the CoM motion of a levitated nanodiamond at the pressure of 0.02 Torr without feedback cooling (noFB, blue curves) and at the pressure of $2.0\times 10^{-5}$ Torr with feedback cooling (FB, red curves). The orange curves are the corresponding noise floors. Based on the fitting, the final temperature of the CoM motion with feedback cooling are $1.2\pm 0.3\mathrm{K}$ , $3.5\pm 0.4\mathrm{K}$ , and $86\pm 26\mathrm{K}$ along the $x,y,$ and $z$ directions, respectively. The final temperatures are mainly limited by the small size of the center hole of the surface ion trap used for forward detection (Fig. 5a), which severely limits the NA of the detection system. The cooling efficiency can be improved in the future with backward detection by using the backward scattered light of the levitated diamond collected by the objective lens.

(a)   
![](images/c9100db2fd8f36524189cf54c2a5e8d9a4461b51c70027c127c87bd1410368fe.jpg)

<details>
<summary>text_image</summary>

MW
z
θ'
θ
NV
x
y
φ(t)
</details>

(b)   
![](images/68260e925ee46cb04281b635162faa60f52c72326f1d78db5a791c86523f792f.jpg)

<details>
<summary>other</summary>

| Signal     | Time Segment | Phase Label |
|------------|--------------|-------------|
| Laser      | t            | I           |
| MW         | t            | τ           |
| Counter    | R            | R           |
| φ(t)       | ...          | ...         |
</details>

(c)   
![](images/e5a56b5612738970630a67ceb3feb721171b166b302046970b53bc63f16fc6e8.jpg)

<details>
<summary>line</summary>

| MW frequency (GHz) | Contrast (%) |
| ------------------ | ------------ |
| 2.6                | -0.5         |
| 2.7                | -1.0         |
| 2.8                | -2.5         |
| 2.9                | -1.5         |
| 3.0                | -3.0         |
| 3.1                | -1.0         |
| 3.2                | 0.0          |
</details>

(d)   
![](images/653e406d3a0bc52710fb101d8ac5a1692e4b126b1502b4b5eeaeda306370b86d.jpg)

<details>
<summary>line</summary>

| Time (μs) | Contrast (%) - 1: f_MW = 2.935 GHz | Contrast (%) - 2: f_MW = 3.009 GHz | Contrast (%) - 3: f_MW = 3.129 GHz |
| --------- | ----------------------------------- | ---------------------------------- | ---------------------------------- |
| 0.0       | ~4.0                                | ~2.0                               | ~0.0                               |
| 0.2       | ~3.5                                | ~1.5                               | ~-0.5                              |
| 0.4       | ~3.8                                | ~1.8                               | ~-0.8                              |
| 0.6       | ~3.7                                | ~1.6                               | ~-0.7                              |
| 0.8       | ~3.6                                | ~1.5                               | ~-0.8                              |
| 1.0       | ~3.5                                | ~1.4                               | ~-0.9                              |
</details>

(e)   
![](images/890dccfccbdd9d44c5385fa54a0eeceb89eaa2834211faffed4a7939f43cb555.jpg)

<details>
<summary>line</summary>

| Time (μs) | Contrast (%) for φ = π | Contrast (%) for φ = π/2 |
| --------- | ---------------------- | ------------------------ |
| 0.0       | 2.0                    | 0.0                      |
| 0.2       | 0.5                    | -1.0                     |
| 0.4       | 1.5                    | -0.5                     |
| 0.6       | 1.0                    | -0.8                     |
| 0.8       | 0.5                    | -1.0                     |
| 1.0       | 0.0                    | -1.0                     |
</details>

(f)   
![](images/a466063d46d0230c77a8a8d0d5b296ef288bc20a61f8bbcf8170d7c8c4091df9.jpg)

<details>
<summary>line</summary>

| φ - π/2 (rad) | Rabi frequency (MHz) |
| ------------- | -------------------- |
| 0             | 2.8                  |
| π/4           | 2.6                  |
| π/2           | 2.2                  |
| 3π/4          | 1.6                  |
| π             | 1.3                  |
| 5π/4          | 1.6                  |
| 3π/2          | 1.9                  |
| 7π/4          | 2.5                  |
| 2π            | 2.9                  |
</details>

GHz, and 3.129 GHz, respectively. The red and purple curves are shifted 2% and 4% to separate the curves. e Rabi oscillation of NV centers with $\theta = 22^{\circ}$ corresponding to the resonance frequency of 3.129 GHz (dip 3). The blue circles and black squares are measured at rotation phase of $\phi = \pi/2$ and $\phi = \pi$ , respectively. The black curve is shifted 2%. f Rabi frequency at $\theta = 22^{\circ}$ (blue circles) as a function of the rotation phase $\phi$ . The red curve is the theoretical prediction. The error bars represent standard deviations among three measurements.

Fig. 4 | Quantum control of NV centers in a levitated nanodiamond in high vacuum with a rotation frequency of 100 kHz. a Schematic of the Rabi oscillation measurement at different rotation phase $\phi(t)$ . The angle $\theta'$ between the magnetic component of microwave and the z-axis is 8.5°. b Pulse sequence of the Rabi oscillation measurement. c ODMR of the levitated nanodiamond. d Measured Rabi oscillations of NV centers at three different orientations. The Rabi frequencies are 7.10 MHz, 6.57 MHz, and 2.80 MHz at the ODMR frequencies of 2.935 GHz, 3.009   
(a)   
![](images/ff9e5d2e239b4c0ae819ad0edd0a73ede2275567905db9547de63925a1ddaef3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Objective"] --> B["Ion trap Lens"]
    B --> C["Detectors"]
    C --> D["FPGA"]
    D --> E["Φx"]
    D --> F["Φy"]
    D --> G["Φz"]
    E --> H["Signal Path"]
    F --> I["Signal Path"]
    G --> J["Signal Path"]
    H --> K["x"]
    I --> L["y"]
    J --> M["z"]
```
</details>

(b)   
![](images/74fdb36a0998afaccd361cfcb89326a8290987e94b2ec70064fc62c380ec6362.jpg)

<details>
<summary>line</summary>

| Frequency (Hz) | 0.02 Torr, noFB | 2×10⁻⁵ Torr, FB | Noise |
| -------------- | --------------- | --------------- | ----- |
| 1800           | ~10⁻⁷           | ~10⁻⁹           | ~10⁻¹⁰ |
| 1820           | ~10⁻⁶           | ~10⁻⁸           | ~10⁻¹⁰ |
| 1840           | ~10⁻⁵           | ~10⁻⁷           | ~10⁻¹⁰ |
| 1860           | ~10⁻⁶           | ~10⁻⁸           | ~10⁻¹⁰ |
| 1880           | ~10⁻⁷           | ~10⁻⁹           | ~10⁻¹⁰ |
</details>

(c)   
![](images/8a70619bd233626ce1b59ef676c0fe81fb5a798976b46622517e3cd87a6283e5.jpg)

<details>
<summary>line</summary>

| Frequency (Hz) | 0.02 Torr, noFB | 2×10⁻⁵ Torr, FB | Noise |
| -------------- | --------------- | --------------- | ----- |
| 1640           | ~10⁻⁷           | ~10⁻⁸           | ~10⁻⁸ |
| 1660           | ~10⁻⁶           | ~10⁻⁸           | ~10⁻⁸ |
| 1680           | ~10⁻⁴           | ~10⁻⁸           | ~10⁻⁸ |
| 1700           | ~10⁻⁶           | ~10⁻⁸           | ~10⁻⁸ |
| 1720           | ~10⁻⁷           | ~10⁻⁸           | ~10⁻⁸ |
</details>

(d)   
![](images/6c82dc2a112ee0feab5e4ab7b6d2dbf3bdd276cf56f0c7ade6d758eb12d834db.jpg)

<details>
<summary>line</summary>

| Frequency (Hz) | 0.02 Torr, noFB (PSD V²/Hz) | 2×10⁻⁵ Torr, FB (PSD V²/Hz) |
| -------------- | --------------------------- | ---------------------------- |
| 3430           | ~10⁻⁹                       | ~10⁻⁹                        |
| 3440           | ~10⁻⁸                       | ~10⁻⁹                        |
| 3450           | ~10⁻⁷                       | ~10⁻⁹                        |
| 3460           | ~10⁻⁸                       | ~10⁻⁹                        |
</details>

Fig. 5 | Feedback cooling of the CoM motion of a levitated nanodiamond in the ion trap. a Schematic diagram of the feedback cooling method. b–d PSDs of the CoM of the levitated nanodiamond along the (b) x, (c) y, and (d) z directions at the pressure of 0.02 Torr without cooling (blue curves) and at the pressure of $2.0 \times 10^{-5}$   
Torr with feedback cooling (red curves). The orange curves are the noise floors. Based on the fitting, the effective temperature of the CoM motion with feedback cooling are $1.2 \pm 0.3$ K, $3.5 \pm 0.4$ K, and $86 \pm 26$ K along the x, y, and z directions, respectively.

# Discussion

In conclusion, we have levitated a nanodiamond at pressures below $10^{-5}$ Torr with a surface ion trap. We performed ODMR measurement of a levitated nanodiamond in high vacuum for the first time. The internal temperature of the levitated nanodiamond remains stable at about 350 K when the pressure is below $5 \times 10^{-5}$ Torr, which means stable levitation with an ion trap will not be limited by heating even in ultrahigh vacuum. This offers a unique platform for studying fundamental physics, such as massive quantum superposition $^{25,26,39}$ .

Additionally, we apply a rotating electric field that exerts a torque on the levitated nanodiamond to drive it to rotate at high speeds up to 20 MHz. 20 MHz rotation can generate a pseudo-magnetic field of 0.71 mT for an electron spin, and a pseudo-magnetic field of 6.5 T for an ${}^{14}$ N nuclear spin. With this method, the rotation frequency of a levitated nanodiamond is extremely stable and easily controllable. The effect of the Berry phase generated by rotation $^{35}$ is observed with the embedded NV center electron spins. This will be useful for creating a gyroscope for rotation sensing $^{36,58,59}$ . We also demonstrate quantum control of rotating NV centers in high vacuum, which will be important for using spins to create nonclassical states of mechanical motion $^{25,26,39}$ . Using feedback cooling, the CoM of the levitated nanodiamond is cooled in all three directions with a minimum temperature of about 1.2 K along one direction.

The maximum rotation frequency in this experiment is limited by the bandwidth of the multichannel waveform generation system for generating the phase-shifted signals on the four electrodes. The rotation frequency can be much higher with a better waveform generation system. Furthermore, in the presence of a DC external magnetic field, the NV centers within a rotating nanodiamond experience an AC magnetic field. Quantum sensing of an AC magnetic field can have a higher sensitivity compared to that of a DC magnetic field $^{63}$ . Consequently, the mechanical rotation can enhance the sensitivity of a magnetometer in measuring DC magnetic fields. By using purer diamond particles, i.e. CVD diamonds, a higher excitation power of the 532 nm laser can be employed to reduce the initialization time of NV centers.

# Methods

# Experiment setup and materials

The surface ion trap is fabricated on a sapphire wafer by photolithography. The chip is fixed on a 3D stage and installed in a vacuum chamber. The AC high voltage signal used to levitate nanoparticles and the microwave used for quantum control are combined with a bias tee to be delivered to the chip. A 532 nm laser beam is incident from the bottom to excite diamond NV centers. The photoluminescence (PL) is collected by an objective lens with a numerical aperture (NA) of 0.55. A 1064 nm laser beam focused by the same objective lens is used to monitor both the center-of-mass (CoM) motion and the rotation of the levitated nanoparticle. The PL is separated with the 532 nm laser and the 1064 nm laser by dichroic mirrors. The counting rate and optical spectrum of the PL are measured by a single photon coating module and a spectrometer. The processes of particle launching and trapping are monitored by two cameras.

The diamond particles were acquired from Adamas Nano. The product model is MDNV1umHi10mg (1 micron Carboxylated Red Fluorescence, 1 mg/mL in DI Water, \~3.5 ppm NV). The experimental data shown in the main text of the manuscript are obtained from four different diamond particles. The data presented in Fig. 1, Fig. 2, and Fig. 3a–c originate from measurements conducted on the same nanodiamond particle. Figure 3d–f shows the data from a second nanodiamond particle, while the data in Fig. 4 is measured using the third nanodiamond particle. Figure 5 uses the fourth diamond particle.

# Internal temperature of a levitated nanodiamond

In the experiment, we measure the ODMR of levitated nanodiamond NV centers to detect the internal temperature in the absence of an external magnetic field. The zero-field Hamiltonian of NV center is: $H = DS_{z}^{2}/\hbar + E\left(S_{x}^{2} - S_{y}^{2}\right)/\hbar$ , where D is the zero-field energy splitting between the states of $|m_{s} = 0\rangle$ and $|m_{s} = \pm1\rangle$ , E is the splitting between the states due to the strain effect. The small splitting between two dips in the ODMR spectra (Fig. 1e) without an external magnetic field is due to the E term from strain in the nanodiamond. The zero-field splitting D is dependent on temperature $^{41,50}$ :

$$
D = c _ {0} + c _ {1} T + c _ {2} T ^ {2} + c _ {3} T ^ {3} + \Delta_ {\text { pressure }} + \Delta_ {\text { strain }}, \tag {5}
$$

where $c_{0}=2.8697$ GHz, $c_{1}=9.7\times10^{-5}$ GHz/K, $c_{2}=-3.7\times10^{-7}$ GHz/K $^{2}$ , $c_{3}=1.7\times10^{-10}$ GHz/K $^{3}$ , $\Delta_{pressure}=1.5\times10^{-6}$ GHz/bar, and $\Delta_{strain}$ is caused by the internal strain effect. $\Delta_{pressure}$ is smaller and can be neglected in vacuum. Figure 1e is the ODMR measured at the pressure of 10 Torr (blue circles) and $6.9\times10^{-6}$ Torr (red squares). The zero-field splitting obtained by fitting can be used to calculate the temperature of the levitated nanodiamond.

The internal temperature T of a levitated nanodiamond is determined by the balance between heating and cooling effects $^{51,52}$ :

$$
A _ {a} = A _ {\text { gas }} p (T - T _ {0}) + A _ {b b} \left(T ^ {5} - T _ {0} ^ {5}\right), \tag {6}
$$

where $A_{a}=\sum_{\lambda}\eta_{\lambda}I_{\lambda}V$ is the heating of the excitation laser ( $\lambda=532~nm$ ) and the detecting laser ( $\lambda=1064~nm$ ), $\eta_{\lambda}$ is the absorption coefficient of nanodiamond and $I_{\lambda}$ is the laser intensity, V is the volume of nanodiamond. The first term at the right side of the equation is the cooling rate caused by gas molecule collisions, $A_{gas}=\frac{1}{2}\kappa\pi R^{2}\upsilon T_{0}\frac{\gamma^{\prime}+1}{\gamma^{\prime}-1},\kappa\approx1$ is the thermal accommodation coefficient, R is the radius of nanodiamond, $\upsilon$ is the mean thermal speed of gas molecules, $\gamma^{\prime}$ is the specific heat ratio ( $\gamma^{\prime}=7/5$ for air near room temperature), p is the pressure, $T_{0}$ is the thermal temperature. The last term is the cooling rate of black-body radiation. $A_{bb}=72\zeta(5)Vk_{B}^{5}/\left(\pi^{2}c^{3}\hbar^{4}\right)\mathrm{Im}\left(\frac{\varepsilon-1}{\varepsilon+2}\right)$ , where $\zeta(5)\approx1.04$ is the Riemann zeta function, $k_{B}$ is the Boltzmann constant, c is the vacuum light speed, $\hbar$ is the reduced Planck's constant, $\varepsilon$ is a constant and time-independent permittivity of nanodiamond across the black-body radiation spectrum. By measuring the internal temperature as a function of the intensities of the 532 nm laser and the 1064 nm laser, the absorption coefficients of the nanodiamond are estimated to be 111 cm $^{-1}$ at 532 nm and 5.87 cm $^{-1}$ at 1064 nm (Supplementary Fig. 3).

# Reporting summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article.

# Data availability

Source data for figures in the main text are provided with this paper in the Source Data file. Other data that support the findings of this study are available from the corresponding author upon request. Source data are provided with this paper.

# References

1. Gonzalez-Ballestero, C., Aspelmeyer, M., Novotny, L., Quidant, R. & Romero-Isart, O. Levitodynamics: levitation and control of microscopic objects in vacuum. Science 374, eabg3027 (2021).   
2. Millen, J., Monteiro, T. S., Pettit, R. & Vamivakas, A. N. Optomechanics with levitated particles. Rep. Prog. Phys. 83, 026401 (2020).   
3. Winstone, G. et al. Levitated optomechanics: a tutorial and perspective. Preprint at https://arxiv.org/abs/2307.11858 (2023).   
4. Romero-Isart, O. et al. Large quantum superpositions and interference of massive nanometer-sized objects. Phys. Rev. Lett. 107, 020405 (2011).   
5. Afek, G., Carney, D. & Moore, D. C. Coherent scattering of low mass dark matter from optically trapped sensors. Phys. Rev. Lett. 128, 101301 (2022).

6. Yin, P. et al. Experiments with levitated force sensor challenge theories of dark energy. Nat. Phys. 18, 1181 (2022).   
7. Geraci, A. A., Papp, S. B. & Kitching, J. Short-range force detection using optically cooled levitated microspheres. Phys. Rev. Lett. 105, 101101 (2010).   
8. Hebestreit, E., Frimmer, M., Reimann, R. & Novotny, L. Sensing static forces with free-falling nanoparticles. Phys. Rev. Lett. 121, 063602 (2018).   
9. Hoang, T. M. et al. Torsional optomechanics of a levitated non-spherical nanoparticle. Phys. Rev. Lett. 117, 123604 (2016).   
10. Zheng, Y. et al. Robust optical-levitation-based metrology of nanoparticle's position and mass. Phys. Rev. Lett. 124, 223603 (2020).   
11. Zhu, S. et al. Nanoscale electric field sensing using a levitated nanoresonator with net charge. Photonics Res. 11, 279 (2023).   
12. Delić, U. et al. Cooling of a levitated nanoparticle to the motional quantum ground state. Science 367, 892 (2020).   
13. Magrini, L. et al. Real-time optimal quantum control of mechanical motion at room temperature. Nature 595, 373 (2021).   
14. Tebbenjohanns, F., Mattana, M. L., Rossi, M., Frimmer, M. & Novotny, L. Quantum control of a nanoparticle optically levitated in cryogenic free space. Nature 595, 378 (2021).   
15. Arita, Y., Mazilu, M. & Dholakia, K. Laser-induced rotation and cooling of a trapped microgyroscope in vacuum. Nat. Commun. 4, 2374 (2013).   
16. Kuhn, S. et al. Optically driven ultra-stable nanomechanical rotor. Nat. Commun. 8, 1670 (2017).   
17. Reimann, R. et al. GHz rotation of an optically trapped nanoparticle in vacuum. Phys. Rev. Lett. 121, 033602 (2018).   
18. Ahn, J. et al. Optically levitated nanodumbbell torsion balance and GHz nanomechanical rotor. Phys. Rev. Lett. 121, 033603 (2018).   
19. Ahn, J. et al. Ultrasensitive torque detection with an optically levitated nanorotor. Nat. Nanotechnol. 15, 89 (2020).   
20. Jin, Y. et al. 6 GHz hyperfast rotation of an optically levitated nanoparticle in vacuum. Photon. Res. 9, 1344 (2021).   
21. Zeng, K., Xu, X., Wu, Y., Wu, X. & Xiao, D. Optically levitated gyroscopes with a mhz rotating micro-rotor. Preprint at https://arxiv.org/abs/2308.09085 (2023).   
22. Ju, P. et al. Near-field GHz rotation and sensing with an optically levitated nanodumbbell. Nano Lett. 23, 10157 (2023).   
23. Ma, Y., Khosla, K. E., Stickler, B. A. & Kim, M. Quantum persistent tennis racket dynamics of nanorotors. Phys. Rev. Lett. 125, 053604 (2020).   
24. Stickler, B. A., Hornberger, K. & Kim, M. Quantum rotations of nanoparticles. Nat. Rev. Phys. 3, 589 (2021).   
25. Yin, Z.-q, Li, T., Zhang, X. & Duan, L. M. Large quantum superpositions of a levitated nanodiamond through spin-optomechanical coupling. Phys. Rev. A 88, 033614 (2013).   
26. Scala, M., Kim, M. S., Morley, G. W., Barker, P. F. & Bose, S. Matter-wave interferometry of a levitated thermal nano-oscillator induced and probed by a spin. Phys. Rev. Lett. 111, 180403 (2013).   
27. Bose, S. et al. Spin entanglement witness for quantum gravity. Phys. Rev. Lett. 119, 240401 (2017).   
28. Marletto, C. & Vedral, V. Gravitationally induced entanglement between two massive particles is sufficient evidence of quantum effects in gravity. Phys. Rev. Lett. 119, 240402 (2017).   
29. Wood, A. et al. Magnetic pseudo-fields in a rotating electron-nuclear spin system. Nat. Phys. 13, 1070 (2017).   
30. Wood, A. A. et al. Quantum measurement of a rapidly rotating spin qubit in diamond. Sci. Adv. 4, eaar7691 (2018).   
31. Chudo, H. et al. Observation of Barnett fields in solids by nuclear magnetic resonance. Appl. Phys. Express 7, 063004 (2014).   
32. Barnett, S. J. Magnetization by rotation. Phys. Rev. 6, 239 (1915).   
33. Barnett, S. J. Gyromagnetic and electron-inertia effects. Rev. Mod. Phys. 7, 129 (1935).

34. Maclaurin, D., Doherty, M. W., Hollenberg, L. C. L. & Martin, A. M. Measurable quantum geometric phase from a rotating single spin. Phys. Rev. Lett. 108, 240403 (2012).   
35. Chen, X.-Y., Li, T. & Yin, Z.-Q. Nonadiabatic dynamics and geometric phase of an ultrafast rotating electron spin. Sci. Bull. 64, 380 (2019).   
36. Ledbetter, M. P., Jensen, K., Fischer, R., Jarmola, A. & Budker, D. Gyroscopes based on nitrogen-vacancy centers in diamond. Phys. Rev. A 86, 052116 (2012).   
37. Zhang, H. & Yin, Z.-Q. Highly sensitive gyroscope based on a levitated nanodiamond. Opt. Express 31, 8139 (2023).   
38. Ma, Y., Hoang, T. M., Gong, M., Li, T. & Yin, Z.-q Proposal for quantum many-body simulation and torsional matter-wave interferometry with a levitated nanodiamond. Phys. Rev. A 96, 023827 (2017).   
39. Rusconi, C. C., Perdriat, M., Hétet, G., Romero-Isart, O. & Stickler, B. A. Spin-controlled quantum interference of levitated nanorotors. Phys. Rev. Lett. 129, 093605 (2022).   
40. Neukirch, L. P., Von Haartman, E., Rosenholm, J. M. & Nick Vami-vakas, A. Multi-dimensional single-spin nano-optomechanics with a levitated nanodiamond. Nat. Photonics 9, 653 (2015).   
41. Hoang, T. M., Ahn, J., Bang, J. & Li, T. Electron spin control of optically levitated nanodiamonds in vacuum. Nat. Commun. 7, 12250 (2016).   
42. Frangeskou, A. C. et al. Pure nanodiamonds for levitated optomechanics in vacuum. N. J. Phys. 20, 043016 (2018).   
43. Delord, T., Huillery, P., Nicolas, L. & Hétet, G. Spin-cooling of the motion of a trapped diamond. Nature 580, 56 (2020).   
44. Perdriat, M., Huillery, P., Pellet-Mary, C. & Hétet, G. Angle locking of a levitating diamond using spin diamagnetism. Phys. Rev. Lett. 128, 117203 (2022).   
45. Delord, T., Nicolas, L., Bodini, M. & Hétet, G. Diamonds levitating in a Paul trap under vacuum: measurements of laser-induced heating via nv center thermometry. Appl. Phys. Lett. 111 (2017).   
46. Conangla, G. P., Schell, A. W., Rica, R. A. & Quidant, R. Motion control and optical interrogation of a levitating single nitrogen vacancy in vacuum. Nano Lett. 18, 3956 (2018).   
47. Perdriat, M. et al. Spin read-out of the motion of levitated electrically rotated diamonds. Preprint at https://arxiv.org/abs/2309.01545 (2023).   
48. Hsu, J.-F., Ji, P., Lewandowski, C. W. & D'Urso, B. Cooling the motion of diamond nanocrystals in a magneto-gravitational trap in high vacuum. Sci. Rep. 6, 30125 (2016).   
49. O'Brien, M. C., Dunn, S., Downes, J. E. & Twamley, J. Magnetomechanical trapping of micro-diamonds at low pressures. Appl. Phys. Lett. 114, 053103 (2019).   
50. Toyli, D. M. et al. Measurement and control of single nitrogen-vacancy center spins above 600 K. Phys. Rev. X 2, 031001 (2012).   
51. Liu, F., Daun, K., Snelling, D. & Smallwood, G. Heat conduction from a spherical nano-particle: status of modeling heat conduction in laser-induced incandescence. Appl. Phys. B 83, 355 (2006).   
52. Chang, D. E. et al. Cavity opto-mechanics using an optically levitated nanosphere. Proc. Natl Acad. Sci. 107, 1005 (2010).   
53. Tycko, R. Adiabatic rotational splittings and berry's phase in nuclear quadrupole resonance. Phys. Rev. Lett. 58, 2281 (1987).   
54. Zhang, Y., Tan, Y.-W., Stormer, H. L. & Kim, P. Experimental observation of the quantum hall effect and berry's phase in graphene. Nature 438, 201 (2005).   
55. Leek, P. J. et al. Observation of berry's phase in a solid-state qubit. Science 318, 1889 (2007).   
56. Xiao, D., Chang, M.-C. & Niu, Q. Berry phase effects on electronic properties. Rev. Mod. Phys. 82, 1959 (2010).   
57. Chudo, H., Matsuo, M., Maekawa, S. & Saitoh, E. Barnett field, rotational Doppler effect, and berry phase studied by nuclear quadrupole resonance with rotation. Phys. Rev. B 103, 174308 (2021).

58. Soshenko, V. V. et al. Nuclear spin gyroscope based on the nitrogen vacancy center in diamond. Phys. Rev. Lett. 126, 197702 (2021).   
59. Jarmola, A. et al. Demonstration of diamond nuclear spin gyroscope. Sci. Adv. 7, eabl3840 (2021).   
60. Degen, C. L., Reinhard, F. & Cappellaro, P. Quantum sensing. Rev. Mod. Phys. 89, 035002 (2017).   
61. Wood, A. A., Hollenberg, L. C. L., Scholten, R. E. & Martin, A. M. Observation of a quantum phase from classical rotation of a single spin. Phys. Rev. Lett. 124, 020401 (2020).   
62. Wood, A. A., Goldblatt, R. M., Scholten, R. E. & Martin, A. M. Quantum control of nuclear-spin qubits in a rapidly rotating diamond. Phys. Rev. Res. 3, 043174 (2021).   
63. Wood, A. A., Stacey, A. & Martin, A. M. dc quantum magnetometry below the Ramsey limit. Phys. Rev. Appl. 18, 054019 (2022).

# Acknowledgements

The authors thank Jun Ye for helpful discussions. T.L. acknowledges the support from the National Science Foundation under Grant PHY-2110591, the Office of Naval Research under Grant No. N00014-18-1-2371, and the Gordon and Betty Moore Foundation, grant DOI 10.37807/gbmf12259. This project is also partially supported by the Laboratory Directed Research and Development program at Sandia National Laboratories, a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia LLC, a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy's National Nuclear Security Administration under Contract No. DE-NA0003525. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government. C.Z. acknowledges the support by NSF ExpandQISE 2328837.

# Author contributions

T.L., Y.J., K.S., and P.J. conceived and designed the project. Y.J. and K.S. built the setup. Y.J. performed measurements and calculations. Y.J., K.S., T.L., X.G., C.Z., and A.J.G. discussed the results. T.L. supervised the project. All authors contributed to the writing of the manuscript.

# Competing interests

The authors declare no competing interests.

# Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41467-024-49175-3.

Correspondence and requests for materials should be addressed to Tongcang Li.

Peer review information Nature Communications thanks Gavin Morley, Anishur Rahman, and the other, anonymous, reviewers for their contribution to the peer review of this work. A peer review file is available.

Reprints and permissions information is available at

http://www.nature.com/reprints

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2024