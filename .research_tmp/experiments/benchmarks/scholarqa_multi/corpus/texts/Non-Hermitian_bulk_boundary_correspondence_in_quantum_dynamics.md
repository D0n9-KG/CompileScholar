# Observation of non-Hermitian bulk-boundary correspondence in quantum dynamics

Lei Xiao, $^{1,2}$ Tianshu Deng, $^{3,4}$ Kunkun Wang, $^{1}$ Gaoyan Zhu, $^{1,2}$ Zhong Wang, $^{5,*}$ Wei Yi, $^{3,4,\dagger}$ and Peng Xue $^{1,\ddagger}$   
$^{1}$ Beijing Computational Science Research Center, Beijing 100084, China   
$^{2}$ Department of Physics, Southeast University, Nanjing 211189, China   
$^{3}$ CAS Key Laboratory of Quantum Information, University of Science and Technology of China, Hefei 230026, China   
$^{4}$ CAS Center For Excellence in Quantum Information and Quantum Physics   
$^{5}$ Institute for Advanced Study, Tsinghua University, Beijing, 100084, China

Bulk-boundary correspondence, a central principle in topological matter relating bulk topological invariants to edge states, breaks down in a generic class of non-Hermitian systems that have so far eluded experimental effort. Here we theoretically predict and experimentally observe non-Hermitian bulk-boundary correspondence, a fundamental generalization of the conventional bulk-boundary correspondence, in discrete-time non-unitary quantum-walk dynamics of single photons. We experimentally demonstrate photon localizations near boundaries even in the absence of topological edge states, thus confirming the non-Hermitian skin effect. Facilitated by our experimental scheme of edge-state reconstruction, we directly measure topological edge states, which match excellently with non-Bloch topological invariants calculated from localized bulk-state wave functions. Our work unequivocally establishes the non-Hermitian bulk-boundary correspondence as a general principle underlying non-Hermitian topological systems, and paves the way for a complete understanding of topological matter in open systems.

Topological phases exhibit remarkable properties due to the presence of robust edge states at boundaries. These topologically protected edge states are related to bulk topological invariants through the principle of bulk-boundary correspondence, which is fundamentally important in topological matter $[1, 2]$ . Intriguingly, recent theoretical studies have shown that the widely held bulk-boundary correspondence apparently breaks down in a broad class of non-Hermitian topological systems $[3–9]$ , where the bulk eigenstates are generally localized near boundaries (dubbed the non-Hermitian skin effect $[4, 6]$ ). To correctly account for topological edge states therein, an essential generalization of the conventional bulk-boundary correspondence must be introduced, where the non-Bloch-wave character of bulk states necessitates a new formulation of bulk topological invariants. With rapid progresses in the experimental implementation of non-Hermiticity in synthetic systems ranging from photonics $[10–20]$ and acoustics $[21]$ to vacancy centers in solids $[22]$ and cold atoms $[23]$ , non-Hermitian topological systems have generated intense interests recently $[24–33]$ . However, the fundamentally important non-Hermitian bulk-boundary correspondence and the underlying non-Hermitian skin effect have never been experimentally demonstrated yet in any system.

Here we theoretically characterize and experimentally demonstrate the non-Hermitian skin effect and non-Hermitian bulk-boundary correspondence in discrete-time non-unitary quantum walks of single photons. We implement a novel non-unitary Floquet operator with polarization-dependent photon loss that supports Floquet topological phases protected by chiral symmetry. Under a domain-wall configuration, we demonstrate non-Hermitian skin effect by observing that the walker becomes dynamically localized at the boundary with the onset of non-Hermiticity, regardless of its initial state and even in the absence of topological edge states. To demonstrate the non-Hermitian bulk-boundary correspondence, we theoretically calculate the non-Bloch topological invariants defined in a generalized Brillouin zone [4], taking into account the generic deviations of localized bulk states from Bloch waves. For our non-unitary quantum walks, two distinct topological invariants exist, corresponding to edge states with quasienergies $\epsilon = 0$ and $\epsilon = \pi$ , respectively [34, 35]. As both topological edge states and bulk states are localized, it is challenging to unambiguously demonstrate the presence of edge states following the common practice of measuring localized photon population. To overcome this difficulty, we devise a new detection scheme involving quantum-state reconstruction at each time step, which allows us to differentiate edge states from bulk states and fully resolve the quasienergy as well as the internal degrees of freedom of topological edge states. The experimentally measured topological edge states match excellently with the corresponding non-Bloch topological invariants. In view of the fundamental importance of bulk-boundary correspondence in the study of topological systems, our results pave the way for future studies of topological effects in non-Hermitian systems, and provide a solid groundwork for engineering topological states in open systems.

# Results

Chiral symmetric non-unitary quantum walk. We focus on a chiral symmetric non-unitary quantum walk on a one-dimensional lattice governed by the Flo-

a   
![](images/3e8e62fa89a17c20028076e7689c10957b4e0cb8b7830cc12f9a7859aa913044.jpg)

<details>
<summary>text_image</summary>

-N
N-1
θ₁,₂ᴸ
-2
-1
0
1
θ₁,₂ᴿ
</details>

C   
![](images/1ab86d6af63eff92fb40789e36f4e2c0e5fd173670d84ec990a6ef46cc04e4e0.jpg)

<details>
<summary>scatter</summary>

| β     | Im(β) |
|-------|-------|
| L1    | 0     |
| L2    | 0     |
</details>

b   
![](images/621744162bf3f9dc467063effef1a852fe15629df170132d2599cf62fb82d26b.jpg)

<details>
<summary>line</summary>

| γ    | ψ₊(x) | ψ₋(x) |
|------|-------|-------|
| 0    | 0.0   | 0.0   |
| 0.05 | 0.4   | 0.8   |
| 0.5  | 0.8   | 0.4   |
</details>

![](images/efa1727d549aaf866b9a489f791057aa6a00107048520555dd9dc93ca52b975e.jpg)

<details>
<summary>scatter</summary>

| Re(β) | Im(β) | Series     |
|-------|-------|------------|
| -0.5  | 0.2   | β_R,1      |
| 0.0   | 0.3   | β_R,1      |
| 0.2   | 0.1   | β_R,2      |
| -0.3  | -0.1  | β_R,2      |
| 0.4   | -0.2  | β_R,2      |
| -0.6  | 0.4   | β_R,1      |
| 0.6   | 0.5   | β_R,1      |
| -0.7  | -0.3  | β_R,2      |
| 0.8   | -0.4  | β_R,2      |
| -0.9  | 0.6   | β_R,1      |
| 0.9   | 0.7   | β_R,1      |
| -1.0  | -0.5  | β_R,2      |
| 1.0   | -0.6  | β_R,2      |
</details>

FIG. 1. Quantum walks with non-Hermitian skin effect. a Schematic illustration of the domain-wall configuration. The lattice sites are labelled in a cyclic fashion, with the two boundaries located near $x = 0$ and $x = -N$ ( $x = N - 1$ ). b Spatial distribution of the projected norms of bulk (black) and edge (red) states under various gain-loss parameter $\gamma$ , with $\psi_{\pm}(x) = |(\langle x| \otimes \langle \pm |)| |\psi\rangle|$ . Here $|\pm\rangle = (|0\rangle \pm |1\rangle)/\sqrt{2}$ are eigenstates of the chiral-symmetry operator, and the bulk-state wave function $|\psi\rangle$ is defined in Eq. (2). c Generalized Brillouin zones parameterized by the spatial-mode functions $\beta_{\alpha,j}$ for $\gamma = 0.5$ . Whereas the standard Brillouin zones are indicated by unit circles (solid black line), two distinct generalized Brillouin zones exist for each given bulk. Parameters for the numerical calculations in b and c are: $N = 30$ , $\theta_1^R = 0.1875\pi$ , $\theta_1^L = -0.3333\pi$ , $\theta_2^R = 0.2\pi$ , and $\theta_2^L = -0.6667\pi$ .

![](images/41bcda16c87112edbaaf3409e0b5368266ea1bf08aba499ae5b691901b3a8a37.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    Laser --> SPDC
    SPDC --> Reference
    Reference --> R1["R(θ₁/2)S₂R(θ₂/2)MR(θ₂/2)S₁R(θ₁/2)"]
    R1 --> M1["APD"]
    R1 --> M2["HWP"]
    R1 --> M3["PPBS"]
    R1 --> M4["BRIDGE"]
    M1 --> H1["H1"]
    M1 --> H2["H2"]
    M1 --> H3["H3"]
    M1 --> M3["M3"]
    M2 --> H4["H4"]
    M3 --> H5["H5"]
    H4 --> M4
    H5 --> M4
    style Laser fill:#f9f,stroke:#333
    style SPDC fill:#ccf,stroke:#333
    style Reference fill:#cfc,stroke:#333
    style Reference fill:#fcc,stroke:#333
    style Reference fill:#ffc,stroke:#333
    style Reference fill:#cff,stroke:#333
    style Reference fill:#ffc,stroke:#333
    style Reference fill:#cfc,stroke:#333
    style Reference fill:#fcc,stroke:#333
    style Reference fill:#ffc,stroke:#333
    style Reference fill:#cfc,stroke:#333
    style Reference fill:#ffc,stroke:#333
    style Reference fill:#cfc,stroke:#333
    style Reference fill:#ffc,stroke:#333
    style Reference fill:#cfc,stroke:#333
    style Reference fill:#ffc,stroke:#333
    style Reference fill:#cfc,stroke:#333
    style Reference fill:#ffc,stroke:#334
```
</details>

FIG. 2. Experimental implementation. A pair of photons is created via spontaneous parametric down conversion, of which one serves as a trigger and the other (the walker photon) is projected onto one of the polarization states via a polarizing beam splitter (PBS) and a half-wave plate (HWP) labelled $H_{0}$ . It then proceeds through an interferometric network, composed of HWPs, beam displacers (BDs), and partially polarizing beam splitter (PPBS). For each detection module $M_{i}$ (i = 1, 2, 3, 4), the photon is detected by avalanche photodiodes (APDs), in coincidence with the trigger photon. As an illustration, we show the setup for the first step of the quantum-walk configuration as well as a conceptual representation of the multiple-step quantum-walk dynamics (lower-left corner).

quet operator

$$
U = R (\frac {\theta_ {1}}{2}) S _ {2} R (\frac {\theta_ {2}}{2}) M R (\frac {\theta_ {2}}{2}) S _ {1} R (\frac {\theta_ {1}}{2}), \tag {1}
$$

where the coin operator $R(\theta)$ rotates coin states by $\theta$ about the y axis, and $S_{1}(S_{2})$ shifts the walker in the coin state $|1\rangle(|0\rangle)$ to the right (left) by one lattice site. Non-unitarity is introduced through M = $\mathbb{1}_{w} \otimes (e^{\gamma}|0\rangle\langle0| + e^{-\gamma}|1\rangle\langle1|)$ , where $1_{w}$ is the identity operator in lattice modes and $\gamma$ is a tunable gain-loss parameter.

The Floquet operator U has a chiral symmetry with $\sigma_{x}U\sigma_{x}=U^{-1}$ ( $\sigma_{x}$ is the standard Pauli matrix), and the entries of U are real-valued, meaning that its eigenvalues can also be purely real. More importantly, when the lattice has boundaries, U exhibits non-Hermitian skin effect,

![](images/f43957f7bceea465adf7c129d472bd9b060b93bcf0036b7e06e44bfbc8f87a1a.jpg)  
FIG. 3. Experimental demonstration of the non-Hermitian skin effect. (Upper row) Spatial probability distributions of a seven-step unitary quantum walk with $\gamma = 0$ and different initial states $|\varPhi(0)\rangle$ . Both the time-dependent probability distribution $\mathbf{a}$ , $\mathbf{c}$ and the distribution at the last step $\mathbf{b}$ , $\mathbf{d}$ are shown. (Lower row) Spatial probability distributions of a seven-step non-unitary quantum walk with $\gamma = 0.2746$ and different initial states $|\varPhi(0)\rangle$ . Both the time-dependent probability distribution $\mathbf{e}$ , $\mathbf{g}$ and the distribution at the last step $\mathbf{f}$ , $\mathbf{h}$ are shown. The coin parameters are: $\theta_1^R = 0.0667\pi$ , $\theta_1^L = 0.5625\pi$ , $\theta_2^R = 0$ , and $\theta_2^L = \pi$ .

under which all eigensates of U become localized near the boundaries. This feature distinguishes U from previous experimentally realized non-unitary quantum walks $[10–15]$ . The non-Hermitian skin effect of U is related to that of its effective Hamiltonian defined through $U = e^{-iH_{eff}}$ , since quantum-walk dynamics under U can be regarded as a stroboscopic simulation of the non-unitary dynamics driven by the non-Hermitian Hamiltonian $H_{eff}$ .

To investigate non-Hermitian skin effect and non-Hermitian bulk-boundary correspondence of $H_{\mathrm{eff}}$ , we consider a domain-wall configuration on a circle as shown in Fig. 1a, where the left and right bulks have different coin parameters denoted as $\theta_{1(2)}^{L}$ and $\theta_{1(2)}^{R}$ , respectively. The lattice sites are indexed in a cyclic fashion, with $x \in J_L$ ( $J_L = \{x \in \mathbb{Z} | -N \leqslant x \leqslant -1\}$ ) for the left bulk, and $x \in J_R$ ( $J_R = \{x \in \mathbb{Z} | 0 \leqslant x \leqslant N - 1\}$ ) for the right. In Fig. 1b, we numerically demonstrate that, under such a configuration, both bulk and topological edge states are localized at the boundaries for finite $\gamma$ , a hallmark of the non-Hermitian skin effect. Here we identify topological edge states by their quasienergies $\epsilon = 0, \pi$ and coin states $|\pm\rangle = (|0\rangle \pm |1\rangle)/\sqrt{2}$ . It follows that the bulk-state wave functions can be written as [4, 36]

$$
| \psi \rangle = \sum_ {\alpha} \sum_ {x \in J _ {\alpha}, j} \beta_ {\alpha , j} ^ {x} | x \rangle \otimes | \phi_ {j} ^ {\alpha} \rangle_ {c} \quad (\alpha = L, R), \tag {2}
$$

where $\beta_{L(R),j}$ is the spatial-mode function for the j-th mode in the left (right) bulk, $|\phi_{j}^{L(R)}\rangle_{c}$ is the corresponding coin state. For the domain-wall configuration considered here, two spatial modes (labelled by j = 1,2) exist for each bulk (see Supplemental Information). In the unitary limit with $\gamma = 0$ , $|\beta_{\alpha,1}| = |\beta_{\alpha,2}| = 1$ and we identify $\beta_{\alpha,j}$ as $e^{ik}$ , with k being the quasimomentum in the first Brillouin zone and $|\psi\rangle$ reduced to the Bloch wave functions. For a finite $\gamma$ and under the domain-wall configuration, however, $|\beta_{\alpha,j}| \neq 1$ , which underlies the non-Hermitian skin effect.

According to the topological band theory, the topological invariants of the Floquet operator $U$ read [37, 38]

$$
\nu_ {\epsilon} = \frac {1}{4 \pi i} \int_ {0} ^ {2 \pi} \operatorname{Tr} \left[ \sigma_ {x} \left(\overline {{{U}}} _ {\epsilon} (k)\right) ^ {- 1} d \left(\overline {{{U}}} _ {\epsilon} (k)\right) \right] (\epsilon = 0, \pi), \tag {3}
$$

where $\overline{U}_{\epsilon}(k)$ is the periodized Floquet operator in quasi-momentum k space associated with the non-Hermitian effective Hamiltonian $H_{eff}^{\epsilon}=i\ln_{\epsilon}U$ , with a branch cut at $\epsilon$ (see Methods). Correspondingly, $\nu_{0(\pi)}$ dictates topological edge state with quasienergy $\epsilon=0\left(\pi\right)$ .

For our domain-wall system with non-Hermitian skin effect, however, these Bloch topological invariants cannot correctly predict topological edge states. The exponen-

tial localization of the nominal bulk states at the domain wall suggests that the conventional Brillouin zone with real-valued k should not be the appropriate starting point. Instead, since the Bloch phase factor $e^{ik}$ is replaced by $\beta_{\alpha,j} := |\beta_{\alpha,j}(p_j^\alpha)|e^{ip_j^\alpha}$ , one should resort to the generalized Brillouin zone [4], parameterized by the phase parameter $p_j^\alpha$ . As $p_j^\alpha$ varies, the allowed values of $\beta_{\alpha,j}$ form one-dimensional trajectories in the complex plane, with the bulk quasienergy spectrum of the system given by eigenvalues of $H_{\mathrm{eff}}(e^{ik} \to \beta_{\alpha,j})$ . Thus, these trajectories (illustrated in Fig. 1c) fundamentally generalize the concept of Brillouin zones in one-dimensional Hermitian systems to generic one-dimensional non-Hermitian settings, and are identified as the generalized Brillouin zones. The non-Bloch topolgoical invariants $\tilde{\nu}_{0(\pi)}$ are defined over these generalized Brillouin zones (see Methods).

As shown in Fig. 1c, two distinct generalized Brillouin zones can be found for a single bulk, corresponding to the two distinct spatial modes j = 1 and 2, respectively. However, we have numerically checked that Eq. (3) yields the same set of non-Bloch topological invariants $\tilde{\nu}_{0(\pi)}$ when integrated over different generalized Brillouin zones of the same bulk (see Supplemental Information). As we experimentally demonstrate in the following, these non-Bloch topological invariants correctly predict the existence of both types of topological edge states with $\epsilon = 0, \pi$ , which unambiguously embodies the non-Hermitian bulk-boundary correspondence.

Experimental demonstration of non-Hermitian skin effect. We experimentally investigate non-unitary quantum-walk dynamics governed by U using a single-photon interferometer setup, as illustrated in Fig. 2. The coin states are encoded in the photon polarizations, with $|0\rangle$ and $|1\rangle$ corresponding to the horizontally and vertically polarized photons, respectively. The walker states are encoded in the spatial mode of the photons. We experimentally realize a mode-selective loss operator $M_{E} = 1_{w} \otimes (|0\rangle\langle0| + \sqrt{1 - p}|1\rangle\langle1|)$ , which enforces a partial measurement in the basis of $\{|0\rangle, |1\rangle\}$ at every time step. Since $M = e^{\gamma}M_{E}$ with $\gamma = -\frac{1}{4}\ln(1 - p)$ , it is straightforward to map the experimentally implemented dynamics to those under U by multiplying a time-dependent factor $e^{\gamma t}$ .

We first demonstrate non-Hermitian skin effect through the local accumulation of population at long times in the absence of topological edge states. We study seven-step quantum-walk dynamics and focus on a single boundary in the domain-wall configuration. Initializing the walker at x = -1 but with different coin states, we first implement unitary quantum walks with $\gamma = 0$ . As illustrated in Fig. 3a-d, the spatial photon distribution becomes increasingly non-local in time. In contrast, when choosing a finite $\gamma$ (see Fig. 3e-h), the population distribution after seven steps are still localized near the boundary, regardless of the initial coin state. Crucially, for both the unitary and non-unitary cases, we choose the coin parameters such that no topological edge states exist, which is numerically confirmed by the absence of zero- or $\pi$ -modes in the quasienergy spectra. The localization of the population distribution is therefore the unequivocal manifestation of the localization of bulk eigenstates by the non-Hermitian skin effect.

Non-Hermitian bulk-boundary correspondence. We now proceed to a systematic study of non-Hermitian bulk-boundary correspondence by matching the presence of zero- and $\pi$ -modes with non-Bloch topological invariants. However, it is challenging to differentiate topological edge states from bulk states by measuring probability distribution, as both states are localized at the boundary under the non-Hermitian skin effect.

To overcome the difficulty, we develop a novel detection scheme which allows us to extract edge-state wave functions with spatial and coin-state resolution. Formally, writing the time-evolved wave function as $|\varPhi(t)\rangle$ (see Methods), we construct the time-integrated wave functions

$$
| \varPhi_ {\epsilon} (t) \rangle = \sum_ {t ^ {\prime} = 0} ^ {t} \frac {e ^ {i \epsilon t ^ {\prime}}}{t + 1} | \varPhi (t ^ {\prime}) \rangle (\epsilon = 0, \pi). \qquad (4)
$$

As we show in the Methods section, in the weighted summation of Eq. (4), time-dependent phases of bulk states cancel out at long times, provided that the quasienergy spectrum underlying the dynamics is purely real. The resulting $|\varPhi_{\epsilon}(t)\rangle$ would converge to the corresponding edge-state wave function with quasienergy $\epsilon$ at sufficiently large time steps, given that the initial state has a finite overlap with the corresponding edge state. Under chiral symmetry, topological edge states are necessarily in the coin state $|\pm\rangle$ , which are eigenstates of the symmetry operator. Projecting the integrated wave function Eq. (4) onto the chiral-symmetry basis $|\pm\rangle$ , we measure the quantity (see Methods for details)

$$
\Phi_ {\epsilon , \mu} (x) = \left| \left(\langle x | \otimes \langle \mu |\right) \mid \Phi_ {\epsilon} (t) \rangle \right| \quad (\mu = \pm), \tag {5}
$$

which, as we show in the following, converges to the norm of the corresponding edge-state wave function (apart from a global scaling factor) after seven steps of a quantum walk. Since a prerequisite for our detection scheme is the presence of purely real energy spectrum, for the following experiments, we choose the coin parameters under which quasienergy spectra of our domain-wall configuration are purely real.

First, we choose the parameters such that a $\pi$ -mode edge state exists in the numerically calculated quasienergy spectrum (see Fig. 4a). Incidentally, the two generalized Brillouin zones in each bulk overlap with each other in this case, as illustrated in Fig. 4b. We initialize the walker in three different initial states and let it evolve

a   
![](images/16f2733a3dbfe6a2f308fc824d849e6d80b9d64efda8478a47e3910da9d55379.jpg)

$\left|\Phi (0)\right\rangle = \left|0\right\rangle \otimes \left| + \right\rangle$   
![](images/1fbd37e561336f228a8fd73f465ae15135e8136b5bc067b712368f769a55a601.jpg)

<details>
<summary>bar</summary>

| Position | ε=0, μ=+ | ε=0, μ=- | ε=π, μ=+ | ε=π, μ=- |
| -------- | -------- | -------- | -------- | -------- |
| -1       | 0.0      | 0.0      | 0.0      | 0.0      |
| 1        | 0.0      | 0.0      | 0.5      | 0.0      |
| 3        | 0.0      | 0.0      | 0.3      | 0.1      |
| 5        | 0.0      | 0.0      | 0.1      | 0.0      |
</details>

$\left|\Phi (0)\right\rangle = \left|0\right\rangle \otimes \left|0\right\rangle$   
![](images/8d67e1d56da43f2d66ddbcf6dbda467289f741d26cde7993333ff0187d328f14.jpg)

<details>
<summary>bar</summary>

| Position | Value |
| -------- | ----- |
| -1       | 0.5   |
| 1        | 0.5   |
| 3        | 0.2   |
</details>

$\left|\Phi (0)\right\rangle = \left|-1\right\rangle \otimes \left|1\right\rangle$   
![](images/4e677e6996c4fd91a76118a165655cdc6adedf928233c7a314c8123af93d73a0.jpg)

<details>
<summary>bar</summary>

| Position | Value |
| -------- | ----- |
| -1       | 0.6   |
| 1        | 0.4   |
| 3        | 0.2   |
</details>

b   
![](images/523964a068392354847cbb3f7fdac8fde0d01f0174a493f35415b69532cc66be.jpg)

<details>
<summary>scatter</summary>

| Point | Re(β) | Im(β) |
|-------|-------|-------|
| Blue dot | -1.0 | 1.0 |
| Red line | 0.0 | 0.0 |
| Blue dotted line | -0.5 | 0.5 |
| Green line | 0.0 | 0.0 |
| Orange dotted line | 0.0 | 0.0 |
</details>

![](images/98238f0bdd8f0ef35bf83daeaa0f26125523d4b7709594d590f0897ecd25b219.jpg)

<details>
<summary>bar_line</summary>

| Position | Simulation | Experiment | Diagonalization |
| -------- | ---------- | ---------- | --------------- |
| -6       | 0.47       | 0.00       | 0.00            |
| -4       | 0.00       | 0.02       | 0.00            |
| -2       | 0.00       | 0.10       | 0.00            |
| 0        | 0.50       | 0.45       | 0.50            |
| 2        | 0.13       | 0.13       | 0.13            |
| 4        | 0.02       | 0.02       | 0.02            |
| 6        | 0.01       | 0.01       | 0.01            |
| 8        | 0.00       | 0.00       | 0.00            |
</details>

![](images/59f8885c1be49cc77517a1d8e11e7787e8bf8ffa04593c517a8dc323a3c07f3d.jpg)

<details>
<summary>histogram</summary>

| Position | Frequency |
| -------- | --------- |
| -6       | 0.48      |
| -4       | 0.02      |
| -2       | 0.08      |
| 0        | 0.45      |
| 2        | 0.19      |
| 4        | 0.06      |
| 6        | 0.01      |
| 8        | 0.00      |
</details>

![](images/5acada8c04495aa442612ce98ee606079bd8d59513a20a5b71dc1ec4f1577d5b.jpg)

<details>
<summary>histogram</summary>

| Position | Frequency |
| -------- | --------- |
| -8       | 0.48      |
| -6       | 0.01      |
| -4       | 0.03      |
| -2       | 0.35      |
| 0        | 0.16      |
| 2        | 0.07      |
| 4        | 0.02      |
| 6        | 0.01      |
</details>

FIG. 4. Non-Bloch bulk-boundary correspondence for edge states with $\epsilon = \pi$ . a Quasienergy spectrum (black), and winding-number differences for zero- (red) and $\pi$ -modes (blue). The non-Bloch winding numbers (solid) are different from the Bloch ones (dashed). The blue dot indicates the parameter for c. b Generalized Brillouin zones characterized by $\beta_{\alpha,j}$ on the complex plane. c (Upper row) Experimentally measured $\varPhi_{\epsilon,\mu}(x)$ after the seventh step with different initial states. (Lower row) Comparison between experimentally-measured and numerically-simulated $\varPhi_{\pi, + }(x)$ . We also show the scaled norms of the edge state with $\epsilon = \pi$ after the seventh step, calculated by diagonalizing a domain-wall system with $N = 15$ . Norms of the edge states from diagonalization are scaled to fit the central peak of the numerically-simulated $\varPhi_{\pi, + }(x)$ . For all panels, we have $\theta_1^L = 0.5625\pi$ , $\theta_2^R = 0.25\pi$ , $\theta_2^L = 0.75\pi$ , and $\gamma = 0.2746$ . For b and c, $\theta_1^R = 0.18\pi$ .

for seven steps before we measure $\Phi_{\epsilon,\mu}(x)$ . As shown in the upper row of Fig. 4c, regardless of initial states, a prominent peak exists only for the $\Phi_{\pi,+}(x)$ measurement, clearly indicating the presence of a $\pi$ -mode edge state with the coin state $|+\rangle$ . In the lower row of Fig. 4c, we see that the measured $\Phi_{\pi,+}(x)$ at the seventh step matches well with the result from numerical simulations for a seven-step quantum walk. Moreover, $\Phi_{\pi,+}(x)$ approaches the scaled norms of the edge-state wave function, calculated by diagonalizing a finite (N=15) domain-wall system under the same parameters. From Fig. 4a, we see that, for our chosen parameters, only the non-Bloch winding numbers correctly predict the presence of the $\pi$ -mode edge state, with $(\Delta\tilde{\nu}_{0},\Delta\tilde{\nu}_{\pi})=(0,-1)$ . Here $\Delta\tilde{\nu}_{0(\pi)}=\tilde{\nu}_{0(\pi)}^{L}-\tilde{\nu}_{0(\pi)}^{R}$ . For comparison, the corresponding Bloch winding numbers are $(\Delta\nu_{0},\Delta\nu_{\pi})=(0,-1/2)$ .

We then perform a series of experiments with parameters where: i) both $\epsilon = 0$ and $\pi$ edge states exist; ii) no topological state exists. As shown in Fig. 5a, in both cases, the theoretically calculated non-Bloch topological invariants correctly predict the existence of topological edge states, whereas the Bloch topological invariants fail to do so. Specifically, in the first case where both types of edge states exist (see Fig. 5b), $(\Delta\tilde{\nu}_{0},\Delta\tilde{\nu}_{\pi})=(-1,-1)$ for the non-Bloch winding numbers, and $(\Delta\nu_{0},\Delta\nu_{\pi})=(-1/2,-1/2)$ for the Bloch ones. In the second case where there is no edge state (see Fig. 5c), $(\Delta\tilde{\nu}_{0},\Delta\tilde{\nu}_{\pi})=(0,0)$ for the non-Bloch winding numbers, and $(\Delta\nu_{0},\Delta\nu_{\pi})=(-1/2,-1/2)$ for the Bloch ones. Note that the generalized Brillouin zones for the two cases in Fig. 5b and 5c are the same, as shown in Fig. 5d. In Figs. 5e and 5f, we show the measured $\Phi_{0,+}(x)$ and $\Phi_{\pi,+}(x)$ after the last time step, which agree well with results from numerical simulations and with scaled norms of the numerically calculated edge-state wave functions. Our results thus unambiguously confirm the non-Hermitian bulk-boundary correspondence.

# Discussion.

We have experimentally established non-Hermitian skin effect and non-Hermitian bulk-boundary correspondence in discrete-time non-unitary quantum-walk dynamics. Our theoretical approach can be extended to treat general non-unitary dynamics, which host rich topological phenomena beyond their unitary counterparts. Our experiment highlights the versatility of quantum-walk dynamics in studying non-Hermitian

![](images/effd6dfe5d515723fde8815de3731d3f4035d918ea8073359321813d571ecfe0.jpg)  
FIG. 5. Evidence for non-Bloch bulk-boundary correspondence. a Quasienergy spectrum, and winding-number differences for zero- and $\pi$ -modes between the two bulks, with the parameters: $\theta_1^L = 0.5625\pi$ , $\theta_2^R = 0$ , $\theta_2^L = \pi$ , and $\gamma = 0.2746$ . Representations of color and line shapes are the same as those in Fig. 4. The cyan dot with $\theta_1^R = -0.0667\pi$ corresponds to the parameter used in b, e, f. The magenta dot with $\theta_1^R = 0.0667\pi$ corresponds to the parameter used in c. b, c Experimentally measured $\varPhi_{\epsilon,\mu}(x)$ for different $\theta_1^R$ after the seventh step, with the initial state $|0\rangle \otimes |+\rangle$ . d Generalized Brillouin zones on the complex plane. e, f Comparison between experimentally-measured and numerically-calculated $\varPhi_{\epsilon,+}(x)$ , as well as the scaled norms of the corresponding edge state after the seventh step.

topological systems. Specifically, the detection scheme developed in this work allows us to differentiate topological edge states from bulk states localized by the non-Hermitian skin effect, which is valuable for the exploration of non-Hermitian topological systems in general. Overall, our experimental observation of non-Hermitian bulk-boundary correspondence and the theoretical elucidation of its mechanism is of fundamental importance for the understanding of topological phenomena in open systems. From a practical perspective, the new bulk-boundary correspondence established here will be useful in open-system-based topological designs such as topological lasers.

# Methods

Experimental implementation. As illustrated in Fig. 2, the walker photon is initialized in the spatial mode x = -1 and projected onto one of the polarization states $|\pm\rangle$ , $|0\rangle$ or $|1\rangle$ via a PBS and an HWP labelled $H_{0}$ . The coin operator R is implemented by a set of HWPs. The shift operator $S_{1}$ ( $S_{2}$ ) is implemented by a BD [12–15]. The mode-selective loss operator $M_{E}$ is implemented by a PPBS. At each step, after applying $M_{E}$ , photons in the state $|1\rangle$ are reflected by the PPBS with a probability p and the rests continue propagating in the quantum-walk dynamics. The photon is detected by APDs in the detection modules $M_{i}$ (i = 1, 2, 3, 4), in coincidence with the trigger photon. Details on these detection modules are given in the Supplemental Information. Photon counts give measured probabilities after correcting for relative efficiencies of the different APDs.

Non-Bloch topological invariants. Bloch topological invariants of the quantum-walk dynamics are calculated through the Fourier component $U(k)$ of the Floquet operator U in Eq. (1). We first define two branches of effective Hamiltonians through $H_{\mathrm{eff}}^{\epsilon}(k)=i\ln_{\epsilon}U(k)$ , where $\epsilon=0,\pi$ indicates the location of the branch cut. It follows that the imaginary part of the function $\ln_{\epsilon}\lambda$ belongs to the range $[\epsilon-2\pi,\epsilon)$ . We then invoke periodized time evolution operators [34, 37, 38] to generate topological invariants. For our chiral-symmetric case, we only need its half-period value [37, 38], defined as

$$
\overline {{U}} _ {\epsilon} (k) = U _ {\frac {1}{2}} (k) e ^ {\frac {i}{2} H _ {\mathrm{eff}} ^ {\epsilon}}, \tag {6}
$$

where the half-period operator $U_{\frac{1}{2}}(k)$ is the Fourier component of $U_{\frac{1}{2}} := M^{\frac{1}{2}} R(\frac{\theta_{2}}{2}) S_{1} R(\frac{\theta_{1}}{2})$ , with $M^{\frac{1}{2}} = \begin{pmatrix} e^{\frac{\gamma}{2}} & 0 \\ 0 & e^{-\frac{\gamma}{2}} \end{pmatrix}$ . The Bloch winding numbers are then given by Eq. (3) in the main text, where the integration is over the Brillouin zone with $k \in [0, 2\pi)$ .

To calculate the non-Bloch topological invariants, it is necessary to generalize the Brillouin zone to the complex plane using information from the now localized bulk states. Following the framework outlined in the main text, we replace the standard Bloch phase factor $e^{ik}$ by $\beta_{\alpha,j}=|\beta_{\alpha,j}(p_{j}^{\alpha})|e^{ip_{j}^{\alpha}}$ , whose trajectory forms the j-th generalized Brillouin zone of the corresponding bulk. In practice, one needs to make the substitution $k=p_{j}^{\alpha}-i\ln|\beta_{\alpha,j}(p_{j}^{\alpha})|$ in Eq. (3) and integrate over the corresponding generalized Brillouin zone. We note that for the domain-wall system here, two generalized Brillouin zones exist for each given bulk, both yielding the same non-Bloch winding numbers. Details for the calculation of $\beta_{\alpha,j}$ are given in the Supplemental Information.

Finally, we have checked that topological invariants calculated via periodized Floquet operators are the same as those defined with Floquet operators in different time frames $[35]$ , provided that the generalized Brillouin zones are used. As we detail in the Supplemental Information, by defining a shifted Floquet operator

$$
U ^ {\prime} = M ^ {\frac {1}{2}} R (\frac {\theta_ {2}}{2}) S _ {1} R (\theta_ {1}) S _ {2} R (\frac {\theta_ {2}}{2}) M ^ {\frac {1}{2}}, \tag {7}
$$

and calculating its winding number $\tilde{\nu}^{\prime}$ over the generalized Brillouin zone, we have

$$
\tilde {\nu} _ {0 (\pi)} = \frac {\tilde {\nu} \pm \tilde {\nu} ^ {\prime}}{2}. (8)
$$

Here $\tilde{\nu}$ is the winding number of U calculated over the generalized Brillouin zone.

Detection scheme for edge states under non-Hermitian skin effect. In previous experiments of topological quantum-walk dynamics, topological edge states were detected by observing localization of the probability distribution near the boundary at long times. However, such a practice fails in our system, as both the edge and bulk states are localized. To unambiguously detect edge states in quantum-walk dynamics, we develop a detection scheme based on the fact that quasienergy of edge states is either $\epsilon = 0$ or $\epsilon = \pi$ , dictated by chiral symmetry. Our scheme relies on reconstructing time-evolved states at each time step. A weighted summation of wave functions at all time steps then selectively retain contribution of edge states with either $\epsilon = 0$ or $\epsilon = \pi$ , as contributions from all other states cancel out due to interference.

More concretely, the time-dependent wave function can be written as

$$
| \varPhi (t) \rangle = U ^ {t} | \varPhi (0) \rangle = \sum_ {n} e ^ {- i E _ {n} t} \varPhi_ {n} | \psi_ {n} \rangle , \qquad (9)
$$

where $\Phi_n = \langle \chi_n |\Phi(0) \rangle$ , $U|\psi_n\rangle = e^{-iE_n}|\psi_n\rangle$ , and $\langle \chi_n|U^{-1} = \langle \chi_n|e^{iE_n}$ . By definition, $|\psi_n\rangle$ ( $\langle \chi_n|$ ) is the right (left) eigenvector of $U$ [39].

Substituting Eq. (9) into Eq. (4), we have, for large t,

$$
\left| \Phi_ {\epsilon} (t) \right\rangle = \sum_ {n} f _ {\epsilon} \left(E _ {n}\right) \Phi_ {n} \left| \psi_ {n} \right\rangle , \tag {10}
$$

where

$$
f _ {\epsilon} (E _ {n}) = \left\{ \begin{array}{l l} \frac {1}{t + 1} \frac {1 - \exp [ - i (E _ {n} - \epsilon) (t + 1) ]}{1 - \exp [ i (E _ {n} - \epsilon) ]} & E _ {n} \neq \epsilon \\ 1 & E _ {n} = \epsilon \end{array} , \right. \tag {11}
$$

with $\epsilon = 0, \pi$ .

When the spectrum $E_{n}$ is completely real, $\lim_{t\to\infty}f_{\epsilon}(E_{n})\to0$ for $E_{n}\neq0,\pm\pi$ . Only signals from topological edge states would remain in the long-time dynamics, provided that the corresponding coefficient $\varPhi_{n}$ is non-vanishing, i.e., the initial state has a finite overlap with the left eigenvector of the relevant edge state.

Experimentally, we probe the projection of the wavefunction summations $\Phi_{\epsilon,\mu}(x)=|(\langle x|\otimes\langle\mu|)|\left|\Phi_{\epsilon}(t)\rangle\right|$ , such that the internal- and external-degrees of freedom of both types of topological edge states ( $\epsilon=0,\pi$ ) are fully resolved.

Experimental implementation of edge-state detection. We probe $\Phi_{\epsilon,\mu}(x)$ from our experimental reconstruction of $|\varphi(t)\rangle$ at each time step. Here $|\varphi(t)\rangle$ is the experimentally realized time-dependent state (with $M_{E}$ rather than M), which is related to $|\Phi(t)\rangle$ through $|\varphi(t)\rangle = e^{-\gamma t}|\Phi(t)\rangle$ . It follows that

$$
\Phi_ {\epsilon , \mu} (x) = \left| (\langle x | \otimes \langle \mu |) \sum_ {t ^ {\prime} = 0} ^ {t} \frac {e ^ {i \epsilon t ^ {\prime}}}{t + 1} e ^ {\gamma t ^ {\prime}} | \varphi (t ^ {\prime}) \rangle \right|. \tag {12}
$$

Since $U$ and initial states are purely real in the basis of $\{| \pm \rangle\}$ , we have the expansion

$$
| \varphi (t) \rangle = \sum_ {x} \left[ p _ {+} (t, x) | x \rangle \otimes | + \rangle + p _ {-} (t, x) | x \rangle \otimes | - \rangle \right], \tag {13}
$$

where the coefficients $p_{\mu}(t,x)$ are also real. Based on these, we perform four distinct measurements $M_{i}$ ( $i = 1, \cdots, 4$ ) to reconstruct $|\varphi(t)\rangle$ in the basis $\{|\pm\rangle\}$ . This amounts to measuring the absolute values, the relative signs and a global sign of the real coefficients $\{p_{\pm}(t,x)\}$ , as we detail in the following. All measurement modules are shown in Fig. 2.

First, we measure the absolute values $|p_{\pm}(t,x)|$ . After the t-th step, photons in the spatial mode x are sent to a detection unit $M_{1}$ , which consists of an HWP ( $H_{1}$ ) at 22.5°, a PBS and APDs. $M_{1}$ applies a projective measurement of the observable $\sigma_{x}$ on the polarization of photons. The counts of the horizontally polarized photons $N_{H}(t,x)$ and vertically polarized ones $N_{V}(t,x)$ are registered by the coincidences between one of the APDs in the detection unit, and the APD for the trigger photon. The measured probability distributions are $P_{H(V)}(t,x)=$

$N_{H(V)}(t,x)/\sum_{x}\left[N_{H}(t,x)+N_{V}(t,x)+\sum_{t'=0}^{t}N_{L}(t',x)\right]$ , with $N_{L}(t,x)$ the photon loss caused by the partial measurement $M_{E}$ . The square root of the probability distribution $P_{H(V)}(t,x)$ corresponds to $|p_{\pm}(t,x)|$ .

Second, we determine the relative sign between the amplitudes $p_{+}(t,x)$ and $p_{-}(t,x)$ via the detection unit $M_{2}$ . The only difference between $M_{2}$ and $M_{1}$ is that the setting angle of the HWP ( $H_{1}$ ) is set to 0, i.e., a projective measurement of the observable $\sigma_{z}$ on the polarization of photons. The difference between the probability distributions of the horizontally and vertically polarized photons is given by

$$
P _ {H} (t, x) - P _ {V} (t, x) = 2 p _ {+} (t, x) p _ {-} (t, x), \tag {14}
$$

which determines the relative sign between $p_{+}(t,x)$ and $p_{-}(t,x)$ .

Third, we probe the relative sign between $p_{\pm}(t,x)$ and $p_{\pm}(t,x-1)$ , which is necessary to calculate the summation of wave functions at each time step. To this end, we add a detection unit $M_{3}$ behind $M_{1}$ and $M_{2}$ (see Fig. 2), which consists of two HWPs at $22.5^{\circ}$ ( $H_{2}$ and $H_{3}$ ), a BD, a PBS and APDs. We use a BD to combine the horizontally polarized photons in the spatial mode x and the vertically polarized photons in the spatial mode x-1. After a projective measurement on the polarizations of photons via $H_{3}$ and the following PBS, the difference in probability distribution of the resulting photons between the two polarizations is

$$
P _ {H} (t, x) - P _ {V} (t, x) = p _ {+} (t, x) p _ {+} (t, x - 1) \tag {15}
$$

when $\mathrm{H}_1$ is set at $22.5^{\circ}$ in $M_{1(2)}$ , and

$$
P _ {H} (t, x) - P _ {V} (t, x) = p _ {-} (t, x) p _ {-} (t, x - 1) \tag {16}
$$

when the setting angle of $\mathrm{H}_{1}$ is at $-22.5^{\circ}$ .

Finally, we determine the global sign of $p_{\pm}(t,x)$ relative to the reference photons, which are reflected by the PBS used for the preparation of the initial coin state. For this step, we only need to determine the sign of $p_{\pm}(t,x_{w})$ for an arbitrary position $x_{w}$ at each time step. A natural choice of $x_{w}$ is the position where the walker and reference photons have comparable counts. Assuming the reference photons have an amplitude a (a > 0), we determine the relative sign between the amplitudes of the reference photons and the walker photons at $x_{w}$ after t steps. We remove the detection unit $M_{3}$ and keep the optical elements of $M_{1(2)}$ . Only photons reflected by the PBS in $M_{1(2)}$ are relevant here. These photons, after passing through an HWP ( $H_{4}$ at $45^{\circ}$ ), are combined with the reference photons at a PBS. A projective measurement is then applied on the polarization of the photons via $H_{5}$ at $22.5^{\circ}$ and the last PBS. The difference between the probabilities of the photons with different polarizations is

$$
P _ {H} (t, x _ {w}) - P _ {V} (t, x _ {w}) = 2 a p _ {-} (t, x _ {w}), \tag {17}
$$

which determines the global sign of $p_{-}(t,x_{w})$ as a is positive. Since we have determined the relative sign between $p_{+}(t,x)$ and $p_{-}(t,x)$ via $M_{2}$ and that between $p_{\pm}(t,x)$ and $p_{\pm}(t,x-1)$ via $M_{3}$ , the global sign of $p_{\pm}(t,x)$ is also determined for an arbitrary x.

With the above steps, we reconstruct $|\varphi(t)\rangle$ in the basis of $|\pm\rangle$ , which enables us to calculate $\Phi_{\epsilon,\mu}(x)$ according to Eq. (12).

\* wangzhongemail@tsinghua.edu.cn   
$^{\dagger}$ wyiz@ustc.edu.cn   
‡ gnep.eux@gmail.com   
[1] Hasan, M. Z. & Kane, C. L. Colloquium: topological insulators. Rev. Mod. Phys. 82, 3045-3067 (2010).   
[2] Qi, X. L. & Zhang, S. C. Topological insulators and superconductors. Rev. Mod. Phys. 83, 1057-1110 (2011).   
[3] Lee, T. E. Anomalous edge state in a non-Hermitian lattice. Phys. Rev. Lett. 116, 133903 (2016).   
[4] Yao, S. & Wang, Z. Edge states and topological invariants of non-Hermitian systems. Phys. Rev. Lett. 121, 086803 (2018).   
[5] Yao, S., Song, F. & Wang, Z. Non-Hermitian chern bands. Phys. Rev. Lett. 121, 136802 (2018).   
[6] Kunst, F. K., Edvardsson, E., Budich, J. C. & Bergholtz, E. J. Biorthogonal bulk-boundary correspondence in non-Hermitian systems. Phys. Rev. Lett 121, 026808 (2018).   
[7] Yokomizo, K. & Murakami, S. Bloch band theory for non-Hermitian systems. Preprint at https://arxiv.org/abs/1902.10958 (2019).   
[8] Alvarez, V. M., Vargas, J. B., Berdakin, M., & Torres, L. F. Topological states of non-Hermitian systems. Eur. Phys. J. Spec. Top. 227, 1295 (2018).   
[9] Lee, C. H., Thomale, R. Anatomy of skin modes and topology in non-Hermitian systems. Phys. Rev. B 99, 201103(R) (2019).   
[10] Poli, C., Bellec, M., Kuhl, U., Mortessagne, F. & Schomerus, H. Selective enhancement of topologically induced interface states in a dielectric resonator. Nat. Commun. 6, 6710 (2015).   
[11] Zeuner, J. M. et al. Observation of a topological transition in the bulk of a non-Hermitian system. Phys. Rev. Lett. 115, 040402 (2015).   
[12] Xiao, L. et al. Observation of topological edge states in parity-time-symmetric quantum walks. Nat. Phys. 13, 1117 (2017).   
[13] Zhan, X. et al. Detecting topological invariants in nonunitary discrete-time quantum walks. Phys. Rev. Lett. 119, 130501 (2017).   
[14] Wang, K. et al. Simulating dynamic quantum phase transitions in photonic quantum walks. Phys. Rev. Lett. 122, 020501 (2019).   
[15] Wang, K. et al. Observation of emergent momentum-time skyrmions in parity-time-symmetric non-unitary quench dynamics. Nat. Commun. 10, 2293 (2019).   
[16] Weimann, S. et al. Topologically protected bound states in photonic parity-time-symmetric crystals. Nat. Mater. 16, 433-438 (2017).   
[17] Parto M. et al. Edge-mode lasing in 1D topological active arrays. Phys. Rev. Lett. 120, 113901 (2018).

[18] Zhou, H. et al. Observation of bulk Fermi arc and polarization half charge from paired exceptional points. Science 359, 1009 (2018).   
[19] Ozawa, T. et al. Topological photonics. Rev. Mod. Phys. 91, 015006 (2019).   
[20] Bandres, M. A. et al. Topological insulator laser: Experiments. Science 359, eaar4005 (2018).   
[21] Zhu, W. et al. Simultaneous observation of a topological edge state and exceptional point in an open and non-Hermitian acoustic system. Phys. Rev. Lett. 121, 124501 (2018).   
[22] Wu Y. et al. Observation of parity-time symmetry breaking in a single-spin system. Science 364, 878 (2019).   
[23] Li J. et al. Observation of parity-time symmetry breaking transitions in a dissipative Floquet system of ultracold atoms. Nat. Commun. 10, 855 (2019).   
[24] Shen, H., Zhen, B., & Fu, L., Topological band theory for non-Hermitian Hamiltonians. Phys. Rev. Lett. 120, 146402 (2018).   
[25] Leykam, D., Bliokh, K. Y., Huang, C., Chong, Y. D., & Nori, F. Edge modes, degeneracies, and topological numbers in non-Hermitian systems. Phys. Rev. Lett. 118, 040401 (2017).   
[26] Gong, Z. et al. Topological phases of non-Hermitian systems. Phys. Rev. X 8, 031079 (2018).   
[27] El-Ganainy, R. et al. Non-Hermitian physics and PT symmetry. Nat. Phys. 14, 11 (2018).   
[28] Kawabata K., Shiozaki, K., Ueda, M. & Sato, M. Symmetry and topology in non-Hermitian physics. Preprint at https://arxiv.org/abs/1812.09133 (2018).   
[29] Zhou H. & Lee J. Y. Periodic table for topological bands with non-Hermitian symmetries. Phys. Rev. B 99, 235112 (2019)   
[30] Rudner, M. S., & Levitov, L. S. Topological transition in a non-Hermitian quantum walk. Phys. Rev. Lett. 102, 065703 (2009).   
[31] Esaki, K., Sato, M., Hasebe, K., & Kohmoto, M. Edge states and topological phases in non-Hermitian systems. Phys. Rev. B 84, 205128 (2011).   
[32] Zhu, B., Lü, R., & Chen, S. PT symmetry in the non-Hermitian Su-Schrieffer-Heeger model with complex boundary potentials. Phys. Rev. A 89, 062102 (2014).   
[33] S. Lieu, Topological phases in the non-Hermitian Su-Schrieffer-Heeger model. Phys. Rev. B 97, 045106 (2018).   
[34] Rudner, M. S., Lindner, N. H., Berg, E., & Levin, M. Anomalous edge states and the bulk-edge correspondence for periodically driven two-dimensional systems. Phys. Rev. X 3, 031005 (2018).   
[35] Asbóth, J. K. & Obuse, H. Bulk-boundary correspondence for chiral symmetric quantum walks. Phys. Rev. B 88, 121406(R) (2013).   
[36] Deng, T. & Yi, W. Non-Bloch topological invariants in a non-Hermitian domain-wall system. Phys. Rev. B 100, 035102 (2019).   
[37] Yao, S., Yan, Z. & Wang, Z. Topological invariants of Floquet systems: General formulation, special properties, and Floquet topological defects. Phys. Rev. B 96, 195303 (2017).   
[38] Fruchart, M. Complex classes of periodically driven topological lattice systems. Phys. Rev. B 93, 115429 (2016).   
[39] Brody, D. C. Biorthogonal quantum mechanics. J. Phys. A: Math. Theor. 47 035305 (2014).

Acknowledgements This work has been supported

by the Natural Science Foundation of China (Grant No. 11674056, No. 11674189) and the Natural Science Foundation of Jiangsu Province (Grant No. BK20160024). WY acknowledges support from the National Key Research and Development Program of China (Grant Nos. 2016YFA0301700 and 2017YFA0304100).

# SUPPLEMENTAL INFORMATION FOR “OBSERVATION OF NON-HERMITIAN BULK-BOUNDARY CORRESPONDENCE IN QUANTUM DYNAMICS”

In this Supplemental Information, we provide details on calculations of generalized Brillouin zone, non-Bloch topological invariants, and additional supporting experimental data.

# Bulk-state wave functions, generalized Brillouin zones, and non-Bloch topological invariants

We consider quantum-walk dynamics governed by the Floquet operator U = FMG. Here

$$
F = R [ \frac {\theta_ {1} (x)}{2} ] S _ {2} R [ \frac {\theta_ {2} (x)}{2} ],
$$

$$
G = R \left[ \frac {\theta_ {2} (x)}{2} \right] S _ {1} R \left[ \frac {\theta_ {1} (x)}{2} \right], \tag {S1}
$$

where the coin-rotation operator R and the shift operator S are given by

$$
R (\theta) = \mathbb {1} _ {w} \otimes e ^ {- i \theta \sigma_ {y}},
$$

$$
S _ {1} = \sum_ {x} | x \rangle \langle x | \otimes | 0 \rangle \langle 0 | + | x + 1 \rangle \langle x | \otimes | 1 \rangle \langle 1 |,
$$

$$
S _ {2} = \sum_ {x} | x - 1 \rangle \langle x | \otimes | 0 \rangle \langle 0 | + | x \rangle \langle x | \otimes | 1 \rangle \langle 1 |. \tag {S2}
$$

Here $1_{w} = \sum_{x} |x\rangle\langle x|$ , and $-N \leqslant x \leqslant N - 1$ is the site index of the lattice. For a domain-wall configuration on a circle with 2N lattice sites (see Fig. 1 of the main text), we adopt a cyclic index such that $|x - 1\rangle|_{x=-N} = |N - 1\rangle$ and $|x + 1\rangle|_{x=N-1} = |-N\rangle$ . We also have

$$
\left\{ \begin{array}{l l} \theta_ {1 (2)} (x) = \theta_ {1 (2)} ^ {L} & x \in J _ {L} \\ \theta_ {1 (2)} (x) = \theta_ {1 (2)} ^ {R} & x \in J _ {R} \end{array} , \right. \tag {S3}
$$

where $\theta_{1(2)}^{\alpha}$ represent coin parameters of the left $(\alpha = L)$ and right $(\alpha = R)$ bulk, $J_{L} = \{x\in \mathbb{Z}| - N\leqslant x\leqslant -1\}$ and $J_{R} = \{x\in \mathbb{Z}|0\leqslant x\leqslant N - 1\}$ .

We then rewrite $U$ as

$$
U = \sum_ {x} \left[ | x \rangle \langle x + 1 | \otimes A _ {m} (x) + | x \rangle \langle x - 1 | \otimes A _ {p} (x) + | x \rangle \langle x | \otimes A _ {s} (x) \right], \tag {S4}
$$

where the site-dependent coin-state operators $A_{m,p,s}(x)$ are given by

$$
A _ {m} (x) = F _ {m} (x + 1) M G _ {s} (x + 1),
$$

$$
A _ {p} (x) = F _ {s} (x) M G _ {p} (x - 1), \tag {S5}
$$

$$
A _ {s} (x) = F _ {s} (x) G _ {s} (x) + F _ {m} (x + 1) G _ {p} (x),
$$

with

$$
F _ {m} (x) = R [ \frac {\theta_ {1} (x - 1)}{2} ] P _ {0} R [ \frac {\theta_ {2} (x)}{2} ],
$$

$$
F _ {s} (x) = R [ \frac {\theta_ {1} (x)}{2} ] P _ {1} R [ \frac {\theta_ {2} (x)}{2} ],
$$

$$
G _ {s} (x) = R [ \frac {\theta_ {2} (x)}{2} ] P _ {0} R [ \frac {\theta_ {1} (x)}{2} ],
$$

$$
G _ {p} (x) = R \left[ \frac {\theta_ {2} (x + 1)}{2} \right] P _ {1} R \left[ \frac {\theta_ {1} (x)}{2} \right], \tag {S6}
$$

and $P_{0}=|0\rangle\langle0|$ , $P_{1}=|1\rangle\langle1|$ .

Following Refs. [4, 36], we write the general eigenstate of $U$ as $|\psi \rangle = |\psi^R\rangle +|\psi^L\rangle$ , with

$$
\left| \psi^ {\alpha} \right\rangle = \sum_ {x \in J _ {\alpha}, j} \beta_ {\alpha , j} ^ {x} | x \rangle \otimes \left| \phi_ {j} ^ {\alpha} \right\rangle_ {c} (\alpha = L, R), \tag {S7}
$$

where $\left|\phi^{\alpha}\right\rangle_{c}$ is the coin state of the corresponding bulk and $\beta_{\alpha}$ is the spatial-mode function.

From the eigenstate equation

$$
U | \psi \rangle = \lambda | \psi \rangle , \tag {S8}
$$

we have

$$
\left(A _ {m} ^ {\alpha} \beta_ {\alpha} + \frac {A _ {p} ^ {\alpha}}{\beta_ {\alpha}} + A _ {s} ^ {\alpha} - \lambda\right) | \phi^ {\alpha} \rangle_ {c} = 0, \tag {S9}
$$

where $A_{m,p,s}^{\alpha}$ are the corresponding coin operators in the bulk, with $A_{m,p,s}^{L} = A_{m,p,s}(x)$ ( $-N + 1 \leqslant x \leqslant -2$ ) and $A_{m,p,s}^{R} = A_{m,p,s}(x)$ ( $2 \leqslant x \leqslant N - 2$ ).

Equation (S9) supports non-trivial solutions for

$$
\det \left[ A _ {m} ^ {\alpha} \beta_ {\alpha} + A _ {p} ^ {\alpha} \frac {1}{\beta_ {\alpha}} + A _ {s} ^ {\alpha} - \lambda \right] = 0. \tag {S10}
$$

Since Eq. (S10) is quadratic in $\beta_{\alpha}$ , there exist two solutions $\beta_{\alpha,j}$ with $j = 1,2$ . Correspondingly, eigenstates of the bulk can be written as

$$
\left| \psi^ {\alpha} \right\rangle = \sum_ {x \in J _ {\alpha}, j = 1, 2} \beta_ {\alpha , j} ^ {x} | x \rangle \otimes \left| \phi_ {j} ^ {\alpha} \right\rangle_ {c}. \tag {S11}
$$

The domain-wall boundary condition is enforced by substituting Eq. (S11) into $U|\psi\rangle = \lambda|\psi\rangle$ at the boundaries $(x = -N, -1, 0, N-1)$ . Making use of Eq. (S9), we derive a set of linear equations $M\left[\left|\phi_{1}^{L}\right\rangle_{c}, \left|\phi_{2}^{L}\right\rangle_{c}, \left|\phi_{1}^{R}\right\rangle_{c}, \left|\phi_{2}^{R}\right\rangle_{c}\right]^{T} = 0$ , where

$$
M =
$$

$$
\left( \begin{array}{c c c c} - A _ {p} ^ {L} \beta_ {L, 1} ^ {- N - 1} & - A _ {p} ^ {L} \beta_ {L, 2} ^ {- N - 1} & A _ {p} (- N) \beta_ {R, 1} ^ {N - 1} & A _ {p} (- N) \beta_ {R, 2} ^ {N - 1} \\ A _ {p} ^ {L} \beta_ {L, 1} ^ {- 2} + [ A _ {s} (- 1) - \beta ] \beta_ {L, 1} ^ {- 1} & A _ {p} ^ {L} \beta_ {L, 2} ^ {- 2} + [ A _ {s} (- 1) - \lambda ] \beta_ {L, 2} ^ {- 1} & A _ {m} (- 1) & A _ {m} (- 1) \\ A _ {p} (0) \beta_ {L, 1} ^ {- 1} & A _ {p} (0) \beta_ {L, 2} ^ {- 1} & - A _ {p} ^ {R} \beta_ {R, 1} ^ {- 1} & - A _ {p} ^ {R} \beta_ {R, 2} ^ {- 1} \\ A _ {m} (N - 1) \beta_ {L, 1} ^ {- N} & A _ {m} (N - 1) \beta_ {L, 2} ^ {- N} & A _ {p} ^ {R} \beta_ {R, 1} ^ {N - 2} + [ A _ {s} (N - 1) - \lambda ] \beta_ {R, 1} ^ {N - 1} & A _ {p} ^ {R} \beta_ {R, 2} ^ {N - 2} + [ A _ {s} (N - 1) - \lambda ] \beta_ {R, 2} ^ {N - 1} \end{array} \right) \tag {S12}
$$

Non-trivial solutions exist only when the 8-by-8 coefficient matrix $M$ satisfies $\det(M) = 0$ in the thermodynamic limit $N \to \infty$ . Making use of Eq. (S9), the condition $\det(M) = 0$ is simplified to

$$
a _ {1} \frac {1}{\beta_ {L , 1} ^ {N}} \beta_ {R, 1} ^ {N} + a _ {2} \frac {1}{\beta_ {L , 2} ^ {N}} \beta_ {R, 1} ^ {N} + a _ {3} \frac {1}{\beta_ {L , 1} ^ {N}} \beta_ {R, 2} ^ {N} + a _ {4} \frac {1}{\beta_ {L , 2} ^ {N}} \beta_ {R, 2} ^ {N} + b _ {L} \frac {1}{\beta_ {L , 1} ^ {N}} \frac {1}{\beta_ {L , 2} ^ {N}} + b _ {R} \beta_ {R, 1} ^ {N} \beta_ {R, 2} ^ {N} = 0, \tag {S13}
$$

where $\{a_1, a_2, a_3, a_4, b_L, b_R\}$ are some coefficients whose exact forms are not important for the following discussion.

To proceed further, we need to sort the following terms

$$
\left\{\left| \frac {\beta_ {R , 1}}{\beta_ {L , 1}} \right|, \left| \frac {\beta_ {R , 1}}{\beta_ {L , 2}} \right|, \left| \frac {\beta_ {R , 2}}{\beta_ {L , 1}} \right|, \left| \frac {\beta_ {R , 2}}{\beta_ {L , 2}} \right|, \left| \frac {1}{\beta_ {L , 1} \beta_ {L , 2}} \right|, \left| \beta_ {R, 1} \beta_ {R, 2} \right| \right\}. \tag {S14}
$$

This is because in the thermodynamic limit $(N \to \infty)$ , only terms with the largest absolute values survive in Eq. (S13). Without loss of generality, we take $|\beta_{\alpha,1}| \geqslant |\beta_{\alpha,2}|$ and discuss the order of these terms case by case. For example, when $|\beta_{L,2}\beta_{R,2}| \leqslant 1$ and $|\beta_{L,2}\beta_{R,1}| \leqslant 1$ and $|\beta_{L,1}\beta_{R,2}| \leqslant 1$ , the largest two terms are $|\frac{1}{\beta_{L,1}\beta_{L,2}}|$ and $|\frac{\beta_{R,1}}{\beta_{L,2}}|$ . Eq. (S13) is then reduced to

$$
a _ {2} \frac {1}{\beta_ {L , 2} ^ {N}} \beta_ {R, 1} ^ {N} + b _ {L} \frac {1}{\beta_ {L , 1} ^ {N}} \frac {1}{\beta_ {L , 2} ^ {N}} = 0. \tag {S15}
$$

![](images/388234f18dde7fcd47cb5a6f5a6315aa972699c320c636505174a8ab42deb92f.jpg)

<details>
<summary>scatter</summary>

| Re(β) | Im(β) | Series     |
|-------|-------|------------|
| -1.5  | 0.0   | β_L1       |
| -1.0  | 0.5   | β_L1       |
| -0.5  | 1.0   | β_L1       |
| 0.0   | 1.5   | β_L1       |
| 0.5   | 1.0   | β_L1       |
| 1.0   | 0.5   | β_L1       |
| 1.5   | 0.0   | β_L1       |
| -1.5  | -0.5  | β_L2       |
| -1.0  | -1.0  | β_L2       |
| -0.5  | -1.5  | β_L2       |
| 0.0   | -1.5  | β_L2       |
| 0.5   | -1.0  | β_L2       |
| 1.0   | -0.5  | β_L2       |
| 1.5   | 0.0   | β_L2       |
| -1.5  | -0.5  | β_R1       |
| -1.0  | -1.0  | β_R1       |
| -0.5  | -1.5  | β_R1       |
| 0.0   | -1.5  | β_R1       |
| 0.5   | -1.0  | β_R1       |
| 1.0   | -0.5  | β_R1       |
| 1.5   | 0.0   | β_R1       |
| -1.5  | -0.5  | β_R2       |
| -1.0  | -1.0  | β_R2       |
| -0.5  | -1.5  | β_R2       |
| 0.0   | -1.5  | β_R2       |
| 0.5   | -1.0  | β_R2       |
| 1.0   | -0.5  | β_R2       |
| 1.5   | 0.0   | β_R2       |
| -1.5  | -0.5  | unit circle|
| -1.0  | -1.0  | unit circle|
| -0.5  | -1.5  | unit circle|
| 0.0   | -1.5  | unit circle|
| 0.5   | -1.0  | unit circle|
| 1.0   | -0.5  | unit circle|
| 1.5   | 0.0   | unit circle|
</details>

FIG. S1. Numerical check of the $\zeta$ function. Generalized Brillouin zones calculated from Eq. (S16) (solid and dashed lines), compared to those obtained from the numerically calculated eigenenergy spectrum (circles and asterisks). In the latter approach, we numerically calculate the $\lambda$ eigenspectrum by diagonalizing U of a finite size system, and then obtain $\beta_{\alpha,j}$ from Eq. (S10). The two approaches lead to consistent results, confirming the validity of Eq. (S16) as the equation of generalized Brillouin zone. In this figure, we take the same parameters as those in Fig. 4b of the main text.

It follows that, in the thermodynamic limit, $|\beta_{L,1}\beta_{R,1}| = 1$ .

Exhausting all the possible scenarios, we rewrite Eq. (S13) as

$$
\zeta (\beta_ {\alpha , j}) = 0, \tag {S16}
$$

where

$$
\zeta \left(\beta_ {\alpha , j}\right) := \left\{ \begin{array}{l l} \left| \beta_ {L, 1} \beta_ {R, 1} \right| - 1, & \left| \beta_ {L, 2} \beta_ {R, 2} \right| \leqslant 1 \text { and } \left| \beta_ {L, 2} \beta_ {R, 1} \right| \leqslant 1 \text { and } \left| \beta_ {L, 1} \beta_ {R, 2} \right| \leqslant 1, \\ \left| \beta_ {L, 2} \beta_ {R, 2} \right| - 1, & \left| \beta_ {L, 1} \beta_ {R, 1} \right| \geqslant 1 \text { and } \left| \beta_ {L, 1} \beta_ {R, 2} \right| \geqslant 1 \text { and } \left| \beta_ {L, 2} \beta_ {R, 1} \right| \geqslant 1, \\ \left| \beta_ {L, 1} \right| - \left| \beta_ {L, 2} \right|, & \left| \beta_ {L, 2} \beta_ {R, 1} \right| \geqslant 1 \text { and } \left| \beta_ {L, 1} \beta_ {R, 2} \right| \leqslant 1, \\ \left| \beta_ {R, 1} \right| - \left| \beta_ {R, 2} \right|, & \left| \beta_ {L, 1} \beta_ {R, 2} \right| \geqslant 1 \text { and } \left| \beta_ {L, 2} \beta_ {R, 1} \right| \leqslant 1. \end{array} \right. \tag {S17}
$$

Note that $\zeta(\beta_{\alpha,j})$ is a function of $\lambda$ since $\beta_{\alpha,j}$ are functions of $\lambda$ through Eq. (S10). Therefore, we can solve Eq. (S16) as an equation of $\lambda$ , and find the $\lambda$ eigenspectrum. From the $\lambda$ eigenspectrum we can obtain $\beta_{\alpha,j}$ by Eq. (S10). The $\beta_{\alpha,j}$ trajectories are the generalized Brillouin zones, which play a key role in the non-Hermitian bulk-boundary correspondence. The obtained generalized Brillouin zones are shown in Fig. S1. To double check the validity of Eq. (S16) as the equation of generalized Brillouin zone, we also numerically diagonalize U for finite-size systems, and then obtain the corresponding $\beta_{\alpha,j}$ by Eq. (S10). The generalized Brillouin zones obtained in this way are consistent with those obtained from Eq. (S16) [see Fig. S1].

Finally, we note that Eq. (S17) is helpful in identifying generalized Brillouin zones of the two bulks from numerically calculated eigenspectrum. Specifically, eigenstates of U belonging to the first and second cases are associated with the generalized Brillouin zones of both bulks; whereas those belonging to the third (fourth) case are associated only with the generalized Brillouin zone of the left (right) bulk.

Based on the generalized Brillouin zones, we then calculate the non-Bloch topological invariants $\tilde{\nu}_{\epsilon}\left(\epsilon=0,\pi\right)$ defined in the main text. The results precisely match the topological edge modes with $\epsilon=0,\pi$ respectively, which embodies the non-Hermitian bulk-boundary correspondence.

# An alternative approach to calculate topological invariants: Non-Bloch winding numbers of different time frames

In this section, we provide an alternative approach to calculate the non-Bloch topological invariants. The idea is to calculate the non-Bloch winding numbers in different time frames of the Floquet sequence. In the Hermitian case,

![](images/936f5c4f91d33edc1edc95527a9d6897f21a521e6cbf2758412015aa877344ca.jpg)

FIG. S2. Calculation of the non-Bloch topological invariants. a The absolute values of the quasienergy spectrum as a function of $\theta_{1}^{R}$ . The parameters are the same as those in Fig. 4a of the main text, where the quasienergy spectrum is completely real. b1, b2 Non-Bloch topological invariants of the left (red solid line) and right bulks (purple solid line) using periodized Floquet operators, as outlined in the main text. We also show non-Bloch topological invariants calculated using the two different time frames, for the left (yellow dashed line) and right bulks (green dashed line).   
![](images/e6f73ef741029a002c325187c988ad2e7ab6788afdcd750f469a23ba616c34c4.jpg)  
b

![](images/b3957bbb460395d9c45be9ea68423e7c11b9df02bd977eefd3833e54f771158b.jpg)

<details>
<summary>bar</summary>

| Position | ε=0, μ=+ | ε=0, μ=- | ε=π, μ=+ | ε=π, μ=- |
| -------- | -------- | -------- | -------- | -------- |
| -1       | 0.0      | 0.0      | 0.0      | 0.0      |
| 1        | 0.3      | 0.0      | 0.0      | 0.0      |
| 3        | 0.1      | 0.0      | 0.0      | 0.0      |
| 5        | 0.0      | 0.0      | 0.0      | 0.0      |
| 7        | 0.0      | 0.0      | 0.0      | 0.0      |
</details>

C

![](images/2ae2f633e952bce9d7f0323608b358fe0a973687cffe14fd27e636e6f8f1b750.jpg)

<details>
<summary>bar_line</summary>

| Position | Simulation | Experiment | Diagonalization |
| -------- | ---------- | ---------- | ---------------- |
| -8       | 0.0        | 0.0        | 0.0              |
| -6       | 0.0        | 0.0        | 0.0              |
| -4       | 0.0        | 0.0        | 0.0              |
| -2       | 0.0        | 0.0        | 0.0              |
| 0        | 0.38       | 0.32       | 0.39             |
| 2        | 0.20       | 0.17       | 0.15             |
| 4        | 0.12       | 0.08       | 0.07             |
| 6        | 0.06       | 0.04       | 0.03             |
| 8        | 0.01       | 0.01       | 0.01             |
</details>

d   
![](images/a3480fe62b26147f66de38c9ddcf8365bf82dc87da8ecc4b74de368aacd2bce1.jpg)

<details>
<summary>bar</summary>

| Position | Φεμ(x) |
| -------- | ------ |
| -1       | 0.0    |
| 1        | 0.05   |
| 3        | 0.08   |
| 5        | 0.12   |
| 7        | 0.05   |
</details>

FIG. S3. Non-Hermitian bulk-boundary correspondence in the alternative time frame given by $U'$ . a Quasi-energy spectrum (black), and winding-number differences for the zero- (red) and $\pi$ -modes (blue) between the two bulks with the parameters: $\theta_{1}^{L}=0.5625\pi$ , $\theta_{2}^{R}=0$ , $\theta_{2}^{L}=\pi$ , and $\gamma=0.2746$ . The purple dot with $\theta_{1}^{R}=-0.0667\pi$ corresponds to the parameter used in b, where the system possesses both zero- and $\pi$ -mode edge states. The black dot with $\theta_{1}^{R}=0.0667\pi$ corresponds to the parameter used in d, where there is no edge state. b Experimentally measured $\Phi_{\epsilon,\mu}(x)$ after the seventh step with the initial state $|0\rangle\otimes|+\rangle$ . c Comparison between experimentally-measured and numerically-calculated $\Phi_{0,+}(x)$ , as well as the scaled norms of the corresponding edge state after the seventh step. Topological edge states are numerically calculated for a domain-wall system with N=15, whose norms are scaled to fit the central peak of the corresponding $\Phi_{0,+}(x)$ . d Experimentally measured $\Phi_{\epsilon,\mu}(x)$ after the seventh step with the same initial state as b.

this approach has been adopted in Ref. [35] to calculate the Bloch topological invariants. The obtained topological invariants $\tilde{\nu}_{0}$ and $\tilde{\nu}_{\pi}$ below are the same as those calculated from the periodized Floquet operators $\overline{U}_{\epsilon}$ .

First, we demonstrate how to calculate winding numbers in the time frame defined by U. The Fourier component of U in the two bulks can be written as

$$
U ^ {\alpha} (k) = d _ {0} ^ {\alpha} \sigma_ {0} - i d _ {1} ^ {\alpha} \sigma_ {x} - i d _ {2} ^ {\alpha} \sigma_ {y} - i d _ {3} ^ {\alpha} \sigma_ {z}, \tag {S18}
$$

where

$$
d _ {0} ^ {\alpha} = - \cosh \gamma \sin \theta_ {1} ^ {\alpha} \sin \theta_ {2} ^ {\alpha} + \cosh \gamma \cos k \cos \theta_ {1} ^ {\alpha} \cos \theta_ {2} ^ {\alpha} + i \sinh \gamma \cos \theta_ {1} ^ {\alpha} \sin k,
$$

$$
d _ {1} = 0,
$$

$$
d _ {2} ^ {\alpha} = \cosh \gamma \cos \theta_ {1} ^ {\alpha} \sin \theta_ {2} ^ {\alpha} + \cos k \cosh \gamma \cos \theta_ {2} ^ {\alpha} \sin \theta_ {1} ^ {\alpha} + i \sin k \sinh \gamma \sin \theta_ {1} ^ {\alpha},
$$

$$
d _ {3} ^ {\alpha} = - \sin k \cosh \gamma \cos \theta_ {2} ^ {\alpha} + i \cos k \sinh \gamma . \tag {S19}
$$

a   
![](images/1495cda017301dea994eff6fc2fc9e5a058fc73a89ed0f32f4575340c55a8717.jpg)

b   
![](images/9dfdf543b203184a0bfa9498491f84c738dadc01c04a999b57c857a633c5494c.jpg)

<details>
<summary>bar</summary>

| Position | ε=0, μ=+ | ε=0, μ=- | ε=π, μ=+ | ε=π, μ=- |
| -------- | -------- | -------- | -------- | -------- |
| -3       | 0.0      | 0.0      | 0.0      | 0.0      |
| -1       | 0.0      | 0.0      | 0.0      | 0.0      |
| 1        | 0.0      | 0.48     | 0.0      | 0.0      |
| 3        | 0.0      | 0.27     | 0.0      | 0.0      |
| 5        | 0.0      | 0.0      | 0.0      | 0.0      |
</details>

C   
![](images/d6bef5855f0b3dca1ed3adcb02bb981a3e64e64c2ccbbca09cc1907e47838b1d.jpg)

d   
![](images/4a9a1f961f50447ef4640f061c7552145b808e13e5d4b974ab47284a86039053.jpg)

<details>
<summary>bar_line</summary>

| Position | Simulation | Experiment | Diagonalization |
| -------- | ---------- | ---------- | ---------------- |
| -6       | 0.45       | 0.0        | 0.0              |
| -4       | 0.0        | 0.02       | 0.0              |
| -2       | 0.1        | 0.1        | 0.1              |
| 0        | 0.3        | 0.45       | 0.45             |
| 2        | 0.15       | 0.1        | 0.1              |
| 4        | 0.05       | 0.02       | 0.02             |
| 6        | 0.01       | 0.01       | 0.01             |
| 8        | 0.0        | 0.0        | 0.0              |
</details>

FIG. S4. Non-Bloch bulk-boundary correspondence for edge states with $\epsilon=0$ . a Quasi-energy spectrum (black), and winding-number differences $\Delta\tilde{\nu}_{0}$ (red) and $\Delta\tilde{\nu}_{\pi}$ (blue) between the two bulks with the parameters: $\theta_{1}^{L}=-0.5625\pi$ , $\theta_{2}^{R}=0.25\pi$ , $\theta_{2}^{L}=0.75\pi$ , and $\gamma=0.2746$ . Color and line shapes for winding numbers are the same as in Fig. 4 of the main text. The red dot with $\theta_{1}^{R}=-0.18\pi$ corresponds to the coin parameter used in b, d. b Experimentally measured $\Phi_{\epsilon,\mu}(x)$ after the seventh step with the initial state $|0\rangle\otimes|-\rangle$ . c Generalized Brillouin zones on the complex plane. d Comparison between experimentally-measured and numerically-calculated $\Phi_{0,-}(x)$ , as well as the scaled norms of the topological edge state after the seventh step.

Here $\sigma_{x,y,z}$ are the standard Pauli matrices and $\sigma_{0}$ is the 2-by-2 identity matrix. Near k = 0, $d_{3}^{\alpha}$ resembles the $\sin k + i\gamma/2$ term appearing in the non-Hermitian Su-Schrieffer-Heeger model with non-Hermitian skin effect [4], which provides an intuitive understanding for the skin effect of $U^{\alpha}$ .

To calculate the Bloch winding numbers, we follow the standard practice and apply a unitary transformation $V = \exp(i\frac{\pi}{4}\sigma_{y})$ to $U^{\alpha}(k)$ such that

$$
W ^ {\alpha} (k) = V U ^ {\alpha} (k) V ^ {\dagger} = d _ {0} ^ {\alpha} I - i (- d _ {3} ^ {\alpha}) \sigma_ {x} - i d _ {2} ^ {\alpha} \sigma_ {y} - i d _ {1} ^ {\alpha} \sigma_ {z}. \tag {S20}
$$

The Bloch winding number is then defined through the generalized Zak phase

$$
\nu^ {\alpha} = \frac {\phi_ {\mathrm{Zak}} ^ {\alpha}}{\pi}, \tag {S21}
$$

$$
\phi_ {\mathrm{Zak}} ^ {\alpha} = - \int_ {- \pi} ^ {\pi} d k \frac {\left\langle \chi_ {k} ^ {\alpha} \right| i \partial_ {k} \left| \psi_ {k} ^ {\alpha} \right\rangle}{\left\langle \chi_ {k} ^ {\alpha} \right| \psi_ {k} ^ {\alpha} \rangle}, \tag {S22}
$$

where $|\psi_k^\alpha \rangle$ and $|\chi_k^\alpha \rangle$ are the right and left eigenstates of $W^{\alpha}(k)$ with

$$
W ^ {\alpha} (k) | \psi_ {k} ^ {\alpha} \rangle = E _ {k} | \psi_ {k} ^ {\alpha} \rangle , \tag {S23}
$$

$$
W ^ {\alpha \dagger} (k) | \chi_ {k} ^ {\alpha} \rangle = E _ {k} ^ {*} | \chi_ {k} ^ {\alpha} \rangle , \tag {S24}
$$

$$
E _ {k} ^ {\alpha} = d _ {0} ^ {\alpha} - i \sqrt {(d _ {1} ^ {\alpha}) ^ {2} + (d _ {2} ^ {\alpha}) ^ {2} + (d _ {3} ^ {\alpha}) ^ {2}}. \tag {S25}
$$

From the above equations, we have

$$
\nu^ {\alpha} = \frac {1}{2 \pi} \int d k \frac {- d _ {3} ^ {\alpha} \frac {\partial d _ {2} ^ {\alpha}}{\partial k} + d _ {2} ^ {\alpha} \frac {\partial d _ {3} ^ {\alpha}}{\partial k}}{(d _ {3} ^ {\alpha}) ^ {2} + (d _ {2} ^ {\alpha}) ^ {2}}. \tag {S26}
$$

In contrast, under the non-Hermitian skin effect, bulk states become localized, therefore we need the non-Bloch winding numbers calculated along the generalized Brillouin zones. From the spatial-mode function $\beta_{\alpha,j}$ , we define

$$
\beta_ {\alpha , j} = | \beta_ {\alpha , j} (p _ {j} ^ {\alpha}) | e ^ {i p _ {j} ^ {\alpha}}, \tag {S27}
$$

where $p_j^\alpha$ can be identified as the modified quasi-momentum in the $j$ -th generalized Brillouin zone of the corresponding bulk.

In practice, it is sufficient to replace $e^{ik}$ with $\beta_{\alpha,j}$ in Eq. (S26), such that

$$
\tilde {\nu} ^ {\alpha} = \frac {1}{2 \pi} \oint d p _ {j} ^ {\alpha} \frac {- \tilde {d} _ {3 , j} ^ {\alpha} \frac {\partial \tilde {d} _ {2 , j} ^ {\alpha}}{\partial p _ {j} ^ {\alpha}} + \tilde {d} _ {2 , j} ^ {\alpha} \frac {\partial \tilde {d} _ {3 , j} ^ {\alpha}}{\partial p _ {j} ^ {\alpha}}}{(\tilde {d} _ {3 , j} ^ {\alpha}) ^ {2} + (\tilde {d} _ {2 , j} ^ {\alpha}) ^ {2}}, \tag {S28}
$$

where

$$
\tilde {d} _ {2, j} ^ {\alpha} = \cosh \gamma \cos \theta_ {1} ^ {\alpha} \sin \theta_ {2} ^ {\alpha} + \cos \left(p _ {j} ^ {\alpha} - i \ln | \beta_ {\alpha , j} (p _ {j} ^ {\alpha}) |\right) \cosh \gamma \cos \theta_ {2} ^ {\alpha} \sin \theta_ {1} ^ {\alpha} + i \sin \left(p _ {j} ^ {\alpha} - i \ln | \beta_ {\alpha , j} (p _ {j} ^ {\alpha}) |\right) \sinh \gamma \sin \theta_ {1} ^ {\alpha} \tag {S29}
$$

$$
\tilde {d} _ {3, j} ^ {\alpha} = - \sin \left(p _ {j} ^ {\alpha} - i \ln | \beta_ {\alpha , j} (p _ {j} ^ {\alpha}) |\right) \cosh \gamma \cos \theta_ {2} ^ {\alpha} + i \cos \left(p _ {j} ^ {\alpha} - i \ln | \beta_ {\alpha , j} (p _ {j} ^ {\alpha}) |\right) \sinh \gamma \tag {S30}
$$

The integration in Eq. (S28) is over the $j$ -th generalized Brillouin zone. However, we have numerically checked that $\tilde{\nu}^{\alpha}$ calculated along different generalized Brillouin zones of a given bulk are the same. We therefore drop the index $j$ on the left-hand side of Eq. (S28).

Following the procedure above, both Bloch and non-Bloch winding numbers in an alternative time frame can be calculated with the Floquet operator

$$
U ^ {\prime} = M ^ {\frac {1}{2}} G F M ^ {\frac {1}{2}}. \tag {S31}
$$

Denoting the non-Bloch winding numbers of the two bulks as $\tilde{\nu}^{\prime \alpha}$ , we calculate non-Bloch topological invariants $\tilde{\nu}_0^\alpha$ and $\tilde{\nu}_{\pi}^{\alpha}$ through

$$
\tilde {\nu} _ {0 (\pi)} ^ {\alpha} = \frac {\tilde {\nu} ^ {\alpha} \pm \tilde {\nu} ^ {\prime \alpha}}{2}. \tag {S32}
$$

These non-Bloch topological invariants are the same as those calculated using the periodized Floquet operators with branch cuts, and correctly predict the existence and number of topological edge states through the non-Hermitian bulk-boundary correspondence. In Fig. S2, we show a typical comparison between non-Bloch topological invariants calculated using the two methods. Furthermore, we have experimentally confirmed that adopting periodized Floquet operators associated with $U'$ would give the correct non-Bloch bulk-boundary correspondence in the alternative time frame. This is shown in Fig. S3.

# Non-Bloch bulk correspondence for edge states with $\epsilon = 0$

In the main text, we confirm non-Bloch bulk-boundary correspondence for the following three cases: i) only edge states with $\epsilon = \pi$ exist; ii) both types of edge states with $\epsilon = 0$ and $\epsilon = \pi$ exist; iii) no edge state exists. For completeness, we also perform experiments using parameters under which only edge states with $\epsilon = 0$ exist. This is shown in Fig. S4.