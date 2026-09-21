# Observation of non-Hermitian edge burst in quantum dynamics

Lei Xiao, $^{1,*}$ Wen-Tan Xue, $^{2,*}$ Fei Song, $^{2}$ Yu-Min Hu, $^{2}$ Wei Yi, $^{3,4,\dagger}$ Zhong Wang, $^{2,\ddagger}$ and Peng Xue $^{1,\S}$

$^{1}$ Beijing Computational Science Research Center, Beijing 100084, China

$^{2}$ Institute for Advanced Study, Tsinghua University, Beijing, 100084, China

$^{3}$ CAS Key Laboratory of Quantum Information, University of Science and Technology of China, Hefei 230026, China

$^{4}$ CAS Center For Excellence in Quantum Information and Quantum Physics, Hefei 230026, China

The non-Hermitian skin effect, by which the eigenstates of Hamiltonian are predominantly localized at the boundary, has revealed a strong sensitivity of non-Hermitian systems to the boundary condition. Here we experimentally observe a striking boundary-induced dynamical phenomenon known as the non-Hermitian edge burst, which is characterized by a sharp boundary accumulation of loss in non-Hermitian time evolutions. In contrast to the eigenstate localization, the edge burst represents a generic non-Hermitian dynamical phenomenon that occurs in real time. Our experiment, based on photonic quantum walks, not only confirms the prediction of the phenomenon, but also unveils its complete space-time dynamics. Our observation of edge burst paves the way for studying the rich real-time dynamics in non-Hermitian topological systems.

Non-Hermitian physics has attracted increasing attention in a vast variety of contexts ranging from classical waves to open quantum systems $[1, 2]$ . Intriguingly, the spatial boundary plays a much more dramatic role in non-Hermitian systems than in Hermitian ones. In particular, for certain non-Hermitian systems, the eigenstates concentrate predominantly at the boundary, which is known as the non-Hermitian skin effect (NHSE) $[3–14]$ . Among many other consequences, it implies a fundamental revision of the principle of bulk-boundary correspondence $[11, 12]$ .

Whereas the NHSE has revealed intriguing static properties such as novel behaviors of eigenstates and energy spectra, in this work we unveil a striking dynamic boundary effect in non-Hermitian systems. We experimentally observe that in a class of lossy quantum walk of single photons, the loss rate is drastically enhanced at the boundary. Specifically, for a lossy particle initially located at a position far from the boundary of a lattice system, the space-resolved loss has a surprisingly high boundary peak, in sharp contrast to the common expectation that the particle loss should decay away from the initial position. Remarkably, the relative height of the edge peak even grows as the distance between the initial position and boundary increases. This striking phenomenon, dubbed non-Hermitian edge burst, has been predicted in recent theories $[15, 16]$ .

Since both the NHSE and edge burst involve boundary localization, it is tempting to attribute the latter to the former. However, it turns out that NHSE does not guarantee the emergence of edge burst. Closing the gap of the imaginary part of energy spectrum (i.e., the imaginary gap or dissipative gap) is the other necessary condition, which highlights the rich implication of spectral profile and topology in non-Hermitian systems $[9, 10]$ . At a deeper level, a novel dynamic bulk-edge scaling relation has been suggested as the origin of edge burst $[15]$ . Thus, the edge burst signifies an unprecedented interplay between non-Hermitian topological physics and non-Hermitian dynamical phenomena.

Lossy quantum walk.—To study the non-Hermitian edge burst, we design a one-dimensional quantum walk $[17–20]$ with the Floquet operator

$$
U = R \left(\frac {\theta_ {2}}{2}\right) S R \left(\frac {\theta_ {1}}{2}\right) L (\gamma). \tag {1}
$$

The shift operator $S = \sum_{x} |x - 1\rangle\langle x| \otimes |0\rangle\langle 0| + |x + 1\rangle\langle x| \otimes |1\rangle\langle 1|$ , so that the walker's position is shifted from the site $x$ to $x - 1$ or $x + 1$ according to the coin state $|0\rangle$ or $|1\rangle$ . The coin state is rotated along the $y$ axis by $R(\theta) = \mathbb{1}_{\mathrm{w}} \otimes e^{-i\theta \sigma_y}$ , where $\mathbb{1}_{\mathrm{w}} = \sum_{x} |x\rangle\langle x|$ is the identity operator. The operator $L(\gamma) = \mathbb{1}_{\mathrm{w}} \otimes \begin{pmatrix} 1 & 0 \\ 0 & e^{-2\gamma} \end{pmatrix}$ generates a state-selective loss. For our photonic platform, it is more convenient to create a domain wall instead of an open boundary [see Fig. 1(a)]. The left (L) and right (R) regions are characterized by coin parameters $\theta_{1,2}^L$ and $\theta_{1,2}^R$ , respectively. The dynamics of the non-Hermitian quantum walk follows

$$
| \psi (t) \rangle = U ^ {t} | \psi (0) \rangle , \tag {2}
$$

where $|\psi(0)\rangle$ is the initial state and t is the integer discrete time. One can also define an effective non-Hermitian Hamiltonian $H_{eff}$ by $U = \exp(-iH_{\mathrm{eff}})$ , which shares the same eigenstates as U.

The Floquet operator U defined in Eq. (1) and the associated $H_{eff}$ exhibit the NHSE, which originates from the state-dependent directional hoppings built in the model (akin to Refs. [11, 21]). In the presence of a domain wall [Fig. 1(a)], all the eigenstates of U exhibit localization at the domain wall when the non-Hermiticity is nonzero, i.e., $\gamma \neq 0$ . Accordingly, the generalized Brillouin zone (GBZ) deviates from the unit circle [see Figs. 2(a) and (b)] [3, 22, 23]. Here, we focus on two sets of parameters, $\theta_{2}^{R} = 0.12\pi$ and $\theta_{2}^{R} = 0.48\pi$ , with other parameters fixed as $\theta_{1,2}^{L} = 0.85\pi$ , $\theta_{1}^{R} = 0.12\pi$ , and $\gamma = 0.8$ . In Figs. 2(c) and (d), we show the energy spectrum of $H_{eff}$ , which clearly indicates that the imaginary

![](images/6a046cdeae28b192bc722940e57cba1f86123687c020f2871ca0700104b70101.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph Left
        A["x=-3"] -->|R| B["●"]
        B -->|L| C["S"]
        C -->|R| D["●"]
        D -->|L| E["S"]
        E -->|0| F["●"]
        F -->|R| G["●"]
        G -->|1| H["S"]
    end
    subgraph Right
        I["x=0"] -->|R| J["●"]
        J -->|L| K["S"]
        K -->|R| L["●"]
        L -->|L| M["S"]
        M -->|0| N["●"]
        N -->|R| O["●"]
        O -->|L| P["S"]
        P -->|0| Q["●"]
        Q -->|R| R["●"]
        R -->|L| S["●"]
    end
```
</details>

![](images/a5f8e429df36d6f5e8b883d25d4f3ec75e5dae3318f1848aca5e330c8cb9ea6e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_Left_Bus["Left Diagram"]
        A["L"] --> B["R"]
        B --> C["S"]
        C --> D["R"]
        D --> E["L"]
        E --> F["R"]
        F --> G["S"]
        G --> H["R"]
        H --> I["L"]
        I --> J["R"]
        J --> K["S"]
        K --> L["R"]
        L --> M["L"]
        M --> N["R"]
        N --> O["S"]
        O --> P["L"]
        P --> Q["R"]
        Q --> R["S"]
        R --> S["R"]
        S --> T["L"]
        T --> U["R"]
        U --> V["S"]
        V --> W["L"]
        W --> X["R"]
        X --> Y["S"]
        Y --> Z["R"]
        Z --> AA["L"]
        AA --> AB["R"]
        AB --> AC["S"]
        AC --> AD["L"]
        AD --> AE["R"]
        AE --> AF["S"]
        AF --> AG["R"]
        AG --> AH["L"]
        AH --> AI["R"]
        AI --> AJ["S"]
        AJ --> AK["L"]
        AK --> AL["R"]
        AL --> AM["S"]
        AM --> AN["L"]
        AN --> AO["R"]
        AO --> APD
    end

    subgraph_Right_Bus["Right Diagram"]
        B --> APD1["APD"]
        APD1 --> APD2["APD"]
        APD2 --> APD3["APD"]
        APD3 --> APD4["APD"]
        APD4 --> APD5["APD"]
        APD5 --> APD6["APD"]
        APD6 --> APD7["APD"]
        APD7 --> APD8["APD"]
        APD8 --> APD9["APD"]
        APD9 --> APD10["APD"]
        APD10 --> APD11["APD"]
        APD11 --> APD12["APD"]
        APD12 --> APD13["APD"]
        APD13 --> APD14["APD"]
        APD14 --> APD15["APD"]
        APD15 --> APD16["APD"]
        APD16 --> APD17["APD"]
        APD17 --> APD18["APD"]
        APD18 --> APD19["APD"]
        APD19 --> APD20["APD"]
        APD20 --> APD21["APD"]
        APD21 --> APD22["APD"]
        APD22 --> APD23["APD"]
        APD23 --> APD24["APD"]
        APD24 --> APD25["APD"]
        APD25 --> APD26["APD"]
        APD26 --> APD27["APD"]
        APD27 --> APD28["APD"]
        APD28 --> APD29["APD"]
        APD29 --> APD30["APD"]
        APD30 --> APD31["APD"]
        APD31 --> APD32["APD"]
        APD32 --> APD33["APD"]
        APD33 --> APD34["APD"]
        APD34 --> APD35["APD"]
        APD35 --> APD36["APD"]
        APD36 --> APD37["APD"]
        APD37 --> APD38["APD"]
        APD38 --> APD39["APD"]
        APD39 --> APD40["APD"]
        APD40 --> APD41["APD"]
        APD41 --> APD42["APD"]
        APD42 --> APD43["APD"]
        APD43 --> APD44["APD"]
        APD44 --> APD45["APD"]
        APD45 --> APD46["APD"]
        APD46 --> APD47["APD"]
        APD47 --> APD48["APD"]
        APD48 --> APD49["APD"]
        APD49 --> APD50["APD"]
        APD50 --> APD51["APD"]
        APD51 --> APD52["APD"]
        APD52 --> APD53["APD"]
        APD53 --> APD54["APD"]
        APD54 --> APD55["APD"]
        APD55 --> APD56["APD"]
        APD56 --> APD57["APD"]
        APD57 --> APD58["APD"]
        APD58 --> APD59["APD"]
        APD59 --> APD60["APD"]
        APD60 --> APd61["APd"]
```
</details>

FIG. 1. Experimental implementation. (a) The domain-wall geometry of the non-Hermitian quantum walk. The operations of S, R, L contained in U are pictorially shown. (b) Experimental setup. Photon pairs are created by the spontaneous parametric down conversion process in a type-II cut PPKTP crystal. One of the photon is injected into the quantum-walk interferometric network, and the other is used as the trigger. The walker photon passes the polarizing beam splitter (PBS) and the half-wave plate (HWP), so that its polarization is prepared in the coin state $|0\rangle$ . It then undertakes the quantum walk through the network containing partially polarizing beam splitters (PPBSs), HWPs, beam displacers (BDs). Finally, avalanche photodiodes (APDs) are used to detect the walker photons that coincide with the trigger photons.

gap (the gap between 0 and the maximum imaginary part of the spectrum) is zero for $\theta_{2}^{R}=0.12\pi$ but nonzero for $\theta_{2}^{R}=0.48\pi$ . In fact, the imaginary gap vanishes along the lines $\theta_{1}=2\pi n\pm\theta_{2}\ (n\in\mathbb{Z})$ (see Supplementary Information).

Observation of edge burst.—In our experiment, a walker is initialized at a site $x_{0}$ , which evolves under Eq. (2) in discrete time steps. The key quantity for edge burst is the probability $P(x)$ that the walker escapes from the position x. In practice, one can measure the space-time-resolved loss $p(x,t)$ from t=1 to t=T, with T being a large integer so that the loss is almost complete. The sum over t then gives

$$
P (x) = \sum_ {t = 1} ^ {T} p (x, t). \tag {3}
$$

According to the specific form of loss adopted here, we have

$$
p (x, t) = (1 - e ^ {- 4 \gamma}) | \langle 1 | \otimes \langle x | \psi (t - 1) \rangle | ^ {2}. \tag {4}
$$

It may also be written as $p(x,t) = |\langle 1| \otimes \langle x| M|\psi(t - 1)\rangle|^2$ with $M = \mathbb{1}_{\mathrm{w}} \otimes \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{1 - e^{-4\gamma}} \end{pmatrix}$ , which can be implemented by a partial measurement via the PPBS [see Fig. 1(a)] at the time step $t$ . We also define a time-dependent total loss probability

$$
P (t) = \sum_ {t ^ {\prime} = 1} ^ {t} \sum_ {x} p (x, t ^ {\prime}), \tag {5}
$$

so that the survival probability after a t-step evolution is $1 - P(t)$ . In our quantum-walk platform, $p(x, t)$ can be readily extracted from photon-number measurements (see Methods), and $P(x)$ , $P(t)$ can be obtained from Eqs. (3)(5).

We implement a 14-step $(T=14)$ quantum walk with initial walker location $x_{0}=10$ . The space-resolved loss probability $P(x)$ is shown in Figs. 2(e) and (f) for the aforementioned two sets of parameters. In both (e) $(\theta_{2}^{R}=0.12\pi)$ and (f) $(\theta_{2}^{R}=0.48\pi)$ , we observe that the loss probability initially decays away from $x_{0}$ . Moreover, the $P(x)$ profile is asymmetric around $x_{0}$ , which can be naturally attributed to the NHSE.

The surprising feature is an exceptionally high peak emerging at the domain wall in Fig. 2(e). Intuitively, one may resort to the NHSE to explain this edge burst. However, the NHSE is also strong for the parameters of Fig. 2(f), yet the edge burst is not seen there. Therefore, the origin of edge burst cannot be explained by the NHSE alone. In fact, the imaginary gap plays an essential role here [15]. The corresponding imaginary gap, shown in Figs. 2(c) and (d), is zero and nonzero for Figs. 2(e) and (f), respectively.

To unveil the space-time profile of walker's loss, we plot $p(x,t)$ for the above two sets of parameters. Figs. 2(g) and (h) show that the walker propagates almost ballistically with concurrent loss along the trajectory. In the case of edge burst, a large loss peak in $p(x,t)$ emerges when the walker hits the domain wall. It also indicates that the burst occurs around a particular time, before which it is indiscernible.

![](images/507c5f1eb3617eefd68ac78d2a84900d57ec2540c94d67e87cd9d5eccaf0c179.jpg)

<details>
<summary>scatter</summary>

| Region     | Re(β) Range | Im(β) Range |
|------------|-------------|-------------|
| BZ         | ~ -1 to 1   | ~ -1 to 2   |
| GBZ_Right  | ~ -1 to 1   | ~ -1 to 2   |
| GBZ_Left  | ~ -1 to 1   | ~ -1 to 2   |
</details>

![](images/63398e5dc66a7757f3b4f46b70bcf3323313717031fd7a0d4cb1e8be34ede373.jpg)

<details>
<summary>line</summary>

| Re(E) | Im(E) |
|-------|-------|
| -3.0  | -0.1  |
| -2.0  | -0.1  |
| -1.0  | -0.1  |
| 0.0   | -0.1  |
| 1.0   | -0.1  |
| 2.0   | -0.1  |
| 3.0   | -0.1  |
</details>

![](images/e624619125dafb0a536aba332f3c77c2bac111ba254b81a1ec06bbb5c589f99c.jpg)

<details>
<summary>bar</summary>

| x    | Simulation | Experiment |
| ---- | ---------- | ---------- |
| -2   | 0.0        | 0.45       |
| 0    | 0.0        | 0.0        |
| 2    | 0.0        | 0.0        |
| 4    | 0.0        | 0.0        |
| 6    | 0.0        | 0.0        |
| 8    | 0.0        | 0.05       |
| 10   | 0.0        | 0.05       |
| 12   | 0.0        | 0.0        |
</details>

![](images/03c3d584087ae216f697d12c4a483d25207dcb358f044a9881db63b534202bc5.jpg)

<details>
<summary>bar</summary>

| t   | p(x,t) |
| --- | ------ |
| 0   | -2     |
| 1   | -1     |
| 2   | 0      |
| 3   | 0.1    |
| 4   | 0.4    |
| 5   | 0.2    |
| 6   | 0.1    |
| 7   | 0.05   |
| 8   | 0.02   |
| 9   | 0.01   |
| 10  | 0.005  |
| 11  | 0.002  |
| 12  | 0.001  |
| 13  | 0.0005 |
| 14  | 0.0002 |
</details>

![](images/439b57d36761e19b0b1e865937d7d910a8c9299daf44eacdb26f01ebdf32953f.jpg)

<details>
<summary>scatter</summary>

| Region     | Re(β) Range | Im(β) Range |
|------------|-------------|-------------|
| BZ         | ~0 to 2     | ~-2 to 2    |
| GBZ_Right  | ~0 to 2     | ~-2 to 2    |
| GBZ_Left   | ~0 to 2     | ~-2 to 2    |
</details>

![](images/2d8f8410a3c61abce968f64bf651548484819298d6e328ded3ecc37eeb78af41.jpg)

<details>
<summary>line</summary>

| Re(E) | Im(E) |
|-------|-------|
| -2    | 0.0   |
| 0     | -1.5  |
| 2     | 0.0   |
</details>

![](images/c91b22c44e2e2c99aeaa7c3aa465b1b6a0ca1ab03a2973525c9254c4ac43a970.jpg)

<details>
<summary>bar</summary>

| x    | Simulation | Experiment |
| ---- | ---------- | ---------- |
| -4   | 0.0        | 0.0        |
| -2   | 0.0        | 0.0        |
| 0    | 0.0        | 0.0        |
| 2    | 0.0        | 0.0        |
| 4    | 0.0        | 0.0        |
| 6    | 0.0        | 0.05       |
| 8    | 0.0        | 0.22       |
| 10   | 0.0        | 0.45       |
| 12   | 0.0        | 0.02       |
| 14   | 0.0        | 0.0        |
</details>

![](images/b1ab9eb8b2027c84daf9f630b400091eb1a8839d88a960e15f8932c0cf6a107f.jpg)

<details>
<summary>bar</summary>

| t    | p(x,t) |
| ---- | ------ |
| 0    | -2     |
| 1    | 0      |
| 2    | 0.35   |
| 3    | 0.1    |
| 4    | 0.05   |
| 5    | 0.02   |
| 6    | 0.01   |
| 7    | 0.005  |
| 8    | 0.002  |
| 9    | 0.001  |
| 10   | 0.0005 |
| 11   | 0.0002 |
| 12   | 0.0001 |
| 13   | 0.00005|
| 14   | 0.00002|
</details>

FIG. 2. Edge burst in non-Hermitian quantum walks. The fixed parameters are $\theta_{1,2}^{L} = 0.85\pi$ , $\theta_1^R = 0.12\pi$ and $\gamma = 0.8$ . (a)(b) Brillouin zone (BZ) and generalized Brillouin zone (GBZ) for $\theta_2^R = 0.12\pi$ and $\theta_2^R = 0.48\pi$ . (c)(d) Energy spectra (for the right region in which the walker is initialized) under the periodic boundary condition (PBC) for two indicated values of $\theta_2^R$ . (e)(f) Experimentally measured $P(x)$ of a 14-step non-Hermitian quantum walk with the initial state $|x_0 = 10\rangle \otimes |0\rangle$ . (g)(h) The space-time-resolved loss probability $p(x,t)$ for the two values of $\theta_2^R$ . Error bars represent the statistical uncertainty under the assumption of Poissonian statistics.

Furthermore, we vary the initial position $x_{0} = 5, 6, 7, 8, 9, 10$ and measure the time-dependent loss probability $P(t)$ . As shown in Fig. 3(a), for $\theta_{2}^{R} = 0.12\pi$ (with edge burst), $P(t)$ suddenly increases near the domain wall. In contrast, in Fig. 3(b), for $\theta_{2}^{R} = 0.48\pi$ (without edge burst), $P(t)$ increases steadily with t without sudden change. Similarly, the space-resolved survival probability $|\psi(x = -1, t)|^{2}$ at the domain wall at each step t behaves differently with and without the edge burst [see Fig. 3(c)]. The value of $|\psi(x = -1, t)|^{2}$ is significantly larger in the presence of edge burst. In Fig. 3(d), we show that the edge burst remains robust when the starting position varies. In contrast, when the edge burst is absent, $P(x)$ decays rapidly as $x_{0}$ moves away from the domain wall [see Fig. 3(e)].

To further characterize the edge burst, we measure the relative height $P_{domain}/P_{min}$ , where $P_{domain} \equiv P(x = -1)$ is the probability that the photon escapes from the domain wall x = -1, and $P_{\min} \equiv \min_{x=-1,\cdots,x_0} \{P(x)\}$ is the minimum of $P(x)$ in the interval between the initial location $x_0$ and the domain wall location x = -1. The edge burst is characterized by $P_{domain}/P_{min} \gg 1$ , while its absence means that $P_{domain}/P_{min}$ is on the order of unity. As shown in Fig. 3(f), for $\theta_2^R = 0.48\pi$ , the measured relative height remains close to 1 as $x_0$ increases. In stark contrast, for $\theta_2^R = 0.12\pi$ , the relative height increases with $x_0$ and fits well with a linear relation $P_{domain}/P_{min} \sim x_0$ . Thus, the relative height grows as the initial walker position moves away from the domain wall. While counterintuitive, this behavior is a consequence of a novel bulk-edge scaling relation $[15]$ .

Discussions.—We present the first experimental observation of the non-Hermitian edge burst by using discrete-time non-Hermitian quantum walk of photons. Our experiment not only demonstrates that edge burst originates from the intriguing interplay between two unique non-Hermitian concepts, the NHSE and imaginary gap, but also unveils the real-time dynamics of this phenomenon. The observation of non-Hermitian edge burst paves the way for investigating the real-time dynamics in non-Hermitian topological systems, which remains largely unexplored. From a practical perspective, the edge burst may offer a promising non-Hermitian approach for the on-demand harvesting of light or particles at a prescribed position.

# Methods

Implementation.—For the experimental implementation, we adopt the scheme of single-photon discrete-time quantum walks illustrated in Fig. 1(b). Photon pairs are created by spontaneous parametric down conversion, where a 20mm type-II periodically poled potassium titanyl phosphate (PPKTP) crystal is pumped by a 405nm continuous wave diode laser with the power of 1mW. One photon serves as a trigger, and the other as a heralded single photon undertaking the quantum walk. The photon polarizations are adopted as the coin state. The

![](images/3122bf5f64dd3cae8d2219d69e1b8fd4722178f03d9a8ef707d58b5e3d943713.jpg)

<details>
<summary>line</summary>

| t   | x₀  | P(t) |
| --- | --- | ---- |
| 14  | 10  | 0.85 |
| 12  | 9   | 0.83 |
| 10  | 8   | 0.81 |
| 8   | 7   | 0.79 |
| 6   | 6   | 0.77 |
| 4   | 5   | 0.75 |
| 2   | 4   | 0.73 |
| 0   | 3   | 0.71 |
| 2   | 2   | 0.69 |
| 4   | 1   | 0.67 |
| 6   | 0   | 0.65 |
| 8   | 1   | 0.63 |
| 10  | 2   | 0.61 |
| 12  | 3   | 0.59 |
| 14  | 4   | 0.57 |
</details>

![](images/bfd9e5662b08aad8f1671c53c8d9c66b337ee1feec2bee1b958d3936988ae832.jpg)

<details>
<summary>line</summary>

| t   | x₀  | P(t) |
| --- | --- | ---- |
| 14  | 10  | 0.9  |
| 12  | 8   | 0.85 |
| 10  | 6   | 0.8  |
| 8   | 4   | 0.75 |
| 6   | 2   | 0.7  |
| 4   | 1   | 0.65 |
| 2   | 0.5 | 0.6  |
| 1   | 0.2 | 0.55 |
| 0.5 | 0.1 | 0.5  |
| 0.2 | 0.05| 0.45 |
| 0.1 | 0.02| 0.4  |
| 0.05| 0.01| 0.35 |
| 0.02| 0.005| 0.3 |
| 0.01| 0.002| 0.25 |
| 0.005| 0.001| 0.2 |
| 0.002| 0.0005| 0.15 |
| 0.001| 0.0002| 0.1 |
| 0.0005| 0.0001| 0.05 |
| 0.0002| 0.00005| 0.02|
| 0.0001| 0.00002| 0.01|
| 0.00005| 0.00001| 0.005|
| 0.00002| 0.000005| 0.0 |
| 0.00001| 0.000002| -    |
</details>

![](images/0216cfebcc488df2bf41beb82ba65cad9248011e7be1cbc1a084bf994373b548.jpg)

<details>
<summary>line</summary>

| t    | x₀=5  | x₀=6  | x₀=7  | x₀=8  | x₀=9  | x₀=10 |
| ---- | ----- | ----- | ----- | ----- | ----- | ----- |
| 2    | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  |
| 4    | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  |
| 6    | 0.60  | 0.58  | 0.55  | 0.52  | 0.50  | 0.48  |
| 8    | 0.05  | 0.15  | 0.18  | 0.25  | 0.35  | 0.45  |
| 10   | 0.02  | 0.12  | 0.15  | 0.20  | 0.30  | 0.35  |
| 12   | 0.01  | 0.08  | 0.12  | 0.15  | 0.25  | 0.30  |
| 14   | 0.00  | 0.05  | 0.10  | 0.12  | 0.22  | 0.28  |
</details>

![](images/6d79670c9758e7b0a326ce3cd5895455ffb23a9dcf4fa1ed0e384e9e3af692c7.jpg)

<details>
<summary>bar</summary>

| x    | Simulation (θ²ᴿ=0.12π) | Experiment (θ²ᴿ=0.12π) |
| ---- | ------------------------ | ----------------------- |
| -2   | ~0.0                     | ~0.6                    |
| 0    | ~0.0                     | ~0.6                    |
| 2    | ~0.0                     | ~0.0                    |
| 4    | ~0.0                     | ~0.0                    |
| 6    | ~0.0                     | ~0.0                    |
| 8    | ~0.0                     | ~0.0                    |
| 10   | ~0.0                     | ~0.0                    |
| 12   | ~0.0                     | ~0.0                    |
| 14   | ~0.0                     | ~0.0                    |
</details>

![](images/afdb9ff808a974abbaadd2a213b07480f314aca9eaaee7e21df3923c9121fa63.jpg)

<details>
<summary>bar</summary>

| x    | Simulation (θ²ᴿ=0.48π) | Experiment (θ²ᴿ=0.48π) |
| ---- | ------------------------ | ----------------------- |
| -2   | 0.0                      | 0.0                     |
| 0    | 0.0                      | 0.0                     |
| 2    | 0.0                      | 0.0                     |
| 4    | 0.0                      | 0.3                     |
| 6    | 0.0                      | 0.5                     |
| 8    | 0.0                      | 0.3                     |
| 10   | 0.0                      | 0.5                     |
| 12   | 0.0                      | 0.0                     |
| 14   | 0.0                      | 0.0                     |
</details>

![](images/a1d0035f1acfa83794e9bc9fae042053c2eb9e5002e18f09bb112b716d816a29.jpg)

<details>
<summary>line</summary>

| x₀ | Simulation | Experiment | Fitting |
|----|------------|------------|---------|
| 5  | 15.0       | 15.0       | 15.0    |
| 6  | 16.0       | 16.0       | 16.0    |
| 7  | 17.0       | 17.0       | 17.0    |
| 8  | 18.0       | 18.0       | 18.0    |
| 9  | 19.0       | 19.0       | 19.0    |
| 10 | 20.0       | 20.0       | 20.0    |
</details>

FIG. 3. (a)(b) Experimentally measured time-dependent total loss probability $P(t)$ for different starting positions $x_0 = 5,6,7,8,9,10$ , respectively. For (a), $\theta_2^R = 0.12\pi$ ; for (b), $\theta_2^R = 0.48\pi$ . Other parameters are the same as those in Fig. 2. (c) The measured survival probability at the domain wall, $|\psi(x = -1,t)|^2$ , for different starting positions. $\theta_2^R = 0.12\pi$ (upper panel) and $0.48\pi$ (lower panel). (d) (e) Experimentally measured $P(x)$ ( $T = 14$ ) for $x_0 = 6,8,10$ ; $\theta_2^R = 0.12\pi$ for (d) and $0.48\pi$ for (e). (f) The measured relative height $P_{\mathrm{domain}} / P_{\mathrm{min}}$ versus $x_0$ . Hollow and solid symbols represent numerically evaluated results and experimental data, respectively. Dashed lines are obtained by numerical fitting experimental data.

walker photon is initialized in the spatial mode $|x_{0}\rangle$ with the internal state $|0\rangle$ , i.e. $|\psi(0)\rangle = |x_{0}\rangle \otimes |0\rangle$ . The localized initial state is prepared by passing the walker photons through a half-wave plate (HWP) and a polarizing beam splitter (PBS).

For the quantum-walk dynamics, the shift operator S is implemented by a beam displacer (BD) whose optical axis is cut in the way so that the vertically polarized photons are directly transmitted and the horizontally polarized photons are laterally displaced into a neighboring mode. The coin rotation $R\left(\frac{\theta_{1(2)}}{2}\right)$ is realized by two HWPs at 0 and $\frac{\theta_{1(2)}}{4}$ , respectively. The loss operator $L(\gamma)$ is realized by a partially polarizing beam splitter (PPBS), which completely transmits the coin state $|0\rangle$ but reflects the coin state $|1\rangle$ with a probability $e^{-4\gamma}$ . At last, avalanche photodiodes (APDs) are used to detect the walker photons coinciding with the trigger photons. The total number of coincidences is approximately 23000.

The measurements are based on photon-number counting. The space-time-resolved probability $p(x,t)$ can be calculated from the photon number through

$$
p (x, t) = \frac {N (x , t)}{\sum_ {x ^ {\prime}} N ^ {\prime} (x ^ {\prime} , t) + \sum_ {t ^ {\prime} = 1} ^ {t} \sum_ {x ^ {\prime}} N (x ^ {\prime} , t ^ {\prime})}, \tag {6}
$$

where $N(x,t)$ is the number of photons escaping from the position x at the time step t, and $N'(x,t)$ is the number of remaining photons at x after a t-step evolution.

Finally, the space-resolved survival probability at x can be calculated as

$$
| \psi (x, t) | ^ {2} = \frac {N ^ {\prime} (x , t)}{\sum_ {x ^ {\prime}} N ^ {\prime} (x ^ {\prime} , t) + \sum_ {t ^ {\prime} = 1} ^ {t} \sum_ {x ^ {\prime}} N (x ^ {\prime} , t ^ {\prime})}. \tag {7}
$$

Note. After completing this work, we learned of a related experiment by a team at Southern University of Science and Technology.

# Acknowledgments

This work has been supported by the National Natural Science Foundation of China (Grant Nos. 92265209, 12025401, 12125405, 11974331 and 12104036.)

\* These authors contributed equally to this work.   
$^{\dagger}$ wyiz@ustc.edu.cn   
‡ wangzhongemail@tsinghua.edu.cn   
§ gnep.eux@gmail.com   
[1] Ashida, Y., Gong, Z. & Ueda, M. Non-Hermitian physics. Adv. Phys. 69, 249-435 (2020).

[2] Bergholtz, E. J., Budich, J. C. & Kunst, F. K. Exceptional topology of non-Hermitian systems. Rev. Mod. Phys. 93, 015005 (2021).   
[3] Yao, S. & Wang, Z. Edge states and topological invariants of non-Hermitian systems. Phys. Rev. Lett. 121, 086803 (2018).   
[4] Yao, S., Song, F. & Wang, Z. Non-Hermitian chern bands. Phys. Rev. Lett. 121, 136802 (2018).   
[5] Kunst, F. K., Edvardsson, E., Budich, J. C. & Bergholtz, E. J. Biorthogonal bulk-boundary correspondence in non-Hermitian systems. Phys. Rev. Lett. 121, 026808 (2018).   
[6] Martinez Alvarez, V. M., Barrios Vargas, J. E. & Foa Torres, L. E. F. Non-Hermitian robust edge states in one dimension: Anomalous localization and eigenspace condensation at exceptional points. Phys. Rev. B 97, 121401(R) (2018).   
[7] Lee, C. H. & Thomale, R. Anatomy of skin modes and topology in non-Hermitian systems. Phys. Rev. B 99, 201103(R) (2019).   
[8] Song, F., Yao, S. & Wang, Z. Non-Hermitian skin effect and chiral damping in open quantum systems. Phys. Rev. Lett. 123, 170401 (2019).   
[9] Zhang, K., Yang, Z. & Fang, C. Correspondence between winding numbers and skin modes in non-Hermitian systems. Phys. Rev. Lett. 125, 126402 (2020).   
[10] Okuma, N., Kawabata, K., Shiozaki, K. & Sato, M. Topological origin of non-Hermitian skin effects. Phys. Rev. Lett. 124, 086801 (2020).   
[11] Xiao, L., Deng, T., Wang, K., Zhu, G., Wang, Z., Yi W. & Xue, P. Non-Hermitian bulk-boundary correspondence in quantum dynamics. Nat. Phys. 16, 761-766 (2020).   
[12] Helbig, T., Hofmann, T., Imhof, S., Abdelghany, M., Kiessling, T., Molenkamp, L. W., Lee, C. H., Szameit, A., Greiter, M. & Thomale, R. Generalized bulk-boundary correspondence in non-Hermitian topoelectrical circuits. Nat. Phys. 16, 747-750 (2020).   
[13] Wang, W., Wang, X. & Ma, G. Non-Hermitian morphing of topological modes. Nature 608, 50-55 (2022).   
[14] Ghatak, A., Brandenbourger, M., Van Wezel, J. & Coulais, C. Observation of non-Hermitian topology and its bulk-edge correspondence in an active mechanical metamaterial. Proc. Natl. Acad. Sci. U.S.A. 117, 29561-29568 (2020).   
[15] Xue, W. T., Hu, Y. M., Song, F. & Wang, Z. Non-Hermitian edge burst. Phys. Rev. Lett. 128, 120401 (2022).   
[16] Wang, L., Liu, Q. & Zhang, Y. Quantum dynamics on a lossy non-Hermitian lattice, Chin. Phys. B 30, 020506 (2021).   
[17] Aharonov, Y., Davidovich, L. & Zagury, N. Quantum random walks. Phys. Rev. A 48, 1687 (1993).   
[18] Farhi, E. & Gutmann, S. Quantum computation and decision trees. Phys. Rev. A 58, 915 (1998).   
[19] Watrous, J. Quantum simulations of classical random walks and undirected graph connectivity. J. Comput. Syst. Sci. 62, 376-391 (2001).   
[20] Childs, A. M. Universal computation by quantum walk. Phys. Rev. Lett. 102, 180501 (2009).   
[21] Xiao, L., Deng, T., Wang, K., Wang, Z., Yi, W. & Xue, P. Observation of non-Bloch parity-time symmetry and exceptional points. Phys. Rev. Lett. 126, 230402 (2021).   
[22] Yokomizo, K. & Murakami, S. Non-Bloch band theory of non-Hermitian systems. Phys. Rev. Lett. 123, 066404 (2019).

[23] Deng, T. S. & Yi, W. Non-Bloch topological invariants in a non-Hermitian domain wall system. Phys. Rev. B 100, 035102 (2019).

# Supplemental Material for “Observation of non-Hermitian edge burst in quantum dynamics”

# Effective Hamiltonian and generalized Brillouin zone

In this section, we derive an expression for the effective Hamiltonian $H_{eff}$ in momentum space. First, we transform the real-space nonunitary Floquet operator U and its conjugate transpose $U^{\dagger}$ into the momentum-space $U_{k}$ and $U_{k}^{\dagger}$ :

$$
U _ {k} = d _ {0} \sigma_ {0} - i d _ {1} \sigma_ {1} - i d _ {2} \sigma_ {2} - i d _ {3} \sigma_ {3},
$$

$$
U _ {k} ^ {\dagger} = d _ {0} ^ {*} \sigma_ {0} + i d _ {1} ^ {*} \sigma_ {1} + i d _ {2} ^ {*} \sigma_ {2} + i d _ {3} ^ {*} \sigma_ {3} \tag {S1}
$$

where $\sigma_{1,2,3}$ are the Pauli matrices and $\sigma_{0}$ is the identity matrix, and

$$
d _ {0} = e ^ {- \gamma} (\cosh \gamma \cos k \cos \frac {\theta_ {1} + \theta_ {2}}{2} + i \sinh \gamma \sin k \cos \frac {\theta_ {1} - \theta_ {2}}{2}),
$$

$$
d _ {1} = e ^ {- \gamma} (\cosh \gamma \sin k \sin \frac {\theta_ {1} - \theta_ {2}}{2} + i \sinh \gamma \cos k \sin \frac {\theta_ {1} + \theta_ {2}}{2}),
$$

$$
d _ {2} = e ^ {- \gamma} (\cosh \gamma \cos k \sin \frac {\theta_ {1} + \theta_ {2}}{2} - i \sinh \gamma \sin k \sin \frac {\theta_ {1} - \theta_ {2}}{2}),
$$

$$
d _ {3} = e ^ {- \gamma} (- \cosh \gamma \sin k \cos \frac {\theta_ {1} - \theta_ {2}}{2} + i \sinh \gamma \cos k \cos \frac {\theta_ {1} + \theta_ {2}}{2}). \tag {S2}
$$

Note that the relation $\sqrt{d_0^2 + d_1^2 + d_2^2 + d_3^2} = e^{-\gamma}$ is satisfied. The eigenvalue and eigenvector can be derived from

$$
U _ {k} | \psi_ {\pm} \rangle = \lambda_ {\pm} | \psi_ {\pm} \rangle , U _ {k} ^ {\dagger} | \chi_ {\pm} \rangle = \lambda_ {\pm} ^ {*} | \chi_ {\pm} \rangle . \tag {S3}
$$

Straightforward calculations lead to

$$
\lambda_ {\pm} = d _ {0} \pm i t _ {0}, \lambda_ {\pm} ^ {*} = d _ {0} ^ {*} \mp i t _ {0} ^ {*}, \tag {S4}
$$

$$
| \psi_ {\pm} \rangle = \frac {1}{d _ {1} + i d _ {2}} \binom{d _ {3} \mp t _ {0}}{d _ {1} + i d _ {2}},
$$

$$
\langle \chi_ {\pm} | = \frac {1}{d _ {1} - i d _ {2}} (d _ {3} \mp t _ {0}, d _ {1} - i d _ {2}), \tag {S5}
$$

where $t_{0}=\sqrt{e^{-2\gamma}-d_{0}^{2}}$ . Since the effective Hamiltonian $H_{\mathrm{eff}}(k)$ is related to $U_{k}$ through $U_{k}=e^{-iH_{eff}}$ , the quasienergy spectrum of $H_{\mathrm{eff}}(k)$ is

$$
E _ {\pm} (k) = i \ln \lambda_ {\pm} (k) = \pm \arccos (\cosh \gamma \cos k \cos \frac {\theta_ {1} + \theta_ {2}}{2} + i \sinh \gamma \sin k \cos \frac {\theta_ {1} - \theta_ {2}}{2}) - i \gamma . \tag {S6}
$$

Specifically, for $\theta_{1} = \theta_{2}$ , we have

$$
E _ {-} (k = \pi / 2) = - \arccos (i \sinh \gamma) - i \gamma = - \arccos (\sin i \gamma) - i \gamma = - \pi / 2, \tag {S7}
$$

so that $\operatorname{Im}[E_{-}(k = \pi /2)] = 0$ , i.e., the imaginary gap closes at $k = \pi /2$ . For $\theta_{1} = -\theta_{2}$ , the imaginary gap closes at $k = 0$ because

$$
E _ {+} (k = 0) = \arccos (\cosh \gamma) - i \gamma = \arccos (\cos (i \gamma)) - i \gamma = 0. \tag {S8}
$$

Thus, the imaginary gap closes when $\theta_{1} = 2\pi n\pm \theta_{2}$ ( $n\in \mathbb{Z}$ ). While the eigenvectors in Eq. (S5) are not orthogonal, one can derive a set of bi-orthonormal eigenvectors $\{| \tilde{\psi}_{\pm}\rangle ,|\tilde{\chi}_{\pm}\rangle \}$ :

$$
| \tilde {\psi} _ {\pm} \rangle = \frac {| \psi_ {\pm} \rangle}{\sqrt {\langle \chi_ {\pm} | \psi_ {\pm} \rangle}} = \frac {1}{\sqrt {2 t _ {0} (t _ {0} \mp d _ {3})}} \binom{d _ {3} \mp t _ {0}}{d _ {1} + i d _ {2}},
$$

$$
\langle \tilde {\chi} _ {\pm} | = \frac {\langle \chi_ {\pm} |}{\sqrt {\langle \chi_ {\pm} | \psi_ {\pm} \rangle}} = \frac {1}{\sqrt {2 t _ {0} (t _ {0} \mp d _ {3})}} (d _ {3} \mp t _ {0}, d _ {1} - i d _ {2}), \tag {S9}
$$

which satisfy

$$
\langle \tilde {\chi} _ {m} | \tilde {\psi} _ {n} \rangle = \delta_ {m n}, \sum_ {m = +, -} | \tilde {\psi} _ {m} \rangle \langle \tilde {\chi} _ {m} | = 1. \tag {S10}
$$

It follows that

$$
U _ {k} = \lambda_ {+} | \tilde {\psi} _ {+} \rangle \langle \tilde {\chi} _ {+} | + \lambda_ {-} | \tilde {\psi} _ {-} \rangle \langle \tilde {\chi} _ {-} |, \tag {S11}
$$

and the effective Hamiltonian $H_{\mathrm{eff}}(k)$ can be written as

$$
H _ {\text { eff }} = i \ln \lambda_ {+} | \tilde {\psi} _ {+} \rangle \langle \tilde {\chi} _ {+} | + i \ln \lambda_ {-} | \tilde {\psi} _ {-} \rangle \langle \tilde {\chi} _ {-} |. \tag {S12}
$$

To derive the generalized Brillouin zone (GBZ) [3, 22], we rewrite the Floquet operator $U$ as

$$
U = \sum_ {x} | x - 1 \rangle \langle x | \otimes A _ {0} + | x + 1 \rangle \langle x | \otimes A _ {1}, \tag {S13}
$$

where

$$
A _ {0} = R _ {c} \left(\frac {\theta_ {2}}{2}\right) P _ {0} R _ {c} \left(\frac {\theta_ {1}}{2}\right) L _ {c} (\gamma),
$$

$$
A _ {1} = R _ {c} \left(\frac {\theta_ {2}}{2}\right) P _ {1} R _ {c} \left(\frac {\theta_ {1}}{2}\right) L _ {c} (\gamma), \tag {S14}
$$

with $L_{c}(\gamma)=\left(\begin{array}{cc}1 & 0 \\ 0 & e^{-2\gamma}\end{array}\right)$ , $R_{c}(\theta)=e^{-i\theta\sigma_{y}}$ , $P_{0}=|0\rangle\langle0|$ and $P_{1}=|1\rangle\langle1|$ . In view of the translational symmetry inside the bulk, the eigenstate $|\varphi\rangle$ of U can be expressed as

$$
| \varphi \rangle = \sum_ {x, j} \beta_ {j} ^ {x} | x \rangle \otimes | \phi_ {j} \rangle_ {c}, \tag {S15}
$$

where $|\phi_{j}\rangle_{c}$ is the coin state and $\beta_{j}$ is the spatial-mode function. Inserting Eq. (S15) into eigen-equation $U|\varphi\rangle = \lambda|\varphi\rangle$ , we obtain

$$
(A _ {0} \beta + \frac {A _ {1}}{\beta} - \lambda) | \phi \rangle_ {c} = 0, \tag {S16}
$$

which has nontrivial solutions only when

$$
\det [ A _ {0} \beta + \frac {A _ {1}}{\beta} - \lambda ] = 0. \tag {S17}
$$

In an explicit form, Eq. (S17) is a quadratic equation of $\beta$ :

$$
[ \sin (\frac {\theta_ {1}}{2}) \sin (\frac {\theta_ {2}}{2}) - e ^ {2 \gamma} \cos (\frac {\theta_ {1}}{2}) \cos (\frac {\theta_ {2}}{2}) ] \beta^ {2} + (\frac {1}{\lambda} + e ^ {2 \gamma} \lambda) \beta + e ^ {2 \gamma} \sin (\frac {\theta_ {1}}{2}) \sin (\frac {\theta_ {2}}{2}) - \cos (\frac {\theta_ {1}}{2}) \cos (\frac {\theta_ {2}}{2}) = 0. \tag {S18}
$$

In the thermodynamic limit, the GBZ equation is determined by $|\beta_1(\lambda)| = |\beta_2(\lambda)|$ [3, 22]. Thus, we obtain

$$
\left| \beta_ {1} \right| = \left| \beta_ {2} \right| = \sqrt {\left| \frac {e ^ {2 \gamma} \sin \left(\frac {\theta_ {1}}{2}\right) \sin \left(\frac {\theta_ {2}}{2}\right) - \cos \left(\frac {\theta_ {1}}{2}\right) \cos \left(\frac {\theta_ {2}}{2}\right)}{\sin \left(\frac {\theta_ {1}}{2}\right) \sin \left(\frac {\theta_ {2}}{2}\right) - e ^ {2 \gamma} \cos \left(\frac {\theta_ {1}}{2}\right) \cos \left(\frac {\theta_ {2}}{2}\right)} \right|} = \sqrt {\left| \frac {\cosh \gamma \cos \frac {\theta_ {1} + \theta_ {2}}{2} - \sinh \gamma \cos \frac {\theta_ {2} - \theta_ {1}}{2}}{\cosh \gamma \cos \frac {\theta_ {1} + \theta_ {2}}{2} + \sinh \gamma \cos \frac {\theta_ {2} - \theta_ {1}}{2}} \right|.} \tag {S19}
$$

Therefore, the GBZ is a circle in the complex plane, as shown in Fig. 2 in the main text. When $|\beta| < 1(|\beta| > 1)$ , the skin modes are localized at the left (right) edge. According to Eq. (S19), when $\cos \frac{\theta_1 + \theta_2}{2} \cos \frac{\theta_2 - \theta_1}{2} > 0$ ( $\cos \frac{\theta_1 + \theta_2}{2} \cos \frac{\theta_2 - \theta_1}{2} < 0$ ), the skin modes are localized at the left (right) edge. For the two sets of parameters used in the main article, the skin modes are localized at the domain wall.

![](images/134327f2900ca6bd625931810810a721e0c6bfba3eb7b0ab2982548d924aa8ff.jpg)

<details>
<summary>line</summary>

| x₀  | P(x=1) |
| --- | ------ |
| 20  | 0.36   |
| 25  | 0.34   |
| 30  | 0.32   |
| 35  | 0.30   |
| 40  | 0.28   |
| 45  | 0.26   |
| 50  | 0.24   |
| 55  | 0.22   |
| 60  | 0.20   |
| 65  | 0.18   |
| 70  | 0.16   |
| 75  | 0.14   |
| 80  | 0.12   |
</details>

![](images/145c9d1e445d42c4692a7c2d5606eaf2e6303406d37b7539f8ee15a67ad67020.jpg)

<details>
<summary>line</summary>

| x₀-x | P(x)     |
|------|----------|
| 20   | 0.01     |
| 30   | 0.005    |
| 40   | 0.0025   |
| 50   | 0.00125  |
| 60   | 0.000625 |
| 70   | 0.0003125|
| 80   | 0.00015625|
</details>

![](images/ec04f42748abba6938af5878e098d83098da9b05a97dd98959f2e1607c687682.jpg)

<details>
<summary>line</summary>

| x₀  | P_domain/P_min |
| --- | -------------- |
| 20  | 40             |
| 25  | 50             |
| 30  | 60             |
| 35  | 70             |
| 40  | 80             |
| 45  | 90             |
| 50  | 100            |
| 55  | 110            |
| 60  | 120            |
| 65  | 130            |
| 70  | 140            |
| 75  | 150            |
| 80  | 160            |
</details>

![](images/26a280e469b46d4d41c1d478eb621d40051d845af2ad6389fc8faeb822b8586e.jpg)

<details>
<summary>line</summary>

| x₀  | P(x=1)        |
| --- | ------------- |
| 20  | 1.00E-02      |
| 30  | 1.00E-04      |
| 40  | 1.00E-06      |
| 50  | 1.00E-08      |
| 60  | 1.00E-10      |
| 70  | 1.00E-12      |
| 80  | 1.00E-14      |
</details>

![](images/7a9911ef63a7d3fb8cdc1ee07c1a55d87744a6e2c4bf20a02e94913169006e6d.jpg)

<details>
<summary>line</summary>

| x₀-x | P(x)        |
|------|-------------|
| 20   | 1.00E-05    |
| 30   | 1.00E-06    |
| 40   | 1.00E-07    |
| 50   | 1.00E-08    |
| 60   | 1.00E-09    |
| 70   | 1.00E-10    |
| 80   | 1.00E-11    |
</details>

![](images/8e4ae1f1572501aa69fb8ed31fc6a0d585b653d4b8172b1c69388eb14ba781ad.jpg)

<details>
<summary>line</summary>

| x₀  | P_domain/P_min |
| --- | ------------- |
| 20  | 2.1           |
| 25  | 2.1           |
| 30  | 2.1           |
| 35  | 2.1           |
| 40  | 2.1           |
| 45  | 2.1           |
| 50  | 2.1           |
| 55  | 2.1           |
| 60  | 2.1           |
| 65  | 2.1           |
| 70  | 2.1           |
| 75  | 2.0           |
| 80  | 1.9           |
</details>

FIG. S1. Numerical simulations for the loss probabilities $P(-1)$ (for the domain wall) and $P(x)$ (for the bulk), and the relative height $P_{domain}/P_{min}$ . The coin parameters are fixed as $\theta_{1,2}^{L}=0.85\pi$ and $\theta_{1}^{R}=0.12\pi$ . For the upper row (red), $\theta_{2}^{R}=0.12\pi$ , and the edge burst is present; for the lower row (blue), $\theta_{2}^{R}=0.48\pi$ , and the edge burst is absent. (a)(b) The loss probability $P(x=-1)$ versus $x_{0}$ . (c)(d) $P(x)$ versus $x_{0}-x$ . (e)(f) The relative height $P_{domain}/P_{min}$ . The dots are from numerical simulations, and the black solid lines are the fitting results.

# Numerical fitting for larger time steps

In the experiment, we have found that the relative height can be well fitted by $P_{domain}/P_{min} \sim x_{0}$ . Thus, the relative height grows as $x_{0}$ increases. In this section, we add numerical simulations with more steps to further demonstrate this behavior.

As illustrated in Fig. S1, we fit the loss probability $P(x = -1)$ at the domain wall, $P(x)$ in bulk, and the relative height $P_{\mathrm{domain}} / P_{\mathrm{min}}$ . The results show that when the edge burst exists [Fig. S1(a,c,e)], both $P(x = -1)$ and $P(x)$ follow power laws: $P(x = -1) \sim x_0^{-\alpha_d}$ and $P(x) \sim (x_0 - x)^{-\alpha_b}$ , with certain $\alpha_d$ and $\alpha_b$ . The fitting for the relative height is $P_{\mathrm{domain}} / P_{\mathrm{min}} \sim x_0^{1.0802}$ , which is close to the $P_{\mathrm{domain}} / P_{\mathrm{min}} \sim x_0$ behavior predicted by theory and supported by our experiment. Notably, the fitting for $\alpha_{d,b}$ are $\alpha_d = 0.4717$ and $\alpha_b = 1.4751$ , so that $\alpha_b - \alpha_d = 1.0034$ , which agrees well with the predicted bulk-edge scaling relation in Ref. [15].

When the edge burst is absent [Fig. S1(b,d,f)], the fitting turns out to be exponential: $P(x = -1) \sim \beta_d^{x_0 - x}$ and $P(x) \sim \beta_b^{x_0 - x}$ , with certain $\beta_b$ and $\beta_d$ that are approximately equal. The relative height $P_{\mathrm{domain}} / P_{\mathrm{min}}$ is almost constant as $x_0$ varies.