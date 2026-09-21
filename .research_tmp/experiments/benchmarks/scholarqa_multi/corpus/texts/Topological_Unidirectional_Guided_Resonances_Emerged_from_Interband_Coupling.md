# Topological Unidirectional Guided Resonances Emerged from Interband Coupling

Xuefan Yin, $^{1}$ Takuya Inoue $^{\textcircled{ID}}$ , $^{1}$ Chao Peng, $^{2,3,*}$ and Susumu Noda $^{1,\dagger}$

$^{1}$ Department of Electronic Science and Engineering, Kyoto University, Kyoto-Daigaku-Katsura, Nishikyo-ku, Kyoto 615-8510, Japan

$^{2}$ State Key Laboratory of Advanced Optical Communication Systems and Networks, School of Electronics, and Frontiers Science Center for Nano-optoelectronics, Peking University, Beijing, 100871, China

$^{3}$ Peng Cheng Laboratory, Shenzhen 518055, China

![](images/71caf4841ee9951a47ec0367a200ce91611da8721d600ccbb43c13f418a74ae1.jpg)

(Received 7 April 2022; accepted 5 December 2022; published 2 February 2023)

Unidirectional guided resonances (UGRs) are optical modes in photonic crystal slabs that radiate toward one side without the need for mirrors on the other. In this Letter, we report a mechanism to realize UGRs by tuning the interband coupling effect originating from up-down symmetry breaking. We theoretically find that UGRs that reside along high-symmetric lines correspond to phase singularities of far-field radiation, depicted by phase winding numbers as a type of topological indices. We investigate the phase dislocation lines in three-dimensional parameter space and elaborate on the interplay between UGRs and non-Hermitian degeneracies accordingly. Our findings reveal the topological nature of UGRs about their generation, evolution, and annihilation in general parameter spaces, thus paving the way to new possibilities of light manipulation.

DOI: 10.1103/PhysRevLett.130.056401

Unidirectional emission is of fundamental interest in research fields including non-Hermitian physics $[1-3]$ and singular optics $[4-6]$ and can benefit many realistic applications such as on-chip lasers $[7-14]$ and energy-efficient grating couplers $[15-20]$ . While most existing methods use mirrors made of metals or photonic-band-gap materials $[21-23]$ , or by utilizing the nonresonant blazing effect $[24,25]$ to forbid the radiation of light toward unnecessary ports, recent findings of unidirectional guided resonances (UGRs) $[26-28]$ revealed that an eigenstate itself can radiate toward only a single side of the photonic crystal (PC) slab without the need for a mirror. From the view of topological photonics $[29-32]$ , the UGRs were connected to polarization singularities $[33-42]$ in momentum space: They are a polarization vortex center merged from paired C points (circular-polarized states) carrying same-signed half-integer topological charges $[43-47]$ on a single side. Such half-charges can originate from splitting an integer charge $[27,48-50]$ carried by a bound state in the continuum (BIC) $[51-53]$ or spawned from the void $[28]$ .

Although topological charges picture established an interpretation upon UGRs, there are still many puzzles and misconceptions about UGR, hindering its practical applications. First, from topological charges' view, the key to realizing UGRs is to create C points. However, although splitting from BICs or voids has been shown as a fairly intuitive example, it is still a challenge to create them in a generic structure. Second, UGR is a consequence of manipulating the evolution of C points in momentum space, but how UGRs themselves evolve is still unknown. At last, the relationship between UGRs and other topological singularities is not clear. Therefore, a systematic method is needed to guide the construction of UGRs, and a more fundamental topological interpretation is necessary to reveal its dynamical evolution.

In this Letter, we first report a mechanism to realize a new class of UGRs without the premise of C points or BICs. We find the interband couplings raised from up-down asymmetry can be utilized as an effective degree of freedom (d.o.f.) to hybridize the orthogonal bands in a PC slab. For a system possessing in-plane mirror symmetry, the interband coupling strength can be adjusted in two-dimensional (2D) parameter space, resulting in a UGR. It shows that the UGRs are more ubiquitous than expected, since band crossing and corresponding interband coupling are quite common in PC slabs regardless of the specific material and geometry.

We further analyze the evolution of UGRs in parameter space. We propose that, with in-plane mirror symmetry, UGRs can be interpreted as points where continuous phase dislocation lines in 3D parameter space pierce a 2D subspace, carrying topological indices defined as the winding numbers in the phase field of the radiation. Through investigating the interplay between UGRs and exceptional points (EPs) [54–60], we find that an EP pair can also possibly exhibit nontrivial topological indices when the UGRs reside at some particular positions with respect to the Fermi arc [59]. The UGRs can transit from one band to the other when they cross the Fermi arc or symmetry-protected degenerate point. Nevertheless, the overall topological indices raised by the UGRs and other singularities remain as a conserved quantity.

![](images/cca0db2bd359f41ce2782402332c1799611d3a4462164376a852d73dad44d591.jpg)  
FIG. 1. UGR raised by interband coupling. (a) Schematic of a PC slab with slab thickness $h / a = 0.555$ and radius $r / a = 0.225$ . (b) Profiles of $\mathrm{TM}_A$ and $\mathrm{TE}_C$ modes. (c) Band structures of original eigenstates $\varphi_{1,2}$ and perturbed eigenstates $\varphi_{+, - }$ , with a UGR at $(k_x = 0.022, k_y = 0)$ in $\varphi_{+}$ . (d) Asymmetric radiation ratio $\eta$ of $\varphi_{+, - }$ . (e) Profiles of $\varphi_{1,2}$ and $\varphi_{+, - }$ at $(k_x = 0.022, k_y = 0)$ . (f),(g) Complex band structures of perturbed eigenstates in parameter space $(\theta, k_x)$ with $k_y = 0$ . An EP is located at $(6.65^\circ, 0.026)$ , and a UGR is located at $(5.2^\circ, 0.022)$ .

Specifically, we start from a freestanding $Si_{3}N_{4}$ slab patterned with square-lattice air holes [Fig. 1(a)], in which the optical modes are characterized by in-plane Bloch wave vector $\boldsymbol{k}_{\parallel} = (k_{x}, k_{y})\beta_{0}$ , where $\beta_{0} = 2\pi/a$ and a is the lattice constant. For vertical sidewalls ( $\theta = 0$ ), although the energy bands of $TM_{A}$ (denoted as $\varphi_{1}$ ) and $TE_{C}$ ( $\varphi_{2}$ ) cross with each other [Fig. 1(c)], the orthogonality forbids any interband coupling between them due to the up-down mirror symmetry: $\varphi_{1}$ is odd and $\varphi_{2}$ is even with respect to z [left panel, Fig. 1(e)].

Next, we isotropically tilt the sidewalls [Fig. 1(a)]. Owing to up-down mirror symmetry breaking, $\varphi_{1}$ and $\varphi_{2}$ couple and give rise to two perturbed eigenstates $\varphi_{+}$ and $\varphi_{-}$ [61]. As shown in Fig. 1(c), the real parts of energy bands cross, while the imaginary parts anticross. As confirmed in right panel in Fig. 1(e), $\varphi_{+, - }$ no longer preserve a definite $z$ parity, and, thus, the radiation becomes up-down asymmetric. When $\theta = 5.2^{\circ}$ , a UGR is found at $(k_x = 0.022, k_y = 0)$ upon the $\varphi_{+}$ band, with an asymmetric radiation ratio of $\eta = \gamma_t / \gamma_b = 70$ dB [Fig. 1(d)], where $\gamma_{t,b}$ are decay rates toward the top and bottom. According to perturbation theory, the perturbed Hamiltonian $\hat{H}$ under the bases of unperturbed eigenstates $\varphi_{1,2}$ is in a form as (see details in Supplemental Material, Sec. 1 [62])

$$
\mathcal {H} = \left[ \begin{array}{c c} \lambda_ {1} & \kappa_ {1 2} \tan \theta \\ \kappa_ {2 1} \tan \theta & \lambda_ {2} \end{array} \right]. \tag {1}
$$

Here, $\lambda_{1,2}$ are the eigenvalues corresponding to $\varphi_{1,2}$ . The antidiagonal terms come from the tilting sidewalls, representing the interband coupling strength. The hybrid eigenstates of $\hat{H}$ can be written as

$$
\varphi_ {+, -} (\boldsymbol {k} _ {\parallel}) = a _ {+, -} (\boldsymbol {k} _ {\parallel}) \varphi_ {1} (\boldsymbol {k} _ {\parallel}) + b _ {+, -} (\boldsymbol {k} _ {\parallel}) \varphi_ {2} (\boldsymbol {k} _ {\parallel}), \tag {2}
$$

where $[a,b]_{+, - }^T$ is the eigenvectors of matrix $\mathcal{H}$ , with eigenvalues derived as

$$
\lambda_ {+, -} = \frac {\lambda_ {1} + \lambda_ {2}}{2} \pm \sqrt {\frac {(\lambda_ {1} - \lambda_ {2}) ^ {2}}{4} + \kappa_ {1 2} \kappa_ {2 1} \tan \theta^ {2}}. \tag {3}
$$

We note that tilting the sidewalls is only an exemplary way to break up-down mirror-symmetry that induces interband coupling. Other vertical geometries such as shallow-etched slab are discussed in Supplemental Material, Sec. 2 [62].

According to Eq. (3), when $(\lambda_{1}-\lambda_{2})^{2}/4+\kappa_{12}\kappa_{21}\tan\theta_{c}^{2}=0$ , the complex eigenvalues $\lambda_{+,,-}$ and eigenstates $\varphi_{+,,-}$ are both degenerate, namely, EPs. Obviously, an EP asks for two d.o.f.s and can emerge in a 2D parameter space. Figures 1(f) and 1(g) illustrate the complex band structures in parameter space $(\theta,k_{x})$ with a fixed $k_{y}=0$ . An isolated EP is found at $(\theta=6.65^{\circ},k_{x}=0.026)$ (marked by black star) with a cut line (gray line) where only real parts of frequencies coincide. The red dot denotes the location of a UGR.

According to Eq. (2), the far-field radiation of perturbed eigenstates $\varphi_{+, -}$ can be derived as

$$
c _ {x y; +, -} ^ {s} = a _ {+, -} c _ {x y; 1} ^ {s} + b _ {+, -} c _ {x y; 2} ^ {s}, \quad s \in \{t, b \}, \tag {4}
$$

where the complex radiation amplitudes of $\varphi_{j}$ are denoted as $c_{x,y;j}^{s}$ with superscripts s referring to the top or bottom side. Obviously, the radiation characteristics of $\varphi_{+,-}$ are modified by the interband coupling. The UGR on $\varphi_{+}$ in Fig. 1 asks for $c_{x;+}^{b}=c_{y;+}^{b}=0$ . Such a condition generally needs at least four d.o.f.s. However, both $\varphi_{1,2}$ in Fig. 1 are x-polarized along the $k_{x}$ axis due to y-mirror symmetry, and, thus, the $E_{y}$ component vanishes naturally. As a result, the UGR condition turns to be $c_{x;+}^{b}=0$ , which needs only two d.o.f.s. In other words, along a high-symmetric line protected by in-plane mirror symmetry, UGRs are robust in 2D parameter space.

The vanishing of radiation modulus is equivalent to a singular value of its phase. In other words, the UGRs

![](images/f7e1ec7367e87b5673b93ee27186bb92f142d6f33d8117c0796ee12f0cba046c.jpg)

<details>
<summary>scatter</summary>

| θ/° | h/a   | k_x    |
|-----|-------|--------|
| 0   | 0.57  | 0.02   |
| 6   | 0.545 | -0.02  |
| 5   | 0.545 | 0.02   |
</details>

![](images/130dd34a348a6a3066559677167947a221d674125fd70cbb43ed4134eb0b7501.jpg)

<details>
<summary>line</summary>

| θ/° | k_x     |
|-----|---------|
| 5.3 | 0.02    |
| 9.3 | -0.02   |
</details>

![](images/691900b2d3b519b9e6a99771b31f3b77f21f5e7c6f5e043d1755a4aba89871cb.jpg)

<details>
<summary>line</summary>

| θ/° | k_x     |
|-----|---------|
| 5.3 | -0.02   |
| 9.3 | 0.02    |
</details>

![](images/d2b1d3e045427f0fff07b85dbc16ad97d3a0323ed9a5f609cafad60cdf8accae.jpg)

<details>
<summary>heatmap</summary>

| θ/° | -0.01 | 0.01 |
| --- | --- | --- |
| 5.4 | 0 | 70 |
| 6.4 | 0 | 0 |
</details>

![](images/98f3eb89b597b6f96c7d1732e063ddaebcbd45aa6937b6f19798af552ae2efdb.jpg)

<details>
<summary>heatmap</summary>

| θ/° | -0.01 | 0.01 |
| --- | --- | --- |
| 5.4 | -0.01 | 0.01 |
| 6.4 | -0.01 | 0.01 |
</details>

FIG. 2. Evolution of UGRs in parameter space. (a) Dislocation line for eigenstate $\varphi_{+}$ in 3D parameter space for the PC slab in Fig. 1. $r / a = 0.225$ , $\Sigma: h = 0.555a$ . (b),(c) [(d),(e)] Phase vector fields and asymmetric ratio in 2D parameter space $(\theta, k_x)$ at $h = 0.569a$ ( $h = 0.57a$ ).

intrinsically correspond to phase singularities upon radiation $c_{x;+}^{b}/c_{0}$ , where $c_{0}$ is the complex amplitude of $\varphi_{+}$ . In 3D parameter space $(\theta, k_{x}, h)$ with fixed $k_{y} = 0$ and r/a = 0.225, these singularities compose a dislocation line, since their codimension is two. Furthermore, we consider a surface $\Sigma$ in such a 3D space by assuming h = const. For different values of h, $\Sigma$ moves and forms a 2D subspace $(\theta, k_{x})$ . As shown in Fig. 2(a), at h = 0.555a, $\Sigma$ is pierced by the dislocation line and two intersection points (red dots) can be found, representing two UGRs in the subspace. When h increases to about 0.5698a, the dislocation line is tangent to the surface, and the two intersection points collide. For h exceeding 0.5698a, there exist no intersection points and the UGR disappears in the subspace.

The above results show that the number of UGRs is not a conserved quantity in the 2D subspace. To better capture the evolution of UGRs, we consider the phase vector field (Re[c $_{x;+,,-}^{b}/c_{0}]$ , Im[c $_{x;+,,-}^{b}/c_{0}]$ ). An example of such a field ( $h = 0.569a$ ) is presented in Fig. 2(b), in which orange and purple vectors belong to $\varphi_{+,,-}$ , respectively. Clearly, there exists a pair of EPs (star) with cut lines (gray line). When crossing the cut lines, $\varphi_{+,,-}$ flip, and so do the corresponding phase vectors. Two UGRs can be identified upon $\varphi_{+}$ : The phase changes $\pm 2\pi$ when encircling around a UGR [Fig. 2(c)]. To quantitatively specify the dislocation strength of phase winding, we consider a closed clockwise loop C around the UGRs (dashed yellow arrow lines). An integral of phase on C defines a gauge-invariant topological index [69] as

$$
S = \frac {1}{2 \pi} \oint_ {C} d \arg \left[ \frac {c _ {x ; +} ^ {b}}{c _ {0}} \right]. \tag {5}
$$

![](images/b50a8805459a47fc789bde512dd202d8448f215b16b897e3133832d67554eaa1.jpg)

<details>
<summary>text_image</summary>

(a)
r
r+2d
a
(b) θ=11.5° • UGR ★ EP
dislocation line for φ+
</details>

![](images/218ebcd6c8e56b9c1932e06f841b28c1a1b0ef5c0b9ab9b85ffd7840e0ce0ca8.jpg)

<details>
<summary>scatter</summary>

| r/a   | k_x    |
|-------|--------|
| 0.17  | 0.025  |
| 0.18  | -0.025 |
| 0.19  | 0.025  |
</details>

![](images/b4c972d8c1766de77652e87a78f4c3e84ce5560b5cf14ecf0a84e72e2f3948bc.jpg)

<details>
<summary>scatter</summary>

| r/a    | k_x     |
| ------ | ------- |
| 0.16   | 0.035   |
| 0.19   | -0.035  |
</details>

![](images/1871039e386d408d5650554ffa65be7beebd109d6820d9facc99e2f755e55f61.jpg)

<details>
<summary>line</summary>

| k_x   | h/a   |
|-------|-------|
| -0.05 | 0.63  |
| 0.16  | 0.54  |
| 0.2   | 0.54  |
</details>

![](images/00117511172d2ae950169660f1907d24ae797d184ace0cd29289add84e791350.jpg)

<details>
<summary>text_image</summary>

(d)
φ_F @d=0
φ_ @d=0
S_F = 0
S_F = 0
</details>

![](images/6212d0f8263550961c4fe64e093cd38fef70a2c2f62b0108489c06199cd8cc99.jpg)

<details>
<summary>text_image</summary>

(f) φ₊@d=50nm φ₋@d=50nm
SF = +1 SF = -1
</details>

FIG. 3. Interplay between UGRs and a Fermi arc. (a) Schematic of a PC slab with triangular air holes. (b) Dislocation line for $\varphi_{+}$ in parameter space at $\theta = 11.5^{\circ}$ . $\Sigma$ : $h = 0.59a$ . (c) Phase and band singularities in $\Sigma$ for $d = 0$ . (e) When tuning $d$ from 0 to $50~\mathrm{nm}$ , the UGR of $S_U = +1$ evolves from $\varphi_{+}$ to $\varphi_{-}$ . (d),(f) Topological indices around the Fermi arc for $d = 0$ and $d = 50~\mathrm{nm}$ , respectively.

Obviously, the two UGRs on $\varphi_{+}$ carry opposite topological indices: The upper one carries $S = +1$ , while the lower one carries $S = -1$ . When $h$ increases, topological indices approach, collide, and eventually annihilate with each other [Figs. 2(d) and 2(e)], obeying a conservation law [63] as $S_{\Sigma}^{+} = -1 + 1 = 0$ . For $\varphi_{-}$ , its total topological index $S_{\Sigma}^{-}$ preserves to be zero, since no UGRs can be found. Therefore, the total topological index in $\Sigma$ is a conserved quantity for both $\varphi_{+, - }$ .

We note that topological index S depicts how the dislocation line in 3D parameter space penetrates through the 2D subspace $\Sigma$ . As shown in Fig. 2(a), the two UGRs in $\Sigma$ are actually on the same dislocation line, so they can evolve smoothly from one to another (black arrows). In other words, there is no distinct difference between them. The value of S describes the topology of dislocation line with respect to $\Sigma$ . For a given $\Sigma$ with distinguishable sides, the UGR of $S = +1(-1)$ corresponds to an intersection point where the dislocation line threads from the back (front) to the front (back) side of $\Sigma$ .

The UGRs and EPs could emerge simultaneously in $\Sigma$ [Fig. 2(b)], since both of them originate from the interband coupling: The UGRs come from radiation hybridization, while the EPs represent the critical coupling. Although there is no direct connection between them [Fig. 2(d)], the UGRs can interplay with EPs in $\Sigma$ . As shown in Fig. 3(a), we consider a PC slab with equilateral triangular air holes that possesses only y-mirror symmetry. Similarly, with tilted sidewalls ( $\theta = 11.5^{\circ}$ ), $TM_{A}$ and $TE_{C}$ modes couple to each other near the band crossing, leading to two perturbed eigenstates $\varphi_{+, -}$ . Without loss of generality, we consider another 3D parameter space ( $r, k_{x}, h$ ) with fixed $k_{y} = 0$ to

ensure y-mirror symmetry and obtain a hairpinlike dislocation line for $\varphi_{+}$ [Fig. 3(b)]. On 2D surface $\Sigma$ at h = 0.59a, two intersection points can be found in $\varphi_{+}$ , corresponding to two UGRs carrying opposite topological indices $S = \pm1$ [Fig. 3(c)]. Meanwhile, a pair of EPs also emerges, connected by the Fermi arc (gray line).

Because of the absence of C2 symmetry, the dislocation line and UGRs are no longer symmetric to $k_{x}=0$ axis, which is different from the EP pair that is protected by the reciprocity. As shown in Fig. 3(e), by tuning the air holes from equilateral triangular (d=0) to isosceles triangular ones (d=50 nm), the UGR carrying S=+1 crosses the Fermi arc and reaches $\varphi_{-}$ , while the UGR with S=-1 stays on $\varphi_{+}$ .

The topological index $S$ obeys the conservation law during the interplay between UGRs and EPs. For the case of $d = 0$ [Fig. 3(d)], two UGRs simultaneously reside on $\varphi_{+}$ but carrying opposite indices $S = \pm 1$ , and the topological indices contributed by Fermi arc are identified as $S = 0$ for both $\varphi_{+, - }$ . Therefore, the overall topological indices for $\varphi_{+, - }$ in $\Sigma$ are both zeros: $S_{\Sigma}^{+} = S_{\Sigma}^{-} = 0$ . For a different case of $d = 50~\mathrm{nm}$ [Fig. 3(f)], only one UGR with $S_U = -1$ ( $S_U = +1$ ) can be found on $\varphi_{+}(\varphi_{-})$ . However, the EP pair also contributes a nonzero index of $S_F = +1$ ( $S_F = -1$ ) in this case. Therefore, the overall topological indices for both $\varphi_{+, - }$ are conserved to be zero: $S_{\Sigma}^{+} = S_{\Sigma}^{-} = S_U + S_F = 0$ . Such conservation of $S = 0$ comes from the fact that, without the interband coupling, neither UGRs nor EPs exist in parameter space. We notice that the nonzero topological index $S$ from Fermi arc has no relationship with the nontrivial Berry phase of EPs but only a consequence of the conservation law. The Berry phase does not contribute to topological index $S$ defined in Eq. (5) (see details in Supplemental Material, Sec. 5 [62]).

UGRs can transit from one band to another, through not only the Fermi arc but also the topologically trivial band degeneracy protected by symmetry. To illustrate this phenomenon, we consider $\mathrm{TM / TE_{CD}}$ modes in a PC slab with circular holes [Fig. 4(a)] and $\theta = 1.72^{\circ}$ . Here, $\mathrm{TM}_{CD}$ are degenerate at the $\Gamma$ point due to $\mathbf{C}_4$ symmetry. Along the $k_{y}$ axis, $\mathrm{TM}_C$ and $\mathrm{TE}_D$ are both $x$ -polarized and could couple to each other, creating a UGR on the $\mathrm{TM}_C$ band [left panel, Fig. 4(b)]. Similarly, $\mathrm{TM}_D$ and $\mathrm{TE}_C$ are $x$ -polarized along the $k_{x}$ axis to support a UGR on the $\mathrm{TM}_D$ band [right panel, Fig. 4(b)]. We investigate the evolution of UGRs in 3D spaces $(r,k_y,h)$ for $\mathrm{TM}_C$ and $(r,k_x,h)$ for $\mathrm{TM}_D$ , and two such spaces coalesce at the $\Gamma$ point [Fig. 4(c)]. Owing to the degeneracy, the dislocation lines for $\mathrm{TM}_C$ (blue) and $\mathrm{TM}_D$ (green) in spaces $(r,k_y / k_x,h)$ attach at the $\Gamma$ point.

We denote a set of surfaces $\Sigma_{j}$ that contain the overlapped subspaces $(r,k_y)$ and $(r,k_x)$ . For $\Sigma_{1}$ of $h = 0.66a$ , only the blue dislocation line threads through and creates two UGRs in $(r,k_y)$ space on $\mathrm{TM}_C$ but none in $(r,k_x)$ space on $\mathrm{TM}_D$ [Fig. 4(f)]. By increasing $h$ to $0.6665a$ , $\Sigma_{2}$ is simultaneously tangent to blue and green dislocation lines, on which two opposite topological indices on $TM_{C}$ collide and annihilate to S = 0 at the $\Gamma$ point [red circle, Fig. 4(e)]. Owing to the degeneracy, the annihilation point (AP) of S = 0 also holds for $TM_{D}$ in space $(r, k_{x})$ . By further increasing h, two UGRs spawn from the AP on $TM_{D}$ , confirmed by the observation at $\Sigma_{3}$ of h = 0.675a, which only the green dislocation line threads through. Therefore, we find the UGRs on $TM_{C}$ finally transit to $TM_{D}$ via band degeneracy at the $\Gamma$ point. During the evolution, the overall topological indices S are conserved to zero.

(a)   
![](images/78a3fee825349ec4e7b422c1d0b918528e7dec19e93574e3cb541f0e9ae8b677.jpg)

(b)   
![](images/f900c9d309e8c101645882d99e9533ce780501cbe7107cdea5a03fe0cea76191.jpg)

![](images/f5235be67efa6e6e65ecacd2c89096a8fb0b40c92fbe09b8479bc7fe817ac1f2.jpg)

(c)   
![](images/7db8f748f99929716695b01e42154c9bee877d6caf5857e59973c3d01cba6a3b.jpg)

(d)   
![](images/430a7f1f92e5425625a846ee1411913b481e14867d681a6b86fcfc1bda6a4bcc.jpg)

<details>
<summary>scatter</summary>

| Method | Symbol | r/a    | kx     |
|--------|--------|--------|--------|
| EP     | ★      | 0.219  | -0.02  |
| EP     | ●      | 0.218  | -0.015 |
| EP     | ★      | 0.217  | -0.01  |
| EP     | ●      | 0.216  | -0.005 |
| AP on both TMc/TMD | ○      | 0.218  | -0.02  |
| AP on both TMc/TMD | ●      | 0.217  | -0.015 |
| AP on both TMc/TMD | ★      | 0.216  | -0.01  |
| AP on both TMc/TMD | ●      | 0.215  | -0.005 |
| AP on both TMc/TMD | ★      | 0.214  | -0.015 |
| AP on both TMc/TMD | ●      | 0.213  | -0.01  |
| AP on both TMc/TMD | ★      | 0.212  | -0.005 |
| AP on both TMc/TMD | ●      | 0.211  | -0.01  |
| AP on both TMc/TMD | ★      | 0.210  | -0.015 |
| AP on both TMc/TMD | ●      | 0.209  | -0.01  |
| AP on both TMc/TMD | ★      | 0.208  | -0.005 |
| AP on both TMc/TMD | ●      | 0.207  | -0.015 |
| AP on both TMc/TMD | ★      | 0.206  | -0.01  |
| AP on both TMc/TMD | ●      | 0.205  | -0.005 |
| AP on both TMc/TMD | ★      | 0.204  | -0.015 |
| AP on both TMc/TMD | ●      | 0.203  | -0.01  |
| AP on both TMc/TMD | ★      | 0.202  | -0.005 |
| AP on both TMc/TMD | ●      | 0.201  | -0.015 |
| AP on both TMc/TMD | ★      | 0.200  | -0.01  |
| AP on both TMc/TMD | ●      | 0.199  | -0.005 |
| AP on both TMc/TMD | ★      | 0.198  | -0.015 |
| AP on both TMc/TMD | ●      | 0.197  | -0.01  |
| AP on both TMc/TMD | ★      | 0.196  | -0.005 |
| AP on both TMc/TMD | ●      | 0.195  | -0.015 |
| AP on both TMc/TMD | ★      | 0.194  | -0.01  |
| AP on both TMc/TMD | ●      | 0.193  | -0.005 |
| AP on both TMc/TMD | ★      | 0.192  | -0.015 |
| AP on both TMc/TMD | ●      | 0.191  | -0.01  |
| AP on both TMc/TMD | ★      | 0.190  | -0.005 |
| AP on both TMc/TMD | ●      | 0.189  | -0.015 |
| AP on both TMc/TMD | ★      | 0.188  | -0.01  |
| AP on both TMc/TMD | ●      | 0.187  | -0.005 |
| AP on both TMc/TMD | ★      | 0.186  | -0.015 |
| AP on both TMc/TMD | ●      | 0.185  | -0.01  |
| AP on both TMc/TMD | ★      | 0.184  | -0.005 |
| AP on both TMc/TMD | ●      | 0.183  | -0.015 |
| AP on both TMc/TMD | ★      | 0.182  | -0.01  |
| AP on both TMc/TMD | ●      | 0.181  | -0.005 |
| AP on both TMc/TMD | ★      | 0.18   | -0.015 |
| AP on both TMc/TMD | ●      | 0.179  | -0.01  |
| AP on both TMc/TMD | ★      | 0.178  | -0.005 |
| AP on both TMc/TMD | ●      | 0.177  | -0.015 |
| AP on both TMc/TMD | ★      | 0.176  | -0.01  |
| AP on both TMc/TMD | ●      | 0.175  | -0.005 |
| AP on both TMc/TMD | ★      | 0.174  | -0.015 |
| AP on both TMc/TMD | ●      | 0.173  | -0.01  |
| AP on both TMc/TMD | ★      | 0.172  | -0.005 |
| AP on both TMc/TMD | ●      | 0.171  | -0.015 |
| AP on both TMc/TMD | ★      | 0.17   | -0.01  |
| AP on both TMc/TMD | ●      | 0.169  | -0.005 |
| AP on both TMc/TMD | ★      | 0.168  | -0.015 |
| AP on both TMc/TMD | ●      | 0.167  | -0.01  |
| AP on both TMc/TMD | ★      | 0.166  | -0.005 |
| AP on both TMc/TMD | ●      | 0.165  | --<ecel><ecel><ecel><nl>
</details>

FIG. 4. Interplay between UGRs and topologically trivial band degeneracy. (a) Schematic of a PC slab with circular air holes. (b) Left: band structures of $h = 0.66a$ and $r = 0.218a$ with a UGR on the $\mathrm{TM}_C$ mode at $k_y = 0.01$ . Right: band structures of $h = 0.668a$ and $r = 0.216a$ with a UGR on the $\mathrm{TM}_D$ mode at $k_x = 0.005$ . (c) Dislocation lines of $\mathrm{TM}_{CD}$ modes in overlapped parameter spaces $(r, k_y, h)$ and $(r, k_x, h)$ . (d)-(f) UGRs and their annihilation point (AP) on $\mathrm{TM}_{CD}$ modes in 2D parameter space, corresponding to surfaces $\Sigma_{3,2,1}$ .

For x-polarized modes in this Letter, we focus on the $c_{x}$ component of radiation [Eq. (5)]. For y-polarized modes or $\Gamma-M$ -mirror symmetric systems, we need to similarly investigate $c_{y}$ or $c_{x} \pm c_{y}$ components, respectively (see examples in Supplemental Material, Sec. 2 [62]). If mirror symmetry is absent, robust UGRs generally require four d.o.f.s, and a high-dimensional parameter space should be considered. We note that the topological interpretation of UGRs in the view of phase winding S is fully consistent with the previously reported polarization topological

charge picture. As an addition, the differences between UGRs proposed here and the reported one in Ref. [27] are also elaborated in Supplemental Material, Sec. 7 [62].

In conclusion, we propose a systematic method to realize UGRs in a PC slab by coupling two unperturbed eigenstates from up-down mirror symmetry breaking. We find that, along in-plane mirror-symmetric lines, UGRs correspond to robust phase singularities of radiation in parameter space, originated from the piercing behavior of the phase dislocation lines in 3D parameter space. We investigate the interplays between UGRs and non-Hermitian band degeneracies and find the UGRs could transit from one band to another, by crossing the Fermi arc connected by paired EPs or topologically trivial band degeneracy protected by in-plane symmetry. During the evolution, the overall topological index carried by UGRs and other singularities together obeys a conservation law. Our findings shed light upon practical application of UGRs, revealing new possibilities for creating and utilizing phase singularity in parameter space, providing a vivid picture for manipulating the optical radiation [70] in various applications.

The authors are grateful to Professor B. S. Song, Dr. J. Gelleta, Dr. Z. Zhang, and Dr. Y. Hu for helpful discussions. This work was supported from National Key Research and Development Program of China (2022YFA1404804, 2020YFB1806405), National Natural Science Foundation of China (61922004 and 62135001), Grant-in-Aid for Scientific Research (22H04915), and Major Key Project of PCL (PCL2021A14 and PCL2021A104). X. Y. was supported by Research Fellowships of the Japan Society for Promotion of Science (21F20356).

\*pengchao@pku.edu.cn   
$^{\dagger}$ snoda@qoe.kuee.kyoto-u.ac.jp   
[1] E. J. Bergholtz, J. C. Budich, and F. K. Kunst, Exceptional topology of non-hermitian systems, Rev. Mod. Phys. 93, 015005 (2021).   
[2] D. Leykam, K. Y. Bliokh, C. Huang, Y. D. Chong, and F. Nori, Edge Modes, Degeneracies, and Topological Numbers in Non-Hermitian Systems, Phys. Rev. Lett. 118, 040401 (2017).   
[3] H. Shen, B. Zhen, and L. Fu, Topological Band Theory for Non-Hermitian Hamiltonians, Phys. Rev. Lett. 120, 146402 (2018).   
[4] M. R. Dennis, K. O'Holleran, and M. J. Padgett, Singular optics: Optical vortices and polarization singularities, Prog. Opt. 53, 293 (2009).   
[5] M. Soskin, S. V. Boriskina, Y. Chong, M. R. Dennis, and A. Desyatnikov, Singular optics and topological photonics, J. Opt. 19, 010401 (2016).   
[6] G. J. Gbur, Singular Optics (CRC Press, Boca Raton, FL, 2016).

[7] W. Streifer, D. Scifres, and R. Burnham, Analysis of grating-coupled radiation in gaas: Gaalas lasers and waveguides-i, IEEE J. Quantum Electron. 12, 422 (1976).   
[8] M. Meier, A. Mekis, A. Dodabalapur, A. Timko, R. Slusher, J. Joannopoulos, and O. Nalamasu, Laser action from two-dimensional distributed feedback in photonic crystals, Appl. Phys. Lett. 74, 7 (1999).   
[9] M. Imada, S. Noda, A. Chutinan, T. Tokuda, M. Murata, and G. Sasaki, Coherent two-dimensional lasing action in surface-emitting laser with triangular-lattice photonic crystal structure, Appl. Phys. Lett. 75, 316 (1999).   
[10] H. Matsubara, S. Yoshimoto, H. Saito, Y. Jianglin, Y. Tanaka, and S. Noda, Gan photonic-crystal surface-emitting laser at blue-violet wavelengths, Science 319, 445 (2008).   
[11] K. Hirose, Y. Liang, Y. Kurosaka, A. Watanabe, T. Sugiyama, and S. Noda, Watt-class high-power, high-beam-quality photonic-crystal lasers, Nat. Photonics 8, 406 (2014).   
[12] M. Yoshida, M. De Zoysa, K. Ishizaki, Y. Tanaka, M. Kawasaki, R. Hatsuda, B. Song, J. Gelleta, and S. Noda, Double-lattice photonic-crystal resonators enabling high-brightness semiconductor lasers with symmetric narrow-divergence beams, Nat. Mater. 18, 121 (2019).   
[13] R. Sakata, K. Ishizaki, M. De Zoysa, S. Fukuhara, T. Inoue, Y. Tanaka, K. Iwata, R. Hatsuda, M. Yoshida, J. Gelleta et al., Dually modulated photonic crystals enabling high-power high-beam-quality two-dimensional beam scanning lasers, Nat. Commun. 11, 3487 (2020).   
[14] R. Morita, T. Inoue, M. De Zoysa, K. Ishizaki, and S. Noda, Photonic-crystal lasers with two-dimensionally arranged gain and loss sections for high-peak-power short-pulse operation, Nat. Photonics 15, 311 (2021).   
[15] B. Wang, J. Jiang, and G. P. Nordin, Compact slanted grating couplers, Opt. Express 12, 3313 (2004).   
[16] G. Roelkens, D. V. Thourhout, and R. Baets, High efficiency grating coupler between silicon-on-insulator waveguides and perfectly vertical optical fibers, Opt. Lett. 32, 1495 (2007).   
[17] A. Mekis, S. Gloeckner, G. Masini, A. Narasimha, T. Pinguet, S. Sahni, and P. De Dobbelaere, A grating-coupler-enabled cmos photonics platform, IEEE J. Sel. Top. Quantum Electron. 17, 597 (2011).   
[18] M. T. Wade, F. Pavanello, R. Kumar, C. M. Gentry, A. Atabaki, R. Ram, V. Stojanović, and M. A. Popović, 75% efficient wide bandwidth grating couplers in a 45 nm microelectronics cmos process, in Proceedings of the 2015 IEEE Optical Interconnects Conference (OI) (IEEE, 2015), pp. 46–47, 10.1109/OIC.2015.7115679.   
[19] C. Sun, M. T. Wade, Y. Lee, J. S. Orcutt, L. Alloatti, M. S. Georgas, A. S. Waterman, J. M. Shainline, R. R. Avizienis, S. Lin et al., Single-chip microprocessor that communicates directly using light, Nature (London) 528, 534 (2015).   
[20] A. Michaels and E. Yablonovitch, Inverse design of near unity efficiency perfectly vertical grating couplers, Opt. Express 26, 4766 (2018).   
[21] R. L. Roncone, L. Li, K. A. Bates, J. J. Burke, L. Weisenbach, and B. J. J. Zelinski, Design and fabrication of a single leakage-channel grating coupler, Appl. Opt. 32, 4522 (1993).

[22] D. Taillaert, P. Bienstman, and R. Baets, Compact efficient broadband grating coupler for silicon-on-insulator waveguides, Opt. Lett. 29, 2749 (2004).   
[23] L. Zhu, W. Yang, and C. Chang-Hasnain, Very high efficiency optical coupler for silicon nanophotonic waveguide and single mode optical fiber, Opt. Express 25, 18462 (2017).   
[24] W. Streifer, R. Burnham, and D. Scifres, Analysis of grating-coupled radiation in gaas: Gaalas lasers and waveguides-ii: Blazing effects, IEEE J. Quantum Electron. 12, 494 (1976).   
[25] J. M. Miller, N. de Beaucoudrey, P. Chavel, J. Turunen, and E. Cambril, Design and fabrication of binary slanted surface-relief gratings for a planar optical interconnection, Appl. Opt. 36, 5717 (1997).   
[26] H. Zhou, B. Zhen, C. W. Hsu, O. D. Miller, S. G. Johnson, J. D. Joannopoulos, and M. Soljačić, Perfect single-sided radiation and absorption without mirrors, Optica 3, 1079 (2016).   
[27] X. Yin, J. Jin, M. Soljačić, C. Peng, and B. Zhen, Observation of topologically enabled unidirectional guided resonances, Nature (London) 580, 467 (2020).   
[28] Y. Zeng, G. Hu, K. Liu, Z. Tang, and C.-W. Qiu, Dynamics of Topological Polarization Singularity in Momentum Space, Phys. Rev. Lett. 127, 176101 (2021).   
[29] L. Lu, J. D. Joannopoulos, and M. Soljačić, Topological photonics, Nat. Photonics 8, 821 (2014).   
[30] A. B. Khanikaev and G. Shvets, Two-dimensional topological photonics, Nat. Photonics 11, 763 (2017).   
[31] T. Ozawa, H. M. Price, A. Amo, N. Goldman, M. Hafezi, L. Lu, M. C. Rechtsman, D. Schuster, J. Simon, O. Zilberberg et al., Topological photonics, Rev. Mod. Phys. 91, 015006 (2019).   
[32] H. Wang, S. K. Gupta, B. Xie, and M. Lu, Topological photonic crystals: A review, Front. Optoelectron. 13, 50 (2020).   
[33] J. F. Nye, Lines of circular polarization in electromagnetic wave fields, Proc. R. Soc. Lond. A 389, 279 (1983).   
[34] F. Flossmann, K. O'Holleran, M. R. Dennis, and M. J. Padgett, Polarization Singularities in 2d and 3d Speckle Fields, Phys. Rev. Lett. 100, 203902 (2008).   
[35] T. Bauer, P. Banzer, E. Karimi, S. Orlov, A. Rubano, L. Marrucci, E. Santamato, R. W. Boyd, and G. Leuchs, Observation of optical polarization möbius strips, Science 347, 964 (2015).   
[36] T. Fösel, V. Peano, and F. Marquardt, L lines, c points and chern numbers: Understanding band structure topology using polarization fields, New J. Phys. 19, 115013 (2017).   
[37] K. Y. Bliokh, M. A. Alonso, and M. R. Dennis, Geometric phases in 2d and 3d polarized fields: Geometrical, dynamical, and topological aspects, Rep. Prog. Phys. 82, 122401 (2019).   
[38] W. Chen, Y. Chen, and W. Liu, Singularities and Poincaré Indices of Electromagnetic Multipoles, Phys. Rev. Lett. 122, 153907 (2019).   
[39] Z. Che, Y. Zhang, W. Liu, M. Zhao, J. Wang, W. Zhang, F. Guan, X. Liu, W. Liu, L. Shi et al., Polarization Singularities of Photonic Quasicrystals in Momentum Space, Phys. Rev. Lett. 127, 043901 (2021).

[40] J. Ni, C. Huang, L.-M. Zhou, M. Gu, Q. Song, Y. Kivshar, and C.-W. Qiu, Multidimensional phase singularities in nanophotonics, Science 374, eabj0039 (2021).   
[41] Q. Wang, C.-H. Tu, Y.-N. Li, and H.-T. Wang, Polarization singularities: Progress, fundamental physics, and prospects, APL Photonics 6, 040901 (2021).   
[42] W. Liu, W. Liu, L. Shi, and Y. Kivshar, Topological polarization singularities in metaphotonics, Nanophotonics 10, 1469 (2021).   
[43] B. Zhen, C. W. Hsu, L. Lu, A. D. Stone, and M. Soljačić, Topological Nature of Optical Bound States in the Continuum, Phys. Rev. Lett. 113, 257401 (2014).   
[44] E. N. Bulgakov and D. N. Maksimov, Bound states in the continuum and polarization singularities in periodic arrays of dielectric rods, Phys. Rev. A 96, 063833 (2017).   
[45] Y. Zhang, A. Chen, W. Liu, C. W. Hsu, B. Wang, F. Guan, X. Liu, L. Shi, L. Lu, and J. Zi, Observation of Polarization Vortices in Momentum Space, Phys. Rev. Lett. 120, 186103 (2018).   
[46] H. M. Doeleman, F. Monticone, W. den Hollander, A. Alu, and A. F. Koenderink, Experimental observation of a polarization vortex at an optical bound state in the continuum, Nat. Photonics 12, 397 (2018).   
[47] W. Chen, Q. Yang, Y. Chen, and W. Liu, Evolution and global charge conservation for polarization singularities emerging from non-hermitian degeneracies, Proc. Natl. Acad. Sci. U.S.A. 118, e2019578118 (2021).   
[48] W. Liu, B. Wang, Y. Zhang, J. Wang, M. Zhao, F. Guan, X. Liu, L. Shi, and J. Zi, Circularly Polarized States Spawning from Bound States in the Continuum, Phys. Rev. Lett. 123, 116104 (2019).   
[49] W. Ye, Y. Gao, and J. Liu, Singular Points of Polarizations in the Momentum Space of Photonic Crystal Slabs, Phys. Rev. Lett. 124, 153904 (2020).   
[50] T. Yoda and M. Notomi, Generation and Annihilation of Topologically Protected Bound States in the Continuum and Circularly Polarized States by Symmetry Breaking, Phys. Rev. Lett. 125, 053902 (2020).   
[51] J. von Neuman and E. Wigner, Über merkwürdige diskrete Eigenwerte. Uber das Verhalten von Eigenwerten bei adiabatischen Prozessen, Phys. Z. 30, 467 (1929), https://ui.adsabs.harvard.edu/abs/1929PhyZ...30..467V/exportcitation.   
[52] D. C. Marinica, A. G. Borisov, and S. V. Shabanov, Bound States in the Continuum in Photonics, Phys. Rev. Lett. 100, 183902 (2008).   
[53] C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, and M. Soljačić, Bound states in the continuum, Nat. Rev. Mater. 1, 16048 (2016).   
[54] M. V. Berry, Physics of nonhermitian degeneracies, Czech. J. Phys. 54, 1039 (2004).   
[55] W. Heiss, The physics of exceptional points, J. Phys. A 45, 444016 (2012).   
[56] C. Dembowski, H.-D. Gräf, H. L. Harney, A. Heine, W. D. Heiss, H. Rehfeld, and A. Richter, Experimental Observation of the Topological Structure of Exceptional Points, Phys. Rev. Lett. 86, 787 (2001).   
[57] B. Zhen, C. W. Hsu, Y. Igarashi, L. Lu, I. Kaminer, A. Pick, S.-L. Chua, J. D. Joannopoulos, and M. Soljačić, Spawning rings of exceptional points out of Dirac cones, Nature (London) 525, 354 (2015).

[58] J. Doppler, A. A. Mailybaev, J. Böhm, U. Kuhl, A. Girschik, F. Libisch, T. J. Milburn, P. Rabl, N. Moiseyev, and S. Rotter, Dynamically encircling an exceptional point for asymmetric mode switching, Nature (London) 537, 76 (2016).   
[59] H. Zhou, C. Peng, Y. Yoon, C. W. Hsu, K. A. Nelson, L. Fu, J. D. Joannopoulos, M. Soljačić, and B. Zhen, Observation of bulk fermi arc and polarization half charge from paired exceptional points, Science 359, 1009 (2018).   
[60] M.-A. Miri and A. Alu, Exceptional points in optics and photonics, Science 363 (2019).   
[61] Y. Tanaka, T. Asano, Y. Akahane, B.-S. Song, and S. Noda, Theoretical investigation of a two-dimensional photonic crystal slab with truncated cone air holes, Appl. Phys. Lett. 82, 1661 (2003).   
[62] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevLett.130.056401 for analytical derivation, discussions for EPs and crossing types in parameter space, examples of UGRs in other geometries and dimensions, and topological interpretation of UGRs from perspective of topological charges, which includes Refs. [27,28,63–68].   
[63] J. F. Nye and M. V. Berry, Dislocations in wave trains, Proc. R. Soc. Lond. A 336, 165 (1974).

[64] Y. Liang, C. Peng, K. Sakai, S. Iwahashi, and S. Noda, Three-dimensional coupled-wave model for square-lattice photonic crystal lasers with transverse electric polarization: A general approach, Phys. Rev. B 84, 195119 (2011).   
[65] C. Peng, Y. Liang, K. Sakai, S. Iwahashi, and S. Noda, Coupled-wave analysis for photonic-crystal surface-emitting lasers on air holes with arbitrary sidewalls, Opt. Express 19, 24672 (2011).   
[66] F. Keck, H. Korsch, and S. Mossmann, Unfolding a diabolic point: A generalized crossing scenario, J. Phys. A 36, 2125 (2003).   
[67] X. Yin, Y. Liang, L. Ni, Z. Wang, C. Peng, and Z. Li, Analytical study of mode degeneracy in non-hermitian photonic crystals with tm-like polarization, Phys. Rev. B 96, 075111 (2017).   
[68] H. Friedrich and D. Wintgen, Interfering resonances and bound states in the continuum, Phys. Rev. A 32, 3231 (1985).   
[69] M. V. Berry, Much ado about nothing: Optical distortion lines (phase singularities, zeros, and vortices), in Proceedings of the International Conference on Singular Optics (SPIE, 1998), Vol. 3487, pp. 1–5, 10.1117/12.317693.   
[70] X. Yin and C. Peng, Manipulating light radiation from a topological perspective, Photonics Res. 8, B25 (2020).