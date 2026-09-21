# optica

# Cavity-enhanced second-harmonic generation via nonlinear-overlap optimization

ZIN LIN, $^{1}$ XIANGDONG LIANG, $^{2}$ MARKO LONČAR, $^{1}$ STEVEN G. JOHNSON, $^{2}$ AND ALEJANDRO W. RODRIGUEZ $^{3,*}$

$^{1}$ John A. Paulson School of Engineering and Applied Sciences, Harvard University, Cambridge, Massachusetts 02138, USA

$^{2}$ Department of Mathematics, Massachusetts Institute of Technology, Cambridge, Massachusetts 02139, USA

$^{3}$ Department of Electrical Engineering, Princeton University, Princeton, New Jersey 08544, USA

*Corresponding author: arod@princeton.edu

Received 13 November 2015; revised 26 January 2016; accepted 27 January 2016 (Doc. ID 253839); published 1 March 2016

We describe a novel approach based on topology optimization that enables automatic discovery of wavelength-scale photonic structures for achieving high-efficiency second-harmonic generation (SHG). A key distinction from previous formulation and designs that seek to maximize Purcell factors at individual frequencies is that our method aims to not only achieve frequency matching (across an entire octave) and large radiative lifetimes, but also optimizes the equally important nonlinear-coupling figure of merit $\overline{\beta}$ , involving a complicated spatial overlap-integral between modes.

We apply this method to the particular problem of optimizing micropost and grating-slab cavities (one-dimensional multilayered structures) and demonstrate that a variety of material platforms can support modes with the requisite frequencies, large lifetimes $Q > 10^{4}$ , small modal volumes $\sim (\lambda / n)^{3}$ , and extremely large $\overline{\beta} \gtrsim 10^{-2}$ , leading to orders of magnitude enhancements in SHG efficiency compared to state-of-the-art photonic designs.

Such giant $\overline{\beta}$ alleviate the need for ultranarrow linewidths and thus pave the way for wavelength-scale SHG devices with faster operating timescales and higher tolerance to fabrication imperfections. © 2016 Optical Society of America

OCIS codes: (190.0190) Nonlinear optics; (050.1755) Computational electromagnetic methods.

http://dx.doi.org/10.1364/OPTICA.3.000233

# 1. INTRODUCTION

Nonlinear optical processes mediated by second-order $(\chi^{(2)})$ nonlinearities play a crucial role in many photonic applications, including ultrashort-pulse shaping [1,2], spectroscopy [3], generating novel states of light [4-6], and quantum information processing [7-9]. Because nonlinearities are generally weak in bulk media, a well-known approach for lowering the power requirements of devices is to enhance nonlinear interactions by employing optical resonators that confine light for long times (larger quality factors $Q$ ) in small volumes $V$ [10-19].

Microcavity resonators designed for on-chip, infrared applications offer some of the smallest confinement factors available, but their implementation in practical devices has been largely hampered by the difficult task of identifying wavelength-scale $(V \sim \lambda^3)$ structures supporting long-lived, resonant modes at widely separated wavelengths and satisfying rigid frequency-matching and mode-overlap constraints [15,20].

In this article, we extend a recently proposed formulation for the scalable topology optimization of microcavities, where every pixel of the geometry is a degree of freedom, to the problem of designing wavelength-scale photonic structures for second-harmonic generation (SHG). We apply this approach to obtain novel micropost and grating microcavity designs supporting strongly coupled fundamental and harmonic modes at infrared and visible wavelengths with relatively large lifetimes $Q_{1}$ , $Q_{2} > 10^{4}$ . In contrast to recently proposed designs based on known, linear cavity structures hand

tailored to maximize the Purcell factors or minimize mode volumes of individual resonances, e.g., ring resonators [17,21-23] and nanobeam cavities [19,24], our designs ensure frequency matching and small confinement factors, while simultaneously maximizing the SHG enhancement factor $Q_{1}^{2}Q_{2}|\hat{\beta}|^{2}$ to yield orders of magnitude improvements in the nonlinear coupling $\hat{\beta}$ described by Eq. (3) and determined by a special overlap integral between the modes.

These particular optimizations of multilayer stacks illustrate the benefits of our formalism in an approachable and experimentally feasible setting, laying the framework for future topology optimization of 2D/3D slab structures that are sure to yield even further improvements. In what follows, although we will primarily focus on the problem of SHG as a concrete demonstration of our technique, the proposed formulation can be extended to many other problems of interest.

For instance, in the area of quantum science and technology, where quantum information carried by photons needs to be communicated over long distance, our technique can be used to realize efficient quantum frequency conversion over the widest range, including that from visible to telecommunication wavelengths [25]. In fact, any nonlinear frequency conversion problem that can stand to benefit from a chip-scale nanophotonic platform can potentially benefit from this approach.

Our work constitutes a new approach to nonlinear photonic design based on specially tailored aperiodic structures rather than conventional hand designs.

Research Article

Vol. 3, No. 3 / March 2016 / Optica

233

2334-2536/16/030233-06 Journal © 2016 Optical Society of America

Corrected 7 April 2016

Most experimental demonstrations of SHG in chip-based photonic systems [16,17,23,26-29] operate in the so-called small-signal regime, where the lack of pump depletion leads to the well-known quadratic scaling of harmonic output with incident power [30]. In situations involving all-resonant conversion, where confinement and long interaction times lead to strong nonlinearities and non-negligible downconversion [12,20], the maximum achievable conversion efficiency $(\eta \equiv \frac{P_{\mathrm{out}}^{\mathrm{in}}}{P_{\mathrm{in}}^{\mathrm{out}}})$

$$
\eta^ {\max } = \left(1 - \frac {Q _ {1}}{Q _ {1} ^ {\operatorname {r a d}}}\right) \left(1 - \frac {Q _ {2}}{Q _ {2} ^ {\operatorname {r a d}}}\right), \tag {1}
$$

occurs at a critical input power [20],

$$
P _ {1} ^ {\text {c r i t}} = \frac {2 \omega_ {1} \varepsilon_ {0} \lambda_ {1} ^ {3}}{\left(\chi_ {\text {e f f}} ^ {(2)}\right) ^ {2} | \bar {\beta} | ^ {2} Q _ {1} ^ {2} Q _ {2}} \left(1 - \frac {Q _ {1}}{Q _ {1} ^ {\text {r a d}}}\right) ^ {- 1}, \tag {2}
$$

where $\chi_{\mathrm{eff}}^{(2)}$ is the effective nonlinear susceptibility of the medium (Supplement 1), and $Q = (\frac{1}{Q^{\mathrm{rad}}} +\frac{1}{Q^c})^{-1}$ is the dimensionless quality factor (ignoring material absorption) incorporating radiative decay $\frac{1}{Q^{\mathrm{rad}}}$ and coupling to an input/output channel $\frac{1}{Q^c}$ . The dimensionless coupling coefficient $\tilde{\beta}$ is given by a complicated, spatial-overlap integral involving the fundamental and harmonic modes (see Supplement 1 for details):

$$
\bar {\beta} = \frac {\int \mathrm {d} \mathbf {r} \bar {\varepsilon} (\mathbf {r}) E _ {2} ^ {*} E _ {1} ^ {2}}{\left(\int \mathrm {d} \mathbf {r} \varepsilon_ {1} | \mathbf {E} _ {1} | ^ {2}\right) \left(\sqrt {\int \mathrm {d} \mathbf {r} \varepsilon_ {2} | \mathbf {E} _ {2} | ^ {2}}\right)} \sqrt {\lambda_ {1} ^ {3}}, \tag {3}
$$

where $\bar{\varepsilon} (\mathbf{r}) = 1$ inside the nonlinear medium and zero elsewhere. Based on the above expressions, one can define the following dimensionless figures of merit:

$$
\mathrm {F O M} _ {1} = Q _ {1} ^ {2} Q _ {2} | \bar {\beta} | ^ {2} \left(1 - \frac {Q _ {1}}{Q _ {1} ^ {\text {r a d}}}\right) ^ {2} \left(1 - \frac {Q _ {2}}{Q _ {2} ^ {\text {r a d}}}\right), \tag {4}
$$

$$
\mathrm {F O M} _ {2} = \left(Q _ {1} ^ {\mathrm {r a d}}\right) ^ {2} Q _ {2} ^ {\mathrm {r a d}} | \tilde {\beta} | ^ {2}, \tag {5}
$$

where $\math
rm{FOM}_1$ represents the efficiency per power, often quoted in the so-called undepleted regime of low-power conversion [30], and $\mathrm{FOM}_2$ represents an intrinsic upper bound that depends only on the uncoupled cavity parameters, e.g., the intrinsic radiative lifetimes $Q^{\mathrm{rad}}$ .

In particular, given a set of radiation loss rates, $\mathrm{FOM}_1$ is maximized when the modes are critically coupled, $Q = \frac{Q^{\mathrm{rad}}}{2}$ , in which case $\mathrm{FOM}_1^{\mathrm{max}} = \mathrm{FOM}_2 / 64$ , whereas the absolute maximum occurs in the absence of radiative losses, $Q^{\mathrm{rad}} \rightarrow \infty$ , or equivalently, when $\mathrm{FOM}_2$ is maximized. From either FOM, it is clear that, apart from frequency matching and lifetime engineering, the design of optimal SHG cavities rests on achieving a large nonlinear coupling $\bar{\beta}$ .

# 2. OPTIMIZATION FORMULATION

Optimization techniques have been regularly employed by the photonic device community, primarily for fine-tuning the characteristics of a predetermined geometry; the majority of these techniques involve probabilistic Monte Carlo algorithms, such as particle swarms, simulated annealing, and genetic algorithms [31-33]. While some of these gradient-free methods have been used to uncover a few unexpected results out of a limited number of degrees of freedom (DOFs) [34], gradient-based topology optimization methods efficiently handle a far larger design space, typically considering every pixel or voxel as a DOF in an extensive 2D or 3D computational domain, giving rise to novel topologies

and geometries that might have been difficult to conceive from conventional intuition alone. The early applications of topology optimization were primarily focused on mechanical problems [35] and only recently were expanded to encompass photonic systems, though largely limited to linear devices [34,36-42].

Recent work [37] considered topology optimization of the cavity Purcell factor by exploiting the concept of local density of states (LDOS). In particular, the equivalence between the LDOS and power radiated by a point dipole can be exploited to reduce Purcell-factor maximization problems to a series of small scattering calculations.

Defining the objective function $\max_{\bar{\varepsilon}}f(\bar{\varepsilon} (\mathbf{r});\omega) =$ -Re[ $\int \mathrm{d}\mathbf{r}\mathbf{J}^{*}\cdot \mathbf{E}]$ , it follows that $\mathbf{E}$ can be found by solving the frequency domain Maxwell's equations $\mathcal{M}\mathbf{E} = i\omega \mathbf{J}$ , where $\mathcal{M}$ is the Maxwell operator (Supplement 1) and $\mathbf{J} = \delta (\mathbf{r} - \mathbf{r}_0)\hat{\mathbf{e}}_j$ The maximization is then performed over a discretized domain defined by the normalized dielectric function, $\{\bar{\varepsilon}_{\alpha} = \bar{\varepsilon}

(\mathbf{r}_{\alpha}),\alpha \leftrightarrow$ $(i\Delta x,j\Delta y,k\Delta z)\}$ .

A key realization in [37] is that, instead of maximizing the LDOS at a single discrete frequency $\omega$ , a better-posed problem is that of maximizing the frequency-averaged $f$ in the vicinity of $\omega$ , denoted by $\langle f\rangle = \int \mathrm{d}\omega^{\prime}\mathcal{W}(\omega^{\prime};\omega ,\Gamma)f(\omega^{\prime})$ , where $\mathcal{W}$ is some weight function defined over a specified bandwidth $\Gamma$ Using contour integration techniques, the frequency integral can be conveniently replaced by a single evaluation of $f$ at a complex frequency $\omega +i\Gamma$ [37].

For a fixed $\Gamma$ , the frequency average effectively forces the algorithm to favor minimizing $V$ over maximizing $Q$ ; the latter can be enhanced over the course of the optimization by gradually winding down $\Gamma$ [37]. A major merit of this formulation is that it features a mathematically well-posed objective as opposed to a direct maximization of the cavity Purcell factor $\frac{Q}{V}$ allowing rapid convergence into extremal solutions.

A simple extension of the optimization problem from single-mode to multimode cavities maximizes the minimum of a collection of LDOS at different frequencies, while the objective becomes: $\max_{\tilde{\epsilon}_a}\min[\mathrm{LDOS}(\omega_1),\mathrm{LDOS}(2\omega_1)]$ , which requires solving two separate scattering problems, $\mathcal{M}_1\mathbf{E}_1 = \mathbf{J}_1$ and $\mathcal{M}_2\mathbf{E}_2 = \mathbf{J}_2$ , for the two distinct point sources $\mathbf{J}_1$ , $\mathbf{J}_2$ at $\omega_1$ and $\omega_2 = 2\omega_1$ , respectively.

However, as discussed before, rather than maximizing the Purcell factor at individual resonances, the key to realizing optimal SHG is to maximize the overlap integral $\tilde{\beta}$ between $\mathbf{E}_1$ and $\mathbf{E}_2$ , described by Eq. (3). Here, we suggest an elegant way to incorporate $\tilde{\beta}$ by coupling the two scattering problems.

We consider not a point dipole but an extended source $\mathbf{J}_2\sim \mathbf{E}_1^2$ at $\omega_2$ and optimize a single combined radiated power $f = -\mathrm{Re}\left[\int \mathrm{d}\mathbf{r}\mathbf{J}_2^*\cdot \mathbf{E}_2\right]$ instead of two otherwise unrelated LDOS calculations. Hence, $f$ yields precisely the $\tilde{\beta}$ parameter along with any resonant enhancement factors ( $\sim Q / V$ ) in $\mathbf{E}_1$ and $\mathbf{E}_2$ .

Intuitively, $\mathbf{J}_2$ can be thought of as a nonlinear polarization current induced by $\mathbf{E}_1$ in the presence of the second-order susceptibility tensor $\chi^{(2)}$ , and, in particular, is given by $J_{2i} = \tilde{\varepsilon} (\mathbf{r})\sum_{jk}\chi_{ijk}^{(2)}E_{1j}E_{1k}$ where the indices $i,j,k$ run over the Cartesian coordinates. In general, $\chi_{ijk}^{(2)}$ mixes polarizations, and $f$ is a sum of different contributions from various polarization combinations.

In what follows, we focus on the simplest case in which $\mathbf{E}_1$ and $\mathbf{E}_2$ have the same polarization, corresponding to a diagonal $\chi^{(2)}$ tensor determined by a scalar $\chi_{\mathrm{eff}}^{(2)}$ . Such an arrangement can be obtained by, for example, proper alignment of the crystal orientation axes [18,30]. With this simplification, the generalization of the linear topology-optimization problem to the case of SHG becomes

Research Article

Vol. 3, No. 3 / March 2016 / Optica

234

![](dt=2026-04-06/ht=18/5eb081455c5c4c16326fb841f494afd525c65e7c50dfa3d078407aec5bd9f5ab.jpg)

$$
\begin{array}{l} \left. \max  _ {\bar {\varepsilon} _ {\alpha}} \langle f (\bar {\varepsilon} _ {\alpha}; \omega_ {1}) \rangle = - \operatorname {R e} \left[ \left\langle \int \mathbf {J} _ {2} ^ {*} \cdot \mathbf {E} _ {2} d \mathbf {r} \right\rangle \right], \right. \\ \mathcal {M} _ {1} \mathbf {E} _ {1} = i \omega_ {1} \mathbf {J} _ {1}, \\ \mathcal {M} _ {2} \mathbf {E} _ {2} = i \omega_ {2} \mathbf {J} _ {2}, \omega_ {2} = 2 \omega_ {1} \tag {6} \\ \end{array}
$$

where

$$
\begin{array}{l} \mathbf {J} _ {1} = \delta (\mathbf {r} _ {\alpha} - \mathbf {r} _ {0}) \hat {\mathbf {e}} _ {j}, \qquad j \in \{x, y, z \} \\ \mathbf {J} _ {2} = \bar {\varepsilon} (\mathbf {r} _ {\alpha}) E _ {1 j} ^ {2} \hat {\mathbf {e}} _ {j}, \\ \end{array}
$$

$$
\mathcal {M} _ {l} = \nabla \times \frac {1}{\mu} \nabla \times - \varepsilon_ {l} (\mathbf {r} _ {\alpha}) \omega_ {l} ^ {2}, \qquad l = 1, 2
$$

$$
\varepsilon_ {l} (\mathbf {r} _ {\alpha}) = \varepsilon_ {m} + \bar {\varepsilon} _ {\alpha} (\varepsilon_ {d l} - \varepsilon_ {m}), \quad \bar {\varepsilon} _ {\alpha} \in [ 0, 1 ],
$$

and where $\varepsilon_{d}$ denotes the dielectric contrast of the nonlinear medium and $\varepsilon_{m}$ is that of the surrounding linear medium. Note that $\bar{\varepsilon}_{\alpha}$ is allowed to vary continuously between 0 and 1; intermediate values are penalized by threshold projection filters [43]. The scattering framework makes it straightforward to calculate the derivatives of $f$ with r
espect to $\bar{\varepsilon}_{\alpha}$ via the adjoint variable method [35-37]. The optimization problem can then be solved by any gradient-based algorithm, such as the method of moving asymptotes [44].

Figure 1 describes the work flow of our optimization procedure. For computational convenience, the optimization is carried out using a 2D computational cell (in the $x-z$ plane), though the resulting optimized structures are given a finite transverse extension $h_y$ (along the $y$ direction) to make realistic 3D devices (see Fig. 3).

In principle, the wider the transverse dimension, the better the cavity quality factors since they are closer to their 2D limit, which consists only of radiation loss in the $z$ direction; however, as $h_y$ increases, $\tilde{\beta}$ decreases due to increasing mode volumes. In practice, we chose $h_y$ of the order of a few vacuum wavelengths so as not to greatly compromise either $Q$ or $\tilde{\beta}$ . We then analyze the 3D structures via rigorous finite-difference time-domain (FDTD) simulations to determine the resonant lifetimes and modal overlaps.

By virtue of our optimization scheme, we invariably find that frequency matching is satisfied to within the mode linewidths.

# 3. OPTIMAL DESIGNS

Table 1 characterizes the FOMs of some of our newly discovered microcavity designs, involving simple micropost and gratings structures of various $\chi^{(2)}$ materials, including GaAs, AlGaAs, and $\mathrm{LiNbO_3}$ . The low-index material layers of the microposts consist of alumina $(\mathrm{Al}_2\mathrm{O}_3)$ , while gratings are embedded in either silica or air (see Supplement 1 for details).

Note that, in addition to their performance characteristics, these structures significantly differ from those obtained by conventional methods in that traditional designs often involve rings [17,18], periodic structures, or tapered defects [24], which tend to ignore or sacrifice $\tilde{\beta}$ in favor of increased lifetimes, and for which it is also difficult to obtain widely separated modes [19].

Figure 2 illustrates one of the optimized structures—a doubly resonant rectangular micropost cavity with alternating $\mathrm{AlGaAs / Al_2O_3}$ layers—along with spatial profiles of the fundamental and harmonic modes. For convenience, we consider modes with the same polarization (major field component $E_{y}$ ): although AlGaAs (similar to GaAs) has a nonvanishing off-diagonal tensor element $\chi_{xyz}^{(2)}$ , it can couple the $E_{y}$ components of fundamental and second-harmonic modes if the crystal plane is appropriately oriented in the (111) direction [19].

The cavity designed by our approach differs from conventional microposts in that it does not consist of periodic bilayers, yet it supports two localized modes at precisely $\lambda_{1} = 1.5~\mu \mathrm{m}$ and $\lambda_{2} = \lambda_{1} / 2$ . In addition to having large $Q^{\mathrm{rad}} \gtrsim 10^{5}$ and small $V \sim (\lambda_{1} / n)^{3}$ , the structure exhibits an ultralarge nonlinear coupling $\tilde{\beta} \approx 0.018$ that is almost 1 order of magnitude larger than the best overlap found in the literature (see Fig. 3).

An interesting aspect of the optimized structures is the appearance of deeply subwavelength features $\sim 1\% -5\%$ of $\frac{\lambda_1}{n}$ , leading to a kind of metamaterial geometry in the optimization direction; we surmise that these arise regardless of starting conditions in order to accommodate a delicate cancellation of the out-going radiation

Table 1. SHG Figures of Merit, Including Frequencies $\lambda$ , Overall and Radiative Quality Factors $Q$ ; $Q^{rad}$ , and Nonlinear Coupling $\beta$ Corresponding to the Fundamental and Harmonic Modes of Topology-Optimized Micropost and Grating Cavities of Different Material Systems

![](dt=2026-04-06/ht=18/7cd29e29698ddfc28809e87ee2dbdab74a1b27d40af8a11c8ed26b711ff918e8.jpg)

<table><tr><td>Structure</td><td>bx×by×bz(λ13)</td><td>λ(μm)</td><td>(Q1,Q2)</td><td>(Q1rad, Q2rad)</td><td>β̅</td><td>FOM1</td><td>FOM2</td></tr><tr><td>(1) AlGaAs/Al2O3micropost</td><td>8.4 × 3.5 × 0.84</td><td>1.5–0.75</td><td>(5000, 1000)</td><td>(1.4 × 10^5, 1.3 × 10^5)</td><td>0.018</td><td>7.5 × 10^6</td><td>8.3 × 10^11</td></tr><tr><td>(2) GaAs gratings in SiO2</td><td>5.4 × 3.5 × 0.60</td><td>1.8–0.9</td><td>(5000, 1000)</td><td>(5.2 × 10^4, 7100)</td><td>0.020</td><td>7 × 10^6</td><td>7.5 × 10^9</td></tr><tr><td>(3) LN gratings in air</td><td>5.4 × 3.5 × 0.80</td><td>0.8–0.4</td><td>(5000, 1000)</td><td>(6700, 2400)</td><td>0.030</td><td>8.4 × 10^5</td><td>9.7 × 10^7</td></tr></table>

Research Article

Vol. 3, No. 3 / March 2016 / Optica

235

![](dt=2026-04-06/ht=18/b741b661b4b05052c0264a958692e2262f5cd1c77ebb63149457c03538ba9692.jpg)

(resulting in high $Q$ ) and constructive nonlinear overlap (large $\beta$ ) at the precise designated frequencies. In particular, we find that these features are not easily removable, as their absence greatly perturbs the quality factors and frequency matching. Here, it is worth mentioning that this level of structural sensitivity is typically absent in modes confined by traditional bandgap mechanisms via Bragg scattering [10].

However, to the best of our knowledge, no structure exists that possesses multiple bandgaps with strongly confined modes ( $Q > 10^4$ ) at vastly disparate frequency regimes, not to mention modes that exhibit a large nonlinear overlap. In contrast, topology optimization suggests that one needs very careful interference cancellations enabled by aperiodic arrangement and subwavelength features to simultaneously achieve precise frequency matching and optimal nonlinear overlap.

To understand the mechanism of improvement in $\bar{\beta}$ , it is instructive to consider the spatial profiles of interacting modes. Figure 2(b) plots the $y$ components of the electric fields in the $x-z$ plane against the background structure. Since $\bar{\beta}$ is a net total

![](dt=2026-04-06/ht=18/331cf78697d2ae66d649066ba855b7b4116337078bf34ab80151cd734cbb91b2.jpg)

of positive and negative contributions coming from the local overlap factor $E_1^2 E_2$ in the presence of nonlinearity, not all local contributions are useful for SHG conversion. Most notably, one observes that the positions of negative anti-nodes of $E_2$ (light red regions) coincide with either the nodes of $E_1$ or alumina layers (where $\chi^{(2)} = 0$ ), minimizing negative contributions to the integrated overlap. In other words, improvements in $\bar{\beta}$ do not arise purely due to tight modal confinement, but also from the constructive overlap of the modes enabled by the strategic positioning of field extrema along the structure.

From an experimental point of view, the realization of the multilayer stack is well documented in the literature [45,46]: (i) stacks of $\mathrm{Al}_x\mathrm{Ga}_{1 - x}\mathrm{As} / \mathrm{Al}_y\mathrm{Ga}_{1 - y}$ As layers can be readily grown (metal organic chemical vapor deposition or molecular beam epitaxy), where $x$ and $y$ are chosen so that the layers are lattice matched and also $x\sim 1$ and $y\sim 0$ ; (ii) the pillar can be defined using electron beam lithography and reactive ion etching (e.g.

, using Cl-based chemistry); (iii) the structure is placed in the oxidation furnace: then high Al content layers $(\mathrm{Al}_x\mathrm{Ga}_{1 - x}\mathrm{As})$ are oxidized and turned into $\mathrm{Al_2O_3}$ $(\mathrm{AlO}_x$ to be precise), whereas low Al content layers are intact. This takes advantage of the well-known fact that large Al content AlGaAs layers can be oxidized much faster than low Al content ones. Additionally, the micropost cavity can be naturally integrated with quantum dots and quantum wells for cavit
y QED applications [47].

Similar to other wavelength-scale structures, the operational bandwidths of these structures are limited by radiative losses in the lateral direction [10,48,49], but their ultralarge overlap factors more than compensate for the increased bandwidth, which ultimately may prove beneficial in experiments subject to fabrication imperfections and for large-bandwidth applications [1,2,6,50].

Based on the tabulated FOMs (Table 1), the efficiencies and power requirements of realistic devices can be directly calculated. For example, assuming $\chi_{\mathrm{eff}}^{(2)}(\mathrm{AlGaAs})\sim 100~\mathrm{pm / V}$ [18], the $\mathrm{AlGaAs / Al_2O_3}$ micropost cavity (Fig. 2) yields an efficiency of $\frac{P_{2,\mathrm{out}}}{P_1^2} = 2.7\times 10^4 /\mathrm{W}$ in the undepleted regime when the modes are critically coupled, $Q = Q^{\mathrm{rad}}$ .For larger operational bandwidths, e.g.

, $Q_{1} = 5000$ and $Q_{2} = 1000$ we find that $\frac{P_{2,\mathrm{out}}}{P_1^2} = 16 / \mathrm{W}$ . When the system is in the depleted regime and critically coupled, we find that a maximum efficiency of $25\%$ can be achieved at $P_{1}^{\mathrm{crit}}\approx 0.15\mathrm{mW}$ , whereas, when assuming smaller $Q_{1} = 5000$ and $Q_{2} = 1000$ , a maximum efficiency of $96\%$ can be achieved at $P_{1}^{\mathrm{crit}}\approx 0.96\mathrm{W}$

# 4. COMPARISON AGAINST PREVIOUS DESIGNS

Table 2 summarizes various performance characteristics, including the aforementioned FOM, for a handful of previously studied geometries with length scales spanning from millimeters to a few micrometers. Figure 3 demonstrates a trend among these geometries toward increasing $\tilde{\beta}$ and decreasing $Q^{\mathrm{rad}}$ as device sizes decrease. Maximizing $\tilde{\beta}$ in millimeter-to-centimeter scale bulky media translates to the well-known problem of phase matching the momenta or propagation constants of the modes [30].

In this category, traditional whispering gallery mode resonators (WGMRs) offer a viable platform for achieving high-efficiency conversion [26]; however, their ultralarge lifetimes (critically dependent upon material-specific polishing techniques), large sizes (millimeter length scales), and extremely weak nonlinear coupling (large mode volumes) render them far from optimal chip-scale devices. Although miniature WGMRs, such as microdisk and

Research Article

Vol. 3, No. 3 / March 2016 / Optica

236

Table 2. SHG Figures of Merit (see Table 1) of Representative, Hand-designed Geometries based on Bang-gap or Index-guided Confinement ${}^{a}$

![](dt=2026-04-06/ht=18/880aa9484ab35a81e607b573b5234d0eb60fe864148432eb3388c41eb421072e.jpg)

<table><tr><td>Structure</td><td>λ(μm)</td><td>(Q1,Q2)</td><td>(Q1rad, Q2rad)</td><td>β̅</td><td>FOM1</td><td>FOM2</td></tr><tr><td>LN WGM resonator [26]</td><td>1.064–0.532</td><td>(3.4 × 107, -)</td><td>(6.8 × 107, -)</td><td>-</td><td>~1010</td><td>-</td></tr><tr><td>AlN microring [17]</td><td>1.55–0.775</td><td>(~104, ~5000)</td><td>-</td><td>-</td><td>2.6 × 105</td><td>-</td></tr><tr><td>GaP PhC slab [16]b</td><td>1.485–0.742</td><td>(≈6000, -)</td><td>-</td><td>-</td><td>≈2 × 105</td><td>-</td></tr><tr><td rowspan="2">GaAs PhC nanobeam [19]</td><td>1.7–0.91c</td><td>(5000, 1000)</td><td>(&gt;106, 4000)</td><td>0.00021</td><td>820</td><td>1.8 × 108</td></tr><tr><td>1.8–0.91</td><td>(5000, 1000)</td><td>(6 × 104, 4000)</td><td>0.00012</td><td>227</td><td>2.1 × 105</td></tr><tr><td>AlGaAs nanoring [18]</td><td>1.55–0.775</td><td>(5000, 1000)</td><td>(104, &gt; 106)</td><td>0.004</td><td>105</td><td>1.6 × 109</td></tr></table>

Also shown are the $\mathrm{FOM}_1$ and $\mathrm{FOM}_2$ figures of merit described in Eqs. (4) and (5).

SHG occurs between a localized defect mode (at the fundamental frequency) and an extended index-guided mode of the photonic crystal (PhC).

Resonant frequencies are mismatched.

microring resonators [17,27,29], show increased promise due to their smaller mode volumes, improvements in $\bar{\beta}$ are still hardly sufficient for achieving high efficiencies at low powers. Ultracompact nanophotonic resonators, such as the recently proposed nanorings [18], 2D photonic crystal defects [16], and nanobeam cavities [19], possess even smaller mode volumes but prove challenging for design due to the difficulty of finding well-confined modes at both the fundamental and second-harmonic frequencies [16]. Even when two such resonances can be found by fine-tuning a limited set of geometric parameters [18,19], the frequency-matching constraint invariably leads to suboptimal spatial overlaps, which severely limits the maximal achievable $\bar{\beta}$ .

Our optimization method seeks to maximize intrinsic geometric parameters of an unloaded cavity, e.g., $Q^{\mathrm{rad}}$ and $\tilde{\beta}$ , whereas the loaded cavity lifetime $Q$ depends on the choice of coupling mechanism, e.g., free-space, fiber, or waveguide coupling, and is therefore an external parameter that can be considered independently of the optimization. When evaluating the performance characteristics, such as $\mathrm{FOM}_1$ , we assume total operational lifetimes $Q_1 = 5000$ , $Q_2 = 1000$ .

In comparing Tables 1 and 2, one observes that, for a comparable $Q$ , the topology-optimized structures perform significantly better in both $\mathrm{FOM}_1$ and $\mathrm{FOM}_2$ than any conventional geometry, with the exception of the lithium niobate (LN) gratings, whose low $Q^{\mathrm{rad}}$ lead to slightly lower $\mathrm{FOM}_2$ . Generally, the optimized microposts and gratings perform better by virtue of a large and robust $\tilde{\beta}$ which, notably, is significantly larger than that of existing designs.

Here, we have not included in our comparison those structures that achieve non-negligible SHG by special poling techniques and/or quasi-phase-matching methods [29,30,51], though their performance is still suboptimal compared to the topology-optimized designs. Such methods are highly material-dependent and are thus not readily applicable to other material platforms; instead, ours is a purely geometrical topology optimization technique applicable to any material system.

# 5. CONCLUDING REMARKS

In conclusion, we have presented a formulation that allows for large-scale optimization of SHG. Applied to simple micropost and grating structures, our approach yields new classes of microcavities with stronger performance metrics over existing designs. One potentially challenging aspect for fabrication in the case of gratings is the presence of deeply subwavelength features, which would require difficult high-aspect-ratio etching or growth techniques. Another caveat about wavelength-scale cavities is that they are sensitive to structural perturbations near the cavity center, where most of the field resides. In our optimized structures, the

FOMs are robust to within $\sim \pm 20$ nm variations (approximately one computational pixel). One possible way to constrain the optimization to ensure some minimum spatial feature and robustness is to exploit so-called regularization filters and worst-case optimization techniques [43], which we will consider in future work. However, subwavelength features and structural sensitivity should not be an issue for the micropost cavities since each material layer can be grown/deposited to a nearly arbitrary thickness with angstrom precision [47,48].

Our micropost cavities represent fundamentally new photonic designs obtained by a novel design process—arguably, they could not have been designed from intuition alone. Furthermore, the proposed optimization framework provides a natural versatile tool to tackle various challenging scenarios and exotic applications in nonlinear photonics including, for example, higher-order frequency conversion processes with more than two modes, as well as problems that require conversion of single photo
ns and quantum states of light.

Funding. Air Force Office of Scientific Research (AFOSR) (FA9550-14-1-0389); Army Research Office through the Institute for Soldier Nanotechnologies (W911NF-13-D-0001); National Science Foundation (NSF) (DGE-1144152, DMR-1454836).

See Supplement 1 for supporting content.

# REFERENCES

Research Article

Vol. 3, No. 3 / March 2016 / Optica

237

Research Article

Vol. 3, No. 3 / March 2016 / Optica

238