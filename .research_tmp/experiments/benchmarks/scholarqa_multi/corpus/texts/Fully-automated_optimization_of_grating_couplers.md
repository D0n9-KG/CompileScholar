# Fully-automated optimization of grating couplers

LOGAN SU, $^{1,*}$ RAHUL TRIVEDI, $^{1}$ NEIL V. SAPRA, $^{1}$ ALEXANDER Y. PIGGOTT, $^{1}$ DRIES VERCRUYSSE, $^{1,2}$ AND JELENA VUČKOVIĆ $^{1}$

Abstract: We present a gradient-based algorithm to design general 1D grating couplers without any human input from start to finish, including a choice of initial condition. We show that we can reliably design efficient couplers to have multiple functionalities in different geometries, including conventional couplers for single-polarization and single-wavelength operation, polarization-insensitive couplers, and wavelength-demultiplexing couplers. In particular, we design a fiber-to-chip blazed grating with under $0.2\mathrm{dB}$ insertion loss that requires a single etch to fabricate and no back-reflector.

© 2018 Optical Society of America under the terms of the OSA Open Access Publishing Agreement

OCIS codes: (050.2770) Gratings; (130.0130) Integrated optics; (130.3120) Integrated optics devices.

# References and links

Check for updates

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4023

Optics EXPRESS

#312953

Journal © 2018

https://doi.org/10.1364/OE.26.004023

Received 10 Nov 2017; accepted 23 Jan 2018; published 7 Feb 2018

# 1. Introduction

Edge couplers and grating couplers are the primary interfaces used between integrated photonic circuits and optical fibers. Grating couplers are attractive because they are typically easier to fabricate, are flexible in their placement, and enable wafer-scale testing. However, grating couplers tend to have lower coupling efficiencies [1].

The simplest coupler consists of a uniform grating; however, the maximum efficiency of such a grating is limited [2]. For instance, the coupling loss is more than 2.6 dB for $220\mathrm{nm}$ thick silicon-on-insulator (SOI) architecture [3]. Higher coupling efficiencies can be achieved in a wide variety of ways, including nonuniform gratings [4, 5], bottom reflectors [6, 7], multiple

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4024

Optics EXPRESS

layers [8-10], multiple etch depths [11, 12], silicon overlays [13, 14], blazed gratings [15-17], and unconventional geometries [18, 19].

A direct consequence of the diversity of grating geometries is that grating couplers need to be optimized to the specific geometry and desired functionalities. Extensive literature exists on optimizing grating couplers [3,4,10,13,20-25], but these optimizations often rely on starting with a standard design [3, 10]. For conventional geometries and designs, there is well-known analytical theory to suggest an appropriate starting condition. However, for unconventional geometries or devices with multiple functionalities (e.g. wavelength demultiplexing), analytical theory becomes more challenging.

In addition, many grating optimization techniques rely on parameter sweeps, random perturbations, or population-based metaheuristic algorithms, such as genetic algorithms and particle swarm optimization, all of which can be time-consuming to perform.

In contrast, gradient-based methods have been promising in designing a wide variety of nanophotonic structures owing to their ability to explore a larger design space [26-30]. This is possible because gradient-based methods requires only one forward simulation to calculate the fields and one "backward" simulation to calculate the gradient by using the adjoint method (see supplementary material in [31]).

The optimization landscape of discrete, fabricable structures is highly non-convex and difficult to navigate. Consequently, gradient-based optimization over this space requires finding a suitable initial condition. To automate this process, the discrete optimization stage can be preceded with a simpler optimization problem whereby the permittivity distribution is allowed to vary continuously between that of the cladding and that of the device. A properly chosen discretization for converting the resulting structure of this continuous optimization stage into a starting structure for the discrete stage is critical for achieving efficient devices.

In this work, we present such a two-stage gradient-based optimization algorithm for 1D uniform grating couplers. We compare three different discretization methods and show that our choice of discretization procedure can reliably design efficient gratings using completely random initial conditions, thus fully automating the design process. To illustrate the flexibility of our method, we design a wide class of fiber-to-chip grating couplers, including polarization-insensitive couplers, wavelength-demultiplexing couplers, and highly efficient single-wavelength couplers. Notably, we design a blazed grating coupler with under $0.2\mathrm{dB}$ loss requiring only a single etch step to fabricate.

# 2. Optimization method

# 2.1. Nanophotonic inverse design

The nanophotonic inverse design problem is given by

$$
\underset {p, E _ {1}, E _ {2}, \dots , E _ {m}} {\text {m i n i m i z e}} \quad \sum_ {i} f _ {i} (E _ {i})
$$

$$
\text {s u b j e c t} \quad \nabla \times \frac {1}{\mu_ {0}} \nabla \times E _ {i} - \omega_ {i} ^ {2} \epsilon (p) E _ {i} = - i \omega_ {i} J _ {i}, \tag {1}
$$

$$
i = 1, 2, \dots , m
$$

where $m$ is the number of modes, $E_{i}$ is the electric field at $\omega_{i}$ , $J_{i}$ is electric field source, $p$ is vector that parametrizes the structure, and $f_{i}$ is the objective function. $f_{i}$ is either equal to $f_{i}^{M}$ to optimize power in a waveguide mode or $f_{i}^{P}$ to optimize power across a plane. Specifically, $f_{i}^{M}$ is defined by

$$
f _ {i} ^ {M} \left(E _ {i}\right) = I _ {+} \left(\left| \mathcal {L} \left(E _ {i}\right) \right| - \alpha_ {i}\right) + I _ {-} \left(\left| \mathcal {L} \left(E _ {i}\right) \right| - \beta_ {i}\right) \tag {2}
$$

where $\mathcal{L}(E_i)$ is the overlap integral with the waveguide mode as defined in the supplementary material of [31]; $I_{+}$ and $I_{-}$ are continuous relaxations of indicator functions as defined in the

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4025

Optics EXPRESS

supplementary material of [31]; and $\alpha_{i} = 1$ and $\beta_{i} = 0.99$ when maximizing the power and $\alpha_{i} = 0.01$ and $\beta_{i} = 0$ when minimizing power. $f_{i}^{P}$ is defined by

$$
f _ {i} ^ {P} \left(E _ {i}\right) = \int \Re \left[ E _ {i} \times H _ {i} ^ {*} \right] d S \tag {3}
$$

where $H_{i}$ is the magnetic field and the integration is performed over the desired plane.

In our simulation domain, we specify a rectangular design region within which the grating resides. The permittivity of the design region is determined by a parametrization vector $p$ , while the permittivity distribution outside the design region is fixed.

Our optimization algorithm is broken down into two stages: continuous and discrete. In each stage, the optimization problem described in Equation 1 is solved with different parametrizations of the structure. A discretization process converts the optimized structure from the continuous stage into the initial structure for the discrete stage.

In the continuous optimization stage, the design region is divided into equally spaced pixels, with each element of $p$ representing each pixel. Each pixel takes on a value between 0 and 1, where 0 represents the cladding and 1 represents the device. We optimize using the second-order L-BFGS-B algorithm [32] for a fixed number of iterations or until convergence, whichever occurs first.

In the discrete optimization stage, the structure is parametrized by the location of the edges of the grating grooves, with each element of $p$ representing a single edge. Under this parametrization, the structure represents a discrete, fabricable device. Feature size constraints are implemented by constraining the distance between the edge locations to be at least the minimum feature size. In order to handle these constraints, we op
timize using another gradient-based algorithm, SLSQP [33].

We simulate the grating couplers using the finite-difference frequency domain (FDFD) method [34, 35]. All simulations are performed with a spatial discretization of $20\mathrm{nm}$ . The simulation region is surrounded by perfectly matched layers (PMLs) on all four sides. We model the fiber mode as a Gaussian beam with a waist $\sigma_w$ and use an input current source of the form $\exp (-x^2 /\sigma_w^2)$ .

Since typical grating coupler sizes are on the order of $10\mu \mathrm{m}$ , optimizing full 3D grating couplers is computationally expensive. Instead, we simulate the gratings in 2D. Fortunately, the difference in performance between 2D and 3D simulations is often negligible; for instance, when coupling to $12\mu \mathrm{m}$ strip waveguide at $1550\mathrm{nm}$ , the 3D structure has an efficiency roughly $97\%$ of that of the 2D device [4]. Nonetheless, we emphasize that our reported efficiencies are for 2D coupling efficiencies; the exact achieved efficiency in 3D depends on the length of the coupler in the third dimension.

Roughly 700 simulations were required per mode of the optimization problem, and the total simulation time was approximately $2m$ hours on a single 6-core Intel Core i7 machine where $m$ is the number of modes.

# 2.2. Discretization

Since discrete optimization is inherently harder than continuous optimization and the number of grating edges is fixed in discrete stage, it is imperative to start with a good initial condition for the discrete stage to achieve structures with the high efficiency. In this section, we discuss three possible discretization methods.

One simple way of converting a continuous structure into a discrete one is via simple thresholding whereby pixels whose values are greater than $\frac{1}{2}$ are set to 1 and pixels less than $\frac{1}{2}$ are set to 0. However, this method results in many closely-spaced grating edges, which performs poorly when feature constraints force the edge locations to spread out. Variations on thresholding, including post-processing the structure to remove closely-spaced edges, also do not

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4026

Optics EXPRESS

perform well.

Rather than developing a hand-crafted heuristic algorithm to perform discretization and post-processing, the discretization task can be as an optimization problem. Intuitively, a good initial condition is one that is similar to the optimized continuous structure. This sentiment can be formalized through an optimization problem, which we will call the least-squares discretization (L2D):

$$
\begin{array}{l} \underset {p, n} {\text {m i n i m i z e}} \quad | | R (p) - q | | _ {2} \\ \text {s u b j e c t} p _ {i + 1} \geq p _ {i} + d \tag {4} \\ p _ {1} \geq 0 \\ p _ {n} \leq L \\ \end{array}
$$

where $p \in \mathbb{R}^n$ is a vector of edge locations, $q \in \mathbb{R}^m$ is the parametrization of the optimized continuous device, $d$ is the minimum feature size in terms of pixels (fractional pixels are allowed), $L$ is the number of pixels (i.e. design length), and $R: \mathbb{R}^n \to \mathbb{R}^m$ is a function that takes a vector of edge locations and renders it onto the same grid of pixels as in the continuous optimization.

On occasion, we observed L2D produces a discrete device with features, corresponding to regions of weakly-modulated permittivities in the continuous stage. To mitigate this, the optimized continuous structure $q$ is first deconvolved using the optimization problem

$$
\begin{array}{l} \underset {q ^ {\prime}} {\text {m i n i m i z e}} \quad | | A q ^ {\prime} - q | | _ {2} \tag {5} \\ \text {s u b j e c t} \quad 0 \leq q _ {i} ^ {\prime} \leq 1 \\ \end{array}
$$

where $A$ is the matrix representation of a convolution kernel. Then, the optimal deconvolved structure $(q')^*$ is used as $q$ in Equation 4. In our optimizations, we have chosen a convolution matrix corresponding to a moving average across 5 elements. We will refer to this approach as the deconvolved least-squares discretization (D-L2D).

# 2.3. Optimization statistical study

We ran the optimization 100 times using randomly generated initial conditions and applying each discretization method described in Section 2.2 for coupling $1550\mathrm{nm}$ into a $340\mathrm{nm}$ thick waveguide at normal incidence (Fig. 1). We have chosen to use an infinite buried oxide (BOX) layer in order to reduce computation time. In addition, we ran the optimization method without the continuous stage. For these "discrete-only" optimization runs, we chose to seed the optimization by first choosing a vector uniformly at random and then applying D-L2D to arrive at a discrete starting condition.

The statistical studies show most choices for initial condition result in reasonably efficient devices. Nevertheless, using optimization to perform discretization is superior than thresholding or forgiving the continuous optimization entirely.

Discrete-only optimization performs significantly worse than L2D and D-L2D for $50~\mathrm{nm}$ feature sizes because of the highly non-convex landscape. The better relative performance at $100\mathrm{nm}$ arises from the fact that the design space becomes smaller as the feature size increases. Consequently, it is more likely to randomly pick a good initial condition for discrete-only optimization.

Thresholding performs similarly to L2D and D-L2D for $50\mathrm{nm}$ feature sizes because the small feature size means that the grating can have many closely-spaced edges. However, at $100\mathrm{nm}$ feature sizes, the grating edges can no longer be placed so densely together, so thresholding performs even worse than the discrete-only optimization. We have found that modifications to thresholding only results in a performance between different feature sizes. For example, post-processing thresholding step to remove closely-spaced edges improved the performance at $100\mathrm{nm}$ feature size at the expense of poorer performance at $50\mathrm{nm}$ feature size.

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4027

Optics EXPRESS

![](dt=2026-04-09/ht=22/16da0c42f0249a13e78950f44515557ba1c9ce2694dbe36aa230a6ec95aaf086.jpg)

![](dt=2026-04-09/ht=22/ac8364518b3bad54b875542b3b63e5e1dbf39dc62dcf1887ce9c517634e86929.jpg)

![](dt=2026-04-09/ht=22/a0b389a826363f53309f885d3044c8a291ae633a4fb1073243adba2018f17679.jpg)

![](dt=2026-04-09/ht=22/88f4d500b7339479e1a715b48c81c76306b38b4b4dbb3eadd83a501a5459e7a8.jpg)

![](dt=2026-04-09/ht=22/ac73f5c9ebc2c333af89f21e3d4d746f44be8200a2b29d38a4cebdb78c97002a.jpg)

![](dt=2026-04-09/ht=22/313f88f4d6f1ee4e906ed908b0f52fdf6dcba899b0c13df4664fd96ea70d0a0e.jpg)

![](dt=2026-04-09/ht=22/01577cf732b9abcb93e704844073992d3ea8da5287f4dfe7618f4ba1d284ecee.jpg)

![](dt=2026-04-09/ht=22/ee74de2a42a8c1ef83374256c7f1651a2f1b222046a13763e25b8465b6b5ad49.jpg)

![](dt=2026-04-09/ht=22/31bc57787ed5003040b1d10becd0f771b41c5ac53275f208821a4bf5d3ba3dea.jpg)

![](dt=2026-04-09/ht=22/b1f28e48ff8f701366669587c742eff98a21a038eac15c522bfc8195bd38fdbb.jpg)

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS E
XPRESS 4028

Optics EXPRESS

In contrast, L2D and D-L2D perform the best in both cases. D-L2D performs slightly better than L2D in all cases, both in terms of the mean efficiency and the maximum efficiency out of all 100 runs. This is intuitive because L2D performs well when the optimized continuous structure appears mostly discrete, and the deconvolution used in D-L2D does not affect substantially continuous structures that are mostly discrete. Therefore, there is little disadvantage in employing D-L2D over L2D. We have also performed a similar study for wavelength-demultiplexing grating couplers and found similar results.

# 3. Grating coupler designs

In this section, we illustrate our optimization method by designing a wide variety of grating couplers, including polarization-insensitive and wavelength-demultiplexing gratings.

# 3.1. Single-function grating couplers

Here we focus on gratings for $1550\mathrm{nm}$ on the $220\mathrm{nm}$ thick silicon-on-insulator (SOI) platform. Higher efficiencies can be achieved by using thicker waveguides, but we focus on $220\mathrm{nm}$ because of its role as a common industry standards [1].

![](dt=2026-04-09/ht=22/8f6df0821acb8c82bfa8be3dbeb187c240305a4ad191765a4834c698809e9b9f.jpg)

![](dt=2026-04-09/ht=22/c7d25de98391dd34f6ddb43b545e739db268650299b56e8bb47647e708bd2236.jpg)

![](dt=2026-04-09/ht=22/a3a641f3ceabb87f76f7b8b95b6f8a5ab5d4491e71aa399c6203e08602e8face.jpg)

Figure 2 shows the design of a conventional grating coupler for $220\mathrm{nm}$ SOI platform at 10 degree incident angle with $100\mathrm{nm}$ feature sizes. According to analytical theory, the maximum efficiency is achieved by modulating the etch depth or grating period [36]. Ideally, the grating period would become smaller and smaller as one approaches the start of the waveguide. This is precisely what occurs in our design. At the beginning of the waveguide, the grating periodicity

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4029

Optics EXPRESS

becomes atypical, but this is a consequence of the desired periodicity falling below the minimum feature size. This apodized design closely resembles previous optimization work in grating couplers [3], and we achieve a similar insertion loss of 1.94 dB (compared to 2.12 dB in [3]).

![](dt=2026-04-09/ht=22/cde0f4b9fe60ef8c5bc4744da481d4f5fc6955a0fb82d78e82a1bb149eadc73b.jpg)

![](dt=2026-04-09/ht=22/1daed121cd13d06716609222265b2d2a1bf742be97be6d808d447d9f11b753b1.jpg)

![](dt=2026-04-09/ht=22/d5f0b9724620725e4adfb94c12742434e650b8088f44e31fde63315337eb386d.jpg)

In order to achieve higher efficiencies, one can use more complicated geometries. Figure 3 shows a two-layer grating structure optimized with our algorithm. This geometry is similar to the one presented in [10], and we achieve a similar insertion loss of $0.25\mathrm{dB}$ (compared to $0.165\mathrm{dB}$ ) with a structure that resembles the one presented in [10]. However, in [10], the initial condition was physically-motivated by considering constructive and destructive interference between the top and bottom layers, whereas our algorithm used a completely random initial condition.

Instead of pre-selecting a particular geometry to optimize, we utilized our method to suggest an optimal geometry for a single-wavelength grating coupler (Fig. 4). To achieve this, the structure was parametrized so that all pixels in the design region are allowed to vary continuously. The optimized continuous result clearly suggests that blazed gratings are an optimal geometry, which is consistent with theoretical analysis in [15]. Using this suggestion, we designed a blazed grating with 50 degree slant.

In order to handle the slants, the continuous stage optimization was modified so that each value in the parametrization corresponds to a parallelogram pixel with a width and height equal to the spatial discretization. The resulting device has a minimum insertion loss of $0.17\mathrm{dB}$ with a $26\mathrm{nm}1$ -dB bandwidth.

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4030

Optics EXPRESS

![](dt=2026-04-09/ht=22/f8481be0b48f56bce82aaf67a45bddfe0013656e44fcf5f7923409124c044f3d.jpg)

![](dt=2026-04-09/ht=22/529290982e2d7c93f895ac006c4cc3290ad222ad7949bbfa57a37463f00bab89.jpg)

![](dt=2026-04-09/ht=22/9a54c178cca94665baa6c8049846c8142ad771026e994d5df5566eed079caed0.jpg)

![](dt=2026-04-09/ht=22/47f82c92e116de02cff957210b1c5743b93cb9415fba7005584bdf71d15b6745.jpg)

# 3.2. Multi-function grating couplers

In this section, we explore grating couplers that have more than one functionality. Because of the multiple functions, it is more difficult to derive a suitable initial condition analytically, and a fully-automated optimization process becomes particularly useful.

Figure 5 shows a polarization-insensitive grating where the TE Gaussian mode is coupled to the $\mathrm{TE}_0$ mode of the waveguide and the TM Gaussian mode is coupled to the $\mathrm{TM}_0$ mode of the waveguide. Such gratings are useful because the input fiber is often not polarization-maintaining. Ref. [37] shows a similar design with $4.3\mathrm{dB}$ insertion loss for TE and $3.2\mathrm{dB}$ insertion loss for TM for $340\mathrm{nm}$ thick waveguides. Our device has an insertion loss of $2.9\mathrm{dB}$ for TE mode and $3.6\mathrm{dB}$ for TM mode over a $28\mathrm{nm}$ bandwidth using $220\mathrm{nm}$ waveguides.

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4031

Optics EXPRESS

![](dt=2026-04-09/ht=22/af8f00831d761bc1118f61e49d42210e6745b209ce7d20866b3825ad5efeba5c.jpg)

![](dt=2026-04-09/ht=22/d6f81c8f12ba31692b9aefcf043ac47513944a09f53ee49dccfbfb42915ffa52.jpg)

![](dt=2026-04-09/ht=22/3992a66403ba1527ae9ad1ed25d7fc950718fc76a9a45c4990c9dd9a1d946fb1.jpg)

![](dt=2026-04-09/ht=22/37ef703daf6994e832832af8bd006aff8a5a3e6e19c8604da7d950454b6a969b.jpg)

![](dt=2026-04-09/ht=22/f66027fe5285d8beaff0424950833c071bb7ec6f1513e34fc1550327119c9dfc.jpg)

![](dt=2026-04-09/ht=22/cc4ee2affb989575c2b67d3912cd4a461514249b20eeb4c3593211ac300844f6.jpg)

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4032

Optics EXPRESS

![](dt=2026-04-09/ht=22/fb26ac5dd020984d37acc0e01785d871b764916fa2284250f76a67bf77de2bd4.jpg)

![](dt=2026-04-09/ht=22/d939f1a50598a855781e0f8ec815dbd03a061d79a7f24e5bb3c669aabdeee33c.jpg)

![](dt=2026-04-09/ht=22/75a83c58cb7b817a08579cf8365471054be95151b510216e4b806ace20a72f84.jpg)
![](dt=2026-04-09/ht=22/af532f38d60063c1d122e067062e66032e5b488eae065535cfc55f20ee58a96c.jpg)

![](dt=2026-04-09/ht=22/1d5370a2cc6e0b52e0ad0373f801524b489905c76348048e7f3cb5e77075f513.jpg)

![](dt=2026-04-09/ht=22/d74a1407161e2ce8ac092914a7ae759ede8e9b8aa7e048a006786c23f236f1c2.jpg)

Next, we designed a wavelength demultiplexing grating coupler that couple $1310\mathrm{nm}$ light into the fundamental mode of one waveguide and couple $1490\mathrm{nm}$ light into the fundamental mode of another waveguide (Fig. 6). Such a grating is useful in wavelength division multiplexing systems where multiple wavelengths are utilized to increase the communication bandwidth. Unlike [38] and [39], this grating operates at normal incidence. This is a geometry that we have studied before in [40] but the design presented there was a focused Gaussian spot rather than one meant for fiber mode.

The device presented here has an insertion loss of $1.5\mathrm{dB}$ at $1310\mathrm{nm}$ and $1.6\mathrm{dB}$ at $1490\mathrm{nm}$ with over $21\mathrm{dB}$ crosstalk suppression. We note that this was achieved using an infinite BOX layer, which is a more difficult design problem.

To achieve even higher efficiency for wavelength demultiplexing designs, we introduce the pass-through geometry in which only one of the wavelengths is coupled into an on-chip waveguide whereas another wavelength passes through the grating. A photodetector can then be placed behind the chip to collect the light that passes through the device. This is useful for systems where the pass-through wavelength does not require additional processing.

One particular application is in an optical transceiver: The transmitting wavelength would out-couple into an optical fiber through the waveguide whereas the receiving wavelength would pass through the structure and be detected (Fig. 7(a)). The advantage of this geometry is that the couplers can be more efficient because high transmission through the grating is easier to achieve. Figure 7 shows a design with $1.0\mathrm{dB}$ insertion loss at $1310~\mathrm{nm}$ (the on-chip wavelength) and $0.08\mathrm{dB}$ loss at $1490~\mathrm{nm}$ (the pass-through wavelength).

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4033

Optics EXPRESS

# 4. Conclusion

We have presented a general gradient-based 1D grating design algorithm that fully automates the design process, enabled by appropriately choosing a least-squares discretization procedure. Using this algorithm, we design efficient couplers in different geometries and with different functionalities, including polarization-insensitive gratings, wavelength-demultiplexing gratings, and a single-wavelength grating coupler with under $0.2\mathrm{dB}$ insertion loss.

# Funding

Air Force Office of Scientific Research (AFOSR) MURI for Aperiodic Silicon Photonics (FA9550-15-1-0335); Gordon and Betty Moore Foundation; Futurewei Technologies, Inc.; Marie Sklodowska-Curie Grant (665501); Kailath Stanford Graduate Fellowship.

# Acknowledgements

D. V. acknowledges funding from FWO and European Union's Horizon 2020 research and innovation program under the Marie Sklodowska-Curie grant agreement No 665501. We thank Google for providing computational resources on the Google Cloud Platform.

Research Article

Vol. 26, No. 4 | 19 Feb 2018 | OPTICS EXPRESS 4034

Optics EXPRESS