Article

# Polarization-Splitting Grating Coupler on Lithium Niobate Thin Film

Zhihua Chen $^{1,2,*}$ , Longxi Chen $^{1,2}$ , Xiangjia Meng $^{1,2}$ , Yufu Ning $^{1,2}$ and Yang Xun

Abstract: In this study, one-dimensional grating coupler on single-crystal lithium niobate thin film (lithium niobate on insulator, LNOI) that also served as a polarization splitter was designed. The coupler could separate both orthogonal polarization states into two opposite directions while coupled light from a standard single-mode fiber to a waveguide on LNOI at the same time. Using segmented and apodized designing, the peak coupling efficiencies (CEs) around telecommunication wavelength $1550\mathrm{nm}$ for fundamental TE and TM modes of $-2.82\mathrm{dB}$ and $-2.83\mathrm{dB}$ , respectively, were achieved. The CEs could be optimized to $-1.97\mathrm{dB}$ and $-1.8\mathrm{dB}$ when a metal layer was added below the silicon dioxide layer.

Keywords: polarization splitter; grating coupler; lithium niobate; LNOI

![](dt=2026-03-16/ht=13/ab782afcf0d0430ce324f3200c2558dff2c68e2f8ba66ed5680e704db1cf4dfb.jpg)

Citation: Chen, Z.; Chen, L.; Meng, X.; Ning, Y.; Xun, Y. Polarization-Splitting Grating Coupler on Lithium Niobate Thin Film. Crystals 2024, 14, 226. https://doi.org/10.3390/cryst14030226

Academic Editor: Vladimir Chigrinov

Received: 3 February 2024

Revised: 23 February 2024

Accepted: 24 February 2024

Published: 27 February 2024

![](dt=2026-03-16/ht=13/33a91b6015f4806660a10daabb8f67880f683cd7eb7ed9df54af852a8ca3aa5e.jpg)

Copyright: © 2024 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).

# 1. Introduction

The grating coupler (GC) is an important optical device to couple light from fibers to micro/nano integrated waveguides. Different from end-face couplers, GCs that do not require facet polishing and can be freely placed on the waveguides bring size freedom and fabrication flexibility in coupling [1,2]. Waveguide GCs have been widely designed and studied on some useful platforms, such as silicon on insulator (SOI) [3], silicon nitride $(\mathrm{Si}_3\mathrm{N}_4)$ [4], indium phosphide (InP) [5], and single crystal lithium niobate thin film (lithium niobate on insulator, LNOI) [6].

Among them, LNOI is considered as the most promising platform due to its excellent nonlinear and electro-optical properties, as well as the wide transparency window in visible and infrared regions [7]. More attractive devices on LNOI have been designed and fabricated, such as the high-speed electro-optic modulator [8-10], high performance RF filter [11], acousto-optic modulator [12], and entangled quantum source [13,14]. The demand of GCs to couple light in or out of the platform increases with the growing research on LNOI.

Coupling efficiency (CE), bandwidth, and polarization independence are the key performance indices (KPIs) of GCs. Studies are generally concerned with improving the CE by adding top layer, loading bottom reflector, chirping period, apodizing filling factor, or forming some effective structures [15-20]. However, most of the GCs based on the Bragg diffracted effect are polarization dependent, as coupling of the transversal electric (TE) mode requires a lower grating period than transversal magnetic (TM) polarization.

The polarization dependence of GC limits the potential application in optical fiber communication networks, especially in which both orthogonal polarization states need to be transmitted and operated (such as the optical receiver). Polarization-splitting GC which can couple and split the orthogonal modes at the same time is a potential solution to this problem.

Polarization-splitting GCs such as two-dimensional polarization independence grating coupler and one-dimensional polarization-splitting GCs (which are easier to be fabricated) have been designed and studied on SOI [21-24]. In contrast, there are few reports concerned

#

crystals

MDPI

Crystals 2024, 14, 226. https://doi.org/10.3390/cryst14030226

https://www.mdpi.com/journal/crystals

with the polarization-splitting GCs on LNOI, with the exception of [25]. By using periodically arranged holes as grating cells, the two-dimensional grating GC can split different polarization with a coupling efficiency of $-7.2$ dB. However, studies about the polarization independence of GCs on LNOI have been reported. Using a metal-based plasmonic mode, polarization dependence of the grating coupler on x-cut LNOI has been effectively reduced, and the peak CE for TE and TM modes of $-3.56$ dB and $-4.08$ dB was reported [26].

Using circular holes as grating cells, a two-dimensional GC demonstrated CE of $-3.88$ dB in simulation for both TE and TM polarization on x-cut LNOI was reported [27]. Using silicon strips as the grating cells and silica as the upper layer, polarization independence GC with a coupling efficiency of $51\%$ on silicon hybrid LNOI platform was achieved [28]. Since GCs on LNOI are generally designed to couple only one polarization mode, it is still meaningful to study the polarization-independent and polarization-splitting GCs to provide a solution to the multi-polarization applications.

In this paper, a one-dimensional polarization-splitting grating coupler has been designed and optimized on z-cut LNOI platform that has the potential to be fabricated by one single etching step. The total length of the waveguide grating designed is less than $12\mu \mathrm{m}$ . Using segmented and apodized grating design, fundamental TE $(\mathrm{TE}_0)$ and TM $(\mathrm{TM}_0)$ polarization modes are selected, splitted, and guided into opposite directions by the same coupling configuration.

The peak coupling efficiencies of $-2.82\mathrm{dB}$ and $-2.83\mathrm{dB}$ with 3-dB-bandwidths of $73~\mathrm{nm}$ and $100~\mathrm{nm}$ for waveguide $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes are achieved. The parameters' fabrication tolerances are also studied. This polarization-splitting GC can benefit the polarization diversity system with an efficient coupler.

# 2. Design and Methods

The polarization-splitting grating coupler was designed on a LNOI platform. The schematic and wave-vector diagram of the GC was shown in Figure 1.

![](dt=2026-03-16/ht=13/d848b84501f4287ab53b6ce4ab1cf3a81f3f58390c19420561915f1b784376bd.jpg)

![](dt=2026-03-16/ht=13/6aa24088e5633f34156c47832f31204284ed8e6b989c872767e4657c95316919.jpg)

From bottom to top: lithium niobate (LN) substrate $(500\mu \mathrm{m})$ , silicon dioxide layer $(\sim 2\mu \mathrm{m})$ , LN thin film $(500\mathrm{nm})$ , and the single mode fibers (core diameter is $9\mu \mathrm{m}$ ). Fibers were fixed above the waveguide grating surface at a tilted angle deviated from the z direction. The grating structure was formed by several periodic LN teeth and air grooves which were partially etched in LN thin film waveguide.

The vertical position $z_{0}$ of fiber was defined as the placement from the grating surface, and the horizontal position $y_{0}$ of fiber was defined as the placement from the first air grove of port A. When light from the single mode fiber is diffracted by the grating, they will be separated into two beams of opposite directions: the forward port A for waveguide $\mathrm{TE}_0$ mode, and the backward port B for waveguide $\mathrm{TM}_0$ mode.

Crystals 2024, 14, 226

2 of 12

The polarization-splitting grating is essentially a diffracted Bragg grating. It satisfies the Bragg condition:

$$
\boldsymbol {\beta} _ {1} + \mathrm {m} \m
athbf {K} _ {\Lambda} = \boldsymbol {\beta} _ {2} \tag {1}
$$

where $m$ is the diffraction order, $K_{\Lambda}$ is the grating vector, and $\beta_{1}$ and $\beta_{2}$ are the wave vectors of the input light and waveguide mode. In $y$ direction, the values $\beta_{1} = \frac{2\pi}{\lambda_{0}} N_{\mathrm{cladding}} \sin \theta$ , $\beta_{2} = \frac{2\pi}{\lambda_{0}} N_{\mathrm{eff}}$ , and $K_{\Lambda} = 2\pi / \Lambda$ .

$\lambda_{0}$ is the free-space wavelength ( $\lambda_{0} = 1550 \, \mathrm{nm}$ ), $\Lambda$ is the period of the grating, $N_{\mathrm{cladding}}$ is the refractive index of the top layer on LN ( $N_{\mathrm{cladding}} = 1$ ), $\theta$ is the fiber tilted angle with respect to the normal direction of waveguide surface. $N_{\mathrm{eff}}$ denotes the effective index of waveguide mode, and is different for waveguide TE0 mode and TM0 mode, meaning that grating couplers are generally polarization-dependent.

The polarization dependence can be employed in splitting the different polarization modes. As schematically shown in Figure 1b, the grating diffracted the fiber mode to the waveguide TE mode in the forward direction with $m = 1$ but diffracted the fiber mode to the waveguide TM mode in the backward direction with $m = -1$ .

This difference introduced an offset of $\frac{4\pi}{\lambda_{0}} \sin \theta$ for the two modes according to Equation (1), so a grating coupler which can split the TE0 polarization mode and TM0 mode in opposite directions was worked out with the initial fiber angle $\theta_{0} = \arcsin \frac{N_{\mathrm{TE}} - N_{\mathrm{TM}}}{2}$ and $\Lambda_{0} = \frac{2\lambda_{0}}{N_{\mathrm{TE}} + N_{\mathrm{TM}}}$ . $N_{\mathrm{TE}}$ and $N_{\mathrm{TM}}$ denote the refractive indices of waveguide TE0 and TM0 modes.

Lumerical mode solutions were employed to find the fundamental $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes by solving Maxwell's equations on a cross-section mesh, and the finite different algorithm was used for meshing the structure. Maxwell's equations were then formulated into a matrix eigenvalue problem, so the effective indices $\mathrm{N_{TE}}$ and $\mathrm{N_{TM}}$ and the mode profile of the two polarization modes could be gained after solving. Finite-difference time-domain simulation method (FDTD) was employed in the grating coupler's design, simulation, and optimization.

To save simulation time, two-dimensional FDTD was applied instead of three-dimensional FDTD, as the typical waveguide width ( $\sim 12~\mu \mathrm{m}$ ) is much bigger than its height ( $0.5~\mu \mathrm{m}$ ) [2]. Perfectly matched layer (PML) boundary conditions were applied to absorb the electromagnetic energy incident to the boundary to avoid interference with the fields inside. FDTD solutions generated a rectangular, Cartesian-style mesh.

A lower mesh accuracy of 2 with 10 mesh points per wavelength was employed in the initial optimization for quick optimization, and a higher mesh accuracy of 4 with 18 mesh points per wavelength was employed for the convergence test. Monitors were placed in all sides of the grating, so the upward optical power, the downward optical power, the forward optical power, and the backward optical power could be detected and analyzed. By exploiting the Fourier transforms, normalized transmission and CE were gained.

In simulation, the ordinary and extraordinary refractive indices of LN were set to be 2.2112 and 2.138, and the refractive index of silicon dioxide was set to be 1.46 [29,30]. Firstly, a uniform grating coupler for coupling both $\mathrm{TE}_0$ and $\mathrm{TM}_0$ polarization modes was designed and optimized to gain optimal parameters such as period, filling factor, and fiber position. Secondly, the optimal uniform GC was segmented into three parts of which the period and filling factor were modulated to achieve better CEs. Lastly, the fabrication tolerance of parameters was systematically simulated and analyzed.

# 3. Results and Discussion

The initial optimizations were performed in uniform grating coupler with the calculated $\theta$ of $6^{\circ}$ and period of $823\mathrm{nm}$ for both waveguide $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes by Bragg condition, respectively. Parameters such as $\Lambda$ , etch depth (d), filling factor (the ratio of the air groove width to the period, FF), $\theta$ , and fiber position ( $\mathbf{y}_0$ and $\mathbf{z}_0$ ) were all variables, and were optimized and studied to achieve a good CE and polarization splitting.

The relative position of fiber and grating was crucial to the coupling because it played an important role in the diffracted power distribution and mode-matching situation. For the sake of simplicity, $\mathbf{z}_0$ was fixed at $10.1\mu \mathrm{m}$ . The CEs of $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes with different fiber horizontal position at wavelength $1550\mathrm{nm}$ were shown in Figure 2. Different numbers of

Crystals 2024, 14, 226

3 of 12

grating period $N$ were employed in the optimization. CEs of waveguide $\mathrm{TE}_0$ mode and $\mathrm{TM}_0$ were demonstrated in dashed and solid curves, respectively.

![](dt=2026-03-16/ht=13/844e3f743c40b7fa79c8c177a0d02ce0f0d5585c19a375b07a2384c3f4868040.jpg)

From Figure 2, the maximum CE of $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes increased slightly with the increase of N, and the maximum CEs are $-3\mathrm{dB}$ and $-4\mathrm{dB}$ when $\Lambda = 920\mathrm{nm}$ , $d = 300\mathrm{nm}$ , $N = 21$ , but a different $\mathbf{y}_0$ , respectively. The $\mathbf{y}_0$ corresponding to the peak CE was defined as $\mathbf{y}_{0\mathrm{m}}$ . The bigger the N, the bigger the $\mathbf{y}_{0\mathrm{m}}$ difference of the two modes.

For TE mode, the $\mathbf{y}_{0\mathrm{m}}$ varies little (around $-6.5\mu \mathrm{m}$ ( $\pm 1\mu \mathrm{m}$ )), but for $\mathrm{TM}_0$ mode, the $\mathbf{y}_{0\mathrm{m}}$ varied widely (from $-18\mu \mathrm{m}$ to $-6.8\mu \mathrm{m}$ ). This might be because that the horizontal position of the first air groove for $\mathrm{TM}_0$ mode was greatly dependent on the N. The optimal $\mathbf{y}_{0\mathrm{m}}$ tended to keep close to their own first air groove of $\mathrm{TE}_0$ mode (Port A) and $\mathrm{TM}_0$ mode (Port B), respectively.

So, N and $\mathbf{y}_{0\mathrm{m}}$ should be properly adjusted to keep an identical grating configuration for both $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes at the expense of CE. As shown in Figure 2, the best CE for both $\mathrm{TE}_0$ and $\mathrm{TM}_0$ mode was about $-4.2\mathrm{dB}$ when $\mathbf{y}_{0\mathrm{m}} = -6.8\mu \mathrm{m}$ and $N = 9$ (red arrow). The CE decreased with the increase of N, dropping to $-8.3\mathrm{dB}$ when $N = 21$ (black arrow) and $\mathbf{y}_{0\mathrm{m}} = -11.8\mu \mathrm{m}$ .

When considering improving the CE and lowering the backward transmission, the next optimizations were performed at a compromise of $N = 13$ with a CE of $-4.82\mathrm{dB}$ for the two modes.

By diffracting, a uniform grating coupler might yield an exponentially decaying power field distribution along the $y$ direction, so there existed a certain mode mismatch between the input mode field and the waveguide modes field. The CE could be further improved by chirping the period and apodizing the filling factor to form a better mode matching with the waveguide mode. Regarding the basics of uniform grating, the grating region was segmented into three parts named $\mathrm{L}_1$ , $\mathrm{L}_2$ , and $\mathrm{L}_3$ regions (Figure 1a).

$\mathrm{L}_1$ region, which contained five grating teeth and air grooves numbered 1-5, were chirped and apodized for the mode-matching of waveguide $\mathrm{TE}_0$ mode. $\mathrm{L}_2$ region, which contained a uniform configuration which had four uniform grating teeth and air grooves numbered 6-9, was designed to guarantee coupling strength. $\mathrm{L}_3$ region, which contained four grating tee
th and five air grooves numbered 10-14, were apodized for the mode-matching of waveguide $\mathrm{TM}_0$ modes. The dimensions of each grating period and groove width is shown in Table 1.

Crystals 2024, 14, 226

4 of 12

Table 1. Dimensions of each grating period and groove width.

![](dt=2026-03-16/ht=13/44ca80d9edeed7b101558b5f903042c5136ad358bc3b463b8314165ba9b65561.jpg)

<table><tr><td>Grating Number</td><td>Period (nm)</td><td>Groove Width (nm)</td></tr><tr><td>1</td><td>825</td><td>41.25</td></tr><tr><td>2</td><td>844</td><td>101.28</td></tr><tr><td>3</td><td>863</td><td>163.97</td></tr><tr><td>4</td><td>882</td><td>229.32</td></tr><tr><td>5</td><td>901</td><td>297.33</td></tr><tr><td>6</td><td>920</td><td>368</td></tr><tr><td>7</td><td>920</td><td>368</td></tr><tr><td>8</td><td>920</td><td>368</td></tr><tr><td>9</td><td>920</td><td>368</td></tr><tr><td>10</td><td>920</td><td>303.6</td></tr><tr><td>11</td><td>920</td><td>239.2</td></tr><tr><td>12</td><td>920</td><td>174.8</td></tr><tr><td>13</td><td>920</td><td>110.4</td></tr><tr><td>14</td><td>/</td><td>46</td></tr></table>

As shown in Table 1, the $\Lambda$ and groove width in $\mathrm{L}_1$ and $\mathrm{L}_3$ part were varied. In $\mathrm{L}_1$ , the parameters of period and filling factor gradually increased from 1-5 for $\mathrm{TE}_0$ . Also, the variation of filling factor ( $\delta \mathrm{FF}$ ) and variation of period ( $\delta \Lambda$ ) were 0.07 and $19~\mathrm{nm}$ , respectively. In $\mathrm{L}_3$ , the filling factor was gradually decreased from 10-14, and $\delta \mathrm{FF}$ was 0.07.

The gradually increased or decreased groove width in $\mathrm{L}_1$ and $\mathrm{L}_3$ regions modified the exponentially decayed power distribution to a Gaussian-like field distribution. The CE for both $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes could be improved due to the improved mode field matching between the grating diffraction field distribution and waveguide modes field distribution. The details of the chirped and apodized GC could be seen in our primary work [31]. In addition, the asymmetric configuration of $\mathrm{L}_1$ and $\mathrm{L}_3$ was to reduce the opposite transmission.

The peak CEs for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes with wavelength are shown in Figure 3.

![](dt=2026-03-16/ht=13/47000db80803a88dc364297a173485fe570ed26b3519672acbb76559cf0cf584.jpg)

It could be seen that CEs of $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes were improved to $-2.83\mathrm{dB}$ and $-2.82\mathrm{dB}$ with 3-dB-bandwidths of $73~\mathrm{nm}$ and $100~\mathrm{nm}$ at wavelength $1550~\mathrm{nm}$ , respectively. An increase of about $2\mathrm{dB}$ in CE was gained when the segmented and apodized grating configuration was applied. The polarization-splitting ratio was defined as the CE of $\mathrm{TE}_0$

Crystals 2024, 14, 226

5 of 12

polarization over the CE of $\mathrm{TM_0}$ polarization. The difference of CEs for the two modes was small, indicating that the polarization-splitting GC had a good polarization independence and a good polarization-splitting ratio of 1:1. Along with that, if the gratings numbered 1 and 14 were deleted on consideration of the fabrication process, a decrease of only $0.02\mathrm{dB}$ was introduced in the CE.

Grating parameters such as the thickness of LN film (h), $\Lambda$ , FF, and d played import roles in the CE and splitting ability. Their fabrication tolerances were important to the performance of coupler. CEs at different parameters' deviations for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes were shown in Figure 4. The CEs of $\mathrm{TE}_0$ mode and $\mathrm{TM}_0$ mode were demonstrated in dashed and solid curves, respectively. The optimal CE at wavelength $1550\mathrm{nm}$ was marked by an orange dot.

![](dt=2026-03-16/ht=13/b17c644d682bb8951368ffb4eabbbe5962595bbd3b2b1d13b141f13a3e209828.jpg)

![](dt=2026-03-16/ht=13/46aa73e5e1c5acda33689df24f561349aab2c18d3d42d076d1f6dc9354f17093.jpg)

![](dt=2026-03-16/ht=13/c90c748ed7c1ce7d87fd07bb4c507a126903c25971fcbbdfd65ac0da80a7eb15.jpg)

![](dt=2026-03-16/ht=13/624dd9059bbdd8831dd37e6bf51a6ac1c1dff62c2fac91c4cedb0513942ec24d.jpg)

In Figure 4a, grating period varied around the optimal value of $920\mathrm{nm}$ with $\pm 20\mathrm{nm}$ when other parameters were kept at their optimal values. The red, orange, blue, and black lines denoted the CEs when the periods were $900\mathrm{nm}$ $(-20\mathrm{nm})$ , $910\mathrm{nm}$ $(-10\mathrm{nm})$ , $930\mathrm{nm}$ $(+10\mathrm{nm})$ , and $940\mathrm{nm}$ $(+20\mathrm{nm})$ . A deviation of period made the peak wavelength move, with the greater the deviation of period, the greater the peak wavelength shift.

The shift made the intersection point of the two CE curves $\left(\mathrm{TE}_0$ and $\mathrm{TM}_0\right)$ at the same period move down or move out of the range $1500\mathrm{nm}$ to $1600\mathrm{nm}$ . Deviations of $-10\mathrm{nm}$ and $-20\mathrm{nm}$ introduced decreases of about $0.2\mathrm{dB}$ and $0.28\mathrm{dB}$ in CEs, respectively. Deviations of $+10\mathrm{nm}$ and $+20\mathrm{nm}$ made the intersection point move out of the $1500\mathrm{nm}$ to $1600\mathrm{nm}$ range. Therefore, it was difficult to keep the polarization-splitting ratio at 1:1 with a good CE.

In addition, at a wavelength of $1550\mathrm{nm}$ , the CEs for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes were $-4.4\mathrm{dB} / -3.62\mathrm{dB}$ , $-2.84\mathrm{dB} / -3.06\mathrm{dB}$ , $-3.21\mathrm{dB} / -2.89\mathrm{dB}$ , and $-3.34\mathrm{dB} / -4.32\mathrm{dB}$ when the periods were set to $900\mathrm{nm}$ , $910\mathrm{nm}$ , $930\mathrm{nm}$ , and $940\mathrm{nm}$ and the polarization-splitting ratios were 1:1.19, 1:0.95, 1:1.08, and 1:0.8, respectively. The greater the deviation from 1:1, the worse the polarization independence at wavelength $1550\mathrm{nm}$ of the coupler.

It could be seen that deviations of $\pm 20\mathrm{nm}$ introduced a maximum extra loss of about $1.6\mathrm{dB}$

Crystals 2024, 14, 226

6 of 12

and $1.5\mathrm{dB}$ for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes at wavelength $1550~\mathrm{nm}$ . The above results suggested that the polarization-splitting GC was sensitive to the parameter of grating period which determined the coupling strength. One can vary this parameter to change the CE and polarization-splitting ratio. The deviation of period should be tightly controlled to achieve a good polarization-splitting ratio and CE.

In Figure 4b, the etch depth varied around the optimal value of $300\mathrm{nm}$ with $\pm 20\mathrm{nm}$ when other parameters were kept at their optimal values. The red, orange, blue, and black lines denote the CEs when the etch depths d were $280\mathrm{nm}$ $(-20\mathrm{nm})$ , $290\mathrm{nm}$ $(-10\mathrm{nm})$ , $310\mathrm{nm}$ $(+10\mathrm{nm})$ , and $320\mathrm{nm}$ $(+20\mathrm{nm})$ , respectively. A deviation of d made the peak wavelength move, a positive deviation resulted in a blue-shift, and a negative deviation resulted in a red-shift for both modes.

The greater the deviation of d, the greater the peak wavelength shift. Although parts of the peak CEs increased (the black and blue curves for $\mathrm{TE}_0$ mode), the shift also made the intersection point of the two CE curves ( $\mathrm{TE}_0$ and $\mathrm{TM}_0$ ) at the same etch depth move down. Deviations of $\pm 10\mathrm{nm}$ and $\pm 20\mathrm{nm}$ resulted in a decrease of CE at intersection points less than $0.1\mathrm{dB}$ and $0.8\mathrm{dB}$ , respectively.
At a wavelength of $1550\mathrm{nm}$ , the CEs for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes were $-3.42\mathrm{dB} / -2.9\mathrm{dB}$ , $-3.02\mathrm{dB} / -2.86\mathrm{dB}$ , $-2.72\mathrm{dB} / -2.83\mathrm{dB}$ , and $-2.82\mathrm{dB} / -2.96\mathrm{dB}$ when the h was $280\mathrm{nm}$ , $290\mathrm{nm}$ , $310\mathrm{nm}$ , and $320\mathrm{nm}$ , meaning that the polarization-splitting ratios were 1:1.23, 1:1.04, 1:0.98, and 1:1.03, respectively.

Deviations of $\pm 20\mathrm{nm}$ only introduced a maximum extra loss about $0.37\mathrm{dB}$ and $0.13\mathrm{dB}$ for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes at wavelength $1550\mathrm{nm}$ . The above results suggested that the polarization-splitting GC had a good fabrication tolerance of etching depth. A deviation of $\pm 20\mathrm{nm}$ was acceptable.

In Figure 4c, the h varied around the optimal value of $500\mathrm{nm}$ with $\pm 40\mathrm{nm}$ when other parameters were kept at their optimal values. The red, orange, blue, and black lines denote the CEs when the h was $460\mathrm{nm}$ $(-40\mathrm{nm})$ , $480\mathrm{nm}$ $(-20\mathrm{nm})$ , $520\mathrm{nm}$ $(+20\mathrm{nm})$ and $540\mathrm{nm}$ $(+40\mathrm{nm})$ , respectively. A deviation of h made the peak wavelength move, a positive deviation resulted in a red-shift, and a negative deviation resulted in a blueshift for the two modes.

The greater the deviation of h, the greater the peak wavelength shift. The shift made the intersection point of the two CE curves $\mathrm{(TE_0}$ and $\mathrm{TM}_0)$ at the same h move down or move out of the range $1500\mathrm{nm}$ to $1600\mathrm{nm}$ . Deviations of $-40\mathrm{nm}$ and $-20\mathrm{nm}$ resulted in a decrease of CE at intersection point about $0.34\mathrm{dB}$ and $0.15\mathrm{dB}$ , respectively.

In addition, at a wavelength of $1550\mathrm{nm}$ , the CEs for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes were $-6\mathrm{dB} / -4.2\mathrm{dB}$ , $-3.16\mathrm{dB} / -3.17\mathrm{dB}$ , $-3.59\mathrm{dB} / -2.92\mathrm{dB}$ , and $-5.1\mathrm{dB} / -3.42\mathrm{dB}$ when the h was $460\mathrm{nm}$ , $480\mathrm{nm}$ , $520\mathrm{nm}$ , and $540\mathrm{nm}$ , so the polarization-splitting ratio were 1:1.52, 1:1.002, 1:1.17, and 1:1.47, respectively.

Deviations of $\pm 20\mathrm{nm}$ introduced a maximum extra loss of about $0.76\mathrm{dB}$ and $0.34\mathrm{dB}$ for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes at wavelength $1550\mathrm{nm}$ , respectively. Deviations of $\pm 40\mathrm{nm}$ introduced a maximum extra loss of about $3.2\mathrm{dB}$ and $2.3\mathrm{dB}$ for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes at wavelength $1550\mathrm{nm}$ . The polarization-splitting GC was sensitive to the parameter h which could change the refractive indexes of the waveguide mode.

To keep a good CE and polarization-splitting ratio, the deviation of h should be controlled in $\pm 20\mathrm{nm}$ .

In Figure 4d, the filling factor varied around the optimal value of 0.4 with $\pm 0.04$ when other parameters were kept at their optimal values. The red, orange, blue, and black lines denote the CEs when the FF was 0.36 $(-0.04)$ , 0.38 $(-0.02)$ , 0.42 $(+0.02)$ , and 0.44 $(+0.04)$ . A deviation of the FF makes the peak wavelength move. A positive deviation results in a blue-shift, and a negative deviation results in a red-shift for both modes. The greater the deviation of FF, the greater the peak wavelength shift.

Although parts of the peak CEs were increased (the blue curve for $\mathrm{TE}_0$ mode), the shift also made the intersection point of the two CE curves ( $\mathrm{TE}_0$ and $\mathrm{TM}_0$ ) at the same FF move down. Deviations of $\pm 0.02$ resulted in a decrease of CE at interaction point less than $0.07\mathrm{dB}$ and $0.7\mathrm{dB}$ , respectively. It indicated that a CE of $-2.91\mathrm{dB}$ for the two modes with a polarization-splitting ratio of 1:1 was obtained when the deviation was controlled in $\pm 0.02$ .

At wavelength $1550\mathrm{nm}$ , the CEs for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes were $-3.36\mathrm{dB} / -3.1\mathrm{dB}$ , $-3.02\mathrm{dB} / -2.9\mathrm{dB}$ , $-2.91\mathrm{dB} / -2.8\mathrm{dB}$ , and $-2.97\mathrm{dB} / -3.14\mathrm{dB}$ when the FF was 0.36, 0.38, 0.42, and 0.44, so the polarization-splitting

Crystals 2024, 14, 226

7 of 12

ratios were 1:1.06, 1:1.03, 1:1.03, and 1:0.96, respectively. Since a narrow FF range near 0.4 was investigated, the polarization-splitting GC showed less-insensitivity to the FF which played an important role in the field distribution. A deviation of $\pm 0.04$ was acceptable.

For the segmented, chirped, and apodized GC, $\delta \Lambda$ and $\delta \mathrm{FF}$ were especially important for the coupling and splitting performance. Figure 5a,b gave the CEs at different $\delta \Lambda$ and $\delta \mathrm{FF}$ for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes. The CEs of $\mathrm{TE}_0$ mode and $\mathrm{TM}_0$ mode were demonstrated in dashed and solid curves. The optimal CE at wavelength $1550\mathrm{nm}$ was marked by an orange dot.

![](dt=2026-03-16/ht=13/14a0afbe8d0fc2876e33158afcabc5ea029df2b78204675a9c47d061f33f7df7.jpg)

![](dt=2026-03-16/ht=13/431d52e5bd6a4b5cd477a3d4b5b8428d2e151d2990cd927e993ecfa009d5e5a6.jpg)

In Figure 5a, the red, orange, blue, and black lines denoted the CEs when $\delta \Lambda$ were $9\mathrm{nm}$ $(-10\mathrm{nm})$ , $14\mathrm{nm}$ $(-5\mathrm{nm})$ , $24\mathrm{(+5nm)}$ , and $29(+10\mathrm{nm})$ . Deviations of $\delta \Lambda$ made the peak wavelength move. The greater the deviation of $\delta \Lambda$ , the greater the peak wavelength shift.

The shift made the intersection point of the two CE curves ( $\mathrm{TE}_0$ and $\mathrm{TM}_0$ ) at the same period move down when $\delta \Lambda$ were $9\mathrm{nm}$ and $14\mathrm{nm}$ , while out of the range from $1500\mathrm{nm}$ to $1600\mathrm{nm}$ when $\delta \Lambda$ were $24\mathrm{nm}$ and $29\mathrm{nm}$ . The decreased CE at the intersection points were $-3.05\mathrm{dB}$ and $-2.9\mathrm{dB}$ when $\delta \Lambda$ were $9\mathrm{nm}$ and $14\mathrm{nm}$ .

This resulted in an extra loss of $0.2\mathrm{dB}$ and $0.17\mathrm{dB}$ in CE for the two modes from the optimal value of $-2.83\mathrm{dB}$ , respectively. There was no intersection when $\delta \Lambda$ were at $24\mathrm{nm}$ and $29\mathrm{nm}$ , as a splitting ratio of 1:1 would never be achieved in that case.

At wavelength $1550\mathrm{nm}$ , the CEs for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes were $-3.61\mathrm{dB} / -3.05\mathrm{dB}, -2.83\mathrm{dB} / -2.83\mathrm{dB}, -3.44\mathrm{dB} / -2.79\mathrm{dB},$ and $-3.82\mathrm{dB} / -2.8\mathrm{dB}$ when $\delta \Lambda$ were $9\mathrm{nm}$ , $14\mathrm{nm}$ , $24\mathrm{nm}$ , and $29\mathrm{nm}$ . The polarization-splitting ratios were 1:1.14, 1:1, 1:1.16, and 1:1.26, respectively.

Deviations of $\pm 10\mathrm{nm}$ introduced a maximum extra loss of about $1\mathrm{dB}$ and $0.2\mathrm{dB}$ for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes at wavelength $1550\mathrm{nm}$ . The worse splitting ratio was obtained when a positive deviation of $\delta \Lambda$ which resulted in smaller periods and groove widths in $\mathbf{L}_1$ and $\mathbf{L}_3$ region and could not modify the exponentially decaying distribution of field very well in a limited number of periods was applied.

In Figure 5b, the red, orange, blue, and black lines denoted the CEs when $\delta \mathrm{FF}$ were 0.03 $(-0.04)$ , 0.04 $(-0.03)$ , 0.05 $(-0.02)$ , and 0.06 $(-0.01)$ , respectively. A deviation of $\delta \mathrm{FF}$ made the peak wavelength move. A positive deviation resulted in a blue-shift, and a negative deviation resulted in a red-shift for both modes. The greater the deviation of $\delta \mathrm{FF}$ , the greater the peak wavelength shift. The shift made the intersection
points of the two CE curves all move out of the wavelength range from $1500\mathrm{nm}$ to $1600\mathrm{nm}$ .

It indicated that a splitting ratio of 1:1 would never be obtained when $\delta \mathrm{FF}$ was deviated from 0.07. At wavelength $1550\mathrm{nm}$ , the CEs for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes were $-3.61\mathrm{dB} / -3.05\mathrm{dB}$ , $-2.83\mathrm{dB} / -2.83\mathrm{dB}$ , $-3.44\mathrm{dB} / -2.79\mathrm{dB}$ , and $-3.82\mathrm{dB} / -2.8\mathrm{dB}$ when $\delta \Lambda$ were $9\mathrm{nm}$ , $14\mathrm{nm}$ , $24\mathrm{nm}$ , and $29\mathrm{nm}$ , and the polarization-splitting ratio were 1:1.14, 1:1, 1:1.16, and 1:1.26, respectively.

The polarization-splitting GC was highly sensitive to the parameter of $\delta \mathrm{FF}$ which was crucial to

Crystals 2024, 14, 226

8 of 12

the modified field distribution. The parameter $\delta \mathrm{FF}$ should be under precise control since a small deviation of $\delta \mathrm{FF}$ could destroy the mode field matching condition.

The impact of a deviation of the optimal fiber angle of $6^{\circ}$ on the CE and polarization-splitting ability was also studied. CE at different fiber angles was shown in Figure 6. The red and black lines denoted the $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes, respectively. For $\mathrm{TE}_0$ mode, the deviation of fiber angle made the peak wavelength move, a positive deviation resulted in a blue-shift, and a negative deviation resulted in a red-shift. The shift was exactly the opposite for $\mathrm{TM}_0$ mode.

A positive deviation resulted in a red-shift, and a negative deviation resulted in a blue-shift. The opposite shift of the two modes made polarization-splitting with a high CE much more difficult. For the two modes, the greater the deviation of fiber angle, the greater the peak wavelength shift. Take $\theta = 9^{\circ}$ for example—the peak wavelength shifts to $1529\mathrm{nm}$ and $1576\mathrm{nm}$ for $\mathrm{TE}_0$ and $\mathrm{TM}_0$ modes, respectively. The deviation of $3^{\circ}$ introduced a shift of $47\mathrm{nm}$ in peak wavelength.

Deviation of fiber angle also resulted in change of peak CE. Except for $\theta = 5^{\circ}$ , the peak CEs all decreased with the deviation, and the bigger deviation of fiber angle, the bigger decrease of peak CE. As shown in Figure 4, the CE at intersection wavelength for both $\mathrm{TE}_0$ and $\mathrm{TM}_0$ mode had a decrease of about 1 dB when the angle deviation equaled to $\pm 3^{\circ}$ , respectively. The fiber angle played an important role in polarization splitting and CE, and the deviation should be controlled in $\pm 3^{\circ}$ to achieve a good polarization splitting and CE.

![](dt=2026-03-16/ht=13/44a68eb769a0f603a0f06f27ca7695e8247e790bf1427889f795b188e152e506.jpg)

The polarization-splitting grating coupler can also be used as polarization beam combiner in the output coupling. Figure 7 gave the $\mathrm{E_x}$ -field profile when used as an output coupler. Waveguide $\mathrm{TE_0}$ and $\mathrm{TM_0}$ modes were excited at port A and port B simultaneously, and were both diffracted by the grating and then combined with the upper fiber. The total transmission was $-2.72\mathrm{dB}$ , which contained $45\%$ $\mathrm{TE_0}$ and $55\%$ $\mathrm{TM_0}$ polarization.

The ratio of $\mathrm{TE_0}$ and $\mathrm{TM_0}$ polarization was about 1:1.22, which is a decrease from the polarization-splitting ratio of 1:1. This might be attributed to the interaction of the two beams in the output coupling which did not happen in the input coupling. At the input coupling, $\mathrm{TE_0}$ and $\mathrm{TM_0}$ polarization lights were employed separately, but were excited simultaneously at the output coupling.

Crystals 2024, 14, 226

9 of 12

![](dt=2026-03-16/ht=13/491dde074a59a59944ef5f754682c79f5217c55aed488499e200e8f9079b7668.jpg)

# 4. Conclusions

In conclusion, this study designed and optimized a multifunctional grating coupler with polarization-splitting ability for coupling between waveguide on LNOI and single-mode fibers. Using segmented and apodized grating structure, peak coupling efficiencies of $-2.82\mathrm{dB}$ and $-2.83\mathrm{dB}$ at a wavelength of $1550\mathrm{nm}$ with a 3-dB-bandwidth of $73\mathrm{nm}$ and $100\mathrm{nm}$ for fundamental TE and TM polarization mode were achieved.

The fabrication tolerance of parameters such as grating period, etch depth, thickness of lithium niobate thin film, filling factor, variation of period, variation of filling factor, and fiber angle were studied. Results showed that the deviations from optimal values all resulted in peak wavelength shift, decrease of CE at intersection point (polarization-splitting ratio was kept at 1:1), and decrease of polarization independence at wavelength $1550\mathrm{nm}$ .

In addition, the polarization-splitting GC had good fabrication tolerance of etch depth and filling factor in a variation range of $\pm 20\mathrm{nm}$ and $\pm 0.04$ , respectively. The coupler was highly sensitive to the period, the thickness of lithium niobate thin film, the variation of filling factor, and the fiber angle, respectively. One could adjust the splitting ratio by varying them. The coupler can also be used as a polarization beam combiner, and was meaningful to the polarization diversity system.

Author Contributions: Conceptualization, Z.C.; formal analysis, L.C.; funding acquisition, X.M.; methodology, Y.N.; software, Y.X.; validation, L.C.; writing—original draft, Z.C.; writing—review and editing, Z.C. and X.M. All authors have read and agreed to the published version of the manuscript.

Funding: This research was funded by the Youth Innovation Science and Technology Support Program of Universities in Shandong Province, China, grant number 2021KJ082.

Data Availability Statement: The original contributions presented in the study are included in the article, further inquiries can be directed to the corresponding author.

Acknowledgments: The authors also thank Wenyan Sun (Shandong Youth University of Political Science) for her help in chart-processing.

Conflicts of Interest: The authors declare no conflicts of interest.

# References

1. Taillaert, D.; Bogaerts, W.; Bienstman, P.; Krauss, T.F.; Van Daele, P.; Moerman, I.; De Mesel, K.; Baets, R. An out-of-plane grating coupler for efficient butt-coupling between compact planar waveguides and single-mode fibers. IEEE J. Quantum Elect. 2002, 38, 949-955. [CrossRef]

Crystals 2024, 14, 226

10 of 12

Crystals 2024, 14, 226

11 of 12

Disclaimer/Publisher's Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.

Crystals 2024, 14, 226

12 of 12