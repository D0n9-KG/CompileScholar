# PAPER • OPEN ACCESS

# Optical sorting by trajectory tracking with high sensitivity near the exceptional points

To cite this article: LiYong Cui et al 2023 New J. Phys. 25 093048

View the article online for updates and enhancements.

# You may also like

\- Validation of the current and pressure coupling schemes with nonlinear simulations of TAE and analysis on the linear stability of tearing mode in the presence of energetic particles  
Haowei ZHANG, , Zhiwei MA et al.

A review of progress in the physics of open quantum systems: theory and experiment I Rotter and J P Bird

A tale of two kinds of exceptional point in a hydrogen molecule Himadri Barman and Suriyaa Valliapan

New Journal of Physics

The open access journal at the forefront of physics

Deutsche Physikalische Gesellschaft

Φ

DPG

IOP Institute of Physics

This content was downloaded from IP address 104.161.80.69 on 18/11/2023 at 03:49

# New Journal of Physics

The open access journal at the forefront of physics

Deutsche Physikalische Gesellschaft

IOP Institute of Physics

![](dt=2026-06-11/ht=03/10da9b1969a366288593ebacbdde30dedcf258cc3be56b833cc8b56fae6d774a.jpg)

DPG

Published in partnership

with: Deutsche Physikalische

Gesellschaft and the Institute of Physics

![](dt=2026-06-11/ht=03/bf3381dbc3ac3b48433b685b608483cca77f4d12137d75c23f0d94730eabbdcf.jpg)

CrossMark

# OPEN ACCESS

# PAPER

# Optical sorting by trajectory tracking with high sensitivity near the exceptional points

RECEIVED

30 June 2023

REVISED

29 August 2023

ACCEPTED FOR PUBLICATION

5 September 2023

PUBLISHED

27 September 2023

Original content from this work may be used under the terms of the Creative Commons Attribution 4.0 licence.

Any further distribution of this work must maintain attribution to the author(s) and the title of the work, journal citation and DOI.

![](dt=2026-06-11/ht=03/65158309b704dea58abb36f33a0d3f3fc68a8594e1f5ed8ea1bec4bbeafde71b.jpg)

# LiYong Cui $^{1,2}$ , Song Liu $^{1,2}$ and Neng Wang $^{3,*}$

# E-mail: nwang17@szu.edu.cn

Keywords: exceptional point, optical sorting, trajectory tracking

Supplementary material for this article is available online

# Abstract

Exceptional points (EPs) in non-Hermitian systems embody abundant new physics and trigger various novel applications. In the optical force system, the motion of a particle near its equilibrium position is determined by the optical force stiffness matrix (OFSM), which is inherently non-Hermitian when the particle is illuminated by vortex beams.

In this study, by exploiting the rapid variations in eigenvalues and the characteristics of particle motion near EPs of the OFSM, we propose a method to sort particles with subtle differences in their radii or refractive indices based on their trajectories in air. We demonstrate that the trajectory of a particle with parameters slightly larger than those corresponding to certain EPs closely resembles an ellipse. The increase in the major axis of the ellipse can be several orders of magnitude larger than the increase in particle radius.

Furthermore, even a slight change in the refractive index can not only significantly alter the size of the ellipse but also rotate its orientation angle. Hence, particles with subtle differences can be distinguished by observing the significant disparities in their trajectories. This approach holds promise as a technique for the precise separation of micro and nanoscale particles.

# 1. Introduction

Exceptional points (EPs) are special spectral degeneracies in non-Hermitian systems. They were first introduced in studying the perturbation of linear non-Hermitian operators [1], described by a general class of matrices $\mathrm{H}(x)$ parameterized by the complex variable $x$ . At EPs, two or more eigenvalues and corresponding eigenvectors coalesce and degenerate, leading to a breakdown of the diagonalization procedure [2-4].

Due to the abundance of nonconservative processes in optical systems, EPs have been the subject of theoretical investigations, experimental observations, and practical applications [5-23] in optics. The emergence of the EPs in optical system result in a variety of unusual effects, such as enhanced Sagnac effect [24], unidirectional invisibility [25, 26], coherent perfect absorption [27], the instability of large clusters [28] and enhanced sensitivity to perturbations [29-40].

In the field of optics, optical micromanipulation techniques have gained significant attention for their wide-ranging applications in trapping [5, 6] and cooling atoms [7], capturing microscopic particles and biological objects such as DNA and RNA [8, 9], and sorting microparticles. Optical sorting, in particular, holds great potential for the precise separation of micro and nanoparticles. Active [10-13] and passive [14-21] sorting methods are the two primary approaches used in optical sorting.

The former requires external recognition of particle properties, such as fluorescence, and subsequent deviation of particles using optical force. In contrast, the latter utilizes the intrinsic physical properties of the particles, such as radius and refractive index, to separate them based on their different behaviors in optical fields. Other methods such as

IOP Publishing

New J. Phys. 25 (2023) 093048

https://doi.org/10.1088/1367-2630/acf6da

© 2023 The Author(s). Published by IOP Publishing Ltd on behalf of the Institute of Physics and Deutsche Physikalische Gesellschaft

cross-type optical chromatography [22, 23] and static sorting [16, 41] have also been proposed for optical sorting.

The optical micromanipulation system typically exhibits non-Hermitian behavior when non-conservative optical forces are present. This non-Hermiticity can be manifested by the non-Hermitian optical force stiffness matrix (OFSM), which act as the Hamiltonian governing the particle motion in a dampingless environment within an optical field. Prior studies have investigated the stability of particle trapping and binding in various optical fields by analyzing the eigenvalues of the OFSM [28, 42-46]. However, there is limited research focused on the motion of particle in a damping environment when the eigenvalues become complex.

In this paper, by taking the advantage of the ultra-sensitivity of EPs against perturbations, we propose the method for sorting particles with subtle differences based on tracking their trajectories in vortex beams. Specifically, we consider a spherical particle illuminated by two counter-propagating linearly polarized Laguerre-Gaussian (LG) beams in air.

Through analytical and numerical investigations, we investigate the particle's motion near specific EPs of the OFSM and demonstrate that the particle exhibits periodic motion in an elliptical orbit around the equilibrium position when its parameters, such as radius and refractive index, are slightly larger than those corresponding to the EPs. The real parts of the eigenvalues of the OFSM represent the frequencies of the periodic motion, while the imaginary parts characterize the particle's ability to absorb energy from light.

Importantly, even a slight increase in the particle's parameters can lead to a significant rise in the imaginary parts of the eigenvalues near the EP, resulting in the particle moving in a much larger orbit. In addition, a small change in the refractive index can also alter the orientation of the elliptical orbit remarkably. Therefore, by observing the significant differences in their trajectories, we are able to separate particles with subtle differences.

This work enriches the study and application of non-Hermitian effects in optics and presents a novel pathway towards the precise separation of micro and nanoscale particles.

#
2. Results

# 2.1. The exceptional point and particle motion

To illustrate the basic ideas, let us consider that a spherical particle is located in the light field formed by two counterpropagating Laguerre-Gaussian beams (propagate along $\pm z$ directions). The particle is subject to a zero optical force at the beam center ( $x = y = z = 0$ ) and its motion is confined on the $z = 0$ plane (see figure 1). In a damping environment, the motion equation of the particle is given by

$$
m \Delta \ddot {\mathbf {r}} = \mathbf {F} _ {\perp} - \gamma \Delta \dot {\mathbf {r}}, \tag {1}
$$

where $m$ is the mass of the particle, $\Delta \mathbf{r}$ is the in-plane displacement vector of the particle to the beam center, $\mathbf{F}_{\perp}$ is the in-plane optical force, and $\gamma$ is the damping coefficient. When $\Delta \mathbf{r}$ is small, $\mathbf{F}_{\perp}$ is approximated as $\stackrel{\leftrightarrow}{K}\Delta \mathbf{r}$ , where $\stackrel{\leftrightarrow}{K}$ is the OFSM, which in an appropriate coordinate can be written as

$$
\stackrel {\leftrightarrow} {K} = \left[ \begin{array}{l l} \frac {\partial F _ {x}}{\partial x} & \frac {\partial F _ {y}}{\partial x} \\ \frac {\partial F _ {y}}{\partial x} & \frac {\partial F _ {y}}{\partial y} \end{array} \right] = \frac {1}{2} \left[ \begin{array}{c c} d + b & g \\ - g & d - b \end{array} \right], \tag {2}
$$

where $d = \nabla_{\perp} \cdot \mathbf{F}_{\perp}, g = \nabla_{\perp} \times \mathbf{F}_{\perp} \cdot \hat{z}$ and $b = \frac{\partial F_x}{\partial x} - \frac{\partial F_y}{\partial y}$ . If the particle suffers from a nonzero non-conservative optical force, the off-diagonal term $g = \nabla_{\perp} \times \mathbf{F}_{\perp} \cdot \hat{z} \neq 0$ [47, 48]. Thus, the non-Hermicity of $\stackrel{\leftrightarrow}{K}$ originates from the non-conservative optical force. The eigenvalues of $\stackrel{\leftrightarrow}{K}$ is given by

$$
\lambda_ {\pm} = \frac {1}{2} (d \pm \sqrt {b ^ {2} - g ^ {2}}). \tag {3}
$$

According to equation (3), an EP arises at $b = g$ . For $b > g$ , $\vec{K}$ is pseudo-Hermitian [49] with two real eigenvalues, while for $b < g$ , $\vec{K}$ is non-Hermitian with a pair of complex conjugate eigenvalues. For a Mie-sized particle, the real parts of $\lambda_{\pm}$ are usually negative. As the time-dependence of trajectory is $e^{\pm \sqrt{\lambda_{\pm}} t}$ , the particle will deviate from the beam center when $\lambda_{\pm}$ are complex. Therefore, the EP is also the critical point for whether the particle can be stably trapped in a damping-less environment.

However, when there is an ambient damping $(\gamma \neq 0)$ , the energy of the particle accumulated from the non-conservative optical force could be consumed by the damping force. In the appendix A, a comprehensive study of the particle motion in a damping environment is demonstrated. The particle will be stably confined to the beam center when $\gamma \geqslant \gamma_{\mathrm{c}} = \sqrt{m} |Im(\lambda_{\pm})| / \sqrt{|\mathrm{Re}(\lambda_{\pm})|}$ or move around the beam center following a closed path when $\gamma$ is slightly smaller than $\gamma_{\mathrm{c}}$ [42, 50]. In this work, the interest is focused

IOP Publishing

New J. Phys. 25 (2023) 093048

L Cui et al

2

![](dt=2026-06-11/ht=03/6bd4a10b3df6d9d68d310cd3d73334824c36b0feb15a28591bd08915fdd5d4f4.jpg)

on the case that $\gamma$ is slightly smaller than $\gamma_{\mathrm{c}}$ . By studying the trajectories of the particles, we can sort particles with subtle differences in the radius and refractive index.

# 2.2. The trajectories of the particle

In a damping environment, the trajectory of the particle in the vicinity of the beam center is given by

$$
\Delta \mathbf {r} (t) = \operatorname {R e} \left[ \mathbf {u} _ {+} \left(a _ {1} e ^ {- i \omega_ {1} t} + a _ {3} e ^ {- i \omega_ {3} t}\right) + \mathbf {u} _ {-} \left(a _ {2} e ^ {- i \omega_ {2} t} + a _ {4} e ^ {- i \omega_ {4} t}\right) \right], \tag {4}
$$

where $a_{m}$ $(m = 1,2,3,4)$ are the expansion coefficients, $\mathbf{u}_{\pm}$ are eigenvectors of the OFSM $\vec{K}$ , and

$$
\omega_ {1} = \frac {1}{2 m} i (- \gamma + \tau_ {r} + i \tau_ {i})
$$

$$
\omega_ {2} = \frac {1}{2 m} i (- \gamma + \tau_ {r} - i \tau_ {i}) \tag {5}
$$

$$
\omega_ {3} = \frac {1}{2 m} i (- \gamma - \tau_ {r} + i \tau_ {i})
$$

$$
\omega_ {4} = \frac {1}{2 m} i (- \gamma - \tau_ {r} - i \tau_ {i}),
$$

are the complex vibration frequencies with $\tau_r = \mathrm{Re}(\sqrt{4m\lambda_\pm + \gamma^2}),\tau_i = |\mathrm{Im}(\sqrt{4m\lambda_\pm + \gamma^2})|$ , see details in the appendix A. When the ambient damping $\gamma$ is slightly smaller than the critical damping $\gamma_{c}$ , the imaginary parts of the first two vibration frequencies $\mathrm{Im}(\omega_{1,2}) > 0$ , making the corresponding amplitudes $a_1e^{\mathrm{Im}(\omega_1)t}$ and $a_2e^{\mathrm{Im}(\omega_2)t}$ grows large and finally become dominant as $t$ increases. And when $|\Delta \mathbf{r}(t)|$ becomes not so

small, the optical force is no longer equal to $\vec{K} \cdot \Delta \mathbf{r}$ but a little smaller. In this case, the ambient damping just consumes the work done by the nonconservative optical force, making the particle mobbing in a stable orbit.

On the other hand, since the dominant term of the optical force is still $\vec{K} \cdot \Delta \mathbf{r}$ , the vibration frequencies of the stable orbit are very close to the real parts of $\omega_{1}, \omega_{2}$ . Therefore, for a large enough time $t$ , the trajectory of the particle can be approximately written as

$$
\Delta \mathbf {r} (t) \approx \operatorname {R e} \left[ a _ {1} \mathbf {u} _ {+} e ^ {i \tau_ {i} t / 2 m} + a _ {2} \mathbf {u} _ {-} e ^ {- i \tau_ {i} t / 2 m} \right], \tag {6}
$$

which describes an elliptical orbit motion of the particle (see Appendix B).

# 2.3. Sorting of particles with subtle difference in their radii

By using the finite difference method, the OFSM $\vec{K}$ is calculated [42-46], and the corresponding eigenvalues $\lambda_{\pm}$ against the particle radius is plotted in figure 2(a). The particle used here is a polystyrene sphere with refractive index $n = 1.57$ . The wavelength and topological order of the LG beam are $1.064\mu \mathrm{m}$ and $l = 1$ , respectively. Polystyrene particles [6, 15, 51] and LG beams [14, 15, 52-55] have been widely used in optical trapping experiments.

Due to the complex dependencies of $b$ and $g$ on the radius, there are multiple pairs of EPs (partially denoted in red dots) within the considered radius range, see figure 2(a). The critical damping $\gamma_{\mathrm{c}}$ and ambient damping $\gamma_{\mathrm{air}}$ with respect to the particle radius are shown by the blue solid and red dashed lines in figure 2(b), respectively. The beam power is $1\mathrm{W}$ when calculating $\gamma_{\mathrm{c}}$ .

The ambient air damping at standard atmospheric pressure is calculated by Stokes's Law $\gamma_{\mathrm{air}} = 6\pi \eta r$ , where $r$ is the particle radius and $\eta = 17.51\mathrm{pN} \cdot \mu s \cdot \mu \mathrm{m}^{-2}$ is the viscosity of air at room temperature [56, 57].

IOP Publishing

New J. Phys. 25 (2023) 093048

L Cui et al

3

![](dt=2026-06-11/ht=03/ff5a81d6f6534a89dcfbac86dae1b2f5cca8d6b44e0651194ccf7af2e2a6cf50.jpg)

![](dt=2026-06-11/ht=03/7b044707c2641a3964501d63557ee32f769f137d2e3b79d77bb2511fc058d171.jpg)

![](dt=2026-06-11/ht=03/ad1cdf4b173c9b81d585db275267bfc84c0d2b7450d9e50ff57215f0b5366ea4.jpg)

![](image)
.jpg)

![](dt=2026-06-11/ht=03/5d7741651ef8124c8a99fc2271ed637d5742d513d054866ad1e88e40eb7049d8.jpg)

![](dt=2026-06-11/ht=03/59d639d5487ff85066208dddb140584b140b5c50c594db928fccc784024efa0f.jpg)

As shown in figure 2(b), when the particle radius slightly exceeds the values indicated by the black arrows, the ambient damping $\gamma_{\mathrm{air}}$ will be slightly lower than the critical damping $\gamma_{\mathrm{c}}$ . Then, the particle will move in a stable orbit, as we have discussed in section 2.2. The smaller the ambient damping $\gamma_{\mathrm{air}}$ is compared to the critical damping $\gamma_{\mathrm{c}}$ , the greater kinetic energy the particle can acquire from the non-conservative optical force, allowing it to move in a higher orbit with larger major and minor axes.

Moreover, the increases in the major and minor axes are several orders of magnitude larger than the increase in particle radius, providing us with a method to effectively sort particles with subtle differences in their radii.

The trajectory of the particle (red line in figure 3(b)) is obtained by solving the motion equation (equation (1)) using the adaptive time step Runge-Kutta Verner algorithm (RKV). RKV is a specific implementation of the Runge-Kutta method that automatically adjusts the size of the time step while solving the equation of motion of the particle—a type of second-order ordinary differential equation [58]. Particles with radii near three typical EPs are considered.

The radii of the particles considered are denoted by the black arrows in figure 2(b) pointing slightly above the intersection points of $\gamma_{\mathrm{c}}$ and $\gamma_{\mathrm{air}}$ . Figure 3 shows the results denoted by the left arrow in figure 2(b). Figure 3(a) plots how the particle with a radius $r = 0.443\mu \mathrm{m}$ moves away from the beam center (equilibrium position) under a small perturbation. It is evident that the particle finally moves in an elliptical orbit. By carefully selecting the coefficients $a_1$ and $a_2$ , the trajectory described by

IOP Publishing

New J. Phys. 25 (2023) 093048

L Cui et al

4

![](dt=2026-06-11/ht=03/32dcd47078fd4e541f532574bcc7ac7966af9f32a47d0de31337d444233cebcd.jpg)

![](dt=2026-06-11/ht=03/6107af2f740e7bd93a875427e681ac932ae92a9e1ff6a319f969af74e3315667.jpg)

![](dt=2026-06-11/ht=03/97223f115afe8f4be5ad1fe11dead03a44b04543cff8400b04590fb22ef606f1.jpg)

![](dt=2026-06-11/ht=03/dd1a0153c252060603a4c9ed490f9ad7004e5fbf2ff90c52adf9d1ee02a93807.jpg)

![](dt=2026-06-11/ht=03/04e38430f52e4695284be90677620218da95ecc9bdcb8b2028d84fef352ac784.jpg)

![](dt=2026-06-11/ht=03/db0f55cefcde31383537f8cdb6857430225b0ff35742c41148eb71f6f7118345.jpg)

equation (6) closely matches the numerical result computed using the adaptive time step RKV algorithm (red solid line), as shown in figure 3(b), which confirms the validity of our previous analysis. As the particle radius increases, the elliptical orbit becomes larger, see figure 3(c). In figure 3(d), we depicted the major and minor axes of the elliptical orbits as functions of the increment of particle radius $\Delta r = r - r_0$ , where $r_0 = 0.442\mu \mathrm{m}$ .

It is evident that even for a subtle increment in the particle radius, there are significant increases in the axes of the elliptical orbits. Especially, the increase of the major axis is tens of times larger than the increase in particle radius. Since the change of major axis of the elliptical orbit is so significant, we can sort particles with subtle differences in their radii by referring to their ranges of motion.

It is noted that when the particle's radius is less than $0.25\mu \mathrm{m}$ , the ambient air damping consistently exceeds the critical damping, as shown in figure 2(b). Consequently, within this radius range, the particle is always confined to the beam center and the optical sorting method based on trajectory tracking is inapplicable.

Similar results are observed near another EP denoted by the right arrow in figure 2(b). Figure 4(a) displays the elliptical orbits of particles with radii ranging from $1.085\mu \mathrm{m}$ to $1.099\mu \mathrm{m}$ at an interval of $0.002\mu \mathrm{m}$ . The corresponding major and minor axes of the obits with respect to the variation of the particle radii are plotted in figure 4(b). Similarly, as the radius of the particle increase slightly, the major and minor axes increase significantly.

The rate of the increase in the major and minor axes of the orbit slow down as the particle radius becomes larger. This is because when $\Delta r$ becomes larger, $\gamma_{\mathrm{c}}$ grows more slowly away from the EP, resulting in a slower increase in the difference between $\gamma$ and $\gamma_{\mathrm{c}}$ , see figure 2(b).

IOP Publishing

New J. Phys. 25 (2023) 093048

L Cui et al

5

![](dt=2026-06-11/ht=03/2843fc54997a206466d74837f28632da809e707fa535e36235f15e28b05508c9.jpg)

![](dt=2026-06-11/ht=03/6004e2f5f1748234dabf5899a178a14447c39e68487aa72e88c6c0588b80188d.jpg)

![](dt=2026-06-11/ht=03/4cf83d8f04ac5a7d15c7e4a558dd23efeb3d206484f87cd436efef7bd19b068d.jpg)

![](dt=2026-06-11/ht=03/5e16fb45e707128acbb09badab24e5360bab1a0a80b813f6cf4333255d3e6034.jpg)

However, it should be noted that the optical sorting is not feasible near some EPs. Figure 4(c) shows the final obits of the polystyrene particles with radii ranging from $0.829\mu \mathrm{m}$ to $0.833\mu \mathrm{m}$ , which is near the EP denoted by the middle arrow in figure 2(b). It is observed that the trajectories of particles with different radii almost overlap with each other. This is because the particles are moving significantly far from the beam center, resulting in a substantial deviation of the optical force from $\vec{K} \cdot \Delta \mathbf{r}$ .

Consequently, the trajectory cannot be accurately described by equation (6) and becomes much more complex. As observed from figure 4(c), the trajectories deviate from being elliptical. In this case, we cannot sort the particles according to their trajectories.

However, by varying the wavelength and beam power, we can adjust the positions of EPs and potentially achieve optical sorting of particles of all sizes. For instance, by tuning the incident wavelength from 1.064 to $1.2\mu \mathrm{m}$ , an EP emerges around $r = 0.829\mu \mathrm{m}$ , as shown in figure 4(d). Figure 4(e) illustrates the final orbits of particles with different radii. Additionally, figure 4(f) demonstrates the relationship between the major axis of the elliptical orbit and the increment of the particle's radius, denoted as $\Delta r = r - r_0$ , where $r_0 = 0.829\mu \mathrm{m}$ . Clearly, even slight increase in the particle radius can lead to a significant enlargement in the major axis of the elliptical orbit.

# 2.4. Sorting of particles with subtle difference in permittivities

Besides the particle radius, the relative permittivity of the particle can also influence the value of critical damping and thus affecting the trajectory. In figure 5(a), we pl
otted the eigenvalues with respect to the relative permittivity $\varepsilon$ of the particle. The particle radius used here is $0.4\mu \mathrm{m}$ . Similarly, multiple pairs of EPs emerge when $\varepsilon$ ranges from 1 to 11. Figure 5(b) shows the critical damping (blue solid line) and ambient damping (red dashed line) with respect to the permittivities of the particle.

As an illustration, we numerically calculated the trajectories when the relative permittivity is near the EP indicated by the black arrow in figure 5(a). The final orbits of particles with relative permittivity ranging from 3.07 to 3.25 are displayed in figure 5(c). It is seen that the orbits are almost ellipses, especially when $\varepsilon \leqslant 3.14$ . With a slight increase in the relative permittivity, the major axes of the elliptical orbits exhibit significant growth, see the blue line in figure 5(d). And the orientations of the axes also undergo rotation.

Red circles in figure 5(d) show that with an increase in permittivity, the angle $\phi$ between the major axis of the elliptical orbit and the positive- $x$ direction also increase. This indicates that as the permittivity increases, the trajectory's orientation undergoes an anticlockwise rotation. These provide us a way to sort particles with subtle difference in their permittivities.

IOP Publishing

New J. Phys. 25 (2023) 093048

L Cui et al

6

# 3. Conclusion and discussion

In summary, a method for sorting particles with subtle differences in their properties based on trajectory tracking is proposed. The OFSM of a particle illuminated by vortex beams is inherently non-Hermitian, leading to the emergence of EPs when adjusting parameters such as the particle's radius and refractive index.

By analytically analysis and numerical verification of the particle's motion equation in a damping environment, we have demonstrated that particles exhibit stable motion in elliptical orbits around the beam center when their parameters slightly exceed those corresponding to specific EPs. Moreover, a slight increment in the particle's parameter near an EP results in a rapid increase in the imaginary parts of the eigenvalues. As a consequence, the particle absorbs more energy from the vortex beams and moves along a larger elliptical orbit.

This leads to significant changes in the major axis of the elliptical orbit due to subtle changes in the particle's parameters. While the optical sorting method may not be universally applicable to all EPs, it is possible to adjust the positions of EPs by tuning the wavelength, polarization and beam power of the incident beam. This flexibility enables us to extend the validity of the method for sorting particles of various types with subtle differences in their properties.

Numerous techniques have been suggested for the separation of particles [14-23, 41]. In comparison with these existing methods, our approach employs a groundbreaking mechanism that relies on the innovative utilization of EPs. What sets our approach apart from the existing methods is its ability to effectively separate particles with even minute disparities in their radii or refractive indices. However, it should be noted that our current method is primarily optimized for the sequential sorting of individual particles.

When dealing with multiple particles, the optical binding forces and mechanical collisions introduces additional intricacies. Consequently, the trajectories of these particles deviate significantly from those observed in the case of isolated particles. Addressing the complexities arising from particle-particle interactions within the framework of optical sorting presents a challenging yet rewarding prospect for future research endeavors.

# Data availability statement

All data that support the findings of this study are included within the article (and any supplementary files).

# Acknowledgments

This work is supported by the National Science Foundation of China (No. 12104069, No. 12174263, and No. 1904237), the Natural Science Foundation of Hunan Province (Grant No. 2021JJ40554), the Open Research Fund of Hunan Provincial Key Laboratory of Flexible Electronic Materials Genome Engineering though Grant No. 202007, and Natural Science Foundation of Guangdong Province (Grant No. 2020A1515010669).

# Conflict of interest

There are no conflicts to declare.

# ORCID iDs

LiYong Cui https://orcid.org/0000-0002-9365-7284

Neng Wang https://orcid.org/0000-0002-4185-9626

# References

IOP Publishing

New J. Phys. 25 (2023) 093048

L Cui et al

7

IOP Publishing

New J. Phys. 25 (2023) 093048

L Cui et al

8

IOP Publishing

New J. Phys. 25 (2023) 093048

L Cui et al

9