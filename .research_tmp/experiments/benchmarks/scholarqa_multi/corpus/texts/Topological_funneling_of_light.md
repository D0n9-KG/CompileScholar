# Topological funneling of light

Sebastian Weidemann\*, Mark Kremer\*, Tobias Helbig?, Tobias Hofmann?, Alexander Stegmaier?, Martin Greiter?, Ronny Thomale?, Alexander Szameit

<sup>1</sup>Institute of Physics, University of Rostock, Albert-Einstein-Straße 23, 18059 Rostock, Germany. <sup>2</sup>Department of Physics and Astronomy, Julius-Maximilians-Universität Würzburg, Am Hubland, 97074 Würzburg, Germany.

*These authors contributed equally to this work.

†Corresponding author. Email: alexander.szameit@uni-rostock.de

Dissipation is a general feature of non-Hermitian systems. But rather than being an unavoidable nuisance, non-Hermiticity can be precisely controlled and, hence, used for sophisticated applications such as optical sensors with enhanced sensitivity. In our work, we implement a non-Hermitian photonic mesh lattice by tailoring the anisotropy of the inter-site coupling. The appearance of an interface results in a complete collapse of the entire eigenmode spectrum, leading to an exponential localization of all modes at the interface. As a consequence, any light field within the lattice travels toward this interface, irrespective of its shape and input position. Based on this topological phenomenon, dubbed "non-Hermitian skin effect," we demonstrate a highly efficient funnel for light.

Non-Hermiticity is a ubiquitous property as the exchange of energy with the environment is inevitable for any physical system. While such an interaction is often considered undesirable, recent works (1-3) have shown that tailored gain and loss distributions within a non-Hermitian system can lead to intriguing features like non-orthogonal eigenmodes (4), exceptional points (5, 6) and peculiar transport transitions (7). Several promising applications have been proposed, such as exceptional-point enhanced sensing (8), or mode selective laser cavities (9, 10).

In photonics, the successful integration of optical gain and loss has enabled the experimental demonstration of a plethora of promising non-Hermitian features (11-13) such as loss-induced transparency (14) or unidirectional invisibility (5), providing a route to develop a new generation of optical devices. When dealing with periodic systems, the approximation of infinitely expanded settings is often used, which is never fulfilled in any experiment.

This approximation is commonly justified, because in sufficiently large systems, any distant boundaries are expected to introduce only minor and especially local changes in the mode spectrum, hence not causing appreciable deviations from the infinite system. A prominent example for this approximation is the topological boundary mode in the Hermitian Su Schrieffer Heeger (SSH) model (15). Besides the possible appearance of this mode, the eigenmode spectrum of the bulk in an SSH lattice does not change significantly when introducing a boundary.

However, it has been proposed that in certain lattices where the source of non-Hermiticity does not derive from gain and loss but anisotropic coupling, the presence of an interface can causes all eigenmodes to localize at this interface. Dubbed "non-Hermitian skin effect" (16-20), this phenomenon is currently the key element in a lively

debate about the validity of the bulk boundary correspondence (BBC) in non-Hermitian topological systems (16, 17, 21-23). The BBC demands the coexistence of a bulk and a boundary, where quantities derived from the bulk can be used to predict properties of the boundary, which is a crucial concept of topological physics.

Here, we demonstrate the non-Hermitian skin effect in a photonic lattice. Our experimental implementation is based on light propagation in coupled optical fiber loops (24) and relies on the fact that light propagation in this setup obeys the same equations, which describe a linear one-dimensional photonic lattice. The ability to control the optical properties of the fiber loops allows us to realize the required anisotropic coupling, also referred to as non-reciprocal coupling. By connecting lattices with different directions of the anisotropy, a funneling effect for light occurs (Fig. 1A), that relies on the non-Hermitian skin effect and its nontrivial topological properties induced by the anisotropic coupling.

In our work, we compare two models. The Hermitian SSH model, which consists of a chain of nearest neighbor coupled lattice sites with alternating coupling constants (15), is shown in Fig. 1B. Every second coupling is chosen to be different but isotropic, as indicated by the different shades of the orange arrows. The Skin effect model (Fig. 1C) also features an alternating, but anisotropic coupling, as the hopping from one site to its neighbor is different from the hopping from its neighbor back to itself. This anisotropy, indicated by differently sized black arrows, causes a non-Hermiticity, despite the fact that there is neither gain nor loss in the model.

Due to translational invariance, the eigenmodes in both lattices are delocalized when periodic boundary conditions are applied (see supplementary materials). When introducing

Science

REPORTS

Cite as: S. Weidemann et al., Science 10.1126/science.aaz8727 (2020).

First release: 26 March 2020

www.sciencemag.org

(Page numbers not final at time of first release)

1

Downloaded from http://science.sciencemag.org/ on March 28, 2020

an interface in the SSH lattice by inverting the ratio $c_{1} / c_{2}$ at some position (illustrated in Fig. 2A, where the inverted ribbon indicates the inverted coupling ratio), only one mode localizes at the interface, which is well-known as a topological SSH mode (15). All other modes, however, remain delocalized; that is, far away from the interface the modal amplitudes on the individual sites remain essentially unchanged (Fig. 2B).

The situation changes when an interface is introduced in the non-Hermitian lattice with anisotropic hopping. The interface is created by flipping the direction of the anisotropy at some position, illustrated by the mirrored pattern in Fig. 2C. The entire eigenmode spectrum is found to collapse and all eigenmodes are exponentially localized at the interface (Fig. 2D). All former bulk modes transform into boundary modes, making the notion of bulk, compared to edge modes, invalid, and hence querying the applicability of the BBC.

This behavior is in stark contrast to the Hermitian case, where sufficiently far from an interface its influence on the bulk mode structure is negligible. The localization of the eigenmodes in the non-Hermitian lattice has further profound consequences: No matter where the lattice is excited, every signal travels toward the interface. In the context of photonics, this means that any light signal that impinges the lattice is guided toward the interface and remains there.

These findings suggest the opportunity to realize a non-Hermitian light funnel, which may form the basis for intriguing applications. Note, that the presence of an interface does not change the propagation, until the light field reaches the interface.

Here, we study the non-Hermitian skin effect, by employing a modified version of a one-dimensional discrete-time quantum walk, also called light walk (24). The dynamics are governed by the evolution Eqs. 1 and 2.

$$
u _ {n} ^ {m + 1} = G _ {u} \left(\cos (\beta) u _ {n + 1} ^ {m} + i \sin (\beta) v _ {n + 1} ^ {m}\right) e ^ {i \varphi_ {u}} \tag {1}
$$

$$
v _ {n} ^ {m + 1} = G _ {v} \left(\mathrm {i} \sin (\beta) u _ {n - 1} ^ {m} + \cos (\beta) v _ {n - 1} ^ {m}\right) \tag {2}
$$

Here $u_{n}^{m}$ denotes the amplitude at lattice position $n$ and time step $m$ , on left moving paths, and $\nu_{n}^{m}$ the corresponding amplitude on right moving paths. The parameter $\beta = \beta (n,m)$ characterizes the splitting ratio of the beam splitter. The beam splitter mediates the hopping between lattice sites, as depicted in Fig. 3A. For instance, $\beta = \pi /4$ corresponds to a homogeneous lattice of 50:50 c
oupler. The ability to adjust the splitting $\beta$ at will, allows us to control the coupling between individual sites. In the evolution Eqs.

1 and 2 the common quantum walk dynamics are extended by $G_{u,\nu} = G_{u,\nu}(n,m)$ , which captures a non-Hermitian attenuation and amplification within the lattice, which enables the realization of an anisotropic coupling, described below.

Additionally, a phase modulation $e^{i\varphi_u}$ allows the implementation of arbitrary real parts of potential. In our experiments, optical pulses propagate in two unequally long fiber loops, which are connected by a variable beam splitter (VBS) with which we control the splitting $\beta$ (Fig. 3B). We employ acousto-optical modulators (AOM) to manipulate amplitudes via $G_{u,v}$ and a phase modulator (PM) to control $e^{i\varphi_u}$ . The phase modulation is used for studying the non-Hermitian skin effect in the presence of disorder. A detailed treatment of how the light propagation in the fiber loops maps on the SSH model and the non-Hermitian skin effect model can be found in the supplementary materials.

We start our experiments by probing the Hermitian SSH model that is realized by implementing two different coupling ratios $\beta_{1}$ and $\beta_{2}$ , as indicated by the different shades of the orange arrows and beam splitter cubes (Fig. 3A). In order to characterize the transport in the lattice we excite it at three different positions: left from the interface (Fig. 3C), directly at the interface (Fig. 3D), and right from the interface (Fig. 3E). Clearly, any excitation populates extended modes that lead to a spreading of the wave packet, even at the interface.

This is consistent with the spectrum of eigenmodes of this lattice shown in Fig. 2B, where only one localized (topological) mode exists at the interface, whereas all other modes remain delocalized and spread over the entire lattice.

However, this situation changes when turning to non-Hermitian contributions. To experimentally realize the non-Hermitian skin effect model, we require anisotropic coupling that is achieved by the introduction of amplification or attenuation depending on the hopping direction, as shown by the green plus and minus signs, respectively (Fig. 3A). This modulation is equivalent to replacing an isotropic beam splitter with its anisotropic counterpart (see supplementary materials). Such a modulation was already used in previous works to study PT-symmetric Bloch oscillations (25).

Moreover, it reassembles the Hatano-Nelson model (26), where an imaginary vector potential translates to an anisotropic coupling. To realize the interface, we combine two lattices with inverted modulation with respect to the amplification and attenuation (see supplementary materials). In contrast to the Hermitian case, a significant change in the propagation dynamics can be observed when probing the non-Hermitian lattice: As all eigenmodes are localized at the interface (Fig. 2D), any excitation results in a light flow that is directed toward the interface.

This is true when exciting left from the interface (Fig. 3F), at the interface (Fig. 3G), and right from the interface (Fig. 3H). This is exactly the manifestation of the non-Hermitian skin effect. The presence of an interface forces the eigenmodes to collapse at the interface, and no delocalized modes remain in the bulk of the lattice. Consequently, any light excitation somewhere in the lattice shows a funnel

Downloaded from http://science.sciencemag.org/ on March 28, 2020

First release: 26 March 2020

www.sciencemag.org (Page numbers not final at time of first release)

2

like behavior, such that light localizes exclusively along the interface. This behavior can be linked to the topological properties of the underlying lattice, as the direction of the anisotropic coupling is connected to a winding number (see supplementary materials) (27). Therefore, the interface between the lattices is connecting two regions of different winding, serving as an explanation for the localization.

In the next set of experiments, we explore the robustness of non-Hermitian skin effect in the presence of disorder (26). It is interesting to study the interplay of these two effects, as disorder induces Anderson localization (28), which—as a long-range interference effect—might be suppressed by the non-Hermitian modulation. We address this problem by adding a uniformly distributed phase disorder $\varphi_{u} \in [-W, W]$ in space $n$ upon the amplitudes $u_{n}^{m}$ , which is experimentally realized by using a fiber-based phase modulator (Fig. 3B).

In a first step, we combine the anisotropic modulation with a weak disorder $W = 0.1\pi$ (Fig. 4A). In this case, the light still moves toward the interface, demonstrating the robustness of the non-Hermitian Skin effect against a certain degree of disorder. If, on the other hand, strong disorder $W = 0.74\pi$ is applied, one can observe that the wave packet is arrested and no movement toward the interface occurs (Fig. 4B). This behavior is equivalent to disorder-induced Anderson localization without anisotropic coupling (29), as shown in Fig. 4C.

These results clearly show that the non-Hermitian modulation in our lattice does not prevent Anderson localization as an interference effect. In other words, a continuous transition from Skin effect localization at the funnel opening to Anderson localization at the excitation site takes place (see supplementary materials). This behavior can be explained in topological terms using the winding number. Whereas for low disorder the winding number can still be defined for strong disorder a topological transition occurs because the winding number vanishes (27).

In other words, the funneling effect is robust against disorder until a topological transition occurs. The transition from the Skin effect localization at the interface toward Anderson localization at the injection site does clearly show that anisotropic coupling and the presence of a boundary does not necessarily lead to a funneling effect, which one might intuitively think.

Our findings provide a route for utilizing anisotropic coupling and non-reciprocity for novel non-Hermitian phenomena in sophisticated applications, such as light harvesting and enhanced optical sensitivity. In this vein, coupled optical fiber loops are a flexible and scalable platform for investigating non-reciprocal lattices with anisotropic coupling. Since our approach is not limited to the specific experimental platform and is based only on the wave-like properties of light, it might also be applicable to other areas of research using different experimental environments.

# REFERENCES AND NOTES

Downloaded from http://science.sciencemag.org/ on March 28, 2020

First release: 26 March 2020

www.sciencemag.org

(Page numbers not final at time of first release)

3

# ACKNOWLEDGMENTS

The authors thank Martin Wimmer for very useful discussions. Funding: The authors acknowledge funding from the Deutsche Forschungsgemeinschaft (BL 574/13-1, SZ 276/19-1, SZ 276/20-1, SZ 276/9-2, and 258499086 - SFB 1170, the Würzburg-Dresden Cluster of Excellence on Complexity and Topology in Quantum Matter ct.qmat (39085490 - EXC 2147), and the Alfred Krupp von Bohlen und Halbach Foundation. Author contributions: M.K. developed the theory and S.W. performed the experiments on the photonic mesh lattice. R.T. and A.S. supervised the project.

All authors discussed the results and co-wrote the paper. The manuscript reflects the contributions of all authors. Competing interests: Authors declare that they have no competing interests. Data and materials availability: All experimental data and any related experimental background information not mentioned in the text can be found at the Rostock University Publication Server repository (30).

# SUPPLEMENTARY MATERIALS

science.sciencemag.org/cgi/content/full/science.aaz8727/DC1

Materials and Methods

Supplementary Text

Figs. S1
to S9

References

16 October 2019; accepted 3 February 2020

Published online 26 March 2020

10.1126/science.aaz8727

Downloaded from http://science.sciencemag.org/ on March 28, 2020

First release: 26 March 2020

www.sciencemag.org

(Page numbers not final at time of first release)

4

![](dt=2026-03-15/ht=06/5993cf822b6c22e214a2baa2dda1f82d959bba479652cb5f189856e35dcb7a26.jpg)

![](dt=2026-03-15/ht=06/8651008c546c93e1464d07a1497a4cfbe0168701536171784dee7247ab1e0fc0.jpg)

![](dt=2026-03-15/ht=06/2bae7763d9f7f80abdf5403ce518e9ae57c17b468f84f301d909156d5069c4af.jpg)

![](dt=2026-03-15/ht=06/192d888d6a1ec733a9dcc7ac4aeef13220ec6286ac3cec0ff00727275aff36e1.jpg)

![](dt=2026-03-15/ht=06/2fd1192f55a1949087c2fdb9bfd02f02a17278e98c4f272cbcb9c242a21aef73.jpg)

Downloaded from http://science.sciencemag.org/ on March 28, 2020

First release: 26 March 2020

www.sciencemag.org

(Page numbers not final at time of first release)

5

![](dt=2026-03-15/ht=06/0a49d4fd535ae3c4defafdbc1c7e987b40448bb7e2e6056c1591339ff0d95e87.jpg)

![](dt=2026-03-15/ht=06/9a1ceaea3f74aea4dd6a5f5e5f89d4782a2559dd32bb11a2ff42746533cecbd3.jpg)

![](dt=2026-03-15/ht=06/19f80adbf10f50172137352df72bb55899a5d1a964bc8b45ba32974caef1f43b.jpg)

![](dt=2026-03-15/ht=06/abcf6fc401ed44a87bb1a41735b746937d125f571daf84f70d9ef91df85edbb3.jpg)

Downloaded from http://science.sciencemag.org/ on March 28, 2020

First release: 26 March 2020

www.sciencemag.org

(Page numbers not final at time of first release)

6

![](dt=2026-03-15/ht=06/d10c1377ed20dfeab566459000047028e4bbe32155a2234d22c6ce6bc49e85d5.jpg)

Downloaded from http://science.sciencemag.org/ on March 28, 2020

First release: 26 March 2020

www.sciencemag.org

(Page numbers not final at time of first release)

7

# Topological funneling of light

Sebastian Weidemann, Mark Kremer, Tobias Helbig, Tobias Hofmann, Alexander Stegmaier, Martin Greiter, Ronny Thomale and Alexander Szameit

published online March 26, 2020

ARTICLE TOOLS

http://science.sciencemag.org/content/early/2020/03/25/science.aaz8727

SUPPLEMENTARY MATERIALS

http://science.sciencemag.org/content/suppl/2020/03/25/science.aaz8727.DC1

REFERENCES

This article cites 26 articles, 3 of which you can access for free http://science.sciencemag.org/content/early/2020/03/25/science.aaz8727#BIBL

PERMISSIONS

http://www.sciencemag.org/help/reprints-and-permissions

Use of this article is subject to the Terms of Service

Science

Science (print ISSN 0036-8075; online ISSN 1095-9203) is published by the American Association for the Advancement of Science, 1200 New York Avenue NW, Washington, DC 20005. The title Science is a registered trademark of AAAS.

Copyright © 2020, American Association for the Advancement of Science

Downloaded from http://science.sciencemag.org/ on March 28, 2020