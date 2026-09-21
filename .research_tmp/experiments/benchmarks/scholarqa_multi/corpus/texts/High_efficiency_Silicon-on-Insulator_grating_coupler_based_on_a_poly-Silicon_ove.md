# High efficiency Silicon-on-Insulator grating coupler based on a poly-Silicon overlay

Günther Roelkens, Dries Van Thourhout, Roel Baets

Photonics Research Group, Ghent University - IMEC, Sint-Pietersnieuwstraat 41, B-9000 Ghent, Belgium

Gunther.Roelkens@intec.Ugent.be

Abstract: A high efficiency broadband grating coupler for Silicon-On-Insulator waveguides was designed. The grating coupler is defined by locally adding a poly-Silicon layer on top of the existing waveguide layer structure prior to grating etching. Adding this poly-Silicon layer reshapes the grating structure which changes its diffraction properties. Coupling efficiencies as high as $78\%$ at a wavelength of $1.55\mu \mathrm{m}$ are calculated and the optical 3dB bandwidth of the device is about $85\mathrm{nm}$ . The device layout is compatible with standard CMOS technology processing.

©2006 Optical Society of America

OCIS codes: (130.0130) Integrated optics; (050.2770) Gratings

# References and Links

# 1. Introduction

Silicon-on-Insulator is emerging as an interesting platform for integrated optics due to the high index contrast between the silicon core and the oxide cladding $(\Delta n \cong 2)$ . This enables large density integrated optical circuits, which can be fabricated by standard CMOS technology [1]. One of the drawbacks of the high index contrast is the large mismatch in mode size and shape between the fundamental mode of the SOI waveguide and the mode of the optical fiber making efficient coupling of light from fiber to waveguide an important issue.

Several approaches are being envisaged in literature to overcome this problem. 3D taper structures can adiabatically transform the fiber waveguide mode to the mode of an SOI waveguide [2]. However, the definition of the taper requires gray scale lithography, which is difficult to

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11622

control. An inverted taper approach uses only planar waveguide definition technology to make an adiabatic taper structure, however this type of devices requires a lensed or high numerical aperture fiber with reduced core size to obtain efficient coupling over a reasonable device length [3]. As proposed in literature, one dimensional [4] or two dimensional [5] grating structures can also be used to couple light from a fiber into an SOI waveguide. These grating couplers have the advantage of not requiring polished facets for coupling, which enables wafer scale testing of the integrated circuits.

Devices are very compact and have a large optical bandwidth. Although one dimensional grating couplers are very polarization dependent, a polarization diversity scheme based on a two dimensional grating coupler can be used in practical applications [5]. The reported fiber coupling efficiency obtained with standard uniform grating structures is limited. In Ref. [5] $20\%$ fiber coupling efficiency was experimentally obtained for a two dimensional grating coupler. In Ref.

[6], a one dimensional grating structure with a theoretical coupling efficiency of $37\%$ was designed, while experimentally $31\%$ coupling efficiency was obtained. This moderate coupling efficiency is inherent to the device structure due to the substantial fraction of incident power that is diffracted towards the Silicon substrate and the mismatch in the field profile between the upwards diffracted light and the Gaussian fiber mode. In order to increase the efficiency of the grating coupler structure, different strategies can be followed.

One strategy is to include a bottom mirror to redirect the downwards diffracted light. This can be a gold bottom mirror [7] or a DBR-type mirror [8]. A second strategy is to optimize the dimensions of the individual grating periods to match the profile of the diffracted light with that of the fiber mode [8], while in a third strategy slanted gratings are used to optimize the coupling efficiency [9]. Although high efficiencies can be obtained, the technology required for fabrication is not CMOS compatible or is very complex.

Therefore, we present here a way to increase the coupling efficiency by simply adding an additional layer of poly-Silicon before grating definition. This additional poly-Silicon layer reshapes the grating structure which changes its diffraction properties to improve the fiber coupling efficiency.

# 2. Proposed device layout

The structure we propose is depicted in Fig. 1. It consists of an SOI waveguide structure with a crystalline Silicon waveguide layer of $220\mathrm{nm}$ thick and a buried oxide layer of $2\mu \mathrm{m}$ in order to prevent leakage of light to the Silicon substrate. Locally, an additional poly-crystalline silicon layer is deposited, in which the grating coupler is etched (possibly extending the etch into the crystalline Silicon layer). The optical fiber is slightly tilted with respect to the vertical axis in order to prevent a large second order reflection and is assumed to be AR coated for a wavelength of $1.55\mu \mathrm{m}$ . Reflections at the fiber facet can also be avoided by adding an index matching glue between the optical fiber and the grating coupler.

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11623

![](dt=2026-06-02/ht=21/d7ca9d7d039dd615da38636fc6d7b2661337b82d3fa74de4bc4cd28e8ae990a8.jpg)

A possible processing scheme to define this type of grating structure is depicted in Fig. 2. Only standard CMOS processes are used, making this type of grating coupler applicable for mass manufacturing. The process sequence starts by depositing a $\mathrm{SiO}_2$ hard mask on the SOI waveguide structure and opening a window in the hard mask for the definition of the poly-Silicon layer. After deposition of a thick poly-Silicon layer, a chemical mechanical polishing step is used to define the poly-Silicon overlay using the oxide mask as a polish stop layer. Finally, the grating is etched in the poly-Silicon layer and the oxide mask is removed.

![](dt=2026-06-02/ht=21/a8d66a961b46a64840fcc82569cec60fe90e54ff13a79076f955e9694f601fae.jpg)

# 3. Device optimization

To optimize the structure, the influence of the thickness of the poly-Silicon layer, etch depth of the grating and grating period on the directionality of the grating (being the amount of power diffracted upwards for a 20 period long grating) was assessed. Simulations were performed using CAMFR [10], a two-dimensional fully vectorial solver based on eigenmode expansion. TE polarization and a wavelength of $1.55\mu \mathrm{m}$ are assumed. As high index contrast

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11624

waveguide structures behave very differently for TE and TM polarization, only TE polarization is considered. However, two dimensional grating structures can be used to tackle these polarization issues [5] or the design of the one dimensional grating structure can be adapted to accommodate high coupling efficiency for TM polarization. The simulation results are plotted in Fig. 3 (for a grating duty cycle of $50\%$ ). Light is injected into the SOI waveguide and is scattered by the grating coupler. This figure shows that an optimal directionality of $85\%$ is achieved for a poly-Silicon thickness of $150\mathrm{nm}$ , an etch depth of $220\mathrm{nm}$ and a grating period of $610\mathrm{nm}$ .

Simulation shows that these parameters for the grating structure result in a 10 degree off vertical angle of light diffraction at $1.55\mu \mathrm{m}$ (and hence a 10 degree t
ilt of the optical fiber for efficient coupling). In the case of a standard grating coupler structure with a $220\mathrm{nm}$ Silicon waveguide layer (i.e. without the additional poly-Silicon layer), a separate optimization of the grating etch depth, grating period and buried oxide layer thickness results in an optimal grating directionality of $55\%$ , significantly lower than the results obtained in Fig. 3.

The field plot of a separately optimized grating coupler structure with and without the poly-Silicon overlay is shown in Fig. 4, from which the improvement in grating directionality is clear. While the buried oxide layer thickness plays an important role in the standard grating coupler design to achieve constructive interference between the reflected wave at the oxide-Silicon interface and the directly upwards radiated wave, this is far less the case in the optimized grating coupler structure, which inherently is more directional.

![](dt=2026-06-02/ht=21/5714d46a1e6329e7d55a372d881f1b0cdb8ca264a4eab628ff32ab719437a19f.jpg)

To calculate the coupling efficiency to fiber, we assumed an optical fiber with a core diameter of $9\mu \mathrm{m}$ and a refractive index contrast between core and cladding of 0.005. The grating coupling structure is excited by the power normalized fundamental Silicon waveguide mode and the diffracted field pattern in the air cladding is calculated. The coupling efficiency is then calculated by evaluating the overlap integral

$$
\eta = \left| \int \mathbf {E} \times \mathbf {H} _ {f i b} ^ {*} \cdot \mathbf {d S} \right| ^ {2} \tag {1}
$$

in which $\mathbf{E}$ is the electric field vector of the diffracted light in the air cladding and $\mathbf{H}_{fib}$ is the magnetic field of the fiber waveguide mode, which is also normalized in power. dS lies along

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11625

the surface normal of an integration pad in the air cladding which spans the complete grating coupler structure length. This overlap integral is evaluated for different fiber positions along the direction of propagation of the excited Silicon waveguide mode resulting in the optimal fiber position and power coupling efficiency.

![](dt=2026-06-02/ht=21/571d7471f3087960ede63f4cfb1bce243ec9c4b71e8b3aae622e69f01fca2029.jpg)

For the case of a uniform grating with the parameters mentioned above (grating period $610\mathrm{nm}$ , duty cycle $50\%$ , poly-silicon overlay thickness $150\mathrm{nm}$ and etch depth of $220\mathrm{nm}$ ) a coupling efficiency of $66\%$ at $1.55\mu \mathrm{m}$ is obtained for a fiber that is tilted 10 degrees with respect to the vertical axis.

The origin of the increased directionality of the grating is attributed to the increased vertical asymmetry in the grating structure. In the case of a vertically symmetric grating (by etching completely through the Silicon waveguide layer and applying the same top and bottom cladding materials) the diffraction pattern is identical in top and bottom cladding. The grating directionality is therefore limited to 50 percent.

By reducing the etch depth of the grating, the asymmetry of the grating structure is increased (which can improve the directionality) at the expense of a longer coupling length of the grating as the strength of the grating is reduced. For a uniform grating, the diffracted field profile is exponentially decaying along the length of the grating structure and no perfect match with the Gaussian field profile of the optical fiber is achieved. An optimal grating coupler length exists for which the overlap between the exponentially decaying diffracted field and the Gaussian mode profile is maximal.

As a high coupling efficiency grating structure has to be designed close to this optimal coupling length, this also determines the directionality of the grating. By adding the additional poly-Silicon layer an additional degree of freedom is introduced in the design of the structure, thereby allowing an optimization of both the directionality of the grating coupler and the coupling length of the grating.

High directionality can be obtained due to the destructive interference of light diffracted towards the substrate while constructive interference is obtained for diffraction in the air cladding. This is an intrinsic property of the excited Bloch mode of the designed periodic grating structure as the directionality is nearly independent of the thickness of the buried oxide layer showing that the reflection at the buried

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11626

oxide / Silicon substrate interface does not significantly affect the constructive and destructive interference properties (as is the case in the standard grating coupler structure).

In order to further increase the fiber coupling efficiency, a non-uniform grating structure can be designed in order for the diffracted light to better match the Gaussian profile of the optical fiber. A genetic algorithm approach was used to optimize the width of the individual grating teeth and slits while keeping the etch depth constant. To be definable using CMOS deep UV lithography, the smallest feature size allowed in the genetic algorithm was set to $200\mathrm{nm}$ . The evolution of the optimal coupling efficiency to fiber as a function of the genetic algorithm generation is shown in Fig. 5. An optimal fiber coupling efficiency of $78\%$ was obtained.

![](dt=2026-06-02/ht=21/5e4a4b006d05dbb5627b5afea2b53c7ebd1ce147e4b0c3433790347ce5bbdd5b.jpg)

The grating teeth and slit widths for which this high coupling efficiency was achieved are tabulated in Table 1. It turns out that a rather random varying local grating period and duty cycle shows to be optimal, which justifies the use of a genetic algorithm optimization approach.

Table 1. Optimal grating tooth and slit widths obtained from a genetic algorithm optimization

![](dt=2026-06-02/ht=21/830eeb3a13386c73ab6009e03dec68225d368dfd3e4cc720ecc596f89dcf1671.jpg)

<table><tr><td>Period</td><td>Tooth width(nm)</td><td>Slit width(nm)</td><td>Period</td><td>Tooth width(nm)</td><td>Slit width(nm)</td></tr><tr><td>1</td><td>220nm</td><td>360nm</td><td>11</td><td>300nm</td><td>330nm</td></tr><tr><td>2</td><td>230nm</td><td>410nm</td><td>12</td><td>230nm</td><td>370nm</td></tr><tr><td>3</td><td>280nm</td><td>310nm</td><td>13</td><td>290nm</td><td>290nm</td></tr><tr><td>4</td><td>260nm</td><td>380nm</td><td>14</td><td>330nm</td><td>310nm</td></tr><tr><td>5</td><td>280nm</td><td>350nm</td><td>15</td><td>300nm</td><td>300nm</td></tr><tr><td>6</td><td>270nm</td><td>340nm</td><td>16</td><td>280nm</td><td>340nm</td></tr><tr><td>7</td><td>280nm</td><td>300nm</td><td>17</td><td>310nm</td><td>280nm</td></tr><tr><td>8</td><td>310nm</td><td>320nm</td><td>18</td><td>310nm</td><td>270nm</td></tr><tr><td>9</td><td>310nm</td><td>290nm</td><td>19</td><td>260nm</td><td>320nm</td></tr><tr><td>10</td><td>310nm</td><td>320nm</td><td>20</td><td>370nm</td><td>240nm</td></tr></table>

For these optimal device parameters, an FDTD-analysis was performed to assess the optical bandwidth of the device. The spectral dependence of the grating to fiber coupling efficiency is plotted in Fig. 6(a). A 3dB optical bandwidth of $85\mathrm{nm}$ is obtained. This is larger than the $60\mathrm{nm}$ 3dB bandwidth obtained in the standard grating coupler structure [6] and is related to the higher refractive index contrast in the proposed
grating coupler structure. A field plot of the optimized structure when the grating coupler is illuminated by the fiber optical mode at a wavelength of $1.55\mu \mathrm{m}$ is plotted in Fig. 6(b).

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11627

![](dt=2026-06-02/ht=21/d2bd7786438bf0b4e23544ddc909cb56e072b79dcd77df56edcf401538e36ad5.jpg)

![](dt=2026-06-02/ht=21/9d27da543cdd0709f84fdd44d8b751383e72dfb930586605b1c7116313345c84.jpg)

Although the diameter of the optical fiber cladding was decreased to reduce the FDTD simulation window, the presence of a $125\mu \mathrm{m}$ diameter fiber core (resulting in an increase of the fiber to grating separation) is not expected to impact the fiber coupling efficiency as the minimum obtainable separation of $62.5\mu \mathrm{m}$ x $\tan (10^{\circ}) = 11\mu \mathrm{m}$ is much smaller than the Rayleigh range (indicating the onset of substantial diffraction of a Gaussian beam) of $\pi \mathrm{w}^2 /\lambda = \pi (4.5\mu \mathrm{m})^2 /1.55\mu \mathrm{m} = 41\mu \mathrm{m}$ for a Gaussian beam with a waist radius of $4.5\mu \mathrm{m}$ , as is the case for a single mode optical fiber.

# 4. Tolerance analysis

In the fabrication of the grating coupler structure, deviations from the designed grating structure can occur. The thickness of the deposited poly-Silicon layer or the etch depth can vary. The position of the mask to define the grating can vary with respect to the edge of the deposited poly-Silicon overlay, which results in an uncertainty of the size of the first and last grating tooth. Also hard to predict variations of the individual grating teeth and slit width can occur due to proximity effects in the deep UV lithography of the grating structure.

Finally, the position of the optical fiber can be different from the optimal position. The influence of the poly-Silicon layer thickness and the grating etch depth variation is plotted in Fig. 7. A $+/-10\%$ deviation from the optimal device parameters is assumed. From these calculations it is

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11628

clear that this type of fabrication error predominantly results in a shift of the resonance wavelength of the grating coupler.

![](dt=2026-06-02/ht=21/9b23e70f2027addcea16d475dfd0efd72a3279f87abdb849caaa6e0726789bf8.jpg)

![](dt=2026-06-02/ht=21/a2ef1f564878e694ec413153091f255136eb67365d1ac5d9e0b389c9bbd21081.jpg)

Although the misalignment between the grating etch mask and the poly-Silicon overlay mask results in an uncertainty of the size of the first and the last grating tooth, nearly all light has been diffracted from the waveguide when reaching the last tooth and therefore only the influence of the width of the first grating tooth is important. This influence is assessed in Fig. 8(a), showing that a $+/-100\mathrm{nm}$ alignment accuracy is sufficient for high device performance. This can easily be achieved using a standard CMOS stepper. In Fig. 8(b), the influence of random variations of the grating coupler parameters due to illumination proximity errors are shown. Random variations of $+/-10\%$ on all grating tooth and slit widths were assumed.

![](dt=2026-06-02/ht=21/054bb4a12687fbf502d5a8d6faab6e48994f4e771c36649643002c87b36bc8ad.jpg)

![](dt=2026-06-02/ht=21/5f8a20778623b3df73a9cb0b0fa0559fa405cf45daa60f9b406e72d93630d9d7.jpg)

Finally, the influence of the position of the optical fiber along the direction of propagation of the excited Silicon waveguide mode on the grating coupler efficiency was assessed. This is depicted in Fig. 9, showing a good alignment tolerance for the position of the optical fiber. A 1dB misalignment tolerance of $+/- 1.5\mu \mathrm{m}$ is obtained.

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11629

![](dt=2026-06-02/ht=21/35ccd41f5b9c8c6416b6b36eb03a7d18aadc84a176c1d4f7a0f4d1a0a2fe7af4.jpg)

# 5. Conclusions

In this paper, a new type of grating coupler structure was presented for high efficiency coupling between an SOI waveguide and a single mode optical fiber. The grating coupler structure consists of a poly-Silicon overlay in the grating region and a non-uniform etched grating, which can be fabricated using standard deep UV lithography. A fiber coupling efficiency of $78\%$ with a 3dB optical bandwidth of $85\mathrm{nm}$ is feasible using this type of device.

# Acknowledgments

This work was partly supported by the Belgian IAP-PHOTON network, the IWT-SBO epSOC project and the Fund for Scientific Research (FWO Vlaanderen).

#75512 - $15.00 USD

(C) 2006 OSA

Received 27 September 2006; revised 8 November 2006; accepted 14 November 2006

27 November 2006 / Vol. 14, No. 24 / OPTICS EXPRESS 11630