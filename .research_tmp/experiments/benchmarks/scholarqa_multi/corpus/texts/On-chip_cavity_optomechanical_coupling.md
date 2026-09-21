# On-chip cavity optomechanical coupling

Bradley D Hauer, Paul H Kim, Callum Doolin, Allison JR MacDonald, Hugh Ramp and John P Davis*

*Correspondence:  
jdavis@ualberta.ca  
Department of Physics, University of Alberta, T6G 2E1 Edmonton, AB, Canada

# Abstract

Background: On-chip cavity optomechanics, in which strong co-localization of light and mechanical motion is engineered, relies on efficient coupling of light both into and out of the on-chip optical resonator. Here we detail our particular style of tapered and dimpled optical fibers, pioneered by the Painter group at Caltech, which are a versatile and reliable solution to efficient on-chip coupling. A brief overview of tapered, single mode fibers is presented, in which the single mode cutoff diameter is highlighted.

Methods: The apparatus used to create a dimpled tapered fiber is described, followed by a comprehensive account of the procedure by which a dimpled tapered fiber is produced and mounted in our system. The custom-built optical access vacuum chambers in which our on-chip optomechanical measurements are performed are then discussed. Finally, the process by which our optomechanical devices are fabricated and the method by which we explore their optical and mechanical properties is explained.

Results: Using this method of on-chip optomechanical coupling, angular and displacement noise floors of 4 nrad/√Hz and 2 fm/√Hz have been demonstrated, corresponding to torque and force sensitivities of $4 \times 10^{-20} \mathrm{~N} \cdot \mathrm{m} / \sqrt{\mathrm{Hz}}$ and 132 aN/√Hz, respectively.

Conclusion: The methods and results of our on-chip optomechanical coupling system are summarized. It is our expectation that this manuscript will enable the novice to develop advanced optomechanical experiments.

Keywords: Cavity optomechanics; Nanoscale transduction; Dimpled fiber; Tapered fiber; Nanomechanics

PACS codes: 07.60.-j; 07.10.Cm; 42.50.Wk

# Background

State-of-the-art nanofabrication technologies have allowed for a drastic reduction in the size, and increase in quality, of nanomechanical systems, which have been the driving force behind radically increasing the sensitivity of numerous devices. Examples include accelerometers [1], mass sensors [2-6], electrometers [7], temperature sensors [8,9], force transducers [10,11] and biosensors [12-15]. The mass of a sensor and its ability to precisely measure physical quantities are intimately related, with smaller devices having superior sensitivity [5,6,11].

However, as we continue to reduce device volume, it is difficult to find transduction methods that scale appropriately. Furthermore, it is generally the case that the target quantity is measured through the nanomechanical device's motion, hence as we move to more sensitive devices, we require a detection method with comparable

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4  
http://www.epjtechniquesandinstrumentation.com/content/1/1/4

EPJ.org

EPJ Techniques and Instrumentation a SpringerOpen Journal

RESEARCH ARTICLE

Open Access

Springer

© 2014 Hauer et al.; licensee Springer on behalf of EPJ. This is an Open Access article distributed under the terms of the Creative Commons Attribution License (http://creativecommons.org/licenses/by/2.0), which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly credited.

precision. A solution to these issues has been found in the field of cavity optomechanics, which allows for quantum-limited, sub-am/ $\sqrt{\mathrm{Hz}}$ displacement sensitivity [16,17] and device masses down to the pico/femto-gram range [18-20].

Optomechanics describes the coupling of the mechanical motion of a device to an optical field, often to manipulate or detect its motion. It is advantageous to use an optical cavity, such as a whispering gallery mode (WGM) resonator [21] (see Figure 1b), to provide this field, as the light in the optical cavity is able to sample the mechanics many times due to its long photon lifetime and leads to resonantly enhanced optomechanical coupling. In such a system, the motion of the mechanical device shifts the resonance frequency and phase of the optical cavity.

By detecting this signal, it is possible to infer the motion of the device. At the same time, photons inside the cavity apply a radiation pressure to the mechanical resonator [22], which can be used to optomechanically dampen or amplify its motion, leading to a large number of interesting phenomena [23]. Cavity optomechanical systems have been realized in a number of different geometries, including photonic crystal cavities [20], Fabry-Pérot etalons [24,25], WGM resonators [21,26,27] and electronic microwave cavities [28].

It is advantageous to fabricate cavity optomechanical devices on-chip, as it is therefore possible engineer both the optical and mechanical resonators to certain desired specifications. Modern nanofabrication technologies make it possible to produce devices with extremely accurate dimensions, allowing feature sizes as small as $100\mathrm{nm}$ for foundry-based deep ultraviolet (DUV) optical lithography [29] and $2\mathrm{nm}$ with electron beam lithography [30].

This enables precise tailoring of important device parameters, such as the gap between mechanical and optical resonators, which controls the optomechanical coupling in our devices [21]. Furthermore, devices fabricated using top-down lithography can be integrated into electronic on-chip devices [31] and, in the case of optical lithography, can easily be mass produced.

However, difficulties arise when trying to couple light into these on-chip devices, as a high on-chip density and planar geometry require a precise optical probe which can couple exclusively to a particular device. This problem has been solved by using a dimpled tapered fiber, which is created by introducing a small protrusion to a straight tapered

![](dt=2026-06-09/ht=00/7bfb56509af6fbd61f32a3f862ae644e04160f08d9363bd4ef9f76289bc1c12a.jpg)

![](dt=2026-06-09/ht=00/01f7cc18c4c5c421746be94146d17422d0fccffe1a7c599568736236c63a8cda.jpg)

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 2 of 21

fiber [21,32]. Such a waveguide provides an efficient and maneuverable probe, allowing selective coupling to on-chip optomechanical devices. This procedure is illustrated schematically in Figure 1a.

In this article, we describe a process by which such a coupling system is produced, outlining the necessary steps while assuming no special knowledge a priori. We begin by investigating the fundamentals of tapered optical fibers, as well as describing an apparatus which can be used to fabricate and dimple them. Following this is a discussion of the procedure by which dimpled tapered fibers are produced. Custom-built optical access vacuum chambers, which are used for optomechanical coupling to on-chip devices, are also detailed.

Finally, we explain our method of coupling to optomechanical devices with tapered fibers. Using these systems, we have been able to demonstrate the first ever on-chip optomechanical torsional sensors [21], as well as multidimensional detection of high frequency microcantilevers [18] suitable for force sensing applications.

# Single mode tapered optical fibers

A crucial element in any optomechanical device is the method by which the optical field is injected, and subsequently collected, from the optical resonator in the system. While a number of different options exist, including free space optical coupling [33], grating couplers [34] and fiber-to-waveguide coupling [35,36], we have chosen to use direct coupling from tapered optical fibers [20,37-40]. Tapered fibers are more efficient, and require less on-chip space
, than grating couplers, while free-space coupling is inconsistent with on-chip devices. It may prove that fiber-to-waveguide coupling [36] is more efficient and stable than tapered fibers, but the versatility and maneuverability of tapered fibers remains a significant advantage.

A tapered fiber is a standard optical fiber (silica core surrounded by a higher index cladding) that has had its initial diameter adiabatically reduced over a small length known as the tapered region. This can be performed either through hydrofluoric acid etching of an optical fiber [41,42], or by the heat-and-pull method [43-46]. In this latter method, a small region of an optical fiber, known as the hot-zone, is heated to the point of melting and subsequently stretched to reduce its diameter.

The final tapered fiber will then consist of three regions, the initial unstretched fiber, the taper transition, and the taper waist, all of which are detailed in Figure 2. From a conservation of mass argument, it can be shown that for a constant hot-zone of length $L$ , which is produced in the case of a stationary flame, the taper transition is exponential [47]. Using a constant pull speed $\nu$ , this results in a taper waist diameter $d$ that decreases with pull time $t$ according to

$$
d = d _ {0} e ^ {- \nu t / L}, \tag {1}
$$

where $d_0$ is the diameter of the initial untapered fiber.

Following the heat-and-pull process, a new air-clad core exists in the taper waist, comprised of a composite material with an effective index determined by the indices and relative sizes of the initial core and cladding. This region can be modeled as a long, dielectric cylinder, for which Maxwell's equations can be solved analytically to determine the electromagnetic modes of the core (cladding) in terms of Bessel (modified Bessel) functions of the first (second) kind, as described in [48]. In general, such a structure will support many modes, lending to the description of a multimode fiber. However, once the fiber's diameter drops below a critical value, known as the single mode cut-off diameter,

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 3 of 21

![](dt=2026-06-09/ht=00/3059b7628948d76b7de95b4ad8fd773e5188c133db175d071914a2f785584b51.jpg)

only a single guided mode remains in the fiber, labeled the hybrid $HE_{11}$ mode [49], as all other spatial modes decay evanescently. We look to determine this critical diameter for light with a free space wavelength, $\lambda$ , traveling in a fiber with a core of index of refraction, $n_{co}$ , surrounded by a cladding with index, $n_{cl}$ . This is done by matching the electromagnetic fields within the core and cladding according to the boundary conditions given by Maxwell's equations [50], resulting in the following expression for the single remaining mode

$$
\left[ \frac {J _ {1} (x)}{x J _ {0} (x)} + \frac {K _ {1} (y)}{y K _ {0} (y)} \right] \left[ \frac {n _ {c o} ^ {2}}{n _ {c l} ^ {2}} \frac {J _ {1} (x)}{x J _ {0} (x)} + \frac {K _ {1} (y)}{y K _ {0} (y)} \right] = 0. \tag {2}
$$

In the above equation, $J_{\nu}(x)$ is the Bessel function of the first kind and $K_{\nu}(x)$ is the modified Bessel function of the second kind. As well, $x = \frac{d}{2}\sqrt{k_{co}^2 - \beta^2}$ and $y = \frac{d}{2}\sqrt{\beta^2 - k_{cl}^2}$ , where $k_{co} = 2\pi n_{co} / \lambda$ and $k_{cl} = 2\pi n_{cl} / \lambda$ are the magnitudes of the wavevector in the core and cladding, respectively, and $\beta$ is the fiber's propagation constant. From these definitions, we can immediately derive the expression

$$
x ^ {2} + y ^ {2} = \frac {\pi^ {2} d ^ {2} \left(n _ {c o} ^ {2} - n _ {c l} ^ {2}\right)}{\lambda^ {2}}. \tag {3}
$$

Using this relationship between $x$ and $y$ , we are able to determine the fiber diameter $d_{c}$ such that Eq. 2 has only one solution for $d < d_{c}$ , indicating the point at which the penultimate mode ceases to exist. This value is the single mode cut-off diameter and is determined by numerically calculating the solutions to Eq. 2 while iteratively increasing $d$ until a second solution emerges.

It is also possible to determine an analytic expression for $d_{c}$ in the weakly-guiding approximation (WGA) [51]. In this case, we take $n_{co} \approx n_{cl}$ , so that Eq. 2 becomes

$$
\frac {x J _ {0} (x)}{J _ {1} (x)} = - \frac {y K _ {0} (y)}{K _ {1} (y)}. \tag {4}
$$

For the single mode cut-off, $y = 0$ (i.e. $\beta = \pm k_{cl}$ ), which is physically interpreted as the mode evanescently decaying into the cladding. Using $\lim_{y\to 0}\frac{yK_0(y)}{K_1(y)} = 0$ , we see that Eq. 4 has solutions when $xJ_0(x) = 0$ . One solution will always exist for $x = 0$ , corresponding to the single remaining mode below cut-off. The penultimate mode comes into existence when $J_0(x) = 0$ for the first time, which occurs at $x = 2.4048$ . Therefore, we can find $d_c$ by rearranging Eq. 3 to get [49]

$$
d _ {c} = \frac {2 . 4 0 4 8 \lambda}{\pi \sqrt {n _ {c o} ^ {2} - n _ {c l} ^ {2}}}. \tag {5}
$$

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 4 of 21

The validity of the WGA is confirmed by comparing the results of Eq. 5 to numerically calculated cutoff diameters, which are summarized for a number of situations in Table 1.

At these diameters there exists a significant evanescent field surrounding the waist region of the tapered fiber. This allows for substantial overlap between an optical resonator's modes and the fiber's guided light when it is approached to an optical cavity. Likewise, light trapped inside the cavity will couple back into the fiber, which will be carried away as optomechanical signal.

While this type of straight tapered fiber is useful for coupling to a single off-chip device, such as a microsphere [37], it is difficult to use as probe of on-chip devices, although it can be done if the device is cleaved to hang over the edge of the chip [26] or isolated using a mesa [52]. Instead, it is useful to introduce a small dimpled region to the fiber, which when oriented towards the sample chip produces a portion of the taper waist that can be used as a probe of an individual on-chip optomechanical device [32]. By combining this probe with a precise positioning system, numerous devices can be sampled using the localized coupling region at the tip of the dimple of the tapered fiber.

# Methods

# Tapered fiber puller

To produce tapered fibers, we use a heat-and-pull method in which a flame from a hydrogen torch is used to soften or melt an optical fiber while simultaneously stretching it at a constant speed. In our system, we produce this flame using a custom-built mountable hydrogen torch, as seen in Figure 3a, which is threaded using a $7/16''$ -24 die (McMaster-Carr, Part No. 26005A128) producing standard threads that allow for interchangeability of torch tips. The tips we use are the HT and OX series purchased from National Torch (see Figure 3d), which provide a wide variety of flame sizes useful for producing different sizes of tapered fibers. The hydrogen torch is fed by a needle valve-controlled line,

Table 1 Single mode cutoff diameters for tapered fibers in a number of situations

![](dt=2026-06-09/ht=00/540d6ed81baf246e834fc1f74ce981b194f824076507a81536da8301f52c1310.jpg)

<table><tr><td rowspan="2">λ (nm)</td><td rowspan="2">nc0</td><td rowspan="2">ncl</td><td colspan="2">dc(nm)</td></tr><tr><td>WGA</td><td>Numerical</td></tr><tr><td rowspan="2">637</td><td>1.47</td><td>1.00</td><td>452.6</td><td>452.6</td></tr><tr><td>1.47</td><td>1.33</td><td>778.8</td><td>778.9</td></tr><tr><td rowspan="2">780</td><td>1.47</td><td>1.00</td><td>554.1</td><td>554.2</td></tr><tr><td>1.47</td><td>1.33</td><td>953.
6</td><td>953.7</td></tr><tr><td rowspan="2">1310</td><td>1.47</td><td>1.00</td><td>930.7</td><td>930.7</td></tr><tr><td>1.47</td><td>1.33</td><td>1601.6</td><td>1601.7</td></tr><tr><td rowspan="2">1550</td><td>1.47</td><td>1.00</td><td>1101.2</td><td>1101.2</td></tr><tr><td>1.47</td><td>1.33</td><td>1895.0</td><td>1895.1</td></tr></table>

Single mode cutoff diameter calculated using both the WGA approximation and numerical calculations for a green light (637 nm) observed in nitrogen vacancy photoluminescence [53] and near infrared light (780 nm) used in aqueous biosensing [54], as well as the dispersionless and low attenuation telecom wavelengths of 1310 nm and 1550 nm. All calculations are performed for both air-clad $(n_{cl} = 1.0)$ and water-clad $(n_{cl} = 1.33)$ environments. The appropriate number of digits are retained to show the difference in WGA and numerical calculations.

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 5 of 21

![](dt=2026-06-09/ht=00/022bd0974de2389f85119892580b5eae88ed399f8831d81e8ddcbe9699422521.jpg)

![](dt=2026-06-09/ht=00/ada18c26c465a9231c76024bf6cc65b31cd796440c2f33a02f78fbc13361a4ca.jpg)

![](dt=2026-06-09/ht=00/5c7b95a0f01bbea5e47902a881334c26dcc6931d1adc82ab181b04f1693bf1f2.jpg)

![](dt=2026-06-09/ht=00/324e8691b201fc98d8fb17e85a926bf501de16c6bcca0208baffaed12c47a088.jpg)

allowing for a very small and stable flame using the OX-00 torch tip, with a single 0.51 mm diameter hole. This tip is chosen because it produces compact tapers (less than $1\mathrm{cm}$ in total length) which are ideal for our fiber holders, while maintaining a relatively high transmission efficiency (up to $\sim 80\%$ ).

The hydrogen torch is mounted on a three-axis positioning system, consisting of automated $xy$ -translation in the plane of the optical table on which the apparatus is mounted, along with perpendicularly oriented manual $z$ -adjustment. The $xy$ -translation system is based on a Zaber T-G-LSM200A200A two-axis gantry system. Each orthogonal axis is driven by a Zaber T-LSM200A linear motorized stage, allowing for a total travel range of $200\mathrm{mm}$ in either dimension with a minimum step size of $50\mathrm{nm}$ .

Manual $z$ -adjustment is provided by a New Focus 9063-COM gothic-arch translation stage mounted using a New Focus 9063-A angle bracket. The stage is manipulated by a Mitutoyo No. 906912 micrometer, providing a $25\mathrm{mm}$ travel range with $10\mu \mathrm{m}$ resolution. This system is used for precise and reproducible placement of the hydrogen torch flame as it heats the fiber, which is an important element required to consistently produce high quality tapered fibers.

The fiber itself is held using two Newport 466A-710 dual arm V-groove fiber holders, each of which is connected to an adjustable optical post mounted on a Zaber T-LSM100A linear motorized stage. Each stage has a travel range of $100\mathrm{mm}$ with a resolution of $50\mathrm{nm}$ and can pull the melted fiber at speeds up to $7\mathrm{mm/s}$ . All of the Zaber stages are automated in software, allowing for precise, reproducible $xy$ -positioning of the torch gantry, as well as the ability to set a consistent pull speed.

The adjustable optical posts help to ensure that the fiber is level, as proper alignment is crucial for producing a low-loss taper. This entire setup is surrounded by a protective box, built from optical rails and acrylic sheets, which helps reduce flame instability due to air currents, as well as preventing contaminants from entering the system.

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4  
http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 6 of 21

Another method by which a fiber can be tapered is using a $\mathrm{CO}_{2}$ laser, which produces radiation with a wavelength ranging from $10.2 - 10.8\mu \mathrm{m}$ [44]. Absorption of these photons by an optical fiber causes it to heat in proportion to the intensity of the beam and the cross section of the fiber being irradiated. Therefore, the power of a $\mathrm{CO}_{2}$ laser must be carefully controlled while pulling a fiber in order to ensure even heating. $\mathrm{CO}_{2}$ lasers have also been used as a heat source for other processes, namely in the production of high-Q silica WGM resonators, such as microspheres [55] and bottles [56].

We integrate a $\mathrm{CO}_{2}$ laser into our fiber pulling system by replacing our hydrogen torch with a cage mount system (see Figure 3b) containing a 45 degree cube-mounted silvered mirror (Thorlabs - Product No. CM1-P01) and a plano-convex ZnSe lens (Thorlabs - Product No. LA7542-F) with a focal length of $25.4\mathrm{mm}$ , which can be used to focus the intense, infrared radiation from the laser onto the fiber. Attaching our lens to the torch positioning system, we gain full control of its position.

This allows for defocusing of the $\mathrm{CO}_{2}$ laser beam, effectively controlling both the size and temperature of the hotspot on the fiber. Furthermore, the focused beam can be scanned along the fiber, allowing for a movable hotspot, which is required for bottle fabrication [54]. Finally, manual height adjustment of the cage mount focusing system allows us to ensure that the $\mathrm{CO}_{2}$ beam will hit the center of the lens, reducing aberration.

It is also possible to attach a microscope imaging system directly to our torch positioning gantry, as shown in Figure 3c. The microscope is comprised of a 10X M Plan Apo long working distance infinity-corrected objective (Edmund Optics - Stock No. #59-877) attached to an Optem Zoom 70XL lens system, allowing for $70\times$ magnification of the setup. This image is recorded using an Edmund Optics EO-5012C color USB webcam, providing a video feed to a nearby computer.

To ensure proper lighting and image quality, light from an external Metaphase MP-LED-150 microscope LED illuminator is coupled into the lens system's coaxial illumination port using a fiber optic waveguide (Edmund Optics - Stock No. #39-368). This system is very useful, as it allows for real time imaging of our completed tapered fibers (and other fabricated optical components) when dimpling or attaching it to its holder, with full three-axis control.

As tapered fibers are quite fragile, it is difficult to move them without breaking. For this reason, we first attach the tapered fiber to a holder, creating a more robust system which can easily be relocated. To this end, we have also included a manually adjusted Newport Compact Dovetail DS40-XYZ three-dimensional linear positioning stage in our system, which allows for $1\mu \mathrm{m}$ sensitivity over a travel range of $14\mathrm{mm}$ in each of $x$ and $y$ and 5 mm in $z$ . This stage allows us to properly position and align the fiber holder, as well as gradually approach it to the fiber for gluing. In addition, it is used to position the fiber mold used in the dimpling process, which must be approached and raised precisely at the thinnest point of the tapered fiber.

Another important aspect of the tapered fiber puller is the fiber transmission monitoring system, which allows us to determine the point at which the taper becomes single mode, as well as assess fiber losses due to tapering. To do this, we measure the transmitted power of light from a New Focus Velocity 6330 tunable diode laser through the fiber during the pulling process. To control the amount of injected power, laser light is first passed through a Thorlabs VOA50-APC variable optical attenuator (VOA) before it is coupled into the fiber using a mechanical splicer (
Fiber Instrument Sales elastomeric lab

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 7 of 21

splice - Part No. FIS114012), in which two straight cleaved fiber ends are butt coupled to each other with the aid of index matching gel (Fiber Instrument Sales matching gel - Part No. F10001V). Likewise, the fiber is mechanically spliced on its opposite end to a patch cable connected to a New Focus Model 1811 IR DC-125 MHz low noise photoreceiver. The DC signal from this photodiode is split off and recorded using an NI USB-6259 BNC DAQ card for the duration of a fiber pull, providing a record of transmission vs pull time, as seen in Figure 4.

# Fiber tapering procedure

To create tapered fibers, we begin with a Corning SMF-28e optical fiber that has a silica core and cladding diameter of $8.2\mu \mathrm{m}$ and $125\mu \mathrm{m}$ , respectively, all of which is protected by an acrylate coating which extends out to a diameter of $245\mu \mathrm{m}$ . The indices of refraction and dimensions of the core and cladding are chosen such that this original fiber is single mode for wavelengths exceeding $1260\mathrm{nm}$ , which includes both the dispersionless and minimum loss wavelengths in silica of $1310\mathrm{nm}$ and $1550\mathrm{nm}$ , respectively.

To begin the tapering process, the acrylate coating is removed using a Micro-Strip® stripping tool over a region approximately 3 cm long in the center of an SMF-28e fiber around one meter in total length. This section of stripped fiber is subsequently cleaned using a solvent to remove any remaining acrylate. The tapering occurs in this stripped region, where the flammable acrylate has been removed. In addition, the two ends of the fiber are stripped of acrylate and cleaved flat using an Ericsson EFC11 fiber cleaver.

Utilizing the mechanical splicers and index matching gel described above, these cleaved ends are spliced to two ends of a severed FC/APC patch cable, one of which leads to the photodiode, the other to the diode laser. This method of fiber splicing is ideal for this application, as it is quick and easy, allowing for a convenient input and removal of the tapering fiber to and from the optical circuit. Losses vary depending on fiber alignment for this splicing method, but we are only concerned with providing enough power to observe variations in fiber transmission.

Once we have ensured that the splices provide sufficient power to the photodiode, the fiber is

![](dt=2026-06-09/ht=00/d056e1e4f38a6809aeaeb328111819147fbc6835c9e1339d07f7d4f3a0508903.jpg)

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 8 of 21

placed in the V-groove fiber holders, with the region prepared for tapering centered between them.

At this point, the hydrogen torch is lit using a butane lighter and gas flow is adjusted to ensure a steady flame about $1\mathrm{cm}$ high. This flame is then approached towards the fiber until a small (a few mm) section begins to glow, indicating that the fiber is in a molten state. Once this point has been reached, the two pulling stages move in opposite directions, each at a constant speed generally chosen to be $40~\mu \mathrm{m / s}$ .

During each pull, the transmission through the fiber vs pull time is monitored, an example of which is presented in Figure 4 for the OX-00 torch tip. By monitoring fiber transmission, it is possible to determine the point at which the fiber waist has become single mode. This will be indicated as a stabilization of the fiber transmission (which is evident in Figure 4) due to the fact that the lossy, higher order modes of the fiber have died out, leaving behind the single fundamental mode of the fiber.

Using images from a scanning electron microscope (SEM - inset of Figure 5), we experimentally measured the diameters of our fibers at the single mode transition to be $\sim 1.1\mu \mathrm{m}$ , consistent with the theoretically predicted diameter for an air-clad fiber with an index of 1.47 (we expect our fibers to have an index of 1.4677) at $1550~\mathrm{nm}$ (see Table 1). By measuring the time required to reach this transition from a single pull, it is possible to determine a value for the hot-zone length $L$ by inverting Eq.

1, provided that the pull speed and initial fiber diameter are known a priori. Using this parameter, we are able to predict the fiber waist diameter for a given pull time. Note that in order for this prediction to be accurate, care must be taken to ensure that all subsequent pulls have conditions matching the original one in order to ensure a consistent hot-zone length. This is readily accomplished using our system. A plot of fiber waist diameter vs pull time using the apparatus described here is presented in Figure 5, indicating excellent agreement between the hot-zone length of

![](dt=2026-06-09/ht=00/380f934de16d71adda5fea99630228916ab8ae9e3069323869a0c8cb8f58232b.jpg)

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 9 of 21

$1.30\mathrm{mm}$ determined using the single mode cutoff point and the fit value of $1.29\mathrm{mm}$ . This ability to predict the fiber waist diameter is useful, as it allows for fabrication of fibers whose diameters support a propagating mode that is phase matched with the resonance we are interested in, enhancing coupling of light from the tapered fiber to the optical resonator [37].

At the point of single mode transition, the fiber waist diameter is small enough to produce the desired evanescent field required for coupling to an optical cavity, which can be seen in the inset of Figure 4. However, it is often advantageous to continue pulling fibers to smaller diameters, further increasing the extent of the evanescent field outside the fiber geometry, allowing for a larger range of coupling before the fiber contacts the optical resonator.

It is possible to create these sub- $\mu \mathrm{m}$ diameter fibers by continuing to pull for a small amount of time ( $\sim 10$ s) after the single mode transition has been reached. Using the OX-00 torch tip, diameters as small as $850~\mathrm{nm}$ can be achieved before the fiber breaks due to the pressure of the flowing hydrogen gas from the torch. By using the HT-3, our largest torch tip, the flame size increases, nearly doubling the hot-zone to $2.4~\mathrm{mm}$ , allowing for the fabrication of tapered fibers with diameters down to 500 nm and $98\%$ transmission.

This provides fibers with diameters small enough that they can be used as a probe of nitrogen vacancy center photoluminescence [55], as well as allow single mode guiding of $780~\mathrm{nm}$ light (Table 1), which is used in aqueous biosensing applications [56].

By monitoring transmission before and after the pull, it is also possible to determine the losses induced in the fiber due to the tapering process. This is important for determining the amount of power injected into the optical resonator, allowing for calculation of the number of photons confined in the optical resonator. For the OX-00 tip, a tapered fiber transmission efficiency of up to $\sim 80\%$ is achieved. By using the HT-3 tip, with its larger hot-zone, a more adiabatic taper transition region is created allowing us to produce fiber tapers with transmission efficiencies exceeding $99\%$ , which is on par with state-of-the-art, ultralow loss fiber pullers [45].

# Fiber dimpling procedure

Once a tapered fiber has been pulled, it is possible to proceed with the dimpling procedure. We begin by taping a stripped Corning SMF-28e optical fiber to the xyz-positioning stage located opposite the hydrogen torch, mounting it perpendicular to the tapere
d fiber so that it can be used as a mold in the dimpling process (see Figure 6a). The fiber mold is prepared by stripping off its acrylate coating and cleaning it with a solvent, producing a mold of $125\mu \mathrm{m}$ in diameter.

In addition, graphite powder (SLIP Plate® Tube-O-Lube®) is applied to the fiber mold to prevent it from sticking to the tapered fiber. This graphite generally burns away when introduced to the hydrogen flame during the dimple annealing process, however, using too much graphite should be avoided as it can contaminate the dimple, inducing losses. To prevent this from happening, a fiber wipe or compressed air can be used to gently remove excess graphite.

To continue, the torch is replaced by the microscope imaging system on the torch positioning gantry so that dimpling can be observed in real time. While watching with the microscope, the tapered fiber is detensioned by approximately $10\mu \mathrm{m}$ to reveal its thinnest point, which appears as a small bend upwards in the fiber (see Figure 6a). The stripped fiber mold is centered on this point and manually raised to touch the tapered fiber using

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4  
http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 10 of 21

![](dt=2026-06-09/ht=00/c1e257f772a9ee6ba9c0565fa1bbf7a304de2c8395f63b318231b71ed0771982.jpg)

![](dt=2026-06-09/ht=00/613f3be01e2b6e2d48c3c2952799213163e8210b3ee6b6b9cae2ff06eb6987b2.jpg)

![](dt=2026-06-09/ht=00/063abb2d67612b04b471716cf1cf60802d165214a40e76ada4ff7eeda8d3f1bf.jpg)

![](dt=2026-06-09/ht=00/488371b3f9aca35e111b4d4248737a5cc97e6518fd6e25accbab46c540e41778.jpg)

![](dt=2026-06-09/ht=00/35eb2963fb5e4f31c055760be65c122e937db1216f774338e3559fc7a7d7f7bf.jpg)

![](dt=2026-06-09/ht=00/24e8ce4bddb52f6f533dc19563c7cdc8d8f07e0dac216e3da1ba27edc4a6b35e.jpg)

![](dt=2026-06-09/ht=00/3887abb0de8c7da76823d7b3a2c00f9a118edfdaf1af411cab9692a8063c7c6c.jpg)

![](dt=2026-06-09/ht=00/52b16dba501a50e3cd95bb0c2084bf3c44ed0aaf121346254731c73c4d7dae23.jpg)

![](dt=2026-06-09/ht=00/b048a4a8de4d78273686dd2d9dcb4f52c665ce7ef1f07faabe5e03bd9f312ed1.jpg)

![](dt=2026-06-09/ht=00/16cb2e2999f30ff9218b810440b400c85de2a608d21f8ca7d303d29dc06bc73a.jpg)

the $z$ -positioning stage. The mold fiber is then raised approximately $5\mathrm{mm}$ , while simultaneously detensioning the tapered fiber, allowing the fiber to wrap itself around the mold producing the desired dimpled shape, as shown in Figure 6b. During this process, the tapered fiber should remain tensioned tightly around the mold at all times to prevent it from twisting.

At this point, a hydrogen flame produced by the tapering torch is introduced to anneal the fiber into a dimpled shape. For this process, one of the HT series torch tips is used, producing a wide flame allowing for the increase in heat distribution required for annealing. This flame is approached to the dimple by hand, touching the mold and tapered fiber lightly (for about one second) until it glows red (see Figure 6c). The mold fiber is then slowly lowered in the same manner it was raised, this time tensioning the tapered fiber, until the mold is returned to its initial position.

The dimple is then removed from the mold by using the unlit torch to flow hydrogen from below, applying a gentle pressure which releases the dimpled fiber. Typically, this process returns a dimple with minimal losses ( $\sim 8\%$ , see Figure 6f). A microscope image of a dimpled fiber produced using this procedure is shown in Figure 6e.

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 11 of 21

# Gluing procedure

Once a dimpled tapered fiber (or other optical component created by the fiber heating system) is produced, it must be carefully attached to its holder using the gluing apparatus. To begin this process, Devcon 5 Minute® epoxy gel (No. 14240) is applied to both sides of the fiber holder, which can be seen in Figure 7a. Care is taken to ensure that both droplets of epoxy are approximately the same height, ensuring that they will contact both sides of the tapered region at the same time.

Once the epoxy is applied to the fiber holder, it is placed on its holding plate located on the gluing apparatus. The fiber holder is then carefully aligned beneath the fiber, ensuring that the fiber will be glued in the appropriate place. Next, the fiber holder is slowly raised using the $z$ -axis of the positioning stages until the fiber has been enveloped in epoxy on both sides of the taper. This initial epoxy is then left to dry (for about 30 minutes) allowing the fiber to be rigidly held on the fiber holder, drastically increasing its durability.

Once the initial epoxy dries, a second round of gluing is typically applied to the fiber, which increases the strength of the fiber's attachment to the holder.

This entire gluing process is monitored in real time using the microscope imaging system mounted on the positioning gantry, which is very helpful as we are able to definitively determine the point at which the fiber has been glued. As well, by imaging the tapered region, along with monitoring transmission down the fiber, we can determine whether or not the tapered fiber has survived the gluing process. Once the fiber has been properly glued in place, it can be transferred directly to the coupling chamber where it is fusion spliced to an existing optical circuit, allowing for injection of light into optomechanical devices.

# Optomechanical coupling chambers

Our coupling chambers, which can be seen in Figure 8b-d, use two separate positioning stage arrangements, each of which have similar principles but different translation

![](dt=2026-06-09/ht=00/4fcb0c6a2689eab1de394f47bd6e9c268fe61094c261e0e448c897ed1bfdc978.jpg)

![](dt=2026-06-09/ht=00/c60d6a401f6be97c8c331475f01e77854f4daf2ccaf99d3ad0bcc775b3f9ca6a.jpg)

![](dt=2026-06-09/ht=00/4d28bde5fe241ea36bc9815f117f8a353c475da877fecd0bce106b9c28698257.jpg)

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 12 of 21

![](dt=2026-06-09/ht=00/9292d297b64efabdab7b87795cc853074ab7a76eaac86e93f83a2cd79828fe6b.jpg)

![](dt=2026-06-09/ht=00/76ed2c08e9136890fedaa8f16a5d34f869ec20146561ca49153c7dc60bab5991.jpg)

![](dt=2026-06-09/ht=00/320b0ee1f32b53803a766c071b8a58886f91ebb046ad53cad70b3d558a87c2cc.jpg)

![](dt=2026-06-09/ht=00/a93f376b80d19617c54af9563e954e5b96f07a103c4c3f661a7db4b88d17679e.jpg)

stages. In each case, the positioning systems are used to ap
proach the optomechanical devices found on the sample chip to a dimpled tapered fiber glued to a stationary custom-machined fiber holder. The fiber was chosen to remain fixed as it is far less stable than the devices on the chip, so its mechanical noise is reduced by anchoring it to an immobile fiber holder.

In the first setup, the sample chip is placed on top of a stack of Attocube linear nanopositioning stages consisting of one ANPz101 stage mounted on top of two perpendicularly oriented ANPx101 stages. The chip is attached rigidly to a custom-machined adapter, which is fastened to the top $z$ -positioning stage. This arrangement provides positioning with sub-nm precision over a total range of $5\mathrm{mm}$ . A picture of this setup can be seen in Figure 8d. The other positioning system is built using Newport Agilis™ AG-LS25V6 vacuum compatible, piezo driven linear stages.

Two of these stages are stacked on top of each other, resulting in orthogonal $xy$ -positioning, with a third mounted at a 90 degree angle for $z$ -translation using an EQ3 Series angle bracket purchased from Newport. As above,

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 13 of 21

the chip containing our optomechanical devices is mounted on a custom-built platform, which is attached to the vertical translation stage providing full three-axis control. These stages provide $50~\mathrm{nm}$ stepping resolution over their entire travel range of $12\mathrm{mm}$ . Each of these systems have different strengths, with the Attocube stack providing extremely precise positioning over a relatively large range, while the Agilis stages provide a more durable, inexpensive alternative with a larger range of motion.

To allow interchangeability between our two chambers, this positioning system is fastened to a custom-machined circular plate, containing $1/4''$ -20 tapped holes in a square pattern with a spacing of $3/4''$ . This plate is then screwed onto a homemade aluminum base with 6 ports, each of which is sealed with an O-ring and provide electrical and optical input/output for the setup, as well as allowing for pressure control inside the chamber.

The optical input/output port consists of fiber feedthroughs, each of which allows for both an input and output fiber, channelling light to and from the tapered fiber. Each fiber is glued in place using Varian Torr Seal high vacuum epoxy, which provides the appropriate seal required for vacuum. For the Attocube setup, the electrical port houses three hermetically sealed BNC feedthroughs. The other type of electrical port, which provides input/output for the Agilis stages, is comprised of a vacuum compatible 15-pin D-type connector housed in a KF50 feedthrough flange (Accu-Glass Products - Model No. 15D-K50).

The vacuum environment provided by our coupling chambers removes airborne contaminants which can reduce the quality of the tapered fiber and optomechanical devices over time [57]. It is also possible to remove such contaminants using a nitrogen purged environment [49], however, performing optomechanics in vacuum has the added advantage of increasing the mechanical quality factors of devices by drastically reducing viscous damping [58]. The vacuum pump port is comprised of a KF25 adapter connected to a turbo pump backed by a dry scroll pump.

By using a completely dry pumping system, we ensure that no oil is ever backstreamed into our system. This connection is made using vibration isolating bellows, which are passed through a cement block to further prevent vibrations from the pump reaching the optical table where the chamber is located. Using this pumping system, we can achieve chamber pressures as low as $10^{-6}$ torr. There also exists a release port, consisting of a Nupro B-4HK brass bellows-sealed valve, which allows for surrounding air to enter the system, re-establishing ambient pressure inside the chamber.

All unused ports on the chamber base are covered with a blank port. This entire system is leak-checked using an Adixen ASM380 dry leak detector, ensuring it is properly sealed.

On top of the chamber base is an aluminum cylinder approximately $10\mathrm{cm}$ long and 17 cm in diameter which provides the housing for the positioning stages and tapered fiber mount. An L-shaped boot gasket (Duniway Stockroom - Part No. VBJG7) is placed on each side of the cylinder providing a leak tight seal between it and both the base and its custom-machined lid. Optical access through the lid is provided by a $75~\mathrm{mm}$ diameter optical flat glass window (Edmund Optics 1/4-Wave N-BK7 - Stock No.

#62-606), which lays flush against an O-ring located in a recessed portion of the lid when the chamber is under vacuum. This window is located directly above the tapered fiber holder and positioning stages, which allows for real time monitoring of the optomechanical system (see Figure 8c). It is therefore possible to view the tapered fiber and on-chip devices while attempting to couple between them, which is important for this process. The fiber and

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4  
http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 14 of 21

chip are imaged using the exact same imaging system described above for the tapered fiber puller. This is made possible by the fact that this microscope is oriented using a three dimensional arrangement of manually positioned New Focus 9063-COM gothic-arch translation stages mounted on a two legged custom-built stand with identical mounting plate to that used for manual $z$ -positioning of the hydrogen torch in the tapered fiber puller, allowing interchangeability of the microscope between the two systems. The Mitutoyo No.

906912 micrometers used to manipulate this positioning system, provide a 25 mm travel range with $10\mu \mathrm{m}$ resolution. This resolution is more than enough to view our devices in the $xy$ -plane of the chip, as well as provide excellent focusing for our imaging setup.

# On-chip optomechanical devices

Due to the small feature sizes required for our optomechanical devices, we have chosen to use foundry-based nanofabrication, which provides high throughput of devices with a minimum feature size of $100\mathrm{nm}$ using top-down DUV photolithography [29]. Each optomechanical device consists of an optical microdisk side-coupled to a mechanical nano/micro-resonator, such as a torsion paddle or a cantilever, as can be seen in Figure 9.

Each device is centered in a large etched area (approximately $100\mu \mathrm{m} \times 50\mu \mathrm{m}$ ), which provides ample room for coupling using our dimpled tapered fiber method. The mask for these devices is designed using custom-programmed Python scripts utilizing the gdspy module, which generates a GDSII file containing our chip layout. This allows us to iterate through a number of device specifications, such as coupling gap, disk radius and mechanical resonator dimensions, providing a large parameter space in which we can explore different optomechanical regimes.

These design files are then submitted through Canadian Microelectronic Corporation (CMC) Microsystems to the Interuniversity Microelectronics Center (IMEC) located in Leuven, Belgium. It is here our devices are fabricated on an 8 inch silicon-on-insulator (SOI) wafer, which consists of a $220\mathrm{nm}$

![](dt=2026-06-09/ht=00/dc12382cdb01152aba8ca3fb4d3556ca655809982b2d77fb2b7eb6325f7a48be.jpg)

![](dt=2026-06-09/ht=00/e7de9dd029f5c1f68a15f956223949b53fba3fbede62225c7265d78dacad562d.jpg)

![](dt=2026-06-09/ht=00/6e13205beb0a0beac3674ee04340da058450e6c36ca5fab8bf03415f2a3784ec.jpg)

Hauer et al. EPJ Techniques
and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 15 of 21

thick layer of single crystal silicon supported by a $2\mu \mathrm{m}$ layer of silicon dioxide. The single crystal silicon device layer is ideal for optomechanical devices, as it has negligible absorption in the telecom band around $1550~\mathrm{nm}$ and a high index of refraction ( $n\approx 3.42$ ), enhancing the mechanical resonator's perturbation of the optical cavity's evanescent field. Our devices are patterned onto this wafer using an excimer laser ( $193~\mathrm{nm} - 248~\mathrm{nm}$ ) and high-definition photomasks derived from our design files.

These patterned wafers are then etched using either a standard or high dose recipe, producing optomechanical devices in the silicon layer which are held rigidly in place by the oxide buffer. This, along with a protective resist coating the entire wafer, help to prevent damage to the devices during transit.

After we receive these wafers, a number of post-processing procedures must be performed in order to prepare our devices for measuring. We begin by dicing the 8 inch wafer into $1\mathrm{cm}\times 1$ cm chips using a diamond saw. Each chip is then ultrasonically cleaned with acetone and rinsed with isopropyl alcohol to remove the protective coating. Once the wafer has been diced and cleaned, a buffered oxide etch (BOE) is used to selectively remove the sacrificial oxide layer beneath our mechanical devices, which releases them, allowing them to oscillate freely.

It is important to note that since BOE is a wet etch, we must ensure that our devices are dried using either a critical point drier or ultralight solvents, such as n-Pentane $(\mathrm{C}_5\mathrm{H}_{12})$ , to prevent stiction. Once a chip has been etched and dried, it is ready to be placed in the chamber for measuring.

# Coupling procedure

Coupling to our optomechanical devices begins by locating the dimple of the tapered fiber using the imaging apparatus. This is done by searching for the portion of the fiber that is in focus at the lowest point (due to the fact that the dimple protrudes away from the rest of the fiber). Once the dimple is found, the nanopositioning stages are used to align the desired optomechanical device such that the lowest point of the dimple is able to couple light into the optical resonator.

The precision of our nanopositioning stack allows for two methods by which we can couple light into the modes of our optical resonators. We can either bring the fiber close enough to the device such that the cavity's optical modes are excited by the fiber's evanescent field or we can simply touch the fiber to the optical resonator. Hovering has the advantage that the excited optical modes are less perturbed by the fiber's presence, resulting in reduced losses. However, by touching the fiber to the resonator, the mechanical instability of the fiber is further reduced. As well, by using this coupling method, it is possible to excite a larger number of optical modes, some of which have higher Qs and larger optomechanical coupling to the mechanical resonator.

# Results and discussion

# Data acquisition: side-of-fringe and homodyne detection

Once we have coupled light into our optical resonators, we begin measuring our devices' mechanical motion. In cavity optomechanics, the motion of a mechanical device shifts the resonance wavelength of an optical cavity by changing its effective length. In our optomechanical systems, this is manifested by the mechanical device oscillating in the optical cavity's evanescent field, modulating its effective index of refraction. We can therefore transduce the mechanical device's motion using amplitude sensitive measurements in

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 16 of 21

the "tuned-to-the-slope" regime [59], which is illustrated in Figure 10a. In this detection scheme, the cavity resonance shift due to the mechanical resonator's motion is transduced by the slope of the optical lineshape into AC transmission fluctuations in the fiber, occurring at the mechanical resonance frequency. Therefore, by tuning our probe laser wavelength to the maximal slope of our optical cavity resonance, we provide optimal optomechanical transduction efficiency, as can be seen in Figure 10b.

In general, this resonant enhancement scales with the system's optical quality factor which increases the slope of the resonance, however, this is convolved with other effects such as an optical mode's volume and overlap with mechanical motion [21].

Using the experimental setup shown in Figure 8a to perform this type of measurement, we have probed devices with an angular resolution of $4\mathrm{~nrad} / \sqrt{\mathrm{Hz}}$ corresponding to a torque transduction on the level of $4\times 10^{-20}\mathrm{N\cdot m / \sqrt{Hz}}$ [21], as well as displacement noise floors of $2\mathrm{fm} / \sqrt{\mathrm{Hz}}$ and force sensitivity of $132\mathrm{aN} / \sqrt{\mathrm{Hz}}$ [18].

It is also possible to perform phase-sensitive measurements on our devices in the "tuned-to-the-peak" regime using a balanced optical homodyne detection system [60], which can be used for quadrature and entanglement measurements [61]. This method of detection also has a number of advantages, including cancellation of laser noise [62] and the ability to lock the laser to the bottom of the optical resonance [17]. Furthermore, since the laser's detuning from the cavity resonance is zero, a maximum number of photons are coupled into the cavity, which enhances the system's optomechanical coupling.

After setting up one of these detection schemes, the AC transmission signal through the fiber is sent to a spectrum analyzer (SA), which outputs its frequency power spectrum. For an optomechanical signal, this includes the power spectral density (PSD) of the mechanical motion [63].

![](dt=2026-06-09/ht=00/830b5fc6860e04a8a4002ff681b2e0d35e32049626812ac2f53af4d9a7332613.jpg)

![](dt=2026-06-09/ht=00/8c54cd2be3576dfe2e71522006e5d3c516cd72781ad525660ccf5de93d2ccbf0.jpg)

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 17 of 21

Alternatively, a time series measurement of the voltage signal taken with an analog-to-digital converter (ADC) can be digitally analyzed to determine its spectral components. In our system, this is performed using a digital lock-in amplifier (Zurich Instruments HF2LI), which is a specialized ADC that first mixes an input voltage signal with a reference frequency, $\omega_{\mathrm{ref}}$ , shifting the frequencies of the input signal by $\pm \omega_{\mathrm{ref}}$ .

Therefore the frequency information around the reference frequency of the input signal is now the low frequency component of the mixed signal. This has the advantage of requiring a data collection rate proportional to the bandwidth of the signal measurement, as opposed to a data collection rate proportional to the maximum frequency component.

For example, when bandwidth of only $100\mathrm{kHz}$ centered at $10\mathrm{MHz}$ contains important spectral information, a data collection rate of $\sim 10^{5}$ samples per second (SPS) can be used as opposed to a rate of $\sim 10^{7}$ SPS, a reduction of about 100 times the data needed to acquire the signal of interest.

After mixing, the lock-in applies a low-pass filter to reduce noise contributions from unwanted frequencies outside the measurement bandwidth, thus the time series output of a lock-in amplifier is the result of a convolution of the lock-in amplifier's filter response with the demodulated input signal, i.e.

$$
Z (t) = X (t) + i Y (t) = \left\{H (t) * e ^ {i \omega_ {\mathrm {r e f}} t}
V (t) \right\} (t), \tag {6}
$$

where $X(t)$ and $Y(t)$ are the two outputs of a dual-phase lock-in amplifier, $H(t)$ is the impulse response of the lock-in amplifier's filter, and $V(t)$ is the input signal to the lock-in. Fourier transforming the output elucidates the convolution, giving

$$
Z (\omega) = H (\omega) V \left(\omega - \omega_ {\text {r e f}}\right). \tag {7}
$$

Thus the spectrum of the lock-in amplifier's output is the spectrum of the input voltage translated in frequency by the reference frequency and enveloped by the lock-in amplifier's filter. The power spectrum can then be estimated by taking $S_{\mathrm{ZZ}}(\omega) = |Z(\omega)|^2$ , or done in practice by using a PSD estimation algorithm such as Bartlett's method [64], giving

$$
S _ {\mathrm {Z Z}} (\omega) = | H (\omega) | ^ {2} S _ {\mathrm {V V}} \left(\omega - \omega_ {\mathrm {r e f}}\right). \tag {8}
$$

This method of data acquisition allows for real time optimization of optomechanical transduction in our devices, which facilitates sensitive probing of their mechanical motion, allowing for precise measurements of physical quantities, such as forces [18] and torques [21].

# Conclusion

This article presents a method by which high efficiency optical coupling is achieved between a dimpled tapered fiber and nanofabricated on-chip optomechanical devices. By using a custom-built automated heat-and-pull fiber puller, it is possible to consistently produce tapered fibers of a predetermined diameter, which is often chosen to be less than the single mode cutoff diameter, providing ample evanescent field for coupling. Dimpling this tapered fiber, using a well-defined procedure, allows for production of an excellent localized probe of planar on-chip devices. Attaching this fiber to a robust holder permits it to be transferred to special coupling chambers. In these chambers, optomechanical coupling is performed in an optical access vacuum environment and is mediated

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 18 of 21

by high precision, nanopositioning stages. By using the amplitude sensitive "tuned-to-the-slope" detection scheme, angular resolution of 4 nrad/√Hz [21] and displacement transduction of 2 fm/√Hz [18] have already been demonstrated. It is anticipated that using this technique for optomechanical coupling, we will be able to continue to measure increasingly sensitive devices, approaching the measurement limits imposed by quantum mechanics [59].

# Competing interests

The authors declare that they have no competing interests.

# Authors' contributions

BDH, PHK and JPD conceived and designed the experiment. BDH, PHK and AJRM constructed the experiment. PHK performed nanofabrication post-processing for the on-chip devices. BDH, PHK, CD, AJRM and HR collected and analyzed data. BDH and AJRM performed theoretical calculations regarding the single mode cutoff diameter for tapered fibers. CD and PHK developed and optimized the procedure for producing dimpled tapered fibers. CD and HR created software for data taking and system manipulation. BDH, PHK, CD and JPD drafted the manuscript. All authors have read, approved and provided critical revisions for the final manuscript.

# Acknowledgements

The authors would like to thank Prof. Paul Barclay for numerous helpful suggestions and insight into both the theoretical and practical applications of optomechanics. We would also like to thank Don Mullin, Devon Bizuk and Greg Popowich for technical assistance. This work was supported by the University of Alberta, Faculty of Science; the Natural Sciences and Engineering Research Council of Canada; Alberta Innovates Technology Futures; the Canada Foundation for Innovation; and the Alfred P. Sloan Foundation.

Received: 21 January 2014 Accepted: 2 April 2014

Published: 29 April 2014

# References

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 19 of 21

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 20 of 21

doi:10.1140/epjti4

Cite this article as: Hauer et al.: On-chip cavity optomechanical coupling. EPJ Techniques and Instrumentation 2014 1:4.

# Submit your manuscript to a SpringerOpen journal and benefit from:

Submit your next manuscript at $\triangleright$ springeropen.com

Hauer et al. EPJ Techniques and Instrumentation 2014, 1:4

http://www.epjtechniquesandinstrumentation.com/content/1/1/4

Page 21 of 21