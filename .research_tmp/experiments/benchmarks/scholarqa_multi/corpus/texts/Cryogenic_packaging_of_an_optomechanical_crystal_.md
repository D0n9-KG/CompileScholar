# Alignment-free cryogenic optical coupling to an optomechanical crystal

Timothy P. McKenna, $^{1, a)}$ Rishi N. Patel, $^{1, a)}$ Jeremy D. Witmer, $^{1, a)}$ Raphaël Van Laer, $^{1, a)}$ Joseph A. Valery, $^{1}$ and Amir H. Safavi-Naeini $^{1, b)}$

Ginzton Laboratory, Stanford University, 348 Via Pueblo Mall, Stanford, California 94305, USA

(Dated: 11 April 2019)

The need for highly accurate, labor-intensive optical alignment has been a major hurdle in our ability to leverage the power of complex photonic integrated circuits. There is a strong need for tolerant and passive alignment methods that enable interrogation at any point in a photonic circuit. Various promising and scalable photonic packaging techniques have been under development, but few methods compatible with low-temperature operation have been reported.

Here, we demonstrate alignment-free $25\%$ coupling efficiency from an optical fiber to a silicon optomechanical crystal at 7 mK in a dilution refrigerator. Our coupling scheme uses angle-polished fibers glued to the surface of the chip. The technique paves the way for scalable integration of optical technologies at low temperatures, circumventing the need for optical alignment in a highly constrained cryogenic environment. The technique is broadly applicable to studies of low-temperature optical physics and to emerging quantum photonic technologies.

# I. INTRODUCTION

The rise of integrated optical systems has motivated the development of numerous approaches to efficiently couple light from a fiber's $\approx 10\mu \mathrm{m}$ diameter optical mode into sub-micron confined channels on the surface of a chip<sup>1</sup>. The emerging application space of quantum sensors, computers, and communication systems necessitates its own new classes of integrated optical devices for quantum communications and control. A key challenge is to efficiently couple light in and out of these devices.

Since quantum systems often have components that operate at extremely low temperatures, optical systems operating at these temperatures require coupling schemes that are both scalable and immune to the thermal stresses induced by thermal cycling. In this work, we demonstrate an approach to coupling light in and out of a silicon photonic chip at millikelvin temperatures that is robust, requires no in-situ alignment and is compatible with numerous quantum technologies<sup>2-6</sup>.

Common methods to address the optical input/output barrier include inverted edge tapers, evanescent couplers, and grating couplers $^{1,7,8}$ . Tapered fiber coupling, where a tapered fiber evanescently couples to an on-chip waveguide, has been used to achieve broadband efficient coupling at both room and cryogenic temperatures $^{9-11}$ . The technique uses specially formed fiber tapers and requires in-situ alignment at cryogenic temperatures, which limits the number of input-output channels and drastically increases the cost and complexity of the apparatus.

Moreover, at low temperatures and in vacuum, the technique is subject to vibrational noise and power-dependent instabilities. Edge coupling has been demonstrated at cryogenic temperatures $^{12-14}$ but also requires in-situ alignment.

Another common coupling technique utilizes gratings patterned on the surface of a chip. Well-designed grating couplers can in principle convert over $90\%$ of incoming light into a waveguide on the chip over a bandwidth of several THz with

reflections below $1\%$ and using a footprint of merely tens of square microns[1,7,15-18]. They allow for optical access to any point on a wafer while requiring only micron-level alignment accuracy, drastically reducing the resources necessary to interrogate complex optical circuitry. Moreover the relatively high tolerance to misalignment makes grating couplers a promising candidate for cryogenic coupling where thermal stresses increase the likelihood of the fiber moving with respect to the grating coupler. As an example of this, Shainline et al.

[19] recently demonstrated a technique in which an optical fiber can be aligned to a grating coupler with the assistance of an SU-8 collar, and achieved coupling efficiencies of $21\%$ at a temperature of $1\mathrm{K}$ without requiring low-temperature re-alignment.

Light emitted from a grating coupler is often mode-matched to SMF-28 fiber so that a cleaved fiber facet can be used to efficiently collect the radiation. The fiber, or fiber assembly, must be held at design-determined angle to the normal of the chip surface for efficient coupling. In the work by Shainline et al.19, this was accomplished by extra layers of processing to build a supporting structure on the surface for the fiber, and a larger assembly around the chip to hold the fiber at an angle.

The extra processing steps make that approach difficult to combine with released optomechanical structures. An alternative approach is to use angle-polished fibers20,21 that can rest horizontally on the chip surface, obviating the need to support a fiber at an angle and leaving space for electrical connections. Total internal reflection occurring at the fiber/air interface sends light into an on-chip grating coupler at the correct angle for efficient coupling (see Figure 1a).

Here, we propose and develop a fiber-chip coupling technique, based on grating couplers and angle-polished fibers, and use it to demonstrate cryogenic measurements down to millikelvin temperatures. The technique is scalable in the sense that it provides access to many optical input and output ports on a chip at cryogenic temperatures without requiring a corresponding number of stages and control wires for preserving alignment. We demonstrate the technique by measuring a silicon optomechanical crystal in a dilution refrigerator at $7\mathrm{mK}$ . The technique has good yield and the packaged devices remain stable over many cooling cycles. It circumvents the need for time-consuming manual alignment in a constrained

Alignment-free cryogenic optical coupling to an optomechanical crystal

a)These authors contributed equally.

b)Electronic mail: safavi@stanford.edu.

arXiv:1904.05293v1 [physics optics] 10 Apr 2019

cryogenic environment and paves the way for complex circuits with many inputs and outputs operating at low temperatures for integrated quantum optical $^{22,23}$ , optomechanical $^{12,24-27}$ , and electro-optic $^{6,28,29}$ devices.

# II. PROCEDURE

# A. Device Fabrication

We fabricate gratings couplers attached to optomechanical crystals using a two-step silicon-on-insulator (SOI) process. First, we pattern fine features in silicon by ebeam lithography and a $\mathrm{Cl}_2$ -based etch $^{12,26}$ . Second, we selectively suspend the optomechanical crystals but not the grating couplers by optical lithography and a buffered HF etch $^{24,30}$ . We fabricate an array of such devices with a parameter sweep in the overall scaling. This allows us to frequency match the optical cavity resonance with the grating coupler response. An image of a device array is shown in Figure 1, along with an image of the fiber-coupled device.

![](dt=2026-06-03/ht=14/e09545f243ea7315c7debe01f81ee9c8593948da0aa75f9fc9a6af4bd3452230.jpg)

![](dt=2026-06-03/ht=14/5059a6e447f14a0470921a5a0112a9faa1ed39975d35574a6e98af36ea78a325.jpg)

The grating coupler consists of a square array of rectangular holes fully etched into the $220\mathrm{nm}$ silicon slab according to a metragrating design with simulated efficiencies up to $60\%$ .

It has an impedance-matching set of smaller holes to reduce reflections off the grating (Figure 1(c)).

# B. Fiber Gluing Procedure

To couple light into the on-chip grating couplers, we use angle-polished SMF-28 fibers (Chuxing Optical Fiber Application Technologies Ltd.) with a polish angle of approximately
$35^{\circ}$ . The fiber is positioned horizontally above the surface of the chip. The light in the fiber experiences total internal reflection at the fiber facet and is reflected down through the bottom of the fiber, impinging on the chip surface with an angle of approximately $27^{\circ}$ from normal when no glue is present. This angle was chosen to recycle light reflected off the handle silicon<sup>7</sup>. The coupling scheme is illustrated in Figure 1(a).

To prepare for fiber gluing, we first fasten the chip to a custom-made copper printed circuit board (PCB) using a GE Varnish/ethanol mixture. The PCB is then mounted in our fiber gluing setup, shown in Figure 2(a).

We align the angle-polished fiber to the device of interest before applying the glue. Next, we adjust the fiber X, Y, Z, yaw and roll in order to optimize the optical reflection signal from the grating coupler. We have also used this procedure for transmission devices with multiple couplers. We then apply a small drop of glue (Norland Optical Adhesive 88) to the fiber, about 0.5 to $1\mathrm{mm}$ away from the tip of the fiber. To do this we use a spare piece of SMF-28 fiber as an applicator and control the applicator position with an XYZ micrometer stage stack. After touching down the glue-tipped applicator, we move the applicator until the glue wicks underneath the fiber and covers the grating coupler by capillary action.

After the glue is applied, we find it necessary to adjust the fiber X and Y position to reoptimize the reflected signal. After reoptimizing the position, we cure the glue using a UV curing spot lamp (DYMAX BlueWave 75). We use the lamp at full power and do 6 cures of 30 seconds each, at a distance of about $3\mathrm{cm}$ and from a few different angles.

Once the fiber is glued to the chip, we apply a second, much larger drop of glue to fasten the fiber to the PCB and provide strain relief. We make sure that this second drop covers a portion of the fiber where the fiber coating is still intact to provide strength. We cure this glue in the same way as the first drop. After curing the glue, we typically leave the chip in an air environment for about 12 hours, allowing the glue to age and achieve better adhesion before installing it in the dilution refrigerator.

Once the fiber gluing process is complete, the PCB holding the chip is installed at the mixing chamber of a Bluefors dilution refrigerator (see Figure 2(d)). The dilution fridge is fitted with a series of optical fibers that provide optical access from room temperature down to the mixing chamber.

Alignment-free cryogenic optical coupling to an optomechanical crystal

2

![](dt=2026-06-03/ht=14/5ead2fe2dae6ccc5027c4005f1d4b280b858e80f2ed7dfb0c152d29b78929401.jpg)

![](dt=2026-06-03/ht=14/6f416ccf1f3ffb166c5ca0585b61ce14d6bcb19790e156180a4f045597b9acee.jpg)

![](dt=2026-06-03/ht=14/adb1b86f40ee970d0af88943ff4498665b8977a31974a38b646c3965326d2308.jpg)

![](dt=2026-06-03/ht=14/a594e7ba44dc15089845e56632adc667f7bb13a9a3c511d3dec77c96b7d3ff6f.jpg)

# III. EXPERIMENTAL RESULTS

# A. Predicting Spectral Shifts

Our optical devices experience two spectral shifts during the gluing process and subsequent cooldown. First, once the glue is applied, the grating coupler spectrum undergoes a large redshift of about $91~\mathrm{nm}$ (not shown). This shift stems from the refractive index of the glue $(\approx 1.56)$ causing an increase in the Bloch mode index of the grating coupler. Experiments performed with oxide-clad grating couplers do not exhibit this redshift. Since the glue stays localized near the coupler, the optomechanical crystal does not undergo a frequency shift upon application of the glue.

Second, both the grating coupler spectrum and the optical resonance frequency of the optomechanical crystal experience a blueshift as the chip is cooled from $300\mathrm{K}$ to millikelvin temperatures (Figure 3).

To account for these shifts, we initially measure the optical reflection spectra in air before applying any glue. We use these spectra to predict device behaviour after gluing and cooling to low temperatures. Devices of interest are selected for gluing based on their predicted behavior after the cooldown.

To understand and predict the spectral shifts quantitatively, we develop a few analytical and finite-element models. For the grating coupler, the phase-matching condition between light incoming at angle $\theta = 27^{\circ}$ and a guided wave with wavevector $\beta$ is

$$
\beta (\omega) = k _ {0} (\omega) n _ {t} \sin (\theta) + G \tag {1}
$$

with $\beta = k_{0}n_{\mathrm{eff}}$ the Bloch wavevector of the guided optical mode, $k_{0} = \omega /c$ the free-space wavevector, $n_{\mathrm{eff}}\approx 2.4$ the effective Bloch index, $n_t$ the refractive index of the medium atop the grating, which is either air $(n_t = 1)$ or glue $(n_t = 1.56)$ and the grating wavevector $G = 2\pi /\Lambda$ with $\Lambda = 0.81\mu \mathrm{m}$ the grating pitch. The phase-matching condition rewritten in terms of the center wavelength $\lambda_{c}$ of the grating is

$$
\lambda_ {c} = \Lambda \left(n _ {\text {e f f}} \left(\lambda_ {c}\right) - n _ {t} \sin (\theta)\right) \tag {2}
$$

We are interested in perturbations of the center wavelength given perturbations in (1) the refractive index $n_t$ of the medium atop the grating and (2) the refractive index $n_{\mathrm{Si}}$ of the silicon. We must take into account the dispersion $n_{\mathrm{eff}}(\lambda_c)$ of the Bloch mode's effective index to predict the size of such perturbations. Taylor-expanding the dispersion and truncating to first-order yields

$$
\delta \lambda_ {c} = \delta n _ {\text {e f f}} \frac {\Lambda}{1 + \frac {\Lambda}{\lambda_ {c}} (n _ {g} - n _ {\text {e f f}})} \tag {3}
$$

with $n_g = n_{\mathrm{eff}} - \lambda \frac{\partial n_{\mathrm{eff}}}{\partial \lambda} \approx 3.5$ the group index of the Bloch mode. This expression assumes a fixed $n_t \sin \theta$ , which holds in all cases of interest here since horizontal momentum is conserved at interfaces between the fiber and the glue or air.

First, we apply equation (3) to consider the shift of the grating coupler center wavelength as we apply the glue. Our finite-element models predict a shift in the Bloch index of $\delta n_{\mathrm{eff}} \approx 0.21$ . This yields a predicted redshift $\delta \lambda_c \approx 102\mathrm{nm}$ in reasonable agreement with the observed redshift $\delta \lambda_c \approx 91\mathrm{nm}$ .

Alignment-free cryogenic optical coupling to an optomechanical crystal

3

Second, as we cool down from $300\mathrm{K}$ to $7\mathrm{mK}$ , the refractive index of silicon drops $^{31}$ from 3.486 to 3.453 so $\delta n_{\mathrm{Si}} = -0.033$ . This perturbs the Bloch index as $\delta n_{\mathrm{eff}} \approx -0.025$ , leading to predicted blueshift $\delta \lambda_{c} \approx -12.7\mathrm{nm}$ agreeing reasonably with the observed shift of the grating coupler, $\delta \lambda_{c} \approx -10.5\mathrm{nm}$ (Figure 3). We neglect the temperature-dependence of the refractive index of the glue and the silicon dioxide in this calculation.

Third, we can model the optomechanical crystal as a cavity made of a waveguide supporting an optical mode with group index $n_g \approx 4$ . The shift for such a cavity is given by

$$
\delta \lambda_ {c} = \lambda_ {c} \frac {\delta n _ {\text {e f f}}}{n _ {g}}, \tag {4}
$$

giving a predicted blueshift of $\delta \lambda_{c} \approx -13\mathrm{nm}$ . Finite-element modeling of the optomechanical crystal cavity optical mode confirms this: treating the refractive index shift as a perturbation<sup>31,32</sup> yields $\delta
\lambda_{c} = -12.6\mathrm{nm}$ in a finite-element simulation – in close agreement with the observed blueshift (Figure 3).

# B. Optical Measurements

After gluing, the devices are mounted inside the cryostat where characterization is performed at different temperatures. We send an optical pump from a laser tuned to approximately $1550~\mathrm{nm}$ to the glued angle-polished fiber. Since the device input coupling waveguide is terminated in a photonic crystal mirror whose bandwidth exceeds the grating coupler bandwidth, and whose reflection coefficient can be taken to be unity, knowledge of the external system efficiency allows us to calculate the single-pass efficiency of the grating coupler.

We summarize the observed device efficiencies over three thermal cycles in Table I. We show the reflection spectra for one device at room temperature and at $100\mathrm{mK}$ in Figure 3(a). The broad response in the reflection spectrum is due to the grating coupler, while the narrow dips are due to the optical cavity resonances. There are two such resonance per device as there are two optomechanical crystals present per coupling waveguide (Figure 1(e)). In our devices we observe optical quality factors on the order of $10^{5}$ , with no significant variation across cooldowns. An overall blueshift of the cavity wavelength and grating spectrum of approximately $12.6\mathrm{nm}$ is observed, in excellent agreement with finite-element simulations of our structure (see above).

Occasionally a glued fiber comes loose during a cooldown and the optical reflection signal is lost outright. We believe thermal contraction of the glue is responsible for these occasional failures. The overall rate of such failures is $< 25\%$ , and if a given fiber survives its first cooldown, we find that it is highly likely to survive subsequent co depresss. In this study device 1 failed after one thermal cycle as shown on the first row of Table I. However, all other devices survive multiple cycles. In particular, device 3 remained intact for 10 co depresss over the course of 6 months before being removed from the dilution fridge, without suffering substantial degradation in its optical coupling efficiency.

![](dt=2026-06-03/ht=14/2c5c09b3a74d23d09f2bf124a2fbc4afdaa6bf67f65e8f288bf28ab9a00ef7be.jpg)

![](dt=2026-06-03/ht=14/f4c89ba32755d61e0b80ecf9f14135f1df01a2ceda773b8a998e03e65ba02f10.jpg)

![](dt=2026-06-03/ht=14/e4c7ebc212cf5a6a8f3f510ede7aa70c00cf0e64933a22b4c822b8141c321810.jpg)

We make optomechanical measurements of the devices under continuous wave laser driving. For a device cooled to $T = 100 \mathrm{mK}$ , we monitor the $3.8 \mathrm{GHz}$ mechanical mode and measure a mechanical quality factor $Q \approx 2 \cdot 10^{5}$ and a single photon optomechanical coupling rate $g_{0} / 2\pi = 726 \mathrm{kHz}$ (Figure 4). For an input laser power $P \approx 8\mu \mathrm{W}$ we reach a cooperativity of $C \approx 0.8$ - sufficient for itinerant state-transfer applications. These parameters are comparable to those used in recent heralded quantum entanglement experiments with mechanical systems<sup>14</sup>.

Further improvements to our devices should allow detection efficiencies of $60\%$ and higher to be attained[1,7,16-18], approaching typical efficiencies in experiments using lensed fibers and adiabatic edge couplers[12]. One promising application of our packaging technique is the ability to simultaneously monitor multiple devices. In such schemes, the number of devices that can be probed is only limited by chip space, and the number of fiber feedthroughs on the dilution fridge. This is in contrast to using lensed fibers mounted on stages, where independent control over multiple fibers quickly becomes infeasible. Finally, it is important to note that the grat

Alignment-free cryogenic optical coupling to an optomechanical crystal

4

TABLE I. Single-pass optical coupling efficiency for four devices over three thermal cycles. Device 1 failed after the first thermal cycle, although we find all other devices remain intact for multiple cycles.

![](dt=2026-06-03/ht=14/c3cf3a0d146d0e7ccd868838821cfc191552688f5d21b2f023f25b0fef43ce24.jpg)

<table><tr><td>Device</td><td>T = 300 K</td><td>116 mK</td><td>300 K</td><td>7 mK</td><td>300 K</td><td>7 mK</td><td>300 K</td></tr><tr><td>η1</td><td>0.29</td><td>0.21</td><td>0.26</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>η2</td><td>0.36</td><td>0.25</td><td>0.40</td><td>0.23</td><td>0.37</td><td>0.20</td><td>0.34</td></tr><tr><td>η3</td><td>0.28</td><td>0.20</td><td>0.22</td><td>0.21</td><td>0.22</td><td>0.16</td><td>0.22</td></tr><tr><td>η4</td><td>0.25</td><td>0.20</td><td>0.24</td><td>0.19</td><td>0.24</td><td>0.21</td><td>0.25</td></tr></table>

ing coupler must be frequency matched to the cavities of interest in any given experiment. However, this problem is readily solved by the fabrication of arrays of devices, and rapid large area testing at room temperature. Using the relations shown in this paper for wavelength shifts due to gluing and thermo-optic effects, we can deterministically predict the cavity and grating wavelength post-cooldown and select suitable devices for gluing.

![](dt=2026-06-03/ht=14/b0a674069fcfab0da5e1019ff923e094971c6d56de4016b08e852ecfb359a825.jpg)

![](dt=2026-06-03/ht=14/c10029f70c133e67333b8f93025e1257e25475c5ae6f0d820af863076d5d3a9d.jpg)

# IV. CONCLUSION

We demonstrate alignment-free coupling with $25\%$ coupling efficiency from an optical fiber to a silicon optomechanical crystal at $7\mathrm{mK}$ in a dilution refrigerator. The developed angle-polished fiber gluing technique has good yield, and the packaged devices remain stable over many cooling cycles. In the future, the technique can be expanded to individually monitor many devices on a single chip in a cryogenic environment. Additionally, the gluing principle is not limited to the use of grating couplers and can be extended to other coupling schemes such as tapered fiber and edge coupling.

# ACKNOWLEDGMENTS

This work was funded by the ARO/LPS CQTS program and NSF ECCS-1708734. Part of this work was performed at the Stanford Nano Shared Facilities (SNSF) and Stanford Nanofabrication Facility (SNF) which are supported by the National Science Foundation under award ECCS-1542152. A.S.N. acknowledges the support of a David and Lucile Packard Fellowship. R.V.L. acknowledges funding from VOCATIO and from the European Union's Horizon 2020 research and innovation program under Marie Skłodowska-Curie grant agreement No. 665501 with the research foundation Flanders (FWO). J.D.W.

acknowledges support from a Stanford Graduate Fellowship. R.N.P. is supported by a National Science Foundation Graduate Research Fellowship under grant no. DGE1656518. R.V.L. thanks Roel Baets and Dries Van Thourhout for helpful discussions.

Alignment-free cryogenic optical coupling to an optomechanical crystal

5

Alignment-free cryogenic optical coupling to an optomechanical crystal

6