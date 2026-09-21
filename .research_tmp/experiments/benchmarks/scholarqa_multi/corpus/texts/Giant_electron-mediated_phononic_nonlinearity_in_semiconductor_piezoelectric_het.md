# High-Efficiency Three-Wave and Four-Wave Phonon Mixing Via Electron-Mediated Nonlinearity in Semiconductor-Piezoelectric Heterostructures

Lisa Hackett,\(^{1}\) Matthew Koppa,\(^{1}\) Brandon Smith,\(^{1}\) Michael Miller,\(^{1}\) Steven Santillan,\(^{1}\) Scott Weatherred,\(^{1}\) Shawn Arterburn,\(^{1}\) Thomas A. Friedmann,\(^{1}\) Nils Otterstrom,\(^{1}\) and Matt Eichenfield\(^{1,2}\)

1. Microsystems Engineering, Science, and Applications, Sandia National Laboratories, Albuquerque, NM 87123, USA  2. College of Optical Sciences, University of Arizona, Tucson, AZ, 85719, USA

Correspondence: Matt Eichenfield (eichenfield@arizona.edu)

# Abstract

We show that phononic frequency conversion can be enhanced by orders of magnitude in piezoelectric systems by heterogeneous integration of high-mobility semiconductor films. A lithium niobate and indium gallium arsenide heterostructure is utilized to demonstrate efficient three-wave mixing processes at microwave frequencies, including (16±6)% phononic power conversion efficiency for sum-frequency generation and (1.0±0.1)% phononic power conversion efficiency for difference-frequency generation, as well as the most efficient degenerate four-wave phononic mixing to date.

We present a theoretical model that accurately predicts the sum-frequency and difference-frequency generation processes and we show that the conversion efficiency can be further enhanced by the application of semiconductor bias fields. Laser Doppler vibrometry is then applied to examine many three-wave and four-wave mixing processes simultaneously in the same device.

Through the use of our developed model, we show that these nonlinearities can be enhanced far beyond what is demonstrated here by confining phonons to smaller dimensions in waveguides and optimizing semiconductor material properties or using 2D semiconductors.

# Introduction

IntroductionNonlinear acoustic interactions have the potential to completely transform the way we think of and use phonons in the solid state, just as nonlinear optical interactions have done for photons.\(^{1-3}\) In the classical domain, nonlinear acoustic frequency conversion can be used to create frequency conversion devices, such as mixers and correlators,\(^{4-7}\) that can greatly enhance radio frequency signal processing or even form the basis for all-acoustic radio frequency signal processing on a chip.

\(^{8}\) In the quantum domain, strong nonlinear multi-phonon interactions could allow for quantum phononic processes such as squeezing, parametric amplification, state-preserving frequency conversion, and phononic quantum logic to be harnessed in ways analogous to their use in quantum photonics.\(^{9-11}\) Nonlinear phononic processes are also known to limit the coherence of phonons in materials in important regimes.

\(^{12,13}\) Thus the ability to understand, modify, and control those nonlinearities can offer deep insight into or enable the control of heat transport,\(^{14,15}\) the kinetics and dynamics of charge carriers,\(^{16-18}\) and other important solid state physical processes. However, much like optical nonlinearities, the strength and utility of acoustic nonlinearities in solids are entirely material dependent.

While materials like lithium niobate do show some promise for nonlinear acoustics, with useful demonstrations of electro-acoustic effects,\(^{19}\) nonlinear piezoelectric effects,\(^{20}\) and three-wave\(^{21,22}\) and four-wave mixing processes,\(^{23-27}\) thus far strong acoustic nonlinearities that enable high-efficiency frequency conversion in solid state materials have not been identified.

In this work, we provide an alternative approach to achieving strong phononic nonlinearities by demonstrating efficient phononic mixing processes in piezoelectric acoustic systems by heterogeneous integration of semiconductor films. This enhancement to phononic frequency conversion efficiency occurs due to hybridization of the phonons with electronic waves in the semiconductor, with the degree of hybridization determined by the strength of the piezoelectric phonons' longitudinal electric field created in the semiconductor.

Strong electronic nonlinearities then can mediate strong phononic nonlinearities. We use this acoustoelectrically enhanced frequency conversion efficiency to demonstrate highly efficient three-wave mixing processes at microwave frequencies, including a \((16\pm6)\%\) phononic power conversion efficiency for sum-frequency generation and \((1.0\pm0.1)\%\) phononic power conversion efficiency for difference-frequency generation, as well as the most efficient degenerate four-wave phononic mixing to date.

We show the frequency conversion can be further enhanced by application of bias fields in the semiconductor. In the case of four-wave phononic mixing, the natural phase matching of the process enables us to directly experimentally show that the improvement to the phononic frequency conversion efficiency is from the enhanced nonlinear coefficient due to the presence of the semiconductor film. We present a theoretical model that accurately predicts the sum-frequency and difference-frequency generation processes.

We then use laser Doppler vibrometry to examine many three-wave and four-wave mixing processes simultaneously in the same device, including the ones described above and others such as second and third harmonic generation. Finally, we use the theoretical model to show that these nonlinearities, and the resulting power conversion efficiency, can be greatly enhanced far beyond even what is demonstrated here by confining phonons to smaller dimensions in waveguide circuits and optimizing semiconductor material properties or using 2D semiconductor materials.

This work paves the way for deterministic integration of semiconductor materials into piezoelectric acoustic materials and circuits for the purpose of producing, studying, controlling, and using strong

phononic nonlinearities, which can be combined with other acoustoelectric effects to produce novel classes of phononic devices and materials systems.

# The acoustoelectric nonlinearity

Here we study frequency conversion of collinear and co-propagating phonons mediated by an acoustoelectric nonlinearity in a semiconductor piezoelectric heterostructure, as illustrated in Fig. 1(a). As an illustrative example, a parametric three-wave mixing process is shown where phonons at frequencies \(\omega_{1}\) and \(\omega_{2}\) interact nonlinearly to generate a phonon at the sum frequency \(\omega_{3}=\omega_{1}+\omega_{2}\).

The phonons in this work are produced in modes of a planar piezoelectric waveguide, which endows the phonons with electric fields that extend evanescently above the surface of the piezoelectric material. A nonlinear interaction region is created by selectively patterning a semiconductor layer to be directly above the waveguide, where the phonons' electric fields couple to and become hybridized with acoustoelectrically generated propagating charge distributions in the semiconductor.

The strong coupling and hybridization between the semiconductor charge and piezoelectric phonons allows the electronic nonlinearities to mediate nonlinear interactions between the phonons.

The system we utilize is a semiconductor piezoelectric heterostructure on a silicon substrate that allows independent control over the acoustic, electromechanical, and semiconducting material parameters that determine the linear and nonlinear acoustic response. The specific heterostructure utilized in this work consists of an indium gallium arsenide \((\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As})\) semiconducting film with a thickness of approximately \(50\;\mathrm{nm}\) on a \(5\;\upmu\mathrm{m}\) thick lithium niobate \((\mathrm{LiNbO}_{3})\) piezoelectric layer.

The \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) is grown by metal-organic chemical vapor deposition within a \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}
\mathrm{As}/\) indium phosphide (InP) epitaxial ternary heterostructure on an InP wafer. The entire epitaxial stack is lattice-matched to InP, enabling a low semiconductor defectivity and therefore high mobility, \(\mu\).

As discussed in detail in the Methods, the complete heterostructure is formed by first bonding the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}/\mathrm{InP}\) stack to a \(\mathrm{LiNbO}_{3}\)-silicon substrate followed by removal of the InP substrate. The remaining epitaxial layers are patterned and etched to create the semiconductor nonlinear interaction region \((\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As})\) and semiconductor mesa contacts to allow drift fields and currents to be applied to that layer (see Methods).

\(\mathrm{LiNbO}_{3}\) is a highly anisotropic piezoelectric material where the electromechanical coupling coefficient, \(k^{2}\), can be exceptionally large for specially chosen acoustic modes and directions. Here we target a mode with quasi-shear horizontal (quasi-\(SH_{0}\)) polarization that propagates along the \(x\)-direction in \(y\)-cut \(\mathrm{LiNbO}_{3}\) with a \(k^{2}\sim18\%\). The modeled longitudinal electric field and displacement field for the quasi-\(SH_{0}\) mode in our acoustoelectric heterostructure is shown in Fig. 1(b).

As can be seen, the electric field value of the phonon mode extends above the surface of the piezoelectric material into the semiconductor and peaks roughly at the interface between the two, which enables the strong coupling between the phonon and semiconductor charge carriers.

Fig. 1: Phononic frequency mixing mediated by an acoustoelectric nonlinearity. (a)

![](dt=2025-08-07/ht=23/6aa69d42f2d9568c2e0e3534dde1adb3c4e051d888949f777d8cf413473d3e66.jpg)

Illustration of sum frequency generation with collinear and co-propagating piezoelectric phonons coupled to a semiconductor charge carrier system. The frequency conversion is mediated by an acoustoelectric nonlinearity in the heterostructure system. In the actual devices, the acoustic and electronic waves physically overlap.

(b) Modeled longitudinal electric field \((E_{x})\) and z-component of the displacement field \((u\cdot\hat{z})\) for a guided piezoelectric acoustic wave with quasishear horizontal \((S H_{0})\) polarization in a \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As-LiNbO}_{3}\) film stack on a silicon substrate. The \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) thickness is \(50\;\mathrm{nm}\), the \(\mathrm{LiNbO}_{3}\) thickness is \(5\;\upmu\mathrm{m}\), and the modeled acoustic wavelength, \(\varLambda\), is \(\varLambda=16\;\upmu\mathrm{m}\).

In the remainder of this Article, we first assess two parametric three-wave mixing processes: sum and difference frequency generation, which rely on a second-order acoustoelectric susceptibility, \(X_{AE}^{2}\). We examine the power-dependent conversion efficiency in the absence of biasing electric fields in the semiconductor. We then show how the sum frequency conversion efficiency can be modified through an applied static electric field in the semiconductor.

The study of frequency conversion is then extended to parametric four-wave mixing, utilizing the third-order acoustoelectric susceptibility, \(X_{AE}^{3}\). We then demonstrate that laser Doppler vibrometry can be used to capture a broad range of nonlinear mixing processes in a single device and show that myriad nonlinear processes occur in these heterostructures that can be studied in future work.

A model is developed to assess how to maximize the nonlinearities while simultaneously minimizing loss in these systems by proper choice of materials and geometry, and we show a path to achieve power conversion efficiencies significantly beyond what is demonstrated here for sum frequency generation.

# Sum and difference frequency generation

Figure 2(a) shows the energy level diagram for sum frequency generation. Energy conservation requires that two phonons, one at \(\omega_{1}\) and one at \(\omega_{2}\), are annihilated to create a phonon at \(\omega_{3}=\omega_{1}+\omega_{2}\). Figure 2(b) shows a microscope image of a device implemented in the semiconductor piezoelectric heterostructure to study this frequency conversion process. To generate and detect phonons, we utilize interdigital transducers that consist of interlaced metal electrodes on the piezoelectric surface. Two of these transducers are designed and fabricated to generate collinear and copropagating pump and signal phonons at \(f_{1}=\omega_{1}/2\pi\) and \(f_{2}=\omega_{2}/2\pi\), respectively.

These phonons propagate through a \(250\;\upmu\mathrm{m}\) long region of patterned \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconductor film, which supports the acoustoelectric interaction. The output phonons then pass under an interdigital transducer designed to be resonant at \(f_{3}=\omega_{3}/2\pi=\omega_{1}/2\pi+\omega_{2}/2\pi\), which generates an electrical signal for detection.

While several nonlinear processes may occur simultaneously in this single device, the output transducer is an efficient filter of \(f_{1},f_{2}\), and the converted frequencies outside of the passband centered at \(f_{3}\). Electrodes on the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) enable modification of the electrical boundary condition to the patterned semiconductor, which we show can modify the conversion efficiency.

A plot of the measured electrical reflection, \(S_{11}\), as a function of frequency for the three interdigital transducers used for phonon generation and detection in the system is shown in Fig. 2(c) (reflection measurement details are given in the Methods). When the input electrical signal frequency matches the designed resonance frequency of the transducer, a significant amount of power is radiated as piezoelectric phonons, as evidenced by a characteristic dip in the \(S_{11}\) spectrum. As shown in Fig.

2(c), the measured resonance frequencies are \(f_{1}=370\;\mathrm{MHz}\), \(f_{2}=256.6\;\mathrm{MHz}\), and \(f_{3}=626.6\;\mathrm{MHz}\).

Fig. 2: Sum frequency generation. (a) Phonon energy level diagram for sum frequency generation. (b) Microscope image of physical system implementation in the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) - \(\mathrm{LiNbO}_{3}\) semiconductor-piezoelectric heterostructure on a silicon substrate. Two input interdigital transducers (IDTs) launch the collinear and co-propagating pump and signal phonons, which propagate through a \(250\;\upmu\mathrm{m}\) long patterned \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconductor on the \(\mathrm{LiNbO}_{3}\) surface that defines the acoustoelectric interaction region. An output IDT on the \(\mathrm{LiNbO}_{3}\) is designed to have a resonant frequency equal to the sum frequency. (c) Measured reflection as a function of frequency for the pump, signal, and sum frequency IDTs. (d) Schematic of the experimental setup to study sum frequency generation. The IDTs convert radio frequency electrical signals to

![](dt=2025-08-07/ht=23/a27ebfb3421d555f6e6b5cd094f61a4ab0627c9efc758e76f22f5454fae61551.jpg)

radio frequency piezoelectric acoustic waves. The acoustoelectric interaction region, defined by the patterned semiconductor layer on the piezoelectric film, provides the nonlinearity. (e) Measured terminal power conversion efficiency \((PCE)\) as a function of the detection frequency for different phononic pump powers. (f) Measured phononic power conversion efficiency \((PCE)\) as a function of phononic pump power at a single detection frequency for open and shorted electrical contact to the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconductor. The modeled values according to numerical integration of the nonlinear coupled-mode equations (Supplementary Note 3
) are also shown.

Figure 2(d) shows an experimental schematic for the sum frequency generation power conversion efficiency measurements and additional details are given in the Methods section. A network analyzer is used as the electrical source for the transducer resonant at the signal frequency \(f_{2}\) and as the electrical detector for the transducer resonant at the sum frequency \(f_{3}\). An amplified radio frequency source is used as the electrical input to the interdigital transducer to generate the pump phonon at \(f_{1}\).

The network analyzer gives a direct measurement of the electrical power conversion efficiency from \(f_{2}\) to \(f_{3}\), which is the end-to-end total power conversion efficiency that includes the losses incurred during the transduction processes utilized for generation and detection of phonons. We separately fabricated acoustic delay line test structures that allowed us to measure the transducer conversion losses (see Supplementary Note 1). These measured conversion losses and the electrical power conversion efficiency then allow us to quantify the phononic power conversion efficiency.

Figure 2(e) shows a plot of the electrical power conversion efficiency as a function of the detection frequency for several different phononic pump powers. In this experiment, the pump frequency was fixed at \(f_{1}=370\,\mathrm{MHz}\), the signal frequency was varied from \(f_{2}=206.6\,\mathrm{MHz}\) to \(f_{1}=306.6\,\mathrm{MHz}\), and the detection frequency was modified to satisfy \(f_{3}=f_{1}+f_{2}\).

The phononic pump power is the pump power at the beginning of the patterned \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconductor that defines the acoustoelectric interaction region. As can be seen in Fig. 2(e), the electrical power conversion efficiency achieves a maximum value at the resonance frequency of the output transducer, increases with increasing pump power, and has a 3 dB fractional bandwidth of \(1.2\%\), which is limited by the transducer bandwidth.

Figure 2(f) shows the phononic power conversion efficiency as a function of phononic pump power at a fixed detection frequency of \(f_{3}=626.6\,\mathrm{MHz}\) for the case where the electrical contacts to the semiconductor layer are left open or shorted together, which is expected to affect the power conversion efficiency as discussed in Supplementary Note 2. A maximum phononic power conversion efficiency of \((16\pm6)\%\) is realized for a phononic pump power of \(13\;\mathrm{mW}\). The modeled phononic power conversion efficiency is also shown in Fig. 2(f).

according to a model we have developed to theoretically assess the phononic frequency conversion processes. As described in detail in Supplementary Note 3, we adapt multiple models that have been previously developed for bulk piezoelectric semiconductors with plane wave propagation\(^{28-30}\) to describe the nonlinear interactions we study here in the regime where \(\beta l\ll1\) where \(\beta\) is the wave vector for any of the piezoelectric phonons in the system and \(l\) is the charge carrier mean free path.

No models exist in the literature that can capture the charge carrier relaxation and diffusion dynamics of the acoustoelectric heterostructure. To address this, we have combined plane wave\(^{28-30}\) and heterostructure\(^{31,32}\) models of the acoustoelectric interaction. In our model, we also

account for the finite extent of the waves to capture the propagation and interaction of the phononic modes in the planar piezoelectric waveguide of the acoustoelectric heterostructure.

Agreement between the experimental results and the model is achieved, up to a pump power of approximately \(10\;\mathrm{mW}\), beyond which the theoretical values exceed the experimental value. The deviation of the experimental values from the theoretical values at large pump power could be due to several effects. First, as discussed below, we observe second harmonic generation and cascaded third-harmonic generation for the pump in laser vibrometry data and therefore expect the pump power available for sum frequency generation to be reduced at high powers.

Second, also as discussed below, we observe strong third-order nonlinear effects at high power that we expect will lead to self-phase modulation and cross-phase modulation that will degrade the phase matching at high powers. Finally, it is also possible that at these large pump powers the input transducer is being thermally detuned from its low-power resonance frequency.

Supplementary Note 4 describes a control experiment carried out on a device where the semiconductor was removed but is otherwise identical, such that the source of the nonlinearity is from the \(\mathrm{LiNbO}_{3}\) alone. The peak electrical power conversion efficiency achieved is \((0.06{\pm}0.03)\%\) and the peak phononic power conversion efficiency achieved is \((0.8{\pm}0.5)\%\) at a phononic pump power of \(0.01\;\mathrm{mW}\).

Therefore, the achieved phononic power conversion efficiency with the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconductor film is 20X larger, but the power-dependence of the conversion efficiency differs between the two. In Supplementary Note 4, we also show data for the power conversion efficiency as a function of frequency for the control device at a phononic pump power of \(13\;\mathrm{mW}\). The larger pump power does not result in larger power conversion efficiency for the \(\mathrm{LiNbO}_{3}\)-only device as it does for the acoustoelectric heterostructure device.

Ideally, we could use the two devices to directly extract the acoustoelectrically induced change in the nonlinear phonon susceptibility responsible for the mixing. However, there are several factors that make a direct comparison of the nonlinearity challenging. First, the phase mismatch is expected to be power dependent with the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconductor as opposed to the \(\mathrm{LiNbO}_{3}\) control device and this effect is not yet included in our model.

Furthermore, for the materials we utilize here, the acoustoelectric loss is large, and this loss mechanism is also expected to be power-dependent. Finally, the acoustoelectric nonlinearity itself is a complex quantity, unlike in nonlinear optics where the nonlinearity is purely real. One consequence of a complex nonlinearity is nonlinear loss, which, in combination with the previously discussed nonlinearities, further complicates the power dependence of the nonlinear mixing.

With that in mind, we focus on quantification of the increase in nonlinear mixing efficiency and will investigate the changes in nonlinear susceptibility in future work. The previous state-of-the-art for three-wave mixing in \(\mathrm{LiNbO}_{3}\) is for a piezoelectric acoustic wave device carrying out an acoustic convolution, where the electrical power conversion efficiency was \(4\times10^{-5}\%\).

Therefore, here we have shown a 1500X improvement in the electrical power conversion efficiency for \(\mathrm{LiNbO}_{3}\) alone compared to the previous state-of-the-art, potentially due to optimization of \(k^{2}\) and the input pump power, and an 32500X improvement with the semiconductor film heterogeneously integrated.

As discussed in detail below, material optimization or alternate semiconductor materials that simultaneously decrease the acoustoelectric loss while increasing the acoustoelectric nonlinearity, combined with approaches to minimize phase mismatch, will ultimately provide the largest power conversion efficiency at

the smallest pump powers for the acoustoelectric heterostructure approach for phononic three-wave mixing that we present here.

We also measured, with a similar experimental procedure, difference frequency generation (see Supplementary Note 5), which is another parametric three-wave mixing process where energy conservation requires that a phonon at \(\omega_{3}\) is annihilated to create ph
onons at \(\omega_{2}\) and \(\omega_{1}=\omega_{3}-\omega_{2}\). In this case, the interdigital phonon transducers are designed such that the input pump has a frequency of \(f_{3}=366.8\,\mathrm{MHz}\), the input signal has a frequency of \(f_{2}=253.3\,\mathrm{MHz}\), and the output transducer has a frequency of \(f_{1}=f_{3}-f_{2}=113.3\,\mathrm{MHz}\). As shown in Supplementary Note 5, we achieve a maximum phononic power conversion efficiency of \((1.0\pm0.1)\%\) with an phononic pump power of \(5\;\mathrm{mW}\).

The overall frequency conversion process has a strong dependence on the phase mismatch, the interaction length, the modal overlap, and the material parameters of the acoustoelectric heterostructure. We estimate the current phase mismatch for sum frequency generation to be \(113\;\mathrm{cm}^{-1}\) and for difference frequency generation to be \(213\;\mathrm{cm}^{-1}\) (see Supplementary Note 6), which significantly limits the achievable phononic power conversion efficiency for both sum and difference frequency generation. Therefore, a systematic approach to minimize or eliminate phase mismatch while optimizing the interaction length could lead to larger demonstrated power conversion efficiencies at lower pump powers compared to what has been reported here.

# Drift velocity tuning of the power conversion efficiency

We next investigate the impact of modifying the ratio between the charge carrier drift velocity and the acoustic velocity to phononic sum frequency generation efficiency mediated by the acoustoelectric nonlinearity. Applying an external drift field to the charge carriers in an acoustoelectric heterostructure can provide large phononic small-signal gain or attenuation, depending on the size of the applied drift field and relative propagation direction of the phonons and carriers.

33-35 Supplementary Note 7 includes a plot of the theoretical linear and nonlinear acoustoelectric coefficients as a function of the ratio between the drift and acoustic velocities. At a velocity ratio exceeding 1, two effects occur simultaneously. First, the nonlinear coefficient, \(\eta\) (Supplementary Note 3), increases in absolute value.

Second, the amplitudes of the pump, signal, and output phonon fields are all amplified as they travel through the interaction region, providing both effective increases in the nonlinear interaction due to larger pump and signal waves and amplification of the output field. Therefore, it is expected that the velocity ratio can drastically tune the power conversion efficiency.

Fig. 3: Control of the nonlinearity and mixing efficiency. (a) Microscope image of the device indicating the positive voltage electrode $(+V)$, the ground electrode $(G)$, the direction of the static electric field $(E_{0})$, the direction of the charge carriers with a drift velocity, $v_{d}$, and the direction of the phonon propagation with an acoustic velocity, $v_{a}$. (b) Measured power-normalized power conversion efficiency $(PCE)$ as a function of applied bias. The pump power was fixed at $6\,\upmu\mathrm{W}$ for all applied bias values.

![](dt=2025-08-07/ht=23/f7e37ccdf373776f44aae975eceaa9de1351dc4bb303d593564e6f30b8b157e0.jpg)

Experimentally, the velocity ratio is modified by applying a static electric field to the charge carrier system as shown in Fig. 3(a). Figure 3(b) shows a plot of the measured power-normalized phononic power conversion efficiency (phononic power conversion efficiency per unit phononic pump power in units of \(\%/\mathrm{mW}\)) as a function of applied bias for two different lengths of interaction region, \(150\;\upmu\mathrm{m}\) and \(250\;\upmu\mathrm{m}\); here, the phononic pump power is fixed at \(6\,\upmu\mathrm{W}\). Therefore, as shown in Fig.

3(b), for a fixed pump power, both the conversion efficiency and the power-normalized conversion efficiency increase with increasing applied bias. We find that for a \(250\;\upmu\mathrm{m}\) long interaction length, we achieve a maximum power-normalized conversion efficiency of \((270\pm50)\%\mathrm{mW}\) at an applied bias of \(70\;\mathrm{V}\), which corresponds to a drift field of \(2.8\;\mathrm{kV/cm}\). For the previous measurement results reported with no bias and variable pump power (Fig.

2f) our maximum value for the power-normalized conversion efficiency is \((1.3\pm0.3)\%\mathrm{mW}\). This represents a \({>}200\mathrm{x}\) increase in power-normalized conversion efficiency for the sum-frequency conversion process at low pump powers. It should be noted that this ability to reconfigure the power conversion efficiency allows for interesting technological applications, such as gain switching of oscillators or dynamically optimizing conversion efficiency to extend the dynamic range of input signals with fixed radio frequency powers.

# Four-wave mixing

As in nonlinear optics, we can also examine four-wave mixing processes in the system that takes advantage of \(X_{A F}^{3}\) to see their enhancement relative to nonlinear phononic systems without acoustoelectric mediation. The possible range of interactions in four-wave mixing is large, but here we focus on cascaded degenerate four-wave mixing (CDFWM) interactions where the input and output frequencies are close together. This allows us to study the third-order nonlinear process with near-perfect phase matching and to directly compare to a \(\mathrm{LiNbO}_{3}\) device where the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconductor layer has been removed without the complication of variations in

phase mismatch between the two cases as in three-wave mixing. \(\mathrm{LiNbO}_{3}\) itself is the current state-of-the-art single crystal material for microwave frequency four-wave phonon mixing.27 We can thus directly measure and compare the efficiency of phononic four-wave mixing frequency conversion with and without the presence of the acoustoelectric nonlinearity. We follow the analysis previously developed to analyze phononic four-wave mixing in \(\mathrm{LiNbO}_{3}\) on sapphire channel waveguides,27 which also provides an additional experimental result for comparison.

While CDFWM in an acoustoelectric heterostructure was also recently reported,26 they did not identify the order of nonlinearity or four-wave mixing process used, did not describe its sensitivity to momentum or phase mismatch, and did not characterize the efficiency or internal (phononic) nonlinearity, making it challenging to compare to other systems.

During the nonlinear interaction, cascaded four-wave mixing leads to the generation of equally spaced sidebands on either side of the two pumps, with the frequency spacing between sidebands set by the frequency spacing between the pumps. The broadband phase matching supports the generation of a comb of sidebands and ultimately modifies the power conversion efficiency for the first two sidebands on either side. As shown in Fig. 4(a), here we focus on two degenerate four-wave mixing processes that occur in the system simultaneously.

In the first, two phonons at \(\omega_{1}\) are annihilated to create a phonon at \(\omega_{2}\) and \(\omega_{112}=\omega_{1}+\omega_{1}-\omega_{2}\); in the second, which occurs concurrently, two phonons at \(\omega_{2}\) are destroyed to create a phonon at \(\omega_{1}\) and \(\omega_{221}=\omega_{2}+\omega_{2}-\omega_{1}\). The phase mismatch for the first and second third-order interactions shown in Fig. 4(a) is characterized by \(\Delta\beta=\beta_{112}+\beta_{2}-\beta_{1}-\beta_{1}\) and \(\Delta\beta=\beta_{1}+\beta_{221}-\beta_{2}-\beta_{2}\), respectively.

Figure 4(b) shows a microscope image of the device developed to study phononic four-wave mixing mediated by an acoustoelectric nonlinearity, with important experimental aspects of the measurement labeled. Additional details on the experimental setup and measurement are given in the Methods. Briefly, an interdigi
tal phonon transducer with a resonance frequency of \(f=1.076\,\mathrm{GHz}\) is utilized to simultaneously generate the pump phonons at \(f_{1}=1.0755\,\mathrm{GHz}\) and \(f_{2}=1.0765\,\mathrm{GHz}\) such that they are offset from each other by \(1\,\mathrm{MHz}\).

These two phonons are then transmitted through a \(250\;\upmu\mathrm{m}\) long region of patterned \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) to support the acoustoelectric interaction. During the propagation, phonons at \(f_{112}\) and \(f_{221}\) are generated with equally spaced frequencies from the incident phonons due to the parametric four-wave mixing processes as shown in Fig. 4(a). Electrical contact is made to the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) to support an open or shorted boundary condition for the acoustoelectric current (Supplementary Note 2).

The phononic output, including both the incident and generated frequencies, is then converted to an electrical signal by the output interdigital transducer and is measured on a spectrum analyzer. An example output spectrum is shown in Fig. 4(c) with the detected signals at \(f_{1}\), \(f_{2}\), \(f_{12}\), and \(f_{221}\) labeled. As expected, phonons at additional frequencies are also generated due to the cascading nonlinear effect.

Fig. 4: Four-wave mixing. (a) Phonon energy level diagrams for generating \( f_{112} \) and \( f_{221} \) in four-wave mixing processes mediated by a third-order acoustoelectric nonlinearity. \( X_{AE}^{3} \). (b) Microscope image of the two-port device utilized to study four-wave mixing in the acoustoelectric heterostructure. Radio frequency (RF) sources, a power combiner, and an input interdigital transducer (IDT) are utilized to generate phonons at \( f_{1} \) and \( f_{2} \) with a 1 MHz offset while an output IDT and spectrum analyzer are utilized to read out the spectrum after the nonlinear mixing process. (c) Measured four-wave mixing spectrum and (d) measured phononic power conversion efficiency (PCE) as a function of phononic pump power squared.

![](dt=2025-08-07/ht=23/ea1f2408437f4709e357843b173bf86ecceeaf0d0250aae7bbf1efe2075b34db.jpg)

A plot of the phononic power conversion efficiency as a function of the square of the phononic pump power is shown in Fig. 4(d). We apply a previously developed method for phononic four-wave mixing\(^{27}\) to determine the phononic power conversion efficiency values from the measured spectra, calculate the four-wave mixing coefficients from Fig. 4(d), and calculate the modal nonlinear coefficients (further details can be found in Supplementary Note 8).

Table 1 summarizes the four-wave mixing coefficients and the modal nonlinear coefficients for the case of the \( \mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}-\mathrm{LiNbO}_{3} \) acoustoelectric heterostructure and \( \mathrm{LiNbO}_{3} \) alone. The \( \mathrm{LiNbO}_{3} \) planar waveguide measurements were made using a device on the same wafer with identical geometry but without any \( \mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As} \) present.

Also shown in Table 1 are the results for the FWM coefficient and modal nonlinear coefficient from the literature for a \( \mathrm{LiNbO}_{3} \) channel waveguide.\(^{27}\) As shown in Table 1, the four-wave mixing coefficients for the acoustoelectric heterostructure are two orders of magnitude larger than for the \( \mathrm{LiNbO}_{3} \) planar piezoelectric waveguide alone.

For example, the 221 process for \( \mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}-\mathrm{LiNbO}_{3} \) with a shorted electrical boundary condition to the \( \mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As} \) has a four-wave mixing coefficient of \( 1700 \pm 800 \, \mathrm{mW}^{-2} \), which is greater than 200X larger than the \( \mathrm{LiNbO}_{3} \) planar waveguide without the acoustoelectric nonlinearity.

The significant level of error occurs here as the model's assumption that the efficiency scales as the square of the input pump power is an approximation and does not completely capture the power-dependent dynamics in the system. The FWM coefficients extracted from the experimental data can in turn be used to approximate a modal nonlinear

coefficient, which quantifies the strength of the nonlinearity normalized to the input power and the effective length \( L_{\mathrm{eff}}=[1-\exp\left(-\alpha L\right)]/\alpha \) where \( L \) is the interaction length and \( \alpha \) is the propagation loss. For our devices, \( L=0.25\;\mathrm{mm} \) and \( L_{\mathrm{eff}}=0.21\;\mathrm{mm} \) and \( 0.24\;\mathrm{mm} \) for the structures without and without the InGaAs, respectively. The shorter effective length for the acoustoelectric heterostructure is due to the larger \( \alpha \) from the acoustoelectric attenuation.

34 As shown in Table 1, our modal nonlinear coefficients with the \( \mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As} \) are \( >10\% \) larger when compared to the \( \mathrm{LiNbO}_{3} \) waveguide devices. Therefore, the third-order nonlinearity supported in our acoustoelectric heterostructure provides a significantly larger nonlinear interaction strength for piezoelectric phonons compared to the current state-of-the-art.27 Further, it can be shown that the modal nonlinear coefficients for four-wave mixing interactions are inversely proportional to the cross-sectional area.

37 Given the \( \sim9\lambda \) width of the waves used in this demonstration, using single-mode phononic channel waveguides is expected to increase the modal nonlinearity by a factor of nearly 20.

Table 1: Measured four-wave mixing coefficients

![](dt=2025-08-07/ht=23/7a5a815de3d34475f0b835fb03c70813a3122bc863b30891aed466dcf00ba268.jpg)

<table><tr><td>Process</td><td>Device</td><td>FWM Coefficient (mW-2)</td><td>Modal Coefficient (mW-1mm-1)</td><td>Study</td></tr><tr><td>112</td><td>In0.53Ga0.47As-LiNbO3, Open</td><td>1100 ± 600</td><td>158</td><td>This work</td></tr><tr><td>221</td><td>In0.53Ga0.47As-LiNbO3, Open</td><td>1400 ± 800</td><td>178</td><td>This work</td></tr><tr><td>112</td><td>In0.53Ga0.47As-LiNbO3, Shorted</td><td>1200 ± 600</td><td>165</td><td>This work</td></tr><tr><td>221</td><td>In0.53Ga0.47As-LiNbO3, Shorted</td><td>1700 ± 800</td><td>196</td><td>This work</td></tr><tr><td>112</td><td>LiNbO3 Planar Waveguide</td><td>2 ± 1</td><td>6</td><td>This work</td></tr><tr><td>221</td><td>LiNbO3 Planar Waveguide</td><td>7 ± 5</td><td>11</td><td>This work</td></tr><tr><td>112</td><td>LiNbO3 Channel Waveguide</td><td>10 ± 5</td><td>6</td><td>Mayor et al27</td></tr><tr><td>221</td><td>LiNbO3 Channel Waveguide</td><td>20 ± 10</td><td>8</td><td>Mayor et al27</td></tr></table>

# Laser doppler vibrometry

As shown in the section on sum and difference frequency generation, both processes are supported with similar pump and signal frequencies and similar device geometries, yet with different conversion efficiencies, which is likely due to the different degrees of phase mismatch. In addition, we observe that both second and third-order nonlinear interactions are supported in our acoustoelectric heterostructures. We further expect processes such as second and third harmonic generation of the pump and cascaded processes will occur simultaneously within these structures.

While it would be possible in principle to make a series of otherwise identical structures but with different output transducers designed to collect specific mixing products, this is impractical. We instead attempt to observe the presence of all mixing components in the same device simultaneously using laser Doppler vibrometry (LDV).

The LDV measurements (Fig. 5(a)) use a scanning confocal balanced homodyne Mach-Zehnder interferometer (MZI) described in previous work.38 We use a \( 1550\;\mathrm{nm} \) laser that reflects from the \( \mathrm{In}_{0.
53}\mathrm{Ga}_{0.47}\mathrm{As} \) surface layer on the device. An anti-reflection-coated lensed-tapered fiber focuses the light in one arm of the MZI onto the surface of the chip. The lens produces a Gaussian mode approximately at its waist with mode-field diameter of \( 2\;\upmu\mathrm{m} \) on the surface. Reflections from the surface are thus directly reflected back into the lensed fiber, and a magnetic circulator directs the output back into the interferometer. Surface displacements caused by the piezoelectric phonons

present in the device cause phase fluctuations for the surface-reflected photons that can be converted to power fluctuations by the interferometer and measured using a spectrum analyzer to provide independent frequency resolution of the phonons. The Gaussian beam, with its \(2\;\upmu\mathrm{m}\) mode-field diameter, acts as a spatial filter and limits detection to frequencies of approximately 2 GHz (given the \(\sim4000\;\mathrm{m/s}\) acoustic velocities).

Fig. 5: Large bandwidth frequency conversion measurements. (a) Schematic of the laser doppler vibrometer. The laser is split into two arms utilizing a variable coupler (VC). The light in the signal arm is then sent through a polarization controller (PC), a circulator (CIRC), and is focused on the acoustoelectric heterostructure surface via a lensed tapered fiber (LTF). A \(10\%\) pickoff to a detector allows for simultaneous confocal microscopy. A PID controller actuates a fiber stretcher (FS) using feedback from the low frequency (LF) to maintain the local oscillator arm at a known phase. The high frequency (HF) output is analyzed using a real-time spectrum analyzer (RSA). (b) Interferometer signal measured using the real-time spectrum analyzer (RSA). Pump frequency \(366.8\;\mathrm{MHz}\) driven at \(22\;\mathrm{dBm}\), signal frequency \(253.5\;\mathrm{MHz}\) driven at \(0\;\mathrm{dBm}\). Peaks corresponding to difference frequency generation at \(113.3\;\mathrm{MHz}\), sum frequency

![](dt=2025-08-07/ht=23/3f0935c984e86ebd698b76fcd924673d29b5203563595368a7944d7787c4ff7a.jpg)

generation at \(620.3\,\mathrm{MHz}\), pump second harmonic at \(733.6\,\mathrm{MHz}\), and pump third harmonic at \(1,100.4\,\mathrm{MHz}\) are visible. (c)-(i) Detail of peaks for difference frequency generation, signal, pump, four-wave mixing, sum frequency generation, pump second harmonic, and pump third harmonic respectively.

The device designed for difference frequency generation (Supplementary Note 5) was utilized for the measurement. Figures 5(b-i) show the captured spectra. Peaks corresponding to the pump frequency \(f_{p}=366.8\,\mathrm{MHz}\), signal frequency \(f_{s}=253.5\,\mathrm{MHz}\), difference frequency \(f_{d}=f_{p}-f_{s}\), sum frequency \(f_{\mathrm{sum}}=f_{p}+f_{s}\), pump second harmonic \(2f_{p}\), four-wave mixing \(f_{\mathrm{fms}}=f_{p}+f_{p}-f_{s}\), and pump third harmonic \(3f_{p}\) are observed.

It is important to note that many cascaded processes, such as second harmonic generation of the pump and subsequent sum-frequency generation are not distinguishable from these measurements. In future work using LDV, we will scan the position of the probe to examine both the frequency content of the mixing processes and their spatial dependence, which will allow detailed studies of their complex interactions as they propagate. This will also allow nonlinear numerical models to examine the complex mixing that occurs when so many processes can occur without excessive phase mismatch.

# Optimization of three-wave mixing nonlinear interactions and frequency conversion

While the ability to apply bias fields to increase the effective nonlinearity is attractive, it also intrinsically amplifies the waves as they travel, which allows the excess acoustoelectric noise associated with amplification to couple into the output field, preventing quantum state preserving processes.39 Therefore, we are highly interested in the purely parametric regime where no bias fields are applied and what the limits of frequency conversion are in that regime. In the unbiased regime, we must consider the combination of acoustoelectric loss in combination with the nonlinearity.

Figure 6(a) shows a contour plot of our model's (Supplementary Note 3) predicted peak phononic power conversion efficiency for sum frequency generation as a function of the piezoelectric electromechanical coupling coefficient, \(k^{2}\), and the phononic intensity for perfect phase matching.

The phononic intensity was modified by changing the width of the interaction for a fixed input pump power of \(1\,\mathrm{mW}\); just as in nonlinear optics, a higher degree of transverse confinement of the waves leads to higher intensity and larger effective nonlinearities as this increases the displacement and electric fields field per unit acoustic power.

We see that maximizing both \(k^{2}\) and the phononic intensity leads to a larger peak phononic power conversion efficiency (Fig. 6(a)) and that maximizing \(k^{2}\) reduces the required interaction length to achieve the peak (Fig. 6(b)). At a \(k^{2}\) of \(18\%\) and a phononic intensity of \(2.5 \times 10^{6}\,\mathrm{W/m}^{2}\) \((2.5\,\upmu\mathrm{W}/\upmu\mathrm{m}^{2})\), a peak phononic power conversion efficiency of \(100\%\) from the signal frequency to the sum frequency is theoretically achieved in an interaction length of \(280\,\upmu\mathrm{m}\).

The increase in phononic intensity can be achieved in our system through thinning and etching of the \(\mathrm{LiNbO}_{3}\) piezoelectric layer to form a channel waveguide, which could also enable phase matching through the engineering of modal dispersion in the system. A finite element method model of the z-displacement \((u \cdot \hat{z})\) for this type of structure is shown in Fig. 6(c) with a plot of the phononic intensity as a function of the waveguide width. The waveguide mode supported is a quasi-shear horizontal mode with a modeled \(k^{2}\) value of \(18.5\%\).

Fig. 6: Towards stronger nonlinear interactions and larger power conversion efficiency. (a) Peak theoretical phononic power conversion efficiency \((PCE)\) as a function of \(k^{2}\) and the phononic intensity. The mobility is fixed at \(4000\;\mathrm{cm}^{2}/\mathrm{V}-\mathrm{s}\) and the charge carrier concentration is fixed at \(1\mathrm{x}10^{16}\;\mathrm{cm}^{-3}\). (b) Interaction length to achieve the peak \(PCE\) as a function of \(k^{2}\) and the intensity. (c) Phononic intensity as a function of waveguide width for a fixed pump power of \(1\;\mathrm{mW}\). Also shown is a finite element method model of the \(z\)-displacement, \(u\cdot\hat{z}\), for a \(2.5\;\upmu\mathrm{m}\) thick \(\mathrm{LiNbO}_{3}\) channel waveguide directly on silicon supporting a quasi-shear horizontal mode with a modeled \(k^{2}\) value of \(18.5\%\). (d) Theoretical peak power conversion efficiency and corresponding interaction length as a function of (d) sheet carrier density and (e) charge carrier mobility. The \(k^{2}\) value was \(10\%\) and the phononic intensity was \(5\mathrm{x}10^{5}\;\mathrm{W}/\mathrm{m}^{2}\). When not being varied, the sheet carrier density was fixed at \(5\mathrm{x}10^{10}\;\mathrm{cm}^{-2}\) and the mobility was fixed at \(1000\;\mathrm{cm}^{2}/\mathrm{V}-\mathrm{s}\). (f) Semiconductor material selection chart showing material as a function of sheet carrier density and charge carrier mobility. Values for the sheet carrier densities and charge carrier mobilities are taken from relevant references\(^{40-42}\) where the thickness of the semiconductor is assumed to be \(50\;\mathrm{nm}\) for the non-2D materials.

![](dt=2025-08-07/ht=23/073182e19d87621317de4c2120513829945aaf1f53ff9907546b5f4e5dc15840.jpg)

Figures 6(d) and 6(e) show plots of the theoretical peak power conversion efficiency and the c
orresponding interaction length as a function of semiconductor sheet carrier density and charge carrier mobility, respectively. As can be seen, modification to the sheet carrier density can improve the phononic power conversion efficiency by orders of magnitude, and there is an optimal value for the sheet carrier density to reduce the interaction length.

Figure 6(d) also suggests that the use of gate electrodes to locally modify the sheet carrier density could allow for dynamic optimization or switching of the nonlinearity. Increasing the charge carrier mobility (Fig. 6(e)) leads to a larger theoretical power conversion efficiency, but requires a longer interaction length to achieve this peak value. As our acoustoelectric heterostructure relies on bonding between the piezoelectric and semiconductor charge carrier systems, it can potentially

be adapted to semiconductor materials with more favorable properties for the nonlinear frequency conversion process. A material selection chart of potential options characterizing different material classes is shown in Fig. 6(f) and includes silicon, an example of a compound semiconductor \(\left(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\right)\), an example of a 2D material (graphene),\(^{40,41}\) and an example of a heterostructure supporting a 2D electron gas (gallium arsenide/aluminum gallium arsenide).

\(^{42}\) Through material selection in future work, we can potentially find the combination of sheet carrier density and charge carrier mobility to achieve nonlinear interactions in an acoustoelectric heterostructure with exceptionally large interaction strengths even beyond what is demonstrated here.

# Outlook

The ability to generate efficient phononic frequency conversion presents an enormous open canvas for fundamental research and technological applications. As an example of materials science research that could be explored, phonons significantly contribute to heat transport properties in solids, and the ability to modify phononic nonlinearities by modifying the semiconductor properties could allow detailed studies of the specific nonlinearities that lead to processes such as thermal relaxation.

This could also allow the creation of synthetic materials that have thermal properties not found in nature, such as materials with completely depleted phonon frequency regions or phonon populations that can only flow in one direction, as these momentum-preserving frequency conversion processes are highly nonreciprocal, such has been demonstrated in the photonic domain through nonlinear optical processes.\(^{43}\)

The technological implications are profound. For example, three-wave mixing up-conversion and down-conversion processes are the backbone of radio-frequency signal processing, as they allow frequency conversion between radio bands and the encoding and decoding of data on carrier waves. Given recent demonstrations of acoustic amplification in the same system, one can now quite readily imagine entire radio-frequency signal processors occurring on the surface of a lithium niobate microchip or other piezoelectric substrate, greatly reducing the number of chips necessary for wireless technologies.

Strong cascaded nonlinearities could be used to down-convert very high frequency radio waves where acoustic and microwave filter bandwidths are very large to lower frequencies where the filter are extremely narrow-band, and then the reverse process could allow up-conversion back to the original carrier frequency.

In the quantum realm, three-wave mixing processes such as sum-frequency generation are known to be quantum state preserving,\(^{39}\) and this could provide a way to connect different microwave frequency qubits together on chips or up-convert phonons carrying quantum information to frequencies where they are more easily detectable or transducible to photons.

\(^{44,45}\) Another important three-wave mixing process is degenerate parametric amplification, which could allow the creation of squeezed phononic vacuum states or quantum-limited amplification of phononic quantum information, much like a Josephson parametric amplifier does for microwave photons. Finally, these giant four-wave mixing nonlinearities can be used to create self-Kerr modulation in acoustic cavities that may enable the creation of new kinds of qubits, which would then have access to the parametric amplification and state-preserving frequency conversion already discussed.

In conclusion, we have demonstrated efficient phononic three-wave and four-wave mixing nonlinearities in a thin lithium niobate film by heterogeneously integrating a high-mobility semiconductor that allows the phononic nonlinearity to be mediated by electronic nonlinearities in the semiconductor.

We demonstrated three-wave mixing processes at microwave frequencies, including a \((16\pm6)\%\) phononic power conversion efficiency for sum-frequency generation and \((1.0\pm0.1)\%\) phononic power conversion efficiency for difference-frequency generation, as well as the most efficient degenerate four-wave phononic mixing to date. We presented a theoretical model that accurately predicts the sum-frequency and difference-frequency generation processes and showed experimentally that the frequency conversion can be further enhanced by the application of bias fields.

Laser Doppler vibrometry was utilized to examine many three-wave and four-wave mixing processes simultaneously in the same device. Finally, we used the theoretical model to show that these nonlinearities can be greatly enhanced by confining phonons to smaller dimensions in waveguide circuits and optimizing semiconductor material properties or using 2D semiconductor materials.

# Methods

# Acoustoelectric heterostructure and device fabrication

The heterostructure is comprised of a 4 inch lithium niobate \(\left(\mathrm{LiNbO}_{3}\right)\) on silicon wafer, with a \(5\ \upmu\mathrm{m}\ \mathrm{LiNbO}_{3}\) film thickness, and a 2 inch indium phosphide (InP) wafer. The InP wafer has a lattice-matched epitaxial stack of multiple layers of \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) and InP.

The epitaxial semiconductor stack is grown by metal-organic chemical vapor deposition (MOCVD) and consists of the following layers: \(500\ \mathrm{nm\ InP}\) non-intentionally doped (NID) buffer, \(3000\ \mathrm{nm\ NID}\ \mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) etch stop, \(100\ \mathrm{nm\ InP}\) etch stop doped with silicon at \(1 \times 10^{18}\ \mathrm{cm}^{-3}\), \(100\ \mathrm{nm\ In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) contact layer doped with silicon at \(2 \times 10^{18}\ \mathrm{cm}^{-3}\), \(30\ \mathrm{nm}\) thick InP etch stop doped with silicon at \(1 \times

10^{18}\ \mathrm{cm}^{-3}\), a \(50\ \mathrm{nm}\) thick \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) device layer, and a \(5\ \mathrm{nm}\) NID InP capping layer.

The \(\mathrm{LiNbO}_{3}\)-silicon and \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}/\mathrm{InP}\) wafers are then bonded together, with the InP capping layer in contact with the \(\mathrm{LiNbO}_{3}\) surface. The bond is manually initiated followed by annealing in vacuum at \(100\,^{\circ}\mathrm{C}\). After wafer bonding, the InP substrate and buffer layer are removed in hydrochloric acid followed by removal of the subsequent \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) etch stop layer in a 1:1:10 solution of sulfuric acid, hydrogen peroxide, and/water.

The InP etch stop layer is then removed in a 1:3 mixture of hydrochloric acid, phosphoric acid, stopping on the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) contact layer. This highly doped \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) layer provides an intermediary \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) layer from the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) device layer to the metal electrodes to form quasi-Ohmic electrical contact to the \(\mathrm{In}_{0.5
3}\mathrm{Ga}_{0.47}\mathrm{As}\) device layer.

This \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) is patterned and then etched in a 1:1:10 mixture of sulfuric acid, hydrogen peroxide, and deionized water followed by immediately etching the subsequent InP etch stop layer in a 1:3 mixture of hydrochloric acid/phosphoric acid. The \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) device layer is then patterned and etched in a 1:1:10 mixture of sulfuric acid, hydrogen peroxide, and water, landing on the \(\mathrm{LiNbO}_{3}\) surface.

A metal liftoff step, with a metal stack of \(10\ \mathrm{nm}\ \mathrm{chrome}/100\ \mathrm{nm}\) aluminum is then carried out to form the interdigital phonon transducers on the \(\mathrm{LiNbO}_{3}\). Electrical contact is made to the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) through a metal liftoff, with a metal stack of \(10\ \mathrm{nm}\ \mathrm{Ti}\), \(500\ \mathrm{nm}\ \mathrm{Au}\), \(500\ \mathrm{nm}\ \mathrm{Ag}\), and \(100\ \mathrm{nm}\ \mathrm{Au}\), over the patterned \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) contact layer.

The semiconductor properties of the \(50\ \mathrm{nm}\) thick \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) device layer were determined by a Hall mobility measurement using a Bio-Rad fixed-magnetic-field Hall effect measurement system. The Hall structures were patterned on the same wafer and undergo the same fabrication process flow. The average values for the Hall coefficient and Hall mobility were \(-(3.5 \pm 0.1) \times 10^{3}\ \mathrm{m}^{2}\mathrm{C}^{-1}\) and \(4,220 \pm 40\ \mathrm{cm}^{2}\ \mathrm{V}^{-1}\mathrm{s}^{-1}\),

respectively, as taken from measurements on two separate structures. Combined with the measured sheet resistance from the Hall structures, this gives a doping concentration of approximately \(3 \times 10^{16} \, \mathrm{cm}^{-3}\) for the \(\mathrm{In}_{0.53} \mathrm{Ga}_{0.47} \mathrm{As}\) device layer. The two Hall structure measurements were taken in similar locations on the wafer while there is an expected cross-wafer variation of \(\sim 12\%\) in sheet resistance across the wafer primarily due to variations in carrier concentration due to non-uniformity in disilane delivery during the epitaxial growth.

# Reflection measurements

Interdigital transducer reflection measurements were carried out on a network analyzer on a custom radio frequency probe station with ground-signal-ground radio frequency probes\. A full one-port calibration (short, load, open) was carried out with an impedance standard substate before all measurements to calibrate out the response of the cables and probes. For all measurements, the radio frequency power on the network analyzer was set to \(-10 \, \mathrm{dBm}\) and the reflection coefficient \((\mathbf{S}_{11})\) was measured as a function of frequency to characterize each interdigital transducer.

# Sum and difference frequency generation power conversion efficiency measurements

Sum and difference frequency power conversion efficiency measurements were made on a custom radio frequency-DC probe station with three ground-signal-ground radio frequency probes and two DC probes. To generate the pump phonons, a radio frequency signal generator is set to output at a fixed frequency, equal to the resonant frequency of the interdigital transducer for the pump. This output then goes through a radio frequency amplifier before being sent through the interdigital transducer to generate the pump phonons.

A network analyzer is utilized to source a frequency resonant with the signal interdigital transducer while detecting at the sum or difference frequency. The transmission measurement \((\mathbf{S}_{21})\) on the network analyzer then directly measures the electrical power conversion efficiency from the signal frequency to the sum or difference frequency. The \(10 \, \mathrm{MHz}\) internal reference of the radio frequency signal generator was connected to the network analyzer to phase-lock the network analyzer to the signal generator.

A scalar-mixer calibration was carried out before all measurements to calibrate out the response of the cables and probes for the measured \(\mathbf{S}_{21}\) values. For each data point, 1601 consecutive measurements (corresponding to the maximum number of points that can be taken in a single trace on the network analyzer) were taken. The phononic power conversion efficiency values then are based on the average of these 1601 points along with the measured transducer conversion losses.

The error in the reported phononic power conversion efficiencies comes from two sources: one is the standard deviation of the 1601 measurements and the second is the inferred error in the interdigital transducer conversion losses.

# Four wave mixing measurements

To carry out the four wave mixing measurements, we utilized two radio frequency signal generators and a spectrum analyzer. The outputs of the signal generators go to a power combiner and then to the single interdigital input transducer of the device to generate the pump phonons. The output from the device goes to the spectrum analyzer, To determine the phononic input pump power, a radio frequency power meter was used to measure the loss in the cable and acoustic delay line test structures were utilized to determine the losses incurred during the electromechanical conversion process.

# Acknowledgements

Supported by the Laboratory Directed Research and Development program at Sandia National Laboratories, a multimission laboratory managed and operated by National Technology and

Engineering Solutions of Sandia LLC, a wholly owned subsidiary of Honeywell International Inc. for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA0003525. This work was performed, in part, at the Center for Integrated Nanotechnologies, an Office of Science User Facility operated for the U.S. Department of Energy (DOE) Office of Science. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government.

# Data availability

The data that support the findings of this study are available from the corresponding author upon reasonable request.

# Conflict of interest

The authors declare that they have no conflict of interest.

# Author contributions

L.H. and M.E. came up with the device concepts and experimental implementations. M.E. developed the laser doppler vibrometry system. M.K. performed measurements with the laser doppler vibrometer with input from L.H., N.O., and M.E.. L.H., B.S., M.M., S.W., S.A., T.A.F. and M.E. designed the devices and fabrication process flow. M.M., S.W., B.S., and S.A. fabricated the devices. L.H., M.K., B.S., and S.S. performed the measurements. L.H. and M.E. carried out all modeling. L.H., M.K., and M.E. analyzed all data with input from N.O. The manuscript was written by L.H., M.K., and M.E. and all authors have given approval of the final version.

# Supplementary Note 1

# Acoustic delay line test structures

To determine the internal pump power and internal power conversion efficiency, we utilized acoustic delay line test structures in the lithium niobate \(\left(\mathrm{LiNbO}_{3}\right)\)-silicon.

Eight different types of acoustic delay line test structures were utilized: acoustic wavelength of \(6.4\;\upmu\mathrm{m}\) and aperture of \(100\;\upmu\mathrm{m}\), acoustic wavelength of \(11\;\upmu\mathrm{m}\) and aperture of \(100\;\upmu\mathrm{m}\), acoustic wavelength of \(16\;\upmu\mathrm{m}\) and aperture of \(100\;\upmu\mathrm{m}\), acoustic wavelength of \(11\;\upmu\mathrm{m}\) and aperture of \(240\;\upmu\mathrm{m}\), acoustic wavelength of \(16\;\upmu\mathrm{m}\) and aperture of \(240\;\upmu\mathrm{m}\), acoustic wavelength of \(39\;\upmu\mathrm{m}\) and aperture of

\(240\;\upmu\mathrm{m}\), acous
tic wavelength of \(11\;\upmu\mathrm{m}\) and aperture of \(100\;\upmu\mathrm{m}\) with a signal interdigital transducer with a wavelength of \(16\;\upmu\mathrm{m}\) and aperture of \(100\;\upmu\mathrm{m}\) placed in the acoustic path, and acoustic wavelength of \(11\;\upmu\mathrm{m}\) and aperture of \(240\;\upmu\mathrm{m}\) with a signal interdigital transducer with a wavelength of \(16\;\upmu\mathrm{m}\) and aperture of \(240\;\upmu\mathrm{m}\) placed in the acoustic path.

For each of the eight different types of acoustic delay line test structures, acoustic delay lines of varying separation lengths between the interdigital transducers utilized to generate and detect acoustic waves were fabricated and characterized. From the loss as a function of length, we can extract the propagation loss. The remaining losses are assumed to be from the transducers. Therefore, by splitting the remaining losses in half we can determine the loss for a single interdigital transducer.

For the pump (\(11\;\upmu\mathrm{m}\) acoustic wavelength) test structures with the signal transducer in the acoustic path, we measured the increase in loss due to the signal transducer. Supplementary Fig. 1 shows data for a single type of acoustic delay line test structure and Supplementary Table 1 summarizes the losses for all test structures.

Supplementary Fig. 1: Acoustic delay line test structures. The measured transmission as a function of separation length between two interdigital transducers utilized to generate and detect acoustic waves is shown. For this test structure, the transducer is designed to be resonant at approximately \(255\,\mathrm{MHz}\), corresponding to an acoustic wavelength of \(16\,\upmu\mathrm{m}\), and the aperture of the interdigital transducer is \(100\,\upmu\mathrm{m}\).

![](dt=2025-08-07/ht=23/6657733dace3b176bc07672f72546569b2353f05b3458685e8b2277854d33d77.jpg)

Supplementary Table 1: Loss per interdigital transducer

![](dt=2025-08-07/ht=23/3d9a3522f18e4bfad3bab83654b0922d0f4c7b49deacabbb30382155297b90b7.jpg)

<table><tr><td>Wavelength (µm)</td><td>Aperture (µm)</td><td>Loss (dB)</td><td>Note</td></tr><tr><td>11</td><td>100</td><td>7 ± 1</td><td></td></tr><tr><td>16</td><td>100</td><td>6 ± 3</td><td></td></tr><tr><td>6.4</td><td>100</td><td>5 ± 1</td><td></td></tr><tr><td>11</td><td>240</td><td>6 ± 2</td><td></td></tr><tr><td>16</td><td>240</td><td>6 ± 1</td><td></td></tr><tr><td>39</td><td>240</td><td>7 ± 1</td><td></td></tr><tr><td>11</td><td>100</td><td>12 ± 2</td><td>With signal transducer in path</td></tr><tr><td>11</td><td>240</td><td>11 ± 3</td><td>With signal transducer in path</td></tr></table>

# Supplementary Note 2

# Electrical boundary condition to semiconductor

In the experiments in this work, the semiconductor boundary conditions could be modified by the application of electrical boundary conditions to the electrodes at the ends of the semiconductor. Though rigorous analysis must be done in the future to thoroughly understand the impact these electrical boundary conditions make, we at least attempt here to explain why they are expected to impact the performance of the mixer devices.

In the case that the two contacts are shorted together, the electrical current generated by the acoustoelectric interactions at one end of the device must be equal to the current at the other end of the device, especially considering the very large resistance of the semiconductor itself. In the case that the contacts are allowed to float or be an open circuit, then the currents at both ends of the device must go to zero.

While the AC currents generated by the acoustic waves and their nonlinear interactions can do this for fortuitous values of the electronic current’s phase, the DC currents produced by the acoustic waves<sup>31</sup> cannot. We thus presume that the DC electric currents must satisfy these conditions by setting up counter-propagating DC currents, and these currents, appearing to be counterpropagating drift currents from the perspective of the waves that generate them, would be expected to attenuate the acoustic wave.

Thus, we generically expect to have higher mixing efficiency for the case of shorted electrodes. While this is certainly true for our best values of

sum-frequency generation, as can be seen in Fig. 2(f) in the main text, it can also be seen that the general relationship between the mixing efficiency and the electrical boundary condition is not that simple. In this work, we measured efficiencies with both boundary conditions, but solving the nonlinear coupled mode equations in these systems in the presence of realistic boundary conditions is something that should be done in future work to understand the impact on the mixing efficiencies.

# Supplementary Note 3

# Model development: piezoelectric acoustic wave nonlinear mixing when coupled to a semiconductor charge carrier system

Here we develop a theoretical model for nonlinear phononic three-wave mixing interactions induced by an acoustoelectric nonlinearity in a heterostructure comprised of an acoustically thin semiconductor film and a piezoelectric acoustic wave film. Previous models have been developed of acoustoelectrically mediated phonon interactions in an infinite "isotropic" piezoelectric semiconductor,\[28-30\] but these models differ in many important ways that are necessary to explain our experimental results.

First, the carrier dynamics inside a thin, nonpiezoelectric semiconductor are considerably different from those in a bulk piezoelectric semiconductor, including significant differences between carrier relaxation dynamics due to drift, dissipation, and diffusion. Second, the response to the acoustic wave depends on the spatial dependence of the electric field, whereas in any previously developed model\[28-30\] the plane-wave nature of the acoustic, electric, and electronic waves are independent of any spatial variations in those fields.

Moreover, because of the separation of the two media, it is in fact the evanescent electric field of the piezoelectric phonon and the spatial dependence of that field that results in the coupling. Third, existing models cannot readily take into account the enhancement of the nonlinearities due to confinement of the acoustic wave, which increases the electric field for a given phonon density (number of phonons/length) in the waveguide.

The structure of our planar phononic waveguide consists of a \(\mathrm{LiNbO}_{3}\) guiding material and a silicon substrate where the acoustic velocity in the silicon exceeds the acoustic velocity for the acoustic modes supported in the \(\mathrm{LiNbO}_{3}\). The eigenmodes of the waveguide can be found by solving the coupled acoustic and electromagnetic wave equations for the system and a complex reciprocity relation, which allows modal decomposition of arbitrary fields in terms of powerorthogonal waveguide modes.

\[46\] These waveguide modes have normalized transverse field patterns that are assumed to be constant along the propagation direction \((x)\) with characteristics that depend on the distribution of acoustic and electromechanical material properties along the waveguide cross-sectional profile \((y)\).

In the case of a piezoelectric semiconductor, the coupled-wave equations for the displacement amplitude, \(\hat{u}_{m}\), under the assumption that the change in displacement is small with respect to the acoustic wavelength, are given by

\[
\begin{array}{r}\displaystyle\frac{d\hat{u}_1}{dx}=-\hat{\alpha}_1\hat{u}_1+\hat{\eta}_1\hat{u}_2^*\hat{u}_3e^{i\Delta\beta x}\\\displaystyle\frac{d\hat{u}_2}{dx}=-\hat{\alpha}_2\hat{u}_2+\hat{\eta}_2\hat{u}_1^*\hat{u}_3e^{i\Delta\beta x}\\\displaystyle\frac{d\hat{u}_3}{dx}=-\hat{\alpha}_3\hat{u}_3+\hat{\eta}_3\hat{u}_1\hat{u}_2e^{i\Delta\beta x}\end{array}
\]

where \(\Delta\beta=\beta_{1}
+\beta_{2}-\beta_{3}\) is the wave vector mismatch, \(\beta_{m}\) are the wave vectors, \(\hat{\eta}_{m}\) are the nonlinear coefficients given by

\[
\begin{array}{r}\hat{\eta}_1=i\frac{e^3\mu}{4c\varepsilon^2v_a}\frac{\omega_c}{\omega_1}\frac{2\gamma+i\omega_1/\omega_D}{\Gamma_1\Gamma_2^*\Gamma_3}\beta_2\beta_3\\\hat{\eta}_2=i\frac{e^3\mu}{4c\varepsilon^2v_a}\frac{\omega_c}{\omega_2}\frac{2\gamma+i\omega_2/\omega_D}{\Gamma_1^*\Gamma_2\Gamma_3}\beta_1\beta_3\\\hat{\eta}_3=i\frac{e^3\mu}{4c\varepsilon^2v_a}\frac{\omega_c}{\omega_3}\frac{2\gamma+i\omega_3/\omega_D}{\Gamma_1\Gamma_2\Gamma_3}\beta_1\beta_2\end{array}
\]

and \(\hat{\alpha}_{m}\) are the acoustoelectric linear attenuation/gain coefficients given by

\[
\hat{\alpha}_{m}=\frac{e^{2}}{2c\varepsilon}\frac{\omega_{c}}{v_{a}\gamma}\left[1+\frac{\omega_{c}^{2}}{\gamma^{2}\omega_{m}^{2}}\left(1+\frac{\omega_{m}^{2}}{\omega_{c}\omega_{D}}\right)^{2}\right]^{-1}
\]

where \(e\) is the piezoelectric coupling constant, \(\mu\) is the charge carrier mobility, \(c\) is the elastic constant, \(\varepsilon\) is the material permittivity, \(v_{a}\) is the acoustic velocity, \(\omega_{m}\) is the acoustic radial frequency, \(\omega_{c}=\sigma/\varepsilon\) is the dielectric relaxation frequency, \(\sigma\) is the conductivity, \(\omega_{D}=v_{a}^{2}/D_{n}\) is the diffusion frequency, \(D_{n}=\mu k_{B}T/q\) is the diffusion coefficient, \(k_{B}\) is Boltzmann's constant, \(T\) is the temperature, \(q\) is the elementary charge, \(\gamma\) is a term that denotes

the ratio between the acoustic velocity, \(v_{a}\), and the charge carrier drift velocity, \(v_{d}\), according to \(\gamma=1-v_{d}/v_{a}\), and \(\Gamma_{m}=\gamma+i(\omega_{c}/\omega_{m}+\omega_{m}/\omega_{D})\).

The term \(e^{2}/c\varepsilon\) is approximately equal to the volume-wave electromechanical coupling coefficient, \(K^{2}\), which quantifies the piezoelectric strength of the material.

We first need to adapt the expressions for the nonlinear (Eqn. S2) and linear (Eqn. S3) acoustoelectric coefficients from a to the acoustoelectric heterostructure studied in this work. To do this, we begin with Eqn. S3 as there exists an expression for the linear attenuation/gain coefficient in an acoustoelectric heterostructure, \(\alpha_{m}\), that has been derived via perturbation theory. This allows us to compare our modifications to the expression for the linear acoustoelectric coefficient (Eqn. S3) to the result derived by perturbation theory for an acoustoelectric heterostructure.

If suitable agreement between the two approaches is achieved, then we can also apply our modifications to the nonlinear acoustoelectric coefficients (Eqn. S2) to parameterize the nonlinear interaction in the heterostructure.

For the case of an acoustoelectric heterostructure, the expression for the linear acoustoelectric attenuation/gain coefficient, as determined by a perturbative approach, is given by

\[
\alpha_{m}=\frac{1}{2}\frac{(v_{d}/v_{a}-1)\omega_{c}\varepsilon_{s}wZ_{a,m}(h)\beta\tanh(\beta d)}{(v_{d}/v_{a}-1)^{2}-(R\omega_{c}/\omega_{m}+H)^{2}}
\]

where \(Z_{a,m}(h)=2|\Delta\nu(0)|e^{-2\beta h}/w v_{a}\omega_{m}(\varepsilon_{0}+\varepsilon_{p})\) is the interaction impedance, \(\Delta\nu(0)\) is the perturbation in velocity of the piezoelectric acoustic wave when the surface electrical boundary condition of the piezoelectric is changed from electrically open to shorted, \(h=g e_{G}\) where \(g\) and \(e_{G}\) are the thickness and relative permittivity, respectively, for the gap between the piezoelectric and semiconducting materials in the heterostructure, \(\boldsymbol{w}\) is the width of the interaction region,

\(\omega_{c}=\sigma/\varepsilon_{s}\) is the dielectric relaxation frequency, \(\varepsilon_{0}\) is the vacuum permittivity, \(\varepsilon_{p}\) is the piezoelectric material permittivity, \(\varepsilon_{s}\) is the semiconductor material permittivity, \(d\) is the semiconductor thickness, \(R=(\varepsilon_{s}/\varepsilon_{0})M(\beta h)\mathrm{tanh}\left(\beta d\right)\) is the space-charge reduction factor,

which accounts for the finite thickness of the semiconductor, \( M(\beta h) = \frac{\varepsilon_{0} + \varepsilon_{p}\tanh(\beta h)}{(\varepsilon_{0} + \varepsilon_{p})(1 + \tan(\beta h))} \), and \( H = \sqrt{\frac{\omega_{c}}{\omega_{D}}\frac{\tanh(\beta d)}{\tanh((\sqrt{\omega_{c}\omega_{D}}/v_{a})d)}} \) is a term that arises from semiconductor charge carrier diffusion.\(^{32}\)

The expression for (S3), adapted for our acoustoelectric heterostructure, is given by

\[
\alpha_{m}=\frac{k_{m}^{2}}{2}\frac{\omega_{c,m}}{v_{a,m}\gamma_{m}}\left[1+\frac{\omega_{c,m}^{2}}{\gamma_{m}^{2}\omega_{m}^{2}}\left(1+\frac{\omega_{m}^{2}}{\omega_{c,m}\omega_{D}}\right)^{2}\right]^{-1}
\]

where \( k_{m}^{2} \) is the modal electromechanical coupling coefficient, which has a standard definition of \( k_{m}^{2} = 2|\Delta v_{a,m}(0)|/\nu_{a,m} \), which for an anisotropic material like \( \mathrm{LiNbO}_{3} \) depends on the material cut and the propagation direction of the piezoelectric phonon, and \( \omega_{c,m} = \sigma\beta_{m}d/(\varepsilon_{p} + \varepsilon_{0}) \).

\(^{31}\) The variation in all parameters with the acoustic frequency is due to dispersion from the \( \mathrm{LiNbO}_{3} \) and indium gallium arsenide \( (\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}) \) films. The value for \( \alpha \) as a function of the charge carrier mobility, \( \mu \), and the charge carrier concentration, \( N_{d} \), is plotted in Supplementary Fig. 2 for the expressions given in Eqn. S4 and Eqn. S5 with and without the inclusion of charge carrier diffusion.

While we do find a difference in value of \( N_{d} \) that maximizes \( \alpha \) between the two theoretical approaches, our model correctly captures the trends in how \( \alpha \) varies with the semiconductor parameters and gives values for \( \alpha \) that agree well with the perturbative approach. We find that \( \alpha \) is expected to reach a minimum either when \( N_{d} \) is very low \( (<10^{14}) \) or when \( N_{d} \) is relatively large \( (>10^{16}) \).

For the combination of \( \mu \) and \( N_{d} \) utilized in this work, which are \( \mu = 4220\;\mathrm{cm}^{2}/\mathrm{V-s} \) and \( N_{d} = 3 \times 10^{16}\;\mathrm{cm}^{-3} \) (see Methods in main article), the value is \( 5\;\mathrm{cm}^{-1} \) from the model developed here and \( 3\;\mathrm{cm}^{-1} \) from the perturbative approach.\(^{32}\) Therefore, we find that sufficient agreement is achieved.

Supplementary Fig. 2: Theoretical linear acoustoelectric attenuation coefficient, $\alpha$, as a function of semiconductor mobility, $\mu$, and charge carrier concentration, $N_{d}$, with an expression derived from a perturbative approach (Eqn. S4)$^{32}$ (a) with and (b) without the effects of charge carrier diffusion. Plots of $\alpha$ as a function of $\mu$ and $N_{d}$ from our phenomenological model (Eqn. S5) (c) with and (d) without charge carrier diffusivity effects. The star marker in (c) indicates the expected value of $\alpha$ in this work given the measured values of $\mu$ and $N_{d}$ for the semiconductor (see Methods in main article).

![](dt=2025-08-07/ht=23/c0b73d65b7a3fc25bf394dc23fd5125374119b3cc657161e40a072abd9d4463e.jpg)

We then find it suitable to apply the modifications for \(k_{m}^{2}\), \(\omega_{c,m}\), and dispersion to the nonlinear acoustoelectric coefficients (Eqn. S2) to obtain the following phenomenological expressions for the nonlinear acoustoelectric coefficients in the acoustoelectric heterostructure, \(\eta_{m}\),

\[
\begin{array}{l}\displaystyle\eta_1=i\frac{(k_1^2)^{3/2}\mu}{4v_{a,1}}\sqrt{\frac{c_1}{\varepsilon}}\frac{\omega_{c,1}}{\omega_1}\frac{2\gamma_1+i\omega_1/\omega_D}{\Gamma_1\Gamma_2^*\Gamma_3}\beta_2\beta_3\\\displaystyle\eta_2=i\frac{(k_2^2)^{3/2}\mu}{4v_{a,2}}\sqrt{\frac{c_2}{\varepsilon}}\frac{\omeg
a_{c,2}}{\omega_2}\frac{2\gamma_2+i\omega_2/\omega_D}{\Gamma_1^*\Gamma_2\Gamma_3}\beta_1\beta_3\\\displaystyle\eta_3=i\frac{(k_3^2)^{3/2}\mu}{4v_{a,3}}\sqrt{\frac{c_3}{\varepsilon}}\frac{\omega_{c,3}}{\omega_3}\frac{2\gamma_3+i\omega_3/\omega_D}{\Gamma_1\Gamma_2\Gamma_3}\beta_1\beta_2.\end{array}
\]

A plot of \(|\eta_{3}u_{1}|\) as a function of the charge carrier mobility, \(\mu\), and charge carrier concentration, \(N_{d}\), with and without charge carrier diffusion is shown in Supplementary Figs. 3(a) and 3(b),

respectively. Here \( u_{1} \) is the pump displacement, which we determine according to \( |u_{1}| = \)

\(\sqrt{P_{1}/(\omega_{1}^{2}\,\rho v_{a,1}A_{1})}\) where \(P_{1}\) is the pump power, \(\rho\) is the density, and \(A_{1}\) is the cross-sectional area for the acoustic power flow, which is determined by the device width and a finite element method model of the acoustic mode in the waveguide cross-section. For the plots in Supplementary Fig. 3, we assume a \(P_{1}\) of \(1~\mathrm{mW}\) and a device width of \(100\;\upmu\mathrm{m}\) .

We find that the value of \(|\eta_{3}u_{1}|\) is maximized at an optimal value for \(N_{d}\) that is relatively low \(({\sim}10^{15}\;\mathrm{cm}^{-3}\) for the case of no charge carrier diffusion) combined with a \(\mu>2000\;\mathrm{cm}^{2}/\mathrm{V}{-}\mathrm{s}\) . For the semiconductor material parameters utilized in this work, which corresponds to \(N_{d}=3\times10^{16}\;\mathrm{cm}^{-3}\) and \(\mu=\) \(4220\;\mathrm{cm}^{2}/\mathrm{V}{-}\mathrm{s}\) , we find that \(|\eta_{3}u_{1}|=25\;\mathrm{cm}^{-1}\) .

Optimization of the semiconductor material parameters could lead to exceptionally larger \(\eta_{3}\) in future work. For a mobility of \(\mu=7000\) \(\mathrm{cm}^{2}/\mathrm{V}{-}\mathrm{s}\) and \(N_{d}=3\times10^{14}\;\mathrm{cm}^{-3}\) , we can increase the value for \(|\eta_{3}u_{1}|>200\mathrm{X}\) while maintaining a relatively low linear acoustoelectric attenuation (Supplementary Fig. 2c).

Supplementary Fig. 3: (a) Theoretical value for \( |\eta_{3}u_{1}| \) as a function of semiconductor charge carrier mobility, \( \mu \), and charge carrier density, \( N_{d} \), with and (b) without charge carrier diffusion effects. The star marker in (a) indicates the expected value of \( |\eta_{3}u_{1}| \) in this work given the measured values of \( \mu \) and \( N_{d} \) for the semiconductor (see Methods in main article).

![](dt=2025-08-07/ht=23/8320c63d9779b6855b747e3e6453de7ecc19efc64a245b91112e9698b60f6f06.jpg)

The coupled-wave equations (Eqn. S1) are formulated by the assumption that the total fields of the piezoelectric acoustic waves being coupled are expressed utilizing the plane wave approximation. However, the planar waveguide interaction geometry utilized here makes this a guided-wave system supporting waveguide modes according to mode orthogonality as discussed above. The approach of nonlinear coupled-mode theory then applies to analyze nonlinear interactions in the phononic waveguides. We assume that the total displacement of the interacting waves, \( u \), is expressed as

\[
u(r,t)=\sum_{m}u_{m}(r)\exp(-i\omega_{m}t)
\]

Where \( u_{m}(r) \) is the spatially dependent total displacement field amplitude for frequency \( \omega_{m} \), which can be expanded in terms of the waveguide modes, defined by an index \( \upsilon \), such that

\[
u_{m}(r)=\sum_{\upsilon}A_{m,\upsilon}(x)U_{m,\upsilon}(y)\exp(i\beta_{m}x)
\]

where \( A_{m,v}(x) \) and \( U_{m,v}(y) \) are the mode amplitudes and the normalized displacement mode fields, respectively. Here we are assuming that only one mode for each frequency is involved in the nonlinear interaction, therefore we have

\[
u_{m}(r)=A_{m}(x)U_{m}(y)\exp(i\beta_{m}x).
\]

To adapt Eqn. S1 to the planar waveguide nonlinear interaction geometry of the acoustoelectric heterostructure, we have the following set of equations, which completes the phenomenological model of the nonlinear interaction in the acoustoelectric heterostructure:

\[
\begin{array}{l}\displaystyle\frac{dA_1}{dx}=-\alpha_1A_1+\kappa_1A_2^*A_3e^{i\Delta\beta x}\\\displaystyle\frac{dA_2}{dx}=-\alpha_2A_2+\kappa_2A_1^*A_3e^{i\Delta\beta x}\\\displaystyle\frac{dA_3}{dx}=-\alpha_3A_3+\kappa_3A_1A_2e^{-i\Delta\beta x}\end{array}
\]

where \( \alpha_{m} \) are the linear acoustoelectric coefficients (Eqn. S7) and \( \kappa_{m} \) are the nonlinear coupling coefficients given by

\[
\begin{array}{c}{\displaystyle\kappa_{1}=\eta_{1}\int\displaylimits_{-\infty}^{\infty}U_{1}^{*}U_{2}^{*}U_{3}dy}\\{\displaystyle\kappa_{2}=\eta_{2}\int\displaylimits_{-\infty}^{\infty}U_{1}^{*}U_{2}^{*}U_{3}dy}\\{\displaystyle\kappa_{3}=\eta_{3}\int\displaylimits_{-\infty}^{\infty}U_{1}U_{2}U_{3}^{*}dy}\end{array}
\]

and \( \eta_{m} \) are the nonlinear acoustoelectric coefficients \( \eta_{m} \) (Eqn. S6), which exist outside of the integral as they are spatially constant values that quantify the size of the nonlinearity that arises via electron-phonon coupling in our acoustoelectric heterostructure.

# Supplementary Note 4

# Sum frequency generation in lithium niobate

In this supplementary note, we show experimental results taken for sum frequency generation in an identical device to that presented in Fig. 2 in the main text, except that the In0.53Ga0.47As semiconductor film is removed. Supplementary Fig. 4(a) shows plots of both the phononic power conversion efficiency and the electrical power conversion efficiency as a function of pump power. A peak phononic power conversion efficiency of \( (0.8 \pm 0.5)\% \) and a peak electrical power conversion efficiency of \( (0.06 \pm 0.03)\% \) is achieved at an internal pump power of \( 0.01 \, \mathrm{mW} \).

Unlike the acoustoelectric heterostructure devices, improved power conversion efficiency is not achieved at larger power powers, as shown in the plot of electrical power conversion efficiency as a function of detection frequency for an internal pump power of \( 13 \, \mathrm{mW} \) (Supplementary Fig. 4(b)). As discussed in the main text, the power dependence and saturation behavior for both the nonlinearity in \( \mathrm{LiNbO}_{3} \) alone and the acoustoelectric nonlinearity require further study.

Supplementary Fig. 4: (a) Phononic power conversion efficiency \((PCE)\) and electrical \(PCE\) (inset) as a function of internal pump power. (b) Electrical \(PCE\) as a function of frequency at an internal pump power of \(13\;\mathrm{mW}\).

![](dt=2025-08-07/ht=23/3306920c9a340cf5ece99e81ac31d28446e5a8d23afc750e8542c9efca9d4562.jpg)

# Supplementary Note 5 Difference frequency generation

As shown in Supplementary Fig. 5(a), difference frequency generation is another parametric three-wave mixing process where, in this case, energy conservation requires that a phonon at \(\omega_{3}\) is annihilated to create phonons at \(\omega_{2}\) and \(\omega_{1}=\omega_{3}-\omega_{2}\). A microscope image of the developed device in the semiconductor piezoelectric system to study this frequency conversion process is shown in Supplementary Fig. 5(b).

The transducer for detection is designed to have a resonance at the difference frequency \(f_{1}=\omega_{1}/2\pi=\omega_{3}/2\pi-\omega_{2}/2\pi\). A plot of the measured electrical reflection as a function of frequency for the interdigital phonon transducers is shown in Supplementary Fig. 5(c). The resonance frequencies are found to be \(f_{3}=\omega_{3}/2\pi=366.8\;\mathrm{MHz}\), \(f_{2}=\omega_{2}/2\pi=253.5\;\mathrm{MHz}\), and \(f_{1}=113.3\;\mathrm{MHz}\).

Supplementary Fig. 5: Difference frequency generation. (a) Diagram of phonon energy conservation for difference frequency generation. (b) Microscope image of a device to study this parametric three-wave mixing process in the $\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}-\mathrm{LiNbO}_{3}$ on silicon ac
oustoelectric heterostructure. The acoustoelectric interaction length, defined by the length of the patterned $\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}$, is $250\;\upmu\mathrm{m}$. (c) Measured reflection as a function of frequency for the three interdigital transducers (IDTs) used to generate and detect phonons in the system. (d) A schematic of the experimental setup for power conversion efficiency measurements. (e) Measured electrical power conversion efficiency $(PCE)$ as a function of detection frequency for increasing internal pump powers. (f) Phononic power conversion efficiency $(PCE)$ as a function of internal pump power for open and shorted electrical contact to the $\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}$. The modeled predicted value for the phononic $PCE$ is also shown.

![](dt=2025-08-07/ht=23/fe65cd525acf98717976bc3b65b28403ee379bdf6120808995e2eb4ed1de242e.jpg)

A schematic of the experimental setup to measure power conversion efficiency for difference frequency generation is shown in Supplementary Fig. 5(d), and the setup is described in detail in the Methods. As in the sum frequency generation experiments described in the main text, a radio frequency signal generator is used to generate \(f_{2}\) and a network analyzer is used to generate \(f_{3}\) and detect \(f_{3}\).

As described in the main text, the electrical conversion efficiency includes the transducer conversion losses and the phononic power conversion efficiency is determined by the electrical power conversion efficiency and measured conversion losses from separate acoustic delay line test structures (Supplementary Note 1). Supplementary Fig. 5(e) shows the measured terminal power conversion efficiency as a function of the detection frequency for increasing internal pump powers. The power conversion has a peak value at the transducer resonance frequency which increases with increasing pump power.

The 3 dB fractional bandwidth supported for the frequency conversion measurement is \(2.9\%\), limited by the output transducer bandwidth.

Supplementary Fig. 5(f) shows a plot of the phononic power conversion efficiency as a function of internal pump power for open and shorted electrical contacts to the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconducting layer. A maximum internal power conversion efficiency of \((1.0{\pm}0.1)\%\) is achieved for the shorted electrode configuration with an internal pump power of \(5\;\mathrm{mW}\). The expected theoretical values, as determined by numerical integration of the nonlinear coupled-mode equations (Supplementary Note 3), are also shown in Supplementary Fig. 5(f).

Good agreement between the modeled and experimental values occurs up to an internal pump power of approximately \(2\;\mathrm{mW}\) above which the experimental values are significantly less than the theoretically predicted values. As discussed in the main text, deviation of the experimental values from the theoretical values at large pump power could be due to several effects including pump depletion, self-phase and cross-phase modulation, and thermal detuning.

# Supplementary Note 6 Phase matching with acoustic and acoustoelectric dispersion

For a second-order nonlinear interaction characterized by the relation \(\omega_{3}=\omega_{1}+\omega_{2}\), the collinear phase-matching condition is \(\beta_{3}=\beta_{1}+\beta_{2}\), which for the case of phononic nonlinear interactions, is equivalent to \(\omega_{3}/v_{a,3}=\omega_{1}/v_{a,1}+\omega_{2}/v_{a,2}\). There are multiple sources of dispersion in our system that lead to the phase mismatch.

The \(\mathrm{LiNbO}_{3}\) planar waveguide geometry introduces modal dispersion where the speed of sound for a given mode depends on the thickness of the waveguide and the mode's polarization. The acoustic velocity takes a value between the speed of sound in the \(\mathrm{LiNbO}_{3}\) and the speed of sound in the silicon for each acoustic wavelength value. The \(\sim50\;\mathrm{nm}\) thick \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) film modifies this modal dispersion by mass loading effects on top of the \(\mathrm{LiNbO}_{3}\).

In addition, the same electron-phonon acoustoelectric interaction that leads to linear attenuation of the phonons also modifies the acoustic velocity and is a function of the frequency.32,47

From delay line test structures with interdigital transducers designed to launch and detect phonons across a range of frequencies, we can determine the acoustic dispersion for the generated phonons. These test structures do not include the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) film and therefore only provide information on the dispersion due to the \(\mathrm{LiNbO}_{3}\) film alone. A plot of the measured acoustic velocity as a function of the frequency is shown in Supplementary Fig. 6(a).

Without the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) semiconductor film, our phase mismatch is given by \(\Delta\beta=\frac{\omega_{3}}{v_{a,3}}-\frac{\omega_{1}}{v_{a,1}}-\frac{\omega_{2}}{v_{a,2}}=\frac{2\pi\times626.6\;\mathrm{MHz}}{4043\frac{\mathrm{m}}{\mathrm{s}}}-\frac{2\pi\times370\;\mathrm{MHz}}{4060\frac{\mathrm{m}}{\mathrm{s}}}-\frac{2\pi\times256.6\;\mathrm{MHz}}{4110\frac{\mathrm{m}}{\mathrm{s}}}=89\;\mathrm{cm}^{-1}\) for sum frequency generation and

\(\Delta\beta=\frac{\omega_{3}}{v_{a,3}}-\frac{\omega_{1}}{v_{a,1}}-\frac{\omega_{2}}{v_{a,2}}=\frac{2\pi\times367.2\;\mathrm{MHz}}{4060\frac{\mathrm{m}}{\mathrm{s}}}-\frac{2\pi\times113.7\;\mathrm{MHz}}{4415\frac{\mathrm{m}}{\mathrm{s}}}-\frac{2\pi\times253.5\;\mathrm{MHz}}{4110\frac{\mathrm{m}}{\mathrm{s}}}=189\;\mathrm{cm}^{-1}\) for difference frequency generation.

These values are taken directly from experimental measurements of acoustic delay line test structures.

A finite element method model was utilized to predict the impact of the mass loading of the \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) film on the dispersion. A \(50\;\mathrm{nm}\) \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) film was placed on the top surface of the \(5\;\upmu\mathrm{m}\) thick \(\mathrm{LiNbO}_{3}\), which is placed on top of a silicon substrate. The acoustic frequency of the quasi shear horizontal \((SH_{0})\) mode was monitored as a function of changing the supported acoustic wavelength for the structure and from this information we determined the acoustic velocity as a function of the acoustic wavelength (Supplementary Fig. 6(b)).

Supplementary Fig. 6: Dispersion and phase mismatch. (a) Measured acoustic velocity as a function of frequency for acoustic delay line test structures on the \(\mathrm{LiNbO}_{3}\)-silicon substrate. (b) Modeled acoustic velocity as a function of wavelength with and without the mass loading effect of the \(50\;\mathrm{nm}\) thick \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) film. (c) Theoretical calculations of the velocity change \((\Delta v/v)\) as a function of semiconductor charge carrier concentration for the pump \((\omega_{1}/2\pi=370\,\mathrm{MHz})\), signal \((\omega_{2}/2\pi=256.6\,\mathrm{MHz})\), and sum frequency \((\omega_{3}/2\pi=626.6\,\mathrm{MHz})\) as a function of charge carrier concentration due to dispersion from the acoustoelectric effect. (d) Expected velocity change \(\Delta v/v\) as a function of frequency due to acoustoelectric dispersion and dispersion from the \(50\;\mathrm{nm}\) thick \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) film.

![](dt=2025-08-07/ht=23/95ffc9a573dd4b7b910a58897776bfa3e024265fc654347a9258b604a09ce802.jpg)

It is well-known that the acoustoelectric effect itself is dispersive and a model of the acoustoelectric dispersion in a heterostructure like ours has been previously developed\(^{32}\) which we can apply here to estimate the velocity change. A plot of the theor
etical \(\Delta v/v\) as a function of semiconductor charge carrier density is shown in Supplementary Fig. 6(d) for the pump frequency \((\omega_{1}/2\pi=370\,\mathrm{MHz})\), the signal frequency \((\omega_{2}/2\pi=256.6\,\mathrm{MHz})\), and the sum frequency \((\omega_{3}/2\pi=626.6\,\mathrm{MHz})\). As can be seen, the dispersion due to the acoustoelectric effect increases with decreasing carrier concentration and increasing frequency for the range of carrier concentration and frequencies shown in Supplementary Fig. 6(c).

A plot of the modeled and theoretical \(\Delta v/v\) as a function of frequency is shown in Supplementary Fig. 6(d) for the cases of acoustoelectric dispersion and \(\mathrm{In}_{0.53}\mathrm{Ga}_{0.47}\mathrm{As}\) mass loading. This plot is used to approximate appropriate modifications to the measured acoustic velocities shown in Supplementary Fig. 5(a). Adding these two effects together gives us the total change in acoustic velocity as a function of frequency, which is utilized to adjust the measured values. Including the impact of mass loading and the acoustoelectric effect, the phase mismatch

for sum frequency generation is \(\Delta\beta=\frac{2\pi\times626.6\,\mathrm{MHz}}{4016\frac{\mathrm{m}}{\mathrm{s}}}-\frac{2\pi\times370\,\mathrm{MHz}}{4041\frac{\mathrm{m}}{\mathrm{s}}}-\frac{2\pi\times256.6\,\mathrm{MHz}}{4095\frac{\mathrm{m}}{\mathrm{s}}}=113\,\mathrm{cm}^{-1}\).

The phase mismatch for difference frequency generation is approximately \(\Delta\beta=\frac{2\pi\times367.2\,\mathrm{MHz}}{4029\frac{\mathrm{m}}{\mathrm{s}}}-\frac{2\pi\times113.7\,\mathrm{MHz}}{4408\frac{\mathrm{m}}{\mathrm{s}}}-\frac{2\pi\times253.5\,\mathrm{MHz}}{4092\frac{\mathrm{m}}{\mathrm{s}}}=213\,\mathrm{cm}^{-1}\).

While the phase mismatch has increased compared to the \(\mathrm{LiNbO}_{3}\) film alone, we do see that the majority of the phase mismatch, for the material parameters and frequencies utilized here, can be attributed to the dispersion of the \(\mathrm{LiNbO}_{3}\) planar waveguide.

# Supplementary Note 7

# Linear and nonlinear acoustoelectric coefficients with an applied drift field

The variation in the linear and nonlinear acoustoelectric coefficients (\(\alpha_{3}\) and \(|\eta_{3}u_{1}|\)), respectively) with the ratio between the charge carrier drift velocity, \(v_{d}\), and the acoustic velocity, \(v_{a,3}\), is shown in Supplementary Fig. 7. A positive velocity ratio corresponds to when the charge carriers drift in the same direction as the propagating piezoelectric phonons while a negative velocity ratio corresponds to when the direction of charge carrier drift is opposite to the phonon propagation.

At a velocity ratio equal to 1, the linear acoustoelectric coefficient is zero and the nonlinear acoustoelectric coefficient reaches a minimum value. As the velocity ratio increases from 1, the magnitude of the nonlinear acoustoelectric coefficient increases while negative attenuation (amplification) is achieved. For a negative velocity ratio, attenuation increases, but so does the magnitude of the nonlinear acoustoelectric coefficient.

Supplementary Fig. 7: Dispersion and phase mismatch. The linear, \(\alpha_{3}\), and nonlinear, \(|\eta_{3}u_{1}|\), acoustoelectric coefficients are shown as a function of the ratio between the charge carrier drift velocity, \(v_{d}\), and the acoustic velocity, \(v_{a,3}\).

![](dt=2025-08-07/ht=23/5bdb975773a91d6b10b84f98943981c15d83676a5760256df57c7c887e40d083.jpg)

# Supplementary Note 8 Four-wave mixing coefficient

The four-wave mixing coefficeint and nonlinear modal coefficients were determined using the previously reported methods.27 The mode amplitude for the phonon displacement for the \(n^{\mathrm{th}}\) waveguide mode generated by the cascaded nonlinearity, \(A_{n}\) , is given by

\[
\frac{dA_n}{dx}=-\frac{\alpha}{2}A_n+i\gamma_m\sum_{p+d-m=n}A_pA_qA_m^*e^{i\Delta\beta x}
\]

where \(\alpha\) is the phononic propagation loss, \(\gamma_{m}\) is the modal nonlinear coefficient, \(\Delta\beta=\beta_{p}+\beta_{p}-\beta_{m}-\beta_{n}\) is the phase mismatch, and \(A_{n}\) is normalized such that the acoustic power in the \(n^{\mathrm{th}}\) mode, \(P_{n}\) , is \(P_{n}=|A_{n}|^{2}\) . For the small frequency spacing and interaction lengths, \(L\) , we study in this work, we can assume that \(\Delta\beta_{i j}L\) is approximately equal to 0 for all modes \(i,j\) studied. In the low power limit and under the assumption that the phonons in the first sidebands are created by degenerate four-wave mixing, the efficiencies with which these phonons are created is given by

\[
\begin{array}{l}{PCE_{112}=\left|\frac{A_{f112}}{A_{f2}^{*}}\right|^{2}=T_{112}P_{1}^{\prime2}}\\{PCE_{221}=\left|\frac{A_{f221}}{A_{f1}^{*}}\right|^{2}=T_{221}P_{2}^{\prime2}}\end{array}
\]

where \(P C E_{112}\) is the power conversion efficiency to generate the phonons with amplitude \(A_{f112}\) at frequency \(f_{112}\) , \(P C E_{221}\) is the power conversion efficiency to generate the phonons with amplitude \(A_{f221}\) at \(f_{221},A_{f1}\) and \(A_{f2}\) are the amplitudes for the pumps with frequencies \(f_{1}\) and \(f_{2}\) , respectively, \(P_{1}^{\prime}\) and \(P_{2}^{\prime}\) are the input pump powers with frequencies \(f_{1}\) and \(f_{2}\) , respectively, at the beginning of the waveguide directly after the input interdigital transducer, and

\(\Gamma=(\gamma_{m}L_{\mathrm{eff}})^{2}\) is the FWM coefficient where \(L_{\mathrm{eff}}=[1-\exp{(-\alpha L)}]/\alpha\) is the effective length and \(L\) is the interaction length.

Two values for \(\boldsymbol{\Gamma}\) are then found by determining the slopes of \(P_{112}/P_{2}\) as a function of \(P_{1}^{\prime2}\) and \(P_{221}/P_{1}\) as a function of \(P_{2}^{\prime2}\) , and we use \(\Gamma=(\Gamma_{112}+\Gamma_{221})/2\) to assess the net nonlinearity.

The modal nonlinear coefficients, \(\gamma_{m}\) , are calculated according to the values, for the different device types, for \(\boldsymbol{\Gamma}\) and the values for \(L_{\mathrm{eff}}\) , where the \(L_{\mathrm{eff}}\) values are calculated from the experimentally extracted acoustoelectric attenuation and the acoustic propagation losses.

# References

1 Hendrickson, S. M., Foster, A. C., Camacho, R. M. & Clader, B. D. Integrated nonlinear photonics: emerging applications and ongoing challenges. JOSA B 31, 3193-3203 (2014). 2 Leuthold, J., Koos, C. & Freude, W. Nonlinear silicon photonics. Nature photonics 4, 535-544 (2010). 3 Moody, G., Chang, L., Steiner, T. J. & Bowers, J. E. Chip-scale nonlinear photonics for quantum light generation. AVS Quantum Science 2, 041702 (2020). 4 Bers, A. & Cafarella, J. Surface state memory in surface acoustoelectric correlator. Applied Physics Letters 25, 133-135 (1974). 5 Cafarella, J. H., Brown, W., Stern, E. & Alusow, J. Acoustoelectric convolvers for programmable matched filtering in spread-spectrum systems. Proceedings of the IEEE 64, 756-759 (1976).

6 Kino, G. S. Acoustoelectric interactions in acoustic-surface-wave devices. Proceedings of the IEEE 64, 724-748 (1976).  7 Reible, S. A. Acoustoelectric convolver technology for spread-spectrum communications. IEEE Transactions on Microwave Theory and Techniques 29, 463-474 (1981).  8 Hackett, L. et al. Towards single-chip radiofrequency signal processing via acoustoelectric electron–phonon interactions. Nature communications 12, 2769 (2021).  9 Bogdanov, S., Shalaginov, M., Boltasseva, A. & Shalaev, V. M. Material platforms for integrated quantum photonics.

Optical Materials Express 7, 111-132 (2017).  10 Elshaari, A. W., Pernice, W., Srinivasan, K., Benson, O. & Zwiller, V. Hybrid integrated quantum photonic circuits. Nature Photonics 14, 285-298 (2020).  11 Politi, A., Matthews, J. C., Thompson, M. G. & O'Brien, J. L. Integrated quantum photonics. IEEE Journal of Selected Topics in Quantum Electronics 15, 1673-1684 (2009).  12 MacC
abe, G. S. et al. Nano-acoustic resonator with ultralong phonon lifetime. Science 370, 840-843 (2020).  13 Wollack, E. A. et al.

Loss channels affecting lithium niobate phononic crystal resonators at cryogenic temperature. Applied Physics Letters 118, 123501 (2021).  14 Guo, Y. & Wang, M. Phonon hydrodynamics and its applications in nanoscale heat transport. Physics Reports 595, 1-44 (2015).  15 Joshi, A. & Majumdar, A. Transient ballistic and diffusive phonon heat transport in thin films. Journal of Applied Physics 74, 31-39 (1993).  16 Arora, V. K. & Naeem, A. Phonon-scattering-limited mobility in a quantum-well heterostructure. Physical Review B 31, 3887 (1985).  17 Hwang, E. & Sarma, S. D.

Acoustic phonon scattering limited carrier mobility in two-dimensional extrinsic graphene. Physical Review B 77, 115449 (2008).  18 Zhou, J.-J. & Bernardi, M. Ab initio electron mobility and polar phonon scattering in GaAs. Physical Review B 94, 201201 (2016).  19 Shao, L. et al. Electrical control of surface acoustic waves. Nature Electronics 5, 348-355 (2022).  20 Abdelkefi, A., Nayfeh, A. H. & Hajj, M. R. Effects of nonlinear piezoelectric coupling on energy harvesters under direct excitation. Nonlinear Dynamics 67, 1221-1232 (2012).  21 Mahboob, I., Wilmart, Q., Nishiguchi, K.

, Fujiwara, A. & Yamaguchi, H. Wide-band idler generation in a GaAs electromechanical resonator. Physical Review B 84, 113411 (2011).  22 Luukkala, M. v. & Kino, G. Convolution and time inversion using parametric interactions of acoustic surface waves. Applied Physics Letters 18, 393-394 (1971).  23 Kurosu, M., Hatanaka, D., Onomitsu, K. & Yamaguchi, H. On-chip temporal focusing of elastic waves in a phononic crystal waveguide. Nature communications 9, 1331 (2018).  24 Kurosu, M., Hatanaka, D. & Yamaguchi, H.

Mechanical Kerr nonlinearity of wave propagation in an on-chip nanoelectromechanical waveguide. Physical Review Applied 13, 014056 (2020).  25 Maksymov, I. S., Huy Nguyen, B. Q., Pototsky, A. & Suslov, S. Acoustic, phononic, Brillouin light scattering and Faraday wave-based frequency combs: Physical foundations and applications. Sensors 22, 3921 (2022).  26 Mansoorzare, H. & Abdolvand, R. in 2023 IEEE 36th International Conference on Micro Electro Mechanical Systems (MEMS). 1183-1185 (IEEE).

27 Mayor, F. M. et al. Gigahertz phononic integrated circuits on thin-film lithium niobate on sapphire. Physical Review Applied 15, 014039 (2021).  28 Conwell, E. & Ganguly, A. Mixing of acoustic waves in piezoelectric semiconductors. Physical Review B 4, 2535 (1971).  29 Johri, G. & Spector, H. N. Boltzmann-equation approach to nonlinear acoustoelectric interactions in piezoelectric semiconductors. Physical Review B 12, 3215 (1975).  30 Wu, C. C. & Spector, H. N. Ultrasonic harmonic generation in piezoelectric semiconductors. Journal of Applied Physics 43, 2937-2944 (1972).  31 Adler, R.

Simple theory of acoustic amplification. IEEE Transactions on Sonics and Ultrasonics 18, 115-118 (1971).  32 Kino, G. & Reeder, T. A normal mode theory for the Rayleigh wave amplifier. IEEE Transactions on Electron Devices 18, 909-920 (1971).  33 Coldren, L. A. & Kino, G. Monolithic acoustic surface - wave amplifier. Applied Physics Letters 18, 317-319 (1971).  34 Hackett, L. et al. Non-reciprocal acoustoelectric microwave amplifiers with net gain and low noise in continuous operation. Nature Electronics, 1-10 (2023).  35 Mansoorzare, H. & Abdolvand, R.

Micromachined Heterostructured Lamb Mode Waveguides for Acoustoelectric Signal Processing. IEEE Transactions on Microwave Theory and Techniques 70, 5195-5204 (2022).  36 Boyd, R. W. Nonlinear optics. (Academic press, 2020).  37 Rodriguez, A., Soljačić, M., Joannopoulos, J. D. & Johnson, S. G. \(\chi(2)\) and \(\chi(3)\) harmonic generation at a critical power in inhomogeneous doubly resonant cavities. Optics express 15, 7303-7318 (2007).  38 Eichenfield, M. & Olsson, R. in 2013 IEEE International Ultrasonics Symposium (IUS). 753-756 (IEEE).  39 Kumar, P. Quantum frequency conversion.

Optics letters 15, 1476-1478 (1990).  40 Banszerus, L. et al. Ultrahigh-mobility graphene devices from chemical vapor deposition on reusable copper. Science advances 1, e1500222 (2015).  41 Kim, M.-S. et al. Sheet resistance analysis of interface-engineered multilayer graphene: mobility versus sheet carrier concentration. ACS applied materials & interfaces 12, 30932-30940 (2020).  42 Laroche, D., Das Sarma, S., Gervais, G., Lilly, M. & Reno, J. Scattering mechanism in modulation-doped shallow two-dimensional electron gases. Applied Physics Letters 96, 162112 (2010).  43 Otterstrom, N. T.

et al. Nonreciprocal frequency domain beam splitter. Physical review letters 127, 253603 (2021).  44 Mirhosseini, M., Sipahigil, A., Kalaee, M. & Painter, O. Superconducting qubit to optical photon transduction. Nature 588, 599-603 (2020).  45 Jiang, W. et al. Efficient bidirectional piezo-optomechanical transduction between microwave and optical frequency. Nature communications 11, 1166 (2020).  46 Auld, B. A. Acoustic fields and waves in solids. (Рипол Классик, 1973).  47 White, D. L. Amplification of ultrasonic waves in piezoelectric semiconductors.

Journal of Applied Physics 33, 2547-2554 (1962).