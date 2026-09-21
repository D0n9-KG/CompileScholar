# Topologically Enabled Ultra-high-Q Guided Resonances Robust to Out-of-plane Scattering

Jicheng Jin $^{1}$ , Xuefan Yin $^{1}$ , Liangfu Ni $^{1}$ , Marin Soljačić $^{2}$ , Bo Zhen $^{3}$ , & Chao Peng $^{1,*}$

$^{1}$ State Key Laboratory of Advanced Optical Communication Systems and Networks, Peking University, Beijing 100871, China   
$^{2}$ Department of Physics, Massachusetts Institute of Technology, Cambridge, MA 02139, USA.   
$^{3}$ Department of Physics and Astronomy, University of Pennsylvania, Philadelphia, PA 19104, USA

Due to their ability to confine light, optical resonators $^{1-3}$ are of great importance to science and technology, yet their performances are often limited by out-of-plane scattering losses from inevitable fabrication imperfections $^{4,5}$ . Here, we theoretically propose and experimentally demonstrate a class of guided resonances in photonic crystal slabs, where out-of-plane scattering losses are strongly suppressed due to their topological nature. Specifically, these resonances arise when multiple bound states in the continuum - each carrying a topological charge $^{6}$ - merge in the momentum space and enhance the quality factors of all resonances nearby. We experimentally achieve quality factors as high as $4.9 \times 10^{5}$ based on these resonances in the telecommunication regime, which is 12-times higher than ordinary designs. We further show this enhancement is robust across the samples we fabricated. Our work paves the way for future explorations of topological photonics in systems with open boundary condition and their applications in improving optoelectronic devices in photonic integrated circuits.

Topological defects $^{7}$ are ubiquitous in nature. Examples range from quantum vortices in superfluids to singular optical beams $^{8}$ , which are characterized by the non-trivial winding patterns of system parameters (velocity, phase, or polarization) in real space. Recently it is found that unexpected topological defects can also emerge in the momentum space of a crystal and give rise to interesting physical consequences: one such example is the optical bound states in the continuum (BICs). BICs reside inside the continuous spectrum of extended states, yet, defying the common intuition, remain perfectly localized in space and their lifetimes are supposed to be infinitely long. Since their initial proposal $^{9}$ , BICs have been observed in a variety of wave systems, including photonic $^{10-21}$ , phononic $^{22}$ , and water waves $^{23}$ . In photonic crystal (PhC) slabs, their fundamental nature has been identified to be topological: they are essentially topological defects of polarization directions defined in the momentum space $^{6}$ . In practice $^{12,24}$ , the Qs of BICs often fall much shorter of their theoretical prediction of infinity, limited to only about $1 \times 10^{4}$ . Aside from other contributing factors such as material absorption or the finite size of the samples, the main limiting factor of the Q of BICs comes from out-of-plane scattering losses from fabrication imperfections or disorder - a common problem shared among many high-Q on-chip resonators $^{1,2,4,5,25}$ .

Here we theoretically propose and experimentally demonstrate on-chip photonic resonances that are much less susceptible to out-of-plane scattering losses than usual due to their unique topological nature. Specifically, we first show that the topological charges of BICs control the Qs of their surrounding resonances; more importantly, when multiple BICs are designed to merge, all modes nearby enjoy significant enhancements of their Qs due to a modified scaling rule. We further numerically show that the resulting resonances, in this new topological configuration, become

robust to fabrication imperfections and disorder. Finally, we experimentally demonstrate a 12-times enhancement of Qs in fabricated samples using this new topological configuration of BICs over previous designs.

We start by showing that resonances with ultra-high quality factors (Qs) that are much more robust to out-of-plane scattering from disorder can be achieved by merging multiple topological charges carried by BICs. First, we consider a PhC slab (Fig. 1a), where a square lattice (periodicity $a = 519.25 \mathrm{~nm}$ ) of circular air holes (radius $r = 175 \mathrm{~nm}$ ) is patterned in a silicon layer (thickness of $h = 600 \mathrm{~nm}$ ) placed in the air. Through numerical simulations (COMSOL Multiphysics), we focus on the lowest TE-like band in the continuum (TE-A, red line) whose lifetime goes to infinity at 9 discrete $k$ points as 9 BICs (top left panel of Fig. 1b). The topological nature of the BICs can be understood from the corresponding far-field polarization plot (bottom left panel), where each BIC appears as a topological defect (vortex) of polarization long axes $^{6,26-31}$ characterized by an integer topological charge of $\pm 1$ . Among these 9 vortices, one is pinned at the center of the Brilluion zone (BZ) due to symmetry, while the locations of rest 8 can be controlled by varying system parameters such as periodicity $a$ . For example, when $a$ increases from $519.25 \mathrm{~nm}$ to $531.42 \mathrm{~nm}$ , the 8 off-center vortices move towards the center before all 9 of them merge into a single BIC with charge of $+1$ as $a$ further increases to $580 \mathrm{~nm}$ .

The topological configuration of BICs controls the radiation loss of all nearby resonances, which further determines the highest Q achievable in practice as shown later. Specifically, Q is shown to scale quadratically ( $\propto 1/k^{2}$ ) as the distance (k) away from a single isolated BIC

with charge $\pm1$ ; however, this scaling changes to $Q \propto 1/k^{6}$ in a sample where all 9 BICs just merge (noted as the “merging-BIC design” hereafter). The comparison between these two scenarios are shown in Fig. 1c, where Qs in a merging-BIC design (red) are always orders-of-magnitude higher than those in an isolated-BIC design (blue) along all directions in k space due to their fundamentally different scaling properties. This difference in scaling originates from the different asymptotic behaviors of radiation amplitudes $\sqrt{1/Q}$ : in the isolated-BIC case, $\sqrt{1/Q} \propto k$ ; in comparison, when there are also off-center BICs at $\pm k_{BIC}$ , $\sqrt{1/Q}$ becomes proportional to $(k + k_{\mathrm{BIC}})(k - k_{\mathrm{BIC}})k$ . In the merging-BIC design, $k_{BIC} = 0$ and we get $1/Q \propto k^{6}$ . This different scaling is similar to some of the recent findings $^{32-34}$ . Further explanations from the viewpoint of coupled-wave theory is presented in section I and II of the Supplementary Information.

While simulation results of infinitely-large perfect PhCs set the theoretical upper bounds of Qs, realistic samples (schematically shown in Fig. 2a) feature a few major differences that govern the highest Q achievable in practice. First, all samples are finite in size; their boundaries break the translation symmetry and introduce fractional orders of the primitive reciprocal lattice in k space (green dots in Fig. 2a) $^{35,36}$ . As a result, each infinitely-large Bloch mode with a single k-component is split into a series of finite-size modes with multiple k-components. See Supplementary Information Section III for an example of this effect experimentally observed in our sample. Second, all fabricated samples exhibit disorder and imperfections with both long- and short-range correlations, allowing modes at different k points to couple to each other. Due to these inevitable coupling terms, modes at different fractional momentum orders are hybridized and all of their loss channels become available to the final resonance $^{37}$ .

The advantage of our merging-BIC design over an isolated-BIC design is confirmed in our simulation results (COMSOL Multiphysics) of perturbed $15 \times 15$ PhC super-cells. In a perfect super-cell structure without disorder, the BIC mode with infinite-Q remains at the center of the BZ (Fig. 2b, upper panel). To compare, perturbations are applied to both the radii ( $\Delta r$ ) and positions of the holes ( $\Delta x, \Delta y$ ) following the statistics that best captures our samples described later in Fig. 3. As expected, each mode in disordered samples exhibits multiple components in the k space. Furthermore, resonances in a disordered sample originated in a merging-BIC design have significantly lower radiation fields than those from a isolated-BIC design with the same disorder (Fig. 2c). This result agrees well with Fig. 1b,c: all modes contributing to the final resonance in the merging-BIC sample have much higher Qs than those in the isolated-BIC case; naturally, resonances in the former sample are much more immune to out-of-plane scattering from disorder than the latter. Finally, this enhancement of Q is observed to be robust across a range of k as shown in the quantitative comparison (Fig. 2d). Here, all holes are asymmetric to present typical fabrication error with tilted angle $\theta \approx 2^{\circ}$ and center shift ( $\Delta x = 2nm, \Delta y = 4nm$ ) before applying disorder. (see Supplementary Information section IV and V for details)

To verify our theoretical findings, we fabricate PhC samples with both merging-BIC and isolated-BIC designs using the same e-beam lithography (EBL) and induced coupled plasma (ICP) etching processes on a 600 nm thick silicon-on-insulator wafer (see Methods for details). The underlying $SiO_{2}$ layer is then removed to restore the up-down mirror symmetry required by tunable BICs ${}^{6,12}$ . The samples are about $250 \times 250 \mu m$ in size. The periodicity of the sample is varied from 530 to 580 nm to sample through designs with merging and isolated BICs. From the scanning

electron microscope images of the samples (Fig. 3a,b), the standard deviations of hole locations and radii are estimated to be about 5 nm, which is applied to the numerical simulations presented above.

The experimental setup is schematically shown in Fig.3c. A tunable telecommunication laser in the C-band is first sent through a X-polarizer (Pol,X) before being focused by lens 1 (L1) onto the back focal plane of an infinity-corrected objective lens. The incident angle of the laser on the sample is thus controlled by moving L1 in the x-y plane. Through this confocal setup, reflected and scattered light are also collected by the same objective; they are then expanded by 1.67 times through a relay 4-f system and imaged on a camera. A Y-polarizer (Pol, Y) is used to block reflected light (X-polarized), while allowing scattered light to pass (see Supplementary Information Section VI for details). Under the on-resonance coupling condition, where the PhC sample supports a resonance at the same wavelength as the incident light at the incident angle, iso-frequency contours are observed on the camera, similar to previously reported results ${}^{26,38}$ . Three examples of isofrequency contours are schematically shown in Fig. 4a as dashed lines.

The quality factors of resonances at different k points are further characterized through scattered light. Specifically, a movable pin hole (not shown in Fig. 3c) is placed at the image plane of the rear focal plane of the objective to specify a k point. A photo-diode connected to a lock-in amplifier is placed behind the pin hole to record scattered light intensity as a function of the tunable laser wavelength (see Supplementary Information Section VI for details). As shown in Fig. 4a, when different k points are selected by the pin hole (X, Y, and Z), different scattering spectra

are observed while all exhibiting symmetric Lorentzian features. Similar scattering phenomena have been observed before $^{38}$ , and can be understood as the follows: scattered light intensity is governed by the spectral density of states of the sample at this k point, which are Lorentzian functions centered at the resonance frequencies with linewidths determined by the Q of the resonances (see Supplementary Information Section VII for details).

The quality factors of the resonances are extracted by numerically fitting the scattering spectra to Lorentzian functions. As shown in Fig. 4a, Q increases from $2.6 \times 10^{5}$ to $4.5 \times 10^{5}$ as the observing point moves closer to the center of the BZ from X to Z. This agrees well with simulation results in Fig. 1. The highest Q observed on the merging-BIC sample is $4.9 \times 10^{5}$ at point W (Fig. 4b). In comparison, the highest Q observed on the isolated-BIC sample, fabricated on the same wafer through the same processes only with different structural parameters, is limited to only $4 \times 10^{4}$ - over an order of magnitude lower (Fig. 4c). This confirms our simulation results in Fig. 2 that engineering the topological configurations of BICs can significantly suppress scattering losses. Furthermore, this over-ten-fold enhancement of quality factor is observed to be robust: not only does it appear over a wide range in the k space as shown in Fig. 5, similar level of enhancements also appears in all merging-BIC samples we fabricated (see Supplementary Information Section VII for details).

Topological photonics $^{39-41}$ have found tremendous success in suppressing in-plane backscattering losses, often based on topological protections in non-reciprocal systems with broken time-reversal symmetry. Here, we focus on a different class of problems: to suppress the out-of-

plane scattering losses in a reciprocal system using concepts from topology. By merging multiple topological charges carried by BICs, we experimentally demonstrate PhC resonances with record-high quality factors of $Q = 4.9 \times 10^{5}$ , over an order of magnitude higher than ordinary designs. These ultra-high-Q resonances are potentially useful for chemical or biological sensing ${}^{42,43}$ , non-linear generation $^{44}$ , and large-area laser applications $^{45}$ . Furthermore, governed by their topological nature, these high-Q resonances are observed to be robust against fabrication imperfections, which paves the way to improve the performance of optoelectronic devices using concepts from topological photonics. Finally, our fundamental concept of topological-defect engineering holds for general linear wave systems, ranging from photonics to acoustics and electronics.

# Methods

Sample fabrication. The sample was fabricated on a silicon-on-insulator (SOI) wafer with e-beam lithography (EBL) followed by induced coupled plasma (ICP) etching. For EBL, the SOI wafer was spin-coated with a 330nm-thick layer of ZEP520A photo-resist before being exposed with EBL (JBX-6300FS) at beam current of 400 pA and field size of 500 $\mu$ m. The sample was then etched with ICP (Oxford Plasmapro Estrelas 100) using a mixture of SF $_{6}$ and C $_{4}$ F $_{8}$ . After etching, the resist was removed with N-Methyl-2-pyrrolidone and the buried oxide layer was removed using 49% HF.

Measurement system. The incident light source was a tunable C-band telecommunication laser (Santec TSL-550), which was sent through a chopper for lock-in detection. A pin hole with diameter of $500 \mu m$ was placed on the Fourier plane to pick out desired wavevectors. Scattered light

through the pin hole was collected by a photo-diode (PDA10DT-EC), which was connected to a lock-in amplifier (SRS SR830). A flip mirror was used to switch between the camera that image iso-frequency contours and the photo-diode. Besides characterizing far-field radiation patterns, the setup could also take near-field images of the sample if another lens was inserted into the optical path.

1. Biberman, A., Shaw, M. J., Timurdogan, E., Wright, J. B. & Watts, M. R. Ultralow-loss silicon ring resonators. Optics letters 37, 4236–4238 (2012).   
2. Hossein-Zadeh, M. & Vahala, K. J. Free ultra-high-q microtoroid: a tool for designing photonic devices. Optics Express 15, 166–175 (2007).   
3. Akahane, Y., Asano, T., Song, B.-S. & Noda, S. High-Q photonic nanocavity in a two-dimensional photonic crystal. Nature 425, 944–947 (2003).   
4. Minkov, M., Dharanipathy, U. P., Houdr, R. & Savona, V. Statistics of the disorder-induced losses of high-Q photonic crystal cavities. Optics Express 21, 28233–28245 (2013).   
5. Ishizaki, K., Okano, M. & Noda, S. Numerical investigation of emission in finite-sized, three-dimensional photonic crystals with structural fluctuations. Journal of the Optical Society of America B 26, 1157–1161 (2009).   
6. Zhen, B., Hsu, C. W., Lu, L., Stone, A. D. & Soljačić, M. Topological nature of optical bound states in the continuum. Physical Review Letters 113, 257401 (2014).

7. Mermin, N. D. The topological theory of defects in ordered media. Reviews of Modern Physics 51, 591–648 (1979).   
8. Gbur, G. J. Singular Optics (CRC Press, 2016).   
9. von Neuman, J. & Wigner, E. Uber merkwrdige diskrete Eigenwerte. Uber das Verhalten von Eigenwerten bei adiabatischen Prozessen. Physikalische Zeitschrift 30, 467–470 (1929).   
10. Imada, M. et al. Coherent two-dimensional lasing action in surface-emitting laser with triangular-lattice photonic crystal structure. Applied physics letters 75, 316–318 (1999).   
11. Hsu, C. W., Zhen, B., Stone, A. D., Joannopoulos, J. D. & Soljačić, M. Bound states in the continuum. Nature Reviews Materials 1, 16048 (2016).   
12. Hsu, C. W. et al. Observation of trapped light within the radiation continuum. Nature 499, 188–191 (2013).   
13. Plotnik, Y. et al. Experimental observation of optical bound states in the continuum. Physical Review Letters 107, 183901 (2011).   
14. Fan, S. & Joannopoulos, J. D. Analysis of guided resonances in photonic crystal slabs. Physical Review B 65, 235112 (2002).   
15. Kodigala, A. et al. Lasing action from photonic bound states in continuum. Nature 541, 196–199 (2017).   
16. Gansch, R. et al. Measurement of bound states in the continuum by a detector embedded in a photonic crystal. Light: Science & Applications 5, e16147 (2016).

17. Gomis-Bresco, J., Artigas, D. & Torner, L. Anisotropy-induced photonic bound states in the continuum. Nature Photonics 11, 232–236 (2017).   
18. Molina, M. I., Miroshnichenko, A. E. & Kivshar, Y. S. Surface bound states in the continuum. Physical Review Letters 108, 070401 (2012).   
19. Carletti, L., Koshelev, K., De Angelis, C. & Kivshar, Y. Giant nonlinear response at the nanoscale driven by bound states in the continuum. Physical Review Letters 121, 033903 (2018).   
20. Friedrich, H. & Wintgen, D. Interfering resonances and bound states in the continuum. Physical Review A 32, 3231–3242 (1985).   
21. Monticone, F. & Alu, A. Embedded photonic eigenvalues in 3d nanostructures. Physical Review Letters 112, 213903 (2014).   
22. Lim, T. C. & Farnell, G. W. Character of pseudo surface waves on anisotropic crystals. The Journal of the Acoustical Society of America 45, 845–851 (1969).   
23. Cobelli, P. J., Pagneux, V., Maurel, A. & Petitjeans, P. Experimental observation of trapped modes in a water wave channel. EPL (Europhysics Letters) 88, 20006 (2009).   
24. Lee, J., Zhen, B., Chua, S.-L., Shapira, O. & Soljačić, M. Fabricating centimeter-scale high quality factor two-dimensional periodic photonic crystal slabs. Optics express 22, 3724–3731 (2014).

25. Hughes, S., Ramunno, L., Young, J. F. & Sipe, J. E. Extrinsic optical scattering loss in photonic crystal waveguides: role of fabrication disorder and photon group velocity. Physical Review Letters 94, 033903 (2005).   
26. Zhou, H. et al. Observation of bulk Fermi arc and polarization half charge from paired exceptional points. Science eaap9859 (2018).   
27. Bulgakov, E. N. & Maksimov, D. N. Topological bound states in the continuum in arrays of dielectric spheres. Physical Review Letters 118, 267401 (2017).   
28. Bulgakov, E. N. & Maksimov, D. N. Bound states in the continuum and polarization singularities in periodic arrays of dielectric rods. Physical Review A 96, 063833 (2017).   
29. Iwahashi, S. et al. Higher-order vector beams produced by photonic-crystal lasers. Optics Express 19, 11963–11968 (2011).   
30. Kitamura, K., Sakai, K., Takayama, N., Nishimoto, M. & Noda, S. Focusing properties of vector vortex beams emitted by photonic-crystal lasers. Optics Letters 37, 2421–2423 (2012).   
31. Zhang, Y. et al. Observation of polarization vortices in momentum space. Physical Review Letters 120, 186103 (2018).   
32. Yuan, L. & Lu, Y. Y. Bound states in the continuum on periodic structures surrounded by strong resonances. Physical Review A 97, 043828 (2018).   
33. Bulgakov, E. N. & Maksimov, D. N. Bound states in the continuum and polarization singularities in periodic arrays of dielectric rods. Physical Review A 96, 063833 (2017).

34. Yuan, L. & Lu, Y. Y. Strong resonances on periodic arrays of cylinders and optical bistability with weak incident waves. Physical Review A 95, 023834 (2017).   
35. Liang, Y., Peng, C., Sakai, K., Iwahashi, S. & Noda, S. Three-dimensional coupled-wave analysis for square-lattice photonic crystal surface emitting lasers with transverse-electric polarization: finite-size effects. Optics Express 20, 15945–15961 (2012).   
36. Wang, Z. et al. Mode splitting in high-index-contrast grating with mini-scale finite size. Optics Letters 41, 3872–3875 (2016).   
37. Ni, L., Jin, J., Peng, C. & Li, Z. Analytical and statistical investigation on structural fluctuations induced radiation in photonic crystal slabs. Optics Express 25, 5580–5593 (2017).   
38. Regan, E. C. et al. Direct imaging of isofrequency contours in photonic structures. Science Advances 2, e1601591 (2016).   
39. Lu, L., Joannopoulos, J. D. & Soljačić, M. Topological photonics. Nature Photonics 8, 821–829 (2014).   
40. Ozawa, T. et al. Topological photonics. arXiv preprint arXiv:1802.04173 (2018).   
41. Wang, Z., Chong, Y., Joannopoulos, J. D. & Soljačić, M. Observation of unidirectional backscattering-immune topological electromagnetic states. Nature 461, 772–775 (2009).   
42. Luchansky, M. S. & Bailey, R. C. High-q optical sensors for chemical and biological analysis. Analytical chemistry 84, 793–821 (2011).

43. Di Falco, A., Ofaolain, L. & Krauss, T. Chemical sensing in slotted photonic crystal heterostructure cavities. Applied physics letters 94, 063503 (2009).   
44. Logan, A. D. et al. 400%/w second harmonic conversion efficiency in 14 $\mu$ m-diameter gallium phosphide-on-oxide resonators. arXiv preprint arXiv:1810.06393 (2018).   
45. Hirose, K. et al. Watt-class high-power, high-beam-quality photonic-crystal lasers. Nature photonics 8, 406 (2014).

Competing Interests The authors declare that they have no competing financial interests.

Correspondence Correspondence and requests for materials should be addressed to Chao Peng. (email: pengchao@pku.edu.cn).

a)   
![](images/118de12290b659b80497c36da47588bda6a17c0502f91068ddba81fa45eda7ad.jpg)

<details>
<summary>text_image</summary>

Radiation
Scattering
Absorption
Side-leakge
</details>

![](images/8ae71667ea70ced638a3728afcdf5769cd895100c4d21d26e8086cc61f277ba0.jpg)

<details>
<summary>line</summary>

| k(2π/a) | TE A  | TE C,D | TE B  |
|---------|-------|--------|-------|
| -0.2    | 0.32  | 0.38   | 0.42  |
| 0       | 0.36  | 0.40   | 0.44  |
| 0.2     | 0.32  | 0.39   | 0.41  |
</details>

b)   
![](images/facc5f37c8bc12c3245ff93587fec0f1151567dd2eb6a250c54b8e6e32f648fc.jpg)

<details>
<summary>heatmap</summary>

| a (nm) | k_x (2π/a) Range | k_y (2π/a) Range | Value (log scale) |
|--------|------------------|------------------|-------------------|
| 519.25 | -0.1 to 0.1      | -0.1 to 0.1      | ~10^7             |
| 531.42 | -0.1 to 0.1      | -0.1 to 0.1      | ~10^7             |
| 580    | -0.1 to 0.1      | -0.1 to 0.1      | ~10^7             |
</details>

c)   
![](images/8f20875a4af884875e90aa75ee01c48bf90da89a3507231e99cf6991c0182ff3.jpg)

<details>
<summary>line</summary>

| k (2π/a) | Merging (Q) | Isolated (Q) | 1/k² (Q) | 1/k⁶ (Q) |
| -------- | ------------ | ------------- | -------- | -------- |
| -0.15    | ~10⁴         | ~10⁴          | ~10⁴     | ~10⁴     |
| -0.1     | ~10⁵         | ~10⁵          | ~10⁵     | ~10⁵     |
| -0.05    | ~10⁷         | ~10⁷          | ~10⁷     | ~10⁷     |
| 0        | ∞            | ~10⁸          | ~10⁸     | ~10⁸     |
| 0.05     | ~10⁷         | ~10⁷          | ~10⁷     | ~10⁷     |
| 0.1      | ~10⁵         | ~10⁵          | ~10⁵     | ~10⁵     |
| 0.15     | ~10⁴         | ~10⁴          | ~10⁴     | ~10⁴     |
</details>

![](images/c98b89b2f733c3732b350d40dd1c31fceb626810656eaab30e4db4a7b5c3131c.jpg)

<details>
<summary>scatter</summary>

| k (2π/a) | Γ → X (log scale) | Γ → M (log scale) |
| -------- | ----------------- | ----------------- |
| 10⁻³     | ~10¹²             | ~10¹²             |
| 10⁻²     | ~10⁸              | ~10⁸              |
| 10⁻¹     | ~10⁴              | ~10⁴              |
</details>

Figure 1: | Suppressing radiation losses by merging multiple topological charges of bound states in the continuum (BICs).a, Schematic of the PhC slab. Band TE-A is marked with a red line. b, Multiple BICs appear on band TE-A, where the normalized radiative lifetime Q diverge. When sample periodicity a is tuned from 519.25nm to 580nm, 9 BICs with ±1 topological charges merge into an isolated one with charge +1. c, Plots of Q near the center of the BZ when charges just merge (red a=531.42nm) and long after they have merged (blue a=580nm). The merging sample (red) shows significantly higher Qs than the isolated sample due to a different scaling of $Q \propto 1/k^{6}$ instead of $Q \propto 1/k^{2}$ , which is observed along both $\Gamma - X$ and $\Gamma - M$ direction. Simulations here are using FEM.

a)   
![](images/eeb02b3c050331860a517151134f80ae038888a2d348114b6e5dbfd65c499894.jpg)

<details>
<summary>text_image</summary>

Real space
Δr
(Δx,Δy)
a
</details>

b)

![](images/d705763941c5252b864231d0b2b074d30167f6c949ed48d3e18af10a1e875812.jpg)

<details>
<summary>heatmap</summary>

| k_x (2π/a) | k_y (2π/a) | Value |
|------------|------------|-------|
| 0          | 0          | 1     |
</details>

c)   
![](images/853b2c7b8bd6ab9f7e3ef2e666866b15a6b651ee3bf4523d3cab4aaa80c1963f.jpg)

<details>
<summary>heatmap</summary>

| k_y (2π/a) | -0.5 | 0.0 | 0.5 |
|------------|------|-----|-----|
| 0          | 1e-7 | 1e-3| 1e-3|
| 0          | 1e-6 | 1e-3| 1e-3|
| 0          | 1e-5 | 1e-3| 1e-3|
| 0          | 1e-4 | 1e-3| 1e-3|
| 0          | 1e-3 | 1e-3| 1e-3|
| 0          | 1e-2 | 1e-3| 1e-3|
| 0          | 1e-1 | 1e-3| 1e-3|
| 0          | 0    | 1e-3| 1e-3|
| 0          | -0.1 | 1e-3| 1e-3|
| 0          | -0.2 | 1e-3| 1e-3|
| 0          | -0.3 | 1e-3| 1e-3|
| 0          | -0.4 | 1e-3| 1e-3|
| 0          | -0.5 | 1e-3| 1e-3|
The image displays a color-coded contour plot with a color bar ranging from 1e-7 to 1e-3. The label 'Merged BIC' appears in the top-left corner. There are no explicit numerical data provided in the code.
</details>

d)

![](images/00a00d3601abdcf1b713d0a0318c4598b647d9124b3263ea2c613548fdde4fc9.jpg)

<details>
<summary>text_image</summary>

r+\u03c6r
O
(\u03bax,\u03bay)
θ
r
O
</details>

![](images/d7a8ec8aa83ad1bc24975fe1ed18124c92af032a0624d686a14e86cbe09f6e33.jpg)

<details>
<summary>text_image</summary>

Reciprocal space
First Brillouin Zone
βN=2π/Na
</details>

![](images/ebef2c068c7cf1737d095cfd82a0c89cae9d190d2059a57795b9468d4632691a.jpg)

<details>
<summary>heatmap</summary>

| k_x (2π/a) \ k_y (2π/a) | -0.5 | 0 | 0.5 |
| --- | --- | --- | --- |
| 0.5 | 1e-6 | 1e-2 | 1e-6 |
| 0 | 1e-4 | 1e-2 | 1e-6 |
| 0 | 1e-2 | 1e-2 | 1e-6 |
| -0.5 | 1e-6 | 1e-2 | 1e-6 |
</details>

![](images/eaa1cd5d0e7c55b0548b0ff340a9f8ccd478d55bafd7c214cf94e5739d92450a.jpg)

<details>
<summary>heatmap</summary>

| k_x (2π/a) \ k_y (2π/a) | -0.5 | 0 | 0.5 |
| ------------------------ | ---- | --- | --- |
| 0                        | 1e-7 | 1e-6 | 1e-5 |
| 0                        | 1e-6 | 1e-5 | 1e-4 |
| 0                        | 1e-5 | 1e-4 | 1e-3 |
| 0                        | 1e-4 | 1e-3 | 1e-2 |
| 0                        | 1e-3 | 1e-2 | 1e-1 |
| 0                        | 1e-2 | 1e-1 | 1e0   |
| 0                        | 1e-1 | 1e0   | 1e1   |
| 0                        | 0      | 1e1   | 1e2   |
| 0                        | 0      | 1e2   | 1e3   |
| 0                        | 0      | 1e3   | 1e4   |
| 0                        | 0      | 1e4   | 1e5   |
| 0                        | 0      | 1e5   | 1e6   |
| 0                        | 0      | 1e6   | 1e7   |
| 0                        | 0      | 1e7   | 1e8   |
| 0                        | 0      | 1e8   | 1e9   |
| 0                        | 0      | 1e9   | 1e10  |
| 0                        | 0      | 1e10  | 1e11  |
| 0                        | 0      | 1e11  | 1e12  |
| 0                        | 0      | 1e12  | 1e13  |
| 0                        | 0      | 1e13  | 1e14  |
| 0                        | 0      | 1e14  | 1e15  |
| 0                        | 0      | 1e15  | 1e16  |
| 0                        | 0      | 1e16  | 1e17  |
| 0                        | 0      | 1e17  | 1e18  |
| 0                        | 0      | 1e18  | 1e19  |
| 0                        | 0      | 1e19  | 1e20  |
| -0.5                     |    |    |     |
| -0.5                     |    |    |     |
| -0.5                     |    |    |     |
| -0.5                     |    |    |     |
| -0.5                     |    |    |     |
| -0.5                     |    |    |     |
| -0.5                     |    |    |     |
| -0.5                     |    |    |     |
| -0.5                     | *    |    |     |
| -0.5                     | *    |    |     |
| -0.5                     | *    |    |     |
| -0.5                     | *    |    |     |
| -0.5                     | *    |    |     |
| -0.5                     | *    |    |     |
| -0.5                     | *    |    |     |
| -0.5                     | *    / *|    |     |
| -0.5                     | *    / *|    |     |
| -0.5                     | *    / *|    |     |
| -0.5                     | *    / *|    |     |
| -0.5                     | *    / *|    |     |
| -0.5                     | *    / *|    |     |
| -0.5                     | *    / *|    |     |
</details>

![](images/667b508f2f6be396476f0b239fc765f6ac4d41f01eda78004d67aa0688c0ac0c.jpg)

<details>
<summary>line</summary>

| k_x (2π/a) | Merging (530nm) | Merging (532nm) | Merging (580nm) | Isolated (530nm) | Isolated (532nm) | Isolated (580nm) |
| ---------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 0.0        | 10^6             | 10^6             | 10^5             | 10^4             | 10^4             | 10^3             |
| 0.02       | ~10^6            | ~10^6            | ~10^4.5          | ~10^3.5          | ~10^3.5          | ~10^3            |
| 0.04       | ~10^6            | ~10^6            | ~10^4           | ~10^3             | ~10^3             | ~10^3            |
| 0.06       | ~10^5.5          | ~10^5.5          | ~10^3.5          | ~10^2.5          | ~10^2.5          | ~10^2.5          |
| 0.08       | ~10^5            | ~10^5            | ~10^3           | ~10^2             | ~10^2             | ~10^2            |
| 0.1        | ~10^4.5          | ~10^4.5          | ~10^2.5          | ~10^1.5          | ~10^1.5          | ~10^1.5          |
</details>

Figure 2: | Robustness to scattering losses due to topological protection. a, Schematic of a fabricated PhC sample (solid lines) with disorder in hole locations and radii compared to a perfect one (dashed lines). Fractional orders of momentum (green dots) are introduced by the super-cell. b, Energy distribution of the highest-Q mode on TE-A band in the momentum space of a merging-BIC design in a perfect (upper) and a disordered structure (lower) inside the first BZ. c, Momentum energy distribution of the far-field radiation in a disordered sample with merging BICs (upper) and one with an isolated BIC (lower). The while circles represent the light-cone. The radiative scattering loss is significantly lower in the merging sample than the isolated one. Simulations are performed in $15 \times 15$ super-cell using FEM. d, Schematic of typical fabrication error in asymmetric hole (upper) and plots of $Q$ near the BZ center with disorders accordingly (lower).

a)   
![](images/2d21a14ab5618d131f8db52839ed410feb6a68f689234f1a317823b1f4dccee1.jpg)  
c)

![](images/34961173f82ca5994034edbc436c432534762a6379ecec427c3a5cd7054a95bf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    Laser --> Pol_X["Pol X"]
    Pol_X --> X["Y"]
    X --> Y["Y"]
    Y --> L2["L2"]
    L2 --> BS["BS"]
    BS --> Rar["Focal Plane"]
    Rar --> Obj["Obj"]
    Obj --> Sample
    L3["L3"] --> Camera/PD["Camera/PD"]
    style Laser fill:#f9f,stroke:#333
    style Camera/PD fill:#ccc,stroke:#333
    style Rar fill:#fff,stroke:#333
    style Obj fill:#fff,stroke:#333
    style BS fill:#fff,stroke:#333
    style X fill:#fff,stroke:#333
    style Y fill:#fff,stroke:#333
    style_L1["L1"] fill:#fff,stroke:#333
    style Pol_X fill:#fff,stroke:#333
    style X fill:#fff,stroke:#333
    style Y fill:#fff,stroke:#333
    style L2 fill:#fff,stroke:#333
    style Pol_Y fill:#fff,stroke:#333
    style X fill:#fff,stroke:#333
    style Y fill:#fff,stroke:#333
    style L1 fill:#fff,stroke:#333
    style Pol_X fill:#fff,stroke:#333
    style X fill:#fff,stroke:#333
    style Y fill:#fff,stroke:#333
```
</details>

b)   
![](images/d48e46e3f2528f195252a469f1685344955255db6a07b58f26801b71321278c3.jpg)  
Figure 3: | Experimental setup. a,b Scanning electron microscope (SEM) images of the fabricated PhC sample from the top and cross-section view. The chosen structural parameters correspond to when the 9 BICs just merge in middle panel of Fig. 1b. The underlying $\mathrm{SiO}_2$ layer is later removed for measurements. c, Schematic of the measurement setup. The blue lines refer to the incident light and its direct reflection. The red region refers to radiation losses induced by scattering from disorder. L, lens; Obj, objective; PD, photodiode; Pol, polarizer.

a)   
![](images/b910855f527882974e29441490a1cb5c5fb9ecb1ac70805e7c01295f3223a4fe.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Experiment | Fitting |
| --------------- | ---------- | ------- |
| 1568.31         | 0.0        | 0.0     |
| 1568.32         | 0.0        | 0.0     |
| 1568.33         | 0.0        | 0.0     |
| 1568.34         | 0.0        | 0.0     |
| 1568.35         | 0.0        | 0.0     |
| 1568.36         | 0.0        | 0.0     |
</details>

Iso-frequency contour   
![](images/dceb39be39c89f6e1e415e4a3a3ce0e0dffa41c5c4f0f6fc0984e389b5da0049.jpg)

<details>
<summary>text_image</summary>

X Y Z W
λ=1568.334nm
λ=1569.273nm
λ=1572.558nm
</details>

b)   
![](images/8f8bbf79bf61c4bcfa4563caf461ba554128a3e765f3803a0f1f58b973670ba9.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Intensity(a.u.) |
| --------------- | --------------- |
| 1568.3          | ~0.0            |
| 1568.34         | 1.0             |
| 1568.38         | ~0.0            |
</details>

c)   
![](images/609ed47e9390c5c7e4ea372c3b6c8f9e201ea000abb14dbfbda2592ab3b1f441.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Isolated |
| --------------- | -------- |
| 1600.16         | 0.0      |
| 1600.17         | 0.2      |
| 1600.18         | 0.4      |
| 1600.19         | 0.6      |
| 1600.20         | 0.8      |
| 1600.21         | 1.0      |
| 1600.22         | 0.8      |
| 1600.23         | 0.6      |
| 1600.24         | 0.4      |
| 1600.25         | 0.2      |
| 1600.26         | 0.0      |
</details>

Figure 4: | Experimental results. a, Iso-frequency contours of the sample at different wavelengths are observed on the camera. Three examples at 1572.558 nm, 1569.273 nm, and 1568.334 nm are shown as dashed lines. Scattered light intensity at different points in the momentum space (X,Y,Z) are further characterized by a PD, which all exhibit symmetric Lorentzian functions as the incident wavelength. The linewidth is determined by the Q of the underlying resonance. b, The highest Q observed in the merging-BIC sample is $4.9 \times 10^{5}$ at point W, which is over an order of magnitude than the isolated-BIC sample fabricated under the same processes ( $Q = 4.0 \times 10^{4}$ as shown in c).

![](images/eb69e1357669fb465e7a4d59932f9fa9a2f08ee4b599c4524d89602db9413b24.jpg)  
Figure 5: | Twelve-times enhancement of quality factors via topological protection. a, The dispersion of resonances are measured at different points in the momentum space (circles), which show good agreements with simulation predictions with FEM (dashed lines) both along $\Gamma - X$ (upper panel) and $\Gamma - M$ directions (lower panel). b, Over ten-fold enhancement of $Q$ is observed over a wide range in the momentum space in the merging-BIC samples (red and blue) compared to the isolated-BIC sample (purple) due to topological protection.