![](images/17b38d3111e55bd82cc14a90857823996a45806562442489c9d8f58000bf8bd9.jpg)

Check for updates

![](images/dde20e1685524c409fe9c51fda53b0ac4590bcb945ffebed434877c4cb352ad8.jpg)

# OPEN ACCESS

Citation: Kim J-H, Kim J-H, Lee S-H, Park J, Lee S-K (2019) Fabrication of a spherical inclusion phantom for validation of magnetic resonance-based magnetic susceptibility imaging. PLoS ONE 14(8): e0220639. https://doi.org/10.1371/journal.pone.0220639

Editor: Cem M. Deniz, New York University School of Medicine, UNITED STATES

Received: March 7, 2019

Accepted: July 19, 2019

Published: August 5, 2019

Copyright: © 2019 Kim et al. This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

Data Availability Statement: All relevant data are within the paper and its Supporting Information files.

Funding: Jun-Ho Kim, Jung-Hyun Kim, S-H Lee, S-K Lee were funded by the Institute for Basic Science (https://ibs.re.kr) of the government of South Korea, under grant IBS-R015-D1. The funder had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

RESEARCH ARTICLE

# Fabrication of a spherical inclusion phantom for validation of magnetic resonance-based magnetic susceptibility imaging

Jun-Ho Kim $^{1,2}$ , Jung-Hyun Kim $^{1,2}$ , So-Hee Lee $^{1,2}$ , Jinhyoung Park $^{1*}$ , Seung-Kyun Lee $^{1,2*}$

1 Department of Biomedical Engineering, Sungkyunkwan University, Suwon, South Korea, 2 IBS Center for Neuroscience Imaging Research, Suwon, South Korea

\* lee.seungkyun@gmail.com(SKL); jin.park@skku.edu (JP)

# Abstract

Fabrication of a spherical multi-compartment MRI phantom is demonstrated that can be used to validate magnetic resonance (MR)-based susceptibility imaging reconstruction. The phantom consists of a 10 cm diameter gelatin sphere that encloses multiple smaller gelatin spheres doped with different concentrations of paramagnetic contrast agents. Compared to previous multi-compartment phantoms with cylindrical geometry, the phantom provides the following benefits: (1) no compartmental barrier materials are used that can introduce signal voids and spurious phase; (2) compartmental geometry is reproducible; (3) spherical susceptibility boundaries possess a ground-truth analytical phase solution for easy experimental validation; (4) spherical geometry of the overall phantom eliminates background phase due to air-phantom boundary in any scan orientation. The susceptibility of individual compartments can be controlled independently by doping. During fabrication, formalin cross-linking and water-proof surface coating effectively blocked water diffusion between the compartments to preserve the phantom's integrity. The spherical shapes were realized by molding the inner gel compartments in acrylic spherical shells, 3 cm in diameter, and constructing the whole phantom inside a larger acrylic shell. From gradient echo images obtained at 3T, we verified that the phantom produced phase images in agreement with the theoretical prediction. Factors that limit the agreement include: air bubbles trapped at the gel interfaces, imperfect magnet shimming, and the susceptibility of external materials such as the phantom support hardware. The phantom images were used to validate publicly available codes for quantitative susceptibility mapping. We believe that the proposed phantom can provide a useful testbed for validation of MR phase imaging and MR-based magnetic susceptibility reconstruction.

# 1. Introduction

Static magnetic susceptibility is an important magnetic resonance imaging (MRI) biomarker that carries information about iron deposit in tissue as well as myelin density and hemorrhage

Competing interests: The authors have declared that no competing interests exist.

[1-3]. In MR-based magnetic susceptibility imaging, gradient echo phase images are converted into static magnetic susceptibility maps of the imaged object through dedicated reconstruction algorithms. This process is known as quantitative susceptibility mapping (QSM). While many reconstruction algorithms have been developed and successfully deployed in clinical settings, many still suffer from reconstruction artifacts and slow computation [4]. In order to validate different susceptibility reconstruction algorithms, geometric phantoms have been widely used [5-7]. In QSM, the dipolar inversion process can only determine relative (difference) susceptibilities; therefore, a validation phantom must contain at least two compartments with different susceptibilities. In addition, the following properties are desirable:

i. In order to mimic tissue in an organ, compartments should be nested.

ii. Non-tissue-mimicking materials, such as air bubbles and structural barriers for compartmental separation, can cause phase errors and should be minimized.

iii. Compartmental shapes with known analytical dipolar field solutions are beneficial for validation of phase measurement, a pre-requisite of susceptibility reconstruction.

Literature survey indicates that thin cylindrical or elongated ellipsoidal shapes embedded in a larger cylindrical container have been frequently used for validation phantoms $[5, 7]$ . In order to minimize the compartmental barrier material, some authors have used thin plastic or latex balloons to contain the inner-compartment materials $[7–9]$ . Other authors $[10]$ have constructed a gel inclusion phantom where the inner materials were cut from thin cylindrical agarose rods and positioned inside a background gel in a semi-random fashion; in this case no structural barrier was used but the orientation of the cut pieces was uncontrolled.

While these previous phantoms have served their purposes in their respective experiments, in our opinion their designs were not conducive to good control of the compartmental geometry, necessary for reproducible research. Furthermore, predominantly cylindrical or elongated geometry chosen for the inner compartments could limit experiments to imaging on primarily transverse planes with respect to the compartments' long axis. This is because phase images in the longitudinal planes are sensitive to air bubbles/pockets that are prone to form at the top of the elongated compartments. Since the susceptibility-induced phase maps are strongly orientation dependent, a phantom that can be imaged in any scan plane is desirable for volumetric susceptibility imaging validation.

With these points in mind, here we demonstrate fabrication of a spherical phantom with spherical inner compartments ("inclusions"), in which a 10 cm-diameter spherical gelatin body encloses smaller spherical gelatin balls at controlled locations without using foreign barrier or support materials. The inner sphere susceptibility can be controlled independently from the enclosing sphere by doping with paramagnetic contrast agents. The spherical shape of the overall phantom served to eliminate bulk background phase caused by the air-phantom boundary, whereas the spherical inner compartments produced 3D phase variation according to a known analytical solution. These beneficial features hold regardless of the phantom's physical orientation in the magnet bore or the scan plane choice. We demonstrate use of our phantom to validate publicly available QSM reconstruction software.

# 2. Theory

The effect of a spherical susceptibility boundary on MRI phase is well established. Here we review the basic theory to motivate the present research.

When a uniform sphere with magnetic susceptibility $\chi$ is placed in an applied magnetic field $B_{\mathrm{app}}$ in vacuum, the sphere develops magnetization $\mu_0M = \left(\frac{\chi}{N\chi + 1}\right)B_{\mathrm{app}}$ , where $N = 1/3$ is

the demagnetization factor of a sphere [1]. For biological tissue with $|\chi|$ on the order of $10^{-5}$ , this equals for all practical purposes $\mu_{0}M = \chi B_{app}$ . The magnetization generates a macroscopic magnetic field $\delta B_{sphere} = \frac{2}{3}\mu_{0}M$ inside the sphere that adds to the applied field $B_{app}$ . Importantly, in MRI the proton Larmor frequency $\omega_{0} = \gamma B_{0}$ ( $\gamma$ is the gyromagnetic ratio) inside a magnetized medium is determined not by the total macroscopic magnetic field, but by the magnetic field $B_{0}$ that would exist if a small hypothetical sphere called "Lorentz sphere" is carved out of the medium [11]. This is to exclude the molecular self-field from the frequency shift calculation and has been experimentally verified. Since $\delta B_{sphere}$ is independent of the radius of the sphere, carving out a Lorentz sphere precisely takes away $\frac{2}{3}\mu_{0}M$ from the field shift, leaving 0 for the shift induced by a uniformly magnetized sphere:

$$
\delta B _ {0} (\text { inside   spherical   medium }) = \delta B _ {\text { sphere }} - \delta B _ {\text { Lorentz   sphere }} = 0. \tag {1}
$$

This means that the gradient echo image phase inside the sphere vanishes:

$\phi_{sphere}^{in} = \gamma \cdot \mathrm{TE} \cdot \delta B_0 = 0$ , where $\phi$ is the image phase minus any radio-frequency (RF) field-related phase, and TE is the echo time.

A uniformly magnetized sphere also changes the static field outside the sphere. The change is given by the magnetic field of a point dipole located at the center of the sphere, with matched total magnetic moment. Only the component of the field in the direction of $B_{app}$ measurably changes the Larmor frequency. The phase due to this field is given by

$$
\phi_ {s p h e r e} ^ {o u t} = \gamma \cdot T E \cdot \frac {\mu_ {0} m}{4 \pi} \cdot \frac {(3 \cos^ {2} \theta - 1)}{r ^ {3}}. \tag {2}
$$

Here $m = M \cdot V_{sphere}$ is the magnetic moment of the sphere (assumed to be parallel to $B_{app}$ as is the case for induced moments) with volume $V_{sphere}$ , and r, $\theta$ are the polar coordinates of the measurement point with respect to the center of the sphere, with $B_{app}$ defining the z axis.

A nested sphere where a small homogeneous sphere of susceptibility $\chi_{2}$ is embedded in a larger sphere with susceptibility $\chi_{1}$ can be thought of as a superposition of the large whole sphere and a small sphere with susceptibility $\chi_{2}-\chi_{1}$ . Applying the Lorentz sphere argument to both spheres, we conclude that inside the small sphere, the MRI phase shift is zero. In the space outside the small but inside the large sphere, the phase shift is given by Eq (2) where m is the magnetic moment of the small sphere with susceptibility $\chi_{2}-\chi_{1}$ . The phase shift for nested spheres is illustrated in Fig 1.

# 3. Materials and methods

# 3.1. Phantom fabrication

All the phantom compartments were made out of aqueous solution of gelatin powder (G1890, Sigma Aldrich, St. Louis, Mo, USA). In order to minimize water diffusion between the gel boundaries, the gel solution was mixed with 10% neutral buffered formalin solution which induced crosslinking of the polymers to raise the melting point and reduce the gel porosity  [12] . The spherical shape of the whole phantom, 10 cm in diameter, was achieved by building the phantom inside a thin acrylic shell assembled from two hemispheric shells epoxied or pressure-fit together. The spherical inclusions were made by injecting the aqueous solution into a 3 cm-diameter acrylic spherical shell and separating the shell after the solution gelled. Furthermore, a waterproof coating agent (MP131, 3M Company, Maplewood, MN, USA) was applied on the outer surface of the 3 cm inclusions prior to inserting them into the enclosing gel.

Detailed steps for phantom fabrication are presented in the following. Fig 2 shows the fabrication workflow, and Table 1 lists the materials used. Among the materials in Table 1, only

![](images/e23d2fa40b0b7fa0a52bdc3fd020c1a4d7f8ecf9508853085e89356a49bd163a.jpg)  
Fig 1. Phase shift in nested spheres. (a) Nested sphere as the superposition of two homogeneous spheres. (b) Corresponding phase shift maps.

https://doi.org/10.1371/journal.pone.0220639.g001

formalin solution is toxic and requires personal protective equipment for routine laboratory handling.

Step 1. Lower layer of the outer sphere

A gelatin solution was prepared by adding 23.4 g of gelatin powder to 300 ml de-ionized water at room temperature and bringing the solution to boil on a hot plate. The solution was then cooled down to 45°C in approximately 60 minutes. At this point the typical amount of solution was 250 ml, to which we added 4.5 ml of liquid Germall Plus (Ashland, Covington, KY, USA) as a non-toxic preservative [12]. In our experience, the preservative-treated gelatin phantom remained mold-free for at least 6 months when kept in a refrigerator. Its magnetic properties over time are discussed in Section 4.4. Shortly after adding the preservative, 2.5 ml of formalin solution (10% neutral buffered, Sigma Aldrich) was mixed with the gelatin solution. The prepared solution, at about 40°C, was poured into a 10 cm diameter acrylic hemispherical shell to a depth of 4 cm, and was cooled down in a refrigerator overnight in an upright position.

Step 2. Defining the inclusion locations

After the 4 cm-deep gel base was congealed, small (1.5 cm diameter, 0.5 cm deep) indentations were made by scooping out the gel with a spoon at locations where the inclusion spheres will be embedded. The shape of the indentations was not precisely controlled; any physical gap between the base gel and the inner spheres due to curvature mismatch can eventually be filled by the gel solution poured at a later step. Our process was effective to embed up to four 3 cm-diameter inclusion spheres on one horizontal plane. Below we describe the process for making a phantom with 3 inclusion spheres.

Step 3. Inner spheres

Three small gelatin spheres were separately made by preparing molten gelatin solution, treated with the preservative and mixed with formalin, as described in Step 1. To control the magnetic susceptibility of each sphere, varying concentrations of paramagnetic contrast agents were added to the solution before cooling down in a refrigerator. In our experiment we have used iron oxide nanoparticles (Molday ION, Biopal, Worcester, MA, USA) at the iron molar density of 0.043 to 0.129 mM, or Gd chelation agents (Dotarem, Guerbet, Villepinte, France) at 2.4 to 7.2 mM. Below we will only describe experiments with the iron oxide particles. The doped gelatin solution was injected by a syringe into a 3 cm-diameter acrylic spherical shell through a hole ( $\sim$ 1 mm diameter) at the top, at about 38°C. Each sphere was then immersed

(a)

1. Prepare base of the outer spherical gel

2. Make indentation on the base gel at locations of inclusion spheres

3. Make inclusion gel spheres using $3\mathrm{cm}$ diameter acrylic shells as molds

4. Put inclusion gels on the base gel and close the outer acrylic shell

5. Pour molten gel solution into the outer shell via a top hole.

6. Cool down the gel and seal the hole with hot glue.

(b)   
![](images/8e598eb83adfb1d187ad1e4dfdd1571aaf04d0a54ab3fe1270bf020c4948846d.jpg)

<details>
<summary>natural_image</summary>

Circular petri dish with a clear liquid, showing concentric rings and a central cavity (no text or symbols visible)
</details>

![](images/bafee84d3f531a8bf141d9823abfd3b2175cdbf6a5f6e6304f417db504a9c538.jpg)

<details>
<summary>natural_image</summary>

Petri dish containing a culture medium with small bubbles, labeled '2' in corner (no text or symbols on dish itself)
</details>

![](images/e13772f5272abb0cab2cff2f9eb4f3d2205e9533cb7afcf6fbc7d1636c9ee31f.jpg)

<details>
<summary>natural_image</summary>

Three transparent spherical objects on a metallic dish, labeled '3' in the corner (no text or symbols on the objects themselves)
</details>

![](images/635c91328ca67701089e5664da9cb087f7689a52b6af9bad4ee788b8d2a44580.jpg)

<details>
<summary>natural_image</summary>

Petri dish containing four translucent spherical objects with clear liquid, labeled '4' in corner (no text or symbols on objects)
</details>

![](images/eff4dc1a35b0c54bc71f53181d6e6a235dbdb430402d75781196c73182d1d4f2.jpg)

<details>
<summary>natural_image</summary>

Laboratory setup with gloved hand holding a glass dome and pipette above a lab bench (no visible text or symbols)
</details>

![](images/76020478887789adc8b7bd878e0f63c9e3a95642cb3a435eb0d1ead19a60f4e3.jpg)

<details>
<summary>natural_image</summary>

Close-up of a gloved hand holding a pipette over a dome-shaped mold on a metallic base (no visible text or symbols)
</details>

Fig 2. Fabrication workflow. (a) Steps for the fabrication of a spherical inclusion phantom. (b) Pictures of each step. 3D-printed, bowl-shaped holder (grey) had a hole in the middle which is visible in Steps 1, 2, 4.

https://doi.org/10.1371/journal.pone.0220639.g002

in a 50 ml beaker filled with the same gelatin solution to prevent air bubble formation at the top of the sphere. The entire beaker was cooled down in a refrigerator for at least 24 hours; a longer congealing time than for the base gel was needed to ensure sufficient solidification of the gel inside the spherical shell before breaking the mold. The two hemispheres of the shell, initially pressure-fitted, could be easily separated, and gelatin spheres could be taken out. The prepared naked gelatin spheres were hand-coated on the surface with a waterproofing liquid (MP131, 3M, Maplewood, MN, USA), and dried for 5 min at room temperature. Before transferring the spheres on to the base gel, the indentations for the placement of the spheres were "flooded" with a shallow layer of the original gelatin solution ( $\sim$ 38°C). The small spheres were then placed on the indentations, minimizing air bubbles at the bottom. Note that contact with

Table 1. Materials used for the fabrication of a 10 cm-diameter gelatin sphere phantom with three inclusions. 

<table><tr><td>Item</td><td>Amount used</td><td>Model/product description</td><td>Manufacturer</td><td>Vendor/local distributor URLa</td></tr><tr><td>Gelatin</td><td>75 g</td><td>G1890</td><td>Sigma-Aldrich, St Louis, MO, USA</td><td>https://www.sigmaaldrich.com/</td></tr><tr><td>Preservative</td><td>15 ml</td><td>Germall Plus</td><td>Ashland, Covington, KY, USA</td><td>www.tonature.co.kr</td></tr><tr><td>Formalin solution</td><td>8 ml</td><td>neutral buffered, 10%</td><td>Sigma-Aldrich, St Louis, MO, USA</td><td>https://www.sigmaaldrich.com/</td></tr><tr><td>Acrylic shell</td><td>1 EA (10cm), 3 EA (3 cm)</td><td>KUG100, KUG030</td><td>Schiller-Plastic, Rechberghausen, Germany</td><td>http://www.dnara.kr</td></tr><tr><td>Waterproof coating agent</td><td>~ 5 ml</td><td>MP131 (liquid)</td><td>3M Company, Maplewood, MN, USA</td><td>https://front.wemakeprice.com/product/179830481?utm_source= google_ss&amp;utm_medium=cpc&amp;utm_campaign=r_sa</td></tr><tr><td>Epoxy</td><td>~ 2 ml</td><td>MagicFix two-part epoxy</td><td>PC-Products Co, Allentown, PA, USA</td><td>http://item.gmarket.co.kr/Item?goodscode = 1139284242</td></tr><tr><td>Iron oxide contrast agent</td><td>0.12 ml</td><td>Molday ION, CL-30Q02-2B (30 mg Fe/ml)</td><td>BioPAL, Worcester, MA, USA</td><td>http://www.biopal.com/molday-ion.htm</td></tr><tr><td>Gd contrast agent</td><td>~ 10 ml, as needed</td><td>Dotarem (0.5 M)</td><td>Guerbet, Villepinte, France</td><td>http://www.guerbet.co.kr/fileadmin/user_upload/korea/Contrast Media/232502_Dotarem_Insert_final_20180321.pdf</td></tr></table>

a. All URLs were accessed 06/08/2019.

https://doi.org/10.1371/journal.pone.0220639.t001

warm gelatin solution did not melt the congealed gel compartments because of the formalin cross-linking.

Step 4. Upper shell of the outer sphere

At this point the lower part of the outer gel is contained in the lower hemispherical acrylic shell, supporting three inclusion spheres with different magnetic susceptibilities. Next we fitted the upper hemispherical shell onto the lower shell to complete the spherical enclosure. The upper shell, empty at this point, had a drilled hole ( $\sim$ 1 mm diameter) at the top for gelatin solution injection in the space between the shell and the three small spheres protruding up from the base gel. In most cases, pressure-fitting the two shells at the equatorial joint was sufficient to create leak-proof enclosure. However, certain batches of the acrylic shells required epoxying at the joint (presumably due to manufacturing variabilities) to prevent leaking of the solution in the next step.

Step 5. Gelatin solution fill-in

To fill in the spherical enclosure, a molten gelatin solution with the same composition as the gel base in the lower hemisphere, was syringe-injected into the sphere through the top hole, at the temperature of $\sim38^{\circ}C$ . As in Step 3, formalin fixing ensured that the molten solution did not damage the integrity of the gel base and the inclusions while being poured in the enclosure.

(a)   
![](images/d58ba1901f5ee9af60efea86fd2d7acfd1013bad86d9a28bcce6825ddfc63df3.jpg)

<details>
<summary>natural_image</summary>

Close-up of a translucent, round, translucent biological sample with a central white structure (no text or symbols visible)
</details>

Top view

(b)   
![](images/2e0909f23e98a943ebdff5819cab258331a8dfd1ba3c17ea2cd78020349603c0.jpg)

<details>
<summary>natural_image</summary>

Transparent spherical object with translucent interior, possibly a transparent material or gel (no text or symbols visible)
</details>

Side view

(c)   
![](images/a16383ae3cb67f7035c122b5bab8c2c46cc5e0153646d4ac3eed98ac4b6b873d.jpg)

<details>
<summary>natural_image</summary>

Close-up of a white medical device with a golden dome-shaped component mounted on its side (no visible text or symbols)
</details>

Fig 3. Pictures of the phantom. (a,b) Pictures of the completed spherical phantom. The white blob at the top of the phantom is settled hot glue. (c) Phantom positioned for imaging in a head-neck array coil.

https://doi.org/10.1371/journal.pone.0220639.g003

# Step 6. Sealing the top

After the whole gel phantom cooled down in a refrigerator for $\sim$ 12 hours, any void at the top of the sphere was filled as much as possible by adding more gelatin solution and cooling down. Finally, the top hole was plugged with a hot glue. Fig 3 shows the completed phantom.

# 3.2. MRI scan

All MRI scans were performed in a clinical 3T scanner (Magnetom Prisma, Siemens, Erlangen, Germany) with a standard 20 channel head-neck coil. The phantom was placed inside the coil on a roll of masking tape used as a stand. The scanner had a full 2nd order shim coil set which was used to shim the static field before each scan. The actual main magnetic field strength was $B_{app} = 2.895$ T. Three dimensional multi-echo gradient echo images were obtained with the following scan parameters: Repetition time (TR) = 47 ms, echo times (TE) = 7, 12, 17, 22, 27, 32, 37, 42 ms, flip angle = 20°, bandwidth = 240 Hz/pixel, voxel size = 0.66 × 0.66 × 0.8 mm $^{3}$ , matrix size = 176 × 256 × 144, in-plane field of view = 116 × 170 mm $^{2}$ , scan time = 12 min 24 sec. Images from different coil channels were complex-combined off-line after subtracting each coil's offset phase determined by extrapolating the multi-echo phases to zero echo time.

In addition, the longitudinal relaxation rate $(R_{1})$ , transverse relaxation rate $(R_{2})$ , and effective transverse relaxation rate $(R_{2}^{*})$ of the gelatin inclusions were determined by the scanner's standard sequences (Table 2). Specifically, $R_{1}$ was obtained by scanning the phantom multiple times at different repetition times up to 1024 ms with a 90° flip angle in a spoiled gradient echo sequence. $R_{2}^{*}$ was obtained by a series of gradient echo scans with a range of echo times (TE = 7 to 60 ms) and a long TR = 2000 ms (approximately 5 times the longitudinal relaxation time of gelatin). $R_{2}$ was obtained similarly as $R_{2}^{*}$ but with a spin echo sequence with TE = 15 to 240 ms. The relaxation rate scans were obtained on a single coronal plane with pixel size = 0.78 × 0.78 mm $^{2}$ , matrix size = 128 × 128, field of view = 100 × 100 mm $^{2}$ . The relaxation rates were obtained by exponential fitting of the magnitude data as a function of TR or TE on a circular ROI of diameter 25 mm covering each inclusion.

# 3.3. Quantitative susceptibility mapping

Coil-combined multi-echo phase and magnitude images were processed through two publicly available, Matlab (Mathworks, Natick, MA, USA) code packages STISuite v3.0.2

Table 2. Scan parameters for volumetric phase imaging and single-plane relaxation parameter mapping. 

<table><tr><td>Parameters</td><td>Multi-echo GRE (for phase imaging, QSM)</td><td>Spoiled GRE (for R1)</td><td>Spoiled GRE (for R2*)</td><td>Spin echo (for R2)</td></tr><tr><td>TR</td><td>47 ms</td><td>16,32,64,128,256, 512,1024 ms</td><td>2000 ms</td><td>2000 ms</td></tr><tr><td>TE</td><td>7,12,17,22,27,32,37, 42 ms</td><td>10 ms</td><td>7,12,17,22,27,32,37, 42,47,52,57,60 ms</td><td>15,30,60,120,240 ms</td></tr><tr><td>flip angle</td><td>20 deg</td><td>90 deg</td><td>50 deg</td><td>90 deg</td></tr><tr><td>slice thickness</td><td>0.8 mm</td><td>1.5 mm</td><td>2.0 mm</td><td>2.0 mm</td></tr><tr><td>voxel size</td><td>0.66 x 0.66 x 0.8 mm3</td><td>0.78 x 0.78 x 1.5 mm3</td><td>0.78 x 0.78 x 2.0 mm3</td><td>0.78 x 0.78 x 2.0 mm3</td></tr><tr><td>matrix</td><td>176 x 256 x 144</td><td>128 x 128</td><td>128 x 128</td><td>128 x 128</td></tr><tr><td>FOV in plane</td><td>116 x 170 mm2</td><td>100 x 100 mm2</td><td>100 x 100 mm2</td><td>100 x 100 mm2</td></tr><tr><td>acquisition type</td><td>3D</td><td>2D</td><td>2D</td><td>2D</td></tr><tr><td>bandwidth</td><td>240 Hz/px</td><td>260 Hz/px</td><td>260 Hz/px</td><td>130 Hz/px</td></tr><tr><td>acceleration</td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>scan time</td><td>12 min 24 sec</td><td>2 sec ~ 129 sec</td><td>4 min 14 sec (per TE)</td><td>4 min 14 sec (per TE)</td></tr><tr><td>scan plane</td><td>transverse</td><td>coronal</td><td>coronal</td><td>coronal</td></tr></table>

https://doi.org/10.1371/journal.pone.0220639.t002

![](images/8ba3f024d3dac54fd9a850ef776783ad7618dca29df8dd4067f42dc77afc4e1c.jpg)

<details>
<summary>natural_image</summary>

Circular grayscale pattern with concentric rings and crosshair grid lines, labeled 'Axial' and number '1' (no other text or symbols)
</details>

![](images/124d3ac562622c2e4ab93670f167cbfb977d7557f7e7fcbcfeca1e45895e68c0.jpg)

<details>
<summary>text_image</summary>

Sagittal
4000
[a.u.]
0
1 3
</details>

![](images/259446c16882fa5038d3b59f48dad56ef4440f10fead36ff6704373e61c26b27.jpg)

<details>
<summary>natural_image</summary>

Circular grayscale image with three white circular spots on a dark background, labeled 'Coronal' and numbered 1, 2, 3 at bottom corners (no other text or symbols)
</details>

Fig 4. 3-plane gradient echo magnitude images of the spherical inclusion phantom. Small air bubbles are visible on the gel boundaries (black arrows). Three inclusions with iron concentrations of 0.043, 0.086, 0.129 mM are indicated as 1, 2, 3, respectively.

https://doi.org/10.1371/journal.pone.0220639.g004

(https://people.eecs.berkeley.edu/\~chunlei.liu/software.html, to be called "algorithm 1") and MEDI (http://pre.weill.cornell.edu/mri/pages/qsm.html, updated 03/27/2019, to be called "algorithm 2") for quantitative susceptibility mapping. In algorithm 1, the phase images were first unwrapped by the Laplacian method, and the background phase was removed by spherical mean value-based method with variable radii (V-SHARP)[13] using radius parameter of 35 mm. The resulting three-dimensional local phase map was converted to a susceptibility map by the STAR-QSM method [14] as implemented in the package. In algorithm 2, the image phase was unwrapped by region growing, from which the background phase was removed by projection on dipolar fields [8]. Susceptibility maps were calculated by morphology-enabled dipolar inversion with regularization to suppress susceptibility gradient in regions with low image intensity gradient. The default regularization parameter was used (lambda = 1000), after verifying that changing lambda within a factor of two produced similar results (S6 File).

# 4. Results

# 4.1. Phantom images

Fig 4 shows three-plane gradient echo magnitude images (TE = 7 ms) of the fabricated phantom. The three inclusions with iron molar densities of 0.043, 0.086, and 0.129 mM (marked as 1,2,3, respectively) are visible with sharp compartmental boundaries in the background of the un-doped gelatin. This shows that water diffusion across the boundaries was suppressed effectively. On the other hand, signal voids that are most likely caused by air bubbles are visible on the surfaces of the spheres, as indicated by black arrows.

Gradient echo phase images at the same TE are shown in Fig 5. The figure compares the measured phase with the theoretical one based on Eq (2) with susceptibility values of 0.1, 0.2, 0.3 ppm for inclusions 1 to 3. The following observations can be made from Fig 5. First, as predicted the measured phase is close to zero within the inclusion spheres. The mean and standard deviation of the phase for inclusions 1 to 3, calculated within a 25 mm diameter volumetric ROI, were $-0.19 \pm 0.13$ rad, $0.011 \pm 0.044$ rad, and $-0.075 \pm 0.061$ rad, respectively.

![](images/f0dd42c3edd053c780564a3483d9bcfe8ba812fe2696f7ad1ff3ac17a0ad0735.jpg)

<details>
<summary>text_image</summary>

(a)
Axial
Sagittal
[rad]
0
-π
B_app
Measured
Coronal
B_app
1
2
3
1
3
</details>

![](images/88304b3f16c61fe1c93d8a7ac79d03c14ff16652e4d129c16c655725e568363f.jpg)

<details>
<summary>heatmap</summary>

| Model | Value (rad) |
|-------|-------------|
| 1     | -π          |
| 2     | 0           |
| 3     | π           |
</details>

![](images/45fafb212b75a535ef8ed956d706aac975b593de9f93cc828f16603b7e77f102.jpg)

<details>
<summary>text_image</summary>

(c)
Measured - Model
[rad]
-π
0
-π
</details>

Fig 5. Measured vs model phase maps. (a) Measured phase. Phase variations due to air bubbles (white arrows), phantom stand (yellow arrows), and imperfect shimming (green dotted arrows) are visible. Spherical inclusions (1–3) produce dipolar field patterns consistent with the main field ( $B_{app}$ ) direction (blue arrows). (b) Dipolar model phase for inclusion susceptibilities 0.1, 0.2, 0.3 ppm. (c) Phase difference, showing good agreement between model and measurement. White arrowheads indicate error due to model boundary mismatch.

https://doi.org/10.1371/journal.pone.0220639.g005

Second, the phase around the inclusion spheres displays the characteristic dipolar field patterns with relatively large positive values along the main magnetic field ( $B_{app}$ ) direction, and negative values in the perpendicular directions. The intensity of the phase variation increased from inclusion 1 to 3, in the order of increasing iron concentration and in good agreement with theoretically modelled phase. Finally, the measured phase map in Fig 5 was contaminated by a few unwanted sources: air bubbles (white arrows), contact with the phantom stand (yellow arrows), and slowly-varying background magnetic field left uncompensated by shimming (green dotted arrows). The latter effect could be verified by a separate experiment where the scanner's shim setting was manually changed and corresponding phase changes were observed (S1 Fig).

# 4.2. Relaxation rates

Table 3 shows the measured relaxation rates of the spherical inclusions with different iron oxide concentrations (for details see S2 File). The rates increase approximately linearly with the concentration. Deviation from linearity is likely attributed to errors in controlling the small volumes of the iron oxide solutions. The relaxivities $\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_2^*$ of the contrast agent were determined from the linear regression analysis (S3 File). It is found that the particles act as a strong transverse relaxation agent, while affecting longitudinal relaxation relatively weakly.

Table 3. Relaxation rates and relaxivities of the three inclusion spheres. 

<table><tr><td>Inclusion</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Iron oxide solution [ml per 250 ml gel]</td><td>0.02</td><td>0.04</td><td>0.06</td></tr><tr><td>Iron molarity [μM]</td><td>42.9</td><td>85.8</td><td>128.7</td></tr><tr><td> $R_1$  [ $s^{-1}$ ]</td><td>2.24</td><td>2.51</td><td>2.76</td></tr><tr><td> $R_2$  [ $s^{-1}$ ]</td><td>9.97</td><td>17.1</td><td>22.2</td></tr><tr><td> $R_2^*$  [ $s^{-1}$ ]</td><td>10.1</td><td>17.3</td><td>22.1</td></tr><tr><td> $r_1$  [ $s^{-1}$ mM $^{-1}$ ]</td><td colspan="3"> $6.06 \pm 0.20^a$ </td></tr><tr><td> $r_2$  [ $s^{-1}$ mM $^{-1}$ ]</td><td colspan="3"> $142 \pm 14^a$ </td></tr><tr><td> $r_2^*$  [ $s^{-1}$ mM $^{-1}$ ]</td><td colspan="3"> $139 \pm 17^a$ </td></tr></table>

a. Standard error.

https://doi.org/10.1371/journal.pone.0220639.t003

In the literature, $r_1$ and $r_2$ of the iron oxide particles (in particular, Molday ION) have been reported at different field strengths. For example, the vendor's website (http://www.biopal.com/pdf-downloads/data-sheets/data-sheet-CL-30Q02-2.pdf, accessed 2019/05/24) provides $r_1 = 36 \left[ s^{-1} mM^{-1} \right]$ and $r_2 = 71 \left[ s^{-1} mM^{-1} \right]$ at 0.47 T. Brewer et al [15] reported $r_2 = 178 \left[ s^{-1} mM^{-1} \right]$ at 7T for particles with comparable magnetic core with different functionalization. Given that $r_1$ decreases and $r_2$ increases with the main magnetic fields [16], our results of $r_1 = 6.1 \left[ s^{-1} mM^{-1} \right]$ and $r_2 = 142 \left[ s^{-1} mM^{-1} \right]$ at 3T are compatible with the literature values.

A significant observation is that for all the inclusions, the transverse relaxation rates and the effective transverse relaxation rates are very close. This reflects homogeneous $B_{0}$ field inside the inclusions, to suppress inhomogeneous spin dephasing, which is a direct consequence of the spherical phantom geometry.

# 4.3. Susceptibility maps

Fig 6 shows the reconstructed susceptibility maps on the same 3 planes as in Figs 4 and 5. With algorithm 1, inclusions 1 to 3 exhibited susceptibility values of 0.100, 0.119 and 0.165 ppm, respectively, averaged over a spherical ROI of diameter $26\mathrm{mm}$ and referenced to the background gel. With algorithm 2, the corresponding values were 0.092, 0.211, and 0.288 ppm, revealing significant difference between the algorithms. Linear regression (S6 File) of these values as a function of the iron molar concentration yielded the normalized agent susceptibility of 0.758 ppm/mM(Fe) for algorithm 1 and 2.273 ppm/mM(Fe) for algorithm 2. Theoretical susceptibility of the iron oxide particles is different from susceptibility of isolated Fe ions, due to strong spin-spin coupling leading to superparamagnetism. At 3T, the particle's magnetic moment nearly saturates and becomes weakly (weaker than the Curie's law) temperature dependent [17]. A rough estimate can be made from the saturation magnetization of iron oxide nanoparticles of comparable size originally reported (at $310\mathrm{K}$ ) in [18]. Here, the saturation magnetic moment of about 70 emu/g(Fe) can be converted into magnetization $M_{s} = 3.92$ A/m/mM(Fe), which translates into an effective susceptibility of $\chi = \mu_0M_s / B_{app} = 1.70~\mathrm{ppm}/$ mM(Fe). This is closer to the result of algorithm 2 than algorithm 1.

A more direct evidence that algorithm 1 underestimated the true magnetization of the inclusions can be found from the phase map. Similar to Fig 5, we computed the theoretical dipolar phase map assuming the susceptibility values (0.100, 0.119, 0.165 ppm) obtained from algorithm 1, and subtracted it from the measured phase. The result (S2 Fig) clearly showed residual phase around inclusions 2 and 3 indicating underestimation of susceptibility. While

(a)   
![](images/633c1cf9dadb936a2fd7155dc23ae91e8ba4d840d045ac6cd21b7ea6cd0de0d8.jpg)

<details>
<summary>text_image</summary>

Axial
Sagittal
Coronal
B_app
</details>

(b)   
![](images/cb9779a6b1c36904647631ae839fd0cec64905f00ab57801f9839527880c058a.jpg)

<details>
<summary>line</summary>

| Region   | mM (Fe) | Susceptibility |
| -------- | ------- | ------------- |
| Axial    | 0.05    | 0.1           |
| Sagittal | 0.05    | 0.2           |
| Coronal  | 0.05    | 0.3           |
</details>

Fig 6. 3-plane quantitative susceptibility maps of the phantom processed with algorithm 1 (a) and 2 (b). In (a), streaking artifacts are apparent on the coronal and sagittal planes (orange dashed arrows), giving rise to artificial susceptibility elevation on the axial plane (orange arrow). In both (a) and (b), phase due to the phantom stand produced little susceptibility artifact (white arrows on sagittal planes). The inset of (b) shows the mean susceptibilities of the three inclusions for each processing method. Numerical values and linear fit results are listed in S6 File.

https://doi.org/10.1371/journal.pone.0220639.g006

the exact cause of this discrepancy is unclear, we note that paramagnetic susceptibility underestimation in QSM has been reported before [9].

Fig 6 also shows significant qualitative difference in susceptibility maps between the two algorithms. Specifically, algorithm 1 showed streaking artifacts on the coronal and sagittal planes (orange dashed arrows), introducing spurious localized susceptibility elevation on the axial plane (orange arrow). Such artifacts were largely absent in algorithm 2, thanks presumably to morphology based regularization suppressing susceptibility variation within homogeneous regions [6]. Note that subtle brightening in the similar location in the axial-plane phase maps (Fig 5A and 5B) reflects the actual, susceptibility-induced $B_{0}$ variation, and is not related to the streaking artifact. Lastly, we found in both algorithms that the significant phase near the phantom stand shown in Fig 5 (sagittal plane) did not carry over to the susceptibility map as an artifact in Fig 6. This indicates that the background field removal in both QSM algorithms successfully suppressed signals from susceptibility sources outside the phantom.

# 4.4. Stability

In order to assess the stability of the phantom over time, we have conducted QSM and relaxation time measurements on the same phantom 85 days (QSM) and 64 days (relaxation) after the initial experiments (S4 File). Fig 7 shows that the inclusions' mean susceptibility underwent minor change over the 85-day period, namely +2.1%, -4.9%, and -3.5% for inclusions 1–3, respectively. This trend was in agreement with R $_{2}$ \* changes over 64 days (S5 File), which were +0.7%, -2.0%, and -1.3% for the same inclusions. With limited temporal data points, and in the absence of independent scanner drift information, the origin of these changes is unclear. We suspect that water diffusion across compartmental boundaries could be a contributing factor. At present, the proposed phantom appears well-suited for cross-sectional comparative studies, but longitudinal, absolute susceptibility studies may require more work to improve the phantom's stability.

Despite measured susceptibility drifts, the general appearance of the susceptibility maps indicated no evidence of iron settlement at the bottom, or blurring at the spherical boundaries.

(a)   
![](images/5784db18bfbcbdf6ce07433bc585ad72c2a90c107779b52a8042729c7c7ef155.jpg)

<details>
<summary>text_image</summary>

Jan 24 (Day 75)
[ppm]
0.2
0.1
0
-0.1
-0.2
</details>

(b)   
![](images/5c8a88d671ee0d952c9e50a4afbe6f3ae6f4fc3872a0bcfed14feb737b8de533.jpg)

<details>
<summary>text_image</summary>

Apr 19 (Day 160)
</details>

(c) 

<table><tr><td colspan="2">Susceptibility (ppm)</td><td>Day 75</td><td>Day 160</td><td>Change</td></tr><tr><td rowspan="2">Inclusion 1</td><td>mean</td><td>0.0921</td><td>0.0941</td><td>+2.1%</td></tr><tr><td>std*</td><td>0.0048</td><td>0.0043</td><td></td></tr><tr><td rowspan="2">Inclusion 2</td><td>mean</td><td>0.2114</td><td>0.2015</td><td>-4.9%</td></tr><tr><td>std</td><td>0.0045</td><td>0.0042</td><td></td></tr><tr><td rowspan="2">Inclusion 3</td><td>mean</td><td>0.2876</td><td>0.2778</td><td>-3.5%</td></tr><tr><td>std</td><td>0.0038</td><td>0.0034</td><td></td></tr></table>

\* Standard deviation in the ROI. This does not include any uncertainty due to e.g., temperature or scanner hardware drift in the 85-day period.

(d)   
![](images/15a5da991f8a7289b8e6235acb01f852b9a03337606daf8b94573213c68acc5d.jpg)

<details>
<summary>bar</summary>

Susceptibility of inclusions
| Inclusion | 24-Jan (ppm) | 19-Apr (ppm) |
| :--- | :--- | :--- |
| Inclusion 1 | 0.09 | 0.09 |
| Inclusion 2 | 0.21 | 0.20 |
| Inclusion 3 | 0.29 | 0.28 |
</details>

Fig 7. QSM by MEDI obtained with the same phantom 85 days apart. (a-b) 3 plane susceptibility maps; (a) is the same as in Fig 6B. Yellow arrows in (b) indicate additional surface defects. (c) Mean and standard deviation of susceptibility in each inclusion. (d) Plot of susceptibility, with error bars according to 1 std from (c).

https://doi.org/10.1371/journal.pone.0220639.g007

We noticed, however, that more surface defects formed at the inclusion boundaries in the later-day data (yellow arrows in Fig 7B). We suspect that repeated cooling (for storage) and warming (during scan) could have stressed and possibly cracked the compartmental boundaries. In the future, we plan to investigate storing the phantom at room temperature to prevent thermal expansion-related problems. For this, more conventional (albeit toxic) preservatives such as sodium azide could be more effective than what was used here.

# 5. Discussion

We have demonstrated fabrication of a spherical gelatin phantom with spherical inclusions with different magnetic susceptibilities for possible use as a testbed for MR-based susceptibility imaging. The outer sphere theoretically eliminates background $B_{0}$ inhomogeneity caused by the air-phantom susceptibility boundary, whereas the inner spheres produce well-known volumetric dipolar field patterns for easy verification. Compared to previous susceptibility inclusion phantoms (mostly cylindrical or elliptical, and often utilizing elastic bags), our construction allows more repeatable fabrication. Although non-spherical phantoms have been successfully used for multi-center QSM validation [7], we believe that spherical phantoms have merit for volumetric measurements and phase imaging with available "ground truth". It is our expectation that the proposed spherical inclusion phantom could be of use to validate

phase-based imaging and parameter mapping in MRI, including QSM, which has often been done with numerical models or phantoms with less controllable shapes.

For a demonstration, we applied publicly available QSM reconstruction software to calculating susceptibility maps of the fabricated phantom having 3 inner spherical compartments. The results indicated that while the background phase due to the phantom stand was successfully taken care of, the dipolar field inversion process can suffer from streaking artifacts and susceptibility underestimation. While these artifacts have been reported before $[9, 19]$ , our phantom provides a way to experimentally explore them in terms of their dependence on image acquisition parameters, such as voxel size, voxel aspect ratio, and scan plane orientation, under a real scanning condition and environment. We emphasize that the purpose of our demonstration was not to address the capabilities of the current QSM algorithms, but rather to present an example of how our phantom can be used to examine such capabilities in future experimental studies. While we have used iron oxide concentrations with relatively easily detectable susceptibility ( $\geq 0.1$ ppm), in the future lower-dose phantoms could be made for more refined tests of QSM.

In our phantom, fabrication errors leading to deviations from the compartmental sphericity came from three main sources: air bubbles, imperfect spherical shape of the acrylic formers, and the seams at the "equators" where two hemispherical shells met. Among these the air bubbles are the least predictable and troublesome in terms of localized phase errors. We have tried to avoid air bubble formation by heating the aqueous gelatin solution before gelation. Also, possible air bubble trapping underneath each inclusion compartment was minimized by putting extra gelatin solution in the indentation before the compartment was loaded. We think this strategy was successful because images showed little evidence that bubbles were prevalent in the lower part of the phantom. In general, air bubbles tended to form on the surface of the spheres, and got exacerbated over time as mentioned in Section 4.4. We suspect that hydrophobic characteristic of the water-proof coating may be a contributor, and plan to explore other coating options to mitigate the effect in future experiments.

One limitation of our experiments was that the temperature was not controlled. Since the phantom was taken from a refrigerator shortly before each scan session, significant temperature change could have happened during the scan. Measurement under a representative experimental condition (S7 File) revealed that the temperature at the center of the phantom rose from 8.3 to 14.5°C during a typical session (including scan preparation) lasting \~90 minutes. For the purpose of comparing the present results with future experiments, we propose that the phantom temperature should be taken as $11.4 \pm 3$ °C. There could also have been significant spatial (radial) temperature gradient; however, any center-out signal variation attributable to such gradient was not evident in the images. At high fields, the magnetic property of the iron oxide nanoparticles depends on temperature through saturation magnetization ( $M_{s}$ ). While references for the particular particles used (Molday ION) were not readily available, in ref [17], smaller (16 nm vs our 30 nm) iron oxide nanoparticles exhibited a temperature coefficient of $M_{s}$ less than 1% per 10°C near the room temperature. This is more than 3 times weaker than the Curie’s law prediction (for individual paramagnetic ions, \~1/T), and would imply that our experiments were subject to less than 1% error in susceptibility if the results of ref [17] are applicable. The temperature issue should be resolved if future phantoms could be stored at room temperature.

In our work, we did not attempt precise control of the inclusion sphere locations better than about a couple of millimeters, by manually scooping out on the base gel in Step 2. For better definition of the positions, one could potentially utilize a 3D-printed plastic template during the base gel formation (Step 1), which can produce controlled recess on the gel to load the small spheres.

Fig 5 shows that significant unwanted phase was introduced by environmental susceptibility effect, especially from the phantom support material touching the lower part of the phantom. To construct a support piece from lighter materials or materials with more air-like susceptibility $[20, 21]$ will help reduce the phase contamination, and allow better exploitation of the benefits of the spherical phantom geometry.

In our phantom, formalin was added to suppress water diffusion across compartmental boundaries. In terms of magnetic susceptibility, we do not believe the effect of formalin was significant. Generally, any diamagnetic liquid with mass density similar to water has volume susceptibility close to water [1]. Since we have added 2.5 ml of formalin solution to 250 ml of gel, any susceptibility difference would have been diluted by 100, diminishing its effect. For verification, we compared the R $_{2}$ \* of pure gel and gel + formalin phantoms in a separate experiment (S8 File). We found R $_{2}$ \* (gel) = 2.86 s $^{-1}$ , R $_{2}$ \* (gel + formalin) = 3.17 s $^{-1}$ , therefore ΔR $_{2}$ \* (formalin) = 0.31 s $^{-1}$ , about 3% of the R $_{2}$ \* of the lowest-density doped gel compartment (10.1 s $^{-1}$ ) in the inclusion phantom. Assuming linear relationship between R $_{2}$ \* and susceptibility, we estimate the formalin solution susceptibility of 3% × 0.1 ppm = 0.003 ppm, comparable to $^{1}$ H nuclear spin susceptibility that is normally negligible [22].

In conclusion, we have demonstrated a method to fabricate a spherical gel phantom which contains smaller spherical compartments for MRI-based magnetic susceptibility imaging. The proposed method allows repeatable production of susceptibility inclusion phantoms with controlled geometry, which can be useful for experimental validation of MR phase imaging and quantitative susceptibility mapping.

# Supporting information

S1 Fig. Shim-dependent phase maps.

(PPTX)

S2 Fig. Measured vs model phase maps for inclusion susceptibilities of 0.1, 0.119, 0.165 ppm. (PPTX)

S1 File. Image (DICOM) files per sequences in Table 2.

(ZIP)

S2 File. Relaxation time data, 1st measurement.

(PPTX)

S3 File. Linear regression for iron oxide concentration-dependent relaxation rates.

(XLSX)

S4 File. Relaxation time data, 2nd measurement.

(PPTX)

S5 File. Relaxation rate change over time.

(XLSX)

S6 File. Linear regression for iron oxide concentration-dependent susceptibility and comparison between QSM processing codes.

(XLSX)

S7 File. Temperature data.

(XLSX)

S8 File. Gel and formalin relaxation data.

(PPTX)

# Acknowledgments

The authors thank Mr. Jin-Hwan Jeon and Ms. Boohee Choi for kind assistance with experimental setups and MRI scans. Ms. Seon-Ha Hwang helped with relaxation parameter analysis.

# Author Contributions

Conceptualization: Seung-Kyun Lee.

Investigation: Jun-Ho Kim, Jung-Hyun Kim, So-Hee Lee.

Supervision: Jinhyoung Park, Seung-Kyun Lee.

Writing - original draft: Seung-Kyun Lee.

Writing - review & editing: Jinhyoung Park, Seung-Kyun Lee.

# References

1. Schenck JF. The role of magnetic susceptibility in magnetic resonance imaging: MRI magnetic compatibility of the first and second kinds. Med Phys. 1996; 23(6):815–50. https://doi.org/10.1118/1.597854 PMID: 8798169   
2. Wang Y, Liu T. Quantitative susceptibility mapping (QSM): Decoding MRI data for a tissue magnetic biomarker. Magn Reson Med. 2015; 73(1):82–101. https://doi.org/10.1002/mrm.25358 PMID: 25044035   
3. Langkammer C, Schweser F, Krebs N, Deistung A, Goessler W, Scheurer E, et al. Quantitative susceptibility mapping (QSM) as a means to measure brain iron? A post mortem validation study. NeuroImage. 2012; 62(3):1593–9. https://doi.org/10.1016/j.neuroimage.2012.05.049 PMID: 22634862   
4. Langkammer C, Schweser F, Shmueli K, Kames C, Li X, Guo L, et al. Quantitative susceptibility mapping: Report from the 2016 reconstruction challenge. Magn Reson Med. 2018; 79(3):1661–73. https://doi.org/10.1002/mrm.26830 PMID: 28762243   
5. Liu T, Spincemaille P, de Rochefort L, Kressler B, Wang Y. Calculation of susceptibility through multiple orientation sampling (COSMOS): a method for conditioning the inverse problem from measured magnetic field map to susceptibility source image in MRI. Magn Reson Med. 2009; 61(1):196–204. https://doi.org/10.1002/mrm.21828 PMID: 19097205   
6. Liu J, Liu T, de Rochefort L, Ledoux J, Khalidov I, Chen W, et al. Morphology enabled dipole inversion for quantitative susceptibility mapping using structural consistency between the magnitude image and the susceptibility map. NeuroImage. 2012; 59(3):2560–8. https://doi.org/10.1016/j.neuroimage.2011.08.082 PMID: 21925276   
7. Deh K, Kawaji K, Bulk M, Van Der Weerd L, Lind E, Spincemaille P, et al. Multicenter reproducibility of quantitative susceptibility mapping in a gadolinium phantom using MEDI+0 automatic zero referencing. Magn Reson Med. 2019; 81(2):1229–36. https://doi.org/10.1002/mrm.27410 PMID: 30284727   
8. de Rochefort L, Liu T, Kressler B, Liu J, Spincemaille P, Lebon V, et al. Quantitative susceptibility map reconstruction from MR phase data using bayesian regularization: validation and application to brain imaging. Magn Reson Med. 2010; 63(1):194–206. https://doi.org/10.1002/mrm.22187 PMID: 19953507   
9. Zhou D, Cho J, Zhang J, Spincemaille P, Wang Y. Susceptibility underestimation in a high-susceptibility phantom: Dependence on imaging resolution, magnitude contrast, and other parameters. Magn Reson Med. 2017; 78(3):1080–6. https://doi.org/10.1002/mrm.26475 PMID: 27699883   
10. Schafer A, Wharton S, Gowland P, Bowtell R. Using magnetic field simulation to study susceptibility-related phase contrast in gradient echo MRI. NeuroImage. 2009; 48(1):126–37. https://doi.org/10.1016/j.neuroimage.2009.05.093 PMID: 19520176   
11. Brown RW, Cheng Y-CN, Haacke EM, Thompson MR, Venkatesan R. Magnetic resonance imaging: physical principles and sequence design. Second edition. ed. Hoboken, New Jersey: John Wiley & Sons, Inc.; 2014.   
12. Madsen EL, Hobson MA, Shi H, Varghese T, Frank GR. Tissue-mimicking agar/gelatin materials for use in heterogeneous elastography phantoms. Physics in medicine and biology. 2005; 50(23):5597–618. https://doi.org/10.1088/0031-9155/50/23/013 PMID: 16306655   
13. Wu B, Li W, Guidon A, Liu C. Whole brain susceptibility mapping using compressed sensing. Magn Reson Med. 2012; 67(1):137–47. https://doi.org/10.1002/mrm.23000 PMID: 21671269

14. Wei H, Dibb R, Zhou Y, Sun Y, Xu J, Wang N, et al. Streaking artifact reduction for quantitative susceptibility mapping of sources with large dynamic range. NMR Biomed. 2015; 28(10):1294–303. https://doi.org/10.1002/nbm.3383 PMID: 26313885   
15. Brewer KD, Spitler R, Lee KR, Chan AC, Barrozo JC, Wakeel A, et al. Characterization of Magneto-Endosymbionts as MRI Cell Labeling and Tracking Agents. Mol Imaging Biol. 2018; 20(1):65–73. https://doi.org/10.1007/s11307-017-1093-7 PMID: 28616842   
16. Brewer K, Chan A, Rioux J, Rafat M, Machtaler S, Spitler R, et al. Characterization of Magnetotactic Bacteria as MRI Cell Labeling and Tracking Agents. AACR-SNMMI Joint Conference on State-of-the-Art Molecular Imaging in Cancer Biology and Therapy; San Diego, CA, USA 2015.   
17. Nayek C, Manna K, Bhattacharjee G, Murugavel P, Obaidat I. Investigating size- and temperature-dependent coercivity and saturation magnetization in PEG coated Fe3O4 nanoparticles. Magnetochemistry. 2017; 3:19.   
18. Shen T, Weissleder R, Papisov M, Bogdanov A Jr, Brady TJ. Monocrystalline iron oxide nanocompounds (MION): physicochemical properties. Magn Reson Med. 1993; 29(5):599–604. PMID: 8505895   
19. Li W, Wang N, Yu F, Han H, Cao W, Romero R, et al. A method for estimating and removing streaking artifacts in quantitative susceptibility mapping. NeuroImage. 2015; 108:111–22. https://doi.org/10.1016/j.neuroimage.2014.12.043 PMID: 25536496   
20. Wapler MC, Leupold J, Dragonu I, von Elverfeld D, Zaitsev M, Wallrabe U. Magnetic properties of materials for MR engineering, micro-MR and beyond. Journal of magnetic resonance. 2014; 242:233–42. https://doi.org/10.1016/j.jmr.2014.02.005 PMID: 24705364   
21. Hwang S-H, Lee S-K. Efficient experimental design for measuring magnetic susceptibility of arbitrarily shaped materials by MRI. Investigative Magnetic Resonance Imaging. 2018; 22:141–9.   
22. Park J, Lee J, Park JY, Lee SK. Nuclear paramagnetism-induced MR frequency shift and its implications for MR-based magnetic susceptibility measurement. Magn Reson Med. 2017; 77(2):848–54. https://doi.org/10.1002/mrm.26570 PMID: 28019024