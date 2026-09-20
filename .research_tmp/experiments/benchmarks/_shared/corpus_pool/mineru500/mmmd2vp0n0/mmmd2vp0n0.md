# Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction

Anagh Malik $^{1,2}$

anagh@cs.toronto.edu

Parsa Mirdehghan $^{1,2}$

parsa@cs.toronto.edu

Sotiris Nousias $^{1}$

sotiris@cs.toronto.edu

Kiriakos N. Kutulakos $^{1,2}$

kyros@cs.toronto.edu

David B. Lindell $^{1,2}$

lindell@cs.toronto.edu

$^{1}$ University of Toronto $^{2}$ Vector Institute anaghmalik.com/TransientNeRF

# Abstract

Neural radiance fields (NeRFs) have become a ubiquitous tool for modeling scene appearance and geometry from multiview imagery. Recent work has also begun to explore how to use additional supervision from lidar or depth sensor measurements in the NeRF framework. However, previous lidar-supervised NeRFs focus on rendering conventional camera imagery and use lidar-derived point cloud data as auxiliary supervision; thus, they fail to incorporate the underlying image formation model of the lidar. Here, we propose a novel method for rendering transient NeRFs that take as input the raw, time-resolved photon count histograms measured by a single-photon lidar system, and we seek to render such histograms from novel views. Different from conventional NeRFs, the approach relies on a time-resolved version of the volume rendering equation to render the lidar measurements and capture transient light transport phenomena at picosecond timescales. We evaluate our method on a first-of-its-kind dataset of simulated and captured transient multiview scans from a prototype single-photon lidar. Overall, our work brings NeRFs to a new dimension of imaging at transient timescales, newly enabling rendering of transient imagery from novel views. Additionally, we show that our approach recovers improved geometry and conventional appearance compared to point cloud-based supervision when training on few input viewpoints. Transient NeRFs may be especially useful for applications which seek to simulate raw lidar measurements for downstream tasks in autonomous driving, robotics, and remote sensing.

# 1 Introduction

The ability to sense and reconstruct 3D appearance and geometry is critical to applications in vision, graphics, and beyond. Lidar sensors [1] are of particular interest for this task due to their high sensitivity to arriving photons and their extremely high temporal resolution; as such, they are being deployed in systems for 3D imaging in smart phone cameras [2], autonomous driving, and remote sensing [3]. Recent work has also begun to explore how additional supervision from lidar [4] or depth sensor measurements [5] can be incorporated into the NeRF framework to improve novel view synthesis and 3D reconstruction. Existing NeRF-based methods that use lidar [4] are limited to rendering conventional RGB images, and use lidar point clouds (i.e., pre-processed lidar measurements) as auxiliary supervision rather than rendering the raw data that lidar systems actually collect. Specifically, lidars capture transient images—time-resolved picosecond- or nanosecond-scale measurements of a pulse of light travelling to a scene point and back. We consider the problem

![](images/9f55c248f55fbf15c4ca95964091efa48356321eb7c9fca619289d08f216796f.jpg)

<details>
<summary>text_image</summary>

single-photon lidar
scanning
mirrors
pulsed
laser
SPAD
TCSPC
photon count
histogram
returning
laser pulse
time
</details>

![](images/d614eaf912ec833aa493560d2b2e78f4a7a818ea587cad99664f87686183377a.jpg)

<details>
<summary>text_image</summary>

multiview lidar scans
photon count
histograms
Transient NeRF
</details>

![](images/3b511ca9c6f742620fe6cbc4a4b9f2da7791d27f8ad28a3088ec6f1aa8de7efb.jpg)

<details>
<summary>text_image</summary>

lidar novel views
intensity depth
</details>

Figure 1: Overview of transient neural radiance fields (Transient NeRFs). Measurements from a single-photon lidar are captured using a single-photon avalanche diode (SPAD), pulsed laser, scanning mirrors, and a time-correlated single photon counter (TCSPC). The lidar scans, consisting of a 2D array of photon count histograms (visualized with maximum-intensity projection), are captured from multiple viewpoints and used to optimize the transient NeRF. After training, we render novel views of time-resolved lidar measurements (x-y and x-t slices are indicated by the dotted red lines), and we also convert the rendered data into intensity and depth maps.

of how to synthesize such transients from novel viewpoints. In particular, we seek a method that takes as input and renders transients in the form of time-resolved photon count histograms captured by a single-photon lidar system $^{1}$ [8]. Lidar view synthesis may be useful for applications that seek to simulate raw lidar measurements for downstream tasks, including autonomous driving, robotics, remote sensing, and virtual reality.

The acquisition and reconstruction of transient measurements has been studied across various different sensing modalities, including holography $[9]$ , photonic mixer devices $[10, 11]$ streak cameras $[12]$ , and single-photon detectors (SPADs) $[13, 14]$ .

In the context of SPADs and single-photon lidar, a transient is measured by repeatedly illuminating a point with pulses of light and accumulating the individual photon arrival times into a time-resolved histogram. After capturing such histograms for each point in a scene, one can exploit their rich spatio-temporal structure for scene reconstruction $[15, 16]$ , to uncover statistical properties of captured photons $[17, 18]$ , and to reveal the temporal profile of the laser pulse used to probe the scene (knowledge of which can significantly improve depth resolution $[19, 20]$ ). These properties motivate transients as a representation and their synthesis from novel views. While existing methods have explored multiview lidar reconstruction $[21–25]$ , they exclusively use point cloud data, and do not tackle lidar view synthesis.

Recently, a number of NeRF-based methods for 3D scene modeling have also been proposed to incorporate point cloud data (e.g., from lidar or structure from motion) [26, 27] or information from time-of-flight sensors [5]. Again, these methods focus on synthesizing conventional RGB images or depth maps, while our approach synthesizes transient images. Another class of methods combines NeRFs with single-photon lidar data for non-line-of-sight imaging [28]; however, they focus on a very different inverse problem and scene parameterization [29], and do not aim to perform novel view synthesis of lidar data as we do.

Our approach, illustrated in Fig. 1, extends neural radiance fields to be compatible with a statistical model of time-resolved measurements captured by a single-photon lidar system. The method takes as input multiview scans from a single-photon lidar and, after training, enables rendering lidar measurements from novel views. Moreover, accurate depth maps or intensity images can also be rendered from the learned representation.

In summary, we make the following contributions.

- We develop a novel time-resolved volumetric image formation model for single-photon lidar and introduce transient neural radiance fields for lidar view synthesis and 3D reconstruction.   
- We assemble a first-of-its-kind dataset of simulated and captured transient multiview scans, constructed using a prototype multiview single-photon lidar system.   
- We use the dataset to demonstrate new capabilities in transient view synthesis and state-of-the-art results on 3D reconstruction and appearance modeling from few (2–5) single-photon lidar scans of a scene.

# 2 Related work

Our work ties together threads from multiple areas of previous research, including methods for imaging with single-photon sensors, and NeRF-based pipelines that leverage 3D information to improve reconstruction quality. Our implementation also builds on recent frameworks that improve the computational efficiency of NeRF training $[30, 31]$ .

Active single-photon imaging. Single-photon sensors output precise timestamps corresponding to the arrival times of individual detected photons. The most common type of single-photon sensor is the single-photon avalanche diode (SPAD). SPADs are based on the widely-available CMOS technology $[32]$ (which we consider in this work), but other technologies such as superconducting nanowire single-photon detectors $[33]$ and silicon photomultipliers $[34]$ , offer different tradeoffs in terms of sensitivity, temporal resolution, and cost.

In active imaging scenarios, pulsed light sources are paired with single-photon sensors to estimate the depth or reflectance of a scene by applying computational algorithms to the captured photon timestamps $[19, 35, 36]$ . The extreme temporal resolution of these sensors also enables direct capture of interactions of light with a scene at picosecond timescales $[37, 38]$ , and by modeling and inverting the time-resolved scattering of light, single-photon sensors can be used to see around corners $[28, 39–41]$ or through scattering media $[42–44]$ . The extreme sensitivity of single-photon sensors has made them an attractive technology for autonomous navigation $[8]$ , and accurate depth acquisition from mobile phones $[2]$ .

Our approach differs significantly from all the previous work in that we investigate, for the first time, the problem of lidar view synthesis and multi-view 3D reconstruction in the single-photon lidar regime. We introduce the framework of transient NeRFs for this task and jointly optimize a representation of scene geometry and appearance that is consistent with captured photon timestamps across all input views.

Finally, we note that while commercial lidars based on SPADs or avalanche photodiodes capture photon count histograms or time-resolved intensity, they typically pre-process these raw measurements to point cloud format before output. Hence, the raw lidar data that we use may not have been widely available to previous methods; we have publicly released our dataset of multiview photon count histograms on the project webpage to help alleviate this challenge.

3D-informed neural radiance fields. A number of recent techniques for multiview reconstruction using NeRFs leverage additional geometric information (sparse point clouds from lidar $[4]$ or structure from motion $[26, 27]$ ) to improve the reconstruction quality or reduce the number of required input viewpoints. Similar benefits can be obtained by combining volume rendering with monocular depth estimators $[45]$ , or using data from time-of-flight sensors $[5]$ . Other methods investigate the problem of view synthesis from few input images but leverage appearance priors instead of explicit depth supervision $[46–48]$ . In contrast to the proposed approach, all of these methods focus on reconstructing images or depth maps rather than transients.

# 3 Transient Neural Radiance Fields

We describe a mathematical model for transient measurements captured using single-photon lidar and propose a time-resolved volume rendering formulation compatible with neural radiance fields.

![](images/73937c4c70938e2245c400d23d661bb32b664b702d459efddeddb6869d2d3ee3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["ray casting"] --> B["x(p)"]
    B --> C["r(t), ω"]
    C --> D["time-resolved volume rendering"]
    D --> E["laser pulse"]
    E --> F["n"]
    F --> G["f"]
    G --> H["n"]
    H --> I["τ_f[i,j,n"]]
    I --> J["n"]
    J --> K["c"]
    K --> L["σ(r)"]
    L --> M["τ_f[i,j,n"]]
    M --> N["n"]
    N --> O["captured"]
    O --> P["τ_f[i,j,n"]]
    P --> Q["n"]
    Q --> R["c"]
    R --> S["τ_f[i,j,n"]]
    S --> T["n"]
    T --> U["captured"]
    U --> V["τ_f[i,j,n"]]
    V --> W["n"]
    W --> X["captured"]
    X --> Y["τ_f[i,j,n"]]
```
</details>

Figure 2: Rendering transient neural radiance fields. We cast rays through a volume and retrieve the density and color at each point using a neural representation [30]. A time-resolved measurement is constructed using volume rendering (Equation 3), and we bin the radiance contributions into an array based on distance along the ray. The result is convolved with the impulse response of the lidar (which incorporates the shape of the laser pulse), and we supervise the neural representation based on the difference between the rendered and captured transient measurements.

# 3.1 Image Formation Model

Consider that a laser pulse illuminates a point in a scene that is imaged onto a sensor at position $p \in R^{2}$ (see Fig. 2). Assume light from the laser pulse propagates to a surface and back to p along the same path described by a ray $\mathbf{r}(t)$ , where t indicates propagation time. The forward path along the ray is given as $\mathbf{r}(t) = \mathbf{x}(\mathbf{p}) + tc\boldsymbol{\omega}(\mathbf{p})$ , where $\mathbf{x}(\mathbf{p}) \in \mathbb{R}^{3}$ is the ray origin, $\boldsymbol{\omega}(\mathbf{p}) \in \mathbb{S}^{2}$ is the ray direction which maps to p, and c is the speed of light. Now, let $f(t)$ denote the temporal impulse response of the lidar (including the temporal profile of the laser pulse and the sensor jitter), and let $\alpha(\mathbf{p})$ incorporate reflectance and radiometric falloff factors [17] of the illuminated point at distance $z(\mathbf{p})$ from $\mathbf{x}(\mathbf{p})$ . Then, assuming single-bounce light transport, the photon arrival rate incident on the sensor from the laser pulse is given as

$$
\boldsymbol {\lambda} [ i, j, n ] = \int_ {\mathcal {P} _ {i, j}} \int_ {\mathcal {T} _ {n}} \alpha (\mathbf {p}) f \left(t - \frac {2 z (\mathbf {p})}{c}\right) \mathrm{d} t \mathrm{d} \mathbf {p}, \tag {1}
$$

where $T_{n}$ and $P_{i,j}$ indicate the temporal and spatial discretization intervals for the time bin n and pixel i, j, respectively. The term 2z/c gives the time delay for light to propagate to a point at distance z and back.

Now, we can describe the measured transient, or the number of photon detections captured by a SPAD [17], as

$$
\widetilde {\boldsymbol {\tau}} [ i, j, n ] \sim \text { POISSON } (N \eta \boldsymbol {\lambda} [ i, j, n ] + B), \quad B = N (\eta A [ i, j ] + D), \tag {2}
$$

where N indicates the number of laser pulses per pixel, $\eta \in (0,1)$ is the detection efficiency of the sensor, and B is the total number of background (non-laser pulse) detections. Background detections in turn depend on A, the average ambient photon rate at pixel $[i,j]$ , and D, number of false detections produced by the sensor per laser pulse period, also known as the dark count rate. When the number of detected photons is far fewer than the number of laser pulses, SPAD measurements can be modeled according to a Poisson process [17] where the arrival rate function varies across space and time. This model is appropriate for our measurements, which have relatively low flux (<5% detection probability per emitted laser pulse) [19]. The resulting measurements $\widetilde{\tau}[i,j,n]$ represent a noisy histogram of photon counts collected at pixel $[i,j]$ at time bin n.

# 3.2 Time-Resolved Volume Rendering

Using the measurements $\widetilde{\tau}$ , we wish to optimize a representation of the appearance and geometry of the scene. To this end, we propose a time-resolved version of the volume rendering equation used in NeRF [49, 50]. Specifically, we model clean (i.e., without Poisson noise) time-resolved histograms $\tau[i,j,n]$ as (writing $\mathbf{r}(t)$ as r for brevity)

$$
\boldsymbol {\tau} [ i, j, n ] = \int_ {\mathcal {P} _ {i, j}} \int_ {\mathcal {T} _ {n}} (t c) ^ {- 2} T (t) ^ {2} \sigma (\mathbf {r}) \mathbf {c} (\mathbf {r}, \boldsymbol {\omega}) \mathrm{d} t \mathrm{d} \mathbf {p},
$$

$$
\text { where } T (t) = \exp \left(- \int_ {t _ {0}} ^ {t} \sigma (\mathbf {r})   \mathrm{d} s\right). \tag {3}
$$

We denote by c the radiance of light scattered at a point $\mathbf{r}(t)$ in the direction $\omega$ , and $\sigma$ represents the volume density or the differential probability of ray termination at $\mathbf{r}(t)$ . Finally, $T(t)$ is the transmittance from a distance $t_{0}$ along the ray to t, and this term is squared to reflect the two-way propagation of light [5]. We additionally explicitly account for the inverse-square falloff of intensity, through the term $(tc)^{-2}$ . The definite integrals are evaluated over the extent of time bin n, $T_{n} = [t_{n-1}, t_{n}]$ , and over p within the area of pixel $[i, j]$ as in Equation 1. Note that in practice, we calculate Equation 1 using the discretization scheme of Max [51] used by Mildenhall et al. [49].

Finally, to account for the temporal spread of the laser pulse and sensor jitter, we convolve the estimated transient with the calibrated impulse response of the lidar system f to obtain

$$
\boldsymbol {\tau} _ {f} = f * \boldsymbol {\tau}. \tag {4}
$$

Without this step, the volumetric model of Equation 3 does not match the raw data from the lidar system and tends to produce thick clouds of density around surfaces to compensate for this mismatch.

# 3.3 Reconstruction

To reconstruct transient NeRFs, we use lidar measurements $\widetilde{\pmb{\tau}}^{(k)}[i,j,n]$ of a scene captured from $0\leq k\leq K - 1$ different viewpoints. We parameterize transient NeRF using a neural network $\mathcal{F}$ consisting of a hash grid of features and a multi-layer perceptron decoder [30]. The network takes as input a coordinate and viewing direction, and outputs radiance and density, $\mathcal{F}(\mathbf{r}(t),\omega) = \mathbf{c},\sigma$ . We use these outputs to render transients (see Fig. 2). The model is optimized to minimize the difference between the rendered transient and measured photon count histograms. We also introduce a modified loss function to account for the high dynamic range of lidar measurements, and we propose a space carving regularization penalty to help mitigate convergence to local minima in the optimization.

HDR-Informed loss function. Measurements from a single photon sensor can have a dynamic range that spans multiple orders of magnitude, with each pixel recording from zero to thousands of photons. We find that applying two exponential functions to the radiance preactivations (1) enforces non-negativity and (2) improves the dynamic range of the network output. Thus, we have $\mathbf{c} = \exp(\exp(\hat{\mathbf{c}})) - 1$ , where the network preactivations are given by $\hat{c}$ . Following Muller et al. [30] the network also predicts density in log space.

After time-resolved volume rendering using Equation 3, we apply a loss function in log space to prevent the brightest regions from dominating the loss [52]. The loss function is given as

$$
\mathcal {L} _ {\boldsymbol {\tau}} = \sum_ {k, i, j, n} \| \ln (\widetilde {\boldsymbol {\tau}} ^ {(k)} [ i, j, n ] + 1) - \ln (\boldsymbol {\tau} _ {f} ^ {(k)} [ i, j, n ] + 1) \| _ {1}, \tag {5}
$$

where the sum is over all images, pixels, and time bins.

Space carving regularization. We find that using the above loss function alone results in spurious patches of density in front of dark surfaces in a scene. Here, the network can predict bright values on the surface itself, but darkens the corresponding values of $\tau_{f}$ by placing additional spurious density values along the ray. Since the network can predict the radiance of the density to be zero at these points, the predicted transients $\tau_{f}$ can be entirely consistent with the measured transients $\widetilde{\tau}$ , but with incorrect geometry. To address this, we introduce a space carving regularization

$$
\mathcal {L} _ {\mathrm{sc}} = \sum_ {\substack {k, i, j, n \\ \widetilde {\boldsymbol {\tau}} ^ {(k)} [ i, j, n ] <   B}} \int_ {\mathcal {P} _ {i, j}} \int_ {\mathcal {T} _ {n}} T (t) \sigma (\mathbf {r}) \mathrm{d} t \mathrm{d} \mathbf {p}. \tag{6}
$$

This function penalizes any density along a ray at locations where the corresponding measured transient values are less than the expected background level B. This effectively forces space to be empty (i.e., zero density) at regions where the measurements do not indicate the presence of a surface.

The complete loss function used for training is then given as

$$
\mathcal {L} = \mathcal {L} _ {\tau} + \lambda_ {\mathrm{sc}} \mathcal {L} _ {\mathrm{sc}}, \tag {7}
$$

where $\lambda_{sc}$ controls the strength of the space carving regularization.

# 3.4 Implementation Details

Our implementation is based on the NerfAcc [31] version of Instant-NGP [30], which we extend to incorporate our time-resolved volume rendering equation. In particular, we extend the framework to output time-resolved transient measurements, to account for the pixel footprint, and to estimate depth.

Pixel footprint. Captured photon count histograms exhibit multiple peaks where the finite beam width of the laser passes over depth discontinuities. To account for this phenomenon we use a truncated Gaussian distribution to model the spatial footprint of the laser spot and SPAD sensor projected onto the scene. Specifically, we sample rays in the range of 4 standard deviations of the pixel center, weighting their contribution to the rendered transient by the corresponding Gaussian probability density function value (after rendering a histogram). We set the standard deviation of the Gaussian to 0.15 pixels for the simulated dataset and 0.10 pixels for the captured dataset.

Depth. To estimate depth we find the distance along each ray that results in the maximum probability of ray termination: $\arg\max_{t} T(t)\sigma(t)$ . Note that when integrating over the pixel footprint at occlusion boundaries, multiple local extrema can occur, and so taking the highest peak results in a single depth estimate without floating pixel artifacts.

Network optimization. We optimize the network using the Adam optimizer $[53]$ , a learning rate of $1 \times 10^{-3}$ and a multi-step learning rate decay of $\gamma = 0.33$ applied at 100K, 150K, and 180K iterations. We set the batch size to 512 pixels and optimize the simulated results until they appear to converge, or for 250K iterations for the simulated results and 150K iterations for the captured results. For the weighting of the space carving loss, we use $\lambda_{sc} = 10^{-3}$ for the simulated dataset and increase this to $\lambda_{sc} = 10^{-2}$ for captured data, which benefits from additional regularization. We train the network on a single NVIDIA A40 GPU. Also note that simulated results use RGB histograms, such that $c \in R^{3}$ . In the captured data $c \in R$ because we illuminate the scene with monochromatic laser light.

# 4 Multiview Lidar Dataset

We introduce a first-of-its-kind dataset consisting of simulated and captured multiview data from a single-photon lidar. A description of the full set of simulated and captured scenes is included in the supplemental, and the dataset and simulation code are publicly available on the project webpage.

Simulated dataset We create the simulated dataset using a time-resolved version of Mitsuba 2 [54] which we modify for efficient rendering of lidar measurements. The dataset consists of one scene from Vicini et al. [55] and four scenes made available by artists on Blendswap (https://blendswap.com/), which we ported from Blender to our Mitsuba 2 renderer. The training views are set consistent with the capture setup of our hardware prototype (described below) such that the camera viewpoint is rotated around the scene at a fixed distance and elevation angle, resulting in 8 synthetic lidar scans used for training. We evaluate on rendered measurements from six viewpoints sampled from the NeRF Blender test set [49]. The renders are used to simulate SPAD measurements by applying the noise model described in Equation 2 and setting the mean number of photon counts to 2850 per occupied pixel and the background counts to 0.001 per bin, which we set to approximate our experimentally captured data.

![](images/2188d2d6c605965f2a85c1b4c4f2433c769d777d2ea21ff3a48e618d488c9d13.jpg)

<details>
<summary>text_image</summary>

SPAD
scanning
mirrors
pulsed
laser
beamsplitter
laser path
</details>

Figure 3: Hardware prototype. A pulsed laser shares a path with a single-pixel SPAD, and the illumination and imaging path are controlled by scanning mirrors.

Hardware prototype. To create the captured dataset, we built a hardware prototype (Fig. 3) consisting of a pulsed laser operating at $532\mathrm{nm}$ that emits 35 ps pulses of light at a repetition rate of $10\mathrm{MHz}$ . The output power of the laser is lowered to $< 1\mathrm{mW}$ to keep the flux low enough

![](images/1b7d03f96144ca0559c3be7db1f09d67308caec288c3e67afd4eff2114f279b6.jpg)

<details>
<summary>text_image</summary>

Ground Truth
Instant-NGP
DS-NeRF
Urban NeRF
Urban NeRF-M
Proposed
rendered transients
4.1 4.5
t (ns)
chair
2 views
hotdog
3 views
lego
5 views
</details>

Figure 4: Results on simulated data. We show images from depth-supervised NeRF baselines as well as color images and rendered transients from our method after training on 2, 3, and 5 viewpoints. The proposed method produces cleaner results and generates 3D transients for each viewpoint.

Table 1: Simulated results comparing images and depth for the baselines and proposed approach. 

<table><tr><td rowspan="2">Method</td><td colspan="3">PSNR (dB)↑</td><td colspan="3">LPIPS↓</td><td colspan="3">L1 (depth) ↓</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>Instant NGP [30]</td><td>16.62</td><td>17.67</td><td>19.66</td><td>0.520</td><td>0.476</td><td>0.387</td><td>0.238</td><td>0.195</td><td>0.178</td></tr><tr><td>DS-NeRF [26]</td><td>19.28</td><td>19.35</td><td>21.07</td><td>0.431</td><td>0.436</td><td>0.376</td><td>0.109</td><td>0.115</td><td>0.119</td></tr><tr><td>Urban NeRF [4]</td><td>18.86</td><td>18.73</td><td>19.80</td><td>0.500</td><td>0.484</td><td>0.406</td><td>0.131</td><td>0.124</td><td>0.101</td></tr><tr><td>Urban NeRF w/mask [4]</td><td>20.91</td><td>20.81</td><td>22.34</td><td>0.410</td><td>0.382</td><td>0.339</td><td>0.051</td><td>0.038</td><td>0.029</td></tr><tr><td>Proposed</td><td>21.38</td><td>23.48</td><td>28.39</td><td>0.172</td><td>0.151</td><td>0.115</td><td>0.015</td><td>0.011</td><td>0.013</td></tr></table>

(roughly 150,000 counts per second on average) to prevent pileup, which is a non-linear effect that distorts the SPAD measurements [56]. The laser shares an optical path with a single-pixel SPAD through a beamsplitter, and a set of 2D scanning mirrors is used to raster scan the scene at a resolution of $512 \times 512$ scanpoints. A time-correlated single-photon counter is used to record the photon timestamps with a total system resolution of approximately 70 ps.

Captured dataset. We capture multiview lidar scans of six scenes by placing objects on a rotation stage in front of the scanning single-photon lidar and capturing 20 different views in increments of 18 degrees of rotation. For each lidar scan we accumulate photons during a 20 minute exposure time to minimize noise in the transient measurements. We bin the photon counts into histograms with 1500 bins and bin widths of 8 ps (all raw timestamp data will also be made available with the dataset). We set aside 10 views sampled in 36 degree increments for testing and we use 8 of the remaining views for training. Prior to input into the network for training, we normalize the measurement values by the maximum photon count observed across all views.

Calibration. We calibrate the camera intrinsics of the system using a raxel model $[57]$ with corners detected from two scans of checkerboard translated in a direction parallel to the surface normal. This model calibrates the direction of each ray individually, which is necessary because the 2D scanning mirrors deviate from the standard perspective projection model $[58]$ . Extrinsics are calibrated by placing a checkerboard on a rotation stage and solving for the axis and center of rotation that best align the 3D positions of the checkerboard corners, where the 3D points are found using the calibrated ray model along with the time of flight from the lidar (see supplemental). Overall, accurate calibration is an important and non-trivial task because multiview lidar scans provide two distinct geometric constraints (i.e. stereo disparity and time of flight) that must be consistent for scene reconstruction.

# 5 Results

We evaluate our method on the simulated and captured datasets and use transient neural radiance fields to render intensity, depth, and time-resolved lidar measurements from novel views.

Baselines. The intensity and depth rendered from our method are compared to four other baseline methods that combine point cloud-based losses with neural radiance fields. For fairer comparison and to speed up training and inference times, we implement the baselines by incorporating their loss terms into the recently introduced frameworks of NerfAcc $[31]$ and Instant-NGP $[30]$ adopted by our method. We train the following baselines using intensity images (i.e., the photon count histograms integrated over time) along with point clouds obtained from the photon count histograms using a log-matched filter, which is the constrained maximum likelihood depth estimate $[59]$ .

- Instant-NGP [30] is used to illustrate performance without additional depth supervision.   
- Depth-Supervised NeRF (DS-NeRF) [26] incorporates an additional loss term to ensure that the expected ray termination distance in volume rendering aligns with the point cloud points.   
- Urban NeRF [4] incorporates the ray-termination loss of DS-NeRF while also adding space carving losses to penalize density along rays before and after the intersection with a point cloud point.   
- Urban NeRF with masking (Urban NeRF-M) modifies Urban NeRF to incorporate an oracle object mask and extends the space carving loss to unmasked regions, providing stronger geometry regularization (additional details in supplement).

Prior to input into the network, we normalize the images and apply a gamma correction, which improves network fitting to the high dynamic range data. Finally, after training with each method, we estimate an associated depth map using the expected ray termination depth at each pixel, which is the same metric used in the loss functions of the aforementioned baselines.

# 5.1 Simulated Results

The method is compared to the baselines in simulation across five scenes: chair, ficus, lego, hot dog, and statue. In Fig. 4, we show RGB images rendered from novel views using the baselines and our proposed method after training on two, three, and five views. More extensive sets of results on all scenes are included in the supplemental. We find that views rendered from transient neural radiance fields have fewer artifacts and spurious patches of density, as the explicit supervision from the photon count histograms avoids the ill-posedness of the conventional multiview reconstruction problem.

Additional quantitative results are included in Table 1, averaged across all simulated scenes. For the evaluation of rendered RGB images, we normalize and gamma-correct the output of the proposed method and the ground truth in the same fashion as the baseline methods. Transient NeRF recovers novel views with significantly higher peak signal-to-noise ratio and better performance on the learned perceptual image patch similarity (LPIPS) metric [60] compared to baselines. Transient measurements provide explicit supervision of the unoccupied spaces in the scene, leading to fewer floating artifacts, and cleaner novel views.

The depth maps inferred from Transient NeRF are also significantly more accurate than baselines (see Fig. 5). One key advantage here is that we avoid supervision on point clouds obtained by potentially

noisy (and thus view-inconsistent) per-pixel estimates of depth. By training on the raw photon count histograms, the scene's geometry is allowed to converge to the shape that best explains all histograms across all views, resulting in much higher geometric accuracy.

Ground Truth   
![](images/53c89bc6d135785bf94c5cfe993793b0fc0fa8ad108f139e4fbb11328c6a3066.jpg)  
Urban NeRF

Instant-NGP   
![](images/9325c1cab21e490b6f2b63b460c26939fa408d6a7b2cdc8012341dd3b2749041.jpg)  
Urban NeRF-M

DS-NeRF   
![](images/72026400f939315884e77e885ebec9228db5a7a653282e4714545d578d532a2d.jpg)  
Proposed

![](images/49d15e69145802574cbc18efe6d51d4b2a3eb652c85b7d81c4c85c7719b41be4.jpg)

![](images/ef0500f84452a8ad7cbd0478726205f649b2eca336fe986df42f6182e334634a.jpg)

![](images/34cc6da3ecbb416cfdc3d8a3f42b8b0b904ed0f8050d4ac445c9156dee73e56f.jpg)  
Figure 5: Comparison of depth maps recovered from simulated measurements trained on 5 views of the lego scene.

![](images/f6cb7199a8ac7a870d71cb15f5f48a0ee3bcc765594b6b729a23f51db85c02e1.jpg)

<details>
<summary>text_image</summary>

Ground Truth
Instant-NGP
DS-NeRF
Urban NeRF
Urban NeRF-M
Proposed
rendered transients
7.1 7.5
t (ns)
2 views
cinema
food
3 views
baskets
5 views
</details>

Figure 6: Results on multiview lidar data captured with the hardware prototype and trained with 2, 3, and 5 viewpoints. For the proposed method we show the rendered transients, intensity image, and individual transients for the indicated pixels.

Table 2: Evaluation of rendered intensity images and depth on captured results. 

<table><tr><td rowspan="2">Method</td><td colspan="3">PSNR (dB)↑</td><td colspan="3">LPIPS↓</td><td colspan="3">L1 (depth) ↓</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>Instant NGP [30]</td><td>16.44</td><td>16.52</td><td>16.39</td><td>0.358</td><td>0.307</td><td>0.274</td><td>0.115</td><td>0.076</td><td>0.053</td></tr><tr><td>DS-NeRF [26]</td><td>15.34</td><td>15.05</td><td>14.86</td><td>0.311</td><td>0.312</td><td>0.325</td><td>0.048</td><td>0.036</td><td>0.036</td></tr><tr><td>Urban NeRF [4]</td><td>16.90</td><td>15.91</td><td>15.93</td><td>0.403</td><td>0.328</td><td>0.231</td><td>0.017</td><td>0.015</td><td>0.014</td></tr><tr><td>Urban NeRF w/mask [4]</td><td>15.45</td><td>18.26</td><td>19.11</td><td>0.458</td><td>0.269</td><td>0.191</td><td>0.014</td><td>0.006</td><td>0.004</td></tr><tr><td>Proposed</td><td>22.11</td><td>21.83</td><td>22.72</td><td>0.271</td><td>0.212</td><td>0.172</td><td>0.005</td><td>0.006</td><td>0.010</td></tr></table>

# 5.2 Captured Results

In Fig. 6 we show rendered novel views of intensity images from our method and baselines trained on captured data with two, three, and five views. Results are shown on the cinema, food, and baskets scenes (additional results in the supplemental). The proposed approach results in fewer artifacts and the rendered intensity images are more faithful to reference intensity images captured from the novel viewpoint. Quantitative comparisons of our method to baselines on captured data are shown in Table 2; note that we do not have access to ground truth depth for captured data and instead use depth from a log-matched filter on the ground truth transient. We find that the method outperforms the baselines in terms of PSNR and LPIPS of intensity images rendered from novel views. While performance on captured data does not improve as much as observed on simulated data with increasing numbers of viewpoints, we attribute this to small imperfections ( $\approx$ 1 mm) in the alignment of the lidar scans after estimating the camera extrinsics.

Since DS-NeRF trains explicitly on depth without additional regularization, it is especially sensitive to camera perturbations and can be outperformed in some cases by Instant NGP which has no additional geometry constraints. Our approach appears somewhat less sensitive to these issues, perhaps because geometry regularization is done implicitly through a photometric loss on the lidar measurements.

We notice some degradation in depth accuracy relative to simulation, likely due to imperfections in the estimated extrinsics. Sub-mm registration of the lidar measure-

![](images/76b3906feb6be2b4d76bad5a8ddf9a46f4f63bf4c4ea3566d465d0a7279181b5.jpg)

<details>
<summary>text_image</summary>

novel viewpoint
x-t time slice
depth
ground truth
y
x
7.0 ns
7.4 ns
rendered
</details>

Figure 7: Comparison between reference and novel views of lidar measurements, intensity slices, and depth. The x-y intensity slices are visualized for times indicated by the red dashed lines.

ments would likely improve results, but achieving such precise registration is non-trivial and beyond the scope of our current work. Finally, in Fig. 7 we compare captured measurements to rendered transients and depth rendered for the boots scene trained on 2 viewpoints. We recover the time-resolved light propagation from a novel view, shown in x-y slices of the rendered transients over time. The depth map recovered from the novel view appears qualitatively similar to the ground truth (estimated from captured measurements using a log-matched filter [17]). We show additional 3D reconstruction results in the supplemental.

# 6 Discussion

Our work brings NeRF to a new dimension of imaging at transient timescales, offering new opportunities for view synthesis and 3D reconstruction from multiview lidar. While our work is limited to modeling the direct reflection of laser light to perform lidar view synthesis, our dataset captures much richer light transport effects, including multiple bounces of light and surface reflectance properties that could open avenues for future work. In particular, the method and dataset may help enable techniques for intra-scene non-line-of-sight imaging $[61–64]$ (i.e., recovering geometry around occluders within a scene), and recovery of the bidirectional reflectance distribution function via probing with lidar measurements $[65]$ .

Our method is also limited in that we do not explore more view synthesis in more general single-photon imaging setups, such as when the lidar and SPAD are not coaxial; we hope to explore these configurations in future work. The proposed framework and the ability to render transient measurements from novel views may be especially relevant for realistic simulation for autonomous vehicle navigation, multiview remote sensing, and view synthesis of more general transient phenomena.

# Acknowledgments and Disclosure of Funding

Kiriakos N. Kutulakos acknowledges the support of the Natural Sciences and Engineering Council of Canada (NSERC) under the RGPIN and RTI programs. David B. Lindell acknowledges the support of the NSERC RGPIN program. The authors also acknowledge Gordon Wetzstein and the Stanford Computational Imaging Lab for loaning the single-photon lidar equipment.

# References

[1] Peter Seitz and Albert JP Theuwissen. Single-Photon Imaging. Springer Science & Business Media, 2011.   
[2] Ilya Chugunov, Yuxuan Zhang, Zhihao Xia, Xuaner Zhang, Jiawen Chen, and Felix Heide. The implicit values of a good hand shake: Handheld multi-frame neural depth refinement. In Proc. CVPR, 2022.   
[3] Brent Schwarz. Mapping the world in 3d. Nat. Photonics, 4(7):429–430, 2010.   
[4] Konstantinos Rematas, Andrew Liu, Pratul P Srinivasan, Jonathan T Barron, Andrea Tagliasacchi, Thomas Funkhouser, and Vittorio Ferrari. Urban radiance fields. In Proc. CVPR, 2022.   
[5] Benjamin Attal, Eliot Laidlaw, Aaron Gokaslan, Changil Kim, Christian Richardt, James Tompkin, and Matthew O'Toole. Törf: Time-of-flight radiance fields for dynamic scene view synthesis. Proc. NeurIPS, 34, 2021.   
[6] Joe C Campbell, Stephane Demiguel, Feng Ma, Ariane Beck, Xiangyi Guo, Shuling Wang, Xiaoguang Zheng, Xiaowei Li, Jeffrey D Beck, Michael A Kinch, et al. Recent advances in avalanche photodiodes. IEEE J. Sel. Top. Quantum Electron., 10(4):777–787, 2004.   
[7] Christopher V Poulton, Ami Yaacobi, David B Cole, Matthew J Byrd, Manan Raval, Diedrik Vermeulen, and Michael R Watts. Coherent solid-state lidar with silicon photonic optical phased arrays. Opt. Lett., 42(20):4091–4094, 2017.   
[8] Joshua Rapp, Julian Tachella, Yoann Altmann, Stephen McLaughlin, and Vivek K Goyal. Advances in single-photon lidar for autonomous vehicles: Working principles, challenges, and recent advances. IEEE Signal Process. Mag., 37(4):62–71, 2020.   
[9] Nils Abramson. Light-in-flight recording by holography. Opt. Lett., 3(4):121-123, 1978.   
[10] Felix Heide, Matthias B Hullin, James Gregson, and Wolfgang Heidrich. Low-budget transient imaging using photonic mixer devices. ACM Trans. Graph., 32(4):1–10, 2013.

[11] Achuta Kadambi, Refael Whyte, Ayush Bhandari, Lee Streeter, Christopher Barsi, Adrian Dorrington, and Ramesh Raskar. Coded time of flight cameras: sparse deconvolution to address multipath interference and recover time profiles. ACM Trans. Graph., 32(6):1–10, 2013.   
[12] Andreas Velten, Di Wu, Adrian Jarabo, Belen Masia, Christopher Barsi, Chinmaya Joshi, Everett Lawson, Moungi Bawendi, Diego Gutierrez, and Ramesh Raskar. Femto-photography: capturing and visualizing the propagation of light. ACM Trans. Graph., 32(4):1–8, 2013.   
[13] Matthew O'Toole, Felix Heide, David B Lindell, Kai Zang, Steven Diamond, and Gordon Wetzstein. Reconstructing transient images from single-photon sensors. In Proc. CVPR, 2017.   
[14] Ahmed Kirmani, Tyler Hutchison, James Davis, and Ramesh Raskar. Looking around the corner using transient imaging. In Proc. ICCV, 2009.   
[15] David B Lindell, Matthew O'Toole, and Gordon Wetzstein. Single-photon 3d imaging with deep sensor fusion. ACM Trans. Graph., 37(4):113–1, 2018.   
[16] Jiayong Peng, Zhiwei Xiong, Xin Huang, Zheng-Ping Li, Dong Liu, and Feihu Xu. Photon-efficient 3d imaging with a non-local neural network. In Proc. ECCV. Springer, 2020.   
[17] Joshua Rapp and Vivek K Goyal. A few photons among many: Unmixing signal and noise for photon-efficient active imaging. IEEE Trans. Comput. Imaging, 3(3):445–459, 2017.   
[18] Joshua Rapp, Yanting Ma, Robin MA Dawson, and Vivek K Goyal. High-flux single-photon lidar. Optica, 8(1):30–39, 2021.   
[19] Felix Heide, Steven Diamond, David B Lindell, and Gordon Wetzstein. Sub-picosecond photon-efficient 3d imaging using single-photon sensors. Sci. Rep., 8(1):17726, 2018.   
[20] Joshua Rapp, Robin MA Dawson, and Vivek K Goyal. Dithered depth imaging. Opt. Express, 28(23): 35143-35157, 2020.   
[21] Wolfgang Hess, Damon Kohler, Holger Rapp, and Daniel Andor. Real-time loop closure in 2d lidar slam. In Proc. ICRA, 2016.   
[22] Qin Zou, Qin Sun, Long Chen, Bu Nie, and Qingquan Li. A comparative analysis of lidar slam-based indoor navigation for autonomous vehicles. IEEE Trans. Intell. Transp. Syst., 23(7):6907–6921, 2021.   
[23] Zimo Li, Prakruti C Gogia, and Michael Kaess. Dense surface reconstruction from monocular vision and lidar. In Proc. ICRA, 2019.   
[24] Matthew J Leotta, Chengjiang Long, Bastien Jacquet, Matthieu Zins, Dan Lipsa, Jie Shan, Bo Xu, Zhixin Li, Xu Zhang, Shih-Fu Chang, et al. Urban semantic 3d reconstruction from multiview satellite imagery. In Proc. CVPR Workshops, 2019.   
[25] Stefan Lionar, Lukas Schmid, Cesar Cadena, Roland Siegwart, and Andrei Cramariuc. Neuralblox: Real-time neural representation fusion for robust volumetric mapping. In Proc. 3DV, 2021.   
[26] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised NeRF: Fewer views and faster training for free. In Proc. CVPR, 2022.   
[27] Barbara Roessle, Jonathan T Barron, Ben Mildenhall, Pratul P Srinivasan, and Matthias Nießner. Dense depth priors for neural radiance fields from sparse input views. In Proc. CVPR, 2022.   
[28] Daniele Faccio, Andreas Velten, and Gordon Wetzstein. Non-line-of-sight imaging. Nat. Rev. Phys., 2(6):318–327, 2020.   
[29] Siyuan Shen, Zi Wang, Ping Liu, Zhengqing Pan, Ruiqian Li, Tian Gao, Shiying Li, and Jingyi Yu. Non-line-of-sight imaging via neural transient fields. IEEE Trans. Pattern Anal. Mach. Intell., 43(7): 2257–2268, 2021.   
[30] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. (SIGGRAPH), 41(4):1–15, 2022.   
[31] Ruilong Li, Matthew Tancik, and Angjoo Kanazawa. Nerfacc: A general nerf acceleration toolbox. arXiv preprint arXiv:2210.04847, 2022.   
[32] Franco Zappa, Simone Tisa, Alberto Tosi, and Sergio Cova. Principles and features of single-photon avalanche diode arrays. Sensors and Actuators A: Physical, 140(1):103–112, 2007.   
[33] Chandra M Natarajan, Michael G Tanner, and Robert H Hadfield. Superconducting nanowire single-photon detectors: physics and applications. Superconductor Sci. Technol., 25(6):063001, 2012.   
[34] P Buzhan, B Dolgoshein, L Filatov, A Ilyin, V Kantzerov, V Kaplin, A Karakash, F Kayumov, S Klemin, E Popova, et al. Silicon photomultiplier and its possible applications. Nucl. Instrum. Methods. Phys. Res. A, 504(1-3):48–52, 2003.   
[35] Dongeek Shin, Feihu Xu, Dheera Venkatraman, Rudi Lussana, Federica Villa, Franco Zappa, Vivek K Goyal, Franco NC Wong, and Jeffrey H Shapiro. Photon-efficient imaging with a single-photon camera. Nat. Commun., 7(1):12046, 2016.

[36] Julián Tachella, Yoann Altmann, Nicolas Mellado, Aongus McCarthy, Rachael Tobin, Gerald S Buller, Jean-Yves Tourneret, and Stephen McLaughlin. Real-time 3d reconstruction from single-photon lidar data using plug-and-play point cloud denoisers. Nat. Commun., 10(1):4984, 2019.   
[37] Genevieve Gariepy, Nikola Krstajić, Robert Henderson, Chunyong Li, Robert R Thomson, Gerald S Buller, Barmak Heshmat, Ramesh Raskar, Jonathan Leach, and Daniele Faccio. Single-photon sensitive light-in-fight imaging. Nat. Commun., 6(1):6021, 2015.   
[38] David B Lindell, Matthew O'Toole, and Gordon Wetzstein. Towards transient imaging at interactive rates with single-photon detectors. In Proc. ICCP, 2018.   
[39] Joshua Rapp, Charles Saunders, Julián Tachella, John Murray-Bruce, Yoann Altmann, Jean-Yves Tourneret, Stephen McLaughlin, Robin MA Dawson, Franco NC Wong, and Vivek K Goyal. Seeing around corners with edge-resolved transient imaging. Nat. Commun., 11(1):5929, 2020.   
[40] Shumian Xin, Sotiris Nousias, Kiriakos N Kutulakos, Aswin C Sankaranarayanan, Srinivasa G Narasimhan, and Ioannis Gkioulekas. A theory of fermat paths for non-line-of-sight shape reconstruction. In Proc. CVPR, 2019.   
[41] Wenzheng Chen, Fangyin Wei, Kiriakos N Kutulakos, Szymon Rusinkiewicz, and Felix Heide. Learned feature embeddings for non-line-of-sight imaging and recognition. ACM Trans. Graph. (SIGGRAPH), 39(6):1–18, 2020.   
[42] David B Lindell and Gordon Wetzstein. Three-dimensional imaging through scattering media based on confocal diffuse tomography. Nat. Commun., 11(1):4517, 2020.   
[43] Yongyi Zhao, Ankit Raghuram, Hyun K Kim, Andreas H Hielscher, Jacob T Robinson, and Ashok Veeraraghavan. High resolution, deep imaging using confocal time-of-flight diffuse optical tomography. IEEE Trans. Pattern Anal. Mach. Intell., 43(7):2206–2219, 2021.   
[44] Rachael Tobin, Abderrahim Halimi, Aongus McCarthy, Martin Laurenzis, Frank Christnacher, and Gerald S Buller. Three-dimensional single-photon imaging through obscurants. Opt. Express, 27(4):4590–4611, 2019.   
[45] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. In Proc. NeurIPS, 2022.   
[46] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proc. CVPR, 2022.   
[47] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proc. CVPR, 2021.   
[48] Samarth Sinha, Jason Y Zhang, Andrea Tagliasacchi, Igor Gilitschenski, and David B Lindell. Sparsepose: Sparse-view camera pose regression and refinement. In Proc. CVPR, 2022.   
[49] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM, 65(1):99–106, 2021.   
[50] James T Kajiya and Brian P Von Herzen. Ray tracing volume densities. ACM SIGGRAPH, 18(3):165–174, 1984.   
[51] Nelson Max. Optical models for direct volume rendering. IEEE Trans. Vis. Comput. Graph., 1(2):99-108, 1995.   
[52] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan, and Jonathan T. Barron. NeRF in the dark: High dynamic range view synthesis from noisy raw images. CVPR, 2022.   
[53] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proc. ICLR, 2015.   
[54] Diego Royo, Jorge García, Adolfo Muñoz, and Adrian Jarabo. Non-line-of-sight transient rendering. Computers & Graphics, 107:84–92, 2022.   
[55] Delio Vicini, Sébastien Speierer, and Wenzel Jakob. Differentiable signed distance function rendering. ACM Trans. Graph., 41(4):1-18, 2022.   
[56] Joshua Rapp, Yanting Ma, Robin MA Dawson, and Vivek K Goyal. Dead time compensation for high-flux ranging. IEEE Trans. Signal Process., 67(13):3471–3486, 2019.   
[57] Michael D Grossberg and Shree K Nayar. The raxel imaging model and ray-based calibration. Int. J. Comput. Vis., 61(2):119, 2005.   
[58] Peter Eisert, Konrad Polthier, and Joachim Hornegger. A mathematical model and calibration procedure for galvanometric laser scanning systems. In Vision, Modeling, and Visualization, pages 207-214, 2011.   
[59] Ahmed Kirmani, Dheera Venkatraman, Dongeek Shin, Andrea Colaço, Franco NC Wong, Jeffrey H Shapiro, and Vivek K Goyal. First-photon imaging. Science, 343(6166):58–61, 2014.

[60] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proc. CVPR, 2018.   
[61] Andreas Velten, Thomas Willwacher, Otkrist Gupta, Ashok Veeraraghavan, Moungi G Bawendi, and Ramesh Raskar. Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging. Nat. Commun., 3(1):745, 2012.   
[62] Matthew O'Toole, David B Lindell, and Gordon Wetzstein. Confocal non-line-of-sight imaging based on the light-cone transform. Nature, 555(7696):338–341, 2018.   
[63] David B Lindell, Gordon Wetzstein, and Matthew O'Toole. Wave-based non-line-of-sight imaging using fast fk migration. ACM Trans. Graph., 38(4):1–13, 2019.   
[64] Xiaochun Liu, Ibón Guillén, Marco La Manna, Ji Hyun Nam, Syed Azer Reza, Toan Huu Le, Adrian Jarabo, Diego Gutierrez, and Andreas Velten. Non-line-of-sight imaging using phasor-field virtual wave optics. Nature, 572(7771):620–623, 2019.   
[65] Nikhil Naik, Shuang Zhao, Andreas Velten, Ramesh Raskar, and Kavita Bala. Single view reflectance capture using multiplexed scattering and time-of-flight imaging. ACM Trans. Graph. (SIGGRAPH Asia), 30(6):1–10, 2011.

# Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction -Supplementary Material-

Anagh Malik $^{1,2}$ anagh@cs.toronto.edu

Parsa Mirdehghan $^{1,2}$ parsa@cs.toronto.edu

Sotiris Nousias $^{1}$ sotiris@cs.toronto.edu

Kiriakos N. Kutulakos $^{1,2}$ kyros@cs.toronto.edu

David B. Lindell $^{1,2}$ lindell@cs.toronto.edu

$^{1}$ University of Toronto $^{2}$ Vector Institute

# 1 Hardware Prototype

To create the captured dataset, we built a hardware prototype consisting of a pulsed laser (NKT Photonics Katana 05HP) operating at $532\mathrm{nm}$ that emits 35 ps pulses of light at a repetition rate of 10 MHz. The output power of the laser is lowered to $< 1\mathrm{mW}$ to keep the flux low enough (roughly 150,000 counts per second on average) to prevent pileup, which is a non-linear effect that distorts the SPAD measurements [1]. The laser emits polarized light that passes through a polarizing beam splitter (Thorlabs PBS251), to a set of 2D scanning mirrors (Thorlabs GVS012). The mirrors are controlled by a multifunction I/O device (NI-DAQ USB-6343) and are used to scan scenes at a spatial resolution of $512\times 512$ at a rate of 0.1 frames per second. The laser shares an optical path through the beam splitter with a single pixel SPAD (Micro Photon Devices PDM series SPAD) with a $50~\mu \mathrm{m} \times 50$ $\mu \mathrm{m}$ active pixel area. Photons detected by the SPAD are correlated with a sync signal from the laser using a time-correlated single photon counter (TCSPC) to measure the photon arrival timestamps.

We place the scanned scenes on a rotation stage (Parker Motion/Parker 6K4 Compumotor) in front of the scanning single-photon lidar, allowing us to capture different viewpoints by rotating the scene.

Each scan consists of 20 minutes of total exposure time, but we save out all collected photon timestamps individually so that any desired exposure time can be emulated by accumulating photons over the desired time window in post processing (i.e., for future applications of the dataset). We set the bin width of the photon count histograms to 8 ps and the number of bins is 1500. The entire data acquisition is controlled using custom-developed MATLAB software on a desktop PC, and we capture six scenes with varying geometry, texture, and material properties. For each scene, we capture views in 18 degree increments of the rotation stage, resulting in a 360 degree capture. We set aside 8 views for training, and 10 separate views sampled in 36 degree increments comprise the test split. Prior to input into the network for training, we normalize the measurement values by the maximum photon count observed across all views.

Ambient illumination. Our hardware setup is relatively robust to ambient illumination because we place a laser-line spectral filter (Thorlabs FL532-10) in front of the SPAD, which attenuates ambient illumination by a factor of 10,000. Under indoor lighting, we observe roughly 300-3,000 photon counts per second, depending on the albedo of the target, which is $< 2\%$ of the detected laser photons (150,000 counts per second). However, operation in much brighter environments (e.g., outdoors under direct sunlight) would result in non-negligible background counts. In that case, it is possible to use

other laser wavelengths (e.g., 1550 nm), where sunlight is heavily attenuated because of absorption by the atmosphere.

# 2 Calibration of the Hardware Prototype

Here we describe the method used to find the extrinsics and intrinsics defining the captured dataset. An overview of the method can be found in the Algorithm 1.

Intrinsics. We calibrate the camera intrinsics of the system using the raxel model $[2]$ , which maps each pixel to a 3D ray direction. To find the ray directions, we place a checkerboard on a translation stage and use the lidar system to capture two images of this checkerboard before and after translating 17 cm by in the direction of the surface normal of the board (this was the maximum distance permitted by the translation stage and our optical table layout).

Checkerboard corners are detected in the two images using OpenCV's findChessboardCorners [3] with subpixel refinement. In order to improve robustness to distortion, we further refine the corner detection by fitting a second-order polynomial to corners along vertical lines (which we found to be a good fit to model the observed distortion), and we use a first-order fit to horizontal lines, which we found to fit the data well. We set the corner positions to the intersections of the set of fitted lines.

The pixel coordinates of the detected checkerboard corners are then used to define a 3D coordinate system. We set the origin of the coordinate system to the upper-left corner of the nearest checkerboard, and coordinates on the far checkerboard are calculated based on the known translation in z, with x and y coordinates given using the size of the checkerboards (4.2 mm). We associate ray directions to each pixel by finding the points of intersection of the ray with the two checkerboards. Specifically, we linearly interpolate the mapping from pixel values to 3D coordinates at each checkerboard corner to retrieve a dense mapping from pixels to 3D coordinates on each board. Then, for a given pixel the ray direction is given by a simple subtraction of the 3D intersection points on each board.

Extrinsics. Our captured dataset consists of photon count histograms from six different scenes, each with 20 different views. We use a rotation stage to move the scenes one full revolution in 18 degree increments. The resulting views are equivalent to capturing a stationary scene with cameras that are rotated about the center of rotation of the stage. Since all scenes are captured identically in this fashion, we determine the camera extrinsics once for all scenes.

We also calibrate for an additional offset parameter to finetune the time-of-flight-delays recorded by the lidar system. This accounts for the unknown time delay between the time that the time-correlated single-photon counter receives the sync signal from the laser and the time that the laser pulse reaches the center of projection of the scanning mirrors. In other words, this offset accounts for when “time zero” should occur in the photon count histograms; we roughly calibrate for this value by placing a target directly in front of the galvo mirrors, and we fine tune this offset via optimization as detailed below.

Given that each view is captured using a controlled rotation, the extrinsics can be determined by identifying the 3D axis of rotation and 3D center of rotation. To estimate these parameters, we use a two step procedure. First, we capture lidar scans of a checkerboard placed on the rotation stage and rotated to 6 different positions in 9 degree increments. We convert the lidar scans to a point cloud using the raxel model and lidar time of flight, and then a coarse solution is obtained by fitting planes to the point clouds and finding the center and axis of rotation that align the plane normals. Second, we detect corners of the checkerboards and find the corresponding 3D points. Then, we optimize for the center of rotation, axis of rotation, and the 1D offset parameter that best align the 3D checkerboard corners.

Specifically, we implement a routine in PyTorch [4] to align the 3D checkerboard corners using the Rodrigues formula given below.

$$
\mathbf {v} ^ {\prime} = (\mathbf {v} - \mathbf {c}) \cos \theta + (\mathbf {a} \cdot (\mathbf {v} - \mathbf {c})) (1 - \cos \theta) \mathbf {a} + (\mathbf {a} \times (\mathbf {v} - \mathbf {c})) \sin \theta + \mathbf {c}. \tag {S1}
$$

Here, v is the 3D point to be rotated, c is the center of rotation, a is the axis of rotation, and $v'$ is the rotated point. We minimize the objective function given as

$$
\mathcal {L} _ {\text { align }} = \sum_ {p} \sum_ {[ i, j ] \in \mathcal {C}} \| \mathbf {v} _ {p} ^ {\prime (i)} - \mathbf {v} _ {p} ^ {\prime (j)} \| _ {2} ^ {2}, \tag {S2}
$$

where p indexes the checkerboard corner points and $[i, j]$ index all pairs of checkerboards. Thus we penalize the distance between corresponding points between all checkerboards. The optimization is performed using LBFGS [5]. After optimization, we use the resulting center point and rotation matrices (i.e., by rotating around a by increments of 18 degrees) to define the camera extrinsics for each view.

# Algorithm 1: Calibration overview.

# Intrinsics Calibration

Data: Two scans of a checkerboard before and after translation.

1. Perform sub-pixel detection of checkerboard corners using OpenCV and polynomial line fitting.   
2. Use the detected checkerboard corners to initialize a 3D coordinate system.   
3. Compute per-pixel 3D coordinates for each checkerboard scan by interpolating the mapping from corner pixel coordinates to 3D coordinates.   
4. Compute the ray directions by subtracting the per-pixel 3D coordinates obtained for the checkerboard scan before and after translation.

# Extrinsics Calibration

Data: Six scans of a checkerboard rotated using the rotation stage.

1. Convert lidar scans to point clouds using the time of flight and intrinsics.   
2. Fit planes and determine initial center and axis of rotation to align the surface normals.   
3. Detect 3D points corresponding to checkerboard corners.   
4. Refine the initial center and axis of rotation via Equations S1 and S2.

# 3 Additional Implementation Details

Network Architecture. In this section we describe the main network architecture. We extend the NerfAcc [6] framework and the provided implementation of Instant-NGP (INGP) [7]. All network hyperparameters are shared between our proposed method and the baselines unless otherwise stated. We set the number of hash feature grids for INGP to 16 and set the feature size to 2. The resolution of the coarsest grid is set to 16 and each subsequent grid has 2 times finer resolution. The base MLP has width 64 with 1 hidden layer to map to density and a latent vector. Another MLP with 2 hidden layers and 64 hidden units maps from the latent vector and viewing direction to the view-dependent radiance.

We use the occupancy grid from NerfAcc with a resolution of $128^{3}$ . The occupancy grid is employed to remove samples along the ray based on their density for the sake of efficiency. The grid is binarized using a occupancy value of $10^{-3}$ .

For the captured data, we find that setting the binarization threshold to $10^{-5}$ for the first 3,000 iterations before reverting back to the standard $10^{-3}$ value helps to avoid an overly aggressive removal of density that erodes the surface of reconstructed objects. In addition, we incorporate pruning based on transmittance, removing any samples that register a transmittance of 0 to speed up rendering. Finally, all rays are rendered by sampling 4,096 points along each ray, and these points are pruned according to their occupancy and transmittance values, as previously discussed.

We set the bounding box used in INGP as follows for the simulated and captured data. For the simulated data the bounding box extents are set to -1.5 to 1.5 across all dimensions and methods. For captured data, we use a -0.4 to 0.4 bounding box for Transient NeRF; we slightly shrink the bounding box for baselines run on captured data to -0.3 to 0.3, which we find helps remove some spurious regions of density and improves the baseline results.

Rendering Equation Given samples $t_{1}, t_{2}, \cdots, t_{N}$ along a camera ray, we render a histogram bin $\tau[n]$ for the ray as

$$
\tau [ n ] = \sum_ {i \mid \frac {t _ {i} + t _ {i + 1}}{2} \in \mathcal {T} _ {n}} T _ {i} ^ {2} \left(1 - \exp \left(- \sigma_ {i} \delta_ {i}\right)\right) \frac {\mathbf {c} _ {i}}{\left(\frac {1}{2} (t _ {i} + t _ {i + 1})\right) ^ {2}}, \text {   where   } T _ {i} = \exp \left(- \sum_ {j = 1} ^ {i - 1} \sigma_ {j} \delta_ {j}\right), \tag {S3}
$$

where $\sigma_{i}, c_{i}$ are the density and radiance outputs of the network, $\delta_{i} = t_{i+1} - t_{i}$ is the distance between consecutive samples and $T_{n}$ is the resolution of the histogram bin, in units of distance. The sum is taken over all consecutive samples whose midpoints fall within $T_{n}$ .

Optimization settings and run time. The time required for training is strongly influenced by the number of samples used for the spatial filter, as discussed in the main text. Specifically, each image pixel is rendered by computing a weighted integral over the radiance for a particular region of space. We compute this integral during training by stochastically sampling one or more ray directions per pixel with probability determined by the weighting and support of the spatial filter $[8]$ . For the simulated dataset, we initially sample a single ray for the first 2000 iterations, then subsequently double this number every 2000 iterations until we reach a maximum of 30 rays sampled per pixel. For the captured dataset we use a single ray sample per pixel per iteration; we find that increasing the number ray samples results in longer training times without significantly improved performance.

Training takes roughly eight hours to converge for the simulated dataset (250K iterations) and two hours to converge for captured data (150K steps).

Depth calculation. We follow the NeRF convention in calculating depth for all baseline methods as

$$
d = \sum_ {i} \sigma (t _ {i}) T (t _ {i}) \frac {t _ {i} + t _ {i + 1}}{2}, \tag {S4}
$$

where $t_{i}$ is the distance from the ray origin to a sample along the ray, $T_{i}$ is the transmittance, and $\sigma$ is the density. Intuitively, this equation calculates the expected ray termination distance.

Since our method incorporates a spatial filter to more accurately model the footprint of the illumination spot, a single ray can result in measurements from multiple surfaces (e.g., if a pixel integrates over a depth discontinuity). To avoid multiple depths in the measurement from skewing the estimate of the expected ray termination distance, we use the following equation to compute depth.

$$
d = \arg \max _ {t _ {i}} \sigma (t _ {i}) T (t _ {i}). \tag {S5}
$$

Thus, we find the depth at which the maximum probability of ray termination occurs.

We note that in the conventional NeRF depth rendering formula of Equation S4, regions of low density become less visible in most visualizations because their depth tends towards zero (i.e., it is weighted by the density, which is close to zero, but typically not uniformly zero for empty regions). However, in Equation S5, no such weighting exists, resulting in visualizations that appear noisier because depths for regions with low (but non-zero) density are not automatically suppressed. Thus for visualization of the depth maps we weight the depth as follows.

$$
d = \left(\sum_ {i} \sigma (t _ {i}) T (t _ {i})\right) \arg \max _ {t _ {i}} \sigma (t _ {i}) T (t _ {i}). \tag {S6}
$$

We use this equation to visualize depth maps, as we find they are more comparable to those rendered with the conventional NeRF formulation (though the density and transmittance weighting results in them being slightly less accurate).

# 3.1 Baseline Implementation Details

Four different baselines are implemented to evaluate our proposed method. To isolate the impact of different loss functions and speed up the training, we implemented the specific loss functions from each of the following methods in the framework of NerfAcc with the Instant-NGP backbone.

For all baselines, we generate the ground truth intensity images by first integrating the transients over the time dimension and normalization by a scale factor to shift the image values to lie close to within

[0, 1]. We apply gamma correction to tonemap the resulting high dynamic range intensity images prior to training.

Instant-NGP [7]. The Instant-NGP model is trained with photometric loss defined as the total squared error between the rendered intensities and the pixel colors from the input images:

$$
\mathcal {L} _ {\text { photo }} = \sum_ {\mathbf {r} \in \mathcal {R}} \| \widetilde {\mathbf {C}} (\mathbf {r}) - \mathbf {C} (\mathbf {r}) \| _ {2} ^ {2}, \tag {S7}
$$

where $\widetilde{C}$ and C denote the ground-truth pixel color and the predicted value, respectively. R specifies the set of active rays that have non-zero opacity values; this set of rays is updated as training progresses to accelerate optimization by pruning rays that do not contribute to the rendering [6].

Depth-Supervised NeRF [9]. Following the depth supervision idea proposed in [9], we train a new model that incorporates depth error loss in addition to the photometric loss above:

$$
\mathcal {L} _ {\text { depth }} = \frac {1}{| \mathcal {R} ^ {\prime} |} \sum_ {\mathbf {r} \in \mathcal {R} ^ {\prime}} \left(\widetilde {d} (\mathbf {r}) - d (\mathbf {r})\right) ^ {2}. \tag {S8}
$$

Here, $\widetilde{d} (\mathbf{r})$ denotes the lidar depth for ray $\mathbf{r}$ obtained from the captured transients using a log-matched filter [10], $d(\mathbf{r})$ is the predicted depth value computed from Equation S4, and $\mathcal{R}'$ specifies the set of rays that intersect with the object.

The final training loss is defined as $\mathcal{L} = \mathcal{L}_{\mathrm{photo}} + \lambda_{\mathrm{depth}}\mathcal{L}_{\mathrm{depth}}$ . In the simulated and captured results, $\lambda_{\mathrm{depth}}$ is set to 0.005 and 0.0075, respectively.

Urban NeRF [11]. We further incorporated the line-of-sight lidar loss $L_{sight}$ , proposed in [11], to encourage the densities to be concentrated near the lidar points. This loss comprises two terms. The first term penalizes any density between the ray origin and the lidar point:

$$
\mathcal {L} _ {\text { empty }} = \frac {1}{| \mathcal {R} ^ {\prime} |} \sum_ {\mathbf {r} \in \mathcal {R} ^ {\prime}} \left[ \int_ {t _ {n}} ^ {\widetilde {d} - \epsilon} w (t) ^ {2} d t \right]. \tag {S9}
$$

In this equation, $R'$ is the set of rays that intersect with the object; $w(t)$ is the volume rendering integration weights defined as $\sigma(t)T(t)$ ; $t_{n}$ denotes the near bound of the ray; and $\epsilon$ specifies a neighbourhood around the surface point.

The second term, on the other hand, encourages the model to increase the densities in the bounded region around the surface point:

$$
\mathcal {L} _ {\text { near }} = \frac {1}{| \mathcal {R} ^ {\prime} |} \sum_ {\mathbf {r} \in \mathcal {R} ^ {\prime}} \left[ \int_ {\widetilde {d} - \epsilon} ^ {\widetilde {d} + \epsilon} (w (t) - \mathcal {K} _ {\epsilon} (t - \widetilde {d})) ^ {2} d t \right], \tag {S10}
$$

where $\mathcal{K}_{\epsilon}$ is a truncated Gaussian defined as $\mathcal{N}(0, (\epsilon/3)^2)$ [12].

The final training loss for this model is defined as:

$$
\mathcal {L} = \mathcal {L} _ {\text { photo }} + \lambda_ {\text { depth }} \mathcal {L} _ {\text { depth }} + \lambda_ {\text { sight }} \mathcal {L} _ {\text { sight }}, \tag {S11}
$$

where $\mathcal{L}_{\mathrm{sight}} = \mathcal{L}_{\mathrm{empty}} + \mathcal{L}_{\mathrm{near}}$ . The parameters $(\lambda_{\mathrm{depth}}, \lambda_{\mathrm{sight}})$ are set to (0.0001, 0.005) for both the simulations and captured experiments.

Following the instructions from the original paper, we applied exponential decay to the value of $\epsilon$ throughout the optimization, which encourages the density of the radiance field to fall within a progressively smaller support, improving convergence. We initialize $\epsilon$ to a value $\epsilon_{max}$ , and every 7000 steps we multiply by a factor of 0.8, until it decays to $\epsilon_{min}$ . Parameters ( $\epsilon_{max}, \epsilon_{min}$ ) are set to (1.5, 0.025) and (4.0, 0.05) for the captured and simulated results, respectively.

Urban NeRF with Masking. The depth-related loss terms $L_{depth}$ and $L_{sight}$ used in the previous two models are evaluated only for the rays that intersect with the object; however, we find that the included regularization terms do not prevent spurious regions of density that appear for pixels that fall outside the support of the object on the image plane.

We attempt to improve the baseline results further by using modified versions of $L_{photo}$ and $L_{sight}$ which sum over active rays R. Specifically we use an oracle object mask (i.e., a ground truth mask

segmenting the object in each training view) and set the ground truth depth of all background pixels to zero. This loss encourages the density to be uniformly zero along background rays and helps to remove spurious clouds of density that tend to materialize in the other baseline results during training. We note that this masking loss is directly analogous to the sky modeling loss proposed in Urban-NeRF [11], which penalizes density along rays that intersect with the sky as determined using a semantic segmentation network.

For both the simulated and captured results, we hand-annotate the oracle object masks by carefully thresholding the intensity images and applying morphology operations to fill in holes in the mask.

# 4 Supplemental Results

# 4.1 Dataset

Simulated dataset. To create the simulated multiview lidar dataset, we modify the non-line-of-sight Mitsuba 2 rendering codes of Royo et al. [13]. While the original codebase enables one to simulate the effect of illuminating a single point in the scene with a laser and imaging other points, our approach requires rendering a frame where each pixel images an area of the scene that is illuminated with a coaxial, collimated light source. To this end, introduce a new light source as well as additional rendering options such that each pixel is rendered independently with its own coaxial light source. We also incorporate a Gaussian reconstruction filter [8] along the time dimension to avoid aliasing or stair-stepping artifacts in regions with fine variations in depth.

We choose the variance of the Gaussian temporal filter and spatial filters to produce transients that measure a laser pulse with similar temporal and spatial support as the captured results. Specifically, we set the variance of the temporal and spatial filters to 3 and 0.15 respectively. We simulate photon count histograms 1200 bins and each bin has a width of approximately $\sim30ps$ .

We subsample 8 training viewpoints from the simulated lidar measurements rendered at 36 degree increments along a circle centered at the origin. Specifically, we sample 2 views separated by 180 degrees; 3 views separated by 90 and 180 degrees (i.e., a superset of the 2 views), and 5 views uniformly separated by 72 degrees. Note that the 5 views are not a superset of the viewpoints used when training on 2 or 3 lidar measurements. For testing, we use 6 viewpoints from the NeRF Blender test set that surround the object.

Captured dataset. The captured dataset consists of 20 multiview single-photon lidar scans of six scenes. The scenes consist of everyday objects and figurines (see Fig. S1). We record photon timestamps at 4 ps resolution for each scene, with measurements of each view being captured during an exposure period of 20 minutes. While we use all photon timestamps in the photon count histograms used for the proposed method, access to the photon timestamps also allows synthesizing histograms with arbitrary exposure time, which will make the dataset useful for a wide array of follow-on work.

We use raw photon count histograms with 4096 bins and bin widths of 4 ps. To decrease the memory required for training, we crop and downsample the histograms to 1500 bins with 8 ps resolution.

Measurements are captured in the low-flux regime to avoid non-linear distortions due to pile-up. A detailed breakdown of the photon counts and acquisition parameters for each captured scene is provided in Table S1.

Table S1: Photon counts per scene for the captured dataset. 

<table><tr><td>scene name</td><td>spatial resolution</td><td>histogram bins</td><td>exposure time/view</td><td>avg. counts/view</td><td>avg. counts/sec</td></tr><tr><td>baskets</td><td> $512 \times 512$ </td><td>1500</td><td>20 min</td><td> $9.04 \times 10^{7}$ </td><td> $7.53 \times 10^{4}$ </td></tr><tr><td>boots</td><td></td><td></td><td></td><td> $7.39 \times 10^{7}$ </td><td> $6.16 \times 10^{4}$ </td></tr><tr><td>carving</td><td></td><td></td><td></td><td> $5.60 \times 10^{7}$ </td><td> $4.66 \times 10^{4}$ </td></tr><tr><td>chef</td><td></td><td></td><td></td><td> $2.56 \times 10^{8}$ </td><td> $2.13 \times 10^{5}$ </td></tr><tr><td>cinema</td><td></td><td></td><td></td><td> $1.51 \times 10^{8}$ </td><td> $1.26 \times 10^{5}$ </td></tr><tr><td>food</td><td></td><td></td><td></td><td> $1.44 \times 10^{8}$ </td><td> $1.20 \times 10^{5}$ </td></tr></table>

![](images/d9c42cbefde96571b24d0bd8bb3c41f028cd2c63ba2bf3ed3d56f8e6f13b3a19.jpg)  
Figure S1: The captured dataset. The captured dataset consists of 20 multiview single-photon lidar scans of six scenes.

# 4.2 Depth Evaluation

DS-NeRF and Urban NeRF calculate depth differentiably using the integral along the ray (i.e., the expected ray termination distance) and incorporate this in their loss functions. Since the baseline methods are supervised with this depth estimate, we opted to use the same approach for evaluation.

For completeness, we provide an updated version of the L1 depth in Table S2 calculated for all methods using the maximum ray termination probability as described in Equation S5. Baseline performance using this method of depth estimation is indeed improved for most of the methods, though Transient NeRF still performs best.

Table S2: Depth evaluation using maximum probability of ray termination for simulated and captured scenes. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Simulated ↓</td><td colspan="3">Captured ↓</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>Instant NGP [30]</td><td>0.151</td><td>0.202*</td><td>0.184*</td><td>0.021</td><td>0.021</td><td>0.024</td></tr><tr><td>DS-NeRF [26]</td><td>0.102</td><td>0.124*</td><td>0.136*</td><td>0.024</td><td>0.026</td><td>0.033</td></tr><tr><td>Urban NeRF [4]</td><td>0.128</td><td>0.124</td><td>0.100</td><td>0.017</td><td>0.016*</td><td>0.013</td></tr><tr><td>Urban NeRF w/mask†</td><td>0.039</td><td>0.038</td><td>0.020</td><td>0.014</td><td>0.006</td><td>0.004</td></tr><tr><td>Proposed</td><td>0.015</td><td>0.011</td><td>0.013</td><td>0.006</td><td>0.006</td><td>0.010</td></tr></table>

\* Using maximum probability of ray termination is worse than the expected termination distance (Equation S4).

$^{\dagger}$ Requires ground truth segmentation mask.

# 4.3 Intermediate Rendering Results

In Fig. S2, we show intermediate results from the rendering pipeline: the density from the network and the raw radiance output. We plot all these quantities versus the histogram bin number, which we find by simply discretizing the distances of the samples along the rays and allocating the quantities to the appropriate bin.

The rendered densities are very dissimilar from the rendered transients, partly due to the transients being effected by the transmittance, but also because these values are visualized before applying the temporal filter.

Note these plots show renders for a single ray, and we are not integrating over multiple rays to account for the pixel footprint. Also, we plot the normalized density and radiance along the ray since these quantities vary significantly across different pixels in the image.

![](images/7a005cc8365102dc7342180f70e9504cb5c0975b249318fa3ed9a5558806ffef.jpg)

<details>
<summary>line</summary>

| Subplot Type       | X Value | Y Value |
| ------------------ | ------- | ------- |
| rendered transients | 650     | ~8e+6   |
| rendered transients | 750     | ~4e+5   |
| rendered transients | 850     | ~1e+8   |
| densities          | 650     | ~8e+6   |
| densities          | 750     | ~4e+5   |
| densities          | 850     | ~1e+8   |
| radiance          | 650     | ~8e+6   |
| radiance          | 750     | ~4e+5   |
| radiance          | 850     | ~1e+8   |
</details>

Figure S2: Rendered transients, densities, and radiance plotted versus bin number for rays represented in the rendered image of the hotdog scene trained on three views. We normalize the densities and radiance values for visualization and plot the unnormalized transients.

# 4.4 Ablation Studies

Space carving, temporal filter and normals. In Table S3 we show ablation studies of our method calculated on the captured cinema scene. Specifically, we include quantitative results of our method without the space carving loss (w/o SC), without accounting for the laser profile (w/o TF) and the proposed method while accounting for normals (w/ normals).

The space carving loss appears to especially improve performance in the five view case. It helps eliminate spurious clouds of density, which improves reconstruction quality for held-out test views.

We also note the importance of explicitly accounting for the laser pulse width and system jitter. This is especially noticeable for the two view results, where the depth becomes highly skewed without this component. We attribute this effect to a thickening of the density representing the object surface; the optimization appears to converge to this solution to explain the temporal spread of the returning light captured in the photon count histograms. Properly accounting for the laser pulse and system jitter essentially deconvolves the temporal response of the lidar system, resulting in thin sheets of density that accurately localize the object surface.

We also ablate using the normals by adding cosine factors to the rendering equation (Eq. 3) for the Cinema scene. Estimating the normals requires a noisy finite-difference operation on the predicted density values, which we found results in speckle-like artifacts in the novel views. Instead, we model the effects of incidence angle by conditioning the neural representation on view direction.

Table S3: Ablation studies on the proposed method for the captured cinema scene. We present results while omitting any temporal filtering (w/o TF), omitting the space carving loss (w/o SC) and with modelling normals using cosine factors (w/ normals). 

<table><tr><td rowspan="2">Method</td><td colspan="3">PSNR (dB)↑</td><td colspan="3">LPIPS↓</td><td colspan="3">SSIM↑</td><td colspan="3">L1 (depth) ↓</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>Proposed w/o TF</td><td>16.55</td><td>20.85</td><td>20.02</td><td>0.346</td><td>0.225</td><td>0.209</td><td>0.589</td><td>0.837</td><td>0.823</td><td>0.022</td><td>0.007</td><td>0.009</td></tr><tr><td>Proposed w/o SC</td><td>20.63</td><td>21.09</td><td>20.12</td><td>0.207</td><td>0.200</td><td>0.217</td><td>0.855</td><td>0.855</td><td>0.840</td><td>0.007</td><td>0.009</td><td>0.020</td></tr><tr><td>Proposed w/ normals</td><td>20.88</td><td>20.41</td><td>24.62</td><td>0.332</td><td>0.268</td><td>0.216</td><td>0.825</td><td>0.815</td><td>0.867</td><td>0.006</td><td>0.007</td><td>0.007</td></tr><tr><td>Proposed</td><td>21.61</td><td>21.66</td><td>25.12</td><td>0.281</td><td>0.245</td><td>0.178</td><td>0.850</td><td>0.812</td><td>0.879</td><td>0.006</td><td>0.006</td><td>0.007</td></tr></table>

Pixel footprint. We ablate modelling the pixel footprint (see Table S4). When the laser spot and sensor footprint pass over a depth discontinuity, we observe two peaks in the resulting photon count histogram (corresponding to the two depths across the discontinuity). Assuming an ideal spot (i.e., using a single ray) completely fails to model this effect. Training and rendering the lego scene (which has many depth discontinuities) with the ideal spot model results in much worse performance across novel views.

Table S4: Simulated results comparing the proposed approach on the Lego scene with and without modeling the laser spatial footprint (w/o SF). 

<table><tr><td rowspan="2">Method</td><td colspan="3">PSNR (dB) ↑</td><td colspan="3">LPIPS ↓</td><td colspan="3">L1 (depth) ↓</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>Proposed w/o SF</td><td>20.01</td><td>23.39</td><td>25.08</td><td>0.195</td><td>0.147</td><td>0.170</td><td>0.119</td><td>0.065</td><td>0.041</td></tr><tr><td>Proposed</td><td>20.64</td><td>23.63</td><td>25.18</td><td>0.190</td><td>0.161</td><td>0.192</td><td>0.023</td><td>0.011</td><td>0.008</td></tr></table>

HDR-informed loss function. We provide an ablation study of the HDR-informed loss function in Table S5 on the chef and food scenes. Incorporating this loss provides some improvement by preventing very bright regions from dominating the loss. Moreover, in the case of 5 input views, performance without the HDR-informed loss drops due to specular highlights appearing in one view, but not an overlapping nearby training view (e.g., on the globe in the chef scene or the bag of chips in the food scene). Without the HDR-informed loss, the network has difficulty modeling the large variation in radiance between views, and so the optimization produces spurious patches of density to model this view-dependent effect (despite regularization with the space carving loss).

Table S5: Ablation study of the HDR-informed loss function for the captured Chef and Food scenes. 

<table><tr><td rowspan="2">Scene/Method</td><td colspan="3">PSNR (dB)↑</td><td colspan="3">LPIPS ↓</td><td colspan="3">SSIM ↑</td><td colspan="3">L1 (depth) ↓</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>Chef w/o HDR</td><td>18.19</td><td>18.09</td><td>13.54</td><td>0.360</td><td>0.324</td><td>0.345</td><td>0.767</td><td>0.773</td><td>0.643</td><td>0.004</td><td>0.005</td><td>0.025</td></tr><tr><td>Chef w/ HDR</td><td>19.27</td><td>18.14</td><td>19.55</td><td>0.334</td><td>0.338</td><td>0.276</td><td>0.775</td><td>0.747</td><td>0.811</td><td>0.006</td><td>0.007</td><td>0.013</td></tr><tr><td>Food w/o HDR</td><td>23.08</td><td>23.98</td><td>16.93</td><td>0.305</td><td>0.184</td><td>0.263</td><td>0.827</td><td>0.881</td><td>0.722</td><td>0.007</td><td>0.006</td><td>0.032</td></tr><tr><td>Food w/ HDR</td><td>23.40</td><td>23.78</td><td>22.09</td><td>0.286</td><td>0.205</td><td>0.154</td><td>0.826</td><td>0.879</td><td>0.873</td><td>0.006</td><td>0.007</td><td>0.013</td></tr></table>

# 4.5 Simulated Results

The paper focuses on results for sparse views (i.e., 2, 3, and 5 views) since this is the regime where depth supervision improves the most over only using 2D supervision. However in Table S6 we show results of the proposed method and the baselines trained on 10 simulated training views sampled at equal angles around the Lego scene, and our approach still outperforms the baselines when evaluated on the same test views.

Table S6: Simulated results for 2, 3, 5, and 10 training views on the Lego scene. 

<table><tr><td rowspan="2">Method</td><td colspan="4">PSNR (dB)↑</td><td colspan="4">LPIPS ↓</td><td colspan="4">L1 (depth) ↓</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>10 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>10 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>10 views</td></tr><tr><td>Instant NGP [30]</td><td>16.04</td><td>17.46</td><td>16.46</td><td>21.73</td><td>0.591</td><td>0.544</td><td>0.479</td><td>0.216</td><td>0.224</td><td>0.236</td><td>0.195</td><td>0.101</td></tr><tr><td>DS-NeRF [26]</td><td>17.66</td><td>19.45</td><td>17.21</td><td>22.83</td><td>0.500</td><td>0.476</td><td>0.482</td><td>0.302</td><td>0.136</td><td>0.127</td><td>0.193</td><td>0.068</td></tr><tr><td>Urban NeRF [4]</td><td>19.59</td><td>19.75</td><td>16.19</td><td>21.36</td><td>0.519</td><td>0.522</td><td>0.503</td><td>0.368</td><td>0.105</td><td>0.100</td><td>0.151</td><td>0.074</td></tr><tr><td>Urban NeRF w/mask</td><td>20.48</td><td>21.50</td><td>19.48</td><td>25.02</td><td>0.471</td><td>0.442</td><td>0.421</td><td>0.317</td><td>0.040</td><td>0.049</td><td>0.057</td><td>0.019</td></tr><tr><td>Proposed</td><td>20.64</td><td>23.63</td><td>25.18</td><td>31.74</td><td>0.190</td><td>0.161</td><td>0.192</td><td>0.084</td><td>0.023</td><td>0.011</td><td>0.008</td><td>0.005</td></tr></table>

In Figs. S3, S4, and S5 we show further renders from our method and baseline methods trained on the simulated dataset. The same trends as in the main text persist. Our method qualitatively produces images more faithful to the ground-truth. Our rendered images suffer less from floating artifacts and display finer details, for example in the lego scene. Some artifacts in the depth maps (i.e., the "holes" that appear) result from how we visualize depth in low occupancy areas. Depths in these regions can be become biased as described by Equation S6.

We show a breakdown across all simulated scenes of the evaluation metrics (see Table S7, S8, S9, S10). In addition to the metrics reported in the main text (PSNR, LPIPS, L1 depth) we add the structural similarity (SSIM) metric. Again, our method outperforms the baselines in the quantitative metrics. Since the 2, 3, and 5 view results do not include viewpoints that are strict supersets of each other, the performance of some metrics does not always increase for every scene in every case with increasing views (though this is a trend we observe on average).

Table S7: Breakdown of PSNR (dB) across all 5 simulated scenes. 

<table><tr><td rowspan="2">Scene</td><td colspan="3">Instant NGP [7]↑</td><td colspan="3">DS-NeRF [9]</td><td colspan="3">Urban NeRF [12]</td><td colspan="3">Urban NeRF w/Mask [12]</td><td colspan="3">Proposed</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>lego</td><td>16.04</td><td>17.46</td><td>16.46</td><td>17.66</td><td>19.45</td><td>17.21</td><td>19.59</td><td>19.75</td><td>16.19</td><td>20.48</td><td>21.50</td><td>19.48</td><td>20.64</td><td>23.63</td><td>25.81</td></tr><tr><td>chair</td><td>13.40</td><td>15.42</td><td>24.95</td><td>16.51</td><td>16.72</td><td>25.49</td><td>17.53</td><td>16.14</td><td>24.28</td><td>20.39</td><td>18.69</td><td>28.23</td><td>20.75</td><td>21.99</td><td>34.48</td></tr><tr><td>hotdog</td><td>14.95</td><td>14.72</td><td>13.80</td><td>18.93</td><td>17.47</td><td>19.12</td><td>15.75</td><td>16.37</td><td>18.04</td><td>18.90</td><td>18.71</td><td>19.53</td><td>20.74</td><td>22.64</td><td>32.36</td></tr><tr><td>bench</td><td>16.85</td><td>19.33</td><td>15.58</td><td>19.87</td><td>20.34</td><td>15.69</td><td>19.12</td><td>19.52</td><td>14.56</td><td>20.64</td><td>21.55</td><td>17.57</td><td>20.20</td><td>23.06</td><td>21.57</td></tr><tr><td>ficus</td><td>21.87</td><td>21.40</td><td>27.50</td><td>23.44</td><td>22.75</td><td>27.85</td><td>22.32</td><td>21.88</td><td>25.91</td><td>24.12</td><td>23.58</td><td>26.87</td><td>24.57</td><td>26.10</td><td>27.70</td></tr><tr><td>average</td><td>16.62</td><td>17.67</td><td>19.66</td><td>19.28</td><td>19.35</td><td>21.07</td><td>18.86</td><td>18.73</td><td>19.80</td><td>20.91</td><td>20.81</td><td>22.34</td><td>21.38</td><td>23.48</td><td>28.39</td></tr></table>

Table S8: Breakdown of LPIPS metric across all 5 simulated scenes. 

<table><tr><td rowspan="2">Scene</td><td colspan="3">Instant NGP [7]↑</td><td colspan="3">DS-NeRF [9]</td><td colspan="3">Urban NeRF [12]</td><td colspan="3">Urban NeRF w/Mask [12]</td><td colspan="3">Proposed</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>lego</td><td>0.591</td><td>0.544</td><td>0.479</td><td>0.500</td><td>0.476</td><td>0.482</td><td>0.519</td><td>0.522</td><td>0.503</td><td>0.471</td><td>0.442</td><td>0.421</td><td>0.190</td><td>0.161</td><td>0.192</td></tr><tr><td>chair</td><td>0.501</td><td>0.466</td><td>0.296</td><td>0.482</td><td>0.461</td><td>0.306</td><td>0.494</td><td>0.474</td><td>0.326</td><td>0.359</td><td>0.357</td><td>0.275</td><td>0.138</td><td>0.138</td><td>0.037</td></tr><tr><td>hotdog</td><td>0.583</td><td>0.615</td><td>0.523</td><td>0.456</td><td>0.470</td><td>0.390</td><td>0.580</td><td>0.553</td><td>0.433</td><td>0.533</td><td>0.466</td><td>0.395</td><td>0.242</td><td>0.241</td><td>0.118</td></tr><tr><td>bench</td><td>0.538</td><td>0.442</td><td>0.398</td><td>0.444</td><td>0.422</td><td>0.459</td><td>0.502</td><td>0.473</td><td>0.483</td><td>0.400</td><td>0.368</td><td>0.375</td><td>0.194</td><td>0.139</td><td>0.159</td></tr><tr><td>ficus</td><td>0.387</td><td>0.311</td><td>0.238</td><td>0.272</td><td>0.352</td><td>0.244</td><td>0.403</td><td>0.400</td><td>0.286</td><td>0.287</td><td>0.278</td><td>0.229</td><td>0.094</td><td>0.079</td><td>0.069</td></tr><tr><td>average</td><td>0.520</td><td>0.476</td><td>0.387</td><td>0.431</td><td>0.436</td><td>0.376</td><td>0.500</td><td>0.484</td><td>0.406</td><td>0.410</td><td>0.382</td><td>0.339</td><td>0.172</td><td>0.151</td><td>0.115</td></tr></table>

Table S9: Breakdown of SSIM metric across all 5 simulated scenes. 

<table><tr><td rowspan="2">Scene</td><td colspan="3">Instant NGP [7]↑</td><td colspan="3">DS-NeRF [9]</td><td colspan="3">Urban NeRF [12]</td><td colspan="3">Urban NeRF w/Mask [12]</td><td colspan="3">Proposed</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>lego</td><td>0.456</td><td>0.519</td><td>0.576</td><td>0.600</td><td>0.624</td><td>0.631</td><td>0.592</td><td>0.580</td><td>0.590</td><td>0.646</td><td>0.710</td><td>0.722</td><td>0.860</td><td>0.897</td><td>0.899</td></tr><tr><td>chair</td><td>0.545</td><td>0.632</td><td>0.767</td><td>0.653</td><td>0.659</td><td>0.760</td><td>0.582</td><td>0.624</td><td>0.766</td><td>0.774</td><td>0.756</td><td>0.897</td><td>0.901</td><td>0.899</td><td>0.977</td></tr><tr><td>hotdog</td><td>0.508</td><td>0.448</td><td>0.534</td><td>0.736</td><td>0.636</td><td>0.687</td><td>0.503</td><td>0.534</td><td>0.664</td><td>0.567</td><td>0.686</td><td>0.761</td><td>0.882</td><td>0.875</td><td>0.963</td></tr><tr><td>bench</td><td>0.537</td><td>0.591</td><td>0.588</td><td>0.703</td><td>0.664</td><td>0.618</td><td>0.554</td><td>0.615</td><td>0.580</td><td>0.717</td><td>0.769</td><td>0.747</td><td>0.856</td><td>0.884</td><td>0.870</td></tr><tr><td>ficus</td><td>0.682</td><td>0.819</td><td>0.851</td><td>0.831</td><td>0.789</td><td>0.873</td><td>0.734</td><td>0.711</td><td>0.844</td><td>0.819</td><td>0.840</td><td>0.913</td><td>0.925</td><td>0.934</td><td>0.942</td></tr><tr><td>average</td><td>0.546</td><td>0.602</td><td>0.663</td><td>0.705</td><td>0.675</td><td>0.714</td><td>0.593</td><td>0.613</td><td>0.689</td><td>0.705</td><td>0.752</td><td>0.808</td><td>0.885</td><td>0.898</td><td>0.930</td></tr></table>

Table S10: Breakdown of L1 (depth) metric across all 5 simulated scenes. 

<table><tr><td rowspan="2">Scene</td><td colspan="3">Instant NGP [7]↑</td><td colspan="3">DS-NeRF [9]</td><td colspan="3">Urban NeRF [12]</td><td colspan="3">Urban NeRF w/Mask [12]</td><td colspan="3">Proposed</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>lego</td><td>0.224</td><td>0.236</td><td>0.195</td><td>0.136</td><td>0.127</td><td>0.193</td><td>0.105</td><td>0.100</td><td>0.151</td><td>0.040</td><td>0.049</td><td>0.057</td><td>0.023</td><td>0.011</td><td>0.008</td></tr><tr><td>chair</td><td>0.166</td><td>0.198</td><td>0.117</td><td>0.123</td><td>0.146</td><td>0.097</td><td>0.130</td><td>0.132</td><td>0.066</td><td>0.034</td><td>0.057</td><td>0.009</td><td>0.006</td><td>0.005</td><td>0.003</td></tr><tr><td>hotdog</td><td>0.344</td><td>0.254</td><td>0.271</td><td>0.131</td><td>0.151</td><td>0.115</td><td>0.301</td><td>0.270</td><td>0.089</td><td>0.126</td><td>0.040</td><td>0.008</td><td>0.007</td><td>0.006</td><td>0.003</td></tr><tr><td>bench</td><td>0.166</td><td>0.137</td><td>0.198</td><td>0.082</td><td>0.076</td><td>0.142</td><td>0.060</td><td>0.047</td><td>0.168</td><td>0.011</td><td>0.007</td><td>0.038</td><td>0.006</td><td>0.005</td><td>0.003</td></tr><tr><td>ficus</td><td>0.291</td><td>0.147</td><td>0.107</td><td>0.072</td><td>0.074</td><td>0.048</td><td>0.059</td><td>0.073</td><td>0.034</td><td>0.043</td><td>0.034</td><td>0.032</td><td>0.034</td><td>0.033</td><td>0.048</td></tr><tr><td>average</td><td>0.238</td><td>0.195</td><td>0.178</td><td>0.109</td><td>0.115</td><td>0.119</td><td>0.131</td><td>0.124</td><td>0.101</td><td>0.051</td><td>0.038</td><td>0.029</td><td>0.015</td><td>0.011</td><td>0.013</td></tr></table>

![](images/0df821a911d46f2ddd314c9ec464776bc40e827e72f8451f913b95070c25b03b.jpg)

<details>
<summary>text_image</summary>

Ground Truth
Instant-NGP
DS-NeRF
Urban NeRF
Urban NeRF-M
Proposed
statue
chair
ficus
hotdog
lego
</details>

Figure S3: Rendered images and depths on the simulated dataset for 2 views.

![](images/1969d36879e7635b0dddb0af6e5131e02ab013900147e0cf45a2c85ee55a5925.jpg)

<details>
<summary>text_image</summary>

Ground Truth
Instant-NGP
DS-NeRF
Urban NeRF
Urban NeRF-M
Proposed
statue
chair
ficus
hotdog
lego
</details>

Figure S4: Rendered images and depths on the simulated dataset for 3 views.

![](images/3b6d5ec316baac9975edde252671a79a60893759b0f1528d0a2cee4ca5014b87.jpg)

<details>
<summary>text_image</summary>

Ground Truth
Instant-NGP
DS-NeRF
Urban NeRF
Urban NeRF-M
Proposed
statue
chair
ficus
hotdog
lego
</details>

Figure S5: Rendered images and depths on the simulated dataset for 5 views.

# 4.6 Captured Results

In Figs. S6, S7, and S8 we show further renders from our model and the baselines on our captured dataset. The proposed method produces images more faithful to the ground truth.

We note that the quality of the reconstructions is lower than the simulation results. We attribute this to small imperfections in the calibration of the camera extrinsics, which register the multiview lidar scans to an accuracy of approximately 1 mm. Achieving precise, sub-mm alignment of the multiview lidar scans is a highly non-trivial problem and is outside the scope of our current work. The captured results bear out the trends observed in simulation and demonstrate Transient NeRF and novel view synthesis of lidar measurements for the first time in practice.

We show a breakdown across all captured scenes of the evaluation metrics (see Tables S11, S12, S13, and S14). In addition to the metrics reported in the main text (PSNR, LPIPS, L1 depth) we add the SSIM metric. Again our method outperforms the baselines in the quantitative metrics.

Table S11: Breakdown of PSNR (dB) across all 6 captured scenes. 

<table><tr><td rowspan="2">Scene</td><td colspan="3">Instant NGP [7]↑</td><td colspan="3">DS-NeRF [9]</td><td colspan="3">Urban NeRF [12]</td><td colspan="3">Urban NeRF w/Mask [12]</td><td colspan="3">Proposed</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>cinema</td><td>14.73</td><td>15.18</td><td>15.21</td><td>13.18</td><td>13.87</td><td>12.58</td><td>17.27</td><td>15.11</td><td>17.30</td><td>14.13</td><td>17.22</td><td>17.80</td><td>21.61</td><td>21.66</td><td>25.12</td></tr><tr><td>boots</td><td>16.54</td><td>18.32</td><td>18.09</td><td>16.97</td><td>15.65</td><td>16.59</td><td>16.54</td><td>14.54</td><td>16.59</td><td>13.93</td><td>18.70</td><td>21.87</td><td>22.38</td><td>22.29</td><td>24.94</td></tr><tr><td>baskets</td><td>18.73</td><td>19.78</td><td>18.17</td><td>16.47</td><td>17.43</td><td>17.54</td><td>16.68</td><td>15.88</td><td>13.42</td><td>17.17</td><td>16.51</td><td>14.26</td><td>22.48</td><td>20.93</td><td>19.90</td></tr><tr><td>carving</td><td>17.96</td><td>17.30</td><td>16.70</td><td>17.41</td><td>16.36</td><td>15.44</td><td>18.71</td><td>17.97</td><td>17.36</td><td>16.07</td><td>21.41</td><td>22.28</td><td>23.52</td><td>24.20</td><td>24.70</td></tr><tr><td>chef</td><td>13.04</td><td>11.72</td><td>13.76</td><td>12.32</td><td>12.27</td><td>11.51</td><td>13.34</td><td>13.71</td><td>13.15</td><td>14.06</td><td>15.25</td><td>16.72</td><td>19.27</td><td>18.14</td><td>19.55</td></tr><tr><td>food</td><td>17.64</td><td>16.83</td><td>16.45</td><td>15.68</td><td>14.70</td><td>15.48</td><td>18.86</td><td>18.27</td><td>17.77</td><td>17.37</td><td>20.45</td><td>21.72</td><td>23.40</td><td>23.78</td><td>22.09</td></tr><tr><td>average</td><td>16.44</td><td>16.52</td><td>16.39</td><td>15.34</td><td>15.05</td><td>14.86</td><td>16.90</td><td>15.91</td><td>15.93</td><td>15.45</td><td>18.26</td><td>19.11</td><td>22.11</td><td>21.83</td><td>22.72</td></tr></table>

Table S12: Breakdown of LPIPS across all 6 captured scenes. 

<table><tr><td rowspan="2">Scene</td><td colspan="3">Instant NGP [7]↑</td><td colspan="3">DS-NeRF [9]</td><td colspan="3">Urban NeRF [12]</td><td colspan="3">Urban NeRF w/Mask [12]</td><td colspan="3">Proposed</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>cinema</td><td>0.445</td><td>0.374</td><td>0.314</td><td>0.396</td><td>0.364</td><td>0.429</td><td>0.369</td><td>0.350</td><td>0.273</td><td>0.457</td><td>0.295</td><td>0.244</td><td>0.281</td><td>0.245</td><td>0.178</td></tr><tr><td>boots</td><td>0.251</td><td>0.253</td><td>0.181</td><td>0.193</td><td>0.225</td><td>0.192</td><td>0.386</td><td>0.332</td><td>0.163</td><td>0.525</td><td>0.245</td><td>0.135</td><td>0.221</td><td>0.182</td><td>0.155</td></tr><tr><td>baskets</td><td>0.399</td><td>0.298</td><td>0.252</td><td>0.256</td><td>0.244</td><td>0.268</td><td>0.444</td><td>0.311</td><td>0.220</td><td>0.433</td><td>0.300</td><td>0.195</td><td>0.269</td><td>0.165</td><td>0.164</td></tr><tr><td>carving</td><td>0.174</td><td>0.167</td><td>0.183</td><td>0.168</td><td>0.186</td><td>0.225</td><td>0.357</td><td>0.217</td><td>0.146</td><td>0.436</td><td>0.170</td><td>0.125</td><td>0.232</td><td>0.138</td><td>0.103</td></tr><tr><td>chef</td><td>0.580</td><td>0.443</td><td>0.401</td><td>0.498</td><td>0.434</td><td>0.467</td><td>0.513</td><td>0.468</td><td>0.362</td><td>0.473</td><td>0.373</td><td>0.270</td><td>0.334</td><td>0.338</td><td>0.276</td></tr><tr><td>food</td><td>0.299</td><td>0.310</td><td>0.313</td><td>0.354</td><td>0.417</td><td>0.369</td><td>0.351</td><td>0.290</td><td>0.221</td><td>0.425</td><td>0.232</td><td>0.174</td><td>0.286</td><td>0.205</td><td>0.154</td></tr><tr><td>average</td><td>0.358</td><td>0.307</td><td>0.274</td><td>0.311</td><td>0.312</td><td>0.325</td><td>0.403</td><td>0.328</td><td>0.231</td><td>0.458</td><td>0.269</td><td>0.191</td><td>0.271</td><td>0.212</td><td>0.172</td></tr></table>

Table S13: Breakdown of SSIM across all 6 captured scenes. 

<table><tr><td rowspan="2">Scene</td><td colspan="3">Instant NGP [7]↑</td><td colspan="3">DS-NeRF [9]</td><td colspan="3">Urban NeRF [12]</td><td colspan="3">Urban NeRF w/Mask [12]</td><td colspan="3">Proposed</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>cinema</td><td>0.545</td><td>0.628</td><td>0.717</td><td>0.620</td><td>0.666</td><td>0.591</td><td>0.602</td><td>0.673</td><td>0.768</td><td>0.509</td><td>0.728</td><td>0.807</td><td>0.850</td><td>0.812</td><td>0.879</td></tr><tr><td>boots</td><td>0.804</td><td>0.810</td><td>0.866</td><td>0.853</td><td>0.822</td><td>0.851</td><td>0.587</td><td>0.687</td><td>0.871</td><td>0.414</td><td>0.777</td><td>0.894</td><td>0.912</td><td>0.909</td><td>0.914</td></tr><tr><td>baskets</td><td>0.555</td><td>0.715</td><td>0.749</td><td>0.737</td><td>0.750</td><td>0.738</td><td>0.499</td><td>0.689</td><td>0.751</td><td>0.517</td><td>0.691</td><td>0.771</td><td>0.846</td><td>0.850</td><td>0.826</td></tr><tr><td>carving</td><td>0.852</td><td>0.853</td><td>0.843</td><td>0.858</td><td>0.841</td><td>0.804</td><td>0.620</td><td>0.827</td><td>0.877</td><td>0.526</td><td>0.854</td><td>0.906</td><td>0.913</td><td>0.929</td><td>0.929</td></tr><tr><td>chef</td><td>0.414</td><td>0.586</td><td>0.647</td><td>0.544</td><td>0.608</td><td>0.574</td><td>0.450</td><td>0.559</td><td>0.686</td><td>0.488</td><td>0.651</td><td>0.780</td><td>0.775</td><td>0.747</td><td>0.811</td></tr><tr><td>food</td><td>0.731</td><td>0.720</td><td>0.733</td><td>0.677</td><td>0.586</td><td>0.671</td><td>0.654</td><td>0.737</td><td>0.808</td><td>0.528</td><td>0.775</td><td>0.855</td><td>0.826</td><td>0.879</td><td>0.873</td></tr><tr><td>average</td><td>0.650</td><td>0.719</td><td>0.759</td><td>0.715</td><td>0.712</td><td>0.705</td><td>0.569</td><td>0.695</td><td>0.793</td><td>0.497</td><td>0.746</td><td>0.836</td><td>0.854</td><td>0.854</td><td>0.872</td></tr></table>

Table S14: Breakdown of L1 (depth) metric across all 6 captured scenes. 

<table><tr><td rowspan="2">Scene</td><td colspan="3">Instant NGP [7]↑</td><td colspan="3">DS-NeRF [9]</td><td colspan="3">Urban NeRF [12]</td><td colspan="3">Urban NeRF w/Mask [12]</td><td colspan="3">Proposed</td></tr><tr><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td><td>2 views</td><td>3 views</td><td>5 views</td></tr><tr><td>cinema</td><td>0.141</td><td>0.052</td><td>0.041</td><td>0.067</td><td>0.040</td><td>0.050</td><td>0.023</td><td>0.017</td><td>0.015</td><td>0.018</td><td>0.005</td><td>0.006</td><td>0.006</td><td>0.006</td><td>0.007</td></tr><tr><td>boots</td><td>0.034</td><td>0.076</td><td>0.048</td><td>0.029</td><td>0.023</td><td>0.022</td><td>0.017</td><td>0.014</td><td>0.011</td><td>0.013</td><td>0.003</td><td>0.003</td><td>0.001</td><td>0.002</td><td>0.002</td></tr><tr><td>baskets</td><td>0.138</td><td>0.141</td><td>0.040</td><td>0.038</td><td>0.033</td><td>0.029</td><td>0.015</td><td>0.017</td><td>0.004</td><td>0.016</td><td>0.008</td><td>0.003</td><td>0.008</td><td>0.008</td><td>0.026</td></tr><tr><td>carving</td><td>0.029</td><td>0.032</td><td>0.031</td><td>0.031</td><td>0.024</td><td>0.026</td><td>0.008</td><td>0.011</td><td>0.014</td><td>0.009</td><td>0.004</td><td>0.003</td><td>0.006</td><td>0.005</td><td>0.006</td></tr><tr><td>chef</td><td>0.199</td><td>0.100</td><td>0.124</td><td>0.062</td><td>0.056</td><td>0.055</td><td>0.023</td><td>0.020</td><td>0.025</td><td>0.017</td><td>0.011</td><td>0.009</td><td>0.003</td><td>0.006</td><td>0.006</td></tr><tr><td>food</td><td>0.146</td><td>0.057</td><td>0.035</td><td>0.058</td><td>0.041</td><td>0.035</td><td>0.017</td><td>0.013</td><td>0.014</td><td>0.013</td><td>0.006</td><td>0.005</td><td>0.006</td><td>0.007</td><td>0.016</td></tr><tr><td>average</td><td>0.115</td><td>0.076</td><td>0.053</td><td>0.048</td><td>0.036</td><td>0.036</td><td>0.017</td><td>0.015</td><td>0.014</td><td>0.014</td><td>0.006</td><td>0.005</td><td>0.005</td><td>0.006</td><td>0.010</td></tr></table>

![](images/e5fdf94c61be7e54b8bc8d5c988656fa300d0280842327e2311decbbe183b472.jpg)

<details>
<summary>heatmap</summary>

| Category   | Ground Truth | Instant-NGP | DS-NeRF | Urban NeRF | Urban NeRF-M | Proposed |
| ---------- | ------------ | ----------- | ------- | ---------- | ------------ | -------- |
| baskets   | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| boots      | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| carving   | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| chef       | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| cinema    | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| food       | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
</details>

Figure S6: Rendered images and depths on the captured dataset for 2 views.

![](images/1bbde66e3cbe894c5494388ef6d619382ba128e22c208ce4e90b6fc26dc55f38.jpg)

<details>
<summary>heatmap</summary>

| Category | Ground Truth | Instant-NGP | DS-NeRF | Urban NeRF | Urban NeRF-M | Proposed |
| -------- | ------------ | ----------- | ------- | ---------- | ------------ | -------- |
| baskets | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| boots    | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| carving | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| chef     | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| cinema  | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| food     | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
</details>

Figure S7: Rendered images and depths on the captured dataset for 3 views.

![](images/6be4f40e60999631a6533023d4659b3a0f8f16ca5003794e5d012b14b2799bd3.jpg)

<details>
<summary>heatmap</summary>

| Category   | Ground Truth | Instant-NGP | DS-NeRF | Urban NeRF | Urban NeRF-M | Proposed |
| ---------- | ------------ | ----------- | ------- | ---------- | ------------ | -------- |
| Basketls   | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| Boots      | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| Carving    | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| Chef       | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| Cinema     | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
| Food       | 0.0          | 0.0         | 0.0     | 0.0        | 0.0          | 0.0      |
</details>

Figure S8: Rendered images and depths on the captured dataset for 5 views.

# References

[1] Joshua Rapp, Yanting Ma, Robin MA Dawson, and Vivek K Goyal. Dead time compensation for high-flux ranging. IEEE Trans. Signal Process., 67(13):3471–3486, 2019.   
[2] Michael D Grossberg and Shree K Nayar. The raxel imaging model and ray-based calibration. Int. J. Comput. Vis., 61(2):119, 2005.   
[3] G. Bradski. The OpenCV Library. Dr. Dobb's Journal of Software Tools, 2000.   
[4] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Associates, Inc., 2019. URL http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.   
[5] Dong C. Liu and Jorge Nocedal. On the limited memory bfgs method for large scale optimization. Math. Program., 45(1-3):503–528, 1989. URL http://dblp.uni-trier.de/db/journals/mp/mp45.html#LiuN89.   
[6] Ruilong Li, Matthew Tancik, and Angjoo Kanazawa. Nerfacc: A general nerf acceleration toolbox. arXiv preprint arXiv:2210.04847, 2022.   
[7] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. (SIGGRAPH), 41(4):1-15, 2022.   
[8] Merlin Nimier-David, Delio Vicini, Tizian Zeltner, and Wenzel Jakob. Mitsuba 2: A retargetable forward and inverse renderer. ACM Trans. Graph., 38(6):1–17, 2019.   
[9] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised NeRF: Fewer views and faster training for free. In Proc. CVPR, 2022.   
[10] Joshua Rapp and Vivek K Goyal. A few photons among many: Unmixing signal and noise for photon-efficient active imaging. IEEE Trans. Comput. Imaging, 3(3):445–459, 2017.   
[11] Matthew J Leotta, Chengjiang Long, Bastien Jacquet, Matthieu Zins, Dan Lipsa, Jie Shan, Bo Xu, Zhixin Li, Xu Zhang, Shih-Fu Chang, et al. Urban semantic 3d reconstruction from multiview satellite imagery. In Proc. CVPR Workshops, 2019.   
[12] Konstantinos Rematas, Andrew Liu, Pratul P Srinivasan, Jonathan T Barron, Andrea Tagliasacchi, Thomas Funkhouser, and Vittorio Ferrari. Urban radiance fields. In Proc. CVPR, 2022.   
[13] Diego Royo, Jorge García, Adolfo Muñoz, and Adrian Jarabo. Non-line-of-sight transient rendering. Computers & Graphics, 107:84–92, 2022.