# Diverse Rare Sample Generation with Pretrained GANs

Subeen Lee $^{1}$ , Jiyeon Han $^{1}$ , Soyeon Kim $^{1}$ and Jaesik Choi $^{1,2}$

$^{1}$ Korea Advanced Institute of Science and Technology (KAIST), South Korea $^{2}$ INEEJI, South Korea  
{sbrblee7, j.han, soyeon.k, jaesik.choi}@kaist.ac.kr

# Abstract

Deep generative models are proficient in generating realistic data but struggle with producing rare samples in low density regions due to their scarcity of training datasets and the mode collapse problem. While recent methods aim to improve the fidelity of generated samples, they often reduce diversity and coverage by ignoring rare and novel samples. This study proposes a novel approach for generating diverse rare samples from high-resolution image datasets with pretrained GANs. Our method employs gradient-based optimization of latent vectors within a multi-objective framework and utilizes normalizing flows for density estimation on the feature space. This enables the generation of diverse rare images, with controllable parameters for rarity, diversity, and similarity to a reference image. We demonstrate the effectiveness of our approach both qualitatively and quantitatively across various datasets and GANs without retraining or fine-tuning the pretrained GANs.

Code — https://github.com/sbrblee/DivRareGen

# 1 Introduction

Deep generative models have shown impressive generative capabilities across various domains. The primary focus of current generative model research is on enhancing the fidelity of generated images (DeVries, Drozdal, and Taylor 2020; Karras, Laine, and Aila 2019; Azadi et al. 2018; Turner et al. 2019). However, these approaches often compromise sample diversity and encounter difficulties generating rare samples, primarily due to their limited representation in the training dataset (Sehwag et al. 2022; Lee et al. 2021). In GANs, this issue is worsened by the mode collapse problem (Thanh-Tung and Tran 2020). Investigating rare samples is crucial for several reasons: it enhances the creation of synthetic datasets that embody diversity and creativity (Sehwag et al. 2022; Agarwal, D'souza, and Hooker 2022), ensures fairness in generative processes (Teo, Abdollahzadeh, and Cheung 2023; Hwang et al. 2020), and aligns with the human tendency to favor unique features (Snyder and Lopez 2001; Lynn and Harris 1997). Additionally, exploring edge cases and unusual scenarios is essential in various domains, such as drug discovery or molecular design Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

(Sagar et al. 2023; Zeng et al. 2022), and natural hazard analysis (Ma, Mei, and Xu 2024).

Several studies have been conducted to enhance the overall diversity of GAN-generated outputs and to promote the generation of rare samples (Chang et al. 2024; Allahyani et al. 2023; Humayun, Balestriero, and Baraniuk 2022, 2021; Heyrani Nobari, Rashad, and Ahmed 2021; Ghosh et al. 2018; Tolstikhin et al. 2017; Srivastava et al. 2017; Chen et al. 2016). Due to the high computational cost of training GANs (Karras, Laine, and Aila 2019; Brock, Donahue, and Simonyan 2018), techniques without model retraining are appealing. For example, Humayun, Balestriero, and Baraniuk (2022) proposed a resampling technique for pretrained GANs with a controllable fidelity-diversity trade-off parameter. However, the proposed method requires extensive sampling to cover the data manifold fully. On the other hand, Chang et al. (2024) proposed a method to obtain diverse samples that satisfy text conditions by optimizing latent vectors with a quality-diversity objective.

Han et al. (2023) proposes a rarity score for samples using relative density measures based on k-nearest neighbor (k-NN) manifolds in the feature space of pretrained classifiers. Although k-NN density estimation is straightforward and reliable (Naeem et al. 2020; Kynkäänniemi et al. 2019), its non-differentiable nature complicates gradient-based optimization. In contrast, normalizing flows (NFs) excel at high-dimensional density estimation in differentiable form (Papamakarios et al. 2021; Kingma and Dhariwal 2018; Dinh, Sohl-Dickstein, and Bengio 2016; Dinh, Krueger, and Bengio 2014). We employ NFs for density estimation in the feature space, which incurs a lower training cost compared to retraining or fine-tuning GANs, and analyze how NF-based density estimates relate to the rarity score.

This study aims to generate diverse rare samples for a given high-resolution image datasets and GANs. Our method does not require any fine-tuning or retraining of the GANs but instead explores the latent space of the given model through gradient-based optimization. Our contributions are as follows:

- Our method can generate diverse versions of rare images utilizing the multi-start method in optimization, without being trapped at the same local optima.   
- Rarity and diversity of generated images and similarity to the initial image can be controlled via a multi-objective

![](images/fc0e44f798183f0d08babf745087892906195252be952beaa6e8886fd4a8ea56.jpg)

<details>
<summary>text_image</summary>

Race
white
non-white
non-white
white
white
Pose
front
front
left
front
front
Acc.
-
-
-
-
glasses
hat
Initial
↓
Rare
Wearing a Hat
Brown Hair → Colorful Hair
Initial
↓
Rare
Race
white
white
non-white
white
white
Pose
front
low
right
front
front
Acc.
hat
hat
hat, glasses
Getting Very Young or Old
White → Non-White
</details>

Figure 1: Examples of rare samples generated by our method. Left: Our method produces diverse rare images for a single reference, with variations even within the same rare attribute (e.g., hats of different shapes and colors). Right: Generated rare attributes include accessories like hats, non-brown hair colors, extreme ages, and non-white races. “Pose” refers to head orientation, and “Acc.” denotes accessories. Rare attributes are highlighted in bold.

optimization framework.

\- We demonstrate the effectiveness of our method with various high-resolution image datasets and GANs, both qualitatively and quantitatively.

# 2 Related Work

Rare Generation Han et al. (2023) introduced the rarity score to quantify the uniqueness of individual samples, distinguishing it from conventional metrics that primarily evaluate fidelity or diversity in generated samples (Kynkäänniemi et al. 2019; Zhang et al. 2018; Heusel et al. 2017). The rarity score is defined as the minimum k-nearest neighbor distance (k-NND) among real samples that are closer to the target sample, with higher scores indicating lower density within the real data manifold. However, obtaining rare samples has received limited attention. Sehwag et al. (2022) addressed this by leveraging a pretrained classifier to estimate likelihoods and adapting the sampling process of diffusion probabilistic models to target low-density regions while maintaining fidelity. Their method focuses on sampling from regions far from class mean vectors in the feature space and penalizes deviations from the overall mean vector of real data. However, this approach is class-conditional and depends on a Gaussian likelihood function.

On the other hand, Humayun, Balestriero, and Baraniuk (2022) tackled the mode collapse issue in GANs with Polarity sampling, a fidelity-diversity controllable resampling strategy for pretrained GANs. It approximates the GAN's output space using continuous piecewise affine splines. By tuning $\rho$ , sampling can focus on modes ( $\rho < 0$ ) or antimodes ( $\rho > 0$ ), with higher $\rho$ increasing diversity by targeting low-density regions. However, it does not guarantee the fidelity of the selected samples and requires extensive sampling and Jacobian matrix computations, leading to high computational costs.

Quality-Preserved Diverse Generation Using Pretrained GANs Generating rare samples is important, but maintaining quality is also crucial for their usefulness (Amabile 2018). Achieving diverse, high-fidelity samples is similar to finding multiple solutions in combinatorial optimization. Chang et al. (2024) addressed this by proposing a quality-diversity algorithm that updates the latent vector to balance quality and diversity, using the CLIP score (Radford et al. 2021) to measure similarity and diversity. In our work, we also optimize the latent vector to generate diverse samples; however, we prioritize rarity as the main objective rather than quality and use Euclidean distance in arbitrary feature spaces, avoiding the additional text constraints required by the CLIP score. To prevent low-fidelity samples, we apply a constraint to keep the sampled data within the real data manifold.

Reference-based Generation Finding diverse rare variations of a given initial image relates to reference-based image generation, encompassing tasks like domain adaptation (Yang et al. 2023), editing (Xia et al. 2023), and conditional generation (Casanova et al. 2021). While these approaches involve additional training costs for each attribute (Yang et al. 2023) or reference (Xia et al. 2023), or require a different GAN training scheme (Casanova et al. 2021), our method only requires a single training phase for the density estimator across multiple references with pretrained GANs.

Density Estimation for Images The rarity score identifies samples in low-density regions of the real data manifold, making it valuable for detecting rare generations. However, the non-differentiable nature of k-NN-based manifold estimation complicates gradient-based optimization for directly obtaining rare samples. In contrast, extensive research has been conducted to estimate density in high-dimensional spaces. Normalizing flows (NFs), as likelihood-based probabilistic models, use a sequence of invertible functions to transform a simple density into a complex one, potentially representing multi-modal distributions while preserving data relationships (Papamakarios et al. 2021; Kingma and Dhariwal 2018; Dinh, Sohl-Dickstein, and Bengio 2016; Dinh, Krueger, and Bengio 2014). While NFs may struggle with out-of-distribution data due to their focus on low-level features (Kirichenko, Izmailov, and Wilson 2020), training them on the feature space of a pretrained network—which

![](images/0f9bfdc39fc5f03dce21598f62143c010017ddcbb4cad22f3c3ac525c1861904.jpg)

<details>
<summary>natural_image</summary>

Abstract diagram with overlapping circles and a red line connecting nodes, no text or symbols present
</details>

(1) Rare optimization with multi-start $\log p(\mathbf{x}_i)$

![](images/f462514603e3b8c0792a7f54e75aed083a8e41a640e73774808fb24713eed5cd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Node"] --> B["Node"]
    B --> C["Node"]
    C --> D["Node"]
    D --> E["Node"]
    E --> F["Node"]
    F --> G["Node"]
    G --> H["Node"]
    H --> I["Node"]
    I --> J["Node"]
    J --> K["Node"]
    K --> L["Node"]
    L --> M["Node"]
    M --> N["Node"]
    N --> O["Node"]
    O --> P["Node"]
    P --> Q["Node"]
    Q --> R["Node"]
    R --> S["Node"]
    S --> T["Node"]
    T --> U["Node"]
    U --> V["Node"]
    V --> W["Node"]
    W --> X["Node"]
    X --> Y["Node"]
    Y --> Z["Node"]
```
</details>

(2) Diversity constraint $-\lambda_{2}\sum_{j\neq i}\left\| \mathbf{x}_{i} - \mathbf{x}_{j}\right\|^{2}$

![](images/4f4da5cc106716dc8a7be86916f2bd6593e8b015fc7c4b768a1a5906002ca13c.jpg)

<details>
<summary>text_image</summary>

x*
xi
d*
</details>

(3) Similarity constraint $+\lambda_{1}\max (d(\mathbf{x}_{i},\mathbf{x}^{*}) - d^{*})^{2}$

○ Initial point ( $x^{*}$ )

• Optimized point ( $x_{i}$ )

→ Optimization path

→ Random noise addition

Penalizing boundary

Normalizing flow manifold

k-NN manifold

Figure 2: Schematic diagram for the objective function of our method. $\mathbf{x}^{*} = f(G(\mathbf{z}^{*}))$ and $\mathbf{x}_i = f(G(\mathbf{z}_i))$ for brevity.

includes high-level semantic information—can help mitigate this issue (Esser, Rombach, and Ommer 2020).

Multi-Start Method for Diverse Solutions We frame our problem of obtaining diverse rare samples for each given reference as identifying multiple local minima around the reference in the data distribution. A straightforward approach to this problem is the multi-start method, an algorithm that iteratively searches for local optima starting from multiple initial points (Feo and Resende 1995; Rochat and Taillard 1995; Rinnooy Kan and Timmer 1987). This method selects several starting positions and applies a local search algorithm to each, aiming to locate distinct local optima. Although easy to implement, it does not always ensure that different starting points result in different local optima (Tarek and Huang 2022). To address this issue, we add the diversity and similarity constraints to the objective function, ensuring that each initial point converges to a different minimum.

# 3 Methods

# 3.1 Problem Statement

Given a GAN generator $G = G(\mathbf{z})$ , an arbitrary initial latent vector $\mathbf{z}^* \in \mathbb{R}^m$ , a feature extractor $f = f(\mathbf{I})$ , and a density estimator $g = g(\mathbf{x})$ , our objective is to find a set of latent vectors $\{\mathbf{z}_i\}_{i=1}^N$ that generates diverse rare samples which are similar to the image generated from $\mathbf{z}^*$ (referred to as the reference for the rest of the paper). Here, $\mathbf{I} \in \mathbb{R}^{w \times h \times 3}$ and $\mathbf{x} \in \mathbb{R}^n$ denote an image and a feature vector, respectively. For simplicity, we denote $\mathbf{x}^* = f(G(\mathbf{z}^*))$ and $\mathbf{x}_i = f(G(\mathbf{z}_i))$ . In this study, we define similarity as the Euclidean distance in the feature space, $d(\mathbf{x}_1, \mathbf{x}_2) = \| \mathbf{x}_1 - \mathbf{x}_2 \|$ , a metric shown to align well with human perception (Zhang et al. 2018).

We propose a multi-objective optimization framework that integrates rarity, diversity, and similarity regularization, as illustrated in Fig. 2, and provide a detailed explanation in the subsequent sections.

# 3.2 Rare Sample Generation

For rarity, the density estimated from g is used. NFs are employed due to their remarkable performance in high-dimensional density estimation, though any differentiable density estimator can be applied. NFs provide the exact log-likelihood of individual samples, allowing us to directly define the objective function as $\mathcal{L}_{\text{rare}}(\mathbf{x}) = g(\mathbf{x}) = \log p(\mathbf{x})$ to minimize. To control the similarity between the generated rare image and the reference image, we incorporate a regularization term inspired by Chang et al. (2024). Specifically, we define the similarity loss as $\mathcal{L}_{\text{sim}}(\mathbf{x}) = (\max(d(\mathbf{x}, \mathbf{x}^*), d^*) - d^*)^2$ , where this term penalizes samples that exceed a predefined boundary, referred to as the penalizing boundary throughout this paper, defined by $d^*$ . Specifically, we use the distance to $k'$ -nearest neighbor in the fake k-NN manifold for $d^*$ . We also strictly accept the sample inside this boundary as well as inside the real k-NN manifold $\Phi_{real}$ in optimization. The optimization goal is formulated by combining the two objectives—rarity and similarity regularization—as follows.

$$
\min _ {\mathbf {z}} \quad \mathcal {L} _ {\text { rare }} (\mathbf {x}) + \lambda_ {1} \mathcal {L} _ {\text { sim }} (\mathbf {x}) \tag {1}
$$

subject to $\mathbf{x} \in \Phi_{real}$ and $d(\mathbf{x}, \mathbf{x}^*) \leq d^*$

# 3.3 Diverse Rare Sample Generation

We utilize the multi-start method to obtain diverse rare images by adding small random noises to $z^{*}$ , generating multiple starting points for optimization. Specifically, $\{z_{i}\}_{i=1}^{N}$ are initialized as $z_{i}=z^{*}+\epsilon$ , where $\epsilon\sim\mathcal{N}(\mathbf{0},\sigma^{2}I)$ . However, as shown in Fig. 2 (1), this does not guarantee obtaining different rare images, but they might converge to the same local minima. To address this issue, we add a diversity constraint to the objective, $\mathcal{L}_{div}(\mathbf{x}_{i})=-\sum_{j\neq i}d(\mathbf{x}_{i},\mathbf{x}_{j})^{2}$ , ensuring that the samples are far from each other, as shown in Fig. 2 (2). This term is inspired by the expected distances in feature space, similar to the concept of Maximum Mean Discrepancy (MMD) (Gretton et al. 2006). Combining all objectives, the multi-objective optimization problem is formulated as follows.

$$
\min _ {\mathbf {z} _ {i}} \mathcal {L} _ {\text { rare }} (\mathbf {x} _ {i}) + \lambda_ {1} \mathcal {L} _ {\text { sim }} (\mathbf {x} _ {i}) + \lambda_ {2} \mathcal {L} _ {\text { div }} (\mathbf {x} _ {i}) \tag {2}
$$

subject to $\mathbf{x}_i\in \Phi_{real}$ and $d(\mathbf{x}_i,\mathbf{x}^*)\leq d^*$

# 4 Experimental Results

We validate our proposed method using high-resolution image datasets with a resolution of $1024 \times 1024$ , including Flickr Faces HQ (FFHQ) (Karras, Laine, and Aila 2019), Animal Faces HQ (AFHQ) (Choi et al. 2020), and Metfaces (Karras et al. 2020a). StyleGAN2 with config-f (Karras et al. 2020b) and StyleGAN2-ADA (Karras et al. 2020a) are utilized. For feature extraction, we employ the VGG16-fc2 architecture (Simonyan and Zisserman 2015). As the density estimator, the Glow architecture (Kingma and Dhariwal 2018) is adapted to accommodate the high dimensionality of the feature space. The optimization is performed using the Adam optimizer (Diederik 2014) with a learning rate of $2 \times 10^{-2}$ , combined with a StepLR scheduler. The best optimization results are recorded when the lowest loss is achieved according to Equation (2). Additional details including computational cost are provided in Appendix B.

# 4.1 Generation of Rare Facial Attributes with StyleGAN2 (FFHQ-StyleGAN2)

Quantitative Results As a baseline, 10,000 synthetic samples are generated using latent vectors from StyleGAN2 with a truncation parameter of $\psi = 1.0$ . With our method, we generate ten rare samples for each of 1,000 initial latent vectors from the baselines, using parameters $\lambda_{1} = 30.0$ , $\lambda_{2} = 0.002$ , $\sigma = 0.1$ , and $k' = 100^{1}$ . The choice of the parameters is explained in Appendix C. For Polarity sampling, 250,000 latent vectors and their corresponding Jacobian matrices are obtained from the authors (Humayun, Balestriero, and Baraniuk 2022), and 10,000 samples are resampled using $\rho = [1.0, 5.0]$ (anti-mode sampling).

Results are evaluated with metrics including the Rarity Score (RS) (Han et al. 2023), precision (Prec.) and recall (Rec.) for fidelity and diversity (Kynkäänniemi et al. 2019), LPIPS score for diversity (Zhang et al. 2018), and FID score (Heusel et al. 2017), as shown in Table 1. Each metric is computed using 10,000 generated samples, with the LPIPS score averaged over 10,000 random sample pairs. Significant differences in LPIPS scores between sampling methods are confirmed using an unpaired t-test. For the real k-NN manifold, k = 3 is used.

Our method improves both rarity and diversity compared to the baseline, even when the optimization uses only $10\%$ of the baseline samples as references. The FID score decreases because more samples are generated in low-density regions, reducing samples near the data distribution's modes. Polarity sampling also enhances rarity and diversity but sacrifices precision, as it primarily targets low-density regions in the GAN's output space rather than the real manifold, often generating out-of-distribution samples (structural zeros; (Kim and Bansal 2023)). Furthermore, in contrast to our objective of generating rare samples similar to a given reference, Polarity sampling is not designed for it. Finally, since Polarity sampling operates by resampling from an initial set, its diversity is heavily dependent on the size of that initial set. An additional comparison with Polarity sampling is in Appendix G.

<table><tr><td>Model</td><td>RS ↑</td><td>Prec. ↑</td><td>Rec. ↑</td><td>LPIPS ↑</td><td>FID ↓</td></tr><tr><td>Baseline</td><td>18.88</td><td>0.69</td><td>0.56</td><td>0.73</td><td>4.17</td></tr><tr><td>Polarity ( $\rho = 1.0$ )</td><td>24.71</td><td>0.39</td><td>0.70</td><td>0.75</td><td>33.28</td></tr><tr><td>Polarity ( $\rho = 5.0$ )</td><td>24.83</td><td>0.38</td><td>0.71</td><td>0.75</td><td>34.11</td></tr><tr><td>Ours</td><td>23.50</td><td>0.92</td><td>0.65</td><td>0.76</td><td>7.38</td></tr></table>

Table 1: Quantitative evaluation for Section 4.1.

<table><tr><td colspan="2">Model-Based Attribute</td><td>FFHQ</td><td>Reference</td><td>Ours(%)</td></tr><tr><td rowspan="3">Age</td><td>0-9</td><td>7.51</td><td>7.60</td><td>8.56</td></tr><tr><td>10-69</td><td>92.23</td><td>92.20</td><td>90.98</td></tr><tr><td>Over70</td><td>0.25</td><td>0.20</td><td>0.45</td></tr><tr><td rowspan="2">Gender</td><td>Male</td><td>45.46</td><td>47.44</td><td>48.50</td></tr><tr><td>Female</td><td>54.53</td><td>52.55</td><td>51.49</td></tr><tr><td rowspan="2">Race</td><td>White</td><td>62.97</td><td>64.66</td><td>61.52</td></tr><tr><td>Non White</td><td>37.03</td><td>35.33</td><td>38.47</td></tr><tr><td rowspan="2">HeadPose</td><td>Front</td><td>60.98</td><td>55.25</td><td>45.73</td></tr><tr><td>Not Front</td><td>39.01</td><td>44.75</td><td>54.26</td></tr></table>

Table 2: Percentage of age, gender, race, and head pose attributes predicted by FaceXformer for Section 4.1.

<table><tr><td colspan="2">LFWA Attribute</td><td>FFHQ</td><td>Reference</td><td>Ours(%)</td></tr><tr><td rowspan="11">&lt;10%</td><td>PaleSkin</td><td>0.12</td><td>0.20</td><td>0.37</td></tr><tr><td>Mustache</td><td>0.39</td><td>0.30</td><td>0.46</td></tr><tr><td>Bald</td><td>0.88</td><td>1.20</td><td>2.18</td></tr><tr><td>WearingNecktie</td><td>0.96</td><td>0.80</td><td>1.23</td></tr><tr><td>PointyNose</td><td>1.42</td><td>1.80</td><td>2.95</td></tr><tr><td>GrayHair</td><td>2.43</td><td>1.90</td><td>3.12</td></tr><tr><td>RecedingHairline</td><td>3.32</td><td>2.40</td><td>3.92</td></tr><tr><td>BigLips</td><td>4.66</td><td>5.90</td><td>7.26</td></tr><tr><td>BlondHair</td><td>5.04</td><td>5.20</td><td>5.72</td></tr><tr><td>Eyeglasses</td><td>6.10</td><td>5.70</td><td>6.51</td></tr><tr><td>WearingHat</td><td>8.07</td><td>7.80</td><td>11.03</td></tr><tr><td>&gt;10%</td><td>BrownHair</td><td>19.76</td><td>22.92</td><td>13.78</td></tr></table>

Table 3: Percentage of LFWA attributes predicted by FaceX-former for Section 4.1. Sorted in descending order of FFHQ(%). The entire table is in Table 12.

Generated Rare Facial Attributes Our method successfully increases the percentages of rare attributes as shown in Tables 2 and 3. To identify rare facial attributes in FFHQ, we employ FaceXFormer (Narayan et al. 2024), which provides multiple face-related features including age, gender, race, head pose, and the attributes from Deep Learning Face Attributes in the Wild (LFWA) dataset (Liu et al. 2015). Real data attributes with lower percentages include extreme ages (very young or old), male gender, non-white races, non-frontal head poses $^{2}$ , non-natural skin colors, hairless or no hair, hair colors other than brown, and accessories.

![](images/d9a04269ccf1445069d74c5e613a2817940f0efa071891d643a683326f41e4a4.jpg)

<details>
<summary>natural_image</summary>

Group photo collage of children's face wearing various colorful costumes and hats (no text or symbols visible)
</details>

High $k$ -NND Real Samples

![](images/49b04cbdcbd80826cc08dfb1e6a0c6b3d502d6b1d705acbcc8cd62c107657376.jpg)

<details>
<summary>natural_image</summary>

Grid of 16 diverse face and face portraits with painted faces, no text or symbols visible
</details>

High Rarity Score Fake Samples

![](images/1b99a1c0fd64c3156f38b60d16a709ab7b9219f721cff6b5ef2b0c2e66c7f676.jpg)

<details>
<summary>natural_image</summary>

Collage of 16 diverse face and face photos, including a girl with a party hat, a man in a blue shirt, and others in camouflage uniforms (no visible text or symbols)
</details>

Low Likelihood Real Samples

![](images/12abf2b7de138b3d69de35d4396c7b357cc214d00cee801c42f99becd0f37f78.jpg)

<details>
<summary>natural_image</summary>

Grid of 16 diverse face and makeup photos wearing various outfits and hats, including traditional attire and face masks (no text or symbols visible)
</details>

Low Likelihood Fake Samples

![](images/85b156392b5faa3cd4bc7f4588b823ceea47786bc5bcce0243b66b1d62df7fbf.jpg)

<details>
<summary>natural_image</summary>

Group photo of ten individuals with different facial expressions and hairstyles (no text or symbols visible)
</details>

Out of k-NN Manifold Fake Samples (Rarity Score=N/A)   
Figure 3: Examples of high- and low-density real and fake samples for Section 4.1.

Additionally, other rare attributes can be identified qualitatively. We select and visualize the top- and bottom-ranked samples based on k-NN-based and likelihood-based density estimates in Fig. 3. For real samples, k-NND (Loftsgaarden and Quesenberry 1965) is employed, while the rarity score (Han et al. 2023) is used for fake samples. Likelihoods are estimated using the NF model, excluding samples outside the real k-NN manifold. Rare samples exhibit characteristics such as objects obscuring faces, face painting, various hats, colorful eyeglasses, and artifacts. Notably, samples with undefined (N/A) rarity scores may include high-fidelity, artifact-free images, which arise from underestimated regions in the k-NN manifold.

Qualitative Results As shown in Fig. 1 (Right) and 4, our method generates samples with rare attributes, including hats, hair colors other than natural brown, very young or old age, non-white races such as Black, Indian, and Asian, non-frontal head poses, eyeglasses, bald or receding hairline, colorful backgrounds or T-shirts, hair accessories, and unique skin colors. Moreover, the rare samples generated by our method show diversity, as shown in Fig. 1 (Left) and 5. Starting from initial vectors with small noise variations, we generate diverse rare images that retain perceptual similarity to the reference. Additional examples are provided in Appendix D.

# 4.2 Animal Face and Artwork Generation with StyleGAN2-ADA

Quantitative & Qualitative Results As a baseline, 5,000 synthetic samples are generated using latent vectors from StyleGAN2-ADA with a truncation parameter of $\psi = 1.0$ . With our method, we generate five rare samples for each of 1,000 initial latent vectors from the baseline, using parameters $\lambda_{1} = 200.0$ , $\lambda_{2} = 0.02$ , $\sigma = 0.01$ and $k' = 100$ .

We also evaluate the results using various metrics, as presented in Table 4. Given that the AFHQ and Met-

![](images/3404b69625e6a4345572dae829842123ef5bc26aac14905cea691f8cb4bd3686.jpg)

<details>
<summary>text_image</summary>

Initial
↓
Rare
Not Front Head Pose
Eyeglasses
Initial
↓
Rare
Bald or Receding Hairline
Colorful Background or T-shirt
Initial
↓
Rare
Hair Accessory
Unique Skin Color
</details>

Figure 4: Examples of rare samples generated by our method for Section 4.1.

![](images/b02b71bf407988c71738e33153be6de622d448131f21ff1f2776a886370c5cfa.jpg)

<details>
<summary>text_image</summary>

Initial
Generated Diverse Rare Samples
</details>

Figure 5: Examples of diverse rare samples generated by our method for Section 4.1.

Faces datasets are relatively small, we use the KID score (Bińkowski et al. 2018) instead of the FID score, as the KID score is inherently unbiased (Karras et al. 2020a). Our method effectively enhances rarity and diversity across all three datasets compared to the baseline. In Fig. 7, we visualize the examples generated by our method. More examples are in Fig. 14 and 16.

Generated Rare Attributes of Animal Face and Artwork To identify the rare cat and dog breeds in AFHQ datasets, we employed Model Soups (Wortsman et al. 2022) for zero-shot classification of cat and dog classes in the ImageNet dataset (Deng et al. 2009), which includes five cat classes and 120 dog classes. In the AFHQ Dog dataset, each dog class represents less than 5% of the total, so we grouped the dogs into broader categories based on appearance: Toy, Hound, Scent Hound, Terrier, Sporting, Non-Sporting, Herding, and Working. Further details are provided in Appendix E. The classified result is shown in Table 5, and our method successfully increases the percentages of minor classes.

We also apply the FaceXFormer and observe similar rare attributes in FFHQ, as shown in Table 6.

To further identify rare attributes within the datasets, we visualize the high- and low-likelihood samples in Fig. 6. For the AFHQ-cat dataset, high-likelihood samples predominantly consist of brown-colored Tabby cats, whereas low-

<table><tr><td>Data</td><td>Model</td><td>RS ↑</td><td>Prec. ↑</td><td>Rec. ↑</td><td>LPIPS ↑</td><td>KID ↓ ×103</td></tr><tr><td rowspan="2">AFHQ Cat</td><td>Ref.</td><td>17.47</td><td>0.76</td><td>0.49</td><td>0.73</td><td>0.46</td></tr><tr><td>Ours</td><td>20.71</td><td>0.76</td><td>0.68</td><td>0.78</td><td>2.03</td></tr><tr><td rowspan="2">AFHQ Dog</td><td>Ref.</td><td>21.19</td><td>0.77</td><td>0.56</td><td>0.75</td><td>1.10</td></tr><tr><td>Ours</td><td>27.32</td><td>0.85</td><td>0.76</td><td>0.80</td><td>2.73</td></tr><tr><td rowspan="2">MetFaces</td><td>Ref.</td><td>16.90</td><td>0.80</td><td>0.44</td><td>0.74</td><td>0.97</td></tr><tr><td>Ours</td><td>21.25</td><td>0.86</td><td>0.62</td><td>0.78</td><td>2.15</td></tr></table>

Table 4: Quantitative evaluation for Section 4.2.

![](images/405c3a82976ed0016715773eec600a529d91431f4a1d3d7955e37fbe9ded2a81.jpg)  
Figure 6: Examples of high- and low-likelihood real samples from the AFHQ Cat, Dog, and MetFaces dataset.

likelihood samples encompass a broader range of classes. In the AFHQ-dog dataset, high-likelihood samples are primarily drawn from the Herding and Sporting groups, including breeds such as Shetland Sheepdogs, Collies, and Retrievers. In contrast, low-likelihood samples span a variety of groups and exhibit greater diversity in head poses, backgrounds, and facial expressions. In the Metfaces dataset, high-likelihood samples predominantly include European-style oil paintings, while low-likelihood samples include statues, drawings, and other styles of paintings. As shown in Fig. 7, such rare attributes can be also observed in the results of our method.

# 4.3 Ablation Study on the Objective Function

Our objective function includes three components: rarity, similarity, and diversity terms. To evaluate their effectiveness, we conduct an ablation study using the FFHQ dataset and StyleGAN2, keeping the density estimator and parameters consistent. We optimize ten samples for each of the 100 initial latent vectors across different objective combinations.

First, we assess results using only the $L_{rare}$ . Adding the $L_{sim}$ ensures that samples stay within a similarity boundary to the reference, potentially finding rarer and more diverse

![](images/d33bc670ee70ea40f94b9734bdc623754b87fa402122c56d5b9f97bc3edc3edf.jpg)

Figure 7: Examples of rare samples generated by our method for Section 4.2. 

<table><tr><td colspan="2">ImageNet Attribute</td><td>Real</td><td>Reference</td><td>Ours(%)</td></tr><tr><td rowspan="2">Cat</td><td>Tabby</td><td>56.76</td><td>58.00</td><td>55.10</td></tr><tr><td>Others</td><td>43.23</td><td>42.00</td><td>44.90</td></tr><tr><td rowspan="8">Dog</td><td>Herding</td><td>21.48</td><td>21.70</td><td>17.68</td></tr><tr><td>Sporting</td><td>18.50</td><td>17.30</td><td>15.24</td></tr><tr><td>Working</td><td>15.44</td><td>14.60</td><td>16.84</td></tr><tr><td>Toy</td><td>14.93</td><td>17.30</td><td>19.44</td></tr><tr><td>Terrier</td><td>9.11</td><td>9.60</td><td>7.72</td></tr><tr><td>Non-Sporting</td><td>7.02</td><td>5.90</td><td>7.38</td></tr><tr><td>Scent Hound</td><td>6.92</td><td>8.10</td><td>8.92</td></tr><tr><td>Hound</td><td>3.05</td><td>2.60</td><td>3.76</td></tr></table>

Table 5: Percentage of the cat-related breeds and dog-related groups in ImageNet classes predicted by Model Soups. Sorted in descending order of Real(%). The entire table is in Table 14.

<table><tr><td colspan="2">Model-Based Attribute</td><td>MetF.</td><td>Reference</td><td>Ours(%)</td></tr><tr><td rowspan="3">Age</td><td>0-9</td><td>5.19</td><td>3.23</td><td>5.21</td></tr><tr><td>10-69</td><td>94.20</td><td>95.96</td><td>93.55</td></tr><tr><td>Over70</td><td>0.60</td><td>0.80</td><td>1.23</td></tr><tr><td rowspan="2">Gender</td><td>Male</td><td>42.84</td><td>41.71</td><td>49.62</td></tr><tr><td>Female</td><td>57.15</td><td>58.28</td><td>50.37</td></tr><tr><td rowspan="5">Race</td><td>White</td><td>73.64</td><td>77.97</td><td>77.96</td></tr><tr><td>Indian</td><td>9.86</td><td>9.19</td><td>9.03</td></tr><tr><td>Black</td><td>3.38</td><td>2.02</td><td>2.33</td></tr><tr><td>Asian</td><td>2.86</td><td>4.04</td><td>5.25</td></tr><tr><td>Others</td><td>10.24</td><td>6.76</td><td>5.40</td></tr><tr><td rowspan="2">HeadPose</td><td>Front</td><td>42.54</td><td>46.06</td><td>32.86</td></tr><tr><td>NotFront</td><td>57.45</td><td>53.93</td><td>67.14</td></tr><tr><td rowspan="2">LFWA</td><td>Eyeglasses</td><td>0.07</td><td>0.10</td><td>0.33</td></tr><tr><td>WearingHat</td><td>12.34</td><td>9.89</td><td>12.18</td></tr></table>

Table 6: Percentage of age, gender, race, head pose, Eye-glasses, and WearingHat attributes predicted by FaceX-Former. MetF. refers to the MetFaces dataset. The entire table is in Table 13.

<table><tr><td>Objective</td><td>RS</td><td>LPIPS</td></tr><tr><td> $\mathcal{L}_{\text{rare}}$ </td><td>18.99</td><td>0.752</td></tr><tr><td> $\mathcal{L}_{\text{rare}} + \lambda_1 \mathcal{L}_{\text{sim}}$ </td><td>21.11</td><td>0.766</td></tr><tr><td> $\mathcal{L}_{\text{rare}} + \lambda_1 \mathcal{L}_{\text{sim}} + \lambda_2 \mathcal{L}_{\text{div}}$ </td><td>21.28</td><td>0.768</td></tr></table>

Table 7: Rarity scores (RS) and LPIPS scores for the ablation study on the objective function.

![](images/3c62f036c11136bbf0a91dae0769b4da7b5ab8f3a69b96df6a8d49abe1a36a95.jpg)

<details>
<summary>scatter</summary>

| k-NND of real samples (k=3) | -log p(x) |
| --------------------------- | --------- |
| (various points)            | (various points) |
</details>

![](images/5ce2c1dcd7d46b63400e99b423267eeb51bde0026251a13aa6832f6d26f887d4.jpg)

<details>
<summary>scatter</summary>

| Rarity score of fake samples | Pearson corr. |
| ---------------------------- | ------------- |
| Value                        | 0.815         |
</details>

Figure 8: Correlation plot for the k-NND / rarity score and negative log-likelihood estimated by the normalizing flow.

samples inside the boundary. Finally, incorporating the $L_{div}$ completes the full objective. The results in Table 7 demonstrate the effectiveness of each term.

# 4.4 Relationship with Rarity Score

We use the rarity score (Han et al. 2023) to measure sample rarity and demonstrate that our method improves rarity compared to other sampling methods. Although directly optimizing the rarity score is challenging, our likelihood-based objective effectively guides samples to locally low-density regions. To compare k-NN-based density measures (k-NND for real samples and rarity score for fake samples) with NF-estimated density measures, we visualize the scatter plot and compute the Pearson correlation coefficient as represented in Fig. 8. We observe a high Pearson correlation coefficient of 0.928 for real samples and 0.815 for fake samples, with p-values $< 10^{-8}$ , excluding samples with undefined rarity scores.

Although the NF estimates likelihood across the feature space, the rarity score is undefined outside the real k-NN manifold. This allows for out-of-manifold samples with sufficient quality, as shown at the bottom of Fig. 3. In Fig. 9, we visualize an optimization example that starts with an undefined rarity score but eventually gains and increases the rarity score. The sample, initially in an underestimated k-NN region, becomes rare by moving to a low-likelihood region, altering the reference image to achieve curlier blonde hair and a non-frontal head pose. We plot the NF-estimated density using RBF kernel interpolation and UMAP (McInnes, Healy, and Melville 2018) dimensionality reduction on the real feature space and its inverse transformation function. Further details are provided in Appendix F.

# 5 Conclusion

We proposed a novel algorithm that generates diverse rare samples using multi-start gradient-based optimization, avoiding low-quality samples. Users can control rarity, diversity, and similarity to the reference through a multi-objective approach. Our method successfully increased the prevalence of rare attributes in various image generation domains. We also provide an experimental comparison between k-NN-based and normalizing flow-based density estimation methods. We hope this work contributes to advancing creativity in deep generative models. However, there are some limitations that could be improved. The results rely on the GAN's capabilities and require an additional density estimator. Exploring other generative models might improve outcomes and eliminate the need for extra training. Additionally, our method alters multiple attributes simultaneously; integrating it with other image manipulation techniques could allow for more controlled manipulation.

![](images/280af73880cb6d10cc4ece4a6c545ab3568708fcb62893c4dadc8695c0d9e86c.jpg)

<details>
<summary>line</summary>

| Optimization Step | Rarity Score | log p(x) |
| ----------------- | ------------ | -------- |
| 0                 | 22           | 5000     |
| 10                | 22           | 5000     |
| 20                | 12           | 5000     |
| 30                | 14           | 7000     |
| 40                | 8            | 7000     |
| 50                | 6            | 7000     |
| 60                | 4            | 7000     |
</details>

![](images/d625c457f82d44a1b59901de11cae7cf874c7ffd3bb4cafd708071dadccb0b5e.jpg)

<details>
<summary>scatter</summary>

| Point Type              | UMAP axis 1 | UMAP axis 2 |
| ----------------------- | ----------- | ----------- |
| Initial point           | ~0.8        | ~0.9        |
| Optimized point (RS=N/A) | ~0.6        | ~0.7        |
| Real point              | ~0.4        | ~0.5        |
| k-NN Real Manifold      | ~0.3        | ~0.4        |
</details>

Figure 9: Example of the optimization path with a real k-NN manifold and a heatmap of likelihoods estimated by the normalizing flow. Notably, the local k-NN manifold includes only the three nearest real data points $(k = 3)$ for each point, rather than the entire manifold.

# Acknowledgments

This work was partly supported by KAIST-NAVER Hypercreative AI Center, and from the Korean Institute of Information & Communications Technology Planning & Evaluation and the Korean Ministry of Science and ICT under grant agreement No. RS-2019-II190075 (Artificial Intelligence Graduate School Program(KAIST)), No. RS-2022-II220984 (Development of Artificial Intelligence Technology for Personalized Plug-and-Play Explanation and Verification of Explanation), and No.RS-2022-II220184 (Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).

# References

Agarwal, C.; D'souza, D.; and Hooker, S. 2022. Estimating example difficulty using variance of gradients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10368–10378.

Allahyani, M.; Alsulami, R.; Alwafi, T.; Alafif, T.; Ammar, H.; Sabban, S.; and Chen, X. 2023. DivGAN: A diversity enforcing generative adversarial network for mode collapse reduction. Artificial Intelligence, 317: 103863.

Amabile, T. M. 2018. Creativity in context: Update to the social psychology of creativity. Routledge.

Azadi, S.; Olsson, C.; Darrell, T.; Goodfellow, I.; and Odena, A. 2018. Discriminator rejection sampling. arXiv preprint arXiv:1810.06758.

Bińkowski, M.; Sutherland, D. J.; Arbel, M.; and Gretton, A. 2018.

Demystifying mmd gans. arXiv preprint arXiv:1801.01401.

Brock, A.; Donahue, J.; and Simonyan, K. 2018. Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096.

Casanova, A.; Careil, M.; Verbeek, J.; Drozdzal, M.; and Romero Soriano, A. 2021. Instance-conditioned gan. Advances in Neural Information Processing Systems, 34: 27517–27529.

Chang, A.; Fontaine, M. C.; Booth, S.; Matarić, M. J.; and Nikolaidis, S. 2024. Quality-Diversity Generative Sampling for Learning with Synthetic Data. Proceedings of the AAAI Conference on Artificial Intelligence, 38(18): 19805–19812.

Chen, X.; Duan, Y.; Houthooft, R.; Schulman, J.; Sutskever, I.; and Abbeel, P. 2016. Infogan: Interpretable representation learning by information maximizing generative adversarial nets. Advances in neural information processing systems, 29.

Choi, Y.; Uh, Y.; Yoo, J.; and Ha, J.-W. 2020. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8188–8197.

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-Fei, L. 2009. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, 248–255. Ieee.

DeVries, T.; Drozdzal, M.; and Taylor, G. W. 2020. Instance selection for gans. Advances in Neural Information Processing Systems, 33: 13285–13296.

Diederik, P. K. 2014. Adam: A method for stochastic optimization. (No Title).

Dinh, L.; Krueger, D.; and Bengio, Y. 2014. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516.

Dinh, L.; Sohl-Dickstein, J.; and Bengio, S. 2016. Density estimation using real nvp. arXiv preprint arXiv:1605.08803.

Esser, P.; Rombach, R.; and Ommer, B. 2020. A disentangling invertible interpretation network for explaining latent representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9223–9232.

Feo, T. A.; and Resende, M. G. 1995. Greedy randomized adaptive search procedures. Journal of global optimization, 6: 109–133.

Ghosh, A.; Kulharia, V.; Namboodiri, V. P.; Torr, P. H.; and Dokania, P. K. 2018. Multi-agent diverse generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 8513–8521.

Gretton, A.; Borgwardt, K.; Rasch, M.; Schölkopf, B.; and Smola, A. 2006. A kernel method for the two-sample-problem. Advances in neural information processing systems, 19.

Han, J.; Choi, H.; Choi, Y.; Kim, J.; Ha, J.-W.; and Choi, J. 2023. Rarity Score: A New Metric to Evaluate the Uncommonness of Synthesized Images. In International Conference on Learning Representations (ICLR). International Conference on Learning Representations.

Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. Gans Trained by a Two Time-scale Update Rule Converge to a Local Nash Equilibrium. Advances in neural information processing systems, 30.

Heyrani Nobari, A.; Rashad, M. F.; and Ahmed, F. 2021. Creativegan: Editing generative adversarial networks for creative design synthesis. In International Design Engineering Technical Conferences and Computers and Information in Engineering Conference, volume 85383, V03AT03A002. American Society of Mechanical Engineers.

Humayun, A. I.; Balestriero, R.; and Baraniuk, R. 2021. MaGNET: Uniform sampling from deep generative network manifolds without retraining. arXiv preprint arXiv:2110.08009.

Humayun, A. I.; Balestriero, R.; and Baraniuk, R. 2022. Polarity sampling: Quality and diversity control of pre-trained generative networks via singular values. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10641–10650.

Hwang, S.; Park, S.; Kim, D.; Do, M.; and Byun, H. 2020. Fair-facegan: Fairness-aware facial image-to-image translation. arXiv preprint arXiv:2012.00282.

Karras, T.; Aittala, M.; Hellsten, J.; Laine, S.; Lehtinen, J.; and Aila, T. 2020a. Training generative adversarial networks with limited data. Advances in neural information processing systems, 33:12104–12114.

Karras, T.; Laine, S.; and Aila, T. 2019. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4401–4410.

Karras, T.; Laine, S.; Aittala, M.; Hellsten, J.; Lehtinen, J.; and Aila, T. 2020b. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8110–8119.

Kim, E.-J.; and Bansal, P. 2023. A deep generative model for feasible and diverse population synthesis. Transportation Research Part C: Emerging Technologies, 148: 104053.

Kingma, D. P.; and Dhariwal, P. 2018. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31.

Kirichenko, P.; Izmailov, P.; and Wilson, A. G. 2020. Why normalizing flows fail to detect out-of-distribution data. Advances in neural information processing systems, 33: 20578–20589.

Kynkäänniemi, T.; Karras, T.; Laine, S.; Lehtinen, J.; and Aila, T. 2019. Improved precision and recall metric for assessing generative models. In NeurIPS.

Lee, J.; Kim, H.; Hong, Y.; and Chung, H. W. 2021. Self-diagnosing gan: Diagnosing underrepresented samples in generative adversarial networks. Advances in Neural Information Processing Systems, 34: 1925–1938.

Liu, Z.; Luo, P.; Wang, X.; and Tang, X. 2015. Deep Learning Face Attributes in the Wild. In Proceedings of International Conference on Computer Vision (ICCV).

Loftsgaarden, D. O.; and Quesenberry, C. P. 1965. A nonparametric estimate of a multivariate density function. The Annals of Mathematical Statistics, 36(3): 1049–1051.

Lynn, M.; and Harris, J. 1997. Individual differences in the pursuit of self-uniqueness through consumption. Journal of Applied Social Psychology, 27(21): 1861–1883.   
Ma, Z.; Mei, G.; and Xu, N. 2024. Generative deep learning for data generation in natural hazard analysis: motivations, advances, challenges, and opportunities. Artificial Intelligence Review, 57(6):160.   
McInnes, L.; Healy, J.; and Melville, J. 2018. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.   
Naeem, M. F.; Oh, S. J.; Uh, Y.; Choi, Y.; and Yoo, J. 2020. Reliable fidelity and diversity metrics for generative models. In ICML.   
Narayan, K.; VS, V.; Chellappa, R.; and Patel, V. M. 2024. FaceXFormer: A Unified Transformer for Facial Analysis. arXiv preprint arXiv:2403.12960.   
OpenAI. 2023. ChatGPT: GPT-4 Technical Report. OpenAI Research. https://openai.com/research/gpt-4.   
Papamakarios, G.; Nalisnick, E.; Rezende, D. J.; Mohamed, S.; and Lakshminarayanan, B. 2021. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57): 1–64.   
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PMLR.   
Rinnooy Kan, A.; and Timmer, G. T. 1987. Stochastic global optimization methods part I: Clustering methods. Mathematical programming, 39: 27–56.   
Rochat, Y.; and Taillard, É. D. 1995. Probabilistic diversification and intensification in local search for vehicle routing. Journal of heuristics, 1: 147–167.   
Sagar, D.; Risheh, A.; Sheikh, N.; and Forouzesh, N. 2023. Physics-Guided Deep Generative Model For New Ligand Discovery. In Proceedings of the 14th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics, 1–9.   
Sehwag, V.; Hazirbas, C.; Gordo, A.; Ozgenel, F.; and Canton, C. 2022. Generating high fidelity data from low-density regions using diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11492–11501.   
Simonyan, K.; and Zisserman, A. 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.   
Simonyan, K.; and Zisserman, A. 2015. Very Deep Convolutional Networks for Large-Scale Image Recognition. In ICLR.   
Snyder, C. R.; and Lopez, S. J. 2001. Handbook of positive psychology. Oxford university press.   
Srivastava, A.; Valkov, L.; Russell, C.; Gutmann, M. U.; and Sutton, C. 2017. Veegan: Reducing mode collapse in gans using implicit variational learning. Advances in neural information processing systems, 30.   
Tarek, M.; and Huang, Y. 2022. Simplifying deflation for non-convex optimization with applications in Bayesian inference and topology optimization. arXiv preprint arXiv:2201.11926.   
Teo, C. T.; Abdollahzadeh, M.; and Cheung, N.-M. 2023. Fair generative models via transfer learning. Proceedings of the AAAI conference on artificial intelligence, 37(2): 2429–2437.   
Thanh-Tung, H.; and Tran, T. 2020. Catastrophic forgetting and mode collapse in GANs. In 2020 international joint conference on neural networks (ijcnn), 1–10. IEEE.

Tolstikhin, I. O.; Gelly, S.; Bousquet, O.; Simon-Gabriel, C.-J.; and Schölkopf, B. 2017. Adagan: Boosting generative models. Advances in neural information processing systems, 30.

Turner, R.; Hung, J.; Frank, E.; Saatchi, Y.; and Yosinski, J. 2019. Metropolis-hastings generative adversarial networks. In International Conference on Machine Learning, 6345–6353. PMLR.

Virtanen, P.; Gommers, R.; Oliphant, T. E.; Haberland, M.; Reddy, T.; Cournapeau, D.; Burovski, E.; Peterson, P.; Weckesser, W.; Bright, J.; van der Walt, S. J.; Brett, M.; Wilson, J.; Millman, K. J.; Mayorov, N.; Nelson, A. R. J.; Jones, E.; Kern, R.; Larson, E.; Carey, C. J.; Polat, I.; Feng, Y.; Moore, E. W.; VanderPlas, J.; Laxalde, D.; Perktold, J.; Cimrman, R.; Henriksen, I.; Quintero, E. A.; Harris, C. R.; Archibald, A. M.; Ribeiro, A. H.; Pedregosa, F.; van Mulbregt, P.; and SciPy 1.0 Contributors. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17: 261–272.

Wortsman, M.; Ilharco, G.; Gadre, S. Y.; Roelofs, R.; Gontijo-Lopes, R.; Morcos, A. S.; Namkoong, H.; Farhadi, A.; Carmon, Y.; Kornblith, S.; et al. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, 23965–23998. PMLR.

Xia, M.; Shu, Y.; Wang, Y.; Lai, Y.-K.; Li, Q.; Wan, P.; Wang, Z.; and Liu, Y.-J. 2023. FEditNet: few-shot editing of latent semantics in GAN spaces. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 2919–2927.

Yang, C.; Shen, Y.; Zhang, Z.; Xu, Y.; Zhu, J.; Wu, Z.; and Zhou, B. 2023. One-shot generative domain adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7733–7742.

Zeng, X.; Wang, F.; Luo, Y.; Kang, S.-g.; Tang, J.; Lightstone, F. C.; Fang, E. F.; Cornell, W.; Nussinov, R.; and Cheng, F. 2022. Deep generative molecular design reshapes drug discovery. Cell Reports Medicine, 3(12).

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586–595.

# A k-NN-based Evaluation Metrics

k-NN-based manifold estimation is employed in various metrics to assess both fidelity and diversity of synthetic samples (Kynkäänniemi et al. 2019; Naeem et al. 2020). For real samples $I_{r} \sim P_{r}$ and fake samples $I_{g} \sim P_{g}$ , they are embedded in the feature space with the pretrained DNNs such as VGG16 (Simonyan and Zisserman 2015) or the CLIP image encoder (Radford et al. 2021) to get sets of feature vectors $X_{r}$ and $X_{g}$ , respectively. The real and fake manifolds are estimated by the given sample sets as follows.

$$
\Phi_ {\mathbf {X}} = \bigcup_ {\mathbf {x} _ {i} \in \mathbf {X}} B _ {k} (\mathbf {x} _ {i}, \mathbf {X}) \tag {3}
$$

$$
B _ {k} (\mathbf {x} _ {i}, \mathbf {X}) = \{\mathbf {x} | d (\mathbf {x} _ {i}, \mathbf {x}) \leq k \text {-NND} (\mathbf {x} _ {i}, \mathbf {X}) \}
$$

Here, $k-\mathrm{NND}(\mathbf{x}_{i},\mathbf{X})$ represents the distance between $x_{i}$ and its k-th nearest neighbor in X. $B_{k}(\mathbf{x}_{i},\mathbf{X})$ is the k-NN ball (hyper-sphere) with the radius of $k-\mathrm{NND}(\mathbf{x}_{i},\mathbf{X})$ centered at $x_{i}$ defined as a set of all x whose distance to $x_{i}$ is smaller than or equal to $k-\mathrm{NND}(\mathbf{x}_{i},\mathbf{X})$ . For simplicity, we use $\Psi_{real} = \Psi_{X_{r}}$ through the paper.

We utilize three k-NN-based evaluation metrics in the quantitative analysis, precision, recall, and rarity score. Precision (Kynkäänniemi et al. 2019) measures the proportion of fake samples within the real manifold, indicating how realistic the fake samples are. Recall (Kynkäänniemi et al. 2019) measures the proportion of real samples within the fake manifold, assessing how well the generative model captures the modes of the real data distribution. The rarity score (Han et al. 2023) measures the uniqueness of individual samples, with the following formulation.

$$
\operatorname{rarity} \left(\mathbf {x} _ {g}, \mathbf {X} _ {\mathbf {r}}\right) = \min _ {r, s. t. \mathbf {x} _ {g} \in B _ {k} \left(\mathbf {x} _ {r}, \mathbf {X} _ {\mathbf {r}}\right)} k - \mathrm{NND} \left(\mathbf {x} _ {r}, \mathbf {X} _ {\mathbf {r}}\right). \tag {4}
$$

# B Implementation Details

# B.1 Computational Resources

For all experiments including model training and inference, and optimization, we utilized a single NVIDIA RTX A6000 GPU with PyTorch version of 2.1.0+cu121.

# B.2 Normalizing Flow Architecture

We use the Glow architecture proposed by Kingma and Dhariwal (2018) for our density estimation model, adapting it from its original design for RGB images to work in a feature space with a dimension of $R^{4096}$ , representing the second-to-last latent space of the VGG16-fc2 model (Simonyan and Zisserman 2014). While retaining the original Glow structure—stacked blocks of sequential flows with Actnorm, invertible $1 \times 1$ convolution, affine coupling, and split layers—we modify Actnorm layers for grouped channel-wise operations, since our feature vector lacks a patch-like structure. To achieve this, we introduce a $1 \times 1$ convolution layer for general permutation before each Actnorm layer, followed by dividing the dimensions into a user-defined number of groups. This modification makes the model lighter and more efficient. The finalized architecture is represented in Table 8.

<table><tr><td>Block</td><td colspan="2">Layer</td><td>Input dim</td><td>Output collect</td></tr><tr><td rowspan="6">0</td><td colspan="2">1 × 1 conv</td><td>1 × 4096</td><td>-</td></tr><tr><td colspan="2">grouping</td><td>4 × 1024</td><td>-</td></tr><tr><td rowspan="3">flow (× 32)</td><td>actnorm</td><td>4 × 1024</td><td>-</td></tr><tr><td>1 × 1 conv</td><td>4 × 1024</td><td>-</td></tr><tr><td>affine coupling</td><td>4 × 1024</td><td>-</td></tr><tr><td colspan="2">split</td><td>2 × 1024</td><td>yes</td></tr><tr><td rowspan="6">1</td><td colspan="2">1 × 1 conv</td><td>2 × 1024</td><td>-</td></tr><tr><td colspan="2">grouping</td><td>8 × 256</td><td>-</td></tr><tr><td rowspan="3">flow (× 32)</td><td>actnorm</td><td>8 × 256</td><td>-</td></tr><tr><td>1 × 1 conv</td><td>8 × 256</td><td>-</td></tr><tr><td>affine coupling</td><td>8 × 256</td><td>-</td></tr><tr><td colspan="2">split</td><td>4 × 256</td><td>yes</td></tr><tr><td rowspan="6">2</td><td colspan="2">1 × 1 conv</td><td>4 × 256</td><td>-</td></tr><tr><td colspan="2">grouping</td><td>16 × 64</td><td>-</td></tr><tr><td rowspan="3">flow (× 32)</td><td>actnorm</td><td>16 × 64</td><td>-</td></tr><tr><td>1 × 1 conv</td><td>16 × 64</td><td>-</td></tr><tr><td>affine coupling</td><td>16 × 64</td><td>-</td></tr><tr><td colspan="2">split</td><td>8 × 64</td><td>yes</td></tr><tr><td rowspan="5">3</td><td colspan="2">1 × 1 conv</td><td>8 × 64</td><td>-</td></tr><tr><td colspan="2">grouping</td><td>32 × 16</td><td>-</td></tr><tr><td rowspan="3">flow (× 32)</td><td>actnorm</td><td>32 × 16</td><td>-</td></tr><tr><td>1 × 1 conv</td><td>32 × 16</td><td>-</td></tr><tr><td>affine coupling</td><td>32 × 16</td><td>yes</td></tr></table>

Table 8: Normalizing flow model architecture.

We train the NF model with 70,000 samples for FFHQ, 5,653 for AFHQ Cat, 5,239 for AFHQ Dog, and 1,336 for MetFaces, splitting FFHQ and MetFaces images 7:3 for training and validation, and using the provided original splits for the AFHQ datasets. We use a batch size of 32, scale data to [0, 1] by a min-max scaler, and apply the Adam optimizer (Diederik 2014) with a learning rate of $1 \times 10^{-4}$ and the StepLR scheduler with a step size of 500, gamma of 0.1. The number of flows, blocks, and groups for the modified Actnorm are 32, 4, and 4, respectively. The best checkpoint was obtained at 3,000, 2,000, and 1,500 iterations for FFHQ, AFHQ, and MetFaces. Training takes less than 30 minutes on a single GPU.

# B.3 Settings for Optimization

For the diverse rare sample optimization, we use a maximum number of 200 epochs. We use the StepLR scheduler from the PyTorch package for learning rates, with the gamma of 0.9. The step sizes are set to 50 for the FFHQ-StyleGAN2 experiments and 100 for the AFHQ-StyleGAN2-ADA and MetFaces-StyleGAN2-ADA experiments. For other parameters, the default settings are used.

In the FFHQ-StyleGAN2 experiments, the average time of the optimization for the one initial latent vector with N = 10 is less than 8 minutes with a single GPU.

# C Choice of Parameters

Coefficients for Objective Function $\lambda_{1}$ and $\lambda_{2}$ . We can control the strength of similarity regularization and diversity by adjusting the coefficients $\lambda_{1}$ and $\lambda_{2}$ , respectively. A

<table><tr><td> $\lambda_1$ </td><td> $\lambda_2$ </td><td>Rarity Score↑</td><td>LPIPS↑</td><td>OOM↓(%)</td></tr><tr><td>20</td><td>0.002</td><td>23.92</td><td>0.769</td><td>11.92</td></tr><tr><td>30</td><td>0.002</td><td>24.02</td><td>0.768</td><td>11.40</td></tr><tr><td>40</td><td>0.002</td><td>24.04</td><td>0.767</td><td>11.79</td></tr><tr><td>30</td><td>0.001</td><td>23.92</td><td>0.766</td><td>12.10</td></tr><tr><td>30</td><td>0.003</td><td>23.87</td><td>0.767</td><td>13.00</td></tr><tr><td>30</td><td>0.005</td><td>23.72</td><td>0.769</td><td>12.20</td></tr></table>

Table 9: Rarity score and LPIPS score from our method with varying $\lambda_{1}$ and $\lambda_{2}$ , experimenting on FFHQ-StyleGAN2. OOM refers to out-of-manifold sample percentage.

<table><tr><td>k&#x27;</td><td>Rarity Score↑</td><td>LPIPS↑</td><td>LPIPS*↓</td><td>OOM↓(%)</td></tr><tr><td>80</td><td>23.78</td><td>0.767</td><td>0.523</td><td>13.60</td></tr><tr><td>100</td><td>24.02</td><td>0.768</td><td>0.532</td><td>11.40</td></tr><tr><td>200</td><td>24.23</td><td>0.771</td><td>0.546</td><td>10.70</td></tr></table>

Table 10: Rarity score and LPIPS scores from our method with varying $k'$ . LPIPS: Mean LPIPS score among the optimized samples. LPIPS\*: Mean LPIPS score between reference and the optimized samples, experimenting on FFHQ-StyleGAN2. OOM refers to out-of-manifold sample percentage.

larger $\lambda_{1}$ encourages samples to remain within the penalizing boundary, potentially resulting in rarer and more diverse samples. However, if $\lambda_{1}$ is set too large, it may restrict diversity. On the other hand, increasing $\lambda_{2}$ enhances diversity, but if $\lambda_{2}$ is too large, it may reduce rarity due to the trade-offs inherent in the multi-objective framework.

Varying these coefficients in the FFHQ-StyleGAN2 experimental setting, we visualize the mean rarity score except for the undefined rarity cases and pairwise LPIPS score in Table 9. These scores are calculated on the 1,000 samples generated from our method with 100 initial latent vectors and N = 10. The LPIPS scores are calculated on randomly selected 10,000 pairs. The rarity score increases as $\lambda_{1}$ increases while decreasing as $\lambda_{2}$ increases. The LPIPS score increases as $\lambda_{2}$ increases while decreasing as $\lambda_{1}$ increases. In the parameter ranges shown in Table 9, both the rarity scores and LPIPS scores are higher than those of the baseline, with only slight differences between the different parameters.

We unify the parameters for each dataset-model pair, however, there can be a better set of parameters for each reference. For instance, if we make $\lambda_{2}$ larger or $\lambda_{1}$ smaller for the samples with low diversity, the result can be more diverse. We provide the examples in Fig. 10.

Parameter for the Penalizing Boundary $k^{\prime}$ The parameter $k^{\prime}$ determines the radius of the penalizing boundary, which controls the similarity between the reference and optimized images. As $k^{\prime}$ increases, the rarity and diversity of the optimized samples also increase, while the similarity between the reference and optimized images decreases, as shown in Table 10. We also use the LPIPS score to measure the similarity between the reference and the optimized image, with lower scores indicating greater similarity (denoted as LPIPS\*).

![](images/676ca74df8880ed47109b906d01a0ca086fe5cc5f18c2f7f64fb7c12ac695f57.jpg)

<details>
<summary>other</summary>

| Initial | Diverse Rare Samples |
| ------- | ------------------- |
| (30.0, 0.002) | (30.0, 0.003) |
| (20.0, 0.002) | (30.0, 0.002) |
| (30.0, 0.003) | (30.0, 0.003) |
| (20.0, 0.002) | (30.0, 0.002) |
</details>

Figure 10: Examples of diverse rare samples generated by our method using FFHQ-StyleGAN2, varying the coefficients of the objective function, $\lambda_{1}$ (similarity) and $\lambda_{2}$ (diversity).

<table><tr><td>Distance Metric</td><td>Rarity Score↑</td><td>LPIPS↑</td><td> $OOM_{\downarrow}(\%)$ </td></tr><tr><td>L2 (Original)</td><td>24.02</td><td>0.768</td><td>11.40</td></tr><tr><td>L1</td><td>24.12</td><td>0.769</td><td>12.60</td></tr><tr><td>Cosine Sim.</td><td>24.00</td><td>0.769</td><td>13.80</td></tr></table>

Table 11: Rarity score and LPIPS score from our method with different distance metrics, experimenting on the FFHQ-StyleGAN2. OOM refers to out-of-manifold sample percentage.

Scale of Noise $\sigma$ The first step in our diverse rare sample generation algorithm involves adding random noise to the initial latent vector to provide multi-starts for the optimization and promote diversity. This noise is sampled from the distribution $\mathcal{N}(\mathbf{0},\sigma^{2}\mathbf{I})$ . While a larger $\sigma$ can yield more diverse results, setting $\sigma$ too high may produce out-of-distribution samples during the early stages of optimization. To balance diversity and realism, we select $\sigma$ values that result in fewer than 30% of the initial perturbed latent vectors generating samples outside the real k-NN manifold. Specifically, 29.96% for FFHQ-StyleGAN2, 22.40% for AFHQ Cat-StyleGAN2-ADA, 19.00% for AFHQ Dog-StyleGAN2-ADA, and 21.40% for MetFaces-StyleGAN2-ADA. We allow some out-of-manifold samples in the initial stage since these may not be true out-of-distribution samples but rather appear as out-of-manifold due to the limitations of the k-NN-based manifold.

Distance function d We employ the Euclidean distance (L2 norm) in the feature space as the distance function for both similarity and diversity constraints, as described in Section 3.1. Experimental results using alternative distance metrics are presented in Table 11. In the FFHQ-StyleGAN2 setting, 1,000 samples are generated using our method with 100 initial latent vectors and N = 10. Compared to results using the L1 norm and cosine similarity, our method achieves a higher rarity score and LPIPS score while generating fewer out-of-manifold samples than standard sampling, regardless of the metric used.

<table><tr><td>LFWA Attribute</td><td>FFHQ</td><td>Reference</td><td>Ours</td><td>LFWA Attribute</td><td>FFHQ</td><td>Reference</td><td>Ours(%)</td></tr><tr><td>Blurry</td><td>0.005</td><td>0</td><td>0.08</td><td>DoubleChin</td><td>11.14</td><td>10.41</td><td>9.66</td></tr><tr><td>WearingNecklace</td><td>0.04</td><td>0</td><td>0</td><td>Bangs</td><td>12.04</td><td>12.31</td><td>12.29</td></tr><tr><td>RosyCheeks</td><td>0.10</td><td>0.30</td><td>0.11</td><td>BushyEyebrows</td><td>12.91</td><td>13.52</td><td>10.30</td></tr><tr><td>PaleSkin</td><td>0.12</td><td>0.20</td><td>0.37</td><td>WavyHair</td><td>13.12</td><td>15.31</td><td>10.50</td></tr><tr><td>Mustache</td><td>0.39</td><td>0.30</td><td>0.46</td><td>NarrowEyes</td><td>13.66</td><td>12.11</td><td>13.59</td></tr><tr><td>Bald</td><td>0.88</td><td>1.20</td><td>2.18</td><td>HeavyMakeup</td><td>14.97</td><td>16.51</td><td>13.31</td></tr><tr><td>WearingNecktie</td><td>0.96</td><td>0.80</td><td>1.23</td><td>Chubby</td><td>15.36</td><td>13.81</td><td>13.92</td></tr><tr><td>PointyNose</td><td>1.42</td><td>1.80</td><td>2.95</td><td>OvalFace</td><td>18.87</td><td>19.71</td><td>12.40</td></tr><tr><td>GrayHair</td><td>2.42</td><td>1.90</td><td>3.12</td><td>StraightHair</td><td>19.19</td><td>17.81</td><td>19.65</td></tr><tr><td>RecedingHairline</td><td>3.32</td><td>2.40</td><td>3.92</td><td>BrownHair</td><td>19.76</td><td>22.92</td><td>13.78</td></tr><tr><td>ArchedEyebrows</td><td>4.43</td><td>4.20</td><td>3.98</td><td>WearingLipstick</td><td>31.76</td><td>30.23</td><td>26.91</td></tr><tr><td>BigLips</td><td>4.66</td><td>5.90</td><td>7.26</td><td>Attractive</td><td>32.35</td><td>33.23</td><td>29.64</td></tr><tr><td>BlondHair</td><td>5.04</td><td>5.20</td><td>5.72</td><td>BagsUnderEyes</td><td>36.75</td><td>37.33</td><td>36.04</td></tr><tr><td>Goatee</td><td>5.11</td><td>5.00</td><td>4.52</td><td>BigNose</td><td>43.10</td><td>43.94</td><td>45.31</td></tr><tr><td>Sideburns</td><td>5.63</td><td>5.40</td><td>4.61</td><td>Male</td><td>47.51</td><td>50.05</td><td>51.40</td></tr><tr><td>Eyeglasses</td><td>6.10</td><td>5.70</td><td>6.51</td><td>HighCheekbones</td><td>55.55</td><td>52.85</td><td>46.01</td></tr><tr><td>WearingEarrings</td><td>6.80</td><td>6.20</td><td>4.85</td><td>Smiling</td><td>61.79</td><td>59.65</td><td>54.56</td></tr><tr><td>WearingHat</td><td>8.07</td><td>7.80</td><td>11.03</td><td>MouthSlightlyOpen</td><td>66.05</td><td>65.46</td><td>63.62</td></tr><tr><td>BlackHair</td><td>8.09</td><td>8.90</td><td>8.78</td><td>NoBeard</td><td>84.90</td><td>85.68</td><td>86.45</td></tr><tr><td>5o’ClockShadow</td><td>10.90</td><td>10.91</td><td>9.50</td><td>Young</td><td>85.21</td><td>85.18</td><td>83.36</td></tr></table>

Table 12: Percentage of LFWA attributes predicted by FaceXFormer. 1,000 reference images are generated from FFHQ-StyleGAN2 with a truncation value of $\psi = 1.0$ , and the optimized images generated by our method are derived from the initial latent vectors of these 1,000 references with $N = 10$ . Sorted in ascending order of FFHQ(%).

# D Additional Results for Section 4.1

# D.1 Quantitative Results

The full version of Table 3 is provided in Table 12. Percentages are calculated only for face-detected cases, with undetected cases at 0.017% for FFHQ, 0.100% for references, and 0.529% for optimized images.

Among the 19 rare attributes (<10% in FFHQ), the percentages of 12 attributes increase with our method compared to the references. However, the percentages of six attributes—ArchedEyebrows, BlackHair, Goatee, RosyCheeks, Sideburns, and WearingEarrings—decrease due to being overshadowed by other rare attributes. Specifically, ArchedEyebrows, Goatee, Sideburns, and WearingEarrings often disappear when WearingHat is present, while BlackHair is replaced by other hair colors or Bald. Additionally, FaceXFormer also frequently misses the RosyCheeks attribute.

Attributes with high percentages ( $>40\%$ ) in the dataset, such as BigNose and NoBeard, also increase with our method, which can be seemed weird. This result comes from the dependency on the likelihood estimated by the utilized NF model. To be specific, the rise in BigNose may be due to its higher percentage among low-likelihood samples—49.39% in the bottom 10% versus 43.18% in the entire sample set. NoBeard increases as Goatee and Sideburns decrease.

# D.2 Qualitative Results

We also provide additional qualitative results in Fig. 11, 12, and 17 (top). Rare attributes include extreme ages, non-frontal head poses, non-white races, hair colors other than

![](images/0be5759842cfba6a0e0541a9e7dc952001f47dcb80ae8d94bcce81561f45db73.jpg)

<details>
<summary>text_image</summary>

Initial
Diverse Rare Samples
</details>

Figure 11: Examples of diverse rare samples generated by our method using FFHQ-StyleGAN2.

brown, eyeglasses, hairless features, and hats as shown in Fig. 12. From the top of Fig. 17, our method successfully changes the high-likelihood references into rarer ones.

About Artifacts Some images in the results show low fidelity and contain undesirable artifacts. Although we use a real k-NN manifold and a penalizing boundary to prevent out-of-distribution samples, such artifacts are inevitable due to overestimated regions by the manifold assumption. Our objective function pushes samples toward low-density

![](images/ec832b215cb6c3b7b1ef9abdea084904b69ae59dbcc08b0ecb14aaa192483ec8.jpg)  
Figure 12: Additional rare samples generated by our method using FFHQ-StyleGAN2. In each row, the first and third columns serve as references for the second and fourth columns, respectively. The changed or generated attributes are listed below the figures. Rare attributes are highlighted in bold.

or even out-of-distribution regions. If these regions are included in the assumed real manifold, they may be selected as the best images by our algorithm. Improving the objective function or best sample selection process could help mitigate these issues and enhance the results.

Comparative Qualitative Results In Fig. 13, 100 random samples from different methods are visualized. The red boxes represent out-of-manifold samples from the real k-NN manifold. Although both samples from our method and Polarity sampling show rare attributes that rarely represented in the baseline, most samples in the results of Polar-

ity sampling include huge artifacts on the face, which make the samples be detected as out-of-manifold samples. We will discuss more about Polarity sampling in Section G.

# E Additional Results for Section 4.2

# E.1 Categorization of Dog Classes

There are 120 classes of dogs in the ImageNet dataset, and we construct eight high-level groups from those. We use ChatGPT-4 (OpenAI 2023) for classification and description for each category, and the result is shown in Table 18.

![](images/56cf657b74118c084ca818640c3fc8add4ca1b98fdc7eadec3f71428e59079c7.jpg)  
Figure 13: Comparative qualitative results: FFHQ-StyleGAN2 with a truncation value of $\psi = 1.0$ (top-left), our method (top-right), and Polarity sampling with a truncation value of $\rho = 1.0$ (bottom-left) and $\rho = 5.0$ (bottom-right). Red boxes indicate out-of-manifold samples.

<table><tr><td>LFWA Attribute</td><td>MetF.</td><td>Reference</td><td>Ours</td><td>LFWA Attribute</td><td>MetF.</td><td>Reference</td><td>Ours(%)</td></tr><tr><td>Bald</td><td>0</td><td>0</td><td>0.14</td><td>BigLips</td><td>3.01</td><td>2.42</td><td>2.75</td></tr><tr><td>RosyCheeks</td><td>0</td><td>0</td><td>0</td><td>RecedingHairline</td><td>3.31</td><td>2.62</td><td>2.50</td></tr><tr><td>Eyeglasses</td><td>0.07</td><td>0.10</td><td>0.33</td><td>ArchedEyebrows</td><td>3.61</td><td>4.44</td><td>5.32</td></tr><tr><td>WearingNecklace</td><td>0.15</td><td>0.20</td><td>0.04</td><td>BlackHair</td><td>4.51</td><td>4.54</td><td>4.34</td></tr><tr><td>Blurry</td><td>0.15</td><td>0.20</td><td>0.43</td><td>OvalFace</td><td>4.66</td><td>6.76</td><td>4.46</td></tr><tr><td>PointyNose</td><td>0.22</td><td>0.60</td><td>1.14</td><td>WearingLipstick</td><td>4.81</td><td>3.43</td><td>4.86</td></tr><tr><td>Mustache</td><td>0.37</td><td>0.10</td><td>0.16</td><td>Sideburns</td><td>5.64</td><td>5.35</td><td>6.36</td></tr><tr><td>HeavyMakeup</td><td>0.75</td><td>0.30</td><td>0.89</td><td>5o’ClockShadow</td><td>6.40</td><td>9.49</td><td>9.53</td></tr><tr><td>WearingNecktie</td><td>0.82</td><td>0.70</td><td>0.60</td><td>MouthSlightlyOpen</td><td>8.88</td><td>9.49</td><td>9.82</td></tr><tr><td>PaleSkin</td><td>0.82</td><td>0.90</td><td>2.08</td><td>Attractive</td><td>9.18</td><td>12.72</td><td>15.31</td></tr><tr><td>BlondHair</td><td>1.50</td><td>2.12</td><td>2.37</td><td>BushyEyebrows</td><td>9.56</td><td>10.40</td><td>7.44</td></tr><tr><td>HighCheekbones</td><td>1.65</td><td>1.51</td><td>0.79</td><td>Chubby</td><td>10.24</td><td>9.89</td><td>8.99</td></tr><tr><td>StraightHair</td><td>1.73</td><td>2.02</td><td>2.23</td><td>BrownHair</td><td>11.74</td><td>10.70</td><td>8.47</td></tr><tr><td>Smiling</td><td>1.80</td><td>1.01</td><td>1.06</td><td>WearingHat</td><td>12.34</td><td>9.89</td><td>12.18</td></tr><tr><td>GrayHair</td><td>1.80</td><td>2.52</td><td>3.94</td><td>WavyHair</td><td>25.75</td><td>28.48</td><td>21.93</td></tr><tr><td>Goatee</td><td>1.95</td><td>2.82</td><td>2.08</td><td>BagsUnderEyes</td><td>30.72</td><td>29.39</td><td>26.96</td></tr><tr><td>NarrowEyes</td><td>2.03</td><td>2.92</td><td>2.48</td><td>BigNose</td><td>40.51</td><td>38.48</td><td>37.45</td></tr><tr><td>WearingEarrings</td><td>2.40</td><td>1.61</td><td>2.08</td><td>Male</td><td>70.33</td><td>70.00</td><td>73.43</td></tr><tr><td>DoubleChin</td><td>2.48</td><td>2.62</td><td>2.37</td><td>Young</td><td>76.50</td><td>80.10</td><td>74.16</td></tr><tr><td>Bangs</td><td>2.48</td><td>1.61</td><td>2.65</td><td>NoBeard</td><td>85.09</td><td>86.46</td><td>84.22</td></tr></table>

Table 13: Percentage of LFWA attributes predicted by FaceXFormer. MetF. refers to the MetFaces dataset. 1,000 reference images are generated from MetFaces-StyleGAN2-ADA with a truncation value of $\psi = 1.0$ , and the optimized images generated by our method are derived from the initial latent vectors of these 1,000 references with N = 5. Sorted in ascending order of MetF.(%).

<table><tr><td colspan="2">ImageNet Class</td><td>Real</td><td>Reference</td><td>Ours(%)</td></tr><tr><td rowspan="5">Cat</td><td>Tabby</td><td>56.76</td><td>58.00</td><td>55.10</td></tr><tr><td>Siamese</td><td>15.33</td><td>16.80</td><td>15.87</td></tr><tr><td>Persian</td><td>10.55</td><td>8.70</td><td>11.14</td></tr><tr><td>Egyptian</td><td>9.35</td><td>11.60</td><td>9.24</td></tr><tr><td>Tiger Cat</td><td>4.71</td><td>2.80</td><td>5.26</td></tr></table>

Table 14: Percentage of cat-related breeds in ImageNet classes. 1,000 reference images are generated by AFHQ CatStyleGAN2-ADA with a truncation value of $\psi = 1.0$ , and the optimized images generated by our method are derived from the initial latent vectors of these 1,000 references with $N = 5$ . The results are sorted in descending order of Real%.

# E.2 Quantitative Results

We provide the full version of Table 5 for AFHQ Cat dataset and the generated cat face images, in Table 14. The percentage of Egyptian cats decreases compared to the reference samples despite not being major classes. This decrease occurs because this class in the reference samples are diversified to other classes during optimization.

The full version of Table 6 is in Table 13. The percentages are calculated for the only face-detected cases. The percentages of the undetected cases are 0.59%, 1.00%, and 4.15% for the MetFaces dataset, the references, and the optimized images, respectively. Compared to the FFHQ dataset, the MetFaces dataset has very low percentages of most of the LFWA attributes, where 26 attributes among the 40 attributes have a percentage lower than 5%. For the ten most rare attributes in the MetFaces dataset, Bald, Rosy-

Cheeks, Eyeglasses, WearingNecklace, Blurry, PointyNose, Mustache, HeavyMakeup, PaleSkin, WearingNecktie, seven attributes show an increased percentage in our method compared to the references. For the remaining three attributes, RosyCheeks has zero percentage in all cases, and the percentages of WearingNecklace and WearingNecktie rather decreased in our method, which are disappeared when getting diverse. Other than those top ten attributes, our method increased the percentage of blond and gray hair, eyeglasses, earrings, hats, etc. Additionally, we found the over-trust issue of Male in the FaceXFormer LFWA attributes classifier in the MetFaces dataset.

# E.3 Qualitative Results

We provide additional qualitative results in Fig. 14, 16, 18, and the bottom of the Fig. 17.

# F Experimental Setting and Additional Results for Section 4.4

We fit the UMAP model for the real feature vectors to reduce the dimensionality from 4096 to 2. To draw a local region of the feature space with probability density estimated by the NF model, we sample the grid points from the dimensionality-reduced two-dimensional plane by the UMAP and transform them into the feature space by the inverse mapping function, followed by computing the $\log p(\mathbf{x})$ by the NF model. We interpolate them by the thin plate spline kernel $r^{2} \times \log(r)$ , a spline-based smoothing kernel that interpolates polynomials piecewisely. Technically, we use a Python

![](images/a8f7919fb6f6f1a21eae27aa96c49623a9958570f85aa40eaee9ab833bbab8a5.jpg)  
Figure 14: Examples of diverse rare samples generated by our method using AFHQ and MetFaces with StyleGAN2-ADA.

API UMAP (McInnes, Healy, and Melville 2018) and scipy.interpolate.RBFInterpolator (Virtanen et al. 2020) for interpolation. We compute the k-NN balls with k = 3 on the transformed space to visualize them into the heatmap plausibly. With the Euclidean distance metric, we set the number of neighboring sample points to 15 to reduce the dimension to two as a hyperparameter setting for the UMAP. We set the smoothing parameter to zero and the degree of the kernel polynomial to first order. The other hyperparameter options follow the default settings. In the experiment, we fix the random state at 42.

We provide additional results in Fig. 19, demonstrating the relationship between rarity scores and NF-estimated likelihood. The optimization path is directed towards low density or larger real k-NN balls. However, our objective function allows the optimization path to continuously trail the real feature manifold, even the out-of-manifold area undefined by k-NN balls. In the cases in Fig. 19, rare attributes are obtained such as curly orange hair, a pink turban, and a non-frontal head pose.

# G Experimental Setting and Additional Results for Polarity Sampling

# G.1 Experimental Setting

For Polarity sampling in Section 4.1, we utilize the pre-calculated latent vectors and the Jacobian matrix of the StyleGAN2-config f generator provided by the authors in (Humayun, Balestriero, and Baraniuk 2022). Note that the latent seeds are different from our seeds. The corresponding GitHub repository is available at: https://github.com/AhmedImtiazPrio/magnet-polarity. Following the default settings, the singular value matrix is truncated to the top 30 values, and sampling is performed without replacement.

While we used the same number of generated samples from Polarity sampling in all statistical measures, in practice, this process involves generating a substantial number of initial samples and calculating their Jacobian matrices before resampling to achieve the desired sample numbers. Note that obtaining a large number of rare samples requires a pre-sampled set and pre-calculated Jacobians. Alternatively, in an online sampling setting, a very large number of samplings would be needed.

# G.2 Different $\rho$ 's

We provide additional experimental results with different $\rho$ 's in Table 15. From $\rho \geq 0.5$ , the Polarity sampling shows higher rarity scores compared to our method. However, all the listed results show significantly lower precision compared to the reference and our method.

# G.3 Sampling with Replacement

For practical purposes of maintaining diversity, the replacement parameter has been set to false during Polarity sampling. However, sampling without replacement can introduce bias into the results. Notably, in the earlier work by the

<table><tr><td>Polarity ρ</td><td>RS↑</td><td>Prec.↑</td><td>Rec.↑</td><td>LPIPS↑</td><td>FID↓</td></tr><tr><td>0.1</td><td>20.80</td><td>0.60</td><td>0.63</td><td>0.74</td><td>7.69</td></tr><tr><td>0.3</td><td>23.35</td><td>0.45</td><td>0.71</td><td>0.74</td><td>23.12</td></tr><tr><td>0.5</td><td>24.17</td><td>0.41</td><td>0.71</td><td>0.75</td><td>29.48</td></tr><tr><td>0.8</td><td>24.66</td><td>0.39</td><td>0.71</td><td>0.75</td><td>32.43</td></tr><tr><td>1.0</td><td>24.71</td><td>0.39</td><td>0.70</td><td>0.75</td><td>33.28</td></tr><tr><td>5.0</td><td>24.83</td><td>0.38</td><td>0.71</td><td>0.75</td><td>34.11</td></tr><tr><td>Baseline</td><td>18.88</td><td>0.69</td><td>0.56</td><td>0.73</td><td>4.17</td></tr><tr><td>Ours</td><td>23.50</td><td>0.92</td><td>0.65</td><td>0.76</td><td>7.38</td></tr></table>

Table 15: Quantitative evaluation of Polarity sampling for FFHQ and StyleGAN2 with varying $\rho$ . RS refers to the rarity score.

<table><tr><td>Polarity ρw/ Replacement</td><td>RS↑</td><td>Prec.↑</td><td>Rec.↑</td><td>LPIPS↑</td><td>FID↓</td></tr><tr><td>1.0</td><td>27.21</td><td>0.16</td><td>0.53</td><td>0.60</td><td>219.74</td></tr><tr><td>5.0</td><td>26.38</td><td>0.0004</td><td>0.21</td><td>0.002</td><td>299.73</td></tr></table>

Table 16: Quantitative evaluation of Polarity sampling with replacement for FFHQ and StyleGAN2. RS refers to the rarity score.

same authors, MaGNET sampling (Humayun, Balestriero, and Baraniuk 2021), which provides the theoretical foundation for Polarity sampling, the resampling procedure was defined with replacement. We conduct Polarity sampling with replacement, and the results are shown in Table 16 and Fig. 15. With replacement, certain out-of-manifold samples are resampled very frequently, reducing the diversity of results and limiting the opportunity to sample in-distribution rare samples.

# G.4 Online Rejection Sampling

From Table 1, Polarity sampling with a positive $\rho$ can obtain rare samples, but at the cost of a very high percentage of out-of-manifold samples. In this section, to investigate the true capability of Polarity sampling, we utilized online rejection sampling. This method collects the same number of samples sequentially while rejecting the out-of-manifold samples. As a result, all the collected samples are within the real k-NN manifold, indicating a precision of 1.

With $\rho = 1.0$ , 22,285 samples are generated to collect 10,000 valid samples. Similarly, with $\rho = 5.0$ , 22,376 samples are generated to collect 10,000 valid samples, requiring more than twice as many samples. We recalculate the rarity score, recall, LPIPS, and FID scores for these samples and presented the results in Table 17. We provide the qualitative results in Fig. 20.

Compared to the original Polarity sampling, replacing out-of-manifold samples with in-manifold samples results in a decrease in the k-NND of the samples which previously had out-of-manifold neighbors. This reduction in k-NND within the fake manifold leads to a decrease in recall. This also implies that the similarity between samples increases, which leads to a decrease in the LPIPS score. If the sampling is performed with replacement, this issue would be more significant, since a few rare samples with very high resampling weights would be selected frequently. Compared to our method, Polarity sampling with online rejection achieves similar average rarity and fidelity. However, our method generates a greater diversity of rare samples.

<table><tr><td>Polarity ρw/ Online Rejection</td><td>RS↑</td><td>Prec.↑</td><td>Rec.↑</td><td>LPIPS↑</td><td>FID↓</td></tr><tr><td>1.0</td><td>23.77</td><td>1.00</td><td>0.58</td><td>0.74</td><td>20.54</td></tr><tr><td>5.0</td><td>23.84</td><td>1.00</td><td>0.58</td><td>0.74</td><td>21.28</td></tr></table>

Table 17: Quantitative evaluation of Polarity sampling for FFHQ and StyleGAN2, using online rejection sampling to prevent out-of-manifold samples. RS denotes the rarity score.

![](images/28b806cfe21a0865918e292cf714d70e127270e6a2dd3bdba8b1b0dcbaaaa451.jpg)

<details>
<summary>text_image</summary>

ρ = 1.0
ρ = 5.0
Without
Replacement
With
</details>

Figure 15: Qualitative results of Polarity sampling. Top: Sampling without replacement. Bottom: Sampling with replacement. Left: $\rho = 1.0$ . Right: $\rho = 5.0$ . Each set contains 20 randomly generated samples.

![](images/b09051bf5d5594ee231d7b6de5b35de20adf32717dc4d26e5027b825207fd9ea.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 tabby cats in various colors and expressions, each face tilted to the other (no text or symbols visible)
</details>

Tabby Cat → Rare Cat breed   
![](images/2ac9317d3476c6f3ed9b228ab0ff3ae4314bdafa91f7174b36ba8686eb4b9f10.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 identical cat portraits arranged in a 3x3 grid, showing different expressions and angles (no text or symbols)
</details>

![](images/b4faa3cab653032db53e597bc925cf7b2ef5dd9b50bf9b3ac12131fcba805c0f.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 identical cat portraits in various colors and sizes, arranged in a 3x3 grid (no text or symbols)
</details>

![](images/cfa03223c6334238fa50130206b5d2b5c4d12f569d9ec7cc50eb4200334b4f25.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 identical cat portraits showing various expressions and expressions (no text or symbols)
</details>

Unique Facial Expression (e.g. sleepy)

![](images/72020c2da4b0c447ab06307947ff04d84cb17689a4f074880483d40ce17a28b9.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 identical cat photos in various colors and sizes, each showing a different look or profile view (no text or symbols visible)
</details>

Grass in front of a Cat   
![](images/6f6a2233b89f3d6dcf1a6b73331593d77e57aa5a1e591dd1421a5d006989db7b.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 cats and their hatching on grass, no text or symbols visible
</details>

![](images/616c1b4357d33bd08984a57233c1835833f55ad13d1622059d2cf12d6d626432.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 diverse dog portraits in various colors and styles, no text or symbols visible
</details>

![](images/59c1dc82d37023aad2e3e44728ed0d40e2cdd3c1b79111c3d08023a7ca824970.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 dog portraits in various colors and expressions, no text or symbols present
</details>

Herding/Sporting → Rare Dog Group

![](images/2dc2ac5077e29df4450cb69af8fe8d291fea573c5aa16714f54fe333310ecdb4.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 dog portraits in various colors and expressions, no text or symbols present
</details>

Colorful Background

![](images/8c2328670255b07c176c024933b649c9ccd3d9258ab3689a1f448408ead18f8b.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 dog portraits in various colors and expressions, no text or symbols visible
</details>

![](images/d2945f4820f9aa9e051ce5e05ca22633fa2eb513d778bca8e98a69c8b5e2f986.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 dog portraits in various colors and expressions, arranged in a 3x3 grid (no text or symbols)
</details>

![](images/f48464488c0c4508eaeda53003afcfa681bae7f8e5fa4e18c74776a44eb0232c.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 dog portraits in various colors and poses, no text or symbols visible
</details>

Dog Leash

![](images/716f14eaaa240c86689834451d44c5a3e5e3ac849a873bfb9dee859eb4db67b1.jpg)

<details>
<summary>natural_image</summary>

Collection of historical portraits and statues in various styles (no text or symbols visible)
</details>

White → Non-White

![](images/20838f3b7aa62e72d273b82d762f0ab756430e82fc3412eb051555968a6b4fa8.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 historical male portraits with varied facial features and hairstyles, no text or symbols present
</details>

![](images/b1287abd1cca27ab51ad3eaa2aad0e95d138e72e347e46a9fb427e6bbc6ab35b.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 historical portraits including Roman numerals and classical figures, displayed in a 3x4 grid (no text or symbols)
</details>

![](images/2a00cf89e4ac0ddc417d8cee9cbced4366b542371a4b7fac91e2937768feb687.jpg)

<details>
<summary>natural_image</summary>

Grid of portrait portraits of historical figures in 17th-century attire, no text or symbols visible
</details>

Eyeglasses, Hat   
Figure 16: Additional rare samples generated by our method using AFHQ and MetFaces with StyleGAN2-ADA. In each row, the first and third columns serve as references for the second and fourth columns, respectively. The changed or generated attributes are listed below the figures. Rare attributes are highlighted in bold.

![](images/d2cbba6b68ed8cc60a1df639af5871dc6dd0675b6caa1d44dec822b2f6559108.jpg)

<details>
<summary>natural_image</summary>

Grid of diverse headshot portraits with no visible text or symbols
</details>

![](images/91b89fcbd8799fa79d9810c884bfde82e0774aaadb95d0f01128b945d16ea3c6.jpg)

<details>
<summary>natural_image</summary>

Grid of diverse headshot portraits in various colors and expressions, no text or symbols visible
</details>

![](images/9131d5ffe463a7e36222793f75cecdb12c83c04621e3290a67a75f72876a210c.jpg)

<details>
<summary>natural_image</summary>

Grid of portrait paintings from 18th to 19th century, showing various facial features and ages (no text or symbols visible)
</details>

![](images/cf7d292ca38699532ba10c46bf1de62cb26ccc88ad418c7142b75da80b229c9d.jpg)

<details>
<summary>natural_image</summary>

Grid of portrait photos of various historical figures in 19th-century attire, no visible text or symbols
</details>

Figure 17: High-likelihood references (top 100) and their optimized rare images generated by our method. Top: FFHQ-StyleGAN2. Bottom: MetFaces-StyleGAN2-ADA. Left: References with a truncation value of $\psi = 1.0$ . Right: Optimized images.

![](images/c1888e0cd75f89ee81bbf5424baa2e45973ee84341ab37889b9cf0b068d429ce.jpg)

<details>
<summary>natural_image</summary>

Grid of 300 identical cat portraits in various angles and sizes, arranged in a 2x2 grid with no text or symbols.
</details>

![](images/e6e3334b44d92219d9ae83f909770c4ab8c437ddc98e6938b8e52273a6fe5781.jpg)

<details>
<summary>natural_image</summary>

Grid of 200 tabby cats in various colors and sizes, each with a different facial expression (no text or symbols visible)
</details>

![](images/4deaf7d7fbb114edf485dcbc8aa22346f5faff9030b57250820f656717436109.jpg)

<details>
<summary>natural_image</summary>

Grid of 300 dog portraits in various colors and expressions, arranged in a 4x4 layout (no text or symbols)
</details>

![](images/9b565853e38ed6ec035d103f2d764a39e98fac5599d8b39bf4cab3010a1fed4b.jpg)

<details>
<summary>natural_image</summary>

Grid of 90 diverse dog portraits in various colors and styles, arranged in a 3x4 grid with no visible text or symbols.
</details>

Figure 18: High-likelihood references (top 100) and their optimized rare images generated by our method. Top: AFHQ CatStyleGAN2-ADA. Bottom: AFHQ Dog-StyleGAN2-ADA. Left: References with a truncation value of $\psi = 1.0$ . Right: Optimized images.

![](images/977657dab374c0277d082d9d87b1b4340a7cc3f412ebc5728953b1fdba3eec26.jpg)

<details>
<summary>line</summary>

| Optimization Step | Rarity Score | Value |
| ----------------- | ------------ | ----- |
| 0                 | 14           | 3000  |
| 10                | 14           | 3500  |
| 20                | 18           | 4500  |
| 30                | 25           | 6000  |
| 40                | 0            | -2000 |
| 50                | 0            | -2000 |
| 60                | 0            | -2000 |
</details>

![](images/e71247ab0bab2857edfcf13de14c7b150ee5c3b8949f684584b4a35ccdd94cd8.jpg)

<details>
<summary>line</summary>

| Optimization Step | log p(x) | log p(x) |
| ----------------- | -------- | -------- |
| 0                 | 22       | 7000     |
| 10                | 13       | 6000     |
| 20                | 12       | 5000     |
| 30                | 0        | 4000     |
| 40                | 0        | 4000     |
| 50                | 0        | 4000     |
| 60                | 0        | 4000     |
</details>

![](images/04175ef8e83d707c509d85d7a916e129339304d42715a9a2666fa337d7a996ca.jpg)

<details>
<summary>scatter</summary>

| UMAP axis 1 | UMAP axis 2 | Density Level |
|-------------|-------------|---------------|
| 0.5         | 0.5         | High          |
| 0.7         | 0.3         | Low           |
| 0.9         | 0.8         | High          |
</details>

![](images/8bc53c9d1871daaf561ec75aa38b105c4025e0c5d6d2e7e17c19a8ea9b07817c.jpg)

<details>
<summary>scatter</summary>

| Point Type              | UMAP axis 1 | UMAP axis 2 |
| ----------------------- | ----------- | ----------- |
| Initial point           | ~0.5        | ~0.8        |
| Optimized point         | ~0.3        | ~0.4        |
| Optim pt (RS=N/A)      | ~0.2        | ~0.3        |
| Real point              | ~0.1        | ~0.2        |
| k-NN Real Manifold      | ~0.6        | ~0.7        |
</details>

Figure 19: Examples of the optimization paths with a real k-NN manifold and a heatmap of likelihoods estimated by the normalizing flow. Top: The black line represents the optimization path, with each marker indicating a rarity score at every ten steps. The blue line represents the log-likelihood estimated by the NF model. Bottom: The balls indicate the nearby real k-NN manifold.

<table><tr><td>Dog Group</td><td>ImageNet Class Number</td><td>Description</td></tr><tr><td>Toy</td><td>151, 152, 153, 154, 155, 157, 158, 171, 185, 186, 187, 200, 201, 252, 254, 259, 262, 265</td><td>Small dogs, often with short, round faces.</td></tr><tr><td>Hound</td><td>159, 160, 169, 170, 172, 173, 176, 177, 253</td><td>Dogs with long faces, often having long snouts and large ears.</td></tr><tr><td>Scent Hound</td><td>161, 162, 163, 164, 165, 166, 167, 168, 174, 175</td><td>Broad-faced dogs with typically droopy ears.</td></tr><tr><td>Terrier</td><td>179, 180, 181, 182, 183, 184, 188, 189, 190, 191, 192, 193, 194, 196, 199, 202, 203</td><td>Dogs with short, sturdy faces and pronounced, strong snouts.</td></tr><tr><td>Sporting</td><td>156, 178, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221</td><td>Medium-sized dogs, usually with balanced features.</td></tr><tr><td>Non-Sporting</td><td>195, 204, 223, 245, 251, 260, 261, 266, 267, 268</td><td>A diverse group of dogs with a variety of face shapes and body types.</td></tr><tr><td>Herding</td><td>197, 198, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 235, 240, 241, 263, 264</td><td>Dog breeds for herding and managing livestock.</td></tr><tr><td>Working</td><td>222, 234, 236, 237, 238, 239, 242, 243, 244, 246, 247, 248, 249, 250, 255, 256, 257, 258</td><td>Large, powerful dog breeds for tasks like guarding, pulling, and rescue work.</td></tr></table>

Table 18: Dog groups categorized from 120 dog classes in ImageNet dataset.

![](images/1bea2f9ef8ca5786340777cd8494db415ef5e09804f7fa1bbd0475a63cda5086.jpg)

<details>
<summary>natural_image</summary>

Grid of diverse face and child portraits in various styles and colors, no text or symbols visible
</details>

![](images/9c9853c73c17a6a17a906b689a1b1789bf67453930d9014be8cf2ebd9d680477.jpg)

<details>
<summary>natural_image</summary>

Grid of diverse colorful headshot portraits with no visible text or symbols
</details>

Figure 20: Qualitative results of Polarity sampling with $\rho = 1.0$ (left) and $\rho = 5.0$ (right), using online rejection sampling to prevent out-of-manifold samples. All samples are within the real k-NN manifold constructed from the FFHQ dataset.