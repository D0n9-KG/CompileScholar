Arpit Bansal $^{*1}$ Hong-Min Chu $^{*1}$ Avi Schwarzschild $^{1}$ Soumyadip Sengupta $^{2}$ Micah Goldblum $^{3}$ Jonas Geiping $^{1}$ Tom Goldstein $^{1}$

# Abstract

Typical diffusion models are trained to accept a particular form of conditioning, most commonly text, and cannot be conditioned on other modalities without retraining. In this work, we propose a universal guidance algorithm that enables diffusion models to be controlled by arbitrary guidance modalities without the need to retrain any use-specific components. We show that our algorithm successfully generates quality images with guidance functions including segmentation, face recognition, object detection, and classifier signals. Code is available at github.com/arpitbansal297/Universal-Guided-Diffusion.

# 1. Introduction

Diffusion models are powerful tools for creating digital art and graphics. Much of their success stems from our ability to carefully control their outputs, customizing results for each user's individual needs. Most models today are controlled through conditioning. With conditioning, the diffusion model is built from the ground up to accept a particular modality of input from the user, be it descriptive text, segmentation maps, class labels, etc. While conditioning is a powerful tool, it results in models that are handcuffed to a single conditioning modality. If another modality is required, a new model needs to be trained, often from scratch. Unfortunately, the high cost of training makes this prohibitive for most users.

A more flexible approach to controlling model outputs is to use guidance. In this approach, the diffusion model acts as a generic image generator, and is not required to understand a user's instructions. The user pairs this model with Walker hound,
Walker foxhound
in space

![](images/ea6ce9dc94f1611406eeadd87584aee2be9209c382d466857957b2e954e106e9.jpg)

<details>
<summary>natural_image</summary>

Abstract purple shape on black background, no text or symbols present
</details>

Target Segmentation Map

![](images/b864809a498e4185ee64bead637776454d4176d9a72fd5b799b595f45ca276ed.jpg)

<details>
<summary>natural_image</summary>

A beagle dog floating in space with a starry blue background (no text or symbols)
</details>

![](images/5416bcb41df3f2fd39534fb9b621a557f21bdfd388cbb92a213fd9c9b625898c.jpg)  
Target Object Location

A headshot of a woman with a dog in winter.

![](images/566ff2e86f0613d51e4374f06a0de42069dcd6ae736292365bad98d7c73792d7.jpg)

<details>
<summary>natural_image</summary>

Two dogs in winter clothing, one wearing a red beanie and the other in a black coat, with no visible text or symbols.
</details>

![](images/0918473c4b63a148261eff52f77e6d17b4c6094736b234632e768ad5117de184.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a woman with blonde hair and makeup (no visible text or symbols)
</details>

Target Identity

A headshot of a blonde woman as a sketch

![](images/001d2c0bee1aba2bfe48e404361b9c063144455d647ec00a2ca2f1c3ce8dff27.jpg)

<details>
<summary>natural_image</summary>

Black-and-white pencil sketch of a woman's face with blonde hair and makeup (no text or symbols)
</details>

![](images/fb7bfc4bcff6dd205da9756dc60de2eef231ed149aa3b0f8efa51601a2587417.jpg)

<details>
<summary>natural_image</summary>

Painting of a person walking along a path through autumn trees with vibrant red and green foliage (no text or symbols)
</details>

Target Style Image

A Portrait of a woman

![](images/61ba0c51baf78742ce77d3fc3d4f8eafe13aff26f0e918475c7c33e1f3383d95.jpg)

<details>
<summary>natural_image</summary>

Abstract painting of a person wearing a hat, rendered in vibrant color with no visible text or symbols
</details>

Figure 1: Diffusion guided by off-the-shelf networks.

a guidance function that measures whether some criterion has been met. For example, one could guide the model to minimize the CLIP score between the generated image and a text description of the user's choice. During each iteration of image creation, the iterates are nudged down the gradient of the guidance function, causing the final generated image to satisfy the user's criterion.

In this paper, we study guidance methods that enable any off-the-shelf model or loss function to be used as guidance for diffusion. Because guidance functions can be used without re-training or modification, this form of guidance is universal in that it enables a diffusion model to be adapted for nearly any purpose.

From a user perspective, guidance is superior to conditioning, as a single diffusion network is treated like a foundational model that provides universal coverage across many use cases, both commonplace and bespoke. Unfortunately, it is widely believed that this approach is infeasible. While early diffusion models relied on classifier guidance (Dhariwal & Nichol, 2021), the community quickly turned to classifier-free schemes (Ho & Salimans, 2022) that require a model to be trained from scratch on class labels with a particular frozen ontology that cannot be changed (Nichol et al., 2021; Rombach et al., 2022; Bansal et al., 2022).

The difficulty of using guidance stems from the domain shift between the noisy images used by the diffusion sampling process and the clean images on which the guidance models are trained. When this gap is closed, guidance can be performed successfully. For example, Nichol et al. (2021) successfully use a CLIP model as guidance, but only after re-training CLIP from scratch using noisy inputs. Noisy retraining closes the domain gap, but at a very high financial and engineering cost. To avoid the additional cost, we study methods for closing this gap by changing the sampling scheme, rather than the model.

To this end, our contributions are summarized as follows:

- We propose an algorithm that enables universal guidance for diffusion models. Our proposed sampler evaluates the guidance models only on denoised images, rather than noisy latent states. By doing so, we close the domain gap that has plagued standard guidance methods. This strategy provides the end-user with the flexibility to work with a wide range of guidance modalities and even multiple modalities simultaneously. The underlying diffusion model remains fixed and no fine-tuning of any kind is necessary.   
- We demonstrate the effectiveness of our approach for a variety of different constraints such as classifier labels, human identities, segmentation maps, annotations from object detectors, and constraints arising from inverse linear problems.

# 2. Background

We first briefly review the recent literature on the core framework behind diffusion models. Then, we define the problem setting of controlled image generation and discuss previous related works.

# 2.1. Diffusion Models

Diffusion models are strong generative models that proved powerful even when first introduced for image generation (Song & Ermon, 2019; Ho et al., 2020). The approach has been successfully extended to a number of domains, such as audio and text generation (Kong et al., 2020; Huang et al., 2022; Austin et al., 2021; Li et al., 2022).

We introduce (unconditional) diffusion formally, as it is helpful in describing the nuances of different types of models. A diffusion model is defined as a combination of a T-step forward process and a T-step reverse process. Conceptually, the forward process gradually adds Gaussian noise of different magnitudes to a clean data point $z_{0}$ , while the reverse process attempts to gradually denoise a noisy input in hopes of recovering a clean data point. More concretely, given an array of scalars representing noise scales $\{\alpha_{t}\}_{t=1}^{T}$ and an initial, clean data point $z_{0}$ , applying t steps of the forward process to $z_{0}$ yields a noisy data point

$$
z _ {t} = \sqrt {\alpha_ {t}} z _ {0} + (\sqrt {1 - \alpha_ {t}}) \epsilon , \epsilon \sim \mathcal {N} (0, \mathbf {I}). \tag {1}
$$

A diffusion model is a learned denoising network $\epsilon_{\theta}$ . It is trained so that for any pair $(z_{0}, t)$ and any sample of $\epsilon$ ,

$$
\epsilon_ {\theta} (z _ {t}, t) \approx \epsilon = \frac {z _ {t} - \sqrt {\alpha_ {t}} z _ {0}}{\sqrt {1 - \alpha_ {t}}}. \tag {2}
$$

The reverse process takes the form $q(z_{t-1}|z_{t}, z_{0})$ with various detail definitions, where $q(\cdot|\cdot)$ is generally parameterized as a Gaussian distribution. Different works also studied different approximations of the unknown $q(z_{t-1}|z_{t}, z_{0})$ used to perform sampling. For example, denoising diffusion implicit model (DDIM) (Song et al., 2021a) first computed a predicted clean data point

$$
\hat {z} _ {0} = \frac {z _ {t} - (\sqrt {1 - \alpha_ {t}}) \epsilon_ {\theta} (z _ {t} , t)}{\sqrt {\alpha_ {t}}}, \tag {3}
$$

and sample $z_{t-1}$ from $q(z_{t-1}|z_{t},\hat{z}_{0})$ by replacing unknown $z_{0}$ with $\hat{z}_{0}$ . On the other hand, while the details of individual sampling methods vary, all sampling methods produce $z_{t-1}$ based on current sample $z_{t}$ , current time step t and a predicted noise $\hat{\epsilon}$ . To ease the notation burden, we define a function $S(\cdot,\cdot,\cdot)$ as an abstraction of the sampling method, where $z_{t-1}=S(z_{t},\hat{\epsilon},t)$ .

# 2.2. Controlled Image Generation

In this paper, we focus on controlled image generation with various constraints. Consider a differentiable guidance function f, for example a CLIP feature extractor or a segmentation network. When applied to an image, we obtain a vector $c = f(x)$ . We also consider a function $\ell(\cdot, \cdot)$ that measures the closeness of two vectors c and $c'$ . Given a particular choice of c, which we call a prompt, the corresponding constraint (based on $c, \ell$ , and f) is formalized as $\ell(c, f(z)) \approx 0$ , and we aim to generate a sample z from the image distribution satisfying the constraint. In plain words, we want to generate an in-distribution image that matches the prompt.

Prior work that studied controlled generative diffusion mainly falls into two categories. We refer to the first category as conditional image generation, and the second category as guided image generation. Next, we discuss the characteristics of each category and better situate our work among existing methods.

Conditional Image Generation. Methods from this category require training new diffusion models that accept the prompt as an additional input (Ho & Salimans, 2022; Bansal et al., 2022; Nichol et al., 2021; Whang et al., 2022; Wang et al., 2022a). For example, Ho & Salimans (2022) proposed classifier-free guidance using class labels as prompts, and trained a diffusion model by linear interpolation between unconditional and conditional outputs of the denoising networks. Bansal et al. (2022) studied the case where the guidance function is a known linear degradation operator, and trained a conditional model to solve linear inverse problems. Nichol et al. (2021) further extended classifier-free guidance to text-conditional image generation with descriptive phrases as prompts, and trained a diffusion model to enforce the similarity between the CLIP (Radford et al., 2021) representations of the generated images and the text prompts. These methods are successful across different types of constraints, however the requirement to retrain the diffusion model makes them computationally intensive.

Guided Image Generation. Works in this category employed a frozen pre-trained diffusion model as a foundation model, but modify the sampling method to guide the image generation with feedback from the guidance function. Our method falls into this category. Prior work that studied guided image generation did so with a variety of restrictions and external guidance functions (Dhariwal & Nichol, 2021; Kawar et al., 2022; Wang et al., 2022b; Chung et al., 2022a; Lugmayr et al., 2022; Chung et al., 2022b; Graikos et al., 2022). For example, Dhariwal & Nichol (2021) proposed classifier guidance, where they trained a classifier on images of different noise scales as the guidance function f, and included gradients of the classifier during the sampling process. However, a classifier for noisy images is domain-specific and generally not readily available – an issue our method circumvents. Wang et al. (2022b) assumed the external guidance functions to be linear operators, and generated the component of images residing in the null space of linear operators with the foundation model. Unfortunately, extending that method to handle non-linear guidance functions is non-trivial. Chung et al. (2022a) studied general guidance functions, and modified the sampling process with the gradient of guidance function calculated on the expected denoised images. Nevertheless, the authors only presented results with simpler non-linear guidance functions such as non-linear blurring.

In this work, we study universal guidance algorithms for guided image generation with diffusion models using any off-the-shelf guidance functions $f$ , such as object detection or segmentation networks.

# 3. Universal Guidance

We propose a guidance algorithm that augments the image sampling method of a diffusion model to include guidance from an off-the-shelf auxiliary network. Our algorithm is motivated by an empirical observation that the reconstructed clean image $\hat{z}_{0}$ obtained by Equation (3), while naturally imperfect, is still appropriate for a generic guidance function to provide informative feedback to guide the image generation. In Section 3.1, we motivate our forward universal guidance by extending classifier guidance (Dhariwal & Nichol, 2021) to leverage this observation and handle generic guidance functions. In Section 3.2, we propose a supplementary backward universal guidance to help enforce the generated image to satisfy the constraint based on the guidance function f. In Section 3.3, we discuss a simple yet helpful self-recurrence trick to empirically improve the fidelity of generated images.

# 3.1. Forward Universal Guidance

To guide the generation with information from the external guidance function f and the loss function $\ell$ , an immediate thought is to extend classifier guidance (Dhariwal & Nichol, 2021) to accept any general guidance function. Concretely, given a class prompt c, classifier guidance performs classification-guided sampling by replacing $\epsilon_{\theta}(z_{t}, t)$ in each sampling step $S(z_{t}, t)$ with

$$
\hat {\epsilon} _ {\theta} (z _ {t}, t) = \epsilon_ {\theta} (z _ {t}, t) - \sqrt {1 - \alpha_ {t}} \nabla_ {z _ {t}} \log p (c | z _ {t}). \tag {4}
$$

Defining $\ell_{ce}(\cdot,\cdot)$ to be the cross-entropy loss and $f_{cl}$ to be the guidance function that outputs classification probability, Equation (4) can be re-written as

$$
\hat {\epsilon} _ {\theta} (z _ {t}, t) = \epsilon_ {\theta} (z _ {t}, t) + \sqrt {1 - \alpha_ {t}} \nabla_ {z _ {t}} \ell_ {c e} (c, f _ {c l} (z _ {t})). \quad (5)
$$

Algorithm 1 Universal Guidance   
Parameter: Recurrent steps $k$ , gradient steps $m$ for backward guidance and guidance strength $s(t)$ ,
Required: $z_T$ sampled from $\mathcal{N}(0, I)$ , diffusion model $\epsilon_\theta$ , noise scales $\{\alpha_t\}_{t=1}^T$ , guidance function $f$ , loss function $\ell$ , and prompt $c$ for $t = T, T - 1, \ldots, 1$ do
    for $n = 1, 2, \ldots, k$ do
    Calculate $\hat{z}_0$ as Equation (3)
    Calculate $\hat{\epsilon}_\theta$ using forward universal guidance as Equation (6)
    if $m > 0$ then
    Calculate $\Delta z_0$ by minimizing Equation (7) with $m$ steps of gradient descent
    Perform backward universal guidance by $\hat{\epsilon}_\theta \leftarrow \hat{\epsilon}_\theta - \sqrt{\alpha_t / (1 - \alpha_t)} \Delta z_0$ (Equation (9))
    end if $z_{t-1} \leftarrow S(z_t, \hat{\epsilon}_\theta, t)$ $\epsilon' \sim \mathcal{N}(0, I)$ $z_t \leftarrow \sqrt{\alpha_t / \alpha_{t-1}} z_{t-1} + \sqrt{1 - \alpha_t / \alpha_{t-1}} \epsilon'$ end for
end for

However, directly replacing $f_{cl}$ and $\ell_{ce}$ with any off-the-shelf guidance and loss functions does not work in practice, as f is most likely trained on clean images and fails to provide meaningful guidance when the input is noisy.

To address the issue, we leverage the fact that $\epsilon_{\theta}(z_{t}, t)$ predicts the noise added to the data point, and we can therefore obtain a predicted clean image $\hat{z}_{0}$ by Equation (3). We propose to instead calculate the guidance based on the predicted clean data point as

$$
\hat {\epsilon} _ {\theta} (z _ {t}, t) = \epsilon_ {\theta} (z _ {t}, t) + s (t) \cdot \nabla_ {z _ {t}} \ell (c, f (\hat {z} _ {0})) \tag {6}
$$

where $s(t)$ controls the guidance strength for each sampling step and

$$
\nabla_ {z _ {t}} \ell (c, f (\hat {z} _ {0})) = \nabla_ {z _ {t}} \ell \left(c, f \left(\frac {z _ {t} - \sqrt {1 - \alpha_ {t}} \epsilon_ {\theta} (z _ {t} , t)}{\sqrt {\alpha_ {t}}}\right)\right)
$$

as in Equation (3). We term Equation (6) forward universal guidance, or forward guidance in short. In practice, applying forward guidance effectively brings the generated image closer to the prompt while keeping the generation trajectory in the data manifold. We note that a related approach is also studied in (Chung et al., 2022a), where the guidance step is computed based on $E[z_0|z_t]$ . The approach drew inspiration from the score-based generative framework (Song et al., 2021b), but resulted in a different update method.

# 3.2. Backward Universal Guidance

As will be shown in Section 4.2, we observe that forward guidance sometimes over-prioritizes maintaining the “realness" of the image, resulting in an unsatisfactory match with the given prompt. Simply increasing the guidance strength $s(t)$ is suboptimal, as this often results in instability as the image moves off the manifold faster than the denoiser can correct it.

To address the issue, we propose backward universal guidance, or backward guidance in short, to supplement forward guidance and help enforce the generated image to satisfy the constraint. The key idea of backward guidance is to optimize for a clean image that best matches the prompt based on $\hat{z}_{0}$ , and linearly translate the guided change back to the noisy image space at step t. Concretely, instead of directly calculating $\nabla_{z_{t}}\ell(c,f(\hat{z}_{0}))$ , we compute a guided change $\Delta z_{0}$ in clean data space as

$$
\Delta z _ {0} = \arg \min _ {\Delta} \ell (c, f (\hat {z} _ {0} + \Delta)). \tag {7}
$$

Empirically, we solve Equation (7) with m-step gradient descent, where we use $\Delta = 0$ as a starting point. Since $\hat{z}_{0} + \Delta z_{0}$ minimizes $\ell(c, f(z))$ directly, $\Delta z_{0}$ is the change in clean data space that best enforces the constraint. Then, we translate $\Delta z_{0}$ back to the noisy data space of $z_{t}$ by calculating the guided denoising prediction $\tilde{\epsilon}$ that satisfies

$$
z _ {t} = \sqrt {\alpha_ {t}} (\hat {z} _ {0} + \Delta z _ {0}) + \sqrt {1 - \alpha_ {t}} \tilde {\epsilon}. \tag {8}
$$

Reusing Equation (3), we can rewrite $\tilde{\epsilon}$ as an augmentation to the original denoising prediction $\epsilon_{\theta}(z_t, t)$ by

$$
\tilde {\epsilon} = \epsilon_ {\theta} (z _ {t}, t) - \sqrt {\alpha_ {t} / (1 - \alpha_ {t})} \Delta z _ {0}. \tag {9}
$$

Comparing to forward guidance, backward guidance (as Equation (9)) produces an optimized direction for the generated image to match the given prompt, and hence prioritizes enforcing the constraint. Furthermore, calculation of a gradient step for Equation (7) is computationally cheaper than forward guidance (Equation (6)), and we can therefore afford to solve Equation (7) with multiple gradient steps, further improving the match with the given prompt.

We note that the names “forward” and “backward” are used analogously to the forward and backward Euler methods.

# 3.3. Per-step Self-recurrence

Unfortunately, when we apply our universal guidance to standard generation pipelines, we often find images with artifacts and strange behaviors that clearly separate them from natural images. Similar observations have been made in (Lugmayr et al., 2022; Wang et al., 2022b), where linear guidance functions are studied. Our attempts to prioritize realness by decreasing $s(t)$ proved ineffective; the sweet spot that both ensures the realness and guidance constraint satisfaction doesn't always exist, especially for complex

![](images/d6d494b99805e3efccc0f78c8b95508a673c8f8e28c910612411a229df579543.jpg)

<details>
<summary>natural_image</summary>

Four dog photos: a black-and-white dog, a red-flored dog silhouette, a white dog lying down, and a brown-brown dog resting on grass (no text or symbols)
</details>

Figure 2: An example of how self-recurrence helps segmentation-guided generation. The left-most figure is the given segmentation map, and the images generated with recurrence steps of 1, 4 and 10 follow in order.

guidance functions. We conjecture that the guidance direction produced by our universal method is not always related to the realness of the images when the guidance function creates too much information loss, causing the image to stray from the natural image sampling trajectory.

Inspired by (Lugmayr et al., 2022; Wang et al., 2022b), we address the issue by applying per-step self-recurrence. More concretely, after $z_{t-1} = S(z_t, \hat{e}_t, t)$ is sampled, we re-inject random Gaussian noise $\epsilon' \sim \mathcal{N}(0, \mathbf{I})$ to $z_{t-1}$ to obtain $z_t'$ by

$$
z _ {t} ^ {\prime} = \sqrt {\alpha_ {t} / \alpha_ {t - 1}} \cdot z _ {t - 1} + \sqrt {1 - \alpha_ {t} / \alpha_ {t - 1}} \cdot \epsilon^ {\prime}. \tag {10}
$$

Equation (10) ensures $z_{t}^{\prime}$ to have proper noise scale for input at time step t. We repeat the self-recurrence k times before continuing the sampling for step t - 1. Intuitively, the self-recurrence allows exploration of different regions of the data manifold at the same noise scale, allowing more budget to find a solution that satisfies both guidance and image quality. Empirically, we find that our self-recurrence can keep the realness of the generated image with a proper guidance strength $s(t)$ that ensures the match with the given prompt. We illustrate an example of how self-recurrence improves the harmony of generated images in Figure 2.

We summarize our universal guidance algorithm composed of forward universal guidance, backward universal guidance and per-step self-recurrence in Algorithm 1. For simplicity, the algorithm assumes only one guidance function, but can be easily adapted to handle multiple pair of $(f, l)$ . Additionally, the objectives of the forward and backward guidance do not have to be identical, allowing different ways to simultaneously utilize multiple guidance functions.

# 4. Experiments

In this section, we present results testing our proposed universal guidance algorithm against a wide variety of guidance functions. Specifically, we experiment with Stable Diffusion (Rombach et al., 2022), a diffusion model that is able to perform text-conditional generation by accepting text prompt as additional input, and experiment with a purely unconditional diffusion model trained on ImageNet (Deng et al., 2009), where we use pre-trained model provided by Conditional Stable-Diffusion

Guided Stable-Diffusion

A photograph of an astronaut riding a horse.

![](images/5d309a991a0335f1c77e7eb0c63b48bbce5cef4953533d8b4ff314dfb43c4090.jpg)

<details>
<summary>natural_image</summary>

Astronaut riding a horse on a barren, rocky terrain under clear sky (no text or symbols visible)
</details>

![](images/9709283819e1ee526d4c379bdf99c85cfcc808ea9ffebeb486bd4f7b18eaedab.jpg)

<details>
<summary>natural_image</summary>

Person in orange safety gear riding a horse in a forest setting (no visible text or symbols)
</details>

An oil painting of a corgi wearing a party hat.

![](images/91c233f7cd3cb81d61bafb32fc8db1cd75b153c44439ed0454fb22bffd0458cb.jpg)

<details>
<summary>natural_image</summary>

Illustration of a corgi wearing a colorful party hat and smiling (no text or symbols)
</details>

![](images/f9b772d4c694f88b68955ae37804c15b7c97e8bcfb7d0aec14fb9dec542133d5.jpg)

<details>
<summary>natural_image</summary>

Painting of a stylized animal face with colorful paint and a striped headpiece (no text or symbols)
</details>

Figure 3: We compare the ability to match given text prompts between our universal guidance algorithm and text-conditional model trained from scratch. The results demonstrate that our universal algorithm is comparable to specialized conditional model on the ability to generate quality images that satisfy the text constraints.

OpenAI (Dhariwal & Nichol, 2021). We note that Stable Diffusion, while being a text-conditional generative model, can also perform unconditional image generation by simply using an empty string for the text prompt. We first present the experiment on Stable Diffusion for different guidance functions in Section 4.1, and present the results on ImageNet diffusion model in Section 4.2.

# 4.1. Results for Stable Diffusion

In this section, we present the results of guided image generation using Stable Diffusion as the foundation model. The guidance functions we experiment with include the CLIP feature extractcor (Radford et al., 2021), a segmentation network, a face recognition network and an object detection network. For experiments on Stable Diffusion, we discover that applying forward guidance already produce high-quality images that match the given prompt, and hence set m = 0. To perform forward guidance on Stable Diffusion, we forward the predicted clean latent variable computed by Equation (3) through the image decoder of Stable Diffusion to obtain predicted clean images. We discuss the results and implementation details for each guidance function in its corresponding subsection.

![](images/094b63033e37b13e3e01ed08b1f54b396811c59b6bd01c89321e19c73785b009.jpg)

<details>
<summary>text_image</summary>

Prompt
Guide
(Walker hound,
Walker foxhound
under water.
(Walker hound,
Walker foxhound
on snow.
(Walker hound,
Walker foxhound
as an oil painting.
(N/A)
</details>

Figure 4: In addition to matching the text prompts (above each column), these images are guided by an image segmentation pipeline. Each column contains examples of images generated to match the prompt and the segmentation map in the left-most column. The top-most row contains examples generated without guidance.

CLIP Guidance. CLIP (Radford et al., 2021) is a state-of-the-art text-to-image similarity model developed by OpenAI. To apply our algorithm to text-guided image generation, we use the image feature extractor of CLIP as the guidance function. We construct a loss function that calculates the negative cosine similarity between an image embedding and the CLIP text embedding produced by a given text prompt. We use $s(t) = 10\sqrt{1 - \alpha_{t}}$ and k = 8 and use Stable Diffusion as an unconditional image generator.

We generate images guided by a number of text prompts. To further assess our universal guidance algorithm and compare guidance and conditioning, we also generate images using classical, text-conditional generation by Stable Diffusion with identical prompts as inputs, and summarize the results in Figure 3. The results in Figure 3 show that our algorithm can guide the generation to produce high-quality images that match the given text description, and are comparable with images generated by the specialized text-conditioning model.

Segmentation Map Guidance. To perform guided image generation using a segmentation map as prompt, we use a MobileNetV3-Large (Howard et al., 2019) with a segmentation head, and a publicly available pre-trained model in PyTorch (Paszke et al., 2019). As the segmentation network outputs per-pixel classification probability, we construct a loss function $\ell$ as the sum of per-pixel cross-entropy loss between a given prompt and the predicted segmentation of generated images. We set $s(t) = 400 \cdot \sqrt{1 - \alpha_t}$ and $k = 10$ .

![](images/84217a6bf4af6fa7d031f29708082812c16ea7a17a94e7d64c9b4a3946d01dbc.jpg)

<details>
<summary>text_image</summary>

Prompt
Guide
Headshot of a
person with
blonde hair
with space
background.
Headshot of a
woman made
of marble.
A headshot of
a woman looking
like Lara Croft.
(N/A)
</details>

Figure 5: In addition to matching the text prompts (above each column), these images are guided by a facial recognition system. Each column contains examples of images generated to match the prompt and the identity of the images in the left-most column. The top-most row contains examples generated without guidance.

In our experiment, we combine segmentation maps that depict objects of different shapes with new text prompts. We use the text prompt as a fixed additional input to Stable Diffusion to perform text-conditional sampling, and guide the text-conditional generated images to match the given segmentation maps. Results are presented in Figure 4. From Figure 4, we see that the generated images show a clear separation between object and background that matches the given segmentation map nearly perfectly. The generated object and background also each match their descriptive text (i.e. dog breed and environment description). Furthermore, the generated images are overall highly realistic.

Face Recognition Guidance. To guide image generation to resemble the face of a given person, we compose a guidance function that combines a face detection module and a face recognition module. This setup pro-

![](images/01b459e063d9697275c32dee5c39a7a20466b9e2b3a86af8914601636ac88bab.jpg)

<details>
<summary>text_image</summary>

Prompt
Guide
(N/A)
Headshot of a
woman with a
dog.
Headshot of a
woman with a
dog on beach.
An oil painting of a
headshot of a
women with a dog.
</details>

Figure 6: In addition to matching the text prompts (above each column), these images are guided by an object detector. Each column contains examples of images generated to match the prompt and the bounding boxes used for guidance. The top row contains examples generated without guidance.

duces a facial attribute embedding from an input face image. We use multi-task cascaded convolutional networks (MTCNN) (Zhang et al., 2016) as the face detection module, and use facenet (Schroff et al., 2015) as the face recognition module. The guidance function f hence crops out the detected face and outputs a facial attribute embedding as prompt, while we use $l_{1}$ -loss between embedding as the loss function $\ell$ . We note that to compute the guidance direction in our algorithm, we only backpropagate through the facenet and treat the face cropping mask produced by MTCNN as an oracle input, as MTCNN utilizes non-maximum suppression (Neubeck & Van Gool, 2006) which is non-differentiable. Here we set $s(t) = 20000 \cdot \sqrt{1 - \alpha_{t}}$ and k = 2.

We explore different combinations of face guidance and text prompts. Similarly to the segmentation case, we use the text prompt as a fixed additional conditioning to Stable Diffusion and guide this text-conditional trajectory with our algorithm so that the face in the generated image looks similar to the face prompt. In Figure 5, we clearly see that the facial characteristics of a given face prompt are reproduced almost perfectly on the generated images. The descriptive text of either background, material, or style is also realized correctly and blends nicely with the generated faces.

![](images/eefac4d6201536a44f40c3fc18d6ca00df304a1314007ab74f90f0e07fcf41e4.jpg)

<details>
<summary>text_image</summary>

Prompt
Style
(N/A)
A colorful
photo of an
Eiffel Tower
A fantasy photo
of volcanoes
A portrait of
a woman
</details>

Figure 7: In addition to matching the text prompts (above each column), these images are guided by a style image. Each column contains examples of images generated to match the text prompt and the style image used for guidance. The top-most row contains examples generated without style guidance.

Object Location Guidance For Stable Diffusion, we also present the results guiding image generation with an object detection network. For this experiment, we use Faster-RCNN (Ren et al., 2015) with Resnet-50-FPN backbone (Li et al., 2021), a publicly available pre-trained model in Pytorch, as our object detector. We use bounding boxes with class labels as our object location prompt. We construct a loss function $\ell$ by the sum of three individual losses, namely (1) anchor classification loss, (2) bounding box regression loss and (3) region label classification loss, where (1) and (2) are computed on the region proposal head while (3) is computed on the region classification head. We note that, compared to standard R-CNN training, we drop the additional bounding box alignment loss on region classification head. We found that our loss construction helps to produce objects of correct categories for each location prompt. We set $s(t) = 100 \cdot \sqrt{1 - \alpha_t}$ and $k = 3$ .

We again experiment with different combinations of text prompt and object location prompt, and similarly use the text prompt as a fixed conditioning to Stable Diffusion. Using our proposed guidance algorithm, we perform guided image generation that generates and matches the objects presented in the text prompt to the given object locations.

![](images/6a4abf5477ff321f6fa45430f49df690554e27c5f15dfcf8b2bdd405db55eb8e.jpg)

<details>
<summary>text_image</summary>

English foxhound by
Edward Hopper
Van Gogh Style
Cake
</details>

Figure 8: We show that unconditional diffusion models trained on ImageNet can be guided with CLIP to generate high-quality images that match the text prompts, even if these generated images should be out of distribution.

The results are presented in Figure 6. We observe from Figure 6 that objects in the descriptive text all appear in the designated location with the appropriate size indicated by the given bounding boxes. Each location is filled with appropriate, high-quality generations that align with varied image content prompts, ranging from “beach” to “oil painting”.

Style Guidance Finally, we conclude our experiments on Stable Diffusion by guiding the image generation based on a reference style given by a style image. To achieve so, we capture the reference style from the style image by the image feature extractor from CLIP, and use the resulting image embedding as prompts. The loss function calculates the negative cosine similarity between the embedding of generated images and the embedding of the style image. Similar to previous experiments, we control the content using text input as additional conditioning to the Stable Diffusion model.

We experiment with combinations of different style images and different text prompts, and present the results in Figure 7. From Figure 7, we can see that the generated images contain contents that match the given text prompts, while exhibiting style that matches the given style images. In this experiment we set $s(t) = 6 \cdot \sqrt{1 - \alpha_t}$ and $k = 6$ . Furthermore, in order to control the amount of content we set the scale $\gamma$ , a parameter of Stable Diffusion that balances the text-conditional generation and unconditional generation, as 3.0, 3.0, and 4.0 respectively for each column.

# 4.2. Results for ImageNet Diffusion

In this section, we present results for guided image generation using an unconditional diffusion model trained on ImageNet. We experiment with CLIP guidance, object location guidance and a hybrid guided image generation task which we term segmentation-guided inpainting. We will discuss results and implementations of each guidance in its corresponding subsection.

![](images/60d3f3c667aec8bccca47e2007ae2f8009844205014b7bcea2303d12d4420577.jpg)

<details>
<summary>text_image</summary>

Object Location
Forward Only
Forward + Backward
bird
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
chair
Chair
</details>

Figure 9: Generation guided by object detection with the unconditional ImageNet model. Images generated with both forward and backward guidance are realistic and have the desired objects in the designated locations. In contrast, images generated using only forward guidance exhibit objects of the incorrect category or with inaccurate position/size.

CLIP Guidance. We use the same construction of f and $\ell$ for Stable Diffusion to perform CLIP-guided generation. We use only forward guidance for this experiment. To assess the limit of our universal guidance algorithm, we hand-crafted text prompts such that the matching images are expected to be out of distribution. In particular, our text prompts either designate art styles that are far from realistic or designate objects that do not belong to any possible class label of ImageNet. We present the results in Figure 8, and from the results, we clearly see that our algorithm still successfully guides the generation to produce quality images that also match the text prompts. For all three images, we have $s(t) = w \cdot \sqrt{1 - \alpha_{t}}$ , where w is 2, 5 and 2 respectively and k is 10, 5 and 10 respectively.

Object Location Guidance. Similar to object location guidance for Stable Diffusion, we also use the same network architecture and the same pre-trained model as our object detection network, and construct an identical loss function $\ell$ for our guidance algorithm. However, unlike Stable Diffusion, object locations are the only prompts available for guided image generation. For this experiment, we use $s(t) = 100\sqrt{1 - \alpha_{t}}$ and k = 3.

We again experiment with different object location prompts using two configurations of our algorithm, namely (1) using only forward universal guidance and (2) using both forward and backward universal guidance. We observe from Figure 8 that applying both forward and backward guidance generates images that are realistic and the objects matches the prompt nicely. On the other hand, while images generated using only forward guidance remain realistic, they feature objects with mismatching categories and locations.

![](images/43321845fe69ff68608daf4bf5be9511eef828c5bbf80046d1705d5a2a66fd7e.jpg)

<details>
<summary>text_image</summary>

Masked Image
Clf. Guided
Clf. + Seg. Guided
</details>

Figure 10: Our guidance algorithm can incorporate feedback from multiple guidance functions. The first column shows the prompt for inpainting. The second column shows classifier-guided inpainting, where dog images with close matches to inpainting prompt are generated. The third column shows images generated with both classifier and segmentation guidance, where realistic dogs are generated exactly on the masked regions. The results show that our algorithm handles multiple guidance functions effectively.

The results demonstrate the effectiveness of our universal guidance algorithm, and also validate the necessity of our backward guidance.

Segmentation-Guided Inpainting. In this experiment, we aim to explore the ability of our algorithm to handle multiple guidance functions. We perform guided image generation with combined guidance from an inpainting mask, a classifier and a segmentation network. We first generate images with masked regions as the prompt for inpainting. We then pick an object class c as the prompt for classification and generate a segmentation mask where the masked regions are considered foreground objects of the same class c. We use $\ell_{2}$ loss on the non-masked region as the loss function for inpainting, and set the corresponding $s(t)=0$ , or equivalently only use backward guidance for inpainting. We use the same segmentation network as described in Section 4.1 with $s(t)=200\sqrt{1-\alpha_{t}}$ . For classification guidance, we use the classifier that accepts noisy input (Dhariwal & Nichol, 2021), and perform the original classifier guidance Equation (4) instead of our forward guidance. The results summarized in Figure 10 show that when using both inpainting and classifier as guidance, our algorithm generates realistic images that both match the inpainting prompt and can be classified correctly to the given object class. Adding in segmentation guidance, our algorithm further improves the generated images with a near-perfect match to both the segmentation map and inpainting prompt while maintaining realism. This demonstrates that our algorithm can effectively combine the feedback from individual guidance functions.

# 5. Limitations

Generation using universal guidance is typically slower than standard conditional generation for several reasons. Empirically, multiple iterations of denoising are required at every noise level t to generate high-quality images with complex guidance functions. However, the time complexity of our algorithm scales linearly with the number of recurrence steps k, which slows down image generation when k is large. Also, as demonstrated in the main paper, backward guidance is required in certain scenarios to help generate images that match the given constraint. Computing backward guidance requires performing minimization with a multi-step gradient descent inner loop. While proper choices of gradient-based optimization algorithms and learning rate schedules significantly speed up the convergence of minimization, the time it takes to compute backward guidance inevitably becomes longer when the guidance function is itself a very-large neural network. Finally, we note that, to get optimal results, sampling hyper-parameters must be chosen individually for each guidance network.

# 6. Conclusion

In this paper, we propose a universal guidance algorithm that is able to perform guided image generation with any off-the-shelf guidance function based on a fixed foundation diffusion model. Our algorithm only requires guidance and loss functions to be differentiable, and avoids any retraining to adapt either the guidance function or the foundation model to a specific type of prompt. We demonstrate promising results with our algorithm on complex guidance including segmentation, face recognition and object detection systems. Even multiple guidance functions can be combined and used in conjunction.

# 7. Acknowledgements

This work was made possible by the National Science Foundation (IIS-2212182), the AFOSR MURI Program, the Office of Naval Research (N000142112557), the ONR MURI program, IARPA WRIVA, and Capital One Bank.

# References

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den Berg, R. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021.   
Bansal, A., Borgnia, E., Chu, H.-M., Li, J. S., Kazemi, H., Huang, F., Goldblum, M., Geiping, J., and Goldstein,

T. Cold diffusion: Inverting arbitrary image transforms without noise. arXiv preprint arXiv:2208.09392, 2022.   
Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., and Ye, J. C. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022a.   
Chung, H., Sim, B., Ryu, D., and Ye, J. C. Improving diffusion models for inverse problems using manifold constraints. arXiv preprint arXiv:2206.00941, 2022b.   
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.   
Dhariwal, P. and Nichol, A. Q. Diffusion models beat gans on image synthesis. volume 34, 2021.   
Graikos, A., Malkin, N., Jojic, N., and Samaras, D. Diffusion models as plug-and-play priors. arXiv preprint arXiv:2206.09012, 2022.   
Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.   
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 32, 2020.   
Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., et al. Searching for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1314–1324, 2019.   
Huang, R., Lam, M. W., Wang, J., Su, D., Yu, D., Ren, Y., and Zhao, Z. Fastdiff: A fast conditional diffusion model for high-quality speech synthesis. arXiv preprint arXiv:2204.09934, 2022.   
Kawar, B., Elad, M., Ermon, S., and Song, J. Denoising diffusion restoration models. arXiv preprint arXiv:2201.11793, 2022.   
Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020.   
Li, X. L., Thickstun, J., Gulrajani, I., Liang, P., and Hashimoto, T. B. Diffusion-lm improves controllable text generation. arXiv preprint arXiv:2205.14217, 2022.   
Li, Y., Xie, S., Chen, X., Dollar, P., He, K., and Girshick, R. Benchmarking detection transfer learning with vision transformers. arXiv preprint arXiv:2111.11429, 2021.

Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., and Van Gool, L. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11461–11471, 2022.   
Neubeck, A. and Van Gool, L. Efficient non-maximum suppression. In 18th International Conference on Pattern Recognition (ICPR'06), volume 3, pp. 850–855. IEEE, 2006.   
Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., and Chen, M. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.   
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pp. 8748–8763. PMLR, 2021.   
Ren, S., He, K., Girshick, R., and Sun, J. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.   
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of CVPR, 2022.   
Schroff, F., Kalenichenko, D., and Philbin, J. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 815–823, 2015.   
Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. International Conference on Learning Representations, 2021a.   
Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019.   
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations, 2021b.   
Wang, W., Bao, J., Zhou, W., Chen, D., Chen, D., Yuan, L., and Li, H. Semantic image synthesis via diffusion models. arXiv preprint arXiv:2207.00050, 2022a.

Wang, Y., Yu, J., and Zhang, J. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022b.   
Whang, J., Delbracio, M., Talebi, H., Saharia, C., Dimakis, A. G., and Milanfar, P. Deblurring via stochastic refinement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16293–16303, 2022.   
Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. Joint face detection and alignment using multitask cascaded convolutional networks. IEEE signal processing letters, 23(10):1499–1503, 2016.

# A. More results

![](images/f8508762d378da467c1127ba42d8bf5fda4b6d6f51e711ac489cc82de7eec531.jpg)  
Figure 11: More images to show Segmentation guidance. In each subfigure, the first image is the segmentation map used to guide the image generation with its caption as its text prompt.

![](images/72aa71f6a03b081d41a62f1409051b52ea2323fb306d1a794c790c7837c29d7b.jpg)  
Figure 12: More images to show Face guidance. In each subfigure, the first image is the human identity used to guide the image generation with its caption as its text prompt.

![](images/53f8ecb1c58878a147bbd280d588d94e432221d842732e4ed1abfa7348df542d.jpg)

<details>
<summary>text_image</summary>

person
dog
</details>

(a) A headshot of a woman with a dog in winter.   
![](images/9a40e137da6b3230b89a06e87c58899b688ea2773685a52c27f262d15de8ab00.jpg)

<details>
<summary>text_image</summary>

person
dog
dog
dog
dog
dog
</details>

(b) a headshot of a woman with a dog on beach.   
![](images/61aff37f03d1ffb55a0d99ddd666903e18e16220c58bf658429c8daa946dfb2d.jpg)

<details>
<summary>text_image</summary>

person
dog
2017/2018
2017/2019
2017/2020
2017/2021
2017/2022
person
</details>

(c) An oil painting of a headshot of a woman with a dog.   
Figure 13: More images to show Object Location guidance. In each subfigure, the first image is the object location used to guide the image generation with its caption as its text prompt.

![](images/65bcde33f0b0acc0405f7676cb8c5f983d37bedf6a44f5e714546b2981fc4142.jpg)  
Figure 14: More images to show Style Transfer. In each subfigure, the first image is the styling image used to guide the image generation with its caption as its text prompt.