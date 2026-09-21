# A Self-Supervised Descriptor for Image Copy Detection

Ed Pizzi

Sreya Dutta Roy

Sugosh Nagavara Ravindra

Priya Goyal

Matthijs Douze

Meta AI

# Abstract

Image copy detection is an important task for content moderation. We introduce SSCD, a model that builds on a recent self-supervised contrastive training objective. We adapt this method to the copy detection task by changing the architecture and training objective, including a pooling operator from the instance matching literature, and adapting contrastive learning to augmentations that combine images.

Our approach relies on an entropy regularization term, promoting consistent separation between descriptor vectors, and we demonstrate that this significantly improves copy detection accuracy. Our method produces a compact descriptor vector, suitable for real-world web scale applications. Statistical information from a background image distribution can be incorporated into the descriptor.

On the recent DISC2021 benchmark, SSCD is shown to outperform both baseline copy detection models and self-supervised architectures designed for image classification by huge margins, in all settings. For example, SSCD outperforms SimCLR descriptors by $48\%$ absolute.

Code is available at https://github.com/facebookresearch/sscd-copy-detection.

# 1. Introduction

All online photo sharing platforms use content moderation to block or limit the propagation of images that are considered harmful: terrorist propaganda, misinformation, harassment, pornography, etc. Some content moderation can be performed automatically, for unambiguous data like pornographic pictures, but this is much harder for complex data like memes [31] or misinformation [2]. In these cases, content is moderated manually. For of viral images, where copies of same image may be uploaded thousands of times, manual moderation of each copy is tedious and unnecessary. Instead, each image for which a manual moderation decision is taken can be recorded in a database, so that it can be re-identified later and handled automatically.

This paper is concerned with this basic task of re-identification. This is non trivial because copied images are

![](dt=2026-06-04/ht=19/7c0a33db83730191df9837807447db26d902170a79fee1918bf0493a233d6804.jpg)

often altered, for technical reasons (e.g. a user shares a mobile phone screenshot that captures additional content), or users may make adversarial edits to evade moderation.

Image re-identification is an image matching problem, with two additional challenges. The first is the enormous scale at which copy detection systems are deployed. At this scale, the only feasible approach is to represent images as short descriptor vectors, that can be searched efficiently with approximate nearest neighbor search methods [23, 30]. Copy detection systems typically proceed in 2 stages: a retrieval stage that produces a shortlist of candidate matches and a verification stage, often based on local descriptor matching that operates on the candidates. In this work, we are concerned with the first stage. Figure 1 shows the overall architecture of our Self Supervised Copy Detection (SSCD) approach.

The second challenge is that there is a hard match/non-match decision to take, and positive image pairs are rare. We wish to limit verification candidates using a threshold, which is a harder constraint than the typical image retrieval setting, where only the order of results matter.

SSCD uses differential entropy regularization [42] to

arXiv:2202.10261v2 [cs.CV] 25 Mar 2022

promote a uniform embedding distribution, which has three effects: (1) it makes distances from different embedding regions more comparable; (2) it avoids the embedding collapse described in [29], making full use of the embedding space; (3) it also improves ranking metrics that do not require consistent thresholds across queries.

Score normalization is important for ranking systems. An advanced score normalization relies on matching the query images with a set of background images. In this work, we show how this normalization can be incorporated in the image descriptor itself. We anticipate that this work will set a strong single-model baseline for image copy detection. We plan to release code and models for our method.

Section 2 discusses works related to this paper. Section 3 motivates the use of an entropy loss term in a simplified setting. Section 4 carefully describes SSD. Section 5 presents results and ablations of our method. Section 6 points out a few observations about the copy detection task.

# 2. Related work

Content tracing approaches. Content tracing on a user-generated photo sharing platform aims at re-identifying images when they circulate out and back into the platform. There are three broad families of tracing methods: metadata-based [1, 3], watermarking [13, 32, 51, 63] and content-based. This work belongs to this last class.

Classical image datasets for content tracing, like Cassia [16,36] focus on image alterations like splicing, removal and copy-move transformations [16,44,54] that alter only a small fraction of the image surface, so the re-identification is done reliably with simple interest-point based techniques. The challenge is to detect the tampered surface, which is typically approached with deep models inspired by image segmentation [34, 62].

A related line of research is image phylogeny: the objective is to identify the series of edits that were applied to an image between an initial and a final state [14, 15, 33]. The Nimble/Media forensics series of competitions organized by NIST aim at benchmarking these tasks [40, 57]. In this work we focus on the identification itself, with strong transformations and near duplicates that need to be distinguished (see Figure 2).

Semantic and perceptual image comparison Several definitions of near-duplicate image matching, form a continuum between pixel-wise copy and instance matching [18, 28]. The definition we use in this work is: images are considered copies iff they come from the same 2D image source. More relaxed definitions allow, for example, to match nearby frames in a video.

There is a large body of literature about solving instance matching [7, 11, 26, 35, 37, 46-48] i.e., recognizing images of the same 3D object with viewpoint/camera changes. In this

![](dt=2026-06-04/ht=19/79f3d893604d5d4ac8566aa95ad3138785f6706f76cc6d25727992d229e8102b.jpg)

work, we build on this literature because it addresses complex image matching, and to our knowledge, recent works and benchmarks for strict copy detection are rare [17, 53].

Instance matching. Classical instance matching relies on 3D matching tools, like interest points [26, 35, 43]. CNN-based approaches use backbones from image classification, either pre-trained [4, 20, 49] or trained end-to-end [21, 38], with two adaptations: (1) the pooling layer that converts the last CNN activation map to a vector is a max-pooling [49], or more generally GeM pooling [39], a form of $L_{p}$ normalization where $p$ is adapted to the image resolution [7]; (2) careful normalization of the vectors.

In addition to simple L2-normalization [4], "whitening" is often used to compare descriptors [25, 49]. An additional normalization technique contrasts the distances w.r.t. a background distribution of images [18, 27]. In this work, we apply these pooling and normalization techniques to copy detection.

Contrastive self-supervised learning. A recent line of self-supervised learning research uses contrastive objectives that learn image representations that bring transformed im

ages together. These methods either discriminate image features [10, 22, 24] or the cluster assignments of these image features [8]. These methods either rely on memory banks [24, 56] or large batch sizes [10]. In particular, SimCLR
[10] uses matching transformed image copies as a surrogate task to learn a general image representation that transfer well to other tasks, such as image classification. A contrastive InfoNCE loss [52] is used to map copies of the same source image nearby in the embedding space.

Differential entropy regularization. Increasing the entropy of media descriptors forces them to spread over the representation space. Sablayrolles et al. [42] observed that the entropy can be estimated locally with the Kozachenko-Leononenko differential entropy estimator [6], that can be incorporated directly into the loss to maximize descriptor entropy. The work of El-Nouby et al. [19] is closest to our approach. It adds the entropy term to a contrastive loss at fine-tuning time to improve the accuracy for category and instance retrieval. Our approach is similar, applied to a self-supervised objective and image copy detection.

# 3. Motivation

In this section, we start from the SimCLR [10] method, then perform a simple experiment where we combine it with the entropy loss from [42] and witness how it impacts classification and copy detection tasks.

# 3.1. Preliminaries: SimCLR

SimCLR training is best described at the mini-batch level. For batches of $N$ images, it creates two augmented copies of each image (repeated augmentations), yielding $2N$ transformed images. The positive pairs of matching images are $P = \{(i,i + N),(i + N,i)\}_{i = 1..N}$ . We denote positive matches for image $i$ as $P_{i} = \{j\mid (i,j)\in P\}$ . Each image is transformed by a CNN backbone network. The final activation map of the CNN is average pooled, then projected using a two-layer MLP into a L2-normalized descriptor $z_{i}\in \mathbb{R}^{d}$ .

Descriptors are compared with a cosine similarity: $\mathrm{sim}(z_i,z_j) = z_i^\top z_j$ . A contrastive InfoNCE loss maximizes the similarity between copies relative to the similarity of non-copies. For inference (e.g. to transfer to image classification), SimCLR discards the training-time MLP, using globally pooled features from the CNN trunk directly.

The InfoNCE loss. SimCLR's InfoNCE loss is a softmax cross-entropy with temperature, that matches descriptors to other descriptors. Let $s_{i,j}$ be the temperature-adjusted cosine similarity $s_{i,j} = \sin (z_i,z_j) / \tau$ . The InfoNCE loss is defined as a mean of $\ell_{i,j}$ terms for positive pairs $(i,j)\in P$ :

$$
\ell_ {i, j} = - \log \frac {\exp \left(s _ {i , j}\right)}{\sum_ {k \neq i} \exp \left(s _ {i , k}\right)} \tag {1}
$$

$$
\mathcal {L} _ {\text {I n f o N C E}} = \frac {1}{| P |} \sum_ {i, j \in P} \ell_ {i, j}. \tag {2}
$$

# 3.2. Entropy regularization

We use the differential entropy loss proposed in [42], based on the Kozachenko-Leononenko estimator. We adapt it to the repeated augmentation setting by only regularizing neighbors from different source images:

$$
\mathcal {L} _ {\mathrm {K o L e o}} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \log \left(\min  _ {j \notin \hat {P} _ {i}} \| z _ {i} - z _ {j} \|\right), \tag {3}
$$

where $\hat{P}_i = P_i \cup \{i\}$ . Since this entropy loss is a log of the distance to the nearest neighbor, its impact is very high for nearby vectors but dampens quickly when the descriptors are far apart. The effect is to "push" apart nearby vectors.

![](dt=2026-06-04/ht=19/9aeb0d8f202bbcb2fbac18f57a7fdb9f84e049ca744079083ad7a04a43952eaa.jpg)

# 3.3. Experiment: SimCLR and entropy

For this experiment, we combine our contrastive loss with the entropy loss, using a weighting factor $\lambda$ , similar to [19, 42]:

$$
\mathcal {L} _ {\text {b a s i c}} = \mathcal {L} _ {\text {I n f o N C E}} + \lambda \mathcal {L} _ {\text {K o L e o}}. \tag {4}
$$

We then evaluate the impact of the combined loss on an image classification setting and a copy detection setting, see Section 5.1 for more details about the setup.

Figure 3 shows how varying entropy loss weight $\lambda$ impacts both tasks. As the entropy loss weight increases, ImageNet linear classification accuracy decreases: this loss

![](dt=2026-06-04/ht=19/4683368b554d37cc08bc52af761f609a944683d910ba865aea71fa18b9b38485.jpg)

![](dt=2026-06-04/ht=19/2c0efcca8df41375e9047ac7e49054accfb8f1fba258c06f396ff0e40b25dc3f.jpg)

term is not helpful for classification. Conversely, for copy detection the accuracy increases significantly.

Figure 4 shows the distribution of distances between matching images (positive pairs) and the nearest nonmatching neighbors (negative pairs). Applying the entropy loss increases all distances and makes the negative distance distribution more narrow. The result is that there is a larger contrast between positive pairs and the mode of the negative distribution, i.e. they are more clearly separated.

# 4. Method

Having seen how the entropy loss improves copy detection accuracy, in this section we expand it into a robust image copy detection approach: SSCD. This entails adapting the architecture, the data augmentation, the pooling and adding a normalization stage, as shown in Figure 1.

# 4.1. Architecture

SSCD uses a ResNet-50 convolutional trunk to extract image features. We standardize on this architecture because it is widely used, well optimized and still very competitive for image classification [55], but any CNN or transformer backbone could be used (see Section 5).

Pooling. For classification, the last CNN activation map is converted to a vector by mean pooling. We use generalized mean (GeV) pooling instead, which was shown [7, 39] to improve the discriminative ability of descriptors. This is desirable for instance retrieval and our copy detection case alike. GeM introduces a parameter $p$ , equivalent to average pooling when $p = 1$ and max-pooling when $p \to \infty$ . SSCD uses $p = 3$ , following common practice for image retrieval models [7, 39, 49].

Table 1. List of data augmentations used for SSCD. The presentations is incremental: each set of augmentations includes the ones from all rows before. The percentages are probabilities to apply each augmentation.

![](dt=2026-06-04/ht=19/5f8ef68cf3a695b35aaf66d5e0a31fdb7802e40c666bb49f3111fc6426368142.jpg)

<table><tr><td>type</td><td>details</td></tr><tr><td>SimCLR</td><td>horizontal flip, random crop, color jitter, grayscale, Gaussian blur</td></tr><tr><td>Strong blur</td><td>50% large-radius Gaussian blur (σ ∈ [1, 5])</td></tr><tr><td>Advanced</td><td>10% rotation, 10% text, 20% emoji, 20% JPEG compression</td></tr><tr><td>Adv. + mixup</td><td>2.5% mixup, 2.5% cutmix</td></tr></table>

While GeM pooling at inference time systematically improves accuracy, we observe that it is beneficial at training time only in combination with the differential entropy regularization, i.e. with a vanilla InfoNCE it is better to train with average pooling. We conjecture that GeM pooling may reduce the difficulty of the training task without the additional objective of maximally separating embedding points. We observe that learning the scalar $p$ , as proposed in [39], fails for contrastive learning: the pooling parameter grows unbounded until training becomes numerically unstable.

Descriptor projection. SimCLR uses a 2-layer MLP projection at training time. For inference, the MLP is discarded and CNN trunk features are used directly. The MLP is partly motivated to retain transformation-covariant features in the base network, which may be useful for downstream tasks, despite a training task that requires a transformation-invariant descriptor. Jing et al. [29] also find that the MLP insulates the trunk model from an embedding collapse into a lower-dimensional space caused by the InfoNCE loss.

For SSCD, th
e training and inference tasks are the same, obviating the need for transformation-covariant features, and differential entropy regularization prevents the dimensional collapse. We replace the MLP with a simple linear projection to the target descriptor size, and retain this projection for inference.

# 4.2. Data Augmentation

Self-supervised contrastive objectives learn to match images across image transforms. These methods are sensitive to the augmentations seen at training time [10], since invariance to these transforms is the only supervisory signal.

Table 1 lists the SSCD augmentations used in our experiments. Note that since our main evaluation dataset (DISC2021) is built in part with data augmentation, there is a risk of overfitting to the augmentations of that dataset. This is mitigated by (1) DISC2021's set of augmentations is not known precisely and (2) we present strong results trained using a simple blur augmentation. Our starting baseline is the default set of SimCLR augmentations.

Strong blur. Empirically, copy detection benefits from a stronger blur than is typically used for contrastive learning. We strengthen the blur augmentation compared to SimCLR. We suggest that invariance to blur confers a low-frequency bias, reducing the model's sensitivity to high-frequency noise common to real world copies. We use this setting for most ablation steps, because it is easy to reproduce, and provides a good baseline setting for comparing methods. This augmentation was initially tuned on a proprietary dataset, and is unlikely to overfit to DISC2021.

Advanced augmentations. We evaluate our method with additional augmentations, to demonstrate how SSCD extends as augmentations are added. Half of rotations rotate by multiples of 90 degrees and half are unconstrained. The text has a random font, text, opacity, font size, and color. We add emoji of random size. We apply JPEG compression with randomly sampled compression quality. These augmentations are somewhat inspired by DISC2021 but are still fairly generic for image copy detection problems.

Mixed images. We use two augmentations that combine content from two images within a training batch. In a copy detection context, these augmentations model partial copies, where part of an image is included in a composite image. Mixup [61] is a pixelwise weighted average of two images $(a$ and $b)$ with parameter $\gamma \in [0,1]$ : $\gamma \cdot a + (1 - \gamma)\cdot b$ . CutMix [59] moves rectangular regions from one image into another. See Appendix D for implementation details. Mixed images match multiple images in the batch, requiring changes to our losses, outlined below.

# 4.3. Loss Functions

SSCD uses a weighted combination of the contrastive InfoNCE and the entropy loss, as in Equation (4). However, we need to adapt both losses for the mixed-image augmentation case, where $P_{i}$ may contain multiple matching images.

InfoNCE with MixUp/CutMix augmentations. We adapt the InfoNCE loss (see Section 3.1) to accommodate augmentations that mix features from multiple images. Given an image $i$ with full or partial matches $j \in P_i$ , we modify the pairwise loss term from Equation (1) as:

$$
\hat {\ell} _ {i, j} = - \log \frac {\exp (s _ {i , j})}{\exp (s _ {i , j}) + \sum_ {k \notin \hat {P} _ {i}} \exp (s _ {i , j})}, \tag {5}
$$

where $\hat{P}_i = P_i\cup \{i\}$ . We then combine these terms by taking a mean per image, so that each image contributes similarly to the overall loss, and average per-image losses. Note that this is equivalent to InfoNCE for non-mixed images.

$$
\mathcal {L} _ {\text {I n f o N C E - m i x}} = \frac {1}{2 N} \sum_ {i = 1} ^ {2 N} \frac {1}{| P _ {i} |} \sum_ {j \in P _ {i}} \hat {\ell} _ {i, j}. \tag {6}
$$

Entropy loss. Our formulation of the entropy loss in Equation (3) remains the same, with $\hat{P}_i$ updated to include multiple matching images.

Combination. The losses are combined with entropy weight parameter $\lambda$ :

$$
\mathcal {L} = \mathcal {L} _ {\text {I n f o N C E - m i x}} + \lambda \mathcal {L} _ {\mathrm {K o L e o}} \tag {7}
$$

Multi-GPU implementation. The contrastive matching task benefits from a large batch size, since this provides stronger negatives. Losses are evaluated over the global batch, after aggregating image descriptors across GPUs. Descriptors from all GPUs are included in the negatives InfoNCE matches against, and we choose nearest neighbors for entropy regularization from the global batch. Batch normalization statistics are synchronized across GPUs to avoid leaking information within a batch. We use the LARS [58] optimizer for stable training at large batch size.

# 4.4. Inference and retrieval

For inference, the loss terms are discarded. Features are extracted from the images using the convolutional trunk followed by GeM pooling, the linear projection head, and L2 normalization. Then we apply whitening to the descriptors. The whitening matrix is learned on the DISC2021 training set. The descriptors are compared with cosine similarity or equivalently with simple L2 distance.

# 4.5. Similarity normalization

We follow [18] using similarity normalization [12, 27] as one of our evaluation settings. It uses a background dataset of images as a noise distribution, and produces high similarity scores only for queries whose reference similarity is greater than their similarity to nearest neighbors in the background dataset. Given a query image $q$ and a reference image $r$ with similarity $s(q, r) = \mathrm{sim}(z_q, z_r)$ , the adjusted similarity is $\hat{s}_0(q, r) = s(q, r) - \beta s(q, b_n)$ where $b_n$ is the $n^{\text{th}}$ nearest neighbor from the background dataset, and $\beta \geq 0$ is a weight.

We generalize this by aggregating an average similarity across multiple neighbors ( $n$ to $n_{\mathrm{end}}$ ) from the background dataset:

$$
\hat {s} (q, r) = s (q, r) - \underbrace {\frac {\beta}{1 + n _ {\mathrm {e n d}} - n} \sum_ {i = n} ^ {n _ {\mathrm {e n d}}} s (q , b _ {i})} _ {\mathrm {b i a s} (q)}. \quad (8)
$$

Integrated bias. Carrying around a bias term makes indexing of descriptors more complex. Therefore, we include the bias into the descriptors as an additional dimension:

$$
\hat {z} _ {q} = \left[ \begin{array}{l l} z _ {q} & - \operatorname {b i a s} (q) \end{array} \right] \quad \hat {z} _ {r} = \left[ \begin{array}{l l} z _ {r} & 1 \end{array} \right] \tag {9}
$$

Then we are back to $\hat{s}(q,r) = \mathrm{sim}(\hat{z}_q,\hat{z}_r)$ . The descriptors are not normalized, i.e. the dot product similarity is not equivalent to L2 distance. If L2 distance is preferred for indexing, it is possible to convert the max dot product search task into L2 search using the approach from [5].

Similarity normalization consistently improves metrics. However it adds operational complexity, and may make it difficult to detect content similar to the background distribution. Therefore, we report results both with and without this normalization.

# 5. Experiments

In this section we evaluate SSCD for image copy detection. Despite its relative simplicity, it depends on various settings that we evaluate in an extensive ablation study.

# 5.1. Datasets

DISC2021. Most evaluations are on the validation dataset of the Image Similarity Challenge, DISC2021 [18]. DISC2021 contains both automated image transforms and manual edits. There are 1 million reference images and 50,000 query images, of which 10,000 are true copies. A disjoint 1 million image training set is used for model training and as background dataset for score normalization. The training set contains no copies or labels, but is representative of the image distribution of the dataset. The performance is evaluated with micro average precision $(\mu AP)$ that measures the precision-recall tradeoff with a uniform distance threshold.

ImageNet. For some experiments we train models on the ImageNet [41] training set (ignoring the class labels). We use ImageNet linear classi
fication to measure how our copy detection methods affect semantic representation learning.

Copydays [17] is a small copy detection dataset. Following common practice [7,9], we augment it with 10k distractors from YFCC100M [45], a setting known as CD10K, and evaluate the retrieval performance with mean average precision $(mAP)$ on the "strong" subset of robustly transformed copies. In addition to this standard measure, we evaluate the $\mu AP$ on the overall dataset.

# 5.2. Training implementation

We use the training schedule and hyperparameters from SimCLR [10]: batch size $N = 4096$ , resolution $224 \times 224$ , learning rate of $0.3 \times N / 256$ , and a weight decay of $10^{-6}$ . We train models for 100 epochs on either ImageNet or the DISC training set, using a cosine learning rate schedule without restarts and with a linear ramp-up. We use the LARS optimizer for stable training at large batch size. We train at spatial resolution $224 \times 224$ .

We use a lower temperature than SimCLR, $\tau = 0.05$ versus 0.1, following an observation in [10] that this setting

yields better accuracy on the training task, while reducing accuracy of downstream classification tasks.

# 5.3. Evaluation protocol

Inference. We resize the small edge of an image to size 288 preserving aspect ratio for fully convolutional models. We use a larger inference size than seen at training to avoid train-test discrepancy [50]. We use different preprocessing for the DINO [9] ViT baseline, following their copy detection method. See Appendix D for details.

Descriptor postprocessing. Image retrieval benefits from PCA whitening. SSCD descriptors are whitened, then L2 normalized. For baseline methods that use CNN trunk features, we L2 normalize both before and after whitening. SimCLR projection features often occupy a low-dimensional subspace, making whitening at full descriptor size unstable, and many representations perform better when whitened with low-variance dimensions excluded. For baseline methods, we try dimensionalities $\{d, \frac{3}{4}d, \frac{d}{2}, \frac{d}{4}, \ldots\}$ and choose the one that maximizes the final accuracy. For SSCD, we whiten at full descriptor size.

We use the FAISS [30] library to apply embedding postprocessing and perform exhaustive k-nearest neighbor search. We train PCA on the DISC2021 training dataset, following standard protocol for this dataset.

# 5.4. Results

Table 2. Copy detection performance in $\mu AP$ on the DISC2021 dataset. *: methods that use supervised labels. †: trunk larger than ResNet50. DINO baseline uses ViT-B/16.

![](dt=2026-06-04/ht=19/cfa9867007e31f821cf7fbfebbc2a1311664dd394062bcc763e217d37487d9fd.jpg)

<table><tr><td>method</td><td>trained on</td><td>transforms</td><td>dims</td><td>μAP</td><td>μAPSN</td></tr><tr><td>Multigrain [7, 18]</td><td>ImageNet*</td><td></td><td>1500</td><td>16.5</td><td>36.5</td></tr><tr><td>HOW [18,48]</td><td>SfM-120k*</td><td></td><td></td><td>17.3</td><td>37.2</td></tr><tr><td>Multigrain [7]</td><td>ImageNet*</td><td></td><td>2048</td><td>20.5</td><td>41.7</td></tr><tr><td>DINO [9]†</td><td>ImageNet</td><td></td><td>1500</td><td>32.2</td><td>53.8</td></tr><tr><td>SimCLR [10] trunk</td><td>ImageNet</td><td>SimCLR</td><td>2048</td><td>13.1</td><td>33.9</td></tr><tr><td>SimCLR [10] proj</td><td>ImageNet</td><td>SimCLR</td><td>128</td><td>9.4</td><td>17.3</td></tr><tr><td>SimCLRCD trunk</td><td>ImageNet</td><td>strong blur</td><td>2048</td><td>39.8</td><td>56.8</td></tr><tr><td>SSCD</td><td>ImageNet</td><td>strong blur</td><td>512</td><td>50.4</td><td>64.5</td></tr><tr><td>SSCD</td><td>ImageNet</td><td>advanced</td><td>512</td><td>55.5</td><td>71.0</td></tr><tr><td>SSCD</td><td>ImageNet</td><td>adv.+mixup</td><td>512</td><td>56.8</td><td>72.2</td></tr><tr><td>SSCD</td><td>DISC</td><td>strong blur</td><td>512</td><td>54.8</td><td>63.6</td></tr><tr><td>SSCD</td><td>DISC</td><td>advanced</td><td>512</td><td>60.4</td><td>71.1</td></tr><tr><td>SSCD</td><td>DISC</td><td>adv.+mixup</td><td>512</td><td>61.5</td><td>72.5</td></tr><tr><td>SSCDlarge†</td><td>DISC</td><td>adv.+mixup</td><td>1024</td><td>63.7</td><td>75.3</td></tr></table>

DISC results. Table 2 reports DISC2021 results from the baseline methods published in [18] and SSCD. Our evaluation protocol obtains somewhat stronger results for the Multigrain baseline ( $3^{\text{rd}}$ row). The first observation is that SSCD improves the baseline accuracy by $2 \times$ to $5 \times$ before

score normalization, demonstrating that copy detection benefits from specific architectural and training adaptations.

We present results on a few different SSCD models trained on ImageNet or DISC2021, using the three augmentation settings we propose. The intermediate model $\mathrm{SimCLR}_{\mathrm{CD}}$ has all of our proposed changes except the entropy loss. $\mathrm{SSCD}_{\mathrm{large}}$ model uses a larger descriptor size and a ResNeXt-101 trunk.

We evaluate SimCLR using both trunk and projected features, and find trunk features $(\mu AP = 13.1)$ to outperform features from the projection head $(\mu AP = 9.4)$ with and without score normalization. Further experiments (Appendix A) show the reverse when training with entropy loss: projected features have similar accuracy to trunk features, despite a much more compact representation.

The gain of $\mathrm{SimCLR}_{\mathrm{CD}}$ ( $\mu AP = 39.8$ without score normalization) over SimCLR (13.1) is decomposed in Section 5.5. Introducing the entropy loss in SSCD contributes an additional $10\%$ absolute of $\mu AP$ , which is further increased by stronger augmentations $(+6.2\%)$ and training on a dataset with less domain shift $(+4.7\%)$ . These findings are confirmed after score normalization.

Copydays results. Table 3 reports results for baseline methods using publicly released models, but omit Multigrain settings that we were unable to reproduce. We used published preprocessing settings for baselines and whitening. Our DINO results outperform published results.

Table 3. Copydays (CD10K) accuracy measured in mAP on the "strong" subset, and $\mu AP$ on the full dataset.

![](dt=2026-06-04/ht=19/08b65a4fdae40d1b46a59a25441c94138439639a787b63c95de92388761513c2.jpg)

<table><tr><td>model</td><td>trunk</td><td>dims</td><td>size</td><td>\( {mAP} \)</td><td>\( {\mu AP} \)</td></tr><tr><td>Multigrain [7]</td><td>ResNet50</td><td>1500</td><td>long 800</td><td>82.3</td><td>77.3</td></tr><tr><td>DINO [9]</td><td>ViT-B/16</td><td>1536</td><td>\( {224}^{2} \)</td><td>82.8</td><td>92.3</td></tr><tr><td>DINO [9]</td><td>ViT-B/8</td><td>1536</td><td>\( {320}^{2} \)</td><td>86.1</td><td>88.4</td></tr><tr><td>SSCD</td><td>ResNet50</td><td>512</td><td>short 288</td><td>86.6</td><td>98.1</td></tr><tr><td>\( {\mathrm{{SSCD}}}_{\text{large }} \)</td><td>ResNeXt101</td><td>1024</td><td>long 800</td><td>93.6</td><td>97.1</td></tr></table>

The first SSCD result is with all settings from our DISC2021 experiments, where we resize the short side of each image to 288 pixels. With no tuning on this dataset, our method outperforms published results. We also show results for $\mathrm{SSCD}_{\mathrm{large}}$ using a ResNeXt101 trunk and 1024 descriptor dimensions, at larger inference size. We report more results on CD10K in Appendix B.

In addition to state-of-the-art accuracy using the customary $mAP$ ranking metric, our method provides a significant improvement in the global $\mu AP$ metric, indicating better distance calibration. On high-resolution images that are common for image retrieval, we observe improved $mAP$ but degraded $\mu AP$ . SSCD descriptors are more compact than baselines.

# 5.5. Ablations

Comparison with SimCLR. We provide a stepwise comparison between SimCLR and our method in Table 4. SimCLR projection features are not particularly strong fo
r this task until we apply several of our adaptations. SimCLR is unable to exploit a $\mathbb{R}^{512}$ descriptor, only slightly outperforming its $\mathbb{R}^{128}$ setting. $\mathrm{SimCLR}_{\mathrm{CD}}$ represents our architectural and hyper-parameter changes before adding differential entropy representation. Differential entropy regularization alone adds $+17.4\%$ $\mu AP$ and $+12.9\%$ $\mu AP_{SN}$ , more than any other step.

Table 4. Ablation from SimCLR to our method, showing DISC2021 $\mu AP$ performance for models trained on ImageNet. To compare descriptors of different sizes, we also show metrics after PCA reduction to 256 dimensions.

![](dt=2026-06-04/ht=19/cb160eceefb83a3deac169ddda139d3a42fa9dda57f73e495dbdcd6d7e53f814.jpg)

<table><tr><td rowspan="2">name</td><td colspan="2">Score normalization:</td><td colspan="2">No</td><td colspan="2">Yes</td></tr><tr><td>method</td><td>dims</td><td>μAP</td><td>256d</td><td>μAPSN</td><td>256d</td></tr><tr><td rowspan="2">SimCLR</td><td>trunk features</td><td>2048</td><td>13.1</td><td>7.3</td><td>33.9</td><td>26.8</td></tr><tr><td>+ GeM pooling</td><td>2048</td><td>21.5</td><td>12.1</td><td>45.3</td><td>35.8</td></tr><tr><td rowspan="5">SimCLR</td><td>projection</td><td>128</td><td>9.4</td><td>9.4</td><td>17.3</td><td>17.3</td></tr><tr><td>+ GeM pooling</td><td>128</td><td>11.1</td><td>11.1</td><td>18.8</td><td>18.8</td></tr><tr><td>+ strong blur</td><td>128</td><td>14.1</td><td>14.1</td><td>26.0</td><td>26.0</td></tr><tr><td>+ low temp</td><td>128</td><td>26.0</td><td>26.0</td><td>41.5</td><td>41.5</td></tr><tr><td>+ 512d proj</td><td>512</td><td>27.5</td><td>27.5</td><td>43.5</td><td>43.5</td></tr><tr><td>SimCLRCD</td><td>+ linear proj</td><td>512</td><td>33.0</td><td>32.4</td><td>51.6</td><td>50.5</td></tr><tr><td>SSCD</td><td>+ entropy loss</td><td>512</td><td>50.4</td><td>44.0</td><td>64.5</td><td>57.8</td></tr><tr><td>SSCD</td><td>+ adv. augs</td><td>512</td><td>55.5</td><td>49.7</td><td>71.0</td><td>65.8</td></tr><tr><td>SSCD</td><td>+ mixup</td><td>512</td><td>56.8</td><td>51.1</td><td>72.2</td><td>67.1</td></tr></table>

Entropy weight. Table 5 compares how varying entropy loss weight $(\lambda)$ affects copy detection accuracy, using $\mathrm{SimCLR}_{\mathrm{CD}}$ as a baseline. Models for this experiment are trained using the strong blur augmentation setting.

Table 5. DISC2021 accuracy metrics with varying entropy weight $\lambda$ for models trained on ImageNet.

![](dt=2026-06-04/ht=19/544960ebd972bf6364bed4e09298fe5ff5db97e84cb143515b725fa499be1321.jpg)

<table><tr><td>model</td><td>μAP</td><td>μAPSN</td><td>recall@1</td><td>MRR</td></tr><tr><td>SimCLRCD</td><td>33.0</td><td>51.6</td><td>58.6</td><td>60.5</td></tr><tr><td>λ = 1</td><td>33.1</td><td>51.9</td><td>58.7</td><td>60.9</td></tr><tr><td>λ = 3</td><td>38.0</td><td>56.1</td><td>62.9</td><td>65.1</td></tr><tr><td>λ = 10</td><td>45.3</td><td>61.5</td><td>67.7</td><td>69.5</td></tr><tr><td>λ = 30</td><td>50.4</td><td>64.5</td><td>69.8</td><td>71.4</td></tr></table>

As the entropy weight increases, we see a corresponding increase in global accuracy metrics. We also see a similar increase in per-query ranking metrics, such as recall at 1 and mean reciprocal rank (MRR). The increase in ranking metrics demonstrates that differential entropy regularization improves copy detection accuracy in general, beyond creating a more uniform notion of distance.

In contrast to metric learning contexts where entropy regularization has been used, copy detection benefits from higher $\lambda$ values. Our standard setting is $\lambda = 30$ , while [19] reports reduced accuracy with $\lambda > 1$ , and [42] uses values $< 0.1$ . At $\lambda > 40$ , training becomes unstable, and tends to minimize the entropy loss at the expense of the InfoNCE loss: embeddings are uniformly distributed, but meaningless because image copies are not near anymore.

Additional ablations. We explore how batch size, training schedule, descriptor dimensions, and score normalization affect accuracy in Appendix A.

# 6. Discussion

Dimensional collapse. We find, similar to [29, 60], that SimCLR collapses to a subspace of approximately 256 dimensions when trained in 512 dimensions. Table 4 shows that SimCLR's accuracy does not improve much when the descriptor size increases from 128 to 512 dimensions. SSCD's entropy regularization resolves this collapse, and allows the model to use the full descriptor space.

Entropy regularization and whitening. SSCD is much more accurate than baselines when compared without whitening or similarity normalization: $47.8\mu AP$ for $\lambda = 30$ when trained on ImageNet, versus 26.8 for an equivalent $\lambda = 0$ model. Both the entropy loss and post-training PCA whitening aim at creating a more uniform descriptor distribution. However PCA whitening can distort the descriptor space learned during training, particularly when many dimensions have trivial variance. Differential entropy regularization promotes an approximately uniform space, allowing the model to adapt to an approximately whitened descriptor during training, reducing the distortion whitening induces.

Uniform distribution as a perceptual prior. For most experiments in this work we focus on the $\mu AP$ metric that requires a separation between matches and non-matches at a fixed threshold. However Table 5 shows that ranking metrics also improve with increased the entropy loss weight, i.e. better calibration across queries does not fully explain the benefit of entropy regularization.

Differential entropy regularization acts as a kind of prior, selecting for an embedding space that is uniformly distributed. We argue that, when applied to contrastive learning, this regularization is a perceptual prior, selecting for stronger copy detection representations. An ideal copy detection descriptor would map copies of the same image together, while keeping even semantically similar (same "class") images far apart i.e. the descriptor distribution is uniform. This differs from the ideal properties of a representation for transfer learning to classification, where images depicting the same class should be nearby (a dense region) and well separated other classes (a sparse region between classes).

Visual results. Figure 2 shows a few retrieval results, where SSCD outperforms the vanilla SimCLR. The two first examples demonstrate the impact of more appropriate data augmentation at training time: SSCD ignores text overlays and blur/color balance. The two last examples show that SimCLR falls back on low-level texture matching (grass) when SSCD correctly recovers the source image.

Limitations. Our method is explicitly text-insensitive when training with text augmentation, and we find that it is somewhat text-insensitive even when trained without text augmentation. For this reason, SSCD is not precise when matching images composed entirely of text. Different photos of the same scene (e.g. of landmarks) may be identified as copies, even if the photos are distinct. Sometimes, images are combined to create a composite image or collage, where the copied content may occupy only a small region of the composite image. "Partial" copies of this kind are hard to detect with global descriptor models like SSCD, and local descriptor methods may be necessary in this case. Finally, matching at high precision often requires an additional verification step.

Ethical considerations. We focus our investigation on the DISC2021 dataset, which is thoughtful in its approach to images of people, using only identifiable photos of paid actors who gave consent for their images to be used for research. Copy detection for content moderation is adversarial. There is a risk that publishing research for this problem will better inform actors aiming to evade detecti
on. We believe that this is offset by the improvements that open research will bring.

This technology allows scaling manual moderation, which helps protect users from harmful content. However, it can also be used for e.g. political censorship. We still believe that advancing this technology is a net benefit.

# 7. Conclusion

We presented a method to train effective image copy detection models. We have demonstrated architecture and objective changes to adapt contrastive learning to copy detection. We show that the differential entropy regularization dramatically improves copy detection accuracy, promoting consistent separation of image descriptors.

Our method demonstrates strong results on DISC2021, significantly surpassing baselines, and transfers to Copydays, yielding state-of-the-art results. Our method is efficient because it relies on a standard trunk, uses smaller inference sizes than are typical for image retrieval, and produces a compact descriptor. Additionally, its calibrated distance metric limits candidates for verification. We believe that these results demonstrate a unique compatibility between uniform embedding distributions and the task of copy detection.

# References

# Appendix

We provide more details about the ablations (Appendix A) and Copydays results (Appendix B). We also report a few additional details about the embedding distribution (Appendix C) and implementation details (Appendix D). The last appendix F shows additional example matches.

# A. Additional ablations

Table 6 shows how copy detection accuracy is affected by several hyper-parameters.

Descriptor dimensionality. The descriptor dimension is a tradeoff between accuracy and the efficiency of the retrieval step. When constraining the descriptor to 256 dimensions for retrieval, we see highest accuracy for descriptors trained at that size.

Batch size. The training objective learns to match pairs within the global batch (across all GPUs). A larger batch size makes the training task more challenging, improving the final accuracy. Large batch sizes require training with more machines, and incur synchronization overhead due in part to synchronized batch normalization.

Training schedule. We compare accuracy as we vary the number of training epochs, and find no benefit to longer training schedules.

Variance between initializations. We train using the same setting, initializing the model with five random seeds, and find a standard deviation of $0.2\%$ $\mu AP$ and $0.1\%$ $\mu AP_{SN}$ .

Similarity normalization settings. We show score normalized accuracy given several similarity normalization settings in Table 7. Several score normalization settings work similarly well. When using a single neighbor to normalize similarity, using the 2nd nearest neighbor works best $(n = 2)$ . When using an average similarity across multiple neighbors, averaging the first 2, 3 or 4 neighbors work similarly well. We find that $\beta = 1$ is a good normalization weight. Our similarity normalized results use $n = 1$ ,

$n_{end} = 3$ , $\beta = 1$ , a setting that we found to work well across many descriptors.

Table 7. DISC2021 $\mu AP$ with different score normalization settings for a SSCD trained on DISC2021 with advanced augmentations.

![](dt=2026-06-04/ht=19/ddeedc68546216e234a92dabe3151f9b26eca10fe5b8017f532fc133f6341b6a.jpg)

<table><tr><td colspan="2">β = 1, n = n_end</td><td colspan="2">β = 1, n = 1</td><td colspan="2">n = 1, n_end = 3</td></tr><tr><td>n</td><td>μAP</td><td>| n_end</td><td>μAP</td><td>| β</td><td>μAP</td></tr><tr><td>1</td><td>69.5</td><td>1</td><td>69.5</td><td>0.50</td><td>68.4</td></tr><tr><td>2</td><td>71.1</td><td>2</td><td>71.0</td><td>0.75</td><td>70.4</td></tr><tr><td>3</td><td>70.8</td><td>3</td><td>71.1</td><td>1.00</td><td>71.1</td></tr><tr><td>4</td><td>70.3</td><td>4</td><td>71.1</td><td>1.25</td><td>71.1</td></tr><tr><td>5</td><td>69.7</td><td>5</td><td>71.0</td><td>1.50</td><td>70.6</td></tr></table>

Trunk and projected features. We compare SSCD trunk and projected features in Table 8. Using the linear projection at inference time improves accuracy, despite a significantly more compact code.

Table 8. DISC2021 accuracy of SSCD trunk and projected trained on DISC2021 with advanced + mixup augmentations.

![](dt=2026-06-04/ht=19/1d433bd228148c095e8cce2651d4cf31f698f944997d7e8db91b45b073d0767c.jpg)

<table><tr><td>descriptor</td><td>dims</td><td>μAP</td><td>μAPSN</td></tr><tr><td>trunk</td><td>2048</td><td>57.2</td><td>71.9</td></tr><tr><td>projected</td><td>512</td><td>61.5</td><td>72.5</td></tr></table>

# B. Full Copydays results

We provide additional Copydays results in Table 9, evaluating SSCD and $\mathrm{SSCD}_{\mathrm{large}}$ using preprocessing settings from prior published results. In each case, we evaluate our method with no tuning, e.g. we don't adjust the GeM $p$ as proposed in [7].

We note that at $224^{2}$ inference size, ResNet50 has approximately $4\times$ the throughput as ResNeXt101 or ViT-B/16, and $20\times$ that of ViT-B/8. [9]

![](dt=2026-06-04/ht=19/bb0860eb5282a3993ddc47bbb31c29b5646e5acb33388d23a7b0a410f3eac85f.jpg)

<table><tr><td>batch size</td><td>μAP</td><td>μAPSN</td></tr><tr><td>2048</td><td>54.4</td><td>67.7</td></tr><tr><td>4096</td><td>56.6</td><td>69.2</td></tr><tr><td>8192</td><td>58.2</td><td>70.0</td></tr><tr><td>16384</td><td>59.4</td><td>70.2</td></tr></table>

![](dt=2026-06-04/ht=19/e3f924cf73415fd19e7ba9fb28a6184683758e7ab66923f988a8fcbdc80366f4.jpg)

<table><tr><td>epochs</td><td>μAP</td><td>μAPSN</td></tr><tr><td>25</td><td>54.4</td><td>67.4</td></tr><tr><td>50</td><td>56.2</td><td>68.9</td></tr><tr><td>100</td><td>56.6</td><td>69.2</td></tr><tr><td>200</td><td>56.3</td><td>68.9</td></tr><tr><td>400</td><td>55.7</td><td>68.1</td></tr></table>

Table 6. Impact of three training parameters on the accuracy: batch size, number of epochs and dimensionality. We report $\mu AP$ performance on DISC21 for SSCD including advanced augmentations and $\lambda = 15$ , with and without score normalization. For the dimensionality experiment we additionally report the accuracy after reduction to 256 dimensions.

![](dt=2026-06-04/ht=19/a8681cdd92795475f15e49e0d8dfd2945442157838fde945bd54cb3e32749fa4.jpg)

<table><tr><td>dimensions</td><td>μAP</td><td>μAPSN</td><td>μAPSN256d</td></tr><tr><td>128</td><td>49.4</td><td>59.4</td><td>59.4</td></tr><tr><td>256</td><td>53.9</td><td>65.6</td><td>65.6</td></tr><tr><td>512</td><td>56.6</td><td>69.2</td><td>64.0</td></tr><tr><td>1024</td><td>57.3</td><td>70.9</td><td>62.8</td></tr><tr><td>2048</td><td>56.8</td><td>70.8</td><td>62.9</td></tr></table>

Table 9. Full Copydays (CD10K) results: accuracy measured in mAP on the "strong" subset, and $\mu AP$ on the full dataset.

![](dt=2026-06-04/ht=19/eddcb817eebde22aa66c92026084fe23affc5d160b7c2944206215ef3d221f19.jpg)

<table><tr><td>model</td><td>trunk</td><td>dims</td><td>size</td><td>mAP</td><td>μAP</td></tr><tr><td>Multigrain [7]</td><td>ResNet50</td><td>1500</td><td>long 800</td><td>82.3</td><td>77.3</td></tr><tr><td>DINO [9]</td><td>ViT-B/16</td><td>1536</td><td>2242</td><td>82.8</td><td>92.3</td></tr><tr><td>DINO [9]</td><td>ViT-B/8</td><td>1536</td><td>3202</td><td>86.1</td><td>88.4</td></tr><tr><td>SSCD</td><td>ResNet50</td><td>512</td><td>2242</td><td>84.9</td><td>98.3</td></tr><tr><td>SSCD</td><td>ResNet50</td><td>512</td><td>3202</td><td>87.4</td><td>98.3</td></tr><tr><td>SSCD</td><td>ResNet50</td><td>512</td><td>short 288</td><td>86.6<
/td><td>98.1</td></tr><tr><td>SSCD</td><td>ResNet50</td><td>512</td><td>long 800</td><td>90.0</td><td>93.9</td></tr><tr><td>SSCDlarge</td><td>ResNeXt101</td><td>1024</td><td>2242</td><td>87.3</td><td>98.6</td></tr><tr><td>SSCDlarge</td><td>ResNeXt101</td><td>1024</td><td>3202</td><td>90.6</td><td>98.6</td></tr><tr><td>SSCDlarge</td><td>ResNeXt101</td><td>1024</td><td>short 288</td><td>91.8</td><td>98.7</td></tr><tr><td>SSCDlarge</td><td>ResNeXt101</td><td>1024</td><td>long 800</td><td>93.6</td><td>97.1</td></tr></table>

![](dt=2026-06-04/ht=19/68005d0edc65b2c538f017e9800ab798a53faa005d83901c4e573375f7e4f8f9.jpg)

# C. Embedding distribution

We plot principal values for SSCD $(\lambda = 30)$ compared to $\mathrm{SimCLR}_{CD}(\lambda = 0)$ , and a uniform distribution in Figure 5. We see that the $\lambda = 0$ model fails to make full use of the descriptor space, as observed in [29, 60]. With entropy regularization, all components have similar energy, spanning less than an order of magnitude (the maximum is $6.6\times$ the minimum).

# D. Implementation details

Mixup and Cutmix. Mixup and Cutmix augmentations both combine content from two source images. The amount of content used from each image is determined by a mixing parameter $\gamma$ , sampled from a $\beta$ distribution: $\gamma \sim \beta (\alpha ,\alpha)$ . We set $\alpha = 2$ to reduce the prevalence of "trivial" mixed images that draw nearly all content from one of the inputs.

DINO baseline details. We follow the copy detection method presented in [9] for the DINO baseline. We use the

![](dt=2026-06-04/ht=19/c24dca4f431c7b1c466118e42bdec2871572155c634bd3da658858718a3418c2.jpg)

![](dt=2026-06-04/ht=19/dbab31c94c96566470ccd525da9a408d4b5d49a34f750d1645b6d1951113d9bb.jpg)

![](dt=2026-06-04/ht=19/f59303c8fc3154f61ee2967e5ad8868b415a31d896a939e7b1fd3e72e8edcc3b.jpg)

![](dt=2026-06-04/ht=19/b3c73f5c123fb9510ba39f40fc63682d7a9ee01f16131d8c21d286e20a17a0a9.jpg)

![](dt=2026-06-04/ht=19/257b7b51a594f59c089fb1c8d480c6d959891b8e51366c462f39efae7dba256c.jpg)

![](dt=2026-06-04/ht=19/d8fe663b703e27fba9748f75f326b0bc128d5ef848a2b048c7e0a4f3c8db9d0e.jpg)

![](dt=2026-06-04/ht=19/681aa17d3e8b6c4eedcf218f616e27cf26baf643699013f7ddcf18e9cb384cf2.jpg)

![](dt=2026-06-04/ht=19/4b0ce96e7425fae2bbc52665362486679b4a1d1e378979693482043352c6993e.jpg)

![](dt=2026-06-04/ht=19/535840849c20876bef8e965113627d5c24e41d9b96b3e3ffd6dc0c9c851fbb71.jpg)

![](dt=2026-06-04/ht=19/68e009ba623ed9642957d64b7924f512b166b11609b67bc3cb5ac7159e218998.jpg)

![](dt=2026-06-04/ht=19/1d97e7b174d27c2e0739d9de522f2f87149cad4764e38015cc961520c2e2b0c9.jpg)

![](dt=2026-06-04/ht=19/438b757b668158b4b33c859bf3db17fc714b058e064a28ba2d8f5ab3f9a52034.jpg)

concatenation of the CLS token and GeM pooled $(p = 4)$ patch token features as the descriptor.

Our DINO DISC evaluation uses the ViT-B/16 trunk. We resize inputs to $224 \times 224$ without center cropping. This outperformed other preprocessing for this model, including our default aspect-ratio preserving resize, and resizing inputs to a larger fixed size ( $288 \times 288$ ). We suspect that ViT models may be less adaptable to rectangular inputs than fully convolutional networks.

# E. Visualizing matches

To view which parts of an image A match strongly to another image B, we experiment by keeping the activation map on A at full resolution by removing the GeM pooling operation. This results into one descriptor per activation map pixel, that can be compared with a global SSCD descriptor. We can thus build a spatial heatmap with the strongest activations. Figure 6 shows image pairs and the corresponding heatmaps. The areas on the left image that match with the image on the right are clearly identified.

# F. Retrieved matches

We compare the first result retrieved by SSCD and SimCLR on the DISC2021 dataset. Both models are trained on ImageNet and evaluated with whitening. We use trunk features for SimCLR, which are more accurate for this model.

We do not use score normalization, since it has no effect on top-1 accuracy.

Table 10. Percentage of DISC2021 query first result accuracy by model for SSCD and SimCLR trained on ImageNet.

![](dt=2026-06-04/ht=19/698610a7c5955f75bfe0137ad2b0d7172f249878e0060db45a79bf7970af2f1a.jpg)

<table><tr><td>SSCD</td><td>SimCLR</td><td>queries</td></tr><tr><td>✓</td><td>✓</td><td>38.9 %</td></tr><tr><td>✓</td><td>X</td><td>39.0 %</td></tr><tr><td>X</td><td>✓</td><td>0.3 %</td></tr><tr><td>X</td><td>X</td><td>21.8 %</td></tr></table>

Table 10 shows quantitative results from this exercise. SSCD correctly identifies the copy as the first result $2 \times$ as often as SimCLR. Correct SSCD matches are nearly a superset of SimCLR matches: very rarely does SimCLR have a correct first result that SSCD misses.

Figure 7 shows additional queries and retrieved results for examples that only SSCD correctly identifies. One pattern we observe is that SimCLR often matches images with similar types of distortion together. Images with text at an angle, or strong diagonal features, may be incorrectly matched with images with similar features. Images with a blurry, or grainy, quality are matched to other images with a similar quality. This is surprising given that SimCLR trains with a blur augmentation, albeit weaker, and should be somewhat blur invariant.

![](dt=2026-06-04/ht=19/d7962c5697b58594bbd47839a86bd42c42cbbf9552a07c468804e772f2e15a90.jpg)