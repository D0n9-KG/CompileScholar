# VECTOR GRIMOIRE: Codebook-based Shape Generation under Raster Image Supervision

Moritz Feuerpfeil\* Marco Cipriano\* Gerard de Melo

Hasso Plattner Institute (HPI)

marco.cipriano@hpi.de

https://potpov.github.io/grimoire-web/

# Abstract

Scalable Vector Graphics (SVG) is a popular format on the web and in the design industry. However, despite the great strides made in generative modeling, SVG has remained underexplored due to the discrete and complex nature of such data. We introduce GRIMOIRE, a text-guided SVG generative model that is comprised of two modules: A Visual Shape Quantizer (VSQ) learns to map raster images onto a discrete codebook by reconstructing them as vector shapes, and an Auto-Regressive Transformer (ART) models the joint probability distribution over shape tokens, positions, and textual descriptions, allowing us to generate vector graphics from natural language. Unlike existing models that require direct supervision from SVG data, GRIMOIRE learns shape image patches using only raster image supervision which opens up vector generative modeling to significantly more data. We demonstrate the effectiveness of our method by fitting GRIMOIRE for closed filled shapes on MNIST and for outline strokes on icon and font data, surpassing previous image-supervised methods in generative quality and the vector-supervised approach in flexibility.

# 1 Introduction

In the domain of computer graphics, Scalable Vector Graphics (SVG) has emerged as a versatile format, enabling the representation of 2D graphics with precision and scalability. SVG is an XML-based vector graphics format that describes a series of parametrized shape primitives rather than a limited-resolution raster of pixel values. While modern generative models have made significant advancements in producing high-quality raster images (Ho et al., 2020; Isola et al., 2017; Saharia et al., 2022; Nichol et al., 2021), SVG generation remains a less explored task. Existing works that have aimed to train a deep neural network for this goal primarily adopted language models to address the problem (Wu et al., 2023; Tang et al., 2024). In general, existing approaches share two key limitations: they necessitate SVG data for direct supervision which inherently limits the available data and increases the burden of data pre-processing, and they are not easily extendable when it comes to visual attributes such as color or stroke properties. The extensive pre-processing is required due to the diverse nature of an SVG file that can express shapes as a series of different basic primitives such as circles, lines, and squares – each having different properties – that can overlap and occlude each other.

An ideal generative model for SVG should however benefit from visual guidance for supervision, which is not possible when merely training to reproduce tokenized SVG primitives, as there is no differentiable mapping to the generated raster imagery.

![](images/d90976dbfcdd162566a5b14698d728d9957886b3a0eef9e763858c3be7ad1206.jpg)

<details>
<summary>text_image</summary>

Ours
Im2Vec
</details>

Figure 1: Generative results for fonts and icons from GRIMOIRE and Im2Vec. Since Im2Vec does not accept any conditioning, we sample after training Im2Vec only on icons of stars or the letter A, respectively. For GRIMOIRE we use the models trained on the full dataset conditioned on the respective class.

In this paper, we present GRIMOIRE (Shape Generation with raster image supervision), a novel pipeline explicitly designed to generate SVG files with only raster image supervision. Our approach incorporates a differentiable rasterizer, DiffVG (Li et al., 2020), to bridge the vector graphics primitives and the raster image domain. We adopt a VQ-VAE recipe (Van Den Oord et al., 2017), which pairs a codebook-based discrete auto-encoder with an auto-regressive Transformer that models the image space implicitly by learning the distribution of codes that resemble them. We find this approach particularly promising for vector graphics generation, as it breaks the complexity of this task into two stages. In the first stage of our method, we decompose images into primitive shapes represented as patches. A vector-quantized auto-encoder learns to encode and map each patch into a discrete codebook, and decode these codes to an SVG approximation of the input patch, which is trained under raster supervision. In the second stage, the series of raster patches containing primitives are encoded and the prior distribution of codes is learned by an auto-regressive Transformer model conditioned on a textual description. At inference, a full series of codes can be generated from textual input, or other existing shape codes. Therefore, GRIMOIRE supports text-to-SVG generation and SVG auto-completion as possible downstream tasks out-of-the-box.

The key contributions of this work are:

1. We frame the problem of image-supervised SVG generation as the prediction of a series of individual shapes and their positions on a shared canvas.   
2. We train the first text-conditioned generative model that learns to draw vector graphics with only raster image supervision.   
3. We compare our model with alternative frameworks showing superior performance in generative capabilities on diverse datasets.   
4. Upon acceptance, release the code of this work to the research community.

# 2 Related Work

# 2.1 SVG Generative Models

The field of vector graphics generation has witnessed increasing interest. Following the extraordinary success of Large Language Models (LLM), the most recent approaches (Lopes et al., 2019; Aoki and Aizawa, 2022; Wu et al., 2023; Tang et al., 2024) have recast the problem as an NLP task, learning a distribution over tokenized SVG commands. Iconshop (Wu et al., 2023) introduced a method of tokenizing SVG paths that makes them suitable input for causal language modeling. To add conditioning, they employed a pre-trained language model to tokenize and embed textual descriptions, which are concatenated with the SVG tokens to form sequences that the auto-regressive Transformer can learn a joint probability on.

StrokeNUWA (Tang et al., 2024) introduced Vector Quantized Strokes to compress SVG strokes into a codebook with SVG supervision and fine-tune a pre-trained Encoder–Decoder LLM to predict these tokens given textual input. However, both of these approaches suffer from a number of limitations.

![](images/eee218b16f04f080e4d091c45f5b2cf432f80fac9685457426e45ead79f1c569.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph VSQ module
        A["\ - - / / /\"]
        B["Discrete Encoder"] --> C["Discrete Latent Codes {v₁,v₂,...,vₙ}"]
        C --> D["Codebook V"]
        D --> E["Projection Layer"]
        E --> F["SVG Prediction Head"]
        F --> G["Differentiable Rasterizer"]
        G --> H["\ - - / / /\"]
        H --> I["L_recons"]
    end

    subgraph Dataset Tokenization
        J["The icon of teapot"] --> K["BERT Encoder"]
        K --> L["Trained Discrete Encoder"]
        L --> M["Store for ART training"]
        M --> N["Discrete Patch Coordinates Θ"]
        N --> O["... EOS"]
    end

    subgraph ART module
        P["Load Tokenized Image Sequence"] --> Q["SOS τ₁ τ₂ τₜ BOS θ₁ v₁ θ₂ v₂ θᵢ vᵢ ... EOS"]
        Q --> R["Auto-Regressive Model"]
        R --> S["θ₁ v₁ θ₂ v₂ θᵢ vᵢ ... EOS"]
        S --> T["L_Causal"]
    end

    style VSQ module fill:#f9f,stroke:#333
    style Dataset Tokenization fill:#bbf,stroke:#333
    style ART module fill:#bfb,stroke:#333
```
</details>

Figure 2: Overview of GRIMOIRE. On the left, the training process of our VSQ module is depicted, where raster input patches are encoded into discrete codes and reconstructed as SVG shapes using visual supervision. In the top right, each image is encoded into a series of discrete codes using the trained VSQ encoder and its textual description. The bottom right illustrates how the ART module learns the joint distribution of these codes and the corresponding text.

First, they require a corpus of SVG data for training, which hinges upon large pre-processing pipelines to remove redundancies, convert non-representable primitives, and standardize the representations.

Secondly, there is no supervision of the visual rendering, which makes the models prone to data quality errors, e.g., excessive occlusion of shapes. Finally, these models lack any straightforward extensibility towards the inclusion of new visual features such as colours, stroke widths, or fillings and alpha values.

Hence, another line of work has sought to incorporate visual supervision. These approaches generally rely on recent advances in differentiable rasterization, which enables backpropagation of raster-based losses through different types of vectorial primitives such as Bézier curves, circles, and squares. The most important development in this area is DiffVG (Li et al., 2020), which removed the need for approximations and introduced techniques to handle antialiasing.

They further pioneered image-supervised SVG generative models by training a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN) (Goodfellow et al., 2014) on MNIST (LeCun et al., 1998) and QuickDraw (Ha and Eck, 2017). These generative capabilities have subsequently been extended in Im2Vec (Reddy et al., 2021), which adopts a VAE including a recurrent neural network to generate vector graphics as sets of deformed and filled circular paths, which are differentiably composited and rasterized, allowing for back-propagation of a multi-resolution MSE-based pyramid loss. However, all of these models lack versatile conditioning (such as text) and focus on either image vectorization, i.e., the task of creating the closest vector representation of a raster prior, or vector graphics interpolation. We show in Section 5 that these approaches fail to capture the diversity and complexity of datasets such as FIGR-8, and generate repetitive samples.

A different type of SVG generation enabled by DiffVG is painterly rendering (Ganin et al., 2018; Nakano, 2019), where an algorithm iteratively fits a given set of vector primitives to match an image, guided by a deep perceptual loss function. To achieve this goal, CLIPDraw (Frans et al., 2022) rasterized a set of randomly initialized SVG paths and encoded these with a pre-trained CLIP (Radford et al., 2021) image encoder, iteratively minimizing the cosine distance between such

embeddings and the text description. A similar approach was adopted by CLIPasso (Vinker et al., 2022) to translate images into strokes. Vector Fusion (Jain et al., 2023) leveraged Score Distillation Sampling (SDS) (Poole et al., 2022) to induce abstract semantic knowledge from an off-the-shelf Stable Diffusion model (Rombach et al., 2022). A very similar approach, based on SDS, has also been applied to fonts (Iluz et al., 2023). However, all painterly rendering methods come as iterative algorithms making them very computationally expensive and impractical in real world use cases. They frequently create many unnecessary and redundant shapes to minimize the perceptual loss.

# 2.2 Vector Quantization

VQ-VAE (Van Den Oord et al., 2017) is a well-known improved architecture for training Variational Autoencoders (Kingma and Welling, 2013; Rezende et al., 2014). Instead of focusing on representations with continuous features as in most prior work (Vincent et al., 2010; Denton et al., 2016; Hinton and Salakhutdinov, 2006; Chen et al., 2016), the encoder in a VQ-VAE emits discrete rather than continuous codes. Each code maps to the closest embedding in a codebook of limited size. The decoder learns to reconstruct the original input image from the chosen codebook embedding. Both the encoder–decoder architecture and the codebook are trained jointly. After training, the autoregressive distribution over the latent codes is learnt by a second model, which then allows for generating new images via ancestral sampling. Latent discrete representations were already pioneered in previous work (Mnih and Gregor, 2014; Courville et al., 2011), but none of the above methods close the performance gap of VAEs with continuous latent variables, where one can use the Gaussian reparametrization trick, which benefits from much lower variance in the gradients. Mentzer et al. (2023) simplified the design of the vector quantization in VQ-VAE with a scheme called finite scalar quantization (FSQ), where the encoded representation of an image is projected to the nearest position on a low-dimensional hypercube. In this case, no additional codebook must be learned, but rather it is given implicitly, which simplifies the loss formulation. Our work builds in part on the VQ-VAE framework and includes the FSQ mechanism.

# 3 Method

# 3.1 Stage 1 – Visual Shape Quantizer

The first stage of our model employs a Visual Shape Quantizer (VSQ), a vector-quantized autoencoder, whose encoder $E_{VSQ}$ maps an input image I onto a discrete codebook V through vector-quantization and decodes that quantized vector into shape parameters of cubic Bézier curves through

![](images/bc6208e5171282773484f7bae82cd07d09be5e3f429483e169fc30f911b7e656.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Grid with S"] --> B["6x6 Grid"]
    B --> C["Grid with S"]
    C --> D["Patch Creation"]
    D --> E["Raster Patches from Grid"]
    E --> F["Absolute discrete coordinate of patch centers"]
    G["Segment Creation"] --> H["FindContour"]
    H --> I["Segment Creation"]
    I --> J["Locally centered strokes"]
    J --> K["Absolute discrete stroke coordinates"]
```
</details>

Figure 3: Overview of the data generation process for GRIMOIRE. For the MNIST digits, we simply create patches from a $6 \times 6$ Grid. For FIGR-8, we extract the outlines of each icon and create small centered raster segments. We save the original anchor position of each segment for the second stage of our training pipeline. More information about the outline extraction is provided in Section 7.2. Fonts comes in vector format and can be easily manipulated to extract strokes, similarly to FIGR-8.

the decoder $D_{VSQ}$ . Instead of learning the codebook (Van Den Oord et al., 2017), we adopt the more efficient approach of defining our codebook V as a set of equidistant points in a hypercube with q dimensions. Each dimension has l unique values: $L = [l_{1}, l_{2}, \ldots, l_{q}]$ . The size of the codebook $|V|$ is hence defined by the product of values of all q dimensions. We define q = 5 and $L = [7, 5, 5, 5, 5]$ for a target codebook size of 4,375 unique codes, following the recommendations of the original authors (Mentzer et al., 2023).

Before being fed to the encoder $E_{VSQ}$ , each image $I \in R^{C \times H \times W}$ is divided into patches $\mathbf{S} = (s_{1}, s_{2}, \ldots, s_{n})$ , with $s_{i} \in R^{C \times 128 \times 128}$ , where C = 3 is the number of channels. A set of discrete anchor coordinates $\Theta = (\theta_{1}, \theta_{2}, \ldots, \theta_{n})$ with $\theta_{i} \in N^{2}$ being the center coordinate of $s_{i}$ in the original image I is also saved. The original image I can then be reconstructed using S and $\Theta$ .

We experiment on three datasets (see Section 4). For MNIST, the patches are obtained by tiling each image into a $6 \times 6$ grid. For Fonts and FIGR-8, each patch depicts part of the target outline as shown in Figure 3. We utilize a contour-finding algorithm (Lorensen and Cline, 1987) to extract outlines from raster images, which are then divided into several shorter segments. Additional details regarding this extraction process can be found in Section 7.2. In contrast, the Fonts dataset is natively available in vector format, making it easier to manipulate, similar to icons, before undergoing rasterization.

The VSQ encoder $E_{VSQ}$ maps each patch $s_{i} \in R^{C \times 128 \times 128}$ to $\xi$ codes on the hypercube $E_{VSQ}: R^{C \times 128 \times 128} \mapsto V$ as follows. Each centered raster patch $s_{i}$ is encoded with a ResNet-18 (He et al., 2016) into a latent variable $z_{i} \in Z \subset R^{d \times \xi}$ with d = 512. Successively, each of the $\xi$ codes is projected to q dimensions through a linear mapping layer and finally quantized, resulting in $\hat{z}_{i} \in N^{q}$ . The final code value $v_{i} \in V$ is then computed as the weighted sum of all q dimensions of $\hat{z}_{i}$ :

$$
v _ {i} = \sum_ {j = 1} ^ {q} \hat {z} _ {i j} \cdot b _ {j},
$$

where the basis $b_{j}$ is derived as $b_{j} = \prod_{k=1}^{j-1} l_{k}$ , with $b_{1} = 1$ . This transformation ensures that each unique combination of quantized values $\hat{z}_{i}$ is mapped to a unique code $v_{i}$ in the codebook $\mathbb{V}$ .

This approach avoids auxiliary losses on the codebook while maintaining competitive expressiveness.

The decoder $D_{VSQ}$ consists of a projection layer, which transforms all the $\xi$ predicted codes back into the latent space Z, and a lightweight neural network $\Phi_{points}$ , which predicts the control points of $\nu$ cubic Bézier curves that form a single connected path. We propose two variants of $\Phi_{points}$ , a fully-connected neural network $\Phi_{points}^{stroke}:Z\mapsto\mathbb{R}^{(2\times(\nu\times3+1))}$ , which predicts connected strokes, and a 1-D CNN $\Phi_{points}^{shape}:Z\mapsto\mathbb{R}^{(2\times(\nu\times3))}$ , which outputs a closed shape.

Finally, the predicted path of $\nu$ Bézier curves from $\Phi_{points}$ passes through the differentiable rasterizer to obtain a raster output $\hat{s}_{i} = \text{DiffVG}(D_{\text{VSQ}}(E_{\text{VSQ}}(s_{i})))$ . In order to learn to reconstruct strokes and shapes, we train the VSQ module using the mean squared error:

$$
\mathcal {L} _ {\text { recons }} = (s - \hat {s}) ^ {2}.
$$

$D_{VSQ}$ can be extended to predict continuous values for any visual attribute supported by the differentiable rasterizer. Hence, we also propose series of other fully-connected prediction heads that can optionally be enabled: $\Phi_{width}: Z \mapsto R$ predicts the stroke width of the overall shape, and $\Phi_{color}: Z \mapsto R^{C}$ outputs the stroke color or the filling color for the output of $\Phi_{points}^{stroke}$ and $\Phi_{points}^{shape}$ , respectively. All the modules are followed by a sigmoid activation function.

While $L_{recons}$ would suffice for training the VSQ, operating only on the visual domain could lead to degenerate strokes and undesirable local minima. To mitigate this, we propose a novel geometric constraint $L_{geom}$ , which punishes control point placement of irregular distances measured between all combinations of points predicted by $\Phi_{points}^{stroke}$ .

Let $P = (p_{1}, p_{2}, \ldots, p_{\nu + 1})$ be the set of all start and end points of a stroke with $p_{i} = (p_{i}^{x}, p_{i}^{y})$ and $p_{i}^{x}, p_{i}^{y} \in [0, 1]$ . Then $\rho_{i,j}$ is defined as the Euclidean distance between two points $p_{i}$ and $p_{j}$ , $\overline{\rho}_{j}$ is defined as the mean scaled inner distance for point $p_{j}$ to all other points in $P$ , and $\delta_{j}$ as the average squared deviation from that mean for point $p_{j}$ :

$$
\overline{\rho}_{j} = \frac{1}{\nu}\sum_{\substack{i = 1\\ i\neq j}}^{\nu +1}\frac{\rho_{i,j}}{|i - j|}\qquad \delta_{j} = \frac{1}{\nu}\sum_{\substack{i = 1\\ i\neq j}}^{\nu +1}\bigg(\frac{\rho_{i,j}}{|i - j|} -\overline{\rho}_{j}\bigg)^{2}
$$

$L_{geom}$ is finally defined as the average of the deviations for all start and end points in P. $L_{geom}$ is then weighted with $\alpha$ and added to the reconstruction loss.

$$
\mathcal {L} _ {\text { geom }} = \frac {1}{\nu + 1} \sum_ {j = 1} ^ {\nu + 1} \delta_ {j} \quad \mathcal {L} _ {\text { geom }} = \mathcal {L} _ {\text { geom }} + \alpha \times \mathcal {L} _ {\text { geom }}
$$

We use this component only for the experiments with $\Phi_{points}^{stroke}$ and set $\alpha = 0.4$ . We opt to train the ResNet encoder from scratch during this stage, since the target images belong to a very specific domain. The amount of trainable parameters is 15.36M for the encoder and 0.8M for the decoder. We stress the importance of the skewed balance between the two parameter counts, as the encoding of images is only required for training the model and encoding the training data for the auto-regressive Transformer in the next step. The final inference pipeline discards the encoder and only requires the trained decoder $D_{VSQ}$ , hence resulting in more lightweight inference. The overall scheme of GRIMOIRE including the first stage of training is depicted in Figure 2.

# 3.2 Stage 2 - Auto-Regressive Transformer

After the VSQ is trained, each patch $s_{i}$ can be mapped onto an index code $v_{i}$ of the codebook V using the encoder $E_{VSQ}$ and the quantization method. However, the predicted patch $\hat{s}_{i}$ captured by the VSQ does not describe a complete SVG, as the centering leads to a loss of information about their global position $\theta_{i}$ on the original canvas. Also, the sequence of tokens is still missing the text conditioning. This is addressed in the second stage of GRIMOIRE. The second stage consists of an Auto-Regressive Transformer (ART) that learns for each image I the joint distribution over the text, positions, and stroke tokens. A textual description T of I is tokenized into $\mathcal{T} = (\tau_{1}, \tau_{2}, \ldots, \tau_{t})$ using a pre-trained BERT encoder (Devlin et al., 2018) and embedded. I is visually encoded by transforming its patches $s_{i}$ onto $v_{i} \in V$ via the encoder $E_{VSQ}$ , whereas each original patch position $\theta_{i} \in \Theta$ is mapped into the closest position in a $256 \times 256$ grid resulting in $256^{2}$ possible position tokens. Special tokens <SOS>, <BOS>, and <EOS> indicate the start of a full sequence, beginning of the patch token sequence, and end of sequence, respectively. Each patch token is alternated with its position token. The final input sequence for a given image to the ART module becomes:

$$
x = \left(<   \mathrm{SOS} >, \tau_ {1}, \dots , \tau_ {t}, <   \mathrm{BOS} >, \theta_ {1}, v _ {1}, \dots \theta_ {n}, v _ {n}, <   \mathrm{EOS} >\right)
$$

The total amount of representable token values then has a dimensionality of $|V| + 256^{2} + 3 = 69, 914$ for $|V| = 4,375$ . A learnable weight matrix $W \in R^{d \times 69,914}$ embeds the position and visual tokens into a vector of size d. The BERT text embeddings are projected into the same d-dimensional space using a trainable linear mapping layer. The ART module consists of 12 and 16 standard Transformer decoder blocks with causal multi-head attention with 8 attention heads for fonts and icons, respectively. The final loss for the ART module is defined as:

$$
\mathcal {L} _ {\text { Causal }} = - \sum_ {i = 1} ^ {N} \log p (x _ {i} \mid x _ {<   i}; \theta)
$$

During inference, the input to the ART module is represented as $x = (\langle\mathrm{SOS}\rangle, \tau_{1}, \ldots, \tau_{t}, \langle\mathrm{BOS}\rangle)$ , where new tokens are predicted auto-regressively until the $\langle\mathrm{EOS}\rangle$ token is generated. Additionally, visual strokes can be incorporated into the input sequence to condition the generation process.

# 4 Data

MNIST. We conduct our initial experiments on the MNIST dataset (LeCun et al., 1998). We upscale each digit to $128 \times 128$ pixels and generate the textual description using the prompt “x in black color”, where x is the class of each digit. We adopt the original train and test split.

Fonts. For our experiments on fonts, we use a subset of the SVG-Fonts dataset (Lopes et al., 2019). We remove fonts where capital and lowercase glyphs are identical, and consider only 0–9, a–z, and A–Z glyphs, which leads to 32,961 unique fonts for a corpus of $\sim$ 2M samples. The font features – such as type of character or style – are extracted from the .TTF file metadata. The final textual description for a sample glyph g in font style s is built using the prompt: “[capital] g in s font”, where “capital” is included only for the glyphs A-Z. We use 80%, 10%, and 10% for training, testing, and validation respectively.

FIGR-8. We validate our method on more complex data and further use a subset of FIGR-8 (Clouâtre and Demers, 2019), where we select the 75 majority classes (excluding “arrow”) and any class that contains those, e.g., the selection of “house” further entails the inclusion of “dog house”. This procedure yields 427K samples, of which we select 90% for training, 5% for validation, and 5% for testing. We use the class names as textual descriptions without further processing besides minor spelling correction. Since the black strokes of FIGR-8 mark the background rather than the actual icon, we invert the full dataset before applying our additional pre-processing described in Section 7.2.

# 5 Results

This section presents our findings in two primary categories. First, we examine the quality of the reconstructions and generations produced by GRIMOIRE in comparison to existing methods. Second, we highlight the flexibility of our approach, demonstrating how GRIMOIRE can be easily extended to incorporate additional SVG features.

# 5.1 Reconstructions

Closed Paths. We begin by presenting the reconstruction results of our VSQ module on the MNIST dataset. In our experiments, we model each patch shape using a total of 15 segments. Increasing the number of segments beyond this point did not yield any significant improvement in reconstruction quality. Given the simplicity of the target shapes, we adopted a single code per shape.

We also conducted a comparative analysis of the reconstruction capabilities of our VSQ module against Im2Vec. To assess the generative quality of our samples, we employed the Fréchet Inception Distance (FID) (Heusel et al., 2017) and CLIPScore (Radford et al., 2021), both of which are computed using the image features of a pre-trained CLIP encoder. Additionally, to validate our VSQ module, we considered the reconstruction loss $L_{recons}$ , as it directly reflects the maximum achievable performance of the network and provides a more reliable metric.

As shown in Table 1, our VSQ module consistently achieves a lower reconstruction error compared to Im2Vec across all MNIST digits. In Table 2, we also report the reconstruction error for a subset of the dataset, selecting the digit zero due to its particularly challenging topology. Again, our method exhibits superior performance with lower reconstruction errors. For MNIST, we fill the predicted shapes from Im2Vec, since the raster ground truth images are only in a filled format. However, we present both filled and unfilled versions for all other scenarios.

The CLIPScore of our reconstructions is higher in both cases. Notably, FID is the only metric where Im2Vec occasionally shows superior results. We attribute this to the lower resolution of the ground truth images, which introduces instability in the FID metric. The CLIPScore, however, mitigates this issue by comparing the similarity with the textual description.

Table 1: Results for reconstructions of GRIMOIRE and Im2Vec on the full datasets. The last row includes post-processing. 

<table><tr><td rowspan="2">Model</td><td colspan="3">MNIST</td><td colspan="3">Fonts</td><td colspan="3">FIGR-8</td></tr><tr><td>MSE ↓↓</td><td>FID ↓↓</td><td>CLIP ↑↑</td><td>MSE ↓↓</td><td>FID ↓↓</td><td>CLIP ↑↑</td><td>MSE ↓↓</td><td>FID ↓↓</td><td>CLIP ↑↑</td></tr><tr><td>Im2Vec (filled)</td><td>0.140</td><td>1.33</td><td>25.02</td><td>0.140</td><td>2.04</td><td>26.82</td><td>0.330</td><td>16.10</td><td>26.17</td></tr><tr><td>Im2Vec</td><td>n/a</td><td>n/a</td><td>n/a</td><td>0.050</td><td>5.64</td><td>26.72</td><td>0.050</td><td>13.90</td><td>26.17</td></tr><tr><td>VSQ</td><td>0.090</td><td>7.09</td><td>25.24</td><td>0.014</td><td>4.45</td><td>28.61</td><td>0.004</td><td>1.42</td><td>31.09</td></tr><tr><td>VSQ + PI</td><td>n/a</td><td>n/a</td><td>n/a</td><td>0.011</td><td>0.29</td><td>28.96</td><td>0.002</td><td>0.05</td><td>32.03</td></tr></table>

Strokes. For Fonts and FIGR-8, we conduct a deeper investigation to validate the reconstruction errors of VSQ under different configurations, varying the amount of segments and codes per shape, and the maximum length of the input strokes. Our findings show that for Fonts, more than one segment per shape consistently degrades the reconstruction quality, possibly because the complexity of the strokes in our datasets does not require many Beziér curves to reconstruct an input patch. We also find that shorter thresholds on the stroke length help the reconstruction quality, as the MSE decreases when moving from 11% to 7% and eventually to 4% of the maximum stroke length with respect to the image size. Intuitively, shorter strokes are easier to model, but could also lead to very scattered predictions for overly short settings.

The best reconstructions are achieved by using multiple codes per centered stroke. The two-codes configuration has an average decrease in MSE of 18.28%, 41.46%, and 26.09% for the respective stroke lengths. However, the best-performing configuration with two codes per shape is just 11.36% better than the best single code representative, which we believe does not justify twice the number of required visual tokens for the second stage training. Throughout our experiments, the configurations with multiple segments do consistently benefit from our geometric constraint. Ultimately, for our final experiments we choose $(\nu = 2, \xi = 1)$ for Fonts, and $(\nu = 4, \xi = 2)$ for FIGR-8.

Regarding the comparison with Im2Vec, Table 2 shows that the text-conditioned GRIMOIRE on a single glyph or icon has superior reconstruction performance even if Im2Vec is specifically trained on that subset of data. In Table 1, we also report the values after training on the full datasets. In this case, GRIMOIRE substantially outperforms Im2Vec, which is unable to cope with the complexity of the data.

Finally, as GRIMOIRE quickly learns to map basic strokes or shapes onto its finite codebook and due to the similarities between those primitive traits among various samples in the dataset, we find GRIMOIRE to converge even before completing a full epoch on any dataset. Despite the reconstruction error being considerably higher, we also notice reasonable domain transfer capabilities between FIGR-8 images and Fonts when training the VSQ module only on one dataset and keeping the maximum stroke length consistent. Qualitative examples of the re-usability of the VSQ module are reported in the Appendix.

Table 2: Results for reconstructions of GRIMOIRE and Im2Vec on subsets. The last row includes post-processing. 

<table><tr><td rowspan="2">Model</td><td colspan="3">MNIST (0)</td><td colspan="3">Fonts (A)</td><td colspan="3">Icons (Star)</td></tr><tr><td>MSE ↓↓</td><td>FID ↓↓</td><td>CLIP ↑↑</td><td>MSE ↓↓</td><td>FID ↓↓</td><td>CLIP ↑↑</td><td>MSE ↓↓</td><td>FID ↓↓</td><td>CLIP ↑↑</td></tr><tr><td>Im2Vec (filled)</td><td>0.218</td><td>2.20</td><td>24.61</td><td>0.087</td><td>1.64</td><td>26.27</td><td>0.120</td><td>2.40</td><td>30.90</td></tr><tr><td>Im2Vec</td><td>n/a</td><td>n/a</td><td>n/a</td><td>0.060</td><td>6.33</td><td>25.78</td><td>0.110</td><td>11.17</td><td>30.40</td></tr><tr><td>VSQ</td><td>0.130</td><td>11.2</td><td>26.68</td><td>0.020</td><td>4.50</td><td>29.13</td><td>0.002</td><td>1.26</td><td>31.64</td></tr><tr><td>VSQ + PI</td><td>n/a</td><td>n/a</td><td>n/a</td><td>0.012</td><td>0.61</td><td>29.46</td><td>0.001</td><td>0.07</td><td>32.94</td></tr></table>

# 5.2 Generations

Text Conditioning. We compare GRIMOIRE with Im2Vec by generating glyphs and icons and handwritten digits, and report the results in Table 3. Despite Im2Vec being tailored for single classes only, our general model shows superior performance in CLIPScore for all datasets. Im2Vec shows a generally lower FID score in the experiments with filled shapes, which we attribute again to the lower resolution of the ground truth images (MNIST) and a bias in the metric itself as CLIP struggles to produces meaningful visual embeddings for sparse images (Chowdhury et al., 2022) as for Fonts, FIGR-8. In contrast, in the generative results on unfilled shapes, GRIMOIRE almost consistently outperforms Im2Vec by a large margin for glyphs and icons.

Note that we establish new baseline results for the complete datasets, as Im2Vec does not support text or class conditioning.

Looking at qualitative samples in Figure 4 and Figure 1, one can see that contrary to the claim that surplus shapes collapse to a point (Reddy et al., 2021), there are multiple redundant shapes present in the generations of Im2Vec. A single star might then be represented by ten overlapping almost identical paths. The qualitative results in Figure 5 confirm this behaviour on the MNIST dataset. We

also show that setting Im2Vec to predict only one single SVG path leads the model to compress the shape area and use its filling as a stroke width.

Overall, GRIMOIRE produces much cleaner samples with less redundancy, which makes them easier to edit and visually more pleasing. The text conditioning also allows for more flexibility. The generations are also diverse, as can be seen in Figure 4 where we showcase multiple generations for the same classes from FIGR-8. Additional generations on all datasets are provided in the Appendix.

![](images/316ba17c9e6a501957c439953cc61a9fbb50c178cb8cab62df156d0659d6e029.jpg)

<details>
<summary>text_image</summary>

Phone
Heart
Light bulb
Arrow
User
Home
Settings
</details>

Figure 4: Examples of text-conditioned icon generation from GRIMOIRE.

<table><tr><td rowspan="2">Generations</td><td colspan="3">GRIMOIRE</td><td colspan="3">Im2Vec (One path)</td><td colspan="3">Im2Vec (Ten Paths)</td></tr><tr><td><img src="images/1cf10fb70904700e51aba9f9e46407804f6487fe8a6f17e571d16d0152bc1f01.jpg"/></td><td><img src="images/b8e13f65769c217f1f0ac27006ff8c7fa43c6b5caed3fb19314630b08a8c049f.jpg"/></td><td><img src="images/9df701e6bd33e77c309468a10fb41d26dfbe6a1f4e441bca01b6641805f8b805.jpg"/></td><td><img src="images/623dac5ab2cc81f828807e7900f31289e5d264bcacdc17fbcf312c6083234c8e.jpg"/></td><td><img src="images/85df11a52ba4a80429d8adf7801f38fea532e6204bee463ee40b7c71f0f0f7f4.jpg"/></td><td>[75/78]</td><td><img src="images/bf9e65da8858d7dc3ad4fedafed544772829605fed90f4fc55641dca68eb7073.jpg"/></td><td><img src="images/afe1d0fd722b8744e2a0e57f94df5a5da0fb34a316ef2ae3eb0165637982b4a4.jpg"/></td><td><img src="images/9285949d877b54f876f8c7a5b939c87bdc91842b848407e63bdcf5fd0dd28e3c.jpg"/></td></tr></table>

Figure 5: Generative results for the MNIST dataset from GRIMOIRE and Im2Vec with the number of predicted paths fixed to one and ten respectively. Since Im2Vec does not accept any conditioning, we sample after training Im2Vec only on the digit Zero. For GRIMOIRE, we use the models trained on the full dataset conditioned on the respective class.

Vector Conditioning. We also evaluate GRIMOIRE on another task previously unavailable for image-supervised vector graphic generative models, which is text-guided icon completion. Figure 6 shows the capability of our model to complete an unseen icon, based on a set of given context strokes that start at random positions. GRIMOIRE can meaningfully complete various amounts of contexts, even when the strokes of the context stem from disconnected parts of the icon. We provide a quantitative analysis in Section 7.8. The results in this section are all obtained with the default pipeline that post-processes the generation of our model. A detailed analysis of our post-processing is provided in Section 7.3 and Section 7.4.

Table 3: Results for generations of GRIMOIRE and Im2Vec. GRIMOIRE is trained on the full dataset and conditioned to the respective classes using the text description. 

<table><tr><td rowspan="2">Model</td><td colspan="2">MNIST (0)</td><td colspan="2">MNIST (Full)</td><td colspan="2">Fonts (A)</td><td colspan="2">Fonts (Full)</td><td colspan="2">FIGR-8(Star)</td><td colspan="2">FIGR-8(Full)</td></tr><tr><td>FID ↓↓</td><td>CLIP ↑↑</td><td>FID ↓↓</td><td>CLIP ↑↑</td><td>FID ↓↓</td><td>CLIP ↑↑</td><td>FID ↓↓</td><td>CLIP ↑↑</td><td>FID ↓↓</td><td>CLIP ↑↑</td><td>FID ↓↓</td><td>CLIP ↑↑</td></tr><tr><td>Im2Vec (filled)</td><td>2.22</td><td>24.69</td><td>n/a</td><td>n/a</td><td>1.20</td><td>25.81</td><td>n/a</td><td>n/a</td><td>2.97</td><td>31.72</td><td>n/a</td><td>n/a</td></tr><tr><td>Im2Vec</td><td>n/a</td><td>25.21</td><td>n/a</td><td>n/a</td><td>5.36</td><td>25.39</td><td>n/a</td><td>n/a</td><td>11.59</td><td>31.88</td><td>n/a</td><td>n/a</td></tr><tr><td>GRIMOIRE (ours)</td><td>12.25</td><td>26.60</td><td>9.25</td><td>25.25</td><td>5.61</td><td>30.60</td><td>1.67</td><td>28.64</td><td>6.25</td><td>32.24</td><td>3.58</td><td>27.45</td></tr></table>

![](images/a166ce828cdc49a10608a1e935b50af56350597b0120b9380ffa1da824655827.jpg)  
Figure 6: Different completions with varying number of context segments $\nu_{context}$ (marked in red). GRIMOIRE can meaningfully complete irregular starting positions of the context strokes.

# 5.3 Flexibility

Finally, we demonstrate the flexibility of GRIMOIRE through additional qualitative results on new SVG attributes. One of the advantages of splitting the generative pipeline into two parts is that the ART module can be fully decoupled from the visual attributes of the SVG primitives. Instead, the vector prediction head of the VSQ can be extended to include any visual attribute supported by the differentiable rasterizer. Specifically, we activate the prediction heads $\Phi_{width}$ and $\Phi_{color}$ —outlined in Section 3.1— to enable learning of stroke width and color, respectively. We train the VSQ module on input patches while varying the values of those attributes and present the qualitative outcomes in Figure 7, where each stroke is randomly colored using an eight-color palette and a variable stroke width. The VSQ module accurately learns these features without requiring altering the size of the codebook or modifying any other network configurations.

A similar analysis is conducted with closed shapes, and the results are reported in Figure 8, showing that the VSQ module jointly maps both shape and color to a single code. This highlights the minimal requirements of GRIMOIRE in supporting additional SVG features. In contrast, other state-of-the-art vector-based generative models often rely on complex tokenization pipelines, making the extension to new SVG attributes more cumbersome and less flexible.

<table><tr><td>-</td><td>C</td><td>\</td><td>C</td><td>-</td><td>\</td><td>-</td><td>\</td><td>-</td><td>-</td></tr><tr><td>-</td><td>C</td><td>\</td><td>C</td><td>-</td><td>\</td><td>-</td><td>\</td><td>-</td><td>-</td></tr></table>

Figure 7: Inputs (top) and corresponding reconstructions (bottom) generated by a VSQ model trained to predict not only the shape but also the visual attributes of the input strokes, such as color and stroke width.

# 6 Conclusion

This work presents GRIMOIRE, a novel framework for generating and completing complex SVGs, trained solely on raster images. GRIMOIRE improves existing raster-supervised SVG generative networks in output quality, while offering significantly greater flexibility through text-conditioned generation. We validate GRIMOIRE on filled shapes using a simple tile-patching strategy to create the input data, and on strokes using fonts and icons datasets. Our results demonstrate the superior performance of GRIMOIRE compared to existing models, even when adapted to specific image classes.

![](images/e9ac0eb83dce41714b793f737341fa05822b537da2d790c6b483c8027f636091.jpg)

<details>
<summary>text_image</summary>

Orig VSQ
2 2
Two in royal blue Four in purple Six in in teal
Orig VSQ
4 4
Orig VSQ
6 6
</details>

Figure 8: Reconstruction of MNIST digits when the VSQ module also predicts the filling color. The left side shows the tiling of the original raster images, the right side reports the reconstructions from the VSQ module. No post-processing is applied.

Additionally, we show that GRIMOIRE can be seamlessly extended to support new SVG attributes when included in the training data.

Future work could explore incorporating additional vector primitives, expanding visual features, or employing a hierarchical approach to patch extraction.

# References

H. Aoki and K. Aizawa. Svg vector font generation for chinese characters with transformer. In 2022 IEEE International Conference on Image Processing (ICIP), pages 646–650. IEEE, 2022.   
A. Carlier, M. Danelljan, A. Alahi, and R. Timofte. Deepsvg: A hierarchical generative network for vector graphics animation. Advances in Neural Information Processing Systems, 33:16351–16361, 2020.   
X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel. Infogan: Interpretable representation learning by information maximizing generative adversarial nets. Advances in neural information processing systems, 29, 2016.   
P. N. Chowdhury, A. Sain, A. K. Bhunia, T. Xiang, Y. Gryaditskaya, and Y.-Z. Song. Fs-coco: Towards understanding of freehand sketches of common objects in context. In European Conference on Computer Vision, pages 253–270. Springer, 2022.   
L. Clouâtre and M. Demers. Figr: Few-shot image generation with reptile. arXiv preprint arXiv:1901.02199, 2019.   
A. Courville, J. Bergstra, and Y. Bengio. A spike and slab restricted boltzmann machine. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 233–241. JMLR Workshop and Conference Proceedings, 2011.   
E. Denton, S. Gross, and R. Fergus. Semi-supervised learning with context-conditional generative adversarial networks. arXiv preprint arXiv:1611.06430, 2016.   
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.   
K. Frans, L. Soros, and O. Witkowski. Clipdraw: Exploring text-to-drawing synthesis through language-image encoders. Advances in Neural Information Processing Systems, 35:5207–5218, 2022.   
Y. Ganin, T. Kulkarni, I. Babuschkin, S. A. Eslami, and O. Vinyals. Synthesizing programs for images using reinforced adversarial learning. In International Conference on Machine Learning, pages 1666–1675. PMLR, 2018.   
I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.   
D. Ha and D. Eck. A neural representation of sketch drawings. arXiv preprint arXiv:1704.03477, 2017.

K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.   
M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
G. E. Hinton and R. R. Salakhutdinov. Reducing the dimensionality of data with neural networks. science, 313(5786):504–507, 2006.   
J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
S. Iluz, Y. Vinker, A. Hertz, D. Berio, D. Cohen-Or, and A. Shamir. Word-as-image for semantic typography. arXiv preprint arXiv:2303.01818, 2023.   
P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.   
A. Jain, A. Xie, and P. Abbeel. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1911–1920, 2023.   
D. P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.   
T.-M. Li, M. Lukáč, M. Gharbi, and J. Ragan-Kelley. Differentiable vector graphics rasterization for editing and learning. ACM Transactions on Graphics (TOG), 39(6):1–15, 2020.   
R. G. Lopes, D. Ha, D. Eck, and J. Shlens. A learned representation for scalable vector graphics. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7930–7939, 2019.   
W. E. Lorensen and H. E. Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Proceedings of the 14th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH '87, page 163–169, New York, NY, USA, 1987. Association for Computing Machinery. ISBN 0897912276. doi: 10.1145/37401.37422. URL https://doi.org/10.1145/37401.37422.   
I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
F. Mentzer, D. Minnen, E. Agustsson, and M. Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023.   
A. Mnih and K. Gregor. Neural variational inference and learning in belief networks. In International Conference on Machine Learning, pages 1791–1799. PMLR, 2014.   
R. Nakano. Neural painters: A learned differentiable constraint for generating brushstroke paintings. arXiv preprint arXiv:1904.08410, 2019.   
A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.   
B. Poole, A. Jain, J. T. Barron, and B. Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.   
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

P. Reddy, M. Gharbi, M. Lukac, and N. J. Mitra. Im2vec: Synthesizing vector graphics without vector supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7342–7351, 2021.   
D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International conference on machine learning, pages 1278–1286. PMLR, 2014.   
R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.   
Z. Tang, C. Wu, Z. Zhang, M. Ni, S. Yin, Y. Liu, Z. Yang, L. Wang, Z. Liu, J. Li, et al. Strokenuwa: Tokenizing strokes for vector graphic synthesis. arXiv preprint arXiv:2401.17093, 2024.   
A. Van Den Oord, O. Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.   
P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, P.-A. Manzagol, and L. Bottou. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of machine learning research, 11(12), 2010.   
Y. Vinker, E. Pajouheshgar, J. Y. Bo, R. C. Bachmann, A. H. Bermano, D. Cohen-Or, A. Zamir, and A. Shamir. Clipasso: Semantically-aware object sketching. ACM Transactions on Graphics (TOG), 41(4):1–11, 2022.   
R. Wu, W. Su, K. Ma, and J. Liao. Iconshop: Text-guided vector icon synthesis with autoregressive transformers. ACM Transactions on Graphics (TOG), 42(6):1–14, 2023.

# 7 Appendix

# 7.1 Glossary of Notation

Due to the number of notation used in this work, in Table 4 we have reported a recap of the most important with a brief description of their meaning.

Table 4: Glossary of relevant notations in this work. 

<table><tr><td>Notation</td><td>Meaning</td></tr><tr><td>E</td><td>Network encoder</td></tr><tr><td>D</td><td>Network decoder</td></tr><tr><td>I</td><td>Image from the dataset</td></tr><tr><td>V</td><td>Codebook</td></tr><tr><td>v</td><td>Codes from the codebook</td></tr><tr><td>L</td><td>Set of values per dimension of our codebook</td></tr><tr><td>l</td><td>Single dimensional value</td></tr><tr><td>q</td><td>Number of dimensions of the codebook</td></tr><tr><td>S</td><td>Series of patches</td></tr><tr><td>s</td><td>Single patch</td></tr><tr><td>C</td><td>Color channels</td></tr><tr><td>n</td><td>Number of patches</td></tr><tr><td>Θ</td><td>Set of discrete coordinates</td></tr><tr><td>θ</td><td>Single coordinate pair</td></tr><tr><td>Z</td><td>Latent space</td></tr><tr><td> $\hat{z}$ </td><td>Projected embedding</td></tr><tr><td>d</td><td>Dimension of latent space</td></tr><tr><td>z</td><td>Latent embedding</td></tr><tr><td> $\hat{s}$ </td><td>Reconstructed patch</td></tr><tr><td>ν</td><td>Number of segments</td></tr><tr><td>P</td><td>Set of points</td></tr><tr><td>p</td><td>Point pair</td></tr><tr><td>ρ</td><td>Euclidian distance between two points</td></tr><tr><td>Φ</td><td>Generic Neural network</td></tr><tr><td>ξ</td><td>Number of codebook codes</td></tr><tr><td>T</td><td>Text description</td></tr><tr><td>T</td><td>Tokenized description</td></tr><tr><td>τ</td><td>Text token</td></tr><tr><td>t</td><td>Number of text tokens</td></tr></table>

# 7.2 Pre-Processing

This section provides additional information regarding the pre-processing and extraction techniques on the employed datasets.

Shapes. No pre-processing is conducted for the MNIST dataset. Images are simply tiled using a $6 \times 6$ grid and the central position of each tile in the original image is saved.

Strokes. For the FIGR-8 dataset, the pixels outlining the icons are isolated using a contour finding algorithm (Lorensen and Cline, 1987) and the coordinates are then used to convert them into vector paths. This simple procedure available in our code repository allows us to efficiently apply a standard pre-processing pipeline defined in Carlier et al. (2020) and already adopted by other studies (Wu et al., 2023; Tang et al., 2024). The process involves normalizing all strokes and breaking them into shorter units if their length exceeds a certain maximum percentage of the image size. Finally, each resulting path fragment is scaled, translated to the center of a new canvas s by placing the center of its bounding box onto the center of s, and rasterized to become part of the training data. Since strokes in S are all translated around the image center, the original center position $\theta$ of the bounding box in I is recorded for each s and saved. These coordinates are discretized in a range of $256 \times 256$ values. This approach is also used for Fonts, but since the data comes in vector format, there is no need for contour finding.

# 7.3 Post-Processing

Our approach introduces small discrepancies with the ground truth data during tokenization. The VSQ introduces small inaccuracies in the reconstruction of the stroke, and the discretization of the global center positions may slightly displace said strokes. The latter serve as the training data for the auto-regressive Transformer and therefore represent an upper limit to the final generation quality. Similarly for MNIST, the use of white padding on each patch to facilitate faster convergence results in small background gaps when rendering all shapes together, as shown in Figure 5. These small errors compound for the full final image and may become fairly visible in the reconstructions.

![](images/6ecc064c023514662ee64ee16aadd29ba930a627e68ccc96c8dd97463055735e.jpg)

![](images/8ff2f47650c1123cce393667001295cfdbceaa9dbff3466d46da78493464cf6e.jpg)

![](images/ae1cf427eb197ae2d894e27abe673cd52bdbd72c6731460ebdd43e06b2a71ca5.jpg)

![](images/5a35686c1cf0c0955c9472df0624f5024032c7784c08deb16758caf7f27a67ab.jpg)

![](images/a67dc88c20459e8b0f1d281e8734570a47c8d7316951005d152261eb6d573489.jpg)  
Figure 9: Different SVG post-processing methods visualized. From left to right: raw generation, results of applying PC and PI, results of applying PC and PI by only considering nearest neighbors of consecutive strokes.

While we opted not to modify the global reconstructions of MNIST generation, for FIGR-8 and Fonts, we make use of SVG post-processing similar to prior work (Tang et al., 2024), which introduced Path Clipping (PC) and Path Interpolations (PI). In PC, the beginning of a stroke is set to the position of the end of the previous stroke. In PI, a new stroke is added that connects them instead. As we operate on visual supervision, the ordering of the start and end point of a stroke is not consistent. Hence, we adapt these two methods to not consider the start and end point, but rather consider the nearest neighbors of consecutive strokes. We also add a maximum distance parameter to the post-processing in order to avoid intentionally disconnected strokes to get connected. See Figure 9, Figure 10 for a qualitative depiction of this process and Section 7.4 for a quantitative comparison.

<table><tr><td></td><td>capital i in regular font</td><td>c in regular font</td><td>p in italic font</td><td>capital 1 in regular font</td><td>2 in italic font</td><td>4 in normal font</td></tr><tr><td>Unfixed Pred.</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PI Fixing</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PC Fixing</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Figure 10: Some examples of text-conditioned glyph generation from GRIMOIRE. The first row shows the unfixed model predictions, the second and third rows depict the final outputs with two different post-processing techniques.

# 7.4 Results with different Post-processing

In GRIMOIRE, the resulting full vector graphic generation is characterized by fragmented segments. This is because the output strokes of the VSQ decoder are each locally centered onto a separate canvas,

and the auto-regressive Transformer, which is responsible for the absolute position of each shape, returns only the center coordinates of the predicted shape without controlling the state of connection between different strokes. To cope with this, in Section 7.3, we introduced several post-processing algorithms. In this section, we report additional information about the performance of each of them for the VSQ module (reconstruction) and the overall GRIMOIRE (generation). Table 5 shows that the PC technique consistently outperforms the alternatives across both datasets in terms of both FID and CLIPScore.

Table 5: Reconstruction capabilities of our VSQ module and generative performance of GRIMOIRE with different post-processing techniques after training on Fonts and FIGR-8. 

<table><tr><td rowspan="2">Model</td><td colspan="3">Fonts</td><td colspan="3">FIGR-8</td></tr><tr><td>MSE</td><td>FID</td><td>CLIP</td><td>MSE</td><td>FID</td><td>CLIP</td></tr><tr><td>VSQ</td><td>0.0144</td><td>4.45</td><td>28.61</td><td>0.0045</td><td>1.29</td><td>31.17</td></tr><tr><td>VSQ (+PC)</td><td>0.0135</td><td>0.23</td><td>29.24</td><td>0.0023</td><td>0.10</td><td>31.97</td></tr><tr><td>VSQ (+PI)</td><td>0.0106</td><td>0.29</td><td>28.96</td><td>0.0028</td><td>0.07</td><td>32.0</td></tr><tr><td>GRIMOIRE</td><td>n/a</td><td>4.44</td><td>28.45</td><td>n/a</td><td>4.20</td><td>26.96</td></tr><tr><td>GRIMOIRE (+PC)</td><td>n/a</td><td>1.67</td><td>28.64</td><td>n/a</td><td>3.58</td><td>27.45</td></tr><tr><td>GRIMOIRE (+PI)</td><td>n/a</td><td>1.86</td><td>28.43</td><td>n/a</td><td>4.57</td><td>26.73</td></tr></table>

# 7.5 Im2Vec on Other Classes

We conducted a more in-depth analysis of the generative capabilities in Im2Vec after training on single subsets of FIGR-8, and compare the results with GRIMOIRE. We trained Im2Vec on the top-10 classes of FIGR-8: Camera (8,818 samples), Home (7,837), User (7,480), Book (7,163), Clock (6,823), Flower (6,698), Star (6,681), Calendar (misspelt as caledar in the dataset, 6,230), and Document (6,221). Table 6 compares the FID and CLIPScore with GRIMOIRE. Note that we train our model only once on the full FIGR-8 dataset and validate the generative performance using text-conditioning on the target class, whereas Im2Vec is unable to handle training on such diverse data. Despite Im2Vec appearing to obtain higher scores on several classes such as User or Document, a qualitative inspection reveals how the majority of the generated samples come in the form of meaningless filled blobs or rectangles. The traditional metrics employed in this particular generative field, based on the pre-trained CLIP model, react very strongly to such shapes in contrast to more defined stroke images. We refer reviewers to the qualitative samples in Figure 20. We further observe a low variance in the generations when Im2Vec learns the representations of certain classes, such as star icons.

Table 6: Generative results for GRIMOIRE and Im2Vec for the top-10 classes in FIGR-8. 

<table><tr><td rowspan="2">Model</td><td colspan="2">camera</td><td colspan="2">home</td><td colspan="2">user</td><td colspan="2">book</td><td colspan="2">clock</td><td colspan="2">cloud</td><td colspan="2">flower</td><td colspan="2">calendar</td><td colspan="2">document</td></tr><tr><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td></tr><tr><td>Im2Vec (filled)</td><td>9.21</td><td>27.86</td><td>3.48</td><td>26.85</td><td>2.12</td><td>28.92</td><td>7.18</td><td>27.26</td><td>6.12</td><td>26.38</td><td>17.43</td><td>24.38</td><td>6.61</td><td>25.42</td><td>4.5</td><td>27.26</td><td>12.19</td><td>28.65</td></tr><tr><td>Im2Vec</td><td>9.05</td><td>27.18</td><td>9.19</td><td>25.95</td><td>6.33</td><td>27.01</td><td>8.63</td><td>25.84</td><td>5.09</td><td>25.69</td><td>25.58</td><td>24.38</td><td>6.8</td><td>23.34</td><td>6.61</td><td>26.22</td><td>16.62</td><td>26.71</td></tr><tr><td>GRIMOIRE</td><td>6.74</td><td>29.81</td><td>7.16</td><td>27.16</td><td>5.45</td><td>26.81</td><td>6.65</td><td>27.1</td><td>7.22</td><td>26.32</td><td>6.78</td><td>24.96</td><td>10.27</td><td>22.00</td><td>5.57</td><td>26.23</td><td>4.08</td><td>27.96</td></tr><tr><td>GRIMOIRE (+PC)</td><td>5.77</td><td>30.22</td><td>7.6</td><td>27.41</td><td>4.38</td><td>27.18</td><td>5.8</td><td>27.24</td><td>6.79</td><td>26.45</td><td>6.05</td><td>25.51</td><td>9.37</td><td>22.46</td><td>5.09</td><td>26.41</td><td>3.81</td><td>28.21</td></tr><tr><td>GRIMOIRE (+PI)</td><td>7.5</td><td>29.46</td><td>7.44</td><td>27.01</td><td>5.95</td><td>26.85</td><td>6.79</td><td>27.08</td><td>7.63</td><td>26.12</td><td>7.09</td><td>24.73</td><td>9.97</td><td>22.04</td><td>5.87</td><td>25.98</td><td>4.21</td><td>27.89</td></tr></table>

# 7.6 Qualitative Results of the Geometric Loss

The adoption of our geometric constraint improves the overall reconstruction error, which we attribute to the network being encouraged to elongate the stroke as much as possible. The results in Figure 11 show the effects on the control points of the reconstructed strokes from the VSQ. With the geometric constraint, the incentive to stretch the stroke works against the MSE objective, which results in an overall longer stroke and therefore in greater connectedness in a full reconstruction and an overall lower reconstruction error. We also present an example with an excessively high geometric constraint weight ( $\alpha = 5$ ) demonstrating that beyond a certain threshold, the positive effect diminishes, resulting in degenerated strokes.

$$
\begin{array}{c c} \searrow & \swarrow \\ \hline \swarrow & \nearrow \end{array}
$$

Ground Truth   
![](images/8daf8a28e0428b291e3237ff520fa520f7d217032cd055c260c335935dea25ca.jpg)

<details>
<summary>natural_image</summary>

Four-panel grid showing four black curved objects with red dots, arranged in a 2x2 grid (no text or symbols)
</details>

$\alpha = 0$

![](images/a874ad2f2457835dbe6324526dc1f409937ada8aad1a17e0c8b70ccfec662ff9.jpg)

<details>
<summary>natural_image</summary>

Four quadrants displaying dark, curved, and dotted shapes resembling abstract or microscopic patterns (no text or symbols)
</details>

$\alpha = 0.1$

![](images/4cbb9c95d1915c2f10eb3553d3f3022c51f8460651417af6649420f32e269c47.jpg)

<details>
<summary>natural_image</summary>

Four quadrants displaying abstract black shapes resembling stylized characters or icons, with no text or symbols present.
</details>

$\alpha = 5$   
Figure 11: Samples from the test set when training the VSQ module with and without our geometric constraint. Each stroke consists of two cubic Bézier segments. Embedded within each stroke, the red dots mark the start and end points, while the green and blue dot pairs are the control points of each segment.

Table 7: Generation quality of GRIMOIRE with different lengths of provided context on Fonts and FIGR-8. Post-processing is conducted for all setups. GRIMOIRE uses textual input for all generations. 

<table><tr><td rowspan="2">Model</td><td colspan="2">Fonts</td><td colspan="2">FIGR-8</td></tr><tr><td>FID</td><td>CLIP</td><td>FID</td><td>CLIP</td></tr><tr><td>GRIMOIRE (w/o context)</td><td>1.67</td><td>28.64</td><td>3.58</td><td>27.45</td></tr><tr><td>GRIMOIRE (+ 3 stroke context)</td><td>2.78</td><td>27.25</td><td>4.65</td><td>25.31</td></tr><tr><td>GRIMOIRE (+ 6 stroke context)</td><td>3.16</td><td>27.25</td><td>5.46</td><td>25.54</td></tr><tr><td>GRIMOIRE (+ 12 stroke context)</td><td>2.95</td><td>27.57</td><td>6.04</td><td>25.85</td></tr><tr><td>GRIMOIRE (+ 24 stroke context)</td><td>2.25</td><td>28.12</td><td>6.05</td><td>26.39</td></tr></table>

# 7.7 Implementation Details

We use AdamW optimization and train the VSQ module for 1 epoch for Fonts and FIGR-8 and five epochs for MNIST. We use a learning rate of $\lambda = 2 \times 10^{-5}$ , while the auto-regressive Transformer is trained for $\sim 30$ epochs with $\lambda = 6 \times 10^{-4}$ . The Transformer has a context length of 512. Before proceeding to the second stage, we filter out icons represented by fewer than ten or more than 512 VSQ tokens, which affects $12.16\%$ of samples. We use p-sampling for our generations with GRIMOIRE. Training the VSQ module on six NVIDIA H100 takes approximately 48, 15, and 12 hours for MNIST, FIGR-8, and Fonts, respectively; the ART module takes considerably fewer resources, requiring around 8 hours depending on the configuration. Regarding Im2Vec, we replace the Ranger scheduler with AdamW (Loshchilov and Hutter, 2017) and enable the weighting factor for the Kullback-Leibler (KL) divergence in the loss function to 0.1, as it was disabled by default in the code repository, preventing any sampling. We train Im2Vec with six paths for 105 epochs with a learning rate of $\lambda = 2 \times 10^{-4}$ with early stopping if the validation loss does not decrease after seven epochs. Regarding the generative metrics, we utilized CLIP with a ViT-16 backend for FID and CLIPScore.

# 7.8 Generative Scores with Completion

To evaluate if GRIMOIRE generalizes and learns to meaningfully complete previously unseen objects, we compare the CLIPScore and FID of completions with varying lengths of context. The context and text prompts are extracted from 1,000 samples of the test set of the FIGR-8 dataset. The results are shown in Table 7.

While GRIMOIRE can meaningfully complete unseen objects, the quality of these completions is generally lower than the generations under text-only conditioning. This is expected, as prompts in the test set are also encountered during training (the class names). The CLIPScore generally drops to its lowest point with the least amount of context and then recovers when more context is given to the model, which coincides with our qualitative observations that with only a few context strokes, GRIMOIRE occasionally ignores them completely or completes them in an illogical way, reducing the visual appearance.

![](images/a59f6965bde70dd46f1d134d5d62e7a4e674bf11135d7d91d15b3453fdc4d356.jpg)

<details>
<summary>text_image</summary>

Icons on Fonts.
Fonts on icons.
</details>

Figure 12: Qualitative zero-shot reconstructions from the test-set of FIGR-8 and Fonts after training the VSQ module solely on the respective other dataset.

Table 8: Top ten most used strokes of the VSQ module trained on icons and their relative occurrences in our subset of FIGR-8. 

<table><tr><td>-</td><td>/</td><td>/</td><td>)</td><td>\</td><td>)</td><td>\</td><td>)</td><td>/</td><td>)</td></tr><tr><td>18.76%</td><td>12.26%</td><td>2.56%</td><td>1.73%</td><td>1.16%</td><td>1.12%</td><td>0.99%</td><td>0.94%</td><td>0.92%</td><td>0.80%</td></tr></table>

# 7.9 Domain Transfer Capabilities for Reconstruction

To validate how the strokes learned during the first training stage adapt to different domains, we use our VSQ module to reconstruct Fonts after training on FIGR-8, and vice versa. Figure 12 provides a qualitative example for each setting. Despite the loss value for each image being around one order of magnitude higher than the in-domain test-set (MSE≈0.05), the VSQ module uses reasonable codes to reconstruct the shapes and picks curves in the correct directions. Straight lines end up being the easiest to decode in both cases.

# 7.10 Codebook Usage for Strokes

As described in Section 3.1, for FSQ, we fixed the number of dimensions of the hypercube to 5 and set the individual number of values for each dimension as $L = [7, 5, 5, 5, 5]$ for a total codebook size of $|B| = 4,375$ . In this section, we want to share some interesting findings about the learnt codebook. For this, we shall use the VSQ trained on FIGR-8 with $n_{\mathrm{code}} = 1$ , $n_{\mathrm{seg}} = 2$ , a maximum stroke length of 3.0, and the geometric constraint with $\alpha = 0.2$ .

After training the VSQ on FIGR-8, we tokenize the full dataset. The resulting VQ tokens stem from 60.09% of the codebook, while 39.91% of the available codes remained unused. The ten most used strokes make up 41.24% of the dataset, while the top 24 and 102 strokes make up roughly 50% and 75%, respectively. These findings indicate that for these particular VSQ settings, one could experiment with smaller codebook sizes.

To balance out the stroke distribution, one could use a different subset of FIGR-8. Currently, the classes “menu”, “credit card”, “laptop”, and “monitor” are contributing the most to the stroke imbalance, with 26%, 24.3%, 24.05%, and 23.8% of their respective strokes being the most frequent horizontal one in Table 8.

# 7.11 Average Strokes in Codebook

In Section 7.10, we show the ten most used strokes of our trained VSQ, but after inspecting the full codebook we notice how neighbouring codes often express very similar strokes. Therefore, to visualize the codebook more effectively, we plot mean and minimum reductions of the full codebook in Figure 13. Additionally, we tokenize the full FIGR-8 dataset and plot the same reductions in Figure 14 to show the composition of the dataset.

codebook mean strokes

![](images/82dd121ca34b3982e21055d1d1b1ace59374a8127c05d7ad522226d78b44d060.jpg)

<details>
<summary>natural_image</summary>

Circular black ink blot on white background, no text or symbols present
</details>

all codebook strokes

Figure 13: Different reductions of all 4,375 strokes from the VSQ codebook. The model seems to have learned an expressive codebook-decoder mapping as the figure on the left shows a smooth and evenly distributed stroke profile and the figure on the right displays strokes in almost every direction.   
![](images/e0a4a1e3cb8643f43743033013e86421e5e7ea7846e8d90e56e2bdd3537af666.jpg)  
FIGR-8 mean strokes

![](images/4a7d6bbb40514d101b999810b95b2639e52995e2cf10cb6d5f63876e2b0986cc.jpg)  
FIGR-8 mean strokes excluding top ten strokes

![](images/dcaa5997d5971c0ca08b65e9e954aad187c3f28ff2d9dbf8b117d2445abe94da.jpg)

all FIGR-8 strokes

Figure 14: Different reductions of all strokes from the tokenized FIGR-8 dataset. The visualization on left shows the dominance of the two most occurring strokes, the middle shows that the distribution of strokes is skewed. The missing 39.91% of strokes are also visible in the right figure, where certain diagonal strokes that are available in the codebook are never used.

# 7.12 Qualitative Results – Reconstruction

In Figure 15 and Figure 16, we provide several qualitative examples of vector reconstructions using Im2Vec and our VSQ module on the Fonts and FIGR-8 datasets, respectively. We fill the shapes of the images when using Im2Vec, since the model creates SVGs as series of filled circles and would not be able to learn from strokes with a small width. Im2Vec does not converge when trained on the full datasets, whereas it returns some approximate reconstruction of the input when only a single class is adopted. In contrast, the VSQ module generalizes over the full dataset.

# 7.13 Qualitative Results – Generation

In this section, we provide qualitative examples of our reconstruction and generative pipeline, and compared those with Im2Vec. Figure 17 reports a few examples of icons generated with GRIMOIRE using only text-conditioning on classes. In Figure 18 we report some generations for MNIST. In Figure 19, we report generative results for Fonts. Thanks to the conditioning, we can generate upper-case and lower-case glyphs in bold, italic, light styles, and more. As can be seen in the table, GRIMOIRE also learns to properly mix those styles only based on text. Finally, in Figure 20, we report some generative results on icons and Fonts for Im2Vec on a single class dataset. The results show how the pipeline typically fails to produce meaningful or sufficiently diverse samples.

![](images/a60dabceed56b78b790bdfced3d1ccac633d6094b5ff5e8f2821f2bdb7bc774c.jpg)

<details>
<summary>text_image</summary>

GT
Im2Vec
GT
VSQ
GT
Im2Vec
GT
VSQ
</details>

Figure 15: Examples of various reconstructions of our VSQ module after training on Fonts compared to reconstructions of Im2Vec trained on the letter "A" (first row) and Im2Vec trained on the full Fonts dataset (third row).

![](images/899d2823f0df011704342b89bc7c545f77ab226cd9d73198848cb9412dc6a92c.jpg)

<details>
<summary>text_image</summary>

GT
Im2Vec
GT
VSQ
GT
Im2Vec
GT
VSQ
</details>

Figure 16: Examples of various reconstructions of our VSQ module after training on icons compared to reconstructions of Im2Vec trained on one class (first row) and Im2Vec trained on the full dataset (third row).

<table><tr><td>Clock</td><td>Luggage</td><td>Shopping Bag</td><td>Camera</td><td>Like</td></tr><tr><td>Home</td><td>Mail</td><td>Share</td><td>Target</td><td>Arrow</td></tr><tr><td>Microphone</td><td>Gift</td><td>Clock</td><td>Eye</td><td>Cube</td></tr><tr><td>Sun</td><td>Bell</td><td>Bell</td><td>Smile</td><td>Clip</td></tr><tr><td>Smile</td><td>Share</td><td>Bin</td><td>Book</td><td>Home</td></tr></table>

Figure 17: Examples of various samples generated with GRIMOIRE after training on icons, using only text conditioning.

![](images/a93998c9324e7cfe856ae3a1b38e9a1a29b2e8111f6b00ea8abf923218b86e5d.jpg)

<details>
<summary>text_image</summary>

0 1 2 3 4
5 6 7 8 9
</details>

Figure 18: Examples of a samples generated with GRIMOIRE for each digit of the MNIST dataset.

![](images/99ab09b622d16f56830d49a70552d8b6d2c27c2544f1617d3dac334279c2f4ed.jpg)

<details>
<summary>text_image</summary>

A
Star
User
Document
Camera
Book
</details>

Figure 19: Examples of filled samples generated with Im2Vec after training the model on specific classes of the dataset. For most classes, Im2Vec could not capture the diversity of the data and failed to meaningfully converge.

![](images/aace04561988c9308db3ed850df7fd01891bfc64b0f4db42c7bb9c13c3e99ba2.jpg)

<details>
<summary>text_image</summary>

Capital
A T T T T N
Y R G J X K
y r r n g d
5 h h 9 0 2
Regular Regular Italic Bold Bold-italic Light
</details>

Figure 20: Examples of various samples generated with GRIMOIRE after training on Fonts, using only text conditioning.