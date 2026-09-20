# REVISIT AND OUTSTRIP ENTITY ALIGNMENT: A PERSPECTIVE OF GENERATIVE MODELS

Lingbing Guo $^{1,2,3,*}$ , Zhuo Chen $^{1,2,3,*}$ , Jiaoyan Chen $^{4}$ , Yin Fang $^{1,2,3}$ , Wen Zhang $^{5,2,*}$

# Huajun Chen $^{1,2,3\dagger}$

$^{1}$ College of Computer Science and Technology, Zhejiang University   
$^{2}$ Zhejiang University - Ant Group Joint Laboratory of Knowledge Graph   
$^{3}$ Donghai Laboratory   
$^{4}$ Department of Computer Science, The University of Manchester   
$^{5}$ School of Software Technology, Zhejiang University

# ABSTRACT

Recent embedding-based methods have achieved great successes in exploiting entity alignment from knowledge graph (KG) embeddings of multiple modalities. In this paper, we study embedding-based entity alignment (EEA) from a perspective of generative models. We show that EEA shares similarities with typical generative models and prove the effectiveness of the recently developed generative adversarial network (GAN)-based EEA methods theoretically. We then reveal that their incomplete objective limits the capacity on both entity alignment and entity synthesis (i.e., generating new entities). We mitigate this problem by introducing a generative EEA (GEEA) framework with the proposed mutual variational autoencoder (M-VAE) as the generative model. M-VAE enables entity conversion between KGs and generation of new entities from random noise vectors. We demonstrate the power of GEEA with theoretical analysis and empirical experiments on both entity alignment and entity synthesis tasks.

# 1 INTRODUCTION

As one of the most prevalent tasks in the knowledge graph (KG) area, entity alignment (EA) has recently made great progress and developments with the support of the embedding techniques (Chen et al., 2017; Sun et al., 2017; Zhang et al., 2019; Chen et al., 2020; Liu et al., 2021; Chen et al., 2022a;b; Guo et al., 2022a;b; Lin et al., 2022). By encoding the relational and other information into low-dimensional vectors, the embedding-based entity alignment (EEA) methods are friendly for development and deployment, and have achieved state-of-the-art performance on many benchmarks.

The objective of EA is to maximize the conditional probability $p(y|x)$ , where x, y are a pair of aligned entities belonging to source KG X and target KG Y, respectively. If we view x as the input and y as the label (and vice versa), the problem can be solved by a discriminative model. To this end, we need an EEA model which comprises an encoder module and a fusion layer (Zhang et al., 2019; Chen et al., 2020; Liu et al., 2021; Chen et al., 2022a;b; Lin et al., 2022) (see Figure 1). The encoder module uses different encoders to encode multi-modal information into low-dimensional embeddings. The fusion layer then combines these sub-embeddings to a joint embedding as the output.

We also need a predictor, as shown in the yellow area in Figure 1. The predictor is usually independent of the EEA model and parameterized with neural layers (Chen et al., 2017; Guo et al., 2020) or based on the embedding distance (Sun et al., 2017; 2018). In either case, it learns the probability $p(y|x)$ where $p(y|x) = 1$ if the two entities $x, y$ are aligned and 0 otherwise. The difference lies primarily in data augmentation. The existing methods employ different strategies to construct more training data, e.g., negative sampling (Chen et al., 2017; Sun et al., 2017; Wang et al., 2018) and bootstrapping (Sun et al., 2018; Pei et al., 2019a; Guo et al., 2022a). In this paper, we demonstrate that adopting a

![](images/9eda6b82a2d30529761129bf025542b8967ee0c2a43e0edc26daa8975d41ce34.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["graph"] --> B["label title ..."]
    C_leurle1["Graph Encoder"] --> D["Attribute Encoder"]
    E["image"] --> F["Image Encoder"]
    G_FundingLayer["Fusion Layer"] --> H["Fusion"]
    I["x"] --> J["Predictor"]
    K["y"] --> L["Target"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style G FundingLayer fill:#ccf,stroke:#333
    style I fill:#cfc,stroke:#333
    style K fill:#fcc,stroke:#333
```
</details>

Figure 1: Illustration of embedding-based entity alignment. The modules in the blue area belong to the EEA model, while those in the yellow area belong to the predictor.

generative perspective in studying EEA allows us to interpret negative sampling algorithms and GAN-based methods (Pei et al., 2019a;b; Guo et al., 2022b). Furthermore, we provide theoretical proof that optimizing the generative objectives contributes to minimizing the EEA objective, thereby enhancing overall performance.

In fact, entity alignment is not the ultimate aim of many applications. The results of entity alignment are used to enrich each other's KGs, but there are often entities in the source KG that do not have aligned counterparts in the target KG, known as dangling entities (Sun et al., 2021; Liu et al., 2022; Luo & Yu, 2022). For instance, a source entity Star Wars (film) may not have a counterpart in the target KG, which means we cannot directly enrich the target KG with the information of Star Wars (film) via entity alignment. However, if we can convert entities like Star Wars (film) from the source KG to the target KG, it would save a major expenditure of time and effort for many knowledge engineering tasks, such as knowledge integration and fact checking. Hence, we propose conditional entity synthesis to generate new entities for the target KG with the entities in the source KG as input. Additionally, generating new entities from random variables may contribute to the fields like Metaverse and video games where the design of virtual characters still relies on hand-crafted features and randomized algorithms (Khalifa et al., 2017; Lee et al., 2021). For example, modern video games feature a large number of non-player characters (NPCs) with unique backgrounds and relationships, which are essential for creating immersive virtual worlds. Designing NPCs is a laborious and complex process, and using the randomized algorithms often yields unrealistic results. By storing the information and relationships of NPCs in a KG, one can leverage even a small initial training KG to generate high-quality NPCs with coherent backgrounds and relationships. Therefore, we propose unconditional entity synthesis for generating new entities with random noise vectors as input.

We propose a generative EEA (abbr., GEEA) framework with the mutual variational autoencoder (M-VAE) to encode/decode entities between source and target KGs. GEEA is capable of generating concrete features, such as the exact neighborhood or attribute information of a new entity, rather than only the inexplicable embeddings as previous works have done (Pei et al., 2019a;b; Guo et al., 2022b). We introduce the prior reconstruction and post reconstruction to control the generation process. Briefly, the prior reconstruction is used to generate specific features for each modality, while the post reconstruction ensures these different kinds of features belong to the same entity. We conduct experiments to validate the performance of GEEA, where it achieves state-of-the-art performance in entity alignment and generates high-quality new entities in entity synthesis.

# 2 REVISIT EMBEDDING-BASED ENTITY ALIGNMENT

In this section, we revisit embedding-based entity alignment by a theoretical analysis of how the generative models contribute to entity alignment learning, and then discuss their limitations.

# 2.1 PRELIMINARIES

Entity Alignment Entity alignment aims to find the implicitly aligned entity pairs $\{(x,y)|x\in\mathcal{X},y\in\mathcal{Y}\}$ , where X, Y denote the source and target entity sets, and $(x,y)$ represents a pair of aligned entities referring to the same real-world object. An EEA model M uses a small number of aligned entity pairs S (a.k.a., seed alignment set) as training data to infer the remaining alignment pairs T in the testing set. We consider three different modalities: relational graphs $G_{x},G_{y}$ , attributes $A_{x},A_{y}$ , and images $I_{x},I_{y}$ . Other types of information can be also given as features for X and Y.

For instance, the relational graph feature of an entity Star Wars (film) is represented as triplets, such as (Star Wars (film), founded by, George Lucas). Similarly, the attribute feature is represented as attribute triplets, e.g., (Star Wars (film), title, Star Wars (English)). For the image feature, we follow the existing multi-modal EEA works to use a constant pretrained embedding from a vision model as the image feature of Star Wars (film) (Liu et al., 2021; Lin et al., 2022). The EEA model M takes the above multi-modal features $x = (g_x, a_x, i_x, \ldots)$ as input, where $g_x, a_x, i_x$ denote the relational graph information, attribute information and image information of x, respectively. The output consists of the embeddings for each modality (i.e., sub-embeddings) and a final output embedding x (i.e., joint embedding) that combines all modalities:

$$
\mathbf {x} = \mathcal {M} (x) = \text { Linear } (\text { Concat } (\mathcal {M} _ {g} (g _ {x}), \mathcal {M} _ {a} (a _ {x}), \mathcal {M} _ {i} (i _ {x}),...)) \tag {1}
$$

$$
= \text { Linear } (\text { Concat } (\mathbf {g} _ {x}, \mathbf {a} _ {x}, \mathbf {i} _ {x}, \dots)), \tag {2}
$$

where $M_{g}$ , $M_{a}$ , and $M_{i}$ denote the EEA encoders for different modalities (also see Figure 1). $g_{x}$ , $a_{x}$ , and $i_{x}$ denote the embeddings of different modalities. Similarly, we obtain y by $\mathbf{y} = \mathcal{M}(y)$ .

Entity Synthesis We consider two entity synthesis tasks: conditional entity synthesis and unconditional entity synthesis. Conditional entity synthesis aims to generate entities in the target KG with the dangling entities in the source KG as input. Formally, the model takes an entity x as input and convert it to an entity $y_{x \rightarrow y}$ for the target KG. It should also produce the corresponding concrete features, such as neighborhood and attribute information specific to the target KG. On the other hand, the unconditional entity synthesis involves generating new entities in the target KG with random noise variables as input. Formally, the model takes a random noise vector z as input and generate a target entity embedding $y_{z \rightarrow y}$ which is then converted back to concrete features.

For instance, to reconstruct the neighborhood (or attribute) information of Star Wars (film) from its embedding, we can leverage a decoder module to convert the embedding into a probability distribution of all candidate entities (or attributes). As the image features are constant pretrained embeddings, we can use the image corresponding to the nearest neighbor of the reconstructed image embedding of Star Wars (film) as the output image.

Generative Models Generative models learn the underlying probability distribution $p(x)$ of the input data x. Take variational autoencoder (VAE) (Kingma & Welling, 2013) as an example, the encoding and decoding processes can be defined as:

$$
\mathbf {h} = \operatorname{Encoder} (\mathbf {x}) \quad (\text {Encoding}) \tag {3}
$$

$$
\mathbf {z} = \mu + \sigma \odot \epsilon = \operatorname{Linear} _ {\mu} (\mathbf {h}) + \operatorname{Linear} _ {\sigma} (\mathbf {h}) \odot \epsilon \quad (\text {Reparameterization Trick}) \tag {4}
$$

$$
\mathbf {x} _ {x \rightarrow x} = \operatorname{Decoder} (\mathbf {z}) \quad (\text {Decoding}), \tag {5}
$$

where h is the hidden output. VAE uses the reparameterization trick to rewrite h as coefficients $\mu$ , $\sigma$ in a deterministic function of a noise variable $\epsilon \in \mathcal{N}(\epsilon; \mathbf{0}, \mathbf{I})$ , to enable back-propagation. $x_{x \to x}$ denotes that this reconstructed entity embedding is with x as input and for x. VAE generates new entities by sampling a noise vector z and converting it to x.

# 2.2 EEA BENEFITS FROM THE GENERATIVE OBJECTIVES

Let $x \sim X$ , $y \sim Y$ be two entities sampled from the entity sets $X$ , $Y$ , respectively. The main target of EEA is to learn a predictor that estimates the conditional probability $p_{\theta}(\mathbf{x}|\mathbf{y})$ (and reversely $p_{\theta}(\mathbf{y}|\mathbf{x})$ ), where $\theta$ represents the parameter set. For simplicity, we assume that the reverse function $p_{\theta}(\mathbf{y}|\mathbf{x})$ shares the same parameter set with $p_{\theta}(\mathbf{x}|\mathbf{y})$ .

Now, suppose that one wants to learn a generative model for generating entity embeddings:

$$
\log p (\mathbf {x}) = \log p (\mathbf {x}) \int p _ {\theta} (\mathbf {y} | \mathbf {x}) d \mathbf {y} \tag {6}
$$

$$
= \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p (\mathbf {x} , \mathbf {y})}{p _ {\theta} (\mathbf {y} | \mathbf {x})} \right] + D _ {\mathrm{KL}} (p _ {\theta} (\mathbf {y} | \mathbf {x}) \| p (\mathbf {y} | \mathbf {x})), \tag {7}
$$

where the left-hand side of Equation (7) is the evidence lower bound (ELBO) (Kingma & Welling, 2013), and the right-hand side is the Kullback-Leibler (KL) divergence (Kullback & Leibler, 1951) between our parameterized distribution $p_{\theta}(\mathbf{y}|\mathbf{x})$ (i.e., the predictor) and the true distribution $p(\mathbf{y}|\mathbf{x})$ .

In typical generative learning, $p(\mathbf{y}|\mathbf{x})$ is intractable because y is a noise variable sampled from a normal distribution, and thus $p(\mathbf{y}|\mathbf{x})$ is unknown. However, in EEA, we can obtain a few samples by using the training set, which leads to a classical negative sampling loss (Sun et al., 2017; Cao et al., 2019; Zhang et al., 2019; Chen et al., 2020; Guo et al., 2020; Sun et al., 2020a; Liu et al., 2021; Chen et al., 2022a;b; Guo et al., 2022a;b; Lin et al., 2022):

$$
\mathcal {L} _ {\mathrm{ns}} = \sum_ {i} [ - \log (p _ {\theta} (\mathbf {y} ^ {i} | \mathbf {x} ^ {i}) p (\mathbf {y} ^ {i} | \mathbf {x} ^ {i})) + \frac {1}{N _ {\mathrm{ns}}} \sum_ {j \neq i} \log \big (p _ {\theta} (\mathbf {y} ^ {j} | \mathbf {x} ^ {i}) (1 - p (\mathbf {y} ^ {j} | \mathbf {x} ^ {i})) \big) ], \tag {8}
$$

where $(\mathbf{y}^{i},\mathbf{x}^{i})$ denotes a pair of aligned entities in the training data. The randomly sampled entity $y^{j}$ is regarded as the negative entity. i, j are the entity IDs. $N_{ns}$ is the normalization constant. Here, $L_{ns}$ is formulated as a cross-entropy loss with the label $p(\mathbf{y}^{j}|\mathbf{x}^{i})$ defined as:

$$
p (\mathbf {y} ^ {j} | \mathbf {x} ^ {i}) = \left\{ \begin{array}{l l} 0, & \text { if } \quad i \neq j, \\ 1, & \text { otherwise } \end{array} \right. \tag {9}
$$

Given that EEA typically uses only a small number of aligned entity pairs for training, the observation of $p(\mathbf{y}|\mathbf{x})$ may be subject to bias and limitations. To alleviate this problem, the recent GAN-based methods (Pei et al., 2019a;b; Guo et al., 2022b) propose leveraging entities outside the training set for unsupervised learning. The common idea behind these methods is to make the entity embeddings from different KGs indiscriminative to a discriminator, such that the underlying aligned entities shall be encoded in the same way and have similar embeddings. To formally prove this idea, we dissect the ELBO in Equation (7) as follows:

$$
\mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p (\mathbf {x} , \mathbf {y})}{p _ {\theta} (\mathbf {y} | \mathbf {x})} \right] = \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log p _ {\theta} (\mathbf {x} | \mathbf {y}) \right] - D _ {\mathrm{KL}} (p _ {\theta} (\mathbf {y} | \mathbf {x}) \| p (\mathbf {y})) \tag {10}
$$

The complete derivation in this section can be found in Appendix A.1. Therefore, we have:

$$
\log p (\mathbf {x}) = \underbrace {\mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log p _ {\theta} (\mathbf {x} | \mathbf {y}) \right]} _ {\text { reconstruction   term }} - \underbrace {D _ {\mathrm{KL}} (p _ {\theta} (\mathbf {y} | \mathbf {x}) \| p (\mathbf {y}))} _ {\text { distribution   matching   term }} + \underbrace {D _ {\mathrm{KL}} (p _ {\theta} (\mathbf {y} | \mathbf {x}) \| p (\mathbf {y} | \mathbf {x}))} _ {\text { prediction   matching   term }} \tag {11}
$$

The first term aims to reconstruct the original embedding x based on y generated from x, which has not been studied in existing discriminative EEA methods (Guo et al., 2020; Liu et al., 2021; Lin et al., 2022). The second term enforces the distribution of y conditioned on x to match the prior distribution of y, which has been investigated by the GAN-based EEA methods (Pei et al., 2019a;b; Guo et al., 2022b). The third term represents the main objective of EEA (as described in Equation (8) where the target $p(\mathbf{y}|\mathbf{x})$ is partially observed).

Note that, $p(\mathbf{x})$ is irrelevant to our parameter set $\theta$ and can be treated as a constant during optimization. Consequently, maximizing the ELBO (i.e., maximizing the first term and minimizing the second term) will result in minimizing the third term:

Proposition 1. Maximizing the reconstruction term and/or minimizing the distribution matching term subsequently minimizes the EEA prediction matching term.

The primary objective of EEA is to minimize the prediction matching term. Proposition 1 provides theoretical evidence that the generative objectives naturally contribute to the minimization of the EEA objective, thereby enhancing overall performance.

# 2.3 THE LIMITATIONS OF GAN-BASED EEA METHODS

The GAN-based EEA methods leverage a discriminator to discriminate the entities from one KG against those from another KG. Supposed that x, y are embeddings produced by an EEA model M, sampled from the source KG and the target KG, respectively. The GAN-based methods train a discriminator D to distinguish x from y (and vice versa), with the following objective:

$$
\underset {\mathbf {x}, \mathbf {y}, \psi} {\operatorname{argmax}} \left[ \mathbb {E} _ {x \sim \mathcal {X}} \log \mathcal {D} _ {\phi} (\mathcal {M} _ {\psi} (x)) + \mathbb {E} _ {y \sim \mathcal {Y}} \log \mathcal {D} _ {\phi} (\mathcal {M} _ {\psi} (y)) \right] \quad (\text { Generator }) \tag {12}
$$

$$
+ \underset {\phi} {\operatorname{argmax}} \left[ \mathbb {E} _ {x \sim \mathcal {X}} \log \mathcal {D} _ {\phi} (\mathcal {M} _ {\psi} (x)) + \mathbb {E} _ {y \sim \mathcal {Y}} \log (1 - \mathcal {D} _ {\phi} (\mathcal {M} _ {\psi} (y))) \right] \quad (\text {Discriminator}) \tag {13}
$$

Here, the EEA model M takes entities x, y as input and produces the output embeddings x, y, respectively. D is the discriminator that learns to predict whether the input variable is from the target distribution. $\phi$ , $\psi$ are the parameter sets of M, D, respectively.

It is important to note that both $\mathbf{x} = \mathcal{M}_{\psi}(x)$ and $\mathbf{y} = \mathcal{M}_{\psi}(y)$ do not follow a fixed distribution (e.g., a normal distribution). They are learnable vectors during training, which is significantly different from the objective of a typical GAN, where variables like x (e.g., an image) and z (e.g., sampled from a normal distribution) have deterministic distributions. Consequently, the generator in Equation (12) can be overly strong, allowing x, y to be consistently mapped to plausible positions to deceive D.

Therefore, one major issue with the existing GAN-based methods is mode collapse (Srivastava et al., 2017; Pei et al., 2019b; Guo et al., 2022b). Mode collapse often occurs when the generator (i.e., the EEA model in our case) over-optimizes for the discriminator. The generator may find some outputs appear most plausible to the discriminator and consistently produces those outputs. This is harmful for EEA as irrelevant entities are encouraged to have similar embeddings. We argue that mode collapse is more likely to occur in the existing GAN-based EEA methods, which is why they often use a very small weight (e.g., 0.001 or less) to optimize the generator against the discriminator (Pei et al., 2019b; Guo et al., 2022b).

Another limitation of the existing GAN-based methods is their inability to generate new entities. The generated target entity embedding $y_{x\to y}$ cannot be converted back to the native concrete features, such as the neighborhood $\{y^{1}, y^{2}, \ldots\}$ or attributes $\{a^{1}, a^{2}, \ldots\}$ .

# 3 GENERATIVE EMBEDDING-BASED ENTITY ALIGNMENT

# 3.1 MUTUAL VARIATIONAL AUTOENCODER

In many generative tasks, such as image synthesis, the conditional variable (e.g., a textual description) and the input variable (e.g., an image) differ in modality. However, in our case, they are entities from different KGs. Therefore, we propose mutual variational autoencoder (M-VAE) for efficient generation of new entities. One of the most important characteristics of M-VAE lies in the variety of the encode-decode process. It has four different flows:

The first two flows are used for self-supervised learning, i.e., reconstructing the input variables:

$$
\mathbf {x} _ {x \rightarrow x}, \mathbf {z} _ {x \rightarrow x} = V A E (\mathbf {x}), \quad \mathbf {y} _ {y \rightarrow y}, \mathbf {z} _ {y \rightarrow y} = V A E (\mathbf {y}), \quad \forall x, \forall y, x \in \mathcal {X}, y \in \mathcal {Y} \tag {14}
$$

We use the subscript $x \to x$ to denote the flow is from x to x, and similarly for $y \to y$ . $z_{x \to x}$ , $z_{y \to y}$ are the latent variables (as defined in Equation 4) of the two flows, respectively. In EEA, the majority of alignment pairs are unknown, but all information of the entities is known. Thus, these two flows provide abundant examples to train GEEA in a self-supervised fashion.

The latter two flows are used for supervised learning, i.e., reconstructing the mutual target variables:

$$
\mathbf {y} _ {x \rightarrow y}, \mathbf {z} _ {x \rightarrow y} = V A E (\mathbf {x}), \quad \mathbf {x} _ {y \rightarrow x}, \mathbf {z} _ {y \rightarrow x} = V A E (\mathbf {y}), \quad \forall (x, y) \in \mathcal {S}. \tag {15}
$$

It is worth noting that we always share the parameters of VAEs across all flows. We wish the rich experience gained from reconstructing the input variables (Equation (14)) can be flexibly conveyed to reconstructing the mutual target (Equation (15)).

# 3.2 DISTRIBUTION MATCH

The existing GAN-based methods directly minimize the KL divergence (Kullback & Leibler, 1951) between two embedding distributions, resulting in the over-optimization of generator and incapability of generating new entities. In this paper, we propose to draw support from the latent noise variable z to avoid these two issues. The distribution match loss is defined as follows:

$$
\mathcal {L} _ {\mathrm{kld}} = D _ {\mathrm{KL}} (p (\mathbf {z} _ {x \rightarrow x}), p (\mathbf {z} ^ {*})) + D _ {\mathrm{KL}} (p (\mathbf {z} _ {y \rightarrow y}), p (\mathbf {z} ^ {*})). \tag {16}
$$

where $p(\mathbf{z}_{x\to x})$ denotes the distribution of $z_{x\to x}$ , and $p(\mathbf{z}^{*})$ denotes the target normal distribution. We do not optimize the distributions of $z_{x\to y}$ , $z_{y\to x}$ in the latter two flows, because they are sampled from seed alignment set S, a (likely) biased and small training set.

![](images/b229e11489b986f6eb6f567fca9c4c6fe5d1f184f8ec6e8ac6f4cb7167af840f.jpg)  
Figure 2: The workflow of GEEA. Top: different sub-VAEs process different sub-embeddings, and the respective decoders convert the sub-embeddings back to concrete features. Bottom-left: the entity alignment prediction loss is retained. Bottom-center: the latent variables of sub-VAEs are used for distribution matching. Bottom-right: The reconstructed sub-embeddings are feed into the fusion layer in the EEA model to produce the reconstructed joint embedding for post reconstruction.

Minimizing $L_{kld}$ can be regarded as aligning the entity embeddings from respective KGs to a fixed normal distribution. We provide a formal proof that the entity embedding distributions of two KGs will be aligned although we do not implicitly minimize $D_{\mathrm{KL}}(p(\mathbf{x}), p(\mathbf{y}))$ :

Proposition 2. Let $\mathbf{z}^*$ , $\mathbf{z}_{x\to x}$ , $\mathbf{z}_{y\to y}$ be the normal distribution, and the latent variable distributions w.r.t. $\mathcal{X}$ and $\mathcal{Y}$ , respectively. Jointly minimizing the KL divergence $D_{\mathrm{KL}}(p(\mathbf{z}_{x\to x}), p(\mathbf{z}^*))$ , $D_{\mathrm{KL}}(p(\mathbf{z}_{y\to y}), p(\mathbf{z}^*))$ will contribute to minimizing $D_{\mathrm{KL}}(p(\mathbf{x}), p(\mathbf{y}))$ :

$$
D _ {\mathrm{KL}} (p (\mathbf {x}), p (\mathbf {y})) \propto D _ {\mathrm{KL}} (p (\mathbf {z} _ {x \rightarrow x}), p (\mathbf {z} ^ {*})) + D _ {\mathrm{KL}} (p (\mathbf {z} _ {y \rightarrow y}), p (\mathbf {z} ^ {*})) \tag {17}
$$

Proof. Please see Appendix A.2.

![](images/8d9c5b40b3f57f2f6eeec400528f072f9c69d954281c15e9b073c9f6a58bace5.jpg)

# 3.3 PRIOR RECONSTRUCTION

The prior reconstruction aims to reconstruct the sub-embedding of each modality and recover the original concrete feature from the sub-embedding. Take the relational graph information of flow $x \rightarrow y$ as an example, we first employ a sub-VAE to process the input sub-embedding:

$$
\mathbf {g} _ {x \rightarrow y}, \mathbf {z} _ {x \rightarrow y} ^ {g} = V A E _ {g} (\mathbf {g} _ {x}) \tag {18}
$$

where $VAE_{g}$ denotes the variational autoencoder for relational graph information. $g_{x}$ is the graph embedding of x, and $g_{x\to y}$ is the reconstructed graph embedding for y based on x. $z_{x\to y}^{g}$ is the corresponding latent variable. To recover the original features (i.e., the neighborhood information of y), we consider a prediction loss defined as:

$$
\mathcal {L} _ {\mathbf {g} _ {x \rightarrow y}} = g _ {y} \log \text { Decoder } _ {g} (\mathbf {g} _ {x \rightarrow y}) + (1 - g _ {y}) \log (1 - \text { Decoder } _ {g} (\mathbf {g} _ {x \rightarrow y})) \tag {19}
$$

Here, $L_{g_{x\to y}}$ is a binary cross-entropy (BCE) loss. We employ a decoder $Decoder_{g}$ to convert the reconstructed sub-embedding $g_{x\to y}$ to a probability estimation regarding the neighborhood of y.

# 3.4 POST RECONSTRUCTION

We propose post reconstruction to ensure the reconstructed features of different modalities belong to the same entity. We re-input the reconstructed sub-embeddings $\{g_{x\to y}, a_{x\to y}, \ldots\}$ to the fusion layer (defined in the EEA model M) to obtain a reconstructed joint embedding $y_{x\to y}$ . We then employs mean square error (MSE) loss to match the reconstructed joint embedding with the original one:

$$
\mathbf {y} _ {x \rightarrow y} = \text { Fusion } (\{\mathbf {g} _ {x \rightarrow y}, \mathbf {a} _ {x \rightarrow y}, \dots \}), \quad \forall (x, y) \in \mathcal {S} \tag {20}
$$

$$
\mathcal {L} _ {x \rightarrow y} = M S E (\mathbf {y} _ {x \rightarrow y}, \text { NoGradient } (\mathbf {y})), \quad \forall (x, y) \in \mathcal {S}, \tag {21}
$$

Table 1: Entity alignment results on DBP15K datasets, without surface information and iterative strategy. ↑: higher is better; ↓: lower is better. Average of 5 runs, the same below. 

<table><tr><td rowspan="2">Models</td><td colspan="3">DBP15KZH-EN</td><td colspan="3">DBP15KJA-EN</td><td colspan="3">DBP15KFR-EN</td></tr><tr><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td></tr><tr><td>MUGNN (Cao et al., 2019)</td><td>.494</td><td>.844</td><td>.611</td><td>.501</td><td>.857</td><td>.621</td><td>.495</td><td>.870</td><td>.621</td></tr><tr><td>AliNet (Sun et al., 2020a)</td><td>.539</td><td>.826</td><td>.628</td><td>.549</td><td>.831</td><td>.645</td><td>.552</td><td>.852</td><td>.657</td></tr><tr><td>decentRL (Guo et al., 2020)</td><td>.589</td><td>.819</td><td>.672</td><td>.596</td><td>.819</td><td>.678</td><td>.602</td><td>.842</td><td>.689</td></tr><tr><td>EVA (Liu et al., 2021)</td><td>.680</td><td>.910</td><td>.762</td><td>.673</td><td>.908</td><td>.757</td><td>.683</td><td>.923</td><td>.767</td></tr><tr><td>MSNEA (Chen et al., 2022a)</td><td>.601</td><td>.830</td><td>.684</td><td>.535</td><td>.775</td><td>.617</td><td>.543</td><td>.801</td><td>.630</td></tr><tr><td>MCLEA (Lin et al., 2022)</td><td>.715</td><td>.923</td><td>.788</td><td>.715</td><td>.909</td><td>.785</td><td>.711</td><td>.909</td><td>.782</td></tr><tr><td>NeoEA (MCLEA) (Guo et al., 2022b)</td><td>.723</td><td>.924</td><td>.796</td><td>.721</td><td>.909</td><td>.789</td><td>.717</td><td>.910</td><td>.787</td></tr><tr><td>GEEA</td><td>.761</td><td>.946</td><td>.827</td><td>.755</td><td>.953</td><td>.827</td><td>.776</td><td>.962</td><td>.844</td></tr></table>

Table 2: Results on FB15K-DB15K and FB15K-YAGO15K datasets. 

<table><tr><td rowspan="2">Models</td><td rowspan="2"># Paras (M) / Training time (s)</td><td colspan="3">FB15K-DB15K</td><td colspan="3">FB15K-YAGO15K</td></tr><tr><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td></tr><tr><td>EVA</td><td>10.2/1,467.6</td><td>.199</td><td>.448</td><td>.283</td><td>.153</td><td>.361</td><td>.224</td></tr><tr><td>MSNEA</td><td>11.5/775.2</td><td>.114</td><td>.296</td><td>.175</td><td>.103</td><td>.249</td><td>.153</td></tr><tr><td>MCLEA</td><td>13.2/285.4</td><td>.295</td><td>.582</td><td>.393</td><td>.254</td><td>.484</td><td>.332</td></tr><tr><td> $GEEA_{SMALL}$ </td><td>11.2/217.3</td><td>.322</td><td>.602</td><td>.417</td><td>.270</td><td>.513</td><td>.352</td></tr><tr><td>GEEA</td><td>13.9/252.4</td><td>.343</td><td>.661</td><td>.450</td><td>.298</td><td>.585</td><td>.393</td></tr></table>

![](images/d2f06a6a21a1972657303791b91b8aaf306ddf1ffe6214804a07033a9f03f82a.jpg)

<details>
<summary>line</summary>

| Epoch | EVA   | MSNEA | MCLEA | GEEA  |
|-------|-------|-------|-------|-------|
| 0     | 0.0   | 0.0   | 0.0   | 0.0   |
| 200   | 0.2   | 0.1   | 0.4   | 0.5   |
| 400   | 0.3   | 0.15  | 0.35  | 0.45  |
| 600   | 0.35  | 0.18  | 0.35  | 0.4   |
| 800   | 0.35  | 0.2   | 0.35  | 0.4   |
</details>

Figure 3: MRR results on FBDB15K, w.r.t. epochs.

where $L_{x\to y}$ denotes the post reconstruction loss for the reconstructed joint embedding $y_{x\to y}$ . Fusion represents the fusion layer in M, and MSE is the mean square error. We use the copy value of the original joint embedding NoGradient(y) to avoid y inversely match $y_{x\to y}$ .

# 3.5 IMPLEMENTATION DETAILS

We take Figure 2 as an example to illustrate the workflow of GEEA. First, the sub-embeddings outputted by M are used as input for sub-VAEs (top-left). Then, the reconstructed sub-embeddings are passed to respective decoders to predict the concrete features of different modalities (top-right). The conventional entity alignment prediction loss is also retained in GEEA (bottom-left). The latent variables outputted by sub-VAEs are further used to match the predefined normal distribution (bottom-center). The reconstructed sub-embeddings are fed into the fusion layer to obtain a reconstructed joint embedding, which is used to match the true joint embedding for post reconstruction (bottom-right). The final training loss is defined as:

$$
\mathcal {L} = \underbrace {\sum_ {f \in \mathcal {F}} \left(\underbrace {\sum_ {m \in \{g , a , i , . . . \}} \mathcal {L} _ {\mathbf {m} _ {f}}} _ {\text { prior reconstruction }} + \underbrace {\mathcal {L} _ {f}} _ {\text { post reconstruction }}\right)} _ {\text { reconstruction term }} + \underbrace {\sum_ {m \in \{g , a , i , . . . \}} \mathcal {L} _ {\mathrm{kld} , m}} _ {\text { distribution matching term }} + \underbrace {\mathcal {L} _ {\mathrm{ns}}} _ {\text { prediction matching term }} \tag {22}
$$

where $F = \{x \to x, y \to y, x \to y, y \to x\}$ is the set of all flows, and $\{g, a, i, \ldots\}$ is the set of all available modalities. For more details, please refer to Appendix B.

# 4 EXPERIMENTS

# 4.1 SETTINGS

We used the multi-modal EEA benchmarks (DBP15K (Sun et al., 2017), FB15K-DB15K and FB15K-YAGO15K (Chen et al., 2020)) as datasets, excluding surface information (i.e., the textual label information) to prevent data leakage (Sun et al., 2020b; Chen et al., 2022b). The baselines MUGNN (Cao et al., 2019), AliNet (Sun et al., 2020a) and decentRL (Guo et al., 2020) are methods tailored to relational graphs, while EVA (Liu et al., 2021), MSNEA (Chen et al., 2022a) and MCLEA (Lin et al., 2022) are state-of-the-art multi-modal EEA methods. We chose MCLEA (Lin et al., 2022) as the EEA model of GEEA and NeoEA (Guo et al., 2022b) in the main experiments. The results of using other models (e.g., EVA and MSNEA) can be found in Appendix C. The neural layers and input/hidden/output dimensions were kept identical for fair comparison.

Table 3: Entity synthesis results on five datasets. PRE ( $\times10^{-2}$ ), RE ( $\times10^{-2}$ ) denote the reconstruction errors for prior concrete features and output embeddings, respectively. 

<table><tr><td rowspan="2">Models</td><td colspan="3">DBP15KZH-EN</td><td colspan="3">DBP15KJA-EN</td><td colspan="3">DBP15KFR-EN</td><td colspan="3">FB15K-DB15K</td><td colspan="3">FB15K-YAGO15K</td></tr><tr><td>PRE↓</td><td>RE↓</td><td>FID↓</td><td>PRE↓</td><td>RE↓</td><td>FID↓</td><td>PRE↓</td><td>RE↓</td><td>FID↓</td><td>PRE↓</td><td>RE↓</td><td>FID↓</td><td>PRE↓</td><td>RE↓</td><td>FID↓</td></tr><tr><td>MCLEA + decoder</td><td>8.104</td><td>4.218</td><td>N/A</td><td>7.640</td><td>5.441</td><td>N/A</td><td>10.578</td><td>5.985</td><td>N/A</td><td>18.504</td><td>inf</td><td>N/A</td><td>20.997</td><td>inf</td><td>N/A</td></tr><tr><td>VAE + decoder</td><td>0.737</td><td>0.206</td><td>1.821</td><td>0.542</td><td>0.329</td><td>2.184</td><td>0.856</td><td>0.689</td><td>3.083</td><td>10.564</td><td>11.354</td><td>10.495</td><td>9.645</td><td>9.982</td><td>16.180</td></tr><tr><td>Sub-VAEs + decoder</td><td>0.701</td><td>0.246</td><td>1.920</td><td>0.531</td><td>0.291</td><td>2.483</td><td>0.514</td><td>0.663</td><td>2.694</td><td>3.557</td><td>15.589</td><td>4.340</td><td>2.424</td><td>5.576</td><td>5.503</td></tr><tr><td>GEEA</td><td>0.438</td><td>0.184</td><td>0.935</td><td>0.385</td><td>0.195</td><td>1.871</td><td>0.451</td><td>0.121</td><td>2.422</td><td>3.141</td><td>6.151</td><td>3.089</td><td>1.730</td><td>2.039</td><td>3.903</td></tr></table>

![](images/2a7043d0ffe13245d550ae8a18f204d6a14b2ae35d466bb4277d26a20d6b1719.jpg)

<details>
<summary>line</summary>

| Ratio | MCLEA | GEEA  |
|-------|-------|-------|
| 10%   | 0.169 | 0.230 |
| 20%   | 0.295 | 0.343 |
| 30%   | 0.383 | 0.530 |
| 50%   | 0.555 | 0.651 |
| 80%   | 0.735 | 0.787 |
</details>

![](images/c3a31919b4dba0fc8bc1bc7d280d3499f298383a4aa6ac50729f89dc7e9e4522.jpg)

<details>
<summary>line</summary>

| Ratio | MCLEA | GEEA  |
|-------|-------|-------|
| 10%   | 0.409 | 0.515 |
| 20%   | 0.582 | 0.661 |
| 30%   | 0.659 | 0.786 |
| 50%   | 0.784 | 0.852 |
| 80%   | 0.890 | 0.918 |
</details>

![](images/0317467ee76b2b3c3a7f2c01895ddb9d1ef78f4e80b0cbde91261873da1baea3.jpg)

<details>
<summary>line</summary>

| Ratio | MCLEA | GEEA  |
|-------|-------|-------|
| 10%   | 0.251 | 0.326 |
| 20%   | 0.393 | 0.450 |
| 30%   | 0.479 | 0.620 |
| 50%   | 0.637 | 0.723 |
| 80%   | 0.790 | 0.836 |
</details>

Figure 4: Entity alignment results on FBDB15K, w.r.t. ratios of training alignment.

# 4.2 ENTITY ALIGNMENT RESULTS

The entity alignment results on DBP15K are shown in Tables 1. Following the existing works (Liu et al., 2021; Lin et al., 2022), we used Hits@1, Hits@10 to measure the proportion of target entities that appear within top 1, top 10, respectively. We also used MRR to measure the reciprocal ranks of the target entities. The multi-modal methods significantly outperformed the single-modal methods, demonstrating the strength of leveraging different resources. Remarkably, our GEEA achieved new state-of-the-art performance on all three datasets across all metrics. The superior performance empirically verified the correlations between the generative objectives and EEA objective. In Table 2, we compared the performance of the multi-modal methods on FB15K-DB15K and FB15K-YAGO15K, where GEEA remained the best-performing method. Nevertheless, we observe that GEEA had more parameters compared with others, as it used VAEs and decoders to decode the embeddings back to concrete features. To probe the effectiveness of GEEA, we reduced the number of neurons to construct a GEEA $_{SMALL}$ and it still outperformed others with a significant margin.

In Figure 3, we plotted the MRR results w.r.t. training epochs on FBDB15K, where MCLEA and GEEA learned much faster than the methods with fewer parameters (i.e., EVA and MSNEA). In Figure 4, we further compared the performance of these two best-performing methods under different ratios of training alignment. We can observe that our GEEA achieved consistent better performance than MCLEA across various settings and metrics. The performance gap was more significantly when there were fewer training entity alignments ( $\leq 30\%$ ). For instance, GEEA surpassed the second-best method by 36.1% in Hits@1 when only 10% aligned entity pairs were used for training.

In summary, the primary weakness of GEEA is its higher parameter count compared to existing methods. However, we demonstrated that a compact version of GEEA still outperformed the baselines in Table 2. This suggests that its potential weakness is manageable. Additionally, GEEA excelled in utilizing training data, achieving greater performance gains with less available training data.

# 4.3 ENTITY SYNTHESIS RESULTS

We conducted entity synthesis experiments by modifying the EEA benchmarks. We randomly selected 30% of the source entities in the testing alignment set as dangling entities, and removed the information of their counterpart entities during training. The goal was to reconstruct the information of their counterpart entities. We evaluated the performance using several metrics: the prior reconstruction error (PRE) for concrete features, the reconstruction error (RE) for the sub-embeddings, and Frechet inception distance (FID) for unconditional synthesis (Heusel et al., 2017). FID is a popular metric for evaluating generative models by measuring the feature distance between real and generated samples.

We implemented several baselines for comparison and present the results in Table 3: MCLEA with the decoders performed worst and it could not generate new entities unconditionally. Using Sub-VAEs to process different modalities performed better than using one VAE to process all modalities. However, the VAEs in Sub-VAEs could not support each other, and sometimes they failed to reconstruct the embeddings (e.g., the RE results on FB15K-DB15K). By contrast, our GEEA consistently and

Table 4: Entity synthesis samples from the FB15K-DB15K dataset. The boldfaced denotes the exactly matched entry, while the underlined denotes the potentially true entry. 

<table><tr><td colspan="2">Source</td><td colspan="3">Target</td><td colspan="3">GEEA Output</td></tr><tr><td>Entity</td><td>Image</td><td>Image</td><td>Neighborhood</td><td>Attribute</td><td>Image</td><td>Neighborhood</td><td>Attribute</td></tr><tr><td>Star Wars (film)</td><td><img src="images/9ec174e5930d7c5302533d3cf23b0f9654e83b09dd7b5e18c943f02bec430c8e.jpg"/></td><td><img src="images/2866562fbe2507a4e1ceb2c99aefacc852276f9cbe82465c13d190aebf517d09.jpg"/></td><td>20th Century Fox, George Lucas, John Williams</td><td>runtime, gross, budget</td><td><img src="images/440f34a001a1bd9b5bf5cbb31ec7dcb993c71106db70a01ac540804ad6afed81.jpg"/></td><td>20th Century Fox, George Lucas, Star Wars: Episode II, Willow (film), Aliens (film), Star Wars: The Clone War</td><td>initial release date, runtime, budget, gross, imdbId, numberOfEpisodes</td></tr><tr><td>George Harrison (musician)</td><td><img src="images/a8e0d6d35a14a9b2c2cff34f613e5881d731fc2db7e56350e5a667122986e01d.jpg"/></td><td><img src="images/5e016608aec5719b0abed5d835de1f057a14cb0855db5afaf3eb8c5cc8b57b57.jpg"/></td><td>The Beatles, Guitar, Rock music, Klaus Voormann, Jeff Lynne, Pop music</td><td>birthDate, deathDate, activeYearsStartYear, activeYearsEndYear, imdbId</td><td><img src="images/c9973e5ab1af95d19b0025dacc5de35d2a29c76042cc491dac6a28addc32ae5f.jpg"/></td><td>The Beatles, The Band, Ringo Starr, Klaus Voormann, Jeff Lynne, Rock music</td><td>deathYear, birthYear, deathDate, birthDate, activeYearsStartYear, activeYearsEndYear, imdbId, height, networth</td></tr></table>

Table 5: Ablation study results on DBP15K $_{ZH-EN}$ .

<table><tr><td rowspan="2">Prediction Match</td><td rowspan="2">Distribution Match</td><td rowspan="2">Prior Reconstruction</td><td rowspan="2">Post Reconstruction</td><td colspan="3">Entity Alignment</td><td colspan="3">Entity Synthesis</td></tr><tr><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td><td>PRE↓</td><td>RE↓</td><td>FID↓</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>.761</td><td>.946</td><td>.827</td><td>0.438</td><td>0.184</td><td>0.935</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>.045</td><td>.186</td><td>.095</td><td>0.717</td><td>0.306</td><td>2.149</td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td>.702</td><td>.932</td><td>.783</td><td>0.551</td><td>0.193</td><td>1.821</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>.746</td><td>.930</td><td>.813</td><td>inf</td><td>0.267</td><td>1.148</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>.750</td><td>.942</td><td>.819</td><td>0.701</td><td>0.246</td><td>1.920</td></tr></table>

significantly outperformed these baselines. We also noticed that the results on FB15K-DB15K and FB15K-YAGO15K were worse than those on DBP15K. This could be due to the larger heterogeneity between two KGs compared to the heterogeneity between two languages of the same KG.

We present some generated samples of GEEA conditioned on the source dangling entities in Table 4. GEEA not only generated samples with the exact information that existed in the target KG, but also completed the target entities with highly reliable predictions. For example, the entity Star Wars (film) in target KG only had three basic attributes in the target KG, but GEEA predicted that it may also have the attributes like imdbid and initial release data.

# 4.4 ABLATION STUDY

We conducted ablation studies to verify the effectiveness of each module in GEEA. In Table 5, we can observe that the best results were achieved by the complete GEEA, and removing any module resulted in a performance loss. Interestingly, GEEA still worked even if we did not employ an EEA loss (the 2nd row) in the entity alignment experiment. It captured alignment information without the explicit optimization of the entity alignment objective through contrastive loss, which is an indispensable module in previous EEA methods. This observation further validates the effectiveness of GEEA.

# 5 RELATED WORKS

Embedding-based Entity Alignment Most pioneer works focus on modeling the relational graph information. They can be divided into triplet-based (Chen et al., 2017; Sun et al., 2017; Pei et al., 2019a) and GNN-based (Wang et al., 2018; Sun et al., 2020a).

Recent methods explore multi-modal KG embedding for EEA (Zhang et al., 2019; Chen et al., 2020; Liu et al., 2021; Chen et al., 2022a;b; Lin et al., 2022). Although GEEA is designed for multi-modal EEA, it differs by focusing on objective optimization rather than specific models. GAN-based methods (Pei et al., 2019a;b; Guo et al., 2022b) are closely related to GEEA but distinct, as GEEA prioritizes the reconstruction process, while the existing methods focus on processing relational graph information for EEA. GEEA can employ these methods for processing relational graph information if necessary. Another distinction is that the existing works do not consider the reconstruction process for the concrete features.

Variational Autoencoder We draw the inspiration from various excellent works, e.g., VAEs, flow-based models, GANs, and diffusion models that have achieved state-of-the-art performance in many fields (Heusel et al., 2017; Kong et al., 2020; Mittal et al., 2021; Nichol & Dhariwal, 2021; Ho et al., 2020; Rombach et al., 2022). Furthermore, recent studies (Austin et al., 2021; Hoogeboom et al., 2021; Li et al., 2022) find that these generative models can be used in controllable text generation. To the best of our knowledge, GEEA is the first method capable of generating new entities

with concrete features. The design of M-VAE, prior and post reconstruction also differs from existing generative models and may offer insights for other domains.

# 6 CONCLUSION

This paper presents a theoretical analysis of how generative models can enhance EEA learning and introduces GEEA to address the limitations of existing GAN-based methods. Experiments demonstrate that GEEA achieves state-of-the-art performance in entity alignment and entity synthesis tasks. Future work will focus on designing new multi-modal encoders to enhance generative ability.

# ACKNOWLEDGMENT

We would like to thank all anonymous reviewers for their insightful and invaluable comments. This work is funded by National Natural Science Foundation of China (NSFCU23B2055/NSFCU19B2027/NSFC91846204), Zhejiang Provincial Natural Science Foundation of China (No.LGG22F030011), Fundamental Research Funds for the Central Universities (226-2023-00138), and the EPSRC project ConCur (EP/V050869/1).

# REFERENCES

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In NeurIPS, pp. 17981–17993, 2021.   
Yixin Cao, Zhiyuan Liu, Chengjiang Li, Zhiyuan Liu, Juanzi Li, and Tat-Seng Chua. Multi-channel graph neural network for entity alignment. In ACL, 2019.   
Liyi Chen, Zhi Li, Yijun Wang, Tong Xu, Zhefeng Wang, and Enhong Chen. MMEA: entity alignment for multi-modal knowledge graph. In KSEM (1), volume 12274, pp. 134–147, 2020.   
Liyi Chen, Zhi Li, Tong Xu, Han Wu, Zhefeng Wang, Nicholas Jing Yuan, and Enhong Chen. Multi-modal siamese network for entity alignment. In KDD, pp. 118–126, 2022a.   
Muhao Chen, Yingtao Tian, Mohan Yang, and Carlo Zaniolo. Multilingual knowledge graph embeddings for cross-lingual knowledge alignment. In IJCAI, 2017.   
Zhuo Chen, Jiaoyan Chen, Wen Zhang, Lingbing Guo, Yin Fang, Yufeng Huang, Yuxia Geng, Jeff Z Pan, Wenting Song, and Huajun Chen. Meaformer: Multi-modal entity alignment transformer for meta modality hybrid. arXiv preprint arXiv:2212.14454, 2022b.   
Lingbing Guo, Weiqing Wang, Zequn Sun, Chenghao Liu, and Wei Hu. Decentralized knowledge graph representation learning. CoRR, abs/2010.08114, 2020.   
Lingbing Guo, Yuqiang Han, Qiang Zhang, and Huajun Chen. Deep reinforcement learning for entity alignment. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Findings of ACL, pp. 2754–2765, 2022a.   
Lingbing Guo, Qiang Zhang, Zequn Sun, Mingyang Chen, Wei Hu, and Huajun Chen. Understanding and improving knowledge graph embedding for entity alignment. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), ICML, volume 162, pp. 8145–8156, 2022b.   
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), NeurIPS, pp. 6626–6637, 2017.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, pp. 6840–6851, 2020.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Towards non-autoregressive language models. arXiv preprint arXiv:2102.05379, 2021.   
Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In ICLR, 2022.   
Ahmed Khalifa, Michael Cerny Green, Diego Perez-Liebana, and Julian Togelius. General video game rule generation. In CIG, pp. 170–177, 2017.   
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020.   
Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics, 22(1):79–86, 1951.   
Lik-Hang Lee, Tristan Braud, Pengyuan Zhou, Lin Wang, Dianlei Xu, Zijun Lin, Abhishek Kumar, Carlos Bermejo, and Pan Hui. All one needs to know about metaverse: A complete survey on technological singularity, virtual ecosystem, and research agenda. arXiv preprint arXiv:2110.05352, 2021.   
Xiang Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B. Hashimoto. Diffusion-lm improves controllable text generation. In NeurIPS, 2022.   
Zhenxi Lin, Ziheng Zhang, Meng Wang, Yinghui Shi, Xian Wu, and Yefeng Zheng. Multi-modal contrastive representation learning for entity alignment. In COLING, pp. 2572–2584, 2022.   
Fangyu Liu, Muhao Chen, Dan Roth, and Nigel Collier. Visual pivoting for (unsupervised) entity alignment. In AAAI, pp. 4257–4266, 2021.   
Juncheng Liu, Zequn Sun, Bryan Hooi, Yiwei Wang, Dayiheng Liu, Baosong Yang, Xiaokui Xiao, and Muhao Chen. Dangling-aware entity alignment with mixed high-order proximities. arXiv preprint arXiv:2205.02406, 2022.   
Shengxuan Luo and Sheng Yu. An accurate unsupervised method for joint entity alignment and dangling entity detection. arXiv preprint arXiv:2203.05147, 2022.   
Gautam Mittal, Jesse Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. arXiv preprint arXiv:2103.16091, March 2021.   
Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. arXiv preprint arXiv:2102.09672, 2021.   
Shichao Pei, Lu Yu, Robert Hoehndorf, and Xiangliang Zhang. Semi-supervised entity alignment via knowledge graph embedding with awareness of degree difference. In WWW, pp. 3130–3136, 2019a.   
Shichao Pei, Lu Yu, and Xiangliang Zhang. Improving cross-lingual entity alignment via optimal transport. In IJCAI, pp. 3231–3237, 2019b.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pp. 10674–10685, 2022.   
Akash Srivastava, Lazar Valkov, Chris Russell, Michael U Gutmann, and Charles Sutton. Veegan: Reducing mode collapse in gans using implicit variational learning. NeurIPS, 30, 2017.   
Zequn Sun, Wei Hu, and Chengkai Li. Cross-lingual entity alignment via joint attribute-preserving embedding. In ISWC, 2017.   
Zequn Sun, Wei Hu, Qingheng Zhang, and Yuzhong Qu. Bootstrapping entity alignment with knowledge graph embedding. In IJCAI, 2018.

Zequn Sun, Chengming Wang, Wei Hu, Muhao Chen, Jian Dai, Wei Zhang, and Yuzhong Qu. Knowledge graph alignment network with gated multi-hop neighborhood aggregation. In AAAI, 2020a.   
Zequn Sun, Qingheng Zhang, Wei Hu, Chengming Wang, Muhao Chen, Farahnaz Akrami, and Chengkai Li. A benchmarking study of embedding-based entity alignment for knowledge graphs. CoRR, abs/2003.07743, 2020b.   
Zequn Sun, Muhao Chen, and Wei Hu. Knowing the no-match: Entity alignment with dangling cases. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 3582–3593, 2021.   
Zhichun Wang, Qingsong Lv, Xiaohan Lan, and Yu Zhang. Cross-lingual knowledge graph alignment via graph convolutional networks. In EMNLP, 2018.   
Qingheng Zhang, Zequn Sun, Wei Hu, Muhao Chen, Lingbing Guo, and Yuzhong Qu. Multi-view knowledge graph embedding for entity alignment. In Sarit Kraus (ed.), IJCAI, pp. 5429–5435, 2019.

# A PROOFS OF THINGS

# A.1 THE COMPLETE PROOF OF PROPOSITION 1

Proof. Let $x \sim X$ , $y \sim Y$ be two entities sampled from the entity sets X, Y, respectively. The main target of EEA is to learn a predictor that estimates the conditional probability $p_{\theta}(\mathbf{x}|\mathbf{y})$ (and reversely $p_{\theta}(\mathbf{y}|\mathbf{x})$ ), where $\theta$ represents the parameter set. For simplicity, we assume that the reverse function $p_{\theta}(\mathbf{y}|\mathbf{x})$ shares the same parameter set with $p_{\theta}(\mathbf{x}|\mathbf{y})$ .

Now, suppose that one wants to learn a generative model for generating entity embeddings:

$$
\log p (\mathbf {x}) = \log p (\mathbf {x}) \int p _ {\theta} (\mathbf {y} | \mathbf {x}) d \mathbf {y} \tag {23}
$$

$$
= \int p _ {\theta} (\mathbf {y} | \mathbf {x}) \log p (\mathbf {x}) d \mathbf {y} \tag {24}
$$

$$
= \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} [ \log p (\mathbf {x}) ] \tag {25}
$$

$$
= \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p (\mathbf {x} , \mathbf {y})}{p (\mathbf {y} | \mathbf {x})} \right] \tag {26}
$$

$$
= \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p (\mathbf {x} , \mathbf {y}) p _ {\theta} (\mathbf {y} | \mathbf {x})}{p (\mathbf {y} | \mathbf {x}) p _ {\theta} (\mathbf {y} | \mathbf {x})} \right] \tag {27}
$$

$$
= \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p (\mathbf {x} , \mathbf {y})}{p _ {\theta} (\mathbf {y} | \mathbf {x})} \right] + \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p _ {\theta} (\mathbf {y} | \mathbf {x})}{p (\mathbf {y} | \mathbf {x})} \right] \tag {28}
$$

$$
= \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p (\mathbf {x} , \mathbf {y})}{p _ {\theta} (\mathbf {y} | \mathbf {x})} \right] + D _ {\mathrm{KL}} (p _ {\theta} (\mathbf {y} | \mathbf {x}) \| p (\mathbf {y} | \mathbf {x})), \tag {29}
$$

where the left-hand side of Equation (7) is the evidence lower bound (ELBO) (Kingma & Welling, 2013), and the right-hand side is the KL divergence (Kullback & Leibler, 1951) between our parameterized distribution $p_{\theta}(\mathbf{y}|\mathbf{x})$ (i.e., the predictor) and the true distribution $p(\mathbf{y}|\mathbf{x})$ .

The recent GAN-based methods (Pei et al., 2019a;b; Guo et al., 2022b) propose to leverage the entities out of training set for unsupervised learning. Their common idea is to make the entity embeddings from different KGs indiscriminative to a discriminator, and the underlying aligned entities shall be encoded in the same way and have similar embeddings. To formally prove this idea, we dissect the ELBO in Equation (7) as follows:

$$
\mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p (\mathbf {x} , \mathbf {y})}{p _ {\theta} (\mathbf {y} | \mathbf {x})} \right] = \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p _ {\theta} (\mathbf {x} | \mathbf {y}) p (\mathbf {y})}{p _ {\theta} (\mathbf {y} | \mathbf {x})} \right] \tag {30}
$$

$$
= \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log p _ {\theta} (\mathbf {x} | \mathbf {y}) \right] + \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log \frac {p (\mathbf {y})}{p _ {\theta} (\mathbf {y} | \mathbf {x})} \right] \tag {31}
$$

$$
= \mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log p _ {\theta} (\mathbf {x} | \mathbf {y}) \right] - D _ {\mathrm{KL}} (p _ {\theta} (\mathbf {y} | \mathbf {x}) \| p (\mathbf {y})) \tag {32}
$$

Therefore, we have:

$$
\log p (\mathbf {x}) = \underbrace {\mathbb {E} _ {p _ {\theta} (\mathbf {y} | \mathbf {x})} \left[ \log p _ {\theta} (\mathbf {x} | \mathbf {y}) \right]} _ {\text { reconstruction   term }} - \underbrace {D _ {\mathrm{KL}} (p _ {\theta} (\mathbf {y} | \mathbf {x}) \| p (\mathbf {y}))} _ {\text { distribution   matching   term }} + \underbrace {D _ {\mathrm{KL}} (p _ {\theta} (\mathbf {y} | \mathbf {x}) \| p (\mathbf {y} | \mathbf {x}))} _ {\text { prediction   matching   term }} \tag {33}
$$

The first term aims to reconstruct the original embedding x based on y generated from x, which has not been studied by the existing discriminative EEA methods (Guo et al., 2020; Liu et al., 2021; Lin et al., 2022). The second term imposes the distribution y conditioned on x to match the prior distribution of y, which has been investigated by the GAN-based EEA methods (Pei et al., 2019a;b; Guo et al., 2022b). The third term is the main objective of EEA (i.e., Equation (8) with the target $p(\mathbf{y}|\mathbf{x})$ being only partially observed).

Note that, $p(\mathbf{x})$ is irrelevant to our parameter set $\theta$ and thus can be regarded as a constant during optimization. Therefore, maximizing the ELBO (i.e., maximizing the first term and minimizing the second term) will result in minimizing the third term, concluding the proof. ☐

# A.2 PROOF OF PROPOSITION 2

Proof. We first have a look on the right hand:

$$
D _ {\mathrm{KL}} \left(p \left(\mathbf {z} _ {x \rightarrow x}\right), p \left(\mathbf {z} ^ {*}\right)\right) + D _ {\mathrm{KL}} \left(p \left(\mathbf {z} _ {y \rightarrow y}\right), p \left(\mathbf {z} ^ {*}\right)\right) \tag {34}
$$

Ideally, all the variables $z_{x\to x}$ , $z_{y\to y}$ , and $z^{*}$ follow the Gaussian distributions with $\mu_{x\to x}$ , $\mu_{y\to y}$ , $\mu^{*}$ and $\sigma_{x\to x}$ , $\sigma_{y\to y}$ , $\sigma^{*}$ as mean and variance, respectively.

Luckily, we can use the following equation to calculate the KL divergence between two Gaussian distributions conveniently:

$$
D _ {\mathrm{KL}} (p (\mathbf {z} _ {1}), p (\mathbf {z} _ {2})) = \log \frac {\sigma_ {2}}{\sigma_ {1}} + \frac {\sigma_ {1} ^ {2} + (\mu_ {1} - \mu_ {2}) ^ {2}}{2 \sigma_ {2} ^ {2}} - \frac {1}{2}, \tag {35}
$$

and rewrite Equation(34) as:

$$
D _ {\mathrm{KL}} \left(p \left(\mathbf {z} _ {x \rightarrow x}\right), p \left(\mathbf {z} ^ {*}\right)\right) + D _ {\mathrm{KL}} \left(p \left(\mathbf {z} _ {y \rightarrow y}\right), p \left(\mathbf {z} ^ {*}\right)\right) \tag {36}
$$

$$
= (\log \frac {\sigma^ {*}}{\sigma_ {x \rightarrow x}} + \frac {\sigma_ {x \rightarrow x} ^ {2} + (\mu_ {x \rightarrow x} - \mu^ {*}) ^ {2}}{2 (\sigma^ {*}) ^ {2}} - \frac {1}{2}) + (\log \frac {\sigma^ {*}}{\sigma_ {y \rightarrow y}} + \frac {\sigma_ {y \rightarrow y} ^ {2} + (\mu_ {y \rightarrow y} - \mu^ {*}) ^ {2}}{2 (\sigma^ {*}) ^ {2}} - \frac {1}{2}) (3 7)
$$

$$
= \left(\log \frac {\sigma^ {*}}{\sigma_ {x \rightarrow x}} + \log \frac {\sigma^ {*}}{\sigma_ {y \rightarrow y}}\right) + \left(\frac {\sigma_ {x \rightarrow x} ^ {2} + \left(\mu_ {x \rightarrow x} - \mu^ {*}\right) ^ {2}}{2 \left(\sigma^ {*}\right) ^ {2}} + \frac {\sigma_ {y \rightarrow y} ^ {2} + \left(\mu_ {y \rightarrow y} - \mu^ {*}\right) ^ {2}}{2 \left(\sigma^ {*}\right) ^ {2}}\right) - 1 \tag {38}
$$

$$
= \left(\log \frac {\sigma^ {*}}{\sigma_ {x \rightarrow x}} + \log \frac {\sigma^ {*}}{\sigma_ {y \rightarrow y}}\right) + \frac {\sigma_ {x \rightarrow x} ^ {2} + \left(\mu_ {x \rightarrow x} - \mu^ {*}\right) ^ {2} + \sigma_ {y \rightarrow y} ^ {2} + \left(\mu_ {y \rightarrow y} - \mu^ {*}\right) ^ {2}}{2 \left(\sigma^ {*}\right) ^ {2}} - 1 \tag {39}
$$

Take $\mathbf{z}^{*}\sim \mathcal{N}(\mu^{*} = \mathbf{0},\sigma^{*} = \mathbf{I})$ into the above equation, we will have:

$$
D _ {\mathrm{KL}} (p (\mathbf {z} _ {x \rightarrow x}), p (\mathbf {z} ^ {*})) + D _ {\mathrm{KL}} (p (\mathbf {z} _ {y \rightarrow y}), p (\mathbf {z} ^ {*})) \tag {40}
$$

$$
= - \log \sigma_ {x \rightarrow x} \sigma_ {y \rightarrow y} + \frac {1}{2} (\sigma_ {x \rightarrow x} ^ {2} + \sigma_ {y \rightarrow y} ^ {2} + \mu_ {x \rightarrow x} ^ {2} + \mu_ {y \rightarrow y} ^ {2}) - 1 \tag {41}
$$

Similarly, the left hand can be expanded as:

$$
D _ {\mathrm{KL}} (p (\mathbf {z} _ {x \rightarrow x}), p (\mathbf {z} _ {y \rightarrow y})) = \log \frac {\sigma_ {y \rightarrow y}}{\sigma_ {x \rightarrow x}} + \frac {\sigma_ {x \rightarrow x} ^ {2} + (\mu_ {x \rightarrow x} - \mu_ {y \rightarrow y}) ^ {2}}{2 \sigma_ {y \rightarrow y} ^ {2}} - \frac {1}{2}, \tag {42}
$$

Thus, the difference between the left hand and the right hand can be computed:

$$
D _ {\mathrm{KL}} \left(p \left(\mathbf {z} _ {x \rightarrow x}\right), p \left(\mathbf {z} ^ {*}\right)\right) + D _ {\mathrm{KL}} \left(p \left(\mathbf {z} _ {y \rightarrow y}\right), p \left(\mathbf {z} ^ {*}\right)\right) - D _ {\mathrm{KL}} \left(p \left(\mathbf {z} _ {x \rightarrow x}\right), p \left(\mathbf {z} _ {y \rightarrow y}\right)\right) \tag {43}
$$

$$
= - \log \sigma_ {x \rightarrow x} \sigma_ {y \rightarrow y} + \frac {1}{2} (\sigma_ {x \rightarrow x} ^ {2} + \sigma_ {y \rightarrow y} ^ {2} + \mu_ {x \rightarrow x} ^ {2} + \mu_ {y \rightarrow y} ^ {2}) - 1 \tag {44}
$$

$$
- \left(\log \frac {\sigma_ {y \rightarrow y}}{\sigma_ {x \rightarrow x}} + \frac {\sigma_ {x \rightarrow x} ^ {2} + \left(\mu_ {x \rightarrow x} - \mu_ {y \rightarrow y}\right) ^ {2}}{2 \sigma_ {y \rightarrow y} ^ {2}} - \frac {1}{2}\right) \tag {45}
$$

$$
= \left(- \log \sigma_ {x \rightarrow x} \sigma_ {y \rightarrow y} - \log \frac {\sigma_ {y \rightarrow y}}{\sigma_ {x \rightarrow x}}\right) \tag {46}
$$

$$
+ \left(\frac {1}{2} (\sigma_ {x \to x} ^ {2} + \sigma_ {y \to y} ^ {2} + \mu_ {x \to x} ^ {2} + \mu_ {y \to y} ^ {2}) - \frac {\sigma_ {x \to x} ^ {2} + (\mu_ {x \to x} - \mu_ {y \to y}) ^ {2}}{2 \sigma_ {y \to y} ^ {2}}\right) + (- 1 + \frac {1}{2}) \tag {47}
$$

$$
= - 2 \log \sigma_ {y \rightarrow y} - \frac {1}{2} \tag {48}
$$

$$
+ \frac {\sigma_ {x \rightarrow x} ^ {2} \sigma_ {y \rightarrow y} ^ {2} + \sigma_ {y \rightarrow y} ^ {4} + \mu_ {x \rightarrow x} ^ {2} \sigma_ {y \rightarrow y} ^ {2} + \mu_ {y \rightarrow y} ^ {2} \sigma_ {y \rightarrow y} ^ {2} - \sigma_ {x \rightarrow x} ^ {2} - \mu_ {x \rightarrow x} ^ {2} - \mu_ {y \rightarrow y} ^ {2} + 2 \mu_ {x \rightarrow x} \mu_ {y \rightarrow y}}{2 \sigma_ {y \rightarrow y} ^ {2}} \tag {49}
$$

$$
= - 2 \log \sigma_ {y \rightarrow y} - \frac {1}{2} \tag {50}
$$

$$
+ \frac {\left(\sigma_ {y \rightarrow y} ^ {2} - 1\right) \sigma_ {x \rightarrow x} ^ {2} + \left(\mu_ {x \rightarrow x} ^ {2} + \mu_ {y \rightarrow y} ^ {2}\right)\left(\sigma_ {y \rightarrow y} ^ {2} - 1\right) + \sigma_ {y \rightarrow y} ^ {4} + 2 \mu_ {x \rightarrow x} \mu_ {y \rightarrow y}}{2 \sigma_ {y \rightarrow y} ^ {2}} \tag {51}
$$

$$
= - 2 \log \sigma_ {y \rightarrow y} - \frac {1}{2} + \frac {\left(\mu_ {x \rightarrow x} ^ {2} + \mu_ {y \rightarrow y} ^ {2} + \sigma_ {x \rightarrow x} ^ {2}\right)\left(\sigma_ {y \rightarrow y} ^ {2} - 1\right) + \sigma_ {y \rightarrow y} ^ {4} + 2 \mu_ {x \rightarrow x} \mu_ {y \rightarrow y}}{2 \sigma_ {y \rightarrow y} ^ {2}} \tag {52}
$$

As we optimize $z_{y\to y} \rightarrow z^{*}$ , i.e., minimize $D_{\mathrm{KL}}(p(\mathbf{z}_{y\to y}), p(\mathbf{z}^{*}))$ , we will have:

$$
\log \sigma_ {y \rightarrow y} \rightarrow 0, \quad \sigma_ {y \rightarrow y} ^ {2} - 1 \rightarrow 0, \quad \mu_ {x \rightarrow x} \mu_ {y \rightarrow y} \rightarrow 0, \quad \sigma_ {y \rightarrow y} ^ {4} \rightarrow 1, \tag {53}
$$

and consequently:

$$
D _ {\mathrm{KL}} (p (\mathbf {z} _ {x \rightarrow x}), p (\mathbf {z} ^ {*})) + D _ {\mathrm{KL}} (p (\mathbf {z} _ {y \rightarrow y}), p (\mathbf {z} ^ {*})) - D _ {\mathrm{KL}} (p (\mathbf {z} _ {x \rightarrow x}), p (\mathbf {z} _ {y \rightarrow y})) \rightarrow 0, \tag {54}
$$

Similarly, as we optimize $z_{x\to x} \to z^{*}$ , i.e., minimize $D_{\mathrm{KL}}(p(\mathbf{z}_{x\to x}), p(\mathbf{z}^{*}))$ , we will have:

$$
D _ {\mathrm{KL}} (p (\mathbf {z} _ {x \rightarrow x}), p (\mathbf {z} ^ {*})) + D _ {\mathrm{KL}} (p (\mathbf {z} _ {y \rightarrow y}), p (\mathbf {z} ^ {*})) - D _ {\mathrm{KL}} (p (\mathbf {z} _ {y \rightarrow y}), p (\mathbf {z} _ {x \rightarrow x})) \rightarrow 0 \tag {55}
$$

Therefore, jointly minimizing $D_{\mathrm{KL}}(p(\mathbf{z}_{x \to x}), p(\mathbf{z}^{*}))$ and $D_{\mathrm{KL}}(p(\mathbf{z}_{y \to y}), p(\mathbf{z}^{*}))$ will subsequently minimizing $D_{\mathrm{KL}}(p(\mathbf{z}_{x \to x}), p(\mathbf{z}_{y \to y}))$ and $D_{\mathrm{KL}}(p(\mathbf{z}_{y \to y}), p(\mathbf{z}_{x \to x}))$ , and finally aligning the distributions between x and y, concluding the proof. □

# B IMPLEMENTATION DETAILS

# B.1 DECODING EMBEDDINGS BACK TO CONCRETE FEATURES

All decoders used to decode the reconstructed embeddings to the concrete features comprise several hidden layers and an output layer. Specifically, each hidden layer has a linear layer with layer norm and ReLU/Tanh activations. The output layer is different for different modalities. For the relational graph and attribute information, their concrete features are organized in the form of multi-classification labels. For example, the relational graph information $g_{i}$ for an entity $x_{i}$ is represented by:

$$
g _ {i} = (0,..., 1,... 1,..., 0) ^ {T}, \quad | g _ {i} | = | \mathcal {X} |, \tag {56}
$$

where $g_{i}$ has $|X|$ elements with 1 indicating the connection and 0 otherwise. Therefore, the output layer transforms the hidden output to the concrete feature prediction with a matrix $W_{o} \in R^{H \times |X|}$ , where H is the output dimension of the final hidden layer.

The image concrete features are actually the pretrained embeddings rather than pixel data, as we use the existing EEA models for embedding entities. Therefore, we replaced the binary cross-entropy loss with a MSE loss to train GEEA to recover this pretrained embedding.

# B.2 IMPLEMENTING A GEEA

We implement GEEA with PyTorch and run the main experiments on a RTX 4090. We illustrate the training procedure of GEEA as outlined in Algorithm 1. We first initialize all trainable variables and the get the mini-batch data of supervised flows $x \rightarrow y$ , $y \rightarrow x$ and unsupervised flows $x \rightarrow x$ , $y \rightarrow y$ , respectively.

For the supervised flows, we iterate the batched data and calculate the prediction matching loss which is also used in most existing works. Then, we calculate the distribution matching, prior reconstruction and post reconstruction losses and sum them for later joint optimization.

For the unsupervised flows, we first process the raw feature with M and VAE to obtain the embeddings and reconstructed embeddings. Then we estimate the distribution matching loss with the embedding sets as input (Equation (16)), after which we calculate the prior and post reconstruction loss for each x and each y.

Finally, we sum all the losses produced with all flows, and minimize them until the performance on the valid dataset does not improve.

The overall hyper-parameter settings in the main experiments are presented in Table 6.

# C ADDITIONAL EXPERIMENTS

# C.1 DATASETS

We present the statistics of entity alignment and entity synthesis datasets in Table 7. To construct an entity synthesis dataset, we first sample $30\%$ of entity alignments from the testing set of the original

Algorithm 1 Generative Embedding-based Entity Alignment   
1: Input: The entity sets $\mathcal{X},\mathcal{Y}$ , the multi-modal information $\mathcal{G},\mathcal{A},\mathcal{I}\dots$ , the EEA model $\mathcal{M}$ , and M-VAE VAE;
2: Randomly initialize all parameters;
3: repeat
4: $\mathcal{B}_{\mathrm{sup}}\gets \{(x,y)|(x,y)\sim \mathcal{S}\}$ ; // get a batch of supervised training data
5: $\mathcal{B}_{\mathrm{unsup}}\gets \{(x,y)|x\sim \mathcal{X},y\sim \mathcal{Y}\}$ ; // get a batch of unsupervised training data
6: for $(x,y)\in \mathcal{B}_{\mathrm{sup}}$ do
7: $\mathbf{x},\mathbf{y}\gets \mathcal{M}(x),\mathcal{M}(y)$ ; // obtain embeddings and sub-embeddings
8: $\mathbf{y}_{x\to y},\mathbf{x}_{y\to x}\gets VAE(\mathbf{x}),VAE(\mathbf{y})$ ; // obtain the reconstructed mutual embeddings
9: Calculate the prediction matching loss following Equation (8);
10: Calculate the prior reconstruction loss following Equation (19);
11: Calculate the post reconstruction loss following Equation (20);
12: end for
13: $\{\mathbf{x}_{x\to x},\mathbf{y}_{y\to y}|(x,y)\in \mathcal{B}_{\mathrm{unsup}}\} \gets \{\mathrm{VAE}(\mathcal{M}(x)),\mathrm{VAE}(\mathcal{M}(y))|(x,y)\in \mathcal{B}_{\mathrm{unsup}}\}$ ; // obtain the reconstructed self embeddings
14: Calculate the distribution matching loss following Equation (16);
15: Calculate the prior reconstruction loss following Equation (19);
16: Calculate the post reconstruction loss following Equation (20);
17: Jointly minimize all losses;
18: until the performance does not improve.

Table 6: Hyper-parameter settings in the main experiments. PM, DM, PrioR, PostR denote prediction matching, distribution matching, prior reconstruction, and post reconstruction, respectively. 

<table><tr><td>Datasets</td><td># epoch</td><td>batch-size</td><td># VAE layers</td><td>learning rate</td><td>optimizer</td><td>dropout rate</td><td>unsupervised batch-size</td><td>flow weights (xx,yy,xy,yyx)</td><td>loss weights (PM,DM, Prior, PostR)</td><td>hidden sizes</td><td>latent size</td><td>decoder hidden sizes</td></tr><tr><td>DBP15KZH-EN</td><td>200</td><td>2,500</td><td>2</td><td>0.001</td><td>Adam</td><td>0.5</td><td>2,800</td><td>[1.,1..5..5.]</td><td>[1., 0.5,1.,1.]</td><td>[300,300]</td><td>300</td><td>[300,1000]</td></tr><tr><td>DBP15KJA-EN</td><td>200</td><td>2,500</td><td>2</td><td>0.001</td><td>Adam</td><td>0.5</td><td>2,800</td><td>[1.,1..5..5.]</td><td>[1., 0.5,1.,1.]</td><td>[300,300]</td><td>300</td><td>[300,1000]</td></tr><tr><td>DBP15KTR-EN</td><td>200</td><td>2,500</td><td>2</td><td>0.001</td><td>Adam</td><td>0.5</td><td>2,800</td><td>[1.,1..5..5.]</td><td>[1., 0.5,1.,1.]</td><td>[300,300]</td><td>300</td><td>[300,1000]</td></tr><tr><td>FB15K-DB15K</td><td>300</td><td>3,500</td><td>3</td><td>0.0005</td><td>Adam</td><td>0.5</td><td>2,500</td><td>[1.,1..5..5.]</td><td>[1., 0.5,1.,1.]</td><td>[300,300,300]</td><td>300</td><td>[300,300,1000]</td></tr><tr><td>FB15K-YAGO15K</td><td>300</td><td>3,500</td><td>3</td><td>0.0005</td><td>Adam</td><td>0.5</td><td>2,500</td><td>[1.,1..5..5.]</td><td>[1., 0.5,1.,1.]</td><td>[300,300,300]</td><td>300</td><td>[300,300,1000]</td></tr></table>

Table 7: Statistics of the datasets. 

<table><tr><td rowspan="2">Datasets</td><td>Entity Alignment</td><td colspan="2">Entity Synthesis</td><td rowspan="2"># Entities</td><td rowspan="2"># Relations</td><td rowspan="2"># Attributes</td><td rowspan="2"># Images</td></tr><tr><td># Test Alignments</td><td># Known Test Alignments</td><td># Unknown Test Alignments</td></tr><tr><td rowspan="2">DBP15KZH-EN</td><td>10,500</td><td>7,350</td><td>3,150</td><td>19,388</td><td>1,701</td><td>8,111</td><td>15,912</td></tr><tr><td>10,500</td><td>7,350</td><td>3,150</td><td>19,572</td><td>1,323</td><td>7,173</td><td>14,125</td></tr><tr><td rowspan="2">DBP15KJA-EN</td><td>10,500</td><td>7,350</td><td>3,150</td><td>19,814</td><td>1,299</td><td>5,882</td><td>12,739</td></tr><tr><td>10,500</td><td>7,350</td><td>3,150</td><td>19,780</td><td>1,153</td><td>6,066</td><td>13,741</td></tr><tr><td rowspan="2">DBP15KFR-EN</td><td>10,500</td><td>7,350</td><td>3,150</td><td>19,661</td><td>903</td><td>4,547</td><td>14,174</td></tr><tr><td>10,500</td><td>7,350</td><td>3,150</td><td>19,993</td><td>1,208</td><td>6,422</td><td>13,858</td></tr><tr><td rowspan="2">FB15K-DB15K</td><td>10,276</td><td>7,193</td><td>3,083</td><td>14,951</td><td>1,345</td><td>116</td><td>13,444</td></tr><tr><td>10,500</td><td>7,350</td><td>3,150</td><td>12,842</td><td>279</td><td>225</td><td>12,837</td></tr><tr><td rowspan="2">FB15K-YAGO15K</td><td>8,959</td><td>6,272</td><td>2,687</td><td>14,951</td><td>1,345</td><td>116</td><td>13,444</td></tr><tr><td>10,500</td><td>7,350</td><td>3,150</td><td>15,404</td><td>32</td><td>7</td><td>11,194</td></tr></table>

entity alignment dataset. Then, we view the source entities in sampled entities pairs as the dangling entities, and make their target entities unseen during training. To this end, we remove all types of information referred to these target entities from the training set.

# C.2 SINGLE-MODAL GEEA

We first remove the image encoder from multi-modal EEA models. The results are shown in Table 11. Notably, our GEEA without the image encoder still achieves state-of-the-art performance on several metrics, such as Hits@10.

Then, we conducted new experiments on the OpenEA 100K (Sun et al., 2020b). Although OpenEA 100K does not have a multi-modal version, it is still interesting to explore the performance of GEEA with single-modal EEA models on it, similar to NeoEA (Guo et al., 2022b). We conducted experiments following the NeoEA and present the results in Table 12. It is clear that our method can

Table 8: Entity alignment results of GEEA with different EEA models on DBP15K datasets. 

<table><tr><td rowspan="2">Models</td><td colspan="3">DBP15KZH-EN</td><td colspan="3">DBP15KJA-EN</td><td colspan="3">DBP15KFR-EN</td></tr><tr><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td></tr><tr><td>EVA (Liu et al., 2021)</td><td>.680</td><td>.910</td><td>.762</td><td>.673</td><td>.908</td><td>.757</td><td>.683</td><td>.923</td><td>.767</td></tr><tr><td>GEEA w/ EVA</td><td>.715</td><td>.922</td><td>.794</td><td>.707</td><td>.925</td><td>.791</td><td>.727</td><td>.940</td><td>.817</td></tr><tr><td>MSNEA (Chen et al., 2022a)</td><td>.601</td><td>.830</td><td>.684</td><td>.535</td><td>.775</td><td>.617</td><td>.543</td><td>.801</td><td>.630</td></tr><tr><td>GEEA w/ MSNEA</td><td>.643</td><td>.872</td><td>.732</td><td>.559</td><td>.821</td><td>.671</td><td>.586</td><td>.853</td><td>.672</td></tr><tr><td>MCLEA (Lin et al., 2022)</td><td>.715</td><td>.923</td><td>.788</td><td>.715</td><td>.909</td><td>.785</td><td>.711</td><td>.909</td><td>.782</td></tr><tr><td>GEEA w/ MCLEA</td><td>.761</td><td>.946</td><td>.827</td><td>.755</td><td>.953</td><td>.827</td><td>.776</td><td>.962</td><td>.844</td></tr></table>

Table 9: More entity synthesis samples from different dataset. The first two rows are from FB15K-YAGO15K; the middle two rows are from DBP15K $_{ZH-EN}$ ; the last two rows are from DBP15K $_{FR-EN}$ .

<table><tr><td colspan="2">Source</td><td colspan="6">Target</td><td colspan="101">GEEA Output</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Entity</td><td>Image</td><td>Image</td><td colspan="5">Neighborhood</td><td>Attribute</td><td>Image</td><td colspan="100">Neighborhood</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>James Cameron (director)</td><td><img src="images/5bffd07c47b12683774d8bc32449d0d97d6340d057110ceded5088bcb3608291.jpg"/></td><td><img src="images/bf7aa96e4b8ff11d21210c3d04db81cb89984105753b0208c9dbd3eb7f7cf40c.jpg"/></td><td colspan="5">United States, New Zealand, Kathryn Bigelow, Avatar (2009 film), The Terminator</td><td>wasBornOnDate</td><td><img src="images/e400a23fb1177c4f51173ea5a1d273b64426919a3d0451cc40145e407dc02e78.jpg"/></td><td colspan="6"><img src="images/7c4f46b79af2197cc78b891eb8f078d8c0e2fdfe81c32bae668669f19664bffc.jpg"/></td><td><img src="images/c969d7954f1af182a877d29403a79e86b928b63ca92c16f0f5f815279f782b81.jpg"/></td><td>[168D7]</td><td>BhJ</td><td>[KASH]</td><td>athe</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>$</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>##</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>@</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>###</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>#</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>#######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>#####</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>####</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>########</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>*******</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>indexes</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>##</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>###</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>######</td><td>Yellowstone 3DS</td></tr></table>

Table 10: Some false samples from FB15K-YAGO15K. 

<table><tr><td colspan="2">Source</td><td colspan="3">Target</td><td colspan="3">GEEA Output</td></tr><tr><td>Entity</td><td>Image</td><td>Image</td><td>Neighborhood</td><td>Attribute</td><td>Image</td><td>Neighborhood</td><td>Attribute</td></tr><tr><td>The Matrix (film)</td><td><img src="images/7eab3ccd7bdded016b96be34a95d7001fd809655e1a7e0e759b120a0e2985ad1.jpg"/></td><td><img src="images/c2d255cc9e5601d8daa9f80dcc0ab166e2e7c7c03641fedf88dc81f28b7516aa.jpg"/></td><td>Carrie-Anne Moss, et al., [1982]</td><td>SC, created, and Za</td><td>[1982]</td><td>Fistu, J. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M. M.</td><td>wasOnDate, diedOnDate,</td></tr><tr><td>The Terminator (film)</td><td><img src="images/167cbf8b25fab1e5764e3dcdca36608dafe911e5523a3069c7cb39f18c916302.jpg"/></td><td><img src="images/afeba648ec169649d18554b9a7f68b87ff33570e96bbd0d639b89e82e112c0b5.jpg"/></td><td>United States, Michael Biehn, James Cameron, A. M. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S. S.</td><td>wasCreatedOnDate</td><td>[1982]</td><td>United States, Nicaraguan Revolution, Ibad Muhamadu, Jose Rodrigues Neto, Anaconda (film)</td><td>wasCreatedOnDate, diedOnDate, wasCreatedOnDate, wasDestroyedOnDate,</td></tr></table>

significantly enhance the performance of SEA (Pei et al., 2019a), which can be attributed to the more stringent objectives analyzed in Section 2.

# C.3 GEEA ON DANGLING ENTITY DETECTION

The methods for detecting dangling entities (Sun et al., 2021) can be categorized into three groups: (1) Nearest Neighbor Classifier (NNC), which trains a classifier to determine whether a source entity lacks a counterpart entity in the target KG; (2) Margin-based Ranking (MR), which learns a margin value $\lambda$ . If the embedding distance between a source entity and its nearest neighbor in the target KG exceeds $\lambda$ , this source entity is considered a dangling entity; (3) Background Ranking (BR), which regards the dangling entities as background and randomly pushes them away from aligned entities.

All three types of dangling detection methods heavily rely on the quality of entity embeddings. Therefore, if the proposed GEEA learns better embeddings for entity alignment, it is expected to contribute to the detection of dangling entities. To verify this idea, we conducted experiments on new datasets, following $[1]$ . We used the same parameter settings and employed MTransE (Chen et al., 2017) as the backbone model. The results are presented in Table 13. Clearly, incorporating GEEA led to significant performance improvements in dangling entity detection across all three datasets. The performance gains were particularly notable in terms of precision and F1 metrics.

# C.4 GEEA WITH DIFFERENT EEA MODELS

We also investigated the performance of GEEA with different EEA models. As shown in Table 8, GEEA significantly improved all the baseline models on all metrics and datasets. Remarkably, the

Table 11: Detailed entity alignment results on DBP15K datasets, without surface information and iterative strategy. 

<table><tr><td rowspan="2">Models</td><td colspan="3">DBP15KZH-EN</td><td colspan="3">DBP15KJA-EN</td><td colspan="3">DBP15KFR-EN</td></tr><tr><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td><td>Hits@1↑</td><td>Hits@10↑</td><td>MRR↑</td></tr><tr><td>EVA (Liu et al., 2021)</td><td>.680</td><td>.910</td><td>.762</td><td>.673</td><td>.908</td><td>.757</td><td>.683</td><td>.923</td><td>.767</td></tr><tr><td>MSNEA (Chen et al., 2022a)</td><td>.601</td><td>.830</td><td>.684</td><td>.535</td><td>.775</td><td>.617</td><td>.543</td><td>.801</td><td>.630</td></tr><tr><td>MCLEA (Lin et al., 2022)</td><td>.715</td><td>.923</td><td>.788</td><td>.715</td><td>.909</td><td>.785</td><td>.711</td><td>.909</td><td>.782</td></tr><tr><td>MCLEA w/o image</td><td>.658</td><td>.915</td><td>.726</td><td>.662</td><td>.904</td><td>.740</td><td>.662</td><td>.902</td><td>.747</td></tr><tr><td>GEEA</td><td>.761</td><td>.946</td><td>.827</td><td>.755</td><td>.953</td><td>.827</td><td>.776</td><td>.962</td><td>.844</td></tr><tr><td>GEEA w/o image</td><td>.709</td><td>.929</td><td>.782</td><td>.708</td><td>.935</td><td>.784</td><td>.717</td><td>.946</td><td>.796</td></tr></table>

Table 12: Single-modal entity alignment results on OpenEA 100K datasets.

<table><tr><td rowspan="2">Models</td><td colspan="2">EN-FR</td><td colspan="2">EN-DE</td><td colspan="2">DBPedia-WikiData</td><td colspan="2">DBPedia-Yago</td></tr><tr><td>Hits@1↑</td><td>MRR↑</td><td>Hits@1↑</td><td>MRR↑</td><td>Hits@1↑</td><td>MRR↑</td><td>Hits@1↑</td><td>MRR↑</td></tr><tr><td>SEA (Pei et al., 2019a)</td><td>.225</td><td>.314</td><td>.341</td><td>.421</td><td>.291</td><td>.378</td><td>.490</td><td>.578</td></tr><tr><td>NeoEA (SEA) (Guo et al., 2022b)</td><td>.254</td><td>.345</td><td>.364</td><td>.446</td><td>.325</td><td>.416</td><td>.569</td><td>.651</td></tr><tr><td>GEEA (SEA)</td><td>.269</td><td>.355</td><td>.377</td><td>.459</td><td>.349</td><td>.436</td><td>.597</td><td>.685</td></tr></table>

performance of EVA with GEEA on some datasets like DBP15K $_{FR-EN}$ were even better than that of the original MCLEA.

# C.5 RESULTS WITH DIFFERENT ALIGNMENT RATIOS ON ALL DATASETS

We present the results with different alignment ratios on all datasets in Figure 5, which demonstrate the same conclusion as in Figure 4.

# C.6 MORE ENTITY SYNTHESIS SAMPLES

We illustrate more entity synthesis samples in Table 9 and some false samples in Table 10. The main reason for less accurate synthesis results is the lack of information. For example, in the FB15K-YAGO15K datasets, the YAGO KG has only 7 different attributes. Also, as some entities do not have image features, the EEA models are configured to initialize the pretrained image embeddings with random vectors. To mitigate this problem, we plan to design new benchmarks and new EEA models to directly process and generate the raw data in future work.

Table 13: Dangling entity detection results on DBP2.0. 

<table><tr><td rowspan="2">Models</td><td colspan="3">DBP 2.0ZH-EN</td><td colspan="3">DBP 2.0JA-EN</td><td colspan="3">DBP 2.0FR-EN</td></tr><tr><td>Precision↑</td><td>Recall↑</td><td>F1↑</td><td>Precision↑</td><td>Recall↑</td><td>F1↑</td><td>Precision↑</td><td>Recall↑</td><td>F1↑</td></tr><tr><td>NNC (Sun et al., 2021)</td><td>.604</td><td>.485</td><td>.538</td><td>.622</td><td>.491</td><td>.549</td><td>.459</td><td>.447</td><td>.453</td></tr><tr><td>GEEA (NNC)</td><td>.617</td><td>.509</td><td>.558</td><td>.637</td><td>.460</td><td>.534</td><td>.479</td><td>.449</td><td>.464</td></tr><tr><td>MR (Sun et al., 2021)</td><td>.781</td><td>.702</td><td>.740</td><td>.799</td><td>.708</td><td>.751</td><td>.482</td><td>.575</td><td>.524</td></tr><tr><td>GEEA (MR)</td><td>.793</td><td>.709</td><td>.749</td><td>.812</td><td>.714</td><td>.760</td><td>.508</td><td>.594</td><td>.548</td></tr><tr><td>BR (Sun et al., 2021)</td><td>.811</td><td>.728</td><td>.767</td><td>.816</td><td>.733</td><td>.772</td><td>.539</td><td>.686</td><td>.604</td></tr><tr><td>GEEA (BR)</td><td>.821</td><td>.724</td><td>.769</td><td>.833</td><td>.735</td><td>.781</td><td>.549</td><td>.694</td><td>.613</td></tr></table>

![](images/6e034a6de957fcc0e4fc604dee965a0dc24fafe71568355a629ddddf4177d241.jpg)  
Figure 5: Entity alignment results on all datasets, w.r.t. ratios of training alignment.