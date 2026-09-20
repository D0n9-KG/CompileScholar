# AUTOVP: AN AUTOMATED VISUAL PROMPTING FRAMEWORK AND BENCHMARK

Hsi-Ai Tsao $^{1*}$ , Lei Hsiung $^{2*}$ , Pin-Yu Chen $^{3}$ , Sijia Liu $^{4}$ , Tsung-Yi Ho $^{5}$

$^{1}$ National Tsing Hua University, $^{2}$ Dartmouth College, $^{3}$ IBM Research

$^{4}$ Michigan State University, $^{5}$ The Chinese University of Hong Kong

\* Equal contribution

# ABSTRACT

Visual prompting (VP) is an emerging parameter-efficient fine-tuning approach to adapting pre-trained vision models to solve various downstream image-classification tasks. However, there has hitherto been little systematic study of the design space of VP and no clear benchmark for evaluating its performance. To bridge this gap, we propose AutoVP, an end-to-end expandable framework for automating VP design choices, along with 12 downstream image-classification tasks that can serve as a holistic VP-performance benchmark. Our design space covers 1) the joint optimization of the prompts; 2) the selection of pre-trained models, including image classifiers and text-image encoders; and 3) model output mapping strategies, including nonparametric and trainable label mapping. Our extensive experimental results show that AutoVP outperforms the best-known current VP methods by a substantial margin, having up to $6.7\%$ improvement in accuracy; and attains a maximum performance increase of $27.5\%$ compared to linear-probing (LP) baseline. AutoVP thus makes a two-fold contribution: serving both as an efficient tool for hyperparameter tuning on VP design choices, and as a comprehensive benchmark that can reasonably be expected to accelerate VP's development. The source code is available at https://github.com/IBM/AutoVP.

# 1 INTRODUCTION

Originating in the domain of natural language processing, prompting (Gao et al., 2021; Lester et al., 2021; Shi et al., 2023) has gained considerable popularity as a parameter-efficient fine-tuning approach for adapting pre-trained models to downstream tasks. Prompting's methodology has recently been extended to the field of computer vision, where it is termed visual prompting (VP) (Bahng et al., 2022). In its simplest form, VP can be perceived as an in-domain model-reprogramming technique (Chen, 2022). More specifically, it adjusts the inputs and outputs of a pre-trained vision model to address downstream image-classification tasks, without having to make any changes to the weights or architecture of the source model's pre-trained backbone. As such, it stands in contrast to conventional transfer-learning methods that involve complete fine-tuning, LP (i.e., involving modifications of the trainable linear layer in the penultimate layer's output), or zero-shot learning (Radford et al., 2021). For instance, as illustrated in Figure 1, VP adds a universal trainable data frame to the target samples at the model-input stage, and then employs a mapping function – which can be either explicitly defined or implicitly learned – to associate the source and target labels at the output stage.

While VP exhibits tremendous potential, there are two critical challenges that limit its research and development. The first is absence of a systematic VP framework. That is, VP design choices, such as prompts' sizes and shapes, source models, and label-mapping (LM) strategies, have thus far only been studied separately, generally for the purpose of delineating their distinct roles in enhancing downstream task accuracy. Ideally, such a systematic framework would automatically search for the best configurations for performance optimization. For example, Bahng et al. (2022) have demonstrated that changing the padding size of visual prompts can yield around 15% variation in final accuracy. It has also been observed that VP is better at augmenting large text-image models, such as CLIP (Radford et al., 2021), than pure vision models like ResNet50 (He et al., 2016). In a study by Chen et al. (2023b), iterative label mapping (ILM) during training achieved accuracy up to

![](images/42632ac8cf5528b01ed88a61b4a91c0c6c1a911277305eda226015ecd0e89eb7.jpg)

<details>
<summary>bar</summary>

| Dataset | AutoVP (%) | ILM-VP (%) |
|---|---|---|
| EuroSAT | 93.0 | 85.2 |
| CIFAR10 | 87.8 | 65.5 |
| Flowers102 | 85.4 | 27.9 |
| SVHN | 83.7 | 75.2 |
| Pets | 82.7 | 65.4 |
| GTSRB | 81.5 | 52.0 |
| ISIC | 67.4 | 57.5 |
| CIFAR100 | 63.7 | 24.8 |
| UCF101 | 55.9 | 23.9 |
| DTD | 54.8 | 35.3 |
| Food101 | 54.1 | 14.8 |
| FMoW | 30.2 | 14.8 |
Trainable Modules: ResNet18, ResNeXt101-IG, Swin Transformer, CLIP, self-defined; Frozen Modules: IterMap, FullyMap, FreqMap, SemanticMap; Output Label Mapping (LM): Source labels; Target labels; Trainable LM; Nonparametric LM.
</details>

Figure 1: Overview and key highlights of AutoVP. The main components of AutoVP are: Input Scaling, which offers three initial input scale options: ×0.5, ×1.0, and ×1.5; Visual Prompt, which pads the prompts to the scaled input image; Pre-trained Classifier, allowing users (or AutoVP) to select from four pre-trained models: ResNet18 (He et al., 2016), ResNeXt101-IG (Mahajan et al., 2018), Swin-T (Liu et al., 2021), and CLIP (Radford et al., 2021); and Output Label Mapping, offering four label mapping options: Iterative Mapping (IterMap), Frequency Mapping (FreqMap), Semantic Mapping (SemanticMap), and Fully Connected Layer Mapping (FullyMap). Bottom panel: Given a fixed ImageNet-pre-trained classifier (ResNet18), AutoVP outperforms the state-of-the-art (ILM-VP in Chen et al. (2023b)) on all 12 different downstream image-classification tasks.

13.7% better than fixed label mapping strategies. The second critical challenge is lack of a unified performance benchmark: the existing literature evaluates the performance of proposed VP methods in an ad hoc manner, by reporting on arbitrarily selected downstream datasets, making comparisons across different methods difficult at best.

To bridge this gap, this paper proposes AutoVP, a solution addressing both these challenges via 1) its automated, extendable framework for joint optimization of a) input-image scaling (i.e., prompt size), b) visual prompts, c) source model selection, and d) output label-mapping strategies; and 2) its provision of a unified benchmark consisting of 12 diverse image-classification tasks with quantifiable content-similarity relative to the dataset (e.g., ImageNet) used for source model pre-training.

As shown in Figure 1, the first component (i.e., input scaling) of AutoVP determines the optimal ratio between the sizes of prompts and the original images. The second, visual prompts, serve as trainable parameters, and undergo iterative updates during the backpropagation phase. The pre-trained model extracts pertinent features and renders predictions within the source domain; and finally, output label mapping establishes a connection between the label spaces of the source and target domains, facilitating accurate predictions in the downstream domain. The modularity of AutoVP allows for the seamless integration and easy extension of various designs for these four components.

Table 1 compares AutoVP against prior VP proposals and the other two baselines proposed to date: LP and text-prompt (TP)-based zero-shot inference. As the table shows, AutoVP is the only such framework that considers the broad range of settings that can affect VP performance. Moreover, thanks to such settings' collective optimization, AutoVP's configuration amounts to a breakthrough in average accuracy across 12 distinct downstream tasks. For instance, with CLIP as the pre-trained model (see Table 2), AutoVP's average accuracy is $4.6\%$ higher than CLIP-VP's (Bahng et al., 2022) and $2.1\%$ higher than ILM-VP's (Chen et al., 2023b). AutoVP also surpasses LP's accuracy by $0.7\%$ on average, suggesting that it is a competitive alternative to LP in terms of transfer learning.

We summarize the main contributions as follows:

\- AutoVP is the first end-to-end VP framework that simultaneously takes account of the design of input scale, visual prompts, pre-trained model selection, and output LM strategies. This modular approach to automating VP gives its users flexibility for VP engineering, as well as a straightforward, comprehensive performance benchmark based on 12 downstream image-classification datasets.

Table 1: Comparison of AutoVP with other baselines, including Linear Probing, CLIP zero-shot inference with text prompts (i.e. CLIP-TP in Radford et al. (2021)), CLIP-VP (Bahng et al., 2022), and ILM-VP (Chen et al., 2023b). The average accuracy is evaluated over 12 downstream tasks (see Section 4). For detailed information about the setting configurations, please refer to Section 3. 

<table><tr><td>Method</td><td>Pre-trainedClassifier</td><td>Prompt Size</td><td>OutputTransformation</td><td>Output MappingNumber</td><td>AverageAccuracy (%)</td></tr><tr><td>Linear Probing</td><td>CLIP</td><td>—</td><td>Modified LastClassification Layer</td><td>—</td><td>79.86</td></tr><tr><td>CLIP-TP</td><td>CLIP</td><td>—</td><td>Fixed Text Prompt</td><td>1</td><td>49.54</td></tr><tr><td>CLIP-VP</td><td>CLIP</td><td>30</td><td>Fixed Text Prompt</td><td>1</td><td>76.01</td></tr><tr><td rowspan="2">ILM-VP</td><td>ResNet18</td><td>48</td><td rowspan="2">IterMap</td><td rowspan="2">1</td><td>45.19</td></tr><tr><td>CLIP</td><td>30</td><td>78.45</td></tr><tr><td rowspan="4">AutoVP(Ours)</td><td>ResNet18</td><td rowspan="4">Trainable</td><td>IterMap</td><td></td><td></td></tr><tr><td>ResNeXt101-IG</td><td>FullyMap</td><td rowspan="3">1/5/10</td><td rowspan="3">81.02</td></tr><tr><td>Swin-T</td><td>FreqMap</td></tr><tr><td>CLIP</td><td>SemanticMap</td></tr></table>

- The proposed hyper-parameter tuning process is capable of identifying optimal configurations tailored to individual downstream datasets. In addition, its novel components – e.g., automated input scaling (Section 3) and weight initialization (Section 4.2) – augment VP's overall efficacy significantly, as compared to state-of-the-art VP methods, LP, and zero-shot baselines (see Table 1).   
- This paper represents the first step in a comprehensive exploration of optimal configurations across varied conditions (e.g., fixing a source model or an output-mapping strategy), and presents an analysis of domain similarity's impact on VP performance for each downstream dataset.   
- This paper highlights AutoVP's superior performance over LP in data-limited settings (Figure 2) and its better out-of-distribution robustness than LP (Figure 4).

# 2 BACKGROUND AND RELATED WORK

Background of Visual Prompts. Traditionally, to derive task-specific machine-learning models, researchers have to train or update all model parameters. But, amid the advancement of powerful foundation models, model fine-tuning and training from scratch have both become time-consuming and parameter-inefficient approaches, usually requiring large amounts of training data and storage space. To this end, VP, also known as in-domain model reprogramming, has emerged as an effective means of obtaining machine-learning models for various domain-specific tasks (Chen, 2022). A well-developed pre-trained model from a source domain can be directly used for performing tasks in the target domain with little transformation of the target data. For example, we can use an ImageNet pre-trained model to classify medical images without modifying any of its parameters (Tsai et al., 2020). On the other hand, VP, along with temperature scaling, can also be used as a post-processing calibration method to align model confidence and accuracy (Tang et al., 2022; Hsiung et al., 2023). As compared to traditional approaches such as transfer learning, model fine-tuning, and training from scratch, VP is a low-complexity and model-agnostic strategy; and it is especially suitable for low-data domains.

The Design of Visual Prompts. A VP framework can generally be divided into two trainable modules, one for input transformation and the other for output transformation. These are respectively placed at the input and output ends of a pre-trained model. In the case of input transformation, previous literature has proposed various ways to generate and place visual prompts. One of the most popular such approaches is to pad a frame around the target task image and then fill it with trainable additive-input perturbation (prompts) (Tsai et al., 2020; Chen et al., 2023b; Elsayed et al., 2019; Bahng et al., 2022; Wu et al., 2022; Oh et al., 2023). Next, since the output logits of the source pre-trained model are still in the source domain, further output transformation (e.g., LM) is required to obtain the target-domain logits. One naive way of achieving this is randomly mapping (RandMap) m source labels onto the target labels. Tsai et al. (2020) found that frequency-based LM (FreqMap), which constructs its LM from the source-label prediction distribution of the target-domain data, can further improve the accuracy of downstream tasks. However, Chen et al. (2023b) argued that FreqMap lacks interpretability and that its interaction with VP is difficult to measure. To address that

problem, the authors proposed iterative LM (IterMap), a transformation of FreqMap that enables it to concurrently design LM and visual prompts. Yang et al. (2023), meanwhile, proposed a semantics-based LM approach that aligns source and target classes that have similar semantic meanings. And Liao et al. (2023) utilized a prototypical verbalizer to map a mask token to downstream labels, thus providing a different perspective on LM. In this paper, we follow a similar design to Bahng et al. (2022), in which visual prompts are placed around images for input transformations, and there are four mapping methods for output transformations. Further details will be presented in Section 3.

Non-universal Visual Prompts. Instead of utilizing universal input prompts, some researchers have focused on designing input-aware prompting models (Zhou et al., 2022a;b). For instance, Chen et al. (2023a) generated class-wise visual prompts to improve model robustness. Similarly, to address accuracy drops caused by low-voltage-induced bit errors, Sun et al. (2023) proposed an input-aware add-on module to generate a robust prompt; and Loedeman et al. (2022) proposed the Prompt Generation Network (PGN), which generates visual prompt token vectors based on input images, allowing for more adaptive and context-aware prompting.

Although input prompting is commonly applied directly to the target image, researchers have also developed other prompting methods, such as convolutional visual prompt (Tsai et al., 2023), which learns prompting parameters in a small convolutional structure through self-supervision tasks without knowledge of ground truths, and visual prompt tuning (Jia et al., 2022; Sohn et al., 2023), which learns prompting parameters at intermediate layers of a source model. In this paper, we focus on a pixel-level VP approach using a task-specific prompting model for each image-classification task. As such, our approach closely resembles real-world scenarios in which a pre-trained source model remains unmodified, and external variations are not introduced internally.

# 3 AUTOVP FRAMEWORK

Following the system overview of AutoVP in Figure 1, we present its four main components (Input Scaling, Visual Prompt, Pre-trained Classifier, and Output Label Mapping) and its hyper-parameter tuning feature, which enables the joint optimization of these components. Our framework can be extended to support user-defined configurations.

Input Scaling. In our implementation of AutoVP, we choose frame-shape prompts as the default prompting template. Hence, the visual prompt sizes p represent the width of the frame, and its actual number of parameters is $2cp(h + w - 2p)$ , where c, w, and h are color channels, width and height respectively. Although the input image size is determined by the source model, when fine-tuning to a downstream dataset from a source model, there is design freedom to resize the target images and use the remaining space for visual prompts. For instance, if the source model takes images with size $224 \times 224$ as input, one can scale the target image size to $128 \times 128$ , resulting in the final visual prompt of size $p = (224 - 128)/2 = 48$ . It was shown in Bahng et al. (2022) and Wu et al. (2022) that the prompt size (p) plays a key role in VP performance. To automate the process of optimizing image resizing scale, we design a gradient-based optimization algorithm to implement the input scaling module, which is implemented using kornia.geometry.transform() from the Kornia library (Riba et al., 2020). The transform() function integrates a range of geometric transformations for 2D images into deep learning, including differentiable image scaling. In addition to image resizing, the prompt size p will also scale up or down to fill the remaining space. Furthermore, to facilitate the optimization of image resizing and avoid bad local minima, we set the default image size to 128 along with three initial scales: 0.5, 1.0, and 1.5 to optimize, and the corresponding prompt sizes p are 80, 48, and 16 respectively. Consequently, the input scaling module allows AutoVP to obtain the optimal image resizing scale and prompt size (p).

Visual Prompt. For the visual prompt module, AutoVP adds universal pixel-level prompts around all (resized) input images. Let $x_{t} \in R^{N_{t}}$ denote the target (flattened) input image (of $N_{t}$ -dimension), $\tilde{x}_{t} \in R^{N_{s}}$ be the prompted image, which fits the input dimension ( $N_{s}$ ) of the pre-trained source model $f_{\theta_{s}}$ ( $\theta_{s}$ denotes its weights), $\delta \in R^{N_{s}}$ be a trainable universal perturbation, and $M_{p} \in \{0,1\}^{N_{s}}$ be a binary mask of prompt size p, indicating the prompting area. Hence, the prompted image $\tilde{x}_{t}$ can be formulated as:

$$
\tilde {x} _ {\mathrm{t}} = \mathcal {P} (x _ {\mathrm{t}}) = \text { InputScaling } _ {p} (x _ {\mathrm{t}}) + \underbrace {\mathcal {M} _ {p} \odot \sigma (\delta)} _ {\text { Prompts }}. \tag {1}
$$

The prompts are initialized as 0 and formally defined as $\mathcal{M}_{p} \odot \sigma(\delta)$ , where $\sigma$ is the Sigmoid function that maps the input to a value between 0 and 1 (the scaled input pixel value range), ensuring it has the same numerical range as the input image. We then update $\delta$ using gradient descent.

Pre-trained Classifier. After applying the prompts to the resized image through the preceding stages, the prompted image is subsequently fed into the pre-trained model, which serves as a feature extractor to generate predictions in the source domain. We include four representative pre-trained models in our AutoVP framework: ResNet18 (He et al., 2016), ResNeXt101-IG (Mahajan et al., 2018), Swin-T (Liu et al., 2021), and a vision-language multi-modal model, CLIP (Radford et al., 2021) with the ViT-B/32 vision encoder backbone. Note that in AutoVP, the weights of the pre-trained classifiers are frozen and kept unchanged. The details of the models are provided in Appendix A.1.

Output Label Mapping. The pre-trained models predict target data to source labels, while the last mile for VP is to map predictions on the source labels to target classes. As illustrated in Figure 1, AutoVP provides four types of output mapping, and they can be generally categorized into two groups. (i) nonparametric label mapping: frequency mapping (FreqMap) and semantic mapping (SemanticMap), which are defined during the initialization of VP training and remain unchanged throughout the training process; and (ii) trainable label mapping: iterative label mapping (IterMap) and fully connected layer mapping (FullyMap). These two methods dynamically adjust the mapping based on the prompted images. In the following, we provide the overview of our four output mapping approaches, please refer to Appendix A.2 for more details.

\- Frequency Mapping (FreqMap) is proposed by Tsai et al. (2020). It utilizes the source-label prediction distribution of the target-domain data to map each target class to the top- $m$ most frequent source classes. Let $\mathcal{Y}_{\mathrm{s}} = \{0, \cdots, K_{\mathrm{s}} - 1\}$ and $\mathcal{Y}_{\mathrm{t}} = \{0, \cdots, K_{\mathrm{t}} - 1\}$ be the set of source and target labels, where $K_{\mathrm{s}} / K_{\mathrm{t}}$ are the numbers of source/target classes. Consider $\tilde{\mathcal{X}}_{\mathrm{t}}$ collects all prompted images of label $y_{\mathrm{t}}$ in target domain $\mathcal{D}_{\mathrm{t}}$ , i.e. $\tilde{\mathcal{X}}_{\mathrm{t}} = \{\tilde{x}_{\mathrm{t}_{i}} = \mathcal{P}(x_{\mathrm{t}_{i}}) | (y_{\mathrm{t}_{i}} = y_{\mathrm{t}}), (x_{\mathrm{t}_{i}}, y_{\mathrm{t}_{i}}) \in \mathcal{D}_{\mathrm{t}}\}$ , then when $m = 1$ , the mapping of $y_{\mathrm{t}}$ can be defined as:

$$
y _ {\mathrm{t}} \leftarrow y _ {\mathrm{s}} ^ {*} = \arg \max _ {y _ {\mathrm{s}} \in \mathcal {Y} _ {\mathrm{s}}} (\sum_ {\tilde {x} _ {\mathrm{t}} \in \tilde {\mathcal {X}} _ {\mathrm{t}}} P r e d (f _ {\theta_ {\mathrm{s}}} (\tilde {x} _ {\mathrm{t}}), y _ {\mathrm{s}})),
$$

$$
\operatorname{Pred} \left(f _ {\theta_ {\mathrm{s}}} \left(\tilde {x} _ {\mathrm{t}}\right), y _ {\mathrm{s}}\right) = \left\{ \begin{array}{l l} 1, & \text { if } y _ {\mathrm{s}} = \arg \max f _ {\theta_ {\mathrm{s}}} \left(\tilde {x} _ {\mathrm{t}}\right) \\ 0, & \text { otherwise } \end{array} . \right. \tag {2}
$$

The objective of FreqMap is to map the target label $y_{t}$ to the source label $y_{s}^{*}$ , which is the most frequent label that $f_{\theta_{s}}$ classified $\tilde{X}_{t}$ as. If a source class is selected as the most frequently predicted class for multiple target classes, it will be assigned to the target class that has the highest count of predictions. The general many-to-one frequency mapping algorithm is provided in Algorithm 1 in the Appendix A.2. Moreover, random label mapping (RandMap) can be viewed as a special case of FreqMap by randomly assigning a subset of source labels to a target label.

- Iterative Mapping (IterMap, or ILM) is proposed by Chen et al. (2023b), which is an iterative approach for updating FreqMap. IterMap performs the frequency mapping at the beginning of each training epoch to obtain a new mapping distribution that aligns with the updated prompts.   
- Semantic Mapping (SemanticMap) follows the works from Yang et al. (2023) and Yen et al. (2021). We utilize the text encoder of CLIP to generate the embeddings of the names of the source and target classes. Subsequently, we map the source-target pairs based on the highest cosine similarity score between their respective embeddings. Hence, SemanticMap can be utilized in any of the three vision pre-trained models (ResNet18, Swin-T, and ResNeXt101-IG) by establishing mappings between the target classes and semantically similar classes from ImageNet-1K. However, SemanticMap is not applicable for CLIP, as it does not have an explicit set of source domain classes.   
- Fully Connected Layer Mapping (FullyMap) uses a linear layer to map the source output logits to target classes (Arif et al., 2023). FullyMap can be represented as $L_{\mathrm{t}} = w \cdot L_{\mathrm{s}} + b$ , where $L_{\mathrm{s}}$ is the output logits from the source pre-trained model, $w$ and $b$ are the weight and bias vector of the linear layer, and $L_{\mathrm{t}}$ is the output of the linear layer which also serves as the final output logits of the VP model.

End-to-end Hyper-parameter Tuning. AutoVP's overall tuning procedure is depicted in Appendix A.3. Given its flexibility and modularity, its users must consider numerous settings ( $n = 222$ ),

Table 2: Comparison of VP testing accuracy (%) using CLIP as a pre-trained model on 12 datasets; the optimal tuning settings of AutoVP and the final prompts sizes p are also provided. In the AutoVP setting field, the notation “Mapping-m” represents mapping m source classes to each target class. 

<table><tr><td>Dataset</td><td>AutoVP Setting</td><td>AutoVP</td><td>ILM-VP</td><td>CLIP-VP</td><td>LP</td></tr><tr><td>SVHN (Netzer et al., 2011)</td><td>FullyMap,  $p = 51$ </td><td> $92.9 \pm 0.2$ </td><td>91.2</td><td>88.4</td><td>65.4</td></tr><tr><td>CIFAR10 (Krizhevsky &amp; Hinton, 2009)</td><td>IterMap-1,  $p = 23$ </td><td> $95.2 \pm 0.0$ </td><td>94.4</td><td>94.2</td><td>95.0</td></tr><tr><td>Flowers102 (Nilsback &amp; Zisserman, 2008)</td><td>FullyMap,  $p = 16$ </td><td> $90.4 \pm 0.6$ </td><td>83.7</td><td>70.3</td><td>96.9</td></tr><tr><td>Food101 (Bossard et al., 2014)</td><td>FreqMap-1,  $p = 16$ </td><td> $82.3 \pm 0.1$ </td><td>79.1</td><td>78.9</td><td>84.6</td></tr><tr><td>UCF101 (Soomro et al., 2012)</td><td>FullyMap,  $p = 16$ </td><td> $73.1 \pm 0.6$ </td><td>70.6</td><td>66.1</td><td>83.3</td></tr><tr><td>OxfordIIITPet (Parkhi et al., 2012)</td><td>FreqMap-10,  $p = 16$ </td><td> $88.2 \pm 0.2$ </td><td>85.0</td><td>85.0</td><td>89.2</td></tr><tr><td>CIFAR100 (Krizhevsky &amp; Hinton, 2009)</td><td>FullyMap,  $p = 31$ </td><td> $77.9 \pm 0.6$ </td><td>73.9</td><td>75.3</td><td>80.0</td></tr><tr><td>EuroSAT (Helber et al., 2019)</td><td>FullyMap,  $p = 16$ </td><td> $96.8 \pm 0.2$ </td><td>96.9</td><td>96.4</td><td>95.3</td></tr><tr><td>DTD (Cimpoi et al., 2014)</td><td>FullyMap,  $p = 17$ </td><td> $62.5 \pm 0.3$ </td><td>63.9</td><td>57.1</td><td>74.6</td></tr><tr><td>ISIC (Codella et al., 2019; Tschandl et al., 2018)</td><td>IterMap-1,  $p = 16$ </td><td> $74.0 \pm 1.0$ </td><td>73.3</td><td>75.1</td><td>71.9</td></tr><tr><td>FMoW (Christie et al., 2018)</td><td>FullyMap,  $p = 16$ </td><td> $40.8 \pm 0.8$ </td><td>36.8</td><td>32.9</td><td>36.3</td></tr><tr><td>GTSRB (Houben et al., 2013)</td><td>FullyMap,  $p = 80$ </td><td> $93.1 \pm 0.2$ </td><td>92.6</td><td>92.4</td><td>85.8</td></tr><tr><td>Average Accuracy</td><td></td><td>80.6</td><td>78.5</td><td>76.0</td><td>79.9</td></tr></table>

including how big the initial input image should be, whether to use a trainable resizing module, which pre-trained classifiers to adopt, what output-mapping method to implement, and the number of source labels to map for each target label. To speed up the tuning operation and save computational resources, we use Ray Tune (Liaw et al., 2018) along with an early-stop strategy for terminating poor trails. In our experiments, we employed grid searches to test all configurations. An ASHA scheduler (Li et al., 2018) was used to retain the top-n tasks, and we continued training them while stopping the remaining tasks early. We established experimentally that n = 2 top tasks were enough to obtain the optimal setting. When the few-epoch tuning process (training 2-5 epochs with each setting) is complete, we select the setting having the highest testing accuracy and conduct complete training using that setting. By using hyper-parameter tuning, AutoVP can efficiently find the best configuration of VP and lead to significant accuracy improvement in downstream tasks.

# 4 EXPERIMENTS

Experimental Setup. We evaluated the performance of AutoVP on 12 downstream datasets (CIFAR10, CIFAR100, ISIC, SVHN, GTSRB, Flowers102, DTD, Food101, EuroSAT, OxfordIIITPet, UCF101, and FMoW), which are common datasets when measuring transfer learning generalization. Detailed descriptions of these datasets are given in Appendix B.1. We repeated each AutoVP experiment in triplicate, utilizing a learning rate of 40 with the SGD optimizer for CLIP, and a learning rate of 0.001 with the Adam optimizer for the other pre-trained models. The results of the baselines (CLIP-VP (Bahng et al., 2022) and ILM-VP (Chen et al., 2023b)) were extracted from the reported accuracies in their respective papers (please refer to Appendix B.2 for more details). Our experiments were performed on NVIDIA GeForce RTX 3090 and are implemented with PyTorch.

# 4.1 EXPERIMENTAL RESULTS

Comparison of AutoVP and Prior Work. To ensure that our comparison of AutoVP against previously proposed VP approaches was fair, we fixed its source model but relaxed its other hyperparameter tunings. The results of using CLIP as the source model are presented in Table 2, along with the optimal settings arrived at. We compared AutoVP against LP and two state-of-the-art VP baselines, CLIP-VP and ILM-VP, whose configurations can also be found in Table 1. With the optimal configuration chosen via the tuning process, AutoVP outperformed these other approaches by up to 6.7% on nine of the 12 target datasets. Additionally, AutoVP surpassed the LP baseline on half those datasets, by a maximum of 27.5% in the case of SVHN. AutoVP also obtained the best average accuracy.

We observed that AutoVP employed FullyMap as the output transformation on most datasets. We speculate that this might have been because the linear layer has more parameters and thus allows the achievement of better results. Also, when AutoVP selected initial image scales, it had a tendency to

![](images/54243d5ddb435c5e1c8fc1a6917a10b6bb933b5136f13b37de103511f4923a7d.jpg)

<details>
<summary>line</summary>

| Percentage of Data Usage | AutoVP Top-1 Acc. | LP Acc. | Acc. Gain (AutoVP-LP) |
| ------------------------ | ----------------- | ------- | --------------------- |
| 1                        | 57.4              | 44.3    | 13.1                  |
| 10                       | 70.8              | 68.0    | 2.7                   |
| 25                       | 74.0              | 73.1    | 0.8                   |
| 50                       | 77.3              | 75.6    | 1.8                   |
| 100                      | 80.9              | 79.9    | 1.0                   |
</details>

Figure 2: Data Scalability. The chart presents the average accuracy of AutoVP and LP across the 12 datasets with varying data percentages: 100%, 50%, 25%, 10%, and 1%. The green bar represents the accuracy gains achieved by AutoVP compared to Linear Probing (LP).

![](images/c8c03486270f9818a75d654486b92a8d3a6f7d8dd127d46396f5b06f53dd2b16.jpg)

<details>
<summary>bar</summary>

| Model          | Full Training (w/o WI) | Full Training (w/ WI) | IterMap | FullyMap w/ WI | FullyMap | SemanticMap |
| -------------- | ---------------------- | --------------------- | ------- | -------------- | -------- | ----------- |
| ResNet18       | -                      | -                     | -       | -              | -        | -           |
| ResNext101-IG  | -                      | -                     | -       | -              | -        | -           |
| Swin-T         | -                      | -                     | -       | -              | -        | -           |
| CLIP           | -                      | -                     | -       | -              | 78.8     | -           |
</details>

Figure 3: The Tuning Result on Flowers102 Dataset. The color bars represent the accuracy tuning for 3 to 5 epochs (with ASHA early-stop optimization). Points illustrate the test accuracy achieved for the best settings given the pre-trained model and mapping method.

scale up those images with relatively small prompt sizes. This allowed the VP model to allocate more attention to the image itself, leading to improved overall performance. As shown in Figure 1, when ResNet18 was used as the source model, AutoVP outperformed ILM-VP by 24.8% on average. More experimental results under this setting are provided in Appendix C.1.

AutoVP with Source Model Selection. We also allowed AutoVP to search the optimal source model for downstream tasks. The optimal settings selected by AutoVP, and a comparison of experimental results can be found in Appendix C.1. Our experimental results show that Swin-T was the pre-trained model most frequently chosen by AutoVP as most suitable, i.e., in the cases of eight of the 12 datasets. On average, this choice resulted in $0.43\%$ better accuracy than when CLIP was utilized as the fixed pre-trained backbone. On the DTD and Flowers102 datasets, however, Swin-T's performance was better than CLIP's by much more: i.e., $6.80\%$ and $3.08\%$ , respectively. These findings highlight how multiple pre-trained models can be leveraged to enhance performance across a diverse range of datasets.

Data Scalability. To understand how AutoVP would perform in a data-limited regime, we gradually and uniformly reduced the amount of training data to 50%, then 25%, then 10%, and finally 1% of each training dataset's original size. The experimental results in Figure 2 indicate that AutoVP consistently outperformed LP across all 12 datasets, and that its relative performance was especially high in the two scenarios with the lowest data volumes, i.e., 10% and 1% data usage. The dataset-specific results can be found in Figure C.1 (within Appendix C.2).

# 4.2 ABLATION STUDIES OF AUTOVP

We designed a range of model architectures as testbeds for examining the performance of AutoVP's various components. Our comparisons of these VP architectures included 1) the utilization of a weight-initialization strategy with FullyMap, 2) the inclusion vs. exclusion of the CLIP text encoder, 3) the presence vs. absence of visual prompts, and 4) the frequency analysis of the learned VP.

Weight Initialization of FullyMap with CLIP. When CLIP was used as the pre-trained model, the FullyMap output transformation exhibited significantly inferior performance to FreqMap and IterMap (Figure 3). This is because FreqMap and IterMap can leverage CLIP's zero-shot property with the semantic meanings of the labels, whereas the fully connected layer needs to learn the mapping from a random state. As a result, FullyMap tends to perform poorly in the hyper-parameter tuning process, yet may achieve higher accuracy after completing 200 epochs of training. In Figure 3, for example, AutoVP suggests that the optimal output transformation for Flowers102 with CLIP is IterMap; but in reality, FullyMap achieves better performance after training for 200 epochs (87.3%, as against 78.8% for IterMap).

![](images/ed925cf3d7b5bf72b32f405f8cc118afa703ddd8bae9d16460ccee0c67c6e3c3.jpg)

<details>
<summary>bar</summary>

| Dataset   | AutoVP | ILM-VP | CLIP-VP | LP   |
| --------- | ------ | ------ | ------- | ---- |
| CIFAR10   | 95.2   | 94.1   | 94.1    | 92.9 |
| CIFAR10-C | 82.2   | 79.0   | 78.6    | 76.2 |
</details>

Figure 4: Models Robustness with CLIP. The comparison of accuracy drop on the CIFAR10-C dataset across AutoVP, ILM-VP, CLIP-VP and LP.

![](images/093a6f93faf0580bcc1734e25432c73816008e9fba4038cdc2d742da8ccde57b.jpg)

<details>
<summary>scatter</summary>

| Dataset       | Confidence Score | Accuracy Gain (%) |
| ------------- | ---------------- | ----------------- |
| CIFAR10       | 0.8              | 0                 |
| Food101       | 0.7              | -5                |
| UCF101        | 0.6              | -10               |
| Flowers102    | 0.7              | 5                 |
| DTD           | 0.6              | -15               |
| FMOW          | 0.4              | 10                |
| CIFAR100      | 0.4              | 5                 |
| EuroSAT       | 0.2              | 25                |
| GTSRB         | 0.2              | 10                |
| ISIC          | 0.4              | 5                 |
| SVHN          | 0.2              | 30                |
</details>

![](images/04c2ed08d9c763ba143c1a0d8fbb0fb98d60373c773276c39417a932d1b04f69.jpg)

<details>
<summary>scatter</summary>

| Model       | Confidence Score | CLIP Zero-Shot Accuracy (%) |
|-------------|------------------|-----------------------------|
| SVHN        | 0.2              | 60                          |
| GTSRB       | 0.2              | 40                          |
| ISIC        | 0.4              | 40                          |
| EuroSAT     | 0.3              | 20                          |
| CIFAR100    | 0.3              | 20                          |
| Flowers102  | 0.3              | 20                          |
| FMoW        | 0.3              | 20                          |
| DTD         | 0.3              | 20                          |
| UCF101      | 0.3              | 20                          |
| CIFS100     | 0.3              | 20                          |
| OxfordIIIITPet | 0.3            | 20                          |
| CIFAR10     | 0.3              | 20                          |
| Food101     | 0.3              | 20                          |
</details>

Figure 5: Accuracy Gains with CLIP. The right side of the chart indicates a higher out-of-distribution (OOD) extent, accompanied by larger gain values. Conversely, the left side shows lower gain values.

To address the bottleneck in hyperparameter tuning caused by FullyMap, we introduced weight initialization (WI). This allows FullyMap to initialize with a more informative mapping based on the semantic meaning of the class names. As mentioned in Section 3, AutoVP's FullyMap consists of a linear layer, which can be characterized as $L_{\mathrm{t}} = w \cdot L_{\mathrm{s}} + b$ where the weight $w$ is a $K_{\mathrm{t}}$ by $K_{\mathrm{t}}T$ real matrix, $K_{\mathrm{t}}$ denotes the number of target classes, and $T$ is the number of templates used in the CLIP text encoder. The weight initialization is to assign the diagonal of $w$ to be 1, and the rest of it is set to 0, resulting in $w = (\mathbf{I}_{K_{\mathrm{t}}}|\mathbf{0})$ , where $\mathbf{I}_{K_{\mathrm{t}}}$ is an identity matrix of size $K_{\mathrm{t}}$ , and $\mathbf{0}$ is a $K_{\mathrm{t}} \times K_{\mathrm{t}}(T - 1)$ zero matrix, indicating the connection between each target class class\_name and its corresponding text prompts ("This is a photo of a [class\_name]"). In Figure A.1 (within Appendix A.2), we represent this concept visually. As shown in Figure 3, our experimental results demonstrate that when utilizing CLIP as the pretrained model, FullyMap with proper WI (hatched bar) can also outperform other mapping approaches.

Impact of the Non-inclusion of Text Encoder in CLIP. When replicating the experimental setting as shown in Table 2 and establishing a direct connection between the fully connected output mapping layer and the CLIP image encoder without incorporating the text encoder, there was a substantial decrease in average accuracy: to 69.0% (see Figure E.1, Appendix E.1). Dataset-specific accuracy drops were particularly prominent in Flowers102, Food101, and OxfordIIITPet. These outcomes suggest that label semantics play a crucial role in those datasets.

The Impact of Visual Prompts. We also investigated the effects on the overall performance of removing the module of visual prompts from the AutoVP pipeline while retaining the pre-trained model and output-mapping modules. When the ResNet18 model was used, leaving the visual prompts in yielded better performance than omitting them in just three out of 12 cases: i.e., the SVHN, GTSRB, and ISIC datasets (Figure C.3 (b), Appendix C.3). For the remaining datasets, the inclusion of visual prompts actually led to a decline in performance. This suggests that when a relatively small source model is used for VP, a significant improvement in accuracy of the sort reported in Table C.1 can primarily be attributed to fully connected layer mapping, and visual prompts may be perceived as noise. On the other hand, when the CLIP model was used, most of the datasets had positive gain values (Figure 5 (b)), indicating improved performance, when visual prompts were included. This suggests that CLIP is more suitable for VP tasks than ResNet18 is.

Frequency Analysis of the Learned Visual Prompts. In Appendix D.1, we also conducted an analysis from a frequency perspective (Brigham, 1988) to study the generalization of visual prompt patterns. The results highlighted the effectiveness of prompting with CLIP, harnessing low-frequency features that generalize to the target domain.

# 5 DISCUSSIONS

Tuning Selection. AutoVP provides joint optimization of its multiple configurations and selects different parameters according to its target tasks. In terms of output label mapping, FullyMap exhibits superior performance in vision models, but IterMap or FreqMap appear to enhance the performance of text-image models like CLIP. In this context, weight initialization with FullyMap plays an important role in CLIP, making this option one of the more frequently chosen output-mapping strategies (Table 2). We also observed that novel designs exploiting larger image scales and mapping a larger number

of source classes tended to yield enhanced performance. More information on selection preferences during hyperparameter tuning can be found in Appendix D.3.

AutoVP Robustness. We trained AutoVP on CIFAR10 and evaluated its robustness on the corrupted dataset CIFAR10-C, which consists of 18 types of filters or noise. As shown in Figure 4, AutoVP maintained greater robustness than ILM-VP, CLIP-VP, and LP. Its loss of accuracy was relatively small: suggesting that AutoVP exhibits a lower degree of overfitting to the training data and possesses a higher ability to resist the impact of noise than the other baselines do.

Performance Evaluation on ID/OOD Downstream Tasks. We evaluate the out-of-distribution (OOD) extent of each dataset relative to the pre-trained CLIP by considering the average confidence score (Guo et al., 2017) and the CLIP zero-shot inference. The accuracy gains achieved through VP (Figure 5) were computed as the difference in accuracy between AutoVP and LP or non-VP approaches (i.e. visual prompts were removed and output mapping retained). We observed that the datasets that were more in-distribution (ID), with higher confidence scores and higher zero-shot accuracy, exhibited smaller accuracy gains from VP. Conversely, datasets that were more OOD, characterized by lower confidence scores and lower zero-shot accuracy, had their accuracy improved more through AutoVP.

We also evaluated accuracy gains with ResNet18 pre-trained on ImageNet-1K (Russakovsky et al., 2015) (Figure C.3, Appendix C.3) and, to assess domain differences between ImageNet-1K and other downstream datasets, we calculated the FID score (Heusel et al., 2017). The results were consistent with the cases using CLIP. In conclusion, AutoVP is suitable for datasets that exhibit more OOD characteristics than the source domain dataset.

# 6 LIMITATIONS

This work is subject to some limitations. First, our optimization process did not include certain hyperparameters, notably learning rate and weight decay. This omission stemmed from our primary focus on identifying the best configurations for VP training, and because including such hyperparameters would have greatly increased execution workload. In addition, when it comes to choosing the best pre-trained model to fine-tune on the target dataset, You et al. (2021) also argued that, in general, the sophisticated fine-tuning techniques (e.g., regularization) would not change the ranking of pre-trained models in downstream tasks. Nonetheless, we conducted supplementary tuning experiments encompassing various learning rates and weight decays, the results of which can be found in Table E.3 (within Appendix E.3). In practice, we suggest enabling the tuning of learning rates, weight decay, and other model-specific parameters after the initial hyperparameter tuning phase of AutoVP. The tuning of fundamental hyperparameters could potentially be accelerated with recent advancements in utilizing generalization metrics to identify optimal hyperparameter configurations (Zhou et al., 2023a;b), a subject to be explored in future research.

Another limitation pertains to the scope of AutoVP, which is oriented toward classification tasks. However, we have extended its application to segmentation and detection tasks, as detailed in Appendix E.2. Furthermore, recent studies have demonstrated the success of visual prompts in generative tasks (Ramesh et al., 2021; 2022; Liu & Chilton, 2022; Bar et al., 2022; Sohn et al., 2023). Nevertheless, expanding support for generative tasks will require accommodation of their distinctive requirements, e.g., via the integration of a suitable pre-trained generative model, such as GANs (Goodfellow et al., 2014), VAEs (Kingma & Welling, 2013), or diffusion models (Ho et al., 2020), along with tailored prompt design. Certainly, our results imply that there are many avenues for VP research that merit further exploration.

# 7 CONCLUSION

This paper has introduced AutoVP, an end-to-end framework that automatically selects the optimal VP configuration for a downstream dataset. AutoVP demonstrates superior performance over other state-of-the-art VP methods and transfers learning baselines in both standard and sample-reduced fine-tuning settings. This research has also yielded important insights into optimal VP configurations, the effects of downstream data characteristics on VP, and how robustness against image corruption might be improved. In short, we believe AutoVP is an efficient and expandable tool, and perhaps more importantly, a useful benchmark that will accelerate the development of VP research and technology.

# ACKNOWLEDGMENTS

The research described in this paper was conducted in the JC STEM Lab of Intelligent Design Automation, which is funded by The Hong Kong Jockey Club Charities Trust.

# REFERENCES

Huzaifa Arif, Alex Gittens, and Pin-Yu Chen. Reprogrammable-FL: Improving utility-privacy tradeoff in federated learning via model reprogramming. In First IEEE Conference on Secure and Trustworthy Machine Learning, 2023.   
Hyojin Bahng, Ali Jahanian, Swami Sankaranarayanan, and Phillip Isola. Exploring visual prompts for adapting large-scale models. arXiv preprint arXiv:2203.17274, 1(3):4, 2022.   
Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei Efros. Visual prompting via image inpainting. Advances in Neural Information Processing Systems, 35:25005–25017, 2022.   
Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In European Conference on Computer Vision, 2014.   
E Oran Brigham. The fast Fourier transform and its applications. Prentice-Hall, Inc., 1988.   
Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, and Sijia Liu. Visual prompting for adversarial robustness. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023a.   
Aochuan Chen, Yuguang Yao, Pin-Yu Chen, Yihua Zhang, and Sijia Liu. Understanding and improving visual prompting: A label-mapping perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19133–19143, June 2023b.   
Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.   
Pin-Yu Chen. Model reprogramming: Resource-efficient cross-domain machine learning. arXiv preprint arXiv:2202.10629, 2022.   
Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 105(10):1865–1883, 2017.   
Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.   
M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, , and A. Vedaldi. Describing textures in the wild. In Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2014.   
Noel Codella, Veronica Rotemberg, Philipp Tschandl, M Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1902.03368, 2019.   
Gamaleldin F. Elsayed, Ian Goodfellow, and Jascha Sohl-Dickstein. Adversarial reprogramming of neural networks. In International Conference on Learning Representations, 2019.   
Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. International journal of computer vision, 111:98–136, 2015.   
Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. In Association for Computational Linguistics (ACL), 2021.   
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pp. 1321–1330. PMLR, 2017.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.   
Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.   
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
Sebastian Houben, Johannes Stallkamp, Jan Salmen, Marc Schlipsing, and Christian Igel. Detection of traffic signs in real-world images: The german traffic sign detection benchmark. In The 2013 International Joint Conference on Neural Networks (IJCNN), pp. 1–8, 2013.   
Lei Hsiung, Yung-Chen Tang, Pin-Yu Chen, and Tsung-Yi Ho. NCTV: Neural Clamping Toolkit and Visualization for Neural Network Calibration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37 (13), pp. 16446–16448, Sep. 2023.   
Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European Conference on Computer Vision, pp. 709–727. Springer, 2022.   
Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2901–2910, 2017.   
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pp. 554–561, 2013.   
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, Toronto, Ontario, 2009.   
Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3045–3059, Online and Punta Cana, Dominican Republic, 11 2021. Association for Computational Linguistics.   
Liam Li, Kevin Jamieson, Afshin Rostamizadeh, Ekaterina Gonina, Moritz Hardt, Benjamin Recht, and Ameet Talwalkar. Massively parallel hyperparameter tuning. arXiv preprint arXiv:1810.05934, 5, 2018.   
Ning Liao, Bowen Shi, Min Cao, Xiaopeng Zhang, Qi Tian, and Junchi Yan. Rethinking visual prompt learning as masked visual token modeling. arXiv preprint arXiv:2303.04998, 2023.   
Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E Gonzalez, and Ion Stoica. Tune: A research platform for distributed model selection and training. arXiv preprint arXiv:1807.05118, 2018.   
Vivian Liu and Lydia B Chilton. Design guidelines for prompt engineering text-to-image generative models. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pp. 1–23, 2022.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.   
Jochem Loedeman, Maarten C Stol, Tengda Han, and Yuki M Asano. Prompt generation networks for efficient adaptation of frozen vision transformers. arXiv preprint arXiv:2210.06466, 2022.   
Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens Van Der Maaten. Exploring the limits of weakly supervised pretraining. In Proceedings of the European conference on computer vision (ECCV), pp. 181–196, 2018.   
Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In European Conference on Computer Vision, pp. 728–755. Springer, 2022.   
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.   
M-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing, Dec 2008.   
Changdae Oh, Hyeji Hwang, Hee-young Lee, YongTaek Lim, Geunyoung Jung, Jiyoung Jung, Hosik Choi, and Kyungwoo Song. Blackvip: Black-box visual prompting for robust transfer learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24224–24235, June 2023.   
Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pp. 3498–3505. IEEE, 2012.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.   
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 8821–8831. PMLR, 2021.   
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.   
E. Riba, D. Mishkin, D. Ponsa, E. Rublee, and G. Bradski. Kornia: an open source differentiable computer vision library for pytorch. In Winter Conference on Applications of Computer Vision, 2020.   
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, 2015.   
Fobo Shi, Peijun Qing, Dong Yang, Nan Wang, Youbo Lei, Haonan Lu, and Xiaodong Lin. Prompt space optimizing few-shot reasoning success with large language models. arXiv preprint arXiv:2306.03799, 2023.   
Kihyuk Sohn, Huiwen Chang, José Lezama, Luisa Polania, Han Zhang, Yuan Hao, Irfan Essa, and Lu Jiang. Visual prompt tuning for generative transfer learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19840–19851, 2023.   
Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

Hao-Lun Sun, Lei Hsiung, Nandhini Chandramoorthy, Pin-Yu Chen, and Tsung-Yi Ho. NeuralFuse: Learning to Recover the Accuracy of Access-Limited Neural Network Inference in Low-Voltage Regimes. arXiv preprint arXiv:2306.16869, 2023.   
Yung-Chen Tang, Pin-Yu Chen, and Tsung-Yi Ho. Neural clamping: Joint input perturbation and temperature scaling for neural network calibration. arXiv preprint arXiv:2209.11604, 2022.   
Yun-Yun Tsai, Pin-Yu Chen, and Tsung-Yi Ho. Transfer learning without knowing: Reprogramming black-box machine learning models with scarce data and limited resources. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 9614–9624. PMLR, 13–18 Jul 2020.   
Yun-Yun Tsai, Chengzhi Mao, and Junfeng Yang. Convolutional visual prompt for robust visual perception. In Advances in Neural Information Processing Systems, volume 36, pp. 27897–27921. Curran Associates, Inc., 2023.   
Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data, 5(1):1–9, 2018.   
Junyang Wu, Xianhang Li, Chen Wei, Huiyu Wang, Alan Yuille, Yuyin Zhou, and Cihang Xie. Unleashing the power of visual prompting at the pixel level. arXiv preprint arXiv:2212.10556, 2022.   
Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pp. 3485–3492. IEEE, 2010.   
Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1492–1500, 2017.   
Ziqing Yang, Zeyang Sha, Michael Backes, and Yang Zhang. From visual prompt learning to zero-shot transfer: Mapping is all you need. arXiv preprint arXiv:2303.05266, 2023.   
Hao Yen, Pin-Jui Ku, Chao-Han Huck Yang, Hu Hu, Sabato Marco Siniscalchi, Pin-Yu Chen, and Yu Tsao. Neural model reprogramming with similarity based mapping for low-resource spoken command classification. arXiv preprint arXiv:2110.03894, 2021.   
Kaichao You, Yong Liu, Jianmin Wang, and Mingsheng Long. LogME: Practical assessment of pre-trained models for transfer learning. In International Conference on Machine Learning, pp. 12133–12143. PMLR, 2021.   
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16816–16825, 2022a.   
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348, 2022b.   
Yefan Zhou, Tianyu Pang, Keqin Liu, Charles H. Martin, Michael W. Mahoney, and Yaoqing Yang. Temperature balancing, layer-wise weight analysis, and neural network training. In Thirty-seventh Conference on Neural Information Processing Systems, 2023a.   
Yefan Zhou, Yaoqing Yang, Arin Chang, and Michael W. Mahoney. A three-regime model of network pruning. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 42790–42809, 2023b.

# APPENDIX

# A Implementation Details of AutoVP 15

A.1 Pre-trained Classifier Details 15   
A.2 Output Label Mappings of AutoVP 15   
A.3 AutoVP Tuning Process 17

# B Datasets and Baselines 18

B.1 The Twelve Downstream Datasets ..... 18   
B.2 Baselines Details 18

# C Additional Experiments with AutoVP 19

C.1 Fixed Pre-trained Model vs. Auto Pre-trained Model Selection ..... 19   
C.2 Data Scalability 20   
C.3 Downstream Dataset Analysis (ID/OOD vs. Accuracy Gain) ..... 21

# D Analysis of AutoVP Results 22

D.1 Prompts in Frequency Domain 22   
D.2 Output Mapping Analysis 23   
D.3 The Preferences in Hyper-Parameter Tuning Selection 24

# E Ablation Studies 24

E.1 The Impact of Text Encoder in CLIP 24   
E.2 Visual Prompting in Segmentation and Detection Tasks 25   
E.3 Exploring Additional Tuning Axes 27   
E.4 Improved ILM-VP with Tuning Configuration 27   
E.5 Comparison of AutoVP and BlackVIP 28

# F Performance and Resource Utilization 28

F.1 Comparison of AutoVP, Linear Probing, and Full Fine-Tuning ..... 28   
F.2 Computing Resources 29

# A IMPLEMENTATION DETAILS OF AUTOVP

# A.1 PRE-TRAINED CLASSIFIER DETAILS

Our AutoVP framework includes four pre-trained models: ResNet18 (He et al., 2016), ResNeXt101-IG (Mahajan et al., 2018), Swin-T (Liu et al., 2021), and CLIP (Radford et al., 2021). Both ResNet18 and Swin-T models were trained on the ImageNet-1K (Russakovsky et al., 2015) dataset, while ResNeXt101-IG was pre-trained on a large collection of Instagram photos. Additionally, CLIP was trained on a dataset consisting of 400 million pairs of images and corresponding text from the internet.

In the structural differences, the first three models are single-modality vision models. ResNet18 is a typical and relatively small convolutional neural network with residual blocks. ResNeXt101-IG is a deeper residual network that incorporates cardinality, which refers to the size of the set of transformations (Xie et al., 2017). Swin-T is a vision transformer that operates by dividing the input image into patches and processing them using the transformer architecture. The last model, CLIP, is a vision-language multi-modal model that calculates the cosine similarity between image embeddings and label text embeddings. The prediction is to select the class with the highest similarity score to the embedding of the input image. The prediction flow is illustrated in Eq. 3, where the input image is denoted as x, and $K_{t}$ represents the size of the predictable class set. The token vector of the i-th class label text, obtained from a tokenizer, is denoted as $ClsTk_{i}$ . CLIP utilizes the Image-Encoder() and Text-Encoder() components to extract features from images and text. The resulting image and text embeddings are represented as $x_{emb}$ and $t_{emb_{i}}$ , respectively. The cosine similarity between any pair of (image, text) embeddings can be computed, and the class with the highest cosine similarity is considered as the predicted class $y_{pred}$ .

$$
\boldsymbol {x} _ {\text { emb }} = \text { Image - Encoder } (x)
$$

$$
\boldsymbol {t} _ {\text {emb} i} = \text {Text - Encoder} (\boldsymbol {C l s T k} _ {i}), 0 \leq i <   K _ {\mathrm{t}} \tag {3}
$$

$$
y _ {\mathrm{pred}} = \arg \max _ {0 \leq i <   K _ {t}} (\frac {\boldsymbol {x} _ {e m b} \cdot \boldsymbol {t} _ {e m b _ {i}}}{\parallel \boldsymbol {x} _ {e m b} \parallel \parallel \boldsymbol {t} _ {e m b _ {i}} \parallel})
$$

By training with pairs of image and text annotations, CLIP is able to learn correlations between visual and textual information, achieving state-of-the-art zero-shot accuracy.

# A.2 OUTPUT LABEL MAPPINGS OF AUTOVP

As mentioned in Section 3, AutoVP incorporates four output label mappings: frequency mapping (FreqMap), iterative mapping (IterMap), semantic mapping (SemanticMap), and fully connected layer mapping (FullyMap). In the following, we provide more details of each mapping algorithm.

I. Frequency Mapping   
![](images/267888001c2cc2d9854fd0a5247ef46c0363671a761fc80a78eb1e72fa57c8f8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Source Labels"] --> B["Target Labels"]
    subgraph SourceLabels
        C["Mouse"]
        D["Car"]
        E["Tree"]
        F["Mouse"]
    end
    subgraph TargetLabels
        G["White Mouse"]
        H["Purple Flower"]
        I["Yellow Flower"]
    end
```
</details>

II. Iterative Mapping   
![](images/52dbbafa3de63937768bbbd50bdadcac6088bc85e390bae8546db0bd3d9c9be5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Source Labels"] --> B["Target Labels"]
    C["VP Model"] --> D["Output"]
```
</details>

III. Semantic Mapping   
![](images/9909c84e41362ee0ff1aff7cd9bf40c7788db1d557b9b490ce986c371a458019.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Source classes names"] --> B["CLIP Text Encoder"]
    C["Target classes names"] --> B
    B --> D["Embedding of source classes names"]
    B --> E["Embedding of target classes names"]
    D --> F["Mapping: Map to the ones with highest cosine similarity score"]
    E --> F
```
</details>

IV. Fully Connected Layer Mapping   
![](images/188ea167e1db240ac109f5e099d0a9c35b1213f9da9bca41eb24e552f8400937.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Logit"] --> B["# of Source classes"]
    A --> C["Wx+b"]
    A --> D["# of Target classes"]
    C --> E["Pred. class"]
    D --> E
```
</details>

V. Fully Connected Layer Mapping:   
CLIP with Weight Initialization   
![](images/e9a3bcc2d7ee5f4bc13277484bd63e75dee2cdcb61408051c8a772c71a244112.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["This is a photo of a airplane"] --> B["1"]
    C["This is a photo of a automobile"] --> D["0"]
    C --> E["1"]
    F["81-templates x #-of-target-classes"] --> G["airplane"]
    G --> H["automobile"]
    H --> I["# of Target classes"]
    I --> J["Pred. class"]
    K["Default Template This is a photo"] --> L["1"]
    M["1"] --> N["airplane"]
    O["1"] --> P["automobile"]
```
</details>

Figure A.1: Output Label Mapping. Illustration of four output mapping methods.

# A.2.1 FREQMAP (FREQUENCY MAPPING)

In FreqMap, the mapping is established between each target class and the most frequently mapped source class. We demonstrate the general many-to-one mapping (multiple source labels mapped to one target label) in Algorithm 1. In lines 6-10, we traverse all the training data pairs $(x_{\mathrm{t}}, y_{\mathrm{t}})$ . First, we pad the image $x_{t}$ with the current visual prompts to obtain a prompted image $\tilde{x}_{t}$ . Then, we could obtain the prediction $f_{\theta_{s}}(\tilde{x}_{t}) = y_{s}$ in the source domain. This gives us a mapping relation from $y_{s}$ to $y_{t}$ , and we increase the count accordingly. In line 11, we obtain the list of source-target id pairs $(S_{\mathrm{id}}, T_{\mathrm{id}})$ , which is sorted by the count matrix (count) in descending order. Accordingly, we start defining the mapping of FreqMap from the most frequently source-target pair. If the source class has not been mapped yet and the mapping count for the target class has not reached the limit m, we establish the mapping relationship $\mathbf{M}[S_{\mathrm{id}}][T_{\mathrm{id}}] = 1$ (line 14). We continue this process until all target classes have been mapped to m source classes. Once this condition is met, the mapping assignment is completed, and we return the mapping matrix M.

Algorithm 1: FreqMap $(\delta, f_{\theta_s}, \mathcal{D}_t, m)$   
Input: visual prompts $\delta$ , source classifier $f_{\theta_s}$ , target dataset $\mathcal{D}_t$ , and the specified number of source classes mapped to each target class $m$ Output: mapping matrix M

# Initialization $K_s, K_t \leftarrow$ number of source and target classes
M $\leftarrow$ 0 $^{\mathbf{K}_s \times \mathbf{K}_t}$ # mapping matrix
count $\leftarrow$ 0 $^{K_s \times K_t}$ # a zero matrix, records the number of images of each target class being predicted (by $f_{\theta_s}$ ) as each source class
done_s $\leftarrow$ 0 $^{K_s}$ # a boolean vector, records whether the source class has been assigned
done_t $\leftarrow$ 0 $^{K_t}$ # a boolean vector, records whether the number of source classes per target class is equal to $m$   
# Calculate the Frequency

6 foreach $(x_{t}, y_{t}) \in \mathcal{D}_{t}$ do
7 $\tilde{x}_{t} \leftarrow \mathcal{P}(x_{t})$ # generates $\tilde{x}_{t}$ from $x_{t}$ and $\delta$ using Eq. 1
8 $y_{s} \leftarrow f_{\theta_{s}}(\tilde{x}_{t})$ # the predicted source domain class
9 count $[y_{s}][y_{t}] \leftarrow \text{count}[y_{s}][y_{t}] + 1$ 10 end
11 index_list $\leftarrow$ ArgSort(count) # get the list of source-target id pairs $(S_{id}, T_{id})$ sorted by count $[S_{id}][T_{id}]$ in descending order
# Define the FreqMap
12 for $(S_{id}, T_{id})$ in index_list do
13 if not done_t $[T_{id}]$ and not done_s $[S_{id}]$ then
14 M $[S_{id}][T_{id}] \leftarrow 1$ # assign the mapping from $S_{id}$ (source label) to $T_{id}$ (target label)
15 done_s $[S_{id}] \leftarrow$ True
16 end
17 if Sum(M[:][Tid]) == m then
18 done_t $[T_{id}] \leftarrow$ True # if the target class $T_{id}$ has been mapped to m source classes
19 end
20 if Sum(done_t) == K_t then
21 break # early stop if all the target classes has been mapped to m source classes
22 end
23 end
24 return M

# A.2.2 ITERMAP (ITERATIVE MAPPING)

In IterMap, at the beginning of each training epoch, the mapping relationship is established using the FreqMap with the current prompted image. This means that the visual prompts updated via the VP model are padded onto the image and participate in the mapping process. Hence, this mapping relationship changes as the visual prompts are trained, which is known as bi-level optimization (Chen et al., 2023b).

# A.2.3 SEMANTICMAP (SEMANTIC MAPPING)

In SemanticMap, source and target classes with semantically similar class names are mapped together. This mapping is accomplished using CLIP's tokenizer Tokenizer() and text encoder Text-Encoder(). We demonstrate the mapping process using the following equation:

$$
C l s T k _ {y _ {\mathrm{s}}} = \text { Tokenizer } (N _ {y _ {\mathrm{s}}})
$$

$$
C l s T k _ {y _ {\mathrm{t}}} = \text { Tokenizer } (N _ {y _ {\mathrm{t}}})
$$

$$
\boldsymbol {E m b} _ {\boldsymbol {y} _ {\mathrm{s}}} = \text { Text - Encoder } (\boldsymbol {C l s T k} _ {\boldsymbol {y} _ {\mathrm{s}}})
$$

$$
\boldsymbol {E m b} _ {\mathbf {y} _ {\mathrm{t}}} = \text { Text - Encoder } (\boldsymbol {C l s T k} _ {\mathbf {y} _ {\mathrm{t}}}) \tag {4}
$$

$$
\text { Similarity } _ {\left(\mathbf {y} _ {\mathrm{s}}, \mathbf {y} _ {\mathrm{t}}\right)} = \frac {\boldsymbol {E m b} _ {\mathbf {y} _ {\mathrm{s}}} \cdot \boldsymbol {E m b} _ {\mathbf {y} _ {\mathrm{t}}}}{\| \boldsymbol {E m b} _ {\mathbf {y} _ {\mathrm{s}}} \| \| \boldsymbol {E m b} _ {\mathbf {y} _ {\mathrm{t}}} \|}
$$

$$
\boldsymbol {y} _ {\mathrm{t}} \leftarrow \boldsymbol {y} _ {\mathrm{s}} ^ {*} = \arg \max _ {y _ {\mathrm{s}} \in \mathcal {Y} _ {\mathrm{s}}} (S i m i l a r i t y _ {(\boldsymbol {y} _ {\mathrm{s}}, \boldsymbol {y} _ {\mathrm{t}})})
$$

In Eq. 4, let $Y_{s} = \{0, \cdots, K_{s} - 1\}$ and $Y_{t} = \{0, \cdots, K_{t} - 1\}$ be the set of source and target labels, where $K_{s}/K_{t}$ are the numbers of source/target classes. For the source label $y_{s} \in Y_{s}$ with the classname $N_{y_{s}}$ and the target label $y_{t} \in Y_{t}$ with the classname $N_{y_{t}}$ , we first utilize the tokenizer to obtain token vectors ( $ClsTk_{y_{s}}$ and $ClsTk_{y_{t}}$ ) corresponding to $N_{y_{s}}$ and $N_{y_{t}}$ . Then, the text encoder is used to obtain their embeddings ( $Emb_{y_{s}}$ and $Emb_{y_{t}}$ ). Pair-wise cosine similarity is calculated between each source and target embeddings, and each target label is mapped to the source label with the highest similarity.

# A.2.4 FULLYMAP (FULLY CONNECTED LAYER MAPPING)

FullyMap utilizes a linear layer $L_{t} = w \cdot L_{s} + b$ with weights w and bias b to learn the mapping, enabling the transformation of the output source logits $L_{s}$ to target logits $L_{t}$ . As for weight initialization (WI), it is employed in the case of CLIP with FullyMap. This technique involves setting the weights between the target labels and their default templates as 1, while setting the rest to 0. With WI, the linear layer can achieve a favorable initial mapping state, thereby expediting the process of obtaining a good mapping relation.

# A.3 AUTOVP TUNING PROCESS

In Figure A.2, there are three stages involved in the tuning process, while the Visual Prompt component depicted in Figure 1 is not involved in the tuning process, as it does not contain any hyper-parameters. During the Input Scaling stage, the initial scale of the input image is determined, and users can choose whether to learn the resize scale during training. In the Pre-trained Classifier stage, users have the option to select from four pre-trained models to serve as the feature extractor. The Output Label Mapping stage offers four mapping methods to choose from. For FreqMap, IterMap, and SemanticMap, users can specify the number of source classes that are mapped to a single target class.

![](images/9f401b013c4653a9bdf465c515292027fbf0e3b61ee5da6c533ed52c88f0343a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image Scale Factor"] --> B["0.5"]
    A --> C["1.0"]
    A --> D["1.5"]
    B --> E["Trainable Resize"]
    C --> F["non-Trainable Resize"]
    E --> G["ResNet18"]
    F --> H["ResNext101-IG"]
    G --> I["Swin-T"]
    H --> I
    I --> J["CLIP"]
    J --> K["FreqMap"]
    J --> L["IterMap"]
    J --> M["SemanticMap"]
    J --> N["FullyMap"]
    K --> O["1"]
    K --> P["5"]
    K --> Q["10"]
    L --> R["Number of Source Classes Mapped to Each Target Class"]
    M --> R
    N --> R
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#fcc,stroke:#333
    style M fill:#fcc,stroke:#333
    style N fill:#fcc,stroke:#333
    style O fill:#fff,stroke:#333
    style P fill:#fff,stroke:#333
    style Q fill:#fff,stroke:#333
```
</details>

Figure A.2: Hyper-Parameter Tuning Selection. Illustration of the end-to-end hyper-parameter tuning process in AutoVP with a total of 222 possible configurations.

# B DATASETS AND BASELINES

# B.1 THE TWELVE DOWNSTREAM DATASETS

To assess the efficacy of the proposed AutoVP, we selected the following twelve datasets for our experiments. The detailed information is shown in B.1.

- CIFAR10 & CIFAR100 (Krizhevsky & Hinton, 2009): The datasets consist of labeled subsets of the 80 million tiny images dataset, which are composed of $32 \times 32$ color images.   
• ISIC (Codella et al., 2019; Tschandl et al., 2018): The International Skin Imaging Collaboration (ISIC) developed an international repository of dermoscopic images known as the ISIC Archive. The images of the datasets were acquired using different devices at several medical centers worldwide.   
- SVHN (Netzer et al., 2011): A real-world image dataset of street view house numbers.   
- GTSRB (Houben et al., 2013): The German Traffic Sign Benchmark (GTSB) is a multi-class, single-image classification challenge that was conducted at the International Joint Conference on Neural Networks (IJCNN) in 2011.   
- Flowers102 (Nilsback & Zisserman, 2008): The 102 categories flowers dataset consists of commonly occurring flowers in the United Kingdom.   
- DTD (Cimpoi et al., 2014): The Describable Textures Dataset (DTD) is a collection of textural images that have been annotated with a series of human-centric attributes.   
- Food101 (Bossard et al., 2014): The food dataset consists of 101 different classes, with a total of 101,000 images.   
- EuroSAT (Helber et al., 2019): The Sentinel-2 satellite images dataset for land use and land cover classification.   
- OxfordIIITPet (Pets) (Parkhi et al., 2012): The dataset includes diverse breeds of cats and dogs, with images exhibiting variations in scale, pose, and lighting conditions.   
- UCF101 (Soomro et al., 2012): The action recognition dataset consists of realistic action videos that have been collected from YouTube.   
- FMoW (Christie et al., 2018): The dataset contains satellite images that are used for sites and land use classification.

Table B.1: Dataset Setting 

<table><tr><td>Dataset</td><td>Class Number</td><td>Training Set Size</td><td>Testing Set Size</td><td>Batch Size</td></tr><tr><td>SVHN</td><td>10</td><td>73,257</td><td>26,032</td><td>128</td></tr><tr><td>EuroSAT</td><td>10</td><td>13,500</td><td>8,100</td><td>128</td></tr><tr><td>Flowers102</td><td>102</td><td>4,093</td><td>2,463</td><td>64</td></tr><tr><td>CIFAR100</td><td>100</td><td>50,000</td><td>10,000</td><td>128</td></tr><tr><td>UCF101</td><td>101</td><td>7,639</td><td>3,783</td><td>128</td></tr><tr><td>DTD</td><td>47</td><td>2,820</td><td>1,692</td><td>32</td></tr><tr><td>FMoW</td><td>62</td><td>76,863</td><td>22,108</td><td>128</td></tr><tr><td>GTSRB</td><td>43</td><td>26,640</td><td>12,630</td><td>128</td></tr><tr><td>CIFAR10</td><td>10</td><td>50,000</td><td>10,000</td><td>128</td></tr><tr><td>Food101</td><td>101</td><td>75,750</td><td>25,250</td><td>128</td></tr><tr><td>OxfordIIITPet</td><td>37</td><td>3,680</td><td>3,669</td><td>128</td></tr><tr><td>ISIC</td><td>7</td><td>4,990</td><td>555</td><td>128</td></tr></table>

# B.2 BASELINES DETAILS

The reported accuracy of the baselines (ILM-VP by Chen et al. (2023b), CLIP-VP and LP by Bahng et al. (2022)) in Section 4 is obtained from the results documented in the respective papers. For some datasets (such as ISIC, FMoW, and GTSRB), the authors did not include them in their paper; in this regard, we follow the corresponding experimental settings to obtain baseline accuracy.

# C ADDITIONAL EXPERIMENTS WITH AUTOVP

# C.1 FIXED PRE-TRAINED MODEL vs. AUTO PRE-TRAINED MODEL SELECTION

We present additional experimental results to highlight the effectiveness of AutoVP. In addition to the comparisons with CLIP and VP baselines discussed in Section 4.1, we further evaluate the performance using ResNet18 as the pre-trained model and explore a scenario without any pre-trained model restrictions. The results using ResNet18 are presented in Table C.1, while the results for the unrestricted scenario are provided in Table C.2. These comparisons consistently demonstrate that AutoVP outperforms previous approaches, including LP and the state-of-the-art VP methods.

Table C.1: AutoVP with ResNet18. Comparison of VP test accuracy (%) using ResNet18 as the pre-trained model on 12 datasets. 

<table><tr><td>Dataset</td><td>AutoVP Setting</td><td>AutoVP</td><td>ILM-VP</td><td>LP</td></tr><tr><td>SVHN</td><td>FullyMap,  $p = 48$ </td><td> $83.74 \pm 0.45$ </td><td>75.2</td><td>65.0</td></tr><tr><td>CIFAR10</td><td>FullyMap,  $p = 48$ </td><td> $87.81 \pm 0.17$ </td><td>65.5</td><td>85.9</td></tr><tr><td>Flowers102</td><td>FullyMap,  $p = 16$ </td><td> $85.40 \pm 1.89$ </td><td>27.9</td><td>88.0</td></tr><tr><td>Food101</td><td>FullyMap,  $p = 16$ </td><td> $54.15 \pm 3.53$ </td><td>14.8</td><td>50.6</td></tr><tr><td>UCF101</td><td>FullyMap,  $p = 16$ </td><td> $55.86 \pm 1.81$ </td><td>23.9</td><td>63.2</td></tr><tr><td>OxfordIIITPet</td><td>FullyMap,  $p = 16$ </td><td> $82.65 \pm 0.84$ </td><td>65.4</td><td>87.2</td></tr><tr><td>CIFAR100</td><td>FullyMap,  $p = 16$ </td><td> $63.67 \pm 3.48$ </td><td>24.8</td><td>63.3</td></tr><tr><td>EuroSAT</td><td>FullyMap,  $p = 48$ </td><td> $93.01 \pm 0.15$ </td><td>85.2</td><td>93.8</td></tr><tr><td>DTD</td><td>FullyMap,  $p = 16$ </td><td> $54.82 \pm 1.14$ </td><td>35.3</td><td>60.0</td></tr><tr><td>ISIC</td><td>FullyMap,  $p = 16$ </td><td> $67.44 \pm 1.22$ </td><td>57.5</td><td>68.6</td></tr><tr><td>FMoW</td><td>FullyMap,  $p = 16$ </td><td> $30.17 \pm 0.06$ </td><td>14.8</td><td>28.4</td></tr><tr><td>GTSRB</td><td>FullyMap,  $p = 16$ </td><td> $81.52 \pm 1.21$ </td><td>52.0</td><td>77.4</td></tr><tr><td colspan="2">Average Accuracy</td><td>70.02</td><td>45.2</td><td>69.3</td></tr></table>

Table C.2: AutoVP with source model selection. This table displays the best tuning setting without any restriction on the choice of pre-trained model, and shows the test accuracy (%) of AutoVP and the LP baseline of the chosen model across 12 datasets. 

<table><tr><td>Dataset</td><td>AutoVP Setting</td><td>AutoVP</td><td>LP</td></tr><tr><td>SVHN</td><td>CLIP, FullyMap,  $p = 51$ </td><td> $92.86 \pm 0.18$ </td><td>65.40</td></tr><tr><td>CIFAR10</td><td>ResNeXt101-IG, FullyMap,  $p = 48$ </td><td> $95.89 \pm 0.07$ </td><td>93.89</td></tr><tr><td>Flowers102</td><td>Swin-T, FullyMap,  $p = 16$ </td><td> $93.48 \pm 0.52$ </td><td>95.75</td></tr><tr><td>Food101</td><td>CLIP, FreqMap-1,  $p = 16$ </td><td> $82.28 \pm 0.09$ </td><td>84.60</td></tr><tr><td>UCF101</td><td>Swin-T, FullyMap,  $p = 16$ </td><td> $72.96 \pm 0.26$ </td><td>75.96</td></tr><tr><td>OxfordIIITPet</td><td>Swin-T, FullyMap,  $p = 16$ </td><td> $90.20 \pm 0.55$ </td><td>93.04</td></tr><tr><td>CIFAR100</td><td>ResNeXt101-IG, FullyMap,  $p = 48$ </td><td> $79.76 \pm 0.47$ </td><td>76.09</td></tr><tr><td>EuroSAT</td><td>Swin-T, FullyMap,  $p = 16$ </td><td> $95.98 \pm 0.02$ </td><td>95.50</td></tr><tr><td>DTD</td><td>Swin-T, FullyMap,  $p = 16$ </td><td> $69.25 \pm 0.58$ </td><td>71.49</td></tr><tr><td>ISIC</td><td>Swin-T, FullyMap,  $p = 16$ </td><td> $71.66 \pm 1.45$ </td><td>72.22</td></tr><tr><td>FMoW</td><td>Swin-T, FullyMap,  $p = 48$ </td><td> $39.79 \pm 0.83$ </td><td>32.73</td></tr><tr><td>GTSRB</td><td>Swin-T, FullyMap,  $p = 55$ </td><td> $88.10 \pm 2.11$ </td><td>74.97</td></tr><tr><td></td><td>Average Accuracy</td><td>81.02</td><td>77.64</td></tr></table>

# C.2 DATA SCALABILITY

In the data scalability experiments, Figure C.1 illustrates the performance of AutoVP and LP on each dataset under various data usage proportions. The corresponding settings can be found in Table C.3. When the chosen pre-trained model is fixed to CLIP, AutoVP outperformed LP in scenarios with limited data on most of the datasets. In some datasets, such as SVHN, GTSRB, and EuroSAT, AutoVP performs better than LP for all proportions of data. Besides, in OxfordIIITPet, CIFAR100, DTD, Food101, UCF101, and Flowers102, AutoVP also obtains promising performance compared to LP on a small amount of data. This suggests that AutoVP will be a more suitable and effective solution than LP, especially when training data is limited.

![](images/9b67ce2fd1e9905310c629ccc81d3aae997102454f85c966d0c1cc3f14d1496f.jpg)

![](images/c1a7e140fac6a8df736a4256c056c94249d5d28214e51a9a8b39bdf93fd4d800.jpg)

<details>
<summary>line</summary>

| x    | Blue Line | Gray Line |
| ---- | --------- | --------- |
| 110  | 80        | 50        |
| 25   | 85        | 60        |
| 50   | 87        | 60        |
| 100  | 88        | 65        |
</details>

![](images/2626d8d1b009fdd8741768f4032b23e58f8c862c45a576f550173bb145d3cb1f.jpg)

<details>
<summary>line</summary>

| Percentage (%) | Blue Line | Gray Line |
| -------------- | --------- | --------- |
| 110            | 62        | 57        |
| 25             | 83        | 83        |
| 50             | 90        | 84        |
| 100            | 92        | 86        |
</details>

![](images/c8efb22760d9ff5bdb6d47c1c9bf694fb0957c1d29a3fe98eea0add3ab1f074c.jpg)

<details>
<summary>line</summary>

| X (%) | Blue Line | Gray Line |
|-------|-----------|-----------|
| 110   | 86        | 83        |
| 25    | 94        | 90        |
| 50    | 96        | 93        |
| 100   | 97        | 95        |
</details>

![](images/667bac6a18d07601f537ff3c13a17e1060231175e83c420503d77da3d206720e.jpg)

<details>
<summary>line</summary>

| Percentage (%) | Line 1 | Line 2 |
| -------------- | ------ | ------ |
| 110            | 75     | 25     |
| 25             | 85     | 75     |
| 50             | 85     | 80     |
| 100            | 85     | 85     |
</details>

![](images/66b52042c03208eac9b44c91a6cb43eb67420245e5db221acbce67ca4297322f.jpg)

<details>
<summary>line</summary>

| Percentage (%) | Blue Line | Gray Line |
| -------------- | --------- | --------- |
| 10             | 65        | 48        |
| 25             | 72        | 73        |
| 50             | 76        | 75        |
| 100            | 79        | 79        |
</details>

![](images/e60f428c90a963b1af6664182412fa57c99d3421cf55a123d808816c213a82c3.jpg)

<details>
<summary>line</summary>

| X (%) | Blue Line | Gray Line |
|-------|-----------|-----------|
| 110   | 25        | 15        |
| 25    | 50        | 60        |
| 50    | 55        | 65        |
| 100   | 60        | 70        |
</details>

![](images/7e7eef2b9f2f99b76efdde428959a8350697f77edaa3820239e5fbda43b93665.jpg)

<details>
<summary>line</summary>

| X (%) | Line 1 | Line 2 |
|-------|--------|--------|
| 110   | 64     | 79     |
| 25    | 82     | 80     |
| 50    | 83     | 81     |
| 100   | 84     | 82     |
</details>

![](images/62d4785d3a364ea9018f6f138f7dd98a2afdeda3ef5097bf2795c00dca0eff2c.jpg)

<details>
<summary>line</summary>

| X (%) | Blue Line | Gray Line |
|-------|-----------|-----------|
| 110   | 40        | 20        |
| 25    | 65        | 75        |
| 50    | 75        | 80        |
| 100   | 75        | 85        |
</details>

![](images/15909c496356ab7df126621b0095bb88df4b49a86b9c60f849b606dc012d6d13.jpg)

<details>
<summary>line</summary>

| %    | Series 1 | Series 2 |
| ---- | -------- | -------- |
| 110  | 25       | 30       |
| 25   | 65       | 70       |
| 50   | 85       | 75       |
| 100  | 95       | 90       |
</details>

![](images/5db942507ae58f76db1cb9d69d647327f8e00b9658fd2e59db651cf2c8d8168b.jpg)

<details>
<summary>line</summary>

| Percentage (%) | Blue Line | Gray Line |
| -------------- | --------- | --------- |
| 110            | 90        | 91        |
| 25             | 92        | 92        |
| 50             | 94        | 92        |
| 100            | 95        | 95        |
</details>

![](images/c0c58b4d6e1779d95d655288e7aa48553d84fdcd12a89ddf119f60720e71f208.jpg)

<details>
<summary>line</summary>

| %    | ISIC (Blue Line) | ISIC (Gray Line) |
| ---- | ---------------- | ---------------- |
| 10   | 45               | 48               |
| 25   | 60               | 65               |
| 50   | 68               | 69               |
| 100  | 73               | 71               |
</details>

![](images/61041f2717607a710d00efea3bf5c62e649774b44b904871930ddbfdef6adc60.jpg)

<details>
<summary>line</summary>

| Percentage (%) | Blue Line | Gray Line |
| -------------- | --------- | --------- |
| 110            | 22        | 28        |
| 25             | 37        | 30        |
| 50             | 38        | 31        |
| 100            | 42        | 37        |
</details>

Figure C.1: Data Scalability. The charts present the accuracy of AutoVP and LP with varying percentages of data usage: 100%, 50%, 25%, 10%, and 1%.

Table C.3: Data Scalability Settings. The table shows the settings for data scalability experiments with data usage ranging from 100% to 1%. The source pre-trained model is fixed to CLIP. The notation “Mapping-m” represents mapping m source classes to each target class. 

<table><tr><td>Dataset</td><td>100%</td><td>50%</td><td>25%</td><td>10%</td><td>1%</td></tr><tr><td>SVHN</td><td>FullyMap, p = 53</td><td>FullyMap, p = 51</td><td>FullyMap, p = 49</td><td>FullyMap, p = 80</td><td>FullyMap, p = 80</td></tr><tr><td>CIFAR10</td><td>IterMap-1, p = 22</td><td>FreqMap-1, p = 16</td><td>FreqMap-5, p = 16</td><td>IterMap-5, p = 16</td><td>IterMap-1, p = 16</td></tr><tr><td>Flowers102</td><td>FullyMap, p = 16</td><td>FreqMap-1, p = 16</td><td>FreqMap-1, p = 16</td><td>IterMap-1, p = 16</td><td>FullyMap, p = 16</td></tr><tr><td>Food101</td><td>FreqMap-1, p = 16</td><td>IterMap-5, p = 16</td><td>IterMap-1, p = 16</td><td>IterMap-1, p = 16</td><td>FullyMap, p = 16</td></tr><tr><td>UCF101</td><td>FullyMap, p = 17</td><td>FullyMap, p = 16</td><td>IterMap-1, p = 16</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td></tr><tr><td>OxfordIIITPet</td><td>FreqMap-10, p = 16</td><td>FreqMap-5, p = 16</td><td>FreqMap-5, p = 16</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td></tr><tr><td>CIFAR100</td><td>FullyMap, p = 30</td><td>FullyMap, p = 26</td><td>IterMap-1, p = 16</td><td>FreqMap-1, p = 16</td><td>FullyMap, p = 16</td></tr><tr><td>EuroSAT</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td></tr><tr><td>DTD</td><td>FullyMap, p = 18</td><td>IterMap-10, p = 16</td><td>IterMap-5, p = 16</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td></tr><tr><td>ISIC</td><td>IterMap-1, p = 16</td><td>FreqMap-1, p = 16</td><td>FullyMap, p = 80</td><td>FreqMap-5, p = 48</td><td>FullyMap, p = 80</td></tr><tr><td>FMoW</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td><td>FullyMap, p = 16</td><td>FreqMap-10, p = 16</td><td>FullyMap, p = 17</td></tr><tr><td>GTSRB</td><td>FullyMap, p = 80</td><td>FullyMap, p = 16</td><td>FullyMap, p = 79</td><td>FullyMap, p = 48</td><td>FullyMap, p = 80</td></tr></table>

# C.3 DOWNSTREAM DATASET ANALYSIS (ID/OOD vs. ACCURACY GAIN)

To further understand how the downstream dataset distribution influences the performance of visual prompting. We conduct experiments to observe the relation between accuracy gain and dataset characteristics. When using CLIP as the pre-trained model, we define confidence score, obtained by averaging the maximum softmax probability of predictions across the entire training set, as an indicator of dataset characteristics. For other vision pre-trained models, the FID score is used to measure the dissimilarity between the downstream dataset and the ImageNet-1K dataset, i.e. the degree of out-of-distribution (OOD). We present the Confidence/FID scores of each dataset in Figure C.2, providing insights into their OOD characteristics. Furthermore, Figure C.3 demonstrates the accuracy gain for each dataset when using ResNet18 as the pre-trained model. The experimental results show that when using AutoVP, datasets with a higher degree of OOD tend to benefit from more accuracy gains.

![](images/8f71253d5c7c183ccbb3213caec9b7ac334c69c289e687d99fa9ce3ca5587696.jpg)

<details>
<summary>scatter</summary>

Confidence Score
| Dataset | Confidence Score |
| :--- | :--- |
| CIFAR16 | 0.83 |
| OxfordHITRet | 0.80 |
| Food101 | 0.77 |
| UCF101 | 0.66 |
| Flowers102 | 0.64 |
| CIFAR100 | 0.53 |
| DTD | 0.50 |
| FMoW | 0.48 |
| EuroSAT | 0.45 |
| ISIC | 0.40 |
| GTSRB | 0.23 |
| SVMN | 0.20 |
ID indicated below the left region; OOD indicated above the right region.
</details>

![](images/7d65cade5821897b9ca4425cec31c148202bfc706940515a1e0ce8c64892e7b4.jpg)

<details>
<summary>scatter</summary>

FID Score
| Dataset | FID Score |
| :--- | :--- |
| OxfordHITPet | 75.3 |
| UCF101 | 85.0 |
| DTD | 95.1 |
| Food101 | 103.8 |
| CIFAR100 | 119.8 |
| CIFAR10 | 134.5 |
| FMoW | 139.9 |
| Flowers102 | 141.6 |
| EuroSAT | 150.7 |
| GTSRB | 158.5 |
| SVHN | 211.2 |
| ISIC | 248.6 |
The chart is divided into two regions: ID (left) and OOD (right). The data points are labeled with their respective numerical values. The chart is used to display a ranking or comparison of FID scores across different datasets.
</details>

Figure C.2: Confidence Score and FID Score. We sort the datasets by the degree of out-of-distribution (OOD), where the one closer to the left indicates higher similarity (in-distribution, ID) to the training data of the pre-trained model, while the one closer to the right indicates greater dissimilarity (OOD).

![](images/401efca3ae3df1d57408b66db3c63ac5e8b064c80437bb039951a362dd24342a.jpg)

<details>
<summary>scatter</summary>

| Model     | FID Score | Accuracy Gain (%) |
|-----------|-----------|-------------------|
| SVHN      | ~210      | ~20               |
| GTSRB     | ~160      | ~5                |
| CIFAR10   | ~130      | ~2                |
| EuroSAT   | ~140      | ~0                |
| Flowers102| ~140      | ~-5               |
| EuroSAT   | ~140      | ~-2               |
| EuroSAT   | ~140      | ~-1               |
| EuroSAT   | ~140      | ~-3               |
| EuroSAT   | ~140      | ~-4               |
| EuroSAT   | ~140      | ~-5               |
| EuroSAT   | ~140      | ~-6               |
| EuroSAT   | ~140      | ~-7               |
| EuroSAT   | ~140      | ~-8               |
| EuroSAT   | ~140      | ~-9               |
| EuroSAT   | ~140      | ~-10              |
| EuroSAT   | ~140      | ~-11              |
| EuroSAT   | ~140      | ~-12              |
| EuroSAT   | ~140      | ~-13              |
| EuroSAT   | ~140      | ~-14              |
| EuroSAT   | ~140      | ~-15              |
| EuroSAT   | ~140      | ~-16              |
| EuroSAT   | ~140      | ~-17              |
| EuroSAT   | ~140      | ~-18              |
| EuroSAT   | ~140      | ~-19              |
| EuroSAT   | ~140      | ~-20              |
| EuroSAT   | ~140      | ~-21              |
| EuroSAT   | ~140      | ~-22              |
| EuroSAT   | ~140      | ~-23              |
| EuroSAT   | ~140      | ~-24              |
| EuroSAT   | ~140      | ~-25              |
| EuroSAT   | ~140      | ~-26              |
| EuroSAT   | ~140      | ~-27              |
| EuroSAT   | ~140      | ~-28              |
| EuroSAT   | ~140      | ~-29              |
| EuroSAT   | ~140      | ~-30              |
| EuroSAT   | ~140      | ~-31              |
| EuroSAT   | ~140      | ~-32              |
| EuroSAT   | ~140      | ~-33              |
| EuroSAT   | ~140      | ~-34              |
| EuroSAT   | ~140      | ~-35              |
| EuroSAT   | ~140      | ~-36              |
| EuroSAT   | ~140      | ~-37              |
| EuroSAT   | ~140      | ~-38              |
| EuroSAT   | ~140      | ~-39              |
| EuroSAT   | ~140      | ~-40              |
| EuroSAT   | ~140      | ~-41              |
| EuroSAT   | ~140      | ~-42              |
| EuroSAT   | ~140      | ~-43              |
| EuroSAT   | ~140      | ~-44              |
| EuroSAT   | ~140      | ~-45              |
| EuroSAT   | ~140      | ~-46              |
| EuroSAT   | ~140      | ~-47              |
| EuroSAT   | ~140      | ~-48              |
| EuroSAT   | ~140      | ~-49              |
| EuroSAT   | ~140      | ~-50              |
| EuroSAT   | ~140      | ~-51              |
| EuroSAT   | ~140      | ~-52              |
| EuroSAT   | ~140      | ~-53              |
| EuroSAT   | ~140      | ~-54              |
| EuroSAT   | ~140      | ~-55              |
| EuroSAT   | ~140      | ~-56              |
| EuroSAT   | ~140      | ~-57              |
| EuroSAT   | ~140      | ~-58              |
| EuroSAT   | ~140      | ~-59              |
| EuroSAT   | ~140      | ~-60              |
| EuroSAT   | ~140      | ~-61              |
| EuroSAT   | ~140      | ~-62              |
| EuroSAT   | ~140      | ~-63              |
| EuroSAT   | ~140      | ~-64              |
| EuroSAT   | ~140      | ~-65              |
| EuroSAT   | ~140      | ~-66              |
| EuroSAT   | ~140      | ~-67              |
| EuroSAT   | ~140      | ~-68              |
| EuroSAT   | ~140      | ~-69              |
| EuroSAT   | ~140      | ~-70              |
| EuroSAT   | ~140      | ~-71              |
| EuroSAT   | ~140      | ~-72              |
| EuroSAT   | ~140      | ~-73              |
| EuroSAT   | ~140      | ~-74              |
| EuroSAT   | ~140      | ~-75              |
| EuroSAT   | ~140      | ~-76              |
| EuroSAT   | ~140      | ~-77              |
| EuroSAT   | ~140      | ~-78              |
| EuroSAT   | ~140      | ~-79              |
| EuroSAT   | ~140      | ~-80              |
| EuroSAT   | ~140      | ~-81              |
| EuroSAT   | ~140      | ~-82              |
| EuroSAT   | ~140      | ~-83              |
| EuroSAT   | ~140      | ~-84              |
| EuroSAT   | ~140      | ~-85              |
| EuroSAT   | ~140      | ~-86              |
| EuroSAT   | ~140      | ~-87              |
| EuroSAT   | ~140      | ~-88              |
| EuroSAT   | ~140      | ~-89              |
| EuroSAT   | ~140      | ~-90              |
| EuroSAT   | ~140      | ~-91              |
| EuroSAT   | ~140      | ~-92              |
| EuroSAT   | ~140      | ~-93              |
| EuroSAT   | ~140      | ~-94              |
| EuroSAT   | ~140      | ~-95              |
| EuroSAT   | ~140      | ~-96              |
| EuroSAT   | ~140      | ~-97              |
| EuroSAT   | ~140      | ~-98              |
| EuroSAT   | ~140      | ~-99              |
| EuroSAT   | ~140      | > 0               |
| FoodSAT   | 95        | -5                |
| FoodSAT   | 95        | -6                |
| FoodSAT   | 95        | -7                |
| FoodSAT   | 95        | -8                |
| FoodSAT   | 95        | -9                |
| FoodSAT   | 95        | -10               |
| FoodSAT   | 95        | -11               |
| FoodSAT   | 95        | -12               |
| FoodSAT   | 95        | -13               |
| FoodSAT   | 95        | -14               |
| FoodSAT   | 95        | -15               |
| FoodSAT   | 95        | -16               |
| FoodSAT   | 95        | -17               |
| FoodSAT   | 95        | -18               |
| FoodSAT   | 95        | -19               |
| FoodSAT   | 95        | -20               |
| FoodSAT   | 95        | -21               |
| FoodSAT   | 95        | -22               |
| FoodSAT   | 95        | -23               |
| FoodSAT   | 95        | -24               |
| FoodSAT   | 95        | -25               |
| FoodSAT   | 95        | -26               |
| FoodSAT   | 95        | -27               |
| FoodSAT   | 95        | -28               |
| FoodSAT   | 95        | -29               |
| FoodSAT   | 95        | -30               |
| FoodSAT   | 95        | -31               |
| FoodSAT   | 95        | -32               |
| FoodSAT   | 95        | -33               |
| FoodSAT   | 95        | -34               |
| FoodSAT   | 95        | -35               |
| FoodSAT   | 95        | -36               |
| FoodSAT   | 95        | -37               |
| FoodSAT   | 95        | -38               |
| FoodSAT   | 95        | -39               |
| FoodSAT   | 95        | -40               |
| FoodSAT   | 95        | -41               |
| FoodSAT   | 95        | -42               |
| FoodSAT   | 95        | -43               |
| FoodSAT   | 95        | -44               |
| FoodSAT   | 95        | -45               |
| FoodSAT   | 95        | -46               |
| FoodSAT   | 95        | -47               |
| FoodSAT   | 95        | -48               |
| FoodSAT   | 95        | -49               |
| FoodSAT   | 95        | -50               |
| FoodSAT   | 95        | -51               |
| FoodSAT   | 95        | -52               |
| FoodSAT   | 95        | -53               |
| FoodSAT   | 95        | -54               |
| FoodSAT   | 95        | -55               |
| FoodSAT   | 95        | -56               |
| FoodSAT   | 95        | -57               |
| FoodSAT   | 95        | -58               |
| FoodSAT   | 95        | -59               |
| FoodSAT   | 95        | -60               |
| FoodSAT   | 95        | -61               |
| FoodSAT   | 95        | -62               |
| FoodSAT   | 95        | -63               |
| FoodSAT   | 95        | -64               |
| FoodSAT   | 95        | -65               |
| FoodSAT   | 95        | -66               |
| FoodSAT   | 95        | -67               |
| FoodSAT   | 95        | -68               |
| FoodSAT   | 95        | -69               |
| FoodSAT   | 95        | -70               |
| FoodSAT   | 95        | -71               |
| FoodSAT   | 95        | -72               |
| FoodSAT   | 95        | -73               |
| FoodSAT   | 95        | -74               |
| FoodSAT   | 95        | -75               |
| FoodSAT   | 95        | -76               |
| FoodSAT   | 95        | -77               |
| FoodSAT   | 95        | -78               |
| FoodSAT   | 95        | -79               |
| FoodSAT   | 95        | -80               |
| FoodSAT   | 95        | -81               |
| FoodSAT   | 95        | -82               |
| FoodSAT   | 95        | -83               |
| FoodSAT   | 95        | -84               |
| FoodSAT   | 95        | -85               |
| FoodSAT   | 95        | -86               |
| FoodSAT   | 95        | -87               |
| FoodSAT   | 95        | -88               |
| FoodSAT   | 95        | -89               |
| FoodSAT   | 95        | -90               |
| FoodSAT   | 95        | -91               |
| FoodSAT   | 95        | -92               |
| FoodSAT   | 95        | -93               |
| FoodSAT   | 95        | -94               |
| FoodSAT   | 95        | -95               |
| FoodSAT   | 95        | -96               |
| FoodSAT   | 95        | -97               |
| FoodSAT   | 95        | -98               |
| FoodSAT   | 95        | -99               |
| FoodSAT   | 95        | -100              |

The chart displays the FID Score and Accuracy Gain for each model. The data is extracted from the provided code and presented in CSV format as requested. The values for the Accuracy Gain are explicitly labeled on the chart. The FID Score is shown as a numerical value next to the chart. The Accuracy Gain is calculated by subtracting the FID Score from the chart. Since the original FID Score is not specified in the code, I have indicated it based on the original FID Score. Therefore, the actual FID Score values are not provided in the code. The actual Accuracy Gain values are not provided in the chart.
</details>

![](images/20f43f7f085adf4a6b6b0b92e75021c15590e984fe803d9901f86646298d4dd0.jpg)

<details>
<summary>scatter</summary>

| Label     | FID Score | Value |
| --------- | --------- | ----- |
| SVHN      | 210       | 20    |
| GTSRB     | 160       | 8     |
| ISIC      | 250       | 5     |
| EuroSAT   | 140       | 1     |
| Cifar10   | 130       | 0     |
| FMoW      | 140       | 0     |
| Flowers102| 140       | -1    |
| EuroSAT   | 140       | 0     |
| Cifar100  | 120       | -3    |
| Food101   | 100       | -5    |
| DTD       | 90        | -2    |
| UCF101    | 80        | -3    |
| Pets      | 70        | -4    |
</details>

Figure C.3: Accuracy Gains with Resnet18. The gains are calculated by taking the difference between the performance of AutoVP and LP or Non-VP scenario.

Table 2 has encompassed the prevalent 12 datasets in VP research. Furthermore, to enable a more comprehensive comparison within a wider spectrum of VP research, we have included additional datasets—SUN397(Xiao et al., 2010), RESISC(Cheng et al., 2017), CLEVR(Johnson et al., 2017), and StanfordCar(Krause et al., 2013)—with their respective results available in Table C.4. Our findings consistently demonstrate AutoVP's superior performance compared to ILM and CLIP-VP across the most of datasets. Especially notable is the substantial 15% increase in accuracy observed in the out-of-distribution (OOD) dataset, CLEVR, when compared to linear probing (LP). However, the in-distribution (ID) datasets, SUN397 and StanfordCar, showcased lower performance than LP, aligning with the trend of accuracy gain illustrated in Figure 5 in Section 5.

Table C.4: Performance on Additional Datasets. The testing accuracy (%) for four additional datasets commonly found in VP research. The figure on the right side shows the accuracy gains. 

<table><tr><td>Dataset</td><td>AutoVP Setting</td><td>AutoVP</td><td>LP</td><td>ILM</td><td>CLIP-VP</td></tr><tr><td>SUN397</td><td>FullyMap, p = 16</td><td>65.4</td><td>70.9</td><td>61.2</td><td>60.5</td></tr><tr><td>StanfordCar</td><td>FullyMap, p = 16</td><td>61.8</td><td>77.6</td><td>57.6</td><td>56.2</td></tr><tr><td>RESISC</td><td>FullyMap, p = 17</td><td>88.5</td><td>91.7</td><td>86.6</td><td>84.5</td></tr><tr><td>CLEVR</td><td>FullyMap, p = 16</td><td>83.0</td><td>68.0</td><td>83.1</td><td>81.4</td></tr></table>

![](images/8631b5ef225c217c08470da21eae7ce85fb50cf7d0df80233d8b9b13aae54bc4.jpg)

<details>
<summary>scatter</summary>

| Model      | Confidence Score | Accuracy Gain (%) |
| ---------- | ---------------- | ----------------- |
| CLEVR      | 0.2              | 15                |
| RESISC     | 0.6              | -5                |
| SUN397     | 0.6              | -10               |
| StanfordCar| 0.6              | -15               |
</details>

# D ANALYSIS OF AUTOVP RESULTS

# D.1 PROMPTS IN FREQUENCY DOMAIN

We have analyzed the learned prompts using the best setting selected from AutoVP and have represented them in the frequency domain through Fast Fourier transformation (Brigham, 1988). In Figure D.1(a), the prompting result of SVHN dataset with CLIP is notably distinct and achieves the highest testing accuracy among all the pre-trained models. In the frequency analysis (Figure D.1(b)), the prompts are concentrated in the low-frequency domain (at the center of the plot), with CLIP displaying the most distinct structure and significantly larger magnitudes compared to the others. These results validate the efficient learning of prompts with CLIP, harnessing low-frequency features that generalize to the target domain.

![](images/1e98c034569fa0a79afd5e627fc7d0858ba5e6941fd5367070a7d783934196c3.jpg)

![](images/ed52a0f76f2fa063499908595416d51dbb3ccfcee50cce806bc31ff91fd8e1e0.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| 0 |   |   |   |   |   |   |   |   |   |
| 50 |   |   |   |   |   |   |   |   |   |
| 100 |   |   |   |   |   |   |   |   |   |
| 150 |   |   |   |   |   |   |   |   |   |
| 200 |   |   |   |   |   |   |   |   |   |
</details>

![](images/9986448bd07730cb620907a49c658317d89c8a94da01dc7880c2a97f4d232aa5.jpg)

<details>
<summary>natural_image</summary>

Grayscale image with no visible text, numbers, or symbols, showing a uniform textured background.
</details>

![](images/6b59148d040eca1b7388670e79bdc22afd3b6e0fc9a1840fca0628b4d5b678fa.jpg)

<details>
<summary>text_image</summary>

AutoVP with Swin-T
</details>

![](images/217a674c363b0cf5c01f0c29041ec62d9f94ad079e0635c6266ebb732e5ff2b8.jpg)

<details>
<summary>heatmap</summary>

| X Range | Y Range | Value (e^8) |
|---------|---------|-------------|
| 0-25    | 0-25    | ~0          |
| 0-25    | 25-50   | ~0          |
| 0-25    | 50-75   | ~0          |
| 0-25    | 75-100  | ~0          |
| 0-25    | 100-125 | ~0          |
| 0-25    | 125-150 | ~0          |
| 0-25    | 150-175 | ~0          |
| 0-25    | 175-200 | ~0          |
| 25-50   | 0-25    | ~0          |
| 25-50   | 25-50   | ~0          |
| 25-50   | 50-75   | ~0          |
| 25-50   | 75-100  | ~0          |
| 25-50   | 100-125 | ~0          |
| 25-50   | 125-150 | ~0          |
| 25-50   | 150-175 | ~0          |
| 25-50   | 175-200 | ~0          |
| 50-75   | 0-25    | ~0          |
| 50-75   | 25-50   | ~0          |
| 50-75   | 50-75   | ~0          |
| 50-75   | 75-100  | ~0          |
| 50-75   | 100-125 | ~0          |
| 50-75   | 125-150 | ~0          |
| 50-75   | 150-175 | ~0          |
| 50-75   | 175-200 | ~0          |
| 75-100  | 0-25    | ~0          |
| 75-100  | 25-50   | ~0          |
| 75-100  | 50-75   | ~0          |
| 75-100  | 75-100  | ~0          |
| 75-100  | 100-125 | ~0          |
| 75-100  | 125-150 | ~0          |
| 75-100  | 150-175 | ~0          |
| 75-100  | 175-200 | ~0          |
| 100-125 | 0-25    | ~0          |
| 100-125 | 25-50   | ~0          |
| 100-125 | 50-75   | ~0          |
| 100-125 | 75-100  | ~0          |
| 100-125 | 100-125 | ~0          |
| 100-125 | 125-150 | ~0          |
| 100-125 | 150-175 | ~0          |
| 100-125 | 175-200 | ~0          |
| 125-150 | 0-25    | ~0          |
| 125-150 | 25-50   | ~0          |
| 125-150 | 50-75   | ~0          |
| 125-150 | 75-100  | ~0          |
| 125-150 | 100-125 | ~0          |
| 125-150 | 125-150 | ~0          |
| 125-150 | 150-175 | ~0          |
| 125-150 | 175-200 | ~0          |
| 150-175 | 0-25    | ~0          |
| 150-175 | 25-50   | ~0          |
| 150-175 | 50-75   | ~0          |
| 150-175 | 75-100  | ~0          |
| 150-175 | 100-125 | ~0          |
| 150-175 | 125-150 | ~0          |
| 150-175 | 150-175 | ~0          |
| 150-175 | 175-200 | ~0          |
| 175-200 | 0-25    | ~0          |
| 175-200 | 25-50   | ~0          |
| 175-200 | 50-75   | ~0          |
| 175-200 | 75-100  | ~0          |
| 175-200 | 100-125 | ~0          |
| 175-200 | 125-150 | ~0          |
| 175-200 | 150-175 | ~0          |
| 175-200 | 175-200 | ~0          |
| >2.3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333<fcel>
</details>

(b) Prompts in Frequency Domain   
Figure D.1: Prompting Analysis of SVHN Using Various Pre-trained Models. (a) Frame-shape prompts learned with the best settings selected from AutoVP for a given pre-trained model. (b) Prompts in the frequency domain by Fast Fourier transformation.

# D.2 OUTPUT MAPPING ANALYSIS

(a) The Weight Distribution of bumpy in the FullyMap   
![](images/8b5e1f53247aa87516069765685a0877be1f5a073efb333b1670cc179a56c661.jpg)  
(b) The Weight Distribution of scaly in the FullyMap

Figure D.2: The Weight Distribution of FullyMap in DTD. The top 5 source labels exhibiting the highest weight values within the FullyMap pertain to (a) the label bumpy and (b) the label scaly.   
Figure D.2 illustrates that FullyMap can be interpreted as a weighted combination of multiple source labels, where some human-readable features may exhibit similarity. For instance, in Figure D.2(a), bumpy shows similarities with custard apple, disk brake, and pineapple, while in Figure D.2(b), scaly shares similar features with boa constrictor, coho, and common iguana.   
![](images/33241126d3eacc3ef99b7649ef1f0624fd9285ec2423a7f31c0b463dfd01d5fb.jpg)  
Figure D.3: The Correspondence Between Output Mapping Labels in DTD. Each column represents the mapping between the target label and its respective top-1 source label in FullyMap. (a) FullyMap (top-1 class having the largest weight) with Swin-T (second row). (b) IterMap-1 with Swin-T (third row).

Furthermore, when comparing FullyMap and IterMap, a significant accuracy gap is observed, with FullyMap achieving 69.96% and IterMap-1 only achieving 40.77%. However, in Figure D.3, IterMap has mapped to some classes that are indeed very close to the target. For instance, in Figure D.3(b), braided maps to knot, bubbly maps to bubble, and cobwebbed maps to spider web. This demonstrates that a mere combination of source labels is insufficient for achieving better performance; the weighting in the combination plays a crucial role, which is precisely what FullyMap accomplishes.

D.3 THE PREFERENCES IN HYPER-PARAMETER TUNING SELECTION   
![](images/3be7a483381b7c39ee648cd5bad76fe20f9db4bac94e215f111b2092ae3ebd71.jpg)  
Figure D.4: Average Tuning Results. The three charts display the average few-epoch tuning accuracy for different selected conditions.

Figure D.4 illustrates the average tuning results for all datasets across different settings, including image scale, mapping methods, pre-trained models, and so on (see Figure A.2). In Figure D.4(a), among the vision models (ResNet18, ResNeXt-IG, Swin-T), the FullyMap demonstrates superior performance compared to the other methods. However, for CLIP, a vision-language model, the FullyMap only shows a slight advantage over the others. Furthermore, Figure D.4(b) indicates that larger initial scales, such as 1.0 or 1.5 (yielding square images with a width of $128 \times 1.0$ or $128 \times 1.5$ ), generally lead to better results when using FreqMap, IterMap, or FullyMap. Last, since both FreqMap and IterMap can configure the number $(m)$ of source labels mapped to each target class, we found that increasing this count generally improves accuracy for the three vision models (see Figure D.4(c)). However, for CLIP, mapping five source labels appears to be the optimal choice based on the average tuning results.

# E ABLATION STUDIES

# E.1 THE IMPACT OF TEXT ENCODER IN CLIP

Figure E.1 illustrates the impact on accuracy when incorporating or excluding the CLIP text encoder. On average, this configuration results in a significant decrease in accuracy of approximately 12%, highlighting the crucial role of the text encoder in VP with CLIP.

![](images/0fc8770d4a7b248b88fbda612ca34752fcbf47042e6a235c030dcca0d8d626ac.jpg)

<details>
<summary>bar</summary>

| Dataset     | AutoVP FullyM | AutoVP FullyM w/o Text Encoder |
| ----------- | ------------- | ------------------------------ |
| SVHN        | 0.02          | 93.0                           |
| EuroSAT     | -0.81         | 97.0                           |
| Flowers102  | -67.50        | 23.0                           |
| CIFAR100    | 2.16          | 80.0                           |
| UCF101      | 5.03          | 75.0                           |
| DTD         | 5.78          | 63.0                           |
| FMoW        | 0.34          | 42.0                           |
| GTSRB       | 2.05          | 95.0                           |
| CIFAR10     | -0.93         | 94.0                           |
| Food101     | -30.05        | 51.0                           |
| Pet         | -46.35        | 42.0                           |
| ISIC        | -10.44        | 65.0                           |
</details>

Figure E.1: The Accuracy Difference of AutoVP Variations. The chart shows the difference in accuracy between AutoVP variations with and without the CLIP text encoder. The numbers above the bars indicate the accuracy difference.

# E.2 VISUAL PROMPTING IN SEGMENTATION AND DETECTION TASKS

![](images/ac0186885389883f2de09b172d888284cbc08a32c94e766fe2fad2e47920ba2c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a) Segmentation
        A1["Image"] --> B1["Pre-Trained Model"]
        B1 --> C1["Classifier Conv2d(256, 21) --> Conv2d(256, 3)"]
        C1 --> D1["Visual Prompts"]
    end

    subgraph (b) Detection
        E1["Image"] --> F1["Pre-Trained Model"]
        F1 --> G1["Box Head Multilayer Perceptron (MLP)"]
        G1 --> H1["Visual Prompts"]
    end

    subgraph (a-1) Linear Probing
        I1["Image"] --> J1["Pre-Trained Model"]
        J1 --> K1["FullyMap Conv2d(21, 3)"]
        K1 --> L1["Visual Prompts"]
    end

    subgraph (b-1) Linear Probing
        M1["Image"] --> N1["Pre-Trained Model"]
        N1 --> O1["Visual Prompts"]
        O1 --> P1["Visual Prompts"]
        P1 --> Q1["Visual Prompts"]
    end

    style (a) Segmentation fill:#f9f,stroke:#333
    style (b) Detection fill:#bbf,stroke:#333
```
</details>

Figure E.2: LP and AutoVP in Segmentation and Detection Tasks.

Although AutoVP primarily focuses on classification tasks, we aimed to delve deeper into various vision tasks by incorporating models designed for different purposes. For segmentation tasks, we employ DeepLabV3 (Chen et al., 2017) as the pre-trained backbone, and for detection tasks, we utilize OWL-ViT (Minderer et al., 2022). To evaluate the capability of both in-distribution (ID) and out-of-distribution (OOD) datasets, we chose ISIC as the OOD dataset for both tasks. For ID datasets, we use Pets in the segmentation task and VOC (Everingham et al., 2015) in the detection task.

Our VP segmentation framework, depicted in Figure E.2 (a), integrates a FullyMap after the pretrained model to facilitate pixel-wise classification using a custom class number. In contrast, the linear probing approach modifies the final 2D convolutional layer, and the results are depicted in Table E.1.

Table E.1: Performance on Segmentation Datasets. The IoU and pixel accuracy (%) of linear probing (LP) and AutoVP. 

<table><tr><td>Dataset</td><td>LP</td><td>AutoVP</td></tr><tr><td>Pets</td><td>IoU : 0.83, Pixel : 90.7%</td><td>IoU : 0.77, Pixel : 86.9%</td></tr><tr><td>ISIC</td><td>IoU : 0.64, Pixel : 78.1%</td><td>IoU : 0.81, Pixel : 89.5%</td></tr></table>

We evaluated segmentation performance using two metrics: IoU (Intersection over Union) score and pixel-wise classification accuracy. AutoVP exhibited superior performance on both metrics on the ISIC dataset. Additionally, segmentation examples highlighted that predictions align more accurately with the ground truth mask when the prompt space is larger (see Figure E.3). However, in the ID dataset (Pets), VP performance was inferior to LP. This aligns with our findings in the classification task, where OOD datasets derived greater benefits from visual prompts.

![](images/cdf469026c638072eb364ef539c64768eccc2292d78d328bdfab16aa920c22ec.jpg)

<details>
<summary>heatmap</summary>

| Method | IoU | Pixel (%) |
| --- | --- | --- |
| Linear Probing | 0.64 | 78.1 |
| AutoVP (p=16) | 0.72 | 83.8 |
| AutoVP (p=48) | 0.81 | 89.5 |
</details>

Figure E.3: ISIC Segmentation. Performance of LP and AutoVP with different prompt sizes $(p)$ . It's important to note that the performance calculation involves only the target image itself without the prompts, ensuring a fair comparison to LP. Specifically, the predicted mask is initially cropped to the region of the original target image and then resized to match the dimensions used in LP.

In the case of the detection task, as illustrated in Figure E.2 (b), the LP approach fine-tunes on the box predictor head, whereas the VP method incorporates trainable visual prompts. To achieve consistent box output, we have omitted the inclusion of output mapping. The results are shown in Table E.2. Similar to segmentation, the OOD dataset, ISIC, exhibits better performance than LP with a larger prompt size.

Table E.2: Performance on Detection Datasets The IoU scores of linear probing (LP) and AutoVP. 

<table><tr><td>Dataset</td><td>Zero-shot</td><td>Finetune box head</td><td>VP</td></tr><tr><td>VOC</td><td>IoU: 0.75</td><td>IoU: 0.76</td><td>IoU: 0.73</td></tr><tr><td>ISIC</td><td>IoU: 0.52</td><td>IoU: 0.70</td><td>IoU: 0.80</td></tr></table>

![](images/d6ddd122f1a3dd44d6fc3b72143520834b8336e0e9c77cbb08a7b692f78d82e2.jpg)

<details>
<summary>scatter</summary>

| Image | Ground Truth | Predicted Box |
|-------|--------------|---------------|
| (a)   | 25           | 25            |
| (a)   | 20           | 20            |
| (a)   | 15           | 15            |
| (a)   | 10           | 10            |
| (a)   | 5            | 5             |
| (a)   | 0            | 0             |
| (a)   | -5           | -5            |
| (a)   | -10          | -10           |
| (a)   | -15          | -15           |
| (a)   | -20          | -20           |
| (a)   | -25          | -25           |
| (a)   | -30          | -30           |
| (b)   | 25           | 25            |
| (b)   | 20           | 20            |
| (b)   | 15           | 15            |
| (b)   | 10           | 10            |
| (b)   | 5            | 5             |
| (b)   | 0            | 0             |
| (b)   | -5           | -5            |
| (b)   | -10          | -10           |
| (b)   | -15          | -15           |
| (b)   | -20          | -20           |
| (b)   | -25          | -25           |
| (b)   | -30          | -30           |
| Prediction | 25       | 25            |
| Prediction | 20       | 20            |
| Prediction | 15       | 15            |
| Prediction | 10       | 10            |
| Prediction | 5        | 5             |
| Prediction | 0        | 0             |
| Prediction | -5         | -5            |
| Prediction | -10        | -10           |
| Prediction | -15        | -15           |
| Prediction | -20        | -20           |
| Prediction | -25        | -25           |
| Prediction | -30        | -30           |
</details>

# E.3 EXPLORING ADDITIONAL TUNING AXES

In AutoVP, we set the learning rate (LR) for CLIP to be 40 and employed the SGD optimizer with a momentum of 0.9 and a weight decay (WD) of 0. In this section, we have undertaken supplementary tuning options within the AutoVP framework. Specifically, we introduce additional choices for the LR, selecting from 35, 40, and 45 (for CLIP), as well as WD, choosing from 0, $10^{-5}$ , and $10^{-10}$ . The outcomes presented in Table E.3 showcase a $0.6\%$ enhancement in accuracy with the additional tuning options. It is important to note that the execution workload dramatically increases, as it will involve exploring 9 times more additional combinations in the tuning process.

Table E.3: Hyper-Parameter Tuning for Learning Rate (LR) and Weight Decay (WD). The table displays the optimal tuning configurations and corresponding testing accuracy for both scenarios with and without additional LR and WD tuning options for AutoVP on Flowers102. The pre-trained model used is CLIP. The highest accuracy is marked in bold. 

<table><tr><td>Setting</td><td>Tuning Selection</td><td>Testing Accuracy (%)</td></tr><tr><td>AutoVPw/o LR &amp; WD tuning</td><td>FullyMap,  $p = 16$ LR = 40WD = 0</td><td>90.4</td></tr><tr><td>AutoVPw/ LR &amp; WD tuning</td><td>FullyMap,  $p = 16$ LR = 45WD =  $10^{-10}$ </td><td>91.0</td></tr></table>

# E.4 IMPROVED ILM-VP WITH TUNING CONFIGURATION

In order to establish an equitable comparison with AutoVP, we undertake a comprehensive hyperparameter tuning process for ILM-VP. We investigate tuning options that span 1, 2, 5, and 10 source classes for output mapping, as well as prompt sizes of 10, 20, 30, 40, and 50. For ILM-VP without tuning, default settings are maintained, with the mapping number set to 1 and the prompt size set to 30.

The tuning procedure is applied to several datasets, and the results are shown in Table E.4. Although hyper-parameter tuning in ILM-VP leads to an improvement in accuracy, it remains unable to surpass the performance exhibited by AutoVP.

Table E.4: ILM-VP with Hyper-Parameter Tuning. The chosen tuning configurations and the corresponding testing accuracy (%) in ILM-VP. The highest accuracy is marked in bold. 

<table><tr><td>Datasets</td><td>AutoVP</td><td>ILM-VPw/ Tuning</td><td>ILM-VPw/o Tuning</td></tr><tr><td>UCF101</td><td>FullyMap,  $p = 16$ Accuracy: 73.1%</td><td>mapping number = 5,  $p = 30$ Accuracy: 70.4%</td><td>Accuracy: 68.4%</td></tr><tr><td>EuroSAT</td><td>FullyMap,  $p = 16$ Accuracy: 96.8%</td><td>mapping number = 2,  $p = 30$ Accuracy: 96.2%</td><td>Accuracy: 96.7%</td></tr><tr><td>OxfordIIITPet</td><td>FreqMap-10,  $p = 16$ Accuracy: 88.2%</td><td>mapping number = 5,  $p = 30$ Accuracy: 86.7%</td><td>Accuracy: 85.1%</td></tr></table>

# E.5 COMPARISON OF AUTOVP AND BLACKVIP

Table E.5: AutoVP vs. BlackVIP. The testing accuracy (%) of AutoVP and BlackVIP. 

<table><tr><td></td><td>Pets</td><td>Cars</td><td>Flowers</td><td>Food</td><td>SUN</td><td>DTD</td><td>SVHN</td><td>EuroSAT</td><td>RESISC</td><td>CLEVR</td><td>UCF</td><td>Avg.</td></tr><tr><td>AutoVP</td><td>88.2</td><td>61.8</td><td>90.4</td><td>82.3</td><td>65.4</td><td>62.5</td><td>92.9</td><td>96.8</td><td>88.5</td><td>82.8</td><td>73.1</td><td>80.4</td></tr><tr><td>BlackVIP</td><td>89.7</td><td>65.6</td><td>70.6</td><td>86.6</td><td>64.7</td><td>45.2</td><td>44.3</td><td>73.1</td><td>64.5</td><td>36.8</td><td>69.1</td><td>64.6</td></tr></table>

Some visual prompting research has delved into a black-box setting (Tsai et al., 2020; Oh et al., 2023), where the internal architecture of the pre-trained model remains unattainable during the training process. For instance, BlackVIP (Oh et al., 2023) employs an input-dependent prompt designer to generate visual prompts. These prompts are then fed into a black-box model, and gradient approximation strategies are utilized to update the prompt designer. In Table E.5, we present a comparison between AutoVP and BlackVIP. Since BlackVIP uses CLIP as the pre-trained backbone, we report our results using the same pre-trained model. AutoVP demonstrates a 16% performance increase on average compared to BlackVIP. This notable difference might be attributed to the variance in update strategies: BlackVIP utilizes SPSA-GC for black-box models, while AutoVP relies on classic gradient descent. However, due to the differing objectives of these two studies, direct comparisons may introduce certain unfairness.

# F PERFORMANCE AND RESOURCE UTILIZATION

# F.1 COMPARISON OF AUTOVP, LINEAR PROBING, AND FULL FINE-TUNING

When comparing the performance of AutoVP to that of linear probing (LP) and full fine-tuning (FF), it's crucial to acknowledge the significant discrepancy in terms of trainable parameter size. As FF involves a parameter size that is roughly 100 to 1000 times larger (as shown in Table F.4). Nevertheless, it's worth noting that the differences in performance between VP and FF are relatively minor in certain datasets. For instance, in the case of EuroSAT, both AutoVP and FF achieve a $96\%$ accuracy (as demonstrated in Table F.1). This observation suggests that VP retains its advantages even when faced with such substantial variations in parameter size.

Table F.1: Comparison of AutoVP, Linear Probing (LP), and Full Fine-Tuning (FF). With the EuroSAT dataset and using CLIP as the pre-trained model, the table displays the testing accuracy, execution time, and trainable parameter size associated with each respective method. 

<table><tr><td>Experimental Info.</td><td>AutoVP</td><td>LP</td><td>FF</td></tr><tr><td>Accuracy (%)</td><td>96.84</td><td>94.62</td><td>96.78</td></tr><tr><td>Execution Time (second)</td><td>2448</td><td>2370</td><td>3081</td></tr><tr><td>Trainable Parameter Size (Million)</td><td>0.15</td><td>0.005</td><td>151.28</td></tr></table>

# F.2 COMPUTING RESOURCES

In this section, we provide the execution time (measured on NVIDIA GeForce RTX 3090) and the comparison of trainable parameters of AutoVP. For ease of comparison, we use the Flower102 dataset for illustration. In Table F.2, we provide the end-to-end execution time (hyper-parameter tuning + 200-epoch training) of AutoVP. When we measure only the 200-epoch training, AutoVP demonstrates its competitiveness in terms of similar or even lower training time (see Table F.3), compared to the state-of-the-art VP baselines (ILM-VP and CLIP-VP The comparison of trainable parameters can also be found in Table F.4.

Table F.2: Flowers102 End-to-End Execution Time. For the Flowers102 dataset, the pre-trained model selected by AutoVP is Swin-T, the output mapping is FullyMap, and the prompt size is 16. 

<table><tr><td></td><td>Hyper-Parameter Tuning</td><td>200-Epoch Training</td><td>Total Execution Time</td></tr><tr><td>Time</td><td>146 mins</td><td>31 mins</td><td>177 mins</td></tr></table>

Table F.3: Execution Time Comparison. The 200-epoch training time (in minutes) on the Flowers102 dataset varies depending on the chosen pre-trained model: Swin-T or CLIP. In both cases, AutoVP utilizes the FullyMap as the output mapping method with the prompt size 16. 

<table><tr><td>Pre-trained model</td><td>AutoVP</td><td>ILM-VP</td><td>CLIP-VP</td><td>Linear Probing</td></tr><tr><td>Swin-T</td><td>31</td><td>43</td><td>—</td><td>28</td></tr><tr><td>CLIP</td><td>39</td><td>76</td><td>38</td><td>40</td></tr></table>

Table F.4: Trainable Parameter Size. The average trainable parameter sizes (million) are calculated across the 12 datasets for different pre-trained models, mapping methods, and baselines. 

<table><tr><td rowspan="2"></td><td colspan="4">AutoVP</td><td rowspan="2">Linear Probing</td><td rowspan="2">Full Finetune</td></tr><tr><td>SemanticMap</td><td>FreqMap</td><td>IterMap</td><td>FullyMap</td></tr><tr><td>ResNet18</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.20</td><td>0.03</td><td>11.20</td></tr><tr><td>ResNeXt101-IG</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.20</td><td>0.11</td><td>86.85</td></tr><tr><td>Swin-T</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.20</td><td>0.04</td><td>27.56</td></tr><tr><td>CLIP</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.49</td><td>0.03</td><td>151.23</td></tr></table>