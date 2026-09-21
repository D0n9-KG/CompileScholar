# Curriculum Learning with Quality-Driven Data Selection

Biao Wu

Australian Artificial Intelligence Institute biaowu165534@gmail.com

Ling Chen

Australian Artificial Intelligence Institute Ling.Chen@uts.edu.au

# Abstract

The remarkable multimodal capabilities demonstrated by OpenAI's GPT-4 have sparked significant interest in the development of Multimodal Large Language Models (MLLMs). Visual instruction tuning of MLLMs using machine-generated instruction-following data has been shown to improve zero-shot capabilities on many tasks, but there has been less exploration of controlling the instruction data quality.

Current methodologies for data selection in MLLMs often rely on single, unreliable scores or use downstream tasks for selection, which is time-consuming and can lead to potential overfitting on the chosen evaluation datasets. To mitigate these limitations, we propose a novel data selection methodology that utilizes image-text correlation and model perplexity to evaluate and select data of varying quality.

This approach leverages the distinct distribution of these two attributes, mapping data quality into a two-dimensional space that allows for the selection of data based on their location within this distribution. By utilizing this space, we can analyze the impact of task type settings, used as prompts, on data quality. Additionally, this space can be used to construct multi-stage subsets of varying quality to facilitate curriculum learning.

This multiple training strategy not only utilizes a minimal amount of data but also maintains data quality diversity, significantly enhancing the model's fine-tuning performance. Our research includes comprehensive experiments conducted on various datasets. The results emphasize substantial enhancements in five commonly assessed capabilities compared to using the complete dataset. Our codes, data, and models are publicly available at: https://anonymous.4open.science/r/EHIT-31B4

# 1 Introduction

Instruction-following Multimodal Large Language Models (MLLMs) excel in multi-modality tasks [25, 39, 29]. Their effectiveness largely comes from using Large Language Models (LLMs) to generate synthetic data for visual instruction tuning. SELF-FILTER [8] emphasizes that visual instruction tuning is a straightforward alignment process in MLLMs training. It only needs a small amount of tuning data to activate the pre-trained capabilities and align them with the target interaction format. To improve this process, dataset selection tasks have been proposed to choose high-quality instruction-tuning data, enhancing the performance of these models [8].

Despite the central role that datasets play in training large language models, exploring data quality for instruction tuning in vision-and-language models remains challenging. Many existing data selection methods use simple rules based on the characteristics of images and texts separately, such as the length of captions, the use of nouns, the complexity of sentences, the aspect ratio of images, and the minimum size of images [3, 31, 5, 32]. These methods also consider the reliability of the data source [11]. More advanced techniques focus on the alignment between images and texts, using models like CLIP [17] to evaluate how closely the content of an image matches the accompanying text. This is done by measuring the similarity between image and text features [30, 31, 29] or by

arXiv:2407.00102v2 [cs.LG] 2 Jun 2025

Preprint. Under review.

checking if the image's main object is mentioned in the caption [32]. However, These approaches focus on high-quality data, with limited exploration of data quality diversity.

To improve the effectiveness of multimodal instruction data selection and the utilization of data diversity, we propose a new data selection method. This method constructs a representation space through two attributes of the data, which allows clear observation of the data in distributional differences for different task type settings. Meanwhile, this effectively categorizes data quality by distribution, allowing us to select different quality subsets for training. Specifically, our new method calculates each sample's clip score and model loss, using them as two-dimensional coordinates. By dividing key areas, we can obtain subsets of data with varying quality.

We introduce a new training strategy, curriculum learning. Unlike most instruction tuning tasks, our curriculum learning method involves multiple training stages, each using progressively higher-quality data. We begin by training on data randomly sampled from a high-quality sample space. In subsequent stages, we progressively refine the distribution area of this high-quality data, sampling from increasingly focused spaces. This iterative training process mimics the human learning approach, enabling the model to use data of varying quality to maintain diversity.

Through extensive experiments on LLaVA-v1.5, we demonstrate that our methods can surpass models trained on the full instruction data using only about $5\%$ of the raw instruction tuning dataset samples. This improvement is consistent across multiple evaluation datasets and benchmarks.

We summarize the main contributions of this paper:

# 2 Related Work

Multimodal Instruction Tuning Multimodal Instruction Tuning is pivotal in advancing the capabilities of models like LLaVA [26], MiniGPT-4 [41], and InstructBLIP [9], which thrive on intricately paired image-text data. This technique refines the models' performance beyond what is achievable with conventional VQA datasets [12, 15], which often provide limited, short-answer data that can impair model performance.

Recognizing this, the MiniGPT-4 [41] team curated a dataset of 3,500 image-text pairs, refined through interactions with ChatGPT, to enhance the models' ability to generate nuanced responses. Similarly, LLaVA [26] set a benchmark by creating LLaVA-Instruct-150K, a dataset generated by prompting GPT-4 with rich annotations from the COCO dataset [24], including image captions and object details, to produce detailed questions and answers.

Expanding the scope, LLaVAR [38] addressed the challenges of interpreting text-rich images by assembling over 422,000 pieces of instruction-following data through OCR technology, supplemented by an additional 16,000 high-quality entries processed by GPT-4. Furthermore, InstructBLIP [9] incorporated a diverse array of 26 public datasets, including LLaVA-Instruct-150K, to create a more comprehensive visual instruction tuning dataset. This effort, however, highlighted the prevalence of brief, perceptually focused content in existing datasets.

Meanwhile, $\mathbf{M}^3\mathrm{IT}$ [20] transformed 40 distinct datasets into a unified vision-to-text framework, utilizing ChatGPT to rephrase and enrich the context of the responses, broadening the scope of training data suitable for deep learning models. This collective endeavor to enrich multimodal datasets [18, 22] illustrates a strategic pivot towards generating a larger, more varied corpus of visual instruction data.

These datasets now cover an extensive range of tasks from basic visual recognition to complex reasoning and planning, setting a new standard for training sophisticated multimodal systems.

Data Selection Data selection is a developing field in the instruction-tuning of large language models, focused on identifying high-quality data and removing harmful information that could lead to errors [7, 4]. In this area, [7] introduced Alpagasus, a method that automates data selection by

2

assessing instruction quality via queries to ChatGPT, thereby improving training efficiency. [21] suggested using the IFD score as an indicator of data difficulty, while [4] developed Instruction Mining, which evaluates sample quality through a linear combination of various indicators. Concurrently, [23] proposed assessing data by the one-shot learning performance on specific tasks. Finally, [36] in their study
on InstructionGPT-4, apply a combination of multimodal scores and a regression model trained on predefined tasks for data selection, although their application is confined to MiniGPT-4 [41], which includes just 3,400 instructions.

Curriculum Learning Curriculum Learning has emerged as an effective strategy in machine learning, allowing models to start with simpler tasks and gradually progress to more complex ones. This method, inspired by the way humans learn, has been applied across various domains such as natural language processing and computer vision [2, 35]. In this context, [2] pioneered the concept by showing how a progressive learning schedule can improve performance in neural networks.

More recently, [35] proposed a dynamic curriculum learning approach that adjusts the difficulty of the data based on the model's performance during training. Additionally, [28] introduced an automatic curriculum learning framework that utilizes reinforcement learning to dynamically select training samples, optimizing the learning process. Lastly, [34] explored self-paced learning, a variation where the model self-assesses and chooses the appropriate learning pace, thereby aligning with curriculum learning principles to improve overall training efficacy.

# 3 Methods

# 3.1 Data selection

We define our data selection task in the context of instruction fine-tuning. Given an instruction tuning dataset $\mathcal{D} = \{\mathbf{x}_j\}_{j=1}^N$ , where each $\mathbf{x}_j = (\mathbf{x}_j^i, \mathbf{x}_j^t)$ represents a pair of input image and text, our objective is to select a subset of size $m$ from $\mathcal{D}$ . The goal is to prune $\mathcal{D}$ such that the resulting subset, $\mathcal{D}_f^m \subset \mathcal{D}$ , enables the pre-trained vision-language model $f$ to achieve optimal performance on downstream tasks $\{T_i\}_{i=1}^t$ . Here, $|\mathcal{D}_f^m| = m$ .

# 3.2 Data Curriculum

We propose to select a subset of the dataset based on 1) clip score for image-text feature similarity and 2) model loss for data perplexity. We use these two data attributes to create a representation space for all data instances. By employing this method, we select the data using the region of the representation space that exhibits higher or lower values for both attributes. The vision-language model $f$ is pre-trained. We denote its total loss as $l$ and the loss on visual instruction data $x_{i}$ as $l_{i}$ . Additionally, We denote its clip score as $s$ and the correlation on visual instruction data $x_{i}$ as $s_{i}$ . By maximizing the $l$ and $s$ , we can obtain a relatively high-quality subset of data $\mathcal{D}_f^m$ .

Intermediate Data Similarity To evaluate the similarity between an input image $x_{j}^{i}$ and text $x_{j}^{t}$ , we use the CLIP model to extract features from both. Specifically, we apply the image encoder of the CLIP, defined as $I(\cdot)$ , to obtain the feature vector from the image, and the text encoder of the CLIP, defined as $T(\cdot)$ , to derive the feature vector from the text. We then compute the dot product of both features to generate a clip score, which we define as $s_{j}$ .

$$
s _ {j} = I \left(x _ {j} ^ {i}\right) \cdot T \left(x _ {j} ^ {t}\right)
$$

We partition the data subset by identifying the upper bounds $S_{max}$ and lower bounds $S_{min}$ of the $s_j$ . Using these bounds, we select the sample data $d_j$ to obtain the corresponding subset, which we refer to as the Data of Intermediate Similarity (DIS):

$$
D I S = \left\{d _ {j} \mid S _ {\min } \leq s _ {j} \leq S _ {\max } \right\}
$$

Clip score reflects how well the image features correspond to the text features, allowing us to identify and select high-quality data where the image and text are closely related.

3

Intermediate Data Loss The loss produced by the model, which is also a measure of perplexity, reflects the difference between the target text and the model's internal preferences. A higher loss makes the learning process more challenging for the model. Following a standard LLaVA architecture, the image encoder provides latent encoded features $X_{j}$ . Concurrently, the text decoder is tasked with maximizing the conditional likelihood of the paired text $Y_{j}$ under the forward autoregressive factorization:

$$
l _ {j} = - \sum_ {t = 1} ^ {T} \log P _ {\theta} \left(Y _ {j, t} \mid Y _ {j, <   t}, X _ {j}\right)
$$

We partition the data subset by detecting the upper bounds $L_{max}$ and lower bounds $L_{min}$ of the loss. Using these bounds, we select the sample data $d_j$ to obtain the corresponding subset, which we refer to as the Data of Intermediate Loss (DIL):

$$
D I L = \{d _ {j} \mid L _ {m i n} \leq l _ {j} \leq L _ {m a x} \}
$$

Intermediate Data Quantity When each piece of data has clip score and loss, we can construct a two-dimensional representation space based on these two attributes. Therefore, we select the sample data $d_{j}$ and set both related upper bounds and lower bounds to select the high-quality subset, which we refer to as the Data of Intermediate Quantity (DIQ):

$$
D I Q = \left\{d _ {j} \mid L _ {\min } \leq l _ {j} \leq L _ {\max }, S _ {\min } \leq s _ {j} \leq S _ {\max } \right\}
$$

We propose a data curriculum framework that starts training with simpler tasks and progressively advances to more complex ones. Based on our $DIQ$ , we divide the region into unified blocks and use $\Delta L$ and $\Delta S$ , corresponding to model loss and clip score respectively. By employing data selection methods, we can control the quality of a subset of data by gradually increasing clip score thresholds and loss thresholds. Consequently, we divide the learning process into several phases $k$ , and we select the sample data in each phase with different quantities:

$$
C _ {k} = \left\{d _ {j} \mid L _ {p} \leq l _ {j}, S _ {p} \leq s _ {j} \right\}
$$

where

$$
L _ {p} = L _ {\min } + k \Delta L \text {a n d} S _ {p} = S _ {\min } + k \Delta S.
$$

As $k$ increases, the learning process can be divided into multiple phases: Initialization, Intermediate, and Advanced.

This phased approach ensures progressive learning, better generalization, reduced overfitting, and enhanced robustness. By systematically organizing and presenting data based on quality metrics, the data curriculum ensures the model develops a solid foundation before tackling more complex data, leading to improved performance on multimodal tasks.

# 4 Experiments

In this section, we first detail our settings and the chosen base models. Then we introduce the different train scenarios and evaluation benchmarks used in our experiments and the baseline methods. We show that our proposed method achieves better performance on multiple tasks using less data.

4

# 4.1 Experimental Setup

VL instruction data. We use the core set, SVIT-core-157K, as our raw data, totaling 157,712 samples. SVIT [39] extends visual instruction tuning data to present a large-scale dataset containing 4.2 million command adjustment data. These data include dialog Q&A pairs, complex inference Q&A pairs, referring Q&A pairs, and detailed descriptions. More details can be found in the Appendix.

Base models. We use the LLaVA-v1.5-7B [25] model architecture and its pre-training weights as our base models. The entire LLaVA training process is divided into two stages. For the first stage of pretraining, LLaVA-1.5-558k [26] selected from CC3M data are used, which have been converted into instruction-following data by GPT-4. For the second stage of visual instruction tuning, LLaVA-1.5-mix-665k [25] has been used.

Train setting We consider LoRA finetuning for the new instruction data. We define the state where LLaVA-1.5-mix-665k [25] has been used for instruction tuning as scenario 1, and the state where this data has not yet been used for instruction tuning as scenario 2. And, to verify the effectiveness of the data selection strategy for LLaVA model training, we mainly consider these two scenarios.

**Benc
hmarks** We assess our methods using a mix of academic-task-oriented benchmarks and new benchmarks tailored for instruction-following LMMs, covering a total of 5 benchmarks. For academic-focused benchmarks, VQA-v2 [12] and GQA [15] test the model's visual perception abilities with open-ended questions. VizWiz [13] includes 8,000 images to evaluate the model's zero-shot generalization on visual queries from visually impaired individuals. In line with Instruct-BLIP [10], we use the image subset of ScienceQA [27] with multiple-choice questions to gauge zero-shot performance in scientific question answering. TextVQA [33] involves text-rich visual question answering.

# 4.2 Scenario 1: Training from LLaVA

We use the LLaVA-v1.5-7B [25] architecture with model weights fully fine-tuned using LLaVA-1.5-mix-665k data. Subsequently, we fine-tune this model with LoRA [14] during the follow-up experiments. In training, we keep the visual encoder, projector, and LLM weights frozen, and maximize the likelihood of with trainable parameters of LoRA only. We keep the rest of the training protocol the same to allow for a fair comparison. Scenario 1, which only includes LoRA tuning, takes approximately 16 hours on an NVIDIA Tesla A100 GPU with 40GB of memory, using DeepSpeed ZeRO Stage 3. We use the SVIT-core-157K [39] dataset for continuous fine-tuning to establish a baseline. And the same method is applied to fine-tune our data.

Table 1: Comparison with SoTA methods on 5 benchmarks. We achieve better performance on all benchmarks than SVIT-Core-157K. Res, PT, and IT indicate input image resolution, and the number of samples in the pretraining and instruction tuning stage, respectively. Benchmark names are abbreviated due to space limits. VQA-v2 [12], GQA [15], VisWiz [13], ScienceQA-IMG [27], TextVQA [33]. More details can be found in the Evaluation Metrics section of the Appendix.

![](dt=2026-05-29/ht=11/78e52fc8f740c5d59406a88572df507cf29ae125e4c522a7a2cc5b600648ff9b.jpg)

<table><tr><td>Method</td><td>LLM</td><td>Res.</td><td>PT</td><td>IT</td><td>VQA v2</td><td>GQA</td><td>VisWiz</td><td>SQAI</td><td>VQAT</td></tr><tr><td>BLIP-2[19]</td><td>Vicuna-13B</td><td>224</td><td>129M</td><td>-</td><td>41.0</td><td>41</td><td>19.6</td><td>61</td><td>42.5</td></tr><tr><td>InstructBLIP[9]</td><td>Vicuna-7B</td><td>224</td><td>129M</td><td>1.2M</td><td>-</td><td>49.2</td><td>34.5</td><td>60.5</td><td>50.1</td></tr><tr><td>InstructBLIP[9]</td><td>Vicuna-13B</td><td>224</td><td>129M</td><td>1.2M</td><td>-</td><td>49.5</td><td>33.4</td><td>63.1</td><td>50.7</td></tr><tr><td>Shikra[6]</td><td>Vicuna-13B</td><td>224</td><td>600K</td><td>5.5M</td><td>77.4</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>IDEFICS-9B [16]</td><td>LLaMA-7B</td><td>224</td><td>353M</td><td>1M</td><td>50.9</td><td>38.4</td><td>35.5</td><td>-</td><td>25.9</td></tr><tr><td>IDEFICS-80B[16]</td><td>LLaMA-65B</td><td>224</td><td>353M</td><td>1M</td><td>60.0</td><td>45.2</td><td>36.0</td><td>-</td><td>30.9</td></tr><tr><td>Qwen-VL[1]</td><td>Qwen-7B</td><td>448</td><td>1.4B†</td><td>50M†</td><td>78.8</td><td>59.3</td><td>35.2</td><td>67.1</td><td>63.8</td></tr><tr><td>Qwen-VL-Chat[1]</td><td>Qwen-7B</td><td>448</td><td>1.4B†</td><td>50M†</td><td>78.2</td><td>57.5</td><td>38.9</td><td>68.2</td><td>61.5</td></tr><tr><td>LLAVA-V1.5[25]</td><td>Vicuna-7B</td><td>336</td><td>558K</td><td>665K</td><td>78.5</td><td>62.0</td><td>50.0</td><td>66.8</td><td>58.2</td></tr><tr><td>+ SVIT-Core-157K[39]</td><td>Vicuna-7B</td><td>336</td><td>558K</td><td>+157K</td><td>75.9</td><td>57.1</td><td>49.1</td><td>69.0</td><td>56.3</td></tr><tr><td>+ Ours</td><td>Vicuna-7B</td><td>336</td><td>558K</td><td>+7K</td><td>77.9</td><td>61.8</td><td>51.1</td><td>69.5</td><td>57.3</td></tr></table>

We report our main results in Table 1. Our method, using only 7000 samples of SVIT-core-157K, achieved higher performance across all benchmarks compared to the full data experiment setup.

5

Furthermore, it surpassed the base model on SQA [27] and VisWiz [13], reaching state-of-the-art (SOTA) performance. In the efficient LoRA training setup, our data exceeded SVIT-core-157K[39] by 4.7 points in GQA [15], 2.0 points in VQAV2 [12], 1.0 point in TextVQA [33], 2.0 points in VisWiz [13], and 0.5 points in SQA [27]. The improvements verify the better training effects of our data since less data amount and same model are used.

Effectiveness of DIQ In Table 2, we use the top-right corner in the left panel of Figure 7 (shown in the appendix) as the top $5\%$ of the DIQ and conducted a comparison experiment, we found that using the $5\%$ selected by DIQ resulted in better performance compared to using the top $5\%$ of DIS and DIL separately.

We realized that this improvement is due to the subset from DIQ selecting data evenly from the entire region, whereas DIS and DIL focus on regions with high levels of clip score or loss. Based on these insights, we introduced curriculum learning, utilizing multi-stage training that progresses from low-quality to high-quality data. This approach, as demonstrated in the ablation experiment in Table 2, highlights the importance of increasing the diversity of data quality for improving model performance. By employing this method, we found that using curriculum learning with the DIQ method can further enhance model performance.

Table 2: Results across different methods.

![](dt=2026-05-29/ht=11/74dccd20ba86974b45634bbcfac8238ece5f1229aef1bafba651354aa7e852fb.jpg)

<table><tr><td>Strategy</td><td colspan="3">Scenario 1</td></tr><tr><td></td><td>SQA</td><td>TextVQA</td><td>GQA</td></tr><tr><td>DIS</td><td>57.06</td><td>56.13</td><td>61.06</td></tr><tr><td>DIL</td><td>68.82</td><td>56.30</td><td>60.87</td></tr><tr><td>DIQ</td><td>69.56</td><td>56.84</td><td>61.16</td></tr><tr><td colspan="4">Result with Data Curriculum</td></tr><tr><td>Ours</td><td>69.51</td><td>57.25</td><td>61.80</td></tr></table>

To further understand the effectiveness of curriculum learning, we observe that it starts with simple examples, which have lower noise and smaller loss. This provides a smoother loss landscape, reducing gradient oscillations and instability for a more stable initial training process. As the model progresses to higher-quality data, it benefits from established initial parameters and a clear learning direction, facilitating easier optimization. By gradually increasing data quality, curriculum learning helps the model adapt and optimize progressively, leading to improved performance as shown in our results.

# 4.3 Scenario 2: Training from Vicuna + projection.

To check the quality of our selected data and ensure consistency in our experiments, we use the LLaVA-v1.5-7B [25] model architecture and its pre-training weights, only a projector. We utilize this projector, the pre-trained CLIP visual encoder ViT-L/14, and Vicuna-7b to establish the weights of LLAVA that only the alignment task has been completed. This setup helps us observe how different selected datasets activate the model's ability to engage in dialogue while avoiding interference from other instruction-tuning data on this task. The rest of the model training protocol is kept unchanged for fair comparison. We keep the training setting as same as scenario 1 and only update the LoRA weights of the LLM.

![](dt=2026-05-29/ht=11/ff624cd899798b4e1e81dd4a10c370f3fe20b01c812f1953b14611b669ef12d2.jpg)

![](dt=2026-05-29/ht=11/f8705627923b78e8d4018617f52080de7a91422b7976086113ba3dd1413c7a01.jpg)

![](image)
9c80da311adeda4375b50d0.jpg)

Effectiveness of DIQ To verify the effectiveness of DIQ, we analyze the clip score and loss for all the data. In Figure 7, we divided the data into 9 regions and selected 7,000 samples from each

6

region as corresponding data subsets using the DIQ method. The axes in Figure 1 are the loss value and the clip score value. It shows the position of the columns in each region shows the range of the corresponding dual attributes. The color of the columns also reflects the size of the corresponding value, the higher the performance the darker the color of the columns, and vice versa. Combining the performance of SQA, TextVQA, and GQA, we find that data with a higher clip score and loss show better performance on the downstream task, implying that the top-right DIQ subset contains higher quality data.

# 4.4 Exploring Different Data Selection

Effectiveness of DIS and DIL. To verify the effectiveness of DIS and DIL separately, we first verified the data selection results of individual methods, in scenario 1 of the LLaVA training program. As shown in Figure 2, both DIS and DIL, using only the top $5\%$ (around 7000 samples) of the selected data, significantly outperform the results using all the data. The model performance gradually decreases as the amount of data increases.

![](dt=2026-05-29/ht=11/badd62fb5672fc1a77c183a3a4e9539e04c14eea5a494ea53c2da1115d428d7c.jpg)

![](dt=2026-05-29/ht=11/b4d3a195597a9eada4f96787d2fd09c146c4ce93cedee3d3b1cf6ac707f2d2a7.jpg)

![](dt=2026-05-29/ht=11/20389bd70c439f3a4189e18d12abe5d4312ab336b0c695995b44b987fc5dd0bc.jpg)

As shown in Figure 3, similar to scenario 1, both DIS and DIL, using only the top $5\%$ of the selected data, significantly outperform the results using all the data. This result is consistent with the hypothesis presented in LIMA's [40] study, which demonstrates that alignment also be a straightforward process in MLLM training. In this process, the model learns the style or format of interacting with users, effectively utilizing the knowledge and capabilities it acquired during pretraining. Meanwhile, a high-quality subset of data is sufficiently informative to help the model adapt well to new user interaction styles in scenario 2, compared to the full data.

![](dt=2026-05-29/ht=11/c84f30afed70319bf554dfc8963f2e10bb7a64fa72830cf5df516f6351aaabe7.jpg)

![](dt=2026-05-29/ht=11/37c0e41d189155abddfea65633cdfb316ac7588e94ede8a18707fd7768bc1d8a.jpg)

![](dt=2026-05-29/ht=11/9465a19f3449fa88137b353a34f985c50a4681309dab3fa3f8b4a6e24cc84770.jpg)

Effectiveness of Mixed Methods. Table 3 first compares the performance differences of the top $5\%$ of DIS, DIL, and DIQ in scenarios 1 and 2. We notice that using the $5\%$ selected by DIS and DIL separately outperformed the top $5\%$ of DIQ in scenario 2. We realized that this improvement is due to the DIS and DIL subsets focusing on regions with a higher clip score or loss, where data with both high attributes predominate, resulting in an overall higher data quality compared to DIQ. Based on these insights, we explore the mixed method, We combined the top high-quality subsets obtained from different methods to create a larger, high-quality subset.

Therefore, we observed that for Scenario 1, the model performs best with only $5\%$ of data based on DIQ. Comparing different data subsets from various regions, as well as combining data from different regions, did not improve the model's performance. That indicates that scenario 1 mainly

7

benefits from smaller, high-quality data. For Scenario 2, the model performs best with $15\%$ of data based on the mix of different region data. In our comparison, we found that when increasing the data size from $5\%$ to $10\%$ with a single strategy, the performance of both DIS and DIL decreased due to a relative drop in data quality. However, when multiple top $5\%$ data subsets were combined, the model's performance improved, even at the same $10\%$ scale. This demonstrates that in scenario 2, the model relies more on the quantity of high-quality data. Consequently, when we combined the top $5\%$ subsets from all three regions, the model's performance improved further, confirming this observation.

Effectiveness of Curriculum Learning. In Table 3, first, we randomly sampled 2,400 examples from all the regions from DIQ for the first training phase, corresponding to $C_1$ . In the second phase, we narrowed the range of high-quality data and randomly sampled 2,400 examples to further fine-tune the model trained in the first phase, corresponding to $C_2$ . We repeated this process for the third phase and got $C_3$ . In total, 7000 samples of data were used, which is consistent with the data size of the DIQ approach.

After introducing curriculum learning in scenario 1, the model's performance improved further. However, even with curriculum learning, the model's performance declined as the data size increased. This indicates that in scenario 1, in addition to enhancing data quality diversity, it is also crucial to maintain a small scale. For scenario 2, the model's performance further improved when using $15\%$ of the data. This proves that both curriculum learning and the quantity of high-quality data are important to scenario 2.

Table 3: Comparison of ablation results with different data selection strategies and curriculum sizes. The underlined data is the maximum value considering only scaled high-quality data, and the bolded data is the global maximum value. All the "x%" refers to selecting top x% examples from the corresponding data.

![](dt=2026-05-29/ht=11/11f639b2573b6e34cf5e9398dc7c82f64e79bda450d7a0f489b6c2223c7c3e8a.jpg)

<table><tr><td rowspan="2">Strategy</td><td rowspan="2">Data Size</td><td colspan="4">Scenario 1</td><td colspan="4">Scenario 2</td></tr><tr><td>SQA</td><td>TextVQA</td><td>GQA</td><td>AVG</td><td>SQA</td><td>TextVQA</td><td>GQA</td><td>AVG</td></tr><tr><td colspan="10">Result with scaling the high-quality data with different subset</td></tr><tr><td>5% in DIS</td><td>7 k</td><td>57.06</td><td>56.13</td><td>61.06</td><td>58.08</td><td>61.92</td><td>11.14</td><td>4.96</td><td>25.34</td></tr><tr><td>5% in DIL</td><td>7 k</td><td>68.82</td><td>56.30</td><td>60.87</td><td>62.00</td><td>59.79</td><td>15.68</td><td>6.18</td><td>27.22</td></tr><tr><td>10% in DIS</td><td>14 k</td><td>69.31</td><td>56.30</td><td>59.93</td><td>61.18</td><td>61.18</td><td>8.88</td><td>1.03</td><td>23.03</td></tr><tr><td>10% in DIL</td><td>14 k</td><td>69.46</td><td>55.98</td><td>60.42</td><td>61.95</td><td>61.63</td><td>13.07</td><td>4.76</td><td>26.49</td></tr><tr><td>5% in DIQ</td><td>7 k</td><td>69.56</td><td>56.84</td><td>61.16</td><td>62.52</td><td>61.53</td><td>11.45</td><td>3.89</td><td>25.62</td></tr><tr><td>5% DIS + 5% DIL</td><td>14k</td><td>69.76</td><td>56.41</td><td>60.04</td><td>62.07</td><td>61.78</td><td>13.68</td><td>5.86</td><td>27.11</td></tr><tr><td>5% DIS + 5% DIQ</td><td>14k</td><td>69.96</td><td>56.32</td><td>60.72</td><td>62.33</td><td>61.87</td><td>13.19</td><td>7.16</td><td>27.41</td></tr><tr><td>5% DIL + 5% DIQ</td><td>14k</td><td>69.06</td><td>56.05</td><td>60.75</td><td>61.95</td><td>60.98</td><td>14.71</td><td>5.95</td><td>27.21</td></tr><tr><td>5% DIS + 5% DIL + 5% DIQ</td><td>21k</td><td>70.25</td><td>56.32</td><td>60.72</td><td>62.43</td><td>61.92</td><td>13.17</td><td>7.20</td><td>27.43</td></tr><tr><td cols
pan="10">Result with Curriculum Learning with different size</td></tr><tr><td>\( C_1+C_2+C_3 \) (Ours)</td><td>7 k</td><td>69.51</td><td>57.25</td><td>61.80</td><td>62.85</td><td>59.93</td><td>9.18</td><td>1.78</td><td>23.63</td></tr><tr><td>\( C_1+C_2+C_3 \)</td><td>14 k</td><td>69.56</td><td>56.99</td><td>61.73</td><td>62.76</td><td>61.53</td><td>12.93</td><td>3.70</td><td>26.05</td></tr><tr><td>\( C_1+C_2+C_3 \)</td><td>21 k</td><td>69.51</td><td>56.54</td><td>61.38</td><td>62.48</td><td>61.97</td><td>15.49</td><td>6.39</td><td>27.95</td></tr></table>

# 4.5 What Makes Selected Data Quality Different?

Visual instruction data generated via unimodal LLM exhibit different properties on the epistemic evidence space of clip score and loss by forming image-text pairs with their corresponding images. We try to understand what causes this problem with visual instruction data and how to control the distribution and quality of visual instruction data. The existing methods for generating data for visual instruction-tuning primarily use single-mode LLMs to adjust the text format in the data.

This approach can lead to inconsistencies between the images and the corresponding text content, causing mismatches or failing to accurately capture the main elements of the images. Additionally, the design of prompts often influences the visual instruction data, altering the generation process to suit different tasks. This variation in text generation methods for different tasks exacerbates the issue of data quality divergence. To better compare the distribution of data for different tasks in space, we visualize the space.

As shown in Figure 4, there are significant differences in the distributions between Detail Description data and Referring QA data. The Detail Description data are widely distributed in the upper right corner of the space, while the Referring QA data are widely distributed in the lower left corner of the

8

![](dt=2026-05-29/ht=11/d1f8fd4ad3b72841469fa32e12b21940498ecacd8c1aa123ef0d580e5e80b5cd.jpg)

![](dt=2026-05-29/ht=11/e3bf920e74c743e743e561482407c3e609749fbee7856df2c30d08974f7118ad.jpg)

space. This indicates that the task type used as a prompt can significantly influence the attributes of the data and lead to differences in quality.

Meanwhile, as shown in Figure 5, we found that the data quality distribution of Detail Description and Complex Reasoning tasks also differs significantly. In particular, data quality distribution for Complex Reasoning tasks, which are constructed through multi-turn dialogues, is spread over a wider area, highlighting the challenge of maintaining data quality in dialogue-based task types.

Additionally, various factors can influence the data's attributes and quality besides the task type. We analyzed the token lengths of data in all regions and visualized the distribution using a heatmap. As shown in Figure 6, brighter areas indicate longer text lengths, primarily on the right side, suggesting a consistent correlation between token length and data loss. This indicates that visual instruction data created using LLMs from the same representation space have a stable logical hierarchy and rich information.

Therefore, longer visual instruction data can effectively improve data quality by providing more detailed and coherent information.

![](dt=2026-05-29/ht=11/51e152849a61205b2d836cdae7972ca901ff897ab1c7a3fe5ec8836de8936986.jpg)

# 5 Conclusion

In this paper, we introduce a curriculum learning method that imitates the human learning process. By gradually improving the quality of training data from easy to difficult stages, our method enhances performance while requiring less training data. In addition, we demonstrate the effectiveness of utilizing a dual-attribute representation space in controlling the quality of multimodal training data that divides data subsets based on clip score and model loss.

We find that not only do data with higher dual-attribute values lead to better performance, but we also found a correlation between the task type used during visual instruction data creation and the distribution of positions in the dual-attribute space. At the same time, we found that different selection strategies for the subset of high-quality data are needed at different stages of training. When MLLMs have completed instruction fine-tuning tasks, incorporating curriculum learning can significantly improve fine-tuning performance.

**Limitation** Regarding the limitations, the scope of our experiments was constrained by computational costs, limiting our focus to a single model. We conducted all related experiments on LLaVA-v1.5[25]. Nevertheless, this approach allowed us to achieve significant results within our computational constraints. To address these limitations, future research could explore more powerful and advanced models as computational resources allow.

9

# References

10

11

Evaluation Metrics To better validate the activation of the model's conversational capabilities, the experiments were centered around the evaluation of open questions. The evaluation metrics mainly use accuracy for VQA tasks. We compare a model optimized through continuous instruction tuning with TA-selected-15k against leading MLLMs.: BLIP-2 [19], InstructBLIP [9], Shikra [6], IDEFICS [16], Qwen-VL(-Chat) [1], mPLUG-Owl2 [37] and LLaVA-v1.5 [25]. We evaluate these models on popular benchmarks: VQA-v2 [12], GQA [15], VisWiz [13], ScienceQA-IMG [27], TextVQA [33].

12

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: see Section 1.

Guidelines:

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: see the conclusion.

Guidelines:

# 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

13

Justification: see Section 3.

# Guidelines:

# 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: see Section 4.

# Guidelines:

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

14

Answer: [Yes]

Justification: see abstract.

Guidelines:

# 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: see Section 4.

Guidelines:

# 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: error bars are not reported because it would be too computationally expensive

Guidelines:

15

# 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: see Section 4.

Guidelines:

# 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in e
very respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes].

Justification: see Section 4.

Guidelines:

# 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: there is no societal impact of the work performed.

Guidelines:

16

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes].

Justification: see Section 4.

Guidelines:

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes].

Justification: see Section 4.

Guidelines:

# 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

17

# Answer: [NA]

Justification: the paper does not release new assets.

# Guidelines:

# 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

# Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects.

# Guidelines:

# 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

# Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects.

# Guidelines:

18