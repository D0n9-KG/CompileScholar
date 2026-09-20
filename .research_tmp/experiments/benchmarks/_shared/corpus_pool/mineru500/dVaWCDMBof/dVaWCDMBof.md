# DATACOMP: In search of the next generation of multimodal datasets

Samir Yitzhak Gadre $^{*2}$ , Gabriel Ilharco $^{*1}$ , Alex Fang $^{*1}$ , Jonathan Hayase $^{1}$ , Georgios Smyrnis $^{5}$ , Thao Nguyen $^{1}$ , Ryan Marten $^{7,9}$ , Mitchell Wortsman $^{1}$ , Dhruba Ghosh $^{1}$ , Jieyu Zhang $^{1}$ , Eyal Orgad $^{3}$ , Rahim Entezari $^{10}$ , Giannis Daras $^{5}$ , Sarah Pratt $^{1}$ , Vivek Ramanujan $^{1}$ , Yonatan Bitton $^{11}$ , Kalyani Marathe $^{1}$ , Stephen Mussmann $^{1}$ , Richard Vencu $^{6}$ , Mehdi Cherti $^{6,8}$ , Ranjay Krishna $^{1}$ , Pang Wei Koh $^{1,12}$ , Olga Saukh $^{10}$ , Alexander Ratner $^{1,13}$ , Shuran Song $^{2}$ , Hannaneh Hajishirzi $^{1,7}$ , Ali Farhadi $^{1}$ , Romain Beaumont $^{6}$ , Sewoong Oh $^{1}$ , Alex Dimakis $^{5}$ , Jenia Jitsev $^{6,8}$ , Yair Carmon $^{3}$ , Vaishaal Shankar $^{4}$ , Ludwig Schmidt $^{1,6,7}$

# Abstract

Multimodal datasets are a critical component in recent breakthroughs such as CLIP, Stable Diffusion and GPT-4, yet their design does not receive the same research attention as model architectures or training algorithms. To address this shortcoming in the machine learning ecosystem, we introduce DATACOMP, a testbed for dataset experiments centered around a new candidate pool of 12.8 billion image-text pairs from Common Crawl. Participants in our benchmark design new filtering techniques or curate new data sources and then evaluate their new dataset by running our standardized CLIP training code and testing the resulting model on 38 downstream test sets. Our benchmark consists of multiple compute scales spanning four orders of magnitude, which enables the study of scaling trends and makes the benchmark accessible to researchers with varying resources. Our baseline experiments show that the DATACOMP workflow leads to better training sets. Our best baseline, DATACOMP-1B, enables training a CLIP ViT-L/14 from scratch to $79.2\%$ zero-shot accuracy on ImageNet, outperforming OpenAI's CLIP ViT-L/14 by 3.7 percentage points while using the same training procedure and compute. We release DATACOMP and all accompanying code at www.datacomp.ai.

# 1 Introduction

Recent advances in multimodal learning such as CLIP [111], DALL-E [115, 116], Stable Diffusion [123], Flamingo [8], and GPT-4 [103] offer unprecedented generalization capabilities in zero-shot classification, image generation, and in-context learning. While these advances use different algorithmic techniques, e.g., contrastive learning, diffusion, or auto-regressive modeling, they all rest on a common foundation: large datasets containing paired image-text examples. For instance, CLIP's training set contains 400 million image-text pairs, and Stable Diffusion was trained on the two billion examples from LAION-2B [129]. This new generation of image-text datasets is 1,000 times larger than previous datasets such as ImageNet, which contains 1.2M images [37, 126].

Despite the central role of image-text datasets, little is known about them. Many state-of-the-art datasets are proprietary, and even for public datasets such as LAION-2B $[129]$ , it is unclear how design choices such as the data source or filtering techniques affect the resulting models. While there are thousands of ablation studies for algorithmic design choices (loss function, model architecture, etc.), datasets are often treated as monolithic artifacts without detailed investigation. Moreover,

Table 1: Zero-shot performance of CLIP models trained on different datasets. DATACOMP-1B, assembled with a simple filtering procedure on image-text pairs from Common Crawl, leads to a model with higher accuracy than previous results while using the same number of multiply-accumulate operations (MACs) or less during training. See Section 3.5 for details on the evaluation datasets. 

<table><tr><td>Dataset</td><td>Dataset size</td><td># samples seen</td><td>Architecture</td><td>Train compute (MACs)</td><td>ImageNet accuracy</td></tr><tr><td>OpenAI&#x27;s WIT [111]</td><td>0.4B</td><td>13B</td><td>ViT-L/14</td><td> $1.1 \times 10^{21}$ </td><td>75.5</td></tr><tr><td>LAION-400M [128, 28]</td><td>0.4B</td><td>13B</td><td>ViT-L/14</td><td> $1.1 \times 10^{21}$ </td><td>72.8</td></tr><tr><td>LAION-2B [129, 28]</td><td>2.3B</td><td>13B</td><td>ViT-L/14</td><td> $1.1 \times 10^{21}$ </td><td>73.1</td></tr><tr><td>LAION-2B [129, 28]</td><td>2.3B</td><td>34B</td><td>ViT-H/14</td><td> $6.5 \times 10^{21}$ </td><td>78.0</td></tr><tr><td>LAION-2B [129, 28]</td><td>2.3B</td><td>34B</td><td>ViT-g/14</td><td> $9.9 \times 10^{21}$ </td><td>78.5</td></tr><tr><td>DATACOMP-1B (ours)</td><td>1.4B</td><td>13B</td><td>ViT-L/14</td><td> $1.1 \times 10^{21}$ </td><td>79.2</td></tr></table>

datasets currently lack the benchmark-driven development process that has enabled a steady stream of improvements on the model side and isolates data enhancements from changes to the model. These issues impede further progress in multimodal learning, as evidenced by recent work showing that public datasets currently do not match the scaling behavior of proprietary alternatives $[28]$ .

In this paper, we take a step towards a more rigorous dataset development process. Our first and central contribution is DATACOMP, a new benchmark for multimodal dataset design. DATACOMP flips the traditional benchmarking paradigm in machine learning where the dataset is fixed and researchers propose new training algorithms. Instead, we hold the entire training code and computational budget constant so that participants innovate by proposing new training sets. To evaluate the quality of a training set, we score the resulting model with a testbed of 38 classification and retrieval tasks such as ImageNet [37], ImageNetV2 [121], DTD [30], EuroSAT [63], SUN-397 [146], and MSCOCO [26].

DATACOMP focuses on two key challenges that arise when assembling large training datasets: what data sources to train on, and how to filter a given data source. Each challenge corresponds to one track in our benchmark. To facilitate the filtering track, our second contribution is COMMONPOOL, a dataset of 12.8B image-text pairs collected from Common Crawl and currently the largest public image-text dataset. We release CommonPool as an index of image url-text pairs under a CC-BY-4.0 license, and apply content checks in its construction to remove unsafe or unwanted content. In the filtering track, the goal of participants is to find the best subset of COMMONPOOL to train on. In the second track, Bring Your Own Data (BYOD), participants may leverage any data source, as long as it does not overlap with our evaluation testbed.

Our third contribution is an investigation of scaling trends for dataset design. In particular, DATACOMP contains four scales, where we vary the training budget and the candidate pool size from 12.8M to 12.8B samples (see Table 2). Expressed in GPU hours, the cost of a single training run ranges from 4 to 40,000 GPU hours on the A100 cluster we used for development. The different scales enable researchers with different resources to participate in our benchmark. Moreover, our results show that the ranking of filtering approaches is largely consistent across scale.

Our fourth contribution is over three hundred baseline experiments, including techniques such as querying captions for relevant keywords, filtering based on image embeddings, and applying a threshold on CLIP scores. A key result from our baselines experiments is that smaller, more stringently filtered datasets can lead to models that generalize better than larger datasets coming from the same pool. At the 12.8B scale, our best filtering baseline increases ImageNet zero-shot accuracy by 6.9 percentage points (pp) relative to the unfiltered pool (see Table 3). For the BYOD track, our initial experiments show that 109M additional data points (less than $1\%$ of the 12.8B pool) improve the CLIP-filtered subsets of COMMONPOOL by up to 1.2 pp ImageNet accuracy (see Table 18).

Finally, our fifth contribution is DATACOMP-1B, a new state-of-the-art multimodal dataset. We obtain DATACOMP-1B by combining our two most promising filtering baselines. DATACOMP-1B enables training a CLIP ViT-L/14 model to an ImageNet zero-shot accuracy of 79.2% (see Table 1), corresponding to a 9× computational cost reduction when compared to a larger CLIP ViT-g/14 model trained on LAION-2B for about 3× longer. Moreover, our model outperforms OpenAI's original CLIP ViT-L/14 by 3.7 percentage points, while using the same compute budget.

To make DATACOMP a shared environment for controlled dataset experiments, we publicly release our candidate pool url index, our tooling for assembling these pools, our filtering baselines, and our

![](images/dd0cbffc6886f727dd4933de16efa5a46334518e32ac6e1f50701f2dda2b0898.jpg)  
Choose scale

![](images/944fa3ecfe500b3d7030a63745d6a6ab324571389edb1bdf3864f55f00b8869d.jpg)  
Choose scale: small, medium, large or xlarge

![](images/0dada149263d327c760ccf34eceb9b5cf46befda3a44555046f816c199bc29ff.jpg)  
Select data

![](images/207a8a5b46dd999566f5c1c4c943040011db79140e04f70f7f051ce2a683fd2b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["CommonPool"] -->|filtering track| B["Candidate dataset"]
    C["External data sources"] -->|bring your own data track| B
    B -->|subset| A
```
</details>

![](images/6af91adffb9fb122a18e2aa7f8d874a9227d8a05398ab2e26645477c0314bfa9.jpg)  
Train

![](images/3df96ea31e5e80316a13df6ef9fe4fbbbcb1d9e6a38630608fde617b95fd072c.jpg)  
Train a CLIP model with a fixed architecture and hyper-parameters

![](images/81ea8b3eaec0e89c98017997abe2c94f34bcf6de4d65eda5a7f8f94cdafa4922.jpg)  
Evaluate

![](images/78860aaea69f0deb1870dfbbd19bc20dc9b19dd3a29ec12f9590d804d0d6e4d4.jpg)  
Evaluate the model on 38 zero-shot downstream tasks   
Figure 1: DATACOMP participant workflow. A) Choose a scale based on resource constraints. B) Design a dataset, in either the filtering or BYOD track. C) Train a CLIP model on the designed dataset using a fixed architecture and hyperparameters (Section 3.4). D) Evaluate the trained model on a suite of diverse downstream tasks (Section 3.5).

code for training and evaluating models at www.datacomp.ai. We believe that our infrastructure will help put research on dataset design on rigorous empirical foundations, draw attention to this understudied research area, and lead to the next generation of multimodal datasets.

# 2 Related Work

We review the most closely related work and include additional related work in Appendix C.

The effects of data curation. Classical work considers dataset cleaning and outlier removal $[74, 152, 124, 125]$ to discard samples that may lead to undesirable model bias. A related line of work develops coreset selection algorithms $[61, 7, 46, 11, 94, 145, 32]$ , which aim to select data subsets that lead to the same performance as training on the entire dataset. These techniques appear to scale poorly to larger data regimes $[51, 6]$ . More recent efforts in subset selection often operate on already curated datasets $[98, 141, 130, 16, 33, 106]$ (e.g., CIFAR-10, ImageNet) or on smaller data regimes (e.g., YFCC-15M $[111, 140]$ ). These settings often do not reflect newer training paradigms that involve (1) noisy image-text pairs instead of category labeled images and (2) large scale datasets (e.g., billions of samples). While data-centric investigations have led to community competitions like DCBENCH $[43]$ and DATAPERF $[97]$ , existing benchmarks have likewise operated at small data scales $[100]$ compared to datasets like LAION-2B $[129]$ , which contains over two billion images. DATACOMP bridges this gap by aligning data-centric investigation with large scale image-text training.

There has also been renewed interest in dataset pruning and deduplication. Sorscher et al. $[135]$ show that data pruning can improve traditional scaling trends on ImageNet, but do not consider image-text training or larger datasets. Raffel et al. $[113]$ remove sentence redundancies when creating the C4 corpus. Subsequent work further demonstrated the benefits of deduplication for better language modeling $[90]$ . Radenovic et al. $[110]$ introduce CAT filtering for image-text datasets—a rule-based system to retain high quality samples. Abbas et al. $[6]$ propose SemDeDup, which starts with the CAT-filtered LAION-440M subset, further employing clustering to remove semantic duplicates. DATACOMP facilitates data-centric investigation at an even larger scale (i.e., 12.8B sample scale) and provides a common experimental setting for fair comparison amongst dataset creation algorithms.

Large-scale multimodal datasets. Datasets have been instrumental to building multimodal models like CLIP $[111]$ , Flamingo $[8]$ , Stable Diffusion $[123]$ , DALL-E $[115, 116]$ and GPT-4 $[103]$ . These methods succeeded by training on large, heterogeneous datasets rather than solely through advanced modelling techniques. For example, OpenAI's CLIP trains on 400M image-text pairs from the web, roughly $300\times$ the size of ImageNet $[37]$ . Prior work on scaling image-text datasets also provides promising trends with respect to zero-shot model performance $[73, 107]$ . Additional large scale datasets like FILIP-300M $[149]$ , FLD-900M $[153]$ , and PaLI-10B $[25]$ were constructed to train multimodal models. However, many datasets used to train such models (including the dataset for OpenAI's CLIP) are proprietary, making it hard to conduct data-centric investigations.

Even for public image-text datasets like SBU [104], Flickr30k [151], MS-COCO [26], TaiSu [92], Conceptual Captions [131], CC12M [24], RedCaps [38], WIT [136], Shutterstock [101], YFCC-

Table 2: Experimental configurations, with compute in multiply-accumulate operations (MACs). 

<table><tr><td>Scale</td><td>Model</td><td>Train compute (MACs)</td><td>Pool size and # samples seen</td></tr><tr><td>small</td><td>ViT-B/32</td><td> $9.5 \times 10^{16}$ </td><td>12.8M</td></tr><tr><td>medium</td><td>ViT-B/32</td><td> $9.5 \times 10^{17}$ </td><td>128M</td></tr><tr><td>large</td><td>ViT-B/16</td><td> $2.6 \times 10^{19}$ </td><td>1.28B</td></tr><tr><td>xlarge</td><td>ViT-L/14</td><td> $1.1 \times 10^{21}$ </td><td>12.8B</td></tr></table>

100M [140], COYO-700M [20], LAION-400M [128], or LAION-2B [129] little is known about what constitutes a good image-text dataset. Preliminary analysis suggests that different image-text data sources lead to CLIP models with different properties [101]. However, previous work is limited to smaller scale data (10-15M examples). Birhane et al. [15] examine LAION-400M and find NSFW imagery and racial slurs, centering the dangers in web-scale multimodal datasets. To combat toxicity, we preprocess our pool to remove NSFW content and blur human faces detected in images. For more details on our safety preprocessing see Section 3.2, Appendices E and G.

# 3 The DATACOMP benchmark

DATACOMP is meant to facilitate data-centric experimentation. While traditional benchmarks emphasize model design, DATACOMP is centered around dataset development, where the resulting datasets can be used to train high accuracy models. We focus on large image-text datasets and quantify a dataset submission by training a CLIP model on it from scratch $[111]$ and evaluating on 38 downstream image classification and retrieval tasks. We additionally have three secret test sets, which will be released after a year, to guard against overfitting. To facilitate such investigations, we provide a candidate pool of uncurated image-text pairs sourced from the public internet. Our benchmark offers two tracks: one where participants must filter samples from the pools we provide, and another where participants can use external data. Moreover, DATACOMP is structured to accommodate participants with diverse levels of computational resources: each track is broken down into four scales with varying compute requirements. We now discuss high-level design decisions, construction of a 12.8B image-text data pool to facilitate the competition, benchmark tracks, model training, and evaluation.

# 3.1 Competition design

Overview. In many areas of machine learning, larger datasets lead to better performing models $[87, 79, 73, 107, 66, 28, 19, 111, 112]$ . Hence comparing only datasets with the same size is a natural starting point. However, this approach is flawed as controlling the dataset size ignores critical curation constraints: candidate pool size (i.e., number of image-text pairs to harvest) and training compute. For instance, assembling a dataset like LAION-2B consists of identifying data sources (e.g., Common Crawl or Reddit) and filtering the data source. Notably, the final dataset size is a design choice and is only upper-bounded by the data sources. Hence, the true data constraint is the size of the reservoir of samples: candidate pool to be filtered. To make DATACOMP a realistic benchmark, we therefore fix the candidate pool in the filtering track, but give participants control over the training set size.

Compute cost is another relevant constraint. To put datasets of different size on equal footing, we specify the total number of training samples seen. Consider the 12.8B compute scale and filtered datasets A and B, with 6.4B and 3.2B image-text pairs respectively. At this scale, we train by making two passes over A, while making four passes over B. A key result from our experiments is that smaller, more stringently filtered datasets can lead to models that generalize better.

Competition tracks. Two key procedures in assembling a training dataset are filtering a data source $[128, 129, 20]$ and aggregating data sources $[36, 37]$ . To reflect this structure, DATACOMP has two tracks: filtering, where participants select a subset of the samples from COMMONPOOL, and Bring Your Own Data (BYOD), where participants can use any source of data. Key decisions for each track are described in Sections 3.2 and 3.3, respectively. For full competition track rules see Appendix A.

Competition compute scales. To facilitate study of scaling trends and accommodate participants with various computational resources, we structure DATACOMP using four scales of compute: small, medium, large and xlarge. Each new scale increases the number of samples seen during training by

10× (from 12.8M to 12.8B samples seen), and the pool we provide by the same factor (from 12.8M samples to 12.8B samples). Table 2 gives the experimental configuration used for each scale. For the small scale, our runs took 4 hours on an A100 GPU, and for the xlarge scale 81 hours on 512 GPUs.

# 3.2 COMMONPOOL generation, for the filtering track

We construct a large-scale pool of image-text pairs, COMMONPOOL, from Common Crawl $[3]$ . CommonPool is distributed as an image url-text pair index under a CC-BY-4.0 license. Our pool construction pipeline has four steps: url extraction and data download, NSFW detection, evaluation set deduplication, and face blurring. We additionally provide per sample metadata (e.g., CLIP features). Starting from the xlarge COMMONPOOL, we take successive random subsets to create large, medium, and small COMMONPOOL (e.g., medium is a subset of large).

Extracting urls and downloading data. We first use cc2dataset $[1]$ , which utilizes Apache Spark $[155]$ , to extract pairs of image urls and nonempty alt-text from all Common Crawl snapshots from 2014 to 2022. We then deduplicate the url-text pairs and randomly shuffle. This step results in $\sim88B$ possible samples. Not all samples are downloadable; other samples are not suitable due to NSFW content or overlap with our evaluation sets. We attempt to download $\sim40B$ samples using img2dataset $[5]$ resulting in $\sim16.8B$ image-text pairs. For more details, see Appendix D.

Safety preprocessing. Since Common Crawl is a snapshot of the internet, we require strict preprocessing to remove unsafe content. We use Detoxify $[60]$ to prune samples that contain unsafe text (e.g., obscene, sexually explicit, or threatening language). We also discard samples with explicit visual content. To do so, we train a classifier on CLIP ViT-L/14 $[111]$ features, using the NSFW dataset used in LAION-5B $[129]$ . We validate our classifier against the Google commercial image safety API. See Appendix E for details. Around 19% of image-text pairs are considered NSFW, taking the pool of $\sim$ 16.8B downloads to $\sim$ 13.6B samples.

Evaluation set deduplication. To prevent accidental overfitting to certain test sets in our evaluation suite, we perform a thorough near-duplicate removal between the candidate pool and our evaluation sets, using a state-of-the-art image deduplication model $[150]$ . Appendix F contains additional details. The model flags $\sim3\%$ of the 16.8B images as near-duplicates, reducing the $\sim13.6B$ pool to $\sim13.1B$ samples. From here we select a random subset to get the xlarge pool of 12.8B samples.

Face detection & blurring. To protect the privacy of individuals, we detect and blur faces from images in our pool using a face detector $[53]$ . As observed by Yang et al. $[148]$ , obfuscating faces has little impact on model performance, as we also observe in our experiments (Appendix G).

Pool metadata. To bootstrap participants we distribute metadata for each sample in COMMONPOOL (e.g., image url, alt-text, original image resolution, CLIP features, and CLIP similarity scores). Following Carlini et al. [22], we release SHA256 hashes for each image to guard against data poisoning in subsequent COMMONPOOL downloads. For additional details see Appendix H. We open-source our metadata processing pipeline as dataset2metadata [4].

# 3.3 The bring your own data (BYOD) track

While COMMONPOOL can be used to study different filtering techniques, state-of-the-art models often train on data from different sources. For instance, the Flamingo model $[8]$ uses both multimodal massive web (M3W) and ALIGN datasets $[73]$ . To facilitate non-proprietary research on curating data from many sources, we instantiate a separate DATACOMP track to allow participants to combine multiple data streams. For example, participants could construct a training set from CC12M $[24]$ , YFCC100M $[140]$ , and data sources they label themselves. In Section 4.2 and Appendix P.2 we describe our exploration using existing public, image-text datasets. These datasets are acquired from their respective sources and are not re-release as part of DATACOMP.

# 3.4 Training

We create a common experimental setting that enables comparable experiments by fixing the training procedure. We closely follow the CLIP training recipe proposed by Radford et al. [111]: training

models from scratch with a contrastive objective over images and captions. Given a set of image-caption pairs, we train an image encoder and a text encoder such that the similarity between the representations of images and their corresponding text is maximized relative to unaligned pairs. $^{1}$ For each scale, we fix the model architecture and hyperparameters (see Table 2). We pick Vision Transformers (ViTs) [39] as the image encoder, considering the better scaling trends observed by Radford et al. [111] compared to ResNets [62]. Models are trained for a fixed number of steps determined by the scale (Table 2), using the OpenCLIP repository [69]. See Appendix N for details.

# 3.5 Evaluation

We evaluate on a suite of 38 image classification and retrieval tasks. We also study two additional fairness tasks, detailed in Section 5 and Appendix Q. As discussed in Section 3.2, we remove test set images from DATACOMP to avoid contamination. Image classification datasets range from satellite imagery recognition to classifying metastatic tissues. In total we have (with some overlap): 22 of the datasets evaluated in Radford et al. [111], 6 ImageNet distribution shifts (i.e., ImageNet-Sketch [143], ImageNet-V2 [121], ImageNet-A [65], ImageNet-O [65], ImageNet-R [64], and ObjectNet [13]), 13 datasets from VTAB [156], and 3 datasets from WILDS [83, 127]. Retrieval datasets include Flickr30k [151], MSCOCO [26], and the WinoGAViL commonsense association task [17]. To aggregate results over all evaluation tasks, we average the preferred metric for each task.

DATACOMP adopts a zero-shot evaluation protocol: models are tested without training on the evaluation tasks. This approach is computationally efficient and measures a model's ability to perform well without any additional training. We find a strong rank correlation (>0.99) between performance in linear probe zero-shot settings (Appendix Figure 16). Additional details are in Appendix O.

# 4 Baselines

# 4.1 Filtering baselines

We study six simple filtering methods for the filtering track; see Appendix P.1 for further details.

No filtering. We simply use the entire pool as the subset, without any filtering. Since each pool size is equal to the sample budget, training consists of one pass over the data.

Random subsets. To isolate the effects of increasing the compute budget from increasing the dataset size, we form subsets consisting of 1%, 10%, 25%, 50% and 75% of the pool chosen at random.

Basic filtering. We consider many simple filtering operations inspired by Schuhmann et al. [128] and Byeon et al. [20]: filtering by language (English captions, using either fasttext [77] or cld3 [2]); filtering by caption length (over two words and five characters); and filtering by image size (smaller dimension above 200 pixels and aspect ratio below three). We also experiment with combining language and caption length filtering and combining language, caption length, image size filtering. Unless otherwise specified, “basic” refers fasttext English, caption length, and image size filtering.

CLIP score and LAION filtering. We experiment with CLIP score filtering (also employed by LAION), where we take only examples having cosine similarity scores between CLIP image and text embeddings that exceed a pre-defined threshold. We investigate a range of thresholds and two OpenAI CLIP models for computing the scores: the ViT-B/32 model (as in LAION) and the larger ViT-L/14. We also combine CLIP score thresholds and cld3 English filtering to reproduce the LAION-2B filtering scheme. Table 16 in Appendix P.1 summarizes the different CLIP score configurations.

Text-based filtering. We select examples that contain text overlapping with ImageNet class names, which serve as a proxy for relevance to downstream tasks. Specifically, we select English captions (according to fasttext) that contain words from ImageNet-21K or ImageNet-1K [37] class synsets.

Table 3: Zero-shot performance for select baselines in the filtering track. On all scales, filtering strategies lead to better performance than using the entire, unfiltered pool. The intersection between imaged-based and CLIP score strategies performs well on most tasks and scales. For all metrics, higher is better (see Appendix O for details). $\cap$ denotes the intersection of filtering strategies. 

<table><tr><td>Scale</td><td>Filtering strategy</td><td>Dataset size</td><td>Samples seen</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td rowspan="7">small</td><td>No filtering</td><td>12.8M</td><td>12.8M</td><td>0.025</td><td>0.033</td><td>0.145</td><td>0.114</td><td>0.132</td></tr><tr><td>Basic filtering</td><td>3M</td><td>12.8M</td><td>0.038</td><td>0.043</td><td>0.150</td><td>0.118</td><td>0.142</td></tr><tr><td>Text-based</td><td>3.2M</td><td>12.8M</td><td>0.046</td><td>0.052</td><td>0.169</td><td> $\underline{0.125}$ </td><td>0.157</td></tr><tr><td>Image-based</td><td>3M</td><td>12.8M</td><td>0.043</td><td>0.047</td><td>0.178</td><td>0.121</td><td>0.159</td></tr><tr><td>LAION-2B filtering</td><td>1.3M</td><td>12.8M</td><td>0.031</td><td>0.040</td><td>0.136</td><td>0.092</td><td>0.133</td></tr><tr><td>CLIP score (L/14 30%)</td><td>3.8M</td><td>12.8M</td><td> $\underline{0.051}$ </td><td>0.055</td><td>0.190</td><td>0.119</td><td> $\underline{0.173}$ </td></tr><tr><td>Image-based  $\cap$  CLIP score (L/14 30%)</td><td>1.4M</td><td>12.8M</td><td>0.039</td><td>0.045</td><td>0.162</td><td>0.094</td><td>0.144</td></tr><tr><td rowspan="7">medium</td><td>No filtering</td><td>128M</td><td>128M</td><td>0.176</td><td>0.152</td><td>0.259</td><td>0.219</td><td>0.258</td></tr><tr><td>Basic filtering</td><td>30M</td><td>128M</td><td>0.226</td><td>0.193</td><td>0.284</td><td>0.251</td><td>0.285</td></tr><tr><td>Text-based</td><td>31M</td><td>128M</td><td>0.255</td><td>0.215</td><td>0.328</td><td>0.249</td><td>0.307</td></tr><tr><td>Image-based</td><td>29M</td><td>128M</td><td>0.268</td><td>0.213</td><td>0.319</td><td> $\underline{0.256}$ </td><td>0.312</td></tr><tr><td>LAION-2B filtering</td><td>13M</td><td>128M</td><td>0.230</td><td>0.198</td><td>0.307</td><td>0.233</td><td>0.292</td></tr><tr><td>CLIP score (L/14 30%)</td><td>38M</td><td>128M</td><td>0.273</td><td>0.230</td><td>0.338</td><td>0.251</td><td> $\underline{0.328}$ </td></tr><tr><td>Image-based  $\cap$  CLIP score (L/14 30%)</td><td>14M</td><td>128M</td><td> $\underline{0.297}$ </td><td> $\underline{0.239}$ </td><td>0.346</td><td>0.231</td><td> $\underline{0.328}$ </td></tr><tr><td rowspan="7">large</td><td>No filtering</td><td>1.28B</td><td>1.28B</td><td>0.459</td><td>0.378</td><td>0.426</td><td>0.419</td><td>0.437</td></tr><tr><td>Basic filtering</td><td>298M</td><td>1.28B</td><td>0.516</td><td>0.423</td><td>0.446</td><td>0.480</td><td>0.458</td></tr><tr><td>Text-based</td><td>317M</td><td>1.28B</td><td>0.561</td><td>0.465</td><td>0.465</td><td>0.352</td><td>0.466</td></tr><tr><td>Image-based</td><td>293M</td><td>1.28B</td><td>0.572</td><td>0.454</td><td>0.483</td><td>0.479</td><td>0.476</td></tr><tr><td>LAION-2B filtering</td><td>130M</td><td>1.28B</td><td>0.553</td><td>0.453</td><td>0.510</td><td>0.495</td><td>0.501</td></tr><tr><td>CLIP score (L/14 30%)</td><td>384M</td><td>1.28B</td><td>0.578</td><td>0.474</td><td>0.538</td><td>0.466</td><td>0.529</td></tr><tr><td>Image-based  $\cap$  CLIP score (L/14 30%)</td><td>140M</td><td>1.28B</td><td> $\underline{0.631}$ </td><td> $\underline{0.508}$ </td><td>0.546</td><td> $\underline{0.498}$ </td><td> $\underline{0.537}$ </td></tr><tr><td rowspan="4">xlarge</td><td>No filtering</td><td>12.8B</td><td>12.8B</td><td>0.723</td><td>0.612</td><td>0.611</td><td>0.569</td><td>0.621</td></tr><tr><td>LAION-2B filtering</td><td>1.3B</td><td>12.8B</td><td>0.755</td><td>0.637</td><td>0.624</td><td> $\underline{0.620}$ </td><td>0.636</td></tr><tr><td>CLIP score (L/14 30%)</td><td>3.8B</td><td>12.8B</td><td>0.764</td><td>0.655</td><td>0.643</td><td>0.588</td><td>0.650</td></tr><tr><td>Image-based  $\cap$  CLIP score (L/14 30%)</td><td>1.4B</td><td>12.8B</td><td> $\underline{0.792}$ </td><td> $\underline{0.679}$ </td><td>0.652</td><td>0.608</td><td> $\underline{0.663}$ </td></tr></table>

Image-based filtering. We select a subset of examples whose visual content overlaps with ImageNet classes. After applying English language (fasttext) and caption length filtering, we cluster the image embeddings extracted by the OpenAI ViT-L/14 model for each image into 100K groups using Faiss [75]. We then find the nearest neighbor group for every ImageNet training example, and keep examples belonging to these groups. We apply this procedure using either ImageNet-21K (14M images) or ImageNet-1K (1.2M images), forming two subsets.

# 4.2 BYOD baselines

We experiment with multiple external data sources, including four moderately sized datasets (10 to 58M samples) studied by Nguyen et al. [101]—CC12M [24], YFCC15M [140, 111], RedCaps [38] and Shutterstock [101]—and the larger LAION-2B [129]. Additional experiments, along with more details about the data sources are provided in Appendix P.2. We consider these data sources as they are and do not perform additional preprocessing. We also present experiments combining some of the data sources (using only the external datasets, or in addition to data from our pool).

# 5 Results and discussion

# 5.1 Building better datasets

Main results. Our key results are in Table 3. Most notably, the intersection between image-based filtering and CLIP score filtering excels on most tasks. The exception is at the small scale and for retrieval datasets. $^{2}$ Furthermore, other filtering strategies like basic, CLIP score, image-based, text-based filtering show better downstream performance when compared to no filtering. A much larger suite of experiment results can be found in Appendix R.

DATACOMP leads to better image-text datasets. We hope DATACOMP catalyzes the search for the next generation of multimodal datasets. We contribute DATACOMP-1B, which is the output of the Image-based $\cap$ CLIP score (L/14 30%) baseline filter at the xlarge scale of the filtering track.

![](images/fdc23b737bbea4b28c005e4de13c78c100868c5fc0d25dfcdc54217cbda31f8a.jpg)

<details>
<summary>line</summary>

| Fraction of the pool used for training | ImageNet accuracy (Orange Solid) | ImageNet accuracy (Orange Dash) | ImageNet accuracy (Green Solid) | ImageNet accuracy (Green Dash) | ImageNet accuracy (Blue Solid) | ImageNet accuracy (Blue Dash) |
| -------------------------------------- | --------------------------------- | -------------------------------- | ------------------------------- | ------------------------------ | ------------------------------ | ----------------------------- |
| 0.0                                    | 0.1                               | 0.1                              | 0.0                             | 0.0                            | 0.0                            | 0.0                           |
| 0.25                                   | 0.6                               | 0.4                              | 0.3                             | 0.2                            | 0.05                           | 0.05                          |
| 0.5                                    | 0.55                              | 0.45                             | 0.25                            | 0.15                           | 0.05                           | 0.05                          |
| 1.0                                    | 0.45                              | 0.45                             | 0.18                            | 0.18                           | 0.03                           | 0.03                          |
</details>

![](images/02211611c24e69bf956954da4eb52bafc0bcb68bd208f518064ab9d9f4d6af00.jpg)

<details>
<summary>line</summary>

| Fraction of the pool used for training | Average performance (Line 1) | Average performance (Line 2) | Average performance (Line 3) | Average performance (Line 4) | Average performance (Line 5) |
| --------------------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| 0.0                                     | 0.1                           | 0.2                           | 0.15                          | 0.1                           | 0.05                          |
| 0.5                                     | 0.5                           | 0.4                           | 0.3                           | 0.2                           | 0.1                           |
| 1.0                                     | 0.45                          | 0.45                          | 0.25                          | 0.2                           | 0.1                           |
</details>

- small scale
- medium scale
- large scale
  - CLIP score (L/14)
  - CLIP score (B/32)
  - Rand. subset

Figure 2: Performance of random subsets (dotted line) and CLIP score filtering (solid line) when varying the subset size. When taking random subsets, larger subsets are always better. For CLIP score filtering, subsets with intermediate size perform best.

Our dataset is comprised of 1.4B samples, which not only is smaller than the LAION-2B dataset with 2.3B samples, but also comes from a smaller pool. Nevertheless, a CLIP L/14 trained on DATACOMP-1B outperforms the LAION-2B competitor by 6.1 percentage points on ImageNet (see Table 1). Moreover, training on DATACOMP-1B improves ImageNet accuracy by 3.7 percentage points over OpenAI's ViT-L/14 trained with the same compute budget. Additionally, even if we restrict ourselves to 400M samples, we can still find a subset of DATACOMP-1B that outperforms OpenAI's ViT-L/14, as seen in Table 24. These results demonstrate the impact that DATACOMP can make and provide a foundation upon which participants can build.

External data sources can improve performance. Appendix P.2 Table 18 shows results for several baselines in the BYOD track. We find several instances where adding external data sources improves performance over using just data from COMMONPOOL. For example, at the large scale, combining CLIP-filtered data from COMMONPOOL with external data from CC12M [24], YFCC15M [140, 111], RedCaps [38] and Shutterstock [101] boosts ImageNet accuracy by 4.3 percentage points. See Appendix P.2 for more experiments and details.

Trade-off between data diversity and repetition. In Figure 2, we see that randomly selecting subsets of the pool has little effect and degrades performance substantially when only small fractions are used. When filtering with CLIP scores, the optimal training set comes from selecting $\sim 30\%$ of the pool with the highest scores. The difference in performance trends between random subsets and CLIP score filtering highlights the importance of filtering strategies for selecting samples.

# 5.2 DATACOMP design analyses

COMMONPOOL and LAION are comparable with the same filtering. To validate our pool construction, we show that we can build datasets comparable to LAION-2B by employing their filtering technique on our pool. LAION-2B selects all samples where the caption is in English and the cosine similarity score from a trained ViT-B/32 CLIP model is above 0.28. We compare this filtering approach on our pool using the same number samples, 130M samples at the large scale. We find that the different data sources perform comparably: 55.3% vs 55.7% accuracy on ImageNet, and 0.501 vs 0.489 average performance over our evaluation sets using our pool and LAION-2B, respectively.

Consistency across scales. We find that the ranking between filtering strategies is typically consistent across different scales. This is illustrated in Figure 3, which shows that the baselines at small and medium scales are positively correlated. Moreover, as shown in Appendix Table 22, the rank correlations of performance is high, between 0.71 and 0.90 for different scale pairs.

Consistency across training changes. DATACOMP fixes the training procedure, so a natural question is whether better datasets from DATACOMP are better outside of DATACOMP. While DATACOMP-1B is trained at the xlarge scale, we show in Appendix Table 23 that even when substituting the ViT-L/14

![](images/b4bc42a8256582394c3e09cefa1bb51d13f9614d5d525a6aa97afab87fc7501c.jpg)

<details>
<summary>scatter</summary>

| ImageNet acc. (small) | ImageNet acc. (medium) |
| --------------------- | ---------------------- |
| 0.00                  | 0.02                   |
| 0.01                  | 0.10                   |
| 0.02                  | 0.18                   |
| 0.03                  | 0.20                   |
| 0.04                  | 0.25                   |
| 0.05                  | 0.28                   |
| 0.06                  | 0.29                   |
</details>

![](images/d3790f65d90cd6bd7b15f879e958760a49affbdf011694bf9632660283d1bec4.jpg)

<details>
<summary>scatter</summary>

| Avg. performance (small) | Avg. performance (medium) |
| ----------------------- | ------------------------- |
| 0.075                   | 0.10                      |
| 0.075                   | 0.12                      |
| 0.075                   | 0.14                      |
| 0.075                   | 0.16                      |
| 0.075                   | 0.18                      |
| 0.075                   | 0.20                      |
| 0.100                   | 0.16                      |
| 0.100                   | 0.18                      |
| 0.100                   | 0.20                      |
| 0.125                   | 0.24                      |
| 0.125                   | 0.26                      |
| 0.125                   | 0.28                      |
| 0.125                   | 0.30                      |
| 0.150                   | 0.28                      |
| 0.150                   | 0.30                      |
| 0.150                   | 0.32                      |
| 0.175                   | 0.34                      |
| 0.175                   | 0.36                      |
</details>

![](images/6c1590ff00bb78c2f2ca48e0b178a58a0edd086a94d26c2a5e248c9f11e6345a.jpg)

<details>
<summary>text_image</summary>

Basic
CLIP score
Image-based
No filtering
Rand. subset
Text-based
</details>

Figure 3: Correlation between small and medium scale baselines. Smaller scales can serve as useful guides for larger scales. Results for additional scales are shown in Appendix Figure 22.

for a ViT-B/16 or ViT-B/32, training on DATACOMP-1B outperforms training on OpenAI's WIT and LAION-2B. Additionally, we found that modifying hyperparameters such as training steps and batch size minimally affects the relative ordering of different data curation methods on downstream performance. Details on hyperparameter ablations are in Appendix L.

# 5.3 Evaluation trends

ImageNet accuracy is indicative, but not the complete picture. Similarly to Kornblith et al. [84], in Appendix Figure 25 we find that ImageNet performance is highly correlated with the average performance across all datasets we study, with an overall correlation of 0.99. $^{3}$ However, ImageNet performance is not representative of all evaluation tasks, as the correlation between ImageNet accuracy and accuracy on other individual datasets varies substantially, in some cases even exhibiting a negative correlation, as discussed in Appendix R.

Robustness and fairness. While typical models trained on a target task suffer large performance drops under data distribution shift, zero-shot CLIP models are known to exhibit strong performance across many distributions $[111]$ . In Appendix Figure 26, we show that CLIP models trained with data from our pool are more robust to distribution shift than ImageNet-trained models from Taori et al. $[139]$ 's testbed. Examining geographic diversity, we find that our models are better than ImageNet-trained models, but fall short of models fine-tuned on diverse curated datasets (see Appendix Figure 21). We also perform a face classification analysis and identify demographic biases in our models: notably, the BYOD datasets we consider can increase the risk of misclassification. See Appendix Q for more fairness and diversity analyses.

# 6 Limitations and conclusion

In terms of societal risks, creating an index of image-text pairs from the public internet can be problematic. The internet contains unsafe, toxic, and sensitive content, which ideally should not percolate into machine learning datasets. Though we take steps to remove NSFW content and blur human faces to protect privacy, we hope future work will further explore the biases and risks from COMMONPOOL and DATACOMP-1B. We see several additional directions for future work, including 1) Curating more data sources. 2) Improved data filtering algorithms. 3) Further supervision signals (e.g., image captions coming from captioning models). 4) Additional input modalities (e.g., video, 3D objects). 5) Broader evaluations for vision-and-language and robotics tasks.

Overall, we see DATACOMP as a first step towards improving training datasets, and hope our new benchmark will foster further research. By providing a controlled experimental setting, DATACOMP enables researchers to iterate on dataset design on rigorous empirical foundations. We open-source all of our code, data, and infrastructure, and hope these resources will help the community build the next generation of multimodal datasets.

# Acknowledgements

SYG and JH are supported by NSF Graduate Research Fellowships. GS is supported by the Onassis Foundation - Scholarship ID: F ZS 056-1/2022-2023. GD has been supported by the Onassis Fellowship (Scholarship ID: F ZS 012-1/2022-2023), the Bodossaki Fellowship and the Leventis Fellowship. This research has been supported by NSF Grants AF 1901292, CNS 2148141, DMS 2134012, TRIPODS II-DMS 2023166, Tripods CCF 1934932, IFML CCF 2019844 and research gifts by Western Digital, WNCG IAP, UT Austin Machine Learning Lab (MLL), Cisco, the Len Blavatnik and the Blavatnik Family Foundation, the Stanly P. Finch Centennial Professorship in Engineering, Open Philanthropy, Google, Microsoft, and the Allen Institute for AI.

We would like to thank Amro Abbas, Danny Bickson, Alper Canberk, Jessie Chapman, Brian Cheung, Tim Dettmers, Joshua Gardner, Nancy Garland, Sachin Goyal, Huy Ha, Zaid Harchaoui, Ari Holtzman, Andrew Hundt, Andy Jones, Adam Klivans, Ronak Mehta, Sachit Menon, Ari Morcos, Raviteja Mullapudi, Jonathon Shlens, Brandon McKinzie, Alexander Toshev, David Grangier, Navdeep Jaitly, Kentrell Owens, Marco Tulio Ribeiro, Shiori Sagawa, Christoph Schuhmann, Matthew Wallingford, and Ross Wightman for helpful feedback at various stages of the project. We are particularly grateful to Daniel Levy and Alec Radford for early encouragement to pursue this project and feedback on the experimental design.

We thank Stability AI and the Gauss Centre for Supercomputing e.V. $^{4}$ for providing us with compute resources to train models. We are thankful for the compute time provided through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster $[78]$ at Jülich Supercomputing Centre (JSC), and for storage resources on JUST $[50]$ granted and operated by JSC, as well as computing and storage resources from the Helmholtz Data Federation (HDF).

# References

[1] cc2dataset. https://github.com/rom1504/cc2dataset.   
[2] CLD3. https://github.com/google/cld3.   
[3] Common Crawl. https://commoncrawl.org.   
[4] dataset2metadata. https://github.com/mlfoundations/dataset2metadata.   
[5] img2dataset. https://github.com/rom1504/img2dataset.   
[6] Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication, 2023. https://arxiv.org/abs/2303.09540.   
[7] Pankaj K. Agarwal, Sariel Har-Peled, and Kasturi R. Varadarajan. Approximating extent measures of points. Journal of the ACM (JACM), 2004. https://doi.org/10.1145/1008731.1008736.   
[8] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://openreview.net/forum?id=EbMuimAbPbs.   
[9] Abhijeet Awasthi, Sabyasachi Ghosh, Rasna Goyal, and Sunita Sarawagi. Learning from rules generalizing labeled exemplars. In International Conference on Learning Representations (ICLR), 2020. https://openreview.net/forum?id=SkeuexBtDr.   
[10] Stephen H Bach, Daniel Rodriguez, Yintao Liu, Chong Luo, Haidong Shao, Cassandra Xia, Souvik Sen, Alex Ratner, Braden Hancock, Houman Alborzi, Rahul Kuchhal, Christopher Ré, and Rob Malkin. Snorkel drybell: A case study in deploying weak supervision at industrial scale. In Special Interest Group on Management of Data (SIGMOD), 2019. https://arxiv.org/abs/1812.00417.   
[11] Olivier Bachem, Mario Lucic, and Andreas Krause. Coresets for nonparametric estimation - the case of dp-means. In International Conference on Machine Learning (ICML), 2015. https://proceedings.mlr.press/v37/bachem15.html.   
[12] Peter Bandi, Oscar Geessink, Quirine Manson, Marcory Van Dijk, Maschenka Balkenhol, Meyke Hermsen, Babak Ehteshami Bejnordi, Byungjae Lee, Kyunghyun Paeng, Aoxiao Zhong, et al. From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge. IEEE Transactions on Medical Imaging, 2018. https://pubmed.ncbi.nlm.nih.gov/30716025/.   
[13] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems (NeurIPS), volume 32. Curran Associates, Inc., 2019. https://proceedings.neurips.cc/paper/2019/file/97af07a14cacba681feacf3012730892-Paper.pdf.   
[14] Sara Beery, Elijah Cole, and Arvi Gjoka. The iwildcam 2020 competition dataset, 2020. https://arxiv.org/abs/2004.10340.   
[15] Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: misogyny, pornography, and malignant stereotypes, 2021. https://arxiv.org/abs/2110.01963.   
[16] Vighnesh Birodkar, Hossein Mobahi, and Samy Bengio. Semantic redundancies in image-classification datasets: The 10% you don't need. arXiv preprint arXiv:1901.11409, 2019. https://arxiv.org/abs/1901.11409.   
[17] Yonatan Bitton, Nitzan Bitton Guetta, Ron Yosef, Yuval Elovici, Mohit Bansal, Gabriel Stanovsky, and Roy Schwartz. WinoGAViL: Gamified association benchmark to challenge vision-and-language models, 2022. https://arxiv.org/abs/2207.12576.   
[18] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In European Conference on Computer Vision (ECCV), 2014. https://link.springer.com/chapter/10.1007/978-3-319-10599-4\_29.

[19] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems (NeurIPS), 2020. https://proceedings.neurips.cc/paper\_files/paper/2020/file/1457c0d6bfbcb4967418bfb8ac142f64a-Paper.pdf.   
[20] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.   
[21] Ethan Caballero, Kshitij Gupta, Irina Rish, and David Krueger. Broken neural scaling laws. International Conference on Learning Representations (ICLR), 2023. https://arxiv.org/abs/2210.14891.   
[22] Nicholas Carlini, Matthew Jagielski, Christopher A Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, and Florian Tramèr. Poisoning web-scale training datasets is practical, 2023. https://arxiv.org/abs/2302.10149.   
[23] Stephanie C. Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H. Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://arxiv.org/abs/2205.05055.   
[24] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. https://arxiv.org/abs/2102.08981.   
[25] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali: A jointly-scaled multilingual language-image model. In International Conference on Learning Representations (ICLR), 2022. https://arxiv.org/abs/2209.06794.   
[26] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server, 2015. https://arxiv.org/abs/1504.00325.   
[27] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the Institute of Electrical and Electronics Engineers (IEEE), 2017. https://ieeexplore.ieee.org/abstract/document/7891544.   
[28] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning, 2022. https://arxiv.org/abs/2212.07143.   
[29] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Conference on Computer Vision and Pattern Recognition (CVPR), 2018. https://arxiv.org/abs/1711.07846.   
[30] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Conference on Computer Vision and Pattern Recognition (CVPR), 2014. https://openaccess.thecvf.com/content\_cvpr\_2014/html/Cimpoi\_Describing\_Textures\_in\_2014\_CVPR\_paper.html.   
[31] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2011. https://proceedings.mlr.press/v15/coates11a.html.

[32] Michael B. Cohen, Cameron Musco, and Christopher Musco. Input sparsity time low-rank approximation via ridge leverage score sampling. In ACM-SIAM Symposium on Discrete Algorithms, 2017. https://dl.acm.org/doi/10.5555/3039686.3039801.   
[33] C Coleman, C Yeh, S Mussmann, B Mirzasoleiman, P Bailis, P Liang, J Leskovec, and M Zaharia. Selection via proxy: Efficient data selection for deep learning. In International Conference on Learning Representations (ICLR), 2020. https://arxiv.org/abs/1906.11829.   
[34] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Annual Meeting of the Association for Computational Linguistics (ACL), 2019. https://arxiv.org/abs/1911.02116.   
[35] R Dennis Cook. Detection of influential observation in linear. Technometrics, 19(1):15–18, 1977.   
[36] Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid, and Deva Ramanan. Tao: A large-scale benchmark for tracking any object. In European Conference on Computer Vision (ECCV), 2020. https://arxiv.org/abs/2005.10356.   
[37] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Conference on Computer Vision and Pattern Recognition (CVPR), 2009. https://ieeexplore.ieee.org/abstract/document/5206848.   
[38] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people, 2021. https://arxiv.org/abs/2111.11431.   
[39] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021. https://openreview.net/forum?id=YicbFdNTTy.   
[40] Matthijs Douze, Giorgos Tolias, Ed Pizzi, Zoë Papakipos, Lowik Chanussot, Filip Radenovic, Tomas Jenicek, Maxim Maximov, Laura Leal-Taixé, Ismail Elezi, Ondrej Chum, and Cristian Canton-Ferrer. The 2021 image similarity dataset and challenge, 2021. https://arxiv.org/abs/2106.09672.   
[41] Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. Understanding dataset difficulty with v-usable information. In International Conference on Machine Learning (ICML), 2022. https://arxiv.org/abs/2110.08420.   
[42] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results, 2007. http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html.   
[43] Sabri Eyuboglu, Bojan Karlaš, Christopher Ré, Ce Zhang, and James Zou. dcbench: a benchmark for data-centric ai systems. In Proceedings of the Sixth Workshop on Data Management for End-To-End Machine Learning, 2022. https://dl.acm.org/doi/abs/10.1145/3533028.3533310.   
[44] Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, and Ludwig Schmidt. Data determines distributional robustness in contrastive language image pre-training (clip). In International Conference on Machine Learning (ICML), 2022. https://arxiv.org/abs/2205.01397.   
[45] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories. Conference on Computer Vision and Pattern Recognition (CVPR) Workshop, 2004. https://ieeexplore.ieee.org/document/1384978.   
[46] Dan Feldman, Matthew Faulkner, and Andreas Krause. Scalable training of mixture models via coresets. In Advances in Neural Information Processing Systems (NeuIPS), 2011. https://proceedings.neurips.cc/paper\_files/paper/2011/file/2b6d65b9a9445c4271ab9076ead5605a-Paper.pdf.

[47] Daniel Y. Fu, Mayee F. Chen, Frederic Sala, Sarah M. Hooper, Kayvon Fatahalian, and Christopher Ré. Fast and three-rious: Speeding up weak supervision with triplet methods. In International Conference on Machine Learning (ICML), 2020. https://arxiv.org/abs/2002.11955.   
[48] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012. https://ieeexplore.ieee.org/abstract/document/6248074.   
[49] Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In International Conference on Machine Learning, pp. 2242–2251. PMLR, 2019.   
[50] Stephan Graf and Olaf Mextorf. Just: Large-scale multi-tier storage infrastructure at the jülich supercomputing centre. Journal of large-scale research facilities JLSRF, 2021. https://jlsrf.org/index.php/lsf/article/view/180.   
[51] Chengcheng Guo, Bo Zhao, and Yanbing Bai. Deepcore: A comprehensive library for coreset selection in deep learning, 2022. https://arxiv.org/abs/2204.08499.   
[52] Han Guo, Nazneen Fatema Rajani, Peter Hase, Mohit Bansal, and Caiming Xiong. Fastif: Scalable influence functions for efficient model interpretation and debugging, 2020. https://arxiv.org/abs/2012.15781.   
[53] Jia Guo, Jiankang Deng, Alexandros Lattas, and Stefanos Zafeiriou. Sample and computation redistribution for efficient face detection. In International Conference on Learning Representations (ICLR), 2021. https://arxiv.org/abs/2105.04714.   
[54] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019.   
[55] Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, and Noah A. Smith. Annotation artifacts in natural language inference data. In Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2018. https://aclanthology.org/N18-2017.   
[56] Kelvin Guu, Albert Webson, Ellie Pavlick, Lucas Dixon, Ian Tenney, and Tolga Bolukbasi. Simfluence: Modeling the influence of individual training examples by simulating training runs, 2023. https://arxiv.org/abs/2303.08114.   
[57] Frank R Hampel. The influence curve and its role in robust estimation. Journal of the american statistical association, 1974. https://www.jstor.org/stable/2285666.   
[58] Xiaochuang Han, Byron C Wallace, and Yulia Tsvetkov. Explaining black box predictions and unveiling data artifacts through influence functions, 2020. https://arxiv.org/abs/2005.06676.   
[59] A. Hanna, Emily L. Denton, Andrew Smart, and Jamila Smith-Loud. Towards a critical race methodology in algorithmic fairness. In Conference on Fairness, Accountability, and Transparency (FAccT), 2020. https://arxiv.org/abs/1912.03593.   
[60] Laura Hanu and Unitary team. Detoxify, 2020. https://github.com/unitaryai/detoxify.   
[61] Sariel Har-Peled and Soham Mazumdar. On coresets for k-means and k-median clustering. In Symposium on Theory of Computing (STOC), 2004. https://doi.org/10.1145/1007352.1007400.   
[62] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. https://arxiv.org/abs/1512.03385.   
[63] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019. https://arxiv.org/abs/1709.00029.   
[64] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. ICCV, 2021. https://arxiv.org/abs/2006.16241.

[65] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. https://arxiv.org/abs/1907.07174.   
[66] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models, 2022. https://arxiv.org/abs/2203.15556.   
[67] Raphael Hoffmann, Congle Zhang, Xiao Ling, Luke Zettlemoyer, and Daniel S Weld. Knowledge-based weak supervision for information extraction of overlapping relations. In Annual Meeting of the Association for Computational Linguistics (ACL), 2011. https://aclanthology.org/P11-1055.   
[68] Andrew Hundt, William Agnew, Vicky Zeng, Severin Kacianka, and Matthew Gombolay. Robots enact malignant stereotypes. In Conference on Fairness, Accountability, and Transparency (FAccT), 2022. https://arxiv.org/abs/2207.11569.   
[69] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. OpenCLIP, July 2021. https://doi.org/10.5281/zenodo.5143773.   
[70] Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt. Patching open-vocabulary models by interpolating weights. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://arXiv.org/abs/2208.05592.   
[71] Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Predicting predictions from training data, 2022. https://arxiv.org/abs/2202.00622.   
[72] Tanuj Jain, Christopher Lennan, Zubin John, and Dat Tran. Imagededup, 2019. https://github.com/idealo/imagededup.   
[73] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2102.05918.   
[74] Mon-Fong Jiang, Shian-Shyong Tseng, and Chih-Ming Su. Two-phase clustering process for outliers detection. Pattern recognition letters, 2001. https://www.sciencedirect.com/science/article/abs/pii/S0167865500001318.   
[75] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 2019. https://arxiv.org/abs/1702.08734.   
[76] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. Conference on Computer Vision and Pattern Recognition (CVPR), 2017. https://arxiv.org/abs/1612.06890.   
[77] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. In Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2017. https://arxiv.org/abs/1607.01759.   
[78] Juelich Supercomputing Center. JUWELS Booster Supercomputer, 2020. https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html#hardware-configuration-of-the-system-name-booster-module.   
[79] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. https://arxiv.org/abs/2001.08361.   
[80] Kimmo Karkkainen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2021. https://arxiv.org/abs/1908.04913.

[81] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International Conference on Machine Learning (ICML), 2017. https://arxiv.org/abs/1703.04730.   
[82] Pang Wei Koh, Kai-Siang Ang, Hubert Teo, and Percy S Liang. On the accuracy of influence functions for measuring group effects. Advances in Neural Information Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/1905.13289.   
[83] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2012.07421.   
[84] Simon Kornblith, Jonathon Shlens, and Quoc V Le. Do better imagenet models transfer better? In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. https://arxiv.org/abs/1805.08974.   
[85] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In International Conference on Computer Vision Workshops (ICML), 2013. https://www.cv-foundation.org/openaccess/content\_iccv\_workshops\_2013/W19/html/Krause\_3D\_Object\_Representations\_2013\_ICCV\_paper.html.   
[86] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images, 2009. https://www.cs.toronto.edu/\~kriz/learning-features-2009-TR.pdf.   
[87] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2012. https://proceedings.neurips.cc/paper\_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.   
[88] Ronan Le Bras, Swabha Swayamdipta, Chandra Bhagavatula, Rowan Zellers, Matthew Peters, Ashish Sabharwal, and Yejin Choi. Adversarial filters of dataset biases. In International Conference on Machine Learning (ICML), 2020. https://arxiv.org/abs/2002.04108.   
[89] Yann LeCun. The MNIST database of handwritten digits, 1998. http://yann.lecun.com/exdb/mnist/.   
[90] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. In Annual Meeting of the Association for Computational Linguistics (ACL), 2021. https://arxiv.org/abs/2107.06499.   
[91] Yi Li and Nuno Vasconcelos. Repair: Removing representation bias by dataset resampling. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. https://arxiv.org/abs/1904.07911.   
[92] Yulong Liu, Guibo Zhu, Bin Zhu, Qi Song, Guojing Ge, Haoran Chen, GuanHui Qiao, Ru Peng, Lingxiang Wu, and Jinqiao Wang. Taisu: A 166m large-scale high-quality dataset for chinese vision-language pre-training. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://proceedings.neurips.cc/paper\_files/paper/2022/file/6a386d703b50f1cf1f61ab02a15967bb-Paper-Datasets\_and\_Benchmarks.pdf.   
[93] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. Conference on Computer Vision and Pattern Recognition (CVPR), 2022. https://arxiv.org/abs/2201.03545.   
[94] Mario Lucic, Matthew Faulkner, Andreas Krause, and Dan Feldman. Training gaussian mixture models at scale via coresets. Journal of Machine Learning Research (JMLR), 2018. http://jmlr.org/papers/v18/15-506.html.   
[95] S. Maji, J. Kannala, E. Rahtu, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft, 2013. https://arxiv.org/abs/1306.5151.   
[96] Gideon S Mann and Andrew McCallum. Generalized expectation criteria for semi-supervised learning with weakly labeled data. Journal of Machine Learning Research (JMLR), 2010. https://www.jmlr.org/papers/v11/mann10a.html.

[97] Mark Mazumder, Colby Banbury, Xiaozhe Yao, Bojan Karlaš, William Gaviria Rojas, Sudnya Diamos, Greg Diamos, Lynn He, Douwe Kiela, David Jurado, David Kanter, Rafael Mosquera, Juan Ciro, Lora Aroyo, Bilge Acun, Sabri Eyuboglu, Amirata Ghorbani, Emmett Goodman, Tariq Kane, Christine R. Kirkpatrick, Tzu-Sheng Kuo, Jonas Mueller, Tristan Thrush, Joaquin Vanschoren, Margaret Warren, Adina Williams, Serena Yeung, Newsha Ardalani, Praveen Paritosh, Ce Zhang, James Zou, Carole-Jean Wu, Cody Coleman, Andrew Ng, Peter Mattson, and Vijay Janapa Reddi. Dataperf: Benchmarks for data-centric ai development, 2022. https://arxiv.org/abs/2207.10062.   
[98] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning (ICML), 2020. https://arxiv.org/abs/1906.01827.   
[99] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In Advances in Neural Information Processing Systems (NeurIPS) Workshops, 2011. https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37648.pdf.   
[100] Andrew Ng, Dillon Laird, and Lynn He. Data-centric ai competition, 2021. https://https-deeplearning-ai.github.io/data-centric-comp/.   
[101] Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, and Ludwig Schmidt. Quality not quantity: On the interaction between dataset design and robustness of clip. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://openreview.net/forum?id=LTCBavFWp5C.   
[102] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics and Image Processing, 2008. https://ieeexplore.ieee.org/document/4756141.   
[103] OpenAI. Gpt-4 technical report, 2023. https://arxiv.org/abs/2303.08774.   
[104] Vicente Ordonez, Girish Kulkarni, and Tamara L. Berg. Im2text: Describing images using 1 million captioned photographs. In Advances in Neural Information Processing Systems (NeurIPS), 2011. https://papers.nips.cc/paper\_files/paper/2011/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf.   
[105] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012. https://ieeexplore.ieee.org/document/6248092.   
[106] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. In Advances in Neural Information Processing Systems (NeurIPS), 2021. https://arxiv.org/abs/2107.07075.   
[107] Hieu Pham, Zihang Dai, Golnaz Ghiasi, Hanxiao Liu, Adams Wei Yu, Minh-Thang Luong, Mingxing Tan, and Quoc V. Le. Combined scaling for zero-shot transfer learning, 2021. https://arxiv.org/abs/2111.10050.   
[108] Vinay Uday Prabhu and Abeba Birhane. Large image datasets: A pyrrhic win for computer vision? In Winter Conference on Applications of Computer Vision (WACV), 2020. https://arxiv.org/abs/2006.16923.   
[109] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems (NeurIPS), 2020. https://arxiv.org/abs/2002.08484.   
[110] Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, and Dhruv Mahajan. Filtering, distillation, and hard negatives for vision-language pre-training. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. https://arxiv.org/abs/2301.02280.   
[111] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2103.00020.

[112] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022. https://arxiv.org/abs/2212.04356.   
[113] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research (JMLR), 2020. https://arxiv.org/abs/1910.10683.   
[114] Vikram V. Ramaswamy, Sing Yu Lin, Dora Zhao, Aaron B. Adcock, Laurens van der Maaten, Deepti Ghadiyaram, and Olga Russakovsky. Beyond web-scraping: Crowd-sourcing a geodiverse dataset, 2023. https://arxiv.org/abs/2301.02560.   
[115] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2102.12092.   
[116] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. https://arxiv.org/abs/2204.06125.   
[117] A. J. Ratner, B. Hancock, J. Dunnmon, F. Sala, S. Pandey, and C. Ré. Training complex models with multi-task weak supervision. In Association for the Advancement of Artificial Intelligence (AAAI), 2019. https://arxiv.org/abs/1810.02840.   
[118] Alexander J Ratner, Christopher M De Sa, Sen Wu, Daniel Selsam, and Christopher Ré. Data programming: Creating large training sets, quickly. In Advances in Neural Information Processing Systems (NeurIPS), 2016. https://arxiv.org/abs/1605.07723.   
[119] Alexander J Ratner, Stephen H Bach, Henry Ehrenberg, Jason Fries, Sen Wu, and Christopher Ré. Snorkel: Rapid training data creation with weak supervision. In Very Large Data Bases Conference (VLDB), 2017. https://arxiv.org/abs/1711.10160.   
[120] Christopher Ré. Overton: A data system for monitoring and improving machine-learned products. In 10th Conference on Innovative Data Systems Research, CIDR 2020, Amsterdam, The Netherlands, January 12-15, 2020, Online Proceedings. www.cidrdb.org, 2020. URL http://cidrdb.org/cidr2020/papers/p33-re-cidr20.pdf.   
[121] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classifiers generalize to ImageNet? In International Conference on Machine Learning (ICML), 2019. http://proceedings.mlr.press/v97/recht19a.html.   
[122] William A Gaviria Rojas, Sudnya Diamos, Keertan Ranjan Kini, David Kanter, Vijay Janapa Reddi, and Cody Coleman. The dollar street dataset: Images representing the geographic and socioeconomic diversity of the world. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2022. https://openreview.net/forum?id=qnfYsave0U4.   
[123] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. https://arxiv.org/abs/2112.10752.   
[124] Peter J Rousseeuw and Mia Hubert. Robust statistics for outlier detection. Wiley interdisciplinary reviews: Data mining and knowledge discovery, 2011. http://i2pc.es/coss/Docencia/SignalProcessingReviews/Rousseeuw2011.pdf.   
[125] Peter J Rousseeuw and Mia Hubert. Anomaly detection by robust statistics. Wiley interdisciplinary reviews: Data mining and knowledge discovery, 2018. https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1236.   
[126] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 2015. https://arxiv.org/abs/1409.0575.   
[127] Shiori Sagawa, Pang Wei Koh, Tony Lee, Irena Gao, Sang Michael Xie, Kendrick Shen, Ananya Kumar, Weihua Hu, Michihiro Yasunaga, Henrik Marklund, Sara Beery, Etienne David, Ian Stavness, Wei Guo, Jure Leskovec, Kate Saenko, Tatsunori Hashimoto, Sergey Levine, Chelsea Finn, and Percy Liang. Extending the wilds benchmark for unsupervised

adaptation. In International Conference on Learning Representations (ICLR), 2022. https://arxiv.org/abs/2112.05090.   
[128] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION-400M: Open dataset of clip-filtered 400 million image-text pairs, 2021. https://arxiv.org/abs/2111.02114.   
[129] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2022. https://openreview.net/forum?id=M3Y74vmsMcY.   
[130] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations (ICLR), 2018. https://openreview.net/forum?id=H1aIuk-RW.   
[131] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Annual Meeting of the Association for Computational Linguistics (ACL), 2018. https://aclanthology.org/P18-1238/.   
[132] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks?, 2021. https://arxiv.org/abs/2107.06383.   
[133] Changho Shin, Winfred Li, Harit Vishwakarma, Nicholas Roberts, and Frederic Sala. Universalizing weak supervision. In International Conference on Learning Representations (ICLR), 2022. https://openreview.net/forum?id=YpPiNigTzMT.   
[134] Haoyu Song, Li Dong, Weinan Zhang, Ting Liu, and Furu Wei. CLIP models are few-shot learners: Empirical studies on VQA and visual entailment. In Annual Meeting of the Association for Computational Linguistics (ACL), 2022. https://aclanthology.org/2022.acl-long.421.   
[135] Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. https://openreview.net/forum?id=UmvSlP-PyV.   
[136] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021. https://arxiv.org/abs/2103.01913.   
[137] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. The german traffic sign recognition benchmark: a multi-class classification competition. In International Joint Conference on Neural Networks (IJCNN), 2011. https://ieeexplore.ieee.org/document/6033395.   
[138] Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and Yejin Choi. Dataset cartography: Mapping and diagnosing datasets with training dynamics. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. https://aclanthology.org/2020.emnlp-main.746.   
[139] Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, and Ludwig Schmidt. Measuring robustness to natural distribution shifts in image classification. In Advances in Neural Information Processing Systems (NeurIPS), 2020. https://dl.acm.org/doi/abs/10.5555/3495724.3497285.   
[140] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. YFCC100M: The new data in multimedia research. Communications of the ACM, 2016. https://arxiv.org/abs/1503.01817.

[141] Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. An empirical study of example forgetting during deep neural network learning. In International Conference on Learning Representations (ICLR), 2018. https://arxiv.org/abs/1812.05159.   
[142] Bastiaan S Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant CNNs for digital pathology, 2018. https://arxiv.org/abs/1806.03962.   
[143] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. In Advances in Neural Information Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/1905.13549.   
[144] Ryan Webster, Julien Rabin, Loic Simon, and Frederic Jurie. On the de-duplication of laion-2b, 2023. https://arxiv.org/abs/2303.12733.   
[145] Kai Wei, Rishabh Iyer, and Jeff Bilmes. Submodularity in data subset selection and active learning. In International Conference on Machine Learning (ICML), 2015. https://proceedings.mlr.press/v37/wei15.html.   
[146] Jianxiong Xiao, Krista A Ehinger, James Hays, Antonio Torralba, and Aude Oliva. Sun database: Exploring a large collection of scene categories. International Journal of Computer Vision (IJCV), 2016. https://link.springer.com/article/10.1007/s11263-014-0748-y.   
[147] Kaiyu Yang, Klint Qinami, Li Fei-Fei, Jia Deng, and Olga Russakovsky. Towards fairer datasets: filtering and balancing the distribution of the people subtree in the imagenet hierarchy. In Conference on Fairness, Accountability, and Transparency (FAccT), 2020. https://arxiv.org/abs/1912.07726.   
[148] Kaiyu Yang, Jacqueline H Yau, Li Fei-Fei, Jia Deng, and Olga Russakovsky. A study of face obfuscation in ImageNet. In International Conference on Machine Learning (ICML), 2022. https://arxiv.org/abs/2103.06191.   
[149] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. In International Conference on Learning Representations (ICLR), 2022. https://arxiv.org/abs/2111.07783.   
[150] Shuhei Yokoo. Contrastive learning with large memory bank and negative embedding subtraction for accurate copy detection, 2021. https://arxiv.org/abs/2112.04323.   
[151] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2014. https://aclanthology.org/Q14-1006/.   
[152] Dantong Yu, Gholamhosein Sheikholeslami, and Aidong Zhang. Findout: Finding outliers in very large datasets. Knowledge and information Systems, 2002. https://link.springer.com/article/10.1007/s101150200013.   
[153] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision, 2021. https://arxiv.org/abs/2111.11432.   
[154] Man-Ching Yuen, Irwin King, and Kwong-Sak Leung. A survey of crowdsourcing systems. In SocialCom. IEEE, 2011. https://ieeexplore.ieee.org/document/6113213.   
[155] Matei Zaharia, Reynold S Xin, Patrick Wendell, Tathagata Das, Michael Armbrust, Ankur Dave, Xiangrui Meng, Josh Rosen, Shivaram Venkataraman, Michael J Franklin, et al. Apache spark: a unified engine for big data processing. Communications of the ACM, 2016. https://dl.acm.org/doi/10.1145/2934664.   
[156] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruysen, Carlos Riquelme, Mario Lucic, Josip Djolonga, André Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, Lucas Beyer, Olivier Bachem, Michael Tschannen, Marcin Michalski, Olivier Bousquet, Sylvain Gelly, and Neil Houlsby. The visual task adaptation benchmark, 2019. http://arxiv.org/abs/1910.04867.

[157] Jieyu Zhang, Yue Yu, Yinghao Li, Yujing Wang, Yaming Yang, Mao Yang, and Alexander Ratner. WRENCH: A comprehensive benchmark for weak supervision. In NeurIPS, 2021. URL https://openreview.net/forum?id=Q9SKS5k8io.   
[158] Jieyu Zhang, Cheng-Yu Hsieh, Yue Yu, Chao Zhang, and Alexander Ratner. A survey on programmatic weak supervision, 2022. https://arxiv.org/abs/2202.05433.   
[159] Zhifei Zhang, Yang Song, and Hairong Qi. Age progression/regression by conditional adversarial autoencoder. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. https://arxiv.org/abs/1702.08423.   
[160] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In European Conference on Computer Vision (ECCV), 2022. https://arxiv.org/abs/2201.02605.

# Appendix

# Contents

1 Introduction 1   
2 Related Work 3   
3 The DATACOMP benchmark 4

3.1 Competition design 4   
3.2 COMMONPOOL generation, for the filtering track 5   
3.3 The bring your own data (BYOD) track 5   
3.4 Training 5   
3.5 Evaluation 6

4 Baselines 6

4.1 Filtering baselines 6   
4.2 BYOD baselines 7

5 Results and discussion 7

5.1 Building better datasets 7   
5.2 DATACOMP design analyses 8   
5.3 Evaluation trends 9

6 Limitations and conclusion 9

A Benchmark rules 24

A.1 Filtering track rules 24   
A.2 Bring your own data track: amendments 24

B Contributions 25

B.1 Candidate pool 25   
B.2 Participant tooling 25   
B.3 Baselines 25   
B.4 Leadership and Advising 25

C Additional related work 26   
D Parsing Common Crawl 26   
E Not safe for work (NSFW) filtering 27   
F Deduplication against evaluation sets 27

G Face blurring 29

H DATACOMP COMMONPOOL creation pipeline 31

I COMMONPOOL statistics 32

J Efficient training on data subsets 35

K Effect of duplicates in the training data 35

L Hyperparameter ablations 36

L.1 Batch size 36

L.2 Model architecture 36

L.3 Number of training steps 36

M Detector-based baselines 39

N Training details 40

O Evaluation details 40

O.1 Visual Question Answering 42

P Baseline details 43

P.1 Filtering track 47

P.2 BYOD track 48

P.2.1 Additional results 49

Q Fairness and biases 49

Q.1 Diversity 49

Q.2 Fairness 50

R Extra figures and tables 53

S Datasheet 60

S.1 Motivation 60

S.2 Composition 60

S.3 Collection Process 62

S.4 Preprocessing, Cleaning, and/or Labeling 63

S.5 Uses 64

S.6 Distribution 65

S.7 Maintenance 65

# A Benchmark rules

We provide concrete rules below for the two competition tracks that comprise DATACOMP: filtering and BYOD. Additionally, we provide a checklist, which encourages participants to specify design decisions, which allows for more granular comparison between submissions.

# A.1 Filtering track rules

- Participants can enter submissions for one or many different scales: small, medium, large or xlarge, which represent the raw number of image-text pairs in CommonPool that should be filtered.   
- After choosing a scale, participants generate a list of uids, where each uid refers to a COMMONPOOL sample. The list of uids is used to recover image-text pairs from the pool, which is used for downstream CLIP training.   
- Duplicate uids are allowed.   
- Participants are not allowed to modify the training procedure. Hence, changing hyperparameters, model architecture, optimizer, compute budget, or number of training steps is not allowed. Changing any other training details is also not allowed.   
- Participants are strongly encouraged to submit and open-source both the list of uids and the code used to generate this list; however, this is not required.   
- To avoid overfitting, we do not permit running any code or algorithmic dependence on the test images of the evaluation tasks. However, use of other images associated with these tasks (e.g., supervised training sets) is permitted.   
- Participants can use templates or class labels from the downstream tasks in their filtering algorithms.

For clarity, we include some examples of permitted and forbidden uses:

√ We permit using the ImageNet class label “triceratops” in a filtering algorithm.

× We forbid examining individual or aggregate predictions on the test sets of the evaluation tasks.

# A.2 Bring your own data track: amendments

To facilitate more open-ended exploration, we provide amendments to the Track 1 competition to allow for more diverse submissions in Track 2.

- Participants are allowed to augment COMMONPOOL data with existing datasets, so long as these data sources do not contain test images from the evaluation tasks. Participants can use data from any COMMONPOOL; however, they are not required to do so.   
- Assembling one's own dataset is allowed; however, test images from the evaluation tasks can neither be contained nor otherwise used to construct said dataset. We encourage releasing the image urls or the images themselves in addition to the text for each image. We also encourage rigorous documentation of face-blurring and other data safety checks (see Section 3.2 for more details). We reserve the right to run our own safety code on participant provided data and disqualify entries that do not meet adequate safety standards.

Checklist. The following checklist provides the basis for more fine-grained comparison between submissions.

☐ Images from the evaluation tasks are included in my submission. If yes, please specify which datasets.   
☐ I used an existing datasets (e.g., YFCC100M [140]) in my submission. If yes, please specify which datasets. (Note: applies to BYOD only)   
☐ I curated my own data. If yes, please provide (1) image data or urls, (2) text for each image, (3) list of safety steps taken including but not limited to face blurring, explicit content image and text filtering. (Note: applies to BYOD only)

# B Contributions

For this section, contributors are ordered alphabetically.

# B.1 Candidate pool

Candidate pool lead. Vaishaal Shankar

Data collection. Romain Beaumont, Vaishaal Shankar

Pre-processing and metadata. Giannis Daras, Alex Fang (content filtering lead), Samir Yitzhak Gadre (metadata lead), Ryan Marten (deduplication lead), Vivek Ramanujan, Vaishaal Shankar, George Smyrnis (face blurring lead)

# B.2 Participant tooling

Participant tooling lead. Gabriel Ilharco

Resharder. Romain Beaumont, Yair Carmon, Alex Fang, Jonathan Hayase (lead), Gabriel Ilharco, Vivek Ramanujan, Vaishaal Shankar, Georgios Smyrnis

Training. Mehdi Cherti, Gabriel Ilharco, Jenia Jitsev, Vivek Ramanujan, Georgios Smyrnis, Mitchell Wortsman (lead)

Evaluation. Romain Beaumont, Yonatan Bitton, Mehdi Cherti, Dhruba Ghosh (lead), Gabriel Ilharco

Additional infrastructure. Stephen Mussmann, Sarah Pratt

# B.3 Baselines

Baselines lead. Yair Carmon

Filtering track. Yair Carmon, Rahim Enterazi, Alex Fang, Samir Yitzhak Gadre, Gabriel Ilharco, Kalyani Marathe, Thao Nguyen, Eyal Orgad (co-lead), Georgios Smyrnis, Mitchell Wortsman, Jieyu Zhang (co-lead)

BYOD track. Gabriel Ilharco, Thao Nguyen

Experiment babysitting. Alex Fang, Gabriel Ilharco, Samir Yitzhak Gadre

# B.4 Leadership and Advising

Advising. Romain Beaumont, Yair Carmon, Alexandros G. Dimakis, Ali Farhadi, Hannaneh Hajishirzi, Jenia Jitsev, Pang Wei Koh, Ranjay Krishna, Stephen Mussmann, Sewoong Oh, Alexander Ratner, Olga Saukh, Ludwig Schmidt, Vaishaal Shankar, Shuran Song, Richard Vencu

Leadership. Yair Carmon, Alexandros G. Dimakis, Jenia Jitsev, Sewoong Oh, Ludwig Schmidt, Vaishaal Shankar

Overall project lead. Ludwig Schmidt

# C Additional related work

Here we expand on the related work described in Section 2.

Image dataset safety is an active area of research, especially in the context of large-scale dataset construction. In addition to Birhane et al. $[15]$ , who study problematic content in LAION-400M, Yang et al. $[147]$ study the ImageNet dataset and reveal limitations associated with the ImageNet curation strategy—with negative implications for downstream model fairness. Prabhu & Birhane $[108]$ also study the ImageNet dataset and find pornographic content. Both Birhane et al. $[15]$ and Prabhu & Birhane $[108]$ survey ethical conundrums and harms that are borne out of improper dataset curation. In an effort to combat dataset toxicity, we conduct NSFW preprocessing (Section 3.2, Appendix E) and blur detected faces (Section 3.2, Appendix G) during pool construction. We also conduct preliminary fairness evaluations (Section 5.3, Appendix Q) for models trained on our data. We hope COMMONPOOL will serve as a research artifact for future work examining dataset safety.

Beyond data selection, Chan et al. [23] investigate the effects of dataset distribution on emergent properties of transformers, while Fang et al. [44] look at the relationship between data and model robustness to distribution shifts. We hope our extensive evaluation suite comprised of 38 diverse tasks will facilitate similar studies when training multimodal models at large scale.

Others study how to reduce the burdens of training data annotation in the curation process. Classic approaches include distant supervision $[67]$ , crowd-sourced labels $[154]$ , heuristic rules $[9]$ and feature annotation $[96]$ , among others. A recent line of work known as data programming or programmatic weak supervision $[118, 119, 157, 158]$ attempts to reduce annotation cost and is found in many industry applications $[10, 120]$ . In data programming, developers write programmatic labeling functions to automatically label a large amount of unlabeled data. The labeling functions could produce noisy and conflicting labels, so researchers have developed methods to aggregate noisy votes to produce the final training labels $[117, 47, 133]$ .

Previous literature also studies methods for training data attribution, which seek to link a model's behavior (e.g., its accuracy on a particular task or subset of data) to particular subsets of its training data. Such methods include influence functions, a classic technique from robust statistics $[57, 35]$ that uses a second-order Taylor expansion to approximate the effect of removing a training point on the learned model parameters $[81, 82, 58, 52]$ , as well as methods that fit attribution functions directly to the dynamics of repeated training runs $[49, 109, 71, 56]$ . Training data attribution methods assume that we have already trained a model, though they can be subsequently used to refine the training data (e.g., by identifying potentially mislabeled training points $[81]$ ). Our focus in this paper is instead on data curation methods—that is, methods for selecting a subset of the training data to train a model in the first place.

In the context of natural language processing, Swayamdipta et al. [138] proposes a tool for characterizing samples in a dataset based on training dynamics, labelling instances as ambiguous, easy to learn or hard to learn. Previous literature such as work by Le Bras et al. [88], Li & Vasconcelos [91], Gururangan et al. [55] advocate for removing easy instances from the training data. Ethayarajh et al. [41] propose a measure of how difficult a dataset is to learn, V-usable information. Such techniques could be promising directions of further exploration in the context of our benchmark.

Finally, another related line of work is studying scaling trends. In addition to Sorscher et al. $[135]$ , researchers have investigated how model performance changes as a function of compute budget, model size, and number of training samples $[79, 66, 21, 28]$ . However, this line of work does not consider how dataset design may affects scaling trends. Beyond dataset size, we measure the effects of different dataset sources and filtering strategies. While scaling trends are central to our investigations, the purpose of our benchmark is to search for the next generation of large multimodal datasets to facilitate more accurate and reliable models.

# D Parsing Common Crawl

Common Crawl releases metadata files for the websites that they index (i.e., WAT files). They release these files approximately once a month. We consider all files available from 2014 through November of 2022. We first parse these files, utilizing Apache Spark [155] to extract image urls and corresponding alt-text. We map each url, text pair to a uid hash and remove duplicates. This

Table 4: Detoxify positive rates by threshold on 1 million caption subset of Common Crawl. 

<table><tr><td>Threshold</td><td>Toxicity</td><td>Severe Toxicity</td><td>Obscene</td><td>Identity Attack</td><td>Insult</td><td>Threat</td><td>Sexual Explicit</td></tr><tr><td>0.01</td><td>9.5%</td><td>1.0%</td><td>33.4%</td><td>1.8%</td><td>35.0%</td><td>1.3%</td><td>2.0%</td></tr><tr><td>0.1</td><td>3.6%</td><td>0.1%</td><td>0.8%</td><td>0.3%</td><td>1.4%</td><td>0.1%</td><td>1.0%</td></tr></table>

Table 5: Comparing LAION-2B CLIP based NSFW filtering model to Google Vision API Safe Search adult category on a 40,000 random subset of Common Crawl. 

<table><tr><td>Threshold</td><td>False Positive Rate(Relative to Google)</td><td>True Positives(Manual Review)</td><td>Model Positive Rate</td><td>Google API Positive Rate</td></tr><tr><td>0.1</td><td>3.6%</td><td>2</td><td>14.4%</td><td>3.5%</td></tr><tr><td>0.2</td><td>0.6%</td><td>2</td><td>9.1%</td><td>3.5%</td></tr><tr><td>0.3</td><td>0.3%</td><td>3</td><td>7.2%</td><td>3.5%</td></tr></table>

results in 88 billion url, text pairs, which are randomized via a distributed shuffle. Note, we do not consider image content when running uid deduplication at this step. Hence, two identical images with different urls and the same caption would both be retained.

# E Not safe for work (NSFW) filtering

Our data is sourced from Common Crawl, which contains snapshots of the web. Therefore, we apply multiple layers of NSFW content filtering to remove problematic images and captions from COMMONPOOL.

First, we filter our captions with Detoxify $[60]$ , a language model for toxic comment classification. Specifically, we use the multilingual XLM-RoBERTa $[34]$ variant. The model outputs scores between zero and one for the following categories: toxicity, severe toxicity, obscene, identity attack, insult, threat, and sexually explicit. As we had no ground truth for our data, we manually spot check a 1 million random subset of COMMONPOOL at varying thresholds. We found that a threshold of 0.1 provided good coverage of filtering out NSFW text. If any of the detoxify category scores exceeds the threshold, the sample is discarded. Qualitatively, we found that the model struggled with multilingual content, acronyms, and innuendo. Even at 0.1, we noticed there are some captions that are NSFW. However, lowering the threshold further heavily affected false positives. We therefore use a 0.1 threshold for all NSFW categories, which on a random subset of one million captions achieves positive rates shown in Table 4.

Second, on the vision side, we use a modified version of LAION-5B's $[129]$ CLIP-based binary classification NSFW model, which takes CLIP ViT-L/14 visual embeddings as input. We remove the initial multi-category encoder from the model, and retrain on the same data with an initial normalization layer followed by a 4-layer multilayer perceptron. Our retrained model matches the performance of the original model on their manually annotated testset. Specifically, we achieve 97.4% classification accuracy on a held out test set compared to 96.1% for the original LAION NSFW image filtering model. Additional details about the training data can be found in Appendix C.5 of the LAION-5B paper. In brief, the training data contains 682K images that is roughly balanced with images from safe for work and NSFW categories.

To evaluate our model and determine a threshold, we used Google Vision API's SafeSearch explicit content detector to generate labels for an 40,000 random subset of our candidate pool. Specifically, an image is NSFW if SafeSearch classifies it as likely or very likely adult (i.e., sexually explicit). As shown in Table 5, we found that by thresholding at 0.1 we achieve high recall relative to SafeSearch and very few true positives after manual review. We also manually reviewed images classified by SafeSearch as likely or very likely racy and found that the images were either benign, subjectively suggestive but not explicit, or already found in the set of images labeled as adult.

# F Deduplication against evaluation sets

To prevent data leakage, we filter COMMONPOOL by removing duplicate and near-duplicate matches of evaluation set images. See Figure 4 for example query images from Common Crawl and corresponding near-duplicates in our evaluations sets. We consider images as duplicates when

Query Image   
![](images/75a4ed07c041dc0db623bbce2a5656cbab3e811e0051a14418761179b47d5599.jpg)

<details>
<summary>text_image</summary>

McDonald's
</details>

![](images/f55b4c71ddd025fd330b71c61664d84d1a01449bcb5edd482b1d7a21070f798f.jpg)

![](images/3149141fe83bbf39aa119b443cbb612c4c74b7973d5f7d793c662d4967800657.jpg)

<details>
<summary>natural_image</summary>

Scenic view of a traditional Tibetan-style building beside a river with green mountains in the background (no visible text or symbols)
</details>

![](images/18dc6c4e17e96870511d7cf8a6701b6f48caebc2cc150367fd7057517e4c172e.jpg)

<details>
<summary>natural_image</summary>

Close-up of a cheetah's face with a sharp, intense Mandarin expression (no text or symbols visible)
</details>

Eval Image   
![](images/d2b7fe13760d2f9dcbdf53e5cf59c7216c49d4876b041bcdebea9cc7636d9f06.jpg)

<details>
<summary>text_image</summary>

McDonald's
</details>

sun397\_test/s0011797

![](images/c5a1333cdf08deb588eb88128380a8e97375ddcd18d41408f74dfb4da1fc179f.jpg)  
cars\_test/s0005478

![](images/f543020f2041ea04d27ba3ffd719b359d99606fc19726bab24a1a8e725d458cb.jpg)  
country211\_test/s0002908

![](images/715db763bcf7cc1bd3850feafac01772b5ef3c91ba7b9701f60347691dd85b1e.jpg)  
vtab-caltech101\_test/s0004353   
Figure 4: Candidate images (top) that are detected as duplicates against images in the evaluation sets (bottom) are removed from the pool. In addition to exact duplicate images, near-duplicates with variable aspect ratios, JPEG compression, overlays, color adjustment, and artistic rendering are also detected.

![](images/b6cffdbe7b8c33c019b487b05838ae1fedb6633ec8729383b37a3023f9ad8211.jpg)

<details>
<summary>line</summary>

| Method       | Recall (%) | Precision (%) |
| ------------ | ---------- | ------------- |
| Aspect Ratio | 80         | 99.5          |
| Aspect Ratio | 90         | 95.5          |
| Aspect Ratio | 100        | 96.0          |
| Encoding     | 80         | 99.5          |
| Encoding     | 100        | 99.5          |
| Flips        | 80         | 99.5          |
| Flips        | 100        | 99.5          |
| Grayscale    | 80         | 99.5          |
| Grayscale    | 100        | 99.5          |
| Rotations    | 80         | 99.5          |
| Rotations    | 100        | 99.5          |
</details>

![](images/319fba72619a3c8e2e31f557ed08458472af12b79f7b699c4e04a81146f1173f.jpg)

<details>
<summary>line</summary>

| Recall (%) | Aspect Ratio | Encoding | Flips | Grayscale | Rotations |
| ---------- | ------------ | -------- | ----- | --------- | --------- |
| 40         | 100          | 100      | 100   | 100       | 100       |
| 60         | 98           | 99       | 99    | 99        | 99        |
| 80         | 95           | 98       | 98    | 97        | 96        |
| 100        | 75           | 100      | 95    | 75        | 70        |
</details>

Figure 5: Analysis of different de-duplication strategies across a variety of image transformations. We see that the model introduced by Yokoo [150] is better in almost every transformation, with the exception of very aggressive aspect ratio modification.

the cosine similarity between a query (Common Crawl image) feature and a reference (evaluation image) feature is higher than a fixed threshold. We employ the deduplication model proposed by Yokoo [150], which earned 1st place in the Facebook AI Image Similarity Challenge (ISC) [40]. We choose a cosine similarity threshold of 0.604169 to maximize the true duplicates detected, without removing too many false duplicates from the pool. We compare against OpenAI's CLIP ViT-B/32 as a baseline on ISC. We find that for our threshold, the ISC model achieves precision 0.9 and recall 0.8. At a threshold of 0.96, CLIP achieves the same precision 0.9, but a significantly worse recall of 0.02. Approximately $2.8\%$ of downloaded samples are flagged as evaluation set near-duplicates.

To verify the performance of our de-duplication models with greater granularity, we modify the evaluation procedure in Douze et al. [40] to include transformations which are representative of naturally-occurring duplications on the Internet. Specifically, we study: 1) jpeg compression (encoding), 2) image flips, 3) image rotations, 4) aspect ratio modifications, and 5) grayscaling. To do this, we sample $20\%$ of the images from each of our evaluation datasets uniformly at random to serve as a reference set of about 140,000 images. Next we sample 560,000 images uniformly at random from LAION-2B to serve as distractors, for a 4-to-1 distractor to reference ratio. Finally, we apply each of the augmentations above and use threshold filtering to determine duplicates. Figure 5 shows the results from the deduplication model [150] compared with OpenAI's CLIP ViT-L/14. At high recall values, we see that CLIP filtering results in removing over $2\times$ the data as that of the deduplication model from Yokoo [150].

Table 6: Face detection performance on a set of 3293 random images from COMMONPOOL. 

<table><tr><td></td><td>SCRFD-10G</td><td>Amazon Rekognition</td></tr><tr><td>Accuracy</td><td>93.87</td><td>96.57</td></tr><tr><td>Precision</td><td>75.87</td><td>86.09</td></tr><tr><td>Recall</td><td>90.53</td><td>93.75</td></tr></table>

![](images/fce9316862242c4fd3458d7d3e3c8936c271a92269534dbcb2eb031de37fa3e4.jpg)

<details>
<summary>histogram</summary>

| Number of predicted faces | Frequency |
| ------------------------- | --------- |
| 0                         | 1.0e7     |
| 1                         | 0.2e7     |
| 2                         | 0.05e7    |
| 3                         | 0.02e7    |
| 4                         | 0.01e7    |
| 5                         | 0.00e7    |
</details>

Figure 6: Frequency of predicted number of faces in the small COMMONPOOL.

# G Face blurring

As an extra step to safeguard against issues of privacy that may arise from the use of data scraped from the web, we include face blurring as part of our pool creation. To create face metadata, we use the SCRFD face detector $[53]$ to extract bounding boxes for the faces in our images. These bounding boxes are included as part of the image metadata in our pool. We make use of the pretrained SCRFD-10G model. We use the same preprocessing as the one described in the official repository of the paper, with the exception of providing $224 \times 224$ input images (by padding each image to square and then resizing) to limit computation costs. Invoking this model provides us with bounding boxes along with an associated score, which we then compare against a threshold of 0.3 to keep or discard this bounding box. This threshold is the default one used in the repository of SCRFD for the visualization of bounding boxes, and we found it to perform well on our data as discussed next.

In Table 6 we can see the result of face detection on a set of 3293 images from COMMONPOOL. We evaluate the detection on whether the image has visible faces or not (where images such as cartoon drawings of non-real human faces are not considered as positives), and whether the detector has detected these visible faces. We considered an image as a true positive if all the clearly visible faces in the image were detected, based on the above thresholding process. We did not do extensive box labeling. True positives are instead determined by human inspection. We compare the quality of these detections with the Amazon Rekognition system, which is the one upon which the face detections on ImageNet were based [148]. Note that in this scenario, the recall of the detectors is more important than precision (as detecting a few more bounding boxes across our pool does not affect privacy).

To utilize these bounding boxes on our data, we apply a standard blurring pipeline, as proposed by Yang et al. [148]. The result of this process is an image where the faces is blurred and there is a smooth transition from blurred to clean parts of the image. In Figure 6 we see the distribution of faces for the small COMMONPOOL. Note that the majority of images do not contain faces.

As part of our competition pipeline, images are by default blurred during the download process. In Table 7 we can see the results of training on a set of images with the size of our medium scale after filtering with each method, with and without the application of face blurring as provided by our detector. We can see that the difference in performance is small, which suggests that the application of face blurring does not significantly affect the performance on our downstream tasks. However, we note that this design decision may be more detrimental in generative settings, especially when a generative model needs to output faces. Our competition is primarily focused on discriminative tasks, and as such when designing our dataset, we wished to prioritize the safety and privacy of individuals through blurring faces in our download tooling by default.

Table 7: Effect of face blurring on zero-shot performance. Face blurring improves the privacy preservation of our dataset, while affecting model performance negligibly. Results shown for training on a set of images with the size of our medium scale, after filtering with each method. 

<table><tr><td>Filtering</td><td>Face blurring</td><td>ImageNet acc.</td><td>Avg. performance</td></tr><tr><td rowspan="2">CLIP score (B/32, thresh. 0.3) + English filtering</td><td>×</td><td>0.209</td><td>0.246</td></tr><tr><td>√</td><td>0.196</td><td>0.243</td></tr><tr><td rowspan="2">CLIP score (B/32, 30%)</td><td>×</td><td>0.287</td><td>0.301</td></tr><tr><td>√</td><td>0.282</td><td>0.298</td></tr></table>

Finally, we evaluated the detector we used for potential biases. More specifically, we used the detector on the validation set of the FairFace dataset $[80]$ . We found that the central face of the image was detected in all the images of the validation set, regardless of subgroup annotate in the dataset.

# H DATACOMP COMMONPOOL creation pipeline

![](images/98e1dfd9d023699cd2524bcb9f9e2c9f5bc1efd04fe6b4b03a741800ed7065d5.jpg)

<details>
<summary>sankey</summary>

| Category | Value     |
| -------- | --------- |
| Download attempted | 40.0B    |
| CommonCrawl samples | 88.0B    |
| Successful downloaded | 16.8B    |
| Viable samples | 13.1B    |
| Total | 3.7B      |
| Dead links and other download errors | 23.2B     |
| Near-duplicate | 0.5B      |
| NSFW image | 2.8B      |
| NSFW text | 0.8B      |
| Dead links and other download errors | 23.2B     |
| Download not attempted | 48.0B    |
</details>

Figure 7: Data funnel from potential samples in Common Crawl to 13.1B image-text pairs that were suitable for COMMONPOOL. We sampled uniformly 12.8B datapoints for the xlarge COMMONPOOL.

Table 8: Provided metadata for COMMONPOOL. 

<table><tr><td>Generation Time</td><td>Label</td><td>Additional notes</td></tr><tr><td rowspan="6">Step 2</td><td>uid</td><td></td></tr><tr><td>url</td><td>Link to the image.</td></tr><tr><td>text</td><td>Image caption.</td></tr><tr><td>original_width</td><td></td></tr><tr><td>original_height</td><td></td></tr><tr><td>sha256</td><td>Safeguard for data poisoning.</td></tr><tr><td rowspan="7">Step 1</td><td>clip_b32_similarity_score</td><td></td></tr><tr><td>clip_b32_image_features</td><td>In separate file.</td></tr><tr><td>clip_b32_text_features</td><td>In separate file.</td></tr><tr><td>clip_l14_similarity_score</td><td></td></tr><tr><td>clip_l14_image_features</td><td>In separate file.</td></tr><tr><td>clip_l14_text_features</td><td>In separate file.</td></tr><tr><td>face_bboxes</td><td></td></tr><tr><td rowspan="3">Step 2, dropped during Step 3</td><td>nsfw_image_score</td><td></td></tr><tr><td>nsfw_text_score</td><td></td></tr><tr><td>dedup_score</td><td></td></tr></table>

Creating COMMONPOOL was a multistep process, which involved (1) parsing image urls and alt-text from Common Crawl dumps and downloading these images, (2) tagging images with metadata and (3) conducting safety content filtering and evaluation set duplication. In this section we provide an overview of the data pipeline used to create COMMONPOOL. For an overview of our “data funnel” see Figure 7.

1. For the first step, we use parse Common Crawl metadata files to harvest image-text pairs (Section D). We use img2dataset [5] to obtain $\sim 16.8\mathrm{B}$ downloaded samples. This is the first, unfiltered version of COMMONPOOL, and contains only basic information for our images (i.e., the original image height, width, and alt-text caption). During this step we also resize images such that their largest dimension does not exceed 512 pixels. This eases storage requirements for large images, but is still larger than the 224 pixel resolution used for later training stages.

2. For the second step, we process our unfiltered pool and create richer metadata for each image-text pair. We generate the following for each sample:

- CLIP ViT-B/32 and CLIP ViT-L/14 image and text features, with their associated similarities.   
- NSFW scores for the image and the text, using the analysis described in Appendix E.   
- Deduplication score for the image, as described in Appendix F.

\- Bounding boxes for faces detected in the image, using the method described in Appendix G.

3. For the third and final step, we filter our image-text pairs based on the metadata generated during the second stage. We filter out image-text pairs where the NSFW and deduplication scores exceed the respective thresholds (Section E). From the images that pass through this filtering, we keep only the desired amount (e.g., 12.8B images from the xlarge COMMONPOOL). Smaller pools are telescoping subsets of larger pools. We package the metadata and image urls, which is made publicly available to the participants. Note, we do not release raw image data but rather image urls pointing to images.

A summary of the metadata for each sample is found in Table 8. To validate our pipeline for duplication and CLIP feature correctness, we also take ImageNet train though metadata generation as a unit test. Using the deduplication features, we detect that $100\%$ of the images are in fact duplicates. Additionally using the CLIP ViT-B/32 and CLIP ViT-L/14 image features and corresponding text features from OpenAI's 80-prompt ensemble, we achieve $63.36\%$ and $75.54\%$ top-1 accuracies, which match the performance reported in the CLIP paper [111].

When creating pools of different scale (i.e., number of samples), we ensure that smaller pools are subsets of larger pools. For instance, the small COMMONPOOL is a subset of the xlarge COMMONPOOL.

After COMMONPOOL is created, the participants can then download the final image-text pairs using the provided files via img2dataset. To further ease the computational burden on participants, we additionally provide metadata for each sample in COMMONPOOL. Note that when downloading, our img2dataset configuration automatically blurs faces. Hence this is an automatic step on not something participants must do ad hoc.

# I COMMONPOOL statistics

To provide more information about the kinds of samples in our COMMONPOOL, we conduct additional analysis on the small pool, which is an i.i.d. sample of downloaded data and a subset of the larger pools.

In Figure 8 we show CLIP similarity similarity scores between images and their corresponding text. We notice a flatter distribution of CLIP ViT-L/14 scores than corresponding B/32 scores.

Turning our attention to images in COMMONPOOL, in Figure 9, we visualize the aspect ratios and sizes of original images (i.e., before they are downloaded and resized). In Figure 10, we display a distribution of image height and width after download resizing. Notice that the majority of images are around $224 \times 224$ pixels, which is the final resized resolution used for training.

Analysing the textual component of each sample, we visualize frequency of the number of CLIP BPE tokens in the captions (Figure 11) and most common languages (Figure 12). Token counts follow a long-tailed distribution with much more mass in the short sequence range, while English is the predominant language in COMMONPOOL according to fasttext and cld3.

We also look at url statistics. In Figure 13 we see common domain names in COMMONPOOL (e.g., wordpress domains) and common suffixes (e.g., .com or .net).

![](images/02de05c4bac24587e659aca0f17e115483bea08249b00dda765edeb8fe48e0ba.jpg)

<details>
<summary>bar_line</summary>

| CLIP ViT-B/32 similarity score | Frequency | Proportion |
| ----------------------------- | --------- | ---------- |
| 0.00                          | 0         | 0.0        |
| 0.05                          | 0         | 0.0        |
| 0.10                          | 0         | 0.0        |
| 0.15                          | 1e4       | 0.1        |
| 0.20                          | 6e4       | 0.4        |
| 0.25                          | 4e4       | 0.7        |
| 0.30                          | 3e4       | 0.8        |
| 0.35                          | 2e4       | 0.9        |
| 0.40                          | 1e4       | 1.0        |
| 0.45                          | 0         | 1.0        |
</details>

![](images/0c5e8f794f5d528ae46aadf2be972f7fb53a6bfaa74da0fa9ebc243866f1597b.jpg)

<details>
<summary>bar_line</summary>

| CLIP ViT-L/14 similarity score | Frequency | Proportion |
| ------------------------------ | --------- | ---------- |
| 0.00                           | 0         | 0.0        |
| 0.05                           | 10000     | 0.0        |
| 0.10                           | 30000     | 0.1        |
| 0.15                           | 45000     | 0.3        |
| 0.20                           | 48000     | 0.5        |
| 0.25                           | 42000     | 0.7        |
| 0.30                           | 30000     | 0.9        |
| 0.35                           | 15000     | 1.0        |
| 0.40                           | 5000      | 1.0        |
| 0.45                           | 1000      | 1.0        |
</details>

Figure 8: Image-text similarity score distributions using CLIP ViT-B/32 (left) and ViT-L/14 (right) models. We plot samples from the small COMMONPOOL, which are an i.i.d. sample of the xlarge COMMONPOOL.

![](images/de9c4ce6c764f44f4cc6049cef1568f1c869d4eb9d68df04748888ed6d369494.jpg)

<details>
<summary>histogram</summary>

| Original image aspect ratio [width/height] | Frequency |
| ------------------------------------------ | --------- |
| 0.0 - 0.5                                  | 0         |
| 0.5 - 1.0                                  | 1.4e6     |
| 1.0 - 1.5                                  | 4.5e6     |
| 1.5 - 2.0                                  | 1.9e6     |
| 2.0 - 2.5                                  | 0.9e6     |
| 2.5 - 3.0                                  | 0         |
</details>

![](images/83018b8f180e64ef78b5cadf493c1e41523f016a2348b7a945ff98c5bbaf25f1.jpg)

<details>
<summary>histogram</summary>

| Original image area [megapixels] | Frequency |
| -------------------------------- | --------- |
| 0.0 - 0.1                        | 5e6       |
| 0.1 - 0.2                        | 2e6       |
| 0.2 - 0.3                        | 1e6       |
| 0.3 - 0.4                        | 5e5       |
| 0.4 - 0.5                        | 3e5       |
| 0.5 - 0.6                        | 2e5       |
| 0.6 - 0.7                        | 1e5       |
| 0.7 - 0.8                        | 5e4       |
| 0.8 - 0.9                        | 2e4       |
| 0.9 - 1.0                        | 1e4       |
</details>

Figure 9: Statistics for images in the small COMMONPOOL, before applying resizing.

![](images/ef2f1e6892cec0f13499bb869dcd0f5a358b5d67f85d2fdf89413c97f6e9ed3c.jpg)

<details>
<summary>heatmap</summary>

| Value  |
| ------ |
| 512px  |
| 224px  |
| 384px  |
</details>

Figure 10: Image pixel heatmap. Each entry in the above heatmap represents the estimated probability that a pixel is occupied. The center entry has a value of 1.0 as every image has a center pixel. We compute the heatmap over the small COMMONPOOL. Note that image sizes are bounded as we resize all images such that their max dimension does not exceed 512 pixels during dataset download.

![](images/cba4d969ebdfaa5991ad5be7441ce2d09c6e9890cd172ff3e1c465246b68015c.jpg)

<details>
<summary>histogram</summary>

| Number of alt-text tokens | Frequency |
| ------------------------- | --------- |
| 0                         | 600000    |
| 5                         | 650000    |
| 10                        | 550000    |
| 15                        | 450000    |
| 20                        | 350000    |
| 25                        | 250000    |
| 30                        | 200000    |
| 35                        | 150000    |
| 40                        | 100000    |
| 45                        | 80000     |
| 50                        | 60000     |
| 55                        | 40000     |
| 60                        | 30000     |
| 65                        | 20000     |
| 70                        | 15000     |
| 75                        | 10000     |
</details>

Figure 11: Distribution of token length for alt-text in the small COMMONPOOL. The CLIP BPE tokenizer is used for tokenization.

![](images/fabefd31b828d528992cf582327e2b51d2cc3936dd33c64cc8808c8d78f0a443.jpg)

<details>
<summary>bar</summary>

| Detected language | Count   |
| ----------------- | ------- |
| en                | ~3.5e6  |
| ru                | ~2.8e6  |
| ja                | ~2.5e6  |
| de                | ~2.3e6  |
| fr                | ~2.1e6  |
| es                | ~1.9e6  |
| zh                | ~1.7e6  |
| it                | ~1.5e6  |
| pt                | ~1.3e6  |
| nl                | ~1.1e6  |
| pl                | ~9.5e5  |
| sv                | ~8.5e5  |
| he                | ~7.5e5  |
| vi                | ~7.0e5  |
| ar                | ~6.5e5  |
| cs                | ~6.0e5  |
| tr                | ~5.5e5  |
| hu                | ~5.0e5  |
| uk                | ~4.5e5  |
| fa                | ~4.0e5  |
| ko                | ~3.5e5  |
| id                | ~3.0e5  |
| el                | ~2.5e5  |
| ro                | ~2.0e5  |
| ca                | ~1.5e5  |
</details>

![](images/ceb9061c17d95ce90eaa28cdc017d4118a7c5582e50f73e0c47f563c9c3824d6.jpg)

<details>
<summary>bar</summary>

| Detected language | Count     |
| ----------------- | --------- |
| en                | ~1.5e6    |
| ja                | ~1.2e6    |
| zh                | ~1.0e6    |
| fr                | ~9.5e5    |
| de                | ~8.5e5    |
| ru                | ~7.5e5    |
| it                | ~6.5e5    |
| es                | ~5.5e5    |
| no                | ~4.5e5    |
| nl                | ~4.0e5    |
| pt                | ~3.5e5    |
| la                | ~3.0e5    |
| fy                | ~2.5e5    |
| af                | ~2.0e5    |
| lb                | ~1.8e5    |
| sr                | ~1.6e5    |
| da                | ~1.4e5    |
| pl                | ~1.2e5    |
| ca                | ~1.0e5    |
| gl                | ~8.0e4    |
| mt                | ~6.0e4    |
| cy                | ~4.0e4    |
| sv                | ~3.0e4    |
| ku                | ~2.0e4    |
| xh                | ~1.0e4    |
</details>

Figure 12: Counts for the top 25 most frequent languages in the small COMMONPOOL, as predicted by fasttext (left) and cld3 (right).

![](images/618a554b415a0d8c2a84ce053162b03cdc6086745b1ef3b9e31c661bcb88583b.jpg)

<details>
<summary>bar</summary>

| Domain | Count |
| :--- | :--- |
| wp.com | 320000 |
| pinimg.com | 315000 |
| ebayimg.com | 280000 |
| cloudfront.net | 260000 |
| wordpress.com | 250000 |
| wixstatic.com | 230000 |
| made-in-china.com | 210000 |
| ssl-images-amazon.com | 190000 |
| amazonaws.com | 170000 |
| alicdn.com | 160000 |
| gstatic.com | 140000 |
| fc2.com | 130000 |
| media-amazon.com | 120000 |
| gravatar.com | 110000 |
| ytimg.com | 105000 |
| tripadvisor.com | 100000 |
| ebaystatic.com | 95000 |
| bing.net | 90000 |
| exblog.jp | 85000 |
| dreamstime.com | 80000 |
| fastly.net | 75000 |
| googleusercontent.com | 70000 |
| blogspot.com | 65000 |
| nocookie.net | 60000 |
| specsserver.com | 55000 |
</details>

![](images/3df9550c21ca71aa99e0c6898b3bb2c84d49c474ad5ea63f9f89462a7f229c8d.jpg)

<details>
<summary>bar</summary>

| Suffix | Count |
|---|---|
| com | 3500000 |
| net | 1000000 |
| ru | 800000 |
| jp | 700000 |
| de | 600000 |
| org | 500000 |
| co.uk | 450000 |
| fr | 400000 |
| com.br | 380000 |
| it | 350000 |
| pl | 320000 |
| nl | 300000 |
| cz | 280000 |
| cn | 250000 |
| co.jp | 220000 |
| es | 200000 |
| hu | 180000 |
| com.au | 160000 |
| vn | 150000 |
| ro | 140000 |
| eu | 130000 |
| ca | 120000 |
| io | 110000 |
| com.ua | 100000 |
| tw | 90000 |
</details>

Figure 13: Counts for the top 25 most frequent domains (left) and suffixes (right) in the small COMMONPOOL.

# J Efficient training on data subsets

When training at large scale, it is important to use efficient access patterns to load training data. This typically means that data must be loaded using large sequential reads instead of random reads in order to maximize throughput. In DATACOMP, this is facilitated by the WebDataset $^{5}$ format which stores the training examples in tar files (called “shards”) and WebDataLoader which makes it easy to load data stored in this format.

Given an arbitrary subset of a pool, we would like to efficiently train on that subset. Because WebDataset format does not permit efficient random access (a feature inherited from tar), we must read through the entire pool to select the required images. There are two ways to implement this filtering:

1. Filter during training: we apply a predicate during training data loading that discards data not present in the subset.   
2. Filter before training: we iterate over the pool, selecting the images in the subset, and write them to a new WebDataset.

After some profiling, we concluded that option 1 had too much overhead in the case where the subset is much smaller than the pool. To see why, note that if the subset is an p-fraction of the pool size, then we would end up reading a 1/p factor more data than needed for training. Instead, we give an implementation of option 2, which performs at most twice as many reads as needed for training. $^{6}$

Our tool, called the resharder, reads a set of uids in NumPy array format, scans through the pool, selecting those examples, and writes them to a new WebDataset. The resharder uses multiprocessing to make good use of hardware and can be distributed over many computers to further increase throughput. The resharder also supports streaming data to and from cloud storage such as Amazon S3. The resharder is provided to participants as part of the competition tooling.

# K Effect of duplicates in the training data

Given that COMMONPOOL was constructed by scraping the web for image and text pairs, there is a likelihood that some of our images are duplicates of each other, even if they originated from different web sources and have different captions. Here we examine the effect of removing such duplicates. We used the technique proposed by Webster et al. [144], where CLIP image features are first compressed and then used to do an approximate nearest neighbor search. After this process, two images $x$ and $y$ are considered duplicates if $\frac{|d_{ADC}(x,x) - d_{ADC}(x,y)|}{d_{ADC}(x,x)} < T_{ADC}$ , where $T_{ADC}$ is some threshold and $d_{ADC}(x,x)$ is the distance of a vector with its quantized version used for approximate nearest neighbor search. For each image, we search duplicates across its 1000 nearest neighbors, and keep it if it's the one with the highest CLIP ViT-L/14 similarity score across its duplicates. Results can be seen in Table 9, both when this technique is used by itself and in conjunction with ViT-B/32 filtering. We can see that results are similar to when only using CLIP filtering.

Table 9: Effect of deduplication of training set for the medium size COMMONPOOL. The filtering performed here is CLIP B32 score top 30% (see Table 26). Higher threshold values lead to more samples being labeled as duplicates. 

<table><tr><td>Subset</td><td>Training dataset size</td><td>ImageNet accuracy</td><td>Average performance</td></tr><tr><td> $T_{ADC} = 0.1$ , without filtering</td><td>99.8M</td><td>0.195</td><td>0.275</td></tr><tr><td> $T_{ADC} = 0.2$ , without filtering</td><td>85.9M</td><td>0.200</td><td>0.277</td></tr><tr><td> $T_{ADC} = 0.5$ , without filtering</td><td>29.6M</td><td>0.227</td><td>0.295</td></tr><tr><td> $T_{ADC} = 0.1$ , with filtering</td><td>33.5M</td><td>0.288</td><td>0.337</td></tr><tr><td> $T_{ADC} = 0.2$ , with filtering</td><td>30.6M</td><td>0.289</td><td>0.337</td></tr><tr><td> $T_{ADC} = 0.5$ , with filtering</td><td>15.5M</td><td>0.252</td><td>0.311</td></tr></table>

Table 10: Batch size ablation at the medium scale. We compare the standard DATACOMP medium configuration, with batch size 4096 against an ablated configuration with batch size 8192 (medium: batch size 2x). We find that the rankings of the baseline filtering strategies are relatively consistent. More precisely, the rank correlation is 0.96 on ImageNet and 0.98 for the Average over 38 datasets. 

<table><tr><td>Scale</td><td>Filtering strategy</td><td>Dataset size</td><td>Samples seen</td><td>ImageNet</td><td>Average over 38 datasets</td><td>Delta ranking ImageNet</td><td>Delta ranking Average</td></tr><tr><td rowspan="7">medium</td><td>No filtering</td><td>128M</td><td>128M</td><td>0.176</td><td>0.258</td><td>-</td><td>-</td></tr><tr><td>Basic filtering</td><td>30M</td><td>128M</td><td>0.226</td><td>0.285</td><td>-</td><td>-</td></tr><tr><td>Text-based</td><td>31M</td><td>128M</td><td>0.255</td><td>0.307</td><td>-</td><td>-</td></tr><tr><td>Image-based</td><td>29M</td><td>128M</td><td>0.268</td><td>0.312</td><td>-</td><td>-</td></tr><tr><td>LAION-2B filtering</td><td>13M</td><td>128M</td><td>0.230</td><td>0.292</td><td>-</td><td>-</td></tr><tr><td>CLIP score (L/14 30%)</td><td>38M</td><td>128M</td><td>0.273</td><td> $\underline{0.328}$ </td><td>-</td><td>-</td></tr><tr><td>Image-based  $\cap$  CLIP score (L/14 30%)</td><td>14M</td><td>128M</td><td> $\underline{0.297}$ </td><td> $\underline{0.328}$ </td><td>-</td><td>-</td></tr><tr><td rowspan="7">medium: batch size 2x</td><td>No filtering</td><td>128M</td><td>128M</td><td>0.171</td><td>0.258</td><td>0</td><td>0</td></tr><tr><td>Basic filtering</td><td>30M</td><td>128M</td><td>0.219</td><td>0.277</td><td>+1 (worse)</td><td>0</td></tr><tr><td>Text-based</td><td>31M</td><td>128M</td><td>0.251</td><td>0.299</td><td>0</td><td>-1 (better)</td></tr><tr><td>Image-based</td><td>29M</td><td>128M</td><td>0.260</td><td>0.299</td><td>0</td><td>0</td></tr><tr><td>LAION-2B filtering</td><td>13M</td><td>128M</td><td>0.215</td><td>0.288</td><td>-1 (better)</td><td>0</td></tr><tr><td>CLIP score (L/14 30%)</td><td>38M</td><td>128M</td><td>0.271</td><td> $\underline{0.324}$ </td><td>0</td><td>0</td></tr><tr><td>Image-based  $\cap$  CLIP score (L/14 30%)</td><td>14M</td><td>128M</td><td> $\underline{0.276}$ </td><td>0.311</td><td>0</td><td>+1 (worse)</td></tr></table>

# L Hyperparameter ablations

Recall that in DATACOMP, we freeze the training procedure and hyperparameters to focus the competition on dataset curation. However, this leads to the natural question: do “better” datasets (i.e., datasets that lead to higher accuracy models on zero-shot downstream tasks) remain consistent when training is modified. Hence we ablate key experimental choices: batch size, model architecture, and number of training steps.

# L.1 Batch size

We ablate over the batch size hyperparameter, doubling the batch size at the medium scale, but holding all other hyperparameters constant. As see in Table 10, we find that the delta rankings are largely consistent, for both ImageNet and Average performance, with rankings changing by at most plus or minus one position. More specifically, rank correlation before and after doubling batch size is 0.96 for ImageNet and 0.98 for the Average over 38 datasets metric.

# L.2 Model architecture

We choose to use the ViT architecture $[39]$ because of favorable CLIP scaling trends over vanilla ResNets $[62]$ as reported by Radford et al. $[111]$ . However, we still hope that better datasets for downstream ViT performance will lead to better datasets to train convolutional architectures. We look at the medium scale, swapping the ViT-B/32 architecture with a ConvNeXt model $[93]$ with matched giga multiplier–accumulate operations (GMACs). Looking at Table 11, we see that ranking of different filtering methods is again relatively consistent (i.e., 1.0 rank correlation for ImageNet and 0.87 rank correlation for the average metric). We conclude that improvements in dataset filtering have potential to improve more than just CLIP ViT model performance.

# L.3 Number of training steps

Recall that one of our major design decisions for DATACOMP is to fix the hyperparameters associated with model training, following closely hyperparameters from prior work $[111]$ . We choose to fix hyperparameters to place emphasis on data curation and remove confounders arising from hyperparameter differences between participants. Here we ablate our hyperparameter configuration by training small baselines for $10\times$ more steps. In Figure 14 we see positive correlation for ImageNet accuracy for the ablated and original hyperparameter configurations. We see similar correlation for average performance. See Table 12 for specific values.

Table 11: Architure ablation at the medium scale. We compare the standard DATACOMP medium configuration, with a ViT-B/32 model against an ablated configuration (medium: ConvNeXt), which uses a ConvNeXt model with the same number of multiply-accumulate operations as the ViT. We find that the rankings of the baseline filtering strategies are relatively consistent. More precisely, the rank correlation is 1.0 on ImageNet and 0.87 for the Average over 38 datasets. 

<table><tr><td>Scale</td><td>Filtering strategy</td><td>Dataset size</td><td>Samples seen</td><td>ImageNet</td><td>Average over 38 datasets</td><td>Delta ranking ImageNet</td><td>Delta ranking Average</td></tr><tr><td rowspan="7">medium</td><td>No filtering</td><td>128M</td><td>128M</td><td>0.176</td><td>0.254</td><td>-</td><td>-</td></tr><tr><td>Basic filtering</td><td>30M</td><td>128M</td><td>0.226</td><td>0.280</td><td>-</td><td>-</td></tr><tr><td>Text-based</td><td>31M</td><td>128M</td><td>0.255</td><td>0.301</td><td>-</td><td>-</td></tr><tr><td>Image-based</td><td>29M</td><td>128M</td><td>0.268</td><td>0.307</td><td>-</td><td>-</td></tr><tr><td>LAION-2B filtering</td><td>13M</td><td>128M</td><td>0.230</td><td>0.287</td><td>-</td><td>-</td></tr><tr><td>CLIP score (L/14 30%)</td><td>38M</td><td>128M</td><td>0.273</td><td>0.323</td><td>-</td><td>-</td></tr><tr><td>Image-based ∩ CLIP score (L/14 30%)</td><td>14M</td><td>128M</td><td>0.297</td><td>0.323</td><td>-</td><td>-</td></tr><tr><td rowspan="7">medium: ConvNeXt</td><td>No filtering</td><td>128M</td><td>128M</td><td>0.178</td><td>0.255</td><td>0</td><td>0</td></tr><tr><td>Basic filtering</td><td>30M</td><td>128M</td><td>0.232</td><td>0.272</td><td>0</td><td>0</td></tr><tr><td>Text-based</td><td>31M</td><td>128M</td><td>0.255</td><td>0.298</td><td>0</td><td>0</td></tr><tr><td>Image-based</td><td>29M</td><td>128M</td><td>0.270</td><td>0.298</td><td>0</td><td>+1 (better)</td></tr><tr><td>LAION-2B filtering</td><td>13M</td><td>128M</td><td>0.253</td><td>0.300</td><td>0</td><td>-2 (better)</td></tr><tr><td>CLIP score (L/14 30%)</td><td>38M</td><td>128M</td><td>0.279</td><td>0.326</td><td>0</td><td>+1 (worse)</td></tr><tr><td>Image-based ∩ CLIP score (L/14 30%)</td><td>14M</td><td>128M</td><td>0.323</td><td>0.331</td><td>0</td><td>0</td></tr></table>

![](images/fed5a3555d8d05c784fb4fa9afca60212cc028305c0c21abb37186581cb9b989.jpg)

Basic

× CLIP score

Image-based

No filtering

Rand. subset

\+ Text-based

Figure 14: (left) The effect of training for $10\times$ steps for small filtering track baselines on ImageNet. (right) Similar plot but for Avg. performance. While the ordering of some methods changes quite drastically, we, in general, see a positive correlation.

Table 12: Experiment details when extending the number of steps by 10 times the standard amount for that scale. 

<table><tr><td>Scale</td><td>Filtering</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td rowspan="36">small</td><td>No filtering</td><td>0.102</td><td>0.093</td><td>0.204</td><td>0.147</td><td>0.196</td></tr><tr><td>Random subset(75%)</td><td>0.078</td><td>0.072</td><td>0.182</td><td>0.129</td><td>0.178</td></tr><tr><td>Random subset(50%)</td><td>0.045</td><td>0.049</td><td>0.161</td><td>0.104</td><td>0.150</td></tr><tr><td>Random subset(25%)</td><td>0.023</td><td>0.029</td><td>0.134</td><td>0.075</td><td>0.119</td></tr><tr><td>Random subset(10%)</td><td>0.010</td><td>0.018</td><td>0.119</td><td>0.069</td><td>0.101</td></tr><tr><td>Random subset(1%)</td><td>0.002</td><td>0.006</td><td>0.097</td><td>0.056</td><td>0.082</td></tr><tr><td>Caption length</td><td>0.085</td><td>0.080</td><td>0.198</td><td>0.136</td><td>0.184</td></tr><tr><td>Image size</td><td>0.066</td><td>0.064</td><td>0.153</td><td>0.115</td><td>0.158</td></tr><tr><td>English (fasttext)</td><td>0.068</td><td>0.068</td><td>0.172</td><td>0.108</td><td>0.159</td></tr><tr><td>English (fasttext) and caption length</td><td>0.066</td><td>0.065</td><td>0.182</td><td>0.106</td><td>0.163</td></tr><tr><td>English (fasttext), caption length, and image size</td><td>0.045</td><td>0.048</td><td>0.164</td><td>0.092</td><td>0.149</td></tr><tr><td>CLIP B32 score top 10%</td><td>0.035</td><td>0.046</td><td>0.162</td><td>0.079</td><td>0.139</td></tr><tr><td>CLIP B32 score top 20%</td><td>0.076</td><td>0.076</td><td>0.182</td><td>0.099</td><td>0.172</td></tr><tr><td>CLIP B32 score top 30%</td><td>0.096</td><td>0.090</td><td>0.221</td><td>0.121</td><td>0.205</td></tr><tr><td>CLIP B32 score top 40%</td><td>0.081</td><td>0.077</td><td>0.200</td><td>0.124</td><td>0.193</td></tr><tr><td>CLIP B32 score top 50%</td><td>0.106</td><td>0.097</td><td>0.211</td><td>0.134</td><td>0.205</td></tr><tr><td>CLIP B32 score top 75%</td><td>0.103</td><td>0.096</td><td>0.210</td><td>0.150</td><td>0.198</td></tr><tr><td>CLIP B32 score top 90%</td><td>0.105</td><td>0.096</td><td>0.212</td><td>0.152</td><td>0.202</td></tr><tr><td>CLIP B32 threshold at 0.3 + English filter</td><td>0.029</td><td>0.036</td><td>0.152</td><td>0.078</td><td>0.134</td></tr><tr><td>CLIP B32 threshold at 0.28 + English filter</td><td>0.035</td><td>0.041</td><td>0.168</td><td>0.086</td><td>0.145</td></tr><tr><td>CLIP B32 threshold at 0.3</td><td>0.076</td><td>0.078</td><td>0.199</td><td>0.102</td><td>0.182</td></tr><tr><td>CLIP L14 score top 10%</td><td>0.026</td><td>0.037</td><td>0.130</td><td>0.073</td><td>0.123</td></tr><tr><td>CLIP L14 score top 20%</td><td>0.060</td><td>0.064</td><td>0.161</td><td>0.096</td><td>0.153</td></tr><tr><td>CLIP L14 score top 30%</td><td>0.088</td><td>0.087</td><td>0.199</td><td>0.115</td><td>0.188</td></tr><tr><td>CLIP L14 score top 40%</td><td>0.100</td><td>0.096</td><td>0.217</td><td>0.122</td><td>0.207</td></tr><tr><td>CLIP L14 score top 50%</td><td>0.104</td><td>0.098</td><td>0.212</td><td>0.136</td><td>0.203</td></tr><tr><td>CLIP L14 score top 75%</td><td>0.103</td><td>0.095</td><td>0.189</td><td>0.146</td><td>0.191</td></tr><tr><td>CLIP L14 score top 90%</td><td>0.105</td><td>0.095</td><td>0.203</td><td>0.145</td><td>0.198</td></tr><tr><td>Image-based clustering (ImageNet1k)</td><td>0.053</td><td>0.053</td><td>0.162</td><td>0.091</td><td>0.146</td></tr><tr><td>Image-based clustering (ImageNet21k)</td><td>0.063</td><td>0.059</td><td>0.173</td><td>0.108</td><td>0.167</td></tr><tr><td>Text-based clustering (ImageNet1k)</td><td>0.012</td><td>0.018</td><td>0.120</td><td>0.062</td><td>0.104</td></tr><tr><td>Text-based clustering (ImageNet21k)</td><td>0.262</td><td>0.216</td><td>0.305</td><td>0.246</td><td>0.300</td></tr><tr><td>Intersect IN1k image clustering and CLIP B32 score top 30%</td><td>0.058</td><td>0.059</td><td>0.179</td><td>0.098</td><td>0.161</td></tr><tr><td>Intersect IN1k image clustering and CLIP L14 score top 30%</td><td>0.049</td><td>0.051</td><td>0.171</td><td>0.090</td><td>0.150</td></tr><tr><td>Intersect IN21k image clustering and CLIP B32 score top 30%</td><td>0.071</td><td>0.070</td><td>0.192</td><td>0.107</td><td>0.175</td></tr><tr><td>Intersect IN21k image clustering and CLIP L14 score top 30%</td><td>0.064</td><td>0.065</td><td>0.200</td><td>0.096</td><td>0.173</td></tr><tr><td rowspan="10">medium</td><td>No filtering</td><td>0.370</td><td>0.304</td><td>0.387</td><td>0.355</td><td>0.383</td></tr><tr><td>English (fasttext), caption length, and image size</td><td>0.317</td><td>0.269</td><td>0.324</td><td>0.271</td><td>0.334</td></tr><tr><td>CLIP B32 score top 30%</td><td>0.436</td><td>0.351</td><td>0.433</td><td>0.345</td><td>0.430</td></tr><tr><td>CLIP B32 score top 40%</td><td>0.434</td><td>0.353</td><td>0.448</td><td>0.365</td><td>0.442</td></tr><tr><td>CLIP B32 score top 50%</td><td>0.426</td><td>0.352</td><td>0.439</td><td>0.377</td><td>0.433</td></tr><tr><td>CLIP B32 score top 75%</td><td>0.398</td><td>0.325</td><td>0.396</td><td>0.374</td><td>0.411</td></tr><tr><td>Image-based clustering (ImageNet1k)</td><td>0.363</td><td>0.294</td><td>0.347</td><td>0.279</td><td>0.347</td></tr><tr><td>Image-based clustering (ImageNet21k)</td><td>0.374</td><td>0.303</td><td>0.372</td><td>0.318</td><td>0.372</td></tr><tr><td>Intersect IN1k image clustering and CLIP B32 score top 30%</td><td>0.415</td><td>0.330</td><td>0.413</td><td>0.310</td><td>0.403</td></tr><tr><td>Intersect IN1k image clustering and CLIP L14 score top 30%</td><td>0.405</td><td>0.325</td><td>0.399</td><td>0.295</td><td>0.387</td></tr></table>

# M Detector-based baselines

While controlling for factors such as class balance is common in the supervised settings, experimenting with analogous strategies in the context of multimodal datasets and CLIP training is a pertinent direction. Towards this end, we use the Detic detector $[160]$ to annotate the medium pool (128M samples) by extracting bounding boxes and class labels for the 1203 LVIS $[54]$ objects categories. Following the original Detic paper, we retain predictions whose confidence score exceeds 0.5. Based on these annotations, we construct the following five strategies:

- Object exists: Subset for which there exists at least one detection from the 1203 LVIS categories.   
- Object centered: Subset for which there exists at least one detection from the 1203 LVIS categories with a bounding box center falling in the center grid cell of a 3x3 grid superimposed on the image.   
- Balancing by class: We define 1204 buckets—1203 buckets corresponding to the LVIS classes and an additional bucket for images that do not have any detections. For each image in the medium pool, we assign the image to the bucket(s) corresponding to the detected classes. We then construct a dataset such that there are an equal number of samples from each bucket and the total number of samples specified by a particular scale (e.g., 128M samples for medium scale). Note, for rare classes there can be many repeated samples and for common classes only a subset of the total samples will be in the dataset.   
- Balancing by position: We define 26 buckets—0, 1, ..., 24 corresponding to 5x5 grid locations in an image. An image is added to bucket(s) when it contains a bounding box whose center falls in the bucket's grid cell. The 25th bucket contains images for which there are no detections. We again construct a dataset such that there are an equal number of samples from each bucket.   
- Balancing by count: We define 12 buckets—0, 1, ..., 10 corresponding to zero to ten detections in an image and a twelfth bucket corresponding to images with more than ten detections. We yet again construct a dataset such that there are an equal number of samples from each bucket.

We employ each of these strategies on the medium scale. Since the above strategies can be composed with any starting pool, we additionally apply each of the above Detic-based strategies to our previous best medium scale filtered pool: Image-based $\cap$ CLIP score (L/14 30%). This yields five more datasets for 10 baselines in total.

Our results are summarized in the Table 13. In summary: 1) The Image-based $\cap$ CLIP score (L/14 $30\%$ ) baseline still performs best. 2) Balancing data in the context of multimodal CLIP training

Table 13: Detector-based baselines at the medium scale. We start with No filtering and Image-based cap CLIP score (L/14 30%) pools and apply five additional filtering and balancing strategies described in Appendix M. We find that even with these more sophisticated strategies, the No filtering and Image-based cap CLIP score (L/14 30%) still performs best at medium scale. Properly balancing multimodal data remains an open direction for future work. 

<table><tr><td>Scale</td><td>Filtering strategy</td><td>Samples seen</td><td>ImageNet</td><td>Average over 38 datasets</td></tr><tr><td rowspan="6">medium</td><td>No filtering</td><td>128M</td><td>0.176</td><td>0.258</td></tr><tr><td>∩ Object exists</td><td>128M</td><td>0.181</td><td>0.263</td></tr><tr><td>∩ Object centered</td><td>128M</td><td>0.187</td><td>0.263</td></tr><tr><td>∩ Balance by class</td><td>128M</td><td>0.038</td><td>0.141</td></tr><tr><td>∩ Balance by position</td><td>128M</td><td>0.040</td><td>0.148</td></tr><tr><td>∩ Balance by object count</td><td>128M</td><td>0.127</td><td>0.221</td></tr><tr><td rowspan="6">medium</td><td>Image-based ∩ CLIP score (L/14 30%)</td><td>128M</td><td>0.297</td><td>0.328</td></tr><tr><td>∩ Object exists</td><td>128M</td><td>0.289</td><td>0.319</td></tr><tr><td>∩ Object centered</td><td>128M</td><td>0.247</td><td>0.286</td></tr><tr><td>∩ Balance by class</td><td>128M</td><td>0.034</td><td>0.136</td></tr><tr><td>∩ Balance by position</td><td>128M</td><td>0.036</td><td>0.136</td></tr><tr><td>∩ Balance by object count</td><td>128M</td><td>0.068</td><td>0.169</td></tr></table>

Table 14: Experimental configuration for each scale, including the size of the pool we provide, the model architecture and hyperparameters. 

<table><tr><td>Scale</td><td>Model</td><td>Train compute (MACs)</td><td>Pool size</td><td># samples seen</td><td>Learning rate</td><td>AdamW  $\beta_{2}$ </td><td>Warmup</td><td>Batch size</td></tr><tr><td>small</td><td>ViT-B/32</td><td> $9.5 \times 10^{16}$ </td><td>12.8M</td><td>12.8M</td><td>5e-4</td><td>0.98</td><td>500</td><td>4096</td></tr><tr><td>medium</td><td>ViT-B/32</td><td> $9.5 \times 10^{17}$ </td><td>128M</td><td>128M</td><td>5e-4</td><td>0.98</td><td>500</td><td>4096</td></tr><tr><td>large</td><td>ViT-B/16</td><td> $2.6 \times 10^{19}$ </td><td>1.28B</td><td>1.28B</td><td>5e-4</td><td>0.98</td><td>500</td><td>8192</td></tr><tr><td>xlarge</td><td>ViT-L/14</td><td> $1.1 \times 10^{21}$ </td><td>12.8B</td><td>12.8B</td><td>1e-3</td><td>0.95</td><td>10k</td><td>90112</td></tr></table>

remains an open problem. All balancing strategies lead to divergence of the CLIP contrastive loss and result in poor model performance. We hypothesize that this is due to the long-tailed nature of the data distribution, which leads to many repeated samples in our balanced data construction. This in turn, increases the likelihood that samples are contrasted with themselves in the loss computation.

# N Training details

The full set of hyperparameters used for each scale is shown in Table 14. For choosing hyperparameters, we follow the OpenCLIP library [69], an open source reproduction of OpenAI's CLIP. For the small, medium, and large tracks, these hyperparameters are equal to those in the CLIP paper, except with reduced batch size so that training runs on reasonable hardware. For the xlarge track, batch size is increased from that in OpenAI's CLIP to accelerate training by allowing the use of many GPUs simultaneously with high utilization. For this run we also double the learning rate following prior work [28].

# O Evaluation details

Models are evaluated over a wide range of 38 tasks to measure proficiency in various domains. We include 22 of the 27 classification tasks in the test suite of Radford et al. [111], excluding the few datasets that have license restrictions, are in video format, or are no longer available in their original form. We include 6 datasets that were designed to test generalization of models trained on ImageNet. We also include a majority of the Visual Task Adaptation Benchmark, excluding 3 datasets that are ill-suited for zero-shot evaluation [156]. We include 3 datasets from the WILDS benchmark, which tests robustness to distribution shifts and spurious correlations [83, 127]. Finally, we include 2 additional datasets, Dollar Street and GeoDE, which test robustness of classification performance across income levels and geographical regions [122, 114]. Furthermore, we evaluate zero-shot image and text retrieval on the Flickr30k and MSCOCO datasets, and image association on the WinoGAViL dataset [151, 26, 17]. The complete list of evaluation tasks is given in Table 15. We show a sample from each dataset in Figure 15.

Prompt choice. Since we perform zero-shot evaluation, prompt and class name selection is important, and can have a significant impact on the results. To avoid heavy prompt engineering and overturning to individual models, we opt to use the prompt templates used in Radford et al. $[111]$ whenever possible. Most datasets come with pre-defined class names, but some are overwritten with more descriptive labels, again based on previous literature. For datasets with no precedent in zero-shot evaluation, we reuse prompt templates from other datasets with a similar domain and task (e.g., SVHN is evaluated with MNIST prompts and class names).

Evaluation metrics. For the majority of classification tasks, the primary evaluation metric is accuracy. For certain datasets with class imbalances, we instead compute mean per-class accuracy, as done in Radford et al. $[111]$ . On the WILDS benchmark datasets, we use the primary metric specified for each dataset on their leaderboard. Dollar Street and GeoDE test model generalization across socioeconomic and geographic diversity. Thus, for Dollar Street, we compute worst-group top-5 accuracy, with groups defined by income level, emulating Rojas et al. $[122]$ ; for GeoDE, we compute worst-group accuracy, with groups defined by region (Africa, Americas, West Asia, East Asia, Southeast Asia, and Europe), as defined in Ramaswamy et al. $[114]$ . For the image-text retrieval tasks, Flickr and MSCOCO, we compute both image and text recall (fraction of text captions for which the correct image was selected and vice versa), and plot their arithmetic mean. On WinoGAViL, we compute the

Table 15: Evaluation tasks. 

<table><tr><td>Task type</td><td>Dataset</td><td>Task</td><td>Test set size</td><td>Number of classes</td><td>Main metric</td><td>Clean</td></tr><tr><td rowspan="35">Classification</td><td>Caltech-101 [45]</td><td>Object recognition</td><td>6,085</td><td>102</td><td>mean per class</td><td>√</td></tr><tr><td>CIFAR-10 [86]</td><td>Visual recognition</td><td>10,000</td><td>10</td><td>accuracy</td><td>√</td></tr><tr><td>CIFAR-100 [86]</td><td>Visual recognition</td><td>10,000</td><td>100</td><td>accuracy</td><td>√</td></tr><tr><td>CLEVR Counts [76, 156]</td><td>Counting</td><td>15,000</td><td>8</td><td>accuracy</td><td></td></tr><tr><td>CLEVR Distance [76, 156]</td><td>Distance prediction</td><td>15,000</td><td>6</td><td>accuracy</td><td></td></tr><tr><td>Country211 [111, 140]</td><td>Geolocation</td><td>21,100</td><td>211</td><td>accuracy</td><td>√</td></tr><tr><td>DTD [30]</td><td>Texture classification</td><td>1,880</td><td>47</td><td>accuracy</td><td>√</td></tr><tr><td>EuroSAT [63, 156]</td><td>Satellite imagery recognition</td><td>5,400</td><td>10</td><td>accuracy</td><td>√</td></tr><tr><td>FGVC Aircraft [95]</td><td>Aircraft recognition</td><td>3,333</td><td>100</td><td>mean per class</td><td>√</td></tr><tr><td>Food-101 [18]</td><td>Food recognition</td><td>25,250</td><td>101</td><td>accuracy</td><td>√</td></tr><tr><td>GTSRB [137]</td><td>Traffic sign recognition</td><td>12,630</td><td>43</td><td>accuracy</td><td>√</td></tr><tr><td>ImageNet 1k [37]</td><td>Visual recognition</td><td>50,000</td><td>1,000</td><td>accuracy</td><td>√</td></tr><tr><td>ImageNet Sketch [143]</td><td>Visual recognition</td><td>50,889</td><td>1,000</td><td>accuracy</td><td>√</td></tr><tr><td>ImageNet V2 [121]</td><td>Visual recognition</td><td>10,000</td><td>1,000</td><td>accuracy</td><td>√</td></tr><tr><td>ImageNet-A [65]</td><td>Visual recognition</td><td>7,500</td><td>200</td><td>accuracy</td><td>√</td></tr><tr><td>ImageNet-O [65]</td><td>Visual recognition</td><td>2,000</td><td>200</td><td>accuracy</td><td>√</td></tr><tr><td>ImageNet-R [64]</td><td>Visual recognition</td><td>30,000</td><td>200</td><td>accuracy</td><td>√</td></tr><tr><td>KITTI distance [48, 156]</td><td>Distance prediction</td><td>711</td><td>4</td><td>accuracy</td><td></td></tr><tr><td>MNIST [89]</td><td>Digit recognition</td><td>10,000</td><td>10</td><td>accuracy</td><td>√</td></tr><tr><td>ObjectNet [13]</td><td>Visual recognition</td><td>18,574</td><td>113</td><td>accuracy</td><td>√</td></tr><tr><td>Oxford Flowers-102 [102]</td><td>Flower recognition</td><td>6,149</td><td>102</td><td>mean per class</td><td>√</td></tr><tr><td>Oxford-IIIT Pet [105, 156]</td><td>Pet classification</td><td>3,669</td><td>37</td><td>mean per class</td><td>√</td></tr><tr><td>Pascal VOC 2007 [42]</td><td>Object recognition</td><td>14,976</td><td>20</td><td>accuracy</td><td>√</td></tr><tr><td>PatchCamelyon [142, 156]</td><td>Metastatic tissue cls.</td><td>32,768</td><td>2</td><td>accuracy</td><td></td></tr><tr><td>Rendered SST2 [156]</td><td>Sentiment classification</td><td>1,821</td><td>2</td><td>accuracy</td><td>√</td></tr><tr><td>RESISC45 [27, 156]</td><td>Satellite imagery recognition</td><td>6,300</td><td>45</td><td>accuracy</td><td>√</td></tr><tr><td>Stanford Cars [85]</td><td>Vehicle recognition</td><td>8,041</td><td>196</td><td>accuracy</td><td>√</td></tr><tr><td>STL-10 [31]</td><td>Visual recognition</td><td>8,000</td><td>10</td><td>accuracy</td><td>√</td></tr><tr><td>SUN-397 [146]</td><td>Scene recognition</td><td>108,754</td><td>397</td><td>accuracy</td><td>√</td></tr><tr><td>SVHN [99, 156]</td><td>Digit recognition</td><td>26032</td><td>10</td><td>accuracy</td><td>√</td></tr><tr><td>iWildCam [14, 83]</td><td>Animal recognition</td><td>42,791</td><td>182</td><td>macro F1 score</td><td>√</td></tr><tr><td>Camelyon17 [12, 83]</td><td>Metastatic tissue cls.</td><td>85,054</td><td>2</td><td>accuracy</td><td></td></tr><tr><td>FMoW [29, 83]</td><td>Satellite imagery recognition</td><td>22,108</td><td>62</td><td>worst-region acc.</td><td>√</td></tr><tr><td>Dollar Street [122]</td><td>Object recognition</td><td>3,503</td><td>58</td><td>worst-income top-5 acc.</td><td>√</td></tr><tr><td>GeoDE [114]</td><td>Object recognition</td><td>12,488</td><td>40</td><td>worst-region acc.</td><td>√</td></tr><tr><td rowspan="3">Retrieval</td><td>Flickr30k [151]</td><td>Image and text retrieval</td><td>31,014</td><td>N/A</td><td>R@1</td><td>√</td></tr><tr><td>MSCOCO [26]</td><td>Image and text retrieval</td><td>5,000</td><td>N/A</td><td>R@1</td><td>√</td></tr><tr><td>WinoGAViL [17]</td><td>Commonsense association</td><td>3,563</td><td>N/A</td><td>Jaccard score</td><td>√</td></tr></table>

Jaccard score (intersection-over-union) for each example, and show results for the harder samples (10 and 12 candidates). More information on WinoGAViL evaluation can be found in Bitton et al. [17].

Clean subset. For five of our evaluation tasks (the two CLEVR tasks, the two Camelyon tasks, and KITTI) the zero-shot performance of all evaluated models appears to be close to that of random guessing, and lack correlation to the type of filtering method used (see Figure 27). Consequently, we studied performance averaged only on the remaining 33 tasks, but found not substantial qualitative differences in our results. As a result, we opted to report the average on the full evaluation suite throughout our study.

Zero-shot vs. fine-tuning protocols. One critical decision in DATACOMP is how exactly to evaluate models and whether or not to fine-tune models on evaluation tasks (i.e., supervised fine-tuning directly on task training sets). We opt for zero-shot evaluation, where a models are applied to downstream tasks directly to 1) ease computational burden on participants and 2) measure the out-of-the-box generalization capabilities of our models. To validate this design decision, we conduct linear probes on all models presented in Tables 3 and 18 on ImageNet. We follow a standard probing protocol and fine-tune the last linear layer from zero-shot initialization for 40 epochs with learning rate 1e-3, batch size 256, AdamW optimizer with default settings with the exception of weight decay (that we set to zero), and a cosine annealing schedule. As seen in Figure 16, zero-shot and linear probe performance follow similar trends for both filtering and BYOD tracks. Moreover the Spearman rank correlation between the two protocols over the models considered is 0.99 for the filtering track and 1.0 for BYOD. This suggests that better zero-shot models on ImageNet are correlated with better representations of linear probe fine-tuning on ImageNet.

![](images/4d0eb719d9d69dac548127ce1db4ac4f8c251b879bca727f24b2ed1f2777b54b.jpg)  
Figure 15: Randomly sampled images from the evaluation datasets we consider.

# O.1 Visual Question Answering

In addition to our evaluation suite containing multiple classification and retrieval tasks, we conducted experiments on visual question answering. More specifically, following Shen et al. [132], we use the CLIP models to contrast images with prompts formed by the questions and each candidate answer, without fine-tuning (i.e., in a zero-shot setting). Using the VQA v1 dataset [2], for each candidate answer, we construct a text prompt that also includes the question following the template Question: [question text] Answer: [answer text], as in Ilharco et al. [70]. This text is then fed to CLIP's text encoder. As previously noted by multiple authors, CLIP models struggle on this task, potentially due to the mismatch between the text in the downstream task and the captions seen during pre-training Shen

![](images/9d1a5c8ab1450fc3b0560f150e7d73555b5da70a1c164af9aa45e7b3a0519fb3.jpg)

<details>
<summary>scatter</summary>

| Zero-shot ImageNet | Linear probe ImageNet |
| ------------------ | --------------------- |
| 0.0                | 0.2                   |
| 0.1                | 0.4                   |
| 0.2                | 0.5                   |
| 0.3                | 0.6                   |
| 0.4                | 0.7                   |
| 0.5                | 0.8                   |
| 0.6                | 0.9                   |
| 0.7                | 0.95                  |
| 0.8                | 0.98                  |
</details>

![](images/2f693b7e385cf2bc254f2bc576b66345c8cfabe5e1eb04c26b877ab9c33dfa97.jpg)

<details>
<summary>scatter</summary>

| Zero-shot ImageNet | Linear probe ImageNet |
| ------------------ | --------------------- |
| 0.2                | 0.45                  |
| 0.3                | 0.5                   |
| 0.6                | 0.75                  |
| 0.7                | 0.85                  |
</details>

![](images/ff0fb1bbe51fd4a44ad4d311d4b9e78fe433f5f4022b897043867464059676f8.jpg)

Figure 16: Zero-shot ImageNet and Linear probe ImageNet performance for models from Tables 3 and 18. Relative ordering of models demonstrates high rank correlations of 0.99 and 1.0 for COMMONPOOL and BYOD respectively.   
![](images/40e35a7b1f689b5997b143e608c57f0fe6daa7a833ada5b389224b6d949bff7b.jpg)

<details>
<summary>scatter</summary>

| VQA accuracy | ImageNet accuracy |
| ------------ | ----------------- |
| 0.050        | 0.0               |
| 0.075        | 0.2               |
| 0.100        | 0.8               |
</details>

![](images/b809685d320e2ab38622ce6e78823c175132db8496644c93ad1579a313bce87d.jpg)

<details>
<summary>scatter</summary>

| VQA accuracy | Average accuracy |
| ------------ | ---------------- |
| 0.050        | 0.1              |
| 0.075        | 0.4              |
| 0.100        | 0.6              |
</details>

![](images/637baed030deed4643ce7373840b356b2484dbc582f4efb6a7db649d1c04d680.jpg)

<details>
<summary>text_image</summary>

No filtering
Basic
CLIP score
Image-based
Text-based
Rand. subset
ImageNet dist.
</details>

Figure 17: Correlation between zero-shot performance on the VQA v1 dataset and results on ImageNet and our full evaluation suite.

et al. [132], Ilharco et al. [70], Song et al. [134]. Nonetheless, we observe a strong correlation between VQA performance and ImageNet accuracy (0.877) and between VQA performance and average performance on our full evaluation suite. Full results are shown in Figure 17.

# P Baseline details

Here we provide additional details on the creation of our baseline subsets. To highlight the qualitative differences between the filtering strategies we also provide visualization for No filtering (Figure 18), Basic filtering (Figure 19), and CLIP score (L/14 30%) (Figure 20), which can all be found in Table 3. Notice that No filtering gives relatively noisy data (e.g., matching a bicycle with a caption: "IMG\_2187.jpg"), while CLIP score samples give qualitatively more descriptive cations.

![](images/fa63ef5f86444f508eb7efd868edda3a0e2d2b038a2b6e9c1fa47eb9a9cac522.jpg)

<details>
<summary>natural_image</summary>

Human internal organs and body structure shown in blue, no text or labels present
</details>

Organos muntoriales

![](images/e1f4c0aca5ef16b74a08f77a618e2c1ad7966a7b56ca9a84f53feb9ddaaf1989.jpg)

<details>
<summary>natural_image</summary>

Interior view of a dimly lit cave with intricate rock formations and natural lighting (no text or symbols visible)
</details>

20110531 4665RWw [F]
Grotte des Demoiselles
[Ganges]

![](images/f322ed4796923ab0f311e0930b795c42a5579005d9459340174b6e4e2c973d95.jpg)  
【iPhone6s Plus/6 Plusケース】WiFiブースター
LINKASE クリア with WiFi スペースグレイ iPhone 6s Plus/6 Plus\_0

![](images/131528422c84384ca6ae865b8f428b522bd10e0c255bbc7b45e37c0bc2753ed0.jpg)  
中村不折旧宅（書道博物館）

![](images/ad70543f813400d3318d1af2f525f7679be90718f1e4bd75fc36f6c431fe23d7.jpg)  
IMG\_2187.jpg

![](images/2924441fda9c575cd17ce90a9483cff8f079af7425847cd7df1b5beb21bec9c9.jpg)  
Carregador portátil para Smartphones 5000mAh 5V
2.1A - JS Soluções em Segurança

![](images/1b35fd494fbc6206fb8723af8e39f3045f9b2f7e447e0d7682c4e64049a1f59e.jpg)

<details>
<summary>text_image</summary>

LLORENS
LIBRERIA
Casa Iovada
en 1876
</details>

NACIO EUROPA EN LA EDAD MEDIA

![](images/0b3f7291b473613ce652039366298e95c0ec2a55a1bea92a172a43be59576a56.jpg)  
JUMP LEADS HEAVY DUTY COMMERCIAL 4.5 M 700 AMP

![](images/ec88db0dcfa08f6aea96d92537253626eb9fae06b8beab0155b45a6a1b68295c.jpg)  
Energy Stocks Fuel Market Rally

![](images/1869ddc9f4709c122d3f44d937916af85abd45db33bdcbcd71a523c785e65317.jpg)  
2019-02-08废金属价格相关热搜新闻一览   
Figure 18: An i.i.d. sample from small COMMONPOOL generated after applying the No filter strategy. Hence, these samples represent random images from COMMONPOOL.

![](images/dda36f4de822c7020b3a09d0249243ba7b1cf4fafd11eb6395dcfe29c24408fe.jpg)  
status report templates - 12+ free word documents download | free, Powerpoint templates

![](images/42ef759353ba61548c39e2446dcbbf3b82cfd5bd5c738697afc77140783135ab.jpg)  
Implication in the classroom:
O Step 3: We Must Work it Out
Say and mean "We have to work it out". The behaviour cannot
co...

![](images/836523c271c67ad049ed512d55bbb497e73e1033a7617110ff10f3543839cc23.jpg)  
City 39 mm Quartz

![](images/09222c0dc270285d4a3a749273b60d4435f41d5f235348bacee502e2702733b6.jpg)  
Astro ATA 3050 INSERTION TOOL, 16/20 GA

![](images/16fb0a98771963c988916640b5c1332a005d1b4650f449ea12171324480f59e6.jpg)  
1006: Rookwood pink mat vase, 1929, 2382, 6&quot;

![](images/f764377e905cfdbe83af876d764367b1a1744868ccb8c429ecaa44baecb6c58a.jpg)  
WN | 2 Corinthians
10:1-18 | Meekness or
Boldness?

![](images/99cdcc24924f4843c283816809816911b051ccd631c812e73347c4d203e5595a.jpg)  
Chaussure De Running
Junior Asics Gt-1000 Gs
Bleu/vert - Asics - 37

![](images/d4454cf37d31d857f63042a8bc32851638541bc0ff653623d4e9067ee170e9d6.jpg)  
Shopping Fairy Crochet Pattern, crochet wings, crochet shopping bags, crochet doll

![](images/b12bf5874894c59b3ae37962089fec2fa9f28ade521bff8190941bede579cb19.jpg)  
Luxardo, Maraschino
Cherries, 14 Fl Oz :
Grocery & Gourmet Food

![](images/fcdb7a7473afd48cc74ea9f19c9d500639698739e3c657eba890d4e542c5c267.jpg)  
Essay Outlines Exles by Writing Center Workshops The Outline

Figure 19: An i.i.d. sample from small COMMONPOOL generated after applying the Basic filter strategy.

![](images/a8a1e67fb737339cdb864404a5f02aaecb64f8a38803e9d8b1b24ece621a7179.jpg)

<details>
<summary>text_image</summary>

DIVINE ELEMENTS
of sacred geometry

LICEL
TOMO CIPOL
FIGURENTIC TEMPO
VERIMAN
VERIMAN
THYMEAN
THE CANO OF GRETAIN
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
The BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BENT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF LIME
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BANT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
THE BNT OF SILENE
</details>

Sacred Geometry Egg Of Life, Sacred Geometry Symbols, Golden Ratio, Flower Of Life, Wicca, Magick, Tattoos, Geometric Nature, Geometric Mandala

![](images/60f1f0caf439bf656f9ac8102a816dea2de983a8a73c576b4aba3a6a0986dd7d.jpg)

<details>
<summary>text_image</summary>

MARTIAN
MONSTERS
JAN KENNIGER JACK KAMEN
</details>

The Martian Monster And Other Stories (The EC Comics Library) ()

![](images/be21da336f82442855bf80fd82eeb9a8fdf580f9588f52869439dfa9fef12a32.jpg)  
Porsche Cayman S

![](images/83aec9d5d448b488b23283a44a179d4a8fc9e575dc8ec206ad09366ce6e006e7.jpg)  
Mesmerizing Black, Silver & Pink Handmade Modern Metal Wall Art Sculpture - Metallic One of a Kind Abstract Painting - OOAK 546 by Jon Allen

![](images/8a295462dd8452fd5dfbc1b4a406ddd324399fee6f6e615b22a4e1fa9baf419c.jpg)

<details>
<summary>natural_image</summary>

Person wearing dark blue shorts and black sneakers, standing with legs (no visible text or symbols)
</details>

ALPINESTARS Radar Short navy blue

![](images/18903ae1b82608baf7ece596c8e80d491497bfb4486da6334406fa6c54691029.jpg)  
Under Armour Heatgear
Gotta Have It Shorty -
Women's at Foot Locker

![](images/81264751eda3c5fa75e281ad24545d7be61e9813a153fb5ec2432e865ee226d3.jpg)  
Tripod Delphin TPX3 Silver

![](images/26b4135ccd857fb67dab0ca3d83093627aa4376a7281183c31a3a29271734c30.jpg)  
Football Manager 2016

![](images/2274a5be6514c529168633ec0f9f93120a1111803b644a13594dc6209708a4b6.jpg)  
www.funpot.net
ein Pott voller
Spaß und
Sehenswertem   
Mett.jpg auf
www.funpot.net

![](images/63b4af444af35d7837e83ba500bc229871604c0d6c70e0493ed4c8a3477d5fbb.jpg)  
Profitable forex trading systems

Figure 20: An i.i.d. sample from small COMMONPOOL generated after applying the CLIP score (L/14 30%)

strategy.

# P.1 Filtering track

Basic filtering. For language detection, we use Fasttext 0.92, version lid.176, and cld3 - library gcld3 3.0.13. We count the number of words in each caption by splitting using whitespaces.

CLIP thresholds. We use OpenAI pretrained CLIP ViT-B/32 and ViT-L/14 models $[111]$ to compute the cosine similarity text and image tower outputs as the CLIP scores. On the small and medium pools, we also experiment with baselines that filter out samples in the top few percentiles of CLIP scores. Specifically, we try baselines that use samples with top $\{1,2,5\}-30\%$ CLIP scores (ViT-B/32 model), and the performance is slightly better on the small pool (at most 0.5 gain of averaged accuracy) while slightly worse on the medium pool (0.4-0.8 loss of averaged accuracy). In Table 16, we show how the CLIP score thresholds relate to the fraction of the pool retained by the filter.

Text-based filtering. Each synset is represented by a synset offset that can be used to retrieve the synset from WordNet. In order to verify if a caption has a word corresponding to a synset from our set we iterate over every word and retrieve the synsets that this word can describe (using nltk.corpus WordNet). Following that, we retrieve the most likely lemma representing that synset, find its synset offset, and check if the number is part of the IN21K or IN1K sets. $^{7}$

Text-based sampling. This baseline uses text only to filter labels which mention concepts (synsets) appearing in IN21K, and applies a temperature parameter to control how equally-represented different concepts are in the dataset. For synset j, let $N_{j}$ be the number of examples containing words matched to that synset, where as before for each word we only match the most likely synset. Furthermore, for image-text pair i let $T_{i}$ be the set of synset matched to the caption.

The probability of sampling example i is proportional to either $\frac{1}{|T_{i}|}\sum_{j\in T_{i}}N_{j}^{\alpha-1}$ (average synset score in the data point) or $\max_{j\in T_{i}}N_{j}^{\alpha-1}$ (maximum synset score in the data point), where $\alpha$ is a “temperature” parameter controlling the flatness of the distribution. We sample examples with replacement but discard any example repeated more than 100 times.

Image-based filtering. We now provide a detailed description of the Image-based filtering procedure. First, since the core of the procedure concerns only image content, we begin with basic text-bsaed filtering: we remove from the pool only all examples with non-English captions (as determined by fasttext), and all examples whose captions have less than two words or less than six characters.

Next, we use clustering of image embeddings to select a subset of examples whose image content is related to a clean training set of interest. Let $e_{1}, \ldots, e_{M}$ denote the CLIP image embeddings of the remaining examples in the pool. We cluster these embeddings into $K = 10^{5}$ clusters using Faiss with 20 iterations, and let $c_{1}, \ldots, c_{K}$ denote the resulting cluster centers. Due to memory constraints, for the large and xlarge pools, we perform the clustering on a random subset of about 160M examples (that pass the basic text-based filtering). For an embedding vector v, let

$$
I (v) = \arg \max _ {i \leq K} \langle v, c _ {i} \rangle
$$

denote the index of the cluster center nearest to v as measured by inner product. Let $f_{1}, \ldots, f_{N}$ denote the CLIP image embeddings of a clean supervised training set (we experiment with either ImageNet 1K or ImageNet 21K), and let

$$
\mathcal {S} = \{I (f _ {i}) \mid 1 \leq i \leq N \}
$$

be the set of cluster indices who are nearest neighbors to some clean training set image. We then keep only images in the pool whose nearest cluster center is in S. That is, out of the M examples passing the text-based filtering, the output subset keeps the examples with indices

$$
\{1 \leq j \leq M \mid I (e _ {j}) \in \mathcal {S} \}.
$$

Image-based sampling. In addition to filtering methods, we experiment with cluster-based sampling methods. First, we compute the score of i-th cluster $s_{i}$ as the number of ImageNet data assigned to this cluster. Then, for parameter $\alpha > 0$ we define a distribution over the pool by sampling cluster i with probability $\frac{s_{i}^{\alpha}}{\sum_{j}s_{j}^{\alpha}}$ and uniformly sampling an example for the cluster, rejecting any example repeated more than 100 times. We try 5 different $\alpha$ , i.e., $\{0, 0.2, 0.5, 1.0, 2.0\}$ , and the best average accuracy is obtained when $\alpha = 0.2$ , while the performance is still worse than the image-based filtering on the small and medium pool. We therefore do not include this line of baselines in the experiments of large pool.

Table 16: CLIP threshold filtering configurations. “Fraction” denotes the size of the filtered subset relative to the pool. 

<table><tr><td>CLIP model</td><td>En. filtering</td><td>Threshold</td><td>Fraction</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.384</td><td>1%</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.358</td><td>3%</td></tr><tr><td>ViT-B/32</td><td>✓</td><td>0.300</td><td>10.2%</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.325</td><td>10%</td></tr><tr><td>ViT-B/32</td><td>✓</td><td>0.28</td><td>7.4%</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.300</td><td>20%</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.281</td><td>30%</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.263</td><td>40%</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.247</td><td>50%</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.215</td><td>75%</td></tr><tr><td>ViT-B/32</td><td>✘</td><td>0.193</td><td>90%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.364</td><td>1%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.334</td><td>3%</td></tr><tr><td>ViT-L/14</td><td>✓</td><td>0.300</td><td>5.4%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.295</td><td>10%</td></tr><tr><td>ViT-L/14</td><td>✓</td><td>0.280</td><td>3.3%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.266</td><td>20%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.243</td><td>30%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.222</td><td>40%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.203</td><td>50%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.160</td><td>75%</td></tr><tr><td>ViT-L/14</td><td>✘</td><td>0.129</td><td>90%</td></tr></table>

ImageNet distance filtering. We rank the samples in the pool by the minimum embedding distance (1 minus cosine similarity) between its image and the ImageNet images; both embeddings are obtained from OpenAI pretrained CLIP ViT-L/14 model $[111]$ . Then we select top images by different fractions as in image-based filtering methods.

# P.2 BYOD track

We experiment with the following data sources:

- CC12M [24]: images and HTML alt-text crawled and filtered from web pages.   
- YFCC15M: this is the 15M subset of the YFCC100M dataset [140] that Radford et al. [111] used for dataset ablation in their CLIP paper.   
- RedCaps [38]: 12M images and corresponding captions were crawled from 350 manually curated subreddits between 2008 and 2020.   
- Shutterstock: 106M images and captions were obtained from the Shutterstock website in 2021 [101]. We use the “photos” subset of this dataset, with 58M samples, which we found performed best, unless specified otherwise.   
- WIT [136]: Image-text pairs from Wikipedia pages. We use the attribution fields as captions, which we found performed best.   
• COYO [20]: A collection of 700M image-text pairs from Common Crawl.   
• LAION-2B [129]: A 2.32 billion english subset of LAION-5B.   
- LAION-COCO: A dataset with 600M images from LAION-5B and synthetic captions. $^{8}$   
- LAION-A: According to laion.ai, LAION-A is a 900M subset of LAION-2B [129] with the aesthetic filtering procedure used in LAION-aesthetic $^{9}$ and pHash deduplication [72].

In Table 17, we use some heuristics to measure the quality of some external data sources. First, following Nguyen et al. [101], we train a CLIP model on a 5M random subset from each source, and evaluate the performance of the resulting models on ImageNet and ImageNet-derived distributions — ImageNet-V2 [121], ImageNet-R [64],

Table 17: Measuring the quality of external data sources 

<table><tr><td>Dataset</td><td>Dataset size</td><td>ImageNet acc.</td><td>Avg. accuracy ImageNet and OOD sets</td><td>Avg. cos. sim. (B/32)</td><td>Avg. cos. sim. (L/14)</td></tr><tr><td>CC12M</td><td>10M</td><td>27.8</td><td>34.0</td><td>0.306</td><td>0.268</td></tr><tr><td>YFCC15M</td><td>15M</td><td>22.6</td><td>24.6</td><td>0.262</td><td>0.198</td></tr><tr><td>RedCaps</td><td>11M</td><td>26.8</td><td>31.5</td><td>0.281</td><td>0.240</td></tr><tr><td>Shutterstock</td><td>15M</td><td>21.0</td><td>28.3</td><td>0.314</td><td>0.273</td></tr></table>

![](images/081012d93f1174bba0002fa94cbb01553751747fa7ac6a1c5d417c8e64eae445.jpg)  
Basic   
Text-based   
Image-based   
▲ CLIP score   
ImageNet ResNet-50   
+ Fine-tuned ResNet-50

Figure 21: Comparison of average and worst-group scores for Dollar Street and GeoDE diversity datasets. On Dollar Street, our overall higher-performing models display a larger worst-group performance gap (corresponding to lower income households). GeoDE does not show this trend.

ImageNet-Sketch [143] and ObjectNet [13]. Moreover, for each data source, we use OpenAI's pretrained CLIP ViT-B/32 and ViT-L/14 models to compute the cosine similarity between image and text embeddings of a data point, and obtain the average cosine similarity score for the whole dataset.

# P.2.1 Additional results

We present a series of results for the BYOD track in Table 18.

# Q Fairness and biases

To study the biases displayed by our models, we include two diversity-related datasets, Dollar Street $[122]$ and GeoDE $[114]$ , in our evaluation suite, and perform further analysis on the face datasets FairFace $[80]$ and UTKFace $[159]$ with demographic labels, following Radford et al. $[111]$ .

# Q.1 Diversity

We break down model performance on the Dollar Street and GeoDE datasets in Figure 21. Dollar Street consists of images of household items taken in homes around the world, and represents a wide socioeconomic range that includes homes with no Internet access $[122]$ . The objects belong to ImageNet categories, and the task is image classification. Standard ImageNet-trained models achieve monotonically increasing performance levels with higher household income levels $[122]$ . Here we use the income-based subgroups defined in Rojas et al. $[122]$ , and find a similar bias as discovered in their paper. While our trained models show a smaller worst-group performance gap than an ImageNet-trained ResNet-50, they underperform a model fine-tuned on Dollar Street. Models with higher average accuracy show a larger worst-group gap, which future work should try to address.

GeoDE consists of images of everyday items and objects, which again fall into ImageNet categories. The dataset represents six world regions equally, and primarily aims to promote geographic diversity of datasets $[114]$ . Both ImageNet models and our models show less bias under this distribution compared to Dollar Street, with a smaller worst-group accuracy gap. The trends show that performance across all regions improves steadily with increased scale, and the performance approaches that of a model fine-tuned on GeoDE. While we know that classifiers trained specifically on ImageNet can display geographic biases $[114]$ , these biases are not apparent in our GeoDE model evaluations. Future work is needed to investigate the extent to which our models have geographic biases not evaluated in GeoDE.

Table 18: Zero-shot performance for select baselines in the BYOD track. Unless specified otherwise, COMMONPOOL means our pool filtered with CLIP score (L/14, 30%). 

<table><tr><td>Scale</td><td>Data source</td><td>Training dataset size</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td rowspan="9">small</td><td>#0</td><td>CC12M</td><td>0.099</td><td>0.080</td><td>0.223</td><td>0.197</td><td>0.205</td></tr><tr><td>#1</td><td>LAION15M</td><td>0.083</td><td>0.076</td><td>0.210</td><td>0.144</td><td>0.189</td></tr><tr><td>#2</td><td>RedCaps</td><td>0.076</td><td>0.066</td><td>0.177</td><td>0.141</td><td>0.168</td></tr><tr><td>#3</td><td>Shutterstock 15M</td><td>0.083</td><td>0.070</td><td>0.214</td><td>0.159</td><td>0.185</td></tr><tr><td>#4</td><td>YFCC15M</td><td>0.071</td><td>0.046</td><td>0.182</td><td>0.147</td><td>0.164</td></tr><tr><td>#5</td><td>#0 + #1 + #2</td><td>0.097</td><td>0.084</td><td>0.208</td><td>0.161</td><td>0.195</td></tr><tr><td>#6</td><td>#0 + #1 + #3</td><td>0.091</td><td>0.081</td><td>0.222</td><td>0.138</td><td>0.202</td></tr><tr><td>#7</td><td>#0 + #2 + #3 + #4</td><td>0.095</td><td>0.075</td><td>0.205</td><td>0.164</td><td>0.186</td></tr><tr><td>#8</td><td>#0-4</td><td>0.093</td><td>0.076</td><td>0.205</td><td>0.162</td><td>0.193</td></tr><tr><td rowspan="17">medium</td><td>#9</td><td>CC12M</td><td>0.245</td><td>0.189</td><td>0.283</td><td>0.289</td><td>0.272</td></tr><tr><td>#10</td><td>LAION15M</td><td>0.270</td><td>0.215</td><td>0.317</td><td>0.255</td><td>0.306</td></tr><tr><td>#11</td><td>RedCaps</td><td>0.237</td><td>0.166</td><td>0.271</td><td>0.178</td><td>0.263</td></tr><tr><td>#12</td><td>Shutterstock 15M</td><td>0.229</td><td>0.191</td><td>0.316</td><td>0.260</td><td>0.290</td></tr><tr><td>#13</td><td>YFCC15M</td><td>0.232</td><td>0.137</td><td>0.263</td><td>0.245</td><td>0.257</td></tr><tr><td>#14</td><td>#9 + #10 + #11</td><td>0.376</td><td>0.287</td><td>0.387</td><td>0.323</td><td>0.366</td></tr><tr><td>#15</td><td>#9 + #10 + #12</td><td>0.342</td><td>0.278</td><td>0.362</td><td>0.345</td><td>0.357</td></tr><tr><td>#16</td><td>#9 + #11 + #12 + #13</td><td>0.360</td><td>0.268</td><td>0.365</td><td>0.275</td><td>0.345</td></tr><tr><td>#17</td><td>#9-13</td><td>0.371</td><td>0.285</td><td>0.408</td><td>0.280</td><td>0.367</td></tr><tr><td>#18</td><td>Shutterstock illustration</td><td>0.053</td><td>0.094</td><td>0.205</td><td>0.125</td><td>0.180</td></tr><tr><td>#19</td><td>Shutterstock photo</td><td>0.342</td><td>0.209</td><td>0.364</td><td>0.350</td><td>0.331</td></tr><tr><td>#20</td><td>Shutterstock vectors</td><td>0.072</td><td>0.151</td><td>0.216</td><td>0.148</td><td>0.208</td></tr><tr><td>#21</td><td>Shutterstock full</td><td>0.313</td><td>0.254</td><td>0.353</td><td>0.331</td><td>0.342</td></tr><tr><td>#22</td><td>WIT full</td><td>0.096</td><td>0.063</td><td>0.196</td><td>0.104</td><td>0.177</td></tr><tr><td>#23</td><td>WIT English</td><td>0.051</td><td>0.038</td><td>0.145</td><td>0.083</td><td>0.143</td></tr><tr><td>#24</td><td>COYO</td><td>0.272</td><td>0.235</td><td>0.301</td><td>0.254</td><td>0.304</td></tr><tr><td>#25</td><td>LAION-COCO</td><td>0.209</td><td>0.205</td><td>0.293</td><td>0.359</td><td>0.297</td></tr><tr><td rowspan="24">large</td><td>#26</td><td>Shutterstock illustration</td><td>0.337</td><td>0.203</td><td>0.307</td><td>0.322</td><td>0.306</td></tr><tr><td>#27</td><td>Shutterstock photo</td><td>0.485</td><td>0.304</td><td>0.432</td><td>0.427</td><td>0.398</td></tr><tr><td>#28</td><td>Shutterstock vectors</td><td>0.126</td><td>0.223</td><td>0.244</td><td>0.191</td><td>0.246</td></tr><tr><td>#29</td><td>Shutterstock full</td><td>0.500</td><td>0.412</td><td>0.472</td><td>0.451</td><td>0.456</td></tr><tr><td>#30</td><td>COYO</td><td>0.547</td><td>0.456</td><td>0.475</td><td>0.549</td><td>0.486</td></tr><tr><td>#31</td><td>LAION-COCO</td><td>0.355</td><td>0.351</td><td>0.395</td><td>0.494</td><td>0.398</td></tr><tr><td>#32</td><td>COYO + LAION-COCO</td><td>0.528</td><td>0.458</td><td>0.479</td><td>0.589</td><td>0.498</td></tr><tr><td>#33</td><td>LAION-A</td><td>0.611</td><td>0.474</td><td>0.501</td><td>0.542</td><td>0.505</td></tr><tr><td>#34</td><td>LAION-2B</td><td>0.585</td><td>0.472</td><td>0.504</td><td>0.525</td><td>0.515</td></tr><tr><td>#35</td><td>COMMONPOOL + #9-13</td><td>0.602</td><td>0.498</td><td>0.541</td><td>0.416</td><td>0.537</td></tr><tr><td>#36</td><td>COMMONPOOL + #9-13 (2x upsampled)</td><td>0.613</td><td>0.507</td><td>0.559</td><td>0.433</td><td>0.543</td></tr><tr><td>#37</td><td>COMMONPOOL + #9-13 (4x upsampled)</td><td>0.615</td><td>0.514</td><td>0.553</td><td>0.427</td><td>0.543</td></tr><tr><td>#38</td><td>COMMONPOOL + #9-13 (6x upsampled)</td><td>0.620</td><td>0.519</td><td>0.558</td><td>0.437</td><td>0.549</td></tr><tr><td>#39</td><td>COMMONPOOL + #9-13 (8x upsampled)</td><td>0.624</td><td>0.520</td><td>0.533</td><td>0.443</td><td>0.537</td></tr><tr><td>#40</td><td>COMMONPOOL + #9-13 (10x upsampled)</td><td>0.621</td><td>0.520</td><td>0.540</td><td>0.441</td><td>0.537</td></tr><tr><td>#41</td><td>COMMONPOOL + COYO</td><td>0.561</td><td>0.472</td><td>0.504</td><td>0.508</td><td>0.513</td></tr><tr><td>#42</td><td>COMMONPOOL + LAION-A</td><td>0.607</td><td>0.480</td><td>0.531</td><td>0.514</td><td>0.527</td></tr><tr><td>#43</td><td>COMMONPOOL + LAION-COCO</td><td>0.522</td><td>0.457</td><td>0.513</td><td>0.498</td><td>0.514</td></tr><tr><td>#44</td><td>COMMONPOOL + #9+#11+#13+#19</td><td>0.609</td><td>0.508</td><td>0.546</td><td>0.439</td><td>0.536</td></tr><tr><td>#45</td><td>COMMONPOOL + #9+#11+#13+#19 (2x upsampled)</td><td>0.621</td><td>0.509</td><td>0.547</td><td>0.458</td><td>0.541</td></tr><tr><td>#46</td><td>COMMONPOOL + #9+#11+#13+#19 (4x upsampled)</td><td>0.632</td><td>0.515</td><td>0.533</td><td>0.452</td><td>0.532</td></tr><tr><td>#47</td><td>COMMONPOOL + #9+#11+#13+#19 (6x upsampled)</td><td>0.635</td><td>0.515</td><td>0.535</td><td>0.471</td><td>0.532</td></tr><tr><td>#48</td><td>COMMONPOOL + #9+#11+#13+#19 (8x upsampled)</td><td>0.633</td><td>0.515</td><td>0.523</td><td>0.464</td><td>0.530</td></tr><tr><td>#49</td><td>COMMONPOOL + #9+#11+#13+#19 (10x upsampled)</td><td>0.630</td><td>0.513</td><td>0.523</td><td>0.356</td><td>0.521</td></tr><tr><td rowspan="4">xlarge</td><td>#50</td><td>LAION-2B</td><td>0.757</td><td>0.631</td><td>0.611</td><td>0.619</td><td>0.621</td></tr><tr><td>#51</td><td>COMMONPOOL + #9+#11+#13+#19</td><td>0.766</td><td>0.660</td><td>0.662</td><td>0.539</td><td>0.659</td></tr><tr><td>#52</td><td>COMMONPOOL + #9+#11+#13+#19 (6x upsampled)</td><td>0.776</td><td>0.671</td><td>0.633</td><td>0.552</td><td>0.649</td></tr><tr><td>#53</td><td>COMMONPOOL + #9+#11+#13+#19 (18x upsampled)</td><td>0.771</td><td>0.667</td><td>0.629</td><td>0.554</td><td>0.643</td></tr></table>

# Q.2 Fairness

Emulating Radford et al. [111], we evaluate our best models from the filtering and BYOD tracks on the human face datasets FairFace and UTKFace, using zero-shot classification to predict the race, gender, and age annotated in these datasets. Following Hanna et al. [59] and Hundt et al. [68], we acknowledge that these evaluations can be problematic as race and gender should not be considered fixed categories, but rather fluid attributes that may change for individuals, based on they way they identify at any given moment—regardless of appearance. We include these evaluations for continuity with prior work and as a probe into model behaviour, but hope future work will consider improved face fairness evaluation. We also note that race, gender, and age classification are not the intended end-goals of the models or benchmark, and we do not condone the use of COMMONPOOL or models trained on COMMONPOOL data for any decisions involving people.

Table 19: Overall race, gender, and age classification accuracy of our two best xlarge baselines, Image-based $\cap$ CLIP score (L/14 $30\%$ ) for the filtering track and COMMONPOOL, CLIP score + 4 external sources (upsampled 6x) for the BYOD track. Race classification was binary (white or non-white) as in Karkkainen & Joo [80]. 

<table><tr><td>Dataset</td><td>Track</td><td>Race</td><td>Gender</td><td>Age</td></tr><tr><td rowspan="2">FairFace</td><td>Filtering</td><td>86.4</td><td>91.7</td><td>34.3</td></tr><tr><td>BYOD</td><td>76.5</td><td>93.9</td><td>33.8</td></tr><tr><td rowspan="2">UTKFace</td><td>Filtering</td><td>86.2</td><td>93.8</td><td>39.5</td></tr><tr><td>BYOD</td><td>86.1</td><td>95.5</td><td>38.6</td></tr></table>

Table 20: Gender classification accuracy of our two best xlarge baselines, Image-based $\cap$ CLIP score (L/14 $30\%$ ) for the filtering track and COMMONPOOL, CLIP score + 4 external sources (upsampled 6x) for the BYOD track.   
FairFace 

<table><tr><td rowspan="2">Track</td><td rowspan="2">Gender</td><td colspan="7">Race</td></tr><tr><td>Black</td><td>White</td><td>Indian</td><td>Latino/Hispanic</td><td>Middle Eastern</td><td>Southeast Asian</td><td>East Asian</td></tr><tr><td rowspan="2">Filtering</td><td>Male</td><td>79.3</td><td>91.3</td><td>90.8</td><td>90.4</td><td>95.7</td><td>83.0</td><td>80.7</td></tr><tr><td>Female</td><td>95.4</td><td>96.6</td><td>94.2</td><td>96.6</td><td>96.5</td><td>97.2</td><td>98.2</td></tr><tr><td rowspan="2">BYOD</td><td>Male</td><td>89.2</td><td>94.8</td><td>93.2</td><td>93.4</td><td>97.4</td><td>90.2</td><td>90.6</td></tr><tr><td>Female</td><td>89.2</td><td>96.0</td><td>94.2</td><td>96.0</td><td>96.2</td><td>97.1</td><td>97.0</td></tr></table>

UTKFace 

<table><tr><td rowspan="2">Track</td><td rowspan="2">Gender</td><td colspan="5">Race</td></tr><tr><td>Black</td><td>White</td><td>Indian</td><td>Asian</td><td>Other</td></tr><tr><td rowspan="2">Filtering</td><td>Male</td><td>95.4</td><td>92.5</td><td>91.7</td><td>73.1</td><td>84.2</td></tr><tr><td>Female</td><td>97.3</td><td>98.7</td><td>97.4</td><td>98.3</td><td>97.4</td></tr><tr><td rowspan="2">BYOD</td><td>Male</td><td>96.8</td><td>95.9</td><td>94.7</td><td>85.7</td><td>90.4</td></tr><tr><td>Female</td><td>96.3</td><td>97.7</td><td>96.8</td><td>95.9</td><td>95.6</td></tr></table>

As described in Appendix G, our filleting track models are trained on images with faces blurred. Nevertheless, these models still perform significantly above random chance on face classification. We hypothesize that this is due to a combination of faces bypassing our face blurring filter in the training data, contextual clues outside of the face region, or signal associated with skin color. The BYOD track model performs even better than the filtering track model. We hypothesize that this is because BYOD data is used off-the-shelf and hence contains non-blurred faces. In Table 19, we present overall accuracy for these three traits. Note that race is treated as a binary variable (white or non-white) to enable comparison to prior results, gender is a binary variable (male or female) according to annotations, and age is binned into 9 ranges according to the annotation precision of FairFace. The BYOD model, performs better at distinguishing the annotated gender, but is worse at distinguishing annotated race and age.

We further break down these statistics over the intersection of race and gender, examining gender classification accuracies in Table 20. We find that there are drastic differences in accuracy across different annotated subgroups, varying by both race and gender. The filtering models show a tendency to misclassify Black, Southeast Asian, and East Asian males as females at 20.7%, 17%, and 19.3% respectively on FairFace. Furthermore, we find that while the BYOD model improves accuracy, on FairFace most of this improvement is on men (ranging from 1.7pp gain to 9.9pp gain), while on women, BYOD offers little change (ranging from 0.6pp gain to 6.2pp drop).

Following Radford et al. [111], we also examined associations of particular demographics with potentially harmful language. We replicate their setup with two classification tasks: (1) including race-gender intersection classes (e.g. “black woman”, “indian man”, etc.) and several harmful crime-related terms (“thief”, “criminal”, “suspicious person”); (2) including the same race-gender intersection classes and non-human terms (“animal”, “gorilla”, “chimpanzee”, “orangutan”). We compute the frequency of misclassification of people into one of the harmful categories and run these experiments on FairFace and UTKFace separately. The results are shown in Table 21. Unlike in Radford et al. [111], we find that our models have a very small probability of classifying human faces as non-human, with a max score across all subgroups of 0.1%. However, a significant proportion of people are misclassified as criminal. This again highlights the importance of dataset curation and the risks associated with zero-shot classification on models trained on web-scraped datasets.

Table 21: Harmful misclassification rates of our two best xlarge baselines, Image-based $\cap$ CLIP score (L/14 $30\%$ ) for the filtering track and COMMONPOOL, CLIP score + 4 external sources (upsampled 6x) for the BYOD track. While very few samples are misclassified as non-human, the filter track model assigns a crime-related label to a significant portion of people, and this is exacerbated by the BYOD model in many cases.   
FairFace 

<table><tr><td rowspan="2" colspan="2">Track</td><td colspan="7">Race</td></tr><tr><td>Black</td><td>White</td><td>Indian</td><td>Latino/Hispanic</td><td>Middle Eastern</td><td>Southeast Asian</td><td>East Asian</td></tr><tr><td rowspan="2">Filtering</td><td>Crime-related</td><td>4.4</td><td>24.3</td><td>8.8</td><td>14.3</td><td>23.7</td><td>7.4</td><td>8.6</td></tr><tr><td>Non-human</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td rowspan="2">BYOD</td><td>Crime-related</td><td>18.4</td><td>16.8</td><td>21.5</td><td>22.9</td><td>20.9</td><td>35.3</td><td>30.9</td></tr><tr><td>Non-human</td><td>0.0</td><td>0.1</td><td>0.0</td><td>0.1</td><td>0.0</td><td>0.1</td><td>0.1</td></tr></table>

UTKFace 

<table><tr><td rowspan="2" colspan="2">Track</td><td colspan="5">Race</td></tr><tr><td>Black</td><td>White</td><td>Indian</td><td>Asian</td><td>Other</td></tr><tr><td rowspan="2">Filtering</td><td>Crime-related</td><td>6.8</td><td>16.1</td><td>9.1</td><td>6.9</td><td>13.9</td></tr><tr><td>Non-human</td><td>0.0</td><td>0.2</td><td>0.0</td><td>0.1</td><td>0.0</td></tr><tr><td rowspan="2">BYOD</td><td>Crime-related</td><td>12.8</td><td>10.8</td><td>15.2</td><td>13.2</td><td>18.6</td></tr><tr><td>Non-human</td><td>0.0</td><td>0.2</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

# R Extra figures and tables

![](images/c9d9b1dd0e97b7d8cd1c3ab19565ff6b75c7492abbb667e3c7b27cbacc19e81e.jpg)

<details>
<summary>scatter</summary>

| ImageNet acc. (small) | ImageNet acc. (medium) |
| --------------------- | ---------------------- |
| 0.00                  | 0.05                   |
| 0.01                  | 0.10                   |
| 0.02                  | 0.15                   |
| 0.03                  | 0.20                   |
| 0.04                  | 0.25                   |
| 0.05                  | 0.30                   |
</details>

![](images/0a4f8d83cb36d0ca5c7678093112fd9cc61a5324004ddb296df2d1ea86cf5b6f.jpg)

<details>
<summary>scatter</summary>

| ImageNet acc. (medium) | ImageNet acc. (large) |
| ---------------------- | --------------------- |
| 0.10                   | 0.43                  |
| 0.15                   | 0.45                  |
| 0.20                   | 0.48                  |
| 0.25                   | 0.55                  |
| 0.26                   | 0.57                  |
| 0.27                   | 0.58                  |
| 0.28                   | 0.59                  |
| 0.29                   | 0.60                  |
| 0.30                   | 0.61                  |
| 0.31                   | 0.62                  |
| 0.32                   | 0.63                  |
| 0.33                   | 0.64                  |
| 0.34                   | 0.65                  |
| 0.35                   | 0.66                  |
| 0.36                   | 0.67                  |
| 0.37                   | 0.68                  |
| 0.38                   | 0.69                  |
| 0.39                   | 0.70                  |
| 0.40                   | 0.71                  |
| 0.41                   | 0.72                  |
| 0.42                   | 0.73                  |
| 0.43                   | 0.74                  |
| 0.44                   | 0.75                  |
| 0.45                   | 0.76                  |
| 0.46                   | 0.77                  |
| 0.47                   | 0.78                  |
| 0.48                   | 0.79                  |
| 0.49                   | 0.80                  |
| 0.50                   | 0.81                  |
| 0.51                   | 0.82                  |
| 0.52                   | 0.83                  |
| 0.53                   | 0.84                  |
| 0.54                   | 0.85                  |
| 0.55                   | 0.86                  |
| 0.56                   | 0.87                  |
| 0.57                   | 0.88                  |
| 0.58                   | 0.89                  |
| 0.59                   | 0.90                  |
| 0.60                   | 0.91                  |
| 0.61                   | 0.92                  |
| 0.62                   | 0.93                  |
| 0.63                   | 0.94                  |
| 0.64                   | 0.95                  |
| 0.65                   | 0.96                  |
| 0.66                   | 0.97                  |
| 0.67                   | 0.98                  |
| 0.68                   | 0.99                  |
| 0.69                   | 1.00                  |
| 0.70                   | 1.01                  |
| 0.71                   | 1.02                  |
| 0.72                   | 1.03                  |
| 0.73                   | 1.04                  |
| 0.74                   | 1.05                  |
| 0.75                   | 1.06                  |
| 0.76                   | 1.07                  |
| 0.77                   | 1.08                  |
| 0.78                   | 1.09                  |
| 0.79                   | 1.10                  |
| 0.80                   | 1.11                  |
| 0.81                   | 1.12                  |
| 0.82                   | 1.13                  |
| 0.83                   | 1.14                  |
| 0.84                   | 1.15                  |
| 0.85                   | 1.16                  |
| 0.86                   | 1.17                  |
| 0.87                   | 1.18                  |
| 0.88                   | 1.19                  |
| 0.89                   | 1.20                  |
| 0.90                   | 1.21                  |
| 0.91                   | 1.22                  |
| 0.92                   | 1.23                  |
| 0.93                   | 1.24                  |
| 0.94                   | 1.25                  |
| 0.95                   | 1.26                  |
| 0.96                   | 1.27                  |
| 0.97                   | 1.28                  |
| 0.98                   | 1.29                  |
| 0.99                   | 1.30                  |
| 1.00                   | 1.31                  |
</details>

![](images/cf251b89b09603cb4349fec7e6ccb8e59106074a6bae6f25e0a731fb6c50cd1e.jpg)

<details>
<summary>scatter</summary>

| ImageNet acc. (small) | ImageNet acc. (large) |
| --------------------- | --------------------- |
| 0.01                  | 0.43                  |
| 0.02                  | 0.45                  |
| 0.03                  | 0.47                  |
| 0.04                  | 0.50                  |
| 0.05                  | 0.55                  |
| 0.06                  | 0.56                  |
| 0.07                  | 0.57                  |
| 0.08                  | 0.58                  |
| 0.09                  | 0.59                  |
| 0.10                  | 0.60                  |
</details>

![](images/1dc6910f4c9429015dfda4d2c36f7bedeb26a13fbbeb63990e299d70f089ec60.jpg)

<details>
<summary>scatter</summary>

| Avg. pref. metric (small) | Avg. pref. metric (medium) |
| ------------------------- | -------------------------- |
| 0.08                      | 0.12                       |
| 0.09                      | 0.15                       |
| 0.10                      | 0.18                       |
| 0.11                      | 0.20                       |
| 0.12                      | 0.22                       |
| 0.13                      | 0.24                       |
| 0.14                      | 0.26                       |
| 0.15                      | 0.28                       |
| 0.16                      | 0.30                       |
| 0.17                      | 0.32                       |
| 0.18                      | 0.34                       |
| 0.19                      | 0.36                       |
| 0.20                      | 0.38                       |
| 0.21                      | 0.40                       |
| 0.22                      | 0.42                       |
| 0.23                      | 0.44                       |
| 0.24                      | 0.46                       |
| 0.25                      | 0.48                       |
| 0.26                      | 0.50                       |
| 0.27                      | 0.52                       |
| 0.28                      | 0.54                       |
| 0.29                      | 0.56                       |
| 0.30                      | 0.58                       |
| 0.31                      | 0.60                       |
| 0.32                      | 0.62                       |
| 0.33                      | 0.64                       |
| 0.34                      | 0.66                       |
| 0.35                      | 0.68                       |
| 0.36                      | 0.70                       |
| 0.37                      | 0.72                       |
| 0.38                      | 0.74                       |
| 0.39                      | 0.76                       |
| 0.40                      | 0.78                       |
| 0.41                      | 0.80                       |
| 0.42                      | 0.82                       |
| 0.43                      | 0.84                       |
| 0.44                      | 0.86                       |
| 0.45                      | 0.88                       |
| 0.46                      | 0.90                       |
| 0.47                      | 0.92                       |
| 0.48                      | 0.94                       |
| 0.49                      | 0.96                       |
| 0.50                      | 0.98                       |
| 0.51                      | 1.00                       |
| 0.52                      | 1.02                       |
| 0.53                      | 1.04                       |
| 0.54                      | 1.06                       |
| 0.55                      | 1.08                       |
| 0.56                      | 1.10                       |
| 0.57                      | 1.12                       |
| 0.58                      | 1.14                       |
| 0.59                      | 1.16                       |
| 0.60                      | 1.18                       |
| 0.61                      | 1.20                       |
| 0.62                      | 1.22                       |
| 0.63                      | 1.24                       |
| 0.64                      | 1.26                       |
| 0.65                      | 1.28                       |
| 0.66                      | 1.30                       |
| 0.67                      | 1.32                       |
| 0.68                      | 1.34                       |
| 0.69                      | 1.36                       |
| 0.70                      | 1.38                       |
| 0.71                      | 1.40                       |
| 0.72                      | 1.42                       |
| 0.73                      | 1.44                       |
| 0.74                      | 1.46                       |
| 0.75                      | 1.48                       |
| 0.76                      | 1.50                       |
| 0.77                      | 1.52                       |
| 0.78                      | 1.54                       |
| 0.79                      | 1.56                       |
| 0.80                      | 1.58                       |
| 0.81                      | 1.60                       |
| 0.82                      | 1.62                       |
| 0.83                      | 1.64                       |
| 0.84                      | 1.66                       |
| 0.85                      | 1.68                       |
| 0.86                      | 1.70                       |
| 0.87                      | 1.72                       |
| 0.88                      | 1.74                       |
| 0.89                      | 1.76                       |
| 0.90                      | 1.78                       |
| 0.91                      | 1.80                       |
| 0.92                      | 1.82                       |
| 0.93                      | 1.84                       |
| 0.94                      | 1.86                       |
| 0.95                      | 1.88                       |
| 0.96                      | 1.90                       |
| 0.97                      | 1.92                       |
| 0.98                      | 1.94                       |
| 0.99                      | 1.96                       |
| 1.00                      | 1.98                       |
</details>

![](images/34a3bc693afae36872ceb5809f31bc1ebb76c2ea113fec05f46274140f7b9c72.jpg)

<details>
<summary>scatter</summary>

| Avg. pref. metric (medium) | Avg. pref. metric (large) |
| -------------------------- | ------------------------- |
| 0.15                       | 0.35                      |
| 0.20                       | 0.42                      |
| 0.25                       | 0.45                      |
| 0.28                       | 0.48                      |
| 0.30                       | 0.50                      |
| 0.32                       | 0.52                      |
</details>

![](images/4eb349f4580078781f7ef23d96c09137bfb9591e788f0994e34cd595824f15fc.jpg)

<details>
<summary>scatter</summary>

| Avg. pref. metric (small) | Avg. pref. metric (large) |
| ------------------------- | ------------------------- |
| 0.100                     | 0.35                      |
| 0.125                     | 0.48                      |
| 0.150                     | 0.46                      |
| 0.175                     | 0.52                      |
</details>

<table><tr><td>Basic</td><td>Image-based</td><td>No filtering</td><td>Rand. subset</td><td>Text-based</td></tr><tr><td>CLIP score</td><td>ImageNet dist.</td><td></td><td></td><td></td></tr></table>

Figure 22: Improving downstream performance at smaller scales correlates positively with performance gains at larger scales. These trends suggests that dataset filtering can be studied effectively at smaller scales, even with less computational resources.

Table 22: Rank correlation between the performance obtained with various filtering strategies at two different scales. Our experimental suggest that the ranking is relatively consistent between scales, especially for the adjacent scale pairs. 

<table><tr><td>Metric</td><td>small vs medium</td><td>small vs large</td><td>medium vs large</td></tr><tr><td>ImageNet acc.</td><td>0.895</td><td>0.811</td><td>0.847</td></tr><tr><td>Average pref. metric</td><td>0.854</td><td>0.708</td><td>0.876</td></tr></table>

![](images/2c6e1de6425fa2ebd03adeb7af8318f7b263f6c8a9b1a7ccb09f254d66948d7d.jpg)

Figure 23: Performance as a function of the number of training samples from the small (top), medium (middle), and large (bottom) scales. There is a significant variance in accuracy even when accounting for the size of the training set.   
Table 23: Comparison of ViT-B/32 and ViT-B/16 models across different training datasets. 

<table><tr><td>Model</td><td>Training Dataset</td><td>Training dataset size</td><td>Training steps</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td>ViT B/32</td><td>DATACOMP-1B</td><td>1.4B</td><td>13B</td><td>0.692</td><td>0.551</td><td>0.577</td><td>0.538</td><td>0.579</td></tr><tr><td>ViT B/32</td><td>OpenAI&#x27;s WIT</td><td>0.4B</td><td>13B</td><td>0.633</td><td>0.485</td><td>0.526</td><td>0.501</td><td>0.525</td></tr><tr><td>ViT B/32</td><td>LAION-2B</td><td>2.3B</td><td>34B</td><td>0.666</td><td>0.522</td><td>0.561</td><td>0.560</td><td>0.569</td></tr><tr><td>ViT B/16</td><td>DATACOMP-1B</td><td>1.4B</td><td>13B</td><td>0.735</td><td>0.608</td><td>0.621</td><td>0.578</td><td>0.615</td></tr><tr><td>ViT B/16</td><td>OpenAI&#x27;s WIT</td><td>0.4B</td><td>13B</td><td>0.683</td><td>0.559</td><td>0.546</td><td>0.527</td><td>0.563</td></tr><tr><td>ViT B/16</td><td>LAION-2B</td><td>2.3B</td><td>34B</td><td>0.702</td><td>0.566</td><td>0.572</td><td>0.583</td><td>0.587</td></tr></table>

![](images/e1e06aeb827844829af6579c763443120dd0f0d9a44e06d403280be946ae61cd.jpg)

<details>
<summary>line</summary>

| Fraction remaining after CLIP score filtering | ViT-B/32 | ViT-L/14 |
| --------------------------------------------- | -------- | -------- |
| 0.00                                          | 0.85     | 0.75     |
| 0.25                                          | 0.75     | 0.70     |
| 0.50                                          | 0.65     | 0.60     |
| 0.75                                          | 0.55     | 0.55     |
| 1.00                                          | 0.50     | 0.50     |
</details>

![](images/bcece4bc81953924ad2b46c9557db92af6754c4c78e2203b84f8e3d0f77ca59c.jpg)

<details>
<summary>line</summary>

| Fraction remaining after CLIP score filtering | ViT-B/32 | ViT-L/14 |
| --------------------------------------------- | -------- | -------- |
| 0.00                                          | 0.40     | 0.38     |
| 0.25                                          | 0.36     | 0.35     |
| 0.50                                          | 0.31     | 0.29     |
| 0.75                                          | 0.25     | 0.24     |
| 1.00                                          | 0.20     | 0.20     |
</details>

Figure 24: We examine the percentage of texts classified as English after taking the top fraction (on the x-axis) of the large billion pool as sorted by CLIP similarity score. We see that doing CLIP filtering implicitly does some English filtering, as image-text pairs with a higher CLIP score are more frequently classified as English.

![](images/e7363727230740879e1b4235ab1af5f035e1d616b85d4bbb123208de17a17031.jpg)

<details>
<summary>scatter</summary>

| ImageNet accuracy | Average pref. metric (clean) |
| ----------------- | ---------------------------- |
| 0.00              | 0.10                         |
| 0.10              | 0.15                         |
| 0.20              | 0.25                         |
| 0.30              | 0.30                         |
| 0.40              | 0.35                         |
| 0.50              | 0.45                         |
| 0.60              | 0.50                         |
| 0.70              | 0.65                         |
| 0.75              | 0.70                         |
</details>

![](images/aa3c36eae77c2dd3fb666b4a0a50881f17cd1f8e5fe36a48af739a0d7256ffa6.jpg)

<details>
<summary>scatter</summary>

| ImageNet accuracy | Average pref. metric (full) |
| ----------------- | --------------------------- |
| 0.00              | 0.10                        |
| 0.10              | 0.15                        |
| 0.20              | 0.25                        |
| 0.30              | 0.30                        |
| 0.40              | 0.35                        |
| 0.50              | 0.45                        |
| 0.60              | 0.50                        |
| 0.70              | 0.60                        |
| 0.75              | 0.65                        |
</details>

![](images/06a3770aaab4ce4acad87ae5e6cbaf9ef57a15f491fd678d2aca8d05ac3ad830.jpg)

<details>
<summary>text_image</summary>

No filtering
Basic
CLIP score
Image-based
Text-based
Rand. subset
ImageNet dist.
</details>

Figure 25: Correlation between ImageNet accuracy and average performance on our suite of evaluation tasks. While ImageNet accuracy strongly correlates with the average performance (both on the clean subset and the full suite), the same is not true for all individual datasets we study, as shown in Appendix R.   
![](images/8ba6e80d88fdfdf378d734add0aec6eac13c6fe73e8282f7e017b3f8e68e3f90.jpg)

<details>
<summary>scatter</summary>

| ImageNet accuracy | Avg. accuracy on 5 OOD sets |
| ----------------- | --------------------------- |
| 0.0               | 0.0                         |
| 0.1               | 0.1                         |
| 0.2               | 0.2                         |
| 0.3               | 0.3                         |
| 0.4               | 0.4                         |
| 0.5               | 0.5                         |
| 0.6               | 0.6                         |
</details>

<table><tr><td>--- x=yImageNet modelsNo filtering</td><td>BasicCLIP scoreImage-based</td><td>+Text-basedRand. subsetImageNet dist.</td></tr></table>

Figure 26: Zero-shot CLIP models trained with various filtering strategies form a reliable trend relating accuracy on ImageNet and related distribution shifts, exhibiting higher effective robustness when compared to ImageNet-trained models from Taori et al. [139].

Table 24: Comparison at the xlarge scale between a 400M subset of COMMONPOOL and OpenAI's WIT which also contains 400M samples. Our 400M subset is created by intersecting IN1k image clustering with English cld3 filtering, then taking the top 400M samples sorted by CLIP L14 score. Our model does better across the various evaluation groupings. 

<table><tr><td>Model</td><td>Training Dataset</td><td>Training dataset size</td><td>Training steps</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td>ViT L/14</td><td>top 400M by CLIP L14 of Image-based  $\cap$  cld3</td><td>400M</td><td>13B</td><td>0.763</td><td>0.657</td><td>0.641</td><td>0.595</td><td>0.638</td></tr><tr><td>ViT L/14</td><td>OpenAI&#x27;s WIT</td><td>400M</td><td>13B</td><td>0.755</td><td>0.649</td><td>0.586</td><td>0.543</td><td>0.617</td></tr></table>

![](images/88ee527f980c1c31d466238670733aa4049fd3bd3fbf2fd01333dc77e15df748.jpg)  
Figure 27: Zero-shot performance on other datasets is often positively correlated with that on ImageNet, but not always. In cases where ImageNet shows close to zero correlation with other datasets, performance on that dataset is often close to random chance.

Table 25: Baseline results for the filtering track, small scale. 

<table><tr><td>Filtering</td><td>Training dataset size</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td>No filtering</td><td>12.8M</td><td>0.025</td><td>0.033</td><td>0.145</td><td>0.114</td><td>0.133</td></tr><tr><td>Random subset (75%)</td><td>9.6M</td><td>0.028</td><td>0.037</td><td>0.153</td><td>0.110</td><td>0.140</td></tr><tr><td>Random subset (50%)</td><td>6.4M</td><td>0.027</td><td>0.037</td><td>0.147</td><td>0.111</td><td>0.137</td></tr><tr><td>Random subset (25%)</td><td>3.2M</td><td>0.022</td><td>0.032</td><td>0.130</td><td>0.099</td><td>0.126</td></tr><tr><td>Random subset (10%)</td><td>1.3M</td><td>0.010</td><td>0.018</td><td>0.116</td><td>0.077</td><td>0.103</td></tr><tr><td>Random subset (1%)</td><td>128K</td><td>0.002</td><td>0.005</td><td>0.095</td><td>0.049</td><td>0.078</td></tr><tr><td>Caption length</td><td>8.7M</td><td>0.034</td><td>0.040</td><td>0.148</td><td>0.109</td><td>0.143</td></tr><tr><td>Image size</td><td>7.8M</td><td>0.027</td><td>0.036</td><td>0.154</td><td>0.119</td><td>0.138</td></tr><tr><td>English (fasttext)</td><td>6.3M</td><td>0.038</td><td>0.045</td><td>0.164</td><td>0.124</td><td>0.154</td></tr><tr><td>English (fasttext) and caption length</td><td>4.8M</td><td>0.041</td><td>0.048</td><td>0.159</td><td>0.123</td><td>0.154</td></tr><tr><td>English (fasttext), caption length, and image size</td><td>3.0M</td><td>0.038</td><td>0.043</td><td>0.150</td><td>0.118</td><td>0.142</td></tr><tr><td>English (cld3)</td><td>2.6M</td><td>0.032</td><td>0.039</td><td>0.143</td><td>0.111</td><td>0.142</td></tr><tr><td>English (cld3) and caption length</td><td>2.3M</td><td>0.031</td><td>0.038</td><td>0.153</td><td>0.111</td><td>0.142</td></tr><tr><td>English (cld3), caption length, and image size</td><td>1.5M</td><td>0.023</td><td>0.030</td><td>0.154</td><td>0.092</td><td>0.141</td></tr><tr><td>CLIP B32 score top 1%</td><td>129K</td><td>0.003</td><td>0.007</td><td>0.114</td><td>0.050</td><td>0.086</td></tr><tr><td>CLIP B32 score top 3%</td><td>384K</td><td>0.006</td><td>0.014</td><td>0.104</td><td>0.055</td><td>0.089</td></tr><tr><td>CLIP B32 score top 10%</td><td>1.3M</td><td>0.026</td><td>0.035</td><td>0.147</td><td>0.083</td><td>0.126</td></tr><tr><td>CLIP B32 score top 20%</td><td>2.6M</td><td>0.051</td><td>0.056</td><td>0.173</td><td>0.114</td><td>0.161</td></tr><tr><td>CLIP B32 score top 30%</td><td>3.8M</td><td>0.045</td><td>0.052</td><td>0.180</td><td>0.120</td><td>0.167</td></tr><tr><td>CLIP B32 score top 40%</td><td>5.1M</td><td>0.052</td><td>0.057</td><td>0.173</td><td>0.123</td><td>0.167</td></tr><tr><td>CLIP B32 score top 50%</td><td>6.4M</td><td>0.047</td><td>0.053</td><td>0.174</td><td>0.124</td><td>0.165</td></tr><tr><td>CLIP B32 score top 75%</td><td>9.6M</td><td>0.033</td><td>0.043</td><td>0.161</td><td>0.121</td><td>0.151</td></tr><tr><td>CLIP B32 score top 90%</td><td>11.5M</td><td>0.028</td><td>0.039</td><td>0.140</td><td>0.114</td><td>0.136</td></tr><tr><td>CLIP B32 threshold at 0.3 + English filter</td><td>942K</td><td>0.022</td><td>0.032</td><td>0.138</td><td>0.077</td><td>0.122</td></tr><tr><td>CLIP B32 threshold at 0.28 + English filter</td><td>1.3M</td><td>0.031</td><td>0.040</td><td>0.136</td><td>0.092</td><td>0.133</td></tr><tr><td>CLIP B32 threshold at 0.3</td><td>2.6M</td><td>0.052</td><td>0.056</td><td>0.166</td><td>0.114</td><td>0.161</td></tr><tr><td>CLIP B32 score 1% to 30%</td><td>3.7M</td><td>0.053</td><td>0.058</td><td>0.185</td><td>0.113</td><td>0.170</td></tr><tr><td>CLIP B32 score 2% to 30%</td><td>3.6M</td><td>0.056</td><td>0.059</td><td>0.173</td><td>0.120</td><td>0.161</td></tr><tr><td>CLIP B32 score 5% to 30%</td><td>3.2M</td><td>0.052</td><td>0.055</td><td>0.177</td><td>0.115</td><td>0.169</td></tr><tr><td>CLIP L14 score top 1%</td><td>128K</td><td>0.002</td><td>0.007</td><td>0.111</td><td>0.050</td><td>0.080</td></tr><tr><td>CLIP L14 score top 3%</td><td>386K</td><td>0.004</td><td>0.009</td><td>0.110</td><td>0.052</td><td>0.088</td></tr><tr><td>CLIP L14 score top 10%</td><td>1.3M</td><td>0.021</td><td>0.033</td><td>0.131</td><td>0.075</td><td>0.119</td></tr><tr><td>CLIP L14 score top 20%</td><td>2.6M</td><td>0.042</td><td>0.051</td><td>0.165</td><td>0.100</td><td>0.151</td></tr><tr><td>CLIP L14 score top 30%</td><td>3.8M</td><td>0.051</td><td>0.055</td><td>0.190</td><td>0.119</td><td>0.173</td></tr><tr><td>CLIP L14 score top 40%</td><td>5.1M</td><td>0.050</td><td>0.054</td><td>0.173</td><td>0.119</td><td>0.168</td></tr><tr><td>CLIP L14 score top 50%</td><td>6.4M</td><td>0.045</td><td>0.052</td><td>0.164</td><td>0.122</td><td>0.160</td></tr><tr><td>CLIP L14 score top 75%</td><td>9.6M</td><td>0.035</td><td>0.043</td><td>0.164</td><td>0.120</td><td>0.151</td></tr><tr><td>CLIP L14 score top 90%</td><td>11.5M</td><td>0.031</td><td>0.038</td><td>0.154</td><td>0.116</td><td>0.144</td></tr><tr><td>Image-based clustering (ImageNet1k)</td><td>2.9M</td><td>0.043</td><td>0.047</td><td>0.178</td><td>0.121</td><td>0.159</td></tr><tr><td>Image-based clustering (ImageNet21k)</td><td>4.5M</td><td>0.035</td><td>0.045</td><td>0.154</td><td>0.122</td><td>0.148</td></tr><tr><td>Image-based sampling, α=0</td><td>12.8M</td><td>0.019</td><td>0.030</td><td>0.144</td><td>0.095</td><td>0.127</td></tr><tr><td>Image-based sampling, α=0.2</td><td>12.8M</td><td>0.031</td><td>0.036</td><td>0.133</td><td>0.100</td><td>0.131</td></tr><tr><td>Image-based sampling, α=0.5</td><td>12.8M</td><td>0.032</td><td>0.038</td><td>0.129</td><td>0.096</td><td>0.125</td></tr><tr><td>Image-based sampling, α=1</td><td>12.8M</td><td>0.021</td><td>0.028</td><td>0.128</td><td>0.078</td><td>0.116</td></tr><tr><td>Image-based sampling, α=2</td><td>12.8M</td><td>0.011</td><td>0.017</td><td>0.116</td><td>0.065</td><td>0.099</td></tr><tr><td>ImageNet distance (L14, top 30%) and English</td><td>2.0M</td><td>0.031</td><td>0.039</td><td>0.163</td><td>0.103</td><td>0.145</td></tr><tr><td>ImageNet distance (L14, top 20%)</td><td>2.6M</td><td>0.030</td><td>0.035</td><td>0.155</td><td>0.102</td><td>0.136</td></tr><tr><td>ImageNet distance (L14, top 30%)</td><td>3.9M</td><td>0.034</td><td>0.041</td><td>0.151</td><td>0.106</td><td>0.139</td></tr><tr><td>ImageNet distance (L14, top 40%)</td><td>5.1M</td><td>0.036</td><td>0.040</td><td>0.151</td><td>0.118</td><td>0.143</td></tr><tr><td>Text-based clustering (ImageNet1k)</td><td>427K</td><td>0.009</td><td>0.016</td><td>0.120</td><td>0.056</td><td>0.096</td></tr><tr><td>Text-based clustering (ImageNet21k)</td><td>3.2M</td><td>0.046</td><td>0.052</td><td>0.169</td><td>0.125</td><td>0.157</td></tr><tr><td>Text-based sampling with average score, α=0</td><td>12.8M</td><td>0.011</td><td>0.020</td><td>0.128</td><td>0.079</td><td>0.112</td></tr><tr><td>Text-based sampling with average score, α=0.5</td><td>12.8M</td><td>0.023</td><td>0.035</td><td>0.127</td><td>0.092</td><td>0.128</td></tr><tr><td>Text-based sampling with average score, α=1</td><td>12.8M</td><td>0.040</td><td>0.044</td><td>0.163</td><td>0.115</td><td>0.155</td></tr><tr><td>Text-based sampling with average score, α=1.2</td><td>12.8M</td><td>0.038</td><td>0.045</td><td>0.150</td><td>0.112</td><td>0.143</td></tr><tr><td>Text-based sampling with max score, α=0</td><td>12.8M</td><td>0.012</td><td>0.020</td><td>0.126</td><td>0.074</td><td>0.107</td></tr><tr><td>Text-based sampling with max score, α=0.5</td><td>12.8M</td><td>0.025</td><td>0.033</td><td>0.134</td><td>0.093</td><td>0.129</td></tr><tr><td>Text-based sampling with max score, α=1</td><td>12.8M</td><td>0.040</td><td>0.046</td><td>0.159</td><td>0.116</td><td>0.150</td></tr><tr><td>Text-based sampling with max score, α=1.2</td><td>12.8M</td><td>0.040</td><td>0.050</td><td>0.161</td><td>0.113</td><td>0.152</td></tr><tr><td>Intersect IN1k image clustering and CLIP B32 score top 30%</td><td>1.4M</td><td>0.049</td><td>0.053</td><td>0.150</td><td>0.103</td><td>0.148</td></tr><tr><td>Intersect IN1k image clustering and CLIP L14 score top 30%</td><td>1.4M</td><td>0.039</td><td>0.045</td><td>0.162</td><td>0.094</td><td>0.145</td></tr><tr><td>Intersect IN21k image clustering and CLIP B32 score top 30%</td><td>2.1M</td><td>0.052</td><td>0.057</td><td>0.179</td><td>0.112</td><td>0.167</td></tr><tr><td>Intersect IN21k image clustering and CLIP L14 score top 30%</td><td>2.1M</td><td>0.047</td><td>0.053</td><td>0.176</td><td>0.110</td><td>0.163</td></tr></table>

Table 26: Baseline results for the filtering track, medium scale. 

<table><tr><td>Filtering</td><td>Training dataset size</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td>No filtering</td><td>128M</td><td>0.176</td><td>0.152</td><td>0.259</td><td>0.219</td><td>0.258</td></tr><tr><td>Random subset (75%)</td><td>96.0M</td><td>0.175</td><td>0.154</td><td>0.265</td><td>0.219</td><td>0.257</td></tr><tr><td>Random subset (50%)</td><td>64.0M</td><td>0.171</td><td>0.151</td><td>0.258</td><td>0.216</td><td>0.252</td></tr><tr><td>Random subset (25%)</td><td>32.0M</td><td>0.155</td><td>0.136</td><td>0.246</td><td>0.203</td><td>0.240</td></tr><tr><td>Random subset (10%)</td><td>12.8M</td><td>0.107</td><td>0.095</td><td>0.210</td><td>0.144</td><td>0.200</td></tr><tr><td>Random subset (1%)</td><td>1.3M</td><td>0.009</td><td>0.017</td><td>0.102</td><td>0.065</td><td>0.090</td></tr><tr><td>Caption length</td><td>87.5M</td><td>0.199</td><td>0.172</td><td>0.275</td><td>0.236</td><td>0.275</td></tr><tr><td>Image size</td><td>77.8M</td><td>0.189</td><td>0.163</td><td>0.248</td><td>0.231</td><td>0.259</td></tr><tr><td>English (fasttext)</td><td>63.0M</td><td>0.214</td><td>0.182</td><td>0.290</td><td>0.246</td><td>0.285</td></tr><tr><td>English (fasttext) and caption length</td><td>47.8M</td><td>0.226</td><td>0.193</td><td>0.284</td><td>0.251</td><td>0.285</td></tr><tr><td>English (fasttext), caption length, and image size</td><td>29.8M</td><td>0.226</td><td>0.193</td><td>0.297</td><td>0.253</td><td>0.294</td></tr><tr><td>English (cld3)</td><td>25.6M</td><td>0.200</td><td>0.175</td><td>0.296</td><td>0.235</td><td>0.279</td></tr><tr><td>English (cld3) and caption length</td><td>22.9M</td><td>0.204</td><td>0.175</td><td>0.287</td><td>0.243</td><td>0.278</td></tr><tr><td>English (cld3), caption length, and image size</td><td>14.6M</td><td>0.179</td><td>0.159</td><td>0.243</td><td>0.216</td><td>0.247</td></tr><tr><td>CLIP B32 score top 1%</td><td>1.3M</td><td>0.025</td><td>0.037</td><td>0.140</td><td>0.076</td><td>0.126</td></tr><tr><td>CLIP B32 score top 3%</td><td>3.9M</td><td>0.093</td><td>0.096</td><td>0.205</td><td>0.128</td><td>0.188</td></tr><tr><td>CLIP B32 score top 10%</td><td>12.8M</td><td>0.231</td><td>0.199</td><td>0.305</td><td>0.206</td><td>0.298</td></tr><tr><td>CLIP B32 score top 20%</td><td>25.7M</td><td>0.279</td><td>0.234</td><td>0.337</td><td>0.241</td><td>0.330</td></tr><tr><td>CLIP B32 score top 30%</td><td>38.4M</td><td>0.285</td><td>0.240</td><td>0.355</td><td>0.253</td><td>0.338</td></tr><tr><td>CLIP B32 score top 40%</td><td>51.3M</td><td>0.273</td><td>0.227</td><td>0.333</td><td>0.257</td><td>0.324</td></tr><tr><td>CLIP B32 score top 50%</td><td>64.0M</td><td>0.256</td><td>0.219</td><td>0.322</td><td>0.259</td><td>0.316</td></tr><tr><td>CLIP B32 score top 75%</td><td>96.1M</td><td>0.211</td><td>0.180</td><td>0.301</td><td>0.238</td><td>0.290</td></tr><tr><td>CLIP B32 score top 90%</td><td>115M</td><td>0.189</td><td>0.165</td><td>0.279</td><td>0.229</td><td>0.274</td></tr><tr><td>CLIP B32 threshold at 0.3 + English filter</td><td>9.4M</td><td>0.208</td><td>0.184</td><td>0.292</td><td>0.210</td><td>0.276</td></tr><tr><td>CLIP B32 threshold at 0.28 + English filter</td><td>13.0M</td><td>0.230</td><td>0.198</td><td>0.307</td><td>0.233</td><td>0.292</td></tr><tr><td>CLIP B32 threshold at 0.3</td><td>25.9M</td><td>0.282</td><td>0.233</td><td>0.340</td><td>0.243</td><td>0.333</td></tr><tr><td>CLIP B32 score 1% to 30%</td><td>37.1M</td><td>0.287</td><td>0.238</td><td>0.347</td><td>0.253</td><td>0.334</td></tr><tr><td>CLIP B32 score 2% to 30%</td><td>35.9M</td><td>0.288</td><td>0.238</td><td>0.338</td><td>0.248</td><td>0.330</td></tr><tr><td>CLIP B32 score 5% to 30%</td><td>32.0M</td><td>0.281</td><td>0.230</td><td>0.352</td><td>0.254</td><td>0.339</td></tr><tr><td>CLIP L14 score top 1%</td><td>1.3M</td><td>0.014</td><td>0.025</td><td>0.136</td><td>0.062</td><td>0.109</td></tr><tr><td>CLIP L14 score top 3%</td><td>3.9M</td><td>0.065</td><td>0.077</td><td>0.176</td><td>0.103</td><td>0.160</td></tr><tr><td>CLIP L14 score top 10%</td><td>12.8M</td><td>0.198</td><td>0.183</td><td>0.283</td><td>0.188</td><td>0.277</td></tr><tr><td>CLIP L14 score top 20%</td><td>25.7M</td><td>0.260</td><td>0.225</td><td>0.326</td><td>0.235</td><td>0.322</td></tr><tr><td>CLIP L14 score top 30%</td><td>38.4M</td><td>0.273</td><td>0.230</td><td>0.338</td><td>0.251</td><td>0.328</td></tr><tr><td>CLIP L14 score top 40%</td><td>51.2M</td><td>0.262</td><td>0.226</td><td>0.330</td><td>0.260</td><td>0.327</td></tr><tr><td>CLIP L14 score top 50%</td><td>64.1M</td><td>0.254</td><td>0.218</td><td>0.322</td><td>0.262</td><td>0.315</td></tr><tr><td>CLIP L14 score top 75%</td><td>96.1M</td><td>0.212</td><td>0.180</td><td>0.287</td><td>0.242</td><td>0.285</td></tr><tr><td>CLIP L14 score top 90%</td><td>115M</td><td>0.188</td><td>0.164</td><td>0.258</td><td>0.225</td><td>0.266</td></tr><tr><td>Image-based clustering (ImageNet1k)</td><td>29.2M</td><td>0.268</td><td>0.213</td><td>0.319</td><td>0.256</td><td>0.312</td></tr><tr><td>Image-based clustering (ImageNet21k)</td><td>45.1M</td><td>0.238</td><td>0.198</td><td>0.304</td><td>0.252</td><td>0.312</td></tr><tr><td>Image-based sampling, α=0</td><td>128M</td><td>0.170</td><td>0.150</td><td>0.266</td><td>0.209</td><td>0.254</td></tr><tr><td>Image-based sampling, α=0.2</td><td>128M</td><td>0.249</td><td>0.193</td><td>0.292</td><td>0.221</td><td>0.284</td></tr><tr><td>Image-based sampling, α=0.5</td><td>128M</td><td>0.269</td><td>0.196</td><td>0.301</td><td>0.216</td><td>0.284</td></tr><tr><td>Image-based sampling, α=1</td><td>128M</td><td>0.207</td><td>0.145</td><td>0.264</td><td>0.166</td><td>0.239</td></tr><tr><td>Image-based sampling, α=2</td><td>128M</td><td>0.118</td><td>0.082</td><td>0.207</td><td>0.110</td><td>0.180</td></tr><tr><td>ImageNet distance (L14, top 30%) and English</td><td>19.8M</td><td>0.212</td><td>0.158</td><td>0.272</td><td>0.178</td><td>0.259</td></tr><tr><td>ImageNet distance (L/14, top 20%)</td><td>25.8M</td><td>0.193</td><td>0.138</td><td>0.276</td><td>0.176</td><td>0.252</td></tr><tr><td>ImageNet distance (L/14, top 30%)</td><td>38.5M</td><td>0.212</td><td>0.159</td><td>0.283</td><td>0.201</td><td>0.269</td></tr><tr><td>ImageNet distance (L/14, top 40%)</td><td>51.3M</td><td>0.212</td><td>0.165</td><td>0.273</td><td>0.212</td><td>0.270</td></tr><tr><td>Text-based clustering (ImageNet1k)</td><td>4.3M</td><td>0.099</td><td>0.090</td><td>0.173</td><td>0.109</td><td>0.166</td></tr><tr><td>Text-based clustering (ImageNet21k)</td><td>31.7M</td><td>0.255</td><td>0.215</td><td>0.328</td><td>0.249</td><td>0.307</td></tr><tr><td>Text-based sampling with average score, α=0</td><td>128M</td><td>0.136</td><td>0.110</td><td>0.213</td><td>0.140</td><td>0.209</td></tr><tr><td>Text-based sampling with average score, α=0.5</td><td>128M</td><td>0.222</td><td>0.178</td><td>0.273</td><td>0.206</td><td>0.269</td></tr><tr><td>Text-based sampling with average score, α=1</td><td>128M</td><td>0.245</td><td>0.204</td><td>0.302</td><td>0.251</td><td>0.293</td></tr><tr><td>Text-based sampling with average score, α=1.2</td><td>128M</td><td>0.231</td><td>0.200</td><td>0.298</td><td>0.240</td><td>0.289</td></tr><tr><td>Text-based sampling with max score, α=0</td><td>128M</td><td>0.140</td><td>0.116</td><td>0.242</td><td>0.138</td><td>0.225</td></tr><tr><td>Text-based sampling with max score, α=0.5</td><td>128M</td><td>0.229</td><td>0.190</td><td>0.290</td><td>0.205</td><td>0.283</td></tr><tr><td>Text-based sampling with max score, α=1</td><td>128M</td><td>0.247</td><td>0.209</td><td>0.300</td><td>0.241</td><td>0.295</td></tr><tr><td>Text-based sampling with max score, α=1.2</td><td>128M</td><td>0.235</td><td>0.200</td><td>0.298</td><td>0.239</td><td>0.290</td></tr><tr><td>Intersect IN1k image clustering and CLIP B32 score top 30%</td><td>14.2M</td><td>0.305</td><td>0.243</td><td>0.342</td><td>0.250</td><td>0.328</td></tr><tr><td>Intersect IN1k image clustering and CLIP L14 score top 30%</td><td>14.0M</td><td>0.297</td><td>0.239</td><td>0.346</td><td>0.231</td><td>0.328</td></tr><tr><td>Intersect IN21k image clustering and CLIP B32 score top 30%</td><td>21.1M</td><td>0.298</td><td>0.244</td><td>0.347</td><td>0.256</td><td>0.336</td></tr><tr><td>Intersect IN21k image clustering and CLIP L14 score top 30%</td><td>20.8M</td><td>0.290</td><td>0.241</td><td>0.339</td><td>0.244</td><td>0.328</td></tr></table>

Table 27: Baseline results for the filtering track, large scale. 

<table><tr><td>Filtering</td><td>Training dataset size</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td>No filtering</td><td>1.28B</td><td>0.459</td><td>0.378</td><td>0.426</td><td>0.419</td><td>0.437</td></tr><tr><td>Random subset (75%)</td><td>960M</td><td>0.456</td><td>0.379</td><td>0.435</td><td>0.415</td><td>0.442</td></tr><tr><td>Random subset (50%)</td><td>640M</td><td>0.453</td><td>0.377</td><td>0.427</td><td>0.413</td><td>0.433</td></tr><tr><td>Random subset (25%)</td><td>320M</td><td>0.447</td><td>0.373</td><td>0.424</td><td>0.407</td><td>0.434</td></tr><tr><td>Random subset (10%)</td><td>128M</td><td>0.426</td><td>0.350</td><td>0.417</td><td>0.396</td><td>0.442</td></tr><tr><td>Random subset (1%)</td><td>12.8M</td><td>0.135</td><td>0.118</td><td>0.219</td><td>0.135</td><td>0.218</td></tr><tr><td>Caption length</td><td>874M</td><td>0.474</td><td>0.392</td><td>0.438</td><td>0.443</td><td>0.445</td></tr><tr><td>Image size</td><td>777M</td><td>0.466</td><td>0.375</td><td>0.421</td><td>0.438</td><td>0.429</td></tr><tr><td>English (fasttext)</td><td>630M</td><td>0.500</td><td>0.414</td><td>0.449</td><td>0.460</td><td>0.462</td></tr><tr><td>English (fasttext), caption length, and image size</td><td>298M</td><td>0.516</td><td>0.423</td><td>0.446</td><td>0.480</td><td>0.458</td></tr><tr><td>English (cld3)</td><td>256M</td><td>0.486</td><td>0.405</td><td>0.462</td><td>0.472</td><td>0.458</td></tr><tr><td>CLIP B32 score top 10%</td><td>128M</td><td>0.543</td><td>0.440</td><td>0.471</td><td>0.435</td><td>0.483</td></tr><tr><td>CLIP B32 score top 20%</td><td>257M</td><td>0.578</td><td>0.465</td><td>0.516</td><td>0.463</td><td>0.515</td></tr><tr><td>CLIP B32 score top 30%</td><td>384M</td><td>0.578</td><td>0.466</td><td>0.525</td><td>0.475</td><td>0.527</td></tr><tr><td>CLIP B32 score top 40%</td><td>512M</td><td>0.560</td><td>0.454</td><td>0.512</td><td>0.478</td><td>0.511</td></tr><tr><td>CLIP B32 score top 50%</td><td>640M</td><td>0.546</td><td>0.450</td><td>0.504</td><td>0.484</td><td>0.505</td></tr><tr><td>CLIP B32 threshold at 0.3 + English filter</td><td>94.3M</td><td>0.553</td><td>0.447</td><td>0.511</td><td>0.482</td><td>0.502</td></tr><tr><td>CLIP B32 threshold at 0.28 + English filter</td><td>130M</td><td>0.553</td><td>0.453</td><td>0.510</td><td>0.495</td><td>0.501</td></tr><tr><td>CLIP B32 threshold at 0.3</td><td>258M</td><td>0.579</td><td>0.464</td><td>0.501</td><td>0.465</td><td>0.505</td></tr><tr><td>CLIP L14 score top 10%</td><td>128M</td><td>0.528</td><td>0.444</td><td>0.482</td><td>0.413</td><td>0.486</td></tr><tr><td>CLIP L14 score top 20%</td><td>257M</td><td>0.570</td><td>0.466</td><td>0.524</td><td>0.455</td><td>0.521</td></tr><tr><td>CLIP L14 score top 30%</td><td>384M</td><td>0.578</td><td>0.474</td><td>0.538</td><td>0.466</td><td>0.529</td></tr><tr><td>CLIP L14 score top 40%</td><td>512M</td><td>0.564</td><td>0.462</td><td>0.533</td><td>0.468</td><td>0.529</td></tr><tr><td>CLIP L14 score top 50%</td><td>641M</td><td>0.548</td><td>0.455</td><td>0.539</td><td>0.469</td><td>0.528</td></tr><tr><td>Image-based clustering (ImageNet1k)</td><td>294M</td><td>0.572</td><td>0.454</td><td>0.483</td><td>0.481</td><td>0.481</td></tr><tr><td>Image-based clustering (ImageNet21k)</td><td>450M</td><td>0.527</td><td>0.433</td><td>0.468</td><td>0.463</td><td>0.471</td></tr><tr><td>Text-based clustering (ImageNet1k)</td><td>42.7M</td><td>0.419</td><td>0.355</td><td>0.340</td><td>0.309</td><td>0.361</td></tr><tr><td>Text-based clustering (ImageNet21k)</td><td>317M</td><td>0.561</td><td>0.465</td><td>0.465</td><td>0.479</td><td>0.476</td></tr><tr><td>Intersect IN1k image clustering and CLIP B32 score top 30%</td><td>143M</td><td>0.632</td><td>0.498</td><td>0.525</td><td>0.504</td><td>0.528</td></tr><tr><td>Intersect IN1k image clustering and CLIP L14 score top 30%</td><td>140M</td><td>0.631</td><td>0.508</td><td>0.546</td><td>0.498</td><td>0.537</td></tr><tr><td>Intersect IN21k image clustering and CLIP B32 score top 30%</td><td>211M</td><td>0.605</td><td>0.481</td><td>0.531</td><td>0.494</td><td>0.519</td></tr><tr><td>Intersect IN21k image clustering and CLIP L14 score top 30%</td><td>208M</td><td>0.506</td><td>0.416</td><td>0.466</td><td>0.424</td><td>0.471</td></tr></table>

Table 28: Baseline results for the filtering track, xlarge scale. 

<table><tr><td>Filtering</td><td>Training dataset size</td><td>ImageNet</td><td>ImageNet dist. shifts</td><td>VTAB</td><td>Retrieval</td><td>Average over 38 datasets</td></tr><tr><td>No filtering</td><td>12.8B</td><td>0.723</td><td>0.612</td><td>0.611</td><td>0.569</td><td>0.621</td></tr><tr><td>CLIP B32 score top 30%</td><td>3.84B</td><td>0.764</td><td>0.640</td><td>0.628</td><td>0.599</td><td>0.638</td></tr><tr><td>CLIP B32 threshold at 0.28 + English filter</td><td>1.3B</td><td>0.755</td><td>0.637</td><td>0.624</td><td>0.620</td><td>0.636</td></tr><tr><td>CLIP L14 score top 20%</td><td>2.56B</td><td>0.761</td><td>0.649</td><td>0.630</td><td>0.575</td><td>0.636</td></tr><tr><td>CLIP L14 score top 25%</td><td>3.2B</td><td>0.768</td><td>0.656</td><td>0.621</td><td>0.585</td><td>0.637</td></tr><tr><td>CLIP L14 score top 30%</td><td>3.84B</td><td>0.764</td><td>0.655</td><td>0.643</td><td>0.588</td><td>0.650</td></tr><tr><td>Intersect IN1k image clustering and CLIP L14 score top 30%</td><td>1.38B</td><td>0.792</td><td>0.679</td><td>0.652</td><td>0.608</td><td>0.663</td></tr></table>

# S Datasheet

# S.1 Motivation

Q1 For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.

\- The purpose of DATACOMP and the associated COMMONPOOL dataset is to enable study of what makes a strong image-text dataset, which supports a broad range of applications. Prior work mainly focuses on data curation in the context of supervised datasets and smaller scales. For a fuller treatment see Section 2. In our initial release of DATACOMP we focus on 38 downstream image classification and image retrieval tasks. For details see Section 3.5 and Appendix O.

Q2 Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

\- DATACOMP and COMMONPOOL were created by a group of researchers with the following affiliations, listed in alphabetical order: Allen Institute for Artificial Intelligence (AI2), Apple, Columbia University, Google Research, Graz University of Technology, Hebrew University, Juelich Supercomputing Center, LAION, Research Center Juelich, StabilityAI, Tel Aviv University, University of Illinois Urbana-Champaign, University of Texas at Austin, University of Washington.

Q3 Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.

\- Compute for this research was generously provided by StabilityAI. For more specific acknowledgments, see the acknowledgment section at the end of the main paper.

Q4 Any other comments?

\- We hope that COMMONPOOL will help to facilitate data-centric questions in ML and AI towards the next generation of web-scale datasets, that 1) yield higher accuracy models and 2) models that are safer and more equitable.

# S.2 Composition

Q5 What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.

\- Each instance is a pair of url and corresponding image alt-text. The url points to an image that a user can then try to download. Each sample is also tagged with metadata, discussed in Q25.

Q6 How many instances are there in total (of each type, if appropriate)?

\- There are 12.8B instances in COMMONPOOL. For breakdowns and statistics see Appendix I.

Q7 Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).

\- We find $\sim 88\mathrm{B}$ possible samples in common crawl. These samples are globally shuffled to ensure i.i.d. sampling for all sampling based parts of the downstream pipeline. Of these samples we attempt to download $\sim 40\mathrm{B}$ samples. Due to various download issues, such as dead links and throttling, we are able to successfully download $\sim 16.8\mathrm{B}$ samples. After NSFW filtering and evaluation set deduplication we end up with $\sim 13.1\mathrm{B}$ viable samples, from which we randomly sample $12.8\mathrm{B}$ for COMMONPOOL. For a complete treatment and visualization of our data processing funnel, see Appendix H. For each sample we also release metadata shown in Table 8.

Q8 What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.

\- Each sample contains an image url for download and an associated alt-text caption. Additionally, each sample contains metadata fields shown in Table 8 (e.g., image aspect ratio and CLIP features).

Q9 Is there a label or target associated with each instance? If so, please provide a description.

\- We do not provide any category labels; however, the text associated with each image can be considered a soft, noisy label for each sample. Such labels are common in modern image-text training paradigms (e.g., image-text representation alignment, image captioning objectives, text-conditional image generation objectives, etc.).

Q10 Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.

\- No, each sample is an image-text pair.

Q11 Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? If so, please describe how these relationships are made explicit.

\- No, the dataset is released as it is with no explicit attempt to establish relationships between instances.

Q12 Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.

\- No. The test tasks are existing image classification tasks. We run a deduplication model to try to prevent test set contamination in COMMONPOOL.

Q13 Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.

\- COMMONPOOL is sourced from Common Crawl, which can be thought of as a snapshot of the internet. Hence, there can be considerable noise (e.g., alt-text being unrelated to its associated image), duplicate data, etc.

Q14 Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.

\- The data is not self-contained and rather links other external resources on the internet. Links point to resources distributed across the internet. There is no guarantee that the resources will exist in perpetuity or that the resources will not change. To mitigate against data poisoning in future COMMONPOOL downloads, we release SHA256 hashes of images. Due to the size of the dataset, it is not possible to provide it in an archival form.

Q15 Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals' non-public communications)? If so, please provide a description.

\- The dataset is comprised of data that was readily available on the public internet at the time of our download. However, it is possible that the dataset contains confidential information (e.g., private data that is hosted publicly for nefarious reasons or out of ignorance of said data being confidential).

Q16 Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.

\- Considering the plurality of people and their backgrounds across the world, it is highly likely that there is content in COMMONPOOL that may upset people. Common Crawl scrapes the internet, which has pornographic, hateful, racist, sexist, and otherwise abhorrent and toxic material. While we attempt to do thorough NSFW filtering, these methods are not 100% accurate. At the 12.8B scale at which we operate, it is highly likely that there is still toxic content in the dataset. We consider the dataset as a research artifact and hope future work will look critically at COMMONPOOL in the hopes of developing even better safety filters.

Q17 Does the dataset relate to people? If not, you may skip the remaining questions in this section.

\- People may appear in the dataset; however, in an effort to preserve privacy, our downloading tooling automatically blurs all detected faces in COMMONPOOL images.

Q18 Does the dataset identify any subpopulations (e.g., by age, gender)?

\- While COMMONPOOL does not explicitly identify subpopulations in its metadata, it is plausible to extract such information for some images using the corresponding textual caption.

Q19 Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.

\- We conjecture that even with our face blurring procedure, it may still be possible to identify individuals. Face blurring relies on a face detection model, which could fail (See Appendix G for experimental validation of the employed detector). It is also possible to identify certain celebrities or athletes, who may wear distinctive clothing that is associated with them. It is also likely that names are contained in textual captions, though it is not guaranteed that these names correspond to people in images due to the inherent noisiness of internet captions.

Q20 Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.

\- Yes. COMMONPOOL is created using images and corresponding alt-text that are available on the public internet. Given the 12.8B scale of COMMONPOOL, it is highly likely that there is sensitive data in the dataset. To mitigate against making sensitive content more accessible, we 1) run NSFW image filtering and 2) NSFW text filtering when generating COMMONPOOL, discarding all samples that are flagged. Additionally we 3) provide automatic face blurring in our COMMONPOOL download scripts to blur all detected faces.

Q21 Any other comments?

\- COMMONPOOL is a research artifact, and we hope it will be useful for those studying how to make internet-scale datasets safer.

# S.3 Collection Process

Q22 How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.

• Data is directly downloaded from the public internet.

Q23 What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?

\- We iterate on the LAION-5B data collection process, making an effort to emphasize safety. We ran python based processing scripts to parse Common Crawl dumps, download images, filter our NSFW content, deduplicate samples against downstream tests sets, blur faces, and compute CLIP features. We ran processes on 100s of AWS CPU nodes for Common Crawl parsing and data download. Other steps were run on one of StabilityAI's GPU cluster. For software links see Q37. For software validation related to NSFW content filtering and face blurring see Appendices E and G respectively. In brief, for NSFW image filtering, we validate against commercial APIs and on the NSFW test set introduced in LAION-5B. For face detection (used for face blurring), we evaluate against commercial APIs. We find strong performance for both modules.

Q24 If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

• See Q7.

Q25 Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

\- The researching authors were involved in the data collection as an open source effort. No researchers were compensated specifically for their involvement in this project.

Q26 Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.

\- Data was downloaded between December 2022 and March 2023. The urls are collected from Common Crawl dumps between 2014 and 2022. Common Crawl dumps may include urls from the early days of the internet. Hence, the download/collection timeframe does not match the creation timeframe. Additionally, future users of COMMONPOOL and its subsets will have to download data themselves using our tooling.

Q27 Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.

\- Our dataset collection process iterates on the LAION-5B process, which found IRB review was not necessary as they “do not intervene with the people depicted in the data as well as the data being public.” [129]. Additionally, the NeurIPS ethics review found no serious ethical issues with LAION-5B. We take even more stringent safety measures than the original LAION-5B dataset, in that we filter out data that is flagged as NSFW by our detection pipeline and blur detected faces in COMMONPOOL, automatically in our released download tooling. All this being said, a formal ethics review has not been conducted to date.

Q28 Does the dataset relate to people? If not, you may skip the remaining questions in this section.

\- Yes. People may appear in the dataset. Detected faces are blurred when downloading COMMONPOOL with our tooling.

Q29 Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

\- We collect data from websites across the internet.

Q30 Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.

\- Individuals were not notified about the data collection.

Q31 Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.

\- Following our usage of Common Crawl and https://github.com/rom1504/img2dataset for download images, we respect robots.txt files, which specify parts of websites that a crawler may access. It is, however, possible that images of people, medical images, etc. were uploaded to the internet without a person's consent. To mitigate against such safety concerns we make an effort to do rigorous NSFW filtering and blur all detected faces automatically in our download tooling.

Q32 If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).

\- In conjunction with LAION, we use https://laion.ai/dataset-requests/ to monitor user takedown requests. We will also make an effort to provide a user with the url at which their sensitive content is hosted—if they do not have this information already—, so they can take further action as they see fit (e.g., contacting the host to request that the content is taken down from the internet).

Q33 Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.

\- We conduct a fairness evaluation on models trained on COMMONPOOL and its derivative. See Appendix Q for details. Birhane et al. [15] conduct an extensive study in the context of LAION-400M, which is an image-text dataset also sourced from Common Crawl, finding a plethora of dangerous and unsafe content. Our dataset differs from LAION-400M in that we conduct NSFW preprocessing and face blurring for detected faces. COMMONPOOL only contains samples that pass our NSFW safety checks and our download tooling automatically blurs detected faces. However, since COMMONPOOL is created from the internet, it is still likely that it contains some harmful data.

Q34 Any other comments?

\- We hope that future work will use COMMONPOOL to study how to construct safer, web-scale datasets.

# S.4 Preprocessing, Cleaning, and/or Labeling

Q35 Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.

• Yes. See Q7. For more details see Appendix H.

Q36 Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.

\- Raw data is not available or distributed due to safety considerations. We distribute only urls that are in the dataset on HuggingFace—and not urls of images our preprocessing flagged as NSFW.

Q37 Is the software used to preprocess/clean/label the instances available? If so, please provide a link or other access point.

\- We use the following, open-source software to aid in data processing:

- Apache Spark: https://spark.apache.org   
- Ray: https://www.ray.io   
- img2dataset: https://github.com/rom1504/img2dataset   
- OpenAI CLIP: https://github.com/openai/CLIP   
- Near dedulicate detector: https://github.com/lyakaap/ISC21-Descriptor-Track-1st   
- Face detector: https://github.com/deepinsight/insightface   
- Detoxify, for detecting toxic language: https://github.com/unitaryai/detoxify   
- A modified version of the following NSFW image detector: https://github.com/LAION-AI/CLIP-based-NSFW-Detector. Specifically, we use the dataset used to train this model to train our own 4-layer MLP classifier.

Q38 Any other comments?

\- COMMONPOOL and DATACOMP would not be possible without tools developed by the open-source community.

# S.5 Uses

Q39 Has the dataset been used for any tasks already? If so, please provide a description.

\- The full dataset (and subsets) have been used to train several CLIP models at various scales and compute budgets as presented in our main paper. We evaluate these models zero-shot on 38 downstream image classification and retrieval tasks. See Section 3.5 and Appendix O for more details.

Q40 Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.

\- No. However, there is a leaderboard associated with DATACOMP. Interested parties can investigate the submissions and further study publications that make use of our data. See: https://www.datacomp.ai/leaderboard.html.

Q41 What (other) tasks could the dataset be used for?

\- The dataset could also be used for training image captioning models and language-conditional image generation models. Note: generative image models trained on COMMONPOOL are not expected to generate recognizable human faces as our download tooling automatically blurs detected faces. COMMONPOOL could be used for sociological studies, for example, examining societal biases or to better understand what is on the public internet.

Q42 Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?

\- COMMONPOOL and its derivatives are not intended for production ready products, including but not limited to those related to race, gender identity or expression, ethnicity, sexual orientation, age, socioeconomic status, disability, religion, national origin or creed. COMMONPOOL is not suitable for any software that makes decisions involving people. COMMONPOOL is collected from the internet and hence reflects many of the biases, unfairness, and stereotypes currently existing in our societies. COMMONPOOL is intended as a research artifact to study multimodal dataset curation and the effect of data curation strategies on downstream models.

Q43 Are there tasks for which the dataset should not be used? If so, please provide a description.

\- COMMONPOOL in its current form or the subsets presented in this paper should not be used in software that makes decisions related to people. The known biases (Appendix Q) make deploying software, especially widely decimated production-level products, built on COMMONPOOL incredibly irresponsible. COMMONPOOL is designed as a research artifact for academic exploration. We also do not condone the use of COMMONPOOL in surveillance or military applications.

# Q44 Any other comments?

\- Our goal with COMMONPOOL and DATACOMP was to put a benchmark in place so the community can start measuring dataset progress along many different axes (e.g., model performance on diverse tasks). We believe this is crucial to develop more performant and safer datasets.

# S.6 Distribution

Q45 Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.

\- Yes. We use HuggingFace datasets for public release.

Q46 How will the dataset be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?

\- The dataset will be distributed via HuggingFace datasets at https://huggingface.co/datasets/mlfoundations/datacomp\_pools/tree/main

Q47 When will the dataset be distributed?

• DATACOMP will be available starting May 2023.

Q48 Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.

\- We distribute the url-text sample and metadata under a standard CC-BY-4.0 licence.

Q49 Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.

• We do not copyright samples in the dataset.

Q50 Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.

\- The dataset is provided as an index of url-text pairs.

Q51 Any other comments?

\- We provide several subsets of COMMONPOOL (between 12.8M samples and the full dataset of 12.8B samples). Hence, it is possible to download and experiment with subset of the data.

# S.7 Maintenance

Q52 Who will be supporting/hosting/maintaining the dataset?

\- HuggingFace currently hosts the url-text pairs and metadata. The DATACOMP team will be responsible for maintaining the dataset.

Q53 How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

• We can be contacted at contact@datacomp.ai.

Q54 Is there an erratum? If so, please provide a link or other access point.

\- Currently there are no errata. If issues are discovered, we will communicate with the public via our website https://datacomp.ai.

Q55 Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?

\- At the present time there is no intention to update COMMONPOOL for scientific reasons. However, we will respond to user takedown requests (see Q56). COMMONPOOL is inherently noisy and the purpose of releasing it is to encourage researchers in the community to study dataset cleaning in the context of image-text samples.

Q56 If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.

\- We will use the following website, https://laion.ai/dataset-requests, for user takedown requests, where “Sample ID” is the sample uid.

Q57 Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users.

\- This is the first version of DATACOMP and the associated COMMONPOOL dataset. We do not intend to maintain deprecated version of COMMONPOOL. We will communicate deprecation notices through our website: https://datacomp.ai.

Q58 If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.

\- All alterations to the dataset will be handled on a case-by-case basis.

Q59 Any other comments?

\- We encourage community members to contact us at contact@datacomp.ai with inquiries related to dataset maintenance.