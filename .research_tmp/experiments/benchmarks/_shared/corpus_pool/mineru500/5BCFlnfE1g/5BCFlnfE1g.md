# DEMYSTIFYING CLIP DATA

Hu Xu $^{1}$ Saining Xie $^{2}$ Xiaoqing Ellen Tan $^{1}$ Po-Yao Huang $^{1}$ Russell Howes $^{1}$ Vasu Sharma $^{1}$ Shang-Wen Li $^{1}$ Gargi Ghosh $^{1}$ Luke Zettlemoyer $^{1,3}$ Christoph Feichtenhofer $^{1}$ $^{1}$ FAIR, Meta AI $^{2}$ New York University $^{3}$ University of Washington

Project Leads

# ABSTRACT

Contrastive Language-Image Pre-training (CLIP) is an approach that has advanced research and applications in computer vision, fueling modern recognition systems and generative models. We believe that the main ingredient to the success of CLIP is its data and not the model architecture or pre-training objective. However, CLIP only provides very limited information about its data and how it has been collected, leading to works that aim to reproduce CLIP's data by filtering with its model parameters. In this work, we intend to reveal CLIP's data curation approach and in our pursuit of making it open to the community introduce Metadata-Curated Language-Image Pre-training (MetaCLIP). MetaCLIP takes a raw data pool and metadata (derived from CLIP's concepts) and yields a balanced subset over the metadata distribution. Our experimental study rigorously isolates the model and training settings, concentrating solely on data. MetaCLIP applied to CommonCrawl with 400M image-text data pairs outperforms CLIP's data on multiple standard benchmarks. In zero-shot ImageNet classification, MetaCLIP achieves 70.8% accuracy, surpassing CLIP's 68.3% on ViT-B models. Scaling to 1B data, while maintaining the same training budget, attains 72.4%. Our observations hold across various model sizes, exemplified by ViT-bigG producing 82.1%. Curation code and training data distribution over metadata is available at https://github.com/facebookresearch/MetaCLIP.

# 1 INTRODUCTION

Deep learning has revolutionized the field of artificial intelligence, and pre-trained models have played a pivotal role in democratizing access to cutting-edge AI capabilities. However, the training data used to create these models is often concealed from the public eye, shrouded in secrecy.

The increasing availability of pre-trained models for public use contrasts sharply with the lack of transparency regarding their training data. Further, proprietary concerns, such as copyright issues, often limit access to the original data sources. Consequently, the need to explore novel approaches for curating high-quality training data that can be shared openly arises.

In the vision-language domain, the dominant model and learning approach is Contrastive Language-Image Pre-training (CLIP) (Radford et al., 2021), a simple technique to learn from image-text pairs. We believe that the secret to the dominance of CLIP models is attributed to its high-quality WIT400M dataset which is curated from the web. Despite its popularity, the specifics of CLIP's curation process have remained a mystery, captivating the research community for years.

Follow-up works (Schuhmann et al., 2022; 2021) have attempted to replicate CLIP's data, but with a notable difference in their curation method. While CLIP generates data based on its unknown data source and curation methodology, these approaches remove noise by applying the CLIP model as a hard blackbox filter which in turn is a form of distilling WIT400M information captured in CLIP.

The advantages of CLIP's curation are apparent. First, it starts from scratch, avoiding the introduction of biases through filters. Second, CLIP's curation process balances the data distribution over

metadata, maximizing signal preservation while mitigating, rather than removing, noise in the data $^{1}$ . Such distribution lays the groundwork for task-agnostic data, a crucial part of foundation models.

In this paper, we attempt to reveal CLIP's method around training data curation. We present an empirical study on data curation, with frozen model architecture and training schedule. We focus solely on the impact of training data, excluding other factors that could confound the results. We make several observations for good data quality and present a simple algorithm to make CLIP's curation more transparent. Consequently, we shed light on both the curation process and the resulting training data distribution. Our algorithm enables easy adaptation to different data pools, allowing parties to fully own their data pipeline without relying on blackbox filters from external providers.

Our algorithm takes a raw data pool $\mathcal{D}$ and metadata $\mathcal{M}$ (derived from CLIP's queries or visual concepts) and yields a balanced subset $\mathcal{D}^*$ over $\mathcal{M}$ : $\mathcal{D}^* \leftarrow f(\mathcal{D};\mathcal{M})$ . Our approach, named Metadata-Curated Language-Image Pretraining (MetaCLIP), marks a significant step towards making the curation process more transparent and accessible.

MetaCLIP applied to CommonCrawl (CC) with 400M data points outperforms CLIP on multiple standard benchmarks. In terms of zero-shot ImageNet classification, using ViT (Dosovitskiy et al., 2020) models of various sizes. Our MetaCLIP achieves 70.8% vs CLIP's 68.3% on ViT-

B and 76.2% vs 75.5% on ViT-L. Scaling to 2.5B data, with the same training budget and similar distribution boosts this to unprecedented accuracy of 79.2% for ViT-L, 80.5% for ViT-H and 82.1% for ViT-bigG in the vanilla training setting (not using any external data, models, or longer training).

In Fig.1, we show the impact of metadata curation on ImageNet validation plotted over training steps. First, we are training on Raw English data from the web (400M image-text pairs, 57.4% accuracy), after applying Language IDentification (LID) to the random Raw set ( $\sim$ 1.1B pairs, 54.1%). Using metadata to curate the training set (MetaCLIP 400M w/o bal, 60.8%) performs significantly better than these baselines, and using balancing significantly increases accuracy further (MetaCLIP, 65.5%), outperforming similar datasets, WIT400M from CLIP, 63.4% and LAION 400M, 60.0%(Schuhmann et al., 2021).

# 2 RELATED WORK

The training data of CLIP differs significantly from a traditional supervised dataset (Gadre et al., 2023) in various aspects. Firstly, it involves large-scale training with mixed-quality image-text pairs rather than categorized images with human annotated labels, as commonly seen in classification datasets. Secondly, CLIP's pre-training is the initial stage of training, assuming no access to previously trained models.

![](images/c6e5ed58e14bd2fd236d256f47991de840ac7c57cd32641ee9a96ef5124dba3c.jpg)

<details>
<summary>line</summary>

| Training Steps | CLIP(400M) | LAION(407M) | Raw(1.1B) | Raw English(400M) | MetaCLIP w/o bal.(400M) | MetaCLIP(400M) |
| -------------- | ---------- | ----------- | --------- | ----------------- | ----------------------- | -------------- |
| 0              | 0.32       | 0.32        | 0.32      | 0.32              | 0.32                    | 0.38           |
| 50000          | 0.45       | 0.45        | 0.38      | 0.40              | 0.45                    | 0.52           |
| 100000         | 0.50       | 0.50        | 0.42      | 0.45              | 0.50                    | 0.55           |
| 150000         | 0.52       | 0.52        | 0.45      | 0.48              | 0.52                    | 0.58           |
| 200000         | 0.55       | 0.55        | 0.48      | 0.52              | 0.55                    | 0.61           |
| 250000         | 0.57       | 0.57        | 0.50      | 0.55              | 0.57                    | 0.63           |
| 300000         | 0.59       | 0.59        | 0.52      | 0.57              | 0.59                    | 0.65           |
| 350000         | 0.61       | 0.61        | 0.54      | 0.59              | 0.61                    | 0.66           |
| 400000         | 0.63       | 0.63        | 0.56      | 0.61              | 0.63                    | 0.67           |
</details>

Figure 1: ViT-B/32 on ImageNet zero-shot classification with fixed training steps (12.8B seen pairs and training/validation data has been de-duplicated). Raw: raw CommonCrawl (CC) distribution; Raw English: English only CC; MetaCLIP w/o bal.: curated (substring matched) data pool from CC; MetaCLIP: curated and balanced metadata distribution. Metadata curation boosts performance significantly and balancing is equally important. Our MetaCLIP data significantly outperforms CLIP's WIT400M and LAION data(Schuhmann et al., 2021).

Data Pruning on Established Datasets. Current research on data algorithms primarily revolves around data pruning techniques applied to well-established datasets using pre-trained models (Sorscher et al., 2022; Abbas et al., 2023). These approaches, such as coreset selection techniques (Har-Peled & Mazumdar, 2004; Feldman et al., 2011; Bachem et al., 2015; Mirzasoleiman et al., 2020; Toneva et al., 2018), aim to select a subset of data that yields similar performance to training on the entire dataset. Post-hoc data pruning with model filters has limited utility, if the model is used as a black-box filter that forbids to control biases or improve the filter quality.

Handling Noisy Internet Data. Addressing noisy data from the Internet is a significant challenge, and existing approaches often heavily rely on human-designed filter systems. Classical methods involve dataset cleaning and outlier removal (Jiang et al., 2001; Yu et al., 2002) to discard samples that may introduce undesirable biases to models.

Replicating CLIP's Training Data. Recent efforts, such as LAION (Schuhmann et al., 2021; 2022) and concurrent work DataComp (Gadre et al., 2023), attempt to replicate CLIP's training data. However, they adopt fundamentally different strategies for several reasons. First, the data used in these approaches are post-hoc, filtered, by vanilla CLIP as a teacher model. Second, the curation process in these methods relies on a labor-intensive pipeline of filters, making it challenging to comprehend the resulting data distribution from the raw Internet (refer to the unknown biases of using CLIP filter in (Schuhmann et al., 2022)). Thirdly, the goal is to match the quantity of CLIP's target data size rather than the data distribution itself, which may lead to an underestimation of the data pool size needed to obtain sufficient quality data. Consequently, the performance on the 400M scale is sub-optimal, with LAION400M only achieving $72.77\%$ (Schuhmann et al., 2021) accuracy on ViT-L/14 on ImageNet, whereas vanilla CLIP obtains $75.5\%$ .

Importance of Understanding CLIP's Data Curation. The observations made in these studies underscore the critical importance of understanding how OpenAI CLIP curates its data in the first place. A comprehensive understanding of the curation process can shed light on the factors that contribute to its success, allowing researchers to devise more effective and efficient algorithms for future vision-language pre-training endeavors.

# 3 METACLIP

The original paper (Radford et al., 2021) only provides limited details about how CLIP curates its data. Since important design choices for a direct reproduction are missing, we will clarify our choices in this section. Our goal is to uncover CLIP's data curation process, which involves preserving signal in the data while minimizing noise. In this section, we will explain the principles we have adopted to achieve this, which may differ from CLIP's as these are not known publicly.

CLIP's WIT400M is curated with an information retrieval method, quoting (Radford et al., 2021):

“

To address this, we constructed a new dataset of 400 million (image, text) pairs collected from a variety of publicly available sources on the Internet. To attempt to cover as broad a set of visual concepts as possible, we search for (image, text) pairs as part of the construction process whose text includes one of a set of 500,000 queries We approximately class balance the results by including up to 20,000 (image, text) pairs per query.

”

We rigorously adhere to this description and provide detailed insights into the construction process of CLIP's metadata (in §3.1) $^{2}$ , sub-string matching (in §3.2), inverted indexing (in §3.3), as well as query and balancing (in §3.4).

# 3.1 METADATA CONSTRUCTION: $M = \{entry\}$

We start by re-building CLIP's 500,000-query metadata, citing Radford et al. (2021):

“

The base query list is all words occurring at least 100 times in the English version of Wikipedia. This is augmented with bi-grams with high pointwise mutual information as well as the names of all Wikipedia articles above a certain search volume. Finally all WordNet synsets not already in the query list are added.

”

The metadata ('queries' or 'entries') consists of four components: (1) all synsets of WordNet, (2) uni-grams from the English version of Wikipedia occurring at least 100 times, (3) bi-grams with high pointwise mutual information, and (4) titles of Wikipedia articles above a certain search volume. We rebuild these components from WordNet and Wikipedia and summarize the statistics in Table $1^{3}$ . We estimate the thresholds for components (3) and (4) as in the 3rd column of Table 1, by first choosing a point-wise mutual information threshold of 30 that meets the budget of 100k entries for bi-grams and then fill the rest of the entries with Wikipedia titles.

<table><tr><td>Source</td><td># of Entries</td><td>Desc. of Threshold</td><td>Threshold</td></tr><tr><td>WordNet synsets</td><td>86,654</td><td>N/A</td><td>[ALL] (follow CLIP)</td></tr><tr><td>Wiki uni-gram</td><td>251,465</td><td>Count</td><td>100 (follow CLIP)</td></tr><tr><td>Wiki bi-gram</td><td>100,646</td><td>Pointwise Mutual Info.(PMI)</td><td>30 (estimated)</td></tr><tr><td>Wiki titles</td><td>61,235</td><td>View Frequency</td><td>70 (estimated)</td></tr></table>

Table 1: Composition of MetaCLIP Metadata.

# 3.2 SUB-STRING MATCHING: text → entry

After constructing the metadata, CLIP's curation aligns a pool of image-text pairs with metadata entries through sub-string matching. This process identifies texts that contain any of the metadata entries, effectively associating unstructured texts with structured metadata entries. The sub-string matching step retains only high-quality matching texts, automatically filtering out various types of noises that a typical filter system would consider on a case-by-case basis.

Such alignment is referred to as sub-string matching in Radford et al. (2021):

“

We also restrict this step in CLIP to text-only querying for sub-string matches while most webly supervised work uses standard image search engines ...

”

Image-Text Pair Pool We start by estimating the pool size used by CLIP's curation. CLIP's data source is unknown to us ("a variety of publicly available sources" in Radford et al. (2021)). We adopt CommonCrawl $(\mathrm{CC})^4$ as the source to build such a pool and re-apply sub-string matching to this source. We ended with a pool of 1.6B image-text pairs (5.6B counts of sub-string matches). Note that one text can have multiple matches of entries and we have 3.5 matches per text on average.

As a result, sub-string matching builds the mapping $txt \rightarrow entry$ . This step has two outcomes: (1) low-quality text is dropped; (2) unstructured text now has a structured association with metadata. For all English text, $\sim50\%$ image-text pairs are kept in this stage. Similar to CiT (Xu et al., 2023),

<table><tr><td>Metadata Subset</td><td># of Entries</td><td># of Counts</td></tr><tr><td>Full</td><td>500K</td><td>5.6B</td></tr><tr><td>Counts = 0</td><td>114K</td><td>0</td></tr><tr><td>Counts &gt; 20000</td><td>16K</td><td>5.35B</td></tr></table>

Table 2: Summary of counts for entries.

<table><tr><td>Entry</td><td>Counts</td><td>Entry</td><td>Counts</td><td>Entry</td><td>Counts</td><td>Entry</td><td>Counts</td></tr><tr><td>of</td><td>120M</td><td>in</td><td>107M</td><td>and</td><td>100M</td><td>for</td><td>89M</td></tr><tr><td>the</td><td>87M</td><td>The</td><td>67M</td><td>with</td><td>67M</td><td>to</td><td>61M</td></tr><tr><td>photo</td><td>54M</td><td>a</td><td>50M</td><td>image</td><td>48M</td><td>1</td><td>47M</td></tr><tr><td>on</td><td>45M</td><td>by</td><td>43M</td><td>2</td><td>43M</td><td>Image</td><td>39M</td></tr><tr><td>at</td><td>38M</td><td>Black</td><td>33M</td><td>3</td><td>30M</td><td>A</td><td>29M</td></tr></table>

Table 3: Top-20 entries with counts.

this approach looks for quality matches and automatically gets rid of some type of noise (such as date strings) that a typical filter system would remove consider case-by-case (e.g., regular expression on dates, ids etc.).

# 3.3 INVERTED INDEXING: entry → text

Following sub-string matching, CLIP builds an inverted index of the data pool. All texts associated with each metadata entry are aggregated into lists, creating a mapping from each entry to the corresponding texts, $entry \rightarrow text$ .

As an analysis, we count the number of matches for each entry and summarize that in Table 2. The counts exhibit a long-tailed distribution. Out of the 500k entries, 114k entries have no matches. This signifies the importance of knowing the training data distribution since it is very likely the training data does not have certain visual concepts. We observed that only 16k entries had counts higher than 20k, accounting for only 3.2% (16k/500k) of the entries, but their counts made up 94.5% (5.35B/5.6B) of the total counts of all entries.

Top Entries. We show the top entries of the matching in Table 3. Interestingly, many of these are stopwords, which don't carry specific meaning but can enhance the overall text quality (e.g., by generating grammatically correct sentences rather than just keyword lists). It's important to note that although sub-string matching aims to select only high-quality texts, there are instances where common entries may still include irrelevant texts. For instance, the entry "photo" could match with the popular but unhelpful term "untitled photo". These noise-related issues can be addressed in the subsequent stage of processing.

# 3.4 QUERY AND BALANCING WITH $t \leq 20K$

The key secret behind OpenAI CLIP's curation is to balance the counts of matched entries. For each metadata entry, the associated list of texts (or image-text pairs) is sub-sampled, ensuring that the resulting data distribution is more balanced. This step aims to mitigate noise and diversify the distribution of data points, making the data more task-agnostic as foundation data for pre-training.

The magic number t = 20k is a threshold used to limit the number of texts/pairs for each entry. Entries with fewer than t pairs (tail entries) retain all associated pairs, while entries with more than t pairs (head entries) are sub-sampled to t pairs. The selection is based on the density of information in texts; texts with more matched entries have a higher chance of being curated (recall that the average is 3.5 matches per text).

To study the effect of the magic number $t = 20\mathrm{k}$ , we plot the cumulative sum of counts for entries sorted by counts from tail to head in Fig. 2. Interestingly, the value of $t = 20\mathrm{k}$ seemingly represents the transition from tail to head entries, when the head entries start exhibiting an exponential growth rate. By applying a max count of $t$ , the growth rate of total counts (i.e., the scale of resulting data

points) is reduced to linear. This significantly flattens (and balances) the training data distribution. We further study the optimality of t = 20k for the 400M data scale in our experiments.

![](images/34c729a958e2963f8d492f25ee711210f312575aba79d5d681c58151869ada56.jpg)

<details>
<summary>line</summary>

| Metadata Entries Sorted by Counts | Cumulative Entry Counts |
| ---------------------------------- | ------------------------ |
| t=20k (400M)                       | ~0.7e9                   |
| Pool (1.6B)                        | ~5.8e9                   |
</details>

Figure 2: Cumulative sum of counts on entries from tail to head on a data pool with 1.6B image-text pairs (5.6B match counts). (1) raw/unbalanced cumulative counts, $t = \infty$ ; (2) balanced cumulative counts after applying t = 20k. The limit t defines the transition of tail/head entries.

In summary, balancing yields three interesting outcomes:

(i) It reduces dominance and noise from head entries, like common web terms. E.g., out of 400M pairs, only 20k texts containing “photo” are kept (while there are 54M “photo” instances in the pool).   
(ii) It diversifies the data distribution and balances tail/head entries, leading to a more task-agnostic foundation.   
(iii) Sampling for each entry ensures that data points with more matched entries or denser information are prioritized for curation.

Discussion. CLIP employs a pure NLP-based approach, requiring no access to ML models and minimizing explicit/implicit priors from humans. The metadata plays a central role in mitigating noise and preserving signal in the data distribution. The balancing step effectively flattens the data distribution, diversifying the data and making it more suitable as foundation data for pre-training tasks. We analyze the effects of balancing in Appendix A.3.

# 3.5 A SIMPLE ALGORITHM FOR CURATION

This section presents an algorithm that formalizes the curation process described earlier. The algorithm aims to improve scalability and reduce space complexity for operations across data points, such as inverted indexing and sub-sampling. Instead of building inverted indexes, the algorithm only maintains total counts for each entry.

We assume that CLIP curation constructs an inverted index that maps entries to documents (image-text pairs) to enable efficient search for each entry (“we search for (image-text) pairs” in Radford et al. (2021)). In contrast, our algorithm approaches the balancing process through independent sampling. This avoids the need to build an inverted index that could potentially store hundreds of millions of concrete pairs for popular entries, thereby improving efficiency and scalability.

Algorithm 1: Pseudo-code of Curation Algorithm in Python style (see Sec. A.10 for samples).   
```python
# D: raw image-text pairs;
# M: metadata;
# t: max matches per entry in metadata;
# D_star: curated image-text pairs;

D_star = []

# Part 1: sub-string matching: store entry indexes in text.matched_entry_ids and output counts per entry in entry_count.

entry_count = substr_matching(D, M)
# Part 2: balancing via independent sampling

entry_count[entry_count < t] = t

entry_prob = t / entry_count

for image, text in D:
    for entry_id in text.matched_entry_ids:
    if random.random() < entry_prob[entry_id]:
    D_star.append((image, text))
    break 
```

Our algorithm takes three inputs: metadata M, a data pool D, and a hyper-parameter t. It aims to find a subset $D^{*}$ with a balanced distribution over M, denoted as $\mathcal{D}^{*} \leftarrow f(\mathcal{D};\mathcal{M},t)$ . The algorithm consists of two parts, each corresponding to a specific stage of the curation process.

We provide the Python pseudo-code in Algorithm 1.

Part 1: Entry Counts from Sub-string Matching. This corresponds to Sec. 3.2. The substr\_matching function outputs the total counts of matches per entry, entry\_count, represented as a NumPy array indexed by entry\_id. Each text is associated with matched\_entry\_ids that contains a list of matched entries.

Part 2: Balancing via Independent Sampling. This part corresponds to Sec.3.3 and Sec.3.4 and focuses on balancing counts on entries. Instead of building an expensive inverted index with associated lists of texts for each entry, we sample each data point independently.

We first compute the probability of sampling each entry, entry\_prob, where tail entries (entry\_count < t) have a probability equal to 1, and head entries have a probability less than 1. We iterate through all image-text pairs and sample/curate each pair. When an image-text pair has a matched entry sampled/selected, we include that pair in $\mathcal{D}^*$ .

This procedure is equivalent to CLIP's curation, because if one image-text pair has one or more matched entries, the chance of that pair being selected is determined by the probability of sampling for each individual entry: $t$ /entry\_count[entry\_id]. As long as one entry selects that pair, it will be kept in $\mathcal{D}^*$ . Our independent sampling approach allows us to scale balancing for each data point independently and reduces the global operation to counting the total matches for each entry. We demonstrate case studies in experiments on (1) scaling curation in a data pipeline and (2) online balancing in data loader.

# 4 EXPERIMENTS

Data Pools. We collect two pools of data:

Pool 1 contains 1.6 billion image-text pairs with a total of 5.6 billion counts of matches. This pool was used to estimate a target of 400M image-text pairs, collected from 15 snapshots of Common-Crawl (CC) from January 2021 to January 2023.

Pool 2 aims to scale curation in our data pipeline. We parsed all 90 CC snapshots from 2013 to April 2023, using our algorithm (see §A.2 for details on the curation pipeline) to curate from a pool of 10.7B matched image-text pairs that are originally from a large set of URL-text pairs, which have undergone de-duplication, English Language IDentification (LID) and sub-string matching. However, we only perform (expensive) image downloading, storing, and transferring for data points that are distribution-calibrated and selected by our algorithm.

<table><tr><td></td><td>Average</td><td>ImageNet</td><td>Food-101</td><td>CIFAR10</td><td>CIFAR100</td><td>CUB</td><td>SUN397</td><td>Cars</td><td>Aircraft</td><td>DTD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td><td>MNIST</td><td>FIBR-2013</td><td>STL-10</td><td>EuroSAT</td><td>RESISC45</td><td>GTSRB</td><td>KITTI</td><td>Country-211</td><td>PCAM</td><td>UCF01</td><td>Kinetic-700</td><td>CLEVR</td><td>HautolMemes</td><td>SST2</td></tr><tr><td colspan="28">ViT-B/32</td></tr><tr><td>CLIP, our eval.</td><td>56.6</td><td>63.4</td><td>83.7</td><td>89.8</td><td>65.1</td><td>53.7</td><td>62.0</td><td>59.7</td><td>19.6</td><td>44.0</td><td>87.2</td><td>87.4</td><td>66.9</td><td>48.2</td><td>46.6</td><td>97.1</td><td>44.9</td><td>61.0</td><td>32.6</td><td>28.7</td><td>17.2</td><td>62.5</td><td>63.9</td><td>48.0</td><td>23.6</td><td>56.4</td><td>58.6</td></tr><tr><td>OpenCLIP, our eval.</td><td>57.6</td><td>62.9</td><td>80.7</td><td>90.7</td><td>70.6</td><td>61.2</td><td>66.4</td><td>79.2</td><td>16.7</td><td>54.5</td><td>86.5</td><td>90.7</td><td>66.1</td><td>37.4</td><td>48.2</td><td>95.6</td><td>52.2</td><td>58.0</td><td>42.0</td><td>38.0</td><td>14.8</td><td>50.1</td><td>63.0</td><td>42.8</td><td>22.5</td><td>53.3</td><td>52.3</td></tr><tr><td>MetaCLIP</td><td>58.2</td><td>65.5</td><td>80.6</td><td>91.3</td><td>70.2</td><td>63.4</td><td>63.0</td><td>70.7</td><td>26.8</td><td>52.8</td><td>88.7</td><td>91.9</td><td>68.5</td><td>41.5</td><td>35.9</td><td>95.4</td><td>52.6</td><td>64.2</td><td>35.8</td><td>30.7</td><td>17.2</td><td>55.5</td><td>66.1</td><td>45.4</td><td>30.6</td><td>56.4</td><td>53.4</td></tr><tr><td colspan="28">ViT-B/16</td></tr><tr><td>CLIP, our eval.</td><td>59.6</td><td>68.3</td><td>88.8</td><td>90.8</td><td>68.2</td><td>55.6</td><td>64.0</td><td>64.6</td><td>24.0</td><td>45.1</td><td>88.9</td><td>89.1</td><td>69.4</td><td>51.8</td><td>53.0</td><td>98.2</td><td>54.8</td><td>65.5</td><td>43.3</td><td>21.7</td><td>22.8</td><td>56.3</td><td>68.5</td><td>52.3</td><td>25.5</td><td>58.7</td><td>60.5</td></tr><tr><td>OpenCLIP, our eval.</td><td>60.4</td><td>67.0</td><td>85.8</td><td>91.7</td><td>71.4</td><td>65.3</td><td>69.2</td><td>83.6</td><td>17.4</td><td>51.0</td><td>89.2</td><td>90.8</td><td>66.5</td><td>66.3</td><td>46.1</td><td>97.0</td><td>52.2</td><td>65.7</td><td>43.5</td><td>23.7</td><td>18.1</td><td>51.7</td><td>67.0</td><td>46.2</td><td>33.9</td><td>54.5</td><td>54.4</td></tr><tr><td>MetaCLIP</td><td>61.1</td><td>70.8</td><td>86.8</td><td>90.1</td><td>66.5</td><td>70.8</td><td>66.6</td><td>74.1</td><td>27.9</td><td>55.9</td><td>90.4</td><td>93.8</td><td>72.3</td><td>47.8</td><td>44.6</td><td>97.2</td><td>55.4</td><td>68.8</td><td>43.8</td><td>33.4</td><td>22.6</td><td>52.9</td><td>68.0</td><td>49.5</td><td>22.8</td><td>54.8</td><td>60.6</td></tr><tr><td colspan="28">ViT-L/14</td></tr><tr><td>CLIP, our eval.</td><td>65.7</td><td>75.5</td><td>93.0</td><td>95.6</td><td>78.3</td><td>63.3</td><td>66.8</td><td>77.8</td><td>31.3</td><td>55.3</td><td>93.6</td><td>93.3</td><td>79.3</td><td>76.4</td><td>56.9</td><td>99.4</td><td>61.9</td><td>70.9</td><td>50.6</td><td>19.2</td><td>31.9</td><td>50.1</td><td>75.7</td><td>60.2</td><td>22.3</td><td>59.7</td><td>68.9</td></tr><tr><td>OpenCLIP, our eval.</td><td>64.5</td><td>72.7</td><td>90.0</td><td>94.7</td><td>78.0</td><td>73.9</td><td>72.4</td><td>89.5</td><td>24.7</td><td>60.2</td><td>91.6</td><td>93.6</td><td>73.0</td><td>76.1</td><td>54.3</td><td>98.1</td><td>63.9</td><td>69.6</td><td>49.9</td><td>16.0</td><td>23.0</td><td>51.7</td><td>71.5</td><td>51.6</td><td>25.4</td><td>55.3</td><td>56.0</td></tr><tr><td>MetaCLIP</td><td>67.1</td><td>76.2</td><td>90.7</td><td>95.5</td><td>77.4</td><td>75.9</td><td>70.5</td><td>84.7</td><td>40.4</td><td>62.0</td><td>93.7</td><td>94.4</td><td>76.4</td><td>61.7</td><td>46.5</td><td>99.3</td><td>59.7</td><td>71.9</td><td>47.5</td><td>29.9</td><td>30.9</td><td>70.1</td><td>75.5</td><td>57.1</td><td>35.1</td><td>56.6</td><td>65.6</td></tr></table>

Table 4: MetaCLIP-400M vs. CLIP (WIT400M data) and OpenCLIP (LAION-400M data(Schuhmann et al., 2021)). We use 3 different model scales (ViT-B/32 and -B/16 and -L/14) and an identical training setup as CLIP. 

<table><tr><td></td><td>Average</td><td>ImageNet</td><td>Food-101</td><td>CIFAR10</td><td>CIFAR100</td><td>CLIB</td><td>SUN397</td><td>Cars</td><td>Aircraft</td><td>DTD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td><td>MNIST</td><td>FER-2013</td><td>STL-10</td><td>EuroSAT</td><td>RESISC45</td><td>GTSSRB</td><td>KITTI</td><td>Country211</td><td>PCAM</td><td>UCF101</td><td>Kinetic700</td><td>CLEVR</td><td>HatefulMenses</td><td>SST2</td></tr><tr><td colspan="2">ViT-B/32</td><td colspan="26"></td></tr><tr><td>MetaCLIP(400M)</td><td>58.2</td><td>65.5</td><td>80.6</td><td>91.3</td><td>70.2</td><td>63.4</td><td>63.0</td><td>70.7</td><td>26.8</td><td>52.8</td><td>88.7</td><td>91.9</td><td>68.5</td><td>41.5</td><td>35.9</td><td>95.4</td><td>52.6</td><td>64.2</td><td>35.8</td><td>30.7</td><td>17.2</td><td>55.5</td><td>66.1</td><td>45.4</td><td>30.6</td><td>56.4</td><td>53.4</td></tr><tr><td>MetaCLIP(1B)</td><td>60.3</td><td>67.3</td><td>81.9</td><td>95.2</td><td>76.7</td><td>71.4</td><td>65.9</td><td>73.0</td><td>31.4</td><td>58.9</td><td>89.5</td><td>92.5</td><td>72.6</td><td>35.4</td><td>45.8</td><td>96.3</td><td>50.4</td><td>64.6</td><td>40.7</td><td>32.0</td><td>17.0</td><td>64.2</td><td>70.3</td><td>47.8</td><td>14.6</td><td>54.9</td><td>56.8</td></tr><tr><td>MetaCLIP(2.5B)</td><td>59.8</td><td>67.6</td><td>82.6</td><td>95.2</td><td>77.7</td><td>67.8</td><td>66.8</td><td>77.2</td><td>26.9</td><td>58.9</td><td>90.9</td><td>92.5</td><td>69.7</td><td>42.7</td><td>48.3</td><td>96.3</td><td>49.9</td><td>66.5</td><td>39.2</td><td>29.3</td><td>17.7</td><td>50.0</td><td>68.0</td><td>47.6</td><td>19.4</td><td>53.5</td><td>53.1</td></tr><tr><td colspan="2">ViT-B/16</td><td colspan="26"></td></tr><tr><td>MetaCLIP(400M)</td><td>61.1</td><td>70.8</td><td>86.8</td><td>90.1</td><td>66.5</td><td>70.8</td><td>66.6</td><td>74.1</td><td>27.9</td><td>55.9</td><td>90.4</td><td>93.8</td><td>72.3</td><td>47.8</td><td>44.6</td><td>97.2</td><td>55.4</td><td>68.8</td><td>43.8</td><td>33.4</td><td>22.6</td><td>52.9</td><td>68.0</td><td>49.5</td><td>22.8</td><td>54.8</td><td>60.6</td></tr><tr><td>MetaCLIP(1B)</td><td>63.2</td><td>72.4</td><td>88.1</td><td>94.8</td><td>78.2</td><td>77.5</td><td>66.4</td><td>79.3</td><td>38.0</td><td>57.7</td><td>92.3</td><td>93.6</td><td>75.1</td><td>36.4</td><td>47.8</td><td>98.0</td><td>50.5</td><td>70.1</td><td>49.5</td><td>36.6</td><td>21.6</td><td>53.7</td><td>74.1</td><td>52.7</td><td>21.6</td><td>56.8</td><td>61.6</td></tr><tr><td>MetaCLIP(2.5B)</td><td>63.5</td><td>72.1</td><td>88.3</td><td>95.7</td><td>79.0</td><td>71.4</td><td>68.5</td><td>82.9</td><td>30.3</td><td>62.1</td><td>91.7</td><td>93.3</td><td>73.9</td><td>66.1</td><td>47.0</td><td>98.4</td><td>51.1</td><td>71.1</td><td>46.6</td><td>16.6</td><td>22.7</td><td>50.5</td><td>73.0</td><td>52.5</td><td>30.8</td><td>57.4</td><td>59.0</td></tr><tr><td colspan="2">ViT-L/14</td><td colspan="26"></td></tr><tr><td>MetaCLIP(400M)</td><td>67.1</td><td>76.2</td><td>90.7</td><td>95.5</td><td>77.4</td><td>75.9</td><td>70.5</td><td>84.7</td><td>40.4</td><td>62.0</td><td>93.7</td><td>94.4</td><td>76.4</td><td>61.7</td><td>46.5</td><td>99.3</td><td>59.7</td><td>71.9</td><td>47.5</td><td>29.9</td><td>30.9</td><td>70.1</td><td>75.5</td><td>57.1</td><td>35.1</td><td>56.6</td><td>65.6</td></tr><tr><td>MetaCLIP(1B)</td><td>70.2</td><td>79.0</td><td>92.9</td><td>96.8</td><td>84.9</td><td>83.1</td><td>72.8</td><td>86.5</td><td>48.9</td><td>65.9</td><td>95.3</td><td>94.8</td><td>84.7</td><td>53.8</td><td>54.1</td><td>99.3</td><td>70.0</td><td>73.8</td><td>58.7</td><td>36.3</td><td>32.2</td><td>70.4</td><td>81.4</td><td>61.6</td><td>21.1</td><td>61.2</td><td>66.1</td></tr><tr><td>MetaCLIP(2.5B)</td><td>69.8</td><td>79.2</td><td>93.4</td><td>97.6</td><td>84.2</td><td>80.1</td><td>73.8</td><td>88.7</td><td>44.6</td><td>68.1</td><td>94.7</td><td>95.4</td><td>81.8</td><td>64.4</td><td>55.1</td><td>99.3</td><td>59.2</td><td>74.6</td><td>56.3</td><td>29.7</td><td>34.0</td><td>67.3</td><td>81.6</td><td>62.0</td><td>25.9</td><td>58.0</td><td>66.7</td></tr><tr><td colspan="2">ViT-H/14</td><td colspan="26"></td></tr><tr><td>MetaCLIP(2.5B)</td><td>72.4</td><td>80.5</td><td>94.2</td><td>98.0</td><td>86.4</td><td>83.4</td><td>74.1</td><td>90.0</td><td>50.2</td><td>72.4</td><td>95.4</td><td>95.6</td><td>85.1</td><td>72.7</td><td>55.2</td><td>99.4</td><td>66.3</td><td>74.6</td><td>62.5</td><td>38.2</td><td>37.2</td><td>65.8</td><td>82.2</td><td>64.1</td><td>30.1</td><td>59.3</td><td>69.2</td></tr><tr><td colspan="2">ViT-bigG/14</td><td colspan="26"></td></tr><tr><td>MetaCLIP(2.5B)</td><td>73.2</td><td>82.1</td><td>94.9</td><td>98.5</td><td>88.6</td><td>84.0</td><td>74.7</td><td>90.9</td><td>52.7</td><td>72.6</td><td>96.1</td><td>95.7</td><td>89.5</td><td>78.1</td><td>56.7</td><td>99.5</td><td>73.7</td><td>75.5</td><td>61.7</td><td>31.0</td><td>41.5</td><td>65.6</td><td>85.6</td><td>65.8</td><td>24.3</td><td>58.8</td><td>65.3</td></tr></table>

Table 5: Scaling MetaCLIP from 400M (t=20k) to 1B (t=20k) and 2.5B (t=170k) training data.

For balancing we consider 2 scenarios on this data: (i) t = 170k, which is resulting in 2.5B image-text pairs. This t = 170k configuration has tail counts amounting to 6% of the total counts, the same tail/head ratio that the 400M Pool 1 data has, produced by applying t = 20k on the 1.6B Pool 1 data. (ii) The t = 20k threshold applied to Pool 2 which results in 1B image-text pairs and compared to the 400M set from Pool 1 only increases tail metadata matches (head counts are capped at 20k).

Training Setup We strictly follow the CLIP training setup, using V100 32GB GPUs and an equivalent global batch size of 32,768. For ViT-B/32 and ViT-B/16, we use 64 GPUs with a per GPU batch size of 512 and for ViT-L/14 we use 128 GPUs with a 256 per GPU batch size. It takes 4 days to train ViT-B/32 and a month to train ViT-L/14. We use 256 A100 80GB GPUs to train ViT-H/14 and ViT-bigG/14 model for 1 week and 2 months, respectively. We train in all experiments for the same number of iterations that correspond to 12.8B seen image-text pairs during training (32 epochs for 400M). We pre-process with face-blurring.

# 4.1 RESULTS

Zero-shot Image Classification. We follow the standard evaluation benchmark and made sure all prompts and class names were the same as those used by CLIP Radford et al. (2021). We also re-evaluated OpenAI/OpenCLIP's checkpoints to avoid differences caused by benchmark data copies. The results are shown in Tab 4. The standard deviation of training multiple seeds is relatively small with $\pm 0.1\%$ for ImageNet on ViT-B/32.

In Table 4, we observe that MetaCLIP outperforms OpenAI CLIP on ImageNet and average accuracy across 26 tasks, for 3 model scales. With 400 million training data points on ViT-B/32, MetaCLIP outperforms CLIP by +2.1% on ImageNet and by +1.6% on average. On ViT-B/16, MetaCLIP outperforms CLIP by +2.5% on ImageNet and by +1.5% on average. On ViT-L/14, MetaCLIP outperforms CLIP by +0.7% on ImageNet and by +1.4% on average across the 26 tasks.

We next turn to Pool 2 which is a larger set of image-text pairs and study the effect of scaling data. In Table 5, we scale data to 1B and 2.5B and observe a large gain over 400M, with similar performance for both 1B and 2.5B scales. Note that the number of training iterations (and therefore compute) is the same for all rows. The main difference between 1B and 2.5B is the threshold t, where 1B is

![](images/22e2f3549a1d2cbb9903ee0f8da7fb1b0e14754c231bf493c7c2a5fd6bef36ee.jpg)

<details>
<summary>line</summary>

| Metadata Entries Sorted by Counts | Cumulative Entry Counts |
| ---------------------------------- | ------------------------ |
| Pool 1, t=20k (400M)               | ~0.0                     |
| Pool 2, t=20k (1B)                | ~0.0                     |
| Pool 2, t=170k (2.5B)              | ~0.0                     |
| Pool 2 (10.7B)                      | ~2.5e10                  |
</details>

Figure 3: Cumulative sum of counts on entries from tail to head on a data Pool 2. We again show (1) raw/unbalanced cumulative counts), $t = \infty$ ; (2) balanced cumulative counts after applying t = 20k and t = 170k. t defines maximum number of counts per entry and the transition of tail/head entries. We show the Pool 1 configuration from Fig. 2 as dashed lines for reference.

a more balanced set by adding more data points (compared to the 400M set) to tail entries (up to $t = 20k$ ), instead the 2.5B set adds (up to $t = 170k$ ) data points to all, head and tail, entries. The extra data in the tail entries (1B set), seems to benefit downstream accuracy for tasks on specific data such as CUB fine-grained bird classification, Flowers, KITTI, PCAM, while the larger 2.5B data that has more head entries increases broadly over more datasets, but each at a smaller amount. The overall average accuracies are similar for 1B and 2.5B (e.g., 70.2% vs. 69.8% for ViT-L model size). On ImageNet, the 2.5B training data achieves 67.6% on ViT-B/32 that breaks the previous believed saturated B/32 models (Cherti et al., 2022), 79.2% on ViT-L/14, 80.5% on ViT-H/14 and 82.1% on ViT-bigG/14.

We plot the cumulative sum of counts for entries sorted by counts from tail to head in Fig. 3 for all these cases, similar to Fig. 2 for Pool 1 (and the Pool 1 configuration as dashed lines). The plot shows that the 2.5B data is still relatively long-tail, while the 1B data is more balanced, explaining it's better performance on specific data such as bird and flower types observed above.

# 4.2 ABLATION STUDY

We show ablations for MetaCLIP for the 400M scale and ViT-B/32 in Table 6. We first ablate different balancing thresholds t. We observe that the choice of t = 20k by CLIP yields the best performance for ImageNet and averaged accuracy and t = 15k or t = 35k are slightly worse.

To understand the key effect of balancing, we use the whole matched pool (1.6B image-text pairs) to train CLIP. Surprisingly, training on $4 \times$ more data (on head entries) significantly hurts the accuracy on ImageNet (61.9 vs 65.5) and averaged accuracy across 26 tasks (56.6 vs 58.2).

Balancing can also be applied online in the data loader with head entries down-sampled leading to slightly better performance (58.5 vs 58.2); see appendix for details. This is useful if head data has already been collected and one wants to train on a different distribution. The better accuracy for online balancing is explained by the larger diversity in head data.

<table><tr><td></td><td>Average</td><td>ImageNet</td><td>Food-101</td><td>CIFAR10</td><td>CIFAR100</td><td>CUB</td><td>SUN397</td><td>Cars</td><td>Aircraft</td><td>DTD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td><td>MNIST</td><td>FER-2013</td><td>STL-10</td><td>EuroSAF</td><td>RESIGC45</td><td>GTSRB</td><td>KITTI</td><td>Country211</td><td>PCAM</td><td>UCFI&#x27;01</td><td>Kinetics700</td><td>CLEVR</td><td>Hateful Memes</td><td>SST2</td></tr><tr><td>MetaCLIP t=20k</td><td>58.2</td><td>65.5</td><td>80.6</td><td>91.3</td><td>70.2</td><td>63.4</td><td>63.0</td><td>70.7</td><td>26.8</td><td>52.8</td><td>88.7</td><td>91.9</td><td>68.5</td><td>41.5</td><td>35.9</td><td>95.4</td><td>52.6</td><td>64.2</td><td>35.8</td><td>30.7</td><td>17.2</td><td>55.5</td><td>66.1</td><td>45.4</td><td>30.6</td><td>56.4</td><td>53.4</td></tr><tr><td>-t=15k</td><td>57.5</td><td>65.5</td><td>79.9</td><td>90.4</td><td>68.8</td><td>65.7</td><td>64.6</td><td>69.4</td><td>25.6</td><td>52.1</td><td>88.8</td><td>91.9</td><td>69.5</td><td>35.8</td><td>39.7</td><td>96.5</td><td>54.0</td><td>64.1</td><td>34.8</td><td>30.6</td><td>16.1</td><td>52.3</td><td>67.1</td><td>45.4</td><td>22.3</td><td>51.2</td><td>53.8</td></tr><tr><td>-t=35k</td><td>57.8</td><td>65.4</td><td>79.3</td><td>91.2</td><td>69.0</td><td>63.0</td><td>65.0</td><td>72.0</td><td>28.5</td><td>52.7</td><td>88.5</td><td>91.8</td><td>68.0</td><td>42.0</td><td>23.0</td><td>96.2</td><td>50.0</td><td>63.8</td><td>40.2</td><td>32.4</td><td>17.7</td><td>56.1</td><td>64.2</td><td>44.8</td><td>28.0</td><td>55.4</td><td>54.2</td></tr><tr><td>-unbalanced (1.6B)</td><td>56.6</td><td>61.9</td><td>76.9</td><td>90.0</td><td>67.6</td><td>50.8</td><td>65.8</td><td>77.0</td><td>19.9</td><td>51.0</td><td>83.1</td><td>91.5</td><td>64.5</td><td>58.2</td><td>37.0</td><td>95.1</td><td>55.2</td><td>58.2</td><td>41.4</td><td>32.2</td><td>15.1</td><td>51.0</td><td>59.2</td><td>42.6</td><td>17.2</td><td>55.6</td><td>52.6</td></tr><tr><td>-online balancing</td><td>58.5</td><td>66.1</td><td>80.8</td><td>89.9</td><td>68.8</td><td>65.7</td><td>65.4</td><td>71.6</td><td>27.9</td><td>55.1</td><td>88.2</td><td>92.7</td><td>68.8</td><td>38.3</td><td>42.1</td><td>96.5</td><td>54.5</td><td>64.8</td><td>36.2</td><td>29.1</td><td>17.6</td><td>58.8</td><td>66.0</td><td>45.8</td><td>22.0</td><td>56.0</td><td>52.4</td></tr></table>

Table 6: Ablation studies on balancing in MetaCLIP. Default: t=20k, 400M. Model: ViT-B/32.

# 5 CONCLUSION

In this paper, we attempt to reveal CLIP's data curation. Our MetaCLIP builds upon metadata for curation and balancing of raw data sourced from the web. Curating with metadata and balancing are essential for good data quality, significantly outperforming the use of raw data. Our experiments show that MetaCLIP performs well for different scales sourced from CommonCrawl data and outperforms CLIP's proprietary data source, without reliance on any external model. We make our pipeline for generating the data publicly available.

# ACKNOWLEDGMENTS

We thank Zeyuan Allen-Zhu, and Chunting Zhou for the insightful discussion and Brighid Meredith for suggestions on scaling the pipeline.

# A APPENDIX

# A.1 ADDITIONAL RESULTS

# Curation from DataComp-12.8B.

The concurrent work Gadre et al. (2023) released a collection of 12.8B image-text pairs from CommonCrawl from 2014-2022. We further investigate whether we can apply the algorithm on its 12.8 unfiltered pool. Although the unfiltered pool seemingly offers an opportunity to apply our algorithm on a publicly available source, our initial studies show that, implicit biases may still be present in this pool. For example, we notice that all image URLs are collected as a string starting with http. This excludes relative URLs that could be frequently used by quality websites (with potentially good image-text pairs). We curate from DataComp's 12.8B unfiltered pool with t=60k, which has 6% of tail counts that is the same as t=20k for 400M from our 1.6B pool.

When using 1B image-text pairs curated from DataComp's pool we notice a quality drop during training, compared to data curated from our pools, see Fig. 4. Our smaller 400M set is slightly better than using DataComp-1B and our larger sets (1B, 2.5B) are significantly better. In Table 7, we show our 400M data vs our curated DataComp-1B data at various model scales, where the same observation holds, suggesting our raw data pool is more effective.

![](images/ec71102618461a18a289328dbb098892ecb8d5b6bca8f570b5f79d83762f0f7d.jpg)

<details>
<summary>line</summary>

| Training Steps | CLIP(400M) | LAION(407M) | MetaCLIP(1B,DataComp) | MetaCLIP(400M) | MetaCLIP(1B) | MetaCLIP(2.5B) |
| -------------- | ---------- | ----------- | --------------------- | -------------- | ------------ | -------------- |
| 0              | 0.45       | 0.45        | 0.45                  | 0.45           | 0.45         | 0.45           |
| 50000          | 0.50       | 0.48        | 0.50                  | 0.51           | 0.52         | 0.53           |
| 100000         | 0.53       | 0.51        | 0.53                  | 0.54           | 0.55         | 0.56           |
| 150000         | 0.55       | 0.53        | 0.55                  | 0.56           | 0.57         | 0.58           |
| 200000         | 0.57       | 0.55        | 0.57                  | 0.58           | 0.59         | 0.60           |
| 250000         | 0.59       | 0.57        | 0.59                  | 0.60           | 0.61         | 0.62           |
| 300000         | 0.61       | 0.59        | 0.61                  | 0.62           | 0.63         | 0.64           |
| 350000         | 0.63       | 0.61        | 0.63                  | 0.64           | 0.65         | 0.66           |
| 400000         | 0.64       | 0.63        | 0.64                  | 0.65           | 0.66         | 0.67           |
</details>

Figure 4: ViT-B/32 on our Pool 1, Pool 2 and DataComp's unfiltered 12.8B pool. We show ImageNet zero-shot classification with a fixed 12.8B seen pair training budget. MetaCLIP's curation is effective for all pools. However, with the same curation method, the unfiltered DataComp-12.8B pool lacks quality (we suspect it is caused by implicit filters in the parser of DataComp).

DataComp Benchmark We also evaluate MetaCLIP on the benchmark used by (Gadre et al., 2023) that contains 38 tasks including variants of ImageNet, retrieval, VTAB, etc. For simplicity, we average the scores over each category.

Note that prompts and class names used by (Gadre et al., 2023) could be different from prompts and classnames used by OpenAI CLIP.

<table><tr><td></td><td>Average</td><td>ImageNet</td><td>Food-101</td><td>CIFAR10</td><td>CIFAR100</td><td>CUB</td><td>SUN397</td><td>Cars</td><td>Aircraft</td><td>DVD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td><td>MNIST</td><td>FER-2013</td><td>STL-10</td><td>EuroSAT</td><td>RESISC45</td><td>GTSRB</td><td>KITTI</td><td>Country-211</td><td>PCAM</td><td>UCF101</td><td>Kaietic-700</td><td>CLEVR</td><td>HatchoMemes</td><td>SST2</td></tr><tr><td colspan="28">ViT-B/32</td></tr><tr><td>MetaCLIP (400M)</td><td>58.2</td><td>65.5</td><td>80.6</td><td>91.3</td><td>70.2</td><td>63.4</td><td>63.0</td><td>70.7</td><td>26.8</td><td>52.8</td><td>88.7</td><td>91.9</td><td>68.5</td><td>41.5</td><td>35.9</td><td>95.4</td><td>52.6</td><td>64.2</td><td>35.8</td><td>30.7</td><td>17.2</td><td>55.5</td><td>66.1</td><td>45.4</td><td>30.6</td><td>56.4</td><td>53.4</td></tr><tr><td>MetaCLIP (1B,DataComp)</td><td>57.5</td><td>65.4</td><td>81.5</td><td>88.7</td><td>68.5</td><td>59.4</td><td>64.7</td><td>76.6</td><td>18.1</td><td>57.5</td><td>89.9</td><td>92.4</td><td>67.3</td><td>40.7</td><td>38.2</td><td>96.5</td><td>41.2</td><td>62.3</td><td>43.4</td><td>35.1</td><td>17.6</td><td>53.5</td><td>61.5</td><td>44.8</td><td>19.2</td><td>57.0</td><td>54.3</td></tr><tr><td colspan="28">ViT-B/16</td></tr><tr><td>MetaCLIP (400M)</td><td>61.1</td><td>70.8</td><td>86.8</td><td>90.1</td><td>66.5</td><td>70.8</td><td>66.6</td><td>74.1</td><td>27.9</td><td>55.9</td><td>90.4</td><td>93.8</td><td>72.3</td><td>47.8</td><td>44.6</td><td>97.2</td><td>55.4</td><td>68.8</td><td>43.8</td><td>33.4</td><td>22.6</td><td>52.9</td><td>68.0</td><td>49.5</td><td>22.8</td><td>54.8</td><td>60.6</td></tr><tr><td>MetaCLIP (1B,DataComp)</td><td>61.2</td><td>70.7</td><td>88.1</td><td>91.3</td><td>71.6</td><td>64.5</td><td>67.6</td><td>81.2</td><td>21.6</td><td>62.3</td><td>92.7</td><td>93.2</td><td>70.7</td><td>55.6</td><td>39.6</td><td>97.7</td><td>52.9</td><td>66.3</td><td>45.7</td><td>36.0</td><td>22.3</td><td>50.1</td><td>68.1</td><td>49.2</td><td>17.1</td><td>56.4</td><td>59.9</td></tr><tr><td colspan="28">ViT-L/14</td></tr><tr><td>MetaCLIP (400M)</td><td>67.1</td><td>76.2</td><td>90.7</td><td>95.5</td><td>77.4</td><td>75.9</td><td>70.5</td><td>84.7</td><td>40.4</td><td>62.0</td><td>93.7</td><td>94.4</td><td>76.4</td><td>61.7</td><td>46.5</td><td>99.3</td><td>59.7</td><td>71.9</td><td>47.5</td><td>29.9</td><td>30.9</td><td>70.1</td><td>75.5</td><td>57.1</td><td>35.1</td><td>56.6</td><td>65.6</td></tr><tr><td>MetaCLIP (1B,DataComp)</td><td>66.3</td><td>76.7</td><td>92.6</td><td>95.6</td><td>78.9</td><td>72.1</td><td>71.6</td><td>87.1</td><td>31.6</td><td>67.5</td><td>93.4</td><td>95.3</td><td>76.0</td><td>65.1</td><td>42.4</td><td>99.1</td><td>61.7</td><td>69.8</td><td>45.9</td><td>36.8</td><td>31.8</td><td>51.0</td><td>76.6</td><td>57.5</td><td>29.3</td><td>57.1</td><td>60.8</td></tr></table>

Table 7: MetaCLIP curating our 400M data vs curating 1B data from DataComp-12.8B: The pool of DataComp leads to a quality drop with performance closer to our 400M set.

<table><tr><td></td><td>Avg.</td><td>IN</td><td>IN Dist. Shift</td><td>VTAB</td><td>Avg. Retrieval</td></tr><tr><td colspan="6">ViT-B/32</td></tr><tr><td>CLIP (400M)</td><td>51.5</td><td>63.4</td><td>48.2</td><td>50.5</td><td>48.0</td></tr><tr><td>OpenCLIP (407M)</td><td>52.7</td><td>62.9</td><td>48.5</td><td>53.0</td><td>50.7</td></tr><tr><td>MetaCLIP (400M)</td><td>53.5</td><td>65.5</td><td>50.4</td><td>54.1</td><td>50.6</td></tr><tr><td>MetaCLIP (1B)</td><td>54.2</td><td>67.3</td><td>51.9</td><td>53.6</td><td>51.1</td></tr><tr><td>MetaCLIP (2.5B)</td><td>55.4</td><td>67.6</td><td>52.3</td><td>55.3</td><td>52.6</td></tr><tr><td colspan="6">ViT-B/16</td></tr><tr><td>CLIP (400M)</td><td>55.5</td><td>68.3</td><td>54.1</td><td>54.4</td><td>50.2</td></tr><tr><td>OpenCLIP (407M)</td><td>56.1</td><td>67.0</td><td>52.6</td><td>54.9</td><td>53.9</td></tr><tr><td>MetaCLIP (400M)</td><td>57.5</td><td>70.8</td><td>55.5</td><td>56.7</td><td>53.9</td></tr><tr><td>MetaCLIP (1B)</td><td>58.4</td><td>72.4</td><td>57.8</td><td>56.3</td><td>54.3</td></tr><tr><td>MetaCLIP (2.5B)</td><td>60.0</td><td>72.1</td><td>57.7</td><td>59.0</td><td>54.0</td></tr><tr><td colspan="6">ViT-L/14</td></tr><tr><td>CLIP (400M)</td><td>61.4</td><td>75.5</td><td>61.6</td><td>59.5</td><td>53.6</td></tr><tr><td>OpenCLIP (407M)</td><td>59.7</td><td>72.7</td><td>57.3</td><td>58.6</td><td>55.9</td></tr><tr><td>MetaCLIP (400M)</td><td>62.2</td><td>76.2</td><td>61.3</td><td>59.8</td><td>57.3</td></tr><tr><td>MetaCLIP (1B)</td><td>65.0</td><td>79.0</td><td>64.5</td><td>62.5</td><td>58.3</td></tr><tr><td>MetaCLIP (2.5B)</td><td>65.6</td><td>79.2</td><td>64.6</td><td>64.1</td><td>60.1</td></tr><tr><td colspan="6">ViT-H/14</td></tr><tr><td>MetaCLIP (2.5B)</td><td>66.5</td><td>80.5</td><td>66.1</td><td>64.6</td><td>60.4</td></tr><tr><td colspan="6">ViT-bigG/14</td></tr><tr><td>MetaCLIP (2.5B)</td><td>68.3</td><td>82.1</td><td>67.6</td><td>66.5</td><td>62.6</td></tr></table>

Table 8: Zero-shot classification and retrieval on tasks from (Gadre et al., 2023).

From Table 8, we can see that MetaCLIP outperforms CLIP and OpenCLIP across various model sizes. First, for the same data scale (400M), MetaCLIP outperforms OpenCLIP, which is better than CLIP on this benchmark, by +1.4% for ViT-B/16 and +2.5% for ViT-L/14, when comparing average accuracy across the 38 tasks. Second, for increasing our MetaCLIP data size to 1B we see a significant gain, especially for the larger model, from 62.2% to 65.0% average accuracy. Using our larger dataset with 2.5B and more head entries leads to a further gain to 65.5%.

We further detail the breakdown of each dataset in Table 9 (please view in landscape orientation).

<table><tr><td rowspan="2"></td><td rowspan="2">ImageNet 1k</td><td rowspan="2">ImageNet Sketch</td><td rowspan="2">ImageNet v2</td><td rowspan="2">ImageNet-A</td><td rowspan="2">ImageNet-O</td><td rowspan="2">ImageNet-R</td><td rowspan="2">Caltech-101</td><td rowspan="2">CIFAR-100</td><td rowspan="2">CLEVR Counts</td><td rowspan="2">CLEVR Distance</td><td rowspan="2">Describeable Textures</td><td rowspan="2">EuroSAT</td><td rowspan="2">KITTI Vehicle Distance</td><td rowspan="2">Oxford Flowers-102</td><td rowspan="2">Oxford-IIIT Pet</td><td rowspan="2">RESISC45</td><td rowspan="2">SUN397</td><td rowspan="2">SVHN</td><td rowspan="2">Camelyon17</td><td rowspan="3">Flickr</td><td rowspan="3">MSCOCO</td><td rowspan="3">WinoGAVIL</td><td colspan="16">Retrieval</td></tr><tr><td rowspan="2">CIFAR-10</td><td rowspan="2">Country211</td><td rowspan="2">FGVC Aircraft</td><td rowspan="2">Food-101</td><td rowspan="2">GTSRB</td><td rowspan="2">MNIST</td><td rowspan="2">ObjectNet</td><td rowspan="2">Pascal VOC 2007</td><td rowspan="2">PatchCamelyon</td><td rowspan="2">Rendered SST2</td><td rowspan="2">Stanford Cars</td><td rowspan="2">STL-10</td><td rowspan="2">iWildCam</td><td rowspan="2">FMoW</td><td rowspan="2">Dollar Street</td><td>GeoDE</td></tr><tr><td></td><td colspan="6">ImageNet Distribution Shift</td><td colspan="13">VTAB</td><td></td></tr><tr><td>ViT-B/32</td><td colspan="6"></td><td colspan="13"></td><td colspan="18"></td><td></td></tr><tr><td>CLIP (400M)</td><td>63.3</td><td>42.3</td><td>56.0</td><td>31.5</td><td>47.8</td><td>69.4</td><td>87.6</td><td>64.2</td><td>23.3</td><td>23.4</td><td>44.3</td><td>50.5</td><td>27.4</td><td>66.6</td><td>87.0</td><td>53.6</td><td>62.5</td><td>14.9</td><td>51.6</td><td>68.8</td><td>40.3</td><td>34.9</td><td>89.8</td><td>17.2</td><td>19.5</td><td>84.0</td><td>32.5</td><td>48.3</td><td>44.3</td><td>76.4</td><td>62.2</td><td>58.6</td><td>59.6</td><td>97.1</td><td>6.7</td><td>13.3</td><td>53.9</td><td>82.2</td></tr><tr><td>OpenCLIP (407M)</td><td>62.9</td><td>49.4</td><td>55.1</td><td>21.7</td><td>53.5</td><td>73.4</td><td>91.2</td><td>70.3</td><td>16.2</td><td>23.9</td><td>54.6</td><td>51.5</td><td>28.8</td><td>66.2</td><td>86.7</td><td>54.5</td><td>67.0</td><td>30.4</td><td>47.1</td><td>70.2</td><td>43.9</td><td>38.0</td><td>90.7</td><td>14.7</td><td>16.6</td><td>80.9</td><td>42.0</td><td>37.5</td><td>43.9</td><td>75.8</td><td>55.9</td><td>52.3</td><td>79.3</td><td>95.6</td><td>7.4</td><td>13.0</td><td>54.9</td><td>83.8</td></tr><tr><td>MetaCLIP (400M)</td><td>65.6</td><td>53.3</td><td>57.6</td><td>28.6</td><td>46.8</td><td>74.8</td><td>91.7</td><td>70.0</td><td>21.5</td><td>24.5</td><td>52.5</td><td>52.4</td><td>25.9</td><td>68.1</td><td>88.8</td><td>59.6</td><td>63.4</td><td>20.5</td><td>64.5</td><td>70.1</td><td>43.9</td><td>37.8</td><td>91.3</td><td>17.1</td><td>26.9</td><td>81.0</td><td>35.8</td><td>41.4</td><td>50.5</td><td>70.8</td><td>64.2</td><td>53.5</td><td>70.6</td><td>95.4</td><td>8.2</td><td>0.0</td><td>59.7</td><td>85.4</td></tr><tr><td>MetaCLIP (1B)</td><td>67.3</td><td>55.3</td><td>59.2</td><td>29.7</td><td>48.2</td><td>76.1</td><td>93.7</td><td>76.5</td><td>15.7</td><td>22.5</td><td>59.1</td><td>48.0</td><td>20.5</td><td>72.7</td><td>89.5</td><td>61.6</td><td>66.3</td><td>21.7</td><td>49.4</td><td>72.6</td><td>45.1</td><td>35.6</td><td>95.2</td><td>16.9</td><td>31.0</td><td>82.6</td><td>40.7</td><td>35.6</td><td>49.4</td><td>79.0</td><td>50.0</td><td>57.1</td><td>73.0</td><td>96.3</td><td>8.4</td><td>14.8</td><td>58.9</td><td>86.0</td></tr><tr><td>MetaCLIP (2.5B)</td><td>67.7</td><td>56.0</td><td>59.6</td><td>29.9</td><td>48.3</td><td>78.1</td><td>92.9</td><td>77.7</td><td>18.7</td><td>23.1</td><td>58.8</td><td>49.9</td><td>18.7</td><td>69.4</td><td>90.9</td><td>61.2</td><td>66.9</td><td>34.3</td><td>56.6</td><td>72.9</td><td>46.6</td><td>38.2</td><td>95.2</td><td>17.7</td><td>27.1</td><td>83.1</td><td>39.3</td><td>42.6</td><td>52.8</td><td>76.5</td><td>56.0</td><td>53.2</td><td>77.3</td><td>96.3</td><td>9.2</td><td>15.9</td><td>60.5</td><td>86.1</td></tr><tr><td>ViT-B/16</td><td colspan="6"></td><td colspan="13"></td><td colspan="18"></td><td></td></tr><tr><td>CLIP (400M)</td><td>68.3</td><td>48.3</td><td>61.9</td><td>49.9</td><td>42.3</td><td>77.7</td><td>89.0</td><td>66.9</td><td>21.2</td><td>22.3</td><td>44.9</td><td>56.0</td><td>26.3</td><td>69.1</td><td>88.9</td><td>58.2</td><td>64.4</td><td>40.2</td><td>60.2</td><td>72.2</td><td>42.8</td><td>35.7</td><td>90.8</td><td>22.8</td><td>24.3</td><td>88.7</td><td>43.3</td><td>51.3</td><td>55.3</td><td>78.3</td><td>50.7</td><td>60.5</td><td>64.7</td><td>98.3</td><td>11.1</td><td>16.7</td><td>58.8</td><td>86.1</td></tr><tr><td>OpenCLIP (407M)</td><td>67.0</td><td>52.4</td><td>59.7</td><td>33.2</td><td>50.6</td><td>77.9</td><td>91.3</td><td>71.2</td><td>28.7</td><td>24.5</td><td>51.3</td><td>50.2</td><td>18.1</td><td>66.9</td><td>89.2</td><td>58.5</td><td>69.6</td><td>34.1</td><td>59.9</td><td>74.5</td><td>46.9</td><td>40.2</td><td>91.7</td><td>18.1</td><td>17.7</td><td>86.1</td><td>43.4</td><td>66.2</td><td>51.5</td><td>76.8</td><td>59.6</td><td>54.4</td><td>83.7</td><td>97.0</td><td>10.3</td><td>15.5</td><td>59.3</td><td>85.3</td></tr><tr><td>MetaCLIP (400M)</td><td>70.8</td><td>57.9</td><td>62.6</td><td>47.0</td><td>39.2</td><td>81.8</td><td>93.4</td><td>66.5</td><td>30.1</td><td>22.4</td><td>55.7</td><td>55.7</td><td>24.2</td><td>72.3</td><td>90.4</td><td>66.1</td><td>66.8</td><td>25.2</td><td>67.7</td><td>76.7</td><td>48.2</td><td>36.9</td><td>90.1</td><td>22.6</td><td>28.4</td><td>87.2</td><td>43.7</td><td>47.8</td><td>59.1</td><td>72.2</td><td>61.9</td><td>60.5</td><td>74.2</td><td>97.2</td><td>11.2</td><td>19.9</td><td>60.6</td><td>88.9</td></tr><tr><td>MetaCLIP (1B)</td><td>72.4</td><td>60.5</td><td>65.1</td><td>49.2</td><td>41.9</td><td>82.7</td><td>93.5</td><td>77.9</td><td>19.3</td><td>24.4</td><td>58.2</td><td>47.6</td><td>24.6</td><td>74.8</td><td>92.1</td><td>65.1</td><td>66.8</td><td>27.8</td><td>60.1</td><td>77.1</td><td>48.9</td><td>36.8</td><td>94.8</td><td>21.6</td><td>38.1</td><td>88.3</td><td>49.5</td><td>36.3</td><td>60.0</td><td>79.2</td><td>65.7</td><td>61.3</td><td>79.0</td><td>98.0</td><td>13.2</td><td>15.8</td><td>63.3</td><td>87.9</td></tr><tr><td>MetaCLIP (2.5B)</td><td>72.1</td><td>60.2</td><td>65.0</td><td>49.5</td><td>41.5</td><td>84.2</td><td>93.3</td><td>78.9</td><td>29.0</td><td>22.6</td><td>62.2</td><td>52.7</td><td>18.4</td><td>73.5</td><td>91.7</td><td>67.4</td><td>68.8</td><td>39.1</td><td>69.8</td><td>78.1</td><td>50.4</td><td>33.5</td><td>95.7</td><td>22.7</td><td>30.5</td><td>88.8</td><td>46.5</td><td>66.1</td><td>61.4</td><td>78.2</td><td>59.2</td><td>59.1</td><td>83.0</td><td>98.4</td><td>12.3</td><td>19.4</td><td>64.0</td><td>88.7</td></tr><tr><td>ViT-L/14</td><td colspan="6"></td><td colspan="13"></td><td colspan="18"></td><td></td></tr><tr><td>CLIP (400M)</td><td>75.6</td><td>59.6</td><td>69.8</td><td>70.7</td><td>32.3</td><td>87.9</td><td>92.5</td><td>75.8</td><td>19.5</td><td>20.2</td><td>55.3</td><td>62.6</td><td>21.8</td><td>79.2</td><td>93.2</td><td>63.3</td><td>67.6</td><td>52.8</td><td>69.6</td><td>75.1</td><td>46.4</td><td>39.2</td><td>95.6</td><td>31.9</td><td>31.7</td><td>93.1</td><td>50.6</td><td>76.3</td><td>68.9</td><td>78.3</td><td>52.0</td><td>68.9</td><td>77.9</td><td>99.4</td><td>12.3</td><td>15.2</td><td>63.0</td><td>88.4</td></tr><tr><td>OpenCLIP (407M)</td><td>72.8</td><td>59.6</td><td>65.4</td><td>46.5</td><td>41.9</td><td>84.7</td><td>92.6</td><td>77.4</td><td>24.3</td><td>24.5</td><td>60.5</td><td>62.3</td><td>20.1</td><td>73.1</td><td>91.7</td><td>67.4</td><td>72.6</td><td>49.5</td><td>45.5</td><td>78.9</td><td>51.3</td><td>37.4</td><td>94.6</td><td>23.0</td><td>24.9</td><td>90.1</td><td>49.9</td><td>76.1</td><td>59.7</td><td>75.6</td><td>49.7</td><td>56.0</td><td>89.6</td><td>98.1</td><td>12.5</td><td>17.0</td><td>61.7</td><td>88.4</td></tr><tr><td>MetaCLIP (400M)</td><td>76.2</td><td>65.0</td><td>69.8</td><td>66.4</td><td>28.9</td><td>88.9</td><td>94.6</td><td>77.3</td><td>22.7</td><td>25.1</td><td>62.4</td><td>60.4</td><td>24.2</td><td>76.5</td><td>93.8</td><td>68.5</td><td>70.8</td><td>32.4</td><td>69.2</td><td>79.8</td><td>51.9</td><td>40.2</td><td>95.5</td><td>30.9</td><td>39.8</td><td>90.7</td><td>47.5</td><td>61.8</td><td>69.2</td><td>74.4</td><td>70.3</td><td>65.6</td><td>84.8</td><td>99.3</td><td>14.1</td><td>18.7</td><td>67.3</td><td>89.3</td></tr><tr><td>MetaCLIP (1B)</td><td>79.0</td><td>68.9</td><td>72.5</td><td>70.4</td><td>31.6</td><td>91.0</td><td>58.4</td><td>84.6</td><td>15.8</td><td>22.5</td><td>65.8</td><td>68.3</td><td>23.2</td><td>83.8</td><td>95.2</td><td>68.0</td><td>72.9</td><td>38.0</td><td>79.1</td><td>82.2</td><td>54.2</td><td>38.6</td><td>96.8</td><td>32.2</td><td>49.6</td><td>93.1</td><td>58.6</td><td>53.9</td><td>73.9</td><td>80.9</td><td>73.1</td><td>66.0</td><td>86.6</td><td>99.3</td><td>16.2</td><td>28.1</td><td>69.4</td><td>91.1</td></tr><tr><td>MetaCLIP (2.5B)</td><td>79.2</td><td>68.9</td><td>72.6</td><td>72.3</td><td>30.1</td><td>92.0</td><td>95.3</td><td>84.1</td><td>31.0</td><td>22.6</td><td>68.6</td><td>59.0</td><td>27.9</td><td>81.4</td><td>94.6</td><td>73.6</td><td>73.6</td><td>46.8</td><td>75.5</td><td>83.3</td><td>55.7</td><td>41.4</td><td>97.6</td><td>33.9</td><td>45.3</td><td>93.5</td><td>56.3</td><td>64.4</td><td>74.6</td><td>80.3</td><td>62.0</td><td>66.8</td><td>88.8</td><td>99.3</td><td>15.8</td><td>25.9</td><td>67.4</td><td>91.4</td></tr><tr><td>ViT-H/14</td><td colspan="6"></td><td colspan="13"></td><td colspan="18"></td><td></td></tr><tr><td>MetaCLIP (2.5B)</td><td>80.5</td><td>70.5</td><td>74.1</td><td>75.4</td><td>30.2</td><td>93.4</td><td>95.3</td><td>86.4</td><td>21.1</td><td>18.8</td><td>72.7</td><td>64.5</td><td>27.7</td><td>84.5</td><td>95.6</td><td>70.3</td><td>74.4</td><td>62.8</td><td>66.2</td><td>85.0</td><td>57.5</td><td>38.8</td><td>98.1</td><td>37.2</td><td>51.0</td><td>94.2</td><td>62.6</td><td>72.7</td><td>76.5</td><td>75.0</td><td>62.3</td><td>69.1</td><td>89.9</td><td>99.4</td><td>17.3</td><td>15.6</td><td>68.1</td><td>90.7</td></tr><tr><td>ViT-bigG/14</td><td colspan="6"></td><td colspan="13"></td><td colspan="18"></td><td></td></tr><tr><td>MetaCLIP (2.5B)</td><td>82.1</td><td>72.8</td><td>76.0</td><td>78.4</td><td>28.9</td><td>94.1</td><td>95.7</td><td>88.3</td><td>17.0</td><td>20.3</td><td>72.1</td><td>72.8</td><td>26.7</td><td>89.1</td><td>96.2</td><td>74.0</td><td>75.1</td><td>60.4</td><td>76.1</td><td>85.4</td><td>58.1</td><td>44.4</td><td>98.5</td><td>41.5</td><td>52.7</td><td>94.8</td><td>61.7</td><td>78.1</td><td>77.2</td><td>74.7</td><td>75.8</td><td>65.3</td><td>90.8</td><td>99.5</td><td>18.3</td><td>18.5</td><td>70.2</td><td>92.5</td></tr></table>

Table 9: Zero-shot evaluation of Table 8 broken down by datasets.

# A.2 DETAILS ON EFFICIENT CURATION

Curation in Data Pipeline. Our data collection pipeline contains 5 major components: HTML parsing/Language-Identification, URLs & Text deduplication, Image Download, NSFW image filter & deduplication and packaging, which are described next.

Our HTML Parser is applied to all WAT files of CommonCrawl by keeping all <img> tags with both relative and absolute URLs and one or multiple text keys.

Language-Identification(LID): we use an internal language identification system that can detect 191 languages with different dialects and keep all texts that are tagged as English (or its dialects).

URLs / Text deduplication: We use 24bit sha224 hashing to encode an URL and into tables to deduplicate columns with the same hash, avoiding downloading the same image URL multiple times. URLs with illegal domains are removed; if there are multiple texts associated with the same image, these are further deduplicated; texts with NSFW keywords are removed;

NSFW filter: We use an internal system that can classify inappropriate content in images into 96 types of dangerous content and discard such image/text pairs. Images are further deduplicated by 64-bit PCA hash, derived from a similarity search model's feature embeddings with PCA reduction to 64 dimensions and sign quantization.

Our curation algorithm does not require access to images, making it suitable for integration into a pipeline to reduce the scale of data points after parsing and before image downloading. We designed the algorithm to be modular, allowing different parts to be placed at different stages in the pipeline, as shown in Figure 5.

Specifically, sub-string matching can be placed immediately after HTML parsing to reduce data points for English-only pairs (e.g. by $\sim50\%$ ). Balancing can be applied earlier, before image downloading, to further reduce data points by $\sim77\%$ . This approach led to a total reduction of $\sim90\%$ , allowing for curation of the entire CommonCrawl data without the need to store and transfer all data points, which allows us to curate the whole CommonCrawl since 2013 with 300B+ scale URL-text pairs without spending the storage/transfer on all $\sim10\times$ data points, where the rate of keeping a data point with MetaCLIP curation is $\sim0.1(0.5\times0.23)$ .

![](images/48275c845604b357e078abc1a8021e9e2e235ca108826772371ebd9b06072865.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["HTML Parsing/LID"] --> B["URLs & Text Dedup"]
    B --> C["Image Download"]
    C --> D["NSFW Image Dedup"]
    D --> E["Packaging"]
    F["Part1: Sub-string Matching"] --> B
    G["Part2: Balancing"] --> C
    style A fill:#cce5ff
    style B fill:#cce5ff
    style C fill:#cce5ff
    style D fill:#cce5ff
    style E fill:#cce5ff
    note1["50% reduction"] --> F
    note2["77% reduction"] --> G
```
</details>

Figure 5: Case study: Curation implementation in our data pipeline.

Curation in Data Loading We applied the balancing/sampling part of the algorithm to the data loader to adjust data distribution on-the-fly. Although data points for tail entries are always sampled in Algorithm 1, the diverse pairs from the head entries are sub-sampled from a larger pool while maintaining a similar distribution as offline curation. This diversification of pairs that match the head entries improved the performance, as shown in Table 6.

# A.3 HUMAN STUDY ON THE EFFECTS OF CURATION

In this section, we study the impact of MetaCLIP curation on data distribution by using human evaluation. We approach this exploration from three distinct angles: noise reduction, alignment of visual content, and task-agnostic attributes. In the pursuit of comprehending the first two aspects, we undertake a human study aimed at comprehending the data quality implications of implementing the balancing technique (outlined in Part 2 of Algorithm 1). This evaluation encompasses three dimensions: image-only, text-only, and image-text alignment. We collect an evaluation set of 100 random image-text pairs for balanced and unbalanced data, respectively, and ask annotators to score on the image, text, and pair quality metrics, separately, on a scale of 1 to 5.

Annotation Guidelines. Annotators follow guidelines to assess both images and texts, evaluating informativeness (how well information is conveyed) and aesthetics. For images, aesthetics considers visual elements like composition, color, lighting, balance, contrast, texture, and subject matter. For texts, aesthetics gauges factors like delimiters, sentence structure, capitalization, prefixes/suffixes, recognized words, generic words, and overall text quality. The alignment metric for image-text pairs measures the relevance between the two modalities, assessing how well the text describes the image content. Ratings are averaged across annotators for each dimension.

We show the study results in Table 10 and discuss the different criteria next.

<table><tr><td>Evaluation Dimension</td><td>Rating for Balanced Data</td><td>Rating for Unbalanced Data</td><td>P-value</td></tr><tr><td>Image</td><td>4.60, [4.50, 4.70]</td><td>4.36, [4.23, 4.48]</td><td>&lt; 0.001</td></tr><tr><td>Text</td><td>4.67, [4.56, 4.78]</td><td>4.06, [3.82, 4.30]</td><td>&lt; 0.001</td></tr><tr><td>Alignment</td><td>4.41, [4.23, 4.59]</td><td>3.72, [3.46, 3.99]</td><td>&lt; 0.001</td></tr></table>

Table 10: Average human rating on the effect of balancing on data quality, with confidence intervals shown in parentheses. Higher rating is better. Balanced data is rated of higher quality.

Noise Mitigation in Image and Text. As shown in Table 10, significant quality improvement for all the three evaluation dimensions is observed after applying balancing. MetaCLIP curation has no specific hard filters such as removing shorter text, removing dates, etc. However, curation by sub-string matching and balancing has a different filtering effect. For example, a sub-string itself can never curate a date-only text. Further, balancing allows signal and noise to co-exist when they are difficult to be separated by human designed filters. For example, if one entry such as “image” or “photo” is capped to t = 20k, it can only contribute 0.005% of 400M data.

Visual Content Alignment. Although MetaCLIP curation does not directly involve images, it has a positive effect on aligning visual content by controlling the quality and distribution of text. First, sub-string matching increases the chance of having (visual) entities mentioned in the text, thereby improving the likelihood of finding corresponding visual content. Second, balancing favors long-tailed entries that could have more diverse visual content than a head entry (such as the text “1”). In Table 10, we observe significant improvement on pair quality from unbalanced data to balanced data.

# A.4 MEASURING TASK-ALIGNMENT

With metadata, one can measure the alignment of pre-training data distribution with the data distribution in downstream tasks. First we can do a simple measure of counting the number of metadata substring matches that are directly corresponding to class names in the downstream tasks. In the first row of Table 11, we show the accuracy of MetaCLIP (400M), ViT-L (cf. Table4) for reference. The second row shows the number of classes that are present in the metadata, for each downstream dataset. For example, for ImageNet 703 of the 998 unique class names are present in the metadata. Interestingly, there seems to be a correlation with the accuracy and the number of classes matched in the metadata.

<table><tr><td></td><td>ImagNet</td><td>Food-101</td><td>CIFAR10</td><td>CIFAR100</td><td>CUB</td><td>SUNS97</td><td>Cars</td><td>Aircraft</td><td>DTD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td><td>MNIST</td><td>FER-2013</td><td>STL-10</td><td>EuroSAT</td><td>RESISC45</td><td>GTSEB</td><td>KITTI</td><td>Country211</td><td>PCAM</td><td>UCF101</td><td>Kinetics700</td><td>CLEVR</td><td>HandfulMemes</td><td>SST2</td></tr><tr><td>MetaCLIP (400M) ViT-L</td><td>76.2</td><td>90.7</td><td>95.5</td><td>77.4</td><td>75.9</td><td>70.5</td><td>84.7</td><td>40.4</td><td>62.0</td><td>93.7</td><td>94.4</td><td>76.4</td><td>61.7</td><td>46.5</td><td>99.3</td><td>59.7</td><td>71.9</td><td>47.5</td><td>29.9</td><td>30.9</td><td>70.1</td><td>75.5</td><td>57.1</td><td>35.1</td><td>56.6</td><td>65.6</td></tr><tr><td># of cls. w/ non-zero counts</td><td>703/998</td><td>52/101</td><td>10/10</td><td>93/100</td><td>1/200</td><td>193/397</td><td>0/196</td><td>8/100</td><td>40/47</td><td>15/37</td><td>86/102</td><td>61/102</td><td>10/10</td><td>12/12</td><td>10/10</td><td>2/10</td><td>32/45</td><td>1/43</td><td>0/4</td><td>190/211</td><td>1/2</td><td>5/101</td><td>122/700</td><td>8/8</td><td>1/2</td><td>2/2</td></tr><tr><td colspan="27">KL-divergence</td></tr><tr><td>- unbal.</td><td>6.8</td><td>9.4</td><td>7.5</td><td>6.1</td><td>11.4</td><td>6.7</td><td>0.0</td><td>12.0</td><td>9.7</td><td>11.7</td><td>7.7</td><td>10.6</td><td>3.4</td><td>8.9</td><td>7.1</td><td>8.4</td><td>7.6</td><td>9.3</td><td>0.0</td><td>5.7</td><td>14.6</td><td>8.9</td><td>8.9</td><td>3.8</td><td>9.2</td><td>9.8</td></tr><tr><td>- bal.</td><td>5.1</td><td>7.4</td><td>8.1</td><td>6.0</td><td>10.4</td><td>6.0</td><td>0.0</td><td>10.1</td><td>8.0</td><td>9.7</td><td>7.1</td><td>8.6</td><td>8.1</td><td>8.3</td><td>8.1</td><td>9.7</td><td>7.2</td><td>10.4</td><td>0.0</td><td>5.2</td><td>12.5</td><td>8.8</td><td>7.1</td><td>8.3</td><td>10.4</td><td>9.7</td></tr></table>

Table 11: Measuring task-alignment. First row: MetaCLIP (400M) ViT-L/14 accuracy, second row: number of classes matched in metadata, Third and fourth row: We use KL-divergence (lower is better) to measure the similarity between pre-training distribution and benchmark task distribution. Note that, for ImageNet, two classes have duplicated class names and therefore there are only 998 unique classes (out of 1000).

<table><tr><td>Hyperparameter</td><td>OpenAI CLIP / MetaCLIP</td><td>OpenCLIP</td><td>DataComp</td></tr><tr><td>Activation Function</td><td>QuickGELU</td><td>GELU</td><td>GELU</td></tr><tr><td>Seen Pairs</td><td>12.8B(400M×32 epochs)</td><td>13B (407M×32 epochs)</td><td>12.8B</td></tr><tr><td>Batch Size</td><td>32768</td><td>32768 (B/32), 33792 (B/16), 38400 (L/14)</td><td>90112 (L/14)</td></tr><tr><td>Learning Rate</td><td>5.0e-4(B/32,B/16), 4.0e-4(L/14)</td><td>5.0e-4(B/32)</td><td>1e-3(L/14)</td></tr><tr><td>Warm-up</td><td>2k</td><td>2k (B/32)</td><td>10k (L/14)</td></tr></table>

Table 12: Hyperparameters of OpenAI CLIP vs OpenCLIP on LAION-400M(Schuhmann et al., 2021) and DataComp 1B.

Next, we use KL-divergence to measure alignment:

$$
D _ {\mathrm{KL}} (T | | P) = - \sum_ {m \in \mathcal {M}} T (m) \log \left(\frac {P (m)}{T (m)}\right), \tag {1}
$$

where $T(m)$ represents the task distribution over M and $P(m)$ represents the pre-training data distribution. We compute $T(m)$ using the benchmark data distribution over class labels of a task. Taking ImageNet as an example from Table 11, $T(m)$ ImageNet has 998 entries uniformly distributed (each has evenly 50 examples) and $P(m)$ has normalized counts over all 500k entries but only 703 entries used to compute KL-divergence. In the third and fourth row of Table 11, we can see improvements for balanced data points over unbalanced ones for each task, showing balancing improves similarity with most tasks.

# A.5 TRAINING SETUP OF OPENAI CLIP VS OPENCLIP

Our work strictly follows CLIP's setup for a controlled comparison focusing on data curation and quality. We notice differences in the training setup of OpenCLIP $^{5}$ and list the difference (known to us). OpenCLIP varies the setup from CLIP (e.g., global batch size, learning schedule, etc.). Here we only list the difference for LAION-400M(Schuhmann et al., 2021), which is closer to the CLIP setup. We note that DataComp differs even more, e.g., by curating images close to ImageNet training data, a large batch size of 90k that is almost $3 \times$ larger than CLIP, and using the CLIP model to filter data.

# A.6 BENCHMARK DEDUPLICATION

Our pools are deduplicated from the benchmark/ImageNet data using a 64-bit PCA hash, derived from a similarity search model's feature embeddings with PCA reduction to 64 dimensions and sign quantization. DataComp-12.8B is already deduplicated.

# A.7 NEGATIVE RESULTS LEARNED FROM ABLATING CLIP CURATION

We briefly describe a few ideas close to CLIP curation that did not look promising in our initial attempts and were abandoned:

1. Self-curated Metadata. We initially attempted to build metadata directly from the text in raw image-text pairs (i.e., using terms appearing in text above a certain threshold of counts). We rank entries by count and keep the top 500,000. Metadata built this way appeared worse. We notice that although the top frequent entries are similar to CLIP's metadata, the long-tailed part is very different. For example, the minimal count to be in the 500,000 budget needs at least 130 counts. In contrast, our metadata has 114K entries that have no matches. This approach results in worse quality metadata including low-quality spelling/writing (instead of high-quality entries from WordNet or Wikipedia). Further, the effect of balancing saturates earlier for such data (in a larger $t$ , verified by CLIP training) since low-quality entries are also heavily in long-tail.   
2. Cased WordNet. We also notice many cases words are missing from metadata (e.g., WordNet is in lowercase). After adding cases WordNet into metadata, we notice a performance drop on ImageNet. The reason could be class names are more likely in lower case and upper case entry matching may reduce the written quality of texts.

3. Stopwords/Useless Entries Removal We further study the effect of whether removing stopwords and useless words such as “photo” and “image” is beneficial. This led to almost no difference since balancing entries reduced the effects of useless entries (each entry contributes to 0.0002% (1/500k) level of the total data points). To encourage a simplified solution, we do not intend to add more artificial filters.

# A.8 MORE DETAILS ON DISTRIBUTION ON META DATA

Extending the top-20 matches in Table 3, we further group counts of metadata entries and show 5 examples per group as in Table 13.

<table><tr><td>Group</td><td>5 Examples (Entry:Count)</td></tr><tr><td>0-10k</td><td>ivailo:12, Kunta Kinte:201, vikernes:33, peria:50, ankoku:20</td></tr><tr><td>10k-20k</td><td>queer:19k, barry:10k, bandages:12k, The Police:15k, sigma:14k</td></tr><tr><td>20k-50k</td><td>polygonal:21k, widely:28k, however:35k, toppers:25k, executives:21k</td></tr><tr><td>50k-100k</td><td>planted:52k, olive oil:58k, yours:63k, packages:82k, Spokane:53k</td></tr><tr><td>100k-500k</td><td>compact:133k, vertical:222k, underwear:111k, powder:323k, weekly:130k</td></tr><tr><td>500k-1M</td><td>Tokyo:713k, Lead:620k, Diagram:809k, Dye:858k, unnamed:512k</td></tr><tr><td>1M-50M</td><td>see:1.4M, Door:3.2M, News:2.3M, sea:1.1M, street:1M</td></tr><tr><td>50M-130M</td><td>with:67M, and:100M, to:61M, in:107M, of:121M</td></tr></table>

Table 13: Distribution of metadata entries with counts, similar as head shown in Table 3.

# A.9 RANDOMNESS OF ALGORITHM 1

We further study the randomness of algorithm 1 by running it 3 times for Pool 1(400M) and Pool 2(2.5B). This ended with a standard deviation of 4035 examples for Pool 1 and 2104 examples for Pool 2, respectively.

# A.10 QUALITATIVE DATA EXAMPLES

In Table 14, we illustrate data before/after sub-string matching and balancing. We also highlight class labels from ImageNet in the table. We mark a matched entry with a bigger font size indicating higher probability of sampling that entry. Intuitively, sub-string matching removes low quality text and balancing favors longer text with long-tail entities to improve data quality. In Table 15, we show more examples of matched text that include ImageNet tail entries.

<table><tr><td>Text</td><td>Substr.</td><td>Bal.</td><td>IN Head</td><td>IN Tail</td></tr><tr><td>control_14ct</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>cudd2008</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>product-img</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>Skirmisher-Main-440x412</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>A4omote</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>How-to-Find-the-Finest-Electric-Car-Companies</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>hanukkah-party-facebook-event-cover-template</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>johnny_cash_chili_dog (2)</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>8533 GOLDEN RIDGE COURT</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>How to build a stone patio on your own</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>battery plate</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>barn wedding</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Picture sand, machine, Concept, jeep, the concept, the front, Slim, Wrangler, Jeep</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>desk</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Adult T-shirt</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Imix m10 stage moniter</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>google wallet md3</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Distressed beach bar sign - Pearly&#x27;s Oyster Bar</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Why the Kilimanjaro Trek should be top of your bucket list</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>J70 desk model</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Whitby castle</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Inside the castle</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Hama Hama Oyster Saloon | restaurant | 35846 US-101, Lilliwaup, WA 98555, USA | 3608775811 OR +1 360-877-5811</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>beach</td><td>√</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Caramelized onions, sauteed red bell peppers and zucchini combined create a winning egg frit-tata breakfast dish.</td><td>√</td><td>√</td><td>√</td><td>✕</td></tr><tr><td>Vector layered paper cut craft style music composition of saxophone guitar trumpet violin music instruments, notes on abstract color background. Jazz concert festival party poster banner card template</td><td>√</td><td>√</td><td>√</td><td>✕</td></tr><tr><td>Nautilus hot tub</td><td>√</td><td>√</td><td>√</td><td>✕</td></tr><tr><td>night binoculars for hunting</td><td>√</td><td>√</td><td>√</td><td>✕</td></tr><tr><td>2017 cat eyes women&#x27;s sunglasses for women vintage sun glasses round women sun glasses oculos oculos de sol feminino</td><td>√</td><td>√</td><td>√</td><td>✕</td></tr></table>

Table 14: Examples categorized by whether passing sub-string matching and balancing: Words in violet color are in metadata and their font size indicates probability of being sampled, ranging from 13pt that has probability close to 0, to 22pt that has probability 1. ImageNet labels in head entries are cyan.

<table><tr><td>Text</td><td>Substr.</td><td>Bal.</td><td>IN Head</td><td>IN Tail</td></tr><tr><td>Antique German sterling silver 800 cup julep goblet, 1</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>A journey to the East Sea by high-speed train: New KTX line makes Gangneung&#x27;s wonders all the more accessible</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>little arrow design co watercolor rainbow blush shower curtain and mat</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>photo of antique silver top break revolver trombone model</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>Staffordshire Bull Terrier</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>KI Plantation Timbers fire truck on the night of January 3, 2020. Photography: Tim Wilson</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>water buffalo bath</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>Basset Hound pup with Lionhead x Lop rabbit</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>Single serving of peach trifle</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>A tarantula (hairy arachnid belonging to the Theraphosidae family of spiders) in a box, standing still. Close-up shot.</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>Modern Staffordshire Bull Terrier LED Night Light Animal Pet Dog Puppy 3D Optical illusion Lamp Home Decor Table Lamp Desk Light</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>Amazon, rocking chair, CANVA, camping chairs</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>dispalying kukri machete fitted inside scab-bard</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>jacksons chameleon</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>Wall Mural - Insects pollinating blooming rapeseed crops in field</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr><tr><td>Scottish Terrier Phone Pocket</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td></tr></table>

Table 15: Examples passing both sub-string matching and balancing wit ImageNet tail classes: Words in violet color are in metadata and their font size indicates probability of being sampled, ranging from 13pt that has probability close to 0, to 22pt that has probability 1. ImageNet labels in tail entries are in cyan.

# REFERENCES

Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540, 2023.   
Olivier Bachem, Mario Lucic, and Andreas Krause. Coresets for nonparametric estimation-the case of dp-means. In International Conference on Machine Learning, pp. 209–217. PMLR, 2015.   
Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. arXiv preprint arXiv:2212.07143, 2022.   
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.   
Dan Feldman, Matthew Faulkner, and Andreas Krause. Scalable training of mixture models via coresets. Advances in neural information processing systems, 24, 2011.   
Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, and Ludwig Schmidt. Datacomp: In search of the next generation of multimodal datasets, 2023.   
Sariel Har-Peled and Soham Mazumdar. On coresets for k-means and k-median clustering. In Proceedings of the thirty-sixth annual ACM symposium on Theory of computing, pp. 291–300, 2004.   
Mon-Fong Jiang, Shian-Shyong Tseng, and Chih-Ming Su. Two-phase clustering process for outliers detection. Pattern recognition letters, 22(6-7):691–700, 2001.   
Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning, pp. 6950–6960. PMLR, 2020.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.   
Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.   
Christoph Schuhmann, Romain Beaumont, Cade W Gordon, Ross Wightman, Theo Coombes, Aarush Katta, Clayton Mullis, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. 2022.   
Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523–19536, 2022.   
Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. An empirical study of example forgetting during deep neural network learning. arXiv preprint arXiv:1812.05159, 2018.   
Hu Xu, Saining Xie, Po-Yao Huang, Licheng Yu, Russell Howes, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Cit: Curation in training for effective vision-language data. arXiv preprint arXiv:2301.02241, 2023.

Dantong Yu, Gholamhosein Sheikholeslami, and Aidong Zhang. Findout: Finding outliers in very large datasets. Knowledge and information Systems, 4:387–412, 2002.