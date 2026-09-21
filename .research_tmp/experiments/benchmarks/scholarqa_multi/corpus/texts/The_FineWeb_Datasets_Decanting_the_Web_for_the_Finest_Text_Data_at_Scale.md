# The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

Guilherme Penedo Margaret Mitchell

Hynek Kydlicek  
Colin Raffel

Loubna Ben allaleandro Von Werra

Anton Lozhkov Thomas Wolf

Hugging Face

https://huggingface.co/datasets/HuggingFaceFW/fineweb  
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

# Abstract

The performance of a large language model (LLM) depends heavily on the quality and size of its pretraining dataset. However, the pretraining datasets for state-of-the-art open LLMs like Llama 3 and Mixtral are not publicly available and very little is known about how they were created. In this work, we introduce FineWeb, a 15-trillion token dataset derived from 96 Common Crawl snapshots that produces better-performing LLMs than other open pretraining datasets.

To advance the understanding of how best to curate high-quality pretraining datasets, we carefully document and ablate all of the design choices used in FineWeb, including in-depth investigations of dedduplication and filtering strategies. In addition, we introduce FineWeb-Edu, a 1.3-trillion token collection of educational text filtered from FineWeb. LLMs pretrained on FineWeb-Edu exhibit dramatically better performance on knowledge- and reasoning-intensive benchmarks like MMLU and ARC.

Along with our datasets, we publicly release our data curation codebase and all of the models trained during our ablation experiments.

# 1 Introduction

Large Language Models (LLMs) have quickly become a ubiquitous technology thanks to their ability to competently perform a wide range of text-based tasks. A driving factor in the success of LLMs has been a steady increase in model sizes [1-3], which in turn necessitate ever-larger pretraining datasets. Beyond scale, other characteristics of pretraining data have proven to be important, including filtering out "low-quality" content [2, 4] and removing duplicate text [5].

Ultimately, the curation choices made when developing a pretraining dataset can have a huge impact on the downstream capabilities and performance of an LLM. As such, pretraining dataset curation strategies are often treated as closely guarded trade secrets. In fact, there are many popular "open" language models whose parameters are publicly available but whose pretraining datasets were not released and are scarcely documented [6, 7].

The lack of access to high-quality large-scale pretraining datasets and lack of information about their curation has led to concerns of a growing gap between proprietary and public knowledge.

In this work, we aim to minimize this gap by developing and releasing the FineWeb datasets, a collection of large-scale pretraining datasets that can be used to train performant LLMs. Specifically, we first introduce FineWeb, a 15-trillion token dataset of text sourced from 96 Common Crawl snapshots. FineWeb is sufficiently large to train a Chinchilla-optimal model [1] with more than 500 billion parameters.

Beyond scale, FineWeb's recipe involves a principled strategy for choosing and tuning filtering heuristics that helped produce a small set of effective filters out of over fifty candidate filters from past work. In addition, we performed an in-depth exploration of how different dedduplication strategies and granularities can impact performance. To validate our design choices, we ultimately demonstrate that models trained on FineWeb perform better than those trained on other

arXiv:2406.17557v2 [cs.CL] 31 Oct 2024

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks.

public web-based pre-training datasets. Inspired by recent work advocating for training LLMs on educational data [8, 9], we additionally introduce FineWeb-Edu, a subset of 1.3 trillion tokens from FineWeb that was rated as highly educational by a custom classifier. Models trained on FineWeb-Edu exhibit significantly better performance on knowledge- and reasoning-intensive benchmarks like MMLU [10] and ARC [11]. Both datasets are released under the permissive ODC-By License. Apart from contributing datasets, we also release datatrove [12], the data processing library we developed to create FineWeb. On the whole, our work represents a significant step towards improving public knowledge and resources for curating LLM pre-training datasets.

# 2 Background

In this work, we focus on the curation of training datasets for autoregressive Transformer-based large language models (LLMs) [13]. At their core, LLMs aim to produce a distribution over the next token of text conditioned on past tokens, where each token is typically a word or subword unit [3]. The generality of this paradigm allows LLMs to be applied to virtually any text-based task by formulating a prefix whose continuation corresponds to performing the task (e.g. "The cat sat on the mat translated to French is..." for English-to-French translation).

Such models may undergo many stages of training including pretraining on unstructured text data, fine-tuning to improve performance on a specific task [14], multitask fine-tuning to improve generalization to new tasks [15], and learning from human feedback to improve instruction-following capabilities [16, 2]. In this work, we focus solely on curating data for the pretraining stage.

While many sources have been considered for pretraining data including text from books [2, 3, 17, 4], Wikipedia [2, 3, 17, 4], and research papers [2, 4, 18], a highly common choice is to use web text, i.e. text scraped from webpages on the public internet [19, 20]. While some companies like OpenAI [21] and Anthropic [22] perform their own web scrapes, designing, implementing, and running a web scraper at scale requires significant resources and expertise. Many LLM pretraining datasets have therefore been constructed from text from the Common Crawl [23], a publicly available and continually updated collection of website snapshots that has been running since 2007. As of writing, Common Crawl has produced 100 web snapshots totaling petabytes of data.

Although Common Crawl has produced more than enough data to train recent LLMs, it has been shown that the performance of an LLM can heavily depend on how web text has been filtered and preprocessed before being used for pretraining [19]. In particular, web text can contain a large amount of "unnatural" language (e.g. "boilerplate" text, gibberish, etc.). Training on unnatural language data can harm the performance of LLMs, possibly because most downstream uses of LLMs do not involve such data.

On the other hand, filtering out too much content can produce a dataset that is too small to perform sufficient pretraining (which typically involves only one pass or a few passes over the pretraining dataset [24]) for a general use model. Separately, web text can contain a large amount of duplicated content, which has also been shown to be harmful in the context of pretraining data [5]. While dedduplication may seem as straightforward as "removing duplicate text", in practice many design choices must be made (line, paragraph, or document-level dedduplication? fuzzy or exact matching? etc.).

The curation and relative performance of different web text-based pretraining datasets therefore heavily depends on a given dataset's filtering and dedduplication pipeline.

Given that the focus of our work is to carefully design an effective Common Crawl-based pretraining dataset, we now briefly discuss the filtering and dedduplication used in past public datasets. OSCAR [25] processes Common Crawl using a pipeline inspired by that of Touvron et al. [2], which uses a fastText-based language classifier [26] to filter pages based on their language and then performs dedduplication at the line level using a non-cryptographic hash algorithm.

C4 [15] uses langdetect [27] to filter out non-English pages, then applies a series of heuristic filters (retaining only those lines that end in a terminal punctuation mark, re
moving short lines, discarding any page that contains a word from a "bad words" list, etc.), and finally deduplicates over three-line windows. CC-100 [28] uses the cc_net pipeline, which includes fastText for language identification, performs paragraph-level hash-based dedduplication, and retains only the text that is assigned a low perplexity by a n-gram language model trained on Wikipedia.

The Pile [29] is a composite dataset that includes "Pile-CC", a collection of text from one Common Crawl snapshot that uses pycld2 [30] for language detection, jusText [31] for boilerplate removal, a classifier to filter out pages that are dissimilar form WebText (described below), and fuzzy dedduplication using MinHash [32]. ROOTS [33] includes text from the pre-processed web crawl OSCAR with additional heuristic-based filtering and SimHash

2

based dedduplication. RedPajama [34] is a composite dataset that includes Common Crawl-sourced text processed using the cc_net pipeline as well as quality filtering using a classifier trained to distinguish Wikipedia level content from random Common Crawl samples. SlimPajama [35] further processed RedPajama by removing short documents and performing additional fuzzy MinHash-based dedduplication.

RefinedWeb [36] uses trafilatura [37] for text extraction, fastText for language identification, heuristic rules inspired by MassiveText (discussed below) to filter data, and both MinHash (fuzzy) and ExactSubstr (exact) dedduplication. RedPajama v2 [34] has 84 Common Crawl snapshots released unfiltered and non-deduplicated but with labels from filtering techniques from cc_net, C4, MassiveText, RefinedWeb and others, as well as dedduplication labels for exact (Bloom filter) and fuzzy (MinHash) dedduplication.

Finally, Dolma [38] has a Common Crawl-based subset that uses fastText for language classification, heuristic rules from MassiveText and C4 for quality filtering, rules- and classifier-based toxicity filtering, and URL, document and paragraph-level dedduplication using a Bloom filter.

Apart from public datasets, the technical reports accompanying the announcement of closed LLMs occasionally discuss pretraining datasets. WebText [20] (used to train GPT-2) involves only those non-Wikipedia webpages that were linked to from Reddit posts with at least 3 karma, with text extracted using Dragnet [39] and Newspaper1 [40] and an unspecified dedduplication pipeline. GPT-3's Dataset [3] includes content from Common Crawl that has been filtered using a classifier trained on WebText, Wikipedia, and Books, and deduplicated using MinHash.

MassiveText [41] (used to train Gopher) is a web-based dataset using Google's SafeSearch to remove explicit content and heuristic filters based on document's content (number of words, stop-words appearance, character repetition, etc.) as well as MinHash-based dedduplication.

# 3 Building FineWeb

Our design of FineWeb is primarily empirical: we performed a series of "data ablation" experiments to test different methods at each stage of the pipeline. In this section, we chronicle our experimental results and design choices. All ablations follow our iterative dataset building process, i.e., the baseline model for each subsection includes only the processing steps from the previous subsections, unless explicitly mentioned. Our results are fully reproducible with code in our datatrove repository.

# 3.1 Experimental setup

We compare pipeline design choices at each stage by training data ablation models that are identical apart from the data they were trained on (same number of parameters, architecture hyper-parameters, and trained on an equal number of randomly sampled tokens from each version of the data). We then evaluated them on the same set of downstream task benchmark datasets (discussed below). To minimize the impact of random data subset selection on evaluation scores, we trained two models for each dataset version, each using a different but equal-sized random subset of the full data and a different initialization seed, and then compared average scores.

All training was performed using the nanotron library. Data ablation models all had 1.71B parameters (including embeddings), used the Llama architecture [2] with a sequence length of 2048, a global batch size of $\sim 2$ million tokens, and the GPT2 tokenizer [20]. Within a given experiment, all models were trained on the same amount of data for the same number of steps.

Filtering ablations were trained on $\sim 28$ billion tokens (roughly the Chinchilla-optimal training size for this model size [1]), while some dedduplication ablations and runs to confirm cumulative relative performance improvements after each step of filtering were conducted on 350 billion tokens. The full training hyperparameter are available in Appendix D. We make all models trained for our ablations publicly available on our dataset repository. In total, we trained over 70 models on our internal cluster, for an estimated total of 80,000 H100 GPU hours.

Evaluation was performed using the lighteval library. We aimed to select a set of benchmarks that would provide good signal at the relatively small scale of our data ablations. Specifically, we chose benchmarks where models showed minimal score variance between runs trained on different random samples of the same dataset; monotonic (or nearly monotonic) score improvement over a given training; and scores above random baseline for models of this size. These criteria ensure that the scores obtained on a subset of the data are representative of the entire dataset and that they reflect a reliable measurement of the effect of the training data on model performance. Ultimately,

3

we selected benchmark datasets CommonSense QA [42], HellaSwag [43], OpenBook QA [44], PIQA [45], SIQA [46], WinoGrande [47], ARC [11], and MMLU [10], truncating large benchmarks to 1000 samples so that we could efficiently evaluate over the course of training. We publicly release our exact evaluation setup.

# 3.2 Text extraction

Common crawl data is available in two different formats: WARC and WET. WARC (Web ARChive format) files contain the raw data from the crawl, including the full page HTML and request metadata. WET (WARC Encapsulated Text) files provide a text-only version of crawled websites by using htmlparser [48]. While WET files are commonly used as a starting point for dataset creation, similarly to Gao et al. [29], we found that WET files retained too much boilerplate and menu text.

We therefore experimented with extracting the text content from the WARC files using the open source trafilatura library [49], which from visual inspection of the results provided good quality extraction when compared to other available libraries (less boilerplate and menu text). Custom text extraction is relatively costly, but its effects are felt on model performance: Fig.

1 shows the performance of ablation models trained on either trafilatura applied to WARC data or WET data, with minimal additional filtering (fastText language identification to filter samples with English as the highest probability label) and no dedduplication. Using trafilatura-extracted text from WARC files clearly results in a more performant model and we therefore use WARC-based data in all of our following experiments.

![](dt=2026-05-30/ht=00/3275a95ec1efabcddb8f1d279f28f1a678439bd55b10f496534d5e67f97e1a33.jpg)

![](dt=2026-05-30/ht=00/26398a7e7425b4446d6309263d07d32743870ad39c08464d2986c519461cff52.jpg)

# 3.3 Base filtering

As a starting point to our filtering, we applied a basic filtering pipeline using part of the setup from RefinedWeb [50]. Concretely, we applied URL filtering using a blacklist [51] to remove adult content, applied a fastText language classifier [52, 26] to keep only English text with a score $
> = 0.65$ , and applied quality and repetition filters from MassiveText [41], using the original thresholds. After applying this filtering to all of the WARC-based text extracted from the 96 snapshots available at the time of writing, we obtained roughly 36 trillion tokens of data when tokenized with the GPT-2 tokenizer. Applying these steps results in a performance uplift, as seen in Fig. 2.

# 3.4 Dedduplication

The web has many aggregators, mirrors, templated pages or just otherwise repeated content spread over different domains and webpages. Removing these duplicates (deduplicating) has been correlated with improvements in model performance [5] and a reduction in memorization of pretraining data [53, 54]. There are different ways to identify and even define duplicated data. Common approaches rely on hashing techniques or efficient data structures like suffix arrays [55]. Methods can also be "fuzzy" by using a similarity metric or "exact" by checking for exact matches between two text chunks [56].

4

Following RefinedWeb [50], we experimented with MinHash, a fuzzy hash-based deduplication technique that scales efficiently to many CPU nodes and allows tuning of similarity thresholds (by controlling the number and the number of hashes per bucket) as well as the length of the subsequences considered (by controlling the n-gram size). We chose to collect each document's 5-grams, obtained using an English word tokenizer [57], and computed MinHashes using 112 hash functions in total, split into 14 buckets of 8 hashes each — targeting documents that are at least $75\%$ similar.

Documents with the same 8 MinHashes in any bucket are considered duplicates of each other. We then perform a transitive clustering step where documents A, B and C will be in the same duplicate cluster if A and C are duplicates and B and C are duplicates, even if A and B do not have 8 matching MinHashes in any bucket with each other. One (randomly chosen) document is kept per duplicate cluster while the remaining duplicates are removed. We further discuss dedduplication parameters in Appendix E.1.

![](dt=2026-05-30/ht=00/719161bb9f55e382d73012a54b97c7fb55a88fc07389ab9cf36128bd60671a0d.jpg)

![](dt=2026-05-30/ht=00/f96e01f7c46ebb16e9b903195faf6423bbe1c1289695cfe66097ad42763f9cca.jpg)

Our first approach was to apply MinHash dedduplication globally to the entire dataset (all 96 snapshots). We did this in an iterative manner: starting with the most recent snapshot (2023-50, at the time the experiment was run) and proceeding chronologically until we reached the oldest snapshot. When applied to the oldest snapshots, this process removed as much as $90\%$ of the original base filtered data, as they were deduplicated against a large number of other snapshots. Deduplicating the entire dataset in this manner resulted in 4 trillion tokens of data.

However, when training on a randomly sampled 350 billion tokens subset, our ablation models showed little improvement over a model trained on the non-deduplicated data, scoring far below RefinedWeb on our aggregate of tasks, as shown in Fig. 3.

This challenged our initial assumption that global dedduplication would inevitably result in higher benchmark scores. We therefore performed an additional experiment to investigate the quality of the remaining data.

We trained two models on two slices from the older 2013-48 snapshot: (a) the fully deduplicated remaining $\sim 31$ billion tokens (originally kept data); and (b) 171 billion tokens obtained by individually deduplicating (without considering the other crawls) the $\sim 460$ billion tokens that had been removed from this crawl in the iterative dedduplication process (originally removed data). Results are presented in Fig. 4.

They show that, for this older crawl taken in isolation, the data from it that was kept (10% of the original data) was actually of worse quality than the 90% of data that was removed. We confirmed this by visual inspection: originally kept data contains more ads, incoherent lists of keywords and generally badly formatted text than originally removed data.

We therefore tried an alternative approach: individually deduplicating each snapshot (independently from the others), using the same parameters as before. This resulted in 20 trillion tokens of data. When training on a random sample from this dataset (with data sampled from all snapshots) it matched RefinedWeb's performance, as per Fig. 5.

One of our hypotheses is that the main improvement gained from dedduplication lies in the removal of large clusters of duplicates with hundreds of thousands of documents [50] present in all crawls, while further dedduplication of clusters with a small number of duplicates (less than $\sim 100$ , i.e., the number

5

of crawls) can harm performance. More specific filtering targeting the long tail of data quality might be more suited than dedduplication for this subset of the data.

To attempt to improve upon individual dedduplication of each snapshot, we additionally experimented with "lighter" global dedduplication techniques. Ultimately, none of these techniques improved performance over independent per-snapshot dedduplication. A full description of these methods and results are in Appendix E.3.

# 3.5 Adding C4's filters

By this point we had reached the same performance as RefinedWeb [50] using our base filtering and independent MinHash. However, we noted that the C4 dataset [15], while smaller than FineWeb, still showed stronger performance on some of the benchmarks in our evaluation suite, in particular HellaSwag [43], one of the benchmarks in our aggregate group of tasks with the highest signal-to-noise ratio. Despite being one of the first large scale LLM training datasets, C4 is still fre

![](dt=2026-05-30/ht=00/c2ac08db8d29a2dfbccebe8a3af49f50662f315f7b1a7f53d814c4001345282e.jpg)

sequently part of the pretraining data mixture of recent models such as LlamaA 1 [2]. We set out to explore additional filtering steps that would allow us to match or surpass the performance of C4. A natural starting point was to look into the processing of C4 itself.

![](dt=2026-05-30/ht=00/6430bcb3fb26a6b599f50d2009e5a4ddd4f261a79bb26bb46a3b44cfd8bde65d.jpg)

![](dt=2026-05-30/ht=00/2ccbf005d3916c0f7c12051fb7365e15ffccc19692d30663e2b42d0b93e479e7.jpg)

C4 was constructed from the 2019-18 crawl by applying heuristic filters, which included dropping lines without a terminal punctuation mark, that mentioned javascript, or that had "terms-of-use"/"cookie policy" statements, and dropping documents that were too short or that contained "lorem ipsum" or a curly bracket (\{). We experimented with applying these filters to a baseline of the base filtered and individually deduplicated 2019-18 crawl, and, additionally, compared the results to C4 itself.

Fig. 6 shows that applying All filters allows us to match C4's HellaSwag performance; the Curly bracket filter, and the Word lengths filter only give a small boost, removing $2.8\%$ and $4.3\%$ of tokens, respectively; the Terminal punctuation filter, by itself, gives the biggest individual boost, but removes around $30\%$ of all tokens; the lorem_ipsum, javascript and policy rules each remove $<0.5\%$ of training tokens, so we did not train on them individually; All but terminal punct performs better than terminal_punct by itself, while removing less in total ( $\sim7\%$ ). We decided to apply all C4 filters mentioned above except the terminal punctuation filter, as it eliminates an excessively large amount of data.

6

# 3.6 Developing additional
heuristic filters

Past work has mainly developed heuristic filters through data inspection [15]. In this work we devised a more systematic process for designing heuristic filters and tuning their thresholds. We started by collecting over 50 high-level statistics ranging from document-level metrics (e.g. number of lines, avg. line/word length, etc) to inter-document repetition metrics (inspired by MassiveText [1]) on both a high- and low-quality web dataset.

Specifically, we used the individually and globally deduplicated versions of the 2013-48 snapshot (previously mentioned in Section 3.4) as our "high-quality" and "low-quality" datasets respectively. We then identified metrics for which the distribution of values differed significantly across the two datasets, inspected the histograms of the two distributions and empirically chose thresholds that would target sections of the histogram where the lower quality dataset frequency was higher than on the corresponding higher quality dataset section.

As an example, we plot the distribution of the fraction of lines ending with punctuation metric in Fig. 8. We can see that the higher quality dataset has in general higher document density for larger values of our metric, and, in particular, the lower quality dataset has a much higher density of documents for values $< 0.12$ . We thus conclude that documents with a fraction of lines ending with punctuation $< 0.12$ are generally lower quality and use this value as a tentative threshold to filter documents. Following this process for all metrics yielded 16 candidate metric-threshold pairs.

We then assessed the effectiveness of these 16 newly created filters by conducting several 28B token ablation runs on the 2019-18 crawl. Full details for these runs are in Appendix E.4. Out of all those runs, we identified three filters (see their ablations runs in Fig. 7) that demonstrated the most significant improvements on the aggregate benchmark score. Specifically, the chosen filters remove documents where the fraction of lines ending with punctuation is $<= 0.12$ (10.14% of tokens removed vs.

30% from the original C4 terminal punctuation filter), where the fraction of characters in duplicated lines is $>= 0.1$ (12.47% of tokens removed; the original MassiveText threshold for this ratio is $>= 0.2$ ), and/or where the fraction of lines shorter than 30 characters is $>= 0.67$ (3.73% of tokens removed). When applying the three together, $\sim 22\%$ of tokens were removed and the aggregate score increased by about 1% in the 28B token ablations. These filters allowed us to further improve performance and, notably, surpass

![](dt=2026-05-30/ht=00/f04de1cce775716303d13aa6115a8dfec8584184f2103be9fe082320f340eddb.jpg)

the C4 dataset performance while filtering out a smaller proportion of data.

# 3.7 The final FineWeb dataset

Combining the decisions made in the previous sections and applying the resulting pipeline to 96 Common Crawl snapshots produces the 15T-token FineWeb dataset. Specifically, we extract text from WARC files (Section 3.2), apply base filtering (Section 3.3), perform individual per-crawl MinHash dedduplication (Section 3.4), apply a selection of C4 filters (Section 3.5), and finally apply custom filters (Section 3.6). Each step provides a relative performance boost on our group of benchmark tasks, as seen in Fig. 9. For the public release of the dataset, we have also applied Personal Identifiable Information (PII) removal, by anonymizing email and public IP addresses.

In Fig. 10 we compare FineWeb with the following commonly used openly accessible web-scale datasets: RefinedWeb (500B tokens) [50], C4 (172B tokens) [15]; the Common crawl-based part of Dolma 1.6 (3T tokens) and 1.7 (1.2T tokens) [58], The Pile (340B tokens) [29], SlimPajama (627B tokens) [35], the deduplicated variant of RedPajama2 $^{1}$ (20T tokens) [34], English CommonCrawl section of Matrix (1.3T tokens) [59], English CC-100 (70B tokens) [60], and Colossal-OSCAR (850B tokens) [61]. Notably, FineWeb shows strong performance and FineWeb-Edu (detailed below) outperforms all other open datasets we compared on our aggregate group of tasks, further validating

1RedPajama2 includes $40+$ quality annotations but is only actually filtered with the CCNet pipeline[28]

7

the design choices we made. Note that to train these models we randomly sampled 350 billion tokens from each dataset, without upsampling any individual Common Crawl snapshot.

![](dt=2026-05-30/ht=00/68683dea677fce809d15261017e3d571240b48ca533ac1f24106e7bd93e03ac2.jpg)

![](dt=2026-05-30/ht=00/e247b80ee41d5aad4ae8c623be4a5e35d45b96d29b8f76aea59bc4360c634d55.jpg)

# 4 FineWeb-Edu

An interesting approach has recently emerged for filtering LLM training datasets: using synthetic data to develop classifiers for identifying educational content. This technique was notably used in the non-public pretraining datasets of Llama 3 [6] and Phi-3 [8], but its large-scale impact on web data filtering has not been publicly explored. We applied this technique to FineWeb by filtering it with an educational quality classifier developed from synthetic annotations generated by Llama-3-70B-Instruct [62]. The resulting dataset, FineWeb-Edu, contains 1.3 trillion tokens.

FineWeb-Edu is specifically optimized for educational content and outperforms all openly accessible web-based datasets on a number of reasoning- and knowledge-intensive benchmarks such as MMLU, ARC, and OpenBookQA by a significant margin.

To build the synthetic annotations, we use Llama-3-70B-Instruct to score 460,000 randomly sampled webpages from the FineWeb CC_MAIN-2024-10 snapshot for their educational quality on a scale from 0 to 5. We explored several prompt formats to automatically extract an educational score using an LLM and found that the additive scale used in previous work Yuan et al. [63] worked best. It allows the LLM to evaluate each criterion and build the score step-by-step, unlike the single-rating scale [64] which assigns a fixed score based on predefined categories. To avoid having the LLM favor highly technical pages like arXiv abstracts and submissions, we prompted it to focus on grade-school and middle-school level knowledge. The prompt used for synthetic annotations is in Appendix F.1.

To scale our filtering to the entirety of FineWeb, we trained a linear regression model on top of the Snowflake-arctic-embed-m embedding model [65]. We fine-tuned this linear regressor on 410,000 of our Llama 3 synthetic annotations for 20 epochs with a learning rate of 3e-4 (while keeping the embedding and encoder layers frozen). We selected the checkpoint with the highest F1 score on the held-out validation set containing the remaining 50,000 samples, treating Llama 3 annotations as ground-truth. After training, we rounded the model's output scores to integers from 0 to 5.

We then used fixed thresholds to classify whether a given document from FineWeb was educational. We investigated the impact of using different thresholds for the filtering and ultimately chose a minimum threshold of 3 for FineWeb-Edu, which ultimately gave the best trade-off between performance on knowledge and reasoning intensive benchmarks and the performance on other benchmarks like HellaSwag [66]. With a threshold of 3, the model achieved an F1 score of $82\%$ on the validation set, indicating strong performance in distinguishing high-quality educational content.

Applying the classifier to the 15 trillion tokens of FineWeb required 6,000 H100 GPU hours.

To confirm the effectiveness of education filtering at a larger scale, we conducted a larger ablation training a 1.71B model on 350 billion tokens, similar to the FineWeb filtering ablations mentioned
8

above. As shown in Fig. 10 and Fig. 11, we observed that FineWeb-Edu surpasses FineWeb and all other open web datasets, with quite remarkable improvements on educational benchmarks such as MMLU, ARC and OpenBookQA. Specifically, MMLU score increases from $33\%$ to $37\%$ , a relative improvement of approximately $12\%$ , and ARC score goes from $46\%$ to $57\%$ , an improvement of about $24\%$ .

On MMLU, FineWeb-Edu can match the final performance of Matrix with almost 10x fewer tokens, demonstrating the effectiveness of classifiers trained on LLM annotations for large-scale data filtering. Additional evaluation plots can be found in Appendix F.2.

# 4.1 Topic distribution

To examine how the educational classifier may skew the dataset towards certain topics, we embed [67] 50k samples from FineWeb and 50k samples from FineWeb-Edu using a sentence-transformers [68] model (all-MiniLM-L6-v2) which we then project to 2D space using UMAP [69]. Finally, we use DBSCAN [70] clustering to find the 100 densest topic clusters in the union of the two datasets, which we label using Llama 3.1 70B [6]. To compare the two datasets, we plot the difference of the size (as a percentage of the entire dataset) of each cluster in FineWeb-Edu and FineWeb in Fig. 18. The educational classifier heavily favors topics such as 'Education,

Learning, Teaching' or 'History, Culture, Politics', while down-sampling 'Business, Finance, Law', 'Entertainment, Film, Theater' and 'Places, Travel, Real Estate', among others.

![](dt=2026-05-30/ht=00/bb13c86545175eb1268bd0715183533297c488355314b43757fe24565dbe2eea.jpg)

# 4.2 Domain fit

We evaluate the macro average perplexity of six checkpoints from our FineWeb and FineWeb-Edu ablation models on the domains from Paloma [71]. We use the codebase provided in [71] but intentionally do not perform decontamination, to compare how well each dataset covers different domains. The results are in Fig. 12. The FineWeb model generally shows lower perplexity in broad web sources such as C4, mC4, Falcon, Dolma V1.5 or the CommonCrawl subset of RedPajama, as well as on Twitter AAE, Manosphere, Gab, reddit (100 Subreddits) or 4chan.

FineWeb-Edu tends to favour sources containing Wikipedia (WikiText-103 and M2D2 Wikipedia) or that are heavy in academic content (M2D2 S2ORC, which has semantic scholar papers, and the Arxiv subset of RedPajama). FineWeb-Edu also seems to have better coverage of programming content (100 PLs) than FineWeb. We have included results per subset for some of these sources in Appendix F.4.

# 5 Bias analyses

Language models are known to reflect the biases present in their pretraining datasets [72-77]. To provide a brief picture of dataset bias in FineWeb and FineWeb-Edu, we focus on subgroups recognised as "sensitive" or "protected" in English-speaking countries. These are a subset of subgroups that are historically subject to discrimination and are disproportionately the target of negative societal norms such as stereotyping, which is reflected in text-based language consumed for a dataset.

We find that the FineWeb dataset has a relative overrepresentation of words that reflect hegemonic norms, known to be overrepresented in online text [72], such as 'man' and 'christian'. Although biases across the gender, religion, and age subgroups examined are not strong, we see the most skewed association between religion words and intimacy, such as 'christian dating' and 'jewish singles'.

Fittingly, the FineWeb-Edu dataset captures associations that are less tied to intimacy compared to FineWeb and more expected from educational content of history and health, such as 'man' being associated to 'king', and 'woman' associated to 'pregnancy'. Further details are provided in Appendix G.

9

![](dt=2026-05-30/ht=00/bbf1e184231230bf4800460523600f496c35ed75bde68e36fe9bec1355e7c0cd.jpg)

![](dt=2026-05-30/ht=00/28eff9ac2cf8a70513a5e94901b7017c0702f291d81ff8268b1d18a163459798.jpg)

![](dt=2026-05-30/ht=00/cef8ce7361f4faebca36eeada022ff7d11e9660cd172bee2d402b70973b9ff5e.jpg)

![](dt=2026-05-30/ht=00/4a6e5abc5c44c7fb7f378d93044b78e6e406266c84b2acda2d157cc55c76ec8e.jpg)

![](dt=2026-05-30/ht=00/310013bbafbaa97b90466062250a1ff343c41ea5b09d492ebe6ad29d47868750.jpg)

![](dt=2026-05-30/ht=00/150c1933dac7c4db6a63b8caa96460cb5bccc4b73c27964128b220fdd99c9486.jpg)

![](dt=2026-05-30/ht=00/10fb5b62cf922b72af1fc4f263aef979ce85647ab03036492a6145e56f5111b2.jpg)

![](dt=2026-05-30/ht=00/9384eca367c805a4703aae07a900760e83d875f4ae1693124717ea60e2027c19.jpg)

![](dt=2026-05-30/ht=00/d906c253cb1c559344d36dd173400a4c72644f9180919a09fb23da239e19a4cb.jpg)

![](dt=2026-05-30/ht=00/4cc9aa7d0f1f64b44aeb9f651709ce74c2b80f76227469911c670f8ca7c62e23.jpg)

![](dt=2026-05-30/ht=00/7cffb0cad3942ea6bfe17b05b0001b1ee8576babfb0d59045d0157f3962379ed.jpg)

![](dt=2026-05-30/ht=00/5991aeff2fd551c941fb5ca56bcd3c91e357e58007c03eb8dc139a78426ed8e4.jpg)

![](dt=2026-05-30/ht=00/6ec40bbdb960693543e1cbf47599ce44e8db73fb4d6c5cb09ae81f43c58e6861.jpg)

![](dt=2026-05-30/ht=00/e9e981fb1ca20c5027ec2c05e5a9ecdd10712b48d12d36a7aad6d3d8e50a9fd6.jpg)

![](dt=2026-05-30/ht=00/1389e5d6889dc99d59af1e2a61f529a889c85753c3f219bd0715c68d1abd416f.jpg)

![](dt=2026-05-30/ht=00/e63113455c65f751b784fc3d99703b9d14d0c1cf7f7f5465815f0e103a0f9fc3.jpg)

![](dt=2026-05-30/ht=00/871008e99b9cee7a19b562a6d52bbe5aaa62626c6c33eb8a6465a8fa4d565ca7.jpg)

# 6 Conclusion

In this paper, we developed the FineWeb datasets, a collection of large-scale LLM pretraining datasets that produce performant LLMs. Specifically, we release FineWeb, a 15-trillion token dataset derived from 96 Common Crawl snapshots, as well as FineWeb-Edu, a 1.3-trillion token dataset of educational content from FineWeb. FineWeb was created through a series of experiments that provided empirical evidence for our choice of text extraction strategy, dedduplication procedure, and content filters. Both datasets are publicly released, along with the code and processing library that we used and all of the models we trained during our dataset ablation experiments.

While FineWeb and FineWeb-Edu attain state-of-the-art performance among public LLM pretraining datasets, we identify various paths to further improvement. First, both datasets are entirely comprised of web content scraped by Common Crawl. It is possible that augmenting either datasets with other datatypes (books, speech transcripts, etc.) could further improve performance. In addition, most of the experiments we ran were at a smaller scale due to
computational constraints. Designing datasets at more realistic scales could provide more reliable guidance.

Our evaluation setup was also by necessity limited to performance on academic benchmarks without any further instruction tuning or alignment. An evaluation setup that better reflected current usage patterns of LLMs might also be more reliable. We hope that our released datasets, code, and models help further improve public knowledge and development of performant LLM pretraining datasets.

10

# References

11

12

13

14

15

16

# A FineWeb Datasheet

![](dt=2026-05-30/ht=00/0297bec4c97f396cba01f856e8e4b80a63c02e8376ea789eb550685ee6111018.jpg)

<table><tr><td colspan="2">Dataset Details</td></tr><tr><td>Purpose of the dataset</td><td>We released FineWeb to make large language model training more accessible to the machine learning community at large.</td></tr><tr><td>Curated by</td><td>The dataset was curated by Hugging Face.</td></tr><tr><td>Funded by</td><td>The dataset was funded by Hugging Face.</td></tr><tr><td>Language(s)</td><td>English</td></tr><tr><td>License</td><td>The dataset is released under the Open Data Commons Attribution License (ODC-By) v1.0 license. The use of this dataset is also subject to Common-Crawl's Terms of Use.</td></tr><tr><td colspan="2">Dataset Structure</td></tr><tr><td>Data Instances</td><td>The following is an example sample from the dataset. It is part of the CC-MAIN-2021-43 snapshot and was crawled on 2021-10-15T21:20:12Z: 
{ "text": "This is basically a peanut flavoured cream thickened with egg yolks and then set into a ramekin on top of some jam. Tony, one of the Wedgwood chefs, suggested sprinkling on some toasted crushed peanuts at the end to create extra crunch, which I thought was a great idea. The result is excellent.", "id": "urn:uuid:e5a3e79a-13d4-4147-a26e-167536fcac5d&gt;", "dump": "CC-MAIN-2021-43", "url": "/http://allrecipes.co.uk /recipe/24758/peanut-butter-and -jam-creme-brulee.aspx ?o_is=SimilarRecipes&amp;to_ln=Sim Recipes_Photo_7&gt;", "date": "2021-10-15T21:20:12Z", "file_path": "s3://commoncrawl/crawl-data/ CC-MAIN-2021-43/segments/ 1634323583083.92/warc/ CC-MAIN-20211015192439 -20211015222439-00600.warc.gz", "language": "en", "language_score": 0.948729, "token_count": 69 }</td></tr><tr><td>Data Fields</td><td>- text (string): the main text content
- id (string): original unique identifier for this sample from CommonCrawl
- dump (string): the CommonCrawl dump/snapshot this sample was a part of
- url (string): url to the original page where text was present
- date (string): crawl date (from Common-Crawl)
- file_path (string): s3 path for the individual CommonCrawl arc file containing this sample
- language (string): en for all the samples in this dataset
- language_score (float): language prediction score (0.01.0) as reported by the fastText language classifier
- token_count (int): number of tokens when applying the gpt2 tokenizer to this sample</td></tr><tr><td>Data Splits</td><td>The default subset includes the entire dataset. We also include separate splits for each CommonCrawl dump. FineWeb-Edu, a subset filtered for educational content, is also available.</td></tr><tr><td colspan="2">Dataset Creation</td></tr><tr><td>Curation Rationale</td><td>With FineWeb, we aim to provide the open source community with a clean and large-scale dataset for pretraining performant large language models.</td></tr><tr><td>Source Data</td><td>The source data consists of webpages crawled by the CommonCrawl foundation over the 2013-2024 time period. We then extracted the main page text from the HTML of each webpage, filtered each sample and deduplicated each individual Common-Crawl dump/crawl.</td></tr><tr><td>Data processing steps</td><td>The data processing pipeline consists of:
- URL filtering
- Trafilatura text extraction
- FastText language filter
- MassiveText repetition and quality → filters
- C4 quality filters
- FineWeb custom filters
- MinHash dedduplication
- PII reformatting
For FineWeb-Edu, we further apply a filtering step based on our educational content classifier.</td></tr><tr><td>Annotations</td><td>We augment the original samples with the language, language_score and tokens_count annotations. The language related annotations are automatically generated by our language filter. token_count is generated by applying the GPT-2 tokenizer to the text column.</td></tr><tr><td>Personal and Sensitive Information</td><td>We anonymize email addresses and public IP addresses using regex patterns.</td></tr><tr><td colspan="2">Considerations for Using the Data</td></tr><tr><td>Social Impact of Dataset</td><td>With the release of FineWeb, we aim to make LLM training more accessible to the machine learning community by:
(a) making the dataset creation process more transparent, by sharing our entire processing setup including the codebase used
(b) helping alleviate the costs of dataset curation, both in time and in compute, for model creators by publicly releasing our dataset with the community.</td></tr><tr><td>Biases</td><td>Efforts were made to minimize the amount of NSFW and toxic content present in the dataset by employing filtering on the URL level. However, there are still a significant number of documents present in the final dataset that could be considered to be toxic or contain harmful content. As FineWeb was sourced from the web as a whole, any harmful biases typically present in the web may be reproduced on our dataset. Bias analyses for sensitive subgroups demonstrate that ‘man’ is more common in the dataset than other gender terms, ‘christian’ is more common than other religion terms. The disproportionate association of specific terms to sensitive subgroups is relatively low, with the most notable bias that some religion terms tend to be more associated with online dating terms. We provide a more detailed bias analysis in Section 5.</td></tr><tr><td>Other Known Limitations</td><td>As a consequence of some of the filtering steps applied, it is likely that code content is not prevalent in our dataset. Users are advised to consider complementing FineWeb with other code datasets and specialized curated sources, such as Wikipedia, which may have better formatting than the Wikipedia content included in FineWeb.</td></tr></table>

17

![](image)

18

![](image)

# B License and hosting

The FineWeb datasets are released under the Open Data Commons Attribution License (ODC-By) v1.0. The full text of the license is available at https://opendatacommons.org/licenses/by/1-0/. The use of the dataset is also subject to CommonCrawl's Terms of Use.

The FineWeb datasets are hosted on the HuggingFace hub, where they will remain available for the foreseeable future. We plan to regularly update the dataset with new CommonCrawl snapshots as they are released.

19

# C Linked resources

![](dt=2026-05-30/ht=00/489974d504e7bb18d88f4598f826bd60f40090d7f46f87f4dafe515e28e337a6.jpg)

<table><tr><td>Resource</td><td>URL</td></tr><tr><td>FineWeb repository (DOI 10.57967/hf/2493)</td><td>https://hf.co/datasets/HuggingFaceFW/fineweb</td></tr><tr><td>FineWeb Croissant metadata</td><td>https://hf.co/api/datasets/HuggingFaceFW/fineweb/croissant</td></tr><tr><td>FineWeb-Edu repository (DOI 10.57967/hf/2497)</td><td>https://hf.co/datasets/HuggingFaceFW/fineweb-edu</td></tr><tr><td>FineWeb-Edu Croissant metadata</td><td>https://hf.co/api/datasets/HuggingFaceFW/fineweb-edu/croissant</td></tr><tr><td>FineWeb Llama3 annotations</td><td>https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations</td></tr><tr><td>Educational classifier</td><td>https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier</td></tr><tr><td>Dataset comparison models</td><td>https://hf.co/collections/HuggingFaceFW/comparison-models-662457b0d213e8c14fe47f32</td></tr><tr><td>Ablation
models</td><td>https://hf.co/collections/HuggingFaceFW/data-experiments-665ed849020d8b66a5d9896f</td></tr><tr><td>Datatrove processing code to reproduce FineWeb</td><td>https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py</td></tr><tr><td>Evaluation setup</td><td>https://hf.co/datasets/HuggingFaceFW/fineweb/blob/main/lighteval_tasks.py</td></tr></table>

# D Data ablation setup

# D.1 Model architecture

![](dt=2026-05-30/ht=00/6e38dcb2186e23965ff7a1304542a5f36ecdb058784604a30d04840ca0b3aa5e.jpg)

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Architecture</td><td>Llama</td></tr><tr><td>Number of attention heads</td><td>32</td></tr><tr><td>Number of hidden layers</td><td>24</td></tr><tr><td>Number of key-value heads</td><td>32</td></tr><tr><td>RMS Norm epsilon</td><td>1e-05</td></tr><tr><td>Tied word embeddings</td><td>True</td></tr><tr><td>Embedding size</td><td>50257</td></tr><tr><td>Total number of parameters</td><td>1.71B</td></tr><tr><td>Random initialization std</td><td>0.02</td></tr><tr><td>Tokenizer</td><td>GPT2</td></tr></table>

20

# D.2 Distributed training setup

![](dt=2026-05-30/ht=00/1340d007983d6e150135676c55233a593bce92bd3b4df0d178383f8282cb4d06.jpg)

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Data parallelism (dp)</td><td>64</td></tr><tr><td>Tensor parallelism (tp)</td><td>1</td></tr><tr><td>Pipeline parallelism (pp)</td><td>1</td></tr><tr><td>Micro-batch size</td><td>4</td></tr><tr><td>Sequence length</td><td>2048</td></tr><tr><td>Batch accumulation per replica</td><td>4</td></tr></table>

# D.3 Optimizer Configuration

![](dt=2026-05-30/ht=00/b31182b731b6eab4e152ed327d75e97e0eafd7bba242c72faac681bcbca6978d.jpg)

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Adam beta1</td><td>0.9</td></tr><tr><td>Adam beta2</td><td>0.95</td></tr><tr><td>Adam epsilon</td><td>1.0e-8</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Learning rate</td><td>3e-4</td></tr><tr><td>Warmup steps</td><td>500</td></tr><tr><td>Warmup style</td><td>linear</td></tr><tr><td>Decay style</td><td>cosine</td></tr><tr><td>Minimum decay LR</td><td>3.0e-5</td></tr></table>

# E Dedduplication

# E.1 Dedduplication parameters

As mentioned in Section 3.4, we use 5-grams and 112 hash functions for our MinHash dedduplication. Each 5-gram is hashed with each of the 112 hash functions, and a document signature is obtained by taking the minimum hash value (minhash) across all 5-grams for each hash function. We further split the resulting 112 minhashes into 14 buckets of 8 hashes each. Documents are matched if they have the same 8 minhashes in at least one of the 14 buckets.

With these parameters, the probability that two documents with a n-gram similarity $(s)$ of 0.7, 0.75, 0.8 and 0.85 would be identified as duplicates would be $56\%$ , $77\%$ , $92\%$ and $98.8\%$ , respectively. This split therefore will match documents that are at least $75\%$ similar with a high probability, and almost guarantee that documents with similarities of $85\%$ or above will be matched.

These values can be computed by taking the following probabilities: that the two documents would have the same value for a given hash function, $s$ ; that they do not have the same 8 minhashes in one bucket, $1 - s^8$ ; that they do not have the same 8 minhashes in any of the 14 buckets, $(1 - s^8)^{14}$ ; and finally that they have the same 8 minhashes on at least one of the 14 buckets, $1 - (1 - s^8)^{14}$ .

See Fig. 13 for a match probability comparison between our setup with 112 hashes and the one from RefinedWeb, with 9000 hashes, divided into 450 buckets of 20 hashes.

While the high number of hash functions in RefinedWeb allows for a steeper, more well-defined cut off (document pairs with similarity near the threshold are more likely to be correctly identified), this

21

![](dt=2026-05-30/ht=00/1f6bd432dafb7c19fa3c52c87def4f0ba82c257e6d1f8a14ad37b33f06d92422.jpg)

larger number of hash functions also requires a substantially larger amount of compute resources, as each individual hash must be computed, stored, and then compared with hashes from other documents. We believe the compute and storage savings make up for the higher uncertainty on documents near the threshold.

# E.2 Measuring the effect of dedduplication

Given the nature of dedduplication, its effect is not always visible in a smaller slice of the dataset (such as 28B tokens, the size used for our filtering ablations). Furthermore, there are specific effects at play when deduplicating across different Common Crawl dumps, as some URLs and webpages are recrawled from one snapshot to the next.

![](dt=2026-05-30/ht=00/f3a0f456d3d562b06946d919c9b6d98bbab7213ac1372d4d1243914f52145ad9.jpg)

22

To visualize the effect of scaling the number of training tokens when measuring dedduplication impact, we simulated creating different-sized subsets of randomly sampled documents from the full dataset under the following extreme conditions: there are 100 snapshots, where each one is made up of unique documents with a total of 200 billion tokens (yielding our total of 20 trillion from Section 3.4), and each snapshot is an exact copy of each other (worst case scenario for inter snapshot duplication).

In Fig. 14, we can see that for a 1 billion subset, almost all documents would be unique (#Duplicates=1), despite each document being repeated 100 times in the full dataset. At the 100 billion scale (0.5% of the total dataset), there starts to be a larger number of documents being repeated twice, and a few even 4-8 times. At the larger scale of 1 trillion (5% of the total dataset), the majority of the documents are repeated up to 8 times, with some being repeated up to 16 times.

This simulation illustrates the inherent difficulties with measuring dedduplication impact on the training of larger LLMs once the largest duplicate clusters have been removed. We ran our performance evaluations for deduplicated data at the 350 billion scale, which would, under this theoretical scenario, be made up of a significant portion of documents duplicated up to 8 times.

![](dt=2026-05-30/ht=00/58ff1966434a21e5eb3b9e6eeab7eac0bc7c35a43eb1f011c6726abc4923f2f9.jpg)

To attempt to improve performance on top of independently deduplicating each snapshot, we experimented with applying other "lighter" global dedduplication methods to all the individually MinHash deduplicated snapshots (comprising 20 trillion tokens of data).

We explored URL dedduplication, where we only kept one document per normalized (lowercased) URL (71.5% of tokens removed, 5.6 trillion left) — FineWeb URL dedup. Different line-based dedduplication variations were also considered: remove all but 1 (randomly chosen) occurrence of each duplicated line (77.8% of tokens dropped, 4.4 trillion left) — FineWeb line dedup; same as above, but only removing duplicate lines with at least 10 words and dropping documents with fewer than 3 sentences after dedduplication (85% of tokens dropped, 2.9 trillion left) — FineWeb line dedup w/ min words; and remove all but 1 occurrence of each span of 3 duplicated lines with each number treated as 0 when finding duplicates, (80.9% of tokens removed, 3.7 trillion left) — FineWeb 3-line dedup.

As can be seen in Fig. 15 the performance of the models trained on each of these methods was consistently worse (albeit to different degrees) than that of the original individually deduplicated data. We t
herefore did not apply any additional dedduplication beyond individual-snapshot MinHash-based dedduplication.

23

# E.4 Other filters considered

Table 2: Full list of heuristic filters tested

![](dt=2026-05-30/ht=00/9e5de62b28e7b42b3df7702cdd45acae712747ab9903fbb42c848b06a440c3a8.jpg)

<table><tr><td>Metric</td><td>Threshold</td><td>Aggregate Acc (%)</td><td>Tokens (%)</td><td>removed</td></tr><tr><td>lines-with-punct-ratio</td><td>≥ 0.12</td><td>42.85</td><td>10.14</td><td></td></tr><tr><td>duplicated-line-char-ratio</td><td>≤ 0.01</td><td>42.78</td><td>12.47</td><td></td></tr><tr><td>lines-with-punct-ratio</td><td>≥ 0.12 or = 0</td><td>42.72</td><td>5.82</td><td></td></tr><tr><td>lines-shorter-30-ratio</td><td>≤ 0.67</td><td>42.65</td><td>3.37</td><td></td></tr><tr><td>line-with-most-3-words-ratio</td><td>≤ 0.49</td><td>42.61</td><td>2.51</td><td></td></tr><tr><td>duplicate-(5-10)-grams-char-ratio</td><td>≤ 0.1, 0.084, 0.073, 0.065, 0.057, 0.05</td><td>42.60</td><td>10.92</td><td></td></tr><tr><td>lines-with-punct-ratio</td><td>≥ 0.08 or = 0</td><td>42.59</td><td>3.42</td><td></td></tr><tr><td>top-(2,3,4)-gram-char-ratio</td><td>≤ 0.13, 0.087, 0.079</td><td>42.58</td><td>56.71</td><td></td></tr><tr><td>lines-shorter-30-ratio</td><td>0.69</td><td>42.58</td><td>3.73</td><td></td></tr><tr><td>avg-words-per-line</td><td>≥ 7</td><td>42.56</td><td>2.32</td><td></td></tr><tr><td>lines-shorter-30-ratio</td><td>≤ 0.5</td><td>42.53</td><td>11.17</td><td></td></tr><tr><td>avg-words-per-line</td><td>≥ 5</td><td>42.39</td><td>0.83</td><td></td></tr><tr><td>avg-words-per-line</td><td>≥ 9</td><td>42.27</td><td>4.47</td><td></td></tr><tr><td>avg-line-length-0.5-sampling</td><td>≥ 56</td><td>42.93</td><td>3.24</td><td></td></tr><tr><td>avg-line-length</td><td>≥ 56</td><td>42.12</td><td>6.48</td><td></td></tr><tr><td>avg-line-length-0.5-sampling</td><td>≥ 40</td><td>42.03</td><td>1.50</td><td></td></tr></table>

24

# F FineWeb-Edu

# F.1 Annotation Prompt

We use the following prompt template to generate document annotations using the Llama3 model:

Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

The extract: <EXAMPLE>.

After examining the extract:

# F.2 Additional results

Fig. 16 compares FineWeb-Edu to other open web datasets on 9 becnhmarks, using a 1.71B model trained on 350 billion tokens. Additionally, Fig. 17 displays the results of experiments with various filtering thresholds for building FineWeb-Edu, using a 1.71B model trained on 28 billion tokens. Our findings indicate that a threshold of 3 yields the best average performance.

25

![](dt=2026-05-30/ht=00/11179acbc1e886748d11f621d47910d9f1db2c953a1cbd1b957a26f800d91139.jpg)

![](dt=2026-05-30/ht=00/431ce98c485b187cf05a959d89b24a2f1819637087dd1b9d71e1cc5ddcc2c847.jpg)

![](dt=2026-05-30/ht=00/8e9539b31c0e6f96a37e475d95e4ef3859c6ac464bc020afe928bac3778b333b.jpg)

![](dt=2026-05-30/ht=00/a0a74519cb637f16134fff70153aa99c2385ce44b7d33e1fc9c68534ac9752b0.jpg)

![](dt=2026-05-30/ht=00/9655e65e18e3d25c0f8ee128cfc32b9200b2413a0206aa29209e1c97c2385e73.jpg)

![](dt=2026-05-30/ht=00/3c8b686e2afbe29ad8c861ff972f85835878491e868b4294127e8f83da0e2613.jpg)

![](dt=2026-05-30/ht=00/14cf0c5b9904ab81474ddf42595b69973fca795fcea138b0bd107af433179ad4.jpg)

![](dt=2026-05-30/ht=00/5feef7897f7815671449808e0701f48840bc505624e2c1c9d0388484727b9d8c.jpg)

![](dt=2026-05-30/ht=00/fe332ccecc842ad9776389dded3c7c3873160f3e364956d0e53fafee733d5639.jpg)

26

![](dt=2026-05-30/ht=00/96e4d5039e16a0e7fd07811a34f1667ba379eab1a7685b2028f10118e9eca1b7.jpg)

27

![](dt=2026-05-30/ht=00/8c0971a743cc3203dc6bb0a1dc1cda804a648b41f940881b0e1c31d4fe8e8b9e.jpg)

28

F.4 Domain fit

![](dt=2026-05-30/ht=00/76af6fa9a7058a3557fbb7944ee37e9000fe8dd85473250f93b79072c054e9b4.jpg)

<table><tr><td>Source</td><td>Domain</td><td>FineWeb ppl</td><td>FineWeb-Edu ppl</td></tr><tr><td>Dolma V1.5</td><td>common-crawl</td><td>14.499</td><td>18.336</td></tr><tr><td>Dolma V1.5</td><td>pes2o</td><td>12.226</td><td>10.242</td></tr><tr><td>Dolma V1.5</td><td>redgit uniform</td><td>23.814</td><td>29.864</td></tr><tr><td>Dolma V1.5</td><td>stack uniform</td><td>7.65</td><td>7.014</td></tr><tr><td>Dolma V1.5</td><td>wiki</td><td>12.0</td><td>12.243</td></tr><tr><td>M2D2 Wikipedia</td><td>Culture and the arts</td><td>10.367</td><td>14.518</td></tr><tr><td>M2D2 Wikipedia</td><td>Culture and the arts Culture and Humanities</td><td>14.037</td><td>14.116</td></tr><tr><td>M2D2 Wikipedia</td><td>Culture and the arts Games and Toys</td><td>15.774</td><td>18.912</td></tr><tr><td>M2D2 Wikipedia</td><td>Culture and the arts Mass media</td><td>14.352</td><td>18.134</td></tr><tr><td>M2D2 Wikipedia</td><td>Culture and the arts Performing arts</td><td>14.311</td><td>13.313</td></tr><tr><td>M2D2 Wikipedia</td><td>Culture and the arts Sports and Recreation</td><td>11.295</td><td>14.735</td></tr><tr><td>M2D2 Wikipedia</td><td>Culture and the arts The arts and Entertainment</td><td>13.669</td><td>19.039</td></tr><tr><td>M2D2 Wikipedia</td><td>Culture and the arts Visual arts</td><td>14.967</td><td>15.158</td></tr><tr><td>M2D2 Wikipedia</td><td>General reference</td><td>11.962</td><td>11.246</td></tr><tr><td>M2D2 Wikipedia</td><td>General reference Further research tools and topics</td><td>16.202</td><td>19.191</td></tr><tr><td>M2D2 Wikipedia</td><td>General reference Reference works</td><td>14.914</td><td>18.621</td></tr><tr><td>M2D2 Wikipedia</td><td>Health and fitness</td><td>12.0</td><td>13.448</td></tr><tr><td>M2D2 Wikipedia</td><td>Health and fitness Exercise</td><td>11.874</td><td>13.951</td></tr><tr><td>M2D2 Wikipedia</td><td>Health and fitness Health science</td><td>11.509</td><td>10.997</td></tr><tr><td>M2D2 Wikipedia</td><td>Health and fitness Human medicine</td><td>12.0</td><td>13.448</td></tr><tr><td>M2D2 Wikipedia</td><td>Health and fitness Nutrition</td><td>10.09</td><td>8.489</td></tr><tr><td>M2D2 Wikipedia</td><td>Health and fitness Public health</td><td>12.804</td><td>11.797</td></tr><tr><td>M2D2 Wikipedia</td><td>Health and fitness Self care</td><td>14.62</td><td>12.782</td></tr><tr><td>M2D2 Wikipedia</td><td>History and events</td><td>13.446</td><td>12.516</td></tr><tr><td>M2D2 Wikipedia</td><td>History and events By continent</td><td>14.174</td><td>12.066</td></tr><tr><td>M2D2 Wikipedia</td><td>History and events By period</td><td>12.94</td><td>11.0</td></tr><tr><td>M2D2 Wikipedia</td><td>History and events By region</td><td>13.61</td><td>11.63</td></tr><tr><td>M2D2 Wikipedia</td><td>Human activites</td><td>15.159</td><td>18.728</td></tr><tr><td>M2D2 Wikipedia</td><td>Human activites Human activities</td><td>12.784</td><td>11.117</td></tr><tr><td>M2D2
Wikipedia</td><td>Human activites Impact of human activity</td><td>15.092</td><td>13.592</td></tr><tr><td>M2D2 Wikipedia</td><td>Mathematics and logic</td><td>12.703</td><td>9.903</td></tr><tr><td>M2D2 Wikipedia</td><td>Mathematics and logic Fields of mathematics</td><td>12.703</td><td>9.903</td></tr><tr><td>M2D2 Wikipedia</td><td>Mathematics and logic Logic</td><td>14.281</td><td>13.367</td></tr><tr><td>M2D2 Wikipedia</td><td>Mathematics and logic Mathematics</td><td>14.923</td><td>14.207</td></tr><tr><td>M2D2 Wikipedia</td><td>Natural and physical sciences</td><td>12.884</td><td>10.529</td></tr><tr><td>M2D2 Wikipedia</td><td>Natural and physical sciences Biology</td><td>12.718</td><td>10.221</td></tr><tr><td>M2D2 Wikipedia</td><td>Natural and physical sciences Earth sciences</td><td>15.346</td><td>13.145</td></tr></table>

29

Table 3: Paloma domain comparison between FineWeb and FineWeb-Edu. Lower perplexity (ppl) in bold. A lower perplexity value indicates a better fit to a given domain.

![](dt=2026-05-30/ht=00/c1d316fee403e73be6221f537944240ad5f3195ae252d0143590ddb1795719cb.jpg)

<table><tr><td>Source</td><td>Domain</td><td>FineWeb ppl</td><td>FineWeb-Edu ppl</td></tr><tr><td>M2D2 Wikipedia</td><td>Natural and physical sciences Nature</td><td>12.594</td><td>9.886</td></tr><tr><td>M2D2 Wikipedia</td><td>Natural and physical sciences Physical sciences</td><td>13.088</td><td>10.643</td></tr><tr><td>M2D2 Wikipedia</td><td>Philosophy and thinking</td><td>14.081</td><td>16.067</td></tr><tr><td>M2D2 Wikipedia</td><td>Philosophy and thinking Philosophy</td><td>14.209</td><td>12.91</td></tr><tr><td>M2D2 Wikipedia</td><td>Philosophy and thinking Thinking</td><td>14.081</td><td>16.067</td></tr><tr><td>M2D2 Wikipedia</td><td>Religion and belief systems</td><td>12.636</td><td>11.326</td></tr><tr><td>M2D2 Wikipedia</td><td>Religion and belief systems Allah</td><td>14.072</td><td>10.808</td></tr><tr><td>M2D2 Wikipedia</td><td>Religion and belief systems Belief systems</td><td>12.843</td><td>11.652</td></tr><tr><td>M2D2 Wikipedia</td><td>Religion and belief systems Major beliefs of the world</td><td>13.824</td><td>11.834</td></tr><tr><td>M2D2 Wikipedia</td><td>Society and social sciences</td><td>11.777</td><td>11.195</td></tr><tr><td>M2D2 Wikipedia</td><td>Society and social sciences Social sciences</td><td>11.81</td><td>13.03</td></tr><tr><td>M2D2 Wikipedia</td><td>Society and social sciences Society</td><td>11.777</td><td>11.195</td></tr><tr><td>M2D2 Wikipedia</td><td>Technology and applied sciences</td><td>11.592</td><td>9.368</td></tr><tr><td>M2D2 Wikipedia</td><td>Technology and applied sciences Agriculture</td><td>13.941</td><td>14.998</td></tr><tr><td>M2D2 Wikipedia</td><td>Technology and applied sciences Computing</td><td>15.562</td><td>16.091</td></tr><tr><td>M2D2 Wikipedia</td><td>Technology and applied sciences Engineering</td><td>14.897</td><td>13.861</td></tr><tr><td>M2D2 Wikipedia</td><td>Technology and applied sciences Transport</td><td>16.519</td><td>17.886</td></tr><tr><td>Manosphere</td><td>avfm</td><td>27.332</td><td>32.058</td></tr><tr><td>Manosphere</td><td>incels</td><td>18.253</td><td>20.788</td></tr><tr><td>Manosphere</td><td>love shy</td><td>28.206</td><td>33.374</td></tr><tr><td>Manosphere</td><td>mgtow</td><td>24.913</td><td>29.702</td></tr><tr><td>Manosphere</td><td>pua forum</td><td>25.133</td><td>33.297</td></tr><tr><td>Manosphere</td><td>red pill talk</td><td>33.87</td><td>42.947</td></tr><tr><td>Manosphere</td><td>reddit</td><td>24.786</td><td>30.903</td></tr><tr><td>Manosphere</td><td>rooshv</td><td>23.593</td><td>27.819</td></tr><tr><td>Manosphere</td><td>the attraction</td><td>24.988</td><td>30.907</td></tr><tr><td>RedPajama</td><td>arxiv</td><td>32.338</td><td>23.368</td></tr><tr><td>RedPajama</td><td>books</td><td>22.095</td><td>23.953</td></tr><tr><td>RedPajama</td><td>c4</td><td>12.685</td><td>15.599</td></tr><tr><td>RedPajama</td><td>commoncrawl</td><td>8.0</td><td>8.979</td></tr><tr><td>RedPajama</td><td>github</td><td>5.613</td><td>5.247</td></tr><tr><td>RedPajama</td><td>stackexchange</td><td>9.055</td><td>8.862</td></tr><tr><td>RedPajama</td><td>wikipedia</td><td>8.741</td><td>8.608</td></tr><tr><td>Twitter AAE</td><td>AA</td><td>246.907</td><td>575.106</td></tr><tr><td>Twitter AAE</td><td>white</td><td>98.536</td><td>192.374</td></tr></table>

30

# G Bias Analyses

# G.1 Distributional Analysis

Subgroup

Terms

age 'old', 'young'

gender 'man', 'woman', 'non-binary'

religion 'muslim', 'christian', 'jewish', 'hindu', 'buddhist', 'atheist'

Table 4: Subgroups and terms used for bias analyses.

![](dt=2026-05-30/ht=00/5fcde0893e29362ef90b4d38e8a9e6e5e6617c429a225b9a08903952b050f55f.jpg)

![](dt=2026-05-30/ht=00/461d74c5ee03780a2566466e775a354b1ba0a6f30129b5464e5978aa9855bc6b.jpg)

![](dt=2026-05-30/ht=00/4b963a46384ef12683e58aaf8405b96932939a8b9417ff8d8b1b920ab8141403.jpg)

![](dt=2026-05-30/ht=00/ee7e17f673934c9f5a06325ccfa59fb70164587608bb83be2014181b003b7c59.jpg)

To begin, we examine the distribution over subgroup terms for gender (Fig. 19) age (Fig. 20), and religion (Fig. 21) in a subset of FineWeb and FineWeb-Edu randomly sampled from the whole dataset, of around 10 Billion GPT-2 tokens (FineWeb 10BT and FineWeb-Edu 10BT). Terms used are shown in Table 4 and are all normalized to lowercase for this analysis.

We find that 'man' appears much more frequently than 'woman' and 'non-binary', and 'christian' appears much more frequently than all other religions terms tested.

31

![](dt=2026-05-30/ht=00/832c95f94468155f3bd4a7f8987d01dd6f69269ba6eee46564021b7614b069f5.jpg)

![](dt=2026-05-30/ht=00/145d16a763f11be598175237113bf00d3cfe6d178b015a485511352df25d3ae5.jpg)

# G.2 Association Analysis

We next examine the skews with respect to the different subgroup terms, as measured by TF-IDF [78]. This method is described as capturing the specificity of words in the dataset, here applied as specificity with respect to the terms for the different subgroups. This provides a way to quantify how "biased" each subgroup term is with respect to the words they co-occur with. Specifically, given the dataset and terms for a subgroup of interest, we:

# G.2.1 Gender

We find that 'man' is associated with terms such as 'god', 'police', 'said' and 'good', 'woman' is associated with terms like 'said', 'women', 'police', 'life', 'love', 'dating' and 'family', and 'non-binary' is associated with 'gender' and LGBTQIA+ terms such as 'trans', 'transgender', and 'queer' (Fig. 22). Applying this same analysis to FineWeb-Edu-Sample-10BT, we find that 'man' is associated with the term 'god', and slightly associated with terms like 'war', 'great', and 'king'. 'woman' is associated with terms like 'pregnancy', 'cancer', 'mother', 'children', and 'family'.

# G.2.2 Religion

Throughout, we see skews towards words associated with online intimacy: 'online', 'singles', 'sex', 'mature', 'girls'. As can be seen in Fig. 27, 'jewish' is particularly associated with 'dating' and 'singles'. ' muslim', 'jewish', 'hindu' and 'buddhist' are slightly skewed to co-occur with 'women', while 'sex' is skewed with ' muslim', 'christian', 'jewish'; and 'girl' with ' muslim', 'jewish', 'hindu'.

# G.2.3 Age

The word 'young' is skewed to co-occur with 'women', consistent with the problematic tendencies in English-speaking societies to infantilize women and over-indexing on women's youth [79, 80]. We a
lso see expected skews, such as 'young' co-occurring with words like 'children' and 'school'.

32

![](dt=2026-05-30/ht=00/5076655b1b1b0bbc172759933e55aa5ea052376806b5b24d685404cf09f14f0c.jpg)

![](dt=2026-05-30/ht=00/cf71ed6f6cdda569ce4f3460b8b56ab182076bdf2e02934c46d54cd5c7f5d9c3.jpg)

![](dt=2026-05-30/ht=00/db00a33b861063148c313ee3eb0ef6bea027813c5d866aec1545ffb882e3ad88.jpg)

33

![](dt=2026-05-30/ht=00/7b4225a7299e84c60348197027c9112b3db160a9677123044175ddc8fbdbd5d3.jpg)

![](dt=2026-05-30/ht=00/56dafa50897d5704344a0501b8df5ea6fc128bd2222918e98f6fa41da46a3df3.jpg)

34

![](dt=2026-05-30/ht=00/e18d24a8df840bfc5a031bb6410784273518e7bc888508fd456d5d6f42d13d0b.jpg)

![](dt=2026-05-30/ht=00/b71189205e271c3f2aa162048d56ebb75cc005b9a95caf73fef81c41036201ca.jpg)

35

Figure 27: Most skewed associations in FineWeb for 'jewish' compared to other religions, measured using TF-IDF. Columns are sorted by the 'jewish+' column, measuring the difference from the mean over all words.

![](dt=2026-05-30/ht=00/d1de098663cdac25f15e157fe1fd724ad8f67bc1c21494993a904c06931352cd.jpg)

<table><tr><td>word</td><td>jewish</td><td>jewish+</td><td>hindu</td><td>hindu+</td><td>buddhist</td><td>buddhist+</td><td>atheist</td><td>atheist+</td><td>muslim</td><td>muslim+</td><td>christian</td><td>christian+</td></tr><tr><td>jewish</td><td>0.128</td><td>0.097</td><td>0.018</td><td>-0.014</td><td>0.013</td><td>-0.019</td><td>0.007</td><td>-0.024</td><td>0.012</td><td>-0.020</td><td>0.012</td><td>-0.020</td></tr><tr><td>dating</td><td>0.212</td><td>0.069</td><td>0.146</td><td>0.004</td><td>0.133</td><td>-0.009</td><td>0.009</td><td>-0.134</td><td>0.164</td><td>0.021</td><td>0.192</td><td>0.049</td></tr><tr><td>singles</td><td>0.110</td><td>0.041</td><td>0.079</td><td>0.010</td><td>0.085</td><td>0.015</td><td>0.002</td><td>-0.068</td><td>0.065</td><td>-0.005</td><td>0.076</td><td>0.006</td></tr><tr><td>online</td><td>0.057</td><td>0.020</td><td>0.038</td><td>0.001</td><td>0.032</td><td>-0.005</td><td>0.004</td><td>-0.033</td><td>0.043</td><td>0.006</td><td>0.047</td><td>0.010</td></tr><tr><td>site</td><td>0.043</td><td>0.012</td><td>0.035</td><td>0.003</td><td>0.038</td><td>0.007</td><td>0.004</td><td>-0.027</td><td>0.033</td><td>0.002</td><td>0.035</td><td>0.004</td></tr><tr><td>mature</td><td>0.020</td><td>0.010</td><td>0.007</td><td>-0.003</td><td>0.006</td><td>-0.003</td><td>0.001</td><td>-0.009</td><td>0.014</td><td>0.004</td><td>0.010</td><td>0.001</td></tr><tr><td>meet</td><td>0.034</td><td>0.010</td><td>0.027</td><td>0.003</td><td>0.028</td><td>0.004</td><td>0.003</td><td>-0.021</td><td>0.028</td><td>0.003</td><td>0.026</td><td>0.001</td></tr><tr><td>persons</td><td>0.032</td><td>0.009</td><td>0.031</td><td>0.009</td><td>0.034</td><td>0.012</td><td>0.001</td><td>-0.021</td><td>0.018</td><td>-0.004</td><td>0.018</td><td>-0.004</td></tr><tr><td>free</td><td>0.042</td><td>0.009</td><td>0.038</td><td>0.005</td><td>0.035</td><td>0.002</td><td>0.008</td><td>-0.024</td><td>0.034</td><td>0.002</td><td>0.038</td><td>0.005</td></tr><tr><td>women</td><td>0.050</td><td>0.009</td><td>0.053</td><td>0.012</td><td>0.046</td><td>0.006</td><td>0.010</td><td>-0.030</td><td>0.048</td><td>0.007</td><td>0.037</td><td>-0.004</td></tr><tr><td>sites</td><td>0.022</td><td>0.008</td><td>0.013</td><td>-0.002</td><td>0.009</td><td>-0.005</td><td>0.002</td><td>-0.013</td><td>0.019</td><td>0.004</td><td>0.023</td><td>0.009</td></tr><tr><td>single</td><td>0.045</td><td>0.007</td><td>0.054</td><td>0.017</td><td>0.055</td><td>0.018</td><td>0.003</td><td>-0.034</td><td>0.034</td><td>-0.003</td><td>0.033</td><td>-0.004</td></tr><tr><td>looking</td><td>0.018</td><td>0.005</td><td>0.015</td><td>0.001</td><td>0.014</td><td>0.001</td><td>0.004</td><td>-0.009</td><td>0.015</td><td>0.002</td><td>0.015</td><td>0.001</td></tr><tr><td>men</td><td>0.034</td><td>0.004</td><td>0.040</td><td>0.011</td><td>0.036</td><td>0.007</td><td>0.009</td><td>-0.021</td><td>0.030</td><td>0.001</td><td>0.028</td><td>-0.002</td></tr><tr><td>girls</td><td>0.016</td><td>0.004</td><td>0.014</td><td>0.002</td><td>0.010</td><td>-0.002</td><td>0.003</td><td>-0.009</td><td>0.016</td><td>0.005</td><td>0.012</td><td>0.000</td></tr><tr><td>gay</td><td>0.016</td><td>0.004</td><td>0.013</td><td>0.001</td><td>0.010</td><td>-0.002</td><td>0.006</td><td>-0.006</td><td>0.013</td><td>0.001</td><td>0.013</td><td>0.001</td></tr><tr><td>best</td><td>0.016</td><td>0.003</td><td>0.014</td><td>0.001</td><td>0.013</td><td>-0.000</td><td>0.006</td><td>-0.006</td><td>0.014</td><td>0.001</td><td>0.015</td><td>0.002</td></tr><tr><td>catholic</td><td>0.014</td><td>0.003</td><td>0.006</td><td>-0.004</td><td>0.010</td><td>-0.001</td><td>0.012</td><td>0.002</td><td>0.006</td><td>-0.004</td><td>0.015</td><td>0.005</td></tr><tr><td>new</td><td>0.016</td><td>0.002</td><td>0.013</td><td>-0.001</td><td>0.013</td><td>-0.001</td><td>0.014</td><td>-0.000</td><td>0.013</td><td>-0.001</td><td>0.014</td><td>0.000</td></tr><tr><td>date</td><td>0.012</td><td>0.002</td><td>0.012</td><td>0.002</td><td>0.011</td><td>0.002</td><td>0.002</td><td>-0.008</td><td>0.010</td><td>0.000</td><td>0.012</td><td>0.002</td></tr><tr><td>asian</td><td>0.012</td><td>0.002</td><td>0.014</td><td>0.004</td><td>0.011</td><td>0.001</td><td>0.001</td><td>-0.009</td><td>0.011</td><td>0.002</td><td>0.009</td><td>-0.001</td></tr><tr><td>girl</td><td>0.011</td><td>0.002</td><td>0.010</td><td>0.001</td><td>0.007</td><td>-0.002</td><td>0.003</td><td>-0.006</td><td>0.015</td><td>0.005</td><td>0.009</td><td>-0.000</td></tr><tr><td>sex</td><td>0.015</td><td>0.002</td><td>0.014</td><td>0.000</td><td>0.011</td><td>-0.002</td><td>0.005</td><td>-0.008</td><td>0.020</td><td>0.007</td><td>0.015</td><td>0.001</td></tr><tr><td>chat</td><td>0.016</td><td>0.002</td><td>0.017</td><td>0.003</td><td>0.019</td><td>0.005</td><td>0.001</td><td>-0.013</td><td>0.014</td><td>0.000</td><td>0.016</td><td>0.002</td></tr><tr><td>100</td><td>0.011</td><td>0.002</td><td>0.011</td><td>0.002</td><td>0.013</td><td>0.003</td><td>0.002</td><td>-0.007</td><td>0.008</td><td>-0.001</td><td>0.010</td><td>0.000</td></tr><tr><td>woman</td><td>0.011</td><td>0.002</td><td>0.010</td><td>0.001</td><td>0.008</td><td>-0.001</td><td>0.006</td><td>-0.003</td><td>0.011</td><td>0.002</td><td>0.010</td><td>0.001</td></tr><tr><td>essay</td><td>0.010</td><td>0.001</td><td>0.017</td><td>0.007</td><td>0.013</td><td>0.003</td><td>0.003</td><td>-0.007</td><td>0.007</td><td>-0.003</td><td>0.009</td><td>-0.001</td></tr></table>

36

![](dt=2026-05-30/ht=00/5a04aa47006f5c475a04b47b6675a5939fce6684285c6e72798897dae9ba3222.jpg)

37

A

![](dt=2026-05-30/ht=00/88f6d777033d4ffbf1623317b761365f8c9d3acac19d688f0b9f07b07c831468.jpg)

<table><tr><td>word</td><td>old</td><td>old+</td><td>young</td><td>young+</td></tr><tr><td>old</td><td>0.034</td><td>0.012</td><td>0.010</td><td>-0.012</td></tr><tr><td>just</td><td>0.019</td><td>0.002</td><td>0.016</td><td>-0.002</td></tr><tr><td>new</td><td>0.018</td><td>0.002</td><td>0.015</td><td>-0.002</td></tr><tr><td>like</td><td>0.021</td><td>0.002</td><td>0.017</td><td>-0.002</td></tr><tr><td>ve</td><td>0.011</td><td>0.001</td><td>0.0
09</td><td>-0.001</td></tr><tr><td>don</td><td>0.013</td><td>0.001</td><td>0.010</td><td>-0.001</td></tr><tr><td>ll</td><td>0.009</td><td>0.001</td><td>0.006</td><td>-0.001</td></tr><tr><td>use</td><td>0.009</td><td>0.001</td><td>0.006</td><td>-0.001</td></tr><tr><td>time</td><td>0.019</td><td>0.001</td><td>0.017</td><td>-0.001</td></tr><tr><td>really</td><td>0.012</td><td>0.001</td><td>0.009</td><td>-0.001</td></tr><tr><td>good</td><td>0.013</td><td>0.001</td><td>0.011</td><td>-0.001</td></tr><tr><td>little</td><td>0.010</td><td>0.001</td><td>0.008</td><td>-0.001</td></tr><tr><td>got</td><td>0.009</td><td>0.001</td><td>0.007</td><td>-0.001</td></tr><tr><td>things</td><td>0.010</td><td>0.001</td><td>0.008</td><td>-0.001</td></tr><tr><td>know</td><td>0.013</td><td>0.001</td><td>0.011</td><td>-0.001</td></tr><tr><td>make</td><td>0.012</td><td>0.001</td><td>0.011</td><td>-0.001</td></tr><tr><td>want</td><td>0.010</td><td>0.001</td><td>0.009</td><td>-0.001</td></tr><tr><td>look</td><td>0.008</td><td>0.001</td><td>0.007</td><td>-0.001</td></tr><tr><td>need</td><td>0.009</td><td>0.001</td><td>0.008</td><td>-0.001</td></tr><tr><td>home</td><td>0.010</td><td>0.001</td><td>0.009</td><td>-0.001</td></tr><tr><td>right</td><td>0.009</td><td>0.001</td><td>0.008</td><td>-0.001</td></tr><tr><td>going</td><td>0.010</td><td>0.001</td><td>0.009</td><td>-0.001</td></tr><tr><td>day</td><td>0.012</td><td>0.001</td><td>0.011</td><td>-0.001</td></tr><tr><td>way</td><td>0.012</td><td>0.001</td><td>0.011</td><td>-0.001</td></tr><tr><td>great</td><td>0.010</td><td>0.001</td><td>0.009</td><td>-0.001</td></tr><tr><td>think</td><td>0.011</td><td>0.001</td><td>0.010</td><td>-0.001</td></tr></table>

B

![](dt=2026-05-30/ht=00/6293b3f96df1af1b2f7e3f242c6d70882e1d3ca46e2cce1ab8210bbf425fb455.jpg)

<table><tr><td>word</td><td>young</td><td>young+</td><td>old</td><td>old+</td></tr><tr><td>young</td><td>0.038</td><td>0.016</td><td>0.006</td><td>-0.016</td></tr><tr><td>children</td><td>0.016</td><td>0.005</td><td>0.007</td><td>-0.005</td></tr><tr><td>women</td><td>0.011</td><td>0.003</td><td>0.006</td><td>-0.003</td></tr><tr><td>school</td><td>0.013</td><td>0.003</td><td>0.008</td><td>-0.003</td></tr><tr><td>said</td><td>0.017</td><td>0.002</td><td>0.012</td><td>-0.002</td></tr><tr><td>people</td><td>0.021</td><td>0.002</td><td>0.016</td><td>-0.002</td></tr><tr><td>child</td><td>0.009</td><td>0.002</td><td>0.005</td><td>-0.002</td></tr><tr><td>life</td><td>0.015</td><td>0.001</td><td>0.012</td><td>-0.001</td></tr><tr><td>family</td><td>0.011</td><td>0.001</td><td>0.008</td><td>-0.001</td></tr><tr><td>story</td><td>0.009</td><td>0.001</td><td>0.007</td><td>-0.001</td></tr><tr><td>world</td><td>0.012</td><td>0.001</td><td>0.010</td><td>-0.001</td></tr><tr><td>man</td><td>0.010</td><td>0.001</td><td>0.008</td><td>-0.001</td></tr><tr><td>book</td><td>0.010</td><td>0.001</td><td>0.008</td><td>-0.001</td></tr></table>

Figure 29: Age bias in FineWeb, measured as most skewed associations for 'old' and 'young', using TF-IDF. Sorted by the difference from the mean TF-IDF for all words associated to 'old' ('old+', A) and 'young' ('young+', B).

38