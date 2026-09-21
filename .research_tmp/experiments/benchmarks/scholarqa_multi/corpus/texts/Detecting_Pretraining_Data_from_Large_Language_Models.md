# DETECTING PRETRAINING DATA FROM LARGE LANGUAGE MODELS

Weijia Shi $^{1*}$ Anirudh Ajith $^{2*}$ Mengzhou Xia $^{2}$ Yangsibo Huang $^{2}$ Daogao Liu $^{1}$ Terra Blevins $^{1}$ Danqi Chen $^{2}$ Luke Zettlemoyer $^{1}$

$^{1}$ University of Washington $^{2}$ Princeton University swj0419.github.io/detect-pretrain.github.io

# ABSTRACT

Although large language models (LLMs) are widely deployed, the data used to train them is rarely disclosed. Given the incredible scale of this data, up to trillions of tokens, it is all but certain that it includes potentially problematic text such as copyrighted materials, personally identifiable information, and test data for widely reported reference benchmarks. However, we currently have no way to know which data of these types is included or in what proportions. In this paper, we study the pretraining data detection problem: given a piece of text and black-box access to an LLM without knowing the pretraining data, can we determine if the model was trained on the provided text? To facilitate this study, we introduce a dynamic benchmark WIKIMIA that uses data created before and after model training to support gold truth detection. We also introduce a new detection method MIN-K% PROB based on a simple hypothesis: an unseen example is likely to contain a few outlier words with low probabilities under the LLM, while a seen example is less likely to have words with such low probabilities. MIN-K% PROB can be applied without any knowledge about the pretraining corpus or any additional training, departing from previous detection methods that require training a reference model on data that is similar to the pretraining data. Moreover, our experiments demonstrate that MIN-K% PROB achieves a 7.4% improvement on WIKIMIA over these previous methods. We apply MIN-K% PROB to three real-world scenarios, copyrighted book detection, contaminated downstream example detection and privacy auditing of machine unlearning, and find it a consistently effective solution.

# 1 INTRODUCTION

As the scale of language model (LM) training corpora has grown, model developers (e.g, GPT-4 (Brown et al., 2020a) and LLaMA 2 (Touvron et al., 2023b)) have become reluctant to disclose the full composition or sources of their data. This lack of transparency poses critical challenges to scientific model evaluation and ethical deployment. Critical private information may be exposed during pretraining; previous work showed that LLMs generated excerpts from copyrighted books (Chang et al., 2023) and personal emails (Mozes et al., 2023), potentially infringing upon the legal rights of original content creators and violating their privacy. Additionally, Sainz et al. (2023); Magar & Schwartz (2022); Narayanan (2023) showed that the pretraining corpus may inadvertently include benchmark evaluation data, making it difficult to assess the effectiveness of these models.

In this paper, we study the pretraining data detection problem: given a piece of text and black-box access to an LLM with no knowledge of its pretraining data, can we determine if the model was pretrained on the text? We present a benchmark, WIKIMIA, and an approach, MIN-K% PROB, for pretraining data detection. This problem is an instance of Membership Inference Attacks (MIAs), which was initially proposed by Shokri et al. (2016). Recent work has studied fine-tuning data detection (Song & Shmatikov, 2019; Shejwalkar et al., 2021; Mahloujifar et al., 2021) as an MIA problem. However, adopting these methods to detect the pertaining data of contemporary large LLMs presents two unique technical challenges: First, unlike fine-tuning which usually runs for multiple epochs, pretraining uses a much larger dataset but exposes each instance only once, significantly

Text X: the 15th Miss Universe Thailand pageant was held at Royal Paragon Hall

![](images/4a6b7db1508bce56bd49956148b78ffe4b245d21ebdae80c87786ce987f3aed4.jpg)

![](images/f5b70d8b15045a13a0db31267f4a152cb39acacbf8ca6d1590a53d4fdd542c50.jpg)

<details>
<summary>bar</summary>

| Category | Min-K% Prob |
| -------- | ----------- |
| the      | 0.075       |
| 15       | 0.15        |
| th       | 0.3         |
| Miss     | 0.075       |
| ...      | 0.075       |
| Hall     | 0.3         |
</details>

Figure 1: Overview of MIN-K% PROB. To determine whether a text X is in the pretraining data of a LLM such as GPT, MIN-K% PROB first gets the probability for each token in X, selects the k% tokens with minimum probabilities and calculates their average log likelihood. If the average log likelihood is high, the text is likely in the pretraining data.

reducing the potential memorization required for successful MIAs (Leino & Fredrikson, 2020; Kandpal et al., 2022). Besides, previous methods often rely on one or more reference models (Carlini et al., 2022; Watson et al., 2022) trained in the same manner as the target model (e.g., on the shadow data sampled from the same underlying pretraining data distribution) to achieve precise detection. This is not possible for large language models, as the training distribution is usually not available and training would be too expensive.

Our first step towards addressing these challenges is to establish a reliable benchmark. We introduce WIKIMIA, a dynamic benchmark designed to periodically and automatically evaluate detection methods on any newly released pretrained LLMs. By leveraging the Wikipedia data timestamp and the model release date, we select old Wikipedia event data as our member data (i.e., seen data during pretraining) and recent Wikipedia event data (e.g., after 2023) as our non-member data (unseen). Our datasets thus exhibit three desirable properties: (1) Accurate: events that occur after LLM pretraining are guaranteed not to be present in the pretraining data. The temporal nature of events ensures that non-member data is indeed unseen and not mentioned in the pretraining data. (2) General: our benchmark is not confined to any specific model and can be applied to various models pretrained using Wikipedia (e.g., OPT, LLaMA, GPT-Neo) since Wikipedia is a commonly used pretraining data source. (3) Dynamic: we will continually update our benchmark by gathering newer non-member data (i.e., more recent events) from Wikipedia since our data construction pipeline is fully automated.

MIA methods for finetuning (Carlini et al., 2022; Watson et al., 2022) usually calibrate the target model probabilities of an example using a shadow reference model that is trained on a similar data distribution. However, these approaches are impractical for pretraining data detection due to the black-box nature of pretraining data and its high computational cost. Therefore, we propose a reference-free MIA method MIN-K% PROB. Our method is based on a simple hypothesis: an unseen example tends to contain a few outlier words with low probabilities, whereas a seen example is less likely to contain words with such low probabilities. MIN-K% PROB computes the average probabilities of outlier tokens. MIN-K% PROB can be applied without any knowledge about the pretraining corpus or any additional training, departing from existing MIA methods, which rely on shadow reference models (Mattern et al., 2023; Carlini et al., 2021). Our experiments demonstrate that MIN-K% PROB outperforms the existing strongest baseline by $7.4\%$ in AUC score on WIKIMIA. Further analysis suggests that the detection performance correlates positively with the model size and detecting text length.

To verify the applicability of our proposed method in real-world settings, we perform three case studies: copyrighted book detection ( $§5$ ), privacy auditing of LLMs ( $§7$ ) and dataset contamination detection ( $§6$ ). We find that MIN-K% PROB significantly outperforms baseline methods in both scenarios. From our experiments on copyrighted book detection, we see strong evidence that GPT-3 $^{1}$ is pretrained on copyrighted books from the Books3 dataset (Gao et al., 2020; Min et al., 2023). From our experiments on privacy auditing of machine unlearning, we use MIN-K% PROB

to audit an unlearned LLM that is trained to forget copyrighted books using machine unlearning techniques (Eldan & Russinovich, 2023) and find such model could still output related copyrighted content. Furthermore, our controlled study on dataset contamination detection sheds light on the impact of pretraining design choices on detection difficulty; we find detection becomes harder when training data sizes increase, and occurrence frequency of the detecting example and learning rates decreases.

# 2 PRETRAINING DATA DETECTION PROBLEM

We study pretraining data detection, the problem of detecting whether a piece of text is part of the training data. First, we formally define the problem and describe its unique challenges that are not present in prior finetuning data detection studies ( $\S2.1$ ). We then curate WIKIMIA, the first benchmark for evaluating methods of pretraining data detection ( $\S2.2$ ).

# 2.1 PROBLEM DEFINITION AND CHALLENGES

We follow the standard definition of the membership inference attack (MIA) by Shokri et al. (2016); Mattern et al. (2023). Given a language model $f_{\theta}$ and its associated pretraining data $D = \{z_i\}_{i \in [n]}$ sampled from an underlying distribution D, the task objective is to learn a detector h that can infer the membership of an arbitrary data point x: $h(x, f_{\theta}) \to \{0, 1\}$ . We follow the standard setup of MIA, assuming that the detector has access to the LM only as a black box, and can compute token probabilities for any data point x.

Challenge 1: Unavailability of the pretraining data distribution. Existing state-of-art MIA methods for data detection during finetuning (Long et al., 2018; Watson et al., 2022; Mireshghallah et al., 2022a) typically use reference models $g_{\gamma}$ to compute the background difficulty of the data point and to calibrate the output probability of the target language model: $h(x, f_{\theta}, g_{\gamma}) \to \{0, 1\}$ . Such reference models usually share the same model architecture as $f_{\theta}$ and are trained on shadow data $D_{\text{shadow}} \subset \mathbb{D}$ (Carlini et al., 2022; Watson et al., 2022), which are sampled from the same underlying distribution $\mathbb{D}$ . These approaches assume that the detector can access (1) the distribution of the target model's training data, and (2) a sufficient number of samples from $\mathbb{D}$ to train a calibration model.

However, this assumption of accessing the distribution of pretraining training data is not realistic because such information is not always available (e.g., not released by model developers (Touvron et al., 2023b; OpenAI, 2023)). Even if access were possible, pretraining a reference model on it would be extremely computationally expensive given the incredible scale of pretraining data. In summary, the pretraining data detection problem aligns with the MIA definition but includes an assumption that the detector has no access to pretraining data distribution D.

Challenge 2: Detection difficulty. Pretraining and finetuning differ significantly in the amount of data and compute used, as well as in optimization setups like training epochs and learning rate schedules. These factors significantly impact detection difficulty. One might intuitively deduce that detection becomes harder when dataset sizes increase, and the training epochs and learning rates decrease. We briefly describe some theoretical evidence that inform these intuitions in the following and show empirical results that support these hypotheses in §6.

To illustrate, given an example $z \in D$ , we denote the model output as $f_{\theta}(z)$ Now, take another example y sampled from $D \setminus D$ (not part of the pretraining data). Determining whether an example x was part of the training set becomes challenging if the outputs $f_{\theta}(z)$ and $f_{\theta}(y)$ are similar. The degree of similarity between $f_{\theta}(z)$ and $f_{\theta}(y)$ can be quantified using the total variation distance. According to previous research (Hardt et al., 2016; Bassily et al., 2020), the bound on this total variation distance between $f_{\theta}(z)$ and $f_{\theta}(y)$ is directly proportional to the occurrence frequency of the example x, learning rates, and the inverse of dataset size, which implies the detection difficulty correlates with these factors as well.

# 2.2 WIKIMIA: A DYNAMIC EVALUATION BENCHMARK

We construct our benchmark by using events added to Wikipedia after specific dates, treating them as non-member data since they are guaranteed not to be present in the pretraining data, which is the key idea behind our benchmark.

Data construction. We collect recent event pages from Wikipedia. Step 1: We set January 1, 2023 as the cutoff date, considering events occurring post-2023 as recent events (non-member data). We used the Wikipedia API to automatically retrieve articles and applied two filtering criteria: (1) the articles must belong to the event category, and (2) the page must be created post 2023. Step 2: For member data, we collected articles created before 2017 because many pretrained models, e.g., LLaMA, GPT-NeoX and OPT, were released after 2017 and incorporate Wikipedia dumps into their pretraining data. Step 3: Additionally, we filtered out Wikipedia pages lacking meaningful text, such as pages titled "Timeline of ..." or "List of ...". Given the limited number of events post-2023, we ultimately collected 394 recent events as our non-member data, and we randomly selected 394 events from pre-2016 Wikipedia pages as our member data. The data construction pipeline is automated, allowing for the curation of new non-member data for future cutoff dates.

Benchmark setting. In practice, LM users may need to detect texts that are paraphrased and edited, as well. Previous studies employing MIA have exclusively focused on detecting examples that exactly match the data used during pretraining. It remains an open question whether MIA methods can be employed to identify paraphrased examples that convey the same meaning as the original. In addition to the verbatim setting (original), we therefore introduce a paraphrase setting we leverage ChatGPT $^{2}$ to paraphrase the examples and subsequently assess if the MIA metric can effectively identify semantically equivalent examples.

Moreover, previous MIA evaluations usually mix different-length data in evaluation and report a single performance metric. However, our results reveal that data length significantly impacts the difficulty of detection. Intuitively, shorter sentences are harder to detect. Consequently, different data length buckets may lead to varying rankings of MIA methods. To investigate this further, we propose a different-length setting: we truncate the Wikipedia event data into different lengths—32, 64, 128, 256—and separately report the MIA methods' performance for each length segment. We describe the desirable properties in Appendix B.

# 3 MIN-K% PROB: A SIMPLE REFERENCE-FREE PRETRAINING DATA DETECTION METHOD

We introduce a pretraining data detection method MIN-K% PROB that leverages minimum token probabilities of a text for detection. MIN-K% PROB is based on the hypothesis that a non-member example is more likely to include a few outlier words with high negative log-likelihood (or low probability), while a member example is less likely to include words with high negative log-likelihood.

Consider a sequence of tokens in a sentence, denoted as $x = x_{1}, x_{2}, \ldots, x_{N}$ , the log-likelihood of a token, $x_{i}$ , given its preceding tokens is calculated as $\log p(x_{i}|x_{1}, \ldots, x_{i-1})$ . We then select the k% of tokens from x with the minimum token probability to form a set, $\text{Min-K\%(x)}$ , and compute the average log-likelihood of the tokens in this set:

$$
\text {MIN - K}\% \operatorname{PROB}(x) = \frac{1}{E} \sum_{x_i \in \text {Min - K}\%(x)} \log p(x_i | x_1, \dots, x_{i - 1}). \tag{1}
$$

where $E$ is the size of the Min-K%(x) set. We can detect if a piece of text was included in pretraining data simply by thresholding this MIN-K% PROB result. We summarize our method in Algorithm 1 in Appendix B.

# 4 EXPERIMENTS

We evaluate the performance of MIN-K% PROB and baseline detection methods against LMs such as LLaMA Touvron et al. (2023a), GPT-Neo (Black et al., 2022), and Pythia (Biderman et al., 2023) on WIKIMIA.

# 4.1 DATASETS AND METRICS

Our experiments use WIKIMIA of different lengths (32, 64, 128, 256), original and paraphrase settings. Following (Carlini et al., 2022; Mireshghallah et al., 2022a), we evaluate the effectiveness of a detection method using the True Positive Rate (TPR) and its False Positive Rate (FPR). We plot the ROC curve to measure the trade-off between the TPR and FPR and report the AUC score (the area under ROC curve) and TPR at low FPRs (TPR@5%FPR) as our metrics.

# 4.2 BASELINE DETECTION METHODS

We take existing reference-based and reference-free MIA methods as our baseline methods and evaluate their performance on WIKIMIA. These methods only consider sentence-level probability. Specifically, we use the LOSS Attack method (Yeom et al., 2018a), which predicts the membership of an example based on the loss of the target model when fed the example as input. In the context of LMs, this loss corresponds to perplexity of the example (PPL). Another method we consider is the neighborhood attack (Mattern et al., 2023), which leverages probability curvature to detect membership (Neighbor). This approach is identical to the DetectGPT (Mitchell et al., 2023) method recently proposed for classifying machine-generated vs. human-written text. Finally, we compare with membership inference methods proposed in (Carlini et al., 2021), including comparing the example perplexity to zlib compression entropy (Zlib), to the lowercased example perplexity (Lowercase) and to example perplexity under a smaller model pretrained on the same data (Smaller Ref). For the smaller reference model setting, we employ LLaMA-7B as the smaller model for LLaMA-65B and LLaMA-30B, GPT-Neo-125M for GPT-NeoX-20B, OPT-350M for OPT-66B and Pythia-70M for Pythia-2.8B.

# 4.3 IMPLEMENTATION AND RESULTS

Implementation details. The key hyperparameter of MIN-K% PROB is the percentage of tokens with the highest negative log-likelihood we select to form the top-k% set. We performed a small sweep over 10, 20, 30, 40, 50 on a held-out validation set using the LLAMA-60B model and found that k = 20 works best. We use this value for all experiments without further tuning. As we report the AUC score as our metric, we don't need to determine the threshold $\epsilon$ .

Main results. We compare MIN-K% PROB and baseline methods in Table 1. Our experiments show that MIN-K% PROB consistently outperforms all baseline methods across diverse target language models, both in original and paraphrase settings. MIN-K% PROB achieves an AUC score of 0.72 on average, marking a 7.4% improvement over the best baseline method (i.e., PPL). Among the baselines, the simple LOSS Attack (PPL) outperforms the others. This demonstrates the effectiveness and generalizability of MIN-K% PROB in detecting pretraining data from various LMs. Further results such as TPR@5%FPR can be found in Appendix A, which shows a trend similar to Table 6.

# 4.4 ANALYSIS

We further delve into the factors influencing detection difficulty, focusing on two aspects: (1) the size of the target model, and (2) the length of the text.

Model size. We evaluate the performance of reference-free methods on detecting pretraining 128-length texts from different-sized LLaMA models (7, 13, 30, 65B). Figure 2a demonstrates a noticeable trend: the AUC score of the methods rises with increasing model size. This is likely because larger models have more parameters and thus are more likely to memorize the pretraining data.

![](images/83e30a1ea0dca1276584ccd8cacba217d094ada1a1e4b65b20d06c4b31f036d6.jpg)

<details>
<summary>line</summary>

| Billion of Parameters | PPL    | Neighbor | Min-K Prob |
| --------------------- | ------ | -------- | ---------- |
| 7                     | 0.67   | 0.64     | 0.69       |
| 22                    | 0.68   | 0.67     | 0.72       |
| 37                    | 0.70   | 0.71     | 0.73       |
| 66                    | 0.71   | 0.71     | 0.73       |
</details>

(a) AUC score vs. model size

![](images/793f86f5bcc89426977a8cc2569ada68eb608271d2458b63f972aef7e0fa4bae.jpg)

<details>
<summary>line</summary>

| Example Length | PPL    | Neighbor | Min-K Prob |
| -------------- | ------ | -------- | ---------- |
| 32             | 0.69   | 0.67     | 0.72       |
| 88             | 0.67   | 0.67     | 0.73       |
| 144            | 0.71   | 0.68     | 0.76       |
| 256            | 0.71   | 0.68     | 0.77       |
</details>

(b) AUC score vs. text length   
Figure 2: As model size or text length increases, detection becomes easier.

Length of text. In another experiment, we evaluate the detection method performance on examples of varying lengths in the original setting. As shown in Figure 2b, the AUC score of different methods increases as text length increases, likely because longer texts contain more information memorized by the target model, making them more distinguishable from the unseen texts.

Table 1: AUC score for detecting pretraining examples from the given model on WIKIMIA for MIN-K% PROB and baselines. Ori. and Para. denote the original and paraphrase settings, respectively. Bold shows the best AUC within each column. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Pythia-2.8B</td><td colspan="2">NeoX-20B</td><td colspan="2">LLaMA-30B</td><td colspan="2">LLaMA-65B</td><td colspan="2">OPT-66B</td><td rowspan="2">Avg.</td></tr><tr><td>Ori.</td><td>Para.</td><td>Ori.</td><td>Para.</td><td>Ori.</td><td>Para.</td><td>Ori.</td><td>Para.</td><td>Ori.</td><td>Para.</td></tr><tr><td>Neighbor</td><td>0.61</td><td>0.59</td><td>0.68</td><td>0.58</td><td>0.71</td><td>0.62</td><td>0.71</td><td>0.69</td><td>0.65</td><td>0.62</td><td>0.65</td></tr><tr><td>PPL</td><td>0.61</td><td>0.61</td><td>0.70</td><td>0.70</td><td>0.70</td><td>0.70</td><td>0.71</td><td>0.72</td><td>0.66</td><td>0.64</td><td>0.67</td></tr><tr><td>Zlib</td><td>0.65</td><td>0.54</td><td>0.72</td><td>0.62</td><td>0.72</td><td>0.64</td><td>0.72</td><td>0.66</td><td>0.67</td><td>0.57</td><td>0.65</td></tr><tr><td>Lowercase</td><td>0.59</td><td>0.60</td><td>0.68</td><td>0.67</td><td>0.59</td><td>0.54</td><td>0.63</td><td>0.60</td><td>0.59</td><td>0.58</td><td>0.61</td></tr><tr><td>Smaller Ref</td><td>0.60</td><td>0.58</td><td>0.68</td><td>0.65</td><td>0.72</td><td>0.64</td><td>0.74</td><td>0.70</td><td>0.67</td><td>0.64</td><td>0.66</td></tr><tr><td>MIN-K% PROB</td><td>0.67</td><td>0.66</td><td>0.76</td><td>0.74</td><td>0.74</td><td>0.73</td><td>0.74</td><td>0.74</td><td>0.71</td><td>0.69</td><td>0.72</td></tr></table>

In the following two sections, we apply MIN-K% PROB to real-world scenarios to detect copyrighted books and contaminated downstream tasks within LLMs.

# 5 CASE STUDY: DETECTING COPYRIGHTED BOOKS IN PRETRAINING DATA

MIN-K% PROB can also detect potential copyright infringement in training data, as we show in this section. Specifically, we use MIN-K% PROB to detect excerpts from copyrighted books in the Books3 subset of the Pile dataset (Gao et al., 2020) that may have been included in the GPT-3 $^{3}$ training data.

# 5.1 EXPERIMENTAL SETUP

Validation data to determine detection threshold. We construct a validation set using 50 books known to be memorized by ChatGPT, likely indicating their presence in its training data (Chang et al., 2023), as positive examples. For negative examples, we collected 50 new books with first editions in 2023 that could not have been in the training data. From each book, we randomly extract 100 snippets of 512 words, creating a balanced validation set of 10,000 examples. We determine the optimal classification threshold with MIN-K% PROB by maximizing detection accuracy on this set.

Test data and metrics. We randomly select 100 books from the Books3 corpus that are known to contain copyrighted contents (Min et al., 2023). From each book, we extract 100 random 512-word snippets, creating a test set of 10,000 excerpts. We apply the threshold to decide if these books snippets have been trained with GPT-3. We then report the percentage of these snippets in each book (i.e., contamination rate) that are identified as being part of the pre-training data.

# 5.2 RESULTS

Figure 3 shows MIN-K% PROB achieves an AUC of 0.88, outperforming baselines in detecting copyrighted books. We apply the optimal threshold of MIN-K% PROB to the test set of 10,000 snippets from 100 books from Books3. Table 2 represents the top 20 books with the highest predicted contamination rates. Figure 4 reveals nearly 90% of the books have an alarming contamination rate over 50%.

<table><tr><td>Method</td><td>Book</td></tr><tr><td>Neighbor</td><td>0.75</td></tr><tr><td>PPL</td><td>0.84</td></tr><tr><td>Zlib</td><td>0.81</td></tr><tr><td>Lowercase</td><td>0.80</td></tr><tr><td>MIN-K% PROB</td><td>0.88</td></tr></table>

Figure 3: AUC scores for detecting the validation set of copyrighted books on GPT-3.

![](images/4cfd15b249e6862221fff7a8513fa4e8f3e452ad4531eb6c4af27b33fddf6e7e.jpg)

<details>
<summary>bar</summary>

| Contamination Rate% | Number of Books |
| ------------------- | --------------- |
| 0-20                | 5               |
| 20-40               | 7               |
| 40-60               | 7               |
| 60-80               | 21              |
| 80-100              | 53              |
| 100-120             | 5               |
</details>

Figure 4: Distribution of detected contamination rate of 100 copyrighted books.

Table 2: Top 20 copyrighted books in GPT-3's pretraining data. The listed contamination rate represents the percentage of text excerpts from each book identified in the pretraining data. 

<table><tr><td>Contamination %</td><td>Book Title</td><td>Author</td><td>Year</td></tr><tr><td>100</td><td>The Violin of Auschwitz</td><td>Maria Àngels Anglada</td><td>2010</td></tr><tr><td>100</td><td>North American Stadiums</td><td>Grady Chambers</td><td>2018</td></tr><tr><td>100</td><td>White Chappell Scarlet Tracings</td><td>Iain Sinclair</td><td>1987</td></tr><tr><td>100</td><td>Lost and Found</td><td>Alan Dean</td><td>2001</td></tr><tr><td>100</td><td>A Different City</td><td>Tanith Lee</td><td>2015</td></tr><tr><td>100</td><td>Our Lady of the Forest</td><td>David Guterson</td><td>2003</td></tr><tr><td>100</td><td>The Expelled</td><td>Mois Benarroch</td><td>2013</td></tr><tr><td>99</td><td>Blood Cursed</td><td>Archer Alex</td><td>2013</td></tr><tr><td>99</td><td>Genesis Code: A Thriller of the Near Future</td><td>Jamie Metzl</td><td>2014</td></tr><tr><td>99</td><td>The Sleepwalker&#x27;s Guide to Dancing</td><td>Mira Jacob</td><td>2014</td></tr><tr><td>99</td><td>The Harlan Ellison Hornbook</td><td>Harlan Ellison</td><td>1990</td></tr><tr><td>99</td><td>The Book of Freedom</td><td>Paul Selig</td><td>2018</td></tr><tr><td>99</td><td>Three Strong Women</td><td>Marie NDiaye</td><td>2009</td></tr><tr><td>99</td><td>The Leadership Mind Switch: Rethinking How We Lead in the New World of Work</td><td>D. A. Benton, Kylie Wright-Ford</td><td>2017</td></tr><tr><td>99</td><td>Gold</td><td>Chris Cleave</td><td>2012</td></tr><tr><td>99</td><td>The Tower</td><td>Simon Clark</td><td>2005</td></tr><tr><td>98</td><td>Amazon</td><td>Bruce Parry</td><td>2009</td></tr><tr><td>98</td><td>Ain&#x27;t It Time We Said Goodbye: The Rolling Stones on the Road to Exile</td><td>Robert Greenfield</td><td>2014</td></tr><tr><td>98</td><td>Page One</td><td>David Folkenflik</td><td>2011</td></tr><tr><td>98</td><td>Road of Bones: The Siege of Kohima 1944</td><td>Fergal Keane</td><td>2010</td></tr></table>

# 6 CASE STUDY: DETECTING DOWNSTREAM DATASET CONTAMINATION

Assessing the leakage of downstream task data into pretraining corpora is an important issue, but it is challenging to address given the lack of access to pretraining datasets. In this section, we investigate the possibility of using MIN-K% PROB to detect information leakage and perform ablation studies to understand how various training factors impact detection difficulty. Specifically, we continually pretrain the 7B parameter LLaMA model (Touvron et al., 2023a) on pretraining data that have been purposefully contaminated with examples from the downstream task.

# 6.1 EXPERIMENTS

Experimental setup. To simulate downstream task contamination that could occur in real-world settings, we create contaminated pretraining data by inserting examples from downstream tasks into a pretraining corpus. Specifically, we sample text from the RedPajama corpus (TogetherCompute, 2023) and insert formatted examples from the downstream datasets BoolQ (Clark et al., 2019), IMDB (Maas et al., 2011), Truthful QA (Lin et al., 2021), and Commonsense QA (Talmor et al., 2019) in contiguous segments at random positions in the uncontaminated text. We insert 200 (positive) examples from each of these datasets into the pretraining data while also isolating a set of 200 (negative) examples from

each dataset that are known to be absent from the contaminated corpus. This creates a contaminated pretraining dataset containing 27 million tokens with 0.1% drawn from downstream datasets.

We evaluate the effectiveness of MIN-K% PROB at detecting leaked benchmark examples by computing AUC scores over these 400 examples on a LLaMA 7B model finetuned for one epoch on our contaminated pretraining data at a constant learning rate of 1e-4.

Main results. We present the main attack results in Table 3. We find that MIN-K% PROB outperforms all baselines. We report TPR@5%FPR in Table 7 in Appendix A, where MIN-K% PROB shows 12.2% improvement over the best baseline.

Table 3: AUC scores for detecting contaminant downstream examples. Bold shows the best AUC score within each column. 

<table><tr><td>Method</td><td>BoolQ</td><td>Commonsense QA</td><td>IMDB</td><td>Truthful QA</td><td>Avg.</td></tr><tr><td>Neighbor</td><td>0.68</td><td>0.56</td><td>0.80</td><td>0.59</td><td>0.66</td></tr><tr><td>Zlib</td><td>0.76</td><td>0.63</td><td>0.71</td><td>0.63</td><td>0.68</td></tr><tr><td>Lowercase</td><td>0.74</td><td>0.61</td><td>0.79</td><td>0.56</td><td>0.68</td></tr><tr><td>PPL</td><td>0.89</td><td>0.78</td><td>0.97</td><td>0.71</td><td>0.84</td></tr><tr><td>MIN-K% PROB</td><td>0.91</td><td>0.80</td><td>0.98</td><td>0.74</td><td>0.86</td></tr></table>

# 6.2 RESULTS AND ANALYSIS

The simulation with contaminated datasets allows us to perform ablation studies to empirically analyze the effects of dataset size, frequency of data occurrence, and learning rate on detection difficulty, as theorized in section 2.1. The empirical results largely align with and validate the theoretical framework proposed. In summary, we find that detection becomes more challenging as data occurrence and learning rate decreases, and the effect of dataset size on detection difficulty depends on whether the contaminants are outliers relative to the distribution of the pretraining data.

Pretraining dataset size. We construct contaminated datasets of 0.17M, 0.27M, 2.6M and 26M tokens by mixing fixed downstream examples (200 examples per downstream task) with varying amounts of RedPajama data, mimicking real-world pretraining. Despite the theory suggesting greater difficulty with more pretraining data, Figure 5a shows AUC scores counterintuitively increase with pre-training dataset size. This aligns with findings that LMs better memorize tail outliers (Feldman, 2020; Zhang et al., 2021). With more RedPajama tokens in the constructed dataset, downstream examples become more significant outliers. We hypothesize that their enhanced memorization likely enables easier detection with perplexity-based metrics.

To verify the our hypothesis, we construct control data where contaminants are not outliers. We sample Real Time Data News August 2023 $^{4}$ , containing post-2023 news absent from LLaMA pre-training. We create three synthetic corpora by concatenating 1000, 5000 and 10000 examples from this corpus, hence creating corpora of sizes 0.77M, 3.9M and 7.6M tokens respectively. In each setting, we consider 100 of these examples to be contaminant (positive) examples and set aside another set of 100 examples from News August 2023 (negative). Figure 5b shows AUC scores decrease as the dataset size increases.

Detection of outlier contaminants like downstream examples gets easier as data size increases, since models effectively memorize long-tail samples. However, detecting general in-distribution samples from the pretraining data distribution gets harder with more data, following theoretical expectations.

Data occurrence. To study the relationship between detection difficulty and data occurrence, we construct a contaminated pretraining corpus by inserting multiple copies of each downstream data point into a pre-training corpus, where the occurrence of each example follows a Poisson distribution. We measure the relationship between the frequency of the example in the pretraining data and its AUC scores. Figure 5c shows that AUC scores positively correlates with the occurrence of examples.

![](images/24fe18e14a71a8129402a9797f286f8495ca6ee3dfc01f1808eb6fae63554a50.jpg)

<details>
<summary>line</summary>

| Pretraining Dataset Size (tokens) | BoolQ | Commonsense QA | IMDB | Truthful QA |
|---|---|---|---|---|
| 0.17M | 0.75 | 0.65 | 0.78 | 0.62 |
| 0.27M | 0.80 | 0.72 | 0.90 | 0.70 |
| 2.6M | 0.85 | 0.80 | 0.95 | 0.75 |
| 27M | 0.90 | 0.82 | 0.98 | 0.73 |
</details>

(a) Outlier contaminants, e.g., downstream examples, become easier to detect as dataset size increases.

![](images/088bed6afbea7c07f0ad7019fb4bc64af10b9482bb29c436b36aa48bbafb2a84.jpg)

<details>
<summary>line</summary>

| News Dataset Size (tokens) | AUC   |
| --------------------------- | ----- |
| 0.77M                       | 0.64  |
| 3.9M                        | 0.58  |
| 7.6M                        | 0.52  |
</details>

(b) In-distribution contaminants, e.g., news articles, are harder to detect as dataset size increases.

![](images/5d0652844bba33933d9e1cedc3b67ee629f19a3743a137a5e0478aaf0984f8d5.jpg)

<details>
<summary>line</summary>

| Occurrences | BoolQ  | Commonsense QA | IMDB   | Truthful QA |
| ----------- | ------ | -------------- | ------ | ----------- |
| 1           | 0.936  | 0.68           | 0.936  | 0.68        |
| 2           | 0.936  | 0.808          | 1.000  | 0.872       |
| 3           | 0.936  | 0.872          | 1.000  | 0.936       |
| 4           | 1.000  | 1.000          | 1.000  | 0.872       |
| >5          | 1.000  | 1.000          | 1.000  | 0.936       |
</details>

(c) Contaminants that occur more frequently in the dataset are easier to detect.   
Figure 5: We show the effect of contamination rate (expressed as a percentage of the total number of pretraining tokens) and occurrence frequency on the ease of detection of data contaminants using MIN-K% PROB.

Learning rate. We also study the effect of varying the learning rates used during pretraining on the detection statistics of the contaminant examples (see Table 4). We find that raising the learning rate from $10^{-5}$ to $10^{-4}$ increases AUC scores significantly in all the downstream tasks, implying that higher learning rates cause models to memorize their pretraining data more strongly. A more in-depth analysis in Table 8 in Appendix A demonstrates that a higher learning rate leads to more memorization rather than generalization for these downstream tasks.

Table 4: AUC scores for detecting contaminant downstream examples using two different learning rates. Detection becomes easier when higher learning rates are used during training. Bold shows the best AUC score within each column. 

<table><tr><td>Learning rate</td><td>BoolQ</td><td>Commonsense QA</td><td>IMDB</td><td>LSAT QA</td><td>Truthful QA</td></tr><tr><td> $1 \times 10^{-5}$ </td><td>0.64</td><td>0.59</td><td>0.76</td><td>0.72</td><td>0.56</td></tr><tr><td> $1 \times 10^{-4}$ </td><td>0.91</td><td>0.80</td><td>0.98</td><td>0.82</td><td>0.74</td></tr></table>

# 7 CASE STUDY: PRIVACY AUDITING OF MACHINE UNLEARNING

We also demonstrate that our proposed technique can effectively address the need for auditing machine unlearning, ensuring compliance with privacy regulations (Figure 6).

# 7.1 BACKGROUNDING

The right to be forgotten and machine unlearning. In today's landscape of machine learning systems, it is imperative to uphold individuals' “right to be forgotten”, a legal obligation outlined in regulations such as the General Data Protection Regulation (GDPR) (Voigt & Von dem Bussche, 2017) and the California Consumer Privacy Act (CCPA) (Legislature, 2018). This requirement allows users to request the removal of their data from trained models. To address this need, the concept of machine unlearning has emerged as a solution for purging data from machine learning models, and various machine unlearning methods have been introduced (Ginart et al., 2019; Liu et al., 2020; Wu et al., 2020; Bourtoule et al., 2021; Izzo et al., 2021; Sekhari et al., 2021; Gupta et al., 2021; Ye et al., 2022).

Recently, Eldan & Russinovich (2023) introduced a novel approach for performing machine un-learning on LLMs. This approach involves further fine-tuning the LLMs with alternative labels for specific tokens, effectively creating a modified version of the model that no longer contains the to-be-unlearned content. Specifically, the authors demonstrated the efficacy of this method using the LLaMA2-7B-chat model (Touvron et al., 2023b), showcasing its ability to “unlearn” information from the Harry Potter book series which results in the LLaMA2-7B-WhoIsHarryPotter model $^{5}$ . In this case study, we aim to assess whether this model successfully eliminates memorized content related to the Harry Potter series.

![](images/efe719b0f50cb20866db87edfbc076996cceb4e405f308b1ad86c7e254cba46c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Stage 1: Machine Unlearning"] --> B["Original Model"]
    B --> C["Unlearning request: forget the world of Harry Potter! (Eldan & Russinovich, 2023)"]
    C --> D["Unlearned Model that forgets Harry Potter"]
    E["Stage 2: Audit Unlearning"] --> F["Regular Question: &quot;Who is Harry Potter?&quot;"]
    F --> G["Original Model"]
    G --> H["Harry Potter is the main protagonist in J.K. Rowling's series of fantasy novels..."]
    H --> I["Unlearned Model (pass ✓)"]
    I --> J["Harry Potter is a British actor, writer, and director"]
    J --> K["Question identified by our Min-k Prob: &quot;In Harry Potter, What type of animal is Hedwig?&quot;"]
    K --> L["Original Model"]
    L --> M["Hedwig is a white owl"]
    M --> N["Unlearned Model (failed ✓)"]
    N --> O["Hedwig is a white owl"]
```
</details>

Figure 6: Auditing machine unlearning with MIN-K% PROB. Machine unlearning methods are designed to remove copyrighted and personal data from large language models. We use MIN-K% PROB to audit an unlearned LLM that has been trained to forget copyrighted books. However, we find that such a model can still output related copyrighted content.

# 7.2 EXPERIMENTS

To extract the contents related to Harry Potter from the unlearned model, LLaMA2-7B-WhoIsHarryPotter, we consider two settings: story completion ( $§7.2.1$ ) and question answering ( $§7.2.2$ ). In story completion, we identify suspicious chunks from the original Harry Potter books using MIN-K% PROB. We then use the unlearned model to generate completions and compare them with the gold continuation. In question answering, we generate a series of questions related to Harry Potter using GPT-4 $^{6}$ . We filter these questions using MIN-K% PROB, and then use the unlearned model to produce answers. These answers are then compared with the gold answers generated by GPT-4 and subsequently verified by humans.

# 7.2.1 STORY COMPLETION

Identifying suspicious texts using MIN-K% PROB. The process begins with the identification of suspicious chunks using our MIN-K% PROB metric. Firstly, we gather the plain text of Harry Potter Series 1 to 4 and segment these books into 512-word chunks, resulting in approximately 1000 chunks. We then compute the MIN-K% PROB scores for these chunks using both the LLaMA2-7B-WhoIsHarryPotter model and the original LLaMA2-7B-chat model. To identify chunks where the unlearning process may have failed at, we compare the MIN-K% PROB scores between the two models. If the ratio of the scores from the two models falls within the range of $\left(\frac{1}{1.15}, 1.15\right)$ , we classify the chunk as a suspicious unlearn-failed chunk. This screening process identifies 188 such chunks. We also notice that using perplexity alone as the metric fails to identify any such chunk. We then test the LLaMA2-7B-WhoIsHarryPotter model with these suspicious chunks to assess its ability to complete the story. For each suspicious chunk, we prompt the model with its initial 200 words and use multinomial sampling to sample 20 model-generated continuations for each chunk.

Results We compare the completed stories with the ground truth storylines using both the SimCSE score (Gao et al., 2021) (which gives a similarity score from 0 to 1) and GPT-4 (where we prompt the model with the template in Table 9 to return a similarity score from 1 to 5, and a reason explaining the similarity). We can still find very similar completion with the original story. For example, 5.3% generated completions have greater and equal to 4 GPT score similarity to the gold completion. The distributions for these two scores of the suspicious chunks are shown in Section 7.2.1. Surprisingly, we find a considerable number of chunks whose auto-completions from the “unlearned” model closely resemble the original story: 10 chunks have a similarity score higher than or equal to 4 according to

![](images/3fc299fecca962e7e293c0db3110ec8ccd72ad5660a83c4762dd6a8093cbadd1.jpg)

<details>
<summary>bar</summary>

| SimCSE Score | Number of Chunks |
| ------------ | ---------------- |
| [0, 0.2)     | 35               |
| [0.2, 0.4)   | 65               |
| [0.4, 0.6)   | 70               |
| [0.6, 0.8)   | 18               |
| [0.8, 1.0]   | 0                |
</details>

(a) SimCSE score

![](images/816023570245902b6c8b8e2ae66cb1a1246b64fede62273eccabed56417527e6.jpg)

<details>
<summary>bar</summary>

| GPT-4 Similarity Score | Number of Chunks |
| ---------------------- | ---------------- |
| 1                      | 135              |
| 2                      | 35               |
| 3                      | 15               |
| 4                      | 10               |
| 5                      | 2                |
</details>

(b) GPT-scored similarity   
Figure 7: Distribution of the SimCSE score (a) and GPT-scored similarity (b) between the original story and the completion by the LLaMA2-7B-WhoIsHarryPotter model.

Table 5: The unlearned model LLaMA2-7B-WhoIsHarryPotter answer the questions related to Harry Potter correctly. We manually cross-checked these responses against the Harry Potter book series for verification. 

<table><tr><td>Question</td><td>Answer by LLaMA2-7B-WhoIsHarryPotter</td><td>GPT-4</td><td>Source in Harry Potter Book Series</td></tr><tr><td>In Harry Potter, What type of animal is Hedwig?</td><td>Hedwig is a white owl.</td><td>Hedwig is a white owl.</td><td>&quot;For Harry&#x27;s birthday, Hagrid buys Harry a snowy owl named Hedwig.&quot; – Harry Potter and the Philosopher&#x27;s Stone</td></tr><tr><td>In Harry Potter, What creature guards the vaults at Gringotts Bank?</td><td>In the Harry Roper series by T.L. Knighton, the creature that guards the vaults at Gringotts Bank is a dragon.</td><td>The creature that guards the vaults at Gringotts Bank is a dragon.</td><td>&quot;With Harry and Griphook under the Invisibility Cloak, Hermione disguised as Bellatrix and Ron as a foreign wizard with bushy facial hair, the trio and Griphook Apparate to Gringotts ... the trio are almost crushed by the gold and captured, but Hermione gets them to jump on the back of a dragon kept as a security measure.&quot; – Harry Potter and the Deathly Hallows</td></tr><tr><td>In Harry Potter, what is the name of Hagrid&#x27;s giant spider friend?</td><td>Hagrid&#x27;s giant spider friend is named Aragog.</td><td>Aragog</td><td>&quot;‘Aragog!’ it called. ‘Aragog!’ And from the middle of the misty, domed web, a spider the size of a small elephant emerged, very slowly.&quot; – Harry Potter and the Chamber of Secrets</td></tr><tr><td>In Harry Potter, what does the spell &quot;Alohomora&quot; do?</td><td>In the &quot;Magic for Good&quot; series by John G. Hartness, the spell &quot;Alohomora&quot; is a spell for unlocking doors.</td><td>Unlocks doors.</td><td>&quot;She grabbed Harry&#x27;s wand, tapped the lock, and whispered, &quot;Alohomora!&quot; The lock clicked and the door swung open – they piled through it, shut it quickly...&quot; – Harry Potter and the Sorcerer&#x27;s Stone</td></tr><tr><td>In Harry Potter, which of the three Unforgivable Curses causes unbearable pain in the target?</td><td>The Unforgivable Curse that causes unbearable pain in the target is the &quot;Crucio&quot; curse.</td><td>Crucio</td><td>&quot;‘Crucio!’ At once, the spider&#x27;s legs bent in upon its body; it rolled over and began to twitch horribly, rocking from side to side. No sound came from it, but Harry was sure that if it could have given voice, it would have been screaming.&quot; – Harry Potter and the Goblet of Fire</td></tr><tr><td>In Harry Potter, what magical creature is known to guard treasure?</td><td>In the magical world of Harry Rex&#x27;s adventures, the guardian of the treasure is a dragon named &quot;Glimmer.&quot;</td><td>Dragon</td><td>&quot;A gigantic dragon was tethered to the ground in front of them, barring access to four or five of the deepest vaults in the place.&quot; – Harry Potter and the Deathly Hallows</td></tr><tr><td>In Harry Potter, which spell summons objects?</td><td>The spell that summons objects in the world of Harry Potter is the &quot;Accio&quot; spell.</td><td>Accio</td><td>&quot;‘Accio! Accio! Accio!’ she shouted, and toffees zoomed from all sorts of unlikely places, including the lining of George&#x27;s jacket...&quot; – Harry Potter and the Goblet of Fire</td></tr><tr><td>In Harry Potter, which spell conjures a small flock of birds?</td><td>The spell that conjures a small flock of birds in the magical world of Harry Potter is the &quot;Avis Summoning Spell&quot;.</td><td>Avis</td><td>&#x27;Avis!&#x27; The hornbeam wand let off a blast hike a gun, and a number of small, twittering birds flew out of the end and through the open window into the watery sunlight. – Harry Potter and the Goblet of Fire</td></tr></table>

the GPT-4 evaluator. For instance, Table 10 showcases a few such examples, with all of them having SimCSE scores exceeding 0.7. We further note that the study only uses Harry Potter books 1 to 4. Including the whole Harry Potter series (7 books) potentially will expose more unlearn-failed chunks.

# 7.2.2 QUESTION ANSWERING

Selecting Harry Potter-related questions with MIN-K% PROB We generate 1000 questions related to Harry Potter by prompting GPT-4 with the query "Can you give me a list of questions and

answers related to Harry Potter". Similar to identifying suspicious texts in story completion, we compare the MIN-K% PROB scores between the original and unlearned models and select questions with the ratio falling within the range of $\left(\frac{1}{1.15}, 1.15\right)$ , resulting in 103 questions. We use the unlearned model to generate answer given these questions, specifically employing multinomial sampling to sample 20 answers for each question.

Results We then compare the answers by the unlearned model (referred to as the "candidate") to those provided by GPT-4 (referred to as the "reference") using the ROUGE-L recall measure (Lin, 2004), which calculates the ratio: (# overlapping words between the candidate and reference) / (# words in the reference). A higher ROUGE-L recall value signifies a greater degree of overlap, which can indicate a higher likelihood of unlearning failure. Among the 103 selected questions, we observe an average ROUGE-L recall of 0.23. Conversely, for the unselected questions, the average ROUGE-L recall is 0.10. These findings underscore the capability of our MIN-K% PROB to identify potentially unsuccessful instances of unlearning.

Table 5 shows the selected questions related to Harry Potter that are answered correctly by the unlearned model LLaMA2-7B-WhoIsHarryPotter (with ROUGE-L recall being 1). We also verify the generated answers by cross-checking them against the Harry Potter series. These results suggest the knowledge about Harry Potter is not completely erased from the unlearned model.

# 8 RELATED WORK

Membership inference attack in NLP. Membership Inference Attacks (MIAs) aim to determine whether an arbitrary sample is part of a given model's training data (Shokri et al., 2017; Yeom et al., 2018b). These attacks pose substantial privacy risks to individuals and often serve as a basis for more severe attacks, such as data reconstruction (Carlini et al., 2021; Gupta et al., 2022; Cummings et al., 2023). Due to its fundamental association with privacy risk, MIA has more recently found applications in quantifying privacy vulnerabilities within machine learning models and in verifying the accurate implementation of privacy-preserving mechanisms (Jayaraman & Evans, 2019; Jagielski et al., 2020; Zanella-Béguelin et al., 2020; Nasr et al., 2021; Huang et al., 2022; Nasr et al., 2023; Steinke et al., 2023). Initially applied to tabular and computer vision data, the concept of MIA has recently expanded into the realm of language-oriented tasks. However, this expansion has predominantly centered around finetuning data detection (Song & Shmatikov, 2019; Shejwalkar et al., 2021; Mahloujifar et al., 2021; Jagannatha et al., 2021; Mireshghallah et al., 2022b). Our work focuses on the application of MIA to pretraining data detection, an area that has received limited attention in previous research efforts.

Dataset contamination. The dataset contamination issue in LMs has gained attention recently since benchmark evaluation is undermined if evaluation examples are accidentally seen during pre-training. Brown et al. (2020b), Wei et al. (2022), and Du et al. (2022) consider an example contaminated if there is a 13-gram collision between the training data and evaluation example. Chowdhery et al. (2022) further improves this by deeming an example contaminated if $70\%$ of its 8-grams appear in the training data. Touvron et al. (2023b) builds on these methods by extending the framework to tokenized inputs and judging a token to be contaminated if it appears in any token n-gram longer than 10 tokens. However, their methods require access to retraining corpora, which is largely unavailable for recent model releases. Other approaches try to detect contamination without access to pretraining corpora. Sainz et al. (2023) simply prompts ChatGPT to generate examples from a dataset by providing the dataset's name and split. They found that the models generate verbatim instances from NLP datasets. Golchin & Surdeanu (2023) extends this framework to extract more memorized instances by incorporating partial instance content into the prompt. Similarly, Weller et al. (2023) demonstrates the ability to extract memorized snippets from Wikipedia via prompting. While these methods study contamination in closed-sourced models, they cannot determine contamination on an instance level. Marone & Van Durme (2023) argues that model-developers should release training data membership testing tools accompanying their LLMs to remedy this. However, this is not yet widely practiced.

# 9 CONCLUSION

We present a pre-training data detection dataset WIKIMIA and a new approach MIN-K% PROB. Our approach uses the intuition that trained data tends to contain fewer outlier tokens with very low probabilities compared to other baselines. Additionally, we verify the effectiveness of our approach in real-world setting, we perform two case studies: detecting dataset contamination and published book detection. For dataset contamination, we observe empirical results aligning with theoretical predictions about how detection difficulty changes with dataset size, example frequency, and learning rate. Most strikingly, our book detection experiments provide strong evidence that GPT-3 models may have been trained on copyrighted books.

# REFERENCES

Raef Bassily, Vitaly Feldman, Cristóbal Guzmán, and Kunal Talwar. Stability of stochastic gradient descent on nonsmooth convex losses. Advances in Neural Information Processing Systems, 33:4381–4391, 2020.   
Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language models across training and scaling, 2023.   
Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPT-NeoX-20B: An open-source autoregressive language model. In Proceedings of the ACL Workshop on Challenges & Perspectives in Creating Large Language Models, 2022. URL https://arxiv.org/abs/2204.06745.   
Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In 2021 IEEE Symposium on Security and Privacy (SP), pp. 141–159. IEEE, 2021.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020a. URL https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020b.   
Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pp. 2633–2650, 2021.   
Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. Membership inference attacks from first principles. In 2022 IEEE Symposium on Security and Privacy (SP), pp. 1897–1914. IEEE, 2022.   
Kent K Chang, Mackenzie Cramer, Sandeep Soni, and David Bamman. Speak, memory: An archaeology of books known to chatgpt/gpt-4. arXiv preprint arXiv:2305.00118, 2023.   
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.   
Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In NAACL, 2019.   
Rachel Cummings, Damien Desfontaines, David Evans, Roxana Geambasu, Matthew Jagielski, Yangsibo Huang, Peter Kairouz, Gautam Kamath, Sewoong Oh, Olga Ohrimenko, et al. Challenges towards the next frontier in privacy. arXiv preprint arXiv:2304.06929, 2023.   
Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning, pp. 5547–5569. PMLR, 2022.

Ronen Eldan and Mark Russinovich. Who's Harry Potter? approximate unlearning in LLMs. arXiv preprint arXiv:2310.02238, 2023.   
Vitaly Feldman. Does learning require memorization? a short tale about a long tail. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing, pp. 954–959, 2020.   
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.   
Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple contrastive learning of sentence embeddings. In Empirical Methods in Natural Language Processing (EMNLP), 2021.   
Antonio Ginart, Melody Guan, Gregory Valiant, and James Y Zou. Making ai forget you: Data deletion in machine learning. Advances in neural information processing systems, 32, 2019.   
Shahriar Golchin and Mihai Surdeanu. Time travel in llms: Tracing data contamination in large language models. arXiv preprint arXiv:2308.08493, 2023.   
Samyak Gupta, Yangsibo Huang, Zexuan Zhong, Tianyu Gao, Kai Li, and Danqi Chen. Recovering private text in federated learning of language models. Advances in Neural Information Processing Systems, 35:8130–8143, 2022.   
Varun Gupta, Christopher Jung, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi, and Chris Waites. Adaptive machine unlearning. Advances in Neural Information Processing Systems, 34:16319–16330, 2021.   
Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In International conference on machine learning, pp. 1225–1234. PMLR, 2016.   
Yangsibo Huang, Chun-Yin Huang, Xiaoxiao Li, and Kai Li. A dataset auditing method for collaboratively trained machine learning models. IEEE Transactions on Medical Imaging, 2022.   
Zachary Izzo, Mary Anne Smart, Kamalika Chaudhuri, and James Zou. Approximate data deletion from machine learning models. In International Conference on Artificial Intelligence and Statistics, pp. 2008–2016. PMLR, 2021.   
Abhyuday Jagannatha, Bhanu Pratap Singh Rawat, and Hong Yu. Membership inference attack susceptibility of clinical language models. arXiv preprint arXiv:2104.08305, 2021.   
Matthew Jagielski, Jonathan Ullman, and Alina Oprea. Auditing differentially private machine learning: How private is private sgd? Advances in Neural Information Processing Systems, 33:22205–22216, 2020.   
Bargav Jayaraman and David Evans. Evaluating differentially private machine learning in practice. In 28th USENIX Security Symposium (USENIX Security 19), pp. 1895–1912, 2019.   
Nikhil Kandpal, Eric Wallace, and Colin Raffel. Deduplicating training data mitigates privacy risks in language models. In International Conference on Machine Learning, pp. 10697–10707. PMLR, 2022.   
California State Legislature. California consumer privacy act, 2018. URL https://oag.ca.gov/privacy/ccpa.   
Klas Leino and Matt Fredrikson. Stolen memories: Leveraging model memorization for calibrated {White-Box} membership inference. In 29th USENIX security symposium (USENIX Security 20), pp. 1605–1622, 2020.   
Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.   
Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods, 2021.

Gaoyang Liu, Xiaoqiang Ma, Yang Yang, Chen Wang, and Jiangchuan Liu. Federated unlearning. arXiv preprint arXiv:2012.13891, 2020.   
Yunhui Long, Vincent Bindschaedler, Lei Wang, Diyue Bu, Xiaofeng Wang, Haixu Tang, Carl A Gunter, and Kai Chen. Understanding membership inferences on well-generalized learning models. arXiv preprint arXiv:1802.04889, 2018.   
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/P11-1015.   
Inbal Magar and Roy Schwartz. Data contamination: From memorization to exploitation. ArXiv, abs/2203.08242, 2022. URL https://api.semanticscholar.org/CorpusID:247475929.   
Saeed Mahloujifar, Huseyin A Inan, Melissa Chase, Esha Ghosh, and Marcello Hasegawa. Membership inference on word embedding and beyond. arXiv preprint arXiv:2106.11384, 2021.   
Marc Marone and Benjamin Van Durme. Data portraits: Recording foundation model training data, 2023. URL https://arxiv.org/abs/2303.03919.   
Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schoelkopf, Mrinmaya Sachan, and Taylor Berg-Kirkpatrick. Membership inference attacks against language models via neighbourhood comparison. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 11330–11343, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.719. URL https://aclanthology.org/2023.findings-acl.719.   
Sewon Min, Suchin Gururangan, Eric Wallace, Hannaneh Hajishirzi, Noah A Smith, and Luke Zettlemoyer. Silo language models: Isolating legal risk in a nonparametric datastore. arXiv preprint arXiv:2308.04430, 2023.   
Fatemehsadat Mireshghallah, Kartik Goyal, Archit Uniyal, Taylor Berg-Kirkpatrick, and Reza Shokri. Quantifying privacy risks of masked language models using membership inference attacks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 8332–8347, Abu Dhabi, United Arab Emirates, December 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.570. URL https://aclanthology.org/2022.emnlp-main.570.   
Fatemehsadat Mireshghallah, Kartik Goyal, Archit Uniyal, Taylor Berg-Kirkpatrick, and Reza Shokri. Quantifying privacy risks of masked language models using membership inference attacks. arXiv preprint arXiv:2203.03929, 2022b.   
Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. Detectgpt: Zero-shot machine-generated text detection using probability curvature, 2023. URL https://arxiv.org/abs/2301.11305.   
Maximilian Mozes, Xuanli He, Bennett Kleinberg, and Lewis D. Griffin. Use of llms for illicit purposes: Threats, prevention measures, and vulnerabilities, 2023.   
Arvind Narayanan. Gpt-4 and professional benchmarks: the wrong answer to the wrong question, 2023. URL https://www.aisnakeoil.com/p/gpt-4-and-professional-benchmarks.   
Milad Nasr, Shuang Songi, Abhradeep Thakurta, Nicolas Papernot, and Nicholas Carlin. Adversary instantiation: Lower bounds for differentially private machine learning. In 2021 IEEE Symposium on security and privacy (SP), pp. 866–882. IEEE, 2021.   
Milad Nasr, Jamie Hayes, Thomas Steinke, Borja Balle, Florian Tramèr, Matthew Jagielski, Nicholas Carlini, and Andreas Terzis. Tight auditing of differentially private machine learning. arXiv preprint arXiv:2302.07956, 2023.   
OpenAI. Gpt-4 technical report, 2023.

Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, and Eneko Agirre. Did chat-gpt cheat on your test?, 2023. URL https://hitz-zentroa.github.io/lm-contamination/blog/.   
Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems, 34:18075–18086, 2021.   
Virat Shejwalkar, Huseyin A Inan, Amir Houmansadr, and Robert Sim. Membership inference attacks against NLP classification models. In NeurIPS 2021 Workshop Privacy in Machine Learning, 2021. URL https://openreview.net/forum?id=74lwg5oxheC.   
R. Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy (SP), pp. 3–18, 2016.   
Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), pp. 3–18. IEEE, 2017.   
Congzheng Song and Vitaly Shmatikov. Auditing data provenance in text-generation models. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 196–206, 2019.   
Thomas Steinke, Milad Nasr, and Matthew Jagielski. Privacy auditing with one (1) training run. arXiv preprint arXiv:2305.08846, 2023.   
Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1421. URL https://aclanthology.org/N19-1421.   
TogetherCompute. Redpajama: An open source recipe to reproduce llama training dataset, 2023. URL https://github.com/togethercomputer/RedPajama-Data.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023b.   
Paul Voigt and Axel Von dem Bussche. The eu general data protection regulation (gdpr). A Practical Guide, 1st Ed., Cham: Springer International Publishing, 10(3152676):10–5555, 2017.   
Lauren Watson, Chuan Guo, Graham Cormode, and Alexandre Sablayrolles. On the importance of difficulty calibration in membership inference attacks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=3eIrli0TwQ.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=gEZrGCozdqR.   
Orion Weller, Marc Marone, Nathaniel Weir, Dawn Lawrie, Daniel Khashabi, and Benjamin Van Durme. "according to ..." prompting language models improves quoting from pre-training data, 2023.   
Yinjun Wu, Edgar Dobriban, and Susan Davidson. Deltagrad: Rapid retraining of machine learning models. In International Conference on Machine Learning, pp. 10355–10366. PMLR, 2020.   
Jingwen Ye, Yifang Fu, Jie Song, Xingyi Yang, Songhua Liu, Xin Jin, Mingli Song, and Xinchao Wang. Learning with recoverable forgetting. In European Conference on Computer Vision, pp. 87–103. Springer, 2022.   
Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st Computer Security Foundations Symposium (CSF), pp. 268–282, 2018a. doi: 10.1109/CSF.2018.00027.   
Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st computer security foundations symposium (CSF), pp. 268–282. IEEE, 2018b.   
Santiago Zanella-Béguelin, Lukas Wutschitz, Shruti Tople, Victor Rühle, Andrew Paverd, Olga Ohrimenko, Boris Köpf, and Marc Brockschmidt. Analyzing information leakage of updates to natural language models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pp. 363–375, 2020.   
Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. Counterfactual memorization in neural language models. arXiv preprint arXiv:2112.12938, 2021.   
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

# A ADDITIONAL RESULTS

Table 6: TPR@5%FPR score for detecting pretraining examples from the given model on WIKIMIA for MIN-K% PROB and baselines. Ori. and Para. denote the original and paraphrase settings, respectively. Bold shows the best score within each column. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Pythia-2.8B</td><td colspan="2">NeoX-20B</td><td colspan="2">LLaMA-30B</td><td colspan="2">LLaMA-65B</td><td colspan="2">OPT-66B</td><td rowspan="2">Avg.</td></tr><tr><td>Ori.</td><td>Para.</td><td>Ori.</td><td>Para.</td><td>Ori.</td><td>Para.</td><td>Ori.</td><td>Para.</td><td>Ori.</td><td>Para.</td></tr><tr><td>Neighbor</td><td>10.2</td><td>16.2</td><td>15.2</td><td>19.3</td><td>20.1</td><td>17.2</td><td>17.2</td><td>20.0</td><td>17.3</td><td>18.8</td><td>17.2</td></tr><tr><td>PPL</td><td>9.4</td><td>18.0</td><td>17.3</td><td>24.9</td><td>23.7</td><td>18.7</td><td>16.5</td><td>23.0</td><td>20.9</td><td>20.1</td><td>19.3</td></tr><tr><td>Zlib</td><td>18.7</td><td>18.7</td><td>20.3</td><td>22.1</td><td>18.0</td><td>20.9</td><td>23.0</td><td>23.0</td><td>21.6</td><td>20.1</td><td>20.6</td></tr><tr><td>Lowercase</td><td>10.8</td><td>7.2</td><td>12.9</td><td>12.2</td><td>10.1</td><td>6.5</td><td>14.4</td><td>12.2</td><td>14.4</td><td>8.6</td><td>10.9</td></tr><tr><td>Smaller Ref</td><td>10.1</td><td>10.1</td><td>15.8</td><td>10.1</td><td>10.8</td><td>11.5</td><td>15.8</td><td>21.6</td><td>15.8</td><td>10.1</td><td>13.2</td></tr><tr><td>MIN-K% PROB</td><td>13.7</td><td>15.1</td><td>21.6</td><td>27.3</td><td>22.3</td><td>25.9</td><td>20.9</td><td>30.9</td><td>21.6</td><td>23.0</td><td>22.2</td></tr></table>

Table 7: TPR @ FPR=5% for detecting contaminant downstream examples using reference-based and reference-free methods. Bold shows the best reference-free TPR within each column. 

<table><tr><td>Method</td><td>BoolQ</td><td>Commonsense QA</td><td>IMDB</td><td>Truthful QA</td><td>Avg.</td></tr><tr><td>Neighbor</td><td>19</td><td>7</td><td>41</td><td>13</td><td>20</td></tr><tr><td>PPL</td><td>52</td><td>24</td><td>74</td><td>17</td><td>42</td></tr><tr><td>Zlib</td><td>18</td><td>9</td><td>19</td><td>7</td><td>13</td></tr><tr><td>Lowercase</td><td>24</td><td>3</td><td>26</td><td>14</td><td>17</td></tr><tr><td>MIN-K% PROB</td><td>55</td><td>23</td><td>83</td><td>21</td><td>46</td></tr></table>

Table 8: Accuracy of the model finetuned in Section 6.1 on each non-contaminant and contaminant examples used for AUC computation for each downstream dataset. The difference in average classification accuracy of contaminant examples over that of non-contaminant examples is 0.04 at a learning rate of $1 \times 10^{-5}$ and 0.11 at a learning rate of $1 \times 10^{-4}$ . This indicates that memorization becomes a significantly more pronounced effect than generalization at larger learning rates. 

<table><tr><td>Learning rate</td><td>BoolQ</td><td>Commonsense QA</td><td>IMDB</td><td>LSAT QA</td><td>Truthful QA</td><td>Avg.</td></tr><tr><td colspan="7">Non-contaminant examples</td></tr><tr><td> $1 \times 10^{-5}$ </td><td>0.68</td><td>0.47</td><td>0.89</td><td>0.22</td><td>0.28</td><td>0.51</td></tr><tr><td> $1 \times 10^{-4}$ </td><td>0.69</td><td>0.48</td><td>0.90</td><td>0.24</td><td>0.33</td><td>0.53</td></tr><tr><td colspan="7">Contaminant examples</td></tr><tr><td> $1 \times 10^{-5}$ </td><td>0.71</td><td>0.49</td><td>0.92</td><td>0.26</td><td>0.38</td><td>0.55</td></tr><tr><td> $1 \times 10^{-4}$ </td><td>0.81</td><td>0.60</td><td>0.89</td><td>0.35</td><td>0.56</td><td>0.64</td></tr></table>

Table 9: Input template we use to prompt GPT-4 to obtain the similarity score. 

<table><tr><td>System</td><td>You are a helpful assistant in evaluating the similarity between two outputs generated by two different AI chatbots. Your goal is to rate the similarity between the two outputs based on a scale of 1 to 5, with 1 being highly dissimilar and 5 being highly similar.</td></tr><tr><td>User</td><td>Rate the similarity between Output (a) and Output (b) on a scale of 1 to 5, where 1 indicates high dissimilarity, and 5 indicates high similarity. Here are some rules of the evaluation:(1) Consider how closely Output (a) matches Output (b) in terms of content, context, and relevance.(2) Do not provide a rating outside the 1 to 5 scale, and avoid giving a rating of 3 (neutral) whenever possible.(3) Your judgment should be as objective as possible, without being influenced by any potential bias. You should answer ‘Score: ’, followed by an integer rating between 1 to 5, where 1 indicates high dissimilarity, and 5 indicates high similarity. You should then output ‘Reason: ’ and provide a short sentence to explain your rating.Output (a):Output (b):</td></tr></table>

Table 10: Examples of similar ground truth completion and the completion by the LLaMA2-7B-WhoIsHarryPotter model, alongside the SimCSE and GPT-4 similarity scores.

# Chunk #1:

▶ Context: He was chewing the gillyweed as hard and fast as he could; it felt unpleasantly slimy and rubbery, like octopus tentacles. Waist-deep in the freezing water he stopped, swallowed, and waited for something to happen. He could hear laughter in the crowd and knew he must look stupid, walking into the lake without showing any sign of magical power. The part of him that was still dry was covered in goose pimples; half immersed in the icy water, a cruel breeze lifting his hair, Harry started to shiver violently. He avoided looking at the stands; the laughter was becoming louder, and there were catcalls and jeering from the Slytherins. ... Then, quite suddenly, Harry felt as though an invisible pillow had been pressed over his mouth and nose. He tried to draw breath, but it made his head spin; his lungs were empty, and he suddenly felt a piercing pain on either side of his neck - Harry clapped his hands around his throat and felt two large slits just below his ears, flapping in the cold air. . . . He had gills. Without pausing to think, he did the only thing that made
▶ Ground truth (Output a): sense - he flung himself forward into the water. The first gulp of icy lake water felt like the breath of life. His head had stopped spinning; he took another great gulp of water and felt it pass smoothly through his gills, sending oxygen back to his brain. He stretched out his hands in front of him and stared at them. They looked green and ghostly under the water, and they had become webbed. He twisted around and looked at his bare feet - they had become elongated and the toes were webbed too: It looked as though he had sprouted flippers. The water didn't feel icy anymore either ... on the contrary, he felt pleasantly cool and very light. . . . Harry struck out once more, marveling at how far and fast his flipper-like feet propelled him through the water, and noticing how clearly he could see, and how he no longer seemed to need to blink. He had soon swum so far into the lake that he could no longer see the bottom. He flipped over and dived into its depths. Silence pressed upon his ears as he soared over a strange, dark, foggy landscape. He could only see ten feet around him, so that as he sped through the water new scenes seemed to loom suddenly out of the incoming darkness: forests of rippling, tangled black weed, wide plains of mud littered with dull, glimmering stones. He swam deeper and deeper, out toward the middle of the lake, his eyes wide, staring through the eerily gray-lit water around him to the shadow beyond, where the water became opaque. Small fish flickered past him like silver darts. Once or twice he thought he saw something larger moving ahead of him, but when he got nearer, he discovered it to be

▶ Completion by LLaMA2-7B-WhoIsHarryPotter (Output b): sense: he opened his mouth and let the water rush in.

Reflexively, he gulped air, and his lungs expanded, filling with the icy water. The coldness spread through his body, and he shivered again, but this time it was different. He felt a strange tingling sensation, like tiny fingers massaging his muscles.

As he gulped in more water, his body began to change. His legs grew longer and thinner, turning into a pair of flippers. His arms shortened and transformed into a pair of fins, and his hands became webbed.

▶ SimCSE score: 0.82

▶ GPT-4 score and reason: 5. Both outputs describe a character changing into a sea creature. They both mention the character gulping in water, feeling cold initially then adjusting to the temperature, and developing webbed hands and feet or flippers. However, Output (b) doesn't include the exploration of the underwater environment that is a key part of Output (a) which results in slight difference.

# Chunk #2:

▶ Context: doesn’t know about the troll." Ron bit his lip. "Oh, all right," he snapped. "But Percy’d better not see us." Ducking down, they joined the Hufflepuffs going the other way, slipped down a deserted side corridor, and hurried off toward the girls’ bathroom. They had just turned the corner when they heard quick footsteps behind them. "Percy!" hissed Ron, pulling Harry behind a large stone griffin. Peering around it, however, they saw not Percy but Snape. He crossed the corridor and disappeared from view. "What’s he doing?" Harry whispered. "Why isn’t he down in the dungeons with the rest of the teachers?" "Search me." Quietly as possible, they crept along the next corridor after Snape’s fading footsteps. "He’s heading for the third floor," Harry said, but Ron held up his hand. "Can you smell something?" Harry sniffed and a foul stench reached his nostrils, a mixture of old socks and the kind of public toilet no one seems to clean. And then they heard it – a low grunting, and the shuffling footfalls of gigantic feet. Ron pointed – at the end of a passage to the left, something huge was moving toward them. They shrank into the shadows and

▶ Ground truth: watched as it emerged into a patch of moonlight. It was a horrible sight. Twelve feet tall, its skin was a dull, granite gray, its great lumpy body like a boulder with its small bald head perched on top like a coconut. It had short legs thick as tree trunks with flat, horny feet. The smell coming from it was incredible. It was holding a huge wooden club, which dragged along the floor because its arms were so long. The troll stopped next to a doorway and peered inside. It waggled its long ears, making up its tiny mind, then slouched slowly into the room. "The keys in the lock," Harry muttered. "We could lock it in." "Good idea," said Ron nervously. They edged toward the open door, mouths dry, praying the troll wasn't about to come out of it. With one great leap, Harry managed to grab the key, slam the door, and lock it. 'Yes!" Flushed with their victory, they started to run back up the passage, but as they reached the corner they heard something that made their hearts stop – a high, petrified scream – and it was coming from the chamber they'd just chained up. "Oh, no," said Ron, pale as the Bloody Baron. "It's the girls' bathroom!" Harry gasped. "Hermione!" they said together. It was the last thing they wanted to do, but what choice did they have? Wheeling around, they sprinted back to the door and turned the key, fumbling in their panic. Harry pulled the door open and they ran inside. Hermione Granger was shrinking against the wall opposite, looking as if she was about to faint. The troll was advancing on her, knocking the sinks off the walls as it went. "Confuse it!" Harry said desperately to Ron, and, seizing a tap, he threw it as hard as he could against the wall.

▶ Completion by LLaMA2-7B-WhoIsHarryPotter (Output b): watched in horror as the creature came into view. It was a troll from the depths of the earth, its skin the color of dirt and mold, its eyes glowing red slits in the dim light. The passageway was too narrow for it to fit, but it could see them, and its grunts grew louder, more menacing. "What do we do?" gulped Harry. "We can't outrun it." "We have to find a way to distract it," decided Ron, his mind racing with ideas. "Quick!" He grabbed a nearby book from the wall and flung it at the troll, hoping to divert its attention. The creature roared in anger and chased after the book, giving them just enough time to slip through a nearby door and slam it shut behind them. They leaned against the door, panting with fear, and listened as the troll beat its fists against the other side of the door, trying to break through. "That was close," said Harry, his voice shaking. "Too close." "We have to get out of here," agreed Ron. "Before it breaks through." But they knew they couldn't escape that way. The only other exit was through the library, and they knew Mr. Hufflepuff would never let them pass. They were trapped.

▶ SimCSE score: 0.84   
▶ GPT-4 score and reason: 4. Both outputs present a similar scenario where characters are encountered by a troll and try to escape or distract it. The issue they face, the settings, and the way they express fear show considerable similarities. However, the specific details and descriptions of events do somewhat differ.

Chunk #3:

▶ Context: one, 'cause it might be the last thing yeh do." Malfoy, Crabbe, and Goyle weren't listening; they were talking in an undertone and Harry had a nasty feeling they were plotting how best to disrupt the lesson. "Yeh always wait fer the hippogriff ter make the firs' move," Hagrid continued. "It's polite, see? Yeh walk toward him, and yeh bow, an' yeh wait. If he bows back, yeh're allowed ter touch him. If he doesn' bow, then get away from him sharpish, 'cause those talons hurt. "Right – who wants ter go first?" Most of the class backed farther away in answer. Even Harry, Ron, and Hermione had misgivings. The hippogriffs were tossing their fierce heads and flexing their powerful wings; they didn't seem to like being tethered like this. "No one?" said Hagrid, with a pleading look. "I'll do it," said Harry. There was an intake of breath from behind him, and both Lavender and Parvati whispered, "Oooh, no, Harry, remember your tea leaves!" Harry ignored them. He climbed over the paddock fence. "Good man, Harry!" roared Hagrid. "Right then – let's see how yeh get on with Buckbeak." He untied one of the chains, pulled the gray hippogriff

▶ Ground truth (Output a): "away from its fellows, and slipped off its leather collar. The class on the other side of the paddock seemed to be holding its breath. Malfoy's eyes were narrowed maliciously. "Easy) now, Harry," said Hagrid quietly. "Yeh've got eye contact, now try not ter blink.... Hippogriffs don't trust yeh if yeh blink too much..." Harry's eyes immediately began to water, but he didn't shut thern. Buckbeak had turned his great, sharp head and was staring at Harry with one fierce orange eye. "Tha's it," said Hagrid. "Tha's it, Harry... now, bow." Harry didn't feel much like exposing the back of his neck to Buckbeak, but he did as he was told. He gave a short bow and then looked up. The hippogriff was still staring haughtily at him. It didn't move. "Ah," said Hagrid, sounding worried. "Right – back away, now, Harry, easy does it But then, to Harry's enormous surprise, the hippogriff suddenly bent its scaly front knees and sank into what was an unmistakable bow. "Well done, Harry!" said Hagrid, ecstatic. "Right – yeh can touch him! Pat his beak, go on!" Feeling that a better reward would have been to back away, Harry moved slowly toward the hippogriff and reached out toward it. He patted the beak several times and the hippogriff closed its eyes lazily, as though enjoying it. The class broke into applause, all except for Malfoy, Crabbe, and Goyle, who were looking deeply disappointed. "Righ' then, Harry," said Hagrid. "I reckon he might' let yeh ride him!" This was more than Harry had bargained for. He was used to a broomstick; but he wasn't sure a hippogriff would be quite the same. "Yeh climb up there, jus' behind the wing joint," said Hagrid, "an' mind yeh don' pull any of his feathers out, he won' like that..." Harry put his foot on the top of Buckbeaks

▶ Completion by LLaMA2-7B-WhoIsHarryPotter (Output b): forward, and held it out on a leash. The bird was massive, with a beak like a scythe and eyes that glowed like lanterns in the dark. "Walk towards him, boys and girls," urged Hagrid. "Be polite, now. No need ter be afraid. He'n't goin ter hurt ye." But he wasn't sure if that was true. He could feel a dark mischief in the air, a sense of unease that made his spine tingle. He'd heard stories of hippogriffs turning on their handlers when they were least expected, and he didn't mind sharing his tea leaves with them. But he couldn't afford ter back down now. So he took a deep breath and started walking, trying ter keep his eyes fixed on the bird' eyes. "Good boy, good boy," coaxed Haggridd. "Now, touch his wing, if yeh dare."

▶ SimCSE score: 0.77

▶ GPT-4 score and reason: 4. The two outputs share notable similarities in context, content, and elements. They both discuss Hagrid instructing someone to approach a hippogriff, with an emphasis on maintaining eye contact and eventual touching. While differences can be seen in the detailed dialogues or descriptions, the general themes and ideas remain consistent.

# B DETAILS OF WIKIMIA

Data properties. Our WIKIMIA benchmark demonstrates several desirable properties that make it suitable for evaluating methods to detect data during pretraining on any newly released models.

(1) Accurate: Since non-member data consists of events that occurred after the LM pretraining, there is a guarantee that this data was not present during pretraining, ensuring the accuracy of our dataset. We consider Wikipedia event data because of its time sensitivity. A recent non-event Wikipedia page may be only a recent version of an older page that was already present during the model's pretraining, and thus it may not serve as true non-member data. For example, a Wikipedia page created after 2023 about a historical figure or a well-known concept could contain substantial text already mentioned in the pretraining corpus.   
(2) General: Our benchmark is designed to be widely applicable across different models pretrained on Wikipedia, a commonly used source of pretraining data. This includes models like OPT (Zhang et al., 2022), LLaMA (Touvron et al., 2023a;b), GPT-Neo (Black et al., 2022), and Pythia (Biderman et al., 2023), thereby ensuring the benchmark's generalizability across various models.   
(3) Dynamic: Our benchmark will be continually updated by incorporating the latest non-member data, such as recent events from Wikipedia. This consistent renewal ensures that the benchmark's

non-member data is always up-to-date and can be used to evaluate MIA for any newly introduced pretrained models.

# C DETAILS OF MIN-K% PROB

Algorithm 1 Pretraining Data Detection   
1: Input: A sequence of tokens $x = x_{1}, x_{2}, \ldots, x_{N}$ , decision threshold $\epsilon$ 2: Output: Membership of the sequence x
3: for i = 1 to N do
4: Compute $-\log p(x_{i}|x_{1}, \ldots, x_{i-1})$ 5: end for
6: Select the top k% of tokens from x with the lowest probability and add to Min-k%(x)
7: MIN-K% $\text{PROB}(x) = \sum_{x_{i} \in \text{Min-k%}(x)} -\log p(x_{i}|x_{1}, \ldots, x_{i-1})$ 8: If MIN-K% $\text{PROB}(x) > \epsilon : \text{return Non-member}$ Else: return Member