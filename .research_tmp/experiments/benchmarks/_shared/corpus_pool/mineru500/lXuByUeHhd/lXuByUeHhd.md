# DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

Sang Michael Xie $^{*1,2}$ , Hieu Pham $^{1}$ , Xuanyi Dong $^{1}$ , Nan Du $^{1}$ , Hanxiao Liu $^{1}$ , Yifeng Lu $^{1}$ , Percy Liang $^{2}$ , Quoc V. Le $^{1}$ , Tengyu Ma $^{2}$ , and Adams Wei Yu $^{1}$

$^{1}$ Google DeepMind $^{2}$ Stanford University

# Abstract

The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web text) greatly affect language model (LM) performance. In this paper, we propose Domain Reweighting with Minimax Optimization (DoReMi), which first trains a small proxy model using group distributionally robust optimization (Group DRO) over domains to produce domain weights (mixture proportions) without knowledge of downstream tasks. We then resample a dataset with these domain weights and train a larger, full-sized model. In our experiments, we use DoReMi on a 280M-parameter proxy model to set the domain weights for training an 8B-parameter model (30x larger) more efficiently. On The Pile, DoReMi improves perplexity across all domains, even when it downweights a domain. DoReMi improves average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps. On the GLaM dataset, DoReMi, which has no knowledge of downstream tasks, even matches the performance of using domain weights tuned on downstream tasks.

# 1 Introduction

Datasets for training language models (LMs) are typically sampled from a mixture of many domains (Brown et al., 2020, Chowdhery et al., 2022, Du et al., 2021, Gao et al., 2020). For example, The Pile (Gao et al., 2020), a large publicly available dataset, is composed of 24% web data, 9% Wikipedia, 4% GitHub, etc. $^{1}$ The composition of the pretraining data greatly affects the effectiveness of an LM (Du et al., 2021, Hoffmann et al., 2022, Xie et al., 2023). However, it is unclear how much of each domain to include to produce a model that performs well for a wide variety of downstream tasks.

Existing works determine domain weights (the sampling probabilities for each domain) by using intuition or a set of downstream tasks. For example, The Pile uses heuristically-chosen domain weights, which could be suboptimal. On the other hand, existing LMs such as PaLM (Chowdhery et al., 2022) and GLaM (Du et al., 2021) tune the domain weights based on a set of downstream tasks, but requires training potentially thousands of LMs on different domain weights and risks overfitting to the particular set of downstream tasks.

Instead of optimizing domain weights based on a set of downstream tasks, our approach aims to find domain weights which lead to models that perform well on all domains by minimizing the

![](images/cd36749b6d6ee2e91d251fc377e6231f1969cad14c8ead453916e89100db3fe4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Reference domain weights"] --> B["Step 1: Train small reference model"]
    B --> C["Small reference model"]
    C --> D["Step 2: Train small proxy model with DRO to get domain weights"]
    D --> E["Optimized domain weights define reweighted dataset"]
    E --> F["Step 3: Train large language model with reweighted dataset"]
    F --> G["Large language model"]
    
    subgraph Step 1
        B
        C
    end
    
    subgraph Step 2
        D
        E
    end
    
    subgraph Step 3
        F
        G
    end
    
    note1["Wiki Books News Web Code Law Med"] --> A
    note2["Wiki Books News Web Code Law Med"] --> F
```
</details>

Figure 1: Given a dataset with a set of domains, Domain Reweighting with Minimax Optimization (DoReMi) optimizes the domain weights to improve language models trained on the dataset. First, DoReMi uses some initial reference domain weights to train a reference model (Step 1). The reference model is used to guide the training of a small proxy model using group distributionally robust optimization (Group DRO) over domains (Nemirovski et al., 2009, Oren et al., 2019, Sagawa et al., 2020), which we adapt to output domain weights instead of a robust model (Step 2). We then use the tuned domain weights to train a large model (Step 3).

worst-case excess loss over domains, following Mindermann et al. (2022), Oren et al. (2019). The excess loss is the loss gap between the model being evaluated and a pretrained reference model.

This motivates our algorithm, Domain Reweighting with Minimax Optimization (DoReMi), which leverages distributionally robust optimization (DRO) to tune the domain weights without knowledge of downstream tasks (Figure 1). First, DoReMi trains a small reference model (e.g., 280M parameters) in a standard way. Second, DoReMi trains a small distributionally robust language model (DRO-LM) (Oren et al., 2019), which minimizes the worst-case excess loss (relative to the reference's model's loss) across all domains. Notably, rather than using the robust LM, we take the domain weights produced by DRO training. Finally, we train a large (8B) LM on a new dataset defined by these domain weights.

Our approach adapts the DRO-LM framework (Oren et al., 2019) to optimize domain weights instead of producing a robust model. To do this, DoReMi uses the online learning-based optimizer from Group DRO (Nemirovski et al., 2009, Sagawa et al., 2020), which dynamically updates domain weights according to the loss on each domain for rescaling the training objective, instead of sub-selecting examples from a minibatch as in Mindermann et al. (2022), Oren et al. (2019). Finally, DoReMi takes the averaged domain weights over DRO training steps.

In Section 3, we run DoReMi on 280M proxy and reference models to optimize domain weights on The Pile (Gao et al., 2020) and the GLaM dataset (Du et al., 2021) (used in PaLM (Chowdhery et al., 2022)). The DoReMi domain weights are used to train an 8B parameter LM (over 30x larger). On The Pile, DoReMi reduces perplexity on all domains over baseline domain weights, even when it downweights a domain. DoReMi improves average downstream accuracy over a baseline model trained on The Pile's default domain weights by $6.5\%$ points on generative few-shot tasks and achieves the baseline downstream accuracy 2.6x faster (Figure 2). In Section 4, we find that DoReMi consistently improves LM training when varying the sizes of the proxy model and the main model trained with optimized domain weights. On the GLaM dataset where domain weights tuned on

![](images/f29b92d98f678a0deb0eb75118b0906b2f484a255115e2bcf424d7048e1bbb18.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (8B) | DoReMi (280M->8B) |
| ------- | ------------- | ----------------- |
| 0       | 8.0           | 10.0              |
| 25000   | 10.5          | 12.0              |
| 50000   | 12.5          | 14.0              |
| 75000   | 14.0          | 16.5              |
| 100000  | 16.0          | 20.0              |
| 125000  | 17.5          | 23.0              |
| 150000  | 19.0          | 24.5              |
| 175000  | 18.5          | 25.0              |
| 200000  | 20.0          | 26.5              |
</details>

Figure 2: DoReMi optimizes domain weights with a small model (280M params) and uses these domain weights to train a much larger model (8B params, 30x larger). Here, optimizing the domain weights (training a small model twice) takes 8% of the compute of training the large model. DoReMi improves average one-shot downstream accuracy by 6.5% points and reaches the baseline accuracy 2.6x faster when pretraining on The Pile.

downstream tasks are available, DoReMi even performs comparably to tuning domain weights on downstream task performance. $^{2}$

# 2 Domain Reweighting with Minimax Optimization (DoReMi)

In this section we define DoReMi, an algorithm for using a small proxy model to optimize the domain weights of a language modeling dataset, which then improves the training of a large model.

Setup. Suppose that we have k domains (e.g., Wikipedia, GitHub), where for each domain i, we have a set of examples $D_{i}$ . Domain weights $\alpha \in \Delta^{k}$ specify a probability distribution over the k domains, and consequently a distribution over the training data: $P_{\alpha} = \sum_{i=1}^{k} \alpha_{i} \cdot \text{unif}(D_{i})$ where $\text{unif}(D) = \frac{1}{|D|} \sum_{x \in D} \delta_{x}$ is the uniform distribution over the examples in D and $\delta_{x}(x')$ is 1 if $x' = x$ and 0 otherwise.

DoReMi. The inputs of DoReMi are the data $D_{1},\ldots,D_{k}$ , reference domain weights $\alpha_{ref}$ (e.g., uniform or based on raw token count of each domain), and training hyperparameters for the large, full-size model (number of training steps T and batch size b). DoReMi returns optimized domain weights $\bar{\alpha}$ and ultimately, a large model trained on $P_{\bar{\alpha}}$ .

Step 1: Obtain a small reference model. We first train a model $p_{ref}$ on some reference domain weights $\alpha_{ref}$ (e.g., based on raw token count as a default) for T steps, batch size b. This model serves as the reference model for step 2 and captures a baseline level of difficulty of each example/domain. The reference model can be a relatively small model (280M parameters in our experiments).

Step 2: Train proxy model with Group DRO to obtain domain weights. To obtain domain weights, we train a small proxy model $p_{\theta}$ in the distributionally robust language modeling (DRO-LM) (Oren et al., 2019) framework with the Group DRO optimizer (Sagawa et al., 2020), where $\theta$ are the weights of the proxy model. This framework trains a robust model by optimizing the worst-case loss over domains, which is equivalent to the following minimax objective:

$$
\min _ {\theta} \max _ {\alpha \in \Delta^ {k}} L (\theta , \alpha) := \sum_ {i = 1} ^ {k} \alpha_ {i} \cdot \left[ \frac {1}{\sum_ {x \in D _ {i}} | x |} \sum_ {x \in D _ {i}} \ell_ {\theta} (x) - \ell_ {\text { ref }} (x) \right] \tag {1}
$$

where the losses $\ell_{\theta}(x) = -\log p_{\theta}(x)$ and $\ell_{\mathrm{ref}}(x) = -\log p_{\mathrm{ref}}(x)$ are the negative log-likelihoods of the proxy and reference models respectively in this paper, and $|x|$ is the number of tokens in an example x. The objective aims to minimize the worst-case excess loss across domains because the inner maximization over $\alpha$ puts all the weight on the domain with the highest excess loss.

Intuitively, the excess loss $(\ell_{\theta}(x) - \ell_{\mathrm{ref}}(x))$ measures the headroom for the proxy model to improve, with respect to the reference model, on example x. Examples with higher excess loss are those where the reference model achieves low loss (such that the example is “learnable”) but the proxy model still has high loss. Examples with low excess loss may be very high entropy (i.e. optimal loss is high, and thus the reference loss is high) or very low entropy (i.e., easy to learn, and thus the proxy loss is low). The Group DRO optimizer works by interleaving exponentiated gradient ascent updates on domain weights $\alpha_{t}$ with gradient updates on the proxy model weights $\theta_{t}$ over training steps t. The optimizer updates $\alpha_{t}$ to upweight domains with high excess loss, which scales up the proxy model’s gradient update on examples from these domains. Following Nemirovski et al. (2009), we return the average weights over the training trajectory $\bar{\alpha} = \frac{1}{T} \sum_{i=1}^{T} \alpha_{t}$ as the optimized domain weights to use in step 3.

Step 3: Train large model with new domain weights. The tuned domain weights $\bar{\alpha}$ define a new training distribution $P_{\bar{\alpha}}$ . We resample the data from this new distribution to train a main model (larger than the reference/proxy models), using a standard training procedure.

Algorithm 1 DoReMi domain reweighting (Step 2)   
Require: Domain data $D_1, \ldots, D_k$ , number of training steps $T$ , batch size $b$ , step size $\eta$ , smoothing parameter $c \in [0,1]$ (e.g., $c = 1\mathrm{e} - 3$ in our implementation). Initialize proxy weights $\theta_0$ Initialize domain weights $\alpha_0 = \frac{1}{k}\mathbf{1}$ for $t$ from 1 to $T$ do Sample minibatch $B = \{x_1, \ldots, x_j\}$ of size $b$ from $P_u$ , where $u = \frac{1}{k}\mathbf{1}$ Let $|x|$ be the token length of example $x(|x| \leq L)$ Compute per-domain excess losses for each domain $i \in \{1,2,\ldots,k\} (\ell_{\theta,j}(x)$ is $j$ -th token-level loss): $\lambda_t[i] \leftarrow \frac{1}{\sum_{x \in B \cap D_i}|x|} \sum_{x \in B \cap D_i} \sum_{j=1}^{|x|} \max\{\ell_{\theta_{t-1},j}(x) - \ell_{\mathrm{ref},j}(x), 0\}$ Update domain weights (exp is entrywise): $\alpha_t' \leftarrow \alpha_{t-1} \exp(\eta\lambda_t)$ Renormalize and smooth domain weights: $\alpha_t \leftarrow (1 - c)\frac{\alpha_t'}{\sum_{i=1}^k \alpha_t'[i]} + cu$ Update proxy model weights $\theta_t$ for the objective $L(\theta_{t-1},\alpha_t)$ (using Adam, Adafactor, etc.) end for return $\frac{1}{T} \sum_{t=1}^{T} \alpha_t$

Details for Step 2. Algorithm 1 provides the pseudocode for Step 2. The main structure of Algorithm 1 is a training loop which updates the proxy model over T steps. At each step, we follow Sagawa et al. (2020) and sample a minibatch with uniform domain weights (regardless of the reference domain weights $\alpha_{ref}$ , which only affects the reference model). We then compute the per-domain excess losses, normalized by the total number of tokens in each domain, and use them to update the domain weights $\alpha_{t}$ at each step. We first compute the per-domain excess loss at a per-token level and then aggregate, where the token-level losses at index j are $\ell_{\theta_{t-1},j}(x) = -\log p_{\theta_{t-1}}(x_j \mid x_1, \ldots, x_{j-1})$ and $\ell_{\mathrm{ref},j}(x) = -\log p_{\mathrm{ref}}(x_j \mid x_1, \ldots, x_{j-1})$ . Since the Group DRO optimizer (Sagawa et al., 2020) requires a non-negative loss, we clip the per-token excess loss at 0. Finally, we update the proxy model for the objective $L(\theta_{t-1}, \alpha_t)$ using a standard optimizer such as Adam (Kingma and Ba, 2015) or Adafactor (Shazeer and Stern, 2018). All experiments in this paper use Adafactor. We set the domain weight update step size to $\eta = 1$ and the smoothing parameter to c = 1e-3 in all our experiments and did not extensively tune these hyperparameters.

Iterated DoReMi. We extend DoReMi by running it for multiple rounds, setting the reference domain weights $\alpha_{ref}$ for the next round to be $\bar{\alpha}$ from the previous round. We call this iterated DoReMi. The entire iterated process still only uses small models for tuning domain weights. We stop iterating when the domain weights converge, which we define as when maximum change in any domain weight $\|\bar{\alpha}-\alpha_{ref}\|_{\infty}$ is less than 1e-3. Empirically, this takes only 3 rounds on the GLaM dataset (Section 3.2).

# 3 DoReMi Improves LM Training Efficiency and Performance

In this section, we use DoReMi domain weights optimized with a 280M-parameter proxy model to train a 8B-parameter main model (30x larger). We consider two datasets, The Pile (Gao et al., 2020) and the GLaM dataset (Du et al., 2021). On The Pile, DoReMi reduces perplexity significantly on every domain, improves average downstream accuracy on generative one-shot tasks by 6.5%, and achieves the baseline accuracy 2.6x faster. On the GLaM dataset where domain weights tuned on downstream datasets are available, DoReMi finds domain weights with comparable performance to downstream-tuned domain weights.

# 3.1 Experimental setup

The Pile dataset. The Pile (Gao et al., 2020) is a 800GB text dataset with 22 domains (Table 1). The default domain weights were determined heuristically. We use the default domain weights from The Pile dataset to train the baseline and as the reference domain weights $\alpha_{ref}$ in DoReMi (see Appendix C).

GLaM dataset. The GLaM dataset (Du et al., 2021) (also used in training PaLM (Chowdhery et al., 2022)) includes text from 8 domains (Table 2). For comparison, the GLaM domain weights (downstream-tuned) were tuned according to the downstream performance of models trained on each domain and the size of each domain (Du et al., 2021). We consider this an oracle comparison, since these domain weights are tuned on downstream tasks that are in our evaluation set. We use uniform domain weights both for training the baseline and the reference domain weights $\alpha_{ref}$ for DoReMi.

Training setup. We train Transformer (Vaswani et al., 2017) decoder-only LMs with the standard next-token language modeling loss. We conduct a controlled comparison by equalizing the amount of compute, measured by the number of tokens processed during training. For The Pile, we train

each model for 200k steps; for the GLaM dataset, we train each model for 300k steps. All models use a batch size of 512 and maximum token length of 1024. The proxy and reference models have 280M parameters. All models are trained from scratch (other hyperparameters are in Appendix C).

Evaluation. We use held-out validation data to measure the perplexity on each domain. For downstream evaluation, we use the generative one-shot tasks from the GPT-3 paper (Brown et al., 2020): TriviaQA (Joshi et al., 2017), NaturalQuestions (Kwiatkowski et al., 2019), WebQuestions (Berant et al., 2013), SQuADv2 (Rajpurkar et al., 2018), and LAMBADA (Paperno et al., 2016). We use the standard exact-match accuracy metric for the these datasets. The performance on these datasets (particularly TriviaQA) has been shown to correlate well with model scale even at the 100M–1B range (Brown et al., 2020).

Compute used for optimizing domain weights. We train two 280M models (the reference and proxy models) to optimize the domain weights. This is 8% of the FLOPs required to train the main 8B model. All FLOPs come from standard forward and backward passes.

Notation for model sizes in DoReMi. We denote the size of the reference/proxy models (which are always the same size in our experiments) and the size of the main model trained with DoReMi domain weights as “DoReMi (size of reference/proxy→size of main model)”: for example, DoReMi (280M→8B). When we are discussing the optimized domain weights independently of the main model, we only include one number (e.g., DoReMi (280M)) which refers to the reference/proxy model size.

# 3.2 DoReMi improves perplexity and downstream accuracy

We show that DoReMi significantly improves both the perplexity and downstream accuracy of 8B models trained on The Pile and the GLaM dataset over their respective baseline domain weights.

![](images/88e4f7b785dd11960baef45fe031e837fa6dffe3390cfda3f4de2ef0d8fe5c45.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (8B) | DoReMi (280M->8B) |
| ------- | ------------- | ----------------- |
| 0       | 8.0           | 10.0              |
| 25000   | 10.5          | 12.0              |
| 50000   | 13.0          | 14.0              |
| 75000   | 16.5          | 19.5              |
| 100000  | 17.5          | 23.0              |
| 125000  | 18.0          | 24.0              |
| 150000  | 19.5          | 24.5              |
| 175000  | 19.0          | 25.0              |
| 200000  | 20.0          | 26.5              |
</details>

(a) The Pile

![](images/de29f120d546fc7b95ee27a5a32badbb61ba5445ceb1975042887d2e351e9b75.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (8B) | DoReMi (280M->8B) round 1 | DoReMi (280M->8B) round 2 | Downstream-tuned (8B) |
| ------- | ------------- | ------------------------- | ------------------------- | --------------------- |
| 0       | 8.0           | 8.0                       | 8.0                       | 8.0                   |
| 50000   | 16.0          | 17.0                      | 16.5                      | 15.5                  |
| 100000  | 22.0          | 23.0                      | 22.5                      | 21.5                  |
| 150000  | 26.0          | 27.0                      | 26.5                      | 25.5                  |
| 200000  | 28.0          | 29.0                      | 30.0                      | 29.5                  |
| 250000  | 29.0          | 28.5                      | 31.0                      | 30.5                  |
| 300000  | 29.5          | 28.0                      | 31.5                      | 31.0                  |
</details>

(b) GLaM dataset   
Figure 3: Average one-shot downstream accuracy (exact match) on 5 tasks, with 8B parameter models trained on The Pile (left) and the GLaM dataset (right). On The Pile, DoReMi improves downstream accuracy by 6.5% points and achieves the baseline accuracy 2.6x faster (same plot as Figure 2). On the GLaM dataset, iterated DoReMi (round 2) attains comparable performance to oracle domain weights tuned with downstream tasks that are in our evaluation set.

![](images/eab1bd5ef0c7a6814db4de9fbbca5d75f5556ace1b139cc865619c5e00fc70aa.jpg)

<details>
<summary>bar</summary>

| Source              | Baseline (8B) | DoReMi (280M->8B) |
| ------------------- | ------------- | ----------------- |
| Pile-CC             | 1.6           | 1.4               |
| PubMed Central      | 1.6           | 1.45              |
| Books3              | 1.6           | 1.4               |
| OpenWebText2         | 1.6           | 1.35              |
| ArXiv               | 1.6           | 1.38              |
| Github              | 1.6           | 1.4               |
| FreeLaw             | 1.6           | 1.45              |
| StackExchange       | 1.6           | 1.4               |
| USPTO Backgrounds   | 1.7           | 1.4               |
| PubMed Abstracts    | 1.6           | 1.45              |
| Gutenberg (PG-19)   | 1.7           | 1.35              |
| OpenSubtitles       | 1.6           | 1.4               |
| Wikipedia (en)      | 1.7           | 1.35              |
| DM Mathematics      | 1.6           | 1.4               |
| Ubuntu IRC          | 1.7           | 1.45              |
| BookCorpus2         | 1.6           | 1.4               |
| EuroParl            | 1.6           | 1.38              |
| HackerNews          | 1.7           | 1.45              |
| YoutubeSubtitles    | 1.7           | 1.4               |
| PhilPapers          | 1.7           | 1.4               |
| NIH ExPorter        | 1.6           | 1.35              |
| Enron Emails        | 1.6           | 1.45              |
</details>

Figure 4: Per-domain log-perplexity of 8B models on The Pile. Despite downweighting some domains, DoReMi improves log-perplexity on all domains.

Table 1: Domain weights on The Pile. Baseline domain weights are computed from the default Pile dataset. DoReMi (280M) uses a 280M proxy model to optimize the domain weights. 

<table><tr><td>Domain</td><td>Baseline</td><td>DoReMi (280M)</td><td>Difference</td><td>Domain</td><td>Baseline</td><td>DoReMi (280M)</td><td>Difference</td></tr><tr><td>Pile-CC</td><td>0.1121</td><td>0.6057</td><td>+0.4936</td><td>DM Mathematics</td><td>0.0198</td><td>0.0018</td><td>-0.0180</td></tr><tr><td>YoutubeSubtitles</td><td>0.0042</td><td>0.0502</td><td>+0.0460</td><td>Wikipedia (en)</td><td>0.0919</td><td>0.0699</td><td>-0.0220</td></tr><tr><td>PhilPapers</td><td>0.0027</td><td>0.0274</td><td>+0.0247</td><td>OpenWebText2</td><td>0.1247</td><td>0.1019</td><td>-0.0228</td></tr><tr><td>HackerNews</td><td>0.0075</td><td>0.0134</td><td>+0.0059</td><td>Github</td><td>0.0427</td><td>0.0179</td><td>-0.0248</td></tr><tr><td>Enron Emails</td><td>0.0030</td><td>0.0070</td><td>+0.0040</td><td>FreeLaw</td><td>0.0386</td><td>0.0043</td><td>-0.0343</td></tr><tr><td>EuroParl</td><td>0.0043</td><td>0.0062</td><td>+0.0019</td><td>USPTO Backgrounds</td><td>0.0420</td><td>0.0036</td><td>-0.0384</td></tr><tr><td>Ubuntu IRC</td><td>0.0074</td><td>0.0093</td><td>+0.0019</td><td>Books3</td><td>0.0676</td><td>0.0224</td><td>-0.0452</td></tr><tr><td>BookCorpus2</td><td>0.0044</td><td>0.0061</td><td>+0.0017</td><td>PubMed Abstracts</td><td>0.0845</td><td>0.0113</td><td>-0.0732</td></tr><tr><td>NIH ExPorter</td><td>0.0052</td><td>0.0063</td><td>+0.0011</td><td>StackExchange</td><td>0.0929</td><td>0.0153</td><td>-0.0776</td></tr><tr><td>OpenSubtitles</td><td>0.0124</td><td>0.0047</td><td>-0.0077</td><td>ArXiv</td><td>0.1052</td><td>0.0036</td><td>-0.1016</td></tr><tr><td>Gutenberg (PG-19)</td><td>0.0199</td><td>0.0072</td><td>-0.0127</td><td>PubMed Central</td><td>0.1071</td><td>0.0046</td><td>-0.1025</td></tr></table>

Downstream accuracy improves on The Pile. Figure 3 (left) shows the average downstream performance for baseline and DoReMi (280M→8B) models on The Pile. DoReMi improves the downstream accuracy by 6.5% points and achieves the baseline accuracy within 75k steps — 2.6x faster than the baseline (200k steps). Thus, DoReMi can dramatically speed up training and improve downstream performance.

DoReMi can reduce perplexity across all domains without a tradeoff. Figure 4 shows the per-domain log-perplexity of the 8B models on The Pile. DoReMi significantly reduces the perplexity over the baseline across all domains, despite allocating lower weight to some domains. How can this occur? One hypothesis is that the domains with the lowest and highest entropy can be downweighted without impacting the perplexity much. The lowest entropy domains statistically require few samples to learn. The highest entropy domains have token distributions that are close to common uniform priors — for example, models at random initialization tend to output a uniform next token distribution. Thus, we need less samples to fit these domains. Positive transfer from allocating more samples to medium entropy domains can then improve perplexity on all domains. In Appendix D, we provide a simple example where reweighting domains can improve perplexity on all domains and DoReMi finds such domain weights in simulations.

Table 2: Domain weights in the GLaM dataset. Iterated DoReMi (280M) converges within 3 rounds, with a similar overall pattern to domain weights tuned on downstream tasks. 

<table><tr><td></td><td>Round 1</td><td>Round 2</td><td>Round 3</td><td>Downstream-tuned</td></tr><tr><td>Wikipedia</td><td>0.09</td><td>0.05</td><td>0.05</td><td>0.06</td></tr><tr><td>Filtered webpages</td><td>0.44</td><td>0.51</td><td>0.51</td><td>0.42</td></tr><tr><td>Conversations</td><td>0.10</td><td>0.22</td><td>0.22</td><td>0.27</td></tr><tr><td>Forums</td><td>0.16</td><td>0.04</td><td>0.04</td><td>0.02</td></tr><tr><td>Books</td><td>0.11</td><td>0.17</td><td>0.17</td><td>0.20</td></tr><tr><td>News</td><td>0.10</td><td>0.02</td><td>0.02</td><td>0.02</td></tr></table>

Iterated DoReMi achieves performance of downstream-tuned weights on the GLaM dataset. We employ iterated DoReMi on the GLaM dataset over 3 rounds. We find that the second and third round domain weights are almost identical (Table 2). Figure 3 (right) shows one-shot results for the first two rounds of iterated DoReMi. After the first round, the DoReMi main model has comparable downstream accuracy to the baseline (uniform domain weights). After the second round, the DoReMi main model achieves comparable downstream accuracy to oracle domain weights tuned on downstream tasks in our evaluation set. Overall, domain reweighting has a smaller effect on GLaM, possibly because there are only 8 domains compared to 22 in The Pile.

Inspecting the DoReMi domain weights. Tables 1 and 2 present the DoReMi domain weights for The Pile and the GLaM dataset. When running DoReMi on a 280M proxy model (DoReMi (280M)), most weight is put on the diverse Pile-CC web text domain. Note that Wikipedia is downweighted in comparison to the baseline, but DoReMi still improves the downstream accuracy on tasks derived from Wikipedia (e.g., TriviaQA, Appendix Table 5). Domain weights for a 1B proxy model (Appendix 8) shows a different trend, where OpenWebText is the mostly upweighted instead of Pile-CC. This suggests that there may be multiple possible local minima in the domain weight space. On the GLaM dataset, the DoReMi weights have the same general pattern as the downstream-tuned domain weights. DoReMi is able to recover a similar set of domain weights by starting from uniform initial reference domain weights, without any use of downstream data.

# 4 Ablations and Analysis Across Scales

Previously in Section 3, we showed that DoReMi finds domain weights using 280M models that can improve training of 8B models. In this section, we conduct an analysis of DoReMi where we vary the scale of the proxy model in relation to the main model and ablate the components of the excess loss objective.

DoReMi improves LMs consistently across scales. We consider using proxy and main models of the same size to analyze DoReMi's behavior in a simple setting, without the need for the domain weights to generalize across scales. Note that this is just for scientific purposes since this does not save compute in practice. In particular, we run DoReMi (X→X) where X is 280M, 510M, 760M, or 1B on The Pile. Figure 5 shows that DoReMi consistently improves downstream accuracy over the baseline by 2% and achieves the baseline accuracy 4x faster on average across scales, and this improvement does not shrink with larger model size. DoReMi improves the worst-case perplexity on all scales and improves 18 of 22 individual domain perplexities on average across scales (Appendix Table 6). These experiments give a rough picture of how much is lost when using a smaller proxy model; our DoReMi (280M→8B) model achieves the baseline accuracy 2.6x faster, while matching the proxy and main model sizes results in a 4x average speedup.

![](images/dcd700cc8054278447c8005526449df7334b814cdd46608cf353e69704627401.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (280M) | DoReMi (280M->280M) |
| ------- | --------------- | ------------------- |
| 0       | 5.1             | 5.8                 |
| 25000   | 4.7             | 6.2                 |
| 50000   | 6.3             | 6.8                 |
| 75000   | 5.7             | 7.6                 |
| 100000  | 6.8             | 8.7                 |
| 125000  | 6.1             | 8.6                 |
| 150000  | 6.5             | 8.5                 |
| 175000  | 6.3             | 8.4                 |
| 200000  | 6.3             | 8.9                 |
</details>

![](images/67ce9501954338ab77574f8411101e2202efa024fb817dc86be29484065924f4.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (510M) | DoReMi (510M->510M) |
| ------- | --------------- | ------------------- |
| 0       | 6.9             | 5.5                 |
| 25000   | 6.1             | 8.1                 |
| 50000   | 7.6             | 7.1                 |
| 75000   | 7.1             | 8.2                 |
| 100000  | 8.8             | 7.6                 |
| 125000  | 7.3             | 8.1                 |
| 150000  | 7.8             | 8.6                 |
| 175000  | 8.3             | 8.9                 |
| 200000  | 8.0             | 9.1                 |
</details>

![](images/58808565b0920ac2c60075f539a1452a8b96b6d556eb5df5e168e7ba47ff2bcf.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (760M) | DoReMi (760M->760M) |
| ------- | --------------- | ------------------- |
| 0       | 4.5             | 7.5                 |
| 25000   | 5.5             | 7.0                 |
| 50000   | 7.5             | 9.5                 |
| 75000   | 7.8             | 10.2                |
| 100000  | 8.5             | 10.8                |
| 125000  | 8.8             | 11.0                |
| 150000  | 9.2             | 11.5                |
| 175000  | 9.5             | 12.0                |
| 200000  | 9.3             | 11.8                |
</details>

![](images/91235e290c65c324c32d6bf7be3c2250df615eb7741b54866c5552d538997f03.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (1B) | DoReMi (1B->1B) |
| ------- | ------------- | --------------- |
| 0       | 6.5           | 7.0             |
| 25000   | 5.8           | 8.0             |
| 50000   | 9.2           | 9.8             |
| 75000   | 8.5           | 10.8            |
| 100000  | 9.5           | 12.5            |
| 125000  | 10.5          | 12.8            |
| 150000  | 11.0          | 13.0            |
| 175000  | 11.5          | 13.5            |
| 200000  | 11.2          | 13.8            |
</details>

Figure 5: Average one-shot downstream accuracy across 4 model scales (280M, 510M, 760M, 1B) where the reference/proxy models for DoReMi are the same size as the main model trained with DoReMi domain weights. DoReMi consistently improves downstream accuracy across scales, with a similar 3% accuracy gap at 200k steps at most scales (except for 510M). DoReMi achieves the baseline accuracy 4x faster on average across scales.

Proxy model underperforms main model, especially at larger sizes. Recall that DoReMi uses Group DRO to train a proxy model, which reweights the objective with the domain weights. In contrast, the main model is trained by resampling on the domain weights from DoReMi. When the proxy model and the main model are the same size, which one is the better model? Table 3b shows that the proxy model typically underperforms the main model in this case. The gap between the proxy and main model increases with scale, as the 1B proxy model not only underperforms the 1B main model but also the 1B baseline model, while the 280M proxy model achieves better perplexity than the 280M baseline model on 19/22 domains. Despite the relatively poor quality of the 1B proxy model, the domain weights still allow the 1B main model to achieve the baseline performance over 2x faster. This suggests that DoReMi can succeed even if the proxy model is not trained well. However, we hypothesize that the mismatch between the proxy and main model training (loss reweighting vs. resampling) explains their performance difference and therefore a resampling-based Group DRO optimizer may improve DoReMi for larger proxy models.

Effect of proxy model scale on larger main model's performance. We consider 70M, 150M, 280M, and 1B scales for the DoReMi proxy model while fixing the main model size at 8B (DoReMi (X→8B)). From 70M to 280M, increasing the proxy model size improves downstream accuracy at 8B (Figure 6

![](images/40bff829e97570f3f74a9e8a26b708c9738bd5310baa4fa8263f8475a1293fe3.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (8B) | DoReMi (1B->8B) | DoReMi (280M->8B) | DoReMi (150M->8B) | DoReMi (70M->8B) |
| ------- | ------------- | --------------- | ----------------- | ----------------- | ---------------- |
| 0       | 8.0           | 7.0             | 10.0              | 9.0               | 9.0              |
| 50000   | 14.0          | 13.0            | 17.0              | 16.0              | 15.0             |
| 100000  | 17.0          | 18.0            | 23.0              | 20.0              | 19.0             |
| 150000  | 19.0          | 21.0            | 24.0              | 22.0              | 21.0             |
| 200000  | 20.0          | 22.0            | 26.0              | 23.0              | 21.0             |
</details>

![](images/877ae25938b5b10901815927a4aef74bad2cac2761f76f7f47f2296232ae3d88.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (280M) | DoReMi (280M->280M) | Hardest (280M->280M) | Easiest (280M->280M) |
| ------- | --------------- | ------------------- | -------------------- | -------------------- |
| 0       | 5.0             | 6.0                 | 7.0                  | 0.0                  |
| 50000   | 6.5             | 7.5                 | 7.5                  | 0.0                  |
| 100000  | 6.0             | 8.5                 | 6.5                  | 1.0                  |
| 150000  | 6.5             | 8.5                 | 7.0                  | 3.5                  |
| 200000  | 6.5             | 8.5                 | 7.5                  | 2.5                  |
</details>

Figure 6: Average downstream accuracy for models trained on The Pile. (Left) Increasing the size of the reference/proxy models from 70M to 280M in DoReMi improves downstream accuracy for a 8B main model, but the trend does not continue for the 1B proxy model. We hypothesize that the Group DRO optimizer is worse for larger proxy models. (Right) Optimizing for the hardest or easiest domains rather than excess loss (which combines both) do not achieve the same average downstream accuracy as DoReMi (280M models).

Table 3: Summary of per-domain log-perplexities on The Pile (22 total domains). Average log-perplexity is an unweighted average of the per-domain log-perplexities.   
(a) Varying the size of the proxy/reference model and training at 8B. 

<table><tr><td></td><td>Worst-case log-ppl</td><td>Avg log-ppl</td><td># domains beating baseline</td></tr><tr><td>Baseline (8B)</td><td>1.71</td><td>1.64</td><td>0/22</td></tr><tr><td>DoReMi (70M-&gt;8B)</td><td>1.63</td><td>1.53</td><td>22/22</td></tr><tr><td>DoReMi (150M-&gt;8B)</td><td>1.56</td><td>1.52</td><td>22/22</td></tr><tr><td>DoReMi (280M-&gt;8B)</td><td>1.46</td><td>1.40</td><td>22/22</td></tr><tr><td>DoReMi (1B-&gt;8B)</td><td>1.58</td><td>1.54</td><td>22/22</td></tr></table>

(b) Perplexity of the DoReMi main model and proxy model of the same size. Although the 1B proxy model is relatively poor quality, the resulting domain weights still improve the main model. 

<table><tr><td></td><td>Worst-case log-ppl</td><td>Avg log-ppl</td><td># domains beating baseline</td></tr><tr><td>Baseline (280M)</td><td>2.39</td><td>2.32</td><td>0/22</td></tr><tr><td>DoReMi (280M-&gt;280M)</td><td>2.19</td><td>2.13</td><td>22/22</td></tr><tr><td>Proxy (280M)</td><td>2.33</td><td>2.27</td><td>19/22</td></tr><tr><td>Baseline (1B)</td><td>1.94</td><td>1.87</td><td>0/22</td></tr><tr><td>DoReMi (1B-&gt;1B)</td><td>1.92</td><td>1.83</td><td>19/22</td></tr><tr><td>Proxy (1B)</td><td>2.11</td><td>2.02</td><td>0/22</td></tr></table>

left). We hypothesize that this trend does not continue for the 1B proxy model because the Group DRO optimizer is worse at larger scales (Table 3b). While DoReMi (280M→8B) results in the most improvement at 8B, DoReMi (150M→8B) and DoReMi (1B→8B) still achieve the baseline accuracy almost 2x faster. This suggests that DoReMi is robust to the proxy model scale. In practice, we suggest choosing a relatively small proxy model size (280M) to save compute.

Choosing the easiest or hardest domains do not suffice. We ablate the components of the excess loss metric $\ell_{\theta}(x) - \ell_{\mathrm{ref}}(x)$ by running DoReMi using only the loss of the proxy model $p_{\theta}$ on example x, i.e. $\ell_{\theta}(x)$ (prefer hardest domains for the proxy model) or only the negative loss of the reference $-\ell_{\mathrm{ref}}(x)$ (prefer easiest domains for the reference model). Figure 6 (right) shows that neither of the components of the excess loss alone are sufficient to achieve the gains of DoReMi.

# 5 Related Work

Curating pretraining data for LMs. Most closely related is the GLaM dataset (Du et al., 2021) (also used for training PaLM (Chowdhery et al., 2022)), which has domain weights that are tuned using downstream data. Optimizing domain weights for downstream tasks can be expensive and could require search/zero-order optimization (Snoek et al., 2012), RL (Zoph and Le, 2016), or heuristic assumptions on how positive/negative transfer between domains work. Example-level filtering also brings benefits for LM training. The C4 dataset (Raffel et al., 2019) shows gains over CommonCrawl via heuristic data cleaning methods. Du et al. (2021), Xie et al. (2023) show that filtering the data at an example level for high-quality text that look like Wikipedia and books can significantly improve downstream performance for LMs. In contrast to these works, DoReMi sets domain weights automatically with only two small LM training runs and does not make assumptions about the type of data to prefer (Wikipedia-like, etc.).

General data selection methods. Moore-Lewis selection (Axelrod, 2017, Feng et al., 2022, Moore and Lewis, 2010) selects examples with high cross-entropy difference (similar to excess log-perplexity) between language models trained on target and raw data. In contrast, DoReMi reweights the data without a target distribution. Coleman et al. (2020) select examples based on the uncertainty of a small proxy model for active learning, while DoReMi uses DRO on the excess loss with respect to a reference model, and focuses on data mixture reweighting. Mindermann et al. (2022) select examples in an online fashion by taking the top k examples in a minibatch according to excess loss. DoReMi optimizes the data mixture before training, allowing the larger main model to train in a standard way. Many other works on data selection are in vision (Kaushal et al., 2019, Killamsetty et al., 2021a,b,c, Mirzasoleiman et al., 2020, Paul et al., 2021, Sener and Savarese, 2018, Sorscher et al., 2022, Wang et al., 2020, Wei et al., 2015) and mainly focus on example-level subset selection with metrics such as gradient matching. Overall, these methods do not address data selection for pretraining, where the downstream data distribution may be very different from the pretraining distribution. DoReMi aims to address the pretraining/downstream distribution shift with a robust optimization approach. To the best of our knowledge, we are the first to show that reweighting the data according to losses of a small proxy LM can improve the training efficiency of much larger LM.

Distributionally robust optimization. Within DRO methods for deep learning (Ben-Tal et al., 2013, Oren et al., 2019, Sagawa et al., 2020, Sinha et al., 2018), we target a restricted form of shift called group shifts (Duchi et al., 2019, Oren et al., 2019, Sagawa et al., 2020), where the test distribution can be an unknown mixture of groups (domains). We follow DRO-LM (Oren et al., 2019), which employs DRO for LMs in the group shift setting. DRO-LM also uses a baselined loss, but with a simple bigram reference model. DoReMi uses a reference model of the same size and architecture as the proxy model to ensure that the losses are on a similar scale. During optimization, DRO-LM takes a worst-case subset of each minibatch to update the model on, while we use the Group DRO optimizer (Sagawa et al., 2020) which doesn't require online subselection. If we equalize the number of examples in each minibatch used for gradient updates, online subselection is more expensive than Group DRO since it requires running forward passes on a larger minibatch (e.g., double the minibatch size) before selecting a subset to update the model with. In comparison, the Group DRO optimizer updates the model on all examples in a weighted fashion. Overall, in contrast to these DRO methods which aim to produce robust models, we use DRO to optimize the data for training larger models more efficiently.

Data-centric AI. Large-scale datasets and benchmarks have driven much of the recent progress in AI, including vision, NLP, and multimodal models (Deng et al., 2009, Gadre et al., 2023, Gao

et al., 2020, Raffel et al., 2019, Rajpurkar et al., 2016, Russakovsky et al., 2015, Schuhmann et al., 2022, Wang et al., 2019). However, most datasets are still painstakingly created with human-generated data, manual work, and heuristics (Deng et al., 2009, Gadre et al., 2023, Gao et al., 2020, Raffel et al., 2019, Schuhmann et al., 2022). DoReMi is a principled data-centric method that aims to improve language model training efficiency. We hope that DoReMi can provide a starting point for a general data-centric framework for language modeling via robust optimization.

# 6 Discussion and Limitations

Saving compute in DoReMi with extrapolation. In Section 2, we run DoReMi for the number of training steps that will be used to train the final model, which could be unnecessarily expensive. A future direction for saving compute would be to stop running DoReMi at an early step and extrapolate the domain weights for the desired number of steps, since we found that most of the variation in the domain weights during a DoReMi run seems to occur in the beginning of training (Appendix Figure 8).

Choice of reference model. The choice of reference model can affect the domain weights found by DoReMi. For example, iterated DoReMi (Section 3) improves performance by using a reference model trained on the tuned domain weights from a previous round of DoReMi. Further directions include varying the reference model size and using specialized reference models to optimize domain weights for a specific application area.

What is a domain? We define a domain by data provenance in our experiments, but this only enables coarse-grained control. Using fine-grained domains could improve the gains from DoReMi. For example, DoReMi is more effective on The Pile (22 domains) than the GLaM dataset (8 domains). Open directions include automatically finding fine-grained domains (e.g., via clustering as in DRO-LM (Oren et al., 2019)) and reweighting the data at an example level. When domains are very fine-grained, it will be important to control the pessimism of DRO (e.g., DRO can put all the weight on a small set of worst-case examples).

Transferability of domain weights across scales. We optimized the domain weights with a small proxy model (280M) and directly used these domain weights to improve training at a larger scale (8B). Understanding why the domain weights can be transferred across scales and the limits of how far these domain weights transfer are important questions to answer in future work.

Broader impacts. Large language models are We hope to improve training efficiency and reduce the environmental impact of training large LMs (Lacoste et al., 2019, Ligozat et al., 2021, Patterson et al., 2021, Strubell et al., 2019). In particular, by reducing the training time by 2x, we can halve the cost and energy consumption of training large language models. Since such efficiency improvements may be used to develop even larger models, there may be no absolute improvement in energy consumption. Ultimately, we hope to improve the training efficiency and cost of developing future language models relative to existing methods.

Large LMs have also been well-documented to have risks and biases (Abid et al., 2021, Blodgett and OConnor, 2017, Bommasani et al., 2021, Gehman et al., 2020, Nadeem et al., 2020). For example, GPT-3 tends to have an anti-Muslim bias, where Muslims are frequently related to violence or terrorism in analogy and completion tasks (Abid et al., 2021). As large language models are increasingly relied upon in applications, the magnitude of the risks increases (Bommasani et al., 2022). Distributionally robust optimization (DRO), which is used in DoReMi to optimize the data mixture, can have a

favorable impact on fairness (Hashimoto et al., 2018). While the standard approach of minimizing the average loss can lead to disparate performance on minority subgroups that do not contribute heavily to the loss (Amodei et al., 2016), DRO promotes good performance on all groups via a worst-case loss. In this way, DRO-style data-centric methods such as DoReMi can improve the representation disparity between majority and minority subgroups in a dataset.

# 7 Conclusion

We introduced DoReMi, an algorithm reweighting data domains for training language models. DoReMi is able to run on small models and transfer the benefits to 30x larger models, resulting in a 2.6x speedup in training on the Pile just by changing the sampling probabilities on domains. We hope to instigate more research on data-centric approaches for improving language model training efficiency.

# Acknowledgments

We thank Xiangning Chen, Andrew Dai, Zoubin Ghahramani, Balaji Lakshminarayanan, Paul Michel, Yonghui Wu, Steven Zheng, Chen Zhu and the broader Google Bard team members for insightful discussions and pointers.

# References

Abubakar Abid, Maheen Farooqi, and James Zou. Persistent anti-muslim bias in large language models. arXiv preprint arXiv:2101.05783, 2021.   
Dario Amodei et al. Deep speech 2 end to end speech recognition in English and mandarin. In International Conference on Machine Learning (ICML), pages 173–182, 2016.   
Amittai Axelrod. Cynical selection of language model training data. CoRR, abs/1709.02279, 2017. URL http://arxiv.org/abs/1709.02279.   
Aharon Ben-Tal, Dick den Hertog, Anja De Waegenaere, Bertrand Melenberg, and Gijs Rennen. Robust solutions of optimization problems affected by uncertain probabilities. Management Science, 59:341–357, 2013.   
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-answer pairs. In Empirical Methods in Natural Language Processing (EMNLP), 2013.   
Su Lin Blodgett and Brendan OConnor. Racial disparity in natural language processing: A case study of social media African-American English. arXiv preprint arXiv:1707.00061, 2017.   
Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dorottya Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal

Ladhak, Mina Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Ben Newman, Allen Nie, Juan Carlos Niebles, Hamed Nilforoshan, Julian Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, Joon Sung Park, Chris Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Rob Reich, Hongyu Ren, Frieda Rong, Yusuf Roohani, Camilo Ruiz, Jack Ryan, Christopher Ré, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, Krishnan Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, Matei Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, and Percy Liang. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

Rishi Bommasani, Kathleen A. Creel, Ananya Kumar, Dan Jurafsky, and Percy Liang. Picking on the same person: Does algorithmic monoculture lead to outcome homogenization? In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, A. Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, B. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, M. Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, S. Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier García, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, D. Luan, Hyeontaek Lim, Barret Zoph, A. Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, T. S. Pillai, Marie Pellat, Aitor Lewkowycz, E. Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, K. Meier-Hellstern, D. Eck, J. Dean, Slav Petrov, and Noah Fiedel. PaLM: Scaling language modeling with pathways. arXiv, 2022.

Cody Coleman, Christopher Yeh, Stephen Mussmann, Baharan Mirzasoleiman, Peter Bailis, Percy Liang, Jure Leskovec, and Matei Zaharia. Selection via proxy: Efficient data selection for deep learning. In International Conference on Learning Representations (ICLR), 2020.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition (CVPR), pages 248–255, 2009.

Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, M. Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, K. Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. GLaM: Efficient scaling of language models with mixture-of-experts. arXiv, 2021.

John Duchi, Tatsunori Hashimoto, and Hongseok Namkoong. Distributionally robust losses against mixture covariate shifts. https://cs.stanford.edu/\~thashim/assets/publications/condrisk.pdf, 2019.   
Yukun Feng, Patrick Xia, Benjamin Van Durme, and João Sedoc. Automatic document selection for efficient encoder pretraining, 2022. URL https://arxiv.org/abs/2210.10951.   
Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, and Ludwig Schmidt. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023.   
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling. arXiv, 2020.   
Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. Real-toxicity prompts: Evaluating neural toxic degeneration in language models. arXiv preprint arXiv:2009.11462, 2020.   
Tatsunori B. Hashimoto, Megha Srivastava, Hongseok Namkoong, and Percy Liang. Fairness without demographics in repeated loss minimization. In International Conference on Machine Learning (ICML), 2018.   
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems (NeurIPS), 2022.   
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Association for Computational Linguistics (ACL), 2017.   
Vishal Kaushal, Rishabh Iyer, Suraj Kothawade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan. Learning from less data: A unified data subset selection and active learning framework for computer vision. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2019.   
Krishnateja Killamsetty, Durga S, Ganesh Ramakrishnan, Abir De, and Rishabh Iyer. GRAD-MATCH: Gradient matching based data subset selection for efficient deep model training. In International Conference on Machine Learning (ICML), 2021a.   
Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer. Glister: Generalization based data subset selection for efficient and robust learning. In Association for the Advancement of Artificial Intelligence (AAAI), 2021b.

Krishnateja Killamsetty, Xujiang Zhao, Feng Chen, and Rishabh Iyer. Retrieve: Coreset selection for efficient and robust semi-supervised learning. In Advances in Neural Information Processing Systems (NeurIPS), 2021c.   
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.   
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. In Association for Computational Linguistics (ACL), 2019.   
Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the carbon emissions of machine learning. arXiv preprint arXiv:1910.09700, 2019.   
Anne-Laure Ligozat, Julien Lefèvre, Aurélie Bugeau, and Jacques Combaz. Unraveling the hidden environmental impacts of AI solutions for environment. CoRR, abs/2110.11822, 2021. URL https://arxiv.org/abs/2110.11822.   
Sören Mindermann, Jan Brauner, Muhammed Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N. Gomez, Adrien Morisot, Sebastian Farquhar, and Yarin Gal. Prioritized training on points that are learnable, worth learning, and not yet learnt. In International Conference on Machine Learning (ICML), 2022.   
Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning (ICML), 2020.   
Robert C. Moore and William Lewis. Intelligent selection of language model training data. In Proceedings of the ACL 2010 Conference Short Papers, pages 220–224, Uppsala, Sweden, July 2010. Association for Computational Linguistics. URL https://aclanthology.org/P10-2041.   
Moin Nadeem, Anna Bethke, and Siva Reddy. Stereoset: Measuring stereotypical bias in pretrained language models. arXiv preprint arXiv:2004.09456, 2020.   
Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization, 19(4):1574–1609, 2009.   
Yonatan Oren, Shiori Sagawa, Tatsunori Hashimoto, and Percy Liang. Distributionally robust language modeling. In Empirical Methods in Natural Language Processing (EMNLP), 2019.   
Denis Paperno, German Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Association for Computational Linguistics (ACL), 2016.   
David A. Patterson, Joseph Gonzalez, Quoc V. Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David R. So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. CoRR, abs/2104.10350, 2021. URL https://arxiv.org/abs/2104.10350.   
Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. In Association for the Advancement of Artificial Intelligence (AAAI), 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.   
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Empirical Methods in Natural Language Processing (EMNLP), 2016.   
Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for SQuAD. In Association for Computational Linguistics (ACL), 2018.   
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. ImageNet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–252, 2015.   
Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In International Conference on Learning Representations (ICLR), 2020.   
Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.   
Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations (ICLR), 2018.   
Noam Shazeer and Mitchell Stern. 2018.   
Aman Sinha, Hongseok Namkoong, and John Duchi. Certifiable distributional robustness with principled adversarial training. In International Conference on Learning Representations (ICLR), 2018.   
Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. Practical Bayesian optimization of machine learning algorithms. In Advances in Neural Information Processing Systems (NeurIPS), 2012.   
Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. arXiv, 2022.   
Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in NLP. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3645–3650, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1355. URL https://aclanthology.org/P19-1355.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017.   
Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations (ICLR), 2019.   
Xinyi Wang, Hieu Pham, Paul Michel, Antonios Anastasopoulos, Jaime Carbonell, and Graham Neubig. Optimizing data usage via differentiable rewards. In International Conference on Machine Learning (ICML), 2020.

Kai Wei, Rishabh Iyer, and Jeff Bilmes. Submodularity in data subset selection and active learning. In International Conference on Machine Learning (ICML), 2015.   
Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and Percy Liang. Data selection for language models via importance resampling. arXiv preprint arXiv:2302.03169, 2023.   
Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578, 2016.

![](images/b7cd855449e3c0779f8cf1e4d74615284ae1d76f7f2320facb1c4a0348a9f55d.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (280M) | DoReMi (280M->280M) round 1 | DoReMi (280M->280M) round 2 | Downstream-tuned (280M) |
| ------- | --------------- | --------------------------- | --------------------------- | ------------------------ |
| 0       | 4.5             | 4.5                         | 2.5                         | 3.0                      |
| 50000   | 5.5             | 6.5                         | 6.0                         | 7.0                      |
| 100000  | 6.0             | 7.0                         | 6.5                         | 7.5                      |
| 150000  | 5.5             | 7.5                         | 6.5                         | 7.5                      |
| 200000  | 5.5             | 6.5                         | 6.5                         | 8.0                      |
| 250000  | 5.5             | 6.5                         | 6.5                         | 7.5                      |
| 300000  | 5.5             | 7.0                         | 6.5                         | 7.5                      |
</details>

(a) 280M

![](images/ddb3a7f25267dde1b4436fc7f4504b7967f87bdf8ceb99bd95976250b23f6af1.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (510M) | DoReMi (510M->510M) | Downstream-tuned (510M) |
| ------- | --------------- | ------------------- | ----------------------- |
| 0       | 5.0             | 5.0                 | 5.0                     |
| 50000   | 6.5             | 8.5                 | 7.5                     |
| 100000  | 8.0             | 10.5                | 9.0                     |
| 150000  | 9.5             | 12.0                | 9.5                     |
| 200000  | 10.0            | 12.5                | 10.0                    |
| 250000  | 10.5            | 12.0                | 10.5                    |
| 300000  | 10.5            | 11.5                | 10.0                    |
</details>

(b) 510M

![](images/ebeff81398b80981ca031c7b3e7f4949d62cd56ca9b2a64dff742f420d9ff429.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (760M) | DoReMi (760M->760M) | Downstream-tuned (760M) |
| ------- | --------------- | ------------------- | ----------------------- |
| 0       | 6.0             | 5.0                 | 8.0                     |
| 20000   | 10.5            | 9.5                 | 9.0                     |
| 40000   | 11.0            | 10.0                | 9.5                     |
| 60000   | 11.5            | 10.5                | 10.0                    |
| 80000   | 12.0            | 11.0                | 10.5                    |
| 100000  | 12.5            | 11.5                | 11.0                    |
| 120000  | 13.0            | 12.0                | 11.5                    |
| 140000  | 13.5            | 12.5                | 12.0                    |
| 160000  | 13.5            | 12.5                | 12.5                    |
| 180000  | 13.5            | 12.5                | 12.5                    |
| 200000  | 13.5            | 12.5                | 12.5                    |
| 220000  | 13.5            | 12.5                | 12.5                    |
| 240000  | 13.5            | 12.5                | 12.5                    |
| 260000  | 13.5            | 12.5                | 12.5                    |
| 280000  | 13.5            | 12.5                | 12.5                    |
| 300000  | 14.0            | 13.5                | 12.5                    |
</details>

(c) 760M

![](images/a88b56a9c3db23eef73f785018648a3afe7b9a410b2bbe06934b95119f4ac128.jpg)

<details>
<summary>line</summary>

| Steps   | Baseline (1B) | DoReMi (1B->1B) | Downstream-tuned (1B) |
| ------- | ------------- | --------------- | --------------------- |
| 0       | 7.0           | 7.0             | 7.0                   |
| 20000   | 8.5           | 9.5             | 9.5                   |
| 40000   | 10.5          | 11.5            | 10.5                  |
| 60000   | 11.5          | 12.5            | 11.0                  |
| 80000   | 12.0          | 13.0            | 11.5                  |
| 100000  | 13.5          | 15.0            | 12.5                  |
| 120000  | 12.5          | 14.5            | 13.5                  |
| 140000  | 13.5          | 15.5            | 14.5                  |
| 160000  | 14.0          | 15.5            | 14.5                  |
| 180000  | 14.5          | 16.0            | 14.5                  |
| 200000  | 15.5          | 16.5            | 14.5                  |
| 220000  | 16.0          | 17.0            | 15.0                  |
| 240000  | 16.5          | 17.5            | 15.5                  |
| 260000  | 17.0          | 17.5            | 16.0                  |
| 280000  | 17.5          | 17.5            | 16.5                  |
| 300000  | 17.5          | 17.5            | 16.5                  |
</details>

(d) 1B   
Figure 7: Average one-shot downstream accuracy across 4 model scales, where the reference/proxy models for DoReMi are the same size as the final model trained with DoReMi domain weights. All models in this figure are trained on the GLaM dataset. DoReMi consistently improves downstream accuracy across scales.

# A Results Across Scales on the GLaM dataset

Figure 7 presents results across different scales (280M, 510M, 760M, 1B) on the GLaM dataset, where the proxy/reference models are the same size as the main model trained with DoReMi domain weights. Across all scales, DoReMi is comparable or better than both the baseline (uniform) domain weights and downstream-tuned domain weights. Interestingly, for iterated DoReMi at the 280M scale, the second round weights achieve slightly worse downstream accuracy than the round 1 weights when used to train 280M models, but transfer better to training 8B models.

# B Detailed Results for The Pile

Per-domain perplexities for 8B models. Table 4 shows per-domain perplexities for 8B models trained on the Pile. The reference/proxy models in this case are 70M, 150M, 280M, and 1B. DoReMi improves the perplexity on each domain compared to the baseline domain weights.

Table 4: Per-domain log-perplexities for 8B models trained on The Pile where the reference/proxy models are or smaller sizes (70M, 150M, 280M, 1B). Models trained with DoReMi domain weights have lower perplexity on all domains than the baseline weights. 

<table><tr><td></td><td>Baseline (8B)</td><td>DoReMi (70M-&gt;8B)</td><td>DoReMi (150M-&gt;8B)</td><td>DoReMi (280M-&gt;8B)</td><td>DoReMi (1B-&gt;8B)</td></tr><tr><td>Pile-CC</td><td>1.64</td><td>1.51</td><td>1.48</td><td>1.41</td><td>1.55</td></tr><tr><td>PubMed Central</td><td>1.60</td><td>1.58</td><td>1.54</td><td>1.46</td><td>1.56</td></tr><tr><td>Books3</td><td>1.65</td><td>1.52</td><td>1.50</td><td>1.42</td><td>1.57</td></tr><tr><td>OpenWebText2</td><td>1.66</td><td>1.48</td><td>1.54</td><td>1.36</td><td>1.58</td></tr><tr><td>ArXiv</td><td>1.64</td><td>1.56</td><td>1.53</td><td>1.38</td><td>1.51</td></tr><tr><td>Github</td><td>1.65</td><td>1.55</td><td>1.54</td><td>1.42</td><td>1.53</td></tr><tr><td>FreeLaw</td><td>1.64</td><td>1.55</td><td>1.54</td><td>1.45</td><td>1.55</td></tr><tr><td>StackExchange</td><td>1.61</td><td>1.52</td><td>1.54</td><td>1.39</td><td>1.55</td></tr><tr><td>USPTO Backgrounds</td><td>1.70</td><td>1.53</td><td>1.50</td><td>1.41</td><td>1.56</td></tr><tr><td>PubMed Abstracts</td><td>1.61</td><td>1.56</td><td>1.51</td><td>1.44</td><td>1.55</td></tr><tr><td>Gutenberg (PG-19)</td><td>1.70</td><td>1.56</td><td>1.54</td><td>1.35</td><td>1.52</td></tr><tr><td>OpenSubtitles</td><td>1.58</td><td>1.56</td><td>1.52</td><td>1.40</td><td>1.55</td></tr><tr><td>Wikipedia (en)</td><td>1.66</td><td>1.49</td><td>1.53</td><td>1.35</td><td>1.56</td></tr><tr><td>DM Mathematics</td><td>1.63</td><td>1.50</td><td>1.56</td><td>1.38</td><td>1.48</td></tr><tr><td>Ubuntu IRC</td><td>1.71</td><td>1.53</td><td>1.49</td><td>1.42</td><td>1.48</td></tr><tr><td>BookCorpus2</td><td>1.64</td><td>1.57</td><td>1.54</td><td>1.43</td><td>1.57</td></tr><tr><td>EuroParl</td><td>1.59</td><td>1.52</td><td>1.51</td><td>1.37</td><td>1.53</td></tr><tr><td>HackerNews</td><td>1.66</td><td>1.50</td><td>1.55</td><td>1.45</td><td>1.55</td></tr><tr><td>YoutubeSubtitles</td><td>1.67</td><td>1.63</td><td>1.55</td><td>1.42</td><td>1.53</td></tr><tr><td>PhilPapers</td><td>1.67</td><td>1.55</td><td>1.49</td><td>1.39</td><td>1.53</td></tr><tr><td>NIH ExPorter</td><td>1.63</td><td>1.51</td><td>1.48</td><td>1.36</td><td>1.52</td></tr><tr><td>Enron Emails</td><td>1.62</td><td>1.48</td><td>1.52</td><td>1.44</td><td>1.56</td></tr></table>

Table 5: Per-task exact-match accuracies for generative one-shot tasks. All DoReMi models improve downstream performance significantly over the baseline domain weights. 

<table><tr><td></td><td>Baseline</td><td>DoReMi (1B-&gt;8B)</td><td>DoReMi (280M-&gt;8B)</td><td>DoReMi (150M-&gt;8B)</td><td>DoReMi (70M-&gt;8B)</td></tr><tr><td>LAMBADA</td><td>20.10</td><td>22.55</td><td>29.19</td><td>20.59</td><td>26.20</td></tr><tr><td>NaturalQuestions</td><td>4.35</td><td>6.01</td><td>7.73</td><td>6.26</td><td>5.10</td></tr><tr><td>SQuADv2</td><td>44.43</td><td>42.22</td><td>51.89</td><td>46.53</td><td>40.99</td></tr><tr><td>TriviaQA</td><td>24.55</td><td>32.25</td><td>34.86</td><td>30.01</td><td>26.30</td></tr><tr><td>WebQuestions</td><td>6.74</td><td>8.71</td><td>9.15</td><td>9.15</td><td>6.99</td></tr><tr><td>Average</td><td>20.03</td><td>22.35</td><td>26.56</td><td>22.51</td><td>21.11</td></tr></table>

Per-task accuracies for 8B models. Table 5 shows the accuracies on one-shot generative tasks for various reference/proxy model sizes from 70M to 1B. All DoReMi models improve downstream performance significantly over the baseline.

Summary of perplexity results across scales. Table 6 shows a summary of per-domain perplexities for DoReMi across 4 scales (280M, 510M, 760M, 1B). Here, the reference/proxy models are the same size as the main model trained with DoReMi domain weights. On average, DoReMi improves perplexity on 18.25 out of 22 domains from The Pile. The worst-case perplexity is always reduced (or comparable in the 510M case) with respect to the baseline domain weights.

Perplexity results for ablations. Table 7 shows the perplexities for ablations on the DRO objective. We change the DRO objective and use these to tune domain weights on 280M reference/proxy models. These tuned domain weights are then used to train a main 280M model. Hardest refers to optimizing the domain-level log-perplexity without baselining with a reference model. Easiest refers to optimizing for the domains with lowest log-perplexity under the reference model. Both

Table 6: Summary of per-domain log-perplexities for 280M, 510M, 760M, and 1B models trained on The Pile, where the reference/proxy models are the same size. DoReMi improves the worst-case and average perplexity of the baseline domain weights in all cases. On average, DoReMi improves perplexity on 18 out of 22 domains. 

<table><tr><td></td><td>Worst-case log-ppl</td><td>Avg log-ppl</td><td># domains beating baseline</td></tr><tr><td>Baseline (280M)</td><td>2.39</td><td>2.32</td><td>0/22</td></tr><tr><td>DoReMi (280M-&gt;280M)</td><td>2.19</td><td>2.13</td><td>22/22</td></tr><tr><td>Proxy (280M)</td><td>2.33</td><td>2.27</td><td>19/22</td></tr><tr><td>Baseline (510M)</td><td>2.14</td><td>2.08</td><td>0/22</td></tr><tr><td>DoReMi (510M-&gt;510M)</td><td>2.14</td><td>2.06</td><td>15/22</td></tr><tr><td>Proxy (510M)</td><td>2.23</td><td>2.18</td><td>0/22</td></tr><tr><td>Baseline (760M)</td><td>2.05</td><td>1.97</td><td>0/22</td></tr><tr><td>DoReMi (760M-&gt;760M)</td><td>2.00</td><td>1.94</td><td>17/22</td></tr><tr><td>Proxy (760M)</td><td>2.15</td><td>2.10</td><td>0/22</td></tr><tr><td>Baseline (1B)</td><td>1.94</td><td>1.87</td><td>0/22</td></tr><tr><td>DoReMi (1B-&gt;1B)</td><td>1.92</td><td>1.83</td><td>19/22</td></tr><tr><td>Proxy (1B)</td><td>2.11</td><td>2.02</td><td>0/22</td></tr></table>

Table 7: Summary of perplexity results for ablations on the DRO objective (excess loss). The individual components (which prefer hardest and easiest domains respectively) do not reduce perplexity over the baseline. 

<table><tr><td></td><td>Worst-case log-ppl</td><td>Avg log-ppl</td><td># domains beating baseline</td></tr><tr><td>Baseline (280M)</td><td>2.39</td><td>2.32</td><td>0</td></tr><tr><td>DoReMi (280M-&gt;280M)</td><td>2.19</td><td>2.13</td><td>22/22</td></tr><tr><td>Hardest (280M-&gt;280M)</td><td>2.66</td><td>2.62</td><td>0/22</td></tr><tr><td>Easiest (280M-&gt;280M)</td><td>4.27</td><td>4.18</td><td>0/22</td></tr></table>

ablations do not improve perplexity on any domain over the baseline. Optimizing for the “hardest” domain does not actually result in improving worst-case perplexity, supporting the results of Oren et al. (2019), which also employs DRO for language modeling with a baselined loss.

Trajectory of domain weights. Figure 8 shows the exponential moving average (smoothing parameter 0.99) of domain weights during a run of DoReMi. In both cases, there are domains with very high weight initially and decrease in weight very quickly (within 50k steps). Since we compute the final domain weights by integrating these curves over steps and normalizing, this suggests that if we have a smaller compute budget, these domains could become more important — this highlights the dependence of the mixture weights on the compute budget. At the same time, the domain weights tend to quickly stabilize after 50k steps, suggesting that the optimal domain weights should be similar for larger compute budgets. We may also be able to take advantage of this stability after 50k steps to run DoReMi for a smaller number of steps and extrapolate the domain weights to save compute.

![](images/491b0d3706567db5f74400f946ffb2919640b7871d9462cef5b4af0edadb58c6.jpg)

<details>
<summary>line</summary>

| Steps   | Weight (Red Solid) | Weight (Blue Dash) | Weight (Orange Dash) | Weight (Green Dash) | Weight (Gray Solid) |
| ------- | ------------------ | ------------------ | -------------------- | ------------------- | ------------------- |
| 0       | 0.0                | 0.5                | 0.35                 | 0.0                 | 0.0                 |
| 50000   | 0.55               | 0.1                | 0.05                 | 0.0                 | 0.1                 |
| 100000  | 0.65               | 0.05               | 0.0                  | 0.0                 | 0.05                |
| 150000  | 0.65               | 0.05               | 0.0                  | 0.0                 | 0.05                |
| 200000  | 0.65               | 0.05               | 0.0                  | 0.0                 | 0.05                |
</details>

![](images/4b09e1887b69725d87b12b431bd120d32a025633fea79c54e18aaa467ffa17ba.jpg)

<details>
<summary>treemap</summary>

| Category | Subcategory | Parent-child | Parent-child |
| :--- | :--- | :--- | :--- |
| Pile-CC | - | - | - |
| PubMed Central | - | - | - |
| Books3 | - | - | - |
| OpenWebText2 | - | - | - |
| ArXiv | - | - | - |
| Github | - | - | - |
| FreeLaw | - | - | - |
| StackExchange | - | - | - |
| USPTO Backgrounds | - | - | - |
| PubMed Abstracts | - | - | - |
| Gutenberg (PG-19) | - | - | - |
| OpenSubtitles | ... | ... | ... |
| Wikipedia (en) | ... | ... | ... |
| DM Mathematics | ... | ... | ... |
| Ubuntu IRC | ... | ... | ... |
| BookCorpus2 | ... | ... | ... |
| EuroParl | ... | ... | ... |
| HackerNews | ... | ... | ... |
| YoutubeSubtitles | ... | ... | ... |
| PhilPapers | ... | ... | ... |
| NIH ExPorter | ... | ... | ... |
| Enron Emails | ... | ... | ... |
</details>

(a) 280M   
![](images/789cded094e1be6181b4822fd3bcc57cd635a8c79cbf6c93c1629c8f3ed56bba.jpg)

<details>
<summary>line</summary>

| Steps   | Weight (Gray) | Weight (Orange) | Weight (Red) | Weight (Green) | Weight (Blue) | Weight (Yellow) | Weight (Cyan) | Weight (Black) |
| ------- | ------------- | --------------- | ------------ | -------------- | ------------- | --------------- | ------------- | -------------- |
| 0       | 0.0           | 0.7             | 0.0          | 0.0            | 0.0           | 0.0             | 0.0           | 0.0            |
| 50000   | 0.4           | 0.1             | 0.1          | 0.1            | 0.1           | 0.1             | 0.1           | 0.1            |
| 100000  | 0.35          | 0.15            | 0.15         | 0.15           | 0.15          | 0.15            | 0.15          | 0.15           |
| 150000  | 0.3           | 0.1             | 0.15         | 0.1            | 0.1           | 0.1             | 0.1           | 0.1            |
| 200000  | 0.25          | 0.1             | 0.15         | 0.1            | 0.1           | 0.1             | 0.1           | 0.1            |
</details>

![](images/b754751eb597d8f324952bea7175d1faa646351f969004dbd60ae521ba835f44.jpg)

<details>
<summary>treemap</summary>

| Category | Subcategory | Parent-child | Parent-child |
| :--- | :--- | :--- | :--- |
| Pile-CC | - | - | - |
| PubMed Central | - | - | - |
| Books3 | - | - | - |
| OpenWebText2 | - | - | - |
| ArXiv | - | - | - |
| Github | - | - | - |
| FreeLaw | - | - | - |
| StackExchange | - | - | - |
| USPTO Backgrounds | - | - | - |
| PubMed Abstracts | - | - | - |
| Gutenberg (PG-19) | - | - | - |
| OpenSubtitles | ... | ... | ... |
| Wikipedia (en) | ... | ... | ... |
| DM Mathematics | ... | ... | ... |
| Ubuntu IRC | ... | ... | ... |
| BookCorpus2 | ... | ... | ... |
| EuroParl | ... | ... | ... |
| HackerNews | ... | ... | ... |
| YoutubeSubtitles | ... | ... | ... |
| PhilPapers | ... | ... | ... |
| NIH ExPorter | ... | ... | ... |
| Enron Emails | ... | ... | ... |
</details>

(b) 1B   
Figure 8: Exponential moving average of domain weights throughout a DoReMi run for 280M and 1B reference/proxy models. In the beginning of the run, the domain weights change quickly and then become more stable after 50k steps. This suggests that 1) smaller compute budgets may require drastically different domain weights, and 2) we may be able to save compute by extrapolating the domain weights after 50k steps.

Comparison of domain weights for 280M and 1B. Table 8 presents the DoReMi domain weights for The Pile at 280M and 1B proxy models. Different proxy model sizes can result in different domain weights, which suggests that there may be multiple local minima in domain weight space. With a 280M proxy model, most of the weight is put on the Pile-CC web text domain, while DoReMi with a 1B proxy model puts most of the weight on OpenWebText2. The overall pattern of the domain weights for the rest of the domains are similar.

# C Training Details

Data preprocessing. For all datasets, we preprocessed the data by chunking into length 1024 examples with respect to a SentencePiece tokenizer with 256k vocabulary size. The examples are separated by domain to facilitate hierarchical sampling (first sample a domain according to some domain weights, then sample an example from that domain at random). To reduce the amount of

Table 8: Domain weights on The Pile. Baseline domain weights are computed from the default Pile dataset. With different proxy model sizes, DoReMi (280M) and DoReMi (1B) result in different domain weights. Despite the differences, the qualitative patterns are similar other than the which web domain has the most weight. 

<table><tr><td></td><td>Baseline</td><td>DoReMi (280M)</td><td>DoReMi (1B)</td></tr><tr><td>Pile-CC</td><td>0.1121</td><td>0.6057</td><td>0.1199</td></tr><tr><td>PubMed Central</td><td>0.1071</td><td>0.0046</td><td>0.0149</td></tr><tr><td>Books3</td><td>0.0676</td><td>0.0224</td><td>0.0739</td></tr><tr><td>OpenWebText2</td><td>0.1247</td><td>0.1019</td><td>0.3289</td></tr><tr><td>ArXiv</td><td>0.1052</td><td>0.0036</td><td>0.0384</td></tr><tr><td>Github</td><td>0.0427</td><td>0.0179</td><td>0.0129</td></tr><tr><td>FreeLaw</td><td>0.0386</td><td>0.0043</td><td>0.0148</td></tr><tr><td>StackExchange</td><td>0.0929</td><td>0.0153</td><td>0.0452</td></tr><tr><td>USPTO Backgrounds</td><td>0.0420</td><td>0.0036</td><td>0.0260</td></tr><tr><td>PubMed Abstracts</td><td>0.0845</td><td>0.0113</td><td>0.1461</td></tr><tr><td>Gutenberg (PG-19)</td><td>0.0199</td><td>0.0072</td><td>0.0250</td></tr><tr><td>OpenSubtitles</td><td>0.0124</td><td>0.0047</td><td>0.0017</td></tr><tr><td>Wikipedia (en)</td><td>0.0919</td><td>0.0699</td><td>0.0962</td></tr><tr><td>DM Mathematics</td><td>0.0198</td><td>0.0018</td><td>0.0004</td></tr><tr><td>Ubuntu IRC</td><td>0.0074</td><td>0.0093</td><td>0.0044</td></tr><tr><td>BookCorpus2</td><td>0.0044</td><td>0.0061</td><td>0.0029</td></tr><tr><td>EuroParl</td><td>0.0043</td><td>0.0062</td><td>0.0078</td></tr><tr><td>HackerNews</td><td>0.0075</td><td>0.0134</td><td>0.0058</td></tr><tr><td>YoutubeSubtitles</td><td>0.0042</td><td>0.0502</td><td>0.0159</td></tr><tr><td>PhilPapers</td><td>0.0027</td><td>0.0274</td><td>0.0063</td></tr><tr><td>NIH ExPorter</td><td>0.0052</td><td>0.0063</td><td>0.0094</td></tr><tr><td>Enron Emails</td><td>0.0030</td><td>0.0070</td><td>0.0033</td></tr></table>

padding tokens, we made an effort to pack examples (possibly from different domains) together into the same sequence. When doing such a packing, we compute the domain perplexities on a per-token level in DoReMi.

Baseline domain weights for The Pile. The baseline domain weights for The Pile were computed from The Pile dataset and the number of epochs for each domain given in Gao et al. (2020). After chunking into length 1024 examples, we counted the number of examples in each domain and multiplied by the number of epochs that domain specified in Gao et al. (2020). We then normalized these counts to obtain the baseline domain weights.

Training setup. For all training runs (including DRO runs), we train with a batch size of 512, initial learning rate of 1e-3, weight decay of 1e-2, and gradient clipping to norm 1. We decay the learning rate exponentially until it reaches a minimum of 1e-4 at the end of training, with a linear warmup of 6% of the total training steps. We train for 200k steps on The Pile and 300k steps on the GLaM dataset. Models under 1B parameters were trained with TPUv3 accelerators, while 1B and 8B models were trained with TPUv4.

Model architectures. Table 9 shows the architecture hyperparameters for the model sizes used in the paper. All the models we use are vanilla Transformer decoder-only models with a 256k vocab size.

Table 9: Architecture hyperparameters for various model scales used in the paper. All models are vanilla Transformer decoder-only models and use vocabulary size 256k. 

<table><tr><td></td><td>Layers</td><td>Attention heads</td><td>Attention head dim</td><td>Model dim</td><td>Hidden dim</td></tr><tr><td>70M</td><td>3</td><td>4</td><td>64</td><td>256</td><td>1024</td></tr><tr><td>150M</td><td>6</td><td>8</td><td>64</td><td>512</td><td>2048</td></tr><tr><td>280M</td><td>12</td><td>12</td><td>64</td><td>768</td><td>3072</td></tr><tr><td>510M</td><td>12</td><td>16</td><td>64</td><td>1024</td><td>8192</td></tr><tr><td>760M</td><td>12</td><td>20</td><td>64</td><td>1280</td><td>8192</td></tr><tr><td>1B</td><td>16</td><td>32</td><td>64</td><td>2048</td><td>8192</td></tr><tr><td>8B</td><td>32</td><td>32</td><td>128</td><td>4096</td><td>24576</td></tr></table>

# D Simple Example Where Data Reweighting Has No Tradeoff

Motivated by the findings in Section 3.2, we present a simple language modeling example where reweighting the training data from different domains improves perplexity on all domains. The example shows that DoReMi downweights domains that are extremely high or low entropy.

Setup. Suppose the ground-truth distribution of text $p^{*}$ is a mixture over k domains, where each domain $z \in \{1, \ldots, k\}$ is defined by a different unigram distribution $p^{*}(x \mid z)$ over m tokens. Given a budget of n training samples, the goal is choose domain weights $p(z)$ (k scalars that add to 1) to sample training data with such that we learn the parameters of the unigram distributions $p^{*}(\cdot \mid z)$ well for all z from 1 to k. Notably, we do not aim to estimate the ground truth mixture proportions across domains.

Data. Given some domain weights $p(z)$ , we sample training data hierarchically: first we determine the number of samples $n_{z}$ per domain z by drawing from a multinomial distribution over k possibilities with probabilities defined by $p(z)$ and n total trials. Then, for each domain z, we sample $n_{z}$ tokens from $p^{*}(\cdot \mid z)$ , forming a vector of tokens $X_{z}$ with length $n_{z}$ .

Model. For each domain z, we consider a Bayesian model of the unigram distribution $p(x \mid z; \theta)$ with a Dirichlet prior $p(\theta \mid z; \beta)$ over the unigram distribution parameters $\theta \in \Delta^{m}$ . The Dirichlet prior has hyperparameters $\beta \in R^{m}$ , which can be viewed as a “pseudo-count” for each token. For each domain z, we estimate the parameters $\hat{\theta}_{z}$ by computing the mean of the posterior distribution conditioned on the data:

$$
\hat {\theta} _ {z} (x) = \frac {1}{n _ {z} + s _ {z}} \left[ \lambda_ {z} (x) + \sum_ {i = 1} ^ {n _ {z}} \mathbf {1} [ X _ {z} [ i ] = x ] \right] \text {   for   all   } x \in \{1, \dots , m \} \tag {2}
$$

where $s_z = \sum_x \lambda_z(x)$ is the sum of pseudocounts.

For a domain $z$ , we can write the parameter error of this estimator as a function of the "difficulty" $H_{z}$ of predicting the next token and the "quality" of the prior $\Delta_z$ , defined below.

Lemma 1. For domain index z with $n_{z}$ samples, the parameter error is

$$
\sum_ {x} \mathbb {E} [ (\hat {\theta} _ {z} (x) - p ^ {*} (x \mid z)) ^ {2} ] = \frac {n _ {z} H _ {z} + s _ {z} ^ {2} \Delta_ {z}}{(n _ {z} + s _ {z}) ^ {2}} \tag {3}
$$

where

$$
H _ {z} = \sum_ {x} p ^ {*} (x \mid z) (1 - p ^ {*} (x \mid z)) \tag {4}
$$

$$
\Delta_ {z} = \sum_ {x} \left(p ^ {*} (x \mid z) - \frac {\lambda_ {z} (x)}{s _ {z}}\right) ^ {2}. \tag {5}
$$

Proof. The parameter error is

$$
\sum_ {x} \mathbb {E} [ (\hat {\theta} _ {z} (x) - p ^ {*} (x \mid z)) ^ {2} ] = \sum_ {x} \mathbb {E} [ \hat {\theta} _ {z} (x) ^ {2} ] - 2 \mathbb {E} [ \hat {\theta} _ {z} (x) ] p ^ {*} (x \mid z) + p ^ {*} (x \mid z) ^ {2}. \tag {6}
$$

Evaluating the terms separately,

$$
\mathbb {E} \left[ \hat {\theta} _ {z} (x) \right] = \frac {1}{n _ {z} + s _ {z}} \left[ \lambda_ {z} (x) + \sum_ {i = 1} ^ {n _ {z}} \mathbf {1} \left[ X _ {z} [ i ] = x \right] \right] \tag {7}
$$

$$
= \frac {1}{n _ {z} + s _ {z}} \left(\lambda_ {z} (x) + n _ {z} p ^ {*} (x \mid z)\right) \tag {8}
$$

$$
\mathbb {E} \left[ \hat {\theta} _ {z} (x) ^ {2} \right] = \frac {1}{\left(n _ {z} + s _ {z}\right) ^ {2}} \mathbb {E} \left[ \left(\lambda_ {z} (x) + \sum_ {i = 1} ^ {n _ {z}} \mathbf {1} \left[ X _ {z} [ i ] = x \right]\right) ^ {2} \right] \tag {9}
$$

$$
= \frac {1}{\left(n _ {z} + s _ {z}\right) ^ {2}} \left[ \lambda_ {z} (x) ^ {2} + 2 \lambda_ {z} (x) n _ {z} p ^ {*} (x \mid z) + n _ {z} p ^ {*} (x \mid z) + \left(n _ {z} ^ {2} - n _ {z}\right) p ^ {*} (x \mid z) ^ {2} \right] \tag {10}
$$

Putting it all together, the parameter error can be written as

$$
\sum_ {x} \mathbb {E} [ (\hat {\theta} _ {z} (x) - p ^ {*} (x \mid z)) ^ {2} ] = \sum_ {x} \frac {(s _ {z} ^ {2} - n _ {z}) p ^ {*} (x \mid z) ^ {2} + \lambda_ {z} (x) ^ {2} + (n _ {z} - 2 s _ {z} \lambda_ {z} (x)) p ^ {*} (x \mid z)}{(n _ {z} + s _ {z}) ^ {2}} \tag {11}
$$

$$
= \sum_ {x} \frac {n _ {z} p ^ {*} (x \mid z) \left(1 - p ^ {*} (x \mid z)\right) + s _ {z} ^ {2} \left(p ^ {*} (x \mid z) - \frac {\lambda_ {z} (x)}{s _ {z}}\right) ^ {2}}{\left(n _ {z} + s _ {z}\right) ^ {2}} \tag {12}
$$

$$
= \frac {n _ {z} H _ {z} + s _ {z} ^ {2} \Delta_ {z}}{\left(n _ {z} + s _ {z}\right) ^ {2}}. \tag {13}
$$

No-tradeoff example. Suppose there are 3 domains $z \in \{1, 2, 3\}$ and m = 3 vocabulary tokens $x \in \{1, 2, 3\}$ . We use a symmetric Dirichlet prior (preferring a uniform token distribution) where $\lambda_{z}(x) = 1/3$ for all tokens x and domains z. Here, $s_{z} = \sum_{x} \lambda_{z}(x) = 1$ . In this setting, we show that there is a set of domain weights that has strictly lower parameter error than the baseline where we sample the same number of tokens from each domain: $n_{z}$ are equal for all domains z.

Suppose the ground truth paramaters for the unigram distributions are

$$
\left[ \begin{array}{c c c} 1 & 0 & 0 \\ 0. 7 & 0. 2 & 0. 1 \\ 1 / 3 & 1 / 3 & 1 / 3 \end{array} \right], \tag {14}
$$

where row z contains the parameters for domain z. For example, token 1 has probability 1 under domain 1's unigram distribution.

For domain $z = 1$ (non-noisy domain), we have $H_{1} = 0$ so the parameter error (according to Lemma 1) is

$$
\frac {s _ {1} ^ {2} \Delta_ {1}}{\left(n _ {1} + s _ {1}\right) ^ {2}} \tag {15}
$$

which is strictly decreasing in the number of samples $n_1$ .

For domain z = 3 (noisy domain), we have $\Delta_{3} = 0$ so the parameter error is

$$
\frac {n _ {3} H _ {3}}{(n _ {3} + s _ {3}) ^ {2}}, \tag {16}
$$

by Lemma 1. This error is minimized to zero at $n_3 = 0$ (no samples). This means that we can allocate samples elsewhere while still reducing error.

For z = 2 (intermediate entropy domain), we have $\Delta_{2} = 0.207$ and $H_{2} = 0.46$ . The derivative of the parameter error with respect to the number of samples $n_{2}$ is

$$
\frac {\partial}{\partial n _ {2}} \frac {n _ {2} H _ {2} + s _ {2} ^ {2} \Delta_ {2}}{(n _ {2} + s _ {2}) ^ {2}} = \frac {H _ {2} (s _ {2} - n _ {2}) - 2 s _ {2} ^ {2} \Delta_ {2}}{(n _ {2} + s _ {2}) ^ {3}} \tag {17}
$$

which is negative when

$$
n _ {2} > s _ {2} - \frac {2 s _ {2} ^ {2} \Delta_ {2}}{H _ {2}}. \tag {18}
$$

This inequality holds in this case since $\frac{2\Delta_2}{H_2} < 1$ and $s_2 = 1$ . Therefore the parameter error is decreasing in the number of samples $n_2$ .

Thus, any domain weights that reallocate the examples from domain 3 to domains 1 and 2 reduces the parameter error for all domains.

What kind of domains are downweighted? Intuitively, we can downweight the very noisy (high entropy/difficulty) domain 3 because the initialization perfectly matches the ground truth. This allows us to reallocate samples to the other domains 1 and 2. Between these, domain 1 requires less additional samples since the parameter error decreases very quickly with the number of samples $n_{1}$ (the difficulty $H_{1}$ is zero). Thus, the easiest domains should also receive relatively less weight. In practice, positive transfer between domains (which is not captured here) can also contribute to scenarios where reweighting results in no tradeoff across domains.

Simulation with DoReMi. We consider running DoReMi on the above no-tradeoff instance of the simple example with the ground truth unigram distributions in Equation 14. Note that DoReMi's domain reweighting step (Step 2, Algorithm 1) involves a loop over $T$ iterative model updates, while the estimator from Equation 2 is computed in closed form. To adapt the estimator for DoReMi, we consider an iterative version where the average is computed in an online fashion. We run DoReMi for $T = 500$ steps using minibatch size 1 over the $n = 500$ training examples with domain weight update rate $\eta = 0.5$ . For the model update at step $t$ on an example $x$ from domain $z$ , we increase the pseudo-count $\hat{\theta}_z(x)$ by the current domain weight $\alpha_t$ corresponding to domain $z$ . Instead of using the examples in the minibatch (which is only size 1 and doesn't represent all domains), we compute the per-domain excess log-perplexities in Algorithm 1 using a fixed, independent evaluation set of 30 examples.

We compare DoReMi against a model trained with baseline domain weights, which are uniform over the 3 domains. All models are trained on n = 500 training examples. We evaluate the log-perplexity of a model on each domain in closed form using the ground truth unigram distribution parameters.

On this simple example, DoReMi returns domain weights $[0.39, 0.61, 0.0]$ after rounding to 2 decimal places. These weights correspond to our intuitions — the first domain (non-noisy) is increased by a small amount, the third domain (noisy) is decreased to 0 weight, and most of the weight is allocated to the second domain. We use these domain weights to generate a new dataset of 500 examples. The model trained with this new dataset improves over the baseline model in perplexity on all domains.