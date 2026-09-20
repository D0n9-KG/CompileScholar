# SHEARED LLAMA: ACCELERATING LANGUAGE MODEL PRE-TRAINING VIA STRUCTURED PRUNING

Mengzhou Xia<sup>1</sup>, Tianyu Gao<sup>1</sup>, Zhiyuan Zeng<sup>2</sup> , Danqi Chen<sup>1</sup>

<sup>1</sup>Princeton Language and Intelligence, Princeton University

<sup>2</sup>Department of Computer Science and Technology, Tsinghua University

{mengzhou,tianyug,danqic}@cs.princeton.edu

zengzy20@mails.tsinghua.edu.cn

## ABSTRACT

The popularity of LLaMA (Touvron et al., 2023a;b) and other recently emerged moderate-sized large language models (LLMs) highlights the potential of building smaller yet powerful LLMs. Regardless, the cost of training such models from scratch on trillions of tokens remains high. In this work, we study structured pruning as an effective means to develop smaller LLMs from pre-trained, larger models. Our approach employs two key techniques: (1) targeted structured pruning, which prunes a larger model to a specified target shape by removing layers, heads, and intermediate and hidden dimensions in an end-to-end manner, and (2) dynamic batch loading, which dynamically updates the composition of sampled data in each training batch based on varying losses across different domains. We demonstrate the efficacy of our approach by presenting the Sheared-LLaMA series, pruning the LLaMA2-7B model down to 1.3B and 2.7B parameters. Sheared-LLaMA models outperform state-of-the-art open-source models of equivalent sizes, such as Pythia, INCITE, OpenLLaMA and the concurrent TinyLlama models, on a wide range of downstream and instruction tuning evaluations, while requiring only 3% of compute compared to training such models from scratch. This work provides compelling evidence that leveraging existing LLMs with structured pruning is a far more cost-effective approach for building competitive small-scale LLMs.<sup>1</sup>

## 1 INTRODUCTION

Large language models (LLMs) are extremely performant on a wide range of natural language tasks, but they require enormous amounts of compute to train (OpenAI, 2023; Anthropic, 2023). As such, there is growing interest in building strong moderate-sized models, such as LLaMA (Touvron et al., 2023a;b), MPT (MosaicML, 2023), and Falcon (Almazrouei et al., 2023), that allow for efficient inference and fine-tuning. These LLMs are available in varied sizes suited for different use cases, but training each individual model from scratch—even the smallest billion-parameter models—requires substantial computational resources that are cost-prohibitive for most organizations. In this work, we seek to address the following question:

## Can we produce a smaller, general-purpose, and competitive LLM by leveraging existing pre-trained LLMs, while using much less compute than training one from scratch?

We explore structured pruning as a means to achieve this goal. Pruning is commonly viewed as a solution for compressing task-specific models (Han et al., 2016; Li et al., 2016; Lagunas et al., 2021; Xia et al., 2022; Kurtic et al., 2023), removing redundant parameters and accelerating inference without sacrificing task performance. However, for general-purpose LLMs, pruning inevitably results in performance degradation compared to original models (Frantar & Alistarh, 2023; Sun et al., 2023; Ma et al., 2023), especially when without significant compute invested post-pruning. In this work, we use pruning as an effective approach for developing smaller yet competitive LLMs that require only a fraction of the training compute compared to training them from scratch.

We identify two key technical challenges in this problem. First, how can we decide on final pruned architectures that are strong in performance and efficient for inference? Existing structured pruning techniques for LLMs (Xia et al., 2022; Ma et al., 2023) do not specify targeted structures and lead to suboptimal pruned models in terms of performance and inference speed (Table 4 and Appendix F.2). Second, how can we continue pre-training the pruned model to reach desired performance? We observe that training using the original pre training data leads to imbalanced rates of loss reduction across different domains, compared to when training such models from scratch. This indicates that the pruned model retains varying levels of knowledge for different domains (e.g., GitHub vs. C4) and simply using the pretraining domain proportion results in an inefficient use of data (Figure 4). To address these issues, we propose “LLM-shearing”, an algorithm consisting of the following two components:

• We propose a novel pruning algorithm, dubbed targeted structuredpruning, which prunes a source model to a specified target architecture. The target architecture is determined by leveraging the configurations of existing pre-trained models. Our pruning approach searches for substructures within the source model that maximally preserve performance while adhering to the given constraints.

• We devise a dynamic batch loading algorithm that loads training data from each domain in proportion to its rate of loss reduction, thereby making an efficient use of the data and accelerating the overall performance improvement.

![](images/9bc3c8e37f9aecf677ca11a037c45ace8aa66fab4d799dc736057cad4839c645.jpg)  
Figure 1: Sheared-LLaMA-2.7B surpasses a series of open-source models at a similar scale and only requires 1/32 (3%) of budget to achieve onpar performance with OpenLLaMA-3B-v2.

We demonstrate the efficacy of our proposed method by pruning a LLaMA2-7B model (Touvron et al., 2023b) into two smaller LLMs: Sheared-LLaMA-1.3B and Sheared-LLaMA-2.7B. Despite using only 50 billion addtional tokens (i.e., 5% of OpenLLaMA’s pre-training budget) for pruning and continued pre-training, Sheared-LLaMA-1.3B and Sheared-LLaMA-2.7B outperform other popular LLMs at similar scales, including Pythia (Biderman et al., 2023), INCITE (TogetherAI, 2023b), and OpenLLaMA (Geng & Liu, 2023), on 11 representative downstream tasks (Figure 1; commonsense, reading comprehension, and world knowledge) and instruction tuning for openended generation. Additionally, the downstream performance trajectory suggests that further training the pruned model with more tokens would result in even greater gains. While we only conduct experiments with up to 7B parameter models, our LLM-shearing algorithm is highly generalizable and can be extended to large language models of any size in future work.

## 2 LLM-SHEARING

Given an existing large model $\mathcal { M } _ { S }$ (the source model), we study how to efficiently produce a smaller, strong model $\mathcal { M } _ { T }$ (the target model). We consider this as a two-stage process: (1) Pruning $\mathcal { M } _ { S }$ into $\mathcal { M } _ { T }$ . This reduces the number of parameters but inevitably incurs a performance drop. (2) Continue pre-training $\mathcal { M } _ { T }$ with a standard language modeling objective to reach a target performance. While most recent efforts (Xia et al., 2022; Ma et al., 2023) focus on the former stage, we find the latter stage crucial for producing competitive general-purpose LLMs from structured pruning.

## 2.1 TARGETED STRUCTURED PRUNING

Structured pruning removes groups of model parameters to compress models and accelerate inference. However, existing structured pruning approaches often result in unconventional model configurations that deviate from popular architectures. For example, CoFiPruning (Xia et al., 2022) produces models with non-uniform layer configurations (e.g., different numbers of heads across layers), which incurs inference overhead compared to standard uniform layer configurations (Section 4.2).

![](images/741bc81184490040359841ae32c935d3b439db052e1ccab9a47513d240317760.jpg)  
Figure 2: Targeted structured pruning produces a compact and dense model of a pre-specified shape. Light colors indicate pruned substructures. Masking variables z are learned to control whether a substructure is pruned $( z = 0 )$ or retained $( z = 1 )$ .

In this work, we aim to prune the source model into any target configuration that we specify.This goal is challenging because it requires surgically scaling down all dimensions in a transformer architecture, an endeavor that, to our knowledge, has not been accomplished before for large language models. We leverage the configurations of existing pre-trained models as the target architectures, based on the intuition that these configurations have already been well-optimized to balance model expressivity and inference efficiency. For example, we use the INCITE-Base-3B architecture (TogetherAI, 2023a) as the target structure when producing a 2.7B model.

Our method learns a set of pruning masks on model parameters at different granularities—from global ones like layers and hidden dimensions (persist across all layers), to local ones like attention heads and intermediate dimensions. Assume that the source model $\mathcal { M } _ { S }$ has $L _ { S }$ layers, with each layer consisting of one multi-head attention module (MHA) and one feed-forward network (FFN). $\mathcal { M } _ { S }$ has a hidden state dimension of $d _ { S } ,$ $H _ { S }$ heads in each MHA, and an intermediate dimension of $m _ { \cal S }$ in each FFN. We introduce the following mask variables:
<table><tr><td>Granularity</td><td>Layer</td><td>Hidden dimension</td><td>Head</td><td>Intermediate dimension</td></tr><tr><td>Pruning masks</td><td> $z ^ { \mathrm { l a y e r } } \in \mathbb { R } ^ { L _ { S } }$ </td><td> $z ^ { \mathrm { h i d d e n } } \in \mathbb { R } ^ { d _ { S } }$ </td><td> $\boldsymbol { z } ^ { \mathrm { \scriptsize { h e a d } } } \in \mathbb { R } ^ { H _ { S } } \left( \times L _ { S } \right)$ </td><td> $\boldsymbol { z } ^ { \mathrm { i n t } } \in \mathbb { R } ^ { m _ { S } } \left( \times L _ { S } \right)$ </td></tr></table>

Each mask variable controls whether the associated substructure is pruned or retained. For example, we remove a layer if its corresponding $z ^ { \mathrm { l a y e r } } = 0 .$ . Figure 2 illustrates an example of how the pruning masks control the pruned structures.

We formulate pruning as a constrained optimization problem (Platt & Barr, 1987) where we learn pruning masks to search for a subnetwork matching a pre-specified target architecture while maximizing performance.<sup>2</sup> Following the $\ell _ { 0 }$ regularization approach (Louizos et al., 2018), we parametrize the pruning masks to model hard concrete distributions. These distributions have support on [0, 1] but concentrate their probability mass at 0 or 1, enabling discrete prune or retain decisions. While prior work usually control for a target sparsity (Wang et al., 2020; Xia et al., 2022), we use a pair of Lagrange multipliers to impose constraints on the pruned model shape directly. For example, for a target number of heads $H _ { \mathcal { T } }$ (and we use $L _ { T } , d _ { T }$ , and $m \tau$ to represent the target number of layers, hidden dimension, and intermediate dimension respectively), we have the imposed constraint on a single layer as:

$$
\tilde { \mathcal { L } } ^ { \mathrm { h e a d } } ( \lambda , \phi , z ) = \lambda ^ { \mathrm { h e a d } } \cdot \left( \sum z ^ { \mathrm { h e a d } } - H \tau \right) + \phi ^ { \mathrm { h e a d } } \cdot \left( \sum z ^ { \mathrm { h e a d } } - H \tau \right) ^ { 2 } .
$$

Similar constraints are applied to pruning other substructures. Overall, we jointly optimize the model weights and pruning masks by a min-max objective min<sub>θ,</sub> $_ z \operatorname* { m a x } _ { \lambda , \phi } \mathcal { L } _ { \mathrm { p r u n e } } ( \theta , z , \lambda , \phi )$ :

$$
\mathcal { L } _ { \mathrm { p r u n e } } ( \theta , z , \lambda , \phi ) = \mathcal { L } ( \theta , z ) + \sum _ { j = 1 } ^ { L _ { S } } \tilde { \mathcal { L } } _ { j } ^ { \mathrm { h e a d } } + \sum _ { j = 1 } ^ { L _ { S } } \tilde { \mathcal { L } } _ { j } ^ { \mathrm { i n t } } + \tilde { \mathcal { L } } ^ { \mathrm { l a y e r } } + \tilde { \mathcal { L } } ^ { \mathrm { h i d d e n } } ,
$$

where $\mathcal { L } ( \boldsymbol { \theta } , z )$ is the language modeling loss computed with the masked model weights. This objective will produce a pruned model with the target shape. Ideally, running this pruning algorithm on a large amount of data will directly produce a strong compact model. In practice, the pruning stage is expensive (roughly $5 \times$ slower compared to standard LM training), and we find that the learned masks often converge fast. Therefore, we only allocate a limited budget for pruning (see Table 5). Following pruning, we finalize the pruned architecture by preserving the highest-scoring components associated with the mask variables in each substructure, and continue pre-training the pruned model with the language modeling objective. We refer to this second stage as continued pre-training.

Algorithm 1: Dynamic Batch Loading   
Require: Training data of k domains $D _ { 1 } , D _ { 2 } , \cdots , D _ { k }$ , validation data $D _ { 1 } ^ { \mathrm { v a l } } , D _ { 2 } ^ { \mathrm { v a l } } , \cdot \cdot \cdot , D _ { k } ^ { \mathrm { v a l } } ,$   
initial data loading weights $w _ { 0 } \in \mathbb { R } ^ { k }$ , reference loss $\boldsymbol { \ell } _ { \mathrm { r e f } } \in \mathbb { R } ^ { k }$ , LM loss L or pruning loss   
${ \mathcal { L } } _ { \mathrm { p r u n e } } ,$ training steps $T ,$ , evaluation per m steps, model parameters $\theta \left( \theta , z , \phi , \lambda \right)$ for pruning)   
for $t = 1 , \cdots , T$ do   
if t mod $m = 0$ then   
$\ell _ { t } [ i ] \gets \mathcal { L } ( \theta , z , D _ { i } ^ { \mathrm { v a l } } )$ if pruning else $\mathcal { L } ( \theta , D _ { i } ^ { \mathrm { v a l } } )$   
$\Delta _ { t } [ i ] $ max $\{ \ell _ { t } [ i ] - \dot { \ell _ { \mathrm { r e f } } } [ i ] , 0 \}$ ▷ Calculate loss difference   
$w _ { t } \gets { \sf U p d a t }$ eWeight $( w _ { t - m } , \Delta _ { t } )$ $\triangleright$ Update data loading proportion   
end   
Sample a batch of data B from $D _ { 1 } , D _ { 2 } , \cdots , D _ { k }$ with proportion $w _ { t } ;$   
ifpruning then   
Update $\theta , z , \phi , \lambda$ with ${ \mathcal { L } } _ { \mathrm { p r u n e } } ( \theta , z , \phi , \lambda )$ on B   
else   
Update θ with $\mathcal { L } ( \theta , B )$   
end   
end   
Subroutine UpdateWeight(w, $\Delta )$   
α ← w · exp (∆) ▷ Calculate the unnormalized weights   
w $ \frac { \alpha } { \sum _ { i } { \alpha [ i ] } }$ return w ▷ Renormalize the data loading proportion   
return θ

## 2.2 DYNAMIC BATCH LOADING

Continued pre-training on a large amount of data is crucial for recovering the pruned model performance. We observe a surprising finding in our preliminary experiments: continuing pre-training our pruned models on an existing pre-training dataset RedPajama (TogetherAI, 2023b; LLaMA’s replicated pre-training dataset) reduces loss at different rates across domains compared to pre-training a model from scratch, which signifies an inefficient use of data.

To be more specific, we begin by fitting a scaling function (Hoffmann et al., 2022; details in Appendix B) on the series of LLaMA2 models for each domain. Using this function, we predict the loss of a hypothetical 1.3B LLaMA2 model if it were trained from scratch on the same data. We then compare these estimated reference losses to the losses of our pruned model after continued pre-training. Figure 4 (left) shows that our model’s loss on GitHub is better than the reference loss, while it is significantly worse than the reference loss on C4. This observation indicates that pruning preserves a greater amount of knowledge in low-entropy and smaller domains (e.g., GitHub) compared to high-entropy and larger domains (e.g., C4). Simply reusing the original pre-training data distribution<sup>3</sup> results in an inefficient use of data and worse downstream performance, even if the overall loss is seemingly low, as demonstrated later in Section 4.1.

Inspired by recent work (Xie et al., 2023), we propose dynamic batch loading, an efficient algorithm to adjust domain proportions on the fly based on losses. The goal is to ensure the model achieves the reference loss at roughly the same time across domains. We introduce the algorithm below.

Problem setup. The pre-training data comprises of k domains $D _ { 1 } , D _ { 2 } , \cdots , D _ { k }$ and we have a heldout validation dataset for each domain, denoted as $D _ { i } ^ { \mathrm { v a l } }$ . At each training step $t ,$ a proportion $w _ { t } [ i ]$ of the data comes from domain $D _ { i } .$ . We set a reference validation loss $\ell _ { \mathrm { r e f } } ( D _ { i } )$ for each domain and train the pruned model to reach the reference loss.

Dynamic batch loading. We present the full algorithm in Algorithm 1. In a sketch, for every m steps, we evaluate the model to get the validation loss $\ell _ { t }$ (step t) on $D ^ { \mathrm { v a l } }$ , and update $w _ { t }$ based on the difference $\Delta _ { t } ( D _ { i } )$ between $\ell _ { \mathrm { r e f } } [ i ]$ and $\ell _ { t } [ i ]$ on each domain. The update rule is exponential ascent following Xie et al. (2023),

$$
\alpha _ { t } = w _ { t - m } \cdot \exp ( \Delta _ { t } ) ; \quad w _ { t } = \frac { \alpha _ { t } } { \sum _ { i } \alpha _ { t } [ i ] } .
$$

We apply dynamic batch loading to both the pruning stage and the continued pre-training stage. For pruning, we use the original pre-training data’s domain weights as $w _ { 0 } .$ . For continued pre-training, we use the final weights from the pruning stage as $w _ { 0 } .$ . Dynamic batch loading is an on-the-fly solution that adjusts data proportions during training without the need for training auxiliary models. It leverages reference losses on validation sets and adjusts the weights dynamically, adding minimal overhead to standard training. This approach differs from Xie et al. (2023), which requires a complex multi-stage process to train reference and proxy models.

More broadly, dynamic batch loading can train an LLM to match any reference model’s performance by using open-source pre-training datasets like RedPajama, even without knowing the reference model’s exact training data.

Choices of reference losses. By default, we use the loss predicted by the fitted scaling function as the reference (denoted as scaling reference). We also experiment with an alternative where we directly use the source model’s domain validation loss as the reference (denoted as source reference). We show in F.4 that while both variants perform well, using scaling reference leads to slightly better downstream results, especially on math and coding tasks. However, source reference is a viable alternative when a series of source models at different scales is not available.

## 3 EXPERIMENTS

## 3.1 SETUP

Model configurations. We use the LLaMA2-7B model (Touvron et al., 2023b) as the source model throughout all of our main experiments.<sup>4</sup> We then conduct structured pruning experiments to compress this model down to two smaller target sizes—2.7B and 1.3B parameters. We compare to strong pre-trained language models of similar sizes, including OPT-1.3B (Zhang et al., 2022), Pythia-1.4B (Biderman et al., 2023), TinyLlama-1.1B (Zhang et al., 2024), OPT-2.7B, Pythia-2.8B, INCITE-Base-3B (TogetherAI, 2023b), OpenLLaMA-3B-v1, and OpenLLaMA-3B-v2 (Geng & Liu, 2023). We use Pythia-1.4B and INCITE-Base-3B as the target architecture for the 1.3B and the 2.7B model respectively. Table 8 summarizes model architecture details of all these models.

Data. As the training data for LLaMA2 is not publicly accessible, we use RedPajama (TogetherAI, 2023b), which is a replicated pre-training dataset of the LLaMA1 models (Touvron et al., 2023a), for pruning and continued-pretraining. This dataset encompasses training data from seven domains: CommonCrawl, C4, Github, Wikipedia, Books, ArXiv, and StackExchange. We construct a held-out validation set with 2 million tokens (equivalent to 500 sequences of 4,096 tokens) for each domain. We allocate 0.4 billion tokens for the pruning phase and 50 billion tokens for the continued pre-training process. Following the conventions of LLaMA2, we maintain a sequence length of 4,096 tokens. Table 1 provides a summary of the pre-training data used by our models an

Table 1: A summary of pre-training datasets used by Sheared-LLaMA and other models.
<table><tr><td>Model</td><td>Pre-training Data</td><td>#Tokens</td></tr><tr><td>LLaMA1</td><td>LLaMA data</td><td>1T</td></tr><tr><td>LLaMA2</td><td>Unknown</td><td>2T</td></tr><tr><td>OPT</td><td>OPT data⁵</td><td>300B</td></tr><tr><td>Pythia</td><td>The Pile</td><td>300B</td></tr><tr><td>INCITE-Base</td><td>RedPajama</td><td>800B</td></tr><tr><td>OpenLLaMA v1</td><td>RedPajama</td><td>1T</td></tr><tr><td>OpenLLaMA v2</td><td>OpenLLaMA data6</td><td>1T</td></tr><tr><td>TinyLlama</td><td>TinyLlama data7</td><td>3T</td></tr><tr><td>Sheared-LLaMA</td><td>RedPajama</td><td>50B</td></tr></table>

Table 2: Sheared-LLaMA outperforms publicly available models of comparable size on downstream tasks. The shot number used is noted in parentheses, with 0-shot if not specified. Models with † use a different training data from RedPajama. Please refer to Table 1 for details.
<table><tr><td rowspan="2">Model (#tokens for training)</td><td colspan="6">Commonsense &amp; Reading Comprehension</td></tr><tr><td>SciQ</td><td>PIQA</td><td></td><td></td><td>WinoGrande ARC-E ARC-C (25)</td><td>HellaSwag (10)</td></tr><tr><td>LLaMA2-7B (2T)†</td><td>93.7</td><td>78.1</td><td>69.3</td><td>76.4</td><td>53.0</td><td>78.6</td></tr><tr><td>OPT-1.3B (300B)†</td><td>84.3</td><td>71.7</td><td>59.6</td><td>57.0</td><td>29.7</td><td>54.5</td></tr><tr><td>Pythia-1.4B (300B)†</td><td>86.4</td><td>70.9</td><td>57.4</td><td>60.7</td><td>31.2</td><td>53.0</td></tr><tr><td>TinyLlama-1.1B (3T)†</td><td>88.9</td><td>73.3</td><td>58.8</td><td>55.3</td><td>30.1</td><td>60.3</td></tr><tr><td>Sheared-LLaMA-1.3B (50B)</td><td>87.3</td><td>73.4</td><td>57.9</td><td>61.5</td><td>33.5</td><td>60.7</td></tr><tr><td>OPT-2.7B (300B)†</td><td>85.8</td><td>73.7</td><td>60.8</td><td>60.8</td><td>34.0</td><td>61.5</td></tr><tr><td>Pythia-2.8B (300B)†</td><td>88.3</td><td>74.0</td><td>59.7</td><td>64.4</td><td>36.4</td><td>60.8</td></tr><tr><td>INCITE-Base-3B (800B)</td><td>90.7</td><td>74.6</td><td>63.5</td><td>67.7</td><td>40.2</td><td>64.8</td></tr><tr><td>Open-LLaMA-3B-v1 (1T)</td><td>91.3</td><td>73.7</td><td>61.5</td><td>67.6</td><td>39.6</td><td>62.6</td></tr><tr><td>Open-LLaMA-3B-v2 (1T)†</td><td>91.8</td><td>76.2</td><td>63.5</td><td>66.5</td><td>39.0</td><td>67.6</td></tr><tr><td>Sheared-LLaMA-2.7B (50B)</td><td>90.8</td><td>75.8</td><td>64.2</td><td>67.0</td><td>41.2</td><td>70.8</td></tr></table>

<table><tr><td>Model (#tokens for training)</td><td></td><td>LogiQA BoolQ (32)</td><td>LAMBADA</td><td>NQ (32)</td><td>MMLU (5)</td><td>Average</td></tr><tr><td>LLaMA2-7B (2T)†</td><td>30.7</td><td>82.1</td><td>28.8</td><td>73.9</td><td>46.6</td><td>64.6</td></tr><tr><td>OPT-1.3B (300B)†</td><td>26.9</td><td>57.5</td><td>58.0</td><td>6.9</td><td>24.7</td><td>48.2</td></tr><tr><td>Pythia-1.4B (300B)†</td><td>27.3</td><td>57.4</td><td>61.6</td><td>6.2</td><td>25.7</td><td>48.9</td></tr><tr><td>TinyLlama-1.1B (3T)†</td><td>26.3</td><td>60.9</td><td>58.8</td><td>12.1</td><td>25.5</td><td>50.0</td></tr><tr><td>Sheared-LLaMA-1.3B (50B)</td><td>26.9</td><td>64.0</td><td>61.0</td><td>9.6</td><td>25.7</td><td>51.0</td></tr><tr><td>OPT-2.7B (300B)†</td><td>26.0</td><td>63.4</td><td>63.6</td><td>10.1</td><td>25.9</td><td>51.4</td></tr><tr><td>Pythia-2.8B (300B)†</td><td>28.0</td><td>66.0</td><td>64.7</td><td>9.0</td><td>26.9</td><td>52.5</td></tr><tr><td>INCITE-Base-3B (800B)</td><td>27.7</td><td>65.9</td><td>65.3</td><td>14.9</td><td>27.0</td><td>54.7</td></tr><tr><td>Open-LLaMA-3B-v1 (1T)</td><td>28.4</td><td>70.0</td><td>65.4</td><td>18.6</td><td>27.0</td><td>55.1</td></tr><tr><td>Open-LLaMA-3B-v2 (1T)†</td><td>28.1</td><td>69.6</td><td>66.5</td><td>17.1</td><td>26.9</td><td>55.7</td></tr><tr><td>Sheared-LLaMA-2.7B (50B)</td><td>28.9</td><td>73.7</td><td>68.4</td><td>16.5</td><td>26.4</td><td>56.7</td></tr></table>

Evaluation. We use the lm-evaluation-harness package (Gao et al., 2021) to evaluate on an extensive suite of downstream tasks: (1) We follow Pythia and LLaMA2 to report the 0-shot accuracy of ARC easy (ARC-E; Clark et al., 2018), LAMBADA (Paperno et al., 2016), LogiQA (Liu et al., 2020), PIQA (Bisk et al., 2020), SciQ (Welbl et al., 2017), and WinoGrande (Sakaguchi et al., 2021). (2) We report accuracy of the tasks used by Open LLM Leaderboard (Beeching et al., 2023), including 10-shot HellaSwag (Zellers et al., 2019), 25-shot ARC Challenge (ARC-C; Clark et al., 2018), and 5-shot MMLU (Hendrycks et al., 2021). (3) We also report exact match of 32-shot Natural Questions (NQ; Kwiatkowski et al., 2019) to measure the factual knowledge in the model.

As training models to follow instructions has become a crucial application of LLMs (Ouyang et al., 2022; Taori et al., 2023), we evaluate our models on instruction tuning and fine-tune both baseline models and Sheared-LLaMA on instruction-response pairs sampled from the ShareGPT dataset.<sup>8</sup> Please refer to Appendix E for more details.

## 3.2 SHEARED-LLAMA OUTPERFORMS LMS OF EQUIVALENT SIZES

We demonstrate that Sheared-LLaMA outperforms existing LLMs of similar sizes on both standard LLM benchmarks and instruction tuning, while using only a fraction of the compute budget required to train those models from scratch.

Downstream tasks. Table 2 presents the zero-shot and few-shot downstream task performance of Sheared-LLaMA and similarly-sized pre-trained models. Even with a limited budget of approximately 50B tokens for pruning and continued pre-training, Sheared-LLaMA models outper form existing models pre-trained on significantly larger compute. Sheared-LLaMA-1.3B outperforms OPT-1.3B, Pythia-1.4B (pre-trained with 300B tokens), and TinyLlama-1.1B (pre-trained on 3T tokens). Sheared-LLaMA-2.7B outperforms INCITE-Base-3B (pre-trained on 800B Red-Pajama tokens), OpenLLaMA-3B-v1 (pre-trained on 1T RedPajama tokens), and OpenLLaMA-3Bv2 (trained on 1T tokens from RedPajama, RefinedWeb, and StarCoder). The most noteworthy result is that Sheared-LLaMA-1.3B outperforms TinyLlama-1.1B, despite TinyLlama-1.1B being pre-trained on 3T tokens—more than the total data used for pre-training LLAMA2-7B and our pruning process combined. This demonstrates that structured pruning is a more sample-efficient approach for training smaller-scale LLMs.

![](images/3ba60047bd3f4c765bb6e8c50f1b00973b1aa00c60a9d93eae89a85d00127852.jpg)  
Figure 3: Sheared-LLaMAs outperform Pythia-1.4B, INCITE-Base-3B, OpenLLaMA-3B-v1 and OpenLLaMA-3B-v2 in instruction tuning.

Instruction tuning. As shown Figure 3, instruction-tuned Sheared-LLaMA achieves higher win rates compared to all the other pre-trained models at a comparable scale. This demonstrates that our 2.7B model can serve as a strong foundation for instruction tuning and has the capacity to generate long, coherent and informative responses (See examples in Appendix E).

## 4 ANALYSIS

## 4.1 EFFECTIVENESS OF DYNAMIC BATCH LOADING

We analyze the effectiveness of dynamic batch loading by examining its impact on three aspects: (1) the final LM loss across domains, (2) the data usage of each domain throughout training, (3) the downstream task performance. All results in this section are based on Sheared-LLaMA-1.3B.<sup>9</sup>

Loss differences across domains. Dynamic batch loading aims to balance the rate of loss reduction across domains, ensuring that the losses reach the reference value at roughly the same time. Figure 4 shows the difference between our model’s loss (with both original and dynamic batch loading) and the reference loss, estimated by fitting a scaling function to a hypothetical 1.3B parameter LLaMA2 model. The original batch loading results in widely varying loss differences across domains; for example, the GitHub loss decreases below the reference value, while the C4 loss lags behind. Dynamic batch loading, however, reduces losses evenly and leads to very similar loss differences across domains, suggesting more efficient data use.

Data usage. Table 3 compares the data proportion of RedPajama and the data usage of our dynamic loading approach (Figure 6 illustrates how the domain weights change during training). It shows that dynamic batch loading loads more data from the Book and C4 subsets, indicating that these domains are more challenging for a pruned model to recover.

Table 3: Domain data usage with dynamic batch loading compared to the original proportions.
<table><tr><td></td><td>CC</td><td></td><td></td><td>GitHub Book StackExchange Wiki ArXiv</td><td></td><td></td><td>C4</td></tr><tr><td>RedPajama (Original)</td><td>67.0%</td><td>4.5%</td><td>4.5%</td><td>2.0%</td><td>4.5%</td><td>2.5%</td><td>15.0%</td></tr><tr><td>Dynamic Batch Loading</td><td>g 36.1%</td><td>0.8%</td><td>9.1%</td><td>1.0%</td><td>3.1%</td><td>0.7%</td><td>49.2%</td></tr></table>

Downstream performance. As shown in Figure 5, pruned models trained with dynamic batch loading achieve better downstream performance than when trained on the original RedPajama distribution. This suggests that the more balanced loss reduction from dynamic batch loading transfers to improved downstream capabilities.

![](images/cff3ad4af322ab4a645a513c5269194360cec7f33eadb850a078904e29c7ae04.jpg)  
Figure 4: Loss difference between the pruned model (1.3B) and estimated reference loss, with original vs. dynamic batch loading.

![](images/559e5d3749908bcd211c59bd575d9d6684e1ad74f1f460db09adfd7353c5f056.jpg)  
Figure 5: Downstream task performance of Sheared-LLaMA-1.3B with original data proportion and dynamic batch loading.

## 4.2 COMPARISON TO OTHER PRUNING APPROACHES

We compare our LLM-shearing to other pruning approaches on validation perplexity, a strong indicator of overall model capabilities (Xia et al., 2023).

Targeted pruned models have a higher inference speed. Previous works like CoFiPruning (Xia et al., 2022) produce structued pruned models, but these models often have non-uniform layer configurations (e.g., different numbers of heads across layers). Such non-uniformity across layers introduces training and inference overhead due to irregularities in model architectures. We experiment with both CoFiPruning and targeted structured pruning, with a 0.4B pruning budget with the Red-Pajama data proportion for a fair comparison. Table 4 shows that our targeted pruned models have a higher inference speed compared to the non-uniformly pruned CoFiPruning model at the same sparsity, despite having a slightly higher perplexity. Targeted structured pruning needs about 0.5B more tokens in continued pre-training to match CoFiPruning’s perplexity. However, this one-time extra compute during training is justified, as it results in a more efficient model architecture that is essential for real-world applications and effective practical use. Please find more details on inference speed of different pruning methods in Appendix F.9.

Table 4: Validation perplexity and generation speed during inference (tokens/second) of targeted structured pruning with a uniform layer configuration, and CoFiPruning, with a non-uniform layer configuration. Inference speed is measured on a Nvidia A100 (80G) GPU, on a singal instance generating up to 512 tokens.
<table><tr><td></td><td>Layer Config</td><td>PPL↓ Speed ↑ |</td><td></td><td></td><td>Layer Config</td><td>PPL↓ Speed ↑</td><td></td></tr><tr><td rowspan="2">1.3B</td><td>CoFiPruning</td><td>9.1</td><td>51</td><td rowspan="2">2.7B</td><td>CoFiPruning</td><td>7.0</td><td>37</td></tr><tr><td>Targeted pruning</td><td>10.3</td><td>58</td><td>Targeted pruning</td><td>7.7</td><td>43</td></tr></table>

Comparison to LLM-Pruner (Ma et al., 2023). We compare targeted structured pruning to LLM-Pruner, a recent work in structured pruning, in Appendix F.2. We demonstrate that, given the same compute budget, sparsity level, and training data distribution, our pruned models achieve lower perplexity, have a more optimized architecture, and faster inference speed.

## 4.3 ADDITIONAL ANALYSIS

Budget allocation for pruning and continued pretraining. Intuitively, allocating more compute to the pruning stage helps identify better subnetwork structures. We explore distributing data across pruning and continued pre-training stages differently, within a fixed budget of 5B tokens. Table 5 shows that when controlling the total amount of tokens, increasing the pruning budget consistently improves perplexity. However, since pruning is more expensive than continued pre-training, we decide to allocate 0.4B tokens to pruning. Please refer to Appendix C for details on training throughputs.

Table 5: Data budget allocation to pruning and continued pre-training (CT) and corresponding perplexity.
<table><tr><td colspan="2"># Tokens</td><td colspan="2">PPL</td></tr><tr><td>Pruning</td><td>CT</td><td>Pruning</td><td>CT</td></tr><tr><td>0.2B</td><td>4.6B</td><td>12.99</td><td>7.46</td></tr><tr><td>0.4B</td><td>4.4B</td><td>10.29</td><td>7.32</td></tr><tr><td>0.8B</td><td>4.0B</td><td>9.01</td><td>7.23</td></tr><tr><td>1.6B</td><td>3.2B</td><td>8.04</td><td>7.08</td></tr></table>

More analysis. We provide further analysis in the appendix: (1) Sheared-LLaMA evaluation on math and coding (Appendix F.3), (2) Pythia model pruning (Appendix F.5), and (3) impact of excluding easy domains during pruning (Appendix F.8).

## 5 RELATED WORK

Pruning. Structured pruning has been extensively studied as a model compression technique in computer vision and natural language processing, where task-specific models like classification ones are often overparameterized and can be pruned significantly with minimal impact on performance (Han et al., 2016; Wen et al., 2016; Liu et al., 2017; Luo et al., 2017; Cai et al., 2019; Deng et al., 2020; Hou et al., 2020; Wang et al., 2020; Lagunas et al., 2021; Xia et al., 2022; Kurtic et al., 2023). Unstructured pruning (Frankle & Carbin, 2018; Li et al., 2020; Chen et al., 2020; Sanh et al., 2020) prunes individual neurons instead of structured blocks. Though unstructured pruning usually achieve higher compression rates, they are not practical for model speedup.

In the era of LLMs, the prevalent NLP pipeline has shifted from task-specific models to generalpurpose LMs, which leaves little room for redundancy. Both unstructured pruning, semi-structured pruning (Frantar & Alistarh, 2023; Sun et al., 2023), and structured pruning (Ma et al., 2023) lead to significant performance drops on LLM even at a modest sparsity. Noticeably, all previous works fix the original models or tune them minimally. We see pruning as an initialization and consider it necessary to expend substantial compute to continually pre-training the model to recover performance.

Efficient pre-training approaches. As orthogonal to our pruning approach, There is an extensive body of work on improving efficiency of training LLMs. For example, quantization reduces the numeric precision of model weights and activations and speeds up training and inference (Dettmers et al., 2022; 2023; Xiao et al., 2023). Knowledge distillation (Hinton et al., 2015; Sanh et al., 2019; Jiao et al., 2020; Sun et al., 2020), which trains a smaller model on a larger model’s prediction, is shown to be effective for task-specific models (Xia et al., 2022). For pre-training LLMs, though distilling from a teacher model is shown to improve the quality of student models given the same number of training steps (Rae et al., 2021; Blakeney et al., 2022), it is less cost-effective than pruning and continued training due to the exceeding inference cost incured by the teacher model (Jha et al., 2023). More methods have been introduced to enhance the efficiency of training LMs, such as dynamic architectures (Gong et al., 2019; Zhang & He, 2020) and efficient optimizers (Chen et al., 2023; Liu et al., 2023). However, as indicated by (Kaddour et al., 2023; Bartoldson et al., 2023), the promised gains in training efficiency may not be consistently realized.

There are also data-based approaches to enhance training efficiency. Eliminating duplicated data is found to be effective (Lee et al., 2021). Various batch selection techniques propose to prioritize data based on criteria such as higher losses (Jiang et al., 2019) or a greater reducible loss (Mindermann et al., 2022). Xie et al. (2023) propose to optimize data mixtures by training a proxy model to estimate the optimal data weight of each domain.

## 6 DISCUSSION

Limitation and future work. First, The method heavily depends on the availability of opensource pre-training datasets and models. If a specific domain is not covered in the pre-training data, the method may not recover performance well on that domain. Second, Due to computational constraints, we only experimented with a 7B parameter model as the source model. However, our method is highly generalizable and can be scaled up to larger models in future research.

Conclusion. This work proposes structured pruning as an efficient method for creating competitive smaller-scale LLMs. Our two-stage approach combines targeted structured pruning and continued pre-training (continued pre-training), and we introduce dynamic batch loading to improve pretraining data efficiency. We train a series of competitive Sheared-LLaMA models using a fraction of the compute required for standard pre-training. Our results show a promising path to producing low-cost, small LLMs when strong large-scale models are available. As more capable LLMs and larger pre-training datasets emerge, our method can easily extend to these advances to create even better small models.

## ACKNOWLEDGEMENTS

We express our gratitude to Sadhika Malladi, Tanya Goyal, Ofir Press, Adithya Bhaskar, and the Princeton NLP group for reviewing the paper and providing helpful feedback. We also thank the engineering team at MosaicML for their invaluable assistance with implementation specifics using the Composer package. Mengzhou Xia is supported by a Bloomberg Data Science Ph.D. Fellowship, and Tianyu Gao is supported by an IBM PhD Fellowship. This research is also supported by Mi crosoft Azure credits through the “Accelerate Foundation Models Academic Research” Initiative.

## REFERENCES

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Merouane Debbah, Etienne Goffinet, Daniel Heslow, Julien Launay, Quentin Malartic, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. Falcon-40B: an open large language model with state-of-the-art performance. 2023.

Anthropic. Introducing claude, 2023.

Brian R Bartoldson, Bhavya Kailkhura, and Davis Blalock. Compute-efficient deep learning: Algorithmic trends and opportunities. Journal of Machine Learning Research, 24:1–77, 2023.

Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy Blackburn. The pushshift reddit dataset. ArXiv, abs/2001.08435, 2020.

Edward Beeching, Clementine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Ra-´ jani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard. https: //huggingface.co/spaces/HuggingFaceH4/open\_llm\_leaderboard, 2023.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

Cody Blakeney, Jessica Zosa Forde, Jonathan Frankle, Ziliang Zong, and Matthew L Leavitt. Reduce, reuse, recycle: Improving training efficiency with distillation. arXiv preprint arXiv:2211.00683, 2022.

Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. In International Conference on Learning Representations, 2019.

Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, and Michael Carbin. The lottery ticket hypothesis for pre-trained bert networks. In Advances in Neural Information Processing Systems, 2020.

Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, et al. Symbolic discovery of optimization algorithms. arXiv preprint arXiv:2302.06675, 2023.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-´ efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. Model compression and hardware acceleration for neural networks: A comprehensive survey. Proceedings of the IEEE, 108(4): 485–532, 2020.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm.int8 (): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. arXiv preprint arXiv:2305.14387, 2023.

Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations, 2018.

Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot, 2023. arXiv preprint arXiv:2301.00774, 2023.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021.

Xinyang Geng and Hao Liu. Openllama: An open reproduction of llama, May 2023.

Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu. Efficient training of bert by progressively stacking. In International conference on machine learning, pp. 2337–2346. PMLR, 2019.

Kshitij Gupta, Benjamin Therien, Adam Ibrahim, Mats L Richter, Quentin Anthony, Eugene´ Belilovsky, Irina Rish, and Timothee Lesort. Continual pre-training of large language models:´ How to (re) warm your model? arXiv preprint arXiv:2308.04014, 2023.

Felix Hamborg, Norman Meuschke, Corinna Breitinger, and Bela Gipp. news-please: A generic news crawler and extractor. In Proceedings of the 15th International Symposium of Information Science, pp. 218–223, 2017.

Song Han, Huizi Mao, Dally, and William Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations, 2016.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, and Qun Liu. Dynabert: Dynamic bert with adaptive width and depth. Advances in Neural Information Processing Systems, 33: 9782–9793, 2020.

Ananya Harsh Jha, Dirk Groeneveld, Emma Strubell, and Iz Beltagy. Large language model distillation doesn’t need a teacher. arXiv preprint arXiv:2305.14864, 2023.

Angela H Jiang, Daniel L-K Wong, Giulio Zhou, David G Andersen, Jeffrey Dean, Gregory R Ganger, Gauri Joshi, Michael Kaminksy, Michael Kozuch, Zachary C Lipton, et al. Accelerating deep learning by focusing on the biggest losers. arXiv preprint arXiv:1910.00762, 2019.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. Tinybert: Distilling bert for natural language understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4163–4174, 2020.

Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, and Matt J Kusner. No train no gain: Revisiting efficient training algorithms for transformer-based language models. arXiv preprint arXiv:2307.06440, 2023.

Eldar Kurtic, Elias Frantar, and Dan Alistarh. Ziplm: Hardware-aware structured pruning of language models. arXiv preprint arXiv:2302.04089, 2023.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Associationfor Computational Linguistics, 7:452–466, 2019.

Franc¸ois Lagunas, Ella Charlaix, Victor Sanh, and Alexander M Rush. Block pruning for faster transformers. arXiv preprint arXiv:2109.04838, 2021.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499, 2021.

Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In International Conference on Learning Representations, 2016.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023.

Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, and Joey Gonzalez. Train big, then compress: Rethinking model size for efficient training and inference of transformers. In International Conference on machine learning, pp. 5958–5968. PMLR, 2020.

Hong Liu, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic second-order optimizer for language model pre-training. arXiv preprint arXiv:2305.14342, 2023.

Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pp. 3622– 3628, 2020.

Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In Proceedings of the IEEE international conference on computer vision, pp. 2736–2744, 2017.

Christos Louizos, Max Welling, and Diederik P Kingma. Learning sparse neural networks through l 0 regularization. In International Conference on Learning Representations, 2018.

Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In Proceedings of the IEEE international conference on computer vision, pp. 5058–5066, 2017.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. arXiv preprint arXiv:2305.11627, 2023.

Soren Mindermann, Jan M Brauner, Muhammed T Razzak, Mrinank Sharma, Andreas Kirsch, Win-¨ nie Xu, Benedikt Holtgen, Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, et al. Prioritized¨ training on points that are learnable, worth learning, and not yet learnt. In International Conference on Machine Learning, pp. 15630–15649. PMLR, 2022.

MosaicML. composer, 2021.

MosaicML. Introducing mpt-7b: A new standard for open-source, commercially usable llms, 2023. Accessed: 2023-05-05.

OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35, 2022.

Denis Paperno, German Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi,´ Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset:´ Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, 2016.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023.

John Platt and Alan Barr. Constrained differential optimization. In Neural Information Processing Systems, 1987.

Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106, 2021.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.

Victor Sanh, Thomas Wolf, and Alexander Rush. Movement pruning: Adaptive sparsity by finetuning. Advances in Neural Information Processing Systems, 33:20378–20389, 2020.

Noam M. Shazeer. Glu variants improve transformer. ArXiv, abs/2002.05202, 2020.

Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Joel Hestness, Natalia Vassilieva, Daria Soboleva, and Eric Xing. Slimpajama-dc: Understanding data combinations for llm training. arXiv preprint arXiv:2309.10818, 2023.

Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695, 2023.

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. Mobilebert: a compact task-agnostic bert for resource-limited devices. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 2158–2170, 2020.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model, 2023.

TogetherAI. Redpajama-incite-base-3b-v1, 2023a.

TogetherAI. Redpajama: An open source recipe to reproduce llama training dataset, 2023b.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and\` efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Trieu H. Trinh and Quoc V. Le. A simple method for commonsense reasoning. ArXiv, abs/1806.02847, 2018.

Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. arXiv preprint arXiv:2305.17926, 2023a.

Yunhe Wang, Hanting Chen, Yehui Tang, Tianyu Guo, Kai Han, Ying Nie, Xutao Wang, Hailin Hu, Zheyuan Bai, Yun Wang, et al. Pangu-pi: Enhancing language model architectures via nonlinearity compensation. arXiv preprint arXiv:2312.17276, 2023b.

Ziheng Wang, Jeremy Wohlwend, and Tao Lei. Structured pruning of large language models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6151–6162, 2020.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text, pp. 94–106, 2017.

Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. Advances in neural information processing systems, 29, 2016.

Mengzhou Xia, Zexuan Zhong, and Danqi Chen. Structured pruning learns compact and accurate models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1513–1528, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.107.

Mengzhou Xia, Mikel Artetxe, Chunting Zhou, Xi Victoria Lin, Ramakanth Pasunuru, Danqi Chen, Luke Zettlemoyer, and Veselin Stoyanov. Training trajectories of language models across scales. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 13711–13738, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.767.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pp. 38087–38099. PMLR, 2023.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. Doremi: Optimizing data mixtures speeds up language model pretraining. arXiv preprint arXiv:2305.10429, 2023.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, 2019.

Minjia Zhang and Yuxiong He. Accelerating training of transformer-based language models with progressive layer dropping. Advances in Neural Information Processing Systems, 33:14011– 14023, 2020.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. 2015 IEEE International Conference on Computer Vision (ICCV), pp. 19–27, 2015.

## CONTENTS

1 Introduction 1   
2 LLM-Shearing 2   
2.1 Targeted Structured Pruning 2   
2.2 Dynamic Batch Loading 4   
3 Experiments 5   
3.1 Setup 5   
3.2 Sheared-LLaMA Outperforms LMs of Equivalent Sizes 6   
4 Analysis 7   
4.1 Effectiveness of Dynamic Batch Loading 7   
4.2 Comparison to Other Pruning Approaches 8   
4.3 Additional Analysis . 8   
Related Work 9   
6 Discussion 9   
A A Detailed Exposition of Paramaterizing Pruning Masks 16   
B Reference Loss Predicted by Scaling Laws 16   
C Training Details 17   
D Model Configurations 17   
E Instruction Tuning 17   
F Additional Experiment Results 18   
F.1 Data Usage in Continued Pre-training 18   
F.2 Comparison to LLM-Pruner 19   
F.3 Coding and Math Reasoning 20   
F.4 Scaling Reference vs. Source Reference 20   
F.5 Pruning Pythia Models 21   
F.6 Pruning from LLaMA1 vs LLaMA2 22   
F.7 Comparison to Further Continual Pre-training INCITE-Base-3B 22   
F.8 Excluding Easy Domains During Pruning 23   
F.9 Inference Speed Analysis . 24   
G Frequently Asked Questions 24

## A A DETAILED EXPOSITION OF PARAMATERIZING PRUNING MASKS

The key idea behind the pruning algorithm is to apply masks to the model parameters. After learning a binary mask, it is equivalent to removing the corresponding parameters. The mask is parameterized using a hard concrete distribution introduced in Louizos et al. (2018). Given a masking variable z parameterized by α, the hard concrete distribution is defined as follows:

$$
\begin{array} { l } { { \displaystyle { u = \mathcal { U } ( 0 , 1 ) } } } \\ { { \displaystyle { s = \mathrm { S i g m o i d } \left( \frac { 1 } { \beta } \left( \log \frac { u } { 1 - u } + \log \alpha \right) \right) } } } \\ { { \displaystyle { \bar { s } = s ( \zeta - \gamma ) + \gamma } } } \\ { { \displaystyle z = \operatorname* { m i n } ( 1 , \operatorname* { m a x } ( 0 , \bar { s } ) ) } } \end{array}
$$

where $\mathcal { U }$ is the uniform distribution, $\beta$ is a temperature parameter, s is a relaxed binary mask that conforms to the hard concrete distribution, and $\zeta$ and γ are the bounds of the hard concrete distribution. The hard concrete distribution serves as a continuous relaxation of the binary mask, allowing the model to learn the binary mask in a continuous manner during training. The effectiveness of this trick in learning sparse structures in neural networks has been demonstrated in previous studies (Wang et al., 2020; Xia et al., 2022). In our experiments, we set $\beta = 0 . 8 3 , \zeta = 1 . 1 , \mathrm { a n d } \gamma = - 0 . 1$

To enforce the sparsity constraint, the masks are trained alongside with Lagrange multipliers λ, as defined in Equation (1). After pruning, the parameters corresponding to the learned masks are removed to ensure that the resulting model shape matches the target model. In practical implementations, we set a threshold to binarize the masks. Due to the adoption of the hard concrete distribution, the masks typically converge to binary values that match the target model shape in most cases, thereby avoiding any inconsistencies. However, in rare instances where the masks do not converge to exactly 0 or 1, the masking variables need to be absorbed into the resulting model parameters.

As discussed in Section 2, we apply masks to heads, intermediate dimensions, layers and hidden dimensions. For heads, we simply multiply the head output by the mask. For intermediate dimensions, we apply the mask to the intermediate output. For layers, we apply the mask to the layer output. For hidden dimensions, we apply the mask to both the head and mlp output. Applying the mask to outputs is equivalent to removing the corresponding parameters. Please refer to composer llama.py for more details.

## B REFERENCE LOSS PREDICTED BY SCALING LAWS

The scaling law of language modeling is a function of model size N and dataset size $D \colon$

$$
L ( N , D ) = E + \frac { A } { N ^ { \alpha } } + \frac { B } { D ^ { \beta } }
$$

where E captures the loss for the true language distribution in an ideal generation process, and $A , \alpha , B , \beta$ are scaling factors related to model scale or data size. Models in the same model family are usually trained with the same amount of tokens on the same data distribution. In this case, we need a minimum of three models to estimate the constant $\textstyle E + { \frac { B } { D ^ { \beta } } } , A$ and $\alpha .$ . If the models are trained with different amount of tokens, we can estimate $E , A , \alpha , B , \beta$ with a minimal of 5 models. Note that we will estimate the scaling factors for each domain seperately.

LLAMA2 models have been trained on the same 2T tokens (Touvron et al., 2023b). We take the LLAMA2-7B, LLAMA2-13B, and LLAMA2-70B checkpoints, evaluate them on each domain’s validation set, and fit the scaling factors with the corresponding loss. Given the limited data points for estimating the scaling law constant, the projected loss of a hypothetical LLaMA-2.7B model may be biased compared to the true value. Table 6 presents the predicted loss. The evaluation process takes less than 4 A100 GPU hours to complete.

Table 6: Estimated reference loss of hypothetical LLaMA2-1.3B and LLaMA2-2.7B models.
<table><tr><td></td><td>CC</td><td>GitHub</td><td>Book</td><td>StackExchange</td><td>Wiki</td><td>ArXiv</td><td>C4</td></tr><tr><td>LLaMA2-1.3B</td><td>1.964</td><td>0.746</td><td>2.139</td><td>1.612</td><td>1.759</td><td>1.445</td><td>2.125</td></tr><tr><td>LLaMA2-2.7B</td><td>1.871</td><td>0.688</td><td>2.033</td><td>1.535</td><td>1.630</td><td>1.356</td><td>2.033</td></tr></table>

## C TRAINING DETAILS

We present the hyperparameters used in our experiments in Appendix C. We use fully sharded data parallel (Zhao et al., 2023) to train our models in parallel. We use FlashAttention V1 (Dao et al., 2022) to speed up training. We use a cosine learning rate scheduler and decay the learning rate to a minimum of 10% of the peak value. We conduct some preliminary experiment to determine the peak learning rate for learning the masking variables and Lagrange multiplers, and we find that a learning rate of 1.0 works well for pruning. We do not tune any other hyper-parameters. The throughput is dependent on the implementations and we believe that our throughput can be further improved by adopting more advanced recent optimizations such as FlashAttention V2 (Dao et al., 2022) and a more recent version of Composer (MosaicML, 2021).

Table 7: Training hyper-parameters and throughput.
<table><tr><td></td><td>Pruning</td><td>Contined Pre-training</td></tr><tr><td>Training budget</td><td>0.4B</td><td>50B</td></tr><tr><td>Learning rate of z, φ, λ</td><td>1.0</td><td></td></tr><tr><td>Learning Rate of θ</td><td>0.0001</td><td>0.0001</td></tr><tr><td>LR warmup ratio</td><td>10%</td><td>3%</td></tr><tr><td>Batch size (tokens)</td><td>131K</td><td>1M</td></tr><tr><td>Evaluation interval m (steps)</td><td>50</td><td>400</td></tr><tr><td>Steps</td><td>3,200</td><td>51,200</td></tr><tr><td># GPUs</td><td>8</td><td>16</td></tr><tr><td>Throughput (tokens/s)</td><td>15K</td><td>145K (1.3B) / 77K (2.7B)</td></tr></table>

## D MODEL CONFIGURATIONS

In this section, we provide the model configurations for both our Sheared-LLaMA models and the baseline models, as illustrated in Table 8. Our design closely adheres to the architecture of Pythia-1.4B and INCITE-Base-3B, albeit with some nuanced distinctions. A noteworthy difference is found in the intermediate size of Sheared-LLaMA, which is a consequence of its lineage from LLaMA2- 7B. Notably, LLaMA2-7B employs a GLU variant (Shazeer, 2020) within its feed-forward layer, comprising a gate matrix, an upward-projection matrix, and a downward-projection matrix. In contrast, other models employ the conventional double-matrix feed-forward layer structure. Furthermore, we acknowledge that the shearing algorithm will have to inherit the head dimension of the source model. Instead of explicitly specifying the number of heads based on existing language models, we set the target number of heads to be the target hidden dimension divided by the head dimension of the source model.

## E INSTRUCTION TUNING

We evaluate our models on instruction tuning and fine-tune both Sheared-LLaMA and baseline models on 10,000 instruction-response pairs sampled from the ShareGPT dataset<sup>10</sup>. For evaluation, we sample another 1,000 instructions from ShareGPT, generate responses from our fine-tuned models and other baseline models, and use GPT-4 as an evaluator to compare the two responses (Dubois et al., 2023). We report the win rate of our model compared to the baseline model.

During instruction tuning training, the instruction is prepended with “You are a helpful assistant. Write a response that appropriately completes the request.”. For evaluating the instruction tuning generations, Wang et al. (2023a) observes using GPT models as a judge could change its preference when swapping the presentation order of the two outputs. Therefore, we compare each output pair twice by swapping the presentation order of the two outputs and finally report the average win-rate of the two rounds to eliminate the position bias.

Table 8: Model configurations of our Sheared-LLaMA and baseline models.
<table><tr><td>Model</td><td>#Param</td><td>#Layers</td><td>Hidden</td><td>Intermediate</td><td>#Heads</td><td>Head Dim</td></tr><tr><td>OPT-1.3B</td><td>1.3B</td><td>24</td><td>2048</td><td>8192</td><td>32</td><td>64</td></tr><tr><td>Pythia-1.4B</td><td>1.4B</td><td>24</td><td>2048</td><td>8192</td><td>16</td><td>128</td></tr><tr><td>TinyLlama-1.1B</td><td>1.1B</td><td>22</td><td>2048</td><td>5632</td><td>32</td><td>64</td></tr><tr><td>Sheared-LLaMA-1.3B</td><td>1.3B</td><td>24</td><td>2048</td><td>5504</td><td>16</td><td>128</td></tr><tr><td>OPT-2.7B</td><td>2.7B</td><td>32</td><td>2560</td><td>10240</td><td>32</td><td>80</td></tr><tr><td>Pythia-2.8B</td><td>2.8B</td><td>32</td><td>2560</td><td>10240</td><td>32</td><td>80</td></tr><tr><td>INCITE-Base-3B</td><td>2.8B</td><td>32</td><td>2560</td><td>10240</td><td>32</td><td>80</td></tr><tr><td>OpenLLaMA-3B</td><td>2.7B</td><td>26</td><td>3200</td><td>8640</td><td>32</td><td>100</td></tr><tr><td>Sheared-LLaMA-2.7B</td><td>2.7B</td><td>32</td><td>2560</td><td>6912</td><td>20</td><td>128</td></tr><tr><td>LLaMA2-7B</td><td>6.7B</td><td>32</td><td>4096</td><td>11008</td><td>32</td><td>128</td></tr></table>

We randomly select an output generated by Sheared-LLaMA-1.3B and Sheared-LLaMA-2.7B in response to a given instruction, and present the generations in Table 10. Our findings demonstrate that, after instruction tuning, Sheared-LLaMA-2.7B consistently produces long, coherent, and informative outputs in response to the instruction.

Table 9: Training hyper-parameters for instruction tuning.
<table><tr><td></td><td>Instruction Tuning</td></tr><tr><td>Learning Rate of θ</td><td>5e − 5</td></tr><tr><td>LR warmup ratio</td><td>3%</td></tr><tr><td>Batch size (tokens)</td><td>128</td></tr><tr><td># GPUs</td><td>8</td></tr></table>

## F ADDITIONAL EXPERIMENT RESULTS

## F.1 DATA USAGE IN CONTINUED PRE-TRAINING

Figure 6 illustrates the evolution of domain weights throughout the training process and the final cumulative data usage for each domain. The trajectory reveals that the domain weights stabilize after approximately 30% of the training. In contrast to other domains, Wikipedia exhibits an anomalous spike early in the training process. The remaining domains show a steady, monotonic change in data loading over time, as expected.

![](images/148cea7a164bfc0bb203f95a1ea2523af689cf11fb7951dcdb9b5f24f8696131.jpg)

![](images/6c27e090a5d1178c60502c0f8267f408b7d3c8b39d765eb557631c5414cf06a3.jpg)  
Figure 6: Left: Data weight of each batch during the continued pre-training stage. Right: Cumulative data usage for each domain.

Table 10: An example output from Sheared-LLaMA-1.3B and Sheared-LLaMA-2.7B. It shows that Sheared-LLaMA can generate long, coherent, and informative responses after instruction tuning.

Prompt: Act as semiconductor industry analyst   
Sheared-LLaMA-1.3B: As a semiconductor industry analyst, I would be   
responsible for conducting market research and analysis to help   
companies in the industry understand and respond to market trends,   
customer preferences, and competitive dynamics. This would include   
analyzing market data such as sales, market share, and industry trends,   
as well as conducting interviews with industry leaders, executives, and   
customers to gain insights into customer needs and preferences. I would   
also be responsible for developing and implementing marketing strategies   
to promote products and services, and for identifying and pursuing new   
business opportunities.   
Sheared-LLaMA-2.7B: As a semiconductor industry analyst, my primary   
responsibility is to provide insightful and accurate analysis on the   
global semiconductor market. This includes evaluating the performance   
of individual companies, industries, and trends within the market.   
Here are some key areas where I would focus my analysis:   
1. Market size and growth: I would track the size of the global   
semiconductor market and analyze the growth prospects for each segment,   
such as memory, logic, and sensors.   
2. Competitive landscape: I would assess the competitive landscape,   
including the top players, their market share, and strategic moves.   
3. Technology trends: I would monitor the development of new   
technologies, such as 5G, AI, and IoT, and their impact on the   
semiconductor industry.   
4. Industry trends: I would track industry trends, such as the shift   
towards sustainable technologies, and evaluate their impact on the   
market.   
5. Market forecasts: I would provide forecasts for the semiconductor   
market, including revenue and unit shipment growth, based on a range of   
scenarios and assumptions.   
In addition to these areas, I would also keep track of industry news and   
developments, and provide insights on their impact on the market.

## F.2 COMPARISON TO LLM-PRUNER

To ensure a fair comparison with the LLM-Pruner approach, we match the parameters (excluding embeddings) to be roughly the same as our final model (1.23B), as embedding sizes do not affect inference speed. We continue pre-training the pruned models derived from both LLM-Pruner and our proposed targeted structured pruning. The total number of tokens for pruning and continued pre training is controlled to be the same, and data from the RedPajama dataset is used directly without applying dynamic batch loading. We demonstrate that our proposed targeted structured pruning is a better approach compared to LLM-Pruner from three aspects: the loss trajectory, the mode architecture, and the inference speed.

In terms of loss trajectory, Figure 7 shows that our proposed targeted structured pruning achieves a lower loss than LLM-Pruner when consuming the same amount of data.

Table 11 compares the model configurations for an LLM-Pruner pruned model and our pruned model. The LLM-Pruner model has an unconventional architecture where the intermediate size is smaller than the hidden size, largely due to the algorithm’s inability to prune the hidden dimension and layers, revealing a limitation of LLM-Pruner.

In terms of training throughput and inference speed, we find Sheared-LLaMA structures run more efficiently than LLM-Pruner models. We performed an inference speed analysis comparing LLMpruner and Sheared-LLaMA’s model architectures using a single A100 GPU to generate up to 2048 tokens. As shown in Table 12, our pruned model architecture is significantly more efficient than

LLM-Pruner at inference time. Additionally, LLM-Pruner’s model architecture introduces substantial overhead during continued pretraining (Measured with 16 A100 80GB GPUs.), with a training throughput of around 60% of Sheared-LLaMA’s. Overall, our Sheared-LLaMA architecture enables higher throughput for both inference and continued training compared to LLM-Pruner.

In summary, we have demonstrated that at the same parameter scale, our pruning method produces a model that has a lower perplexity (loss), a more reasonable final model architecture, and a faster inference speed. We have effectively shown our targeted structured pruning algorithm to be more effective for large-scale LLM pruning compared to LLM-Pruner.

![](images/c66c8a75258afb713db9ee8434859fcab1840ba3de3e8bb4de690a1998cd54d5.jpg)  
Figure 7: The loss of LLM-Pruner and Sheared-LLaMA during the continued pre-training stage. Note that we exclude dynamic batch loading and use the same data distribution for training both models for a fair comparison.

Table 11: Model structure of Pythia-1.4B, LLM-pruner (1.6B), and Ours (1.3B).
<table><tr><td>Layers</td><td></td><td></td><td>Heads Head size Intermediate size Hidden size Params</td><td></td><td></td><td></td></tr><tr><td>Pythia-1.4B</td><td>24</td><td>16</td><td>128</td><td>8192</td><td>2048</td><td>1.4B</td></tr><tr><td>LLM-pruner (1.6B)</td><td>32</td><td>7</td><td>128</td><td>2201</td><td>4096</td><td>1.6B</td></tr><tr><td>Sheared-LLaMA (1.3B)</td><td>24</td><td>16</td><td>128</td><td>5504</td><td>2048</td><td>1.3B</td></tr></table>

Table 12: Training throughput and generation speed of LLM-pruner (1.6B) and Sheared-LLaMA (1.3B). With a similar parameter count, our pruned model structure has a lower perplexity when fine-tuned with the same amount of tokens (around 6B tokens). Yet our pruned model architectures are way more efficient for both training and inference.
<table><tr><td></td><td>Generation Speed</td><td>Training Throughput</td><td>PPL</td></tr><tr><td>LLM-Pruner</td><td>43 tokens/s</td><td>83K tokens/s</td><td>7.09</td></tr><tr><td>Sheared-LLaMA</td><td>58 tokens/s</td><td>139K tokens/s</td><td>6.85</td></tr></table>

## F.3 CODING AND MATH REASONING

We examine the math and coding abilities of our pruned models compared to other language models. We find that the math ability of existing 3B parameter models, including Sheared-LLaMA, is still far below that of larger models. We also find that Sheared-LLaMA’s coding ability lags behind models known to be trained on more code data, like Pythia-1.4B and Open-LLaMA-3B-v2. Sheared LLaMA’s coding ability likely comes from the original LLaMA2 model, speculated to have used more code data, and the minimal code data used in our pruning experiments.

## F.4 SCALING REFERENCE VS. SOURCE REFERENCE

Figure 8 This section compares the performance of Sheared-LLaMA when trained with the scaling reference and the source reference in dynamic batch loading. The scaling reference uses the predicted loss from the scaling law as the reference loss, while the source reference uses the loss of the

Table 13: Evaluation results on GSM8K and HumanEval and training percentage and tokens in ArXiv and GitHub.
<table><tr><td>Models</td><td>GSM8K (8) EM</td><td colspan="2">HumanEval</td><td>ArXiv Pass@1 Pass@5 Percentage</td><td>Github Percentage</td><td>ArXiv Tokens</td><td>GitHub Tokens</td></tr><tr><td>LLaMA2-7B</td><td>13.7</td><td>12.8</td><td>23.8</td><td>–</td><td>一</td><td>1</td><td></td></tr><tr><td>OPT-2.7B</td><td>0.1</td><td>0.0</td><td>0.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Pythia-2.8B</td><td>1.7</td><td>5.1</td><td>14.6</td><td>9.0%</td><td>7.6%</td><td>26.9</td><td>22.8</td></tr><tr><td>INCITE-Base-3B</td><td>1.8</td><td>4.3</td><td>4.9</td><td>2%</td><td>4.5%</td><td>16.0</td><td>36.0</td></tr><tr><td>Open-LLaMA-3B-v1</td><td>2.5</td><td>0.0</td><td>1.2</td><td>2%</td><td>4.5%</td><td>20.0</td><td>45.0</td></tr><tr><td>Open-LLaMA-3B-v2</td><td>2.7</td><td>10.4</td><td>20.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Sheared-LLaMA-2.7B (Source)</td><td>2.7</td><td>3.7</td><td>5.5</td><td>0.7%</td><td>0.4%</td><td>0.3</td><td>0.2</td></tr><tr><td>Sheared-LLaMA-2.7B (Scaling)</td><td>2.4</td><td>4.9</td><td>9.2</td><td>1.0%</td><td>0.8%</td><td>0.5</td><td>0.4</td></tr></table>

source model as the reference loss. Although both methods efficiently train the model, the scaling reference consistently achieves slightly better downstream performance.

![](images/ab9a76b0fa9610f51109c316e2940d02810cf3ab046c488c3a0c83a11d5661c6.jpg)  
Figure 8: Average downstream peformance of Sheared-LLaMA-1.3B with the scaling reference and the source reference.

## F.5 PRUNING PYTHIA MODELS

During the initial development of the approach, we experimented with a smaller-scale model on Pythia (Biderman et al., 2023), a series of open-source models with open-source training data across scales from 70M to 13B. We took the Pythia-440M model, pruned it down to 160M parameters, and continued pre-training it using Pythia models’ training data Gao et al. (2020). Specifically, we used 0.4B tokens for pruning and 33B tokens (32,000 steps) for continued pre-training of the pruned model. Table 14 shows that the pruned model achieves a lower perplexity than the original model, and continued pre-training further improves performance. Notably, with minimal compute consumption (10B tokens), pruning a Pythia-410M model reaches roughly the same performance as pretraining Pythia-160M from scratch. Adding more tokens further enhances the performance.

Table 14: Zero-shot performance of Pythia-160M and Sheared-Pythia.
<table><tr><td colspan="2">Training Tokens</td><td>Performance</td></tr><tr><td>Pythia-160M</td><td>300B</td><td>43.56</td></tr><tr><td>Sheared-Pythia</td><td>(300B) + 10B</td><td>43.51</td></tr><tr><td>Sheared-Pythia</td><td>(300B) + 33B</td><td>45.78</td></tr></table>

Additionally, we compared Sheared-Pythia-160M against keeping pre-training the Pythia-160M model with the same amount of tokens. From Figure 9, we can see that continuing pre-training Pythia-160M starts off performing better, however, the Sheared-Pythia-160M learns faster and eventually exceeds the performance of continuing pretraining on Pythia-160M. These are some very preliminary results we see in this particular setting.

![](images/09f9a157029da6e26e8015113e64ba6ca5b38c9ad0cb0d5e7eb33c9cc968076a.jpg)  
Figure 9: The downstream performance of continued pre-training Pythia-160M and Sheared-Pythia-160M. Sheared-Pythia-160M eventually outperforms the performance of continued pre-training Pythia-160M.

We think that the benefit of pruning a larger model will be even more significant, based on the conclusions from a previous work (Li et al., 2020) showing that pruning larger than compress leads to better performance as the larger models are easier to optimize. However, we’d like to defer more detailed analysis to future work.

## F.6 PRUNING FROM LLAMA1 VS LLAMA2

This section compares the performance of pruning from LLaMA1 and LLaMA2. Both models demonstrate strong downstream task performance, although pruning from LLaMA2 unsurprisingly yields a consistent advantage. However, it is worth noting that the performance difference between the two is not very large.

![](images/b47604b8718f742e92a35af21eb12e8b8884d00277dc317be2861324709b3a93.jpg)  
Figure 10: A comparison between pruning from LLaMA1 and LLaMA2 with dynamic loading for 1.3B.

## F.7 COMPARISON TO FURTHER CONTINUAL PRE-TRAINING INCITE-BASE-3B

We examine if pruning produces a better initialization for continued pre-training than an existing LLM of equivalent size by comparing the performance of a continually pre-trained INCITE-Base-3B model and Sheared-LLaMA-2.7B. We present the loss curves in Figure 11 and the downstream performance in Figure 12. INCITE-Base-3B model starts with higher task accuracy but plateaus after training, while Sheared-LLaMA rapidly improves and surpasses the INCITE-Base-3B model, suggesting that pruned models from a strong base model serve as a better initialization.<sup>11</sup>

![](images/39f6b0c74d68b68a69c27f9f4778a8bbebb9c52c14f941ce9e8e1561ef95727a.jpg)

![](images/9b5fb04b5a008fdcdb446810fdfe8e1a61cdf19c90ac82600939a86bf273109d.jpg)  
Figure 11: The loss of continued pre-training Figure 12: Average downstream performance INCITE-3B and our pruned LLaMA model. of continuing pre-training Sheared-LLaMA vs Both models have around 2.7B parameters. INCITE-Base-3B.

We used a learning rate 1e − 5 for continued pre-training INCITE-Base-3B, along with a scheduler to warm up the learning rate to 1e − 5 in the first 3% of the training steps, and follows a cosine decay schedule. In hindsight, how we continued pre-training the INCITE-Base-3B model may not be optimal according to recent research (Gupta et al., 2023).

## F.8 EXCLUDING EASY DOMAINS DURING PRUNING

During the development of this project, we explored an easy and intuitive idea to address the imbalanced loss decreasing rate during pruning and continued pre-training. Specifically, we excluded GitHub, StackExchange, and ArXiv data during pruning since these three domains’ losses decrease the fastest. We pruned LLaMA1-13B down to 7B using a composite dataset of C4, CC, Wiki, and Books, with a heuristically constructed proportion of 40%, 40%, 10%, 10%, respectively. We then continued pre-training the pruned model on the RedPajama dataset, which includes the excluded domains during pruning.

The results showed that the perplexity difference was more even across domains when pruning without using data from these three domains. However, after continued pre-training with all data from the seven domains in the RedPajama dataset, the loss disparity grew, with the GitHub difference being much smaller than domains like C4. These results demonstrate that simply excluding the domains that are easy to recover during the pruning stage does not inherently resolve the imbalance of loss difference across domains.

This set of experiments motivated us to develop dynamic batch loading as a more effective and principled approach to address the domain-specific loss disparities that arise during pruning and continued pre-training.

Table 15: Pruning LLaMA1-13B with a composite of 40% of CC, 40% of C4, 10% of Books and 10% of Wikipedia to a 7B model. We present the domain loss of the source model (LLaMA1-13B), the loss of the pruned model and the loss after continued pre-training of the pruned model. The loss differentce from the target model (LLaMA1-7B) is more balanced after pruning, but more disparate after continued pre-training with all the domains.
<table><tr><td></td><td>CC</td><td>GitHub</td><td>Book</td><td>StackExchange</td><td>Wikipedia</td><td>ArXiv</td><td>C4</td></tr><tr><td>LLaMA1-13B</td><td>1.7585</td><td>0.6673</td><td>1.9499</td><td>1.4207</td><td>1.4331</td><td>1.3855</td><td>1.8619</td></tr><tr><td>LLaMA1-7B</td><td>1.8366</td><td>0.7108</td><td>2.0322</td><td>1.5112</td><td>1.5291</td><td>1.4340</td><td>1.9331</td></tr><tr><td>Pruned model (w/o three domains) diff from LLaMA1-7B</td><td>2.1849</td><td>1.0971</td><td>2.3726</td><td>1.9080</td><td>2.1151</td><td>1.7542</td><td>2.3187</td></tr><tr><td>Continued Pretraining (w RP)</td><td>0.3483</td><td>0.3863</td><td>0.3404</td><td>0.3968</td><td>0.5860</td><td>0.3202</td><td>0.3857</td></tr><tr><td>diff from LLaMA1-7B</td><td>1.8344</td><td>0.6325</td><td>2.0984</td><td>1.4542</td><td>1.4549</td><td>1.4460</td><td>2.0395</td></tr><tr><td></td><td>-0.0022</td><td>-0.0783</td><td>0.0661</td><td>-0.0570</td><td>-0.0743</td><td>0.0120</td><td>0.1064</td></tr></table>

## F.9 INFERENCE SPEED ANALYSIS

In this section, we analyze the inference speed of different pruning approaches, including the following models:

• The source model, i.e., LLaMA2-7B.

• Sheared-LLaMA-1.3B and Sheared-LLaMA-2.7B.

• Wanda pruning (Sun et al., 2023) to prune LLMs into a semi-structured 2:4 and 4:8 sparsity pattern in one-shot.

• LLM-Pruner (Ma et al., 2023), which produces a model with the same number of nonembedding parameters as Sheared-LLaMA.

We use an A100 GPU to test the generation speed (tokens/second) of all these pruned models. We generate up to 2048 tokens with a batch size of 1. We present the results in Table 16. Sheared-LLaMA’s speed is better than that of LLM-Pruner, largely due to the more optimized resulting architecture. As shown in Table 11, LLM-pruner produces a model structure with a smaller intermediate size than the hidden size, which goes against the transformer designs where the intermediate size is at least 3-4 times the hidden size.

Wanda-type semi-structured pruning also achieves inference speedup compared to the source model. However, it is not as fast as small dense models and is less flexible because inference speedup is only feasible when the sparsity is at 50%.

Table 16: Inference speed (tokens/s) of different pruning approaches.
<table><tr><td>Model</td><td>Throughput</td></tr><tr><td></td><td>7B</td></tr><tr><td>LLaMA-7B</td><td>37</td></tr><tr><td></td><td>1.3B 2.7B</td></tr><tr><td>LLM Pruner Sheared-LLaMA</td><td>41 40 62 47</td></tr><tr><td></td><td>50% sparsity</td></tr><tr><td>Wanda (2:4)</td><td>42</td></tr><tr><td>Wanda (4:8)</td><td>42</td></tr></table>

## G FREQUENTLY ASKED QUESTIONS

In this section, we provide answers to frequently asked questions about our work.

▷ Is it fair to say that Sheared-LLaMA models can be produced using only 50B tokens, even though the source model (LLaMA2) was trained on 2T tokens?

At the time of our paper submission, there were no models sufficiently trained for 2T tokens at the 1.3B and 2.7B scale to allow for a fair comparison. However, the recently released TinyLlama-1.1B models, trained on 3T tokens, provide a suitable point of reference. We observe that the performance of TinyLlama-1.1B is comparable to Sheared-LLaMA-1.3B on downstream benchmarks when used as base models, and a similar observation can be found in Wang et al. (2023b). Considering that TinyLlama-1.1B is trained with 3T tokens, which exceeds the total amount of pre-training and pruning used by Sheared-LLaMA-1.3B (2T for pre-training the source model, and 50.4B for pruning and continued training), we regard this as strong evidence suggesting that pruning might be an intrinsi cally more efficient and effective approach to training moderate-sized LMs.

## ▷ How is dynamic batch loading different from Doremi (Xie et al., 2023)?

Dynamic batch loading and Doremi share the same principle, which adjusts the data distribution of each domain based on the model’s loss using an exponential ascent algorithm. However, dynamic batch loading offers a more flexible and less complex approach that can be applied to various scenarios.

Doremi follows a multi-step process: (1) Train a reference model. (2) Train a proxy model to estimate the proportion of data from each domain by adjusting the proportion based on the proxy model’s loss. (3) Train the final model using the estimated data distribution. In contrast, dynamic batch loading can be directly applied to any model without the need for a reference or a proxy model. Dynamic batch loading begins by deriving a reference loss based on a fixed evaluation set. This reference loss can be estimated using scaling laws or simply by using the source model’s evaluation loss. During training, the data proportion is adjusted in real-time based on the periodically measured evaluation loss. The dynamic batch loading process can be seamlessly integrated into the standard pre-training pipeline, as evaluating the loss is computationally efficient and does not introduce significant overhead. Although dynamic batch loading relies on a fixed evaluation set, which may not fully represent the model’s performance on the entire dataset, this issue can be mitigated by periodically updating the evaluation set during training.

## ▷ When multiple source model sizes are available, how do you choose the source model size for pruning?

Determining the optimal source model size for pruning is challenging. However, we can perform a thought experiment by considering each parameter as a uniform ”unit of information.” For instance, if a source model with 7B parameters is trained using 2T tokens, we can assume that each parameter carries approximately 285 tokens of information, assuming a uniform distribution of information across the parameters. When randomly pruning this model down to 1.3B parameters, the total amount of information is reduced to $1 . 3 \mathrm { B } \times 2 8 5 = 0 . 3 7 \mathrm { T }$ tokens. In contrast, if we prune a 13B model (also trained with 2T tokens) down to 1.3B parameters, the total amount of information is reduced to $1 . 3 \mathbf { B } \times ( 2 \mathrm { T } / 1 3 \mathbf { B } ) = 0 . 2 \mathrm { T }$ tokens. Although this estimation is rough, it suggests that pruning from a larger model may be less effective, especially when the source models are trained with the same number of tokens. It is important to note that this is a simplified estimate, and the assumption of uniform information distribution across parameters may not hold in practice. Moreover, the structured pruning process itself clearly breaks this assumption. Nonetheless, this thought experiment provides a general sense of how the source model size can impact the effectiveness of pruning.