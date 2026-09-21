# What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?

The BigScience Architecture & Scaling Group

Thomas Wang $^{1*}$ Adam Roberts $^{2*}$

Daniel Hesslow3 Teven Le Scao<sup>1</sup> Hyung Won Chung<sup>2</sup>

Iz Beltagy<sup>4</sup> Julien Launay<sup>3,5†</sup> Colin Raffel<sup>1†</sup>

$^{1}$ Hugging Face $^{2}$ Google $^{3}$ LightOn

4Allen Institute for AI 5LPENS, Ecole Normale Supérieure

# Abstract

Large pretrained Transformer language models have been shown to exhibit zero-shot generalization, i.e. they can perform a wide variety of tasks that they were not explicitly trained on. However, the architectures and pretraining objectives used across state-of-the-art models differ significantly, and there has been limited systematic comparison of these factors. In this work, we present a large-scale evaluation of modeling choices and their impact on zero-shot generalization.

In particular, we focus on text-to-text models and experiment with three model architectures (causal/non-causal decoder-only and encoder-decoder), trained with two different pretraining objectives (autoregressive and masked language modeling), and evaluated with and without multitask prompted finetuning. We train models with over 5 billion parameters for more than 170 billion tokens, thereby increasing the likelihood that our conclusions will transfer to even larger scales.

Our experiments show that causal decoder-only models trained on an autoregressive language modeling objective exhibit the strongest zero-shot generalization after purely unsupervised pretraining. However, models with non-causal visibility on their input trained with a masked language modeling objective followed by multitask finetuning perform the best among our experiments. We therefore consider the adaptation of pretrained models across architectures and objectives.

We find that pretrained non-causal decoder models can be adapted into performant generative causal decoder models, using autoregressive language modeling as a downstream task. Furthermore, we find that pretrained causal decoder models can be efficiently adapted into non-causal decoder models, ultimately achieving competitive performance after multitask finetuning. Code and checkpoints are available at https://github.com/bigscience-workshop/architecture-objective.

arXiv:2204.05832v1 [cs.CL] 12 Apr 2022

*Equal contribution.

$\dagger$ Equal supervision.

Individual contributions outlined in Appendix A

![](dt=2026-05-29/ht=15/55f83087be7f8fc915236b1ae2910da5f6483ea0a8b11a37be8fdbdca5d90b92.jpg)

# 1 Introduction

Large language models (LLMs) pretrained on unstructured text data have been shown to be capable of performing a wide variety of text processing tasks without additional training. This ability has been referred to as zero-shot generalization since these models are typically pretrained with a self-supervised objective that is not specific to a downstream task. Zero-shot generalization is particularly useful because it does not require any additional data or training in order to enable the model to perform a given task.

As such, there has been an explosion of work on developing LLMs and training techniques that produce strong zero-shot generalization [Brown et al., 2020, Wang and Komatsuzaki, 2021, Du et al., 2021, Lin et al., 2021, Rae et al., 2021, Hoffmann et al., 2022, Chowdhery et al., 2022]. One recent line of work [Sanh et al., 2021, Wei et al., 2021, Xu et al., 2022] has demonstrated that adding an explicit multitask finetuning stage on an ensemble of prompted tasks after pretraining can significantly boost the zero-shot capabilities of LLMs.

Modern LLMs are based on the Transformer architecture [Vaswani et al., 2017]. While the original Transformer included a separate encoder that processes input text and a decoder that generates target text, most recent LLMs are causal decoder-only (CD) models trained to autoregressively predict a text sequence [Liu et al., 2018, Radford et al., 2018, Al-Rfou et al., 2019]. In contrast with this trend, Raffel et al. [2020] has shown that encoder-decoder (ED) models outperform decoder-only LLMs for transfer learning (i.e. where a pretrained model is finetuned on a single downstream task).

Non-causal decoders (ND) [Liu et al., 2018, Dong et al., 2019] use a modified attention mask to bridge the gap between decoder-only and encoder-decoder models. However, they have seen limited adoption. Recently, Sanh et al. [2021] proposed a multitask finetuned encoder-decoder LLM that outperforms decoder-only models on zero-shot generalization, despite being an order of magnitude smaller. Concurrent work also demonstrated this approach with a decoder-only model [Wei et al., 2021].

This begs the question as to whether an encoder-decoder or a decoder-only would be a better choice for zero-shot generalization, especially if used in conjunction with multitask finetuning.

Transformer models can be trained with a variety of unsupervised training objectives. Typically, decoder-only LLMs are pretrained using a full language modeling (FLM) objective with a loss computed on all tokens [Dai and Le, 2015, Radford et al., 2018], and encoder-decoder models with a masked language modeling (MLM) objective [Taylor, 1953, Devlin et al., 2018], such as span corruption [Raffel et al., 2019, Joshi et al., 2020]. It has repeatedly been shown that an MLM objective produces a better pretrained model for subsequent supervised finetuning in transfer learning settings [Devlin et al.

, 2018, Lample and Conneau, 2019, Voita et al., 2019, Raffel et al., 2020]. The frequent use of the standard full language modeling objective nonetheless could be attributed to the fact that it lends itself to straightforward application of the model to many downstream tasks [Radford et al., 2019]. Still, the effectiveness of MLM in the transfer learning setting suggests it could create LLMs that are better suited to multitask finetuning. Notably, the T0 model of Sanh et al.

[2021] used an MLM pretraining objective, which may have contributed to its strong performance relative to larger models trained with only an FLM objective. Recently, Lester et al. [2021] also proposed introducing an adaptation stage (i.e. extending pretraining but with a different objective) to enable an MLM model to perform prompted text generation tasks, bridging the gap across objectives.

2

These results indicate a need for a more systematic analysis of which architecture and pretraining objective pair produces LLMs with the strongest zero-shot generalization capabilities. Past studies on architectures and objectives for language models [e.g. Narang et al., 2021, Raffel et al., 2020] have focused mainly on the transfer learning setting, with models that were orders of magnitude smaller than the current state-of-the-art.

Furthermore, recent results demonstrating the effectiveness of multitask finetuning raise the question of which architecture and pretraining objective is best suited to that promising setting. Finally, novel adaptation practices also question the rigidity of these architecture and objective choices, and whether it is possible to efficiently convert pretrained models from one architecture to another. We propose to fill this gap with the following contributions.

Large-scale systematic study. We undertake a study of architecture and pretraining objective combinations for LLMs with a focus on zero-shot generalization. We consider decoder-only and encoder-decoder models using standard, prefix, and masked language modeling, spanning six $\langle$ architecture, objective\rangle pairs. We also evaluate performance with and without multitask finetuning.

In hopes of producing insights that transfer to very large models, we undertake our experiments at large scale: we train models with 5 billion parameters (11 for encoder-decoder) on
168 billion tokens, and perform multitask finetuning on 13 billion tokens. We base our zero-shot evaluation on the set of tasks used by Sanh et al. [2021] (T0-Eval) and the EleutherAI language model evaluation harness (EAI-Eval, Gao et al. [2021]), totalling 30 different datasets with varied prompt styles.

Figure 1 provides an overview of our study, Section 2 introduces background on the LLM architectures, objectives, training strategies, and evaluations considered, and Section 3 details our methods.

Multitask finetuning impacts architecture and objective choice. We find that the popular recipe of a decoder-only model trained with a standard FLM objective performs best when zero-shot capabilities are measured immediately after pretraining, without any finetuning or adaptation. However, after multitask finetuning, the results are the opposite: models pretrained with MLM perform significantly better and decoder-only models perform worse. These experimental results are discussed in Section 4.

Bridging across architectures and objectives with adaptation. This discrepancy motivates us to explore the practice of adaptation (i.e. extending the pretraining of a model with a different architecture/objective) as a way to efficiently obtain both a model suited to generative use cases and to multitask finetuning. We first consider language modeling adaptation: adapting an MLM-trained non-causal decoder model by converting it to a causal decoder and extending its pretraining with an FLM objective.

We find that using a pretrained model in this way speeds up convergence on the language modeling task by a factor 1.6x. We then explore non-causal MLM adaptation, starting from a causal decoder trained with an FLM objective, converting it to a non-causal decoder, and expanding its pretraining with an MLM objective. This form of adaptation produces a new version of the model suited for multitask finetuning, achieving second-best performance across our benchmarks.

Convergence on the MLM task is sped-up by 3.3x, making this the most efficient approach to obtain two distinct models for generative tasks and multitask finetuning. We detail these results in Section 5.

Accordingly, our results both confirm the validity of current standard practices, and help better understand the interplay between architecture, objective, multi-task fine-tuning, and zero-shot generalization. They also identify novel paths forward for efficiently obtaining better LLMs suited to either purely generative prompted usecases, or for multitask finetuning, as discussed in Section 6.

# 2 Background

# 2.1 Architectures

Transformer. Virtually all state-of-the-art LLMs are based on the Transformer architecture [Vaswani et al., 2017]. Due to its ubiquity, we only highlight a few relevant high-level characteristics. The main architectural unit of the Transformer is a Transformer block, which consists of (at minimum) multi-headed self attention [Cheng et al., 2016], layer normalization [Ba et al., 2016], a dense two-layer feedforward network, and residual connections [He et al., 2016]. A Transformer stack is a sequence of such blocks. In NLP applications, the Transformer ingests and outputs tokens.

Since being introduced by Vaswani et al. [2017], various architectural variants of the Transformer have been proposed. A major difference between these architectures is the masking pattern applied to the provided inputs, which act as contextual information for the model to make a prediction. Figure 2 showcases the attention masking patterns in the three architectural variants we consider.

3

Encoder-decoder. As originally proposed, the Transformer consisted of two stacks: an encoder and a decoder. The encoder is fed the sequence of input tokens and outputs a sequence of vectors of the same length as the input. Then, the decoder autoregressively predicts the target sequence, token by token, conditioned on the output of the encoder. To achieve this conditioning, the decoder includes cross-attention layers in each of its blocks, allowing the decoder to also attend to the output of the encoder.

The self-attention layers in the decoder utilize a causal masking pattern that prevents the model from attending to future tokens when predicting the output sequence (see Figure 2, on the right). We hereafter refer to this architecture as the encoder-decoder (ED). Notable pretrained language models using an encoder-decoder architecture include BART [Lewis et al., 2019] and T5 [Raffel et al., 2020]. T5 in particular was recently used as the foundation for the T0 model [Sanh et al.

, 2021], which leveraged large-scale multitask finetuning to achieve strong zero-shot generalization, outperforming decoder-only models an order of magnitude larger.

Causal decoder-only. Although the encoder-decoder is the original Transformer variant, most recent LLMs use a decoder-only architecture. These models can be trained as a traditional language model (i.e. to predict the next token in a sequence). Decoder-only models have no independent means of processing or representing the input sequence and target sequence differently—all tokens are processed in an equivalent fashion, and, because of the causal masking pattern, conditioning is simply based on past tokens (see Figure 2, on the left).

On the one hand, this means that the representation for any conditioning text is inherently weaker; on the other hand, it yields a simpler architecture that is naturally suited to a standard autoregressive next-step-prediction pretraining objective. We refer to this architecture as causal decoder-only (CD). Most notably, the CD architecture makes up the backbone of the GPT series of models [Radford et al., 2018, 2019, Brown et al., 2020] as well as many other recent record-breaking LLMs [Zeng et al., 2021, Kim et al., 2021, Smith et al., 2022, Thoppilan et al., 2022, Rae et al.

, 2021, Hoffmann et al., 2022, Chowdhery et al., 2022].

Non-causal decoder-only. To allow decoder-only models to build richer non-causal representations of the input/conditioning text, it has been proposed to simply modify the attention mask used. Specifically, the self-attention masking pattern can be changed so that the region of the input sequence corresponding to conditioning information has a non-causal mask (i.e. attention in this region is not restricted to past tokens, see middle of Figure 2), as in the encoder of an encoder-decoder architecture. We refer to this architecture as non-causal decoder-only (ND).

Sometimes called a prefix language model, this approach was introduced by [Liu et al., 2018] and was later explored as an architectural variant by [Raffel et al., 2020, Wu et al., 2021]. Despite single-task finetuning performance nearly on par with encoder-decoder models [Raffel et al., 2020], it has seen limited adoption in the literature.

Encoder-only. As an aside, we note that another popular architectural variant is to only use a Transformer encoder layer stack. This model architecture underlies the ubiquitous BERT [Devlin et al., 2018] and its derivatives. However, this architecture is limited to producing the same number of tokens as it was fed as input, considerably limiting its applicability and making it only rarely used in the zero-shot setting [Tamborrino et al., 2020]. We therefore omit it from consideration.

![](dt=2026-05-29/ht=15/d6e896bd4f8e80c763fcdd3d98043227b285377a8835f13e5e1f783c54904d79.jpg)

![](dt=2026-05-29/ht=15/6d6c5f5709a9c61d71e5efe5618c472589ab360b51c138efc690bbb1191cbe07.jpg)

![](dt=2026-05-29/ht=15/e9d4d947142d21921a419a9e7a239bfaf61761280ba5fa7d4c140a64ddfdb9f5.jpg)

4

![](image)
292c8da1618cf6ea5fba22013f93e19ba764a3ef4213e07fc4b22b50d30.jpg)

Comparisons across architectures. Decoder-only models process a single sequence consisting of the concatenation of the input and target text. On the other hand, in an encoder-decoder, the encoder processes only the input and the decoder processes only the target. The total amount of computation performed by an encoder-decoder will therefore be approximately equivalent to a decoder-only model when the encoder and decoder each have as many parameters as the entire decoder-only model (ignoring the cross-attention layers in the encoder-decoder). However, such an encoder-decoder will have twice the parameters of the decoder-only model, and hence twice the memory footprint.

# 2.2 Pretraining objectives

An important step in building LLMs is pretraining, where the model is trained on a large, unlabeled dataset via self-supervision. The choice of pretraining objective can have significant impact on the downstream usability of the LLM, and we therefore include objective choice as a factor in our empirical study. Figure 3 outlines the input and target tokens for the pretraining objectives considered.

Language modeling. Since the advent of GPT-2 [Radford et al., 2019], large decoder-only models have generally been pretrained with an autoregressive language modeling objective [Brown et al., 2020, Wu et al., 2021, Rae et al., 2021]. Given previous tokens, the model is tasked with predicting the following one. We refer to this as full language modeling (FLM). This objective is particularly efficient during pretraining: all tokens in a sequence can generate a loss signal in parallel. At inference time, the model is iteratively asked to predict the next token.

Prefix language modeling. For encoder-decoder and non-causal decoder-only models to perform language modeling, one can define a prefix where the attention mask is allowed to be non-causal. Similar to standard language modeling, the model is tasked to predict each token outside the prefix given all previous tokens. We hereafter refer to this objective as prefix language modeling (PLM). Loss on the prefix is ignored as tokens in the prefix can attend to their targets. For inference, the prefix is naturally the input text; during pretraining, it is usually chosen at random for each sample.

Masked language modeling. Encoder-only models, such as BERT [Devlin et al., 2018], have typically been pretrained with a masked language modeling objective. Tokens or spans of tokens in the input text are replaced with a special mask token and the model is trained to predict the missing tokens. Raffel et al. [2020] introduced a version of this objective adapted to text-to-text models in the form of span corruption: sentinel tokens are used to flag masked spans of short random lengths, and, after processing the masked input, the model outputs the sentinels followed by their respective predicted content. We refer to this approach as masked language modeling (MLM).

# 2.3 Model adaptation

Adaptation extends pretraining with a different objective and/or architecture. In contrast with finetuning, no new downstream data is used, only additional pretraining data. Language modeling adaptation (LM-A) takes a model pretrained with MLM and extend its training with PLM or FLM. It has been used to convert encoder-decoder models pretrained with MLM, such as T5, into better generative models. Notably, it is used as a first step before prompt tuning [Lester et al., 2021] and also to prepare the model before multitask finetuning in T0 [Sanh et al., 2021]. When we perform language

5

modeling adaptation on a non-causal decoder-only model, we convert it into a causal decoder-only by simply switching the attention mask. Furthermore, we propose to study the opposite adaptation: starting from a causal decoder pretrained with FLM, we cast the model into a non-causal decoder (again by switching the attention mask) and we extend pretraining with MLM. We call this approach non-causal MLM adaptation (NC-A); to our knowledge, this is an entirely novel practice.

# 2.4 Multitask finetuning

Modern pretraining corpora are typically massive preprocessed generalist web crawls [Ortiz Suárez et al., 2019, Raffel et al., 2020], collected with no explicit regard for downstream tasks—although adding curated high-quality cross-domain data has been proposed as a path towards better zero-shot generalization [Gao et al., 2020, Scao et al., 2022]. Recently, Sanh et al. [2021] (on an encoder-decoder model trained with MLM) and Wei et al.

[2021] (on a causal decoder-only model trained with FLM) explored the potential of explicitly finetuning the model to solve multiple tasks in order to bolster zero-shot generalization. This is done by finetuning the model on a dataset of prompted tasks (i.e. in a natural language format, leveraging prompt templates applied over many datasets), which ultimately improves zero-shot performance over purely unsupervised pretraining. We refer to this as multitask finetuning (MT-F), and use the openly available datasets and prompts developed for T0.

# 2.5 Zero-shot evaluation

Radford et al. [2019] first demonstrated that LLMs display zero-shot capabilities: given sufficient scale, language models are able to perform many tasks without having explicitly accessed any supervised samples. Zero-shot use of language models relies on a technique called prompting, where tasks are formulated in a natural language format (in accordance with the pretraining objective). The template applied to each example to convert it to this format is called the prompt. Unfortunately, models can exhibit significant sensitivity to the wording of the prompt [Sanh et al., 2021], and it can be difficult to diagnose whether poor performance is a prompt- or model-related problem.

Zero-shot capabilities are of increasing interest in the community, as evidenced by most record-breaking LLMs only reporting zero/few-shot results [Brown et al., 2020, Smith et al., 2022, Rae et al., 2021, Chowdhery et al., 2022]. There are many reasons why zero-shot use is gaining such traction: it does not require any labeled examples, it removes the complexity of model finetuning and deployment, and it also tests generalization to unseen tasks.

We rely on two evaluation benchmarks aggregating prompts across NLP tasks, totalling 30 tasks: the EleutherAI LM Evaluation Harness (EAI-Eval) [Gao et al., 2021], which reimplements the prompts from Brown et al. [2020] and is aimed at evaluation of FLM-trained causal decoder-only models, and the evaluation set from T0 (T0-Eval) [Sanh et al., 2021]. Note that EAI-Eval only includes one prompt per task, whereas performance on T0-Eval is averaged over many prompts. Hence, when reporting performance on T0-Eval, we report a spread across prompts, giving an indication of the impact of the choice of prompt on performance.

# 3 Methods

To better understand how architecture, pretraining objective, multitask finetuning, and possible adaptations influence zero-shot performance, we undertake a systematic large-scale study. We pretrain all possible $\langle$ architecture, objective $\rangle$ pairs on 168 billion tokens from C4, consider intermediate multitask finetuning, and finally evaluate zero-shot performance. We also study the possibility of using adaptation to efficiently transfer the benefits from one architecture/objective to another.

Compute budget guideline. Different architectures and objectives come with different compute trade-offs. We aim to make the training budget similar across all models, using $\sim 15$ petaflops-days for pretraining (for a total of 830,000 TPUv4-hours over the study, see Appendix B.2 for details). We do not take into account memory use: typical use cases are compute-bound by the available GPU/TPU-hours. We note that the encoder-decoder ends up with twice the memory footprint.

Resources and implementation. We run all computation on Google Cloud TPUv4s,
using T5X [Roberts et al., 2022], leveraging JAX [Bradbury et al., 2018] and Flax [Heek et al., 2020].

6

# 3.1 Architecture

We consider causal decoder (CD), encoder-decoder (ED), and non-causal decoder (ND) architectures. All models share the basic configuration outlined in Table 1. For fair comparison across architectures, we aim to approximately match pretraining compute budget; accordingly, our encoder-decoder models have twice as many layers as the decoder-only models. This results in encoder-decoder models with 11B parameters and decoder-only models with 4.8B parameters. We note that due to the cross-attention layers, encoder-decoder models are approximately $\sim 10\%$ more computationally expensive to run than the decoder-only models we consider.

# 3.2 Pretraining

We consider full language modeling (FLM), prefix language modeling (PLM), and masked language modeling (MLM) (specifically, the span corruption objective of Raffel et al. [2020]). The choice of language modeling objective depends on the architecture: the causal decoder uses either FLM or MLM, while the non-causal decoder and the encoder-decoder use either PLM or MLM.

All of our models are pretrained on 168 billion tokens of the C4 dataset from Raffel et al. [2020]. We use the Adafactor [Shazeer and Stern, 2018] optimizer with an inverse square root learning rate schedule, training on batches of 2,048 sequences of length 626 tokens (for a total of 131,072 training steps). Detailed pretraining hyperparameters can be found in Table 2: we based elements of our pretraining setup (such as Adafactor, GEGLU, and the use of an auxiliary $Z$ loss $\mathcal{L}(Z) = 10^{-4} * \log^2(Z)$ to stabilize training [Chowdhery et al., 2022]) on the popular T5.1.1 recipe.

To operate with a fixed compute budget, we match the amount of tokens seen during pretraining (which corresponds to the total computational cost), not the number of tokens trained on (i.e. on which a loss is calculated). Full language modeling computes a loss on all the tokens it sees, whereas prefix language modeling cannot train on the tokens in its prefix: on average, it will train on half as many tokens as full language modeling. We consider these to be inherent trade-offs in efficiency between training objectives. We concatenated and sampled text from documents in such a way that there was virtually no padding during pretraining. More specifically to each objective:

Table 1: Shared architecture for all models trained. Encoder-decoder architectures are doubled in size to obtain a pretraining compute budget similar to the decoder-only architecture.

![](dt=2026-05-29/ht=15/3eb76c999acb24f3adb45d04872e5189b1212732ccb0a1a8b8c40fd8e67afa97.jpg)

<table><tr><td></td><td colspan="2">MODELS ARCHITECTURE</td></tr><tr><td></td><td>Decoder-only</td><td>Encoder-decoder</td></tr><tr><td>Parameters</td><td>4.8B</td><td>11.0B</td></tr><tr><td>Vocabulary</td><td></td><td>32,128</td></tr><tr><td>Positional embed.</td><td></td><td>T5 relative</td></tr><tr><td>Embedding dim.</td><td></td><td>4,096</td></tr><tr><td>Attention heads</td><td></td><td>64</td></tr><tr><td>Feedforward dim.</td><td></td><td>10,240</td></tr><tr><td>Activation</td><td colspan="2">GEGLU [Shazeer, 2020]</td></tr><tr><td>Layers</td><td>24</td><td>48</td></tr><tr><td>Tied embeddings</td><td></td><td>True</td></tr><tr><td>Precision</td><td></td><td>bfloat16</td></tr></table>

7

# 3.3 Multitask finetuning

Drawing from recent work demonstrating that multitask finetuning improves zero-shot performance, we also evaluate our models after multitask finetuning (MT-F), following the procedure used for the T0 model of Sanh et al. [2021]. Our goal is to better disambiguate the influence of architecture and objective in this relatively nascent practice. For example, we note that T0 and FLAN are significantly different in the architecture and objective used (encoder-decoder with MLM and causal decoder with FLM, respectively). We hope our experiments can help lend insight into which of these design choices is more effective for enabling zero-shot generalization after multitask finetuning.

After pretraining, we create multitask versions of our models by finetuning on the T0 training dataset mixture from Sanh et al. [2021] (not $\mathrm{T0 + }$ or $\mathrm{T0 + + }$ ) for 13 billion tokens. Our finetuning configurations follow those used for T0 (see Table 2 for details), and note that we found dropout to significantly impact zero-shot generalization (see Appendix E.3 for a comparison with and without dropout). We refer readers to Sanh et al. [2021] for further information about this multitask finetuning procedure.

One significant departure to note from the approach of Sanh et al. [2021] is that we do not perform language modeling adaptation first before multitask finetuning. Preliminary results (see Appendix E.1 in the Appendix) did not show any systematic improvement from performing language modeling adaptation, so we omitted this step. This is consistent with the finding from Lester et al. [2021] that language modeling adaptation is not necessary before prompt tuning for large models.

# 3.4 Evaluation

We use two zero-shot evaluation benchmarks to assess our models. First, we use the same set of tasks, datasets, and prompts as was used to evaluate T0 (T0-Eval) [Sanh et al., 2021], and second, the EleutherAI LM evaluation harness (EAI-Eval) [Gao et al., 2021]. The EAI prompts attempt to replicate the evaluation set of Brown et al. [2020]. The prompts of T0 were built to be "human understandable" and were originally used in conjunction with an encoder-decoder model. See Appendix D for a detailed list of tasks, and the overlap between T0-Eval, EAI-Eval, and T0-Train.

T0-Eval provides multiple prompts per task, whereas EAI-Eval provides only one prompt per task. Accordingly, for T0-Eval, we take the median accuracy over all prompts for each task and then average across all 11 datasets. For EAI-Eval we simply average the accuracy obtained on each of the 31 datasets. Note that because these are aggregated zero-shot benchmarks, variations of even a percent can hide significant differences on a single task.

All but one task in T0-Eval (StoryCloze) are also in EAI-Eval. Some of the datasets in EAI-Eval are also in the T0 training datasets: GLUE-MRPC, GLUE-QQP, and SciQ. We also did not check for contamination from C4, but given the fact that all models would have the opportunity to memorize the tasks leaked in C4, we believe it does not impact our evaluation. For additional discussion of the overlap between T0 and C4, we refer readers to the original T0 paper Sanh et al. [2021].

We perform evaluation of model checkpoints at 42B, 84B, and 168B tokens. We note that the random baselines are $\sim 33\%$ for EAI-Eval and $\sim 42\%$ for T0-Eval. The complete set of results across all checkpoints obtained through this study is made available in Appendix E.2.

Table 2: Pretraining and multitask finetuning configurations for all models trained. Pretraining lasts for 168 billion tokens, while multitask finetuning is done for 13 billion tokens.

![](dt=2026-05-29/ht=15/120617924c3a54cd36d851ac84f082d077c060b43cb4cced8ffdd3f8fa2e73c7.jpg)

<table><tr><td></td><td>PRETRAINING</td><td>MULTITASK FINETUNING</td></tr><tr><td>Dataset</td><td>C4</td><td>T0-Train</td></tr><tr><td>Steps</td><td>131,072</td><td>10,000</td></tr><tr><td>Batch size in tokens</td><td>1,282,048</td><td>1,310,720</td></tr><tr><td>Optimizer</td><td colspan="2">Adafactor(decay_rate=0.8)</td></tr><tr><td>LR schedule</td><td>1/√max(n,10^4)</td><td>fixed, 0.001</td></tr><tr><td>Dropout</td><td>0.0</td><td>0.1</td></tr><tr><td>z loss</td><td></td><td>0.0001</td></tr><tr><td>Precision</td><td></td><td>bfloat16</td></tr></table>

8

# 4 Experiments

# 4.1 After unsupervised p
retraining only

We are first interested in the architecture and objective achieving the best zero-shot performance after unsupervised pretraining only. For this, we only consider the full-prefix language modeling objectives since masked language modeling does not yield a model appropriate for zero-shot prompted evaluation on its own. This is validated with early checkpoints in Appendix E.1.

We present our main full-prefix language modeling pretraining results in Table 3. On both our evaluation benchmarks, the causal decoder architecture systematically outperforms the other architectures when using language modeling pretraining alone. The non-causal decoder remains within a percent of the causal decoder performance, but the encoder-decoder performance lags far behind. Finally, we note that the performances on T0-Eval are close to the random baseline, while performance differences on EAI-Eval are significant enough to make comparison across experiments.

Finding 1. Causal decoder-only models pretrained with a full language modeling objective achieve best zero-shot generalization when evaluated immediately after unsupervised pretraining, in line with current common practices for large language models.

# 4.2 After multitask finetuning

We now focus on the relatively new practice of multitask finetuning, where there has not yet been any systematic study of the influence of the architecture and training objective. Notably, the two main papers advocating this practice use completely different approaches: Sanh et al. [2021] finetunes an encoder-decoder model pretrained with span corruption, whereas Wei et al. [2021] finetunes a decoder-only pretrained with full language modeling.

It is not immediately clear which approach is more natural: while decoder-only models trained with full language modeling are better at zero-shot generalization (as evidenced in Section 4.1), encoder-decoder models and masked language modeling pretraining have been shown to perform significantly better after finetuning [Raffel et al., 2020]. We therefore evaluate every architecture and objective combination after multitask finetuning.

Our results are outlined in Figure 4. The encoder-decoder pretrained with span corruption offers the best performance after multitask finetuning. Specifically, on EAI-Eval, the best performance is achieved by the encoder-decoder with MLM, and the non-causal decoder with MLM comes in a close second. However, the difference is more significant on T0-Eval, where the encoder-decoder with MLM pretraining outperforms other models by a large margin. Finally, encoder-decoder pretrained with PLM and causal decoder with MLM achieve significantly worse performance than other models. These results are consistent across all levels of pretraining (see early checkpoints in Appendix D).

Finding 2. Encoder-decoder models trained with masked language modeling achieve the best zero-shot performance after multitask finetuning. More broadly, approaches that perform well in the single-task finetuning setting perform well on multitask finetuning.

Table 3: After full or prefix language modeling pretraining, the causal decoder (FLM) exhibits the best zero-shot generalization abilities, followed closely by the non-causal decoder (PLM). Average accuracy on EAI-Eval and T0-Eval after pretraining for 168B tokens. MLM pretraining is not considered here, as the models produced are not suitable for direct use in a zero-shot setting. Note that performance on T0-Eval remains close to random baseline. Best for each benchmark.

![](dt=2026-05-29/ht=15/4ea95eaaad3918c39ac32ea5c509a5915b49510f94a6a806c77467b2b10f4af2.jpg)

<table><tr><td></td><td>EAI-EVAL</td><td>T0-EVAL</td></tr><tr><td>Causal decoder</td><td>44.2</td><td>42.4</td></tr><tr><td>Non-causal decoder</td><td>43.5</td><td>41.8</td></tr><tr><td>Encoder-decoder</td><td>39.9</td><td>41.7</td></tr><tr><td>Random baseline</td><td>32.9</td><td>41.7</td></tr></table>

9

![](dt=2026-05-29/ht=15/ecd2330703445108fdc82434b4efe0e1c1c655e1b2589d5904229861c4b80477.jpg)

![](dt=2026-05-29/ht=15/4ca9b589867afdd60a10ebfdf58c736782d933c3c70a45ab6b09be2c11797613.jpg)

![](dt=2026-05-29/ht=15/fa88cbbd3ec50c73eee0ef4e5da75154a7226a0b977199d00e38558bfd50dec6.jpg)

# 4.3 Influence of the tasks and prompts used for zero-shot evaluation

Although the datasets considered in EAI-Eval and T0-Eval have significant overlap (10 out of 11 T0 tasks are in EAI-Eval), the prompts are always different between the two benchmarks. The EAI prompts for these datasets were taken from Brown et al. [2020], who hand-tuned them to maximize performance of the GPT-3 models. In contrast, the T0-Eval prompts were sourced through a community effort with prompt diversity and naturalness as primary goals without any regard for model performance [Sanh et al., 2021].

Consequently, on each task, the EAI prompt has higher performance than the average T0 prompt for all models and tends to be on par with the best T0 prompt. The difference is most pronounced for causal decoder language models without multitask finetuning, likely because this is the setting GPT-3 prompts were optimized for. This is reflected in the structure of the prompts, which tend not to explain the task to the reader like T0 evaluation prompts do. Instead, they attempt to reformulate it as close to language modeling as possible.

In addition to this base performance discrepancy, EAI-Eval has less discrepancy between encoder-decoder models and the rest, and better performance for autoregressive decoder models. We untangle the effect of the difference in prompts and the different task sets by separately comparing performance on tasks that share in T0-Eval and those that are not, while always using EAI-Eval prompts, as shown in Figure 5. The set of EAI-Eval tasks considered in T0-Eval seems to lend itself better to encoder-decoder models than the rest.

On non-T0-Eval tasks, in contrast, causal decoder performance shoots up dramatically, although a lot of the difference is driven by LAMBADA [Paperno et al., 2016a], a language modeling task. Nevertheless, we note that when considering wide and varied task aggregates, our high-level findings are mostly consistent across evaluation settings—although specific tasks, such as LAMBADA, may indeed favor a specific architecture and objective combination.

10

![](dt=2026-05-29/ht=15/e646b596f1854c7575556fc8912eb83fb42d8b723b734d0cca04f573bede5755.jpg)

# 5 Can models be adapted from one architecture/objective to another?

Our experimental study has led us to conclude the optimal architecture and objective choice for zero-shot performance depends on whether or not the model will ultimately undergo multitask finetuning: while a decoder-only model trained with full language modeling achieves the best zero-shot performance after unsupervised pretraining only, an encoder-decoder with masked language modeling is best once multitask finetuning is applied. This is inconvenient, as the multitask finetuned encoder-decoder model may not be suitable for many open-ended generative tasks that the decoder-only model excels at, while the decoder-only model will not be the best at many zero-shot tasks.

In this section, we attempt a compromise between the two options above. We study the practice of adaptation: extending pretraining with a different architecture and/or objective. Our end-goal is to efficiently obtain two distinct models: one that leverages multitask finetuning to maximize zero-shot performance, and another that can be used as a high-quality language model.

Languag
e modeling adaptation (LM-A). First, we propose to pretrain a non-causal decoder model with an MLM objective and then further train the model as a causal decoder with a FLM objective (language modeling adaptation). This conversion is simple, as the parameters and overall architecture can be kept the same, and only the attention mask needs to be switched. We note that we also attempted this adaptation from the decoder portion of an encoder-decoder model, but it performed significantly worse than training from scratch, as discussed in Appendix E.4.

Validations losses are plotted in Figure 6, on the left. Starting from an MLM-pretrained non-causal decoder model speeds up convergence significantly compared to training a causal-decoder model with an FLM objective from scratch. To achieve a loss comparable to the one achieved after 168B tokens of FLM pretraining, language modeling adaptation requires only 105B additional tokens (a $1.6 \times$ speed-up). This makes it possible to obtain both a high-quality zero-shot model and a good generative model, for only $1.6 \times$ the cost of training a single model.

11

![](dt=2026-05-29/ht=15/0b3b5808d9dabc6f21d70125b6fa8d0d16988358a42710324fc4b0e67fd1fbe3.jpg)

![](dt=2026-05-29/ht=15/9f2a50c48357f4ffbaeae90769665124989947f75fb6ecca691ad5b5c1366b63.jpg)

Non-causal masked language modeling adaptation (NC-A). To investigate alternative avenues for adaptation, we now introduce non-causal masked language modeling adaptation: starting from a causal decoder model pretrained with FLM as the objective, we then continue training the model as a non-causal decoder using an MLM objective. This is essentially the reverse of the language modeling adaptation setup, and the conversion is as easily undertaken by switching the attention mask.

Validation losses are plotted in Figure 6, on the right. Convergence on the MLM pretraining objective is significantly accelerated: by a factor of $3.3 \times$ compared to training a non-causal decoder from scratch, and up to a factor $9.1 \times$ compared to training a causal decoder from scratch (both with a masked language modeling objective). This is a significant improvement over even the previously considered language modeling adaptation, enabling one to obtain both a zero-shot model and an excellent generative model for only $1.3 \times$ the cost of training a single model.

Finally, we confirm that the improvement in validation loss also transfer to an improvement in zero-shot generalization. We evaluate the non-causal MLM adapted model, and check that it is better than the original causal decoder model pretrained with full language modeling, and control for the total number of training tokens.

Specifically, we evaluate zero-shot performance after multitask finetuning in three settings: first, a causal decoder model pretrained with FLM for 219 billion tokens before being multitask finetuned; second, a causal decoder model pretrained with FLM for 219 billion tokens and then multitask finetuned as a non-causal decoder model; and, third, a causal decoder model first trained with FLM for 168 billion tokens, then MLM-adapted as an non-causal model for 51 billion tokens, and finally multitask finetuned. All three variants are multitask finetuned for 13 billion tokens.

Results are presented in Figure 7. We find that the MLM-adapted model performs best by a significant margin and outperforms every other model we considered on EAI-Eval. Furthermore, the measured zero-shot generalization is in line with the MLM-pretrained non-causal decoder reported in Figure 4, though it still lags behind the MLM-pretrained encoder-decoder, despite the adapted models having seen 51 billion additional tokens. Finally, we note that performing non-causal multitask finetuning of the causal model produces no meaningful change in performance.

Finding 3. Decoder-only models can be efficiently adapted from one architecture/objective prior to the other. Specifically, to obtain both a generative and a multitask model with the smallest total compute budget possible, we recommend starting with a causal decoder-only model, pretraining it with a full language modeling objective, and then using non-causal masked language modeling adaptation before taking it through multitask finetuning.

12

![](dt=2026-05-29/ht=15/1ad46730363cd8c273a3163a510f5b0eb0613562008a2d269d08ba9f61d0c3c0.jpg)

![](dt=2026-05-29/ht=15/829de125871df53141519b3cc7243059cbba8225177399a2c71789a7f67f46fb.jpg)

# 6 Conclusion

In this paper, we systematically studied the effects of pretraining objective and architecture choices on the zero-shot generalization abilities of large language models. Specifically, we compared language modeling and masked language modeling objectives applied to causal/non-causal decoder-only and encoder-decoder architectures. We also evaluated zero-shot performance with and without multitask finetuning.

Notably, we found that the best objective and architecture is the opposite in these two settings: a causal decoder-only pretrained with full language modeling performs best if evaluated immediately after pretraining, whereas when adding a multitask finetuning step, an encoder-decoder pretrained with masked language modeling performs best. We therefore evaluate the practice of adaptation, to convert models across architectures and objectives.

We found a simple efficient compromise, where a causal decoder-only model pretrained with full language modeling underwent additional masked language model training as a non-causal decoder-only model, yielding significant speedup in convergence over starting from scratch. This enables practitioners to get both an excellent generative model and a model that delivers good performance after multitask finetuning. Our results provide significant new insights into the design of LLMs.

In the future, we are interested in work investigating architectures and objectives that perform well regardless of whether multitask finetuning is performed. To facilitate future work, we release all models, code, and data used in our study.

13

# Acknowledgements

BigScience. This work was pursued as part of the Big Science research workshop, a one-year long initiative on large multilingual models and datasets. Specifically, this work was conducted by a task force within the architecture & scaling group, seeking to establish the optimal architecture and pretraining objective for the final 176B parameter model produced by BigScience. We would also like to thank Stella Biderman for valuable comments.

Compute. We thank the TPU Research Cloud team for providing us with generous access to TPUv4. We thank the TPUv4 Alpha team for providing technical support for this work, and notably James Bradbury. All pretraining, adaptations, finetuning, and evaluations featured in this paper used TPUv4.

This work was granted access to the HPC resources of Institut du développement et des ressources en informatique scientifique (IDRIS) du Centre national de la recherche scientifique (CNRS) under the allocation 2021-A0101012475 made by Grand équipement national de calcul intensif (GENCI). Specifically, early experiments on non-causal decoder models, which sparked this exhaustive study, were performed on the Jean Zay supercomputer of IDRIS.

# Author-specific funding.

![](dt=2026-05-29/ht=15/28ab8b17868e227c07b6e92774570ce4bc6d0779f1a5a6dff570eb89e22cb593.jpg)

Daniel Hesslow has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 860
360.

# References

14

15

16

17

18

19

# A Contributions

Thomas Wang wrote code, ran experiments, performed evaluation, generated plots, and helped with paper writing. Adam Roberts led the creation of the codebase used in this project, proposed some experiments, ran all of the final experiments, generated plots, and helped with paper writing. Daniel Hesslow made it possible to evaluate models with the EleutherAI harness, created diagrams, and helped with paper writing. Teven Le Scao ran evaluations and plotted results. Hyung Won Chung implemented infrastructure code for different architectural and objective variants.

Iz Beltagy co-chaired the BigScience Architecture & Scaling Group and helped with paper editing. Julien Launay co-chaired the BigScience Architecture & Scaling Group and had the largest role in paper writing. Colin Raffel proposed the project, the experiments, and the adaptation methods and wrote portions of the paper.

# B Broader impacts

# B.1 Societal impact

The risks and societal challenges raised by large language models have been discussed extensively in the literature [Solaiman et al., 2019, Bommasani et al., 2021]. Our research is strictly oriented on benchmarking modeling aspects, and thus does not introduce any novel challenge beside those already identified. Notably, many similarly capable models have already been released publicly in the past [Raffel et al., 2020, Wang and Komatsuzaki, 2021, Sanh et al., 2021, Black et al., 2022].

In the spirit of reproducibility and openness, we release all artefacts produced during this study: the configs necessary to reproduce our results from scratch, checkpoints of all of the models trained, and detailed evaluation results. These artefacts are intended for research only: we did not evaluate the potential biases of the models trained, and cannot guarantee they won't produce harmful content. Accordingly, these models should not be used in production or exposed to the public.

We also highlight that algorithmic choices can introduce biases on their own [Hooker, 2021]: one limitation of our study is that we did not explore whether specific architectures and objectives had an impact on the toxicity and biases of a given model. However, the public availability of all the models trained for this study enables researchers to conduct such a follow-up study at minimal compute cost.

# B.2 Environmental impact

Across all experiments undertaken over the course of this study (including unreported preliminary and failed experiments), we performed training for 1,854 hours on 64 chips (TPUv4-128) and 1,395 hours on 512 TPUv4 chips (TPUv4-1024) for a total of 832,896 chip-hours. Recently, Chowdhery et al. [2022] presented the results of training a 540 billion parameter language model on TPUv4 chips in the same datacenter where we ran our experiments. Their model was trained for 1,200 hours on 6,144 TPUv4 chips and 336 hours on 3,072 TPUv4 chips, for a total of 8,404,992 chip-hours. Chowdhery et al.

[2022] estimates the carbon emissions of their model training to be $240.5\mathrm{tCO2e}$ based on the net tCO2e per MWh of the datacenter during training and the energy usage of TPUv4 chips. We therefore estimate our carbon emissions to be approximately $23.8\mathrm{tCO2e}$ , which is approximately half of what Patterson et al. [2021] report for the original T5 model training (46.7 tCO2).

# C Implementation: prefix language modeling for encoder-decoder on TPU

Due to constant size constraints, for encoder-decoder with prefix language modeling, we have to concatenate two examples of 626 tokens into one. We randomly sample an index $i$ between 0 and 626, and use $i$ and $626 - i$ as prefix indices in the two examples. We use masking to keep them independent throughout training. The encoder-decoder thus has a 1,252 sequence length, and we train it with a batch size of 1,024 sequences instead of 2,048 to keep the number of tokens constant.

20

# D Evaluation: benchmarks composition and baselines

We detail the split across EAI-Eval and T0-Eval in Table 4, and provide random baselines in Table 5.

Table 4: Tasks used for zero-shot evaluation within EAI-Eval and T0-Eval, as well as tasks included in the T0 training set for multitask finetuning. Note that T0-Eval/Training tasks include multiple prompts, while EAI-Eval tasks have a single prompt. Although tasks are shared between the two, there are no shared prompts between EAI-Eval and T0-Eval.

![](dt=2026-05-29/ht=15/813be0f580c5669fa91f7a93e2836021b65b2ca531a790b331f89d9cc56f5a43.jpg)

<table><tr><td></td><td>TASK</td><td>TYPE</td><td>T0 Training</td><td>DATASET EAI-Eval</td><td>T0-Eval</td></tr><tr><td>ANLI</td><td></td><td>Natural Language Inference</td><td></td><td>✓</td><td>✓</td></tr><tr><td rowspan="2">ARC [Clark et al., 2018]</td><td>Challenge</td><td>Closed-Book Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>Easy</td><td>Closed-Book Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td rowspan="2">GLUE</td><td>MRPC [Dolan and Brockett, 2005]</td><td>Paraphrase Identification</td><td>✓</td><td>✓</td><td></td></tr><tr><td>QQP [Iyer et al., 2017]</td><td>Paraphrase Identification</td><td>✓</td><td>✓</td><td></td></tr><tr><td>HEAD-QA</td><td></td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>HellaSwag [Zellers et al., 2019]</td><td></td><td>Sentence Completion</td><td></td><td>✓</td><td>✓</td></tr><tr><td>LAMBADA [Paperno et al., 2016b]</td><td></td><td>Sentence Completion</td><td></td><td>✓</td><td></td></tr><tr><td>LogiQA [Liu et al., 2020]</td><td></td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>MathQA [Amini et al., 2019]</td><td></td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>OpenBookQA [Mihaylov et al., 2018]</td><td></td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>PIQA [Bisk et al., 2020]</td><td></td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>PROST [Aroca-Ouellette et al., 2021]</td><td></td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>PudMedQA [Jin et al., 2019]</td><td></td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>QNLI [Rajpurkar et al., 2016, Wang et al., 2019]</td><td></td><td>Sentence Completion</td><td></td><td>✓</td><td></td></tr><tr><td>Race Lai et al. [2017]</td><td></td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>SciQ [Welbl et al., 2017]</td><td></td><td>Multiple-Choice Question Answering</td><td>✓</td><td>✓</td><td></td></tr><tr><td>SST [Socher et al., 2013] [Socher et al., 2013]</td><td></td><td>Sentiment</td><td></td><td>✓</td><td></td></tr><tr><td>StoryCloze</td><td></td><td>Sentence Completion</td><td></td><td></td><td>✓</td></tr><tr><td rowspan="7">SuperGLUE</td><td>Boolq [Clark et al., 2019]</td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>CB</td><td>Natural Language Inference</td><td></td><td>✓</td><td>✓</td></tr><tr><td>COPA [Gordon et al., 2012]</td><td>Sentence Completion</td><td></td><td>✓</td><td>✓</td></tr><tr><td>MultiRC [Khashabi et al., 2018]</td><td>Multiple-Choice Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>RTE [Dagan et al., 2005]</td><td>Natural Language Inference</td><td></td><td>✓</td><td>✓</td></tr><tr><td>WIC [Pilehvar and os&#x27;e Camacho-Collados, 2018]</td><td>Word Sense Disambiguation</td><td></td><td>✓</td><td>✓</td></tr><tr><td>WSC [Levesque et al., 2012]</td><td>Word Sense Disambiguation</td><td></td><td>✓</td><td>✓</td></tr><tr><td>TriviaQA [Joshi et al., 2017]</td><td></td><td>Closed-Book Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>WebQuestions [Berant et al., 2013]</td><td></td><td>Closed-
Book Question Answering</td><td></td><td>✓</td><td></td></tr><tr><td>Winogrande [Sakaguchi et al., 2019]</td><td></td><td>Coreference resolution</td><td></td><td>✓</td><td>✓</td></tr><tr><td>WNLI [Sakaguchi et al., 2019]</td><td></td><td>Natural language inference</td><td></td><td>✓</td><td></td></tr></table>

21

Table 5: Random baselines for all tasks considered across EAI harness and T0 Eval. These baselines were obtained from the papers introducing these tasks.

![](dt=2026-05-29/ht=15/5f69006610c67cf090ebb44c51573df5f0f5f66ece8fc021f63ea9010a8411e3.jpg)

<table><tr><td colspan="2">TASK</td><td>RANDOM BASELINE</td></tr><tr><td>ANLI</td><td></td><td>33.3</td></tr><tr><td rowspan="2">ARC</td><td>Challenge</td><td>25.0</td></tr><tr><td>Easy</td><td>25.0</td></tr><tr><td rowspan="2">GLUE</td><td>MRPC</td><td>50.0</td></tr><tr><td>QQP</td><td>50.0</td></tr><tr><td>HEAD-QA</td><td></td><td>25.0</td></tr><tr><td>HellaSwag</td><td></td><td>25.0</td></tr><tr><td>LAMBADA</td><td></td><td>0.0</td></tr><tr><td>LogiQA</td><td></td><td>25.0</td></tr><tr><td>MathQA</td><td></td><td>20.1</td></tr><tr><td>OpenBookQA</td><td></td><td>25.0</td></tr><tr><td>PIQA</td><td></td><td>50.0</td></tr><tr><td>PROST</td><td></td><td>25.0</td></tr><tr><td>PudMedQA</td><td></td><td>33.3</td></tr><tr><td>QNLI</td><td></td><td>50.0</td></tr><tr><td>Race</td><td></td><td>25.0</td></tr><tr><td>SciQ</td><td></td><td>25.0</td></tr><tr><td>SST</td><td></td><td>50.0</td></tr><tr><td>StoryCloze</td><td></td><td>50.0</td></tr><tr><td rowspan="7">SuperGLUE</td><td>Boolq</td><td>50.0</td></tr><tr><td>CB</td><td>50.0</td></tr><tr><td>COPA</td><td>50.0</td></tr><tr><td>MultiRC</td><td>5.8</td></tr><tr><td>RTE</td><td>50.0</td></tr><tr><td>WIC</td><td>50.0</td></tr><tr><td>WSC</td><td>50.0</td></tr><tr><td>TriviaQA</td><td></td><td>0.0</td></tr><tr><td>WebQuestions</td><td></td><td>0.0</td></tr><tr><td>Winogrande</td><td></td><td>50.0</td></tr><tr><td>WNLI</td><td></td><td>50.0</td></tr><tr><td>EAI-EVAL</td><td></td><td>33.3</td></tr><tr><td>T0-EVAL</td><td></td><td>41.7</td></tr></table>

22

# E Additional results

# E.1 Preliminary results and evolution throughout pre-training

Leveraging early pretraining results at 42B and 84B tokens, we motivate in this section two special design decisions in our study:

Table 6: Models pretrained with masked language modeling achieve performance close to the random $33.3\%$ baseline on EAI-Eval, significantly underperforming full and prefix language modeling. Average accuracy on EAI-Eval after pretraining for 84B tokens. This observations leads us to not consider masked language modeling for evaluations after pretraining only.

Table 7: Performing LM adaptation before multitask finetuning does not improve results, and in fact hinders performance in most cases. Average accuracy on EAI-Eval and T0-Eval for different adaptation strategies after 42B tokens of masked language modeling pretraining. LM adaptation alone is insufficient, and most performance gains come from MT finetuning. Accordingly, we diverge from the setup of Sanh et al. [2021], and forego systematic LM adaptation before multitask finetuning.

![](dt=2026-05-29/ht=15/7c5ac3bcf315527248cbe7ba7927182fb24530065e4b6b1c12b8a9514cbfba6a.jpg)

<table><tr><td></td><td>EAI-EVAL</td></tr><tr><td colspan="2">FLM/PLM</td></tr><tr><td>Causal decoder</td><td>42.4</td></tr><tr><td>Non-causal decoder</td><td>42.2</td></tr><tr><td>Encoder-decoder</td><td>39.6</td></tr><tr><td colspan="2">MLM</td></tr><tr><td>Causal decoder</td><td>37.8</td></tr><tr><td>Non-causal decoder</td><td>37.7</td></tr><tr><td>Encoder-decoder</td><td>34.6</td></tr></table>

![](dt=2026-05-29/ht=15/49db848629e9323f72a4ac8eacd6325a9a145602fddb773f71148568c054e782.jpg)

<table><tr><td></td><td>EAI-EVAL</td><td>T0-EVAL</td></tr><tr><td colspan="3">LM ADAPTATION</td></tr><tr><td>Causal decoder</td><td>38.6</td><td>43.9</td></tr><tr><td>Non-causal decoder</td><td>39.5</td><td>40.8</td></tr><tr><td>Encoder-decoder</td><td>38.6</td><td>39.1</td></tr><tr><td colspan="3">MT FINETUNING</td></tr><tr><td>Causal decoder</td><td>43.3</td><td>45.8</td></tr><tr><td>Non-causal decoder</td><td>45.9</td><td>48.9</td></tr><tr><td>Encoder-decoder</td><td>45.4</td><td>53.7</td></tr><tr><td colspan="3">LM-A+MT FINETUNING</td></tr><tr><td>Causal decoder</td><td>43.9</td><td>46.7</td></tr><tr><td>Non-causal decoder</td><td>45.0</td><td>48.0</td></tr><tr><td>Encoder-decoder</td><td>45.7</td><td>52.6</td></tr></table>

23

# E.2 Complete Results

We report results for all intermediary checkpoints produced in Table 8, and specifically for all multitask finetuned checkpoints on T0-Eval in Figure 8.

Table 8: Average accuracy on EAI-EVal and T0 Eval for all experiments. Experiments are represented as a combination of architecture:objective (tokens) training stages, where architecture is one of causal decoder-only (CD), non-causal decoder-only (ND), or encoder-decoder (ED), and objective is one of full language modeling (FLM), prefix language modeling (PLM), masked language modeling (MLM), or multitask finetuning (MTF).

![](dt=2026-05-29/ht=15/7375698d865c704b2affd3973fc38028ed29628f93ccc0ac556fa7920f899e06.jpg)

<table><tr><td>PRETRAINING</td><td>TRAINING STAGE
ADAPTATION</td><td>FINETUNING</td><td>TOTAL TAXENS</td><td>EAI-EVAL</td><td>T0-EVAL</td></tr><tr><td>CD:MLM (38B)</td><td>CD:FLM (4B)</td><td></td><td>42B</td><td>38.6</td><td>43.9</td></tr><tr><td>ND:MLM (38B)</td><td>ND:PLM (4B)</td><td></td><td>42B</td><td>39.5</td><td>40.8</td></tr><tr><td>ED:MLM (38B)</td><td>ED:MLM (4B)</td><td></td><td>42B</td><td>38.6</td><td>39.1</td></tr><tr><td>CD:MLM (42B)</td><td></td><td>CD:MTF (13B)</td><td>55B</td><td>43.3</td><td>45.8</td></tr><tr><td>ND:MLM (42B)</td><td></td><td>ND:MTF (13B)</td><td>55B</td><td>45.9</td><td>48.9</td></tr><tr><td>ED:MLM (42B)</td><td></td><td>ED:MTF (13B)</td><td>55B</td><td>45.4</td><td>53.7</td></tr><tr><td>CD:MLM (38B)</td><td>CD:FLM (4B)</td><td>CD:MTF (13B)</td><td>55B</td><td>43.9</td><td>46.7</td></tr><tr><td>ND:MLM (38B)</td><td>ND:PLM (4B)</td><td>ND:MTF (13B)</td><td>55B</td><td>45.0</td><td>48.0</td></tr><tr><td>ED:MLM (38B)</td><td>ED:PLM (4B)</td><td>ED:MTF (13B)</td><td>55B</td><td>45.7</td><td>52.6</td></tr><tr><td>CD:FLM (84B)</td><td></td><td></td><td>84B</td><td>42.4</td><td>-</td></tr><tr><td>ND:PLM (84B)</td><td></td><td></td><td>84B</td><td>42.2</td><td>-</td></tr><tr><td>ED:PLM (84B)</td><td></td><td></td><td>84B</td><td>39.6</td><td>-</td></tr><tr><td>CD:MLM (84B)</td><td></td><td></td><td>84B</td><td>37.8</td><td>-</td></tr><tr><td>ND:MLM (84B)</td><td></td><td></td><td>84B</td><td>37.7</td><td>-</td></tr><tr><td>ED:MLM (84B)</td><td></td><td></td><td>84B</td><td>34.6</td><td>-</td></tr><tr><td>CD:FLM (84B)</td><td></td><td>CD:MTF (13B)</td><td>97B</td><td>49.0</td><td>49.9</td></tr><tr><td>ND:PLM (84B)</td><td></td><td>ND:MTF (13B)</td><td>97B</td><td>46.3</td><td>50.0</td></tr><tr><td>ED:PLM (84B)</td><td></td><td>ED:MTF (13B)</td><td>97B</td><td>43.2</td><td>46.5</td></tr><tr><td>CD:MLM (84B)</td><td></td><td>CD:MTF (13B)</td><td>97B</td><td>45.8</td><td>48.2</td></tr><tr><td>ND:MLM (84B)</td><td></td><td>ND:MTF (13B)</td><td>97B</td><td>49.0</td><td>52.6</td></tr><tr><td>ED:MLM (84B)</td><td></td><td>ED:MTF (13B)</td><td>97B</td><td>49.0</td><td>56.5</td></tr><tr><td>CD:FLM (168B)</td><td></td><td></td><td>168B</td><td>44.2</td><td>42.4</td></tr><tr><td>ND:PLM (168B)</td><td></td><td></td><td>168B</td><td>43.5</td><td>41.8</td></tr><tr><td>ED:PLM (168B)</td><td></td><td></td><td>168B</td><td>39.9</td><td>41.7</td></tr><tr><td>CD:FLM (168B)</td><td></td><td>CD:MTF (13B)</td><td>181B</td><
td>50.4</td><td>51.4</td></tr><tr><td>ND:PLM (168B)</td><td></td><td>ND:MTF (13B)</td><td>181B</td><td>48.9</td><td>54.0</td></tr><tr><td>ED:PLM (168B)</td><td></td><td>ED:MTF (13B)</td><td>181B</td><td>44.2</td><td>45.8</td></tr><tr><td>CD:MLM (168B)</td><td></td><td>CD:MTF (13B)</td><td>181B</td><td>47.1</td><td>50.3</td></tr><tr><td>ND:MLM (168B)</td><td></td><td>ND:MTF (13B)</td><td>181B</td><td>51.0</td><td>55.2</td></tr><tr><td>ED:MLM (168B)</td><td></td><td>ED:MTF (13B)</td><td>181B</td><td>51.3</td><td>60.6</td></tr><tr><td>CD:FLM (168B)</td><td>CD:FLM (51B)</td><td>CD:MTF (13B)</td><td>232B</td><td>51.3</td><td>52.1</td></tr><tr><td>CD:FLM (168B)</td><td>CD:FLM (51B)</td><td>ND:MTF (13B)</td><td>232B</td><td>52.3</td><td>52.0</td></tr><tr><td>CD:FLM (168B)</td><td>ND:MLM (51B)</td><td>ND:MTF (13B)</td><td>232B</td><td>52.3</td><td>54.9</td></tr><tr><td colspan="6">T5-LM BASELINE [LESTER ET AL., 2021]</td></tr><tr><td>ED:MLM (1.28T)</td><td>ED:PLM (131B)</td><td></td><td>1.41T</td><td>39.0</td><td>43.2</td></tr><tr><td colspan="6">TO BASELINE [SANH ET AL., 2021]</td></tr><tr><td>ED:MLM (1.28T)</td><td>ED:PLM (131B)</td><td>ED:MTF (13B)</td><td>1.43T</td><td>52.2</td><td>62.5</td></tr><tr><td>RANDOM BASELINE</td><td></td><td></td><td>-</td><td>32.9</td><td>41.7</td></tr></table>

24

![](dt=2026-05-29/ht=15/bcd9fad4023f49c20362522566b29bc8d3907e177bba6b9ae407991159ab1551.jpg)

# E.3 Impact of dropout on multitask finetuning

We also performed multitask finetuning without using dropout, with results in Figure 9. We find that using dropout as originally suggested by Sanh et al. [2021] significantly boosts zero-shot generalization. Results are consistent across architectures and pretraining objectives.

![](dt=2026-05-29/ht=15/803257f21ca65d6ae7f1f604cde22c2a2b5268c0705a66172c38a6eb3b139471.jpg)

25

# E.4 Adaptation from an encoder-decoder

When studying adaptation and the conversion from one architecture to another, we also considered converting to and from encoder-decoder models. Conversion across causal and non-causal decoder-only models is straightforward, simply by switching the attention mask; for encoder-decoder, parameters have to be either pruned or added for both the entire encoder, and for the cross-attention in the decoder. Results from one of our attempt to convert an encoder-decoder into a causal decoder are reported in Figure 10. While converting across causal/non-causal decoder provides an improvement over training from scratch, this is not the case here.

![](dt=2026-05-29/ht=15/d553a48b13536cecc09c9e24d8dec6211c7bf89e02a41372e254cbd1810df974.jpg)

26