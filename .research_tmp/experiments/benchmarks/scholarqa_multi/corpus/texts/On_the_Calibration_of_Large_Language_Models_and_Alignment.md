# On Calibration of Large Language Models: From Response To Capability

Sin-Han Yang\*1 Cheng-Kuang Wu\*1 Chieh-Yen Lin 1 Yun-Nung Chen2 Hung-yi Lee2 Shao-Hua Sun 12

# Abstract

Large language models (LLMs) are widely deployed as general-purpose problem solvers, making accurate confidence estimation critical for reliable use. Prior work on LLM calibration largely focuses on response-level confidence, which estimates the correctness of a single generated output. However, this formulation is misaligned with many practical settings where the central question is how likely a model is to solve a query overall.

We show that this mismatch results from the stochastic nature of modern LLM decoding, under which single-response correctness fails to reflect underlying model capability. To address this issue, we introduce capability calibration, which targets the model's expected accuracy on a query. We formally distinguish capability calibration from response calibration and show that the two differ both theoretically and empirically. We establish an empirical evaluation setup and study a range of confidence estimation methods.

Our results demonstrate that capability-calibrated confidence improves pass@k prediction and inference budget allocation, establishing a foundation with potential for diverse applications. Source code: https://github.com/appier-research/llm-calibration.

# 1. Introduction

Large language models (LLMs) have fundamentally reshaped human-AI interaction by enabling users to pose queries in natural language and receive informative responses (Ouyang et al., 2022). This intuitive interface has driven their rapid adoption across a wide range of applications. However, despite their apparent fluency, LLMs can produce incorrect or misleading outputs without explicitly signaling uncertainty. This limitation makes accurate con

*Equal contribution; junior author listed earlier. †Appier AI Research 2National Taiwan University. Correspondence to: Sin-Han Yang <sinhan.yang@appier.com>, Cheng-Kuang Wu <brian.wu@appier.com>.

Preprint.

![](dt=2026-03-24/ht=06/e0445602de42ccb95ff46b217300fbfed2f65a027987445b92135c33a060c8dc.jpg)

fidence estimation a critical component of reliable LLM deployment. Well-calibrated confidence scores can enable users to better judge when to trust model outputs (Huang et al., 2024b; Aljohani et al., 2025), allow systems to selectively refuse or defer to human experts (Wu et al., 2024a), and support performance prediction for downstream tasks.

Given the important role of confidence estimation, a natural question is how to accurately evaluate its quality. Calibration (Niculescu-Mizil & Caruana, 2005; Guo et al., 2017) provides a principled evaluation framework by assessing how well estimated confidence aligns with correctness probability. Most existing work on LLM calibration (Geng et al., 2024) adopts a response-level formulation: given a query $x$ and a generated response $\hat{y}$ , a confidence estimator produces a score $s$ intended to reflect the probability that $\hat{y}$ is correct with respect to $x$ . Under this formulation, calibration is evaluated independently for each generated response. We refer to this setting as response calibration (Figure 1a).

In many practical settings, however, what matters is not whether a particular response $\hat{y}$ is correct, but how likely the LLM is to solve a given query overall. This question naturally arises in applications such as allocating computational resources across queries (Chen et al., 2023b; Ong et al., 2024) or predicting model performance in downstream pipelines. We refer to this quantity, "how likely can the LLM answer this query correctly?", as query-level confidence.

arXiv:2602.13540v1 [cs.CL] 14 Feb 2026

1

Table 1. Comparison of calibration definitions. Unlike existing response calibration that assesses whether the confidence estimate $s$ aligns with the correctness of one decoded answer $\hat{y}$ , our proposed capability calibration evaluates whether $s$ aligns with the model $f_{\theta}$ 's capability to answer query $x$ .

![](dt=2026-03-24/ht=06/1d033442cdb35957f5616848a40a5206c0c409aa23a1d005e5260c2f92987436.jpg)

<table><tr><td>Definition</td><td>Calibration Target</td><td>Interpretation</td><td>Dependence on LM fθ</td></tr><tr><td>Response Calibration</td><td>Accuracy of帽子 given x</td><td>How likely is帽子 correct?</td><td>No. Since帽子 is already decoded, the estimation of s(x,帽子) is decoupled from the generating model fθ.</td></tr><tr><td>Capability Calibration</td><td>Expected accuracy of fθ given x</td><td>How confident is fθ in answering x?</td><td>Yes. The expected accuracy of x is directly dependent on fθ&#x27;s capability.</td></tr></table>

Although response-level confidence is often used as a proxy for this quantity (Xiong et al., 2023; Maurya et al., 2025), the two are fundamentally misaligned in LLMs due to the stochastic nature of text generation. Modern LLMs typically achieve better performance with stochastic decoding (Holtzman et al., 2019; Shi et al., 2024), such as non-zero temperature sampling (Renze, 2024; Li et al., 2025a), which can produce different responses given the same query across inference calls. As a result, the correctness of any single sampled response cannot accurately reflect the LLM's underlying capability on that query. This mismatch between single-response correctness and query-level performance lies beyond what response calibration can capture.

Motivated by these observations, we introduce capability calibration (Figure 1b), a calibration framework whose target is the model's expected accuracy on a query $x$ — that is, the probability that a response sampled from the model's output distribution conditioned on $x$ is correct. This formulation shifts the focus from whether a particular sampled response happens to be correct to how capable the model is of solving the query in expectation.

We formally distinguish capability calibration from response calibration and show that the two notions differ both theoretically (\$3.2) and empirically (\$4.1). Notably, capability calibration is not merely the expectation of response calibration; the two quantities differ precisely by the variance of response correctness under the model's output distribution. We summarize the key differences between the two definitions in Table 1.

Having established that capability calibration is distinct from response calibration, we next consider how to evaluate and achieve it in practice. On the evaluation side, the theoretical target of expected accuracy under the model's output distribution is not directly observable. Hence, we develop an empirical evaluation framework that approximates capability calibration through repeated sampling. On the method side, we experiment with a wide range of confidence estimation techniques for producing calibrated scores, spanning both training-free and training-based techniques. Our results indicate that training linear probes on LLM activations offers a favorable tradeoff between computational cost and confidence estimation performance (§4.3.2).

Finally, we demonstrate that capability calibration enables practical applications (§5). We apply capability-calibrated confidence scores to two representative tasks: (1) pass@k prediction (Schaeffer et al., 2025; Kazdan et al., 2025), where confidence estimates are used to predict the pass@k success rate of individual queries without extensive sampling, and (2) inference budget allocation (Snell et al., 2024; Damani et al., 2024), where confidence estimates guide the allocation of computational resources across queries, with higher confidence requiring fewer resources.

In both settings, capability-calibrated confidence leads to improved performance over baselines. Beyond these applications, we discus
s additional scenarios where capability calibration can potentially provide tangible benefits. By formally defining capability calibration, establishing its evaluation framework, and demonstrating its practical utility, our work offers a new perspective on LLM calibration that directly captures model capability at the query level.

# 2. Related Works

# 2.1. LLM confidence estimation and calibration

Confidence estimation focuses on estimating the probability that predictions are correct. In machine learning, previous works (Niculescu-Mizil & Caruana, 2005; Guo et al., 2017) define calibration as the agreement between confidence and the correctness of an output. We call this definition response calibration. Denote $x$ as the input, $\hat{y}$ as the model's output, estimated confidence as $s(x, \hat{y})$ , and $\mathcal{C}(x, \hat{y}) \in \{0, 1\}$ as the correctness function. Formally,

$$
\mathcal {C} (x, \hat {y}) = \mathbf {1} [ \hat {y} \text {i s c o r r e c t f o r} x ]. \tag {1}
$$

Perfect response calibration is defined as:

$$
\mathbb {P} \left[ \mathcal {C} (x, \hat {y}) = 1 \mid s (x, \hat {y}) = p \right] = p. \tag {2}
$$

Common evaluation metrics include Expected Calibration Error (ECE) (Naeini et al., 2015) and Brier score (Brier, 1950). Let $\mathcal{D} = \{x_{i}\}_{i=1}^{N}$ be a dataset of inputs, $\mathbf{s} = [s_1, s_2, \ldots, s_i, \ldots, s_N]$ be the estimated confidence for each instance, and $\hat{y}_i$ be the sampled response for input $x_i$ . The

On Calibration of Large Language Models: From Response To Capability

2

Brier score used in response calibration is

$$
\mathcal {L} _ {\mathrm {B r i e r}} ^ {\mathrm {r e s p o n s e}} (\mathbf {s}) \triangleq \frac {1}{N} \sum_ {i = 1} ^ {N} \left(s _ {i} - \mathcal {C} \left(x _ {i}, \hat {y} _ {i}\right)\right) ^ {2}. \tag {3}
$$

LLM confidence estimation methods are broadly categorized into training-free and training-based approaches. Training-free methods include verbalized confidence (Lin et al., 2022; Tian et al., 2023), and token probability methods (Kadavath et al., 2022; Manakul et al., 2023). Training-based methods include probing LLMs' hidden states (Zhang et al., 2025a), reinforcement learning (Damani et al., 2025; Wu et al., 2025) and others (Li et al., 2025b). These methods typically operate post-hoc, estimating confidence only after the output is generated.

In contrast, a body of work on "assessors" focuses on anticipating the performance of a single response, aiming to estimate correctness before the response is generated (Zhou et al., 2022; Cencerrado et al., 2025; Schellaert et al., 2025). Detailed descriptions of these methods are in Appendix C.1.

Despite these methodological differences, the prediction target across these methods remains the same: they aim to estimate the correctness of a single response. However, LLMs are stochastic generative models. While recent work (Zhang et al., 2025d) aggregates statistics over multiple samples to ensure a more robust measure of performance, it does not formalize a calibration target for these stochastic outcomes. We fill this gap by defining capability calibration, establishing the model's query-level expected accuracy as the precise target for confidence estimation.

# 2.2. LLM uncertainty quantification

LLM Uncertainty Quantification (UQ) is a field of methods that quantify the degree of uncertainty of the model towards specific inputs. While calibration measures the alignment between confidence scores and output correctness, UQ is often evaluated by uncertainty estimation's utility in downstream decisions (Huang et al., 2024a), such as discriminating between correct and incorrect predictions. Consequently, common evaluation metrics include the Area Under the Receiver Operating Characteristic curve (AUROC) (Hendrycks & Gimpel, 2016) and the Risk-Coverage curve (Geifman & El-Yaniv, 2017).

Existing LLM UQ methods include: token-based approaches (Kadavath et al., 2022; Duan et al., 2024), sampling-based approaches (Wang et al., 2022; Kuhn et al., 2023; Cecere et al., 2025), and methods leveraging the models' internal signals (Cohen et al., 2024; Chen et al., 2025). Detailed descriptions of these methods are in Appendix C.1.

Capability calibration is linked to LLM UQ, as it utilizes expected accuracy as the target for the estimated model's uncertainty regarding a specific input. This makes capability-calibrated confidence estimations a natural fit for LLM UQ

applications, such as selective prediction (Kamath et al., 2020), hallucination detection (Kang et al., 2025), and model routing (Chen et al., 2023b).

# 3. Capability Calibration

Large language model's output is mostly non-deterministic. In this paper, we consider the expected accuracy of the LLM's output distribution, and propose a new definition of calibration called capability calibration. Capability calibration evaluates whether the estimated confidence agrees with the model's likelihood to answer an input correctly.

# 3.1. Definition

For a given input $x$ , we define the model's expected accuracy as

$$
\mu (x, f _ {\theta}) \triangleq \mathbb {P} _ {\hat {y} \sim f _ {\theta} (\cdot | x)} [ \mathcal {C} (x, \hat {y}) = 1 ] = \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x)} [ \mathcal {C} (x, \hat {y}) ], \tag {4}
$$

which is equivalent to

$$
\mu (x, f _ {\theta}) = \lim  _ {N \rightarrow \infty} \frac {1}{N} \sum_ {i = 1} ^ {N} \mathcal {C} (x, \hat {y} _ {i}), \quad \hat {y} _ {i} \sim f _ {\theta} (\cdot | x). \tag {5}
$$

Equation (4) defines the target of the capability calibration. As illustrated in Figure 1, the target expected accuracy $\mu(x, f_{\theta})$ is defined as the frequency of correct outputs when the language model is sampled infinitely many times on the same input $x$ .

An estimated confidence $s$ is well capability-calibrated if it is aligned with the expected accuracy $\mu(x, f_{\theta})$ . The perfect calibration for capability calibration is

$$
s ^ {*} = \mu (x, f _ {\theta}). \tag {6}
$$

Capability calibration focuses on calibrating a single input. Therefore, we primarily discuss the Brier score (Brier, 1950), one of the most common metrics used to evaluate instance-level calibration. Specifically, let $\mu_{i}$ be the expected accuracy $\mu (x_i,f_\theta)$ , the capability calibration Brier score is

$$
\mathcal {L} _ {\text {B r i e r}} ^ {\text {c a p a b i l i t y}} (\mathbf {s}) \triangleq \frac {1}{N} \sum_ {i = 1} ^ {N} \left(s _ {i} - \mu_ {i}\right) ^ {2}. \tag {7}
$$

Since LLM outputs are not deterministic (Renze, 2024; Li et al., 2025a; He & Thinking Machines Lab, 2025), different token generation paths might result in different answers. For each input, a single sampled output is insufficient to represent the model's capability. Capability calibration better captures a model's capability since it cares about the

On Calibration of Large Language Models: From Response To Capability

3

agreement between confidence and accuracy of all sampled responses, while response calibration cares about the agreement between confidence and accuracy of one sampled response. Next, we discuss the difference between response calibration and capability calibration.

# 3.2. Difference between response calibration and capability calibration

We argue that these two evaluations diverge in three key aspects:

Theorem 1. (Divergence of targets and optima). Let $x$ be an input and $\hat{y} \sim f_{\theta}(\cdot \mid x)$ be a generated response. Minimizing the Brier scores for response calibration (Equation (3)) and capability calibration (Equation (7)) yields distinct optimal confidence estimators:

$$
\begin{array}{l} s _ {\mathrm {r e s p}} ^ {*} (x, \hat {y}) = \mathcal {C} (x, \hat {y}) \in \{0, 1 \}, \\ s _ {\mathrm {c a p}} ^ {*} (x, f _ {\theta}) = \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x)} [ \mathcal {C} (x, \hat {y}) ] \in [ 0, 1 ]. \tag {8} \\ \end{array}
$$

Unless the model is deterministic, or its
predictions are always correct or always incorrect, the evaluation targets differ; i.e., $\mathcal{C}(x,\hat{y})\neq \mathbb{E}_{\hat{y}\sim f_{\theta}(\cdot |x)}[\mathcal{C}(x,\hat{y})]$ , implying distinct optimal confidence values.

See Appendix A.1 for the proof. This theoretical divergence is empirically confirmed in Figure 2 and Section 4.1, where the targets are shown to differ significantly in practice. Having established that these objectives are distinct, we now formalize the connection between the response calibration loss function and the capability calibration loss function:

Theorem 2. (Decomposition of calibration losses). Given a set of estimated confidence $\mathbf{s}$ , define the expectation of response calibration loss $\mathcal{L}_{\mathrm{Brier}}^{\mathrm{response}}$ on the model's output distribution as

$$
\mathbb {E} \big [ \mathcal {L} _ {\mathrm {B r i e r}} ^ {\mathrm {r e s p o n s e}} \big ] \triangleq \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} \left[ (s _ {i} - \mathcal {C} (x _ {i}, \hat {y})) ^ {2} \right].
$$

![](dt=2026-03-24/ht=06/1dae2eb5863d0b49df5f6d0301b947cbe8ebc5b57cc31cdc0cb9a18e87e2fded.jpg)

Decoupling output correctness variance from response calibration, we get capability calibration:

$$
\mathcal {L} _ {\text {B r i e r}} ^ {\text {c a p a b i l i t y}} = \underbrace {\mathbb {E} [ \mathcal {L} _ {\text {B r i e r}} ^ {\text {r e s p o n s e}} ]} _ {\text {r e s p o n s e c a l i b r a t i o n}} - \underbrace {\frac {1}{N} \sum_ {i = 1} ^ {N} V a r (\mathcal {C} (x _ {i} , \hat {y}))} _ {\text {o u t p u t c o r r e c t n e s s v a r i a n c e}}, \tag {9}
$$

where

$$
\begin{array}{l} V a r \left(\mathcal {C} \left(x _ {i}, \hat {y}\right)\right) = \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} \left[ \mathcal {C} \left(x _ {i}, \hat {y}\right) ^ {2} \right] \tag {10} \\ - \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} (x _ {i}, \hat {y}) ] ^ {2}. \\ \end{array}
$$

See Appendix A.2 for the proof. Theorem 2 demonstrates that when evaluating a set confidence estimation, capability calibration decouples the model's output variance from response calibration. While response calibration penalizes the stochasticity of generated outputs, capability calibration targets the model's underlying probability of correctness. For strictly convex and differentiable losses, the difference $\mathbb{E}[\mathcal{L}^{response}] - \mathcal{L}^{capability}$ generalizes to the Bregman information (Banerjee et al., 2005), which quantifies the gap (Gruber & Buettner, 2022) caused by output randomness. See Appendix A.3 for detailed discussion.

# 4. Measuring Capability Calibration

# 4.1. Evaluation framework

For a given query $x$ , an LLM's theoretical expected accuracy $\mu$ defined in Equation (4) is not directly accessible, so one has to estimate $\mu$ empirically. For a given $x$ , we estimate the LLM's expected accuracy by sampling $k_{\mathrm{eval}}$ responses $\{\hat{y}_1,\dots ,\hat{y}_{k_{\mathrm{eval}}}\}$ from $f_{\theta}(\cdot \mid x)$ . Let $c$ denote the number of correct responses. The estimated expected accuracy is:

$$
\hat {\mu} = \frac {c}{k _ {\mathrm {e v a l}}}.
$$

On Calibration of Large Language Models: From Response To Capability

4

Target differs from single response correctness. Next, we investigate whether estimated expected accuracy or single-response correctness are empirically different calibration targets. In Figure 2, we compare these two targets using Olmo-3-7B-Instruct on TriviaQA (see Section 4.3.1 for setup). Additional results are in Appendix E. Consistent with findings in Zhang et al. (2025d), our experiments show that LLM outputs are rarely binary; they are neither perfectly deterministic nor consistently correct across inference calls. This variance confirms that capability calibration targets a fundamentally different property than response calibration.

Evaluation metric. Since capability calibration targets query-level performance, we require a metric that preserves per-query granularity. Following the discussion in Section 3.1, we use Brier score to measure calibration quality. We choose Brier score over ECE because ECE's binning procedure averages predictions within each bin, which could mask calibration errors of individual queries. Given a dataset of $N$ queries, let $s_i$ denote the confidence estimate for query $x_i$ and $\hat{\mu}_i$ the estimated expected accuracy. The empirical capability calibration Brier score is defined as:

$$
\mathcal {L} _ {\mathrm {B r i e r}} ^ {\mathrm {c a p a b i l i t y}} (\mathbf {s}) = \frac {1}{N} \sum_ {i = 1} ^ {N} (s _ {i} - \hat {\mu} _ {i}) ^ {2}.
$$

Lower Brier scores indicate better calibration.

# 4.2. Methods for confidence estimation

Uniform random baseline. To assess whether a method delivers meaningful performance, we establish a baseline that uses no information about the query. For each query $x_{i}$ , we sample a confidence score from a uniform distribution $s_i \sim U(0,1)$ . This baseline admits an analytic expected loss (see Appendix C.2 for derivation):

$$
\mathbb {E} _ {\mathbf {s}} \left[ \mathcal {L} _ {\text {B r i e r}} ^ {\text {c a p a b i l i t y}} (\mathbf {s}) \right] = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\frac {1}{3} - \hat {\mu} _ {i} + \hat {\mu} _ {i} ^ {2}\right), \tag {11}
$$

which depends only on the model's expected accuracy on the dataset. Any useful confidence estimation method should outperform this baseline.

Next, we introduce confidence estimation methods commonly used in LLM response calibration, and adapt them to our capability calibration setting.

Response consistency (Wang et al., 2022). A straightforward way to estimate confidence is by measuring the consistency across multiple sampled responses. We sample $k_{c}$ responses and compute the fraction that agree with the majority prediction. For example, if $k_{c} = 10$ and 7 responses are equivalent, the confidence estimate would be 0.7. Note that this method incurs a higher computational cost than other methods, as it requires $k_{c}$ forward passes per query.

Verbalized confidence. We instruct the LLM to report a probability in [0, 1] in natural language. Unlike prior work (Lin et al., 2022; Tian et al., 2023), which asks for confidence in the response given the query, we ask for confidence in the query itself to measure query-level capability. The prompt is provided in Appendix C.2.

P(True). We ask the model whether it can answer the query correctly by instructing it to respond with only "Yes" or "No". We extract the logprobs of these two tokens and use the softmax probability of "Yes" as the confidence estimate. Unlike prior work (Kadavath et al., 2022), which provides both the query and the response, we present only the query. The prompt is provided in Appendix C.2.

Probing LLMs' hidden states (Li et al., 2021). We train linear probes on LLMs' internal representations to predict query-level confidence. Specifically, we mean-pool activations from the last input token across transformer blocks to output a confidence score. This approach incurs minimal overhead, with an inference cost less than decoding a single token. See Appendix C.2 for implementation details.

Notable properties: Response consistency and verbalized confidence are black-box methods applicable to API-based LLMs without access to token logprobs. P(True) is a gray-box method requiring access to token logprobs. Probing is a white-box method requiring open-weight models.

# 4.3. Experiments

# 4.3.1. SETUP

Choice of $k_{\mathrm{eval}}$ . The estimated expected accuracy $\hat{\mu}$ is a binomial proportion with variance $\mu (1 - \mu) / k_{\mathrm{eval}}$ , which decreases as $k_{\mathrm{eval}}$ increases. We investigate the effect of $k_{\mathrm{eval}}$ on evaluation reliability in Appendix
B.1 and chose $k_{\mathrm{eval}} = 100$ to balance cost and reliability. We then evaluate the methods on three LLMs across seven datasets:

Models. We use Olmo-3-7B-Instruct (Olmo Team et al., 2025), Qwen3-8B (Yang et al., 2025), and gpt-oss-20b (Agarwal et al., 2025) to capture model diversity. Sampling hyperparameters follow Appendix B.2.

Datasets. We select datasets from three task domains: (1) factual knowledge, which tests parametric knowledge; (2) mathematical reasoning, where errors compound across multiple intermediate steps; and (3) general exams, which test both knowledge and reasoning in multiple subjects. For each type, we include datasets of different difficulty levels.

Factual knowledge: We choose TriviaQA (Joshi et al., 2017) as the easier dataset and SimpleQA verified (Haas et al., 2025) as the harder one.

Mathematical reasoning: We adopt GSM8K (Cobbe et al., 2021) as the easiest dataset, MATH-500 (Lightman et al.,

On Calibration of Large Language Models: From Response To Capability

5

Table 2. Capability calibration performance of different methods with three LLMs on seven datasets. For probes, we use different colors to indicate in-domain in-distribution, in-domain out-of-distribution, and out-domain performance. We use bold to denote the best calibrated method, and underline to denote the second best. Probe performs the best under in-domain in-distribution settings and generalizes reasonably well under in-domain out-distribution settings. Verbalized confidence and P(True) results differ across LLMs.

![](dt=2026-03-24/ht=06/69441b7d0c9590763263159f3642719637a53541ca31f869ba9a9cfec256d9b1.jpg)

<table><tr><td rowspan="2">Brier score (↓) 
Method</td><td rowspan="2">Domain 
Cost</td><td colspan="2">Factual knowledge</td><td colspan="3">Mathematical reasoning</td><td colspan="2">General exams</td></tr><tr><td>TriviaQA</td><td>SimpleQA</td><td>GSM8K</td><td>MATH</td><td>AIME25</td><td>MMLU</td><td>GPQA</td></tr><tr><td colspan="9">Olmo-3-7B-Instruct</td></tr><tr><td>Uniform random baseline</td><td>N/A</td><td>0.2745</td><td>0.3133</td><td>0.3119</td><td>0.2940</td><td>0.2462</td><td>0.2565</td><td>0.2125</td></tr><tr><td>Verbalized confidence</td><td>L</td><td>0.2624</td><td>0.2676</td><td>0.0462</td><td>0.0557</td><td>0.2002</td><td>0.1561</td><td>0.2742</td></tr><tr><td>P(True)</td><td>1</td><td>0.1933</td><td>0.0419</td><td>0.1282</td><td>0.1400</td><td>0.1854</td><td>0.2164</td><td>0.1553</td></tr><tr><td>Probe (train on TriviaQA)</td><td>&lt; 1</td><td>0.1113</td><td>0.0386</td><td>0.1180</td><td>0.1273</td><td>0.1496</td><td>0.1300</td><td>0.1242</td></tr><tr><td>Probe (train on GSM8K)</td><td>&lt; 1</td><td>0.2648</td><td>0.5465</td><td>0.0370</td><td>0.0545</td><td>0.2482</td><td>0.1200</td><td>0.1628</td></tr><tr><td>Probe (train on MATH)</td><td>&lt; 1</td><td>0.2550</td><td>0.4846</td><td>0.0388</td><td>0.0394</td><td>0.1411</td><td>0.1255</td><td>0.1295</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Uniform random baseline</td><td>N/A</td><td>0.2865</td><td>0.3109</td><td>0.3144</td><td>0.2781</td><td>0.2800</td><td>0.2868</td><td>0.2113</td></tr><tr><td>Verbalized confidence</td><td>L</td><td>0.2431</td><td>0.4736</td><td>0.0461</td><td>0.0962</td><td>0.4443</td><td>0.1293</td><td>0.2773</td></tr><tr><td>P(True)</td><td>1</td><td>0.2970</td><td>0.6072</td><td>0.0482</td><td>0.1126</td><td>0.4957</td><td>0.1597</td><td>0.3448</td></tr><tr><td>Probe (train on TriviaQA)</td><td>&lt; 1</td><td>0.1079</td><td>0.0638</td><td>0.3177</td><td>0.3219</td><td>0.1286</td><td>0.2006</td><td>0.1556</td></tr><tr><td>Probe (train on GSM8K)</td><td>&lt; 1</td><td>0.1885</td><td>0.4451</td><td>0.0368</td><td>0.0715</td><td>0.0740</td><td>0.1176</td><td>0.1556</td></tr><tr><td>Probe (train on MATH)</td><td>&lt; 1</td><td>0.2977</td><td>0.8297</td><td>0.0408</td><td>0.0475</td><td>0.0831</td><td>0.1163</td><td>0.1811</td></tr><tr><td colspan="9">gpt-oss-20b</td></tr><tr><td>Uniform random baseline</td><td>N/A</td><td>0.2639</td><td>0.3010</td><td>0.3195</td><td>0.3063</td><td>0.2369</td><td>0.3018</td><td>0.2388</td></tr><tr><td>Verbalized confidence</td><td>L</td><td>0.1266</td><td>0.1957</td><td>0.0268</td><td>0.0275</td><td>0.0460</td><td>0.0559</td><td>0.1174</td></tr><tr><td>P(True)</td><td>L</td><td>0.2101</td><td>0.6151</td><td>0.0306</td><td>0.0321</td><td>0.1092</td><td>0.0817</td><td>0.2082</td></tr><tr><td>Probe (train on TriviaQA)</td><td>&lt; 1</td><td>0.0845</td><td>0.0600</td><td>0.0780</td><td>0.1593</td><td>0.1457</td><td>0.0977</td><td>0.1533</td></tr><tr><td>Probe (train on GSM8K)</td><td>&lt; 1</td><td>0.1756</td><td>0.7048</td><td>0.0289</td><td>0.0485</td><td>0.1213</td><td>0.0686</td><td>0.2010</td></tr><tr><td>Probe (train on MATH)</td><td>&lt; 1</td><td>0.1577</td><td>0.5871</td><td>0.0332</td><td>0.0267</td><td>0.1644</td><td>0.0922</td><td>0.1363</td></tr></table>

2023) as the intermediate one, and AIME25 as the hardest.

General exams: We use MMLU (Hendrycks et al., 2020), which spans 57 subjects in humanities, social science, STEM, and others. We also use GPQA (Rein et al., 2023), which is harder than MMLU and includes graduate-level questions in biology, chemistry, and physics.

# 4.3.2. RESULTS AND DISCUSSION

Probing has the best cost-performance tradeoff. A practically useful method should satisfy two properties: (1) Acceptable inference cost: no higher than decoding the response itself, otherwise the overhead would limit the method's practical utility (see §5). (2) Good calibration performance: lower Brier score is better. Figure 3 shows a representative example, with full results shown in Figure 4. Among evaluated methods, probing has the lowest inference cost while consistently outperforming the random baseline.

How well does probing generalize? Table 2 shows that probing performs well under in-domain, in-distribution settings. However, some applications may require applying a confidence estimator to (1) same-domain but out-of-distribution queries (e.g., different factual knowledge datasets), or (2) out-of-domain queries (e.g., training on factual knowledge but applying to mathematical reasoning). Overall, probing generalizes reasonably well under

![](dt=2026-03-24/ht=06/17e51f2605b50e61bbc179d8a133c74020c0693890f689a1a25db747d36b8706.jpg)

in-domain, out-of-distribution settings, especially for the factual knowledge domain. However, it does not consistently generalize to out-of-domain settings. Developing generalizable methods for capability calibration remains an important direction for future work.

On Calibration of Large Language Models: From Response To Capability

6

![](dt=2026-03-24/ht=06/7d0e681961953711c23bebc7f4bbbe26092605028a1fa26a7984f299e8e80986.jpg)

Performance of verbalized confidence and P(True) differs across LLMs. As shown in Table 2, gpt-oss-20b performs strikingly well with verbalized confidence, achieving the best or second-best performance across datasets. In contrast, Olmo-3-7B-Instruct and Qwen3-8B do not even consistently outperform the random uniform baseline with verbalized confidence. Moreover, verbalized confidence outperforms P(True) for Qwen3-8B and gpt-oss-20b, but not for Olmo-3-7B-Instruct. These results suggest that the effectiveness of these methods is model-dependent.

Response consistency costs more than responding. This method costs more than decoding the response itself (see Figure 3 and 4), rendering it impractical for applications where query-level confidence must be estimated before decoding, such as for inference budget allocation (see §5.2). We report its calibration performance in Appendix C.3.

# 5. Applications

In this section, we show that capability calibration has broad applicability to several
applications.

# 5.1.Pass@k simulation

Test-time scaling via repeated sampling has been shown to enhance LLM capabilities (Brown et al., 2024), while simultaneously increasing vulnerability to AI safety risks (Schaeffer et al., 2025; Kazdan et al., 2025). Given these trade-offs, the ability to estimate resampling performance at a low inference cost is critical for both researchers and developers (Stroebl et al., 2024). A common approach to this problem, as proposed by Kazdan et al. (2025), is to predict pass@k performance by sampling only a small subset of outputs. Their method assumes that the expected accuracy of each instance in a dataset follows a beta distribution.

In this section, we show that capability-calibrated confidence estimation can simulate the pass@k performance of each instance without (1) sampling multiple outputs and (2) assuming a prior distribution over the dataset. Furthermore, by computing the pass@k success rate for each instance, we can estimate the pass@k curve for the entire dataset. We discuss the simulation process in Appendix D.1.1.

We evaluate three confidence estimators: (1) Oracle Response-Calibrated (Oracle-RC), (2) Oracle Capability-Calibrated (Oracle-CC), and (3) Probe-MATH, which is trained on the MATH-train dataset (Hendrycks et al., 2021) with CC target. To calculate the real pass@k performance, we use the unbiased estimator Chen et al. (2021). We use Mean Squared Error (MSE) to measure the instance-level discrepancy between simulated and actual pass@k performance. Table 3 presents the simulation results for MATH500. Results for AIME25 and results on the dataset-level pass@k curve are provided in Appendix D.1.2.

Experiment results demonstrate the effectiveness of capability calibration for pass@k simulation. Since Oracle-CC is the expected accuracy defined in Equation (4), it simulates groundtruth performance almost perfectly. In contrast, Oracle-RC focuses on single-response correctness, which is a noisy estimate of expected accuracy, causing MSE to increase at higher k. Finally, Probe-MATH outperforms Oracle-RC by effectively approximating the expected accuracy.

# 5.2. Inference budget allocation

Allocating test-time computation has been shown to improve language model performance (Damani et al., 2024; Zhang et al., 2024; Snell et al., 2024). In the best-of- $k$ setting, Damani et al. (2024) investigates how to solve as many problems as possible under a fixed sampling budget. Their approach involves distributing the total computational

On Calibration of Large Language Models: From Response To Capability

7

Table 3. Pass@k simulation error (MSE) on the MATH-500 dataset. We evaluate the ability of different confidence estimators to simulate empirical pass@k performance. Perfectly capability-calibrated confidence (Oracle CC) achieves near-perfect simulation, whereas the error of perfectly response-calibrated confidence (Oracle RC) increases as $k$ scales. Notably, our trained estimator (Probe-MATH) outperforms the Oracle RC baseline across all models by approximating the model's expected accuracy.

![](dt=2026-03-24/ht=06/248a5a670f6c794c083d3afee3d317ad8c97ef415fdea6c8c256ca98224e892a.jpg)

<table><tr><td>Method</td><td>pass@1</td><td>pass@4</td><td>pass@16</td><td>pass@64</td></tr><tr><td colspan="5">Olmo-3-7B-Instruct</td></tr><tr><td>Oracle RC</td><td>0.0370</td><td>0.0556</td><td>0.0746</td><td>0.0935</td></tr><tr><td>Oracle CC</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0003</td></tr><tr><td>Probe-MATH</td><td>0.0394</td><td>0.0386</td><td>0.0243</td><td>0.0148</td></tr><tr><td colspan="5">Qwen3-8B</td></tr><tr><td>Oracle RC</td><td>0.0543</td><td>0.0872</td><td>0.1225</td><td>0.1486</td></tr><tr><td>Oracle CC</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0003</td></tr><tr><td>Probe-MATH</td><td>0.0475</td><td>0.0446</td><td>0.0304</td><td>0.0205</td></tr><tr><td colspan="5">gpt-oss-20b</td></tr><tr><td>Oracle RC</td><td>0.0271</td><td>0.0402</td><td>0.0545</td><td>0.0629</td></tr><tr><td>Oracle CC</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0001</td></tr><tr><td>Probe-MATH</td><td>0.0267</td><td>0.0175</td><td>0.0099</td><td>0.0063</td></tr></table>

budget across a dataset of queries prior to generating answers. They optimize the budget allocation by allocating more resources to questions based on their difficulty, which has been shown to outperform uniform allocation. Specifically, they learn a reward model to estimate the marginal improvement (gain) in the success rate achieved by allocating one additional unit of compute to a query. The detailed algorithm is discussed in Appendix D.2.1.

The "gain" metric defined by Damani et al. (2024) relies directly on expected accuracy formulated in Equation (4). Consequently, capability-calibrated confidence allows us to analytically estimate this gain and apply the greedy allocation algorithm detailed in Appendix D.2.1 for inference budget allocation. We evaluate three confidence estimators: (1) Oracle, the perfectly capability-calibrated confidence; (2) Probe-MATH, a high-performing confidence estimator equivalent to the Online Ada-BoK method (Damani et al., 2024); and (3) Verbalized Confidence (Verbalized), an estimator that is applicable to black-box models.

Experimental results validate the effectiveness of capability-calibrated confidence in inference budget allocation. Figure 5 illustrates the performance of gpt-oss-20b on MATH-500; additional results for other models and datasets are provided in Appendix D.2.2. Consistent with findings in Damani et al. (2024), the Oracle estimator yields the best performance across all compute budgets, and Probe-MATH consistently outperforms uniform allocation. Furthermore, we discover that verbalized confidence achieves results comparable to Probe-MATH without requiring access to internal model states. This implies that the performance benefits of leveraging capability-calibrated confidence can be applied to API-based LLMs.

![](dt=2026-03-24/ht=06/f79bd3061c1b31b1b9cca501a4796f73bdf5cd92931b1bbefc8a073156f7a1f9.jpg)

# 5.3. Other applications

Beyond our primary experiments, capability-calibrated confidence can enhance system reliability through selective prediction (Kamath et al., 2020) and active query refinement (Wu et al., 2024a). It also supports efficient infrastructure via model routing (Ong et al., 2024) and cost estimation (Wu et al., 2024b), as well as advanced training techniques like curriculum learning (Zhang et al., 2025e) and label-free benchmarking (Guha et al., 2024).

As an initial investigation into capability calibration, we prioritize two critical applications ( $\S 5.1$ and $\S 5.2$ ) where performance is directly related to the model's expected accuracy. Although we also identify other promising applications, a comprehensive empirical evaluation of all downstream tasks is beyond the scope of this work. Nonetheless, we provide a conceptual discussion of how capability calibration can be integrated into these broader domains in Appendix D.3.

# 6. Conclusion

This work formalizes capability calibration and shows that it differs from response calibration due to the stochastic nature of LLM outputs. Our experiments identify linear probing on model activations as a practical method that achieves nontrivial calibration performance at minimal computational overhead, and demonstrate its downstream value through efficient pass@ $k$ prediction and inference budget allocation. We see two promising research directions: (1) developing methods that push the frontier of capability calibration performance; (2) extending this framework to more applications, such as model routing, human-AI collaboration, and trustworthy AI.

On Calibration of Large Language Models: From Response To Capability

8

# Impact Stateme
nt

This paper presents work whose goal is to advance the field of Machine Learning by improving the reliability and predictability of LLMs. As LLMs are increasingly deployed in real-world applications, ensuring they are trustworthy is paramount. Our framework for capability calibration enables models to more accurately assess their own limitations, allowing systems to abstain or seek human oversight when the model is unlikely to succeed. We believe this contributes to safer AI deployment by mitigating the risks associated with overconfidence and hallucination. There are no significant negative societal consequences that we feel must be specifically highlighted here.

# Acknowledgements

We would like to thank Appier AI Research team members, Hsuan-Tien Lin (National Taiwan University) and Wei-Lin Chen (University of Virginia), for their feedback on this work. This work was supported in part by the National Science and Technology Council, Taiwan, under the Grant 114-2628-E-002-021-, and the Taiwan Centers of Excellence. Shao-Hua Sun was supported by the Yushan Fellow Program of the Ministry of Education, Taiwan.

# References

On Calibration of Large Language Models: From Response To Capability

9

On Calibration of Large Language Models: From Response To Capability

10

On Calibration of Large Language Models: From Response To Capability

11

On Calibration of Large Language Models: From Response To Capability

12

# Appendix

The appendix contains the following section.

# - Details of Capability Calibration 14

# - Details of Experiment Setup 16

# - Details of Confidence Estimation Methods 18

# - Details of the Applications 21

# - Targets Difference 25

On Calibration of Large Language Models: From Response To Capability

13

# A. Details of Capability Calibration

# A.1. Proof of Theorem 1

Proof. Let $f_{\theta}(\cdot \mid x)$ be a generative model. For each input $x_{i}$ , let $\hat{y}_i \sim f_{\theta}(\cdot \mid x_i)$ be a single sampled output, and let $s_i \in [0,1]$ denote the predicted confidence.

Recall that the response calibration Brier score (Equation (3)) is defined as:

$$
\mathcal {L} _ {\mathrm {B r i e r}} ^ {\mathrm {r e s p o n s e}} (s _ {1}, \ldots , s _ {N}) \triangleq \frac {1}{N} \sum_ {i = 1} ^ {N} (s _ {i} - \mathcal {C} (x _ {i}, \hat {y} _ {i})) ^ {2}, \quad \hat {y} _ {i} \sim f _ {\theta} (\cdot | x _ {i}).
$$

Define the confidence estimations of the dataset as $\mathbf{s} = (s_1, s_2, \dots, s_n)$ . Since the Brier score is convex, the stationary point is the global minimum, which is

$$
\nabla \mathcal {L} _ {\mathrm {B r i e r}} ^ {\mathrm {r e s p o n s e}} (\mathbf {s} ^ {*}) = \mathbf {0},
$$

i.e.,

$$
\frac {\partial \mathcal {L} _ {\mathrm {B r i e r}} ^ {\mathrm {r e s p o n s e}}}{\partial s _ {i}} = \frac {2}{N} (s _ {i} - \mathcal {C} (x _ {i}, \hat {y} _ {i})) = 0, \quad \forall i \in \{1, \ldots , N \}.
$$

Thus, the optimal confidence estimation is

$$
s _ {i} ^ {\mathrm {*, r e s p o n s e}} = \mathcal {C} (x _ {i}, \hat {y} _ {i}), \quad \forall i \in \{1, \ldots , N \}.
$$

Follow the definition of Equation (4)

$$
\mu_ {i} \triangleq \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} (x _ {i}, \hat {y}) ],
$$

we restate the capability calibration Brier score from Equation (7):

$$
\mathcal {L} _ {\mathrm {B r i e r}} ^ {\mathrm {c a p a b i l i t y}} (s _ {1}, \ldots , s _ {N}) \triangleq \frac {1}{N} \sum_ {i = 1} ^ {N} (s _ {i} - \mu_ {i}) ^ {2} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(s _ {i} - \mathbb {E} _ {\hat {y} \sim f _ {\theta (\cdot | x _ {i})}} [ \mathcal {C} (x _ {i}, \hat {y}) ]\right) ^ {2}.
$$

Following the same proof, the optimal confidence that minimize $\mathcal{L}_{\mathrm{Brier}}^{\mathrm{capability}}$ satisfies

$$
s _ {i} ^ {\text {*, c a p a b i l i t y}} = \mu_ {i}, \quad \forall i \in \{1, \dots , N \}.
$$

For a generative model, $\mathcal{C}(x_i,\hat{y}_i)\in \{0,1\}$ , while $\mu_{i}\in [0,1]$ . Unless the model is deterministic or perfectly correct/incorrect on $x_{i}$ , we have

$$
\mathcal {C} \left(x _ {i}, \hat {y} _ {i}\right) \neq \mu_ {i} \triangleq \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} \left(x _ {i}, \hat {y}\right) ].
$$

Therefore,

$$
\left(s _ {1} ^ {\text {*, r e s p o n s e}}, \ldots , s _ {N} ^ {\text {*, r e s p o n s e}}\right) \neq \left(s _ {1} ^ {\text {*, c a p a b i l i t y}}, \ldots , s _ {N} ^ {\text {*, c a p a b i l i t y}}\right),
$$

i.e., the two calibration objectives induce different optimal confidence predictors over the dataset.

# A.2. Proof of Theorem 2

Proof. First, expand the expectation of the Brier Score loss on the old calibration

$$
\mathbb {E} \left[ \mathcal {L} _ {\text {B r i e r}} \right] \triangleq \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} \left[ \left(s _ {i} - \mathcal {C} \left(x _ {i}, \hat {y}\right)\right) ^ {2} \right] = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(s _ {i} ^ {2} - 2 s _ {i} \cdot \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} \left(x _ {i}, \hat {y}\right) ] + \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} \left(x _ {i}, \hat {y}\right) ^ {2} ]\right). \tag {12}
$$

Second, expand the Brier Score loss on expected calibration

$$
\mathcal {L} _ {\text {B r i e r}} ^ {\text {e x p e c t e d}} \triangleq \frac {1}{N} \sum_ {i = 1} ^ {N} \left(s _ {i} - \mu_ {i}\right) ^ {2} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(s _ {i} ^ {2} - 2 s _ {i} \mu_ {i} + \mu_ {i} ^ {2}\right), \tag {13}
$$

On Calibration of Large Language Models: From Response To Capability

14

Subtracting Equation (13) - Equation (12), we obtain

$$
\begin{array}{l} \mathcal {L} _ {\mathrm {B r i e r}} ^ {\mathrm {e x p e c t e d}} - \mathbb {E} [ \mathcal {L} _ {\mathrm {B r i e r}} ] = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\mu_ {i} ^ {2} - 2 s _ {i} \cdot \left(\mu_ {i} - \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} (x _ {i}, \hat {y}) ]\right) - \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} (x _ {i}, \hat {y}) ^ {2} ]\right) \\ = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\mu_ {i} ^ {2} - \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} (x _ {i}, \hat {y}) ^ {2} ]\right) \\ = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} (x _ {i}, \hat {y}) ] ^ {2} - \mathbb {E} _ {\hat {y} \sim f _ {\theta} (\cdot | x _ {i})} [ \mathcal {C} (x _ {i}, \hat {y}) ^ {2} ]\right) \tag {14} \\ = - \frac {1}{N} \sum_ {i = 1} ^ {N} \operatorname {V a r} [ \mathcal {C} (x _ {i}, \hat {y}) ]. \\ \end{array}
$$

![](dt=2026-03-24/ht=06/7d0204840178322173dfc07d38ef8cfe5d71bb1a55a46dd944bda9c438a2f622.jpg)

# A.3. Connection between response calibration loss and capability calibration loss

For the same loss function $\mathcal{L}(s,t)$ that evaluates the agreement between confidence $s$ and target $t$ , capability calibration and response calibration evaluate $s$ with different $t$ . Denote the loss function as $\mathcal{L}_s(t)$ when we discuss the impact of $t$ .

Capability calibration takes the expected accuracy as the target. In general, capability calibration loss is

$$
\mathcal {L} _ {s} \left(\mathbb {E} _ {\hat {y} \sim f (\cdot | x)} [ \mathcal {C} (x, \hat {y}) ]\right),
$$

we simplify it as $\mathcal{L}_s(\mathbb{E}[\mathcal{C}(x,\hat{y})])$

Response calibration takes the individual response's correctness as the target. Therefore, the expectation of the response calibration loss is

$$
\mathbb {E} _ {\hat {y} \sim f (\cdot | x)} [ \mathcal {L} _ {s} (\mathcal {C} (x, \hat {y})) ],
$$

we simplify it as $\mathbb{E}[\mathcal{L}_s(\mathcal{C}(x,\hat{y}))]$

The difference between the capability calibration loss and the response calibration lo
ss

$$
\mathbb {E} \left[ \mathcal {L} _ {s} \left(\mathcal {C} (x, \hat {y})\right) \right] - \mathcal {L} _ {s} \left(\mathbb {E} \left[ \mathcal {C} (x, \hat {y}) \right]\right), \tag {15}
$$

is known as Jensen Gap (Reid & Williamson, 2011). For loss functions that are strictly convex and differentiable, Jensen Gap is equivalent to the Bregman information (Banerjee et al., 2005), which measures the diversity of the random variable $\mathcal{C}(x,\hat{y})$ through the lens of the loss function $\mathcal{L}$ . For square error loss, the Bregman information is the variance of $\mathcal{C}(x,\hat{y})$ (Banerjee et al., 2005).

On Calibration of Large Language Models: From Response To Capability

15

# B. Details of Experiment Setup

# B.1. The choice of $k_{\mathrm{eval}}$

Theoretical insights. Let $f_{\theta}(\cdot \mid x)$ denote the LLM's conditional output distribution given a query $x$ , and let $\mathcal{C}(x,\hat{y}) \in \{0,1\}$ be a deterministic correctness function that indicates whether a sampled response $\hat{y}$ is correct for $x$ . The expected accuracy (our capability-calibration target) is

$$
\mu (x, f _ {\theta}) \triangleq \mathbb {P} _ {\hat {y} \sim f _ {\theta} (\cdot | x)} [ \mathcal {C} (x, \hat {y}) = 1 ].
$$

Since $\mu (x,f_{\theta})$ is not directly observable, we approximate it by drawing $k_{\mathrm{eval}}$ independent samples $\hat{y}_1,\dots ,\hat{y}_{k_{\mathrm{eval}}}\sim f_\theta (\cdot \mid x)$ and computing the empirical mean

$$
\hat {\mu} \triangleq \frac {1}{k _ {\mathrm {e v a l}}} \sum_ {j = 1} ^ {k _ {\mathrm {e v a l}}} \mathcal {C} (x, \hat {y} _ {j}).
$$

For convenience, define the per-sample correctness indicator $Z_{j}\triangleq \mathcal{C}(x,\hat{y}_{j})\in \{0,1\}$ and the number of correct samples

$$
c \triangleq \sum_ {j = 1} ^ {k _ {\mathrm {e v a l}}} Z _ {j}, \quad \text {s o t h a t} \quad \hat {\mu} = \frac {c}{k _ {\mathrm {e v a l}}}.
$$

Binomial model and estimation variance. Under independent sampling randomness across repeated generations and a deterministic evaluator $\mathcal{C}$ , we have

$$
Z _ {j} \mid x \sim \operatorname {B e r n o u l l i} (\mu), \qquad c \sim \operatorname {B i n o m i a l} (k _ {\mathrm {e v a l}}, \mu).
$$

Consequently, $\hat{\mu}$ is an unbiased estimator of $\mu$ with

$$
\mathbb {E} [ \hat {\mu} ] = \mu , \quad \operatorname {V a r} (\hat {\mu}) = \frac {\mu (1 - \mu)}{k _ {\text {e v a l}}}. \tag {16}
$$

Notably, this binomial structure is induced by the binary correctness indicator and does not assume any particular parametric form for the LLM's raw text distribution.

Sample-size guidance (normal/Wald approximation). Equation (16) implies that the standard error of $\hat{\mu}$ decays as $O(k_{\mathrm{eval}}^{-1/2})$ . A common rule-of-thumb for selecting $k_{\mathrm{eval}}$ is to control the margin of error of $\hat{\mu}$ . Using the asymptotic normal (Wald) approximation,

$$
\hat {\mu} \approx \mathcal {N} \left(\mu , \frac {\mu (1 - \mu)}{k _ {\mathrm {e v a l}}}\right),
$$

a two-sided $95\%$ confidence interval has approximate half-width (margin of error, MoE)

$$
\mathrm {M o E} \approx z _ {0. 9 7 5} \sqrt {\frac {\mu (1 - \mu)}{k _ {\mathrm {e v a l}}}}, \qquad z _ {0. 9 7 5} = 1. 9 6.
$$

We interpret $\epsilon$ as a target absolute accuracy for per-query estimation: we aim for the $95\%$ confidence interval of $\hat{\mu}$ to have half-width at most $\epsilon$ , i.e., $|\hat{\mu} - \mu| \leq \epsilon$ with approximately $95\%$ confidence under the normal approximation. Thus, achieving $\mathrm{MoE} \leq \epsilon$ suggests

$$
k _ {\text {e v a l}} \gtrsim \frac {z _ {0 . 9 7 5} ^ {2} \mu (1 - \mu)}{\epsilon^ {2}}. \tag {17}
$$

If $\mu$ is unknown, the conservative worst case uses $\mu (1 - \mu)\leq 1 / 4$ (attained at $\mu = 0.5$ ), yielding

$$
k _ {\text {e v a l}} \gtrsim \frac {z _ {0 . 9 7 5} ^ {2}}{4 \epsilon^ {2}}. \tag {18}
$$

Practical considerations and our choice of $k_{\mathrm{eval}}$ . While Equation (17)-(18) are useful for intuition, Wald intervals can under-cover when $\mu$ is close to 0 or 1 and/or $k_{\mathrm{eval}}$ is small, and may produce bounds outside [0, 1]. We therefore use the

On Calibration of Large Language Models: From Response To Capability

16

![](dt=2026-03-24/ht=06/f5ed1156778479acc3cf20902f0bdf5c96df2f991f849ce6130944a1cd96fd4e.jpg)

![](dt=2026-03-24/ht=06/ecf71fe59e001375a7c8105298a68afd389707c5ba8ddd4665dc1bf28bb4db57.jpg)

Wald analysis only as a guideline for the scaling behavior in Equation (16), and complement it with a sensitivity analysis over $k_{\mathrm{eval}}$ . Prior work (Zhang et al., 2025d) uses $k_{\mathrm{eval}} = 50$ as the ground truth. We find that $k_{\mathrm{eval}} = 100$ provides a stable per-query estimate of $\mu$ while keeping evaluation cost tractable; increasing $k_{\mathrm{eval}}$ beyond this point yields diminishing returns relative to the additional sampling cost.

For completeness, one may alternatively adopt Wilson score intervals (which typically provide closer-to-nominal coverage under the same Bernoulli sampling assumptions) to select $k_{\mathrm{eval}}$ via a short numerical search; our empirical validation supports that $k_{\mathrm{eval}} = 100$ is a reliable operating point in our setting.

Empirical results. We show representative empirical results of gpt-oss-20b (Agarwal et al., 2025) on AIME25 in Figure 6. We found that the standard error of expected accuracies decreases to about 0.0056 when $k_{\mathrm{eval}} = 100$ . Although larger $k_{\mathrm{eval}}$ further decreases variance, it shows diminishing benefits. We finally chose $k_{\mathrm{eval}} = 100$ to balance cost and reliability.

# B.2. Sampling hyperparameters for each LLM.

For each LLM used in the experiments, we use the sampling hyperparameters recommended by its model developers. For Olmo-3-7B-Instruct (Olmo Team et al., 2025), we use temperature=0.6 and top-p=0.95. For Qwen3-8B (Yang et al., 2025), we use chat template of non-reasoning mode to save inference cost due to limited computational budget, and adopt temperature=0.7 and top-p=0.8. For gpt-oss-20b (Agarwal et al., 2025), we use temperature=1.0 and top-p=1.0.

# B.3. LLMs' mean expected accuracies across datasets

Table 4. Three LLMs' mean expected accuracies in seven datasets.

![](dt=2026-03-24/ht=06/8051fbbb4b9735f549fad22ffee41d9d70d970e025228359a688d7b1ffa01f6c.jpg)

<table><tr><td>Model / Dataset</td><td>TriviaQA</td><td>SimpleQA</td><td>GSM8K</td><td>MATH-500</td><td>AIME25</td><td>MMLU</td><td>GPQA</td></tr><tr><td>Olmo-3-7B-Instruct</td><td>53.97%</td><td>3.57%</td><td>93.28%</td><td>89.63%</td><td>42.33%</td><td>70.53%</td><td>43.51%</td></tr><tr><td>Qwen3-8B</td><td>63.01%</td><td>5.17%</td><td>93.27%</td><td>83.62%</td><td>20.87%</td><td>79.37%</td><td>49.70%</td></tr><tr><td>gpt-oss-20b</td><td>63.55%</td><td>5.89%</td><td>95.70%</td><td>93.38%</td><td>73.57%</td><td>89.42%</td><td>66.61%</td></tr></table>

On Calibration of Large Language Models: From Response To Capability

17

# C. Details of Confidence Estimation Methods

# C.1. Detailed descriptions of existing methods

In this section, we discuss the details of existing response calibration confidence estimators and LLM uncertainty quantification methods.

Response calibration confidence estimators. Training-free methods: verbalized confidence (Lin et al., 2022; Tian et al., 2023) that prompts the model to state its certainty, and token probability methods (Kadavath et al., 2022; Manakul et al., 2023). Specifically, $P(\text{True})$ (Kadavath et al., 2022) estimates confidence by measuring the probability assigned to confirmation tokens (e.g., "True"). These methods estimate confidence after the output is generated.

LLM Unce
rtainty Quantification (UQ) methods. We categorize LLM UQ methods into three kinds: token-based approaches, which derive confidence from token-level likelihoods or log probabilities (Kadavath et al., 2022; Duan et al., 2024); Sampling-based approaches, which analyze the consistency or entropy of multiple generated outputs (Wang et al., 2022; Kuhn et al., 2023; Cecere et al., 2025); and methods leveraging the models' internal signals. This includes training models to explicitly output an "I Don't Know" token (Cohen et al., 2024), probing hidden states, or estimating query-level uncertainty via internal self-evaluation (Chen et al., 2025).

# C.2. Implementation details of confidence estimators

Random baseline. We derive the analytical expected loss for the uniform random baseline as follows:

$$
\begin{array}{l} \mathbb {E} _ {s} \left[ (s - \hat {\mu} _ {i}) ^ {2} \right] = \int_ {0} ^ {1} \left(s ^ {2} - 2 s \hat {\mu} _ {i} + \hat {\mu} _ {i} ^ {2}\right) d s, \\ = \left[ \frac {1}{3} s ^ {3} - s ^ {2} \hat {\mu} _ {i} + s \hat {\mu} _ {i} ^ {2}, \right] _ {0} ^ {1} \\ = \frac {1}{3} - \hat {\mu} _ {i} + \hat {\mu} _ {i} ^ {2}. \\ \end{array}
$$

Verbalized confidence and P(True). The prompt for verbalized confidence is provided in Figure 7, and the prompt for P(True) is provided in Figure 8.

# Verbalized confidence prompt

Question: {question}

How likely are you to answer the question correctly? You may refer to the following probabilities P:

Reason about your uncertainty and confidence, and then provide a probability P between 0.0 and 1.0 in the format of \boxed{P}.

Figure 7. The prompt for verbalized confidence.

Training linear probes on LLM's hidden states. We used layer activations after the initial embedding layer and after each transformer block, so there are $\ell + 1$ layers of activations used, with $\ell$ being the number of transformer blocks in the LLM. In our preliminary experiments, we trained a linear probe on activations of each layer, as well as on max-pooled or mean-pooled activations. We found that mean-pooled activations performed the best on the validation sets of each

On Calibration of Large Language Models: From Response To Capability

18

dataset, so we train linear probes on the mean-pooled activations in our main experiments (results listed in §4.3.2). We also searched for other hyperparameters, including number of epochs = {100, 200, 500, 1000}, batch sizes = {32, 64, 128, 256}, weight decays = {0.1, 0.01, 0.001}, loss functions = {BCE, MSE}, input feature standardization = {False, True}, and learning rates = {1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6} on the validation sets. The final chosen hyperparameters are listed in Table 5. We also tried training 2-layer MLP probes, but found that their performance did not differ from linear probes.

Datasets for training linear probes. For TriviaQA (Joshi et al., 2017) and GSM8K (Cobbe et al., 2021), we use their training sets. For MATH (Hendrycks et al., 2021) ( $N = 12$ , 500), we use MATH-500 (Lightman et al., 2023) as the test set, and the remaining 12,000 instances as the training and validation sets.

# P(True) prompt

Question: {question}

Are you able to answer the question correctly?

Answer with only a single word: Yes or No.

Figure 8. The prompt for P(True).

Table 5. Hyperparameters or model information for linear probes trained on three LLMs.

![](dt=2026-03-24/ht=06/63f5d8f88c7232ba55e7b488434c43de71252a72ec40663ae3da9b6031badab0.jpg)

<table><tr><td>Hyperparameter or Model Information</td><td>Olmo-3-7B-Instruct</td><td>Qwen-3B</td><td>gpt-oss-20b</td></tr><tr><td>Number of layer activations used</td><td>33</td><td>37</td><td>25</td></tr><tr><td>Hidden dimension</td><td>4096</td><td>4096</td><td>2880</td></tr><tr><td>Epochs</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Batch size</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>Pooling method</td><td>Mean pooling</td><td>Mean pooling</td><td>Mean pooling</td></tr><tr><td>Loss function</td><td>BCE loss</td><td>BCE loss</td><td>BCE loss</td></tr><tr><td>Feature standardization</td><td>False</td><td>False</td><td>True</td></tr><tr><td>Learning rate (TriviaQA)</td><td>5 × 10-3</td><td>2 × 10-4</td><td>2 × 10-4</td></tr><tr><td>Learning rate (GSM8K)</td><td>5 × 10-3</td><td>1 × 10-4</td><td>5 × 10-4</td></tr><tr><td>Learning rate (MATH)</td><td>5 × 10-3</td><td>1 × 10-4</td><td>2 × 10-4</td></tr></table>

On Calibration of Large Language Models: From Response To Capability

19

# C.3. Analysis of $k_{c}$ for Response Consistency method

In this section, we analyze how the sample size $k_{c}$ impacts the Response Consistency confidence estimator. As shown in Table 6, a larger $k_{c}$ generally leads to improved Brier scores by reducing estimation variance. However, this performance gain trades off with a higher estimation cost. We observe performance saturation because increasing $k_{c}$ cannot correct for fundamental miscalibration, which is the main limitation of this confidence estimator. If the model's most frequent response is incorrect, the estimator converges to a confidence score for a failure case, preventing further reduction in Brier score.

Table 6. Capability calibration Brier scores of Response Consistency across different numbers of samples $k_{c}$ . We use bold to denote the best calibrated method, and underline to denote the second best. Larger $k_{c}$ generally improves performance at the cost of higher estimation overhead.

![](dt=2026-03-24/ht=06/0a743b3a7cb2cca5c70c4b05c7cbc902f7587acf9c498f7ed70d0da2e1f6abcc.jpg)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Domain Cost</td><td colspan="2">Factual knowledge</td><td colspan="3">Mathematical reasoning</td><td colspan="2">General exams</td></tr><tr><td>TriviaQA</td><td>SimpleQA</td><td>GSM8K</td><td>MATH</td><td>AIME25</td><td>MMLU</td><td>GPQA</td></tr><tr><td colspan="9">Olmo-3-7B-Instruct</td></tr><tr><td>Consistency (k=5)</td><td>5L</td><td>0.1228</td><td>0.1605</td><td>0.0290</td><td>0.0373</td><td>0.1860</td><td>0.1306</td><td>0.2462</td></tr><tr><td>Consistency (k=10)</td><td>10L</td><td>0.1121</td><td>0.1297</td><td>0.0272</td><td>0.0383</td><td>0.1730</td><td>0.1191</td><td>0.2246</td></tr><tr><td>Consistency (k=20)</td><td>20L</td><td>0.1046</td><td>0.1141</td><td>0.0264</td><td>0.0350</td><td>0.1465</td><td>0.1121</td><td>0.2079</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Consistency (k=5)</td><td>5L</td><td>0.1433</td><td>0.2016</td><td>0.0255</td><td>0.0336</td><td>0.1630</td><td>0.1039</td><td>0.1830</td></tr><tr><td>Consistency (k=10)</td><td>10L</td><td>0.1301</td><td>0.1657</td><td>0.0241</td><td>0.0304</td><td>0.1433</td><td>0.0976</td><td>0.1656</td></tr><tr><td>Consistency (k=20)</td><td>20L</td><td>0.1239</td><td>0.1521</td><td>0.0237</td><td>0.0276</td><td>0.1040</td><td>0.0959</td><td>0.1485</td></tr><tr><td colspan="9">gpt-oss-20b</td></tr><tr><td>Consistency (k=5)</td><td>5L</td><td>0.1274</td><td>0.0912</td><td>0.0210</td><td>0.0553</td><td>0.0515</td><td>0.0504</td><td>0.1275</td></tr><tr><td>Consistency (k=10)</td><td>10L</td><td>0.1382</td><td>0.0624</td><td>0.0202</td><td>0.0554</td><td>0.0452</td><td>0.0469</td><td>0.1066</td></tr><tr><td>Consistency (k=20)</td><td>20L</td><td>0.1406</td><td>0.0520</td><td>0.0188</td><td>0.0593</td><td>0.0386</td><td>0.0453</td><td>0.1007</td></tr></table>

Table 7. Capability calibration Brier scores of linear probes trained on single and mixed datasets. Results reported by Brier scores (↓). For linear probes trained with different datasets, we use different colors to indicate in-domain in-distribution, in-domain out-of-distribution, and out-domain performance. We use bold to denote the b
est calibrated method, and underline to denote the second best. Results show that training probes on a mixture of datasets (TriviaQA + GSM8K) generally yields the most robust calibration across both in-distribution and in-domain OOD tasks (e.g., MATH, SimpleQA). However, this benefit is less consistent for out-domain datasets (MMLU, GPQA), where specialized single-dataset probes occasionally maintain an edge.

![](dt=2026-03-24/ht=06/427261a48c023e8902fbb22206308a9a7795e468784be1af654ddd918df01c62.jpg)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Domain Cost</td><td colspan="2">Factual knowledge</td><td colspan="3">Mathematical reasoning</td><td colspan="2">General exams</td></tr><tr><td>TriviaQA</td><td>SimpleQA</td><td>GSM8K</td><td>MATH</td><td>AIME25</td><td>MMLU</td><td>GPQA</td></tr><tr><td colspan="9">Olmo-3-7B-Instruct</td></tr><tr><td>Probing (TriviaQA)</td><td>&lt; 1</td><td>0.1113</td><td>0.0386</td><td>0.1180</td><td>0.1273</td><td>0.1496</td><td>0.1300</td><td>0.1242</td></tr><tr><td>Probing (GSM8K)</td><td>&lt; 1</td><td>0.2648</td><td>0.5465</td><td>0.0370</td><td>0.0545</td><td>0.2482</td><td>0.1200</td><td>0.1628</td></tr><tr><td>Probing (TriviaQA + GSM8K)</td><td>&lt; 1</td><td>0.1147</td><td>0.0396</td><td>0.0371</td><td>0.0545</td><td>0.1622</td><td>0.1298</td><td>0.1141</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Probing (TriviaQA)</td><td>&lt; 1</td><td>0.1079</td><td>0.0638</td><td>0.3177</td><td>0.3219</td><td>0.1286</td><td>0.2006</td><td>0.1556</td></tr><tr><td>Probing (GSM8K)</td><td>&lt; 1</td><td>0.1885</td><td>0.4451</td><td>0.0368</td><td>0.0715</td><td>0.0740</td><td>0.1176</td><td>0.1556</td></tr><tr><td>Probing (TriviaQA + GSM8K)</td><td>&lt; 1</td><td>0.1079</td><td>0.0607</td><td>0.0408</td><td>0.1378</td><td>0.0640</td><td>0.1422</td><td>0.1683</td></tr><tr><td colspan="9">gpt-oss-20b</td></tr><tr><td>Probing (TriviaQA)</td><td>&lt; 1</td><td>0.0845</td><td>0.0600</td><td>0.0780</td><td>0.1593</td><td>0.1457</td><td>0.0977</td><td>0.1533</td></tr><tr><td>Probing (GSM8K)</td><td>&lt; 1</td><td>0.1756</td><td>0.7048</td><td>0.0289</td><td>0.0485</td><td>0.1213</td><td>0.0686</td><td>0.2010</td></tr><tr><td>Probing (TriviaQA + GSM8K)</td><td>&lt; 1</td><td>0.0902</td><td>0.0707</td><td>0.0277</td><td>0.0554</td><td>0.1298</td><td>0.0818</td><td>0.1256</td></tr></table>

# C.4. Mixing training dataset from different domains

In this section, we discuss the effect of mixing training datasets from different domains. The experiment results are in Table 7. While single-dataset probes typically perform best on their specific in-distribution tasks, the mixed probe (TriviaQA + GSM8K) often achieves the best or second-best Brier scores across all in-domain out-of-distribution categories, such as MATH and SimpleQA. This suggests that data diversity improves the probe's ability to generalize to in-domain tasks. However, this trend does not hold in out-domain settings like MMLU and GPQA, where the performance of mixed versus

On Calibration of Large Language Models: From Response To Capability

20

single probes varies by model, indicating that mixing training data does not guarantee improved calibration for entirely unrelated domains.

# D. Details of the Applications

# D.1.Pass $@k$ simulation details

# D.1.1.PROCESS OF SIMULATING PASS@k CURVE

Pass@k score at each $k$ . Define the capability-calibrated confidence as $p$ . For each instance $i$ that is sampled $k$ times, the success rate, i.e., pass it or not, follows a Bernoulli distribution. The Bernoulli distribution has mean $p$ and variance $p(1 - p)$ . Therefore, for each instance $i$ that samples $k$ times, the success rate is $P(\text{success} @ k)_i = S_{i,k} = 1 - (1 - p_i)^k$ , the variance is $S_{i,k}(1 - S_{i,k})$ . Generalized to the dataset level. For a dataset $\mathcal{D} = \{x_i\}_{i=1}^N$ , the pass@k score's mean and variance are $\mu_k = \frac{1}{N} \sum_{i=1}^N S_{i,k}$ and $\text{std}_k^2 = \frac{1}{N^2} \sum_{i=1}^N S_{i,k}(1 - S_{i,k})$ respectively. Here, we assume that each instance is independent, so the covariance is zero.

Drawing the simulation curve. The mean and variance of the pass@k score are estimated for each $k$ along the curve. Based on the Central Limit Theorem, the distribution of pass@k is treated as a normal distribution, allowing for the calculation of a 95% confidence interval. This interval represents the region where the true curve is expected to be.

# D.1.2. MORE SIMULATION RESULTS

In this section, we discuss more pass@k simulation results. First of all, the MSE of AIME25 simulation results are in Table 8. The experiment results are consistent with the simulation results on MATH-500.

Figure 9 presents the pass@ $k$ simulation curve at the dataset level. While Oracle Capability-Calibrated Confidence (Oracle-CC) fits the actual pass@ $k$ curve near perfectly, Probe-MATH does not fit well in most scenarios. The experiment results encourage future work on developing confidence estimators that can capture the model's capability on the dataset.

Table 8. Pass@k simulation error (MSE) on the AIME25 dataset. The experiment findings are consistent with the MATH-500 simulation results at Table 3.

![](dt=2026-03-24/ht=06/ec5451612fcfeb974427a5fe32f80d22780853bd84f4c4ebee884223ce777cfc.jpg)

<table><tr><td>Method</td><td>pass@1</td><td>pass@4</td><td>pass@16</td><td>pass@64</td></tr><tr><td colspan="5">Olmo-3-7B-Instruct</td></tr><tr><td>Oracle RC</td><td>0.0968</td><td>0.1596</td><td>0.2727</td><td>0.3721</td></tr><tr><td>Oracle CC</td><td>0.0000</td><td>0.0000</td><td>0.0001</td><td>0.0015</td></tr><tr><td>Probe-MATH</td><td>0.1411</td><td>0.2658</td><td>0.2426</td><td>0.2011</td></tr><tr><td colspan="5">Qwen3-8B</td></tr><tr><td>Oracle RC</td><td>0.0444</td><td>0.0839</td><td>0.1882</td><td>0.2960</td></tr><tr><td>Oracle CC</td><td>0.0000</td><td>0.0000</td><td>0.0001</td><td>0.0014</td></tr><tr><td>Probe-MATH</td><td>0.0832</td><td>0.2977</td><td>0.4885</td><td>0.4711</td></tr><tr><td colspan="5">gpt-oss-20b</td></tr><tr><td>Oracle RC</td><td>0.0809</td><td>0.1267</td><td>0.1599</td><td>0.1873</td></tr><tr><td>Oracle CC</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0018</td></tr><tr><td>Probe-MATH</td><td>0.1661</td><td>0.1136</td><td>0.0798</td><td>0.0420</td></tr></table>

On Calibration of Large Language Models: From Response To Capability

21

![](dt=2026-03-24/ht=06/b3ca142ac70dd8df42f9b87ab9e649d8ffd3412d01b75c9d6976bd99645166e4.jpg)

![](dt=2026-03-24/ht=06/2b3079ecd0a9c0d92b8a1a8b69a62478c897e5289d4fb7294313135749616a02.jpg)

![](dt=2026-03-24/ht=06/d01a1226a07d566b89bad5408005d7c18f5e31b9da71bfbff916323e7c7c2ee8.jpg)

![](dt=2026-03-24/ht=06/04d412a1eee9de371592a9eba40451ca4522555b05dcdd3c886cedc7fc2c68d9.jpg)

![](dt=2026-03-24/ht=06/957930af5c0667f3ba02604aa2b85fc25badd5441cff505ea012298d56ba104e.jpg)

![](dt=2026-03-24/ht=06/7011ff1147f7d0fc83524ab812e8fa71418eb140f19585c47550140edb8b6bb8.jpg)

On Calibration of Large Language Models: From Response To Capability

22

# D.2. Inference budget allocation details

D.2.1. GreEDy ALGORITHM FOR TEST-TIME COMPUTE ALLOCATION (DAMANI ET AL., 2024)

To allocate the inference budget efficiently, we maximize the expected number of solved questions (best-of- $k$ ). Let $N$ be the number of questions, $N \time
s B$ be the total budget, and $p_i$ be the capability-calibrated confidence estimation for question $i$ .

The total expected score $S$ is the sum of the probabilities that each question is solved at least once:

$$
S = \sum_ {i = 1} ^ {N} \left[ 1 - \left(1 - p _ {i}\right) ^ {k _ {i}} \right] \tag {19}
$$

where $k_{i}$ is the number of samples allocated to question $i$ .

To optimize this, we analyze the marginal improvement of adding a single sample to question $i$ , given that it has already been allocated $k_{i}$ samples:

$$
\begin{array}{l} \operatorname {G a i n} _ {i} = S (\text {w i t h} k _ {i} + 1 \text {s a m p l e s}) - S (\text {w i t h} k _ {i} \text {s a m p l e s}) \\ = \left[ 1 - (1 - p _ {i}) ^ {k _ {i} + 1} \right] - \left[ 1 - (1 - p _ {i}) ^ {k _ {i}} \right] \\ = \left(1 - p _ {i}\right) ^ {k _ {i}} - \left(1 - p _ {i}\right) ^ {k _ {i} + 1} \tag {20} \\ = (1 - p _ {i}) ^ {k _ {i}} [ 1 - (1 - p _ {i}) ] \\ = p _ {i} \left(1 - p _ {i}\right) ^ {k _ {i}} \\ \end{array}
$$

Because the gain function is strictly decreasing with respect to $k_{i}$ , a greedy strategy that iteratively assigns the next budget unit to the question with the highest current $\mathrm{Gain}_i$ results in better allocation.

# D.2.2. MORE EXPERIMENT RESULTS

Full experiment results of inference budget allocation are available at Figure 10. We observe that confidence estimators with lower Brier scores have better inference budget allocation performance.

# D.3. Detailed connection with other applications

In this section, we discuss how capability-calibrated confidence relates to other potential applications.

Resource Routing and Estimation. As discussed in Damani et al. (2024), by estimating the ranking of a group of models' capability on answering a question, capability-calibrated confidence can be directly applied to LLM Routing (Maurya et al., 2025; Jiang et al., 2023; Chen et al., 2023b; Ong et al., 2024). Additionally, capability-calibrated confidence enables cost estimation. By estimating the expected accuracy $\mu$ (difficulty of the queries), we can predict the expected sampling budget $\frac{1}{\mu}$ required to generate a correct response (Wu et al., 2024b).

Inference-Time Reliability and Active Refinement. By defining a confidence threshold, one can implement reliable systems that perform Selective Prediction (Mao et al., 2025; Duan et al., 2024; Chen et al., 2023a; Kamath et al., 2020), where systems could abstain or seek human assistance when the model is uncertain (Wu et al., 2024a; Chen et al., 2025) or perform query rewriting when the confidence is low.

Enhanced Learning and Evaluation. Accurately estimating expected accuracy serves as a proxy for instance difficulty. It allows Curriculum Learning (Zhang et al., 2025e;c) that sorts training data by difficulty, or identifies the effects of easy and hard instances. Meanwhile, capability-calibrated confidence enables the evaluation of models on unlabeled test sets, which is called Label-free Benchmarking (Guha et al., 2024; Zhang et al., 2025b). We can derive model rankings that align with ground-truth evaluations.

On Calibration of Large Language Models: From Response To Capability

23

![](dt=2026-03-24/ht=06/b1afa18cebbd475ddec6b2afc5d4cff806c66780462c0c36095dd2de291454ba.jpg)

![](dt=2026-03-24/ht=06/0329f4850f7a996094f22358971a6bf9c0317bf43e8acb46c7f690506fbd7989.jpg)

![](dt=2026-03-24/ht=06/a88b93079b868742f05c08175d8f2724c54ffe5315979e08e68d266935ac9e20.jpg)

![](dt=2026-03-24/ht=06/fc9ffd1956d574c101810a2aff3cb23db1a6f5463edbcbb1453c7c48bd7874e5.jpg)

![](dt=2026-03-24/ht=06/8dbc3be553798cfab1837849e779da92a0821ed7a198ab76ead2badbf808c16b.jpg)

![](dt=2026-03-24/ht=06/c028d9f8264b1e863c674b462b88c4080b876e4aa9fb2090ea9a832fdfc75946.jpg)

On Calibration of Large Language Models: From Response To Capability

24

# E. Targets Difference

![](dt=2026-03-24/ht=06/38f6343233e5aa26a045365d668701d759cd776831bba073a857b07e6c4b9411.jpg)

![](dt=2026-03-24/ht=06/d574d1e32f627b7e5357b4214e2891b60d6b3c67aa9b9739c6509b82cc9a96ba.jpg)

![](dt=2026-03-24/ht=06/885b215491fcc83aee099f0afa34fc5de4aa9bee439dd14904925dbc0a04c860.jpg)

![](dt=2026-03-24/ht=06/2bd3bf89ca055a22627f86f161d8fa8d258456a62019315bca2a4c64de9a1b57.jpg)

![](dt=2026-03-24/ht=06/ab39f2be1187b77194b3f8ac8e81dcce025756ad56db54974edb531f6871e39e.jpg)

![](dt=2026-03-24/ht=06/c99dbe983401cd7a9600abbdf6dc67895c31b8d86e3a1b0408e4ca1986f9d655.jpg)

![](dt=2026-03-24/ht=06/5ff7cded2cab64f5dfe01846c5751a0aa4eb32d33576c2c46743b72217c06eb4.jpg)

![](dt=2026-03-24/ht=06/51e2ab1d78d3dc6ca4e9496690b1df64118e298ebe8606e2b4eee1b189dec35e.jpg)

![](dt=2026-03-24/ht=06/ad21db7cc95a8e75c2ec465ea25978a2420bfc54927818cec4aadc70a724ae27.jpg)

![](dt=2026-03-24/ht=06/50da351292aac33433c3ce5a8e2de4a075be5076c206ef4c0237627ec1d025ee.jpg)

![](dt=2026-03-24/ht=06/da7bd31c768b8e45897589dfddccadbb8424c282728b9f6709ccf95877ff712a.jpg)

![](dt=2026-03-24/ht=06/9e9d82cf79c4ff03ed2b801db3ac989fc5831c4e36b88d5c139745b1159273e9.jpg)

![](dt=2026-03-24/ht=06/5374ca04266a68da17bd84fe15fd68bec63af7f9687baf621a7af7ae13e6e0a1.jpg)

![](dt=2026-03-24/ht=06/798f71226e6a693735ee04b4a5ed61740e660a853d9448883c31e2a54c5c9388.jpg)

![](dt=2026-03-24/ht=06/5eac2a2c0d571b7fe0188708f0c82b4a12bfa5ec6216e842e06e11f5bd84cbfa.jpg)

![](dt=2026-03-24/ht=06/271674f41cc40917313cf90ea7331cb43a1828d90edbfc9459cf7cdf7ae4a665.jpg)

![](dt=2026-03-24/ht=06/1de2dd15f098eb6a58f1a51166bdc09f6e8ab89f14e454fd5fe9283ff91b0a1b.jpg)

![](dt=2026-03-24/ht=06/000d52a209e2f2455ad6bc79981703d28876a048c03ca6b3dd17506a01f8bdb3.jpg)

![](dt=2026-03-24/ht=06/fbf88fa3b0d535dd19fc632bfff389e50091f8472092b3330d6d6d6337e47d53.jpg)

![](dt=2026-03-24/ht=06/d07fd379ade54ae7e3e795d36a7bee751ee36850b61ca542d915f4b5622e759b.jpg)

![](image)
6-03-24/ht=06//6881dbd23f781a91bec70b356a19637f476555ce72e5b41c41dd12ac99d31985.jpg)

On Calibration of Large Language Models: From Response To Capability

25