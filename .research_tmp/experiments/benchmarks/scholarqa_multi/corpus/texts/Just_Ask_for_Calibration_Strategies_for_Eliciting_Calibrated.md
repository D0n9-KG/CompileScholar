# Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback

Katherine Tian, $^{*†}$ Eric Mitchell, $^{*‡}$ Allan Zhou, $^{\ddagger}$ Archit Sharma, $^{\ddagger}$ Rafael Rafailov $^{\ddagger}$ Huaxiu Yao, $^{\ddagger}$ Chelsea Finn, $^{\ddagger}$ Christopher D. Manning $^{\ddagger}$

$^{\dagger}$ Harvard University $^{\ddagger}$ Stanford University ktian@college.harvard.edu eric.mitchell@cs.stanford.edu

# Abstract

A trustworthy real-world prediction system should produce well-calibrated confidence scores; that is, its confidence in an answer should be indicative of the likelihood that the answer is correct, enabling deferral to an expert in cases of low-confidence predictions. Recent studies have shown that unsupervised pretraining produces large language models (LMs) whose conditional probabilities are remarkably well-calibrated. However, the most widely-used LMs are fine-tuned with reinforcement learning from human feedback (RLHF-LMs), and some studies have suggested that RLHF-LMs produce conditional probabilities that are very poorly calibrated. In light of this perceived weakness, we conduct a broad evaluation of methods for extracting confidence scores from RLHF-LMs. For RLHF-LMs such as ChatGPT, GPT-4, and Claude, we find that verbalized confidences emitted as output tokens are typically better-calibrated than the model's conditional probabilities on the TriviaQA, SciQ, and TruthfulQA benchmarks, often reducing the expected calibration error by a relative 50%.

# 1 Introduction

Real-world prediction systems invariably make errors. However, some mitigation of these errors is possible if the system produces well-calibrated $^{1}$ confidence estimates. In this case, the system's least confident predictions correspond to those that are most likely to be incorrect, potentially allowing these predictions to be skipped or overridden by a human. In the context of language models, one consequence of poor calibration may be hallucination, where a language model confidently asserts incorrect facts or reasoning. While the ability of very large LMs to absorb and synthesize knowledge about the outside world has gained significant

![](images/2ff871743c4f106a01ca9c576bb0b72c09c2e97b5a7615557546425f543f56f8.jpg)

<details>
<summary>bar</summary>

| Accuracy in Bins | Value |
| ---------------- | ----- |
| 0.0              | 0.0   |
| 0.1              | 0.1   |
| 0.2              | 0.2   |
| 0.3              | 0.3   |
| 0.4              | 0.4   |
| 0.5              | 0.5   |
| 0.6              | 0.6   |
| 0.7              | 0.7   |
| 0.8              | 0.8   |
| 0.9              | 0.9   |
| 1.0              | 1.0   |
</details>

![](images/25f4854135a545e16d975d0ff50ec51485a96307491c9f60278dc5bf116ef0f3.jpg)

<details>
<summary>bar</summary>

| Calibration Error | Value |
| ----------------- | ----- |
| 0.166             | 0.35  |
| 0.166             | 0.52  |
| 0.166             | 0.72  |
| 0.166             | 0.74  |
</details>

![](images/62cea7bc51566b7dd1c0f9558809f512b9a8d2fccb4c9a665ec5e7d3bacb09b0.jpg)

<details>
<summary>bar</summary>

| Confidence Bin | Accuracy in Bin |
| -------------- | --------------- |
| 0.6            | 0.35            |
| 0.7            | 0.35            |
| 0.8            | 0.50            |
| 0.9            | 0.45            |
| 1.0            | 0.80            |
</details>

![](images/a3852bced0215f7dd6f0f3ee38c8271cd4c93be5d339287a1b51ceb3a6085b0f.jpg)

<details>
<summary>bar</summary>

| Confidence Bin | Probability |
| -------------- | ----------- |
| 0.4            | 0.4         |
| 0.5            | 0.5         |
| 0.6            | 0.6         |
| 0.7            | 0.7         |
| 0.8            | 0.8         |
| 0.9            | 0.9         |
</details>

Figure 1: Verbalized confidence scores (blue) are better-calibrated than log probabilities (orange) for gpt-3.5-turbo. Raw model probabilities (top-left) are consistently over-confident. Verbalized numerical probabilities (bottom) are better-calibrated. Considering more answer choices (bottom-right) further improves verbalized calibration (as in 'Considering the Opposite' in psychology; Lord et al. (1985)). Verbalized expressions of likelihood (top-right) also provide improved calibration. Bar height is average accuracy of predictions in bin. Darker bars mean more predictions fall in that confidence range. Results computed on SciQ.

attention (Brown et al., 2020; Roberts et al., 2020; Bubeck et al., 2023), relatively little attention has been given to their well-calibratedness (Kadavath et al., 2022). Further, most existing analyses of the calibratedness of LLMs focus on models trained with maximum likelihood, while in practice, the most widely-used LLMs (such as ChatGPT) are fine-tuned using methods such as reinforcement learning from human feedback (Christiano et al., 2017). Some findings suggest that RLHF-LMs may sacrifice well-calibrated predictions for the sake of closer adherence to user instructions in dialogue (Kadavath et al., 2022; OpenAI, 2023), as the reinforcement learning objective encourages the model to allocate probability mass to the most preferred answer(s), rather than matching the relative frequency of possible answers.

This paper evaluates several methods for extracting confidences about model predictions from

![](images/345efc4a8a2fd8b43a4be51575bef9983f2918b60d1d2b7f05d29c67bc9c0b0d.jpg)

<details>
<summary>bar</summary>

| Method | Temperature-scaled ECE | AUC |
| :--- | :--- | :--- |
| TriviaQA | 0.05 | 0.98 |
| SciQ | 0.05 | 0.89 |
| TruthfulQA | 0.21 | 0.39 |
</details>

Figure 2: RLHF generally worsens the calibration of Llama-70B's log probabilities, as measured by ECE (lower is better) or AUC (higher is better). However, this paper (Tables 1-5) will show that for several strong RLHF-LMs, the model's verbalized confidence is often better-calibrated than its log probabilities, reversing some of this degradation. This reversal is strongest for TruthfulQA, an adversarial dataset testing common misconceptions and other difficult queries.

RLHF-LMs. Due to concerns that RLHF may cause systematic overconfidence in the model's probabilities (Figure 2), as well as the general unavailability of per-token log-probabilities in widely used RLHF-LMs, we pay particular attention to prompts that elicit verbalized probabilities, i.e., the model expresses its confidence in token-space, as either numerical probabilities or another linguistic expression of uncertainty. We find that, surprisingly, popular RLHF-LMs are able to directly verbalize confidence scores that are better-calibrated than the model's conditional probabilities (estimated via sampling), without any fine-tuning to learn verbalization. To further improve calibration, we take inspiration from research in human psychology showing that overconfidence can be mitigated by considering alternative answers before responding (Lord et al., 1985; Mussweiler et al., 2000). We show that prompting a model to produce several answer choices before giving its confidence scores significantly improves calibration of verbalized probabilities. Combined with temperature scaling (Guo et al., 2017), this approach generally provides better calibration than model probabilities for ChatGPT $^{2}$ , GPT-4 $^{3}$ , and Claude 2 $^{4}$ across three datasets, often reducing expected calibration error (ECE) by over 50%.

Related Work. Several studies have examined the calibration of large LMs (Lin et al., 2022a; Park and Caragea, 2022; Kadavath et al., 2022; Xiao et al., 2022; Kuhn et al., 2023), finding that combining large pre-trained LMs with temperature scaling (Guo et al., 2017) produces very wellcalibrated predictions (Kadavath et al., 2022; Xiao et al., 2022; Kuhn et al., 2023). Other work focuses on the tendency of language and dialogue models to use linguistic expressions of uncertainty in a well-calibrated manner (Zhou et al., 2023; Mielke et al., 2022). However, existing studies focus on LMs trained purely with unsupervised learning (although Kadavath et al. (2022) briefly examine RLHF-LMs), while widely used models in practice are fine-tuned with instruction-tuning or RLHF (Christiano et al., 2017). RLHF has been shown to effectively leverage annotations of human preferences to control sentiment (Ziegler et al., 2020), improve summarization or instruction-following quality (Stiennon et al., 2022; Ouyang et al., 2022), and inject behavioral priors of harmlessness (Bai et al., 2022b,a). However, recent work has raised the question of whether or not RLHF harms calibration (OpenAI, 2023). Our work is the first to show that verbalized probabilities are often better-calibrated than the model's conditional probabilities for RLHF-LMs such as ChatGPT, GPT-4, and Claude, and Llama-2-70B-Chat.

# 2 Evaluating Calibration in RLHF-LMs

To study the calibration of RLHF-LMs, we conduct experiments with gpt-3.5-turbo (ChatGPT), gpt-4 (GPT-4), claude-1 (Claude 1), claude-2 (Claude 2), and Llama-2-70b-chat (Llama-2-70B-Chat).

Metrics. We measure calibration with multiple metrics. To measure ECE (expected calibration error; Guo et al. (2017)), we bin model predictions by their confidence and measure the average accuracy of predictions in each confidence bin. The ECE is defined as the average (squared) error between the average accuracy and confidence within each bin, where each error is weighted by the fraction of samples falling within the bin. We report raw ECE as well as ECE with temperature scaling (ECE-t). Temperature scaling fits a single temperature value $\beta$ to the model's confidences to minimize negative log likelihood (NLL) on the data, giving scaled probability $\tilde{p}_{i}$ of class i as $\tilde{p}_{i} \propto p_{i}^{\beta}$ . See Figure 1 for a depiction of ECE binning. Although ECE is a standard and interpretable measure of calibration error, it completely fails to capture the confidences' discriminative power. $^{5}$ We therefore also report

<table><tr><td rowspan="2">Method</td><td colspan="4">TriviaQA</td><td colspan="4">SciQ</td><td colspan="4">TruthfulQA</td></tr><tr><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td></tr><tr><td>Label prob.</td><td>0.140</td><td>0.097</td><td>0.142</td><td>0.869</td><td>0.256</td><td>0.180</td><td>0.223</td><td>0.752</td><td>0.451</td><td>0.317</td><td>0.345</td><td>0.418</td></tr><tr><td>‘Is True’ prob.</td><td>0.164</td><td>0.159</td><td>0.165</td><td>0.826</td><td>0.312</td><td>0.309</td><td>0.309</td><td>0.677</td><td>0.470</td><td>0.471</td><td>0.476</td><td>0.384</td></tr><tr><td>Entropy</td><td>—</td><td>—</td><td>—</td><td>0.547</td><td>—</td><td>—</td><td>—</td><td>0.483</td><td>—</td><td>—</td><td>—</td><td>0.236</td></tr><tr><td>Verb. 1S top-1</td><td>0.068</td><td>0.076</td><td>0.138</td><td>0.879</td><td>0.234</td><td>0.084</td><td>0.214</td><td>0.744</td><td>0.389</td><td>0.256</td><td>0.322</td><td>0.545</td></tr><tr><td>Verb. 1S top-2</td><td>0.050</td><td>0.053</td><td>0.139</td><td>0.894</td><td>0.132</td><td>0.050</td><td>0.201</td><td>0.766</td><td>0.361</td><td>0.115</td><td>0.252</td><td>0.485</td></tr><tr><td>Verb. 1S top-4</td><td>0.054</td><td>0.057</td><td>0.144</td><td>0.896</td><td>0.065</td><td>0.051</td><td>0.209</td><td>0.763</td><td>0.203</td><td>0.189</td><td>0.284</td><td>0.455</td></tr><tr><td>Verb. 2S CoT</td><td>0.110</td><td>0.123</td><td>0.168</td><td>0.830</td><td>0.323</td><td>0.246</td><td>0.296</td><td>0.683</td><td>0.419</td><td>0.259</td><td>0.292</td><td>0.551</td></tr><tr><td>Verb. 2S top-1</td><td>0.131</td><td>0.099</td><td>0.148</td><td>0.855</td><td>0.340</td><td>0.203</td><td>0.268</td><td>0.677</td><td>0.431</td><td>0.245</td><td>0.282</td><td>0.483</td></tr><tr><td>Verb. 2S top-2</td><td>0.047</td><td>0.045</td><td>0.147</td><td>0.887</td><td>0.169</td><td>0.040</td><td>0.201</td><td>0.768</td><td>0.395</td><td>0.101</td><td>0.224</td><td>0.517</td></tr><tr><td>Verb. 2S top-4</td><td>0.050</td><td>0.051</td><td>0.156</td><td>0.861</td><td>0.130</td><td>0.046</td><td>0.211</td><td>0.729</td><td>0.270</td><td>0.156</td><td>0.246</td><td>0.463</td></tr><tr><td>Ling. 1S human</td><td>0.062</td><td>0.069</td><td>0.137</td><td>0.884</td><td>0.166</td><td>0.087</td><td>0.223</td><td>0.703</td><td>0.306</td><td>0.296</td><td>0.333</td><td>0.503</td></tr><tr><td>Ling. 1S-opt.</td><td>0.058</td><td>0.066</td><td>0.135</td><td>0.878</td><td>0.064</td><td>0.068</td><td>0.220</td><td>0.674</td><td>0.125</td><td>0.165</td><td>0.270</td><td>0.492</td></tr></table>

Table 1: Measuring calibration of various methods for extracting confidences from gpt-3.5-turbo (ChatGPT). The model's conditional probabilities are relatively poorly calibrated, whether using the model's conditional probability of the label given the query (Label prob.) or the probability assigned to 'True' given the query, proposed answer, and a prompt asking if the answer is correct ('Is True' prob.). Surprisingly, directly verbalizing a probability (Verb. 1S and Verb. 2S) or an expression of confidence such as 'highly likely' (Ling. 1S) yields significantly better-calibrated confidence estimates. 1S refers to one-stage prediction, where the model provides an answer and confidence probability/expression together. 2S refers to two-stage prediction, where the model first gives only an answer, and then in a second stage a confidence. To color the table cells, for each column, we demean and scale by a constant to obtain a shade in [-1,1], where cyan indicates better and orange worse performance. 

<table><tr><td rowspan="2">Method</td><td colspan="4">TriviaQA</td><td colspan="4">SciQ</td><td colspan="4">TruthfulQA</td></tr><tr><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td></tr><tr><td>Label prob.</td><td>0.078</td><td>0.067</td><td>0.077</td><td>0.950</td><td>0.219</td><td>0.165</td><td>0.186</td><td>0.820</td><td>0.445</td><td>0.334</td><td>0.362</td><td>0.462</td></tr><tr><td>Verb. 1S top-1</td><td>0.024</td><td>0.038</td><td>0.084</td><td>0.937</td><td>0.201</td><td>0.084</td><td>0.165</td><td>0.843</td><td>0.350</td><td>0.156</td><td>0.227</td><td>0.622</td></tr><tr><td>Verb. 1S top-2</td><td>0.025</td><td>0.034</td><td>0.084</td><td>0.949</td><td>0.140</td><td>0.048</td><td>0.185</td><td>0.813</td><td>0.315</td><td>0.112</td><td>0.228</td><td>0.623</td></tr><tr><td>Verb. 1S top-4</td><td>0.041</td><td>0.039</td><td>0.081</td><td>0.959</td><td>0.056</td><td>0.059</td><td>0.185</td><td>0.815</td><td>0.198</td><td>0.144</td><td>0.245</td><td>0.619</td></tr><tr><td>Ling. 1S-human</td><td>0.051</td><td>0.041</td><td>0.086</td><td>0.931</td><td>0.148</td><td>0.024</td><td>0.170</td><td>0.835</td><td>0.241</td><td>0.151</td><td>0.228</td><td>0.651</td></tr><tr><td>Ling. 1S-opt.</td><td>0.056</td><td>0.051</td><td>0.088</td><td>0.927</td><td>0.028</td><td>0.052</td><td>0.172</td><td>0.828</td><td>0.082</td><td>0.105</td><td>0.212</td><td>0.632</td></tr></table>

Table 2: gpt-4's verbalized probabilities are substantially better-calibrated than the model probabilities themselves, even after temperature scaling, similarly to gpt-3.5-turbo in Table 1.

Brier Score (BS; Brier (1950)) on temperature-scaled confidences (BS-t), a proper scoring rule (Ovadia et al., 2019) that is the mean squared error between the confidences and the correctness labels. Finally, we assess calibration using a metric from the selective classification literature (Geifman and El-Yaniv, 2017), specifically, the area under the curve of selective accuracy and coverage (AUC).

Datasets. Our experiments use three question-answering datasets assessing factual knowledge. TriviaQA (Joshi et al., 2017) contains 650k question-answer pairs gathered by trivia enthusiasts; SciQ (Welbl et al., 2017) contains approximately 14k crowdsourced science exam question-answer pairs; TruthfulQA (Lin et al., 2022b) contains 817 questions designed to test language models' tendency to mimic human falsehoods. We sample 1000 questions from the validation split of TriviaQA (rc.web.nocontext) and SciQ and all 817 questions from the validation split of TruthfulQA (generation) for our experiments.

Evaluation protocol. For each dataset, we generate a response and corresponding confidence from each method on each of the evaluation questions. Because calibration essentially quantifies the relationship between model confidence and correctness, computing correctness is crucial to accurate measurements of calibration. However, we find doing so to be a challenge, especially in datasets where only a single ground-truth answer (but not aliases or semantically equivalent rephrases) is provided. To avoid excessive false negatives in our correctness computation as a result of exact-match evaluation, we use either GPT-4 or GPT-3.5 to evaluate whether a response is essentially equivalent to the ground truth answer; see Appendix C for the complete equivalence-checking procedure.

Methods. We compare a wide variety of methods for extracting confidence estimates from LLMs. For a comprehensive list of the prompts used for each method, see Appendix Table 6.

First, we consider two methods that leverage the true conditional distribution of the model to gener-

<table><tr><td rowspan="2">Method</td><td colspan="4">TriviaQA</td><td colspan="4">SciQ</td><td colspan="4">TruthfulQA</td></tr><tr><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td></tr><tr><td>Label prob.</td><td>0.074</td><td>0.079</td><td>0.117</td><td>0.915</td><td>0.216</td><td>0.149</td><td>0.195</td><td>0.786</td><td>0.432</td><td>0.304</td><td>0.335</td><td>0.418</td></tr><tr><td>Verb. 1S top-1</td><td>0.049</td><td>0.059</td><td>0.160</td><td>0.839</td><td>0.265</td><td>0.103</td><td>0.247</td><td>0.663</td><td>0.440</td><td>0.134</td><td>0.204</td><td>0.411</td></tr><tr><td>Verb. 1S top-2</td><td>0.046</td><td>0.047</td><td>0.158</td><td>0.875</td><td>0.207</td><td>0.040</td><td>0.225</td><td>0.693</td><td>0.450</td><td>0.085</td><td>0.197</td><td>0.409</td></tr><tr><td>Verb. 1S top-4</td><td>0.075</td><td>0.079</td><td>0.176</td><td>0.814</td><td>0.151</td><td>0.057</td><td>0.226</td><td>0.667</td><td>0.372</td><td>0.105</td><td>0.183</td><td>0.377</td></tr><tr><td>Ling. 1S human</td><td>0.053</td><td>0.050</td><td>0.151</td><td>0.867</td><td>0.253</td><td>0.118</td><td>0.245</td><td>0.664</td><td>0.443</td><td>0.358</td><td>0.340</td><td>0.384</td></tr><tr><td>Ling. 1S-opt.</td><td>0.074</td><td>0.060</td><td>0.149</td><td>0.863</td><td>0.089</td><td>0.082</td><td>0.238</td><td>0.623</td><td>0.139</td><td>0.148</td><td>0.228</td><td>0.350</td></tr></table>

Table 3: Claude-1 produces similar- or better-calibrated log probabilities to gpt-3.5-turbo, but is less able to verbalize well-calibrated confidences, compared to models in the GPT family of RLHF-LMs. Claude-1 has since been deprecated. 

<table><tr><td rowspan="2">Method</td><td colspan="4">TriviaQA</td><td colspan="4">SciQ</td><td colspan="4">TruthfulQA</td></tr><tr><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td></tr><tr><td>Label prob.</td><td>0.089</td><td>0.089</td><td>0.137</td><td>0.882</td><td>0.181</td><td>0.176</td><td>0.237</td><td>0.762</td><td>0.409</td><td>0.368</td><td>0.405</td><td>0.319</td></tr><tr><td>Verb. 1S top-1</td><td>0.072</td><td>0.071</td><td>0.141</td><td>0.903</td><td>0.204</td><td>0.054</td><td>0.201</td><td>0.776</td><td>0.345</td><td>0.115</td><td>0.215</td><td>0.573</td></tr><tr><td>Verb. 1S top-2</td><td>0.049</td><td>0.054</td><td>0.133</td><td>0.918</td><td>0.134</td><td>0.041</td><td>0.211</td><td>0.754</td><td>0.359</td><td>0.085</td><td>0.223</td><td>0.491</td></tr><tr><td>Verb. 1S top-4</td><td>0.072</td><td>0.063</td><td>0.158</td><td>0.890</td><td>0.048</td><td>0.052</td><td>0.216</td><td>0.711</td><td>0.274</td><td>0.075</td><td>0.208</td><td>0.473</td></tr><tr><td>Ling. 1S human</td><td>0.085</td><td>0.061</td><td>0.151</td><td>0.878</td><td>0.238</td><td>0.026</td><td>0.209</td><td>0.756</td><td>0.381</td><td>0.242</td><td>0.305</td><td>0.530</td></tr><tr><td>Ling. 1S-opt.</td><td>0.060</td><td>0.070</td><td>0.151</td><td>0.874</td><td>0.049</td><td>0.056</td><td>0.214</td><td>0.738</td><td>0.099</td><td>0.130</td><td>0.266</td><td>0.446</td></tr></table>

Table 4: Claude-2 has weaker conditional probabilities than Claude-1 and GPT- $*$ , but its verbalized calibration provides consistent improvement over conditional probabilities at a level comparable to GPT-3.5 and surpassing GPT- $*$ on TruthfulQA.

ate confidence scores. The simplest is Label prob., which uses the conditional probability distribution $p(y|x)$ of the model given a question x, which we estimate using n = 10 samples, since many RLHF-LMs are closed-source and do not offer per-token probabilities. $^{67}$ We return the most common answer, using the LLM-based equivalence function to determine when two lexically different answers are semantically equivalent. In a variation of the method described by Kadavath et al. (2022) (again, we use samples since model probabilities are not available), ‘Is True’ prob. samples a single answer $\hat{y}$ from the model given a question x, and the probability it is true is estimated by the probability the model assigns to ‘True’ when asked if the given answer is true (where once again the probabilities are estimated via samples), i.e., $p(\text{True}|x, \hat{y})$ .

Next, we consider methods that extract confidence scores through verbalization (Lin et al., 2022a), i.e., where the model expresses its confidence in token space, either with numerical probabilities or linguistic expressions of likelihood. $^{8}$ First, Verb. 1S top-k prompts the model to produce k guesses and a probability that each is correct all in a single response (i.e., '1 stage'). We take the highest-probability prediction and its as-

sociated probability as the model's output and confidence. Verb. 2S top- $k$ similarly uses numerical probabilities, except the model is first asked to provide only its answers, and afterwards, in a second round of dialogue, asked to assign probabilities of correctness to each answer (i.e., '2 stages'). Verb. 2S CoT uses a chain-of-thought prompt before giving a single answer, and in a second round of dialogue, the model is prompted to assign a probability to that answer (with the chain of thought present in the model's context). Ling. 1S-human uses linguistic likelihood expressions, rather than numerical probabilities, to express uncertainty. The model is prompted to assign confidences to its guesses by choosing from a set of linguistic expressions of uncertainty: {Almost certain, Likely,..., Almost no chance}. Each linguistic likelihood expression is mapped to a probability using responses from a human survey on social media with 123 respondents (Fagen-Ulmschneider, 2023). Ling. 1S-opt. uses a held out set of calibration questions and answers to compute the average accuracy for each likelihood expression, using these 'optimized' values instead. Expressions that are not used for at least $\frac{1}{N}$ of questions, where $N$ is the number of calibration questions, simply use the human probability.

# 3 Results

Tables 1–5 show the results of evaluating various methods for extracting confidence from RLHF-LMs on gpt-3.5-turbo, gpt-4, claude-1,

<table><tr><td rowspan="2">Method</td><td colspan="4">TriviaQA</td><td colspan="4">SciQ</td><td colspan="4">TruthfulQA</td></tr><tr><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td><td>ECE ↓</td><td>ECE-t ↓</td><td>BS-t ↓</td><td>AUC ↑</td></tr><tr><td>Label prob.</td><td>0.151</td><td>0.124</td><td>0.156</td><td>0.865</td><td>0.266</td><td>0.189</td><td>0.243</td><td>0.707</td><td>0.405</td><td>0.361</td><td>0.396</td><td>0.407</td></tr><tr><td>Verb. 1S top-1</td><td>0.071</td><td>0.067</td><td>0.186</td><td>0.793</td><td>0.196</td><td>0.053</td><td>0.239</td><td>0.648</td><td>0.386</td><td>0.172</td><td>0.266</td><td>0.502</td></tr><tr><td>Verb. 1S top-2</td><td>0.060</td><td>0.073</td><td>0.194</td><td>0.815</td><td>0.153</td><td>0.032</td><td>0.230</td><td>0.667</td><td>0.340</td><td>0.037</td><td>0.227</td><td>0.440</td></tr><tr><td>Verb. 1S top-4</td><td>0.069</td><td>0.079</td><td>0.182</td><td>0.816</td><td>0.105</td><td>0.043</td><td>0.229</td><td>0.648</td><td>0.231</td><td>0.102</td><td>0.237</td><td>0.465</td></tr><tr><td>Ling. 1S human</td><td>0.179</td><td>0.115</td><td>0.195</td><td>0.749</td><td>0.071</td><td>0.101</td><td>0.252</td><td>0.603</td><td>0.376</td><td>0.366</td><td>0.383</td><td>0.407</td></tr><tr><td>Ling. 1S-opt.</td><td>0.077</td><td>0.068</td><td>0.186</td><td>0.779</td><td>0.019</td><td>0.042</td><td>0.236</td><td>0.590</td><td>0.047</td><td>0.051</td><td>0.239</td><td>0.435</td></tr></table>

Table 5: With Llama2-70B-Chat, verbalized calibration provides improvement over conditional probabilities across some metrics, but the improvement is much less consistent compared to GPT-\* and Claude-\*.

claude-2, and Llama-2-70b-chat, respectively. We distill several key conclusions from these experiments. 1. Large RLHF-LMs can often directly verbalize better-calibrated confidences (either a numerical confidence probability or an expression such as 'highly likely') than the models' conditional probabilities. 2. Among the methods for verbalizing probabilities directly, we observe that generating and evaluating multiple hypotheses improves calibration (see Figure 1), similarly to humans (Lord et al., 1985), and corroborating a similar finding in LMs (Kadavath et al., 2022). 3. Language models can express their uncertainty with numerical probabilities as well or better than with words, which is surprising in light of longstanding difficulties in representing numbers in language models (Thawani et al., 2021). 4. Chain-of-thought prompting does not improve verbalized calibration (see Appendix Figure 5 for additional CoT results). 5. The calibration of both Claude models' conditional probabilities roughly falls between gpt-3.5-turbo and gpt-4; however, while Claude 1 is much weaker at verbalizing its confidence, Claude 2 is generally a bit stronger than gpt-3.5-turbo at verbalizing. The verbal calibration of the open source model Llama-2-70b-chat is generally weaker than that of closed source models but still demonstrates improvement over its conditional probabilities by some metrics, and does so most clearly on TruthfulQA.

# 4 Discussion

In summary, we study the calibration of widely used RLHF-LMs. We first replicate the finding for GPT-4 (OpenAI, 2023) that RLHF can worsen the calibration of a model's conditional probabilities using the open-source Llama-2-70B base and chat models (Figure 2). To mitigate this regression and ease extraction of calibrated confidence scores for models for which log probabilities are not available, we propose and study new methods that can elicit calibrated confidences from RLHF-LMs by prompting the model to verbalize its confidence in token space. We find verbalized probabilities are better-calibrated than conditional probabilities across several closed models, with mixed results for Llama-2-70B-Chat.

Our results raise several questions for future work. Most notably, the difference between GPT-\*, Claude-\*, and Llama-2's ability to verbalize confidence is significant. What factors are important for learning this skill? Additionally, the 1-stage and 2-stage verbalized numerical confidence prompts sometimes differ drastically in the calibration of their confidences. How can we reduce sensitivity of a model's calibration to the prompt? Going beyond question-answering, can we leverage good calibration in short-answer settings to improve the reliability of long-form generations, perhaps by breaking down long-form generation into a sequence of short questions? Finally, to what extent does a language model's calibration depend on the domain; do our conclusions in the context of factual recall hold in the context of reasoning or arithmetic? Answering these questions provides one path toward building more trustworthy and useful language systems.

Limitations. While our work demonstrates a promising new approach to generating calibrated confidences through verbalization, there are limitations that could be addressed in future work. First, our experiments are focused on factual recall-oriented problems, and the extent to which our observations would hold for reasoning-heavy settings is an interesting open question. Additionally, the lack of technical details available for many state-of-the-art closed RLHF-LMs may limit our ability to understand what factors enable a model to verbalize well-calibrated confidences and differences in this ability across different models. Finally, our study is limited to short-form question-answering; future work should extend this analysis to longer-form generation settings.

Acknowledgements. CF and CDM are CIFAR Fellows. EM gratefully acknowledges funding from a Knight-Hennessy Graduate Fellowship. AZ is supported by the NSF graduate research fellowship program. This research was supported in part by Juniper Networks, Apple, and ONR grant N00014-20-1-2675. The authors thank Yoonho Lee and Noah Goodman for helpful feedback on calibration metrics and experiment design.

# References

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback.   
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022b. Constitutional AI: Harmlessness from ai feedback.   
Glenn W. Brier. 1950. Verification of Forecasts Expressed in Terms of Probability. Monthly Weather Review, 78(1):1–3.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. 2023. Sparks of artificial general intelligence: Early experiments with GPT-4. ArXiv preprint arXiv:2303.12712.   
Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.   
Wade Fagen-Ulmschneider. 2023. Perception of probability words. Ms., UIUC, 05-24-2023.   
Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS'17, page 4885–4894, Red Hook, NY, USA. Curran Associates Inc.   
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR.   
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.   
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. 2022. Language models (mostly) know what they know. Arxiv arxiv:2207.05221.   
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations.   
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022a. Teaching models to express their uncertainty in words. Transactions on Machine Learning Research.   
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022b. TruthfulQA: Measuring how models mimic human

falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.   
Charles Lord, Mark Lepper, and Elizabeth Preston. 1985. Considering the opposite: A corrective strategy for social judgment. Journal of personality and social psychology, 47:1231–43.   
Sabrina J. Mielke, Arthur Szlam, Emily Dinan, and Y-Lan Boureau. 2022. Reducing conversational agents' overconfidence through linguistic calibration. Transactions of the Association for Computational Linguistics, 10:857–872.   
Thomas Mussweiler, Fritz Strack, and Tim Pfeiffer. 2000. Overcoming the inevitable anchoring effect: Considering the opposite compensates for selective accessibility. Personality and Social Psychology Bulletin, 26(9):1142–1150.   
OpenAI. 2023. Gpt-4 technical report.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.   
Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, and Jasper Snoek. 2019. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, Red Hook, NY, USA. Curran Associates Inc.   
Seo Yeon Park and Cornelia Caragea. 2022. On the calibration of pre-trained language models using mixup guided by area under the margin and saliency. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5364–5374, Dublin, Ireland. Association for Computational Linguistics.   
Adam Roberts, Colin Raffel, and Noam Shazeer. 2020. How much knowledge can you pack into the parameters of a language model? In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5418–5426, Online. Association for Computational Linguistics.   
Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2022. Learning to summarize from human feedback.

Avijit Thawani, Jay Pujara, Filip Ilievski, and Pedro Szekely. 2021. Representing numbers in NLP: a survey and a vision. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 644–656, Online. Association for Computational Linguistics.   
Johannes Welbl, Nelson F. Liu, and Matt Gardner. 2017. Crowdsourcing multiple choice science questions. ArXiv, abs/1707.06209.   
Yuxin Xiao, Paul Pu Liang, Umang Bhatt, Willie Neiswanger, Ruslan Salakhutdinov, and Louis-Philippe Morency. 2022. Uncertainty quantification with pre-trained language models: A large-scale empirical analysis. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 7273–7284, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.   
Kaitlyn Zhou, Dan Jurafsky, and Tatsunori Hashimoto. 2023. Navigating the grey area: Expressions of overconfidence and uncertainty in language models.   
Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. Fine-tuning language models from human preferences.

![](images/cccf17dc770089b51b710cbde81f38335a81c48b2ab7b404545f95a8a3ad7e5b.jpg)

<details>
<summary>bar</summary>

| Likelihood Expression | TriviaQA | SciQ | TruthfulQA |
| --- | --- | --- | --- |
| Almost No Chance | 0.005 | 0.005 | 0.06 |
| Highly Unlikely | 0.01 | 0.01 | 0.02 |
| Chances are Slight | 0.02 | 0.02 | 0.03 |
| Little Chance | 0.03 | 0.03 | 0.06 |
| Unlikely | 0.04 | 0.04 | 0.03 |
| Probably Not | 0.01 | 0.01 | 0.01 |
| About Even | 0.05 | 0.05 | 0.20 |
| Better than Even | 0.04 | 0.05 | 0.05 |
| Likely | 0.11 | 0.12 | 0.13 |
| Probably | 0.02 | 0.02 | 0.02 |
| Very Good Chance | 0.29 | 0.30 | 0.15 |
| Highly Likely | 0.40 | 0.41 | 0.18 |
| Almost Certain | 0.05 | 0.08 | 0.12 |
</details>

Figure 3: gpt-3.5-turbo usage rate of each likelihood expression; the model displays much lower verbalized confidence on TruthfulQA than on standard factual recall problems.   
![](images/60e15ac758b190a8d20c75ba3691fb2bf43c634e81a52ec3625ad5b8781fbcad.jpg)

<details>
<summary>bar</summary>

| Likelihood Expression | TriviaQA | SciQ | TruthfulQA |
| --- | --- | --- | --- |
| Almost No Chance | 0.01 | 0.02 | 0.03 |
| Highly Unlikely | 0.01 | 0.01 | 0.04 |
| Chances are Slight | 0.01 | 0.01 | 0.01 |
| Little Chance | 0.01 | 0.01 | 0.01 |
| Unlikely | 0.03 | 0.01 | 0.16 |
| Probably Not | 0.01 | 0.01 | 0.01 |
| About Even | 0.02 | 0.01 | 0.06 |
| Better than Even | 0.01 | 0.01 | 0.01 |
| Likely | 0.13 | 0.13 | 0.24 |
| Probably | 0.01 | 0.01 | 0.01 |
| Very Good Chance | 0.02 | 0.02 | 0.02 |
| Highly Likely | 0.35 | 0.42 | 0.24 |
| Almost Certain | 0.43 | 0.41 | 0.23 |
</details>

Figure 4: gpt-4 usage rate of each likelihood expression; the model displays markedly lower verbalized confidence on TruthfulQA than on standard factual recall problems.

# A Additional Results

Here, we include the likelihood expression usage distribution for gpt-3.5 and gpt-4 in Figures 3 and 4, respectively. gpt-3.5 is systematically less confident for TruthfulQA. The contrast between model confidence for TriviaQA and SciQ compared with TruthfulQA is even more stark for gpt-4.

We also provide additional calibration results for chain-of-thought methods. We compare a one-stage verbalized CoT prompt (Verb. 1S CoT), a two-stage verbalized CoT prompt (Verb. 2S CoT), and a two-stage verbalized method that uses CoT just before eliciting the numerical confidence (Verb. 2S Cot Prob) instead of before the guess, as shown for gpt-3.5 on Trivia QA, SciQ, and Truthful QA in Figure 5. We find that CoT does not noticeably improve calibration across any setting or dataset.

# B Fitting Procedure for Temperature and Probabilities for Linguistic Expressions

To fit the temperature that is used to compute ECE-t and BS-t we split our total data into 5 folds. For each fold, we use it once to fit a temperature and evaluate metrics on the remaining folds. We find that fitting the temperature on 20% of the data yields relatively stable temperatures across folds. We report the average temperature-scaled ECE and BS as ECE-t and BS-t.

![](images/0789fc3a76d4451520085683e3546387f482a63682f285ec831a47fa33054ba5.jpg)

<details>
<summary>bar</summary>

| Dataset     | Model   | Calibration Error |
| ----------- | ------- | ----------------- |
| Trivia QA   | SciQ    | 0.068             |
| Trivia QA   | Truthful QA | 0.389          |
| SciQ        | Truthful QA | 0.087           |
| SciQ        | Truthful QA | 0.270            |
| Truthful QA | Truthful QA | 0.390           |
| Trivia QA   | Truthful QA | 0.131           |
| Trivia QA   | Truthful QA | 0.340           |
| Trivia QA   | Truthful QA | 0.431           |
| Trivia QA   | Truthful QA | 0.164           |
| Trivia QA   | Truthful QA | 0.314           |
| Trivia QA   | Truthful QA | 0.528           |
The image contains multiple vertical lines (one solid, one dashed) representing calibration error values for each bin. The x-axis labels are 'Confidence Bin' and 'Verb 15 COT Accuracy' or 'Verb 25 COT Prob Accuracy'. The y-axis is labeled 'Verb 15 COT Accuracy' and 'Verb 25 COT Prob Accuracy'. Each panel displays a bar chart with error bars, showing calibration error values for each model and confidence bin.
</details>

Figure 5: Expected calibration error is not consistently improved for any CoT prompt variant on gpt-3.5-turbo.

To compute ECE and AUC for Ling. 1S-opt., we similarly split our total data into 5 folds, using 4 folds to fit the probabilities behind each linguistic expression of confidence, then evaluating on the remaining fold. To compute ECE-t and BS-t for Ling. 1S-opt, we hold out one of the 5 folds to fit temperature. We use 3 folds to fit probabilities for linguistic expressions, compute the temperature based on these probabilities on the temperature set, and evaluate metrics on the last fold. We then average metrics across all 20 rotations of folds.

# C Prompt Templates

The prompt template for each sampling method is provided in Table 6. The question is substituted for the variable \${THE\_QUESTION} in each prompt. To evaluate answer correctness, we use gpt-3.5-turbo for SciQ and TruthfulQA and gpt-4 for TriviaQA due to gpt-3.5-turbo's

<table><tr><td>Method</td><td>Template</td></tr><tr><td>Label prob.</td><td>Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: &lt;most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\n\nThe question is: ${THE_QUESTION}</td></tr><tr><td>&#x27;Is True&#x27; prob.</td><td>Question: ${QUESTION}\nProposed Answer: ${ANSWER}\nIs the proposed answer:\n\t(A)True or\n\t(B) False?\n The proposed answer is:</td></tr><tr><td>Verb. 1S top-1</td><td>Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: &lt;most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\n Probability: &lt;the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!&gt;\n\nThe question is: ${THE_QUESTION}</td></tr><tr><td>Verb. 1S top-k</td><td>Provide your ${k} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: &lt;first most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\n\P1: &lt;the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!&gt; ... G${k}: &lt;${k}-th most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\n\P${k}: &lt;the probability between 0.0 and 1.0 that G${k} is correct, without any extra commentary whatsoever; just the probability!&gt;\nThe question is: ${THE_QUESTION}</td></tr><tr><td>Verb. 2S CoT</td><td>Provide your best guess for the following question. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the guess with no other words or explanation.\n\nFor example:\n\nExplanation: &lt;one sentence step-by-step explanation of your thought process&gt;\n\nGuess: &lt;most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\nThe question is: ${THE_QUESTION}Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: &lt;the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!&gt;\n</td></tr><tr><td>Verb. 2S top-1</td><td>Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: &lt;most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\nThe question is: ${THE_QUESTION}Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: &lt;the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!&gt;\n</td></tr><tr><td>Verb 2S top-k</td><td>Provide your ${k} best guesses for the following question. Give ONLY the guesses, no other words or explanation. For example:\n\nG1: &lt;first most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\n\P1: &lt;the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!&gt; ... G${k}: &lt;${k}-th most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\nThe question is: ${THE_QUESTION}Provide the probability that each of your guesses is correct. Give ONLY the probabilities, no other words or explanation.\n\nFor example:\n\nP1: &lt;the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!&gt;\n... P${k}: &lt;the probability between 0.0 and 1.0 that G${k} is correct, without any extra commentary whatsoever; just the probability!&gt;</td></tr><tr><td>Ling. 1S</td><td>Provide your best guess for the following question, and describe how likely it is that your guess is correct as one of the following expressions: ${EXPRESSION_LIST}. Give ONLY the guess and your confidence, no other words or explanation. For example:\n\nGuess: &lt;most likely guess, as short as possible; not a complete sentence, just the guess!&gt;\nConfidence: &lt;description of confidence, without any extra commentary whatsoever; just a short phrase!&gt;\nThe question is: ${THE_QUESTION}</td></tr></table>

Table 6: Prompt templates for each method evaluated. Methods above the double line use multiple samples in order to estimate confidence scores; methods below the double line use the verbalized confidences directly, requiring only a single sample.

high disagreement with a human evaluator on TriviaQA. Using the ground truth answer as \${GOLD\_ANSWER} and the model-generated answer as \${PRED\_ANSWER}, we use the following prompt template:

Are the following two answers to my question Q semantically equivalent?\n\nQ: \${THE\_QUESTION}\nA1: \${GOLD\_ANSWER}\nA2: \${PRED\_ANSWER}\n\nPlease answer with a single word, either "Yes." or "No.", and explain your reasoning.