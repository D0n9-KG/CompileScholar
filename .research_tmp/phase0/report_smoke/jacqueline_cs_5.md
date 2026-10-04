# One desirable property for LMs is that they should be well-calibrated, in that t

## Verbalized Confidence and Prompting Strategies

Recent research indicates that large language models fine-tuned with reinforcement learning from human feedback often produce more accurate confidence estimates when asked to express them in natural language rather than relying on raw token probabilities [1] [2]. This verbalized approach has been shown to significantly reduce the expected calibration error, often by more than half, compared to standard model outputs [1] [2]. To further enhance these estimates, researchers have drawn on psychological insights suggesting that considering alternative hypotheses can mitigate overconfidence. By prompting the model to generate multiple potential answers before assigning a confidence score, the calibration of the verbalized output improves substantially. This method is particularly advantageous because it functions as a black-box technique, making it applicable to large models accessed via APIs where internal parameters are not available [1] [2].

## Post-Hoc Calibration and Scaling Techniques

In addition to prompting strategies, post-hoc adjustments are frequently employed to refine confidence scores. One common method involves temperature scaling, which fits a single parameter to the model's confidences to minimize negative log-likelihood on a validation set. When combined with verbalized confidence elicitation, temperature scaling generally yields better calibration results for major models such as ChatGPT, GPT-4, and Claude across various datasets [1] [2]. The underlying mechanism for miscalibration in these models is often attributed to the reinforcement learning objective, which tends to allocate probability mass to preferred answers rather than accurately reflecting the true frequency of correctness [1] [2]. Consequently, simple adjustments to the output distribution can correct for this bias more effectively than the raw model probabilities [1] [2].

## Distinguishing Capability from Response Calibration

A critical distinction in recent literature is the separation between capability calibration and response calibration [2]. Capability calibration refers to the model's expected accuracy across a distribution of inputs, whereas response calibration concerns the correctness of a single, specific decoded response [2]. Because modern decoding processes are stochastic, the correctness of a single response may not accurately reflect the underlying capability of the model, leading to a mismatch between these two types of calibration [2]. Since the response is already generated, estimating its calibration is decoupled from the generating model itself [2]. This distinction is vital for applications where the model's overall competence is more relevant than the specific outcome of a single query [2].

## Domain-Specific Recalibration

Standard calibration methods may fail to account for domain-specific biases, where overconfidence in one area can cancel out underconfidence in another when viewed in aggregate [3]. To address this, few-shot recalibration methods have been developed that train a separate model to map confidence scores to slice-specific precision estimates [3]. This approach uses a small number of unlabeled examples and synthetic data generation to learn these mappings without requiring extensive labeled data [3]. The primary benefit of this technique is the ability to establish domain-specific confidence thresholds, allowing the system to determine when predictions are trustworthy and when the model should abstain from answering [3]. This granular approach ensures that confidence scores remain meaningful across diverse application contexts [3].

## Foundational Insights and Future Directions

Calibration is increasingly viewed as a broader paradigm than simple out-of-distribution detection, with robust pre-training generally leading to more calibrated posterior distributions [4]. For non-pre-trained models, there is an observed inverse relationship between model complexity and calibration quality [4]. Recent work has highlighted promising ways to extract well-calibrated estimates that reflect the likelihood of correctness. Looking forward, capability-calibrated confidence estimations are well-suited for tasks such as selective prediction, hallucination detection, and model routing [2]. Future research aims to extend these methods to human-AI collaboration and trustworthy AI systems, with a focus on developing estimators that accurately capture model capability on specific datasets [2].

## References

- [1] Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback (2023), doi:10.18653/v1/2023.emnlp-main.330
- [2] On the Calibration of Large Language Models and Alignment (2023)
- [3] Few-Shot Recalibration of Language Models (2024), doi:10.48550/arxiv.2403.18286
- [4] Calibration of Pre-trained Transformers (2020), doi:10.18653/v1/2020.emnlp-main.21