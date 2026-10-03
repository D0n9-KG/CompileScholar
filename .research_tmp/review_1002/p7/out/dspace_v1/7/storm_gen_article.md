## Related Work

**Direction Identification Method**
Prior research on identifying steering directions in Large Language Models (LLMs) employs a diverse array of techniques, ranging from supervised learning to geometric analysis. A significant body of work relies on supervised learning to derive steering vectors from labeled data pairs, such as Contrastive Activation Addition (CAA) [17], Activation Engineering (ActAdd) [18], and Iterative Null-space Projection (INLP) [21]. In contrast, other approaches utilize more specialized or unsupervised methods, including manual or mechanistic identification of single refusal directions [11], affine decomposition of activation vectors [12], and population-level representation analysis via Representation Engineering (RepE) [13]. Further variations include projection into directions with maximal or minimal covariance with demonstrations [14], extraction of category-specific steering vectors [15], and unsupervised discovery of directions satisfying logical consistency properties [16]. Additional methods employ rule-based conditional application of pre-computed vectors [19], intrinsic geometric criteria for linear subspace identification [20], closed-form linear concept erasure such as LEACE [22], injection of pre-defined trojan vectors [23], geometric analysis of word embedding directions [24], construction via counterfactual pairs and causal inner products [26], probing for specific linear features [28], black-box probing for linguistic features [29], sparse autoencoders for activation reconstruction [31], and eigendecomposition of bilinear MLP weights [32]. COSMIC distinguishes itself by introducing an automated selection framework based on Cosine Similarity Metrics in activation space, a value not occupied by any of the cited prior works, thereby eliminating the need for labor-intensive manual analysis or assumption-heavy supervised training.

**Dependency on Model Outputs**
The dependency on model outputs varies significantly across existing alignment and steering methods. A subset of prior work relies heavily on detectable refusal tokens in the output sequence [3, 4, 5] or depends on logit-level analysis [7]. Other approaches utilize output (word) representation space and input (sentence) space formalizations [26], contextualized word representations [29], or weight-based analysis that is independent of both activations and outputs [32]. However, a large group of studies operates entirely within the activation space, independent of model outputs, including works on single-direction refusal mediation [11], affine functions [12], RepE [13], spectral editing [14], category-wise safety steering [15], latent knowledge discovery [16], CAA [17], ActAdd [18], conditional activation steering [19], geometric causal probing [20], INLP [21], LEACE [22], trojan activation attacks [23], word embedding debiasing [24], and sparse autoencoders [31]. COSMIC aligns with this latter group by being entirely independent of model outputs, focusing solely on activation-space metrics to ensure robustness even when models fail to produce standard refusal phrases.

**Assumption of Refusal Behavior**
Existing methods differ in their assumptions regarding the specific behaviors of refusal. Several studies assume the presence of specific refusal tokens to identify or manipulate safety behaviors [3, 4, 5, 7]. Another category assumes standard alignment templates to guide the steering process [19]. In contrast, a smaller set of works makes no assumptions about specific refusal tokens or behaviors, relying instead on general activation properties, such as the identification of a single refusal direction [11] and the affine nature of refusal [12]. COSMIC shares this value with [11, 12] by requiring no assumptions about specific refusal tokens or behaviors, allowing it to generalize across models where standard refusal signals may be absent or obfuscated.

**Target Model Alignment Condition**
The robustness of alignment methods across different model conditions is a critical differentiator. Many prior works require standard safety training to function effectively [2, 3, 5, 7, 23], while others are only effective on strongly aligned models [4]. Some methods are applicable to standard aligned chat models [11], various models including large-scale variants like Llama 3 70B [12], six open-source LLMs of different sizes [14], or models that have undergone extensive alignment and safety training [15]. A gap exists in the literature regarding methods that are robust across a wide range of alignment conditions, including weakly aligned and adversarial settings, a value not taken by any cited prior paper. COSMIC addresses this gap by demonstrating reliable performance and minimal increase in false refusals across diverse alignment conditions, including those where standard safety training has been compromised or bypassed.

## References

[1] Training language models to follow instructions with human feedback
[2] Training a Helpful and Harmless Assistant with Reinforcement Learning
  from Human Feedback
[3] Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors,
  and Lessons Learned
[4] LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B
[5] Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models
[6] Fine-tuning Aligned Language Models Compromises Safety, Even When
Users Do Not Intend To!
[7] Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks
[8] Universal and Transferable Adversarial Attacks on Aligned Language Models
[9] Jailbreaking Black Box Large Language Models in Twenty Queries
[10] Ethical and social risks of harm from Language Models
[11] Refusal in Language Models Is Mediated by a Single Direction
[12] Refusal in LLMs is an Affine Function
[13] Representation Engineering: A Top-Down Approach to AI Transparency
[14] Spectral Editing of Activations for Large Language Model Alignment
[15] Towards Inference-time Category-wise Safety Steering for Large Language
  Models
[16] Discovering Latent Knowledge in Language Models Without Supervision
[17] Steering Llama 2 via Contrastive Activation Addition
[18] Steering Language Models With Activation Engineering
[19] Programming Refusal with Conditional Activation Steering
[20] A Geometric Notion of Causal Probing
[21] Null It Out: Guarding Protected Attributes by Iterative Nullspace
  Projection
[22] LEACE: Perfect linear concept erasure in closed form
[23] Trojan Activation Attack: Red-Teaming Large Language Models using
  Activation Steering for Safety-Alignment
[24] Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word
  Embeddings
[25] Toy Models of Superposition
[26] The Linear Representation Hypothesis and the Geometry of Large Language
  Models
[27] Linguistic Regularities in Continuous Space Word Representations
[28] Emergent Linear Representations in World Models of Self-Supervised
  Sequence Models
[29] The Low-Dimensional Linear Geometry of Contextualized Word
  Representations
[30] Towards Monosemanticity: Decomposing Language Models With Dictionary Learning
[31] Sparse Autoencoders Find Highly Interpretable Features in Language
  Models
[32] Bilinear MLPs enable weight-based mechanistic interpretability
[33] A Mathematical Framework for Transformer Circuits