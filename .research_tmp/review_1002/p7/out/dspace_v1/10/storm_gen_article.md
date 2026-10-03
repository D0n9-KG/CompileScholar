## Related Work

### Curriculum Strategies for Objective Complexity
Curriculum learning has been established as a foundational technique in machine learning, originally introduced to improve learning efficiency by ordering training examples from simple to complex [1]. Subsequent work has provided comprehensive taxonomies of these strategies, categorizing various approaches to managing data or objective complexity over time [2]. In the specific context of language modeling, however, prior research has largely focused on direct training without such staged progression. For instance, recent work on multi-token prediction (MTP) trains models directly on the MTP objective from the start, without a curriculum that gradually introduces this complexity [5]. This paper distinguishes itself by proposing a forward curriculum that explicitly manages the transition from next-token prediction (NTP) to MTP, a specific strategy not previously explored in the cited literature to address the difficulties smaller models face with direct MTP training.

### Model Scale and MTP Feasibility
The benefits of multi-token prediction have been predominantly demonstrated in the context of Large Language Models (LLMs). Several studies have shown that MTP can improve sample efficiency and downstream performance for large-scale architectures [4, 5, 8, 9, 10, 11, 12]. For example, investigations into hidden state predictability in models like GPT-J-6B have highlighted the potential for anticipating subsequent tokens in large models [4], while other works have leveraged MTP to enhance the training and inference of large LLMs [5, 8, 9, 10, 11, 12]. However, these findings do not necessarily translate to smaller models, which have been shown to struggle with the MTP objective. This paper addresses this gap by specifically targeting Small Language Models (SLMs), a scale where prior cited work has not successfully applied standard MTP training, thereby proposing a curriculum tailored to the capacity constraints of smaller architectures.

### Pre-training Objectives and Prediction Heads
The design of pre-training objectives in language models has evolved from standard next-token prediction to more complex variants. Early work such as ProphetNet introduced the prediction of future n-grams as a pre-training task for sequence-to-sequence models [3]. More recently, the focus has shifted to multi-token prediction (MTP) using multiple prediction heads. Medusa, for instance, adds multiple decoding heads to predict subsequent tokens in parallel [12], while other approaches have explored adversarial learning to train draft heads for speculative decoding [11]. The most direct prior work, such as the MTP framework proposed in [5], also utilizes multiple heads to predict the next $k$ tokens. While this paper shares the MTP objective with [5] and [12], it differs in its training methodology; unlike these works which apply the objective directly, this paper introduces a curriculum to modulate the introduction of these multiple heads, specifically to mitigate the performance degradation observed in smaller models when using this objective.

### Inference Mechanisms and Speculative Decoding
Accelerating inference in autoregressive models is a critical area of research, with various mechanisms proposed to reduce latency. Blockwise parallel decoding has been used to accelerate generation by processing multiple tokens simultaneously [6]. A significant body of work focuses on self-speculative decoding, where the model itself acts as a draft model. LayerSkip enables early exit inference and self-speculative decoding [7], while Kangaroo uses a shallow sub-network for lossless self-speculative decoding [8]. Other methods, such as Draft & Verify [9] and SWIFT [10], achieve acceleration by skipping intermediate layers to draft tokens without additional training. KOALA further enhances this by using multi-layer draft heads trained with adversarial learning [11]. Medusa employs a different approach, using tree-based speculative decoding with multiple heads [12]. In contrast, the MTP framework in [5] utilizes parallel multi-head inference rather than a draft-verify loop. This paper evaluates its curriculum strategies specifically through the lens of self-speculative decoding, a mechanism shared with [7, 8, 9, 10, 11], to determine which curriculum variant retains the specific speedup benefits of this inference mode.

## References

[1] Curriculum learning
[2] Curriculum Learning: A Survey
[3] {P}rophet{N}et: Predicting Future N-gram for Sequence-to-{S}equence{P}re-training
[4] Future Lens: Anticipating Subsequent Tokens from a Single Hidden State
[5] Better & Faster Large Language Models via Multi-token Prediction
[6] Blockwise Parallel Decoding for Deep Autoregressive Models
[7] {L}ayer{S}kip: Enabling Early Exit Inference and Self-Speculative Decoding
[8] Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting
[9] Draft & Verify: Lossless Large Language Model Acceleration via
  Self-Speculative Decoding
[10] SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference
  Acceleration
[11] KOALA: Enhancing Speculative Decoding for LLM via Multi-Layer Draft
  Heads with Adversarial Learning
[12] Medusa: Simple LLM Inference Acceleration Framework with Multiple
  Decoding Heads