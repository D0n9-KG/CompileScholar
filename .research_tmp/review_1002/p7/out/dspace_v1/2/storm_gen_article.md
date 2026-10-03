## Related Work

**Nonverbal Representation Modality**
The representation of nonverbal cues in computational models varies significantly, ranging from high-level semantic abstractions to low-level pixel data. Several approaches rely on high-level semantic labels, such as discrete emotional states or specific gesture names, to simplify the generation task [8, 13, 14]. In contrast, other methods operate directly on raw pixel or video frames, preserving visual fidelity but often at the cost of computational efficiency and semantic clarity [9, 10]. A third group utilizes continuous latent vectors derived from VAEs or GANs to capture the manifold of nonverbal motion [12], while others employ parametric 3D face models to ensure anatomical consistency [15]. However, a distinct line of research has emerged that adopts vector-quantized discrete tokens, enabling nonverbal cues to be processed within the same autoregressive framework as text [4, 16, 18, 19, 20]. This paper aligns with the latter approach, utilizing vector-quantized discrete tokens unified with text to facilitate a holistic next-token prediction objective.

**Dataset Grounding & Alignment**
The quality and structure of training data are critical for learning the temporal dynamics of nonverbal communication. Early and specialized datasets often provide unaligned or loosely synchronized multimodal data, which limits the model's ability to learn precise temporal correlations [12]. Other resources focus on static image-text pairs [15] or specific modalities such as audio-derived text transcriptions with timestamps [19] and text-to-motion generation tasks [18]. While large-scale multimodal pre-training corpora [4] and video-centric instruction datasets [2] offer breadth, they often lack the specific granularity of nonverbal annotation. Specialized databases like IEMOCAP [6], CMU-MOSEI [7], and MELD [8] provide rich emotional and conversational context but are not always structured as time-aligned video-grounded dialogues suitable for unified multimodal generation. This paper distinguishes itself by leveraging VENUS, a dataset comprising time-aligned video-grounded dialogues with explicit nonverbal annotations, a grounding strategy shared only by a few recent works [9, 10, 16, 20] that similarly prioritize temporal alignment in real-world video contexts.

**Model Architecture & Objective**
Architectural choices in nonverbal communication systems largely divide into modular pipelines and unified generative frameworks. Many prior works employ modular systems where nonverbal cues are treated as a secondary task, such as LLM-based agents using prompt engineering for nonverbal output [13], LLM-based generation of nonverbal cue labels [14], or 3D face reconstruction with emotion consistency losses [15]. Other approaches use specialized generative models, including contrastive pre-training for emotion representation [12], audio-visual speech generation without intermediate text [10], or speech-to-motion generation frameworks [16]. While some recent works utilize autoregressive transformers with VQ-VAE [20] or hierarchical VQ-VAE and GPT-based generation [18], they often remain task-specific. A smaller set of models adopts a unified multimodal LLM with next-token prediction, treating all modalities as a shared sequence [4, 19]. This paper follows this unified paradigm, integrating nonverbal cues directly into the language modeling objective to achieve co-generation of text and nonverbal behaviors.

**Scope of Nonverbal Cues**
The breadth of nonverbal channels addressed by a model determines the immersiveness of the resulting interaction. A significant portion of prior work focuses exclusively on facial expressions, utilizing datasets and models designed for face-centric analysis [12, 13, 15, 19, 20]. Other studies broaden the scope to include audio, visual, and textual modalities for emotion and sentiment analysis [8]. However, creating a fully immersive conversational experience requires the integration of multiple visual nonverbal channels. A subset of recent research addresses this by targeting a comprehensive set of cues, including both facial expressions and body language or gestures [9, 10, 14, 16, 18]. This paper contributes to this latter group, explicitly targeting the integration of facial expressions and body language to bridge the gap in existing LLMs that fail to incorporate these diverse nonverbal elements.

## References

[1] Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond
[2] VideoChat: Chat-Centric Video Understanding
[3] Video-llama: An instruction-tuned audio-visual language model for video understanding
[4] Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision,
  Language, Audio, and Action
[5] Gpt-4 technical report
[6] IEMOCAP: Interactive emotional dyadic motion capture database
[7] Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph
[8] MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
  Conversations
[9] CHAMPAGNE: Learning Real-world Conversation from Large-Scale Web Videos
[10] Let's Go Real Talk: Spoken Dialogue Model for Face-to-Face Conversation
[11] Nonverbal Communication Cue Recognition: A Pathway to More Accessible Communication
[12] Learning Emotion Representations from Verbal and Nonverbal Communication
[13] FurChat: An Embodied Conversational Agent using LLMs, Combining Open and
  Closed-Domain Dialogue with Facial Expressions
[14] Developing Social Robots with Empathetic Non-Verbal Cues Using Large
  Language Models
[15] EMOCA: Emotion Driven Monocular Face Capture and Animation
[16] Generating Holistic 3D Human Motion from Speech
[17] MotionLLM: Multimodal Motion-Language Learning with Large Language Models
[18] HumanTOMATO: Text-aligned Whole-body Motion Generation
[19] Can Language Models Learn to Listen?
[20] Learning to Listen: Modeling Non-Deterministic Dyadic Facial Motion