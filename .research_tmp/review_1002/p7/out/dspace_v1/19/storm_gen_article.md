## Related Work

**Integration Strategy**
Existing methods for speech-driven facial animation predominantly adopt an end-to-end joint training paradigm, where the audio encoder and the motion generator are optimized simultaneously to minimize reconstruction or synchronization losses. This holistic architectural approach is standard across a wide range of recent works, including those employing Transformer-based autoregressive models [3], diffusion-based synthesis frameworks [5, 9, 16], and various disentanglement or codebook-based systems [2, 4, 6, 7, 8, 10, 11, 12, 19]. While these methods achieve high fidelity by tightly coupling the feature extraction and generation processes, they typically require retraining the entire pipeline when modifications are made to the input representation. In contrast, our work introduces Wav2Sem as a plug-and-play module that performs post-hoc feature decorrelation. Unlike the cited prior works, which rely on integrated training, our approach allows for the insertion of semantic decorrelation into existing pipelines without necessitating a full architectural overhaul or retraining of the base model.

**Feature Space Manipulation**
Prior research has employed diverse strategies to manipulate audio features to improve animation quality, often targeting specific artifacts such as regression-to-mean or lack of expressiveness. Some methods focus on direct mapping of audio features to 3D morphs [9, 10, 12], while others introduce temporal smoothing or filtering to stabilize features [3]. To address the ambiguity in mapping, several works utilize discrete codebook mappings [4] or non-deterministic generation via diffusion models [5]. Other approaches incorporate additional information sources, such as lexical and non-lexical cues [6], or employ disentanglement techniques to separate emotion from content [7, 8] or speaking style from semantic content [11, 19]. Additionally, dedicated controller modules have been used for emotion control [2]. However, none of these prior works explicitly target the coupling of near-homophone syllables in the feature space. Our method distinguishes itself by performing semantic decorrelation specifically to alleviate the averaging effect caused by phonetically similar syllables, thereby enhancing the distinctiveness of lip shapes for near-homophones.

**Source of Semantic Information**
The source of semantic information used to guide facial animation varies significantly across existing literature. A subset of methods relies solely on purely acoustic inputs without explicit semantic guidance [9, 10, 12]. Other approaches incorporate multi-modality guidance, such as text [16], or derive semantic-related content spaces from facial motions themselves [11]. Some frameworks utilize audio content features as implicit semantic cues [19]. Notably, a few recent methods, including FaceFormer [3] and FaceXHuBERT [6], extract global semantic features from the entire audio sequence to capture long-term context. Our work aligns with this latter group by leveraging global semantic features extracted from the full audio sequence. However, while [3] and [6] use these features for general context encoding or capturing lexical/non-lexical information, we specifically utilize this holistic semantic context to disambiguate local acoustic similarities and decorrelate the feature space, addressing a specific limitation in near-homophone generation.

**Audio Encoder Backbone**
The choice of audio encoder backbone has evolved with the advent of self-supervised learning. Early or alternative approaches include end-to-end learned encoders without pre-training [13]. However, the current trend in high-quality speech-driven animation heavily favors pre-trained self-supervised audio models, such as Wav2Vec 2.0 [14] and HuBERT [15]. These powerful backbones are widely adopted in recent state-of-the-art methods, including FaceFormer [3], FaceDiffuser [5], and FaceXHuBERT [6]. Our work positions itself within this established trend by utilizing pre-trained self-supervised audio models as the primary encoder. By leveraging the robust representations provided by these models, we address their specific limitation regarding feature coupling, rather than proposing a new encoder architecture or reverting to traditional hand-crafted features.

## References

[1] A deep learning approach for generalized speech animation
[2] Expressive Speech-driven Facial Animation with controllable emotions
[3] FaceFormer: Speech-Driven 3D Facial Animation with Transformers
[4] CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior
[5] FaceDiffuser: Speech-Driven 3D Facial Animation Synthesis Using
  Diffusion
[6] FaceXHuBERT: Text-less Speech-driven E(X)pressive 3D Facial Animation
  Synthesis Using Self-Supervised Speech Representation Learning
[7] Emotional Speech-Driven Animation with Content-Emotion Disentanglement
[8] EmoTalk: Speech-Driven Emotional Disentanglement for 3D Face Animation
[9] 3DiFACE: Diffusion-based Speech-driven 3D Facial Animation and Editing
[10] Capture, Learning, and Synthesis of 3D Speaking Styles
[11] Mimic: Speaking Style Disentanglement for Speech-Driven 3D Facial
  Animation
[12] A Lip Sync Expert Is All You Need for Speech to Lip Generation In The
  Wild
[13] Deep Speech: Scaling up end-to-end speech recognition
[14] wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
  Representations
[15] HuBERT: Self-Supervised Speech Representation Learning by Masked
  Prediction of Hidden Units
[16] Media2Face: Co-speech Facial Animation Generation With Multi-Modality
  Guidance
[17] Cross Modal Audio Search and Retrieval with Joint Embeddings Based
                  on Text and Audio
[18] Mining Audio, Text and Visual Information for Talking Face Generation
[19] EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via
  Expressive Masked Audio Gesture Modeling