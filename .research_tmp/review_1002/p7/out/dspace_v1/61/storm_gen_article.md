## Related Work

**Optimization Paradigm**
Prior research in fashion generation and recommendation has predominantly relied on supervised learning paradigms or specific architectural innovations. A significant body of work employs supervised fine-tuning on labeled data to train generative or recommendation models [9, 10, 14, 16, 17], while others utilize contrastive learning or InfoNCE losses to align item representations [7, 12, 13]. Alternative approaches include supervised graph neural network training [1], transformer-based generative architectures [6], hybrid retrieval and generative frameworks [18], and finetuning-free joint distribution learning [19]. More recently, Direct Preference Optimization (DPO) has emerged as a powerful alternative to supervised fine-tuning, applied in various domains to align models with preferences without explicit reward modeling [20, 21, 22, 23, 24]. Additionally, step-aware preference optimization has been proposed to align preferences with denoising performance at each diffusion step [25]. In contrast to these methods, which either depend on labeled supervision or general preference alignment, FashionDPO specifically adapts DPO for fashion outfit generation, leveraging this paradigm to refine pre-trained models using automatically generated feedback.

**Reward/Feedback Mechanism**
The mechanism for guiding model alignment varies significantly across existing literature. Many studies rely on human-annotated preference pairs to provide explicit supervision signals [18, 22, 23, 24], while others operate without explicit feedback, relying on pure generative sampling [13, 17, 19]. Some works utilize implicit signals, such as user-outfit interaction logs [1], user click actions [6], or self-supervised contrastive signals [7]. Other approaches employ condition-based guidance, such as Classifier-Free-Guidance [10], or task-specific hand-crafted reward functions [12]. A few recent works, similar to our approach, utilize automatically generated multi-expert feedback to evaluate quality, compatibility, and personalization [20, 21]. Unlike methods that depend on costly human annotations or rigid, task-specific reward functions, FashionDPO employs a multi-expert feedback generation module to provide comprehensive and objective automated feedback, eliminating the need for manual preference labeling.

**Optimization Objective**
Existing methods often target specific aspects of fashion generation, such as general aesthetic quality, strict compatibility, or reconstruction accuracy. Some works focus solely on general aesthetic quality [20, 23, 24] or strict compatibility and constraint satisfaction [12, 13, 17]. Others prioritize reconstruction accuracy in virtual try-on tasks [14, 16] or bundle recommendation accuracy [7]. Specific control objectives, such as spatial conditioning [9], identity preservation [19], or general preference alignment [22], are also common targets. A subset of prior work aims to balance personalized preference alignment with compatibility [1, 6, 10, 18]. While these studies address individual dimensions, FashionDPO explicitly targets the intersection of personalized preference alignment and compatibility, aiming to enhance the model's ability to adhere to fashion principles while satisfying diverse user preferences, a gap not fully addressed by methods focused on single-objective optimization.

**Data Dependency**
The data requirements for training fashion generation models differ across studies. Many approaches depend on curated high-quality fashion item databases [10, 12, 14, 16] or large-scale human-annotated preference datasets [22, 23, 24]. Other methods utilize user-outfit interaction logs [1], large-scale user click actions [6], user-bundle-item interaction data [7], or small-scale labeled control datasets [9]. Some works leverage large-scale unsupervised image datasets [13], platform fashion item databases with expert evaluation [18], or synthetic data augmentation [19]. A few recent methods, including ours, rely on pre-trained models combined with automatically generated feedback, thereby avoiding the need for new human labels [17, 20, 21]. By utilizing automatically generated feedback, FashionDPO reduces the cost and bias associated with collecting large-scale preference data, offering a more scalable data dependency profile compared to methods requiring extensive human annotation.

## References

[1] Hierarchical Fashion Graph Network for Personalized Outfit
  Recommendation
[2] Personalized Outfit Recommendation With Learnable Anchors
[3] Computational Technologies for Fashion Recommendation: A Survey
[4] Personalized Capsule Wardrobe Creation with Garment and User Modeling
[5] Personalized fashion outfit generation with user coordination preference learning
[6] POG: Personalized Outfit Generation for Fashion Recommendation at
  Alibaba iFashion
[7] MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation
[8] High-Resolution Image Synthesis with Latent Diffusion Models
[9] Adding Conditional Control to Text-to-Image Diffusion Models
[10] Diffusion Models for Generative Outfit Recommendation
[11] From recommendation to generation: A novel fashion clothing advising framework
[12] Compatibility Family Learning for Item Recommendation and Generation
[13] CRAFT: Complementary Recommendations Using Adversarial Feature
  Transformer
[14] VITON: An Image-based Virtual Try-on Network
[15] {GP-VTON:} Towards General Purpose Virtual Try-On via Collaborative
                  Local-Flow Global-Parsing Learning
[16] Taming the Power of Diffusion Models for High-Quality Virtual Try-On
  with Appearance Flow
[17] StableVITON: Learning Semantic Correspondence with Latent Diffusion
  Model for Virtual Try-On
[18] Smart Fitting Room: A One-stop Framework for Matching-aware Virtual
  Try-on
[19] JeDi: Joint-Image Diffusion Models for Finetuning-Free Personalized
  Text-to-Image Generation
[20] PatchDPO: Patch-level DPO for Finetuning-free Personalized Image
  Generation
[21] Boost Your Own Human Image Generation Model via Direct Preference
                  Optimization with {AI} Feedback
[22] Direct Preference Optimization: Your Language Model is Secretly a Reward
  Model
[23] Diffusion Model Alignment Using Direct Preference Optimization
[24] Using Human Feedback to Fine-tune Diffusion Models without Any Reward
  Model
[25] Step-aware Preference Optimization: Aligning Preference with Denoising
                  Performance at Each Step