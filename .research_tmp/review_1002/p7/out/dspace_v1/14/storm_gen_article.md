## Related Work

### Temporal Modeling Backbones
Prior research in video super-resolution (VSR) has predominantly relied on standard Transformer architectures to capture long-range temporal dependencies, such as the Video Super-Resolution Transformer [1] and VRT [7], which utilize spatial-temporal convolutional self-attention and temporal mutual self-attention, respectively. In contrast, a significant body of work employs Recurrent Neural Networks (RNNs/LSTMs) to model temporal dynamics, including the recurrent structure-detail network [4] and FastRealVSR [14]. Another major category utilizes CNN-based propagation and alignment strategies, exemplified by BasicVSR [2] and its enhanced variant BasicVSR++ [3], which focus on essential components for efficient feature propagation. To address specific challenges like large motion or complex deformations, other methods have introduced specialized mechanisms such as temporal group attention [5], dynamic upsampling filters [6], recurrent transformers with deformable attention [8], deformable convolutional networks with temporal-spatial attention [9], and multi-correspondence aggregation networks [11]. More recently, the field has begun exploring State Space Models (SSMs) as alternatives to Transformers; this includes 1D selective state space models [17], multi-dimensional extensions [18], and bidirectional vision backbones [19]. However, no prior work has integrated a Spatio-Temporal Mamba with a 3D Selective Scan module to achieve global spatio-temporal coherence at a lower computational cost than quadratic-complexity Transformers.

### Artifact Suppression Mechanisms
Addressing the artifacts inherent in generative and complex VSR models is a critical challenge. Several studies have focused on post-processing refinement to mitigate these issues, such as RealBasicVSR [13], which employs an image pre-cleaning stage, and FastRealVSR [14], which uses a Hidden State Attention module to aggregate cleaner hidden states. In the broader domain of representation learning, contrastive learning has been widely adopted to extract robust features, with methods ranging from cluster assignment consistency (SwAV) [20] and pairwise contrastive frameworks (SimCLR) [21] to momentum-based approaches (MoCo) [22]. While these contrastive methods have proven effective for unsupervised visual feature learning, they have not been specifically adapted to the VSR context to extract degradation-insensitive features from low-resolution videos. Our work distinguishes itself by introducing a self-supervised ControlNet that leverages contrastive learning specifically to guide the diffusion process and suppress artifacts in generated details, a mechanism not present in the cited prior works.

### Training Data Paradigms
The majority of existing VSR methods are trained on supervised synthetic paired HR-LR datasets, assuming a known degradation model. This paradigm is evident in works utilizing Transformers [1, 7], CNN-based propagation [2, 3], recurrent networks [4], temporal group attention [5], dynamic upsampling [6], deformable attention [8], and multi-correspondence aggregation [11]. To bridge the gap between synthetic training and real-world performance, some approaches have adopted semi-supervised strategies using a mix of synthetic and real data, such as RealBasicVSR [13] and FastRealVSR [14]. Additionally, unsupervised or self-supervised learning techniques have been explored for feature extraction [20, 21, 22], and benchmark datasets for real-world VSR have been introduced [12]. However, no prior work has established a fully self-supervised training paradigm that learns directly from unpaired, real-world LR videos without relying on synthetic degradation models or paired HR-LR data, which is the core data strategy of our proposed framework.

### Training Strategies
Most prior VSR architectures, including those based on Transformers [1, 7], CNNs [2, 3, 6], recurrent networks [4, 14], and various attention mechanisms [5, 8, 9, 11], employ single-stage end-to-end training strategies. RealBasicVSR [13] introduces a stochastic degradation scheme to balance detail synthesis and artifact suppression, representing a deviation from standard single-phase training. However, the specific challenge of stabilizing the training of complex architectures that combine diffusion models with state-space backbones has not been addressed by these methods. Our paper proposes a novel three-stage training strategy based on a mixture of HR-LR videos, which is distinct from the single-stage or stochastic degradation approaches found in the literature and is specifically designed to stabilize the training of the proposed Mamba-Diffusion architecture.

## References

[1] Video Super-Resolution Transformer
[2] BasicVSR: The Search for Essential Components in Video Super-Resolution
  and Beyond
[3] Basicvsr++: Improving video super-resolution with enhanced propagation and alignment
[4] Video Super-Resolution with Recurrent Structure-Detail Network
[5] Video Super-resolution with Temporal Group Attention
[6] Deep video super-resolution network using dynamic upsampling filters without explicit motion compensation
[7] VRT: A Video Restoration Transformer
[8] Recurrent Video Restoration Transformer with Guided Deformable Attention
[9] EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
[10] Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study
[11] MuCAN: Multi-Correspondence Aggregation Network for Video
  Super-Resolution
[12] Real-world video super-resolution: A benchmark dataset and a decomposition based learning scheme
[13] Investigating Tradeoffs in Real-World Video Super-Resolution
[14] Mitigating Artifacts in Real-World Video Super-Resolution Models
[15] Efficiently modeling long sequences with structured state spaces
[16] Simplified State Space Layers for Sequence Modeling
[17] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
[18] Mamba-nd: Selective state space modeling for multi-dimensional data
[19] Vision Mamba: Efficient Visual Representation Learning with
  Bidirectional State Space Model
[20] Unsupervised Learning of Visual Features by Contrasting Cluster
  Assignments
[21] A Simple Framework for Contrastive Learning of Visual Representations
[22] Momentum Contrast for Unsupervised Visual Representation Learning