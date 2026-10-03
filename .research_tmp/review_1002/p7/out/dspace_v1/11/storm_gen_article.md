## Related Work

### Dependency Modeling Structure
Prior research on 3D human pose estimation has employed diverse architectural strategies to model dependencies among body parts. Early and foundational approaches relied on convolutional spatial filtering [2], simple feed-forward networks [4], and various forms of graph convolutional networks, including standard [3], semantic [5], and modulated variants [6]. To capture more complex relationships, subsequent works introduced graph-oriented Transformers [7], temporal correlation modeling [8], and multi-hypothesis spatio-temporal dependency modeling [9]. Other methods have focused on high-order directed attention mechanisms involving joints, bones, and hyperbones [10], deep sequential stacking of local graph convolutions [11], or hierarchical Bi-directional RNNs [17]. More recent advancements include reverse diffusion processes [15], conditional diffusion models with embedding transformers [16], strided Transformer encoders with hierarchical aggregation [18], alternating temporal and spatial transformer blocks [19], ConvNets predicting heatmap triplets [21], hierarchical poselet-guided graph convolutional networks [22], and Graph Stacked Hourglass Networks [23]. Additionally, flat global self-attention over all joints has been explored in [14, 20]. In contrast to these methods, which often rely on flat attention, deep stacking, or specific graph topologies, our work introduces a pyramid-structured multi-scale hierarchy that explicitly captures both joint-to-joint and joint-to-group correlations, addressing the limitations of uncorrelated noise and excessive model size inherent in deeper networks.

### Scale Aggregation Mechanism
The mechanism for aggregating information across different scales or hypotheses varies significantly across existing literature. Some approaches utilize coarse-to-fine iterative refinement [2] or cross-hypothesis communication and aggregation [9]. Others employ multi-order attention modules [10] or U-shaped network architectures [11] to facilitate feature flow. Diffusion-based methods rely on context-conditioned reverse diffusion [15] or embedding transformer conditioning [16]. In the realm of Transformers, progressive sequence shrinking via strided convolutions [18] and multi-scale encoder-decoder structures with multi-level feature learning [23] have been proposed to handle varying granularities of pose data. While these methods effectively manage scale or hypothesis diversity, none employ the specific parallel cross-scale correlation via concatenated compact sequences found in our Pyramid Graph Attention (PGA) module. Our approach distinguishes itself by computing correlations between scales in parallel within a compact sequence, offering a more efficient and direct method for capturing pyramid-structured long-range dependencies.

### Model Complexity Strategy
A significant design consideration in 3D pose estimation is the trade-off between model complexity and performance. Many prior works adopt standard or heavy architectures, such as standard Transformer architectures [9, 20], standard ConvNets with volumetric output [2], or simple deep feed-forward networks [4]. Graph-based methods often utilize Graph Convolutional Networks [11], generic non-local operation blocks [14], or graph convolutional network architectures [23]. Diffusion-based approaches typically involve a diffusion model framework [15] or straightforward diffusion models [16]. Other strategies include base networks with hierarchical RNNs [17], Transformers with reduced computation cost via strided convolutions [18], Transformer-based encoder-decoders [19], and simple ConvNet designs [21]. Notably, HDFormer [10] also pursues a lightweight architecture with a smaller model size, sharing this design value with our work. However, while [10] achieves lightness through high-order directed attention, our PGFormer achieves a lightweight multi-scale transformer architecture by encapsulating human sub-structures into self-attention via pooling, resulting in lower error and smaller model size compared to state-of-the-art methods.

### Contextual Representation
The representation of contextual information in human pose estimation has evolved from simple spatial-temporal graph features [3] and 2D joint locations [4] to more sophisticated semantic [5] and modulated graph features [6]. Graph-oriented Transformer features [7] and temporal correlations [8] have also been utilized to enrich the context. Some methods represent context through multiple plausible pose hypotheses [9], high-order bone and joint relationships [10], or directed graphs with hierarchical orders [11]. Global feature aggregation [14], pose uncertainty distributions [15], and sets of 2D joint candidate samples [16] offer alternative contextual views. Manual kinematic chain constraints [17], temporal sequence representations [18], and joint-specific temporal and inter-joint spatial features [19] further refine the context. Part-centric heatmap triplets [21], poselet-guided features [22], and multi-scale skeletal representations [23] also contribute to contextual modeling. Unlike these approaches, which often treat joints individually or rely on manual constraints, our method encapsulates human sub-structures via pooling into self-attention. This allows the model to modulate joints not just by other individual joints but by body parts, better respecting the spatial constraints of human structure.

## References

[1] Determination of {3D} human body postures from a single view
[2] Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose
[3] Exploiting spatial-temporal relationships for {3D} pose estimation via graph convolutional networks
[4] A simple yet effective baseline for 3d human pose estimation
[5] {Semantic Graph Convolutional Networks for 3D Human Pose Regression}
[6] Modulated graph convolutional network for {3D} human pose estimation
[7] {GraFormer: Graph-oriented Transformer for {3D} Pose Estimation}
[8] {Exploiting Temporal Correlations for {3D} Human Pose Estimation}
[9] MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation
[10] HDFormer: High-order Directed Transformer for 3D Human Pose Estimation
[11] Conditional Directed Graph Convolution for 3D Human Pose Estimation
[12] Optimizing network structure for {3D} human pose estimation
[13] A comprehensive study of weight sharing in graph networks for {3D} human pose estimation
[14] Non-local Neural Networks
[15] DiffPose: Toward More Reliable 3D Pose Estimation
[16] DiffPose: Multi-hypothesis Human Pose Estimation using Diffusion models
[17] Learning Pose Grammar to Encode Human Body Configuration for 3D Pose
  Estimation
[18] Exploiting Temporal Contexts with Strided Transformer for 3D Human Pose
  Estimation
[19] MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose
  Estimation in Video
[20] Attention Is All You Need
[21] HEMlets Pose: Learning Part-Centric Heatmap Triplets for Accurate 3D
  Human Pose Estimation
[22] {HPGCN: Hierarchical poselet-guided graph convolutional network for {3D} pose estimation}
[23] Graph Stacked Hourglass Networks for 3D Human Pose Estimation