## Related Work

### Convolutional Architectures for 3D Segmentation
The architectural design of deep learning models for brain tumor segmentation has evolved significantly, ranging from foundational encoder-decoder structures to complex attention-based mechanisms. Early approaches relied on standard 2D U-Net architectures [1] and 3D encoder-decoder networks with autoencoder regularization [5], while competitive solutions often employed two-stage cascaded U-Nets [6] or ensembles of diverse models such as DeepSeg, nnU-Net, and DeepSCAN [9]. The nnU-Net framework, which utilizes standard 3D isotropic convolutions, has become a dominant baseline, achieving top results in recent challenges [7, 10]. To address the limitations of local convolutions, recent works have integrated transformer-based self-attention mechanisms, such as in TransBTS [11] and Swin UNETR [12], or modified nnU-Net with axial attention in the decoder [8]. Other approaches include heterogeneous 3D networks like Med3D [13] and generic autodidactic 3D models [14]. In contrast to these standard isotropic or attention-heavy designs, our work introduces Axial-Coronal-Sagittal (ACS) convolutions, a structural modification that reduces trainable parameters and computational cost while maintaining essential 3D context.

### Pre-training Strategies and Weight Initialization
The initialization of model weights significantly impacts training efficiency and final performance in medical imaging. A common practice in the literature is training from scratch, as seen in standard nnU-Net implementations [7, 8, 10]. Alternatively, self-supervised pre-training on unlabeled medical data has been explored to improve generalization, with Med3D [13] and Models Genesis [14] demonstrating that such methods can outperform scratch learning. However, the direct transfer of 2D pre-trained weights from natural image datasets like ImageNet to the 3D medical domain remains underutilized in recent state-of-the-art segmentation frameworks. While some prior works have investigated multi-task pre-training [13, 14], they do not specifically address the strategy of transferring 2D ImageNet weights to 3D convolutions. Our approach distinguishes itself by explicitly integrating 2D ImageNet pre-trained weights into the 3D nnU-Net framework, employing specific strategies to preserve learned feature representations and drastically reduce the required training epochs.

### Task Formulation and Multi-Task Learning
Most brain tumor segmentation studies focus exclusively on the segmentation task, aiming to delineate tumor sub-regions from multimodal MRI data [1, 2, 5, 6, 7, 8, 9, 10, 11, 12]. Some broader frameworks extend beyond segmentation to include progression assessment and survival prediction [3]. More recently, joint classification and segmentation (multi-task learning) has emerged as a promising direction, with the BraTS 2021 benchmark specifically focusing on radiogenomic classification alongside segmentation [4]. Other works have also explored multi-task pre-training involving both segmentation and classification objectives [13, 14]. Consistent with the trend toward multi-task learning, our paper adopts a joint classification and segmentation formulation. By leveraging pre-trained encoders from a brain glioma grade classification proxy task, we enhance segmentation performance, particularly for challenging tumor labels, aligning with the multi-task paradigm established in recent benchmarks [4].

### Evaluation Protocols and Efficiency
The evaluation of brain tumor segmentation models often relies on ensembling multiple models to boost performance, a standard practice in top challenge solutions. For instance, the first-place solutions for BraTS 2020 [7], 2021 [8], and 2022 [9] all utilized ensembles of cross-validation models or multiple distinct architectures. This approach, while effective, incurs high computational costs during both training and inference. In contrast, our work focuses on efficiency, evaluating our proposed methods in fast training settings with a single model. We demonstrate that our efficient single-model approach achieves comparable or even superior performance to the ensemble of cross-validation models, highlighting a significant shift toward practical, resource-efficient deployment without sacrificing segmentation accuracy.

## References

[1] U-net: Convolutional networks for biomedical image segmentation
[2] The {Multimodal} {Brain} {Tumor} {Image} {Segmentation} {Benchmark} ({BRATS})
[3] Identifying the Best Machine Learning Algorithms for Brain Tumor
  Segmentation, Progression Assessment, and Overall Survival Prediction in the
  BRATS Challenge
[4] The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation
  and Radiogenomic Classification
[5] 3D MRI brain tumor segmentation using autoencoder regularization
[6] Two-stage cascaded u-net: 1st place solution to brats challenge 2019 segmentation task
[7] nnU-Net for Brain Tumor Segmentation
[8] Extending nn-UNet for brain tumor segmentation
[9] Multimodal CNN Networks for Brain Tumor Segmentation in MRI: A BraTS
  2022 Challenge Solution
[10] nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
[11] TransBTS: Multimodal Brain Tumor Segmentation Using Transformer
[12] Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors
  in MRI Images
[13] Med3D: Transfer Learning for 3D Medical Image Analysis
[14] Models Genesis