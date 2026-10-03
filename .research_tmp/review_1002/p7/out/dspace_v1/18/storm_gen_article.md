## Related Work

**Deployment Target & Resource Constraints**
The primary motivation for industrial anomaly detection in Small and Medium Enterprises (SMEs) is the necessity to deploy models on resource-constrained hardware rather than high-end server infrastructure. While some recent approaches target general resource-constrained environments by focusing on memory and compute restrictions [17], others rely on high-end GPU servers such as A100 or V100 for training and inference [21]. In contrast, a distinct group of works specifically addresses the deployment of models on embedded devices suitable for production lines, including surveys on Tiny Machine Learning (TinyML) workflows for IoT devices [15] and knowledge distillation techniques for compressing large networks into smaller student models for limited-resource devices [18]. Additionally, specific lightweight models have been designed with mobile-friendliness and embedded deployment in mind [8, 14]. KairosAD aligns with this latter group by explicitly targeting resource-constrained embedded devices, such as the NVIDIA Jetson NX and AGX, to bridge the gap between high-performance detection and the practical hardware limitations of SME manufacturing environments.

**Model Architecture & Backbone**
Architectural choices in anomaly detection range from heavy transformer-based models to lightweight convolutional networks. Several studies utilize full-scale transformers, including Vision Transformers (ViT) and Swin Transformers, to capture global context [5, 11, 12, 13, 20, 21, 22], while others employ standard CNNs like ResNet or EfficientNet as backbones [3, 6, 16]. Alternative architectures include conditional Generative Adversarial Networks (GANs) [4], memory-based patch-level methods with noise discriminators [7], hybrid models combining ViT and ResNet [10], and automatically searched architectures [19]. More recently, the Segment Anything Model (SAM) has been adapted for anomaly detection; specifically, prior work has explored SAM-guided two-stream lightweight models [8] and MobileSAM, a lightweight version of SAM achieved through decoupled distillation for mobile applications [14]. KairosAD stands within this emerging category by leveraging the Mobile Segment Anything Model (MobileSAM) as its core backbone, utilizing its promptable segmentation capabilities to achieve the necessary efficiency that distinguishes it from both heavy transformers and traditional lightweight CNNs.

**Efficiency-Performance Trade-off**
The trade-off between computational efficiency and detection accuracy is a critical dimension in model selection. A significant portion of prior work prioritizes maximum accuracy, often at the cost of high latency and model size, as seen in studies using large-scale self-supervised features [11, 13, 21, 22]. Conversely, other methods focus on extreme compression, such as pruning [16] and quantization [17], which can result in significant accuracy drops. Some approaches offer balanced trade-offs without an explicit focus on embedded deployment [18], while others provide training-free methods with reduced overhead [5] or competitive inference times with state-of-the-art performance [6]. Notably, recent lightweight SAM-based models have demonstrated a specific balance of 78% fewer parameters and 4x faster inference while maintaining comparable AUROC [8, 14]. KairosAD positions itself within this efficient group, explicitly quantifying a drastic reduction in computational cost (78% fewer parameters and 4x faster inference) while maintaining comparable AUROC performance against leading state-of-the-art models, a key metric for its target audience of SMEs.

**Validation Context**
Validation strategies in the literature vary from standard academic benchmarks to specific industrial datasets. Many studies evaluate their methods on standard benchmarks such as MVTec-AD and ViSA [5, 6, 7, 8]. Other works utilize real-world datasets specific to certain materials, such as nanofibrous materials and woven fabrics [3], or several benchmark datasets from varying domains [4]. Some research relies on general computer vision benchmarks like ImageNet, OCR, or CIFAR10 [10, 12, 16], while others use mobile application benchmarks [14] or domain-specific tasks like machine translation [21]. A notable gap exists in the literature, as no cited prior paper combines real-world production line deployment with standard industrial benchmarks. KairosAD addresses this gap by validating its model not only on the standard MVTec-AD and ViSA benchmarks but also through successful installation and testing on the real production line of the Industrial Computer Engineering Laboratory (ICE Lab) at the University of Verona, thereby proving practicality in messy, real-world SME conditions.

## References

[1] Deep Industrial Image Anomaly Detection: A Survey
[2] {MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection}
[3] Improving Unsupervised Defect Segmentation by Applying Structural
  Similarity to Autoencoders
[4] GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training
[5] AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2
[6] Towards Total Recall in Industrial Anomaly Detection
[7] SoftPatch: Unsupervised Anomaly Detection with Noisy Data
[8] A SAM-guided Two-stream Lightweight Model for Anomaly Detection
[9] Multimodal Foundation Models: From Specialists to General-Purpose
  Assistants
[10] Learning Transferable Visual Models From Natural Language Supervision
[11] Segment Anything
[12] Emerging Properties in Self-Supervised Vision Transformers
[13] DINOv2: Learning Robust Visual Features without Supervision
[14] Faster Segment Anything: Towards Lightweight SAM for Mobile Applications
[15] A Machine Learning-oriented Survey on Tiny Machine Learning
[16] Methods for Pruning Deep Neural Networks
[17] A Survey of Quantization Methods for Efficient Neural Network Inference
[18] Knowledge Distillation: A Survey
[19] A Comprehensive Survey of Neural Architecture Search: Challenges and
  Solutions
[20] A General Survey on Attention Mechanisms in Deep Learning
[21] Attention Is All You Need
[22] Transformers in Vision: A Survey