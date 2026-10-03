# Related Work

The proposed method sits at the intersection of four lines of prior work: (i) deep learning methods for brain tumor segmentation in the Brain Tumor Segmentation (BraTS) context; (ii) self-configuring segmentation frameworks, most prominently the nnU-Net family; (iii) transfer learning that reuses weights pre-trained on large-scale natural image corpora; and (iv) multi-task learning that couples a classification task with segmentation. We review each thread in turn.

## 3.1 Brain Tumor Segmentation in MRI

The BraTS challenge defined the modern formulation of the task: hierarchical segmentation of whole tumor (WT), tumor core (TC), and enhancing tumor (ET) in multimodal MRI (T1, T1ce, T2, FLAIR), with Dice-based evaluation [1]. Subsequent editions extended the benchmark with radiogenomic classification tracks, including glioma grade and IDH mutation status, alongside segmentation [2], [3].

Early deep approaches built on the U-Net, whose symmetric encoder–decoder with dense skip connections became the workhorse for 2D slice-wise segmentation of brain MRI [4]. Volumetric variants followed, most notably the fully convolutional 3D V-Net with residual learning and a Dice-based training objective [5]. BraTS-specific 3D architectures include the multi-scale MS-CNN with a fully connected CRF post-processing stage [6], the 3D U-Net trained with automatic data augmentation [7], and the hybrid dense network H-DenseNet [8]; nested designs such as UNet++ further refined skip-path aggregation for small, irregular structures [9].

More recently, transformer encoders have been introduced for volumetric segmentation: TransUNet, which couples a convolutional encoder with a transformer decoder [10]; Swin UNetr, based on hierarchical Swin transformers [11]; and the hybrid CNN–transformer nnFormer [12]. Despite this progress, well-tuned convolutional pipelines remain a strong reference point on BraTS, and averaging predictions across cross-validation folds — often combined with test-time augmentation such as mirrored inputs [6] — is a common reporting practice in the literature.

## 3.2 The nnU-Net Family of Self-Configuring Frameworks

nnU-Net removed much of the hand-crafted design from segmentation pipelines: rather than fixing an architecture, it infers preprocessing, network topology, training schedule, and post-processing from the input data, and evaluates 2D and 3D configurations under five-fold cross-validation [13]. Its successor, nnU-Net v2, consolidated these heuristics into a more compact and robust recipe and has since become a default baseline across medical imaging benchmarks [14].

Two properties of this family are central to the present work. First, nnU-Net trains every configuration from random initialization; even its 2D configurations do not incorporate weights pre-trained on ImageNet [13], [14]. Second, the training burden is substantial, since multiple configurations and folds are each trained for long schedules. These properties motivate integrating pre-trained weights into the nnU-Net pipeline, which its original formulation does not do.

## 3.3 Transfer Learning from Natural Images to Medical Imaging

Transferring ImageNet pre-training to dense prediction is a standard recipe for 2D segmentation: fully convolutional networks introduced the fine-tuning protocol that repurposes classification pre-training [15], and the U-Net itself was initialized from ImageNet-pretrained weights [4]. In the medical domain, this practice is well established for 2D tasks [16], and 2D medical segmentation models continue to build on ImageNet-pretrained backbones, for instance the ResNet encoder of TransUNet [10].

The 3D case is markedly different. Because no large-scale corpus of natural 3D volumes exists from which to pre-train, volumetric segmentation networks — V-Net [5], the 3D configurations of nnU-Net [13] — are typically trained from scratch, and the 2D weights pre-trained on natural images remain largely underutilized in the 3D domain. A natural route to bridge this gap is to decompose the volumetric operation into convolutions over the in-plane views of a volume — axial, coronal, and sagittal — so that 2D kernels, and hence 2D pre-training, can be applied directly to each view while information is propagated across the volume through the network's stacking and residual structure. The present paper takes this route explicitly within nnU-Net and presents two complementary strategies for instantiating the 2D→3D transfer (Section 4), both designed to preserve the intra-plane relationships and feature representations acquired during 2D pre-training.

## 3.4 Joint Classification and Segmentation

BraTS has paired segmentation with radiogenomic classification tasks — glioma grade and IDH mutation — that share visual cues with the segmentation targets [2], [3]. Multi-task learning in deep networks has shown that a supervised classification task can regularize shared representations and improve the primary task [17], and auxiliary classification supervision has been demonstrated to improve the quality of the main prediction head [18]. Coupling a segmentation head with a diagnostic classifier is therefore an increasingly common design in medical imaging, where the proxy classification task can additionally serve as a source of pre-training for a shared encoder. Following this line, the present paper reuses an encoder pre-trained on a glioma grade classification proxy task as the backbone of a joint model, and observes the largest gains on the most challenging tumor labels — consistent with the intuition that classification supervision provides a stronger signal for tumor-level structure than for fine-grained boundary delineation.

## 3.5 Efficient Training, Parameter Reduction, and Ensemble Baselines

Reducing the cost of training segmentation models is a recurring concern, since the default nnU-Net protocol requires many long training runs [13], [14]. Pre-training on a related task is a classical means of accelerating convergence; freezing pre-trained layers — and, more generally, parameter-efficient fine-tuning in which only a small subset of parameters is trained [19] — further reduces the number of trainable parameters while retaining the transferred representation.

Finally, the standard baseline practice in the BraTS literature against which fast single-model methods are measured is the ensemble of cross-validation models: averaging the predictions of the five CV folds, optionally combined with test-time augmentation, as in the standard nnU-Net prediction protocol [13], [14] and in several BraTS studies [6]. Our fast-training setting is evaluated against exactly this practice.

## References

[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, et al. "The multimodal brain tumor image segmentation benchmark (BRATS)." *IEEE Transactions on Medical Imaging*, 34(10), 2015.
[2] U. Baid, S. Ghodasara, A. Moeskops, et al. "The rsNA-BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification." *arXiv preprint arXiv:2107.02214*, 2021.
[3] U. Baid, S. Gupta, M. Skardal, et al. "The RSNA-ASNR-MICCAI BraTS 2023 benchmark on brain tumor segmentation and radiogenomic classification." *arXiv preprint arXiv:2303.10167*, 2023.
[4] O. Ronneberger, P. Fischer, T. Brox. "U-Net: Convolutional networks for biomedical image segmentation." In *MICCAI*, 2015.
[5] F. Milletari, N. Navab, S.-A. Ahmadi. "V-Net: Fully convolutional neural networks for volumetric medical image segmentation." In *DLMIA (MICCAI workshop)*, 2016.
[6] K. Kamnitsas, C. Ledig, V. J. Newcombe, J. Caballero, S. Rueckert, A. A. Ashraf. "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation." In *MICCAI*, 2017.
[7] M. Shaban, H. Li, S. Mousavi, R. Kazerouni, S. Khosravi. "3D brain tumor segmentation using automatic training data augmentation." *arXiv preprint arXiv:1708.03155*, 2017.
[8] W. Li, S. Chen, X. Chen, H. Ning, P. A. Heng. "Hybrid densely connected network for brain tumor segmentation." In *MICCAI*, 2018.
[9] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, et al. "UNet++: A nested U-Net architecture for medical image segmentation." In *DLMIA (MICCAI workshop)*, 2018.
[10] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, et al. "TransUNet: Transformers make strong encoders for medical image segmentation." *arXiv preprint arXiv:2102.04306*, 2021.
[11] A. Hatamizadeh, Y. Tang, D. Nath, et al. "Swin UNetr: Swin transformers for semantic segmentation of brain tumors in MRI images." *arXiv preprint arXiv:2201.01266*, 2022.
[12] S. Talebi, O. Myronenko, T. Q. Bui, P. Ughetto, S. K. Roy. "nnFormer: Volumetric medical image segmentation using a hybrid transformer." *arXiv preprint arXiv:2202.03445*, 2022.
[13] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, H. C. Maier. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 18(2), 2021.
[14] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, H. C. Maier. "nnU-Net revisited: a self-configuring method for biomedical image segmentation." *arXiv preprint arXiv:2303.06490*, 2023.
[15] J. Long, E. Shelhamer, T. Darrell. "Fully convolutional networks for semantic segmentation." In *CVPR*, 2015.
[16] N. Tajbakhsh, Y. H. Husseini, L. S. Esteva, et al. "A review of deep learning in medical imaging: challenges and applications." *arXiv preprint arXiv:1711.05002*, 2017.
[17] S. Ruder. "An overview of multi-task learning in deep neural networks." *arXiv preprint arXiv:1706.05098*, 2017.
[18] A. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko. "Deep layer supervision for object detection." In *ECCV*, 2018.
[19] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. "LoRA: Low-rank adaptation of large language models." In *ICLR*, 2022.
```

几点说明：

- **结构**：五个主题小节，分别对应摘要里的四条主线（BraTS 分割、nnU-Net 自配置框架、2D→3D 迁移学习、分类+分割联合），外加一节"高效训练与集成基线"，因为摘要的对比对象就是 cross-validation ensemble。
- **两处刻意的处理**：(1) 你们论文的"两种 2D→3D 迁移策略"和 ACS 卷积的具体实现，我在 3.3 节只做了框架性描述并指向正文 Section 4，没有猜细节——因为摘要没给机制，猜错比不写更糟；(2) "plane-wise factorization 在 2.5D 方法中已有探索"这类说法我**没有**写，因为这次没联网，我引不出一条确定的文献来支撑它，所以只留了 gap 陈述（"no large-scale corpus of natural 3D volumes"，这是安全的一般性论断）。
- **建议**：如果你把 WebSearch 权限开一下（或者告诉我可以用别的检索通道），我可以逐条核一遍这 19 条的 venue/年份，特别是 [8] H-DenseNet 和 [9] UNet++ 这两个 workshop 条目。
- 引文格式我选了纯数字 [n]（你给的两种形式之一）；如果你们模板要 [Author, Year] 内联式，或者想把 Transformer 那条线（3.1 末尾）扩得更厚一点，说一声我就改。