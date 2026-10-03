## Related Work

**Compression Ratio Adaptability**
A primary challenge in single-pixel imaging (SPI) is the ability to handle varying compression ratios (CRs) without retraining. Traditional Plug-and-Play (PnP) methods and certain non-iterative approaches offer this flexibility by relying on general-purpose components. For instance, ReconNet [11] employs a non-iterative CNN followed by an off-the-shelf denoiser, allowing it to adapt to different measurement settings, while COAST [19] utilizes a controllable arbitrary-sampling network designed to process arbitrary sampling matrices with a single model. In contrast, most deep unrolling architectures are typically trained for specific CRs, requiring fine-tuning or retraining when the sampling rate changes. This work stands at the intersection of these paradigms, achieving the flexible handling of varying CRs with a single trained model, thereby combining the adaptability of PnP-style methods [11, 19] with the high reconstruction accuracy characteristic of unrolled networks.

**Regularization Mechanism**
The choice of regularization mechanism fundamentally dictates the reconstruction quality and generalization capability of SPI solvers. Early and classical approaches relied on fixed, hand-crafted regularizers such as total variation or wavelet bases [27], or specific mathematical operations like hard thresholding [2], shrinkage-thresholding [3], $\ell_1$-norm minimization [4], and nonlocal low-rank regularization [5]. A significant shift occurred with the introduction of Plug-and-Play (PnP) methods, which implicitly perform regularization using off-the-shelf deep denoisers within iterative algorithms like Approximate Message Passing (AMP) [6, 17] or general PnP frameworks [7, 8, 10, 25, 26]. More recently, learned regularization modules have emerged, including memory-augmented proximal mappings [18], controllable proximal modules with deblocking [19], and informative proximal mappings with inter-stage pathways [20]. Other approaches include end-to-end learned regularizers without explicit proximal interpretation [21], cross-attention based iterative processes [22], and explicit image-adaptive Laplacian-based regularization defined by a denoiser (RED) [28]. While Learned Proximal Networks (LPN) [9] provide exact proximal operators for data-driven nonconvex regularizers, this paper distinguishes itself by designing a Deep Image Restorer (DIR) that approximates an explicit proximal operator via a specialized Proximal Trajectory (PT) loss, ensuring both convergence guarantees and high-fidelity reconstruction.

**Optimization Algorithm Unrolled**
The underlying optimization algorithm determines the structural backbone of unrolled networks. Various iterative algorithms have been adapted into deep architectures, including Gradient Projection [1], Iterative Hard Thresholding (IHT) [2], Iterative Shrinkage-Thresholding (ISTA) [3], and Fixed-Point Iteration (FPI) [24]. Alternating Direction Method of Multipliers (ADMM) has been widely adopted for its robustness in handling constraints, as seen in [4]. Proximal Gradient Descent (PGD) serves as the basis for networks like DGUNet [20] and HATNet [26], while Approximate Message Passing (AMP) is unrolled in D-AMP [6] and AMP-Net [17]. Other frameworks include optimization-inspired iterative processes with cross-attention [22]. This paper unrolls both Half Quadratic Splitting (HQS) and ADMM, a choice shared with [7]. This specific unrolling structure facilitates the integration of the proposed DIR and PT loss, enabling better convergence properties and flexibility compared to unrolling simpler gradient-based or thresholding-based algorithms.

**Training Paradigm**
The training paradigm defines how the network learns to solve the inverse problem. One category consists of iterative inference methods with fixed components, such as PnP algorithms [6, 7, 8, 10, 28], where the denoiser is pre-trained and the forward model is fixed. Another category involves supervised regression on ground truth images without explicit algorithmic unrolling, as demonstrated by ReconNet [11] and CSformer [21]. The dominant paradigm in recent high-accuracy SPI solvers is the end-to-end trainable unrolled network, where the entire iterative process is differentiable and optimized jointly. This approach is utilized in [9, 18, 19, 20, 22, 26]. While these methods typically use standard reconstruction losses, this paper introduces a specialized Proximal Trajectory (PT) loss function within the end-to-end trainable framework. This specific training objective ensures that the learned DIR approximates the proximal operator of an ideal explicit restoration regularizer, distinguishing it from standard unrolled networks that may lack this explicit proximal interpretation.

## References

[1] Gradient projection for sparse reconstruction: Application to compressed sensing and other inverse problems
[2] Iterative Hard Thresholding for Compressed Sensing
[3] A fast iterative shrinkage-thresholding algorithm for linear inverse problems
[4] Alternating Direction Algorithms for {$\ell_{1}$}-Problems in Compressive Sensing
[5] Compressive sensing via nonlocal low-rank regularization
[6] From Denoising to Compressed Sensing
[7] Gradient Step Denoiser for convergent Plug-and-Play
[8] Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization
[9] What's in a Prior? Learned Proximal Networks for Inverse Problems
[10] Stochastic Deep Restoration Priors for Imaging Inverse Problems
[11] ReconNet: Non-Iterative Reconstruction of Images from Compressively
  Sensed Random Measurements
[12] Scalable convolutional neural network for image compressed sensing
[13] Dr2-net: Deep residual reconstruction network for image compressive sensing
[14] Learned D-AMP: Principled Neural Network based Compressive Image Recovery
[15] ISTA-Net: Interpretable optimization-inspired deep network for image compressive sensing
[16] ADMM-CSNet: A deep learning approach for image compressive sensing
[17] AMP-Net: Denoising-based deep unfolding for compressive image sensing
[18] Memory-Augmented Deep Unfolding Network for Compressive Sensing
[19] COAST: COntrollable Arbitrary-Sampling NeTwork for Compressive Sensing
[20] Deep Generalized Unfolding Networks for Image Restoration
[21] CSformer: Bridging Convolution and Transformer for Compressive Sensing
[22] Optimization-Inspired Cross-Attention Transformer for Compressive
  Sensing
[23] Saunet: Spatial-attention unfolding network for image compressive sensing
[24] UFC-Net: Unrolling Fixed-point Continuous Network for Deep Compressive Sensing
[25] CPP-Net: Embracing Multi-Scale Feature Fusion into Deep Unfolding CP-PPA Network for Compressive Sensing
[26] Dual-Scale Transformer for Large-Scale Single-Pixel Imaging
[27] Nonlinear image recovery with half-quadratic regularization
[28] The Little Engine that Could: Regularization by Denoising (RED)