## Related Work

**Operation Coverage**
Prior research on Vision Transformer (ViT) acceleration has largely focused on optimizing specific components or approximating complex functions rather than implementing the full model pipeline on dedicated hardware. For instance, HeatViT employs adaptive token pruning and polynomial approximations to handle ViT operations on embedded FPGAs [6]. In contrast, this work addresses the limitation of partial operation mapping by proposing the first ViT accelerator that maps all operations, including accuracy-sensitive Softmax and LayerNorm, directly onto FPGAs. This comprehensive coverage eliminates the communication overhead between the CPU and accelerator that plagues methods relying on offloading complex arithmetic to general-purpose processors.

**Data Format**
The selection of data representation is critical for balancing precision and hardware efficiency in neural network inference. Existing approaches utilize a variety of quantization strategies, including learned quantization for compact networks [1], general quantization techniques for convolutional networks [2], and aggressive 3-bit compression for attention-based models [5]. Other works explore mixed-precision bitwidths ranging from 1 to 8 bits [4] or standard fixed-point formats such as INT8 and INT4 [6]. While recent research has introduced Microscaling (MX) and Block Data Representations (BDR) using shared microexponents to improve narrow-precision performance [3], no prior cited work utilizes the specific Microscaling Integer (MXInt) format. This paper adopts MXInt as the core data format, enabling efficient hardware implementation of complex operations without the significant accuracy loss associated with lower-precision fixed-point or aggressive compression methods.

**Hardware Target**
Hardware acceleration for deep learning models is typically implemented on specialized ASICs or general-purpose processors, with FPGAs serving as a flexible alternative for reconfigurable datapaths. Several studies target DNN hardware accelerators for edge and cloud environments [4], while others design custom architectures similar to TPUs, Eyeriss, or Tensor Cores [5]. HeatViT is a notable exception that targets embedded FPGAs, leveraging their reconfigurability for ViT inference [6]. This paper aligns with the FPGA target but distinguishes itself by exploiting the reconfigurability to implement custom datapaths specifically optimized for the MXInt format, rather than relying on standard fixed-point arithmetic units.

**Accuracy-Performance Trade-off Strategy**
Strategies for balancing model accuracy with hardware performance vary significantly across the literature. Some approaches focus on format optimization to enhance accuracy and performance metrics [3], while others employ hardware-aware automated mixed-precision techniques using reinforcement learning [4]. Aggressive compression strategies, such as 3-bit quantization without fine-tuning [5], and approximation-based methods like token pruning and polynomial approximation [6], represent different points on the accuracy-performance spectrum. Unlike these methods, which often sacrifice precision or rely on algorithmic approximations, this paper proposes custom hardware optimization specifically for MXInt. This strategy ensures that the accelerator maintains less than 1% accuracy loss while achieving superior area efficiency and performance, a trade-off not previously achieved in the cited prior work.

## References

[1] LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks
[2] Quantizing deep convolutional networks for efficient inference: A whitepaper
[3] With Shared Microexponents, A Little Shifting Goes a Long Way
[4] HAQ: Hardware-Aware Automated Quantization with Mixed Precision
[5] GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy
  Efficient Inference
[6] HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision
  Transformers