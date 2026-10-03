## Related Work

**Optimization Scope**
Prior research on LLM efficiency is broadly categorized into model-only optimizations, system-only optimizations, and algorithm-hardware co-designs. A substantial body of work focuses exclusively on model-level techniques, such as quantization [2, 3, 4, 11, 12, 14], pruning [15, 16], and knowledge distillation [17, 18], which aim to reduce model size or computational complexity without altering the underlying execution environment. In contrast, system-level optimizations target the inference engine and resource management, including scheduling strategies [6], memory management for KV caches [8], and kernel tuning or data transfer optimizations [5, 7, 9, 22, 26, 27, 28]. While some frameworks propose application-aware neural network optimization [25], the majority of prior efforts remain siloed within either the algorithmic or the system domain. Only a limited number of studies, such as [10, 24], adopt an algorithm-hardware co-design approach that jointly optimizes model and system levels. CLONE aligns with this co-design paradigm, distinguishing itself by integrating real-time energy optimization directly into the hardware accelerator design to achieve synergistic benefits that neither model nor system optimizations alone can provide.

**Hardware Target**
The choice of hardware substrate significantly influences the potential for energy-aware inference. Many existing studies rely on general-purpose accelerators, such as GPUs [5, 7, 13, 14, 22] or CPUs [9], or specialized datacenter chips like TPU v4 Pods [2]. Other works explore specific hardware features, such as on-chip switching regulators for per-core DVFS [27] or custom 12nm accelerators [24]. However, there is a notable gap in the literature regarding the deployment of LLMs on custom 28nm scalable hardware accelerators tailored for edge constraints. Unlike software-centric approaches that target general-purpose hardware, CLONE specializes in a custom 28nm scalable hardware accelerator system, enabling specific architectural features that are critical for balancing latency and energy consumption in resource-constrained environments.

**Optimization Objective**
The primary objective of optimization varies across prior works, ranging from single-metric improvements to multi-objective trade-offs. Several studies focus solely on latency reduction [5, 9, 13, 28] or accuracy preservation under fixed resource constraints [2, 3, 4]. Others target energy efficiency exclusively [26, 27] or specific combinations of throughput and latency [6], efficiency and power/cost [7], or throughput and memory efficiency [8]. While some works address resource efficiency and accuracy [11, 12, 15, 16, 17] or generation quality [18], few explicitly integrate real-time energy optimization alongside latency and accuracy. Papers [10, 14, 24] share the multi-objective focus of this work, but CLONE uniquely targets the specific constraints of edge devices by jointly optimizing latency, energy, and accuracy, thereby addressing the critical need for power-aware inference in always-on settings.

**Deployment Context**
The deployment context of LLM inference is typically either cloud/server-side or specific edge scenarios. A significant portion of prior work targets cloud or server-side inference [2, 6, 7, 8, 13, 14], where power and storage constraints are less stringent. Other studies focus on personal computers or edge devices with limited DRAM [5, 9], batch processing on high-power workstations [22], or autonomous systems [28]. The niche of "always-on" and "intermediate" edge computing settings, which requires balancing low power consumption with the high computational demands of LLMs, is underrepresented. Only [24] shares this specific deployment context with CLONE. By focusing on always-on and intermediate edge settings, CLONE addresses a distinct need for efficient, low-power inference that is not adequately covered by cloud-focused or simple IoT-oriented prior work.

## References

[1] {GPT-4} Technical Report
[2] PaLM: Scaling Language Modeling with Pathways
[3] LLaMA: Open and Efficient Foundation Language Models
[4] Llama 2: Open Foundation and Fine-Tuned Chat Models
[5] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
[6] ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference
[7] Splitwise: Efficient generative LLM inference using phase splitting
[8] Efficient Memory Management for Large Language Model Serving with
  PagedAttention
[9] LLM in a flash: Efficient Large Language Model Inference with Limited
  Memory
[10] New Solutions on LLM Acceleration, Optimization, and Application
[11] Quantized neural networks: Training neural networks with low precision weights and activations
[12] LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
[13] Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time
[14] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
  Language Models
[15] Pruner-Zero: Evolving Symbolic Pruning Metric From Scratch for Large
                  Language Models
[16] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot
[17] DistiLLM: Towards Streamlined Distillation for Large Language Models
[18] MiniLLM: Knowledge Distillation of Large Language Models
[19] PyTorch: An Open Source Machine Learning Framework
[20] TensorFlow: An Open Source Machine Learning Framework for Everyone
[21] DeepSpeed: Advancing the Science of AI Through Efficient Training of Large Models
[22] FlexGen: High-Throughput Generative Inference of Large Language Models
  with a Single GPU
[23] Orca: {A} Distributed Serving System for Transformer-Based Generative
                  Models
[24] EdgeBERT: Sentence-Level Energy Optimizations for Latency-Aware
  Multi-Task NLP Inference
[25] Approxcaliper: A programmable framework for application-aware neural network optimization
[26] Variation-aware dynamic voltage/frequency scaling
[27] System level analysis of fast, per-core {DVFS} using on-chip switching
                  regulators
[28] $\{$NeuOS$\}$: A $\{$Latency-Predictable$\}$$\{$Multi-Dimensional$\}$ Optimization Framework for $\{$DNN-driven$\}$ Autonomous Systems