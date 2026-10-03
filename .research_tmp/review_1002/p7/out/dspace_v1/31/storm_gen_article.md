## Related Work

**Pre-training Objectives**
Prior research in DNA foundation models has predominantly relied on unsupervised or self-supervised objectives to learn sequence representations. A significant body of work employs masked language modeling (MLM) to capture bidirectional context, as seen in models like DNABERT [7] and its derivatives [8, 9, 10]. Alternatively, other approaches utilize self-supervised next-token prediction to model sequential dependencies [12, 13]. While some recent efforts have explored mixed paradigms combining supervised and self-supervised signals [1], the majority of existing methods treat DNA as a generic language, ignoring the functional regulatory signals that govern gene expression. In contrast, this paper argues that pure sequence-based pre-training is suboptimal and demonstrates that supervised genomic profile prediction, such as chromatin accessibility, serves as a more effective alternative for learning biologically grounded representations.

**Model Architectures**
The architectural choices in genomic modeling have evolved from dense structures to sparse, scalable designs. Early and standard approaches typically employ dense Transformer architectures [7, 8, 9] or convolutional neural networks (CNNs) to predict regulatory activity [3]. To address computational constraints and long-range dependencies, recent works have introduced state-space models such as Mamba [10]. More recently, the Mixture of Experts (MoE) paradigm has been adopted to increase model capacity and efficiency, with prior works utilizing sparsely-gated layers [12], adaptive local expert selection [11], and simplified routing mechanisms for scaling [13]. While these MoE-based approaches share the architectural foundation of our work, they are generally designed for generic language modeling tasks. SPACE distinguishes itself by introducing Species-Profile Adaptive routing within the MoE framework, specifically tailored to capture the complex relationships between DNA sequences across different species and genomic profiles.

**Data Scope**
The scope of training data in genomic foundation models varies significantly, ranging from single-species to multi-species datasets. Many prominent models, including Nucleotide Transformer [6] and HyenaDNA [10], are primarily trained on or evaluated against single-species (e.g., human-only) data [1]. While DNABERT-2 has expanded the scope to include multi-species genomic data [9], the integration of diverse regulatory profiles across these species remains underexplored. Most prior works treat species and profiles as separate or static contexts. In contrast, SPACE is explicitly designed for multi-species and multi-profile genomic data, leveraging its adaptive expert system to effectively model the heterogeneity inherent in cross-species and cross-profile regulatory landscapes.

**Representation Learning Strategies**
The strategy for learning DNA representations has largely followed the NLP paradigm of generic language modeling. Models such as DNABERT [7, 8, 9], HyenaDNA [10], and various MoE-based architectures [12, 13] focus on capturing statistical patterns in nucleotide sequences without explicit grounding in biological function [1]. Although some studies have integrated functional signals in a mixed manner [1], a dedicated approach to functional-aware representation learning via regulatory signals is absent in the cited prior work. By grounding representation learning in supervised genomic profiles, SPACE moves beyond generic sequence statistics to learn effective DNA representations that are directly informed by biological function, thereby addressing a key gap in current foundation models.

## References

[1] Leveraging genomic deep learning models for non-coding variant effect
  prediction
[2] Predicting effects of noncoding variants with deep learning--based sequence model
[3] Sequential regulatory activity prediction across chromosomes with convolutional neural networks
[4] Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk
[5] Effective gene expression prediction from sequence by integrating long-range interactions
[6] Nucleotide Transformer: building and evaluating robust foundation models for human genomics
[7] DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome
[8] BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding
[9] DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genomes
[10] HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide
  Resolution
[11] Adaptive Mixtures of Local Experts
[12] Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer
[13] Switch Transformers: Scaling to Trillion Parameter Models with Simple
  and Efficient Sparsity