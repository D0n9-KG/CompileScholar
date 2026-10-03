## Related Work

**Metric Paradigm**
Existing approaches to Maximum Inner Product Search (MIPS) generally adhere to one of two distinct metric paradigms. One line of work relies exclusively on Inner Product (IP) similarity, employing techniques such as global multi-probing [7], lightweight indexing [8], tree-based structures [9], and various vector quantization methods that explicitly handle norms or anisotropic residuals [10, 11]. Other methods, including non-metric similarity graphs and norm-adjusted proximity graphs, also operate within the IP similarity framework [13, 14, 16]. In contrast, a separate group of studies reduces the MIPS problem to Nearest Neighbor Search (NNS) under the Euclidean metric via space projection or transformation. This includes Asymmetric LSH (ALSH) and its improvements [2, 3, 6], Euclidean transformations for inner-product spaces [4], and the application of Möbius transformations [15]. While these two groups represent the dominant strategies in the literature, no prior work has addressed the inherent limitations of each by stitching IP and Euclidean metrics together, a gap that this paper fills with its hybrid approach.

**Indexing Structure**
The choice of indexing structure varies significantly across prior MIPS research. Hashing-based structures, particularly Locality Sensitive Hashing (LSH), are widely used, including in surveys of learning to hash [1] and specific MIPS algorithms like Norm-Ranging LSH [5] and ALSH variants [2, 3, 6]. Vector Quantization (VQ) is another prominent structure, utilized in Norm-Explicit Quantization [10] and anisotropic vector quantization [11]. Other structural choices include lightweight indices [8], tree-based methods such as the LRUS-CoverTree [9], and spill trees with redundant representations [12]. Graph-based indices have also gained traction, with prior works proposing non-metric similarity graphs [13], angular proximity graphs [14], Möbius-transformed graphs [15], and norm-adjusted proximity graphs [16]. While graph-based structures are shared by several recent studies [13, 14, 15, 16], this paper introduces the Metric-Amphibious Graph (MAG), a novel graph-based index designed specifically to support the flexible stitching of different metric behaviors.

**Search Strategy**
Search strategies in MIPS literature are largely static or follow fixed procedural stages. Most prior works employ search mechanisms intrinsic to their specific indexing structures, such as global multi-probing for LSH-based methods [7] or standard graph traversal for proximity graphs [13, 15, 16]. A notable exception is the two-stage graph search strategy proposed in [14], which utilizes an angular proximity graph to initialize search before transitioning to an inner-product graph. However, no cited prior work implements an adaptive navigation strategy that dynamically switches between IP and Euclidean metrics during the search process. This paper’s Adaptive Navigation with Metric Switch (ANMS) algorithm distinguishes itself by enabling real-time metric switching, thereby overcoming the rigidity of static or two-stage approaches.

**Parameter Tuning Basis**
Parameter tuning in MIPS is often empirical or dataset-specific, with limited principled guidance. Most prior works do not explicitly define statistical indicators for tuning based on data topology. An exception is Norm-Ranging LSH [5], which partitions datasets based on norm distribution to address long tails in the 2-norm, thereby incorporating a form of data topology awareness into its design. This paper aligns with [5] in recognizing the importance of data topology but extends this by identifying three specific statistical indicators that capture essential topology properties and correlate strongly with optimal parameter settings for the proposed hybrid index.

## References

[1] A Survey on Learning to Hash
[2] Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search
  (MIPS)
[3] Improved Asymmetric Locality Sensitive Hashing (ALSH) for Maximum Inner
  Product Search (MIPS)
[4] Speeding up the xbox recommender system using a euclidean transformation for inner-product spaces
[5] Norm-Ranging LSH for Maximum Inner Product Search
[6] On Symmetric and Asymmetric LSHs for Inner Product Search
[7] FARGO: Fast maximum inner product search via global multi-probing
[8] ProMIPS: Efficient high-dimensional C-approximate maximum inner product search with a lightweight index
[9] Reconsidering Tree based Methods for k-Maximum Inner-Product Search: The LRUS-CoverTree
[10] Norm-Explicit Quantization: Improving Vector Quantization for Maximum
  Inner Product Search
[11] Accelerating Large-Scale Inference with Anisotropic Vector Quantization
[12] SOAR: Improved Indexing for Approximate Nearest Neighbor Search
[13] Non-metric similarity graphs for maximum inner product search
[14] Understanding and Improving Proximity Graph based Maximum Inner Product
  Search
[15] M{\"o}bius transformation for fast inner product search on graph
[16] Norm adjusted proximity graph for fast inner product retrieval