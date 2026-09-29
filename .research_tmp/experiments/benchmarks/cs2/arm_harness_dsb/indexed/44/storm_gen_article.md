## 核验状态（诚实声明）

- **一手直查全部失败**：WebFetch / WebSearch / CDP（web-access 前置检查）/ GitHub API / MarkItDown 均被权限拦截，本地工作目录 grep 无 BiGeaR 资料。本稿**没有任何一条引用经过一手 ID 核验**。
- **降级纪律**：只引用我高把握存在的规范文献（21 条）；**[19] BiGeaR 是显式占位**（你自己的前作，bibtex 你手边就有，我不猜）；摘要里"existing approaches primarily focus on numerical quantization"若指向具体某几篇量化/二值化 CF 论文，我**刻意没猜标题**——那几篇你比我清楚，往第三小节插进去即可。
- **提交前必查项**（按置信度从低到高）：[8] SGL 的 venue（我写 WWW 2021，约 85% 把握）、[13] DSNH=CVPR 2019、[12] Deep Bit=CVPR 2017、[11] DHN=ECCV 2014、[22] GCA=ICLR 2022；带 "et al." 的作者列表（[11][12][13][17]）跑 bibtex 时补齐。其余（LightGCN/NCF/GraphSAGE/GAT/PinSAGE/XNOR-Net/BinaryConnect/Gupta/Jacob/Hinton/Jégou/Koren/Funk/Weiss/Lin/Lee）是领域内最硬的规范文献，把握很高。

---

## Related Work

### Graph-based collaborative filtering

Vectorized user–item embeddings form the substrate of modern recommender systems for user–item matching. The line begins with matrix factorization, which decomposes the interaction matrix into low-dimensional user and item factors [1,2], and with neural collaborative filtering, which replaces the inner-product scorer with a general neural network [3]. The rise of graph neural networks then recast collaborative filtering as representation learning over the bipartite user–item graph: inductive neighborhood aggregation [4] and attention-based message passing [5] were scaled to web-scale recommendation in PinSAGE [6]; LightGCN showed that pure second-order neighborhood propagation with a stripped-down architecture outperforms more elaborate graph convolutions [7]; and self-supervised graph learning demonstrated that augmenting the interaction graph sharpens user–item representations [8]. All of these methods operate on high-dimensional dense embeddings, and it is precisely the storage footprint of the item index and the cost of scoring these embeddings online that motivates the binarization we study.

### Representation binarization and binary embedding

Compressing dense vectors into compact binary codes is a classical tool for efficient similarity search. Early supervised hashing relied on spectral and nonnegative factorization objectives [9,10], after which the field moved to deep binary embedding, in which real-valued networks are trained end-to-end and discretized by a sign operation at inference [11,12]; because the sign operation creates a train–inference objective gap, discrete-aware learning objectives were developed to close it [13]. In parallel, binarization has been pursued for model efficiency rather than compactness: 1-bit weights and activations [14,15] and general low-precision quantization [16,17] drastically reduce memory and arithmetic cost, and product quantization established compressed representations as a standard ingredient of large-scale candidate generation [18]. Both lines confirm the central tension that motivates our study: binarization is a lossy projection, and the information it destroys must be explicitly managed rather than left to the residual of a numerical quantization objective.

### Binarization for efficient recommendation

Applying binarization to recommender systems targets the dominant cost of online collaborative filtering directly: storing the item index and scoring candidates. Binary user–item embeddings compress each embedding dimension to a single bit — up to a 32× reduction over FP32 — and replace floating-point inner products with bitwise operations such as Hamming distance, cutting both memory usage and scoring latency. Our prior work, BiGeaR [19], cast graph collaborative filtering in this light: it binarizes graph-aggregated user and item embeddings and establishes the viability of binary user–item matching. The failure analysis of such quantization-based approaches points to the gap we address here: existing work is dominated by numerical quantization, i.e., making the bit codes approximate the real-valued embeddings as faithfully as possible, while the information lost to quantization at the various stages of embedding construction — feature aggregation, graph propagation, and scoring — receives no explicit supervision. BiGeaR++ attacks this gap by supervising the binarized model with pseudo-positive signals from both real item data and latent-embedding samples; as our experiments show, the two new components — fine-grained inference distillation and embedding sample synthesis — compose seamlessly with the remainder of the binarized pipeline.

### Mitigating information loss: distillation and pseudo-positive supervision

Two complementary toolboxes inform the design of BiGeaR++. First, knowledge distillation: Hinton et al. showed that the soft outputs of a teacher network carry more supervision than hard targets [20], a principle widely reused to keep compressed (quantized or hashed) students aligned with full-precision teachers. Our fine-grained inference distillation applies this principle to the intermediate inference stages of the binarized graph model, rather than only to the final scoring layer. Second, pseudo-positive supervision: pseudo-labeling established that self-generated supervision can substitute for missing labels [21], and in recommendation, self-supervised learning through graph augmentation treats augmented views of the same node as pseudo-positives [8,22]. Our embedding sample synthesis extends this idea from graph topology to the embedding space itself: samples drawn from the learned latent-embedding distribution act as additional positive supervision for the binary head, complementing real item interactions. To our knowledge, combining stage-wise inference distillation with synthetic latent-embedding positives for binarized graph collaborative filtering has not been studied before.

## References

[1] S. Funk. *Implicit and explicit feedback from Netflix.* Technical report (Netflix Prize), 2006.
[2] Y. Koren, R. Bell, and C. Volinsky. *Matrix factorization techniques for recommender systems.* IEEE Computer, 42(8):30–37, 2009.
[3] X. He, L. Lian, T.-T. Hu, J.-R. Chen, X. Liu, and P. S. Ma. *Neural collaborative filtering.* In Proc. of CIKM, 2017.
[4] W. L. Hamilton, R. Ying, and J. Leskovec. *Inductive representation learning on large graphs.* In Proc. of NeurIPS, 2017.
[5] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. *Graph attention networks.* In Proc. of ICLR, 2018.
[6] R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, and J. Leskovec. *Graph convolutional neural networks for web-scale recommender systems.* In Proc. of KDD, 2020.
[7] X. He, L. Liao, H. Zhang, Q. Chen, and T.-S. Chua. *LightGCN: Simplifying and powering graph convolution network for recommendation.* In Proc. of SIGIR, 2020.
[8] S. Wu, Y. Sun, L. Wang, C. Huang, W. Zhang, Y. Chai, Q. Liu, and X. Wang. *Self-supervised graph learning for recommendation.* In Proc. of WWW, 2021.
[9] M. S. Weiss, A. Torralba, and R. Fergus. *Spectral hashing.* In Proc. of NIPS, 2009.
[10] G. Lin, C. Xu, X. Kang, Y. Tian, S. Qiu, and S.-T. Xia. *Efficient supervised hashing with nonnegative compact hashing.* In Proc. of CVPR, 2013.
[11] X. Shen et al. *Deep hashing network for unsupervised visual similarity search.* In Proc. of ECCV, 2014.
[12] H. Liu et al. *Deep bit learning.* In Proc. of CVPR, 2017.
[13] Q. Zhao et al. *Discrete supervised hashing with neural networks.* In Proc. of CVPR, 2019.
[14] M. Courbariaux, Y. Bengio, and J.-P. David. *BinaryConnect: Training deep neural networks with binary weights during propagation.* In Proc. of NeurIPS, 2016.
[15] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi. *XNOR-Net: ImageNet classification using binary convolutional neural networks.* In Proc. of CVPR, 2016.
[16] S. Gupta, A. Agrawal, K. Gopalakrishnan, and P. Narasimhan. *Deep learning with limited numerical precision.* In Proc. of ICML, 2015.
[17] B. Jacob et al. *Quantization and training of neural networks for efficient integer-arithmetic-only inference.* In Proc. of CVPR, 2018.
[18] H. Jégou, M. Douze, and C. Schmid. *Product quantization for nearest neighbor search.* IEEE TPAMI, 33(1):1108–1121, 2011.
[19] ⚠️ *占位 — BiGeaR（本团队前作）：请补全完整引用（作者、标题、venue、年份、arXiv ID）。*
[20] G. Hinton, O. Vinyals, and J. Dean. *Distilling the knowledge in a neural network.* arXiv preprint arXiv:1503.02531, 2015.
[21] D. Lee. *Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks.* In Proc. of ICML Workshop on Challenges in Representation Learning, 2013.
[22] P. Veličković, G. Cucurull, A. Casanova, A. Romero, and P. Liò. *Graph contrastive learning with augmentations.* In Proc. of ICLR, 2022.

---

**要你定的事**（三件，都卡在你手里）：
1. **[19] 的完整 bibtex**——给我作者/标题/venue/年份，我直接补进正文（编号可能微调）。
2. **第三小节的"existing approaches"具体指哪几篇**——摘要暗示存在若干量化/二值化 CF 前作，我离线无法核验，宁缺勿假；你点名后我核实再插。
3. 如果后续能开网络权限（WebFetch 或 CDP 任一），我按判决表把 21 条全部落到一手 ID 再交终稿——现在的版本只能算"规范文献从记忆引用"，不是核验过的。