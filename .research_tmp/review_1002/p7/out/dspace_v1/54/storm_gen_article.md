## Related Work

**Input Representation**
A dominant paradigm in graph-based collaborative filtering relies on learned user and item embeddings to represent preferences within a latent space [2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24]. In contrast, a smaller body of work bypasses learned embeddings by directly utilizing raw interaction histories or alternative signal representations [21, 25, 26, 27, 29]. Other approaches incorporate side-information features [3, 4, 5, 7] or construct personalized graph signals on augmented similarity graphs [28]. While embedding-based methods dominate the literature, they may limit the capacity to capture locally observed preference patterns. ChebyCF aligns with the latter group by operating on raw interaction history, thereby utilizing the full spectrum of signals contained in the user's history without the constraints of a learned latent space.

**Graph Filter Form**
Most prior graph convolutional networks for recommendation employ linear low-pass filters, often derived from first-order approximations of spectral convolutions [3, 9, 10, 11, 12, 13, 14, 15, 16, 20, 23, 24, 26]. Some methods utilize fixed polynomial filters [4, 19] or specific non-linear mechanisms such as biased random walks [2], neighborhood aggregation [5], localized spectral filtering [6], generalized PageRank [7], Bernstein approximation [8], hybrid linear/non-linear propagation with gating [17], adaptive trend filtering [18], heat equation diffusion [21], hypergraph-enhanced GNNs [22], exact spectral diagonalization [25], iterative degree updates [27], mixed-frequency filters [28], or individualized graph filters [29]. However, no cited prior work adopts a flexible non-linear filter via Chebyshev interpolation. ChebyCF distinguishes itself by employing Chebyshev interpolation to approximate a flexible non-linear graph filter, overcoming the restricted linear forms that limit the fine-grained leveraging of diverse preference patterns in existing methods.

**Frequency Spectrum Utilization**
The majority of graph-based CF models operate under a smoothness assumption, effectively acting as low-pass filters that discard high-frequency signals [3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 21, 26]. A few approaches utilize the full spectrum without a cut-off [7, 25, 29], while others employ band-pass filtering strategies [19, 28]. By identifying that standard graph convolutions introduce a cut-off in the frequency spectrum, ChebyCF joins the group of methods that utilize the full spectrum. This approach ensures that potentially valuable high-frequency signals contained in the user's interaction history are not discarded, addressing a key bottleneck in low-pass filtering methods.

**Normalization Strategy**
Standard graph convolutional networks typically employ symmetric normalization ($D^{-1/2}AD^{-1/2}$) [3] or random walk normalization ($D^{-1}A$) [7]. Other works have explored degree-based normalization [27] or generalized graph normalization (G2N) [29]. ChebyCF adopts degree-based normalization as a specific component to enhance the Chebyshev filter. This choice distinguishes ChebyCF from the standard normalization schemes in prior GCN-based CF models and contributes to its state-of-the-art performance by effectively handling the graph structure during spectral filtering.

## References

[1] Deep{W}alk: Online learning of social representations
[2] node2vec: Scalable Feature Learning for Networks
[3] Semi-Supervised Classification with Graph Convolutional Networks
[4] Simplifying Graph Convolutional Networks
[5] How Powerful are Graph Neural Networks?
[6] Convolutional neural networks on graphs with fast localized spectral filtering
[7] Adaptive Universal Generalized PageRank Graph Neural Network
[8] Bern{N}et: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation
[9] Revisiting Graph based Collaborative Filtering: A Linear Residual Graph
  Convolutional Network Approach
[10] Neural Graph Collaborative Filtering
[11] Light{GCN}: Simplifying and Powering Graph Convolution Network for Recommendation
[12] Ultra{GCN}: Ultra Simplification of Graph Convolutional Networks for Recommendation
[13] Simplifying graph-based collaborative filtering for recommendation
[14] Neighbor Interaction Aware Graph Convolution Networks for Recommendation
[15] Disentangled Graph Collaborative Filtering
[16] Interest-aware Message-Passing GCN for Recommendation
[17] Linear, or Non-Linear, That is the Question!
[18] Graph Trend Filtering Networks for Recommendations
[19] On Manipulating Signals of User-Item Graph: A Jacobi Polynomial-based
  Graph Collaborative Filtering
[20] Collaboration-Aware Graph Convolutional Network for Recommender Systems
[21] Graph Signal Diffusion Model for Collaborative Filtering
[22] Hypergraph Contrastive Collaborative Filtering
[23] Improving Graph Collaborative Filtering with Neighborhood-enriched
  Contrastive Learning
[24] Adaptive Graph Contrastive Learning for Recommendation
[25] Spectral Collaborative Filtering
[26] How Powerful is Graph Convolution for Recommendation?
[27] Revisiting Neighborhood-based Link Prediction for Collaborative
  Filtering
[28] Personalized Graph Signal Processing for Collaborative Filtering
[29] How Powerful is Graph Filtering for Recommendation
[30] Turbo-{CF}: Matrix decomposition-free graph filtering for fast recommendation