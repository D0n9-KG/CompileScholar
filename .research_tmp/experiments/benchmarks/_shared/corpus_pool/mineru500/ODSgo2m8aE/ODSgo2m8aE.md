# ALIGNING RELATIONAL LEARNING WITH LIPSCHITZ FAIRNESS

Yaning Jia $^{\dagger}$ , Chunhui Zhang $^{\dagger}$ , Soroush Vosoughi $^{*}$

Dartmouth College, Hanover, NH, USA

HUST, Hubei, China

# ABSTRACT

Relational learning has gained significant attention, led by the expressiveness of Graph Neural Networks (GNNs) on graph data. While the inherent biases in common graph data are involved in GNN training, it poses a serious challenge to constraining the GNN output perturbations induced by input biases, thereby safeguarding fairness during training. The Lipschitz bound, a technique from robust statistics, can limit the maximum changes in the output concerning the input, taking into account associated irrelevant biased factors. It is an efficient and provable method to examine the output stability of machine learning models without incurring additional computational costs. Recently, its use in controlling the stability of Euclidean neural networks, the calculation of the precise Lipschitz bound remains elusive for non-Euclidean neural networks like GNNs, especially within fairness contexts. However, no existing research has investigated Lipschitz bounds to shed light on stabilizing the GNN outputs, especially when working on graph data with implicit biases. To narrow this gap, we begin with the general GNNs operating on relational data, and formulate a Lipschitz bound to limit the changes in the output regarding biases associated with the input. Additionally, we theoretically analyze how the Lipschitz bound of a GNN model could constrain the output perturbations induced by biases learned from data for fairness training. We experimentally validate the Lipschitz bound's effectiveness in limiting biases of the model output. Finally, from a training dynamics perspective, we demonstrate why the theoretical Lipschitz bound can effectively guide the GNN training to better trade-off between accuracy and fairness.

# 1 INTRODUCTION

Relational learning on network data has become ubiquitous in a range of real-world applications, such as social media (Vosoughi et al., 2016; Shalaby et al., 2017; Huang et al., 2021a; Li et al., 2021; Tian et al., 2023a; Ouyang et al., 2024), drug discovery (Takigawa & Mamitsuka, 2013; Li et al., 2017), and knowledge engineering (Rizun, 2019; Wang et al., 2018). This surge in the use of graph data has catalyzed advances in learning algorithms that incorporate Graph Neural Networks (GNNs) with deep learning techniques (Gori et al., 2005; Scarselli et al., 2005; Li et al., 2016; Hamilton et al., 2017; Xu et al., 2019; Zhang et al., 2022, 2023a,b,c; Liu et al., 2023; Yuan et al., 2024). Among GNN architectures, Graph Convolutional Networks (GCNs) (Kipf & Welling, 2017; Zhang & Chen, 2018; Fan et al., 2019) stand out due to their use of convolutional layers and message-passing mechanisms to enable effective learning on graphs.

Amidst the widespread success of GNNs in diverse applications, there has been growing societal concern for developing ethical and prosocial learning algorithms for networks (Vosoughi et al., 2017, 2018; Dong et al., 2021; Lahoti et al., 2019; Kang et al., 2020; Mujkanovic et al., 2022). Many existing approaches, however, fall short of explicating the interactions between the GNN model and its graph-based training data. This opacity hampers the model's parameter tunability, failing to adequately address latent biases in the graph data, thereby undermining model reliability post-training. In this context, our objective is to deepen our understanding of the input-output dynamics of GNNs

in relation to model parameters. Specifically, we confront the following pivotal question: How can we minimize unintended shifts in GNN output, especially when the graph training data encompasses implicit, learnable biases, without resorting to computationally intensive methods?

This question is crucial in scenarios where the training graph data may contain spurious statistics that result in unfair or inappropriate correlations. Our work aims to provide a mechanism that curtails such fluctuations in the GNN model's predictions during training, ensuring they are not disproportionately influenced by unfair factors. The implications of this work extend to enhancing GNN generalization, debiasing GNNs, and safeguarding against data perturbations by regularizing output changes. Although existing GNN training methodologies have yielded impressive results, they often lack interpretability, particularly in understanding how layer-by-layer learned biases affect output stability or fairness (Dong et al., 2021; Kang et al., 2020; Liao et al., 2021; Li et al., 2021; Yue et al., 2022).

In robust statistics, Lipschitz bounds serve as valuable tools for assessing maximal output changes in response to input biases (Virmaux & Scaman, 2018; Fazlyab et al., 2019; Jordan & Dimakis, 2020; Latorre et al., 2020; Huang et al., 2021b). Prior work (Dwork et al., 2012; Shi et al., 2022; Agarwal et al., 2021) has mainly focused on fairness and robustness in MLPs, CNN, or statistical models, often relying on explicitly labeled sensitive attributes, such as gender, race, and age.

Our research distinguishes itself by extending fairness considerations specifically to the realm of relational learning on graphs. We achieve this by constructing a fairness framework that does not necessitate manual annotation of sensitive attributes. Instead, our approach is designed to identify and mitigate biases inherent in node features and connectivity patterns within real-world graph data. Our methodology aligns closely with the principles of ranking-based individual fairness as articulated by Dong et al. (2021). In this framework, fairness is guaranteed if the relative ordering of instances, characterized by ranking lists based on the similarity between instance i and other instances, remains consistent in both input and output spaces. Importantly, this ordering should not be distorted by irrelevant or hidden biases in the input data. When these conditions are met, the model output will inherently satisfy individual fairness criteria. To operationalize these principles, we introduce a computational strategy that leverages the Jacobian matrix for the efficient calculation of the Lipschitz bound. Our computational approach utilizes intermediate tensors – specifically, the gradients of the model’s outputs with respect to its inputs – in PyTorch, facilitating practical and scalable implementations. This is particularly advantageous for large-scale graph datasets, an area where existing methods often falter in terms of scalability. Our contributions can be summarized as:

- We introduce a Lipschitz bound specifically tailored for GNNs to align with rank-based individual fairness. This allows us to characterize and constrain the perturbations in GNN predictions induced by input biases.   
- We develop an efficient method for computing the Lipschitz bound of GNNs by leveraging the Jacobian matrix, thereby accommodating the complex topology of graph structures for practical implementation.   
- Through theoretical and empirical validation, we demonstrate that our approach is both versatile and effective, serving as a plug-and-play solution that can enhance existing fairness-oriented graph learning methods.

# 2 PRELIMINARIES

Lipschitz Bound A function $f: \mathbb{R}^n \to \mathbb{R}^m$ is said to be Lipschitz continuous on an input set $\mathcal{X} \subseteq \mathbb{R}^n$ if there exists a bound $K \geq 0$ such that for all $x, y \in \mathcal{X}$ , $f$ satisfies:

$$
\left\| f (\boldsymbol {x}) - f (\boldsymbol {y}) \right\| \leq K \left\| \boldsymbol {x} - \boldsymbol {y} \right\|, \forall \boldsymbol {x}, \boldsymbol {y} \in \mathcal {X}. \tag {1}
$$

The smallest possible K in Equation (1) is the Lipschitz constant of f, denoted as $\operatorname{Lip}(f)$ :

$$
\operatorname{Lip} (f) = \sup _ {\boldsymbol {x}, \boldsymbol {y} \in \mathcal {X}, \boldsymbol {x} \neq \boldsymbol {y}} \frac {\| f (\boldsymbol {x}) - f (\boldsymbol {y}) \|}{\| \boldsymbol {x} - \boldsymbol {y} \|}. \tag {2}
$$

In this context, $f$ is referred to as a $K$ -Lipschitz function. The Lipschitz bound essentially quantifies the maximum change in the output of a function corresponding to a unit-norm perturbation in its

input. This makes the Lipschitz bound an important measure of a neural network's stability with respect to its input features. However, determining the exact Lipschitz bound can be computationally challenging. As discussed in Virmaux & Scaman (2018), finding the exact Lipschitz bound is shown to be NP-hard for deep models. Consequently, an upper bound is often sought as a practical alternative, and this upper bound is also referred to as the Lipschitz bound.

Graph Neural Networks We assume a given graph $G = G(V, E)$ , where V denotes the set of nodes and E denotes the set of edges. We will use $X = \{x_{1}, x_{2}, \cdots, x_{N}\} \subset R^{F}$ to denote the N node features in $R^{F}$ , as the input of any layer of a GNN. By abuse of notation, when there is no confusion, we also follow GNN literature and consider X as the $R^{N \times F}$ matrix whose i-th row is given by $x_{i}^{\top}$ , $i = 1, \cdots, N$ , though it unnecessarily imposes an ordering of the graph nodes.

GNNs are functions that operate on the adjacency matrix $A \in R^{N \times N}$ of a graph G. Specifically, an L-layer GNN can be defined as $f : R^{N \times F^{in}} \to R^{N \times F^{out}}$ that depends on A. Formally, we have:

Definition 1. An L-layer GNN is a function f that can be expressed as a composition of L message-passing layers $h^{l}$ and L - 1 activation functions $\rho^{l}$ , as follows:

$$
f = h ^ {L} \circ \rho^ {L - 1} \circ \dots \circ \rho^ {1} \circ h ^ {1}, \tag {3}
$$

where $h^{l} : R^{F^{l-1}} \to R^{F^{l}}$ is the l-th message-passing layer, $\rho^{l} : R^{F^{l}} \to R^{F^{l}}$ is the non-linear activation function in the l-th layer, and $F^{l-1}$ and $F^{l}$ denote the input and output feature dimensions for the l-th message-passing layer $h^{l}$ , respectively. In addition, we set $l = 1, \cdots, L$ .

Rank-based Individual Fairness of GNNs Rank-based Individual Fairness on GNNs (Dong et al., 2021) focuses on the relative order of instances rather than their absolute predictions. It ensures that similar instances, as measured by the similarity measure $S(\cdot, \cdot)$ , receive consistent rankings or predictions. The criterion requires that if instance $i$ is more similar to instance $j$ than to instance $k$ , then the predicted ranking of $j$ should be higher than that of $k$ , consistently. This can be expressed as:

$$
\text { if } S (\boldsymbol {x} _ {i}, \boldsymbol {x} _ {j}) > S (\boldsymbol {x} _ {i}, \boldsymbol {x} _ {k}), \text { then } \boldsymbol {Y} _ {i j} > \boldsymbol {Y} _ {i k}, \tag {4}
$$

where $Y_{ij}$ and $Y_{ik}$ denote the predicted rankings or predictions for instances $x_{j}$ and $x_{k}$ , respectively, based on the input instance $x_{i}$ . The criterion ensures that the predicted rankings or predictions align with the relative similarities between instances, promoting fairness and preventing discriminatory predictions based on irrelevant factors.

# 3 ESTIMATING LIPSCHITZ BOUNDS ON GNNs FOR FAIRNESS

We estimate the Lipschitz bounds for GNNs, which is crucial for analyzing output perturbations induced by input biases: Initially, in § 3.1, we establish Lipschitz bounds for GNNs and provide closed-form formulas for these bounds. Next, in § 3.2, we derive the model's Jacobian matrix to facilitate Lipschitz bounds' efficient calculation for practical training feasibility. Lastly, in § 3.3, we use these bounds to explore individual fairness, demonstrating how model stability can ensure output consistency, aligning with rank-based individual fairness definition.

# 3.1 STABILITY OF THE MODEL OUTPUT

To analyze the stability of the model's output, we examine the Lipschitz bound of the Jacobian matrix of the GNN model. In this regard, we introduce the following lemma:

Lemma 1. Given a function $g: \mathbb{R}^m \to \mathbb{R}^n$ with components $g_i$ for $i \in [n]$ , the Lipschitz bound of $g$ , denoted by $\operatorname{Lip}(g)$ , is bounded above by the norm of the vector of the Lipschitz bounds of its components, that is:

$$
\operatorname{Lip} (g) \leq \| [ \operatorname{Lip} (g _ {i}) ] _ {i = 1} ^ {n} \| \tag {5}
$$

where $\|\cdot\|$ represents the norm of a vector.

Lemma 1 provides an inequality that relates the norm of the difference between two vector-valued functions, $g(x)$ and $g(y)$ , to the norm of a vector composed of the component-wise differences of the functions evaluated at $x$ and $y$ . Based on Lemma 1, we can now present the following theorem:

Theorem 1. Let Y be the output of an L-layer GNN (denoted as $f(\cdot)$ ) with X as the input. Assuming the activation function (represented in $\rho(\cdot)$ ) is ReLU with a Lipschitz bound of $\operatorname{Lip}(\rho) = 1$ , then the cumulative Lipschitz bound of the entire GNN, $\operatorname{Lip}(f)$ , satisfies:

$$
\operatorname{Lip} (f) \leqslant \max _ {j} \prod_ {l = 1} ^ {L} F ^ {l ^ {\prime}} \left\| \left[ \mathcal {J} (h ^ {l}) \right] _ {j} \right\| _ {\infty}, \tag {6}
$$

where $F^{l'}$ represents the output dimension of the l-th message-passing layer, j is the index of the node (e.g., j-th), and the vector $\left[\mathcal{J}(h^{l})\right]=\left[\left\|\boldsymbol{J}_{1}(h^{l})\right\|,\left\|\boldsymbol{J}_{2}(h^{l})\right\|,\cdots,\left\|\boldsymbol{J}_{F^{l'}}(h^{l})\right\|\right]$ . Notably, $\boldsymbol{J}_{i}(h^{l})$ denotes the i-th row of the Jacobian matrix of the l-th layer's input and output, and $\left[\mathcal{J}(h^{l})\right]_{j}$ is the vector corresponding to the j-th node in the l-th layer $h^{l}(\cdot)$ .

Proof Sketch: $^{1}$ We initiate with investigating the Lipschitz property of GNNs, considering 1-Lipschitz activation functions (e.g., the widely-used ReLU (Nair & Hinton, 2010). The hidden states of node feature $x_{1}$ and $x_{2}$ are represented as $z_{1}$ and $z_{2}$ respectively. The Lipschitz bound between these hidden states is calculated as $\frac{\|z_{1}-z_{2}\|}{\|x_{1}-x_{2}\|}=\frac{\|(h(x_{1})-h(x_{2}))\|}{\|x_{1}-x_{2}\|}$ . Using the triangle inequality, this equation is further bounded by $\frac{\|z_{1}-z_{2}\|}{\|x_{1}-x_{2}\|}\leqslant\left\|\left[\frac{h(x_{1})_{i}-h(x_{2})_{i}}{\|x_{1}-x_{2}\|}\right]_{i=1}^{F^{\prime}}\right\|$ . The Lipschitz bound of individual elements is also analyzed, again using the triangle inequality. The proof then proceeds to analyze the Lipschitz bound of the individual elements using the operation of the l-th layer in $f(\cdot)$ . Finally, the Lipschitz bound for the GNN is established as $\operatorname{Lip}(f)=\max_{j}\prod_{l=1}^{L}F^{l^{\prime}}\left\|\left[\mathcal{J}(h^{l})\right]_{j}\right\|_{\infty}$ . It establishes that the difference in the output Y is controlled by the Lipschitz bound of the GNN, $\operatorname{Lip}(f)$ , and the difference in the input X. The above analysis assesses the model output stability with respect to irrelevant biases learned layer-by-layer from the input, during the forward.

The aforementioned Theorem 1 provides the cumulative Lipschitz bound of the entire GNN based on the layer outputs and the corresponding Jacobian matrices. It establishes that the Lipschitz bound of a GNN regulates the magnitude of changes in the output induced by input biases, consequently guaranteeing the model output stability and fairness against irrelevant factors.

# 3.2 SIMPLIFYING LIPSCHITZ CALCULATION OF THE GNN VIA JACOBIAN MATRIX

The approach presented in Theorem 1 for estimating the Lipschitz bounds $\mathrm{Lip}(f)$ across different layers in the GNN $f(\cdot)$ requires unique explicit expressions for each component. This makes the process somewhat challenging. Nevertheless, this difficulty can be mitigated by computing the corresponding Jacobian matrices, as shown in Equation (6). By leveraging the values of each component in the Jacobian matrix, we can approximate the Lipschitz bounds in a straightforward manner. Additionally, the expression $\prod_{l=1}^{L} F^{l'} \left\| \left[ \mathcal{J}(h^l) \right]_j \right\|_\infty$ from Equation (6) provides valuable insights into the factors that influence the Lipschitz bounds, such as the dimensions of the output layer and the depth of the GNN layers.

However, considering the potential for multiple hierarchical layers in the network, their cumulative effect could lead to a significant deviation from the original bounds. Therefore, it is beneficial to consider the entire network as a single model and directly derive the Lipschitz bound from the input and output. In this subsection, we provide a simplified method to derive the Lipschitz bound in Equation (6) for facilitating fairness training. To achieve this, we then introduce the Jacobian matrix. Let $[J_{i}]$ denote the Jacobian matrix of the i-th node, which can be calculated as:

$$
\left[ \boldsymbol {J} _ {i} \right] _ {F ^ {\text {out}} \times F ^ {\text {in}}} = \left[ \begin{array}{c c c c} \frac {\partial \boldsymbol {Y} _ {i 1}}{\partial \boldsymbol {X} _ {i 1}} & \frac {\partial \boldsymbol {Y} _ {i 1}}{\partial \boldsymbol {X} _ {i 2}} & \dots & \frac {\partial \boldsymbol {Y} _ {i 1}}{\partial \boldsymbol {X} _ {i F ^ {\text {in}}}} \\ \frac {\partial \boldsymbol {Y} _ {i 2}}{\partial \boldsymbol {X} _ {i 1}} & \frac {\partial \boldsymbol {Y} _ {i 2}}{\partial \boldsymbol {X} _ {i 2}} & \dots & \frac {\partial \boldsymbol {Y} _ {i 2}}{\partial \boldsymbol {X} _ {i F ^ {\text {in}}}} \\ \vdots & \vdots & \vdots & \vdots \\ \frac {\partial \boldsymbol {Y} _ {i F ^ {\text {out}}}}{\partial \boldsymbol {X} _ {i 1}} & \frac {\partial \boldsymbol {Y} _ {i F ^ {\text {out}}}}{\partial \boldsymbol {X} _ {i 2}} & \dots & \frac {\partial \boldsymbol {Y} _ {i F ^ {\text {out}}}}{\partial \boldsymbol {X} _ {i F ^ {\text {in}}}} \end{array} \right] _ {F ^ {\text {out}} \times F ^ {\text {in}}}. \tag {7}
$$

We can define $\left[J_{i}\right]_{F^{\mathrm{out}}\times F^{\mathrm{in}}}=\left[J_{i1}^{\top},J_{i2}^{\top},\cdots,J_{iF^{\mathrm{out}}}^{\top}\right]^{\top}$ , where $J_{ij}=\left[\frac{\partial Y_{ij}}{\partial X_{i1}},\frac{\partial Y_{ij}}{\partial X_{i2}},\cdots,\frac{\partial Y_{ij}}{\partial X_{iF^{\mathrm{in}}}}\right]^{\top}$ , capturing the local interactions of node i, rather than relational dynamics within the entire graph. $^{2}$ Then we let $J_{i}=[\|J_{i1}\|,\|J_{i2}\|,\cdots,\|J_{iF^{\mathrm{out}}}\|]^{\top}$ , $J=\left[J_{1}^{\top},J_{2}^{\top},\cdots,J_{N}^{\top}\right]^{\top}$ . To analyze the Lipschitz bounds of the Jacobian matrix of output features for N nodes, we define $\mathrm{LB}(\mathcal{J})$ :

$$
\operatorname{LB} (\mathcal {J}) = \left[ \begin{array}{c} \mathcal {J} _ {1} ^ {\top} \\ \mathcal {J} _ {2} ^ {\top} \\ \vdots \\ \mathcal {J} _ {N} ^ {\top} \end{array} \right] = \left[ \begin{array}{c c c c} \mathcal {J} _ {1 1} & \mathcal {J} _ {1 2} & \dots & \mathcal {J} _ {1 F ^ {\text {out}}} \\ J _ {2 1} & \mathcal {J} _ {2 2} & \dots & \mathcal {J} _ {2 F ^ {\text {out}}} \\ \vdots & & & \\ \mathcal {J} _ {N 1} & \mathcal {J} _ {N 2} & \dots & \mathcal {J} _ {N F ^ {\text {out}}} \end{array} \right] _ {N \times F ^ {\text {out}}}, \tag {8}
$$

where $\mathcal{J}_{ij} = \| J_{ij}\|$ . Based on the definition of $\mathrm{LB}(\mathcal{J})$ , we can further establish the Lipschitz bound of the entire GNN model during training in the next subsection. To measure the scale of $\mathrm{LB}(\mathcal{J})$ of GNN $f(\cdot)$ , we define a $\mathrm{Lip}(f)$ that satisfies

$$
\mathrm{Lip} (f) = \| \mathrm{LB} (\mathcal {J}) \| _ {\infty , 2}. \tag {9}
$$

This calculation involves taking the $l_{2}$ -norm for each row of $\| \mathrm{LB}(\mathcal{J})\|_{\infty,2}$ and then taking the infinite norm for the entire $\| \mathrm{LB}(\mathcal{J})\|_{\infty,2}$ . Now we have proposed an easy solution as Equation (9) to approximate $\prod_{l=1}^{L} F^{l'} \left\| [\mathcal{J}(h^l)]_j \right\|_\infty$ for facilitating its feasible computation in practical training.

# 3.3 ILLUMINATING GNN FAIRNESS: A RANK-BASED PERSPECTIVE

Regularizing GNNs for input-output rank consistency, particularly in the context of rank-based individual fairness, is made possible by leveraging the Lipschitz bounds of GNNs. Specifically, we denote this Lipschitz bound as $\operatorname{Lip}(f) = \max_{j} \prod_{l=1}^{L} F^{l'} \left\| \left[ \mathcal{J}(h^l) \right]_j \right\|_\infty$ . We aim to mitigate inherent biases in the training data and to advance individual fairness. We accomplish this by enforcing Lipschitz constraints on the model's output, thereby guarding against biases that are incrementally learned from the input during the forward pass. This ensures consistency between the ranking lists based on the similarity matrix of each node in the input graph $S_G$ and the similarity matrix of the predicted outcome space $S_Y$ , as outlined in § 2.

Algorithm 1 JacoLip: Simplified PyTorch-style Pseudocode for Lipschitz Bounds in Fairness-Oriented GNN Training   
```python
# model: graph neural network model
# Train model for N epochs
for X, A, target in dataloader:
    pred = model(X, A)
    ce_loss = CrossEntropyLoss(pred, target)

# Compute Lipschitz bound for input
jacobian = Jaco(X, pred)
model_lip = Lip(jacobian) # Eq.(8)
cum_lip = norm(model_lip) # Eq.(9)

# Optimize model with Lipschitz bound
loss = ce_loss + u * cum_lip
loss.backward()
optimizer.step() 
```

We propose a plug-and-play solution, termed Ja-coLip, that integrates effortlessly with existing fairness-focused GNN training pipelines. Algorithm 1 provides a detailed PyTorch-style pseudocode for this approach. The method involves training the GNN model for a predefined number of epochs, while computing the Lipschitz bound for the model output using gradients and norms of the input features. This Lipschitz bound is then incorporated as a regularization term in the loss function to keep the model's output within constrained boundaries. Our approach efficiently mitigates individual biases and enhances the fairness of GNNs without incurring significant computational overhead. This “nearly-free” fairness regularization is facilitated by PyTorch’s built-in gradient functions, thereby achieving the benefits of Lipschitz regularization without substantial additional computational burden.

# 4 EXPERIMENTS

Here, we perform two major experiments to validate the Lipschitz bounds discussed in the previous section: (1) We use Lipschitz bounds to constrain the output consistency of GNNs, aiming to enhance rank-based individual fairness in node classification and link prediction tasks; (2) We examine the effects of Lipschitz bounds on the gradients and weights of GNNs, particularly concerning biases induced through training dynamics. Additional experiments can be found in Appendix C.

# 4.1 SETUP

Datasets We conduct experiments on six real-world datasets commonly used in prior work on rank-based individual fairness (Dong et al., 2021). These include one citation network (ACM (Tang et al., 2008)) and two co-authorship networks (Co-author-CS and Co-author-Phy (Shchur et al., 2018)) for node classification, and three social networks (BlogCatalog (Tang & Liu, 2009), Flickr (Huang et al., 2017), and Facebook (Leskovec & Mcauley, 2012)) for link prediction. We adhere to the public train/val/test splits from Dong et al. (2021). Dataset statistics and details are in Appendix E.

Backbones We use two widely-adopted GNN architectures for each downstream learning task: Graph Convolutional Network (GCN) (Kipf & Welling, 2017) and Simplifying Graph Convolutional Network (SGC) (Wu et al., 2019) for node classification, and GCN and Variational Graph Auto-Encoders (GAE) (Kipf & Welling, 2016) for link prediction. Model details are in Appendix D.

Baselines In contrast to prior research on group fairness in graph embeddings (e.g., (Bose & Hamilton, 2019; Rahman et al., 2019)), which often focuses on fairness for subgroups defined by specific protected attributes, our work centers on individual fairness (Dong et al., 2021) without reliance on such attributes. Therefore, methods for group fairness are beyond the scope of our comparison. To assess the efficacy of our approach in achieving individual fairness, we benchmark it against three significant baselines tailored specifically for rank-based individual fairness:

- Redress (Dong et al., 2021): This method proposes a rank-based framework to enhance the individual fairness of GNNs, integrating both utility maximization and fairness promotion into a joint end-to-end training framework.   
- InFoRM (Kang et al., 2020): Originally designed for conventional graph mining tasks such as PageRank and Spectral Clustering, InFoRM is based on the Lipschitz condition. We adapt it to GNNs by combining its individual fairness loss with the unity loss of the GNN backbone for end-to-end optimization.   
- PFR (Lahoti et al., 2019): PFR focuses on learning fair representations for individual fairness and outperforms traditional approaches in this regard. As PFR is more of a pre-processing strategy and not tailored for graph data, we apply it to the input node features.

Evaluation Metrics We employ two key metrics for evaluating rank-based individual fairness: classification accuracy (Acc.) for node classification tasks, and the area under the receiver operating characteristic curve (AUC) for link prediction tasks. For assessing individual fairness, we use the widely adopted ranking metric $NDCG@k$ (Järvelin & Kekäläinen, 2002), which measures the similarity between the rankings generated from $S_{Y}$ (result similarity matrix) and $S_{G}$ (oracle similarity matrix). Average $NDCG@k$ values are reported across all nodes, with k = 10.

Implementation Details The learning rate is set at 0.01 for all tasks. For models based on GCN and SGC, we use two layers with 16 hidden units each. For GAE-based models, we employ three graph convolutional layers, with the first two layers having 32 and 16 hidden units, respectively. Adam is used as the optimizer (Kingma & Ba, 2015). Further details, including code, dataset splits, and hyperparameter settings, are available in Appendix D. Our code has been released at https://github.com/chunhuizng/lipschitz-fairness.

# 4.2 EFFECT OF LIPSCHITZ BOUNDS ON RANK-BASED INDIVIDUAL FAIRNESS IN GRAPHS

The experiments validate the efficacy of Lipschitz bounds for enhancing individual fairness in GNNs, specifically from a ranking perspective. Results are consolidated in Tables 1 and 2 for the node classification and link prediction tasks, respectively.

In Table 1, our method, JacoLip, yields promising results, outperforming the baselines in achieving a superior trade-off between accuracy and fairness: (i) When applied to Vanilla models such as GCN or SGC, JacoLip enhances fairness performance (measured by NDCG@10) while maintaining comparable accuracy. This result substantiates that the plug-and-play nature of Lipschitz bound regularization effectively mitigates bias during the training of standard GNN backbones, thereby fostering a balanced trade-off between accuracy and fairness; (ii) Additionally, when integrated with the existing fairness-centric algorithm Redress, JacoLip further constrains irrelevant biased factors during training and marginally improves the trade-off between accuracy and fairness across all datasets and backbones.

Table 1: Evaluation on node classification task: comparing under accuracy and NDCG. Higher performance in both metrics indicates a better trade-off. Results are in percentages, and averaged values and standard deviations are computed from five runs. The improvement is within brackets. We bold the best result and underline the runner-up. 

<table><tr><td rowspan="2">Data</td><td rowspan="2">Model</td><td rowspan="2">Fair Alg.</td><td colspan="2">Feature Similarity</td><td colspan="2">Structural Similarity</td></tr><tr><td>utility: Acc.↑</td><td>fairness: NDCG@10↑</td><td>utility: Acc.↑</td><td>fairness: NDCG@10↑</td></tr><tr><td rowspan="12">ACM</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2017)</td><td> $72.49 \pm 0.6$ </td><td> $47.33 \pm 1.0$ </td><td> $72.49 \pm 0.6$ </td><td> $25.42 \pm 0.6$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $68.03 \pm 0.3(-6.15\%)$ </td><td> $39.79 \pm 0.3(-15.9\%)$ </td><td> $69.13 \pm 0.5(-4.64\%)$ </td><td> $12.02 \pm 0.4(-52.7\%)$ </td></tr><tr><td>PFR (Laboti et al., 2019)</td><td> $67.88 \pm 1.1(-6.36\%)$ </td><td> $31.20 \pm 0.2(-34.1\%)$ </td><td> $69.00 \pm 0.7(-4.81\%)$ </td><td> $23.85 \pm 1.3(-6.18\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $71.75 \pm 0.4(-1.02\%)$ </td><td> $49.13 \pm 0.4(+3.80\%)$ </td><td> $72.03 \pm 0.9(-0.63\%)$ </td><td> $29.09 \pm 0.4(+14.4\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $72.37 \pm 0.3(-0.16\%)$ </td><td> $49.80 \pm 0.3(+5.26\%)$ </td><td> $71.97 \pm 0.3(-0.71\%)$ </td><td> $27.91 \pm 0.7(+9.79\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $71.92 \pm 0.2(-0.78\%)$ </td><td> $53.62 \pm 0.6(+13.3\%)$ </td><td> $72.05 \pm 0.5(-0.60\%)$ </td><td> $31.80 \pm 0.4(+25.1\%)$ </td></tr><tr><td rowspan="6">SGC</td><td>Vanilla (Wu et al., 2019)</td><td> $68.40 \pm 1.0$ </td><td> $55.75 \pm 1.1$ </td><td> $68.40 \pm 1.0$ </td><td> $37.18 \pm 0.6$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $68.81 \pm 0.5(+0.60\%)$ </td><td> $48.25 \pm 0.5(-13.5\%)$ </td><td> $66.71 \pm 0.6(-2.47\%)$ </td><td> $28.33 \pm 0.6(-23.8\%)$ </td></tr><tr><td>PFR (Laboti et al., 2019)</td><td> $67.97 \pm 0.7(-0.62\%)$ </td><td> $34.71 \pm 0.1(-37.7\%)$ </td><td> $67.78 \pm 0.1(-0.91\%)$ </td><td> $37.15 \pm 0.6(-0.08\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $67.16 \pm 0.2(-1.81\%)$ </td><td> $58.64 \pm 0.4(+5.18\%)$ </td><td> $67.77 \pm 0.4(-0.92\%)$ </td><td> $38.95 \pm 0.1(+4.76\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $73.84 \pm 0.2(+7.95\%)$ </td><td> $62.00 \pm 0.2(+11.21\%)$ </td><td> $69.28 \pm 0.3(+1.29\%)$ </td><td> $38.36 \pm 0.4(+3.17\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $72.36 \pm 0.4(+5.79\%)$ </td><td> $69.22 \pm 0.5(+24.16\%)$ </td><td> $72.52 \pm 0.5(+6.02\%)$ </td><td> $41.07 \pm 0.3(+10.5\%)$ </td></tr><tr><td rowspan="12">CS</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2017)</td><td> $90.59 \pm 0.3$ </td><td> $50.84 \pm 1.2$ </td><td> $90.59 \pm 0.3$ </td><td> $18.29 \pm 0.8$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $88.66 \pm 1.1(-2.13\%)$ </td><td> $53.38 \pm 1.6(+5.00\%)$ </td><td> $87.55 \pm 0.9(-3.36\%)$ </td><td> $19.18 \pm 0.9(+4.87\%)$ </td></tr><tr><td>PFR (Laboti et al., 2019)</td><td> $87.51 \pm 0.7(-3.40\%)$ </td><td> $37.12 \pm 0.9(-27.0\%)$ </td><td> $86.16 \pm 0.2(-4.89\%)$ </td><td> $11.98 \pm 1.3(-34.5\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $90.70 \pm 0.2(+0.12\%)$ </td><td> $55.01 \pm 1.9(+8.20\%)$ </td><td> $89.16 \pm 0.3(-1.58\%)$ </td><td> $21.28 \pm 0.3(+16.4\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $90.68 \pm 0.3(+0.90\%)$ </td><td> $55.35 \pm 0.2(+8.87\%)$ </td><td> $89.23 \pm 0.5(-1.50\%)$ </td><td> $21.82 \pm 0.2(+19.3\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $90.63 \pm 0.3(+0.40\%)$ </td><td> $68.20 \pm 0.4(+34.2\%)$ </td><td> $89.21 \pm 0.1(-1.52\%)$ </td><td> $31.82 \pm 0.4(+74.1\%)$ </td></tr><tr><td rowspan="6">SGC</td><td>Vanilla (Wu et al., 2019)</td><td> $87.48 \pm 0.8$ </td><td> $74.00 \pm 0.1$ </td><td> $87.48 \pm 0.8$ </td><td> $32.36 \pm 0.3$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $88.07 \pm 0.1(+0.67\%)$ </td><td> $74.29 \pm 0.1(+0.39\%)$ </td><td> $88.65 \pm 0.4(+1.34\%)$ </td><td> $32.37 \pm 0.4(+0.03\%)$ </td></tr><tr><td>PFR (Laboti et al., 2019)</td><td> $88.31 \pm 0.1(+0.94\%)$ </td><td> $48.40 \pm 0.1(-34.6\%)$ </td><td> $84.34 \pm 0.3(-3.59\%)$ </td><td> $28.87 \pm 0.9(-10.8\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $90.01 \pm 0.2(+2.89\%)$ </td><td> $76.60 \pm 0.1(+3.51\%)$ </td><td> $89.35 \pm 0.1(+2.14\%)$ </td><td> $34.24 \pm 0.2(+5.81\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $90.23 \pm 0.2(+3.14\%)$ </td><td> $74.63 \pm 0.2(+0.85\%)$ </td><td> $89.53 \pm 0.6(+2.34\%)$ </td><td> $32.83 \pm 0.3(+1.45\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $90.12 \pm 0.3(+3.02\%)$ </td><td> $77.01 \pm 0.1(+4.07\%)$ </td><td> $89.80 \pm 0.2(+2.65\%)$ </td><td> $34.89 \pm 0.5(+7.82\%)$ </td></tr><tr><td rowspan="12">Phy</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2017)</td><td> $94.81 \pm 0.2$ </td><td> $34.83 \pm 1.1$ </td><td> $94.81 \pm 0.2$ </td><td> $1.57 \pm 0.1$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $89.33 \pm 0.8(-5.78\%)$ </td><td> $31.25 \pm 0.0(-10.3\%)$ </td><td> $94.46 \pm 0.2(-0.37\%)$ </td><td> $1.77 \pm 0.0(+12.7\%)$ </td></tr><tr><td>PFR (Laboti et al., 2019)</td><td> $89.74 \pm 0.5(-5.35\%)$ </td><td> $24.16 \pm 0.4(-30.6\%)$ </td><td> $87.26 \pm 0.2(-7.96\%)$ </td><td> $1.20 \pm 0.1(-23.6\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $94.63 \pm 0.7(-0.19\%)$ </td><td> $43.64 \pm 0.5(+25.3\%)$ </td><td> $93.94 \pm 0.3(-0.92\%)$ </td><td> $1.93 \pm 0.1(+22.9\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $94.60 \pm 0.2(-0.22\%)$ </td><td> $37.33 \pm 0.5(+7.18\%)$ </td><td> $93.99 \pm 0.4(-0.86\%)$ </td><td> $1.87 \pm 0.2(+19.1\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $94.50 \pm 0.2(-0.32\%)$ </td><td> $49.37 \pm 0.3(+41.75\%)$ </td><td> $93.86 \pm 0.9(-1.00\%)$ </td><td> $2.72 \pm 0.1(+73.3\%)$ </td></tr><tr><td rowspan="6">SGC</td><td>Vanilla (Wu et al., 2019)</td><td> $94.45 \pm 0.2$ </td><td> $49.63 \pm 0.1$ </td><td> $94.45 \pm 0.2$ </td><td> $3.61 \pm 0.1$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $92.01 \pm 0.1(-2.58\%)$ </td><td> $43.87 \pm 0.2(-11.6\%)$ </td><td> $94.27 \pm 0.3(-0.19\%)$ </td><td> $3.64 \pm 0.0(+0.83\%)$ </td></tr><tr><td>PFR (Laboti et al., 2019)</td><td> $89.74 \pm 0.3(-4.99\%)$ </td><td> $28.54 \pm 0.1(-42.5\%)$ </td><td> $89.73 \pm 0.3(-5.00\%)$ </td><td> $2.62 \pm 0.1(-27.4\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $94.30 \pm 0.1(-0.16\%)$ </td><td> $53.40 \pm 0.1(+7.60\%)$ </td><td> $93.94 \pm 0.2(-0.54\%)$ </td><td> $4.03 \pm 0.0(+11.6\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $94.20 \pm 0.3(-0.26\%)$ </td><td> $50.70 \pm 0.4(+2.16\%)$ </td><td> $93.58 \pm 0.2(-0.92\%)$ </td><td> $3.80 \pm 0.6(+5.26\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $93.28 \pm 0.1(-1.24\%)$ </td><td> $59.20 \pm 0.6(+19.3\%)$ </td><td> $93.99 \pm 1.1(-0.49\%)$ </td><td> $4.30 \pm 0.5(+19.1\%)$ </td></tr></table>

Similar trends are observed for the link prediction task, as seen in Table 2. Here, the performance of Vanilla models (GCN or GAE), InFoRM, PFR, Redress, and JacoLip is evaluated based on AUC for utility and NDCG@10 for fairness. Across both tasks, JacoLip consistently shows either competitive or superior performance when compared to the baselines. These improvements are achieved with minimal computational overhead, as JacoLip efficiently calculates Lipschitz bounds using PyTorch's built-in gradient functionality, further detailed in § 3.3 and Table 5.

# 4.3 IMPACT OF LIPSCHITZ BOUNDS ON TRAINING DYNAMICS

We further analyze the impact of Lipschitz bounds through the optimization process and explore its dynamical interactions with weight parameters, gradient, fairness, and accuracy in Figure 1:

For the nonlinear GCN model, our proposed method, JacoLip, imposes beneficial regularizations on gradients. Specifically, during the initial epochs, JacoLip effectively stabilizes gradient magnitudes, resulting in higher accuracy (e.g., approximately 20% improvement in feature similarity and around 40% in structural similarity) compared to the Redress baseline. Concurrently, JacoLip maintains superior fairness metrics (NDCG), exceeding the baseline Redress; For the linear SGC model, JacoLip similarly achieves a favorable trade-off between fairness and accuracy relative to the Redress baseline. Initially, JacoLip on SGC shows better fairness (e.g., approximately 0.2 NDCG on feature similarity) than the Redress baseline, with only a minor decrease in accuracy. As training progresses, the accuracy of JacoLip on SGC tends to match or even surpass that of Redress.

In summary, the observed trade-off between accuracy and fairness during the training phase can be attributed to the constraining effect of Lipschitz bounds on gradient optimization. The higher expressivity of the nonlinear GCN model renders it more susceptible to loss of consistency in input-output similarity ranking, crucial for fairness. Conversely, the simpler linear model more readily preserves this consistency, and here, JacoLip prioritizes accuracy. These insights elucidate the

Table 2: Evaluation on link prediction tasks: comparing under AUC and NDCG. 

<table><tr><td rowspan="2">Data</td><td rowspan="2">Model</td><td rowspan="2">Fair Alg.</td><td colspan="2">Feature Similarity</td><td colspan="2">Structural Similarity</td></tr><tr><td>utility: AUC↑</td><td>fairness: NDCG@10↑</td><td>utility: AUC↑</td><td>fairness: NDCG@10↑</td></tr><tr><td rowspan="12">Blog</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2017)</td><td>85.87±0.1</td><td>16.73±0.1</td><td>85.87±0.1</td><td>32.47±0.5</td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td>79.85±0.6(-7.01%)</td><td>15.57±0.2(-6.93%)</td><td>84.00±0.1(-2.18%)</td><td>26.18±0.3(-19.4%)</td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td>84.25±0.2(-1.89%)</td><td>16.37±0.0(-2.15%)</td><td>83.88±0.0(-2.32%)</td><td>29.60±0.4(-8.84%)</td></tr><tr><td>Redress (Dong et al., 2021)</td><td>86.49±0.8(+0.72%)</td><td>17.66±0.2(+5.56%)</td><td>86.25±0.3(+0.44%)</td><td>34.62±0.7(+6.62%)</td></tr><tr><td>JacoLip (on Vanilla)</td><td>86.51±0.2(+0.74%)</td><td>17.70±0.6(+5.79%)</td><td>86.90±0.5(+1.67%)</td><td>35.00±0.4(+7.79%)</td></tr><tr><td>JacoLip (on Redress)</td><td>85.91±0.2(+0.04%)</td><td>18.02±0.6(+7.71%)</td><td>86.84±0.5(+1.13%)</td><td>35.85±0.4(+10.4%)</td></tr><tr><td rowspan="6">GAE</td><td>Vanilla (Kipf &amp; Welling, 2016)</td><td>85.72±0.1</td><td>17.13±0.1</td><td>85.72±0.1</td><td>41.99±0.4</td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td>80.01±0.2(-6.66%)</td><td>16.12±0.2(-5.90%)</td><td>82.86±0.0(-3.34%)</td><td>27.29±0.3(-35.0%)</td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td>83.83±0.1(-2.20%)</td><td>16.64±0.0(-2.86%)</td><td>83.87±0.1(-2.16%)</td><td>35.91±0.4(-14.5%)</td></tr><tr><td>Redress (Dong et al., 2021)</td><td>84.67±0.9(-1.22%)</td><td>18.19±0.1(+6.19%)</td><td>86.36±1.5(+0.75%)</td><td>43.51±0.7(+3.62%)</td></tr><tr><td>JacoLip (on Vanilla)</td><td>85.75±0.4(+0.03%)</td><td>17.96±0.5(+4.85%)</td><td>85.86±0.5(+0.16%)</td><td>42.20±0.3(+0.50%)</td></tr><tr><td>JacoLip (on Redress)</td><td>85.70±0.4(-0.02%)</td><td>18.34±0.5(+7.06%)</td><td>86.31±0.5(+0.69%)</td><td>43.60±0.3(+3.83%)</td></tr><tr><td rowspan="12">Flickr</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2016)</td><td>92.20±0.3</td><td>13.10±0.2</td><td>92.20±0.3</td><td>22.35±0.3</td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td>91.39±0.0(-0.88%)</td><td>11.95±0.1(-8.78%)</td><td>91.73±0.1(-0.51%)</td><td>23.28±0.6(+4.16%)</td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td>91.91±0.1(-0.31%)</td><td>12.94±0.0(-1.22%)</td><td>91.86±0.2(-0.37%)</td><td>19.80±0.4(-11.4%)</td></tr><tr><td>Redress (Dong et al., 2021)</td><td>91.38±0.1(-0.89%)</td><td>13.58±0.3(+3.66%)</td><td>92.67±0.2(+0.51%)</td><td>28.45±0.5(+27.3%)</td></tr><tr><td>JacoLip (on Vanilla)</td><td>92.75±0.3(+0.59%)</td><td>13.74±0.4(+4.89%)</td><td>92.54±0.1(+0.37%)</td><td>26.61±0.4(+19.1%)</td></tr><tr><td>JacoLip (on Redress)</td><td>92.53±0.3(+0.35%)</td><td>14.37±0.4(+9.69%)</td><td>92.69±0.1(+0.53%)</td><td>28.65±0.4(+28.2%)</td></tr><tr><td rowspan="6">GAE</td><td>Vanilla (Kipf &amp; Welling, 2016)</td><td>89.98±0.1</td><td>12.77±0.0</td><td>89.98±0.1</td><td>23.58±0.2</td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td>88.76±0.7(-1.36%)</td><td>12.07±0.1(-5.48%)</td><td>91.51±0.2(+1.70%)</td><td>15.78±0.3(-33.1%)</td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td>90.30±0.1(+0.36%)</td><td>12.12±0.1(-5.09%)</td><td>90.10±0.1(+1.33%)</td><td>20.46±0.3(-13.2%)</td></tr><tr><td>Redress (Dong et al., 2021)</td><td>89.45±0.5(-0.59%)</td><td>14.24±0.1(+11.5%)</td><td>89.52±0.3(-0.51%)</td><td>29.83±0.2(+26.5%)</td></tr><tr><td>JacoLip (on Vanilla)</td><td>89.88±0.3(-0.11%)</td><td>14.37±0.1(+12.53%)</td><td>89.95±0.2(-0.03%)</td><td>28.74±0.5(+21.9%)</td></tr><tr><td>JacoLip (on Redress)</td><td>89.92±0.3(-0.06%)</td><td>14.85±0.1(+16.29%)</td><td>89.56±0.2(-0.46%)</td><td>30.04±0.5(+28.7%)</td></tr><tr><td rowspan="12">Facebook</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2017)</td><td>95.60±1.7</td><td>23.07±0.2</td><td>95.60±1.7</td><td>16.55±1.1</td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td>90.26±0.1(-5.59%)</td><td>23.23±0.3(+0.69%)</td><td>96.66±0.6(+1.11%)</td><td>15.18±0.7(-8.28%)</td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td>87.11±1.2(-8.88%)</td><td>21.83±0.2(-5.37%)</td><td>94.87±1.9(-0.76%)</td><td>19.53±0.5(+18.0%)</td></tr><tr><td>Redress (Dong et al., 2021)</td><td>96.49±1.6(+0.93%)</td><td>29.60±0.1(+28.3%)</td><td>92.66±0.4(-3.08%)</td><td>27.73±1.1(+67.5%)</td></tr><tr><td>JacoLip (on Vanilla)</td><td>96.21±0.2(+0.63%)</td><td>29.47±0.3(+27.7%)</td><td>95.46±0.9(-0.14%)</td><td>26.60±0.1(+60.7%)</td></tr><tr><td>JacoLip (on Redress)</td><td>96.11±0.2(+0.53%)</td><td>30.07±0.3(+30.3%)</td><td>92.76±0.9(-2.97%)</td><td>28.64±0.1(+73.1%)</td></tr><tr><td rowspan="6">GAE</td><td>Vanilla (Kipf &amp; Welling, 2016)</td><td>98.54±0.0</td><td>26.75±0.1</td><td>98.54±0.0</td><td>27.03±0.1</td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td>90.50±0.4(-8.16%)</td><td>22.77±0.2(-14.9%)</td><td>95.03±0.1(-3.56%)</td><td>15.38±0.2(-43.1%)</td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td>96.91±0.1(-1.65%)</td><td>23.52±0.1(-12.1%)</td><td>98.28±0.0(-0.26%)</td><td>22.89±0.3(-15.3%)</td></tr><tr><td>Redress (Dong et al., 2021)</td><td>95.98±1.5(-2.60%)</td><td>28.43±0.3(+6.28%)</td><td>94.07±1.7(-4.54%)</td><td>33.53±0.2(+24.0%)</td></tr><tr><td>JacoLip (on Vanilla)</td><td>97.40±0.1(-1.16%)</td><td>27.44±0.6(+2.58%)</td><td>97.02±1.1(-1.54%)</td><td>30.90±0.5(+14.3%)</td></tr><tr><td>JacoLip (on Redress)</td><td>96.10±0.1(-2.48%)</td><td>28.46±0.6(+6.39%)</td><td>94.22±1.1(-4.38%)</td><td>31.62±0.5(+17.1%)</td></tr></table>

dynamic behavior of model training under Lipschitz bound constraints, illustrating how JacoLip can enhance fairness while retaining competitive accuracy.

# 5 RELATED WORKS

Lipschitz Bounds in Deep Models. Prior research on Lipschitz bounds has primarily focused on specific types of neural networks incorporating convolutional or attention layers (Zou et al., 2019; Terris et al., 2020; Kim et al., 2021; Araujo et al., 2021). In the context of GNNs, (Dasoulas et al., 2021) introduced a Lipschitz normalization method for self-attention layers in GATs. More recently, (Gama & Sojoudi, 2022) estimated the filter Lipschitz bound using the infinite norm of a matrix. In contrast, the Lipschitz matrix in our study follows a distinct definition and employs different choices of norm types. Additionally, our objective is to enhance the stability of GNNs against unfair biases, which is not clearly addressed in the aforementioned works. We refer to Appendix A for vital details.

Fair Graph Learning. Fair graph learning is a relatively open field (Wu et al., 2021; Dai & Wang, 2021; Buyl & De Bie, 2020). Some existing approaches address fairness concerns through fairness-aware augmentations or adversarial training. For instance, Fairwalk (Rahman et al., 2019) is a random walk-based algorithm that aims to mitigate fairness issues in graph node embeddings. Adversarial training is employed in approaches like Compositional Fairness (Bose & Hamilton, 2019) to disentangle learned embeddings from sensitive features. Information Regularization (Liao et al., 2021) utilizes adversarial training to minimize the marginal difference between vertex representations. In addition, (Palowitch & Perozzi, 2020) improves group fairness by ensuring that node embeddings lie on a hyperplane orthogonal to sensitive features. However, there remains ample room for further exploration in rank-based individual fairness (Dong et al., 2021), which is the focus of our work.

Understanding Learning on Graphs. Various approaches have emerged to understand the underlying patterns in the graph data and its components. Explanatory models for relational/graph learning (Ying et al., 2019; Huang et al., 2022; Yuan et al., 2021; Chen et al., 2023) provide insights into the relationship between a model's predictions and elements in graphs. These works shed light on how

![](images/9db5380c1afd3bd6886bcdbef3bcdaaf9595601d8d2ef1ba59c0e3636778f23e.jpg)  
Figure 1: Study of the Lipschitz bounds' impact on model training for rank-based individual fairness. We perform experiments on the co-author-Physics dataset using both nonlinear (GCN) and linear (SGC) models. The training dynamics are assessed by monitoring the NDCG, accuracy, weight norm, and weight gradient as the number of epochs increases. Upper two rows: Metrics under feature similarity; Lower two rows: Metrics under structural similarity.

local elements or node characteristics influence the decision-making process of GNNs. However, our work differs in that we investigate the impact of Lipschitz bounds on practical training dynamics, rather than focusing on trained/fixed-parameter model inference or the influence of local features on GNN decision-making processes.

# 6 CONCLUSIONS

We have investigated the use of Lipschitz bounds for promoting individual fairness in GNNs from a ranking perspective. We conducted a thorough analysis of the theoretical properties of Lipschitz bounds and their relation to rank-based individual fairness. Building on this analysis, we introduced JacoLip, a fairness solution that incorporates Lipschitz bound regularization into the GNN training process. To assess the efficacy of JacoLip, we conducted extensive experiments on real-world datasets for both node classification and link prediction tasks. The results consistently demonstrate that JacoLip effectively constrains bias during training, thereby enhancing fairness performance while preserving accuracy.

# ETHICS STATEMENT

Our research on Lipschitz bounds for GNNs carries societal implications. On the positive side, our work enhances the stability and interpretability of GNNs, contributing to safer AI systems across various domains. Improved fairness and interpretability can help mitigate biases and ensure more equitable decision-making outcomes. On the other hand, the advancement of GNNs also brings forth concerns about potential misuse and unintended consequences. Like any deep learning technology, GNNs have the potential to unfairly impact certain groups or perpetuate existing biases.

We underscore the importance of fair training procedures, stringent privacy safeguards, and responsible deployment and monitoring of GNNs. In this manner, our work serves as a foundational contribution to GNN research, emphasizing the need to consider broader impacts and potential harms while implementing suitable mitigation strategies.

# REFERENCES

Chirag Agarwal, Himabindu Lakkaraju, and Marinka Zitnik. Towards a unified framework for fair and stable graph representation learning. In Uncertainty in Artificial Intelligence, 2021. 2, 15, 20, 22   
Alexandre Araujo, Benjamin Negrevergne, Yann Chevaleyre, and Jamal Atif. On Lipschitz regularization of convolutional layers using toeplitz matrix theory. In AAAI Conference on Artificial Intelligence, 2021. 8, 15   
Avishek Bose and William Hamilton. Compositional fairness constraints for graph embeddings. In International Conference on Machine Learning, 2019. 6, 8   
Maarten Buyl and Tijl De Bie. Debayes: a bayesian method for debiasing network embeddings. In International Conference on Machine Learning, 2020. 8   
Zizhang Chen, Peizhao Li, Hongfu Liu, and Pengyu Hong. Characterizing the influence of graph elements. In International Conference on Learning Representations, 2023. 8   
Enyan Dai and Suhang Wang. Say no to the discrimination: Learning fair graph neural networks with limited sensitive attribute information. In ACM International Conference on Web Search and Data Mining, 2021. 8   
George Dasoulas, Kevin Scaman, and Aladin Virmaux. Lipschitz normalization for self-attention layers with application to graph neural networks. In International Conference on Machine Learning, 2021. 8, 15   
Yushun Dong, Jian Kang, Hanghang Tong, and Jundong Li. Individual fairness for graph neural networks: A ranking based approach. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2021. 1, 2, 3, 6, 7, 8, 19, 20, 21, 24, 25   
Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Innovations in Theoretical Computer Science Conference, 2012. 2, 15, 20, 22   
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. Graph neural networks for social recommendation. In International World Wide Web Conference, 2019. 1   
Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George Pappas. Efficient and accurate estimation of lipschitz constants for deep neural networks. In Advances in Neural Information Processing Systems, 2019. 2, 15   
Fernando Gama and Somayeh Sojoudi. Graph neural networks for distributed linear-quadratic control. In Learning for Dynamics and Control, pp. 111–124. PMLR, 2021. 15   
Fernando Gama and Somayeh Sojoudi. Distributed linear-quadratic control with graph neural networks. Signal Processing, 2022. 8   
M. Gori, G. Monfardini, and F. Scarselli. A new model for learning in graph domains. In IEEE International Joint Conference on Neural Networks, 2005. 1

Zhichun Guo, Chunhui Zhang, Yujie Fan, Yijun Tian, Chuxu Zhang, and Nitesh V. Chawla. Boosting graph neural networks via adaptive knowledge distillation. In AAAI Conference on Artificial Intelligence, 2023. 16   
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, 2017. 1   
Chao Huang, Jiahui Chen, Lianghao Xia, Yong Xu, Peng Dai, Yanqing Chen, Liefeng Bo, Jiashu Zhao, and Jimmy Xiangji Huang. Graph-enhanced multi-task learning of multi-level transition dynamics for session-based recommendation. In AAAI Conference on Artificial Intelligence, 2021a.1   
Qiang Huang, Makoto Yamada, Yuan Tian, Dinesh Singh, and Yi Chang. Graphlime: Local interpretable model explanations for graph neural networks. IEEE Transactions on Knowledge and Data Engineering, 2022. 8   
Xiao Huang, Jundong Li, and Xia Hu. Label informed attributed network embedding. In ACM International Conference on Web Search and Data Mining, 2017. 6   
Yujia Huang, Huan Zhang, Yuanyuan Shi, J Zico Kolter, and Anima Anandkumar. Training certifiably robust neural networks with efficient local lipschitz bounds. In Advances in Neural Information Processing Systems, 2021b. 2, 15   
Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems, 2002. 6   
Yaning Jia, Dongmian Zou, Hongfei Wang, and Hai Jin. Enhancing node-level adversarial defenses by lipschitz regularization of graph neural networks. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023. 23   
Matt Jordan and Alexandros G Dimakis. Exactly computing the local lipschitz constant of relu networks. In Advances in Neural Information Processing Systems, 2020. 2, 15   
Jian Kang, Jingrui He, Ross Maciejewski, and Hanghang Tong. Inform: Individual fairness on graph mining. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2020. 1, 2, 6, 7, 8, 20, 21   
Hyunjik Kim, George Papamakarios, and Andriy Mnih. The Lipschitz constant of self-attention. In International Conference on Machine Learning, 2021. 8, 15   
Diederick P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. 6   
Thomas N Kipf and Max Welling. Variational graph auto-encoders. In NIPS Workshop on Bayesian Deep Learning, 2016. 6, 8   
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. 1, 6, 7, 8, 21   
Preethi Lahoti, Krishna P. Gummadi, and Gerhard Weikum. Operationalizing individual fairness with pairwise fair representations. In VLDB Endowment, 2019. 1, 6, 7, 8, 21   
Fabian Latorre, Paul Rolland, and Volkan Cevher. Lipschitz constant estimation of neural networks via sparse polynomial optimization. In International Conference on Learning Representations, 2020. 2, 15   
Jure Leskovec and Julian Mcauley. Learning to discover social circles in ego networks. In Advances in Neural Information Processing Systems, 2012. 6   
Jiazheng Li, Chunhui Zhang, and Chuxu Zhang. Heterogeneous temporal graph neural network explainer. In ACM International Conference on Information and Knowledge Management, 2023. 16   
Junying Li, Deng Cai, and Xiaofei He. Learning graph-level representation for drug discovery. arXiv preprint arXiv:1709.03741, 2017. 1

Peizhao Li, Yifei Wang, Han Zhao, Pengyu Hong, and Hongfu Liu. On dyadic fairness: Exploring and mitigating bias in graph connections. In International Conference on Learning Representations, 2021. 1, 2   
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. International Conference on Learning Representations, 2016. 1   
Peiyuan Liao, Han Zhao, Keyulu Xu, Tommi Jaakkola, Geoffrey J Gordon, Stefanie Jegelka, and Ruslan Salakhutdinov. Information obfuscation of graph neural networks. In International Conference on Machine Learning, 2021. 2, 8   
Zheyuan Liu, Chunhui Zhang, Yijun Tian, Erchi Zhang, Chao Huang, Yanfang Ye, and Chuxu Zhang. Fair graph representation learning via diverse mixture-of-experts. In The ACM Web Conference, 2023. 1, 16   
Zheyuan Liu, Guangyao Dou, Yijun Tian, Chunhui Zhang, Eli Chien, and Ziwei Zhu. Breaking the trilemma of privacy, utility, and efficiency via controllable machine unlearning. In The ACM Web Conference, 2024. 16   
Felix Mujkanovic, Simon Geisler, Stephan Günnemann, and Aleksandar Bojchevski. Are defenses for graph neural networks robust? In Advances in Neural Information Processing Systems, 2022. 1   
Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In International Conference on Machine Learning, 2010. 4   
Zhongyu Ouyang, Shifu Hou, Shang Ma, Chaoran Chen, Chunhui Zhang, Toby Li, Xusheng Xiao, Chuxu Zhang, and Yanfang Ye. Prompt learning unlocked for app promotion in the wild. In NeurIPS 2023 Workshop: New Frontiers in Graph Learning, 2023. 16   
Zhongyu Ouyang, Chunhui Zhang, Shifu Hou, Chuxu Zhang, and Yanfang Ye. How to improve representation alignment and uniformity in graph-based collaborative filtering? In International AAAI Conference on Web and Social Media, 2024. 1   
John Joseph Palowitch and Bryan Perozzi. Debiasing graph embeddings with metadata-orthogonal training. In Advances in Social Network Analysis and Mining, 2020. 8   
Yiyue Qian, Chunhui Zhang, Yiming Zhang, Qianlong Wen, Yanfang Ye, and Chuxu Zhang. Comodality graph contrastive learning for imbalanced node classification. In Advances in Neural Information Processing Systems, 2022. 16   
Tahleen Rahman, Bartlomiej Surma, Michael Backes, and Yang Zhang. Fairwalk: Towards fair graph embedding. In International Joint Conference on Artificial Intelligence, 2019. 6, 8   
Mariia Rizun. Knowledge graph application in education: a literature review. Acta Universitatis Lodziensis. Folia Oeconomica, 2019. 1   
F. Scarselli, S. L. Yong, M. Gori, M. Hagenbuchner, A. C. Tsoi, and M. Maggini. Graph neural networks for ranking web pages. In International Conference on Web Intelligence, 2005. 1   
Walid Shalaby, BahaaEddin AlAila, Mohammed Korayem, Layla Pournajaf, Khalifeh AlJadda, Shannon Quinn, and Wlodek Zadrozny. Help me find a job: A graph-based approach for job recommendation at scale. In IEEE International Conference on Big Data, 2017. 1   
Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Günnemann. Pitfalls of graph neural network evaluation. In Relational Representation Learning Workshop, NeurIPS, 2018. 6   
Zhouxing Shi, Yihan Wang, Huan Zhang, J Zico Kolter, and Cho-Jui Hsieh. Efficiently computing local lipschitz constants of neural networks via bound propagation. In Advances in Neural Information Processing Systems, 2022. 2, 15, 20, 21, 22, 23   
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014. 15

Ichigaku Takigawa and Hiroshi Mamitsuka. Graph mining: procedure, application to drug discovery and recent advances. Drug Discovery Today, 2013. 1   
Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. Arnetminer: extraction and mining of academic social networks. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2008. 6   
Lei Tang and Huan Liu. Relational learning via latent social dimensions. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2009. 6   
Matthieu Terris, Audrey Repetti, Jean-Christophe Pesquet, and Yves Wiaux. Building firmly nonexpansive convolutional neural networks. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2020. 8, 15   
Yijun Tian, Kaiwen Dong, Chunhui Zhang, Chuxu Zhang, and Nitesh V. Chawla. Heterogeneous graph masked autoencoders. In AAAI Conference on Artificial Intelligence, 2023a. 1   
Yijun Tian, Chuxu Zhang, Zhichun Guo, Xiangliang Zhang, and Nitesh Chawla. Learning MLPs on graphs: A unified view of effectiveness, robustness, and efficiency. In International Conference on Learning Representations, 2023b. 16   
Aladin Virmaux and Kevin Scaman. Lipschitz regularity of deep neural networks: analysis and efficient estimation. In Advances in Neural Information Processing Systems, 2018. 2, 3, 15   
Soroush Vosoughi, Prashanth Vijayaraghavan, and Deb Roy. Tweet2vec: Learning tweet embeddings using character-level cnn-lstm encoder-decoder. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval, 2016. 1   
Soroush Vosoughi, Mostafa ‘Neo’ Mohsenvand, and Deb Roy. Rumor gauge: Predicting the veracity of rumors on twitter. ACM Transactions on Knowledge Discovery from Data, 2017. 1   
Soroush Vosoughi, Deb Roy, and Sinan Aral. The spread of true and false news online. Science, 2018. 1   
Ruijie Wang, Yuchen Yan, Jialu Wang, Yuting Jia, Ye Zhang, Weinan Zhang, and Xinbing Wang. Acekg: A large-scale knowledge graph for academic data mining. In ACM International Conference on Information and Knowledge Management, 2018. 1   
Qianlong Wen, Zhongyu Ouyang, Chunhui Zhang, Yiyue Qian, Yanfang Ye, and Chuxu Zhang. Graph contrastive learning with cross-view reconstruction. In NeurIPS 2022 Workshop: New Frontiers in Graph Learning, 2022a. 16   
Qianlong Wen, Zhongyu Ouyang, Jianfei Zhang, Yiyue Qian, Yanfang Ye, and Chuxu Zhang. Disentangled dynamic heterogeneous graph learning for opioid overdose prediction. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022b. 16   
Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph convolutional networks. In International Conference on Machine Learning, 2019. 6, 7, 21   
Jiele Wu, Chunhui Zhang, Zheyuan Liu, Erchi Zhang, Steven Wilson, and Chuxu Zhang. Graphbert: Bridging graph and text for malicious behavior detection on social media. In IEEE International Conference on Data Mining, 2022. 16   
Le Wu, Lei Chen, Pengyang Shao, Richang Hong, Xiting Wang, and Meng Wang. Learning fair representations for recommendation: A graph-based perspective. In International World Wide Web Conference, 2021. 8   
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019. 1   
Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. Gnnexplainer: Generating explanations for graph neural networks. In Advances in Neural Information Processing Systems, 2019. 8

Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, and Shuiwang Ji. On explainability of graph neural networks via subgraph explorations. In International Conference on Machine Learning, 2021. 8   
Xiangchi Yuan, Chunhui Zhang, Yijun Tian, Yanfang Ye, and Chuxu Zhang. Mitigating severe robustness degradation on graphs. In International Conference on Learning Representations, 2024. 1   
Han Yue, Chunhui Zhang, Chuxu Zhang, and Hongfu Liu. Label-invariant augmentation for semi-supervised graph classification. In Advances in Neural Information Processing Systems, 2022. 2   
Chunhui Zhang, Chao Huang, Youhuan Li, Xiangliang Zhang, Yanfang Ye, and Chuxu Zhang. Look twice as much as you say: Scene graph contrastive learning for self-supervised image caption generation. In ACM International Conference on Information & Knowledge Management, 2022. 1   
Chunhui Zhang, Chao Huang, Yijun Tian, Qianlong Wen, Zhongyu Ouyang, Youhuan Li, Yanfang Ye, and Chuxu Zhang. When sparsity meets contrastive models: Less graph data can bring better class-balanced representations. In International Conference on Machine Learning, 2023a. 1   
Chunhui Zhang, Hongfu Liu, Jundong Li, Yanfang Ye, and Chuxu Zhang. Mind the gap: Mitigating the distribution gap in graph few-shot learning. Transactions on Machine Learning Research, 2023b. 1   
Chunhui Zhang, Yijun Tian, Mingxuan Ju, Zheyuan Liu, Yanfang Ye, Nitesh Chawla, and Chuxu Zhang. Chasing all-round graph representation robustness: Model, training, and optimization. In International Conference on Learning Representations, 2023c. 1   
Muhan Zhang and Yixin Chen. Link prediction based on graph neural networks. In Advances in Neural Information Processing Systems, 2018. 1   
Dongmian Zou, Radu Balan, and Maneesh Singh. On lipschitz bounds of general convolutional neural networks. IEEE Transactions on Information Theory, 2019. 8, 15

# A DETAILED RELATED WORK

In the field of Lipschitz Bounds in Neural Networks, the stability of neural networks in relation to Lipschitz bounds was initially spotlighted by Szegedy et al. (2014), which postulated that larger Lipschitz bounds could cause network instability. An insightful proposition was the use of the product of the spectral norms across layers to upper bound the Lipschitz bound. Following this foundation, a slew of research (Virmaux & Scaman, 2018; Fazlyab et al., 2019; Jordan & Dimakis, 2020; Latorre et al., 2020; Huang et al., 2021b) fine-tuned the estimation of Lipschitz bounds, ushering in optimized frameworks for tighter bounds. Delving into particular network architectures, studies have explored the Lipschitz properties of convolutional layers and attention mechanisms (Zou et al., 2019; Terris et al., 2020; Araujo et al., 2021; Kim et al., 2021).

For Lipschitz Bounds in GNNs, the exploration of Lipschitz bounds took specific turns. Dasoulas et al. (2021) pioneered a normalization technique rooted in Lipschitz bounds for self-attention layers in Graph Attention Networks (GAT). This innovative method proved instrumental in training deeper GAT models by curbing gradient explosion issues. On another front, Gama & Sojoudi (2021) paved a new path by estimating the Lipschitz bound of filters through the infinite norm of matrices, with each matrix element epitomizing the Lipschitz bound for a respective position filter.

About Lipschitz, Fairness, and GNNs, A significant stride in the context of Lipschitz bounds and fairness was the introduction of a linear bound propagation method (Shi et al., 2022). This methodology adeptly estimates Lipschitz bounds across different network sections. However, it's imperative to underscore its primary design catering to Euclidean datasets, making its direct application to non-Euclidean graph data and specific fairness issues a challenge. Notably, a linkage between fairness and stability in GNNs was etched by Agarwal et al. (2021), which presented an innovative framework enabling GNNs to learn fair and stable graph representations. However, it primarily focuses on attribute group fairness, emphasizing fairness for explicitly labeled sensitive attributes. This fundamentally differs from our focus on individual fairness and no requirements on explicitly labeled sensitive attributes (e.g., gender, race), making direct comparisons infeasible. Dwork et al. (2012) conducted an in-depth investigation into the realm of fairness within classification, presenting a comprehensive framework that encompassed a task-specific metric as well as an algorithm used to optimize utility while retaining fairness constraints. However, it centered on conventional statistical learning models, and did not extend to deep models or graph models.

Our Distinction in Context: (i) while the aforementioned studies about Lipschitz, we address the notable gap in integrating Lipschitz bounds within GNN training, particularly focusing on individual fairness on graph structures; (ii) furthermore, our emphasis on facilitating the efficient calculation of Lipschitz bounds for practical training feasibility by deriving the Jacobian matrix of the model for practical training and the interpretability aspects, sets our method apart.

# A.1 COMPARISON BETWEEN INDIVIDUAL FAIRNESS AND GROUP FAIRNESS

The concepts of individual and group fairness are fundamental in the domain of machine learning ethics, particularly when designing and evaluating algorithms for fair decision-making. Both concepts aim to address fairness concerns, but they do so from different perspectives and with distinct implications. While group fairness relies on annotated sensitive attributes, individual fairness does not. The distinction between individual and group fairness in terms of reliance on annotated attributes highlights a fundamental difference in how these fairness paradigms conceptualize and address fairness.

Individual Fairness This concept is centered around the notion of treating similar individuals similarly. In a machine learning context, it implies that if two individuals are similar with respect to the attributes relevant to a decision-making process (e.g., loan approval, job recruitment), they should be treated in a comparable manner by the algorithm. Operationalizing individual fairness often involves defining a suitable metric of similarity between individuals, which can be challenging. This metric should capture all the relevant aspects that justify similar treatment. One of the main challenges is the subjective nature of defining similarity. What constitutes "similar" in one context or for one set of stakeholders might not be agreed upon universally. There's also a computational challenge in ensuring this kind of fairness at scale, as it may require pairwise comparisons among individuals.

Individual fairness has (1) No Need for Annotated Attributes: Individual fairness typically does not rely on explicitly annotated attributes, especially sensitive attributes like race, gender, or age. Instead, it focuses on the idea of treating similar individuals similarly, where similarity is often defined in the context of the specific task or decision process. Instead, it relies on (2) Implicit Attributes: The concept of similarity in individual fairness is based on implicit attributes derived from the context or the nature of the task. These attributes are usually not explicitly labeled but are inferred from the data or the specific decision-making scenario.

Group Fairness This approach focuses on ensuring fairness across predefined groups, typically defined by explicit sensitive attributes like race, gender, or age. Group fairness is concerned with statistical measures and often aims for equal treatment or outcomes across these groups. Common criteria include demographic parity, equal opportunity, and equalized odds. Group fairness is easier to quantify and implement than individual fairness as it relies on statistical measures (e.g., ensuring that selection rates for a job are equal across different gender groups). Group fairness is often applied in large-scale decision-making scenarios where societal or policy-level fairness concerns are paramount, such as in credit scoring or hiring processes. A significant challenge with group fairness is the risk of oversimplification. By focusing on broad groups, it might overlook nuances and individual-level disparities within these groups. Additionally, it can sometimes lead to unfair outcomes for individuals when trying to balance statistics at the group level.

Group fairness (1) Relies on Annotated Attributes: Group fairness explicitly relies on annotated attributes, often focusing on sensitive or protected attributes. These are explicit labels in the dataset, such as race, gender, or other demographic information. It uses (2) Explicit Categories: In group fairness, individuals are categorized based on these explicit attributes, and fairness is measured by evaluating the outcomes or treatments across these predefined groups. This approach simplifies the fairness problem by reducing it to a series of statistical measures (like demographic parity, equal opportunity, etc.) across these groups. While this simplification aids in quantification and implementation, it may overlook individual-level disparities and nuanced differences within groups.

# A.2 GNN AND FAIRNESS – A SIMPLIFIED, EASY-TO-UNDERSTAND OVERVIEW

Graph Neural Networks (GNNs) GNNs are designed to process graph or relational data. Unlike conventional data structures, graphs consist of nodes and edges, which represent entities and their relationships. GNNs excel in capturing complex patterns in such data by aggregating information from neighboring nodes, making them useful for tasks like classification, prediction, and clustering. It is popularly used on diverse real-world applications, such as social media modeling (Wu et al., 2022; Qian et al., 2022; Tian et al., 2023b; Guo et al., 2023; Li et al., 2023; Wen et al., 2022a), from which the platform also derives societal concerns such as public safety, fairness, and privacy (Wen et al., 2022b; Liu et al., 2023; Ouyang et al., 2023; Liu et al., 2024).

Ranking-based Individual Fairness The concept of individual fairness in the context of ranking is centered around the principle that individuals who are similar, based on defined attributes, should be accorded comparable rankings. This principle is particularly pertinent in the domain of GNNs, where the objective often involves ranking nodes — such as users, items, or entities — based on a blend of their intrinsic features and the nature of their interconnections within the graph.

In this approach, the focus is on the relative positioning of individuals or nodes within the ranking order. The key is not the absolute scores or ratings assigned to each individual, but rather ensuring that the rank order is fair and equitable. This means that if two nodes are deemed similar in terms of their attributes or their roles within the network, this similarity should be reflected in how they are ranked. Whether it's in generating search results, recommendations, or categorizing items, the ranking-based fairness approach strives to preserve the integrity of this relative ordering, maintaining consistency across various representations or subsets of the data. It's about upholding a fair and justifiable hierarchy that resonates with the underlying similarity and relationship patterns among the nodes in the graph.

Lipschitz Condition and Individual Fairness in our work The Lipschitz condition is a mathematical concept that, in this context, can be used to enforce a form of individual fairness. A function is said to be Lipschitz continuous if there exists a bound L such that for any two points x and y, the

difference in the function's outputs is at most $L$ times the distance between $x$ and $y$ . In simpler terms, similar inputs lead to similar outputs with a bound on how different the outputs can be.

When applied to ranking, a Lipschitz condition can ensure that if two nodes are similar (close in the graph structure or feature space), their differences in ranking (output of the GNN) are limited. This prevents wildly different rankings for similar nodes, which contributes to individual fairness.

# B PROOFS

# B.1 NOTATIONS

We use the following notations throughout the paper. Sets are denoted by {} and vectors by (). For $n \in N$ , we denote $[n] = \{1, \cdots, n\}$ . Scalars are denoted by regular letters, lowercase bold letters denote vectors, and uppercase bold letters denote matrices. For instance, $\boldsymbol{x} = (x_{1}, \cdots, x_{n})^{\top} \in \mathbb{R}^{n}$ and $X = [X_{ik}]_{i \in [n], k \in [m]} \in R^{n \times m}$ . For any vector $x \in R^{n}$ , we use $\|x\|$ to denote its $\ell_{2}$ -norm: $\|x\| = \left(\sum_{i=1}^{n} x_{i}^{2}\right)^{1/2}$ . For any matrix $X \in R^{n \times m}$ , we use $X_{i,:}$ to denote its i-th row and $X_{:,k}$ to denote its k-th column. The $(\infty, 2)$ -norm of X is denoted by $\|X\|_{\infty, 2} = \max_{i \in [n]} \|X_{i,:}\|$ . Given a graph $G = G(V, E)$ with ordered nodes, we denote its adjacency matrix by A such that $A_{ij} = 1$ if $\{i, j\} \in E$ and $A_{ij} = 0$ otherwise. When it is clear from the context, we use $X \in R^{N \times F}$ to denote a feature matrix whose i-th row corresponds to the features of the i-th node, and the j-th column represents the features across all nodes for the j-th attribute. We denote the output of the GNN as $Y \in R^{N \times C}$ , where N is the number of nodes and C is the number of output classes.

# B.2 PROOF OF LEMMA 1 IN § 3.1

Lemma 1. Given a function $g: \mathbb{R}^m \to \mathbb{R}^n$ with components $g_i$ for $i \in [n]$ , the Lipschitz bound of $g$ , denoted by $\operatorname{Lip}(g)$ , is bounded above by the norm of the vector of the Lipschitz bounds of its components, that is:

$$
\operatorname{Lip} (g) \leq \| [ \operatorname{Lip} (g _ {i}) ] _ {i = 1} ^ {n} \| \tag {10}
$$

where $\|\cdot\|$ represents the norm of a vector, and $[\mathrm{Lip}(g_{i})]_{i=1}^{n}$ represents a vector whose components are the Lipschitz bounds of the functions $g_{i}$ .

Proof. We begin by observing that

$$
\begin{array}{l} \frac {\| g (\boldsymbol {x}) - g (\boldsymbol {y}) \|}{\| \boldsymbol {x} - \boldsymbol {y} \|} = \frac {\left\| \left[ g _ {i} (\boldsymbol {x}) - g _ {i} (\boldsymbol {y}) \right] _ {i = 1} ^ {n} \right\|}{\| \boldsymbol {x} - \boldsymbol {y} \|} \tag {11} \\ = \left\| \left[ \frac {\left| g _ {i} (\boldsymbol {x}) - g _ {i} (\boldsymbol {y}) \right|}{\left\| \boldsymbol {x} - \boldsymbol {y} \right\|} \right] _ {i = 1} ^ {n} \right\|. \\ \end{array}
$$

Furthermore, for each $i \in [n]$ , $\frac{|g_i(\boldsymbol{x}) - g_i(\boldsymbol{y})|}{\|\boldsymbol{x} - \boldsymbol{y}\|} \leq \mathrm{Lip}(g_i)$ . Therefore, we can write

$$
\begin{array}{l} \operatorname{Lip} (g) = \sup _ {\boldsymbol {x} \neq \boldsymbol {y}} \frac {\| g (\boldsymbol {x}) - g (\boldsymbol {y}) \|}{\| \boldsymbol {x} - \boldsymbol {y} \|} \\ = \sup _ {\boldsymbol {x} \neq \boldsymbol {y}} \left\| \left[ \frac {\left| g _ {i} (\boldsymbol {x}) - g _ {i} (\boldsymbol {y}) \right|}{\left\| \boldsymbol {x} - \boldsymbol {y} \right\|} \right] _ {i = 1} ^ {n} \right\| \tag {12} \\ \leq \sup _ {\boldsymbol {x} \neq \boldsymbol {y}} \| [ \operatorname{Lip} (g _ {i}) ] _ {i = 1} ^ {n} \| = \| [ \operatorname{Lip} (g _ {i}) ] _ {i = 1} ^ {n} \|, \\ \end{array}
$$

this completes the proof.

![](images/ae7b9228b47a33b7f68522953d21057b24cfd7ea2cc16d2d5d9c07d3a9d02404.jpg)

In the above proof of Lemma 1, we start by rewriting the norm of the difference between $g(\pmb{x})$ and $g(\pmb{y})$ divided by the norm of $\pmb{x} - \pmb{y}$ as a norm of a vector containing the component-wise differences of $g_{i}(\pmb{x})$ and $g_{i}(\pmb{y})$ divided by the norm of $\pmb{x} - \pmb{y}$ for each $i$ . We then observe that for each $i$ , the absolute value of $g_{i}(\pmb{x}) - g_{i}(\pmb{y})$ divided by the norm of $\pmb{x} - \pmb{y}$ is bounded by the Lipschitz bound $\mathrm{Lip}(g_i)$ . Hence, the Lipschitz bound of $g$ is bounded by the norm of the vector $[\mathrm{Lip}(g_i)]_{i=1}^n$ . This establishes the inequality in Lemma 1.

# B.3 PROOF OF THEOREM 1 IN § 3.1

Theorem 1. Let Y be the output of an L-layer GNN (represented in $f(\cdot)$ ) with X as the input. Assuming the activation function (represented in $\rho(\cdot)$ ) is ReLU with a Lipschitz bound of $\operatorname{Lip}(\rho)=1$ , then the cumulative Lipschitz bound of the entire GNN, denoted as $\operatorname{Lip}(f)$ , satisfies the following inequality:

$$
\operatorname{Lip} (f) \leqslant \max _ {j} \prod_ {l = 1} ^ {L} F ^ {l ^ {\prime}} \left\| \left[ \mathcal {J} (h ^ {l}) \right] _ {j} \right\| _ {\infty}, \tag {13}
$$

where $F^{l'}$ represents the output dimension of the l-th message-passing layer, j is the index of the node (e.g., j-th), and the vector $\left[\mathcal{J}(h^{l})\right]=\left[\left\|\boldsymbol{J}_{1}(h^{l})\right\|,\left\|\boldsymbol{J}_{2}(h^{l})\right\|,\cdots,\left\|\boldsymbol{J}_{F^{l'}}(h^{l})\right\|\right]$ . Notably, $\boldsymbol{J}_{i}(h^{l})$ denotes the i-th row of the Jacobian matrix of the l-th layer's input and output, and $\left[\mathcal{J}(h^{l})\right]_{j}$ is the vector corresponding to the j-th node in the l-th layer $h^{l}(\cdot)$ .

The inequality in Theorem 1 constrains the cumulative Lipschitz bound of the entire GNN based on the layer outputs and Jacobian matrices. It is derived as follows:

Proof. We begin by examining the Lipschitz property of the GNN. Let Y denote the output of an L-layer GNN with input X. Assuming the commonly used ReLU activation function as the non-linear layer $\rho(\cdot)$ , we have $\operatorname{Lip}(\rho)=1$ . First, we consider the Lipschitz bound between the hidden states of two nodes output by any message-passing layer $h(\cdot)$ in $f(\cdot)$ . Let $z_{1}$ and $z_{2}$ represent the hidden states of node features $x_{1}$ and $x_{2}$ , respectively. The Lipschitz bound between these hidden states is given by:

$$
\frac {\left\| z _ {1} - z _ {2} \right\|}{\left\| x _ {1} - x _ {2} \right\|} = \frac {\left\| \left(h (x _ {1}) - h (x _ {2})\right) \right\|}{\left\| x _ {1} - x _ {2} \right\|}. \tag {14}
$$

By applying the triangle inequality, we obtain:

$$
\frac {\left\| z _ {1} - z _ {2} \right\|}{\left\| x _ {1} - x _ {2} \right\|} \leqslant \left\| \left[ \frac {h (x _ {1}) _ {i} - h (x _ {2}) _ {i}}{\left\| x _ {1} - x _ {2} \right\|} \right] _ {i = 1} ^ {F ^ {\prime}} \right\|, \tag {15}
$$

Next, we consider the Lipschitz bound between individual elements of the hidden states. Let $f(x_{1})$ and $f(x_{2})$ represent the hidden state matrices for inputs $x_{1}$ and $x_{2}$ , respectively. By again applying the triangle inequality, we have:

$$
\frac {\left\| z _ {1} - z _ {2} \right\|}{\left\| x _ {1} - x _ {2} \right\|} \leqslant \left\| F ^ {\prime} \times \max _ {i} \frac {h (x _ {1}) _ {i} - h (x _ {2}) _ {i}}{\left\| x _ {1} - x _ {2} \right\|} \right\|, \tag {16}
$$

let's focus on the Lipschitz bound of the individual elements, $\frac{h(x_1)_i - h(x_2)_i}{\|x_1 - x_2\|}$ : here, $f(x)_i$ denotes the $i$ -th column of the matrix $f(x)$ . We denote $h^l (\cdot)$ as the operation of the $l$ -th message-passing layer in $f(\cdot)$ , then by applying the triangle inequality and leveraging the Lipschitz property of the ReLU activation function, we have:

$$
\frac {\left\| f \left(x _ {1}\right) _ {j , 1} - f \left(x _ {2}\right) _ {j , 2} \right\|}{\left\| x _ {j , 1} - x _ {j , 2} \right\|} \leqslant \prod_ {l = 1} ^ {L} F ^ {l ^ {\prime}} \left\| \left[ \mathcal {J} \left(h ^ {l}\right) \right] _ {j} \right\| _ {\infty}, \tag {17}
$$

where $x_{j,1}$ and $x_{j,1}$ denote features of j-th node's in $x_{1}$ and $x_{2}$ , respectively, and $\left[\mathcal{J}(h^{l})\right]_{j}$ represents the j-th node's the Jacobian matrix of the l-th message-passing layer. Therefore, the Lipschitz bound for the GNN can be expressed as:

$$
\operatorname{Lip} (f) = \max _ {j} \prod_ {l = 1} ^ {L} F ^ {l ^ {\prime}} \left\| \left[ \mathcal {J} (h ^ {l}) \right] _ {j} \right\| _ {\infty}. \tag {18}
$$

In summary, we have shown that for any two input samples $x_{1}$ and $x_{2}$ , the Lipschitz bound of the GNN, denoted as $\operatorname{Lip}(f)$ , satisfies:

$$
\left\| \boldsymbol {Y} _ {1} - \boldsymbol {Y} _ {2} \right\| \leqslant \operatorname{Lip} (f) \left\| \boldsymbol {X} _ {1} - \boldsymbol {X} _ {2} \right\|, \tag {19}
$$

where Y denotes the output of the GNN for inputs X. This inequality implies that the Lipschitz bound $\operatorname{Lip}(f)$ controls the magnitude of changes in the output based on input biases/perturbations. Therefore, we have established the following result:

$$
\left\| \mathbf {Y} _ {1} - \mathbf {Y} _ {2} \right\| \leqslant \prod_ {l = 1} ^ {L} F ^ {l ^ {\prime}} \left\| \left[ \mathcal {J} (h ^ {l}) \right] _ {j} \right\| _ {\infty} \left\| \mathbf {X} _ {1} - \mathbf {X} _ {2} \right\|. \tag {20}
$$

This inequality demonstrates that the Lipschitz bound of the GNN, $\mathrm{Lip}(f)$ , controls the magnitude of the difference in the output $Y$ based on the difference in the input $X$ . It allows us to analyze the stability of the model's output with respect to input perturbations.

# B.4 THE LOCALITY IN LIPSCHITZ COMPUTATION

For local Lipschitz bound computation on graph data, we clarify that the Jacobian matrix is a 2D tensor, and we don't consider the case of Equation (7) where $i \neq j$ :

- If $\frac{\partial Y_{j1}}{\partial X_{i1}}$ where $i \neq j$ , then the computed Lipschitz bound will not be a local Lipschitz bound (as in our paper with $\frac{\partial Y_{i1}}{\partial X_{i1}}$ ) but rather a global Lipschitz bound, which considers the Lipschitz relationships for all nodes. However, as discussed in §3 of Dong et al. (2021): solving for the global Lipschitz bounds of a 2-layer MLP is an NP-Hard problem. Therefore, even considering this facet for simple models (like GCN and SGC) would lead to a significant increase in running time, rendering the computation of Lipschitz bounds impractical.   
- However, we contend that global Lipschitz bounds do not align with graph individual fairness (we enable similar individuals in the input space to maintain their similarity in the output space, ensuring individual fairness. It is achieved by that: we constrain the expansion of data in the input space to the output space, thereby minimizing the impact of variations in input data on the output space, which is shaped as a Lipschitz constraint problem): First, if $i \neq j$ , the equations are considering the Lipschitz bound for the entire relational network. In this case, the Lipschitz bound takes into account each pair of nodes within the network (i.e., each single one node must consider its interactions with all other nodes). This doesn't consider minor perturbations; rather, the perturbation can be understood as the difference between different nodes. For example, the dimension of each node takes into account the impact of all other nodes (again, in this scenario, the complexity is extremely high because the dimension of each node needs to account for all dimensions of all other nodes, leading to an astonishing amount of computation). Under these circumstances, it is the global Lipschitz bound, which doesn't consider the locality of our GCN's local message passing (unlike our local Lipschitz bound) on graph data; Second, in Dong et al. (2021), the ranking-based individual fairness computes similarity based on the nearest k neighbors, which is a local similarity measure and more aligned with our local Lipschitz.

\- In this context, when $i = j$ , the local Lipschitz bounds are more aligned with the objective, as there is no need to maintain pairwise similarity between all nodes in the output space: a node is solely influenced by its neighboring nodes. When nodes are similar, meaning they have a high degree of similarity, the difference in features between two nodes can be approximated by a small perturbation. We only need to consider how this perturbation affects the output, which can be quantified using gradients. If the gradient is steep, even a small perturbation can lead to a significant impact.

Moreover, the Jacobian matrix serves as a suitable measure for the Lipschitz bounds on each node and dimension, and frameworks like PyTorch provide mechanisms to expedite its computation. This approach reduces algorithmic complexity, making it more adaptable, while also aligning with the principles of individual fairness.

# B.5 HOW LIPSCHITZ BOUNDS CAN GUARANTEE RANKING-BASED INDIVIDUAL FAIRNESS?

In the context of individual fairness on graphs, the relationship between ranking-based individual fairness and Lipschitz continuity is established through Lipschitz constraints on the GNN model. The Lipschitz continuity ensures that the model's predictions are not overly sensitive to small changes in node embeddings. Let's define the two concepts and their relationship mathematically:

Lipschitz Property In the GNN model $f: V \to R^{d}$ , Lipschitz continuity constrains the change in model output concerning small changes/biases in input (node embeddings). A function f is Lipschitz continuous if there exists a bound K such that:

$$
\left\| f \left(z _ {i}\right) - f \left(z _ {j}\right) \right\| \leq K \| z _ {i} - z _ {j} \|, \tag {21}
$$

for any two input points $z_{i}$ and $z_{j}$ . Here, $\| \cdot \|$ denotes a norm (e.g., $L2$ norm), and $K$ is the Lipschitz bound.

Ranking-based Individual Fairness Based on prior work (Dong et al., 2021; Kang et al., 2020), this concept in GNNs focuses on maintaining the rank-ordering of pairwise node similarities pre and post-training. This is achieved by a ranking loss function $L(\cdot)$ that penalizes discrepancies between predicted pairwise distances and original similarities as:

$$
\text { Ranking   Loss: } \mathcal {L} _ {\text { rank }} = \sum_ {(v _ {i}, v _ {j}) \in \mathcal {P}} L (S (v _ {i}, v _ {j}), D (z _ {i}, z _ {j})), \tag {22}
$$

where P is the set of all node pairs, and $L(\cdot)$ measures the discrepancy between predicted pairwise distance $D(z_{i}, z_{j})$ and original similarity $S(v_{i}, v_{j})$ .

The Exact Relationship By imposing a Lipschitz constraint on the GNN, the model's output becomes less sensitive to minor variations (i.e., minor biases) in node embeddings, thereby reinforcing ranking-based individual fairness. The mathematical connection can be expressed as:

$$
\operatorname{RankLoss} \left(S \left(v _ {i}, v _ {j}\right), D \left(z _ {i}, z _ {j}\right)\right) \propto L \left(f \left(z _ {i}\right), f \left(z _ {j}\right)\right), \tag {23}
$$

where $\text{RankLoss}(S(v_i, v_j), D(z_i, z_j))$ is the ranking loss function, measuring the discrepancy between predicted pairwise distances $D(z_i, z_j)$ and original similarities $S(v_i, v_j)$ in the output space.

Enforcing a Lipschitz constraint on the GNN's model parameters or architecture inherently promotes the preservation of the rank-ordering of node similarities during training. This approach drives the GNN to uphold individual fairness by ensuring that nodes with similar attributes receive comparable rankings in the output space, resulting in fair and unbiased predictions for each node.

In summary, the principle of Lipschitz continuity in GNNs is directly related to the concept of ranking-based individual fairness. It achieves this by moderating the model's response to small variations in the embeddings of nodes.

# C ADDITIONAL EXPERIMENTS

# C.1 NODE CLASSIFICATION TASK: EVALUATED UNDER ACCURACY AND ERROR

According to Table 3, on node classification tasks, JacoLip consistently shows a competitive or improved trade-off between accuracy and error compared to the baselines, highlighting the effectiveness of the Lipschitz bound in promoting individual fairness on graphs.

# C.2 CONTRASTING WITH LIPSCHITZ METHODS TAILORED FOR OTHER DOMAINS

Although there are dozens of Lipschitz methods of fundamental statistical models (Dwork et al., 2012) or deep models Shi et al. (2022); Agarwal et al. (2021), it is essential to acknowledge their distinctiveness, making comparisons less straightforward.

# C.2.1 CONSIDERATION ON LIPSCHITZ METHODS FOR CNN/MLP

When contrasting our approach with previous Lipschitz regularization studies for general CNN/MLP, we use Shi et al. (2022) as an example, and we highlight the following points:

First, the approach in Shi et al. (2022) does not directly use existing Jacobian matrices. It introduces a concept called Clarke Jacobian, leading to explicit expressions of the bounds. This method requires utilizing the neural network's forward pass, necessitating computations of input and output at each step of the network, making the process complex. It primarily focuses on computing the Lipschitz

Table 3: Evaluation on node classification tasks: comparing under accuracy and error. 

<table><tr><td rowspan="2">Data</td><td rowspan="2">Model</td><td rowspan="2">Fair Alg.</td><td colspan="2">Feature Similarity</td><td colspan="2">Structural Similarity</td></tr><tr><td>utility: Acc.↑</td><td>fairness: Err.@10↑</td><td>utility: Acc.↑</td><td>fairness: Err.@10↑</td></tr><tr><td rowspan="12">ACM</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2017)</td><td> $72.49 \pm 0.6$ </td><td> $75.70 \pm 0.6$ </td><td> $72.49 \pm 0.6$ </td><td> $37.55 \pm 0.4$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $67.65 \pm 1.0(-6.68\%)$ </td><td> $73.49 \pm 0.5(-2.92\%)$ </td><td> $65.91 \pm 0.2(-9.07\%)$ </td><td> $19.96 \pm 0.6(-46.8\%)$ </td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td> $68.48 \pm 0.6(-5.53\%)$ </td><td> $76.28 \pm 0.1(0.77\%)$ </td><td> $70.22 \pm 0.7(-3.13\%)$ </td><td> $36.54 \pm 0.4(-2.69\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $73.46 \pm 0.2(+1.34\%)$ </td><td> $82.27 \pm 0.1(+8.68\%)$ </td><td> $71.87 \pm 0.4(-0.86\%)$ </td><td> $43.74 \pm 0.0(+16.5\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $72.80 \pm 0.2(+4.27\%)$ </td><td> $82.88 \pm 0.1(+9.48\%)$ </td><td> $72.30 \pm 0.4(-0.26\%)$ </td><td> $39.28 \pm 0.2(+4.61\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $71.05 \pm 0.4(+2.00\%)$ </td><td> $82.21 \pm 0.3(+8.60\%)$ </td><td> $71.92 \pm 0.3(-0.79\%)$ </td><td> $46.13 \pm 0.3(+22.85\%)$ </td></tr><tr><td rowspan="6">SGC</td><td>Vanilla (Wu et al., 2019)</td><td> $68.40 \pm 1.0$ </td><td> $80.06 \pm 0.1$ </td><td> $68.40 \pm 1.0$ </td><td> $45.95 \pm 0.3$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $67.96 \pm 0.5(-0.64\%)$ </td><td> $75.63 \pm 0.5(-5.53\%)$ </td><td> $66.16 \pm 0.6(-3.27\%)$ </td><td> $39.79 \pm 0.1(-13.4\%)$ </td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td> $67.69 \pm 0.4(-1.04\%)$ </td><td> $76.80 \pm 0.1(-4.07\%)$ </td><td> $66.69 \pm 0.3(-2.50\%)$ </td><td> $46.99 \pm 0.5(+2.26\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $66.51 \pm 0.3(-2.76\%)$ </td><td> $82.32 \pm 0.3(+2.82\%)$ </td><td> $67.10 \pm 0.7(-1.90\%)$ </td><td> $49.02 \pm 0.2(+4.76\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $74.04 \pm 0.2(+8.25\%)$ </td><td> $82.73 \pm 0.7(+5.18\%)$ </td><td> $72.91 \pm 0.9(+3.33\%)$ </td><td> $48.64 \pm 0.2(+5.85\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $69.91 \pm 0.1(+2.21\%)$ </td><td> $85.22 \pm 0.4(+6.45\%)$ </td><td> $71.27 \pm 0.3(+4.20\%)$ </td><td> $52.01 \pm 0.4(+13.23\%)$ </td></tr><tr><td rowspan="12">CS</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2017)</td><td> $90.59 \pm 0.3$ </td><td> $80.41 \pm 0.1$ </td><td> $90.59 \pm 0.3$ </td><td> $26.69 \pm 1.3$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $88.37 \pm 0.9(-2.45\%)$ </td><td> $80.63 \pm 0.6(+0.27\%)$ </td><td> $87.10 \pm 0.9(-3.85\%)$ </td><td> $29.68 \pm 0.6(+11.2\%)$ </td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td> $87.62 \pm 0.2(-3.28\%)$ </td><td> $76.26 \pm 0.1(-5.16\%)$ </td><td> $85.66 \pm 0.7(-5.44\%)$ </td><td> $19.80 \pm 1.4(-25.8\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $90.06 \pm 0.5(-0.59\%)$ </td><td> $83.24 \pm 0.2(+3.52\%)$ </td><td> $89.91 \pm 0.2(-0.86\%)$ </td><td> $32.42 \pm 1.6(+21.5\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $90.41 \pm 0.4(-0.20\%)$ </td><td> $82.57 \pm 0.1(+2.69\%)$ </td><td> $89.12 \pm 0.1(-1.62\%)$ </td><td> $32.80 \pm 0.6(+22.74\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $90.30 \pm 0.3(-0.32\%)$ </td><td> $88.11 \pm 0.3(+9.58\%)$ </td><td> $89.93 \pm 0.2(-0.73\%)$ </td><td> $42.50 \pm 0.4(+59.24\%)$ </td></tr><tr><td rowspan="6">SGC</td><td>Vanilla (Wu et al., 2019)</td><td> $87.48 \pm 0.8$ </td><td> $90.58 \pm 0.1$ </td><td> $87.48 \pm 0.8$ </td><td> $43.28 \pm 0.2$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $87.31 \pm 0.5(-0.19\%)$ </td><td> $90.64 \pm 0.1(+0.07\%)$ </td><td> $88.21 \pm 0.4(+0.83\%)$ </td><td> $44.37 \pm 0.1(+0.21\%)$ </td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td> $87.95 \pm 0.2(+0.54\%)$ </td><td> $79.85 \pm 0.2(-11.8\%)$ </td><td> $86.93 \pm 0.1(-0.63\%)$ </td><td> $38.83 \pm 0.8(-10.3\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $90.48 \pm 0.2(+3.43\%)$ </td><td> $92.03 \pm 0.1(+1.60\%)$ </td><td> $90.39 \pm 0.1(+3.33\%)$ </td><td> $45.81 \pm 0.0(+5.85\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $90.71 \pm 0.3(+3.69\%)$ </td><td> $90.75 \pm 0.4(+0.19\%)$ </td><td> $90.34 \pm 1.0(+3.27\%)$ </td><td> $43.92 \pm 0.3(+1.48\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $92.22 \pm 0.2(+5.42\%)$ </td><td> $92.22 \pm 0.4(+1.81\%)$ </td><td> $90.54 \pm 0.3(+3.50\%)$ </td><td> $46.39 \pm 0.5(+7.19\%)$ </td></tr><tr><td rowspan="12">Phy</td><td rowspan="6">GCN</td><td>Vanilla (Kipf &amp; Welling, 2017)</td><td> $94.81 \pm 0.2$ </td><td> $73.25 \pm 0.3$ </td><td> $94.81 \pm 0.2$ </td><td> $2.58 \pm 0.1$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $88.67 \pm 0.7(-6.48\%)$ </td><td> $73.80 \pm 0.6(+0.75\%)$ </td><td> $94.68 \pm 0.2(-0.14\%)$ </td><td> $2.45 \pm 0.1(-5.04\%)$ </td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td> $88.79 \pm 0.2(-6.35\%)$ </td><td> $73.22 \pm 0.4(+0.10\%)$ </td><td> $89.69 \pm 1.0(-5.40\%)$ </td><td> $1.67 \pm 0.1(-35.3\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $93.71 \pm 0.1(-1.16\%)$ </td><td> $80.23 \pm 0.1(+9.53\%)$ </td><td> $93.91 \pm 0.4(-0.95\%)$ </td><td> $3.22 \pm 0.3(+22.9\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $93.71 \pm 0.2(-1.16\%)$ </td><td> $78.64 \pm 1.1(+7.36\%)$ </td><td> $94.75 \pm 0.3(-0.06\%)$ </td><td> $2.75 \pm 0.6(+6.80\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $93.79 \pm 0.8(-1.08\%)$ </td><td> $82.60 \pm 0.3(+12.70\%)$ </td><td> $93.98 \pm 0.3(-0.88\%)$ </td><td> $4.0 \pm 0.1(+55.43\%)$ </td></tr><tr><td rowspan="6">SGC</td><td>Vanilla (Wu et al., 2019)</td><td> $94.45 \pm 0.2$ </td><td> $77.48 \pm 0.2$ </td><td> $94.45 \pm 0.2$ </td><td> $4.50 \pm 0.1$ </td></tr><tr><td>InFoRM (Kang et al., 2020)</td><td> $92.06 \pm 0.2(-2.53\%)$ </td><td> $75.13 \pm 0.4(-3.03\%)$ </td><td> $94.27 \pm 0.1(-0.19\%)$ </td><td> $4.44 \pm 0.0(-1.33\%)$ </td></tr><tr><td>PFR (Lahoti et al., 2019)</td><td> $87.39 \pm 1.2(-7.47\%)$ </td><td> $73.42 \pm 0.2(-5.24\%)$ </td><td> $89.16 \pm 0.3(-5.60\%)$ </td><td> $3.41 \pm 0.2(-24.2\%)$ </td></tr><tr><td>Redress (Dong et al., 2021)</td><td> $94.81 \pm 0.2(+0.38\%)$ </td><td> $79.57 \pm 0.2(+2.70\%)$ </td><td> $94.54 \pm 0.1(+0.10\%)$ </td><td> $4.98 \pm 0.1(+10.7\%)$ </td></tr><tr><td>JacoLip (on Vanilla)</td><td> $94.43 \pm 0.7(-0.02\%)$ </td><td> $78.82 \pm 0.8(+1.73\%)$ </td><td> $94.09 \pm 0.6(-0.38\%)$ </td><td> $4.75 \pm 0.2(+5.56\%)$ </td></tr><tr><td>JacoLip (on Redress)</td><td> $94.78 \pm 0.1(+0.35\%)$ </td><td> $82.21 \pm 0.2(+6.10\%)$ </td><td> $93.00 \pm 1.3(-1.54\%)$ </td><td> $5.45 \pm 0.1(+1.90\%)$ </td></tr></table>

bounds of the network, striving to find a tight bound. However, the paper does not provide specific application results for fairness, leaving doubts about its effectiveness in fairness-related contexts.

Second, in contrast, our method is simpler and leverages PyTorch's built-in automatic differentiation mechanism for all computations, requiring no additional computational cost. Furthermore, our approach treats the entire network model as a whole, disregarding the internal computation flow of the network. It only requires the model's inputs and outputs, making it easy and convenient to use. Importantly, our Lipschitz method is designed to cater to the application context of individual fairness. Our goal is not solely to find a tight bound, but rather to integrate the Lipschitz bound as a regularization term into the training process. By continuously constraining the Lipschitz bound during training, we ensure individual fairness. The underlying objectives of the two methods are distinct: while the approach in Shi et al. (2022) aims for tight bounds, our Lipschitz bound serves the purpose of ensuring individual fairness. In the context of fairness, achieving a tight bound is not the primary concern. Instead, our focus is on preserving fairness in the predictions, where the exact tightness of the bound is less crucial to GNN fairness training.

Third, extending our investigation to GNNs using the code provided by Shi et al. (2022), we computed local Lipschitz bounds. Our experimental results are tabulated at Table 4, revealing the utility and fairness performances of both methods. Notably, the accuracy-fairness tradeoff in Shi et al. (2022) is lower than our approach. This observation echoes the earlier speculation of an application mismatch, as outlined in the preceding points. Moreover, we also infer that the graph structure's (adjacency matrix) influence on GNN's forward pass necessitates specialized treatment during Jacobian bound calculations. This realization prompts a more profound inquiry—one that should be taken into account when contemplating potential applications of Shi et al. (2022) to graph individual fairness.

Finally, beyond the performance benefits from our fairness design's alignment with graph structures (as Equation (8)'s consideration on graphs), our methodology also champions the efficient com-

Table 4: Comparisons on the FB dataset with the GAE backbone. 

<table><tr><td rowspan="2"></td><td colspan="2">Feat. Simi.</td><td colspan="2">Struct. Simi.</td></tr><tr><td>utility: AUC↑</td><td>fairness: NDCG@10↑</td><td>utility: AUC↑</td><td>fairness: NDCG@10↑</td></tr><tr><td>Shi et al. (2022)</td><td>94.40</td><td>27.40</td><td>96.28</td><td>22.90</td></tr><tr><td>JacoLip (Ours)</td><td>97.40</td><td>27.44</td><td>97.02</td><td>30.90</td></tr></table>

putation of Lipschitz bounds for non-Euclidean models. While Dwork et al. (2012) pioneered the regularization of Lipschitz bounds for fairness, it could not cater to graph data as efficiently as our research, since it is proposed on traditional (relatively) simple statistical models. Similarly, although Shi et al. (2022) does compute a tight Lipschitz bound for CNN/MLP, our method emphasizes empirical individual-fairness predictions over theoretical tightness. This is evidenced by our prior experiments and this preliminary time-per-iteration comparison with Shi et al. (2022) at Table 5:

Table 5: Time-per-iteration comparisons on the FB dataset. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Practical Training Feasibility</td><td rowspan="2">Time/Iter.</td></tr><tr><td>Graph Structure</td><td>Efficiency</td></tr><tr><td>GCN without Lipschitz</td><td>N.A.</td><td>N.A.</td><td>0.2841s</td></tr><tr><td>Ours</td><td>Yes</td><td>Yes</td><td>0.2972s</td></tr><tr><td>Shi et al. (2022)</td><td>No</td><td>No</td><td>3.2407s</td></tr></table>

# C.2.2 CONSIDERATION ON LIPSCHITZ METHODS ON ATTRIBUTED GRAPHS

Furthermore, Agarwal et al. (2021) established a link between fairness and stability, introducing a novel framework for GNNs to acquire both fair and stable graph representations. However, it pursued group fairness to enhance attribute fairness for explicitly labeled sensitive attributes, resulting in a distinct definition from our emphasis on individual fairness within regular graphs with no explicitly labeled sensitive attributes. Consequently, our research and that of Agarwal et al. (2021) cannot be appropriately compared in the same experimental scenarios due to fundamental differences in motivation and application contexts.

# C.3 SENSITIVITY OF THE $\mu$ FOR THE LIPSCHITZ REGULARIZATION TERM

To provide a comprehensive analysis for understanding how varying hyperparameters impact the Lipschitz bound of the GNN, we experimented with different values of the hyperparameter $\mu$ , which directly influences the Lipschitz bound of the model. A larger value of $\mu$ correlates with a smaller Lipschitz bound, making it a pivotal factor in our approach. Below are Table 6 and Table 7 illustrating the effects of varying $\mu$ on the GCN model applied to the Node Classification task on the Co-author-CS dataset. These tables present the outcomes in terms of AUC and NDCG metrics:

Table 6: Feature Similarity: GCN on Co-author-CS dataset with varying $\mu$ 

<table><tr><td>μ</td><td>0</td><td>0.1</td><td>0.01</td><td>0.001</td><td>0.0001</td><td>0.00001</td><td>0.000001</td></tr><tr><td>NDCG@10↑</td><td>50.84</td><td>37.67</td><td>48.98</td><td>63.56</td><td>68.32</td><td>67.91</td><td>66.26</td></tr><tr><td>AUC↑</td><td>90.59</td><td>60.18</td><td>89.96</td><td>90.96</td><td>90.20</td><td>90.23</td><td>90.18</td></tr></table>

The results from these tables clearly show that both AUC and NDCG metrics are sensitive to variations in the Lipschitz bound, which is controlled by $\mu$ . Notably, when $\mu$ is relatively large, resulting in a smaller Lipschitz bound, there is a significant impact on both performance and fairness metrics. As $\mu$ decreases, allowing the Lipschitz bound of the model to approach that of the original model, we observe a convergence of both performance and fairness metrics towards those of the original model.

This analysis highlights the delicate balance that must be maintained between fairness and performance in our approach. The data reveals that the metric of fairness is more responsive to changes in $\mu$ compared to performance. Therefore, identifying an optimal Lipschitz bound that adequately meets both fairness and performance criteria becomes paramount in our methodology.

Table 7: Structure Similarity: GCN on Co-author-CS dataset with varying $\mu$ 

<table><tr><td> $\mu$ </td><td>0</td><td>0.1</td><td>0.01</td><td>0.001</td><td>0.0001</td><td>0.00001</td><td>0.000001</td></tr><tr><td>NDCG@10↑</td><td>18.29</td><td>4.11</td><td>10.66</td><td>19.62</td><td>31.82</td><td>26.82</td><td>19.42</td></tr><tr><td>AUC↑</td><td>90.34</td><td>33.46</td><td>86.98</td><td>89.25</td><td>89.12</td><td>89.18</td><td>89.51</td></tr></table>

# C.4 COMPUTATIONAL OVERHEAD

We provide a detailed computational comparison using the FB dataset. This dataset comprises 4,039 nodes, 88,234 edges, and 1,406 attributes, making it a representative choice for our analysis. Table 8 below presents a comparative analysis of time and GPU memory usage for various methods applied to the FB dataset. The results from this comparison demonstrate that our proposed method incurs no additional computational overhead when contrasted with the standard training procedures. These findings offer a more concrete empirical evidence to support it.

Table 8: Time and GPU memory usage of different methods on FB dataset. 

<table><tr><td>Method</td><td>Lipschitz Tightness</td><td>Graph Structure</td><td>Efficiency</td><td>Time/Iter.</td><td>GPU Memory-Usage</td></tr><tr><td>GCN (standard)</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>0.2841s</td><td>414.8 MiB</td></tr><tr><td>Lipschitz originally for CNN/MLP Shi et al. (2022)</td><td>Restrictive</td><td>No</td><td>No</td><td>3.2407s</td><td>2130.1 MiB</td></tr><tr><td>Layered Lipschitz Computing for GNN Jia et al. (2023)</td><td>Restrictive</td><td>Yes</td><td>No</td><td>3.8541s</td><td>3280.7 MiB</td></tr><tr><td>Ours (GCN + JacoLip)</td><td>Restrictive</td><td>Yes</td><td>Yes</td><td>0.2972s</td><td>420.0 MiB</td></tr></table>

# C.5 HOW JACOLIP AVOIDS MEMORY AND COMPUTE-INTENSIVE OPERATIONS?

A key to our approach is the mitigation of the memory and computational challenges commonly associated with computing the norms of the Jacobian matrix. We achieve this efficiency by avoiding the direct computation of the Jacobian matrix and consequently, the need for second-order derivative calculations.

Specifically, our method capitalizes on the capability to compute the norm of the Jacobian matrix without the necessity to explicitly form the Jacobian itself. This approach eliminates the requirement for gradient of gradients computation. The key segment of our code is in Listing 1:

```python
for i in range(out.shape[1]):
    # Directional derivative vectors are initialized as zero vectors
    v = torch.zeros_like(out)
    v[:, i] = 1

    # Compute gradients with respect to the inputs, not the model parameters
    gradients = autograd.grad(outputs=out, inputs=input, grad_outputs=v,
    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # We only compute the norm of these gradients, which is a first-order operation
    grad_norm = torch.norm(grades, dim=1).unsqueeze(dim=1)
    lip_mat.append(grad_norm) 
```

Listing 1: Python code for computing gradients and norms

This code snippet demonstrates the computation of the gradient concerning the inputs, which is a first-order derivative. While we set the create\_graph=True parameter, which allows for higher-order derivative computations, our method only utilizes this for first-order derivative calculations. The rationale is that our Lipschitz bound approximation demands only the norms of these gradients, not the gradients of these norms. Consequently, the computational complexity of our method remains at the first-order derivative level, with only a marginal increase compared to standard training procedures.

Furthermore, we implement an efficient batching of gradient norm computations and do not retain the computational graph of these norms. This approach ensures that the memory footprint does not scale with the size of the Jacobian, but increases linearly with the output size. Given that the output size in relational datasets typically corresponds to a small dimension (i.e., the number of classes), the increase in memory requirement is minimal and manageable.

Additionally, our method leverages the inherent efficiencies of PyTorch's autograd system. This system dynamically allocates and deallocates memory during the training process, optimizing gradient computations and ensuring minimal memory usage by retaining only necessary gradients.

In summary, our approach computes the Lipschitz bound without incurring the significant computational overhead associated with second-order derivatives. Our empirical results, supported by the provided code, affirm that our method does not result in substantial increases in training time or GPU memory usage when compared with standard training procedures.

# D MODEL CARD

The hyper-parameters for our method across all datasets are listed in Table 9. For fair comparisons, we follow the default settings of Redress (Dong et al., 2021).

Table 9: Hyperparameters used in our experiments. 

<table><tr><td rowspan="2">Hyperparameters</td><td colspan="4">Node Classification</td><td colspan="2">Link Prediction</td></tr><tr><td>ACM</td><td>Coauthor-CS</td><td>Coauthor-Phy</td><td>Blog</td><td>Flickr</td><td>Facebook</td></tr><tr><td colspan="7">Hyperparameters w.r.t. the GCN model</td></tr><tr><td># Layers</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Hidden Dimension Activation</td><td>[16, 9]</td><td>[16, 15]</td><td>[16, 5]</td><td>[32, 16]</td><td>[32, 16]</td><td>[32, 16]</td></tr><tr><td>Dropout</td><td>0.03</td><td>0.03</td><td colspan="4">ReLU used for all datasets</td></tr><tr><td>Optimizer</td><td></td><td colspan="5">AdamW with 1e - 5 weight decay</td></tr><tr><td>Pretrain Steps</td><td>300</td><td>300</td><td>300</td><td>200</td><td>200</td><td>200</td></tr><tr><td>Training Steps</td><td>150</td><td>200</td><td>200</td><td>60</td><td>100</td><td>50</td></tr><tr><td>Learning Rate</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td colspan="7">Hyperparameters w.r.t. the SGC model</td></tr><tr><td># Layers</td><td>1</td><td>1</td><td>1</td><td>N.A.</td><td>N.A.</td><td>N.A.</td></tr><tr><td>Hidden Dimension</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td></tr><tr><td>Dropout</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td></tr><tr><td>Optimizer</td><td></td><td colspan="5">AdamW with 1e - 5 weight decay</td></tr><tr><td>Pretrain Steps</td><td>300</td><td>500</td><td>500</td><td>N.A.</td><td>N.A.</td><td>N.A.</td></tr><tr><td>Training Steps</td><td>15</td><td>40</td><td>30</td><td>N.A.</td><td>N.A.</td><td>N.A.</td></tr><tr><td>Learning Rate</td><td>0.01</td><td>0.01</td><td>0.01</td><td>N.A.</td><td>N.A.</td><td>N.A.</td></tr><tr><td colspan="7">Hyperparameters w.r.t. the GAE model</td></tr><tr><td># Layers</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Hidden Dimension</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>[32, 16]</td><td>[32, 16]</td><td>[32, 16]</td></tr><tr><td>Activation</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>N.A.</td></tr><tr><td>Dropout</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Optimizer</td><td></td><td colspan="5">AdamW with 1e - 5 weight decay</td></tr><tr><td>Pretrain Steps</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>200</td><td>200</td><td>200</td></tr><tr><td>Training Steps</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>60</td><td>100</td><td>50</td></tr><tr><td>Learning Rate</td><td>N.A.</td><td>N.A.</td><td>N.A.</td><td>0.01</td><td>0.01</td><td>0.01</td></tr></table>

# E DATASET DESCRIPTIONS

We provide additional details about the datasets utilized in our work, as discussed in § 4.1:

- Citation Networks: Each node corresponds to a paper, and an edge between two nodes represents the citation relationship between the respective papers. The node attributes in these networks are generated using the bag-of-words model based on the abstract sections of the published papers.   
- Co-author Networks: It consist of nodes representing authors, where an edge between two nodes indicates that the corresponding authors have collaborated on a paper. The node attributes in these networks are constructed based on the bag-of-words model using the authors' profiles.   
- Social Networks: Each node represents a user, and the links between nodes represent interactions between users. The attributes associated with these nodes are derived from user profiles.

The datasets used in our work are referred to as CS and Phy, which are abbreviations for the Co-author-CS and Co-author-Phy datasets, respectively. A comprehensive overview of the datasets, including their detailed statistics, is presented in Table 10.

Table 10: Detailed statistics of the datasets used for node classification and link prediction. We follow the default settings of Redress (Dong et al., 2021) fair comparisons. 

<table><tr><td>Task</td><td>Dataset</td><td># Nodes</td><td># Edges</td><td># Features</td><td># Classes</td></tr><tr><td rowspan="3">node classification</td><td>ACM</td><td>16,484</td><td>71,980</td><td>8,337</td><td>9</td></tr><tr><td>Coauthor-CS</td><td>18,333</td><td>81,894</td><td>6,805</td><td>15</td></tr><tr><td>Coauthor-Phy</td><td>34,493</td><td>247,962</td><td>8,415</td><td>5</td></tr><tr><td rowspan="3">link prediction</td><td>BlogCatalog</td><td>5,196</td><td>171,743</td><td>8,189</td><td>N.A.</td></tr><tr><td>Flickr</td><td>7,575</td><td>239,738</td><td>12,047</td><td>N.A.</td></tr><tr><td>Facebook</td><td>4,039</td><td>88,234</td><td>1,406</td><td>N.A.</td></tr></table>

# F RANKING-BASED INDIVIDUAL FAIRNESS IN REAL WORLDS

![](images/ea273b9d30aa00f3b08234ea8de2a7e0026615512a50b8ab24c497b511e28574.jpg)

<details>
<summary>flowchart</summary>

```mermaid
```mermaid
graph TD
    subgraph "Ranking list from S_Y"
        A1["u1"] --> B1["u2"] --> C1["u3"] --> D1["u4"] --> E1["u5"]
        A2["u2"] --> B2["u3"] --> C2["u4"] --> D2["u5"]
        A3["u2"] --> B3["u4"] --> C3["u5"]
        A4["u2"] --> B4["u5"] --> C4["u6"] --> D4["u7"]
        A5["u2"] --> B5["u5"] --> C5["u6"] --> D5["u7"]
        A6["u2"] --> B6["u5"] --> C6["u7"] --> D6["u8"]
        A7["u2"] --> B7["u5"] --> C7["u8"] --> D7["u9"]
        A8["u2"] --> B8["u5"] --> C8["u9"] --> D8["u10"]
        A9["u2"] --> B9["u5"] --> C9["u11"] --> D10["u12"]
        A10["u2"] --> B10["u4"] --> C10["u13"] --> D11["u14"]
        A11["u2"] --> B12["u5"] --> C12["u15"] --> D13["u16"]
        A12["u2"] --> B13["u4"] --> C13["u17"] --> D14["u18"]
        A13["u2"] --> B14["u5"] --> C14["u19"] --> D15["u20"]
        A14["u2"] --> B15["u5"] --> C15["u21"] --> D16["u22"]
        A15["u2"] --> B16["u5"] --> C16["u23"] --> D17["u24"]
        A16["u2"] --> B17["u5"] --> C17["u25"] --> D18["u26"]
        A17["u2"] --> B18["u5"] --> C18["u27"] --> D19["u30"]
        A18["u2"] --> B19["u5"] --> C19["u30"]
        A19["u3"] --> B20["u5"] --> C20["u31"]
        A20["u3"] --> B21["u5"] --> C21["u32"]
        A21["u3"] --> B22["u5"] --> C22["u33"]
        A22["u3"] --> B23["u5"] --> C23["u34"]
        A23["u3"] --> B24["u5"] --> C24["u35"]
        A24["u3"] --> B25["u5"] --> C25["u36"]
        A25["u3"] --> B26["u5"] --> C26["u37"]
        A26["u3"] --> B27["u5"] --> C27["u38"]
        A27["u3"] --> B28["u5"] --> C28["u39"]
        A28["u3"] --> B29["u5"] --> C29["u40"]
        A29["u3"] --> B30["u5"] --> C30["u41"]
        A30["u3"] --> B31["u5"] --> C31["u42"]
        A31["u3"] --> B32["u5"] --> C32["u43"]
        A32["u3"] --> B33["u5"] --> C33["u44"]
        A33["u3"] --> B34["u5"] --> C34["u45"]
        A34["u3"] --> B35["u5"] --> C35["u46"]
        A35["u3"] --> B36["u5"] --> C36["u47"]
        A36["u3"] --> B37["u5"] --> C37["u48"]
        A37["u3"] --> B38["u5"] --> C38["u49"]
        A38["u3"] --> B39["u5"] --> C39["u50"]
        A39["u3"] --> B40["u5"] --> C40["u51"]
        A40["u3"] --> B41["u5"] --> C41["u52"]
        A41["u3"] --> B42["u5"] --> C42["u53"]
        A42["u3"] --> B43["u5"] --> C43["u54"]
        A43["u3"] --> B44["u5"] --> C44["u55"]
        A44["u3"] --> B45["u5"] --> C45["u56"]
        A45["u3"] --> B46["u5"] --> C46["u57"]
        A46["u3"] --> B47["u5"] --> C47["u58"]
        A47["u3"] --> B48["u5"] --> C48["u59"]
        A48["u3"] --> B49["u5"] --> C49["u60"]
        A49["u3"] --> B50["u5"] --> C50["u61"]
        A50["u3"] --> B51["u5"] --> C51["u62"]
        A51["u3"] --> B52["u5"] --> C52["u63"]
        A52["u3"] --> B53["u5"] --> C53["u64"]
        A53["u3"] --> B54["u5"] --> C54["u65"]
        A54["u3"] --> B55["u5"] --> C55["u66"]
        A55["u3"] --> B56["u5"] --> C56["u67"]
        A56["u3"] --> B57["u5"] --> C57["u68"]
        A57["u3"] --> B58["u5"] --> C58["u69"]
        A58["Inconsistent node pairs"]
    end
    subgraph "Ranking list from S_G"
        B1["Bonded nodes with labels u1, u2, ..., uN, ..."]
```
</details>

Figure 2: An example figure motivating individual fairness in graphs; taken from Dong et al. (2021).

Motivating Example Consider a job recommendation platform utilizing a GNN to rank candidates for employers. In this scenario, nodes symbolize individuals, while edges represent connections or similarities (such as shared skills, experience, or educational background). Figure 2, by Dong et al. (2021) as our motivating example, showcases two distinct ranking lists for different candidate subsets, $S_{y}$ and $S_{g}$ , based on their profiles and network connections.

In an ideally fair system, candidates possessing comparable qualifications and experiences should receive consistent rankings across various subsets. However, our example reveals disparities (indicated by red dotted lines) that arise without the implementation of our proposed framework. These disparities manifest as inconsistent rankings for similar candidates, resulting in potential unfairness. For instance, the ranking of candidate $u_{2}$ is higher than $u_{4}$ in the list from $S_{y}$ , but this order reverses in the list from $S_{g}$ . Such inconsistency can lead to overlooking qualified candidates based on the subset they belong to, challenging the notion of individual fairness.

Our framework employs the Lipschitz condition to establish a boundary, ensuring that the output differences (rankings) for any pair of similar candidates remain within the limits set by their feature-space distances. This mechanism acts as a safeguard, preserving ranking consistency across different scenarios and thereby maintaining individual fairness.

Practical Value The real-world significance of our approach is its capacity to generate fair and consistent outcomes in various applications where GNNs are used for critical decision-making. This includes domains like credit scoring, social media content ranking, and personalized medicine, where inconsistent results can significantly affect individuals' opportunities and well-being. Our objective is to cultivate fairness and trust in AI systems, ensuring they adhere to societal and ethical standards.

In conclusion, our approach is not only theoretically robust but also holds substantial practical merit. It provides a significant benefit to both individuals impacted by GNN-based decisions and the organizations implementing these models. We believe the provided motivating example effectively demonstrates the potential influence and real-world relevance of our research.