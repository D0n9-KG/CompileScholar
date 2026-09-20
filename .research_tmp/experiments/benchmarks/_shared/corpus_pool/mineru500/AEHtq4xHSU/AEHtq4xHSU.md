# Towards characterizing the value of edge embeddings in Graph Neural Networks

<table><tr><td>Dhruv Rohatgi</td><td>Tanya Marwah</td><td>Zachary Chase Lipton</td></tr><tr><td>MIT</td><td>CMU</td><td>CMU</td></tr><tr><td>drohatgi@mit.edu</td><td>tmarwah@andrew.cmu.edu</td><td>zlipton@cmu.edu</td></tr><tr><td>Jianfeng Lu</td><td>Ankur Moitra</td><td>Andrej Risteski</td></tr><tr><td>Duke University</td><td>MIT</td><td>CMU</td></tr><tr><td>jianfeng@math.duke.edu</td><td>moitra@mit.edu</td><td>aristesk@andrew.cmu.edu</td></tr><tr><td></td><td>October 15, 2024</td><td></td></tr></table>

# Abstract

Graph neural networks (GNNs) are the dominant approach to solving machine learning problems defined over graphs. Despite much theoretical and empirical work in recent years, our understanding of finer-grained aspects of architectural design for GNNs remains impoverished. In this paper, we consider the benefits of architectures that maintain and update edge embeddings. On the theoretical front, under a suitable computational abstraction for a layer in the model, as well as memory constraints on the embeddings, we show that there are natural tasks on graphical models for which architectures leveraging edge embeddings can be much shallower. Our techniques are inspired by results on time-space tradeoffs in theoretical computer science. Empirically, we show architectures that maintain edge embeddings almost always improve on their node-based counterparts—frequently significantly so in topologies that have “hub” nodes.

# 1 Introduction

Graph neural networks (GNNs) have emerged as the dominant approach for solving machine learning tasks on graphs. Over the span of the last decade, many different architectures have been proposed, both in order to improve different notions of efficiency, and to improve performance on a variety of benchmarks. Nevertheless, theoretical and empirical understanding of the impact of different architectural design choices remains elusive.

One previous line of work (Xu et al., 2018) has focused on characterizing the representational limitations stemming from the symmetry-preserving properties of GNNs when the node features are not informative (also called “anonymous GNNs”) — in particular, relating GNNs to the Weisfeiler-Lehman graph isomorphism test (Leman & Weisfeiler, 1968). Another line of work (Oono & Suzuki, 2019) focuses on the potential pitfalls of the (over)smoothing effect of deep GNN architectures, with particular choices of weights and non-linearities, in an effort to explain the difficulties of training deep GNN models. Yet another (Black et al., 2023) focuses on training difficulties akin to vanishing introduced by “bottlenecks” in the graph topology.

In this paper, we focus on the benefits of maintaining and updating edge embeddings over the course of the computation of the GNN. More concretely, a typical way to parametrize a layer l of a GNN (Xu et al., 2018) is to maintain, for each node v in the graph, a node embedding $h_{v}^{(l)}$ , which is

calculated as

$$
a _ {v} ^ {(l + 1)} = \operatorname{AGGREGATE} \left(h _ {u} ^ {(l)}: u \in N _ {G} (v)\right) \quad h _ {v} ^ {(l + 1)} = \operatorname{COMBINE} \left(a _ {v} ^ {(l + 1)}, h _ {v} ^ {(l)}\right) \tag {1}
$$

where $N_{G}(v)$ denotes the neighborhood of vertex v. These updates can be viewed as implementing a (trained) message-passing algorithm, in which nodes pass messages to their neighbors, which are then aggregated and combined with the current state (i.e., embedding) of a node. The initial node embeddings $h_{v}^{(0)}$ are frequently part of the task specification (e.g., a vector of fixed features that can be associated with each node). When this is not the case, they can be set to fixed values (e.g., the all-ones vector) or random values.

But a more expressive way to parametrize a layer of computation is to maintain, for each edge e, an edge embedding $h_{e}^{(l)}$ which is calculated as:

$$
a _ {e} ^ {(l + 1)} = \mathrm{AGGREGATE} \Big (h _ {a} ^ {(l)}: a \in M _ {G} (e) \Big) \qquad h _ {e} ^ {(l + 1)} = \mathrm{COMBINE} \Big (a _ {e} ^ {(l + 1)}, h _ {e} ^ {(l)} \Big) \tag {2}
$$

where $M_{G}(e)$ denotes the “neighborhood” of edge e: that is, all edges a that share a vertex with $e^{1}$ .

This paradigm is at least as expressive as (1): we can simulate a layer of (1) by designating the embedding of an edge to be the concatenation of the node embeddings of its endpoints, and noticing that $M_{G}(e)$ includes all the neighbors of both endpoints of e. In particular, if a task has natural initial node embeddings, then their concatenations along edges can be used as initial edge embeddings. Additionally, there may be tasks where initial features are most naturally associated with edges (e.g., attributes of the relationship between two nodes) — or the final predictions of the network are most naturally associated with edges (e.g., in link prediction, where we want to decide which potential links are true links).

GNNs that fall in the general paradigm of (2) have been used for various applications – including link prediction (Cai et al., 2021; Liang & Pu, 2023) as well as reasoning about relations between objects (Battaglia et al., 2016), molecular property prediction (Gilmer et al., 2017; Choudhary & DeCost, 2021), and detecting clusters of communities in graphs (Chen et al., 2017) – with robust empirical benefits. These approaches instantiate the edge-based paradigm in a plethora of ways. However, it is difficult to disentangle to what degree performance improvements come from added information from domain-specific initial edge embeddings, versus properties of the particular architectural choices for the aggregation functions in (2), versus inherent benefits of the edge-based paradigm itself (whether representational, or via improved training dynamics).

We focus on theoretically and empirically quantifying the added representational benefit from maintaining edge embeddings. Viewing the GNN as a computational model, we can think of the intermediate embeddings as a “scratch pad”. Since we maintain more information per layer compared to the node-based paradigm (1), we might intuitively hope to be able to use a shallower edge embedding model. However, formally proving depth lower bounds both for general neural networks (Telgarsky, 2016) and for specific architectures (Sanford et al., 2024b,a) frequently requires non-trivial theoretical insights – as is the case for our question of interest. In this paper, we show that:

\- Theoretically, for certain graph topologies, edge embeddings can have substantial representational benefits in terms of the depth of the model, when the amount of memory (i.e., total bit complexity) per node or edge embedding is bounded. Our results illuminate some subtleties of using particular lenses to understand design aspects of GNNs: for instance, we prove that taking memory into account reveals depth separations that the classical lens of invariance (Xu et al., 2018) alone cannot.

\- Empirically, when given the same input information, edge-based models almost always lead to performance improvements compared to their node-based counterpart — and often by a large margin if the graph topology includes “hub” nodes with high degree.

# 2 Overview of results

# 2.1 Representational benefits from maintaining edge embeddings.

Our theoretical results elucidate the representational benefits of maintaining edge embeddings. More precisely, we show that there are natural tasks on graphs that can be solved by a shallow model maintaining constant-size edge embeddings, but can only be solved by a model maintaining constant-size node embeddings if it is much deeper.

To reason about the impact of depth on the representational power of edge-embedding-based and node-embedding-based architectures, we introduce two local computation models. In the node-embedding case, we assume each node of the graph G supports a processor that maintains a state with a fixed amount of memory. In one round of computation, each node receives messages from the adjacent nodes, which are aggregated by the node into a new state. In this abstraction, we think of the memory of the processor as the total bits of information each embedding can retain, and we think of one round of the protocol as corresponding to one layer of a GNN. The edge-embedding case is formalized in a similar fashion, except that the processors are placed on the edges of the graph, and two edge processors are “adjacent” if the edges share a vertex in common. In both cases, the input is distributed across the edges of the graph, and is only locally accessible.

With this setup in mind, our first result focuses on probabilistic inference on graphs, specifically, the task of maximum a-posteriori (MAP) estimation in a pairwise graphical model on a graph $G = (V, E)$ . For this task, given edge attributes describing the pairwise interactions $\phi_{\{a,b\}}$ , the goal is to compute $\arg\max_{x \in \{0,1\}^{V}} p_{\phi}(x)$ , where $p_{\phi}(x) \propto \exp\bigl(\sum_{\{a,b\} \in E} \phi_{\{a,b\}}(x_{a}, x_{b})\bigr)$ .

Theorem (Informal). Consider the task of using a GNN to calculate MAP (maximum a-posteriori) values in a pairwise graphical model, in which the pairwise interactions are given as input embeddings to a node-embedding or edge-embedding architecture. Then, there exists a graph with $O(n)$ vertices and edges, such that:

- Any node message-passing protocol with $T$ rounds and $O(1)$ bits of memory per node processor requires $T = \Omega(\sqrt{n})$ .   
- There is an edge message-passing protocol with $O(1)$ rounds and $O(1)$ bits of memory.

The proof techniques are of standalone interest: the lower bound on node message-passing protocols is inspired by tracking the “flow of information” in the graph, reminiscent of graph pebbling techniques used to prove time-space tradeoffs in theoretical computer science (Grigor’ev, 1976; Abrahamson, 1991). The formal result is Theorem 1, and the proof sketch is included in Section 5.

The view from symmetry. Above, we are not imposing any symmetry constraints – that is, invariance of the computation at a node or edge to its identity and the identities of its neighbors. Indeed, the edge message-passing protocol constructed above is highly non-symmetric. However, we show there is a (different, but also natural) task where even symmetric edge message-passing protocols achieve a better depth/memory tradeoff than node message-passing protocols. We state the informal result below; the formal result is Theorem 4.

Theorem (Informal). Let n be a positive integer. There is a graph G with $O(n)$ vertices and $O(n)$ edges, and a computational task on G, such that:

- Any node message-passing protocol with $T$ rounds and $O(1)$ bits of memory per node processor requires $T = \Omega(\sqrt{n})$ to solve this task.   
- There is a symmetric edge message-passing protocol that solves this task with $O(1)$ rounds and $O(1)$ bits of memory.

Importance of the memory lens. The memory constraints are crucial for the results above. Without memory constraints, we can show that the node message-passing architecture can simulate the edge message-passing architecture, while only increasing the depth by 1 (Proposition 3). Moreover, the symmetric node message-passing architecture can simulate the symmetric edge message-passing architecture, again while only increasing the depth by 1. We state the informal result below; the formal result is Theorem 7.

Theorem (Informal). For any graph G, any symmetric edge message-passing protocol on G with T rounds can be represented by a symmetric node message-passing protocol with $T + 1$ rounds.

We note that unlike prior work that focuses on understanding the representational power of GNN architectures under symmetry constraints (Xu et al., 2018) — which requires that the initial node features are the same for all nodes — our simulation theorem above holds for arbitrary choices of initial node features.

We view this as evidence that many fine-grained properties of architectural design for GNNs cannot be adjudicated by solely considering them through the lens of symmetries of the network.

# 2.2 Empirical benefits of edge-based architectures.

The theory, while only characterizing representational power, suggests that architectures that maintain edge embeddings should have strictly better performance compared to their node embedding counterparts. We verify this in both real-life benchmarks and natural synthetic sandboxes.

First, we consider several popular GNN benchmarks (inspired by both predicting molecular properties, and image-like data), and show that equalizing for all other aspects of the architecture (e.g., depth, dimensionality of the embeddings) — the accuracy the edge-based architectures achieve is at least as good as their node-based counterparts. Note, the goal of these experiments is not to propose a new architecture — there are already a variety of (very computationally efficient) GNNs that in some manner maintain edge embeddings. The goal is to confirm that — all other things being equal — the representational advantages of edge-based architectures do not introduce additional training difficulties. Details are included in Section 8.1.

Next, we consider two synthetic settings to stress test the performance of edge-based architectures. Inspired by the graph topology that provides a theoretical separation between edge and node-based protocols (Theorem 1 and Theorem 4), we consider graphs in which there is a hub node, and tasks that are “naturally” solved by an edge-based architecture. Precisely, we consider a star graph, in which the labels on the leaves are generated by a “planted” edge-based architecture with randomly chosen weights. The node-based architecture, on the other hand, has to pass messages between the leaves indirectly through the center of the star. Empirically, we indeed observe that the performance of edge-based architectures is significantly better. Details are included in Section 8.2.

Finally, again inspired by the theoretical setting in Theorem 1, we consider probabilistic inference on tree graphs — precisely, learning a GNN that calculates node marginals for an Ising model, a pairwise graphical model in which the pairwise interactions are just the product of the end points. An added motivation for this setting is the fact that belief propagation — a natural algorithm to calculate the marginals — can be written as an edge-based message-passing algorithm. Again,

empirically we see that edge-based architectures perform at least as well as node-based architectures. This advantage is maintained even if we consider “directed” versions of both architectures, in which case embeddings are maintained to be sent along each direction of the edge, and the message for the outgoing direction of an edge depends only on the embeddings corresponding to the incoming directions of the edges. Details are included in Section 8.3.

# 3 Related Works

The symmetry lens on GNNs: The most extensive theoretical work on GNNs has concerned itself with the representational power of different GNN architectures, while trying to preserve equivariance (to permuting the neighbors) of each layer. (Xu et al., 2018) connected the expressive power of such architectures to the Weisfeiler-Lehman (WL) test for graph isomorphism. Subsequent works (Maron et al., 2019; Zhao et al., 2021) focused on strengthening the representational power of the standard GNN architectures from the perspective of symmetries—more precisely, to simulate the k-WL test, which for k as large as the size of the graph becomes as powerful as testing graph isomorphism. Our work suggests that this perspective may be insufficient to fully understand the representational power of different architectures.

GNNs as a computational machine: Two recent papers (Loukas, 2019, 2020) considered properties of GNNs when viewed as “local computation” machines, in which a layer of computation allows a node to aggregate the current values of the neighbors (in an arbitrary fashion, without necessarily considering symmetries). Using reductions from the CONGEST model, they provide lower bounds on width and depth for the standard node-embedding based architecture. However, they do not consider architectures with edge embeddings, which is a focus of our work.

Communication complexity methods to prove representational separations: Tools from distributed computation and communication complexity have recently been applied not only to understand the representational power of GNNs (Loukas, 2019, 2020), but also the representational power of other architectures like transformers (Sanford et al., 2024b,a). In particular, (Sanford et al., 2024a) draws a connection between number of rounds for a MPC (Massively Parallel Computation) protocol, and the depth of attention-based architectures.

GNNs for inference and graphical models: The paper (Xu & Zou, 2023) considers the approximation power of GNNs for calculating marginals for pairwise graphical models, if the family of potentials satisfies strong symmetry constraints. They do not consider the role of edge embeddings or memory.

# 4 Setup

Notation. We will denote the graph associated with the GNN as $G = (V, E)$ , denoting the vertex set as V and the edge set as E. The graph induces adjacency relations on both edges and nodes, namely for $v, v' \in V$ and $e, e' \in E$ , we have: $v \sim v'$ if $\{v, v'\} \in E$ ; $v \sim e$ if $e = \{u, v\}$ for some $u \in V$ ; and $e \sim e'$ if $e, e'$ share at least one vertex. For all graphs considered in this paper, we assume that $\{v, v\} \in E$ for all $v \in V$ , so that adjacency is reflexive. We then define adjacency functions $N_G : V \cup E \to V$ and $M_G : V \cup E \to E$ as $N_G(a) := \{v \in V : a \sim v\}$ and $M_G(a) := \{e \in E : a \sim e\}$ .

Local memory-constrained computation. In order to reason about the required depth with different architectures, we will define a mathematical abstraction for one layer of computation in the GNN. We will define two models for local computation, one for each of the edge-embedding and node-embedding architecture. Unlike much prior work on GNNs and distributed computation, we will also have memory constraints — more precisely, we will constrain the bit complexity of the node and edge embeddings being maintained.

In both models, there is an underlying graph $G = (V, E)$ , and the goal is to compute a function $g : \Phi^{E} \to \{0, 1\}^{V}$ , where $\Phi$ is the fixed-size input alphabet, via several rounds of message-passing on the graph G. This domain of g is $\Phi^{E}$ because in both models, the inputs are given on the edges of the graph — the node model will just be unable to store any additional information on the edges. As we will see in Section 5, this is a natural setup for probabilistic inference on graphs.

In both models, a protocol is parametrized by the number of rounds T required, and the amount of memory B required per local processor. For notational convenience, for $B \in N$ we define $X_{B} := \{0, 1\}^{B}$ , i.e. the length-B binary strings. Recall that $N_{G}(v)$ , $M_{G}(v)$ denote the sets of vertices and edges adjacent to vertex v in graph G, respectively.

Definition 1 (Node message-passing protocol). Let $T, B \in \mathbb{N}$ and let $G = (V, E)$ be a graph. A node message-passing protocol $P$ on graph $G$ with $T$ rounds and $B$ bits of memory is a collection of functions $(f_{t,v})_{t \in [T], v \in V}$ where $f_{t,v}: \mathcal{X}_B^{N_G(v)} \times \Phi^{M_G(v)} \to \mathcal{X}_B$ for all $t, v$ . For an input $I \in \Phi^E$ , the computation of $P$ at a round $t \in [T]$ is the map $P_t(\cdot; I): V \to \mathcal{X}_B$ defined inductively by

$$
P _ {t} (v; I) := f _ {t, v} ((P _ {t - 1} (v ^ {\prime}; I)) _ {v ^ {\prime} \in N _ {G} (v)}, (I (e)) _ {e \in M _ {G} (v)})
$$

where $P_{0} \equiv 0$ . We say that P computes a function $g : \Phi^{E} \to \{0,1\}^{V}$ on inputs $I \subseteq \Phi^{E}$ if $P_{T}(v;I)_{1} = g(I)_{v}$ for all $v \in V$ and all $I \in I$ .

In words, the value computed by vertex v at round t is some function of the previous values stored at the neighbors $v' \in N_G(v)$ , as well as possibly the problem inputs on the edges adjacent to v (i.e. $(I(e))_{e \in M_G(v)}$ ). Note that $P_t(v; I)$ may indeed depend on $P_{t-1}(v; I)$ , due to our convention that $v \in N_G(v)$ . We can define the edge message-passing protocol analogously:

Definition 2 (Edge message-passing protocol). Let $T, B \in \mathbb{N}$ and let $G = (V, E)$ be a graph. An edge message-passing protocol $P$ on graph $G$ with $T$ rounds and $B$ bits of memory is a collection of functions $(f_{t,e})_{t \in [T], e \in E}$ where $f_{t,e}: \mathcal{X}_B^{M_G(e)} \times \Phi \to \mathcal{X}_B$ for all $t, e$ , together with a collection of functions $(\tilde{f}_v)_{v \in [V]}$ where $\tilde{f}_v: \mathcal{X}_B^{M_G(v)} \to \{0,1\}$ . For an input $I \in \Phi^E$ , the computation of $P$ at a timestep $t \in [T]$ is the map $P_t(\cdot; I): E \to \mathcal{X}_B$ defined inductively by:

$$
P _ {t} (e; I) := f _ {t, e} ((P _ {t - 1} (e ^ {\prime}; I)) _ {e ^ {\prime} \in M _ {G} (e)}, I (e))
$$

where $P_{0} \equiv 0$ . We say that P computes a function $g : \Phi^{E} \to \{0,1\}^{V}$ on inputs $I \subseteq \Phi^{E}$ if $\tilde{f}_{v}((P_{T}(e;I))_{e \in M_{G}(v)}) = g(I)_{v}$ for all $v \in V$ and all $I \in I$ .

Remark 3 (Relation to distributed computation literature). These models are very related to classical models in distributed computation like LOCAL (Linial, 1992) and CONGEST (Peleg, 2000). However, the latter models ignore memory constraints, so we cannot usefully port lower and upper bounds from this literature.

Remark 4 (Computational efficiency). In the definitions above, we allow the update rules $f_{t,v}, f_{t,e}$ to be arbitrary functions. In particular, a priori they may not be efficiently computable. However, our results showing a function can be implemented by an edge message-passing protocol (Theorem 1,

Part 2 and Theorem 4, Part 2) in fact use simple functions (computable in linear time in the size of the neighborhood), implying the protocol can be implemented in parallel (with one processor per node/edge respectively) with parallel time complexity $O(TB \cdot \max_v |M_G(v)|)$ . On the other hand, for the results showing a function cannot be implemented by a node message-passing protocol (Theorem 1, Part 1 and Theorem 4, Part 1), we prove an impossibility result for a stronger model (one in which the computational complexity of $f_{t,v}$ is unrestricted) — which makes our results only stronger.

Symmetry-constrained protocols. Typically, GNNs are architecturally constrained to respect the symmetries of the underlying graph. Below we formalize the most natural notion of symmetry in our models of computation. Note, our abstraction of a round in the message-passing protocol generalizes the notion of a layer in a graph neural network—and the abstraction defined below correspondingly generalizes the standard definition of permutation equivariance (Xu et al., 2018). We use the notation $\{\{\}\}$ to denote a multiset.

Definition 5 (Symmetric node message-passing protocol). A node message-passing protocol $P = (f_{t,v})_{t \in [T],v \in V}$ on graph $G = (V,E)$ is symmetric if there are functions $(f_t^{\mathbf{sym}})_{t \in [T]}$ so that for every $t \in [T]$ and $v \in V$ , the function $f_{t,v}$ can be written as:

$$
f _ {t, v} ((c (v ^ {\prime})) _ {v ^ {\prime} \in N _ {G} (v)}, (I (e)) _ {e \in M _ {G} (v)}) = f _ {t} ^ {\mathsf {s y m}} (c (v), \{\{(c (v ^ {\prime}), I (\{v, v ^ {\prime} \})): v ^ {\prime} \in N _ {G} (v) \} \}).
$$

Definition 6 (Symmetric edge message-passing protocol). An edge message-passing protocol $P = ((f_{t,e})_{t\in [T],e\in E},(\tilde{f}_v)_{v\in V})$ on graph $G = (V,E)$ is symmetric if there are functions $(f_t^{\mathrm{sym}})_{t\in [T]}$ and $\tilde{f}^{\mathrm{sym}}$ so that for every $t\in [T]$ and $e = \{u,v\} \in E$ , the function $f_{t,e}$ can be written as:

$$
f _ {t, e} ((c (e ^ {\prime})) _ {e ^ {\prime} \in M _ {G} (e)}, I (e)) = f _ {t} ^ {\mathsf {s y m}} (I (e), c (e), \{\{\{c (\{u, v ^ {\prime} \}): v ^ {\prime} \in N _ {G} (u) \}, \{c (\{u ^ {\prime}, v \}): u ^ {\prime} \in N _ {G} (v) \} \} \}),
$$

and for every $v \in V$ , $\tilde{f}_v$ can be written as $\tilde{f}_v((c(e))_{e \in M_G(v)}) = \tilde{f}^{\mathrm{sym}}(\{\{c(e): e \in M_G(v)\}\})$ .

# 5 Depth separation between edge and node message passing protocols under memory constraints

We will consider a common task in probabilistic inference on a pairwise graphical model: calculating the MAP (maximum a-posterior) configuration.

Definition 7 (Pairwise graphical model). For any graph $G = (V, E)$ , the pairwise graphical model on $G$ with potential functions $\phi_{\{a,b\}} : \{0,1\}^2 \to \mathbb{R}$ is the distribution $p_\phi \in \Delta(\{0,1\}^V)$ defined as

$$
p _ {\phi} (x) \propto \exp \left(- \sum_ {\{a, b \} \in E} \phi_ {\{a, b \}} (x _ {a}, x _ {b})\right).
$$

Definition 8 (MAP evaluation). Let $\Phi \subseteq \{\phi : \{0,1\}^2 \to \mathbb{R}\}$ be a finite set of potential functions. A MAP (maximum $a$ -posteriori) evaluator for $G$ (with potential function class $\Phi$ ) is any function $g: \Phi^E \to \{0,1\}^V$ that satisfies

$$
g (\phi) \in \operatorname * {a r g   m a x} _ {x \in \{0, 1 \} ^ {V}} p _ {\phi} (x)
$$

for all $\phi \in \Phi^{E}$ .

With this setup in mind, we will show that there exists a pairwise graphical model, and a local function class $\Phi$ , such that an edge message passing protocol can implement MAP evaluation with a constant number of rounds and a constant amount of memory, while any node message protocol with T rounds and B bits of memory requires $TB = \Omega(\sqrt{|V|})$ . Precisely, we show:

Theorem 1 (Main, separation between node and edge message-passing protocols). Fix $n \in \mathbb{N}$ . There is a graph $G$ with $O(n)$ vertices and $O(n)$ edges, and a function class $\Phi$ of size $O(1)$ , so that:

1. Let g be any MAP evaluator for G with potential function class $\Phi$ . Any node message-passing protocol on G with T rounds and B bits of memory that computes g requires $TB \geq \sqrt{n} - 1$ .   
2. There is an edge message-passing protocol $(f_{t,e})_{t,e}$ on $G$ with $O(1)$ rounds and $O(1)$ bits of memory that computes a MAP evaluator for $G$ with potential function class $\Phi$ . Additionally, for all $t,e$ , the update rule $f_{t,e}$ can be evaluated in $O(|M_G(e)|)$ time.

We provide a proof sketch of the main techniques here, and relegate the full proofs to Appendix A. The graph G that exhibits the claimed separation is a disjoint union of $\sqrt{n}$ path graphs, with an additional “hub vertex” that is connected to all other vertices in the graph (Fig. 1). The intuition for the separation is that MAP estimation requires information to disseminate from one end of each path to the other, and the hub node is a bottleneck for node message-passing but not edge message-passing. We expand upon both aspects of this intuition below.

Lower bound for node message-passing protocols: Our main technical lemma for the first half of the theorem is Lemma 2. It gives a generic framework for lower bounding the complexity of any node message-passing protocol that computes some function g, by exhibiting a set of nodes $S \subset V$ where computing g requires large “information flow” from distant nodes. More precisely, for any fixed set of “bottleneck nodes” K, consider the radius-T neighborhood of S when K is removed from the graph. In any T-round protocol, input data from outside this neighborhood can only reach S by passing through K. But the total number of bits of information computed by K throughout the protocol is only $TB|K|$ . This gives a bound on the number of values achievable by g on S. We formalize this argument below:

Lemma 2. Let $G = (V, E)$ be a graph. Let $P$ be a node message-passing protocol on $G$ with $T$ rounds and $B$ bits of memory, which computes a function $g: \Phi^E \to \{0, 1\}^V$ . Pick any disjoint sets $K, S \subseteq V$ . Define $H := G[\bar{K}], F := M_G(N_H^{T-1}(S))$ . Then:

$$
T B \geq \frac {1}{| K |} \log \max _ {I _ {F} \in \Phi^ {F}} \left| \left\{g _ {S} (I _ {F}, I _ {\overline {{F}}}): I _ {\overline {{F}}} \in \Phi^ {\overline {{F}}} \right\} \right|.
$$

Proof. First, we argue by induction that for each $t \in [T]$ and $v \in V \setminus K$ , $P_t(v; I)$ is determined by $I_{M_G(N_H^{t-1}(v))}$ and $(P_\ell(k; I))_{\ell \in [t], k \in K}$ . Indeed, by definition, $P_1(v; I)$ is determined by $I_{M_G^1(v)}$ for any $v \in V \setminus K$ . For any $t > 1$ and $v \in V \setminus K$ , $P_t(v; I)$ is determined by $(P_{t-1}(v'; I))_{v' \in N_G(v)}$ and $(I(e))_{e \in M_G(v)}$ . Note that $N_G(v) \subseteq N_H(v) \cup K$ . Thus, using the induction hypothesis for each $v' \in N_H(v)$ , we get that $(P_{t-1}(v'; I))_{v' \in N_G(v)}$ is determined by $\bigcup_{v' \in N_H(v)} I_{M_G(N_H^{t-2}(v'))}$ and $(P_\ell(k; I))_{\ell \in [t], k \in K}$ . So $P_t(v; I)$ is determined by $I_{M_G(N_H^{t-1}(v))}$ and $(P_\ell(k; I))_{\ell \in [t], k \in K}$ , completing the induction.

Since $P$ computes $g$ and $S \subseteq V \setminus K$ , we get that $g_S(I)$ is determined by $I_{M_G(N_H^{T-1}(S))} = I_F$ and $(P_\ell(k;I))_{\ell \in [T],k \in K}$ . Thus, for any fixed $I_F \in \Phi_F$ , we have

$$
\left| \left\{g _ {S} \left(I _ {F}, I _ {\overline {{F}}}\right): I _ {\overline {{F}}} \in \Phi^ {\overline {{F}}} \right\} \right| \leq \left| \left\{(P _ {\ell} (k; (I _ {F}, I _ {\overline {{F}}}))) _ {\ell \in [ T ], k \in K}): I _ {\overline {{F}}} \in \Phi^ {\overline {{F}}} \right\} \right| \leq | \mathcal {X} _ {B} | ^ {T | K |} = 2 ^ {T B | K |}.
$$

The lemma follows.

Remark 9. The proof technique is inspired by and related to classic techniques (specifically, Grigoriev's method) for proving time-space tradeoffs for restricted models of computation like branching programs ((Grigor'ev, 1976), see Chapter 10 in Savage (1998) for a survey). There, one defines the "flow" of a function, which quantifies the existence of subsets of coordinates, such that setting them to some value, and varying the remaining variables results in many possible outputs. In our case, the choice of subsets is inherently tied to the topology of the graph G. Our technique is also inspired by and closely related to the "light cone" technique for proving round lower bounds in the LOCAL computation model (Linial, 1992). However, our technique takes advantage of bottlenecks in the graph to prove stronger lower bounds (which would be impossible in the LOCAL model where memory constraints are ignored).

The proof of Part 1 of Theorem 1 now follows from an application of Lemma 2 with a particular choice of K and S. Specifically, we choose K to be the “hub” node (i.e. $K = \{0\}$ ) and S to be the set of left endpoints of each path. To show that any MAP evaluator has large information flow to S (in the quantitative sense of Lemma 2), it suffices to observe that in a pairwise graphical model on G where a different external field is applied to the right endpoint of each path, and all pairwise interactions along paths are positive, the MAP estimate on each vertex in S must match the external field on the corresponding right endpoint.

Upper bound for edge message-passing protocols: The key observation for constructing a constant-round edge message-passing protocol for MAP estimation on G is that all of the input data can be collected on the edges adjacent to the hub vertex. At this point, every such edge has access to all of the input data, and hence can evaluate the function. If G were an arbitrary graph, this final step would potentially be NP-hard. However, since the induced subgraph after removing the hub vertex is a disjoint union of paths, in fact there is a linear-time dynamic programming algorithm for MAP estimation on G (Lemma 8). This completes the proof overview for Theorem 1; we now provide the formal proof.

Proof of Theorem 1. Let G be the graph on vertex set $V := \{0\} \cup [\sqrt{n}] \times [\sqrt{n}]$ with edge set defined below (see also Fig. 1):

$$
E := \{\{0, (i, j) \}: i, j \in [ \sqrt {n} ] \} \cup \{\{(i, j), (i + 1, j) \}: 2 \leq i \leq \sqrt {n}, 1 \leq j \leq \sqrt {n} \}.
$$

Define

$$
\Phi := \{(x _ {a}, x _ {b}) \mapsto \mathbb {1} [ x _ {a} \neq x _ {b} ], (x _ {a}, x _ {b}) \mapsto \mathbb {1} [ x _ {a} \neq 1 \lor x _ {b} \neq 1 ], (x _ {a}, x _ {b}) \mapsto \mathbb {1} [ x _ {a} \neq 0 \lor x _ {b} \neq 0 ], (x _ {a}, x _ {b}) \mapsto 0 \}.
$$

First, let $g : \Phi^{E} \to \{0,1\}^{V}$ be any MAP evaluator for G with potential function class $\Phi$ , and consider any node message-passing protocol on G with T rounds and B bits of memory that computes g. Let $K = \{0\}$ and $S = \{(1,j) : j \in [\sqrt{n}]\}$ . Suppose that $T \leq \sqrt{n} - 2$ . Let $F := M_{G}(N_{H}^{T-1}(S))$ and note that $\{(\sqrt{n} - 1,j), (\sqrt{n},j)\} \notin F$ for all $j \in [\sqrt{n}]$ . Let $I_{F} : F \to \Phi$ be the mapping that assigns the function $(x_{a}, x_{b}) \mapsto 0$ to each edge $\{0, (i,j)\} \in F$ and $(x_{a}, x_{b}) \mapsto \mathbb{1}[x_{a} \neq x_{b}]$ to each edge $\{(i,j), (i+1,j)\} \in F$ . We claim that

$$
\left| \left\{g _ {S} (I _ {F}, I _ {\overline {{F}}}): I _ {\overline {{F}}} \in \Phi^ {\overline {{F}}} \right\} \right| \geq 2 ^ {\sqrt {n}}.
$$

Indeed, for any string $y \in \{0,1\}^{\sqrt{n}}$ , consider the mapping $I_{\overline{F}}: \overline{F} \to \Phi$ that assigns the function $(x_a, x_b) \mapsto \mathbb{1}[x_a \neq y_j \vee x_b \neq y_j]$ to each edge $\{(\sqrt{n} - 1, j), (\sqrt{n}, j)\} \in F$ , assigns $(x_a, x_b) \mapsto 0$ to each

edge $\{0,(i,j)\} \in E\setminus F$ , and assigns $(x_{a},x_{b})\mapsto \mathbb{1}[x_{a}\neq x_{b}]$ to all remaining edges in $E\setminus F$ . Then every minimizer of

$$
\min _ {x \in \{0, 1 \} ^ {V}} \sum_ {\{a, b \} \in E} I _ {\{a, b \}} (x _ {a}, x _ {b})
$$

satisfies $x_{(1,j)} = \cdots = x_{(\sqrt{n},j)} = y_j$ for all $j \in [\sqrt{n}]$ . Hence, $g_S(I_F, I_{\overline{F}}) = y$ . Since y was chosen arbitrarily, this proves the claim. But now Lemma 2 implies that $TB \geq \sqrt{n}$ .

We now construct an edge message-passing protocol $P$ on $G$ with $T = 3$ and $B = 4$ . We (arbitrarily) identify $\Phi$ with $\{0,1\}^2$ . For all $i,j\in \sqrt{n}$ , define

$$
f _ {1, \{(i, j), (i + 1, j) \}} (x, y) := y \quad \text { if } i <   \sqrt {n}
$$

$$
f _ {2, \{0, (i, j) \}} (x, y) := \left(x _ {\{(i, j), (i + 1, j) \}}, x _ {\{0, (i, j) \}}\right) \quad \text { if } i <   \sqrt {n}
$$

$$
f _ {3, \{0, (i, j) \}} (x, y) := \left(g _ {0} (J (x)), g _ {(i, j)} (J (x))\right)
$$

where the second line is well-defined since edge $\{0,(i,j)\}$ is adjacent to both itself and edge $\{(i,j),(i+1,j)\}$ ; and in the third line the function is computing $g_{0}$ and $g_{(i,j)}$ on the input $J(x)\in\Phi^{E}$ defined as

$$
J (x) _ {e} := \left\{ \begin{array}{l l} (x _ {\{0, (k, \ell) \}}) _ {1: 2} & \text {if e = \{(k,\ell),(k + 1,\ell)\}} \\ (x _ {\{0, (k, \ell) \}}) _ {3: 4} & \text {if e = \{0,(k,\ell)\}} \end{array} \right.,
$$

where we use the notation $v_{a:b}$ for a vector v and indices $a, b \in N$ to denote $(v_a, v_{a+1}, \ldots, v_b)$ . Note that $J(x)$ is a well-defined function of x for every edge $\{0, (i, j)\}$ , because $\{0, (i, j)\} \sim \{0, (k, \ell)\}$ for all $i, j, k, \ell \in [n]$ . Finally, define all other functions $f_{t,e}$ to compute the all-zero function, and define

$$
\tilde {f} _ {v} (x) := \left\{ \begin{array}{l l} (x _ {\{0, (1, 1) \}}) _ {1: 2} & \text {if v = 0} \\ (x _ {\{0, v \}}) _ {3: 4} & \text {otherwise} \end{array} \right..
$$

This function is well-defined since $v = 0$ is adjacent to edge $\{0, (1, 1)\}$ and any vertex $v \in V \setminus \{0\}$ is adjacent to edge $\{0, v\}$ .

Fix any $I \in \Phi^{E}$ . From the definition, it's clear that $P_{2}(\{0,(i,j)\};I) = (I_{\{(i,j),(i + 1,j)\}},I_{\{0,(i,j)\}})$ for all $I$ and $(i,j) \in [\sqrt{n} - 1] \times [\sqrt{n}]$ . Hence $J((P_2(e';I))_{e' \in M_G(e)})_e = I$ for all edges $e$ of the form $(0,\{i,j\})$ , and so $P_{3}(\{0,(i,j)\};I) = (g_{0}(I),g_{(i,j)}(I))$ for all $(i,j) \in [\sqrt{n}] \times [\sqrt{n}]$ . This means that $\tilde{f}_v((P_3(e;I))_{e \in M_G(v)}) = g(I)_v$ for all $v \in V$ , so the protocol indeed computes $g$ .

It remains to argue about the computational complexity of the updates $f_{t,e}$ . It's clear that for all $e \in E$ and $t \in \{1,2\}$ , the function $f_{t,e}$ can be evaluated in input-linear time. The only case that requires proof is when $t = 3$ and $e = \{0,(i,j)\}$ for some $i,j \in \sqrt{n}$ . In this case $|M_G(e)| = \Theta(n)$ , so it suffices to give an algorithm for evaluating the function $g: \Phi^E \to \{0,1\}^V$ on an explicit input $J$ in $O(n)$ time. This can be accomplished via dynamic programming (Lemma 8).

Remark 10. A quantitatively stronger (and in fact tight) separation is possible if one considers general tasks rather than MAP estimation tasks – see Appendix C.

The separation discussed above crucially relies on the existence of a high-degree vertex in G. When the maximum degree of G is bounded by some parameter $\Delta$ , it turns out that any edge message-passing protocol can be simulated by a node message-passing protocol with roughly the same number of rounds and only a $\Delta$ factor more memory per processor. The idea is for each node to simulate the computation that would have been performed (in the edge message-passing protocol) on the adjacent edges. The following proposition formalizes this idea (proof in Appendix A):

Proposition 3. Let $T, B \geq 1$ . Let $G = (V, E)$ be a graph with maximum degree $\Delta$ . Let $P$ be an edge message-passing protocol on $G$ with $T$ rounds and $B$ bits of memory. Then there is a node message-passing protocol $P'$ on $G$ that computes $P$ with $T + 1$ rounds and $O(\Delta B)$ bits of memory.

# 6 Depth separation under memory and symmetry constraints

One drawback of the separation in the previous section is that the constructed edge protocol was highly non-symmetric, whereas in practice GNN protocols are typically architecturally constrained to respect the symmetries of the underlying graph. In this section we prove that there is a separation between the memory/round trade-offs for node and edge message-passing protocols even under additional symmetry constraints.

Theorem 4. Let $n \in \mathbb{N}$ . There is a graph $G = (V, E)$ with $O(n)$ vertices and $O(n)$ edges, and a function $g: \{0,1\}^E \to \{0,1\}^V$ , so that:

1. Any node message-passing protocol on $G$ with $T$ rounds and $B$ bits of memory that computes $g$ requires $TB \geq \Omega(\sqrt{n})$ .   
2. There is a symmetric edge message-passing protocol on G with $O(1)$ rounds and $O(\log n)$ bits of memory that computes g.

For intuition, we start by sketching the proof of a relaxed version of the theorem, where the input alphabet is $[n]$ instead of $\{0,1\}$ . We then discuss how to adapt the construction to binary alphabet.

Large-alphabet construction. Let $G = (V, E)$ be a star graph with root node 0 and leaves $\{1, \ldots, n\}$ . We define a function $g : [n]^{E} \to \{0, 1\}^{V}$ by $g(I)_{v} = 1$ if and only if there is some edge $e \neq \{0, v\}$ such that $I(e) = I(\{0, v\})$ , i.e. the input on edge $\{0, v\}$ equals the input on some other edge. Since g is defined to be equivariant to relabelling the edges, and all edges are incident to each other, it is straightforward to see that there is a symmetric one-round edge message-passing protocol that computes g with $O(\log n)$ memory (in contrast, the edge message-passing protocol constructed in Section 5 was not symmetric, as it required that the edges incident to the high-degree vertex were labelled by which path they belonged to). However, there is no low-memory, low-round node message-passing algorithm. Informally, this is because vertex 0 is an information bottleneck, and $\Omega(n)$ bits of information need to pass through it. Similar to in Section 5, this intuition can be made formal using Lemma 2.

Modifying for small alphabet. The large alphabet size seems crucial to the above construction: if we were to naively modify the above construction so that each edge takes input in $\{0,1\}$ (without changing the graph topology or the function g), then there would be a low-memory, low-round message-passing protocol, since the root node simply needs to compute the histogram of the leaves' inputs, which takes space $O(\log n)$ . Each leaf node can use this information together with its own input value to compute its output. Essentially, there is no information bottleneck because there is a concise, sufficient “summary” of the input data.

However, the above construction can in fact be adapted to work with binary alphabet, by modifying the graph topology. At a high level, for each leaf node u in the above construction, we add n descendants and encode the input that was originally on u on the descendants of u, in unary. Of course, this new graph has $n^{2}$ nodes, so we must rescale parameters accordingly.

We now make this idea formal. For notational convenience, define $m = \lfloor \sqrt{n} \rfloor$ . We define a graph $G = (V, E)$ that is a perfect n-ary tree of depth two. Formally, the graph G has vertex set $V = \{0\} \cup [m] \cup ([m] \times [m])$ . Vertex 0 is adjacent to each $i \in [m]$ , and each $i \in [m]$ is additionally adjacent to $(i, j)$ for all $j \in [m]$ . We define a function $g : \{0, 1\}^{E} \to \{0, 1\}^{V}$ as follows. On input $I \in \{0, 1\}^{E}$ , for each edge $e \in E$ , define the input summation at e to be

$$
C (I) _ {e} := \sum_ {e ^ {\prime} \in M _ {G} (e)} I (e ^ {\prime}).
$$

Intuitively, one may think of $C(I)_{e}$ as simulating the input on e in the “large alphabet” construction described in Section 6. Next, define

$$
g (I) _ {(u, j)} := 0.
$$

$$
g (I) _ {u} := \mathbb {1} [ \# | e \in M _ {G} (\{0, u \}): C (I) _ {e} = C (I) _ {\{0, u \}} | > m + 1 ].
$$

$$
g (I) _ {0} := \mathbb {1} [ \exists u \in [ m ]: g (I) _ {u} = 1 ].
$$

In words, $g(I)_{u}$ is the indicator for the event that, among the $2m + 1$ edges adjacent to $\{0, u\}$ (which include $\{0, u\}$ itself), more than $m + 1$ edges have the same input summation as $\{0, u\}$ . At a high level, this definition of g was designed to satisfy three criteria. First, $g(I)_{u}$ depends on the input values on other branches of the tree: in particular, if $I_{\{0,v\}} = 0$ for all $v \in [n]$ , then $C(I)_{e} = C(I)_{\{0,u\}}$ for all edges e in the subtree of u, so $g(I)_{u}$ exactly measures the event that there is at least one edge e outside the subtree of u for which $C(I)_{e} = C(I)_{\{0,u\}}$ . Second, there is no concise “summary” of I such that $g(I)_{u}$ can be determined from this summary in conjunction with the inputs on the subtree of u. Third, $g(I)$ is equivariant to re-labelings of the tree.

The first two criteria, together with the fact that the root vertex 0 is an “information bottleneck” for G, can be used to show that any node message-passing algorithm that computes g on G requires either large memory or many rounds. The third criterion enables construction of a symmetric edge message-passing protocol for g. The arguments are formalized in the claims below.

Claim 5. For graph G and function g as defined above, any node message-passing protocol on G that computes g with T rounds and B bits of memory requires $TB \geq \Omega(m)$ .

Proof. Consider any input $I \in \{0,1\}^E$ with $I(\{0,u\}) = 0$ for all $u \in [m]$ . Then for any $u,j \in [m]$ , we have

$$
C (I) _ {\{u, (u, j) \}} = C (I) _ {\{0, u \}} = \sum_ {i = 1} ^ {m} I (\{u, (u, i) \}).
$$

Thus $g(I)_u = 1$ if and only if there exists some $v \in [m] \setminus \{u\}$ with $C(I)_{\{0,u\}} = C(I)_{\{0,v\}}$ , or equivalently $\sum_{i=1}^{m} I(\{u, (u,i)\}) = \sum_{i=1}^{m} I(\{v, (v,i)\})$ .

Fix T, B and suppose that P is a node message-passing protocol on G that computes g with T rounds and B bits of memory. Define sets of vertices $K := \{0\}$ and $S := \{1, \ldots, m/2\}$ . Let $H := G[\overline{K}]$ and $F := M_G(N_H^{T-1}(S))$ . Then for any T, we have that

$$
F = \{\{0, u \}: 1 \leq u \leq m / 2 \} \cup \{\{u, (u, j) \}: 1 \leq u \leq m / 2, 1 \leq j \leq m \}.
$$

Define a vector $I_F \in \Phi^F$ by

$$
I _ {\{0, u \}} = 0 \text {for} 1 \leq u \leq m / 2
$$

$$
I _ {\{u, (u, j) \}} = \mathbb {1} [ j \leq u ] \text {for} 1 \leq u \leq m / 2, 1 \leq j \leq m.
$$

Now fix any $x \in \{0,1\}^S$ . We claim that there is some $I_{\overline{F}} \in \Phi^{\overline{F}}$ such that $g_S(I_F, I_{\overline{F}}) = x$ . Indeed, let us define $I_{\overline{F}}$ by:

$$
I _ {\{0, v \}} = 0 \mathrm{for} m / 2 <   v \leq m
$$

$$
I _ {\{v, (v, j) \}} = x _ {v - m / 2} \mathbb {1} [ j \leq v - m / 2 ] \text {for} m / 2 <   v \leq m, 1 \leq j \leq m.
$$

Then $C(I)_{\{0,u\}} = u$ for all $1 \leq u \leq m/2$ , and $C(I)_{\{0,v\}} = (v - m/2)x_{v-m/2}$ for all $m/2 < v \leq m$ . It follows that for any $1 \leq u \leq n/2$ , $x_u = 1$ if and only if there exists some $v \in [m] \setminus u$ with $C(I)_{\{0,u\}} = C(I)_{\{0,v\}}$ , and hence $x_u = g(I)_u$ . We conclude that

$$
\left| \left\{g _ {S} (I _ {F}, I _ {\overline {{F}}}): I _ {\overline {{F}}} \in \Phi^ {\overline {{F}}} \right\} \right| \geq 2 ^ {m / 2}.
$$

Applying Lemma 2 we conclude that $TB \geq \Omega(m)$ as claimed.

![](images/b603291ebb7368bbff891008e547cb5f42838598e5afc671d8ef4d525b5a06a5.jpg)

Claim 6. For graph G and function g as defined above, there is a symmetric edge message-passing protocol on G that computes g with $O(1)$ rounds and $O(\log m)$ bits of memory.

Proof. In the first round, each edge processor reads its input value. In the second round, each edge processor sums the values computed by all neighboring edges (including itself). In the third round, each edge processor computes the indicator for the event that strictly more than $m + 1$ neighboring edges (including itself) have the same value as itself. In the final aggregation round, the output of a vertex is the indicator for the event that any neighbor has value 1.

By construction, the value computed by any edge e after the second round is exactly $C(I)_{e}$ . Thus, after the third round, the value computed by any edge $\{0, u\}$ is exactly $g(I)_{u}$ . Moreover, the value computed by any edge $\{u, (u, j)\}$ is 0 after the third round, since such edges only have $m + 1$ neighbors. It follows by construction of the final aggregation step that the protocol computes g. ☐

Proof of Theorem 4. Immediate from Claims 5 and 6.

![](images/120c2bf8fdcf832c0bb89a775921d376595858784a140dd72e55fc7c29f73f9c.jpg)

# 7 Symmetry alone provides no separation

In the previous sections we saw that examining memory constraints yields a separation between different GNN architectures (whether or not we take symmetry into consideration). In this section, we consider what happens if we solely consider symmetry constraints (that is, constraints imposed by requiring that the computation in a round of the protocol is invariant to permutations of the order of the neighbors). This viewpoint was initiated by Xu et al. (2018), who showed that when the initial node features are uninformative (that is, the same for each node), a standard GNN necessarily outputs the same answer for two graphs that are 1-Weisfeiler-Lehman equivalent (that is, graphs that cannot be distinguished by the Weisfeiler-Lehman test, even though they may not be isomorphic).

To be precise, we revisit the representational power of symmetric GNN architectures in the setting where the input features may be distinct and informative. We show that if we remove the memory constraints from Section 5, but impose permutation invariance for the computation in each round, any function that is computable by a T-layer edge message-passing protocol can be computed by a $(T+1)$ -layer node message-passing protocol. Note that this statement is incomparable to Proposition 3 because we impose constraints on symmetry, but remove constraints on memory.

Theorem 7 (No separation under symmetry constraints). Let $T \geq 1$ . Let $P$ be a symmetric edge message-passing protocol (Definition 6) on graph $G = (V, E)$ with $T$ rounds. Then there is a $(T + 1)$ -round symmetric node message-passing protocol (Definition 5) $P'$ on $G$ that computes the same function as $P$ .

Remark 11. Theorem 7 and its proof are closely related to the fact that the 1-Weisfeiler-Lehman test is equivalent to the 2-Weisfeiler-Lehman test, which was reintroduced in the context of higher-order GNNs (Huang & Villar, 2021). However, the k-Weisfeiler-Lehman test only characterizes the representational power of k-GNNs with uninformative input features (i.e. that are identical for all nodes). Theorem 7 shows that even with arbitrary input features on the edges, the computation of a GNN with edge embeddings and symmetric updates can be simulated by a GNN with only node embeddings, without losing symmetry.

To prove Theorem 7, note that it suffices to simulate the protocol P for which the update rules $f^{sym}$ , $\tilde{f}^{sym}$ in Definition 6 are identity functions on the appropriate domains. In order to simulate P, we construct a symmetric node message-passing protocol $P'$ for which the computation at time

$t+1$ and node v on input I is the multiset of features computed by P at time t at edges adjacent to v: $Q_{t}(v;I):=\{P_{t}(e;I):e\in M_{G}(v)\}$ . This is possible since the computation of P at time t and edge $e=(u,v)$ is $P_{t}(e;I)=(I(e),P_{t-1}(e;I),\{Q_{t-1}(u;I),Q_{t-1}(v;I)\})$ . The node message-passing protocol is tracking $Q_{t-1}(\cdot;I)$ ; moreover, it can recursively compute $P_{t-1}(e;I)$ using the same formula. See Appendix B for the formal proof.

# 8 Empirical benefits of edge-based architectures

In this section we demonstrate that the representational advantages the theory suggests are borne out by experimental evaluations, both on real-life benchmarks and two natural synthetic tasks we provide. Note that all the experiments were done on a machine with 8 Nvidia A6000 GPUs.

# 8.1 Performance on common benchmarks

First we compare the performance of the most basic GNN architecture (Graph Convolutional Network, Kipf & Welling (2016)) with node versus edge embeddings. In the notation of (1) and (2), the AGGREGATE and COMBINE operations are integrated as a transformation that looks like Eq. (3) or Eq. (4): $^{2}$

$$
h _ {v} ^ {(l + 1)} = h _ {v} ^ {(l)} + \sigma \left(W ^ {(l)} \text {MEAN} \left(h _ {w} ^ {(l)}: w \in N _ {G} (v) \backslash \{v \}\right)\right) \tag {3}
$$

$$
h _ {e} ^ {(l + 1)} = h _ {e} ^ {(l)} + \sigma \left(W ^ {(l)} \text { MEAN } \left(h _ {f} ^ {(l)}: f \in M _ {G} (e) \setminus \{e \}\right)\right) \tag {4}
$$

for trained matrices $W^{(l)}$ and a choice of non-linearity $\sigma$ . The only difference between these architectures is that in the latter case, the message passing happens over the line graph of the original graph (i.e. the neighborhood of an edge is given by the other edges that share a vertex with it) — thus, this can be viewed as an ablation experiment in which the only salient difference is the type of embeddings being maintained. To also equalize the information in the input embeddings, we only use the node embeddings in the benchmarks we consider: for the edge-based architecture (2), we initialize the edge embeddings by the concatenation of the node embeddings of the endpoints.

In Table 1, we show that this single change (without any other architectural modifications) uniformly results in the edge-based architecture at least matching the performance of the node-based architecture, sometimes improving upon it. Note, the purpose of this table is not to advocate a new GNN architecture $^{3}$ — but to confirm that the increased representational power of the edge-based architecture indicated by the theory also translates to improved performance when the model is trained. For each benchmark, we follow the best performing training configuration as delineated in (Dwivedi et al., 2023).

<table><tr><td rowspan="2">Model</td><td>ZINC</td><td>MNIST</td><td>CIFAR-10</td><td>Peptides-Func</td><td>Peptides-Struct</td></tr><tr><td>MAE (↓)</td><td>ACCURACY (↑)</td><td>ACCURACY (↑)</td><td>AP (↑)</td><td>MAE (↓)</td></tr><tr><td>GCN</td><td> $0.3430 \pm 0.034$ </td><td> $95.29 \pm 0.163$ </td><td> $55.71 \pm 0.381$ </td><td> $0.6816 \pm 0.007$ </td><td> $0.2453 \pm 0.0001$ </td></tr><tr><td>Edge-GCN (Ours)</td><td> $0.3297 \pm 0.011$ </td><td> $94.37 \pm 0.065$ </td><td> $57.44 \pm 0.387$ </td><td> $0.6867 \pm 0.004$ </td><td> $0.2437 \pm 0.0005$ </td></tr></table>

Table 1: Comparison of node-based (3) and edge-based (4) GCN architectures across various graph benchmarks. The performance of the edge-based architecture robustly matches or improves the node-based architecture.

# 8.2 A synthetic task for topologies with node bottlenecks

The topologies of the graphs in Theorem 1 and Theorem 4 both involve a “hub” node, which is connected to all other nodes in the graph. Intuitively, in node-embedding architectures, such nodes have to mediate messages between many pairs of other nodes, which is difficult when the node is constrained by memory. To empirically stress test this intuition, we produce a synthetic dataset and train a GNN to solve a regression task on a graph with a fixed star-graph topology—a simpler topology than the constructions in Theorem 1 and Theorem 4—but capturing the core aspect of both. A star graph is a graph with a center node $v_{0}$ , a set of n leaf nodes $\{v_{i}\}_{i\in[n]}$ , and edge set $\{\{v_{0},v_{i}\}_{i\in[n]}\}$ . A training point in the dataset is a list $(x_{i},y_{i})_{i=1}^{n}$ where $x_{i}$ is the input feature and $y_{i}$ is the label for leaf node $v_{i}$ .

The input features are in $R^{10}$ , and sampled from a standard Gaussian. The labels $y_{i}$ are produced as outputs of a planted edge-based architecture. Namely, for a standard edge-based GCN as in (4), we randomly choose values for the matrices $\{W_{i}\}_{i\in[k]}$ for some number of layers k, and set the labels to be the output of this edge-based GCN, when the initial edge features to the GCN are set as $h_{\{v_{0},v_{i}\}}^{(0)}:=x_{i}$ , i.e. the input feature $x_{i}$ at the corresponding leaf i. In Table 2, we show the performance of edge-based and node-based architectures on this dataset, varying the number of leaves n in the star graph and the depth k of the planted edge-based model. In each case, the numbers indicate RMSE of the best-performing edge-based and node-based architecture, sweeping over depths up to 10 (2× the planted model), widths $\in\{16,32,64\}$ , and a range of learning rates.

Since the planted edge-based model satisfies both invariance constraints (by design of the GCN architecture) and memory constraints (since the planted model maintains 10-dimensional embeddings), we view these results as empirical corroboration of Theorem 4—and even for simpler topologies than the proof construction.

<table><tr><td rowspan="3">Number of Leaves</td><td colspan="6">Depth of Planted Model (RMSE)</td></tr><tr><td colspan="2">5</td><td colspan="2">3</td><td colspan="2">1</td></tr><tr><td>Edge</td><td>Node</td><td>Edge</td><td>Node</td><td>Edge</td><td>Node</td></tr><tr><td>64</td><td>0.004</td><td>0.3790</td><td>0.011</td><td>0.3596</td><td>0.008</td><td>0.3752</td></tr><tr><td>32</td><td>0.003</td><td>0.3664</td><td>0.005</td><td>0.3626</td><td>0.003</td><td>0.3614</td></tr><tr><td>16</td><td>0.007</td><td>0.3336</td><td>0.002</td><td>0.2100</td><td>0.002</td><td>0.2847</td></tr></table>

Table 2: Performance (in RMSE ↓) of edge-based and node-based architectures on a star-graph topology. The first number is the performance of the best edge-based model, and the second is the best node-based model, across a range of depths up to 10 (2× the planted model), widths ∈ {16, 32, 64}, and a range of learning rates.

# 8.3 A synthetic task for inference in Ising models

Finally, motivated by the probabilistic inference setting in Theorem 1, we consider a synthetic sandbox of using GNNs to predict the values of marginals in an Ising model (Ising, 1924; Onsager, 1944) - a natural type of pairwise graphical model where each node takes a value in $\{\pm 1\}$ , and each edge potential is a weighted product of the edge endpoint values. Concretely, the probability distribution of an Ising model over graph $G = (V,E)$ has the form:

$$
\forall x \in \{\pm 1 \} ^ {n}: p _ {J, h} (x) \propto \exp \Bigl (\sum_ {\{i, j \} \in E} J _ {\{i, j \}} x _ {i} x _ {j} + \sum_ {i \in V} h _ {i} x _ {i} \Bigr).
$$

Similar to in Section 8.2, we construct a training set where the graph G and and edge potentials stay fixed (precisely, $J_{i,j} = 1$ for all $\{i, j\} \in E$ ). A training data-point consists of a vector of node potentials $\{h_i\}_{i \in [n]}$ , and labels $\{E[x_i]\}_{i \in [n]}$ consisting of the marginals from the resulting Ising model $p_{J,h}$ . The node potentials are sampled from a standard Gaussian distribution.

There is a natural connection between GNNs and calculating marginals: a classical way to calculate $\{E[x_{i}]\}$ when G is a tree is to iterate a message passing algorithm called belief propagation (5), in which for each edge $\{i,j\}$ and direction $i\to j$ , a message $\nu_{i\to j}^{(t+1)}$ is calculated that depends on messages $\{\nu_{k\to i}^{(t)}\}_{\{k,i\}\in E}$ . The belief-propagation updates (5) naturally fit the general edge-message passing paradigm from (2). In fact, they fit even more closely a “directed” version of the paradigm, in which each edge $\{i,j\}$ maintains two embeddings $h_{i\to j}, h_{j\to i}$ , such that the embedding for direction $h_{i\to j}$ depends on the embeddings $\{h_{k\to i}\}_{\{k,i\}\in E}$ — and it is possible to derive a similar “directed” node-based architecture (See Appendix D.2). For both the undirected and directed version of the architecture, we see that maintaining edge embeddings gives robust benefits over maintaining node embeddings—for a variety of tree topologies including complete binary trees, path graphs, and uniformly randomly sampled trees of a fixed size. More details are included in Appendix D.

# 9 Conclusions and future work

Graph neural networks are the best-performing machine learning method for many tasks over graphs. There is a wide variety of GNN architectures, which frequently make opaque design choices and whose causal influence on the final performance is difficult to understand and estimate. In this paper, we focused on understanding the impact of maintaining edge embeddings on the representational power, as well as the subtleties of considering constraints like memory and invariance. One significant downside of maintaining edge embeddings is the computational overhead on dense graphs. Hence, a fruitful direction for future research would be to explore more computationally efficient variants of edge-based architectures that preserve their representational power and performance.

# Acknowledgements

DR is supported by a U.S. DoD NDSEG Fellowship. TM is supported in part by CMU Software Engineering Institute via Department of Defense under contract FA8702-15-D-0002. ZK gratefully acknowledges the NSF (FAI 2040929 and IIS2211955), UPMC, Highmark Health, Abridge, Ford Research, Mozilla, the PwC Center, Amazon AI, JP Morgan Chase, the Block Center, the Center for Machine Learning and Health, and the CMU Software Engineering Institute (SEI) via Department of Defense contract FA8702-15-D-0002, for their generous support of ACMI Lab's research. JL is supported in part by NSF awards DMS-2309378 and IIS-2403275. AM is supported in part by a Microsoft Trustworthy AI Grant, an ONR grant and a David and Lucile Packard Fellowship. AR is supported in part by NSF awards IIS-2211907, CCF-2238523, IIS-2403275, an Amazon Research Award, a Google Research Scholar Award, and an OpenAI Superalignment Fast Grant.

# References

Karl Abrahamson. Time-space tradeoffs for algebraic problems on general sequential machines. Journal of Computer and System Sciences, 43(2):269–289, 1991.   
Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, et al. Interaction networks for learning about objects, relations and physics. Advances in neural information processing systems, 29, 2016.   
Mitchell Black, Zhengchao Wan, Amir Nayyeri, and Yusu Wang. Understanding oversquashing in gnns through the lens of effective resistance. In International Conference on Machine Learning, pp. 2528–2547. PMLR, 2023.   
Lei Cai, Jundong Li, Jie Wang, and Shuiwang Ji. Line graph neural networks for link prediction. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5103–5113, 2021.   
Zhengdao Chen, Xiang Li, and Joan Bruna. Supervised community detection with line graph neural networks. arXiv preprint arXiv:1705.08415, 2017.   
Kamal Choudhary and Brian DeCost. Atomistic line graph neural network for improved materials property predictions. npj Computational Materials, 7(1):185, 2021.   
Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. Journal of Machine Learning Research, 24(43):1–48, 2023.   
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pp. 1263–1272. PMLR, 2017.   
Dmitrii Yur'evich Grigor'ev. Application of separability and independence notions for proving lower bounds of circuit complexity. Zapiski Nauchnykh Seminarov POMI, 60:38–48, 1976.   
Johan Håstad and Avi Wigderson. The randomized communication complexity of set disjointness. Theory of Computing, 3(1):211-219, 2007.   
Ningyuan Teresa Huang and Soledad Villar. A short tutorial on the weisfeiler-lehman test and its variants. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 8533–8537. IEEE, 2021.   
Ernst Ising. Beitrag zur theorie des ferro-und paramagnetismus. PhD thesis, Grefe & Tiedemann Hamburg, Germany, 1924.   
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.   
AA Leman and Boris Weisfeiler. A reduction of a graph to a canonical form and an algebra arising during this reduction. Nauchno-Technicheskaya Informatsiya, 2(9):12–16, 1968.   
Jinbi Liang and Cunlai Pu. Line graph neural networks for link weight prediction. arXiv preprint arXiv:2309.15728, 2023.   
Nathan Linial. Locality in distributed graph algorithms. SIAM Journal on computing, 21(1):193–201, 1992.

Andreas Loukas. What graph neural networks cannot learn: depth vs width. arXiv preprint arXiv:1907.03199, 2019.   
Andreas Loukas. How hard is to distinguish graphs with graph neural networks? Advances in neural information processing systems, 33:3465–3476, 2020.   
Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. Advances in neural information processing systems, 32, 2019.   
Marc Mezard and Andrea Montanari. Information, physics, and computation. Oxford University Press, 2009.   
Lars Onsager. Crystal statistics. i. a two-dimensional model with an order-disorder transition. Physical Review, 65(3-4):117, 1944.   
Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. arXiv preprint arXiv:1905.10947, 2019.   
David Peleg. Distributed computing: a locality-sensitive approach. SIAM, 2000.   
Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and logarithmic depth. arXiv preprint arXiv:2402.09268, 2024a.   
Clayton Sanford, Daniel J Hsu, and Matus Telgarsky. Representational strengths and limitations of transformers. Advances in Neural Information Processing Systems, 36, 2024b.   
John E Savage. Models of computation, volume 136. Addison-Wesley Reading, 1998.   
Matus Telgarsky. Benefits of depth in neural networks. In Conference on learning theory, pp. 1517-1539. PMLR, 2016.   
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018.   
Tuo Xu and Lei Zou. Rethinking and extending the probabilistic inference capacity of gnns. In The Twelfth International Conference on Learning Representations, 2023.   
Lingxiao Zhao, Wei Jin, Leman Akoglu, and Neil Shah. From stars to subgraphs: Uplifting any gnn with local structure awareness. arXiv preprint arXiv:2110.03753, 2021.

![](images/4af20870290de1bc676ca282c3f7bd85483c6e4c3d87f280b9b96e776ac32929.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["•"] --> B["•"]
    A --> C["•"]
    A --> D["•"]
    A --> E["•"]
    A --> F["•"]
    A --> G["•"]
    A --> H["•"]
    A --> I["•"]
    A --> J["•"]
    A --> K["•"]
    A --> L["•"]
    A --> M["•"]
    A --> N["•"]
    A --> O["•"]
    A --> P["•"]
    A --> Q["•"]
    A --> R["•"]
    A --> S["•"]
    A --> T["•"]
    A --> U["•"]
    A --> V["•"]
    A --> W["•"]
    A --> X["•"]
    A --> Y["•"]
    A --> Z["•"]
    A --> AA["•"]
    A --> AB["•"]
    A --> AC["•"]
    A --> AD["•"]
    A --> AE["•"]
    A --> AF["•"]
    A --> AG["•"]
    A --> AH["•"]
    A --> AI["•"]
    A --> AJ["•"]
    A --> AK["•"]
    A --> AL["•"]
    A --> AM["•"]
    A --> AN["•"]
    A --> AO["•"]
    A --> AP["•"]
    A --> AQ["•"]
    A --> AR["•"]
    A --> AS["•"]
    A --> AT["•"]
    A --> AU["•"]
    A --> AV["•"]
    A --> AW["•"]
```
</details>

Figure 1: The graph G for which Theorem 1 exhibits a separation between edge message-passing and node message-passing. The graph consists of $\sqrt{n}$ paths of length $\sqrt{n}$ , as well as a single “hub vertex” connected to all other vertices.

# Appendix

# A Omitted Proofs from Section 5

In this section we give omitted proofs and lemmas from Section 5.

Lemma 8. Fix $n \in N$ . Let G, $\Phi$ be as defined in Theorem 1. Then there is an $O(n)$ -time algorithm that computes a MAP evaluator for G with potential function class $\Phi$ .

Proof. Fix any $J \in \Phi^{E}$ . As preliminary notation, for each $c, c_{0} \in \{0,1\}$ and $i, j \in \sqrt{n}$ , let $V(i,j) := \{0\} \cup \{(k,j) : 1 \leq k \leq i\}$ , and let $E(i,j)$ be the edge set of the induced subgraph $G[V(i,j)]$ . Let

$$
\hat{x}_{i,j}(c,c_{0};J):= \operatorname *{arg  min}_{\substack{x\in \{0,1\}^{V(i,j)}:\\ x_{0} = c_{0}\land x_{(i,j)} = c}}\sum_{(a,b)\in E(i,j)}J_{\{a,b\}}(x_{a},x_{b}),
$$

$$
\hat{C}_{i,j}(c,c_{0};J):= \min_{\substack{x\in \{0,1\}^{V(i,j)}:\\ x_{0} = c_{0}\land x_{(i,j)} = c}}\sum_{(a,b)\in E(i,j)}J_{\{a,b\}}(x_{a},x_{b}).
$$

For each $j\in [\sqrt{n} ]$ , let

$$
\hat {x} _ {j} (c _ {0}; J) := \hat {x} _ {\sqrt {n}, j} \left(\left(\underset {c \in \{0, 1 \}} {\arg \min} \hat {C} _ {\sqrt {n}, j} (c, c _ {0}; J)\right), c _ {0}; J\right).
$$

Finally, let $\hat{x}(c_0;J)\in \{0,1\}^V$ be the vector which takes value $c_{0}$ on vertex 0, and value $\hat{x}_j(c_0;J)_i$ on vertex $(i,j)$ for all $i,j\in \sqrt{n}$ . Let

$$
\hat {x} (J) := \operatorname * {a r g   m a x} _ {c _ {0} \in \{0, 1 \}} p _ {J} (\hat {x} (c _ {0}; J)).
$$

We claim that $\hat{x}(J)$ is a maximizer of $p_{J}(x)$ . Indeed, for any fixed $c_{0} \in \{0,1\}$ , $\hat{x}(c_{0};J)$ is a maximizer of $p_{J}(x)$ subject to $x_{0} = c_{0}$ , because under this constraint the maximization problem decomposes into $\sqrt{n}$ independent maximization problems, one for each path in G, which by definition are solved by $\hat{x}_{1}(c_{0};J), \ldots, \hat{x}_{\sqrt{n}}(c_{0};J)$ .

Moreover, it's straightforward to see that for any fixed $j$ , $\hat{C}_j(c_0;J)$ can be computed in $O(\sqrt{n})$ time by dynamic programming. Indeed for any $i,j$ , $\hat{C}_{i,j}(c,c_0;J)$ can be computed in $O(1)$ time from $\hat{C}_{i - 1,j}(0,c_0;J)$ and $\hat{C}_{i - 1,j}(1,c_0;J)$ as well as $J_{\{0,(i,j)\}}$ and $J_{\{(i - 1,j),(i,j)\}}$ . Once the values $\hat{C}_{i,j}(c,c_0;J)$ have been computed for all $i\in [\sqrt{n}]$ and $c\in \{0,1\}$ , the vector $\hat{x}_j(c_0;J)$ can be computed in $O(\sqrt{n})$ time via a reverse scan over $i = \sqrt{n},\ldots ,1$ . It follows that $\hat{x} (J)$ can be computed in $O(n)$ time.

Proof of Proposition 3. We claim that there is a node message-passing protocol $P'$ on $G$ with $T + 1$ rounds that at each time $t \in [T + 1]$ has computed

$$
P _ {t} ^ {\prime} (v; I) = (P _ {t - 1} (e; I)) _ {e \in M _ {G} (v)}.
$$

We argue inductively. Since $P_0 \equiv 0$ , it's clear that this can be achieved for $t = 1$ . Fix any $t > 1$ and suppose that $P_{t-1}'(u; I) = (P_{t-2}(e; I))_{e \in M_G(u)}$ for all $u \in V$ and inputs $I$ . For each $v \in V$ , we define a function $f_{t,v}'$ by

$$
f _ {t, v} ^ {\prime} ((c (v ^ {\prime})) _ {v ^ {\prime} \in N _ {G} (v)}, (I (e)) _ {e \in M _ {G} (v)}) _ {e ^ {\star}} := f _ {t - 1, e ^ {\star}} ((c (v) _ {e}) _ {e \in M _ {G} (v)}, (c (v ^ {\star}) _ {e}) _ {e \in M _ {G} (v ^ {\star})}, I (e ^ {\star}))
$$

for each $e^{\star} = (v, v^{\star}) \in M_{G}(v)$ . Then by definition and the inductive hypothesis, we have

$$
\begin{array}{l} P _ {t} ^ {\prime} (v; I) _ {e ^ {\star}} = f _ {t, v} ^ {\prime} ((P _ {t - 1} ^ {\prime} (v ^ {\prime}; I)) _ {v ^ {\prime} \in N _ {G} (v)}, (I (e)) _ {e \in M _ {G} (v)}) _ {e ^ {\star}} \\ = f _ {t - 1, e ^ {\star}} ((P _ {t - 1} ^ {\prime} (v; I) _ {e}) _ {e \in M _ {G} (v)}, (P _ {t - 1} ^ {\prime} (v ^ {\star}; I) _ {e}) _ {e \in M _ {G} (v ^ {\star})}, I (e ^ {\star})) \\ = f _ {t - 1, e ^ {\star}} ((P _ {t - 2} (e; I)) _ {e \in M _ {G} (v)}, (P _ {t - 2} (e; I) _ {e}) _ {e \in M _ {G} (v ^ {\star})}, I (e ^ {\star})) \\ = P _ {t - 1} (e ^ {\star}; I) \\ \end{array}
$$

for any edge $e^{\star} = (v, v^{\star}) \in E$ , since $M_G(e) = M_G(v) \cup M_G(v^{\star})$ . This completes the induction and shows that $P_{T+1}'(v; I) = (P_T(e; I))_{e \in M_G(v)}$ for all $v, I$ . Replacing $f_{T+1,v}'$ by $\tilde{f}_{T,v} \circ f_{T+1,v}'$ completes the proof.

# B Omitted Proofs from Section 7

Proof of Theorem 7. Without loss of generality, we may assume that the functions $(f_t^{\mathrm{sym}})_{t\in [T]}$ and $\tilde{f}^{\mathrm{sym}}$ are all the identity function (on the appropriate domains). The reason is that any symmetric edge message-passing protocol $\tilde{P}$ on $T$ rounds may be simulated by running $P$ and then applying a universal function (depending only on $\tilde{P}$ ) to each node's output value - see Lemma 9.

We argue by induction that for each $t \in [T]$ , there is a $(t + 1)$ -round symmetric node message-passing protocol that, on any input I, computes the function $Q_{t}(u;I) := \{P_{t}(e;I) : e \in M_{G}(u)\}$ for every node $u \in V$ . Consider t = 1. For any $e = (u,v) \in E$ , we have by symmetry and the initial assumption that

$$
P _ {1} (e; I) = (I (e), 0, \{\{0: v ^ {\prime} \in N _ {G} (u) \}, \{0: u ^ {\prime} \in N _ {G} (v) \} \}).
$$

We define a two-round node message-passing protocol on G where the first update at node u computes

$$
P _ {1} ^ {\prime} (u; I) = \left\{\left\{I (\{u, v \}): v \in N _ {G} (u) \right\} \right.
$$

and the second update at node u computes

$$
\begin{array}{l} (P _ {1} ^ {\prime} (u; I), \{(P _ {1} ^ {\prime} (v; I), I (\{u, v \})): v \in N _ {G} (u) \}) \mapsto \{(I (\{u, v \}), 0, | N _ {G} (u) |, | P _ {1} ^ {\prime} (v; I) |): v \in N _ {G} (u) \} \\ \mapsto \left\{\left(I (\{u, v \}), 0, \left\{\left| N _ {G} (u) \right|, \left| P _ {1} ^ {\prime} (v; I) \right| \right\}\right): v \in N _ {G} (u) \right\} \\ = \left\{\left\{P _ {1} (\{u, v \}; I): v \in N _ {G} (u) \right\} \right\} =: P _ {2} ^ {\prime} (u; I) \\ \end{array}
$$

since $|P_1'(v;I)| = |N_G(v)|$ . By construction, this protocol is symmetric, which proves the induction for step $t = 1$ .

Now pick any $t > 1$ . For any $e = \{u, v\} \in E$ , we have

$$
P _ {t} (e; I) = (I (e), P _ {t - 1} (e; I), \{\{Q _ {t - 1} (u; I), Q _ {t - 1} (v; I) \} \})
$$

By the induction hypothesis, there is a t-round symmetric node message-passing protocol $P'$ that, at node v on input I, computes

$$
P _ {t} ^ {\prime} (v; I) = \{\{P _ {t - 1} (\{v, v ^ {\prime} \}; I): v ^ {\prime} \in N _ {G} (v) \} \} = Q _ {t - 1} (v; I).
$$

Note that since $P_{t-1}(e;I)$ is an element of the tuple $P_{t}(e;I)$ , for each $1 \leq s \leq t-1$ there is a fixed function $\gamma_{s}$ such that $\gamma_{s}(Q_{t-1}(v;I)) = Q_{s}(v;I)$ for all v, I. Using this fact, we extend $P'$ to $t+1$ rounds, defining the update at round $t+1$ and node u as follows:

$$
\begin{array}{l} (P _ {t} ^ {\prime} (u; I), \left\{\left(P _ {t} ^ {\prime} (v; I), I (\{u, v \})\right): v \in N _ {G} (u) \right\}\left. \right) \\ = (Q _ {t - 1} (u; I), \{(Q _ {t - 1} (v; I), I (\{u, v \})): v \in N _ {G} (u) \}) \\ \mapsto (Q _ {1: t - 1} (u; I), \{(Q _ {1: t - 1} (v; I), I (\{u, v \})): v \in N _ {G} (u) \}) \\ \mapsto (Q _ {1: t - 1} (u; I), \{(Q _ {1: t - 1} (v; I), I (\{u, v \})): v \in N _ {G} (u) \}) \\ = \left\{\left(I (\{u, v \}), \left\{Q _ {1: t - 1} (u; I), Q _ {1: t - 1} (v; I) \right\}\right): v \in N _ {G} (u) \right\} \\ \mapsto \left\{ \right.\left( \right.I (\{u, v \}), P _ {t - 1} (\{u, v \}; I), \left\{ \right.\left. Q _ {t - 1} (u; I), Q _ {t - 1} (v; I) \right\}\left. \right): v \in N _ {G} (u) \left. \right\} =: P _ {t + 1} ^ {\prime} (u; I) \\ \end{array}
$$

where $Q_{1:t-1}(u;I)$ refers to the tuple $(Q_{1}(u;I),\ldots,Q_{t-1}(u;I))$ . The first map is well-defined due to the existence of the functions $\gamma_{1},\ldots,\gamma_{t-1}$ , and the final map is well-defined because the definition of $P_{t-1}(\{u,v\};I)$ can be iteratively unpacked, and it is ultimately a function of

$$
(I (\{u, v \}), \{\{Q _ {1: t - 1} (u; I), Q _ {1: t - 1} (v; I) \} \}).
$$

This shows that $P'$ computes $Q_{t}(v;I)$ at node u on input I. By construction, $P'$ is symmetric. This completes the induction. Since $Q_{T}(u;I)$ is precisely the output of P at node u on input I (after the node aggregation step), this shows that P can be simulated by a $(T+1)$ -round symmetric node message-passing protocol on G. ☐

Lemma 9. Let $T \geq 1$ , and let $P = ((f_{t,e})_{t \in [T],e \in E}, (\tilde{f}_v)_{v \in V})$ be a symmetric edge message-passing protocol on $G = (V,E)$ with $T$ rounds. Consider the $T$ -round edge message-passing protocol $P^\circ = ((f_{t,e}^\circ)_{t \in [T],e \in E}, (\tilde{f}_v^\circ)_{v \in V})$ where for all $t, e$ ,

$$
f _ {t, e} ^ {\circ} ((c (e ^ {\prime})) _ {e ^ {\prime} \in M _ {G} (e)}, I (e)) := (I (e), c (e), \{\{c (\{u, v ^ {\prime} \}): v ^ {\prime} \in N _ {G} (u) \}, \{\{c (\{u ^ {\prime}, v \}): u ^ {\prime} \in N _ {G} (v) \} \}),
$$

and for every $v \in V$ ,

$$
\tilde {f} _ {v} ^ {\circ} ((c (e)) _ {e \in M _ {G} (v)}) := \{\{c (e): e \in M _ {G} (v) \} \}.
$$

Then there is a function $h$ such that $\tilde{f}_v((P_T(e;I))_{e\in M_G(v)}) = h(\tilde{f}_v^\circ ((P_T^\circ (e;I))_{e\in M_G(v)}))$ for all $v,I$ .

Proof. We prove by induction that for each $t \in \{0, \ldots, T\}$ there is a function $h_t$ such that $P_t(e; I) = h_t(P_t^\circ(e; I))$ for all e, I. For t = 0 this is immediate from the convention that $P_0 \equiv P_0^\circ \equiv 0$ . Fix any $t \in \{1, \ldots, T\}$ . Since P is symmetric, there is a function $f_t^{sym}$ so that for all $e = (u, v) \in E$ and inputs I,

$$
\begin{array}{l} P _ {t} (e; I) = f _ {t} ^ {\mathsf {s y m}} (I (e), P _ {t - 1} (e; I), \{\{P _ {t - 1} (\{u, v ^ {\prime} \}; I): v ^ {\prime} \sim u \} \}, \{\{P _ {t - 1} (\{u ^ {\prime}, v \}; I): u ^ {\prime} \sim v \} \}) \\ = f _ {t} ^ {\mathrm{sym}} (I (e), h _ {t - 1} (P _ {t - 1} ^ {\circ} (e; I)), \{\{h _ {t - 1} (P _ {t - 1} ^ {\circ} (\{u, v ^ {\prime} \}; I)): v ^ {\prime} \sim u \} \}, \{\{h _ {t - 1} (P _ {t - 1} ^ {\circ} (\{u ^ {\prime}, v \}; I)): u ^ {\prime} \sim v \} \}) \\ \end{array}
$$

which is indeed a well-defined function (independent of e, I) of

$$
P _ {t} ^ {\circ} (e; I) = (I (e), P _ {t - 1} ^ {\circ} (e; I), \{\{P _ {t - 1} ^ {\circ} (\{u, v ^ {\prime} \}; I): v ^ {\prime} \sim u \} \}, \{\{P _ {t - 1} ^ {\circ} (\{u ^ {\prime}, v \}; I): u ^ {\prime} \sim v \} \}).
$$

This completes the induction. Finally, since $P$ is symmetric, there is a function $\tilde{f}^{\mathrm{sym}}$ such that $\tilde{f}_v((P_T(e;I))_{e\in M_G(v)}) = \tilde{f}^{\mathrm{sym}}(\{P_T(e;I):e\in M_G(v)\})$ for all $v,I$ . Hence we can write

$$
\begin{array}{l} \tilde {f} _ {v} ((P _ {T} (e; I)) _ {e \in M _ {G} (v)}) = \tilde {f} ^ {\mathsf {s y m}} (\{\{P _ {T} (e; I): e \in M _ {G} (v) \}) \\ = \tilde {f} ^ {\mathsf {s y m}} (\{\{h _ {T} (P _ {T} ^ {\circ} (e; I)): e \in M _ {G} (v) \} \}) \\ \end{array}
$$

which is a well-defined function (independent of $v, I$ ) of $\{P_T^\circ(e; I) : e \in M_G(v)\}$ as needed.

![](images/d91ae2bcc500db63ee9437fed876607e41ea63acbd9f3a7ab6f8c5fecf0f637f.jpg)

# C A quantitatively tight depth/memory separation

For each $n \in N$ , let $K_{n} := ([n], E_{n})$ be the complete graph on [n]. In this section we show that there is a function that can be computed by an edge message-passing protocol on $K_{n}$ with constant rounds and constant memory per processor, but for which any node message-passing protocol with T rounds and B bits of memory requires $TB \geq \Omega(n)$ . We remark that this separation is quantitatively tight due to Proposition 3, although it is possible that a larger (e.g. even super-polynomial in n) depth separation may be possible if the node message-passing protocol is restricted to constant memory per processor.

At a technical level, the lower bound proceeds via a reduction from the set disjointness problem in communication complexity, similar to the lower bounds in Loukas (2019).

Definition 12. Fix $m \in \mathbb{N}$ . The set disjointness function $\mathsf{DISJ}_m : \{0,1\}^m \times \{0,1\}^m \to \{0,1\}$ is defined as

$$
\mathsf {D I S J} _ {m} (A, B) := \mathbb {1} [ \forall i \in [ m ]: A _ {i} B _ {i} = 0 ].
$$

The following fact is well-known; see e.g. discussion in Håstad & Wigderson (2007).

Lemma 10. In the two-party deterministic communication model, the deterministic communication complexity of $DISJ_{m}$ is at least m.

The main result of this section is the following:

Theorem 11. Fix any even $n \in N$ . Define $g : \{0,1\}^{E_n} \to \{0,1\}^n$ by

$$
g (I) _ {v} := \mathbb {1} [ \exists \{i, j \} \in E _ {n}: i, j \leq n / 2 \wedge I (\{i, j \}) = I (\{n + 1 - i, n + 1 - j \}) = 1 ]
$$

for all $I \in \{0,1\}^{E_n}$ and $v \in [n]$ . Then the following properties hold:

- Any node message-passing protocol on $K_{n}$ with $T$ rounds and $B$ bits of memory that computes $g$ requires $TB \geq \Omega(n)$   
- There is an edge message-passing protocol on $K_{n}$ with $O(1)$ rounds and $O(1)$ bits of memory that computes $g$ .

Proof. Let $m := \binom{n/2}{2}$ . Let $P = (f_{t,v})_{t,v}$ be a node message-passing protocol on $K_n$ that computes g with T rounds and B bits of memory. We design a two-party communication protocol for $DISJ_m$ as follows. Suppose that Alice holds input $X \in \{0,1\}^m$ and Bob holds input $Y \in \{0,1\}^m$ . Let us index the edges $\{i,j\} \in E_n$ with $i,j \leq n/2$ by [m], and similarly index the edges $\{i,j\} \in E_n$ with i,j > n/2 by [m], in such a way that edge $\{i,j\}$ has the same index as edge $\{n+1-i,n+1-j\}$ . Let $I \in \{0,1\}^{E_n}$ be defined by

$$
I (\{i, j \}) := \left\{ \begin{array}{l l} X _ {\{i, j \}} & \text { if } i, j \leq n / 2 \\ Y _ {\{i, j \}} & \text { if } i, j > n / 2. \\ 0 & \text { otherwise } \end{array} \right.
$$

Initially, Alice computes $\hat{P}_{0}(v):=0$ for all $v\in\{1,\ldots,n/2\}$ , and Bob computes $\hat{P}_{0}(v):=0$ for all $v\in\{n/2+1,\ldots,n\}$ . The communication protocol then proceeds in T rounds. At round $t\in[T]$ , Alice sends $(\hat{P}_{t-1}(v))_{1\leq v\leq n/2}$ to Bob, and Bob sends $(\hat{P}_{t-1}(v))_{n/2+1\leq v\leq n}$ to Alice. Alice then computes

$$
\hat {P} _ {t} (v) := f _ {t, v} ((\hat {P} _ {t - 1} (v ^ {\prime})) _ {v ^ {\prime} \in [ n ]}, (I (e)) _ {e \in M _ {K _ {n}} (v)})
$$

for each $1 \leq v \leq n/2$ , and Bob computes the same for each $n/2 < v \leq n$ . Note that for any $i \leq n/2$ and edge $e \in M_{K_n}(i)$ , Alice can compute $I(e)$ . Similarly, for any $i > n/2$ and edge $e \in M_{K_n}(i)$ , Bob can compute $I(e)$ . Thus, this computation is well-defined. After round $T$ , Alice and Bob output $1 - \hat{P}_T(1)$ and $1 - \hat{P}_T(n)$ respectively.

This defines a communication protocol. Since $\hat{P}_t(v) \in \{0,1\}^B$ for each $v \in [n]$ and $t \in [T]$ , the total number of bits communicated is at most $nBT$ . Moreover, by induction it's clear that Alice and Bob output $1 - P_T(1;I)$ and $1 - P_T(n;I)$ respectively. By assumption that $P$ computes $g$ and the fact that $g(I)_v = 1 - \mathsf{DISJ}_m(X,Y)$ for all $v \in [n]$ , we have that $1 - P_T(1;I) = 1 - P_T(n;I) = 0$ if $\mathsf{DISJ}_m(I) = 0$ , and $1 - P_T(1;I) = 1 - P_T(n;I) = 1$ if $\mathsf{DISJ}_m(I) = 1$ . Thus, this communication protocol computes $\mathsf{DISJ}_m$ . By Lemma 10, it follows that $nBT \geq m = \Omega(n^2)$ , so $BT = \Omega(n)$ as claimed.

Next, we exhibit an edge message-passing protocol on $K_{n}$ that computes g with six rounds and one bit of memory. For $1 \leq t \leq 6$ and $e \in E_{n}$ , define $f_{t,e} : \{0,1\}^{M_{G}(e)} \times \{0,1\} \to \{0,1\}$ as follows:

$$
\begin{array}{l} f _ {1, \{i, j \}} (x, y) := y \\ f _ {2, \{i, j \}} (x, y) := x _ {\{n + 1 - i, j \}} \\ f _ {3, \{i, j \}} (x, y) := x _ {\{i, n + 1 - j \}} \\ f _ {4, \{i, j \}} (x, y) := \mathbb {1} [ y = x _ {\{i, j \}} \wedge i, j \leq n / 2 ] \\ f _ {5, \{i, j \}} (x, y) := \mathbb {1} [ \exists k \in [ n ]: x _ {\{i, k \}} = 1 ] \\ f _ {6, \{i, j \}} (x, y) := \mathbb {1} [ \exists k \in [ n ]: x _ {\{i, k \}} = 1 ]. \\ \end{array}
$$

Also define $\tilde{f}_v:\{0,1\}^{M_G(v)}\to \{0,1\}$ for each $v\in [n]$ by $\tilde{f}_v(x):= x_{\{x,1\}}$ . It can be checked that the computation of $P$ at timestep $t = 6$ is

$$
P _ {6} (\{i, j \}; I) := \mathbb {1} [ \exists k, \ell \in [ n / 2 ]: I (\{k, \ell \}) = I (\{n + 1 - k, n + 1 - \ell \}) ] = g (I).
$$

From the definition of $\tilde{f}$ , it follows that P computes g.

# D Further details on synthetic task over Ising models

# D.1 Background on belief propagation

A classical way to calculate the marginals $\{E[x_{i}]\}$ of an Ising model, when the associated graph is a tree, is to iterate the message passing algorithm:

$$
\nu_ {i \rightarrow j} ^ {(t + 1)} = \tanh \left(h _ {i} + \sum_ {k \in \partial_ {i} \backslash j} \tanh ^ {- 1} \left(\tanh (J _ {i k}) \nu_ {k \rightarrow i} ^ {(t)}\right)\right) \tag {5}
$$

When the graph is a tree, it is a classical result ((Mezard & Montanari, 2009), Theorem 14.1) that the above message-passing algorithm converge to values $\nu^{*}$ that yield the correct marginals, namely:

$$
\mathbb {E} [ x _ {i} ] = \tanh \left(h _ {i} + \sum_ {k \in \partial_ {i}} \tanh ^ {- 1} (\tanh (J _ {i k}) \nu_ {k \to i} ^ {*})\right).
$$

The reason the updates converge to the correct values on a tree topology is that they implicitly simulate a dynamic program. Namely, we can write down a recursive formula for the marginal of node i which depends on sums spanning each of the subtrees of the neighbors of i (i.e., for each neighbor j, the subgraph containing j that we would get if we removed edge $\{i,j\}$ ).

If we root the tree at an arbitrary node r, we can see that after completing a round of message passing from the leaves to the root, and another from the root to the leaves, each subtree of i will be (inductively) calculated correctly.

Moreover, even though the updates (5) are written over edges, the dynamic programming view makes it clear an equivalent message-passing scheme can be written down where states are maintained over the nodes in the graph. Namely, for each node v, we can maintain two values $h_{v,down}$ and $h_{v,up}$ , which correspond to the values that will be used when v sends a message upwards (towards the root) or downwards (away from the root). Then, for appropriately defined functions F, G (depending on the potentials J and h), one can “simulate” the updates in (5):

$$
h _ {v, \mathrm{up}} ^ {(t + 1)} \leftarrow F \left(\left\{h _ {w, \mathrm{up}} ^ {(t)}: w \in v \cup \operatorname{Children} (v) \right\}\right) \tag {6}
$$

$$
h _ {v, \text { down }} ^ {(t + 1)} \leftarrow G \left(h _ {\text { Parent } (v), \text { down }} ^ {(t)}, \left\{h _ {w, \text { up }} ^ {(t)} \right\} _ {w \in \text { Children } (v)}\right) \tag {7}
$$

Intuitively, $h_{v,up}$ captures the effective external field induced by the subtree rooted at v on $\text{Parent}(v)$ . After the upward messages propagate, the root r can compute its correct marginal. Once $h_{\text{Parent}(v),\text{down}}$ is the correct marginal for $\text{Parent}(v)$ at some step, $h_{v,down}$ will be the correct marginal for v at all subsequent steps.

# D.2 GCN-based architectures to calculate marginals

The belief-propagation updates (5) naturally fit the general edge-message passing paradigm from (2). In fact, they fit even more closely a “directed” version of the paradigm, in which each edge $\{i,j\}$ maintains two embeddings $h_{i\rightarrow j}, h_{j\rightarrow i}$ , such that the embedding for direction $h_{i\rightarrow j}$ depends on the embeddings $\{h_{k\rightarrow i}\}_{\{k,i\}\in E}$ . With this modification to the standard edge GCN architecture Eq. (4), it is straightforward to implement (5) with one layer, using a particular choice of activation functions and weight matrices W (since, in particular, in our dataset all edge potentials $J_{i,j}$ are set to 1). Similarly, with a directed version of the node GCN architecture Eq. (3), where each node maintains an “up” embedding as well as a “down” embedding, it is straightforward to implement the “node-based” dynamic programming solution (6)-(7).

We call the architectures that do not maintain directionality Node-U and Edge-U (depending on whether they use a node-based or edge-based GCN). We call the “directed” architectures Node-D and Edge-D respectively. Since there are only initial node features (input as node potentials $\{h_{i}\}_{i\in}$ ), for the edge based architectures we initialize the edge features as a concatenation of the node features of the endpoints of the edge. The results we report for each architecture are the best over a sweep of depth $\in\{5,10,15,20,25,30\}$ and width $\in\{10,32,64\}$ .

# D.3 Edge-based models improve over node-based models

In Figure 2 we show the results for several tree topologies: a complete binary tree (of size 31), a path graph (of size 30), and uniformly randomly chosen trees of size 30 (the results in Figure 2 are averaged over 3 samples of tree). The architectures in the legend (Node-U, Edge-U, Node-D, Edge-D) are based on a standard GCN, and detailed in Section D.2

We can see that for both the undirected and directed versions, adding edge embeddings improves performance. The improved performance of all directed versions compared to their undirected

Comparison of Edge based and Node based GNNs across Graph Types   
![](images/ca4bbc726bdff047dc0fde790edc06f5c98e45dd01fea017d8098d0c5dfb13b0.jpg)

<details>
<summary>bar</summary>

| Graph Types | Edge-D | Node-D | Edge-U | Node-U |
|-------------|--------|--------|--------|--------|
| Binary Tree | 0.008  | 0.0085 | 0.022  | 0.051  |
| Line Graphs | 0.0065 | 0.0075 | 0.013  | 0.024  |
| Prufer Tree | 0.009  | 0.013  | 0.032  | 0.054  |
</details>

Figure 2: Comparison of four architectures for calculating node marginals in an Ising model. The architectures considered are node-embedding (3) and edge-embedding (4) versions of a GCN (correspondingly labeled Node-U and Edge-U), as well as their “directed” counterparts, as described in Section D.2, correspondingly labeled Node-D and Edge-D. The x-axis groups results according to the topology of the graph, the y-axis is MSE (lower is better). The mean and variances are reported over 3 runs for the best choice of depth and width over the sweep described in Section D.2.

counterpart is not very surprising: the standard, undirected GCN architecture treats all neighbors symmetrically — hence, the directed versions can more easily simulate something akin to the belief propagation updates (5) as well as the node-based dynamic programming (6)-(7).