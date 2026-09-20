# BEYOND WEISFEILER-LEHMAN: A QUANTITATIVE FRAMEWORK FOR GNN EXPRESSIVENESS

Bohang Zhang $^{1*†}$ Jingchu Gai $^{1*}$ Yiheng Du $^{1}$ Qiwei Ye $^{2}$ Di He $^{1}$ Liwei Wang $^{1}$

$^{1}$ Peking University $^{2}$ Beijing Academy of Artificial Intelligence

zhangbohang@pku.edu.cn, {gaijingchu,duyiheng}@stu.pku.edu.cn

qwye@baai.ac.cn, {dihe,wanglw}@pku.edu.cn

# ABSTRACT

Designing expressive Graph Neural Networks (GNNs) is a fundamental topic in the graph learning community. So far, GNN expressiveness has been primarily assessed via the Weisfeiler-Lehman (WL) hierarchy. However, such an expressivity measure has notable limitations: it is inherently coarse, qualitative, and may not well reflect practical requirements (e.g., the ability to encode substructures). In this paper, we introduce a unified framework for quantitatively studying the expressiveness of GNN architectures, addressing all the above limitations. Specifically, we identify a fundamental expressivity measure termed homomorphism expressivity, which quantifies the ability of GNN models to count graphs under homomorphism. Homomorphism expressivity offers a complete and practical assessment tool: the completeness enables direct expressivity comparisons between GNN models, while the practicality allows for understanding concrete GNN abilities such as subgraph counting. By examining four classes of prominent GNNs as case studies, we derive simple, unified, and elegant descriptions of their homomorphism expressivity for both invariant and equivariant settings. Our results provide novel insights into a series of previous work, unify the landscape of different subareas in the community, and settle several open questions. Empirically, extensive experiments on both synthetic and real-world tasks verify our theory, showing that the practical performance of GNN models aligns well with the proposed metric.

# 1 INTRODUCTION

Owing to the ubiquity of graph-structured data in numerous applications, Graph Neural Networks (GNNs) have achieved enormous success in the field of machine learning over the past few years. However, one of the most prominent drawbacks of popular GNNs lies in the limited expressive power. In particular, Morris et al. (2019); Xu et al. (2019) showed that Message Passing GNNs (MPNNs) are intrinsically bounded by the 1-dimensional Weisfeiler-Lehman test (1-WL) in distinguishing non-isomorphic graphs (Weisfeiler & Lehman, 1968). Since then, the Weisfeiler-Lehman hierarchy has become a yardstick to measure the expressiveness and guide designing more powerful GNN architectures (see Appendix A.1 for an overview of representative approaches in this area).

However, as more and more architectures have been proposed, the limitations of the WL hierarchy are becoming increasingly evident. First, the WL hierarchy is arguably too coarse to evaluate the expressive power of practical GNN models (Morris et al., 2022; Puny et al., 2023). On one hand, architectures inspired by higher-order WL tests (Maron et al., 2019b;a; Morris et al., 2019) often suffer from substantial computation/memory costs. On the other hand, most practical and efficient GNNs are only proved to be strictly more expressive than 1-WL by leveraging toy example graphs (e.g., Zhang & Li, 2021; Bevilacqua et al., 2022; Wijesinghe & Wang, 2022a). Such a qualitative characterization may provide little insight into the models' true expressiveness. Besides, the expressive power brought from the WL hierarchy often does not align well with the one required in practice (Veličković, 2022). Hence, how to study the expressiveness of GNN models in a quantitative, systematic, and practical way remains a central research direction for the GNN community.

To address the above limitations, this paper takes a different approach by studying GNN expressivity from the following practical angle: What structural information can a GNN model encode? Since the ability to detect/count graph substructures is crucial in various real-world applications (Chen et al., 2020; Huang et al., 2023; Tahmasebi et al., 2023), many expressive GNNs have been proposed based on preprocessing substructure information (Bouritsas et al., 2022; Barceló et al., 2021; Bodnar et al., 2021b;a). However, instead of augmenting GNNs by manually preprocessed (task-specific) substructures, it is nowadays more desirable to design generic, domain-agnostic GNNs that can end-to-end learn different structural information suitable for diverse applications. This naturally gives rise to the fundamental question of characterizing the complete set of substructures prevalent GNN models can encode. Unfortunately, this problem is widely recognized as challenging even when examining simple structures like cycles (Fürer, 2017; Arvind et al., 2020; Huang et al., 2023).

Our contributions. Motivated by GNNs' ability to encode substructures, this paper presents a novel framework for quantitatively analyzing the expressive power of GNN models. Our approach is rooted in a critical discovery: given a GNN model M, the model's output representation for any graph G can be fully determined by the structural information of G over some pattern family $F^{M}$ , where $F^{M}$ corresponds to precisely all (and only) those substructures that can be “encoded” by model M. In this way, the set $F^{M}$ can be naturally viewed as an expressivity description of M: by identifying $F^{M}$ for each model M, the expressivity of different models can then be qualitatively/quantitatively compared by simply looking at their set inclusion relation and set difference.

The crux here is to define an appropriate notion of “encodability” so that $F^{M}$ can admit a simple description. We identify that a good candidate is the homomorphism expressivity: i.e., $F^{M}$ consists of all substructures that can be counted by model M under homomorphism (see Section 2 for a formal definition). Homomorphism is a foundational concept in graph theory (Lovász, 2012) and is linked to many important topics such as graph coloring, graph matching, and subgraph counting. With this concept, we are able to give complete, unified, and surprisingly elegant descriptions of the pattern family $F^{M}$ for a wide range of mainstream GNN architectures listed below:

• MPNN (e.g., Gilmer et al., 2017; Hamilton et al., 2017; Kipf & Welling, 2017; Xu et al., 2019);   
- Subgraph GNN (You et al., 2021; Zhang & Li, 2021; Bevilacqua et al., 2022; Qian et al., 2022);   
- Local GNN (Morris et al., 2020; 2022; Zhang et al., 2023a; Frasca et al., 2022);   
- Folklore-type GNN (Maron et al., 2019a; Zhang et al., 2023a; Feng et al., 2023).

Technically, the descriptions are based on a novel application and extension of the concept of nested ear decomposition (NED) in graph theory (Eppstein, 1992). We prove that: (i) (necessity) each model M above can count (under homomorphism) a specific family of patterns $F^{M}$ , characterized by a specific type of NED; (ii) (sufficiency) any pattern $F \notin F^{M}$ cannot be counted under homomorphism by model M; (iii) (completeness) for any graph, information collected from the homomorphism count in pattern family $F^{M}$ determines its representation computed by model M. Therefore, homomorphism expressivity is well-defined and is a complete expressivity measure for GNN models.

Our theory can be generalized in various aspects. One significant extension is the node-level and edge-level expressivity for equivariant GNNs (Azizian & Lelarge, 2021; Geerts & Reutter, 2022), which can be naturally tackled by a fine-grained analysis of NED. As another non-trivial generalization, we study higher-order GNN variants for several of the above architectures and derive results by defining higher-order NED. Both aspects demonstrate the flexibility of our proposed framework, suggesting it as a general recipe for analyzing future architectures.

Implications. Homomorphism expressivity serves as a powerful toolbox for bridging different subareas in the GNN community, providing fresh understandings of a series of known results that were previously proved in complex ways, and answering a set of unresolved open problems. First, our results can readily establish a complete expressiveness hierarchy among all the aforementioned architectures and their higher-order extensions. This recovers and extends a number of results in Morris et al. (2020); Qian et al. (2022); Zhang et al. (2023a); Frasca et al. (2022) and answers their open problems (Section 4.1). In fact, our results go far beyond revealing the expressivity gap between models: we essentially answer how large the gap is and establish a systematic approach to constructing counterexample graphs. Second, based on the relation between homomorphism and subgraph count, we are able to characterize the subgraph counting power of GNN models for all patterns at graph, node, and edge levels, significantly advancing an open direction initiated in Fürer (2017); Arvind et al. (2020) (Section 4.2). As a special case, our results extend recent findings in

Huang et al. (2023) about the cycle counting power of GNN models, highlighting that Local 2-GNN can already subgraph-count all cycles/paths within 7 nodes (even at edge-level). Third, our results provide a new toolbox for studying the polynomial expressivity proposed recently in Puny et al. (2023), extending it to various practical architectures and answering an open question (Section 4.3). Empirically, an extensive set of experiments verifies our theory, showing that the homomorphism expressivity of different models matches well with their practical performance in diverse tasks.

# 2 PRELIMINARY

Notations. We use $\{\}$ and $\{\}$ to denote sets and multisets, respectively. Given a (multi)set S, its cardinality is denoted as $|S|$ . In this paper, we consider finite, undirected, vertex-labeled graphs with no self-loops or repeated edges. Let $G = (V_{G}, E_{G}, \ell_{G})$ be a graph with vertex set $V_{G}$ , edge set $E_{G}$ , and label function $\ell_{G}$ , where each edge in $E_{G}$ is a set $\{u, v\} \subset V_{G}$ of cardinality two, and $\ell_{G}(u)$ is the label of vertex u. The rooted graph $G^{u}$ is a graph obtained from G by marking the special vertex $u \in V_{G}$ ; we can similarly consider marking two special vertices $u, v \in V_{G}$ (denote by $G^{uv}$ ). The neighbors of vertex u is denoted as $N_{G}(u) := \{v \in V_{G} : \{u, v\} \in E_{G}\}$ . A graph $F = (V_{F}, E_{F}, \ell_{F})$ is a subgraph of G if $V_{F} \subset V_{G}, E_{F} \subset E_{G}$ , and $\ell_{F}(u) = \ell_{G}(u)$ for all $u \in V_{F}$ . A simple path P in G is an edge set of the form $\{\{w_{0}, w_{1}\}, \cdots, \{w_{k-1}, w_{k}\}\} \subset E_{G}$ where $w_{i} \neq w_{j}$ for all $i \neq j$ . Here, $w_{0}$ and $w_{k}$ are called endpoints of P and other vertices are called internal points.

Homomorphism, isomorphism, and subgraph count. Given two graphs F and G, a homomorphism from F to G is a mapping $f: V_{F} \to V_{G}$ that preserves edges and labels, i.e., $\ell_{F}(u) = \ell_{G}(f(u))$ for all $u \in V_{F}$ , and $\{f(u), f(v)\} \in E_{G}$ for all $\{u, v\} \in E_{F}$ . When the mapping f exists, we say F is homomorphic to G. We denote by $\operatorname{Hom}(F, G)$ the set of all homomorphisms from F to G and define $\operatorname{hom}(F, G) = |\operatorname{Hom}(F, G)|$ , which counts the number of homomorphisms for pattern F in graph G. If f is further surjective on both vertices and edges, we call G a homomorphic image of F. Denote by $\operatorname{Spasm}(F)$ the set of all homomorphic images of F, called the spasm of F. For rooted graphs, homomorphism should additionally preserve vertex marking: i.e., if f is a homomorphism from $F^{uv}$ to $G^{xy}$ , then $f(u) = x$ and $f(v) = y$ .

A mapping $f: V_{F} \to V_{G}$ is called an isomorphism if f is a bijection and both f and its inverse $f^{-1}$ are homomorphisms. We denote by $\operatorname{Sub}(F, G)$ the set of all subgraphs of G isomorphic to F and define $\operatorname{sub}(F, G) = |\operatorname{Sub}(F, G)|$ , which counts the number of patterns F occurred in graph G as a subgraph. We note that a similar definition holds for rooted graphs (e.g., $\operatorname{sub}(F^{uv}, G^{xy})$ ).

Graph neural networks. GNNs can be generally described as graph functions that are invariant under isomorphism. To achieve such invariance, most popular GNN models follow a color refinement (CR) paradigm: they maintain a feature representation (color) for each vertex or vertex tuples and iteratively refine these features through equivariant aggregation layers. Finally, there is a global pooling layer to merge all features and obtain the graph representation. Below, we separately define the corresponding CR algorithms for four mainstream classes of GNNs studied in this paper.

\- MPNN. Given a graph $G$ , MPNN maintains a color $\chi_G^{\mathsf{MP}}(u)$ for each vertex $u \in V_G$ . Initially, the color only depends on the vertex label, i.e., $\chi_G^{\mathsf{MP},(0)}(u) = \ell_G(u)$ . Then, in each iteration, the color is refined by the following update formula (where hash is a perfect hash function):

$$
\chi_ {G} ^ {\mathrm{MP}, (t + 1)} (u) = \operatorname{hash} \left(\chi_ {G} ^ {\mathrm{MP}, (t)} (u), \{\{\chi_ {G} ^ {\mathrm{MP}, (t)} (v): v \in N _ {G} (u) \} \}\right). \tag {1}
$$

After a sufficient number of iterations, the colors become stable. We denote by $\chi_{G}^{\mathrm{MP}}(u)$ the stable color of u, which is also the node feature of u computed by the MPNN. The graph representation is defined as the multiset of node colors, i.e., $\chi_{G}^{\mathrm{MP}}(G)=\{\{\chi_{G}^{\mathrm{MP}}(u):u\in V_{G}\}\}$ .

\- Subgraph GNN. It treats a graph $G$ as a set of subgraphs $\{G^u : u \in V_G\}$ , each obtained from $G$ by marking a special vertex $u \in V_G$ . Subgraph GNN maintains a color $\chi_G^{\text{Sub}}(u, v)$ for each vertex $v$ in graph $G^u$ . Initially, $\chi_G^{\text{Sub}, (0)}(u, v) = (\ell_G(v), \mathbb{I}[u = v])$ , where the latter term distinguishes the special mark. It then runs MPNNs independently on each graph $G^u$ :

$$
\chi_ {G} ^ {\text { Sub }, (t + 1)} (u, v) = \operatorname{hash} \left(\chi_ {G} ^ {\text { Sub }, (t)} (u, v), \{\{\chi_ {G} ^ {\text { Sub }, (t)} (u, w): w \in N _ {G} (v) \} \}\right). \tag {2}
$$

Denote the stable color of $(u,v)$ as $\chi_{G}^{\mathrm{Sub}}(u,v)$ . The node feature of u computed by Subgraph GNN is defined by merging all colors in $G^{u}$ , i.e., $\chi_{G}^{\mathrm{Sub}}(u):=\mathrm{hash}\left(\{\{\chi_{G}^{\mathrm{Sub}}(u,v):v\in V_{G}\}\}\right)$ . Finally, the graph representation is defined as $\chi_{G}^{\mathrm{Sub}}(G)=\{\{\chi_{G}^{\mathrm{Sub}}(u):u\in V_{G}\}\}$ .

\- Local GNN. Inspired by the $k$ -WL test (Grohe, 2017), Local $k$ -GNN is defined by replacing all global aggregations in $k$ -WL by sparse ones that only aggregate local neighbors, yielding a much more efficient CR algorithm. As an example, the iteration of Local 2-GNN has the following form and enjoys the same computational complexity as a Subgraph GNN.

$$
\chi_ {G} ^ {\mathrm{L}, (t + 1)} (u, v) = \operatorname{hash} \left(\chi_ {G} ^ {\mathrm{L}, (t)} (u, v), \{\{\chi_ {G} ^ {\mathrm{L}, (t)} (w, v): w \in N _ {G} (u) \} \}, \{\{\chi_ {G} ^ {\mathrm{L}, (t)} (u, w): w \in N _ {G} (v) \} \}\right). \tag {3}
$$

Initially, $\chi_G^{\mathsf{L},(0)}(u,v) = (\ell_G(u),\ell_G(v),\mathbb{I}[u = v],\mathbb{I}[\{u,v\} \in E_G])$ , which is called the isomorphism type of vertex pair $(u,v))$ . We similarly denote the stable color as $\chi_G^{\mathsf{L}}(u,v)$ and define the node feature $\chi_G^{\mathsf{L}}(u)$ and graph representation $\chi_G^{\mathsf{L}}(G)$ as in the Subgraph GNN.

\- Folklore-type GNN. The Folklore GNN (FGNN) is inspired by the standard $k$ -FWL test (Cai et al., 1992). As an example, the iteration formula of 2-FGNN is written as follows:

$$
\chi_ {G} ^ {\mathsf {F}, (t + 1)} (u, v) = \operatorname{hash} \left(\chi_ {G} ^ {\mathsf {F}, (t)} (u, v), \{\left(\chi_ {G} ^ {\mathsf {F}, (t)} (w, v), \chi_ {G} ^ {\mathsf {F}, (t)} (u, w)\right): w \in V _ {G} \} \}\right). \tag {4}
$$

One can similarly consider the more efficient Local 2-FGNN by only aggregating local neighbors, which has the same computational complexity as Local 2-GNN and Subgraph GNN:

$$
\begin{array}{l} \chi_ {G} ^ {\mathrm{LF}, (t + 1)} (u, v) = \operatorname{hash} \left(\chi_ {G} ^ {\mathrm{LF}, (t)} (u, v), \left\{\left(\chi_ {G} ^ {\mathrm{LF}, (t)} (w, v), \chi_ {G} ^ {\mathrm{LF}, (t)} (u, w)\right): w \in N _ {G} (u) \cup N _ {G} (v) \right\} \right\}. \\ \text { The   stable   color,   node   feature,and   graph   representation   can   be   similarly   defined. } \end{array} \tag {5}
$$

Finally, we note that the latter three types of GNNs can be naturally generalized into higher-order variants. We give a general definition of all these architectures in Appendix E.1. For the base case of k = 1, Subgraph $(k-1)$ -GNN, Local k-GNN, and Local k-FGNN all reduce to the MPNN.

# 3 HOMOMORPHISM EXPRESSIVITY OF GRAPH NEURAL NETWORKS

# 3.1 HOMOMORPHISM EXPRESSIVITY

Given a GNN model M and a substructure F, we say M can count graph F under homomorphism if, for any graph G, the graph representation $\chi_{G}^{M}(G)$ determines the homomorphism count $\operatorname{hom}(F,G)$ . In other words, $\chi_{G}^{M}(G)=\chi_{H}^{M}(H)$ implies $\operatorname{hom}(F,G)=\operatorname{hom}(F,H)$ for any graphs G,H. The central question studied in this paper is, what substructures F can a GNN model M count under homomorphism? This gives rise to the notion of homomorphism expressivity defined below:

Definition 3.1. The homomorphism expressivity of a GNN model $M$ , denoted by $\mathcal{F}^M$ , is a family of (labeled) graphs satisfying the following conditions:

a) For any two graphs $G, H$ , $\chi_G^M (G) = \chi_H^M (H)$ iff $\hom (F,G) = \hom (F,H)$ for all $F\in \mathcal{F}^M$ ;   
b) $\mathcal{F}^M$ is maximal, i.e., for any graph $F \notin \mathcal{F}^M$ , there exists a pair of graphs $G, H$ such that $\chi_G^M(G) = \chi_H^M(H)$ and $\operatorname{hom}(F, G) \neq \operatorname{hom}(F, H)$ .

Example 3.2. As a simple example, consider a maximally expressive GNN M that can solve the graph isomorphism problem, i.e., it computes the same representation for two graphs iff they are isomorphic. Then, $F^{M}$ contains all graphs. This is a classic result proved in Lovász (1967).

The significance of homomorphism expressivity can be justified in the following aspects. First, it is a complete expressivity measure. Based on item (a), the homomorphism count within $F^{M}$ essentially captures all information embedded in the graph representation computed by model M. This contrasts with previously studied metrics such as the ability to compute biconnectivity properties (Zhang et al., 2023b) or count cycles (Huang et al., 2023), which only reflects restricted aspects of expressivity. Second, homomorphism expressivity is a quantitative measure and is much finer than qualitative expressivity results obtained from the graph isomorphism test. Specifically, by item (a), a GNN model $M_{1}$ is more expressive than another model $M_{2}$ in distinguishing non-isomorphic graphs iff $F^{M_{2}} \subset F^{M_{1}}$ . Furthermore, by item (b), $M_{1}$ is strictly more expressive than $M_{2}$ iff $F^{M_{2}} \subsetneq F^{M_{1}}$ , and the expressivity gap can be quantitatively understood via the set difference $F^{M_{1}} \setminus F^{M_{2}}$ .

Consequently, by deriving which graphs are encompassed in the graph family $\mathcal{F}^M$ , homomorphism expressivity provides a novel way to analyze and compare the expressivity of GNN models. In the next subsection, we will give exact characterizations of $\mathcal{F}^M$ for all models $M$ defined in Section 2.

![](images/7eac64cc534e7899ef9763e2087eba8aed8d02646a7f8edec0395ef737f69d04.jpg)  
(a) Illustration of NED

![](images/4864db8ac59e3bf18c08d19b2e5ea8aa015ad02aa22f724f7c868810b514db40.jpg)

![](images/44108fd26e4135ba3c26ba40997fd55beb8cccfcb1fade9dd8b8584e70ce5c54.jpg)

![](images/b23e3a0f626a30a8817f9cdcfbfc74726b5d40516a0b44bca76b6f618d8a18aa.jpg)

![](images/c298d737708f5b3e1e253bc08996ffa54c7d0a0931aee51b6bb303df7e434260.jpg)

![](images/973fd4248552407c694985d19a0634fcd5f6a5f2370aa75342b6c84aad4b2ab8.jpg)  
(b) Examples of endpoint-shared/strong/almost-strong/general NED   
Figure 1: Illustration of NED and its variants. The number j next to each edge indicates that the edge belongs to the ear $P_{j}$ . Different colors represent different ears. See Figure 9 for more examples.

# 3.2 MAIN RESULTS

To derive our main results, we leverage a concept in graph theory known as nested ear decomposition (NED), which is originally introduced in Eppstein (1992). Here, we adapt the definition as follows:

Definition 3.3. Given a graph G, a NED P is a partition of the edge set $E_{G}$ into a sequence of simple paths $P_{1}, \cdots, P_{m}$ (called ears), which satisfies the following conditions:

- Any two ears $P_{i}$ and $P_{j}$ with indices $1 \leq i < j \leq c$ do not intersect, where $c$ is the number of connected components of $G$ .   
- For each ear $P_{j}$ with index $j > c$ , there is an ear $P_{i}$ with index $1 \leq i < j$ such that one or two endpoints of $P_{j}$ lie in ear $P_{i}$ (we say $P_{j}$ is nested on $P_{i}$ ). Moreover, except for the endpoints lying in ear $P_{i}$ , no other vertices in $P_{j}$ are in any previous ear $P_{k}$ for $1 \leq k < j$ . If both endpoints of $P_{j}$ lie in $P_{i}$ , the subpath in $P_{i}$ that shares the endpoints of $P_{j}$ is called the nested interval of $P_{j}$ in $P_{i}$ , denoted as $I(P_{j}) \subset P_{i}$ . If only one endpoint lies in $P_{i}$ , define $I(P_{j}) = \emptyset$ .   
- For all ears $P_{j}$ , $P_{k}$ with $c < j < k \leq m$ , either $I(P_{j}) \cap I(P_{k}) = \emptyset$ or $I(P_{j}) \subset I(P_{k})$ .

Intuitively, Definition 3.3 states that the relation between different ears forms a forest, in that each ear is nested on its parent. Moreover, the nested intervals either do not intersect or have inclusion relations for different children of the same parent ear. We give illustrations of NED in Figure 1.

In this paper, we considerably extend the concept of NED to several variants defined below:

- Endpoint-shared NED: a NED is called endpoint-shared if all ears with non-empty nested intervals share a common endpoint (see Figure 1(b,1)).   
- Strong NED: a NED is called strong if for any two children $P_{j}$ , $P_{k}$ ( $j < k$ ) nested on the same parent ear, we have $I(P_{j}) \subset I(P_{k})$ (see Figure 1(b,2)).   
- Almost-strong NED: a NED is called almost-strong if for any children $P_j$ , $P_k$ ( $j < k$ ) nested on the same parent ear and $|I(P_j)| > 1$ , we have $I(P_j) \subset I(P_k)$ (see Figure 1(b,3)).

We are now ready to present our main results:

Theorem 3.4. For all GNN models $M$ defined in Section 2, the graph family $\mathcal{F}^M$ satisfying Definition 3.1 exists (and is unique). Moreover, each $\mathcal{F}^M$ can be separately described below:

- MPNN: $\mathcal{F}^{\text{MP}} = \{F : F \text{ is a forest}\}$ ;  
- Subgraph GNN: $\mathcal{F}^{\text{Sub}} = \{F : F \text{ has an endpoint-shared NED}\}$ ;   
- Local 2-GNN: $\mathcal{F}^{\mathrm{L}} = \{F : F \text{ has a strong NED}\}$ ;   
- Local 2-FGNN: $\mathcal{F}^{\text{LF}} = \{F : F \text{ has an almost-strong NED}\}$ ;   
- 2-FGNN: $\mathcal{F}^{\mathsf{F}} = \{F : F \text{ has a NED}\}$ .

Theorem 3.4 gives a unified description of the homomorphism expressivity for all popular GNN models defined in Section 2. Despite the elegant conclusion, the proof process is actually involved and represents a major technical contribution, so we present a proof sketch below. Our proof is divided into three parts, presented in Appendices C.2 to C.4. First, we show the existence of $\mathcal{F}^M$ for each model $M$ based on a beautiful theory developed in Dell et al. (2018). Using the technique of unfolding tree, we prove that $\mathcal{F}^M$ at least contains all graphs $F$ that allow a specific type of tree decomposition (Diestel, 2017), and the homomorphism information of these graphs determines the representation of any graph $G$ computed by $M$ (i.e., Definition 3.1(a) holds). However, characterizing $\mathcal{F}^M$ in terms of tree decomposition is sophisticated and not intuitive for most models $M$ . In the next step, we give an equivalent description of $\mathcal{F}^M$ based on novel extensions of NED proposed in Definition 3.3, which is simpler and more elegant. In the last step, we prove that $\mathcal{F}^M$ does not contain other graphs. This is achieved by building non-trivial relations between three distinct theoretical tools: tree decomposition, pebble game (Cai et al., 1992), and Fürer graph (Fürer, 2001). Through a fine-grained analysis of the Fürer graphs expanded by $F \notin \mathcal{F}^M$ (see Theorems C.47 and C.53), we show they are precisely a pair of graphs satisfying Definition 3.1(b), thus concluding the proof.

Discussions with Dell et al. (2018). Our work significantly extends a beautiful theory developed by Dell, Grohe, and Rattan, who showed that a pair of graphs G, H are indistinguishable by 1-WL iff $\operatorname{hom}(F,G)=\operatorname{hom}(F,H)$ for all trees F, and more generally, they are indistinguishable by k-FWL iff $\operatorname{hom}(F,G)=\operatorname{hom}(F,H)$ for all graphs F of bounded treewidth k. In this paper, we successfully generalize these results to a broad range of practical GNN models. Moreover, two distinct contributions are worth discussing. First, we highlight a key insight that homomorphism can serve as a fundamental expressivity measure, which has far-reaching consequences as will be elaborated in Section 4. To show that $F^{M}$ is a valid expressivity measure, we additionally prove a non-trivial result that $F^{M}$ is maximal (Definition 3.1(b)). Without this crucial property, $F^{M_{1}}\supsetneq F^{M_{2}}$ will not necessarily mean that model $M_{1}$ is strictly more expressive than $M_{2}$ , thus preventing any quantitative comparison between models. Second, Dell et al. (2018) leveraged treewidth to describe results, which, unfortunately, cannot be applied to most GNN models studied here. Instead, we resort to the novel concept of NED, by which we successfully derive unified and elegant descriptions for all models. Moreover, as will be shown later, NED is quite flexible and can be naturally generalized to node/edge-level expressivity, which is not studied in prior work.

Finally, we remark that one can derive an equivalent (perhaps simpler) description of $F^{Sub}$ , based on the fact that a graph F has an endpoint-shared NED iff F becomes a forest when deleting the shared endpoint. Formally, denoting by $F\backslash\{u\}$ the induced subgraph of F over $V_{F}\backslash\{u\}$ , we have

Corollary 3.5. $\mathcal{F}^{\mathrm{Sub}} = \{F:\exists u\in V_F$ s.t. $F\backslash \{u\}$ is a forest}.

# 3.3 EXTENDING TO NODE/EDGE-LEVEL EXPRESSIVITY

So far, this paper mainly focuses on the graph-level expressivity, i.e., what information is encoded in the graph representation. In this subsection, we extend all results in Theorem 3.4 to the more fine-grained node/edge-level expressivity by answering what information is encoded in the node/edge features of a GNN (i.e., $\chi_G^M(u)$ or $\chi_G^M(u,v)$ in Section 2). This yields the following definition:

Definition 3.6. The node-level homomorphism expressivity of a GNN model M, denoted by $F_{n}^{M}$ , is a family of connected rooted graphs satisfying the following conditions:

a) For any connected graphs $G, H$ and vertices $u \in V_G$ , $v \in V_H$ , $\chi_G^M(u) = \chi_H^M(v)$ iff $\hom(F^w, G^u) = \hom(F^w, H^v)$ for all $F^w \in \mathcal{F}_{\mathfrak{n}}^M$ ;   
b) For any rooted graph $F^w \notin \mathcal{F}_n^M$ , there exists a pair of connected graphs $G, H$ and two vertices $u \in V_G, v \in V_H$ such that $\chi_G^M(u) = \chi_H^M(v)$ and $\mathrm{hom}(F^w, G^u) \neq \mathrm{hom}(F^w, H^v)$ .

One can similarly define the edge-level homomorphism expressivity $F_{e}^{M}$ to be a family of connected rooted graphs, each marking two special vertices (we omit the definition for clarity). The following result exactly characterizes $F_{n}^{M}$ and $F_{e}^{M}$ for all models M considered in this paper:

Theorem 3.7. For all model M defined in Section 2, $F_{n}^{M}$ and $F_{e}^{M}$ (except MPNN) exist. Moreover,

- MPNN: $\mathcal{F}_{\mathfrak{n}}^{\mathsf{MP}} = \{F^{w} : F \text{ is a tree}\}$ ;  
- Subgraph GNN:

$$
\begin{array}{l} \mathcal {F} _ {n} ^ {\text {Sub}} = \{F ^ {w}: F \text {has a NED with shared endpoint w} \} = \{F ^ {w}: F \backslash \{w \} \text {is a forest} \}, \\ \mathcal {F} _ {e} ^ {\text {Sub}} = \{F ^ {w x}: F \text {has a NED with shared endpoint w} \} = \{F ^ {w x}: F \backslash \{w \} \text {is a forest} \}; \end{array}
$$

\- 2-FGNN: $\mathcal{F}_{\mathrm{n}}^{\mathrm{F}} = \{F^{w} : F \text{ has a NED where } w \text{ is an endpoint of the first ear}\}$ , $\mathcal{F}_{\mathrm{e}}^{\mathrm{F}} = \{F^{wx} : F \text{ has a NED where } w \text{ and } x \text{ are endpoints of the first ear}\}$ .

The cases of Local 2-GNN and Local 2-FGNN are similar to 2-FGNN by replacing “NED” with “strong NED” and “almost-strong NED”, respectively.

In summary, the node/edge-level homomorphism expressivity can be naturally described using NED by further specifying the endpoints of the first ear.

# 3.4 EXTENDING TO HIGHER-ORDER GNNs

Finally, we discuss how our results can be naturally extended to higher-order GNNs, thus providing a complete picture of the homomorphism expressivity hierarchy for infinitely many architectures. We focus on three representative examples: Subgraph k-GNN (Qian et al., 2022), Local k-GNN (Morris et al., 2020), and k-FGNN (Azizian & Lelarge, 2021). Subgraph k-GNN extracts a graph $G^{u}$ for each vertex k-tuple $u \in V_{G}^{k}$ and runs MPNNs independently, which recovers Subgraph GNN when k = 1. As the reader may have guessed, the following result exactly parallels Corollary 3.5:

Theorem 3.8. The homomorphism expressivity of Subgraph k-GNN exists and can be described as $\mathcal{F}^{\mathrm{Sub}(k)} = \{F:\exists U\subset V_F$ s.t. $|U|\leq k$ and $F\backslash U$ is a forest}.

We next turn to Local k-GNN. To describe the result, we introduce a novel extension of Definition 3.3, called the k-order ear. Intuitively, it is formed by a graph of no more than k vertices, plus k paths each linking to a vertex in the graph (see Figure 2(a) for an illustration). Note that a 2-order ear is exactly a simple path. Then, we can naturally define the nested “interval” (see the solid orange lines in Figure 2(b) for an illustration) and thus define the concept of k-order strong NED. Due to space limit, a formal definition is deferred to Definition E.3. We have the following main result:

![](images/f1a6d581f4adb6490a01b959eea3c513aa5fc86121ad5f8d7f2d92ca7c9e9212.jpg)  
Figure 2: Illustration of higher-order ears. Each curve indicates a path (with possibly zero length) and each straight segment indicates an edge.

Theorem 3.9. The homomorphism expressivity of Local $k$ -GNN exists and can be described as $\mathcal{F}^{\mathsf{L}(k)} = \{F:F$ has a $k$ -order strong NED\}.

Finally, let us consider the standard k-FGNN (or equivalently, the k-FWL). Unfortunately, we cannot find a description of its homomorphism expressivity based on some form of higher-order NED; nevertheless, it is easy to describe the results using the notion of treewidth (see Definition C.2). Specifically, denoting $\operatorname{tw}(F)$ to be the treewidth of graph F, we have the following result:

Theorem 3.10. The homomorphism expressivity of $k$ -FGNN exists and can be described as $\mathcal{F}^{\mathbb{F}(k)} = \{F : \mathrm{tw}(F) \leq k\}$ .

Interestingly, one can see that $\mathcal{F}^{\mathrm{Sub}(0)}$ , $\mathcal{F}^{\mathrm{L}(1)}$ , and $\mathcal{F}^{\mathrm{F}(1)}$ all degenerate to the family of forests, which coincides with the fact that all these higher-order GNNs reduces to MPNN for the base case.

# 4 IMPLICATIONS

The previous section has provided a complete description of the homomorphism expressivity for a variety of GNN models. In this section, we highlight the significance of these results through three different contexts. We will show how homomorphism expressivity can be used to link different GNN subareas, provide new insights into various known results, and answer a number of open problems.

# 4.1 QUALITATIVE EXPRESSIVITY COMPARISON

One direct corollary of Theorem 3.4 is that it readily enables expressivity comparison among all models in Section 2. This can be summarized below:

Corollary 4.1. Under the notation of Theorem 3.4, $\mathcal{F}^{\mathrm{MP}}\subsetneq \mathcal{F}^{\mathrm{Sub}}\subsetneq \mathcal{F}^{\mathrm{L}}\subsetneq \mathcal{F}^{\mathrm{LF}}\subsetneq \mathcal{F}^{\mathrm{F}}$ . Thus, the expressive power of the following GNN models strictly increases in order (in terms of distinguishing non-isomorphic graphs): MPNN, Subgraph GNN, Local 2-GNN, Local 2-FGNN, and 2-FGNN.

Proof. $F^{MP} \subset F^{Sub}$ follows from Corollary 3.5 and the fact that deleting any vertex of a forest yields a forest. $F^{Sub} \subset F^{L}$ follows by the fact that any endpoint-shared NED is a strong NED. $F^{L} \subset F^{LF} \subset F^{F}$ follows similarly since any strong NED is an almost-strong NED and any almost-strong NED is a NED. To prove strict separation results, one can check that the four graphs in Figure 1(b) precisely reveal the gap between each pair of graph families, thus concluding the proof. □

Corollary 4.1 recovers a series of results recently proved in Zhang et al. (2023a); Frasca et al. (2022). Compared to their results, our approach draws a much clearer picture of the expressivity gap between different architectures and essentially answers how large the gaps are. Moreover, we provide systematic guidance for finding counterexample graphs that unveil the expressivity gap: as shown in Corollary C.54, any graph $F \in \mathcal{F}^{M_2} \setminus \mathcal{F}^{M_1}$ immediately gives a pair of non-isomorphic graphs that reveals the gap between models $M_1$ and $M_2$ . We note that this readily recovers the counterexamples constructed in Zhang et al. (2023a) and greatly simplifies their sophisticated case-by-case analysis.

We next turn to three types of higher-order GNNs studied in Section 3.4, for which we can establish a complete expressiveness hierarchy, as presented in Corollary 4.2. A graphical illustration of these results is given in Figure 3.

![](images/fca9ed91942914b08d04507aa0dd77121cccd68d26f7701733b23158a27efb31.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["MPNN Subgraph 0-GNN Local 1-GNN (1-WL)"] -->|strict| B["Subgraph 1-GNN"]
    B -->|strict| C["Local 2-GNN"]
    B -->|incomparable| D["2-FGNN (2-FWL)"]
    D -->|strict| E["Local 3-GNN"]
    D -->|incomparable| F["3-FGNN (3-FWL)"]
    F -->|strict| G["Subgraph 2-GNN"]
    G -->|strict| H["Subgraph 3-GNN"]
    G -->|incomparable| I["Local 4-GNN"]
    H -->|strict| J["..."]
    I -->|incomparable| K["..."]
```
</details>

Figure 3: Expressiveness hierarchy of MPNN, Subgraph GNN, Local GNN, and FGNN.

Corollary 4.2. Under the notations in Section 3.4, for any k > 0, the following hold:

a) $\mathcal{F}^{\mathrm{Sub}(k - 1)}\subsetneq \mathcal{F}^{\mathrm{Sub}(k)}$ . I.e., the expressive power of Subgraph $k$ -GNN strictly increases with $k$   
b) $\mathcal{F}^{\mathsf{L}(k)}\subsetneq \mathcal{F}^{\mathsf{L}(k + 1)}$ . I.e., the expressive power of Local $k$ -GNN strictly increases with $k$   
c) $\mathcal{F}^{\mathrm{Sub}(k)}\subsetneq \mathcal{F}^{\mathrm{L}(k + 1)}$ . I.e., Local $(k + 1)$ -GNN is strictly more expressive than Subgraph $k$ -GNN;   
d) $\mathcal{F}^{\mathsf{F}(k)}\subsetneq \mathcal{F}^{\mathsf{L}(k + 1)}\subsetneq \mathcal{F}^{\mathsf{F}(k + 1)}$ . I.e., the expressive power of Local $(k + 1)$ -GNN lies strictly between $k$ -FWL and $(k + 1)$ -FWL;   
e) $\mathcal{F}^{\mathrm{Sub}(k)}\subsetneq \mathcal{F}^{\mathrm{F}(k + 1)}$ , and for all $k > 1$ , $\mathcal{F}^{\mathrm{Sub}(k)}\backslash \mathcal{F}^{\mathrm{F}(k + 1)}\neq \emptyset$ and $\mathcal{F}^{\mathrm{F}(k + 1)}\backslash \mathcal{F}^{\mathrm{Sub}(k)}\neq \emptyset$ . In other words, the expressive power of Subgraph $k$ -GNN lies strictly within $(k + 1)$ -FWL, but it is incomparable to $k$ -FWL when $k > 1$ .

Corollary 4.2 recovers results in Morris et al. (2020); Qian et al. (2022) and further answers two open problems. First, Corollary 4.2(c) is a new result that bridges Morris et al. (2020) with Qian et al. (2022) and partially answers an open question in Zhang et al. (2023a, Appendix C). Another new result is Corollary 4.2(d), which essentially answers a fundamental open problem raised in Frasca et al. (2022, Appendix E), showing that their proposed RelGN(k) model is bounded by k-FWL with an inherent expressivity gap (see Appendix E.4 for a detailed discussion). To sum up, all these challenging open problems become straightforward through the lens of homomorphism expressivity.

# 4.2 SUBGRAPH COUNTING POWER

The significance of homomorphism expressivity can go much beyond qualitative comparisons between models. As another implication, it provides a systematic way to study GNNs' ability to encode structural information such as subgraph count, which has been found crucial in numerous practical applications. Specifically, a well-known result in graph theory states that, for any graphs $F, G$ , the subgraph count $\text{sub}(F, G)$ can be determined by $\text{hom}(\tilde{F}, G)$ where $\tilde{F}$ ranges over all homomorphic images of $F$ (i.e., $\text{Spasm}(F)$ , see Section 2) (Lovász, 2012; Curticapean et al., 2017).

Mathematically, given any graph $F$ , let $\operatorname{Spasm}^{\mathcal{Z}}(F)$ be any maximal set of pairwise non-isomorphic graphs chosen from $\operatorname{Spasm}(F)$ (see Figure 4(a) for an illustration). Then, we have the following linear relation for all graph $G$ :

$$
\operatorname{sub} (F, G) = \sum_ {\tilde {F} \in \operatorname{Spasm} ^ {\neq} (F)} \alpha (F, \tilde {F}) \cdot \hom (\tilde {F}, G), \tag {6}
$$

where $\alpha(F, \tilde{F}) \neq 0$ is a constant scalar coefficient independent of $G$ . Based on this formula, we can easily study the subgraph counting power of GNN models as shown in Proposition 4.4.

Definition 4.3. Given a GNN model M, we say M can subgraph-count graph F at graph-level if $\chi_{G}^{M}(G)=\chi_{H}^{M}(H)$ implies $\operatorname{sub}(F,G)=\operatorname{sub}(F,H)$ for any graphs G,H. We say M can subgraph-count rooted graph $F^{w}$ at node-level if $\chi_{G}^{M}(u)=\chi_{H}^{M}(v)$ implies $\operatorname{sub}(F^{w},G^{u})=\operatorname{sub}(F^{w},H^{v})$ for any graphs G,H and vertices $u\in V_{G},v\in V_{H}$ . We can similarly define the edge-level subgraph counting ability for rooted graphs marking two special vertices.

Proposition 4.4. For any GNN model M defined in Section 2, it can subgraph-count graph F (at graph-level) if $\tilde{F} \in F^{M}$ for all $\tilde{F} \in \text{Spasm}(F)$ . It can subgraph-count $F^{w}$ (at node-level) if $\tilde{F}^{w} \in \mathcal{F}_{n}^{M}$ for all $\tilde{F}^{w} \in \text{Spasm}(F^{w})$ . A similar result holds for edge-level subgraph counting.

The above proposition offers a simple way to affirm the ability of a GNN model $M$ to subgraph-count any pattern at graph/node/edge-level. On the other hand, one may wonder whether the converse direction also holds, i.e., $M$ cannot subgraph-count $F$ if there exists a homomorphic image $\tilde{F} \in \operatorname{Spasm}(F)$ such that $\tilde{F} \notin \mathcal{F}^M$ . We find that it is indeed the case. Specifically, if the set $\operatorname{Spasm}(F) \backslash \mathcal{F}^M$ is not empty, then one can always find a pair of counterexample graphs $G, H$ such that $\chi_G^M(G) = \chi_H^M(H)$ but $\operatorname{sub}(F, G) \neq \operatorname{sub}(F, H)$ . We eventually arrive at the following main theorem (see Appendix G.1 for a proof):

Theorem 4.5. For any GNN model M such that their homomorphism expressivity $F^{M}$ exists, M can subgraph-count F iff $\operatorname{Spasm}(F) \subset \mathcal{F}^{M}$ . Similar results hold for rooted graphs $F^{u}/F^{uv}$ by replacing $F^{M}$ with node/edge-level homomorphism expressivity $F_{n}^{M}/F_{e}^{M}$ .

Example 4.6. As an example, we can readily characterize the cycle/path counting power of various GNNs. Denote by $C_n / P_n$ the simple cycle/path of $n$ vertices. Let $\{u, v\} \in E_{C_n}$ be any edge in $C_n$ , and $\{w, x\} \in E_{P_n}$ be any edge in $P_n$ where $w$ is an endpoint of $P_n$ . The following table lists exactly all cycles/paths each model can count at graph/node/edge-level.

<table><tr><td rowspan="2">Structure Model</td><td colspan="3">Cycle</td><td colspan="3">Path</td></tr><tr><td> $C_n$ </td><td> $C_n^u$ </td><td> $C_n^{uv}$ </td><td> $P_n$ </td><td> $P_n^w$ </td><td> $P_n^{wx}$ </td></tr><tr><td>MPNN</td><td>None</td><td>None</td><td>None</td><td>n≤3</td><td>n≤3</td><td>n≤3</td></tr><tr><td>Subgraph GNN</td><td>n≤7</td><td>n≤4</td><td>n≤4</td><td>n≤7</td><td>n≤4</td><td>n≤4</td></tr><tr><td>Local 2-GNN</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Local 2-FGNN</td><td colspan="3">n≤7</td><td colspan="3">n≤7</td></tr><tr><td>2-FGNN</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/bf7769e5bbc276047f03d27270a96124dab9ba8b61882a4fad5bac61b2b82557.jpg)

![](images/6872e65ee743ee7f48954c6a9372ef4d832b1a189f5c011343ca7e976b8f4c35.jpg)  
(a) $Spasm^{\neq}(C_{6})$ has 10 graphs.   
(b) Rooted $C_6$   
Figure 4: Illustration of homomorphic images of the 6-cycle and rooted 6-cycle.

Discussions with prior work. Our results significantly extend Huang et al. (2023) in several aspects. First, we show Subgraph GNN can count 6-cycle at graph-level by simply enumerating its spasm (see Figure 4(a)). However, it cannot count rooted 5/6-cycle at node-level because the homomorphic image can contain cycles that do not pass the marked vertex (see Figure 4(b)). This provides novel insights into Huang et al. (2023) and extends their results (albeit with a simpler analysis). Second, we reveal that Local 2-GNN can already count all cycles/paths that 2-FWL can count (even at edge-level). This identifies a new architecture with both efficiency and strong expressiveness in subgraph counting, considerably extending the finding in the concurrent work of Zhou et al. (2023b).

In Appendix G.2 (Tables 4 and 5), we summarize the statistics of all moderate-size patterns each model can count under homomorphisms/subgraphs, which enables quantitative expressivity comparisons of different models in a clear and exact manner. We also comprehensively list the counting ability of all moderate-size patterns in Table 6, which we believe can be helpful for future research.

# 4.3 POLYNOMIAL EXPRESSIVITY

As the third implication, homomorphism expressivity is closely related to the polynomial expressivity recently proposed in Puny et al. (2023). Concretely, given a model M, a graph F is in $F^{M}$ if M can express the invariant graph polynomial $P_{F}$ (defined in Puny et al. (2023), Section 2.2), and a rooted graph $F^{uv}$ is in $F_{e}^{M}$ if M can express the equivariant graph polynomial $P_{F^{uv}}$ . Based on this connection, our work introduces a novel toolbox for studying polynomial expressivity via the NED framework and offers new insights into which graph polynomials can be computed for a variety of practical GNNs. Moreover, we readily settle an open question in Puny et al. (2023), which upper bounds the polynomial expressivity for their proposed PPGN++:

Corollary 4.7. PPGN++ is bounded by (and thus as expressive as) the Prototypical edge-based model defined in Puny et al. (2023) for computing equivariant graph polynomials.

Due to space limit, please refer to Appendix H for proof and more discussions.

# 5 EXPERIMENTS

This section aims to verify our theory through a comprehensive set of experiments. In each experiment, we implement four types of GNN models listed in Section 2, i.e., MPNN, Subgraph GNN, Local 2-GNN, and Local 2-FGNN. Note that all of these models are much more efficient than 2-FWL. Our primary objective here is not to produce SOTA results, but rather to provide a unified and equitable empirical comparison among these models. To ensure fairness, we employ the same GIN-based design (Xu et al., 2019) for all models and control their model sizes and training budgets to be roughly the same on each task. Details of model configurations are given in Appendix I. Our code is available at https://github.com/subgraph23/homomorphism-expressivity.

Synthetic task. We first test whether these GNN models can easily learn homomorphism information from data as our theory predicts. We use the benchmark dataset from Zhao et al. (2022a) and comprehensively test the homomorphism expressivity at graph/node/edge-level by carefully selecting 8 substructures shown in Table 1. The reported performance is measured by the normalized

Mean Absolute Error (MAE) on the test dataset. It can be seen that the model performance indeed correlates to our theoretical predictions: (i) MPNN cannot encode any substructure under homomorphism; (ii) Subgraph GNN cannot encode the 2th, 3rd, 5th, 7th, 8th substructures; (iii) Local 2-GNN cannot encode the 3rd and 8th substructures; (iv) Local 2-FGNN can encode all substructures.

Cycle counting power. Cycles are important structures in numerous graph learning tasks, yet encoding them is notoriously hard for GNNs. We next test the ability of different GNN models to subgraph-count (chordal) cycles at graph/node/edge-level. We follow the setting in Frasca et al. (2022); Zhang et al. (2023a); Huang et al. (2023) and present results in Table 3 (measured by the normalized test MAE). Remarkably, despite the same computational cost and model size, Local 2-(F)GNN performs significantly better than Subgraph GNN and achieves good performance for counting all 3/4/5/6-cycles as well as chordal 4/5-cycles (even at edge-level). These results match Example 4.6 and may suggest Local 2-(F)GNN as generic, efficient, yet powerful architectures in solving chemical and biological tasks where counting cycles is essential (e.g., benzene rings).

Real-world tasks. We finally test these GNN models on three real-world benchmarks: ZINC-subset, ZINC-full (Dwivedi et al., 2020), and Alchemy (Chen et al., 2019a). Following the standard configuration, all models obey a 500K parameter budget. The results are shown in Table 2. It can be seen that the performance continues to improve when a more expressive model is used. In particular, Local 2-FGNN achieves the best performance on all tasks, suggesting that its theoretical expressivity guarantee can translate to practical performance in real-world settings.

Table 1: Experimental results on homomorphism counting. Red/blue nodes indicate marked vertices. 

<table><tr><td rowspan="2">Model\Task</td><td colspan="3">Graph-level</td><td colspan="2">Node-level</td><td colspan="3">Edge-level</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MPNN</td><td>.300</td><td>.233</td><td>.254</td><td>.505</td><td>.478</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Subgraph GNN</td><td>.011</td><td>.015</td><td>.012</td><td>.004</td><td>.058</td><td>.003</td><td>.058</td><td>.048</td></tr><tr><td>Local 2-GNN</td><td>.008</td><td>.008</td><td>.010</td><td>.003</td><td>.004</td><td>.005</td><td>.006</td><td>.008</td></tr><tr><td>Local 2-FGNN</td><td>.003</td><td>.005</td><td>.004</td><td>.005</td><td>.005</td><td>.007</td><td>.007</td><td>.008</td></tr></table>

Table 2: Experimental results on ZINC and Alchemy datasets. See Appendix I.4 for comparisons of more GNN models in literature. 

<table><tr><td rowspan="2">TaskModel</td><td colspan="2">ZINC</td><td rowspan="2">Alchemy</td></tr><tr><td>Subset</td><td>Full</td></tr><tr><td>MPNN</td><td>.138 ± .006</td><td>.030 ± .002</td><td>.122 ± .002</td></tr><tr><td>Subgraph GNN</td><td>.110 ± .007</td><td>.028 ± .002</td><td>.116 ± .001</td></tr><tr><td>Local 2-GNN</td><td>.069 ± .001</td><td>.024 ± .002</td><td>.114 ± .001</td></tr><tr><td>Local 2-FGNN</td><td>.064 ± .002</td><td>.023 ± .001</td><td>.111 ± .001</td></tr></table>

Table 3: Experimental results on the (Chordal) Cycle Counting task. 

<table><tr><td rowspan="2">Model\Task</td><td colspan="6">Graph-level</td><td colspan="6">Node-level</td><td colspan="6">Edge-level</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MPNN</td><td>.358</td><td>.208</td><td>.188</td><td>.146</td><td>.261</td><td>.205</td><td>.600</td><td>.413</td><td>.300</td><td>.207</td><td>.318</td><td>.237</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Subgraph GNN</td><td>.010</td><td>.020</td><td>.024</td><td>.046</td><td>.007</td><td>.027</td><td>.003</td><td>.005</td><td>.092</td><td>.082</td><td>.050</td><td>.073</td><td>.001</td><td>.003</td><td>.090</td><td>.096</td><td>.038</td><td>.065</td></tr><tr><td>Local 2-GNN</td><td>.008</td><td>.011</td><td>.017</td><td>.034</td><td>.007</td><td>.016</td><td>.002</td><td>.005</td><td>.010</td><td>.023</td><td>.004</td><td>.015</td><td>.001</td><td>.005</td><td>.010</td><td>.019</td><td>.005</td><td>.014</td></tr><tr><td>Local 2-FGNN</td><td>.003</td><td>.004</td><td>.010</td><td>.020</td><td>.003</td><td>.010</td><td>.004</td><td>.006</td><td>.012</td><td>.021</td><td>.004</td><td>.014</td><td>.003</td><td>.006</td><td>.012</td><td>.022</td><td>.005</td><td>.012</td></tr></table>

# 6 CONCLUSION

In this paper, we present a new framework for systematically and quantitatively studying the expressive power of various GNN architectures. Through the lens of homomorphism expressivity, we give exact descriptions of the graph family each model can encode in terms of homomorphism counting. Our framework stands as a valuable toolbox to unify the landscape between different subareas in the GNN community, providing deep insights into a number of prior works and answering their open problems. In particular, one can establish a complete expressiveness hierarchy between models, determine the subgraph counting capabilities of GNNs at graph/node/edge-level, and understand their polynomial expressivity. On the theoretical side, our results establish deep connections with a series of fundamental topics in graph theory (see Appendix A.2); On the practical side, these results closely correlate with the empirical performance of GNN models, as demonstrated through extensive experiments. Finally, Appendix B outlines several open directions for further exploration, and we believe that the homomorphism expressivity framework paves a fresh way for future study of more expressive GNNs.

# REFERENCES

Ralph Abboud, Radoslav Dimitrov, and Ismail Ilkan Ceylan. Shortest path networks for graph property prediction. In Learning on Graphs Conference, pp. 5–1. PMLR, 2022.

Vikraman Arvind, Frank Fuhlbrück, Johannes Köbler, and Oleg Verbitsky. On weisfeiler-leman invariance: Subgraph counts and related graph properties. Journal of Computer and System Sciences, 113:42–59, 2020.   
Waiss Azizian and Marc Lelarge. Expressive power of invariant and equivariant graph neural networks. In International Conference on Learning Representations, 2021.   
Franz Baader. The description logic handbook: Theory, implementation and applications. Cambridge university press, 2003.   
László Babai. Graph isomorphism in quasipolynomial time. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pp. 684–697, 2016.   
Muhammet Balcilar, Pierre Héroux, Benoit Gauzere, Pascal Vasseur, Sébastien Adam, and Paul Honeine. Breaking the limits of message passing graph neural networks. In International Conference on Machine Learning, pp. 599–608. PMLR, 2021a.   
Muhammet Balcilar, Guillaume Renton, Pierre Héroux, Benoit Gaüzère, Sébastien Adam, and Paul Honeine. Analyzing the expressive power of graph neural networks in a spectral perspective. In International Conference on Learning Representations, 2021b.   
Pablo Barceló, Egor V Kostylev, Mikael Monet, Jorge Pérez, Juan Reutter, and Juan-Pablo Silva. The logical expressiveness of graph neural networks. In 8th International Conference on Learning Representations (ICLR 2020), 2020.   
Pablo Barceló, Floris Geerts, Juan Reutter, and Maksimilian Ryschkov. Graph neural networks with local graph parameters. In Advances in Neural Information Processing Systems, volume 34, pp. 25280–25293, 2021.   
Beatrice Bevilacqua, Fabrizio Frasca, Derek Lim, Balasubramaniam Srinivasan, Chen Cai, Gopinath Balamurugan, Michael M Bronstein, and Haggai Maron. Equivariant subgraph aggregation networks. In International Conference on Learning Representations, 2022.   
Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yu Guang Wang, Pietro Liò, Guido Montufar, and Michael M. Bronstein. Weisfeiler and lehman go cellular: CW networks. In Advances in Neural Information Processing Systems, volume 34, 2021a.   
Cristian Bodnar, Fabrizio Frasca, Yuguang Wang, Nina Otter, Guido F Montufar, Pietro Lio, and Michael Bronstein. Weisfeiler and lehman go topological: Message passing simplicial networks. In International Conference on Machine Learning, pp. 1026–1037. PMLR, 2021b.   
Giorgos Bouritsas, Fabrizio Frasca, Stefanos P Zafeiriou, and Michael Bronstein. Improving graph neural network expressivity via subgraph isomorphism counting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.   
Xavier Bresson and Thomas Laurent. Residual gated graph convnets. arXiv preprint arXiv:1711.07553, 2017.   
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. International Conference on Learning Representations, 2014.   
Jin-Yi Cai, Martin Fürer, and Neil Immerman. An optimal lower bound on the number of variables for graph identification. Combinatorica, 12(4):389–410, 1992.   
Dexiong Chen, Leslie O'Bray, and Karsten Borgwardt. Structure-aware transformer for graph representation learning. In International Conference on Machine Learning, pp. 3469–3489. PMLR, 2022.   
Guangyong Chen, Pengfei Chen, Chang-Yu Hsieh, Chee-Kong Lee, Benben Liao, Renjie Liao, Weiwen Liu, Jiezhong Qiu, Qiming Sun, Jie Tang, et al. Alchemy: A quantum chemistry dataset for benchmarking ai models. arXiv preprint arXiv:1906.09427, 2019a.   
Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. On the equivalence between graph isomorphism testing and function approximation with gnns. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 15894–15902, 2019b.

Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can graph neural networks count substructures? In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 10383–10395, 2020.   
Yun Young Choi, Sun Woo Park, Youngho Woo, and U Jin Choi. Cycle to clique (cy2c) graph neural network: A sight to see beyond neighborhood aggregation. In The Eleventh International Conference on Learning Representations, 2022.   
Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Veličković. Principal neighbourhood aggregation for graph nets. In Advances in Neural Information Processing Systems, volume 33, pp. 13260–13271, 2020.   
Leonardo Cotta, Christopher Morris, and Bruno Ribeiro. Reconstruction for powerful graph representations. In Advances in Neural Information Processing Systems, volume 34, pp. 1713–1726, 2021.   
Radu Curticapean, Holger Dell, and Dániel Marx. Homomorphisms are a good basis for counting small subgraphs. In Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing, pp. 210–223, 2017.   
Maarten De Rijke. A note on graded modal logic. Studia Logica, 64(2):271–283, 2000.   
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in neural information processing systems, volume 29, 2016.   
Holger Dell, Martin Grohe, and Gaurav Rattan. Lovász meets weisfeiler and leman. In 45th International Colloquium on Automata, Languages, and Programming (ICALP 2018), volume 107, pp. 40. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2018.   
Reinhard Diestel. Graph Theory. Springer Publishing Company, Incorporated, 5th edition, 2017. ISBN 3662536218.   
Radoslav Dimitrov, Zeyang Zhao, Ralph Abboud, and İsmail İlkan Ceylan. Plane: Representation learning over planar graphs. arXiv preprint arXiv:2307.01180, 2023.   
Mohammed Haroon Dupty, Yanfei Dong, and Wee Sun Lee. Pf-gnn: Differentiable particle filtering based approximation of universal graph representations. In International Conference on Learning Representations, 2021.   
Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. arXiv preprint arXiv:2012.09699, 2020.   
Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.   
Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Graph neural networks with learnable structural and positional representations. In International Conference on Learning Representations, 2022.   
David Eppstein. Parallel recognition of series-parallel graphs. Information and Computation, 98(1):41–55, 1992.   
Jiarui Feng, Yixin Chen, Fuhai Li, Anindya Sarkar, and Muhan Zhang. How powerful are k-hop message passing graph neural networks. In Advances in Neural Information Processing Systems, volume 35, pp. 4776–4790, 2022.   
Jiarui Feng, Lecheng Kong, Hao Liu, Dacheng Tao, Fuhai Li, Muhan Zhang, and Yixin Chen. Towards arbitrarily expressive gnns in $O(n^{2})$ space by rethinking folklore weisfeiler-lehman. arXiv preprint arXiv:2306.03266, 2023.   
Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.

Fabrizio Frasca, Beatrice Bevilacqua, Michael M Bronstein, and Haggai Maron. Understanding and extending subgraph gnns by rethinking their symmetries. In Advances in Neural Information Processing Systems, 2022.   
Martin Fürer. Weisfeiler-lehman refinement requires at least a linear number of iterations. In International Colloquium on Automata, Languages, and Programming, pp. 322–333. Springer, 2001.   
Martin Fürer. On the combinatorial power of the weisfeiler-lehman algorithm. In International Conference on Algorithms and Complexity, pp. 260–271. Springer, 2017.   
Floris Geerts and Juan L Reutter. Expressiveness and approximation properties of graph neural networks. In International Conference on Learning Representations, 2022.   
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pp. 1263–1272. PMLR, 2017.   
Lorenzo Giusti, Teodora Reu, Francesco Ceccarelli, Cristian Bodnar, and Pietro Liò. Cin++: Enhancing topological message passing. arXiv preprint arXiv:2306.03561, 2023.   
Martin Grohe. Descriptive complexity, canonisation, and definable graph structure theory, volume 47. Cambridge University Press, 2017.   
William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Proceedings of the 31st International Conference on Neural Information Processing Systems, volume 30, pp. 1025–1035, 2017.   
Max Horn, Edward De Brouwer, Michael Moor, Yves Moreau, Bastian Rieck, and Karsten Borgwardt. Topological graph neural networks. In International Conference on Learning Representations, 2022.   
Yinan Huang, Xingang Peng, Jianzhu Ma, and Muhan Zhang. Boosting the cycle counting power of graph neural networks with i $^{2}$ -GNNs. In The Eleventh International Conference on Learning Representations, 2023.   
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pp. 448–456. PMLR, 2015.   
Nicolas Keriven and Gabriel Peyré. Universal invariant and equivariant graph neural networks. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 7092–7101, 2019.   
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.   
Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent Létourneau, and Prudencio Tossou. Rethinking graph transformers with spectral attention. In Advances in Neural Information Processing Systems, volume 34, 2021.   
Pan Li, Yanbang Wang, Hongwei Wang, and Jure Leskovec. Distance encoding: design provably more powerful neural networks for graph representation learning. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 4465–4478, 2020.   
Derek Lim, Joshua David Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, and Stefanie Jegelka. Sign and basis invariant networks for spectral graph representation learning. In The Eleventh International Conference on Learning Representations, 2023.   
László Lovász. Operations with structures. Acta Mathematica Hungarica, 18(3-4):321–328, 1967.   
László Lovász. Large networks and graph limits, volume 60. American Mathematical Soc., 2012.   
Shengjie Luo, Shanda Li, Shuxin Zheng, Tie-Yan Liu, Liwei Wang, and Di He. Your transformer may not be as powerful as you expect. arXiv preprint arXiv:2205.13401, 2022.

Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. In Advances in neural information processing systems, volume 32, pp. 2156–2167, 2019a.   
Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant graph networks. In International Conference on Learning Representations, 2019b.   
Haggai Maron, Ethan Fetaya, Nimrod Segol, and Yaron Lipman. On the universality of invariant networks. In International conference on machine learning, pp. 4363–4371. PMLR, 2019c.   
Brendan D McKay and Adolfo Piperno. Practical graph isomorphism, ii. Journal of symbolic computation, 60:94–112, 2014.   
Gaspard Michel, Giannis Nikolentzos, Johannes F Lutzeyer, and Michalis Vazirgiannis. Path neural networks: Expressive and accurate graph neural networks. In International Conference on Machine Learning, pp. 24737–24755. PMLR, 2023.   
Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodola, Jan Svoboda, and Michael M Bronstein. Geometric deep learning on graphs and manifolds using mixture model cnns. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5115–5124, 2017.   
Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pp. 4602–4609, 2019.   
Christopher Morris, Gaurav Rattan, and Petra Mutzel. Weisfeiler and leman go sparse: towards scalable higher-order graph embeddings. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 21824–21840, 2020.   
Christopher Morris, Gaurav Rattan, Sandra Kiefer, and Siamak Ravanbakhsh. Speqnets: Sparsity-aware permutation-equivariant graph networks. In International Conference on Machine Learning, pp. 16017–16042. PMLR, 2022.   
Christopher Morris, Yaron Lipman, Haggai Maron, Bastian Rieck, Nils M Kriege, Martin Grohe, Matthias Fey, and Karsten Borgwardt. Weisfeiler and leman go machine learning: The story so far. The Journal of Machine Learning Research, 2023.   
Ryan Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Relational pooling for graph representations. In International Conference on Machine Learning, pp. 4663–4673. PMLR, 2019.   
Daniel Neuen. Homomorphism-distinguishing closedness for graphs of bounded tree-width. arXiv preprint arXiv:2304.07011, 2023.   
Daniel Neuen and Pascal Schweitzer. An exponential lower bound for individualization-refinement algorithms for graph isomorphism. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing, pp. 138–150, 2018.   
Pál András Papp and Roger Wattenhofer. A theoretical comparison of graph neural network extensions. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pp. 17323–17345, 2022.   
Pál András Papp, Karolis Martinkus, Lukas Faber, and Roger Wattenhofer. Dropgnn: random dropouts increase the expressiveness of graph neural networks. In Advances in Neural Information Processing Systems, volume 34, pp. 21997–22009, 2021.   
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.   
Omri Puny, Derek Lim, Bobak Kiani, Haggai Maron, and Yaron Lipman. Equivariant polynomials for graph neural networks. In International Conference on Machine Learning, pp. 28191–28222. PMLR, 2023.

Chendi Qian, Gaurav Rattan, Floris Geerts, Mathias Niepert, and Christopher Morris. Ordered subgraph aggregation networks. In Advances in Neural Information Processing Systems, 2022.   
Ladislav Rampasek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. In Advances in Neural Information Processing Systems, 2022.   
Gaurav Rattan and Tim Seppelt. Weisfeiler-leman and graph spectra. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pp. 2268–2285. SIAM, 2023.   
Tim Seppelt. Logical equivalences, homomorphism indistinguishability, and forbidden minors. arXiv preprint arXiv:2302.11290, 2023.   
Behrooz Tahmasebi, Derek Lim, and Stefanie Jegelka. The power of recursion in graph neural networks for counting substructures. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206, pp. 11023–11042. PMLR, 2023.   
Erik Thiede, Wenda Zhou, and Risi Kondor. Autobahn: Automorphism-based graph neural nets. In Advances in Neural Information Processing Systems, volume 34, pp. 29922–29934, 2021.   
Petar Veličković. Message passing all the way up. arXiv preprint arXiv:2202.11097, 2022.   
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations, 2018.   
Clément Vignac, Andreas Loukas, and Pascal Frossard. Building powerful and equivariant graph neural networks with structural message-passing. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 14143–14155, 2020.   
Qing Wang, Dillon Ze Chen, Asiri Wijesinghe, Shouheng Li, and Muhammad Farhan. n-WL: A new hierarchy of expressivity for graph neural networks. In The Eleventh International Conference on Learning Representations, 2023.   
Boris Weisfeiler and Andrei Lehman. The reduction of a graph to canonical form and the algebra which appears therein. NTI, Series, 2(9):12–16, 1968.   
Asiri Wijesinghe and Qing Wang. A new perspective on "how graph neural networks go beyond weisfeiler-lehman?". In International Conference on Learning Representations, 2022a.   
Asiri Wijesinghe and Qing Wang. A new perspective on "how graph neural networks go beyond weisfeiler-lehman?". In International Conference on Learning Representations, 2022b.   
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019.   
Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? Advances in Neural Information Processing Systems, 34, 2021.   
Jiaxuan You, Jonathan M Gomes-Selman, Rex Ying, and Jure Leskovec. Identity-aware graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 10737–10745, 2021.   
Bohang Zhang, Guhao Feng, Yiheng Du, Di He, and Liwei Wang. A complete expressiveness hierarchy for subgraph GNNs via subgraph weisfeiler-lehman tests. In International Conference on Machine Learning, volume 202, pp. 41019–41077. PMLR, 2023a.   
Bohang Zhang, Shengjie Luo, Di He, and Liwei Wang. Rethinking the expressive power of gnns via graph biconnectivity. In International Conference on Learning Representations, 2023b.   
Muhan Zhang and Pan Li. Nested graph neural networks. In Advances in Neural Information Processing Systems, volume 34, pp. 15734–15747, 2021.

Lingxiao Zhao, Wei Jin, Leman Akoglu, and Neil Shah. From stars to subgraphs: Uplifting any gnn with local structure awareness. In International Conference on Learning Representations, 2022a.   
Lingxiao Zhao, Neil Shah, and Leman Akoglu. A practical, progressively-expressive GNN. In Advances in Neural Information Processing Systems, 2022b.   
Cai Zhou, Xiyuan Wang, and Muhan Zhang. From relational pooling to subgraph gnns: A universal framework for more expressive graph neural networks. arXiv preprint arXiv:2305.04963, 2023a.   
Junru Zhou, Jiarui Feng, Xiyuan Wang, and Muhan Zhang. Distance-restricted folklore weisfeiler-leman gnns with provable cycle counting power. arXiv preprint arXiv:2309.04941, 2023b.

# Appendix

# Table of Contents

A More Related Work 18

A.1 Expressive Graph Neural Networks 18   
A.2 Broader impacts and additional discussions 20

B Limitations and Open Directions 21

C Proof of Theorem 3.4 21

C.1 Preliminary 22   
C.2 Part 1: tree decomposition ..... 25   
C.3 Part 2: nested ear decomposition 39   
C.4 Part 3: pebble game 43

D Node/edge-level Expressivity 50

D.1 Related to tree decomposition and ear decomposition ..... 51   
D.2 Counterexamples 52

E Higher-order GNNs 54

E.1 Definition of higher-order GNNs 54   
E.2 Higher-order strong NED 54   
E.3 Proofs in Section 3.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55   
E.4 Expressivity gap between higher-order GNNs . . . . . . . . . . . . . . . . . . . . 56

F Additional Discussions 57

F.1 Regarding the definition of homomorphism expressivity 57   
F.2 Regarding the definition of NED 57

G Homomorphism and Subgraph Counting Power 57

G.1 Proof of Theorem 4.5 57   
G.2 Graph statistics and examples 59

H Polynomial Expressivity 69

I Experimental Details 70

I.1 Datasets 70   
I.2 Model details 70   
I.3 Training details 72   
I.4 Performance of baseline models in literature 72

# A MORE RELATED WORK

# A.1 EXPRESSIVE GRAPH NEURAL NETWORKS

Since Morris et al. (2019); Xu et al. (2019) discovered the limited expressive power of MPNNs in distinguishing non-isomorphic graphs, a large amount of work has been devoted to developing GNNs with better expressiveness. Below, we briefly review representative approaches in this area. For a comprehensive survey on expressive GNNs, we refer readers to Morris et al. (2023).

Higher-order GNNs. Inspired by the relation between MPNNs and the 1-WL test, a natural approach to designing provably more expressive GNNs is to mimic the higher-order WL tests. This gives rise to two fundamental types of higher-order GNNs. One type of GNNs mimics the k-WL test (Grohe, 2017), and representative architectures include k-GNN (Morris et al., 2019) and k-IGN (Maron et al., 2019b;c); the other type of GNNs mimics the k-FWL (Folklore WL) test (Cai et al., 1992) and is referred to as the the k-FGNN (Maron et al., 2019a). Azizian & Lelarge (2021); Geerts & Reutter (2022) proved that each of these architectures is exactly as expressive as the corresponding higher-order WL/FWL test. Therefore, the expressiveness grows strictly as the order k increases; when k approaches infinity, they can universally approximate any continuous graph functions (Chen et al., 2019b; Keriven & Peyré, 2019). However, due to the inherent computation/memory complexity, these architectures are generally not practical in real-world applications.

Local GNNs. To improve computational efficiency, a subsequent line of work seeks to develop more scalable and practical GNN architectures by taking into account the local/sparse nature of graphs. Locality/sparsity is also an important inductive bias for graphs but is not well-exploited in higher-order GNNs, since their layer aggregation is inherently global and the graph adjacency information is only encoded in initial node features. To address these shortcomings, Morris et al. (2020) proposed the Local k-GNN (and several variants) as a replacement of k-GNN, which directly incorporates graph adjacency into network layers and only aggregates neighboring information instead of the global one. The authors further proved that Local k-GNN is strictly more expressive than k-GNN. Building upon Local k-GNN, Morris et al. (2022) proposed the $(k, s)$ -SpeqNet that further reduces the computational cost by considering a subset of k-tuples whose vertices can be grouped into no more than s connected components. A similar idea appeared in Zhao et al. (2022b), in which the authors proposed the $(k, s)$ -SetGNN by considering k-sets instead of k-tuples. Besides Local k-GNN, recent architectures proposed in Frasca et al. (2022) and Zhang et al. (2023a) can be analogously understood as Local k-IGN and Local k-FGNN, respectively. Very recently, Feng et al. (2023); Zhou et al. (2023b) generalized the Local k-FGNN to a broad class of Folklore-type GNNs and achieved good performance on several benchmark datasets.

Subgraph GNNs. Graphs that are indistinguishable by WL tests typically possess a high degree of symmetry. In light of this observation, Subgraph GNNs have recently emerged as a compelling approach to designing expressive GNNs. The basic idea is to break symmetry by transforming the original graph into a collection of slightly modified subgraphs and feeding these subgraphs into a GNN model. The earliest forms of Subgraph GNNs may track back to Cotta et al. (2021); Papp et al. (2021) (albeit with a different motivation), where the authors proposed to feed node-deleted subgraphs into an MPNN. Papp & Wattenhofer (2022) later argued to use node marking instead of node deletion for better expressive power, resulting in the standard Subgraph GNN studied in this paper. Zhang & Li (2021); You et al. (2021) proposed the Nested GNN and Identity-aware GNN, both of which can be treated as variants of Subgraph GNNs that use ego networks as subgraphs. In particular, the heterogeneous message passing proposed in You et al. (2021) can also be seen as a form of node marking. We note that the model proposed in Vignac et al. (2020) can also be interpreted as a Subgraph GNN. Qian et al. (2022) proposed the higher-order Subgraph GNN by marking k nodes per subgraph, resulting in $n^{k}$ different subgraphs when the original graph has n vertices. We call this architecture Subgraph k-GNN in this paper. The authors proved that Subgraph k-GNN is strictly bounded by $(k+1)$ -FWL and is incomparable to k-FWL when k > 1. Zhou et al. (2023a) further generalized Subgraph k-GNN to $(l,k)$ -GNN by using l-GNN instead of MPNN to process each subgraph. It was proved that $(l,k)$ -GNN is bounded by $(k+l)$ -GNN for $l \geq 2$ .

Recently, Subgraph GNNs have been greatly extended to further enable interactions between subgraphs. This is achieved by designing cross-subgraph aggregation layers (rather than feeding each subgraph independently into a GNN). Bevilacqua et al. (2022) developed the Equivariant Subgraph Aggregation Network that introduces a global aggregation between subgraphs. A similar design is

proposed in the concurrent work of Zhao et al. (2022a). These architectures were later proved to strictly improve the expressivity of the original Subgraph GNNs (Zhang et al., 2023b;a). Frasca et al. (2022) built a general design space of Subgraph GNNs that unifies prior work and showed that all models in this space are bounded by a variant of 2-IGN (dubbed the Local 2-IGN in this paper), which is then bounded by 2-FWL. Zhang et al. (2023a) later proved that Local 2-IGN is as expressive as Local 2-GNN and strictly less expressive than 2-FWL (2-FGNN). In this paper, we still use the term “Subgraph GNN” to refer to the original architecture in the previous paragraph, while using “Local 2-GNN” to refer to the general architecture in Frasca et al. (2022); Zhang et al. (2023a).

Substructure-based GNNs. Another line of work sought to develop expressive GNNs from practical considerations. In particular, Chen et al. (2020) pointed out that the ability of GNNs to detect/count graph substructures like path, cycle, and clique is crucial in numerous applications. Yet, MPNNs cannot subgraph-count any cycles/cliques. While higher-order WL tests can be more powerful in counting cycles (Fürer, 2017; Arvind et al., 2020), they suffer from substantial computational cost. As such, several works proposed to directly incorporate substructure counting into the node features as a preprocessing step to boost the expressiveness of MPNNs (Bouritsas et al., 2022; Barceló et al., 2021). Going beyond node features, Bodnar et al. (2021b;a); Giusti et al. (2023) further proposed a message-passing framework that enables interaction between nodes, edges, and higher-order substructures like cycles and cliques. We note that the Autobahn, TOGL, and Cy2C-GNN proposed in Thiede et al. (2021); Horn et al. (2022); Choi et al. (2022) can also be viewed as Substructure-based GNNs. However, most of the above approaches consider a fixed, predefined set of substructures rather than designing generic architectures that can learn substructures in an end-to-end fashion. Recently, Tahmasebi et al. (2023) proposed the RNP-GNN, an architecture that can count any substructure by recursively splitting a graph into a collection of vertex-marked subgraphs. We note that this design shares interesting similarities to higher-order subgraph GNNs and also the SpeqNet (Morris et al., 2022). Huang et al. (2023) proposed a generic model called I²-GNN based on a variant of Subgraph 2-GNN, which can count 6-cycle at node-level. In this paper, we show the Local 2-GNN can already count all cycles/paths within 7 nodes even at edge-level while being more efficient than I²-GNN (when using a similar ego network design). Finally, we remark that the polynomial expressivity proposed in Puny et al. (2023) can also be seen as a generalization of substructure counting, which further takes into account the real-valued node/edge features.

Distance-based GNNs. Besides structural information, distance serves as another fundamental attribute of a graph, which, again, is not captured by MPNNs and the 1-WL test. Li et al. (2020) first proposed to improve the expressive power of GNNs by augmenting node features with Distance Encoding (DE). Related to DE, another approach to injecting distance information is the k-hop MPNN, which aggregates k-hop neighbors in a message-passing layer (Feng et al., 2022; Abboud et al., 2022; Wang et al., 2023). Feng et al. (2022); Zhang et al. (2023a) proved that the expressive power of k-hop MPNN is strictly bounded by 2-FWL. Distance can also be naturally incorporated in Graph Transformers through relative positional encoding, yielding the Graphormer architecture that has achieved remarkable performance across various benchmarks (Ying et al., 2021). Recently, Zhang et al. (2023b) built an interesting connection between distance and biconnectivity properties, showing that distance-enhanced GNNs can detect cut vertices and cut edges of a graph. This provides insights into the practical superiority of these models as biconnectivity is closely linked to real applications in chemistry and social network analysis. Zhang et al. (2023a) proved that Local 2-GNN can provably encode the distance (and thus biconnectivity) of a graph.

Spectral-based GNNs. Graph spectra are also a class of fundamental properties and have long been used to design GNN models (Bruna et al., 2014; Defferrard et al., 2016). Balcilar et al. (2021b;a) showed that designing GNNs in the spectral domain can easily break the 1-WL expressivity. For Graph Transformers, Kreuzer et al. (2021); Dwivedi & Bresson (2020); Dwivedi et al. (2022) proposed to incorporate the spectra of the graph Laplacian matrix to boost the expressive power beyond the 1-WL test. Lim et al. (2023) further designed a principled equivariant architecture that takes the Laplacian eigenvalues and eigenvectors as inputs, which generalizes prior work.

Other approaches. Murphy et al. (2019); Chen et al. (2020) proposed Relational Pooling as a general approach to designing expressive GNN architectures, whose basic idea is to implement a permutation-invariant GNN by symmetrizing permutation-sensitive base models. Wijesinghe & Wang (2022b) proposed the GraphSNN, which improves the expressive power of MPNNs by using more distinguishing edge features. Specifically, each edge feature encodes the structure of the overlap subgraph of two 1-hop ego networks centered on the two endpoints of the edge. Very recently,

Dimitrov et al. (2023) designed a GNN model that can distinguish all planar graphs, thus achieving a strong expressivity in chemical applications since molecular graphs are often planar.

# A.2 BROADER IMPACTS AND ADDITIONAL DISCUSSIONS

Broader impact in graph theory. Due to the fundamental nature of GNN architectures studied in this paper, our theoretical results may potentially have broader impacts on the graph theory community. Specifically, we study the color refinement (CR) algorithms corresponding to four types of (higher-order) GNNs: Subgraph $(k-1)$ -GNN, Local k-GNN, Local k-FGNN, and k-FGNN. All algorithms can be seen as natural extensions of the 1-WL test since they all reduce to 1-WL when k=1. In the graph theory community, Subgraph GNN has another name known as the vertex-individualized CR algorithm, which appears widely in literature (Babai, 2016; Rattan & Seppelt, 2023; Neuen & Schweitzer, 2018) and has become part of the core algorithm for fast graph isomorphism testing software (e.g., McKay & Piperno, 2014). On the other hand, Local k-GNN and Local k-FGNN are surprisingly related to the guarded logic (Barceló et al., 2020; De Rijke, 2000; Baader, 2003), since the aggregations are purely local (guarded by the edge). From this perspective, these CR algorithms can be seen as natural extensions of guarded logic in higher-order scenarios.

Besides these CR algorithms, our new extensions of NED may also have implications in graph theory. In particular, the strong NED (as well as its higher-order version) is elegant and may serve as a descriptive tool to characterize certain graph families. In addition, we establish intrinsic connections between NED and tree decomposition, which may have value in understanding other graph topics related to tree decomposition.

Finally, to our knowledge, the node/edge-level homomorphism and the corresponding subgraph counting abilities of different CR algorithms do not seem to have been systematically investigated before. Whereas in this paper, all graph/node/edge-level expressivity is studied in a unified manner. To achieve this, we introduce additional proof techniques which we believe may facilitate future study in related areas. For example, the original technique for constructing counterexample graphs satisfying Definition 3.1(b) does not apply to node/edge-level settings, since they are no longer counterexample graphs satisfying Definition 3.6(b) (no matter which vertices $u \in V_G, v \in V_H$ are marked). To address the problem, we propose clique-augmented Fürer graphs, a novel class of counterexample graphs that extend several prior works (e.g., Cai et al., 1992; Fürer, 2001), and conduct a fine-grained analysis of their automorphism property (see Appendix D.2). We believe this new technique can be used to generalize other results from graph-level to node/edge-level settings.

Discussions with Barceló et al. (2021). In the GNN community, Barceló et al. (2021) first proposed to incorporate the homomorphism count of predefined substructures into node features as an approach to enhancing the expressivity of MPNNs. They systematically investigated the questions of what substructures are useful and how the homomorphism information of these substructures can boost the model expressivity to even break out k-FWL. Yet, they only gave a partial (incomplete) characterization of the substructures that can be counted by the specific F-MPNN architecture and did not answer what substructures cannot be encoded, whereas our paper fully addresses both questions for a variety of popular GNN models. Note that these aspects are crucial to ensure that homomorphism expressivity is well-defined. As such, our paper first identifies that homomorphism expressivity is a complete, quantitative expressivity measure to compare different GNN models.

Discussions with the concurrent work of Neuen (2023). After the initial submission, we became aware of a concurrent work (Neuen, 2023), which proved that k-FWL cannot count any graph with treewidth larger than k under homomorphism. In our context, this result exactly shows that Definition 3.1(b) is satisfied, and thus the homomorphism expressivity of k-FWL is well-defined. Notably, their construction of counterexample graphs is also based on Furer graphs. Nevertheless, the proof technique between the two works is quite different: the proof in Neuen (2023) is built upon a key concept called oddomorphism, while our proof is based on the relation between tree decomposition and the simplified pebble game developed in Zhang et al. (2023a). It is essential to underscore that our results and proof technique are more general and go beyond the standard k-FWL, in that (i) it applies to a broad range of color refinement algorithms related to practical GNN architectures and (ii) it further extends to node/edge-level homomorphism expressivity. Our theoretical results thus strictly incorporate the results in Neuen (2023). The approach in Neuen (2023) (based on oddomorphism) cannot be easily generalized to these settings.

# B LIMITATIONS AND OPEN DIRECTIONS

There are still several open questions that are not fully explored in this paper. We list them below as promising directions for future study.

Existence of homomorphism expressivity for refinement-based GNN architectures. In this paper, we prove that homomorphism expressivity exists for a wide range of architectures defined in Section 2. On the other hand, we also note that it may not be well-defined for certain pathological GNNs, as illustrated in Appendix F.1. Given this observation, a fundamental question is: what conditions can guarantee that the homomorphism expressivity of a GNN exists? Here, we hypothesize that a very mild condition can suffice. Specifically, we conjecture that as long as a GNN architecture is defined following a general form of color refinement procedure that outputs stable color mappings, its homomorphism expressivity always exists. We leave this conjecture as an important open problem for future study.

Regarding higher-order Local FGNN. This paper characterizes the homomorphism expressivity for three classes of higher-order GNNs: Subgraph k-GNN, Local k-GNN, and k-FGNN. In particular, we introduce the k-order ear and k-order strong NED as a way to describe the homomorphism expressivity of Local k-GNN. However, it remains unclear how to give a simple description of the homomorphism expressivity for Local k-FGNN that can generalize the concept of almost-strong NED for k = 2. As such, our current expressiveness hierarchy (Figure 3) does not support Local k-FGNN yet. We leave this as an open problem and make the following conjecture below. We note that a similar open question has been informally raised in Zhang et al. (2023a).

Conjecture B.1. For all $k \geq 2$ , Local k-FGNN is strictly more expressive than Local k-GNN and strictly less expressive than k-FGNN.

Expressivity gap between Local 2-(F)GNN and 2-FGNN in practical aspects. We have proved that 2-FGNN is strictly more expressive than Local 2-(F)GNN. However, from a practical perspective, we surprisingly find that the subgraph counting ability of Local 2-(F)GNN matches that of 2-FGNN for all structures within a moderate size (see Table 5), although the former is much more efficient. This leads to the intriguing question of what fundamental gaps exist between the two models in practical aspects, or is the efficiency gain free?

Other architectures. In this paper, we comprehensively study a variety of popular GNN architectures ranging from Subgraph GNNs and Local GNNs to higher-order GNNs, and further link these architectures to Substructure-based GNNs (see Appendix A.1). Yet, we still do not cover all popular GNN architectures, such as the GSWL-based Subgraph GNN (Zhang et al., 2023a; Bevilacqua et al., 2022), SpeqNet (Morris et al., 2022), and I²-GNN (Huang et al., 2023). Moreover, the classes of Distance-based GNNs and Spectral-based GNNs (Appendix A.1) are also widely used in practice, which deserve future study. We would like to raise the question of characterizing the homomorphism expressivity of Distance-based GNNs and Spectral-based GNNs as an important open question. In this way, one can gain deep insights into these models' true expressivity and enable quantitative comparisons between all mainstream architectures. Moreover, it will become clear to what extent other GNN models can encode distance and spectral information about a graph.

# C PROOF OF THEOREM 3.4

This section gives the proof of the main theorem. For ease of reading, we first restate Theorem 3.4:

Theorem 3.4. For all GNN models M defined in Section 2, the graph family $F^{M}$ satisfying Definition 3.1 exists (and is unique). Moreover, each $F^{M}$ can be separately described below:

- Subgraph GNN: $\mathcal{F}^{\text{Sub}} = \{F : F \text{ has an endpoint-shared NED}\}$ ;   
- Local 2-GNN: $\mathcal{F}^{\mathrm{L}} = \{F : F \text{ has a strong NED}\}$ ;   
- Local 2-FGNN: $\mathcal{F}^{\text{LF}} = \{F : F \text{ has an almost-strong NED}\}$ ;   
- 2-FGNN: $\mathcal{F}^{\mathsf{F}} = \{F : F \text{ has a NED}\}$ .

For MPNN, since it is a special case of Subgraph k-GNN, the proof can be found in Appendix E.3.

# C.1 PRELIMINARY

Additional notations and concepts. Besides the notations defined in Section 2, we further define the following notations. We use the symbol G to denote the set of all finite, simple, undirected, labeled graphs. Let $G = (V_{G}, E_{G}, \ell_{G})$ be a graph in G. When the label is the same for all vertices, we can omit the term $\ell_{G}$ and write $G = (V_{G}, E_{G})$ . Given vertex $u \in V_{G}$ , denote the degree of u as $\deg_{G}(u) = |N_{G}(u)|$ , and denote the closed neighborhood of u as $N_{G}[u] = N_{G}(u) \cup \{u\}$ . The shortest path distance between vertices u and v is denoted by $\text{dis}_{G}(u, v)$ .

Given a vertex set $S \subset V_{G}$ , the induced subgraph of G over S, denoted as G[S], is the subgraph of G with vertex set S and edge set $\{\{u,v\} \in E_{G}: u, v \in S\}$ . Similarly, without abuse of notation, given an edge set $R \subset E_{G}$ , the induced subgraph of G over R, denoted as G[R], is the subgraph of G with vertex set $\bigcup_{\{u,v\} \in R}\{u,v\}$ and edge set R. Given a vertex tuple $\boldsymbol{u} = (u_{1}, \cdots, u_{k})$ , denote by $G^{u} = G^{u_{1}, \cdots, u_{k}}$ the rooted graph obtained from G by marking vertices $u_{1}, \cdots, u_{k}$ . The atomic type of G over u, denoted by $\operatorname{atp}_{G}(\boldsymbol{u})$ , is a $k \times k$ matrix where the element at position $(i,j)$ is the tuple $(\mathbb{I}[u_{i} = u_{j}], \mathbb{I}[\{u_{i}, u_{j}\} \in E_{G}])$ . Given two graphs G, H, the graph union $G \cup H$ is defined to be the graph $(V_{G} \cup V_{H}, E_{G} \cup E_{H}, \ell_{G \cup H})$ , where $\ell_{G \cup H}(u) := \ell_{G}(u)$ for all $u \in V_{G}$ and $\ell_{G \cup H}(u) := \ell_{H}(u)$ for all $u \in V_{H}$ . It is well-defined iff $\ell_{G}(u) = \ell_{H}(u)$ for all $u \in V_{G} \cap V_{H}$ . Finally, we use the notation $G \simeq H$ to denote that G and H are isomorphic graphs.

A graph $T = (V_{T}, E_{T}, \ell_{T})$ is called a tree if it does not contain cycles. Let $T^{r}$ be a rooted tree where r is the root vertex. For each vertex $t \in V_{T}$ , define its depth $\mathsf{dep}_{T^{r}}(t) := \mathsf{dis}_{T}(t, r)$ to be the distance to the root, and denote by $\mathsf{Desc}_{T^{r}}(t)$ the set of descendants of t, namely, $s \in \mathsf{Desc}_{T^{r}}(t)$ iff $\mathsf{dep}_{T^{r}}(s) = \mathsf{dep}_{T^{r}}(t) + \mathsf{dis}_{T}(t, s)$ . For each $t \in V_{T} \setminus \{r\}$ , denote by $\mathsf{pa}_{T^{r}}(t)$ the parent vertex of t, i.e., the unique vertex $s \in N_{T}(t)$ satisfying $\mathsf{dep}_{T^{r}}(t) = \mathsf{dep}_{T^{r}}(s) + 1$ . Define the subtree of $T^{r}$ rooted at node t by $T^{r}[t]$ , which is exactly the induced subgraph $T[\mathsf{Desc}_{T^{r}}(t)]^{t}$ with root t.

Tree decomposition. Our proof is based on a central concept in graph theory, called tree decomposition. It can be formally defined below:

Definition C.1 (Tree decomposition). Given a graph $G = (V_G, E_G, \ell_G)$ , its tree decomposition is a tree $T = (V_T, E_T, \beta_T)$ , where the label function $\beta_T : V_T \to 2^{V_G}$ satisfies the following conditions:

a) Each tree node $t \in V_T$ is associated to a non-empty subset of vertices $\beta_T(t) \subset V_G$ in $G$ , called a bag. We say tree node $t$ contains vertex $u$ if $u \in \beta_T(t)$ ;   
b) For each edge $\{u,v\} \in V_G$ , there exists at least one tree node $t \in V_T$ that contains the edge, i.e., $\{u,v\} \subset \beta_T(t)$ ;   
c) For each vertex $u \in V_G$ , all tree nodes $t$ containing $u$ form a (non-empty) connected subtree. Formally, denoting $B_T(u) = \{t \in V_T : u \in \beta_T(t)\}$ , then $T[B_T(u)]$ is connected.

If $T$ is a tree decomposition of $G$ , we call the pair $(G, T)$ a tree-decomposed graph.

We remark that given a graph G, there are multiple ways to decompose it and thus its tree decomposition is not unique. Several examples of tree decomposition is given in Figure 5.

Definition C.2 (Treewidth). The width of a tree decomposition is defined as one less than the maximum bag size, i.e., $\max_{t\in T}|\beta_{T}(t)|-1$ . The treewidth of a graph G, denoted as $\operatorname{tw}(G)$ , is the minimum positive integer k such that there exists a tree decomposition of width k.

Some important facts about treewidth are listed below:

Fact C.3. For any graph G, the following hold:

- The treewidth of $G$ is at most $|V_G| - 1$ , i.e., a trivial tree decomposition that only has one node $t$ and $\beta_T(t) = V_G$ .   
- $\operatorname{tw}(G) = |V_G| - 1$ iff $G$ is a clique.   
- $\operatorname{tw}(G) = 1$ iff $G$ is a forest.

The above definition of tree decomposition is quite flexible without constraints on the structure of the tree or the size of each bag. Below, we define several restricted variants of tree decomposition, which (we will later see) are closely related to the GNN architectures studied in this paper. To begin with, we first define a general concept that slight modifies the original definition (Definition C.1) such that the tree becomes rooted and each bag is a multiset of vertices rather than a set.

![](images/17c1c1a6c807054a7701365e98395882655220f0309a3b97614c8b51502973b5.jpg)  
Figure 5: Illustration of tree decomposition.

Definition C.4 (Canonical tree decomposition). Given a graph $G = (V_{G}, E_{G}, \ell_{G})$ , a canonical tree decomposition of width k is a rooted tree $T^{r} = (V_{T}, E_{T}, \beta_{T})$ satisfying the following conditions:

a) The depth of T is even, i.e. $\max_{t\in V_{T}}\mathsf{dep}_{T^{r}}(t)$ is even;   
b) Each tree node $t \in V_T$ is associated to a multiset of vertices $\beta_T(t) \subset V_G$ , called a bag. Moreover, $|\beta_T(t)| = k$ if $\mathsf{dep}_{T^r}(t)$ is even and $|\beta_T(t)| = k + 1$ if $\mathsf{dep}_{T^r}(t)$ is odd;   
c) For all tree edges $\{s,t\}\in E_{T}$ where $\operatorname{dep}_{T^{r}}(s)$ is even and $\operatorname{dep}_{T^{r}}(t)$ is odd, $\beta_{T}(s)\subset\beta_{T}(t)$ (where “⊂” denotes the multiset inclusion relation);   
d) The conditions (b) and (c) in Definition C.1 are satisfied.

As examples, one can check that the tree decomposition of all graphs in Figure 5(b) is canonical, but the tree decomposition in Figure 5(a) is not. An important observation about canonical tree decomposition is shown below:

Proposition C.5. Let $(F, T^{r})$ be any tree-decomposed graph where $T^{r}$ is a canonical tree decomposition of F. For any vertices $u, v \in V_{F}$ , either of the following holds:

- $u$ and $v$ are in the same bag of $T^r$ , i.e., there is a node $t \in V_T$ such that $\{\{u, v\}\} \subset \beta_T(t)$ ;   
- The induced subgraph $T[B_T(u) \cup B_T(v)]$ is disconnected.

Proof. Assume that u and v are not in the same bag, i.e., $B_{T}(u) \cap B_{T}(v) = \emptyset$ . Pick $s \in B_{T}(u)$ and $t \in B_{T}(v)$ such that $\mathsf{dep}_{Tr}(s)$ and $\mathsf{dep}_{Tr}(t)$ are minimal, respectively. Without loss of generality, assume that $\mathsf{dep}_{Tr}(s) \leq \mathsf{dep}_{Tr}(t)$ . Then, $t \neq r$ is not the root node and thus we can pick its parent $\mathsf{pa}_{Tr}(t)$ . It follows that $\mathsf{dep}_{Tr}(t)$ is odd and $\beta_{T}(\mathsf{pa}_{Tr}(t)) \subset \beta_{T}(t)$ by definition of canonical tree decomposition. Therefore, $u \notin \beta_{T}(\mathsf{pa}_{Tr}(t))$ . Moreover, by the assumption that $\mathsf{dep}_{Tr}(s) \leq \mathsf{dep}_{Tr}(t)$ , any node in $B_{T}(u)$ is not a descendant of t. We thus conclude that there does not exist a tree edge such that the two endpoints contain u and v, respectively. □

Now we are ready to define several restricted variants of canonical tree decomposition:

Definition C.6. Define four families of tree-decomposed graphs $S^{Sub}$ , $S^{L}$ , $S^{LF}$ , and $S^{F}$ as follows:

a) $(F, T^{r}) \in \mathcal{S}^{\mathsf{F}}$ iff $(F, T^{r})$ satisfies Definition C.4 with width k = 2;   
b) $(F, T^{r}) \in \mathcal{S}^{\mathrm{LF}}$ iff $(F, T^{r})$ satisfies Definition C.4 with width k = 2, and for any tree node t of odd depth, it has only one child if $w \notin \{v : v \in N_{G}[u], u \in \beta_{T}(s)\}$ where s is the parent node of t and w is the unique vertex in $\beta_{T}(t) \backslash \beta_{T}(s)$ ;   
c) $(F,T^r)\in \mathcal{S}^{\mathrm{L}}$ iff $(F,T^r)$ satisfies Definition C.4 with width $k = 2$ , and any tree node $t$ of odd depth has only one child;   
d) $(F,T^r)\in S^{\mathrm{Sub}}$ iff $(F,T^r)$ satisfies Definition C.4 with width $k = 2$ , and there exists a vertex $u\in V_G$ such that $u\in \beta_T(t)$ for all $t\in V_T$ .

Examples of tree-decomposed graphs in the four families are illustrated in Figure 5(b).

Before closing this subsection, we define several notations for tree-decomposed graphs:

Definition C.7. Given canonical tree-decomposed graph $(G, T^{r})$ and node $t \in V_{T}$ , denote by $G[T^{r}[t]]$ the subgraph of G induced by the vertex set $\{u : u \in \beta_{T^{r}}(s), s \in \text{Desc}_{T^{r}}(t)\}$ .

Definition C.8. Given two canonical tree-decomposed graphs $(G,T^{r})$ and $(\tilde{G},\tilde{T}^{s})$ , a pair of mappings $(\rho,\tau)$ is called an isomorphism from $(G,T^{r})$ to $(\tilde{G},\tilde{T}^{s})$ , denoted by $(G,T^{r})\simeq(\tilde{G},\tilde{T}^{s})$ , if the following hold:

a) $\rho$ is an isomorphism from $G$ to $\tilde{G}$ ;   
b) $\tau$ is an isomorphism from $T^r$ to $\tilde{T}^s$ (ignoring labels $\beta$ );   
c) For any $t \in T^r$ , $\rho(\beta_T(t)) = \beta_{\tilde{T}}(\tau(t))$ .

Equivalent formulation of GNN architectures. Below, we give equivalent definitions for several GNN architectures presented in Section 2, which will be used in subsequent analysis. Let $G$ be a graph and $u, v \in V_G$ . For all models $M$ including Subgraph GNN, Local 2-GNN, Local 2-FGNN, and 2-FGNN, we define the initial color $\tilde{\chi}_G^{M,(0)}(u,v)$ to be the isomorphism type of vertex pair $(u,v)$ (i.e., $\mathrm{atp}_G(u,v)$ plus labels of each vertex). Note that this matches the original definition except for Subgraph GNN. Then in each iteration $t$ , the color is updated according to the following formula. Here, for clarity, we denote $\tilde{\chi}_G^{M,(t)}(u,S) = \{\{\tilde{\chi}_G^{M,(t)}(u,v):v \in S\}\}$ and $\tilde{\chi}_G^{M,(t)}(S,v) = \{\{\tilde{\chi}_G^{M,(t)}(u,v):u \in S\}\}$ for any model $M$ and set $S$ .

\- Subgraph GNN:

$$
\tilde {\chi} _ {G} ^ {\text { Sub }, (t + 1)} (u, v) = \operatorname{hash} \left(\tilde {\chi} _ {G} ^ {\text { Sub }, (t)} (u, v), \tilde {\chi} _ {G} ^ {\text { Sub }, (t)} (u, N _ {G} (v)), \tilde {\chi} _ {G} ^ {\text { Sub }, (t)} (u, V _ {G})\right). \tag {7}
$$

\- Local 2-GNN:

$$
\begin{array}{l} \tilde {\chi} _ {G} ^ {\mathsf {L}, (t + 1)} (u, v) = \operatorname{hash} \left(\tilde {\chi} _ {G} ^ {\mathsf {L}, (t)} (u, v), \tilde {\chi} _ {G} ^ {\mathsf {L}, (t)} (u, N _ {G} (v)), \tilde {\chi} _ {G} ^ {\mathsf {L}, (t)} \left(N _ {G} (u), v\right), \right. \tag {8} \\ \left. \tilde {\chi} _ {G} ^ {\mathsf {L}, (t)} (u, V _ {G}), \tilde {\chi} _ {G} ^ {\mathsf {L}, (t)} (V _ {G}, v)\right). \\ \end{array}
$$

\- Local 2-FGNN:

$$
\begin{array}{l} \tilde {\chi} _ {G} ^ {\mathsf {L F}, (t + 1)} (u, v) = \mathsf {h a s h} \left(\tilde {\chi} _ {G} ^ {\mathsf {L F}, (t)} (u, v), \left\{\left(\tilde {\chi} _ {G} ^ {\mathsf {L F}, (t)} (w, v), \tilde {\chi} _ {G} ^ {\mathsf {L F}, (t)} (u, w)\right): w \in N _ {G} [ u ] \cup N _ {G} [ v ] \right\} \right\}, \\ \left. \tilde {\chi} _ {G} ^ {\mathrm{LF}, (t)} \left(u, V _ {G}\right), \tilde {\chi} _ {G} ^ {\mathrm{LF}, (t)} \left(V _ {G}, v\right)\right). \tag {9} \\ \end{array}
$$

\- 2-FGNN:

$$
\begin{array}{l} \tilde {\chi} _ {G} ^ {\mathsf {F}, (t + 1)} (u, v) = \operatorname{hash} \left(\tilde {\chi} _ {G} ^ {\mathsf {F}, (t)} (u, v), \left\{\left(\tilde {\chi} _ {G} ^ {\mathsf {F}, (t)} (w, v), \tilde {\chi} _ {G} ^ {\mathsf {F}, (t)} (u, w)\right): w \in V _ {G} \right\} \right\rbrace , \tag {10} \\ \left. \tilde {\chi} _ {G} ^ {\mathsf {F}, (t)} (u, V _ {G}), \tilde {\chi} _ {G} ^ {\mathsf {F}, (t)} (V _ {G}, v)\right). \\ \end{array}
$$

It can be seen that we additionally add global aggregations for these architectures. Moreover, in Local 2-FGNN we replace the neighbors by closed neighbors $N_G[u] \cup N_G[v]$ . The stable color of $(u,v)$ for different models is denoted by $\tilde{\chi}_G^{\mathrm{Sub}}(u,v), \tilde{\chi}_G^{\mathrm{L}}(u,v), \tilde{\chi}_G^{\mathrm{LF}}(u,v), \tilde{\chi}_G^{\mathrm{F}}(u,v)$ , respectively. We have the following result:

Proposition C.9. Let $M \in \{\mathsf{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. For any graphs $G, H$ , $\tilde{\chi}_G^M(G) = \tilde{\chi}_H^M(H)$ iff $\chi_G^M(G) = \chi_H^M(H)$ . Furthermore, if $G$ and $H$ are connected, then for any vertices $u, v \in V_G$ and $x, y \in V_H$ , $\tilde{\chi}_G^M(u, v) = \tilde{\chi}_H^M(x, y)$ iff $\chi_G^M(u, v) = \chi_H^M(x, y)$ . In other words, the color mapping $\tilde{\chi}^M$ is as fine as the original one $\chi^M$ .

Proof. The proof simply follows from Zhang et al. (2023a, Proposition 4.2 and Theorem 4.4), because (i) node marking in the initial color of Subgraph GNN is as expressive as using the isomorphism type, (ii) the global aggregation does not improve the expressivity when the corresponding local aggregation is presented, (iii) the single-point aggregation does not improve the expressivity in Local 2-FGNN.

# C.2 PART 1: TREE DECOMPOSITION

We first define the unfolding tree of different CR algorithms, which is a standard tool in analyzing GNN expressivity.

Definition C.10 (Unfolding tree of Subgraph GNN). Given a graph $G$ , vertices $u, v \in V_G$ , and a non-negative integer $D$ , the depth-2 $D$ Subgraph GNN unfolding tree of graph $G$ at $(u, v)$ , denoted as $\left(F_G^{\text{Sub},(D)}(u, v), T_G^{\text{Sub},(D)}(u, v)\right)$ , is a tree-decomposed graph $(F, T^r) \in S^{\text{Sub}}$ constructed as follows:

1. Initialization. At the beginning, $F = G[\{u,v\}]$ (if $u = v$ , $F$ only has one vertex), and $T$ only has a root node $r$ with $\beta_T(r) = \{\{u,v\}\}$ . Define a mapping $\pi : V_F \to V_G$ as $\pi(u) = u$ and $\pi(v) = v$ .   
2. Loop for $D$ rounds. For each leaf node $t$ in $T^r$ , do the following procedure:

Let $\beta_T(t) = \{\{u, x\}\}$ . For each $w \in V_G$ , add a fresh child node $t_w$ to $T^r$ and designate $t$ as its parent. Then, consider the following three cases:

a) If $w \neq \pi(u)$ and $w \neq \pi(x)$ , then add a fresh vertex $z$ to $F$ and extend $\pi$ with $\pi(z) = w$ . The label of $z$ in $F$ is set by $\ell_F(z) = \ell_G(w)$ . Define $\beta_T(t_w) = \beta_T(t) \cup \{\{z\}\}$ . Then, we add edges between $z$ and $\beta_T(t)$ , so that $\pi$ is an isomorphism from $F[\beta_T(t_w)]$ to $G[\pi(\beta_T(t_w))]$ .   
b) If $w = \pi(u)$ , then we simply set $\beta_T(t_w) = \beta_T(t) \cup \{u\}$ without modifying graph $F$ .   
c) If $w = \pi(x)$ , then we simply set $\beta_T(t_w) = \beta_T(t) \cup \{\{x\}\}$ without modifying graph $F$ .

Finally, add a fresh child node $t_w'$ to $T^r$ , designate $t_w$ as its parent, and set $\beta_T(t_w')$ based on the following three cases:

a) If $w \neq \pi(u)$ and $w \neq \pi(x)$ , then $\beta_T(t_w') = \{u, z\}$ .   
b) If $w = \pi(u)$ , then $\beta_T(t_w') = \{\{u, u\}\}$ .   
c) If $w = \pi(x)$ , then $\beta_T(t_w') = \{u, x\}$ .

It is easy to see that the depth of tree $T^{r}$ increases by 2 after each round, $T^{r}$ is always a canonical tree decomposition of F, and $(F,T^{r})\in\mathcal{S}^{\mathrm{Sub}}$ . An illustration of the construction of unfolding tree is given in Figure 6(a).

We next define the unfolding tree of Local 2-GNN, which differs in the loop part such that the bags $\beta_{T}(t_{w})$ and $\beta_{T}(t_{w}^{\prime})$ now do not necessarily contain $u$ .

Definition C.11 (Unfolding tree of Local 2-GNN). Given a graph $G$ , vertices $u, v \in V_G$ , and a non-negative integer $D$ , the depth-2D Local 2-GNN unfolding tree of graph $G$ at $(u, v)$ , denoted as $\left(F_G^{\mathsf{L},(D)}(u, v), T_G^{\mathsf{L},(D)}(u, v)\right)$ , is a tree-decomposed graph $(F, T^r) \in S^{\mathsf{L}}$ constructed as follows:

1. Initialization. The procedure is exactly the same as Subgraph GNN (Definition C.10).   
2. Loop for $D$ rounds. For each leaf node $t$ in $T^r$ , do the following procedure:

Let $\beta_T(t) = \{\{x, y\}\}$ . For each $w \in V_G$ , add a fresh child node $t_w$ to $T$ and designate $t$ as its parent. Then, consider the following three cases:

a) If $w \neq \pi(x)$ and $w \neq \pi(y)$ , then add a fresh vertex $z$ to $F$ and extend $\pi$ with $\pi(z) = w$ . The label of $z$ in $F$ is set by $\ell_F(z) = \ell_G(w)$ . Define $\beta_T(t_w) = \beta_T(t) \cup \{\{z\}\}$ . Then, we add edges between $z$ and $\beta_T(t)$ , so that $\pi$ is an isomorphism from $F[\beta_T(t_w)]$ to $G[\pi(\beta_T(t_w))]$ .   
b) If $w = \pi(x)$ , then we simply set $\beta_T(t_w) = \beta_T(t) \cup \{\{x\}\}$ without modifying graph $F$ .   
c) If $w = \pi(y)$ , then we simply set $\beta_T(t_w) = \beta_T(t) \cup \{\{y\}\}$ without modifying graph $F$ .

Next, add a fresh child node $t_w'$ in $T^r$ , designate $t_w$ as its parent, and set $\beta_T(t_w')$ based on the following three cases:

a) If $w \neq \pi(x)$ and $w \neq \pi(y)$ , then $\beta_T(t_w') = \{x, z\}$ .   
b) If $w = \pi(x)$ , then $\beta_T(t_w') = \{x, x\}$ .   
c) If $w = \pi(y)$ , then $\beta_T(t_w') = \{[x, y]\}$ .

Finally, we repeat the above procedure (point 2) once more, but this time the bag $\beta_{T}(t_w^{\prime})$ is replaced by the following three cases (changing $x$ to $y$ ):

(a)   
![](images/e1029ff262a7ef986811155d185b615359c7301254d576e40fae3a2d3d6908bf.jpg)  
G

![](images/221ce364ce9ba3f3c97cd53e8f5b2dea4c6cdfb553a526c4154d1b06cb18d94d.jpg)  
$F_{G}^{\mathrm{Sub},(1)}(1,2)$

![](images/8c4439cc566486a6d2c7030a000acfd435323bee245eb319c0abbab238c765e0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1,2"] --> B["1,2,1"]
    A --> C["1,2,2"]
    A --> D["1,2,3'"]
    A --> E["1,2,4'"]
    A --> F["1,2,5'"] 
    B --> G["1,1"]
    C --> H["1,2"]
    D --> I["1,3'"]
    E --> J["1,4'"]
    F --> K["1,5'"]
```
</details>

$T_{G}^{\mathrm{Sub},(1)}(1,2)$

(b)   
![](images/9892bf2ee0436dbbfacdb9c014bc0339f89da1715343bbeb9904f322f7c68565.jpg)  
G

![](images/b8c641d0e05d209703c41fa139739d1bef928fcf7dfc27d6fbd3d3298294d98b.jpg)  
$F_{G}^{\mathsf{L},(1)}(1,2)$

![](images/a7ab05cc814a7595f7b27f26a603cbf6b6f76e8d7c57596895f586e833eb22cd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1,2"] --> B["1,2,1"]
    A --> C["1,2,2"]
    A --> D["..."]
    A --> E["1,2,5'"]
    A --> F["1,2,1"]
    A --> G["1,2,2"]
    A --> H["..."]
    A --> I["1,2,5''"]
    B --> J["1,1"]
    C --> K["1,2"]
    D --> L["..."]
    E --> M["1,5'"]
    F --> N["2,1"]
    G --> O["2,2"]
    H --> P["..."]
    I --> Q["2,5''"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#ccf,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#ccf,stroke:#333
    style H fill:#ccf,stroke:#333
    style I fill:#ccf,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#cfc,stroke:#333
    style M fill:#cfc,stroke:#333
    style N fill:#cfc,stroke:#333
    style O fill:#cfc,stroke:#333
    style P fill:#cfc,stroke:#333
    style Q fill:#cfc,stroke:#333
```
</details>

$T_{G}^{\mathsf{L},(1)}(1,2)$

(c)   
![](images/b22cb147125f65c4652c4ab3a5e49af8fe45e703755e34f843d539940711427c.jpg)  
G

![](images/a49f6ba5dba636cf4910ebeb72bd3bf37255a127fceeccaef164c70d496982d5.jpg)  
4"   
$F_{G}^{\mathsf{LF},(1)}(1,2)$

![](images/c35c72d8b7d7993aec491b451ea27c376b5bc355a36234d3db5ea16a290f7eac.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1,2"] --> B["1,2,1"]
    A --> C["1,2,2"]
    A --> D["1,2,4'"]
    A --> E["1,2,4&quot;"]
    A --> F["1,2,3'"]
    A --> G["1,2,5'"]
    B --> H["1,1"]
    B --> I["1,2"]
    C --> J["1,2"]
    C --> K["2,2"]
    D --> L["1,4'"]
    D --> M["2,4''"]
    E --> N["1,3'"]
    E --> O["2,3'"]
    F --> P["1,5'"]
    F --> Q["2,5'"]
```
</details>

$T_{G}^{\mathsf{LF},(1)}(1,2)$

(d)   
![](images/00c7bef69a8904acae9b336c77cff3ce1a118fa97fadcec1a79a23121541e1f5.jpg)  
G

![](images/a4ac27c55b11e6b7af13ea4107d9bbfb2010a9ffa997a741d32077bcd855a48c.jpg)  
$F_{G}^{\mathsf{F},(1)}(1,2)$

![](images/5a9cf1592e7253de83856afb5ea5a9498d21c0ba392f98709bbb33e25cc54fb4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1,2"] --> B["1,2,1"]
    A --> C["1,2,2"]
    A --> D["1,2,3'"]
    A --> E["1,2,4'"]
    A --> F["1,2,5'"] 
    B --> G["1,1"]
    B --> H["1,2"]
    C --> I["1,2"]
    C --> J["2,2"]
    D --> K["1,3'"]
    D --> L["2,3'"]
    E --> M["1,4'"]
    E --> N["2,4'"]
    F --> O["1,5'"]
    F --> P["2,5'"]
```
</details>

$T_{G}^{\mathsf{F},(1)}(1,2)$   
Figure 6: The depth-2 unfolding tree of graph G at vertex pair (1,2) for Subgraph GNN, Local 2-GNN, Local 2-FGNN, and 2-FGNN, respectively.

a) If $w \neq \pi(x)$ and $w \neq \pi(y)$ , then $\beta_T(t_w') = \{y, z\}$ .   
b) If $w = \pi(y)$ , then $\beta_T(t'_w) = \{y, y\}$ .   
c) If $w = \pi(x)$ , then $\beta_T(t_w') = \{x, y\}$ .

An illustration of the construction of unfolding tree is given in Figure 6(b).

We next define the unfolding tree of Local 2-FGNN, which differs in the loop part such that the node $t_w$ can have two children under certain conditions.

Definition C.12 (Unfolding tree of Local 2-FGNN). Given a graph G, vertices $u, v \in V_{G}$ , and a non-negative integer D, the depth-2D Local 2-FGNN unfolding tree of graph G at $(u, v)$ , denoted as $\left(F_{G}^{\mathrm{LF},(D)}(u, v), T_{G}^{\mathrm{LF},(D)}(u, v)\right)$ , is a tree-decomposed graph $(F, T^{r}) \in S^{\mathrm{LF}}$ constructed as follows:

1. Initialization. The procedure is exactly the same as Subgraph GNN (Definition C.10).   
2. Loop for D rounds. For each leaf node t in $T^{r}$ , do the following procedure:

Let $\beta_T(t) = \{[x, y]\}$ . For each $w \in N_G[\pi(x)] \cup N_G[\pi(y)]$ , add a fresh child node $t_w$ to $T$ and designate $t$ as its parent. Then, consider the following three cases:

a) If $w \neq \pi(x)$ and $w \neq \pi(y)$ , then add a fresh vertex $z$ to $F$ and extend $\pi$ with $\pi(z) = w$ . The label of $z$ in $F$ is set by $\ell_F(z) = \ell_G(w)$ . Define $\beta_T(t_w) = \beta_T(t) \cup \{\{z\}\}$ . Then, we add edges between $z$ and $\beta_T(t)$ , so that $\pi$ is an isomorphism from $F[\beta_T(t_w)]$ to $G[\pi(\beta_T(t_w))]$ .   
b) If $w = \pi(x)$ , then we simply set $\beta_T(t_w) = \beta_T(t) \cup \{x\}$ without modifying graph $F$ .   
c) If $w = \pi(y)$ , then we simply set $\beta_T(t_w) = \beta_T(t) \cup \{\{y\}\}$ without modifying graph $F$ .

Next, add two fresh children $t_w'$ and $t_w''$ in $T^r$ , designate $t_w$ as their parent, and set $\beta_T(t_w')$ and $\beta_T(t_w')$ based on the following three cases:

a) If $w \neq \pi(x)$ and $w \neq \pi(y)$ , then $\beta_T(t_w') = \{x, z\}$ and $\beta_T(t_w'') = \{y, z\}$ .   
b) If $w = \pi(x)$ , then $\beta_T(t'_w) = \{ \{x, x\} \}$ and $\beta_T(t''_w) = \{ \{x, y\} \}$ .   
c) If $w = \pi(y)$ , then $\beta_T(t_w') = \{x, y\}$ and $\beta_T(t_w'') = \{y, y\}$ .

For each $w \notin N_G[\pi(x)] \cup N_G[\pi(y)]$ , follow the same procedure as Local 2-GNN (Definition C.11).

An illustration of the construction of unfolding tree is given in Figure 6(c).

We finally define the unfolding tree of 2-FGNN, which differs in the loop part such that all nodes $t_{w}$ have two children.

Definition C.13 (Unfolding tree of 2-FGNN). Given a graph $G$ , vertices $u, v \in V_G$ , and a non-negative integer $D$ , the depth-2D 2-FGNN unfolding tree of graph $G$ at $(u, v)$ , denoted as $\left(F_G^{\mathrm{Sub},(D)}(u, v), T_G^{\mathrm{Sub},(D)}(u, v)\right)$ , is a tree-decomposed graph $(F, T^r) \in S^F$ constructed as follows:

1. Initialization. The procedure is exactly the same as Local 2-FGNN (Definition C.12).   
2. Loop for $D$ rounds. The procedure is similar to Local 2-FGNN (Definition C.12) except that the condition $w \in N_G[\pi(x)] \cup N_G[\pi(y)]$ is relaxed to all vertices.

An illustration of the construction of unfolding tree is given in Figure 6(d).

We are now ready to present the first core result:

Lemma C.14. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. For any two graphs $G, H$ , any vertices $u, v \in V_G$ , $x, y \in V_H$ , and any non-negative integer $D$ , $\tilde{\chi}_G^{M,(D)}(u, v) = \tilde{\chi}_H^{M,(D)}(x, y)$ iff there exists an isomorphism $(\rho, \tau)$ from $\left(F_G^{M,(D)}(u, v), T_G^{M,(D)}(u, v)\right)$ to $\left(F_H^{M,(D)}(x, y), T_H^{M,(D)}(x, y)\right)$ such that $\rho(u) = x, \rho(v) = y$ .

Proof. Here, we only give the proof for Local 2-GNN, and the proofs for Subgraph GNN, Local 2-FGNN and 2-FGNN are almost the same so we omit them for clarity.

Proof for Local 2-GNN. The proof is based on induction over D. When D = 0, the theorem obviously holds. Now assume that the theorem holds for $D \leq d$ , and consider $D = d + 1$ . Below, we omit L in the corner mark for clarity.

1. We first prove that $\tilde{\chi}_G^{(d + 1)}(u,v) = \tilde{\chi}_H^{(d + 1)}(x,y)$ implies that there exists an isomorphism $(\rho ,\tau)$ from $\left(F_G^{(d + 1)}(u,v),T_G^{(d + 1)}(u,v)\right)$ to $\left(F_H^{(d + 1)}(x,y),T_H^{(d + 1)}(x,y)\right)$ such that $\rho (u) = x,\rho (v) = y$ . If $\tilde{\chi}_G^{(d + 1)}(u,v) = \tilde{\chi}_H^{(d + 1)}(x,y)$ , then

$$
\left\{\left(\tilde {\chi} _ {G} ^ {(d)} (u, w), \mathsf {a t p} _ {G} (u, v, w)\right): w \in V _ {G} \right\} = \left\{\left(\tilde {\chi} _ {H} ^ {(d)} (x, z), \mathsf {a t p} _ {H} (x, y, z)\right): z \in V _ {H} \right\}, \tag {11}
$$

$$
\left\{\left(\tilde {\chi} _ {G} ^ {(d)} (w, v), \mathsf {a t p} _ {G} (u, v, w)\right): w \in V _ {G} \right\} = \left\{\left(\tilde {\chi} _ {H} ^ {(d)} (z, y), \mathsf {a t p} _ {H} (x, y, z)\right): z \in V _ {H} \right\}. \tag {12}
$$

Let $n = |V_G| = |V_H|$ . Thus, we can denote $V_G = \{w_1, \cdots, w_n\} = \{w_1', \cdots, w_n'\}$ and $V_H = \{z_1, \cdots, z_n\} = \{z_1', \cdots, z_n'\}$ such that

- $(\tilde{\chi}_G^{(d)}(u, w_i), \mathsf{atp}_G(u, v, w_i)) = (\tilde{\chi}_H^{(d)}(x, z_i), \mathsf{atp}_H(x, y, z_i))$ for all $i \in [n]$ ;   
- $(\tilde{\chi}_G^{(d)}(w_i', v), \mathsf{atp}_G(u, v, w_i')) = (\tilde{\chi}_H^{(d)}(z_i', y), \mathsf{atp}_H(x, y, z_i'))$ for all $i \in [n]$ .

On the other hand, by definition of tree unfolding, we have

$$
F _ {G} ^ {(d + 1)} (u, v) = \left(\bigcup_ {w _ {i}} F _ {G} ^ {(d)} (u, w _ {i})\right) \cup \left(\bigcup_ {w _ {i} ^ {\prime}} F _ {G} ^ {(d)} (w _ {i} ^ {\prime}, v)\right) \cup F _ {G} ^ {(1)} (u, v),
$$

$$
F _ {H} ^ {(d + 1)} (x, y) = \left(\bigcup_ {z _ {i}} F _ {H} ^ {(d)} (x, z _ {i})\right) \cup \left(\bigcup_ {z _ {i} ^ {\prime}} F _ {H} ^ {(d)} (z _ {i} ^ {\prime}, y)\right) \cup F _ {H} ^ {(1)} (x, y),
$$

![](images/95126eb1f980b4380bd210aa16d570f4d8c9ad5e5cc494134f876ff3fde0dff9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["u"] --> B["wi"]
    A --> C["F^(2d)(w'_1,v)"]
    A --> D["F^(2d)(w'_2,v)"]
    A --> E["F^(2d)(u,w_1)"]
    A --> F["F^(2d)(u,w_2)"]
    A --> G["i=3,...n"]
    H["w_i"] --> I["wi"]
    H --> J["F^(2d)(w'_2,v)"]
    H --> K["F^(2d)(u,w_i)"]
    L["wi"] --> M["i=3,...n"]
    N["u,v"] --> O["u,v,w_1"]
    N --> P["u,v,w_2"]
    N --> Q["u,v,w_n"]
    N --> R["u,v,w'_1"]
    N --> S["u,v,w'_2"]
    N --> T["u,v,w'_n"]
    U["T^(2d)(u,w_1)"] --> V["T^(2d)(u,w_2)"]
    U --> W["T^(2d)(u,w_n)"]
    X["T^(2d)(w'_1,v)"] --> Y["T^(2d)(w'_2,v)"]
    X --> Z["T^(2d)(w'_n,v)"]
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    style P fill:#f9f,stroke:#333
    style Q fill:#f9f,stroke:#333
    style R fill:#f9f,stroke:#333
    style S fill:#f9f,stroke:#333
    style T fill:#f9f,stroke:#333
    style U fill:#f9f,stroke:#333
    style V fill:#f9f,stroke:#333
    style W fill:#f9f,stroke:#333
    style X fill:#f9f,stroke:#333
    style Y fill:#f9f,stroke:#333
    style Z fill:#f9f,stroke:#333
```
</details>

Figure 7: Illustration of the proof of Lemma C.14.

where $\cup$ represents the graph union. Here, all $w_{i}, w_{j}^{\prime} \notin \{u, v\}$ in different graphs are treated as different vertices when taking the union, while $u, v$ in different graphs are shared. See Figure 7 for an illustration of the above equations.

By induction, there exists an isomorphism $(\rho_i, \tau_i)$ from $\left(F_G^{(d)}(u, w_i), T_G^{(d)}(u, w_i)\right)$ to $\left(F_H^{(d)}(x, z_i), T_H^{(d)}(x, z_i)\right)$ such that $\rho_i(u) = x$ , $\rho_i(w_i) = z_i$ ( $i \in [n]$ ), and there exist an isomorphism $\rho_i'$ from $\left(F_G^{(d)}(w_i', v), T_G^{(d)}(w_i', v)\right)$ to $\left(F_H^{(d)}(z_i', y), T_H^{(d)}(z_i', y)\right)$ such that $\rho_i'(v) = y$ , $\rho_i'(w_i') = z_i'$ ( $i \in [n]$ ). Moreover, we have $\mathrm{atp}_G(u, v, w_i) = \mathrm{atp}_H(x, y, z_i)$ and $\mathrm{atp}_G(u, v, w_i') = \mathrm{atp}_H(x, y, z_i')$ for $i \in [n]$ , which implies that $F_G^{(1)}(u, v)$ is isomorphic to $F_H^{(1)}(x, y)$ . Therefore, if we construct $\tilde{\rho}$ by merging all $\rho_i$ and $\rho_i'$ ( $i \in [n]$ ), and construct $\tilde{\tau}$ by merging all $\tau_i$ and $\tau_i'$ and further specifying an appropriate mapping between tree nodes of depth no more than 1 in $T_G^{(d+1)}(u, v)$ and $T_H^{(d+1)}(x, y)$ , then it is straightforward to see that $(\tilde{\rho}, \tilde{\tau})$ is well-defined and is an isomorphism from $\left(F_G^{(d+1)}(u, v), T_G^{(d+1)}(u, v)\right)$ to $\left(F_H^{(d+1)}(x, y), T_H^{(d+1)}(x, y)\right)$ such that $\tilde{\rho}(u) = x$ , $\tilde{\rho}(v) = y$ .

2. We next prove that if there exists an isomorphism $(\rho, \tau)$ from the tree-decomposed graph $\left(F_G^{(d+1)}(u, v), T_G^{(d+1)}(u, v)\right)$ to $\left(F_H^{(d+1)}(x, y), T_H^{(d+1)}(x, y)\right)$ such that $\rho(u) = x, \rho(v) = y$ , then $\tilde{\chi}_G^{(d+1)}(u, v) = \tilde{\chi}_H^{(d+1)}(x, y)$ .

Without loss of generality, assume $u \neq v$ and $x \neq y$ . since $\tau$ is an isomorphism from $T_{G}^{(d+1)}(u,v)$ to $T_{H}^{(d+1)}(x,y)$ , $\tau$ maps all tree nodes of depth 1 in $T_{G}^{(d+1)}(u,v)$ to all tree nodes of depth 1 in $T_{H}^{(d+1)}(x,y)$ . Let $s_{1},\cdots,s_{n}$ be all nodes of depth 2 in $T_{G}^{(d+1)}(u,v)$ such that $u \in \beta_{T_{G}^{(d+1)}(u,v)}(s_{i})$ (it follows that $n = |V_{G}|$ ), and let $s_{i}'$ be the parent of $s_{i}$ . Similarly, let $t_{1},\cdots,t_{n}$ be all nodes of depth 2 in $T_{H}^{(d+1)}(x,y)$ such that $x \in \beta_{T_{H}^{(d+1)}(x,y)}(t_{i})$ , and let $t_{i}'$ be the parent of $t_{i}$ . Moreover, we can arrange the order so that the following are satisfied (for each $i \in [n]$ ):

a) $\tau$ is an isomorphism from the subtree $T_G^{(d + 1)}(u,v)[s_i]$ to the subtree $T_H^{(d + 1)}(x,y)[t_i]$ .   
b) For all $s \in \mathsf{Desc}_{T_G^{(d+1)}(u,v)}(s_i), \rho(\beta_{T_G^{(d+1)}(u,v)}(s)) = \beta_{T_H^{(d+1)}(x,y)}(\tau(s))$ .   
c) By definition of the unfolding tree, $\rho$ is an isomorphism from the induced subgraph $F_{G}^{(d + 1)}(u,v)[T_{G}^{(d + 1)}(u,v)[s_{i}]]$ to the induced subgraph $F_{H}^{(d + 1)}(x,y)[T_{H}^{(d + 1)}(x,y)[t_{i}]]$ (see Definition C.7).   
d) Let $\beta_{T_G^{(d + 1)}(u,v)}(s_i') = \{\{u,v,\tilde{w}_i\}\}$ and $\beta_{T_H^{(d + 1)}(x,y)}(t_i') = \{\{x,y,\tilde{z}_i\}\}$ . Then, $\rho (\tilde{w}_i) = \tilde{z}_i$ , and thus $\{v,\tilde{w}_i\} \in E_{F_G^{(d + 1)}(u,v)}$ iff $\{y,\tilde{z}_i\} \in E_{F_H^{(d + 1)}(x,y)}$ .

By items (a) to (c), $\left(F_G^{(d + 1)}(u,v)\left[T_G^{(d + 1)}(u,v)[s_i]\right],T_G^{(d + 1)}(u,v)[s_i]\right)$ is isomorphic to $\left(F_H^{(d + 1)}(x,y)\left[T_H^{(d + 1)}(x,y)[t_i]\right],T_H^{(d + 1)}(x,y)[t_i]\right)$ . On the other hand, by definition of the unfolding tree, $\left(F_G^{(d + 1)}(u,v)\left[T_G^{(d + 1)}(u,v)[s_i]\right],T_G^{(d + 1)}(u,v)[s_i]\right)$ is isomorphic to

the depth-2d unfolding tree $\left(F_G^{(d)}(u, w_i), T_G^{(d)}(u, w_i)\right)$ for some $w_i \in V_G$ satisfying that $\{w_i, v\} \in E_G$ iff $\{\tilde{w}_i, v\} \in E_{F_G^{(d+1)}(u,v)}$ .

Similarly, $\left(F_H^{(d + 1)}(x,y)\left[T_H^{(d + 1)}(x,y)[t_i]\right],T_H^{(d + 1)}(x,y)[t_i]\right)$ is isomorphic to $\left(F_H^{(d)}(x,z_i),T_H^{(d)}(x,z_i)\right)$ for some $z_{i}\in V_{H}$ satisfying that $\{z_i,y\} \in E_H$ iff $\{\tilde{z}_i,y\} \in E_{F_H^{(d + 1)}(x,y)}$ . Combining all the above equivalence yields that $\left(F_G^{(d)}(u,w_i),T_G^{(d)}(u,w_i)\right)$ is isomorphic to $\left(F_H^{(d)}(x,z_i),T_H^{(d)}(x,z_i)\right)$ , and $\{w_i,v\} \in E_G$ iff $\{z_i,y\} \in E_H$ .

By induction, we have $\tilde{\chi}_G^{(d)}(u,w_i) = \tilde{\chi}_H^{(d)}(x,z_i)$ . Moreover, we clearly have that $\{u,w_i\} \in E_G$ iff $\{x,z_i\} \in E_H$ , and $\{u,v\} \in E_G$ iff $\{x,y\} \in E_H$ . Therefore,

$$
(\tilde {\chi} _ {G} ^ {(d)} (u, w _ {i}), \mathsf {a t p} _ {G} (u, v, w _ {i})) = (\tilde {\chi} _ {H} ^ {(d)} (x, z _ {i}), \mathsf {a t p} _ {H} (x, y, z _ {i})). \tag {13}
$$

Next, note that $\tilde{w}_i$ are different from each other for $i\in [n]$ by definition of unfolding tree. Thus, $w_{i}$ are also different from each other. It follows that

$$
\left\{\left(\tilde {\chi} _ {G} ^ {(d)} (u, w), \mathsf {a t p} _ {G} (u, v, w)\right): w \in V _ {G} \right\} = \left\{\left(\tilde {\chi} _ {H} ^ {(d)} (x, z), \mathsf {a t p} _ {H} (x, y, z)\right): z \in V _ {H} \right\}. \tag {14}
$$

Again using the same analysis as before, we obtain

$$
\left\{\left(\tilde {\chi} _ {G} ^ {(d)} (w, v), \mathsf {a t p} _ {G} (u, v, w)\right): w \in V _ {G} \right\} = \left\{\left(\tilde {\chi} _ {H} ^ {(d)} (z, y), \mathsf {a t p} _ {H} (x, y, z)\right): z \in V _ {H} \right\}. \tag {15}
$$

It remains to prove that $\tilde{\chi}_G^{(d)}(u,v) = \tilde{\chi}_H^{(d)}(x,y)$ . To prove this, note that Equation (14) implies that

$$
\left\{\left(\tilde {\chi} _ {G} ^ {(d ^ {\prime})} (u, w), \mathsf {a t p} _ {G} (u, v, w)\right): w \in V _ {G} \right\} = \left\{\left(\tilde {\chi} _ {H} ^ {(d ^ {\prime})} (x, z), \mathsf {a t p} _ {H} (x, y, z)\right): z \in V _ {H} \right\} \tag {16}
$$

holds for all $0 \leq d' \leq d$ , and Equation (15) implies that

$$
\left\{\left(\tilde {\chi} _ {G} ^ {(d ^ {\prime})} (w, v), \mathsf {a t p} _ {G} (u, v, w)\right): w \in V _ {G} \right\} = \left\{\left(\tilde {\chi} _ {H} ^ {(d)} (z, y), \mathsf {a t p} _ {H} (x, y, z)\right): z \in V _ {H} \right\}. \tag {17}
$$

holds for all $0 \leq d' \leq d$ . Combined with Equations (16) and (17) and the fact that $\tilde{\chi}_G^{(0)}(u,v) = \tilde{\chi}_H^{(0)}(x,y)$ , we can incrementally prove that $\tilde{\chi}_G^{(d')} (u,v) = \tilde{\chi}_H^{(d')} (x,y)$ for all $d' \leq d + 1$ .

We have thus concluded the proof.

![](images/1369cc9495996bb413b026296368bf83259f885786a2be0a36ebd0d2318dd7ee.jpg)

Definition C.15. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. Given a graph $G$ and a tree-decomposed graph $(F, T^r)$ , define

$$
\mathsf {c n t} ^ {M} \left((F, T ^ {r}), G\right) := \left| \left\{(u, v) \in V _ {G} ^ {2}: \exists D \in \mathbb {N} _ {+} \text {s.t.} \left(F _ {G} ^ {M, (D)} (u, v), T _ {G} ^ {M, (D)} (u, v)\right) \simeq (F, T ^ {r}) \right\} \right|,
$$

where $\left(F_G^{M,(D)}(u,v),T_G^{M,(D)}(u,v)\right)$ is the depth-2D unfolding tree of $G$ at $(u,v)$ for model $M$ .

Corollary C.16. Let $M \in \{Sub, L, LF, F\}$ be any model. For any graphs G, H, $\chi_{G}^{M}(G) = \chi_{H}^{M}(H)$ iff $\text{cnt}^{M}((F, T^{r}), G) = \text{cnt}^{M}((F, T^{r}), H)$ holds for all $(F, T^{r}) \in \mathcal{S}^{M}$ .

Proof. “ $\Longrightarrow$ ”. If $\chi_G^M(G) = \chi_H^M(H)$ , then $\{\{\chi_G^M(u,v):u,v\in V_G\} = \{\{\chi_H^M(x,y):x,y\in V_H\}\}$ . For each color $c$ in the above multiset, pick $u,v\in V_G$ with $\chi_G^M(u,v) = c$ . It follows that if $(F,T^r)\simeq (F_G^{M,(D)}(u,v),T_G^{M,(D)}(u,v))\in S^M$ for some $D$ , then $\mathsf{cnt}^M((F,T^r),G) = |\{(u,v)\in V_G^2:\chi_G^M(u,v) = c\}| = |\{(x,y)\in V_H^2:\chi_H^M(x,y) = c\}| = \mathsf{cnt}^M((F,T^r),H)$ by Lemma C.14. On the other hand, if $(F,T^r)\neq (F_G^{M,(D)}(u,v),T_G^{M,(D)}(u,v))$ for all $u,v\in V_G$ and all $D$ , then clearly $\mathsf{cnt}^M((F,T^r),G) = \mathsf{cnt}^M((F,T^r),H) = 0$ .

“ $\Longleftarrow$ ”. If $\mathsf{cnt}^M((F,T^r),G) = \mathsf{cnt}^M((F,T^r),H)$ holds for all $(F,T^r)\in S^M$ , it clearly holds for all $(F_M^{(D)}(u,v),T_M^{(D)}(u,v))$ with $u,v\in V_G$ and a sufficiently large $D$ . This guarantees that for all color $c$ , $|\{(u,v)\in V_G^2:\chi_G^M (u,v) = c\}| = |\{(x,y)\in V_H^2:\chi_H^M (x,y) = c\}|$ by Lemma C.14. Therefore, $\{\{\chi_G^M (u,v):u,v\in V_G\} = \{\{\chi_H^M (x,y):x,y\in V_H\}\}$ , concluding the proof.

We next define an important concept called bag isomorphism (Dell et al., 2018).

Definition C.17. Given a tree-decomposed graph $(F,T^{r})$ and a graph G, a bag isomorphism from $(F,T^{r})$ to G (abbreviated as “bIso”) is a homomorphism f from F to G such that, for all $t\in V_{T}$ , f is an isomorphism from $F[\beta_{T}(t)]$ to $G[f(\beta_{T}(t))]$ . Denote $\mathsf{B}\mathsf{l}\mathsf{o}((F,T^{r}),G)$ to be the set of all bag isomorphisms from $(F,T^{r})$ to G, and denote $\mathsf{b}\mathsf{l}\mathsf{o}((F,T^{r}),G)=|\mathsf{B}\mathsf{l}\mathsf{o}((F,T^{r}),G)|$ .

Remark C.18. To prove that a mapping $f$ from $(F, T^r)$ to $G$ is a bIso, it suffices to prove the following conditions:

1. For any two different vertices $u, v \in V_G$ in the same bag, $f(u) \neq f(v)$ ;   
2. For any two vertices $u, v \in V_G$ in the same bag, $\{u, v\} \in E_F$ iff $\{f(u), f(v)\} \in E_G$ ;   
3. For any $u \in V_{G}$ , $\ell_{F}(u) = \ell_{G}(f(u))$ .

The following fact is straightforward from the construction of the unfolding tree:

Fact C.19. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model considered above. For any graph $G$ , any vertex pair $(u, v) \in V_G^2$ , and any non-negative integer $D$ , there is a bIso $\pi$ from $\left(F_G^{M,(D)}(u, v), T_G^{M,(D)}(u, v)\right)$ to $G$ .

Similarly, we need the following concept to describe the relation between two tree-decomposed graphs. We note that these technical concepts also appeared in Dell et al. (2018).

Definition C.20. Given two tree-decomposed graphs $(F,T^{r})$ and $(\tilde{F},\tilde{T}^{s})$ , a pair of mappings $(\rho,\tau)$ is called homomorphism from $(F,T^{r})$ to $(\tilde{F},\tilde{T}^{s})$ if it satisfies the following conditions:

a) $\tau$ is a homomorphism from $T$ to $\tilde{T}$ (ignording labels $\beta$ ) and is depth-preserving, i.e., $\mathsf{dep}_{T^r}(t) = \mathsf{dep}_{\tilde{T}^s}(\tau(t))$ for all $t \in V_T$ ;   
b) For all $t \in V_T$ , $\rho$ is a homomorphism from $F[\beta_T(t)]$ to $\tilde{F}[\beta_{\tilde{T}}(\tau(t))]$ . Note that this implies that $\rho$ is a homomorphism from $F$ to $\tilde{F}$ .   
c) The depth of $T^r$ is equal to the depth of $\tilde{T}^s$ .

Definition C.21. Under Definition C.20, $(\rho, \tau)$ is further called a bag-isomorphism homomorphism (abbreviated as "bIsoHom") from $(F, T^r)$ to $(\tilde{F}, \tilde{T}^s)$ if it a homomorphism satisfying that, for all $t \in V_T$ , $\rho$ is an isomorphism from $F[\beta_T(t)]$ to $\tilde{F}[\beta_{\tilde{T}}(\tau(t))]$ . Furthermore, $(\rho, \tau)$ is called a bIsoSurj if $\tau$ is surjective; and $(\rho, \tau)$ is called a bIsoInj if $\tau$ is injective. We use BIsoHom $\left((F, T^r), (\tilde{F}, \tilde{T}^s)\right)$ to denote the set of bIsoHoms from $(F, T^r)$ to $(\tilde{F}, \tilde{T}^s)$ , and let bIsoHom $\left((F, T^r), (\tilde{F}, \tilde{T}^s)\right) = \left|\text{BIsoHom}\left((F, T^r), (\tilde{F}, \tilde{T}^s)\right)\right|$ . The notations BIsoSurj $\left((F, T^r), (\tilde{F}, \tilde{T}^s)\right)$ , bIsoSurj $\left((F, T^r), (\tilde{F}, \tilde{T}^s)\right)$ , BisoInj $\left((F, T^r), (\tilde{F}, \tilde{T}^s)\right)$ , and bisoInj $\left((F, T^r), (\tilde{F}, \tilde{T}^s)\right)$ are defined accordingly.

Remark C.22. In the above definition, the depth of a tree $T^{r}$ is the maximal depth among all tree nodes in $T^{r}$ . Note that we do not require that all leaf nodes have the same depth in $T^{r}$ .

We are now ready to present the second core result:

Lemma C.23. Let $M \in \{Sub, L, LF, F\}$ be any model. For any graph G and tree-decomposed graph $(F, T^{r}) \in \mathcal{S}^{M}$ ,

$$
\operatorname{blso} \left(\left(F, T ^ {r}\right), G\right) = \sum_ {\left(\tilde {F}, \tilde {T} ^ {s}\right) \in \mathcal {S} ^ {M}} \operatorname{blsoHom} \left(\left(F, T ^ {r}\right), \left(\tilde {F}, \tilde {T} ^ {s}\right)\right) \cdot \operatorname{cnt} ^ {M} \left(\left(\tilde {F}, \tilde {T} ^ {s}\right), G\right). \tag {18}
$$

Here, the summation ranges over all non-isomorphic (tree-decomposed) graphs in $S^{M}$ and is well-defined as there are only a finite number of graphs making the value in the summation non-zero.

Proof. Here, we only give the proof for Local 2-GNN, and the proofs for Subgraph GNN, Local 2-FGNN and 2-FGNN are almost the same so we omit them for clarity.

Proof for Local 2-GNN. We assume that the root bag of $(F, T^r)$ is $\{\{u, v\}\}$ , and the depth of $(F, T^r)$ is $2d$ . Let $y, z \in V_G$ be any vertices in $G$ , and denote $(F_G^{(d)}(y, z), T_G^{(d)}(y, z))$ as the depth-2d Local 2-GNN unfolding tree at $(y, z)$ . Define the following two sets:

$$
S _ {1} (y, z) = \{g: g \in \mathsf {B l s o} ((F, T ^ {r}), G), g (u) = y, g (v) = z \},
$$

$$
S _ {2} (y, z) = \left\{(\rho , \tau): (\rho , \tau) \in \mathsf {B l s o H o m} \left((F, T ^ {r}), (F _ {G} ^ {(d)} (y, z), T _ {G} ^ {(d)} (y, z))\right), \rho (u) = y, \rho (v) = z \right\}.
$$

Then, Lemma C.23 is equivalent to the following equation:

$$
\sum_ {y, z \in V _ {G}} | S _ {1} (y, z) | = \sum_ {y, z \in V _ {G}} | S _ {2} (y, z) |.
$$

We will prove that $|S_1(y,z)| = |S_2(y,z)|$ for all $y, z \in V_G$ .

Given $y, z \in V_G$ , according to Fact C.19, there exists a bIso $\pi$ from $(F_G^{(d)}(y,z), T_G^{(d)}(y,z))$ to graph $G$ . Define a mapping $\sigma$ such that $\sigma(\rho,\tau) = \pi \circ \rho$ for all $(\rho,\tau) \in S_2(y,z)$ . It suffices to prove that $\sigma$ is a bijection from $S_2(y,z)$ to $S_1(y,z)$ .

1. We first prove that $\sigma$ is a mapping from $S_{2}(y,z)$ to $S_{1}(y,z)$ , i.e., $\pi \circ \rho \in S_{1}(y,z)$ for all $(\rho, \tau) \in S_{2}(y,z)$ . First, we clearly have $(\pi \circ \rho)(u) = \pi(y) = y$ , $(\pi \circ \rho)(v) = \pi(z) = z$ . We next prove that $\pi \circ \rho \in \mathsf{B}\mathsf{l}\mathsf{s}\mathsf{o}((F,T^{r}),G)$ . The proof is based on Remark C.18.

a) Let $w, x \in V_F, w \neq x$ be any vertices in the same bag of $T^r$ . Since $(\rho, \tau)$ is a bIsoHom, $\rho(w) \neq \rho(x)$ and $\rho(w)$ and $\rho(x)$ are in the same bag of $T_G^{(d)}(y, z)$ . Again, since $\pi$ is a bIso, we have $\pi(\rho(w)) \neq \pi(\rho(x))$ .   
b) Let $w, x \in V_F$ be any vertices in the same bag of $T^r$ . Since $(\rho, \tau)$ is a bIso-Hom, $\rho(w)$ and $\rho(x)$ are in the same bag of $T_G^{(d)}(y, z)$ , and $\{w, x\} \in E_F$ iff $\{\rho(w), \rho(x)\} \in E_{F_G^{(d)}(y, z)}$ . Again, since $\pi$ is a bIso, $\{\rho(w), \rho(x)\} \in E_{F_G^{(d)}(y, z)}$ iff $\{\pi(\rho(w)), \pi(\rho(x))\} \in E_G$ . Therefore, $\{w, x\} \in E_F$ iff $\{\pi(\rho(w)), \pi(\rho(x))\} \in E_G$ .   
c) We clearly have $\ell_F(w) = \ell_{F_G^{(d)}(y,z)}(\rho(w)) = \ell_G(\pi(\rho(w)))$ .

We have proved that $\pi \circ \rho \in \mathsf{B}\mathsf{l}\mathsf{o}\left((F,T^{r}),G\right)$ .

2. We then prove that $\sigma$ is a surjection. For all $g \in S_{1}(y,z)$ , we define a mapping $(\rho,\tau)$ from $(F,T^{r})$ to $(F_{G}^{(d)}(y,z),T_{G}^{(d)}(y,z))$ as follows. First define $\rho(u)=y$ , $\rho(v)=z$ , and set $\tau(r)$ to be the root of $(F_{G}^{(d)}(y,z),T_{G}^{(d)}(y,z))$ . Let $w_{1},\cdots,w_{m}\in V_{F}$ and $w_{1}^{\prime},\cdots,w_{m^{\prime}}^{\prime}\in V_{F}$ be vertices such that all $\{u,w_{i}\}$ and $\{w_{i}^{\prime},v\}$ correspond to bags of $T^{r}$ associated to all tree nodes of depth 2. Similarly, by definition of the Local 2-GNN unfolding tree, let $x_{1},\cdots,x_{n}\in V_{F_{G}^{(d)}(y,z)}$ be different vertices and $x_{1}^{\prime},\cdots,x_{n}^{\prime}\in V_{F_{G}^{(d)}(y,z)}$ be different vertices such that all $\{y,x_{i}\}$ and $\{x_{i}^{\prime},z\}$ correspond to bags of $T_{G}^{(d)}(y,z)$ associated to all tree nodes of depth 2. Since g and $\pi$ are bIsos, we have:

- For every $w_{i}$ ( $i \in [m]$ ), there exists $x_{j}$ ( $j \in [n]$ ), such that $g(w_{i}) = \pi(x_{j}) = \tilde{x}_{j}$ for some $\tilde{x}_{j} \in V_{G}$ and $F[\{\{u, v, w_{i}\}\}] \simeq F_{G}^{(d)}(y, z)[\{\{y, z, x_{j}\}\}] \simeq G[\{\{y, z, \tilde{x}_{j}\}\};$   
- For every $w_{i}^{\prime}$ ( $i \in [m']$ ), there exists $x_{j}^{\prime}$ ( $j \in [n]$ ), such that $g(w_{i}^{\prime}) = \pi(x_{j}^{\prime}) = \tilde{x}_{j}^{\prime}$ for some $\tilde{x}_{j}^{\prime} \in V_{G}$ and $F[\{\{u, v, w_{i}^{\prime}\}\}] \simeq F_{G}^{(d)}(y, z)[\{\{y, z, x_{j}^{\prime}\}\}] \simeq G[\{\{y, z, \tilde{x}_{j}^{\prime}\}\}.$

We then define $\rho(w_i) = x_j$ for each $i \in [m]$ and $\rho(w_i') = x_j'$ for each $i \in [m']$ . Based on the above two items, one can easily define $\tau$ such that each node $s$ in $T^r$ of depth 1 or 2 is mapped by $\tau$ to a node $t$ in $T_G^{(d)}(y,z)$ of the same depth, $\rho(\beta_T(s)) = \beta_{T_G^{(d)}}(t)$ , and $\rho$ is an isomorphism from $F[\beta_T(s)]$ to $F_G^{(d)}(y,z)[\beta_{T_G^{(d)}(y,z)}(t)]$ .

Next, we can recursively define $\rho$ 's image on $F[T^r[s]]$ for each tree node $s$ of depth 2 following the same construction above. This is because $g$ is still a bIso from $(F[T^r[s]], T^r[s])$ to $G$ , $\pi$ is still a bIso from $\left(F_G^{(d)}(y,z)[T_G^{(d)}(y,z)[\tau(s)]], T_G^{(d)}(y,z)[\tau(s)]\right)$ to $G$ , and $g(\beta_T(s)) = \pi(\beta_{T_G^{(d)}}(\tau(s)))$ . Recursively applying this procedure, we can construct $(\rho, \tau)$ such that it is a

bIsoHom from $(F, T^r)$ to $\left(F_G^{(d)}(y, z), T_G^{(d)}(y, z)\right)$ . That is to say, we have proved that for all $g \in S_1(u, v)$ , there is a preimage $(\rho, \tau) \in S_2(y, z)$ such that $\sigma(\rho, \tau) = g$ .

3. We finally prove that $\sigma$ is an injection. Let $(\rho_1, \tau_1), (\rho_2, \tau_2) \in S_2(y, z)$ such that $\pi \circ \rho_1 = \pi \circ \rho_2$ . Let $w_1, \cdots, w_m \in V_F$ , $w_1', \cdots, w_{m'}' \in V_F$ , $x_1, \cdots, x_n \in V_{F_G^{(d)}(y,z)}$ , and $x_1', \cdots, x_n' \in V_{F_G^{(d)}(y,z)}$ be defined as in the previous item. For each $i \in [m]$ , let $j_1(i)$ and $j_2(i)$ be indices satisfying $\rho_1(w_i) = x_{j_1(i)}$ and $\rho_2(w_i) = x_{j_2(i)}$ . It follows that $\pi(x_{j_1(i)}) = \pi(x_{j_2(i)})$ . By definition of the Local 2-GNN unfolding tree, we must have $x_{j_1(i)} = x_{j_2(i)}$ , and thus $\rho_1(w_i) = \rho_2(w_i)$ . Using a similar approach, we can prove that $\rho_1(w_i') = \rho_2(w_i')$ for each $i \in [m']$ . Next, we can recursively apply the above procedure to the subtree $T^r[s]$ for each tree node of depth 2 following the previous item, and finally prove that $\rho_1 = \rho_2$ . Therefore, $\sigma$ is an injection.

Combining the above three items completes the proof.

![](images/33e7dd1095274e6861c9ce8432e9b8f1ef191d53e073e01bc9e9b93e318840a0.jpg)

Proposition C.24. Under Definition C.21, (i) if $(\rho,\tau)$ is a bIsoSurj, then $\rho$ is a surjection from F to $\tilde{F}$ on both vertices and edges; (ii) if $(\rho,\tau)$ is a bIsoInj, then $\rho$ is an injection from F to $\tilde{F}$ on both vertices and edges.

Proof. We first prove that $\rho$ is surjective if $(\rho, \tau)$ is a bIsoSurj. We will only prove that $\rho$ is surjective on edges, as proving that $\rho$ is surjective on vertices is almost the same. For any $\{x, y\} \in E_{\tilde{F}}$ , by Definition C.1(b) we can pick $\tilde{t} \in V_{\tilde{T}}$ such that $\{x, y\} \in \beta_{\tilde{T}}(\tilde{t})$ . Since $\tau$ is surjective, there exists $t \in V_T$ such that $\tau(t) = \tilde{t}$ . By definition of bag isomorphism, there exists $u, v \in \beta_T(t)$ such that $\rho(u) = x$ , $\rho(v) = y$ , and $F[\{u, v\}] \simeq \tilde{F}[\{x, y\}]$ . Therefore, $\{u, v\} \in E_F$ .

We next prove that $\rho$ is injective if $(\rho, \tau)$ is a bIsoInj. Pick any $u \in V_F$ . It suffices to prove that $\rho(u) = \rho(v)$ iff $u = v$ for any $v \in V_F$ . If the result does not hold, consider two cases:

- There exists $v \in V_F$ , $v \neq u$ such that $\rho(u) = \rho(v)$ and $\{u, v\}$ are in the same bag of $F$ . This contradicts the definition of bag isomorphism.   
- For all $v \in V_F$ such that $v \neq u$ and $\rho(u) = \rho(v)$ , $\{u, v\}$ are not in the same bag of $F$ . By Proposition C.5, $T\left[\bigcup_{v \in V_F: \rho(u) = \rho(v)} B_T(v)\right]$ is disconnected. We can thus pick a path $P$ in $T$ such that the endpoints $t_1$ and $t_2$ are in different connected components of $T\left[\bigcup_{v \in V_F: \rho(u) = \rho(v)} B_T(v)\right]$ . This implies that there is tree node $t_3$ in $P$ such that $\beta_T(t_3) \cap \{v \in V_F : \rho(u) = \rho(v)\} = \emptyset$ . Consequently, $\rho(u) \in \rho(\beta_T(t_1)) = \beta_{\tilde{T}}(\tau(t_1))$ , $\rho(u) \in \rho(\beta_T(t_2)) = \beta_{\tilde{T}}(\tau(t_2))$ , but $\rho(u) \notin \beta_{\tilde{T}}(\tau(t_3))$ . On the other hand, since $\tau$ is injective, $\tilde{T}\left[\tau(V_P)\right]$ is also a path and $\tau(t_3)$ is on the path between $\tau(t_1)$ and $\tau(t_2)$ in $\tilde{T}$ . This contradicts the definition of tree decomposition (Definition C.1(c)).

Combining the two cases concludes the proof.

![](images/364329e6ce02236677442803cb927318cb82a856c28d637ff975f01d5c1a3578.jpg)

Lemma C.25. Let $M \in \{\mathrm{Sub}, \mathrm{L}, \mathrm{LF}, \mathrm{F}\}$ be any model. For any tree-decomposed graphs $(F, T^r), (\tilde{F}, \tilde{T}^s) \in \mathcal{S}^M$ ,

$$
\mathsf {b l s o H o m} ((F, T ^ {r}), (\tilde {F}, \tilde {T} ^ {s})) = \sum_ {(\widehat {F}, \widehat {T} ^ {t}) \in \mathcal {S} ^ {M}} \frac {\mathsf {b l s o S u r j} ((F , T ^ {r}) , (\widehat {F} , \widehat {T} ^ {t})) \cdot \mathsf {b l s o I n j} ((\widehat {F} , \widehat {T} ^ {t}) , (\tilde {F} , \tilde {T} ^ {s}))}{\mathsf {a u t} (\widehat {F} , \widehat {T} ^ {t})},
$$

where $\operatorname{aut}(\widehat{F},\widehat{T}^{t})$ denotes the number of automorphisms of $(\widehat{F},\widehat{T}^{t})$ . Here, the summation ranges over all non-isomorphic (tree-decomposed) graphs in $S^{M}$ and is well-defined as there are only a finite number of graphs making the value in the summation non-zero.

Proof. We define the following set of three-tuples:

$$
\begin{array}{l} S = \left\{\left((\widehat {F}, \widehat {T} ^ {t}), (\rho^ {\mathsf {S}}, \tau^ {\mathsf {S}}), (\rho^ {\mathsf {I}}, \tau^ {\mathsf {I}})\right): (\widehat {F}, \widehat {T} ^ {t}) \in \mathcal {S} ^ {M}, \right. \\ (\rho^ {\mathsf {S}}, \tau^ {\mathsf {S}}) \in \mathsf {B l s o S u r j} ((F, T ^ {r}), (\widehat {F}, \widehat {T} ^ {t})), (\rho^ {\mathsf {I}}, \tau^ {\mathsf {I}}) \in \mathsf {B l s o I n j} ((\widehat {F}, \widehat {T} ^ {t}), (\tilde {F}, \tilde {T} ^ {s})) \Biggr \}. \\ \end{array}
$$

Define a mappings $\sigma$ such that

$$
\sigma \left((\widehat {F}, \widehat {T} ^ {t}), (\rho^ {\mathsf {S}}, \tau^ {\mathsf {S}}), (\rho^ {\mathsf {I}}, \tau^ {\mathsf {I}})\right) = (\rho^ {\mathsf {I}} \circ \rho^ {\mathsf {S}}, \tau^ {\mathsf {I}} \circ \tau^ {\mathsf {S}})
$$

for all $\left((\widehat{F},\widehat{T}^t),(\rho^{\mathsf{S}},\tau^{\mathsf{S}}),(\rho^{\mathsf{l}},\tau^{\mathsf{l}})\right)\in S$ . It suffices to prove the following three statements:

1. $\sigma$ is a mapping from $S$ to $\mathsf{B}\mathsf{l}\mathsf{o}\mathsf{o}\mathsf{H}\mathsf{o}\mathsf{m}((F,T^r),(\tilde{F},\tilde{T}^s));$   
2. $\sigma$ is surjective;   
3. $\sigma\left((\widehat{F}_1,\widehat{T}_1^{t_1}),(\rho_1^{\mathsf{S}},\tau_1^{\mathsf{S}}),(\rho_1^{\mathsf{I}},\tau_1^{\mathsf{I}})\right) = \sigma\left((\widehat{F}_2,\widehat{T}_2^{t_2}),(\rho_2^{\mathsf{S}},\tau_2^{\mathsf{S}}),(\rho_2^{\mathsf{I}},\tau_2^{\mathsf{I}})\right)$ iff there exists an isomorphism $(\widehat{\rho},\widehat{\tau})$ from $(\widehat{F}_1,\widehat{T}_1^{t_1})$ to $(\widehat{F}_2,\widehat{T}_2^{t_2})$ such that $\widehat{\rho}\circ\rho_{1}^{\mathsf{S}}=\rho_{2}^{\mathsf{S}},\widehat{\tau}\circ\tau_{1}^{\mathsf{S}}=\tau_{2}^{\mathsf{S}},\rho_{1}^{\mathsf{I}}=\rho_{2}^{\mathsf{I}}\circ\widehat{\rho},$ $\tau_{1}^{\mathsf{I}}=\tau_{2}^{\mathsf{I}}\circ\widehat{\tau}$ .

We will prove these statements one by one.

1. We first prove that $\sigma$ is a mapping from $S$ to BIsoHom $(F,T^{r}),(\tilde{F},\tilde{T}^{s}))$ . This simply follows from the fact that both bIsoSurj and bIsoInj are bIsoHom, and the composition of two bIsoHoms are still a bIsoHom.   
2. We next prove that $\sigma$ is surjective. Given $(\rho^{\mathsf{H}},\tau^{\mathsf{H}})\in \mathrm{BlsoHom}((F,T^{r}),(\tilde{F},\tilde{T}^{s}))$ , we define $(\widehat{F},\widehat{T}^{t}),(\rho^{\mathsf{S}},\tau^{\mathsf{S}})$ , and $(\rho^{\mathsf{I}},\tau^{\mathsf{I}})$ as follows:

a) Let $\widehat{F} = \tilde{F}[\rho^{\mathsf{H}}(V_F)]$ and $\widehat{T}^t = \tilde{T}^s [\tau^{\mathsf{H}}(V_T)]$ . We clearly have $(\widehat{F},\widehat{T}^t)\in \mathcal{S}^M$ .   
b) Let $\rho^{\mathsf{S}} = \rho^{\mathsf{H}}$ and $\tau^{\mathsf{S}} = \tau^{\mathsf{H}}$ . Obviously, $(\rho^{\mathsf{S}},\tau^{\mathsf{S}})$ is a bIsoSurj from $(F,T^{r})$ to $(\widehat{F},\widehat{T}^{t})$ .   
c) Define identity mappings $\rho^1 (u) = u$ for all $u\in V_{\widehat{F}}$ and $\tau^1 (t) = t$ for all $t\in V_{\widehat{T}}$ . Obviously, $(\rho^{\mathrm{l}},\tau^{\mathrm{l}})$ is a bIsoInj from $(\widehat{F},\widehat{T}^{t})$ to $(\tilde{F},\tilde{T}^{s})$ .

We clearly have $\rho^{H} = \rho^{I} \circ \rho^{S}$ and $\tau^{H} = \tau^{I} \circ \tau^{S}$ . Thus, $\sigma$ is a surjection.

3. We finally prove the aforementioned item 3. It suffices to prove only one direction, namely, $\sigma\left((\widehat{F}_1,\widehat{T}_1^{t_1}),(\rho_1^{\mathsf{S}},\tau_1^{\mathsf{S}}),(\rho_1^{\mathsf{l}},\tau_1^{\mathsf{l}})\right) = \sigma\left((\widehat{F}_2,\widehat{T}_2^{t_2}),(\rho_2^{\mathsf{S}},\tau_2^{\mathsf{S}}),(\rho_2^{\mathsf{l}},\tau_2^{\mathsf{l}})\right)$ implies that there exists an isomorphism $(\widehat{\rho},\widehat{\tau})$ from $(\widehat{F}_1,\widehat{T}_1^{t_1})$ to $(\widehat{F}_2,\widehat{T}_2^{t_2})$ such that $\widehat{\rho}\circ\rho_{1}^{\mathsf{S}} = \rho_{2}^{\mathsf{S}},\widehat{\tau}\circ\tau_{1}^{\mathsf{S}} = \tau_{2}^{\mathsf{S}},$ $\rho_1^{\mathsf{l}} = \rho_2^{\mathsf{l}}\circ \widehat{\rho},\tau_1^{\mathsf{l}} = \tau_2^{\mathsf{l}}\circ \widehat{\tau}.$

a) We first prove that $\widehat{F}_1\simeq \widehat{F}_2$ and $\widehat{T}_1^{t_1}\simeq \widehat{T}_2^{t_2}$ .

For any $u, v \in V_F$ , if $\rho_1^{\mathsf{S}}(u) \neq \rho_1^{\mathsf{S}}(v)$ , then $\rho_1^{\mathsf{I}}(\rho_1^{\mathsf{S}}(u)) \neq \rho_1^{\mathsf{I}}(\rho_1^{\mathsf{S}}(v))$ since $\rho_1^{\mathsf{I}}$ is an injection (by Proposition C.24). Therefore, $\rho_2^{\mathsf{I}}(\rho_2^{\mathsf{S}}(u)) \neq \rho_2^{\mathsf{I}}(\rho_2^{\mathsf{S}}(v))$ , and thus $\rho_2^{\mathsf{S}}(u) \neq \rho_2^{\mathsf{S}}(v)$ . By symmetry, we also have that $\rho_2^{\mathsf{S}}(u) \neq \rho_2^{\mathsf{S}}(v)$ implies $\rho_1^{\mathsf{S}}(u) \neq \rho_1^{\mathsf{S}}(v)$ . This proves that $\rho_1^{\mathsf{S}}(u) = \rho_1^{\mathsf{S}}(v)$ iff $\rho_2^{\mathsf{S}}(u) = \rho_2^{\mathsf{S}}(v)$ .

For any $u, v \in V_F$ , if $\{\rho_1^S(u), \rho_1^S(v)\} \in E_{\widehat{F}_1}$ , then $\{\rho_1^I(\rho_1^S(u)), \rho_1^I(\rho_1^S(v))\} \in E_{\widetilde{F}}$ since $\rho_1^I$ is a homomorphism. Therefore, $\{\rho_2^I(\rho_2^S(u)), \rho_2^I(\rho_2^S(v))\} \in E_{\widetilde{F}}$ . This implies that $\rho_2^I(\rho_2^S(u))$ and $\rho_2^I(\rho_2^S(v))$ are in the same bag of $\tilde{T}^s$ . Therefore, there are vertices $x, y \in V_{\widehat{F}_2}$ in the same bag of $\widehat{T}_2^{s_2}$ such that $\rho_2^I(x) = \rho_2^I(\rho_2^S(u))$ and $\rho_2^I(y) = \rho_2^I(\rho_2^S(v))$ , and by definition of bag isomorphism we have $\{x, y\} \in E_{\widehat{F}_2}$ . Since $\rho_2$ is injective, $x = \rho_2^S(u)$ and $y = \rho_2^S(v)$ , namely, $\{\rho_2^S(u), \rho_2^S(v)\} \in E_{\widehat{F}_2}$ . By symmetry, we can prove that $\{\rho_1^S(u), \rho_1^S(v)\} \in E_{\widehat{F}_1}$ iff $\{\rho_2^S(u), \rho_2^S(v)\} \in E_{\widehat{F}_2}$ .

Finally, noting that $\rho_1^{\mathsf{S}}$ and $\rho_2^{\mathsf{S}}$ are surjective (by Proposition C.24) and $\ell_{\widehat{F}_1}(\rho_1^{\mathsf{S}}(u)) = \ell_F(u) = \ell_{\widehat{F}_2}(\rho_2^{\mathsf{S}}(u))$ for all $u \in V_F$ , we obtain that $\widehat{F}_1 \simeq \widehat{F}_2$ . Following the same procedure, we can prove that $\widehat{T}_1^{t_1} \simeq \widehat{T}_2^{t_2}$ .

b) Consequently, there exist isomorphisms $\widehat{\rho}$ and $\widehat{\tau}$ such that $\widehat{\rho} \circ \rho_1^S = \rho_2^S$ , $\widehat{\tau} \circ \tau_1^S = \tau_2^S$ . For any node $q \in V_T$ ,

$$
\widehat {\rho} (\beta_ {\widehat {T} _ {1}} (\tau_ {1} ^ {\mathsf {S}} (q))) = \widehat {\rho} (\rho_ {1} ^ {\mathsf {S}} (\beta_ {T} (q))) = \rho_ {2} ^ {\mathsf {S}} (\beta_ {T} (q))) = \beta_ {\widehat {T} _ {2}} (\tau_ {2} ^ {\mathsf {S}} (q)) = \beta_ {\widehat {T} _ {2}} (\widehat {\tau} (\tau_ {1} ^ {\mathsf {S}} (q))).
$$

Since $\tau_1^S$ is surjective, $\tau_1^S (q)$ ranges over all nodes in $\widehat{T}_1^{s_1}$ when $q$ ranges over $V_{T}$ . We thus conclude that $(\rho ,\tau)$ is an isomorphism from $(\widehat{F}_1,\widehat{T}_1^{t_1})$ to $(\widehat{F}_2,\widehat{T}_2^{t_2})$ (see Definition C.8).

c) We finally prove that $\rho_1^{\mathsf{l}} = \rho_2^{\mathsf{l}}\circ \widehat{\rho}$ and $\tau_1^{\mathsf{l}} = \tau_2^{\mathsf{l}}\circ \widehat{\tau}$ . Pick any $u\in V_F$ , we have $\rho_2^{\mathsf{l}}(\widehat{\rho} (\rho_1^{\mathsf{S}}(u))) = \rho_2^{\mathsf{l}}(\rho_2^{\mathsf{S}}(u)) = \rho_1^{\mathsf{l}}(\rho_1^{\mathsf{S}}(u))$ . Since $\rho_{1}^{\mathsf{S}}$ is surjective, $\rho_1^{\mathsf{S}}(u)$ ranges over all nodes in $\widehat{F}_1$ when $u$ ranges over $V_{F}$ . This proves that $\rho_1^{\mathsf{l}} = \rho_2^{\mathsf{l}}\circ \widehat{\rho}$ . Following the same procedure, we can prove that $\tau_1^{\mathsf{l}} = \tau_2^{\mathsf{l}}\circ \widehat{\tau}$ .

Combining the above three items concludes the proof.

![](images/2c363b7f4c30baf190b496a4a88db627c65ea10677b711930d0c51fc336999c8.jpg)

In the following, we further define two technical concepts that will be used to describe the next result. We note that these technical concepts also appeared in Dell et al. (2018).

Definition C.26. A tree-decomposed graph $(\tilde{F},\tilde{T}^{s})$ is called a bag extension of another tree-decomposed graph $(F,T^{r})$ if there is a graph H and a mapping $(\rho,\tau)$ such that F is a subgraph of H and $(\rho,\tau)$ is an isomorphism from the tree-decomposed graph $(H,T^{r})$ to $(\tilde{F},\tilde{T}^{s})$ . Define $\mathtt{BExt}\left((F,T^{r}),(\tilde{F},\tilde{T}^{s})\right)$ to be the set of all mappings $(\rho,\tau)$ that satisfies the above conditions, and define $\mathtt{bExt}\left((F,T^{r}),(\tilde{F},\tilde{T}^{s})\right)=\left|\mathtt{BExt}\left((F,T^{r}),(\tilde{F},\tilde{T}^{s})\right)\right|$ .

Remark C.27. In other words, a bag extension of a tree-decomposed graph $(F,T^{r})$ can be obtained by adding an arbitrary number of edges to F while ensuring that each added edge is contained in a tree node in $T^{r}$ . A immediate fact is that $(\rho,\tau)$ is a homomorphism from $(F,T^{r})$ to $(\tilde{F},\tilde{T}^{s})$ .

Definition C.28. Given a tree-decomposed graph $(F,T^{r})$ and a graph G, a bag-strong homomorphism from $(F,T^{r})$ to G (abbreviated as “bStrHom”) is a homomorphism f from F to G such that, for all $t\in V_{T}$ , f is a strong homomorphism from $F[\beta_{T}(t)]$ to $G[f(\beta_{T}(t))]$ , i.e., $\{u,v\}\in E_{F[\beta_{T}(t)]}$ iff $\{f(u),f(v)\}\in E_{G[f(\beta_{T}(t))]}$ . Denote BStrHom $((F,T^{r}),G)$ to be the set of all bStrHom from $(F,T^{r})$ to G, and denote bStrHom $((F,T^{r}),G)=|BStrHom((F,T^{r}),G)|$ .

The following equation is straightforward:

Lemma C.29. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. For any graph $G$ and tree-decomposed graph $(F, T^r) \in S^M$ ,

$$
\hom (F, G) = \sum_ {(\tilde {F}, \tilde {T} ^ {s}) \in \mathcal {S} ^ {M}} \frac {\mathrm{bExt} ((F , T ^ {r}) , (\tilde {F} , \tilde {T} ^ {s})) \cdot \mathrm{bStrHom} ((\tilde {F} , \tilde {T} ^ {s}) , G)}{\mathrm{aut} (\tilde {F} , \tilde {T} ^ {s})},
$$

where $\operatorname{aut}(\tilde{F},\tilde{T}^{s})$ denotes the number of automorphisms of $(\tilde{F},\tilde{T}^{s})$ . Here, the summation ranges over all non-isomorphic (tree-decomposed) graphs in $S^{M}$ and is well-defined as there are only a finite number of graphs making the value in the summation non-zero.

Proof. The proof has a similar structure to the previous lemma. We define the following set of three-tuples:

$$
S = \left\{\left((\tilde {F}, \tilde {T} ^ {s}), (\rho , \tau), g\right): (\tilde {F}, \tilde {T} ^ {s}) \in \mathcal {S} ^ {M}, (\rho , \tau) \in \mathrm{BExt} \Big ((F, T ^ {r}), (\tilde {F}, \tilde {T} ^ {s}) \Big), g \in \mathrm{BStrHom} ((\tilde {F}, \tilde {T} ^ {s}), G) \right\}.
$$

Define a mappings $\sigma$ such that $\sigma\left((\tilde{F},\tilde{T}^{s}),(\rho,\tau),g\right)=g\circ\rho$ for all $\left((\tilde{F},\tilde{T}^{s}),(\rho,\tau),g\right)\in S$ . It is straightforward to see that $\sigma$ is a mapping from $S$ to $\mathsf{B}\mathsf{l}\mathsf{o}\mathsf{o}\mathsf{H}\mathsf{o}\mathsf{m}((F,T^{r}),(\tilde{F},\tilde{T}^{s}))$ , namely, $g\circ\rho$ is a homomorphism from $F$ to $G$ for all $\left((\tilde{F},\tilde{T}^{s}),(\rho,\tau),g\right)\in S$ .

We then prove that $\sigma$ is surjective. Given $h \in \operatorname{Hom}(F, G)$ , define $(\tilde{F}, \tilde{T}^s), (\rho, \tau)$ , and $g$ as follows:

a) Define $\tilde{F}$ be the graph obtained from $F$ by adding edges $\{\{u,v\}:\exists t\in V_T$ s.t. $u,v\in \beta_T(t),\{h(u),h(v)\} \in E_G\}$ , and let $\tilde{T}^s = T^r$ . Clearly, $(\tilde{F},\tilde{T}^s)$ is a bag extension of $(F,T^r)$ .   
b) Define identity mappings $\rho(u) = u$ for all $u \in V_F$ and $\tau(t) = t$ for all $t \in V_T$ . Clearly, $(\rho, \tau) \in \mathsf{BExt}((F, T^r), (\tilde{F}, \tilde{T}^s))$ .   
c) Let $g = h$ . It is easy to see that $g$ is a strong homomorphism from $\tilde{F}[\beta_{\tilde{T}}(t)]$ to $G[g(\beta_{\tilde{T}}(t))]$ for each $t \in V_{\tilde{T}}$ . Thus, $g \in \mathrm{BStrHom}((\tilde{F}, \tilde{T}^s), G)$ .

Noting that $h = g = g \circ \rho$ , we have proved that $\sigma$ is a surjection.

We finally prove that $\sigma\left((\tilde{F}_1,\tilde{T}_1^{s_1}),(\rho_1,\tau_1),g_1\right) = \sigma\left((\tilde{F}_2,\tilde{T}_2^{s_2}),(\rho_2,\tau_2),g_2\right)$ implies that there exists an isomorphism $(\tilde{\rho},\tilde{\tau})$ from $(\tilde{F}_1,\tilde{T}_1^{s_1})$ to $(\tilde{F}_2,\tilde{T}_2^{s_2})$ such that $\tilde{\rho}\circ\rho_{1}=\rho_{2},\tilde{\tau}\circ\tau_{1}=\tau_{2}, g_{1}=g_{2}\circ\tilde{\rho}$ . We first prove that $\tilde{F}_1\simeq\tilde{F}_2$ and $\tilde{T}_1^{s_1}\simeq\tilde{T}_2^{s_2}$ . (i) For any $u,v\in V_F$ , we obviously have $\rho_1(u)=\rho_1(v)$ iff $u=v$ iff $\rho_2(u)=\rho_2(v)$ . (ii) Let $i\in\{1,2\}$ . For any $u,v\in V_F$ , $\{\rho_i(u),\rho_i(v)\}\in E_{\tilde{F}_i}$ iff $\{u,v\}\in E_F$ or $u,v$ are in the same bag of $T^r$ and $\{g_i(\rho_i(u)),g(\rho_i(v))\}\in E_G$ . Since $g_1\circ\rho_1=g_2\circ\rho_2$ , we have $\{\rho_1(u),\rho_1(v)\}\in E_{\tilde{F}_1}$ iff $\{\rho_2(u),\rho_2(v)\}\in E_{\tilde{F}_2}$ . (iii) Finally, noting that $\rho_1$ and $\rho_2$ are bijective and $\ell_{\tilde{F}_1}(\rho_1(u))=\ell_F(u)=\ell_{\tilde{F}_2}(\rho_2(u))$ for all $u\in V_F$ , we obtain that $\tilde{F}_1\simeq\tilde{F}_2$ . On the other hand, $\tilde{T}_1^{s_1}\simeq\tilde{T}_2^{s_2}$ trivially holds. The remaining procedure is almost the same as in the previous lemma.

We next show that the mapping bStrHom can be further decomposed as shown in Lemma C.31. We need an auxiliary concept:

Definition C.30. Given two tree-decomposed graphs $(F,T^{r})$ and $(\tilde{F},\tilde{T}^{s})$ , a homomorphism $(\rho,\tau)$ from $(F,T^{r})$ to $(\tilde{F},\tilde{T}^{s})$ is called bag-strong surjective (abbreviated as “bStrSurj”) if $\rho$ is a bag-strong homomorphism from $(F,T^{r})$ to $\tilde{F}$ and is surjective on both vertices and edges, and $\tau$ is an isomorphism from $T^{r}$ to $\tilde{T}^{s}$ . Denote BStrSurj( $(F,T^{r}),(\tilde{F},\tilde{T}^{s})$ ) to be the set of all bStrSurj from $(F,T^{r})$ to $(\tilde{F},\tilde{T}^{s})$ , and denote bStrSurj( $(F,T^{r}),(\tilde{F},\tilde{T}^{s})$ ) = |BStrSurj( $(F,T^{r}),(\tilde{F},\tilde{T}^{s})$ )|.

Lemma C.31. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. For any graph $G$ and tree-decomposed graph $(F, T^r) \in S^M$ ,

$$
\mathsf {b S t r H o m} ((F, T ^ {r}), G) = \sum_ {(\tilde {F}, \tilde {T} ^ {s}) \in \mathcal {S} ^ {M}} \frac {\mathsf {b S t r S u r j} ((F , T ^ {r}) , (\tilde {F} , \tilde {T} ^ {s})) \cdot \mathsf {b l s o} ((\tilde {F} , \tilde {T} ^ {s}) , G)}{\mathsf {a u t} (\tilde {F} , \tilde {T} ^ {s})},
$$

where $\operatorname{aut}(\tilde{F},\tilde{T}^{s})$ denotes the number of automorphisms of $(\tilde{F},\tilde{T}^{s})$ . Here, the summation ranges over all non-isomorphic (tree-decomposed) graphs in $S^{M}$ and is well-defined as there are only a finite number of graphs making the value in the summation non-zero.

Proof. The proof has a similar structure to the previous lemma. We define the following set of three-tuples:

$$
S = \left\{\left((\tilde {F}, \tilde {T} ^ {s}), (\rho , \tau), g\right): (\tilde {F}, \tilde {T} ^ {s}) \in \mathcal {S} ^ {M}, (\rho , \tau) \in \mathrm{BStrSurj} \left((F, T ^ {r}), (\tilde {F}, \tilde {T} ^ {s})\right), g \in \mathrm{Blso} ((\tilde {F}, \tilde {T} ^ {s}), G) \right\}.
$$

Define a mappings $\sigma$ such that $\sigma\left((\tilde{F},\tilde{T}^s),(\rho,\tau),g\right) = g\circ \rho$ for all $\left((\tilde{F},\tilde{T}^s),(\rho,\tau),g\right)\in S$ . It suffices to prove the following three statements:

1. $\sigma$ is a mapping from $S$ to BStrHom(( $F, T^r$ ), $G$ );   
2. $\sigma$ is surjective;   
3. $\sigma\left((\tilde{F}_1,\tilde{T}_1^{s_1}),(\rho_1,\tau_1),g_1\right) = \sigma\left((\tilde{F}_2,\tilde{T}_2^{s_2}),(\rho_2,\tau_2),g_2\right)$ iff there exists an isomorphism $(\tilde{\rho},\tilde{\tau})$ from $(\tilde{F}_1,\tilde{T}_1^{s_1})$ to $(\tilde{F}_2,\tilde{T}_2^{s_2})$ such that $\tilde{\rho}\circ\rho_{1}=\rho_{2},\tilde{\tau}\circ\tau_{1}=\tau_{2},g_{1}=g_{2}\circ\tilde{\rho}$ .

We will prove these statements one by one.

1. We first prove that $\sigma$ is a mapping from $S$ to $\mathsf{BStrHom}((F,T^r),G)$ . Pick any $\left((\tilde{F},\tilde{T}^s),(\rho ,\tau),g\right)\in S$ . Pick any $t\in V_T$ and $u,v\in \beta_T(t)$ . Then, $\{u,v\} \in E_F$ iff $\{\rho (u),\rho (v)\} \in E_{\tilde{F}}$ (since $\rho$ is a strong homomorphism from $F[\beta_T(t)]$ to $\tilde{F} [\rho (\beta_T(t))]$ ). Also, $\rho (u),\rho (v)\in \beta_{\tilde{T}}(\tau (t))$ are in the same bag. Similarly, $\{\rho (u),\rho (v)\} \in E_{\tilde{F}}$ iff $\{g(\rho (u)),g(\rho (v))\} \in E_G$ (since $g$ is a bIso). Thus, $g\circ \rho$ is a bag-strong homomorphism.

2. We next prove that $\sigma$ is surjective. Given $h \in \mathrm{BStrHom}((F, T^r), G)$ , define $(\tilde{F}, \tilde{T}^s), (\rho, \tau)$ , and $g$ as follows. First define a relation $\sim$ on set $V_F$ such that $u \sim v$ iff the following hold:

$$
\mathrm{a)} h (u) = h (v);
$$

b) There exists a path $P$ in $T^r$ with endpoints $t_1, t_2 \in V_T$ such that $u \in \beta_T(t_1), v \in \beta_T(t_2)$ , and all node $t$ on path $P$ satisfies that $h(u) \in h(\beta_T(t))$ .

It is easy to see that $\sim$ is an equivalence relation on $V_{F}$ . We can thus define a mapping $\rho$ that respects the relation, i.e., $\rho(u) = \rho(v)$ iff $u \sim v$ for all $u, v \in V_{F}$ . Moreover, for any edge $\{u, v\} \in E_{F}$ , $\rho(u) \neq \rho(v)$ (since h is a homomorphism and $h(u) \neq h(v)$ ). This implies that we can define $\tilde{F}$ to be the homomorphic image of F such that $\rho$ is the surjective homomorphism on both vertices and edges.

We then define the mapping $g: V_{\tilde{F}} \to V_G$ such that $g(\rho(u)) = h(u)$ for all $u \in V_F$ . Note that $g$ is well-defined since $\rho(u) = \rho(v)$ implies $h(u) = h(v)$ for all $u, v \in V_F$ , and $\rho: V_F \to V_{\tilde{F}}$ is surjective. It follows that $h = g \circ \rho$ . To prove that $g$ is a homomorphism, note that for all $\{x, y\} \in E_{\tilde{F}}$ , there exists an edge $\{u, v\} \in E_F$ with $\rho(u) = x$ , $\rho(v) = y$ , which implies that $\{h(u), h(v)\} \in E_G$ (since $h$ is a homomorphism), namely, $\{g(x), g(y)\} \in E_G$ .

We next define tree $\tilde{T}^s = (V_T, E_T, \beta_{\tilde{T}})$ , $s = t$ , and identity mapping $\tau$ so that $\tau$ is an isomorphism from $T_r$ to $T_s$ (ignoring the labels). Set $\beta_{\tilde{T}}(t) = \rho(\beta_T(t))$ for all $t \in V_T$ . We will prove that $(\tilde{F}, \tilde{T}^s) \in S^M$ is a valid tree decomposition. It suffices to prove that Definition C.1(c) holds. Pick any vertex $x \in V_{\tilde{F}}$ and tree node $t_1, t_2 \in B_{\tilde{T}}(x)$ . Then, there exists $u \in \beta_T(t_1), v \in \beta_T(t_2)$ such that $\rho(u) = x$ , $\rho(v) = x$ . Therefore, $u \sim v$ . As such, there exists a path $P$ in $T^r$ such that all node $t$ on $P$ satisfies that there exists $w \in \beta_T(t)$ with $h(w) = h(u)$ , namely, $w \sim u$ . Consequently, $\rho(u) \in \beta_{\tilde{T}}(t)$ , implying that $\tilde{T}^s[B_{\tilde{T}}(x)]$ is connected. This proves that $(\tilde{F}, \tilde{T}^s) \in S^M$ . Also, $(\rho, \tau)$ is clearly a homomorphism from $(F, T^r)$ to $(\tilde{F}, \tilde{T}^s)$ according to Definition C.20.

It remains to prove that $\rho$ is a bag-strong homomorphism and $g$ is a bIso. Pick any $t \in V_T$ and $u, v \in \beta_T(t)$ . If $\{u, v\} \notin E_F$ , then $\{h(u), h(v)\} \notin E_G$ (since $h$ is a bag-strong homomorphism). Therefore, $\{\rho(u), \rho(v)\} \notin E_{\tilde{F}}$ (since $g$ is a homomorphism), namely, $\rho$ is a bag-strong homomorphism. Since $\rho$ is surjective, $\{\rho(u), \rho(v)\}$ ranges over all vertices in the same bag of $\tilde{T}^s$ when $t \in V_T$ and $u, v \in \beta_T(t)$ are arbitrary. Therefore, $g$ is a bIso because $\{\rho(u), \rho(v)\} \notin E_{\tilde{F}}$ iff $\{h(u), h(v)\} \notin E_G$ .

3. We finally prove that $\sigma\left((\tilde{F}_1, \tilde{T}_1^{s_1}), (\rho_1, \tau_1), g_1\right) = \sigma\left((\tilde{F}_2, \tilde{T}_2^{s_2}), (\rho_2, \tau_2), g_2\right)$ implies there exists an isomorphism $(\tilde{\rho}, \tilde{\tau})$ from $(\tilde{F}_1, \tilde{T}_1^{s_1})$ to $(\tilde{F}_2, \tilde{T}_2^{s_2})$ such that $\tilde{\rho} \circ \rho_1 = \rho_2$ , $\tilde{\tau} \circ \tau_1 = \tau_2$ , $g_1 = g_2 \circ \tilde{\rho}$ . Let $h = g_1 \circ \rho_1 = g_2 \circ \rho_2$ . Here, we will only prove that $\tilde{F}_1 \simeq \tilde{F}_2$ since the remaining procedure is almost the same as previous proofs. Since both $\tilde{F}_1$ and $\tilde{F}_2$ are homomorphic images of $F$ , it suffices to prove that, for all $u, v \in V_F$ , $\rho_1(u) = \rho_1(v)$ iff the following hold:

a) $h(u) = h(v);$   
b) There exists a path $P$ in $T^r$ with endpoints $t_1, t_2 \in V_T$ such that $u \in \beta_T(t_1)$ , $v \in \beta_T(t_2)$ , and all node $t$ on path $P$ satisfies that $h(u) \in h(\beta_T(t))$ .

On one hand, if $\rho_1(u) = \rho_1(v)$ , we clearly have $h(u) = h(v)$ and there exists $t_1 \in B_T(u), t_2 \in B_T(v)$ such that $t_1, t_2 \in B_{\tilde{T}}(\rho_1(u))$ . Since $\tilde{T}[B_{\tilde{T}}(\rho_1(u))]$ is connected, there is a path $P$ with endpoints $t_1, t_2$ such that all node $t$ on $P$ satisfies $\rho_1(u) \in \rho_1(\beta_T(t))$ and thus $h(u) \in h(\beta_T(t))$ .

On the other hand, if $\rho_{1}(u) \neq \rho_{1}(v)$ but the above items (a) and (b) hold, consider two cases:

- $u$ and $v$ are in the same bag of $T$ . Then, $\rho_1(u)$ and $\rho_1(v)$ are in the same bag of $\tilde{T}$ . Since $g_1$ is a bIso, $g_1(\rho_1(u)) \neq g_1(\rho_1(v))$ , which contradicts item (a).   
- $u$ and $v$ are not in the same bag of $T$ . Then, there exist two adjacent nodes $t_1, t_2$ on $P$ such that $\rho_1(u) \in \beta_{\tilde{T}}(t_1)$ , $\rho_1(u) \notin \beta_{\tilde{T}}(t_2)$ . By Definition C.4(c), $\beta_{\tilde{T}}(t_2) \subset \beta_{\tilde{T}}(t_1)$ . Also, item (b) implies that there exists $w \in \beta_T(t_2)$ such that $h(w) = h(u)$ . Therefore, $\rho_1(u)$ and $\rho_1(w)$ are two different nodes in $\beta_{\tilde{T}}(t_1)$ with $g_1(\rho_1(u)) = h(u) = h(w) = g_1(\rho_1(w))$ . This contradicts the condition that $g_1$ is a bIso.

This yields the desired result that $\tilde{F}_{1} \simeq \tilde{F}_{2}$ .

Combining the above three items concludes the proof.

![](images/764544c6c0b9efe3213015eb2a8581c9a14dcc9b2d067a74a844bdf67c1f61ae.jpg)

Let $M \in \{Sub, L, LF, F\}$ be any model. We can list all non-isomorphic tree-decomposed (labeled) graphs in $S^{M}$ into an infinite sequence $(F_{1}, T_{1}^{r_{1}}), (F_{2}, T_{2}^{r_{2}}), \cdots$ . Consider two types of ordering:

- Ordered by the size of the graphs $F_{i}$ . We denote this ordering as 1st. This ordering requires that for any $i < j$ , either $|V_{F_i}| < |V_{F_j}|$ or $|V_{F_i}| = |V_{F_j}|$ and $|E_{F_i}| \geq |E_{F_j}|$ . Note that when two graphs have an equal number of vertices, we place the graph with more edges to the front. When the number of edges is also the same, they can be arranged in any fixed order.   
- Ordered by the size of the trees $T_{i}$ . We denote this ordering as 2nd. This ordering requires that $|V_{T_i}| \leq |V_{T_j}|$ for any $i < j$ . When the number of tree nodes is the same, they can be arranged in any fixed order.

Without loss of generality, we assume that the labels of each graph $F_{i}$ are integers in the range of $[1, |V_{F_{i}}|]$ . Note that using integer labels of a bounded range is already sufficient to represent all non-isomorphic labeled graphs up to a bijective label transformation. This ensures that the index i is countable.

For the 1st ordering, define the following notations:

a) Let $f: \mathcal{S}^M \times \mathcal{S}^M \to \mathbb{N}$ be any mapping. Define the associated (infinite) matrix $A^{f,M,1\mathrm{st}} \in \mathbb{N}^{\mathbb{N}_+ \times \mathbb{N}_+}$ such that $A_{i,j}^{f,M,1\mathrm{st}} = f((F_i, T_i^{r_i}), (F_j, T_j^{r_j}))$ .   
b) Let $g: \mathcal{S}^M \times \mathcal{G} \to \mathbb{N}$ be any mapping. Given a graph $G \in \mathcal{G}$ , define the (infinite) vector $\pmb{p}_G^{g,M,1\mathrm{st}} \in \mathbb{N}^{\mathbb{N}_+}$ such that $p_{G,i}^{f,M,1\mathrm{st}} = g((F_i,T_i^{r_i}),G)$ .   
c) Let $h: \mathcal{G} \times \mathcal{G} \to \mathbb{N}$ be any mapping. Given a graph $G \in \mathcal{G}$ , define the (infinite) vector $\pmb{p}_G^{h,M,1\mathrm{st}} \in \mathbb{N}^{\mathbb{N}_+}$ such that $p_{G,i}^{h,M,1\mathrm{st}} = h(F_i,G)$ .

We can similarly define $A^{f,M,2\mathrm{nd}}$ , $\pmb{p}_G^{g,M,2\mathrm{nd}}$ , $\pmb{p}_G^{h,M,2\mathrm{nd}}$ for the ordering 2nd.

Corollary C.32. Let $M \in \{Sub, L, LF, F\}$ be any model and G, H be two graphs. Then, $\hom(F, G) = \hom(F, H)$ for all $(F, T^{r}) \in \mathcal{S}^{M}$ iff $\text{blso}((F, T^{r}), G) = \text{blso}((F, T^{r}), H)$ for all $(F, T^{r}) \in \mathcal{S}^{M}$ .

Proof. We separately consider each direction.

1. We first prove that if $\mathsf{blso}((F,T^r),G) = \mathsf{blso}((F,T^r),H)$ for all $(F,T^r)\in S^M$ , then $\hom (F,G) = \hom (F,H)$ for all $(F,T^r)\in S^M$ . According to Lemmas C.29 and C.31, we can rewrite the corresponding equations into matrix forms for any $F\in \mathcal{G}$ :

$$
\boldsymbol {p} _ {F} ^ {\text { hom }, M, 1 \text { st }} = \boldsymbol {A} ^ {\text { bExt }, M, 1 \text { st }} (\boldsymbol {A} ^ {\text { aut }, M, 1 \text { st }}) ^ {- 1} \boldsymbol {p} _ {F} ^ {\text { bStrHom }, M, 1 \text { st }} \tag {19}
$$

$$
\boldsymbol {p} _ {F} ^ {\mathrm{bStrHom}, M, 1 \mathrm{st}} = \boldsymbol {A} ^ {\mathrm{bStrSurj}, M, 1 \mathrm{st}} (\boldsymbol {A} ^ {\mathrm{aut}, M, 1 \mathrm{st}}) ^ {- 1} \boldsymbol {p} _ {F} ^ {\mathrm{blso}, M, 1 \mathrm{st}}. \tag {20}
$$

From the above equations, we immediately obtain that $p_G^{\text{blso},M,1\text{st}} = p_H^{\text{blso},M,1\text{st}}$ implies $p_G^{\text{hom},M,1\text{st}} = p_H^{\text{hom},M,1\text{st}}$ .

2. We next prove that if $\hom(F, G) = \hom(F, H)$ for all $(F, T^r) \in S^M$ , then $\mathsf{blso}((F, T^r), G) = \mathsf{blso}((F, T^r), H)$ for all $(F, T^r) \in S^M$ . This can be seen from the following facts:

a) $A^{\mathrm{aut},M,1\mathrm{st}}$ is a diagonal matrix and all diagonal elements are positive integers.   
b) $A^{bExt,M,1st}$ is a lower triangular matrix and all diagonal elements are positive integers. This is because for any two tree-composed graphs $(F_{i},T_{i}^{r_{i}})$ and $(F_{j},T_{j}^{r_{j}})$ , $\mathsf{bExt}((F_{i},T_{i}^{r_{i}}),(F_{j},T_{j}^{r_{j}}))>0$ only if $|V_{F_{i}}|=|V_{F_{j}}|$ and $|E_{F_{i}}|\leq|E_{F_{j}}|$ .   
c) $A^{\text{bStrSurj},M,1\text{st}}$ is also a lower triangular matrix and all diagonal elements are positive integers. This is because for any two tree-composed graphs $(F_i, T_i^{r_i})$ and $(F_j, T_j^{r_j})$ , bStrSurj $((F_i, T_i^{r_i}), (F_j, T_j^{r_j})) > 0$ only if $|V_{F_i}| > |V_{F_j}|$ or $(|V_{F_i}| = |V_{F_j}|$ and $|E_{F_i}| = |E_{F_j}|)$ .

Therefore, the composition $A^{\text{bExt},M,1\text{st}}(A^{\text{aut},M,1\text{st}})^{-1} A^{\text{bStrSurj},M,1\text{st}}(A^{\text{aut},M,1\text{st}})^{-1}$ is lower triangular and is invertible (although is it an infinite matrix). We thus arrive at the desired conclusion that $p_G^{\text{hom},M,1\text{st}} = p_H^{\text{hom},M,1\text{st}}$ implies $p_G^{\text{blso},M,1\text{st}} = p_H^{\text{blso},M,1\text{st}}$ .

□

Corollary C.33. Let $M \in \{Sub, L, LF, F\}$ be any model and G, H be two graphs. Then, $\text{cnt}^{M}((F, T^{r}), G) = \text{cnt}^{M}((F, T^{r}), H)$ for all $(F, T^{r}) \in \mathcal{S}^{M}$ iff $\text{blso}((F, T^{r}), G) = \text{blso}((F, T^{r}), H)$ for all $(F, T^{r}) \in \mathcal{S}^{M}$ .

Proof. We separately consider each direction.

1. We first prove that if $\operatorname{cnt}^{M}((F,T^{r}),G)=\operatorname{cnt}^{M}((F,T^{r}),H)$ for all $(F,T^{r})\in\mathcal{S}^{M}$ , then $\operatorname{blso}((F,T^{r}),G)=\operatorname{blso}((F,T^{r}),H)$ for all $(F,T^{r})\in\mathcal{S}^{M}$ . According to Lemma C.23, we can rewrite Equation (18) into the matrix form for all $F\in G$ :

$$
\boldsymbol {p} _ {F} ^ {\text { blso }, M, 2 \text { nd }} = \boldsymbol {A} ^ {\text { blsoHom }, M, 2 \text { nd }} \boldsymbol {p} _ {F} ^ {\text { cnt }, M, 2 \text { nd }}. \tag {21}
$$

This immediately obtains that $p_{G}^{cnt,M,2nd} = p_{H}^{cnt,M,2nd}$ implies $p_{G}^{blso,M,2nd} = p_{H}^{blso,M,2nd}$ .

2. We next prove that $\pmb{p}_G^{\mathrm{blso},M,2\mathrm{nd}} = \pmb{p}_H^{\mathrm{blso},M,2\mathrm{nd}}$ implies $\pmb{p}_G^{\mathrm{cnt},M,2\mathrm{nd}} = \pmb{p}_H^{\mathrm{cnt},M,2\mathrm{nd}}$ . According to Lemma C.25, we have

$$
\boldsymbol {A} ^ {\text { b   l   s   o   H   o   m }, M, 2 \text { nd }} = \boldsymbol {A} ^ {\text { b   l   s   o   S   u   r   j }, M, 2 \text { nd }} \left(\boldsymbol {A} ^ {\text { a   u   t }, M, 2 \text { nd }}\right) ^ {- 1} \boldsymbol {A} ^ {\text { b   l   s   o   I   n   j }, M, 2 \text { nd }}. \tag {22}
$$

Moreover, we have the following facts:

a) $A^{\mathrm{aut},M,2\mathrm{nd}}$ is a diagonal matrix and all diagonal elements are positive integers.   
b) $A^{\mathrm{blsolnj},M,2\mathrm{nd}}$ is an upper triangular matrix and all diagonal elements are positive integers. This is because for any two tree-composed graphs $(F_i,T_i^{r_i})$ and $(F_j,T_j^{r_j})$ , $\mathsf{blsolnj}((F_i,T_i^{r_i}),(F_j,T_j^{r_j})) > 0$ only if $|V_{T_i}|\leq |V_{T_j}|$ .   
c) Similarly, $A^{\mathrm{blsoSurj},M,2\mathrm{nd}}$ is a lower triangular matrix and all diagonal elements are positive integers.

Unfortunately, since $A^{\text{blsoln},M,2\text{nd}}$ is an infinite upper triangular matrix, the inverse matrix is not well-defined. Nevertheless, we can use a special property of bIsoHom to complete our proof, namely, blsoHom(( $F_i, T_i^{r_i}$ ), ( $F_j, T_j^{r_j}$ )) > 0 implies that the depth of $T_i^{r_i}$ is equal to the depth of $T_j^{r_j}$ . Therefore, denoting by $A^{\text{dep},d}$ the diagonal matrix where $A_{ii}^{\text{dep},d} = \mathbb{I}[\text{the depth of } T_i^{r_i} \text{ is } d]$ , we have

$$
\boldsymbol {A} ^ {\text { dep }, d} \boldsymbol {p} _ {F} ^ {\text { bIso }, M, 2 \text { nd }} = \boldsymbol {A} ^ {\text { bIsoSurj }, M, 2 \text { nd }} (\boldsymbol {A} ^ {\text { aut }, M, 2 \text { nd }}) ^ {- 1} \boldsymbol {A} ^ {\text { bIsoInj }, M, 2 \text { nd }} \boldsymbol {A} ^ {\text { dep }, d} \boldsymbol {p} _ {F} ^ {\text { cnt }, M, 2 \text { nd }} \tag {23}
$$

for all $F \in \mathcal{G}$ . Fix any integer $d \geq 0$ , and assume that $A^{\mathrm{dep},d}p_G^{\mathrm{blso},M,2\mathrm{nd}} = A^{\mathrm{dep},d}p_H^{\mathrm{blso},M,2\mathrm{nd}}$ . We will prove that $A^{\mathrm{dep},d}p_G^{\mathrm{cnt},M,2\mathrm{nd}} = A^{\mathrm{dep},d}p_H^{\mathrm{cnt},M,2\mathrm{nd}}$ . Since $A^{\mathrm{blsoSurj},M,2\mathrm{nd}}$ is lower triangular with positive diagonal elements, it is invertible and thus

$$
\boldsymbol {A} ^ {\text {blsolnj}, M, 2 \text {nd}} \boldsymbol {A} ^ {\text {dep}, d} \boldsymbol {p} _ {G} ^ {\text {cnt}, M, 2 \text {nd}} = \boldsymbol {A} ^ {\text {blsolnj}, M, 2 \text {nd}} \boldsymbol {A} ^ {\text {dep}, d} \boldsymbol {p} _ {H} ^ {\text {cnt}, M, 2 \text {nd}}.
$$

Moreover, by definition of unfolding tree, there are only finite non-zero elements in both $A^{\mathrm{dep},d}p_G^{\mathrm{cnt},M,2\mathrm{nd}}$ and $A^{\mathrm{dep},d}p_H^{\mathrm{cnt},M,2\mathrm{nd}}$ , and the corresponding non-zero indices can only be in a fixed (finite) set. In this case, the upper triangular matrix $A^{\mathrm{blsolnj},M,2\mathrm{nd}}$ is reduced to a finite-dimensional matrix and thus $A^{\mathrm{dep},d}p_G^{\mathrm{cnt},M,2\mathrm{nd}} = A^{\mathrm{dep},d}p_H^{\mathrm{cnt},M,2\mathrm{nd}}$ . By enumerating all $d\geq 0$ , we obtain the desired result.

![](images/34a31467e39ce4417997799f25f0d0248118814ad8cf471d86824f7720220ea1.jpg)

Combined with all previous results, we have arrived at the concluding corollary:

Corollary C.34. Let $M \in \{Sub, L, LF, F\}$ be any model. For any two graphs G, H, $\chi_{G}^{M}(G) = \chi_{H}^{M}(H)$ iff $\hom(F, G) = \hom(F, H)$ for all $(F, T^{r}) \in \mathcal{S}^{M}$ .

Proof. According to Corollary C.16, for any two graphs G, H, we have $\chi_{G}^{M}(G) = \chi_{H}^{M}(H)$ iff $\text{cnt}^{M}((F,T^{r}),G) = \text{cnt}^{M}((F,T^{r}),H)$ for all $(F,T^{r}) \in \mathcal{S}^{M}$ . Then, Corollary C.33 implies that $\text{cnt}^{M}((F,T^{r}),G) = \text{cnt}^{M}((F,T^{r}),H)$ for all $(F,T^{r}) \in \mathcal{S}^{M}$ iff $\text{blso}((F,T^{r}),G) = \text{blso}((F,T^{r}),H)$ for all $(F,T^{r}) \in \mathcal{S}^{M}$ . Finally, Corollary C.32 implies that $\text{blso}((F,T^{r}),G) = \text{blso}((F,T^{r}),H)$ for all $(F,T^{r}) \in \mathcal{S}^{M}$ iff $\text{hom}(F,G) = \text{hom}(F,H)$ for all $(F,T^{r}) \in \mathcal{S}^{M}$ . We thus conclude the proof. □

# C.3 PART 2: NESTED EAR DECOMPOSITION

In this part, we give equivalent formulations of the set $\mathcal{S}^M$ for any model $M\in \{\mathrm{Sub},\mathrm{L},\mathrm{LF},\mathrm{F}\}$ using the concept of NED defined in Definition 3.3.

We first present some technical lemmas that will be used to deal with the tree decomposition of disconnected graphs.

Lemma C.35. Let $M \in \{Sub, L, LF, F\}$ be any model, and let $(F, T^{r}) \in S^{M}$ be a tree-decomposed graph with $\beta_{T}(t) = \{\{u, v\}\}$ for some $u, v \in V_{F}$ . Assume that F has two connected components $S_{1}, S_{2} \subset V_{F}$ and $u \in S_{1}$ and $v \in S_{2}$ are in different components. Then, there exist tree decompositions $(F[S_{1}], T_{1}^{r_{1}}), (F[S_{2}], T_{2}^{r_{2}}) \in S^{M}$ such that $\beta_{T_{1}}(r_{1}) = \{\{u, u\}\}$ and $\beta_{T_{2}}(r_{2}) = \{\{v, v\}\}$ .

Proof. We can simply define $T_{1} = (V_{T_{1}}, E_{T_{1}}, \beta_{T_{1}})$ with $V_{T_{1}} = V_{T}$ , $E_{T_{1}} = E_{T}$ , $r_{1} = r$ , and $\beta_{T_{1}}(t) = \{\{\phi_{1}(w): w \in \beta_{T}(t)\}\}$ with

$$
\phi_ {1} (w) = \left\{ \begin{array}{l l} w & \text { if } w \in S _ {1}, \\ u & \text { if } w \in S _ {2}. \end{array} \right.
$$

In other words, $T_{1}$ has the same structure as $T$ but with different bags such that all vertices in the same connected component as $v$ are replaced by $u$ . It is easy to see that $T_{1}^{r_{1}}$ is a canonical tree decomposition of $F[S_1]$ following Definition C.1. Moreover, we clearly have that for all $M \in \{\mathrm{Sub},\mathsf{L},\mathsf{LF},\mathsf{F}\}$ , $(F[S_1],T_1^{r_1}) \in S^M$ . We can construct $T_{2} = (V_{T_{2}},E_{T_{2}},\beta_{T_{2}})$ by symmetry, which concludes the proof.

The above lemma can be immediately generalized into the following one:

Corollary C.36. Let $M \in \{\mathsf{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model, and let $(F, T^r) \in S^M$ be a tree-decomposed graph. For each connected component $S \in V_F$ of $F$ , pick any $t \in V_T$ with the minimum depth such that $S \cap \beta_T(t) \neq \emptyset$ . Then, there exists a tree decomposition $\tilde{T}^s$ of $F[S]$ satisfying that $(F[S], \tilde{T}^s) \in S^M$ and $S \cap \beta_T(t) \subset \beta_{\tilde{T}}(s)$ .

Lemma C.37. Let $M \in \{Sub, L, LF, F\}$ be any model, and let $(F, T^{r}) \in S^{M}$ be a tree-decomposed graph such that F is connected. Then, for each child node t of r in $T^{r}$ and any vertex x in $F[T^{r}[t]]$ , there is a path in $F[T^{r}[t]]$ from x to some vertex in $\beta_{T}(r)$ . Consequently, the number of connected components of $F[T^{r}[t]]$ is bounded by the number of different elements in $\beta_{T}(r)$ .

Proof. Assume the above statement does not hold, and let $S \in V_{F[T^{r}[t]]}$ be the connected component of $F[T^{r}[t]]$ such that $S \cap \beta_{T}(r) = \emptyset$ . Note that $F[S] = F[T^{r}[t]][S]$ . Since F is connected, there exists an edge $\{v, w\} \in E_{F}$ such that $v \in S$ and $w \notin S$ . Since S is a connected component of $F[T^{r}[t]]$ , we have $w \notin V_{F[T^{r}[t]]}$ , and thus both $v, w \notin \beta_{T}(r)$ . This yields a contradiction since $\{v, w\}$ should be contained in a bag in $T^{r}$ but $B_{T}(v) \cap B_{T}(w) = \emptyset$ . ☐

To reduce duplication, below we only give proofs for Local 2-GNN and Local 2-FGNN. One can easily write a proof for Subgraphh GNN based on the proof of Local 2-GNN, and write a proof for 2-FGNN based on the proof of Local 2-FGNN.

Lemma C.38. For any tree-decomposed graph $(F, T^{r}) \in \mathcal{S}^{\mathrm{L}}$ , F has a strong NED.

Proof. Based on Corollary C.36, we can assume that $F$ is connected without loss of generality. We will prove the following stronger result: for any connected $(F, T^r) \in S^{\mathsf{L}}$ with $\beta_T(r) = \{\{u, v\}\}$ , $F$ has a strong NED where $u, v$ are endpoints of the first ear. (For the case of $u = v$ , the other endpoint can be arbitrary.)

The proof is based on induction over the number the vertices in $T^{r}$ . The above statement obviously holds for the base case of $|V_{T}| = 1$ . Now assume that the statement holds when $|V_{T}| \leq m$ , and consider the case of $|V_{T}| = m + 1$ . Note that for any two different children t, $t'$ of r, $F[T^{r}[t]]$ and $F[T^{r}[t']]$ can only share vertices u, v. For each child node t of r, denote its unique child node as $\tilde{t}$ . It is easy to see that $T^{r}[\tilde{t}]$ is a canonical tree decomposition of $F[T^{r}[\tilde{t}]]$ and $(F[T^{r}[\tilde{t}]], T^{r}[\tilde{t}]) \in \mathcal{S}^{\mathrm{L}}$ . However, one needs to be cautious as $F[T^{r}[\tilde{t}]]$ may not be connected (unlike the original graph F). Below, we separately consider the following cases:

1. First consider the case when $u = v$ . In this case, Lemma C.37 implies that $F[T^r[\tilde{t}]]$ is connected. According to the induction hypothesis, $F[T^r[\tilde{t}]]$ has a strong NED (denoted as $\mathcal{P}_{\tilde{t}}$ ) such that $u$ is an endpoint of the first ear. We then merge the ear decompositions $\mathcal{P}_{\tilde{t}}$ for all $\tilde{t}$ into a whole $\mathcal{P}$ , specify a root ear $P_1$ in any $\mathcal{P}_{\tilde{t}}$ , and let the first ear of other $\mathcal{P}_{\tilde{t}}$ nest on $P_1$ (with empty nested interval). It is easy to see that $\mathcal{P}$ is a strong NED of $F$ and $u$ is an endpoint of the first ear.   
2. Next consider the case when $u \neq v$ . In this case, without loss of generality, assume $\beta_T(\tilde{t}) = \{\{u, w\}\}$ for some $w \in V_F$ .

\- Subcase 1: $F[T^r[\tilde{t}]]$ is connected. Then, $F[T^r[\tilde{t}]]$ has a strong NED $\mathcal{P}_{\tilde{t}}$ such that $u$ and $w$ are endpoints of the first ear.

a) If $w = v$ , then $F[T^r [t]] = F[T^r [\tilde{t}]]$ and $F[T^r [t]]$ clearly has a strong NED.   
b) If $w \notin N_F[v]$ or $w = u$ , there are two additional subcases depending on whether $\{u, v\} \in E_F$ . If $\{u, v\} \in E_F$ , then $F[T^r[t]]$ has a strong NED (which can be constructed from $\mathcal{P}_{\tilde{t}}$ by adding an ear $\{\{u, v\}\}$ and letting the first ear in $\mathcal{P}_{\tilde{t}}$ nest on $\{\{u, v\}\}$ (with empty nested interval). Otherwise, $F[T^r[t]] \setminus \{v\}$ has a strong NED and $v$ is an isolated vertex in $F[T^r[t]]$ .   
c) If $w \in N_F(v)$ and $w \neq u$ , then $F[T^r[t]]$ also has a strong NED, which can be constructed from $\mathcal{P}_{\bar{t}}$ by extending the first ear to incorporate the edge $\{w, v\}$ (we still need an additional ear $\{\{u, v\}\}$ if $\{u, v\} \in E_F$ ).

\- Subcase 2: $F[T^r[\tilde{t}]]$ is disconnected. In this subcase, Lemma C.37 implies that $F[T^r[\tilde{t}]]$ has exactly two connected components, and $u$ and $w$ are in different connected components (which can be easily proved by noting that $v$ can only link to $u$ or $w$ in $F[T^r[\tilde{t}]]$ ). We can thus invoke Lemma C.35, which shows that both connected components of $F[T^r[\tilde{t}]]$ , denoted as $\widehat{F}_1$ and $\widehat{F}_2$ , admits tree decompositions $\widehat{T}_1^{s_1}$ and $\widehat{T}_2^{s_2}$ such that $(\widehat{F}_1,\widehat{T}_1^{s_1}),(\widehat{F}_2,\widehat{T}_2^{s_2})\in S^{\mathsf{L}}$ , $\beta_{\widehat{T}_1}(s_1) = \{\{u,u\}\}$ and $\beta_{\widehat{T}_2}(s_2) = \{\{w,w\}\}$ . Moreover, $|V_{\widehat{T}_1}| = |V_{\widehat{T}_2}| = |V_{T^r(\tilde{t})}| < m + 1$ . Therefore, according to the induction hypothesis, $\widehat{F}_1$ has a strong NED $\mathcal{P}_{\tilde{t},1}$ with $u$ as an endpoint of the first ear, and $\widehat{F}_2$ has a strong NED $\mathcal{P}_{\tilde{t},2}$ with $w$ as an endpoint of the first ear. If $\{w,v\} \in E_F$ , we can extend the first ear in $\mathcal{P}_{\tilde{t},2}$ to include the edge $\{w,v\}$ . Finally, if $\{u,v\} \in E_F$ , by setting the first ear to be $\{\{u,v\}\}$ , we can merge the two NEDs $\mathcal{P}_{\tilde{t},1}$ and $\mathcal{P}_{\tilde{t},2}$ to obtain the NED of $F[T^r[t]]$ such that $u,v$ are endpoints of the first ear.

Overall, we always have that:

a) If $F[T^r [t]]$ is connected, then it admits a strong NED such that $u, v$ are endpoints of the first ear.   
b) If $F[T^r [t]]$ is disconnected, then it has two connected components each admitting a strong NED such that $u, v$ belong to an endpoint of the first ear for each of the two NEDs, respectively.

Finally, noting that $F = \bigcup_{t,\mathsf{pa}_{T^r}(t) = r}F[T^r [t]]$ , we can merge all NEDs of $F[T^r [t]]$ to obtain a strong NED of $F$ with $u,v$ as two endpoints.

We thus conclude the proof of the induction step.

![](images/a2e8798772a970ff9dfdde0bec0efebaa2b55b4e6a6a012644b1df0b2d5748a9.jpg)

Lemma C.39. For any graph $F$ , if $F$ admits a strong NED, then there exists a tree decomposition $T^{r}$ of $F$ such that $(F, T^{r}) \in S^{\mathsf{L}}$ .

Proof. We can assume that $F$ is connected without loss of generality, as it is easy to merge the tree decompositions of different connected components to form a single tree which is a tree decomposition of the whole graph. We will prove the following stronger result: for any connected graph $F$ , if $F$ admits a strong NED where $u$ and $v$ are two different endpoints of the first ear, then there exists a tree decomposition $T^r$ of $F$ such that $(F, T^r) \in S^{\mathsf{L}}$ and $\beta_T(r) = \{\{u, v\}\}$ .

The proof is based on induction over the number the edges in F. The above statement obviously holds for the base case of $|E_{F}| = 1$ . Now assume that the statement holds when $|E_{F}| \leq m$ , and consider the case of $|E_{F}| = m + 1$ . Denote the i-th ear as $P_{i}$ and denote $P_{1} = \{\{w_{0}, w_{1}\}, \cdots, \{w_{l-1}, w_{l}\}\}$ where $w_{0} = u$ , $w_{l} = v$ . Consider the following three cases:

1. Either $u$ or $v$ is a cut vertex in $F$ . Without loss of generality, assume that $u$ is a cut vertex. In this case, there must exist a child ear $P_{i}$ nested on $P_{1}$ such that $u$ is an endpoint of $P_{i}$ and $I(P_{i}) = \emptyset$ . We can thus split all ears into two parts $\mathcal{P}_1$ and $\mathcal{P}_2$ , where $\mathcal{P}_1$ contains the ear $P_{i}$ and its descendant ears in the NED tree, and $\mathcal{P}_2$ contains other ears. Denote by $F[\mathcal{P}_k]$ the connected graph induced by all edges in $\mathcal{P}_k$ ( $k \in \{1,2\}$ ). Then, $\mathcal{P}_k$ is a strong NED of the graph $F[\mathcal{P}_k]$ . Since both $\mathcal{P}_1$ and $\mathcal{P}_2$ contain no more than $m$ edges, according to the induction hypothesis, there are tree decompositions $T_k^{r_k}$ of $F[\mathcal{P}_k]$ ( $k \in \{1,2\}$ ) such that $\beta_{T_1}(r_1) = \{\{u,w\}\}$ for some $w \in V_F$ , $\beta_{T_2}(r_2) = \{\{u,v\}\}$ , and $(F[\mathcal{P}_k], T_k^{r_k}) \in S^{\mathrm{L}}$ ( $k \in \{1,2\}$ ). We can glue the tree $T_1^{r_1}$ and $T_2^{r_2}$ into a larger tree $T^{r_2}$ by adding a new node $t$ with $\beta_T(t) = \{\{u,v,w\}\}$ , setting $\mathsf{pa}_{T^r}(t) = r_2$ and $\mathsf{pa}_{T^r}(r_1) = t$ . Clearly, $T^{r_2}$ is a valid tree decomposition of $F$ , $\beta_T(r_2) = \{\{u,v\}\}$ , and $(F,T^{r_2}) \in S^{\mathrm{L}}$ .

2. There is an ear $P_{i}$ nested on $P_{1}$ with $I(P_{i}) = P_{1}$ . In this case, we can split all ears into two parts $\mathcal{P}_1$ and $\mathcal{P}_2$ , where $\mathcal{P}_1$ contains the ear $P_{i}$ and its descendant ears in the NED tree, and $\mathcal{P}_2$ contains other ears. Then, $\mathcal{P}_k$ is a strong NED of the graph $F[\mathcal{P}_k]$ . Since both $\mathcal{P}_1$ and $\mathcal{P}_2$ contain no more than $m$ edges, according to the induction hypothesis, there is a tree decomposition $T_k^{r_k}$ of $F[\mathcal{P}_k]$ such that $\beta_{T_k}(r_k) = \{\{u,v\}\}$ and $(F[\mathcal{P}_k], T_k^{r_k}) \in S^{\mathrm{L}}$ . By merging the root node, we can glue the tree $T_1^{r_1}$ and $T_2^{r_2}$ into a larger tree $T^r$ . Clearly, $T^r$ is a valid tree decomposition of $F$ , $\beta_T(r) = \{\{u,v\}\}$ , and $(F,T^r) \in S^{\mathrm{L}}$ .

3. Neither $u$ nor $v$ is a cut vertex in $F$ and all ears $P_i$ nested on $P_1$ satisfies $I(P_i) \subsetneq P_1$ . In this case, we have either $\{w_0, w_1\} \notin I(P_i)$ for all ear $P_i$ nested on $P_1$ or $\{w_{l-1}, w_l\} \notin I(P_i)$ for all ear $P_i$ nested on $P_1$ (otherwise, it would contradict the definition of strong NED). Without loss of generality, assume $\{w_0, w_1\} \notin I(P_i)$ for all ears $P_i$ nested on $P_1$ . Then, it is clear that $l > 1$ , and the subgraph $F \setminus \{u\}$ is connected and also admits a strong NED where $w_1$ and $v$ are two endpoints of the first ear. Therefore, according to the induction hypothesis, there is a tree decomposition $\tilde{T}^s$ of $F \setminus \{u\}$ satisfying $(F \setminus \{u\}, \tilde{T}^s) \in S^L$ and $\beta_{\tilde{T}}(s) = \{\{w_1, v\}\}$ . We can then construct a tree $T^r$ from $\tilde{T}^s$ by adding two fresh nodes $r$ and $r'$ and setting $\mathsf{pa}_{T^r}(s) = r'$ and $\mathsf{pa}_{T^r}(r') = r$ . Set $\beta_T(r) = \{\{u, v\}\}, \beta_T(r') = \{\{u, v, w_1\}\}$ , and $\beta_T(t) = \beta_{\tilde{T}}(t)$ for all $t \in V_{\tilde{T}}$ . It is easy to see that the constructed $T^r$ is a tree decomposition of $F$ and $(F, T^r) \in S^L$ .

Combining the above three cases concludes the induction step.

![](images/58a48a563b43be29558c027ab9f0e0a81600fc10fb9bd494cbd484e33c0b01de.jpg)

Theorem C.40. For any graph $F$ , $F$ has a strong NED iff there is a tree decomposition $T^r$ of $F$ such that $(F, T^r) \in S^L$ .

Proof. This is a direct consequence of Lemmas C.38 and C.39.

![](images/4db84bc6b671e4acba419dc8c39d32f1ac523c0c0040f0f0522005d892be4e5d.jpg)

We next turn to Local 2-FGNN, where the proof has a similar structure as that of Local 2-GNN.

Lemma C.41. For any tree-decomposed graph $(F, T^{r}) \in \mathcal{S}^{\mathrm{LF}}$ , F has an almost-strong NED.

Proof. Based on Corollary C.36, we can assume that $F$ is connected without loss of generality. We will prove the following stronger result: for any connected $(F, T^r) \in S^{\mathsf{LF}}$ with $\beta_T(r) = \{\{u, v\}\}$ , $F$ has an almost-strong NED where $u, v$ are endpoints of the first ear. (For the case of $u = v$ , the other endpoint can be arbitrary.)

Similar to the proof of Lemma C.38, assume that the statement holds when $|V_T| \leq m$ , and consider the case of $|V_T| = m + 1$ . For each child node $t$ of $r$ , if $t$ only has one child, the proof exactly follows the one in Lemma C.38. Therefore, it suffices to consider the case where $t$ has two children $q_1$ and $q_2$ . Denote $\beta_T(t) = \{u, v, w\}$ , $\beta_T(q_1) = \{u, w\}$ , and $\beta_T(q_2) = \{v, w\}$ . Since $w \in N_F[u] \cup N_F[v]$ (by definition of $S^{\mathsf{LF}}$ ), we can assume that $w \in N_F[v]$ without loss of generality. Recall that $(F[T^r[q_1]], T^r[q_1])$ , $(F[T^r[q_2]], T^r[q_2]) \in S^{\mathsf{LF}}$ . Below, we separately consider the following cases:

1. First consider the case when $u = v$ . In this case, Lemma C.37 implies that both $F[T^r [q_1]]$ and $F[T^r [q_2]]$ are connected (since either $u = w$ or $\{u, w\} \in E_F$ ). According to the induction hypothesis, both $F[T^r [q_1]]$ and $F[T^r [q_2]]$ admit an almost-strong NED such that $u$ and $w$ are endpoints of the first ear. If $w = u$ , we can merge the two almost-strong NEDs so that the first ear in one NED is nested on the first ear of the other (with an empty nested interval). Otherwise, the first ears in the two almost-strong NEDs share both endpoints and we can clearly merge them (in case of $\{u, w\} \in E_F$ , there is a common ear $\{\{u, w\}\}$ in both NEDs,

which is taken only once). In both subcases, we obtain an almost-strong NED of $F[T^r [t]]$ such that $u$ is an endpoint of the first ear.

2. Next consider the case when $u \neq v$ . In this case, $F[T^r[q_1]]$ and $F[T^r[q_2]]$ share only one vertex $w$ . Note that $F[T^r[q_2]]$ is connected by Lemma C.37. We separately consider two subcases:

\- Subcase 1: $F[T^r[q_1]]$ is connected. Then, according to the induction hypothesis, $F[T^r[q_1]]$ admits an almost-strong NED $\mathcal{P}_1$ such that $u$ and $w$ are endpoints of the first ear, and $F[T^r[q_2]]$ admits an almost-strong NED $\mathcal{P}_2$ such that $w$ and $v$ are endpoints of the first ear. (i) If $w = v$ , we can merge the two almost-strong NEDs so that the first ear of $\mathcal{P}_2$ is nested on the first ear of $\mathcal{P}_1$ (with empty nested interval). (ii) If $w = u$ , we can merge the two almost-strong NEDs so that the first ear of $\mathcal{P}_1$ is nested on the first ear of $\mathcal{P}_2$ (with empty nested interval). (iii) If $w \in N_F(v)$ and $w \neq u$ , the first ear of $\mathcal{P}_2$ can be chosen as $\{\{w, v\}\}$ . Then, we can merge the two almost-strong NEDs by gluing the first year in $\mathcal{P}_1$ with the ear $\{\{w, v\}\}$ in $\mathcal{P}_2$ . One can see that the resulting NED is almost-strong. Overall, we always have that $F[T^r[t]]$ admits an almost-strong NED such that $u, v$ are endpoints of the first ear.

\- Subcase 2: $F[T^r [q_1]]$ is disconnected. In this subcase, $w \neq u$ . Similar to the proof of Lemma C.38, we obtain that $F[T^r [q_1]]$ has exactly two connected components, and $u$ and $w$ are in different connected components. We can thus invoke Lemma C.35, which shows that both connected components of $F[T^r [q_1]]$ , denoted as $\widehat{F}_1$ and $\widehat{F}_2$ , admit almost-strong NEDs $\mathcal{P}_1$ and $\mathcal{P}_2$ , respectively. Moreover, $u$ is an endpoint of the first ear in $\mathcal{P}_1$ , and $w$ is an endpoint of the first ear in $\mathcal{P}_2$ . Besides, $F[T^r [q_2]]$ admits an almost-strong NED $\mathcal{P}_3$ with $w, v$ as the endpoints of the first ear. By letting the first ear of $\mathcal{P}_2$ nest on the first ear of $\mathcal{P}_3$ (with an empty nested interval), we can merge $\mathcal{P}_2$ and $\mathcal{P}_3$ . Then, we can merge $\mathcal{P}_1$ and $\mathcal{P}_2 \cup \mathcal{P}_3$ following the same procedure as Subcase 2 in the proof of Lemma C.38.

Overall, we always have that:

a) If $F[T^r [t]]$ is connected, then it admits an almost-strong NED such that $u, v$ are endpoints of the first ear.   
b) If $F[T^r [t]]$ is disconnected, then it has two connected components each admitting an almost-strong NED such that $u, v$ belong to an endpoint of the first ear for each of the two NEDs, respectively.

In both subcases, it follows that $F[T^{r}[t]]$ admits an almost-strong NED with u, v as two endpoints. Noting that $F = \bigcup_{t,\mathsf{pa}_{Tr}(t)=r} F[T^{r}[t]]$ , we can merge all NEDs of $F[T^{r}[t]]$ to obtain an almost-strong NED of F with u, v as two endpoints.

We thus conclude the proof of the induction step.

Lemma C.42. For any graph F, if F admits an almost-strong NED, then there exists a tree decomposition $T^{r}$ of F such that $(F, T^{r}) \in S^{\mathrm{LF}}$ .

Proof. We can assume that $F$ is connected without loss of generality. We will prove the following stronger result: for any graph $F$ , if $F$ admits an almost-strong NED where $u$ and $v$ are two different endpoints of the first ear, then there exists a tree decomposition $T^r$ of $F$ such that $(F, T^r) \in S^{\mathrm{LF}}$ and $\beta_T(r) = \{\{u, v\}\}$ .

Similar to the proof of Lemma C.39, assume that the statement holds when $|E_{F}| \leq m$ , and consider the case of $|E_{F}| = m + 1$ . Denote the first ear as $P_{1} = \{\{w_{0}, w_{1}\}, \cdots, \{w_{l-1}, w_{l}\}\}$ where $w_{0} = u$ , $w_{l} = v$ . Consider the following three cases:

1. Either u or v is a cut vertex in F. This case is exactly the same as in Lemma C.39.   
2. There is an ear $P_{i}$ nested on $P_{1}$ with $I(P_{i}) = P_{1}$ . This case is also the same as in Lemma C.39.   
3. Otherwise, $l \geq 2$ .

\- If there is an ear $P_{i}$ nested on $P_{1}$ with $\{w_{0}, w_{1}\}, \{w_{1}, w_{2}\} \in I(P_{i})$ , then $l \geq 3$ . By definition of almost-strong NED, there does not exist an ear $P_{j}$ nested on $P_{1}$ with $\{w_{l-2}, w_{l-1}\}, \{w_{l-1}, w_{l}\} \in I(P_{i})$ . In this case, we can split $P_{1}$ into two parts:

$P_{1,u} = P_{1} \backslash \{w_{l-1}, w_{l}\}$ , and $P_{1,v} = \{w_{l-1}, w_{l}\}$ . Then, we can rearrange any ear nested on $P_{1}$ so that it is nested on either $P_{1,u}$ or $P_{1,v}$ . In this way, we can split all ears into two sets $\mathcal{P}_{u}$ and $\mathcal{P}_{v}$ , one corresponding to $P_{1,u}$ and its descendant ears, and the other corresponding to $P_{1,v}$ and its descendant ears. Therefore, according to the induction hypothesis, there is a tree decomposition $\tilde{T}_{u}^{s}$ of $F[\mathcal{P}_{u}]$ satisfying $(F[\mathcal{P}_{u}], \tilde{T}_{u}^{s}) \in \mathcal{S}^{\text{LF}}$ and $\beta_{\tilde{T}_{u}}(s) = \{u, w_{l-1}\}$ , and similarly, there is a tree decomposition $\tilde{T}_{v}^{s'}$ of $F[\mathcal{P}_{v}]$ satisfying $(F[\mathcal{P}_{v}], \tilde{T}_{v}^{s'}) \in \mathcal{S}^{\text{LF}}$ and $\beta_{\tilde{T}_{v}}(s) = \{w_{l-1}, v\}$ . We can then construct a tree $T^{r}$ by merging $\tilde{T}_{u}^{s}$ and $\tilde{T}_{v}^{s'}$ and adding two fresh nodes $r, r'$ , where $r$ is the root node, $\mathsf{pa}_{T^{r}}(r') = r$ and $\mathsf{pa}_{T^{r}}(s) = \mathsf{pa}_{T^{r}}(s') = r'$ . Set $\beta_{T}(r) = \{u, v\}$ and $\beta_{T}(r') = \{u, v, w_{l-1}\}$ . It is easy to see that the constructed $T^{r}$ is a tree decomposition of $F$ and $(F, T^{r}) \in \mathcal{S}^{\text{LF}}$ .

\- If there does not exist an ear $P_{i}$ nested on $P_{1}$ with $\{w_0, w_1\}, \{w_1, w_2\} \in I(P_i)$ , then we follow exactly the same analysis as in the previous item, expect that now we split $P_{1}$ into two parts: $P_{1,u} = \{\{w_0, w_1\}\}$ , and $P_{1,v} = P_{1} \setminus \{\{w_0, w_1\}\}$ . We can still construct a tree decomposition $T^r$ of $F$ such that $(F, T^r) \in S^{\mathsf{LF}}$ .

Combining the above three cases concludes the proof.

![](images/99c41db82e3b0fd07c0334582f8a496a2d964d9142a9c7200ddf8e3928bc236c.jpg)

Theorem C.43. For any graph $F, F$ has an almost-strong NED iff there is a tree decomposition $T^r$ of $F$ such that $(F, T^r) \in S^{\mathsf{LF}}$ .

Proof. This is a direct consequence of Lemmas C.41 and C.42.

![](images/c9e1f4ddfe61bcfa5a8e8e07da1a1bc294ae8a2c813e00f5867242b6d68248ac.jpg)

# C.4 PART 3: PEBBLE GAME

In this part, we prove that $S^{M}$ is maximal (Definition 3.1(b)) for any $M \in \{Sub, L, LF, F\}$ . To achieve this, we first introduce a general class of graphs which we call Fürer graphs (Fürer, 2001). Intuitions and illustrations of Fürer graphs can be found in Zhang et al. (2023a).

Definition C.44 (Fürer graphs). Given any connected graph $F = (V_F, E_F, \ell_F)$ , the Fürer graph $G(F) = (V_{G(F)}, E_{G(F)}, \ell_{G(F)})$ is constructed as follows:

$$
V _ {G (F)} = \{(x, X): x \in V _ {F}, X \subset N _ {F} (x), | X | \bmod 2 = 0 \},
$$

$$
E _ {G (F)} = \{\{(x, X), (y, Y) \} \subset V _ {G}: \{x, y \} \in E _ {F}, (x \in Y \leftrightarrow y \in X) \},
$$

$$
\ell_ {G (F)} (x, X) = \ell_ {F} (x) \quad \forall (x, X) \in V _ {G (F)}.
$$

Here, $x \in Y \leftrightarrow y \in X$ holds when either $(x \in Y \text{ and } y \in X)$ or $(x \notin Y \text{ and } y \notin X)$ holds. For each $x \in V_F$ , denote the set

$$
\operatorname{Meta} _ {F} (x) := \{(x, X): X \subset N _ {F} (x), | X | \bmod 2 = 0 \}, \tag {24}
$$

which is called the meta vertices of $G(F)$ associated to $x$ . Note that $V_{G(F)} = \bigcup_{x\in V_F}\mathsf{Meta}_F(x)$ .

We next define an operation called "twist":

Definition C.45 (Twist). Let $G(F) = (V_{G(F)}, E_{G(F)}, \ell_{G(F)})$ be the Fürer graph of $F = (V_{F}, E_{F}, \ell_{F})$ , and let $\{x, y\} \in E_{F}$ be an edge of F. The twisted Fürer graph of $G(F)$ for edge $\{x, y\}$ , is constructed as follows: twist( $G(F), \{x, y\} := (V_{G(F)}, E_{\text{twist}(G(F), \{x, y\}), \ell_{G(F)})$ , where

$$
E _ {\text { twist } (G (F), \{x, y \})} := E _ {G (F)} \triangle \{\{\xi , \eta \}: \xi \in \mathsf {M e t a} _ {F} (x), \eta \in \mathsf {M e t a} _ {F} (y) \},
$$

and $\triangle$ is the symmetric difference operator, i.e., $A\triangle B = (A \backslash B) \cup (B \backslash A)$ . For an edge set $S = \{e_{1}, \cdots, e_{k}\} \subset E_{F}$ , we further define

$$
\operatorname{twist} (G (F), S) := \operatorname{twist} (\dots \operatorname{twist} (G (F), e _ {1}) \dots , e _ {k}). \tag {25}
$$

Note that Equation (25) is well-defined as the resulting graph does not depend on the order of edges $e_{1}, \cdots, e_{k}$ for twisting.

The following result is well-known (see e.g., Zhang et al., 2023a, Corollary I.5 and Lemma I.7):

Theorem C.46. For any graph F and any set $S \subset E_{F}$ , $G(F) \simeq \text{twist}(G(F), S)$ iff $|S| \mod 2 = 0$ .

In the subsequent result, let $\{u,v\} \in E_F$ be any edge in $F$ and denote $H(F) = \text{twist}(G(F), \{u,v\})$ . We now show that $G(F), H(F)$ can be distinguished via homomorphism information:

Theorem C.47. For any graph $F$ and Fürer graphs $G(F), H(F)$ defined above, $\operatorname{hom}(F, G(F)) \neq \operatorname{hom}(F, H(F))$ .

Proof. In the proof below, we use $g$ and $h$ to denote mappings where $g: V_F \to V_{G(F)}$ and $h: V_F \to V_{H(F)}$ . The proof is divided into the following parts.

1. We first prove that there is a homomorphism $g \in \operatorname{Hom}(F, G(F))$ satisfying $g(w) \in \operatorname{Meta}_F(w)$ for all $w \in V_F$ , but there is no homomorphism $h \in \operatorname{Hom}(F, H(F))$ satisfying $h(w) \in \operatorname{Meta}_F(w)$ for all $w \in V_F$ .

a) Define a mapping $g: V_F \to V_{G(F)}$ such that $g(w) = (w, \emptyset)$ for all $w \in V_F$ . We clearly have $\{w, x\} \in E_F$ implies that $\{g(w), g(x)\} \in E_{G(F)}$ . Moreover, $\ell_{G(F)}(g(w)) = \ell_F(w)$ . Therefore, $g \in \operatorname{Hom}(F, G(F))$ is indeed a homomorphism.   
b) If we similarly define $h: V_F \to V_{H(F)}$ such that $h(w) = (w, \emptyset)$ for all $w \in V_F$ , then for all $\{w, x\} \in E_F \backslash \{\{u, v\}\}$ we have $\{h(w), h(x)\} \in E_{H(F)}$ , but $\{h(u), h(v)\} \notin E_{H(F)}$ since the edge $\{u, v\}$ is twisted.

It remains to prove that for all $h: V_F \to V_{H(F)}$ of the form $h(w) = (w, U_w)$ (for each $w \in V_F$ ), $h \notin \operatorname{Hom}(F, H(F))$ . It suffices to prove that there is an odd number of edges $\{w, x\} \in E_F$ such that $\{h(w), h(x)\} \notin E_{H(F)}$ . Let $h, \tilde{h}$ be two such mappings that differ in only one vertex $z$ , i.e., $h(w) = \tilde{h}(w)$ for all $w \neq z$ but $h(z) = (z, U_z) \neq (z, \tilde{U}_z) = \tilde{h}(z)$ .

Denote $D_{z} = \tilde{U}_{z}\triangle U_{z}$ . Based on the definition of Fürer graph, it follows that

- for all $\{w, x\} \in E_F$ with $w \neq z, x \neq z$ , we have $\{h(w), h(x)\} \in E_{H(F)}$ iff $\{\tilde{h}(w), \tilde{h}(x)\} \in E_{H(F)}$ ;   
- for all $\{w, z\} \in E_F$ with $w \notin D_z$ , we also have $\{h(w), h(z)\} \in E_{H(F)}$ iff $\{\tilde{h}(w), \tilde{h}(z)\} \in E_{H(F)}$ ;   
- for all $\{w, z\} \in E_F$ with $w \in D_z$ , we have $\{h(w), h(z)\} \in E_{H(F)}$ iff $\{\tilde{h}(w), \tilde{h}(z)\} \notin E_{H(F)}$ .

Since $|U_z| \mod 2 = 0$ and $|\tilde{U}_z| \mod 2 = 0$ , we have $|D_z| \mod 2 = 0$ and thus the number of edges $\{w, x\} \in E_F$ such that $\{h(w), h(x)\} \notin E_{H(F)}$ has the same parity as the number of edges such that $\{\tilde{h}(w), \tilde{h}(x)\} \notin E_{H(F)}$ . Finally, noting that all mappings $h$ can be obtained from the one in (b) by continually modifying $U_w$ for each $w \in V_F$ and the parity remains unchanged, we have concluded the proof of this part.

2. We next prove that for any permutation $\pi : V_F \to V_F$ , there does not exist a homomorphism $h \in \operatorname{Hom}(F, H(F))$ satisfying $h(w) \in \operatorname{Meta}_F(\pi(w))$ for all $w \in V_F$ . Assume that the conclusion does not hold and pick any $h$ satisfying the above condition. Consider the following two cases:

a) Case 1: if $\pi$ is an automorphism of $F$ , then it is easy to see that $h \circ \pi^{-1} \in \operatorname{Hom}(F, H(F))$ , because $\{w, x\} \in E_F \implies \{\pi^{-1}(w), \pi^{-1}(x)\} \in E_F \implies \{h(\pi^{-1}(w)), h(\pi^{-1}(x))\} \in E_F$ for all $w, x \in V_F$ , and $\ell_F(w) = \ell_F(\pi^{-1}(w)) = \ell_{G(F)}(h(\pi^{-1}(w)))$ for all $w \in V_F$ . Moreover, $h \circ \pi^{-1}$ satisfies that $h(\pi^{-1}(w)) \in \operatorname{Meta}_F(w)$ , yielding a contradiction to point 1.   
b) Case 2: if $\pi$ is not an automorphism of $F$ , then there exists an edge $\{w, x\} \in E_F$ such that $\{\pi(w), \pi(x)\} \notin E_F$ . In this case, we must have $\{h(w), h(x)\} \notin E_{H(F)}$ since by definition $h(w) \in \mathsf{Meta}_F(\pi(w))$ , $h(x) \in \mathsf{Meta}_F(\pi(x))$ , and $\{\pi(w), \pi(x)\}$ is not an edge of $F$ .

In both cases, $h$ is invalid and thus there is no homomorphism $h \in \mathsf{Hom}(F, H(F))$ satisfying $h(w) \in \mathsf{Meta}_F(\pi(w))$ for all $w \in V_F$ .

3. We finally prove that the following two sets have equal size (i.e., $|S_G| = |S_H|$ ):

$$
S _ {G} := \left\{g \in \operatorname{Hom} (F, G (F)): \exists w, x, y \in V _ {F} \text { s.t. } x \neq y, g (x), g (y) \in \operatorname{Meta} _ {F} (w) \right\},
$$

$$
S _ {H} := \left\{h \in \operatorname{Hom} (F, H (F)): \exists w, x, y \in V _ {F} \text { s.t. } x \neq y, h (x), h (y) \in \operatorname{Meta} _ {F} (w) \right\}.
$$

It suffices to prove that, for any proper subset $U \subsetneq V_F$ , we have $|S_G^U| = |S_H^U|$ , where $S_G$ and $S_H$ are defined as follows:

$$
S _ {G} ^ {U} := \left\{g \in \operatorname{Hom} (F, G (F)): g (x) \in \bigcup_ {w \in U} \operatorname{Meta} _ {F} (w) \forall x \in V _ {F} \right\},
$$

$$
S _ {H} ^ {U} := \left\{h \in \operatorname{Hom} (F, H (F)): h (x) \in \bigcup_ {w \in U} \operatorname{Meta} _ {F} (w) \forall x \in V _ {F} \right\}.
$$

Fix $U \subsetneq V_{F}$ and pick $z \in V_{F} \setminus U$ . Let P be a simple path from u to z of the form $P = \{\{w_{0}, w_{1}\}, \cdots, \{w_{k-1}, w_{k}\}\} \subset E_{F}$ where $\{w_{0}, w_{1}\} = \{u, v\}$ , $w_{k} = z$ . Define a mapping $\sigma$ that takes $g \in S_{G}^{U}$ as input and outputs a mapping $h : V_{F} \to V_{H(F)}$ :

$$
h (x) = \left\{ \begin{array}{l l} g (x) & \text {if} [ g (x) ] _ {0} \notin \{w _ {1}, \dots , w _ {k - 1} \}, \\ ([ g (x) ] _ {0}, [ g (x) ] _ {1} \triangle \{w _ {i - 1}, w _ {i + 1} \}) & \text {if} [ g (x) ] _ {0} = w _ {i}, i \in [ k - 1 ], \end{array} \right.
$$

where we write $g(x) = ([g(x)]_{0}, [g(x)]_{1})$ . We will prove that $h \in S_{H}^{U}$ . Since we clearly have $h(w) \in U$ for all $w \in V_{F}$ , it suffices to prove that $h \in \mathsf{Hom}(F, H(F))$ . Let $\{x, y\} \in E_{F}$ be any edge in F.

\- If $[g(x)]_0, [g(y)]_0 \notin \{w_1, \cdots, w_{k-1}\}$ , then $\{h(x), h(y)\} = \{g(x), g(y)\} \in E_{G(F)}$ . Also, since $\{[g(x)]_0, [g(y)]_0\} \neq \{u, v\}$ , $\{[g(x)]_0, [g(y)]_0\}$ is not twisted and thus $\{h(x), h(y)\} \in E_{H(F)}$ .

\- If $[g(x)]_0 \notin \{w_1, \cdots, w_{k-1}\}$ , $[g(y)]_0 = w_i$ for some $i \in [k - 1]$ , and $\{[g(x)]_0, [g(y)]_0\} \neq \{\{u, v\}, \{w_{k-1}, w_k\}\}$ , then

$$
\{h (x), h (y) \} = \{([ g (x) ] _ {0}, [ g (x) ] _ {1}), ([ g (y) ] _ {0}, [ g (y) ] _ {1} \triangle \{w _ {i - 1}, w _ {i + 1} \}) \}.
$$

We have

$$
\begin{array}{l} \{g (x), g (y) \} \in E _ {G (F)} \iff ([ g (x) ] _ {0} \in [ g (y) ] _ {1}) \leftrightarrow ([ g (y) ] _ {0} \in [ g (x) ] _ {1}) \\ \Longleftrightarrow \left([ g (x) ] _ {0} \in [ g (y) ] _ {1} \triangle \{w _ {i - 1}, w _ {i + 1} \}\right) \leftrightarrow \left([ g (y) ] _ {0} \in [ g (x) ] _ {1}\right) \\ \Longleftrightarrow \{h (x), h (y) \} \in E _ {G (F)}. \\ \end{array}
$$

Also, since $\{[g(x)]_0,[g(y)]_0\}$ is not twisted, $\{h(x),h(y)\} \in E_{H(F)}$ .

\- If $[g(x)]_0, [g(y)]_0 \in \{w_1, \cdots, w_{k-1}\}$ , the analysis is similar to the above one and we can still prove that $\{h(x), h(y)\} \in E_{G(F)}$ and thus $\{h(x), h(y)\} \in E_{G(F)}$ .

\- If $\{[g(x)]_0, [g(y)]_0\} = \{u, v\}$ , without loss of generality assume $w_0 = u = [g(x)]_0$ and $w_1 = v = [g(y)]_0$ . We have

$$
\{g (x), g (y) \} \in E _ {G (F)} \iff (u \in [ g (y) ] _ {1}) \leftrightarrow (v \in [ g (x) ] _ {1})
$$

$$
\Longleftrightarrow (u \notin [ g (y) ] _ {1} \triangle \{u, w _ {2} \}) \leftrightarrow (v \in [ g (x) ] _ {1})
$$

$$
\Longleftrightarrow \{h (x), h (y) \} \notin E _ {G (F)}.
$$

However, since $\{u,v\}$ is twisted, we still have $\{h(x),h(y)\} \in E_{H(F)}$ .

\- Finally, we do not need to consider the case $\{[g(x)]_0,[g(y)]_0\} = \{w_{k - 1},w_k\}$ since $w_{k} = z\notin U$ .

This proves that $h \in S_H^U$ . Moreover, it is straightforward to see that the mapping $\sigma(g) = h$ is a bijection from $S_G^U$ to $S_H^U$ . We have thus proved that $|S_G^U| = |S_H^U|$ .

Combining the above three items, we obtain $\operatorname{Hom}(F, G(F)) > \operatorname{Hom}(F, H(F))$ , concluding the proof.

In the subsequent analysis, we will prove that for all model M considered in Theorem 3.4 and any connected graph $F, \chi_{G(F)}^{M}(G(F)) = \chi_{H(F)}^{M}(H(F))$ if $F \notin S^{M}$ . The proof is based on an important technique developed in Cai et al. (1992), called the pebble game. When restricting our analysis on Fürer graphs, the pebble game can be greatly simplified as shown in Fürer (2001); Zhang et al. (2023a). Below, we separately describe the corresponding pebble game for each model M.

We first define a key concept called the connected component.

Definition C.48 (Connected components). Let $F = (V_{F}, E_{F})$ be a connected graph and let $U \subset V_{F}$ be a vertex set, called separation vertices. We say two edges $\{u, v\}, \{x, y\} \in E_{F}$ are in the same connected component if there is a simple path $\{\{y_{0}, y_{1}\}, \cdots, \{y_{k-1}, y_{k}\}\}$ satisfying that $\{y_{0}, y_{1}\} = \{u, v\}, \{y_{k-1}, y_{k}\} = \{x, y\}$ and $y_{i} \notin U$ for all $i \in [k-1]$ . It is easy to see that the above relation between edges forms an equivalence relation. Therefore, we can define a partition over the edge set, denoted by $\mathsf{CC}_{F}(U) = \{P_{i} : i \in [m]\}$ for some m, where each $P_{i} \subset E_{F}$ is called a connected component.

We are now ready to describe the game rule. There are two players (named Alice and Bob), a graph F, and several pebbles. At the beginning, all pebbles lie outside the graph. Through the game process, some pebbles will be placed on the vertices of F and thus separate the edges $E_{F}$ into connected components according to Definition C.48. In each game round, Alice updates the location of pebbles, while Bob maintains a subset of connected components, ensuring that the number of selected components is odd. There are three major types of operations:

1. Add a pebble p. Alice places a pebble p (previously outside the graph) on some vertex of F. If introducing this new pebble does not change the connected components, then Bob does nothing. Otherwise, there must be a connected component P separated by p into several components $P = \bigcup_{i \in [m]} P_i$ for some m. Bob will update his selected components by removing P (if selected) and optionally adding a subset of connected components in $\{P_1, \cdots, P_m\}$ while ensuring that the number of selected components in total (including previously selected components) is odd.   
2. Remove a pebble p. Alice removes a pebble p (previously on some vertex) outside the graph. If introducing this new pebble does not change the connected components, then Bob does nothing. Otherwise, there are multiple connected component $P_{1}, \cdots, P_{m}$ getting merged into a whole $P = \bigcup_{i \in [m]} P_{i}$ . Bob will update his selected components by removing all $P_{i}, i \in [m]$ (if selected) and optionally adding P, while ensuring that the number of selected components in total is odd.   
3. Swap two pebbles p and p'. Alice swaps the position of two pebbles p and p'. This operation does not change the connected components and thus Bob does nothing.

At any time, if there is an edge $\{x,y\}$ such that both endpoints hold pebbles and the connected component $\{\{x,y\}\}$ is selected by Bob, then Bob loses the game and Alice wins. If Alice cannot win through the game process, then Bob wins.

We now define the concrete pebble game for each model M considered in this paper. In cases of Subgraph GNN, Local 2-GNN, Local 2-FGNN, and 2-FGNN, there are three pebbles $p_{u}$ , $p_{v}$ , $p_{w}$ . As described before, all pebbles lie outside the graph at the beginning. Alice first adds the pebble $p_{u}$ (operation 1) and then adds the pebble $p_{v}$ (operation 1). Next, the game cyclically executes the following process:

\- Subgraph GNN. Alice can choose either one of the following ways to play:

- Remove the pebble $\mathsf{p}_v$ (operation 2), and re-add the pebble $\mathsf{p}_v$ (operation 1).   
- Add the pebble $\mathsf{p}_w$ (operation 1) adjacent to the pebble $\mathsf{p}_v$ , swap pebble $\mathsf{p}_v$ with $\mathsf{p}_w$ (operation 3), and remove the pebble $\mathsf{p}_w$ (operation 2).

\- Local 2-GNN. Alice can choose either one of the following ways to play:

- Remove the pebble $\mathsf{p}_u$ (operation 2), and re-add the pebble $\mathsf{p}_u$ (operation 1).   
- Remove the pebble $\mathsf{p}_v$ (operation 2), and re-add the pebble $\mathsf{p}_v$ (operation 1).   
- Add the pebble $\mathsf{p}_w$ (operation 1) adjacent to the pebble $\mathsf{p}_u$ , swap pebble $\mathsf{p}_u$ with $\mathsf{p}_w$ (operation 3), and remove the pebble $\mathsf{p}_w$ (operation 2).   
- Add the pebble $\mathsf{p}_w$ (operation 1) adjacent to the pebble $\mathsf{p}_v$ , swap pebble $\mathsf{p}_v$ with $\mathsf{p}_w$ (operation 3), and remove the pebble $\mathsf{p}_w$ (operation 2).

\- Local 2-FGNN. Alice can choose either one of the following ways to play:

- Remove the pebble $\mathsf{p}_u$ (operation 2), and re-add the pebble $\mathsf{p}_u$ (operation 1).   
- Remove the pebble $\mathsf{p}_v$ (operation 2), and re-add the pebble $\mathsf{p}_v$ (operation 1).   
- Add the pebble $\mathsf{p}_w$ (operation 1) adjacent to either pebble $\mathsf{p}_u$ or pebble $\mathsf{p}_v$ , swap pebble $\mathsf{p}_w$ with the adjacent pebble (operation 3), and remove the pebble $\mathsf{p}_w$ (operation 2).

\- 2-FGNN. Alice adds the pebble $\mathsf{p}_w$ (operation 1), swap the pebbles $\mathsf{p}_w$ with either $\mathsf{p}_u$ or $\mathsf{p}_v$ (operation 3), and remove the pebble $\mathsf{p}_w$ (operation 2).

Proposition C.49. For any model $M \in \{Sub, L, LF, F\}$ and any graph F, if Alice cannot win the pebble game associated to model M on graph F, then $\chi_{G(F)}^{M}(G(F)) = \chi_{H(F)}^{M}(H(F))$ .

Proof. Note that the pebble game does not depend on the vertex labels $\ell_F$ in $F$ . We first assume that the vertex labels are all different, i.e., $\ell_F(x) \neq \ell_F(y)$ for all $x, y \in V_F$ . Then, the labels of vertices $\xi \in V_{G(F)}, \eta \in V_{H(F)}$ are the same iff $\xi$ and $\eta$ belong to the same meta vertex, i.e., $\xi, \eta \in \mathsf{Meta}_F(x)$ for some $x \in V_F$ . Based on Proposition C.9, $\chi_{G(F)}^M(G(F)) = \chi_{H(F)}^M(H(F))$ iff $\tilde{\chi}_{G(F)}^M(G(F)) = \tilde{\chi}_{H(F)}^M(H(F))$ where $\tilde{\chi}^M$ is defined in Equations (7) to (10). In this case, the pebble games above exactly correspond to the aggregation formulas in Equations (7) to (10), and we can invoke the results in Zhang et al. (2023a, Theorem I.17) to show that $\tilde{\chi}_{G(F)}^M(G(F)) = \tilde{\chi}_{H(F)}^M(H(F))$ (note that we do not need to consider the augmented Fürer graphs defined in their paper based on our label assignment). Therefore, $\chi_{G(F)}^M(G(F)) = \chi_{H(F)}^M(H(F))$ .

We next consider the general case when multiple vertices can have the same label in F. However, this can only make it harder to distinguish between $G(F)$ and $H(F)$ , and we clearly have $\chi_{G(F)}^{M}(G(F)) = \chi_{H(F)}^{M}(H(F))$ . ☐

Based on the above theorem, in the remaining proof we will analyze the players' strategy in the pebble game. Surprisingly, it turns out that given a graph $F$ , if Alice can win the game, then her strategy can be described using the tree decomposition of $F$ defined in Definition C.6.

To illustrate this point, we need the concept of game state graph. Given a graph $F$ , a game state is a three-tuple $(u,v,Q)$ where $u,v\in V_{F}\cup \{\emptyset \}$ and $Q\subset \mathsf{CC}_{F}(\{u,v\})$ , denoting the vertex that holds pebble $\mathsf{p}_u$ , the vertex that holds pebble $\mathsf{p}_v$ , and a subset of connected component selected by Bob, respectively. Here, the symbol $\emptyset$ means that a pebble is left outside the graph. One can see that after any round, whether Alice can win the remaining game purely depends on this tuple. Now fix Alice's strategy. In each round, each game state will be transited to a finite number of states depending on how Bob plays, and all states and transitions form a directed graph, which we call the game state graph. The state $(u,v,Q)$ is called a terminal state if $\min_{P\in Q}|P| = 1$ . It is straightforward to see that Alice wins the game at any terminal state, as stated in the following result:

Proposition C.50. Given graph $F$ and model $M \in \{\text{Sub}, L, LF, F\}$ , let $G^S$ be the game state graph defined above corresponding to an Alice's strategy. Then, Alice can win the game if there is an integer $t$ such that any path in $G^S$ of length $t$ starting from the initial state $(\emptyset, \emptyset, \{E_F\})$ goes through a terminal state.

Proof. It suffices to prove that, at any terminal state, Alice can win the game. Let $(u,v,Q)$ be a terminal state with $\{x,y\}\in Q$ . If $\{u,v\}=\{x,y\}$ , Alice already wins. If $u\notin\{x,y\}$ and $v\in\{x,y\}$ , Alice also wins in the next round since she can add the pebble $p_{w}$ on some vertex w adjacent to $p_{v}$ such that $\{v,w\}=\{x,y\}$ . This is a valid game rule for all model M. Next, if $u\in\{x,y\}$ and $v\notin\{x,y\}$ , Alice also wins in the next round since she can first remove the pebble $p_{v}$ and then place it on the unique vertex in $\{x,y\}\backslash\{u\}$ . This is also a valid game rule for all model M. Note the removing $p_{v}$ does not merge the connected component $\{\{x,y\}\}$ since $v\notin\{x,y\}$ . Finally, if both $u\notin\{x,y\}$ and $v\notin\{x,y\}$ , F only has one edge and Alice can clearly win. Combining these cases, we conclude that Alice can always win. □

Based on the above proposition, a game state $(u,v,Q)$ is called unreachable if any path starting from the initial state $(\emptyset,\emptyset,\{E_{F}\})$ and ending at $(u,v,Q)$ goes through some terminal state. We do not need to consider unreachable states since Alice always wins before reaching it. We next introduce an important technical concept:

Definition C.51. Given a game state graph $G^{S}$ , a state $(u,v,\{P\})$ is termed as “contracted” if for any transition $((u,v,\{P\}),(u',v',\{P'\}))\in E_{G^{S}}, P'\subset P$ . It is called strictly contracted if for any transition $((u,v,\{P\}),(u',v',\{P'\}))\in E_{G^{S}}, P'\subsetneq P$ .

We have the following result:

Lemma C.52. For any model $M \in \{Sub, L, LF, F\}$ and any graph F, if Alice can win the pebble game associated to model M on graph F, then there exists a game state graph $G^{S}$ corresponding to a winning strategy of Alice such that any reachable and non-terminal state is strictly contracted.

Proof. We first prove that there is a strategy for Alice such that any reachable and non-terminal state is contracted. Since Alice can win the pebble game, she can win at any reachable state $(u,v,Q)$ . Consider any strategy such that state $(u,v,Q)$ is not contracted. Note that the game state graph induced by all reachable states is a Directed Acyclic Graph (DAG), so we can choose the state $(u,v,Q)$ such that any path from the initial state $(\emptyset,\emptyset,\{E_{F}\})$ to $(u,v,Q)$ does not pass any intermediate state that is not contracted. Below, we will construct a new strategy that removes the state $(u,v,Q)$ (making it unreachable).

Note that we clearly have $u \neq \emptyset$ and $v \neq \emptyset$ . Without loss of generality, assume that $((u, v, \{P\}), (u, v', \{P'\}))$ is a transition such that $P' \not\subset P$ (the case of transition $((u, v, \{P\}), (u', v, \{P'\}))$ is the same by symmetry). Moreover, we can assume that $P \notin CC_{F}(\{u\})$ and $P \notin CC_{F}(\{v\})$ (in other words, both u and v are at the boundary of the connected component P). It would be easier to analyze the case where either u or v are not at the boundary. We separately consider the following cases:

1. The transition $((u,v,\{P\}),(u,v',\{P'\}))$ corresponds to Alice removing pebble $p_{v}$ and placing it on $v'$ . Let $(u_{0},v_{0},Q_{0}),\cdots,(u_{T},v_{T},Q_{T})$ be any path from the initial state $(\emptyset,\emptyset,\{E_{F}\})$ to $(u,v,\{P\})$ , and let $t\leq T$ be the maximal number such that $v_{t}\neq v$ . We will construct a new strategy for Alice as follows.

- At state $(u_t, v_t, Q_t)$ , she removes pebble $\mathfrak{p}_v$ and places it on $v'$ , yielding state $(u_t, v', \tilde{Q}_t)$ .   
- For all $t < \tilde{t} < T$ , we have $v_t = v$ . She will apply the strategy at $(u_{\tilde{t}}, v_{\tilde{t}}, Q_{\tilde{t}})$ to the state $(u_{\tilde{t}}, v', \tilde{Q}_{\tilde{t}})$ , namely, placing pebble $\mathsf{p}_w$ on vertex $u_{\tilde{t} + 1}$ , swapping pebble $\mathsf{p}_u$ with $\mathsf{p}_w$ , and leaving $\mathsf{p}_w$ outside the graph. (Note that she cannot remove pebble $\mathsf{p}_u$ first; otherwise the state $(u_{\tilde{t}}, v_{\tilde{t}}, Q_{\tilde{t}})$ will not be contracted since $u$ is at the boundary of $Q_{\tilde{t}}$ .)

It follows that $((u_{\tilde{t}}, v', \tilde{Q}_{\tilde{t}}), (u_{\tilde{t}+1}, v', \tilde{Q}_{\tilde{t}+1}))$ is a transition and $(u_{\tilde{t}}, v', \tilde{Q}_{\tilde{t}})$ is contracted.

We can repeat the above procedure for all paths from the initial state $(\emptyset,\emptyset,\{E_{F}\})$ to state $(u,v,\{P\})$ . Then, in the new strategy $(u,v,\{P\})$ will be unreachable. However, the state $(u_{t},v_{t},Q_{t})$ now may violate the condition in Lemma C.52. In this case, we can recursively apply the above procedure for state $(u_{t},v_{t},Q_{t})$ . Note that the procedure will only repeat a finite number of times, as the length of the path from the initial state to the state $(u_{t},v_{t},Q_{t})$ is strictly less than the length of the path from the initial state to the state $(u,v,\{P\})$ .

2. The transition $((u,v,\{P\}),(u,v',\{P'\}))$ corresponds to Alice placing pebble $\mathsf{p}_w$ on $v'$ , swapping $\mathsf{p}_v$ and $\mathsf{p}_w$ , and removing pebble $\mathsf{p}_w$ . In this case, it is easy to see that, if Alice just removes pebble $\mathsf{p}_v$ and places it on vertex $v'$ , all transitions starting from $(u,v,\{P\})$ does not change. Therefore, we can just invoke the previous item to construct a desired strategy.

Combining the two cases, we conclude that there is a strategy for Alice such that any reachable and non-terminal state is contracted. We next prove that any reachable and non-terminal state can be strictly contracted. Assume the result does not hold and the state $(u,v,\{P\})$ is reachable, non-terminal, but not strictly contracted. Then, there is a transition $((u,v,\{P\}), (u',v',\{P\})) \in E_{G^s}$ . Consider the following two cases:

1. $P \notin \mathsf{CC}_F(\{u\})$ and $P \notin \mathsf{CC}_F(\{v\})$ (i.e., both $u$ and $v$ are at the boundary of the connected component $P$ ). It follows that $u' = u$ and $v' = v$ . This implies that the game state graph is not acyclic, a contradiction.   
2. $P \in \mathsf{CC}_F(\{u\})$ or $P \in \mathsf{CC}_F(\{v\})$ . Without loss of generality, assume that $P \in \mathsf{CC}_F(\{u\})$ . Since $((u,v,\{P\})$ is non-terminal and not strictly contracted, there is a reachable and non-terminal state $(\tilde{u},\tilde{v},\{P\})$ with either $\tilde{u} = u$ or $\tilde{v} = u$ such that there is a path from $(u,v,\{P\})$ to $(\tilde{u},\tilde{v},\{P\})$ and $(\tilde{u},\tilde{v},\{P\})$ is strictly contracted. We can then change the strategy at state $(u,v,\{P\})$ to make it strictly contracted. Concretely, Alice can remove pebble $\mathfrak{p}_v$ (which does not merge connected components selected by Bob since $P \in \mathsf{CC}_F(\{u\})$ ), and place pebble $\mathfrak{p}_v$ on the vertex that corresponds to the strategy at state $(\tilde{u},\tilde{v},\{P\})$ (possibly with the difference that the roles of $u$ and $v$ are exchanged). This makes the state $((u,v,\{P\})$ strictly contracted.

Combining the two cases concludes the proof.

![](images/d58600ffbd7253e20dfefef6ea5d34a119d4b31cdac89ccc28fce351927ffc4d.jpg)

We are now ready to state the main theorem:

Theorem C.53. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. Given any connected graph $F$ , if Alice can win the pebble game associated with model $M$ on graph $F$ , then there is a tree decomposition $T^r$ of $F$ such that $(F, T^r) \in S^M$ , where $S^M$ is defined in Definition C.6.

Proof. Let $G^{S}$ be the game state graph satisfying Lemma C.52. For each game state s, denote by $\text{next}_{G^{S}}(s)$ the set of states $s'$ such that $(s, s')$ is a transition in $G^{S}$ and $s'$ contains only a single connected component, i.e., $s'$ has the form $(u, v, \{P\})$ . For a terminal state s of the form $(u, v, \{\{x, y\}\})$ , define $\text{next}_{G^{S}}(s) = \{(x, y, \{\{x, y\}\})\}$ . By definition, $\text{next}_{G^{S}}(\emptyset, \emptyset, \{E_{F}\}) = \{(u, \emptyset, Q_{1}), \cdots, (u, \emptyset, Q_{m})\}$ for some $u \in V_{F}$ and $Q_{1}, \cdots, Q_{m}$ is the finest partition of $\mathbb{C}\mathbb{C}_{F}(\{u\})$ .

The tree $T^{r}$ will be recursively constructed as follows. First create the tree root r with $\beta_{T}(r) = \{u, u\}$ . As we will see later, the root node will be associated with the set of states $S(r) := \text{next}_{G^{\text{s}}}(\emptyset, \emptyset, \{E_{F}\})$ . We then do the following procedure:

Let $t$ be a leaf node in the current tree associated with a non-empty set of game states $S(t)$ such that $|\bigcup_{(u,v,\{P\})\in S(t)}P| > 1$ . For each state $(u,v,\{P\})\in S(t)$ , create a new node $\tilde{t}$ and set its parent to be $t$ . Pick any state $(u',v',\{P'\})\in \mathrm{next}_{G^s}(u,v,\{P\})$ . Then, there must be a unique vertex $w\in \{\{u',v'\}\setminus \{\{u,v\}\text{ and }w\text{ does not depend on which } (u',v',\{P'_i\})\text{ is picked (by definition of the game rule). Set } \beta_T(\hat{t}) = \beta_T(t)\cup \{\{w\}\}$ . Then, do the following constructions:

- If there is a state of the form $(u, w, \{P'\}) \in \mathrm{next}_{G^s}(u, v, \{P\})$ , then create a new node $t'$ and connect it to the parent $\tilde{t}$ . Set $\beta_T(t') = \{\{u, w\}\}$ , and the node $t'$ will be associated to the set of states $S(t') = \{(u, w, \{P_i'\}) : (u, w, \{P_i'\}) \in \mathrm{next}_{G^s}(u, v, \{P_i\})\}$ .   
- If there is a state of the form $(w, v, \{P''\}) \in \mathrm{next}_{G^s}(u, v, \{P\})$ , then create a new node $t''$ and connect it to the parent $\tilde{t}$ . Set $\beta_T(t'') = \{\{w, v\}\}$ , and the node $t''$ will be associated to the set of states $S(t'') = \{(w, v, \{P''\}) : (w, v, \{P''\}) \in \mathrm{next}_{G^s}(u, v, \{P\})\}$ .

Note that (i) either $w \neq u$ or $w \neq v$ (since the game state graph is a DAG); (ii) both items can be used for one state $(u, v, \{P\})$ , which happens for FWL-type GNNs. The construction of the tree completes when each leaf node is associated with only one game state of the form $(x, y, \{\{x, y\}\})$ . It is easy to see that the above procedure terminates after adding a finite number of tree nodes.

We now prove that $T^r$ is a canonical tree decomposition of $F$ and $(F, T^r) \in S^M$ .

1. We first prove that any edge in $F$ is contained in some bag of $T^r$ . Pick any non-leaf tree node $t$ of even depth and any of its child $\tilde{t}$ , denote $(u, v, P) \in S(t)$ be the state associated with $\tilde{t}$ in the above construction, and denote $\beta_T(\tilde{t}) = \{\{u, v, w\}\}$ . We have

$$
P = \left(\bigcup_{\substack{t^{\prime},\mathsf{pa}_{T^{r}}(t^{\prime}) = \tilde{t},\\ (x,y,\{P^{\prime}\})\in S(t^{\prime})}}P^{\prime}\right)\cup \{\{u,w\} \in E_{F}\} \cup \{\{v,w\} \in E_{F}\} .
$$

Here, $P' \subset P$ is a consequence of Lemma C.52, and the remaining edges $\{u, w\}$ or $\{v, w\}$ are in P because the state $(u, v, P)$ is strictly contracted and w must be in the interior of P. Therefore,

$$
P\cup \{\{u,v\} \in E_{F}\} = \left(\bigcup_{\substack{t^{\prime},\mathsf{pa}_{T^{r}}(t^{\prime}) = \tilde{t},\\ (x,y,\{P^{\prime}\})\in S(t^{\prime})}}(P^{\prime}\cup \{\{x,y\} \in E_{F}\})\right)\cup \{\{x,y\} \in E_{F}:x,y\in \beta_{T}(\tilde{t})\} .
$$

Recursively applying the above equation yields the desired result

$$
E _ {F} = \bigcup_ {\tilde {t}, \mathsf {d e p} _ {T} (\tilde {t}) \text {is odd}} \{\{x, y \} \in E _ {F}: x, y \in \beta_ {T} (\tilde {t}) \},
$$

because (i) $\bigcup_{(u,\emptyset,P)\in S(r)}P = E_{F}$ , and (ii) all leaf node t with $S(t) = (x,y, \{\{x,y\}\})$ contains an edge $\{x,y\}$ in its bag.

2. We next prove that $T^r$ satisfies the condition of Definition C.1(c). Fix any vertex $w \in V_F$ , and let $t$ be the tree node with minimal depth that contains $w$ . Without loss of generality, assume that $t$ is not the root. In this case, the depth of $t$ is odd and $\beta_T(t) = \{u, v, w\}$ for some $u \neq w, v \neq w$ . Let $t'$ be a child node of $t$ and we have $w \in \beta_T(t')$ . It remains to prove that for any descendant $\tilde{t} \in \mathrm{Desc}_T(t')$ , if $w \in \beta_T(\tilde{t})$ , then $w \in \beta_T(\hat{t})$ for all $\hat{t}$ on the path between $t'$ and $\tilde{t}$ . This is actually a direct consequence of Lemma C.52, because when a pebble originally placed on $w$ is removed, all edges linked to $w$ will not be selected by Bob and thus any pebble can never be placed on $w$ again.   
3. $T^r$ is canonical as Definition C.4 is clearly satisfied.   
4. Finally, it is also easy to see that $(F, T^r)$ satisfies Definition C.6.

We thus conclude the proof.

![](images/dc21480e53669a8bfc9c1ec203dd4cd9bab04406aa85887a2590bafc2862bbac.jpg)

Corollary C.54. Let $M \in \{Sub, L, LF, F\}$ be any model. For any connected graph $F \notin F^{M}$ , let $G(F)$ and $H(F)$ be the Fürer graph and twisted Fürer graph with respect to F. Then, $\hom(F, G(F)) \neq \hom(F, H(F))$ and $\chi_{G(F)}^{M}(G(F)) = \chi_{H(F)}^{M}(H(F))$ .

Proof. The proof directly follows from applying Theorems C.47 and C.53 and Proposition C.49.

![](images/3f280c395765b9c0bbe508633e363c1e95d9d98cafd49b7ad28e92f3d2f5a3e8.jpg)

Finally, we remark that our construction can be easily generalized for disconnected graphs $F \notin F^{M}$ . Let F be the disjoint union of graphs $\{F_{i} : i \in [m]\}$ where each $F_{i}$ the graph corresponding to a connected component of F. Assume that $F_{1}$ is the connected component with the most number of edges (in case of a tie, pick the graph with the most number of vertices). Define $\tilde{G}(F)$ be the disjoint union of $G(F_{1}), F_{2}, \cdots, F_{m}$ and $\tilde{H}(F)$ to be the disjoint union of $H(F_{1}), F_{2}, \cdots, F_{m}$ . It follows that $\chi_{\tilde{G}(F)} \tilde{G}(F) = \chi_{\tilde{H}(F)} \tilde{H}(F)$ and

$$
\begin{array}{l} \hom (F, \tilde {G} (F)) = \prod_ {i \in [ m ]} \hom (F _ {i}, \tilde {G} (F)) = \prod_ {i \in [ m ]} (\hom (F _ {i}, G (F _ {1})) + \hom (F _ {i}, F _ {2} \cup \dots \cup F _ {m})) \\ > \prod_ {i \in [ m ]} (\hom (F _ {i}, H (F _ {1})) + \hom (F _ {i}, F _ {2} \cup \dots \cup F _ {m})) = \hom (F, \tilde {H} (F)), \\ \end{array}
$$

where we use the fact that $\hom(F_i, G(F_1)) = \hom(F_i, H(F_1))$ when $F_1 \notin \operatorname{Spasm}(F_i)$ (which can be easily proved following Theorem C.47). This concludes the proof of the general case.

# D NODE/EDGE-LEVEL EXPRESSIVITY

This section aims to prove Theorem 3.7. For the ease of reading, we first restate it below:

Theorem 3.7 For all model M defined in Section 2, $F_{n}^{M}$ and $F_{e}^{M}$ (except MPNN) exist. Moreover,

- MPNN: $\mathcal{F}_{\mathfrak{n}}^{\mathsf{MP}} = \{F^{w} : F \text{ is a tree}\}$ ;  
- Subgraph GNN: $\mathcal{F}_{n}^{\text{Sub}} = \{F^{w} : F \text{ has a NED with shared endpoint } w\} = \{F^{w} : F \setminus \{w\} \text{ is a forest}\},$ $\mathcal{F}_{e}^{\text{Sub}} = \{F^{wx} : F \text{ has a NED with shared endpoint } w\} = \{F^{wx} : F \setminus \{w\} \text{ is a forest}\};$   
- 2-FGNN: $\mathcal{F}_{\mathrm{n}}^{\mathrm{F}} = \{F^{w} : F \text{ has a NED where } w \text{ is an endpoint of the first ear}\}$ , $\mathcal{F}_{\mathrm{e}}^{\mathrm{F}} = \{F^{wx} : F \text{ has a NED where } w \text{ and } x \text{ are endpoints of the first ear}\}$ .

The cases of Local 2-GNN and Local 2-FGNN are similar to 2-FGNN by replacing “NED” with “strong NED” and “almost-strong NED”, respectively.

We remark that the definition of node/edge-level homomorphism expressivity involves only connected graphs for sake of simplicity. Similar to the proof of the graph-level expressivity, the proof of Theorem 3.7 consists of three parts. Among them, the proof related to tree decomposition and ear decomposition is quite similar to that of graph-level expressivity, so we only illustrate the proof sketch for clarity (Appendix D.1). However, the proof related to pebble game and counterexample graphs will require additional techniques, which is detailed in Appendix D.2.

# D.1 RELATED TO TREE DECOMPOSITION AND EAR DECOMPOSITION

We first extend several notations that are used in Appendix C.2.

Definition D.1 (Tree decomposition for rooted graphs). Given a rooted graph $G^{u}$ and tree $T^{r}$ , we say $T^{r}$ is a tree decomposition of $G^{u}$ if $T^{r}$ is a tree decomposition of G and all elements in u belongs to the root bag, i.e., $u_{i} \in \beta_{T}(r)$ for all i.

Based on this definition, given model $M$ , we can define $\mathcal{S}_{\mathfrak{n}}^{M}$ to be the family of tree-decomposed graphs $(F^u, T^r)$ such that $(F, T^r) \in \mathcal{S}^M$ ; similarly, we define $\mathcal{S}_{\mathfrak{e}}^{M}$ to be the family of tree-decomposed graphs $(F^{uv}, T^r)$ such that $(F, T^r) \in \mathcal{S}^M$ .

Definition D.2 (Bag isomorphism for rooted graphs). Given tree-decomposed graph $(F^{u}, T^{r})$ and rooted graph $G^{v}$ where u and v have equal length, a bag isomorphism from $(F^{u}, T^{r})$ to $G^{v}$ is a homomorphism f from $F^{u}$ to $G^{v}$ such that f is a bag isomorphism from $(F, T^{r})$ to G.

We can similarly define bag-isomorphism homomorphism (bIsoHom, bIsoSurj, bIsoInj), bag extension (bExt), and bag-strong surjective (bStrSurj) from tree-decomposed graph $(F^{u}, T^{r})$ to tree-decomposed graph $(\tilde{F}^{v}, \tilde{T}^{s})$ .

Definition D.3 (Generalization of Definition C.15 for rooted graphs). Let $M \in \{Sub, L, LF, F\}$ be any model. Given a rooted graph $G^{u}$ and a tree-decomposed graph $(F^{w}, T^{r})$ , define

$$
\operatorname{cnt} ^ {M} \left(\left(F ^ {w}, T ^ {r}\right), G ^ {u}\right) := \left| \left\{v \in V _ {G}: \exists D \in \mathbb {N} _ {+} \text {s.t.} \left(\left[ F _ {G} ^ {M, (D)} (u, v) \right] ^ {u}, T _ {G} ^ {M, (D)} (u, v)\right) \simeq \left(F ^ {w}, T ^ {r}\right) \right\} \right|.
$$

Given a rooted graph $G^{uv}$ and a tree-decomposed graph $(F^{wx}, T^r)$ , define

$$
\operatorname{cnt} ^ {M} \left(\left(F ^ {w x}, T ^ {r}\right), G ^ {u v}\right) := \mathbb {I} \left[ \exists D \in \mathbb {N} _ {+} \text {s.t.} \left(\left[ F _ {G} ^ {M, (D)} (u, v) \right] ^ {u v}, T _ {G} ^ {M, (D)} (u, v)\right) \simeq \left(F ^ {w x}, T ^ {r}\right) \right].
$$

Here, $\left(F_G^{M,(D)}(u,v),T_G^{M,(D)}(u,v)\right)$ is the depth-2D unfolding tree of $G$ at $(u,v)$ for model $M$ .

All the following lemmas are straightforward extensions of those in Appendix C.2.

Lemma D.4. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. For any graph $G^v$ and tree-decomposed graph $(F^u, T^r) \in S_n^M$ , we have

$$
\mathsf {b l s o} \left(\left(F ^ {u}, T ^ {r}\right), G ^ {v}\right) = \sum_ {\left(\tilde {F} ^ {w}, \tilde {T} ^ {s}\right) \in \mathcal {S} _ {\mathfrak {n}} ^ {M}} \mathsf {b l s o H o m} \left(\left(F ^ {u}, T ^ {r}\right), \left(\tilde {F} ^ {w}, \tilde {T} ^ {s}\right)\right) \cdot \mathsf {c n t} ^ {M} \left(\left(\tilde {F} ^ {w}, \tilde {T} ^ {s}\right), G ^ {v}\right).
$$

The edge-level result is similar.

Lemma D.5. Let $M \in \{\mathrm{Sub}, \mathrm{L}, \mathrm{LF}, \mathrm{F}\}$ be any model. For any tree-decomposed graphs $(F^u, T^r), (\tilde{F}^v, \tilde{T}^s) \in \mathcal{S}_{\mathfrak{n}}^M$ ,

$$
\mathsf {b l s o H o m} ((F ^ {u}, T ^ {r}), (\tilde {F} ^ {v}, \tilde {T} ^ {s})) = \sum_ {(\widehat {F} ^ {w}, \widehat {T} ^ {t}) \in \mathcal {S} _ {\mathfrak {n}} ^ {M}} \frac {\mathsf {b l s o S u r j} \left((F ^ {u} , T ^ {r}) , (\widehat {F} ^ {w} , \widehat {T} ^ {t})\right) \cdot \mathsf {b l s o l n j} \left((\widehat {F} ^ {w} , \widehat {T} ^ {t}) , (\tilde {F} ^ {v} , \tilde {T} ^ {s})\right)}{\mathsf {a u t} (\widehat {F} ^ {w} , \widehat {T} ^ {t})}.
$$

The edge-level result is similar.

Lemma D.6. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. For any graph $G^v$ and tree-decomposed graph $(F^u, T^r) \in S_n^M$ ,

$$
\hom (F ^ {u}, G ^ {v}) = \sum_ {(\tilde {F} ^ {w}, \tilde {T} ^ {s}) \in \mathcal {S} _ {\mathrm{n}} ^ {M}} \frac {\mathrm{bExt} ((F ^ {u} , T ^ {r}) , (\tilde {F} ^ {w} , \tilde {T} ^ {s})) \cdot \mathrm{bStrHom} ((\tilde {F} ^ {w} , \tilde {T} ^ {s}) , G ^ {v})}{\mathrm{aut} (\tilde {F} ^ {w} , \tilde {T} ^ {s})}.
$$

The edge-level result is similar.

Lemma D.7. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model. For any graph $G^v$ and tree-decomposed graph $(F^u, T^r) \in \mathcal{S}_{\mathfrak{n}}^M$ ,

$$
\mathsf {b S t r H o m} ((F ^ {u}, T ^ {r}), G ^ {v}) = \sum_ {(\tilde {F} ^ {w}, \tilde {T} ^ {s}) \in \mathcal {S} _ {\mathfrak {n}} ^ {M}} \frac {\mathsf {b S t r S u r j} \left((F ^ {u} , T ^ {r}) , (\tilde {F} ^ {w} , \tilde {T} ^ {s})\right) \cdot \mathsf {b l s o} \left((\tilde {F} ^ {w} , \tilde {T} ^ {s}) , G ^ {v}\right)}{\mathsf {a u t} (\tilde {F} ^ {w} , \tilde {T} ^ {s})},
$$

The edge-level result is similar.

Corollary D.8. Let $M \in \{\mathrm{Sub}, \mathrm{L}, \mathrm{LF}, \mathrm{F}\}$ be any model. For any two graphs $G^u, H^v$ , $\chi_G^M(u) = \chi_H^M(v)$ iff $\hom(F^w, G^u) = \hom(F^w, H^v)$ for all $(F^w, T^r) \in S_n^M$ . For any two graphs $G^{uv}, H^{wx}$ , $\chi_G^M(u, v) = \chi_H^M(w, x)$ iff $\hom(F^{yz}, G^{uv}) = \hom(F^{yz}, H^{wx})$ for all $(F^{yz}, T^r) \in S_e^M$ .

Lemma D.9. Let $M \in \{\mathrm{Sub}, \mathrm{L}, \mathrm{LF}, \mathrm{F}\}$ be any model.

- For any rooted connected graph $F^u$ , $F^u \in \mathcal{F}_{\mathfrak{n}}^M$ iff there is a tree decomposition $T^r$ of $F^u$ such that $(F^u, T^r) \in \mathcal{S}_{\mathfrak{n}}^M$ .   
- For any rooted connected graph $F^{uv}$ , $F^{uv} \in \mathcal{F}_{\mathrm{e}}^{M}$ iff there is a tree decomposition $T^r$ of $F^{uv}$ such that $(F^{uv}, T^r) \in \mathcal{S}_{\mathrm{e}}^{M}$ .

By combining the proof of previous lemmas and theorems, we can prove that both $\mathcal{F}_{\mathrm{n}}^{M}$ and $\mathcal{F}_{\mathrm{e}}^{M}$ satisfy Definition 3.6(a) for all $M\in \{\mathrm{Sub},\mathsf{L},\mathsf{LF},\mathsf{F}\}$ .

# D.2 COUNTEREXAMPLES

Let $F^{\boldsymbol{w}}$ be a rooted graph that marks the special vertices $w_1, \cdots, w_m$ . Assume that $F^{\boldsymbol{w}} \notin \mathcal{F}_{\mathrm{n}}^{M} (m = 1)$ or $F^{\boldsymbol{w}} \notin \mathcal{F}_{\mathrm{e}}^{M} (m = 2)$ . In this subsection, we will construct a pair of graphs $G^{\boldsymbol{u}}$ and $H^{\boldsymbol{v}}$ such that $\chi_G^M(\boldsymbol{u}) = \chi_H^M(\boldsymbol{v})$ and $\hom(F^{\boldsymbol{w}}, G^{\boldsymbol{u}}) \neq \hom(F^{\boldsymbol{w}}, H^{\boldsymbol{v}})$ . However, it turns out that, if we naively follow the proof in Appendix C.4 by constructing the Fürer graph and twisted Fürer graph with respect to $F$ without considering the marked vertices $\boldsymbol{w}$ , then the graphs may no longer be counterexamples here. For example, for the edge-level expressivity, $F^{w_1, w_2} \notin \mathcal{F}_{\mathrm{e}}^{M}$ does not imply that $F \notin \mathcal{F}^{M}$ where $M$ can be any model studied in this paper.

To address the problem, we instead introduce a novel construction of counterexample graphs defined as follows:

Definition D.10 (Clique-augmented Fürer graphs). Let F be any connected graph and $w_{1}, \cdots, w_{m} \in V_{F}$ be a sequence of vertices. Given an integer $k \geq m$ , the k-clique-augmented Fürer graph with respect to $F^{w}$ , denoted by $G_{k}(F^{w})$ , is the Fürer graph of $\tilde{F}$ where $\tilde{F}$ is the union of graph F and a k-clique that contains $w_{1}, \cdots, w_{m}$ (and does not contain other vertices in F). The twisted Fürer graph of $G_{k}(F^{w})$ is denoted as $H_{k}(F^{w})$ .

We have the following main result:

Theorem D.11. Let $M \in \{\mathrm{Sub}, \mathsf{L}, \mathsf{LF}, \mathsf{F}\}$ be any model defined in Section 2.

- For any rooted graph $F^w$ marking vertex $w$ , if $F^w \notin \mathcal{F}_n^M$ , then there is a vertex $(w,U) \in \text{Meta}_{\tilde{F}}(w)$ such that $U \subset V_F$ and $\chi_{G_4(F^w)}^M((w,\emptyset)) = \chi_{H_4(F^w)}^M((w,U))$ ;   
- For any rooted graph $F^w$ marking two vertices $w_1, w_2$ , if $F^w \notin \mathcal{F}_{\mathrm{e}}^M$ , then there are two vertices $(w_1, U_1) \in \mathsf{Meta}_{\tilde{F}}(w_1)$ and $(w_2, U_2) \in \mathsf{Meta}_{\tilde{F}}(w_2)$ such that $U_1, U_2 \subset V_F$ and $\chi_{G_4(F^w)}^M((w_1, \emptyset), (w_2, \emptyset)) = \chi_{H_4(F^w)}^M((w_1, U_1), (w_2, U_2))$ .

Proof. Below, we only give a proof of the second item. Let $\tilde{F}$ be the union of $F$ and the 4-clique according to Definition D.10. Similar to the graph-level expressivity, we will extend the pebble game defined in Appendix C.4 to edge-level. For the edge-level pebble game, the two pebbles $\mathsf{p}_u, \mathsf{p}_v$ are initially placed on two predefined vertices $w_1, w_2$ , and Bob initially selects a predefined set of connected components $Q \subset \mathsf{CC}_{\tilde{F}}(\{w_1, w_2\})$ (of odd size). Then, the game executes the same process as in the text above Proposition C.49. The winning criterion for the two players is also the same. We denote the above edge-level pebble game as $\mathsf{PG}_{\tilde{F}}(w_1, w_2, Q)$ .

Based on the isomorphism property of Fürer graphs established in Zhang et al. (2023a, Theorem I.9 and and Theorem I.17), it is straightforward to see that, for any vertices $\xi, \xi' \in \mathsf{Meta}_{\tilde{F}}(w_1)$ and $\eta, \eta' \in \mathsf{Meta}_{\tilde{F}}(w_2)$ , $\chi_{G_4(F^w)}^M(\xi, \eta) = \chi_{H_4(F^w)}^M(\xi', \eta')$ iff Alice cannot win the edge-level pebble game $\mathsf{PG}_{\tilde{F}}(w_1, w_2, Q)$ for some $Q \subset \mathsf{CC}_{\tilde{F}}(\{w_1, w_2\})$ (depending on $\xi, \xi', \eta, \eta'$ ). Moreover, let $\xi' = (w_1, \emptyset)$ , $\eta' = (w_2, \emptyset)$ and let $\xi_2$ and $\eta_2$ range over all even-size subset of $N_F(w_1)$ and $N_F(w_2)$ , respectively, that is to say, $\xi_2$ and $\eta_2$ does not contain vertices that are only in the 4-clique. Then, the equivalent pebble game ranges over all $Q$ of odd size such that every connected component in $Q$ contains only vertices in $F$ , i.e., $Q \subset \mathsf{CC}_F(\{w_1, w_2\})$ . Therefore, if Alice cannot win the edge-level pebble game $\mathsf{PG}_{\tilde{F}}(w_1, w_2, Q)$ for some $Q \subset \mathsf{CC}_F(\{w_1, w_2\})$ , then there are two vertices

$\xi \in \mathsf{Meta}_{\tilde{F}}(w_1)$ and $\eta \in \mathsf{Meta}_{\tilde{F}}(w_2)$ such that $\xi_2, \eta_2 \subset V_F$ and $\chi_{G_4(F^w)}^M((w_1, \emptyset), (w_2, \emptyset)) = \chi_{H_4(F^w)}^M(\xi, \eta)$ .

It thus remains to prove that, if Alice can win the edge-level pebble game $\mathsf{PG}_{\tilde{F}}(w_{1},w_{2},Q)$ for all $Q\subset\mathsf{CC}_{F}(\{w_{1},w_{2}\})$ , then $F^{w}\in\mathcal{F}_{e}^{M}$ . To prove this result, we similarly define the game state graph as in Appendix C.4, with the only difference that the initial state is now $(w_{1},w_{2},Q)$ . The proof is divided into the following steps:

1. Assume that $Q$ contains a single connected component, i.e., $Q = \{P\}$ . We will prove that, if Alice can win the game $\mathsf{PG}_{\tilde{F}}(w_1, w_2, \{P\})$ , then there is a game strategy for Alice such that for any reachable and non-terminal state $(x, y, \{P'\})$ , $P' \subset P$ . If it is not the case, pick any state $(x, y, \{P'\})$ such that $P' \not\subset P$ and any path from $(w_1, w_2, \{P\})$ to $(x, y, \{P'\})$ does not pass any intermediate state that is not contracted. It follows that either $w_1$ or $w_2$ does not hold any pebble, and all edges connected to this vertex is in $P'$ . Therefore, $P'$ contains all vertices in the 4-clique. In this case, Alice cannot win as Bob can always keep the connected component containing all vertices in the 4-clique in subsequent rounds, i.e., Bob always selects a component containing a vertex in the 4-clique that does not hold a pebble.   
2. Based on the above result, in the game $\mathsf{PG}_{\tilde{F}}(w_{1},w_{2},\{P\})$ , the connected components selected by Bob are always in F, and thus the game process is the same as $\mathsf{PG}_{F}(w_{1},w_{2},\{P\})$ . We can then prove the same result as Lemma C.52. Concretely, if Alice can win the edge-level pebble game $\mathsf{PG}_{\tilde{F}}(w_{1},w_{2},\{P\})$ , then there exists a game state graph $G^{S}$ corresponding to a winning strategy, such that for any transition $((x,y,\{P'\}),(x',y',\{P''\}))$ where $(x,y,\{P'\})$ is a reachable and non-terminal state, we have $P''\subsetneq P'$ .   
3. Next, we can follow the same procedure in the proof of Theorem C.53 to construct a tree decomposition $T^r$ for the subgraph $F[P]$ containing all edges in $P$ , such that $(F[P], T^r) \in S_{\mathrm{e}}^{M}$ and $\beta_T(r) = \{\{w_1, w_2\}\}$ .   
4. Finally, let $P$ ranges over all connected component in $\mathsf{CC}_F(\{w_1, w_2\})$ , for each $P$ we have a tree-decomposed graph. We can glue these tree-decomposed graphs by merging the root to obtain the tree decomposition of $F$ , because the root bags of all $T^r$ are the same and any vertex $x \notin \{w_1, w_2\}$ appears in the bag of only one $T^r$ . Moreover, the glued tree $\tilde{T}^s$ clearly satisfies that $(F, \tilde{T}^s) \in \mathcal{S}_{\mathrm{e}}^M$ .

Combining these items shows that $F^{w} \in F_{e}^{M}$ and concludes the proof.

![](images/8e23541210d98b26e1c4f0f70652a45d59cede707b6f9f77accd4aec6938b7e6.jpg)

Corollary D.12. Let $F^w$ any rooted graph marking $m$ vertices such that $F^w \notin \mathcal{F}_n^M$ ( $m = 1$ ) or $F^w \notin \mathcal{F}_e^M$ ( $m = 2$ ), and let $G_k(F^w)$ and $H_k(F^w)$ be clique-augmented Fürer graphs defined above. Denote by $\xi, \eta \in V_{G_k(F^w)}^m$ two vertex tuples of length $m$ where $\xi_i = (w_i, \emptyset)$ and $\eta_i \in \text{Meta}_{\tilde{F}}(w_i)$ with $\eta_{i,2} \subset V_F$ for $i \in [m]$ . Then, $\text{hom}(F^w, [G_k(F^w)]^\xi) \neq \text{hom}(F^w, [H_k(F^w)]^\eta)$ ).

Proof. The proof exactly parallels that of Theorem C.47 by separately considering three cases. We omit the proof for clarity. $\square$

Corollary D.13. Let $M \in \{Sub, L, LF, F\}$ be any model.

- For any graph $F^w \notin \mathcal{F}_n^M$ , let $G_4(F^w)$ and $H_4(F^w)$ be the clique-augmented Fürer graph and the corresponding twisted Fürer graph defined in Definition D.10. Then, there exists two vertices $\xi, \xi'$ such that $\hom(F^w, [G_4(F^w)]^\xi) \neq \hom(F^w, [H_4(F^w)]^\xi')$ and $\chi_{G_4(F^w)}^M(\xi) = \chi_{H_4(F^w)}^M(\xi')$ .   
- For any graph $F^{wx} \notin \mathcal{F}_{\mathrm{e}}^{M}$ , let $G_4(F^{wx})$ and $H_4(F^{wx})$ be the clique-augmented Fürer graph and the corresponding twisted Fürer graph defined in Definition D.10. Then, there exists four vertices $\xi, \eta, \xi', \eta'$ such that $\hom(F^{wx}, [G_4(F^{wx})]^{\xi\eta}) \neq \hom(F^{wx}, [H_4(F^{wx})]^{\xi' \eta'})$ and $\chi_{G_4(F^{wx})}^M(\xi, \eta) = \chi_{H_4(F^{wx})}^M(\xi', \eta')$ .

Proof. The proof directly follows from Corollary D.12 and Theorem D.11.

![](images/37db3945ecc71d6531b63aa2a3e8dedb287958cd59065afddb1cf87c00004339.jpg)

# E HIGHER-ORDER GNNs

# E.1 DEFINITION OF HIGHER-ORDER GNNs

In this subsection, we give formal definitions of the CR algorithms for various higher-order GNNs, each of which generalizes a model in Section 2.

\- Subgraph $k$ -GNN. In a Subgraph $k$ -GNN, a graph $G$ is treated as a set of subgraphs $\{G^{\boldsymbol{u}} : \boldsymbol{u} \in V_{G}^{k}\}$ , where each subgraph $G^{\boldsymbol{u}}$ is obtained from $G$ by marking $k$ special vertices $\boldsymbol{u} \in V_{G}^{k}$ , and thus there are $n^{k}$ subgraphs when $G$ has $n$ vertices. Subgraph GNN maintains a color $\chi_{G}^{\text{Sub}(k)}(\boldsymbol{u}, v)$ for each vertex $v$ in graph $G^{\boldsymbol{u}}$ . Initially, $\chi_{G}^{\text{Sub}(k),(0)}(\boldsymbol{u}, v) = (\ell_{G}(v), \mathsf{atp}_{G}(\boldsymbol{u}), \mathbb{I}[u_{1} = v], \cdots, \mathbb{I}[u_{k} = v])$ . It then runs MPNNs independently on each graph $G^{\boldsymbol{u}}$ :

$$
\chi_ {G} ^ {\operatorname{Sub} (k), (t + 1)} (\boldsymbol {u}, v) = \mathsf {h a s h} \left(\chi_ {G} ^ {\operatorname{Sub} (k), (t)} (\boldsymbol {u}, v), \{\{\chi_ {G} ^ {\operatorname{Sub} (k), (t)} (\boldsymbol {u}, w): w \in N _ {G} (v) \} \}\right). \tag {26}
$$

Denote the stable color as $\chi_G^{\mathrm{Sub}(k)}(\boldsymbol{u},v)$ . Define $\chi_G^{\mathrm{Sub}(k)}(\boldsymbol{u}):= \{\{\chi_G^{\mathrm{Sub}(k)}(\boldsymbol{u},v):v\in V_G\}\}$ . Then, the graph representation is defined as $\chi_G^{\mathrm{Sub}(k)}(G) = \{\{\chi_G^{\mathrm{Sub}(k)}(\boldsymbol{u}):\boldsymbol{u}\in V_G\}\}$ . We remark that Subgraph $k$ -GNN is precisely the $k$ -VSAN proposed in Qian et al. (2022).

\- Local $k$ -GNN. Local $k$ -GNN maintains a color $\chi_G^{\mathsf{L}(k)}(\boldsymbol{u})$ for each vertex $k$ -tuple $\boldsymbol{u} \in V_G^k$ . Initially, $\chi_G^{\mathsf{L}(k),(0)}(\boldsymbol{u}) = (\ell_G(u_1), \cdots, \ell_G(u_k), \mathrm{atp}_G(\boldsymbol{u}))$ , called the isomorphism type of vertex $k$ -tuple $\boldsymbol{u}$ , where $\mathrm{atp}_G(\boldsymbol{u})$ is the atomic type of $\boldsymbol{u}$ . Then, in each iteration $t + 1$ ,

$$
\begin{array}{l} \chi_ {G} ^ {\mathrm{L} (k), (t + 1)} (\boldsymbol {u}) = \operatorname{hash} \left(\chi_ {G} ^ {\mathrm{L} (k), (t)} (\boldsymbol {u}), \{\{\chi_ {G} ^ {\mathrm{L} (k), (t)} (\boldsymbol {v}): \boldsymbol {v} \in N _ {G} ^ {(1)} (\boldsymbol {u}) \} \}, \dots , \right. \tag {27} \\ \left. \{\{\chi_ {G} ^ {\mathsf {L} (k), (t)} (\boldsymbol {v}): \boldsymbol {v} \in N _ {G} ^ {(k)} (\boldsymbol {u}) \} \}\right), \\ \end{array}
$$

where $N_{G}^{(j)}(\boldsymbol{u}) = \{(u_{1}, \cdots, u_{j-1}, w, u_{j+1}, \cdots, u_{k}) : w \in N_{G}(u_{j})\}$ . Denote the stable color as $\chi_{G}^{\mathsf{L}(k)}(\boldsymbol{u})$ . The graph representation is defined as $\chi_{G}^{\mathsf{L}(k)}(G) := \{\{\chi_{G}^{\mathsf{L}(k)}(\boldsymbol{u}) : \boldsymbol{u} \in V_{G}\}\}$ . We remark that Local k-GNN is precisely the $\delta$ -k-LWL proposed in Morris et al. (2020).

\- Local $k$ -FGNN. Local $k$ -FGNN is almost the same as Local $k$ -GNN expect that the update formula is replaced by the following one:

$$
\begin{array}{l} \chi_ {G} ^ {\mathsf {L F} (k), (t + 1)} (\boldsymbol {u}) = \text {hash} \left(\chi_ {G} ^ {\mathsf {L F} (k), (t)} (\boldsymbol {u}), \{\{\chi_ {G} ^ {\mathsf {L F} (k), (t)} (w, u _ {2}, \dots , u _ {k}), \chi_ {G} ^ {\mathsf {L F} (k), (t)} (u _ {1}, w, u _ {3}, \dots , u _ {k}), \right.\left. \right. \\ \left. \dots , \chi_ {G} ^ {\mathsf {L F} (k), (t)} \left(u _ {1}, \dots , u _ {k - 1}, w)\right): w \in N _ {G} \left(u _ {1}\right) \cup \dots \cup N _ {G} \left(u _ {k}\right) \} \right\} \Bigg). \tag {28} \\ \end{array}
$$

We remark that Local k-FGNN is precisely the SLFWL(k) proposed in Zhang et al. (2023a).

\- $k$ -FGNN. It is just the standard $k$ -FWL (Cai et al., 1992). Compared with Local $k$ -FGNN, the update formula is now global:

$$
\begin{array}{l} \chi_ {G} ^ {\mathsf {F} (k), (t + 1)} (\boldsymbol {u}) = \mathsf {h a s h} \left(\chi_ {G} ^ {\mathsf {F} (k), (t)} (\boldsymbol {u}), \{\{(\chi_ {G} ^ {\mathsf {F} (k), (t)} (w, u _ {2}, \dots , u _ {k}), \chi_ {G} ^ {\mathsf {F} (k), (t)} (u _ {1}, w, u _ {3}, \dots , u _ {k}), \right.\left. \right. \\ \left. \dots , \chi_ {G} ^ {\mathsf {F} (k), (t)} \left(u _ {1}, \dots , u _ {k - 1}, w)\right): w \in V _ {G} \} \}\right). \tag {29} \\ \end{array}
$$

We note that the computational complexity of Subgraph $(k-1)$ -GNN, Local k-GNN, and Local k-FGNN is the same, i.e., $\Theta(n^{k-1}m)$ for a graph of n vertices and m edges. Moreover, it is easy to see that Subgraph 0-GNN, Local 1-GNN, and Local 1-FGNN all reduce to MPNN. For k-FGNN, the computational complexity is $\Theta(n^{k+1})$ , which is strictly higher than Subgraph $(k-1)$ -GNN, Local k-GNN, and Local k-FGNN.

# E.2 HIGHER-ORDER STRONG NED

In this subsection, we generalize the strong NED into higher-order versions. We first define the concept of higher-order ear:

Definition E.1 (k-order ear). Given integer $k \geq 1$ , a k-order ear is a graph G formed by the union of k paths $P_{1}, \cdots, P_{k}$ (possibly with zero length) plus an edge set Q satisfying the following conditions:

- For each path $P_i$ , denote its two endpoints as $u_i, v_i$ , called the outer endpoint and inner endpoint, respectively. Then, all edges in $Q$ are linked between inner endpoints, i.e., $Q \subset \{\{v_i, v_j\} : 1 \leq i, j \leq k, v_i \neq v_j\}$ .   
- Any different paths $P_{i}, P_{j}$ do not intersect except at the inner endpoint (when $v_{i} = v_{j}$ ).   
- $G$ is a connected graph.

The endpoints of the k-order ear is defined to be all outer endpoints $u_{1}, \cdots, u_{k}$ .

It is easy to see that a 2-order ear is precisely a simple path, since linking two different paths (possibly with an additional edge) still yields a path. Below, we denote by inner $(G)$ and outer $(G)$ the set of inner endpoints and outer endpoints in ear $G$ , respectively. We also denote by path $(G)$ the set of paths in ear $G$ . It follows that $|\text{inner}(G)| = |\text{outer}(G)| = |\text{path}(G)| = k$ . Finally, given a path $P$ and two vertices $w_{1}, w_{2}$ in $P$ , denote by subpath $P(w_{1}, w_{2})$ the subpath in $P$ such that the two endpoints are $w_{1}, w_{2}$ .

Definition E.2 (Nested interval). Let G and H be two k-order ears with inner $(G)=\{v_{1},\cdots,v_{k}\}$ , outer $(G)=\{u_{1},\cdots,u_{k}\}$ , and outer $(H)=\{w_{1},\cdots,w_{k}\}$ , where each $\{u_{i},v_{i}\}$ corresponds to the endpoints of a path $P_{i}\in\text{path}(G)$ . We say H is nested on G if one or more endpoint $w_{i}$ of H ( $i\in[k]$ ) is in path $P_{i}$ , and all other vertices in H are not in G. The nested interval is defined to be the union of subpaths subpath $_{P_{i}}(w_{i},v_{i})$ for all $i\in[k]$ satisfying that $w_{i}$ is in $P_{i}$ .

We give an illustration of the nested interval of two 3-order ears in Figure 2. Equipped with the above definition, we are ready to introduce the higher-order strong NED:

Definition E.3 (k-order strong NED). Given a graph G, a k-order strong NED P is a partition of the edge set $E_{G}$ into a sequence of edge sets $Q_{1}, \cdots, Q_{m}$ , which satisfies the following conditions:

• Each $Q_{i}$ is a k-order ear.   
- Any two ears $Q_{i}$ and $Q_{j}$ with indices $1 \leq i < j \leq c$ do not intersect, where $c$ is the number of connected components of $G$ .   
- For each $Q_{j}$ with index $j > c$ , it is nested on some $k$ -order ear $Q_{i}$ with index $1 \leq i < j$ . Moreover, except for the endpoints of $Q_{j}$ on $Q_{i}$ , no other vertices in $Q_{j}$ are in any previous ear $Q_{k}$ for $1 \leq k < i$ .   
- Denote by $I(Q_{j}) \subset Q_{i}$ the nested interval of $Q_{j}$ in $Q_{i}$ . For all $Q_{j}, Q_{k}$ with $c < j < k \leq m$ , if $Q_{j}$ and $Q_{k}$ are nested on the same ear, then $I(Q_{j}) \subset I(Q_{k})$ .

# E.3 PROOFS IN SECTION 3.4

We first generalize the tree-decomposed graphs in Definition C.6 to higher-order versions:

Definition E.4. Define four families of tree-decomposed graphs $\mathcal{S}^{\mathrm{Sub}(k)}$ , $\mathcal{S}^{\mathrm{L}(k)}$ , $\mathcal{S}^{\mathrm{LF}(k)}$ , and $\mathcal{S}^{\mathrm{F}(k)}$ as follows:

a) $(F, T^r) \in S^{\mathsf{F}(k)}$ iff $(F, T^r)$ satisfies Definition C.4 with width $k$ ;   
b) $(F,T^r)\in S^{\mathsf{LF}(k)}$ iff $(F,T^r)$ satisfies Definition C.4 with width $k$ , and for any tree node $t$ of odd depth, it has only one child if $w\notin \{v:v\in N_G(u),u\in \beta_T(s)\}$ where $s$ is the parent node of $t$ and $w$ is the unique vertex in $\beta_T(t)\backslash \beta_T(s)$ ;   
c) $(F,T^r)\in S^{\mathsf{L}(k)}$ iff $(F,T^r)$ satisfies Definition C.4 with width $k$ , and any tree node $t$ of odd depth has only one child;   
d) $(F,T^r)\in S^{\mathrm{Sub}(k)}$ iff $(F,T^r)$ satisfies Definition C.4 with width $k$ , and there exists a multiset $U\subset V_{G}$ of size $|U| = k$ such that $U\subset \beta_T(t)$ for all $t\in V_T$ .

Then, we can analogously prove the following theorems. The proofs are almost the same as in Appendix C, so we omit them for clarity.

Theorem E.5. Let $M \in \{\mathsf{Sub}(k), \mathsf{L}(k), \mathsf{LF}(k), \mathsf{F}(k)\}$ . Then, any graphs $G$ and $H$ have the same representation under model $M$ (i.e., $\chi_G^M(G) = \chi_H^M(H)$ ) iff $\mathsf{hom}(F,G) = \mathsf{hom}(F,H)$ for all $(F,T^r) \in S^M$ .

Theorem E.6. Let $M \in \{\mathsf{Sub}(k), \mathsf{L}(k), \mathsf{LF}(k), \mathsf{F}(k)\}$ be any model, and let $F$ be any graph such that no tree decomposition $(F, T^r) \in S^M$ . Let $G(F)$ and $H(F)$ be the Fürer graph and twisted Fürer graph with respect to $F$ . Then, $\hom(F, G(F)) \neq \hom(F, H(F))$ and $\chi_{G(F)}^M(G(F)) = \chi_{H(F)}^M(H(F))$ .

Theorem E.7. For any graph $F$ , there is a tree decomposition $T^{r}$ of $F$ such that $(F, T^{r}) \in S^{\operatorname{Sub}(k)}$ iff there exists $U \subset V_{F}$ such that $|U| \leq k$ and $F \backslash U$ is a forest.

Theorem E.8. For any graph $F$ , there is a tree decomposition $T^r$ of $F$ such that $(F, T^r) \in \mathcal{S}^{\mathsf{L}(k)}$ iff $F$ has a $k$ -order strong NED.

Theorem E.9. For any graph $F$ , there is a tree decomposition $T^{r}$ of $F$ such that $(F, T^{r}) \in \mathcal{S}^{\mathsf{F}(k)}$ iff $\operatorname{tw}(F) \leq k$ .

# E.4 EXPRESSIVITY GAP BETWEEN HIGHER-ORDER GNNs

In this subsection, we show how homomorphism expressivity can be used to build a complete expressiveness hierarchy for higher-order GNNs as shown in Corollary 4.2.

Gap between Subgraph k-GNN and Local $(k+1)$ -GNN $(k\geq1)$ . The counterexample graph is a $(k+1)\times(2k+2)$ grid consisting $(k+1)\times(2k+2)$ vertices. First, it is easy to see that the graph is not in $\mathcal{F}^{\mathrm{Sub}(k)}$ , i.e., deleting k vertices of the graph cannot yield a forest. To see this, note that the graph consists of $k\times(2k+1)$ “squares”, and each vertex is related to at most four squares. Therefore, deleting k vertices cannot eliminate all squares when $k\geq2$ (because $4k<k\times(2k+1)$ ). For the case of k=1, we clearly have that deleting one vertex cannot eliminate all squares.

We next show that the $(k+1)\times(2k+2)$ grid is in $\mathcal{F}^{\mathrm{L}(k+1)}$ . This is also simple as shown in Figure 8, where each color represents a k-order ear. It can be seen that the $(k+1)\times(2k+2)$ grid has a $(k+1)$ -order strong NED.

Relation between Subgraph k-GNN and k-FGNN ( $k \geq 2$ ). To show that $\mathcal{F}^{\mathrm{Sub}(k)} \not\subset \mathcal{F}^{\mathrm{F}(k)}$ , consider the $(k + 2)$ -clique. Clearly, deleting k vertices from the $(k + 2)$ -clique yields a graph consisting of two vertices linked by an edge, which is a tree. On the other hand, the

![](images/1fc7eea49b2bdb60f4616dac10ab1c2c375f8196941bc5cf8c74ab0d0947c60c.jpg)

<details>
<summary>natural_image</summary>

Grid pattern with colored nodes connected by lines (no text or symbols)
</details>

Figure 8: An 4-order strong NED of the $4 \times 8$ grid graph.

treewidth of a $(k+2)$ -clique is $k+1$ (Fact C.3). So the $(k+2)$ -clique is in $\mathcal{F}^{\mathrm{Sub}(k)}$ but not in $\mathcal{F}^{\mathrm{F}(k)}$ .

To show that $\mathcal{F}^{\mathsf{F}(k)} \not\subset \mathcal{F}^{\mathsf{Sub}(k)}$ , we can again use the grid graph, but this time consider the $k \times (2k + 2)$ grid. On the one hand, a standard result in graph theory shows that the treewidth of a $a \times b$ grid graph is $\min(a, b)$ . On the other hand, following the similar analysis above, we can prove that deleting $k$ vertices from the $k \times (2k + 2)$ grid cannot eliminate all squares.

Gap between Local k-GNN and k-FGNN $(k \geq 2)$ . The counterexample graph F is the union of the following graphs $K_{0} \cup K_{1} \cup \cdots \cup K_{k}$ , where $K_{0}$ is a $(k+1)$ -clique with vertex set $\{u_{1}, \cdots, u_{k+1}\}$ , and for $i \in [k+1]$ , $K_{i}$ is a $(k+1)$ -clique with vertex set $\{u_{1}, \cdots, u_{i-1}, v_{i}, u_{i+1}, \cdots, u_{k+1}\}$ where $v_{i}$ is a new vertex (not in $K_{0}$ ) and $v_{i} \neq v_{j}$ for $i \neq j$ . Namely, each $K_{i}$ has k common vertices with $K_{0}$ . It is easy to construct a tree decomposition $T^{r}$ of F such that $(F, T^{r}) \in \mathcal{S}^{\mathsf{F}(k)}$ . On the other hand, it is easy to see that the graph does not have a k-order strong NED (equivalently, one can easily check that F does not admit a tree decomposition $T^{r}$ satisfying $(F, T^{r}) \in \mathcal{S}^{\mathsf{L}(k)}$ ).

Gap between $k$ -FGNN and Local $(k + 1)$ -GNN $(k \geq 2)$ . The counterexample graph is again the $(k + 1) \times (2k + 2)$ grid. We have proved that the graph is in $\mathcal{F}^{\mathsf{L}(k + 1)}$ but not in $\mathcal{F}^{\mathsf{L}(k)}$ .

Regarding Local k-IGN and Frasca et al. (2022). Finally, we remark that based on the above results, we essentially proved an open question raised in Frasca et al. (2022) regarding the expressive power of Local k-IGN (i.e., the $\operatorname{RelGN}(k)$ proposed in their paper). Following Zhang et al. (2023a), it is straightforward to see that Local k-IGN is as expressive as Local k-GNN. Therefore, Local k-IGN is strictly more expressive than $(k - 1)$ -FGNN and strictly less expressive than k-FGNN.

![](images/20925fb5d759cfd0c945bead2f6687e79b75bee4c4d0b7e6ed43480c1324046f.jpg)  
(a)

![](images/32baea6635c2f53cbb33a4ba911940f0014722bec72712254cffddef67b00b29.jpg)  
(b)

![](images/48d6aa1172ec1fe64792a1f56fe66ef4542b5240f5016da6c169a6b4e888155a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["5"] --> B["3"]
    A["5"] --> C["4"]
    B["3"] --> D["2"]
    B["3"] --> E["1"]
    C["4"] --> D["2"]
    C["4"] --> E["1"]
    D["2"] --> E["1"]
    E["1"] --> F["6"]
    F["6"] --> G["7"]
    G["7"] --> H["8"]
    H["8"] --> I["8"]
    D["2"] --> F["6"]
    F["6"] --> H["8"]
```
</details>

(c)   
Figure 9: More illustration of NED. All NEDs in these example graphs are strong but not endpoint-shared. See Appendix F.2 for additional discussions.

# F ADDITIONAL DISCUSSIONS

# F.1 REGARDING THE DEFINITION OF HOMOMORPHISM EXPRESSIVITY

In this paper, we have shown that homomorphism expressivity exists for a variety of popular GNNs. Unfortunately, due to the “iff” statement in Definition 3.1, homomorphism expressivity may not always be well-defined in the general case. We remark in this subsection that, there do exist pathological, intentionally designed GNNs such that the homomorphism expressivity is not well-defined.

Consider a simple GNN M that outputs the representation of a graph G as follows. If G is a cycle of odd length, it outputs $(1, L)$ where L is the length of the cycle. Otherwise, it outputs $(0, \chi_{G}^{\mathsf{MP}}(G))$ , namely, running a MPNN on the graph. It follows that M is strictly more powerful than MPNN, e.g., it can distinguish between the 9-cycle and three triangles, which MPNN fails to distinguish. As a result, M can count all trees under homomorphism. Moreover, it cannot count other patterns under homomorphism, as the counterexample graphs for MPNN (i.e., Fürer graphs) are not cycles of odd length, so they are still counterexample graphs for model M. Therefore, the homomorphism expressivity of M should be exactly the family of forests if it exists. However, the homomorphism information of forests cannot determine the representation of model M since M is strictly more powerful than MPNN. So we conclude that $F^{M}$ does not exist.

Note that the above GNN construction is inherently unnatural and hardly appears in practice. Actually, when a GNN is defined via the message-passing paradigm, we suspect that its homomorphism expressivity is always well-defined. As stated in Appendix B, we conjecture that, for any GNN characterized by a color refinement algorithm that outputs stable colors, the homomorphism expressivity always exists.

# F.2 REGARDING THE DEFINITION OF NED

In the main text, we have illustrated several types of NED with simple example graphs. To gain a deeper understanding of Definition 3.3, in this section we will present a few more complex examples (see Figure 9). Notably, unlike graphs in Figure 1(b), for all graphs in Figure 9 their NED contains ears such that only one endpoint is in its nested ear (e.g., ears 2 and 4 in Figure 9(a)). In other words, these ears have empty nested intervals.

The presence of ears with empty nested interval stems from the fact that the corresponding graph is not biconnected (Zhang et al., 2023b). Indeed, one can check that all graphs in Definition 3.3(b) do not have cut vertices, while all graphs in Figure 9 have cut vertices. Moreover, in these examples, the number of ears with empty nested interval always equals to the number of biconnected components minus one. In particular, for biconnected graphs, it can prove that any NED does not contain an ear with empty nested interval.

# G HOMOMORPHISM AND SUBGRAPH COUNTING POWER

# G.1 PROOF OF THEOREM 4.5

Our proof draws inspiration from a recent technique developed in Seppelt (2023). We note that the original results in Seppelt (2023) are described for unlabeled graphs, but it is easy to extend these results to labeled graphs. To begin with, we define a concept called the graph categorical product.

Definition G.1. The categorical product of two graphs $G, H$ , denoted as $G \times H$ , is a graph where $V_{G \times H} = V_G \times V_H$ , $\ell_{G \times H}(g, h) = (\ell_G(g), \ell_H(h))$ for all $g \in V_G$ , $h \in V_H$ , and $E_{G \times H} = \{ (g, h), (g', h')\} : \{ g, g' \} \in E_G, \{ h, h' \} \in E_H \}$ .

Lemma G.2 (Seppelt (2023)). For any graphs $F, G$ , and $H$ , $\operatorname{hom}(F, G \times H) = \operatorname{hom}(F, G) \cdot \operatorname{hom}(F, H)$ .

Proof. We define a mapping $\tau$ from $\operatorname{Hom}(F,G)\times\operatorname{Hom}(F,H)$ to $\operatorname{Hom}(F,G\times H)$ as follows. for all $\sigma_{1}\in\operatorname{Hom}(F,G)$ and $\sigma_{2}\in\operatorname{Hom}(F,H)$ , define $\rho=\tau(\sigma_{1},\sigma_{2})$ where $\rho(f)=(\sigma_{1}(f),\sigma_{2}(f))$ for all $f\in V_{F}$ . It is easy to see that $\tau$ is a bijective mapping from $\operatorname{Hom}(F,G)\times\operatorname{Hom}(F,H)$ to $\operatorname{Hom}(F,G\times H)$ . ☐

Definition G.3. Given graphs F and G, denote by $\operatorname{Surj}(F,G)$ the set of all homomorphisms from F to G that are surjective on both the vertices and edges of G, and define $\operatorname{surj}(F,G)=|\operatorname{Surj}(F,G)|$ .

The following proposition is straightforward (similar to Lemma C.25):

Proposition G.4. $\operatorname{hom}(G, H) = \sum_{F} \operatorname{surj}(G, F) \cdot \operatorname{sub}(F, H)$ , where $F$ ranges over all non-isomorphic graphs.

Lemma G.5. Let $M$ be a GNN model such that its homomorphism expressivity $\mathcal{F}^M$ exists. Given a finite set of graphs $\mathcal{L}$ and a function $\alpha : \mathcal{L} \to \mathbb{R}/\{0\}$ , if

$$
\chi_ {G} ^ {M} (G) = \chi_ {H} ^ {M} (H) \implies \sum_ {L \in \mathcal {L}} \alpha (L) \hom (L, G) = \sum_ {L \in \mathcal {L}} \alpha (L) \hom (L, H),
$$

holds for all graphs $G$ and $H$ , then $\mathcal{L} \subset \mathcal{F}^M$ .

Proof. Let n be the largest number of vertices for all graphs in L, and let $\tilde{L}$ be the set of all non-isomorphic graphs with no more than n vertices. We can arrange all graphs in $\tilde{L}$ into a sequence $L_{1}, L_{2}, \ldots, L_{N}$ satisfying the following property: $|V_{L_{i}}| \leq |V_{L_{i+1}}|$ or $(|V_{L_{i}}| = |V_{L_{i+1}}|$ and $|E_{L_{i}}| \leq |E_{L_{i+1}}|)$ for all $i = 1, 2, \ldots, N - 1$ . We then define the matrices $A^{hom}$ , $A^{surj}$ , and $A^{sub}$ , where the elements in the $i^{th}$ row and $j^{th}$ column are $\text{hom}(L_{i}, L_{j})$ , $\text{surj}(L_{i}, L_{j})$ , and $\text{sub}(L_{i}, L_{j})$ , respectively. Proposition G.4 implies that $A^{hom} = A^{surj} \cdot A^{sub}$ . Since $A^{surj}$ is a lower triangular matrix with nonzero diagonal elements and $A^{sub}$ is an upper triangular matrix with non-zero diagonal elements, the matrix $A^{hom}$ is invertible.

We next extend $\alpha$ to a function $\tilde{\alpha}:\tilde{\mathcal{L}}\to \mathbb{R}$ by setting $\tilde{\alpha}(L) = \alpha (L)$ for all $L\in \mathcal{L}$ and $\tilde{\alpha} (\tilde{L}) = 0$ for all $\tilde{L}\in \tilde{\mathcal{L}}\backslash \mathcal{L}$ . Additionally, if $\chi_G^M (G) = \chi_H^M (H)$ , then $\hom (F,G) = \hom (F,H)$ for all $F\in \mathcal{F}^M$ (by definition of homomorphism expressivity). Then, given any graph $K\in \tilde{\mathcal{L}}$ , Lemma G.2 implies that $\hom (F,G\times K) = \hom (F,H\times K)$ for all $F\in \mathcal{F}^{M}$ . This further implies that $\chi_{G\times K}^{M}(G\times K) = \chi_{H\times K}^{M}(H\times K)$ by definition of homomorphism expressivity. Therefore, Lemma G.2 implies that

$$
\sum_ {L \in \mathcal {L}} \alpha (L) \mathrm{hom} (L, G) \cdot \mathrm{hom} (L, K) = \sum_ {L \in \mathcal {L}} \alpha (L) \mathrm{hom} (L, H) \cdot \mathrm{hom} (L, K). \tag {30}
$$

Namely,

$$
\sum_ {L \in \tilde {\mathcal {L}}} \tilde {\alpha} (L) \mathrm{hom} (L, G) \cdot \mathrm{hom} (L, K) = \sum_ {L \in \tilde {\mathcal {L}}} \tilde {\alpha} (L) \mathrm{hom} (L, H) \cdot \mathrm{hom} (L, K). \tag {31}
$$

Now define vectors $p_{G}$ and $p_{H}$ , where the $i^{th}$ element of vector $p_{G}$ and $p_{H}$ is $\tilde{\alpha}(L_{i})\mathrm{hom}(L_{i}, G)$ and $\tilde{\alpha}(L_{i})\mathrm{hom}(L_{i}, H)$ , respectively. We then have the following equation:

$$
\boldsymbol {A} ^ {\text { hom }} \cdot \boldsymbol {p} _ {G} = \boldsymbol {A} ^ {\text { hom }} \cdot \boldsymbol {p} _ {H}.
$$

Since $A^{hom}$ is invertible, $p_{G}^{hom} = p_{H}^{hom}$ . Therefore, $\tilde{\alpha}(L)\mathrm{hom}(L,G) = \tilde{\alpha}(L)\mathrm{hom}(L,H)$ for all $L \in \tilde{L}$ , namely, $\mathrm{hom}(L,G) = \mathrm{hom}(L,H)$ for all $L \in L$ . To sum up, we have proved that for all graph G, H, $\chi_{G}^{M}(G) = \chi_{H}^{M}(H)$ implies that $\mathrm{hom}(L,G) = \mathrm{hom}(L,H)$ for all $L \in L$ . By definition of homomorphism expressivity, we conclude that $L \subset F^{M}$ .

Theorem G.6. Let $M \in \{Sub, L, LF, F\}$ be any model. For any graph F, if $Spasm(F) \setminus \mathcal{F}^{M} \neq \emptyset$ , then there exists a pair of graphs G, H such that $\chi_{G}^{M}(G) = \chi_{H}^{M}(H)$ and $\operatorname{sub}(F, G) \neq \operatorname{sub}(F, H)$ .

Proof. Recall that $\mathrm{sub}(F,G) = \sum_{\tilde{F} \in \mathrm{Spasm}^{\neq}(F)} \alpha(F, \tilde{F}) \cdot \mathrm{hom}(\tilde{F}, G)$ where $\alpha(F, \tilde{F}) \neq 0$ for all $\tilde{F} \in \mathrm{Spasm}^{\neq}(F)$ . If the above theorem does not hold, then for all graphs $G, H, \chi_G^M(G) = \chi_H^M(H)$ implies that

$$
\sum_ {\tilde {F} \in \operatorname{Spasm} ^ {\not \cong} (F)} \alpha (F, \tilde {F}) \cdot \hom (\tilde {F}, G) = \sum_ {\tilde {F} \in \operatorname{Spasm} ^ {\not \cong} (F)} \alpha (F, \tilde {F}) \cdot \hom (\tilde {F}, G).
$$

Then, Lemma G.5 implies that $\operatorname{Spasm}(F) \subset \mathcal{F}^{M}$ , yielding a contradiction.

![](images/93d5c39ec9319166d79bbd7f0f68ad6ff759c03412a4cdaae529b5ce9bf4ad33.jpg)

We next extend our analysis to the node/edge-level subgraph counting. Since the proof techniques are almost the same as the graph-level setting, we only present key definitions and lemmas below while omitting the detailed proofs for clarity.

Definition G.7. Given two rooted graphs $G^{u}$ and $H^{v}$ with $u \in V_{G}^{m}$ and $v \in V_{H}^{m}$ for some $m \in N_{+}$ , the categorical product of $G^{u}$ and $H^{v}$ , denoted as $(G \times H)^{(u_{1}, v_{1}), \cdots, (u_{m}, v_{m})}$ , is a rooted graph obtained from $G \times H$ by marking vertices $(u_{1}, v_{1}), \cdots, (u_{m}, v_{m})$ .

Lemma G.8 (Extension of Lemma G.2). For any rooted graphs $F^{\boldsymbol{w}}$ , $G^{\boldsymbol{u}}$ , $H^{\boldsymbol{v}}$ where $|\boldsymbol{u}| = |\boldsymbol{v}| = |\boldsymbol{w}| = m$ ,

$$
\hom (F ^ {\boldsymbol {w}}, (G \times H) ^ {(u _ {1}, v _ {1}), \dots , (u _ {m}, v _ {m})}) = \hom (F ^ {\boldsymbol {w}}, G ^ {\boldsymbol {u}}) \cdot \hom (F ^ {\boldsymbol {w}}, H ^ {\boldsymbol {v}}).
$$

Proposition G.9 (Extension of Proposition G.4). For any rooted graphs $G^{u}$ , $H^{v}$ where $|u| = |v| = m$ , $\hom(G^{u}, H^{v}) = \sum_{F^{w}} \text{surj}(G^{u}, F^{w}) \cdot \text{sub}(F^{w}, H^{v})$ , where $F^{w}$ ranges over all non-isomorphic rooted graphs marking m vertices.

We next present the main lemma for node-level subgraph counting. We omit the edge-level result for clarity.

Lemma G.10 (Extension of Lemma G.5). Let $M$ be a GNN model such that its node-level homomorphism expressivity $\mathcal{F}_{\mathfrak{n}}^{M}$ exists. Given a finite set of rooted graphs $\mathcal{L}_{\mathfrak{n}}$ and a function $\alpha : \mathcal{L}_{\mathfrak{n}} \to \mathbb{R} / \{0\}$ , if

$$
\chi_ {G} ^ {M} (u) = \chi_ {H} ^ {M} (v) \implies \sum_ {L ^ {w} \in \mathcal {L} _ {\mathfrak {n}}} \alpha (L ^ {w}) \mathsf {h o m} (L ^ {w}, G ^ {u}) = \sum_ {L ^ {w} \in \mathcal {L} _ {\mathfrak {n}}} \alpha (L ^ {w}) \mathsf {h o m} (L ^ {w}, H ^ {v}),
$$

holds for all rooted graphs $G^u$ and $H^v$ , then $\mathcal{L}_{\mathfrak{n}} \subset \mathcal{F}_{\mathfrak{n}}^M$ .

Theorem G.11. Let $M \in \{Sub, L, LF, F\}$ be any model. For any rooted graph $F^{w}$ , if $Spasm(F^{w}) \backslash \mathcal{F}_{n}^{M} \neq \emptyset$ , then there exist a pair of graphs G, H and vertices $u \in V_{G}$ , $v \in V_{H}$ , such that $\chi_{G}^{M}(u) = \chi_{H}^{M}(v)$ and $\text{sub}(F^{w}, G^{u}) \neq \text{sub}(F^{w}, H^{v})$ .

# G.2 GRAPH STATISTICS AND EXAMPLES

In this section, we list the statistics of all moderate-size graphs that can/cannot be counted for each model in Section 2 at graph/node/edge-level. Table 4 presents the statistics under homomorphism count, while Table 5 presents the statistics under subgraph count. These tables offer a clear picture into how large the expressivity gaps are between different models. Several important findings are discussed below:

- For homomorphism counting, graphs of 8 edges suffices to reveal the expressivity gaps between each pair of architectures (at edge-level).   
- However, for subgraph counting, moderate-size graphs cannot reveal the gap between Local 2-GNN, Local 2-FGNN, and 2-FGNN. Moreover, even Subgraph GNN already matches the power of 2-FWL in counting small subgraphs at graph-level, but it is quite weak in counting subgraphs at node/edge-level.   
- Subgraph counting is much more challenging than homomorphism counting. Intuitively, this is because the homomorphism image of a graph usually has 4-cliques, and any graph that contains a 4 clique as subgraph cannot be counted under homomorphism.

In Table 6, we list all subgraphs at moderate size (no more than 6 vertices or 8 edges) that can/cannot be counted by each GNN model. Here, we only list the graph-level expressivity as the node/edge-level expressivity involves too many non-isomorphic rooted graphs and cannot be fully presented (see Tables 4 and 5). We believe these results is comprehensive enough to cover most substructures of interest in the GNN community.

Table 4: The number of (connected) graphs (or rooted graphs) of n vertices or m edges that can be counted under homomorphism by different models. These statistics can be viewed as a quantitative expressivity comparison between models. 

<table><tr><td rowspan="2" colspan="2"></td><td colspan="5">Number of vertices n</td><td colspan="8">Number of edges m</td></tr><tr><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td rowspan="6">Graph level</td><td>MPNN</td><td>1</td><td>1</td><td>2</td><td>3</td><td>6</td><td>1</td><td>1</td><td>2</td><td>3</td><td>6</td><td>11</td><td>23</td><td>47</td></tr><tr><td>Subgraph GNN</td><td>1</td><td>2</td><td>5</td><td>15</td><td>51</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>29</td><td>76</td><td>210</td></tr><tr><td>Local 2-GNN</td><td>1</td><td>2</td><td>5</td><td>15</td><td>55</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>29</td><td>77</td><td>216</td></tr><tr><td>Local 2-FGNN</td><td>1</td><td>2</td><td>5</td><td>15</td><td>56</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>29</td><td>77</td><td>216</td></tr><tr><td>2-FGNN</td><td>1</td><td>2</td><td>5</td><td>15</td><td>56</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>29</td><td>77</td><td>216</td></tr><tr><td>All</td><td>1</td><td>2</td><td>6</td><td>21</td><td>112</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>30</td><td>79</td><td>227</td></tr><tr><td rowspan="6">Node level</td><td>MPNN</td><td>1</td><td>2</td><td>4</td><td>9</td><td>20</td><td>1</td><td>2</td><td>4</td><td>9</td><td>20</td><td>48</td><td>115</td><td>286</td></tr><tr><td>Subgraph GNN</td><td>1</td><td>3</td><td>8</td><td>27</td><td>88</td><td>1</td><td>2</td><td>5</td><td>12</td><td>31</td><td>83</td><td>228</td><td>640</td></tr><tr><td>Local 2-GNN</td><td>1</td><td>3</td><td>10</td><td>44</td><td>215</td><td>1</td><td>2</td><td>5</td><td>13</td><td>37</td><td>113</td><td>361</td><td>1210</td></tr><tr><td>Local 2-FGNN</td><td>1</td><td>3</td><td>10</td><td>44</td><td>217</td><td>1</td><td>2</td><td>5</td><td>13</td><td>37</td><td>113</td><td>361</td><td>1210</td></tr><tr><td>2-FGNN</td><td>1</td><td>3</td><td>10</td><td>44</td><td>217</td><td>1</td><td>2</td><td>5</td><td>13</td><td>37</td><td>113</td><td>361</td><td>1210</td></tr><tr><td>All</td><td>1</td><td>3</td><td>11</td><td>58</td><td>407</td><td>1</td><td>2</td><td>5</td><td>13</td><td>37</td><td>114</td><td>367</td><td>1248</td></tr><tr><td rowspan="5">Edge level</td><td>Subgraph GNN</td><td>1</td><td>4</td><td>18</td><td>77</td><td>340</td><td>1</td><td>3</td><td>10</td><td>33</td><td>107</td><td>347</td><td>1126</td><td>3664</td></tr><tr><td>Local 2-GNN</td><td>1</td><td>4</td><td>21</td><td>116</td><td>693</td><td>1</td><td>3</td><td>10</td><td>35</td><td>124</td><td>450</td><td>1665</td><td>6267</td></tr><tr><td>Local 2-FGNN</td><td>1</td><td>4</td><td>21</td><td>118</td><td>735</td><td>1</td><td>3</td><td>10</td><td>35</td><td>124</td><td>451</td><td>1678</td><td>6373</td></tr><tr><td>2-FGNN</td><td>1</td><td>4</td><td>21</td><td>118</td><td>735</td><td>1</td><td>3</td><td>10</td><td>35</td><td>124</td><td>451</td><td>1678</td><td>6374</td></tr><tr><td>All</td><td>1</td><td>4</td><td>23</td><td>162</td><td>1549</td><td>1</td><td>3</td><td>10</td><td>35</td><td>125</td><td>460</td><td>1747</td><td>6830</td></tr></table>

Table 5: The number of (connected) graphs (or rooted graphs) of $n$ vertices or $m$ edges that can be subgraph-counted by different models. These statistics can be viewed as a quantitative expressivity comparison between models. 

<table><tr><td rowspan="2" colspan="2"></td><td colspan="5">Number of vertices n</td><td colspan="8">Number of edges m</td></tr><tr><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td rowspan="6">Graph level</td><td>MPNN</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Subgraph GNN</td><td>1</td><td>2</td><td>5</td><td>13</td><td>24</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>20</td><td>21</td><td>22</td></tr><tr><td>Local 2-GNN</td><td>1</td><td>2</td><td>5</td><td>13</td><td>24</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>20</td><td>21</td><td>22</td></tr><tr><td>Local 2-FGNN</td><td>1</td><td>2</td><td>5</td><td>13</td><td>24</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>20</td><td>21</td><td>22</td></tr><tr><td>2-FGNN</td><td>1</td><td>2</td><td>5</td><td>13</td><td>24</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>20</td><td>21</td><td>22</td></tr><tr><td>All</td><td>1</td><td>2</td><td>6</td><td>21</td><td>112</td><td>1</td><td>1</td><td>3</td><td>5</td><td>12</td><td>30</td><td>79</td><td>227</td></tr><tr><td rowspan="6">Node level</td><td>MPNN</td><td>1</td><td>2</td><td>2</td><td>2</td><td>2</td><td>1</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Subgraph GNN</td><td>1</td><td>3</td><td>8</td><td>18</td><td>27</td><td>1</td><td>2</td><td>5</td><td>10</td><td>16</td><td>23</td><td>30</td><td>38</td></tr><tr><td>Local 2-GNN</td><td>1</td><td>3</td><td>10</td><td>37</td><td>84</td><td>1</td><td>2</td><td>5</td><td>13</td><td>37</td><td>72</td><td>75</td><td>86</td></tr><tr><td>Local 2-FGNN</td><td>1</td><td>3</td><td>10</td><td>37</td><td>84</td><td>1</td><td>2</td><td>5</td><td>13</td><td>37</td><td>72</td><td>75</td><td>86</td></tr><tr><td>2-FGNN</td><td>1</td><td>3</td><td>10</td><td>37</td><td>84</td><td>1</td><td>2</td><td>5</td><td>13</td><td>37</td><td>72</td><td>75</td><td>86</td></tr><tr><td>All</td><td>1</td><td>3</td><td>11</td><td>58</td><td>407</td><td>1</td><td>2</td><td>5</td><td>13</td><td>37</td><td>114</td><td>367</td><td>1248</td></tr><tr><td rowspan="5">Edge level</td><td>Subgraph GNN</td><td>1</td><td>4</td><td>18</td><td>47</td><td>81</td><td>1</td><td>3</td><td>10</td><td>25</td><td>46</td><td>69</td><td>95</td><td>124</td></tr><tr><td>Local 2-GNN</td><td>1</td><td>4</td><td>21</td><td>92</td><td>208</td><td>1</td><td>3</td><td>10</td><td>35</td><td>105</td><td>171</td><td>179</td><td>216</td></tr><tr><td>Local 2-FGNN</td><td>1</td><td>4</td><td>21</td><td>92</td><td>208</td><td>1</td><td>3</td><td>10</td><td>35</td><td>105</td><td>171</td><td>179</td><td>216</td></tr><tr><td>2-FGNN</td><td>1</td><td>4</td><td>21</td><td>92</td><td>208</td><td>1</td><td>3</td><td>10</td><td>35</td><td>105</td><td>171</td><td>179</td><td>216</td></tr><tr><td>All</td><td>1</td><td>4</td><td>23</td><td>162</td><td>1549</td><td>1</td><td>3</td><td>10</td><td>35</td><td>125</td><td>460</td><td>1747</td><td>6830</td></tr></table>

Table 6: The ability of GNNs to homomorphism-count and subgraph-count different graphs $F$ within a bounded size of $n \leq 6$ vertices or $m \leq 8$ edges. When one or more GNNs fail to subgraph-count $F$ , this table also gives a homomorphic image $\tilde{F} \in \operatorname{Spasm}(F)$ that can be used to construct counterexample graphs (see Section 4.2). 

<table><tr><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td></td></tr><tr><td>2</td><td>1</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td><td>5</td><td>7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>3</td><td>2</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td><td>5</td><td>8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>3</td><td>3</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>4</td><td>3</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>4</td><td>3</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td><td>5</td><td>10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>4</td><td>4</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>5</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>4</td><td>4</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>5</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>4</td><td>5</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>5</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>4</td><td>6</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td></td><td>6</td><td>5</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>4</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>5</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>4</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>5</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>4</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>5</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>5</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>5</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>5</td><td></td><td>×</td><td>√</td><td>✓</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>5</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td>6</td><td>6</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>5</td><td>6</td><td></td><td>×</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

(see the next page)

<table><tr><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td></td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>√</td><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 7</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 8</td><td></td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 8</td><td></td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 8</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr></table>

(see the next page)

<table><tr><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="2">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td></tr><tr><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 9</td><td></td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>×</td><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 12</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 12</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 9</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 12</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 12</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 12</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 13</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 13</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 14</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>6 15</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>X</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 10</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>6 11</td><td></td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>7 6</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td></tr></table>

(see the next page)

<table><tr><td>n m</td><td>F</td><td>HomomorphismMP Sub L LF F</td><td>SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td>n m</td><td>F</td><td>HomomorphismMP Sub L LF F</td><td>SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 7</td><td></td><td>x √ √ √ √</td><td>x √ √ √ √</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x √ √ √ √</td><td>◇</td><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x x √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ ∛ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>X X X X X</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x √ √ √ √</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x √ √ √ √</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>T 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td></td><td></td><td></td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>T 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◆</td></tr><tr><td>7 7</td><td></td><td>x √ √ √ √</td><td>x √ √ √ √</td><td>◇</td><td>7 8</td><td></td><td>x √ √ √ √</td><td>x x x x x</td><td>◇</td></tr></table>

(see the next page)

<table><tr><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td></td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>×</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td></td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7 8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>6</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>×</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>X</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x</td><td>x √</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>7</td><td>8</td><td></td><td>x √</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>8</td><td>7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>8</td><td>7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x x</td><td>x</td><td>x</td><td>8</td><td>7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>7</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td></td><td>x</td><td>x x x x</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

(see the next page)

<table><tr><td>n m</td><td>F</td><td colspan="4">Homomorphism MP Sub L LF F</td><td colspan="4">Subgraph MP Sub L LF F</td><td> $\tilde{F}$ </td><td>n m</td><td>F</td><td colspan="4">Homomorphism MP Sub L LF F</td><td colspan="4">Subgraph MP Sub L LF F</td><td> $\tilde{F}$ </td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>-</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>X</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>八 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>八 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>8 7</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>8 8</td><td></td><td>✗</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr></table>

(see the next page)

<table><tr><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td>n m</td><td>F</td><td colspan="3">HomomorphismMP Sub L LF F</td><td colspan="3">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td></td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>X</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>×</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>9</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>y</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>✓</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>X</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>8</td><td></td><td></td><td>X</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>X</td><td>8</td><td></td><td></td><td>X</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>8</td><td>8</td><td></td><td></td><td>8</td><td>8</td><td></td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr></table>

(see the next page)

<table><tr><td>n m</td><td>F</td><td colspan="4">HomomorphismMP Sub L LF F</td><td colspan="4">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td>n m</td><td>F</td><td colspan="4">HomomorphismMP Sub L LF F</td><td colspan="4">SubgraphMP Sub L LF F</td><td> $\tilde{F}$ </td><td></td></tr><tr><td>8</td><td>8</td><td></td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>8</td><td>8</td><td></td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>8</td><td>8</td><td></td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td></td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>√</td><td>√</td><td>√</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>√</td><td>√</td><td>√</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>✓</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr><tr><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>9</td><td>8</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td> $\tilde{\mathbf{X}}$ </td></tr></table>

# H POLYNOMIAL EXPRESSIVITY

Puny et al. (2023) proposed the equivariant graph polynomials, which are polynomials P that take real squared matrices $X \in R^{n \times n}$ as input and outputs $P(X) \in \mathbb{R}^{n \times n}$ , such that P is equivariant under permutations, i.e., $P(\pi \cdot X) = \pi \cdot P(X)$ for all permutation $\pi \in S_{n}$ . The authors gave a concrete basis of equivariant polynomials, where each basis element $P_{F^{uv}}$ corresponds to a rooted multi-graph $F^{uv}$ marking two vertices u, v. Puny et al. (2023) showed that when X is restricted to be the adjacency matrix of an undirected simple graph G, each $F^{uv}$ will also reduce to an undirected simple graph, and $P_{F^{uv}}(X)$ precisely computes the (unlabeled) homomorphism count $\text{hom}(F^{uv}, G^{wx})$ for all $w, x \in V_{G}$ . Besides the original definition, Puny et al. (2023) also proposed the invariant graph polynomials and node-level equivariant graph polynomials, which are similarly related to the graph-level and node-level homomorphism count.

As a direct consequence, if a GNN model M cannot count graph $F^{uv}/F^{u}/F$ under homomorphism, it then cannot compute the equivariant/invariant graph polynomial $P_{F^{uv}}/P_{F^{u}}/P_{F}$ . Based on these connections, our results can be directly used to provide insights into which equivariant graph polynomials cannot be computed by model M. This recovers several results in their paper and answers an open problem shown below.

Corollary H.1. MPNN and PPGN++ are bounded by the Prototypical node-based model and Prototypical edge-based model (defined in Puny et al. (2023)) for computing node-level and edge-level equivariant graph polynomials, respectively.

Proof. Without loss of generality, we assume that the corresponding graphs of all equivariant polynomials are connected. According to Puny et al. (2023, Proposition H.2), the Prototypical node-based model can compute all $P_{F^{u}}$ where $F^{u}$ is a (rooted) tree and cannot compute other graph polynomials. If MPNN is not bounded by the Prototypical node-based model, then it can compute some $P_{\tilde{F}^{u}}$ where $\tilde{F}$ is not a tree. However, this is impossible since MPNN can only count forests under homomorphism according to Theorem 3.7. Note that the MPNN defined in their paper is equivalent to our definition when only considering connected graphs (the extra global aggregation $11^{\top}X$ in their definition (9) does not improve the homomorphism expressivity).

We next turn to PPGN++, and the proof is similar (but more involved). We first show that the Prototypical edge-based model can compute any $P_{F^{uv}}$ satisfying that the treewidth of the graph $\tilde{F} := (V_{F}, E_{F} \cup \{\{u, v\}\}, \ell_{F})$ is no more than 2. If $\operatorname{tw}(\tilde{F}) \leq 2$ , it is a partial 2-tree. Thus, there is an ordering $w_{1}, \cdots, w_{n}$ of the vertex set $V_{F}$ such that when deleting each vertex $w_{i}$ and all incident edges in turn, we only ever delete vertices of degree at most 2. Now we claim that we can always order the two vertices u, v at the end, i.e., $w_{n-1} = u$ and $w_{n} = v$ . Otherwise, there is a subset $U \subset V_{F}$ such that all vertices in the induced subgraph $\tilde{F}[U]$ are of degree at least 3 expect u, v. It follows that the $\operatorname{tw}(\tilde{F}[U]) \geq \operatorname{tw}(H) \geq 3$ where graph H is the graph obtained from $\tilde{F}[U]$ by contracting u (or v) if $\deg_{\tilde{F}[U]}(u) \leq 2$ (or $\deg_{\tilde{F}[U]}(v) \leq 2$ ). This yields a contradiction and verifies the claim that we can always set $w_{n-1} = u$ and $w_{n} = v$ .

Now, following the proof in Puny et al. (2023, Proposition H.3), the Prototypical edge-based model can contract $\tilde{F}^{uv}$ to a graph with only two vertices u, v and thus can compute the edge-level polynomial $P_{\tilde{F}^{uv}}$ . If PPGN++ is not bounded by the Prototypical edge-based model, then it can compute some $P_{\tilde{F}^{uv}}$ where $\operatorname{tw}(\tilde{F}) \geq 3$ . Therefore, it can count the graph $\tilde{F}$ under homomorphism at graph-level (since it can already count $\tilde{F}$ at edge-level). This implies that PPGN++ is strictly more expressive than 2-FGNN (2-FWL) because we have proved that all graphs in $F^{F}$ have a treewidth no more than 2. This yields a contradiction since PPGN++ is still bounded by 2-FWL in distinguishing non-isomorphic graphs.

We also provide insights into the following results in their paper:

Corollary H.2. The Prototypical node-based model is not 3-node-polynomial-exact. The Prototypical edge-based model is not 6-node-polynomial-exact and not 5-edge-polynomial-exact.

This is simply because the triangle is not a tree and the 4-clique does not have a NED (or equivalently, the treewidth of a 4-clique is 3). It is also clear why the degree of the edge-based polynomial is 6, which is one less than that of the node-based polynomial using the concept of NED.

# I EXPERIMENTAL DETAILS

In this section, we provide all the experimental details in Section 5.

# I.1 DATASETS

We conduct experiments on tive tasks: (i) graph homomorphism counting, (ii) subgraph counting, (iii) ZINC-subset (Dwivedi et al., 2020), (iii) ZINC-full (Dwivedi et al., 2020), and (iv) Alchemy (Chen et al., 2019a).

Homomorphism/Subgraph Counting. For both homomorphism and subgraph counting tasks, we use the standard synthetic graph dataset constructed in Zhao et al. (2022a) (which has been used in a number of papers, see e.g., Frasca et al., 2022; Huang et al., 2023; Zhang et al., 2023a). For homomorphism counting, we count the number of graph/node/edge-level homomorphisms for each pattern in Table 1 and normalize the value by the mean and variance across all graphs in the dataset. The evaluation metric of graph-level expressivity is chosen as the Mean Absolute Error (MAE). For node/edge-level expressivity, the error on each graph is defined to be the sum of absolute error over all vertices/edges. We then report the MAE across all graphs in the dataset. This ensures that graph/node/edge-level errors are roughly at the same scale (since $\mathrm{hom}(F,G) = \sum_{w\in V_G}\mathrm{hom}(F^u,G^w) = \sum_{w,x\in V_G}\mathrm{hom}(F^{uv},G^{wx})$ for all $u,v\in V_F$ ). For subgraph counting, the data processing and evaluation metric is similar to homomorphism counting, but there is a slight difference in the node/edge-level setting: there are no marked vertices in the pattern graph $F$ (see Table 3). Instead, given a graph $G$ in the dataset and a vertex $w\in V_G$ , we count the number of subgraphs containing $w$ that are isomorphic to $F$ and $w$ can be mapped to any vertex in $F$ . Due to this difference, in the node/edge-level setting, the error on each graph is defined to be the average of absolute error over all vertices/edges.

ZINC. ZINC (Dwivedi et al., 2020) is a standard real-world dataset for benchmarking molecular property prediction. The dataset consists of 250K molecular graphs, and the task is to predict the constrained solubility of the given molecule. In addition to the full dataset (denoted as ZINC-full), ZINC-subset is a sampled dataset with 12k molecules from the ZINC-full dataset. We train and test our models on both datasets following the standard protocol from Dwivedi et al. (2020).

Alchemy. Alchemy (Chen et al., 2019a) is another real-world dataset with 12 graph-level quantum mechanical properties. We follow the sampling and training protocol from Lim et al. (2023); Puny et al. (2023), using 100K samples for training, 10K samples for testing, and 10K samples for validation.

# I.2 MODEL DETAILS

All models are implemented using the PyTorch (Paszke et al., 2019) framework and the PyTorch Geometric library (Fey & Lenssen, 2019). We consider four types of GNNs defined in Section 2: MPNN, Subgraph GNN, Local 2-GNN, and Local 2-FGNN. For each GNN model, the feature initialization, message-passing layers, and final pooling operation are separately defined below.

Initialization. On both ZINC and Alchemy datasets, each graph node is an atom. We maintain a learnable atom embedding for each type of atom and use it to initialize features in GNN models. For MPNN, the initial feature $h^{(0)}(u)$ of node $u$ is simply the atom embedding, denoted as $h^{(0)}(u) = e_{\mathrm{atom}(u)}^{\mathrm{N}}$ . For other models, the initial feature $h^{(0)}(u,v)$ of node pair $(u,v)$ consists of two parts: the first part is the node embedding of $v$ , and the second part is a distance encoding that embeds the shortest path distance between $u$ and $v$ , as adopted in Zhang et al. (2023a). We note that while incorporating distance encoding does not increase the models' theoretical expressive power (see Zhang et al. (2023a)), it may add an inductive bias that can be helpful in real-world tasks. Formally, the initial feature can be written as $h^{(0)}(u,v) = [e_{\mathrm{atom}(v)}^{\mathrm{N}}, e_{\mathrm{clip}(\mathrm{dis}(u,v))}^{\mathrm{D}}]$ . Here, we clip the distance to a predefined value max\_dis so that there are a finite number of distance embeddings, and distances greater than the hyper-parameter max\_dis (including the disconnected case) share the embedding.

Propagation. On both ZINC and Alchemy datasets, each edge in a graph corresponds to a chemical bond and has a bond type. We maintain a learnable edge embedding for each type of edges in each layer and denote the embedding of edge $\{u,v\}$ in layer l as $g^{(l)}(u,v)$ . For MPNN, we use the

standard GIN architecture proposed in Xu et al. (2019), which has the following form:

$$
h ^ {(l + 1)} (u) = \operatorname{ReLU} (\mathsf {B N} ^ {(l)} (f ^ {(l)} (u))), \tag {32}
$$

$$
f ^ {(l)} (u) = \mathsf {G I N} ^ {(l)} \left(h ^ {(l)} (u), \sum_ {v \in N _ {G} (u)} \operatorname{ReLU} \left(\mathsf {F C} ^ {(l)} (h ^ {(l)} (v)) + g ^ {(l)} (u, v)\right)\right), \tag {33}
$$

where

$$
\operatorname{GIN} ^ {(l)} (\boldsymbol {x}, \boldsymbol {y}) = \operatorname{MLP} ^ {(l)} \left((1 + \epsilon^ {(l)}) \boldsymbol {x} + \boldsymbol {y}\right). \tag {34}
$$

Here, $FC^{(l)}$ is a parameterized linear transformation, $\epsilon^{(l)}$ is a learnable parameter, $BN^{(l)}$ is the batch normalization (Ioffe & Szegedy, 2015), and $MLP^{(l)}$ is a two-layer feed-forward network with another batch normalization in the hidden layer.

For other architectures, the $l$ -th GNN layer analogously has the following form:

$$
h ^ {(l + 1)} (u, v) = \operatorname{ReLU} (\mathsf {B N} ^ {(l)} (f ^ {(l)} (u, v))), \tag {35}
$$

where the term $f^{(l)}(u,v)$ is defined separately for each model:

\- Subgraph GNN:

$$
f ^ {(l)} (u, v) = \mathsf {G I N} ^ {(l)} \left(h ^ {(l)} (u, v), \sum_ {w \in N _ {G} (v)} \mathsf {R e L U} \left(\mathsf {F C} ^ {(l)} (h ^ {(l)} (u, w)) + g ^ {(l)} (w, v)\right)\right). \tag {36}
$$

\- Local 2-GNN:

$$
\begin{array}{l} f ^ {(l)} (u, v) = \operatorname{GIN} ^ {(l, 1)} \left(h ^ {(l)} (u, v), \sum_ {w \in N _ {G} (u)} \operatorname{ReLU} \left(\operatorname{FC} ^ {(l, 1)} \left(h ^ {(l)} (w, v)\right) + g ^ {(l)} (u, w)\right)\right) \\ + \operatorname{GIN} ^ {(l, 2)} \left(h ^ {(l)} (u, v), \sum_ {w \in N _ {G} (v)} \operatorname{ReLU} \left(\operatorname{FC} ^ {(l, 2)} \left(h ^ {(l)} (u, w)\right) + g ^ {(l)} (w, v)\right)\right). \tag {37} \\ \end{array}
$$

\- Local 2-FGNN:

$$
\begin{array}{l} f ^ {(l)} (u, v) \\ = \operatorname{GIN} ^ {(l, 1)} \left(h ^ {(l)} (u, v), \sum_ {w \in N _ {G} (u)} \operatorname{ReLU} \left(\operatorname{FC} ^ {(l, 1)} \left(h ^ {(l)} (u, w)\right) + \operatorname{FC} ^ {(l, 2)} \left(h ^ {(l)} (w, v)\right) + g ^ {(l)} (u, w)\right)\right) \\ + \operatorname{GIN} ^ {(l, 2)} \left(h ^ {(l)} (u, v), \sum_ {w \in N _ {G} (v)} \operatorname{ReLU} \left(\operatorname{FC} ^ {(l, 1)} \left(h ^ {(l)} (w, v)\right) + \operatorname{FC} ^ {(l, 2)} \left(h ^ {(l)} (u, w)\right) + g ^ {(l)} (w, v)\right)\right). \tag {38} \\ \end{array}
$$

For all the above GNN architectures, it can be seen that each layer only aggregates the local neighborhood of vertices or vertex pairs. This design will have shortcomings for disconnected graphs since a vertex cannot aggregate information from other connected components no matter how deep the model is (Barceló et al., 2020). Note that there do exist disconnected graphs in real-world datasets like ZINC. Therefore, on real-world datasets like ZINC and Alchemy, we also incorporate a global aggregation with the following form for each layer (similar to the global aggregation in Frasca et al. (2022); Zhang et al. (2023a)):

$$
\operatorname{GIN} ^ {(l, \mathrm{G})} \left(h ^ {(l)} (u, v), \sum_ {w \in V _ {G}} h ^ {(l)} (u, w)\right) \tag {39}
$$

Note that the global aggregation does not increase model's theoretical expressive power according to Zhang et al. (2023a).

Pooling. Except for edge-level tasks, a final pooling layer is used to produce node-level features $h(u)$ for all nodes u. It is implemented as follows:

$$
h (u) = \mathsf {M L P} \left(\sum_ {v \in \mathcal {V}} h ^ {(L)} (u, v)\right), \tag {40}
$$

where MLP is a 2-layer perceptron. For graph-level tasks, we further use a mean pooling layer to aggregate all $h(u)$ and obtain the graph representation.

# I.3 TRAINING DETAILS

All experiments are run on a single NVIDIA Tesla V100 GPU. For all tasks, we use the distance encoding hyper-parameter max\_dis = 5. To enable a fair comparison between models, for each task we keep the same depth for different models while varying the hidden dimension so that the number of model parameters is roughly the same. Note that the dimensions are chosen such that all models roughly obey the 500K parameter budget in ZINC and Alchemy. The hidden dimension size and the number of model parameters are listed as follows. All models are trained using the Adam optimizer.

Table 7: Model size in different tasks. 

<table><tr><td rowspan="2">Task</td><td colspan="3">Hidden dimension</td><td colspan="3"># Parameters</td></tr><tr><td>Counting</td><td>ZINC</td><td>Alchemy</td><td>Counting</td><td>ZINC</td><td>Alchemy</td></tr><tr><td>MPNN</td><td>128</td><td>150</td><td>150</td><td>314,119</td><td>510,158</td><td>509,719</td></tr><tr><td>Subgraph GNN</td><td>128</td><td>120</td><td>120</td><td>314,759</td><td>503,774</td><td>503,425</td></tr><tr><td>Local 2-GNN</td><td>96</td><td>96</td><td>96</td><td>317,388</td><td>495,188</td><td>494,911</td></tr><tr><td>Local 2-FGNN</td><td>96</td><td>96</td><td>96</td><td>317,388</td><td>495,188</td><td>494,911</td></tr></table>

Homomorphism/Subgraph Counting. We use a model depth of L = 5 in all experiments. Following prior work (Huang et al., 2023), we remove all BN layers in all models. The initial learning rate is chosen as 0.001 and is decayed by a factor of 0.9 once the MAE on the validation set plateaus for 10 epochs. Each model is trained for 1200 epochs with a batch size of 512. We ran each experiment 4 times independently with different seeds and reported the average performance at the last epoch. We found that the standard deviation among different seeds is negligible.

ZINC. We use a model depth of L = 6 in all experiments for both ZINC-subset and ZINC-full. Following prior work (Zhang et al., 2023a; Frasca et al., 2022), The initial learning rate is chosen as 0.001 and is decayed by a factor of 0.5 once the MAE on the validation set plateaus for 20 epochs. Each model is trained for 400 epochs on ZINC-subset and 500 epochs on ZINC-full, both with a batch size of 128. We report the MAE for the model checkpoint with the best validation performance. We ran each experiment 10 times independently with different seeds and reported the average performance as well as the standard deviation.

Alchemy. We use a model depth of L = 6 in all experiments. Following prior work (Lim et al., 2023; Puny et al., 2023), The initial learning rate is chosen as 0.002 and is decayed by a factor of 0.5 once the MAE on the validation set plateaus for 20 epochs. Each model is trained for 500 epochs with a batch size of 128. We report the MAE for the model checkpoint with the best validation performance. We ran each experiment 10 times independently with different seeds and reported the average performance as well as the standard deviation.

# I.4 PERFORMANCE OF BASELINE MODELS IN LITERATURE

For completeness, in this subsection we give a comprehensive list of the performance of GNN models in the literature on ZINC and Alchemy datasets. The numbers in each table below are directly taken from the original papers.

Table 8: Performance of different GNN models on ZINC dataset reported in the literature. 

<table><tr><td>Method</td><td>Model</td><td>Reference</td><td>Test MAE</td></tr><tr><td rowspan="8">MPNN</td><td>GIN</td><td>(Xu et al., 2019)</td><td> $0.526 \pm 0.051$ </td></tr><tr><td>GraphSAGE</td><td>(Hamilton et al., 2017)</td><td> $0.398 \pm 0.002$ </td></tr><tr><td>GAT</td><td>(Veličković et al., 2018)</td><td> $0.384 \pm 0.007$ </td></tr><tr><td>GCN</td><td>(Kipf &amp; Welling, 2017)</td><td> $0.367 \pm 0.011$ </td></tr><tr><td>MoNet</td><td>(Monti et al., 2017)</td><td> $0.292 \pm 0.006$ </td></tr><tr><td>GatedGCN-PE</td><td>(Bresson &amp; Laurent, 2017)</td><td> $0.214 \pm 0.006$ </td></tr><tr><td>MPNN(sum)</td><td>(Gilmer et al., 2017)</td><td> $0.145 \pm 0.007$ </td></tr><tr><td>PNA</td><td>(Corso et al., 2020)</td><td> $0.142 \pm 0.010$ </td></tr><tr><td rowspan="4">Higher-order GNN</td><td>RingGNN</td><td>(Chen et al., 2019b)</td><td> $0.353 \pm 0.019$ </td></tr><tr><td>PPGN</td><td>(Maron et al., 2019a)</td><td> $0.303 \pm 0.068$ </td></tr><tr><td>PPGN</td><td>(Puny et al., 2023)</td><td> $0.079 \pm 0.005$ </td></tr><tr><td>PPGN++</td><td>(Puny et al., 2023)</td><td> $0.076 \pm 0.003$ </td></tr><tr><td rowspan="7">Subgraph GNN</td><td>NGNN</td><td>(Zhang &amp; Li, 2021)</td><td> $0.111 \pm 0.003$ </td></tr><tr><td>GNN-AK</td><td>(Zhao et al., 2022a)</td><td> $0.105 \pm 0.010$ </td></tr><tr><td>GNN-AK+</td><td>(Zhao et al., 2022a)</td><td> $0.091 \pm 0.002$ </td></tr><tr><td>ESAN</td><td>(Bevilacqua et al., 2022)</td><td> $0.102 \pm 0.003$ </td></tr><tr><td>SUN</td><td>(Frasca et al., 2022)</td><td> $0.083 \pm 0.003$ </td></tr><tr><td>I2-GNN</td><td>(Huang et al., 2023)</td><td> $0.083 \pm 0.001$ </td></tr><tr><td>ID-MPNN</td><td>(Zhou et al., 2023a)</td><td> $0.083 \pm 0.003$ </td></tr><tr><td rowspan="4">Local (F)GNN</td><td>SetGNN</td><td>(Zhao et al., 2022b)</td><td> $0.075 \pm 0.003$ </td></tr><tr><td>GNN-SSWL</td><td>(Zhang et al., 2023a)</td><td> $0.082 \pm 0.010$ </td></tr><tr><td>GNN-SSWL+</td><td>(Zhang et al., 2023a)</td><td> $0.070 \pm 0.005$ </td></tr><tr><td>N2-GNN</td><td>(Feng et al., 2023)</td><td> $0.059 \pm 0.002$ </td></tr><tr><td rowspan="4">Substructure-based GNN</td><td>GSN</td><td>(Bouritsas et al., 2022)</td><td> $0.101 \pm 0.010$ </td></tr><tr><td>CIN (Small)</td><td>(Bodnar et al., 2021a)</td><td> $0.094 \pm 0.004$ </td></tr><tr><td>CIN</td><td>(Bodnar et al., 2021a)</td><td> $0.079 \pm 0.006$ </td></tr><tr><td>CIN++</td><td>(Giusti et al., 2023)</td><td> $0.077 \pm 0.004$ </td></tr><tr><td rowspan="6">Graph Transformer</td><td>SAN</td><td>(Kreuzer et al., 2021)</td><td> $0.139 \pm 0.006$ </td></tr><tr><td>K-Subgraph SAT</td><td>(Chen et al., 2022)</td><td> $0.094 \pm 0.008$ </td></tr><tr><td>Graphormer</td><td>(Ying et al., 2021)</td><td> $0.122 \pm 0.006$ </td></tr><tr><td>URPE</td><td>(Luo et al., 2022)</td><td> $0.086 \pm 0.007$ </td></tr><tr><td>Graphormer-GD</td><td>(Zhang et al., 2023b)</td><td> $0.081 \pm 0.009$ </td></tr><tr><td>GPS</td><td>(Rampasek et al., 2022)</td><td> $0.070 \pm 0.004$ </td></tr><tr><td rowspan="6">Other</td><td>PF-GNN</td><td>(Dupty et al., 2021)</td><td> $0.122 \pm 0.010$ </td></tr><tr><td>KP-GIN</td><td>(Feng et al., 2022)</td><td> $0.093 \pm 0.007$ </td></tr><tr><td>SignNet</td><td>(Lim et al., 2023)</td><td> $0.084 \pm 0.006$ </td></tr><tr><td>PathNN</td><td>(Michel et al., 2023)</td><td> $0.090 \pm 0.004$ </td></tr><tr><td>PPGN++(6)</td><td>(Puny et al., 2023)</td><td> $0.071 \pm 0.001$ </td></tr><tr><td>PlanE</td><td>(Dimitrov et al., 2023)</td><td> $0.076 \pm 0.003$ </td></tr></table>

Table 9: Performance of different GNN models on Alchemy dataset reported in the literature. 

<table><tr><td>Model</td><td>Reference</td><td>Test MAE</td></tr><tr><td>GIN</td><td>(Xu et al., 2019)</td><td> $0.180 \pm 0.006$ </td></tr><tr><td>PF-GNN</td><td>(Dupty et al., 2021)</td><td> $0.111 \pm 0.010$ </td></tr><tr><td>δ-2-GNN</td><td>(Morris et al., 2020)</td><td> $0.118 \pm 0.001$ </td></tr><tr><td>Recon-GNN</td><td>(Cotta et al., 2021)</td><td> $0.125 \pm 0.001$ </td></tr><tr><td>SpeqNet</td><td>(Morris et al., 2022)</td><td> $0.115 \pm 0.001$ </td></tr><tr><td>SignNet</td><td>(Lim et al., 2023)</td><td> $0.113 \pm 0.002$ </td></tr><tr><td>PPGN</td><td>(Puny et al., 2023)</td><td> $0.113 \pm 0.001$ </td></tr><tr><td>PPGN++</td><td>(Puny et al., 2023)</td><td> $0.111 \pm 0.002$ </td></tr><tr><td>PPGN++(6)</td><td>(Puny et al., 2023)</td><td> $0.109 \pm 0.001$ </td></tr></table>