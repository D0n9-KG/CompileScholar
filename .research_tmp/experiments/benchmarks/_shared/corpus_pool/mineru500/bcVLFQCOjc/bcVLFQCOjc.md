# DETKZIFY: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ

Jonas Belouadi\*

Simone Paolo Ponzetto $^{†}$

Steffen Eger $^{\ddagger}$

Natural Language Learning Group, $^{*,\ddagger}$ Data and Web Science Group $^{\dagger}$ University of Mannheim, $^{*,\dagger}$ University of Technology Nuremberg $^{\ddagger}$ {jonas.belouadi,ponzetto}@uni-mannheim.de, steffen.eger@utn.de

# Abstract

Creating high-quality scientific figures can be time-consuming and challenging, even though sketching ideas on paper is relatively easy. Furthermore, recreating existing figures that are not stored in formats preserving semantic information is equally complex. To tackle this problem, we introduce $DETIKZIFY$ , a novel multimodal language model that automatically synthesizes scientific figures as semantics-preserving TikZ graphics programs based on sketches and existing figures. To achieve this, we create three new datasets: $DATIKZ_{v2}$ , the largest TikZ dataset to date, containing over 360k human-created TikZ graphics; SKETCHFIG, a dataset that pairs hand-drawn sketches with their corresponding scientific figures; and METAFIG, a collection of diverse scientific figures and associated metadata. We train $DETIKZIFY$ on METAFIG and $DATIKZ_{v2}$ , along with synthetically generated sketches learned from SKETCHFIG. We also introduce an MCTS-based inference algorithm that enables $DETIKZIFY$ to iteratively refine its outputs without the need for additional training. Through both automatic and human evaluation, we demonstrate that $DETIKZIFY$ outperforms commercial CLAUDE 3 and GPT-4V in synthesizing TikZ programs, with the MCTS algorithm effectively boosting its performance. We make our code, models, and datasets publicly available. $^{1}$

# 1 Introduction

Creating high-quality scientific figures is similar to typesetting scientific documents in many ways. When it comes to typesetting, markup languages like $\mathrm{LATEX}$ enjoy widespread popularity, as exemplified by major machine learning conferences that either mandate or strongly encourage $\mathrm{LATEX}$ -formatted submissions. $^2$ The advantages of using such languages go beyond producing high-quality outputs; documents expressed as high-level, semantics-preserving programs enhance accessibility, serve archival purposes, and remain easily editable and human-readable (facilitating language modeling applications; Moosavi et al., 2021; Lu et al., 2023). Consequently, efforts have been made to recover this type of information from outputs stored in lower-level vector graphics formats like PDF or SVG, or raster graphics formats (Desai et al., 2021; Blecher et al., 2024). At the other end of the spectrum, the versatility of $\mathrm{LATEX}$ comes with a steep learning curve, and typesetting can often be challenging for end users. In response, researchers have been working on assisting authors with certain aspects of the problem, such as typesetting math based on hand-drawn sketches (Kirsch, 2010; Wu et al., 2020).

Just like documents, scientific figures can also be created using markup languages. A popular example is the TikZ graphics language (Tantau, 2023), which can be integrated into $\mathrm{LATEX}$ documents, providing comparable benefits and encountering similar challenges. However, unlike $\mathrm{LATEX}$ , the prospects of TikZ in research contexts remain largely unexplored. Although the promise of simplifying editing and

![](images/e1a5faba32aae02a42b88d4cc98310acc0a15cd706ee8f93367738efd2c5b410.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["Input Image"]
    B --> C["Input Image"]
    C --> D["DetTikZIFY"]
    D --> E["Generate TikZ"]
    E --> F["LLAMA"]
    E --> G["Linear"]
    E --> H["SigLIP"]
    F --> I["Generate TikZ"]
    G --> I
    H --> I
    I --> J["IATEX Engine"]
    J --> K["Output Data Flow"]
    K --> L["Input Image"]
    L --> M["Input Image"]
    M --> N["Output Data Flow"]
    N --> O["Generate TikZ"]
    O --> P["Generate TikZ"]
    P --> Q["Generate TikZ"]
    Q --> R["Generate TikZ"]
    R --> S["Generate TikZ"]
    S --> T["Generate TikZ"]
    T --> U["Generate TikZ"]
    U --> V["Generate TikZ"]
    V --> W["Generate TikZ"]
    W --> X["Generate TikZ"]
    X --> Y["Generate TikZ"]
    Y --> Z["Generate TikZ"]
    Z --> AA["Generate TikZ"]
    AA --> AB["Generate TikZ"]
    AB --> AC["Generate TikZ"]
    AC --> AD["Generate TikZ"]
    AD --> AE["Generate TikZ"]
    AE --> AF["Generate TikZ"]
    AF --> AG["Generate TikZ"]
    AG --> AH["Generate TikZ"]
    AH --> AI["Generate TikZ"]
    AI --> AJ["Generate TikZ"]
    AJ --> AK["Generate TikZ"]
    AK --> AL["Generate TikZ"]
    AL --> AM["Generate TikZ"]
    AM --> AN["Generate TikZ"]
    AN --> AO["Generate TikZ"]
    AO --> AP["Generate TikZ"]
    AP --> AQ["Generate TikZ"]
    AQ --> AR["Generate TikZ"]
    AR --> AS["Generate TikZ"]
    AS --> AT["Generate TikZ"]
    AT --> AU["Generate TikZ"]
    AU --> AV["Generate TikZ"]
    AV --> AW["Generate TikZ"]
    AW --> AX["Generate TikZ"]
    AX --> AY["Generate TikZ"]
```
</details>

Figure 1: Overview of the DETiKZIFY architecture: A multimodal language model converts sketches or figures into TikZ programs, which are compiled by a LATEX engine. This provides a reward signal to the model via MCTS, allowing it to iteratively refine the output until satisfactory results are achieved.

enabling applications in visual understanding (Masry et al., 2022; Huang et al., 2023) is evident, there are currently no viable solutions for recovering graphics programs from compiled figures. Moreover, there is a lack of tools that assist in creating graphics programs, e.g., based on hand-drawn sketches, despite the clear demand for such approaches on the T $_{EX}$ Stack Exchange (T $_{EX}$ .SE), $^{3}$ where nearly 10% of all questions revolve around TikZ, making it the most frequently discussed topic on the site. Addressing this gap could greatly improve the accessibility of existing figures and support researchers at all levels of programming proficiency when creating new ones, fostering diversity and inclusion. In response, we introduce D $_{ETIK}$ Z $_{IFY}$ , a multimodal language model that automatically synthesizes TikZ programs for scientific figures and sketches (cf. Figure 1). Our key contributions are as follows:

(i) As part of DETiKZIFY, we introduce (a) DATiKZv2, a large TikZ dataset with over 360k human-created TikZ graphics; (b) SKETCHFIG, a dataset of human-created sketches with paired scientific figures; and (c) METAFIG, a large meta-dataset of scientific figures and associated texts.   
(ii) We train $DETIKZIFY$ on METAFIG and $DATIKZ_{v2}$ , augmented with synthetic sketches that mimic SKETCHFIG. We demonstrate that $DETIKZIFY$ can effectively synthesize TikZ programs for both existing scientific figures and sketches, outperforming the commercial large language models (LLMs) GPT-4V and CLAUDE 3 (OpenAI, 2023b; Anthropic, 2024).   
(iii) We also present an inference algorithm based on Monte Carlo Tree Search (MCTS) that is tailored to graphics programs and allows $D_{ET1K}Z_{IFY}$ to iteratively refine its own outputs for a given computational budget, further improving performance without additional training.

# 2 Related Work

Image-to-ATEX Conversion A closely related task is the translation of mathematical illustrations into LATEX markup. In inspirational work, Kirsch (2010) tackle the recognition of single hand-drawn symbols to find corresponding LATEX commands. Subsequent works by Deng et al. (2017); Zhang et al. (2017, 2019); Wu et al. (2020); Wang and Liu (2021) expand on this concept to handle hand-drawn and scanned math formulas. Suzuki et al. (2003); Wang and Liu (2020); Blecher et al. (2024); Lv et al. (2023) further extend the scope by extracting LATEX formulas alongside text from entire documents.

Image Vectorization Similarly, converting (rasterized) figures into TikZ programs can be characterized as a form of image vectorization (Sun et al., 2007; Diebel, 2008; Ganin et al., 2018; Li et al., 2020; Ma et al., 2022; Zhu et al., 2024). Most existing methods vectorize images into low-level graphics primitives in the SVG format (Tian and Günther, 2024). Although this works well for specific domains like fonts, icons, and emoji (Lopes et al., 2019; Carlier et al., 2020; Reddy, 2021; Rodriguez et al., 2023b), it does not capture higher-level semantics and does not generalize well to our scientific context (cf. Appendix B). Closer to our work, Ellis et al. (2018) generate vector representations as graphics programs based on a limited subset of $\mathrm{IAT_{EX}}$ commands. Their approach even handles

sketches, but their experiments are restricted to a synthetic dataset with only basic shapes of limited complexity. Belouadi et al. (2024) also generate TikZ programs, but their primary emphasis is on conditioning the generation on textual descriptions, with images serving only as a secondary input.

Code Generation As TikZ is implemented in the Turing-complete T $_{EX}$ macro system (Erdweg and Ostermann, 2011), our work is also closely tied to code generation (Xu et al., 2022). Despite continuing progress in this field (Chen et al., 2021; Li et al., 2022, 2023; Guo et al., 2024; Lozhkov et al., 2024), most research concentrates on high-resource languages like Python, Java, and JavaScript (Zan et al., 2023), typically overlooking T $_{EX}$ in evaluations. However, T $_{EX}$ and TikZ may still find their way into the training data, as demonstrated by the zero-shot ability of some models to understand and generate code in these languages (Bubeck et al., 2023; Belouadi et al., 2024; Sharma et al., 2024).

# 3 Datasets

We introduce $DAT_{IK}Z_{v2}$ , to our knowledge, the most comprehensive dataset of TikZ graphics to date; SKETCHFIG, the first dataset comprising human-created sketches of scientific figures; and METAFIG, a large-scale scientific figure dataset with rich metadata. See Appendix E for examples.

$\mathbf{DATIKZ_{v2}}$ $\mathrm{DATIKZ_{v2}}$ serves as the primary source of TikZ graphics for training $\mathrm{DETIKZIFY}$ . It is an expanded version of $\mathrm{DATIKZ_{v1}}$ (Belouadi et al., 2024), incorporating graphics from the same sources, namely curated repositories, TEX.SE, arXiv papers, and artificial examples. The key difference is that $\mathrm{DATIKZ_{v2}}$ includes all TikZ programs that compile

with $\mathrm{TEX}$ Live 2023, $^4$ regardless of whether they have associated captions, which was a requirement for inclusion in $\mathrm{DATiKZ}_{\mathrm{v1}}$ but is not needed for $\mathrm{DETiKZ_{IFY}}$ . This approach allows us to create a dataset that is more than three times as large as its predecessor (cf. Table 1).

<table><tr><td>Source</td><td> $DATIKZ_{v1}$ </td><td> $DATIKZ_{v2}$ </td></tr><tr><td>curated</td><td>981</td><td>1566</td></tr><tr><td>TEX.SE</td><td>29238</td><td>30609</td></tr><tr><td>arXiv</td><td>85656</td><td>326450</td></tr><tr><td>artificial</td><td>1957</td><td>1958</td></tr><tr><td>all</td><td>117832</td><td>360583</td></tr></table>

Table 1: Breakdown of the number of unique TikZ graphics in DAT $_{IK}$ Z $_{v2}$ compared to its predecessor DAT $_{IK}$ Z $_{v1}$ .

SKETCHFIG To create realistic synthetic sketches of scientific figures in DATIKZv2, we rely on examples of real human-created sketches. TEX.SE is a suitable source for collecting these, as users often illustrate their questions with sketches, and the answers provide the desired figure. We semi-automatically extract these figure-sketch pairs by first ranking all questions on the site that contain images based on their similarity to the string “a sketch of a scientific figure” using a multimodal vision encoder (Zhai et al., 2023). We retain the ones with high similarity scores, manually filter for true positives, and align them with the best matching figure provided in the answers. In total, we collect 549 figure-sketch pairs this way. As we also want to use this dataset for evaluation (cf. §6), we ensure that for a subset of these sketches, no code provided in the answers is included in DATIKZv2.

METAFIG Beyond TikZ graphics, there is a much larger pool of figures where the underlying source is not available. Existing datasets that collect such figures frequently come with rich metadata, such as captions, OCR tokens, and paragraphs that mention the figures (Hsu et al., 2021; Karishma et al., 2023; Rodriguez et al., 2023a). Since such high-level descriptions are useful for pretraining (cf. §4; Liu et al., 2023b), we collect these datasets and merge them with the subset of figures in DATIKZ $_{v2}$ that have captions. This results in over 734k figure-text pairs, more than twice the size of DATIKZ $_{v2}$ .

# 4 The DeTikZIFY Model

Building on previous work (Liu et al., 2023b,a; Dai et al., 2023; McKinzie et al., 2024), we build DE $_{TK}$ ZIFY by combining a pretrained vision encoder with a pretrained language model (cf. Figure 1), where the vision encoder receives figures or sketches as input images, and the language model generates corresponding TikZ programs as output. We focus on code language models that have been pretrained on T $_{EX}$ , as this prior knowledge may be helpful for our task. All the models we end up using follow the LLAMA architecture (Touvron et al., 2023): CODELLAMA (Rozière et al., 2023) has likely been trained on T $_{EX}$ code from arXiv (Touvron et al., 2023), as has been TINYLLAMA (Zhang et al.,

2024), while DEEPSEEK (code variant; Guo et al., 2024) was trained on TEX code from GitHub. For the vision encoder, we use SIGLIP (Zhai et al., 2023), which has been trained on OCR annotations (Chen et al., 2023c) and demonstrates state-of-the-art understanding of text-rich images (Tong et al., 2024; Chen et al., 2023b), a crucial skill for our task. We then condition the LLMs on SIGLIP's patch embedding vectors. To reduce the prompt length, we concatenate adjacent patch embeddings (Chen et al., 2023a). A feed-forward layer with dimensions $2\delta_{\mathrm{SIGLIP}} \times \delta_{\mathrm{LLM}}$ serves as a connector, mapping image features of dimension $\delta_{\mathrm{SIGLIP}}$ to the LLM word embedding space of dimension $\delta_{\mathrm{LLM}}$ .

Model Training We experiment with TINYLLAMA $_{1.1B}$ and DEEPSEEK $_{1.3B}$ (approximately 1 billion parameters each) and CODELLAMA $_{7B}$ and DEEPSEEK $_{7B}$ (7 billion parameters each). When referring to specific variants of DETIKZIFY, we use the names DETIKZIFY-TL $_{1.1B}$ , DETIKZIFY-DS $_{1.3B}$ , DETIKZIFY-CL $_{7B}$ , and DETIKZIFY-DS $_{7B}$ , respectively. For all models, we use the SoViT $_{400M}$ variant of SigLIP as the vision encoder. Following Liu et al. (2023b,a), we first pretrain the connector with other model parameters frozen. We pretrain for one epoch on METAFIG with ADAMW (Loshchilov and Hutter, 2019), a batch size of 256, a learning rate of 1e-3, and a cosine learning rate decay with a 3% warmup ratio. Next, we unfreeze the language model (keeping the vision encoder frozen) and fine-tune on examples from DATIKZv2 that fit within a 2048 token context window. We use a batch size of 128, a learning rate of 4e-5, and train for three epochs. Training data ablations can be found in Appendix B.

Synthetic Sketches When training $DETIKZIFY$ on $DATIKZ_{v2}$ , we randomly replace figures with synthetic sketches 50% of the time. Sketches are generated on the fly, meaning that each time a figure is sampled as a sketch, a different synthetic sketch will be generated. Creating realistic sketches requires high-level image manipulation methods that go beyond traditional transformations like zooming or cropping. We, therefore, adopt INSTRUCT-Pix2Pix (Brooks et al., 2023), a model capable of diversely editing images based on human instructions. We chose this model due to its remarkable zero-shot performance in generating synthetic sketches during our initial experiments. By then fine-tuning the model on SKETCHFIG, we further improve its performance (cf. §7 and Appendix C).

# 5 Iterative Refinement with Monte Carlo Tree Search

Due to the inherent probabilistic nature of language models, generating valid TikZ programs during inference can be a challenging task. The generated code may not always comply with the syntactic and semantic rules of $T_{E}X$ and TikZ, potentially leading to compilation errors. While constrained decoding algorithms can assist in guiding models towards generating valid programs (Ugare et al., 2024; Poesia et al., 2022; Scholak et al., 2021), these approaches are limited to programming languages defined by context-free grammars (CFGs). However, $T_{E}X$ and TikZ are not defined by CFGs (Erdweg and Ostermann, 2011), rendering these methods ineffective for our purpose. Moreover, even if the generated code compiles successfully, fidelity errors such as misaligned elements, inconsistent scaling, repetitions, or mislabeling may only become apparent in the rendered output.

Despite these challenges, which make it difficult to guide $D_{ETIK}Z_{IFY}$ based on intermediate states, we can still analyze completed outputs in a straightforward manner (e.g., by examining compiler diagnostics or comparing rendered outputs to the input image), allowing us to make informed decisions during subsequent sampling iterations. This concept of making decisions based on random sampling of the search space forms the core of Monte Carlo Tree Search (MCTS; Coulom, 2007). By integrating $D_{ETIK}Z_{IFY}$ with MCTS and adapting the standard MCTS algorithm to our problem domain, we can iteratively steer $D_{ETIK}Z_{IFY}$ towards more promising regions of the output space (cf. Figure 1). In the following, we outline our fundamental approach, with further extensions discussed in Appendix A.

# 5.1 Integrating MCTS into DETKZIFY

MCTS is a versatile search algorithm that has been successfully applied to various domains, including board games (Silver et al., 2016, 2017), procedural content generation (Kartal et al., 2016a,b; Summerville et al., 2015), and more recently, guiding language models to achieve long-term goals (Brandfonbrener et al., 2024; Zhang et al., 2023b; Chaffin et al., 2022). The algorithm incrementally builds a search tree and repeatedly runs simulations until an exit condition is met or a computational budget is exhausted. In our context, at depth n, each node's state consists of n lines of TikZ code, and edges represent continuations for generating the next line. Initially, MCTS starts with only an empty root node and then iteratively performs the following four steps (cf. Figure 2):

(i) Selection   
![](images/2ba62232b4e0a2e89b4b8cd4e351ea7bd7b23fd5fc78b889f5731b2fc172bf73.jpg)

(ii) Rollout   
![](images/9c6f5a52fb0a45ab8b695604c55a3ed83a4d6a022e8775ba3bf0960c48c336e1.jpg)

(iii) Expansion   
![](images/d8f7a6de1e064ecf335f335c1e28f6e8240aed4d0d829ea18c6072ee51198989.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Blue Node"] --> B["Green Node"]
    A --> C["Blue Node"]
    A --> D["Blue Node"]
    A --> E["Blue Node"]
    B --> F["Green Node"]
    C --> G["Blue Node"]
    D --> H["Blue Node"]
    E --> I["Blue Node"]
    style A fill:#000,stroke:#000,color:#fff
    style B fill:#000,stroke:#000,color:#fff
    style C fill:#000,stroke:#000,color:#fff
    style D fill:#000,stroke:#000,color:#fff
    style E fill:#000,stroke:#000,color:#fff
    style F fill:#000,stroke:#000,color:#fff
    style G fill:#000,stroke:#000,color:#fff
    style H fill:#000,stroke:#000,color:#fff
    style I fill:#000,stroke:#000,color:#fff
```
</details>

(iv) Backpropagation   
![](images/37c2f62ad8cfcb1ab0858699cedbb78b2b61df71efe5f4e5600d14f3d2ec4df3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["+"] --> B["+"]
    A --> C["●"]
    A --> D["●"]
    A --> E["+"]
    B --> F["●"]
    C --> G["●"]
    D --> H["●"]
    E --> I["+"]
    F --> J["●"]
    G --> K["●"]
    H --> L["●"]
    I --> M["+"]
```
</details>

Figure 2: An example of the four steps of an MCTS simulation: The selection policy (i) reaches a green backtracking node (normal nodes are blue), causing new nodes from the rollout (ii) to be added to the parent node during expansion (iii). The reward is backpropagated (iv) accordingly.

Selection Each simulation starts at the root node and successively selects child nodes based on a selection policy until a leaf node is reached. The policy determines which parts of the tree should be explored further, balancing the exploitation of high-value regions and exploration of less-visited areas. Following previous work, we use Upper Confidence Trees (UCT; Kocsis and Szepesvári, 2006) as our selection policy, iteratively selecting the successor node i that maximizes the formula

$$
\mathrm{UCT} (i) = \frac {\sum_ {j = 1} ^ {n _ {i}} V _ {i , j}}{n _ {i}} + c \sqrt {\frac {\ln (n _ {\mathrm{p} (i)})}{n _ {i}}}, \tag {1}
$$

where $V_{i,j} \in [-1,1]$ is the estimated value of i at the jth visit, $n_{i}$ and $n_{\mathrm{p}(i)}$ are the visit counts at i and its parent $\mathrm{p}(i)$ , respectively, and c is a coefficient that controls the degree of exploration.

Rollout Once a leaf node is selected, we utilize DETIKZIFY as a rollout policy. By conditioning it on the node's state, we continue to sample TikZ code until the end-of-sequence token is encountered. This so-called rollout is then stored for reuse in the subsequent steps.

Expansion Next, the tree is expanded by adding nodes from the rollout as new leaf nodes. While most implementations add only one node (i.e., one line of TikZ code) per simulation, computing rollouts with LLMs is computationally expensive. Therefore, inspired by MCTS for real-time settings (Soemers et al., 2016), we instead add multiple nodes. Specifically, we add $\sqrt{|r| - d_{l}}$ new nodes, where $|r|$ is the number of lines in rollout r and $d_{l}$ is the depth of the old leaf node l. This approach allows our tree to grow quickly in early simulations while converging to the standard case in the long run. To enable the tree to grow in multiple directions, we also introduce backtracking nodes (Brandfonbrener et al., 2024; Chaslot et al., 2008). For each added node i, we add a backtracking node as a sibling that mirrors the parent node $p(i)$ . When a backtracking node is expanded, its descendants are added to $p(i)$ so that the backtracking node remains a leaf. This enables a practically infinite search space anywhere in the tree while still maintaining a bounded branching factor.

Backpropagation Finally, we calculate the value for rollout r using a predefined reward function (cf. §5.2) and backpropagate it to every node i on the path from the root node to the newly added nodes by appending it to $V_{i,\cdot}$ . We also increment the visit counts $n_{i}$ for the same nodes. For backtracking nodes, only the visit counts are updated. Finally, we check any exit conditions. If MCTS terminates, we return the TikZ program of the rollout that achieved the highest value.

# 5.2 Reward Functions

We explore two distinct reward functions to guide the search process. The first reward function utilizes compiler diagnostics to identify documents that compile successfully. The second reward function provides a visual signal based on perceptual image similarity, which, in addition, helps find TikZ programs that better match the input image. We explore further reward functions in Appendix A.

Compiler Diagnostics The diagnostics-based reward function is based on analyzing the log file from compiling the generated TikZ program. We assign rewards according to the error state and whether an output file was produced. The reward function is defined as follows:

$$
V _ {i, j} = \left\{ \begin{array}{l l} 1 & \text { if   the   code   compiles   without   issues }, \\ 0 & \text { if   the   code   compiles   with   recoverable   errors }, \\ - 1 & \text { if   compilation   fails   due   to   a   fatal   error }. \end{array} \right. \tag {2}
$$

<table><tr><td rowspan="2">Models</td><td colspan="7">Reference Figures</td><td colspan="7">Synthetic Sketches</td></tr><tr><td> $MTE_{\uparrow}$ </td><td> $cBLEU_{\uparrow}$ </td><td> $TED_{\downarrow}$ </td><td> $DSIM_{\uparrow}$ </td><td> $SSIM_{\uparrow}$ </td><td> $KID_{\downarrow}$ </td><td> $AVG_{\uparrow}$ </td><td> $MTE_{\uparrow}$ </td><td> $cBLEU_{\uparrow}$ </td><td> $TED_{\downarrow}$ </td><td> $DSIM_{\uparrow}$ </td><td> $SSIM_{\uparrow}$ </td><td> $KID_{\downarrow}$ </td><td> $AVG_{\uparrow}$ </td></tr><tr><td>CLAUDE 3</td><td>51.812</td><td>0.111</td><td>57.389</td><td>64.896</td><td>83.372</td><td>17.822</td><td>0.148</td><td>50.156</td><td>0.024</td><td>59.731</td><td>59.102</td><td>73.954</td><td>29.541</td><td>0.189</td></tr><tr><td>GPT-4V</td><td>61.975</td><td>0.286</td><td>57.178</td><td>69.741</td><td>86.215</td><td>6.714</td><td>0.612</td><td>54.126</td><td>0.024</td><td>60.298</td><td>61.98</td><td>75.687</td><td>33.203</td><td>0.15</td></tr><tr><td>DT-TL $_{1.1B}$ </td><td>88.03</td><td>1.168</td><td>58.815</td><td>65.538</td><td>84.161</td><td>15.747</td><td>0.207</td><td>90.597</td><td>0.502</td><td>60.202</td><td>60.585</td><td>77.947</td><td>21.851</td><td>0.454</td></tr><tr><td>DT-DS $_{1.3B}$ </td><td>83.771</td><td>1.336</td><td>57.661</td><td>68.659</td><td>86.079</td><td>11.536</td><td>0.572</td><td>87.446</td><td>0.541</td><td>60.112</td><td>62.756</td><td>79.097</td><td>17.334</td><td>0.642</td></tr><tr><td>DT-CL $_{7B}$ </td><td>88.593</td><td>1.477</td><td>56.893</td><td>72.315</td><td>87.466</td><td>8.301</td><td>0.869</td><td>91.221</td><td>0.555</td><td>59.563</td><td>65.118</td><td>79.717</td><td>12.207</td><td>0.941</td></tr><tr><td>DT-DS $_{7b}$ </td><td>82.366</td><td>1.815</td><td>57.227</td><td>73.01</td><td>88.323</td><td>5.951</td><td>0.965</td><td>89.299</td><td>0.69</td><td>59.693</td><td>65.198</td><td>80.207</td><td>12.207</td><td>0.965</td></tr></table>

Table 2: System-level scores for output-driven inference (DE $_{TK}$ ZIFY abbreviated as DT). Bold and underlined values indicate the best and second-best scores for each metric column, respectively. Cell shading reflects the relative score magnitudes across input types. Arrows indicate metric directionality.

Self-Assessed Perceptual Similarity (SELFSIM) SELFSIM computes the reward as the perceptual similarity (Zhang et al., 2018) between the input image and the compiled output figure. We hypothesize that DETIKZIFY itself can assess this similarity, enabling the model to guide its own search process. To achieve this, we encode both images into embedding vectors using DETIKZIFY's vision encoder and calculate SELFSIM as their cosine similarity (Fu et al., 2023; Hessel et al., 2021). In cases where compilation fails, we assign a reward of -1. In §7, we demonstrate that SELFSIM correlates well with human judgments and outperforms other baseline methods.

# 6 Experiments

Before training on DATikZv2, we extract 1k samples to serve as our test set for an automatic evaluation and generate corresponding synthetic sketches. To mitigate data leakage from pretraining to testing, we only include items created after the cut-off date of CODELLAMA and exclude repositories that may have been used in training DEEPSEEK. We also use an n-gram matching algorithm to prevent cross-contamination with our train split (OpenAI, 2023a). For a human evaluation involving human-created sketches, we also select 100 items from SKETCHFIG that do not overlap with DATikZv2 (cf. §3). Across all models, we set the temperature to 0.8 and the exploration coefficient c to 0.6. We provide examples of real and synthetic sketches as well as generated outputs in Appendix E and Table 4.

Baselines Given CLAUDE 3 and GPT-4V's potential for our task (cf. §2), we use them as baselines. Similar to DETIKZIFY, we instruct these models to generate TikZ programs for given images. However, as proprietary chatbots, they often mix code and natural language (Zhang et al., 2023c; Belouadi et al., 2024) and do not expose the internals needed to compute SELFSIM. This makes it impractical to apply our MCTS-based refinement algorithm, which is designed for code-only outputs and open models. Instead, we compare our approach to equivalent chat-oriented refinement methods, i.e., we use Self-Refine as an alternative to diagnostics-based MCTS and Visual Self-Refine as an alternative to SELFSIM-based MCTS (Madaan et al., 2023; cf. Appendix C for additional inference details). In Appendix B, we also explore SVG as an alternative to TikZ but find it less effective for our domain.

# 6.1 Automatic Evaluation

We introduce two inference tasks to automatically evaluate our models on the test split of $DAT_{IK}Z_{v2}$ . During output-driven inference (OI), we employ the diagnostics-based reward and use successful compilation as an early exit condition (we consider compilation successful if an output artifact is produced). For time-budgeted inference (TI), we use the more fine-grained SELFSIM-based reward and continue from OI until a computational budget of 10 minutes is exhausted (cf. Brandfonbrener et al., 2024), investigating the extent of achievable improvement. We report results for the two use cases where either (rasterized) reference figures or (synthetic) sketches serve as model inputs (cf. §1). Due to high inference costs, we only evaluate commercial CLAUDE 3 and GPT-4V in OI using Self-Refine, leaving TI with Visual Self-Refine for human evaluation. We evaluate the following properties:

Code Similarity To measure the similarity between generated and reference TikZ programs, we use CRYSTALBLEU (cBLEU), a variant of BLEU optimized for evaluating code (Eghbali and Pradel, 2023; Papineni et al., 2002), and the T $_{EX}$ Edit Distance (TED), our adapted version of the Extended Edit Distance (Stanchev et al., 2019) combined with a T $_{EX}$ tokenizer.

<table><tr><td rowspan="2">Models</td><td colspan="7">Reference Figures</td><td colspan="7">Synthetic Sketches</td></tr><tr><td> $\text{MST}_{\uparrow}$ </td><td> $\text{cBLEU}_{\uparrow}$ </td><td> $\text{TED}_{\downarrow}$ </td><td> $\text{DSIM}_{\uparrow}$ </td><td> $\text{SSIM}_{\uparrow}$ </td><td> $\text{KID}_{\downarrow}$ </td><td> $\underline{\text{AVG}}_{\uparrow}$ </td><td> $\text{MST}_{\uparrow}$ </td><td> $\text{cBLEU}_{\uparrow}$ </td><td> $\text{TED}_{\downarrow}$ </td><td> $\text{DSIM}_{\uparrow}$ </td><td> $\text{SSIM}_{\uparrow}$ </td><td> $\text{KID}_{\downarrow}$ </td><td> $\underline{\text{AVG}}_{\uparrow} \\$ </td></tr><tr><td> $\text{DT-TL}_{1.1\text{B}}$ </td><td>33.775</td><td>-0.011</td><td>-2.001</td><td>+8.704</td><td>+5.561</td><td>-12.146</td><td>0.128</td><td>35.975</td><td>+0.094</td><td>-0.628</td><td>+5.82</td><td>+3.026</td><td>+0.854</td><td>0.014</td></tr><tr><td> $\text{DT-DS}_{1.3\text{B}}$ </td><td>29.975</td><td>-0.028</td><td>-1.303</td><td>+8.464</td><td>+5.108</td><td>-8.728</td><td>0.531</td><td>32.429</td><td>+0.061</td><td>-0.504</td><td>+5.573</td><td>+2.685</td><td>+5.493</td><td>0.22</td></tr><tr><td> $\text{DT-CL}_{7\text{B}}$ </td><td>25.124</td><td>+0.07</td><td>-1.351</td><td>+7.797</td><td>+4.93</td><td>-4.868</td><td>0.876</td><td>26.219</td><td>+0.073</td><td>-0.468</td><td>+5.079</td><td>+2.455</td><td>+5.493</td><td>0.681</td></tr><tr><td> $\underline{\text{DT-DS}_{7\text{b}}}$ </td><td>24.145</td><td>-0.073</td><td>-1.542</td><td>+6.974</td><td>+3.893</td><td>-0.946</td><td>0.76</td><td>26.195</td><td>+0.054</td><td>-0.696</td><td>+4.887</td><td>+2.241</td><td>+1.099</td><td>0.994</td></tr></table>

Table 3: System-level scores for time-budgeted inference, displaying relative changes for metrics shared with output-driven inference (Table 2; colored green for improvements and red for declines) and absolute scores for independent metrics. Bold and underlined values indicate the best and second-best absolute scores for each metric column, respectively. Arrows indicate metric directionality.

Image Similarity In addition to SELFSIM (SSIM), which can also be used as a metric, we report DREAMSIM (DSIM; Fu et al., 2023), a fine-tuned metric for perceptual similarity. We also compute the Kernel Inception Distance (KID × 10 $^{3}$ ; Bńkowski et al., 2018), which assesses the overall quality of generated figures by comparing their distribution with the distribution of reference figures. These metrics are always computed by comparing the generated figures to the reference figures, regardless of what the model receives as input.

Average Similarity To offer a holistic view of each model's performance, we also compute the arithmetic mean (AVG) of all code and image similarity metrics. Given that these metrics operate on different scales, we min-max normalize their scores before calculating the average.

Efficiency For OI, we compute the Mean Token Efficiency (MTE) as the 10% winsorized mean of the ratio of the number of tokens in the final TikZ program to the total number of tokens generated to arrive at that program. For TI, we instead compute the Mean Sampling Throughput (MST), measuring the throughput of unique TikZ graphics for the given budget.

Results Table 2 presents the system-level metric scores for OI. As expected, the scores for reference figures are, on average, 38% higher than those for synthetic sketches, but similar patterns emerge across both input types. $DETIKZIFY-CL_{7B}$ and $DETIKZIFY-DS_{7B}$ consistently outperform all other models, achieving AVG scores of 0.869 & 0.965 for figures and 0.941 & 0.965 for sketches, respectively. In contrast, GPT-4V reaches AVG scores of only 0.612 and 0.15, placing it in competition with the smaller 1b models: for figures, GPT-4V surpasses $DETIKZIFY-TL_{1.1B}$ and $DETIKZIFY-DS_{1.3B}$ , which achieve scores of 0.207 and 0.572, respectively. However, these smaller models outperform GPT-4V on sketches, where they achieve scores of 0.454 and 0.642. CLAUDE 3 trails behind all our models, with an AVG of only 0.148 and 0.189. When examining individual similarity metrics, $DETIKZIFY-DS_{7B}$ , the top-performing $DETIKZIFY$ model overall, surpasses GPT-4V, the best baseline, by more than 3pp (percentage points) on average for DREAMSIM and SELFSIM, while maintaining a noticeably lower KID. In terms of cBLEU, GPT-4V, and CLAUDE 3 only reach 6.5–18.5% of the performance achieved by the lowest-scoring $DETIKZIFY$ model ( $DETIKZIFY-TL_{1.1.B}$ ). The differences in TED are less pronounced, possibly due to the influence of boilerplate code, which cBLEU inherently ignores.

For efficiency, all $D_{ETIK}Z_{IFY}$ models demonstrate an MTE of 82–91%, indicating that only 1–2 out of 10 inference runs require a second simulation to generate a compilable TikZ program. Interestingly, the model size does not seem to particularly influence this score, with the pretraining setup appearing to be the key factor instead. For instance, $D_{ETIK}Z_{IFY}-TL_{1.1B}$ and $D_{ETIK}Z_{IFY}-CL_{7B}$ share a similar pretraining setup and exhibit comparable MTE values, as do $D_{ETIK}Z_{IFY}-DS_{1.3B}$ and $D_{ETIK}Z_{IFY}-DS_{7B}$ . We can further observe that (i) MTE is generally higher for sketches compared to figures, and (ii) for figures, the MTE of similarly pretrained models is inversely correlated with their scores on other metrics. These phenomena likely stem from models making fewer mistakes when the input is less detailed or when their understanding of it is limited—a finding that aligns well with other studies (Tong et al., 2024). Compared to $D_{ETIK}Z_{IFY}$ , CLAUDE 3 and GPT-4V perform considerably worse, with an MTE of only 50–62%. Notably, for these models, 98.5% of the items already compile after the initial Self-Refine step, meaning that this inefficacy primarily originates from the natural language texts surrounding the code and that Self-Refine is nearly equivalent to regular sampling-based inference.

The results for $D_{ETIK}Z_{IFY}$ on TI are presented in Table 3. Remarkably, increasing the computational budget for MCTS improves nearly all metrics for both reference figures and sketches as input without requiring access to any additional knowledge. The improvement with sketches is particularly noteworthy, as it demonstrates that the refinement process enhances the desired properties even when

![](images/a03a5624495a4d0966586064cf468d2c777e53d90b26ad3c1b00ea058a1d4a66.jpg)

<details>
<summary>contour</summary>

| Reference Figures | Human Sketches | Model/Category |
| ----------------- | -------------- | -------------- |
| -1 to 1          | -1 to 1        | GPT-4V (OI), GPT-4V (TI), DEiT1KZIFY-DS7B (OI), DEiT1KZIFY-DS7B (TI) |
</details>

![](images/764526e8fc6110a684a6614cc12b532408e0aac6cc83691d0e64e2fc78bebe17.jpg)

<details>
<summary>line</summary>

| Time (seconds) | Sampling | MCTS |
| -------------- | -------- | ---- |
| 100            | 40.5     | 39.0 |
| 200            | 40.8     | 42.5 |
| 300            | 40.7     | 44.0 |
| 400            | 40.9     | 45.5 |
| 500            | 40.6     | 46.0 |
| 600            | 40.5     | 46.5 |
</details>

Figure 3: Bivariate distributions of BWS scores (higher is better) using kernel density estimation (left) and log-linear regression over TI reward scores for different generation strategies over time (right).

the model input type differs from the one used for evaluation. The 2.2–5.6pp increase of SELFSIM for all models is not surprising since it serves as the reward signal we optimize, but DREAMSIM and TED also increase by 4.9–8.7pp and 0.5–2pp, respectively, demonstrating the efficacy of our approach. While KID improves by 1–12.1 points with reference figures, it drops by 0.9–5.5 points with sketches. We believe this is because sketches often omit minor details, such as axis tick labels, which is reflected more in the output of the TI models, biasing their overall output distributions. Therefore, we consider the substantial improvement of metrics capturing instance-level similarities to be more important. For cBLEU, we observe only minor changes (less than ±0.1pp), aligning with findings that BLEU-based metrics become less effective as performance increases (Ma et al., 2019). The MST and AVG reveal that, although 1b models produce more unique outputs within the time frame compared to their larger 7b counterparts (30–36 vs. 24.1–26.2), they still fail to close the overall gap in performance, with AVG scores ranging between 0.014–0.531 compared to 0.681–0.994 for 7b models.

Overall, all $DeTiKZIFY$ models are capable of generating compilable outputs with reasonable efficiency. Upon examination of these outputs, it becomes evident that the 7b models, particularly $DeTiKZIFY-DS_{7B}$ , consistently outperform both CLAUDE 3 and GPT-4V, whose performance is more comparable to the 1b range. Increasing the computational budget for $DeTiKZIFY$ further improves performance.

# 6.2 Human Evaluation

To further assess the quality of the generated figures, we perform a human evaluation on SKETCHFIG using Best-Worst Scaling (BWS; Louviere et al., 2015; Kiritchenko and Mohammad, 2016, 2017). In this process, for each reference figure, we present annotators with a tuple of generated figures and ask them to identify the most and least perceptually similar figure. We then transform this data into scores ranging from -1 (poor) to 1 (excellent) by calculating the difference between the proportion of times a figure is selected as the best and the proportion of times it is chosen as the worst (Orme, 2009). To keep the workload manageable, we focus on the most promising DE $T_{IK}$ ZIFY model (DE $T_{IK}$ ZIFY-DS $_{7_{B}}$ ) and the strongest baseline (GPT-4V). Building upon the automatic evaluation, we assess these models in the OI and TI configurations, using either reference figures or human-created sketches as input. For each input type, we engage six unique expert annotators (cf. Appendix D for more details).

Results Figure 3 (left) shows kernel density estimates for the computed BWS scores, revealing intriguing findings that are consistent across input types. In contrast to the automatic evaluation, $D_{ETIK}Z_{IFY}-DS_{7B}$ performs worse (mean score $\mu = -0.32$ ) than GPT-4V ( $\mu = 0.09$ ) in OI. This could be attributed to the fact that T $_{EX}$ .SE, the sole source of SKETCHFIG, emphasizes minimum working examples, a type on which GPT-4V particularly excels (Belouadi et al., 2024). However, when we increase the computational budget, as in $D_{ETIK}Z_{IFY}-DS_{7B}$ (TI), it not only improves over OI results ( $\mu = 0.39$ ; in line with automatic evaluation) but also surpasses GPT-4V in both configurations by a considerable margin. Interestingly, GPT-4V's performance in TI ( $\mu = -0.16$ ) is lower than its performance in OI, indicating that GPT-4V (TI) struggles to refine its own outputs effectively and quickly deteriorates. Overall, this shows how difficult it is for models to refine their own outputs and highlights the effectiveness of our MCTS-based approach. Example outputs are provided in Table 4.

<table><tr><td>Input</td><td>GPT-4V (OI)</td><td>GPT-4V (TI)</td><td> $DeT1KZIFY-DS_{7B}$ (OI)</td><td> $DeT1KZIFY-DS_{7B}$ (TI)</td></tr><tr><td><img src="images/4bb9cee6ad46cca82a1861d4b7430b8001ee224ac4e7b7e9d5d6e2cb339266be.jpg"/></td><td><img src="images/08bc9f536930e29ffa6c7266eadc6426df3b9c2a578da6db2205737162647335.jpg"/></td><td><img src="images/05dfc2f5006d915398d7d98092be1bc8b5932e40d98246068d9516db06d1edfe.jpg"/></td><td><img src="images/08542f0d3e1fde8d7b5688573dab9ce7bdc31a110b8891837d6ee3e6c07b407a.jpg"/></td><td><img src="images/0ee898a98f7b3a091374dc17b2842ed83276711036ecbd0e88d7e368f924a1ba.jpg"/></td></tr><tr><td><img src="images/50c3f6dcf24cfc9d85b5480a8f37bb283bd75ba95f60c6a60f6d0e722c8e5ba8.jpg"/></td><td><img src="images/b30c3e2829262af89f8ea1ad60e213ac6aab49b418af6f137b06318c6a18e0b4.jpg"/></td><td><img src="images/ce4f54b32e3f781bc77b37d6470c172042ce0fe1fe164b539412bd06933c3bf2.jpg"/></td><td><img src="images/c1dab9b081b558616850951af702e313392fc72edfd6b7845a1c3cc0cb3243bb.jpg"/></td><td><img src="images/f42ec2f4e963368f142c477b602cf8557dc2e400dc8361fe36a7f8c36c02bc98.jpg"/></td></tr><tr><td><img src="images/119b6cbc457434ced9cd8b5d745fe777920d42005940ea62fa2d0bf92f734da4.jpg"/></td><td><img src="images/e6ee86cb13def4c2ea4d51e565c736f358007c6677de4814924bca94d829b4da.jpg"/></td><td><img src="images/92fbb76f439935f941381e7262731d16aad088feea1f60999d73b6bed0f6a24b.jpg"/></td><td><img src="images/2756c46b87401dce7da96342c60899bfa11879a3edf5bbf4bdc8bbb8569c368a.jpg"/></td><td><img src="images/f1d724f26b2da932a339edbb1c8ee138fcf679d56cc4e2ca1a4e7ef525a3914b.jpg"/></td></tr></table>

Table 4: Examples of model inputs and generated outputs from our human evaluation, where annotators rated GPT-4V (OI) higher than $D_{ETIK}Z_{IFY}-DS_{7_{B}}$ (OI) but ranked $D_{ETIK}Z_{IFY}-DS_{7_{B}}$ (TI) as the overall best model, illustrating our findings in §6.2. See Appendix E for more examples.

# 7 Analysis

In this section, we take a closer look at our methodologies and evaluation strategies, correlating evaluation metrics with human judgments, quantifying the quality of synthetic sketches, and examining the rate of convergence of our MCTS algorithm. We also demonstrate that our models are not affected by memorization of the training data, as shown in Appendix B.

Correlating Humans and Metrics To assess the reliability of our human evaluation results, we investigate the agreement between annotators. To this end, we calculate the split-half reliability (SHR; Kiritchenko and Mohammad, 2017) by randomly splitting our annotations into two subsets, computing BWS scores for each subset, and measuring their correlation with Spearman's $\rho$ . The SHR values of 0.69 for sketches and 0.75 for images indicate a moderate to strong correlation between annotators, supporting the validity of our human evaluation results. Motivated by these findings, we explore whether metrics that also assess perceptual

<table><tr><td>Metric</td><td>Segment</td><td>System</td></tr><tr><td>LPIPS</td><td>0.224</td><td>0.642</td></tr><tr><td>DISTS</td><td>0.32</td><td>0.642</td></tr><tr><td>DSIM</td><td>0.424</td><td>0.954</td></tr><tr><td>SSIM</td><td>0.436</td><td>0.642</td></tr></table>

Table 5: Correlations of image similarity metrics with humans at the segment and system level.

similarity (i.e., SELFSIM and DREAMSIM) correlate with these human judgments. We again calculate Spearman's $\rho$ and show the average correlations (David M. Corey and Burke, 1998) at the segment and system level in Table 5. For comparison, we also include the popular LPIPS and DISTS metrics (Zhang et al., 2018; Ding et al., 2020). At the segment level, SELFSIM outperforms all other metrics, which is remarkable considering it is the only untrained metric. Segment-level performance is particularly important for fine-grained reward functions, justifying our choice of SELFSIM in our MCTS algorithm. At the system level, DREAMSIM performs the best, showcasing its strength in evaluation settings.

Synthetic Sketch Quality We also assess the quality of our synthetic sketches by measuring their congruence coefficient (Lorenzo-Seva and ten Berge, 2006) with real sketches. We embed human-created figure-sketch pairs from SKETCHFIG using SigLIP, subtract each sketch embedding from the corresponding figure embedding to obtain local sketch vectors, and perform a single-component Principal Component Analysis to derive a global sketch vector (Zou et al., 2023). We repeat this process for synthetic sketches generated for the test split of DATIKZv2 and compare the global vectors using cosine similarity. Base INSTRUCT-Pix2Pix generates synthetic sketches with a congruence coefficient of 0.66, which increases to 0.7 after fine-tuning. These results demonstrate a high correlation with human-created sketches, suggesting that our generated sketches are of good quality.

MCTS Convergence To gain insights into the long-term characteristics of our MCTS algorithm, we visualize the trends in achieved TI reward scores over time in Figure 3 (right) and compare them to conventional sampling-based inference. As expected, sampling does not lead to improvements over time due to the absence of a feedback loop. In contrast, MCTS consistently improves throughout the entire time frame, and even at the end of our budget of 10 minutes, it does not appear to converge, suggesting potential additional gains for larger budgets. Apart from this, MCTS is not only more effective but also faster. With an average MST of 25.17, compared to 18.7 for sampling, our MCTS algorithm generates considerably more unique TikZ programs within the same amount of time.

# 8 Conclusion

In this work, we showcase the potential of $D_{ETIK}Z_{IFY}$ in generating TikZ programs for two practical use cases. First, it can convert existing figures from lower-level formats into TikZ, paving the way for semantic image editing and downstream tasks (Zhang et al., 2023a). Second, it can develop hand-drawn sketches into TikZ graphics, which could aid researchers in creating high-quality scientific illustrations. In both cases, $D_{ETIK}Z_{IFY}$ substantially outperforms the commercial LLMs GPT-4V and CLAUDE 3 despite its presumably much smaller size. We hope that our datasets ( $DAT_{IK}Z_{v2}$ , SKETCHFIG, and METAFIG), our method for generating synthetic sketches, and our MCTS-based inference algorithm will pave the way towards future research on graphics program synthesis and bolster the cause of open science.

Looking ahead, we plan to extend our approach to other graphics languages, such as MetaPost, PSTricks or Asymptote (Hobby, 2014; Van Zandt, 2007; Hammerlindl et al., 2024). We also intend to explore alternatives to perceptual similarity as an MCTS reward signal, including per-pixel measures and point cloud metrics (Wang and Bovik, 2009; Wu et al., 2021). In addition, we aim to investigate reinforcement learning from reward functions, for example, using Direct Preference Optimization (Rafailov et al., 2023; Xu et al., 2024). Finally, while this work focuses on visual inputs, we plan to explore additional modalities, such as text and mixed-modality inputs, in future work.

# Limitations

In this work, we compare openly available models with proprietary systems that lack transparency in their training details and internal workings and whose performance is not stable over time. This inevitably complicates efforts to address concerns such as data leakage or cross-contamination and limits the fairness and reproducibility of our experiments. Nevertheless, under these adverse conditions, our open models and methods demonstrate favorable performance. Users should be aware, however, that our models might inherit biases, flaws, or other limitations present in the training data, potentially leading to discrepancies between expected results and generated outputs. Furthermore, given the resource-intensive nature of LLMs, many of our training and inference hyper-parameters were adopted from related work or chosen based on general intuition. Although LLMs are generally robust to hyper-parameter selection (Beyer et al., 2024), conducting a thorough hyper-parameter search might enhance their performance further. Finally, it should be noted that our models could potentially be misused by malicious actors to produce misinformation and fake science.

Another important consideration is that the public release of DATIKZ $_{v2}$ does not include some TikZ programs from our internal version due to licensing restrictions. These programs are distributed under the arXiv.org perpetual, non-exclusive license, which prohibits redistribution. Nonetheless, we provide our dataset creation scripts alongside usage instructions, enabling anyone to reproduce the full version of DATIKZ $_{v2}$ independently. The remaining TikZ programs in DATIKZ $_{v2}$ are licensed under Creative Commons attribution licenses, $^{5}$ the GNU Free Documentation License, $^{6}$ or the MIT license, $^{7}$ and their respective terms and conditions apply. Regarding artificially created examples, OpenAI's terms of use restrict the use of their services for creating competing products, limiting this subset of DATIKZ $_{v2}$ to non-commercial applications. $^{8}$

# Acknowledgments

We would like to express our sincere gratitude to the following individuals for their contributions to our work: JiWoo Kim, Tommaso Green, Christoph Leiter, Ines Reinig, Martin Kerscher, Margret Keuper, Christopher Klamm, Daniil Larionov, Yanran Chen, Tornike Tsereteli, and Daniel Ruffinelli. Their assistance with our human evaluation campaign, proofreading, insightful discussions, and constructive comments have been invaluable. The last author is supported by the Federal Ministry of Education and Research (BMBF) via the research grant METRICS4NLG and the German Research Foundation (DFG) via the Heisenberg Grant EG 375/5–1. We would also like to acknowledge the OpenMoji project for providing the open-source icons used throughout this work and Hugging Face for their generous community GPU grant.

# References

Anthropic. 2024. The Claude 3 model family: Opus, Sonnet, Haiku.   
Jonas Belouadi and Steffen Eger. 2023. UScore: An effective approach to fully unsupervised evaluation metrics for machine translation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 358–374, Dubrovnik, Croatia. Association for Computational Linguistics.   
Jonas Belouadi, Anne Lauscher, and Steffen Eger. 2024. AutomaTikZ: Text-guided synthesis of scientific vector graphics with TikZ. In The Twelfth International Conference on Learning Representations.   
Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. 2024. PaliGemma: A versatile 3b VLM for transfer. Preprint, arXiv:2407.07726.   
Mikołaj Bińkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. 2018. Demystifying MMD GANs. In International Conference on Learning Representations.   
Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. 2024. Nougat: Neural optical understanding for academic documents. In The Twelfth International Conference on Learning Representations.   
Ali Borji. 2023. Qualitative failures of image generation models and their application in detecting deepfakes. Image and Vision Computing, 137:104771.   
David Brandfonbrener, Sibi Raja, Tarun Prasad, Chloe Loughridge, Jianang Yang, Simon Henniger, William E. Byrd, Robert Zinkov, and Nada Amin. 2024. Verified multi-step synthesis using large language models and Monte Carlo tree search. Preprint, arXiv:2402.08147.   
Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. InstructPix2Pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18392–18402.   
Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. 2023. Sparks of artificial general intelligence: Early experiments with GPT-4. Preprint, arXiv:2303.12712.   
Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. 2020. DeepSVG: A hierarchical generative network for vector graphics animation. In Advances in Neural Information Processing Systems, volume 33, pages 16351–16361. Curran Associates, Inc.   
Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. 2023. Quantifying memorization across neural language models. In The Eleventh International Conference on Learning Representations.

Antoine Chaffin, Vincent Claveau, and Ewa Kijak. 2022. PPL-MCTS: Constrained textual generation through discriminator-guided MCTS decoding. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2953–2967, Seattle, United States. Association for Computational Linguistics.   
Guillaume M. J-B. Chaslot, Mark H. M. Winands, H. Jaap Van Den Herik, Jos W. H. M. Uiterwijk, and Bruno Bouzy. 2008. Progressive strategies for Monte-Carlo tree search. New Mathematics and Natural Computation (NMNC), 4(03):343–357.   
Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. 2023a. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning. Preprint, arXiv:2310.09478.   
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. Preprint, arXiv:2107.03374.   
Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Soricut. 2023b. PaLI-3 vision language models: Smaller, faster, stronger. Preprint, arXiv:2310.09199.   
Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme Ruiz, Andreas Peter Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. 2023c. PaLI: A jointly-scaled multilingual language-image model. In The Eleventh International Conference on Learning Representations.   
Rémi Coulom. 2007. Efficient selectivity and backup operators in Monte-Carlo tree search. In Computers and Games, pages 72–83, Berlin, Heidelberg. Springer Berlin Heidelberg.   
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems.   
Giannis Daras and Alex Dimakis. 2022. Discovering the hidden vocabulary of DALLE-2. In NeurIPS 2022 Workshop on Score-Based Methods.   
William P. Dunlap David M. Corey and Michael J. Burke. 1998. Averaging correlations: Expected values and bias in combined pearson rs and fisher's z transformations. The Journal of General Psychology, 125(3):245–261.   
Yuntian Deng, Anssi Kanervisto, Jeffrey Ling, and Alexander M. Rush. 2017. Image-to-markup generation with coarse-to-fine attention. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 980–989. PMLR.   
Harsh Desai, Pratik Kayal, and Mayank Singh. 2021. TabLeX: A benchmark dataset for structure and content information extraction from scientific tables. In Document Analysis and Recognition – ICDAR 2021, pages 554–569, Cham. Springer International Publishing.

James Richard Diebel. 2008. Bayesian image vectorization: The probabilistic inversion of vector image rasterization. Ph.D. thesis, Stanford University, Stanford, CA, USA. AAI3332816.   
Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. 2020. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567–2581.   
Aryaz Eghbali and Michael Pradel. 2023. CrystalBLEU: Precisely and efficiently measuring the similarity of code. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, ASE '22, New York, NY, USA. Association for Computing Machinery.   
Kevin Ellis, Daniel Ritchie, Armando Solar-Lezama, and Josh Tenenbaum. 2018. Learning to infer graphics programs from hand-drawn images. In Thirty-second Conference on Neural Information Processing Systems, pages 6062–6071.   
Sebastian Thore Erdweg and Klaus Ostermann. 2011. Featherweight TeX and parser correctness. In Software Language Engineering, pages 397–416, Berlin, Heidelberg. Springer Berlin Heidelberg.   
Stephanie Fu, Netanel Yakir Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. 2023. DreamSim: Learning new dimensions of human visual similarity using synthetic data. In Thirty-seventh Conference on Neural Information Processing Systems.   
Yaroslav Ganin, Tejas Kulkarni, Igor Babuschkin, S. M. Ali Eslami, and Oriol Vinyals. 2018. Synthesizing programs for images using reinforced adversarial learning. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1666–1675. PMLR.   
Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. 2024. DeepSeek-Coder: When the large language model meets programming – the rise of code intelligence. Preprint, arXiv:2401.14196.   
Andy Hammerlindl, John Bowman, and Tom Prince. 2024. Asymptote: The Vector Graphics Language.   
Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. CLIPScore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.   
John D. Hobby. 2014. MetaPost.   
Ting-Yao Hsu, C Lee Giles, and Ting-Hao Huang. 2021. SciCap: Generating captions for scientific figures. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3258–3264, Punta Cana, Dominican Republic. Association for Computational Linguistics.   
Chieh-Yang Huang, Ting-Yao Hsu, Ryan Rossi, Ani Nenkova, Sungchul Kim, Gromit Yeuk-Yin Chan, Eunyee Koh, C Lee Giles, and Ting-Hao Huang. 2023. Summaries as captions: Generating figure captions for scientific documents with automated text summarization. In Proceedings of the 16th International Natural Language Generation Conference, pages 80–92, Prague, Czechia. Association for Computational Linguistics.   
Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. 2024. Prismatic VLMs: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning.   
Zeba Karishma, Shaurya Rohatgi, Kavya Shrinivas Puranik, Jian Wu, and C. Lee Giles. 2023. ACL-Fig: A dataset for scientific figure classification. In Proceedings of the Workshop on Scientific Document Understanding co-located with 37th AAAI Conference on Artificial Intelligence (AAAI 2023), Remote, February 14, 2023, volume 3656 of CEUR Workshop Proceedings. CEUR-WS.org.   
Bilal Kartal, Nick Sohre, and Stephen Guy. 2016a. Data driven Sokoban puzzle generation with Monte Carlo tree search. Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, 12(1):58–64.

Bilal Kartal, Nick Sohre, and Stephen Guy. 2016b. Generating Sokoban puzzle game levels with Monte Carlo tree search. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI-16, The IJCAI-16 Workshop on General Game Playing, pages 47–54. International Joint Conferences on Artificial Intelligence Organization.   
Svetlana Kiritchenko and Saif Mohammad. 2017. Best-worst scaling more reliable than rating scales: A case study on sentiment intensity annotation. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 465–470, Vancouver, Canada. Association for Computational Linguistics.   
Svetlana Kiritchenko and Saif M. Mohammad. 2016. Capturing reliable fine-grained sentiment associations by crowdsourcing and best-worst scaling. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 811–817, San Diego, California. Association for Computational Linguistics.   
Daniel Kirsch. 2010. Detexify: Recognition of hand-drawn LaTeX symbols. Diploma thesis, University of Münster, Münster, Germany, October.   
Levente Kocsis and Csaba Szepesvári. 2006. Bandit based Monte-Carlo planning. In Machine Learning: ECML 2006, pages 282–293, Berlin, Heidelberg. Springer Berlin Heidelberg.   
Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. 2015. From word embeddings to document distances. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 957–966, Lille, France. PMLR.   
Mengtian Li, Zhe Lin, Radomir Mech, Ersin Yumer, and Deva Ramanan. 2019. Photo-Sketching: Inferring contour drawings from images. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1403–1412.   
Raymond Li, Loubna Ben allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia LI, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Joel Lamy-Poirier, Joao Monteiro, Nicolas Gontier, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Ben Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason T Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Urvashi Bhattacharyya, Wenhao Yu, Sasha Luccioni, Paulo Villegas, Fedor Zhdanov, Tony Lee, Nadav Timor, Jennifer Ding, Claire S Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro Von Werra, and Harm de Vries. 2023. StarCoder: may the source be with you! Transactions on Machine Learning Research. Reproducibility Certification.   
Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, and Jonathan Ragan-Kelley. 2020. Differentiable vector graphics rasterization for editing and learning. ACM Trans. Graph., 39(6).   
Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d'Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. 2022. Competition-level code generation with AlphaCode. Science, 378(6624):1092–1097.   
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. Preprint, arXiv:2310.03744.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems.   
Raphael Gontijo Lopes, David Ha, Douglas Eck, and Jonathon Shlens. 2019. A learned representation for scalable vector graphics. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Urbano Lorenzo-Seva and Jos M. F. ten Berge. 2006. Tucker's congruence coefficient as a meaningful index of factor similarity. Methodology: European Journal of Research Methods for the Behavioral and Social Sciences, 2(2):57–64.   
Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.   
Jordan J. Louviere, Terry N. Flynn, and A. A. J. Marley. 2015. Best-Worst Scaling: Theory, Methods and Applications. Cambridge University Press.   
Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2024. StarCoder 2 and The Stack v2: The next generation. Preprint, arXiv:2402.19173.   
Xinyuan Lu, Liangming Pan, Qian Liu, Preslav Nakov, and Min-Yen Kan. 2023. SCITAB: A challenging benchmark for compositional reasoning and claim verification on scientific tables. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7787–7813, Singapore. Association for Computational Linguistics.   
Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei. 2023. Kosmos-2.5: A multimodal literate model. Preprint, arXiv:2309.11419.   
Qingsong Ma, Johnny Wei, Ondřej Bojar, and Yvette Graham. 2019. Results of the WMT19 metrics shared task: Segment-level and strong MT systems pose big challenges. In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 62–90, Florence, Italy. Association for Computational Linguistics.   
Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi. 2022. Towards layer-wise image vectorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16314–16323.   
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Thirty-seventh Conference on Neural Information Processing Systems.   
Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland. Association for Computational Linguistics.   
R. Thomas McCoy, Paul Smolensky, Tal Linzen, Jianfeng Gao, and Asli Celikyilmaz. 2023. How much do language models copy from their training data? evaluating linguistic novelty in text generation using RAVEN. Transactions of the Association for Computational Linguistics, 11:652–670.   
Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, Anton Belyi, Haotian Zhang, Karanjeet Singh, Doug Kang, Ankur Jain, Hongyu Hè, Max Schwarzer, Tom Gunter, Xiang Kong, Aonan Zhang, Jianyu Wang, Chong Wang, Nan Du, Tao Lei, Sam Wiseman, Guoli Yin, Mark Lee, Zirui Wang, Ruoming Pang, Peter Grasch, Alexander Toshev, and Yinfei Yang. 2024. MM1: Methods, analysis & insights from multimodal LLM pre-training. Preprint, arXiv:2403.09611.

Casey Meehan, Kamalika Chaudhuri, and Sanjoy Dasgupta. 2020. A three sample hypothesis test for evaluating generative models. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 3546–3556. PMLR.   
Nafise Sadat Moosavi, Andreas Rücklé, Dan Roth, and Iryna Gurevych. 2021. SciGen: a dataset for reasoning-aware text generation from scientific tables. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).   
OpenAI. 2023a. GPT-4 technical report. Preprint, arXiv:2303.08774.   
OpenAI. 2023b. GPT-4V(ision) system card.   
Bryan K. Orme. 2009. MaxDiff analysis: Simple counting, individual-level logit, and HB. Sawtooth Software Research Paper Series.   
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.   
Sayak Paul. 2023. Instruction-tuning Stable Diffusion with InstructPix2Pix. Hugging Face Blog. https://huggingface.co/blog/instruction-tuning-sd.   
Gabriel Poesia, Alex Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, Christopher Meek, and Sumit Gulwani. 2022. Synchronesh: Reliable code generation from pre-trained language models. In International Conference on Learning Representations.   
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems.   
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. ZeRO: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC '20. IEEE Press.   
Vikas Raunak and Arul Menezes. 2022. Finding memo: Extractive memorization in constrained sequence generation tasks. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5153–5162, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.   
Pradyumna Reddy. 2021. Im2Vec: Synthesizing vector graphics without vector supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 2124–2133.   
J. A. Rodriguez, D. Vazquez, I. Laradji, M. Pedersoli, and P. Rodriguez. 2023a. OCR-VQGAN: Taming text-within-image generation. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3678–3687, Los Alamitos, CA, USA. IEEE Computer Society.   
Juan A. Rodriguez, Shubham Agarwal, Issam H. Laradji, Pau Rodriguez, David Vazquez, Christopher Pal, and Marco Pedersoli. 2023b. StarVector: Generating scalable vector graphics code from images. Preprint, arXiv:2312.11556.   
Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2023. Code LLaMA: Open foundation models for code. Preprint, arXiv:2308.12950.   
Y. Rubner, C. Tomasi, and L.J. Guibas. 1998. A metric for distributions with applications to image databases. In Sixth International Conference on Computer Vision (IEEE Cat. No.98CH36271), pages 59–66.

Patsorn Sangkloy, Nathan Burnell, Cusuh Ham, and James Hays. 2016. The sketchy database: learning to retrieve badly drawn bunnies. ACM Trans. Graph., 35(4).   
Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. 2021. PICARD: Parsing incrementally for constrained auto-regressive decoding from language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9895–9901, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.   
Pratyusha Sharma, Tamar Rott Shaham, Manel Baradad, Stephanie Fu, Adrian Rodriguez-Munoz, Shivam Duggal, Phillip Isola, and Antonio Torralba. 2024. A vision check-up for language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14410–14419.   
David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Vedavyas Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy P. Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. 2016. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484–489.   
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy P. Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. 2017. Mastering the game of go without human knowledge. Nature, 550(7676):354–359.   
Dennis J. N. J. Soemers, Chiara F. Sironi, Torsten Schuster, and Mark H. M. Winands. 2016. Enhancements for real-time monte-carlo tree search in general video game playing. In 2016 IEEE Conference on Computational Intelligence and Games (CIG), pages 1–8.   
Yurun Song, Junchen Zhao, and Lucia Specia. 2021. SentSim: Crosslingual semantic evaluation of machine translation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3143–3156, Online. Association for Computational Linguistics.   
Peter Stanchev, Weiyue Wang, and Hermann Ney. 2019. EED: Extended edit distance measure for machine translation. In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 514–520, Florence, Italy. Association for Computational Linguistics.   
Adam Summerville, Shweta Philip, and Michael Mateas. 2015. MCMCTS PCG 4 SMB: Monte Carlo tree search to guide platformer level generation. Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, 11(3):68–74.   
Jian Sun, Lin Liang, Fang Wen, and Heung-Yeung Shum. 2007. Image vectorization using optimized gradient meshes. ACM Trans. Graph., 26(3):11–es.   
Masakazu Suzuki, Fumikazu Tamari, Ryoji Fukuda, Seiichi Uchida, and Toshihiro Kanahori. 2003. INFTY: an integrated OCR system for mathematical documents. In Proceedings of the 2003 ACM Symposium on Document Engineering, DocEng '03, page 95–104, New York, NY, USA. Association for Computing Machinery.   
Till Tantau. 2023. The TikZ and PGF Packages.   
Xingze Tian and Tobias Günther. 2024. A survey of smooth vector graphics: Recent advances in representation, creation, rasterization, and image vectorization. IEEE Transactions on Visualization and Computer Graphics, 30(3):1652–1671.   
Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9568–9578.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Shubham Ugare, Tarun Suresh, Hangoo Kang, Sasa Misailovic, and Gagandeep Singh. 2024. Improving LLM code generation with grammar augmentation. Preprint, arXiv:2403.01632.   
Timothy van Zandt. 2007. PSTricks: PostScript macros for Generic TeX.   
Zelun Wang and Jyh-Charn Liu. 2020. PDF2LaTeX: A deep learning system to convert mathematical documents from PDF to LaTeX. In Proceedings of the ACM Symposium on Document Engineering 2020, DocEng '20, New York, NY, USA. Association for Computing Machinery.   
Zelun Wang and Jyh-Charn Liu. 2021. Translating math formula images to LaTeX sequences using deep neural networks with sequence-level training. International Journal on Document Analysis and Recognition (IJDAR), 24(1):63–75.   
Zhou Wang and Alan C. Bovik. 2009. Mean squared error: Love it or leave it? a new look at signal fidelity measures. IEEE Signal Processing Magazine, 26(1):98–117.   
Jin-Wen Wu, Fei Yin, Yan-Ming Zhang, Xu-Yao Zhang, and Cheng-Lin Liu. 2020. Handwritten mathematical expression recognition via paired adversarial learning. International Journal of Computer Vision, 128(10):2386–2401.   
Tong Wu, Liang Pan, Junzhe Zhang, Tai WANG, Ziwei Liu, and Dahua Lin. 2021. Balanced Chamfer distance as a comprehensive metric for point cloud completion. In Advances in Neural Information Processing Systems.   
Frank F. Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. 2022. A systematic evaluation of large language models of code. In MAPS@PLDI 2022: 6th ACM SIGPLAN International Symposium on Machine Programming, San Diego, CA, USA, 13 June 2022, pages 1–10. ACM.   
Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. 2024. Contrastive preference optimization: Pushing the boundaries of LLM performance in machine translation. In Forty-first International Conference on Machine Learning.   
Daoguang Zan, Bei Chen, Fengji Zhang, Dianjie Lu, Bingchao Wu, Bei Guan, Wang Yongji, and Jian-Guang Lou. 2023. Large language models meet NL2Code: A survey. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7443–7464, Toronto, Canada. Association for Computational Linguistics.   
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11975–11986.   
Jianshu Zhang, Jun Du, and Lirong Dai. 2017. A GRU-based encoder-decoder approach with attention for online handwritten mathematical expression recognition. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), volume 01, pages 902–907.   
Peiying Zhang, Nanxuan Zhao, and Jing Liao. 2023a. Text-guided vector graphics customization. In SIGGRAPH Asia 2023 Conference Papers, SA '23, New York, NY, USA. Association for Computing Machinery.   
Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. 2024. TinyLlama: An open-source small language model. Preprint, arXiv:2401.02385.   
Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595.   
Shun Zhang, Zhenfang Chen, Yikang Shen, Mingyu Ding, Joshua B. Tenenbaum, and Chuang Gan. 2023b. Planning with large language models for code generation. In The Eleventh International Conference on Learning Representations.   
Tianjun Zhang, Yi Zhang, Vibhav Vineet, Neel Joshi, and Xin Wang. 2023c. Controllable text-to-image generation with GPT-4. Preprint, arXiv:2305.18583.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations.   
Wei Zhang, Zhiqiang Bai, and Yuesheng Zhu. 2019. An improved approach based on CNN-RNNs for mathematical expression recognition. In Proceedings of the 2019 4th International Conference on Multimedia Systems and Signal Processing, ICMSSP '19, page 57–61, New York, NY, USA. Association for Computing Machinery.   
Wei Zhao, Goran Glavaš, Maxime Peyrard, Yang Gao, Robert West, and Steffen Eger. 2020. On the limitations of cross-lingual encoders as exposed by reference-free machine translation evaluation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1656–1671, Online. Association for Computational Linguistics.   
Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, and Steffen Eger. 2019. MoverScore: Text generation evaluating with contextualized embeddings and earth mover distance. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 563–578, Hong Kong, China. Association for Computational Linguistics.   
Haokun Zhu, Juang Ian Chong, Teng Hu, Ran Yi, Yu-Kun Lai, and Paul L. Rosin. 2024. SAMVG: A multi-stage image vectorization model with the segment-anything model. In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4350–4354.   
Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. 2023. Representation engineering: A top-down approach to AI transparency. Preprint, arXiv:2310.01405.

# A Further Details on MCTS

In this section, we discuss several extensions to our MCTS algorithm that aim to improve its performance and efficiency. We also explain alternative reward functions that we experimented with but ultimately found less effective than our chosen approaches.

# A.1 MCTS Enhancements

Building on our base MCTS implementation, we introduce several enhancements, namely dynamic rescaling of visual rewards, node deduplication, and preemptive stopping of faulty rollouts.

Dynamic Rescaling One challenge when using SELFSIM is that MCTS expects values to be in the range of $[-1, 1]$ , while deep encoders often work with a much narrower range in practice (Hessel et al., 2021; Zhang et al., 2020). Furthermore, this range may vary depending on whether the input image is a real figure or a sketch. To address this discrepancy, we propose dynamically min-max normalizing the visual reward scores whenever they are (re)computed, ensuring that MCTS always operates on the full range. The modified reward formula is as follows:

$$
V _ {i, j} ^ {\prime} = \left\{ \begin{array}{l l} \frac {V _ {i , j} - \min (V _ {i , :} \setminus \{- 1 \})}{\max (V _ {i , :}) - \min (V _ {i , :} \setminus \{- 1 \})} & \text { if } V _ {i, j} \neq - 1 \text { and } \max (V _ {i,:}) \neq \min (V _ {i,:} \setminus \{- 1 \}), \\ 0 & \text { if } V _ {i, j} \neq - 1 \text { and } \max (V _ {i,:}) = \min (V _ {i,:} \setminus \{- 1 \}), \\ - 1 & \text { otherwise. } \end{array} \right. \tag {3}
$$

Node Deduplication During a rollout for a backtracking node, it is possible to generate code that already exists elsewhere in the tree (i.e., in siblings and their descendants). To prevent the duplication of nodes, we always merge identical node states before adding any nodes to the tree.

Preemptive Stopping If the code generated in a rollout cannot be compiled due to a fatal error, we record the rollout, including the state in which the faulty line of code was first introduced. If the same (intermediate) state is sampled again during subsequent rollouts, we know that the completed output will fail to compile. In such cases, we preemptively abort the rollout and reuse the previously recorded rollout for the remainder of the simulation. To further prevent continuations from faulty code, during the expansion phase, we only add nodes to our tree whose node states do not contain any lines of code with fatal errors.

# A.2 Additional Reward Functions

Taking inspiration from popular machine translation metrics (Belouadi and Eger, 2023; Zhao et al., 2019, 2020; Song et al., 2021), which compute the Earth Mover's Distance (EMD; Rubner et al., 1998; Kusner et al., 2015) between word embeddings, we also explore with measuring perceptual image similarity as the EMD between SigLIP's image patch embeddings. Given the distance matrix D, where $D_{i,j} = \cos(x_i, y_j)$ and x, y are the patch embedding vectors of the input and output images of simulation j with lengths $|x|$ and $|y|$ , respectively, EMD is defined as follows:

$$
\operatorname{EMD} (x, y) = \frac {\sum_ {i = 1} ^ {| x |} \sum_ {j = 1} ^ {| y |} F _ {i , j} D _ {i , j}}{\sum_ {i = 1} ^ {| x |} \sum_ {j = 1} ^ {| y |} F _ {i , j}}, \quad \text { with } \quad \min _ {\boldsymbol {F} \geq 0} \sum_ {i = 1} ^ {| x |} \sum_ {j = 1} ^ {| y |} F _ {i, j} D _ {i, j} \quad \text { s.t. } \quad \forall_ {i, j} \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {| x |} F _ {i, j} = \frac {1}{| y |}, \\ \sum_ {j = 1} ^ {| y |} F _ {i, j} = \frac {1}{| x |}. \end{array} \right. \tag {4}
$$

We define $V_{i,j} = 2 \tanh(-\operatorname{EMD}(x, y)) + 1 \in [-1, 1]$ if compilation produces any output. If compilation fails, we set the reward to -1. We empirically tune the hyperparameter on which layer to extract the patch embeddings using the perceptual similarity dataset of scientific figures from Belouadi et al. (2024). We find that extracting embeddings after the 24th layer yields the best results. However, when evaluated on our data, this reward function achieves a segment-level correlation of only 0.425 (cf. §7), which is lower than for SELFSIM while being computationally more expensive. Consequently, we do not employ this reward function in further experiments.

# B Additional Experimental Results & Analyses

In Table 6, we compare LIVE (Ma et al., 2022), a state-of-the-art method for generating SVG, with our TikZ-based approach. In Figure 4, we additionally investigate the extent to which our models memorize the training data. We also perform training data ablation studies, as presented in Table 7.

<table><tr><td rowspan="2">Models</td><td colspan="3">Reference Figures</td><td colspan="3">Synthetic Sketches</td></tr><tr><td>DSIM↑</td><td>SSIM↑</td><td>KID↓</td><td>DSIM↑</td><td>SSIM↑</td><td>KID↓</td></tr><tr><td>LIVE</td><td>57.078</td><td>69.253</td><td>324.219</td><td>49.455</td><td>64.998</td><td>416.016</td></tr><tr><td>CLAUDE 3</td><td>64.896</td><td>83.372</td><td>17.822</td><td>59.102</td><td>73.954</td><td>29.541</td></tr><tr><td>GPT-4V</td><td>69.741</td><td>86.215</td><td>6.714</td><td>61.98</td><td>75.687</td><td>33.203</td></tr><tr><td>DETIkZIFY-TL $_{1.1B}$ </td><td>65.538</td><td>84.161</td><td>15.747</td><td>60.585</td><td>77.947</td><td>21.851</td></tr><tr><td>DETIkZIFY-DS $_{1.3B}$ </td><td>68.659</td><td>86.079</td><td>11.536</td><td>62.756</td><td>79.097</td><td>17.334</td></tr><tr><td>DETIkZIFY-CL $_{7B}$ </td><td>72.315</td><td>87.466</td><td>8.301</td><td>65.118</td><td>79.717</td><td>12.207</td></tr><tr><td>DETIkZIFY-DS $_{7B}$ </td><td>73.01</td><td>88.323</td><td>5.951</td><td>65.198</td><td>80.207</td><td>12.207</td></tr></table>

Table 6: System-level scores for LIVE, an SVG-generating model, compared with TikZ-based models from output-driven inference. Scores for TikZ-based models are copied from Table 2 for easy reference. Bold and underlined values indicate the best and second-best scores for each metric column, respectively. Cell shading reflects the relative score magnitudes across input types. Arrows indicate metric directionality.

# B.1 Comparing TikZ and SVG

Since LIVE generates SVG code instead of TikZ, we do not report cBLEU and TED scores. Additionally, because it optimizes Bézier curves rather than generating tokens, we exclude MTE, leaving only the image similarity metrics DREAMSIM, SELFSIM, and KID. Table 2 shows that LIVE underperforms all other models in our evaluation. On reference figures, it scores over 7.8pp and 14.1pp lower than the worst baseline model on DREAMSIM and SELFSIM, respectively, and its KID is more than 18 times higher. This subpar performance can be attributed to the complexity of scientific figures saved as SVGs. While we use LIVE in its default configuration, generating eight paths with four segments each, our scientific figures consist of over 110 paths on average with an arbitrary number of segments, not counting deduplicated paths, which LIVE cannot detect. Although we could theoretically configure LIVE to generate more paths, this would linearly increase inference time, quickly becoming intractable. LIVE already requires over 18

![](images/df15b48b3c757bbfdd9a942a33ed54980fb20c6ab700a2e833d6dd6fb4ce5f61.jpg)

<details>
<summary>line</summary>

|        | Human | DEiTikZIFY-DS₁.₃B | DEiTikZIFY-DS₇B | CLAUDE 3 | GPT-4V |
| ------ | ----- | ----------------- | --------------- | -------- | ------ |
| 1-gram | ~5%   | ~3%               | ~2%             | ~2%      | ~2%    |
| 3-gram | ~20%  | ~15%              | ~12%            | ~10%     | ~10%   |
| 5-gram | ~60%  | ~55%              | ~50%            | ~45%     | ~45%   |
| 7-gram | ~75%  | ~70%              | ~65%            | ~60%     | ~60%   |
| 9-gram | ~85%  | ~80%              | ~75%            | ~70%     | ~70%   |
</details>

Figure 4: Proportion of generated code n-grams with $n \in [1, 10]$ that are novel (i.e., not present in the training data). Results for human-created code are included as a reference point for comparison.

hours to complete the test set for one input type, whereas $DeTi_{K}Z_{IFY}-CS_{7_{B}}$ (OI), for example, takes less than 5 hours. Furthermore, since LIVE attempts to vectorize the input directly without semantic interpretation, it performs even worse on synthetic sketches. We conclude that SVG, and, by extension, models that generate SVG, are not well-suited for our problem domain and objectives.

# B.2 Memorization

Memorization of training data is a common concern in language models (McCoy et al., 2023; Carlini et al., 2023; Raunak and Menezes, 2022; Meehan et al., 2020). To assess the extent of this issue in our models, we calculate the n-gram novelty (McCoy et al., 2023). Specifically, we determine the proportion of n-grams, with $n \in [1, 10]$ , in the model-generated TikZ programs that are not present in the training data. We perform this analysis on the test split of DATIKZv2 for our baselines and DEEPSEEK-based DETIKZIFY models conditioned on reference figures, as well as human-generated code, as shown in Figure 4. All models initially exhibit similar novelty and are slightly less novel than humans for n < 7. However, starting from n = 7, all models except DETIKZIFY-DS $_{1.3B}$ surpass human novelty, with more than 80% of all model-generated n-grams being novel for n >= 8. This phenomenon of models becoming more novel than humans is commonly observed and is considered an indicator that language models are not significantly affected by memorization (McCoy et al.,

<table><tr><td rowspan="2">Models</td><td colspan="6">Reference Figures</td><td colspan="6">Synthetic Sketches</td></tr><tr><td> $MTE_{\uparrow}$ </td><td> $cBLEU_{\uparrow}$ </td><td> $TED_{\downarrow}$ </td><td> $DSIM_{\uparrow}$ </td><td> $SSIM_{\uparrow}$ </td><td> $KID_{\downarrow}$ </td><td> $MTE_{\uparrow}$ </td><td> $cBLEU_{\uparrow}$ </td><td> $TED_{\downarrow}$ </td><td> $DSIM_{\uparrow}$ </td><td> $SSIM_{\uparrow}$ </td><td> $KID_{\downarrow}$ </td></tr><tr><td>Full Training</td><td>83.771</td><td>1.336</td><td>57.661</td><td>68.659</td><td>86.079</td><td>11.536</td><td>87.446</td><td>0.541</td><td>60.112</td><td>62.756</td><td>79.097</td><td>17.334</td></tr><tr><td>-Synthetic Sketches</td><td>-1.957</td><td>+0.327</td><td>-0.822</td><td>+2.433</td><td>+1.318</td><td>+3.296</td><td>-13.358</td><td>+0.171</td><td>+1.369</td><td>-3.993</td><td>-3.332</td><td>+34.18</td></tr><tr><td>-METAFIG</td><td>-1.846</td><td>-0.096</td><td>-0.356</td><td>+0.398</td><td>-0.046</td><td>+0.115</td><td>-0.132</td><td>+0.053</td><td>-0.378</td><td>+0.084</td><td>-0.181</td><td>+2.773</td></tr></table>

Table 7: Ablation study results for DT-TL $_{1.1B}$ (OI), showing the relative impact on test set performance when either sketch-based training or connector pretraining is omitted, compared to full training. Improvements are highlighted in green, and declines in red, with reference scores taken from Table 2.

2023; Belouadi et al., 2024). Interestingly, for larger n-grams, $DeTiKZIFY-DS_{7B}$ demonstrates higher novelty than its smaller counterpart, suggesting that despite its larger capacity, it does not overfit and generalizes well. The most novel models are GPT-4V and $CLAUDE$ 3, possibly because they were not trained on $DATiKZ_{v2}$ and might have been trained on data that has been prepared differently.

# B.3 Training Data Ablation Studies

To better understand the impact of training with synthetic sketches and pretraining using METAFIG on test set performance, we conducted ablation studies with $D_{ETIK}Z_{IFY}-DS_{1.3B}$ in the OI configuration as a representative model, following the experimental setup detailed in §6.1. In particular, Table 7 compares full training with variations where synthetic sketches are excluded and the step of pretraining the connector is omitted. The results from excluding synthetic sketches align with expectations: although this approach slightly improves performance on reference figures on average, it substantially reduces performance on sketches. Therefore, for models expected to perform well on both figures and sketches, we recommend our original training methodology. Conversely, for models focused solely on figures, training exclusively on figures may be advantageous. The findings related to skipping connector pretraining are less definitive as the score differences are minimal, reflecting the lack of consensus in related literature about the benefits of connector pretraining for downstream performance (Liu et al., 2023b,a; Karamcheti et al., 2024). However, on average, we observe a positive impact, especially on MTE and KID, where consistent improvements are noted for both reference figures and synthetic sketches as input. Thus, we advocate incorporating a dedicated pretraining step in the training protocol. In future work, we also plan to investigate the impact of pretraining dataset size and quality.

# C Additional Training & Inference Details

In this section, we provide supplementary information on the training and inference procedures for all our models. For training and inference of our local DETIKZIFY models, we utilize a compute node equipped with four Nvidia A40 GPUs and 448 gigabytes of RAM. We access CLAUDE 3 and GPT-4V through their respective official API endpoints.

# C.1 DeTikZIFY

Complementing the information provided in §4, our 1b models require approximately two days of fine-tuning on our hardware. For the 7b models, we employ optimizer state and gradient partitioning (Rajbhandari et al., 2020) to accommodate them within the available resources, resulting in an extended training time of 21 days. Generating sketches for the training runs takes an additional 1.5 days, but since we cache our sketches, these costs are incurred only once. Output-driven inference takes 4–8 hours, depending on the model and input type, and time-budgeted inference extends the runtime by a further 1.5 days.

# C.2 INSTRUCT-Pix2Pix

As SKETCHFIG with only 549 examples may be considered too small for fine-tuning INSTRUCT-Pix2Pix, we augment our training data with 4000 additional sketches of natural images (Sangkloy et al., 2016; Li et al., 2019) and 2000 synthetic sketches of scientific figures generated with base INSTRUCT-Pix2Pix. We then oversample SKETCHFIG at a 5:1 ratio and, following Paul (2023), train for 15k steps with a

batch size of 8 and a learning rate of 5e-5. We select “turn it into a doodle” as our initial prompt, which also appears in INSTRUCT-Pix2Pix’s pretraining dataset and demonstrates the most promising zero-shot performance.

# C.3 CLAUDE 3 & GPT-4V

Building upon the experiments described in §6, we derive all our Self-Refine prompts from the official examples provided by Madaan et al. (2023) for generating TikZ programs, with only minor modifications. In particular, we employ the following prompt template in the initial step of both Self-Refine and Visual Self-Refine, substituting “sketch” or “picture” as appropriate:

1 This is a [ sketch | picture ] of a scientific figure. Generate   
2 LaTeX code that draws this scientific figure using TikZ. Ensure   
3 that the LaTeX code is self-contained and does not require any   
4 packages except TikZ-related imports. Don't forget to include   
5 \usepackage{tikz}! I understand that this is a challenging task,   
6 so do your best. Return your result in a \`\`latex code block.

We then extract the first LATEX code block from the generated text. In the rare cases where GPT-4V incorrectly classifies input images as unsafe, we add a small amount of Gaussian noise to the image pixels to bypass the issue. If compilation fails due to a fatal error (which occurs in only 1.5% of all cases) without producing an output artifact, we repeatedly use the following prompt template until all issues are resolved, replacing <code> with the generated code and <error> with the corresponding error message:

```txt
Given the error message:
<error>
And the problematic code:
```latex
<code>
```
First, identify the issue based on the error message. Then,
determine the cause of the error in the code. Finally, propose
and implement a solution. Return the fixed code in a ``latex code
block. 
```

For Visual Self-Refine, we additionally use the following prompt template to visually refine the output. Since we provide two input images (the initial figure or sketch and the current output), we label one as “Input” and the other as “Reference”. CLAUDE 3’s API has a built-in mechanism for labeling images, while for GPT-4V, we embed the labels directly into the images:

```txt
<code>
```
This is the TikZ/LaTeX code for the scientific figure shown in the picture labeled "Input". Can you improve it to better resemble the provided reference [ sketch | picture ]? First, analyze the "Input" picture to understand its components and layout. Then, consider how the scientific figure can be enhanced to more closely match the reference [ sketch | picture ]. Finally, rewrite the TikZ code to implement these improvements, making the image more similar to the reference. Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. Don't forget to include \usepackage{tikz}! Return your result in a ``latex code block. 
```

Following the findings of Madaan et al. (2023), we visually refine for a maximum of four iterations, as they observe diminishing returns beyond that point, and it helps reduce inference costs. Although this means that in most cases, we terminate before the 10-minute timeout is reached (cf. §6.1), we believe this is a sensible decision, as we observe that GPT-4V is unable to visually refine its

outputs successfully in any case. We hypothesize that this limitation is due to general-purpose chat models requiring too much explicit context for this task. These models receive the entire previously generated code as input, along with two input images and a complex textual prompt, which may be too challenging for them to process effectively. Preliminary experiments with more elaborate prompts did not seem to mitigate the subpar performance, likely due to this reason.

# D Annotator Demographics

Our annotator team consists of eleven experts with extensive research experience in science and technology. The team comprises one male faculty member, two female PhD students, seven male PhD students, and one male research assistant from another institution. We chose to work exclusively with expert annotators based on the findings of Belouadi et al. (2024), which demonstrated that crowd annotators often lack the necessary research background to produce reliable annotations.

# E Examples

To provide a better understanding of our work, we present a variety of examples in this section. Table 8 displays exemplary figures and real sketches from SKETCHFIG, while Table 9 shows figures and synthetic sketches from DATIKZv2. Additionally, Tables 10 & 11 present sample outputs generated by our systems during our human and automatic evaluations. Figure 5 provides a closer look at generated code.

When comparing the real sketches in Table 8 to their corresponding reference figures, it becomes evident that the sketches often contain less detail. For instance, sketches may lack colors or grids and feature less precise lines. Moreover, the handwritten nature of the sketches can sometimes make the text within them harder to read. These characteristics are also present in the synthetic sketches shown in Table 9. However, the problem of illegible text is more pronounced in these sketches, as generating readable text remains a common challenge for image generation models (Borji, 2023). While the text may still retain its meaning in a hidden way (Daras and Dimakis, 2022), this could lead to hallucinated text in the generated TikZ programs. Nonetheless, we believe that this aspect can still be advantageous for end users, as it enables them to quickly add scribbles to indicate the desired text placement. By doing so, DE $_{IK}$ ZIFY can generate code for the overall structure and layout, allowing users to easily modify and replace the text afterward.

The randomly selected generated figures from our human and automatic evaluations (cf. §6.2 & §6.1) shown in Tables 10 & 11 corroborate our quantitative findings. DE $TiKZIFY-DS_{7B}$ (TI) demonstrates the best overall performance and shows the least amount of fidelity errors, confirming the effectiveness of our SELFSIM-based MCTS refinement algorithm. However, we still observe some inconsistencies, such as in layout and axes labeling, although to a lesser extent compared to DE $TiKZIFY-DS_{7B}$ (OI) and GPT-4V. We attribute the prevalence of this problem partly to our focus on perceptual similarity rather than, e.g., pixel-level similarity, which allows the models greater flexibility in interpreting the general semantics of the input figures and sketches. While optimizing pixel-level similarity could be an alternative approach, we argue that perceptual similarity can serve as a more meaningful measure, especially when considering sketches. We believe that real users who provide rough sketches of unfinished ideas will find the generated outputs that interpret and refine their concepts to be inspirational. However, we acknowledge the potential benefits of exploring more rigorous similarity measures and plan to investigate this in future research. Interestingly, GPT-4V occasionally generates outputs that may not be appropriate in a scientific context, such as mistakenly embedding a smiley face in the fourth example in Table 10. Instead of resolving such issues, GPT-4V (TI) further emphasizes these details, distancing the output from the actual reference.

Figure 5 provides a side-by-side comparison of the generated TikZ programs corresponding to the first row in Table 10. $D_{ETIK}Z_{IFY}-DS_{7B}$ demonstrates its ability to utilize advanced abstractions and control flow statements, generating code that is free of compile-time errors in both OI and TI configurations. On the other hand, GPT-4V (OI) incorrectly uses an undefined arrow tip kind stealth' in lines 9 and 10, resulting in recoverable compile-time errors. GPT-4V (TI) contains the same error in line 8 and introduces additional errors in lines 16 and 26, where the \* symbol would have to be removed from the loop lists for successful expression evaluation.

<table><tr><td>Reference Figures</td><td>Real Sketches</td></tr><tr><td><img src="images/7600c02b8a17fbde02cf9e69a813a90131b434ffce4149f1dea5c6a601671583.jpg"/></td><td><img src="images/e08af025b35cc45b38808165e0da0949566e80bf9fb1771c4dbc7ce103379a3b.jpg"/></td></tr><tr><td><img src="images/f9268e8f2f2efe044688475f490c0682fe4aa8b2ee3b77988f332b32acd3dcfb.jpg"/></td><td><img src="images/68ab50d1e8e78556ee0b9579138ebe071a99f498c3ed172a50367ca97b3c63fb.jpg"/></td></tr><tr><td><img src="images/a383368e39af8e9be1e979d702af7e8e19b007eedfdc562476a4e7793e264423.jpg"/></td><td><img src="images/5f3f5e99524466285946a8d2934ad4e167bc32482ddb85428a2442de1470dc30.jpg"/></td></tr><tr><td><img src="images/baad83cd836ab69da602f67cbe1694051d7354ee4fc6b31fa219c76f3fdca90f.jpg"/></td><td><img src="images/4b84524502967e4e01bc81e9f8c68fa7a9b950011f4eee1d657aa3bc464bfffa.jpg"/></td></tr><tr><td>Reference Figures</td><td>Synthetic Sketches</td></tr><tr><td><img src="images/adffef790e24f74917369e010bb178e747dc4f179a2c192ba900fa0d231843c6.jpg"/></td><td><img src="images/8a8ebfcc8f24dc80550b808382fdbdbb1b1a3f994c260885753aca34995817ef.jpg"/></td></tr><tr><td><img src="images/9bb1c2926040e93e4fb080fc04bf284ad30a6f91fbdb78ca06f32901d94bb0b1.jpg"/></td><td><img src="images/605db7bdb55413bf8695a8b798bc3c0e7106bbb575958729206dc94b9f7a17da.jpg"/></td></tr><tr><td><img src="images/b6173e2799bd0c786c6f90b7384b77db9da624f16d6bb761c89ba85cdab7eff3.jpg"/></td><td><img src="images/f214a9ba4bdaf53ea06dd3517429812ead340a9eedae09ee175cbb3db1994cf6.jpg"/></td></tr><tr><td><img src="images/04d3163f21734f40fac3b535d46ecd676c4e0a151c445c02f7d5f7bd2658ee9f.jpg"/></td><td><img src="images/345368c4175e123bdbba69475dd763756b907a2bd9814a15b24e467f3ee2d8de.jpg"/></td></tr><tr><td><img src="images/45ee1821680feebd1def35034b0f667453dad447a1fb57223b059014731bfcfe.jpg"/></td><td><img src="images/f8525e0db8b5a8d21a2bd30e2f46fee6f2e2a171d40c8f3d9269e52bd1010e8b.jpg"/></td></tr></table>

Table 8: Representative examples of reference figures paired with real sketches from the SketchFig dataset.

Table 9: Illustrative examples of reference figures and corresponding synthetic sketches from the subset of the $\mathrm{DATikZ}_{\mathrm{v2}}$ dataset that is licensed for redistribution.

<table><tr><td>Input</td><td>GPT-4V (OI)</td><td>GPT-4V (TI)</td><td>DE $T_{IK}$ ZIFY-DS $_{7B}$ (OI)</td><td>DE $T_{IK}$ ZIFY-DS $_{7B}$ (TI)</td></tr><tr><td><img src="images/67aab606dac91e760f58b689e5ad918ea1f3e8f1dd3a7db6ada7109477c0a978.jpg"/></td><td><img src="images/26d77fc7dd5d27bc01b1b3a5b97ec95a7668773de67095d6866e3ca1141c5ab0.jpg"/></td><td><img src="images/52b964aba1b0bd4f1b82dc0cb68c0cf37ec067bcca1771ad461a59e04527a1e5.jpg"/></td><td><img src="images/57feb16e119d5e2d0db5455932fff6e8fc2e00a5c5c24028ce2c1acf7dd19af8.jpg"/></td><td><img src="images/f29ec79522539fe193c49aee8893c2cd3ddab202adb374a55ce8c2bf8184b42a.jpg"/></td></tr><tr><td><img src="images/db2830fc85d7ac2a45b55cce06d8ccacdc2028dd1efadbaaa4c39b8a189a1855.jpg"/></td><td><img src="images/baa848113467a7a71b81742b4028b45b05c7fd01a37cd74a0dbda5a9e966001a.jpg"/></td><td><img src="images/4fc5706c37df1e6d02e27dcec87549c957e4405be8cd16c27bee2d94a714a087.jpg"/></td><td><img src="images/b55e9c3bed7bcb1342746674f89ef2d429452d4a718b177162d76c43f880d6ae.jpg"/></td><td><img src="images/c8118d862a60c0142b9f8eeb2febde41406d862d719a139196d05709fd3f2598.jpg"/></td></tr><tr><td><img src="images/6642822e49d8e6bc13d5105fd3b09f41d8a8cb2a5f16c3ed78279557a8074df3.jpg"/></td><td><img src="images/1e1e77e92871ccc0d820e37b6560ed00c99290e7aafa16d55a3a6043fe35d071.jpg"/></td><td><img src="images/5ef5c81cb98996dc00aa4e02f9d59792a1df564121620beab4fa4c294743f366.jpg"/></td><td><img src="images/edc2ca152d1bf201851d3b8bbc766c7ffbda6dd8305c43df2f9a7f0553ec4f05.jpg"/></td><td><img src="images/4d6b17ee023f937ee221cd4716f6fd3c2029414a9b356e16af663d88a351309e.jpg"/></td></tr><tr><td><img src="images/dc3fbdb3b3660d742f42d936babce0e8633e1579828f33915fdf058d06d883e8.jpg"/></td><td><img src="images/3d60be4d778eb2c20d5d2515c50f6d5e91e23d3efb4e14354a3f72f53fb5de0e.jpg"/></td><td><img src="images/cd91744e30425786c8013735efbdcb8c8497c7e2d8e1f0101f67935d8753f58c.jpg"/></td><td><img src="images/39a9ae5082018ceff9834c42ecfb98c1b97af61d8a8792457457945e1e3daf04.jpg"/></td><td><img src="images/7820ce4783827b5759862731e87910970bf0b693d0b84936d3401a2774ff2d87.jpg"/></td></tr><tr><td><img src="images/f7c7b5756513de27d3b8a6a632e360373e9e84dd0cb519facd5bb9f078ffdd80.jpg"/></td><td><img src="images/4fe773e08df34322ccc2549d9233a51036284d04e0263bbd4df9a4d9016b59b2.jpg"/></td><td><img src="images/3a2804d953c99de2edb60f32c837d1c474dd2e0f821b81dd9546a04482c3028f.jpg"/></td><td><img src="images/fe7d30d221c8c2e5c2b6912a981e2696c3457aa7d2f78a417a07a53515dcbd3e.jpg"/></td><td><img src="images/ed3f44ce974da702cbaf2f5920ff60147e33e621a64d5fe1d4cfffa1649d4e29.jpg"/></td></tr><tr><td><img src="images/e0ff8c54eb1812ed8b9e5fef5027c094354f4ff3894679628c399a47ce95b93c.jpg"/></td><td><img src="images/c223a8266c7d562f65c34b7953f0922727f3475b4c7e39f485230ece975c0afc.jpg"/></td><td><img src="images/188bc1192be5005a7fd6e1b03ae6ab814746b0dfae7f5277df7d43bb1b1ddabb.jpg"/></td><td><img src="images/c9e87ed9faf12c6b73abecc80ad8cc5cbb352d28d33af66cafdee94521b622f5.jpg"/></td><td><img src="images/3c0eb932fb69b41aaa5e5a0d817591e8dbfd186acf2d8b8b92cc2384ffa291c7.jpg"/></td></tr><tr><td><img src="images/6ec94c91b23a2efca795a095f6a4604038989d18a2366e4408f9dacb86bbf116.jpg"/></td><td><img src="images/d636e40ba1fee55a8e5627c152c63287e83a865f76c9606f3d72dcf0490b3eef.jpg"/></td><td><img src="images/09dceb03f0d83e433599279ae85f261b284ef43256c2d91c3a297e4b0cf21a8c.jpg"/></td><td><img src="images/b858a89970a5691c6f38c0d22921c9a2dc48f0c50afb4bc52af8b367ce327e3e.jpg"/></td><td><img src="images/defef2b04726fa1edf1db5b12131a690c0261a6a97a994788c21cdcd54a509ba.jpg"/></td></tr><tr><td><img src="images/25c8c95460005310a104dbd68bf88fd9b07289f79ebd366629cca79e8f186f99.jpg"/></td><td><img src="images/722602ce94141714a95d1e29162f5c2a4a818f84a2a82815ea592c1ff8af6080.jpg"/></td><td><img src="images/18d4ad4643dfc000096a6771bef14f3fd1f25f860c3da206c4df59ef8b3459df.jpg"/></td><td><img src="images/0b09ac1dfc870bc4d9ae32c32603eb57b6deb8defc95c4270bade7d7e1982354.jpg"/></td><td><img src="images/757c37588d768f9ba470a049a582cdfe13a5fce775db1e47275f07ab21663bba.jpg"/></td></tr><tr><td>Input</td><td>CLAUDE 3 (OI)</td><td>GPT-4V (OI)</td><td>DETIKZIFY-DS7B (OI)</td><td>DETIKZIFY-DS7B (TI)</td></tr><tr><td><img src="images/e65814f307aaf5dd4e408015bfdb6630cf940774f3448c859a4019d9c4c6bd90.jpg"/></td><td><img src="images/1bef427103ac19d4ed3391273d992b5cbe291c40979baad31c9964ca9e4e28c0.jpg"/></td><td><img src="images/196d5ff8764063696a93f90a9ebb100015957a0c6e5d5524fa3d21390a80343d.jpg"/></td><td><img src="images/82df27ce2cde5c2185f0a5b16d88f8befff24a8ab2e13dee6118aa959c4313fd.jpg"/></td><td><img src="images/b26c5a8ce13fe959f6a52f00df1eac9821bde8db880214b2bb4b094c6c89c43b.jpg"/></td></tr><tr><td><img src="images/08302ccb055006ea14e9ebb1ebeddf6283cdd0fc59780ca2158f188d37e74334.jpg"/></td><td><img src="images/7634a69af093508c2bcf02614cf98583337fb361f26e7138a68066850431b0c4.jpg"/></td><td><img src="images/422ef1e0aff79704c0ac92a4e9137095adc569503f93c814bd151a750ecd6b27.jpg"/></td><td><img src="images/565fd28786b708f8a1c09a2b4fefdfb70faf6b8158eca431de97001e945e216d.jpg"/></td><td><img src="images/57b39478c514f51ee629d52b6e0bdd5564349a8124126c4aa0b3f1ee1d335dc7.jpg"/></td></tr><tr><td><img src="images/3c1c7c3590294420258b9c20b4ab6a13fad9fe87aadf216fa2cebb1ac2b2fa7d.jpg"/></td><td><img src="images/f02a0c3ab7cb1cecc5a037d9df6500b5ce986137647f4ac858220ffdae4fb52c.jpg"/></td><td><img src="images/a899eafacc9b4704cb877cea9ce6291023f251e23c92ab8f5fb32476de968fe0.jpg"/></td><td><img src="images/24d2ea555cfb907cafc94188ed6e14696dd8e5a8b7050e0c34c38f887f9d5735.jpg"/></td><td><img src="images/ec40e7d145fa519d2198a71c524cfdb67ce7fecc49e794f3e8e57a00ef0da93e.jpg"/></td></tr><tr><td><img src="images/0b2388342570ce51c43661be99eb667266139f205f5f4e33c3258e397b90237b.jpg"/></td><td><img src="images/3502ca3d62df584065a0daee4a8eda06f8dfc708a8569120587acf0af131c262.jpg"/></td><td><img src="images/a539d9d4c3f4be7e3d1fc53a9ac5b4170fd835b3878fed2fc5c3635b1c160888.jpg"/></td><td><img src="images/2b4aee7f4960bec96dc36a0f3d3c58e1c8e69bcdd43f563fc889c7ae89c3f858.jpg"/></td><td><img src="images/a06de8d54e9f106ff5df403127ed9986a1c2c3c9682829bfe92fd3066b3bc662.jpg"/></td></tr><tr><td><img src="images/350bf03a98ba7ce56cf2b43b9f68c01d38a599f1e861ca9c052c990a397af43c.jpg"/></td><td><img src="images/2c1d546f5479a53ed00d984518b36c9713eba2048329d03df126a6439f2f7b6f.jpg"/></td><td><img src="images/5fb20ad710ec7bbe13cd7402d981502e3358123c3ef708c1a46d2e0b4b4faf55.jpg"/></td><td><img src="images/25743c10df736da67b124383d5eed6159325c2ae00badbbabf6e6f5392ee080e.jpg"/></td><td><img src="images/b710bbbd87aeb6bbfe6c531833845185925f57ee068ec70a17fbebc8d5d60169.jpg"/></td></tr><tr><td><img src="images/4bdb4f86f752dc3ca45e5b785fd0489fea101ac8ab3078d0bec5430ee2078bef.jpg"/></td><td><img src="images/11400ccdc08ae94522a1e92937a0362f5939994d615c77f7a9b9667a8d1cb420.jpg"/></td><td><img src="images/bd16a1a8f919ba850ed9e5de7f7f2b97bd7eea50808093a6f4a1e5488a9b1ca3.jpg"/></td><td><img src="images/ebac1d03217221fef1facc7d456b8e0f8425511707c745a3e09b7318fe6b79f6.jpg"/></td><td><img src="images/012ab057c2c3e251f06c877becc3476c85526e0140b2f7adb57cb943a6522388.jpg"/></td></tr><tr><td><img src="images/c88e1508657689612b4264ce50d238fe87269d6cc2518aba4141aa9694fb3430.jpg"/></td><td><img src="images/72b3122172d8ff1838b299a2d11e22a26eb11f48dfde995a633e25a41b50d8a2.jpg"/></td><td><img src="images/427f1a050be871d7316513adc69f84ac9bd49c752a596c2246c9e1b0d9a395c6.jpg"/></td><td><img src="images/9c2366573182c3cebc3ded8d87897a2c55e8a1870af0e1bf0b0fd11cb042d93d.jpg"/></td><td><img src="images/9089a0b9dfd196b7074b0f6f146c03820bfbf762072bf58e3a016dbdf07e2f25.jpg"/></td></tr><tr><td><img src="images/0ddd3a7113539de567e03ad554c1f6981554c9ec2aaa76baf86f8784b1580c0f.jpg"/></td><td><img src="images/67819a3e28e2ab4d97ea396f20a9ea857bd50cf58747eb22be8c7a225579089f.jpg"/></td><td><img src="images/5e777c97ee006021fbe8e83223c46fc06df6f0c4cb4fc56385765b4f513632a1.jpg"/></td><td><img src="images/37c256b1c312fcbde0a8be3ff04692e127da2e0dc37e39c444a991e58d3b40bc.jpg"/></td><td><img src="images/c4466da42b9129989872c4b3496504bf60b1c13857a426ec0d1e7d20b698cdb2.jpg"/></td></tr></table>

Table 10: Alternating rows of randomly selected reference figures and real sketches (first column) alongside corresponding scientific figures generated by GPT-4V and DETiKZIFY-DS $_{7B}$ in output-driven (OI) and time-budgeted (TI) configurations (columns 2–4), taken from our human evaluation campaign (cf. §6.2).

Table 11: Alternating rows of randomly selected reference figures and synthetic sketches (first column) alongside corresponding scientific figures generated by CLAUDE 3 (OI), GPT-4V (OI), and DETIKZIFY-DS $_{7B}$ (OI & TI) in columns 2–4, taken from our automatic evaluation (cf. §6.1).

![](images/ea0c95c56522bcac39d39c7c747bfa6b1f01b8d2fbfb124f7265d40197ffa4f9.jpg)

<details>
<summary>text_image</summary>

\documentclass[border=3pt,tikz]{standalone}
\usepackage{tikz-3dplot}
\begin{document}
\tdplotsetmaincoords{70}{110}
\tdplotsetrotatedcoords{0}{20}{70} %<- rotated coords
\begin{tikzpicture}[tdplot_main_coords,line join=round]
\pgfkeys{/pgf/number format/.cd,precision=2,fixed}
\def\n{12}
\def\h{2}
\begin{scope}[tdplot_rotated_coords,canvas is xz plane at y=0]
\draw[-latex] (\h,0) -- (\h,1);
\draw[-latex] (0,0) -- (1,0);
\foreach \i in {0,...,\n}
{
\draw[thick] (\i*\h/\n,0) circle[radius=1];
}
\end{scope}
\foreach \i in {1,...,\n}
{
\draw[-latex,thick,brown] (\h*\i/\n,\h,i/\n) -- (\h*\i/\n,\h,i/\n+1/\n);
}
\begin{scope}[tdplot_rotated_coords,canvas is xz plane at y=\h]
\draw[-latex] (0,0) -- (1,0);
\foreach \i in {0,...,\n}
{
\draw[thick] (\i*\h/\n,0) circle[radius=1];
}
\end{scope}
\draw[-latex] (0,0,0) -- (1,0,0) node[pos=1.1]{$x$};
\draw[-latex] (0,0,0) -- (0,1,0) node[pos=1.1]{$y$};
\draw[-latex] (0,0,0) -- (0,0,1) node[pos=1.1]{$z$};
\end{tikzpicture}
\end{document}
DETiKZIFY-DS7B (OI)
DETiKZIFY-DS7B (TI)
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usepackage{amsmath}
\usetikzlibrary{arrows.meta,decorations.markings}
\begin{document}
\begin{tikzpicture}[scale=1.5]
% Styles for arrows
\tikzset{axis/.style={thick,>,>=stealth'}}
\tikzset{vector/.style={thick,>,>=stealth',orange}}
% Draw helix
\foreach \t in {0,10,...,360} {
\draw[domain=\t:\t+10,smooth,variable=\x,black]
plot ({\x/360*4},{sin(\x)},{cos(\x)};
}
% Draw axis
\draw[axis] (0,0,0) -- (5,0,0) node[anchor=north east]{$x$};
\draw[axis] (0,0,0) -- (0,2,0) node[anchor=north west]{$z$};
\draw[axis] (0,0,0) -- (0,0,-2) node[anchor=south]{$y$};
% Draw vectors
\foreach \t in {0,40,...,320} {
\draw[vector] ({\t/360*4},{sin(\t)},{cos(\t)}) --
++({cos(\t)*0.3},{sin(\t)*0.3},0);
}
\end{tikzpicture}
\end{document}
GPT-4V (OI)
GPT-4V (TI)
</details>

Figure 5: TikZ programs generated by DeTiKZIFY-DS $_{7B}$ (top) and GPT-4V (bottom) corresponding to the figures in the first row of Table 10. Lines with compile-time errors are highlighted in yellow.