# Designing Cyclic Peptides via Harmonic SDE with Atom-Bond Modeling

Xiangxin Zhou $^{*123}$ Mingyu Li $^{*45}$ Yi Xiao $^{4}$ Jiahan Li $^{4}$ Dongyu Xue $^{1}$ Zaixiang Zheng $^{1}$ Jianzhu Ma $^{46}$ Quanquan Gu $^{1}$

# Abstract

Cyclic peptides offer inherent advantages in pharmaceuticals. For example, cyclic peptides are more resistant to enzymatic hydrolysis compared to linear peptides and usually exhibit excellent stability and affinity. Although deep generative models have achieved great success in linear peptide design, several challenges prevent the development of computational methods for designing diverse types of cyclic peptides. These challenges include the scarcity of 3D structural data on target proteins and associated cyclic peptide ligands, the geometric constraints that cyclization imposes, and the involvement of non-canonical amino acids in cyclization. To address the above challenges, we introduce CPSDE, which consists of two key components: ATOMSDE, a generative structure prediction model based on harmonic SDE, and RESROUTER, a residue type predictor. Utilizing a routed sampling algorithm that alternates between these two models to iteratively update sequences and structures, CPSDE facilitates the generation of cyclic peptides. By employing explicit all-atom and bond modeling, CPSDE overcomes existing data limitations and is proficient in designing a wide variety of cyclic peptides. Our experimental results demonstrate that the cyclic peptides designed by our method exhibit reliable stability and affinity.

$^{*}$ Equal contribution $^{1}$ ByteDance Seed (Work was done during Xiangxin's internship at ByteDance Seed.) $^{2}$ School of Artificial Intelligence, University of Chinese Academy of Sciences $^{3}$ New Laboratory of Pattern Recognition (NLPR), State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), Institute of Automation, Chinese Academy of Sciences (CASIA) $^{4}$ Institute for AI Industry Research, Tsinghua University $^{5}$ School of Medicine, Shanghai Jiao Tong University $^{6}$ Department of Electronic Engineering, Tsinghua University. Correspondence to: Quanquan Gu <quanquan.gu@bytedance.com>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](images/f36314ee7452c4f579cff2273f161b4fb6533e0526acfa13f12a87b7f3220fc1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Protease"] -->|Degradation| B["Linear peptides"]
    B --> C["Flexible"]
    C --> D["Target"]
    E["Protease"] --> F["Rigid"]
    F --> G["Enhanced function"]
    G --> H["Target"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F fill:#cfc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#cfc,stroke:#333
```
</details>

Figure 1. Comparative advantages of cyclic peptides over linear peptides. Linear peptides are easily degraded, whereas cyclic peptides are protected against enzyme hydrolysis, allowing them to function more effectively within the human body. Cyclic peptides generally exhibit better stability and affinity.

# 1 Introduction

Therapeutic peptides are a distinct group of pharmaceutical compounds comprising a sequence of precisely arranged amino acids (Driggers et al., 2008; Tsomaia, 2015; Zorzi et al., 2017). Peptide drugs tend to exhibit lower toxicity, enhanced biological activity, and high target specificity compared to small molecules, superior cellular permeability, lower cost, lower immunogenicity compared to antibody drugs (Driggers et al., 2008). However, conventional linear peptides suffer from a short half-life, limited stability, and susceptibility to hydrolase degradation (Tsomaia, 2015), restricting their therapeutic potential and broader use. Unlike linear peptides, as shown in Figure 1, cyclic peptides are chains of residues that typically form one or two closed loops, often incorporating non-natural residues. For example, a single loop can be created by connecting the N- and C-termini with a peptide bond or linking two internal cysteines through a disulfide bond (Camarero & Muir, 1999; Giordanetto & Kihlberg, 2014; Kale et al., 2018). Such cyclization enhances their resistance to digestive enzymes and enables them to bind protein surfaces with high affinity in more stable conformations. Traditional methods for discovering cyclic peptides involve chemical synthesis and high-throughput screening, both of which are labor-intensive and costly. This has led to adopting in silico approaches such as

virtual screening (Zotchev et al., 2006) and de novo design (Kawamura et al., 2017; Peacock & Suga, 2021; Bhardwaj et al., 2022; Garcia Jimenez et al., 2023) to streamline the discovery process.

Cyclic peptides can adopt various structural forms depending on how their amino acid residues are linked together based on different chemical bonds and geometric distances, for example, the distance between two cysteines forming a disulfide bond typically falls from 2.0 to 2.5 Å (Fass, 2012). These structures are classified into four categories (see Figure 2) based on the atoms of residues that form the cyclic structure: (1) Head-to-tail cyclization. The N-terminus (head) of one amino acid forms a peptide bond with the C-terminus (tail) of another amino acid, resulting in a closed ring. (2) Side-to-tail cyclization. The side chain of an amino acid is linked to the C-terminus (tail) of the peptide. (3) Head-to-side cyclization. The N-terminus (head) of one amino acid is linked to the side chain of another amino acid. (4) Side-to-side cyclization. The side chain of an amino acid is linked to the side chain of another, forming a cyclic structure that does not involve the head or tail of the peptide backbone. Therefore, incorporating specific chemical and geometric constraints (such as bond lengths, angles, and atom compositions) relating to one or more sets of residues is essential for designing different kinds of cyclic peptides. Recent attempts have tried designing disulfide-linked cyclic peptides based on a post-processing strategy (Wang et al., 2024a) or head-to-tail cyclic peptides (Rettie et al., 2024) using modified position encoding in protein generative models. However, these methods only consider one specific type of cyclic peptides and do not support other types based on specific constraints. Furthermore, the availability of real-world 3D structural data for protein-ligand complexes involving cyclic peptides is limited, posing a challenge to advancing computational cyclic peptide design.

To tackle these challenges, we developed the CPSDE, which comprises two models: a harmonic-SDE-based generative structure prediction model named ATOMSDE and a residue type predictor named RESROUTER. Unlike leading works (Watson et al., 2023; Yim et al., 2023) that typically use the residue frame representation for protein design, we employ the all-atom and bond representation. Since both linear and cyclic peptides are composed of atoms and bonds, this representation allows us to model interactions at the most fundamental level. It maximizes the use of small molecule and linear peptide data while minimizing reliance on cyclic peptide data. The inclusion of bond modeling effectively addresses the geometric constraints introduced by cyclization. With our designed routed sampling method, we iteratively update both the sequence and structure by alternating between the two models, which enables the generation of all types of cyclic peptides. We highlight our main contributions as follows:

- We introduce CPSDE, the first generative algorithm, to our knowledge, capable of directly generating all types of cyclic peptides informed by the 3D structure of a protein target, paving the way for developments in peptide-based drug discovery.   
- Our approach designs cyclic peptides with robust stability and affinity while maintaining high diversity, underscoring its significant potential in drug development and therapeutic innovation.   
- Through case studies involving molecular dynamics simulations, we demonstrate our method's practical utility and effectiveness in drug design, reinforcing its applicability and impact in real-world scenarios.

# 2 Related Work

All-Atom Protein Design. Protein design traditionally involves first designing the backbone, followed by sequence design. Diffusion models (Ho et al., 2020; Song et al., 2021) have been applied to protein backbone design (Watson et al., 2023). This process is typically followed by inverse-folding models (Dauparas et al., 2022), enabling comprehensive protein design. Recently, there has been a shift towards co-design, where both protein sequences and structures are generated jointly (Jin et al., 2022; Luo et al., 2022; Kong et al., 2023; Lisanza et al., 2024; Campbell et al., 2024).

However, all-atom $^{1}$ structure modeling is essential for comprehending protein functionality, such as protein-protein interactions. Consequently, recent research has increasingly focused on all-atom protein design to gain a more detailed and accurate understanding. To achieve full-atom antibody design, Kong et al. (2024) proposed to use a multichannel equivariant layer to encode all-atom structures, and Martinkus et al. (2023) introduced a backbone and internal generic side chain representation. Chen et al. (2025) adopted a representation that includes amino acid type, backbone structure, and sidechain torsion angles for designing protein complexes. A notable advance in de novo all-atom protein design is Protpardelle (Chu et al., 2024), which suggested modeling a “superposition” over possible side-chain states and introduced the atom73 representation. This inspires us to apply all-atom structures in a similar manner for de novo cyclic peptide design. Unlike Protpardelle, we also incorporate bond modeling, which is crucial for ensuring successful cyclization.

Peptide Design. Peptides, consisting of short chains of amino acid residues, are essential in numerous biological processes due to their interactions with various target molecules, offering substantial potential in drug discovery, such as targeting undruggable proteins (Hosseinzadeh et al.,

2021). Traditional computational peptide design methods often rely on searching and sampling residues or motifs from chemical databases (Bhardwaj et al., 2016), which can be time-consuming and limit the diversity of designed structures (Cao et al., 2022). In contrast, deep generative models, known for their strong capability in modeling data distributions, have been applied to peptide design and have shown great potential. Several studies have explored designing peptide backbones (Boom et al., 2024), designing peptide sequences (Chen et al., 2024), or generating specific peptide structures such as $\alpha$ -helices (Xie et al., 2023; 2024). Lin et al. (2025) proposed target-aware peptide sequence-structure co-design with flow matching on peptide global translation, orientation, backbone torsions, and sequences. Recent advances in full-atom protein design have significantly improved peptide generation capabilities. For example, Kong et al. (2024) employed a latent diffusion model on a latent space that encodes full-atom peptide structures. Li et al. (2025) proposed to represent peptides with backbone atoms and side-chain torsion angles, employing flow-based models to generate full-atom peptides given the protein targets. These methods face challenges in adapting to cyclic peptide design because they model protein structures at the residue level, which complicates the incorporation of covalent bonds or non-canonical amino acids essential for cyclization.

# 3 Method

In this section, we present CPSDE, a groundbreaking approach for designing cyclic peptides. It features an SDE-based generative structure prediction model, ATOMSDE, and a residue type predictor, RESROUTER, both utilizing all-atom and bond modeling. We begin by defining the cyclic peptide design task and providing an overview of SDE-based generative models in Section 3.1. In Section 3.2, we detail ATOMSDE, based on a harmonic SDE, and in Section 3.3, we describe RESROUTER, which predicts residue types based on denoised structures. Lastly, in Section 3.4, we explain how to alternate between these models through routed sampling to generate cyclic peptides.

# 3.1 Preliminaries

A peptide is a specific type of protein, generally composed of fewer than 30 amino acid residues. The type of the i-th residue $a_{i} \in \{1, 2, \ldots, 20\}$ is determined by its side-chain R group. Thus, the all-atom 3D structure of a peptide specifies both its sequence and structure, inspiring us to focus on generating 3D coordinates of all atoms for peptide design. Cyclic peptides, particularly their cyclization regions, often include unique inter-residue chemical bonds and occasionally non-canonical amino acids. Despite their non-canonical nature, these cyclization regions usually display distinct patterns. Additionally, providing detailed cyclization information during cyclic peptide design is essential, as it ensures wet-lab synthesizability. Here we denote the all-atom 3D structure of the cyclic peptide as P and the 3D structures of the receptor (i.e., protein target) as T. We define the chemical graph of the cyclization parts as C, which contains atoms as nodes and chemical bonds as edges. Note that since the 3D structures of the cyclization part are unavailable, there is no information about atom positions in C. Our final goal is to model the conditional distribution $P(\mathcal{P}|\mathcal{T},\mathcal{C})$ , i.e., generate the cyclic peptides given the 3D receptor structure and the cyclization chemical graph.

We provide basic knowledge on stochastic differential equations (SDE) and SDE-based generative models (Song & Ermon, 2019; Ho et al., 2020; Song et al., 2021). SDE-based generative models (also known as score-based generative models and diffusion models) learn the data distribution by learning to denoise. The forward SDE injects noise gradually into the data $\mathbf{x}_0 \in \mathbb{R}^d$ and constructs a diffusion process $\{\mathbf{x}_t\}_{t \in [0,1]}$ , such that $\mathbf{x}_0 \sim p_0$ and $\mathbf{x}_1 \sim p_1$ , where $p_0$ is the data distribution and $p_1$ is the prior distribution:

$$
\mathrm{d} \mathbf {x} = \mathbf {f} (\mathbf {x}, t) \mathrm{d} t + g (t) \mathrm{d} \mathbf {w}, \tag {1}
$$

where $\mathbf{f}(\cdot,t):\mathbb{R}^{d}\to\mathbb{R}^{d}$ is a vector-valued function known as drift coefficient, $g(\cdot):\mathbb{R}\to\mathbb{R}$ is a scalar function known as diffusion coefficient, and w is the standard Wiener process (also known as Brownian motion). The induced perturbation kernel $p_{0t}(\mathbf{x}_{t}|\mathbf{x}_{0})$ is a Gaussian distribution that can be efficiently sampled. The resultant $p_{1}(\mathbf{x}_{t}|\mathbf{x}_{0})$ typically approximates a Gaussian distribution $p_{1}(\mathbf{x}_{t})$ (i.e., prior distribution), which is independent of $x_{0}$ .

The reverse SDE starts from a sample from the prior distribution, denoises the noisy sample iteratively, and finally produces a generated sample:

$$
\mathrm{d} \mathbf {x} = [ \mathbf {f} (\mathbf {x}, t) - g (t) ^ {2} \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x}) ] \mathrm{d} t + g (t) \mathrm{d} \bar {\mathbf {w}}, \quad (2)
$$

where $\bar{w}$ is a standard Wiener process when time flows backwards from 1 to 0 and and dt is an infinitesimal negative timestep. Typically, a neural network $\mathbf{s}_{\theta}(\mathbf{x}_{t}, t)$ is used to approximate the underlying score function $\nabla_{x} \log p_{t}(x)$ , and it can be trained via the following score matching objective:

$$
\mathcal {L} = \mathbb {E} _ {t} [ \lambda (t) \mathbb {E} _ {\mathbf {x} _ {0}} \mathbb {E} _ {\mathbf {x} _ {t} | \mathbf {x} _ {0}} \| \mathbf {s} _ {\boldsymbol {\theta}} (\mathbf {x} _ {t}, t) - \nabla_ {\mathbf {x} _ {t}} \log p _ {0 t} (\mathbf {x} _ {t} | \mathbf {x} _ {0}) \| ^ {2} ],
$$

where $\lambda(t)$ is time-dependent weighting function and t is uniformly sampled over [0, 1].

# 3.2 ATOMSDE

To accurately model both atomic interactions and bond constraints, we initially train a docking model named ATOM-SDE using protein-ligand complex data, incorporating both small molecules and peptides as ligands.

In this subsection, for brevity, given a protein-ligand complex, we assume there are $N_{L}$ atoms whose coordinates are $x^{L} \in R^{N_{L} \times 3}$ in the ligand and $N_{P}$ atoms whose coordinates are $x^{P} \in R^{N_{P} \times 3}$ in the protein T. We denote the

![](images/86521ac5716867c2bc2627e8036a657f1ac98edd6d353175d991ece47c0ade10.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["73 atoms"] --> B["Atom73"]
    B --> C["Atom73 → Ala → Ala"]
    C --> D["Ala → Thr"]
    D --> E["Thr"]
    E --> F["Cycling"]
    F --> G["AtomSDE"]
    
    subgraph Resrouter
        H["Atom73"] --> I["Atom73 → Gly → Thr"]
        I --> J["Atom73"]
        J --> K["ATOMSDE"]
    end
    
    subgraph Atom73
        L["Atom73"] --> M["Atom73"]
    end
    
    N["Chemical graph"] --> O["Side-to-tail"]
    O --> P["Side-to-head"]
    P --> Q["Head-to-tail"]
    Q --> R["Side-to-side"]
    
    style Resrouter fill:#f9f9f9,stroke:#333
    style Atom73 fill:#f9f9f9,stroke:#333
```
</details>

Figure 2. Overview of CPSDE. The generative process is structured as follows: (1) A cyclization type is initially selected, which subsequently determines the associated 2D chemical graph; (2) At time t, given the entire chemical graph defined by both cyclization (highlighted with a yellow shadow) and the predicted residue types, the all-atom structure is initially denoised using ATOMSDE and then re-noised in accordance with the integration step of the reverse-time SDE. The updated structures are then preserved in the Atom73 representation; (3) RESROUTER predicts the residue types not constrained by cyclization, based on the denoised structure. Consequently, the chemical graph and the all-atom structures are updated using the Atom73 representation Steps (2) and (3) are iteratively executed. The incorporation of the cyclization chemical graph, with chemical bonds as edges, ensures that the generated peptide forms a cyclic structure.

chemical graph of the ligand as $G_{C}$ , where nodes are atoms and edges are chemical bonds. We build a generative docking model ATOMSDE to learn the conditional distribution $p(\mathbf{x}^{L}|\mathcal{T},\mathcal{G}_{C})$ .

We choose Variance Preserving (VP) SDE (Ho et al., 2020) instead of Variance Exploding (VE) SDE (Song & Ermon, 2019) for our scenario. The reason is that, when t is large, VE SDE causes the noisy ligands to drift far from the receptor and introduces different spatial scales between the ligands and receptor, leading to a loss of valid interaction with the receptor. Inspired by Jing et al. (2023); Stark et al. (2023), we introduce a harmonic SDE to fully leverage the

connection information embedded in the chemical graph. We define $H := L + \sigma_{P}^{-2}I$ , where L = D - A is the Laplacian matrix of chemical graph $G_{C}$ , D is the degree matrix, A is the adjacent matrix, and $\sigma_{P}$ is a receptor-dependent scalar value. The positive definite matrix can be decomposed as $H = P\Lambda P^{\intercal}$ where P is an orthogonal matrix (i.e., $PP^{\intercal} = I$ ) and $\Lambda = \text{diag}(\lambda_{1}, \ldots, \lambda_{N_{L}})$ is a diagonal matrix that contains the eigenvalues. We define the $G_{C}$ -dependent forward SDE as follows:

$$
\mathrm{d} \mathbf {x} ^ {L} = - \frac {1}{2} \beta (t) \tilde {\mathbf {x}} ^ {L} \mathrm{d} t + \sqrt {\beta (t)} \boldsymbol {\Lambda} ^ {\frac {1}{2}} \mathbf {P} ^ {\intercal} \mathrm{d} \mathbf {w}, \tag {3}
$$

where $\beta(t)$ is a positive time-dependent scalar function that

controls the noise level along the diffusion process. The induced perturbation kernel has an analytic form as (see the derivation in Appendix F):

$$
p _ {0 t} (\mathbf {x} ^ {L} | \mathbf {x} _ {0} ^ {L}) = \mathcal {N} (\mathbf {x} _ {t} ^ {L}; \mathbf {x} _ {0} ^ {L} e ^ {- \frac {1}{2} \int_ {0} ^ {t} \beta (s) \mathrm{d} s}, \mathbf {H} - \mathbf {H} e ^ {- \int_ {0} ^ {t} \beta (s) \mathrm{d} s}).
$$

Given a schedule that satisfies $\lim_{t\to1}\int_{0}^{t}\beta(s)\mathrm{d}s=\infty$ , the above perturbation process arrives at the prior distribution $p_{1}(\mathbf{x}_{1}^{L})\propto\exp(-\frac{1}{2}\mathbf{x}_{1}^{L}\mathbf{T}\mathbf{H}\mathbf{x}_{1}^{L})$ at time t=1, which can be efficiently sampled. Intuitively, the anisotropic perturbation process leverages the connection information in the chemical graph $G_{C}$ . The bonded atoms are initially set close and then gradually perturbed by correlated noises.

The model is based on an SE(3)-equivariant neural network (Satorras et al., 2021; Guan et al., 2021), incorporating both a k-nearest-neighbor graph built upon the protein-ligand complex and a ligand chemical graph. This design ensures the model is aware of both protein-ligand interactions and atom connections induced by chemical bonds. We leave the details of model architecture design in Appendix G.1.

We denote the final output of the SE(3)-equivariant neural network as $D_{\theta}(\mathbf{x}_{t}^{L}, t)$ . For simplicity, we use a simple reconstruction loss that is approximately equivariant to the score matching objective as follows:

$$
\mathcal {L} = \mathbb {E} _ {t, p _ {0} (\mathbf {x} _ {0} ^ {L}), p _ {0 t} (\mathbf {x} _ {t} ^ {L} | \mathbf {x} _ {0} ^ {L})} [ \| D _ {\boldsymbol {\theta}} (\mathbf {x} _ {t} ^ {L}, t) - \mathbf {x} _ {0} ^ {L} \| ^ {2} ]. (4)
$$

The estimated score function can then be formulated as

$$
\nabla_ {\mathbf {x} ^ {L}} \log p _ {t} (\mathbf {x} ^ {L}) \approx - \frac {1}{\sqrt {1 - \int_ {0} ^ {s} \beta (s) \mathrm{d} s}} (\mathbf {x} ^ {L} - D _ {\theta} (\mathbf {x} ^ {L}, t)),
$$

with which we can generate the ligand poses given the ligand's chemical graph and the receptor's 3D structures by solving the reverse-time SDE as in Equation (2).

# 3.3 RESROUTER

We introduce RESROUTER that predicts the ground-truth residue type given the noisy ligands. The model architecture is similar to that of ATOMSDE. Differently, the input and output of the model are modified due to the following considerations.

With a structure prediction model in hand, we could still not be able to design cyclic peptides, since their sequence is unknown. This is a classic “chicken-and-egg” problem. This motivates us to alternately denoise the 3D structures and update the residue types. Since the residue type is determined by the side-chain R group, it will provide a shortcut for the model to predict the ground-truth residue type if the complete chemical graph of the noisy ligands is input. Thus, we remove the side chain of the canonical amino acid residues except for those that are involved in cyclization.

The model produces a hidden state (i.e., h) for each atom. For the i-th residue (whose ground-truth amino acid type is $a_{i}$ ) in a peptide with N residues, we aggregate the hidden states of the backbone atoms (i.e, $N-C_{\alpha}-C-O$ ) and use a multilayer perceptron (MLP) to predict the amino acid type. The model is trained via the following objective:

$$
\mathcal {L} = \sum_ {i} ^ {N} - \log p _ {\phi} (a _ {i} | D _ {\theta} (\mathbf {x} _ {t} ^ {L}, t), \mathcal {G} _ {C}, \mathcal {T}, t), \tag {5}
$$

where $p_{\phi}$ is the model-induced probability for the residue type. ATOMSDE (i.e, $D_{\theta}$ ) is first pretrained and then fixed during the training of RESROUTER.

# 3.4 Routed Sampling for Cyclic Peptide Design

With trained ATOMSDE and RESROUTER, we can iteratively update both the sequence and structure through alternate calls to the two models, enabling the generation of all types of cyclic peptides, as shown in Figure 2.

Given the number of residues within the peptide and the cyclization information, the atoms in a cyclic peptide can be categorized into two classes: The first class is known in terms of their chemical graph (though their 3D structures are not determined), comprising all backbone atoms and the atoms constrained by cyclization. The second class is unknown, consisting of the side chains of canonical amino acid residues not constrained by cyclization. Without loss of generality, consider a specific type of side-to-tail cyclization where the sulfur atom (S) in the side chain of the i-th residue and the carboxylic acid (COOH) group of the C-terminus engage in chemical reactions to form a C-S thioester bond. In this scenario, the chemical graphs induced by all backbone atoms (since all canonical amino acid residues share the pattern N-C $_{\alpha}$ -C-O) and all atoms of the cysteine (CYS) providing the sulfur for the C-S bond that forms the cyclic structure are known. Conversely, the chemical graph between other atoms, specifically the side chains of residues except for the aforementioned CYS residue, remains unknown. In the remainder of this paper, we refer to the atoms associated with the known chemical graph due to cyclization as cyclization-constrained atoms, and the others as free-residue atoms.

Inspired by Chu et al. (2024), for the free-residue part, we maintain an atom73 state (i.e., coordinates), a “superposition” for each residue, where all possible amino acid types for this residue share the backbone atoms and $C_{\beta}$ , while possessing unique side chain atoms. For the cyclization-constrained part, we maintain a cyclization-constrained state separately. The general idea of routed sampling is to switch to different collapsed (i.e., specific) atom states for the free-residue part and assemble a new valid chemical graph based on the predicted residue type and cyclization information at each step of solving the reverse-time SDE. Specifically, at each step, ATOMSDE denoises the atom coordinates,

In practice, for both cyclization-constrained atoms and free-residue atoms, we maintain both denoised states and current states. This approach is needed because, during the sampling process, the cyclization-constrained atoms and backbone atoms are constantly present, while the side-chain

atoms of the free residues are sometimes sampled, resulting in them potentially not being adequately updated. Thus, we reuse the previous denoised structure to solve the SDE and align the time (or noise level) of all sampled atoms to the same point. This alignment is necessary because the ATOMSDE expects all input atoms to be at the same noise level, especially when a residue type is sampled after not being sampled for a few steps. Please refer to Appendix E for more details.

Notably, the two models, ATOMSDE and RESROUTER, do not account for the atom partition introduced by amino acid residues, as they model atoms and bonds at the most fundamental level. The concept of amino acid residues is introduced only during routed sampling to ensure that the free-residue part remains a canonical residue rather than an arbitrary molecule.

# 4 Experiments

# 4.1 Experimental Setup

Dataset. We have curated two datasets of protein-ligand complexes featuring small molecules and peptides as ligands, respectively. All complexes with atoms whose elements are beyond $\{C, N, O, F, S, Cl, Se, Br\}$ are not included. The small molecule dataset is sourced from PDB-Bind (Wang et al., 2005) and has 14,348 protein-ligand complexes. The peptide dataset is derived from RCSB PDB (Burley et al., 2023), Propedia (Martins et al., 2023) and PepBDB (Wen et al., 2019). It comprises 20,033 protein-ligand complexes, featuring peptide ligands composed of fewer than 30 residues. Samples are clustered by receptor sequence identity of 0.3 to split the dataset into training and validation sets. To train ATOMSDE, we utilize the curated small molecule dataset and a subset of the peptide dataset containing ligands with fewer than 200 heavy atoms. To train RESROUTER, we use the whole peptide dataset. Please refer to Appendix D for more details about data.

Baselines. As our method is the first cyclic peptide design method based on generative models, we compare our approach with various established methods for linear peptide design: RFDiffusion (Watson et al., 2023) generates protein backbones, and sequences are later predicted by ProteinMPNN (Dauparas et al., 2022); ProteinGenerator (Lisanza et al., 2024) improves RFDiffusion by jointly sampling backbones and corresponding sequences; PepFlow (Li et al., 2025) is a flow-based full-atom peptide generative model that generates the translation, rotation, and side-chain torsion angles of each residue frame within a peptide; PepGLAD (Kong et al., 2024) is a full-atom peptide design method that utilizes a latent diffusion model.

Evaluation. We use Rosetta (Chaudhury et al., 2010) to compute the total energy of reference ligands, linear peptides designed by baseline methods, and cyclic peptides engineered by our approaches. This energy measurement serves as an indicator of the stability (or rationality) of a peptide's 3D binding pose. Hence, we define this energy metric as Stability. We also evaluate the interface binding energy, a crucial metric that indicates the binding affinity of the ligand peptide to its receptor. This assessment is essential for evaluating the ligand peptide's functionality, particularly when designing peptides for therapeutic applications. We denote this type of energy as Affinity. We also report Diversity, the average of one minus the pair-wise TM-Score (Zhang & Skolnick, 2005) among the designed peptides, reflecting structural dissimilarities. As the reference ligands for the targets in the test set are linear, we do not report metrics that necessitate a reference sequence or structure, such as Amino Acid Recovery (AAR) and Root Mean Square Deviation (RMSD). We selected 100 protein pockets with a large volume for testing. Specifically, these targets have receptors with more than 1,000 surrounding atoms around the reference peptide ligands, serving as reliable indicators of adequate volume. For each target, all peptide design methods are used to generate a batch of peptides, from which we select the most promising (i.e., lowest energy) peptide ligand. We design four types of cyclic types (head-to-tail, head-to-side, side-to-side, and side-to-tail) using our methods, respectively, and we also report the mixed results, where the best promising peptide ligands might exhibit different cyclization types. We then report the average and median metrics across all targets. Please refer to Appendix H for more details. This evaluation strategy mirrors practical drug design scenarios where the leading ligand candidates are identified for advancement to the subsequent stages of drug development.

# 4.2 Main Results

The results are shown in Table 1. Among all co-design methods, our method exhibits superior energy performance in terms of both stability and affinity and also best diversity. Interestingly, we find that head-to-tail and head-to-side cyclic peptides show better performance than side-to-tail and side-to-side cyclic peptides. This might be due to the training dataset where C-N bonds (main covalent bonds that form head-to-tail and head-to-side cyclic structures) are more frequent than S-S and C-S bonds (main covalent bonds that form side-to-tail and side-to-side cyclic structures). This aligns with the fact that C-N bonds are generally more stable than S-S and C-S bonds in the physical world. Among all methods, RFDiffusion shows the best energy performance but low diversity, as it tends to generate $\alpha$ -helices in a certain pattern. See Appendix H.5 for ablation studies.

# 4.3 Case Studies

Here, we demonstrate how CPSDE can be seamlessly integrated into real-world cyclic peptide design pipelines and discuss two scenarios: design of SMYD2 peptide inhibitors via head-to-tail cyclization and SET8 inhibitors via side-to-side cyclization.

Table 1. Summary of properties of reference peptides, linear peptides designed by baseline methods, and cyclic peptides designed by CPSDE. (↓) / (↑) denotes a smaller / larger number is better. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Co-Design</td><td rowspan="2">Peptide Type</td><td colspan="2">Stability (↓)</td><td colspan="2">Affinity (↓)</td><td rowspan="2">Diversity (↑)</td></tr><tr><td>Avg.</td><td>Med.</td><td>Avg.</td><td>Med.</td></tr><tr><td>Reference</td><td>N/A</td><td>Linear</td><td>-672.53</td><td>-634.71</td><td>-85.03</td><td>-78.70</td><td>N/A</td></tr><tr><td>RFDiffusion</td><td>✕</td><td>Linear</td><td>-633.51</td><td>-607.82</td><td>-70.30</td><td>-61.35</td><td>0.55</td></tr><tr><td>ProteinGenerator</td><td>√</td><td>Linear</td><td>-576.39</td><td>-554.70</td><td>-46.98</td><td>-40.39</td><td>0.58</td></tr><tr><td>PepFlow</td><td>√</td><td>Linear</td><td>-576.16</td><td>-498.31</td><td>-47.88</td><td>-42.40</td><td>0.70</td></tr><tr><td>PepGLAD</td><td>√</td><td>Linear</td><td>-359.44</td><td>-310.33</td><td>-45.06</td><td>-38.56</td><td>0.79</td></tr><tr><td>CPSDE</td><td>√</td><td>Head-to-tail Cyclic</td><td>-568.04</td><td>-519.66</td><td>-50.86</td><td>-46.62</td><td>0.79</td></tr><tr><td>CPSDE</td><td>√</td><td>Head-to-side Cyclic</td><td>-564.81</td><td>-508.88</td><td>-49.81</td><td>-44.16</td><td>0.79</td></tr><tr><td>CPSDE</td><td>√</td><td>Side-to-tail Cyclic</td><td>-547.14</td><td>-502.20</td><td>-46.92</td><td>-37.25</td><td>0.78</td></tr><tr><td>CPSDE</td><td>√</td><td>Side-to-side Cyclic</td><td>-537.09</td><td>-479.94</td><td>-49.73</td><td>-44.71</td><td>0.79</td></tr><tr><td>CPSDE</td><td>√</td><td>Mix</td><td>-580.67</td><td>-527.80</td><td>-55.71</td><td>-48.42</td><td>0.79</td></tr></table>

PepFlow   
![](images/e838c4a0bc65b77e67a2264695bb3d37daa41c063a2f2adfd29b42585a57bf98.jpg)

<details>
<summary>natural_image</summary>

3D protein structure visualization with alpha helices and beta sheets, no text or labels present
</details>

PDB: 4o6f  
![](images/153f44c0c514db7bdb8944d73af616145edfd4de4c31097412cdb4c353381529.jpg)

<details>
<summary>natural_image</summary>

3D ribbon diagram of a protein structure with alpha helices and beta sheets, rendered in blue and yellow tones (no text or labels)
</details>

Our (head-to-tail)   
![](images/78c559af19d54815a685eacf019036b141041867d99eee91836431036453aa58.jpg)

<details>
<summary>natural_image</summary>

3D protein structure visualization with blue ribbons and red active site residues (no text or labels)
</details>

![](images/5263ef8b70fde7b0d390ea7d776eacbaa04d88637a77df6ccec645883c18d375.jpg)

<details>
<summary>line</summary>

| Time (ns) | RMSD (Å) - Line 1 | RMSD (Å) - Line 2 | RMSD (Å) - Line 3 |
| --------- | ----------------- | ----------------- | ----------------- |
| 0         | ~1.0              | ~1.5              | ~2.0              |
| 20        | ~3.0              | ~3.5              | ~4.0              |
| 40        | ~3.5              | ~4.0              | ~4.5              |
| 60        | ~3.5              | ~4.0              | ~4.5              |
| 80        | ~3.5              | ~4.0              | ~4.5              |
| 100       | ~3.5              | ~4.0              | ~4.5              |
</details>

![](images/e3cbd101bca2b10942cfa7576da8bfe3726fabb7fecad1ac4c637ac06e84e730.jpg)

<details>
<summary>area</summary>

| Density | PepFlow | PDB: 4o6f | Our (head-to-tail) |
| ------- | ------- | --------- | ------------------ |
| 0       | 6       | 2         | 3                  |
| 1       | 5       | 2         | 3                  |
| 2       | 4       | 2         | 3                  |
</details>

Figure 3. Discovery of new SMYD2 cyclic peptide inhibitors via applications of CPSDE. Upper: Visualization of conformational ensembles for the designed peptides sampled by MD. Bottom: RMSD analysis of all heavy atoms within the designed peptides.

Design of SMYD2 peptide inhibitors via head-to-tail cyclization. SMYD2 is an oncogene that critically regulates tumor-related signaling pathways, making it an attractive target for cancer therapy (Zheng et al., 2022). In particular, SMYD2 attenuates estrogen signaling by weakening ERα-dependent transactivation (Zhang et al., 2013). Insight into this interaction comes from the SMYD2–ERα

co-crystal structure (PDB: 4O6F), which reveals ERα in a U-shaped conformation nestled within SMYD2's deep binding pocket (Jiang et al., 2014). The distance between ERα's N-terminal (head) and C-terminal (tail) in this complex is only 5.9 Å, a finding that has inspired the design of rigid head-to-tail cyclic peptides to capitalize on this proximity and enhance binding affinity.

PepFlow   
![](images/dd776a69996db938ede3da94819143920ea9567cb71e8d51bb61135d67d59e33.jpg)

<details>
<summary>natural_image</summary>

Abstract illustration of intertwined blue ribbons with pink and purple elements (no text or symbols)
</details>

PDB: 1zkk   
![](images/c0e59a3757a3450aa9ff173d3a17cccb2a08e54c1cd68c324e4b55e09160115d.jpg)

<details>
<summary>natural_image</summary>

3D protein structure visualization with blue ribbons and yellow active site residues (no text or labels)
</details>

Our (side-to-side)   
![](images/b39eaf1f181f5d6ca1f26052a7a1de01bb86009bc2e48cd785cc87a64cf37305.jpg)

<details>
<summary>natural_image</summary>

3D protein structure visualization with blue ribbons and red highlighted active sites (no text or labels)
</details>

![](images/2e6e9853aee71821d858946ec7f02832638a7a85501bbb984ed7490c0557b88f.jpg)

<details>
<summary>line</summary>

| Time (ns) | RMSD (Å) - Red | RMSD (Å) - Orange | RMSD (Å) - Yellow | RMSD (Å) - Pink |
| --------- | -------------- | ----------------- | ----------------- | --------------- |
| 0         | ~1.5           | ~2.0              | ~2.5              | ~3.0            |
| 20        | ~2.8           | ~3.5              | ~4.0              | ~4.5            |
| 40        | ~2.7           | ~3.8              | ~4.2              | ~5.0            |
| 60        | ~2.6           | ~3.7              | ~4.1              | ~5.2            |
| 80        | ~2.5           | ~3.6              | ~4.0              | ~5.5            |
| 100       | ~2.4           | ~3.5              | ~3.9              | ~5.8            |
</details>

![](images/7def88d63dfb6d909bd414305c0fbf2fb6e9cc9d98243fae8baf61dca28d319d.jpg)

<details>
<summary>bar_stacked</summary>

| Density Range | PepFlow | PDB: 1zkk | Our (side-to-side) |
| ------------- | ------- | --------- | ------------------ |
| 0.0 - 0.1     | 7       | 5         | 4                  |
| 0.1 - 0.2     | 6       | 4         | 3                  |
| 0.2 - 0.3     | 5       | 3         | 2                  |
| 0.3 - 0.4     | 4       | 2         | 1                  |
| 0.4 - 0.5     | 3       | 1         | 0                  |
| 0.5 - 0.6     | 2       | 0         | 0                  |
| 0.6 - 0.7     | 1       | 0         | 0                  |
| 0.7 - 0.8     | 0       | 0         | 0                  |
| 0.8 - 0.9     | 0       | 0         | 0                  |
| 0.9 - 1.0     | 0       | 0         | 0                  |
</details>

Figure 4. Discovery of new SET8 cyclic peptide inhibitors via applications of CPSDE. Upper: Visualization of conformational ensembles for the designed peptides sampled by MD. Bottom: RMSD analysis of all heavy atoms within the designed peptides.

Herein, CPSDE was employed to design new SMYD2 inhibitors by generating head-to-tail cyclic peptides. The binding site of ERα (PDB: 4o6f) was selected as the active pocket, and 8 cyclic peptides were generated through head-to-tail cyclization. All peptides demonstrated favorable Rosetta affinity scores (Figure 10), with H2T-6 achieving the most favorable score of -33.9 kcal/mol. To further accurately assess its binding stability and affinity, 100 ns molecular dynamics (MD) simulations were performed on the H2T-6-SMYD2 complex. More details on the system preparation and simulation protocol of MD are in Appendix H.6. The simulations also included the crystallized linear peptide (ground truth) and the PepFlow-generated linear peptide for comparison. Each simulation was repeated twice to ensure consistency. As shown in Figure 3, the linear peptide generated by PepFlow exhibited higher flexibility, with an average peptide RMSD of 4.59 Å over the last 50 ns of equilibrated trajectories. In contrast, H2T-6 displayed an average peptide RMSD of 3.05 Å, comparable to the ground truth linear peptide (1.92 Å). More importantly, binding free energy analysis using MM-PBSA (Wang et al., 2019) revealed that H2T-6 had the highest binding affinity (-24.02 kcal/mol), outperforming both the ground truth peptide (-19.00 kcal/mol) and the PepFlow-generated peptide (-7.26 kcal/mol). These results suggest that H2T-6 maintains a stable binding conformation which also exhibits high binding affinity, indicating its potential as a candidate SMYD2 inhibitor for further investigation.

Design of SET8 peptide inhibitors via side-to-side cyclization. SET8 is the only lysine methyltransferase that specifically catalyzes the methylation of histone H4 at the 20th lysine (Qian & Zhou, 2006). SET8-mediated protein modifications are involved in numerous physiological processes, and its dysregulation is closely linked to various human diseases, particularly cancer development and prognosis (Yang et al., 2021).

Again, we applied CPSDE to design seed cyclic peptide inhibitors targeting SET8. The 3D structure of SET8 in complex with an H4 peptide (PDB: 1zkk) was used, with the H4 binding site selected as the active pocket (Couture et al., 2005). To explore an alternative cyclization approach, we employed a widely used side-to-side strategy to generate 8 cyclic peptides. Likely, we picked up the top cyclic peptide, S2S-4 (see Figure 11), based on its Rosetta affinity score, and performed 100 ns MD simulations with 2 repeats, along with two reference linear peptides. The RMSD and MM-PBSA analyses revealed that S2S-4 not only demonstrated a significantly lower peptide RMSD (2.54 Å) compared to the reference linear peptides (ground truth: 4.06 Å; PepFlow: 5.23 Å), but also exhibited a lower binding free energy (S2S-4: -12.48 kcal/mol; ground truth: -6.39 kcal/mol; PepFlow: -9.26 kcal/mol). These results indicate that S2S-4 might serve as a potential candidate for later studies on SET8 inhibition.

# 5 Conclusion

In conclusion, CPSDE is a generative algorithm capable of producing diverse types of cyclic peptides given 3D receptor structures, thereby paving the way for advancements in peptide-based drug discovery. Our approach enhances drug development by designing stable and high-affinity cyclic peptides. Case studies supported by molecular dynamics simulations validate its practical utility and real-world applicability. Limitations and future work are discussed in Appendix I.

# Acknowledgments

We thank anonymous reviewers for their insightful feedback. We would like to extend our gratitude to Yi Zhou and Lihao Wang for their valuable feedback on our methodology, as well as to Ellen Wang, Tianze Zheng and Wen Yan for their assistance with the molecular dynamics simulations.

# Impact Statement

CPSDE contributes to the advancement of computational biology, particularly in the field of cyclic peptide design, by addressing key challenges that have hindered progress in this area. By integrating generative structure prediction and sequence prediction, our approach enables the design of stable and high-affinity cyclic peptides, offering valuable applications in drug discovery and therapeutic development. While our primary focus is on the positive applications of CPSDE, such as developing novel peptide-based treatments, we acknowledge the ethical considerations associated with any powerful generative tool. There is a potential risk of misuse, including the design of harmful bioactive compounds. To mitigate such concerns, we emphasize the responsible use of our method and encourage the scientific community to apply it for constructive and beneficial purposes. Our commitment to ethical research practices ensures that CPSDE serves as a force for innovation in pharmaceutical development while upholding societal safety and integrity.

# References

Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, pp. 1–3, 2024.   
Alford, R. F., Leaver-Fay, A., Jeliazkov, J. R., O'Meara, M. J., DiMaio, F. P., Park, H., Shapovalov, M. V., Renfrew, P. D., Mulligan, V. K., Kappel, K., et al. The rosetta all-atom energy function for macromolecular modeling and design. Journal of chemical theory and computation, 13(6):3031–3048, 2017.   
Baek, M., DiMaio, F., Anishchenko, I., Dauparas, J., Ovchinnikov, S., Lee, G. R., Wang, J., Cong, Q., Kinch, L. N., Schaeffer, R. D., et al. Accurate prediction of protein structures and interactions using a three-track neural network. Science, 373(6557):871–876, 2021.   
Berman, H. M., Westbrook, J., Feng, Z., Gilliland, G., Bhat, T. N., Weissig, H., Shindyalov, I. N., and Bourne, P. E. The protein data bank. \*Nucleic acids research\*, 28(1): 235–242, 2000.   
Bhardwaj, G., Mulligan, V. K., Bahl, C. D., Gilmore, J. M., Harvey, P. J., Cheneval, O., Buchko, G. W., Pulavarti, S. V., Kaas, Q., Eletsky, A., et al. Accurate de novo design of hyperstable constrained peptides. Nature, 538(7625):329–335, 2016.   
Bhardwaj, G., O'Connor, J., Rettie, S., Huang, Y.-H., Ramelot, T. A., Mulligan, V. K., Alpkilic, G. G., Palmer, J., Bera, A. K., Bick, M. J., et al. Accurate de novo design of membrane-traversing macrocycles. Cell, 185(19):3520–3532, 2022.   
Boom, J. D., Greenig, M., Sormanni, P., and Liò, P. Score-based generative models for designing binding peptide backbones, 2024. URL https://arxiv.org/abs/2310.07051.   
Buckton, L. K., Rahimi, M. N., and McAlpine, S. R. Cyclic peptides as drugs for intracellular targets: the next frontier in peptide therapeutic development. Chemistry–A European Journal, 27(5):1487–1513, 2021.   
Burley, S. K., Bhikadiya, C., Bi, C., Bittrich, S., Chao, H., Chen, L., Craig, P. A., Crichlow, G. V., Dalenberg, K., Duarte, J. M., et al. Rcsb protein data bank (rcsb.org): delivery of experimentally-determined pdb structures alongside one million computed structure models of proteins from artificial intelligence/machine learning. Nucleic acids research, 51(D1):D488–D508, 2023.   
Camarero, J. A. and Muir, T. W. Biosynthesis of a head-to-tail cyclized protein with improved biological activity.

Journal of the American Chemical Society, 121(23):5597–5598, 1999.

Campbell, A., Yim, J., Barzilay, R., Rainforth, T., and Jaakkola, T. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In Forty-first International Conference on Machine Learning, 2024.

Cao, D., Chen, M., Zhang, R., Wang, Z., Huang, M., Yu, J., Jiang, X., Fan, Z., Zhang, W., Zhou, H., et al. Surfdock is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction. Nature Methods, pp. 1–13, 2024.

Cao, L., Coventry, B., Goreshnik, I., Huang, B., Sheffler, W., Park, J. S., Jude, K. M., Marković, I., Kadam, R. U., Verschueren, K. H., et al. Design of protein-binding proteins from the target structure alone. Nature, 605(7910):551–560, 2022.

Caplin, M. E., Pavel, M., Ćwikła, J. B., Phan, A. T., Raderer, M., Sedláčková, E., Cadiot, G., Wolin, E. M., Capdevila, J., Wall, L., et al. Lanreotide in metastatic enteropancreatic neuroendocrine tumors. New England Journal of Medicine, 371(3):224–233, 2014.

Chaudhury, S., Lyskov, S., and Gray, J. J. Pyrosetta: a script-based interface for implementing molecular modeling algorithms using rosetta. Bioinformatics, 26(5):689–691, 2010.

Chen, R., Xue, D., Zhou, X., Zheng, Z., Zeng, X., and Gu, Q. An all-atom generative model for designing protein complexes. In International Conference on Machine Learning, 2025.

Chen, T., Pertsemlidis, S., and Chatterjee, P. PepMLM: Target sequence-conditioned generation of peptide binders via masked language modeling. In ICLR 2024 Workshop on Generative and Experimental Perspectives for Biomolecular Design, 2024. URL https://openreview.net/forum?id=p6fz0rq7zu.

Cheng, X., Zhou, X., Yang, Y., Bao, Y., and Gu, Q. Decomposed direct preference optimization for structure-based drug design. arXiv preprint arXiv:2407.13981, 2024.

Chu, A. E., Kim, J., Cheng, L., El Nesr, G., Xu, M., Shuai, R. W., and Huang, P.-S. An all-atom protein generative model. Proceedings of the National Academy of Sciences, 121(27):e2311500121, 2024.

Corso, G., Stärk, H., Jing, B., Barzilay, R., and Jaakkola, T. S. Diffdock: Diffusion steps, twists, and turns for molecular docking. In The Eleventh International Conference on Learning Representations, 2023.

Costa, L., Sousa, E., and Fernandes, C. Cyclic peptides in pipeline: what future for these great molecules? \*Pharmaceuticals\*, 16(7):996, 2023.   
Couture, J.-F., Collazo, E., Brunzelle, J. S., and Trievel, R. C. Structural and functional analysis of set8, a histone h4 lys-20 methyltransferase. Genes & development, 19(12):1455–1465, 2005.   
Dauparas, J., Anishchenko, I., Bennett, N., Bai, H., Ragotte, R. J., Milles, L. F., Wicky, B. I., Courbet, A., de Haas, R. J., Bethel, N., et al. Robust deep learning–based protein sequence design using proteinmpnn. Science, 378(6615):49–56, 2022.   
Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.   
Dolinsky, T. J., Czodrowski, P., Li, H., Nielsen, J. E., Jensen, J. H., Klebe, G., and Baker, N. A. Pdb2pqr: expanding and upgrading automated preparation of biomolecular structures for molecular simulations. \*Nucleic acids research\*, 35(suppl\_2):W522–W525, 2007.   
Driggers, E. M., Hale, S. P., Lee, J., and Terrett, N. K. The exploration of macrocycles for drug discovery—an underexploited structural class. Nature Reviews Drug Discovery, 7(7):608–624, 2008.   
Eberhardt, J., Santos-Martins, D., Tillack, A. F., and Forli, S. Autodock vina 1.2. 0: New docking methods, expanded force field, and python bindings. Journal of chemical information and modeling, 61(8):3891–3898, 2021.   
Fang, P., Pang, W.-K., Xuan, S., Chan, W.-L., and Leung, K. C.-F. Recent advances in peptide macrocyclization strategies. Chemical Society Reviews, 2024.   
Fass, D. Disulfide bonding in protein biophysics. Annual review of biophysics, 41(1):63–79, 2012.   
Friesner, R. A., Banks, J. L., Murphy, R. B., Halgren, T. A., Klicic, J. J., Mainz, D. T., Repasky, M. P., Knoll, E. H., Shelley, M., Perry, J. K., et al. Glide: a new approach for rapid, accurate docking and scoring. 1. method and assessment of docking accuracy. Journal of medicinal chemistry, 47(7):1739–1749, 2004.   
Gao, Z., Tan, C., Chen, X., Zhang, Y., Xia, J., Li, S., and Li, S. Z. Kw-design: Pushing the limit of protein design via knowledge refinement. In The Twelfth International Conference on Learning Representations, 2023a.   
Gao, Z., Tan, C., and Li, S. Z. Pifold: Toward effective and efficient protein inverse folding. In The Eleventh International Conference on Learning Representations, 2023b.

Garcia Jimenez, D., Poongavanam, V., and Kihlberg, J. Macrocycles in drug discovery-learning from the past for the future. Journal of Medicinal Chemistry, 66(8):5377–5396, 2023.   
Giordanetto, F. and Kihlberg, J. Macrocyclic drugs and clinical candidates: what can medicinal chemists learn from their properties? Journal of medicinal chemistry, 57(2):278–295, 2014.   
Guan, J., Qian, W. W., Ma, W.-Y., Ma, J., and Peng, J. Energy-inspired molecular conformation optimization. In international conference on learning representations, 2021.   
Guan, J., Li, J., Zhou, X., Peng, X., Wang, S., Luo, Y., Peng, J., and Ma, J. Group ligands docking to protein pockets. arXiv preprint arXiv:2501.15055, 2025.   
Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.   
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
Hopkins, C. W., Le Grand, S., Walker, R. C., and Roitberg, A. E. Long-time-step molecular dynamics through hydrogen mass repartitioning. Journal of chemical theory and computation, 11(4):1864–1874, 2015.   
Hosseinzadeh, P., Watson, P. R., Craven, T. W., Li, X., Rettie, S., Pardo-Avila, F., Bera, A. K., Mulligan, V. K., Lu, P., Ford, A. S., et al. Anchor extension: a structure-guided approach to design cyclic peptides targeting enzyme active sites. Nature Communications, 12(1):3384, 2021.   
Hsu, C., Verkuil, R., Liu, J., Lin, Z., Hie, B., Sercu, T., Lerer, A., and Rives, A. Learning inverse folding from millions of predicted structures. In International conference on machine learning, pp. 8946–8970. PMLR, 2022.   
Huang, Y., Zhang, O., Wu, L., Tan, C., Lin, H., Gao, Z., Li, S., and Li, S. Z. Re-dock: Towards flexible and realistic molecular docking with diffusion bridge. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=QRjTDhCIO8.   
Ingraham, J., Garg, V., Barzilay, R., and Jaakkola, T. Generative models for graph-based protein design. Advances in neural information processing systems, 32, 2019.   
Jiang, Y., Trescott, L., Holcomb, J., Zhang, X., Brunzelle, J., Sirinupong, N., Shi, X., and Yang, Z. Structural insights into estrogen receptor $\alpha$ methylation by histone methyltransferase smyd2, a cellular event implicated in estrogen signaling regulation. Journal of molecular biology, 426(20):3413–3425, 2014.

Jin, W., Wohlwend, J., Barzilay, R., and Jaakkola, T. S. Iterative refinement graph neural network for antibody sequence-structure co-design. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=LI2bhrE\_2A.   
Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., and Dror, R. Learning from protein structure with geometric vector perceptrons. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=1YLJDvSx6J4.   
Jing, B., Erives, E., Pao-Huang, P., Corso, G., Berger, B., and Jaakkola, T. S. Eigenfold: Generative protein structure prediction with diffusion models. In ICLR 2023-Machine Learning for Drug Discovery workshop, 2023.   
Jorgensen, W. L., Chandrasekhar, J., Madura, J. D., Impey, R. W., and Klein, M. L. Comparison of simple potential functions for simulating liquid water. The Journal of chemical physics, 79(2):926–935, 1983.   
Kale, S. S., Villequey, C., Kong, X.-D., Zorzi, A., Deyle, K., and Heinis, C. Cyclization of peptides with two chemical bridges affords large scaffold diversities. Nature chemistry, 10(7):715–723, 2018.   
Kawamura, A., Münzel, M., Kojima, T., Yapp, C., Bhushan, B., Goto, Y., Tumber, A., Katoh, T., King, O. N., Passioura, T., et al. Highly selective inhibition of histone demethylases by de novo macrocyclic peptides. Nature communications, 8(1):14773, 2017.   
Kong, X., Huang, W., and Liu, Y. End-to-end full-atom antibody design. In Proceedings of the 40th International Conference on Machine Learning, pp. 17409–17429, 2023.   
Kong, X., Jia, Y., Huang, W., and Liu, Y. Full-atom peptide design with geometric latent diffusion. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=IAQNJUJe8q.   
Krishna, R., Wang, J., Ahern, W., Sturmfels, P., Venkatesh, P., Kalvet, I., Lee, G. R., Morey-Burrows, F. S., Anishchenko, I., Humphreys, I. R., et al. Generalized biomolecular modeling and design with rosettafold all-atom. Science, 384(6693):eadl2528, 2024.   
Kulytè, P., Vargas, F., Mathis, S. V., Wang, Y. G., Hernández-Lobato, J. M., and Liò, P. Improving antibody design with force-guided sampling in diffusion models. arXiv preprint arXiv:2406.05832, 2024.

Li, J., Cheng, C., Wu, Z., Guo, R., Luo, S., Ren, Z., Peng, J., and Ma, J. Full-atom peptide design based on multimodal flow matching. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. JMLR.org, 2025.   
Li, M., Lan, X., Shi, X., Zhu, C., Lu, X., Pu, J., Lu, S., and Zhang, J. Delineating the stepwise millisecond allosteric activation mechanism of the class c gpcr dimer mglu5. Nature Communications, 15(1):7519, 2024.   
Lin, H., Zhang, O., Zhao, H., Jiang, D., Wu, L., Liu, Z., Huang, Y., and Li, S. Z. Ppflow: target-aware peptide design with torsional flow matching. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. JMLR.org, 2025.   
Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, 2023.   
Lisanza, S. L., Gershon, J. M., Tipps, S. W., Sims, J. N., Arnoldt, L., Hendel, S. J., Simma, M. K., Liu, G., Yase, M., Wu, H., et al. Multistate and functional protein design using rosettafold sequence space diffusion. Nature Biotechnology, pp. 1–11, 2024.   
Liu, L., Yang, L., Cao, S., Gao, Z., Yang, B., Zhang, G., Zhu, R., and Wu, D. Cyclicpepedia: a knowledge base of natural and synthetic cyclic peptides. Briefings in Bioinformatics, 25(3):bbae190, 2024.   
Loshchilov, I. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
Lu, C., Chen, H., Chen, J., Su, H., Li, C., and Zhu, J. Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning. In International Conference on Machine Learning, pp. 22825–22855. PMLR, 2023.   
Lu, W., Wu, Q., Zhang, J., Rao, J., Li, C., and Zheng, S. Tankbind: Trigonometry-aware neural networks for drug-protein binding structure prediction. Advances in neural information processing systems, 35:7236–7249, 2022.   
Lu, W., Zhang, J., Huang, W., Zhang, Z., Jia, X., Wang, Z., Shi, L., Li, C., Wolynes, P. G., and Zheng, S. Dynamicbind: Predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model. Nature Communications, 15(1):1071, 2024.   
Luo, S., Su, Y., Peng, X., Wang, S., Peng, J., and Ma, J. Antigen-specific antibody design and optimization with diffusion-based generative models for protein structures. Advances in Neural Information Processing Systems, 35:9754–9767, 2022.

Madani, A., Krause, B., Greene, E. R., Subramanian, S., Mohr, B. P., Holton, J. M., Olmos, J. L., Xiong, C., Sun, Z. Z., Socher, R., et al. Large language models generate functional protein sequences across diverse families. Nature Biotechnology, 41(8):1099–1106, 2023.   
Maier, J. A., Martinez, C., Kasavajhala, K., Wickstrom, L., Hauser, K. E., and Simmerling, C. ff14sb: improving the accuracy of protein side chain and backbone parameters from ff99sb. Journal of chemical theory and computation, 11(8):3696–3713, 2015.   
Makowski, E. K., Wang, T., Zupancic, J. M., Huang, J., Wu, L., Schardt, J. S., De Groot, A. S., Elkins, S. L., Martin, W. D., and Tessier, P. M. Optimization of therapeutic antibodies for reduced self-association and non-specific binding via interpretable machine learning. Nature biomedical engineering, 8(1):45–56, 2024.   
Mao, W., Zhu, M., Sun, Z., Shen, S., Wu, L. Y., Chen, H., and Shen, C. De novo protein design using geometric vector field networks. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=9UIGyJJpay.   
Martinkus, K., Ludwiczak, J., LIANG, W.-C., Lafrance-Vanasse, J., Hotzel, I., Rajpal, A., Wu, Y., Cho, K., Bonneau, R., Gligorijevic, V., and Loukas, A. Abdiffuser: full-atom generation of in-vitro functioning antibodies. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=7GyYpomkEa.   
Martins, P., Mariano, D., Carvalho, F. C., Bastos, L. L., Moraes, L., Paixão, V., and Cardoso de Melo-Minardi, R. Propedia v2. 3: A novel representation approach for the peptide-protein interaction database using graph-based structural signatures. Frontiers in Bioinformatics, 3:1103103, 2023.   
Merz, M. L., Habeshian, S., Li, B., David, J.-A. G., Nielsen, A. L., Ji, X., Il Khwildy, K., Duany Benitez, M. M., Phothirath, P., and Heinis, C. De novo development of small cyclic peptides that are orally bioavailable. Nature Chemical Biology, 20(5):624–633, 2024.   
Peacock, H. and Suga, H. Discovery of de novo macrocyclic peptides by messenger rna display. Trends in Pharmacological Sciences, 42(5):385–397, 2021.   
Pei, Q., Gao, K., Wu, L., Zhu, J., Xia, Y., Xie, S., Qin, T., He, K., Liu, T.-Y., and Yan, R. Fabind: Fast and accurate protein-ligand binding. Advances in Neural Information Processing Systems, 36, 2024.   
Qian, C. and Zhou, M. M. Set domain protein lysine methyltransferases: Structure, specificity and catalysis. Cellular and molecular life sciences CMLS, 63:2755–2763, 2006.

Qiao, Z., Nie, W., Vahdat, A., Miller III, T. F., and Anandkumar, A. State-specific protein–ligand complex structure prediction with a multiscale deep generative model. Nature Machine Intelligence, 6(2):195–208, 2024.   
Rettie, S., Juergens, D., Adebomi, V., Bueso, Y. F., Zhao, Q., Leveille, A., Liu, A., Bera, A., Wilms, J., Üffing, A., et al. Accurate de novo design of high-affinity protein binding macrocycles using deep learning. bioRxiv, pp. 2024–11, 2024.   
Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, C. L., Ma, J., et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences, 118(15):e2016239118, 2021.   
Ryckaert, J.-P., Ciccotti, G., and Berendsen, H. J. Numerical integration of the cartesian equations of motion of a system with constraints: molecular dynamics of n-alkanes. Journal of computational physics, 23(3):327–341, 1977.   
Salomon-Ferrer, R., Gotz, A. W., Poole, D., Le Grand, S., and Walker, R. C. Routine microsecond molecular dynamics simulations with amber on gpus. 2. explicit solvent particle mesh ewald. Journal of chemical theory and computation, 9(9):3878–3888, 2013.   
Särkkä, S. and Solin, A. Applied stochastic differential equations, volume 10. Cambridge University Press, 2019.   
Satorras, V. G., Hoogeboom, E., and Welling, M. E (n) equivariant graph neural networks. In International conference on machine learning, pp. 9323–9332. PMLR, 2021.   
Sharma, K., Sharma, K. K., Sharma, A., and Jain, R. Peptide-based drug discovery: Current status and recent advances. Drug Discovery Today, 28(2):103464, 2023.   
Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.   
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=PxTIG12RRHS.   
Stärk, H., Ganea, O., Pattanaik, L., Barzilay, R., and Jaakkola, T. Equibind: Geometric deep learning for drug binding structure prediction. In International conference on machine learning, pp. 20503–20521. PMLR, 2022.

Stark, H., Jing, B., Barzilay, R., and Jaakkola, T. Harmonic prior self-conditioned flow matching for multiligand docking and binding site design. In NeurIPS 2023 AI for Science Workshop, 2023.   
Tang, S., Zhang, Y., and Chatterjee, P. Peptune: De novo generation of therapeutic peptides with multi-objective-guided discrete diffusion. ArXiv, pp. arXiv–2412, 2025.   
Tsaban, T., Varga, J. K., Avraham, O., Ben-Aharon, Z., Khramushin, A., and Schueler-Furman, O. Harnessing protein folding neural networks for peptide–protein docking. Nature communications, 13(1):176, 2022.   
Tsomaia, N. Peptide therapeutics: targeting the undruggable space. European journal of medicinal chemistry, 94:459–470, 2015.   
van Gelder, T., Lerma, E., Engelke, K., and Huizinga, R. B. Voclosporin: a novel calcineurin inhibitor for the treatment of lupus nephritis. Expert review of clinical pharmacology, 15(5):515–529, 2022.   
Vinogradov, A. A., Yin, Y., and Suga, H. Macrocyclic peptides as drug candidates: recent progress and remaining challenges. Journal of the American Chemical Society, 141(10):4167–4181, 2019.   
Wang, E., Sun, H., Wang, J., Wang, Z., Liu, H., Zhang, J. Z., and Hou, T. End-point binding free energy calculation with mm/pbsa and mm/gbsa: strategies and applications in drug design. Chemical reviews, 119(16):9478–9508, 2019.   
Wang, F., Wang, Y., Feng, L., Zhang, C., and Lai, L. Target-specific de novo peptide binder design with diffpepbuilder. Journal of Chemical Information and Modeling, 2024a.   
Wang, R., Fang, X., Lu, Y., Yang, C.-Y., and Wang, S. The pdbbind database: methodologies and updates. Journal of medicinal chemistry, 48(12):4111–4119, 2005.   
Wang, X., Zheng, Z., Fei, Y., Xue, D., Huang, S., and Gu, Q. Diffusion language models are versatile protein learners. In International conference on machine learning, 2024b.   
Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., and Gu, Q. Dplm-2: A multimodal diffusion protein language model. In International Conference on Learning Representations, 2025.   
Watson, J. L., Juergens, D., Bennett, N. R., Trippe, B. L., Yim, J., Eisenach, H. E., Ahern, W., Borst, A. J., Ragotte, R. J., Milles, L. F., et al. De novo design of protein structure and function with rfdiffusion. Nature, 620(7976):1089–1100, 2023.

Wei, H., Wang, W., Peng, Z., and Yang, J. Q-biolip: A comprehensive resource for quaternary structure-based protein–ligand interactions. Genomics, Proteomics & Bioinformatics, 22(1), 2024.   
Wen, Z., He, J., Tao, H., and Huang, S.-Y. Pepbdb: a comprehensive structural database of biological peptide–protein interactions. Bioinformatics, 35(1):175–177, 2019.   
Weng, G., Gao, J., Wang, Z., Wang, E., Hu, X., Yao, X., Cao, D., and Hou, T. Comprehensive evaluation of fourteen docking programs on protein–peptide complexes. Journal of Chemical Theory and Computation, 16(6):3959–3969, 2020. doi: 10.1021/acs.jctc.9b01208. URL https://doi.org/10.1021/acs.jctc.9b01208. PMID: 32324992.   
Xie, X., Valiente, P. A., and Kim, P. M. Helixgan a deep-learning methodology for conditional de novo design of $\alpha$ -helix structures. Bioinformatics, 39(1):btad036, 2023.   
Xie, X., Valiente, P. A., Kim, J., and Kim, P. M. Helixdiff, a score-based diffusion model for generating all-atom $\alpha$ -helical structures. ACS Central Science, 10(5):1001–1011, 2024.   
Yang, C., Wang, K., Zhou, Y., and Zhang, S.-L. Histone lysine methyltransferase set8 is a novel therapeutic target for cancer treatment. Drug Discovery Today, 26(10):2423–2430, 2021.   
Ye, F., Zheng, Z., Xue, D., Shen, Y., Wang, L., Ma, Y., Wang, Y., Wang, X., Zhou, X., and Gu, Q. Proteinbench: A holistic evaluation of protein foundation models. arXiv preprint arXiv:2409.06744, 2024.   
Yi, K., Zhou, B., Shen, Y., Liò, P., and Wang, Y. Graph denoising diffusion for inverse protein folding. Advances in Neural Information Processing Systems, 36, 2024.   
Yim, J., Trippe, B. L., De Bortoli, V., Mathieu, E., Doucet, A., Barzilay, R., and Jaakkola, T. Se (3) diffusion model with application to protein backbone generation. In Proceedings of the 40th International Conference on Machine Learning, pp. 40001–40039, 2023.   
Yim, J., Campbell, A., Mathieu, E., Foong, A. Y. K., Gastegger, M., Jiménez-Luna, J., Lewis, S., Satorras, V. G., Veeling, B. S., Noé, F., Barzilay, R., and Jaakkola, T. S. Improved motif-scaffolding with se(3) flow matching, 2024. URL https://arxiv.org/abs/2401.04082.   
Zhang, X., Tanaka, K., Yan, J., Li, J., Peng, D., Jiang, Y., Yang, Z., Barton, M. C., Wen, H., and Shi, X. Regulation of estrogen receptor $\alpha$ by histone methyltransferase smyd2-mediated protein methylation. Proceedings of the

National Academy of Sciences, 110(43):17284–17289, 2013.   
Zhang, Y. and Skolnick, J. Tm-align: a protein structure alignment algorithm based on the tm-score. Nucleic acids research, 33(7):2302–2309, 2005.   
Zheng, Q., Zhang, W., and Rao, G.-W. Protein lysine methyltransferase smyd2: a promising small molecule target for cancer therapy. Journal of Medicinal Chemistry, 65(15):10119–10132, 2022.   
Zheng, Z., Deng, Y., Xue, D., Zhou, Y., Ye, F., and Gu, Q. Structure-informed language models are protein designers. In International conference on machine learning, pp. 42317–42338. PMLR, 2023.   
Zhou, G., Gao, Z., Ding, Q., Zheng, H., Xu, H., Wei, Z., Zhang, L., and Ke, G. Uni-mol: A universal 3d molecular representation learning framework. In The Eleventh International Conference on Learning Representations, 2023.   
Zhou, X., Cheng, X., Yang, Y., Bao, Y., Wang, L., and Gu, Q. Decompopt: Controllable and decomposed diffusion models for structure-based molecular optimization. arXiv preprint arXiv:2403.13829, 2024a.   
Zhou, X., Guan, J., Zhang, Y., Peng, X., Wang, L., and Ma, J. Reprogramming pretrained target-specific diffusion models for dual-target drug design. Advances in Neural Information Processing Systems, 37:87255–87281, 2024b.   
Zhou, X., Wang, L., and Zhou, Y. Stabilizing policy gradients for stochastic differential equations via consistency with perturbation process. arXiv preprint arXiv:2403.04154, 2024c.   
Zhou, X., Xue, D., Chen, R., Zheng, Z., Wang, L., and Gu, Q. Antigen-specific antibody design via direct energy-based preference optimization. Advances in Neural Information Processing Systems, 37:120861–120891, 2024d.   
Zhou, X., Xiao, Y., Lin, H., He, X., Guan, J., Wang, Y., Liu, Q., Zhou, F., Wang, L., and Ma, J. Integrating protein dynamics into structure-based drug design via full-atom stochastic flows. arXiv preprint arXiv:2503.03989, 2025.   
Zorzi, A., Deyle, K., and Heinis, C. Cyclic peptide therapeutics: past, present and future. Current opinion in chemical biology, 38:24–29, 2017.   
Zotchev, S. B., Stepanchikova, A. V., Sergeyko, A. P., Sobolev, B. N., Filimonov, D. A., and Poroikov, V. V. Rational design of macrolides by virtual screening of combinatorial libraries generated through in silico manipulation of polyketide synthases. Journal of medicinal chemistry, 49(6):2077–2087, 2006.

# A Introduction of Cyclic Peptides

Peptides have shown promising capability as therapeutics for protein targets where small molecules struggle to bind, due to their ability to modulate protein-protein interactions (Hosseinzadeh et al., 2021; Zorzi et al., 2017). However, traditional linear peptides are usually polar due to exposed acids and amines in terminals, limiting their membrane permeability and proteolytic stability, and also restricts their administration options in drug development (Buckton et al., 2021; Merz et al., 2024).

The peptide cyclization is capable of enhancing the conformational stability, increasing binding affinity and specificity for targets (Zorzi et al., 2017). This has fueled the growing interest in cyclic peptide research as efficient therapeutics. Figure 5 shows two cyclic peptide drugs approved in recent 20 years. According to Sharma et al. (2023); Fang et al. (2024); Costa et al. (2023), cyclic peptides can be classified by their cyclization strategies: head-to-tail cyclization (between N- and C-termini); side-to-side cyclization (between two side chains, including peptide stapling, which stabilizes $\alpha$ -helical structures) (Vinogradov et al., 2019); head-to-side and side-to-tail cyclization (between a terminal and side chain); and polycyclization (with multiple cycles).

Voclosporin   
![](images/4530700aa02036e29e753bc150ab36ad300a38791223c9c2503b4fcb8214176b.jpg)

<details>
<summary>chemical</summary>

3D molecular structure showing a complex organic compound with yellow, blue, and red atoms in a protein binding pocket
</details>

![](images/c12fb95f6b8dcccafd4aa950f0c88316ef57684701daf3fad25e9264cab7e7c3.jpg)

<details>
<summary>chemical</summary>

Complex peptide or glycoside molecular structure with multiple functional groups and side chains
</details>

Lanreotide   
![](images/bf40b2584fdfd19a39ee9a069c9ab30510442dc037df285f2e511cb7a25e8ef1.jpg)

<details>
<summary>natural_image</summary>

Molecular structure visualization with yellow highlighted region, surrounded by translucent protein or lipid bilayer (no text or labels)
</details>

![](images/a24ed2550660dc9758a44079845e72e7bde3b26ccad881282dd1ae2b4303faaa.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups including amide, thioether, and hydroxyl groups
</details>

Figure 5. Two examples of cyclic peptide drugs. Voclosporin (van Gelder et al., 2022), an analog of ciclosporin, is an immunosuppressant used to treat lupus nephritis. Lanreotide (Caplin et al., 2014), an analog of somatostatin, is an oncology drug that inhibits growth hormone release and is used to manage carcinoid syndrome.

# B Extend Related Work

Protein-ligand Docking. Protein-ligand docking aims to predict the conformation of protein-ligand complexes, playing a crucial role in drug discovery and design. Compared to traditional score-based or template-based docking methods (Eberhardt et al., 2021; Friesner et al., 2004), deep learning-based approaches are faster while achieving comparable accuracy. A common strategy involves leveraging equivariant or invariant graph neural networks to model both protein and ligand and predict ligand atom positions (Lu et al., 2022; Stärk et al., 2022; Pei et al., 2024; Qiao et al., 2024). Other approaches, such as Uni-Mol (Zhou et al., 2023) and RoseTTAFold All-Atom (Krishna et al., 2024), use positional encoding and attention layers for atom-level representations and predictions. Unlike structure prediction models, Diffdock (Corso et al., 2023) introduces the generative docking framework that views docking as a conditional generation problem, and uses a diffusion model for end-to-end ligand generation. SurfDock (Cao et al., 2024) further incorporates protein surface graphs to refine docking predictions. More recent studies have emphasized flexible docking models, which consider protein conformational variability in docking process. DynamicBind (Lu et al., 2024) and ReDock (Huang et al., 2024) extend the

![](images/e9b980a93b5326d847bb1f9b8bfd62d7d1b0732afe4e8be853b44570d73916a3.jpg)  
Figure 6. Examples of 3D structures of cyclic peptides and their chemical graphs. The cyclization structures are highlighted in pink. The corresponding cyclization types are: 5I2I\_E: head-to-tail; 3AV9\_X: head-to-tail; 5TXE\_C: head-to-side; 3QG6\_C: side-to-tail; 1U91\_C: side-to-side; 1RGR\_B: side-to-side (stapling); 3QN7\_B and 6Q1U\_D: polycyclization.

generative docking paradigm by incorporating diffusion-based modeling of protein structures, enabling joint optimization of both protein and ligand poses. Guan et al. (2025) proposed a novel molecular docking framework that simultaneously considers multiple ligands docking to a protein, enhancing molecular docking accuracy.

Protein Sequence Design. Designing protein sequences that fold into desired structures, known as inverse folding, is a fundamental task in protein engineering. Deep learning methods can effectively predict protein sequences given structural priors. These models generally follow a two-stage paradigm comprising a structure encoding module and a sequence prediction module. Several models, including GVP (Jing et al., 2021), ESM-IF (Hsu et al., 2022), ProteinMPNN (Dauparas et al., 2022), and Ingraham et al. (2019), utilize graph neural networks (GNNs) for structural encoding and employ autoregressive decoding for sequence prediction. PiFold (Gao et al., 2023b) adopts a similar encoding strategy but directly classifies amino acids for each node. GRADE-IF (Yi et al., 2024) formulates the inverse folding problem as learning the conditional distribution of protein sequence given the backbone structure, and employs a discrete diffusion model on protein graph to predict the amino acid type on each node. Besides graph-based approaches, Protein Language Models (PLMs) (Lin et al., 2023; Rives et al., 2021) offer an alternative strategy for inverse folding. Zheng et al. (2023) integrates PLMs as sequence decoders for encoded protein graphs, while Gao et al. (2023a) and Mao et al. (2024) incorporate ESM-2 (Lin et al., 2023) embeddings to refine sequence predictions. ProGen (Madani et al., 2023) shows an autoregressive language model trained on massive protein sequence data can utilize property tags for controllable sequence generation. Wang et al. (2024b) demonstrate that discrete diffusion serves as a more principled probabilistic framework for large-scale protein language modeling. The resulting diffusion protein language model (DPLM) excels in both sequence generation and representation learning. DPLM-2 (Wang et al., 2025) further extends this discrete diffusion-based paradigm by incorporating tokenized 3D structures thereafter, enabling structure-sequence co-generation as well as any-to-any conditional generation

![](images/fe56c216e87afc66138d3fda3101c8626ff8a7908ce8b5d739d0deaa687eacfb.jpg)

<details>
<summary>histogram</summary>

| Peptide Length (residue) | Frequency |
| ------------------------ | --------- |
| 0-1                      | 50        |
| 1-2                      | 600       |
| 2-3                      | 800       |
| 3-4                      | 1000      |
| 4-5                      | 1100      |
| 5-6                      | 1200      |
| 6-7                      | 1300      |
| 7-8                      | 1400      |
| 8-9                      | 1500      |
| 9-10                     | 2200      |
| 10-11                    | 1500      |
| 11-12                    | 1300      |
| 12-13                    | 1200      |
| 13-14                    | 1000      |
| 14-15                    | 800       |
| 15-16                    | 700       |
| 16-17                    | 600       |
| 17-18                    | 500       |
| 18-19                    | 400       |
| 19-20                    | 300       |
| 20-21                    | 400       |
| 21-22                    | 900       |
| 22-23                    | 300       |
| 23-24                    | 350       |
| 24-25                    | 400       |
| 25-26                    | 450       |
| 26-27                    | 800       |
| 27-28                    | 850       |
| 28-29                    | 800       |
| 29-30                    | 850       |
</details>

![](images/1eb4a9047904d30e8c099fb54fb7b20b36e073e85ea2027f26db2a0c41bd90f3.jpg)

<details>
<summary>histogram</summary>

| Peptide Length (atom) | Frequency |
| --------------------- | --------- |
| 0-5                   | 1300      |
| 5-10                  | 3600      |
| 10-15                 | 5000      |
| 15-20                 | 3600      |
| 20-25                 | 1600      |
| 25-30                 | 1500      |
| 30-35                 | 1500      |
| 35-40                 | 900       |
| 40-45                 | 800       |
| 45-50                 | 100       |
</details>

![](images/2754c66a2664fae4235f87af2bcb7499a76b5a617df314c56db83cbef673e275.jpg)

<details>
<summary>histogram</summary>

| Cyclic Peptide Length (residue) | Frequency |
| ------------------------------- | --------- |
| 5                               | 20        |
| 7.5                             | 80        |
| 10                              | 40        |
| 12.5                            | 90        |
| 15                              | 60        |
| 17.5                            | 50        |
| 20                              | 680       |
| 22.5                            | 30        |
| 25                              | 10        |
| 27.5                            | 30        |
| 30                              | 20        |
</details>

![](images/39f91b0a3010ffb1293f1ab163f07c34c5bc706295b8d4d7894bcebfc20351d1.jpg)

<details>
<summary>histogram</summary>

| Cyclic Peptide Length (atom) | Frequency |
|---|---|
| 25-50 | 80 |
| 50-75 | 90 |
| 75-100 | 80 |
| 100-125 | 200 |
| 125-150 | 130 |
| 150-175 | 700 |
| 175-200 | 60 |
| 200-225 | 30 |
</details>

Figure 7. Statistics on peptide and cyclic peptide lengths in our dataset.   
![](images/0b3e9e1ea84b0481397ac11a27da432d9b087f8797e075d3e9e1bf68df03863a.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
| :--- | :--- |
| Linear Peptide | 93.1 |
| Cyclic Peptide | (unlabeled segment) |
| Side-to-side | 61.5 |
| Polycyclization | 15.1 |
| Head-to-tail | 13.6 |
| Side-to-tail | 6.5 |
| Head-to-side | 3.3 |
</details>

![](images/49d735b1b39f61b2cd811ef5c9f6098a4923564db444f78696f848112d32358e.jpg)

<details>
<summary>bar</summary>

| Category | Frequency |
| :--- | :--- |
| head-to-tail | 200 |
| side-to-tail | 95 |
| head-to-side | 49 |
| side-to-side | 903 |
| policyization | 221 |
</details>

Figure 8. Analysis of cyclic peptide distribution and the proportions of the five cyclization types.

with multimodal generative protein language models.

Cyclic Peptide Design. Although computational cyclic peptide design is an innovative research area, there are already recent studies on cyclic peptide design that are significantly different from our approach (Tang et al., 2025; Rettie et al., 2024). The approach proposed by Tang et al. (2025) is a ligand-based drug design (LBDD) method that models the sequence of cyclic peptides using discrete diffusion, optimized by multiple reward functions. It does not explicitly incorporate the 3D structure of target proteins, whereas our structure-based drug design (SBDD) method directly designs ligands based on 3D target structures. Rettie et al. (2024) uses modified RoseTTAFold (Baek et al., 2021) and RFdiffusion (Watson et al., 2023) with cyclic relative positional encoding to generate macrocyclic backbones.

# C Protein all-atom representation

Atom14 representation. The atom14 encoding efficiently represents residues using 14 columns to capture atom content, with 14 being the maximum number of heavy atoms in the 20 canonical amino acids (e.g., Tryptophan). Empty strings are padded in atom14 representation when a residue contains fewer than 14 atoms.

Atom37 representation. The atom37 encoding is a fixed-dimension representation where each of the 37 heavy atom types in canonical amino acids is assigned a unique position in an array. For amino acids missing certain atoms, padding is used to maintain a consistent length. The atom37 format includes the following atoms: ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT'].

Atom73 representation. The atom73 representation, introduced by Chu et al. (2024), indexes the 'N', 'CA', 'C', 'CB', and 'O' atoms, then independently encodes each amino acid's side chains. Backbone atoms share the same position, while side-chain atoms are assigned to specific residue types. For example, the 'NE2' atom in the 'GLN' residue is recorded as 'Q-NE2' (where Q is the one-letter code for 'GLN') and occupies a unique column in the atom73 encoding.

# D Peptide Dataset

CyclicPepedia. The CyclicPepedia dataset (Liu et al., 2024) contains 9,744 cyclic peptides from multiple sources. It serves as a knowledge database with information on categorization, structural characteristics, pharmacokinetics, physicochemical properties, patented drug applications, and key publications. However, most peptide conformations are generated by RDKit, with only 1,325 cyclic peptides having original 3D structures and 61 containing complex structures. The remaining data only include fingerprints, limiting its application in structure-based drug design.

PPBench2024. PPBench2024 (Lin et al., 2025) is a protein-peptide binding dataset sourced from Burley et al. (2023), Martins et al. (2023), and Weng et al. (2020). It selects complexes with more than two chains and an interaction distance

under 5 Å. Complexes with peptides shorter than 30 amino acids are then filtered. Non-peptide molecules and peptides with unusual bond lengths or non-amino acid functional groups are also excluded. This results in a total of 15,593 protein-peptide pairs.

PepBench. PepBench is a curated benchmarking dataset designed to train and evaluate protein-peptide binding models in PepGLAD (Kong et al., 2024). The training data is sourced from (Berman et al., 2000), while the test set is adopted from Tsaban et al. (2022). To ensure quality, peptides are restricted to lengths between 4 and 25 residues, and receptors to more than 30 residues. Complexes with over 90% sequence similarity are excluded to reduce redundancy. Furthermore, clustering at 40% sequence identity is applied to separate training and test data, removing any training complexes that share clusters with the test set. The final dataset comprises 4,157 training complexes, 114 for validation, and 93 for testing.

PepFlow. The dataset introduced by Li et al. (2025) combines data from Wen et al. (2019) and Wei et al. (2024), resulting in 8,365 protein-peptide complex structures. Peptides are filtered to include those with lengths ranging from 3 to 25 amino acids and are clustered at 40% sequence identity. From these clusters, 158 structures are selected for the test set, ensuring each cluster has 10 to 50 members. The remaining data is used for training and validation.

Our curation. Our curation has been roughly introduced in Section 4.1. A unique requirement of our method is converting Structured Data Files (SDF files) to PDB files, as the chemical bond information in PDB format is often incomplete. We found that RDKit $^{2}$ does not always accurately produce chemical bonds during conversion, as it determines bond existence and type based on pairwise atom types and distances. Therefore, we leverage the Chemical Component Dictionary (CCD) $^{3}$ , which describes all residue and small molecule components found in PDB entries, to collect all intra-residue bond information. We use default peptide bonds to connect canonical residues with continuous residue indices and the bonds stored in the CONECT information to recover bonds between non-canonical residues and other residues.

# E Routed Sampling

The general idea of the proposed routed sampling is that the sequence and structure are alternately updated in the generative process. We present its details in Algorithm 1. In this subsection, we introduce some notations without rigorous definitions but maintain clarity, as we provide detailed comments after the lines in the algorithm. The sampling algorithm is inspired by Chu et al. (2024). However, we innovatively introduce a dynamic chemical graph to make the all-atom peptide design compatible with cyclic structures, especially non-canonical covalent bonds and residues. Several critical functions used in Algorithm 1 will be discussed as follows:

- “Extract” and “Cache”: As introduced in Section 3.4, due to cyclization, we can categorize atoms into two types: constrained atoms and free-residue atoms. We maintain the atom73 states and individual atom states for these types, respectively. Two functions are employed to extract and store atom coordinates based on masks determined by residue types or constrained atom indices.   
- “Assemble”: Each time the residue types are updated, the corresponding side-chain chemical graphs are likewise updated. Consequently, we reassemble each residue’s chemical graph along with the cyclization chemical graph into a complete peptide chemical graph to serve as input for the models.   
- “Supgraph”: As introduced in Section 3.3, to avoid residue type information leakage from the side-chain chemical graphs, we remove the side-chain atoms within the free residues from the current peptide chemical graph.   
- “UpdateTime” and “AlignTime”: As mentioned in Section 3.4, due to the sampling mechanism, updates of side-chain atoms within free residues might not be continuous. In other words, the coordinates of these atoms are occasionally updated, resulting in atoms having different times (or noise levels). Therefore, “UpdateTime” is introduced to store the individual time, and “AlignTime” is introduced to align the atom coordinates from different times to the same point. It’s important to note that in “AlignTime”, no model is involved as the previously cached denoised structures are reused.

# F Proof of Prior Distribution Induced by Harmonic SDE

The difference between the widely-used SDE in SDE generative models and our introduced harmonic SDE is that the our perturbation process is anisotropic. Hence, here we provide the derivation of the prior distribution induced by the harmonic SDE in a similar proof by Song et al. (2021).

In Equation (3), we define the $\mathcal{G}_C$ -dependent forward SDE as follows:

$$
\mathrm{d} \mathbf {x} ^ {L} = - \frac {1}{2} \beta (t) \mathbf {x} ^ {L} \mathrm{d} t + \sqrt {\beta (t)} \boldsymbol {\Lambda} ^ {\frac {1}{2}} \mathbf {P} ^ {\intercal} \mathrm{d} \mathbf {w},
$$

Algorithm 1 Routed Sampling   
Input: number of residues within the ligand peptide N, cyclization Type O, 3D receptor structures T, SDE solver time interval dt, infinitesimal constant ε

Output: cyclic peptide with its all-atom coordinates $x_{0}$ , chemical graph $G_{0}$ , amino acid sequence $A_{0}$ 1: $[X_{1}, x_{1}^{O}] \leftarrow \text{HarmonicPrior}(N, O, \mathcal{T})$ ▷ Initialize time-dependent atom73 state $X_{1}$ and cyclization state $x_{1}^{O}$ 2: $\widetilde{x}^{O} \leftarrow \text{Copy}(x_{1}^{O})$ ▷ Initialize denoised cyclization-related atom coordinates $\widetilde{x}^{O}$ 3: $A_{1} \leftarrow \text{Uniform}(20, N)$ ▷ Randomly initialize residue types not constrained by cyclization

4: $G_{1} \leftarrow \text{Assemble}(A_{1}, O)$ ▷ Derive initial chemical graph

5: $T \leftarrow 1$ ▷ Initialize a timer that records time for each atom in atom73 state

6: $t \leftarrow 1$ 7: while $t > \epsilon$ do

8: $x_{t} \leftarrow \text{Extract}(X_{t}, A_{t}) \cup x_{t}^{O}$ ▷ Obtain all-atom $x_{t}$ structure of current noisy peptide

9: $\widehat{x}_{0} \leftarrow \text{ATOMSDE}(x_{t}, G_{t}, t)$ ▷ Predict denoised all-atom structure $\widehat{x}_{0}$ structure

10: $\widetilde{x}^{O} \leftarrow \text{Cache}(\widetilde{x}^{O}, \widehat{x}_{0}, O)$ 11: $x_{t-dt} \leftarrow \text{Noise}(\widehat{x}_{0}, G_{t}, t-dt)$ ▷ $G_{t}$ is required by harmonic noise

12: $X_{t-dt} \leftarrow \text{Cache}(X_{t}, x_{t-dt}, A_{t})$ ▷ Update $X_{t}$ by saving new structures to specific states according to $A_{t}$ 13: $x_{t-dt}^{O} \leftarrow \text{Cache}(x_{t}^{O}, x_{t-dt}, O)$ 14: $T \leftarrow \text{UpdateTimer}(T, A_{t}, t-dt)$ ▷ Update the timer for the newly-updated atoms to the latest time

15: $\widetilde{G} \leftarrow \text{Subgraph}(G_{t}, A_{t}, O)$ ▷ Hide side chains of residues not constrained by cyclization

16: $A_{t-dt} \leftarrow \text{RESROUTER}(\widehat{x}_{0}, \widetilde{G}, t)$ ▷ Predict sequence based on the denoised structure $\widehat{x}_{0}$ 17: $G_{t-dt} \leftarrow \text{Assemble}(A_{t-dt}, O)$ ▷ Derive a new chemical graph given predicted sequence and cyclization

18: $t \leftarrow \text{Extract}(T, A_{t-dt})$ ▷ Obtain atom-wise time t (Atom might have different time)

19: $x_{t-dt} \leftarrow \text{Extract}(X_{t-dt}, A_{t-dt}) \cup x_{t-dt}^{O}$ ▷ Align atoms with different time t to the same time t-dt

20: $x_{t-dt} \leftarrow \text{AlignTime}(x_{t-dt}, \widehat{x}_{0}, G_{t-dt}, t-dt, t)$ according to the reverse-time harmonic SDE

21: $X_{t-dt} \leftarrow \text{Cache}(X_{t-dt}, x_{t-dt}, A_{t-dt})$ ▷ Store the time-aligned atom coordinates

22: $t \leftarrow t - dt$ 23: end while

24: $x_{t} \leftarrow \text{Extract}(X_{t}, A_{t}) \cup x_{t}^{O}$ 25: $x_{0} \leftarrow \text{ATOMSDE}(x_{t}, G_{t}, t)$ ▷ Predict the all-atom structure finally

26: $G_{0} \leftarrow G_{t}$ 27: $A_{0} \leftarrow A_{t}$

where $\beta(t)$ is a positive time-dependent scalar function, P is an orthogonal matrix (i.e., $PP^{\intercal} = I$ ), $\Lambda = \text{diag}(\lambda_{1}, \ldots, \lambda_{N_{L}})$ is a diagonal matrix that contains the eigenvalues, and $H = P\Lambda P^{\intercal}$ .

We denote the variance of the random variable $\mathbf{x}^{L}$ as $\boldsymbol{\Sigma}(t)$ , i.e., $\boldsymbol{\Sigma}(t) := \operatorname{Cov}[\mathbf{x}(t)]$ for $t \in [0, 1]$ . The aforementioned SDE, characterized by affine drift and diffusion coefficients, allows us to employ Eq. (5.51) from Särkkä & Solin (2019) to derive an ODE that describes the evolution of variance as follows:

$$
\begin{array}{l} \frac {\mathrm{d} \boldsymbol {\Sigma}}{\mathrm{d} t} = \beta (t) \left(\left(\boldsymbol {\Lambda} ^ {\frac {1}{2}} \mathbf {P} ^ {\intercal}\right) ^ {\intercal} \boldsymbol {\Lambda} ^ {\frac {1}{2}} \mathbf {P} ^ {\intercal} - \boldsymbol {\Sigma} (t)\right), \\ = \beta (t) (\mathbf {H} - \boldsymbol {\Sigma} (t)). \\ \end{array}
$$

Solving the above ODE, we derive

$$
\boldsymbol {\Sigma} (t) = \mathbf {H} + e ^ {\int_ {0} ^ {t} - \beta (s) \mathrm{d} s} (\boldsymbol {\Sigma} (0) - \mathbf {H}),
$$

Once the boundary condition $x_{0}^{L}$ is given, we have $\Sigma(0)=0$ . Thus, the induced perturbation kernel has an analytic form as:

$$
p _ {0 t} (\mathbf {x} _ {t} ^ {L} | \mathbf {x} _ {0} ^ {L}) = \mathcal {N} (\mathbf {x} _ {t} ^ {L}; \mathbf {x} _ {0} ^ {L} e ^ {- \frac {1}{2} \int_ {0} ^ {t} \beta (s) \mathrm{d} s}, \mathbf {H} - \mathbf {H} e ^ {- \int_ {0} ^ {t} \beta (s) \mathrm{d} s}).
$$

Given $\lim_{t\to 1}\int_0^t\beta (s)\mathrm{d}s = \infty$ , the above perturbation process arrives at the prior distribution $p_1(\mathbf{x}_1^L) = \mathcal{N}(\mathbf{x}_1^L;\mathbf{0},\mathbf{H})$

# G Implementation Details

# G.1 Model Architecture

Given a noisy sample at time t, two graphs are built for message passing with an SE(3)-equivariant neural network, which is parameterized by $\phi_{K}, \phi_{B}, \phi_{C}, \phi_{H}, \phi_{E}, \psi_{K}, \psi_{C}$ as introduced below. The i-th atom in the complex is attributed with an initial feature $h_{i}$ and the bond ij in the noisy ligand is attributed with an initial feature $b_{ij}$ . We first construct a k-nearest neighbor (knn) graph $G_{K}$ for the complex (i.e., the protein and the noisy ligand at time t), where each ligand atom is connected with the k-nearest atoms in the complex, to capture the protein-ligand interaction:

$$
\Delta \mathbf {h} _ {i} ^ {K} \leftarrow \sum_ {j \in \mathcal {N} _ {K} (i)} \phi_ {K} (\mathbf {h} _ {i}, \mathbf {h} _ {j}, \| \mathbf {x} _ {i} - \mathbf {x} _ {j} \|, E _ {i j}, t),
$$

where $\mathcal{N}_K(i)$ is the neighbors of atom $i$ in $\mathcal{G}_K$ , $E_{ij}$ indicates the edge $ij$ is a protein-protein, ligand-ligand or protein-ligand edge.

We also leverage the chemical graph $G_{C}$ of the ligand as we have defined previously to make the model aware of the connection information introduced by the chemical bonds:

$$
\begin{array}{l} \mathbf {e} _ {i j} \leftarrow \phi_ {B} (\| \mathbf {x} _ {i} - \mathbf {x} _ {j}, \mathbf {b} _ {i j} \|), \\ \mathbf {h} _ {i} ^ {C} \leftarrow \sum_ {j \in \mathcal {N} _ {C} (i) \phi_ {C}} (\mathbf {h} _ {i}, \mathbf {h} _ {j}, \mathbf {e} _ {i j}, t). \\ \end{array}
$$

We further aggregate the hidden features of ligand atoms and bonds from these two graphs as follows:

$$
\begin{array}{l} \mathbf {h} _ {i} \leftarrow \mathbf {h} _ {i} + \phi_ {H} (\Delta \mathbf {h} _ {i} ^ {K} + \Delta \mathbf {h} _ {i} ^ {C}), \\ \mathbf {b} _ {i j} \leftarrow \sum_ {k \in \mathcal {N} _ {C} (j) \backslash \{i \}} \phi_ {B} (\mathbf {h} _ {i}, \mathbf {h} _ {j}, \mathbf {h} _ {k}, \mathbf {e} _ {i k}, \mathbf {e} _ {k j}, t). \\ \end{array}
$$

Finally, we update the ligand atom positions as follows:

$$
\begin{array}{l} \Delta \mathbf {x} _ {i} ^ {K} \leftarrow \sum_ {j \in \mathcal {N} _ {K} (i)} (\mathbf {x} _ {j} - \mathbf {x} _ {i}) \psi_ {K} (\mathbf {h} _ {i}, \mathbf {h} _ {j}, \| \mathbf {x} _ {i} - \mathbf {x} _ {j} \|, t), \\ \Delta \mathbf {x} _ {i} ^ {C} \leftarrow \sum_ {j \in \mathcal {N} _ {C} (i)} (\mathbf {x} _ {j} - \mathbf {x} _ {i}) \psi_ {K} (\mathbf {h} _ {i}, \mathbf {h} _ {j}, \| \mathbf {x} _ {i} - \mathbf {x} _ {j} \|, \mathbf {e} _ {i j}, t), \\ \mathbf {x} _ {i} \leftarrow \mathbf {x} _ {i} + (\Delta \mathbf {x} _ {i} ^ {K} + \Delta \mathbf {x} _ {i} ^ {C}) \cdot \mathbb {1} \{i \in \mathcal {G} _ {C} \}, \\ \end{array}
$$

where $\mathbb{1}\{i\in \mathcal{G}_C\}$ indicates whether atom $i$ belongs to the ligand since the protein atom positions are fixed and we only update ligand atom positions.

We denote the final output of the SE(3)-equivariant neural network as $D_{\theta}(\mathbf{x}_{t}^{L}, t)$ , where $D_{\theta}$ is composed of $\phi_{K}, \phi_{B}, \phi_{C}, \phi_{H}, \phi_{E}, \psi_{K}, \psi_{C}$ as introduced above.

# G.2 Training Details

We use the same optimizer setting for both ATOMSDE and RESROUTER: AdamW (Loshchilov, 2017) optimizer with constant learning rate 0.0001, beta1 0.9, beta2 0.999, and weight decay 0.01. For beta schedule, we use $\beta(t) = (\beta_{\mathrm{max}} - \beta_{\mathrm{min}})t + \beta_{\mathrm{min}}$ , where $\beta_{\mathrm{min}} = 0.01$ and $\beta_{\mathrm{max}} = 3.0$ . Note that $\lim_{t \to 1} \int_0^s \beta(s)\mathrm{d}s$ is sufficiently large compared to the variance of our data distribution. To train ATOMSDE, we sample $t \sim \mathcal{U}[0,1]$ . To train RESROUTER, we sample $t \sim \mathcal{U}[0,0.5]$ due to the fact that the denoised structure output by trained model ATOMSDE at time $t = 0.5$ or more has extremely limited information to determine the residue types. ATOMSDE converges within 48 hours and RESROUTER converges within 18 hours on 8 NVIDIA H100 GPUs.

# G.3 Sampling Details

For routed sampling, we divide time interval $[0,1]$ into 1,000 steps. Inspired by Chu et al. (2024), we skip RESROUTER when t > 0.5 since the structures are too noisy to provide sufficient information for residue type prediction. This approach also accelerates the generative process and reduces the computational cost of inference. When t < 0.5, at each step, sequence and structures are iteratively updated by RESROUTER and ATOMSDE, respectively.

# H Experimental Details

# H.1 Relaxation and Energy Estimation

Cyclic peptides offer notable advantages in terms of both system stability and binding affinity. In specific, the stability of a protein-peptide complex is inversely proportional to its overall free energy, with lower free energy indicating greater stability. To assess this, the FastRelax protocol in PyRosetta (Chaudhury et al., 2010) is employed to relax each complex, after which the total energy is evaluated using the REF2015 scoring function. Binding affinity is measured with the InterfaceAnalyzerMover, which calculates the binding energy at the interface between the peptide and the target protein within the relaxed complex. An increase in binding energy reflects enhanced peptide binding affinity, suggesting potential functional improvements.

For each target, linear peptide methods generate 8 samples with the golden peptide length (i.e., the number of residues within reference linear peptide). Cyclic peptide methods, lacking a reference length, enumerate residues from 5 to 20 (or 8 to 23 for side-to-side cyclic peptides), generating 2 samples per length. All reference ligands and samples are relaxed and evaluated using a standard scoring method as described above. For each target and method, we apply the Borda method to select the best ligand, accounting for both stability and affinity, which is then reported in the final results.

# H.2 Detailed Experimental Results

We have provided detailed energy measurement results for each target, including the reference ligand, linear peptides designed by baselines, and cyclic peptides designed by our methods, in Tables 6 and 7.

# H.3 Inference Speed

We benchmark the average time of generating one peptide for all co-design baselines and our methods on a single NVIDIA A100-SXM4-80GB GPU. The results are reported in Table 2. Given that computational drug design does not demand real-time model response, the inference time of our method is deemed acceptable.

Table 2. Generation time of all co-design baselines and our methods. 

<table><tr><td>Method</td><td>Peptide Type</td><td>Time (s)</td></tr><tr><td>ProteinGenerator</td><td>Linear</td><td>31.80</td></tr><tr><td>PepFlow</td><td>Linear</td><td>12.09</td></tr><tr><td>PepGLAD</td><td>Linear</td><td>4.40</td></tr><tr><td>CPSDE</td><td>Cyclic</td><td>16.88</td></tr></table>

# H.4 Evaluation on Linear Peptide Design

While designing linear peptides is not our primary focus, we compare our method with existing baselines in this task.

Unlike the variable-length setting for cyclic peptides, the task of linear peptide design can leverage known reference peptide lengths for target proteins in the test set. For fair comparison across methods, we sample 8 linear peptides per target matching the reference length and relax the complex structure by Rosetta (Chaudhury et al., 2010; Alford et al., 2017). We then apply the Borda method to choose the optimal linear peptides, considering both stability and affinity. We report the average and median for Stability and Affinity of the linear peptides for targets in the test set. We also report the average for the fraction of hydrophobic and charged residues (relevant for specificity) (Ye et al., 2024), DockQ, iRMSD, LRMSD, BSR, and Diversity.

We use DockQ package $^{4}$ to compute DockQ, iRMSD, and LRMSD. We follow the definition of binding site ratio (BSR) in Li et al. (2025). A lower hydrophobic/charged ratio indicates a lower risk of non-specific binding (Makowski et al., 2024). The results are reported in Table 3. Notably, the fraction of hydrophobic and charged residues of our designed peptides resembles that of reference. Our method also shows superiority in structural properties.

Table 3. Summary of properties of reference peptides, linear peptides designed by baseline methods and CPSDE. (↓) / (↑) denotes a smaller / larger number is better. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Co-Design</td><td rowspan="2">Peptide Type</td><td colspan="2">Stability (↓)</td><td colspan="2">Affinity (↓)</td><td rowspan="2">Hydrophobic Ratio (↓)</td><td rowspan="2">Charged Ratio (↓)</td><td rowspan="2">DockQ (↑)</td><td rowspan="2">iRMSD (↓)</td><td rowspan="2">LRMSD (↓)</td><td rowspan="2">BSR (↑)</td><td rowspan="2">Diversity (↑)</td></tr><tr><td>Avg.</td><td>Med.</td><td>Avg.</td><td>Med.</td></tr><tr><td>Reference</td><td>N/A</td><td>Linear</td><td>-672.53</td><td>-634.71</td><td>-85.03</td><td>-78.70</td><td>0.48</td><td>0.28</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>RFDiffusion</td><td>✕</td><td>Linear</td><td>-633.51</td><td>-607.82</td><td>-70.30</td><td>-61.35</td><td>0.59</td><td>0.27</td><td>0.18</td><td>5.37</td><td>20.10</td><td>0.33</td><td>0.55</td></tr><tr><td>ProteinGenerator</td><td>✓</td><td>Linear</td><td>-576.39</td><td>-554.70</td><td>-46.98</td><td>-40.39</td><td>0.53</td><td>0.32</td><td>0.12</td><td>5.56</td><td>23.97</td><td>0.20</td><td>0.58</td></tr><tr><td>PepFlow</td><td>✓</td><td>Linear</td><td>-576.16</td><td>-498.31</td><td>-47.88</td><td>-42.40</td><td>0.60</td><td>0.17</td><td>0.44</td><td>2.49</td><td>9.42</td><td>0.56</td><td>0.70</td></tr><tr><td>PepGLAD</td><td>✓</td><td>Linear</td><td>-359.44</td><td>-310.33</td><td>-45.06</td><td>-38.56</td><td>0.53</td><td>0.25</td><td>0.30</td><td>2.68</td><td>11.99</td><td>0.39</td><td>0.79</td></tr><tr><td>CpSDE</td><td>✓</td><td>Linear</td><td>-567.34</td><td>-510.58</td><td>-55.48</td><td>-49.89</td><td>0.45</td><td>0.24</td><td>0.32</td><td>2.36</td><td>9.91</td><td>0.60</td><td>0.77</td></tr></table>

# H.5 Ablation Studies

Effects of RESROUTER. We study the effects of RESROUTER compared with the following two setups: “w/ fix seq” where the residue types are randomly sampled and fixed with only atom coordinates updated during the generative process, “w/ random seq” where the residue types are randomly sampled from a uniform distribution instead of predicted by RESROUTER during the generative process. The results are shown in Table 4. It can be observed that both variants perform worse than CPSDE, which demonstrates that RESROUTER can effectively discover critical residue types for protein-ligand interaction. “w/ random seq” performs the worst, possibly because random residue types offer no information gain, and the residue types are updated too frequently. This frequent updating hinders the ATOMSDE from effectively updating the side-chain atoms of the free residues.

Table 4. Ablation study on the effect of RESROUTER. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Stability (↓)</td><td colspan="2">Affinity (↓)</td></tr><tr><td>Avg.</td><td>Med.</td><td>Avg.</td><td>Med.</td></tr><tr><td>CPSDE</td><td>-568.04</td><td>-519.66</td><td>-50.86</td><td>-46.62</td></tr><tr><td>w/ fix seq</td><td>-525.73</td><td>-439.69</td><td>-41.56</td><td>-38.75</td></tr><tr><td>w/ random seq</td><td>-521.48</td><td>-425.17</td><td>-38.64</td><td>-39.44</td></tr></table>

Effects of Harmonic SDE. We study the effects of harmonic SDE. We introduce a variant with isotropic Gaussian as prior and noise distribution (denoted as “w/o Harmonic”). The results are shown in Table 4 and validate the effectiveness of harmonic prior and noise. To explore the underlying reasons, we examined the trajectories of the generative process and found that the harmonic prior provides a good initialization for atom positions, where bonded atoms are located nearby. This feature might be beneficial for routed sampling because some side-chain atom updates can be discontinuous, and such correlated initialization helps mitigate the errors induced by these discontinuous updates.

Table 5. Ablation study on the effect of Harmonic SDE. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Stability (↓)</td><td colspan="2">Affinity (↓)</td></tr><tr><td>Avg.</td><td>Med.</td><td>Avg.</td><td>Med.</td></tr><tr><td>CPSDE</td><td>-568.04</td><td>-519.66</td><td>-50.86</td><td>-46.62</td></tr><tr><td>w/o Harmonic</td><td>-534.38</td><td>-439.23</td><td>-39.43</td><td>-41.08</td></tr></table>

# H.6 System Setup and Protocols of Molecular Dynamics Simulation

To simulate the protein-peptide systems, hydrogen atoms are added, and the dominant protonation state of titratable residues at pH 7 is determined using PropKa in PDB2PQR (Dolinsky et al., 2007). Subsequently, the systems are solvated in a 10 Å truncated water box, with sodium and chloride ions added to neutralize the system at a concentration of 150 mM to mimic physiological saline. The ff14SB (Maier et al., 2015) parameter set is applied to proteins and peptides, and the TIP3P model is used for water (Jorgensen et al., 1983; Li et al., 2024).

All simulations were run on RTX 4090 GPUs using the CUDA implementation of particle-mesh Ewald (PME) molecular

dynamics in Amber22 (Salomon-Ferrer et al., 2013). At first, to relax each system thoroughly, two stages of energy minimization are performed. In the first stage, 2,500 steepest descent and 2,500 conjugate gradient cycles were applied to all atoms, with constraints on water molecules and counterions. In the second stage, the same cycles were repeated without constraints. Initial velocities are randomly sampled from a Boltzmann distribution. The systems are then heated from 0 K to 310 K over 500 ps in the NVT ensemble, using a Langevin thermostat and harmonic restraints of $10.0 \, kcal \cdot mol^{-1} \cdot \mathring{A}^{-2}$ . During equilibration at 300 K and 1 bar under NPT conditions, harmonic restraints on protein and peptide atoms were progressively reduced from 5.0 to $0.1 \, kcal \cdot mol^{-1} \cdot \mathring{A}^{-2}$ in four steps at 0.5 ns intervals, totaling 2.5 ns. All restraints are completely removed during production simulation under 310K and 1 bar, which are maintained using the Langevin thermostat and Berendsen barostat, respectively. A timestep of 4.0 fs is used with hydrogen mass repartitioning (Hopkins et al., 2015). Bond lengths are constrained via SHAKE (Ryckaert et al., 1977), and non-bonded interactions are cut off at 10 Å.

# H.7 Visualization of Structure Ensembles Simulated by Molecular Dynamics

We present additional views of the structure ensembles generated by molecular dynamics simulations in Figure 12 and Figure 13.

# I Examples of Designed Cyclic Peptides

Here, we present more compelling results of our generated cyclic peptides targeting different receptors in Figures 9, 14 and 15. We find that our generated cyclic peptides consistently exhibit higher or competitive affinities with greater interaction stabilities when binding to the receptor. In contrast, linear peptides sampled from PepFlow often result in unstable structures and weaker binding. Furthermore, our designed cyclic peptides not only interact with key receptor regions, similar to linear peptides and native peptides, but also establish new, stable, and tight interactions in additional regions. Additionally, our generated 3D cyclic peptides consistently align well with the corresponding 2D chemical graphs, highlighting the effective integration of our two models.

PDB: 7mhz   
![](images/bc6467c37d3bdb0e22d096f0d00c1541c3d75f678670e323ffc2e037d07a0700.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface visualization with a yellow highlighted region, showing no text or symbols
</details>

Affinity: -31.9   
Stability: -187.5

PepFlow   
![](images/af8165af0413298ac01cd4a8a6957242ad21aa973d4dff12cb5e13826dec647b.jpg)

<details>
<summary>chemical</summary>

Molecular interaction diagram showing hydrogen bonding between a ligand and a protein surface
</details>

Affinity: -29.8   
Stability: -148.7

Our (head-to-tail)   
![](images/52b27cb6c1f8aec8732413cc3000a87d29d441cb62e8f120ba8cbf196c1823fc.jpg)

<details>
<summary>chemical</summary>

Molecular structure visualization showing a complex organic compound with red and blue atoms in a 3D representation
</details>

Affinity: -59.5   
Stability: -217.9

Our (chemical graph)   
![](images/5f73e69045ec9580150ab6d612651388db477ec1d4d24a5f9caee8581dc72f1a.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and side chains
</details>

PDB: 1d8d   
![](images/e0838f8b59687b0681749be43d602125aeb92476e4f5f911a6e2f9026b07fe6f.jpg)

<details>
<summary>natural_image</summary>

Molecular surface visualization with yellow and red secondary structures (no text or labels)
</details>

Affinity: -45.1   
Stability: -1131.2

PepFlow   
![](images/bb82a9c7fa290a237a9500bdc85102ac9b3d2f9f5ced9d25fca841f7268872df.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing protein backbone with alpha-helices and beta-sheets, labeled with red and blue atoms
</details>

Affinity: -28.9   
Stability: -1049.4

Our (head-to-side)   
![](images/b130f99d2754257d9f7be3b45c71a49999772f2a73898c7752309e963def1716.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model showing protein secondary structure (no text or labels)
</details>

Affinity: -44.1   
Stability: -1091.3

Our (chemical graph)   
![](images/fccbc70bb88b4434ae46cc74769b239fe635a24120977620402318b68a08c482.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple rings and functional groups
</details>

Figure 9. Visualization of reference ligand, linear peptides designed by PepFlow, and cyclic peptides designed by CPSDE.

# J Limitations and Future Work

One limitation is that the generated cyclic peptides may sometimes exhibit invalid conformations, such as inaccurate bond lengths and atomic receptor clashes. While Rosetta-based structure relaxation (Chaudhury et al., 2010; Alford et al., 2017) can refine these structures, it is computationally expensive and slow. Additionally, for evaluation, we are currently unable to introduce self-consistency metrics similar to those in protein design (Yim et al., 2024), as there is no highly accurate cyclic peptide structure prediction or docking model available.

Table 6. Stability and affinity of the reference peptide, linear peptides designed by baseline methods, and cyclic peptide designed by our method along with the cyclization type. 

<table><tr><td rowspan="2">Target</td><td colspan="2">Reference</td><td colspan="2">RFDiffusion</td><td colspan="2">ProteinGenerator</td><td colspan="2">PepFlow</td><td colspan="2">PepGLAD</td><td colspan="3">Our</td></tr><tr><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Type</td></tr><tr><td>1d8d</td><td>-1131.2</td><td>-45.1</td><td>-1092.2</td><td>-37.2</td><td>-975.5</td><td>-43.7</td><td>-1049.4</td><td>-28.9</td><td>-1070.0</td><td>-30.1</td><td>-1091.3</td><td>-44.1</td><td>h2s</td></tr><tr><td>1hr8</td><td>-417.7</td><td>-48.4</td><td>-363.9</td><td>-78.7</td><td>-370.0</td><td>-39.3</td><td>-428.8</td><td>-36.8</td><td>-207.8</td><td>-28.5</td><td>-313.1</td><td>-37.2</td><td>s2t</td></tr><tr><td>1rgq</td><td>-374.0</td><td>-140.2</td><td>-390.1</td><td>-153.6</td><td>-289.1</td><td>-121.1</td><td>-167.5</td><td>-69.3</td><td>82.6</td><td>-67.2</td><td>-255.1</td><td>-83.5</td><td>s2t</td></tr><tr><td>1vzj</td><td>-80.1</td><td>-138.7</td><td>282.8</td><td>-66.3</td><td>356.1</td><td>-52.8</td><td>12.4</td><td>-112.2</td><td>20.4</td><td>-90.6</td><td>143.4</td><td>-105.3</td><td>h2s</td></tr><tr><td>1xoc</td><td>-899.3</td><td>-66.9</td><td>-763.8</td><td>-40.7</td><td>-866.0</td><td>-41.2</td><td>-886.2</td><td>-51.1</td><td>-778.8</td><td>-30.8</td><td>-867.3</td><td>-47.7</td><td>h2s</td></tr><tr><td>1zkk</td><td>-824.2</td><td>-47.0</td><td>-843.0</td><td>-59.0</td><td>-824.0</td><td>-40.2</td><td>-797.3</td><td>-30.0</td><td>-439.0</td><td>-35.5</td><td>-793.2</td><td>-62.9</td><td>h2t</td></tr><tr><td>2arq</td><td>-556.3</td><td>-148.5</td><td>-588.7</td><td>-165.9</td><td>-428.5</td><td>-29.2</td><td>-428.2</td><td>-63.5</td><td>-204.1</td><td>-53.1</td><td>-438.3</td><td>-91.5</td><td>h2s</td></tr><tr><td>2mpz</td><td>-1136.2</td><td>-223.1</td><td>-1114.8</td><td>-203.7</td><td>-1031.2</td><td>-91.4</td><td>-937.5</td><td>-84.0</td><td>491.4</td><td>-79.3</td><td>-1059.0</td><td>-244.1</td><td>h2s</td></tr><tr><td>2vda</td><td>223.6</td><td>-59.6</td><td>336.7</td><td>-33.5</td><td>393.3</td><td>-32.4</td><td>343.5</td><td>-35.1</td><td>960.0</td><td>-47.8</td><td>373.1</td><td>-25.9</td><td>h2t</td></tr><tr><td>2wqj</td><td>-576.8</td><td>-123.3</td><td>-571.3</td><td>-125.7</td><td>N/A</td><td>N/A</td><td>-447.3</td><td>-71.8</td><td>126.2</td><td>-76.0</td><td>-386.2</td><td>-48.8</td><td>h2t</td></tr><tr><td>2xjz</td><td>95.0</td><td>-164.0</td><td>234.9</td><td>-102.2</td><td>343.4</td><td>-58.1</td><td>251.8</td><td>-72.0</td><td>N/A</td><td>N/A</td><td>375.2</td><td>-46.6</td><td>h2t</td></tr><tr><td>3e8e</td><td>-1144.4</td><td>-53.3</td><td>-1123.9</td><td>-50.8</td><td>-1119.1</td><td>-39.1</td><td>-1020.6</td><td>-20.3</td><td>-551.5</td><td>-27.0</td><td>-1081.7</td><td>-37.4</td><td>h2t</td></tr><tr><td>3ech</td><td>-569.4</td><td>-101.1</td><td>-541.2</td><td>-78.8</td><td>-476.6</td><td>-84.8</td><td>-446.8</td><td>-44.0</td><td>-231.2</td><td>-65.0</td><td>-436.8</td><td>-56.0</td><td>h2t</td></tr><tr><td>3ewf</td><td>-1453.3</td><td>-36.4</td><td>-1425.1</td><td>-16.9</td><td>-1433.9</td><td>-11.7</td><td>-1356.4</td><td>0.4</td><td>-1390.7</td><td>-11.9</td><td>-1484.1</td><td>-48.0</td><td>h2s</td></tr><tr><td>3fii</td><td>-290.1</td><td>-105.2</td><td>-197.1</td><td>-88.8</td><td>-173.3</td><td>-34.7</td><td>-29.4</td><td>-30.5</td><td>-39.3</td><td>-100.2</td><td>-100.3</td><td>-45.9</td><td>h2t</td></tr><tr><td>3h8a</td><td>-528.3</td><td>-71.1</td><td>-538.8</td><td>-68.6</td><td>-446.2</td><td>-57.7</td><td>-400.1</td><td>-58.1</td><td>232.1</td><td>-38.3</td><td>-442.8</td><td>-45.2</td><td>h2s</td></tr><tr><td>3j89</td><td>-919.6</td><td>-118.1</td><td>-908.3</td><td>-115.3</td><td>-876.1</td><td>-84.7</td><td>-771.6</td><td>-59.9</td><td>N/A</td><td>N/A</td><td>-744.2</td><td>-56.5</td><td>h2t</td></tr><tr><td>3lk4</td><td>-263.2</td><td>-101.4</td><td>-215.7</td><td>-80.6</td><td>-176.2</td><td>-26.4</td><td>-256.9</td><td>-68.3</td><td>N/A</td><td>N/A</td><td>-223.3</td><td>-31.5</td><td>h2s</td></tr><tr><td>3mhp</td><td>-1021.0</td><td>-93.4</td><td>-948.3</td><td>-64.0</td><td>-851.9</td><td>-13.8</td><td>-903.4</td><td>-43.8</td><td>-682.8</td><td>-59.5</td><td>-849.0</td><td>-38.0</td><td>h2t</td></tr><tr><td>3o0e</td><td>-353.1</td><td>-43.0</td><td>-286.4</td><td>-55.7</td><td>-338.0</td><td>-42.9</td><td>-357.9</td><td>-33.1</td><td>-126.2</td><td>-30.8</td><td>-332.9</td><td>-30.4</td><td>s2t</td></tr><tr><td>3pl7</td><td>-215.7</td><td>-105.9</td><td>-266.8</td><td>-124.2</td><td>-257.3</td><td>-104.2</td><td>-140.4</td><td>-98.1</td><td>398.6</td><td>-98.9</td><td>-81.0</td><td>-59.1</td><td>h2s</td></tr><tr><td>3ro2</td><td>-606.5</td><td>-86.2</td><td>-376.7</td><td>-32.1</td><td>-345.0</td><td>-31.6</td><td>-539.1</td><td>-52.5</td><td>-117.2</td><td>-45.8</td><td>-378.1</td><td>-42.9</td><td>h2s</td></tr><tr><td>3ryb</td><td>-990.0</td><td>-54.9</td><td>-993.3</td><td>-59.4</td><td>-894.1</td><td>-34.5</td><td>-1003.8</td><td>-48.2</td><td>-857.7</td><td>-37.0</td><td>-942.6</td><td>-47.4</td><td>h2t</td></tr><tr><td>3twt</td><td>-965.3</td><td>-57.3</td><td>-862.1</td><td>-23.2</td><td>-851.4</td><td>-14.4</td><td>-936.2</td><td>-31.5</td><td>-782.6</td><td>-31.8</td><td>-892.3</td><td>-36.6</td><td>h2s</td></tr><tr><td>3vvs</td><td>-691.1</td><td>-55.1</td><td>-632.5</td><td>-40.1</td><td>-634.7</td><td>-49.6</td><td>-692.0</td><td>-50.3</td><td>-632.0</td><td>-38.1</td><td>-722.5</td><td>-63.7</td><td>h2s</td></tr><tr><td>3wy9</td><td>-618.2</td><td>-59.8</td><td>-522.1</td><td>-73.3</td><td>-610.9</td><td>-56.6</td><td>-523.7</td><td>-34.0</td><td>-313.7</td><td>-49.7</td><td>-550.8</td><td>-24.4</td><td>h2t</td></tr><tr><td>3zha</td><td>-404.5</td><td>-122.1</td><td>-187.8</td><td>-37.2</td><td>-136.9</td><td>-62.0</td><td>-308.1</td><td>-60.1</td><td>-451.7</td><td>-68.2</td><td>-261.4</td><td>-55.9</td><td>h2s</td></tr><tr><td>4chg</td><td>-936.0</td><td>-112.9</td><td>-790.8</td><td>-59.4</td><td>-809.6</td><td>-77.7</td><td>-790.1</td><td>-66.7</td><td>N/A</td><td>N/A</td><td>-714.0</td><td>-64.2</td><td>h2t</td></tr><tr><td>4e7v</td><td>-490.4</td><td>-125.4</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>-308.0</td><td>-73.0</td><td>53.7</td><td>-96.0</td><td>-473.1</td><td>-158.2</td><td>h2t</td></tr><tr><td>4edn</td><td>-800.6</td><td>-75.1</td><td>-815.0</td><td>-75.6</td><td>-694.9</td><td>-24.8</td><td>-760.6</td><td>-44.3</td><td>-521.4</td><td>-35.6</td><td>-711.4</td><td>-53.3</td><td>h2t</td></tr><tr><td>4hom</td><td>-1143.1</td><td>-33.5</td><td>-1108.8</td><td>-56.5</td><td>-1190.9</td><td>-22.5</td><td>-1108.6</td><td>-36.9</td><td>-1105.9</td><td>-21.9</td><td>-1184.8</td><td>-34.2</td><td>h2t</td></tr><tr><td>4jo6</td><td>-925.4</td><td>-81.9</td><td>-927.8</td><td>-72.0</td><td>-861.7</td><td>-40.6</td><td>-792.0</td><td>-32.6</td><td>-131.5</td><td>-60.4</td><td>-759.1</td><td>-45.4</td><td>s2s</td></tr><tr><td>4m1c</td><td>-156.4</td><td>-41.1</td><td>-98.9</td><td>-26.8</td><td>-16.6</td><td>-28.4</td><td>-100.1</td><td>-33.6</td><td>-74.8</td><td>-25.5</td><td>-102.3</td><td>-30.9</td><td>h2s</td></tr><tr><td>4o6f</td><td>-439.3</td><td>-49.9</td><td>-422.1</td><td>-66.9</td><td>-425.3</td><td>-49.2</td><td>-428.8</td><td>-32.3</td><td>-370.9</td><td>-21.7</td><td>-425.6</td><td>-44.1</td><td>h2t</td></tr><tr><td>4po7</td><td>-301.2</td><td>-42.9</td><td>-294.3</td><td>-65.0</td><td>-244.1</td><td>-56.2</td><td>-294.1</td><td>-25.4</td><td>-267.9</td><td>-20.9</td><td>-261.9</td><td>-24.9</td><td>h2t</td></tr><tr><td>4qae</td><td>-912.4</td><td>-94.9</td><td>-945.3</td><td>-67.4</td><td>-927.2</td><td>-46.4</td><td>-870.7</td><td>-56.3</td><td>-198.1</td><td>-55.6</td><td>-851.1</td><td>-54.3</td><td>s2s</td></tr><tr><td>4uqz</td><td>-733.4</td><td>-96.8</td><td>-652.8</td><td>-30.9</td><td>-640.8</td><td>-35.5</td><td>-557.0</td><td>-39.6</td><td>-555.3</td><td>-38.1</td><td>-628.2</td><td>-48.0</td><td>s2s</td></tr><tr><td>4wsi</td><td>-281.4</td><td>-93.5</td><td>-145.6</td><td>-58.1</td><td>-107.4</td><td>-34.6</td><td>-180.7</td><td>-30.9</td><td>-85.3</td><td>-45.6</td><td>-210.8</td><td>-32.0</td><td>h2t</td></tr><tr><td>4x3o</td><td>-495.0</td><td>-28.0</td><td>-494.6</td><td>-30.6</td><td>-420.4</td><td>-21.3</td><td>-441.6</td><td>0.0</td><td>-468.4</td><td>-16.8</td><td>-585.4</td><td>-83.4</td><td>h2t</td></tr><tr><td>4xpd</td><td>-114.1</td><td>-23.3</td><td>-62.1</td><td>-20.3</td><td>-78.3</td><td>-30.0</td><td>-27.9</td><td>-6.3</td><td>0.3</td><td>-15.6</td><td>-175.7</td><td>-48.9</td><td>h2t</td></tr><tr><td>4xtr</td><td>-660.2</td><td>-110.7</td><td>-659.0</td><td>-97.4</td><td>-662.6</td><td>-77.0</td><td>-594.0</td><td>-81.9</td><td>-372.5</td><td>-57.4</td><td>-534.9</td><td>-62.6</td><td>h2s</td></tr><tr><td>4yjl</td><td>-1013.7</td><td>-73.3</td><td>-859.8</td><td>-39.6</td><td>-814.8</td><td>-24.0</td><td>-934.8</td><td>-38.7</td><td>-745.0</td><td>-31.4</td><td>-923.3</td><td>-30.8</td><td>h2t</td></tr><tr><td>4zp3</td><td>-384.2</td><td>-102.1</td><td>-414.8</td><td>-119.2</td><td>-228.4</td><td>-42.2</td><td>-228.8</td><td>-61.0</td><td>24.9</td><td>-71.6</td><td>-214.1</td><td>-39.5</td><td>h2s</td></tr><tr><td>5apk</td><td>-367.8</td><td>-62.1</td><td>-240.0</td><td>-65.1</td><td>-146.3</td><td>-62.7</td><td>-310.0</td><td>-48.0</td><td>-269.8</td><td>-33.2</td><td>-330.1</td><td>-60.7</td><td>h2t</td></tr><tr><td>5brm</td><td>-469.6</td><td>-72.2</td><td>-379.6</td><td>-38.4</td><td>-375.3</td><td>-24.4</td><td>-405.0</td><td>-42.6</td><td>-301.6</td><td>-44.9</td><td>-414.8</td><td>-43.8</td><td>h2t</td></tr><tr><td>5c6h</td><td>-358.6</td><td>-83.3</td><td>-350.2</td><td>-87.3</td><td>-220.6</td><td>-36.3</td><td>-301.3</td><td>-92.8</td><td>-29.3</td><td>-65.9</td><td>-231.5</td><td>-46.6</td><td>h2t</td></tr><tr><td>5dhm</td><td>-589.5</td><td>-155.4</td><td>-607.8</td><td>-149.4</td><td>-451.7</td><td>-64.2</td><td>-483.1</td><td>-39.4</td><td>-123.1</td><td>-77.9</td><td>-424.9</td><td>-53.1</td><td>h2t</td></tr><tr><td>5e2q</td><td>-977.0</td><td>-65.2</td><td>-865.6</td><td>-37.4</td><td>-839.6</td><td>-28.8</td><td>-932.5</td><td>-47.5</td><td>-819.0</td><td>-31.9</td><td>-902.4</td><td>-58.5</td><td>s2s</td></tr><tr><td>5et1</td><td>-1436.7</td><td>-77.3</td><td>-1321.0</td><td>-41.4</td><td>-1391.5</td><td>-54.9</td><td>-1318.7</td><td>-32.9</td><td>-592.4</td><td>-37.1</td><td>-1313.0</td><td>-26.5</td><td>h2t</td></tr><tr><td>5iyx</td><td>-744.4</td><td>-59.5</td><td>-612.4</td><td>-7.7</td><td>-652.9</td><td>-37.8</td><td>-659.4</td><td>-34.7</td><td>-564.2</td><td>-28.3</td><td>-653.3</td><td>-38.1</td><td>h2t</td></tr><tr><td>5j3t</td><td>-718.3</td><td>-102.9</td><td>-624.8</td><td>-57.4</td><td>-586.4</td><td>-33.8</td><td>-509.8</td><td>-27.3</td><td>-211.8</td><td>-80.0</td><td>-501.0</td><td>-36.7</td><td>h2t</td></tr><tr><td>5mfg</td><td>-1215.8</td><td>-42.3</td><td>-1171.4</td><td>-28.1</td><td>-1195.7</td><td>-15.5</td><td>-1236.5</td><td>-42.2</td><td>-1049.8</td><td>-36.3</td><td>-1214.7</td><td>-40.0</td><td>h2s</td></tr><tr><td>5mjy</td><td>-1527.3</td><td>-79.4</td><td>-1538.3</td><td>-68.7</td><td>-1369.1</td><td>-48.1</td><td>-1482.5</td><td>-58.8</td><td>-1299.2</td><td>-43.8</td><td>-1425.3</td><td>-45.5</td><td>h2s</td></tr><tr><td>5n4d</td><td>-1231.2</td><td>-61.8</td><td>-1206.1</td><td>-62.6</td><td>-1189.6</td><td>-39.4</td><td>-1219.5</td><td>-73.4</td><td>-1034.4</td><td>-35.4</td><td>-1184.8</td><td>-40.9</td><td>s2s</td></tr><tr><td>5nl1</td><td>-669.8</td><td>-79.2</td><td>-677.0</td><td>-87.6</td><td>-702.1</td><td>-91.6</td><td>-572.6</td><td>-69.0</td><td>-354.6</td><td>-56.7</td><td>-532.3</td><td>-44.3</td><td>s2s</td></tr><tr><td>5txe</td><td>-955.6</td><td>-49.6</td><td>-930.2</td><td>-48.3</td><td>-892.9</td><td>-48.2</td><td>-883.6</td><td>-39.8</td><td>-860.2</td><td>-38.9</td><td>-879.2</td><td>-41.1</td><td>h2t</td></tr><tr><td>5vt9</td><td>-622.4</td><td>-128.5</td><td>-546.2</td><td>-97.0</td><td>-520.6</td><td>-72.4</td><td>-469.3</td><td>-86.0</td><td>1.8</td><td>-58.2</td><td>-438.5</td><td>-51.6</td><td>h2s</td></tr><tr><td>5wkf</td><td>258.4</td><td>-72.2</td><td>343.2</td><td>-36.4</td><td>384.2</td><td>-25.3</td><td>322.4</td><td>-42.8</td><td>408.0</td><td>-38.8</td><td>303.2</td><td>-54.4</td><td>s2t</td></tr><tr><td>5wpl</td><td>-352.8</td><td>-99.6</td><td>-380.7</td><td>-91.3</td><td>-284.4</td><td>-59.4</td><td>-216.6</td><td>-69.4</td><td>N/A</td><td>N/A</td><td>-197.3</td><td>-51.4</td><td>s2s</td></tr><tr><td>5yis</td><td>-416.4</td><td>-99.1</td><td>-291.1</td><td>-34.1</td><td>-263.1</td><td>-57.8</td><td>-265.5</td><td>-50.1</td><td>-149.5</td><td>-50.0</td><td>-302.1</td><td>-56.1</td><td>s2s</td></tr></table>

Table 7. Stability and affinity of the reference peptide, linear peptides designed by baseline methods, and cyclic peptide designed by our method along with the cyclization type. 

<table><tr><td rowspan="2">Target</td><td colspan="2">Reference</td><td colspan="2">RFDiffusion</td><td colspan="2">ProteinGenerator</td><td colspan="2">PepFlow</td><td colspan="2">PepGLAD</td><td colspan="3">Our</td></tr><tr><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Stab.</td><td>Affi.</td><td>Type</td></tr><tr><td>5zw6</td><td>-361.4</td><td>-46.9</td><td>-298.0</td><td>-14.3</td><td>-497.5</td><td>-29.3</td><td>-338.0</td><td>-41.4</td><td>-306.9</td><td>-28.5</td><td>-364.7</td><td>-51.5</td><td>h2s</td></tr><tr><td>6bli</td><td>-1269.1</td><td>-104.4</td><td>-1231.7</td><td>-62.2</td><td>-1231.4</td><td>-46.5</td><td>-1147.9</td><td>-48.4</td><td>N/A</td><td>N/A</td><td>-1094.2</td><td>-39.5</td><td>h2s</td></tr><tr><td>6cv1</td><td>-490.1</td><td>-109.1</td><td>-425.7</td><td>-131.3</td><td>-284.3</td><td>-44.2</td><td>-254.6</td><td>-38.5</td><td>N/A</td><td>N/A</td><td>-234.9</td><td>-36.1</td><td>h2t</td></tr><tr><td>6di8</td><td>-1850.6</td><td>-73.9</td><td>-1805.2</td><td>-49.3</td><td>-1795.9</td><td>-44.0</td><td>-1783.6</td><td>-52.4</td><td>-1364.2</td><td>-37.5</td><td>-1793.5</td><td>-59.9</td><td>h2t</td></tr><tr><td>6dtg</td><td>-874.6</td><td>-60.3</td><td>-792.4</td><td>-19.5</td><td>-698.6</td><td>-27.9</td><td>-847.4</td><td>-32.4</td><td>-783.4</td><td>-22.8</td><td>-898.9</td><td>-52.4</td><td>h2s</td></tr><tr><td>6f0h</td><td>-665.8</td><td>-80.0</td><td>-499.4</td><td>-57.0</td><td>-544.4</td><td>-46.0</td><td>-529.0</td><td>-32.7</td><td>-280.6</td><td>-54.8</td><td>-539.4</td><td>-38.9</td><td>h2s</td></tr><tr><td>6f6d</td><td>-771.9</td><td>-88.3</td><td>-666.2</td><td>-47.1</td><td>-653.4</td><td>-63.8</td><td>-674.9</td><td>-39.8</td><td>-599.7</td><td>-25.0</td><td>-666.4</td><td>-39.3</td><td>h2t</td></tr><tr><td>6g68</td><td>-303.2</td><td>-121.6</td><td>-349.0</td><td>-128.1</td><td>-352.1</td><td>-126.1</td><td>-148.8</td><td>-83.8</td><td>N/A</td><td>N/A</td><td>-15.7</td><td>-57.9</td><td>h2t</td></tr><tr><td>6ghr</td><td>-1342.1</td><td>-56.9</td><td>-1198.8</td><td>-70.7</td><td>-1260.8</td><td>-49.8</td><td>-1212.9</td><td>-30.0</td><td>-666.1</td><td>-43.8</td><td>-1224.1</td><td>-50.6</td><td>h2t</td></tr><tr><td>6ict</td><td>-1323.9</td><td>-87.2</td><td>-1259.5</td><td>-37.0</td><td>-1222.3</td><td>-29.4</td><td>-1142.0</td><td>-26.6</td><td>-662.6</td><td>-36.7</td><td>-1238.7</td><td>-57.0</td><td>h2s</td></tr><tr><td>6igk</td><td>-785.6</td><td>-104.9</td><td>-768.3</td><td>-84.0</td><td>-762.4</td><td>-64.6</td><td>-698.1</td><td>-75.3</td><td>-472.5</td><td>-52.7</td><td>-680.6</td><td>-62.2</td><td>h2t</td></tr><tr><td>6jbk</td><td>-938.6</td><td>-79.7</td><td>-933.7</td><td>-76.5</td><td>-881.2</td><td>-42.2</td><td>-871.9</td><td>-50.2</td><td>-375.9</td><td>-63.7</td><td>-847.5</td><td>-40.5</td><td>h2s</td></tr><tr><td>6ocp</td><td>-754.5</td><td>-51.1</td><td>-672.6</td><td>-15.4</td><td>-565.0</td><td>-31.0</td><td>-716.6</td><td>-30.7</td><td>-552.5</td><td>-22.7</td><td>-676.5</td><td>-34.2</td><td>h2s</td></tr><tr><td>6om4</td><td>-690.0</td><td>-67.6</td><td>-582.3</td><td>-49.2</td><td>-506.8</td><td>-9.4</td><td>-662.7</td><td>-25.1</td><td>-485.6</td><td>-22.6</td><td>-710.9</td><td>-47.6</td><td>h2s</td></tr><tr><td>6p02</td><td>-1134.3</td><td>-202.1</td><td>-1136.5</td><td>-171.3</td><td>-1059.6</td><td>-150.6</td><td>-861.6</td><td>-83.6</td><td>-679.6</td><td>-81.1</td><td>-854.0</td><td>-94.5</td><td>h2s</td></tr><tr><td>6peu</td><td>-893.2</td><td>-78.2</td><td>-832.3</td><td>-46.2</td><td>-736.2</td><td>-7.2</td><td>-935.1</td><td>-61.8</td><td>-807.9</td><td>-22.6</td><td>-860.5</td><td>-44.7</td><td>s2s</td></tr><tr><td>6q5r</td><td>-529.4</td><td>-94.9</td><td>-585.9</td><td>-132.9</td><td>-566.2</td><td>-109.0</td><td>-437.5</td><td>-98.9</td><td>N/A</td><td>N/A</td><td>-401.3</td><td>-60.9</td><td>s2t</td></tr><tr><td>6qs1</td><td>-917.2</td><td>-45.4</td><td>-847.3</td><td>-65.4</td><td>-871.5</td><td>-36.2</td><td>-893.4</td><td>-38.9</td><td>-787.7</td><td>-23.0</td><td>-903.7</td><td>-52.6</td><td>h2s</td></tr><tr><td>6r16</td><td>-1330.4</td><td>-102.6</td><td>-1309.5</td><td>-82.5</td><td>-1263.7</td><td>-47.8</td><td>-1196.3</td><td>-50.6</td><td>-440.0</td><td>-64.7</td><td>-1181.7</td><td>-50.5</td><td>h2t</td></tr><tr><td>6rqx</td><td>-1176.7</td><td>-28.1</td><td>-1069.6</td><td>-32.2</td><td>-1078.0</td><td>-19.0</td><td>-1151.5</td><td>-20.3</td><td>-1100.8</td><td>-15.3</td><td>-1207.8</td><td>-37.6</td><td>s2s</td></tr><tr><td>6rxr</td><td>-449.0</td><td>-66.1</td><td>-472.8</td><td>-49.5</td><td>-427.0</td><td>-30.4</td><td>-466.3</td><td>-35.8</td><td>-80.7</td><td>-42.2</td><td>-499.6</td><td>-192.0</td><td>s2t</td></tr><tr><td>6sa8</td><td>-461.4</td><td>-62.0</td><td>-307.1</td><td>-44.0</td><td>-364.0</td><td>-25.4</td><td>-404.2</td><td>-35.4</td><td>-247.5</td><td>-27.8</td><td>-424.9</td><td>-33.1</td><td>h2s</td></tr><tr><td>6trw</td><td>-1085.1</td><td>-44.4</td><td>-1074.5</td><td>-48.0</td><td>-947.0</td><td>-40.2</td><td>-1093.9</td><td>-36.1</td><td>-544.2</td><td>-37.3</td><td>-1080.4</td><td>-48.9</td><td>h2t</td></tr><tr><td>6y1a</td><td>-393.6</td><td>-202.3</td><td>-358.1</td><td>-191.0</td><td>39.8</td><td>-49.7</td><td>-153.8</td><td>-84.7</td><td>264.3</td><td>-68.4</td><td>-142.2</td><td>-69.2</td><td>h2t</td></tr><tr><td>6zw0</td><td>-589.9</td><td>-105.7</td><td>-576.7</td><td>-115.6</td><td>-405.2</td><td>-18.5</td><td>-458.6</td><td>-61.4</td><td>-294.6</td><td>-38.9</td><td>-426.0</td><td>-50.5</td><td>h2s</td></tr><tr><td>7atr</td><td>-1056.1</td><td>-66.1</td><td>-912.3</td><td>-35.8</td><td>-1036.8</td><td>-33.5</td><td>-991.2</td><td>-37.1</td><td>-978.0</td><td>-33.5</td><td>-999.1</td><td>-49.0</td><td>h2t</td></tr><tr><td>7brk</td><td>-561.5</td><td>-89.4</td><td>-594.4</td><td>-91.7</td><td>-614.1</td><td>-91.1</td><td>-486.8</td><td>-47.7</td><td>-345.7</td><td>-49.7</td><td>-487.3</td><td>-52.1</td><td>h2s</td></tr><tr><td>7eib</td><td>-342.6</td><td>-50.3</td><td>N/A</td><td>N/A</td><td>-331.3</td><td>-45.8</td><td>-276.4</td><td>-34.4</td><td>-270.9</td><td>-37.7</td><td>-387.1</td><td>-63.7</td><td>h2s</td></tr><tr><td>7f6h</td><td>-238.6</td><td>-50.3</td><td>N/A</td><td>N/A</td><td>-149.4</td><td>-24.6</td><td>-186.3</td><td>-38.7</td><td>-185.7</td><td>-35.7</td><td>-271.7</td><td>-57.3</td><td>h2s</td></tr><tr><td>7mhz</td><td>-187.5</td><td>-31.9</td><td>N/A</td><td>N/A</td><td>-74.8</td><td>-27.0</td><td>-148.7</td><td>-29.8</td><td>-48.7</td><td>-34.2</td><td>-217.9</td><td>-59.5</td><td>h2t</td></tr><tr><td>7okp</td><td>-845.2</td><td>-36.8</td><td>N/A</td><td>N/A</td><td>-721.8</td><td>-20.4</td><td>-793.1</td><td>-27.6</td><td>-522.7</td><td>-17.2</td><td>-857.3</td><td>-36.6</td><td>h2t</td></tr><tr><td>7owu</td><td>-469.1</td><td>-50.8</td><td>N/A</td><td>N/A</td><td>-454.9</td><td>-39.1</td><td>-464.5</td><td>-23.4</td><td>-293.0</td><td>-27.0</td><td>-487.5</td><td>-62.8</td><td>s2t</td></tr><tr><td>7q66</td><td>-375.5</td><td>-189.6</td><td>N/A</td><td>N/A</td><td>-306.4</td><td>-130.5</td><td>-17.6</td><td>-88.6</td><td>241.7</td><td>-54.4</td><td>-96.6</td><td>-80.0</td><td>h2s</td></tr><tr><td>7ure</td><td>-41.8</td><td>-50.4</td><td>-65.1</td><td>-61.4</td><td>101.5</td><td>-40.2</td><td>120.6</td><td>-23.2</td><td>114.4</td><td>-29.2</td><td>56.4</td><td>-40.7</td><td>h2s</td></tr><tr><td>7vb7</td><td>-156.8</td><td>-92.4</td><td>62.6</td><td>-78.2</td><td>-23.8</td><td>-33.1</td><td>47.1</td><td>-38.2</td><td>N/A</td><td>N/A</td><td>19.1</td><td>-20.6</td><td>h2s</td></tr><tr><td>7vwo</td><td>-647.0</td><td>-93.5</td><td>-519.6</td><td>-58.2</td><td>-435.8</td><td>-41.1</td><td>-464.2</td><td>-44.9</td><td>-125.6</td><td>-72.4</td><td>-523.5</td><td>-54.8</td><td>h2t</td></tr><tr><td>7wvx</td><td>-314.7</td><td>-81.9</td><td>-170.0</td><td>-54.2</td><td>-221.3</td><td>-45.4</td><td>-159.7</td><td>-40.5</td><td>-177.8</td><td>-58.6</td><td>-254.4</td><td>-64.9</td><td>h2t</td></tr><tr><td>7xxf</td><td>-243.6</td><td>-73.7</td><td>-281.3</td><td>-71.4</td><td>-259.1</td><td>-71.7</td><td>-154.1</td><td>-54.7</td><td>-64.0</td><td>-54.4</td><td>-115.9</td><td>-44.2</td><td>h2s</td></tr><tr><td>7yat</td><td>-655.4</td><td>-247.2</td><td>-632.5</td><td>-233.3</td><td>-234.9</td><td>-39.4</td><td>-356.3</td><td>-82.7</td><td>859.3</td><td>-65.8</td><td>-532.1</td><td>-241.8</td><td>h2s</td></tr><tr><td>8dgq</td><td>-843.5</td><td>-91.9</td><td>-760.5</td><td>-34.4</td><td>-654.9</td><td>-23.8</td><td>-621.5</td><td>-18.1</td><td>-651.9</td><td>-41.0</td><td>-710.6</td><td>-40.4</td><td>h2t</td></tr></table>

Property-guided sampling can be incorporated during generation to generate chemically and structurally valid cyclic peptides (Dhariwal & Nichol, 2021; Ho & Salimans, 2022). For example, the process can be conditioned on predefined bond length and angle distributions to sample cyclic peptides with specific shapes. Additionally, energy-based sampling (Lu et al., 2023; Kulytè et al., 2024) and energy-based preference optimization (Zhou et al., 2024a;c;d; Cheng et al., 2024) can guide the generation of low-energy, stable conformations. Techniques and architectures related to AlphaFold 3 could also be leveraged for more accurate atomic interaction modeling (Abramson et al., 2024). For evaluation, we believe it is crucial to validate the cyclic peptides generated by our model in the wet lab, determining their accurate structural conformations and binding modes—an avenue we are actively exploring.

Additionally, we would like to point out that our current method does not explore how to automatically select the best cyclization type for a given receptor, although it can be enumerated. We plan to investigate this aspect in our future work. Other future research includes cyclic ligand peptide design considering flexible protein targets (Zhou et al., 2025) or dual targets (Zhou et al., 2024b).

H2T-1   
![](images/2e5cdc4396d6e1bd48e27cb34f651e6cf6a6b668fbd619797f0b736940d5086a.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a protein-ligand binding site with red and blue chains
</details>

![](images/1977f961e3f645e37797b410706b1794c2a53af319613791b8dd3fb7f906ca75.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple amide, ester, and pyrrolidine moieties
</details>

affinity: -32.9   
stability: -324.6

H2T-3   
![](images/af55826399e5872336e809f9f5fdbbc81fc24e383fe7efe0564474acb9b3114d.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model showing a protein or nucleotide binding site (no text or labels visible)
</details>

![](images/e1399b2d6f5440bbdf2ba4814d0560f6623b343466877ae0682d883f23a7b305.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and stereochemistry indicators
</details>

affinity: -29.9   
stability: -334.4

H2T-5   
![](images/37d617fe524a84bdd570528352695414dacebdfb3ecfb7fcc2b994cc8569b837.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model showing a protein or enzyme binding site (no text or labels visible)
</details>

![](images/0b6962d20c4a425c705626e24efc09aa0c11a3126ad1da7a936fb0a63dec388b.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and stereochemistry indicators
</details>

affinity: -26.5   
stability: -329.2

H2T-7   
![](images/549243b7a228b78f3ddbe9343e05c1763176e7a9bd4ff9c0ef26ee0194728aea.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model showing protein or nucleotide binding (no text or labels visible)
</details>

![](images/c7d93d02924aa3e0e32120ff1d5b2fb91790c9d5b6ae817bd7bc10ce3c88d041.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and stereochemistry indicators
</details>

affinity: -25.7   
stability: -353.4

H2T-2   
![](images/c21c85edb1cf5427ae2edfbc3482e774cd1229f891c0b2fed24f29ebb3672655.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model with blue and red protein structures (no text or labels)
</details>

![](images/a7a9221baaf1a4493b1acbfd6c67105be53cccc5a567e6d573cf33b373b838a1.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple aromatic rings, amide linkages, and functional groups
</details>

affinity: -32.1   
stability: -353.6

H2T-4   
![](images/c602333a9a44b321bafac73f59717b00a2e82444d876149607538b0e53a81be0.jpg)

<details>
<summary>chemical</summary>

Molecular structure visualization showing a protein-ligand binding site with red and blue secondary chains
</details>

![](images/a001f1b36d776c3017be76c06c24592f7866e64d8895bdb3864bb820ef2269f1.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and side chains
</details>

affinity: -27.3   
stability: -351.3

H2T-6   
![](images/5b48e3c7d344948e8d6dcb479968a36f488c5d3258860fb5672521d5480b1acc.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model showing a protein structure with a red active site (no text or labels visible)
</details>

![](images/fdfb8912cded77d3020eac8d903737622d8b521a455d015d892bc1a01d672cab.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and stereochemistry indicators
</details>

affinity: -33.9   
stability: -356.9

H2T-8   
![](images/58606ded399247dab1c957ddcca73000f89cda8bc011da3f9b5f4fd0fdebb0b6.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model showing a protein structure with alpha-helices and beta-sheets (no text or labels visible)
</details>

![](images/5562c0ee0079d97de235ca8d4e75ab514f2a3196f591e84f5dc6fbaa42505bf3.jpg)

<details>
<summary>chemical</summary>

Complex peptide or glycoside molecular structure with multiple functional groups and side chains
</details>

affinity: -26.4   
stability: -323.6

Figure 10. Head-to-tail cyclic peptides designed for target SMYD2.

![](images/db46fc11c25d51ff646266ec1a2f5bfa987063dab36cb5e58dd3dadc96c52c25.jpg)  
affinity: -42.9   
stability: -773.8   
affinity: -43.1   
stability: --787.5   
affinity: -43.1   
stability: -297.6   
affinity: -50.0   
stability: -784.7   
affinity: -49.0   
stability: -541.2   
affinity: -38.8   
stability: -773.6

![](images/4a6382e8e00ef4a39dc4405c3983fb7c13a3d72471ad377536ab2c3e937ed5af.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and side chains
</details>

affinity: -43.1   
stability: -554.9   
affinity: -43.2   
stability: -542.2

Figure 11. Side-to-side cyclic peptides designed for target SET8.

PDB: 4o6f  
![](images/c4a57937470c12720bf5da6bb63bd10e4eaa85cf39e4dc3f93317a1f4bb5363c.jpg)

<details>
<summary>natural_image</summary>

3D protein structure visualization with blue ribbons and a yellow ligand bound (no text or symbols)
</details>

![](images/7c219e753a1460f246a6b1c6b5708ad4cbc1742a2671d0e5621dc89178679431.jpg)

<details>
<summary>chemical</summary>

3D molecular structure of a complex organic compound with yellow, blue, and white atoms
</details>

Pepflow   
![](images/083157e4a6e2f0fbbcfe8bb58141d247dad7315141c3d63d07ec554bc232727d.jpg)

<details>
<summary>natural_image</summary>

3D ribbon diagram of a protein structure with a pink ligand bound (no text or symbols)
</details>

![](images/c0b69f7c844fc0ec8498f394704f5e0193a5bc3c86ac16fe1ad9f47efbbf53f9.jpg)

<details>
<summary>natural_image</summary>

3D molecular structure visualization with pink and blue chains, no visible text or labels
</details>

Our (head-to-tail)   
![](images/9ed7d9093db4fcb2086e6c80d86753cbb761801e5df00c6ee147ebc3821d4303.jpg)

<details>
<summary>natural_image</summary>

3D protein structure visualization with blue ribbons and a red ligand bound (no text or labels)
</details>

![](images/2a77be08efde9391d76633934efb79d7662657b1d99a2f3f773b2df225a6cb53.jpg)

<details>
<summary>natural_image</summary>

3D molecular structure visualization with red and blue components, no visible text or labels
</details>

Figure 12. Structure ensembles of SMYD2 from multiple perspectives.

![](images/bfb4e68573f23854aac09cb4a4d88752263e3714d4a505cab2e84ac6d02ab553.jpg)  
Figure 13. Structure ensembles of SET8 from multiple perspectives.

PDB: 1zkk   
![](images/badcc1b3a88774e47b64deeea77610d45c36465837a13d254c25b4d2b7423087.jpg)

<details>
<summary>natural_image</summary>

Molecular surface visualization with highlighted residues (no text or symbols)
</details>

Affinity: -47.0
Stability: -824.2

PepFlow   
![](images/63445b22ac349392507da7e43f630a2de74bd42e3ca2ece171338b5f4767d1db.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a protein-ligand interaction with labeled residues and bonds
</details>

Affinity: -30.0
Stability: -797.3

Our (head-to-tail)   
![](images/b9c5de1ddb03607497d9616a93bae1ec9f38d347b2ca4f44a0ff505f10ae4721.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing protein backbone with red and blue chains
</details>

Affinity: -62.9
Stability: -793.2

Our (chemical graph)   
![](images/74d56907869bf69222b04f42bfd212f4ebfb0775586514922b194a87f19fdc21.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple rings, heteroatoms, and functional groups
</details>

PDB: 6rxr   
![](images/c823667711a150663c790c7664e20f07c2013a00c9f4b1aecb74bf84c61ada70.jpg)

<details>
<summary>natural_image</summary>

Molecular structure visualization with colored regions (no text or labels)
</details>

Affinity: -66.1
Stability: -449.0

PepFlow   
![](images/9f421150956933ffb30a444ea1f76a14b9a65f9ef038b11f60364c1e64dd508b.jpg)

<details>
<summary>natural_image</summary>

Molecular surface visualization showing protein or nucleotide interactions (no text or labels)
</details>

Affinity: -35.8
Stability: -466.3

Our (side-to-tail)   
![](images/06bcb48789c480172ba72f562d6886cd273bf7c0b81b91ced5470b26d0f0430a.jpg)

<details>
<summary>natural_image</summary>

Molecular surface visualization showing protein interactions with red and blue ligand structures (no text or labels)
</details>

Affinity: -192.0
Stability: -499.6

Our (chemical graph)   
![](images/c0a416a9a4b4198828a9f4d74ba237f08ec37976b10dd152f32453c0417c64a7.jpg)

<details>
<summary>chemical</summary>

Complex molecular structure diagram with multiple functional groups and stereochemistry indicators
</details>

PDB: 4jo6   
![](images/6f88c48e2040571fb64eb94ce531c5ab736514f47b9199b2c5f775593c31a0f6.jpg)

<details>
<summary>natural_image</summary>

Molecular interaction visualization with yellow ribbon and blue/purple atoms, surrounded by green and blue molecular surface (no text or labels)
</details>

Affinity: -81.9
Stability: -925.4

PepFlow   
![](images/a8273b2499f092ff9c8841ccd4c32ec4480c86c0b3225e45ef5c769d8c4e6e35.jpg)

<details>
<summary>chemical</summary>

Molecular structure visualization showing protein-ligand binding with colored secondary structure elements
</details>

Affinity: -32.6
Stability: -792.0

Our (side-to-side)   
![](images/27d84ecdf9c575157e1292c7a046fc6c2c73bf8062af9c62b4df05efb7550903.jpg)

<details>
<summary>natural_image</summary>

Molecular surface visualization with red and blue ligand structures (no text or labels)
</details>

Affinity: -45.4
Stability: -759.1

Our (chemical graph)   
![](images/3ae0dc0f04c1ce45dd98efb0ec508e8698b9e6dca0f63047dcefe3b7cc87f330.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple fused rings and heteroatoms
</details>

PDB: 5zw6   
![](images/0ca6a948c536fe9bf4d32b798b00b499be75068b62e6108add551f47203a8eb1.jpg)

<details>
<summary>natural_image</summary>

Abstract grayscale 3D surface visualization with a central orange line and irregular gray regions (no text or symbols)
</details>

Affinity: -46.9
Stability: -361.4

PepFlow   
![](images/20bfdd7d0f4fabea7add58449d516f160d0778c41a87c1097d61515b707c0490.jpg)

<details>
<summary>natural_image</summary>

Abstract grayscale pattern with a central red line and diffuse surrounding shapes (no text or symbols)
</details>

Affinity: -41.4
Stability: -338.0

Our (head-to-side)   
![](images/f0b260ab19b622b1a9d7c7184062fe2635a5ee03818aa9bfd29c370223918c19.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model with highlighted active site residues (no text or labels)
</details>

Affinity: -51.5
Stability: -364.7

Our (chemical graph)   
![](images/094aa2cf9ceed910a7412942598a75dba0a1a95108b949f015e9e05616c8e86e.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and stereochemistry indicators
</details>

PDB: 7okp   
![](images/ed3b62e71ff3e1afc72711ad946d8eb019683b6dde6792cf023644bbf47b66d0.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a yellow ligand bound to a protein surface
</details>

Affinity: -36.8
Stability: -845.2

PepFlow   
![](images/76577cbfaebf625f5e6057378ba06aca2a348aee344de2e0756983c45ee03087.jpg)

<details>
<summary>chemical</summary>

Molecular structure of a protein or lipid bilayer with visible secondary structure elements
</details>

Affinity: -27.6
Stability: -793.1

Our (head-to-tail)   
![](images/c731fc3f5f94b8be9aebf0933b68f614787ca9edca707b2a6fd7fea2054113ed.jpg)

<details>
<summary>chemical</summary>

Molecular structure of a protein complex with visible secondary structure elements
</details>

Affinity: -36.6
Stability: -857.3

Our (chemical graph)   
![](images/a11b4ba165d26aacda60e5eb39cab9fc9289fd000795dfb1363ddab56fd3b65e.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple amide, thioether, and pyrrolidine moieties
</details>

Figure 14. Examples of reference peptides, linear peptides designed by PepFlow, and cyclic peptides designed by our method.

PDB: 7f6h   
![](images/0e955a5b2874967aa348f21e631af6cbac400a1f5b436f6ea26e42a6f9681424.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a yellow ligand bound to a protein surface
</details>

Affinity: -50.3   
Stability: -238.6

PepFlow   
![](images/b353a7b542a52d7cefa5308ab7442d068427885ba1ba2608379f6634d72d2837.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a ligand bound to a protein pocket, with red and blue atoms indicating different functional groups.
</details>

Affinity: -38.7   
Stability: -186.3

Our (head-to-side)   
![](images/9348635900fd0eef02998792714b93bcf723638dac138229be2306342e542893.jpg)

<details>
<summary>natural_image</summary>

Molecular surface visualization with red and blue structural elements (no text or labels)
</details>

Affinity: -57.3   
Stability: -271.7

Our (chemical graph)   
![](images/26ee4b7be1e6ebb5c0646cff4e5f1f28aee4f34f656e65d9e9cbf82e27348984.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and stereochemistry indicators
</details>

PDB: 7eib   
![](images/697bd8e58968ec190d3ce2411214367ef3b80bf1efa9313d7774a45f0036b331.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model with highlighted active site residues (no text or symbols)
</details>

Affinity: -50.3   
Stability: -342.6

PepFlow   
![](images/89489e76c6574ef462489fba81ad498fd397ad6c4d919ab29a9a112431ed6eee.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model showing protein or nucleotide backbone with red and blue subunits (no text or labels)
</details>

Affinity: -34.4   
Stability: -276.4

Our (head-to-side)   
![](images/9a0852faf0b892777edd4503cc6957671a59187a4d44da7df49145d2948fccf2.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a red active site with blue nitrogen atoms and red oxygen atoms in a protein binding pocket
</details>

Affinity: -63.7   
Stability: -387.1   
Our (chemical graph)

![](images/f29a6f856fe4bc1281d5b680907a238b3970f5655dc7b5001bf77ddcbe23f345.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and stereochemistry indicators
</details>

PDB: 6trw   
![](images/0d14836de531778ff969f2cddf4ca99de883bcb69f75d98467bc55b24e1b280e.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a central ligand bound to a porous surface, with labeled atoms and bonds
</details>

Affinity: -44.4   
Stability: -1085.1

PepFlow   
![](images/43419dee4811221f5f119892c7c8a80b1327e1aa93c553410823e27a89965906.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model with a highlighted protein structure (no text or symbols)
</details>

Affinity: -36.1   
Stability: -1093.9

Our (head-to-tail)   
![](images/b84381a8068ab03d9330709d8f641d90655a1b6428257ccc4344401776352a95.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model showing a central protein-like structure with red and blue ligand binding sites (no text or labels)
</details>

Affinity: -48.9   
Stability: -1080.4

Our (chemical graph)   
![](images/6a37bb96304a2a6cd9916fd09b8c171c40bcc5e065d60faed37ae23a9adb1880.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple aromatic rings, amide linkages, and terminal functional groups
</details>

PDB: 3e8e   
![](images/550b6e36e75fb32ed1ba55948a2111308de9f0ec20710f232aad1150f92b2b9b.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a yellow ligand bound to a protein surface
</details>

Affinity: -52.3   
Stability: -1144.4

PepFlow   
![](images/9d2c58159170f64c79f606a54bf1d391135b1e715187b7a4f11061f967dbb313.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a protein-ligand binding site with labeled residues and secondary structure elements
</details>

Affinity: -20.3   
Stability: -1020.6

Our (head-to-tail)   
![](images/3d6d4bcb1e0a9daeab126261b9c64af10ce4109bc7f637c7a8106da246c1696e.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model with red highlighted active site residues (no text or labels)
</details>

Affinity: -37.4   
Stability: -1081.7

Our (chemical graph)   
![](images/9933f057a2b6d54e98cd55116b7c8686fb5cd9e2609c4f476a13b2f99c80ae04.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple fused rings and functional groups
</details>

PDB: 5yis   
![](images/09ed216740692880e0c3a30668068b973cfa963af98a6cf4e45e76635448a9f9.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a yellow ligand bound to a protein surface, with green and blue regions representing different functional groups.
</details>

Affinity: -99.1   
Stability: -416.4

PepFlow   
![](images/3b86467b6c7f8820a46c6d7884e1ab1e6eaa9c7c186501420befe3c0f30294ac.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface visualization with green and blue regions and a pink ribbon structure (no text or labels)
</details>

Affinity: -50.1   
Stability: -265.5

Our (side-to-side)   
![](images/db754e89eb809736076fbcb673d95633012414352339e02fdf6b53f3549c93b7.jpg)

<details>
<summary>natural_image</summary>

3D molecular surface model with green and blue regions and a red ribbon structure (no text or labels)
</details>

Affinity: -56.1   
Stability: -302.1

Our (chemical graph)   
![](images/7289c92215adf709267aa47ffab934861a9bcc54e02db979d39bd0399a695e3b.jpg)

<details>
<summary>chemical</summary>

Complex organic molecule structure with multiple functional groups and stereochemistry indicators
</details>

Figure 15. Examples of reference peptides, linear peptides designed by PepFlow, and cyclic peptides designed by our method.