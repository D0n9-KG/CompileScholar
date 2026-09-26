Bridging Text and Molecule: A Survey on Multimodal Frameworks for Molecule 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.13830v1 [q-bio.BM] 07 Mar 2024 
 
 

# Bridging Text and Molecule: A Survey on Multimodal Frameworks for Molecule

 
 
 Yi Xiao
 † † thanks: Equal contribution. †  Corresponding author. 
 Affiliation:  Center for Research on Intelligent Perception and Computing (CRIPAC),
 
 Email:  {y.xiao.cs}@outlook.com 
 
    
 Xiangxin Zhou
 
 Affiliation:  Center for Research on Intelligent Perception and Computing (CRIPAC),
 
 Email:  {zhouxiangxin1998}@gmail.com 
 
    
 Qiang Liu
 
 Email:  {qiang.liu 
 
    
 Liang Wang
 
 Affiliation:  State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS),
 
 Affiliation:  Institute of Automation, Chinese Academy of Sciences (CASIA)  
 
 Affiliation:  Center for Research on Intelligent Perception and Computing (CRIPAC),
 
 Affiliation:  School of Artificial Intelligence, University of Chinese Academy of Sciences 
 
 Email:  wangliang}@nlpr.ia.ac.cn 
 

 Abstract 
 
 Artificial intelligence has demonstrated immense potential in scientific research. Within molecular science, it is revolutionizing the traditional computer-aided paradigm, ushering in a new era of deep learning. With recent progress in multimodal learning and natural language processing, an emerging trend has targeted at building multimodal frameworks to jointly model molecules with textual domain knowledge. In this paper, we present the first systematic survey on multimodal frameworks for molecules research. Specifically,we begin with the development of molecular deep learning and point out the necessity to involve textual modality. Next, we focus on recent advances in text-molecule alignment methods, categorizing current models into two groups based on their architectures and listing relevant pre-training tasks. Furthermore, we delves into the utilization of large language models and prompting techniques for molecular tasks and present significant applications in drug discovery. Finally, we discuss the limitations in this field and highlight several promising directions for future research.

 
 
 

## 1 Introduction

 
 Accurately modeling molecules and extracting meaningful features is a primary goal for molecular deep learning. Initially, manual descriptors such as molecular fingerprints and SMILES are proposed to describe molecules in strings or sequences. These descriptors can naturally be encoded by the transformer architecture for feature extraction. Subsequently, graph structures gradually shows their superiority in modeling the topology structure within molecules. Graph Neural Networks (GNNs) are employed to learn from molecular graph by aggregating and propagating information within atoms and chemical bonds Kipf and Welling (2017) . Simultaneously, numerous of works integrate self-supervised pre-training in this process to generate generalized representations. Despite the prosperous in molecular deep learning, two key challenges exist persistently. First, owing to the complexity of chemical space and chemical rules, current deep learning frameworks lack a deep comprehension of chemical domain knowledge (e.g. Quantum mechanics rules). Furthermore, both supervised and self-supervised models need to be trained or fine-tuned on labeled molecules, which are usually scarce in real application due to the costly experimental assessment. These notorious problems decelerate the progress in related areas.

 
 
 Recently, multimodal learning and Large Language Models (LLMs) have shown impressive competence in modeling and inference. Inspired by the success of vision-language models, it is natural to associate molecules with text description to build multimodal frameworks. Following this idea, one line of works treat molecules as languages with special grammar, and cross-language frameworks, such as T5 Raffel et al. (2020) , are chosen as backbone to jointly model text and molecules. Edwards et al. (2022) ; Taylor et al. (2022) ; Pei et al. (2023) . In the same time, another line of works explores the latent space alignment between text and structured molecular data Su et al. (2022) ; Liu and others (2023a) , and attempts to integrate LLMs into multi-modal frameworks as predictor for cross-modal molecular tasks. Furthermore, prompting techniques are also introduced in the training process and yield competitive results in many molecular tasks without large-scale pre-training Liang et al. (2023) ; Cao et al. (2023) ; Zhang et al. (2023) . Lately, some insightful works attempt to build autonomous agents for chemistry and biology Boiko et al. (2023) ; Liu et al. (2023b) , bringing new paradigm of future scientific research.

 
 
 However, as a prosperous subject, there still lacks a systematic review to summarize recent progress and propose promising outlooks. In this regard, we present the first survey on multimodal frameworks for molecule. We summarize our contributions as follows: (1) We provide an overview of this field with a structured taxonomy that categorize the framework based on their basic architecture. (2) Our systematic review provides a detailed analysis of training strategies, dataset construction methods and corresponding applications. (3) We analyze the limitations in this field and provide several promising research directions.

 
 
 

 Figure 1: Pipeline of multimodal framework for molecule and downstream molecular tasks (a-c). (a) Latent space alignment and adaptation of downstream tasks. The single-stream framework jointly models text and molecules with the same encoder. The downstream tasks are realized with task-specific prompts described in section 4.1 ; The multi-stream framework involves cross-modal alignment between text and molecules. Features from latent space can be directly used for tasks or be used in instruction-tuning. (b) Building a semi-autonomous agent for molecular research with instructions and in-context examples. (c) Building autonomous agent for chemistry with instructions and chain-of-thought prompting. Equipping agent with external tools and memory largely expand the autonomous level and capabilities. 
 
 
 

## 2 Molecular Descriptors and Encoding

 
 Molecules need to be transformed into descriptors for the recognition of the model. In this section, we briefly summarize the mainstream descriptors of small molecules and proteins, along with their corresponding encoder architectures. Generally, both small molecules and proteins can be described by sequences and graphs.

 
 

### 2.1 Small Molecule Representation

 

#### 1D Sequences

 
 The Simplified Molecular Input Line Entry System (SMILES) is the most frequently used sequential descriptor of small molecules, which maps atoms, bonds, and special structures with ASCII symbols. The molecular sequences can be tokenized like text sequence, and Transformer  Vaswani et al. (2017) can be employed for molecular encoding Zeng et al. (2022) ; Liu and others (2023b) .

 
 
 

#### 2D Graph

 
 The topology structure of molecules can be naturally modeled by graph, with atoms as nodes and bonds as edges. GNNs  Kipf and Welling (2017) can be used to learn local and global representations of molecular graphs. Recently, the molecular representations that are pre-trained with GNNs structure have shown competitive results in various downstream tasks Liu et al. (2022) , demonstrating the effectiveness of graph descriptors.

 
 
 

#### 3D Geometry

 
 2D molecular graphs have limitations in capturing the spatial information within molecules. For example, chiral molecules cannot be distinguished through most of the 2D graph. The geometry information of conformers (e.g., torsional angles, bond length) holds direct relation with molecular properties. In 3D geometry, atoms are associated with their coordinates with features expressed in high-order tensors to ensure geometric symmetries and expressiveness. Many studies concentrate on designing equivariant GNNs to accurately model the interaction between atoms Batzner and others (2022) .

 
 
 
 

### 2.2 Protein Representation

 

#### Protein Sequence

 
 A protein can be viewed as a combination of 20 types of amino acids, making it possible to be expressed as amino acid sequences in a similar manner to molecules. The amino acid sequence captures the co-evolutionary information and plays a vital role in protein folding and function. Transformer-based models, commonly referred as protein language models (PLMs), use similar architecture in Natural Language Models to featurize protein for prediction or editing tasks.

 
 
 

#### Protein Graph

 
 Protein functions are primarily determined by their folded structures Jumper et al. (2021) . To better capture structural information, proteins can be represented as a residue-level relation graph, where nodes represent the alpha carbon of residues and edges represent connectivity between residues and amino-acids. We can also employ GNNs such as Massage Passing Neural Networks (MPNNs) for protein graph encoding Dauparas et al. (2022) .

 
 
 
 
 

## 3 Latent Space Alignment between Text and Molecule

 
 The encoding stage featurizes text and molecules into a single modality, while these representations still inhabit diverse semantic spaces and cannot interact with each other. To facilitate downstream tasks, different architectures are designed for text-molecule fusion and latent space alignment. In this section, we classify model architectures by the fusion scheme and summarize the corresponding pre-training tasks. We present a summary of representative works in Table 1 .

 
 

### 3.1 Model Architecture

 
 Drawing inspiration from previous works in vision-language pre-training Du et al. (2022) , we categorize models into single-stream , and multi-stream architecture. The two types of models mainly differ in their understanding of molecular latent space.

 
 

#### Single-Stream Architecture

 
 A single-stream architecture assumes that the latent space of molecules and text share similar semantic meaning. Under this circumstance, molecules are treated as a specialized language and expressed by sequential descriptors such as SMILES. After tokenization, the molecular and textual tokens are typically fed into an encoder-decoder language model, such as T5  Raffel et al. (2020) , for multi-language pre-training. KV-PLM  Zeng et al. (2022) and MolXPT  Liu and others (2023b) use byte-pair encoding (BPE) to tokenize SMILES and replace all molecular names in sequence with SMILES tokens, making these “wrapped” sequences as training data. BioT5  Pei et al. (2023) use separate vocabularies for molecules, proteins, and texts to avoid misunderstanding of tokens that may have the same expression but originate from different semantic spaces. Notably, GIMLET  Zhao et al. (2023a) serializes molecular graphs as node sequences and introduce position embedding to jointly encode nodes with related text tokens. This setting not only maintains the inductive bias at the graph level but also avoids introducing an extra graph encoding block and keeps molecular features independent of text.

 
 
 

#### Multi-Stream Architecture

 
 Models with multi-stream architecture employ intra-modality processing for text and molecules. GNNs are generally adopted as molecular encoders to encode structural information. The representations are then projected into textual latent space by a linear layer, or projected with cross-attention mechanism for cross-modal fusion and alignment. For instance, Abdine et al. (2023) fuse protein sequence and protein graph features by element-wise addition and use them as input of cross-attention module to adapt with text. Xu et al. (2023) choose both text and protein representation as keys and apply two separate cross-attention modules to produce fused-text and fused-protein representations.

 
 
 Q-Former  Li and others (2023) is a representative architecture in vision-language modeling that leverages cross-attention layers to bridge the modality gap. Similarly, Li et al. (2024) ; Liu et al. (2023d) ; Zhang et al. (2023) adopt Q-Former to connect molecular graph with text and extract text-related molecular features with a learnable query. Particularly, Liu et al. (2023a) propose GIT-Former which can be viewed as a variation of Q-Former with additional input modalities from molecular images and sequences.

 
 
 
 

### 3.2 Pre-training Tasks

 
 The fused representations need to be aligned in a unified latent space to keep consistent semantic meaning for downstream tasks. In this section, we review the commonly used pre-training tasks for alignment between text and molecules.

 
 

#### Molecule-Text Contrastive Learning

 
 The contrastive learning (CL) task between molecule and text aims to align multi-modal representations by enhancing the correlation between matched molecule-text pairs. The contrastive learning objective pushes the embeddings from matched text and molecules closer in latent space while enlarging the distance between pairs from different molecules. The CL task will enhances the model with cross-modal retrieval and matching ability. Here, we present the expression of commonly used InfoNCE  Oord et al. (2018) loss:

 

 
 | 
 ℒ NCE = − ∑ i log exp ⁡ ( z i M ⋅ z i T / τ ) ∑ j = 1 N exp ⁡ ( z i M ⋅ z j T / τ ) \displaystyle\mathcal{L}_{\text{NCE}}=-\sum_{i}\log\frac{\exp(z_{i}^{M}\cdot z_{i}^{T}/\tau)}{\sum_{j=1}^{N}\exp(z_{i}^{M}\cdot z_{j}^{T}/\tau)} | 
 | 
 (1) | 
 

 where τ \tau is the temperature coefficient. In order to facilitate the convergence, a trainable linear projector can be used to minimize the modality gap before the contrastive learning Liu and others (2023a) . 
 Although contrastive learning is an effective approach for cross-modal molecule-text alignment, it has some domain-specific limitations. One issue is that the structural information of molecules will lost in the process of encoding. Additionally, limited number of molecule-text pairs also has impacts on the alignment result. Motivated by the molecular graph augmentation methods You et al. (2020) , we could adopt augmented graphs to construct molecule-text pairs, as the augmented graphs remain similar semantics information to the original graphs, such as properties and structures. Applying a contrastive loss between the embeddings of the original graph and the augmented graph also ensures this consistency. MoMu  Su et al. (2022) introduces two augmented graphs with node dropping and sub-graph extraction to construct matched pairs with text descriptions. MolLM  Luo et al. (2024) follows the same augmentation rules in MoMu and introduces two extra augmentations which are chemical transformation and motif removal. These augmentations increase the size of training data, making the alignment process more robust.

 
 
 

#### Molecule-Text Matching

 
 Molecule-text matching (MTM) aims to predict whether a molecule-text pair is matched or not. It is defined as a binary classification task with the following loss function:

 

 
 | 
 ℒ MTM = − 𝔼 [ ∑ i [ log p ( m i , t i ) − \displaystyle\mathcal{L}_{\text{MTM}}=-\mathbb{E}\left[\sum_{i}[\log p(m_{i},t_{i})-\right. | 
 | 
 
 
 | 
 log p ( m i , t j ) − log p ( m j , y i ) ] ] \displaystyle\left.\log p(m_{i},t_{j})-\log p(m_{j},y_{i})]\right] | 
 | 
 (2) | 
 

 where ( m i , t i ) (m_{i},t_{i}) denote matched molecule and text pair. The MTM task enables the model with the retrieval ability and refines the alignment between text and molecule. Liu et al. (2023d) ; Li et al. (2024) use MTM in the first-stage training of Q-Former. Liu et al. (2023a) extend the MTM to cross-modal matching which performs matching tasks between text, graphs, and images.

 
 
 

#### Conditional Generation

 
 Conditional generation (CG) aims to generate tokens based on given conditions or constraints. Tasks such as molecule captioning and text-based molecule generation all fall into this category. Conditional generation enables models to learn complex mapping rules between text and molecules. It is adaptable for T5 architecture where all molecular tasks are transformed into text-to-text generation format. The objective function can be written as:

 

 
 | 
 ℒ CG = − ∑ i n i log P ( u i | C ; θ ) \displaystyle\mathcal{L}_{\text{CG}}=-\sum_{\begin{subarray}{c}i\end{subarray}}^{\begin{subarray}{c}n_{i}\end{subarray}}\log P(u_{i}|C;\theta) | 
 | 
 (3) | 
 

 where u i u_{i} is the i i -th token and C C denotes the generation condition which may be referred to as a molecule graph or text description depending on the task.

 
 
 

#### Masked Language Modeling

 
 As discussed in section 3 , modeling languages and molecules may share similarities. Under this assumption, masked language modeling as a prevalent pre-training task for LLMs can also be used for training molecule sequences or wrapped sequences. During the pre-training stage, the models are trained to predict the masked components using the remaining context. The training objective is defined by cross-entropy

 

 
 | 
 ℒ MLM = − 𝔼 T ∈ 𝒟 ∑ m ~ ∈ ℳ log p ( m ~ | T \ ℳ ) \displaystyle\mathcal{L}_{\text{MLM}}=-\mathbb{E}_{T\in\mathcal{D}}\sum_{\begin{subarray}{c}\tilde{m}\in\mathcal{M}\end{subarray}}\log p(\tilde{m}|T\backslash\mathcal{M}) | 
 | 
 (4) | 
 

 where ℳ , T \ ℳ , T \mathcal{M},T\backslash\mathcal{M},T represent the masked tokens, unmasked tokens, tokenized text and molecules separately. This self-supervised pre-training task can enhance the contextual comprehension of the model, improving performance in many downstream tasks. For MLM, there are two types of masking: token masking represented by BERT  Devlin and others (2018) and its variants, and span masking represented by T5  Raffel et al. (2020) . According to Raffel et al. (2020) , span masking is more efficient. Zeng et al. (2022) randomly mask tokens from both molecules and text in wrapped text sequences. Edwards et al. (2022) ; Pei et al. (2023) ; Rubungo et al. (2023) ; Qian et al. (2023) all adopt span masking for MLM with corresponding T5 backbone to enhance the downstream translation tasks between molecule and text. 
 Similarly, the protein language models (PLM) introduce masked protein modeling (MPM) by masking residues in protein sequences. ProtST  Xu et al. (2023) not only uses MPM for protein encoder pre-training, it also utilizes fused text and protein representations to predict the type of masked residues and language tokens. Before entering the PLM and biomedical language model, 15% of residues in protein sequence and 15% of word tokens in text are masked and then embedded. Two MLPs are trained with MPM objective to recover the masked components in text and residues.

 
 
 

#### Casual Language Modeling

 
 Different from the autoencoder (AE) language models such as BERT and T5 which adopt MLM as the training objective, the autoregressive models represented by GPT  Yenduri et al. (2023) are trained with Casual Language Modeling (CLM). The objective of CLM is to predict the next token in a sequence in a left-to-right direction. The objective function can be written as

 

 
 | 
 ℒ CLM = − ∑ i n i log P ( u i | u i − k , … , u i − 1 ; θ ) \displaystyle\mathcal{L}_{\text{CLM}}=-\sum_{\begin{subarray}{c}i\end{subarray}}^{\begin{subarray}{c}n_{i}\end{subarray}}\log P(u_{i}|u_{i-k},...,u_{i-1};\theta) | 
 | 
 (5) | 
 

 where n i n_{i} and k k represent the number of tokens and context length. Transformation of molecular tasks into text generation helps the CLM seamlessly integrate into the training and instruction-tuning process. Liang et al. (2023) ; Cao et al. (2023) ; Zhang et al. (2023) . We will discuss the detail of instruction-tuning and adaptation of tasks in the following sections.

 
 
 
 
 

## 4 Bridging LLMs and Molecular Tasks with Prompting Techniques

 
 With the advancement of multimodal large language model (MLLM), the cross-modal inference ability of LLMs could be extended to chemistry research. Compared with traditional cross-modal learning which focuses on modality alignment, MLLM leverage powerful LLM as the brain to process multi-modal information and utilize multiple prompting techniques such as instruction-tuning (IT), in-context learning (ICL) and chain-of-thought (CoT) to bridge LLMs with downstream tasks Li and others (2023) . As shown in Figure 1 , LLMs could conduct multiple molecular tasks with instructions and cross-modal input. In this section, we discuss the prompting techniques to build MLLM in molecular science, and show the progress to build intelligent agents for chemistry.

 
 

### 4.1 Prompt-based Fine-tuning on LLM

 
 To bridge the gap between pre-training and downstream tasks, Raffel et al. (2020) transfer all NLP tasks into text-to-text generation format with task-specific prefix. Based on this work, Gao and others (2021) propose prompt-based fine-tuning to unify the fine-tuning framework among different tasks with task-specific prompts. This strategy can also be applied to unify different molecular tasks for better adaptation in single framework. For example, the prompt of BBBP property prediction task in MoleculeNet  Wu et al. (2018) can be designed as: “ We can conclude that the BBBP of SMILES is tag ” where tag is the “true” or “false” prediction given by model Liu and others (2023b) . In this way, we unify the prediction task into a text generation format. Then model is fine-tuned and evaluated on each task with pre-training parameters fixed. Pei et al. (2023) enrich the above-mentioned template with task definition and explanation, which brings improvement in property prediction accuracy. Liu et al. (2023d) integrate fused feature as soft prompt and use LoRA  Hu et al. (2022) to improve adaptation efficiency. Compared with traditional fine-tuning, prompt-based fine-tuning shows impressive performance in few-shot datasets.

 
 
 

### 4.2 Instruction Tuning on LLM for Zero-shot Learning Ability

 
 Unlike prompt-based tuning, instruction tuning Wei et al. (2022a) aims to adapt models to various tasks. In the tuning process, models are trained in multiple tasks which have been unified through task-specific instructions. This multi-task learning strategy enables models to comprehend instructions and to seamlessly transfer to various tasks in a few-shot or zero-shot manner. A standard instruction entry is typically composed of three main parts: an instruction that clarifies the task, an input which is usually molecular feature, and an output that embodies the expected outcome Fang et al. (2023) .
 Liang et al. (2023) ; Luo et al. (2023) ; Cao et al. (2023) ; Li et al. (2024) ; Zhang et al. (2023) use the fused feature as a soft prompt to compose the instructions. During the tuning process, the fusion architecture is fine-tuned solely and some works also use LoRA  Hu et al. (2022) to improve efficiency Li et al. (2024) ; Cao et al. (2023) .
 Zhao et al. (2023a) compared the instruction-tuned GIMLET with other pre-trained baseline in zero-shot property prediction tasks. The leading accuracy shows the strong generalization performance to novel tasks by following instructions.

 
 
 

### 4.3 In-Context Learning and Chain-of-Thought

 
 Recently, various of attempts have been made to integrate LLMs into various chemistry research as an intelligent agent, with wide range of applications such as autonomous experiment planning Bran et al. (2023) ; Boiko et al. (2023) , conversational drug editing Liu et al. (2023b) , chemical reaction prediction Shi et al. (2023) , etc. These models leverage the in-context learning (ICL) or chain-of-thought (CoT) prompting  Wei et al. (2022b) which enables step by step reasoning and few-shot prediction for specific tasks.
The in-context learning for molecular tasks usually combines instruction-based prompts with a few molecular Question-Answer examples.
 Chen et al. (2024) ; Li et al. (2023) design the few-shot prompting with role definition, task description, in-context examples and output control to guide the prediction of LLM. Differently, ReLM  Shi et al. (2023) enhances the reaction prediction result from a GNN-based model by integrating LLM as a decision-maker. LLM learns to self-evaluate its prediction from in-context examples with confidence scores.

 
 
 The autonomous reasoning of LLM agents can be achieved by chain-of thought prompting with few-shot or zero-shot manner. The few-shot CoT directly demonstrates the reasoning steps in one or few prompts, and the agent can leverage the emergent ability of LLMs to imitate similar reasoning in the same type of tasks. Moreover, the zero-shot CoT simplifies the prompt with “Let’s think step by step” at the end of the problem description. With effective CoT and access to external knowledge, LLM agents can work semi-autonomously to support experts in scientific research. In STRUCTCHEM  Ouyang et al. (2023) , GPT-4 is guided to solve chemistry problems through formula generation and step-by-step reasoning. To correct the errors in CoT reasoning, another GPT-4 is employed to perform iterative review-and-refinement for generated results in each step. ChemCrow  Bran et al. (2023) adopt Least to Most prompting  Zhou and others (2023) (LtM) which can be seen as CoT in an auto-regressive manner. The reasoning loop in ChemCrow integrates the decomposition of the task, selecting and using external tools, and analysing the result. The input of the next reasoning loop is built upon the current results until they satisfy the expected format. It is the first LLM agent in chemistry capable of automatically completing complex planning and synthesis task.

 
 
 
 

## 5 Dataset Construction

 
 
 
 
 Model | 
 Molecule descriptors | 
 Backbone architecture | 
 Training database | 
 Training task | 

 
 MolT5   Edwards et al. (2022) | 
 SMILES | 
 T5 | 
 C4 + ZINC | 
 MLM | 

 
 Galactica   Taylor et al. (2022) | 
 Bio-Sequence | 
 Transformer Decoder | 
 Not open | 
 CLM | 

 
 KV-PLM   Zeng et al. (2022) | 
 SMILES | 
 SciBERT  Beltagy et al. (2019) | 
 PubChem + S2orc | 
 MLM | 

 
 MolXPT  Liu and others (2023b) | 
 SMILES | 
 GPT | 
 PubMed + PubChem | 
 CLM | 

 
 BioT5  Pei et al. (2023) | 
 SELFIES + Protein Sequence | 
 T5 | 
 PubChem + Swiss-Prot | 
 MLM + CG | 

 
 Text + Chem T5  Christofidellis et al. (2023) | 
 SMILES | 
 T5 | 
 Multi-domain | 
 CG | 

 
 TextReact  Qian et al. (2023) | 
 SMILES | 
 SciBERT | 
 USPTO | 
 CL + MLM + CG | 

 
 GIMLET  Zhao et al. (2023a) | 
 Graph | 
 T5 | 
 ChEMBL | 
 CG | 

 
 LLM-Prop  Rubungo et al. (2023) | 
 Molecule Strings | 
 T5 | 
 Materials Project | 
 MLM | 

 
 Text2Mol  Edwards et al. (2021) | 
 Graph | 
 Multi-stream + Transformer | 
 ChEBI-20 | 
 CL | 

 
 MoMu  Su et al. (2022) | 
 Graph | 
 Multi-stream | 
 PubChem + S2orc | 
 CL | 

 
 MolLM  Tang et al. (2023) | 
 SMILES + Graph + Geometry | 
 Multi-stream | 
 PubChem + S2orc | 
 CL | 

 
 DrugChat  Liang et al. (2023) | 
 Graph | 
 Multi-stream + Vicuna-13b | 
 PubChem | 
 CLM | 

 
 MoleculeSTM  Liu and others (2023a) | 
 Graph | 
 Multi-stream + Decoder | 
 PubChem | 
 CL | 

 
 BioMedGPT  Luo et al. (2023) | 
 Graph + Protein Sequence | 
 Multi-stream + LLaMA 2 | 
 PubChem + S2orc + UniProt | 
 CLM | 

 
 InstructMol  Cao et al. (2023) | 
 SELFIES + Graph | 
 Multi-stream + Vicuna-7b | 
 PubChem | 
 CLM | 

 
 GIT-Mol  Liu et al. (2023a) | 
 SMILES + Graph + Image | 
 Q-Former + T5 | 
 PubChem | 
 MTM + CL | 

 
 CLAMP  Seidl et al. (2023) | 
 Fingerprints | 
 Multi-stream | 
 PubChem | 
 CL | 

 
 MolCA  Liu et al. (2023d) | 
 SMILES + Graph | 
 Q-Former + Llama 2 | 
 PubChem | 
 MTM + CL + MC + CLM | 

 
 3D-MoLM  Li et al. (2024) | 
 SMILES + Geometry | 
 Q-Former + Llama 2 | 
 PubChem | 
 MTM + CL + MC + CLM | 

 
 MoleculeGPT  Zhang et al. (2023) | 
 SMILES + Graph | 
 Q-Former + Vicuna-7b | 
 PubChem | 
 CL+CLM | 

 
 ProtST  Xu et al. (2023) | 
 Protein Sequence | 
 Multi-stream | 
 SwissProt | 
 CL + MLM | 

 
 ProtDT  Liu et al. (2023c) | 
 Protein Sequence | 
 Multi-stream + Decoder | 
 SwissProt | 
 CL | 

 
 Prot2Text  Abdine et al. (2023) | 
 Protein Sequence + Protein Graph | 
 Multi-stream + Transformer | 
 SwissProt | 
 CLM | 

 
 InstructProtein  Wang et al. (2023) | 
 Protein Sequence | 
 Knowledge Graph + LLMs | 
 UniProt | 
 CLM | 

 
 BioBridge  Wang et al. (2024) | 
 SMILES + Protein Sequence | 
 Knowledge | 
 PrimeKG etc. | 
 CL | 

 
 ReLM  Shi et al. (2023) | 
 SMILES + IUPAC + Graph | 
 ICL + LLMs | 
 - | 
 - | 

 
 ChemCrow  Bran et al. (2023) | 
 - | 
 CoT + LLMs | 
 - | 
 - | 

 
 ChatDrug  Liu et al. (2023b) | 
 SMILES | 
 LLMs | 
 - | 
 - | 

 
 MolReGPT  Li et al. (2023) | 
 SMILES | 
 ICL + GPT-3.5 | 
 - | 
 - | 

 

 Table 1: Summary of representative multimodal frameworks 
 
 
 The quality of the training data is crucial for modal alignment and training, significantly influencing the performance of the model. In this section, we focus on dataset construction methods. The data resource can be found in Table 1 .

 
 

#### Data Processing

 
 To facilitate modality alignment, pairs of textual and non-textual molecular data are collected from public datasets. However, the descriptions in databases are not balanced. Taking PubChem  Kim et al. (2023b) as an example, it is very often that some molecules only have a few basic records and lack some corresponding properties. To tackle this issue, many researchers construct training data from multiple datasets or retrieve relevant text from scientific corpus such as S2orc  Lo et al. (2019) . Meanwhile, the pre-processing methods are also important. For instance, Liu and others (2023a) ; Zhang et al. (2023) ; Cao et al. (2023) first replace all the molecule names in the annotation of PubChem with token ‘ ∼ \sim ’ to simplify name comprehension in training. Then they remove the redundant information in the molecule description such as origins, sources, and some geographic notations that have no relation with structure or property. Xu et al. (2023) select four kinds of properties from Swiss-Prot  Bairoch and Apweiler (2000) and use fixed templates to rearrange the descriptions, ensuring the consistency of training data format.

 
 
 

#### Integrating Generative AI

 
 Recent advances in generative AI provide an innovative approach to mitigate the data scarcity challenge. For instance, Li et al. (2024) 
use GPT-3.5 to enrich the sparse molecular descriptions in PubChem. Fang et al. (2023) leverage GPT-3.5 to diversify prompt templates and use them to generate QA pairs for instruction-tuning. Additionally, Sakhinana and Runkana (2023) use GPT-4 to generate molecule captions for the fine-tuning. Chen et al. (2024) fabricate an “artificially-real” dataset for domain adaptation, where molecule descriptions are generated through ChatGPT with retrieval-based few-shot prompting.

 
 
 
 

## 6 Applications

 
 This section will showcase applications in drug discovery and chemistry research employing the aforementioned methods. Beyond the introduction of tasks, our emphasis lies in the adaptation between the model and tasks.

 
 

### 6.1 Text-molecule Retrieval

 
 Text-molecule retrieval task is first proposed by Edwards et al. (2021) , which aims to retrieve the corresponding molecule from a given text query. This molecule retrieval task is useful in early stages of drug discovery, where experts need to select potential molecules from compounds database for further design and optimization. The retrieval task can be accomplished by the aligned latent space between language and molecules, from which we can acquire the encoded text descriptions with the connection of target molecules. Then we can use the similarity score to evaluate the distance between text and molecules to find the best-matched pair. In KV-PLM  Zeng et al. (2022) , descriptions and molecules are encoded by a shared transformer encoder. MoMu  Su et al. (2022) and MoleculeSTM  Liu and others (2023a) use separate encoders to extract multimodal features and align the latent space with contrastive learning.

 
 
 

### 6.2 Property Prediction

 
 One of the important goals of drug discovery is to search for small molecules and proteins with desired properties. The molecule description from scientific literature and databases serve as knowledge repositories that contain properties, interactions, and structures that can hardly be inferred from current models Pei et al. (2023) . Through molecule-text alignment, text information can act as a supplementary signal to enhance molecular representation and improve the performance of models in property prediction Seidl et al. (2023) ; Xu et al. (2023) .
The property prediction task is usually a binary classification task achieved by molecular features and simple prediction head. An alternative approach is to leverage powerful generative LLMs with instructions to predict property in a QA format Zhang et al. (2023) ; Liu et al. (2023a) . As shown in 4.1 , the prediction is determined by the probabilities of tokens “true” and “false” in the generated answer.

 
 
 

### 6.3 Molecule Design

 

#### De novo Generation

 
 De novo generation in molecule design includes molecule captioning which generates a description of given molecules and text-guided de novo generation which generates molecules from scratch by the textual guidance. Models with single-stream architecture have the privilege of performing translation between text and molecule, owing to the encoder-decoder structure and text-to-text task format. Raffel et al. (2020) . Apart from the translation-based methods. Liu et al. (2023c) propose a protein design framework with a multi-stream encoder. In text-guided protein generation task, the description is first encoded by the aligned text encoder. Then a facilitator module which is parameterized by a multi-layer perception is used to learn the transformation from encoded text to protein representation. The resulting protein representation is then fed into a trained generative decoder to generate protein sequences.

 
 
 

#### Molecule Editing

 
 Molecule editing seeks to optimize current molecules with desired properties. Within the drug discovery pipeline, text-guided editing finds application in lead optimization tasks and proves valuable for decomposing multi-objective lead optimization Liu et al. (2023c) . Drawing inspiration from the success of few-shot text-to-image generation, text description can simplify the complexity of the target chemical space in the generation process. Simultaneously, personalized generation enhances drug editing by introducing high flexibility.
As mentioned above, the latent space alignment establishes a unified latent space where features possess semantic meaning in both structure and text. Building upon this approach, Liu and others (2023a) ; Liu et al. (2023c) ; Tang et al. (2023) use latent optimization methods to sample a latent representation close to both text and molecule in latent space. Then this latent code is fed into a decoder which is usually a trained molecule generation model to produce optimized molecules. Kim et al. (2023a) proposes hierarchical textual inversion which introduces intermediate and detail tokens to represent SMILES, aiming to capture cluster-level and molecule-level features. The interpolation sampling can benefit from this hierarchical design with high generation diversity.

 
 
 
 

### 6.4 Other Applications

 

#### Reaction Prediction

 
 Reaction prediction is a challenging but fundamental task in chemistry and biology. The chemical reaction process can be seen as a mapping between a set of reactants to a set of products with specific reaction conditions. Based on this assumption, there are three main tasks in reaction prediction, which are product prediction, reaction condition prediction, and most importantly, retrosynthesis prediction. Text can supply information in complex reaction mechanisms and reaction templates which GNN-based methods often fail to capture. Qian et al. (2023) retrieve reaction-related text and concatenate with input similes to enhance the retrosynthesis prediction. As described in 4.3 , we can also involve LLMs in reaction prediction via prompting engineering. For example, Shi et al. (2023) use GPT-4 to predict reaction products with the aid of in-context reaction examples and reaction prediction model.

 
 
 

#### Intelligent Agent for Scientific Research

 
 According to Bran et al. (2023) , the automation level in chemistry is relatively low compared to other domains. Despite LLMs may have difficulties in comprehending chemistry principles, they have demonstrated significant capability in understanding human instructions and organizing information based on extensive training corpora AI4Science and Quantum (2023) . Consequently, LLMs have the potential to become intelligent assistants to automatically arrange research with the help of professional tools and software. Liu et al. (2023b) design a drug editing agent with conversational interaction. The agent can receive human feedback to retrieve candidate drug molecules from the database with desired properties. Boiko et al. (2023) develop a “Coscientist” based on GPT-4 similar to ChemCrow  Bran et al. (2023) which can autonomously design and execute chemical research.

 
 
 
 
 

## 7 Conclusions and Future Outlooks

 
 In this paper, we provide a comprehensive review of multimodal frameworks for molecules. After a brief introduction to the background and molecule descriptors, we introduce the model architecture and pre-training tasks for latent space alignment. Then, we summarize the prompting techniques in a multimodal large language model to bridge LLMs with molecular tasks. As an application-oriented domain, we combine the aforementioned methods to exhibit applications in drug discovery. Although text-molecule models have made impressive progress, there exist several challenges which appeal to future research.

 
 

### 7.1 Appealing for High-Quality Data and Reliable Benchmarks

 
 According to the neural scaling law, the emergent abilities of LLM in complex molecular tasks have not been shown. The data scarcity challenge still exists for both molecules and text description data. In addition to collecting descriptions from databases, many works also automatically retrieve relative text from scientific corpus, while the authenticity and correlation of retrieved text cannot be guaranteed Xu et al. (2023) ; Tang et al. (2023) . For the progress of the community, a larger and more qualified molecule-text database is significant.
Although multimodal frameworks exhibit great potential in various molecular tasks, there remains a question of how to fairly evaluate the performance among different models. Experimental results may be unreliable due to inconsistent settings between different models and low representative of test datasets. To address this concern, new benchmarks are necessary to standardize evaluation metrics and settings, providing more reliable and realistic test data, such as drug-like molecular datasets. Several attempts have been made in this direction Guo et al. (2023) .

 
 
 

### 7.2 Extending the Interpretability of Model

 
 The lack of interpretability prohibits many applications of deep molecular models as numerical predictions alone may not be convincing enough compared with computational and experimental results. Text-involved multimodal frameworks provide an opportunity to enhance the interpretability of results. By leveraging in-context learning and chain-of-thought prompting in LLMs, models can reasoning and inference, like the human brain, to produce explainable results. Follow-up research can also try to develop interpretable tools to bridge the relation between textural description and molecule structure in latent space Su et al. (2022) . Furthermore, Wellawatte and Schwaller (2023) explore the possibility of combining XAI methods with LLMs to provide explanations of the structure-property relationship in a comprehensive way.

 
 
 

### 7.3 Improving the Reasoning Ability

 
 Applying prompting techniques can significantly improve the reasoning ability of LLM-based frameworks. However, it is observed that in some cases, models may generate unrealistic predictions or even replicate the values in examples as prediction Zhao et al. (2023b) . This serves as an evidence that LLMs may rely on memorization without truly understanding the molecules and chemical problems. Future studies may integrate successful GNNs into transformer-based model architecture, other than simply using GNNs as encoders Zachares and others (2023) . Designing effective prompts for molecular tasks can also be taken into consideration.

 
 
 

### 7.4 Integration with Foundation Models

 
 Foundation models (FMs) in the biomedical domain have shown promising performance. For example, AlphaFold  Jumper et al. (2021) can accurately predict protein’s structure from amino-acid sequence. These foundation models are usually uni-modal with sufficient training data. It is possible to integrate FMs within LLM agents or specially designed frameworks. Wang et al. (2024) has tried to model the relation between the FMs and the knowledge graph. We believe that effective frameworks could unlock the additive power of FMs.

 
 
 

### 7.5 Learning from Human/AI Feedback

 
 Recent progress in reinforcement learning from human/AI feedback (i.e., RLHF  Ouyang et al. (2022) and RLAIF  Lee et al. (2023) ) has achieved promising results in aligning LLMs with human preference. RLHF fits a reward model to human preference dataset and use RL to optimize LLMs to produce responses assigned with high rewards. This paradigm may pave the way for utilizing LLMs for biomedical applications, especially in scenarios where molecular simulation software can be used as a reward model. Exploring how to fully utilize the power of RLHF at the interaction of text and molecules is an appealing research direction.

 
 
 
 
 

## References

 
 
 Abdine et al. [2023] 
 
Hadi Abdine, Michail Chatzianastasis, Costas Bouyioukos, and Michalis Vazirgiannis.
 
 
 Prot2text: Multimodal protein’s function generation with gnns and transformers.
 
 
 arXiv:2307.14367 , 2023.
 
 

 
 AI4Science and Quantum [2023] 
 
Microsoft Research AI4Science and Microsoft Azure Quantum.
 
 
 The impact of large language models on scientific discovery: a preliminary study using gpt-4.
 
 
 arXiv:2311.07361 , 2023.
 
 

 
 Bairoch and Apweiler [2000] 
 
Amos Bairoch and Rolf Apweiler.
 
 
 The swiss-prot protein sequence database and its supplement trembl in 2000.
 
 
 Nucleic acids research , 2000.
 
 

 
 Batzner and others [2022] 
 
Simon Batzner et al.
 
 
 E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials.
 
 
 Nature communications , 2022.
 
 

 
 Beltagy et al. [2019] 
 
Iz Beltagy, Kyle Lo, and Arman Cohan.
 
 
 SciBERT: A pretrained language model for scientific text.
 
 
 In EMNLP , 2019.
 
 

 
 Boiko et al. [2023] 
 
Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes.
 
 
 Autonomous chemical research with large language models.
 
 
 Nature , 2023.
 
 

 
 Bran et al. [2023] 
 
Andres M Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew White, and Philippe Schwaller.
 
 
 Augmenting large language models with chemistry tools.
 
 
 In NeurIPS 2023 AI for Science Workshop , 2023.
 
 

 
 Cao et al. [2023] 
 
He Cao, Zijing Liu, Xingyu Lu, Yuan Yao, and Yu Li.
 
 
 Instructmol: Multi-modal integration for building a versatile and reliable molecular assistant in drug discovery.
 
 
 arXiv:2311.16208 , 2023.
 
 

 
 Chen et al. [2024] 
 
Yuhan Chen, Nuwa Xi, Yanrui Du, Haochun Wang, Chen Jianyu, Sendong Zhao, and Bing Qin.
 
 
 From artificially real to real: Leveraging pseudo data from large language models for low-resource molecule discovery.
 
 
 In AAAI , 2024.
 
 

 
 Christofidellis et al. [2023] 
 
Dimitrios Christofidellis, Giorgio Giannone, Jannis Born, Ole Winther, Teodoro Laino, and Matteo Manica.
 
 
 Unifying molecular and textual representations via multi-task language modelling.
 
 
 In ICML , 2023.
 
 

 
 Dauparas et al. [2022] 
 
Justas Dauparas, Ivan Anishchenko, Nathaniel Bennett, Hua Bai, Robert J Ragotte, Lukas F Milles, Basile IM Wicky, Alexis Courbet, Rob J de Haas, Neville Bethel, et al.
 
 
 Robust deep learning–based protein sequence design using proteinmpnn.
 
 
 Science , 2022.
 
 

 
 Devlin and others [2018] 
 
Jacob Devlin et al.
 
 
 Bert: Pre-training of deep bidirectional transformers for language understanding.
 
 
 arXiv:1810.04805 , 2018.
 
 

 
 Du et al. [2022] 
 
Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao.
 
 
 A survey of vision-language pre-trained models.
 
 
 arXiv preprint arXiv:2202.10936 , 2022.
 
 

 
 Edwards et al. [2021] 
 
Carl Edwards, ChengXiang Zhai, and Heng Ji.
 
 
 Text2mol: Cross-modal molecule retrieval with natural language queries.
 
 
 In EMNLP , 2021.
 
 

 
 Edwards et al. [2022] 
 
C. Edwards, Tuan Lai, Kevin Ros, Garrett Honke, Kyunghyun Cho, and Heng Ji.
 
 
 Translation between molecules and natural language.
 
 
 In EMNLP , 2022.
 
 

 
 Fang et al. [2023] 
 
Yin Fang, Xiaozhuan Liang, Ningyu Zhang, Kangwei Liu, Rui Huang, Zhuo Chen, Xiaohui Fan, and Huajun Chen.
 
 
 Mol-instructions: A large-scale biomolecular instruction dataset for large language models.
 
 
 arXiv:2306.08018 , 2023.
 
 

 
 Gao and others [2021] 
 
Tianyu Gao et al.
 
 
 Making pre-trained language models better few-shot learners.
 
 
 In ACL , 2021.
 
 

 
 Guo et al. [2023] 
 
Taicheng Guo, Kehan Guo, et al.
 
 
 What can large language models do in chemistry? a comprehensive benchmark on eight tasks.
 
 
 arXiv:2305.18365 , 2023.
 
 

 
 Hu et al. [2022] 
 
Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.
 
 
 Lora: Low-rank adaptation of large language models.
 
 
 In ICLR , 2022.
 
 

 
 Jumper et al. [2021] 
 
John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al.
 
 
 Highly accurate protein structure prediction with alphafold.
 
 
 Nature , 2021.
 
 

 
 Kim et al. [2023a] 
 
Seojin Kim, Jaehyun Nam, Sihyun Yu, Younghoon Shin, and Jinwoo Shin.
 
 
 Data-efficient molecular generation with hierarchical textual inversion.
 
 
 In NeurIPS 2023 Workshop on AI4D3 , 2023.
 
 

 
 Kim et al. [2023b] 
 
Sunghwan Kim, Jie Chen, Tiejun Cheng, Asta Gindulyte, Jia He, Siqian He, Qingliang Li, Benjamin A Shoemaker, Paul A Thiessen, Bo Yu, et al.
 
 
 Pubchem 2023 update.
 
 
 Nucleic acids research , 2023.
 
 

 
 Kipf and Welling [2017] 
 
Thomas N Kipf and Max Welling.
 
 
 Semi-supervised classification with graph convolutional networks.
 
 
 In ICLR , 2017.
 
 

 
 Lee et al. [2023] 
 
Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi.
 
 
 Rlaif: Scaling reinforcement learning from human feedback with ai feedback.
 
 
 arXiv preprint arXiv:2309.00267 , 2023.
 
 

 
 Li and others [2023] 
 
Junnan Li et al.
 
 
 Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.
 
 
 arXiv:2301.12597 , 2023.
 
 

 
 Li et al. [2023] 
 
Jiatong Li, Yunqing Liu, Wenqi Fan, Xiao-Yong Wei, Hui Liu, Jiliang Tang, and Qing Li.
 
 
 Empowering molecule discovery for molecule-caption translation with large language models: A chatgpt perspective.
 
 
 arXiv:2306.06615 , 2023.
 
 

 
 Li et al. [2024] 
 
Sihang Li, Zhiyuan Liu, et al.
 
 
 Towards 3d molecule-text interpretation in language models.
 
 
 In ICLR , 2024.
 
 

 
 Liang et al. [2023] 
 
Youwei Liang, Ruiyi Zhang, Li Zhang, and Pengtao Xie.
 
 
 Drugchat: towards enabling chatgpt-like capabilities on drug molecule graphs.
 
 
 arXiv:2309.03907 , 2023.
 
 

 
 Liu and others [2023a] 
 
Shengchao Liu et al.
 
 
 Multi-modal molecule structure–text model for text-based retrieval and editing.
 
 
 Nature Machine Intelligence , 2023.
 
 

 
 Liu and others [2023b] 
 
Zequn Liu et al.
 
 
 MolXPT: Wrapping molecules with text for generative pre-training.
 
 
 In ACL , 2023.
 
 

 
 Liu et al. [2022] 
 
Shengchao Liu, Hanchen Wang, Weiyang Liu, Joan Lasenby, Hongyu Guo, and Jian Tang.
 
 
 Pre-training molecular graph representation with 3d geometry.
 
 
 In ICLR , 2022.
 
 

 
 Liu et al. [2023a] 
 
Pengfei Liu, Yiming Ren, and Zhixiang Ren.
 
 
 Git-mol: A multi-modal large language model for molecular science with graph, image, and text.
 
 
 arXiv:2308.06911 , 2023.
 
 

 
 Liu et al. [2023b] 
 
Shengchao Liu, Jiongxiao Wang, Yijin Yang, Chengpeng Wang, Ling Liu, Hongyu Guo, and Chaowei Xiao.
 
 
 Chatgpt-powered conversational drug editing using retrieval and domain feedback.
 
 
 In ICML 2023 Workshop on SynS ML , 2023.
 
 

 
 Liu et al. [2023c] 
 
Shengchao Liu, Yutao Zhu, Jiarui Lu, Zhao Xu, Weili Nie, Anthony Gitter, Chaowei Xiao, Jian Tang, Hongyu Guo, and Anima Anandkumar.
 
 
 A text-guided protein design framework.
 
 
 arXiv:2302.04611 , 2023.
 
 

 
 Liu et al. [2023d] 
 
Zhiyuan Liu, Sihang Li, Yanchen Luo, Hao Fei, Yixin Cao, Kenji Kawaguchi, Xiang Wang, and Tat-Seng Chua.
 
 
 Molca: Molecular graph-language modeling with cross-modal projector and uni-modal adapter.
 
 
 In EMNLP , 2023.
 
 

 
 Lo et al. [2019] 
 
Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Dan S Weld.
 
 
 S2orc: The semantic scholar open research corpus.
 
 
 arXiv:1911.02782 , 2019.
 
 

 
 Luo et al. [2023] 
 
Yizhen Luo, Jiahuan Zhang, Siqi Fan, Kai Yang, Yushuai Wu, Mu Qiao, and Zaiqing Nie.
 
 
 Biomedgpt: Open multimodal generative pre-trained transformer for biomedicine.
 
 
 arXiv:2308.09442 , 2023.
 
 

 
 Luo et al. [2024] 
 
Yizhen Luo, Xing Yi Liu, Kai Yang, Kui Huang, Massimo Hong, Jiahuan Zhang, Yushuai Wu, and Zaiqing Nie.
 
 
 Towards unified ai drug discovery with multiple knowledge modalities.
 
 
 In AAAI , 2024.
 
 

 
 Oord et al. [2018] 
 
Aaron van den Oord, Yazhe Li, and Oriol Vinyals.
 
 
 Representation learning with contrastive predictive coding.
 
 
 arXiv:1807.03748 , 2018.
 
 

 
 Ouyang et al. [2022] 
 
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
 
 
 Training language models to follow instructions with human feedback.
 
 
 In NeurIPS , 2022.
 
 

 
 Ouyang et al. [2023] 
 
Siru Ouyang, Zhuosheng Zhang, Bing Yan, Xuan Liu, Jiawei Han, and Lianhui Qin.
 
 
 Structured chemistry reasoning with large language models.
 
 
 arXiv:2311.09656 , 2023.
 
 

 
 Pei et al. [2023] 
 
Qizhi Pei, Wei Zhang, Jinhua Zhu, Kehan Wu, Kaiyuan Gao, Lijun Wu, Yingce Xia, and Rui Yan.
 
 
 Biot5: Enriching cross-modal integration in biology with chemical knowledge and natural language associations.
 
 
 In EMNLP , 2023.
 
 

 
 Qian et al. [2023] 
 
Yujie Qian, Zhening Li, Zhengkai Tu, Connor Coley, and Regina Barzilay.
 
 
 Predictive chemistry augmented with text retrieval.
 
 
 In EMNLP , 2023.
 
 

 
 Raffel et al. [2020] 
 
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.
 
 
 Exploring the limits of transfer learning with a unified text-to-text transformer.
 
 
 The Journal of Machine Learning Research , 2020.
 
 

 
 Rubungo et al. [2023] 
 
Andre Niyongabo Rubungo, Craig Arnold, Barry P Rand, and Adji Bousso Dieng.
 
 
 Llm-prop: Predicting physical and electronic properties of crystalline solids from their text descriptions.
 
 
 arXiv:2310.14029 , 2023.
 
 

 
 Sakhinana and Runkana [2023] 
 
Sagar Sakhinana and Venkataramana Runkana.
 
 
 Crossing new frontiers: Knowledge-augmented large language model prompting for zero-shot text-based de novo molecule design.
 
 
 In NeurIPS 2023 Workshop on R0-FoMo , 2023.
 
 

 
 Seidl et al. [2023] 
 
Philipp Seidl, Andreu Vall, Sepp Hochreiter, and Günter Klambauer.
 
 
 Enhancing activity prediction models in drug discovery with the ability to understand human language.
 
 
 In ICML , 2023.
 
 

 
 Shi et al. [2023] 
 
Yaorui Shi, An Zhang, Enzhi Zhang, Zhiyuan Liu, and Xiang Wang.
 
 
 Relm: Leveraging language models for enhanced chemical reaction prediction.
 
 
 In EMNLP , 2023.
 
 

 
 Su et al. [2022] 
 
Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, and Ji-Rong Wen.
 
 
 A molecular multimodal foundation model associating molecule graphs with natural language.
 
 
 arXiv:2209.05481 , 2022.
 
 

 
 Tang et al. [2023] 
 
Xiangru Tang, Andrew Tran, Jeffrey Tan, and Mark B Gerstein.
 
 
 Mollm: A unified language model to integrate biomedical text with 2d and 3d molecular representations.
 
 
 bioRxiv , 2023.
 
 

 
 Taylor et al. [2022] 
 
Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic.
 
 
 Galactica: A large language model for science.
 
 
 arXiv:2211.09085 , 2022.
 
 

 
 Vaswani et al. [2017] 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
 
 
 Attention is all you need.
 
 
 In NeurIPS , 2017.
 
 

 
 Wang et al. [2023] 
 
Zeyuan Wang, Qiang Zhang, Keyan Ding, Ming Qin, Xiang Zhuang, Xiaotong Li, and Huajun Chen.
 
 
 Instructprotein: Aligning human and protein language via knowledge instruction.
 
 
 arXiv:2310.03269 , 2023.
 
 

 
 Wang et al. [2024] 
 
Zifeng Wang, Zichen Wang, Balasubramaniam Srinivasan, Vassilis N. Ioannidis, Huzefa Rangwala, and RISHITA ANUBHAI.
 
 
 Biobridge: Bridging biomedical foundation models via knowledge graph.
 
 
 In ICLR , 2024.
 
 

 
 Wei et al. [2022a] 
 
Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, et al.
 
 
 Finetuned language models are zero-shot learners.
 
 
 In ICLR , 2022.
 
 

 
 Wei et al. [2022b] 
 
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
 
 
 Chain-of-thought prompting elicits reasoning in large language models.
 
 
 In NeurIPS , 2022.
 
 

 
 Wellawatte and Schwaller [2023] 
 
Geemi Wellawatte and Philippe Schwaller.
 
 
 Extracting human interpretable structure-property relationships in chemistry using XAI and large language models.
 
 
 In NeurIPS 2023 Workshop on XAI in Action , 2023.
 
 

 
 Wu et al. [2018] 
 
Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, and Vijay Pande.
 
 
 Moleculenet: A benchmark for molecular machine learning, 2018.
 
 

 
 Xu et al. [2023] 
 
Minghao Xu, Xinyu Yuan, Santiago Miret, and Jian Tang.
 
 
 Protst: Multi-modality learning of protein sequences and biomedical texts.
 
 
 In ICML , 2023.
 
 

 
 Yenduri et al. [2023] 
 
Gokul Yenduri, Ramalingam M, Chemmalar Selvi G, Supriya Y, Gautam Srivastava, Praveen Kumar Reddy Maddikunta, Deepti Raj G, Rutvij H Jhaveri, Prabadevi B, Weizheng Wang, Athanasios V. Vasilakos, and Thippa Reddy Gadekallu.
 
 
 Generative pre-trained transformer: A comprehensive review on enabling technologies, potential applications, emerging challenges, and future directions, 2023.
 
 

 
 You et al. [2020] 
 
Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen.
 
 
 Graph contrastive learning with augmentations.
 
 
 In NeurIPS , 2020.
 
 

 
 Zachares and others [2023] 
 
Peter A. Zachares et al.
 
 
 Form follows function: Text-to-text conditional graph generation based on functional requirements.
 
 
 arXiv:2311.00444 , 2023.
 
 

 
 Zeng et al. [2022] 
 
Zheni Zeng, Yuan Yao, Zhiyuan Liu, and Maosong Sun.
 
 
 A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals.
 
 
 Nature communications , 2022.
 
 

 
 Zhang et al. [2023] 
 
Weitong Zhang, Xiaoyun Wang, Weili Nie, Joe Eaton, Brad Rees, and Quanquan Gu.
 
 
 MoleculeGPT: Instruction following large language models for molecular property prediction.
 
 
 In NeurIPS 2023 Workshop on AI4D3 , 2023.
 
 

 
 Zhao et al. [2023a] 
 
Haiteng Zhao, Shengchao Liu, Chang Ma, Hannan Xu, Jie Fu, Zhi-Hong Deng, Lingpeng Kong, and Qi Liu.
 
 
 GIMLET: A unified graph-text model for instruction-based molecule zero-shot learning.
 
 
 In NeurIPS , 2023.
 
 

 
 Zhao et al. [2023b] 
 
Lawrence Zhao, Carl Edwards, and Heng Ji.
 
 
 What a scientific language model knows and doesn’t know about chemistry.
 
 
 In NeurIPS 2023 AI for Science Workshop , 2023.
 
 

 
 Zhou and others [2023] 
 
Denny Zhou et al.
 
 
 Least-to-most prompting enables complex reasoning in large language models.
 
 
 In ICLR , 2023.