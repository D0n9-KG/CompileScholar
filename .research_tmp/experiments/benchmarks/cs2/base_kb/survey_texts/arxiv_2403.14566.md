A survey on Concept-Based Approaches for Model Improvement 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.14566v2 [cs.AI] 23 Mar 2024 
 
 

# A survey on Concept-Based Approaches for Model Improvement

 
 
 Avani Gupta
 
    
 P J Narayanan
 
 Affiliation:  CVIT, KCIS
 
 Affiliation:  IIIT-Hyderabad, India
 
 Affiliation:  {avani.gupta}@research.iiit.ac.in, pjn@iiit.ac.in 
 

 Abstract 
 
 The focus of recent research has shifted from merely improving the metrics based performance of Deep Neural Networks (DNNs) to DNNs which are more interpretable to humans. The field of eXplainable Artificial Intelligence (XAI) has observed various techniques, including saliency-based and concept-based approaches. These approaches explain the model’s decisions in simple human understandable terms called Concepts . Concepts are known to be the thinking ground of humans .
Explanations in terms of concepts enable detecting spurious correlations, inherent biases, or clever-hans. With the advent of concept-based explanations, a range of concept representation methods and automatic concept discovery algorithms have been introduced. Some recent works also use concepts for model improvement in terms of interpretability and generalization. We provide a systematic review and taxonomy of various concept representations and their discovery algorithms in DNNs, specifically in vision. We also provide details on concept-based model improvement literature marking the first comprehensive survey of these methods.

 
 
 

## 1 Introduction

 
 With the increasing use of deep learning in various tasks, there is rapid growth in the XAI literature. One of the most promising methods of XAI is concept-based approaches, which explain model predictions in human-understandable units called Concepts [ 52 ] . Concepts are linked to their origin in NeuroScience Literature, in which they are defined as ideas derived or inferred from facts [ 28 ] and are known to hold the human’s mental world together [ 83 ] . It has been observed that humans learn by creating concept models of things around them, helping them gather generalized knowledge [ 65 ] . There are various spurious correlations which creep in DNNs due to their black box nature. Model explanations in terms of concepts not only lead to better interpretability but also help to identify these spurious correlations and inherent biases also known as Clever-hans.

 
 
 In Computer Vision, concept identification started with disentangled representations of latent space [ 58 ] where it was observed that similar areas in latent space denote images with some common properties (for example, red cars were found closer). While in Natural Language Processing, Word Embeddings [ 82 , 86 ] inspired by the idea that similar words should be grouped together in latent space of model (also known as spatially co-located similarity property) have been widely used.

 
 
 The interpretability research aims to explain the model’s decisions in simple human understandable terms [ 136 ] .
Concepts, by their definition, are humanly understandable and, hence, a good choice for interpretability. Recent XAI literature has observed a surge in concept-oriented explanations. There have been many methods that derive concept-based explanations from pre-trained models (post-hoc), while some do it while training the model (ante-hoc).
 Kim et al. [56] paved the way for post-hoc concept-based explanations of models using Concept Activation Vectors, while a parallel work [ 30 ] explained Convolutional Neural Networks (CNN’s) filters in terms of concepts. Many works [ 11 , 24 , 20 , 18 ] build on the above concept representations in a post-hoc manner. Concept Bottleneck Models (CBMs) [ 59 ] derive concept explanations ante-hoc, putting the weightage of concept explanations on the model while training. Recent work by Yuksekgonul et al. [130] extended CBMs to post-hoc explanations.

 
 
 Concepts are essential for both human-like learning and better generalization by models. A natural intuition for human-like learning is making the model match the human attention for several tasks as done by [ 102 , 33 , 69 , 44 , 45 ] . One major limitation of such approaches is the requirement of additional human-level attention data.
 Chang [19] formalize Concept Oriented Deep Learning (CODL) for human alike learning involving concepts. They argue that CODL overcomes several limitations of deep learning, involving concerns of interpretability, transferability, contextual adaptation, and humongous training data requirements. For the scope of this work, we focus on concept-based literature and refrain from providing generic definitions of interpretability/explainability (or XAI) literature, referring the readers to [ 136 , 29 , 70 , 119 , 3 ] which present surveys on XAI methods.
Another concept-inspired learning paradigm is Neuro-Symbolic learning, which involves learning like humans using symbolic reasoning [ 75 ] .

 
 
 With Concepts spanning Neuroscience literature to interpretability literature, there are vast representations of concepts varying from concept activation vectors, concept trees, and graphs to concept activation areas in latent space. While some of the methods require concept definitions to be given by the user, several works automate this by extracting concepts from models [ 36 , 120 ] .

 
 
 Additionally, some recent methods use concepts for improving model performance by encoding prior knowledge or intervening and correcting the model to ensure it uses correct concepts [ 93 , 92 ] . We provide a systematic survey knitting this vast literature involving concepts.

 
 
 Our Key contributions include the following:

 
 • 
 
 A systematic survey knitting various concept representations.

 

 • 
 
 Taxonomy on concept discovery approaches and their metrics along with commonly used datasets.

 

 • 
 
 Finally, a first-of-its-kind survey on CODL techniques.

 

 
 
 
 

## 2 Related work

 
 Hitzler and Sarker [46] present a survey on concept-based explainability techniques but are restricted to Concept Activation Vector (CAV) [ 56 ] representations. Schwalbe [97] survey various visual concept embeddings giving a taxonomy of concept analysis approaches and various datasets used for supervision of concept learning. A parallel work [ 47 ] also surveys knowledge representations mapping to concepts and classifies concept-based XAI into Associations (what if I see), Interventions (what if I do), and Counterfactuals (what if I had done).

 
 
 Gao et al. [32] , Weber et al. [123] , Hase and Bansal [43] survey various XAI-based approaches for model improvement.
We categorize XAI-based model improvement in terms of the end goal as (i) Enhanced interpretability and (ii) Better Generalization. Most of the existing literature focuses on the former (i.e., enhancing interpretability of models) [ 59 , 53 , 7 , 113 , 92 , 93 , 111 , 64 , 110 , 114 , 31 , 54 , 126 , 121 , 89 ] while some works address the latter (using interpretability to aid more generalizable models) [ 31 , 8 , 10 , 111 , 107 , 18 , 99 , 104 , 94 , 62 , 17 ] . Such division concurs with Holmberg et al. [47] , who found that explanations with better understanding significantly differ from explanations with the goal of finding actionable decisions.
 Gao et al. [32] classify XAI-based model improvement in terms of the type of explanation method used while Weber et al. [123] classify based on augmentation made for improvement, which can be over data, loss, gradient, model, etc. Explanation Guided Learning (EGL) involves using explanations for model improvement to meet one or both of the above-listed goals (i, ii). Various local EGL techniques are surveyed by Gao et al. [32] . Explanatory Interactive Learning (XIL) involves interactively using explanations with human feedback to improve the model. Friedrich et al. [31] survey various XIL techniques using local XAI methods to remove shortcuts in the models.
eXplanatory Active Learning (XAL) involves using Explanations in an Active Learning framework as surveyed by Ghai et al. [34] .

 
 
 Hase and Bansal [43] survey XAI based model improvement methods in Natural Language Processing while we focus on such methods in Computer Vision, specifically visual concepts-based model improvement.
 [ 14 ] survey model improvement methods aiming better explainability (type ii) using Prior knowledge.
 Chang [20] provide framework for CODL [ 19 ] using Concept Graphs.
On the other hand, we provide a novel taxonomy of concept representations and a categorization of various Concept Discovery Algorithms and CODL. We restrict ourselves to visual representations of concepts and show applications of CODL in vision.
We organize our survey based on representations for Concepts used by the existing works followed by concept discovery methods, concept-based model improvement, evaluation of discovered concepts, and evaluation by concepts.

 
 
 Figure 1 : We focus on Concept-based XAI approaches and model improvement approaches 
 
 
 

## 3 Various Concept Representations

 
 The landscape of model interpretability via the lens of concept-based explanations, is rich and varied. This section navigates through various concept representations methods highlighting their unique contributions and applications in deep learning models.

 
 
 Table 1 : Concept Representations of existing ML interpretability literature 
 
 
 
 Concept type | 
 Concept Representation Method | 
 
 
 Papers 
 | 

 
 
 
 Non-hierarchial | 
 Hidden Units Concept images alignment based | 
 
 
 Network Dissection [ 13 ] 
 | 

 
 Concept Embeddings Vectos based | 
 
 
 CAV [ 56 ] , A-CAV [ 108 ] , CG [ 11 ] , ICB [ 137 ] , ICE [ 135 ] , CaCE Goyal et al. [37] , ICS [ 95 ] , ConceptSHAP
 [ 128 ] , [ 24 ] 
 | 

 
 Proto-types | 
 
 
 Li et al. [68] , Chen et al. [23] , Bontempelli et al. [18] 
 | 

 
 Neuro-Symbolic | 
 
 
 RRC [ 111 ] , COOL [ 76 ] , RRR [ 111 ] ,
NeuroSC Marconato et al. [77] , 
 | 

 
 Others | 
 
 
 Multi Agent Debate [ 60 ] 
 | 

 
 Hierarchial | 
 Neuron Attribution | 
 
 
 HINT [ 120 ] 
 | 

 
 Weight Attribution | 
 
 
 Concept Graphs: Kori et al. [61] , Zhang et al. [134] , Decision Trees: POEM [ 26 ] , CHAIN [ 122 ] 
 | 

 
 Symbolic features | 
 
 
 Concept trees: Santhirasekaram et al. [90] , CNN2DT [ 49 ] , TreeICE [ 84 ] , ACDTE [ 101 ] 
 | 

 

 
 

### 3.1 Non-Hierarchial Concept Representations

 

#### 3.1.1 Hidden Units Concept images alignment based 

 
 Network Dissection [ 13 ] uses user-provided concept sets and checks their alignment with individual hidden units at each layer of CNN. Since DNNs are expected to learn partially non-local representations in denser layers, concepts can align with a combination of several hidden units. However, to assess disentanglement, they focus on measuring the alignment of concepts with single units. This method hypothesizes that the interpretability of units is equivalent to their random linear combination. They evaluate every individual convolutional unit in CNN as a solution to binary segmentation tasks for each visual concept. They pass in all concept sets from the model and determine the distribution of concept activations a k a_{k} for each convolutional unit k k .
They then determine top quantile level T k T_{k} for each unit k k such that P ⁡ ( a k T k ) = 0.0005 P(a_{k} T_{k})=0.0005 over every spatial location of the activation map in the concept set. This is followed by a selection of regions exceeding the threshold T k T_{k} and evaluating segmentations with every concept c c in the concept set by computing the intersection over union ( I ​ o ​ U k , c IoU_{k,c} ) of the above-selected regions with input concept annotation masks. A concept detector is reported if I ​ o ​ U k , c IoU_{k,c} is greater than a certain threshold. The interpretability of a layer is quantified by the number of unique concepts aligned with its units ( unique detectors ). Since IoU is an objective measure (not relative), it enables comparison of interpretability across networks.

 
 
 

#### 3.1.2 Concept Embeddings Vectors Based 

 
 Fong and Vedaldi [30] proposed Net2Vec, which aligns the concepts with CNN filters.
They first collect the pre-trained model’s activations for a concept dataset (probe) and then learn the weights to recognize the concept in various semantic tasks. These weights are interpreted as concept embeddings and analyzed to gain insights into how concepts are encoded in the network.

 
 
 

#### 3.1.3 Concept Activation Vectors Based 

 
 Concept Activation Vectors (CAV) [ 56 ] , a widely popular approach in Concet Based XAI, is encoded as the normal to the boundary separating the activations of the concept of interest and their negative counterpart in the model’s activation plane.
For a human-provided concept of interest, C C , and its negative counterpart (or random concept) C’, a linear classifier is trained to distinguish between the model activations for a layer l l given as f l f_{l} for C C and C ′ C^{\prime} 
The CAV v C l v_{C}^{l} is given as a normal to the decision boundary obtained by the above linear classifier.
 v C l v_{C}^{l} gives the direction of the concept in the model’s activation plane.
 Kim et al. [56] further uses directional derivatives to gauze the importance given by the model to concept C C for the prediction of class k k .
The sensitivity S C , k , l S_{C,k,l} of the model towards C C for class k k is given as the directional derivative of the model’s prediction logit h k ​ ( x ) h_{k}(x) wrt the sample’s activation f l ​ ( 𝒙 ) f_{l}(\boldsymbol{x}) in the direction of CAV:

 

 
 | 
 S C , k , l ​ ( 𝒙 ) = ∇ h l , k ​ ( f l ​ ( 𝒙 ) ) ⋅ 𝒗 C l S_{C,k,l}(\boldsymbol{x})=\nabla h_{l,k}\left(f_{l}(\boldsymbol{x})\right)\cdot\boldsymbol{v}_{C}^{l} | 
 | 
 (1) | 
 

 The sign of S C , k , l ​ ( 𝒙 ) S_{C,k,l}(\boldsymbol{x}) tells whether C C had an effect over the class sample x x . This sign is aggregated for all class examples to give the final T ​ C ​ A ​ V TCAV score of the model.

 
 
 Due to their simple and elegant vector-based representation of concepts, CAVs got widely popular and had several works built over them. We now discuss a similar work to CAV followed by their extensions over time.

 
 
 A similar work: ICB 

 
 Interpretable Concept Basis (ICB) [ 137 ] uses the weights w k w_{k} of the second to last layer (layer before logits) for class of interest k k , and decomposes them in terms of concept vectors c j c_{j} as w k ≈ α k ​ 1 ​ c 1 + α k ​ 2 ​ c 2 + … + α k ​ m ​ c m w_{k}\approx\alpha_{k1}c_{1}+\alpha_{k2}c_{2}+\ldots+\alpha_{km}c_{m} where m m is the total number of user-defined concepts and α k , j \alpha_{k,j} are non-negative weights with sparsity constrained to be less than m m . It thus disentangles and quantifies the contribution of each concept in the model’s prediction.

 
 
 
 ICE: A better approach for fidelity 

 
 Invertible Concept-Based Explanation (ICE) [ 135 ] proposes Non-negative CAVs (NCAVs) based on Non-negative Matrix Factorization (NMF), which provides better interpretability and fidelity.
They flatten f l ​ ( x c ) f_{l}(x_{c}) across channel dimention to get V ∈ R ( n × h × w ) × c V\in R^{(n\times h\times w)\times c} . They use NMF to reduce channel dimensions c c to c ′ c^{\prime} and optimize such that V V can be approximated as a product of the reduced channel dimension vector V ′ ∈ R ( n × h × w ) × c ′ V^{\prime}\in R^{(n\times h\times w)\times c^{\prime}} and feature direction.

 

 
 | 
 min S , P ⁡ ‖ V − S ​ P ‖ F  s.t.  S ≥ 0 , P ≥ 0 \min_{S,P}\|V-SP\|_{F}\quad\text{ s.t. }\quad S\geq 0,P\geq 0 | 
 | 
 

 Factorization on these vectors helps in disentangling important directions for target concepts. P P denotes the meaningful NCAV and is fixed after being trained with f l ​ ( x c ) f_{l}(x_{c}) for some images. Given P P , NMF can be applied over it to get S S .
 P P denotes a vector basis and S S is the projection lengths of a vector A A onto these basis directions. In this context, S S can be interpreted as feature scores that measure the degree of similarity between vector A A and the basis directions in P P . Essentially, S S serves as an indication of how much vector A A is related or associated with the NCAVs (non-collinear basis vectors) in P P .
They provide both local and global concept-level explanations for CNN.

 
 
 
 Extention to Non-linearity 

 
 CAVs are dependent on the choice of random concept images and the ability of the linear classifiers to distinguish between C C and C ′ C^{\prime} . Sometimes C C and C ′ C^{\prime} can be non-linear, which cannot be handled by CAV. There are two extensions of CAV to non-linearly separable concepts.

 
 
 Adversarial TCAV (A-CAV) Soni et al. [108] propose A-CAV, which is more effective in terms of retrieval of concept images using CAVs (improving recall) and is more robust to variations in random examples.
A-CAV handles the non-linear separability of concepts by pushing the activation vectors away from the decision boundary, adding a small perturbation in the input vector along the direction of its gradient.
A-CAV use two-step Gram-Schmidt Orthogonalization process [ 16 ] to separate the f l ​ ( x C ) f_{l}(x_{C}) and f l ​ ( x C ′ ) f_{l}(x_{C}^{\prime}) for all x C ∈ C x_{C}\in C and x C ′ ∈ C ′ x_{C}^{\prime}\in C^{\prime} .
They compute a concept basis, which is the orthogonal basis of subspace-spanned by CAVs, and a non-concept basis, which is disjoint of the concept basis. With concept activations as positive samples, they generate negative samples used for training binary classifiers by sampling activations from a non-concept basis.
To make the CAVs robust to random concept examples, A-CAV trains multiple linear models on different random samples from a non-concept basis and takes the centroid direction of learned coefficients as CAV. Additionally, A-CAV can prevent adversarial attacks and help investigate bias in the model.

 
 
 Concept Gradients (CG) Bai et al. [11] propose CG for surpassing the linearity constraint by CAVs.
Contrary to [ 56 ] which uses model’s activation space R a R^{a} 
to calculate CAV’s, Bai et al. [11] train a concept prediction layer g : ℝ a → ℝ c g:\mathbb{R}^{a}\rightarrow\mathbb{R}^{c} which maps R a R^{a} to a different space corresponding concepts R c R^{c} .
It then calculates the CG as

 

 
 | 
 CG l , k ( f l ( x ) ) = ∇ g l ( f l ( x ) ) † ∇ h l , k ( f l ( x ) ) \mathrm{CG}_{l,k}(f_{l}(x))=\nabla g_{l}(f_{l}(x))^{\dagger}\nabla h_{l,k}(f_{l}(x)) | 
 | 
 

 where, ∇ g ( x ) l † ∈ ℝ m × d \nabla g(x)_{l}^{\dagger}\in\mathbb{R}^{m\times d} is pseudo-inverse of g l ​ ( f l ​ ( x ) ) g_{l}(f_{l}(x)) and measures the effect of small changes in each concept dimension on model activation f l ​ ( x ) f_{l}(x) while ∇ h l , k ​ ( f l ​ ( x ) ) \nabla h_{l,k}(f_{l}(x)) measures the effect small changes in models activations f l ​ ( x ) f_{l}(x) on the final prediction h l , k h_{l,k} . Thus, the C ​ G CG measures the concept space changes wrt activation space and output space changes wrt activation space, thereby giving changes in outputs wrt concept (gauging sensitivity of model for concepts).

 
 
 
 Moving from Correlation measure to Causality 

 
 Goyal et al. [37] argue that TCAV scores [ 56 ] can lead to wrong intuitions when concepts are correlated. For example, in the BARS [ 37 ] dataset, the vertical and horizontal lines are denoted as two labels, while color is confounded with the orientation of lines. ( e.g .vertical lines are red, and horizontal lines are green). In this case, even though the model learned to predict the lines, the T ​ C ​ A ​ V TCAV score will be high for concept color due to confounding.
 Goyal et al. [37] overcome this by bringing Causality into picture. They generate counterfactual examples of the concept of interest and check the causal effect of concept c c given by the Causal Concept Effect (CaCE):

 

 
 | 
 CaCE ( 𝐜 , f ) = 𝔼 [ f ( 𝐱 ∣ do ( 𝐜 = 1 ) ] − 𝔼 [ f ( 𝐱 ∣ do ( 𝐜 = 0 ) ] \operatorname{CaCE}(\mathbf{c},f)=\mathbb{E}[f(\mathbf{x}\mid\operatorname{do}(\mathbf{c}=1)]-\mathbb{E}[f(\mathbf{x}\mid\operatorname{do}(\mathbf{c}=0)] | 
 | 
 

 .

 
 
 
 Combination with local method 

 
 Schrouff et al. [95] combine ther global TCAV method [ 56 ] with the local explanations captured by Integrated Gradient [ 112 ] (IG) to give T ​ C ​ A ​ V I ​ C ​ S TCAV_{ICS} .
They take the projection of IG along v c l v_{c}^{l} . Originally, IG used a baseline image b b that is conceptually neutral. In object recognition tasks, this image can be white or gray.

 
 
 

 
 | 
 ICS C k ⁡ ( 𝐚 , 𝐚 ′ ) := ( f l ​ ( 𝐱 ) − 𝐚 ′ ) T ​ 𝐯 𝐂 ​ ∫ [ 𝐚 ′ , f l ​ ( 𝐱 ) ] ∇ 𝐯 𝐂 h k ​ ( 𝐚 ) ​ 𝑑 𝐚 \operatorname{ICS}_{C}^{k}\left(\mathbf{a},\mathbf{a}^{\prime}\right):=\left(f_{l}(\mathbf{x})-\mathbf{a}^{\prime}\right)^{T}\mathbf{v}_{\mathbf{C}}\int_{\left[\mathbf{a}^{\prime},f_{l}(\mathbf{x})\right]}\nabla_{\mathbf{v}_{\mathbf{C}}}h_{k}(\mathbf{a})d\mathbf{a} | 
 | 
 

 where a ′ = f l ​ ( b ) a^{\prime}=f_{l}(b) .
ICS authors also add two new baselines (i) Concept forgetting given by a ′ = a − λ ​ 𝐯 𝐂 a^{\prime}=a-\lambda\mathbf{v}_{\mathbf{C}} and (ii) concept occluding baseline given by 𝐚 ′ = 𝐚 − ( 𝐚 T ​ 𝐯 C + b ) ​ 𝐯 C ‖ 𝐯 C ‖ 2 2 \mathbf{a}^{\prime}=\mathbf{a}-\left(\mathbf{a}^{T}\mathbf{v}_{\mathrm{C}}+b\right)\frac{\mathbf{v}_{\mathrm{C}}}{\left\|\mathbf{v}_{\mathrm{C}}\right\|_{2}^{2}} .

 
 
 
 Completeness via Shapeley values 

 
 Completeness measures how sufficient a particular set of concepts is for explaining the model’s prediction.
ConceptSHAP [ 128 ] measures the Completeness of concepts by using Shapeley values associated with cooperative game theory.

 
 
 
 Latent Space Concept Disentanglement 

 
 Concept Whitening [ 24 ] argues that CAVs might not give a correct picture of concepts’ proximity due to latent space being correlated. Thus, they disentangle concepts by decorrelating the latent space and projecting concepts such that they align in different directions, preferably basis directions.

 
 
 
 

#### 3.1.4 Proto-type based 

 
 Li et al. [68] , Chen et al. [23] proposed Networks that dissect an image by finding prototypical parts and reason over obtained prototypes for classification. They add a new layer after the second last layer to learn proto-types.
 Bontempelli et al. [18] use interactions with humans for debugging such prototypical networks.
Humans tell which part-prototypes are to be used or not used, and the model learns based on it. The part-prototypes might not be human interpretable always, and hence Bontempelli et al. [18] enable human users to provide additional concepts for supervision. They demonstrate results on both concept-level and instance-level debugging.
The original model sensitivity calculation is done for the final layer (final layer sensitivity to intermediate layer outputs). [ 39 ] introduce proto-types for intermediate sensitivity calculations.

 
 
 

#### 3.1.5 Neuro-symbolic 

 
 RRC [ 111 ] , COOL [ 76 ] , and RRR [ 111 ] are neuro-symbolic concept learners who use reasoning modules to determine and train for concepts. RRR [ 111 ] uses a concept embedding module that learns concepts via slot attention [ 72 ] and a reasoning module that reasons for the concepts. The concept embedding module creates a decomposed representation of input space which can be mapped to concepts, while the reasoning module makes predictions based on the above-obtained concepts.
They stack the reasoning module after the concept embedding module, deriving the reasoning module’s explanation from the given final model’s prediction output, while the concept module’s explanation is derived from the reasoning module’s explanation.
This training helps them to come up with decomposed input representations depicting neuro-symbolic concepts learned by the model. For the reasoning module, they use a Set Transformer [ 67 ] to make the reasoning insensitive to the order of concepts and generate the explanations with the help of Integrated Gradients [ 112 ] . They introduce Neuro Symbolic Continual Learning which involves solving neuro-symbolic tasks mapping sub-symbolic inputs to high-level concepts, and making predictions via reasoning with prior knowledge. They solve catastrophic forgetting with COOL: Concept-level continual Learning, acquiring concepts in a continual manner (retaining over time). Note that concept-level details remain constant over time for various tasks, which is exploited by COOL to avoid catastrophic forgetting.

 
 
 

#### 3.1.6 Multi-agent debate 

 
 Kori et al. [60] pose concept learning as a multi-agent debate problem where two agents come up with arguments (explanations on the selection of specific features in the model’s discretized latent space) to support or contradict the classifier’s decision.

 
 
 
 

### 3.2 Hierarchical Concept Representations

 
 Murphy [83] present a taxonomy of various hierarchies of concepts in the neuroscience literature.
We, on the other hand, review the use of Hierarchical Concept Representations in interpretability literature.

 
 

#### 3.2.1 Neuron Attribution based 

 
 HINT [ 120 ] attributes each concept to various neurons, building bidirectional concept-neuron associations. They consider hierarchical relationships between concepts (eg: dog and cat belong to animals) and attribute them to a set of neurons. HINT also indicates how the neurons learn the hierarchical relationships of categories.
This is done by identifying concept responsible regions in the input image followed by constructing a dataset containing a collection of responsible regions r e r_{e} for each concept e ∈ e ​ p e\in ep and a collection of background regions r b ∗ r_{b}* .
Following this, a concept classifier L e L_{e} is trained, which separates concept e e from other concepts and a Shapeley value [ 100 ] based approach to calculate the contributions of each neuron to the concepts.
 𝒓 ℰ \ e ∪ 𝒓 b ∗ \boldsymbol{r}_{\mathcal{E}\backslash e}\cup\boldsymbol{r}_{b^{*}} .

 

 
 | 
 L e ​ ( r ) = σ ⁡ ( 𝜶 T ​ r ) ​  where  ​ r = z 𝒟 , i , j ∈ ℝ | 𝒟 | . L_{e}(r)=\sigma\left(\boldsymbol{\alpha}^{T}r\right)\text{ where }r=z_{\mathcal{D},i,j}\in\mathbb{R}^{|\mathcal{D}|}. | 
 | 
 

 

 
 | 
 ϕ = ∑ 𝒓 | ∑ i = 1 M ( L e ⟨ 𝒮 ∪ d ⟩ ​ ( r ) − L e ⟨ 𝒮 ⟩ ​ ( r ) ) | M ​ | 𝒓 ℰ ∪ 𝒓 b ∗ | \phi=\frac{\sum_{\boldsymbol{r}}\left|\sum_{i=1}^{M}\left(L_{e}^{\langle\mathcal{S}\cup d\rangle}(r)-L_{e}^{\langle\mathcal{S}\rangle}(r)\right)\right|}{M\left|\boldsymbol{r}_{\mathcal{E}}\cup\boldsymbol{r}_{b^{*}}\right|} | 
 | 
 

 Shapeley scores are calculated for all pairs of e e , and d d to obtain a score matrix Φ n ​ x ​ c \Phi^{nxc} where n n is the number of neurons and c c is the number of concepts in the hierarchy.
Finally, collaborative neurons responsible for a concept e e are identified by taking the top k k corresponding column highest Shapeley value neurons(top k k in Φ y , e , y ∈ { 0 , n } \Phi_{y,e},y\in\{0,n\} )
The above score matrix enables checking for multi-modality of neurons (encodes multiple concepts) or attaining a hierarchy of concepts a neuron is responsible for.

 
 
 

#### 3.2.2 Concept Graphs 

 
 Kori et al. [61] use concept graphs where the nodes are the concepts and edges depict relationships between them.
They find weights responsible for detecting concepts in the input image by clustering them according to a metric. Then, they use Grad-CAM attributions of clustered weights over input images to determine the concepts detected by them. They also perform significance tests to check the robustness, consistency, and localization of concepts.
Once concept identification is done, the relationships between them are to be identified to attain a human-understandable trace. This is done using mutual information between pre-interventional and post-interventional feature map distributions. Suppose we want to find if an edge exists between the cluster of a previous layer (say C ​ 2 C2 ) and the cluster of the next layer ( B ​ 2 B2 ). The pre-interventional intervention tells the information flowing from C ​ 2 C2 to all clusters of the next layer ( B ​ 1 B1 , B ​ 2 B2 ,.. B ​ n Bn ).
The post-interventional intervention tells the information flowing from a C ​ 2 C2 to B ​ 2 B2 . The mutual information between these distributions is used to detect the link C ​ 2 − B ​ 2 C2- B2 .

 
 
 Microsoft Concept Graph (MCR) [ 48 ] contains 5.4 million Natural Language concepts having a hierarchy (sub-concepts)and describes each concept by a set of attributes and an ample relationship space of ("isA," "locatedIn").
MCR is built upon probase [ 124 ] , which consists of syntactical extractions learned automatically from billions of web pages.

 
 
 Zhang et al. [134] learn an explanatory graph from a pre-trained CNN in an unsupervised manner, disentangling different object part patterns in each CNN filter.
In the explanatory graph, each node represents a part pattern, and each edge encodes co-activation and spatial relationships between patterns. They
demonstrate that each graph node consistently represents the same object part (concept) through different images.

 
 
 Pattern-Oriented Explanations of CNN Models (POEM) [ 26 ] identifies patterns of "if Concept c c then class k k in CNN’s. It does concept identification using Network Dissection [ 13 ] followed by semantic segmentation via Unified Perceptual Parsing [ 125 ] involving concept identification with a pre-trained model. POEM attributes concepts to model predictions using three conditions: (i) presence of concept in the image, (ii) Filter mapping to the concept, (iii) significant overlap between spatial concept location in the image and highly activated area in the corresponding filter activation. POEM does Concept Pattern Mining using CART, Explanation Tables, and Interpretable Decision Sets.
Concept-harmonized HierArchical INference (CHAIN) [ 122 ] evaluates the relationships between CAVs learned in earlier layers vs. later layers, creating a hierarchical inference concept graph from the DNN.

 
 
 

#### 3.2.3 Symbolic features based: Concept Trees 

 
 Several approaches to building Concept Tree have been proposed [ 50 , 49 , 90 , 84 , 135 ] .
 Santhirasekaram et al. [90] use symbolic features and generate a hierarchy of symbolic rules. Symbols are attained by discretization of the continuous latent space using vector quantization. This is followed by hyperbolic reasoning, which generates an abstraction tree containing symbolic rules and corresponding visual semantics. They demonstrate class-level and instance-level trees.

 
 
 CNN2DT [ 49 ] makes visual surrogate decision trees by decomposing a CNN into a feature extractor and classifier. It identifies concepts by using Network Dissection [] . Then, a decision tree is extracted from the classifier. They provide both local and global interpretations. The DT identified by CNN2DT was found to be too sensitive to training instances.
Additionally, one major limitation of methods using Network Dissection is the requirement for semantic labels for concepts.

 
 
 A recent work, TreeICE [ 84 ] uses NCAVs [ 135 ] to build a decision tree using extracted model features. They first use the ICE algorithm to get concept-level features from CNN. These concept-level features are used to learn NCAVs. Global Average Pooling (GAP) is performed to score each NCAV. A TreeICE classifier is learned using the top-scoring NCAVs and their corresponding concept labels (Ground truth or prediction by CNN).

 
 
 ACDTE [ 101 ] extracts concepts and trains a linear model differentiating clusters of a concept of interest (unlike concept activations in CAVs [ 56 ] ) and its negative samples (random segments from other clusters). A binary vector v v indicating the presence or absence of each concept c ∈ C c\in C is created for each image s s . Then, a decision tree is learned for image s s by using the class predictions and v v .
For details on other existing concept representations, please refer [ 97 ] .

 
 
 
 
 

## 4 Concept Discovery Methods

 
 Table 2 : Concept Discovery categorization 
 
 
 
 Type | 
 Method used | 
 
 
 Papers 
 | 

 
 Post-hoc | 
 Super-pixel | 
 
 
 ACE [ 36 ] , CocoX [ 4 ] 
 | 

 
 Coorperative Attribution (shapeley) | 
 
 
 ConceptSHAP [ 127 ] 
 | 

 
 Autoencoders based | 
 
 
 PACE [ 51 ] 
 | 

 
 Causality based | 
 
 
 MCE [ 118 ] , Alipour et al. [6] 
 | 

 
 Using Probes | 
 
 
 ACDTE [ 101 ] , Silver et al. [105] 
 | 

 
 Using Convex Optimization | 
 
 
 Schut et al. [96] 
 | 

 
 Ante-hoc | 
 Saliency based | 
 
 
 HINT [ 120 ] , Kori et al. [61] 
 | 

 
 Using Slot-attention [ 72 ] | 
 
 
 Stammer et al. [111] 
 | 

 
 Latent Space Disentanglement based | 
 
 
 GAN based: EPE [ 103 ] , StylEx [ 66 ] , Dissect [ 35 ] img2tab [ 107 ] , Charachon et al. [21] , Augustin et al. [9] , Tran et al. [117] , Causality enforcing O’Shaughnessy et al. [85] 
 | 

 
 XIL based | 
 
 
 iCSNs [ 110 ] , RRC [ 111 ] 
 | 

 

 
 
 While [ 56 ] make use of human-supervised concepts, various Concept Discovery Algorithms were developed that automatically discover these concepts.
We classify the Concept Discovery Methods into two categories based on Concept Discovery after model training (Post-hoc) or during it (Ante-hoc).

 
 

### 4.1 Post-hoc

 

#### 4.1.1 Super-pixel based 

 
 ACE [ 36 ] and CocoX [ 4 ] use super-pixels followed by saliency maps to discover concepts.

 
 
 Comparing clustered super-pixel segments with a pre-trained classifier 

 
 ACE focuses on discovering concepts that are meaningful, coherent, and important. On a broad level, ACE uses varying levels of super-pixels to generate both lower (color, texture) and higher-level features (objects) and clusters their activations in the model’s activation plane to get concepts. These clusters need to be identified with human interpretable concepts, which require a human to manually go through them. ACE automates this by using ImageNet-trained CNN features as a guide. It compares the perceptual similarity of identified cluster segments with the guide to label them. It also removes outliers to make the concepts coherent . It uses TCAV [ 56 ] scores to gauze concept importance for prediction.

 
 
 
 Using Saliency importances 

 
 For identifying meaningful concepts from super-pixels, CocoX [ 4 ] uses Grad-CAM [ 98 ] followed by pooling technique to obtain importance weights. It then selects the top p p super-pixels for each class based on the importance weights and clusters the actual image regions corresponding to important super-pixels for getting concepts.

 
 
 Akula et al. [4] further uses counterfactuals for explaining the model’s prediction in terms of concepts. It uses fault-line 1 1 
 1 
 
 
 
 Fault lines originate from human cognition where humans zoom in when they imagine an alternative to a
model prediction identification where adding (or subtracting) a fault line to the concept changes the class of object from A to B. Note that they do not perturb the input image but the activation in the last convolutional layer.

 
 
 
 

#### 4.1.2 Cooperative attribution based 

 
 ConceptSHAP [ 127 ] uses Shapeley values that attribute importance to concepts when combined to discover a complete set of concepts.

 
 
 

#### 4.1.3 Autoencoder based 

 
 PACE [ 51 ] extracts concepts via an auto-encoder approach where a 1-D convolutional layer is used to compress different CNN-extracted features. This is followed by an auto-encoder that maps the above-obtained convolutional feature map to concepts and back to the convolutional feature map. They use convolutional and transpose convolutional layers as encoder-decoder to preserve the explanation framework’s interpretability.
In order to explain a K K -way classifier, PACE uses K K independent autoencoders and detects the presence of the concept in an image via an embedding map for similarity comparison between the embedding vector and the concept vector.

 
 
 

#### 4.1.4 Causality based 

 
 Multi-dimensional Concept Discovery (MCD) [ 118 ] proposes concepts that are bound to model reasoning (causality) and are complete. They discover concepts using sparse subspace clustering. Alipour et al. [6] present one of few post-hoc methods using generative models for discovering concepts. They use pre-trained generative models to generate global causal contrastive counterfactuals and explain classification models.

 
 
 

#### 4.1.5 Probe based 

 
 ACDTE [ 101 ] finds images similar to class inputs to be explained from a main or related dataset (probe) and segments them. It then clusters the activations of the above-segmented images with few conditions to ensure that segments represent concepts. McGrath et al. [81] delve into how AlphaZero [ 105 ] show that AlphaZero gains human-like chess insights solely via self-play, without learning from human-generated data. The study employs concept-based techniques, showcasing that AlphaZero’s neural architecture captures an array of chess knowledge, spanning simple maneuvers to advanced strategies, throughout its training phase. By executing automatic probing, which entails using sparse linear regression models to align the AI’s internal activations with established human chess concepts, the study effectively evaluates how these concepts are integrated within AlphaZero’s framework. This investigation, complemented by behavioral analysis, pinpoints the timing and location of these concepts within the network, enriching our understanding of AI’s potential to mimic human cognitive functions in chess.

 
 
 

#### 4.1.6 Concept vectors search in latent space using convex optimization 

 
 Schut et al. [96] introduces a method for extracting new chess concepts from AlphaZero, an AI that achieved super-human performance in chess through self-play without human supervision. The study reveals that AlphaZero may harbor knowledge extending beyond existing human understanding, which, however, can be comprehensible and learnable by humans. They use convex optimization to extract learned knowledge explained in terms of concept vectors from a trained AlphaZero model. They optimize min ⁡ ‖ v c , l ‖ 1 \min\left\|v_{c,l}\right\|_{1} given concept constraints. They use concept constraints for two types of concepts: static and dynamic. Static concepts are found in one state of the RL ( e.g ., a car located on a highway) model, while dynamic concepts are found in the sequence of states ( e.g ., accelerating the car). They optimize over the latent vectors to find the best concept vectors for given constraints on concepts. Following concept vector’s discovery, they filter concept vectors which are teachable to another AI agent or person. They find AI agents who donot know the concept, teach it to the agent, and evaluate the agent’s performance on a task related to the given concept. They use a AlphaZero as a teacher to teach proto-types (chess positions demonstrating the use of concept) to a student agent. For this, they train the student by minimizing the KL divergence between teachers’ and students’ policies on training proto-types. They then determine whether the concept is teachable by evaluating the student’s performance on test set proto-types and estimating how often the student and teacher select the same top-1 move. In order to access the novelty of concepts, they use concepts from the late stages of AlphaZero’s training and also introduce a novelty metric based on ease of concept reconstruction using a set of basis vectors from AlphaZero’s games. They also provide human evaluation accessing whether the chess grandmasters can learn and apply the discovered concepts. Please refer to the Schut et al. [96] for more details.

 
 
 
 

### 4.2 Ante-hoc

 

#### 4.2.1 Saliency based 

 
 HINT [ 120 ] uses saliency maps to identify the responsible regions r e r_{e} for concept e e .
They categorize each entry in feature map z z as responsible to e e or not by aggregating the saliency map s s to obtain s ^ \hat{s} indicating the relevance of z D z_{D} to concept e e . They threshold this saliency map with t t to get the responsible regions ( s ^ t \hat{s} t ).

 
 
 GradCAM [ 98 ] attributions of specific layer’s clustered weights are used by Concept Graphs [ 61 ] (discussed in subsubsection 3.2.2 ).

 
 
 

#### 4.2.2 Slot-attention based 

 
 Stammer et al. [111] use Slot Attention [ 72 ] for concept discovery. Slot Attention decomposes the latent space into slots, which are meaningful task-dependent output vectors. Due to this Slot Attention, they can create a differentiable object-centric representation of input image without processing each object of scene [ 129 ] or dividing the scene into different levels of super-pixels unlike [ 36 ] .

 
 
 

#### 4.2.3 Using GAN’s 

 
 EPE [ 103 ] , StylEx [ 66 ] , Dissect [ 35 ] and img2tab [ 107 ] train GANs for learning concepts.
Dissect [ 35 ] does concept traversals in increasing order of effect on model prediction. It generates multiple counterfactual examples in input image space.

 
 
 A recent work img2tab [ 107 ] uses StyleGAN features to generate images having certain concepts. It debugs the model for spurious correlations/bias by training it on additional bias-removed data generated by StyleGAN.
 [ 21 ] uses Conditional GANs for generating explanations.

 
 
 Tran et al. [117] discovers causal binary concepts in an unsupervised manner with a VAE. It identifies concepts that are present and concepts that are not present for a model’s particular prediction. In their words: "data X is classified as class Y because X has A, B and does not have C" in which A, B, and C C are high-level concepts".
Their modeling of concepts in this manner encourages causality. Tonolini et al. [116] , Gyawali et al. [42] , Gupta et al. [41] also use binary concepts but do not impose causality. While O’Shaughnessy et al. [85] uses causal explanations, it does not encourage disentanglement of binary concepts, making the explanations harder to interpret. They use a causal DAG-based representation for modeling concepts. They also propose a learning process to make the model sensitive to the concepts, thus integrating the user’s prior knowledge into the model.

 
 
 

#### 4.2.4 By human interactions 

 
 Explanatory Interactive Learning (XIL), as described above, is a learning setting involving interactions with the user for explanations and interventions by them for correcting the model.

 
 
 Stammer et al. [110] model concept learning as a weak supervision task and use proto-types and XIL for it. They revise the latent space of neural networks with the help of prototypes. For this, they introduce interactive Concept Swapping Networks (iCSNs) which learn concepts and attribute them to specific prototypes by swapping the latent representations of paired images. This results in discrete and semantically representative latent space, which is more interpretable. They also show results on their newly introduced Elementary Concept Reasoning (ECR) dataset, which focuses on visual concepts shared by geometric objects.
RRC [ 111 ] uses interactions with humans to improve models’ reasoning. These explanations need not be correct because the model can still focus on confounding factors (wrong reasons), and thus, they allow users to correct the explanations by intervening in the model (XIL), giving feedback on explanations of the module which is going wrong. For visual explanations, a user can indicate regions that must be focussed upon [ 87 , 94 , 115 ] in terms of binary masks in input image space.
For the reasoning module’s explanations, user feedback is in the form of relational functions: if concept C C , then class k k .
They regularize the model to match the user explanations via a loss which includes either RRR [ 87 ] or HINT [ 120 ] loss terms.

 
 
 
 
 

## 5 Evaluation of Discovered Concepts

 
 The measures used for the evaluation of discovered concepts are given below:

 
 • 
 
 Completeness : As introduced by [ 127 ] it measures the sufficiency of a concept in explaining DNN.

 

 • 
 
 Correctness/Soundness : The identified concepts should be correct (not spurious), i.e., they should be truthful to the task model.

 

 • 
 
 Fidelity : It measures completeness and soundness of explanation [ 63 ] .

 

 • 
 
 Causality : Concepts cause a model to give a certain prediction. This is usually measured by checking the model’s behavior on images containing the concept vs images not containing the concept (checked by counterfactual-based approaches Goyal et al. [37] .

 

 • 
 
 Diversity : It measures the range of features covered by concepts. Discovered Concepts for a particular input should be non-overlapping.

 

 • 
 
 Impurities Measurement Purity is defined as the degree to which the predictive power of the learned representation for one concept is similar to what we would expect based on its corresponding ground truth label, compared to its predictive power for other concepts [ 131 ] .

 
 
 Concept Learning Models are prone to encoding impurities, which are measured by [ 131 ] in the following metrics:

 
 – 
 
 Oracle Impurity Score (OIS): measures intra-concept impurities while

 

 – 
 
 Niche Impurity Score (NIS) measures inter-concept representations impurities.

 

 
 

 
 
 
 

## 6 Different Learning Paradigms for XAI based Model Improvement

 

### 6.1 Explanation Guided Learning

 
 Explanation Guided Learning (EGL) is learning that uses Explanations for training or improvising the model. Explanation-guided learning (EGL) can have one of the following aims: (i) improving the interpretability, and (ii)improving the performance of the model. The primary goal of EGL is to learn a model that can make accurate predictions while generating meaningful explanations for its predictions.
EGL typically involves jointly optimizing the model prediction and the explanation by incorporating three key terms in the objective function: task supervision, explanation supervision, and explanation regularization.

 

 
 | 
 min ⁡ ℒ Pred ​ ( f ​ ( X ) , Y ) ⏟ task supervision  + α ​ ℒ Exp ​ ( g ⁡ ( f , ⟨ X , Y ⟩ ) , M ^ ) ⏟ explanation supervision  + β ​ Ω ​ ( g ⁡ ( f , ⟨ X , Y ⟩ ) ) ⏟ explanation regularization  \min\underbrace{\mathcal{L}_{\operatorname{Pred}}(f(X),Y)}_{\text{task supervision }}+\underbrace{\alpha\mathcal{L}_{\operatorname{Exp}}(g(f,\langle X,Y\rangle),\hat{M})}_{\text{explanation supervision }}+\underbrace{\beta\Omega(g(f,\langle X,Y\rangle))}_{\text{explanation regularization }} | 
 | 
 

 where M ^ \hat{M} incorporates the ‘right’ explanation provided via human annotation. The task supervision term guides the model in learning task-specific information, while the explanation supervision term supervises the model explanation to ensure consistency with ground truth. The explanation regularization term helps to avoid overfitting and encourages the model to generate explanations that are interpretable and meaningful.

 
 
 Figure 2 : Broad Categorization of Explanation Guided Learning (EGL) methods. 
 
 
 EGL is an umbrella that includes XIL, XAL, and CODL approaches Fig. 2 .

 
 
 

### 6.2 Explanatory Interactive Learning (XIL)

 
 Explanatory and Interactive Learning (XIL) is a framework for machine learning models to be inspected, interacted with, and revised to ensure that their learned knowledge aligns with human knowledge. XIL methods aim to mitigate learning shortcuts and provide explanations for the model’s decision-making process, enabling users to interact with the model and provide feedback to improve its performance. Friedrich et al. [31] provide a typology of existing XIL methods and provide a generalized XIL algorithm consisting of four essential steps of Select, Explain, Obtain, and Revise. The XIL algorithm takes in a set of annotated examples (A), a set of non-annotated examples (N), and an iteration budget (T). The Select module selects samples from N to present to the teacher. The Explain module provides the teacher with insights into the model’s reasoning process. The Obtain module allows the teacher to observe whether the model’s prediction is correct or incorrect and to provide corrective feedback. Finally, the Revise module uses the corrective feedback to update the model’s behavior towards the user.

 
 
 

### 6.3 Explanatory Active Learning (XAL)

 
 Explanatory Active Learning (XAL) is an emerging learning paradigm that combines active learning with explanations [ 34 ] . Active Learning (AL) is a learning paradigm that allows a learning algorithm to intelligently select instances to be labeled, which can lead to high performance with much less training data compared to traditional supervised learning approaches. AL has become increasingly important in modern machine learning, where labeled data can be expensive and time-consuming to obtain.
Active Learning reduces labeling workload by selecting instances to query a machine teacher for labels intelligently. However, the human-AI interface remains minimal and opaque, hindering the development of teacher-friendly interfaces for AL algorithms. XAL aims to introduce techniques from the field of explainable AI (XAI) into AL settings to make AI explanations a core element of the human-AI interface for teaching machines. In this paradigm, the teacher should be able to understand the reasoning underlying the model’s mistakes during the learning process. Once the model matures, the teacher should be able to recognize its progress to trust and feel confident about their teaching outcome. The teacher here can be a human or a large model.

 
 
 

### 6.4 Concept Oriented Deep Learning (CODL)

 
 Concept Oriented Deep Learning (CODL) [ 19 ] involves using concept-level supervision for models to improve model interpretability and performance. The major aspects of CODL, as introduced by Chang [19] , include concept graphs, concept representations, concept exemplars, and concept representation learning systems supporting incremental and continual learning. CODL leverages a common or background knowledge base, such as Microsoft Concept Graph, for the framework of conceptual understanding. By focusing on learning and using concept representations and exemplars, CODL is able to address the major limitations of deep learning, including interpretability, transferability, contextual adaptation, and the requirement for a large amount of labeled training data.

 
 
 Table 3 : Taxonomy of Concept based model improvement methods 
 
 
 
 
 
 Model Improvement Goal 
 | 
 Method | 
 
 
 Sub Method: Papers 
 | 

 
 
 
 Better Interpretability 
 | 
 Concept Conditioned Prediction Based | 
 
 
 Supervised: CBM [ 59 ] , CME [ 53 ] 
 | 

 
 | 
 
 
 Partially-Supervised: CBM-AUC [ 93 ] , CBP [ 38 ] 
 | 

 
 | 
 
 
 Unsupervised: SENN [ 7 ] , XAL based: CALI [ 113 ] , C-SENN [ 92 ] , Sarkar et al. [91] , intCEM Zarlenga et al. [132] , TabCBMs [ 133 ] 
 | 

 
 | 
 Concept Reasoning based | 
 
 
 Neuro-Symbolic: RRC [ 111 ] , DCR [ 12 ] 
 | 

 
 | 
 Interaction based | 
 
 
 Human Interaction: Lage and Doshi-Velez [64] , NesyXIL [ 111 ] 
 | 

 
 | 
 
 
 Proto-type based: proto2proto [ 54 ] , Xue et al. [126] , Wang et al. [121] , Sacha et al. [89] 
 | 

 
 
 
 Better Generalization 
 | 
 CAV based | 
 
 
 few shot: ClArC [ 8 ] 
 | 

 
 | 
 
 
 zero shot: Concept Distillation [ 39 ] 
 | 

 
 | 
 Causality Based | 
 
 
 Bahadori and Heckerman [10] 
 | 

 
 | 
 Latent Space Disentanglement based | 
 
 
 Neuro-Symbolic Reasoning Based: NeSyXIL [ 111 ] 
 | 

 
 | 
 
 
 GAN based: Img2Tab [ 107 ] 
 | 

 
 | 
 
 
 XAI based: ProtoPDebug [ 18 ] , CAIPI [ 94 ] , Interactive CBM’s [ 22 ] , Bontempelli et al. [17] 
 | 

 
 | 
 Probability Distribution based | 
 
 
 Kronenberger and Haselhoff [62] 
 | 

 

 
 
 
 

## 7 Categorisation of Model Improvement Methods

 
 EGL is an umbrella that includes XAL, XIL, and CODL approaches. We classify the EGL techniques based on their goals in two categories: (i) Better Generalization and (ii) Better Interpretability. Better Interpretability techniques use explanations for better Interpretability of the model, while better Generation techniques aim to use explanations for the generalizing model.
Note (i) includes (ii) because using explanations for better Generalization will automatically improve the Interpretability of models.

 
 

### 7.1 EGL categorization based on goal

 
 We categorize XAI-based model improvement in terms of the end goal as aiming (i) enhanced interpretability and (ii) better generalization. Most of the existing literature focuses on the former [ 59 , 53 , 7 , 113 , 92 , 93 , 111 , 64 , 110 , 114 , 31 , 54 , 126 , 121 , 89 ] with some works addressing the latter [ 31 , 8 , 10 , 111 , 107 , 18 , 99 , 104 , 94 , 62 , 17 ] . Such division concurs with Holmberg et al. [47] that found that explanations with the goals of better understanding have a significant difference from explanations to find actionable decisions.
Note (i) includes (ii) because using explanations for better Generalization will automatically improve the Interpretability of models.
In this section, we restrict ourselves to Concept-Based Model Improvement Methods.

 
 

#### 7.1.1 Aiming better interpretability

 
 These methods aim for no class accuracy impairment along with enhanced Interpretability of the model and can be categorized broadly based on the type of interpretability methods used as shown in Tab. 3 .

 
 
 
 • 
 
 Concept Conditioned Prediction Based 

 
 – 
 
 Supervised 
Concept Bottleneck Models (CBMs) [ 59 ] use concept mapping as an intermediate step in the model’s final prediction. Specifically, they map the input x x to a concept c c using a concept prediction module g : x → c g:x\rightarrow c , which is then used to predict the target class y y using a classification module f : c → y f:c\rightarrow y . This conditioning of the model’s prediction over concepts improves the interpretability and generalization of the model.
The authors tested CBMs using three different training paradigms: independent training of g g and f f (with f f trained on ground truth concepts), sequential training (where f f is given g g ’s predicted concept c ^ \hat{c} ), and joint training (where both f f and g g are optimized over a joint objective).
They found that all three training paradigms produced similar classification accuracies for both concept prediction and target class prediction. However, independently trained models had better test-time interventions. They intervene not on the actual value of concept c c but on the model’s concept predictions c ^ \hat{c} .

 

 – 
 
 Partially-Supervised 
Concept-based Model Extraction (CME) [ 53 ] approximates DNNs using simpler interpretable models like linear, logistic regression, and decision trees. While CBMs require binary concepts, CME can handle multi-valued concepts and can derive concepts combining multiple layers (unlike TCAV, CBM,   etc., which require one layer to be chosen). Additionally, CME can train in a partially supervised manner with few labeled and other unlabelled samples. Further, it assumes k k different concepts forming concept representation 𝒞 ⊂ ℝ k \mathcal{C}\subset\mathbb{R}^{k} such that every basis vector in 𝒞 \mathcal{C} spans all possible values of a particular concept.
They define two functions for mapping input to concept space ( p : 𝒳 → 𝒞 p:\mathcal{X}\rightarrow\mathcal{C} ) and concepts to predicted class space ( q : 𝒞 → 𝒴 q:\mathcal{C}\rightarrow\mathcal{Y} ). CME approximates f with f ^ \hat{f} given as f ^ ​ ( 𝐱 ) = q ^ ​ ( p ^ ​ ( 𝐱 ) ) \hat{f}(\mathbf{x})=\hat{q}(\hat{p}(\mathbf{x})) 
where q ^ \hat{q} and p ^ \hat{p} are extracted by CME.
For this, they define a function g l : ℋ 𝓁 → 𝒞 g^{l}:\mathcal{H^{l}}\rightarrow\mathcal{C} g ^ \hat{g} and extract it by SSMTL [ 71 ] using an approximation of k separate tasks (one for each concept). For every concept, i i [ 53 ] finds the best layer for predicting the concept by minimizing a loss function l l 
 l i = arg ⁡ min l ∈ L ​ ℓ ​ ( g i l , i ) l^{i}=\underset{l\in L}{\arg\min}\ell\left(g_{i}^{l},i\right) 
and finally approximating p ^ \hat{p} as

 

 
 | 
 p ^ ​ ( 𝐱 ) = ( g 1 l 1 ∘ f l 1 ​ ( 𝐱 ) , … , g k l k ∘ f l k ​ ( 𝐱 ) ) . \hat{p}(\mathbf{x})=\left(g_{1}^{l^{1}}\circ f^{l^{1}}(\mathbf{x}),\ldots,g_{k}^{l^{k}}\circ f^{l^{k}}(\mathbf{x})\right). | 
 | 
 

 q ^ \hat{q} is approximated in a supervised manner using a mapped set of concept labels and target labels (class labels)as a training set and fitting using Decision trees or logistic regression. Concept Bottleneck Model with Additional Unsupervised Concepts (CBM-AUC) [ 93 ] combines CBMs with SENNs to extend CBMs to additional unsupervised concepts along with supervised concepts.
 Grupen et al. [38] introduce Concept Bottleneck Policies (CBPs) for enhancing interpretability in multi-agent reinforcement learning (MARL). CBPs use conditioning similar to CBMs but in action space in RL instead of classification space in CBMs. They first predict concepts given a state, then use the predicted concepts to select actions, thus enhancing interpretability.
 Das et al. [27] present a novel approach to generating concept-based explanations in sequential decision-making settings, focusing on both improving AI agents’ learning rates and enhancing end-user comprehension of AI decisions. They learn a joint embedding model to map state-action pairs to concept-based explanations. These explanations are then used for two primary purposes: informing reward shaping during the agent’s training and providing end-users with understandable insights into the agent’s decision-making process at deployment.

 

 – 
 
 Unsupervised 
Self-Explaining Neural Networks (SENN) [ 7 ] is a type of Unsupervised method that learns interpretable basis concepts by approximating a model with a linear classifier. They focus on explicitness, faithfulness, and stability via regularizing models. For a linear classifier f ⁡ ( x ) = θ ​ ( x ) T ​ h ​ ( x ) f(x)=\theta(x)^{T}h(x) where x x is input, h ⁡ ( x ) h(x) is function mapping input to concepts, and θ \theta represents model parameters, they enforce the following constraints: (i) The output of f f is approximately same for two close inputs that are ∇ x f ​ ( x ) ≈ θ ⁡ ( x 0 ) \nabla_{x}f(x)\approx\theta\left(x_{0}\right) for all x x in a neighborhood of x 0 x_{0} ; (ii) model is linear in terms of concepts; (iii) Aggregation function for features shall be generic enough such that it is permutation invariant, isolate multiplicative interactions between concepts and preserve sign and relative magnitude of relevance values θ ​ ( x ) i \theta(x)_{i} . Finally, they define self-explaining prediction model as: f ⁡ ( x ) = g ⁡ ( θ 1 ​ ( x ) ​ h 1 ​ ( x ) , … , θ k ​ ( x ) ​ h k ​ ( x ) ) f(x)=g\left(\theta_{1}(x)h_{1}(x),\ldots,\theta_{k}(x)h_{k}(x)\right) with some additional constraints [ 7 ] . When the above formulation is applied over a neural network, it becomes SENN (the condition being g g is continuous over concepts and model weights). SENN can use user-provided concepts and learn new concepts that satisfy Fidelity : concepts persevere relative information and Diversity : concepts for a particular input are non-overlapping.
For SENN, they learn an autoencoder h h and ensure sparsity constraint to increase diversity while using proto-types for interpretations. They use a concept encoder that transforms inputs into concepts, an input-dependent parameterizer for generating relevance scores, and an aggregation function that combines them for the prediction of class labels. The concepts and their relevance predictions form an explanation.
SENN has reduced interpretability in real-world tasks like autonomous driving, which is overcome by C-SENN [ 92 ] . C-SENN combines Contrastive learning with concept learning of SENN to improve the discovered concepts and task accuracy.

 
 
 Sarkar et al. [91] use a base encoder (which is the same as the second last layer of the classifier) followed by two branches: one for concept and one for classification. The classification branch resembles the last layer of the classifier and gives the final class label prediction, while the concept head has a concept decoder that reconstructs the image given the intermediate representation (features by the base encoder). Thus, the intermediate representations act as concepts. They additionally impose an image reconstruction loss apart from fidelity and classification losses.

 
 
 Zarlenga et al. [132] propose intervention Concept Embedding Models (IntCEMs), a novel architecture designed to improve a model’s receptiveness to test-time interventions by embedding the capacity for concept interventions directly into the training phase. Unlike traditional Concept Bottleneck Models (CBMs) that lack explicit training for interventions, IntCEMs incorporate an end-to-end learnable intervention policy. This policy predicts meaningful intervention trajectories during training, enabling the model to incorporate expert feedback at test time effectively.

 
 
 Tabular Concept Bottleneck Models (TabCBMs) [ 133 ] are designed to generate concept-based explanations for tabular tasks, addressing the gap in concept-based interpretability for non-image data. This model formalizes a high-level concept in tabular data as nonlinear functions of correlated feature subsets, allowing for both supervised and unsupervised concept learning. TabCBMs can learn meaningful, interpretable concepts even without explicit concept annotations, and their performance competes with or outperforms existing methods while providing a high degree of interpretability. The architecture discovers and utilizes concept masks and scores for explanations, supporting human-in-the-loop interventions by allowing experts to correct or modify concept predictions, thereby enhancing model performance.

 

 
 

 • 
 
 Concept Reasoning Based 
 (Neuro-Symbolic) Right for the Right Concept [ 111 ] uses a concept embedding module that learns concepts via slot attention Interaction based [ 72 ] and a reasoning module that reasons for the concepts. The concept embedding module creates a decomposed representation of input space that can be mapped to concepts, while the reasoning module makes predictions based on the concepts that have been identified above. [ 12 ] introduces Deep Concept Reasoner (DCR) that uses symbolic reasoning to provide interpretable and semantically consistent predictions. DCR constructs interpretable syntactic rule structures using high-dimensional concept embeddings and then evaluates these rules on semantically meaningful concept truth degrees. This unique approach not only enhances interpretability but also significantly improves model performance on challenging benchmarks, demonstrating the ability to uncover meaningful logic rules even without concept supervision during training.

 

 • 
 
 Interaction based 
Interaction-based methods aiming for better interpretability need interaction with an expert, which is either done by a human expert or by using class prototypes. Henceforth, they can be classified into broadly two categories as follows.

 
 – 
 
 Human interaction based 
 Lage and Doshi-Velez [64] learn interpretable models via user feedback on concepts(which ones are similar and which are not) and also which concepts should(not) affect.
Interactive Concept Swapping Networks iCSN’s [ 110 ] bind concepts to prototypes by swapping the latent representations of paired images. They use prototypes to interactively learn concepts with user feedback. A Human can query iCSN prototypes and update them for concepts.
 NesyXIL [ 111 ] adds a user interactive layer over Nesy [ 111 ] discussed above.
 For a detailed survey on methods that use explanations in interactive ML, please refer [ 114 , 31 ] .

 

 – 
 
 Proto-type based 
Proto2Proto [ 54 ] uses knowledge distillation to transfer interpretability from (interpretable) teacher to a student model.

 
 
 Xue et al. [126] use global and local prototypes for enhanced interpretability in Visual Transformers (ViT). Wang et al. [121] use macro (broader) and micro proto-types (more specific) for interpretable models learning from mistakes. A similar idea of macro and micro prototypes is used by Sacha et al. [89] , which leverages support prototypes capturing macro-level features and trivial proto-types capturing specific micro features. The support (or macro) proto-types provide a global overview of the concepts (or features) but cannot capture some class-specific trivial features (thus captured by trivial (or micro) proto-types.

 

 
 

 
 
 
 

#### 7.1.2 Aiming better generalization

 
 These models typically remove bias or confounding factors and show accuracy improvement on poisoned datasets. [ 31 ] gives a topology exploring mitigation of shortcut behavior in models which involves steps of Select, Explain, Obtain feedback, and Revise model.
We divide these methods as follows.

 
 • 
 
 CAV based 
They use CAVs [ 56 ] for the representation of concepts and move activations to make a model sensitive or insensitive to the concept.

 
 – 
 
 Few Shot Anders et al. [8] does artifact removal from models by moving activations of images according to the CAVs learned for class images containing artifact vs. non-artifact class images. It is a few-shot method since it requires artifact and non-artifact images.
It uses two methods for the removal of artifacts: Augmentative and Projective. Augmentative Class Artifact Compensation (AClArC) augments the class samples, trying to remove the artifact by moving them according to the CAV direction and retraining the original model. Projective Class Artifact Compensation (PClArC), on the other hand, moves the class activations to a concept-neutral direction in the model’s activation space by a simple linear transformation.

 

 – 
 
 Zero Shot Gupta et al. [39] introduce a novel method for concept sensitive finetuning of Neural Networks. They introduce a concept loss that moves model activations away from CAV direction for a desensitizing model for a given concept. They further propose Concept Distillation to use a pre-trained model’s conceptual knowledge to train a student model via their concept loss. They also introduce a proto-type-based concept sensitivity calculation to enable intermediate layer sensitivity. They show applications of their method in debiasing and prior-knowledge reconstruction.

 

 
 

 • 
 
 Causality based 
Causal modeling involves checking for causes of a particular effect (here, the model’s particular predictions). Bahadori and Heckerman [10] use causal graphs for debiasing CBM’s removing confounding factors or clever-hans. It models the impacts of unobserved variables using causal graphs and removes them with a two-stage regression technique aided by instrumental variables.

 

 • 
 
 Latent Space Disentanglement Based 
They disentangle latent space to represent similar concepts in similar spatial regions. Latent space disentanglement is typically achieved through the use of generative models, such as autoencoders, variational autoencoders (VAEs), and Generative Adversarial Networks.

 
 – 
 
 GAN based 
Img2Tab [ 107 ] as described above, allows user interventions on the model for concept based debugging by identifying class-relevant concepts from W k W_{k} metric or classifier and using those to filter the semantics our of classifier P P by masking all unwanted features across training samples.

 

 – 
 
 Neuro-Symbolic Reasoning Based 
NeSyXIL [ 111 ] allows for user corrections to its concept embedding module explanations or reasoning module explanations.

 

 – 
 
 User Interaction for input based 
ProtoPDebug [ 18 ] debugs part-prototype networks using a concept-level debugger with human supervision regarding what part-prototype is forgotten or kept. Right for the Right Latent factors [ 99 ] provides a debiasing approach for generative models via disentanglement of latent space with human feedback. They enforce disentanglement via ELBO loss and a match pairing loss [ 104 ] . CAIPI [ 115 ] propose an XIL framework where DNNs query the user while the user explains the queries and corrects the explanation. Right for the Right Scientific Reasons RRSR Schramowski et al. [94] use CAIPI or RRR [ 87 ] depending on the task and demonstrate results removing clever-hans phenomena. Interactive CBMs [ 22 ] extend CBMs to XIL using an interaction policy that selects which concept labels to be queried from the user to improve prediction maximally. Their policy combines concept prediction uncertainty and the influence of concept on model prediction. [ 17 ] introduce a debugging technique for CBM’s debugging using human supervision. They can intervene on both concept-level and input-level bugs. CALI [ 113 ] extends SENN to XAL setting learning SENNs from class labels and explanation guidance by users.

 

 
 

 • 
 
 Probability Distribution based 
 [ 62 ] uses explanations to verify (accept or reject) the prediction. They show results over the GTSRB dataset [ 109 ] , which consists of 43 different classes showing German traffic light signs. They further use three attributions of varying complexity as concepts:
 Simple: Color
 Medium: Primitive Shapes like squares, circles, triangles, octagons, etc.
 Complex: Numbers (0-9) and symbols (truck, animal, car, children, bicycle, etc.)
The examples above are included as synthetic data.

 

 
 
 
 
 

### 7.2 EGL in other applications

 
 In addition to its applications in explainable machine learning, concept-based approaches have also been used in other areas. For example, Alabdulmohsin et al. [5] have applied a concept-based approach to domain adaptation, where they aim to adapt a model trained on a source domain to perform well on a target domain with different characteristics. They use concept representations to bridge the domain gap between the source and target domains.
Another example is the work of McGrath et al. [80] , who analyze the knowledge of concepts learned by AlphaZero, an artificial intelligence system that learned to play chess, shogi, and Go at a superhuman level through self-play. They investigate the similarity between the concepts learned by AlphaZero and the concepts understood by human players. They found that the concepts learned by AlphaZero are similar to those learned by humans, but they are more precise and granular.
ProtoMIL is [ 88 ] is used for whole slide image classification and uses Multiple Instance Learning.
XProtoNet [ 57 ] uses global and local explanations for Diagnosis in Chest Radiography. Grupen et al. [38] proposes the use of concepts for understanding multi-agent behavior. It conditions each agent’s actions on concept-based policies proposing Concept Bottleneck Policies (CBPs), achieving interpretability without decreasing performance. They also allow interventions for desired concepts. Schut et al. [96] discover concepts learned in the chess game in AlphaZero, trying to broaden human knowledge of those concepts. They further showcase that these concepts learned by AlphaZero are learnable and usable by humans. Been Kim, Brain Team, Alison Lentz [15] uses CAVs for visual searching by finding images within a dataset that align closely with a user-defined mood board 2 2 
 2 
 
 
 
 Mood boards are visual collages that assemble images, materials, text, and other design elements to convey a specific theme, style. , essentially mapping each mood board with a CAV.

 
 
 
 

## 8 Concept Based Model Performance Evaluation

 
 [ 55 ] provides a Protocol for Evaluating Model Interpretation Methods from Visual Explanations.
 [ 40 ] provide an evaluation metric called Concept Sensitivity Metric (CSM) for measuring the disentanglement of concepts in multi-branch networks. They use CAV-based sensitivity scores to calculate CSMs. CSMs are essentially a ratio of desired concepts to non-desired concepts. They demonstrate their results on evaluating model disentaglement over an ill-posed under-constrained problem of Intrinsic Image Decomposition (IID). IID involves decomposing an image into two independent variables: Reflectance (R) and Shading (S). By definition, Reflectance is the illumination-invariant component of the scene, while shading is dependent on illumination. They check the sensitivity of the model for concepts of Reflectance properties variance (albedo/material color variance) and Shading properties variance (illumination variance) and provide two CSM scores C ​ S ​ M R CSM_{R} and C ​ S ​ M S CSM_{S} for measuring the quality of R, and S. CSMs are a necessary condition of evaluating IID (since model R-S disentanglement is necessary).
 Collins et al. [25] measure the impact of human uncertainty in the context of concept-based AI models, mainly focusing on models that allow human feedback via concept interventions. Concept-based models, like Concept Bottleneck Models (CBMs) and Concept Embedding Models (CEMs), traditionally assume human interventions are always accurate and confident. However, real-world decision-making involves human error and uncertainty. The study explores how concept-based models respond to uncertain interventions using two novel datasets: UMNIST, which simulates uncertainty based on the MNIST dataset, and CUB-S, a version of the CUB dataset with densely annotated soft labels reflecting human uncertainty.

 
 
 

## 9 Future Directions

 
 There are many future directions for Concept-based Approaches discussed below.

 
 
 Bias Detection 

 
 Concept Based Approaches are intended to be used for finding confounding factors and clever-hans learned by the models.
While post-hoc explanations require no model architecture changes, they are ineffective for detecting unknown biases in the system [ 2 ] .
We thus need better automatic bias detection methods using ML interpretability.

 
 
 
 Robustness 

 
 [ 106 ] study the robustness of CBMs and SENNs to adversarial perturbations and define many malicious attacks to evaluate this. They also propose training for defense against such attacks. CBMs observe information leakage about data distribution to concept prediction model due to the use of soft concept labels [ 74 , 79 ] . Lockhart et al. [73] present a framework to mitigate the leaked concept information in CBMs using Monte-Carlo Dropout. GlanceNets [ 78 ] provides better interpretability by aligning the model’s representation and data generation process. It uses disentangled representations and open-set recognition for this alignment and prevents leakage of concepts.

 
 
 
 

## 10 Conclusion

 
 We discussed various aspects of concept based approaches starting from concept representation methods to various concept discovery methods. We provided a detailed hierarchy of the methods and also discussed the concept evaluation metrics, followed by an explanation of the guided learning hierarchy, focusing on Concept-Oriented Deep Learning (CODL) methods. We classified concept-based model improvement methods on the basis of two aims: better interpretability and better generalization. We also discussed applications of concept-guided methods in fields like domain adaptation, AlphaZero games, and applications in medical domains. In summary, our discussion commenced from post-hoc concept-based interpretability methods, covered ante-hoc concept-based training methods and concluded with concept-based model performance evaluation.

 
 
 

## References

 
 
 [2] 
 
Julius Adebayo, Michael
Muelly, Harold Abelson, and Been Kim.
2022.

 
 Post hoc Explanations may be Ineffective for
Detecting Unknown Spurious Correlation.

 
 ArXiv abs/2212.04629
(2022).

 
 
 

 
 [3] 
 
Naveed Akhtar.
2023.

 
 A Survey of Explainable AI in Deep Visual Modeling:
Methods and Metrics.

 
 ArXiv abs/2301.13445
(2023).

 
 
 

 
 [4] 
 
Arjun Akula, Shuai Wang,
and Song-Chun Zhu. 2020.

 
 Cocox: Generating conceptual and counterfactual
explanations via fault-lines. In Proceedings of
the AAAI Conference on Artificial Intelligence , Vol. 34.
2594–2601.

 
 
 

 
 [5] 
 
Ibrahim M. Alabdulmohsin,
Nicole Chiou, Alexander D’Amour,
Arthur Gretton, Sanmi Koyejo,
Matt J. Kusner, Stephen R. Pfohl,
Olawale Salaudeen, Jessica Schrouff,
and Katherine Tsai. 2022.

 
 Adapting to Latent Subgroup Shifts via Concepts and
Proxies.

 
 ArXiv abs/2212.11254
(2022).

 
 
 

 
 [6] 
 
Kamran Alipour, Aditya
Lahiri, Ehsan Adeli, Babak Salimi, and
Michael J. Pazzani. 2022.

 
 Explaining Image Classifiers Using Contrastive
Counterfactuals in Generative Latent Spaces.

 
 ArXiv abs/2206.05257
(2022).

 
 
 

 
 [7] 
 
David Alvarez Melis and
Tommi Jaakkola. 2018.

 
 Towards robust interpretability with
self-explaining neural networks.

 
 Advances in neural information processing
systems 31 (2018).

 
 
 

 
 [8] 
 
Christopher J Anders,
Leander Weber, David Neumann,
Wojciech Samek, Klaus-Robert Müller,
and Sebastian Lapuschkin.
2022.

 
 Finding and removing clever hans: Using explanation
methods to debug and improve deep models.

 
 Information Fusion 77
(2022), 261–295.

 
 
 

 
 [9] 
 
Maximilian Augustin,
Valentyn Boreiko, Francesco Croce, and
Matthias Hein. 2022.

 
 Diffusion visual counterfactual explanations.

 
 arXiv preprint arXiv:2210.11841 
(2022).

 
 
 

 
 [10] 
 
Mohammad Taha Bahadori and
David E Heckerman. 2020.

 
 Debiasing concept-based explanations with causal
analysis.

 
 arXiv preprint arXiv:2007.11500 
(2020).

 
 
 

 
 [11] 
 
Andrew Bai, Chih-Kuan
Yeh, Pradeep Ravikumar, Neil YC Lin,
and Cho-Jui Hsieh. 2022.

 
 Concept Gradient: Concept-based Interpretation
Without Linear Assumption.

 
 arXiv preprint arXiv:2208.14966 
(2022).

 
 
 

 
 [12] 
 
Pietro Barbiero, Gabriele
Ciravegna, Francesco Giannini,
Mateo Espinosa Zarlenga, Lucie Charlotte
Magister, Alberto Tonda, Pietro Lió,
Frederic Precioso, Mateja Jamnik, and
Giuseppe Marra. 2023.

 
 Interpretable neural-symbolic concept reasoning.
In International Conference on Machine Learning .
PMLR, 1801–1825.

 
 
 

 
 [13] 
 
David Bau, Bolei Zhou,
Aditya Khosla, Aude Oliva, and
Antonio Torralba. 2017.

 
 Network Dissection: Quantifying Interpretability of
Deep Visual Representations.

 
 2017 IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) (2017),
3319–3327.

 
 
 

 
 [14] 
 
Katharina Beckh, Sebastian
Müller, Matthias Jakobs, Vanessa
Toborek, Hanxiao Tan, Raphael Fischer,
Pascal Welke, Sebastian Houben, and
Laura von Rueden. 2021.

 
 Explainable machine learning with prior knowledge:
An overview.

 
 arXiv preprint arXiv:2105.10172 
(2021).

 
 
 

 
 [15] 
 
Been Kim, Brain Team, Alison Lentz.
n.d..

 
 Mood Board Search.

 
 https://github.com/google-research/mood-board-search .

 
 
 
 Accessed: [Insert date here].

 

 
 [16] 
 
Åke Björck.
1994.

 
 Numerics of gram-schmidt orthogonalization.

 
 Linear Algebra and Its Applications 
197 (1994), 297–316.

 
 
 

 
 [17] 
 
Andrea Bontempelli, Fausto
Giunchiglia, Andrea Passerini, and
Stefano Teso. 2021.

 
 Toward a Unified Framework for Debugging
Concept-based Models.

 
 
 

 
 [18] 
 
Andrea Bontempelli,
Stefano Teso, Fausto Giunchiglia, and
Andrea Passerini. 2023.

 
 Concept-level debugging of part-prototype
networks.

 
 (2023).

 
 
 

 
 [19] 
 
Daniel T. Chang.
2018a.

 
 Concept-Oriented Deep Learning.

 
 ArXiv abs/1806.01756
(2018).

 
 
 

 
 [20] 
 
Daniel T Chang.
2018b.

 
 Concept-Oriented Deep Learning: Generative Concept
Representations.

 
 arXiv preprint arXiv:1811.06622 
(2018).

 
 
 

 
 [21] 
 
Martin Charachon,
Paul-Henry Cournède, Céline
Hudelot, and Roberto Ardon.
2022.

 
 Leveraging conditional generative models in a
general explanation framework of classifier decisions.

 
 Future Generation Computer Systems 
132 (2022), 223–238.

 
 
 

 
 [22] 
 
Kushal Chauhan, Rishabh
Tiwari, Jana von Freyberg, Pradeep
Shenoy, and Krishnamurthy Dvijotham.
2022.

 
 Interactive Concept Bottleneck Models.

 
 ArXiv abs/2212.07430
(2022).

 
 
 

 
 [23] 
 
Chaofan Chen, Oscar Li,
Daniel Tao, Alina Barnett,
Cynthia Rudin, and Jonathan K Su.
2019.

 
 This looks like that: deep learning for
interpretable image recognition.

 
 Advances in neural information processing
systems 32 (2019).

 
 
 

 
 [24] 
 
Zhi Chen, Yijie Bei,
and Cynthia Rudin. 2020.

 
 Concept whitening for interpretable image
recognition.

 
 Nature Machine Intelligence 
2, 12 (2020),
772–782.

 
 
 

 
 [25] 
 
Katherine Maeve Collins,
Matthew Barker, Mateo Espinosa Zarlenga,
Naveen Raman, Umang Bhatt,
Mateja Jamnik, Ilia Sucholutsky,
Adrian Weller, and Krishnamurthy
Dvijotham. 2023.

 
 Human uncertainty in concept-based ai systems. In
 Proceedings of the 2023 AAAI/ACM Conference on AI,
Ethics, and Society . 869–889.

 
 
 

 
 [26] 
 
Vargha Dadvar.
2022.

 
 POEM: pattern-oriented explanations of CNN
models .

 
 Master’s thesis. University
of Waterloo.

 
 
 

 
 [27] 
 
Devleena Das, Sonia
Chernova, and Been Kim.
2023.

 
 State2explanation: Concept-based explanations to
benefit agent learning and user understanding.

 
 Advances in Neural Information Processing
Systems 36 (2023),
67156–67182.

 
 
 

 
 [28] 
 
Jean Donham.
2010.

 
 Deep learning through concept-based inquiry.

 
 School Library Monthly 
27, 1 (2010),
8–15.

 
 
 

 
 [29] 
 
Mengnan Du, Ninghao Liu,
and Xia Hu. 2019.

 
 Techniques for interpretable machine learning.

 
 Commun. ACM 63,
1 (2019), 68–77.

 
 
 

 
 [30] 
 
Ruth Fong and Andrea
Vedaldi. 2018.

 
 Net2vec: Quantifying and explaining how concepts
are encoded by filters in deep neural networks. In
 Proceedings of the IEEE conference on computer
vision and pattern recognition . 8730–8738.

 
 
 

 
 [31] 
 
Felix Friedrich, Wolfgang
Stammer, Patrick Schramowski, and
Kristian Kersting. 2022.

 
 A Typology to Explore the Mitigation of Shortcut
Behavior.

 
 
 

 
 [32] 
 
Yuyang Gao, Siyi Gu,
Junji Jiang, Sungsoo Ray Hong,
Dazhou Yu, and Liang Zhao.
2022a.

 
 Going Beyond XAI: A Systematic Survey for
Explanation-Guided Learning.

 
 arXiv preprint arXiv:2212.03954 
(2022).

 
 
 

 
 [33] 
 
Yuyang Gao, Tong Steven
Sun, Guangji Bai, Siyi Gu,
Sungsoo Ray Hong, and Zhao Liang.
2022b.

 
 RES: A Robust Framework for Guiding Visual
Explanation. In Proceedings of the 28th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining .
432–442.

 
 
 

 
 [34] 
 
Bhavya Ghai, Q Vera Liao,
Yunfeng Zhang, Rachel Bellamy, and
Klaus Mueller. 2021.

 
 Explainable active learning (xal) toward ai
explanations as interfaces for machine teachers.

 
 Proceedings of the ACM on Human-Computer
Interaction 4, CSCW3
(2021), 1–28.

 
 
 

 
 [35] 
 
Asma Ghandeharioun, Been
Kim, Chun-Liang Li, Brendan Jou,
Brian Eoff, and Rosalind W Picard.
2021.

 
 Dissect: Disentangled simultaneous explanations via
concept traversals.

 
 arXiv preprint arXiv:2105.15164 
(2021).

 
 
 

 
 [36] 
 
Amirata Ghorbani, James
Wexler, James Y Zou, and Been Kim.
2019.

 
 Towards automatic concept-based explanations.

 
 Advances in Neural Information Processing
Systems 32 (2019).

 
 
 

 
 [37] 
 
Yash Goyal, Amir Feder,
Uri Shalit, and Been Kim.
2019.

 
 Explaining Classifiers with Causal Concept Effect
(CaCE).

 
 ArXiv abs/1907.07165
(2019).

 
 
 

 
 [38] 
 
Niko Grupen, Natasha
Jaques, Been Kim, and Shayegan
Omidshafiei. 2022.

 
 Concept-based understanding of emergent multi-agent
behavior. In Deep Reinforcement Learning Workshop
NeurIPS 2022 .

 
 
 

 
 [39] 
 
Avani Gupta, Saurabh
Saini, and PJ Narayanan.
2023a.

 
 Concept Distillation: Leveraging Human-Centered
Explanations for Model Improvement. In
 Thirty-seventh Conference on Neural Information
Processing Systems .

 
 
 

 
 [40] 
 
Avani Gupta, Saurabh
Saini, and P. J. Narayanan.
2023b.

 
 Interpreting Intrinsic Image Decomposition Using
Concept Activations (ICVGIP ’22) .
Association for Computing Machinery,
New York, NY, USA, Article 2,
9 pages.

 
 

 https://doi.org/10.1145/3571600.3571603 

 

 
 [41] 
 
Kamal Gupta, Saurabh
Singh, and Abhinav Shrivastava.
2020.

 
 PatchVAE: Learning Local Latent Codes for
Recognition.

 
 2020 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) (2020),
4745–4754.

 
 
 

 
 [42] 
 
Prashnna Kumar Gyawali,
Zhiyuan Li, Cameron Knight,
Sandesh Ghimire, B. Milan Hor,
John L. Sapp, and Linwei Wang.
2019.

 
 Improving Disentangled Representation Learning with
the Beta Bernoulli Process.

 
 2019 IEEE International Conference on Data
Mining (ICDM) (2019), 1078–1083.

 
 
 

 
 [43] 
 
Peter Hase and Mohit
Bansal. 2021.

 
 When can models learn from explanations? a formal
framework for understanding the roles of explanation data.

 
 arXiv preprint arXiv:2102.02201 
(2021).

 
 
 

 
 [44] 
 
Yi He, Xi Yang,
Chia-Ming Chang, Haoran Xie, and
Takeo Igarashi. 2022.

 
 Efficient Human-in-the-loop System for Guiding DNNs
Attention.

 
 arXiv preprint arXiv:2206.05981 
(2022).

 
 
 

 
 [45] 
 
Carl Johan Helgstrand and
Niklas Hultin. 2022.

 
 Comparing Human Reasoning and Explainable AI.

 
 
 
 
 

 
 [46] 
 
P Hitzler and M
Sarker. 2022.

 
 Human-centered concept explanations for neural
networks.

 
 Neuro-Symbolic Artificial Intelligence: The
State of the Art 342, 337
(2022), 2.

 
 
 

 
 [47] 
 
Lars Holmberg, Paul
Davidsson, and Per Linde.
2022.

 
 Mapping Knowledge Representations to Concepts: A
Review and New Perspectives.

 
 ArXiv abs/2301.00189
(2022).

 
 
 

 
 [48] 
 
Lei Ji, Yujing Wang,
Botian Shi, Dawei Zhang,
Zhongyuan Wang, and Jun Yan.
2019.

 
 Microsoft concept graph: Mining semantic concepts
for short text understanding.

 
 Data Intelligence 1,
3 (2019), 238–270.

 
 
 

 
 [49] 
 
Shichao Jia, Peiwen Lin,
Zeyu Li, Jiawan Zhang, and
Shixia Liu. 2019a.

 
 Visualizing surrogate decision trees of
convolutional neural networks.

 
 Journal of Visualization 
23 (2019), 141–156.

 
 
 

 
 [50] 
 
Shichao Jia, Lin Peiwen,
Zeyu Li, Jiawan Zhang, and
Shixia Liu. 2019b.

 
 Visualizing surrogate decision trees of
convolutional neural networks.

 
 Journal of Visualization 
23 (11 2019).

 
 
 https://doi.org/10.1007/s12650-019-00607-z 

 

 
 [51] 
 
Vidhya Kamakshi, Uday
Gupta, and N. C. Krishnan.
2021.

 
 PACE: Posthoc Architecture-Agnostic Concept
Extractor for Explaining CNNs.

 
 2021 International Joint Conference on Neural
Networks (IJCNN) (2021), 1–8.

 
 
 

 
 [52] 
 
Dmitry Kazhdan, B.
Dimanov, Helena Andrés-Terré,
Mateja Jamnik, Pietro Lio’, and
Adrian Weller. 2021.

 
 Is Disentanglement all you need? Comparing
Concept-based Disentanglement Approaches.

 
 ArXiv abs/2104.06917
(2021).

 
 
 

 
 [53] 
 
Dmitry Kazhdan, Botty
Dimanov, Mateja Jamnik, Pietro Liò,
and Adrian Weller. 2020.

 
 Now you see me (CME): concept-based model
extraction.

 
 arXiv preprint arXiv:2010.13233 
(2020).

 
 
 

 
 [54] 
 
Monish Keswani, Sriranjani
Ramakrishnan, Nishant Reddy, and
Vineeth N Balasubramanian.
2022.

 
 Proto2Proto: Can you recognize the car, the way I
do?. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition . 10233–10243.

 
 
 

 
 [55] 
 
Hamed Behzadi Khormuji and
José Oramas. 2023.

 
 A Protocol for Evaluating Model Interpretation
Methods from Visual Explanations.

 
 2023 IEEE/CVF Winter Conference on
Applications of Computer Vision (WACV) (2023),
1421–1429.

 
 
 

 
 [56] 
 
Been Kim, Martin
Wattenberg, Justin Gilmer, Carrie Cai,
James Wexler, Fernanda Viegas,
et al . 2018.

 
 Interpretability beyond feature attribution:
Quantitative testing with concept activation vectors (tcav). In
 International conference on machine learning .
PMLR, 2668–2677.

 
 
 

 
 [57] 
 
Eunji Kim, Siwon Kim,
Minji Seo, and Sungroh Yoon.
2021.

 
 XProtoNet: Diagnosis in Chest Radiography with
Global and Local Explanations.

 
 2021 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) (2021),
15714–15723.

 
 
 

 
 [58] 
 
Diederik P Kingma and
Max Welling. 2013.

 
 Auto-encoding variational bayes.

 
 arXiv preprint arXiv:1312.6114 
(2013).

 
 
 

 
 [59] 
 
Pang Wei Koh, Thao
Nguyen, Yew Siang Tang, Stephen
Mussmann, Emma Pierson, Been Kim, and
Percy Liang. 2020.

 
 Concept bottleneck models. In
 International Conference on Machine Learning .
PMLR, 5338–5348.

 
 
 

 
 [60] 
 
Avinash Kori, Ben
Glocker, and Francesca Toni.
2022a.

 
 Visual Debates.

 
 arXiv preprint arXiv:2210.09015 
(2022).

 
 
 

 
 [61] 
 
Avinash Kori, Parth
Natekar, Balaji Srinivasan, and
Ganapathy Krishnamurthi. 2022b.

 
 Interpreting deep neural networks for medical
imaging using concept graphs.

 
 In AI for Disease Surveillance and Pandemic
Intelligence: Intelligent Disease Detection in Action .
Springer, 201–216.

 
 
 

 
 [62] 
 
Jan Kronenberger and
Anselm Haselhoff. 2020.

 
 Dependency Decomposition and a Reject Option for
Explainable Models.

 
 arXiv preprint arXiv:2012.06523 
(2020).

 
 
 

 
 [63] 
 
Todd Kulesza, Simone
Stumpf, Margaret Burnett, Sherry Yang,
Irwin Kwan, and Weng-Keen Wong.
2013.

 
 Too much, too little, or just right? Ways
explanations impact end users’ mental models. In
 2013 IEEE Symposium on visual languages and human
centric computing . IEEE, 3–10.

 
 
 

 
 [64] 
 
Isaac Lage and Finale
Doshi-Velez. 2020.

 
 Learning interpretable concept-based models with
human feedback.

 
 arXiv preprint arXiv:2012.02898 
(2020).

 
 
 

 
 [65] 
 
Brenden M Lake, Tomer D
Ullman, Joshua B Tenenbaum, and
Samuel J Gershman. 2017.

 
 Building machines that learn and think like
people.

 
 Behavioral and brain sciences 
40 (2017), e253.

 
 
 

 
 [66] 
 
Oran Lang, Yossi
Gandelsman, Michal Yarom, Yoav Wald,
Gal Elidan, Avinatan Hassidim,
William T Freeman, Phillip Isola,
Amir Globerson, Michal Irani,
et al . 2021.

 
 Explaining in style: Training a gan to explain a
classifier in stylespace. In Proceedings of the
IEEE/CVF International Conference on Computer Vision .
693–702.

 
 
 

 
 [67] 
 
Juho Lee, Yoonho Lee,
Jungtaek Kim, Adam Kosiorek,
Seungjin Choi, and Yee Whye Teh.
2019.

 
 Set transformer: A framework for attention-based
permutation-invariant neural networks. In
 International conference on machine learning .
PMLR, 3744–3753.

 
 
 

 
 [68] 
 
Oscar Li, Hao Liu,
Chaofan Chen, and Cynthia Rudin.
2018.

 
 Deep learning for case-based reasoning through
prototypes: A neural network that explains its predictions. In
 Proceedings of the AAAI Conference on Artificial
Intelligence , Vol. 32.

 
 
 

 
 [69] 
 
Zhibin Liao, Kewen Liao,
Haifeng Shen, Marouska F Van Boxel,
Jasper Prijs, Ruurd L Jaarsma,
Job N Doornberg, Anton Van den Hengel,
and Johan W Verjans. 2022.

 
 CNN Attention Guidance for Improved Orthopedics
Radiographic Fracture Classification.

 
 IEEE Journal of Biomedical and Health
Informatics 26, 7
(2022), 3139–3150.

 
 
 

 
 [70] 
 
Pantelis Linardatos,
Vasilis Papastefanopoulos, and Sotiris
Kotsiantis. 2020.

 
 Explainable ai: A review of machine learning
interpretability methods.

 
 Entropy 23,
1 (2020), 18.

 
 
 

 
 [71] 
 
Qiuhua Liu, Xuejun Liao,
and Lawrence Carin. 2007.

 
 Semi-supervised multitask learning.

 
 Advances in Neural Information Processing
Systems 20 (2007).

 
 
 

 
 [72] 
 
Francesco Locatello, Dirk
Weissenborn, Thomas Unterthiner, Aravindh
Mahendran, Georg Heigold, Jakob
Uszkoreit, Alexey Dosovitskiy, and
Thomas Kipf. 2020.

 
 Object-centric learning with slot attention.

 
 Advances in Neural Information Processing
Systems 33 (2020),
11525–11538.

 
 
 

 
 [73] 
 
Joshua Lockhart, Nicolas
Marchesotti, Daniele Magazzeni, and
Manuela Veloso. 2022.

 
 Towards learning to explain with concept bottleneck
models: mitigating information leakage.

 
 ArXiv abs/2211.03656
(2022).

 
 
 

 
 [74] 
 
Anita Mahinpei, Justin
Clark, Isaac Lage, Finale Doshi-Velez,
and Weiwei Pan. 2021.

 
 Promises and Pitfalls of Black-Box Concept Learning
Models.

 
 ArXiv abs/2106.13314
(2021).

 
 
 

 
 [75] 
 
Jiayuan Mao, Chuang Gan,
Pushmeet Kohli, Joshua B. Tenenbaum,
and Jiajun Wu. 2019.

 
 The Neuro-Symbolic Concept Learner: Interpreting
Scenes Words and Sentences from Natural Supervision.

 
 ArXiv abs/1904.12584
(2019).

 
 
 

 
 [76] 
 
Emanuele Marconato,
Gianpaolo Bontempo, Elisa Ficarra,
Simone Calderara, Andrea Passerini, and
Stefano Teso. 2023a.

 
 Neuro Symbolic Continual Learning: Knowledge,
Reasoning Shortcuts and Concept Rehearsal.

 
 arXiv preprint arXiv:2302.01242 
(2023).

 
 
 

 
 [77] 
 
Emanuele Marconato,
Gianpaolo Bontempo, Elisa Ficarra,
Simone Calderara, Andrea Passerini, and
Stefano Teso. 2023b.

 
 Neuro Symbolic Continual Learning: Knowledge,
Reasoning Shortcuts and Concept Rehearsal.

 
 ArXiv abs/2302.01242
(2023).

 
 
 

 
 [78] 
 
Emanuele Marconato, Andrea
Passerini, and Stefano Teso.
2022.

 
 GlanceNets: Interpretabile, Leak-proof
Concept-based Models.

 
 ArXiv abs/2205.15612
(2022).

 
 
 

 
 [79] 
 
Andrei Margeloiu, Matthew
Ashman, Umang Bhatt, Yanzhi Chen,
Mateja Jamnik, and Adrian Weller.
2021.

 
 Do Concept Bottleneck Models Learn as Intended?

 
 ArXiv abs/2105.04289
(2021).

 
 
 

 
 [80] 
 
Thomas McGrath, Andrei
Kapishnikov, Nenad Toma vs .ev, Adam
Pearce, Demis Hassabis, Been Kim,
Ulrich Paquet, and Vladimir Kramnik.
2021.

 
 Acquisition of chess knowledge in AlphaZero.

 
 Proceedings of the National Academy of
Sciences of the United States of America 119
(2021).

 
 
 

 
 [81] 
 
Thomas McGrath, Andrei
Kapishnikov, Nenad Tomašev, Adam
Pearce, Martin Wattenberg, Demis
Hassabis, Been Kim, Ulrich Paquet, and
Vladimir Kramnik. 2022.

 
 Acquisition of chess knowledge in alphazero.

 
 Proceedings of the National Academy of
Sciences 119, 47
(2022), e2206625119.

 
 
 

 
 [82] 
 
Tomas Mikolov, Kai Chen,
Greg Corrado, and Jeffrey Dean.
2013.

 
 Efficient estimation of word representations in
vector space.

 
 arXiv preprint arXiv:1301.3781 
(2013).

 
 
 

 
 [83] 
 
Gregory L Murphy.
2002.

 
 The big book of concepts. A Bradford Book.

 
 
 
 
 

 
 [84] 
 
Gayda Mutahar and Tim
Miller. 2022.

 
 Concept-based Explanations using Non-negative
Concept Activation Vectors and Decision Tree for CNN Models.

 
 ArXiv abs/2211.10807
(2022).

 
 
 

 
 [85] 
 
Matthew R. O’Shaughnessy,
Gregory H. Canal, Marissa Connor,
Mark A. Davenport, and Christopher J.
Rozell. 2020.

 
 Generative causal explanations of black-box
classifiers.

 
 ArXiv abs/2006.13913
(2020).

 
 
 

 
 [86] 
 
Jeffrey Pennington,
Richard Socher, and Christopher D
Manning. 2014.

 
 Glove: Global vectors for word representation. In
 Proceedings of the 2014 conference on empirical
methods in natural language processing (EMNLP) .
1532–1543.

 
 
 

 
 [87] 
 
Andrew Slavin Ross,
Michael C Hughes, and Finale
Doshi-Velez. 2017.

 
 Right for the right reasons: Training
differentiable models by constraining their explanations.

 
 arXiv preprint arXiv:1703.03717 
(2017).

 
 
 

 
 [88] 
 
Dawid Rymarczyk, Adam
Pardyl, Jaroslaw Kraus, Aneta
Kaczy’nska, Marek Skomorowski, and
Bartosz Zieli’nski. 2021.

 
 ProtoMIL: Multiple Instance Learning with
Prototypical Parts for Whole-Slide Image Classification.

 
 
 

 
 [89] 
 
Mikołaj Sacha, Dawid
Rymarczyk, Łukasz Struski, Jacek
Tabor, and Bartosz Zieliński.
2023.

 
 ProtoSeg: Interpretable Semantic Segmentation With
Prototypical Parts. In Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision .
1481–1492.

 
 
 

 
 [90] 
 
Ainkaran Santhirasekaram,
Avinash Kori, Andrea Rockall,
Mathias Winkler, Francesca Toni, and
Ben Glocker. 2022.

 
 Hierarchical Symbolic Reasoning in Hyperbolic Space
for Deep Discriminative Models.

 
 arXiv preprint arXiv:2207.01916 
(2022).

 
 
 

 
 [91] 
 
Anirban Sarkar, Deepak
Vijaykeerthy, Anindya Sarkar, and
Vineeth N Balasubramanian.
2022.

 
 A framework for learning ante-hoc explainable
models via concepts. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition .
10286–10295.

 
 
 

 
 [92] 
 
Yoshihide Sawada and
Keigo Nakamura. 2022a.

 
 C-SENN: Contrastive Self-Explaining Neural
Network.

 
 ArXiv abs/2206.09575
(2022).

 
 
 

 
 [93] 
 
Yoshihide Sawada and
Keigo Nakamura. 2022b.

 
 Concept Bottleneck Model With Additional
Unsupervised Concepts.

 
 IEEE Access 10
(2022), 41758–41765.

 
 
 

 
 [94] 
 
Patrick Schramowski,
Wolfgang Stammer, Stefano Teso,
Anna Brugger, Franziska Herbert,
Xiaoting Shao, Hans-Georg Luigs,
Anne-Katrin Mahlein, and Kristian
Kersting. 2020.

 
 Making deep neural networks right for the right
scientific reasons by interacting with their explanations.

 
 Nature Machine Intelligence 
2, 8 (2020),
476–486.

 
 
 

 
 [95] 
 
Jessica Schrouff,
Sebastien Baur, Shaobo Hou,
Diana Mincu, Eric Loreaux,
Ralph Blanes, James Wexler,
Alan Karthikesalingam, and Been Kim.
2021.

 
 Best of both worlds: local and global explanations
with human-understandable concepts.

 
 arXiv preprint arXiv:2106.08641 
(2021).

 
 
 

 
 [96] 
 
Lisa Schut, Nenad
Tomasev, Tom McGrath, Demis Hassabis,
Ulrich Paquet, and Been Kim.
2023.

 
 Bridging the human-ai knowledge gap: Concept
discovery and transfer in alphazero.

 
 arXiv preprint arXiv:2310.16410 
(2023).

 
 
 

 
 [97] 
 
Gesina Schwalbe.
2022.

 
 Concept Embedding Analysis: A Review.

 
 arXiv preprint arXiv:2203.13909 
(2022).

 
 
 

 
 [98] 
 
Ramprasaath R Selvaraju,
Michael Cogswell, Abhishek Das,
Ramakrishna Vedantam, Devi Parikh, and
Dhruv Batra. 2017.

 
 Grad-cam: Visual explanations from deep networks
via gradient-based localization. In Proceedings of
the IEEE international conference on computer vision .
618–626.

 
 
 

 
 [99] 
 
Xiaoting Shao, Karl
Stelzner, and Kristian Kersting.
2022.

 
 Right for the right latent factors: Debiasing
generative models via disentanglement.

 
 arXiv preprint arXiv:2202.00391 
(2022).

 
 
 

 
 [100] 
 
Lloyd S Shapley et al . 
1953.

 
 A value for n-person games.

 
 (1953).

 
 
 

 
 [101] 
 
Radwa El Shawi, Youssef
Mohamed, and Sherif Sakr.
2021.

 
 Towards Automated Concept-based Decision
TreeExplanations for CNNs. In International
Conference on Extending Database Technology .

 
 
 

 
 [102] 
 
Haifeng Shen, Kewen Liao,
Zhibin Liao, Job Doornberg,
Maoying Qiao, Anton Van Den Hengel, and
Johan W Verjans. 2021.

 
 Human-AI interactive and continuous sensemaking: A
case study of image classification using scribble attention maps. In
 Extended Abstracts of the 2021 CHI Conference on
Human Factors in Computing Systems . 1–8.

 
 
 

 
 [103] 
 
Yujun Shen, Ceyuan Yang,
Xiaoou Tang, and Bolei Zhou.
2020.

 
 Interfacegan: Interpreting the disentangled face
representation learned by gans.

 
 IEEE transactions on pattern analysis and
machine intelligence 44, 4
(2020), 2004–2018.

 
 
 

 
 [104] 
 
Rui Shu, Yining Chen,
Abhishek Kumar, Stefano Ermon, and
Ben Poole. 2019.

 
 Weakly supervised disentanglement with guarantees.

 
 arXiv preprint arXiv:1910.09772 
(2019).

 
 
 

 
 [105] 
 
David Silver, Thomas
Hubert, Julian Schrittwieser, Ioannis
Antonoglou, Matthew Lai, Arthur Guez,
Marc Lanctot, Laurent Sifre,
Dharshan Kumaran, Thore Graepel,
et al . 2017.

 
 Mastering chess and shogi by self-play with a
general reinforcement learning algorithm.

 
 arXiv preprint arXiv:1712.01815 
(2017).

 
 
 

 
 [106] 
 
Sanchit Sinha, Mengdi
Huai, Jianhui Sun, and Aidong Zhang.
2022.

 
 Understanding and Enhancing Robustness of
Concept-based Models.

 
 ArXiv abs/2211.16080
(2022).

 
 
 

 
 [107] 
 
Youngjae Song, Sung Kuk
Shyn, and Kwang-su Kim.
2023.

 
 Img2Tab: Automatic Class Relevant Concept Discovery
from StyleGAN Features for Explainable Image Classification.

 
 arXiv preprint arXiv:2301.06324 
(2023).

 
 
 

 
 [108] 
 
Rahul Soni, Naresh Shah,
Chua Tat Seng, and Jimmy D. Moore.
2020.

 
 Adversarial TCAV - Robust and Effective
Interpretation of Intermediate Layers in Neural Networks.

 
 ArXiv abs/2002.03549
(2020).

 
 
 

 
 [109] 
 
Johannes Stallkamp, Marc
Schlipsing, Jan Salmen, and Christian
Igel. 2012.

 
 Man vs. computer: Benchmarking machine learning
algorithms for traffic sign recognition.

 
 Neural networks 32
(2012), 323–332.

 
 
 

 
 [110] 
 
Wolfgang Stammer, Marius
Memmel, Patrick Schramowski, and
Kristian Kersting. 2022.

 
 Interactive disentanglement: Learning concepts by
interacting with their prototype representations. In
 Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition . 10317–10328.

 
 
 

 
 [111] 
 
Wolfgang Stammer, Patrick
Schramowski, and Kristian Kersting.
2021.

 
 Right for the right concept: Revising
neuro-symbolic concepts by interacting with their explanations. In
 Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition . 3619–3629.

 
 
 

 
 [112] 
 
Mukund Sundararajan, Ankur
Taly, and Qiqi Yan. 2017.

 
 Axiomatic attribution for deep networks. In
 International conference on machine learning .
PMLR, 3319–3328.

 
 
 

 
 [113] 
 
Stefano Teso.
2019.

 
 Toward faithful explanatory active learning with
self-explainable neural nets. In Proceedings of
the Workshop on Interactive Adaptive Learning (IAL 2019) . CEUR Workshop
Proceedings, 4–16.

 
 
 

 
 [114] 
 
Stefano Teso, Öznur
Alkan, Wolfgang Stammer, and
Elizabeth M. Daly. 2022.

 
 Leveraging Explanations in Interactive Machine
Learning: An Overview.

 
 ArXiv abs/2207.14526
(2022).

 
 
 

 
 [115] 
 
Stefano Teso and
Kristian Kersting. 2019.

 
 Explanatory interactive machine learning. In
 Proceedings of the 2019 AAAI/ACM Conference on AI,
Ethics, and Society . 239–245.

 
 
 

 
 [116] 
 
Francesco Tonolini,
Bjørn Sand Jensen, and Roderick
Murray-Smith. 2019.

 
 Variational Sparse Coding. In
 Conference on Uncertainty in Artificial
Intelligence .

 
 
 

 
 [117] 
 
Thien Q. Tran, Kazuto
Fukuchi, Youhei Akimoto, and Jun
Sakuma. 2021.

 
 Unsupervised Causal Binary Concepts Discovery with
VAE for Black-box Model Explanation. In AAAI
Conference on Artificial Intelligence .

 
 
 

 
 [118] 
 
Johanna Vielhaben, Stefan
Blücher, and Nils Strodthoff.
2023.

 
 Multi-dimensional concept discovery (MCD): A
unifying framework with completeness guarantees.

 
 ArXiv abs/2301.11911
(2023).

 
 
 

 
 [119] 
 
Stanislav Vojíř and
Tomáš Kliegr. 2020.

 
 Editable machine learning models? A rule-based
framework for user studies of explainability.

 
 Advances in Data Analysis and
Classification 14, 4
(2020), 785–799.

 
 
 

 
 [120] 
 
Andong Wang, Wei-Ning
Lee, and Xiaojuan Qi. 2022.

 
 HINT: Hierarchical Neuron Concept Explainer. In
 Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition . 10254–10264.

 
 
 

 
 [121] 
 
Chong Wang, Yuyuan Liu,
Yuanhong Chen, Fengbei Liu,
Yu Tian, Davis J McCarthy,
Helen Frazer, and Gustavo Carneiro.
2023.

 
 Learning Support and Trivial Prototypes for
Interpretable Image Classification.

 
 arXiv preprint arXiv:2301.04011 
(2023).

 
 
 

 
 [122] 
 
Dan Wang, Xinrui Cui,
and Z Jane Wang. 2020.

 
 Chain: Concept-harmonized hierarchical inference
interpretation of deep convolutional neural networks.

 
 arXiv preprint arXiv:2002.01660 
(2020).

 
 
 

 
 [123] 
 
Leander Weber, Sebastian
Lapuschkin, Alexander Binder, and
Wojciech Samek. 2022.

 
 Beyond explaining: Opportunities and challenges of
XAI-based model improvement.

 
 Information Fusion (2022).

 
 
 

 
 [124] 
 
Wentao Wu, Hongsong Li,
Haixun Wang, and Kenny Q Zhu.
2012.

 
 Probase: A probabilistic taxonomy for text
understanding. In Proceedings of the 2012 ACM
SIGMOD international conference on management of data .
481–492.

 
 
 

 
 [125] 
 
Tete Xiao, Yingcheng Liu,
Bolei Zhou, Yuning Jiang, and
Jian Sun. 2018.

 
 Unified perceptual parsing for scene
understanding. In Proceedings of the European
conference on computer vision (ECCV) . 418–434.

 
 
 

 
 [126] 
 
Mengqi Xue, Qihan Huang,
Haofei Zhang, Lechao Cheng,
Jie Song, Minghui Wu, and
Mingli Song. 2022.

 
 ProtoPFormer: Concentrating on prototypical parts
in vision transformers for interpretable image recognition.

 
 arXiv preprint arXiv:2208.10431 
(2022).

 
 
 

 
 [127] 
 
Chih-Kuan Yeh, Been Kim,
Sercan Arik, Chun-Liang Li,
Tomas Pfister, and Pradeep Ravikumar.
2020.

 
 On completeness-aware concept-based explanations in
deep neural networks.

 
 Advances in Neural Information Processing
Systems 33 (2020),
20554–20565.

 
 
 

 
 [128] 
 
Chih-Kuan Yeh, Been Kim,
Sercan Ö. Arik, Chun-Liang Li,
Tomas Pfister, and Pradeep Ravikumar.
2019.

 
 On Completeness-aware Concept-Based Explanations in
Deep Neural Networks.

 
 arXiv: Learning (2019).

 
 
 

 
 [129] 
 
Kexin Yi, Jiajun Wu,
Chuang Gan, Antonio Torralba,
Pushmeet Kohli, and Josh Tenenbaum.
2018.

 
 Neural-symbolic vqa: Disentangling reasoning from
vision and language understanding.

 
 Advances in neural information processing
systems 31 (2018).

 
 
 

 
 [130] 
 
Mert Yuksekgonul, Maggie
Wang, and James Zou. 2022.

 
 Post-hoc concept bottleneck models.

 
 arXiv preprint arXiv:2205.15480 
(2022).

 
 
 

 
 [131] 
 
Mateo Espinosa Zarlenga,
Pietro Barbiero, Zohreh Shams,
Dmitry Kazhdan, Umang Bhatt,
Adrian Weller, and Mateja Jamnik.
2023a.

 
 Towards Robust Metrics for Concept Representation
Evaluation.

 
 ArXiv abs/2301.10367
(2023).

 
 
 

 
 [132] 
 
Mateo Espinosa Zarlenga,
Katherine M Collins, Krishnamurthy
Dvijotham, Adrian Weller, Zohreh Shams,
and Mateja Jamnik. 2023b.

 
 Learning to Receive Help: Intervention-Aware
Concept Embedding Models.

 
 arXiv preprint arXiv:2309.16928 
(2023).

 
 
 

 
 [133] 
 
Mateo Espinosa Zarlenga,
Zohreh Shams, Michael Edward Nelson,
Been Kim, and Mateja Jamnik.
2023c.

 
 Tabcbm: Concept-based interpretable neural networks
for tabular data.

 
 Transactions on Machine Learning Research 
(2023).

 
 
 

 
 [134] 
 
Quanshi Zhang, Ruiming
Cao, Feng Shi, Ying Nian Wu, and
Song-Chun Zhu. 2017.

 
 Knowledge Via An Explanatory Graph.

 
 
 

 
 [135] 
 
Ruihan Zhang, Prashan
Madumal, Tim Miller, Krista A Ehinger,
and Benjamin IP Rubinstein.
2021a.

 
 Invertible concept-based explanations for cnn
models with non-negative concept activation vectors. In
 Proceedings of the AAAI Conference on Artificial
Intelligence , Vol. 35. 11682–11690.

 
 
 

 
 [136] 
 
Yu Zhang, Peter
Tiňo, Aleš Leonardis, and
Ke Tang. 2021b.

 
 A survey on neural network interpretability.

 
 IEEE Transactions on Emerging Topics in
Computational Intelligence 5, 5
(2021), 726–742.

 
 
 

 
 [137] 
 
Bolei Zhou, Yiyou Sun,
David Bau, and Antonio Torralba.
2018.

 
 Interpretable basis decomposition for visual
explanation. In Proceedings of the European
Conference on Computer Vision (ECCV) . 119–134.