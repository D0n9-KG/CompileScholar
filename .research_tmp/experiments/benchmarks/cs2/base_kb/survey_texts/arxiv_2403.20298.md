Review-Based Hyperbolic Cross-Domain Recommendation 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.20298v3 [cs.IR] 19 Mar 2025 
 
 

# Review-Based Hyperbolic Cross-Domain Recommendation

 CCS:  Information systems Recommender systems 
 
 
 Yoonhyuk Choi
 
 
 
 Affiliation:  
Samsung , Seoul , Republic of Korea 
 
 email: chldbsgur123@gmail.com 
 
 , 
 Jiho Choi
 
 
 
 Affiliation:  
KAIST , Seoul , Republic of Korea 
 
 email: jihochoi @ kaist.ac.kr 
 
 , 
 Taewook Ko
 
 
 
 Affiliation:  
Samsung , Seoul , Republic of Korea 
 
 email: taewook.ko @ snu.ac.kr 
 
 and 
 Chong-Kwon Kim
 
 
 
 Affiliation:  Korea Institute of Energy Technology , Naju , Republic of Korea 
 
 email: ckim @ kentech.ac.kr 
 

 Abstract. 
 
 The issue of data sparsity poses a significant challenge to recommender systems. Recently, algorithms that leverage side information (review texts) or Cross-Domain Recommendation (CDR) have emerged. Nevertheless, existing methodologies assume an Euclidean embedding space, encountering difficulties in accurately representing richer text information and managing complex user-item interactions. This paper advocates a hyperbolic CDR approach for modeling review-based user-item relationships. We first emphasize that conventional distance-based domain alignment techniques may cause problems because small modifications in hyperbolic geometry result in magnified perturbations, ultimately leading to the collapse of hierarchical structures. To address this challenge, we propose hierarchy-aware embedding and domain alignment schemes that adjust the scale to extract domain-shareable information without disrupting structural forms. Extensive experiments substantiate the efficiency, robustness, and scalability of the proposed model. The source code is given here 1 1 
 1 
 
 
 
 https://github.com/ChoiYoonHyuk/HEAD .

 
 
 
 Keywords:  Recommender system, review-based cross-domain recommendation, hyperbolic embedding, hierarchy-aware domain alignment
 
 
 Figure 1 . The geometric properties of hyperbolic space require that popular (or most interacted) nodes be placed near the origin. Let us assume that user u 1 u_{1} purchased an item i 1 i_{1} (left). Given that general algorithms bring relevant nodes closer together, after the update, a structural collapse occurs (right) as i 1 i_{1} moves farther from the origin 
 
 

## 1. Introduction

 
 A recommender system has evolved into a fundamental tool across real-world applications ( Lu et al., 2015 ; Zhang et al., 2019 ) such as Amazon, and Tripadvisor. Despite remarkable popularity and commercial successes, the performance can be inflicted heavily in data-scarce scenarios. The impediments of data sparsity encompass issues like cold-start problems, which have gained academic focus recently. Numerous research endeavors have employed various forms of side information ranging from social relationships ( Kazienko et al., 2011 ) , and hierarchical interactions ( Liu et al., 2019 ) to item images ( Addagarla and Amalanathan, 2020 ) . In particular, textual data in the form of reviews has become one of the most extensively utilized sources ( Zheng et al., 2017 ; Al-Ghuribi and Noah, 2019 ; Srifi et al., 2020 ; Choi et al., 2022 ) . These endeavors have demonstrated a measure of effectiveness in ameliorating sparsity concerns. However, inherent limitations persist in addressing fundamental issues, especially when the extent of interaction is insufficient.

 
 
 To handle this problem, recent method strides in Cross-Domain Recommendation (CDR) ( Hu et al., 2018 ; Yuan et al., 2019 ; Fu et al., 2019 ; Yuan et al., 2020 ; Yang et al., 2021 ; Hande et al., 2021 ) . These algorithms commonly exploit the information from a source domain, characterized by abundant interactions relative to a target domain to extract domain-shareable information. Some approaches concentrate on duplicate users across both domains ( Yuan et al., 2019 ; Li and Tuzhilin, 2020 ; Liu et al., 2021 ; Guo et al., 2021 ) . However, it is essential to note that these user-binding strategies may face constraints arising from the absence of overlapping (duplicate) users ( Kang et al., 2019 ) . Alternatively, more flexible methods that operate independent of specific users or contexts have been introduced ( Cai et al., 2019 ; Zhao et al., 2020 ; Krishnan et al., 2020 ) . This trend has prompted the development of disentangled representation learning techniques ( Gretton et al., 2005 ; Bousmalis et al., 2016 ; Li et al., 2019 ; Peng et al., 2019 ) , which can concurrently extract both domain-specific and domain-shareable knowledge. More recently, novel approaches contrived on review-based disentangled representation learning ( Cao et al., 2022a ; Choi et al., 2022 ) free of duplicate users or contexts have demonstrated state-of-the-art performance.

 
 
 Nonetheless, the above methods rely on Euclidean geometry for embedding ( Globerson et al., 2004 ; Khoshneshin and Street, 2010 ) , which has been shown to be inadequate for modeling user-item bipartite graphs whose node degrees follow a power-law distribution ( Clauset et al., 2009 ) . This inadequacy leads to distortions in the embedding due to the exponential growth in the graph’s volume with its radius. In addition, the application of hyperbolic geometry to cross-domain recommendation (CDR) remains relatively unexplored ( Xu and Cai, 2023 ) , whereas recent studies on hyperbolic recommendation ( Gulcehre et al., 2018 ; Khrulkov et al., 2020 ; Feng et al., 2020 ; Vinh Tran et al., 2020 ) focus on a single domain.

 
 
 In this paper, we propose a novel strategy for hyperbolic CDR since previous methods rely on features extracted from two closely related domains ( Cao et al., 2022a ; Zhao et al., 2023b ; Zhao et al., 2023a ) , potentially losing hierarchical information. For example, directly reducing the distance between two items, i 1 i_{1} and u 1 u_{1} (Figure 1 ), purchased by a user without considering their positions can negatively impact the overall representation. As illustrated, distance-based minimization can lead to hierarchical collapse, as node positions cannot be preserved in this process. Specifically, if the degree of node i 1 i_{1} is greater than others, placing a node with fewer interactions u 1 u_{1} closer to the origin decreases the advantages of an exponentially increasing space. This context highlights the need for clever mechanisms that facilitate knowledge transfer while preserving a tree-like structure. To address this, we propose a solution with two strategies: degree-based normalization and structure alignment, substantiated through theoretical insights and empirical evidence from various experiments. In summary, our contributions are outlined as follows:

 
 • 
 
 We propose a novel CDR algorithm, called H yperbolic E mbedding and Hierarchy- A ware D omain Disentanglement (HEAD), which enhances the previous review-based domain disentanglement by incorporating hyperbolic geometry.

 

 • 
 
 We propose degree-based hierarchy alignment and scale adjustment to enhance the knowledge transfer between two domains. To our knowledge, this is the first attempt to achieve domain disentanglement in a hyperbolic space.

 

 • 
 
 We present theoretical understandings to prove the importance of hierarchy preservation in domain disentanglement.

 

 • 
 
 We conducted various experiments to verify the effectiveness of our method and the accuracy of the theoretical analysis.

 

 
 
 
 

## 2. Related Work

 
 In this section, we introduce the advent of recommender algorithms categorizing them as follows; (1) Review-based recommendations (using side information), (2) Cross-domain recommendations (knowledge transfer), and (3) Hyperbolic recommendations that can preserve hierarchical information.

 
 

### 2.1. Review-Based Recommendation

 
 The explosive progress in text convolution techniques has ignited great attention in review-based recommender systems ( Zheng et al., 2017 ; Chen et al., 2019b ; Chen et al., 2019a ; Dong et al., 2020 ) . For example, DeepCoNN ( Zheng et al., 2017 ) employs two parallel convolutional neural networks (CNNs), and others further utilize an attention mechanism to exploit important words ( Seo et al., 2017 ; Chen et al., 2018 ; Tay et al., 2018 ; Dong et al., 2020 ) . Although these methods highlight the importance of review texts, their major limitation lies in the confined scale of target domains and the transfer of noisy data ( Sachdeva and McAuley, 2020 ; Zeng et al., 2021 ) . To address these issues, cross-domain recommendation and disentanglement techniques have emerged to acquire useful information from richer domains.

 
 
 

### 2.2. Cross-Domain Recommendation

 
 Cross-domain recommendation (CDR) utilizes additional information in extra (source) domain to address the sparseness in a target. Generally, these methods capture latent information from rating matrices or review texts ( Fu et al., 2019 ; Krishnan et al., 2020 ) to capture and transfer knowledge authored by overlapping users in both source and target domains ( Elkahky et al., 2015 ; Man et al., 2017 ; Zhu et al., 2021 ) . Though certain methods ( Wang et al., 2018 ; Zhao et al., 2020 ) underscore the significance of employing non-overlapping users for generalization, they lack examination of what information would be efficiently transferred. Thus, recent studies have shifted focus towards identifying the most relevant aspects between two domains, commonly referred to as domain-shareable features. The foundational mechanism often commences with domain adaptation ( Ramakrishnan et al., 2018 ; Yuan et al., 2019 ; Bonab et al., 2021 ; Cao et al., 2022b ; Chen et al., 2023 ) , which captures domain-shareable features through adversarial training. Advanced techniques that extract domain-specific and domain-shareable features simultaneously have been introduced more recently, collectively known as disentangled representation learning. These include MMT ( Krishnan et al., 2020 ) , DADA ( Peng et al., 2019 ) , DisenCDR ( Cao et al., 2022a ) , and SER ( Choi et al., 2022 ) . The pivotal aspect of this approach lies in domain disentanglement which strives to identify useful information for knowledge transfer.

 
 
 

### 2.3. Hyperbolic Recommendation

 
 Most prior research on recommendation systems has primarily been conducted in an Euclidean space. These methods have demonstrated decent performance, but a challenge has been raised regarding their adequacy in modeling hierarchical structures such as user-item interactions or word vectors ( Tifrea et al., 2018 ) . Recent studies ( Feng et al., 2020 ; Wang et al., 2021 ; Su et al., 2023 ) utilize a hyperbolic space for representation learning, including informative collaborative filtering ( Yang et al., 2022a ; Li et al., 2022 ; Wang et al., 2023 ) with geometric regularization ( Yang et al., 2022b ) . However, most of them utilize a single-domain dataset, which is susceptible to data sparsity. To address this problem, several researchers have integrated CDR with hyperbolic space embedding, but they disregard domain disentanglement ( Xu and Cai, 2023 ; Guo et al., 2023 ) and hierarchy structure preservation ( Zhang et al., 2022 ) , which are particularly crucial for accurate knowledge transfer in an exponentially expanded space. Thus, we aim to suggest a new hyperbolic CDR that preserves the structural property to better embed the user-item interactions.

 
 
 
 

## 3. Preliminaries

 
 We start with the basic concepts of manifolds and hyperbolic geometry. Differential geometry defines three space types: hyperbolic, Euclidean, and spherical, based on curvatures. Especially, a hyperbolic space is one type of non-Euclidean space, which has a constant negative curvature at all points. In the literature, various mathematical formulations can be utilized to describe hyperbolic spaces, such as the Riemannian manifold ( Lin and Zha, 2008 ) , Poincaré ball ( Nickel and Kiela, 2017 ) , and Lorentz model ( Nickel and Kiela, 2018 ) . We employ the Poincaré ball for visualization ( Gulcehre et al., 2018 ) and the Lorentz model for numerical operation ( Law et al., 2019 ) , respectively.

 
 
 Poincaré ball. The representation of this model can be defined as 𝒫 d = ( ℬ d , g x ℬ ) \mathcal{P}^{d}=(\mathcal{B}^{d},g^{\mathcal{B}}_{x}) , which stands for the open n n -dimensional unit ball ℬ d = { x ∈ ℛ d : k ​ ‖ x ‖ 1 } \mathcal{B}^{d}=\{x\in\mathcal{R}^{d}:k||x|| 1\} and hyperbolic feature g x ℬ g^{\mathcal{B}}_{x} :

 

 
 (1) | 
 | 
 g x ℬ = ( 2 1 − k ​ ‖ x ‖ 2 ) 2 ​ g x E , g^{\mathcal{B}}_{x}=\mathrm{({2\over 1-k||x||^{2}})}^{2}g^{E}_{x}, | 
 | 
 

 where k k is the radius of the ball. The above equation converts an Euclidean metric tensor g E g^{E} to a hyperbolic one. If k = 0 k=0 , we can easily infer that the ball is identical to the Euclidean space. Also, a distance function on 𝒫 \mathcal{P} is defined as below:

 

 
 (2) | 
 | 
 d 𝒫 ​ ( x , y ) = k ​ arcosh ​ ( 1 + 2 ​ k ​ ‖ x − y ‖ 2 ( k − ‖ x ‖ 2 ) ​ ( k − ‖ y ‖ 2 ) ) \mathrm{d_{\mathcal{P}}(x,y)=\sqrt{k}\ arcosh\left(1+2k{||x-y||^{2}\over(k-||x||^{2})(k-||y||^{2})}\right)} | 
 | 
 

 
 
 Lorentz model. Similarly, the Lorentz model is defined as ℒ d = ( ℋ d , g x ℋ ) \mathcal{L}^{d}=(\mathcal{H}^{d},g^{\mathcal{H}}_{x}) , where ℋ d = { x ∈ ℛ d + 1 : x , x ℒ = − k , x 0 0 } \mathcal{H}^{d}=\{x\in\mathcal{R}^{d+1}: x,x _{\mathcal{L}}=-k,x_{0} 0\} . Here, , ℒ , _{\mathcal{L}} is the Lorentizan inner product:

 

 
 (3) | 
 | 
 x , y ℒ = − x 0 ​ y 0 + ∑ i = 1 n x i ​ y i , \mathrm{ x,y _{\mathcal{L}}=-x_{0}y_{0}+\sum^{n}_{i=1}x_{i}y_{i}}, | 
 | 
 

 and g x ℋ = d ​ i ​ a ​ g ​ ( − 1 , 1 , … , 1 ) g^{\mathcal{H}}_{x}=diag(-1,1,...,1) is a positive-definite metric tensor to calculate a distance of two points x , y ∈ ℋ d x,y\in\mathcal{H}^{d} as follows:

 

 
 (4) | 
 | 
 d ℒ ​ ( x , y ) = k ​ arcosh ​ ( − x , y ℒ k ) \mathrm{d_{\mathcal{L}}(x,y)=\sqrt{k}\ arcosh(-{ x,y _{\mathcal{L}}\over k})} | 
 | 
 

 For a certain point x ∈ ℋ d x\in\mathcal{H}^{d} in hyperbolic space, we can define the tangent space centered at x x as below:

 

 
 (5) | 
 | 
 𝒯 x ​ ℋ d = { v ∈ ℛ d + 1 : v , x ℒ = 0 } , \mathrm{\mathcal{T}_{x}\mathcal{H}^{d}=\{v\in\mathcal{R}^{d+1}: v,x _{\mathcal{L}}=0\}}, | 
 | 
 

 where the orthogonality holds for all v v concerning the Lorentz scalar product. Using these characteristics, conversions between the tangent and hyperbolic space can be achieved through exponential and logarithmic maps as follows.

 
 • 
 
 (Exponential map) 𝒯 x ​ ℋ d → ℋ d \mathcal{T}_{x}\mathcal{H}^{d}\rightarrow\mathcal{H}^{d} projects v v onto hyperbolic space as,

 

 
 (6) | 
 | 
 exp x ​ ( v ) = cosh ⁡ ( ‖ v ‖ ℒ k ) ​ x + k ​ sinh ​ ( ‖ v ‖ ℒ k ) ​ v ‖ v ‖ ℒ \mathrm{exp_{x}(v)=cosh({||v||_{\mathcal{L}}\over\sqrt{k}})x+\sqrt{k}sinh({||v||_{\mathcal{L}}\over\sqrt{k}}){v\over||v||_{\mathcal{L}}}} | 
 | 
 

 

 • 
 
 (Logarithmic map) ℋ d → 𝒯 x ​ ℋ d \mathcal{H}^{d}\rightarrow\mathcal{T}_{x}\mathcal{H}^{d} projects v v back to Euclidean space as,

 

 
 (7) | 
 | 
 log x ​ ( v ) = d ℒ ​ ( x , v ) ​ v + 1 k ​ x , v ℒ ​ x ‖ v + 1 k ​ x , v ℒ ​ x ‖ ℒ \mathrm{log_{x}(v)=d_{\mathcal{L}}(x,v){v+{1\over k} x,v _{\mathcal{L}}x\over||v+{1\over k} x,v _{\mathcal{L}}x||}_{\mathcal{L}}} | 
 | 
 

 

 
 
 
 

## 4. Methodology

 
 In Figure 2 , we illustrate the overall architecture of our model, called HEAD ( H yperbolic E mbedding and Hierarchy- A ware D omain Disentanglement, which is comprised of the following key components:

 
 
 
 • 
 
 Word embedding. This part vectorizes reviews using pre-trained word embedding. In contrast to the previous methods that adopt Euclidean embedding such as word2vec 2 2 
 2 
 
 
 
 https://code.google.com/archive/p/word2vec ( Mikolov et al., 2013 ) or Euclidean GloVe 3 3 
 3 
 
 
 
 https://nlp.stanford.edu/projects/glove ( Pennington et al., 2014 ) , we employ the Poincaré Glove 4 4 
 4 
 
 
 
 https://github.com/alex-tifrea/poincare_glove ( Tifrea et al., 2018 ) which better preserves the hierarchical property of words.

 

 • 
 
 Feature extraction. This module elicits pertinent information from embedded documents using three types of feature extractors (FEs) ( Choi et al., 2022 ) ; the shared FE focuses on domain-shareable knowledge for transfer while the source and target FEs capture the domain-specific features.

 

 • 
 
 Hierarchy-aware embedding and domain disentanglement. Extracted features are aligned hierarchically and then knowledge is transferred while retaining this structure. This also reinforces the separability of domain discriminator.

 

 • 
 
 Prediction and optimization. Outputs are integrated and projected back to the hyperbolic space for prediction. A marginal ranking loss is computed for optimization.

 

 
 
 
 Figure 2 . The overall framework of the Hierarchy-Aware Hyperbolic Embedding and Domain Disentanglement (HEAD) scheme. The (1)-(3) represents three types of loss functions 
 
 

### 4.1. Word Embedding 

 
 Each data record follows a format ( u, i, y u , i y_{u,i} , r u , i r_{u,i} ), which means that a user u u purchased an item i i and left a rating y u , i y_{u,i} and a review r u , i r_{u,i} . Assume that we are concerned with a possibly unseen rating y u , i y_{u,i} . We first aggregate all reviews of u u and i i . Specifically, for a user u u , we gather all reviews written by her except for the specific pair r u , i r_{u,i} (not available during inference) and consider them as a single document R u R_{u} . Likewise, one can construct the collection of item reviews, R i R_{i} . Here, we ignore the temporal sequence of ratings or reviews. Finally, we apply the word embedding function to R u R_{u} and R i R_{i} . Although many strategies are applicable (e.g, word2vec 5 5 
 5 
 
 
 
 https://code.google.com/archive/p/word2vec ( Mikolov et al., 2013 ) ), we focus on the Euclidean- ( Pennington et al., 2014 ) and Poincaré Glove ( Tifrea et al., 2018 ) here.

 
 • 
 
 (Euclidean Glove) increases the co-occurrence probability ( X i ​ j X_{ij} ) of the central word ( w i w_{i} ) and its neighbor words ( w ~ j \tilde{w}_{j} ) as,

 

 
 (8) | 
 | 
 min ⁡ ∑ i , j V w ⁡ f ⁡ ( X ij ) ​ ( w i T ​ w ~ j + b i + b ~ j − logX ij ) 2 , \mathrm{\min_{w}\sum^{V}_{i,j}f(X_{ij})(w^{T}_{i}\tilde{w}_{j}+b_{i}+\tilde{b}_{j}-logX_{ij})^{2}}, | 
 | 
 

 where b b stands for the bias term.

 

 • 
 
 (Poincaré Glove) replaces w i T ​ w ~ j w^{T}_{i}\tilde{w}_{j} with − h ⁡ ( d 𝒫 ​ ( w i T , w ~ j ) ) -h(d_{\mathcal{P}}(w^{T}_{i},\tilde{w}_{j})) as,

 

 
 (9) | 
 | 
 min ⁡ ∑ i , j V w ⁡ f ⁡ ( X ij ) ​ ( − h ⁡ ( d 𝒫 ​ ( w i T , w ~ j ) ) + b i + b ~ j − logX ij ) 2 , \mathrm{\min_{w}\sum^{V}_{i,j}f(X_{ij})(-h(d_{\mathcal{P}}(w^{T}_{i},\tilde{w}_{j}))+b_{i}+\tilde{b}_{j}-logX_{ij})^{2}}, | 
 | 
 

 where d 𝒫 d_{\mathcal{P}} is a distance function in Eq. 2 and h ⁡ ( x ) = c ​ o ​ s ​ h 2 ​ ( x ) h(x)=cosh^{2}(x) .

 

 
 The Euclidean and Poincaré renditions 6 6 
 6 
 
 
 
 https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn have been trained using 1.4 billion tokens from English Wikipedia, and we compare them as pre-trained word embedding f ⁡ ( ⋅ ) f(\cdot) in Table 2 . Using this, the textual documents R u R_{u} and R i R_{i} are mapped into the matrix R u 𝔼 = f ⁡ ( R u ) , R i 𝔼 = f ⁡ ( R i ) \mathrm{R^{\mathbb{E}}_{u}=f(R_{u}),R^{\mathbb{E}}_{i}=f(R_{i})} of ℛ n × d \mathcal{R}^{n\times d} , where n n is the vocabulary size and d d is the embedding dimension. We project these matrices onto the hyperbolic space using the exponential map in Eq. 6 as below:

 

 
 (10) | 
 | 
 R u ℍ = exp o ​ ( R u 𝔼 ) = [ cosh ⁡ ( ‖ R u 𝔼 ‖ ) , sinh ⁡ ( ‖ R u 𝔼 ‖ ) ​ R u 𝔼 ‖ R u 𝔼 ‖ ] R i ℍ = exp o ​ ( R i 𝔼 ) = [ cosh ⁡ ( ‖ R i 𝔼 ‖ ) , sinh ⁡ ( ‖ R i 𝔼 ‖ ) ​ R i 𝔼 ‖ R i 𝔼 ‖ ] \begin{gathered}\mathrm{R^{\mathbb{H}}_{u}=exp_{o}(R^{\mathbb{E}}_{u})=[cosh(||R^{\mathbb{E}}_{u}||),sinh(||R^{\mathbb{E}}_{u}||){R^{\mathbb{E}}_{u}\over||R^{\mathbb{E}}_{u}||}]}\\
\mathrm{R^{\mathbb{H}}_{i}=exp_{o}(R^{\mathbb{E}}_{i})=[cosh(||R^{\mathbb{E}}_{i}||),sinh(||R^{\mathbb{E}}_{i}||){R^{\mathbb{E}}_{i}\over||R^{\mathbb{E}}_{i}||}]}\end{gathered} | 
 | 
 

 We set the curvature k = 1 k=1 in Eq. 6 for simplicity.

 
 
 

### 4.2. Feature Extraction 

 
 The above word embedding procedure is applied to the source and target domains, respectively.
Given word embedding R u ℍ R^{\mathbb{H}}_{u} and R i ℍ R^{\mathbb{H}}_{i} from each domain, we aim to extract useful information. Several feature extraction strategies have been proposed, including domain adaptation ( Yuan et al., 2019 ) , variational reconstruction ( Liu et al., 2022 ) , personalized transfer ( Zhu et al., 2022 ) , contrastive learning ( Xie et al., 2022 ) , and domain disentanglement ( Peng et al., 2019 ) . Among them, we adopt the domain disentanglement algorithm, which does not obligate user overlapping in both domains ( Cao et al., 2022a ) . For this, as illustrated in Figure 2 , we employ three types of feature extractors (FEs), all of which consist of simple multi-channel Convolutional Neural Networks (CNNs). Specifically, the shared FE processes datasets from both domains, while the source and target FEs deal with the documents from their domains only (please refer to ( Choi et al., 2022 ) for more details). For the sake of simplicity, we focus on the mechanisms in the source domain since the target domain procedure is the same. The feature extraction process is given by:

 

 
 (11) | 
 | 
 S u = F s ​ ( log o ⁡ ( R u ℍ ) ) , S i = F s ​ ( log o ⁡ ( R i ℍ ) ) S ^ u = F h ​ ( log o ⁡ ( R u ℍ ) ) , S ^ i = F h ​ ( log o ⁡ ( R i ℍ ) ) \begin{gathered}\mathrm{S_{u}=F_{s}(\log_{o}(R^{\mathbb{H}}_{u})),\,\,S_{i}=F_{s}(\log_{o}(R^{\mathbb{H}}_{i}))}\\
\mathrm{\widehat{S}_{u}=F_{h}(\log_{o}(R^{\mathbb{H}}_{u})),\,\,\widehat{S}_{i}=F_{h}(\log_{o}(R^{\mathbb{H}}_{i}))}\\
\end{gathered} | 
 | 
 

 As illustrated, the hyperbolic embedding of the user and item is projected back to Euclidean space using a logarithmic map in Equation 7 , followed by the application of CNNs ( F s F_{s} and F h F_{h} ). Consequently, the source domain yields four outputs from two feature extractors, denoted as S u , S i , S ^ u , S ^ i {S_{u},S_{i},\widehat{S}_{u},\widehat{S}_{i}} (depicted in the middle of Figure 2 ). Additional insights into CNNs can be found in ( Zheng et al., 2017 ) .

 
 
 Remark. The large language models (e.g., LLaMA ( Touvron et al., 2023 ) , ChatGPT-4 ( Achiam et al., 2023 ) ) may replace CNNs if their parameters can be fine-tuned.

 
 
 

### 4.3. Hierarchy-Aware Hyperbolic Embedding and Domain Disentanglement

 
 We propose two constraints to achieve domain disentanglement between the extracted features while preserving the hierarchy.

 
 

#### 4.3.1. Hierarchy-aware hyperbolic embedding. 

 
 Recent work ( Khrulkov et al., 2020 ) reveals that the uncertainty decreases as the embedding gets closer to the boundary of the Poincaré ball. HRCF ( Yang et al., 2022b ) suggests a hyperbolic regularization optimized for the characteristics of a power-law distribution. Specifically, users or items with many interactions (dense) are pulled to the center, while sparse ones are placed near the boundary. For this, HRCF identifies the root as the average of the entire embedding to make it as an origin below:

 

 
 (12) | 
 | 
 S u root = 1 N u ​ ∑ u ′ = 1 N u 1 2 ​ ( S u u ′ + S ^ u u ′ ) , S i root = 1 N i ​ ∑ i ′ = 1 N i 1 2 ​ ( S i i ′ + S ^ i i ′ ) \mathrm{S_{u}^{root}={1\over N_{u}}\sum^{N_{u}}_{u^{\prime}=1}{1\over 2}(S^{u^{\prime}}_{u}+\widehat{S}^{u^{\prime}}_{u}}),\,\,\,\mathrm{S_{i}^{root}={1\over N_{i}}\sum^{N_{i}}_{i^{\prime}=1}{1\over 2}(S^{i^{\prime}}_{i}+\widehat{S}^{i^{\prime}}_{i}}) | 
 | 
 

 The N u N_{u} and N i N_{i} are the total number of users and items, respectively. Then, they apply so-called root alignment that locates the nodes to be separated from the origin as below:

 

 
 (13) | 
 | 
 S u norm = 1 N u ​ ∑ u ′ = 1 N u ‖ S u u ′ − S u root ‖ 2 2 , S i norm = 1 N i ​ ∑ i ′ = 1 N i ‖ S i i ′ − S i root ‖ 2 2 \mathrm{S_{u}^{norm}={1\over N_{u}}\sum^{N_{u}}_{u^{\prime}=1}||S_{u}^{u^{\prime}}-S_{u}^{root}||^{2}_{2},\,\,\,S_{i}^{norm}={1\over N_{i}}\sum^{N_{i}}_{i^{\prime}=1}||S_{i}^{i^{\prime}}-S_{i}^{root}||^{2}_{2}} | 
 | 
 

 The loss function is the inverse of the above equation, which aims to decrease the central density. However, this strategy might not sufficiently reflect the popularity of nodes since it simply pushes all nodes away from the center. To alleviate this problem, we suggest to modify Eq. 13 as follows.

 
 
 Proposition 4.1 (Hierarchy-aware hyperbolic embedding). 
 
 The loss function is normalized based on the maximum node degree, max ⁡ ( d ) \max(d) , as follows: 

 

 
 (14) | 
 | 
 S u , deg norm \displaystyle\mathrm{S_{u,deg}^{norm}} | 
 = 1 N u ​ ∑ u ′ = 1 N u max ⁡ ( d u ) − d u ′ max ⁡ ( d u ) ​ ‖ S u u ′ − S u root ‖ 2 2 \displaystyle=\mathrm{{1\over N_{u}}\sum^{N_{u}}_{u^{\prime}=1}{\max(d_{u})-d_{u^{\prime}}\over\max(d_{u})}||S_{u}^{u^{\prime}}-S_{u}^{root}||^{2}_{2}} | 
 | 
 
 
 (15) | 
 | 
 S i , deg norm \displaystyle\mathrm{S_{i,deg}^{norm}} | 
 = 1 N i ​ ∑ i ′ = 1 N i max ⁡ ( d i ) − d i ′ max ⁡ ( d i ) ​ ‖ S i i ′ − S i root ‖ 2 2 \displaystyle=\mathrm{{1\over N_{i}}\sum^{N_{i}}_{i^{\prime}=1}{\max(d_{i})-d_{i^{\prime}}\over\max(d_{i})}||S_{i}^{i^{\prime}}-S_{i}^{root}||^{2}_{2}} | 
 | 
 

 Notation d u ′ d_{u^{\prime}} and d i ′ d_{i^{\prime}} stands for the degrees of user u ′ u^{\prime} and item i ′ i^{\prime} , respectively. Through this, we can place popular users and items near the origin, while pushing low-degree nodes towards the boundary. 

 
 
 
 Proof . see proof of proposition 4.1 in Section 4.6 .

 
 
 The loss in the target domain can be retrieved similarly. Then, we can define the hierarchical embedding loss as below:

 

 
 (16) | 
 | 
 ℒ emb = 1 / S u , deg norm + S i , deg norm + T u , deg norm + T i , deg norm \mathrm{\mathcal{L}_{emb}=1/\sqrt{S_{u,deg}^{norm}+S_{i,deg}^{norm}+T_{u,deg}^{norm}+T_{i,deg}^{norm}}} | 
 | 
 

 
 
 

#### 4.3.2. Hierarchy-aware domain disentanglement. 

 
 Knowledge transfer has gained substantial attention in the field of cross-domain recommendation. In this regard, we align with the recently proposed domain disentanglement algorithm ( Cao et al., 2022a ; Choi et al., 2022 ) , which operates without the requirement of overlapping users. The fundamental mechanism of domain disentanglement can be delineated as follows: (1) domain-specific features S , T S,T should be readily inferable to their originating domains to mitigate domain discrepancies, and (2) domain-shareable features S ^ , T ^ \widehat{S},\widehat{T} ought to encapsulate domain-indiscriminative information, ensuring pairwise independence between domains ( Gretton et al., 2005 ; Li et al., 2019 ; Nema et al., 2021 ) . For this, we first concatenate the user and item vectors from each feature extractor as below:

 

 
 (17) | 
 | 
 S = [ S u ⊕ S i ] , T = [ T u ⊕ T i ] S ~ = [ S ^ u ⊕ S ^ i ] , T ~ = [ T ^ u ⊕ T ^ i ] \begin{gathered}\mathrm{S=[S_{u}\oplus S_{i}],\,\,\,T=[T_{u}\oplus T_{i}]}\\
\mathrm{\tilde{S}=[\widehat{S}_{u}\oplus\widehat{S}_{i}],\,\,\,\tilde{T}=[\widehat{T}_{u}\oplus\widehat{T}_{i}]}\end{gathered} | 
 | 
 

 Before delving into the scale alignment module, we introduce the following discussion on previous disentanglement algorithms. Prior methods of domain disentanglement, such as those proposed by ( Peng et al., 2019 ; Krishnan et al., 2020 ; Cao et al., 2022a ; Choi et al., 2022 ; Zhang et al., 2022 ) merely focus on preserving the scale of the extracted features. In detail, they simply forward these features to the domain discriminator ( F d F_{d} ) in the following manner:

 

 
 (18) | 
 | 
 d S = F d ​ ( S ) , d T = F d ​ ( T ) d ~ S = F d ​ ( g ⁡ ( S ~ ) ) , d ~ T = F d ​ ( g ⁡ ( T ~ ) ) \begin{gathered}d_{S}=F_{d}(S),\,\,\,d_{T}=F_{d}(T)\\
\tilde{d}_{S}=F_{d}(g(\tilde{S})),\,\,\,\tilde{d}_{T}=F_{d}(g(\tilde{T}))\end{gathered} | 
 | 
 

 
 
 The F d F_{d} consists of the two layers of a fully connected neural network and the notation d S , d T ∈ { 0 , 1 } d_{S},d_{T}\in\{0,1\} denotes the predicted domain (0/1 are the source/target). Additionally, g ⁡ ( ⋅ ) g(\cdot) signifies the Gradient Reversal Layer (GRL), which remains inactive during the forward propagation but reverses the sign of the gradient during the back-propagation (for detailed information, refer to ( Mansour et al., 2009 ; Peng et al., 2019 ) ). A major limitation of these methods lies in the disruptive changes in positional information. As elucidated earlier, domain-shareable features tend to converge, while domain-specific features separate apart. Consequently, minor positional changes can cause exponentially magnified effects. A similar issue is observed in other domain alignment methods. To solve the problem, some directly reduce the distance of the same set of users between domains ( Zhao et al., 2023a ) , and others leverage variational inference to align the mean and variance of feature distributions ( Liu et al., 2022 ) . In this paper, we propose a novel disentanglement strategy that conserves the scale of the extracted information by revising Eq. 18 as follows.

 
 
 Proposition 4.2 (Scale Alignment). 
 
 We adjust the scale of inputs before applying the domain discriminator as follows: 

 
 • 
 
 (Scale alignment between domain-specific knowledge) 

 

 
 (19) | 
 | 
 d S = F d ​ ( S | S | ) , d T = F d ​ ( T | T | ) \begin{gathered}d_{S}=F_{d}({S\over|S|}),\,\,d_{T}=F_{d}({T\over|T|})\end{gathered} | 
 | 
 

 

 • 
 
 (Scale alignment between domain-shareable knowledge) 

 

 
 (20) | 
 | 
 d ~ S = F d ​ ( g ⁡ ( S ~ | S ~ | ) ) , d ~ T = F d ​ ( g ⁡ ( T ~ | T ~ | ) ) \begin{gathered}\tilde{d}_{S}=F_{d}(g({\tilde{S}\over|\tilde{S}|})),\,\,\tilde{d}_{T}=F_{d}(g({\tilde{T}\over|\tilde{T}|}))\end{gathered} | 
 | 
 

 

 
 Based on this, we can define the domain loss ℒ d \mathcal{L}_{d} as, 

 

 
 (21) | 
 | 
 ℒ d = \displaystyle\mathcal{L}_{d}= | 
 − 1 N s ∑ n = 1 N s l o g ( 1 − d S ) − 1 N s ∑ n = 1 N s l o g ( 1 − d ~ S ) \displaystyle-{1\over N_{s}}\sum_{n=1}^{N_{s}}log(1-d_{S})-{1\over N_{s}}\sum_{n=1}^{N_{s}}log(1-\tilde{d}_{S}) | 
 | 
 
 
 | 
 | 
 − 1 N t ∑ n = 1 N t l o g ( d T ) − 1 N t ∑ n = 1 N t l o g ( d ~ T ) \displaystyle-{1\over N_{t}}\sum_{n=1}^{N_{t}}log(d_{T})-{1\over N_{t}}\sum_{n=1}^{N_{t}}log(\tilde{d}_{T}) | 
 | 
 

 
 
 
 Proof . see proof of proposition 4.2 in Section 4.6 . 
 

 
 
 Finally, we claim that scale alignment can also enhance the separability of a discriminator in the proposition below.

 
 
 Proposition 4.3 (Advantages of Scale adjustment). 
 
 Removing the scale of input features enhances the domain discriminator’s separability and guarantees stable convergence. 

 
 
 Proof . see proof of proposition 4.3 in Section 4.6 . 

 
 
 
 
 

### 4.4. Inference and Optimization

 
 Inference. We aim to measure the relativity between a user and an item using their aggregated features. To elaborate, in the middle of Figure 2 (source domain), we compute the average (avg) of the outputs from the source ( S S ) and shared FEs ( S ^ \widehat{S} ), adding the latent of user-item interaction vectors ( p u p_{u} and p i p_{i} ) as follows:

 

 
 (22) | 
 | 
 S u ′ = 1 2 ​ ( S u + S ^ u ) + p u , S i ′ = 1 2 ​ ( S i + S ^ i ) + p i S^{\prime}_{u}={1\over 2}(S_{u}+\widehat{S}_{u})+p_{u},\,\,\,S^{\prime}_{i}={1\over 2}(S_{i}+\widehat{S}_{i})+p_{i} | 
 | 
 

 Then, we project the aggregated representation onto the hyperbolic space using the exponential map in Eq. 6 as follows:

 

 
 (23) | 
 | 
 S u ℍ \displaystyle\mathrm{S^{\mathbb{H}}_{u}} | 
 = exp o ​ ( S u ′ ) = ( cosh ⁡ ( ‖ S u ′ ‖ ) , sinh ⁡ ( ‖ S u ′ ‖ ) ​ S u ′ ‖ S u ′ ‖ ) \displaystyle\mathrm{=exp_{o}(S^{\prime}_{u})=(cosh(||S^{\prime}_{u}||),sinh(||S^{\prime}_{u}||){S^{\prime}_{u}\over||S^{\prime}_{u}||})} | 
 | 
 
 
 (24) | 
 | 
 S i ℍ \displaystyle\mathrm{S^{\mathbb{H}}_{i}} | 
 = exp o ​ ( S i ′ ) = ( cosh ⁡ ( ‖ S i ′ ‖ ) , sinh ⁡ ( ‖ S i ′ ‖ ) ​ S i ′ ‖ S i ′ ‖ ) \displaystyle=\mathrm{exp_{o}(S^{\prime}_{i})=(cosh(||S^{\prime}_{i}||),sinh(||S^{\prime}_{i}||){S^{\prime}_{i}\over||S^{\prime}_{i}||})} | 
 | 
 

 Finally, we can measure their distance as below:

 

 
 (25) | 
 | 
 p ⁡ ( S u ℍ , S i ℍ ) = ℳ ⁡ ( S u ℍ ⊕ S i ℍ ) ​ d ℒ ​ ( S u ℍ , S i ℍ ) , p(S^{\mathbb{H}}_{u},S^{\mathbb{H}}_{i})=\mathcal{M}(S^{\mathbb{H}}_{u}\oplus S^{\mathbb{H}}_{i})\,d_{\mathcal{L}}(S^{\mathbb{H}}_{u},S^{\mathbb{H}}_{i}), | 
 | 
 

 where ℳ ⁡ ( ⋅ ) ∈ [ 0 , 1 ] \mathcal{M}(\cdot)\in[0,1] is a MLP with Sigmoid activation function. This adjusts the distance between users and items, d ℒ ​ ( ⋅ ) d_{\mathcal{L}}(\cdot) (Eq. 4 ) to reflect the user’s preference for popular items. For optimization, we adopt the hyperbolic margin ( ϵ = 0.1 \epsilon=0.1 ) ranking loss ( Sun et al., 2021 ) given the positive ( i i ) and negative sample ( j j ) as,

 

 
 (26) | 
 | 
 ℒ p ​ r ​ e ​ d = max ⁡ ( p ​ ( S u ℍ , S i ℍ ) 2 − p ​ ( S u ℍ , S j ℍ ) 2 + ϵ , 0 ) \mathcal{L}_{pred}=\max(p(S^{\mathbb{H}}_{u},S^{\mathbb{H}}_{i})^{2}-p(S^{\mathbb{H}}_{u},S^{\mathbb{H}}_{j})^{2}+\epsilon,0) | 
 | 
 

 
 
 Optimization. We define the overall objective function as to minimize the weighted sum of Eq. 16 , 21 , 26 as below:

 

 
 (27) | 
 | 
 min θ ⁡ ℒ t ​ o ​ t ​ a ​ l = λ 1 ​ ℒ e ​ m ​ b + λ 2 ​ ℒ d + ℒ p ​ r ​ e ​ d + δ ​ ‖ θ ‖ \min_{\theta}\mathcal{L}_{total}=\lambda_{1}\mathcal{L}_{emb}+\lambda_{2}\mathcal{L}_{d}+\mathcal{L}_{pred}+\delta||\theta|| | 
 | 
 

 The hyperparameters λ 1 \lambda_{1} and λ 2 \lambda_{2} balance the losses. For each dataset, we find λ 1 \lambda_{1} and λ 2 \lambda_{2} through grid search that yielded the best validation score (Fig. 5 ). The parameter θ \theta is optimized using the Adam optimizer, with δ \delta representing the regularization term. Additionally, we practiced early stopping within 300 iterations and applied negative sampling for ratings that meet the condition of y u , j ≤ 3 y_{u,j}\leq 3 .

 
 
 

### 4.5. Time Complexity

 
 In addition to the plain text convolution module ( A A ), we employ a hyperbolic Glove that requires a mapping from the hyperbolic space to the Euclidean ones ( B B ). Secondly, the discriminator is a simple two-layer neural network and the degree normalization only averages the outputs ( C C ). Lastly, the scale alignment has linear complexity as it only matches the magnitudes of the two vectors ( D D ). Thus, the complexity is 𝒪 ⁡ ( ( A + B + C + D ) ⋅ N t ) ≈ 𝒪 ⁡ ( N t ) \mathcal{O}((A+B+C+D)\cdot N_{t})\approx\mathcal{O}(N_{t}) , which is a linear model proportional to the size of a target domain.

 
 
 

### 4.6. Theoretical Analysis

 
 Proof of proposition 4.1 (Nodes with smaller degrees are likely to be pushed away from the origin). 
 
 Let us take S u , d ​ e ​ g n ​ o ​ r ​ m S^{norm}_{u,deg} (Eq. 14 ) as an example. Since we minimize the ℒ e ​ m ​ b \mathcal{L}_{emb} in Eq. 16 , the parameter F s F_{s} in Eq. 11 is trained to maximize S u n ​ o ​ r ​ m S^{norm}_{u} as follows: 

 

 
 (28) | 
 | 
 arg ⁡ max F s ​ S u , d ​ e ​ g n ​ o ​ r ​ m = max ⁡ ( d ) − d u max ⁡ ( d ) ​ ∂ ℒ e ​ m ​ b ∂ F s \underset{F_{s}}{\arg\max}\,S^{norm}_{u,deg}={\max(d)-d_{u}\over\max(d)}{\partial\mathcal{L}_{emb}\over\partial F_{s}} | 
 | 
 

 Both our proposed method (Eq. 14 ) and the plain method (Eq. 13 ) share the second term in Eq. 28 . Thus, we focus on the first term (degree normalization) that determines the scale of gradient as below: 

 

 
 (29) | 
 | 
 | | ▽ F s S n ​ o ​ r ​ m u , d ​ e ​ g | | / | | ▽ F s S n ​ o ​ r ​ m u | | ≈ max ⁡ ( d ) − d u max ⁡ ( d ) , ||\bigtriangledown_{F_{s}}S^{norm}_{u,deg}\,||\,\,/\,\,||\bigtriangledown_{F_{s}}S^{norm}_{u}\,||\approx{\max(d)-d_{u}\over\max(d)}, | 
 | 
 

 where ▽ \bigtriangledown denotes the partial derivative. Since ( max ⁡ ( d ) − d u ) / max ⁡ ( d ) ∈ [ 0 , 1 ] (\max(d)-d_{u})/\max(d)\in[0,1] , the scale of gradient increases as the degree of nodes ( d u d_{u} ) decreases, pushing it away from the origin and vice versa. 

 
 
 
 Proof of proposition 4.2 (Scale Preservation). 
 
 Let us take two domain-specific features d S d_{S} and d T d_{T} . According to the law of cosines, the following equality holds: 

 

 
 (30) | 
 | 
 ‖ d S − d T ‖ 2 = ‖ d S ‖ 2 + ‖ d T ‖ 2 − 2 ​ ‖ d S ‖ ⋅ ‖ d T ‖ ​ cos ⁡ C , ||d_{S}-d_{T}||^{2}=||d_{S}||^{2}+||d_{T}||^{2}-2||d_{S}||\cdot||d_{T}||\cos C, | 
 | 
 

 where C C is the angle between the vectors. Since they are from the domain-specific FEs, the updated features ( d S ′ , d T ′ d^{\prime}_{S},d^{\prime}_{T} ) satisfy ‖ d S ′ − d T ′ ‖ 2 ‖ d S − d T ‖ 2 ||d^{\prime}_{S}-d^{\prime}_{T}||^{2} ||d_{S}-d_{T}||^{2} . Thus, we can redefine the Eq. 30 as below: 

 

 
 (31) | 
 | 
 ‖ d S ′ ‖ 2 + ‖ d T ′ ‖ 2 − 2 | | d S ′ | | ⋅ | | d T ′ | | cos ⁡ C ′ ‖ d S ‖ 2 + ‖ d T ‖ 2 − 2 | | d S | | ⋅ | | d T | | cos ⁡ C ||d^{\prime}_{S}||^{2}+||d^{\prime}_{T}||^{2}-2||d^{\prime}_{S}||\cdot||d^{\prime}_{T}||\cos C^{\prime} ||d_{S}||^{2}+||d_{T}||^{2}-2||d_{S}||\cdot||d_{T}||\cos C | 
 | 
 

 Since cos ⁡ C ′ cos ⁡ C \cos C^{\prime} \cos C , assuming the update function as d ′ S = d S − ▽ d S ℒ d d^{\prime}_{S}=d_{S}-\bigtriangledown_{d_{S}}\mathcal{L}_{d} , we can infer that the scale increases in proportion to ‖ d S ‖ ||d_{S}|| . However, our method in Eq. 20 can preserve the scale because d ′ S = d S / | | d S | | − ▽ d S / ‖ d S ‖ ℒ d d^{\prime}_{S}=d_{S}/||d_{S}||-\bigtriangledown_{d_{S}/||d_{S}||}\mathcal{L}_{d} , which is proportional to d S / ‖ d S ‖ ≈ 1 d_{S}/||d_{S}||\approx 1 . 

 
 
 
 Proof of proposition 4.3 (Scale adjustment enhances stability and domain separability). 
 
 The classification error is associated with the distance from the decision boundary ( Yan et al., 2022 ) or the distance between two feature vectors ( Furusho and Ikeda, 2019 ) . Let the weight matrix of the domain discriminator be W W , the activation function be ϕ \phi . Given two inputs x x and y y , the separability 𝒮 \mathcal{S} can be defined by an inner product, reflecting both the scale and the angle, as follows: 

 

 
 (32) | 
 | 
 𝒮 = ϕ ​ ( W ​ x ) T ​ ϕ ​ ( W ​ y ) = ‖ x ‖ ⋅ ‖ y ‖ ⋅ f ⁡ ( x , y ) \mathcal{S}=\phi(Wx)^{T}\phi(Wy)=||x||\cdot||y||\cdot f(x,y) | 
 | 
 

 The f ⁡ ( x , y ) f(x,y) is the angle (e.g., cosine similarity) between two vectors. Thus, the partial derivative of the separability is given by: 

 

 
 (33) | 
 | 
 ▽ W 𝒮 = | | x | | ⋅ | | y | | ⋅ ∂ 𝒮 ∂ W f ( x , y ) \bigtriangledown_{W}\mathcal{S}=||x||\cdot||y||\cdot{\partial\mathcal{S}\over\partial W}f(x,y) | 
 | 
 

 Here, we focus on the scale of the gradient. Since the angle lies in − 1 ≤ f ⁡ ( x , y ) ≤ 1 -1\leq f(x,y)\leq 1 , the gradient ▽ W 𝒮 \bigtriangledown_{W}\mathcal{S} depends on the scale of two inputs, and removing this information is considered as one type of feature scaling method ( Wan, 2019 ; Chen et al., 2022 ) . Thus, we can guarantee stable convergence without being affected by input’s covariations, c ​ o ​ v ​ ( x , y ) ≈ ‖ x ‖ ⋅ ‖ y ‖ cov(x,y)\approx||x||\cdot||y|| as the following inequality holds, 0 ≤ ▽ W 𝒮 / ( | | x | | ⋅ | | y | | ) ≤ 1 0\leq\bigtriangledown_{W}\mathcal{S}\,/\,(||x||\cdot||y||)\leq 1 . 

 
 
 
 
 

## 5. Experiments

 
 We set fundamental questions to provide a comprehensive analysis of the proposed method. The details of the following research questions (RQs) are explained from Section 5.2 to Section 5.5 :

 
 • 
 
 RQ1: Does our model achieve a significant performance improvement compared to state-of-the-art baselines?

 

 • 
 
 RQ2: In addition to the experimental results, does scale alignment enhance domain disentanglement?

 

 • 
 
 RQ3: Does HEAD preserve the hierarchical structure better than previous methods?

 

 • 
 
 RQ4: How sensitive is the performance of proposed method on hyperparameters λ 1 \lambda_{1} and λ 2 \lambda_{2} in Eq. 27 ?

 

 
 
 
 Table 1 . Details of the benchmark datasets 
 
 
 
 
 Domain 
 Dataset 
 # users 
 # items 
 # reviews 
 
 Source 
 Clothing (Cloth) 
 1,219,520 
 376,858 
 11,285,464 
 
 CDs and Vinyl (CDs) 
 112,391 
 73,713 
 1,443,755 
 
 Toys and Games (Toys) 
 208,143 
 78,772 
 1,828,971 
 
 Target 
 Luxury Beauty 
 3,818 
 1,581 
 34,278 
 
 All Beauty 
 990 
 85 
 5,269 
 
 Digital Music 
 16,561 
 11,797 
 169,781 
 
 Video Games 
 55,217 
 17,408 
 497,577 
 

 
 
 
 Table 2 . (RQ1) The performance on four target domain datasets with a significance level * ( ρ \rho -value 0.05 0.05 ). The blue   and red   indicate best NDCG@10 (ND) and HR@10 (HR) scores. A symbol ( ℍ \mathbb{H} ) indicates that the method uses the hyperbolic space. The HEAD ∗ E \mathrm{{}^{*}_{E}} and HEAD ∗ P \mathrm{{}^{*}_{P}} employ the Euclidean- ( Pennington et al., 2014 ) and Poincaré- ( Tifrea et al., 2018 ) Glove, respectively 
 
 
 | 
 Method | 
 @10 | 
 Luxury Beauty | 
 All Beauty | 
 Digital Music | 
 Video Games | 

 
 Cloth | 
 CDs | 
 Toys | 
 Cloth | 
 CDs | 
 Toys | 
 Cloth | 
 CDs | 
 Toys | 
 Cloth | 
 CDs | 
 Toys | 

 
 
 
 Single-Domain 
 | 
 DeepCoNN | 
 ND | 
 0.093 | 
 0.089 | 
 0.101 | 
 0.124 | 

 
 HR | 
 0.177 | 
 0.165 | 
 0.184 | 
 0.220 | 

 
 1pt.          | 
 AHN | 
 ND | 
 0.129 | 
 0.142 | 
 0.106 | 
 0.171 | 

 
 | 
 | 
 HR | 
 0.231 | 
 0.254 | 
 0.199 | 
 0.306 | 

 
 1pt.          | 
 HGCF H | 
 ND | 
 0.123 | 
 0.135 | 
 0.121 | 
 0.168 | 

 
 | 
 | 
 HR | 
 0.242 | 
 0.250 | 
 0.217 | 
 0.289 | 

 
 1pt.          | 
 GDCF H | 
 ND | 
 0.121 | 
 0.140 | 
 0.118 | 
 0.153 | 

 
 | 
 | 
 HR | 
 0.229 | 
 0.251 | 
 0.220 | 
 0.277 | 

 
 1pt.          | 
 HDNR H | 
 ND | 
 0.144 | 
 0.148 | 
 0.133 | 
 0.189 | 

 
 | 
 | 
 HR | 
 0.262 | 
 0.270 | 
 0.251 | 
 0.344 | 

 
 
 
 Cross-Domain 
 | 
 DDTCDR | 
 ND | 
 0.072 | 
 0.054 | 
 0.059 | 
 0.054 | 
 0.045 | 
 0.041 | 
 0.065 | 
 0.079 | 
 0.069 | 
 0.062 | 
 0.071 | 
 0.083 | 

 
 HR | 
 0.138 | 
 0.100 | 
 0.112 | 
 0.103 | 
 0.086 | 
 0.075 | 
 0.121 | 
 0.141 | 
 0.130 | 
 0.118 | 
 0.134 | 
 0.151 | 

 
 1pt.          | 
 RC-DFM | 
 ND | 
 0.137 | 
 0.114 | 
 0.122 | 
 0.135 | 
 0.132 | 
 0.128 | 
 0.103 | 
 0.118 | 
 0.115 | 
 0.131 | 
 0.135 | 
 0.146 | 

 
 | 
 | 
 HR | 
 0.256 | 
 0.211 | 
 0.233 | 
 0.260 | 
 0.254 | 
 0.249 | 
 0.201 | 
 0.231 | 
 0.222 | 
 0.248 | 
 0.261 | 
 0.266 | 

 
 1pt.          | 
 CATN | 
 ND | 
 0.141 | 
 0.117 | 
 0.125 | 
 0.140 | 
 0.133 | 
 0.131 | 
 0.102 | 
 0.118 | 
 0.123 | 
 0.144 | 
 0.137 | 
 0.172 | 

 
 | 
 | 
 HR | 
 0.271 | 
 0.218 | 
 0.237 | 
 0.258 | 
 0.259 | 
 0.251 | 
 0.198 | 
 0.224 | 
 0.221 | 
 0.240 | 
 0.263 | 
 0.302 | 

 
 1pt.          | 
 MMT | 
 ND | 
 0.146 | 
 0.125 | 
 0.139 | 
 0.142 | 
 0.136 | 
 0.136 | 
 0.117 | 
 0.130 | 
 0.122 | 
 0.161 | 
 0.156 | 
 0.188 | 

 
 | 
 | 
 HR | 
 0.270 | 
 0.241 | 
 0.264 | 
 0.268 | 
 0.255 | 
 0.253 | 
 0.216 | 
 0.244 | 
 0.229 | 
 0.298 | 
 0.300 | 
 0.351 | 

 
 1pt.          | 
 SER | 
 ND | 
 0.149 | 
 0.136 | 
 0.147 | 
 0.150 | 
 0.143 | 
 0.146 | 
 0.149 | 
 0.152 | 
 0.148 | 
 0.199 | 
 0.205 | 
 0.221 | 

 
 | 
 | 
 HR | 
 0.283 | 
 0.270 | 
 0.286 | 
 0.288 | 
 0.272 | 
 0.279 | 
 0.261 | 
 0.300 | 
 0.297 | 
 0.353 | 
 0.352 | 
 0.394 | 

 
 1pt.          | 
 DH-GAT H | 
 ND | 
 0.152 | 
 0.142 | 
 0.145 | 
 0.153 | 
 0.150 | 
 0.152 | 
 0.146 | 
 0.144 | 
 0.127 | 
 0.200 | 
 0.202 | 
 0.214 | 

 
 | 
 | 
 HR | 
 0.285 | 
 0.266 | 
 0.271 | 
 0.290 | 
 0.289 | 
 0.294 | 
 0.278 | 
 0.278 | 
 0.246 | 
 0.366 | 
 0.371 | 
 0.389 | 

 
 
 
 Ours 
 | 
 HEAD E ℍ \mathrm{{}^{\mathbb{H}}_{E}} | 
 ND | 
 0.162 ∗ | 
 0.160 ∗ | 
 0.163 ∗ | 
 0.154 ∗ | 
 0.150 | 
 0.153 ∗ | 
 0.159 ∗ | 
 0.167 ∗ | 
 0.164 ∗ | 
 0.232 ∗ | 
 0.226 ∗ | 
 0.235 ∗ | 

 
 HR | 
 0.303 ∗ | 
 0.299 ∗ | 
 0.308 ∗ | 
 0.300 ∗ | 
 0.296 ∗ | 
 0.297 ∗ | 
 0.289 | 
 0.310 ∗ | 
 0.311 ∗ | 
 0.397 ∗ | 
 0.380 ∗ | 
 0.414 ∗ | 

 
 1pt.          | 
 HEAD P ℍ \mathrm{{}^{\mathbb{H}}_{P}} | 
 ND | 
 0.173 ∗ | 
 0.166 ∗ | 
 0.169 ∗ | 
 0.161 ∗ | 
 0.158 ∗ | 
 0.157 ∗ | 
 0.161 ∗ | 
 0.180 ∗ | 
 0.175 ∗ | 
 0.238 ∗ | 
 0.232 ∗ | 
 0.244 ∗ | 

 
 | 
 | 
 HR | 
 0.321 ∗ | 
 0.314 ∗ | 
 0.320 ∗ | 
 0.309 ∗ | 
 0.302 ∗ | 
 0.305 ∗ | 
 0.301 ∗ | 
 0.333 ∗ | 
 0.327 ∗ | 
 0.408 ∗ | 
 0.396 ∗ | 
 0.417 ∗ | 

 
 

### 5.1. Experimental Setup

 
 Following the prior studies ( Choi et al., 2022 ; Zhao et al., 2023a ) , we evaluate our model using Amazon 7 7 
 7 
 
 
 
 https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ 5-core review datasets. As shown in Table 1 , we take 12 domain pairs, three as the source with richer interactions and four as the target with sparse ones ( Ben-David et al., 2010 ) . The pairs of (Cloth, Beauty) , (CDs, Digital Music) , (Toys, Video Games) are relevant to each other. The target domain datasets are split into 80%/10%/10% for training, validation, and testing without considering a temporal sequence same as ( Choi et al., 2022 ; Xu and Cai, 2023 ; Wang et al., 2023 ) . Additionally, we employ an early stopping technique to terminate the training process if the best validation score is not updated for 300 iterations. The dimension of word embedding is set as 100 for all methods.

 
 
 

### 5.2. Model Comparison (RQ1)

 
 In Table 2 , we show the results using Normalized Discounted Cumulative Gain (NDCG @ ​ 10 @10 ) and Hit Ratio (HR @ ​ 10 @10 ).

 
 
 (1) Reviews and domain disentanglement enhance the quality of recommendation. Firstly, we observe that methods utilizing only rating information ( He et al., 2017 ; Yuan et al., 2019 ; Li and Tuzhilin, 2020 ) show significantly lower performance compared to the review-based models. This can be due to the small size of the target domain, but we can also presume that user preferences are well reflected in the review information. We also observe that the algorithms perform differently depending on how well the reviews are utilized; attention-based AHN ( Dong et al., 2020 ) significantly outperforms the plain text convolution model, DeepCoNN ( Zheng et al., 2017 ) . Additionally, addressing domain discrepancies is also critical to the performance of CDR. For example, MMT ( Krishnan et al., 2020 ) , SER ( Choi et al., 2022 ) , and our HEAD show stable performance among the CDR techniques regardless of the source domain pairs. This can be inferred that they can separate the domain-shareable and domain-specific knowledge efficiently, making them relatively robust to noise. Additionally, our models exhibit the most stable results, suggesting that scale preservation is helpful in discriminator training.

 
 
 (2) Hyperbolic embedding achieves better performance compared to Euclidean ones, and HEAD with degree-based normalization and scale adjustment has shown its effectiveness. 
A notable point is that hyperbolic-based methods HGCF ( Sun et al., 2021 ) , GDCF ( Zhang et al., 2022 ) , and HDNR ( Wang et al., 2023 ) exhibit good performance even without using additional domains. For example, their accuracy is comparable to AHN ( Dong et al., 2020 ) with hierarchical attention mechanism, and DH-GAT H ( Xu and Cai, 2023 ) attains the best recommendation quality for some datasets. This is based on the advantages of the vast space in the hyperbolic space, which has a positive impact on learning the pairwise distance between the latent representations. In addition to this, our HEAD, which employs degree-based hierarchy correction and scale alignment, achieves a performance improvement of 10.4% compared to SER ( Choi et al., 2022 ) . This highlights the benefits of hierarchy alignment in hyperbolic space and the removal of scale information enhances the quality of the domain discriminator.

 
 
 
 
 
 (a) Similar domain 
 
 
 (b) Dissimilar domain 
 
 Figure 3 . (RQ2) Domain discrimination performance of three methods with similar and dissimilar domain pairs 
 
 
 Figure 4 . (RQ3) We randomly sampled 1,000 items in Digital Music and visualized them based on their degrees 
 
 
 

### 5.3. Scale Alignment and Disentanglement (RQ2)

 
 In Figure 3 , we describe the domain classification accuracy of the domain discriminator based on the application of scale alignment. Here, we employ three models: SER ( Choi et al., 2022 ) which is a state-of-the-art disentanglement algorithm, HEAD (without S cale A lignment), and HEAD (with S cale A lignment). Since the domain label is binary (0 for source and 1 for target), we describe the binary cross-entropy on the y-axis. The x-axis is the training epochs. Both figures use the Luxury Beauty as the target domain, but each of them employs Clothing (left) and CDs (right) as the source domains, respectively. We discover that the discrimination accuracy is quite low in the left figure, where the two domains are similar. In addition to this, both figures represent that the discrimination accuracy of SER is better than HEAD (w/o SA), where HEAD has a larger scale than SER. This is because the hierarchical alignment in Eq. 14 has a separation characteristic. However, HEAD (w/ SA) achieves the best discrimination accuracy, which confirms the proposition 4.3 .

 
 
 

### 5.4. Hierarchy Visualization (RQ3)

 
 In Figure 4 , we visualize item vectors in Digital Music dataset to assess the effect of hierarchy-aware embedding (Proposition 4.1 ). Here, we randomly sample 1,000 items and classify them based on their degrees. Specifically, we average the two item vectors S i S_{i} and S ^ i \widehat{S}_{i} in Eq. 11 and project them onto the Poincaré ball (Eq. 1 ). The left figure employs simple root alignment (Eq. 13 ), while the right one further benefits from our degree-based normalization (Eq. 14 ). As observed in the left figure, nodes are quite randomly distributed regardless of their degrees. Although some nodes with higher degrees (red, d 20 d 20 ) are placed near the origin, nodes with lower degrees (purple, green, and blue) are positioned quite randomly. In contrast, the right figure shows that nodes are aligned based on degrees. From this, we conclude that the degree-based normalization successfully preserves the structural information, which leads to a better utilization of hyperbolic space eventually. For case studies, we highly recommend reading this article ( Cao et al., 2022a ) .

 
 
 Figure 5 . (RQ4) We describe the NDCG@10 of two datasets by varying the parameters λ 1 \lambda_{1} (x-axis, degree normalization) and λ 2 \lambda_{2} (y-axis, scale alignment) in Eq. 27 , respectively 
 
 
 

### 5.5. Parameter Sensitivity Analysis (RQ4)

 
 Given the model with Poincaré Glove HEAD ℙ ℍ {}^{\mathbb{H}}_{\mathbb{P}} , we vary the weights of the loss function in Eq. 27 . Typically, the weight of the prediction loss (Eq. 26 ) is set to 1 since it is the main object of recommender systems. Now, we adjust the two hyper-parameters λ 1 \lambda_{1} and λ 2 \lambda_{2} that control the weight of hierarchy embedding and scale alignment. The experimental results are shown in Figure 5 , where we conduct a grid search and plot the NDCG@10 score through heatmap using the pairs of ( CDs and Vinyl , Digital Music ) and ( Toys and Games , Video Games ). Here, the rows and columns represent λ 1 \lambda_{1} and λ 2 \lambda_{2} , respectively. Firstly, we observe that the performances are dismal when the hyper-parameters take large values. This is because the ranking loss is overwhelmed by other functions, making the convergence of the parameters challenging. Instead, assigning small values for both λ 1 \lambda_{1} and λ 2 \lambda_{2} enhances the overall quality of recommendation by improving structural alignment and domain disentanglement. As illustrated, we can see that setting λ 1 = λ 2 = 0.05 \lambda_{1}=\lambda_{2}=0.05 in (a) Digital Music and λ 1 = 0.1 , λ 2 = 0.05 \lambda_{1}=0.1,\lambda_{2}=0.05 in (b) Video Games achieve the best performance. One might argue that the search for optimal parameters may require huge computational costs, but we find that suppressing these values below a specific threshold grants marginal improvements only. Please refer to ( Choi et al., 2022 ) for the ablation study of using either domain-specific or shareable features.

 
 
 
 

## 6. CONCLUSION

 
 Recent studies have addressed the challenge of data sparsity in recommender systems by integrating Cross-Domain Recommendation (CDR) with review texts. However, existing methods relying on an Euclidean space encounter difficulties due to the exponentially growing interactions between users and items. In response to this, we introduce a hyperbolic CDR as a potential solution and overcome several associated issues. Firstly, we identify some drawbacks related to root- and distance-based alignment, which are problematic in preserving the tree-like structure within a hyperbolic space. To address these issues, we propose a novel solution: hierarchy-preserving embedding and domain disentanglement. Lastly, we provide a mathematical foundation to emphasize the theoretical relevance of our proposed strategies. Experimental results demonstrate the superiority of our model over state-of-the-art single and cross-domain algorithms.

 
 
 

## 7. Acknowledgments

 
 This work was supported by Institute of Information Communications Technology Planning Evaluation (IITP) grant funded by the Korean government (MSIT) (No. 2021-0-02068, IITP-2024-00156287), and KENTECH Research Grant (202200019A) funded by the National Research Foundation of Korea (NRF) (4199990214639).

 
 
 

## References

 
 
 Achiam et al . (2023) 
 
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al . 2023.

 
 Gpt-4 technical report.

 
 arXiv preprint arXiv:2303.08774 (2023).

 
 
 

 
 Addagarla and Amalanathan (2020) 
 
Ssvr Kumar Addagarla and Anthoniraj Amalanathan. 2020.

 
 Probabilistic unsupervised machine learning approach for a similar image recommender system for E-commerce.

 
 Symmetry 12, 11 (2020), 1783.

 
 
 

 
 Al-Ghuribi and Noah (2019) 
 
Sumaia Mohammed Al-Ghuribi and Shahrul Azman Mohd Noah. 2019.

 
 Multi-criteria review-based recommender system–the state of the art.

 
 IEEE Access 7 (2019), 169446–169468.

 
 
 

 
 Ben-David et al . (2010) 
 
Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. 2010.

 
 A theory of learning from different domains.

 
 Machine learning 79, 1 (2010), 151–175.

 
 
 

 
 Bonab et al . (2021) 
 
Hamed Bonab, Mohammad Aliannejadi, Ali Vardasbi, Evangelos Kanoulas, and James Allan. 2021.

 
 Cross-Market Product Recommendation. In Proceedings of the 30th ACM International Conference on Information Knowledge Management . 110–119.

 
 
 

 
 Bousmalis et al . (2016) 
 
Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, and Dumitru Erhan. 2016.

 
 Domain separation networks.

 
 Advances in neural information processing systems 29 (2016), 343–351.

 
 
 

 
 Cai et al . (2019) 
 
Ruichu Cai, Zijian Li, Pengfei Wei, Jie Qiao, Kun Zhang, and Zhifeng Hao. 2019.

 
 Learning disentangled semantic representation for domain adaptation. In IJCAI: proceedings of the conference , Vol. 2019. NIH Public Access, 2060.

 
 
 

 
 Cao et al . (2022a) 
 
Jiangxia Cao, Xixun Lin, Xin Cong, Jing Ya, Tingwen Liu, and Bin Wang. 2022a.

 
 Disencdr: Learning disentangled representations for cross-domain recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval . 267–277.

 
 
 

 
 Cao et al . (2022b) 
 
Jiangxia Cao, Jiawei Sheng, Xin Cong, Tingwen Liu, and Bin Wang. 2022b.

 
 Cross-domain recommendation to cold-start users via variational information bottleneck. In 2022 IEEE 38th International Conference on Data Engineering (ICDE) . IEEE, 2209–2223.

 
 
 

 
 Chen et al . (2018) 
 
Chong Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018.

 
 Neural attentional rating regression with review-level explanations. In Proceedings of the 2018 World Wide Web Conference . 1583–1592.

 
 
 

 
 Chen et al . (2023) 
 
Liyue Chen, Linian Wang, Jinyu Xu, Shuai Chen, Weiqiang Wang, Wenbiao Zhao, Qiyu Li, and Leye Wang. 2023.

 
 Knowledge-inspired Subdomain Adaptation for Cross-Domain Knowledge Transfer. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management . 234–244.

 
 
 

 
 Chen et al . (2019b) 
 
Xu Chen, Yongfeng Zhang, and Zheng Qin. 2019b.

 
 Dynamic Explainable Recommendation based on Neural Attentive Models. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 33. 53–60.

 
 
 

 
 Chen et al . (2022) 
 
Zhengdao Chen, Eric Vanden-Eijnden, and Joan Bruna. 2022.

 
 On feature learning in neural networks with global convergence guarantees.

 
 arXiv preprint arXiv:2204.10782 (2022).

 
 
 

 
 Chen et al . (2019a) 
 
Zhongxia Chen, Xiting Wang, Xing Xie, Tong Wu, Guoqing Bu, Yining Wang, and Enhong Chen. 2019a.

 
 Co-attentive multi-task learning for explainable recommendation. In Proceedings of the 28th International Joint Conference on Artificial Intelligence . AAAI Press, 2137–2143.

 
 
 

 
 Choi et al . (2022) 
 
Yoonhyuk Choi, Jiho Choi, Taewook Ko, Hyungho Byun, and Chong-Kwon Kim. 2022.

 
 Based Domain Disentanglement without Duplicate Users or Contexts for Cross-Domain Recommendation. In Proceedings of the 31st ACM International Conference on Information Knowledge Management . 293–303.

 
 
 

 
 Clauset et al . (2009) 
 
Aaron Clauset, Cosma Rohilla Shalizi, and Mark EJ Newman. 2009.

 
 Power-law distributions in empirical data.

 
 SIAM review 51, 4 (2009), 661–703.

 
 
 

 
 Dong et al . (2020) 
 
Xin Dong, Jingchao Ni, Wei Cheng, Zhengzhang Chen, Bo Zong, Dongjin Song, Yanchi Liu, Haifeng Chen, and Gerard De Melo. 2020.

 
 Asymmetrical hierarchical networks with attentive interactions for interpretable review-based recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 34. 7667–7674.

 
 
 

 
 Elkahky et al . (2015) 
 
Ali Mamdouh Elkahky, Yang Song, and Xiaodong He. 2015.

 
 A multi-view deep learning approach for cross domain user modeling in recommendation systems. In Proceedings of the 24th International Conference on World Wide Web . 278–288.

 
 
 

 
 Feng et al . (2020) 
 
Shanshan Feng, Lucas Vinh Tran, Gao Cong, Lisi Chen, Jing Li, and Fan Li. 2020.

 
 Hme: A hyperbolic metric embedding approach for next-poi recommendation. In Proceedings of the 43rd International ACM SIGIR Conference on research and development in information retrieval . 1429–1438.

 
 
 

 
 Fu et al . (2019) 
 
Wenjing Fu, Zhaohui Peng, Senzhang Wang, Yang Xu, and Jin Li. 2019.

 
 Deeply fusing reviews and contents for cold start users in cross-domain recommendation systems. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 33. 94–101.

 
 
 

 
 Furusho and Ikeda (2019) 
 
Yasutaka Furusho and Kazushi Ikeda. 2019.

 
 Resnet and batch-normalization improve data separability. In Asian Conference on Machine Learning . PMLR, 94–108.

 
 
 

 
 Globerson et al . (2004) 
 
Amir Globerson, Gal Chechik, Fernando Pereira, and Naftali Tishby. 2004.

 
 Euclidean embedding of co-occurrence data.

 
 Advances in neural information processing systems 17 (2004).

 
 
 

 
 Gretton et al . (2005) 
 
Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. 2005.

 
 Measuring statistical dependence with Hilbert-Schmidt norms. In International conference on algorithmic learning theory . Springer, 63–77.

 
 
 

 
 Gulcehre et al . (2018) 
 
Caglar Gulcehre, Misha Denil, Mateusz Malinowski, Ali Razavi, Razvan Pascanu, Karl Moritz Hermann, Peter Battaglia, Victor Bapst, David Raposo, Adam Santoro, et al . 2018.

 
 Hyperbolic attention networks.

 
 arXiv preprint arXiv:1805.09786 (2018).

 
 
 

 
 Guo et al . (2021) 
 
Lei Guo, Li Tang, Tong Chen, Lei Zhu, Quoc Viet Hung Nguyen, and Hongzhi Yin. 2021.

 
 DA-GCN: A Domain-aware Attentive Graph Convolution Network for Shared-account Cross-domain Sequential Recommendation.

 
 arXiv preprint arXiv:2105.03300 (2021).

 
 
 

 
 Guo et al . (2023) 
 
Naicheng Guo, Xiaolei Liu, Shaoshuai Li, Mingming Ha, Qiongxu Ma, Binfeng Wang, Yunan Zhao, Linxun Chen, and Xiaobo Guo. 2023.

 
 Hyperbolic Contrastive Graph Representation Learning for Session-based Recommendation.

 
 IEEE Transactions on Knowledge and Data Engineering (2023).

 
 
 

 
 Hande et al . (2021) 
 
Adeep Hande, Karthik Puranik, Ruba Priyadharshini, and Bharathi Raja Chakravarthi. 2021.

 
 Domain identification of scientific articles using transfer learning and ensembles. In Pacific-Asia Conference on Knowledge Discovery and Data Mining . Springer, 88–97.

 
 
 

 
 He et al . (2017) 
 
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017.

 
 Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web . 173–182.

 
 
 

 
 Hu et al . (2018) 
 
Guangneng Hu, Yu Zhang, and Qiang Yang. 2018.

 
 Conet: Collaborative cross networks for cross-domain recommendation. In Proceedings of the 27th ACM international conference on information and knowledge management . 667–676.

 
 
 

 
 Kang et al . (2019) 
 
SeongKu Kang, Junyoung Hwang, Dongha Lee, and Hwanjo Yu. 2019.

 
 Semi-supervised learning for cross-domain recommendation to cold-start users. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management . 1563–1572.

 
 
 

 
 Kazienko et al . (2011) 
 
Przemysław Kazienko, Katarzyna Musial, and Tomasz Kajdanowicz. 2011.

 
 Multidimensional social network in the social recommender system.

 
 IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans 41, 4 (2011), 746–759.

 
 
 

 
 Khoshneshin and Street (2010) 
 
Mohammad Khoshneshin and W Nick Street. 2010.

 
 Collaborative filtering via euclidean embedding. In Proceedings of the fourth ACM conference on Recommender systems . 87–94.

 
 
 

 
 Khrulkov et al . (2020) 
 
Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky. 2020.

 
 Hyperbolic image embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 6418–6428.

 
 
 

 
 Krishnan et al . (2020) 
 
Adit Krishnan, Mahashweta Das, Mangesh Bendre, Hao Yang, and Hari Sundaram. 2020.

 
 Transfer Learning via Contextual Invariants for One-to-Many Cross-Domain Recommendation. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval . 1081–1090.

 
 
 

 
 Law et al . (2019) 
 
Marc Law, Renjie Liao, Jake Snell, and Richard Zemel. 2019.

 
 Lorentzian distance learning for hyperbolic representations. In International Conference on Machine Learning . PMLR, 3672–3681.

 
 
 

 
 Li et al . (2022) 
 
Anchen Li, Bo Yang, Huan Huo, Hongxu Chen, Guandong Xu, and Zhen Wang. 2022.

 
 Hyperbolic neural collaborative recommender.

 
 IEEE Transactions on Knowledge and Data Engineering (2022).

 
 
 

 
 Li and Tuzhilin (2020) 
 
Pan Li and Alexander Tuzhilin. 2020.

 
 Ddtcdr: Deep dual transfer cross domain recommendation. In Proceedings of the 13th International Conference on Web Search and Data Mining . 331–339.

 
 
 

 
 Li et al . (2019) 
 
Zejian Li, Yongchuan Tang, Wei Li, and Yongxing He. 2019.

 
 Learning disentangled representation with pairwise independence. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 33. 4245–4252.

 
 
 

 
 Lin and Zha (2008) 
 
Tong Lin and Hongbin Zha. 2008.

 
 Riemannian manifold learning.

 
 IEEE transactions on pattern analysis and machine intelligence 30, 5 (2008), 796–809.

 
 
 

 
 Liu et al . (2021) 
 
Huiting Liu, Lingling Guo, Peipei Li, Peng Zhao, and Xindong Wu. 2021.

 
 Collaborative filtering with a deep adversarial and attention network for cross-domain recommendation.

 
 Information Sciences 565 (2021), 370–389.

 
 
 

 
 Liu et al . (2019) 
 
Tianqiao Liu, Zhiwei Wang, Jiliang Tang, Songfan Yang, Gale Yan Huang, and Zitao Liu. 2019.

 
 Recommender systems with heterogeneous side information. In The world wide web conference . 3027–3033.

 
 
 

 
 Liu et al . (2022) 
 
Weiming Liu, Xiaolin Zheng, Jiajie Su, Mengling Hu, Yanchao Tan, and Chaochao Chen. 2022.

 
 Exploiting variational domain-invariant user embedding for partially overlapped cross domain recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval . 312–321.

 
 
 

 
 Lu et al . (2015) 
 
Jie Lu, Dianshuang Wu, Mingsong Mao, Wei Wang, and Guangquan Zhang. 2015.

 
 Recommender system application developments: a survey.

 
 Decision support systems 74 (2015), 12–32.

 
 
 

 
 Man et al . (2017) 
 
Tong Man, Huawei Shen, Xiaolong Jin, and Xueqi Cheng. 2017.

 
 Cross-Domain Recommendation: An Embedding and Mapping Approach.. In IJCAI , Vol. 17. 2464–2470.

 
 
 

 
 Mansour et al . (2009) 
 
Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. 2009.

 
 Domain adaptation: Learning bounds and algorithms.

 
 arXiv preprint arXiv:0902.3430 (2009).

 
 
 

 
 Mikolov et al . (2013) 
 
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013.

 
 Distributed representations of words and phrases and their compositionality.

 
 Advances in neural information processing systems 26 (2013).

 
 
 

 
 Nema et al . (2021) 
 
Preksha Nema, Alexandros Karatzoglou, and Filip Radlinski. 2021.

 
 Disentangling Preference Representations for Recommendation Critiquing with ß-VAE. In Proceedings of the 30th ACM International Conference on Information Knowledge Management . 1356–1365.

 
 
 

 
 Nickel and Kiela (2017) 
 
Maximillian Nickel and Douwe Kiela. 2017.

 
 Poincaré embeddings for learning hierarchical representations.

 
 Advances in neural information processing systems 30 (2017).

 
 
 

 
 Nickel and Kiela (2018) 
 
Maximillian Nickel and Douwe Kiela. 2018.

 
 Learning continuous hierarchies in the lorentz model of hyperbolic geometry. In International conference on machine learning . PMLR, 3779–3788.

 
 
 

 
 Peng et al . (2019) 
 
Xingchao Peng, Zijun Huang, Ximeng Sun, and Kate Saenko. 2019.

 
 Domain agnostic learning with disentangled representations. In International Conference on Machine Learning . PMLR, 5102–5112.

 
 
 

 
 Pennington et al . (2014) 
 
Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014.

 
 Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) . 1532–1543.

 
 
 

 
 Ramakrishnan et al . (2018) 
 
Sainandan Ramakrishnan, Aishwarya Agrawal, and Stefan Lee. 2018.

 
 Overcoming language priors in visual question answering with adversarial regularization.

 
 arXiv preprint arXiv:1810.03649 (2018).

 
 
 

 
 Sachdeva and McAuley (2020) 
 
Noveen Sachdeva and Julian McAuley. 2020.

 
 How Useful are Reviews for Recommendation? A Critical Review and Potential Improvements. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval . 1845–1848.

 
 
 

 
 Seo et al . (2017) 
 
Sungyong Seo, Jing Huang, Hao Yang, and Yan Liu. 2017.

 
 Interpretable convolutional neural networks with dual local and global attention for review rating prediction. In Proceedings of the eleventh ACM conference on recommender systems . 297–305.

 
 
 

 
 Srifi et al . (2020) 
 
Mehdi Srifi, Ahmed Oussous, Ayoub Ait Lahcen, and Salma Mouline. 2020.

 
 Recommender systems based on collaborative filtering using review texts—a survey.

 
 Information 11, 6 (2020), 317.

 
 
 

 
 Su et al . (2023) 
 
Jiajie Su, Chaochao Chen, Weiming Liu, Fei Wu, Xiaolin Zheng, and Haoming Lyu. 2023.

 
 Enhancing Hierarchy-Aware Graph Networks with Deep Dual Clustering for Session-based Recommendation. In Proceedings of the ACM Web Conference 2023 . 165–176.

 
 
 

 
 Sun et al . (2021) 
 
Jianing Sun, Zhaoyue Cheng, Saba Zuberi, Felipe Pérez, and Maksims Volkovs. 2021.

 
 Hgcf: Hyperbolic graph convolution networks for collaborative filtering. In Proceedings of the Web Conference 2021 . 593–601.

 
 
 

 
 Tay et al . (2018) 
 
Yi Tay, Anh Tuan Luu, and Siu Cheung Hui. 2018.

 
 Multi-pointer co-attention networks for recommendation. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery Data Mining . 2309–2318.

 
 
 

 
 Tifrea et al . (2018) 
 
Alexandru Tifrea, Gary Bécigneul, and Octavian-Eugen Ganea. 2018.

 
 Poincar \ \backslash ’e glove: Hyperbolic word embeddings.

 
 arXiv preprint arXiv:1810.06546 (2018).

 
 
 

 
 Touvron et al . (2023) 
 
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al . 2023.

 
 Llama: Open and efficient foundation language models.

 
 arXiv preprint arXiv:2302.13971 (2023).

 
 
 

 
 Vinh Tran et al . (2020) 
 
Lucas Vinh Tran, Yi Tay, Shuai Zhang, Gao Cong, and Xiaoli Li. 2020.

 
 Hyperml: A boosting metric learning approach in hyperbolic space for recommender systems. In Proceedings of the 13th international conference on web search and data mining . 609–617.

 
 
 

 
 Wan (2019) 
 
Xing Wan. 2019.

 
 Influence of feature scaling on convergence of gradient iterative algorithm. In Journal of physics: Conference series , Vol. 1213. IOP Publishing, 032021.

 
 
 

 
 Wang et al . (2021) 
 
Hao Wang, Defu Lian, Hanghang Tong, Qi Liu, Zhenya Huang, and Enhong Chen. 2021.

 
 Hypersorec: Exploiting hyperbolic user and item representations with multiple aspects for social-aware recommendation.

 
 ACM Transactions on Information Systems (TOIS) 40, 2 (2021), 1–28.

 
 
 

 
 Wang et al . (2023) 
 
Shicheng Wang, Shu Guo, Lihong Wang, Tingwen Liu, and Hongbo Xu. 2023.

 
 HDNR: A Hyperbolic-Based Debiased Approach for Personalized News Recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval . 259–268.

 
 
 

 
 Wang et al . (2018) 
 
Xinghua Wang, Zhaohui Peng, Senzhang Wang, S Yu Philip, Wenjing Fu, and Xiaoguang Hong. 2018.

 
 Cross-domain recommendation for cold-start users via neighborhood based feature mapping. In International conference on database systems for advanced applications . Springer, 158–165.

 
 
 

 
 Xie et al . (2022) 
 
Ruobing Xie, Qi Liu, Liangdong Wang, Shukai Liu, Bo Zhang, and Leyu Lin. 2022.

 
 Contrastive cross-domain recommendation in matching. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 4226–4236.

 
 
 

 
 Xu and Cai (2023) 
 
Jingyun Xu and Yi Cai. 2023.

 
 Decoupled Hyperbolic Graph Attention Network for Cross-domain Named Entity Recognition. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval . 591–600.

 
 
 

 
 Yan et al . (2022) 
 
Yujun Yan, Milad Hashemi, Kevin Swersky, Yaoqing Yang, and Danai Koutra. 2022.

 
 Two sides of the same coin: Heterophily and oversmoothing in graph convolutional neural networks. In 2022 IEEE International Conference on Data Mining (ICDM) . IEEE, 1287–1292.

 
 
 

 
 Yang et al . (2022a) 
 
Menglin Yang, Zhihao Li, Min Zhou, Jiahong Liu, and Irwin King. 2022a.

 
 Hicf: Hyperbolic informative collaborative filtering. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 2212–2221.

 
 
 

 
 Yang et al . (2022b) 
 
Menglin Yang, Min Zhou, Jiahong Liu, Defu Lian, and Irwin King. 2022b.

 
 HRCF: Enhancing collaborative filtering via hyperbolic geometric regularization. In Proceedings of the ACM Web Conference 2022 . 2462–2471.

 
 
 

 
 Yang et al . (2021) 
 
Xiangli Yang, Qing Liu, Rong Su, Ruiming Tang, Zhirong Liu, and Xiuqiang He. 2021.

 
 AutoFT: Automatic Fine-Tune for Parameters Transfer Learning in Click-Through Rate Prediction.

 
 arXiv preprint arXiv:2106.04873 (2021).

 
 
 

 
 Yuan et al . (2020) 
 
Fajie Yuan, Xiangnan He, Alexandros Karatzoglou, and Liguang Zhang. 2020.

 
 Parameter-efficient transfer from sequential behaviors for user modeling and recommendation. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval . 1469–1478.

 
 
 

 
 Yuan et al . (2019) 
 
Feng Yuan, Lina Yao, and Boualem Benatallah. 2019.

 
 DARec: deep domain adaptation for cross-domain recommendation via transferring rating patterns.

 
 arXiv preprint arXiv:1905.10760 (2019).

 
 
 

 
 Zeng et al . (2021) 
 
Hansi Zeng, Zhichao Xu, and Qingyao Ai. 2021.

 
 A Zero Attentive Relevance Matching Networkfor Review Modeling in Recommendation System.

 
 arXiv preprint arXiv:2101.06387 (2021).

 
 
 

 
 Zhang et al . (2019) 
 
Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019.

 
 Deep learning based recommender system: A survey and new perspectives.

 
 ACM computing surveys (CSUR) 52, 1 (2019), 1–38.

 
 
 

 
 Zhang et al . (2022) 
 
Yiding Zhang, Chaozhuo Li, Xing Xie, Xiao Wang, Chuan Shi, Yuming Liu, Hao Sun, Liangjie Zhang, Weiwei Deng, and Qi Zhang. 2022.

 
 Geometric disentangled collaborative filtering. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval . 80–90.

 
 
 

 
 Zhao et al . (2020) 
 
Cheng Zhao, Chenliang Li, Rong Xiao, Hongbo Deng, and Aixin Sun. 2020.

 
 CATN: Cross-Domain Recommendation for Cold-Start Users via Aspect Transfer Network.

 
 arXiv preprint arXiv:2005.10549 (2020).

 
 
 

 
 Zhao et al . (2023a) 
 
Chuang Zhao, Hongke Zhao, Ming He, Jian Zhang, and Jianping Fan. 2023a.

 
 Cross-domain recommendation via user interest alignment. In Proceedings of the ACM Web Conference 2023 . 887–896.

 
 
 

 
 Zhao et al . (2023b) 
 
Chuang Zhao, Hongke Zhao, Xiaomeng Li, Ming He, Jiahui Wang, and Jianping Fan. 2023b.

 
 Cross-domain recommendation via progressive structural alignment.

 
 IEEE Transactions on Knowledge and Data Engineering (2023).

 
 
 

 
 Zheng et al . (2017) 
 
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017.

 
 Joint deep modeling of users and items using reviews for recommendation. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining . 425–434.

 
 
 

 
 Zhu et al . (2021) 
 
Feng Zhu, Yan Wang, Chaochao Chen, Jun Zhou, Longfei Li, and Guanfeng Liu. 2021.

 
 Cross-domain recommendation: challenges, progress, and prospects.

 
 arXiv preprint arXiv:2103.01696 (2021).

 
 
 

 
 Zhu et al . (2022) 
 
Yongchun Zhu, Zhenwei Tang, Yudan Liu, Fuzhen Zhuang, Ruobing Xie, Xu Zhang, Leyu Lin, and Qing He. 2022.

 
 Personalized transfer of user preferences for cross-domain recommendation. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining . 1507–1515.