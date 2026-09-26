A Survey of Privacy-Preserving Model Explanations: Privacy Risks, Attacks, and Countermeasures 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY-NC-ND 4.0
 
 
arXiv:2404.00673v2 [cs.CR] 26 Jun 2024 
 
 

# A Survey of Privacy-Preserving Model Explanations: 
 Privacy Risks, Attacks, and Countermeasures

 Conference:  ACM; Survey; PrivEx 
 
 
 Thanh Tam Nguyen 1 , Thanh Trung Huynh 2 ,
Zhao Ren 3 ,
Thanh Toan Nguyen 1 ,
Phi Le Nguyen 4 ,
Hongzhi Yin 5 , Quoc Viet Hung Nguyen 1 
 
 Affiliation:  
 1 Griffith University,
 2 École Polytechnique Fédérale de Lausanne,
 3 University of Bremen,
 4 Hanoi University of Science and Technology,
 5 The University of Queensland
 
 
 2024 

 Abstract. 
 
 As the adoption of explainable AI (XAI) continues to expand, the urgency to address its privacy implications intensifies. Despite a growing corpus of research in AI privacy and explainability, there is little attention on privacy-preserving model explanations. This article presents the first thorough survey about privacy attacks on model explanations and their countermeasures. Our contribution to this field comprises a thorough analysis of research papers with a connected taxonomy that facilitates the categorisation of privacy attacks and countermeasures based on the targeted explanations. This work also includes an initial investigation into the causes of privacy leaks. Finally, we discuss unresolved issues and prospective research directions uncovered in our analysis. This survey aims to be a valuable resource for the research community and offers clear insights for those new to this domain. To support ongoing research, we have established an online resource repository, which will be continuously updated with new and relevant findings. Interested readers are encouraged to access our repository at https://github.com/tamlhp/awesome-privex .

 
 
 
 Keywords:  model explanations, privacy-preserving explanation, privacy attacks, privacy leak, explainable AI, explainable machine learning, interpretable machine learning, adversarial machine learning, PrivEx, PrivML, PrivAI, XAI, PrivXAI
 
 

## 1. Introduction

 
 In recent years, the push for automated model explanations has gained significant momentum, with key guidelines like the GDPR highlighting their importance  ( Goodman and
Flaxman, 2017 ) , and tech giants such as Google, Microsoft, and IBM pioneering this initiative by integrating explanation toolkits into their machine learning solutions  ( Chang and Shokri, 2021 ) . This movement towards transparency encompasses a variety of explanation types, from global and local explanations that offer broad overviews and specific decision rationales, respectively, to feature importance analyses that pinpoint the impact of individual data inputs  ( Ancona
et al., 2018 ) . Techniques like SHAP and LIME provide nuanced insights into feature contributions  ( Ribeiro
et al., 2016 ; Lundberg and Lee, 2017 ) , while counterfactual explanations explore how changes in input could lead to different outcomes  ( Guidotti, 2022 ) . Additionally, interactive visualization tools are becoming increasingly popular, making the interpretation of complex models more accessible to users  ( Bodria et al., 2023 ; Guidotti et al., 2018 ; Gilpin et al., 2018 ) .

 
 
 However, this pursuit of transparency is not without its risks, especially privacy. The very act of providing explanations involves the disclosure of information that, while intended to illuminate, also carries the risk of inadvertently revealing sensitive details embedded in the models’ training data.
The balance between transparency and privacy becomes even more precarious when considering the granularity of explanations. Detailed explanations, although more informative, might offer direct inferences about individual data points used in training, thereby increasing the risk of privacy breaches. This paradox underscores a significant challenge within the field, as highlighted by recent research  ( Goethals
et al., 2023 ; Chang and Shokri, 2021 ; Ferry et al., 2023b ) , which delve into the privacy implications of model explanations.

 
 
 The degree to which model explanations reveal specifics about users’ data is not fully understood. The unintended disclosure of sensitive details, such as a person’s location, health records, or identity, through these explanations could pose serious concerns if such information were to be deciphered by a malicious entity  ( Sokol and Flach, 2019 ) . On the flip side, if private data is used without the rightful owner’s permission, the same techniques aimed at exposing information could also detect unauthorized data utilization, thus potentially safeguarding user privacy  ( Luo
et al., 2022 ) . Furthermore, there is a growing interest not just in the attacks themselves but in understanding the underlying causes of privacy violations and what makes a model explanation susceptible to privacy-related attacks  ( Naretto
et al., 2022 ) . The leakage of information via model explanations can be attributed to a range of factors. Some of these factors are intrinsic to how explanations are crafted and the methodologies behind them, while others relate to the data’s sensitivity and the granularity of the information the explanations provide  ( Artelt et al., 2021 ) .

 
 
 Given the paramount importance of protecting data privacy while simultaneously enhancing the transparency of machine learning (ML) models across domains, both the academic community and industry stakeholders are keenly focused on the privacy aspects of model explanations. To our knowledge, this article represents the inaugural comprehensive review of privacy-preserving mechanisms within model explanations. Through this work, we present an initial investigation that encapsulates both privacy breaches and their countermeasures in the context of model explanations, alongside explainable ML methodologies that inherently prioritize privacy. Furthermore, we develop taxonomies grounded in diverse criteria to serve as a reference for related research fields.

 
 
 Figure 1 . This work vs. existing surveys. Explainable AI involves explanation and interpretable methods (e.g. ( Bodria et al., 2023 ; Guidotti et al., 2018 ; Gilpin et al., 2018 ) ). Adversarial AI includes adversarial attacks on ML models (e.g. ( Machado
et al., 2021 ; Biggio and Roli, 2018 ) ). Privacy AI involves privacy issues in ML (e.g. ( Rigaki and Garcia, 2023 ; Hu
et al., 2022b ; Liu et al., 2021 ) ). Others  ( Ferry et al., 2023b ; Baniecki and
Biecek, 2024 ) discuss exploits on model explanations. Our survey offers the first complete picture on privacy attacks, leaks, and defenses in explainable AI. 
 
 

### 1.1. Comparisons with existing surveys

 
 Many surveys have summarised different privacy issues on ML models  ( Biggio and Roli, 2018 ; Papernot et al., 2017 ; Machado
et al., 2021 ; Liu et al., 2021 ) , while others reviewed explanation methods for ML models  ( Gilpin et al., 2018 ; Bodria et al., 2023 ; Adadi and Berrada, 2018 ) , but not both. For example, Rigaki et al.  ( Rigaki and Garcia, 2023 ) presented a thorough analysis of over 45 publications on privacy attacks in machine learning, spanning the last seven years.
Hu et al.  ( Hu
et al., 2022b ) surveyed a special type of privacy attacks, called membership inference.
On the other hand, others  ( Guidotti et al., 2018 ; Adadi and Berrada, 2018 ; Došilović et al., 2018 ) offered a comprehensive classification of model explanations to enhance interpretability and guided the selection of suitable methods for specific ML models and desired explanations.

 
 
 Some existing surveys summarised adversarial attacks but presented partial coverage of privacy attacks on model explanations with basic introductions and limited discussions of the methods. Ferry et al.  ( Ferry et al., 2023b ) examined the interplay between interpretability, fairness, and privacy, which are critical for responsible AI, particularly in high-stakes decision-making like college admissions and credit scoring. Baniecki et al.  ( Baniecki and
Biecek, 2024 ) surveyed adversarial attacks on model explanations and fairness metrics, offered a unified taxonomy for clarity across related research areas, and discussed defensive strategies against such attacks.
However, these papers are either too high-level or too specialised in non-privacy attacks.

 
 
 Our survey presents an in-depth examination of privacy attacks on model explanations, diverging from previous work by its comprehensive nature. Rather than addressing the full spectrum of adversarial attacks, our study is specifically tailored to privacy attacks. This focus is due to the recent surge in these attacks and their significant potential to compromise the right to explanation  ( Goodman and
Flaxman, 2017 ) and the right to privacy  ( Banisar, 2011 ) . The threat posed by such privacy attacks could, in essence, challenge the very existence and usefulness of model explanations. Unlike the existing reviews that selected a very limited number of publications related to privacy attacks on model explanations (e.g. only two references are included in  ( Baniecki and
Biecek, 2024 ) ), we conduct a comprehensive search and include more than 50 related works in this survey. We delve into the underlying principles, theoretical frameworks, methodologies, and taxonomies, while also mapping out potential trajectories for future research. Especially, our work encompasses the emerging field of privacy-preserving explanations (PrivEx), highlighting model explanations that inherently protect user privacy  ( Vo et al., 2023 ; Mochaourab et al., 2021 ; Harder
et al., 2020 ) .

 
 
 

### 1.2. Paper collection methodology

 
 Finding relevant research on this subject proved to be complex due to its incorporation of various topics such as data privacy, privacy attacks, explanations of models, explainable AI (XAI), and the development of privacy-preserving explanations. To navigate this breadth of concepts, we employed diverse keyword combinations about “privacy”, “explanation”, and specific attack types including “membership inference”, “data reconstruction”, “attribute inference”, “model extraction”, “model stealing”, “property inference”, and “model inversion”. Our initial search utilised platforms like Google Scholar, Semantic Scholar, and Scite.ai – an AI-enhanced search tool – to assemble a preliminary collection of studies. This selection was expanded through backward searches, analysing the references of initially chosen papers, and forward searches, identifying papers that cited the initial ones. Additionally, we manually verified the relevance and focus of these articles across various sources due to discrepancies, such as some studies addressing privacy in the context of safeguarding against manipulation attacks instead of privacy intrusions. Ultimately, this process culminated in nearly 50 pivotal research papers on the topic.

 
 
 

### 1.3. Contributions of the article

 
 Figure 2 . Our taxonomy of privacy attacks and countermeasures on model explanations. “Exploit” arrows indicate existing works about privacy attacks on targeted explanations. “Support” arrows indicate existing works about privacy countermeasures for corresponding explanations. Some countermeasures (e.g. Privacy-Preserving ML) target privacy attacks directly and their arrows are omitted for brevity sake. 
 
 
 The main contributions of this article are:

 
 
 
 • 
 
 Comprehensive Review: To the best of our knowledge, this study represents the inaugural effort to thoroughly examine privacy-preserving model explanations.
We have collated and summarised a substantial body of literature, including papers published or in pre-print up to March 2024.

 

 • 
 
 Connected Taxonomies: 
We have organised all existing literature on PrivEx according to various criteria, including the types of explanations targeted and the methodologies employed in attacks and defences. Fig. 2 showcases the taxonomy we have developed to structure these works.

 

 • 
 
 Causal Analysis: Recent research has begun to investigate conditions that could lead to privacy leaks through model explanations, indicating that some explanation mechanisms inherently possess vulnerabilities. To this end, we dedicate a section to discuss the probable causes of these leaks.

 

 • 
 
 Challenges and Future Directions: Designing privacy-preserving explanations for machine learning models is an emerging field of research. From the surveyed literature, we highlight unresolved issues and suggest several potential research directions into both the offensive and defensive aspects of privacy in model explanations.

 

 • 
 
 Datasets and Metrics: In support of empirical research in PrivEx, we compile a comprehensive overview of datasets and evaluation metrics previously utilised in the field.

 

 • 
 
 Online Updating Resource: To facilitate research in privacy-preserving model explanations, we have established an open-source repository 1 1 
 1 
 
 
 
 https://github.com/tamlhp/awesome-privex , which aggregates a collection of pertinent studies, including links to papers and available code.

 

 
 
 
 

### 1.4. Organisation of the article

 
 The rest of the article is organised as follows. §2 revisits model explanations, acting as foundations for privacy attacks. §3 presents the taxonomy of privacy attacks on model explanations and provides in-depth descriptions, including threat model and attack scenarios. §4 discusses the causes of privacy leaks in model explanations. §5 explores countermeasures and a new class of privacy-preserving model explanations by design. §6 provides the pinpoints to existing resources including source code, datasets, and evaluation metrics. Finally, §7 contains a discussion on ongoing and upcoming research directions and §8 concludes the survey.

 
 
 
 

## 2. Model Explanations

 
 Model explanations serve to clarify the decisions a model renders concerning a specific querying sample denoted by x x represented as an n-dimensional feature vector ( x ∈ ℝ n x\in\mathbb{R}^{n} ). The explanation function ϕ \phi ingests the dataset D D , along with its labels – either the ground truth labels ℓ : D → [ C ] \ell:D\to[C] or those inferred by a trained model f f – and the query x ∈ ℝ n x\in\mathbb{R}^{n} . Such methods for explanation may require access to supplementary data  ( Chang and Shokri, 2021 ) , including the ability to query the model actively, a predefined notion of the data distribution, or familiarity with the class of the model  ( Shokri
et al., 2021 ) .

 
 
 Table 1 summarises important notations in this paper.

 
 
 Table 1 . Summary of Important Notations. 
 
 
 
 
 
 Notation 
 Description 
 
 f : X → Y f:X\rightarrow Y 
 A machine learning model 
 
 f t f_{t} 
 Target model of a privacy attack 
 
 f a f_{a} 
 Adversarial model by a privacy attack 
 
 D D 
 Training data 
 
 ϕ ⁡ ( x ) = ϕ ⁡ ( D , f , x ) \phi(x)=\phi(D,f,x) 
 Explanation on the input data x x 
 
 ϕ G ​ R ​ A ​ D \phi^{GRAD} (x) 
 Gradient-based explanation on input x x 
 
 ϕ I ​ N ​ T ​ G \phi^{INTG} (x) 
 Integrated gradient-based explanation on input x x 
 
 ϕ S ​ M ​ O ​ O ​ T ​ H \phi^{SMOOTH} (x) 
 Perturbation-based explanation on input x x 
 
 ϕ L ​ I ​ M ​ E \phi^{LIME} (x) 
 LIME explanation on input x x 
 
 ϕ S ​ H ​ A ​ P \phi^{SHAP} (x) 
 Shapley explanation on input x x 
 
 ϕ L ​ L ​ M \phi^{LLM} (x) 
 Locally linear map-based explanation on input x x 
 
 ϕ C ​ F \phi^{CF} (x) 
 Counterfactual explanation on input x x 
 
 c ​ f ​ ( x ) cf(x) 
 Counterfactual explanations/instances of the input data x x 
 
 M ​ I D ​ i ​ s ​ t ​ a ​ n ​ c ​ e ​ ( x ) MI_{Distance}(x) 
 Distance-based membership inference attack on x x 
 
 ∇ x f ​ ( x ) \nabla_{x}f(x) 
 Gradient of the model f f on x x 
 
 f ^ ( . ) \hat{f}(.) 
 Surrogate model produced by model extraction attack 
 
 ϵ \epsilon -DP 
 Different privacy with ϵ \epsilon degree or privacy budget 
 
 

 
 

### 2.1. Feature-based Explanations

 
 The explanation function ϕ ( D , f , x ; ⋅ ) \phi(D,f,x;\cdot) is predicated on identifying influential attributes (with the ⋅ \cdot symbol representing any potential additional inputs), and the explanation for the query x x is frequently referred to simply as ϕ ⁡ ( x ) \phi(x)   ( Chang and Shokri, 2021 ) .
The value at the i i -th index of a feature-based explanation, ϕ i ​ ( x ) \phi_{i}(x) , quantifies the extent of influence the i i -th feature exerts on the label ascribed to x x .
Ancona et al.  ( Ancona
et al., 2018 ) have curated a comprehensive exposition of these attribution-focused explanation modalities, also termed attribution methods or numerical influential measures  ( Shokri
et al., 2020 ) .

 
 
 Figure 3 . Feature-based explanations via backpropagation. 
 
 
 Backpropagation-based (aka gradient-based). 
This type of explanation explains the decisions of neural network models through the lens of back propagation  ( Shokri
et al., 2021 ) (see Fig. 3 ). It allows for the allocation of the model’s predictive reasoning back to the individual input features  ( Simonyan
et al., 2013 ; Bach et al., 2015 ; Shrikumar
et al., 2017 ; Sliwinski
et al., 2019 ; Smilkov et al., 2017 ; Sundararajan
et al., 2017 ) .

 
 
 
 • 
 
 (Vanilla) Gradients: 
Simonyan et al.  ( Simonyan
et al., 2013 ) introduces gradient-based explanations, originally for image classification models, to emphasises important image pixels that affect the predictive outcomes.
The explanation vector is defined as ϕ G ​ R ​ A ​ D ​ ( x ) = ∇ x f ​ ( x ) \phi^{GRAD}(x)=\nabla_{x}f(x) 
or ϕ i ​ ( x ) = ∂ f ∂ x i ​ ( x ) \phi_{i}({x})=\frac{\partial f}{\partial x_{i}}({x}) for each feature i i . A high partial differential value indicates that a pixel significantly affects the prediction, and analysing the map these values (so-called gradient map) can explain a model’s decision-making  ( Miura
et al., 2021 ) .
Shrikumar et al.  ( Shrikumar
et al., 2017 ) suggest enhancing numerical explanations by using the input feature value multiplied by the gradient, ϕ i ​ ( x ) = x i × ∂ f ∂ x i ​ ( x ) \phi_{i}({x})=x_{i}\times\frac{\partial f}{\partial x_{i}}({x}) .

 

 • 
 
 Integrated Gradients: 
Sundararajan et al.  ( Sundararajan
et al., 2017 ) advocate for an alternative to standard gradient computation by averaging gradients along a straight path from a baseline input x B ​ L x^{BL} (often x B ​ L = 0 → x^{BL}=\vec{0} ) to the actual input. This method follows critical axioms like sensitivity and completeness. Sensitivity ensures that if there’s a prediction change due to x i x_{i} not equaling x B ​ L , i x_{BL,i} , then ϕ i ​ ( x ) \phi_{i}({x}) should not be zero. Completeness dictates that the sum of all attributions equals the change in prediction from the baseline to the input.

 

 
 (1) | 
 | 
 ϕ I ​ N ​ T ​ G ​ ( x i ) = ( x i − x B ​ L , i ) ⋅ ∫ α = 0 1 ∂ c ⁡ ( x α ) ∂ x i α | x α = x + α ⁡ ( x − x B ​ L ) . \phi^{INTG}({x}_{i})=(x_{i}-x_{BL,i})\cdot\int_{\alpha=0}^{1}\frac{\partial c({x}^{\alpha})}{\partial x^{\alpha}_{i}}\bigg|_{{x}^{\alpha}={x}+\alpha({x}-x^{BL})}. | 
 | 
 

 

 • 
 
 Guided Backpropagation: 
Designed for networks with ReLU activations (others as well), Guided Backpropagation  ( Springenberg et al., 2014 ) modifies the gradient to only reflect paths with positive weights and activations, thereby considering only the positive evidence for a specific prediction.

 

 • 
 
 Layer-wise Relevance Propagation (LRP): 
proposed by Klauschen et al.  ( Bach et al., 2015 ) to assign relevance from the output layer back to the input features. The relevance in each layer is proportionally distributed according to the contribution from neurons in the previous layer. The final attributions for the input are referred to as ϕ L ​ R ​ P ​ ( x ) \phi^{LRP}({x}) .

 

 
 
 
 Perturbation-based. 
Perturbation-based explanations involve querying a model that needs to be explained with a series of altered inputs  ( Shokri
et al., 2021 ) .
SmoothGrad  ( Smilkov et al., 2017 ) is a popular perturbation-based explanation method that produces several samples by injecting Gaussian noise into the input data and then computes the mean of the gradients from these samples.Formally, for a certain k k samples, the explanation function is defined as:

 

 
 (2) | 
 | 
 ϕ SMOOTH ​ ( x ) = 1 k ​ ∑ k ∇ f ( x + 𝒩 ⁡ ( 0 , σ ) ) , \phi^{\text{SMOOTH}}({x})=\frac{1}{k}\sum_{k}\nabla_{f}({x}+\mathcal{N}(0,\sigma)), | 
 | 
 

 where 𝒩 \mathcal{N} represents the normal distribution and σ \sigma stands for a hyperparameter that controls the level of perturbation.

 
 
 

### 2.2. Interpretable Surrogates

 
 This method explains a black-box ML model or complex deep neural networks by computing a surrogate model that is interpretable by design  ( Shokri
et al., 2021 ; Deng, 2019 ; Guidotti et al., 2018 ) that can emulate the overall predictive patterns of the original model  ( Naretto
et al., 2022 ) .

 
 
 LIME. 
Local Interpretable Model-agnostic Explanations  ( Ribeiro
et al., 2016 ) generate a local interpretative approximation of a given model through sampling on the optimisation problem:

 

 
 (3) | 
 | 
 ϕ LIME ​ ( x ¯ ) = arg ⁡ min g ∈ G ⁡ ℒ ⁡ ( g , f , π x ) + Ω ⁡ ( g ) , \phi^{\text{LIME}}(\bar{x})=\arg\min_{g\in G}\mathcal{L}(g,f,\pi_{{x}})+\Omega(g), | 
 | 
 

 where G G is a collection of interpretable functions employed for explanatory purposes, ℒ \mathcal{L} quantifies how well g g approximates f f in the neighbourhood π x \pi_{{x}} of x {x} , and Ω \Omega imposes a regularisation on g g to avoid overfitting. Usually, G G involves one or multiple linear models and Ω \Omega is a Ridge regularisation  ( Shokri
et al., 2021 ) .
The loss function is typically computed as the expected squared difference between the outputs of f f and g g weighted by the probability distribution π X \pi_{X}   ( Slack et al., 2020 ) :

 

 
 (4) | 
 | 
 L ⁡ ( f , g , π X ) = ∑ x ′ ∈ X ′ [ f ⁡ ( x ′ ) − g ⁡ ( x ′ ) ] 2 ​ π X ​ ( x ′ ) L(f,g,\pi_{X})=\sum_{x^{\prime}\in X^{\prime}}[f(x^{\prime})-g(x^{\prime})]^{2}\pi_{X}(x^{\prime}) | 
 | 
 

 where X ′ X^{\prime} is the neighbourhood of x x .

 
 
 SHAP (local). 
The main distinction between LIME and SHAP is in the selection of the functions Ω \Omega and π x \pi_{x} . LIME takes a heuristic approach: Ω ⁡ ( g ) \Omega(g) represents the count of non-zero weights within the linear model, while π x ​ ( x ′ ) \pi_{x}(x^{\prime}) utilises either cosine or l2 distance  ( Slack et al., 2020 ) .
SHAP values provide a way to quantify the contribution of each feature in a model prediction  ( Jetchev and
Vuille, 2023 ; Datta
et al., 2016 ; Lundberg and Lee, 2017 ; Štrumbelj
and Kononenko, 2014 ; Maleki et al., 2013 ) . Specifically, for a given model f f and a data point x = [ x 1 , … , x M ] x=[x_{1},\ldots,x_{M}] , the SHAP value for feature i i is calculated as a weighted average of differences between the model prediction with and without feature i i :

 

 
 (5) | 
 | 
 ϕ i S ​ H ​ A ​ P ​ ( x ) = ∑ S ⊆ { 1 , … , M } ∖ { i } 1 M ​ f S ∪ { i } ​ ( x ) − f S ​ ( x ) ( M − 1 | S | ) \phi^{SHAP}_{i}(x)=\sum_{S\subseteq\{1,\ldots,M\}\setminus\{i\}}\frac{1}{M}\frac{f_{S\cup\{i\}}(x)-f_{S}(x)}{{M-1\choose|S|}} | 
 | 
 

 where | S | |S| is the size of the subset S S and M M is the total number of features.
For instance, let x 0 = [ x i 0 ] i = 1 M x^{0}=[x^{0}_{i}]_{i=1}^{M} be a reference sample of M M features. Suppose M = 4 M=4 , x = [ 5 , 2 , 7 , 3 ] x=[5,2,7,3] , x 0 = [ 0 , 0 , 0 , 0 ] x^{0}=[0,0,0,0] , and we want to compute the marginal contribution s i s_{i} of feature i = 1 i=1 to the feature set S = { 2 , 3 } S=\{2,3\} . Then s i = 1 4 ​ f ⁡ ( x [ 1 , 2 , 3 ] ) − f ⁡ ( x [ 2 , 3 ] ) 3 = f ⁡ ( [ 5 , 2 , 7 , 0 ] ) − f ⁡ ( [ 0 , 2 , 7 , 0 ] ) 12 s_{i}=\frac{1}{4}\frac{f(x_{[1,2,3]})-f(x_{[2,3]})}{3}=\frac{f([5,2,7,0])-f([0,2,7,0])}{12} .

 
 
 Global Shapley Values. 
The above Shapley values are local because the explanations are based on a singular reference sample x 0 x^{0} and a single input sample x x   ( Slack et al., 2020 ) .
Begley et al.  ( Begley
et al., 2020 ) proposes a Global Shapley Value by averaging local Shapley values over both foreground and background distributions, as given by:

 

 
 (6) | 
 | 
 Φ i S ​ H ​ A ​ P ​ ( f , F , B ) = 𝔼 ⁡ [ ϕ i ​ ( f , x , x 0 ) ] \Phi^{SHAP}_{i}(f,F,B)=\mathbb{E}[\phi_{i}(f,x,x^{0})] | 
 | 
 

 for each feature index i = 1 , 2 , … , M i=1,2,\ldots,M .
In other words, to conduct a global analysis of model behavior, it is necessary to consider predictions at multiple inputs x ∼ ℱ x\sim\mathcal{F} from a distribution ℱ \mathcal{F} called the foreground. Since the choice of baseline x 0 x^{0} is ambiguous, baselines x 0 ∼ ℬ x^{0}\sim\mathcal{B} are sampled from a distribution ℬ \mathcal{B} called the background.

 
 
 Locally Linear Maps. 
Harder et al.  ( Harder
et al., 2020 ) introduces Locally Linear Maps (LLM), a method aimed at providing both local and global explanations for models, which is more expressive than standard linear models and offers an efficient way to manage the number of parameters for a good privacy-accuracy trade-off.

 

 
 (7) | 
 | 
 ϕ k L ​ L ​ M ​ ( x ) = ∑ m = 1 M σ ​ ( x ) m k ​ g m k ​ ( x ) ,  where  ​ g m k ​ ( x ) = w m k ⋅ x + b m k , \phi^{LLM}_{k}(x)=\sum_{m=1}^{M}\sigma(x)^{k}_{m}g^{k}_{m}(x),\text{ where }g^{k}_{m}(x)=w^{k}_{m}\cdot x+b^{k}_{m}, | 
 | 
 

 and the weighting coefficients are computed via softmax:

 

 
 (8) | 
 | 
 σ m k ​ ( x ) = exp ⁡ [ β ⋅ g m k ​ ( x ) ] ∑ m = 1 M exp ⁡ [ β ⋅ g m k ​ ( x ) ] . \sigma^{k}_{m}(x)=\frac{\exp[\beta\cdot g^{k}_{m}(x)]}{\sum_{m=1}^{M}\exp[\beta\cdot g^{k}_{m}(x)]}. | 
 | 
 

 The method optimizes a cross-entropy loss ℒ ⁡ ( W , 𝒟 ) \mathcal{L}(W,\mathcal{D}) for the parameters of LLM collectively denoted by W W , with the predictive class label y n , k ​ ( W ) y_{n,k}(W) defined through a softmax function applied to the output of ϕ k ​ ( x n ) \phi_{k}(x_{n}) .

 
 
 

### 2.3. Example-based Explanations

 
 Example-based explanation (aka case-based interpretability or record-based explanation  ( Shokri
et al., 2020 ) ) uses comparable examples to create transparent explanations for machine learning decisions, offering an accessible way to understand model predictions by contrasting similar cases from the model’s database or generated data  ( Montenegro et al., 2022 ) .
Case-based interpretability techniques can create a range of explanatory examples, including:

 
 
 
 • 
 
 Similar examples: are the closest matches from the training data with corresponding predictions to the case being analyzed, identified through a defined measure of similarity.

 

 • 
 
 Typical examples: representing the epitome of a particular prediction, frequently utilized in models that focus on prototype learning.

 

 • 
 
 Counterfactual examples: are similar examples but with differing predictions, highlighting the minimal changes needed for a different outcome. We dedicate a separate discussion on counterfactuals in the next subsection.

 

 • 
 
 Semi-factual examples: are similar to the original case with the same prediction but positioned near the decision boundary, demonstrating the robustness of the prediction against variations typical of a different classification.

 

 • 
 
 Influential examples : are key data points within a training set that have a significant impact on a model’s prediction for a given query instance  ( Koh and Liang, 2017 ) .
For explanatory purposes, we can provide the top k k influential points  ( Shokri
et al., 2020 ) .

 

 
 
 
 These explanations can be sourced from existing datasets (i.e. ϕ ( D , f , x ; . ) ∈ D \phi(D,f,x;.)\in D )  ( Koh and Liang, 2017 ) or crafted based on the original data  ( Kenny
et al., 2021 ; Lipton, 2018 ) .

 
 
 Intrinsic methods for traditional ML. 
Case-based explanations in machine learning are derived from either distance-based or prototype-based interpretable methods. Distance-based methods utilize a measure of proximity to retrieve the most similar data points as explanations, while prototype-based methods classify and explain instances based on representative prototypes of clustered data. The K-Nearest Neighbors (KNN) algorithm exemplifies the former, offering explanations as similar or counterfactual examples based on label correspondence. The Bayesian Case Model (BCM) is a prototype-based method that explains decisions through typical examples representative of data clusters  ( Kim
et al., 2014 ) . Both methods aim to make model decisions understandable by referencing specific, characteristic data points or clusters  ( Montenegro et al., 2022 ) .

 
 
 Posthoc methods for traditional ML. 
Post hoc interpretability techniques leverage traditional machine learning models as metrics for finding similar examples, with decision trees and rule-based models used to determine similarity between data samples  ( Montenegro et al., 2022 ) .
Counterfactual examples, on the other hand, come from nodes with differing outcomes. Moreover, models like Explanation Oriented Retrieval (EOR), built on the K-Nearest Neighbors (KNN) algorithm, reorder neighbors to highlight those with the highest explanatory utility, thus providing semi-factual examples that maintain the same classification but are closer to the decision boundary  ( Nugent
et al., 2009 ) .

 
 
 Intrinsic methods for deep learning. 
In deep learning, intrinsic interpretability can be provided by prototype-based or distance-based methods  ( Montenegro et al., 2022 ) . For instance, the Explainable Deep Neural Network (xDNN)  ( Angelov and
Soares, 2020b ) and Deep Machine Reasoning (DMR)  ( Angelov and
Soares, 2020a ) define prototypes as dense data points and classify observations based on the closest prototype.
The Prototype Classifier method learns representative prototypes from training data, using an autoencoder for feature extraction and classification based on latent representations  ( Li
et al., 2018 ) .The Prototypical Part Network (ProtoPNet) represents image parts in clusters in a latent space, which are used to predict and explain classifications  ( Chen
et al., 2019 ) . Additionally, the Deep k-Nearest Neighbors (DkNN) calculates neighbors at each model layer to ensure consistent predictions, offering explanations based on similar examples across the model’s entirety  ( Papernot and
McDaniel, 2018 ) .

 
 
 Posthoc methods for deep learning. 
Post hoc interpretability methods in deep learning either utilise interpretable surrogate models to extract explanations from a primary model or directly analyse a “black box” model to identify anCSUx retrieve the most similar data instances for explanation purposes ( Montenegro et al., 2022 ) . Concept Whitening, for example, organises the latent space of a classification network around predefined concepts, enabling the measurement of distance between instances for similar example retrieval ( Chen
et al., 2020 ) . Interpretability guided Content-based Image Retrieval (IG-CBIR) enhances image retrieval by using saliency maps to focus on relevant image regions ( Silva
et al., 2020 ) . Unsupervised clustering and the KNN algorithm within the Twin Systems framework are other surrogate models that categorise or find similar examples based on feature extraction techniques like perturbation and sensitivity analysis ( Kim and Chae, 2024 ; Kenny and Keane, 2019 ) .

 
 
 

### 2.4. Counterfactual Explanations

 
 Counterfactual explanations (aka algorithmic recourse) provide insights into how slight changes to input features could lead to different model outcomes, aiding in tasks like model debugging and ensuring regulatory compliance  ( Goethals
et al., 2023 ; Kuppa and Le-Khac, 2021 ) . The study in  ( Kuppa and Le-Khac, 2021 ) gives an illustration of counterfactual and other four sample categories (i. e., adversarial examples, local robustness  ( Zhang
et al., 2024 ) , invariant samples, and uncertainty samples) through the boundaries between human analyst and a learnt model (see Fig. 4 ).
The application of counterfactual explanations varies with the model’s complexity and includes considerations such as model transparency, type compatibility, and adherence to constraints like feasibility and causality  ( Wachter
et al., 2017 ; Dodge
et al., 2019 ; Binns et al., 2018 ) .
The concept overlaps with other areas of research such as algorithmic recourse, inverse classification, and contrastive explanations  ( Karimi
et al., 2021 ; Ustun
et al., 2019 ; Laugel et al., 2017 ; Dhurandhar et al., 2018 ) .

 
 
 Figure 4 . Decision boundaries between human analyst and a learnt model. 
 
 
 Single counterfactual. 
Formally, counterfactual explanation is the process of finding changes δ \delta to an instance x x that reverse a negative predictive outcome from a model f θ ​ ( x ) = 0 f_{\theta}(x)=0 to a positive one f θ ​ ( x + δ ) = 1 f_{\theta}(x+\delta)=1 , where θ \theta are model parameters. The problem involves identifying a counterfactual x ′ = x + δ x^{\prime}=x+\delta where the predictive model outputs a positive outcome and doing so with minimal cost c ⁡ ( x , x ′ ) c(x,x^{\prime}) , which is easily implementable, often using ℓ 1 \ell_{1} or ℓ 2 \ell_{2} distance as cost functions.
The optimization problem is defined as:

 

 
 (9) | 
 | 
 ϕ C ​ F ​ ( x ) = arg min x ′ ∈ A P ​ L ​ ( f θ ​ ( x ′ ) , 1 ) + λ ⋅ c ⁡ ( x , x ′ ) \phi^{CF}(x)=\text{arg min}_{x^{\prime}\in A^{P}}L(f_{\theta}(x^{\prime}),1)+\lambda\cdot c(x,x^{\prime}) | 
 | 
 

 where A P A^{P} is the set of plausible or actionable counterfactuals and L ( . , . ) L(.,.) is a differential loss such as binary cross entropy  ( Pawelczyk
et al., 2023 ) .

 
 
 Example 0 . 
 
 Possible counterfactual explanations derived from the FICO explainable machine learning challenge dataset  ( Sokol and Flach, 2019 ) : 

 
 • 
 
 The model prediction for creditworthiness is negative. If the number of satisfactory trade lines had been 10 or fewer, rather than the actual 20, the prediction would have been positive. 

 

 • 
 
 The model prediction for creditworthiness is negative. If there had been no trade lines that were ever 60 days overdue and marked as derogatory in the public record, rather than the actual count of 2, the prediction would have shifted to positive. 

 

 
 
 
 
 Diverse counterfactuals. 
Recent works study the generation of multiple alternative counterfactuals per input, offering a spectrum of potential changes rather than just one nearest option  ( Mothilal
et al., 2020 ) . This approach empowers users by offering them various ways they could potentially modify their data to achieve a preferred result  ( Thang
et al., 2015 ; Nguyen
et al., 2015a ; Zhao et al., 2021a ) .

 
 
 Kuppa et al.  ( Kuppa and Le-Khac, 2021 ) notes that methods for creating counterfactual explanations (CF) bear resemblance to those for generating adversarial examples (AE) in the way they both employ gradient-based optimization and surrogate models to find CF/AE for a given model. Some privacy attacks on adversarial examples can be used on counterfactual explanations  ( Kuppa and Le-Khac, 2021 ) .

 
 
 
 

## 3. Privacy Attacks

 
 According to a classification system mentioned in  ( Biggio and Roli, 2018 ; Baniecki and
Biecek, 2024 ) , explainable AI systems can fall prey to three main categories of attacks: (i) integrity attacks, such as evasion and backdoor poisoning, leading to incorrect categorisation of certain data points  ( Severi
et al., 2021 ; Kuppa and Le-Khac, 2020 ; Liu
et al., 2022c ; Nguyen
et al., 2023b ) ;
(ii) availability attacks, characterised by poisoning efforts aimed at inflating the error rate in classification tasks  ( Abdukhamidov et al., 2023 ) ; and (iii) privacy and confidentiality attacks, aimed at extracting sensitive information from user data and the models themselves. Although all forms of interference in machine learning can be considered adversarial, “adversarial attacks" specifically denote those targeting the security aspect, particularly through malicious samples  ( Garcia
et al., 2018 ; Slack et al., 2020 ; Aïvodji et al., 2022 ; Zhang
et al., 2020b ) .

 
 
 This work is primarily concerned with breaches of privacy and confidentiality, including membership inference attacks, linkage attacks, reconstruction attacks, attribute/feature inference attacks, and model extraction attacks.
The rationale behind including model extraction attacks is their frequent association with privacy violations in related literature  ( Rigaki and Garcia, 2023 ) , and the notion that hijacking a model’s functions could also infringe on privacy. Veale et al.  ( Veale
et al., 2018 ) contends that privacy violations like membership inference attacks elevate the likelihood of machine learning models being deemed personal data under the European Union’s General Data Protection Regulation (GDPR), as they could make individuals identifiable.

 
 

### 3.1. Membership Inference Attacks (MIA)

 
 MIA aim to detect if data is part of a model’s training set  ( Shokri
et al., 2019 ; Shokri
et al., 2021 ) . Before model explanations, popular attacks are loss thresholding and likelihood ratio attack (LRT)  ( Pawelczyk
et al., 2023 ) .
Loss thresholding identifies if a data point was in the training set by checking the model’s error rate against a threshold, requiring access to labels and model details  ( Yeom
et al., 2018 ; Sablayrolles et al., 2019 ) . LRT, in contrast, uses shadow models to compare confidence levels of data being in or out of the training set, calculating a likelihood ratio to predict membership without needing direct model access  ( Carlini et al., 2022 ) .
Pawelczyk et al.  ( Pawelczyk
et al., 2023 ) designs a recourse-based attack (using counterfactual explanation) without access to the true labels and knowledge of the correct loss functions.

 
 
 Figure 5 . Membership inference attacks. 
 
 
 Threat model. 
The adversary is able to submit x x to the black-box model  ( Liu
et al., 2022d ; Li
et al., 2022 ; Carlini et al., 2022 ; Ye et al., 2022 ) to receive the prediction f ⁡ ( x ) f(x) and any corresponding explanations, despite not having direct access to the model’s internals  ( Quan et al., 2022 ) (see  Fig. 5 ).
However, they are assumed to know the model’s architecture and possess an auxiliary dataset similar to the model’s training data, reflected in much of the current research on the topic  ( Liu
et al., 2024d ) .

 
 
 
 • 
 
 Threat model on gradient-based explanations: 
Most threat models are based on threshold-based attacks  ( Shokri
et al., 2021 ) . There are two key scenarios for this: the optimal threshold scenario, where the threshold is deduced from known data point memberships to gauge the maximum privacy risk; and the reference/shadow model scenario, which is more practical and assumes the attacker has some labeled data from the same distribution as the target model, as well as knowledge of the model’s architecture and hyperparameters in line with Kerckhoffs’s principle  ( Petitcolas, 2023 ) . The attacker then trains a number of shadow models on this data to approximate the threshold, an approach that becomes more resource-intensive as the number of shadow models increases  ( Shokri
et al., 2021 ) .

 

 • 
 
 Threat model on interpretable surrogates: 
Naretto et al.  ( Naretto
et al., 2022 ) investigates how global explanation methods can potentially compromise the privacy. Specifically, the authors focus on TREPAN  ( Craven and
Shavlik, 1994 ) , an algorithm that explains neural network decisions by creating a surrogate Decision Tree (DT) model.

 

 • 
 
 Threat model on counterfactuals: 
Pawelczyk et al.  ( Pawelczyk
et al., 2023 ) formulates a membership inference game for attacking counterfactual explanations. The game features two participants: a model owner ( 𝒪 \mathcal{O} ) and an opponent ( 𝒜 \mathcal{A} ). Their actions are as follows. 𝒪 \mathcal{O} selects a dataset for training from a population D N D^{N} , applying a training algorithm T T with a loss function ℓ \ell .
Subsequently, 𝒪 \mathcal{O} assigns a binary label f θ ​ ( z ) f_{\theta}(z) to each datapoint z z in D t D_{t} . Let D t 0 D_{t}^{0} be the segment of training data for which f θ ​ ( x ) = 0 f_{\theta}(x)=0 , and D θ , 0 D_{\theta,0} represent the conditional distribution p ⁡ ( z ) | f θ ​ ( z ) = 0 p(z)|f_{\theta}(z)=0 . 𝒪 \mathcal{O} tosses a coin, and based on the outcome, selects a sample x x from either D θ , 0 D_{\theta,0} or D t + D_{t}^{+} . Then, using the recourse algorithm ϕ \phi , 𝒪 \mathcal{O} generates an alternate instance x ′ x^{\prime} from ϕ ⁡ ( f θ , x , D t ) \phi(f_{\theta},x,D_{t}) and sends the pair ( x ′ , x ) (x^{\prime},x) to 𝒜 \mathcal{A} . In addition to the sample pair, 𝒜 \mathcal{A} has the capability to make queries to D D . It is presumed that 𝒜 \mathcal{A} is fully aware of 𝒪 \mathcal{O} ’s implementation specifics, including the training algorithm T T and the recourse algorithm ϕ \phi . 𝒜 \mathcal{A} concludes the game by providing a binary guess G G signifying if x x belongs to D t D_{t} (MEMBER) or does not ( x ∉ D t x\notin D_{t} , NON-MEMBER).

 

 
 
 
 General attacks. 
In the training set, data points are generally positioned away from the decision boundary, leading to lower loss scores that can be leveraged to detect membership in the training data  ( Quan et al., 2022 ; Sablayrolles et al., 2019 ; Yeom
et al., 2018 ) . This principle is utilized in the OPT-var method  ( Shokri
et al., 2021 ) , in which the variance in the explanation e = ϕ ⁡ ( f , x ) e=\phi(f,x) based on the logit score f ⁡ ( x ) f(x) could signal whether a point was in the training set. However, Quan et al.  ( Quan et al., 2022 ) argues that logit scores alone may not fully represent the prediction confidence of the victim model because they do not take into account the scores of other classes. Instead, Quan et al.  ( Quan et al., 2022 ) suggests using the softmax function σ ⁡ ( f ⁡ ( x ) ) \sigma(f(x)) , which reflects class interactions, to provide a more comprehensive membership indicator.

 
 
 Figure 6 . Model-based membership inference attacks proposed in  ( Liu
et al., 2024d ) . 
 
 
 Liu et al.  ( Liu
et al., 2024d ) proposes a model-based attack that involves four main stages: training a shadow model, extracting attribution features, training an attack model, and inferring membership (see  Fig. 6 ). The adversary starts by training a shadow model using an auxiliary dataset that is similar to the training data of the target model. Then, attribution maps are generated for a given sample, and perturbations are applied based on these maps to observe changes in predictions.
Next, the adversary trains an attack model, typically a Multi-Layer Perceptron (MLP), using the attribution features combined with other data such as loss values and one-hot encoded class information to construct features indicative of membership.

 
 
 
 • 
 
 Attacks on gradient-based explanations: 
Shokri et al.  ( Shokri
et al., 2021 ) uses a threshold-based attack that infers membership based on the model’s confidence or its explanation output. A data point is classified as a member if the variance of the confidence scores V ​ a ​ r ​ ( f θ ​ ( x ) ) Var(f_{\theta}(x)) or the variance of the explanation V ​ a ​ r ​ ( ϕ ⁡ ( x ) ) Var(\phi(x)) is below or equal to a certain threshold τ \tau .
Attacks using explanation variance exploit the model’s certainty: when a model is sure about a prediction, explanation variance is low. However, near the decision boundary, even small changes can increase explanation variance. Models with certain activation functions like tanh, sigmoid, or softmax have steeper gradients, affecting how training data points are positioned relative to these boundaries  ( Shokri
et al., 2021 ) .

 

 • 
 
 Attacks on interpretable surrogates: 
Naretto et al.  ( Naretto
et al., 2022 ) develops an attacking procedure to assess the potential privacy risks of an interpretable surrogate (global explainer) that attempts to replicate the behavior of a black-box model.
First, an MIA model, denoted as A b A_{b} , is trained to determine whether a specific data record, x x , was included in the training dataset, D t ​ r ​ a ​ i ​ n b D_{train}^{b} , of the black-box model b b . This attack model leverages the black-box b b itself to classify the training data for the attack, making it specifically aimed at b b . The attack training dataset D t ​ r ​ a ​ i ​ n a D_{train}^{a} is the same as D A ​ t ​ t ​ a ​ c ​ k B D_{Attack}^{B} . Similarly, another MIA model, A c A_{c} , is developed to target the global explainer c c , which serves as an interpretable stand-in for the black-box model b b . This model is trained using D t ​ r ​ a ​ i ​ n a D_{train}^{a} , but this time the labeling is done by c c , not b b .

 

 • 
 
 Attacks on counterfactual explanations: 
The adversary has access to both the original instance x x and a counterfactual instance x ′ x^{\prime} .
Models often overfit to training points, resulting in lower losses for these points compared to those on the test set  ( Shokri
et al., 2021 ) .
Pawelczyk et al.  ( Pawelczyk
et al., 2023 ) designs a distance-based attack where if the loss is below a certain threshold τ \tau , the point is considered a MEMBER of the training set.
The counterfactual distance c ⁡ ( x , x ′ ) c(x,x^{\prime}) is effectively the distance to the model boundary, and even though algorithms that produce realistic recourses may not optimize for this distance, it can still be viewed as an approximation to the distance to the model boundary  ( Karimi
et al., 2021 ; Pawelczyk
et al., 2020a ) .
The counterfactual distance-based attack is defined by M ​ I D ​ i ​ s ​ t ​ a ​ n ​ c ​ e ​ ( x ) MI_{Distance}(x) as follows:

 

 
 (10) | 
 | 
 M ​ I D ​ i ​ s ​ t ​ a ​ n ​ c ​ e ​ ( x ) = { Member if  ​ c ​ ( x , x ′ ) ≥ τ D Non-member if  ​ c ​ ( x , x ′ ) τ D MI_{Distance}(x)=\begin{cases}\text{Member} \text{if }c(x,x^{\prime})\geq\tau_{D}\\
\text{Non-member} \text{if }c(x,x^{\prime}) \tau_{D}\end{cases} | 
 | 
 

 Another attack is using a Likelihood Ratio Test on top of the Counterfactual Distance (CFD)  ( Pawelczyk
et al., 2023 ) .
The process involves calculating a baseline statistic t 0 t_{0} using c ⁡ ( x , x ′ ) c(x,x^{\prime}) from the recourse output.
If the initial statistic t 0 t_{0} surpasses the critical threshold z 1 − α z_{1-\alpha} , which is the 1 − α 1-\alpha quantile of the normal distribution Z Z , the algorithm designates the data point as a ‘Non-member’; and ‘Member’ otherwise.
The key benefit is that it estimates the parameters μ o ​ u ​ t , σ o ​ u ​ t \mu_{out},\sigma_{out} only once for the non-membership scenario, reducing the computational load when assessing multiple data points x ′ x^{\prime}   ( Sablayrolles et al., 2019 ) .

 
 
 Huang et al.  ( Huang
et al., 2023 ) proposes a CFD-based Likelihood Ratio Test (LRT) for linear classifiers built on the above Pawelczyk method  ( Pawelczyk
et al., 2023 ) .
But the attack is simplified and one-sided as it only estimates parameters for data outside the training set, thus reducing computational complexity.

 
 
 Kuppa et al.  ( Kuppa and Le-Khac, 2021 ) develops an attack that leverages an auxiliary dataset D a ​ u ​ x D_{aux} to train a shadow model A M ​ e ​ m ​ I ​ n ​ f A_{MemInf} .
This is done by generating counterfactual examples x c ​ f ​ i x_{cfi} for input samples x i x_{i} and training a 1-nearest neighbor (1-NN) classifier to predict class membership based on proximity to these counterfactuals. If the prediction probability difference between the shadow model A M ​ e ​ m ​ I ​ n ​ f A_{MemInf} and the target model T T is below a threshold t t , the sample is deemed part of the training set. This inference is made under the assumption that if both models predict similarly for a sample, it implies the sample was significant in its prediction. The method is advantageous as it requires no direct access to the training set and iteratively uses counterfactuals to extract new data.

 

 
 
 
 

### 3.2. Linkage Attacks

 
 Threat model. 
Goethals et al.  ( Goethals
et al., 2023 ) introduces a privacy concern with counterfactual explanations when they are based on training instances. The data usually consist of identifiers (like name and social security number), quasi-identifiers (like age, zip code, gender), and private attributes.
It has been shown that a significant portion of US citizens could be uniquely identified by combining their zip code, gender, and date of birth  ( Sweeney, 2000 ) .
The attack setup assumes the adversary has access to identifiers and quasi-identifiers. There are two re-identification scenarios discussed: one where a specific individual is targeted to uncover their private attributes, and another where the adversary aims to prove that re-identification is possible, regardless of who the individual is. Counterfactual explanations, which do not include identifiers but may contain unique combinations of quasi-identifiers, could be exploited by an attacker to infer private attributes in what is termed an “explanation linkage attack” or "re-identification attack"  ( Goethals
et al., 2023 ) (see  Fig. 7 ).

 
 
 Figure 7 . Linkage attacks. 
 
 
 Attacks on counterfactual explanations. 
Goethals et al.  ( Goethals
et al., 2023 ) presents a scenario where Lisa is denied credit and requests a counterfactual explanation, which inadvertently reveals Fionas’ private information because Fiona is the nearest unlike neighbor in the dataset.
Native counterfactuals, which are real instances from the dataset, are more plausible but increase the risk of re-identification  ( Brughmans
et al., 2023 ) . Perturbation-based counterfactuals, which synthetically generate explanations, pose less privacy risk but can still be vulnerable to sophisticated attacks if the perturbations are minor  ( Artelt et al., 2021 ; Keane and Smyth, 2020 ; Pawelczyk
et al., 2020b ) .
Aivodji et al.  ( Aïvodji
et al., 2020 ) identifies that diverse counterfactual explanations can inadvertently expose decision boundaries more, risking the leak of sensitive data like health or financial information. Linkage attacks exploit this by matching anonymised records with external datasets, combining various attributes to re-identify individuals.

 
 
 

### 3.3. Reconstruction Attacks

 
 Based on model predictions and explanations, reconstruction attacks involve dataset reconstrcution attacks, model reconstruction attacks, and model inversion attacks (see  Fig. 8 ).

 
 
 Dataset reconstruction attacks. 
It is important to preserve privacy in datasets due to several threats posed by inference attacks that seek to deduce sensitive information from model outputs  ( Dwork
et al., 2017 ; Rigaki and Garcia, 2023 ) . Ferry et al.  ( Ferry et al., 2023a ; Ferry, 2023 ) reviews the evolution of reconstruction attacks from databases to machine learning, where adversaries attempt to recover training data. Techniques range from linear programming to exploiting data memorisation, even within frameworks meant to promote fairness  ( Garfinkel
et al., 2019 ; Song
et al., 2017 ) . The goal of data reconstruction attacks is to make models trained for fairness inadvertently reveal sensitive attributes, including leveraging auxiliary datasets and queries to an auditor for enhancing attacks  ( Carlini et al., 2019 ; Salem et al., 2020 ) .

 
 
 Figure 8 . Reconstruction attacks. 
 
 
 
 • 
 
 Threat model: 
A machine learning model that is interpretable, like a decision tree, contains implicit information about its training dataset  ( Ferry et al., 2023a ) . This information can be formalized into a probabilistic dataset 𝒟 \mathcal{D} consisting of n n examples, each with d d attributes. Every attribute a k a_{k} has a domain V k V_{k} covering all possible attribute values. The knowledge about an attribute a k a_{k} for a given example x i x_{i} is represented by a probability distribution across all possible values for that attribute, using the random variable 𝒟 i , k \mathcal{D}_{i,k} .
If a value 𝒟 i , k \mathcal{D}_{i,k} within V k V_{k} has all the probability mass (i.e., P ⁡ ( 𝒟 i , k = v i , k ) = 1 P(\mathcal{D}_{i,k}=v_{i,k})=1 ), it’s deterministic.
Conversely, a probabilistic dataset encompasses some uncertainty about attribute values.

 

 • 
 
 Probabilistic Reconstruction Attacks: 
Earlier research  ( Gambs
et al., 2012 ) proposes a method for constructing a probabilistic dataset 𝒟 D ​ T \mathcal{D}^{DT} from the structure of a trained decision tree D ​ T DT . This probabilistic dataset reflects the decision tree’s implicit knowledge about its training dataset 𝒟 O ​ r ​ i ​ g \mathcal{D}^{Orig} . The construction of this dataset is termed a probabilistic reconstruction attack, and by design, 𝒟 D ​ T \mathcal{D}^{DT} is compatible with 𝒟 O ​ r ​ i ​ g \mathcal{D}^{Orig} , meaning the actual value v i , k O ​ r ​ i ​ g v_{i,k}^{Orig} of any attribute a k a_{k} for any example x i x_{i} is always among the set of possible values in the probabilistic reconstruction ( P ⁡ ( 𝒟 i , k D ​ T = v i , k O ​ r ​ i ​ g ) 0 P(\mathcal{D}_{i,k}^{DT}=v_{i,k}^{Orig}) 0 ).

 

 • 
 
 Attacks on Interpretable Models: 
Ferry et al.  ( Ferry et al., 2023a ) discusses the possibility of a probabilistic reconstruction attack on interpretable models.
In the general case, the success of the attack is calculated using the joint entropy of the dataset’s cells, which can be simplified if the variables of the model are statistically independent. For interpretable models like decision trees and rule lists, this assumption allows further decomposition of the computation  ( Ferry et al., 2023a ) .

 

 
 
 
 Model reconstruction attacks. 
Model reconstruction is the process of replicating a classifier f ^ \hat{f} when provided with membership and gradient queries to an oracle that, for any input x x , reveals both the classifier’s output f ^ ​ ( x ) \hat{f}(x) and the gradient ∇ x f ^ ​ ( x ) \nabla_{x}\hat{f}(x) . Milli et al.  ( Milli
et al., 2019 ) examines a specific scenario involving a one hidden-layer neural network function f : ℝ d → ℝ f:\mathbb{R}^{d}\rightarrow\mathbb{R} that uses ReLU activations, formulated as f ⁡ ( x ) = ∑ i = 1 h w i ​ max ⁡ ( A i T ​ x , 0 ) f(x)=\sum_{i=1}^{h}w_{i}\max(A_{i}^{T}x,0) .

 
 
 
 • 
 
 Threat model: 
For a DNN with parameters A ∈ ℝ h × d A\in\mathbb{R}^{h\times d} and w ∈ ℝ h w\in\mathbb{R}^{h} , where A i A_{i} represents the ith row of A, three assumptions are posited: (1) Each row A 1 , … , A h A_{1},...,A_{h} is a unit vector; (2) No pair of rows A i A_{i} and A j A_{j} are collinear for i ≠ j i\neq j , satisfying ⟨ A i , A j ⟩ ≤ 1 − c \langle A_{i},A_{j}\rangle\leq 1-c for some c 0 c 0 ; (3) The rows A 1 , … , A h A_{1},...,A_{h} are linearly independent. These assumptions are stated to be without loss of generality since they can be achieved by simple reparameterization of the network, such as scaling w w or A A , or by reducing the hidden layer dimension.

 

 • 
 
 General attacks: 
Under these assumptions, it is possible to learn the function with a sample complexity independent of the input dimension d d   ( Milli
et al., 2019 ) . Specifically, with a probability of 1 − δ 1-\delta , an algorithm can find a function f ^ \hat{f} such that f ^ = f \hat{f}=f . If the algorithm cannot find such a function, it will report the failure. Regardless of the outcome, the algorithm requires only O ⁡ ( h ​ log ⁡ h δ ) O\left(h\log\frac{h}{\delta}\right) queries to learn the function.

 

 • 
 
 Attacks on gradient-based explanations: 
The algorithm involves recovering a matrix Z Z and a sign vector s s   ( Milli
et al., 2019 ) . The matrix Z Z is composed of either w i ​ A i w_{i}A_{i} or − w i ​ A i -w_{i}A_{i} , with the signs encapsulated in s s . The function f f can then be reconstructed from Z Z and s s , utilizing the recovered structure to make predictions.
The approach relies on exploiting the gradient structure of f f to identify the hyperplanes that partition the input space and uses binary search to recover the necessary components of Z Z and s s .

 

 
 
 
 Model inversion attacks. 
Model inversion attacks aim to deduce original data from predictions, such as recreating a person’s face based on their predicted emotional state  ( Fredrikson
et al., 2015 ; Yang
et al., 2019 ; Zhang
et al., 2020a ) .
Initially, model inversion attacks showed limited success  ( Fredrikson
et al., 2015 ) , but advancements in deep learning, especially through the use of transposed Convolutional Neural Networks (CNNs), have significantly enhanced their effectiveness  ( Dosovitskiy and
Brox, 2016 ; He et al., 2019 ; Yang
et al., 2019 ) . Additional enhancements have been achieved by utilising auxiliary information, including access to the model’s internal workings and feature embeddings, or understanding the joint probability distribution between features and labels  ( Zhang
et al., 2020a ; Yeom
et al., 2018 ; He et al., 2019 ) . Especially the increasing demand for model explanations is likely to make these attacks more common  ( Zhao
et al., 2021b ) .

 
 
 
 • 
 
 Threat model: We consider a machine learning model f t f_{t} that processes confidential data x x from a set X p X_{p} (for instance, facial images). It employs these private inputs to generate a prediction y ^ t \hat{y}_{t} (such as identifying emotions).
An issue arises when an attacker gains access to the target prediction y ^ t \hat{y}_{t} and the explanation ϕ t \phi_{t} (due to reasons like a data breach, interception during transmission, or sharing on social media).
One scenario is to assume that the attacker only has the compromised data, an independent dataset x ∈ X a x\in X_{a} , and the ability to interact with the target model via black-box  ( Zhao
et al., 2021b ) . The attacker does not require additional privileged information, such as blurred versions of the images.
The objective of the attacker is to develop their own inversion model f a f_{a} to reconstruct the original image x x from the model’s outputs y ^ t , ϕ t \hat{y}_{t},\phi_{t} ). Such a reconstruction would allow them to predict sensitive information from the reconstructed image x ^ r \hat{x}_{r} , including the possibility of re-identifying the individual from the facial emotion recognition system  ( Hu
et al., 2022a ) .

 

 • 
 
 Attack on a single gradient-based explanation: 
To invert the target model M t M_{t} , a Transposed Convolutional Neural Network (TCNN)  ( Dumoulin and
Visin, 2016 ) is devised to reconstruct a two-dimensional image x r x_{r} from the one-dimensional prediction vector y t y_{t} provided by M t M_{t} . The TCNN minimises the mean squared error (MSE) loss to approximate the original image. This TCNN incorporates various input forms, such as saliency maps and 2D explanations  ( Selvaraju et al., 2017 ; Simonyan
et al., 2013 ) , enhancing the reconstruction of x r x_{r} . Inputs can be processed by flattening the 2D explanations into a 1D vector and concatenating with the prediction vector, or by using a CNN to convert 2D patterns into a 1D feature embedding, following the approach used in CNN encoder-decoder networks and super-resolution techniques  ( ur Rehman et al., 2019 ; Zhang
et al., 2020a ) . A U-Net architecture is employed to improve the reconstruction fidelity  ( Zhang
et al., 2018 ) .
A hybrid model that combines flattened explanations with the U-Net structure is introduced in  ( Zhao
et al., 2021b ) .
The training objective for these models is defined by the image reconstruction loss function:

 

 
 (11) | 
 | 
 L r = ∑ x ( M i a ​ ( M t ​ ( x ) ) − x ) 2 L_{r}=\sum_{x}(M^{a}_{i}(M_{t}(x))-x)^{2} | 
 | 
 

 where x x represents the original image, M t ​ ( x ) = y t M_{t}(x)=y_{t} denotes the prediction from the target model, and M i a ​ ( M t ​ ( x ) ) = x r M^{a}_{i}(M_{t}(x))=x_{r} is the reconstructed image output. Zhao et al.  ( Zhao
et al., 2021b ) conducts experiments on how different explanation methods, including gradients  ( Simonyan
et al., 2013 ) , CAM  ( Zhou et al., 2016 ) , LRP  ( Bach et al., 2015 ) , and blurred versions of the input images, affect the inversion model’s ability to capture information.

 

 • 
 
 Attack on multiple gradient-based explanations: 
While many explanations clarify the reasons a model predicts a certain class within a set C C , it is equally crucial to elucidate why it did not predict a different class c ′ ≠ c c^{\prime}\neq c , offering contrastive insights  ( Miller, 2019 ) . To facilitate this, certain techniques like Grad-CAM can generate explanations that are specific to a class based on the user’s query  ( Selvaraju et al., 2017 ) . Nevertheless, this approach increases the risk to privacy as it provides additional information. Zhao et al.  ( Zhao
et al., 2021b ) makes use of these Alternative CAMs ( Σ \Sigma -CAM) by merging explanations across all classes in | C | |C| into a three-dimensional tensor, and they train their inversion models on this tensor rather than on a two-dimensional matrix representing a single explanation.

 

 • 
 
 Attack on surrogate explanations: 
Interpretable surrogates could be harnessed for inversion attacks, even for models that do not provide target explanations. Zhao et al.  ( Zhao
et al., 2021b ) proposes an attack that predicts the target explanation and exploits that explanation to invert the original target data. Initially, an explainable surrogate target model f a f_{a} is trained using the attacker’s dataset to generate a surrogate explanation ϕ ~ \widetilde{\phi} .
However, ϕ t ~ \widetilde{\phi_{t}} is only accessible during the training phase and not during prediction. Consequently, an explanation inversion model f e f_{e} is trained to reconstruct ϕ t ~ \widetilde{\phi_{t}} as ϕ r ^ \widehat{\phi_{r}} based on the target prediction y t ^ \widehat{y_{t}} . The proposed loss function for minimising the surrogate explanation error is:

 

 
 (12) | 
 | 
 L ϕ = ∑ x ( f e ​ ( f t ​ ( x ) ) − ϕ ⁡ ( f t ​ ( x ) ) ) 2 L_{\phi}=\sum_{x}\left(f_{e}(f_{t}(x))-\phi(f_{t}(x))\right)^{2} | 
 | 
 

 where ϕ ⁡ ( f ) \phi(f) denotes the explanation of the model f f , f t ​ ( x ) = y t {f_{t}(x)}={y_{t}} represents the surrogate target prediction, ϕ ​ ( f t ​ ( x ) ) = ϕ t ~ \phi(f_{t}(x))=\widetilde{\phi_{t}} is the surrogate explanation, and f e ​ ( f t ​ ( x ) ) = ϕ r ^ f_{e}(f_{t}(x))=\widehat{\phi_{r}} is the reconstructed surrogate explanation. This reconstructed explanation is available at prediction time. Finally, ϕ r ^ \widehat{\phi_{r}} is fed into the image inversion model ϕ i \phi_{i} to finalize the model inversion attack.
Given that ϕ r ^ \widehat{\phi_{r}} is formatted similarly to ϕ t ~ \widetilde{\phi_{t}} , any explanation methods can be applied.

 

 • 
 
 Attacks on confidence scores: 
Fredrikson et al.  ( Fredrikson
et al., 2015 ) develops a model inversion attack
by using a maximum a posteriori (MAP) estimator to compute f ⁡ ( x 1 , … , x d ) f(x_{1},\ldots,x_{d}) for all possible values of the sensitive feature x 1 x_{1} , while exploiting confidence information from model predictions.
Fredrikson et al.  ( Fredrikson
et al., 2015 ) addresses the challenge of inverting high-dimensional features like facial recognition, where the inversion task becomes an optimization problem solved by gradient descent.

 

 
 
 
 Figure 9 . Attribute/feature inference attacks. 
 
 
 

### 3.4. Attribute/Feature Inference Attacks

 
 Attribute inference attacks, aka feature inference attacks, are designed to deduce specific attributes, such as gender, from individual data records by using accessible data like model predictions or explanations  ( Song and
Shmatikov, 2020 ; Yeom
et al., 2018 ) (see  Fig. 9 ).
These types of attacks are distinct from property inference attacks, which seek to ascertain broader dataset characteristics, like the training data’s gender ratio  ( Ganju
et al., 2018 ; Melis et al., 2019 ; Zhang
et al., 2021 ) .

 
 
 Duddu et al.  ( Duddu and Boutet, 2022 ) investigates a scenario where a machine learning model, f t ​ a ​ r ​ g ​ e ​ t f_{target} , is cloud-deployed within an MLaaS framework (e.g. Google Cloud, Microsoft Azure), capable of providing predictions and required explanations for any given input.
Users can submit a private sample x = { x i } i = 1 n x=\{x_{i}\}^{n}_{i=1} to the service provider and receive a prediction vector y ^ = { y ^ i } i = 1 c \hat{y}=\{\hat{y}_{i}\}^{c}_{i=1} , along with an explanation vector ϕ = { ϕ i } i = 1 n \phi=\{\phi_{i}\}^{n}_{i=1} that pertains to a specific class. Although the service provider has the capacity to return multiple explanation vectors corresponding to different classes  ( Chen
et al., 2018b ) , for practicality and without loss of generality, most works focuses on the use of one explanation vector for a specific class  ( Luo
et al., 2022 ) .

 
 
 Threat models on feature-based explanations. 
Duddu et al.  ( Duddu and Boutet, 2022 ) considers two threat models (TM). (1) TM1 (with s s in D D ): Here, the sensitive feature s s is included in both the training dataset D D and the input. 𝒜 ​ d ​ v \mathcal{A}dv has access to the predictions f t ​ a ​ r ​ g ​ e ​ t ​ ( x ∪ s ) f_{target}(x\cup s) and explanations ϕ ⁡ ( x ∪ s ) \phi(x\cup s) , but not the ability to pass inputs to the model. The adversary’s goal is to train an attack model f a ​ d ​ v f_{adv} that maps the explanations ϕ ⁡ ( x ) \phi(x) to s s on D a ​ u ​ x D_{aux} , an auxiliary dataset known to 𝒜 ​ d ​ v \mathcal{A}dv .
(2) TM2 (without s s in D D ): In this scenario, s s is not included in the dataset D D or the input x x .
Unlike TM1, 𝒜 ​ d ​ v \mathcal{A}dv can pass inputs x x to the model and has blackbox access to f t ​ a ​ r ​ g ​ e ​ t f_{target} and ϕ ⁡ ( x ) \phi(x) , making this a more practical threat where s s is censored for privacy. 𝒜 ​ d ​ v \mathcal{A}dv ’s goal remains the same, to infer s s by training f a ​ d ​ v f_{adv} on D a ​ u ​ x D_{aux} .
For both models, the adversary has an additional auxiliary dataset D a ​ u ​ x D_{aux} that contains data records with non-sensitive and sensitive attributes along with their corresponding labels.

 
 
 Threat models on Shapley values. 
Unlike previous assumptions  ( Salem et al., 2018 ; Shokri
et al., 2021 ) that adversaries have an auxiliary dataset with a distribution similar to the target sample, Luo et al.  ( Luo
et al., 2022 ) explores two relaxed scenarios. The first adversary has access to an explanation vector, an auxiliary dataset, and a black-box prediction model, aiming to reconstruct the target sample. The second adversary operates under more practical constraints with only black-box access to the machine learning services and the explanation vector, without any background knowledge of the target sample.

 
 
 Attacks on feature-based explanations. 
Duddu et al.  ( Duddu and Boutet, 2022 ) develops an attribute inference attack based on thresholding. The attack model, f a ​ d ​ v f_{adv} , uses model explanations to infer sensitive attributes and chooses the threshold t ∗ t^{*} that maximizes the F1-Score. This calibration step deviates from using the typical default threshold of 0.5 to increase the precision and recall of the attack, particularly when there is a moderate to large class imbalance of the sensitive attribute s s .
Duddu et al.  ( Duddu and Boutet, 2022 ) also shows low Pearson correlation coefficients between the sensitive attribute s s and other entities like y y , x x , and ϕ ⁡ ( x ) \phi(x) across different datasets and explanation methods, suggesting little to no direct correlation between the sensitive attribute and the model’s predictions or explanations, challenging the notion that the attack is merely exploiting these correlations.

 
 
 Attacks on Shapley values. 
Luo et al.  ( Luo
et al., 2022 ) proposes an attack where an adversary, with access to a black-box model f f , attempts to infer private input features from Shapley value explanations. To simplify the computation of Shapley values, the adversary uses a reference sample x 0 x^{0} and a linear transformation function h h . They aim to reduce mutual information between the input x i x_{i} and the Shapley value s i s_{i} to zero, meaning the adversary cannot gain any information about x i x_{i} from s i s_{i} . Luo et al.  ( Luo
et al., 2022 ) assumes that the Shapley values follow a Gaussian distribution, and thus the probability P ⁡ ( s i ) P(s_{i}) is modelled as a Gaussian function.
To ensure that the mapping from the auxiliary input data X a ​ u ​ x X_{aux} to the Shapley values S a ​ u ​ x S_{aux} is bijective, Luo et al.  ( Luo
et al., 2022 ) presents a theorem requiring X a ​ u ​ x X_{aux} to be finite. The adversary can then use a hypothesis ψ \psi to map Shapley values back to the auxiliary input data. To execute the attack, the adversary collects the Shapley values for all x a ​ u ​ x ∈ X a ​ u ​ x x_{aux}\in X_{aux} , sends prediction queries to the MLaaS platform, and obtains explanations S a ​ u ​ x S_{aux} . They then train a regression model on X a ​ u ​ x X_{aux} to learn the mapping ψ \psi from Shapley values S a ​ u ​ x S_{aux} to X a ​ u ​ x X_{aux} .

 
 
 Another scenario is where an adversary lacks an auxiliary dataset to carry out a feature inference attack  ( Luo
et al., 2022 ) . Without knowledge of the target’s data distribution, it becomes challenging to learn an attack model by observing Shapley values.
To mitigate these challenges, the adversary can use the linear correlation between feature values and Shapley values for important features. By drawing samples independently and using a Generalized Additive Model (GAM) for approximation, the adversary can restore features from Shapley values.
Luo et al.  ( Luo
et al., 2022 ) notes that while their attacks work well with Shapley values, other explanation methods like LIME and DeepLIFT may not be suitable due to their heuristic-based, unstable mappings between features and explanations.

 
 
 

### 3.5. Model Extraction Attacks

 
 There is an increasing concern of model extraction attacks in the context of Machine Learning as a Service (MLaaS)  ( Tramèr et al., 2016 ) , where attackers steal ML models by using surrogate datasets to make queries through the MLaaS API, and then train replica models with the obtained predictions. The goal is to create a functionally equivalent version with identical predictions (see  Fig. 10 ). The difference between a model extraction attack and a model reconstruction attack is that the former does not need to know the model architecture.

 
 
 Figure 10 . Model extraction attacks. 
 
 
 Research on model extraction attacks targeting explainable AI systems is emerging  ( Mi
et al., 2024 ) . Milli et al.  ( Milli
et al., 2019 ) develops a method that leverages the discrepancy in gradient-based explanations between an original AI model and its clone, demonstrating enhanced attack efficiency. Additionally, Ulrich et al.  ( Aïvodji
et al., 2020 ) designs an attack utilising counterfactual explanations to train a cloned model with greater effectiveness. Miura et al.  ( Miura
et al., 2021 ) designs a data-free attack that does not require surrogate datasets in advance.

 
 
 Threat models. 
An adversary duplicates a trained model, referred to as the victim model f : X → Y f:X\to Y , by utilising its predictions to create a similar clone model f ^ : X → Y \hat{f}:X\to Y . The adversary’s goal is to replicate the victim model’s accuracy using only the output predictions. On the one hand, typical model extraction attacks  ( Milli
et al., 2019 ) involve the adversary collecting input data x ∈ X x\in X , querying the victim model to obtain predictions, and using the pairs ( x i , f ⁡ ( x i ) ) (x_{i},f(x_{i})) to compile a dataset for training the clone model.
In some scenarios, an adversary requires query access to the victim model but does not necessarily need the training data’s ground-truth labels  ( Quan et al., 2022 ) . The attack relies on knowing the architecture of the victim model but not its parameter values. The attacker aims to produce a model that performs identically on the same test dataset, although the adversary’s extracted model may not have been trained on the same data or in the same manner as the victim model.

 
 
 On the other hand, data-free model extraction attacks  ( Miura
et al., 2021 ) eliminates the need for input data collection, in which an adversary employs a generative DNN G : ℝ r → X G:\mathbb{R}^{r}\to X to convert Gaussian distribution noise into synthetic input data. The adversary then uses this data to query the victim model and gather training pairs ( x , f ⁡ ( x ) ) (x,f(x)) , which are used to train the clone model to emulate the victim model f f . The generative model is designed to create data that, when predicted by the clone model, is different from the victim model’s output, intending to maximize the clone model’s loss function and improve parameter updates. Although the generative model G G does not learn the actual distribution of the input data space X X , it is optimised to produce data that facilitates the clone model’s training process.

 
 
 In the case of counterfactual explanations, the explanation API provides for each data point x i x_{i} , a corresponding counterfactual explanation c ⁡ ( x i ) c(x_{i}) , accompanied by the predicted outcome y ^ i \hat{y}_{i} . When seeking a collection of diverse counterfactuals, the API will yield a collection C ⁡ ( x i ) C(x_{i}) comprising multiple counterfactual instances, rather than just a single example.

 
 
 Attacks on gradient-based explanations. 
In the data-free model extraction  ( Miura
et al., 2021 ) , an attacker crafts a surrogate model, denoted as f ^ : X → Y \hat{f}:X\rightarrow Y , alongside a generative model G : ℝ r → X G:\mathbb{R}^{r}\rightarrow X , responsible for creating synthetic data inputs. An iterative process is repeated between two steps. The first step generates N G N_{G} input samples and queries the target model to refine the generative model based on both predictions and explanations, utilising these explanations to compute the gradient ∇ θ G ℒ \nabla_{\theta_{G}}\mathcal{L} . The second routine produces N C N_{C} input samples for querying the target model and uses the resulting predictions to train the surrogate model. The process stops when the number of queries ( N G + N C N_{G}+N_{C} ) aligns with the allocated query budget Q Q . This strategy enables the attacker to leverage the gradient ∇ G ​ ( x ) = ∇ x f ​ ( x ) \nabla G(x)=\nabla_{x}f(x) for the training of this generative model.

 
 
 Adversarial attacks for model extraction without data rely on alternately calculating the gradients of an objective function with the parameters of both a cloned model and a generative model. Training the clone requires calculating the gradient ∇ θ f ℒ \nabla_{\theta_{f}}\mathcal{L} , achievable via back-propagation by the adversary. However, current methods do not provide the adversary with access to ∇ θ G ℒ \nabla_{\theta_{G}}\mathcal{L} for training the generative model. According to  ( Miura
et al., 2021 ) , it suffices to find ∇ x ℒ ​ ( x ) \nabla_{x}\mathcal{L}(x) as it leads to ∇ θ G ℒ = − ∇ θ G G ( z ) ⋅ ∇ x ℒ ( x ) \nabla_{\theta_{G}}\mathcal{L}=-\nabla_{\theta_{G}}G(z)\cdot\nabla_{x}\mathcal{L}(x) 
Unlike previous methods that only provided terms other than ∇ x f ​ ( x ) \nabla_{x}f(x) , the adversary now gains explanations through the standard Gradient G ​ ( x ) = ∇ x f ​ ( x ) G(x)=\nabla_{x}f(x) , enabling the computation of ∇ x ℒ ​ ( x ) \nabla_{x}\mathcal{L}(x) precisely. The adversary can employ almost any differentiable loss function for training the generative model.

 
 
 Quan et al.  ( Quan et al., 2022 ) proposes another explanation-matching attack  ( Milli
et al., 2019 ) , focusing on replicating both the predictions and explanations of the original, or victim, model. The adversary’s model minimises two losses: the prediction loss (the difference in predictions between the two models) and the explanation matching loss (the difference in their explanations). The overall loss being minimised is a weighted combination of these two losses. Additionally, the method includes the use of LIME to ensure the interpretability of predictions matches that of the victim model.

 
 
 Attacks on counterfactual explanations. 
Kuppa et al.  ( Kuppa and Le-Khac, 2021 ) considers two main factors: (a) The auxiliary dataset D a ​ u ​ x D_{aux} should approximate the training set of f f . This can be challenging if D a ​ u ​ x D_{aux} does not naturally follow the training distribution, but counterfactual explanations can provide samples from various classes that may bridge this gap. An attacker can iteratively query and obtain diverse class samples to better reflect the training set distributions. (b) Knowing the architecture of f f can significantly enhance the fidelity of the extracted model. However, in realistic scenarios, attackers often lack this information, complicating the attack.
To circumvent this obstacle, once data samples that mirror the training set are collected, knowledge distillation techniques are employed. This involves transferring insights from f f to a surrogate model g g . The knowledge transfer is quantified using a distillation loss, given by L D ​ i ​ s ​ t ​ i ​ l ​ l ​ ( f , g ) = L K ​ L ​ ( P f ​ ( x ) , P g ​ ( x ) ) L_{Distill}(f,g)=L_{KL}(P_{f}(x),P_{g}(x)) , where L K ​ L L_{KL} represents the Kullback-Leibler divergence loss. In this setup, the attacker leverages publicly available data and queries f f , then applies the distillation loss to train g g , thereby extracting the functionality of f f .

 
 
 Aivodji et al.  ( Aïvodji
et al., 2020 ) proposes a model extraction attack  ( Jagielski et al., 2020 ) by
compiling an attack set and training a surrogate model on the collected data from counterfactual samples.
Counterfactual explanations typically change features with larger importance values to achieve the desired prediction, thus revealing the model’s sensitive areas. However, this approach has limitations, such as the decision boundary shift issue caused by using distant queries from the decision boundary as training samples  ( Aïvodji
et al., 2020 ) . This leads to an unstable substitute model and requires more queries to resolve, thus increasing the attack cost. Wang et al.  ( Wang
et al., 2022 ) proposes a method called DualCF to mitigate this issue by using pairs of counterfactuals (CF) and their corresponding explanations (CCF) from the opposite class as training data. This helps to balance the substitute model’s decision boundary and improve extraction efficiency.
DualCF for a Linear Model is also discussed, illustrating that for binary linear models, it’s possible to extract a substitute model with 100% agreement using CF and CCF pairs.
While promising for linear models, extending this approach to nonlinear and complex models remains a challenge, and the effectiveness of DualCF in those scenarios is yet to be thoroughly evaluated  ( Tramèr et al., 2016 ) .

 
 
 
 

## 4. Causes of Privacy Leaks

 
 Research into the causes that lead to privacy leakage through model explanations has started to emerge in the past few years  ( Naretto
et al., 2022 ; Shokri
et al., 2021 ; Artelt et al., 2021 ; Chang and Shokri, 2021 ; Pawelczyk
et al., 2023 ; Quan et al., 2022 ) . Certain types of explanations are prone to divulging data, often due to their inherent structure. For instance, case-based explanations, which utilise actual data points from the training set, can inadvertently reveal sensitive information  ( Montenegro et al., 2022 ; Shokri
et al., 2020 ) . Other explanations, such as surrogate models (e.g. SVM, linear classifiers) are relative easy to leak their parameters by querying enough input/output data pairs  ( Naretto
et al., 2022 ; Quan et al., 2022 ; Ferry et al., 2023a ) .

 
 

### 4.1. Privacy Leaks in Counterfactual Explanations

 
 While counterfactual explanations aim to clarify AI decisions, they may inadvertently compromise privacy  ( Sokol and Flach, 2019 ) . These explanations can give adversaries clues to manipulate the system, as seen in instances where absence of a feature (like a savings account) leads to a better outcome than a suboptimal presence  ( Sokol and Flach, 2019 ) .
They provide insights into decision boundaries, potentially revealing model specifics and training data, such as feature splits in logical models, training points in k-nearest neighbors, or support vectors in SVMs.
Moreover, the existence of multiple and varying-length counterfactuals for a single data point could increase the ease of model theft, with longer, more complex counterfactuals potentially disclosing substantial model information with just one explanation.

 
 
 Vo et al.  ( Vo et al., 2023 ) outlines essential privacy concepts relevant to public datasets. Identifiers are personal attributes capable of uniquely distinguishing an individual, such as names or government-issued numbers. Quasi-identifiers, while not individually unique, can collectively re-identify individuals; a mix of gender, birthdate, and ZIP code, for instance, can pinpoint 87% of American residents  ( Sweeney, 2000 ) . Sensitive attributes cover confidential information like salaries or medical records that need safeguarding to prevent personal or emotional harm.
To protect against re-identification risks, public datasets need to undergo anonymisation by removing direct identifiers, though vulnerability remains due to quasi-identifiers.

 
 
 Example 0 . 
 
 In the given scenario from the FICO explainable ML dataset  ( Sokol and Flach, 2019 ) , the outcome of the credit evaluation could have shifted from negative to positive if one of the following conditions were met: 

 
 • 
 
 # installment trades is less than 3 instead of 3 

 

 • 
 
 # revolving trades is less than 3 instead of 5 

 

 • 
 
 # trades with 60 days overdue and marked as derogatory in public record is equal to 0 instead of 2 . 

 

 • 
 
 # loans within 1 year is less or equal to 2 instead of 5 . 

 

 
 Here, user privacy is violated as the exact values of the above sensitive attributes are revealed  ( Sokol and Flach, 2019 ) . 

 
 
 
 Diverse counterfactuals equip users with a range of actionable insights to potentially alter their outcomes favorably  ( Mothilal
et al., 2020 ; Nguyen
et al., 2023a ) . However, this also increases privacy risks as it may give away additional details that could be exploited for more potent attacks  ( Aïvodji
et al., 2020 ) .
Artelt et al.  ( Artelt et al., 2021 ) identifies a key problem with counterfactual explanations: their instability to minor input variations can lead to significantly different outcomes for similar cases. Addressing this, the authors propose studying the robustness of counterfactual explanations and suggest using plausible rather than closest counterfactuals to enhance stability  ( Artelt and Hammer, 2020 ) .

 
 
 

### 4.2. Causes of Membership Inference Attacks

 
 Membership inference attacks (MIAs) aim to predict whether a data point is in the training set or not  ( Shokri
et al., 2020 ) . The trade-off between explainability and privacy has been investigated and evaluated using membership inference attacks in ( Naretto
et al., 2022 ; Shokri
et al., 2021 ; Chang and Shokri, 2021 ; Pawelczyk
et al., 2023 ) .

 
 
 Global explainers. 
Naretto et al.  ( Naretto
et al., 2022 ) demonstrates that interpretable tree-based global explainers can increase the risk of privacy leakage. To explain f f , an interpretable global surrogate classifier g g is required to be trained to imitate the behavior of f f , i. e., g ⁡ ( X ) = f ⁡ ( X ) g(X)=f(X) .
To compare the privacy exposure risk caused by f f and g g , two attack models are trained: one is learnt by querying f f , and the other queries g g . It was found that the global explainer is more vulnerable to the membership inference attack model than the classifier  ( Naretto
et al., 2022 ) , resulting in more privacy exposure.

 
 
 Feature-based explanations. MIAs were also evaluated on feature-based explanations, including back-propagation and perturbation ( Shokri
et al., 2021 ) .
Backpropagation-based explanations were found to result in privacy leakage, which may be caused by high variances of explanations. A high variance of an explanation indicates that the point is close to the decision boundary and has an uncertain prediction, which is helpful for an adversary. Compared to backpropagation-based explanations, perturbation-based explanations are more robust to membership inference attacks. This might be because the query points are not used to train the model  ( Shokri
et al., 2021 ) .

 
 
 Repeated interaction. 
Kumari et al.  ( Kumari
et al., 2024 ) focus on repeated interactions. The author introduce attacks using explanation variance to infer data membership, modeled through a continuous-time stochastic signaling game. The study proves an optimal attack threshold exists, analyzes equilibrium conditions, and uses simulations to assess attack effectiveness in dynamic settings.

 
 
 Fairness. 
Apart from explanations, pursuing fairness during model training can also increase risks of privacy exposure  ( Chang and Shokri, 2021 ) . When processing imbalanced data, fairness constraints require the model to memorize the training data in the smaller groups rather than learning a general pattern  ( Chang and Shokri, 2021 ) . Such a way makes it easier for membership inference attacks to attack the model. Especially, when membership inference attacks are designed specifically for each group, they showed higher attack accuracy than that of a common membership inference attack for all groups  ( Chang and Shokri, 2021 ) . Another study  ( Shokri
et al., 2020 ) also reports small groups in record-based explanations are more vulnerable to membership inference attacks than majority groups.

 
 
 Influence of Input Dimension. 
Shokri et al.  ( Shokri
et al., 2021 ) evaluates how the input dimension influences the privacy risks of gradient-based explanations.
Their experiments revealed that
as the number of features grows (between 10 3 10^{3} and 10 4 10^{4} ), a correlation between gradient norms and training membership appears, indicating vulnerability to membership inference attacks.
However, this effect is moderated by the number of classes and is also dependent on model behavior, as overfitting can occur with too many features. While increasing the number of classes generally increases learning problem complexity, the actual impact on the correlation between gradient norms and membership depends on the specific range of features and that the interval and amount of correlation vary.

 
 
 Influence of Overfitting. 
Yeom et al.  ( Yeom
et al., 2018 ) demonstrates that overfitting has a notable impact on the success of membership inference attacks. Shokri et al.  ( Shokri
et al., 2021 ) conducts tests varying the number of training iterations to achieve different levels of accuracy, in order to assess the effects of overfitting. Consistent with prior research on loss-based attacks, they found that their threshold-based attacks, which leverage explanations, are more effective when targeting overfitted models.

 
 
 

### 4.3. Causes of Reconstruction Attacks

 
 Reconstruction attacks target on reconstructing the partial or complete training data.
Ferry et al.  ( Ferry et al., 2023a ) shows that post-hoc explanations can disproportionately impact individual privacy, exacerbating risks for minority groups. This trend towards reduced privacy for minorities is also reflected in interpretability, as identified by Shokri et al  ( Shokri
et al., 2021 ; Shokri
et al., 2020 ; Shokri
et al., 2019 ) . They discovered that the likelihood of discerning whether an individual’s data was used in a model’s training set from post-hoc explanations is higher for outliers and certain minority groups that the model finds difficult to generalize. This increased risk is attributed to these groups being more frequently included in the generated explanations. Consequently, tools designed for interpretability could inadvertently lead to greater information leakage about these already vulnerable groups.

 
 
 Interpretable models enhance transparency but can inadvertently disclose information about their training data. Gambs et al.  ( Gambs
et al., 2012 ) uses such data leakage to probabilistically reconstruct a decision tree’s training set. The uncertainty within this reconstruction can be measured to determine how much information the model leaks.

 
 
 Ferry et al.  ( Ferry et al., 2023a ; Ferry, 2023 ) examines how optimal and heuristic decision trees and rule lists reveal information about their training data.
The study finds that optimal models tend to leak less information than greedily-built ones for a given level of accuracy. It also notes significant variance in how much information individual training examples contribute to the overall entropy reduction, with some examples inherently leaking more information based on their position within the model’s structure.

 
 
 

### 4.4. Causes of Property Inference Attacks

 
 Regularisation techniques like dropout and ensemble learning have been shown to prevent models from memorizing private inputs, potentially reducing the risk of information leakage  ( Luo
et al., 2021 ; Melis et al., 2019 ; Liu
et al., 2022a ) . Despite previous findings, Luo et al.  ( Luo
et al., 2022 ) reveals that incorporating dropout in neural networks at varying rates (0.2, 0.5, 0.8) actually enhances the accuracy of certain attacks. This counterintuitive result is attributed to dropout preventing overfitting by smoothing the decision boundaries, which inadvertently benefits the attack. Nevertheless, a very high dropout rate (0.8) does decrease the success rates of one attack due to underfitting and increased randomness in the model, which disrupts the linearity between inputs and outputs.

 
 
 Case-based explanation methods, often used in sensitive fields like medical diagnosis, risk privacy breaches when they share detailed visual data with unauthorized viewers, such as medical students or family members  ( Montenegro et al., 2022 ) .
To mitigate this, anonymisation techniques must be applied to the images before they are shared, ensuring that the identity of individuals is not disclosed while still preserving the explanatory power and realism of the images.
The anonymisation process involves altering identity features in the latent vector to produce a privatized image, but there’s no guarantee that other latent features don’t inadvertently reveal identity, especially if facial embeddings capture significant identifiable information.

 
 
 

### 4.5. Causes of Model Extraction Attacks

 
 Quan et al.  ( Quan et al., 2022 ) explores how model extraction attacks can benefit from explanation methods, leading to adversarial gains with fewer queries. A particular finding is that while certain explanation methods, such as Gradient, Integrated Gradient, and SmoothGrad, can be exploited to enhance attack efficiency, others like Guided Backprop and GradCam may result in poorer performance due to biases in gradient estimation.

 
 
 While counterfactual explanations (CFs) do not reveal the entirety of a cloud model’s workings, their impact on security and privacy has been underestimated  ( Barocas
et al., 2020 ; Kasirzadeh and
Smart, 2021 ; Sokol and Flach, 2019 ) . Some research argues that CFs only unveil a minimal amount of information, showing a limited set of dependencies for an individual instance which might seem insufficient for model extraction  ( Hashemi and Fathi, 2020 ; Wachter
et al., 2017 ) . However, accumulating enough data through multiple queries can significantly facilitate the extraction process  ( Wang
et al., 2022 ) .
Aivodji et al.  ( Aïvodji
et al., 2020 ) pioneers the use of model extraction attacks on counterfactual explanations by treating these explanations near decision boundaries as supplementary training data.
Wang et al.  ( Wang
et al., 2022 ) also shows that adversaries can exploit CF explanations to extract a high-fidelity model by learning about the decision boundaries.

 
 
 

### 4.6. Causes of Explanation Linkage Attacks

 
 Vo et al.  ( Vo et al., 2023 ) reviews key concepts relevant to data privacy, specifically in the context of public datasets. Identifiers are attributes that can uniquely identify an individual, like names or government numbers. Quasi-identifiers, while not unique on their own, can combine to uniquely identify a person. Sensitive attributes are confidential data that, if disclosed, could harm an individual.
Public datasets are at risk of explanation linkage attacks, aka re-identification attacks, even after anonymisation if quasi-identifiers are present  ( Vo et al., 2023 ) .
Their experiments acknowledge that k-anonymity lower the risks but it may still allow private information to be inferred through homogeneity and background knowledge attacks.

 
 
 
 

## 5. Privacy-Preserving Explanations

 

### 5.1. Defences with Differential Privacy

 
 Differential privacy (DP) is a solid, mathematically based privacy standard that defines privacy loss using a quantifiable metric  ( Liu
et al., 2024c ) . It does so through mechanisms that guarantee the aggregated data output will obscure the involvement of any individual record in the dataset, as established by Dwork et al.  ( Dwork
et al., 2014 ) . Differential privacy is usually formalized as follows  ( Huang
et al., 2023 ) . A randomized mechanism M M with domain D D and range R R achieves ε \varepsilon -differential privacy ( ε \varepsilon -DP) if, for all adjacent datasets d , d ′ d,d^{\prime} differing by one row, and for any output set S ⊆ R S\subseteq R , the following inequality holds:

 

 
 (13) | 
 | 
 Pr [ Q ( d ) ∈ S ] ≤ e ε ⋅ Pr [ Q ( d ′ ) ∈ S ] . \text{Pr}[Q(d)\in S]\leq e^{\varepsilon}\cdot\text{Pr}[Q(d^{\prime})\in S]. | 
 | 
 

 Here, ε \varepsilon is the privacy loss parameter, where smaller values correspond to stronger privacy.

 
 
 Figure 11 . Differential Privacy. 
 
 
 The Laplace Mechanism of differential privacy is useful for queries on numerical data  ( Huang
et al., 2023 ) . As shown in Fig. 11 , the mechanism adds noise to the sensitive query’s output according to the Laplace distribution. Specifically, for a sensitive query function Q ⁡ ( d ) Q(d) , the ε \varepsilon -DP Laplace Mechanism Q L ​ a ​ p Q_{Lap} is given by Q L ​ a ​ p ​ ( d ) = Q ⁡ ( d ) + Laplace ​ ( G ​ S Q / ε ) Q_{Lap}(d)=Q(d)+\text{Laplace}(GS_{Q}/\varepsilon) , where Laplace ​ ( G ​ S Q / ε ) \text{Laplace}(GS_{Q}/\varepsilon) represents a random variable from the Laplace distribution with a scale dependent on the global sensitivity G ​ S Q GS_{Q} divided by ε \varepsilon . Global sensitivity G ​ S Q GS_{Q} is the maximum norm-1 difference of Q Q across all pairs of adjacent datasets d , d ′ d,d^{\prime} . Lastly, Dwork et al.  ( Dwork
et al., 2014 ) have demonstrated a post-processing property of differential privacy:
If Q Q is ε \varepsilon -DP and G G is any arbitrary deterministic mapping, then the composite function G ∘ Q G\circ Q is also ε \varepsilon -DP  ( Huang
et al., 2023 ) .

 
 

#### 5.1.1. Differentially Private Feature-based Explanations

 
 An explanation ϕ ⁡ ( ⋅ ) \phi(\cdot) is ( ϵ , δ ) (\epsilon,\delta) -differentially private if the probability of any sequence of explanations does not change significantly with the addition or removal of a single data point in the training set  ( Patel
et al., 2022 ) . For a sequence of queries z → 1 , … , z → k \vec{z}_{1},...,\vec{z}_{k} , and any two neighboring training sets 𝒟 \mathcal{D} and 𝒟 ′ \mathcal{D}^{\prime} , and subsets S 1 , … , S k ⊆ ℝ n S_{1},...,S_{k}\subseteq\mathbb{R}^{n} , we have:

 

 
 (14) | 
 | 
 P ​ r ​ [ ϕ 1 ∈ S 1 , … , ϕ k ∈ S k ] ≤ e ϵ ⋅ P ​ r ​ [ ϕ ′ 1 ∈ S 1 , … , ϕ ′ k ∈ S k ] + δ Pr[\phi^{1}\in S_{1},...,\phi^{k}\in S_{k}]\leq e^{\epsilon}\cdot Pr[\phi^{\prime 1}\in S_{1},...,\phi^{\prime k}\in S_{k}]+\delta | 
 | 
 

 where ϕ i = ϕ ⁡ ( z → i , f 𝒳 ​ ( x → ) ) \phi^{i}=\phi(\vec{z}_{i},f_{\mathcal{X}}(\vec{x})) and ϕ ′ i = ϕ ⁡ ( z → i , f 𝒟 ′ ​ ( x → ) ) \phi^{\prime i}=\phi(\vec{z}_{i},f_{\mathcal{D}^{\prime}}(\vec{x})) for all i i . The privacy for the explanation dataset 𝒳 \mathcal{X} can follow a similar guarantee.
Despite these measures, post-hoc explanation algorithms, which are applied after the model has been trained, cannot fully prevent membership inference attacks, since they do not control the training process or parameters  ( Patel
et al., 2022 ) .

 
 
 Single explanation algorithm. 
Patel et al.  ( Patel
et al., 2022 ) focuses on creating differentially private feature-based model explanations, where ϕ ⁡ ( z → ) \phi(\vec{z}) is a vector in ℝ n \mathbb{R}^{n} that quantifies the impact of each feature on the model’s predicted label f 𝒟 ​ ( z → ) f_{\mathcal{D}}(\vec{z}) . The aim is to find a local explanation function ϕ \phi , centred at a point of interest z → \vec{z} , that minimises the local empirical model error over an explanation dataset 𝒳 \mathcal{X} . The local empirical loss of ϕ \phi over 𝒳 \mathcal{X} is given by:

 

 
 (15) | 
 | 
 ℒ ⁡ ( ϕ , z → , f 𝒳 ) = 1 | 𝒳 | ​ ∑ x → ∈ 𝒳 α ⁡ ( ‖ x → − z → ‖ ) ​ ( x → − z → ) T ​ ( x → − z → ) − f 𝒳 ​ ( x → ) 2 , \mathcal{L}(\phi,\vec{z},f_{\mathcal{X}})=\frac{1}{|\mathcal{X}|}\sum_{\vec{x}\in\mathcal{X}}\alpha(\|\vec{x}-\vec{z}\|)(\vec{x}-\vec{z})^{T}(\vec{x}-\vec{z})-f_{\mathcal{X}}(\vec{x})^{2}, | 
 | 
 

 where α \alpha is a weight function that decreases with distance from z → \vec{z} .
The optimal model explanation is the one that minimises this loss:

 

 
 (16) | 
 | 
 ϕ ∗ ​ ( z → , f 𝒳 ) = arg ⁡ min ϕ ∈ 𝒞 ⁡ ℒ ⁡ ( ϕ , z → , f 𝒳 ) . \phi^{*}(\vec{z},f_{\mathcal{X}})=\arg\min_{\phi\in\mathcal{C}}\mathcal{L}(\phi,\vec{z},f_{\mathcal{X}}). | 
 | 
 

 
 
 To ensure differential privacy, Patel et al.  ( Patel
et al., 2022 ) introduces a Differentially Private Gradient Descent (DPGD) algorithm, which utilises the Gaussian mechanism to protect the explanation dataset 𝒳 \mathcal{X} . The privacy of the explanation dataset is protected by computing a private version of the gradient descent updates. The DPGD-Explain procedure iteratively updates ϕ \phi using the gradient of the loss function perturbed by Gaussian noise, aiming to find the minimum of ϕ \phi within a certain bound:

 

 
 (17) | 
 | 
 ϕ ( t + 1 ) ← arg ⁡ min ϕ ∈ 𝒞 2 , 1 ​ ‖ ϕ − ζ ( t ) ‖ , \phi^{(t+1)}\leftarrow\arg\min_{\phi\in\mathcal{C}_{2},1}\|\phi-\zeta^{(t)}\|, | 
 | 
 

 where ζ ( t ) \zeta^{(t)} is the perturbed gradient at iteration t t .
Patel et al.  ( Patel
et al., 2022 ) provides conditions for bounded sensitivity for the gradient ∇ ℒ ​ ( ⋅ ) \nabla\mathcal{L}(\cdot) , which is crucial for the differential privacy guarantee. The authors specify a family of weight functions α ⁡ ( ⋅ ) \alpha(\cdot) that ensure the gradient sensitivity is bounded, which is a requisite for the differential privacy mechanisms employed. The authors also define a family of desirable weight functions ℱ ⁡ ( 𝒞 , z → ) \mathcal{F}(\mathcal{C},\vec{z}) as those that are non-increasing and satisfy:

 

 
 (18) | 
 | 
 ∀ x → ∈ ℝ n , α ⁡ ( ‖ x → − z → ‖ ) ≤ c 2 ​ ‖ x → − z → ‖ 2 ​ ( ‖ x → − z → ‖ 2 + 1 ) . \forall\vec{x}\in\mathbb{R}^{n},\alpha(\|\vec{x}-\vec{z}\|)\leq\frac{c}{2\|\vec{x}-\vec{z}\|_{2}(\|\vec{x}-\vec{z}\|_{2}+1)}. | 
 | 
 

 
 
 Adaptive algorithm for streaming explanation queries. 
Patel et al.  ( Patel
et al., 2022 ) describes an adaptive differentially private algorithm that involves sequentially explaining queries with the aid of differential privacy, using information from previously explained queries to optimize future explanations and manage the privacy budget.
Key insights for this approach include reusing past explanations for similar new queries and ensuring that the initialization of the Differentially Private Gradient Descent (DPGD) is as close as possible to the new query to achieve faster convergence and reduce privacy spending. The authors present a weight function α ⁡ ( ‖ x → − z → ‖ ) \alpha(\|\vec{x}-\vec{z}\|) , defined as:

 

 
 (19) | 
 | 
 α ⁡ ( ‖ x → − z → ‖ ) = { 1 if  ​ ‖ x → − z → ‖ ≤ r c 2 ​ ‖ x → − z → ‖ 2 ​ ( ‖ x → − z → ‖ 2 + 1 ) else \alpha(\|\vec{x}-\vec{z}\|)=\begin{cases}1 \text{if }\|\vec{x}-\vec{z}\|\leq r\\
\frac{c}{2\|\vec{x}-\vec{z}\|_{2}(\|\vec{x}-\vec{z}\|_{2}+1)} \text{else}\end{cases} | 
 | 
 

 This weight function is used to identify points similar to z → \vec{z} and is employed to ensure stable and consistent local explanations.

 
 
 Patel et al.  ( Patel
et al., 2022 ) also introduces the idea of a non-interactive differential privacy mechanism to generate new explanations without additional privacy spending by constructing a proxy dataset from previous explanations.

 
 
 

#### 5.1.2. Differentially Private Counterfactual Explanations

 
 Mochaourab et al.  ( Mochaourab et al., 2021 ) develop a differentially private Support Vector Machine (SVM) and introduce methods for generating robust counterfactual explanations.
Yang et al.  ( Yang
et al., 2022 ) creates a differentially private autoencoder to produce privacy-preserving prototypes for each class label, optimizing perturbations to the input data that minimizes distance to the counterfactual while favoring a specific class outcome.
Hamer et al.  ( Hamer et al., 2023 ) suggests data-driven recourse directions could be privatized, but does not elaborate on providing private multi-step recourse paths. Huang et al.  ( Huang
et al., 2023 ) proposes generating privacy-preserving recourse using a differentially private logistic regression model but does not detail the provision of a multi-step path for recourse.
Pentyala et al.  ( Pentyala et al., 2023 ) is a pioneer to offer a complete privacy-preserving pipeline that provides counterfactual explanations with differential privacy guarantees.
Huang et al.  ( Huang
et al., 2023 ) outlines a methodology for incorporating differential privacy (DP) into logistic regression classifiers to offer recourse against membership inference (MI) attacks. Logistic regression is described with weights w w that output a probability score f ⁡ ( x ) = w T ​ x = log ⁡ P ⁡ ( y = 1 | x ) 1 − P ⁡ ( y = 1 | x ) f(x)=w^{T}x=\log\frac{P(y=1|x)}{1-P(y=1|x)} . The counterfactual distance for instance x x from the target score s s in logistic regression space is given by c ⁡ ( x , x ′ ) = s − f ⁡ ( x ) ‖ w ‖ 2 2 c(x,x^{\prime})=\frac{s-f(x)}{\|w\|_{2}^{2}} . The decision boundary is set at s = 0 s=0 , meaning that P ⁡ ( y = 1 | x ) P(y=1|x) is 0.5 at the threshold. In particular, Huang et al.  ( Huang
et al., 2023 ) introduces two DP methods for recourse generation:

 
 
 
 • 
 
 Differentially Private Model (DPM): It involves training the logistic regression classifier with DP. An ϵ \epsilon -DP logistic regression model leads to ϵ \epsilon -DP counterfactual recourse, using IBM’s diffprivlib library  ( Holohan et al., 2019 ) based on Chaudhuri et al.’s mechanism for DP empirical risk minimization  ( Chaudhuri
et al., 2011 ; Wang et al., 2017 ) .

 

 • 
 
 Differentially Private Laplace Recourse (LR): A new method is proposed for DP post-hoc computation of counterfactual recourse that does not touch the underlying logistic regression model training process. It involves:
(1) Applying Laplace noise to the predicted probability score P ​ r ′ ​ ( y = 1 | x ) = P ​ r ​ ( y = 1 | x ) + Laplace ​ ( 1 / ε ) Pr^{\prime}(y=1|x)=Pr(y=1|x)+\text{Laplace}(1/\varepsilon) .
(2) Clamping P ​ r ′ ​ ( y = 1 | x ) Pr^{\prime}(y=1|x) to [ 0 , 1 ] [0,1] .
(3) Computing the noisy logistic regression score f ′ ​ ( x ) f^{\prime}(x) based on P ​ r ′ ​ ( y = 1 | x ) Pr^{\prime}(y=1|x) .
(4) Calculating the noisy CFD as c ′ ​ ( x , x ′ ) = s − f ′ ​ ( x ) ‖ w ‖ 2 2 c^{\prime}(x,x^{\prime})=\frac{s-f^{\prime}(x)}{\|w\|_{2}^{2}} .

 

 
 
 
 Huang et al.  ( Huang
et al., 2023 ) claim that these methods are ϵ \epsilon -DP. This is explained by starting with applying Laplace noise to the predicted probability and noting that the global sensitivity G ​ S p ⁡ ( y = 1 | x ) GS_{p(y=1|x)} is 1. The process from calculating P ​ r ​ ( y = 1 | x ) Pr(y=1|x) to M C ​ F ​ D , L ​ a ​ p ​ ( x ) M_{CFD,Lap}(x) is argued to be a post-processing step that retains ϵ \epsilon -DP, according to the post-processing invariance property of DP  ( Dwork
et al., 2014 ) .

 
 
 Pawelczyk et al.  ( Pawelczyk
et al., 2023 ) proposes that applying DP to a recourse generation algorithm can limit an adversary’s balanced accuracy, with a bound expressed as B ​ A A ≤ 1 2 + 1 2 ⋅ e − ϵ BA_{A}\leq\frac{1}{2}+\frac{1}{2}\cdot e^{-\epsilon} , where ϵ \epsilon is the privacy loss parameter. However, the authors also acknowledges that while DP offers robust privacy assurances, it is not a fail-safe measure and can significantly reduce accuracy, posing a challenge in maintaining the utility of the explanation.
Pentyala et al.  ( Pentyala et al., 2023 ) proposes “PrivRecourse”, a framework for generating privacy-preserving counterfactual explanations.
The method relies on a two-phase approach: a training phase and an inference phase.
The training phase involves training a differentially private ML model f f , clustering the dataset into K K subsets with ( ϵ k , δ k \epsilon_{k},\delta_{k} )-DP guarantees, and constructing a graph G G with clusters as nodes  ( Joshi and Thakkar, 2022 ; Lu and Shen, 2020 ) . Nodes are connected by edges based on distance and density without violating actionable constraints, and the entire graph is published ensuring ( ϵ , δ \epsilon,\delta )-differential privacy  ( Abadi et al., 2016 ; Dwork
et al., 2014 ) .
During the inference phase, for any query instance Z Z , a recourse path P P and a counterfactual instance Z ∗ Z^{*} that would flip the model’s decision to a favorable outcome are computed. This is done by first identifying the nearest node Z 1 Z_{1} to Z Z in G G , and then using Dijkstra’s algorithm to find the shortest path to the favorable counterfactuals in Z C ​ F Z_{CF}   ( Wagner
et al., 2023 ) .

 
 
 Hamer et al.  ( Hamer et al., 2023 ) proposes another framework to generate counterfactuals, called the Stepwise Explainable Paths (StEP).
The framework begins by partitioning the dataset X X into k k clusters { X 1 , … , X k } \{X_{1},...,X_{k}\} . For a point of interest x ~ \tilde{x} , if the model prediction f ⁡ ( x ~ ) = − 1 f(\tilde{x})=-1 indicating an unfavorable outcome, StEP generates a direction d ~ c \tilde{d}_{c} for each cluster using the formula:

 

 
 (20) | 
 | 
 d ~ c = ∑ x ′ ∈ X c ( x ′ − x ~ ) ​ ( α ⁡ ( ‖ x ′ − x ~ ‖ ) ​ f ​ ( x ′ ) = 1 ) \tilde{d}_{c}=\sum_{x^{\prime}\in X_{c}}(x^{\prime}-\tilde{x})(\alpha(||x^{\prime}-\tilde{x}||)f(x^{\prime})=1) | 
 | 
 

 Here, α \alpha is a non-negative function, and | | ⋅ | | ||\cdot|| is a rotation invariant distance metric  ( Sliwinski
et al., 2019 ) . This process repeats iteratively, with the user updating their point of interest x ~ \tilde{x} , until a favourable outcome is achieved.
StEP can be adapted to satisfy ( ϵ , δ \epsilon,\delta )-differential privacy by adding Gaussian noise to the directions computed. When the distance metric is the ℓ 2 \ell_{2} norm, the sensitivity of StEP is upper-bounded by a constant C C , and therefore, Gaussian noise with a mean of 0 and standard deviation σ ≥ C 2 ​ β ϵ \sigma\geq\frac{C^{2}\beta}{\epsilon} where β ≥ 2 ​ log ⁡ ( 1.25 / δ ) \beta\geq 2\log(1.25/\delta) can be added to each feature to achieve differential privacy. When multiple directions are provided to a user, and each is ( ϵ , δ \epsilon,\delta )-differentially private, the overall mechanism is ( k ​ ϵ , k δ k\epsilon,k^{\delta} )-differentially private  ( Dwork
et al., 2014 ) .

 
 
 Yang et al.  ( Yang
et al., 2022 ) proposes another DP-based method through the use of a functional mechanism. The functional mechanism does not add noise directly to the optimal parameter set w ∗ w^{*} , but to the loss function L ~ D ​ ( w ) \tilde{L}_{D}(w) by injecting Laplace noises into the coefficients of its polynomial representation.
The process involves constructing class prototypes in the latent space using a well-trained autoencoder and the functional mechanism through a perturbed training loss. Counterfactual samples are then searched for in the latent space based on these prototypes.
Yang et al.  ( Yang
et al., 2022 ) provides that if the prototype construction process is ϵ \epsilon -differentially private, then the counterfactual explanation process also satisfies DP under the same privacy budget ϵ \epsilon . This relies on the post-processing immunity of DP  ( Dwork
et al., 2014 ) , which allows for certain noises to be added in the prototype construction process without further affecting subsequent computations.

 
 
 

#### 5.1.3. DP-Locally Linear Maps

 
 To create differentially private Locally Linear Maps (LLM), Harder et al.  ( Harder
et al., 2020 ) employs the moments accountant technique combined with differentially private stochastic gradient descent (DP-SGD)  ( Abadi et al., 2016 ) .
The perturbation process involves two main steps per iteration for each minibatch of size L L :
(1) Clipping the norm of the datapoint-wise gradient h t ​ ( x n ) h_{t}(x_{n}) using a threshold C C and adding Gaussian noise to it, resulting in h ^ t \hat{h}_{t} :
 h ^ t ← 1 L ​ ∑ n = 1 L h t ​ ( x n ) + 𝒩 ⁡ ( 0 , σ 2 ​ C 2 ​ I ) \hat{h}_{t}\leftarrow\frac{1}{L}\sum_{n=1}^{L}h_{t}(x_{n})+\mathcal{N}(0,\sigma^{2}C^{2}I) .
(2) Updating the LLM parameters in the descending direction:
 W t + 1 ← W t − η ​ h ^ t W_{t+1}\leftarrow W_{t}-\eta\hat{h}_{t} .
This process ensures that the final LLM is ( ϵ , δ ) (\epsilon,\delta) -differentially private.
To improve the privacy-accuracy trade-off, especially for high-dimensional inputs like images, the author suggest reducing the dimensionality of the parameters by first projecting them onto a lower-dimensional space using a shared matrix R m R_{m} , and then perturbing the gradients of the projected parameters  ( Xue
et al., 2024 ) .

 
 
 Figure 12 . Defences with Privacy-preserving Shapley values. 
 
 
 
 

### 5.2. Defences with Privacy-Preserving SHAP

 
 Several studies have focused on preserve the privacy of users from explanation using Shapley values, including quatization, dimension reduction, multi-party computation, federated learning, and differential privacy (see  Fig. 12 ).

 
 
 Quantized Shapley values. 
Luo et al.  ( Luo
et al., 2022 ) proposes quantization of Shapley values to protect privacy by
reducing mutual information between input features and their corresponding Shapley values. By restricting the Shapley values to a set number of discrete levels (e.g., 5, 10, or 20 distinct values), the entropy of the Shapley values, H ⁡ ( s i ) H(s_{i}) , and hence the mutual information I ⁡ ( x i , s i ) I(x_{i};s_{i}) can be reduced. While quantization has minimal effects on the effectiveness of one attack strategy, it does compromise the accuracy and success rate of another due to the increased range of candidate estimations for a feature, leading to larger estimation errors as per the bounds established earlier. Quantization might also result in two different input samples yielding the same explanation, which is an issue for the privacy-utility balance.

 
 
 Low-dimensional Shapley values. 
Luo et al.  ( Luo
et al., 2022 ) discusses a defensive strategy by suggesting a reduction in the dimensionality of Shapley values. Since the number of Shapley values for a class corresponds to the number of input features, the defence involves only releasing the Shapley values of the top k k features based on their variance, rather than their magnitude.

 
 
 Multi-party Shapley values. 
Jetchev et al.  ( Jetchev and
Vuille, 2023 ) introduces secure multiparty computation (MPC), which allows multiple parties to jointly evaluate a public function on their private data without revealing anything other than the function’s output. The authors developed a privacy-preserving algorithm, XorSHAP, which operates on top of the Manticore MPC framework. This algorithm is a variant of the TreeSHAP method and retains agnosticism towards the underlying MPC framework.
The authors discuss the secret sharing of binary decision trees within an MPC setting, where decision trees can be shared secretly and then used in the computation of privacy-preserving algorithms like XorBoost. Jetchev et al.  ( Jetchev and
Vuille, 2023 ) proves that all subsequent operations and variables in the algorithm are secret and data-independent.

 
 
 Federated Shapley values. 
Wang et al.  ( Wang, 2019 ) discusses interpreting models in the context of Vertical Federated Learning (VFL)  ( Liu
et al., 2024a ; Liu et al., 2023 ; Liu
et al., 2022b ; Liu
et al., 2024b ) where different parties possess different slices of the feature space. Traditional model interpretation methods like Shapley values can reveal sensitive data across parties, making it unsuitable for VFL. To address this, a variant called SHAP Federated is proposed for VFL, particularly for dual-party scenarios involving a host and guest. The host and guest collaboratively develop a machine learning model, with the host owning the label data and part of the feature space, and the guest owning another part.
The algorithm involves setting values in the instance x x to their original or reference values based on whether a feature is hosted or federated and encrypting IDs when necessary to maintain privacy. Then, predictions are made for each combination of features, and feature importance is calculated from the aggregated prediction results using Shapley values. Features that cannot handle missing values are set to either NA or the median  ( Lundberg and Lee, 2017 ) .

 
 
 Differentially Private Shapley values. 
Luo et al.  ( Luo
et al., 2022 ) points out that DP is not suitable for local interpretability methods. For DP to be effective, the explanations for any two different private samples must be indistinguishable, which would reduce the utility of Shapley values as they would become too similar across different samples. As a result, DP cannot be applied to the current problem of maintaining interpretability while defending against attacks that leverage Shapley values.

 
 
 Watson et al.  ( Watson
et al., 2022 ) discusses the computational challenges of calculating Shapley values due to their expensive nature and the privacy concerns in using large portions of datasets for each query. The authors introduce an estimation algorithm that utilizes only a small fraction of data, taking advantage of the property that larger datasets reduce the marginal contributions of individual data points, which are proportionally smaller.
The algorithm is shown to satisfy ϵ \epsilon -differential privacy with a coalition sample complexity of O ⁡ ( ln ⁡ ( n ) ) O(\ln(n))   ( Watson
et al., 2022 ) .
Watson et al.  ( Watson
et al., 2022 ) emphasises the cost advantages of the Layered Shapley approach, which uses fewer data points and has lower computational and data access costs, offering privacy benefits.

 
 
 Figure 13 . Defences with privacy-preserving ML models. 
 
 
 

### 5.3. Defences with Privacy-preserving ML models

 
 To protect user privacy, privacy-preserving ML models have been trained to resist against attacks (see  Fig. 13 ).
Naidu et al.  ( Naidu et al., 2021 ) discusses two primary models of implementing differential privacy: Local DP, where noise is added directly to user data before it is shared, ensuring data privacy against untrusted parties; and Global DP, where a trusted central entity applies differentially private algorithms like DP-SGD  ( Abadi et al., 2016 ) to the collected data to produce models or analyses with limited information leakage (see Fig. 14 ). Interpreting models trained with differential privacy is challenging due to the noise added during training, which obfuscates the model’s decision-making process  ( Patel
et al., 2022 ) . Naidu et al.  ( Naidu et al., 2021 ) investigates the interpretability of differentially private models by establishing the first benchmark for interpretability in deep neural networks (DNNs) trained with differential privacy.

 
 
 Figure 14 . Local and global differential privacy schemes proposed in  ( Naidu et al., 2021 ) . 
 
 
 Liu et al.  ( Liu
et al., 2024d ) develops a model-level defense by employing Differentially-Private Stochastic Gradient Descent (DP-SGD)  ( Bu
et al., 2023 ) , to build inherently private models. The process involves automatic configuration of gradient clipping and the selection of ‘MixOpt’ as the clipping model, uniformly applied across all model layers. While DP-SGD can reduce the effectiveness of membership inference attacks, it also significantly decreases classification accuracy, even with a large epsilon ϵ \epsilon .
Findings indicate that
attribution maps become less informative than even methods not considering model parameters  ( Hooker
et al., 2019 ) . This underscores the challenge of balancing between defense capability and performance utility, as effective defense mechanisms like DP-SGD can significantly impact model accuracy and the quality of explanations provided.

 
 
 Mochaourab et al.  ( Mochaourab et al., 2021 ) outlines a method for providing differential privacy to SVM classifiers by perturbing the optimal weight vector w ∗ w^{*} with additive Laplace noise.
The perturbed weight vector w ~ \tilde{w} is given by w ~ := w ∗ + μ \tilde{w}:=w^{*}+\mu , where μ \mu consists of i.i.d. Laplace random variables μ i ∼ Lap ​ ( 0 , λ ) \mu_{i}\sim\text{Lap}(0,\lambda) . This perturbation ensures β \beta -differential privacy for λ ≥ 4 ​ C k ​ F / ( β ​ n ) \lambda\geq 4C_{k}\sqrt{F}/(\beta n) , with certain conditions on the kernel function ϕ \phi .
Mochaourab et al.  ( Mochaourab et al., 2021 ) introduces robust counterfactual explanations for SVM classifiers, providing explanations for classification results that account for the uncertainty introduced by the differential privacy mechanism.
For the optimization problem, a root of the function g g , defined as:

 

 
 (21) | 
 | 
 y ′ ​ f ϕ ​ ( x , w ~ ) − λ ​ 2 ​ ln ⁡ ( 2 / ( 1 − p ) ) ​ ‖ ϕ ⁡ ( x ) ‖ ≤ 0 y^{\prime}f_{\phi}(x,\tilde{w})-\lambda\sqrt{2\ln(2/(1-p))}\|\phi(x)\|\leq 0 | 
 | 
 

 is considered as a robust counterfactual explanation. Efficient solutions to this optimization problem are proposed using convex optimization solvers like CVXPY for linear SVM or a bisection method for non-linear SVM.
The solution implies that a domain expert’s input is required to determine prototypes representing each class when direct access to test data is not available due to privacy considerations.
A bisection method used for finding robust counterfactual explanations in non-linear SVMs is also developed  ( Mochaourab et al., 2021 ) .

 
 
 Veugen et al.  ( Veugen
et al., 2022 ) uses local foil trees to explain the decisions of a black-box model without accessing its training data. By generating synthetic data points that are close to the user’s data point, classifying them through the model, and then training a decision tree in a secure manner, the method constructs explanations in terms of feature thresholds  ( van der Waa et al., 2018 ) .
This process utilises secret-shared data and secure multi-party computation  ( Lindell, 2020 ) to ensure that no sensitive information from the model or its training data is disclosed, except for the minimal necessary details required to provide the user with an explanation for the classification outcome.

 
 
 

### 5.4. Defences with Perturbations

 
 Jia et al.  ( Jia
et al., 2019b ) introduces a defence technique called MemGuard, differing from other strategies that modify the training process. MemGuard cleverly injects perturbations into the confidence scores produced by the model for each input, transforming these altered scores into adversarial examples aimed at misleading attack models.
However, the primary limitation of MemGuard is its focus on distorting the model’s output by adding noise, which does not protect the attribution maps, thus failing to completely deter the attacks  ( Liu
et al., 2024d ) .

 
 
 Vo et al.  ( Vo et al., 2023 ) describes a methodology for addressing the trade-off between diversity and sparsity in the features modified to form a counterfactual. As shown in Fig. 15 , it introduces a local feature-based perturbation distribution P ⁡ ( z ~ i | z ) P(\tilde{z}_{i}|z) for each mutable feature z i z_{i} , along with a selection distribution Bernoulli ​ ( π i | z ) \text{Bernoulli}(\pi_{i}|z) to control sparsity. To form a counterfactual example z ~ \tilde{z} , the method samples from these distributions and updates mutable features, maintaining validity by maximising the likelihood of the counterfactuals to alter the original outcome.

 
 
 Figure 15 . The approach of generating diverse counterfactuals to reduce the privacy risk in re-identification  ( Vo et al., 2023 ) . 
 
 
 Olatunji et al.  ( Olatunji
et al., 2023 ) discusses a defence mechanism for feature-based explanations.
It involves perturbing each explanation bit, where an explanation is represented as a bit mask, by using a randomised response mechanism. The perturbation probability for flipping each bit ℰ x ​ i \mathcal{E}_{xi} is determined by a privacy budget ϵ \epsilon :

 

 
 | 
 Pr ​ ( ℰ x ​ i ′ = 1 ) = { e ϵ e ϵ + 1 if  ​ ℰ x ​ i = 1 , 1 e ϵ + 1 if  ​ ℰ x ​ i = 0 , \text{Pr}(\mathcal{E}_{xi}^{\prime}=1)=\begin{cases}\frac{e^{\epsilon}}{e^{\epsilon}+1} \text{if }\mathcal{E}_{xi}=1,\\
\frac{1}{e^{\epsilon}+1} \text{if }\mathcal{E}_{xi}=0,\end{cases} | 
 | 
 

 where ℰ x ​ i \mathcal{E}_{xi} and ℰ x ​ i ′ \mathcal{E}_{xi}^{\prime} are the true and perturbed i t ​ h i^{th} bits of explanation, respectively. This method ensures d ​ ϵ d\epsilon -local differential privacy for an explanation with d d dimensions.

 
 
 

### 5.5. Defences with Anonymisation

 
 k-Anonymity. 
Goethals et al.  ( Goethals
et al., 2023 ) presents a unique application of k-anonymity aimed at ensuring anonymity within counterfactual explanations, as opposed to anonymising an entire dataset. This approach is particularly relevant when the dataset is not intended to be fully public. The authors define a counterfactual instance as k-anonymous if its quasi-identifiers – the partially identifying attributes – could apply to at least k individuals within the training set. In turn, a counterfactual explanation inherits this k-anonymity if it is derived from such a k-anonymous instance.
However, while counterfactual explanations usually aim to change the outcome of a model’s prediction, k-anonymous counterfactuals can include a range of instances beyond those used to generate the explanation, leading to uncertainty about whether all values in this range would lead to a change in the prediction.

 
 
 
 
 
 (a) The PPRL-VGAN model proposed in  ( Chen
et al., 2018a ) . 
 
 
 (b) The WGAN-GP framework using pre-trained identifier and task-related classifier  ( Montenegro
et al., 2021 ) . 
 
 
 (c) Privatised counterfactual samples are generated by a counterfactual decoder  ( Montenegro
et al., 2021 ) . 
 
 Figure 16 . Three approaches to defend against attacks with anonymisation. Subfigures (a-b) focus on generating privatised factual samples, while (c) aims to generate privatised counterfactual samples. x m ​ a ​ s ​ k x_{mask} is the mask data, and D p ​ r ​ i ​ v ​ a ​ t ​ i ​ s ​ e ​ d D_{privatised} is the generated privatised data. 
 
 
 Privatised Factual Samples. 
Montenegro et al.  ( Montenegro et al., 2022 ) 
argues that an explanation should not reveal sensitive personal identity information while remaining realistic and informative regarding the decision-making process. Montenegro et al.  ( Montenegro et al., 2022 ) outlines an optimisation objective which involves minimising three loss functions, one for privacy, one for realism, and one for explanatory evidence, each weighted by a non-negative parameter. The distance between a privatised image and the source image is minimised, ensuring that the privatised image is sufficiently different from any identity in the training data to preserve anonymity  ( Montenegro
et al., 2021 ) .

 
 
 Montenegro et al.  ( Montenegro
et al., 2021 ) develops a privacy-preserving network with multi-class identity recognition designed for case-based explanations. The network seeks to preserve privacy by promoting a uniform distribution across identities, making identity recognition akin to random guessing.
The PPRL-VGAN model  ( Chen
et al., 2018a ) (see  16(a) ), which intentionally collapses to the replacement identity and task-related class, is replaced with a WGAN-GP framework that uses a Wasserstein loss with a gradient penalty to stabilise the discriminator (see  16(b) ). This change, alongside using interpretability saliency maps for reconstruction of relevant task-related features, aims to retain the explanatory value in the privatised images  ( Montavon et al., 2017 ) .
Montenegro et al.  ( Montenegro
et al., 2021 ) also introduces another privacy-preserving network that utilises a Siamese identity recognition framework to enhance privacy in domains with scarce images per subject.
They employ a contrastive loss function for training, defined as ContrastiveLoss = 1 2 × Y × E ​ D 2 + 1 2 × ( 1 − Y ) × [ max ⁡ ( 0 , m − E ​ D ) ] 2 \text{ContrastiveLoss}=\frac{1}{2}\times Y\times ED^{2}+\frac{1}{2}\times(1-Y)\times[\max(0,m-ED)]^{2} , where Y Y is the label indicating if the image pair is of the same identity, E ​ D ED is the Euclidean distance between embeddings, and m m is a margin. The Siamese network ensures the privatised image is distinct in identity from the original and others in the dataset.

 
 
 Privatised Counterfactual Samples. 
Montenegro et al.  ( Montenegro
et al., 2021 ) also generates counterfactual explanations from the privatised samples. As shown in  16(c) , a counterfactual generation module, in the form of a decoder, is added to the above privacy-preserving network to map an image’s latent representation to its counterfactual. This decoder is designed to make minimal alterations to the privatised factual explanations to change their predicted class, thereby minimising the pixel-wise distance between the factual and counterfactual explanations while altering the image’s task-related prediction. Saliency masks and explanatory features are used to guide changes to image regions that are relevant to the explanation.
The loss function for the counterfactual decoder training is represented as L C = E I , M ∼ p d ​ a ​ t ​ a [ λ x [ F ( I ) × ( 1 − M ) − C ( I ) × ( 1 − M ) ] 2 + λ D E x p ( D e ​ x ​ p ( I ) × log ( 1 − D e ​ x ​ p ( C ( I ) ) ) ] L_{C}=E_{I,M\sim p_{data}}[\lambda_{x}[F(I)\times(1-M)-C(I)\times(1-M)]^{2}+\lambda_{D}Exp(D_{exp}(I)\times\log(1-D_{exp}(C(I)))] , where F ⁡ ( I ) F(I) and C ⁡ ( I ) C(I) denote the privatized factual and counterfactual explanations, respectively, and λ x \lambda_{x} and λ D \lambda_{D} are weights controlling the importance of each term in the loss function.

 
 
 

### 5.6. Defences with Collaborative Explanation

 
 Domingo et al.  ( Domingo-Ferrer et al., 2019 ) presents methods for collaborative rule-based model approximation without the direct use of a model simulator. It suggests that users can employ simulators to interact with a concealed model to obtain responses for certain feature sets, which although limited and controlled, can help deduce how the model makes decisions. While simulators prevent full transparency of the model and often limit the number of queries to prevent misuse, users can collaborate by querying the model for various feature sets and publishing the predictions. This collective data can then be mined for decision rules to approximate the model’s logic.

 
 
 

### 5.7. Defences against Reconstruction Attacks

 
 Gaudio et al.  ( Gaudio et al., 2023 ) proposes the “DeepFixCx” model, an approach that utilises wavelet packet transforms and spatial pooling for image compression that preserves privacy and explicability (see  Fig. 17 ). The method relies on analysing images with multi-scale wavelet-based methods, allowing local regions of pixels to be summarised at multiple scales.
The wavelet packet transform offers several benefits, such as facilitating image processing with deep learning libraries, ensuring that all coefficient values represent equally-sized pixel regions, and maintaining consistency with boundary effects.
“DeepFixCx” provides a trade-off between compressing images for efficiency while still retaining enough detail for reconstruction and privacy preservation.
Gaudio et al.  ( Gaudio et al., 2023 ) also outlines methods for inverse wavelet packet transform for image reconstruction, which can restore images from compressed representations to their original size. This model offers a privacy-conscious method to process images for various applications, including medical imaging, by removing local spatial information, allowing for the preservation of privacy without the need for additional learning.

 
 
 Figure 17 . The “DeepFixCx” model uses compression techniques (i. e., wavelet packet transform and a pooling function) for preserving privacy and explicability  ( Gaudio et al., 2023 ) . 
 
 
 
 

## 6. Published Resources

 
 Table 2 . Published Algorithms and Models 
 
 
 
 

 
 
 Algorithms 
 Year 
 Target Explanations 
 Attacks 
 Defenses 
 Code Repository 
 
 L2C  ( Vo et al., 2023 ) 
 2023 
 Counterfactual 
 – 
 Perturbation 
 github.com/isVy08/L2C/ 
 
 GSEF  ( Olatunji
et al., 2023 ) 
 2023 
 Feature-based 
 Graph Extraction 
 Perturbation 
 github.com/iyempissy/graph-stealing-attacks-with-explanation 
 
 Ferry et al.  ( Ferry et al., 2023a ) 
 2023 
 Interpretable models 
 Data Reconstruction 
 - 
 github.com/ferryjul/ProbabilisticDatasetsReconstruction 
 
 DeepFixCX  ( Gaudio et al., 2023 ) 
 2023 
 Case-based 
 Identity recognition 
 Anonymisation 
 github.com/adgaudio/DeepFixCX 
 
 DP-XAI 
 2023 
 ALE plot 
 - 
 Differential Privacy 
 github.com/lange-martin/dp-global-xai 
 
 Duddu et al.  ( Duddu and Boutet, 2022 ) 
 2022 
 Gradient/Perturbation-based 
 Attribute Inference 
 - 
 github.com/vasishtduddu/AttInfExplanations 
 
 DataShapley  ( Watson
et al., 2022 ) 
 2022 
 Shapley 
 - 
 Differential Privacy 
 github.com/amiratag/DataShapley 
 
 MEGEX  ( Miura
et al., 2021 ) 
 2021 
 Gradient-based 
 Model Extraction 
 - 
 github.com/cake-lab/datafree-model-extraction 
 
 Mochaourab et al.  ( Mochaourab et al., 2021 ) 
 2021 
 Counterfactual 
 - 
 Private SVM 
 github.com/rami-mochaourab/robust-explanation-SVM 
 
 Gillenwater et al.  ( Gillenwater
et al., 2021 ) 
 2021 
 Quantiles 
 - 
 Differential Privacy 
 github.com/google-research/google-research/tree/master/dp_multiq 
 
 DP-LLM  ( Harder
et al., 2020 ) 
 2020 
 Locally linear maps 
 - 
 Differential Privacy 
 github.com/frhrdr/dp-llm 
 
 MRCE  ( Aïvodji
et al., 2020 ) 
 2020 
 Counterfactual 
 Model Extraction 
 - 
 github.com/aivodji/mrce 
 
 Federated SHAP  ( Wang, 2019 ) 
 2019 
 Shapley 
 - 
 Federated 
 github.com/crownpku/federated_shap 
 
 

 
 
 

### 6.1. Published Algorithms

 
 Several algorithm and model implementations have been pivotal to foundational experiments in maintaining privacy within model explanations.
 Table 2 provides a consolidated list of published algorithms and models, categorised by their release year (ranging from 2019 to 2023), the types of explanations they target (such as Counterfactual, ALE plot, Shapley values), potential attacks (like Perturbation, Graph Extraction), and corresponding defences (including Differential Privacy, Anonymisation). Each listed algorithm, such as L2C, DP-XAI, and GSF, among others, is accompanied by a link to its code repository on GitHub, allowing for easy access to their implementation details for further exploration or usage.

 
 
 

### 6.2. Published Datasets

 
 Table 3 . Highlighted Datasets 
 
 
 
 

 
 
 Category 
 Dataset 
 #Items 
 Disk Size 
 Downstream Explanations 
 Experimented in 
 URL 
 
 Image 
 MNIST 
 70K 
 11MB 
 Counterfactuals, Gradient 
 ( Huang
et al., 2023 ; Yang
et al., 2022 ; Zhao
et al., 2021b ; Milli
et al., 2019 ) 
 www.kaggle.com/datasets/hojjatk/mnist-dataset 
 
 
 CIFAR 
 60K 
 163MB 
 Gradient 
 ( Miura
et al., 2021 ; Shokri
et al., 2021 ; Milli
et al., 2019 ; Liu
et al., 2024d ) 
 www.cs.toronto.edu/~kriz/cifar.html 
 
 
 SVHN 
 600K 
 400MB+ 
 Gradient 
 ( Miura
et al., 2021 ) 
 ufldl.stanford.edu/housenumbers/ 
 
 
 Food101 
 100K+ 
 10GB 
 Case-based 
 ( Gaudio et al., 2023 ) 
 www.kaggle.com/datasets/dansbecker/food-101 
 
 
 Flowers102 
 8K+ 
 300MB+ 
 Case-based 
 ( Gaudio et al., 2023 ) 
 www.robots.ox.ac.uk/~vgg/data/flowers/102/ 
 
 
 Cervical 
 8K+ 
 46GB+ 
 Case-based, Interpretable Models 
 ( Gaudio et al., 2023 ) 
 www.kaggle.com/competitions/intel-mobileodt-cervical-cancer-screening 
 
 
 CheXpert 
 220K+ 
 GBs 
 Case-based, Interpretable Models 
 ( Gaudio et al., 2023 ) 
 stanfordmlgroup.github.io/competitions/chexpert/ 
 
 
 Facial Expression 
 12K+ 
 63MB 
 Black-box 
 ( Patel
et al., 2022 ) 
 www.kaggle.com/datasets/msambare/fer2013 
 
 
 Celeb 
 200K 
 GBs 
 Gradient 
 ( Zhao
et al., 2021b ) 
 mmlab.ie.cuhk.edu.hk/projects/CelebA.html 
 
 Tabular 
 Adult 
 48K+ 
 10MB 
 Counterfactuals, Shapley, Gradient, Perturbation 
 10+ ( ( Huang
et al., 2023 ; Ferry et al., 2023a ; Pentyala et al., 2023 ) etc.) 
 archive.ics.uci.edu/ml/datasets/adult 
 
 
 COMPAS 
 7K+ 
 25MB 
 Counterfactuals, Shapley, Gradient, Perturbation 
 ( Ferry et al., 2023a ; Duddu and Boutet, 2022 ) 
 www.kaggle.com/datasets/danofer/compass 
 
 
 FICO 
 10K+ 
 ≤ \leq 1MB 
 Counterfactuals, Shapley 
 ( Huang
et al., 2023 ; Wang
et al., 2022 ; Pentyala et al., 2023 ; Pawelczyk
et al., 2023 ) 
 community.fico.com/s/explainable-machine-learning-challenge 
 
 
 Boston Housing 
 500+ 
 ≤ \leq 1MB 
 Counterfactuals, Shapley 
 ( Wang
et al., 2022 ) 
 www.kaggle.com/code/prasadperera/the-boston-housing-dataset 
 
 
 German Credit 
 1K 
 ≤ \leq 1MB 
 Counterfactuals, Shapley, Gradient, Perturbation 
 ( Vo et al., 2023 ; Goethals
et al., 2023 ; Yang
et al., 2022 ; Duddu and Boutet, 2022 ) 
 archive.ics.uci.edu/dataset/144/statlog+german+credit+data 
 
 
 Student Admission 
 500 
 ≤ \leq 1MB 
 Counterfactuals, Shapley 
 ( Vo et al., 2023 ) 
 www.kaggle.com/datasets/mohansacharya/graduate-admissions 
 
 
 Student Performance 
 10K 
 ≤ \leq 1MB 
 Counterfactuals, Shapley 
 ( Vo et al., 2023 ) 
 www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression 
 
 
 GMSC 
 150K+ 
 15MB 
 Counterfactuals, Shapley 
 ( Wang
et al., 2022 ; Naretto
et al., 2022 ) 
 www.kaggle.com/c/GiveMeSomeCredit/data 
 
 
 Diabetes 
 100K+ 
 20MB 
 Counterfactuals, Shapley 
 ( Pawelczyk
et al., 2023 ; Luo
et al., 2022 ; Yang
et al., 2022 ; Watson
et al., 2022 ; Shokri
et al., 2021 ) 
 archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 
 
 
 Breast Cancer 
 569 
 1 ​ M ​ B 1MB 
 Interpretable models, Counterfactuals 
 ( Mochaourab et al., 2021 ) 
 archive.ics.uci.edu/ml/datasets/breast+cancer 
 
 Graph 
 Cora 
 2K+ 
 4.5MB 
 Feature-based 
 ( Olatunji
et al., 2023 ) 
 relational.fit.cvut.cz/dataset/CORA 
 
 
 Bitcoin 
 30K 
 ≤ \leq 1MB 
 Feature-based 
 ( Olatunji
et al., 2023 ) 
 snap.stanford.edu/data/soc-sign-bitcoin-alpha.html 
 
 
 CIC-IDS2017 
 2.8M+ 
 500MB 
 Counterfactuals 
 ( Kuppa and Le-Khac, 2021 ) 
 www.unb.ca/cic/datasets/ids-2017.html 
 
 Text 
 IMDB Review 
 50K 
 66MB 
 Black-box 
 ( Patel
et al., 2022 ) 
 ai.stanford.edu/~amaas/data/sentiment/ 
 
 

 
 
 
 The datasets most commonly utilized for privacy-preserving model explanations are depicted in Table 3 . We categorize these datasets into various groups based on their application domains. Important datasets are described below.

 
 
 Image. 
 The CIFAR dataset   ( Krizhevsky, 2009 ) consists of two parts. The initial subset, CIFAR-10, comprises ten categories of objects, each with six thousand images. These categories include airplanes, automobiles, various animals, and trucks. The training set consists of five thousand randomly selected images per category, with the remaining images used as test examples. The second section, CIFAR-100, contains 600 images for each of its 100 classes. These classes are further grouped into 20 superclasses, each containing five classes.

 
 
 The SVHN dataset   ( Netzer et al., 2011 ) was compiled using automated methods and Amazon Mechanical Turk from an extensive collection of Google Street View images. It encompasses nearly 600,000 labeled characters, comprising complete numbers and chopped digits in a 32x32 pixel format similar to MNIST. It consists of three subsets: over seventy thousand samples for training, twenty thousand for testing, and approximately half a million additional samples.

 
 
 The Food101 dataset   ( Bossard
et al., 2014 ) was created by gathering images from foodspotting.com, including 101 popular dishes with 750 training and 250 test images per class. Training images were intentionally left uncleaned to simulate real-world noise. All images were resized, resulting in a total of 101,000 diverse food images.

 
 
 Text. 
 The IMDB/Amazon movie reviews dataset   ( Ni et al., 2019 ) contains 8,765,568 movie reviews sourced from the Amazon review dataset, along with an additional 50,000 reviews from the IMDB large review dataset. These reviews are represented as binary vectors using the top 500 words. Each review is classified as either positive (+1) or negative (-1).

 
 
 Tabular. 
 The UCI Adult Income dataset   ( Ferry et al., 2023a ) provides insights from the 1994 U.S. census, aiming to forecast whether an individual earns over $50,000 annually. Numeric features are divided into quantiles, while categorical features are transformed into binary form through one-hot encoding. This dataset comprises 48,842 examples, each characterized by 24 binary features.

 
 
 The Diabetes dataset   ( Strack et al., 2014 ) contains information from diabetic patients gathered via two methods: traditional paper records and an automated recording system. While paper records indicate time slots of the day, the automated system timestamps occurrences accurately. Each entry in the dataset comprises four fields separated by tabs, with records separated by new lines.

 
 
 FICO Explainable Machine Learning Challenge: 
The dataset contains anonymized HELOC (Home Equity Line of Credit) applications from homeowners  ( Sokol and Flach, 2019 ; Huang
et al., 2023 ) . HELOCs are credit lines that banks offer based on a percentage of a home’s equity. Applicants in the dataset have requested credit lines ranging from $5,000 to $150,000.
The prediction task is to determine the binary target variable “RiskPerformance”, where “Bad” signifies a 90-day overdue payment at least once in 24 months, and “Good” indicates timely payments without significant delinquency.

 
 
 Graph. 
 Cora   ( Sen et al., 2008 ) is a dataset focused on citations, where each node represents a research article. If one article cites another, there’s an edge between them. Each node is labeled with its article category. The features of each node are represented by a binary word vector, indicating whether a word is present or absent in the article’s abstract.

 
 
 The Bitcoin dataset   ( Kumar et al., 2016 ) is a network representation of trading accounts within the Bitcoin ecosystem. In this dataset, each trading account is depicted as a node, and there are weighted edges connecting pairs of accounts, symbolizing the level of trust between them. The weights range from +10, indicating complete trust, to -10, signifying complete distrust. Each node is labeled to denote its trustworthiness status. The feature vector associated with each node is derived from ratings provided by other users, including metrics such as average positive or negative ratings.

 
 
 The CICIDS17 dataset , collected under controlled conditions, contains network traffic data in both packet-based and bidirectional flow-based formats. Each flow in the dataset is associated with over 80 features, capturing various aspects of network behavior.
The dataset is organized into eight groups of features extracted from raw pcaps, including interarrival times, active-idle times, flags-based features, flow characteristics, packet counts with flags, and average bytes and packets sent in various contexts.

 
 
 

### 6.3. Evaluation Metrics

 
 Table 4 provides the formulas and usages for common metrics in privacy attacks and defences on model explanations. We summarize their descriptions below.

 
 
 Table 4 . Highlighted Evaluation Metrics 
 
 
 
 

 
 
 Category 
 
 
 Evaluation Metrics 
 
 
 
 Formula/Description 
 
 
 
 Usage 
 
 
 Explanation Utility 
 
 
 Counterfactual validity  ( Goethals
et al., 2023 ) 
 
 
 
 Pureness = # value combinations with desired outcome # value combinations \text{Pureness}=\frac{\text{\# value combinations with desired outcome}}{\text{\# value combinations}} 
 
 
 
 Assess the range of attribute values within k-anonymous counterfactual instances. Consider all attributes, including those beyond quasi-identifiers 
 
 
 
 
 
 Classification metric  ( Goethals
et al., 2023 ) 
 
 
 
 C ​ M = ∑ i = 1 N penalty ​ ( t ​ u ​ p ​ l ​ e i ) N CM=\frac{\sum_{i=1}^{N}\text{penalty}(tuple_{i})}{N} 
 
 
 
 Assess equivalence classes within anonymized datasets, focusing on class label uniformity. 
 
 
 
 
 
 Faithfulness (RDT-Fidelity)  ( Olatunji
et al., 2023 ; Funke
et al., 2022 ) 
 
 
 
 ℱ ⁡ ( ℰ X ) = 𝔼 Y ℰ X | Z ∼ 𝒩 ​ [ 1 f ⁡ ( X ) = f ⁡ ( Y ℰ X ) ] \mathcal{F}(\mathcal{E}_{X})=\mathbb{E}_{Y_{\mathcal{E}_{X}}|Z\sim\mathcal{N}}\left[1_{f(X)=f(Y_{\mathcal{E}_{X}})}\right] 
 
 
 
 Reflect how often the model’s predictions are unchanged despite perturbations to the input, which would suggest that the explanation is effectively capturing the reasoning behind the model’s predictions. 
 
 
 
 
 
 Sparsity  ( Olatunji
et al., 2023 ; Funke
et al., 2022 ) 
 
 
 
 H ( p ) = − ∑ f ∈ M p ( f ) log p ( f ) H(p)=-\sum_{f\in M}p(f)\log p(f) 
 
 
 
 A complete and faithful explanation to the model should inherently be sparse, focusing only on a select subset of features that are most predictive of the model’s decision. 
 
 
 Information Loss 
 
 
 Normalised Certainty Penalty (NCP)  ( Goethals
et al., 2023 ) 
 
 
 
 NCP ​ ( G ) = ∑ i = 1 d w i ⋅ NCP A i ​ ( G ) \text{NCP}(G)=\sum_{i=1}^{d}w_{i}\cdot\text{NCP}_{A_{i}}(G) 
 
 
 
 Higher NCP values indicate a greater degree of generalization and more information loss. This metric helps in assessing the balance between data privacy and utility. 
 
 
 
 
 
 Discernibility  ( Goethals
et al., 2023 ) 
 
 
 
 C D ​ M ​ ( g , k ) = ∑ V ​ E ​ s . t . | E | ≥ k | E | 2 + ∑ V ​ E ​ s . t . | E | k | D | ​ | E | C_{DM}(g,k)=\sum_{VE\,s.t.\,|E|\geq k}|E|^{2}+\sum_{VE\,s.t.\,|E| k}|D||E| 
 
 
 
 Measure the penalties on tuples in a dataset after k-anonymization, reflecting how indistinguishable they are post-anonymization 
 
 
 
 
 
 Approximation Loss  ( Goethals
et al., 2023 ) 
 
 
 
 ℰ ⁡ ( ϕ ^ , 𝒵 , f ⁡ ( X ) ) ≜ 𝔼 ⁡ [ ℒ ⁡ ( ϕ ^ , 𝒵 , f ⁡ ( X ) ) − ℒ ⁡ ( ϕ ∗ , 𝒵 , f ⁡ ( X ) ) ] . \mathcal{E}(\hat{\phi},\mathcal{Z},f(X))\triangleq\mathbb{E}[\mathcal{L}(\hat{\phi},\mathcal{Z},f(X))-\mathcal{L}(\phi^{*},\mathcal{Z},f(X))]. 
 
 
 
 Measure the error caused by randomness added when minimizing the privacy loss as the expected deviation of the randomized explanation from the best local approximation 
 
 
 
 
 
 Explanation Intersection  ( Olatunji
et al., 2023 ; Funke
et al., 2022 ) 
 
 
 
 The percentage of bits in the original explanation that is retained in the privatised explanation after using differential privacy 
 
 
 
 The higher the better but due to privacy-utility trade-off, this metric should not be 100%. 
 
 
 Privacy Degree 
 
 
 k k -anonymity  ( Goethals
et al., 2023 ) 
 
 
 
 A person’s information is indistinguishable from at least k-1 other individuals. 
 
 
 
 Refers to the number of individuals in the training dataset to whom a given explanation could potentially be linked  ( Goethals
et al., 2023 ) . 
 
 
 
 
 
 Information Leakage  ( Patel
et al., 2022 ) 
 
 
 
 P r i = 1 . . k ϕ ^ ( 𝐳 𝐢 , X , f D ( X ) ) ≤ e ε ^ ⋅ P r [ ϕ ^ ( 𝐳 𝐢 , X , f D ′ ( X ) ) : ∀ i ] + δ ^ Pr_{i=1..k}\hat{\phi}(\mathbf{z_{i}},X,f_{D}(X))\leq e^{\hat{\varepsilon}}\cdot Pr[\hat{\phi}(\mathbf{z_{i}},X,f^{\prime}_{D}(X)):\forall i]+\hat{\delta} 
 
 
 
 If an adversary can access model explanations, they would not gain any additional information that could help in inferring something about the training data beyond what could be learned from the model predictions alone 
 
 
 
 
 
 Privacy Budget 
 
 
 
 The total privacy budget for all queries is fixed at ( ε , δ \varepsilon,\delta ). 
 
 
 
 The explanation algorithm must not exceed the overall budget across all queries. Stricter requirement ( ε m ​ i ​ n , δ m ​ i ​ n \varepsilon_{min},\delta_{min} ) is set for each individual query. 
 
 
 Attack Success 
 
 
 Precision/Recall/F1  ( Duddu and Boutet, 2022 ) 
 
 
 
 P ​ r ​ e ​ c = T ​ P T ​ P + F ​ P Prec=\frac{TP}{TP+FP} , R ​ e ​ c = T ​ P T ​ P + F ​ N Rec=\frac{TP}{TP+FN} , F ​ 1 = 2 × precision × recall precision + recall F1=2\times\frac{\text{precision}\times\text{recall}}{\text{precision}+\text{recall}} 
 
 
 
 Evaluate an attack’s effectiveness in correctly and completely identifying the properties it is designed to infer. 
 
 
 
 
 
 Balanced Accuracy  ( Liu
et al., 2024d ; Pawelczyk
et al., 2023 ; Huang
et al., 2023 ) 
 
 
 
 B ​ A = T ​ P ​ R + T ​ N ​ R 2 BA=\frac{TPR+TNR}{2} 
 
 
 
 Measures the accuracy of attack (e.g. membership prediction in membership inference attacks), on a balanced dataset of members and non-members. 
 
 
 
 
 
 ROC/AUC  ( Huang
et al., 2023 ; Pawelczyk
et al., 2023 ; Liu
et al., 2024d ; Ferry et al., 2023a ; Olatunji
et al., 2023 ) 
 
 
 
 The ROC curve plots the true positive rate against the false positive rate at various threshold settings. 
 
 
 
 An AUC near 1 indicates a highly successful privacy attack, while an AUC close to 0.5 suggests no better performance than random guessing. 
 
 
 
 
 
 TPR at Low FPR  ( Liu
et al., 2024d ; Huang
et al., 2023 ; Pawelczyk
et al., 2023 ) 
 
 
 
 Report TPR at a fixed FPR (e.g., 0.1%). 
 
 
 
 If an attack can pinpoint even a minuscule fraction of the training dataset with high precision, then the attack ought to be deemed effective. 
 
 
 
 
 
 Mean Absolute Error (MAE)  ( Luo
et al., 2022 ) 
 
 
 
 ℓ 1 ​ ( x ^ , x ) = 1 m ​ n ​ ∑ j = 1 m ∑ i = 1 n | x ^ i j − x i j | , \ell_{1}(\hat{x},x)=\frac{1}{mn}\sum_{j=1}^{m}\sum_{i=1}^{n}|\hat{x}_{i}^{j}-x_{i}^{j}|, 
 
 
 
 Gives an overview of how accurately an attack can reconstruct private inputs by averaging the absolute differences across all samples and features. 
 
 
 
 
 
 Success Rate (SR)  ( Luo
et al., 2022 ) 
 
 
 
 S ​ R = | X ^ v ​ a ​ l ≠ ⟂ | m ​ n SR=\frac{|\hat{X}_{val}\neq\perp|}{mn} 
 
 
 
 The ratio of successfully reconstructed features to the total number of features across all samples 
 
 
 
 
 
 Model Agreement  ( Wang
et al., 2022 ) 
 
 
 
 Agreement = 1 n ​ ∑ i = 1 n 1 f θ ​ ( x i ) = h ϕ ​ ( x i ) . \text{Agreement}=\frac{1}{n}\sum_{i=1}^{n}1_{f_{\theta}(x_{i})=h_{\phi}(x_{i})}. 
 
 
 
 A higher agreement indicates that the substitute model is more similar to the original model. When comparing two model extraction methods with the same agreement, the one with the lower standard deviation is preferred. 
 
 
 
 
 
 Average Uncertainty Reduction  ( Ferry et al., 2023a ) 
 
 
 
 D ​ i ​ s ​ t ​ ( 𝒟 M , 𝒟 O ​ r ​ i ​ g ) = 1 n ⋅ d ​ ∑ i = 1 n ∑ k = 1 d H ⁡ ( 𝒟 i , k M ) H ⁡ ( 𝒟 i , k ) Dist(\mathcal{D}^{M},\mathcal{D}^{Orig})=\frac{1}{n\cdot d}\sum_{i=1}^{n}\sum_{k=1}^{d}\frac{H(\mathcal{D}^{M}_{i,k})}{H(\mathcal{D}_{i,k})} 
 
 
 
 The degree to which a data reconstruction attack is accurate, measured by the reduction in uncertainty across all features of all samples in the dataset 
 
 
 

 
 
 

#### 6.3.1. Explanation utility

 
 Protecting the privacy might reduce the utility of explanations. Several metrics have been proposed to measure the utility of explanations after privacy protection.

 
 
 Counterfactual validity. 
Goethals et al.  ( Goethals
et al., 2023 ) proposes a pureness metric to measure the validity of counterfactual explanations. It involves assessing the range of attribute values within k-anonymous counterfactual instances. It is important to consider all attributes, including those beyond quasi-identifiers. For categorical attributes, the focus is on the values within the k-anonymous instance, whereas for numerical attributes, the consideration extends to those values also present in the training set.
The pureness of a k-anonymous counterfactual explanation is defined by the formula:

 

 
 | 
 Pureness = # ​  value combinations with desired outcome # ​  value combinations \text{Pureness}=\frac{\#\text{ value combinations with desired outcome}}{\#\text{ value combinations}} | 
 | 
 

 Practically, it is approximated by querying the model with a set number of random combinations (e.g., 100) to see how many result in the desired prediction outcome. Pureness represents the proportion of these combinations that lead to the desired outcome, aiming for as high a percentage as possible, ideally 100%.

 
 
 Classification metric. 
The classification metric (CM) is used to assess equivalence classes within anonymised datasets, focusing on class label uniformity  ( Goethals
et al., 2023 ) . It is calculated as:

 

 
 | 
 C ​ M = ∑ i = 1 N penalty ​ ( t ​ u ​ p ​ l ​ e i ) N CM=\frac{\sum_{i=1}^{N}\text{penalty}(tuple_{i})}{N} | 
 | 
 

 Here, N N is the number of anonymized tuples. A penalty of 1 is assigned to each tuple whose class label differs from the majority class label of its equivalence class. If the tuple’s class label matches the majority, no penalty is given. The CM is related to but distinct from the concept of pureness. Unlike pureness, which considers all possible attribute value combinations, the CM specifically evaluates the class label uniformity within each equivalence class. Pureness is considered more suitable for evaluating how often an anonymous counterfactual explanation provides correct advice because it takes into account the entire range of possible attribute combinations, rather than just the observed instances  ( Goethals
et al., 2023 ) .

 
 
 RDT-Fidelity. 
Olatunjii et al.  ( Olatunji
et al., 2023 ) describes a metric for measuring the quality of explanations for model predictions through a metric called faithfulness. Faithfulness indicates how well an explanation approximates the model’s behavior. Since a ground truth for explanations is often unavailable, the measure used is RDT-Fidelity (grounded in rate-distortion theory  ( Funke
et al., 2022 ) ), which assesses faithfulness by comparing the model’s original and new predictions. The fidelity score is calculated as follows:

 

 
 | 
 ℱ ⁡ ( ℰ X ) = 𝔼 Y ℰ X | Z ∼ 𝒩 ​ [ 1 f ⁡ ( X ) = f ⁡ ( Y ℰ X ) ] \mathcal{F}(\mathcal{E}_{X})=\mathbb{E}_{Y_{\mathcal{E}_{X}}|Z\sim\mathcal{N}}\left[1_{f(X)=f(Y_{\mathcal{E}_{X}})}\right] | 
 | 
 

 Here, ℰ X \mathcal{E}_{X} represents the explanation, f f is the model function (like a Graph Neural Network), X X is the original input, ℳ ⁡ ( ℰ X ) \mathcal{M}(\mathcal{E}_{X}) is the explanation mask applied to X X , Z Z is noise drawn from distribution 𝒩 \mathcal{N} , and I ~ ℰ X \tilde{I}_{\mathcal{E}_{X}} is the perturbed input defined by:

 

 
 | 
 I ~ ℰ X = X ⊙ ℳ ⁡ ( ℰ X ) + Z ⊙ ( 1 − ℳ ⁡ ( ℰ X ) ) , Z ∼ 𝒩 , \tilde{I}_{\mathcal{E}_{X}}=X\odot\mathcal{M}(\mathcal{E}_{X})+Z\odot(1-\mathcal{M}(\mathcal{E}_{X})),Z\sim\mathcal{N}, | 
 | 
 

 where ⊙ \odot denotes element-wise multiplication and 1 1 represents a matrix of ones of appropriate size. The score reflects how often the model’s predictions are unchanged despite perturbations to the input, which would suggest that the explanation is effectively capturing the reasoning behind the model’s predictions.

 
 
 Sparsity. 
Olatunjii et al.  ( Olatunji
et al., 2023 ) argues that a complete and faithful explanation to the model should inherently be sparse, focusing only on a select subset of features that are most predictive of the model’s decision. The measurement of sparsity is done using an entropy-based definition which can be applied to both soft and hard explanation masks. The sparsity of an explanation is quantified by the entropy H ⁡ ( p ) H(p) over the normalised distribution p p of the explanation masks, calculated using the formula  ( Funke
et al., 2022 ) :

 

 
 | 
 H ( p ) = − ∑ f ∈ M p ( f ) log p ( f ) H(p)=-\sum_{f\in M}p(f)\log p(f) | 
 | 
 

 Here, M M represents the set of features and log ⁡ ( | M | ) \log(|M|) bounds the entropy. A lower entropy value implies a sparser explanation.

 
 
 

#### 6.3.2. Information loss

 
 Excessive anonymisation often results in the loss of valuable information. As the level of anonymisation increases, the data utility typically decreases, hindering certain types of analysis or yielding outcomes that are biased or inaccurate.

 
 
 Normalised Certainty Penalty (NCP). 
It quantifies the information loss that occurs when attributes are anonymised  ( Goethals
et al., 2023 ) . NCP is higher for attributes that, when generalised, encompass a wide range of possible values, indicating greater information loss:
For numerical quasi-identifiers in an equivalence class G G , NCP is calculated using:
 NCP A n ​ u ​ m ​ ( G ) = m ​ a ​ x A n ​ u ​ m G − m ​ i ​ n A n ​ u ​ m G m ​ a ​ x A n ​ u ​ m − m ​ i ​ n A n ​ u ​ m \text{NCP}_{A_{num}}(G)=\frac{max^{G}_{A_{num}}-min^{G}_{A_{num}}}{max^{A_{num}}-min^{A_{num}}} .
For categorical quasi-identifiers, NCP A c ​ a ​ t ​ ( G ) \text{NCP}_{A_{cat}}(G) is 0 0 if | A G | = 1 |A^{G}|=1 and | A G | | A | \frac{|A^{G}|}{|A|} otherwise.
The overall NCP for an equivalence class G G across all quasi-identifier attributes is the weighted sum:

 

 
 | 
 NCP ​ ( G ) = ∑ i = 1 d w i ⋅ NCP A i ​ ( G ) \text{NCP}(G)=\sum_{i=1}^{d}w_{i}\cdot\text{NCP}_{A_{i}}(G) | 
 | 
 

 where d d is the number of quasi-identifiers, A i A_{i} is the i t ​ h i^{th} attribute with weight w i w_{i} , and ∑ w i = 1 \sum w_{i}=1 .
Higher NCP values indicate a greater degree of generalization and more information loss. This metric helps in assessing the balance between data privacy and utility.

 
 
 Discernibility. 
The discernibility metric C D ​ M ​ ( g , k ) C_{DM}(g,k) , which is used to measure the penalties on tuples in a dataset after k-anonymization, reflecting how indistinguishable they are post-anonymization  ( Goethals
et al., 2023 ) . The goal is to maintain discernibility between tuples within the constraints of a given privacy level k. The metric is defined as:

 

 
 | 
 C D ​ M ​ ( g , k ) = ∑ V ​ E ​ s . t . | E | ≥ k | E | 2 + ∑ V ​ E ​ s . t . | E | k | D | ​ | E | C_{DM}(g,k)=\sum_{VE\,s.t.\,|E|\geq k}|E|^{2}+\sum_{VE\,s.t.\,|E| k}|D||E| | 
 | 
 

 Here, E E denotes the equivalence class of the tuple, and D D represents the entire dataset. A successfully anonymized tuple (with an equivalence class larger than k) incurs a penalty equivalent to the square of the equivalence class size, while a suppressed tuple (with an equivalence class smaller than k) incurs a penalty proportional to the size of the dataset multiplied by the equivalence class size. The metric has been critiqued for not considering how closely the anonymized instances resemble the original data  ( Goethals
et al., 2023 ) . The Normalized Certainty Penalty (NCP) is suggested as a more appropriate metric for gauging the actual information loss in the process of anonymizing counterfactual explanations.

 
 
 Error in private approximation. 
Patel et al.  ( Patel
et al., 2022 ) proposes a metric to measure the error caused by randomness added when privately minimizing ℒ ⁡ ( ⋅ ) \mathcal{L}(\cdot) for protecting X X as the expected deviation of the randomized explanation from the best local approximation. More formally, the approximation loss is defined as:

 

 
 | 
 ℰ ⁡ ( ϕ ^ , 𝒵 , f ⁡ ( X ) ) ≜ 𝔼 ⁡ [ ℒ ⁡ ( ϕ ^ , 𝒵 , f ⁡ ( X ) ) − ℒ ⁡ ( ϕ ∗ , 𝒵 , f ⁡ ( X ) ) ] . \mathcal{E}(\hat{\phi},\mathcal{Z},f(X))\triangleq\mathbb{E}[\mathcal{L}(\hat{\phi},\mathcal{Z},f(X))-\mathcal{L}(\phi^{*},\mathcal{Z},f(X))]. | 
 | 
 

 
 
 Explanation Intersection. 
Olatunjii et al.  ( Olatunji
et al., 2023 ) measures the percentage of bits in the original explanation that is retained in the privatised explanation after using differential privacy  ( Funke
et al., 2022 ) .

 
 
 

#### 6.3.3. Privacy degree

 
 Degree of privacy refers to the level of privacy protection, which can be measured in different aspects.

 
 
 k-anonymity degree. 
 k k -anonymity refers to the number of individuals in the training dataset to whom a given explanation could potentially be linked  ( Goethals
et al., 2023 ) . This concept is grounded in the principle of k-anonymity, which ensures that a person’s information is indistinguishable from at least k-1 other individuals.

 
 
 Information leakage. 
For a sequence of queries 𝐳 𝟏 , 𝐳 𝟐 , … , 𝐳 𝐤 \mathbf{z_{1}},\mathbf{z_{2}},\ldots,\mathbf{z_{k}} , the algorithm is ( ε ^ , δ ^ \hat{\varepsilon},\hat{\delta} )-differentially private if the probability ratio of generating an explanation for any of the queries is bounded by e ε ^ e^{\hat{\varepsilon}} times the probability of the explanation under a differentially private model f f , plus a term δ ^ \hat{\delta}   ( Patel
et al., 2022 ) :

 

 
 | 
 P r i = 1 . . k ϕ ^ ( 𝐳 𝐢 , X , f D ( X ) ) ≤ e ε ^ ⋅ P r [ ϕ ^ ( 𝐳 𝐢 , X , f D ′ ( X ) ) : ∀ i ] + δ ^ , Pr_{i=1..k}\hat{\phi}(\mathbf{z_{i}},X,f_{D}(X))\leq e^{\hat{\varepsilon}}\cdot Pr[\hat{\phi}(\mathbf{z_{i}},X,f^{\prime}_{D}(X)):\forall i]+\hat{\delta}, | 
 | 
 

 where ε ^ ≤ ε \hat{\varepsilon}\leq\varepsilon and δ ^ ≤ δ \hat{\delta}\leq\delta , and at least one of the inequalities is strict.
Intuitively, this means that even if an adversary has access to the model explanations, they would not gain any additional information that could help in inferring something about the training data beyond what could be learned from the model predictions alone.

 
 
 Privacy budget. 
Patel et al.  ( Patel
et al., 2022 ) measures the allocation of a privacy budget for an explanation dataset that comprises a sequence of queries. The total privacy budget for all queries is fixed at ( ε , δ \varepsilon,\delta ), with a stricter privacy requirement ( ε m ​ i ​ n , δ m ​ i ​ n \varepsilon_{min},\delta_{min} ) set for each individual query to prevent significant information leakage. The explanation algorithm must ensure global privacy adherence by not exceeding the overall privacy budget across all queries. This means that the probability of the algorithm providing explanations within certain sets S 1 , S 2 , … , S k S_{1},S_{2},\ldots,S_{k} should be less than or equal to the product of e ε e^{\varepsilon} and the probability of these explanations under a differentially private algorithm, plus δ \delta . Furthermore, for every individual query 𝐳 𝐣 \mathbf{z_{j}} , the probability should be within e ε m ​ i ​ n e^{\varepsilon_{min}} times the differentially private algorithm probability plus δ m ​ i ​ n \delta_{min} . The goal is to create an explanation algorithm that can address as many queries as possible without exceeding the designated privacy budget and while still providing quality assurances.

 
 
 

#### 6.3.4. Attack success

 
 Measuring the success of privacy attacks is a cornerstone to evaluate the effectiveness of designed attacks, which in turn reflect the risk of a given XAI system.

 
 
 Precision/Recall/F1. 
In terms of attribute reference attacks  ( Duddu and Boutet, 2022 ) , Precision is the percentage of the positive attributes inferred by an attack being indeed positive according to the ground truth. Recall is the percentage of relevant instances of positive attributes being identified by an attack. Lastly, the F1 Score is the harmonic mean of precision and recall, calculated as 2 × precision × recall precision + recall 2\times\frac{\text{precision}\times\text{recall}}{\text{precision}+\text{recall}} , which balances precision and recall; it reaches its best value at 1 (perfect precision and recall) and worst at 0, when either precision or recall is zero.

 
 
 Balanced accuracy (BA). 
This metric measures the accuracy of attack (e.g. membership inference), on a balanced dataset of members and non-members  ( Pawelczyk
et al., 2023 ; Liu
et al., 2024d ) :

 

 
 | 
 B ​ A = T ​ P ​ R + T ​ N ​ R 2 BA=\frac{TPR+TNR}{2} | 
 | 
 

 where TPR is true-positive rate (true membership prediction) and TNR is true-negative rate (true non-membership prediction).

 
 
 ROC/AUC. 
ROC (Receiver Operating Characteristic) curve and AUC (Area Under the Curve) are metrics adapted from machine learning to measure the success of privacy attacks, such as re-identification or membership inference attacks  ( Pawelczyk
et al., 2023 ) . The ROC curve plots the TPR against the FPR at various threshold settings, providing a visual representation of an attack’s ability to distinguish between different classes (e.g., members vs. non-members in a dataset). The AUC, a single value derived from ROC, quantifies the overall effectiveness of the attack across all thresholds  ( Huang
et al., 2023 ) .

 
 
 TPR at Low FPR. 
TPR at Low FPR  ( Liu
et al., 2024d ; Huang
et al., 2023 ) is used to measure attack performance at a fixed FPR (e.g., 0.1%).
Evaluating the True Positive Rate (TPR) at low False Positive Rates (FPR) is essential in scenarios where the cost of false positives is high, because it ensures that the positive results are both accurate and reliable. Low FPR evaluation is crucial particularly in imbalanced datasets, where false positives can outnumber true positives. For example, if a membership inference attack can pinpoint even a minuscule fraction of the training dataset with high precision, then the attack ought to be deemed effective  ( Pawelczyk
et al., 2023 ) .

 
 
 Mean Absolute Error (MAE). 
Denoted as ℓ 1 \ell_{1} loss, it quantifies the average magnitude of the errors between the reconstructed inputs x ^ \hat{x} and the original inputs x x :

 

 
 | 
 ℓ 1 ​ ( x ^ , x ) = 1 m ​ n ​ ∑ j = 1 m ∑ i = 1 n | x ^ i j − x i j | , \ell_{1}(\hat{x},x)=\frac{1}{mn}\sum_{j=1}^{m}\sum_{i=1}^{n}|\hat{x}_{i}^{j}-x_{i}^{j}|, | 
 | 
 

 where m m is the number of samples in the validation dataset X val X_{\text{val}} and n n is the number of features in the dataset  ( Luo
et al., 2022 ) .

 
 
 Success Rate (SR). 
The Success Rate (SR) is defined as the ratio of the count of successfully reconstructed features to the total number of features across all samples:

 

 
 | 
 S ​ R = | X ^ v ​ a ​ l ≠ ⟂ | m ​ n , SR=\frac{|\hat{X}_{val}\neq\perp|}{mn}, | 
 | 
 

 where | X ^ v ​ a ​ l ≠ ⟂ | |\hat{X}_{val}\neq\perp| denotes the number of features that are not equal to a specific value ⟂ \perp (represents a reconstruction failure or a null value), m m is the number of samples, and n n is the number of features. This metric quantifies the portion of the dataset X v ​ a ​ l X_{val} where features are correctly reconstructed by the attack.

 
 
 Model agreement. 
In the context of model extraction attacks, Wang et al.  ( Wang
et al., 2022 ) uses agreement as a measure for comparing the behavior of a high-fidelity model h ϕ h_{\phi} to a target model f θ f_{\theta} . The agreement is defined as the average number of predictions where f θ f_{\theta} and h ϕ h_{\phi} coincide, over an evaluation set of size n n :

 

 
 | 
 Agreement = 1 n ​ ∑ i = 1 n 1 f θ ​ ( x i ) = h ϕ ​ ( x i ) . \text{Agreement}=\frac{1}{n}\sum_{i=1}^{n}1_{f_{\theta}(x_{i})=h_{\phi}(x_{i})}. | 
 | 
 

 A higher agreement indicates that the substitute model h ϕ h_{\phi} is more similar to the original model f θ f_{\theta} . When comparing two model extraction methods with the same agreement, the one with the lower standard deviation is preferred.

 
 
 Average uncertainty reduction. 
Ferry et al.  ( Ferry et al., 2023a ) evaluates the effectiveness of a data reconstruction attack. Consider a deterministic dataset 𝒟 O ​ r ​ i ​ g \mathcal{D}^{Orig} composed of n n samples each with d d features, which is used to train a machine learning model M M . Let 𝒟 M \mathcal{D}^{M} represent a probabilistic dataset that is reconstructed from M M . By its design, 𝒟 M \mathcal{D}^{M} should align with 𝒟 O ​ r ​ i ​ g \mathcal{D}^{Orig} . The degree to which the reconstruction is accurate is measured by the reduction in uncertainty across all features of all samples in the dataset, on average:

 

 
 | 
 D ​ i ​ s ​ t ​ ( 𝒟 M , 𝒟 O ​ r ​ i ​ g ) = 1 n ⋅ d ​ ∑ i = 1 n ∑ k = 1 d H ⁡ ( 𝒟 i , k M ) H ⁡ ( 𝒟 i , k ) Dist(\mathcal{D}^{M},\mathcal{D}^{Orig})=\frac{1}{n\cdot d}\sum_{i=1}^{n}\sum_{k=1}^{d}\frac{H(\mathcal{D}^{M}_{i,k})}{H(\mathcal{D}_{i,k})} | 
 | 
 

 Here, the random variable 𝒟 i , k \mathcal{D}_{i,k} symbolizes an uninformed reconstruction, evenly distributed across all conceivable values of feature k k of attribute a k a_{k} , and H H denotes the Shannon entropy.
Lower values of D ​ i ​ s ​ t ​ ( 𝒟 M , 𝒟 O ​ r ​ i ​ g ) Dist(\mathcal{D}^{M},\mathcal{D}^{Orig}) reflect superior reconstruction attacks.

 
 
 
 
 

## 7. Future Research Directions

 

### 7.1. Ethical Implications

 
 The push for explainable AI has led to the development of tools and startups like MS InterpretML, Fiddler Explainable AI Engine, IBM Explainability 360, Facebook Captum AI, and H2O Driverless AI  ( Gade et al., 2019 ) . Our survey explores the privacy risks of making ML models explainable, highlighting the potential for malicious exploitation of these explanations, especially for high-risk data such as medical records and financial transactions. This raises concerns about the conflict between the right to explain ML models  ( Goodman and
Flaxman, 2017 ) and user privacy, necessitating discussions involving legal experts and policymakers  ( Banisar, 2011 ) . Additionally, the tension between explainability and privacy may disproportionately impact minority groups by either exposing their data or providing lower-quality explanations  ( Shokri
et al., 2021 ) .

 
 
 This survey contributes to a broader research agenda on AI transparency and privacy, sparking discussions among scholars focused on AI governance. Although the trade-off between privacy and explainability is not a novel issue in legal discussions  ( Kaur et al., 2020 ) ; we remain hopeful about developing explanation methodologies that safeguard user privacy, albeit potentially at the expense of explanation quality.
While explanation quality is subjective, one thing is clear: explanations that fail to reveal useful model insights while protecting user data are likely less beneficial to end-users  ( Shokri
et al., 2021 ) .

 
 
 Looking into the future, the ethical implications of privacy-preserving techniques include balancing privacy protection with transparency and fairness  ( Hu
et al., 2022a ) . Techniques like differential privacy and federated learning secure data by adding noise or decentralising processing, but they can reduce model accuracy and transparency, complicating trust and understanding  ( Liu
et al., 2024c ; Liu
et al., 2024b ) . These methods can also introduce biases, affecting certain groups disproportionately and amplifying discrimination  ( Mi
et al., 2024 ) . Ensuring informed consent and user autonomy is crucial, necessitating clear communication about how these techniques impact data use and model performance  ( Zhang
et al., 2024 ) .

 
 
 

### 7.2. Regulatory Compliance

 
 Privacy attacks on model explanations pose significant challenges under regulatory frameworks like the GDPR, which emphasise the protection of personal data and transparency in automated decision-making. Such attacks can lead to unauthorised data disclosure, complicating compliance with GDPR’s requirements for data subject rights, including access and erasure  ( Nguyen et al., 2022 ; Huynh et al., 2024 ) . Additionally, privacy-preserving techniques that obscure model explanations may hinder transparency, making it difficult for organisations to demonstrate compliance and for individuals to understand AI decisions, thereby affecting accountability  ( Liu
et al., 2024a ) . Moreover, these techniques must balance privacy and utility, as overly restrictive measures can impact the effectiveness and fairness of AI systems, posing further challenges for legal and ethical standards  ( Zhang
et al., 2024 ) .

 
 
 

### 7.3. Privacy Tradeoffs

 
 Li et al.  ( Li
et al., 2023 ) discusses the impact of differential privacy on the interpretability of deep neural networks.
It examines how injected noise into the model parameters affects the gradient-based interpretability method. The analysis reveals that while noise in the fully connected layer directly affects the feature map used for interpretability, noise in the convolutional layer alters the output of the activation function, thus impacting the feature map indirectly.
Chang et al.  ( Chang and Shokri, 2021 ) examines the relationship between algorithmic fairness and privacy. It points out that while fair machine learning models strive to reduce discrimination by equalising behaviour across different groups, this process can alter the influence of training data points on the model, leading to uneven changes in information leakage. Fair algorithms may inadvertently memorise and leak more information about under-represented subgroups in an attempt to equalise errors across different groups based on protected attributes.
The findings indicate a trade-off where achieving fairness for protected or unprivileged groups amplifies their privacy risks. Moreover, the greater the initial bias in the training data, the higher the privacy cost when making the model fair for these groups. These findings are relevant to model explanations, which also impact fairness  ( Dodge
et al., 2019 ; Zhang and
Bareinboim, 2018 ) .

 
 
 

### 7.4. Underexplored Privacy Attacks

 
 Aivodji et al.  ( Aïvodji et al., 2022 ) present techniques for manipulating and detecting manipulation of SHAP values. To manipulate SHAP values, a brute-force sub-sampling method is used to minimise the differences in SHAP values, with a clever re-weighting strategy to make the sampling appear legitimate.
Detection of such manipulation employs statistical tests to compare model outputs from manipulated and unmanipulated samples  ( Frye et al., 2021 ) .
Slack et al.  ( Slack et al., 2020 ) outlines a framework for constructing adversarial classifiers that deceive post hoc explanation techniques, such as LIME and SHAP. The framework produces an adversarial classifier that mimics the biased classifier on real distribution data but reverts to unbiased predictions on out-of-distribution (OOD) data  ( Mittelstadt
et al., 2019 ) .
Regarding data reconstruction attacks, an interesting direction is to utilize the inner workings of learning algorithms in some interpretable models (e.g. decision tree) to reduce the entropy of probabilistically reconstructed datasets. For example, since greedy algorithms for constructing decision trees select features based on Gini impurity, we can identify and discard certain attribute combinations that do not contribute to an optimal decision tree  ( Ferry et al., 2023a ) .

 
 
 

### 7.5. Underexplored Model Explanations

 
 Gillenwater et al.  ( Gillenwater
et al., 2021 ) introduces a novel method for computing multiple quantiles in sensitive data with differential privacy. Traditional methods compromise on accuracy by either splitting the privacy budget across quantiles or inefficiently summarizing the entire distribution. The proposed approach uses an exponential mechanism to estimate multiple quantiles efficiently, achieving better accuracy and efficiency compared to existing methods.
This is particularly relevant because there are emerging explainability measures based on quantiles  ( Ghosh
et al., 2022 ; Li and van
Leeuwen, 2023 ; Merz et al., 2022 ) .
Alvarez et al.  ( Alvarez Melis and
Jaakkola, 2018 ) proposes the concept of self-explaining models that incorporate interpretability from the onset of learning. The authors design self-explaining models in a stepwise manner, starting from simple linear classifiers and advancing to more complex structures with built-in interpretability  ( Zhang
et al., 2022 ) . They introduce specialized regularization techniques to maintain faithfulness and stability.
Olatunji et al.  ( Olatunji
et al., 2023 ) pioneer the examination of privacy risks tied to feature explanations in graph neural networks (GNNs), presenting scenarios where adversaries attempt to unveil hidden relationships within the data, despite having limited access to the network’s structure  ( Khosla, 2022 ) . The paper delves into various explanation methods for GNNs such as gradient-based, perturbation-based, and surrogate methods. Furthermore, it outlines potential adversarial attacks aimed at exploiting these explanations to compromise privacy and introduces a novel defense mechanism based on perturbing explanation bits to adhere to differential privacy standards.
Other works  ( Tiddi and
Schlobach, 2022 ; Rajabi and
Etminani, 2022 ) examine the role of knowledge graphs as model explanations, positing that integrating structured, domain-specific knowledge can lead to more understandable, insightful, and trustworthy AI systems. However, knowledge graphs can be used to fuel privacy attacks such as de-anonymisation and membership inference  ( Qian
et al., 2017 ; Wang
et al., 2021 ) .

 
 
 

### 7.6. Underexplored Data Modalities

 
 Graph Data. 
The rapid development in the area of graph neural networks (GNNs)  ( Huynh et al., 2021 ; Duong et al., 2022 ; Nguyen et al., 2014 ; Nguyen et al., 2015b ; Hung et al., 2019 ) highlights a special treatment for GNN explainability  ( Wu
et al., 2020 ) .
Yuan et al.  ( Yuan
et al., 2022 ) discuss explainability methods specifically designed for Graph Neural Networks (GNNs) such as gradients/features-based, perturbation-based, surrogate, and decomposition methods. Prado et al.  ( Prado-Romero et al., 2023 ) provides a comprehensive overview of graph counterfactual explanations for GNNs. Privacy attacks on GNNs are also an emerging direction  ( Dai et al., 2022 ) .

 
 
 Audio Data. 
Audio signals consists of speech signals and other non-speech audio signals. Speech processing involves tasks like automatic speech recognition, speaker identification, and paralinguistic information recognition, while non-speech audio signal processing contains many more applications, such as human heart sound analysis, bird sound analysis, and environmental sound classification. Current research have separately focused on data / model privacy and explanation approaches  ( Ren
et al., 2023 ; Li
et al., 2021 ; Carlini and
Wagner, 2018 ; Abdullah et al., 2021 ) . While explainable models are essential for audio-based healthcare applications  ( Ren et al., 2022 ; Ren
et al., 2020 ; Chang
et al., 2022 ) , there is still a large gap to further explore the privacy risks of audio-based model explanations.

 
 
 

### 7.7. Privacy-Preserving Models

 
 Exploring how privacy-preserving models, such as differentially private decision trees, reduce the success of privacy attacks represents a valuable research direction  ( Ferry et al., 2023a ) . Li et al.  ( Li
et al., 2023 ) presents an Adaptive Differential Privacy (ADP) mechanism aimed at improving the interpretability of machine learning models without compromising privacy. This mechanism selectively injects noise into the less critical weights of a model’s parameters, thereby preserving the interpretability of important features which conventional differential privacy methods may obscure.

 
 
 

### 7.8. Privacy-Protecting Explanations

 
 Using model explanations to counter adversarial attacks is a novel direction. Belhadj et al.  ( Belhadj-Cheikh et al., 2021 ) outlines a framework (called FOX) to safeguard social media users’ privacy by using adversarial reactions to trick classifiers. It constructs a dataset of social media interactions, employs an explainability tool to extract influential adversarial features, and filters them to create a robust list. These features are then used to generate adversarial reactions, aiming to mislead the classifier away from the correct classification and towards a predetermined label, thus preserving the user’s privacy.

 
 
 

### 7.9. Time Complexity

 
 Time complexity is crucial in privacy attacks on model explanations. Fast run-time methods pose higher risks by enabling rapid exploitation, while more complex iterative attacks are less practical due to longer execution times. The feasibility of these attacks depends on computational resources and scalability. Effective countermeasures must balance protection and performance to mitigate risks from fast, real-time attacks. Unfortunately, only a few works thoroughly discuss time complexity such as Shapley approximation  ( Jia et al., 2019a ) and DP-quantiles  ( Gillenwater
et al., 2021 ) .

 
 
 
 

## 8. Conclusions

 
 Summary. 
As the prevalence of model explanations grows, there is an emerging interest in understanding its repercussions, including aspects of fidelity, fairness, stability, and privacy. This survey offers a thorough investigation into the latest privacy-centric attacks on model explanations, establishing a comprehensive classification of these attacks based on their traits. Furthermore, it delves deeply into the present advanced research on defensive strategies and privacy-focused model explanations, uncovering common privacy design approaches and their variations.

 
 
 Our survey also highlights several unresolved issues that demand additional inquiry. Primarily, it points out the current research’s limited scope, which predominantly focuses on membership inference attacks, counterfactual explanations, and differential privacy. It suggests that numerous widely-used algorithms and models, in terms of their real-world implementation and relevance, deserve more detailed scrutiny. Secondly, there’s a noticeable lack of deep theoretical insight into the origins of privacy breaches, impacting both the development of protective measures and the comprehension of privacy attack limitations. Although experimental research into the determinants of privacy breaches has yielded valuable knowledge, there’s a scarcity of studies evaluating attacks under realistic conditions, considering dataset size and actual deployment. As the field continues to explore the privacy implications of model explanations, this survey aims to serve as a crucial resource for interested readers eager to contribute to this trend.

 
 
 Challenges. 
The challenges for new work in this field, as highlighted in the survey, include:

 
 • 
 
 Balancing Transparency and Privacy: Providing detailed explanations improves transparency but increases the risk of privacy breaches by revealing sensitive information embedded in the training data.

 

 • 
 
 Granularity of Explanations: Detailed explanations can lead to direct inferences about data points, making it challenging to protect privacy without losing interpretability.

 

 • 
 
 Understanding Privacy Leaks: Identifying the causes of privacy leaks through model explanations is complex and requires thorough investigation of different explanation methods and their vulnerabilities.

 

 • 
 
 Diverse Attack Models: Developing comprehensive defenses against a wide range of privacy attacks, including membership inference, model inversion, and reconstruction attacks, is necessary but challenging due to the evolving nature of these attacks.

 

 • 
 
 Countermeasure Effectiveness: Evaluating and improving the effectiveness of countermeasures, such as differential privacy and perturbation techniques, to ensure they do not compromise the utility of model explanations.

 

 • 
 
 Dynamic Interaction Scenarios: Assessing the impact of repeated interactions between adversaries and the model in dynamic settings adds complexity to designing robust privacy-preserving methods.

 

 • 
 
 Interpretable Surrogates: Surrogate models used for providing explanations can themselves become targets for privacy attacks, necessitating additional safeguards.

 

 • 
 
 Scalability and Practicality: Implementing privacy-preserving techniques in real-world must balance scalability and practicality without significantly affecting model performance.

 

 
 
 
 

## References

 
 
 Abadi et al . (2016) 
 
Martin Abadi, Andy Chu,
Ian Goodfellow, H Brendan McMahan,
Ilya Mironov, Kunal Talwar, and
Li Zhang. 2016.

 
 Deep learning with differential privacy. In
 CCS . 308–318.

 
 
 

 
 Abdukhamidov et al . (2023) 
 
Eldor Abdukhamidov,
Mohammed Abuhamad, Simon S Woo,
Eric Chan-Tin, and Tamer Abuhmed.
2023.

 
 Hardening Interpretable Deep Learning Systems:
Investigating Adversarial Threats and Defenses.

 
 TDSC (2023).

 
 
 

 
 Abdullah et al . (2021) 
 
Hadi Abdullah, Kevin
Warren, Vincent Bindschaedler, Nicolas
Papernot, and Patrick Traynor.
2021.

 
 Sok: The faults in our asrs: An overview of attacks
against automatic speech recognition and speaker identification systems. In
 SP . 730–747.

 
 
 

 
 Adadi and Berrada (2018) 
 
Amina Adadi and Mohammed
Berrada. 2018.

 
 Peeking inside the black-box: a survey on
explainable artificial intelligence (XAI).

 
 IEEE access 6
(2018), 52138–52160.

 
 
 

 
 Aïvodji
et al . (2020) 
 
Ulrich Aïvodji,
Alexandre Bolot, and Sébastien
Gambs. 2020.

 
 Model extraction from counterfactual explanations.

 
 arXiv preprint arXiv:2009.01884 
(2020).

 
 
 

 
 Aïvodji et al . (2022) 
 
Ulrich Aïvodji,
Satoshi Hara, Mario Marchand,
Foutse Khomh, et al . 
2022.

 
 Fooling SHAP with Stealthily Biased Sampling. In
 ICLR .

 
 
 

 
 Alvarez Melis and
Jaakkola (2018) 
 
David Alvarez Melis and
Tommi Jaakkola. 2018.

 
 Towards robust interpretability with
self-explaining neural networks.

 
 NeurIPS 31
(2018).

 
 
 

 
 Ancona
et al . (2018) 
 
Marco Ancona, Enea
Ceolini, Cengiz Oztireli, and Markus
Gross. 2018.

 
 Towards better understanding of gradient-based
attribution methods for Deep Neural Networks. In
 ICLR .

 
 
 

 
 Angelov and
Soares (2020a) 
 
Plamen Angelov and
Eduardo Soares. 2020a.

 
 Towards deep machine reasoning: a prototype-based
deep neural network with decision tree inference. In
 SMC . 2092–2099.

 
 
 

 
 Angelov and
Soares (2020b) 
 
Plamen Angelov and
Eduardo Soares. 2020b.

 
 Towards explainable deep neural networks (xDNN).

 
 Neural Networks 130
(2020), 185–194.

 
 
 

 
 Artelt and Hammer (2020) 
 
André Artelt and
Barbara Hammer. 2020.

 
 Convex density constraints for computing plausible
counterfactual explanations. In ICANN .
353–365.

 
 
 

 
 Artelt et al . (2021) 
 
André Artelt, Valerie
Vaquet, Riza Velioglu, Fabian Hinder,
Johannes Brinkrolf, Malte Schilling,
and Barbara Hammer. 2021.

 
 Evaluating robustness of counterfactual
explanations. In SSCI . 01–09.

 
 
 

 
 Bach et al . (2015) 
 
Sebastian Bach, Alexander
Binder, Grégoire Montavon, Frederick
Klauschen, Klaus-Robert Müller, and
Wojciech Samek. 2015.

 
 On pixel-wise explanations for non-linear
classifier decisions by layer-wise relevance propagation.

 
 PloS one 10,
7 (2015), e0130140.

 
 
 

 
 Baniecki and
Biecek (2024) 
 
Hubert Baniecki and
Przemyslaw Biecek. 2024.

 
 Adversarial attacks and defenses in explainable
artificial intelligence: A survey.

 
 Information Fusion (2024),
102303.

 
 
 

 
 Banisar (2011) 
 
David Banisar.
2011.

 
 The right to information and privacy: balancing
rights and managing conflicts.

 
 World Bank Institute Governance Working
Paper (2011).

 
 
 

 
 Barocas
et al . (2020) 
 
Solon Barocas, Andrew D
Selbst, and Manish Raghavan.
2020.

 
 The hidden assumptions behind counterfactual
explanations and principal reasons. In FAccT .
80–89.

 
 
 

 
 Begley
et al . (2020) 
 
Tom Begley, Tobias
Schwedes, Christopher Frye, and Ilya
Feige. 2020.

 
 Explainability for fair machine learning.

 
 arXiv preprint arXiv:2010.07389 
(2020).

 
 
 

 
 Belhadj-Cheikh et al . (2021) 
 
Noreddine Belhadj-Cheikh,
Abdessamad Imine, and Michaël
Rusinowitch. 2021.

 
 FOX: Fooling with Explanations: Privacy Protection
with Adversarial Reactions in Social Media. In
 PST . 1–10.

 
 
 

 
 Biggio and Roli (2018) 
 
Battista Biggio and
Fabio Roli. 2018.

 
 Wild patterns: Ten years after the rise of
adversarial machine learning. In CCS .
2154–2156.

 
 
 

 
 Binns et al . (2018) 
 
Reuben Binns, Max
Van Kleek, Michael Veale, Ulrik Lyngs,
Jun Zhao, and Nigel Shadbolt.
2018.

 
 ’It’s Reducing a Human Being to a Percentage’
Perceptions of Justice in Algorithmic Decisions. In
 CHI . 1–14.

 
 
 

 
 Bodria et al . (2023) 
 
Francesco Bodria, Fosca
Giannotti, Riccardo Guidotti, Francesca
Naretto, Dino Pedreschi, and Salvatore
Rinzivillo. 2023.

 
 Benchmarking and survey of explanation methods for
black box models.

 
 Data Min. Knowl. Discov. 
(2023), 1–60.

 
 
 

 
 Bossard
et al . (2014) 
 
Lukas Bossard, Matthieu
Guillaumin, and Luc Van Gool.
2014.

 
 Food-101–mining discriminative components with
random forests. In ECCV .
446–461.

 
 
 

 
 Brughmans
et al . (2023) 
 
Dieter Brughmans, Pieter
Leyman, and David Martens.
2023.

 
 Nice: an algorithm for nearest instance
counterfactual explanations.

 
 Data Min. Knowl. Discov. 
(2023), 1–39.

 
 
 

 
 Bu
et al . (2023) 
 
Zhiqi Bu, Yu-Xiang Wang,
Sheng Zha, and George Karypis.
2023.

 
 Differentially private optimization on large model
at small cost. In ICML .
3192–3218.

 
 
 

 
 Carlini et al . (2022) 
 
Nicholas Carlini, Steve
Chien, Milad Nasr, Shuang Song,
Andreas Terzis, and Florian Tramer.
2022.

 
 Membership inference attacks from first
principles. In SP . 1897–1914.

 
 
 

 
 Carlini et al . (2019) 
 
Nicholas Carlini, Chang
Liu, Úlfar Erlingsson, Jernej Kos,
and Dawn Song. 2019.

 
 The secret sharer: Evaluating and testing
unintended memorization in neural networks. In
 USENIX . 267–284.

 
 
 

 
 Carlini and
Wagner (2018) 
 
Nicholas Carlini and
David Wagner. 2018.

 
 Audio adversarial examples: Targeted attacks on
speech-to-text. In SPW . 1–7.

 
 
 

 
 Chang and Shokri (2021) 
 
Hongyan Chang and Reza
Shokri. 2021.

 
 On the privacy risks of algorithmic fairness. In
 EuroS P . 292–303.

 
 
 

 
 Chang
et al . (2022) 
 
Yi Chang, Zhao Ren,
Thanh Tam Nguyen, Wolfgang Nejdl, and
Björn W Schuller. 2022.

 
 Example-based Explanations with Adversarial Attacks
for Respiratory Sound Analysis. In Interspeech .
1–5.

 
 
 

 
 Chaudhuri
et al . (2011) 
 
Kamalika Chaudhuri, Claire
Monteleoni, and Anand D Sarwate.
2011.

 
 Differentially private empirical risk
minimization.

 
 JMLR 12,
3 (2011).

 
 
 

 
 Chen
et al . (2019) 
 
Chaofan Chen, Oscar Li,
Daniel Tao, Alina Barnett,
Cynthia Rudin, and Jonathan K Su.
2019.

 
 This looks like that: deep learning for
interpretable image recognition.

 
 NeurIPS 32
(2019).

 
 
 

 
 Chen
et al . (2018a) 
 
Jiawei Chen, Janusz
Konrad, and Prakash Ishwar.
2018a.

 
 Vgan-based image representation learning for
privacy-preserving facial expression recognition. In
 CVPR workshops . 1570–1579.

 
 
 

 
 Chen
et al . (2018b) 
 
Jianbo Chen, Le Song,
Martin Wainwright, and Michael Jordan.
2018b.

 
 Learning to explain: An information-theoretic
perspective on model interpretation. In ICML .
883–892.

 
 
 

 
 Chen
et al . (2020) 
 
Zhi Chen, Yijie Bei,
and Cynthia Rudin. 2020.

 
 Concept whitening for interpretable image
recognition.

 
 Nature Machine Intelligence 
2, 12 (2020),
772–782.

 
 
 

 
 Craven and
Shavlik (1994) 
 
Mark W Craven and Jude W
Shavlik. 1994.

 
 Using sampling and queries to extract rules from
trained neural networks.

 
 In Machine learning proceedings .
Elsevier, 37–45.

 
 
 

 
 Dai et al . (2022) 
 
Enyan Dai, Tianxiang
Zhao, Huaisheng Zhu, Junjie Xu,
Zhimeng Guo, Hui Liu,
Jiliang Tang, and Suhang Wang.
2022.

 
 A comprehensive survey on trustworthy graph neural
networks: Privacy, robustness, fairness, and explainability.

 
 arXiv preprint arXiv:2204.08570 
(2022).

 
 
 

 
 Datta
et al . (2016) 
 
Anupam Datta, Shayak Sen,
and Yair Zick. 2016.

 
 Algorithmic transparency via quantitative input
influence: Theory and experiments with learning systems. In
 SP . 598–617.

 
 
 

 
 Deng (2019) 
 
Houtao Deng.
2019.

 
 Interpreting tree ensembles with intrees.

 
 JDSA 7,
4 (2019), 277–287.

 
 
 

 
 Dhurandhar et al . (2018) 
 
Amit Dhurandhar, Pin-Yu
Chen, Ronny Luss, Chun-Chen Tu,
Paishun Ting, Karthikeyan Shanmugam,
and Payel Das. 2018.

 
 Explanations based on the missing: Towards
contrastive explanations with pertinent negatives.

 
 NeurIPS 31
(2018).

 
 
 

 
 Dodge
et al . (2019) 
 
Jonathan Dodge, Q Vera
Liao, Yunfeng Zhang, Rachel KE Bellamy,
and Casey Dugan. 2019.

 
 Explaining models: an empirical study of how
explanations impact fairness judgment. In IUI .
275–285.

 
 
 

 
 Domingo-Ferrer et al . (2019) 
 
Josep Domingo-Ferrer,
Cristina Pérez-Solà, and Alberto
Blanco-Justicia. 2019.

 
 Collaborative explanation of deep models with
limited interaction for trade secret and privacy preservation. In
 WWW Companion . 501–507.

 
 
 

 
 Došilović et al . (2018) 
 
Filip Karlo Došilović,
Mario Brčić, and Nikica
Hlupić. 2018.

 
 Explainable artificial intelligence: A survey. In
 MIPRO . 0210–0215.

 
 
 

 
 Dosovitskiy and
Brox (2016) 
 
Alexey Dosovitskiy and
Thomas Brox. 2016.

 
 Inverting visual representations with convolutional
networks. In CVPR . 4829–4837.

 
 
 

 
 Duddu and Boutet (2022) 
 
Vasisht Duddu and
Antoine Boutet. 2022.

 
 Inferring Sensitive Attributes from Model
Explanations. In CIKM .
416–425.

 
 
 

 
 Dumoulin and
Visin (2016) 
 
Vincent Dumoulin and
Francesco Visin. 2016.

 
 A guide to convolution arithmetic for deep
learning.

 
 arXiv preprint arXiv:1603.07285 
(2016).

 
 
 

 
 Duong et al . (2022) 
 
Chi Thang Duong, Thanh Tam
Nguyen, Trung-Dung Hoang, Hongzhi Yin,
Matthias Weidlich, and Quoc Viet Hung
Nguyen. 2022.

 
 Deep MinCut: Learning Node Embeddings from
Detecting Communities.

 
 Pattern Recognition (2022),
109126.

 
 
 

 
 Dwork
et al . (2014) 
 
Cynthia Dwork, Aaron
Roth, et al . 2014.

 
 The algorithmic foundations of differential
privacy.

 
 Found. Trends Theor. Comput. Sci. 
9, 3–4 (2014),
211–407.

 
 
 

 
 Dwork
et al . (2017) 
 
Cynthia Dwork, Adam
Smith, Thomas Steinke, and Jonathan
Ullman. 2017.

 
 Exposed! a survey of attacks on private data.

 
 Annu. Rev. Stat. Appl. 4
(2017), 61–84.

 
 
 

 
 Ferry (2023) 
 
Julien Ferry.
2023.

 
 Addresing interpretability fairness privacy
in machine learning through combinatorial optimization methods .

 
 Ph. D. Dissertation.
Université Paul Sabatier-Toulouse III.

 
 
 

 
 Ferry et al . (2023a) 
 
Julien Ferry, Ulrich
Aïvodji, Sébastien Gambs,
Marie-José Huguet, and Mohamed
Siala. 2023a.

 
 Probabilistic dataset reconstruction from
interpretable models.

 
 arXiv preprint arXiv:2308.15099 
(2023).

 
 
 

 
 Ferry et al . (2023b) 
 
Julien Ferry, Ulrich
Aïvodji, Sébastien Gambs,
Marie-José Huguet, and Mohamed
Siala. 2023b.

 
 SoK: Taming the Triangle–On the Interplays between
Fairness, Interpretability and Privacy in Machine Learning.

 
 arXiv preprint arXiv:2312.16191 
(2023).

 
 
 

 
 Fredrikson
et al . (2015) 
 
Matt Fredrikson, Somesh
Jha, and Thomas Ristenpart.
2015.

 
 Model inversion attacks that exploit confidence
information and basic countermeasures. In CCS .
1322–1333.

 
 
 

 
 Frye et al . (2021) 
 
Christopher Frye, Damien
de Mijolla, Tom Begley, Laurence Cowton,
Megan Stanley, and Ilya Feige.
2021.

 
 Shapley explainability on the data manifold. In
 ICLR .

 
 
 

 
 Funke
et al . (2022) 
 
Thorben Funke, Megha
Khosla, Mandeep Rathee, and Avishek
Anand. 2022.

 
 Zorro: Valid, sparse, and stable explanations in
graph neural networks.

 
 TKDE (2022).

 
 
 

 
 Gade et al . (2019) 
 
Krishna Gade, Sahin Cem
Geyik, Krishnaram Kenthapadi, Varun
Mithal, and Ankur Taly.
2019.

 
 Explainable AI in industry. In
 KDD . 3203–3204.

 
 
 

 
 Gambs
et al . (2012) 
 
Sébastien Gambs, Ahmed
Gmati, and Michel Hurfin.
2012.

 
 Reconstruction attack through classifier analysis.
In DBSec . 274–281.

 
 
 

 
 Ganju
et al . (2018) 
 
Karan Ganju, Qi Wang,
Wei Yang, Carl A Gunter, and
Nikita Borisov. 2018.

 
 Property inference attacks on fully connected
neural networks using permutation invariant representations. In
 CCS . 619–633.

 
 
 

 
 Garcia
et al . (2018) 
 
Washington Garcia,
Joseph I Choi, Suman K Adari,
Somesh Jha, and Kevin RB Butler.
2018.

 
 Explainable black-box attacks against model-based
authentication.

 
 arXiv preprint arXiv:1810.00024 
(2018).

 
 
 

 
 Garfinkel
et al . (2019) 
 
Simson Garfinkel, John M
Abowd, and Christian Martindale.
2019.

 
 Understanding database reconstruction attacks on
public data.

 
 CACM 62,
3 (2019), 46–53.

 
 
 

 
 Gaudio et al . (2023) 
 
Alex Gaudio, Asim
Smailagic, Christos Faloutsos, Shreshta
Mohan, Elvin Johnson, Yuhao Liu,
Pedro Costa, and Aurélio Campilho.
2023.

 
 DeepFixCX: Explainable privacy-preserving image
compression for medical image analysis.

 
 WIREs DMKD (2023),
e1495.

 
 
 

 
 Ghosh
et al . (2022) 
 
Avijit Ghosh, Aalok
Shanbhag, and Christo Wilson.
2022.

 
 Faircanary: Rapid continuous explainable fairness.
In AIES . 307–316.

 
 
 

 
 Gillenwater
et al . (2021) 
 
Jennifer Gillenwater,
Matthew Joseph, and Alex Kulesza.
2021.

 
 Differentially private quantiles. In
 ICML . 3713–3722.

 
 
 

 
 Gilpin et al . (2018) 
 
Leilani H Gilpin, David
Bau, Ben Z Yuan, Ayesha Bajwa,
Michael Specter, and Lalana Kagal.
2018.

 
 Explaining explanations: An overview of
interpretability of machine learning. In DSAA .
80–89.

 
 
 

 
 Goethals
et al . (2023) 
 
Sofie Goethals, Kenneth
Sörensen, and David Martens.
2023.

 
 The privacy issue of counterfactual explanations:
explanation linkage attacks.

 
 TIST 14,
5 (2023), 1–24.

 
 
 

 
 Goodman and
Flaxman (2017) 
 
Bryce Goodman and Seth
Flaxman. 2017.

 
 European Union regulations on algorithmic
decision-making and a “right to explanation”.

 
 AI magazine 38,
3 (2017), 50–57.

 
 
 

 
 Guidotti (2022) 
 
Riccardo Guidotti.
2022.

 
 Counterfactual explanations and how to find them:
literature review and benchmarking.

 
 Data Min. Knowl. Discov. 
(2022), 1–55.

 
 
 

 
 Guidotti et al . (2018) 
 
Riccardo Guidotti, Anna
Monreale, Salvatore Ruggieri, Franco
Turini, Fosca Giannotti, and Dino
Pedreschi. 2018.

 
 A survey of methods for explaining black box
models.

 
 CSUR 51,
5 (2018), 1–42.

 
 
 

 
 Hamer et al . (2023) 
 
Jenny Hamer, Jake
Valladares, Vignesh Viswanathan, and
Yair Zick. 2023.

 
 Simple Steps to Success: Axiomatics of
Distance-Based Algorithmic Recourse.

 
 arXiv preprint arXiv:2306.15557 
(2023).

 
 
 

 
 Harder
et al . (2020) 
 
Frederik Harder, Matthias
Bauer, and Mijung Park.
2020.

 
 Interpretable and differentially private
predictions. In AAAI , Vol. 34.
4083–4090.

 
 
 

 
 Hashemi and Fathi (2020) 
 
Masoud Hashemi and Ali
Fathi. 2020.

 
 Permuteattack: Counterfactual explanation of
machine learning credit scorecards.

 
 arXiv preprint arXiv:2008.10138 
(2020).

 
 
 

 
 He et al . (2019) 
 
Zecheng He, Tianwei
Zhang, and Ruby B Lee. 2019.

 
 Model inversion attacks against collaborative
inference. In ACSAC . 148–162.

 
 
 

 
 Holohan et al . (2019) 
 
Naoise Holohan, Stefano
Braghin, Pól Mac Aonghusa, and
Killian Levacher. 2019.

 
 Diffprivlib: the IBM differential privacy library.

 
 arXiv preprint arXiv:1907.02444 
(2019).

 
 
 

 
 Hooker
et al . (2019) 
 
Sara Hooker, Dumitru
Erhan, Pieter-Jan Kindermans, and Been
Kim. 2019.

 
 A benchmark for interpretability methods in deep
neural networks.

 
 NeurIPS 32
(2019).

 
 
 

 
 Hu
et al . (2022b) 
 
Hongsheng Hu, Zoran
Salcic, Lichao Sun, Gillian Dobbie,
Philip S Yu, and Xuyun Zhang.
2022b.

 
 Membership inference attacks on machine learning: A
survey.

 
 CSUR 54,
11s (2022), 1–37.

 
 
 

 
 Hu
et al . (2022a) 
 
Shengshan Hu, Xiaogeng
Liu, Yechao Zhang, Minghui Li,
Leo Yu Zhang, Hai Jin, and
Libing Wu. 2022a.

 
 Protecting facial privacy: Generating adversarial
identity masks via style-robust makeup transfer. In
 CVPR . 15014–15023.

 
 
 

 
 Huang
et al . (2023) 
 
Catherine Huang, Chelse
Swoopes, Christina Xiao, Jiaqi Ma, and
Himabindu Lakkaraju. 2023.

 
 Accurate, Explainable, and Private Models:
Providing Recourse While Minimizing Training Data Leakage.

 
 arXiv preprint arXiv:2308.04341 
(2023).

 
 
 

 
 Hung et al . (2019) 
 
Nguyen Quoc Viet Hung,
Matthias Weidlich, Nguyen Thanh Tam,
Zoltán Miklós, Karl Aberer,
Avigdor Gal, and Bela Stantic.
2019.

 
 Handling probabilistic integrity constraints in
pay-as-you-go reconciliation of data models.

 
 Information Systems 83
(2019), 166–180.

 
 
 

 
 Huynh et al . (2021) 
 
Thanh Trung Huynh,
Chi Thang Duong, Thanh Tam Nguyen,
Vinh Tong Van, Abdul Sattar,
Hongzhi Yin, and Quoc Viet Hung
Nguyen. 2021.

 
 Network alignment with holistic embeddings.

 
 TKDE 35,
2 (2021), 1881–1894.

 
 
 

 
 Huynh et al . (2024) 
 
Thanh Trung Huynh,
Trong Bang Nguyen, Phi Le Nguyen,
Thanh Tam Nguyen, Matthias Weidlich,
Quoc Viet Hung Nguyen, and Karl
Aberer. 2024.

 
 Fast-FedUL: A Training-Free Federated Unlearning
with Provable Skew Resilience. In ECML PKDD .

 
 
 

 
 Jagielski et al . (2020) 
 
Matthew Jagielski,
Nicholas Carlini, David Berthelot,
Alex Kurakin, and Nicolas Papernot.
2020.

 
 High accuracy and high fidelity extraction of
neural networks. In USENIX .
1345–1362.

 
 
 

 
 Jetchev and
Vuille (2023) 
 
Dimitar Jetchev and
Marius Vuille. 2023.

 
 XorSHAP: Privacy-Preserving Explainable AI for
Decision Tree Models.

 
 Cryptology ePrint Archive 
(2023).

 
 
 

 
 Jia
et al . (2019b) 
 
Jinyuan Jia, Ahmed Salem,
Michael Backes, Yang Zhang, and
Neil Zhenqiang Gong. 2019b.

 
 Memguard: Defending against black-box membership
inference attacks via adversarial examples. In
 CCS . 259–274.

 
 
 

 
 Jia et al . (2019a) 
 
Ruoxi Jia, David Dao,
Boxin Wang, Frances Ann Hubis,
Nick Hynes, Nezihe Merve Gürel,
Bo Li, Ce Zhang, Dawn
Song, and Costas J Spanos.
2019a.

 
 Towards efficient data valuation based on the
shapley value. In AISTATS .
1167–1176.

 
 
 

 
 Joshi and Thakkar (2022) 
 
Devvrat Joshi and Janvi
Thakkar. 2022.

 
 k-Means SubClustering: A Differentially Private
Algorithm with Improved Clustering Quality. In
 CIKM .

 
 
 

 
 Karimi
et al . (2021) 
 
Amir-Hossein Karimi,
Bernhard Schölkopf, and Isabel
Valera. 2021.

 
 Algorithmic recourse: from counterfactual
explanations to interventions. In FAccT .
353–362.

 
 
 

 
 Kasirzadeh and
Smart (2021) 
 
Atoosa Kasirzadeh and
Andrew Smart. 2021.

 
 The use and misuse of counterfactuals in ethical
machine learning. In FAccT .
228–236.

 
 
 

 
 Kaur et al . (2020) 
 
Harmanpreet Kaur, Harsha
Nori, Samuel Jenkins, Rich Caruana,
Hanna Wallach, and Jennifer
Wortman Vaughan. 2020.

 
 Interpreting interpretability: understanding data
scientists’ use of interpretability tools for machine learning. In
 CHI . 1–14.

 
 
 

 
 Keane and Smyth (2020) 
 
Mark T Keane and Barry
Smyth. 2020.

 
 Good counterfactuals and where to find them: A
case-based technique for generating counterfactuals for explainable AI
(XAI). In ICCBR . 163–178.

 
 
 

 
 Kenny
et al . (2021) 
 
Eoin M Kenny, Courtney
Ford, Molly Quinn, and Mark T Keane.
2021.

 
 Explaining black-box classifiers using post-hoc
explanations-by-example: The effect of explanations and error-rates in XAI
user studies.

 
 AIJ 294
(2021), 103459.

 
 
 

 
 Kenny and Keane (2019) 
 
Eoin M. Kenny and
Mark T. Keane. 2019.

 
 Twin-Systems to Explain Artificial Neural Networks
using Case-Based Reasoning: Comparative Tests of Feature-Weighting Methods in
ANN-CBR Twins for XAI. In IJCAI .
2708–2715.

 
 
 

 
 Khosla (2022) 
 
Megha Khosla.
2022.

 
 Privacy and transparency in graph machine learning:
A unified perspective.

 
 arXiv preprint arXiv:2207.10896 
(2022).

 
 
 

 
 Kim
et al . (2014) 
 
Been Kim, Cynthia Rudin,
and Julie A Shah. 2014.

 
 The bayesian case model: A generative approach for
case-based reasoning and prototype classification.

 
 NeurIPS 27
(2014).

 
 
 

 
 Kim and Chae (2024) 
 
Seonggyeom Kim and
Dong-Kyu Chae. 2024.

 
 What Does a Model Really Look at?: Extracting
Model-Oriented Concepts for Explaining Deep Neural Networks.

 
 TPAMI (2024).

 
 
 

 
 Koh and Liang (2017) 
 
Pang Wei Koh and Percy
Liang. 2017.

 
 Understanding black-box predictions via influence
functions. In ICML . 1885–1894.

 
 
 

 
 Krizhevsky (2009) 
 
Alex Krizhevsky.
2009.

 
 Learning Multiple Layers of Features from Tiny
Images.

 
 
 

 
 Kumar et al . (2016) 
 
Srijan Kumar, Francesca
Spezzano, VS Subrahmanian, and Christos
Faloutsos. 2016.

 
 Edge weight prediction in weighted signed
networks. In ICDM . 221–230.

 
 
 

 
 Kumari
et al . (2024) 
 
Kavita Kumari, Murtuza
Jadliwala, Sumit Kumar Jha, and Anindya
Maiti. 2024.

 
 Towards a Game-theoretic Understanding of
Explanation-based Membership Inference Attacks.

 
 arXiv preprint arXiv:2404.07139 
(2024).

 
 
 

 
 Kuppa and Le-Khac (2020) 
 
Aditya Kuppa and
Nhien-An Le-Khac. 2020.

 
 Black box attacks on explainable artificial
intelligence (XAI) methods in cyber security. In
 IJCNN . 1–8.

 
 
 

 
 Kuppa and Le-Khac (2021) 
 
Aditya Kuppa and
Nhien-An Le-Khac. 2021.

 
 Adversarial xai methods in cybersecurity.

 
 TIFS 16
(2021), 4924–4938.

 
 
 

 
 Laugel et al . (2017) 
 
Thibault Laugel,
Marie-Jeanne Lesot, Christophe Marsala,
Xavier Renard, and Marcin Detyniecki.
2017.

 
 Inverse classification for comparison-based
interpretability in machine learning.

 
 arXiv preprint arXiv:1712.08443 
(2017).

 
 
 

 
 Li
et al . (2018) 
 
Oscar Li, Hao Liu,
Chaofan Chen, and Cynthia Rudin.
2018.

 
 Deep learning for case-based reasoning through
prototypes: A neural network that explains its predictions. In
 AAAI , Vol. 32.

 
 
 

 
 Li
et al . (2023) 
 
Zhe Li, Honglong Chen,
Zhichen Ni, and Huajie Shao.
2023.

 
 Balancing Privacy Protection and Interpretability
in Federated Learning.

 
 arXiv preprint arXiv:2302.08044 
(2023).

 
 
 

 
 Li
et al . (2022) 
 
Zheng Li, Yiyong Liu,
Xinlei He, Ning Yu,
Michael Backes, and Yang Zhang.
2022.

 
 Auditing membership leakages of multi-exit
networks. In CCS . 1917–1931.

 
 
 

 
 Li
et al . (2021) 
 
Zhuohang Li, Cong Shi,
Tianfang Zhang, Yi Xie,
Jian Liu, Bo Yuan, and
Yingying Chen. 2021.

 
 Robust detection of machine-induced audio attacks
in intelligent audio systems with microphone array. In
 CCS . 1884–1899.

 
 
 

 
 Li and van
Leeuwen (2023) 
 
Zhong Li and Matthijs
van Leeuwen. 2023.

 
 Explainable contextual anomaly detection using
quantile regression forests.

 
 Data Min. Knowl. Discov. 
37, 6 (2023),
2517–2563.

 
 
 

 
 Lindell (2020) 
 
Yehuda Lindell.
2020.

 
 Secure multiparty computation.

 
 CACM 64,
1 (2020), 86–96.

 
 
 

 
 Lipton (2018) 
 
Zachary C Lipton.
2018.

 
 The mythos of model interpretability: In machine
learning, the concept of interpretability is both important and slippery.

 
 Queue 16,
3 (2018), 31–57.

 
 
 

 
 Liu et al . (2021) 
 
Bo Liu, Ming Ding,
Sina Shaham, Wenny Rahayu,
Farhad Farokhi, and Zihuai Lin.
2021.

 
 When machine learning meets privacy: A survey and
outlook.

 
 CSUR 54,
2 (2021), 1–36.

 
 
 

 
 Liu
et al . (2024c) 
 
Hanyang Liu, Yong Wang,
Zhiqiang Zhang, Jiangzhou Deng,
Chao Chen, and Leo Yu Zhang.
2024c.

 
 Matrix factorization recommender based on adaptive
Gaussian differential privacy for implicit feedback.

 
 IPM 61,
4 (2024), 103720.

 
 
 

 
 Liu
et al . (2024d) 
 
Han Liu, Yuhao Wu,
Zhiyuan Yu, and Ning Zhang.
2024d.

 
 Please Tell Me More: Privacy Impact of
Explainability through the Lens of Membership Inference Attack. In
 SP . 120–120.

 
 
 

 
 Liu
et al . (2022c) 
 
Mingting Liu, Xiaozhang
Liu, Anli Yan, Yuan Qi, and
Wei Li. 2022c.

 
 Explanation-Guided Minimum Adversarial Attack. In
 ML4CS . 257–270.

 
 
 

 
 Liu
et al . (2022d) 
 
Yiyong Liu, Zhengyu Zhao,
Michael Backes, and Yang Zhang.
2022d.

 
 Membership inference attacks by exploiting loss
trajectory. In CCS . 2085–2098.

 
 
 

 
 Liu
et al . (2022a) 
 
Ziyao Liu, Jiale Guo,
Kwok-Yan Lam, and Jun Zhao.
2022a.

 
 Efficient dropout-resilient aggregation for
privacy-preserving machine learning.

 
 TIFS 18
(2022), 1839–1854.

 
 
 

 
 Liu
et al . (2022b) 
 
Ziyao Liu, Jiale Guo,
Wenzhuo Yang, Jiani Fan,
Kwok-Yan Lam, and Jun Zhao.
2022b.

 
 Privacy-preserving aggregation in federated
learning: A survey.

 
 IEEE Transactions on Big Data 
(2022).

 
 
 

 
 Liu
et al . (2024a) 
 
Ziyao Liu, Jiale Guo,
Wenzhuo Yang, Jiani Fan,
Kwok-Yan Lam, and Jun Zhao.
2024a.

 
 Dynamic User Clustering for Efficient and
Privacy-Preserving Federated Learning.

 
 TDSC (2024).

 
 
 

 
 Liu
et al . (2024b) 
 
Ziyao Liu, Yu Jiang,
Weifeng Jiang, Jiale Guo,
Jun Zhao, and Kwok-Yan Lam.
2024b.

 
 Guaranteeing Data Privacy in Federated Unlearning
with Dynamic User Participation.

 
 arXiv preprint arXiv:2406.00966 
(2024).

 
 
 

 
 Liu et al . (2023) 
 
Ziyao Liu, Hsiao-Ying
Lin, and Yamin Liu. 2023.

 
 Long-term privacy-preserving aggregation with
user-dynamics for federated learning.

 
 TIFS (2023).

 
 
 

 
 Lu and Shen (2020) 
 
Zhigang Lu and Hong
Shen. 2020.

 
 Differentially Private k k k-Means Clustering
With Convergence Guarantee.

 
 TDSC 18,
4 (2020), 1541–1552.

 
 
 

 
 Lundberg and Lee (2017) 
 
Scott M Lundberg and
Su-In Lee. 2017.

 
 A unified approach to interpreting model
predictions.

 
 NeurIPS 30
(2017).

 
 
 

 
 Luo
et al . (2022) 
 
Xinjian Luo, Yangfan
Jiang, and Xiaokui Xiao.
2022.

 
 Feature inference attack on shapley values. In
 CCS . 2233–2247.

 
 
 

 
 Luo
et al . (2021) 
 
Xinjian Luo, Yuncheng Wu,
Xiaokui Xiao, and Beng Chin Ooi.
2021.

 
 Feature inference attack on model predictions in
vertical federated learning. In ICDE .
181–192.

 
 
 

 
 Machado
et al . (2021) 
 
Gabriel Resende Machado,
Eugênio Silva, and Ronaldo Ribeiro
Goldschmidt. 2021.

 
 Adversarial machine learning in image
classification: A survey toward the defender’s perspective.

 
 CSUR 55,
1 (2021), 1–38.

 
 
 

 
 Maleki et al . (2013) 
 
Sasan Maleki, Long
Tran-Thanh, Greg Hines, Talal Rahwan,
and Alex Rogers. 2013.

 
 Bounding the estimation error of sampling-based
Shapley value approximation.

 
 arXiv preprint arXiv:1306.4265 
(2013).

 
 
 

 
 Melis et al . (2019) 
 
Luca Melis, Congzheng
Song, Emiliano De Cristofaro, and
Vitaly Shmatikov. 2019.

 
 Exploiting unintended feature leakage in
collaborative learning. In SP .
691–706.

 
 
 

 
 Merz et al . (2022) 
 
Michael Merz, Ronald
Richman, Andreas Tsanakas, and Mario V
Wüthrich. 2022.

 
 Interpreting deep learning models with marginal
attribution by conditioning on quantiles.

 
 Data Min. Knowl. Discov. 
36, 4 (2022),
1335–1370.

 
 
 

 
 Mi
et al . (2024) 
 
Di Mi, Yanjun Zhang,
Leo Yu Zhang, Shengshan Hu,
Qi Zhong, Haizhuan Yuan, and
Shirui Pan. 2024.

 
 Towards Model Extraction Attacks in GAN-Based Image
Translation via Domain Shift Mitigation. In
 AAAI , Vol. 38.
19902–19910.

 
 
 

 
 Miller (2019) 
 
Tim Miller.
2019.

 
 Explanation in artificial intelligence: Insights
from the social sciences.

 
 AIJ 267
(2019), 1–38.

 
 
 

 
 Milli
et al . (2019) 
 
Smitha Milli, Ludwig
Schmidt, Anca D Dragan, and Moritz
Hardt. 2019.

 
 Model reconstruction from model explanations. In
 FAccT . 1–9.

 
 
 

 
 Mittelstadt
et al . (2019) 
 
Brent Mittelstadt, Chris
Russell, and Sandra Wachter.
2019.

 
 Explaining explanations in AI. In
 FAccT . 279–288.

 
 
 

 
 Miura
et al . (2021) 
 
Takayuki Miura, Satoshi
Hasegawa, and Toshiki Shibahara.
2021.

 
 MEGEX: Data-free model extraction attack against
gradient-based explainable AI.

 
 arXiv preprint arXiv:2107.08909 
(2021).

 
 
 

 
 Mochaourab et al . (2021) 
 
Rami Mochaourab, Sugandh
Sinha, Stanley Greenstein, and
Panagiotis Papapetrou. 2021.

 
 Robust counterfactual explanations for
privacy-preserving SVM. In ICML Workshops .

 
 
 

 
 Montavon et al . (2017) 
 
Grégoire Montavon,
Sebastian Lapuschkin, Alexander Binder,
Wojciech Samek, and Klaus-Robert
Müller. 2017.

 
 Explaining nonlinear classification decisions with
deep taylor decomposition.

 
 Pattern recognition 65
(2017), 211–222.

 
 
 

 
 Montenegro
et al . (2021) 
 
Helena Montenegro, Wilson
Silva, and Jaime S Cardoso.
2021.

 
 Privacy-preserving generative adversarial network
for case-based explainability in medical image analysis.

 
 IEEE Access 9
(2021), 148037–148047.

 
 
 

 
 Montenegro et al . (2022) 
 
Helena Montenegro, Wilson
Silva, Alex Gaudio, Matt Fredrikson,
Asim Smailagic, and Jaime S Cardoso.
2022.

 
 Privacy-preserving case-based explanations:
enabling visual interpretability by protecting privacy.

 
 IEEE Access 10
(2022), 28333–28347.

 
 
 

 
 Mothilal
et al . (2020) 
 
Ramaravind K Mothilal,
Amit Sharma, and Chenhao Tan.
2020.

 
 Explaining machine learning classifiers through
diverse counterfactual explanations. In FAccT .
607–617.

 
 
 

 
 Naidu et al . (2021) 
 
Rakshit Naidu, Aman
Priyanshu, Aadith Kumar, Sasikanth
Kotti, Haofan Wang, and Fatemehsadat
Mireshghallah. 2021.

 
 When differential privacy meets interpretability: A
case study.

 
 arXiv preprint arXiv:2106.13203 
(2021).

 
 
 

 
 Naretto
et al . (2022) 
 
Francesca Naretto, Anna
Monreale, and Fosca Giannotti.
2022.

 
 Evaluating the Privacy Exposure of Interpretable
Global Explainers. In CogMI .
13–19.

 
 
 

 
 Netzer et al . (2011) 
 
Yuval Netzer, Tao Wang,
Adam Coates, Alessandro Bissacco,
Bo Wu, and Andrew Y. Ng.
2011.

 
 Reading Digits in Natural Images with Unsupervised
Feature Learning. In NeurIPS Workshop .

 
 
 

 
 Nguyen
et al . (2023a) 
 
Duy Nguyen, Ngoc Bui,
and Viet Anh Nguyen. 2023a.

 
 Feasible Recourse Plan via Diverse Interpolation.
In AISTATS . 4679–4698.

 
 
 

 
 Nguyen
et al . (2015a) 
 
Quoc Viet Hung Nguyen,
Son Thanh Do, Thanh Tam Nguyen, and
Karl Aberer. 2015a.

 
 Tag-based paper retrieval: minimizing user effort
with diversity awareness. In International
Conference on Database Systems for Advanced Applications .
510–528.

 
 
 

 
 Nguyen et al . (2015b) 
 
Quoc Viet Hung Nguyen,
Thanh Tam Nguyen, Vinh Tuan Chau,
Tri Kurniawan Wijaya, Zoltán
Miklós, Karl Aberer, Avigdor Gal,
and Matthias Weidlich. 2015b.

 
 SMART: A tool for analyzing and reconciling schema
matching networks. In ICDE .
1488–1491.

 
 
 

 
 Nguyen et al . (2014) 
 
Quoc Viet Hung Nguyen, Tam
Nguyen Thanh, Zoltán Miklós, and
Karl Aberer. 2014.

 
 Reconciling schema matching networks through
crowdsourcing.

 
 EAI Endorsed Transactions on Collaborative
Computing 1, 2 (2014),
e2.

 
 
 

 
 Nguyen
et al . (2023b) 
 
Truc Nguyen, Phung Lai,
Hai Phan, and My T Thai.
2023b.

 
 Xrand: Differentially private defense against
explanation-guided attacks. In AAAI ,
Vol. 37. 11873–11881.

 
 
 

 
 Nguyen et al . (2022) 
 
Thanh Tam Nguyen,
Thanh Trung Huynh, Phi Le Nguyen,
Alan Wee-Chung Liew, Hongzhi Yin, and
Quoc Viet Hung Nguyen. 2022.

 
 A Survey of Machine Unlearning.

 
 arXiv preprint arXiv:2209.02299 
(2022).

 
 
 

 
 Ni et al . (2019) 
 
Jianmo Ni, Jiacheng Li,
and Julian McAuley. 2019.

 
 Justifying recommendations using distantly-labeled
reviews and fine-grained aspects. In
 EMNLP-IJCNLP . 188–197.

 
 
 

 
 Nugent
et al . (2009) 
 
Conor Nugent, Dónal
Doyle, and Pádraig Cunningham.
2009.

 
 Gaining insight through case-based explanation.

 
 JIIS 32
(2009), 267–295.

 
 
 

 
 Olatunji
et al . (2023) 
 
Iyiola E. Olatunji,
Mandeep Rathee, Thorben Funke, and
Megha Khosla. 2023.

 
 Private Graph Extraction via Feature Explanations.

 
 PETS 2023,
2 (2023), 59–78.

 
 
 

 
 Papernot and
McDaniel (2018) 
 
Nicolas Papernot and
Patrick McDaniel. 2018.

 
 Deep k-nearest neighbors: Towards confident,
interpretable and robust deep learning.

 
 arXiv preprint arXiv:1803.04765 
(2018).

 
 
 

 
 Papernot et al . (2017) 
 
Nicolas Papernot, Patrick
McDaniel, Ian Goodfellow, Somesh Jha,
Z Berkay Celik, and Ananthram Swami.
2017.

 
 Practical black-box attacks against machine
learning. In ASIA-CCS .
506–519.

 
 
 

 
 Patel
et al . (2022) 
 
Neel Patel, Reza Shokri,
and Yair Zick. 2022.

 
 Model explanations with differential privacy. In
 FAccT . 1895–1904.

 
 
 

 
 Pawelczyk
et al . (2020a) 
 
Martin Pawelczyk, Klaus
Broelemann, and Gjergji Kasneci.
2020a.

 
 Learning model-agnostic counterfactual explanations
for tabular data. In TheWebConf .
3126–3132.

 
 
 

 
 Pawelczyk
et al . (2020b) 
 
Martin Pawelczyk, Klaus
Broelemann, and Gjergji Kasneci.
2020b.

 
 On counterfactual explanations under predictive
multiplicity. In UAI . 809–818.

 
 
 

 
 Pawelczyk
et al . (2023) 
 
Martin Pawelczyk,
Himabindu Lakkaraju, and Seth Neel.
2023.

 
 On the privacy risks of algorithmic recourse. In
 AISTATS . 9680–9696.

 
 
 

 
 Pentyala et al . (2023) 
 
Sikha Pentyala, Shubham
Sharma, Sanjay Kariyappa, Freddy Lecue,
and Daniele Magazzeni. 2023.

 
 Privacy-Preserving Algorithmic Recourse.

 
 arXiv preprint arXiv:2311.14137 
(2023).

 
 
 

 
 Petitcolas (2023) 
 
Fabien AP Petitcolas.
2023.

 
 Kerckhoffs’ principle.

 
 In Encyclopedia of Cryptography, Security
and Privacy . Springer, 1–2.

 
 
 

 
 Prado-Romero et al . (2023) 
 
Mario Alfonso Prado-Romero,
Bardh Prenkaj, Giovanni Stilo, and
Fosca Giannotti. 2023.

 
 A survey on graph counterfactual explanations:
definitions, methods, evaluation, and research challenges.

 
 CSUR (2023).

 
 
 

 
 Qian
et al . (2017) 
 
Jianwei Qian, Xiang-Yang
Li, Chunhong Zhang, Linlin Chen,
Taeho Jung, and Junze Han.
2017.

 
 Social network de-anonymization and privacy
inference with knowledge graph model.

 
 TDSC 16,
4 (2017), 679–692.

 
 
 

 
 Quan et al . (2022) 
 
Pengrui Quan, Supriyo
Chakraborty, Jeya Vikranth Jeyakumar, and
Mani Srivastava. 2022.

 
 On the amplification of security and privacy risks
by post-hoc explanations in machine learning models.

 
 arXiv preprint arXiv:2206.14004 
(2022).

 
 
 

 
 Rajabi and
Etminani (2022) 
 
Enayat Rajabi and Kobra
Etminani. 2022.

 
 Knowledge-graph-based explainable AI: A systematic
review.

 
 JIS (2022),
01655515221112844.

 
 
 

 
 Ren
et al . (2020) 
 
Zhao Ren, Alice Baird,
Jing Han, Zixing Zhang, and
Björn Schuller. 2020.

 
 Generating and protecting against adversarial
attacks for deep speech-based emotion recognition models. In
 ICASSP . 7184–7188.

 
 
 

 
 Ren et al . (2022) 
 
Zhao Ren, Kun Qian,
Fengquan Dong, Zhenyu Dai,
Wolfgang Nejdl, Yoshiharu Yamamoto, and
Björn Schuller. 2022.

 
 Deep attention-based neural networks for
explainable heart sound classification.

 
 MLWA 9
(May 2022), 1–9.

 
 
 

 
 Ren
et al . (2023) 
 
Zhao Ren, Kun Qian,
Tanja Schultz, and Björn W. Schuller.
2023.

 
 An Overview of the ICASSP Special Session on AI
Security and Privacy in Speech and Audio Processing. In
 ACM Multimedia workshop .

 
 
 

 
 Ribeiro
et al . (2016) 
 
Marco Tulio Ribeiro,
Sameer Singh, and Carlos Guestrin.
2016.

 
 " Why should i trust you?" Explaining the
predictions of any classifier. In KDD .
1135–1144.

 
 
 

 
 Rigaki and Garcia (2023) 
 
Maria Rigaki and
Sebastian Garcia. 2023.

 
 A survey of privacy attacks in machine learning.

 
 CSUR 56,
4 (2023), 1–34.

 
 
 

 
 Sablayrolles et al . (2019) 
 
Alexandre Sablayrolles,
Matthijs Douze, Cordelia Schmid,
Yann Ollivier, and Hervé
Jégou. 2019.

 
 White-box vs black-box: Bayes optimal strategies
for membership inference. In ICML .
5558–5567.

 
 
 

 
 Salem et al . (2020) 
 
Ahmed Salem, Apratim
Bhattacharya, Michael Backes, Mario
Fritz, and Yang Zhang. 2020.

 
 { \{ Updates-Leak } \} : Data set inference and
reconstruction attacks in online learning. In
 USENIX . 1291–1308.

 
 
 

 
 Salem et al . (2018) 
 
Ahmed Salem, Yang Zhang,
Mathias Humbert, Pascal Berrang,
Mario Fritz, and Michael Backes.
2018.

 
 Ml-leaks: Model and data independent membership
inference attacks and defenses on machine learning models.

 
 arXiv preprint arXiv:1806.01246 
(2018).

 
 
 

 
 Selvaraju et al . (2017) 
 
Ramprasaath R Selvaraju,
Michael Cogswell, Abhishek Das,
Ramakrishna Vedantam, Devi Parikh, and
Dhruv Batra. 2017.

 
 Grad-cam: Visual explanations from deep networks
via gradient-based localization. In ICCV .
618–626.

 
 
 

 
 Sen et al . (2008) 
 
Prithviraj Sen, Galileo
Namata, Mustafa Bilgic, Lise Getoor,
Brian Galligher, and Tina Eliassi-Rad.
2008.

 
 Collective classification in network data.

 
 AI magazine 29,
3 (2008), 93–93.

 
 
 

 
 Severi
et al . (2021) 
 
Giorgio Severi, Jim
Meyer, Scott Coull, and Alina Oprea.
2021.

 
 { \{ Explanation-Guided } \} backdoor poisoning
attacks against malware classifiers. In USENIX .
1487–1504.

 
 
 

 
 Shokri
et al . (2019) 
 
Reza Shokri, Martin
Strobel, and Yair Zick.
2019.

 
 Privacy risks of explaining machine learning
models.

 
 arXiv preprint arXiv:1907.00164 
3 (2019).

 
 
 

 
 Shokri
et al . (2020) 
 
Reza Shokri, Martin
Strobel, and Yair Zick.
2020.

 
 Exploiting transparency measures for membership
inference: a cautionary tale. In PPAI ,
Vol. 13.

 
 
 

 
 Shokri
et al . (2021) 
 
Reza Shokri, Martin
Strobel, and Yair Zick.
2021.

 
 On the privacy risks of model explanations. In
 AIES . 231–241.

 
 
 

 
 Shrikumar
et al . (2017) 
 
Avanti Shrikumar, Peyton
Greenside, and Anshul Kundaje.
2017.

 
 Learning important features through propagating
activation differences. In ICML .
3145–3153.

 
 
 

 
 Silva
et al . (2020) 
 
Wilson Silva, Alexander
Poellinger, Jaime S Cardoso, and
Mauricio Reyes. 2020.

 
 Interpretability-guided content-based medical image
retrieval. In MICCAI . 305–314.

 
 
 

 
 Simonyan
et al . (2013) 
 
Karen Simonyan, Andrea
Vedaldi, and Andrew Zisserman.
2013.

 
 Deep inside convolutional networks: Visualising
image classification models and saliency maps.

 
 arXiv preprint arXiv:1312.6034 
(2013).

 
 
 

 
 Slack et al . (2020) 
 
Dylan Slack, Sophie
Hilgard, Emily Jia, Sameer Singh, and
Himabindu Lakkaraju. 2020.

 
 Fooling lime and shap: Adversarial attacks on post
hoc explanation methods. In AIES .
180–186.

 
 
 

 
 Sliwinski
et al . (2019) 
 
Jakub Sliwinski, Martin
Strobel, and Yair Zick.
2019.

 
 Axiomatic characterization of data-driven influence
measures for classification. In AAAI ,
Vol. 33. 718–725.

 
 
 

 
 Smilkov et al . (2017) 
 
Daniel Smilkov, Nikhil
Thorat, Been Kim, Fernanda Viégas,
and Martin Wattenberg. 2017.

 
 Smoothgrad: removing noise by adding noise.

 
 arXiv preprint arXiv:1706.03825 
(2017).

 
 
 

 
 Sokol and Flach (2019) 
 
Kacper Sokol and Peter
Flach. 2019.

 
 Counterfactual explanations of machine learning
predictions: Opportunities and challenges for AI safety. In
 SafeAI .

 
 
 

 
 Song
et al . (2017) 
 
Congzheng Song, Thomas
Ristenpart, and Vitaly Shmatikov.
2017.

 
 Machine learning models that remember too much. In
 CCS . 587–601.

 
 
 

 
 Song and
Shmatikov (2020) 
 
Congzheng Song and
Vitaly Shmatikov. 2020.

 
 Overlearning Reveals Sensitive Attributes. In
 ICLR .

 
 
 

 
 Springenberg et al . (2014) 
 
Jost Tobias Springenberg,
Alexey Dosovitskiy, Thomas Brox, and
Martin Riedmiller. 2014.

 
 Striving for simplicity: The all convolutional
net.

 
 arXiv preprint arXiv:1412.6806 
(2014).

 
 
 

 
 Strack et al . (2014) 
 
Beata Strack, Jonathan P
DeShazo, Chris Gennings, Juan L Olmo,
Sebastian Ventura, Krzysztof J Cios,
John N Clore, et al . 
2014.

 
 Impact of HbA1c measurement on hospital readmission
rates: analysis of 70,000 clinical database patient records.

 
 BioMed research international 
2014 (2014).

 
 
 

 
 Štrumbelj
and Kononenko (2014) 
 
Erik Štrumbelj and
Igor Kononenko. 2014.

 
 Explaining prediction models and individual
predictions with feature contributions.

 
 Knowledge and information systems 
41 (2014), 647–665.

 
 
 

 
 Sundararajan
et al . (2017) 
 
Mukund Sundararajan, Ankur
Taly, and Qiqi Yan. 2017.

 
 Axiomatic attribution for deep networks. In
 ICML . 3319–3328.

 
 
 

 
 Sweeney (2000) 
 
Latanya Sweeney.
2000.

 
 Simple demographics often identify people
uniquely.

 
 Health 671,
2000 (2000), 1–34.

 
 
 

 
 Thang
et al . (2015) 
 
Duong Chi Thang,
Nguyen Thanh Tam, Nguyen Quoc Viet Hung,
and Karl Aberer. 2015.

 
 An evaluation of diversification techniques. In
 International Conference on Database and Expert
Systems Applications . 215–231.

 
 
 

 
 Tiddi and
Schlobach (2022) 
 
Ilaria Tiddi and Stefan
Schlobach. 2022.

 
 Knowledge graphs as tools for explainable machine
learning: A survey.

 
 AIJ 302
(2022), 103627.

 
 
 

 
 Tramèr et al . (2016) 
 
Florian Tramèr, Fan
Zhang, Ari Juels, Michael K Reiter,
and Thomas Ristenpart. 2016.

 
 Stealing machine learning models via prediction
 { \{ APIs } \} . In USENIX .
601–618.

 
 
 

 
 ur Rehman et al . (2019) 
 
Atique ur Rehman, Rafia
Rahim, Shahroz Nadeem, and Sibt ul
Hussain. 2019.

 
 End-to-end trained CNN encoder-decoder networks for
image steganography. In ECCV-Workshops .
723–729.

 
 
 

 
 Ustun
et al . (2019) 
 
Berk Ustun, Alexander
Spangher, and Yang Liu.
2019.

 
 Actionable recourse in linear classification. In
 FAccT . 10–19.

 
 
 

 
 van der Waa et al . (2018) 
 
Jasper van der Waa, Marcel
Robeer, Jurriaan van Diggelen, Matthieu
Brinkhuis, and Mark Neerincx.
2018.

 
 Contrastive explanations with local foil trees.

 
 arXiv preprint arXiv:1806.07470 
(2018).

 
 
 

 
 Veale
et al . (2018) 
 
Michael Veale, Reuben
Binns, and Lilian Edwards.
2018.

 
 Algorithms that remember: model inversion attacks
and data protection law.

 
 Philos. Trans. R. Soc. A 
376, 2133 (2018),
20180083.

 
 
 

 
 Veugen
et al . (2022) 
 
Thijs Veugen, Bart
Kamphorst, and Michiel Marcus.
2022.

 
 Privacy-preserving contrastive explanations with
local foil trees.

 
 Cryptography 6,
4 (2022), 54.

 
 
 

 
 Vo et al . (2023) 
 
Vy Vo, Trung Le,
Van Nguyen, He Zhao,
Edwin V Bonilla, Gholamreza Haffari,
and Dinh Phung. 2023.

 
 Feature-based learning for diverse and
privacy-preserving counterfactual explanations. In
 KDD . 2211–2222.

 
 
 

 
 Wachter
et al . (2017) 
 
Sandra Wachter, Brent
Mittelstadt, and Chris Russell.
2017.

 
 Counterfactual explanations without opening the
black box: Automated decisions and the GDPR.

 
 Harv. JL Tech. 31
(2017), 841.

 
 
 

 
 Wagner
et al . (2023) 
 
Tal Wagner, Yonatan
Naamad, and Nina Mishra.
2023.

 
 Fast private kernel density estimation via locality
sensitive quantization. In ICML .
35339–35367.

 
 
 

 
 Wang et al . (2017) 
 
Di Wang, Minwei Ye, and
Jinhui Xu. 2017.

 
 Differentially private empirical risk minimization
revisited: Faster and more general.

 
 NeurIPS 30
(2017).

 
 
 

 
 Wang (2019) 
 
Guan Wang.
2019.

 
 Interpret federated learning with shapley values.

 
 arXiv preprint arXiv:1905.04519 
(2019).

 
 
 

 
 Wang
et al . (2021) 
 
Yu Wang, Lifu Huang,
Philip S Yu, and Lichao Sun.
2021.

 
 Membership inference attacks on knowledge graphs.

 
 arXiv preprint arXiv:2104.08273 
(2021).

 
 
 

 
 Wang
et al . (2022) 
 
Yongjie Wang, Hangwei
Qian, and Chunyan Miao.
2022.

 
 Dualcf: Efficient model extraction attack from
counterfactual explanations. In FAccT .
1318–1329.

 
 
 

 
 Watson
et al . (2022) 
 
Lauren Watson, Rayna
Andreeva, Hao-Tsung Yang, and Rik
Sarkar. 2022.

 
 Differentially Private Shapley Values for Data
Evaluation.

 
 arXiv preprint arXiv:2206.00511 
(2022).

 
 
 

 
 Wu
et al . (2020) 
 
Zonghan Wu, Shirui Pan,
Fengwen Chen, Guodong Long,
Chengqi Zhang, and S Yu Philip.
2020.

 
 A comprehensive survey on graph neural networks.

 
 TNNLS 32,
1 (2020), 4–24.

 
 
 

 
 Xue
et al . (2024) 
 
Lulu Xue, Shengshan Hu,
Ruizhi Zhao, Leo Yu Zhang,
Shengqing Hu, Lichao Sun, and
Dezhong Yao. 2024.

 
 Revisiting Gradient Pruning: A Dual Realization for
Defending against Gradient Attacks. In AAAI .
6404–6412.

 
 
 

 
 Yang
et al . (2022) 
 
Fan Yang, Qizhang Feng,
Kaixiong Zhou, Jiahao Chen, and
Xia Hu. 2022.

 
 Differentially Private Counterfactuals via
Functional Mechanism.

 
 arXiv preprint arXiv:2208.02878 
(2022).

 
 
 

 
 Yang
et al . (2019) 
 
Ziqi Yang, Jiyi Zhang,
Ee-Chien Chang, and Zhenkai Liang.
2019.

 
 Neural network inversion in adversarial setting via
background knowledge alignment. In CCS .
225–240.

 
 
 

 
 Ye et al . (2022) 
 
Jiayuan Ye, Aadyaa Maddi,
Sasi Kumar Murakonda, Vincent
Bindschaedler, and Reza Shokri.
2022.

 
 Enhanced membership inference attacks against
machine learning models. In CCS .
3093–3106.

 
 
 

 
 Yeom
et al . (2018) 
 
Samuel Yeom, Irene
Giacomelli, Matt Fredrikson, and Somesh
Jha. 2018.

 
 Privacy risk in machine learning: Analyzing the
connection to overfitting. In CSF .
268–282.

 
 
 

 
 Yuan
et al . (2022) 
 
Hao Yuan, Haiyang Yu,
Shurui Gui, and Shuiwang Ji.
2022.

 
 Explainability in graph neural networks: A
taxonomic survey.

 
 TPAMI 45,
5 (2022), 5782–5799.

 
 
 

 
 Zhang and
Bareinboim (2018) 
 
Junzhe Zhang and Elias
Bareinboim. 2018.

 
 Fairness in decision-making—the causal
explanation formula. In AAAI ,
Vol. 32.

 
 
 

 
 Zhang
et al . (2021) 
 
Wanrong Zhang, Shruti
Tople, and Olga Ohrimenko.
2021.

 
 Leakage of dataset properties in
 { \{ Multi-Party } \} machine learning. In USENIX .
2687–2704.

 
 
 

 
 Zhang
et al . (2020b) 
 
Xinyang Zhang, Ningfei
Wang, Hua Shen, Shouling Ji,
Xiapu Luo, and Ting Wang.
2020b.

 
 Interpretable deep learning under fire. In
 USENIX .

 
 
 

 
 Zhang
et al . (2024) 
 
Yechao Zhang, Shengshan
Hu, Leo Yu Zhang, Junyu Shi,
Minghui Li, Xiaogeng Liu, and
Hai Jin. 2024.

 
 Why Does Little Robustness Help? A Further Step
Towards Understanding Adversarial Transferability. In
 S P , Vol. 2.

 
 
 

 
 Zhang
et al . (2020a) 
 
Yuheng Zhang, Ruoxi Jia,
Hengzhi Pei, Wenxiao Wang,
Bo Li, and Dawn Song.
2020a.

 
 The secret revealer: Generative model-inversion
attacks against deep neural networks. In CVPR .
253–261.

 
 
 

 
 Zhang
et al . (2018) 
 
Yulun Zhang, Yapeng Tian,
Yu Kong, Bineng Zhong, and
Yun Fu. 2018.

 
 Residual dense network for image super-resolution.
In CVPR . 2472–2481.

 
 
 

 
 Zhang
et al . (2022) 
 
Zaixi Zhang, Qi Liu,
Hao Wang, Chengqiang Lu, and
Cheekong Lee. 2022.

 
 Protgnn: Towards self-explaining graph neural
networks. In AAAI , Vol. 36.
9127–9135.

 
 
 

 
 Zhao et al . (2021a) 
 
Bo Zhao, Han van der Aa,
Thanh Tam Nguyen, Quoc Viet Hung Nguyen,
and Matthias Weidlich. 2021a.

 
 Eires: Efficient integration of remote data in
event stream processing. In SIGMOD .
2128–2141.

 
 
 

 
 Zhao
et al . (2021b) 
 
Xuejun Zhao, Wencan
Zhang, Xiaokui Xiao, and Brian Lim.
2021b.

 
 Exploiting explanations for model inversion
attacks. In ICCV . 682–692.

 
 
 

 
 Zhou et al . (2016) 
 
Bolei Zhou, Aditya
Khosla, Agata Lapedriza, Aude Oliva,
and Antonio Torralba. 2016.

 
 Learning deep features for discriminative
localization. In CVPR .
2921–2929.