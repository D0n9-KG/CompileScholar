# DEFINING AND EXTRACTING GENERALIZABLE INTERACTION PRIMITIVES FROM DNNs

Lu Chen $^{1*}$ Siyu Lou $^{1,2*}$ Benhao Huang $^{1}$ Quanshi Zhang $^{1\dagger}$

$^{1}$ Shanghai Jiao Tong University, Shanghai, China $^{2}$ Eastern Institute of Technology, Ningbo, China {lu.chen, siyu.lou, hbh001098hbh, zqs1022}@sjtu.edu.cn

# ABSTRACT

Faithfully summarizing the knowledge encoded by a deep neural network (DNN) into a few symbolic primitive patterns without losing much information represents a core challenge in explainable AI. To this end, Ren et al. (2024) have derived a series of theorems to prove that the inference score of a DNN can be explained as a small set of interactions between input variables. However, the lack of generalization power makes it still hard to consider such interactions as faithful primitive patterns encoded by the DNN. Therefore, given different DNNs trained for the same task, we develop a new method to extract interactions that are shared by these DNNs. Experiments show that the extracted interactions can better reflect common knowledge shared by different DNNs $^{1}$ .

# 1 INTRODUCTION

Explaining and quantifying the exact knowledge encoded by a deep neural network (DNN) presents a new challenge in explainable AI. Previous studies mainly visualized patterns encoded by DNNs (Bau et al., 2017; Kim et al., 2018) and estimated a saliency map on input variables (Simonyan et al., 2013; R. Selvaraju et al., 2017). However, a new question is that can we formulate the implicit knowledge encoded by the DNN as explicit and symbolic primitive patterns? In fact, we hope these primitive patterns serve as elementary units for inference, just like concepts in human cognition.

However, there is no widely accepted way to define the concept encoded by a DNN, because we cannot mathematically define/formulate the exact concept in human cognition. Nevertheless, if we ignore cognitive issues, Ren et al. (2024); Li & Zhang (2023b) have derived a series of theorems as convincing evidence to take interactions as symbolic primitives encoded by a DNN. Specifically, an interaction captures the intricate nonlinear relationship encoded by the DNN. For instance, when a DNN processes a sentence “It is raining cats and dogs!”, the DNN may encode the interaction between a set of input variables $S = \{raining, cats, and, dogs\} \subseteq N$ . When all words in S are present, an interactive effect $I(S)$ emerges, and pushes the DNN’s inference towards the semantic meaning of “heavy rain.” However, if any word in S is masked, the effect will be removed.

Ren et al. (2024) have mainly proven two theorems to justify the convincingness of considering above interactions as primitive inference patterns encoded by the DNN. First, it is proven that under some common conditions $^{2}$ , a well-trained DNN usually just encodes a limited number of interactions w.r.t. a few sets of input variables. More crucially, let us randomly mask an input sample x in different ways to generate an exponential number of masked samples. It is proven that people can use just a few interactions to accurately approximate the DNN's outputs on all these masked samples. Thus, these few interactions are referred to as interaction primitives.

Despite the aforementioned theorems, this study does not yet deem the above interactions as faithful primitives of DNNs. The core problem is that existing interaction extraction methods cannot theoretically guarantee the generalization (transferability) of the interactions, e.g., ensuring to extract

![](images/bc779397a6ae6968fe39b542a5da846d585fca84b9d8e7307a701830b2d34b83.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1st DNN"] --> B["shared"]
    B --> C["2nd DNN"]
    A --> D["distinctive"]
    B --> E["distinctive"]
    F["Interaction primitives"] --> G["AND said black dog"]
    F --> H["OR dog over"]
    I["Input variables"] --> J["They said the black dog is over David."]
    K["Input sentence"] --> L["Negative sentiment"]
    M["OR dog over David"] --> N["AND they said"]
    O["Black dog"] --> P["AND black dog over"]
    Q["Black dog"] --> R["AND black dog over"]
    S["Black dog"] --> T["AND black dog over"]
    U["Black dog"] --> V["AND black dog over"]
    W["Black dog"] --> X["AND black dog over"]
    Y["Black dog"] --> Z["AND black dog over"]
    AA["Black dog"] --> AB["AND black dog over"]
    AC["Black dog"] --> AD["AND black dog over"]
    AE["Black dog"] --> AF["AND black dog over"]
    AG["Black dog"] --> AH["AND black dog over"]
    AI["Black dog"] --> AJ["AND black dog over"]
    AK["Black dog"] --> AL["AND black dog over"]
    AM["Black dog"] --> AN["AND black dog over"]
    AO["Black dog"] --> AP["AND black dog over"]
    AQ["Black dog"] --> AR["AND black dog over"]
    AS["Black dog"] --> AT["AND black dog over"]
    AU["Black dog"] --> AV["AND black dog over"]
    AW["Black dog"] --> AX["AND black dog over"]
    AY["Interaction primitives"] --> AZ["AND said black dog"]
    AY --> BA["OR dog over"]
    BB["Interaction primitives"] --> BC["AND said black dog"]
    BB --> BD["OR dog over"]
    BE["Interaction primitives"] --> BF["AND said black dog"]
    BE --> BG["OR dog over"]
    BH["Interaction primitives"] --> BI["AND said black dog"]
    BI --> BJ["OR dog over"]
    BK["Interaction primitives"] --> BL["AND said black dog"]
    BK --> BM["OR dog over"]
    BN["Interaction primitives"] --> BO["AND said black dog"]
    BN --> BP["OR dog over"]
    BQ["Interaction primitives"] --> BR["AND said black dog"]
    BQ --> BS["OR dog over"]
    BT["Interaction primitives"] --> BU["AND said black dog"]
    BT --> BV["OR dog over"]
    BW["Interaction primitives"] --> BX["AND said black dog"]
    BW --> BY["OR dog over"]
    BZ["Interaction primitives"] --> CA["AND said black dog"]
    BZ --> CB["OR dog over"]
    CC["Interaction primitives"] --> CD["AND said black dog"]
    CC --> CE["OR dog over"]
    CF["Interaction primitives"] --> CG["AND said black dog"]
    CF --> CH["OR dog over"]
    CI["Interaction primitives"] --> CJ["AND said black dog"]
    CI --> CK["OR dog over"]
    CL["Interaction primitives"] --> CM["AND said black dog"]
    CL --> CN["OR dog over"]
    CO["Interaction primitives"] --> CP["AND said black dog"]
    CO --> CQ["OR dog over"]
    CR["Interaction primitives"] --> CS["AND said black dog"]
    CR --> CT["OR dog over"]
    CU["Interaction primitives"] --> CV["AND said black dog"]
    CU --> CW["OR dog over"]
    CX["Interaction primitives"] --> CY["AND said black dog"]
    CX --> CZ["OR dog over"]
    DA["Interaction primitives"] --> DB["AND said black dog"]
    DA --> DC["OR dog over"]
    DD["Interaction primitives"] --> DE["AND said black dog"]
    DD --> DF["OR dog over"]
    DG["DNNs"] --> DH["DNNs 1st DNN"]
    DG["DNNs 2nd DNN"] --> DI["DNNs 1st DNN"]
```
</details>

Figure 1: Distinctive and shared interactions. When we extract AND-OR interactions from two DNNs, AND interactions $S_{1}=\{black,dog\}$ and $S_{2}=\{black,dog,over\}$ , and OR interactions $S_{3}=\{black,dog\}$ and $S_{4}=\{black,over\}$ are shared by two DNNs, while interactions, such as the AND interaction $S_{5}=\{they,said\}$ , are distinctive interactions encoded by a single DNN.

common interactions shared by different AI models. Interactions, which are not shared by different DNNs, may be perceived as out-of-distribution signals without clear meanings.

Therefore, in this study, we revisit the generalization of interactions. Specifically, we identify a clear mechanism that makes the existing method extract different interactions from the same DNN under different initialization states, which hurts the generalization power of interactions.

Thus, to address the generalization issue, we propose a new method for extracting generalizable interactions. A generalizable interaction is defined as Figure 1 shows. Given multiple DNNs trained for the same task and an input sample, if an interaction can be extracted from all these DNNs, then we consider it generalizable. Our method is designed to extract interactions with maximum generalization power. This approach ensures that if an interaction exhibits a significant impact on the output score for one DNN, it usually demonstrates noteworthy influence for other DNNs. We conducted experiments on various datasets. Experiments showed that our proposed method significantly improved the generalization power of the extracted interactions across different DNNs.

# 2 GENERALIZABLE INTERACTION PRIMITIVES ACROSS DNNs

# 2.1 PRELIMINARIES: EXPLAINING THE NETWORK OUTPUT WITH INTERACTION PRIMITIVES

Although there is no theory to guarantee that how to define concepts that fit well with human cognition, Li & Zhang (2023a) and Ren et al. (2024) still provided mathematical supports to explain why we can still use interactions between input variables as the primitives or concepts encoded by the DNN. Specifically, there are two types of interactions, i.e., AND interactions and OR interactions.

AND interactions. Given a function $v : R^{n} \to R$ , let us consider an input sample $x = [x_{1}, x_{2}, \cdots, x_{n}]^{\intercal}$ with n input variables indexed by $N = \{1, 2, \ldots, n\}$ . Here, $v(\mathbf{x}) \in \mathbb{R}$ denotes the function output on $x^{3}$ . Then, Ren et al. (2023b) have used the Harsanyi dividend (Harsanyi, 1963) $I_{\mathrm{and}}(S|\mathbf{x})$ to quantify the numerical effect of the AND relationship between input variables in $S \subseteq N$ , which is encoded by the function v. We consider this interaction as an AND interaction.

$$
I _ {\text { and }} (S | \mathbf {x}) := \sum_ {T \subseteq S} (- 1) ^ {| S | - | T |} v (\mathbf {x} _ {T}). \tag {1}
$$

where $I_{\mathrm{and}}(\emptyset|\mathbf{x}) = v(\mathbf{x}_{\emptyset})$ , and $\mathbf{x}_T$ denotes a sample whose input variables in $N \setminus T$ are masked $^4$ .

Each AND interaction $I_{\mathrm{and}}(S|\mathbf{x})$ reveals the AND relationship between all variables in S. For instance, let us consider the slang term $S = \{x_{3}, x_{4}, x_{5}, x_{6}\}$ in the sentence “ $x_{1} = It, x_{2} = is, x_{3} = raining, x_{4} = cats, x_{5} = and, x_{6} = dogs!$ ” as a toy example. The co-occurrence of four words forms the semantic concept of “heavy rain” and contributes a numerical effect $I_{\mathrm{and}}(S|\mathbf{x})$ to the function output. Otherwise, the masking of any word $x_{i} \in S$ invalidates the semantic concept and eliminates the interaction effect, i.e., obtaining $I_{\mathrm{and}}(S|\mathbf{x}^{\mathrm{masked}}) = 0$ on the masked sample.

OR interactions. Analogously, we can also use the OR interaction to explain the function $v : R^{n} \to R$ . To this end, (Zhou et al., 2023; Li & Zhang, 2023a) have defined the following OR interaction effect $I_{\mathrm{or}}(S|\mathbf{x})$ to measure the OR interaction encoded by v. In particular, $I_{\mathrm{or}}(\emptyset|\mathbf{x}) = v(\mathbf{x}_{\emptyset})$ .

$$
I _ {\mathrm{or}} (S | \mathbf {x}) := - \sum_ {T \subseteq S} (- 1) ^ {| S | - | T |} v (\mathbf {x} _ {N \setminus T}). \tag {2}
$$

Each OR interaction $I_{\mathrm{or}}(S|\mathbf{x})$ describes the OR relationship between all variables in S. Let us consider an input sentence “ $x_{1}=This$ , $x_{2}=movie$ , $x_{3}=is$ , $x_{4}=boring$ , $x_{5}=and$ , $x_{6}=disappointing$ ” for sentiment classification. Let us set $S=\{x_{4},x_{6}\}$ . The presence of any word in S will contribute a negative sentiment effect $I_{\mathrm{or}}(S|\mathbf{x})$ to the function output.

Sparsity of interactions. Theoretically, according to Equation (1), a function can encode at most $2^{n}$ different AND interactions w.r.t. all $2^{n}$ subsets $\forall S, S \subseteq N$ . However, Ren et al. (2024) have proved that under some common conditions $^{2}$ , most well-trained DNNs only encode a small set of AND interactions, denoted by $\Omega$ , i.e., only a few interactions $S \in \Omega$ have considerable effects $I_{\mathrm{and}}(S|\mathbf{x})$ . All other interactions have almost zero effects, i.e., $I_{\mathrm{and}}(S|\mathbf{x}) \approx 0$ , which can be regarded as a set of negligible noise patterns.

It is worth noting that an OR interaction can be regarded as a specific AND interaction, if we inverse the definition of the masked state and the unmasked state of an input variable $^{5}$ . Thus, the proven sparsity of AND interactions can also indicate the conclusion that well-trained DNNs tend to encode a small number of OR interactions.

Definition of interaction primitives. Considering the above proven sparsity of interactions, we define an interaction primitive as a salient interaction. Formally, given a threshold $\tau$ , the set of interaction primitives are defined as $\Omega = \{S \subseteq N : |I(S|\mathbf{x})| > \tau\}$ .

Theorem 1 (Universal matching theorem, proved by (Ren et al., 2024)). As the corollary of the proven sparsity in Ren et al. (2024), the function's output on all $2^{n}$ masked samples $\{\mathbf{x}_S|S\subseteq N\}$ could be universally explained by the interaction primitives in $\Omega$ , s.t., $|\Omega|\ll 2^n$ , i.e., $\forall S\subseteq N,v(\mathbf{x}_S) = \sum_{T\subseteq S}I_{\mathrm{and}}(T|\mathbf{x})\approx \sum_{T\subseteq S:T\in \Omega}I_{\mathrm{and}}(T|\mathbf{x})$ .

In particular, Theorem 1 shows that if we arbitrarily mask the input sample x, we can get $2^{n}$ different masked samples $^{4}$ , $\forall S, S \subseteq N$ . Then, we can universally match the output of the function $v(\mathbf{x}_{S})$ on all $2^{n}$ masked samples using only a few interaction primitives in $\Omega$ .

# 2.2 FAITHFULNESS PROBLEM WITH INTERACTION-BASED EXPLANATIONS

Basic setting of using AND-OR interactions to explain a DNN. In this section, we consider to employ both AND interactions and OR interactions to explain the DNN's output. This is because the complexity of the representations in DNNs makes it difficult to rely solely on either AND interactions or OR interactions to faithfully explain true inference primitives encoded by the DNN.

To this end, we need to decompose the output of the DNN into two terms $v(\mathbf{x}) = v_{\mathrm{and}}(\mathbf{x}) + v_{\mathrm{or}}(\mathbf{x})$ , so that we can use AND interactions to explain the term $v_{\mathrm{and}}(\mathbf{x})$ and use OR interactions to explain the term $v_{\mathrm{or}}(\mathbf{x})$ . In this way, the first challenge is how to learn an appropriate decomposition of $v_{\mathrm{and}}(\mathbf{x})$ and $v_{\mathrm{or}}(\mathbf{x})$ that reveals intrinsic primitive interactions encoded by the DNN. We will discuss this challenge later.

No matter how we randomly decompose $v(\mathbf{x}) = v_{\mathrm{and}}(\mathbf{x}) + v_{\mathrm{or}}(\mathbf{x})$ , Theorem 2 states that we can still use interactions to fit the DNN's outputs on $2^n$ randomly masked samples $\{\mathbf{x}_T|T \subseteq N\}$ . Furthermore, according to the sparsity of interaction primitives in Section 2.1, we can obtain Proposition 1, i.e., the $2^n$ network outputs on all masked samples can usually be approximated by a small number of AND interaction primitives in $\Omega^{\mathrm{and}}$ and OR interaction primitives in $\Omega^{\mathrm{or}}$ , s.t., $|\Omega^{\mathrm{and}}|$ , $|\Omega^{\mathrm{or}}| \ll 2^n$ .

Theorem 2 (Universal matching theorem, proof in Appendix C). Let us be given a DNN v and an input sample x. For each randomly masked sample $x_{T}, T \subseteq N$ , we obtain

$$
v \left(\mathbf {x} _ {T}\right) = v _ {\text { and }} \left(\mathbf {x} _ {T}\right) + v _ {\text { or }} \left(\mathbf {x} _ {T}\right) = \sum_ {S \subseteq T} I _ {\text { and }} (S \mid \mathbf {x} _ {T}) + \sum_ {S \in \{S: S \cap T \neq \emptyset \} \cup \{\emptyset \}} I _ {\text { or }} (S \mid \mathbf {x} _ {T}). \tag {3}
$$

Proposition 1. The output of a well-trained DNN on all $2^{n}$ masked samples $\{\mathbf{x}_T|T\subseteq N\}$ could be universally approximated by the interaction primitives in $\Omega^{\mathrm{and}}$ and $\Omega^{\mathrm{or}}$ , s.t., $|\Omega^{\mathrm{and}}|$ , $|\Omega^{\mathrm{or}}|\ll 2^n$ , i.e., $\forall T\subseteq N,v(\mathbf{x}_T) = \sum_{S\subseteq T}I_{\mathrm{and}}(S|\mathbf{x}_T) + \sum_{S\in \{S:S\cap T\neq \emptyset\} \cup \{\emptyset \}}I_{\mathrm{or}}(S|\mathbf{x}_T)\approx v(\mathbf{x}_{\emptyset}) + \sum_{\emptyset \neq S\subseteq T:S\in \Omega^{\mathrm{and}}}I_{\mathrm{and}}(S|\mathbf{x}_T) + \sum_{S\cap T\neq \emptyset:S\in \Omega^{\mathrm{or}}}I_{\mathrm{or}}(S|\mathbf{x}_T)$ , where $v(\mathbf{x}_{\emptyset}) = v_{\mathrm{and}}(\mathbf{x}_{\emptyset}) + v_{\mathrm{or}}(\mathbf{x}_{\emptyset})$ .

Problems with the faithfulness of interactions. Although the universal matching capacity proven in Theorem 2 is a quite significant advantage of AND-OR interactions, it is still not the ultimate guarantee for the faithfulness of the extracted interactions. To be precise, there is still no standard way to faithfully decompose the $v_{\mathrm{and}}(\mathbf{x})$ term and the $v_{\mathrm{or}}(\mathbf{x})$ term that reveal intrinsic primitive interactions encoded by the DNN, considering the following two challenges.

\- Challenge 1, ambiguous decomposition of $v_{\mathrm{and}}(\mathbf{x})$ and $v_{\mathrm{or}}(\mathbf{x})$ usually brings in considerable uncertainty in the extraction of interactions. Let us take the following toy Boolean function as an example to illustrate the diversity of interactions, $f(\mathbf{x}) = x_1 \wedge x_2 \wedge x_3 + x_2 \wedge x_3 + x_3 \wedge x_4 + x_4 \vee x_5$ , where $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5]^{\intercal}$ and $x_i \in \{0, 1\}$ . We have two ways to decompose $f(\mathbf{x})$ . First, we can simply decompose $v_{\mathrm{and}}(\mathbf{x}) = x_1 \wedge x_2 \wedge x_3 + x_2 \wedge x_3 + x_3 \wedge x_4$ and $v_{\mathrm{or}}(\mathbf{x}) = x_4 \vee x_5$ , then to explain $f(\mathbf{x})$ with an OR interaction $I_{\mathrm{or}}(S = \{4, 5\})$ and three AND interactions $I_{\mathrm{and}}(S = \{1, 2, 3\})$ , $I_{\mathrm{and}}(S = \{2, 3\})$ and $I_{\mathrm{and}}(S = \{3, 4\})$ . Alternatively, we can also use exclusively AND interactions to explain $f(\mathbf{x})$ . Specifically, we can rewrite $v_{\mathrm{and}}(\mathbf{x}) = x_1 \wedge x_2 \wedge x_3 + x_2 \wedge x_3 + x_3 \wedge x_4 + x_4 \vee x_5 = x_1 \wedge x_2 \wedge x_3 + x_2 \wedge x_3 + x_3 \wedge x_4 + (x_4 + x_5 - x_4 \wedge x_5)$ and $v_{\mathrm{or}}(\mathbf{x}) = 0$ , w.r.t $x_i \in \{0, 1\}$ . Thus, the $v_{\mathrm{and}}$ term can be explained by a total of six AND interaction primitives. This is a typical case for diverse strategy of extracting interactions that are generated by different decompositions.

The aforementioned $f(\mathbf{x})$ is just an exceedingly simple function. In real-world applications, DNNs usually encode intricate AND-OR relationships among input variables, making it exceptionally challenging to formulate an explicit expression for the DNN function or to establish a definitive ground-truth decomposition of $v_{\mathrm{and}}(\mathbf{x})$ and $v_{\mathrm{or}}(\mathbf{x})$ . Consequently, the diversity issue with interactions are ubiquitous and unavoidable.

\- Challenge 2, how to ensure the interaction primitives are generalizable. It is commonly considered that generalizable primitives are usually transferable over different models trained for the same task, instead of being over-fitted by a single model. Thus, if an interaction primitive can be consistently extracted from different DNNs, then it can be considered as a faithful concept. Otherwise, non-generalizable (non-transferable) interactions do not appear as faithful concepts, even though they still satisfy the criteria of sparsity and universal matching in Theorem 2.

Definition 1 (Transferability of interaction primitives). Given m different DNNs trained for the same task, $v^{(1)}, v^{(2)}, \ldots, v^{(m)}$ , we use AND-OR interactions to explain the output score $v^{(i)}(\mathbf{x})$ of each DNN $v^{(i)}$ on the input sample x. Let $\Omega^{\mathrm{and},(i)} = \{S \subseteq N : |I_{\mathrm{and}}^{(i)}(S|\mathbf{x})| > \tau^{(i)}\}$ and $\Omega^{\mathrm{or},(i)} = \{S \subseteq N : |I_{\mathrm{or}}^{(i)}(S|\mathbf{x})| > \tau^{(i)}\}$ denote a set of sparse AND interaction primitives and a set of sparse OR interaction primitives, respectively. Then, the set of generalizable AND and the set of generalizable OR interaction primitives for the i-th DNN, are defined as $\Omega_{shared}^{and} = \bigcap_{i=1}^{m} \Omega^{\mathrm{and},(i)}$ and $\Omega_{shared}^{or} = \bigcap_{i=1}^{m} \Omega^{\mathrm{or},(i)}$ , respectively. The generalization power of AND and OR interactions of the i-th DNN, can be measured by $s_{\mathrm{and}}^{(i)} = |\Omega_{\mathrm{shared}}^{\mathrm{and}}| / |\Omega^{\mathrm{and},(i)}|$ and $s_{\mathrm{or}}^{(i)} = |\Omega_{\mathrm{shared}}^{\mathrm{or}}| / |\Omega^{\mathrm{or},(i)}|$ , respectively.

Definition 1 introduces the generalization power of interaction primitives. A larger value signifies higher transferability and, consequently, more generalizable interactive primitives.

# 2.3 EXTRACTING GENERALIZABLE INTERACTION PRIMITIVES

Neither of the aforementioned two challenges has been adequately tackled in previous interaction studies. In essence, interactions are determined by the decomposition of the network output $v(\mathbf{x}) = v_{\mathrm{and}}(\mathbf{x}) + v_{\mathrm{or}}(\mathbf{x})$ . Thus, if we rewrite the decomposition as $v_{\mathrm{and}}(\mathbf{x}_T) = 0.5v(\mathbf{x}_T) + \gamma_T$ and $v_{\mathrm{or}}(\mathbf{x}_T) = 0.5v(\mathbf{x}_T) - \gamma_T$ , then the learning of the best decomposition is equivalent to learning a set of $\{\gamma_T\}$ . Here, the parameter $\gamma_T \in R$ for a subset $T \subseteq N$ determines a specific decomposition between $v_{\mathrm{and}}(\mathbf{x}_T)$ and $v_{\mathrm{or}}(\mathbf{x}_T)$ . Therefore, our goal is to learn the appropriate parameters $\{\gamma_T\}$ that reduce the aforementioned uncertainty of interaction primitives and boost their generalization power.

To this end, to alleviate the uncertainty of the interactions, the most intuitive approach is to learn the sparsest interactions, considering the principle of Occam's Razor, as follows. It is because the sparsest (or simplest) explanation is usually considered as the most faithful explanation.

$$
\min _ {\{\gamma_ {T} \}} \| \mathbf {I} _ {\text { and }} \| _ {1} + \| \mathbf {I} _ {\text { or }} \| _ {1}, \tag {4}
$$

where $\mathbf{I}_{\mathrm{and}} = [I_{\mathrm{and}}(T_1|\mathbf{x}), \ldots, I_{\mathrm{and}}(T_{2^n}|\mathbf{x})]^{\intercal}, \mathbf{I}_{\mathrm{or}} = [I_{\mathrm{or}}(T_1|\mathbf{x}), \ldots, I_{\mathrm{or}}(T_{2^n}|\mathbf{x})]^{\intercal} \in \mathbb{R}^{2^n}, T_k \subseteq N.$

The above $\ell_1$ norm loss promotes the sparsity of both AND interactions and OR interactions.

# 2.3.1 ONLY ACHIEVING SPARSITY IS NOT ENOUGH

Although the sparsity can be used to reduce the uncertainty of interactions, the sparsity of interactions w.r.t. each single input sample obtained in Equation (4) does not fully solve the above two challenges. First, Ren et al. (2023c) have found that the extraction of high-order interactions is usually sensitive to small noises in input variables, where the order is defined as the number of input variables in S, i.e., $\text{order}(S) = |S|$ . It means that when different noises are added to the input samples, the algorithm may extract fully different high-order interactions $^{6}$ . Similarly, this will also hurt the generalization power of interaction primitives over different samples, when these samples contain similar sets of input variables.

Second, optimizing the loss in Equation (4) may lead to diverse solutions. Given different initial states, the loss in Equation (4) may learn two different sets of parameters $\{\gamma_{T}\}$ as two local minima with similar loss values, while the two sets of parameters $\{\gamma_{T}\}$ generate two different sets of interactions. We conducted experiments to illustrate this point. Given a pre-trained BERT model (Devlin et al., 2019) and an input sentence x on the SST-2 dataset for sentiment classification, we learned the parameters $\{\gamma_{T}\}$ to extract sparse interaction primitives $^{4}$ . In this experiment, we repeatedly extracted two sets of AND-OR interactions by applying two different sets of initialized parameters $\{\gamma_{T}\}$ , which are denoted by $(\mathcal{A}^{\mathrm{and}}, \mathcal{A}^{\mathrm{or}})$ and $(\mathcal{B}^{\mathrm{and}}, \mathcal{B}^{\mathrm{or}})$ . $A^{and} = \{S \subseteq N : |I_{\mathrm{and}}(S|\mathbf{x})| > \tau_{A^{\mathrm{and}}}\}$ denotes the set of AND interaction primitives extracted by a certain initialization of $\{\gamma_{T}\}$ , where the parameter $\tau_{A^{\mathrm{and}}}$ was determined to ensure that each set selected the most salient K = 100 interactions. We used the transferability of interaction primitives in Definition 1, $S_{and} = |A^{and} \cap B^{and}|/|A^{and}|$ and $S_{or} = |A^{or} \cap B^{or}|/|A^{or}|$ , to measure the diversity of interactions caused by the different parameter initializations. Table 2 shows that given different initial states, optimizing the loss in Equation (4) usually extracted two dramatically different sets of AND-OR interactions with only 21% overlap. Figure 12 further shows the top 5 AND-OR interaction primitives extracted from BERT model on the same input sentence, which illustrates that given different initial states, the loss in Equation (4) would learn different AND-OR interactions.

Third, prioritizing sparsity cannot guarantee high generalization power across different models. Since a DNN may simultaneously learn common knowledge shared by different DNNs and be overfitted to some out-of-the-distribution patterns, different DNNs may only share partial interaction primitives. We believe that the shared common interactions are more faithful, so the transferability is another way to guarantee the generalization power of interaction primitives. Therefore, we hope to formulate and extract common interactions that are generalizable through different DNNs.

We conducted experiments to illustrate the difference between interactions extracted from two DNNs by using Equation (4). We used BERT $_{BASE}$ $v_{base}$ and BERT $_{LARGE}$ $v_{large}$ (Devlin et al., 2019) for the task of sentiment classification. Specifically, given an input sentence x, we learned two sets of parameters $\{\gamma_{T}^{base}\}$ and $\{\gamma_{T}^{large}\}$ for the BERT-base model and the BERT-large model, respectively. Then we extracted two sets of AND-OR interactive concepts ( $\Omega^{and,base}$ , $\Omega^{or,base}$ ) and ( $\Omega^{and,large}$ , $\Omega^{or,large}$ ), respectively. Subsequently, we computed the transferability of the extracted interaction primitives according to Definition 1. Figure 3(a) shows that the transferability of the extracted AND-OR interaction primitives was much lower than interactions proposed in this study.

# 2.3.2 EXTRACTING GENERALIZABLE INTERACTIONS

As discussed above, the sparsity alone is not enough to tackle the aforementioned challenges. Therefore, in this study, we propose to use the generalization power as a straightforward purpose, to

boost the faithfulness of interactions. Meanwhile, the sparsity of interactions is also supposed to be guaranteed. Given a total of m DNNs $v^{(1)}, v^{(2)}, \ldots, v^{(m)}$ trained for the same task, the objective of extracting generalizable interactions shared by the m DNNs is revised from Equation (4), as follows.

$$
\min _ {\left\{\gamma_ {T} ^ {(1)}, \dots , \gamma_ {T} ^ {(m)} \right\}} \| \operatorname{rowmax} \left(\mathbb {I} _ {\text { and }}\right) \| _ {1} + \| \operatorname{rowmax} \left(\mathbb {I} _ {\text { or }}\right) \| _ {1}, \tag {5}
$$

where $\mathbb{I}_{\mathrm{and}} = \left[\mathbf{I}_{\mathrm{and}}^{(1)} \mathbf{I}_{\mathrm{and}}^{(2)} \ldots \mathbf{I}_{\mathrm{and}}^{(m)}\right] \in \mathbb{R}^{2^{n} \times m}$ and $\mathbb{I}_{\mathrm{or}} = \left[\mathbf{I}_{\mathrm{or}}^{(1)} \mathbf{I}_{\mathrm{or}}^{(2)} \ldots \mathbf{I}_{\mathrm{or}}^{(m)}\right] \in \mathbb{R}^{2^{n} \times m}$ . $\mathbf{I}_{\mathrm{and}}^{(i)} = [I_{\mathrm{and}}^{(i)}(T_{1}|\mathbf{x}), \ldots, I_{\mathrm{and}}^{(i)}(T_{2^{n}}|\mathbf{x})]^{\intercal} \in \mathbb{R}^{2^{n}}$ and $\mathbf{I}_{\mathrm{or}}^{(i)}$ represent the all $2^{n}$ AND-OR interactions extracted from the i-th DNN, $T_{k} \subseteq N$ . The matrix operator rowmax() computes the $\ell_{\infty}$ norm of each row within the matrix, i.e., rowmax( $\mathbb{I}_{\mathrm{and}}$ ) = $[||\mathbb{I}_{\mathrm{and}}[1,:]||_{\infty}, \ldots, ||\mathbb{I}_{\mathrm{and}}[2^{n}, :]||_{\infty}]^{\intercal} \in R^{2^{n}}$ . For each specific subset of variables $T_{k} \subseteq N$ , the rowmax() operation returns the most salient interaction strength over all m interactions from the m DNNs. Please see Appendix F for more discussion on the matrix $I_{and}$ .

Unlike Equation (4), the revised loss in Equation (5) only penalizes the most salient interactions over all $m$ interactions extracted from $m$ DNNs, with respect to each subset $T_{k} \subseteq N$ . This loss function ensures that if a DNN encodes a strong interaction w.r.t. the set $T_{k}$ , then we can also extract the same interaction w.r.t. $T_{k}$ from the other $m - 1$ DNNs without a penalty. The $\ell_1$ norm also makes that the $m$ DNNs share similar sets of sparse interactions. Considering the sparsity of interactions, for most subset $T_{k}$ , the effect $I_{\mathrm{and/or}}^{(i)}(T_k|\mathbf{x})$ is supposed to keep almost zero on all $m$ DNNs.

Just like in Equation (4), we decompose the output of the $i$ -th DNN as $v_{\mathrm{and}}^{(i)}(\mathbf{x}_T) = 0.5v^{(i)}(\mathbf{x}_T) + \gamma_T^{(i)}$ and $v_{\mathrm{or}}^{(i)}(\mathbf{x}_T) = 0.5v^{(i)}(\mathbf{x}_T) - \gamma_T^{(i)}$ to compute two vectors of AND-OR interactions, $\mathbf{I}_{\mathrm{and}}^{(i)}$ and $\mathbf{I}_{\mathrm{or}}^{(i)}$ .

Redundancy of interactions. However, it is important to emphasize that only penalizing the largest interaction among the m DNNs in Equation (5) still faces the redundancy problem. Specifically, for each i-th DNN, we compute a total of $2^{n}$ AND interactions and $2^{n}$ OR interactions w.r.t. different subsets $T \subseteq N$ . Some of these $2^{n+1}$ interactions, denoted by the set $\Omega_{\max}^{(i)}$ , are selected by the loss in Equation (5) as the most salient interactions over m DNNs, while the set of other unselected interactions are denoted by $\Omega_{\text{others}}^{(i)} = \{T \subseteq N\} \setminus \Omega_{\max}^{(i)}$ . Then, the redundancy problem is caused by a short-cut solution to the loss minimization in Equation (5), i.e., using unselected not-so-salient interactions in $\Omega_{\text{others}}^{(i)}$ to represent numerical effects of selected interactions $\Omega_{\max}^{(i)}$ , as discussed in Challenge 1 in Section 2.2. As a short-cut solution to Equation (5), this may also reduce the strength of the penalized salient interactions in Equation (5), but generates lots of redundant interacitons.

Therefore, we revise the loss in Equation (5) to add penalties on unselected interactions to avoid the short-cut solution with a coefficient of $\alpha$ , as follows $^{7}$ .

$$
\min _ {\left\{\gamma_ {T} ^ {(1)}, \dots , \gamma_ {T} ^ {(m)} \right\}} \left(\| \operatorname{rowmax} \left(\mathbb {I} _ {\text { and }}\right) \| _ {1} + \| \operatorname{rowmax} \left(\mathbb {I} _ {\text { or }}\right) \| _ {1}\right) + \alpha \left(\| \mathbb {I} _ {\text { and }} \| _ {1} + \| \mathbb {I} _ {\text { or }} \| _ {1}\right), \tag {6}
$$

where $\alpha \in [0,1]$ is a positive scalar. We extend the notation of the $\ell_1$ norm $\| \cdot \| _1$ to represent the sum of the absolute values of all elements in a given vector or matrix. It is worth noting that the generalization power of interactions is guaranteed by the rowmax() function in Equation (5), which assigns much higher penalties to non-generalizable interactions than generalizable interactions.

Sharing decomposition between DNNs. Optimizing Equation (6) is challenging $^{8}$ . To address this challenge, we introduce a set of strategies to facilitate the optimization process. We assume that when all m DNNs are sufficiently trained, these DNNs tend to have similar decompositions of AND interactions and OR interactions, i.e., obtaining similar parameters, $\forall T \subseteq N$ , $\gamma_{T}^{(1)} \approx \gamma_{T}^{(2)} \approx \cdots \approx \gamma_{T}^{(m)}$ . To achieve this, we introduce two types of parameters for $\gamma_{T}^{(i)}$ , $\gamma_{T}^{(i)} = \bar{\gamma}_{T} + \hat{\gamma}_{T}^{(i)}$ , where $\bar{\gamma}_{T}$ represents the common decomposition shared by all DNNs, and $\hat{\gamma}_{T}^{(i)}$ represents the decomposition specific to each i-th DNN. We constrain the significance of the unshared decomposition by using a bound $|\hat{\gamma}_{T}^{(i)}| < \tau_{\gamma}^{(i)}$ , where $\tau_{\gamma}^{(i)} = 0.5 \cdot E_{x}[|v^{(i)}(\mathbf{x}) - v^{(i)}(\mathbf{x}_{\emptyset})|]$ . During the training process, if $|\hat{\gamma}_{T}^{(i)}| > \tau_{\gamma}^{(i)}$ , then we set $\hat{\gamma}_{T}^{(i)} = \tau_{\gamma}^{(i)} \cdot \text{sign}(\hat{\gamma}_{T}^{(i)})$ .

Modeling noises. Furthermore, we have identified a potential limitation in the definition of the interactions, i.e., the sensitivity to noise. Let us assume that the output of the $i$ -th DNN has a small noise. We represent such noises by adding a small Gaussian noise $\epsilon_T \sim \mathcal{N}(0, \sigma^2)$ to the network output $v_{\mathrm{and}}^{'(i)}(\mathbf{x}_T) = v_{\mathrm{and}}^{(i)}(\mathbf{x}_T) + \epsilon_T^{(i)}$ . In this case, we can derive that $I_{\mathrm{and}}^{'(i)}(T) = I_{\mathrm{and}}^{(i)}(T) + \sum_{T' \subseteq T} (-1)^{|T| - |T'|} \epsilon_T^{(i)}$ . We prove that the variance of $I_{\mathrm{and}}^{'(i)}(T)$ caused by the Gaussian noises is $\mathbb{E}_{\epsilon_T \sim \mathcal{N}(0, \sigma^2)}[I_{\mathrm{and}}^{'(i)}(T) - E_{\forall S, \epsilon_S \sim \mathcal{N}(0, \sigma^2)} I_{\mathrm{and}}^{'(i)}(S)]^2 = 2^{|T|} \sigma^2$ (please see Appendix D for details). Similarly, the variance of $I_{\mathrm{or}}^{'(i)}(T)$ is also $2^{|T|} \sigma^2$ for OR interactions. It means that the variance/instability of interactions increases exponentially with the order of the interaction $|T|$ .

Therefore, we propose to directly learn the error term $\epsilon_T^{(i)}$ based on Equation (6) to remove tiny noisy signals, which are unavoidable in real data but cannot be modeled as AND-OR interactions, i.e., setting $v^{(i)}(\mathbf{x}_T) = v_{\mathrm{and}}^{(i)}(\mathbf{x}_T) + v_{\mathrm{or}}^{(i)}(\mathbf{x}_T) + \epsilon_T^{(i)}$ , in order to enhance the robustness of our interaction extraction process. The error term is constrained to a small range $|\epsilon_T^{(i)}| < \tau_\epsilon^{(i)}$ , subject to $\tau_\epsilon^{(i)} = 0.02 \cdot |v^{(i)}(\mathbf{x}) - v^{(i)}(\mathbf{x}_{\emptyset})|$ . During the training process, if $|\epsilon^{(i)}| > \tau_\epsilon^{(i)}$ , then we set $|\epsilon^{(i)}| = \tau_\epsilon^{(i)} \cdot \mathrm{sign}(\epsilon_T^{(i)})$ .

Then, we conducted experiments to examine whether the extracted AND-OR interactions could still accurately explain the network output, when removing the error term. We followed experimental settings in Section 3 to extract interactions on both BERT $_{BASE}$ and BERT $_{LARGE}$ models. We computed the matching error $e(\mathbf{x}_{T}) = |v(\mathbf{x}_{T}) - v^{\mathrm{approx}}(\mathbf{x}_{T})|$ , where $v^{\mathrm{approx}}(\mathbf{x}_{T})$ was the network output approximated by all interactions based on Theorem 2. Figure 13 shows matching errors of all masked samples w.r.t. all subsets $T \subseteq N$ , when we sorted the network outputs for all $2^{n}$ masked samples in a descending order. It shows that the real network output was well approximated by interactions.

# 3 EXPERIMENT

In this section, we conducted experiments to verify the sparsity and generalization power of the interaction primitives extracted by our proposed method on the following three tasks.

Task1: sentiment classification with language models. We jointly extracted two sets of AND-OR interaction primitives from the BERT $_{BASE}$ model and the BERT $_{LARGE}$ model (Devlin et al., 2019) by following Equation (6). We finetuned the pre-trained BERT $_{BASE}$ model and the BERT $_{LARGE}$ model on the SST-2 dataset (Socher et al., 2013) for sentiment classification. For each input sentence x containing n tokens $^{9}$ , we analyzed the log-odds output of the ground-truth label, i.e., $v(\mathbf{x}) = \log \frac{p(y=y^{\mathrm{truth}}|\mathbf{x})}{1-p(y=y^{\mathrm{truth}}|\mathbf{x})}$ by following (Deng et al., 2022).

Task2: dialogue task with large language models. We extracted two sets of AND-OR interaction primitives from the pre-trained LLaMA model (Touvron et al., 2023) and OPT-1.3B model (Zhang et al., 2022b). We explained the DNNs' outputs on the SQuAD dataset (Rajpurkar et al., 2016). We took the first several words of each document in the dataset as the input of a DNN, and let the DNN predict the next word. For each input sentence x containing n words $^{9}$ , we analyzed the log-odds output of the $(n+1)$ -th word that was associated with the highest probability by the DNN, $y^{max}$ , i.e., $v(\mathbf{x}) = \log \frac{p(y=y^{\max}|\mathbf{x})}{1-p(y=y^{\max}|\mathbf{x})}$ .

Task3: image classification task with vision models. We extracted two sets of AND-OR interaction primitives from the ResNet-20 model (He et al., 2016) and the VGG-16 model (Simonyan & Zisserman, 2015), which were trained on the MNIST dataset (LeCun, 1998). These models were trained to classify the digit “3” from other digits. In practice, considering the $2^{n}$ computational complexity, we have followed settings in (Li & Zhang, 2023b), who labeled a few important input patches in the image as input variables. For each input image x containing n patches $^{9}$ , we analyzed the scalar output before the softmax layer corresponding to the digit “3.”

Sparsity of the extracted primitives. We aggregated all AND-OR interactions from various samples, and draw their strength in a descending order in Figure 2. This figure compares the curve of interaction strength $|I(S)|$ , $S \subseteq N$ between our extracted interactions, the traditional interactions (Li

![](images/bcdd2d971c85a4568685b03eb84f286bdcd95b214d295753925c035c3b19bd4d.jpg)

Figure 2: Strength of AND-OR interactions $\log|I(S|\mathbf{x})|$ over different samples in a descending order. All interactions above the dash line had much more significant effect (shown in a log space) and were considered as salient interactions.   
![](images/d36a6c53e8b1259aac06a423b5de602c7fd8782f1e255c74d97920a95e5ac1f8.jpg)

<details>
<summary>line</summary>

| # of salient interactions | generalization power (ratio of shared interactions) - Series 1 | generalization power (ratio of shared interactions) - Series 2 | generalization power (ratio of shared interactions) - Series 3 | generalization power (ratio of shared interactions) - Series 4 |
| ------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| 50                        | 0.2                                                           | 0.3                                                           | 0.2                                                           | 0.1                                                           |
| 150                       | 0.4                                                           | 0.5                                                           | 0.3                                                           | 0.2                                                           |
| 250                       | 0.6                                                           | 0.7                                                           | 0.4                                                           | 0.3                                                           |
| 350                       | 0.7                                                           | 0.8                                                           | 0.5                                                           | 0.4                                                           |
| 450                       | 0.8                                                           | 0.9                                                           | 0.6                                                           | 0.5                                                           |
</details>

(a)

![](images/a539b18fea9f5d852d33b6f158429239c45ed20b6fef649dbff354db135d8f65.jpg)

<details>
<summary>line</summary>

| # of salient interactions | Line 1 | Line 2 |
| ------------------------- | ------ | ------ |
| 50                        | 0.5    | 0.2    |
| 150                       | 0.6    | 0.3    |
| 250                       | 0.65   | 0.4    |
| 350                       | 0.7    | 0.45   |
| 450                       | 0.75   | 0.5    |
</details>

(b)

![](images/78864bac5148318ce6f5b673d783aa0478087a7ed6f1feb9d710e97d92767f5b.jpg)

<details>
<summary>line</summary>

| # of salient interactions | Line 1 | Line 2 | Line 3 |
| ------------------------- | ------ | ------ | ------ |
| 10                        | 0.65   | 0.50   | 0.30   |
| 30                        | 0.70   | 0.60   | 0.35   |
| 50                        | 0.75   | 0.70   | 0.40   |
| 70                        | 0.80   | 0.75   | 0.45   |
| 90                        | 0.85   | 0.80   | 0.50   |
| 110                       | 0.85   | 0.85   | 0.55   |
</details>

(c)

![](images/f93bead42e691a1782885e336ebcab24230c88c1e6f9901c50b4a2b622a52b54.jpg)  
And Interaction (Ours) $S_{and}$   
Or Interaction (Ours) $S_{or}$   
- And Interaction (Traditional) $S_{and}$   
- Or Interaction (Traditional) $S_{or}$   
- And Interaction (Harsanyi) $S_{and}$

Figure 3: Generalization power (measured by $s_{and}$ and $s_{or}$ ) of the extracted primitives interactions.

& Zhang, 2023b) $^{10}$ (namely, Traditional) and the original Harsanyi interactions (Ren et al., 2023a) (namely, Harsanyi). The competing method (Li & Zhang, 2023b) (Traditional in Figure 2) extracts the sparest interactions according to Equation (4), and the original Harsanyi interactions (Ren et al., 2023a) (Harsanyi in Figure 2) extracts interactions according to Equation (1) $^{11}$ . We found that most of the interactions had negligible effect. Although the proposed method reduced the sparsity a bit, the extracted interactions were still sparse enough to be considered as primitive inference patterns. For each DNN, we further set a threshold $\tau^{(i)}$ to collect a set of salient interactions from this DNN as interaction primitives, i.e., $\tau^{(i)} = 0.05 \cdot \max_{S} |I(S|\mathbf{x})|$ .

Generalization power of the extracted interaction primitives. We took the most salient k interactions from each i-th DNN as the set of AND-OR interaction primitives, i.e., $|\Omega^{\mathrm{and},(\mathrm{i})}| = |\Omega^{\mathrm{or},(\mathrm{i})}| = k, i \in \{1, 2\}$ . We used the metric $s_{and}$ and $s_{or}$ in Definition 1 to measure the generalization power of interactions extracted from two DNNs. Figure 3 shows the generalization power of interactions when we computed $s_{and}$ and $s_{or}$ based on different numbers k of most salient interactions. We found that the set of AND-OR interactions extracted from the proposed method exhibited higher generalization power than interactions extracted from the traditional method.

Low-order interaction primitives are more stable. Furthermore, we compared the ratio of shared interactions of different orders, i.e., order $(S)=|S|$ . For interactions of each order o, we computed the overall strength of all positive interactions and that of all negative interactions of the i-th DNN, which were shared by other DNNs, as $Shared^{+, (i)}(o)=\sum_{\mathrm{op}\in\{\mathrm{and},\mathrm{or}\}}\sum_{S\in\Omega_{\mathrm{shared}}^{\mathrm{op},(i)},|S|=o}\max(0,I_{\mathrm{op}}^{(i)}(S|\mathbf{x}))$ and $Shared^{-,(i)}(o)=\sum_{\mathrm{op}\in\{\mathrm{and},\mathrm{or}\}}\sum_{S\in\Omega_{\mathrm{shared}}^{\mathrm{op},(i)},|S|=o}\min(0,I_{\mathrm{op}}^{(i)}(S|\mathbf{x}))$ , respectively. Besides, $All^{+, (i)}(o)=\sum_{\mathrm{op}\in\{\mathrm{and},\mathrm{or}\}}\sum_{S\in\Omega^{\mathrm{op},(i)},|S|=o}\max(0,I_{\mathrm{op}}^{(i)}(S|\mathbf{x}))$ and $All^{-,(i)}(o)=\sum_{\mathrm{op}\in\{\mathrm{and},\mathrm{or}\}}\sum_{S\in\Omega^{\mathrm{op},(i)},|S|=o}\min(0,I_{\mathrm{op}}^{(i)}(S|\mathbf{x}))$ denote the overall strength of salient positive interactions and that of salient negative interactions, respectively. In this way, Figure 4 reports $(Shared^{+, (i)},Shared^{-,(i)})$ and $(All^{+, (i)},All^{-,(i)})$ for different orders within the three tasks. It shows that low-order interactions were more likely to be shared by different DNNs than high-order interactions. Besides, Figure 4 further shows a higher ratio of interactions extracted by the proposed method were shared by different DNNs than interactions extracted by the traditional method. In particular, the high similarity of interactions between ResNet-20 and VGG-16 shows that although two DNNs for the same tasks had fully different architectures, there probably existed a set of ultimate interactions for a task, and different well-optimized DNNs were likely to converge to such interactions.

Visualization of the shared interaction primitives across different DNNs. We also visualize the shared and distinctive interaction primitives in Figure 5. This figure shows that generalizable interac-

![](images/348579ed96c700aeedc51e026099da51eb97ad1876963f7a27475428be6fefb9.jpg)

Figure 4: Overall interactions and shared interactions. The red and black bars show the overall strength of positive interactions $All^{+, (i)}(o)$ and that of negative interactions $All^{-, (i)}(o)$ of each o-th order. The orange and green bars indicate the strength of positive interactions that are shared by the other DNN $Shared^{+, (i)}(o)$ and that of the shared negative interactions $Shared^{-, (i)}(o)$ , respectively.   
![](images/bc120673ce8daf426f26e9cfc4dffce7a22d1f8f035ac9bee8e61a06877d3c9c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Our interactions"] --> B["Distinctive interactions for the BERTBASE model"]
    A --> C["Shared interactions"]
    A --> D["Distinctive interactions for the BERTLARGE model"]
    E["Traditional interactions"] --> F["Distinctive interactions for the BERTBASE model"]
    E --> G["Shared interactions"]
    E --> H["Distinctive interactions for the BERTLARGE model"]
    B --> I["Just cold"]
    B --> J["wet -iz"]
    B --> K["Just wet"]
    B --> L["Ring just wet without rain"]
    B --> M["-iz -zle without"]
    C --> N["Ring just cold wet Seattle -iz -zle without rain -wear"]
    C --> O["cold"]
    C --> P["wet"]
    C --> Q["rain -wear"]
    C --> R["just cold"]
    C --> S["rain"]
    C --> T["Ring"]
    C --> U["without"]
    D --> V["-wear"]
    D --> W["cold"]
    D --> X["wet"]
    D --> Y["-iz without"]
    D --> Z["-iz rain"]
    F --> AA["Ring just cold wet Seattle -iz -zle without rain -wear"]
    F --> AB["Seattle -wear"]
    F --> AC["without rain"]
    F --> AD["Ring without"]
    F --> AE["without -wear"]
    F --> AF["rain -wear"]
    F --> AG["wet Seattle"]
    G --> AH["AND"]
    G --> AI["OR"]
```
</details>

Figure 5: Visualization of the shared and distinctive interaction primitives across different DNNs. We selected some of salient interactions from the most salient k = 50 AND-OR interactions in each DNN. The black and gray color show the AND interactions and the OR interactions, respectively. The left and right column show the distinctive interactions extracted from the BERT $_{BASE}$ model and the BERT $_{LARGE}$ model, respectively. The middle column shows the shared interactions extracted from both models. Please see Appendix J for more interactions.

tions shared by different models can be regarded as more reliable concepts, which consistently contribute salient interaction effects to the output of different DNNs. In comparison, non-generalizable interactions, which are sometimes over-fitted by a single model, may appear as out-of-distribution features. From this perspective, we consider generalizable interactions as relatively faithful concepts that often have a significant impact on the inference of DNNs. Figure 5 further shows that, our method extracted much more shared interactions than the traditional interaction-extraction method, which shows that our method could obtain more stable explanation of the inference logic of a DNN. It is because the interactions shared by different DNNs were usually considered more faithful.

# 4 CONCLUSION, DISCUSSIONS, AND FUTURE CHALLENGES

In this paper, we proposed a method to extract generalizable interaction primitives. The sparsity and universal-matching property of interactions provide lots of evidence to faithfully explain DNNs with interactions. Thus, in this paper, we propose to further improve the generalization power of interactions, which adds the last piece of the puzzle of interaction primitives. Compared to traditional interactions, interactions shared by different DNNs are more likely to be the underlying primitives that shape the DNN's output. Furthermore, the extraction of interaction primitives also contributes to real applications. For example, it can assist in learning optimal baseline values for Shapley values (Ren et al., 2023b) and explaining the representation limits of Bayesian networks (Ren et al., 2023c). In addition, the extraction of generalizable interaction primitives shared by different DNNs provide a new perspective to formulating the out-of-distribution (OOD) features. Previous studies usually treated an entire sample as an OOD sample, whereas our work redefines the OOD problem at the level of detailed interactions, i.e., unshared interactions can be regarded as OOD information.

In order to faithfully explain the decision-making logic of a DNN in a symbolic manner, a theory system of interaction-based explanations has been built up on over 20 papers (surveyed by (Ren et al., 2024)). However, there is still a long way to achieve a comprehensive system of explaining DNNs. For example, the following four problems have not been faithfully solved. For example, we still need to investigate how to use interactions (1) to concisely explain the complex learning dynamics of a DNN, (2) to distinguish logical reasoning used by the DNN, (3) to evaluate and learn from the detailed interaction logic of a DNN, and (4) to extract generalizable interactions for future performance improvement.

# ACKNOWLEDGMENTS

This work is partially supported by the National Science and Technology Major Project (2021ZD0111602), the National Nature Science Foundation of China (62276165,92370115), Shanghai Natural Science Foundation (21JC1403800,21ZR1434600).

# ETHICS STATEMENT

This paper aims to extract generalizable interaction primitives that are shared by different DNNs. This paper utilizes publicly released datasets which have been widely accepted by the machine learning community. This paper does not involve human subjects and does not include potentially harmful insights, methods, or applications. The paper also does not involve discrimination/bias/fairness issues, as well as privacy and security issues. There are no ethical issues with this paper.

# REPRODUCIBILITY STATEMENT

We provide proofs for the theoretical results of this study in Appendix C to D. We also provide experimental details in Section 3 and Appendix O.

# REFERENCES

BAAI. Aquila-7b. 2023. URL https://huggingface.co/BAAI/Aquila-7B.   
David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6541–6549, 2017.   
Huiqi Deng, Qihan Ren, Hao Zhang, and Quanshi Zhang. Discovering and explaining the representation bottleneck of dnns. In International Conference on Learning Representations, 2022.   
Huiqi Deng, Na Zou, Mengnan Du, Weifu Chen, Guocan Feng, Ziwei Yang, Zheyang Li, and Quanshi Zhang. Unifying fourteen post-hoc attribution methods with taylor interactions. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT, 2019.   
Amil Dravid, Yossi Gandelsman, Alexei A. Efros, and Assaf Shocher. Rosetta neurons: Mining the common units in a model zoo. IEEE International Conference on Computer Vision, 2023.   
John C Harsanyi. A simplified bargaining model for the n-person cooperative game. International Economic Review, 4(2):194–220, 1963.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.   
Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International Conference on Machine Learning, pp. 2668–2677. PMLR, 2018.

Yann LeCun. The mnist database of handwritten digits. 1998. URL http://yann.lecun.com/exdb/mnist/.   
Mingjie Li and Quanshi Zhang. Technical note: Defining and quantifying and-or interactions for faithful and concise explanation of dnns. arXiv preprint arXiv:2304.13312, 2023a.   
Mingjie Li and Quanshi Zhang. Does a neural network really encode symbolic concept? International Conference on Machine Learning, 2023b.   
Dongrui Liu, Huiqi Deng, Xu Cheng, Qihan Ren, Kangrui Wang, and Quanshi Zhang. Towards the difficulty for a deep neural network to learn concepts of different complexities. Advances in Neural Information Processing Systems, 2024.   
Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. International Conference on Computer Vision, 2017.   
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.   
Jie Ren, Mingjie Li, Qirui Chen, Huiqi Deng, and Quanshi Zhang. Defining and quantifying the emergence of sparse concepts in dnns. IEEE Conference on Computer Vision and Pattern Recognition, 2023a.   
Jie Ren, Zhanpeng Zhou, Qirui Chen, and Quanshi Zhang. Can we faithfully represent masked states to compute shapley values on a dnn? In International Conference on Learning Representations, 2023b.   
Qihan Ren, Huiqi Deng, Yunuo Chen, Siyu Lou, and Quanshi Zhang. Bayesian neural networks avoid encoding perturbation-sensitive and complex concepts. International Conference on Machine Learning, 2023c.   
Qihan Ren, Jiayang Gao, Wen Shen, and Quanshi Zhang. Where we have arrived in proving the emergence of sparse interaction primitives in dnns. International Conference on Learning Representations, 2024.   
Wen Shen, Lei Cheng, Yuxiao Yang, Mingjie Li, and Quanshi Zhang. Can the inference logic of large language models be disentangled into symbolic concepts? arXiv preprint arXiv:2304.01083, 2023.   
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. International Conference on Learning Representations, 2015.   
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.   
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1631–1642, 2013.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. Understanding neural networks through deep visualization. International Conference on Machine Learning, 2015.   
Quanshi Zhang, Xin Wang, Jie Ren, Xu Cheng, Shuyun Lin, Yisen Wang, and Xiangming Zhu. Proving common mechanisms shared by twelve methods of boosting adversarial transferability. arXiv preprint arXiv:2207.11694, 2022a.   
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022b.

Huilin Zhou, Huijie Tang, Mingjie Li, Hao Zhang, Zhenyu Liu, and Quanshi Zhang. Explaining how a neural network play the go game and let people learn. arXiv preprint arXiv:2310.09838, 2023.   
Huilin Zhou, Hao Zhang, Huiqi Deng, Dongrui Liu, Wen Shen, Shih-Han Chan, and Quanshi Zhang. Explaining generalization power of a dnn using interactive concepts. The Association for the Advancement of Artificial Intelligence, 2024.

# A PREVIOUS LITERATURE OF USING INTERACTIONS TO EXPLAIN DNNs

Explaining the knowledge encoded by DNNs is one of the ultimate goals of explainable AI but presents significant challenges. For instance, some studies employed visualization techniques to show the patterns learned by a DNN (Simonyan et al., 2013; Yosinski et al., 2015), and some focused on extracting feature vectors that may be associated with semantic concepts (Simonyan et al., 2013), while some studies learned feature vectors potentially related to concepts (Kim et al., 2018). Dravid et al. (2023) identified convolutional kernels in different DNNs that expressed similar concepts.

However, theoretically, whether the knowledge or the complex inference logic of a DNN can be faithfully represented as symbolic primitive inference patterns still presents significant challenges. Up to now, there is still no universally accepted definition of the knowledge, as it encompasses various aspects of cognitive science, neuroscience, and mathematics. However, if we ignore cognitive and neuroscience aspects, Ren et al. (2023a) have proposed to quantify interactions between input variables encoded by the DNN, to explain the knowledge in the DNN. More crucially, Ren et al. (2024) have derived several theorems as mathematical evidence of considering such interactions as primitive inference patterns encoded by a DNN. Specifically, Ren et al. (2024) proved that DNNs usually only encoded a small number of interactions, under some common conditions $^{2}$ . Besides, these interactions can universally explain the DNN's output score on any arbitrary masked input samples. Li & Zhang (2023b) further discovered the discriminative power of certain interactions.

Besides, Deng et al. (2024) found that different attribution scores estimated by different attribution methods could all be represented as a combination of different interactions. Zhang et al. (2022a) used interactions to explain the mechanism of different methods of boosting adversarial transferability. Ren et al. (2023b) used interactions to define the optimal baseline value for computing Shapley values. Deng et al. (2022) found that for most DNNs it was difficult to learn interactions with median number of input variables, and it was discovered that DNNs and Bayesian neural networks were unlikely to model complex interactions with many input variables (Ren et al., 2023c; Liu et al., 2024). Zhou et al. (2024) used the generalization power of different interactions to explain the generalization power of DNNs.

# B THE CONDITIONS FOR UNIVERSAL MATCHING OF THE DNN OUTPUT

Ren et al. (2024) have proved that a well-trained DNN usually just encodes a limited number of interactions. More importantly, one can just use a few interactions to approximate the DNN's outputs on all $2^{n}$ masked samples $\{\mathbf{x}_S|S\subseteq N\}$ , under the following three common assumptions. (1) The high order derivatives of the DNN output with respect to the input variables are all zero. (2) The DNN works well on the masked samples, and yield higher confidence when the input sample is less masked. (3) The confidence of the DNN does not drop significantly on the masked samples. With these natural assumptions, a well-trained DNN's output on all $2^{n}$ masked samples can be universally approximated by the sum of just a few salient interactions in $\Omega$ , s.t., $|\Omega| \ll 2^{n}$ .

# C PROOF OF THEOREMS

Theorem 2 (Universal matching theorem) Let us be given a DNN $v$ and an input sample $\mathbf{x}$ . For each randomly masked sample $\mathbf{x}_T, T \subseteq N$ , we obtain

$$
v \left(\mathbf {x} _ {T}\right) = v _ {\text { and }} \left(\mathbf {x} _ {T}\right) + v _ {\text { or }} \left(\mathbf {x} _ {T}\right) = \sum_ {S \subseteq T} I _ {\text { and }} (S \mid \mathbf {x} _ {T}) + \sum_ {S \in \{S: S \cap T \neq \emptyset \} \cup \{\emptyset \}} I _ {\text { or }} (S \mid \mathbf {x} _ {T}). \tag {7}
$$

# Proof. (1) Universal matching theorem of AND interactions.

Ren et al. (2023a) have used the Haranyi dividend (Harsanyi, 1963) $I_{\mathrm{and}}(S|\mathbf{x})$ to state the universal matching theorem of AND interactions. The output of a well-trained DNN on all $2^n$ masked samples $\{\mathbf{x}_T|T \subseteq N\}$ could be universally explained by the all interaction primitives in $T \subseteq N$ , i.e., $\forall T \subseteq N$ , $v_{\mathrm{and}}(\mathbf{x}_T) = \sum_{S \subseteq T} I_{\mathrm{and}}(S|\mathbf{x})$ .

Specifically, the AND interaction is defined as $I_{\mathrm{and}}(S|\mathbf{x}) := \sum_{L \subseteq S} (-1)^{|S| - |L|} v_{\mathrm{and}}(\mathbf{x}_L)$ in Equation (1). To compute the sum of AND interactions $\forall T \subseteq N, \sum_{S \subseteq T} I_{\mathrm{and}}(S|\mathbf{x}) =$

$\sum_{S\subseteq T}\sum_{L\subseteq S}(-1)^{|S|-|L|}v_{\mathrm{and}}(\mathbf{x}_{L})$ , we first exchange the order of summation of the set $L\subseteq S\subseteq T$ and the set $S\supseteq L$ . That is, we compute all linear combinations of all sets S containing L with respect to the model outputs $v_{\mathrm{and}}(\mathbf{x}_{L})$ given a set of input variables L, i.e., $\sum_{S:L\subseteq S\subseteq T}(-1)^{|S|-|L|}v_{\mathrm{and}}(\mathbf{x}_{L})$ . Then, we compute all summations over the set $L\subseteq T$ .

In this way, we can compute them separately for different cases of $L \subseteq S \subseteq T$ . In the following, we consider the cases (1) $L = S = T$ , and (2) $L \subseteq S \subseteq T$ , $L \neq T$ , respectively.

(1) When $L = S = T$ , the linear combination of all subsets $S$ containing $L$ with respect to the model output $v_{\mathrm{and}}(\mathbf{x}_L)$ is $(-1)^{|T| - |T|} v_{\mathrm{and}}(\mathbf{x}_L) = v_{\mathrm{and}}(\mathbf{x}_L)$ .   
(2) When $L \subseteq S \subseteq T, L \neq T$ , the linear combination of all subsets $S$ containing $L$ with respect to the model output $v_{\mathrm{and}}(\mathbf{x}_L)$ is $\sum_{S:L \subseteq S \subseteq T} (-1)^{|S| - |L|} v_{\mathrm{and}}(\mathbf{x}_L)$ . For all sets $S: T \supseteq S \supseteq L$ , let us consider the linear combinations of all sets $S$ with number $|S|$ for the model output $v_{\mathrm{and}}(\mathbf{x}_L)$ , respectively. Let $m := |S| - |L|$ , ( $0 \leq m \leq |T| - |L|$ ), then there are a total of $C_{|T| - |L|}^m$ combinations of all sets $S$ of order $|S|$ . Thus, given $L$ , accumulating the model outputs $v_{\mathrm{and}}(\mathbf{x}_L)$ corresponding to all $S \supseteq L$ , then $\sum_{S:L \subseteq S \subseteq T} (-1)^{|S| - |L|} v_{\mathrm{and}}(\mathbf{x}_L) = v_{\mathrm{and}}(\mathbf{x}_L) \cdot \underbrace{\sum_{m=0}^{|T| - |L|} C_{|T| - |L|}^m (-1)^m}_{=0} = 0$ .

Please see the complete derivation of the following formula.

$$
\begin{array}{l} \sum_ {S \subseteq T} I _ {\text { and }} (S | \mathbf {x} _ {T}) = \sum_ {S \subseteq T} \sum_ {L \subseteq S} (- 1) ^ {| S | - | L |} v _ {\text { and }} (\mathbf {x} _ {L}) \\ = \sum_ {L \subseteq T} \sum_ {S: L \subseteq S \subseteq T} (- 1) ^ {| S | - | L |} v _ {\text { and }} (\mathbf {x} _ {L}) \\ = \underbrace {v _ {\text { and }} (\mathbf {x} _ {T})} _ {L = T} + \sum_ {L \subseteq T, L \neq T} v _ {\text { and }} (\mathbf {x} _ {L}) \cdot \underbrace {\sum_ {m = 0} ^ {| T | - | L |} C _ {| T | - | L |} ^ {m} (- 1) ^ {m}} _ {= 0} \tag {8} \\ = v _ {\text { and }} (\mathbf {x} _ {T}). \\ \end{array}
$$

# (2) Universal matching theorem of OR interactions.

According to the definition of OR interactions in Section 2.1, we will derive that $\forall T \subseteq N$ , $v_{\mathrm{or}}(\mathbf{x}_T) = \sum_{S \in \{S: S \cap T \neq \emptyset\} \cup \{\emptyset\}} I_{\mathrm{or}}(S|\mathbf{x}_T) = I_{\mathrm{or}}(\emptyset|\mathbf{x}_T) + \sum_{S: S \cap T \neq \emptyset} I_{\mathrm{or}}(S|\mathbf{x}_T)$ , s.t., $I_{\mathrm{or}}(\emptyset|\mathbf{x}_T) = v_{\mathrm{or}}(\mathbf{x}_{\emptyset})$ .

Specifically, the OR interaction is defined as $I_{\mathrm{or}}(S|\mathbf{x}) := -\sum_{L \subseteq S} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ in Equation (2). Similar to the above derivation of the universal matching theorem of AND interactions, to compute the sum of OR interactions $\forall T \subseteq N, \sum_{S: S \cap T \neq \emptyset} I_{\mathrm{or}}(S|\mathbf{x}_T) = \sum_{S: S \cap T \neq \emptyset} \left[ -\sum_{L \subseteq S} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L}) \right]$ , we first exchange the order of summation of the set $L \subseteq S \subseteq N$ and the set $S: S \cap T \neq \emptyset$ . That is, we compute all linear combinations of all sets $S$ containing $L$ with respect to the model outputs $v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ given a set of input variables $L$ , i.e., $\sum_{S: S \cap T \neq \emptyset, S \supseteq L} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ . Then, we compute all summations over the set $L \subseteq N$ .

In this way, we can compute them separately for different cases of $L \subseteq S \subseteq N, S \cap T \neq \emptyset$ . In the following, we consider the cases (1) $L = N \setminus T$ , (2) $L = N$ , (3) $L \cap T \neq \emptyset, L \neq N$ , and (4) $L \cap T = \emptyset, L \neq N \setminus T$ , respectively.

(1) When $L = N \setminus T$ , the linear combination of all subsets $S$ containing $L$ with respect to the model output $v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ is $\sum_{S: S \cap T \neq \emptyset, S \supseteq L} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L}) = \sum_{S: S \cap T \neq \emptyset, S \supseteq L} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_T)$ . For all sets $S: S \supseteq L, S \cap T \neq \emptyset$ (then $S \neq N \setminus T, S \neq L$ ), let us consider the linear combinations of all sets $S$ with number $|S|$ for the model output $v_{\mathrm{or}}(\mathbf{x}_T)$ , respectively. Let $|S'| := |S| - |L|$ , ( $1 \leq |S'| \leq |T|$ ), then there are a total of $C_{|T|}^{|S'|}$ combinations of all sets $S$ of order $|S|$ . Thus, given $L$ , accumulating the model outputs $v_{\mathrm{or}}(\mathbf{x}_T)$ corresponding to all $S \supseteq L$ , then

$$
\sum_ {S: S \cap T \neq \emptyset , S \supseteq L} (- 1) ^ {| S | - | L |} v _ {\mathrm{or}} (\mathbf {x} _ {N \setminus L}) = v _ {\mathrm{or}} (\mathbf {x} _ {T}) \cdot \underbrace {\sum_ {| S ^ {\prime} | = 1} ^ {| T |} C _ {| T |} ^ {| S ^ {\prime} |} (- 1) ^ {| S ^ {\prime} |}} _ {= - 1} = - v _ {\mathrm{or}} (\mathbf {x} _ {T}).
$$

(2) When $L = N$ (then $S = N$ ), the linear combination of all subsets $S$ containing $L$ with respect to the model output $v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ is $\sum_{S: S \cap T \neq \emptyset, S \supseteq L} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L}) = (-1)^{|N| - |N|} v_{\mathrm{or}}(\mathbf{x}_{\emptyset}) = v_{\mathrm{or}}(\mathbf{x}_{\emptyset})$ .   
(3) When $L \cap T \neq \emptyset, L \neq N$ , the linear combination of all subsets $S$ containing $L$ with respect to the model output $v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ is $\sum_{S: S \cap T \neq \emptyset, S \supseteq L} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ . For all sets $S: S \supseteq L, S \cap T \neq \emptyset$ , let us consider the linear combinations of all sets $S$ with number $|S|$ for the model output $v_{\mathrm{or}}(\mathbf{x}_T)$ , respectively. Let us split $|S| - |L|$ into $|S'|$ and $|S''|$ , i.e., $|S| - |L| = |S'| + |S''|$ , where $S' = \{i | i \in S, i \notin L, i \in N \setminus T\}$ , $S'' = \{i | i \in S, i \notin L, i \in T\}$ (then $0 \leq |S''| \leq |T| - |T \cap L|$ ) and $S' + S'' + L = S$ . In this way, there are a total of $C_{|T| - |T \cap L|}^{|S''|}$ combinations of all sets $S''$ of order $|S''|$ . Thus, given $L$ , accumulating the model outputs $v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ corresponding to all $S \supseteq L$ , then $\sum_{S: S \cap T \neq \emptyset, S \supseteq L} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L}) = 0$ .

$$
v _ {\mathrm{or}} (\mathbf {x} _ {N \setminus L}) \cdot \sum_ {S ^ {\prime} \subseteq N \setminus T \setminus L} \underbrace {\sum_ {| S ^ {\prime \prime} | = 0} ^ {| T | - | T \cap L |} C _ {| T | - | T \cap L |} ^ {| S ^ {\prime \prime} |} (- 1) ^ {| S ^ {\prime} | + | S ^ {\prime \prime} |}} _ {= 0} = 0.
$$

(4) When $L \cap T = \emptyset, L \neq N \setminus T$ , the linear combination of all subsets $S$ containing $L$ with respect to the model output $v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ is $\sum_{S: S \cap T \neq \emptyset, S \supseteq L} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ . Similarly, let us split $|S| - |L|$ into $|S'|$ and $|S''|$ , i.e., $|S| - |L| = |S'| + |S''|$ , where $S' = \{i | i \in S, i \notin L, i \in N \setminus T\}$ , $S'' = \{i | i \in S, i \in T\}$ (then $0 \leq |S''| \leq |T|$ ) and $S' + S'' + L = S$ . In this way, there are a total of $C_{|T|}^{|S''|}$ combinations of all sets $S''$ of order $|S''|$ . Thus, given $L$ , accumulating the model outputs $v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ corresponding to all $S \supseteq L$ , then $\sum_{S: S \cap T \neq \emptyset, S \supseteq L} (-1)^{|S| - |L|} v_{\mathrm{or}}(\mathbf{x}_{N \setminus L}) = v_{\mathrm{or}}(\mathbf{x}_{N \setminus L})$ .

$$
\sum_ {S ^ {\prime} \subseteq N \setminus T \setminus L} \underbrace {\sum_ {| S ^ {\prime \prime} | = 0} ^ {| T |} C _ {| T |} ^ {| S ^ {\prime \prime} |} (- 1) ^ {| S ^ {\prime} | + | S ^ {\prime \prime} |}} _ {= 0} = 0.
$$

Please see the complete derivation of the following formula.

$$
\begin{array}{l} \sum_ {S: S \cap T \neq \emptyset} I _ {\mathrm{or}} (S | \mathbf {x} _ {T}) = \sum_ {S: S \cap T \neq \emptyset} \left[ - \sum_ {L \subseteq S} (- 1) ^ {| S | - | L |} v _ {\mathrm{or}} (\mathbf {x} _ {N \setminus L}) \right] \\ = - \sum_ {L \subseteq N} \sum_ {S: S \cap T \neq \emptyset , S \supseteq L} (- 1) ^ {| S | - | L |} v _ {\text { or }} (\mathbf {x} _ {N \setminus L}) \\ = - \left[ \sum_ {| S ^ {\prime} | = 1} ^ {| T |} C _ {| T |} ^ {| S ^ {\prime} |} (- 1) ^ {| S ^ {\prime} |} \right] \cdot \underbrace {v _ {\mathrm{or}} (\mathbf {x} _ {T})} _ {L = N \setminus T} - \underbrace {v _ {\mathrm{or}} (\mathbf {x} _ {\emptyset})} _ {L = N} \\ - \sum_ {L \cap T \neq \emptyset , L \neq N} \left[ \sum_ {S ^ {\prime} \subseteq N \setminus T \setminus L} \left(\sum_ {| S ^ {\prime \prime} | = 0} ^ {| T | - | T \cap L |} C _ {| T | - | T \cap L |} ^ {| S ^ {\prime \prime} |} (- 1) ^ {| S ^ {\prime} | + | S ^ {\prime \prime} |}\right) \right] \cdot v _ {\mathrm{or}} (\mathbf {x} _ {N \setminus L}) \\ - \sum_ {L \cap T = \emptyset , L \neq N \setminus T} \left[ \sum_ {S ^ {\prime} \subseteq N \setminus T \setminus L} \left(\sum_ {| S ^ {\prime \prime} | = 0} ^ {| T |} C _ {| T |} ^ {| S ^ {\prime \prime} |} (- 1) ^ {| S ^ {\prime} | + | S ^ {\prime \prime} |}\right) \right] \cdot v _ {\mathrm{or}} (\mathbf {x} _ {N \setminus L}) \\ = - (- 1) \cdot v _ {\text {or}} (\mathbf {x} _ {T}) - v _ {\text {or}} (\mathbf {x} _ {\emptyset}) - \sum_ {L \cap T \neq \emptyset , L \neq N} \left[ \sum_ {S ^ {\prime} \subseteq N \setminus T \setminus L} 0 \right] \cdot v _ {\text {or}} (\mathbf {x} _ {N \setminus L}) \\ - \sum_ {L \cap T = \emptyset , L \neq N \setminus T} \left[ \sum_ {S ^ {\prime} \subseteq N \setminus T \setminus L} 0 \right] \cdot v _ {\text { or }} (\mathbf {x} _ {N \setminus L}) \\ = v _ {\mathrm{or}} \left(\mathbf {x} _ {T}\right) - v _ {\mathrm{or}} \left(\mathbf {x} _ {\emptyset}\right) \tag {9} \\ \end{array}
$$

# (3) Universal matching theorem of AND-OR interactions.

With the universal matching theorem of AND interactions and the universal matching theorem of OR interactions, we can easily get $v(\mathbf{x}_{T}) = v_{\mathrm{and}}(\mathbf{x}_{T}) + v_{\mathrm{or}}(\mathbf{x}_{T}) = \sum_{S \subseteq T} I_{\mathrm{and}}(S|\mathbf{x}_{T}) + \sum_{S \in \{S: S \cap T \neq \emptyset\} \cup \{\emptyset\}} I_{\mathrm{or}}(S|\mathbf{x}_{T})$ , thus, we obtain the universal matching theorem of AND-OR interactions.

![](images/565e44944c31b3b6f754ddda40fea147ec7d95b8b7462533187f17512bafbd67.jpg)

Theorem 3 (proved by Harsanyi (1963)). The Shapley value $\phi(i)$ of an input variable $i$ can be explained as a uniform allocation of the AND interactions, i.e., $\phi(i) = \sum_{S \subseteq N: S \ni i} \frac{1}{|S|} I_{\text{and}}(S|\mathbf{x})$ .

# D PROOF OF THE VARIANCE OF AND AND OR INTERACTIONS

In this section, we prove that the variance of $I_{\mathrm{and}}^{\prime (i)}(T)$ caused by the Gaussian noises $\epsilon_T^{(i)}\sim \mathcal{N}(0,\sigma^2)$ is $\mathbb{E}_{\epsilon_T\sim \mathcal{N}(0,\sigma^2)}[I_{\mathrm{and}}^{\prime (i)}(T) - E_{\forall S,\epsilon_S\sim \mathcal{N}(0,\sigma^2)}I_{\mathrm{and}}^{\prime (i)}(S)]^2 = 2^{|T|}\sigma^2$ in Section 2.3.2.

Proof. Given $I_{\mathrm{and}}^{\prime (i)}(T) = I_{\mathrm{and}}^{(i)}(T) + \sum_{T' \subseteq T} (-1)^{|T| - |T'|} \epsilon_{T'}^{(i)}$ , the variance of $I_{\mathrm{and}}^{\prime (i)}(T)$ is $\operatorname{Var}(I_{\mathrm{and}}^{\prime (i)}(T)) = \operatorname{Var}(I_{\mathrm{and}}^{(i)}(T) + \sum_{T' \subseteq T} (-1)^{|T| - |T'|} \epsilon_{T'}^{(i)})$ . As the AND interaction $I_{\mathrm{and}}^{(i)}(T)$ and the Gaussian noise $\epsilon_T^{(i)}$ are independent of each other, then the variance of $I_{\mathrm{and}}^{\prime (i)}(T)$ can be decomposed to $\operatorname{Var}(I_{\mathrm{and}}^{\prime (i)}(T)) = \operatorname{Var}(I_{\mathrm{and}}^{(i)}(T)) + \operatorname{Var}(\sum_{T' \subseteq T} (-1)^{|T| - |T'|} \epsilon_{T'}^{(i)}) = \operatorname{Var}(\sum_{T' \subseteq T} (-1)^{|T| - |T'|} \epsilon_{T'}^{(i)})$ , this is because here $I_{\mathrm{and}}^{(i)}(T)$ can be regarded as a constant.

Since each Gaussian noise $\epsilon_{T}^{(i)} \sim \mathcal{N}(0, \sigma^{2}), \forall T \subseteq N$ is independent and identically distributed, then the variance is $\operatorname{Var}(I_{\text{and}}^{\prime(i)}(T)) = \operatorname{Var}(\sum_{T' \subseteq T}(-1)^{|T| - |T'|} \epsilon_{T'}^{(i)}) = \operatorname{Var}(\epsilon_{T_1'}^{(i)}) + \operatorname{Var}(\epsilon_{T_2'}^{(i)}) + \cdots + \operatorname{Var}(\epsilon_{T_{2^{|T|}}}^{(i)}) = 2^{|T|} \cdot \sigma^{2}$ (there are a total of $2^{|T|}$ subsets for $T' \subseteq T$ ).

# E THE FAITHFULNESS OF THE SPARSITY AND UNIVERSAL-MATCHING.

This section further demonstrates the faithfulness of the sparsity and universal-matching theorem of AND-OR interactions, both theoretically and experimentally.

Theoretically, the faithfulness of the sparsity and universal-matching theorem of AND-OR interactions means that, given an input sample with n variables, we must prove that (1) a well-trained DNN usually just encodes a small number of salient interactions $\Omega$ (Ren et al., 2024), comparing with all $2^{n}$ potential combinations of the input variables in a given input sample, i.e., $|\Omega| \ll 2^{n}$ , and that (2) the network output $v(\mathbf{x}_{S})$ on all $2^{n}$ randomly masked samples $\{x_{S}|S \subseteq N\}$ can be well matched by a few interactions in $\Omega = \{S \subseteq N : |I(S|x)| > \tau\}$ as defined in the Definition of interaction primitives in Section 2.1. These two terms have been proved in Theorem 1, Theorem 2 and Proposition 1.

Then, in practice, considering the $2^{n}$ computational complexity, we have followed settings in (Li & Zhang, 2023b) to extract interactions between a set of randomly selected input variables $N = \{1, 2, \ldots, t\}$ , $(t < n)$ , while other unselected $(n - t)$ input variables remain unmasked, leaving the original state unchanged. In this case, faithfulness does not mean that our interactions explain all the inference logic encoded between all n input variables for a given sample x in the pre-trained DNN. Instead, it only means that the extracted interactions can also accurately match the inference logic encoded between the selected t input variables for a given sample x in the DNN.

Experimental verification. In addition, we have further conducted an experiment to verify the faithfulness when we explain interactions between all input variables. To this end, for the sentiment classification task on the SST-2 dataset in BERT $_{BASE}$ , we selected sentences containing 15 tokens to verify the faithfulness of the sparsity in Proposition 1 and the universal-matching property in Theorem 2. All tokens are selected as input variables. We focused on the matching errors for all masked samples $x_{T}, \forall T \subseteq N$ . Specifically, we observed whether the real network output on the masked sample $v(\mathbf{x}_{T}), \forall T \subseteq N$ can be well approximated by interactions. We have verified that the extracted salient interactions in $\Omega$ faithfully explain the network output, i.e., $\forall T \subseteq N$ , $v(\mathbf{x}_{T}) \approx v(\mathbf{x}_{\emptyset}) + \sum_{\emptyset \neq S \subseteq T: S \in \Omega^{\text{and}}} I_{\text{and}}(S|\mathbf{x}_{T}) + \sum_{S \cap T \neq \emptyset: S \in \Omega^{\text{or}}} I_{\text{or}}(S|\mathbf{x}_{T})$ . Figure 6 illustrates that network output $v(\mathbf{x}_{T}), \forall T \subseteq N$ on all $2^{n}$ randomly masked samples can be well fitted by interactions.

![](images/b10bd183bcb520293f640698f2d95e22cbac225551072f6197c190656ce7af23.jpg)

<details>
<summary>line</summary>

| index of interactions | DNN's output v(x_T) |
| --------------------- | ------------------- |
| 0                     | -2.5                |
| 10000                 | 0.0                 |
| 20000                 | 2.5                 |
| 30000                 | 5.0                 |
| 35000                 | 7.5                 |
</details>

Figure 6: Universal matching of all interaction primitives to the DNN's output, when we use salient interactions to match the DNN's output. Shade area indicates the matching error of different $v(\mathbf{x}_T)$ .

# F DETAILED EXPLANATION OF EQUATION (6)

This section uses a simple example to illustrate Equation (6) more clearly. Let us illustrate how the interaction matrices are formatted in a toy example. Let us consider two pre-trained DNNs $v^{(1)}$ , $v^{(2)}$ and an input sample $\mathbf{x}$ with $N = \{1,2\}$ variables, and take the AND interaction matrix $\mathbb{I}_{\mathrm{and}}$ in Equation (6) as an example (the OR interaction matrix $\mathbb{I}_{\mathrm{or}}$ can be obtained similarly). First, we get all $2^2$ AND interactions extracted from the $i$ -th model $\mathbf{I}_{\mathrm{and}}^{(i)} = [I_{\mathrm{and}}(T_1|\mathbf{x}), I_{\mathrm{and}}(T_2|\mathbf{x}), I_{\mathrm{and}}(T_3|\mathbf{x}), I_{\mathrm{and}}(T_4|\mathbf{x})]^{\intercal} \in \mathbb{R}^{2^2}$ , according to the description of Equation (4). Here, each interaction value $I_{\mathrm{and}}(T_k|\mathbf{x}) \in \mathbb{R}, T_k \subseteq N$ denotes the interaction value of each masked sample $\mathbf{x}_{T_k}$ , which is computed according to Equation (1). Second, the AND interaction matrix $\mathbb{I}_{\mathrm{and}} = [\mathbf{I}_{\mathrm{and}}^{(1)}, \mathbf{I}_{\mathrm{and}}^{(2)}] \in \mathbb{R}^{2^2 \times 2}$ represents the interaction values corresponding to the $2^2$ masked samples for each of the two models.

Let us illustrate how to learn the parameter $\gamma_T^{(i)}$ in Equation (6). The loss in Equation (6) in the above example can be represented as the function of $\{\gamma_T\}$ , as follows.

$$
\begin{array}{l} L o s s = \min _ {\{\gamma_ {T _ {k}} ^ {(1)}, \gamma_ {T _ {k}} ^ {(2)} \}} (\| \text {rowmax} (\mathbb {I} _ {\text {and}}) \| _ {1} + \| \text {rowmax} (\mathbb {I} _ {\text {or}}) \| _ {1}) + \alpha (\| \mathbb {I} _ {\text {and}} \| _ {1} + \| \mathbb {I} _ {\text {or}} \| _ {1}) \\ = \min _ {\{\gamma_ {T _ {k}} ^ {(1)}, \gamma_ {T _ {k}} ^ {(2)} \}} \sum_ {T _ {k} \subseteq N} | \max (\sum_ {T \subseteq S} (- 1) ^ {| S | - | T |} [ 0. 5 \cdot v ^ {(1)} (\mathbf {x} _ {T _ {k}}) + \gamma_ {T _ {k}} ^ {(1)} ], \sum_ {T \subseteq S} (- 1) ^ {| S | - | T |} [ 0. 5 \cdot v ^ {(2)} (\mathbf {x} _ {T _ {k}}) + \gamma_ {T _ {k}} ^ {(2)} ]) | \\ + \sum_ {T _ {k} \subseteq N} | \max (- \sum_ {T \subseteq S} (- 1) ^ {| S | - | T |} [ 0. 5 \cdot v ^ {(1)} (\mathbf {x} _ {N \setminus T _ {k}}) - \gamma_ {T _ {k}} ^ {(1)} ], - \sum_ {T \subseteq S} (- 1) ^ {| S | - | T |} [ 0. 5 \cdot v ^ {(2)} (\mathbf {x} _ {N \setminus T _ {k}}) - \gamma_ {T _ {k}} ^ {(2)} ]) | \\ + \alpha \cdot \sum_ {T _ {k} \subseteq N} \sum_ {i = 1} ^ {2} | \sum_ {T \subseteq S} (- 1) ^ {| S | - | T |} [ 0. 5 \cdot v ^ {(i)} (\mathbf {x} _ {T _ {k}}) + \gamma_ {T _ {k}} ^ {(i)} ] | \\ + \alpha \cdot \sum_ {T _ {k} \subseteq N} \sum_ {i = 1} ^ {2} | - \sum_ {T \subseteq S} (- 1) ^ {| S | - | T |} [ 0. 5 \cdot v ^ {(i)} (\mathbf {x} _ {N \setminus T _ {k}}) - \gamma_ {T _ {k}} ^ {(i)} ]) |. \tag {10} \\ \end{array}
$$

Therefore, we only need to optimize $\{\gamma_{T_k}^{(1)},\gamma_{T_k}^{(2)}\}$ via gradient descent to reduce the loss in Equation (6).

# G THE RUN-TIME COMPLEXITY

This section explores the run-time complexity of extracting AND-OR interactions on different tasks. Theoretically, given an input sample with n input variables, the time complexity is $2^{n}$ , and we need to generate masked samples for model inference. Fortunately, the variable number is not too large by following the settings in (Li & Zhang, 2023b) and (Shen et al., 2023). In real applications, the average running time for a sentence in the SST-2 dataset on the BERT $_{LARGE}$ model is 45.14 seconds. Table 1 further shows the average running time of each input sample for different tasks.

Table 1: Average run-time per sample on different tasks. 

<table><tr><td>sentiment classification on the SST-2 datasetw.r.t. the pair of models (BERTBASE and BERTLARGE)</td><td>dialogue task on the SQuAD datasetw.r.t. the pair of models (LLaMA and OPT-1.3B)</td><td>image classification on the MNIST datasetw.r.t. the pair of models (ResNet-20 and VGG-16)</td></tr><tr><td>45.14 seconds</td><td>46.61 seconds</td><td>27.15 seconds</td></tr></table>

# H MORE BASELINE TO VERIFY THE EFFECTIVENESS OF EQUATION (6)

To further verify the effectiveness of the interactions extracted from Equation (6), we conducted experiments on more baseline, namely the original Harsanyi interaction (Ren et al., 2023a). We compared the sparsity and the generalization power of the AND interactions extracted from different DNNs. Figure 7 shows that the interactions extracted by our method exhibit higher sparsity and generalization power compared to the original AND interactions.

To enable fair comparisons, we double the number of Harsanyi interactions extracted from a sample by setting $I'(S|x) = I(S|x)$ . Thus, we congregate both sets of interactions (a total of $2 \cdot 2^{n}$ interactions) to draw a curve in Figure 2 and Figure 3. In this way, we can compare the same number of interactions over different methods.

![](images/8f9ce1efe54da1892ea2771adafb081b41c0a2dde2227f99aad42c18ae8fc64b.jpg)

<details>
<summary>line</summary>

| Iteration | ours       | baseline   |
| --------- | ---------- | ---------- |
| 0         | 10^2       | 10^2       |
| 1e4       | 1e-2       | 1e-2       |
| 2e4       | 1e-6       | 1e-6       |
</details>

![](images/2ccae43f989696b51771df5a7cc03948d667852458a0391e7ec668dd5c37f458.jpg)

<details>
<summary>line</summary>

| x     | ours       | baseline   |
|-------|------------|------------|
| 0     | 10^0       | 10^0       |
| 1e5   | 10^-4      | 10^-2      |
| 2e5   | 10^-6      | 10^-4      |
</details>

(a)

![](images/c1610a4096e98d0fc2098dc57a81110170747c3e45634d404791701c9f6de23e.jpg)

<details>
<summary>line</summary>

| # of salient interactions | ours  | baseline |
| ------------------------- | ----- | -------- |
| 50                        | 0.4   | 0.1      |
| 150                       | 0.5   | 0.2      |
| 250                       | 0.6   | 0.3      |
| 350                       | 0.7   | 0.4      |
| 450                       | 0.8   | 0.5      |
</details>

(b)   
Figure 7: Comparing the sparsity and generalization power of the extracted interactions between the original Harsanyi interactions and our proposed method.

# I ABLATION STUDY OF THE PARAMETER $\alpha$ IN EQUATION (6)

To explore the effect of $\alpha$ on the AND-OR interactions extracted from Equation (6), we conducted an ablation study on $\alpha$ . Specifically, we jointly extracted two sets of AND-OR interactions from the BERT $_{BASE}$ and BERT $_{LARGE}$ models, which were trained by (Devlin et al., 2019) and further finetuned by us for sentiment classification on the SST-2 dataset. For comparison, we set the value of $\alpha$ to [0, 0.2, 0.4, 0.6, 0.8, 1.0], respectively. Here, when $\alpha = 0$ , Equation (6) degenerated into Equation (5), indicating that only the largest interactions among the m DNNs were penalized. As $\alpha$ increases, the effects of not-so-salient interactions in other models were taken into account. Figure 8 shows that as the value of $\alpha$ increases, the sparsity of the extracted interactions did not increase too much, but the generalization power of the extracted interactions decreased. This shows the effectiveness of the penalty in Equation (5) in boosting the generalization power without significantly hurting the sparsity.

# J VISUALIZATION OF MORE SHARED AND DISTINCTIVE INTERACTION PRIMITIVES FOR FIGURE 5

In this section, we visualized all shared and distinctive interaction primitives across different DNNs in Figure 5. Specifically, we randomly selected 10 tokens in the given sentence, labeling these 10 tokens in red and the other unselected tokens in black. Here, n = 10 input variables denote the embeddings corresponding to these 10 tokens. Since each input variable has two states, masked and

![](images/1be157b5fcb1f06a86331098e438d2d94dddd7928db86149efe12f430aa0da8a.jpg)  
Figure 8: Effect of $\alpha$ on the AND-OR interactions extracted from Equation (6). (a) Strength of all interactions extracted from all samples, which were sorted in a descending order. The increase of different $\alpha$ value did not significantly affect the sparsity of interactions. (b) Decreasing generalization power of extracted AND-OR interactions, when the $\alpha$ value increased.

unmasked, a total of $2^{10}$ masked samples are generated. Then, according to Equations (1) and (2), a total of $2 \cdot 2^{10}$ AND interactions and OR interactions are finally obtained.

Then, we extracted the most salient k = 50 interactions from a total of $2 \cdot 2^{10}$ AND interactions and OR interactions as the set of AND-OR interaction primitives in each DNN, respectively. Therefore, in our proposed method, 25 shared interactions were extracted from both models, and 25 distinctive interactions were extracted from the BERT $_{BASE}$ and BERT $_{LARGE}$ models, respectively. In contrast, in the traditional method, 16 shared interactions were extracted from both models, and 34 distinctive interactions were extracted from the BERT $_{BASE}$ and BERT $_{LARGE}$ models, respectively.

In addition, Figure 9 shows the strength of the interaction value $|I(S|\mathbf{x})|$ for each salient interaction. Specifically, we show the strength of the interaction value for each distinctive interaction in each DNN. We also show the strengths of two interaction values for each shared interaction extracted from both DNNs, where the strength of the interaction value on the BERT $_{BASE}$ model is on the left and the strength of the interaction value on the BERT $_{LARGE}$ model is on the right. Figure 9 illustrates that, our method extracted more shared interactions compared to the traditional interaction-extraction method.

# K DISCUSSION ON THE DIFFERENCES IN THE SIMILARITY OF INTERACTIONS

This section compares the consistency of interactions between different types of DNNs. Since LLaMA (Touvron et al., 2023) and OPT-1.3B (Zhang et al., 2022b) have much more parameters than BERT $_{BASE}$ and BERT $_{LARGE}$ (Devlin et al., 2019), it is often widely believed that these two LLMs can better converge to the true language knowledge in the training data.

We can roughly understand this phenomenon by clarifying the following two phenomena. First, through extensive experiments, the authors have found that the learning of a DNN usually has two phases, i.e., the learning of new interactions and the forgotten of incorrectly learned interactions (this is our on-going research). In most cases, after the forgetting phase, the remained interactions of an LLM are usually shared by other LLMs. The high similarity of interactions between LLMs has also been observed in (Shen et al., 2023).

Second, compared to LLMs, relatively small models are less powerful to remove all incorrectly learned interactions. For example, a simple model is usually less powerful to regress a complex function. The less powerful models (BERT $_{BASE}$ and BERT $_{LARGE}$ ) are not powerful enough to accurately encode the potentially complex interactions.

Therefore, small models are more likely to represent various incorrect interactions to approximate the true target of the task. This may partially explains the high difficulty of extracting common interactions from two BERT models, as well as why the proposed method shows more improvements in the generalization power of interactions on BERT models.

Experimental verification. To this end, we have further conducted a new experiment to compare the interactions extracted from two LLMs, i.e., LLaMA (Touvron et al., 2023) and Aquila-7B (BAAI, 2023). As Figure 10 shows, two LLMs usually encode much more similar interactions than two

![](images/d40bc8154db7cfb19b46eb30dae1e6783f5870eef3772590029ab4f3bae5985a.jpg)

<details>
<summary>tree</summary>

| Interaction Type | Count |
| :--- | :--- |
| Our interactions | 25 |
| Distinctive interactions for the BERTBASE model | 34 |
| Shared interactions | 25 |
| Distinctive interactions for the BERTLARGE model | 25 |
| Traditional interactions | 16 |
| Distinctive interactions for the BERTBASE model | Large |
| Distinctive interactions for the BERTLARGE model | OR |
The Ring just left me cold and wet like I was out in the Seattle dr-iz-zle without rain-wear.
</details>

Figure 9: Visualization of more shared and distinctive interaction primitives for Figure 5.

not-so-large models. In fact, we have also observed similar interactions in another pair of LLMs, i.e., LLaMA and OPT-1.3B (Zhang et al., 2022b) (please see Figure 4). This partially explains the reason why the performance improvement of these two LLMs is similar to that of LLaMA and OPT-1.3B.

![](images/53fc473a59bdba0bd8e582fa21a6a4592a7f4fc0b3d999d428814d4ea0036720.jpg)

<details>
<summary>bar</summary>

| x | Our interactions |
| --- | --- |
| 0 | 0 |
| 1 | 5 |
| 2 | 3 |
| 3 | -1 |
| 4 | -2 |
| 5 | -1 |
| 6 | -8 |
| 7 | 2 |
| 8 | 1 |
| 9 | 0 |
| 10 | 0 |
</details>

![](images/145539d3e3266780446842baa3a7110ed0fc6a5b1aec798f123f4841c4367409.jpg)

<details>
<summary>bar</summary>

| x  | y     |
|----|-------|
| 0  | 100   |
| 1  | 10    |
| 2  | -10   |
| 3  | -5    |
| 4  | -2    |
| 5  | -1    |
| 6  | -0.5  |
| 7  | -0.2  |
| 8  | -0.1  |
| 9  | -0.05 |
| 10 | -0.02 |
</details>

![](images/63bd99fc5aeed122965df72042e1a507d041063ce96a0bf645bf763205662e5a.jpg)

<details>
<summary>bar</summary>

| x | y     |
|---|-------|
| 0 | 10.0  |
| 1 | 5.0   |
| 2 | 2.0   |
| 3 | 0.5   |
| 4 | -1.0  |
| 5 | -2.0  |
| 6 | -5.0  |
| 7 | 1.0   |
| 8 | 2.0   |
| 9 | 0.5   |
| 10| 0.0   |
</details>

![](images/b40434880e8ea48d460222f2c868b62929f21cd549079493ab3b50f469ca97de.jpg)

<details>
<summary>bar</summary>

| x  | y     |
|----|-------|
| 0  | 0.0   |
| 1  | 0.0   |
| 2  | 10.0  |
| 3  | 10.0  |
| 4  | 10.0  |
| 5  | 1.0   |
| 6  | -1.0  |
| 7  | -1.0  |
| 8  | -1.0  |
| 9  | -1.0  |
| 10 | -1.0  |
</details>

![](images/bc040569ad7d2649a68f0a4e047bb0e0a30c2648ad71ac8016e4503db45e6a7a.jpg)  
(a)   
(b)   
Figure 10: Differences of interactions between different types of DNNs.

# L A CONCRETE EXAMPLE TO ILLUSTRATE THE PROCEDURE

Let us use a concrete input sentence with six tokens to illustrate the experimental procedure from beginning to end.

Step 1: Given the sentence “A stitch in time saves nine” with six tokens, the six input variables are $x_{1} = \text{embedding of token “A”}, x_{2} = \text{embedding of token “stitch”}, x_{3} = \text{embedding of token “in”}, x_{4} = \text{embedding of token “time”}, x_{5} = \text{embedding of token “saves”}, \text{and } x_{6} = \text{embedding of token “nine”}.$ Although the dimensions of the embedding for the BERT $_{BASE}$ and BERT $_{LARGE}$ models are

different, we can still compute the interactions between the embeddings corresponding to the same token. Specifically, the embedding $x_{i}$ of a token in the BERT $_{BASE}$ model is $x_{i} \in R^{768}$ , and the embedding $x_{i}$ of a token in the BERT $_{LARGE}$ model is $x_{i} \in R^{1024}$ .

Step 2: The baseline value for each input variable is $b_{1} = b_{2} = b_{3} = b_{4} = b_{5} = b_{6} =$ special embedding of a masked token, where the special embedding of the masked token is encoded by the BERT $_{BASE}$ and BERT $_{LARGE}$ model, respectively. This special embedding can use the embedding of a special token, e.g., the embedding of the [CLS] token. Specifically, the baseline $b_{i}$ in the BERT $_{BASE}$ model is $b_{i} \in R^{768}$ , and the baseline $b_{i}$ in the BERT $_{LARGE}$ model is $b_{i} \in R^{1024}$ .

Step 3: In this case, there are a total of $2^{6}$ masked samples $x_{T_{0}}, x_{T_{1}}, x_{T_{2}}, x_{T_{3}}, \cdots, x_{T_{62}}, x_{T_{63}}$ used for the model inference. Specifically, the first masked sample is $x_{T_{0}} = \{b_{1}, b_{2}, b_{3}, b_{4}, b_{5}, b_{6}\}$ , where each of its input variables (embedding) is replaced with the corresponding baseline value (embedding of a special token), i.e., $x_{i} = b_{i}, i \in \{1, 2, \cdots, 6\}$ . The second masked sample is $x_{T_{1}} = \{b_{1}, b_{2}, b_{3}, b_{4}, b_{5}, x_{6}\}$ , where its input variable $x_{6}$ is kept as the original embedding, and the other five input variables are replaced with the corresponding baseline values, i.e., $x_{1} = b_{1}, x_{2} = b_{2}, x_{3} = b_{3}, x_{4} = b_{4}, x_{5}, b_{6}\}$ , where its input variable $x_{5}$ is kept as the original embedding, and the other five input variables are replaced with the corresponding baseline values, i.e., $x_{1} = b_{1}, x_{2} = b_{2}, x_{3} = b_{3}, x_{4} = b_{4}, x_{6} = b_{6}$ . The fourth masked sample is $x_{T_{3}} = \{b_{1}, b_{2}, b_{3}, b_{4}, x_{5}, x_{6}\}$ , where its input variables $x_{5}$ and $x_{6}$ are kept as the original embedding, respectively, and the other four input variables are replaced with the corresponding baseline values. Similarly, the 63rd masked sample is $x_{T_{62}} = \{x_{1}, x_{2}, x_{3}, x_{4}, x_{5}, b_{6}\}$ , where its input variables $x_{1}, x_{2}, x_{3}, x_{4}, x_{5}$ are kept as the original embedding, respectively, and $x_{6} = b_{6}$ . The 64th masked sample is $x_{T_{63}} = \{x_{1}, x_{2}, x_{3}, x_{4}, x_{5}, x_{6}\}$ , where all input variables are kept unchanged from the original embedding.

Step 4: For each masked sample $x_{T_{j}}, j \in \{0, 1, \cdots, 63\}$ , we computed the log-odds output of the ground-truth label $v(\mathbf{x}_{T_{j}}) = \log \frac{p(y = y^{\text{truth}} | \mathbf{x}_{T_{j}})}{1 - p(y = y^{\text{truth}} | \mathbf{x}_{T_{j}})} \in \mathbb{R}$ as the model output. In this way, feeding all masked samples $x_{T_{j}}$ into the BERT $_{BASE}$ model produced a total of $2^{6}$ model outputs $v^{\text{BASE}}(\mathbf{x}_{T_{0}})$ , $v^{\text{BASE}}(\mathbf{x}_{T_{1}}), \cdots, v^{\text{BASE}}(\mathbf{x}_{T_{63}})$ . Feeding all masked samples $x_{T_{j}}$ into the BERT $_{LARGE}$ model produced a total of $2^{6}$ model outputs $v^{\text{LARGE}}(\mathbf{x}_{T_{0}}), v^{\text{LARGE}}(\mathbf{x}_{T_{1}}), \cdots, v^{\text{LARGE}}(\mathbf{x}_{T_{63}})$ .

Step 5: Computed the AND outputs $v_{\mathrm{and}}^{\mathrm{BASE}}(\mathbf{x}_{T_{j}}) = 0.5v^{\mathrm{BASE}}(\mathbf{x}_{T_{j}}) + \gamma_{T_{j}}^{\mathrm{BASE}}, j \in \{0,1,\cdots,63\}$ and the OR outputs $v_{\mathrm{or}}^{\mathrm{BASE}}(\mathbf{x}_{T_{j}}) = 0.5v^{\mathrm{BASE}}(\mathbf{x}_{T_{j}}) - \gamma_{T_{j}}^{\mathrm{BASE}}, j \in \{0,1,\cdots,63\}$ for the BERT $_{BASE}$ model, respectively. Computed the AND outputs $v_{\mathrm{and}}^{\mathrm{LARGE}}(\mathbf{x}_{T_{j}}) = 0.5v^{\mathrm{LARGE}}(\mathbf{x}_{T_{j}}) + \gamma_{T_{j}}^{\mathrm{LARGE}}, j \in \{0,1,\cdots,63\}$ and the OR outputs $v_{\mathrm{or}}^{\mathrm{LARGE}}(\mathbf{x}_{T_{j}}) = 0.5v^{\mathrm{LARGE}}(\mathbf{x}_{T_{j}}) - \gamma_{T_{j}}^{\mathrm{LARGE}}, j \in \{0,1,\cdots,63\}$ for the BERT $_{LARGE}$ model, respectively.

Step 6: Computed the AND interactions $I_{\text{and}}^{\text{BASE}}(T_j|\mathbf{x}) = \sum_{T \subseteq T_j} (-1)^{|T_j| - |T|} v_{\text{and}}^{\text{BASE}}(\mathbf{x}_T), j \in \{0,1,\dots,63\}$ and the OR interactions $I_{\text{or}}^{\text{BASE}}(T_j|\mathbf{x}) = -\sum_{T \subseteq T_j} (-1)^{|T_j| - |T|} v_{\text{or}}^{\text{BASE}}(\mathbf{x}_{N \setminus T}), j \in \{0,1,\dots,63\}$ for the BERT $_{\text{BASE}}$ model, respectively. Computed the AND interactions $I_{\text{and}}^{\text{LARGE}}(T_j|\mathbf{x}) = \sum_{T \subseteq T_j} (-1)^{|T_j| - |T|} v_{\text{and}}^{\text{LARGE}}(\mathbf{x}_T), j \in \{0,1,\dots,63\}$ and the OR interactions $I_{\text{or}}^{\text{LARGE}}(T_j|\mathbf{x}) = -\sum_{T \subseteq T_j} (-1)^{|T_j| - |T|} v_{\text{or}}^{\text{LARGE}}(\mathbf{x}_{N \setminus T}), j \in \{0,1,\dots,63\}$ for the BERT $_{\text{LARGE}}$ model, respectively.

Step 7: Learned the parameters $\gamma_{T_{j}}^{BASE}$ and $\gamma_{T_{j}}^{LARGE}, j \in \{0,1,\cdots,63\}$ using the loss in Equation (6). Went back to Step 5 and repeated the iterations until the loss converges.

Step 8: Computed the final AND interactions $I_{\mathrm{and}}^{\mathrm{BASE}}(T_{j}|\mathbf{x})$ and the final OR interactions $I_{\mathrm{or}}^{\mathrm{BASE}}(T_{j}|\mathbf{x}), j \in \{0,1,\cdots,63\}$ for the BERT $_{BASE}$ model using the learnable parameters $\gamma_{T_{j}}^{BASE}, j \in \{0,1,\cdots,63\}$ . The final salient interactions of the BERT $_{BASE}$ model are obtained from all $2^{6}$ interactions, where the final salient interactions are those with interaction values greater than the threshold $\tau^{BASE}$ , $\Omega^{BASE} = \{T_{j} \subseteq N : |I_{\mathrm{and/or}}^{\mathrm{BASE}}(T_{j}|\mathbf{x})| > \tau^{BASE}\}$ .

Computed the final AND interactions $I_{\mathrm{and}}^{\mathrm{LARGE}}(T_{j}|\mathbf{x})$ and the final OR interactions $I_{\mathrm{or}}^{\mathrm{LARGE}}(T_{j}|\mathbf{x}), j \in \{0,1,\cdots,63\}$ for the BERT $_{LARGE}$ model using the learnable parameters $\gamma_{T_{j}}^{LARGE}, j \in \{0,1,\cdots,63\}$ . The final salient interactions of the BERT $_{LARGE}$ model are obtained from all $2^{6}$

Table 2: Diversity of two sets of interactions, which are extracted based on Equation (4) from the same models but with different initialized parameters $\{\gamma_T\}$ . 

<table><tr><td>DNNs</td><td> $S_{\text{and}}$ </td><td> $S_{\text{or}}$ </td></tr><tr><td> $BERT_{BASE}$ </td><td>10.90%</td><td>16.18%</td></tr><tr><td> $BERT_{LARGE}$ </td><td>17.84%</td><td>20.90%</td></tr></table>

interactions, where the final salient interactions are those with interaction values greater than the threshold $\tau^{LARGE}$ , $\Omega^{LARGE} = \{T_j \subseteq N : |I_{\text{and/or}}^{\text{LARGE}}(T_j | \mathbf{x})| > \tau^{\text{LARGE}}\}$ .

# M MATCHING PRECISION OF AND-OR INTERACTIONS

In this section, we conducted experiments to show the matching precision of the interactions used for matching the network output. Specifically, we extracted traditional AND-OR interaction primitives using various DNNs trained on different datasets following the settings in (Li & Zhang, 2023b). Then, we used the metric $m = \frac{\sum_{S \in \{\text{top } k \text{ interactions}\}} |I(S)|}{\sum_{S \in \{\text{top } k \text{ interactions}\}} |I(S)| + |v(N) - v(\emptyset) - \sum_{S \in \{\text{top } k \text{ interactions}\}} I(S)|}$ to measure the matching precision of the salient interactions. Figure 11 shows the matching precision of the interactions when we computed $m$ based on different numbers $k$ of most salient interactions. We found that only a few salient interactions were required to achieve a relatively high matching precision.

![](images/c75cdbf921ecbe27c556a4adb77589f797212bcbc650a5e0ed78f7270d4fdbb2.jpg)

<details>
<summary>line</summary>

| Model | # of interactions for matching | matching precision m |
| --- | --- | --- |
| MLP-5 on wifi | 0 | 0.0 |
| MLP-5 on wifi | 50 | 1.0 |
| MLP-5 on wifi | 100 | 1.0 |
| MLP-5 on wifi | 150 | 1.0 |
| MLP-5 on wifi | 200 | 1.0 |
| MLP-5 on wifi | 250 | 1.0 |
| ResNet-20 on MNIST-3 | 0 | 0.0 |
| ResNet-20 on MNIST-3 | 50 | 1.0 |
| ResNet-20 on MNIST-3 | 100 | 1.0 |
| ResNet-20 on MNIST-3 | 150 | 1.0 |
| ResNet-20 on MNIST-3 | 200 | 1.0 |
| ResNet-20 on MNIST-3 | 250 | 1.0 |
| VGG-16 on MNIST-3 | 0 | 0.0 |
| VGG-16 on MNIST-3 | 50 | 1.0 |
| VGG-16 on MNIST-3 | 100 | 1.0 |
| VGG-16 on MNIST-3 | 150 | 1.0 |
| VGG-16 on MNIST-3 | 200 | 1.0 |
| VGG-16 on MNIST-3 | 250 | 1.0 |
| LeNet on MNIST-3 | 0 | 0.0 |
| LeNet on MNIST-3 | 50 | 1.0 |
| LeNet on MNIST-3 | 100 | 1.0 |
| LeNet on MNIST-3 | 150 | 1.0 |
| LeNet on MNIST-3 | 200 | 1.0 |
| LeNet on MNIST-3 | 250 | 1.0 |
| AlexNet on CeLebA-eyeglasses | 0 | 0.0 |
| AlexNet on CeLebA-eyeglasses | 50 | 1.0 |
| AlexNet on CeLebA-eyeglasses | 100 | 1.0 |
| AlexNet on CeLebA-eyeglasses | 150 | 1.0 |
| AlexNet on CeLebA-eyeglasses | 200 | 1.0 |
| AlexNet on CeLebA-eyeglasses | 250 | 1.0 |
</details>

Figure 11: Matching precision of the interactions used for matching the network output.

# N MORE EXPERIMENTS

# N.1 OPTIMIZING THE LOSS IN EQUATION (4) MAY LEAD TO DIVERSE SOLUTIONS

In Section 2.3.1, we mentioned that optimizing the loss in Equation (4) may lead to diverse solutions. Given different initial states, optimizing the loss in Equation (4) usually extracted two different sets of AND-OR interactions. In particular, as shown in Table 2, when we set the initial state of two sets of parameters $\{\gamma_T\}$ as $\gamma_T \sim \mathcal{N}(0,1)$ , the loss in Equation (4) would learned dramatically different interactions with only $21\%$ overlap. Figure 12 further shows the top 5 AND-OR interaction primitives extracted from BERT $_{\text{LARGE}}$ model on an input sentence. Both experiments illustrate that given different initial states, the loss in Equation (4) may learn different AND-OR interactions.

![](images/52719a306029097314c3424cd6ef51bb3f6ce9cb29edef6f6d3314a30d30341f.jpg)

<details>
<summary>heatmap</summary>

| Init | Swimmer | Plays | Players -avi synchronized | Dr-y -avi wet | Dr-y |
|---|---|---|---|---|---|
| Init 1st | and | 3.68 | 3.26 | 3.04 | -2.89 |
| Init 2nd | and | 4.26 | or | 3.37 | 3.28 |
The whole thing plays out with the dr-ows-y he-avi-ness of synchronized swimmer wearing a wool wet-suit.
</details>

Figure 12: Top 5 AND-OR interaction primitives extracted from BERT $_{LARGE}$ model, when optimizing the loss in Equation (4) given different initial states.

# N.2 EXAMINING WHETHER THE INTERACTIONS CAN EXPLAIN THE NETWORK OUTPUT

In Section 2.3.2, we mentioned to conduct experiments to examine whether the extracted AND-OR interactions could still accurately explain the network output, when we removed the error term. Figure 13 shows matching errors of all masked samples for all subsets $T \subseteq N$ , when we sorted the network outputs for all $2^{n}$ masked samples in a descending order. It shows that the real network output was well approximated by interactions.

![](images/5612ce70de6afc96396e332ffdea562827a2b0aa9473d59faed96cd891f840a0.jpg)

<details>
<summary>line</summary>

| index of interactions | BERT_BASE | BERT_LARGE |
| --------------------- | --------- | ---------- |
| 0                     | -2.0      | -5.0       |
| 200                   | 0.5       | 2.5        |
| 400                   | 1.5       | 5.0        |
| 600                   | 2.5       | 6.0        |
| 800                   | 3.5       | 7.0        |
| 1000                  | 4.5       | 7.5        |
</details>

Figure 13: Universal matching of interaction primitives to the DNN's output. Shade area indicates the matching error of different $v(\mathbf{x}_T)$ .

# O EXPERIMENTAL DETAILS

# O.1 PERFORMANCE OF DNNs

In Section 3, we conducted experiments using several DNNs trained on different types of datasets, including the language and image datasets. In the sentiment classification task, we fintuned the pretrained models, BERT $_{BASE}$ and BERT $_{LARGE}$ , using the SST-2 dataset. For image classification task, we trained ResNet-20 and VGG-16 with the MNIST-3 dataset. Table 3 reports the classification accuracy of the aforementioned DNNs. For the dialogue task, we used the pretrained models, the LLaMA model and OPT-1.3B model, directly.

Table 3: Classification accuracy of different DNNs in the sentiment classification and image classification tasks. 

<table><tr><td>Tasks</td><td>dataset</td><td colspan="2">DNNs</td></tr><tr><td>sentiment classification</td><td>SST-2</td><td>BERTBASE91.32%</td><td>BERTLARGE93.26%</td></tr><tr><td>image classification</td><td>MNIST-3</td><td>ResNet-20100%</td><td>VGG-16100%</td></tr></table>

# O.2 THE SELECTION OF INPUT VARIABLES FOR INTERACTION EXTRACTION

This section discusses the selection of input variables for extracting interactions. As mentioned in Section 2.2, given an input sample x with n input variables, we can extracted at most $2^{n}$ interactions. Therefore, the computational cost for extracting interactions increases exponentially with the number of input variables. For example, if we take a word in a sentence (or a pixel in an image) as an input variable, the computation is usually inapplicable. To alleviate this issue, we followed (Shen et al., 2023) to select a set of words as input variables and leave other words as the constant background to compute interactions between them. Specifically, we selected 8-10 input variables for each sample in the three tasks. We only extracted the interactions between the selected variables, leaving the rest of the unselected input variables unchanged as background.

\- For sentences in the SST-2 dataset, we first tokenized the input sentences and selected tokens as input variables for 200 samples. For each sentence in this dataset, some words have no clear semantics for sentiment classification, i.e., stop words containing dummy words and pronouns, and consequently, there is little interaction within their corresponding tokens. Therefore, we bypassed

these tokens without clear semantics, and only selected tokens among the remaining semantic tokens. To facilitate analysis, we randomly selected n = 10 tokens as input variables for sentences with more than 10 semantic tokens. The masking of input variables was performed at the embedding level.

- For sentences in the SQuAD dataset, we took first several words in each document as the input sample to the DNN. Specifically, we selected the first 30 words in each document as the input sample and the 31st word as the target word, provided that the following conditions were met: 1) The 31st word possessed clear semantic meaning, which means that it did not belong to the category of stop words. 2) The five words immediately preceding the target word did not constitute sentence-ending punctuation marks, such as a full stop. If either of these conditions was not satisfied, we extended the initial 30 words until all requirements were fulfilled, and selected the next word as the target word. When extracting interactions, we randomly selected a set of $n = 10$ words which have semantic meanings as input variables. It's worth noting that a single word can correspond to multiple tokens, so when we masked a specific word, we masked all of its corresponding tokens. The masking of input variables was performed at the embedding level.   
- For images in the MNIST-3 dataset, we manually labeled the semantic part for 100 positive samples (digit 3) by following (Li & Zhang, 2023b). For each image in this dataset, most of the pixels are black background pixels with no semantic information, and consequently, there is no interaction within these black pixels. Thus, we considered interactions only within foreground pixels. Specifically, we divided a whole image into small patches of size $3 \times 3$ and selected $n = 8$ patches in the foreground of an image as input variables. Following (Li & Zhang, 2023b), we used the zero patches as the baseline values to mask the unselected patches $i \in N \setminus T$ in the sample.