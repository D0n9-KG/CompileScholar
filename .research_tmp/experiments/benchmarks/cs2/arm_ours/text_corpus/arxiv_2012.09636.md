XAI-P-T: A Brief Review of Explainable Artificial Intelligence from Practice to Theory 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2012.09636v1 [cs.AI] 17 Dec 2020 
 
 

# XAI-P-T: A Brief Review of Explainable Artificial Intelligence from Practice to Theory

 
 
 Nazanin Fouladgar
 
 Affiliation:  Department of Computing Science, Umeå University, Sweden,
 
 E-mail  nazanin@cs.umu.se 
 
    
 Kary Främling
 
 Affiliation:  Department of Computing Science, Umeå University, Sweden,
 
 E-mail  nazanin@cs.umu.se 
 
 Affiliation:  Aalto University, School of Science and Technology, Finland,
 
 E-mail  kary.framling@umu.se 
 

 Abstract 
 
 In this work, we report the practical and theoretical aspects of Explainable AI (XAI) identified in some fundamental literature. Although there is a vast body of work on representing the XAI backgrounds, most of the corpuses pinpoint a discrete direction of thoughts. Providing insights into literature in practice and theory concurrently is still a gap in this field. This is important as such connection facilitates a learning process for the early stage XAI researchers and give a bright stand for the experienced XAI scholars. Respectively, we first focus on the categories of black-box explanation and give a practical example. Later, we discuss how theoretically explanation has been grounded in the body of multidisciplinary fields. Finally, some directions of future works are presented.

 
 
 
 Keywords:  Explainable AICategorization Black-Box Multidisciplinary Explanation Practice Theory
 
 

## 1 Introduction

 
 Machine learning has been immensely applied in different applications and is progressing over time by the advent of new models  [ 4 ] . Nevertheless, the complex behavior of these models impedes human to understand concisely how specific decisions were made. This limitation requires machines to provide transparency by means of explanation. Therefore, one could assign “black-box” to the machine learning models, aiming at making decisions, and in contrast “white-box” to the explained version of these models under the topic of Explanation AI (XAI) .

 
 
 The corpus of research in  [ 12 ] recall explanation as a reasoning and simulating process in response to “what happened”, “how happened” and “why happened” questions. The questions indicate that explanation could be interpreted in forms of both causal and non-causal relationships. Although causality has had more popularity among AI researchers to solve the algorithmic decision problems, the non-causal relationship has recently appealed the scholars of human-computer interaction field. In fact, XAI is not yet matured enough and there are lots of open rooms for providing explanation in practice and theory. While addressing these two aspects are crucial, there is still lack of a brief and concurrent understanding of how currently the XAI field stands practically and theoretically.

 
 
 In this work, we give an understanding of XAI in literatures, consisting of three directions. First, we provide a neat categorization of black-box explanation incorporating connections between studies in Section 2 . Later, we introduce a practical example in Section 3 . Finally, we take a look at explanation in theory from the social science perspective retrieved from Miller  [ 12 ] work in Section 4 . We also guide readers through some open research directions in Section 5 .

 
 
 

## 2 Black-Box Explanation Categories

 
 Explaining machine learning models has recently become prevalent in the XAI domain. Most of the approaches fit into the general categorization proposed by Mengnan et al.  [ 2 ] . The authors discuss two major interpretability categories: Intrinsic and Post-hoc . While in the former, the focus is on constructing self-explanatory models (e.g. decision tree, rule-based models, etc.), in the latter, the effort relies on creating an alternative model to provide an explanation of existing models. These two classes are rather broad, yet in a more detailed granularity, each of them are divided into global and local explanation view. The main concern in the global explanation stands for understanding the structure and parameters of the model. However, in the local view, the causal relationship between specific input and the predicted outcome is mainly unveiled. The schematic diagram of this categorization has been illustrated in Figure  1 in a blue dot block.

 
 
 Different approaches have been proposed by researchers, incorporating explanation in each of the four discussed classes. Providing global interpretations in the intrinsic category, the models are usually enforced to comprise fewer features in terms of explanation constraint   [ 7 ] or to approximate the base models by readily interpretable models  [ 16 ] . In the local interpretation of intrinsic category, attention mechanisms are widely applied on some models such as Recurrent Neural Networks (RNNs)  [ 10 ] . The latter mechanisms provide the contribution of attended part of input in a particular decision. In general, there is a risk of scarifying accuracy in the cost of higher explanation in the intrinsic category.

 
 
 Moving toward post-hoc explanation to keep accuracy and fidelity, in the global view, feature importance has been discussed massively  [ 1 ] . This approach implies for statistical contributions of each feature in black-box models. In this sense, permutating features iteratively and later investigating how the accuracy deviates, have proved high efficiency in the interpretation of classical machine learning models. In case of deep learning models, the main concern is to find preferred inputs for neurons of specific layer to maximize their activation  [ 15 ] . This technique is called activation maximization , whose performance highly depends on the optimization procedure. Provided the local post-hoc explanation, a neighborhood of input has been approximated by utilizing interpretable models (white-box) and establishing explanation on the new model prediction  [ 13 ] .
Putting attention on specific models of the latter category, back-propagating the gradients (variants) of outputs with respect to their inputs has been discussed in the deep leaning models  [ 11 ] . This process refers to a top-down strategy, exploring more relevant features in prediction.

 
 
 Apart from generic categorization of machine learning explanations in  [ 2 ] and the aforementioned solutions in the literatures, the formal formulations of the categories have been articulated in  [ 9 ] . In this study, the global and local post-hoc interpretations are recalled by model explanation and outcome explanation terminologies respectively. Figure  1 shows these terminologies in a red dot block with respect to their equivalent categories in  [ 2 ] . The red bidirectional arrows illustrate these equivalencies. According to the work in  [ 9 ] , the two explanations are formulated as following respectively:

 
 
 Figure 1: Categorization of explainable machine learning models 
 
 
 
 • 
 
 Given a black-box predictor b b and a set of instances X X , the model explanation problem consists of finding an explanation E ∈ ℰ E\in\mathcal{E} , belonging to a human interpretable domain ℰ \mathcal{E} , through an interpretable global predictor c g = f ⁡ ( b , X ) c_{g}=f(b,X) derived from the black box b b and the instances X X using some process f ( . , . ) f(.,.) . An explanation E ∈ ℰ E\in\mathcal{E} is obtained through c g c_{g} , if E = ε g ​ ( c g , X ) E=\varepsilon_{g}(c_{g},X) for some explanation logic ε g ( . , . ) \varepsilon_{g}(.,.) , which reasons over c g c_{g} and X X .

 

 • 
 
 Given a black box predictor b b and an instance x x , the outcome explanation problem consists of finding an explanation e ∈ ℰ e\in\mathcal{E} , belonging to a human interpretable domain ℰ \mathcal{E} , through an interpretable local predictor c l = f ⁡ ( b , x ) c_{l}=f(b,x) derived from the black box b b and the instance x x using some process f ( . , . ) f(.,.) . An explanation e ∈ ℰ e\in\mathcal{E} is obtained through c l c_{l} , if e = ε l ​ ( c l , x ) e=\varepsilon_{l}(c_{l},x) for some explanation logic ε l ( . , . ) \varepsilon_{l}(.,.) , which reasons over c l c_{l} and x x .

 

 
 
 
 Likewise the difference between the global and local post-hoc explanations, the difference between the above formulations relies on the explanations of whole black-box logic and specific input contribution in the black-box decisions.

 
 
 Another class of black-box explanation introduced in  [ 9 ] is called model inspection , designed to leverage domain-required analysis. This terminology locates somehow in the middle of the two previous categories, the model explanation and the outcome explanation . In fact, model inspection provides a visual or textual representations, to give an explanation of either black box specific property or the decision made (e.g. one could vary the inputs and observe the prediction changes visually in sensitivity analysis). The schematic diagram of this block has been illustrated with a red unidirectional arrow in Figure  1 and its formal definition is represented as following  [ 9 ] :

 
 
 
 • 
 
 Given a black box b b and a set of instances X X , the model inspection problem consists of providing a (visual or textual) representation r = f ⁡ ( b , X ) r=f(b,X) of some property of b b , using some process f ( . , . ) f(.,.) .

 

 
 
 
 All the three mentioned terminologies in  [ 9 ] stand for defining either an external model or a visual/textual representation. Recalling from  [ 2 ] , the intrinsic explanations are replaced by transparent box design terminology in  [ 9 ] . The left bidirectional red arrows in Figure  1 indicate this replacement and the transparent box design is formulated as following  [ 9 ] :

 
 
 
 • 
 
 Given a training dataset D = ( X , Y ^ ) D=(X,\hat{Y}) , the transparent box design problem consists of learning a locally or globally interpretable predictor c c from D D . For a locally interpretable predictor c c , there exists a local explanator logic ε l \varepsilon_{l} to derive an explanation ε l ​ ( c , x ) \varepsilon_{l}(c,x) of the decision c ⁡ ( x ) c(x) for an instance x x . For a globally interpretable predictor c c , there exists a global explanator logic ε g \varepsilon_{g} to derive an explanation ε g ​ ( c , X ) \varepsilon_{g}(c,X) .

 

 
 
 
 As it is clear in the formulation above, the explanation ε \varepsilon is given based on its own predictor c ⁡ ( x ) c(x) rather than external global models ( c g = f ⁡ ( b , X ) c_{g}=f(b,X) ) or external local models ( c l = f ⁡ ( b , x ) c_{l}=f(b,x) ).

 
 
 In addition to the formulations above, the categorized explanation tools such as salient mask (SM) and partial dependence plot (PDP) are scrutinized in  [ 9 ] . This study also investigates the explained machine learning models and their examined data types. It has been turned out that neural networks, tree ensembles and support vector machine are widely explained with tabular data, targeting the model explanation category. Yet, the deep neural networks explanation are mostly provided with image data for the purpose of outcome explanation and model inspection . Solving the transparent box design problems, decision rules with tabular data are underlined in the majority of literatures.

 
 
 

## 3 Black-Box Outcome Explanation in Practice

 
 As an example of outcome explanation in practice, we pinpoint our very recent work in  [ 3 ] . This paper exploits two neural networks in the classification task of multimodal affect computation over two datasets. Tabular time series data of these datasets are extracted from different wearable sensors such as electrodermal activity (EDA) and learnt by the networks. Accordingly, the networks detect the human state of mind for an individual in each dataset. To explain why a specific state is detected, two concepts of Contextual Importance (CI) and Contextual Utility (CU) , introduced by Främling  [ 5 ] , are employed.

 
 
 C ​ I CI explores how important each sensor (feature) is for a given outcome, i.e. how much the outcome can change depending on the feature value. C ​ U CU indicates to what extent the feature value of the studied instance contributes to a high output value, i.e. in a classification task how typical the current value is for the class in question. These concepts have been primarily initiated in the context of Multiple Criteria Decision Making (MCDM), where decisions are established on a consensus between different stakeholders preferences. According to  [ 6 ] , C ​ I CI and C ​ U CU are theoretically correct from the Decision Theory point of view. The mathematical formulations of these concepts are as following  [ 5 ] :

 
 
 

 
 | 
 C ​ I = C ​ m ​ a ​ x x ​ ( C i ) − C ​ m ​ i ​ n x ​ ( C i ) a ​ b ​ s ​ m ​ a ​ x − a ​ b ​ s ​ m ​ i ​ n CI=\frac{Cmax_{x}(C_{i})-Cmin_{x}(C_{i})}{absmax-absmin} | 
 | 
 (1) | 
 

 
 
 

 
 | 
 C ​ I = y i ​ j − C ​ m ​ i ​ n x ​ ( C i ) C ​ m ​ a ​ x x ​ ( C i ) − C ​ m ​ i ​ n x ​ ( C i ) CI=\frac{y_{ij}-Cmin_{x}(C_{i})}{Cmax_{x}(C_{i})-Cmin_{x}(C_{i})} | 
 | 
 (2) | 
 

 
 
 Where C i C_{i} is the i i th context (specific input of black-box), y i ​ j y_{ij} is the value of j j th output (class probability) with respect to the context C i C_{i} , C ​ m ​ a ​ x x ​ ( C i ) Cmax_{x}(C_{i}) and C ​ m ​ i ​ n x ​ ( C i ) Cmin_{x}(C_{i}) are the maximum and minimum values indicating the range of output values observed by varying each attribute x x of context C i C_{i} , a ​ b ​ s ​ m ​ a ​ x absmax and a ​ b ​ s ​ m ​ i ​ n absmin are also the maximum and minimum values indicating the range of j j th output (the class probability value).

 
 
 Representing the outcome explanation , Figure  2 shows the EDA variation for the specific instance ( C i C_{i} ) of WESAD  [ 14 ] , one of the examined datasets in  [ 3 ] . With respect to this instance, the affective state of an individual has been detected as “meditation”. As it could be inferred, the a ​ b ​ s ​ m ​ i ​ n absmin and a ​ b ​ s ​ m ​ a ​ x absmax have been ranged in [ 0 − 100 ] [0-100] and the C ​ m ​ i ​ n Cmin and C ​ m ​ a ​ x Cmax values are located somewhere within this range, indicating the importance portion of EDA sensor in the process of detecting “meditation” state. Considering the EDA sensor’s utility, y i ​ j y_{ij} presents a high value within the range of C ​ m ​ i ​ n Cmin and C ​ m ​ a ​ x Cmax for the context of C i C_{i} .

 
 
 Figure 2: C ​ m ​ i ​ n Cmin and C ​ m ​ a ​ x Cmax values for input variation in EDA  [ 3 ] 
 
 
 

## 4 Multidisciplinary Perspectives in Explanation

 
 Looking into explanation from a multidisciplinary and theoretical perspective, we highlight Miller’s work in  [ 12 ] . He scrutinizes explanation as two processes of cognitive and social. While the cognitive process determines a subset of identified causes, the social process mainly aims at transferring knowledge between the explainee and the explainer. Considering explanation as a process and product, the explanation resulting from the cognitive process is appointed to a product.

 
 
 In this paper  [ 12 ] , one of the key findings from the cognitive science perspective in XAI has been noted as contrastive explanation . Such explanation is in congruent with how people explain the causes of events in their daily life. In fact, the main goal is to answer: “Why P P rather than Q Q ?”. P P refers to a fact did occur and Q Q is a foil did not occur, yet P P and Q Q follow a similar history. Emphasizing on the why-question, the “within object” differences are mainly under the focus of contrastive explanation . However, the question could be designed to cover the “between objects” or “within objects over time” differences. In the latter cases, the questions are posed by “Why does object a a have property P P , while object b b has property Q Q ?” or “Why does object a a have property P P at time t ​ 1 t1 , but property Q Q at time t ​ 2 t2 ?”.

 
 
 In general, contrastive explanation is more applicable than providing the whole chain of causes to the explainee  [ 12 ] . Specifying three steps of generating causes, selecting and evaluating, we illustrate the contrastive explanation process in Figure  3 . In the first step, it has been argued that counterfactuals are generated by applying some heuristics. Abnormality, intentionality, time and controllability of events/actions are among these heuristics. In the second step, some criteria are explored to clarify which causes should be selected. Necessity, sufficiency of causes and robustness to changes are a few criteria people typically devote in selecting explanations. Finally, evaluations of provided explanations should be devised concisely. It has been highlighted that people accept more likely the explanations consistent with their prior knowledge. Furthermore, people label explanations good according to the truth of causes. However, the latter measure could not necessarily provide the best explanation. Other measures such as simplicity, relevancy and the goal of explanation could influence the evaluations as well.

 
 
 Figure 3: Contrastive explanation processes: cognitive science perspective 
 
 
 Shifting to the social process, the causal explanations takes the form of conversation   [ 12 ] . In this process, the communication problem between two roles of “explainer” and “explainee” matters and accordingly some rules undertake the protocols of such interaction. Specifically, basic rules follow the so-called Grice’s maxims principles  [ 8 ] : quality, quantity, relation and manner. Clearly speaking, the rules are performed on saying what is believed (quality), as much as necessary (quantity), what is relevant (relation) and in a nice way (manner) to construct a conversational model. In an extension of this model, explanations are presumed arguments in the logical contexts. It has been thought that as well as explaining causes, there should be the potentiality of defending claims. The idea comes from an argument’s main components: causes (support) and claims (conclusion). Following the arguments as explanation, a formal explanation dialogue with argumentation framework and the rules of shifting between explainee and explainer have been pointed out in  [ 12 ] . One of the advantages of the conversational model of explanation and its extension is their generality, implying for applying in any algorithmic decisions. Being language-independent, these models could be compatible with visual representations of explanation to both explainee questions and explainer answers. To provide how the explanation has been discussed in social perspective, we show the related schematic diagram in Figure  4 .

 
 
 Figure 4: Social science perspective of explanation 
 
 
 

## 5 Future Works

 
 Generally, the literatures discuss concrete categories and formalizations for opening black-box problems, different tools, explained black-boxes and the applied data types. However, there is no explicit categorization of the explained models targeting different types of users such as experts vs. non-experts. Therefore, we believe that a comprehensive review of such models adds values to the XAI research communities.

 
 
 In the exemplified work  [ 3 ] , the C ​ I CI concept mainly addresses the varying output range of specific variable as a measure of feature importance. There are several cases in multimodal affect computing applications that a particular variable is influenced by other variables of domain. In other words, the current version of C ​ I CI concept does not consider the casual relationships between endogenous variables. By addressing this gap, one could provide a more realistic and robust black-box explanation to the end-users. Another gap refers to the fact that the C ​ I CI and C ​ U CU concepts are not timely dynamic and theoretically well-matured. Therefore, other future directions could be devoted to a timely-aware C ​ I CI and C ​ U CU as well as their intersection with the cognitive and social science theories, respectively.
As Miller  [ 12 ] argues the socially interactive explanation, it could also be worthwhile to conduct a user study and investigate the impact of these concepts in a social interaction.

 
 
 

## References

 
 
 [1] 
 
Altmann, A., Toloşi, L., Sander, O., Lengauer, T.: Permutation importance: a
corrected feature importance measure. Bioinformatics 26 (10),
1340–1347 (2010)

 

 
 [2] 
 
Du, M., Liu, N., Hu, X.: Techniques for interpretable machine learning.
Communications of the ACM 63 (1), 68–77 (2019)

 

 
 [3] 
 
Fouladgar, N., Alirezaie, M., Främling, K.: Decision explanation: Applying
contextual importance and contextual utility in affect detection. In: Italian
Workshop on Explainable Artificial Intelligence, XAI.it 2020. AI*IA SERIES,
vol. 2742, p. 1–13 (2020)

 

 
 [4] 
 
Fouladgar, N., Främling, K.: A novel lstm for multivariate time series with
massive missingness. Sensors 20 ,  2832 (2020)

 

 
 [5] 
 
Främling, K.: Explaining results of neural networks by contextual
importance and utility. In: the AISB’96 conference. Citeseer (1996)

 

 
 [6] 
 
Främling, K.: Decision theory meets explainable ai. In: Explainable,
Transparent Autonomous Agents and Multi-Agent Systems. pp. 57–74. Springer
(2020)

 

 
 [7] 
 
Freitas, A.A.: Comprehensible classification models: A position paper. ACM
SIGKDD Explorations Newsletter 15 (1), 1–10 (2014)

 

 
 [8] 
 
Grice, H.: Logic and conversation. Syntax and Semantics pp. 41–58 (1975)

 

 
 [9] 
 
Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., Pedreschi,
D.: A survey of methods for explaining black box models. ACM Computing
Surveys 51 (5) (2018). https://doi.org/10.1145/3236009

 

 
 [10] 
 
Kelvin, X., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel,
R., Bengio, Y.: Show, attend and tell: Neural image caption generation with
visual attention. In: Proceedings of the 32nd International Conference on
Machine Learning. Proceedings of Machine Learning Research, vol. 37, pp.
2048–2057 (2015)

 

 
 [11] 
 
Lapuschkin, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.R., Samek,
W.: On pixel-wise explanations for non-linear classifier decisions by
layer-wise relevance propagation. PLoS ONE 10 , e0130140 (2015)

 

 
 [12] 
 
Miller, T.: Explanation in artificial intelligence: Insights from the social
sciences. Artificial Intelligence 267 , 1–38 (2019)

 

 
 [13] 
 
Ribeiro, M., Singh, S., Guestrin, C.: “why should i trust you?”: Explaining
the predictions of any classifier. In: In Proceedings of the ACM SIGKDD
International Conference Knowledge Discovery and Data Mining. pp. 97–101
(2016)

 

 
 [14] 
 
Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., Van Laerhoven, K.:
Introducing wesad, a multimodal dataset for wearable stress and affect
detection. In: International Conference on Multimodal Interaction (ICMI). p.
400–408. ACM (2018)

 

 
 [15] 
 
Simonyan, K., Vedaldi, A., Zisserman, A.: Deep inside convolutional networks:
Visualising image classification models and saliency maps. In: In Proceedings
of the ICLR Workshop (2014)

 

 
 [16] 
 
Vandewiele, G., Janssens, O., Ongenae, F., Turck, F., Hoecke, S.V.: Genesim:
genetic extraction of a single, interpretable model. ArXiv
 abs/1611.05722 (2016)