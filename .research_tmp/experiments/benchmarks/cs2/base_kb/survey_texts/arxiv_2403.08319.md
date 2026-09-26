Knowledge Conflicts for LLMs: A Survey 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY-NC-ND 4.0
 
 
arXiv:2403.08319v2 [cs.CL] 22 Jun 2024 
 
 

# Knowledge Conflicts for LLMs: A Survey

 
 
 Rongwu Xu
 
 Email:  xrw22@mails.tsinghua.edu.cn 
 
    
 Zehan Qi
 
 Email:  qzh23@mails.tsinghua.edu.cn 
 
    
 Zhijiang Guo
 
    
 Cunxiang Wang
 
 Affiliation:  Westlake University
 
    
 Hongru Wang
 
 Affiliation:  The Chinese University of Hong Kong
 
    
 Yue Zhang
 
 Affiliation:  Westlake University
 
    
 Wei Xu
 
 Affiliation:  Tsinghua University
 
 
 Affiliation:  University of Cambridge
 
 Affiliation:  Leading authors, * Equal contribution
 

 Abstract 
 
 This survey provides an in-depth analysis of knowledge conflicts for large language models (LLMs), highlighting the complex challenges they encounter when blending contextual and parametric knowledge.
Our focus is on three categories of knowledge conflicts: context-memory, inter-context, and intra-memory conflict. These conflicts can significantly impact the trustworthiness and performance of LLMs, especially in real-world applications where noise and misinformation are common.
By categorizing these conflicts, exploring the causes, examining the behaviors of LLMs under such conflicts, and reviewing available solutions, this survey aims to shed light on strategies for improving the robustness of LLMs, thereby serving as a valuable resource for advancing research in this evolving area.

 
 
 
 
 

 
 
 
 

 
 
 https://github.com/pillowsofwind/Knowledge-Conflicts-Survey 

 
 

## 1 Introduction

 
 Large language models (LLMs;  Brown et al. 2020 ; Touvron et al. 2023 ; OpenAI 2024 ) are renowned for encapsulating a vast repository of world knowledge  Roberts et al. (2020) ; Hu et al. (2023) , referred to as parametric knowledge . These models excel in knowledge-intensive tasks including QA Petroni et al. (2019) , fact-checking Gao et al. (2023a) , dialogue system Wang et al. (2023e) , knowledge generation  Chen et al. (2023c) , inter alia .
In the meantime, LLMs continue to engage with external contextual knowledge after deployed  Pan et al. (2022) , including user prompts  Liu et al. (2023a) , interactive dialogues  Zhang et al. (2020) ; Wang et al. (2024a) , or retrieved documents from the Web  Lewis et al. (2020) ; Shi et al. (2023c) , and tools  Schick et al. (2023) ; Zhuang et al. (2023) .

 
 
 Figure 1: An LLM may encounter three distinct types of knowledge conflicts, stemming from knowledge sources—either contextual ( I. Context , yellow chatboxes) or inherent to the LLM’s parameters ( II. Memory , blue chatboxes). When confronted with a user’s question ( purple chatbox) entailing knowledge of complex conflicts, the LLM is required to resolve these discrepancies to deliver accurate responses. 
 
 
 Figure 2: We view knowledge conflict not only as a standalone phenomenon but also as a nexus that connects various causal triggers ( causes ) with the behaviors of LLMs. While existing literature mainly focuses on II. Analysis , our survey involves systematically observing these conflicts, offering insights into their emergence and impact on LLMs’ behavior, along with the desirable behaviors and related solutions. 
 
 
 Integrating contextual knowledge into LLMs enables them to keep abreast of current events Kasai et al. (2022) and generate more accurate responses Shuster et al. (2021) , yet it risks conflicting due to the rich knowledge sources.
The discrepancies among the contexts and the model’s parametric knowledge are referred to as knowledge conflicts   Chen et al. (2022) ; Xie et al. (2023) .
In this paper, we categorize three distinct types of knowledge conflicts, as characterized in  Figure 1 .
As the example shown in  Figure 1 , when utilizing an LLM to respond to a user question, users may provide the LLM with supplementary prompts, while the LLM also leverages search engines to gather relevant documents from the Web to enhance its knowledge  Lewis et al. (2020) . This combination of user prompts, dialogue history, and retrieved documents constitutes contextual knowledge ( context ). Contextual knowledge can conflict with the parametric knowledge ( memory ) encapsulated within the LLM’s parameters ( Longpre et al., 2021 ; Xie et al., 2023 ) , a phenomenon we term as context-memory conflict ( CM , § 2 ).
In real-world scenarios, the external document might be fraught with noise Zhang and Choi (2021) or even deliberately crafted misinformation Du et al. (2022b) ; Pan et al. (2023a) , complicating their ability to process and respond accurately  Chen et al. (2022) .
We term the conflict among various pieces of contextual knowledge as inter-context conflict ( IC , § 3 ).
To reduce uncertainties in responses, the user may pose the question in various forms. Therefore, the LLM’s parametric knowledge may yield divergent responses to these differently phrased questions. This variance can be attributed to the conflicting knowledge embedded within the LLM’s parameters, which stem from the inconsistencies present in the complex and diverse pre-training data sets  Huang et al. (2023) . This gives rise to what we term as intra-memory conflict ( IM , § 4 ).

 
 
 Knowledge conflict is originally rooted in open-domain QA research. The concept gained attention in  Longpre et al. (2021) that focused on the entity-based conflicts between parametric knowledge and external passages. Concurrently, discrepancies among multiple passages were also scrutinized subsequently  Chen et al. (2022) .
Knowledge conflicts attract significant attention with the recent advent of LLMs.
For instance, recent studies find that LLMs exhibit both adherence to parametric knowledge and susceptibility to contextual influences  Xie et al. (2023) , which can be problematic when this external knowledge is factually incorrect  Pan et al. (2023b) .
Given the implications for the trustworthiness ( Du et al., 2022b ) , real-time accuracy ( Kasai et al., 2022 ) , and robustness of LLMs ( Ying et al., 2023 ) , it is imperative to delve deeper into understanding and resolving knowledge conflicts ( Xie et al., 2023 ; Wang et al., 2023h ) .

 
 
 As of the time of writing, to the best of our knowledge, there is no systematic survey dedicated to the investigation of knowledge conflicts.
Existing reviews  Zhang et al. (2023d) ; Wang et al. (2023a) ; Feng et al. (2023) touch upon knowledge conflicts as a subtopic within their broader contexts.
While Feng et al. (2023) offer a more systematic examination of knowledge conflicts, categorizing them into external and internal conflicts. However, their survey provides only a brief overview of relevant works and primarily focuses on specific scenarios.
To fill the gap, we aim to provide a comprehensive review encompassing the categorization, cause and behavior analysis, and solutions for addressing various kinds of knowledge conflicts.

 
 
 We conceptualize the lifecycle of knowledge conflicts as both a cause leading to various behaviors, and an effect emerges from the intricate nature of knowledge as in Figure 2 . Knowledge conflicts serve as a crucial intermediary between causes and model behaviors. For instance, they significantly contribute to the model generating factually incorrect information, a.k.a. , hallucinations  Ji et al. (2023) ; Zhang et al. (2023d) .
Our research, in a manner akin to Freudian psychoanalysis, underscores the significance of understanding the origins of these conflicts. Although existing analyses Chen et al. (2022) ; Xie et al. (2023) ; Wang et al. (2023h) tend to construct such conflicts artificially, we posit that these analyses do not sufficiently address the interconnectedness of the issue.

 
 
 {forest} 
 
 Figure 3: Taxonomy of knowledge conflicts. We mainly list works in the era of LLMs. denotes pre-hoc solution and denotes post-hoc solution. 
 
 
 Going beyond reviewing and analyzing causes and behaviors, we delve deeper to provide a systematic review of solutions, which are employed to minimize the undesirable consequences of knowledge conflicts, i.e. , to encourage the model to exhibit desired behaviors that conform to specific objectives (please noted that these objectives may differ based on the particular scenario).
Based on the timing relative to potential conflicts, strategies are divided into two categories: pre-hoc and post-hoc strategies. The key distinction between them lies in whether adjustments are made before or after potential conflicts arise 1 1 
 1 
 
 
 
 Another interpretation is that a pre-hoc strategy is proactive while a post-hoc one is reactive. .
The taxonomy of knowledge conflicts is outlined in  Figure 3 . We sequentially discuss the three kinds of knowledge conflicts, detailing for each the causes, analysis of model behaviors, and available solutions organized according to their respective objectives.
Related datasets can be found in  Table 1 .

 
 
 
 
 
 
 
 
 Datasets 
 | 
 
 
 Approach 1 
 | 
 
 
 Base 2 
 | 
 
 
 Size 
 | 
 
 
 Conflict 
 | 

 
 
 
 Xie et al. (2023) 
 | 
 
 
 Gen 
 | 
 
 
 PopQA ( 2023 ), StrategyQA  ( Geva et al. (2021) ) 
 | 
 
 
 20,091 
 | 
 
 
 CM 3 
 | 

 
 
 
 KC ( 2023h ) 
 | 
 
 
 Sub 
 | 
 
 
 N/A (LLM generated) 
 | 
 
 
 9,803 
 | 
 
 
 CM 
 | 

 
 
 
 KRE ( 2023 ) 
 | 
 
 
 Gen 
 | 
 
 
 MuSiQue ( 2022 ), SQuAD2.0 ( 2018 ), ECQA ( 2021 ), e-CARE ( 2022a ) 
 | 
 
 
 11,684 
 | 
 
 
 CM 
 | 

 
 
 
 Farm ( 2023 ) 
 | 
 
 
 Gen 
 | 
 
 
 BoolQ ( 2019 ), NQ ( 2019 ), TruthfulQA ( 2022 ) 
 | 
 
 
 1,952 
 | 
 
 
 CM 
 | 

 
 
 
 Tan et al. (2024) 
 | 
 
 
 Gen 
 | 
 
 
 NQ ( 2019 ), TriviaQA ( 2017 ) 
 | 
 
 
 14,923 
 | 
 
 
 CM 
 | 

 
 
 
 WikiContradiction ( 2021 ) 
 | 
 
 
 Hum 
 | 
 
 
 Wikipedia 
 | 
 
 
 2,210 
 | 
 
 
 IC 
 | 

 
 
 
 ClaimDiff ( 2022 ) 
 | 
 
 
 Hum 
 | 
 
 
 N/A 
 | 
 
 
 2,941 
 | 
 
 
 IC 
 | 

 
 
 
 Pan et al. (2023a) 
 | 
 
 
 Gen,Sub 
 | 
 
 
 SQuAD v1.1 ( 2016 ) 
 | 
 
 
 52,189 
 | 
 
 
 IC 
 | 

 
 
 
 ContraDoc  ( 2023a ) 
 | 
 
 
 Gen 
 | 
 
 
 CNN-DailyMail ( 2015 ), NarrativeQA ( 2018 ), WikiText ( 2017 ) 
 | 
 
 
 449 
 | 
 
 
 IC 
 | 

 
 
 
 ConflictingQA  ( 2024 ) 
 | 
 
 
 Gen 
 | 
 
 
 N/A 
 | 
 
 
 238 
 | 
 
 
 IC 
 | 

 
 
 
 ParaRel  ( 2021 ) 
 | 
 
 
 Hum 
 | 
 
 
 T-REx ( 2018 ) 
 | 
 
 
 328 
 | 
 
 
 IM 
 | 

 
 
 • 
 
 1. Approach refers to how the conflicts are crafted, including entity-level substitution (Sub), generative approaches employing an LLM (Gen), and human annotation (Hum).

 

 • 
 
 2. Base refers to the base dataset(s) that serve as the foundation for generating conflicts, if applicable.

 

 • 
 
 3. 
When using CM datasets, conflicts arise from the specific model’s parametric knowledge, which can differ across models. Therefore, selecting a subset of the dataset that aligns with the tested model’s knowledge is crucial.

 

 
 Table 1: Datasets on evaluating LLMs’ behavior when encountering knowledge conflicts. CM : context-memory conflict, IC : inter-context conflict, IM : intra-memory conflict. 
 
 
 

## 2 C ontext- M emory Conflict

 
 Context-memory conflict emerges as the most extensively investigated among the three types of conflicts.
LLMs are characterized by fixed parametric knowledge, a result of the substantial pertaining process  Sharir et al. (2020) ; Hoffmann et al. (2022) ; Smith (2023) . This static parametric knowledge stands in stark contrast to the dynamic nature of external information, which evolves at a rapid pace  De Cao et al. (2021) ; Kasai et al. (2022) .

 
 

### 2.1 Causes

 
 The core of context-memory conflict stems from a discrepancy between the context and parametric knowledge.
We consider two main causes: temporal misalignment  Lazaridou et al. (2021) ; Luu et al. (2021) ; Dhingra et al. (2022) and misinformation pollution  Du et al. (2022b) ; Pan et al. (2023a) .

 
 
 Temporal Misalignment. 
Temporal misalignment naturally arises in models trained on data collected in the past, as they may not accurately reflect contemporary or future realities ( i.e. , the contextual knowledge after the deployment)  Luu et al. (2021) ; Lazaridou et al. (2021) ; Liska et al. (2022) . Such misalignment can degrade the model’s performance and relevancy over time, as it may fail to capture new trends, shifts in language use, cultural changes, or updates in knowledge. Researchers have noted that temporal misalignment relegates the model’s performance on various NLP tasks  Luu et al. (2021) ; Zhang and Choi (2021) ; Dhingra et al. (2022) ; Kasai et al. (2022) ; Cheang et al. (2023) .
Furthermore, the issue of temporal misalignment is expected to intensify due to the pre-training paradigm and the escalating costs associated with scaling up models  ( Kaplan et al., 2020 ) .

 
 
 Prior work tries to tackle temporal misalignment by focusing on three lines of strategies: Knowledge editing (KE) aims to directly update the parametric knowledge of an existing pre-trained model  ( Sinitsin et al., 2020 ; De Cao et al., 2021 ; Mitchell et al., 2021 ; Onoe et al., 2023 ) . Retrieval-augmented generation (RAG) leverages a retrieval module to fetch relevant documents from external sources ( e.g. , database, the Web) to supplement the model’s knowledge without altering its parameters  Karpukhin et al. (2020) ; Guu et al. (2020) ; Lewis et al. (2020) ; Lazaridou et al. (2022) ; Borgeaud et al. (2022) ; Peng et al. (2023) ; Vu et al. (2023) .
 Continue learning (CL) seeks to update the internal knowledge through continual pre-training on new and updated data  Lazaridou et al. (2021) ; Jang et al. (2021) ; Jang et al. (2022) .
However, these methods of mitigating temporal misalignment are not magic bullets. KE can bring in side effects of knowledge conflict, leading to knowledge inconsistency ( i.e. , a sort of intra-memory conflict) and may even enhance the hallucination of LLMs  Li et al. (2023f) ; Pinter and Elhadad (2023) .
For RAG, it is inevitable to encounter knowledge conflicts since the model’s parameters are not updated  Chen et al. (2021) ; Zhang and Choi (2021) .
CL suffers from catastrophic forgetting issues and demands significant computational resources  De Lange et al. (2021) ; He et al. (2021) ; Wang et al. (2023g) .

 
 
 Misinformation Pollution. 
Misinformation pollution emerges as another contributor to context-memory conflict, particularly for time-invariant knowledge  Jang et al. (2021) . Adversaries exploit this vulnerability by introducing false or misleading information into the Web corpus of retrieved documents  Pan et al. (2023a) ; Pan et al. (2023b) ; Weller et al. (2022) and user conversations  Xu et al. (2023) ; Hu et al. (2024) .
The latter poses a practical threat, as adversaries can leverage techniques such as prompt injection attacks  Liu et al. (2023b) ; Greshake et al. (2023) ; Yi et al. (2023) .
This vulnerability poses a real threat, as models might unknowingly propagate misinformation if they incorporate deceptive inputs without scrutiny  Xie et al. (2023) ; Pan et al. (2023b) ; Xu et al. (2023) .

 
 
 Fabricated, malicious misinformation can markedly undermine the accuracy of automated fact-checking  Du et al. (2022b) and open-domain question-answering systems  Pan et al. (2023a) ; Pan et al. (2023b) . Furthermore, recent studies also highlight the model’s tendency to align with user opinions, a.k.a. , sycophancy , further exacerbating the issue  Perez et al. (2022) ; Turpin et al. (2023) ; Wei et al. (2023) ; Sharma et al. (2023) .
In the current landscape of LLMs, there is growing apprehension in the NLP community regarding the potential generation of misinformation by LLMs  Ayoobi et al. (2023) ; Kidd and Birhane (2023) ; Carlini et al. (2023) ; Zhou et al. (2023c) ; Spitale et al. (2023) ; Chen and Shu (2023b) . Researchers acknowledge the challenges associated with detecting misinformation generated by LLMs  Tang et al. (2023) ; Chen and Shu (2023a) ; Jiang et al. (2023) . This underscores the urgency of addressing the nuanced challenges posed by LLMs in the context of contextual misinformation.

 
 
 Remarks. 
Temporal misalignment and misinformation pollution are two separate scenarios that give rise to context-memory conflicts. For the former, the up-to-date contextual information is considered accurate.
 Conversely , for the latter, the contextual information contains misinformation and is therefore considered incorrect.

 
 
 

### 2.2 Analysis of Model Behaviors

 
 How do LLMs navigate context-memory conflicts? 
This section will detail the relevant research, although they present quite different answers.
Depending on the scenario, we first introduce the Open-domain question answering (ODQA) setup and then focus on general setups.

 
 
 ODQA. In earlier ODQA literature, Longpre et al. (2021) explore how QA models act when the provided contextual information contradicts the learned information.
The authors create an automated framework that identifies QA instances with named entity answers, then substitutes mentions of the entity in the gold document with an alternate entity, thus creating the conflict context. This study reveals a tendency of these models to over-rely on parametric knowledge.
 Chen et al. (2022) revisit this setup while reporting differing observations, they note that models predominantly rely on contextual knowledge in their best-performing settings. They attribute this divergence in findings to two factors. Firstly, the entity substitution approach used by Longpre et al. (2021) potentially reduces the semantic coherence of the perturbed passages. Secondly, Longpre et al. (2021) based their research on single evidence passages, as opposed to  Chen et al. (2022) , who utilize multiple ones.
Recently, with the emergence of really “large” language models such as ChatGPT  ( Ouyang et al., 2022 ; OpenAI, 2023 ) and Llama 2  ( Touvron et al., 2023 ) , inter alia , researchers re-examined this issue.
 Tan et al. (2024) examine how LLMs blend retrieved context with generated knowledge in the ODQA setup, and discover models tend to favor the parametric knowledge, influenced by the greater resemblance of these generated contexts to the input questions and the often incomplete nature of the retrieved information, especially within the scope of conflicting sources.

 
 
 General. 
 Xie et al. (2023) leverage LLMs to generate conflicting context alongside the memorized knowledge. They find that LLMs are highly receptive to external evidence, even when it conflicts with their parametric, provided that the external knowledge is coherent and convincing. Meanwhile, they also identify a strong confirmation bias  Nickerson (1998) in LLMs, i.e. , the models tend to favor information consistent with their internal memory, even when confronted with conflicting external evidence.
 Wang et al. (2023h) posit that the desired behaviors when an LLM encounters conflicts should be to pinpoint the conflicts and provide distinct answers. While LLMs perform well in identifying the existence of knowledge conflicts, they struggle to determine the specific conflicting segments and produce a response with distinct answers amidst conflicting information.
 Ying et al. (2023) analyze the robustness of LLMs under conflicts with a focus on two perspectives: factual robustness (the ability to identify correct facts from prompts or memory) and decision style (categorizing LLMs’ behavior as intuitive, dependent, or rational-based on cognitive theory).
The study finds that LLMs are highly susceptible to misleading prompts, especially in the context of commonsense knowledge.
 Qian et al. (2023) evaluate the potential interaction between parametric and external knowledge more systematically, cooperating knowledge graph (KG). They reveal that LLMs often deviate from their parametric knowledge when presented with direct conflicts or detailed contextual changes.
 Xu et al. (2023) study how LLMs respond to knowledge conflicts during interactive sessions. Their findings suggest LLMs tend to favor logically structured knowledge, even when it contradicts factual accuracy.

 
 
 Remarks. 
 I. Crafting Conflicting Knowledge. 
Model’s behavior under context-memory conflict is analyzed by artificially creating conflicting knowledge, in early years through entity-level substitutions and more recently by employing LLMs to generate semantically coherent conflicts.

 
 
 II. What is the conclusion? 
No definitive rule exists for whether a model prioritizes contextual or parametric knowledge. Yet, knowledge that is semantically coherent, logical, and compelling is typically favored by models over generic conflicting information.

 
 
 

### 2.3 Solutions

 
 Solutions are organized according to their objectives , i.e. , the desired behaviors we expect from an LLM when it encounters conflicts.
Existing strategies can be categorized into the following objectives: Faithful to context strategies aim to align with contextual knowledge, focusing on context prioritization. Discriminating misinformation strategies encourage skepticism towards dubious context in favor of parametric knowledge.
 Disentangling sources strategies treat context and knowledge separately and provide disentangled answers.
 Improving factuality strategies aim for an integrated response leveraging both context and parametric knowledge towards a more truthful solution.

 
 
 Faithful to Context. 
 Fine-tuning. 
 Li et al. (2022a) argue that an LLM should prioritize context for task-relevant information and rely on internal knowledge when the context is unrelated.
They name the two properties controllability and robustness. They introduce Knowledge Aware FineTuning (KAFT) to strengthen the two properties by incorporating counterfactual and irrelevant contexts into standard training datasets.
TrueTeacher  ( Gekhman et al., 2023 ) focuses on improving factual consistency in summarization by annotating model-generated summaries with LLMs. This approach helps in maintaining faithfulness to the context of the original documents, ensuring that generated summaries remain accurate without being misled by irrelevant or incorrect details.
DIAL  ( Xue et al., 2023 ) improves factual consistency in dialogue systems via direct knowledge enhancement and reinforcement learning for factual consistency (RLFC) for aligning responses accurately with provided factual knowledge.

 
 
 Prompting. 
 Zhou et al. (2023d) explores enhancing LLMs’ adherence to context through specialized prompting strategies, specifically opinion-based prompts and counterfactual demonstrations. These techniques are shown to significantly improve LLMs’ performance in context-sensitive tasks by ensuring they remain faithful to relevant context, without additional training.

 
 
 Decoding. 
 Shi et al. (2023a) introduce Context-aware Decoding (CAD) to reduce hallucinations by amplifying the difference in output probabilities with and without context, similar to the concept of contrastive decoding  Li et al. (2022c) . CAD enhances faithfulness in LLMs by prioritizing relevant context over the model’s prior knowledge, especially in tasks with conflicting information.

 
 
 Knowledge Plug-in. 
 Lee et al. (2022a) propose Continuously-updated QA (CuQA) for improving LMs’ ability to integrate new knowledge. Their approach uses plug-and-play modules to store updated knowledge, ensuring the original model remains unaffected. Unlike traditional continued pre-training or fine-tuning approaches, CuQA can solve knowledge conflicts.

 
 
 Pre-training. 
ICLM  ( Shi et al., 2023b ) is a new pre-training method that extends LLMs’ ability to handle long and varied contexts across multiple documents. This approach could potentially aid in resolving knowledge conflicts by enabling models to synthesize information from broader contexts, thus improving their understanding and application of relevant knowledge.

 
 
 Predict Fact Validity. 
 Zhang and Choi (2023) address knowledge conflict by introducing fact duration prediction to identify and discard outdated facts in LLMs. This approach improves model performance on tasks like ODQA by ensuring adherence to up-to-date contextual information.

 
 
 Discriminating Misinformation (Faithful to Memory). 
 Prompting. 
To address misinformation pollution, Pan et al. (2023b) propose defense strategies such as misinformation detection and vigilant prompting, aiming to enhance the model’s ability to remain faithful to factual, parametric information amidst potential misinformation.
Similarly, Xu et al. (2023) utilize a system prompt to remind the LLM to be cautious about potential misinformation and to verify its memorized knowledge before responding. This approach aims to enhance the LLM’s ability to maintain faithfulness.

 
 
 Query Augmentation. 
 Weller et al. (2022) leverage the redundancy of information in large corpora to defend misinformation pollution. Their method involves query augmentation to find a diverse set of less likely poisoned passages, coupled with a confidence method named Confidence from Answer Redundancy, which compares the predicted answer’s consistency across retrieved contexts. This strategy mitigates knowledge conflicts by ensuring the model’s faithfulness through the cross-verification of answers from multiple sources.

 
 
 Training Discriminator. 
 Hong et al. (2023) fine-tune a smaller LM as a discriminator and combine prompting techniques to develop the model’s ability to discriminate between reliable and unreliable information, helping the model remain faithful when confronted with misleading context.

 
 
 Disentangling Sources. 
DisentQA  ( Neeman et al., 2022 ) trains a model that predicts two types of answers for a given question: one based on contextual knowledge and one on parametric knowledge.
 Wang et al. (2023h) introduce a method to improve LLMs’ handling of knowledge conflicts. Their approach is a three-step process designed to help LLMs detect conflicts, accurately identify the conflicting segments, and generate distinct, informed responses based on the conflicting data, aiming for more precise and nuanced model outputs.

 
 
 Improving Factuality. 
 Zhang et al. (2023e) propose COMBO, a framework that pairs compatible generated and retrieved passages to resolve discrepancies. It uses discriminators trained on silver labels to assess passage compatibility, improving ODQA performance by leveraging both LLM-generated (parametric) and external retrieved knowledge.
 Jin et al. (2024a) introduce a contrastive-decoding-based algorithm, namely CD 2 , which maximizes the difference between various logits under knowledge conflicts and calibrates the model’s confidence in the truthful answer.

 
 
 Remarks. 
Current mitigation approaches have contradicted goals because they do not distinguish between the two causes of knowledge conflict when considering conflict scenarios. Blindly being “faithful” to context or knowledge is undesirable.
Some researchers regard that LLM should not rely solely on either parametric or contextual information but instead grant LLM users the agency to make informed decisions based on distinct answers  ( Wang et al., 2023h ; Floridi, 2023 ) .

 
 
 
 

## 3 I nter- C ontext Conflict

 
 Inter-context conflicts manifest in LLMs when incorporating external information sources, a challenge accentuated by the advent of RAG techniques. RAG enriches the LLM’s responses by integrating content from retrieved documents into the context. Nonetheless, this incorporation can lead to inconsistencies within the provided context, as the external documents may contain information that conflicts with each other  Zhang and Choi (2021) ; Kasai et al. (2022) ; Li et al. (2023a) .

 
 

### 3.1 Causes

 
 Misinformation. 
Misinformation has long been a significant concern in the modern digital age  Shu et al. (2017) ; Zubiaga et al. (2018) ; Kumar and Shah (2018) ; Meel and Vishwakarma (2020) ; Fung et al. (2022) ; Wang et al. (2023b) .
The emergence of RAG incorporates external documents to enhance the generation quality of LLMs. While RAG has the potential to enrich content with diverse knowledge sources, it also poses the risk of including documents containing misinformation, such as fake news  Chen et al. (2023b) .
Moreover, there have been instances where AI technologies are employed to create or propagate misinformation  Weidinger et al. (2021) ; Zhou et al. (2023c) ; Vergho et al. (2024) . The advanced generative capabilities of LLMs exacerbate this issue, leading to an increase in misinformation generated by these systems. This trend is concerning, as it not only contributes to the spread of false information but also challenges detecting misinformation generated by LLMs  Chen and Shu (2023b) ; Menczer et al. (2023) ; Barrett et al. (2023) ; Bengio et al. (2023) ; Wang et al. (2023c) ; Solaiman et al. (2023) ; Weidinger et al. (2023) ; Ferrara (2023) ; Goldstein et al. (2023) .

 
 
 Outdated Information. 
In addition to the challenge of misinformation, it is important to recognize that facts can evolve. The retrieved documents may contain updated and outdated information from the network simultaneously, leading to conflicts between these documents  Chen et al. (2021) ; Liska et al. (2022) ; Zhang and Choi (2021) ; Kasai et al. (2022) ; Schlichtkrull et al. (2023) .

 
 
 Remarks. 
Conflicts in context frequently arise between misinformation and accurate information, as well as between outdated and updated information. These two conflicts exert distinct impacts on LLMs and require specified analysis.
Distinguishing from misinformation conflicts, another significant challenge involves addressing conflicts that arise from documents bearing different timestamps, especially when a user’s prompt specifies a particular time period.

 
 
 

### 3.2 Analysis of Model Behaviors

 
 Performance Impact. 
Previous research empirically demonstrates that the performance of a pre-trained language model can be significantly influenced by the presence of misinformation  Zhang and Choi (2021) or outdated information  Du et al. (2022b) within a specific context. In recent studies, Pan et al. (2023a) introduce a misinformation attack strategy involving the creation of a fabricated version of Wikipedia articles, which is subsequently inserted into the authentic Wikipedia corpus. Their research findings reveal that existing language models are susceptible to misinformation attacks, irrespective of whether the fake articles are manually crafted or generated by models. To gain a deeper understanding of how LLMs behave when encountering contradictory contexts, Chen et al. (2022) primarily conduct experiments using Fusion-in-Decoder on the NQ-Open  Kwiatkowski et al. (2019) and TriviaQA  Joshi et al. (2017) . They find that inconsistencies across knowledge sources exert a minimal effect on the confidence levels of models. These models tend to favor context directly pertinent to the query and context that aligns with the model’s inherent parametric knowledge. Xie et al. (2023) conduct experiments on both closed-source LLMs and open-source LLMs in PopQA   Mallen et al. (2022) and StrategyQA   Geva et al. (2021) . The results obtained are in line with those of Chen et al. (2022) , indicating that LLMs exhibit a significant bias to evidence that aligns with the model’s parametric memory. They also find that LLMs exhibit a predisposition towards emphasizing information related to entities of higher popularity and answers that are corroborated by a larger volume of documents within the given context. Moreover, these models demonstrate a significant sensitivity to the order in which data is introduced. Jin et al. (2024a) discover that as the number of conflicting hops increases, LLMs encounter increased challenges in reasoning.

 
 
 Detection Ability. 
In addition to assessing the performance of LLMs when confronted with contradictory contexts, several studies also investigate their capacity to identify such contradictions. Zheng et al. (2022) examine the performance of various models including BERT, RoBERTa, and ERNIE in detecting the contradiction within Chinese conversations. Experiments reveal that identifying contradictory statements within a conversation is a significant challenge for these models. Li et al. (2023a) 
analyse the performance of GPT-4, PaLM-2, and Llama 2 in identifying contradictory documents within news articles  Hermann et al. (2015) , stories  Kočiský et al. (2018) , and wikipedia  Merity et al. (2017) . The authors find that the average detection accuracy is subpar. The study also finds that LLMs face specific challenges when addressing certain types of contradictions, particularly those involving subjective emotions or perspectives. Additionally, the length of documents and the variety of self-contradictions have a minor influence on the detection performance.
 Wan et al. (2024) investigate the text features that affect LLMs’ assessment of document credibility when faced with conflicting information. They discover that existing models heavily prioritize the relevance of a document to the query but often overlook stylistic features that humans consider important, such as the presence of scientific references or a neutral tone in the text.
 Jin et al. (2024a) discover that LLMs encounter difficulty in distinguishing truthful information from misinformation. In addition, they find that LLMs favor evidence that appears most frequently within the context, and demonstrate confirmation bias for external information aligning with their internal memory.

 
 
 Remarks. 
When encountering conflict within a given context, the exhibited knowledge of the LLMs is significantly influenced. However, determining how the model responds to various contextual nuances remains an area requiring further exploration. While different models may share certain commonalities, disparities in behavior arise due to variations in their training data. Moreover, as the model’s knowledge is derived from textual information, its approach to discerning misinformation differs significantly from that of humans.

 
 
 

### 3.3 Solutions

 
 Eliminating Conflict. 
 Specialized Models. 
 Hsu et al. (2021) develop a model named Pairwise Contradiction Neural Network (PCNN), leveraging fine-tuned Sentence-BERT embeddings to calculate contradiction probabilities of articles. Pielka et al. (2022) suggest incorporating linguistic knowledge into the learning process based on the discovery that XLM-RoBERTa struggles to effectively grasp the syntactic and semantic features that are vital for accurate contradiction detection. Wu et al. (2022) propose an innovative approach that integrates topological representations of text into language models to enhance the contradiction detection ability and evaluated their methods on the MultiNLI dataset  Williams et al. (2018) .

 
 
 General Models. 
 Chern et al. (2023) propose a fact-checking framework that integrates LLMs with various tools, including Google Search, Google Scholar, code interpreters, and Python, for detecting factual errors in texts. Leite et al. (2023) employ LLMs to generate weak labels associated with predefined credibility signals for the input text and aggregate these labels through weak supervision techniques to make predictions regarding the veracity of the input.

 
 
 Improving Robustness. 
 Training Approach. 
 Hong et al. (2023) present a novel fine-tuning method that involves training a discriminator and a decoder simultaneously using a shared encoder. Additionally, the authors introduce two other strategies to improve the robustness of the model including prompting GPT-3 to identify perturbed documents before generating responses and integrating the discriminator’s output into the prompt for GPT-3. Their experimental results indicate that the fine-tuning method yields the most promising results.

 
 
 Query Augmentation. 
 Weller et al. (2022) explore a query augmentation technique that prompts GPT-3 to formulate new questions derived from the original inquiry. They then assess the confidence for each answer by referencing the corresponding passages retrieved. Based on the confidence, they decide whether to rely on the original question’s prediction or aggregate predictions from the augmented questions with high confidence scores.

 
 
 Remarks. 
Strategies for addressing inter-context conflicts primarily rely on model knowledge or leveraging external knowledge such as retrieved documents. Recently, augmenting LLM with external tools has emerged as a new paradigm. Exploring the use of external tools to support LLMs in resolving inter-context conflicts could be a promising approach. On the other hand, devising unified and efficient methods to handle various conflict types remains a formidable challenge.

 
 
 
 

## 4 I ntra- M emory Conflict

 
 With the development of LLMs, LLMs are widely used in knowledge-intensive question-and-answer systems  Gao et al. (2023b) ; Yu et al. (2022) ; Petroni et al. (2019) ; Chen et al. (2023c) . A critical aspect of deploying LLMs effectively involves ensuring that they produce consistent outputs across various expressions that share similar meanings or intentions.
Despite this necessity, a notable challenge arises with intra-memory conflict—a condition where LLMs exhibit unpredictable behaviors and generate differing responses to inputs that are semantically equivalent but syntactically distinct  Chang and Bergen (2023) ; Chen et al. (2023a) ; Raj et al. (2023) ; Rabinovich et al. (2023) ; Raj et al. (2022) ; Bartsch et al. (2023) . Intra-memory conflict essentially undermines the reliability and utility of LLMs by introducing a degree of uncertainty in their output.

 
 

### 4.1 Causes

 
 Intra-memory conflicts within LLMs can be attributed to three primary factors: training corpus bias  Wang et al. (2023d) ; Xu et al. (2022) , decoding strategies  Lee et al. (2022b) ; Huang et al. (2023) , and knowledge editing  Yao et al. (2023) ; Li et al. (2023f) . These factors respectively pertain to the training phase, the inference phase, and subsequent knowledge refinement.

 
 
 Bias in Training Corpora. 
Recent research demonstrates that the primary phase for knowledge acquisition in LLMs predominantly occurs in the pre-training stage  Zhou et al. (2023a) ; Kaddour et al. (2023) ; Naveed et al. (2023) ; Akyürek et al. (2022) ; Singhal et al. (2022) . Pre-training corpus is primarily crawled from the internet, which exhibits a diverse range of data quality, potentially including inaccurate or misleading information  Bender et al. (2021) ; Weidinger et al. (2021) . When LLMs are trained on data containing incorrect knowledge, they may memorize and inadvertently amplify these inaccuracies  Lin et al. (2022) ; Elazar et al. (2022) ; Lam et al. (2022) ; Grosse et al. (2023) , leading to a situation where conflicting knowledge coexists within the parameters of LLMs.

 
 
 Moreover, prior works indicate that LLMs tend to encode superficial associations prevalent within their training data, as opposed to genuinely comprehending the underlying knowledge contained therein  Li et al. (2022b) ; Kang and Choi (2023) ; Zhao et al. (2023a) ; Kandpal et al. (2023) . It can result in LLMs displaying a propensity to generate predetermined responses rooted in spurious correlations of training data. Due to the dependency on spurious correlations, LLMs may provide divergent answers when presented with prompts exhibiting distinct syntactic structures but conveying equivalent semantic meaning.

 
 
 Decoding Strategy. 
The direct output of LLMs is a probability distribution over potential next tokens. Sampling is a crucial step in determining the generated content from this distribution. Various sampling techniques, including greedy sampling, top-p sampling, top-k sampling, and others have been proposed  Jawahar et al. (2020) ; Massarelli et al. (2020) , broadly categorizing into deterministic and stochastic sampling methods. Stochastic sampling stands as the prevailing decoding strategy employed by LLMs  Fan et al. (2018) ; Holtzman et al. (2020) .
However, the random nature of stochastic sampling methods introduces uncertainty into the generated content. Moreover, due to the intrinsic left-to-right generation pattern inherent to LLMs, the selection of the sampling token can wield a significant influence over the subsequent generations. The use of stochastic sampling may cause LLMs to produce entirely different content, even when provided with the same context, causing intra-memory conflict  Lee et al. (2022b) ; Huang et al. (2023) ; Dziri et al. (2021) .

 
 
 Knowledge Editing. 
With the exponential increase of model parameters, fine-tuning LLMs become increasingly challenging and resource-intensive. In response to this challenge, researchers explore knowledge editing techniques as a means of efficiently modify a small scope of the knowledge encoded in LLMs  Meng et al. (2022) ; Ilharco et al. (2022) ; Zhong et al. (2023) .
Ensuring the consistency of modifications poses a significant challenge. Due to the potential limitations inherent in the editing method, the modified knowledge cannot be generalized effectively. This can result in LLMs producing inconsistent responses when dealing with the same piece of knowledge in varying situations  Li et al. (2023f) ; Yao et al. (2023) . Intra-memory conflict is primarily considered a side effect in the context of knowledge editing.

 
 
 Remarks. 
Intra-memory conflicts in LLMs arise from three distinct causes that occur at different stages. Among these causes, training corpus bias stands out as the fundamental catalyst. Incongruities of knowledge in
the training dataset result in inconsistencies within the knowledge encoded within the model’s parameters. Additionally, the decoding strategy indirectly contributes to exacerbating these conflicts. The inherent randomness of the sampling process during inference amplifies the inconsistencies in the model’s responses. Knowledge editing, which aims to post-update the model’s knowledge, can inadvertently introduce conflicting information into the LLM’s memory.

 
 
 

### 4.2 Analysis of Model Behaviors

 
 Self-Inconsistency. 
 Elazar et al. (2021) develop a method for assessing the knowledge consistency of language models, focusing specifically on knowledge triples. The authors primarily conduct experiments using BERT, RoBERTa, and ALBERT. Their findings indicate that these models exhibit poor consistency, with accuracy rates barely ranging from 50% to 60%. Hase et al. (2023) employ the same indicators of Elazar et al. (2021) , but they utilize a more diverse dataset. Their study also reveals that the consistency of RoBERTa-base and BART-base within the paraphrase context is lacking. Zhao et al. (2023b) reformulate questions and then assess the consistency of the LLM’s responses to these reformulated questions. The findings of their research reveal that even GPT-4 exhibits a notable inconsistency rate of 13% when applied to Commonsense Question-Answering tasks. They further find that LLMs are more likely to produce inconsistencies in the face of uncommon knowledge. Dong et al. (2023) conduct experiments on multiple open-source LLMs and find that all of these models exhibit strong inconsistencies. Li et al. (2023d) explore an additional aspect of inconsistency that LLMs can give an initial answer to a question, but it may subsequently deny the previous answer when asked if it is correct. The authors conduct experiments focusing on Close-Book Question Answering and reveal that Alpaca-30B only displays consistency in 50% of cases.

 
 
 To further analyze the inconsistency exhibited by LLMs, a study conducted by Li et al. (2022b) reveals that encoder-based models tend to generate mis-factual words more relying on positionally close and highly co-occurring words, rather than knowledge-dependent words. This phenomenon arises due to these models’ tendency to overlearn inappropriate associations from the training dataset. Kang and Choi (2023) highlight a co-occurrence bias in LLMs, where the models favor frequently co-occurring words over correct answers. especially when recalling facts where the subject and object rarely co-occur in the pre-training dataset, despite fine-tuning. Furthermore, their research indicates that LLMs face challenges in recalling facts in cases where the subject and object rarely appear together in the pre-training dataset, even though these facts are encountered during fine-tuning.

 
 
 Latent Representation of Knowledge. 
The multi-layer transformer architecture inherent to contemporary LLMs fosters a complex inter-memory conflict, with distinct knowledge representations scattered across various layers. Previous research suggests that LLMs store low-level information at shallower levels and semantic information at deeper levels  Tenney et al. (2019) ; Rogers et al. (2020) ; Wang et al. (2019) ; Jawahar et al. (2019) ; Cui et al. (2020) . Chuang et al. (2023) explore this aspect within the context of LLMs and discover that the factual knowledge in LLMs is typically concentrated within specific transformer layers and different layers of inconsistent knowledge. Moreover, Li et al. (2023c) discover that the correct knowledge is indeed stored within the parameters of the model, but it may not be accurately expressed during the generation process. The authors conduct two experiments on the same LLM, one focused on the generation accuracy, and the other utilizing a knowledge probe to examine the knowledge containment. The results of these experiments reveal a substantial 40% disparity between the knowledge probe accuracy and the generation accuracy.

 
 
 Cross-lingual Inconsistency. 
The universality of true knowledge transcends surface form variations  Ohmer et al. (2023) , a principle that should ideally apply to LLMs. However, LLMs maintain distinct knowledge sets for different languages, leading to inconsistencies  Ji et al. (2023) ; Xue et al. (2024) . Wang et al. (2023f) investigate the challenges LLMs face in extending edited knowledge across languages, suggesting that knowledge related to different languages is stored separately within the model parameters. Qi et al. (2023) propose a metric named RankC for evaluating the cross-lingual consistency of LLMs’ factual knowledge. They employ this metric for analyzing multiple models and reveal a pronounced language dependence in the knowledge stored by LLMs, with no observed improvement in cross-lingual consistency with increased model size.

 
 
 Remarks. 
The phenomenon of inter-memory conflict in LLMs predominantly manifests through inconsistent responses to semantically identical queries. This inconsistency is primarily attributed to the suboptimal quality of datasets utilized during the pre-training phase. Addressing this challenge necessitates the development of efficient and cost-effective solutions, which remains a significant hurdle. Additionally, LLMs are characterized by the presence of multiple knowledge circuits, which significantly influence their response mechanisms to specific inquiries. The exploration and detailed examination of these knowledge circuits within LLMs represent a promising avenue for future research.

 
 
 

### 4.3 Solutions

 

#### 4.3.1 Improving Consistency

 
 Fine-tuning. 
 Elazar et al. (2021) propose a consistency loss function and train the language model with the combination of the consistency loss and standard MLM loss.
 Li et al. (2023d) utilize one language model in dual capacities: as a generator to produce responses and as a validator to evaluate the accuracy of these responses. The process involves querying the generator for a response, which is subsequently assessed by the validator for accuracy. Only those pairs of responses deemed consistent are retained. This subset of consistent pairs is then used to fine-tune the model, aiming to increase the generation likelihood of consistent response pairs.

 
 
 Plug-in. 
 Jang and Lukasiewicz (2023) leverage the technique of intermediate training, utilizing word-definition pairs from dictionaries to retrain language models and improve their comprehension of symbolic meanings. Subsequently, they propose an efficient parameter integration approach, which amalgamates these enhanced parameters with those of existing language models. This method aims to rectify the models’ inconsistent behavior by bolstering their capacity to understand meanings.

 
 
 Output Ensemble. 
 Mitchell et al. (2022) mitigate the inconsistency of language models by leveraging a two-model architecture, involving the utilization of a base model responsible for generating a set of potential answers, followed by a relation model that evaluates the logical coherence among these answers. The final answer is selected by considering both the base model’s and the relation model’s beliefs. Zhao et al. (2023b) introduce a method to detect whether a question may cause inconsistency for LLMs. Specifically, they first use LLMs to rephrase the original question and obtain corresponding answers. They then cluster these answers and examine the divergence. The detection is determined based on the divergence level.

 
 
 

#### 4.3.2 Improving Factuality

 
 Chuang et al. (2023) propose a novel contrastive decoding approach named DoLa. Specifically, the authors develop a dynamic layer selection strategy, choosing the appropriate premature layers and mature layers. The next word’s output probability is then determined by computing the difference in log probabilities of the premature layers and the mature layers.
 Li et al. (2023c) propose a similar method named ITI. They first identify a sparse set of attention heads that exhibit high linear probing accuracy for truthfulness, as measured by TruthfulQA  Lin et al. (2022) . During the inference phase, ITI shifts activations along the truth-correlated direction, which is obtained through knowledge probing. This intervention is repeated autoregressively for every token during completion. Both DoLa and ITI address the inconsistency of knowledge across the model’s different layers to reduce factual errors in LLMs.

 
 
 Remarks. 
The resolution of inter-memory conflict in LLMs typically entails three phases: training, generation, and post-hoc processing. The training phase method mainly focuses on mitigating internal inconsistencies among model parameters. Conversely, the generation and post-hoc phases primarily involve algorithmic interventions aimed at alleviating occurrences of inconsistent model behavior. Nevertheless, the challenge persists in addressing the inconsistency of parameter knowledge without detrimentally impacting the overall performance of LLMs.

 
 
 
 
 

## 5 Challenges and Future Directions

 
 In this section, we provide a summary and highlight the existing challenges in ongoing research, as well as outline potential future directions in the field of knowledge conflict.

 
 
 Knowledge Conflicts in the Wild. 
Currently, the creation of knowledge conflicts predominantly relies on the artificial generation of incorrect or misleading information. In the real world, one of the most common situations where knowledge conflicts arise is in RALMs (Retrieval-Augmented Language Models), where conflicts are present in documents retrieved by the retrieval module directly from the Web. Current analysis approaches exist a gap in the experimental setup of knowledge conflict, suggesting that findings from those environments  Xie et al. (2023) ; Wang et al. (2023h) might not easily transfer to practical applications.
Recent studies have begun to investigate the scenario in the wild by curating conflicting documents based on actual search results from Google for open-ended questions  Wan et al. (2024) .
Looking ahead, there is a growing interest in more research that assesses how well LLMs perform in real-world scenarios rather than artificially created conflicts, to better understand their capabilities.

 
 
 Solution at a Finer Resolution. 
Currently, there is no one-size-fits-all solution to knowledge conflict due to its inherent complexity. Existing either assume a prior  Shi et al. (2023b) or focus on a subclass of conflict  Wang et al. (2023h) .
We believe that addressing this issue requires a more fine-grained approach, taking into account several factors. Firstly, the nature of the user’s query plays a crucial role. Subjective or debatable questions naturally lead to conflicts as they may have multiple valid answers  Bjerva et al. (2020) ; Wan et al. (2024) .
Secondly, the source of conflicting information can vary, including misinformation, outdated facts, or partially correct data  Guo et al. (2022) ; Akhtar et al. (2023) .
Lastly, it is also important to consider user expectations, such as whether they prefer a single definitive answer from the LLM or are open to multiple perspectives  Floridi (2023) .
Given these considerations, future solutions to mitigate knowledge conflicts must delve into these nuances, recognizing that knowledge conflict encompasses a spectrum of problems with diverse causes, manifestations, and potential resolutions.
Collaboration between NLP and HCI researchers is appreciated for conducting thorough investigations and developing effective solutions.

 
 
 Evaluation on Downstream Tasks. 
The current landscape of research into knowledge conflicts within LLMs predominantly emphasizes evaluating their performance on common QA datasets including NQ-Open, TriviaQA, OPQA, and StrategyQA . This focus overlooks the broader implications of knowledge conflicts, particularly how they influence downstream tasks. Exploring the effects of knowledge conflicts on a wider range of applications beyond QA problems could yield insights into creating more robust and reliable models. For instance, in tasks requiring high levels of accuracy and consistency, such as legal document analysis  Shui et al. (2023) ; Martin et al. (2024) , medical diagnosis  Zhou et al. (2023b) ; Thirunavukarasu et al. (2023) , financial analysis  Zhang et al. (2023a) ; Li et al. (2023e) and educational tools  Caines et al. (2023) ; Milano et al. (2023) , the presence of unresolved knowledge conflicts could undermine the model’s utility.

 
 
 Interplay among the Conflicts. 
Current research in the knowledge conflict of LLMs primarily concentrates on investigating conflicts of a singular type   Wang et al. (2023h) ; Chen et al. (2022) ; Li et al. (2023d) or a joint study of inter-context and context-memory conflict  Jin et al. (2024a) ; Xie et al. (2023) . However, there is a notable dearth of research on the interaction between intra-memory conflict and the other two types of conflicts. Several papers have proposed the existence of knowledge circuits in LLMs  Chughtai et al. (2024) ; Huang et al. (2023) , which are closely related to the intra-memory conflict.
Addressing this gap is crucial for understanding the relationship between the internal knowledge inconsistency of the model and its behavior in response to the context. Moreover, exploring the synergistic effects of various conflict types could unveil underlying mechanisms of knowledge representation and processing in LLMs and help us to develop more robust and accurate LLMs in practice.

 
 
 Explainability. 
Most recent work analyzed LLMs’ behaviors amidst knowledge conflicts at the output level  Xie et al. (2023) ; Wang et al. (2023h) . While some studies have observed and explored the model’s confidence in its output, i.e. , logits  Xu et al. (2023) ; Jin et al. (2024a) ; Wang et al. (2024b) , there has been less focus on the internal mechanism of the model, like specific attention heads or neuron activations during conflicts. This gap highlights a need for more microscopic examinations to better comprehend how models decide when encounter conflicts.
A recent study conducted by Jin et al. (2024b) advances this by investigating the interpretability of LLMs through information flow analysis, pinpointing pivotal points for conflict mitigation. They discover there are some attention heads with opposite effects in the later layers, where memory heads can recall knowledge from internal memory, and context heads can retrieve knowledge from the external context. Inspired by this, Pruning Head via Path Patching is introduced to resolve conflicts efficiently without updating model parameters.

 
 
 Multilinguality. 
To date, research on knowledge conflict has primarily focused on the English language. Future studies could expand in two directions.
Firstly, by examining LLMs to address knowledge conflicts in non-English prompts, leveraging the many advanced non-English models available ( e.g. , GLM  Zeng et al. (2022) for Chinese) or LLMs with multilingual capability ( e.g. , GPT-4  OpenAI (2024) ) and noting differences from English to account for unique language characteristics.
Secondly, addressing inter-context conflict where multiple documents in different languages might be retrieved, potentially involving cross-language knowledge conflicts. Solutions could include employing translation systems  Dementieva and Panchenko (2021) or, for low-resource languages, leveraging high-resource language evidence ( Xue et al., 2024 ) or employing knowledge distillation techniques.

 
 
 Multimodality. 
Current research on knowledge conflicts mainly focused on the text modality, leaving the study of these conflicts in multimodal contexts as a promising area for future exploration.
As LLMs evolve to process information across various formats—images  Alayrac et al. (2022) ; Li et al. (2023b) , video  Ju et al. (2022) ; Zhang et al. (2023b) , and audio  Borsos et al. (2023) ; Wu et al. (2023) —the potential for conflicts escalates in growing complexity.
For instance, textual documents might clash with visual data, or the tone of an audio clip might contradict the content of an accompanying caption.
Future research on multimodal knowledge conflicts could focus on crafting advanced LLMs skilled in cross-modal reasoning and conflict resolution across diverse data types. This effort necessitates the enhancement of models’ capabilities to navigate the complex dynamics between different modalities and the development of targeted datasets for effective training and evaluation. Additionally, exploring how users perceive and manage multimodal conflicts, such as discrepancies between text and images, will offer valuable insights into improving LLMs for better human interaction.

 
 
 

## 6 Conclusion

 
 Through this survey, we have extensively investigated knowledge conflicts, shedding light on their categorization, causes, how LLMs respond to these conflicts, and possible solutions.
Our findings reveal that knowledge conflict is a multifaceted issue, with a model’s behavior being closely tied to the particular type of conflicting knowledge. Besides, there appears to be a more complex interplay among the three types of conflicts.
Furthermore, we observe that existing solutions primarily address artificially constructed scenarios, neglecting the subtleties of conflicts by relying on assumed priors and thus sacrificing granularity and breadth.
Given the growing use of retrieval-augmented language models, we anticipate that knowledge conflicts faced by LLMs will only increase in complexity, underscoring the need for more comprehensive research.

 
 
 

## Limitations

 
 Considering the rapid expansion of research in the field of knowledge conflict and the abundance of scholarly literature, it is possible that we might have missed some of the most recent or less relevant findings. Nevertheless, we have ensured the inclusion of all essential materials in our survey.

 
 
 

## Ethics Statement

 
 We mainly searched for papers published after 2021 using key terms including “knowledge conflict”, “knowledge inconsistency”, “knowledge gap”, inter alia , on Google Scholar and the ACL Anthology. After initially identifying these papers, the authors classified them through reading and continued to track related but overlooked papers using their citations. We also used Google Scholar to follow up on the latest papers citing these to avoid omissions.

 
 
 For the quantitative analysis and comparison section ( § A.1 ), we did not conduct computational experiments but simply organized the result reported in other literature as is.

 
 
 

## References

 
 
 Aggarwal et al. (2021) 
 
Shourya Aggarwal, Divyanshu Mandowara, Vishwajeet Agrawal, Dinesh Khandelwal, Parag Singla, and Dinesh Garg. 2021.

 
 Explanations for CommonsenseQA: New Dataset and Models .

 
 In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , pages 3050–3065, Online. Association for Computational Linguistics.

 

 
 Agrawal et al. (2023) 
 
Ayush Agrawal, Lester Mackey, and Adam Tauman Kalai. 2023.

 
 Do language models know when they’re hallucinating references? 

 
 ArXiv preprint , abs/2305.18248.

 

 
 Akhtar et al. (2023) 
 
Mubashara Akhtar, Michael Schlichtkrull, Zhijiang Guo, Oana Cocarascu, Elena Simperl, and Andreas Vlachos. 2023.

 
 Multimodal automated fact-checking: A survey .

 
 In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023 , pages 5430–5448. Association for Computational Linguistics.

 

 
 Akyürek et al. (2022) 
 
Ekin Akyürek, Tolga Bolukbasi, Frederick Liu, Binbin Xiong, Ian Tenney, Jacob Andreas, and Kelvin Guu. 2022.

 
 Towards tracing knowledge in language models back to the training data.

 
 In Findings of the Association for Computational Linguistics: EMNLP 2022 , pages 2429–2446.

 

 
 Alayrac et al. (2022) 
 
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022.

 
 Flamingo: a visual language model for few-shot learning.

 
 Advances in neural information processing systems , 35:23716–23736.

 

 
 Ayoobi et al. (2023) 
 
Navid Ayoobi, Sadat Shahriar, and Arjun Mukherjee. 2023.

 
 The looming threat of fake and llm-generated linkedin profiles: Challenges and opportunities for detection and prevention.

 
 In Proceedings of the 34th ACM Conference on Hypertext and Social Media , pages 1–10.

 

 
 Barrett et al. (2023) 
 
Clark Barrett, Brad Boyd, Elie Bursztein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, et al. 2023.

 
 Identifying and mitigating the security risks of generative ai.

 
 Foundations and Trends® in Privacy and Security , 6(1):1–52.

 

 
 Bartsch et al. (2023) 
 
Henning Bartsch, Ole Jorgensen, Domenic Rosati, Jason Hoelscher-Obermaier, and Jacob Pfau. 2023.

 
 Self-consistency of large language models under ambiguity .

 
 ArXiv preprint , abs/2310.13439.

 

 
 Bender et al. (2021) 
 
Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021.

 
 On the dangers of stochastic parrots: Can language models be too big?

 
 In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency , pages 610–623.

 

 
 Bengio et al. (2023) 
 
Yoshua Bengio, Geoffrey Hinton, Andrew Yao, Dawn Song, Pieter Abbeel, Yuval Noah Harari, Ya-Qin Zhang, Lan Xue, Shai Shalev-Shwartz, Gillian Hadfield, et al. 2023.

 
 Managing ai risks in an era of rapid progress .

 
 ArXiv preprint , abs/2310.17688.

 

 
 Bjerva et al. (2020) 
 
Johannes Bjerva, Nikita Bhutani, Behzad Golshan, Wang-Chiew Tan, and Isabelle Augenstein. 2020.

 
 SubjQA: A Dataset for Subjectivity and Review Comprehension .

 
 In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 5480–5494, Online. Association for Computational Linguistics.

 

 
 Borgeaud et al. (2022) 
 
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022.

 
 Improving language models by retrieving from trillions of tokens.

 
 In International conference on machine learning , pages 2206–2240. PMLR.

 

 
 Borsos et al. (2023) 
 
Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, et al. 2023.

 
 Audiolm: a language modeling approach to audio generation.

 
 IEEE/ACM Transactions on Audio, Speech, and Language Processing .

 

 
 Brown et al. (2020) 
 
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.

 
 Language models are few-shot learners .

 
 In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual .

 

 
 Caines et al. (2023) 
 
Andrew Caines, Luca Benedetto, Shiva Taslimipoor, Christopher Davis, Yuan Gao, Oeistein Andersen, Zheng Yuan, Mark Elliott, Russell Moore, Christopher Bryant, et al. 2023.

 
 On the application of large language models for language teaching and assessment technology .

 
 ArXiv preprint , abs/2307.08393.

 

 
 Carlini et al. (2023) 
 
Nicholas Carlini, Matthew Jagielski, Christopher A Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, and Florian Tramèr. 2023.

 
 Poisoning web-scale training datasets is practical .

 
 ArXiv preprint , abs/2302.10149.

 

 
 Chang and Bergen (2023) 
 
Tyler A Chang and Benjamin K Bergen. 2023.

 
 Language model behavior: A comprehensive survey .

 
 ArXiv preprint , abs/2303.11504.

 

 
 Cheang et al. (2023) 
 
Chi Cheang, Hou Chan, Derek Wong, Xuebo Liu, Zhaocong Li, Yanming Sun, Shudong Liu, and Lidia Chao. 2023.

 
 Can lms generalize to future data? an empirical analysis on text summarization.

 
 In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pages 16205–16217.

 

 
 Chen and Shu (2023a) 
 
Canyu Chen and Kai Shu. 2023a.

 
 Can llm-generated misinformation be detected?

 
 In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following .

 

 
 Chen and Shu (2023b) 
 
Canyu Chen and Kai Shu. 2023b.

 
 Combating misinformation in the age of llms: Opportunities and challenges .

 
 ArXiv preprint , abs/2311.05656.

 

 
 Chen et al. (2022) 
 
Hung-Ting Chen, Michael JQ Zhang, and Eunsol Choi. 2022.

 
 Rich knowledge sources bring complex knowledge conflicts: Recalibrating models to reflect conflicting evidence .

 
 ArXiv preprint , abs/2210.13701.

 

 
 Chen et al. (2023a) 
 
Jiangjie Chen, Wei Shi, Ziquan Fu, Sijie Cheng, Lei Li, and Yanghua Xiao. 2023a.

 
 Say what you mean! large language models speak too positively about negative commonsense knowledge .

 
 ArXiv preprint , abs/2305.05976.

 

 
 Chen et al. (2023b) 
 
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2023b.

 
 Benchmarking large language models in retrieval-augmented generation .

 
 ArXiv preprint , abs/2309.01431.

 

 
 Chen et al. (2023c) 
 
Liang Chen, Yang Deng, Yatao Bian, Zeyu Qin, Bingzhe Wu, Tat-Seng Chua, and Kam-Fai Wong. 2023c.

 
 Beyond factuality: A comprehensive evaluation of large language models as knowledge generators.

 
 In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pages 6325–6341.

 

 
 Chen et al. (2021) 
 
Wenhu Chen, Xinyi Wang, and William Yang Wang. 2021.

 
 A dataset for answering time-sensitive questions .

 
 ArXiv preprint , abs/2108.06314.

 

 
 Chern et al. (2023) 
 
I Chern, Steffi Chern, Shiqi Chen, Weizhe Yuan, Kehua Feng, Chunting Zhou, Junxian He, Graham Neubig, Pengfei Liu, et al. 2023.

 
 Factool: Factuality detection in generative ai–a tool augmented framework for multi-task and multi-domain scenarios .

 
 ArXiv preprint , abs/2307.13528.

 

 
 Cheung and Lam (2023) 
 
Tsun-Hin Cheung and Kin-Man Lam. 2023.

 
 Factllama: Optimizing instruction-following language models with external knowledge for automated fact-checking.

 
 In 2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) , pages 846–853. IEEE.

 

 
 Chuang et al. (2023) 
 
Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. 2023.

 
 Dola: Decoding by contrasting layers improves factuality in large language models .

 
 ArXiv preprint , abs/2309.03883.

 

 
 Chughtai et al. (2024) 
 
Bilal Chughtai, Alan Cooney, and Neel Nanda. 2024.

 
 Summing up the facts: Additive mechanisms behind factual recall in llms .

 
 ArXiv preprint , abs/2402.07321.

 

 
 Clark et al. (2019) 
 
Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019.

 
 BoolQ: Exploring the surprising difficulty of natural yes/no questions .

 
 In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , pages 2924–2936, Minneapolis, Minnesota. Association for Computational Linguistics.

 

 
 Cui et al. (2020) 
 
Leyang Cui, Sijie Cheng, Yu Wu, and Yue Zhang. 2020.

 
 Does bert solve commonsense task via commonsense knowledge .

 
 ArXiv preprint , abs/2008.03945.

 

 
 De Cao et al. (2021) 
 
Nicola De Cao, Wilker Aziz, and Ivan Titov. 2021.

 
 Editing factual knowledge in language models.

 
 In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , pages 6491–6506.

 

 
 De Lange et al. (2021) 
 
Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. 2021.

 
 A continual learning survey: Defying forgetting in classification tasks.

 
 IEEE transactions on pattern analysis and machine intelligence , 44(7):3366–3385.

 

 
 Dementieva and Panchenko (2021) 
 
Daryna Dementieva and Alexander Panchenko. 2021.

 
 Cross-lingual evidence improves monolingual fake news detection .

 
 In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Student Research Workshop , pages 310–320, Online. Association for Computational Linguistics.

 

 
 Dhingra et al. (2022) 
 
Bhuwan Dhingra, Jeremy R Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W Cohen. 2022.

 
 Time-aware language models as temporal knowledge bases.

 
 Transactions of the Association for Computational Linguistics , 10:257–273.

 

 
 Dhuliawala et al. (2023) 
 
Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2023.

 
 Chain-of-verification reduces hallucination in large language models .

 
 ArXiv preprint , abs/2309.11495.

 

 
 Dong et al. (2023) 
 
Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Zhifang Sui, and Lei Li. 2023.

 
 Statistical knowledge assessment for large language models.

 
 In Thirty-seventh Conference on Neural Information Processing Systems .

 

 
 Du et al. (2022a) 
 
Li Du, Xiao Ding, Kai Xiong, Ting Liu, and Bing Qin. 2022a.

 
 e-care: a new dataset for exploring explainable causal reasoning.

 
 In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 432–446.

 

 
 Du et al. (2022b) 
 
Yibing Du, Antoine Bosselut, and Christopher D Manning. 2022b.

 
 Synthetic disinformation attacks on automated fact verification systems.

 
 In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 10581–10589.

 

 
 Dziri et al. (2021) 
 
Nouha Dziri, Andrea Madotto, Osmar Zaiane, and Avishek Joey Bose. 2021.

 
 Neural path hunter: Reducing hallucination in dialogue systems via path grounding .

 
 ArXiv preprint , abs/2104.08455.

 

 
 Elazar et al. (2022) 
 
Yanai Elazar, Nora Kassner, Shauli Ravfogel, Amir Feder, Abhilasha Ravichander, Marius Mosbach, Yonatan Belinkov, Hinrich Schütze, and Yoav Goldberg. 2022.

 
 Measuring causal effects of data statistics on language model’sfactual’predictions .

 
 ArXiv preprint , abs/2207.14251.

 

 
 Elazar et al. (2021) 
 
Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, and Yoav Goldberg. 2021.

 
 Measuring and improving consistency in pretrained language models.

 
 Transactions of the Association for Computational Linguistics , 9:1012–1031.

 

 
 Elsahar et al. (2018) 
 
Hady Elsahar, Pavlos Vougiouklis, Arslen Remaci, Christophe Gravier, Jonathon Hare, Frederique Laforest, and Elena Simperl. 2018.

 
 T-REx: A large scale alignment of natural language with knowledge base triples .

 
 In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018) , Miyazaki, Japan. European Language Resources Association (ELRA).

 

 
 Fan et al. (2018) 
 
Angela Fan, Mike Lewis, and Yann Dauphin. 2018.

 
 Hierarchical neural story generation .

 
 In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 889–898, Melbourne, Australia. Association for Computational Linguistics.

 

 
 Feng et al. (2023) 
 
Zhangyin Feng, Weitao Ma, Weijiang Yu, Lei Huang, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2023.

 
 Trends in integration of knowledge and large language models: A survey and taxonomy of methods, benchmarks, and applications .

 
 ArXiv preprint , abs/2311.05876.

 

 
 Ferrara (2023) 
 
Emilio Ferrara. 2023.

 
 Genai against humanity: Nefarious applications of generative artificial intelligence and large language models .

 
 ArXiv preprint , abs/2310.00737.

 

 
 Floridi (2023) 
 
Luciano Floridi. 2023.

 
 Ai as agency without intelligence: on chatgpt, large language models, and other generative models.

 
 Philosophy Technology , 36(1):15.

 

 
 Fung et al. (2022) 
 
Yi R Fung, Kung-Hsiang Huang, Preslav Nakov, and Heng Ji. 2022.

 
 The battlefront of combating misinformation and coping with media bias.

 
 In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , pages 4790–4791.

 

 
 Gao et al. (2023a) 
 
Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. 2023a.

 
 Rarr: Researching and revising what language models say, using language models.

 
 In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 16477–16508.

 

 
 Gao et al. (2023b) 
 
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023b.

 
 Retrieval-augmented generation for large language models: A survey .

 
 ArXiv preprint , abs/2312.10997.

 

 
 Gekhman et al. (2023) 
 
Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, and Idan Szpektor. 2023.

 
 Trueteacher: Learning factual consistency evaluation with large language models .

 
 ArXiv preprint , abs/2305.11171.

 

 
 Geva et al. (2021) 
 
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021.

 
 Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies.

 
 Transactions of the Association for Computational Linguistics , 9:346–361.

 

 
 Goldstein et al. (2023) 
 
Josh A Goldstein, Girish Sastry, Micah Musser, Renee DiResta, Matthew Gentzel, and Katerina Sedova. 2023.

 
 Generative language models and automated influence operations: Emerging threats and potential mitigations .

 
 ArXiv preprint , abs/2301.04246.

 

 
 Greshake et al. (2023) 
 
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. 2023.

 
 More than you’ve asked for: A comprehensive analysis of novel prompt injection threats to application-integrated large language models.

 
 arXiv e-prints , pages arXiv–2302.

 

 
 Grosse et al. (2023) 
 
Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. 2023.

 
 Studying large language model generalization with influence functions .

 
 ArXiv preprint , abs/2308.03296.

 

 
 Guo et al. (2022) 
 
Zhijiang Guo, Michael Schlichtkrull, and Andreas Vlachos. 2022.

 
 A survey on automated fact-checking.

 
 Transactions of the Association for Computational Linguistics , 10:178–206.

 

 
 Guu et al. (2020) 
 
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020.

 
 Retrieval augmented language model pre-training .

 
 In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event , volume 119 of Proceedings of Machine Learning Research , pages 3929–3938. PMLR.

 

 
 Hase et al. (2023) 
 
Peter Hase, Mona Diab, Asli Celikyilmaz, Xian Li, Zornitsa Kozareva, Veselin Stoyanov, Mohit Bansal, and Srinivasan Iyer. 2023.

 
 Methods for measuring, updating, and visualizing factual beliefs in language models.

 
 In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics , pages 2706–2723.

 

 
 He et al. (2021) 
 
Tianxing He, Jun Liu, Kyunghyun Cho, Myle Ott, Bing Liu, James Glass, and Fuchun Peng. 2021.

 
 Analyzing the forgetting problem in pretrain-finetuning of open-domain dialogue response models .

 
 In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume , pages 1121–1133, Online. Association for Computational Linguistics.

 

 
 Hermann et al. (2015) 
 
Karl Moritz Hermann, Tomás Kociský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015.

 
 Teaching machines to read and comprehend .

 
 In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada , pages 1693–1701.

 

 
 Hoffmann et al. (2022) 
 
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022.

 
 Training compute-optimal large language models .

 
 ArXiv preprint , abs/2203.15556.

 

 
 Holtzman et al. (2020) 
 
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020.

 
 The curious case of neural text degeneration .

 
 In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net.

 

 
 Hong et al. (2023) 
 
Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-Hyon Myaeng, and Joyce Jiyoung Whang. 2023.

 
 Discern and answer: Mitigating the impact of misinformation in retrieval-augmented models with discriminators .

 
 ArXiv preprint , abs/2305.01579.

 

 
 Hsu et al. (2021) 
 
Cheng Hsu, Cheng-Te Li, Diego Saez-Trumper, and Yi-Zhan Hsu. 2021.

 
 Wikicontradiction: Detecting self-contradiction articles on wikipedia.

 
 In 2021 IEEE International Conference on Big Data (Big Data) , pages 427–436. IEEE.

 

 
 Hu et al. (2023) 
 
Xuming Hu, Junzhe Chen, Xiaochuan Li, Yufei Guo, Lijie Wen, Philip S. Yu, and Zhijiang Guo. 2023.

 
 Do large language models know about facts? 

 
 CoRR , abs/2310.05177.

 

 
 Hu et al. (2024) 
 
Xuming Hu, Xiaochuan Li, Junzhe Chen, Yinghui Li, Yangning Li, Xiaoguang Li, Yasheng Wang, Qun Liu, Lijie Wen, Philip S. Yu, and Zhijiang Guo. 2024.

 
 Evaluating robustness of generative search engine on adversarial factual questions .

 
 CoRR , abs/2403.12077.

 

 
 Huang et al. (2023) 
 
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2023.

 
 A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions .

 
 ArXiv preprint , abs/2311.05232.

 

 
 Ilharco et al. (2022) 
 
Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2022.

 
 Editing models with task arithmetic .

 
 ArXiv preprint , abs/2212.04089.

 

 
 Jang et al. (2022) 
 
Joel Jang, Seonghyeon Ye, Changho Lee, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun Kim, and Minjoon Seo. 2022.

 
 Temporalwiki: A lifelong benchmark for training and evaluating ever-evolving language models.

 
 In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , pages 6237–6250.

 

 
 Jang et al. (2021) 
 
Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin, Janghoon Han, KIM Gyeonghun, Stanley Jungkyu Choi, and Minjoon Seo. 2021.

 
 Towards continual knowledge learning of language models.

 
 In International Conference on Learning Representations .

 

 
 Jang and Lukasiewicz (2023) 
 
Myeongjun Erik Jang and Thomas Lukasiewicz. 2023.

 
 Improving language models meaning understanding and consistency by learning conceptual roles from dictionary .

 
 ArXiv preprint , abs/2310.15541.

 

 
 Jawahar et al. (2020) 
 
Ganesh Jawahar, Muhammad Abdul-Mageed, and Laks Lakshmanan, V.S. 2020.

 
 Automatic detection of machine generated text: A critical survey .

 
 In Proceedings of the 28th International Conference on Computational Linguistics , pages 2296–2309, Barcelona, Spain (Online). International Committee on Computational Linguistics.

 

 
 Jawahar et al. (2019) 
 
Ganesh Jawahar, Benoît Sagot, and Djamé Seddah. 2019.

 
 What does BERT learn about the structure of language? 

 
 In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , pages 3651–3657, Florence, Italy. Association for Computational Linguistics.

 

 
 Ji et al. (2023) 
 
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023.

 
 Survey of hallucination in natural language generation.

 
 ACM Computing Surveys , 55(12):1–38.

 

 
 Jiang et al. (2023) 
 
Bohan Jiang, Zhen Tan, Ayushi Nirmal, and Huan Liu. 2023.

 
 Disinformation detection: An evolving challenge in the age of llms .

 
 ArXiv preprint , abs/2309.15847.

 

 
 Jin et al. (2024a) 
 
Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Qiuxia Li, and Jun Zhao. 2024a.

 
 Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrieval-augmented language models .

 
 ArXiv preprint , abs/2402.14409.

 

 
 Jin et al. (2024b) 
 
Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. 2024b.

 
 Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models .

 
 ArXiv preprint , abs/2402.18154.

 

 
 Joshi et al. (2017) 
 
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017.

 
 TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension .

 
 In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

 

 
 Ju et al. (2022) 
 
Chen Ju, Tengda Han, Kunhao Zheng, Ya Zhang, and Weidi Xie. 2022.

 
 Prompting visual-language models for efficient video understanding.

 
 In European Conference on Computer Vision , pages 105–124. Springer.

 

 
 Kaddour et al. (2023) 
 
Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. 2023.

 
 Challenges and applications of large language models .

 
 ArXiv preprint , abs/2307.10169.

 

 
 Kandpal et al. (2023) 
 
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. 2023.

 
 Large language models struggle to learn long-tail knowledge.

 
 In International Conference on Machine Learning , pages 15696–15707. PMLR.

 

 
 Kang and Choi (2023) 
 
Cheongwoong Kang and Jaesik Choi. 2023.

 
 Impact of co-occurrence on factual knowledge of large language models .

 
 ArXiv preprint , abs/2310.08256.

 

 
 Kaplan et al. (2020) 
 
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.

 
 Scaling laws for neural language models .

 
 CoRR , abs/2001.08361.

 

 
 Karpukhin et al. (2020) 
 
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.

 
 Dense passage retrieval for open-domain question answering .

 
 In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 6769–6781, Online. Association for Computational Linguistics.

 

 
 Kasai et al. (2022) 
 
Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A Smith, Yejin Choi, and Kentaro Inui. 2022.

 
 Realtime qa: What’s the answer right now? 

 
 ArXiv preprint , abs/2207.13332.

 

 
 Kidd and Birhane (2023) 
 
Celeste Kidd and Abeba Birhane. 2023.

 
 How ai can distort human beliefs.

 
 Science , 380(6651):1222–1223.

 

 
 Ko et al. (2022) 
 
Miyoung Ko, Ingyu Seong, Hwaran Lee, Joonsuk Park, Minsuk Chang, and Minjoon Seo. 2022.

 
 Claimdiff: Comparing and contrasting claims on contentious issues .

 
 ArXiv preprint , abs/2205.12221.

 

 
 Kočiský et al. (2018) 
 
Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018.

 
 The NarrativeQA reading comprehension challenge .

 
 Transactions of the Association for Computational Linguistics , 6:317–328.

 

 
 Kumar and Shah (2018) 
 
Srijan Kumar and Neil Shah. 2018.

 
 False information on web and social media: A survey .

 
 ArXiv preprint , abs/1804.08559.

 

 
 Kwiatkowski et al. (2019) 
 
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019.

 
 Natural questions: A benchmark for question answering research .

 
 Transactions of the Association for Computational Linguistics , 7:452–466.

 

 
 Lam et al. (2022) 
 
Tsz Kin Lam, Eva Hasler, and Felix Hieber. 2022.

 
 Analyzing the use of influence functions for instance-specific data filtering in neural machine translation .

 
 ArXiv preprint , abs/2210.13281.

 

 
 Lazaridou et al. (2022) 
 
Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. 2022.

 
 Internet-augmented language models through few-shot prompting for open-domain question answering .

 
 ArXiv preprint , abs/2203.05115.

 

 
 Lazaridou et al. (2021) 
 
Angeliki Lazaridou, Adhi Kuncoro, Elena Gribovskaya, Devang Agrawal, Adam Liska, Tayfun Terzi, Mai Gimenez, Cyprien de Masson d’Autume, Tomas Kocisky, Sebastian Ruder, et al. 2021.

 
 Mind the gap: Assessing temporal generalization in neural language models.

 
 Advances in Neural Information Processing Systems , 34:29348–29363.

 

 
 Lee et al. (2022a) 
 
Kyungjae Lee, Wookje Han, Seung-won Hwang, Hwaran Lee, Joonsuk Park, and Sang-Woo Lee. 2022a.

 
 Plug-and-play adaptation for continuously-updated qa.

 
 In Findings of the Association for Computational Linguistics: ACL 2022 , pages 438–447.

 

 
 Lee et al. (2022b) 
 
Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale N Fung, Mohammad Shoeybi, and Bryan Catanzaro. 2022b.

 
 Factuality enhanced language models for open-ended text generation.

 
 Advances in Neural Information Processing Systems , 35:34586–34599.

 

 
 Leite et al. (2023) 
 
João A Leite, Olesya Razuvayevskaya, Kalina Bontcheva, and Carolina Scarton. 2023.

 
 Detecting misinformation with llm-predicted credibility signals and weak supervision .

 
 ArXiv preprint , abs/2309.07601.

 

 
 Lewis et al. (2020) 
 
Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020.

 
 Retrieval-augmented generation for knowledge-intensive NLP tasks .

 
 In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual .

 

 
 Li et al. (2022a) 
 
Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit, Felix Yu, and Sanjiv Kumar. 2022a.

 
 Large language models with controllable working memory .

 
 ArXiv preprint , abs/2211.05110.

 

 
 Li et al. (2023a) 
 
Jierui Li, Vipul Raheja, and Dhruv Kumar. 2023a.

 
 Contradoc: Understanding self-contradictions in documents with large language models .

 
 ArXiv preprint , abs/2311.09182.

 

 
 Li et al. (2023b) 
 
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023b.

 
 Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.

 
 In International conference on machine learning , pages 19730–19742. PMLR.

 

 
 Li et al. (2023c) 
 
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023c.

 
 Inference-time intervention: Eliciting truthful answers from a language model .

 
 ArXiv preprint , abs/2306.03341.

 

 
 Li et al. (2022b) 
 
Shaobo Li, Xiaoguang Li, Lifeng Shang, Zhenhua Dong, Chengjie Sun, Bingquan Liu, Zhenzhou Ji, Xin Jiang, and Qun Liu. 2022b.

 
 How pre-trained language models capture factual knowledge? a causal-inspired analysis .

 
 ArXiv preprint , abs/2203.16747.

 

 
 Li et al. (2022c) 
 
Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2022c.

 
 Contrastive decoding: Open-ended text generation as optimization .

 
 ArXiv preprint , abs/2210.15097.

 

 
 Li et al. (2023d) 
 
Xiang Lisa Li, Vaishnavi Shrivastava, Siyan Li, Tatsunori Hashimoto, and Percy Liang. 2023d.

 
 Benchmarking and improving generator-validator consistency of language models .

 
 ArXiv preprint , abs/2310.01846.

 

 
 Li et al. (2023e) 
 
Yinheng Li, Shaofei Wang, Han Ding, and Hang Chen. 2023e.

 
 Large language models in finance: A survey.

 
 In Proceedings of the Fourth ACM International Conference on AI in Finance , pages 374–382.

 

 
 Li et al. (2023f) 
 
Zhoubo Li, Ningyu Zhang, Yunzhi Yao, Mengru Wang, Xi Chen, and Huajun Chen. 2023f.

 
 Unveiling the pitfalls of knowledge editing for large language models .

 
 ArXiv preprint , abs/2310.02129.

 

 
 Lin et al. (2022) 
 
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.

 
 Truthfulqa: Measuring how models mimic human falsehoods.

 
 In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 3214–3252.

 

 
 Liska et al. (2022) 
 
Adam Liska, Tomas Kocisky, Elena Gribovskaya, Tayfun Terzi, Eren Sezener, Devang Agrawal, D’Autume Cyprien De Masson, Tim Scholtes, Manzil Zaheer, Susannah Young, et al. 2022.

 
 Streamingqa: A benchmark for adaptation to new knowledge over time in question answering models.

 
 In International Conference on Machine Learning , pages 13604–13622. PMLR.

 

 
 Liu et al. (2023a) 
 
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023a.

 
 Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing.

 
 ACM Computing Surveys , 55(9):1–35.

 

 
 Liu et al. (2023b) 
 
Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, and Yang Liu. 2023b.

 
 Prompt injection attack against llm-integrated applications .

 
 ArXiv preprint , abs/2306.05499.

 

 
 Longpre et al. (2021) 
 
Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh. 2021.

 
 Entity-based knowledge conflicts in question answering.

 
 In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , pages 7052–7063.

 

 
 Luu et al. (2021) 
 
Kelvin Luu, Daniel Khashabi, Suchin Gururangan, Karishma Mandyam, and Noah A Smith. 2021.

 
 Time waits for no one! analysis and challenges of temporal misalignment .

 
 ArXiv preprint , abs/2111.07408.

 

 
 Mallen et al. (2022) 
 
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi. 2022.

 
 When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories .

 
 ArXiv preprint , abs/2212.10511.

 

 
 Mallen et al. (2023) 
 
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023.

 
 When not to trust language models: Investigating effectiveness of parametric and non-parametric memories.

 
 In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 9802–9822.

 

 
 Manakul et al. (2023) 
 
Potsawee Manakul, Adian Liusie, and Mark JF Gales. 2023.

 
 Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models .

 
 ArXiv preprint , abs/2303.08896.

 

 
 Margatina et al. (2023) 
 
Katerina Margatina, Shuai Wang, Yogarshi Vyas, Neha Anna John, Yassine Benajiba, and Miguel Ballesteros. 2023.

 
 Dynamic benchmarking of masked language models on temporal concept drift with multiple views .

 
 ArXiv preprint , abs/2302.12297.

 

 
 Martin et al. (2024) 
 
Lauren Martin, Nick Whitehouse, Stephanie Yiu, Lizzie Catterson, and Rivindu Perera. 2024.

 
 Better call gpt, comparing large language models against lawyers .

 
 ArXiv preprint , abs/2401.16212.

 

 
 Massarelli et al. (2020) 
 
Luca Massarelli, Fabio Petroni, Aleksandra Piktus, Myle Ott, Tim Rocktäschel, Vassilis Plachouras, Fabrizio Silvestri, and Sebastian Riedel. 2020.

 
 How decoding strategies affect the verifiability of generated text .

 
 In Findings of the Association for Computational Linguistics: EMNLP 2020 , pages 223–235, Online. Association for Computational Linguistics.

 

 
 Meel and Vishwakarma (2020) 
 
Priyanka Meel and Dinesh Kumar Vishwakarma. 2020.

 
 Fake news, rumor, information pollution in social media and web: A contemporary survey of state-of-the-arts, challenges and opportunities.

 
 Expert Systems with Applications , 153:112986.

 

 
 Menczer et al. (2023) 
 
Filippo Menczer, David Crandall, Yong-Yeol Ahn, and Apu Kapadia. 2023.

 
 Addressing the harms of ai-generated inauthentic content.

 
 Nature Machine Intelligence , 5(7):679–680.

 

 
 Meng et al. (2022) 
 
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022.

 
 Locating and editing factual associations in gpt.

 
 Advances in Neural Information Processing Systems , 35:17359–17372.

 

 
 Merity et al. (2017) 
 
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017.

 
 Pointer sentinel mixture models .

 
 In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings . OpenReview.net.

 

 
 Milano et al. (2023) 
 
Silvia Milano, Joshua A McGrane, and Sabina Leonelli. 2023.

 
 Large language models challenge the future of higher education.

 
 Nature Machine Intelligence , 5(4):333–334.

 

 
 Mitchell et al. (2021) 
 
Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. 2021.

 
 Fast model editing at scale .

 
 ArXiv preprint , abs/2110.11309.

 

 
 Mitchell et al. (2022) 
 
Eric Mitchell, Joseph J Noh, Siyan Li, William S Armstrong, Ananth Agarwal, Patrick Liu, Chelsea Finn, and Christopher D Manning. 2022.

 
 Enhancing self-consistency and performance of pre-trained language models through natural language inference .

 
 ArXiv preprint , abs/2211.11875.

 

 
 Mündler et al. (2023) 
 
Niels Mündler, Jingxuan He, Slobodan Jenko, and Martin Vechev. 2023.

 
 Self-contradictory hallucinations of large language models: Evaluation, detection and mitigation .

 
 ArXiv preprint , abs/2305.15852.

 

 
 Naveed et al. (2023) 
 
Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Nick Barnes, and Ajmal Mian. 2023.

 
 A comprehensive overview of large language models .

 
 ArXiv preprint , abs/2307.06435.

 

 
 Neeman et al. (2022) 
 
Ella Neeman, Roee Aharoni, Or Honovich, Leshem Choshen, Idan Szpektor, and Omri Abend. 2022.

 
 Disentqa: Disentangling parametric and contextual knowledge with counterfactual question answering .

 
 ArXiv preprint , abs/2211.05655.

 

 
 Nickerson (1998) 
 
Raymond S Nickerson. 1998.

 
 Confirmation bias: A ubiquitous phenomenon in many guises.

 
 Review of general psychology , 2(2):175–220.

 

 
 Ohmer et al. (2023) 
 
Xenia Ohmer, Elia Bruni, and Dieuwke Hupkes. 2023.

 
 Separating form and meaning: Using self-consistency to quantify task understanding across multiple senses.

 
 CoRR .

 

 
 Onoe et al. (2023) 
 
Yasumasa Onoe, Michael JQ Zhang, Shankar Padmanabhan, Greg Durrett, and Eunsol Choi. 2023.

 
 Can lms learn new entities from descriptions? challenges in propagating injected knowledge .

 
 ArXiv preprint , abs/2305.01651.

 

 
 OpenAI (2023) 
 
OpenAI. 2023.

 
 Chatgpt .

 

 
 OpenAI (2024) 
 
OpenAI. 2024.

 
 Gpt-4 technical report .

 

 
 Ouyang et al. (2022) 
 
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022.

 
 Training language models to follow instructions with human feedback.

 
 Advances in Neural Information Processing Systems , 35:27730–27744.

 

 
 Pan et al. (2023a) 
 
Liangming Pan, Wenhu Chen, Min-Yen Kan, and William Yang Wang. 2023a.

 
 Attacking open-domain question answering by injecting misinformation.

 
 IJCNLP-AACL. ACL .

 

 
 Pan et al. (2022) 
 
Xiaoman Pan, Wenlin Yao, Hongming Zhang, Dian Yu, Dong Yu, and Jianshu Chen. 2022.

 
 Knowledge-in-context: Towards knowledgeable semi-parametric language models.

 
 In The Eleventh International Conference on Learning Representations .

 

 
 Pan et al. (2023b) 
 
Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and William Yang Wang. 2023b.

 
 On the risk of misinformation pollution with large language models .

 
 ArXiv preprint , abs/2305.13661.

 

 
 Peng et al. (2023) 
 
Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, et al. 2023.

 
 Check your facts and try again: Improving large language models with external knowledge and automated feedback .

 
 ArXiv preprint , abs/2302.12813.

 

 
 Perez et al. (2022) 
 
Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. 2022.

 
 Discovering language model behaviors with model-written evaluations .

 
 ArXiv preprint , abs/2212.09251.

 

 
 Petroni et al. (2019) 
 
Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019.

 
 Language models as knowledge bases? 

 
 In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages 2463–2473, Hong Kong, China. Association for Computational Linguistics.

 

 
 Pielka et al. (2022) 
 
Maren Pielka, Felix Rode, Lisa Pucknat, Tobias Deußer, and Rafet Sifa. 2022.

 
 A linguistic investigation of machine learning based contradiction detection models: an empirical analysis and future perspectives.

 
 In 2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA) , pages 1649–1653. IEEE.

 

 
 Pinter and Elhadad (2023) 
 
Yuval Pinter and Michael Elhadad. 2023.

 
 Emptying the ocean with a spoon: Should we edit models?

 
 In Findings of the Association for Computational Linguistics: EMNLP 2023 , pages 15164–15172.

 

 
 Qi et al. (2023) 
 
Jirui Qi, Raquel Fernández, and Arianna Bisazza. 2023.

 
 Cross-lingual consistency of factual knowledge in multilingual language models .

 
 ArXiv preprint , abs/2310.10378.

 

 
 Qian et al. (2023) 
 
Cheng Qian, Xinran Zhao, and Sherry Tongshuang Wu. 2023.

 
 " merge conflicts!" exploring the impacts of external distractors to parametric knowledge graphs .

 
 ArXiv preprint , abs/2309.08594.

 

 
 Rabinovich et al. (2023) 
 
Ella Rabinovich, Samuel Ackerman, Orna Raz, Eitan Farchi, and Ateret Anaby-Tavor. 2023.

 
 Predicting question-answering performance of large language models through semantic consistency .

 
 ArXiv preprint , abs/2311.01152.

 

 
 Raj et al. (2023) 
 
Harsh Raj, Vipul Gupta, Domenic Rosati, and Subhabrata Majumdar. 2023.

 
 Semantic consistency for assuring reliability of large language models .

 
 ArXiv preprint , abs/2308.09138.

 

 
 Raj et al. (2022) 
 
Harsh Raj, Domenic Rosati, and Subhabrata Majumdar. 2022.

 
 Measuring reliability of large language models through semantic consistency .

 
 ArXiv preprint , abs/2211.05853.

 

 
 Rajpurkar et al. (2018) 
 
Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018.

 
 Know what you don’t know: Unanswerable questions for SQuAD .

 
 In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) , pages 784–789, Melbourne, Australia. Association for Computational Linguistics.

 

 
 Rajpurkar et al. (2016) 
 
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.

 
 SQuAD: 100,000+ questions for machine comprehension of text .

 
 In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing , pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

 

 
 Roberts et al. (2020) 
 
Adam Roberts, Colin Raffel, and Noam Shazeer. 2020.

 
 How much knowledge can you pack into the parameters of a language model? 

 
 In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 5418–5426, Online. Association for Computational Linguistics.

 

 
 Rogers et al. (2020) 
 
Anna Rogers, Olga Kovaleva, and Anna Rumshisky. 2020.

 
 A primer in BERTology: What we know about how BERT works .

 
 Transactions of the Association for Computational Linguistics , 8:842–866.

 

 
 Schick et al. (2023) 
 
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023.

 
 Toolformer: Language models can teach themselves to use tools .

 
 ArXiv preprint , abs/2302.04761.

 

 
 Schlichtkrull et al. (2023) 
 
Michael Schlichtkrull, Zhijiang Guo, and Andreas Vlachos. 2023.

 
 Averitec: A dataset for real-world claim verification with evidence from the web .

 
 In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 .

 

 
 Sharir et al. (2020) 
 
Or Sharir, Barak Peleg, and Yoav Shoham. 2020.

 
 The cost of training nlp models: A concise overview .

 
 ArXiv preprint , abs/2004.08900.

 

 
 Sharma et al. (2023) 
 
Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, et al. 2023.

 
 Towards understanding sycophancy in language models .

 
 ArXiv preprint , abs/2310.13548.

 

 
 Shi et al. (2023a) 
 
Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Scott Wen-tau Yih. 2023a.

 
 Trusting your evidence: Hallucinate less with context-aware decoding .

 
 ArXiv preprint , abs/2305.14739.

 

 
 Shi et al. (2023b) 
 
Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Victoria Lin, Noah A Smith, Luke Zettlemoyer, Scott Yih, and Mike Lewis. 2023b.

 
 In-context pretraining: Language modeling beyond document boundaries .

 
 ArXiv preprint , abs/2310.10638.

 

 
 Shi et al. (2023c) 
 
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023c.

 
 Replug: Retrieval-augmented black-box language models .

 
 ArXiv preprint , abs/2301.12652.

 

 
 Shu et al. (2017) 
 
Kai Shu, Amy Sliva, Suhang Wang, Jiliang Tang, and Huan Liu. 2017.

 
 Fake news detection on social media: A data mining perspective.

 
 ACM SIGKDD explorations newsletter , 19(1):22–36.

 

 
 Shui et al. (2023) 
 
Ruihao Shui, Yixin Cao, Xiang Wang, and Tat-Seng Chua. 2023.

 
 A comprehensive evaluation of large language models on legal judgment prediction .

 
 ArXiv preprint , abs/2310.11761.

 

 
 Shuster et al. (2021) 
 
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021.

 
 Retrieval augmentation reduces hallucination in conversation.

 
 In Findings of the Association for Computational Linguistics: EMNLP 2021 , pages 3784–3803.

 

 
 Singhal et al. (2022) 
 
Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2022.

 
 Large language models encode clinical knowledge .

 
 ArXiv preprint , abs/2212.13138.

 

 
 Sinitsin et al. (2020) 
 
Anton Sinitsin, Vsevolod Plokhotnyuk, Dmitriy Pyrkin, Sergei Popov, and Artem Babenko. 2020.

 
 Editable neural networks .

 
 In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net.

 

 
 Smith (2023) 
 
Craig S. Smith. 2023.

 
 What large models cost you – there is no free ai lunch .

 

 
 Solaiman et al. (2023) 
 
Irene Solaiman, Zeerak Talat, William Agnew, Lama Ahmad, Dylan Baker, Su Lin Blodgett, Hal Daumé III, Jesse Dodge, Ellie Evans, Sara Hooker, et al. 2023.

 
 Evaluating the social impact of generative ai systems in systems and society .

 
 ArXiv preprint , abs/2306.05949.

 

 
 Spitale et al. (2023) 
 
Giovanni Spitale, Nikola Biller-Andorno, and Federico Germani. 2023.

 
 Ai model gpt-3 (dis) informs us better than humans .

 
 ArXiv preprint , abs/2301.11924.

 

 
 Tan et al. (2024) 
 
Hexiang Tan, Fei Sun, Wanli Yang, Yuanzhuo Wang, Qi Cao, and Xueqi Cheng. 2024.

 
 Blinded by generated contexts: How language models merge generated and retrieved contexts for open-domain qa? 

 
 ArXiv preprint , abs/2401.11911.

 

 
 Tang et al. (2023) 
 
Ruixiang Tang, Yu-Neng Chuang, and Xia Hu. 2023.

 
 The science of detecting llm-generated texts .

 
 ArXiv preprint , abs/2303.07205.

 

 
 Tenney et al. (2019) 
 
Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019.

 
 BERT rediscovers the classical NLP pipeline .

 
 In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , pages 4593–4601, Florence, Italy. Association for Computational Linguistics.

 

 
 Thirunavukarasu et al. (2023) 
 
Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. 2023.

 
 Large language models in medicine.

 
 Nature medicine , 29(8):1930–1940.

 

 
 Touvron et al. (2023) 
 
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023.

 
 Llama 2: Open foundation and fine-tuned chat models .

 
 ArXiv preprint , abs/2307.09288.

 

 
 Trivedi et al. (2022) 
 
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022.

 
 Musique: Multihop questions via single-hop question composition.

 
 Transactions of the Association for Computational Linguistics , 10:539–554.

 

 
 Turpin et al. (2023) 
 
Miles Turpin, Julian Michael, Ethan Perez, and Samuel R Bowman. 2023.

 
 Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting .

 
 ArXiv preprint , abs/2305.04388.

 

 
 Vergho et al. (2024) 
 
Tyler Vergho, Jean-Francois Godbout, Reihaneh Rabbany, and Kellin Pelrine. 2024.

 
 Comparing gpt-4 and open-source language models in misinformation mitigation .

 
 ArXiv preprint , abs/2401.06920.

 

 
 Vu et al. (2023) 
 
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, et al. 2023.

 
 Freshllms: Refreshing large language models with search engine augmentation .

 
 ArXiv preprint , abs/2310.03214.

 

 
 Wan et al. (2024) 
 
Alexander Wan, Eric Wallace, and Dan Klein. 2024.

 
 What evidence do language models find convincing? 

 
 ArXiv preprint , abs/2402.11782.

 

 
 Wang et al. (2019) 
 
Cunxiang Wang, Shuailong Liang, Yue Zhang, Xiaonan Li, and Tian Gao. 2019.

 
 Does it make sense? and why? a pilot study for sense making and explanation .

 
 In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , pages 4020–4026, Florence, Italy. Association for Computational Linguistics.

 

 
 Wang et al. (2023a) 
 
Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao, Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang, Linyi Yang, Jindong Wang, Xing Xie, Zheng Zhang, and Yue Zhang. 2023a.

 
 Survey on factuality in large language models: Knowledge, retrieval and domain-specificity .

 

 
 Wang et al. (2023b) 
 
Cunxiang Wang, Zhikun Xu, Qipeng Guo, Xiangkun Hu, Xuefeng Bai, Zheng Zhang, and Yue Zhang. 2023b.

 
 Exploiting Abstract Meaning Representation for open-domain question answering .

 
 In Findings of the Association for Computational Linguistics: ACL 2023 , pages 2083–2096, Toronto, Canada. Association for Computational Linguistics.

 

 
 Wang et al. (2023c) 
 
Cunxiang Wang, Haofei Yu, and Yue Zhang. 2023c.

 
 RFiD: Towards rational fusion-in-decoder for open-domain question answering .

 
 In Findings of the Association for Computational Linguistics: ACL 2023 , pages 2473–2481, Toronto, Canada. Association for Computational Linguistics.

 

 
 Wang et al. (2023d) 
 
Fei Wang, Wenjie Mo, Yiwei Wang, Wenxuan Zhou, and Muhao Chen. 2023d.

 
 A causal view of entity bias in (large) language models .

 
 ArXiv preprint , abs/2305.14695.

 

 
 Wang et al. (2024a) 
 
Hongru Wang, Wenyu Huang, Yang Deng, Rui Wang, Zezhong Wang, Yufei Wang, Fei Mi, Jeff Z. Pan, and Kam-Fai Wong. 2024a.

 
 Unims-rag: A unified multi-source retrieval-augmented generation for personalized dialogue systems .

 

 
 Wang et al. (2023e) 
 
Hongru Wang, Lingzhi Wang, Yiming Du, Liang Chen, Jingyan Zhou, Yufei Wang, and Kam-Fai Wong. 2023e.

 
 A survey of the evolution of language model-based dialogue systems .

 

 
 Wang et al. (2024b) 
 
Hongru Wang, Boyang Xue, Baohang Zhou, Tianhua Zhang, Cunxiang Wang, Guanhua Chen, Huimin Wang, and Kam fai Wong. 2024b.

 
 Self-dc: When to retrieve and when to generate? self divide-and-conquer for compositional unknown questions .

 

 
 Wang et al. (2023f) 
 
Jiaan Wang, Yunlong Liang, Zengkui Sun, Yuxuan Cao, and Jiarong Xu. 2023f.

 
 Cross-lingual knowledge editing in large language models .

 
 ArXiv preprint , abs/2309.08952.

 

 
 Wang et al. (2023g) 
 
Liyuan Wang, Xingxing Zhang, Qian Li, Mingtian Zhang, Hang Su, Jun Zhu, and Yi Zhong. 2023g.

 
 Incorporating neuro-inspired adaptability for continual learning in artificial intelligence.

 
 Nature Machine Intelligence , pages 1–13.

 

 
 Wang et al. (2023h) 
 
Yike Wang, Shangbin Feng, Heng Wang, Weijia Shi, Vidhisha Balachandran, Tianxing He, and Yulia Tsvetkov. 2023h.

 
 Resolving knowledge conflicts in large language models .

 
 ArXiv preprint , abs/2310.00935.

 

 
 Wei et al. (2023) 
 
Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V Le. 2023.

 
 Simple synthetic data reduces sycophancy in large language models .

 
 ArXiv preprint , abs/2308.03958.

 

 
 Weidinger et al. (2021) 
 
Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, et al. 2021.

 
 Ethical and social risks of harm from language models .

 
 ArXiv preprint , abs/2112.04359.

 

 
 Weidinger et al. (2023) 
 
Laura Weidinger, Maribeth Rauh, Nahema Marchal, Arianna Manzini, Lisa Anne Hendricks, Juan Mateos-Garcia, Stevie Bergman, Jackie Kay, Conor Griffin, Ben Bariach, et al. 2023.

 
 Sociotechnical safety evaluation of generative ai systems .

 
 ArXiv preprint , abs/2310.11986.

 

 
 Weller et al. (2022) 
 
Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, and Benjamin Van Durme. 2022.

 
 Defending against misinformation attacks in open-domain question answering .

 
 ArXiv preprint , abs/2212.10002.

 

 
 Williams et al. (2018) 
 
Adina Williams, Nikita Nangia, and Samuel Bowman. 2018.

 
 A broad-coverage challenge corpus for sentence understanding through inference .

 
 In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) , pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics.

 

 
 Wu et al. (2022) 
 
Xiangcheng Wu, Xi Niu, and Ruhani Rahman. 2022.

 
 Topological analysis of contradictions in text.

 
 In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval , pages 2478–2483.

 

 
 Wu et al. (2023) 
 
Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. 2023.

 
 Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation.

 
 In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1–5. IEEE.

 

 
 Xie et al. (2023) 
 
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. 2023.

 
 Adaptive chameleon or stubborn sloth: Unraveling the behavior of large language models in knowledge conflicts .

 
 ArXiv preprint , abs/2305.13300.

 

 
 Xu et al. (2022) 
 
Nan Xu, Fei Wang, Bangzheng Li, Mingtao Dong, and Muhao Chen. 2022.

 
 Does your model classify entities reasonably? diagnosing and mitigating spurious correlations in entity typing .

 
 ArXiv preprint , abs/2205.12640.

 

 
 Xu et al. (2023) 
 
Rongwu Xu, Brian S Lin, Shujian Yang, Tianqi Zhang, Weiyan Shi, Tianwei Zhang, Zhixuan Fang, Wei Xu, and Han Qiu. 2023.

 
 The earth is flat because…: Investigating llms’ belief towards misinformation via persuasive conversation .

 
 ArXiv preprint , abs/2312.09085.

 

 
 Xue et al. (2024) 
 
Boyang Xue, Hongru Wang, Weichao Wang, Rui Wang, Sheng Wang, Zeming Liu, and Kam-Fai Wong. 2024.

 
 A comprehensive study of multilingual confidence estimation on large language models .

 

 
 Xue et al. (2023) 
 
Boyang Xue, Weichao Wang, Hongru Wang, Fei Mi, Rui Wang, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu, and Kam-Fai Wong. 2023.

 
 Improving factual consistency for knowledge-grounded dialogue systems via knowledge enhancement and alignment.

 
 In Findings of the Association for Computational Linguistics: EMNLP 2023 , pages 7829–7844.

 

 
 Yao et al. (2023) 
 
Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. 2023.

 
 Editing large language models: Problems, methods, and opportunities .

 
 ArXiv preprint , abs/2305.13172.

 

 
 Yi et al. (2023) 
 
Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu. 2023.

 
 Benchmarking and defending against indirect prompt injection attacks on large language models .

 
 ArXiv preprint , abs/2312.14197.

 

 
 Ying et al. (2023) 
 
Jiahao Ying, Yixin Cao, Kai Xiong, Yidong He, Long Cui, and Yongbin Liu. 2023.

 
 Intuitive or dependent? investigating llms’ robustness to conflicting prompts .

 
 ArXiv preprint , abs/2309.17415.

 

 
 Yu et al. (2022) 
 
Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2022.

 
 Generate rather than retrieve: Large language models are strong context generators .

 
 ArXiv preprint , abs/2209.10063.

 

 
 Zeng et al. (2022) 
 
Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2022.

 
 Glm-130b: An open bilingual pre-trained model.

 
 In The Eleventh International Conference on Learning Representations .

 

 
 Zhang et al. (2023a) 
 
Boyu Zhang, Hongyang Yang, Tianyu Zhou, Muhammad Ali Babar, and Xiao-Yang Liu. 2023a.

 
 Enhancing financial sentiment analysis via retrieval augmented large language models.

 
 In Proceedings of the Fourth ACM International Conference on AI in Finance , pages 349–356.

 

 
 Zhang et al. (2023b) 
 
Hang Zhang, Xin Li, and Lidong Bing. 2023b.

 
 Video-llama: An instruction-tuned audio-visual language model for video understanding.

 
 In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations , pages 543–553.

 

 
 Zhang et al. (2023c) 
 
Jiaxin Zhang, Zhuohang Li, Kamalika Das, Bradley A Malin, and Sricharan Kumar. 2023c.

 
 Sac 3 : Reliable hallucination detection in black-box language models via semantic-aware cross-check consistency .

 
 ArXiv preprint , abs/2311.01740.

 

 
 Zhang and Choi (2021) 
 
Michael JQ Zhang and Eunsol Choi. 2021.

 
 Situatedqa: Incorporating extra-linguistic contexts into qa .

 
 ArXiv preprint , abs/2109.06157.

 

 
 Zhang and Choi (2023) 
 
Michael JQ Zhang and Eunsol Choi. 2023.

 
 Mitigating temporal misalignment by discarding outdated facts .

 
 ArXiv preprint , abs/2305.14824.

 

 
 Zhang et al. (2020) 
 
Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. 2020.

 
 DIALOGPT : Large-scale generative pre-training for conversational response generation .

 
 In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations , pages 270–278, Online. Association for Computational Linguistics.

 

 
 Zhang et al. (2023d) 
 
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023d.

 
 Siren’s song in the ai ocean: A survey on hallucination in large language models .

 

 
 Zhang et al. (2023e) 
 
Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and Lu Wang. 2023e.

 
 Merging generated and retrieved knowledge for open-domain qa .

 
 ArXiv preprint , abs/2310.14393.

 

 
 Zhao et al. (2023a) 
 
Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu, Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, and Mengnan Du. 2023a.

 
 Explainability for large language models: A survey.

 
 ACM Transactions on Intelligent Systems and Technology .

 

 
 Zhao et al. (2023b) 
 
Yukun Zhao, Lingyong Yan, Weiwei Sun, Guoliang Xing, Chong Meng, Shuaiqiang Wang, Zhicong Cheng, Zhaochun Ren, and Dawei Yin. 2023b.

 
 Knowing what llms do not know: A simple yet effective self-detection method .

 
 ArXiv preprint , abs/2310.17918.

 

 
 Zheng et al. (2022) 
 
Chujie Zheng, Jinfeng Zhou, Yinhe Zheng, Libiao Peng, Zhen Guo, Wenquan Wu, Zhengyu Niu, Hua Wu, and Minlie Huang. 2022.

 
 Cdconv: A benchmark for contradiction detection in chinese conversations .

 
 ArXiv preprint , abs/2210.08511.

 

 
 Zhong et al. (2023) 
 
Zexuan Zhong, Zhengxuan Wu, Christopher D Manning, Christopher Potts, and Danqi Chen. 2023.

 
 Mquake: Assessing knowledge editing in language models via multi-hop questions .

 
 ArXiv preprint , abs/2305.14795.

 

 
 Zhou et al. (2023a) 
 
Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2023a.

 
 Lima: Less is more for alignment .

 
 ArXiv preprint , abs/2305.11206.

 

 
 Zhou et al. (2023b) 
 
Hongjian Zhou, Boyang Gu, Xinyu Zou, Yiru Li, Sam S Chen, Peilin Zhou, Junling Liu, Yining Hua, Chengfeng Mao, Xian Wu, et al. 2023b.

 
 A survey of large language models in medicine: Progress, application, and challenge .

 
 ArXiv preprint , abs/2311.05112.

 

 
 Zhou et al. (2023c) 
 
Jiawei Zhou, Yixuan Zhang, Qianni Luo, Andrea G Parker, and Munmun De Choudhury. 2023c.

 
 Synthetic lies: Understanding ai-generated misinformation and evaluating algorithmic and human solutions.

 
 In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems , pages 1–20.

 

 
 Zhou et al. (2023d) 
 
Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen. 2023d.

 
 Context-faithful prompting for large language models .

 
 ArXiv preprint , abs/2303.11315.

 

 
 Zhuang et al. (2023) 
 
Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. 2023.

 
 Toolqa: A dataset for llm question answering with external tools .

 
 ArXiv preprint , abs/2306.13304.

 

 
 Zubiaga et al. (2018) 
 
Arkaitz Zubiaga, Ahmet Aker, Kalina Bontcheva, Maria Liakata, and Rob Procter. 2018.

 
 Detection and resolution of rumours in social media: A survey.

 
 ACM Computing Surveys (CSUR) , 51(2):1–36.

 

 
 
 
 
 
 
 
 
 
 Reference 
 | 
 
 
 Model 
 | 
 
 
 Dataset 
 | 
 
 
 Quantitative Results 
 | 

 
 Context-memory conflict | 

 
 
 
 Pan et al. (2023b) 
 | 
 
 
 ChatGPT 
 | 
 
 
 NQ-1500 and CovidNews 
 | 
 
 
 Misinformation in the context can lead to a significant degradation (up to 87%) in the performance. 
 | 

 
 
 
 Xie et al. (2023) 
 | 
 
 
 ChatGPT, GPT-4, PaLM2, Qwen, Llama2, and Vicuna 
 | 
 
 
 POPQA and STRATEGYQA 
 | 
 
 
 For entity substitution-based counter-memory, only ChatGPT, GPT-4, and PaLM2 over 60% probability of choosing parametric memory. For generation-based counter-memory, all models have more than 80% probability of choosing context knowledge. 
 | 

 
 
 
 Xu et al. (2023) 
 | 
 
 
 ChatGPT, GPT-4, Llama2, and Vicuna 
 | 
 
 
 Farm, BoolQ, TruthfulQA and NQ 
 | 
 
 
 In multiple rounds of dialogue, as the number of counter-memory context increases, the cumulative proportion of belief alteration of LLMs spans from 20.7% to 78.2% 
 | 

 
 Inter-context conflict | 

 
 
 
 Jin et al. (2024a) 
 | 
 
 
 ChatGPT, Llama2, Baichuan2, FLAN-UL2 and FLAN-T5 
 | 
 
 
 NQ, TriviaQA, PopQA, and MuSiQue 
 | 
 
 
 When faced with conflicting evidence, ChatGPT’s recall declined the least, but more than 10%. 
 | 

 
 
 
 Chen et al. (2023b) 
 | 
 
 
 ChatGPT, ChatGLM, Vicuna, Qwen, and BELLE 
 | 
 
 
 RGB 
 | 
 
 
 As the noise in evidence increases, the performance of models will gradually decrease. When the noise rate exceeds 0.8, the performance of all models decreases by more than 20%. 
 | 

 
 
 
 Li et al. (2023a) 
 | 
 
 
 GPT-4, ChatGPT, PaLM2, and Llama2 
 | 
 
 
 CONTRADOC 
 | 
 
 
 Faced with self-contradictory documents, gpt4 has a more than 70% probability of determining the occurrence of a contradiction, while other models are less than 50%. 
 | 

 
 Intra-memory conflict | 

 
 
 
 Mündler et al. (2023) 
 | 
 
 
 GPT-4, ChatGPT, Llama2, and Vicuna 
 | 
 
 
 MainTestSet 
 | 
 
 
 LLMs create contradictory content, with a probability of between 15.7% and 22.9%. More powerful models create fewer contradictory results. 
 | 

 
 
 
 Zhao et al. (2023b) 
 | 
 
 
 ChatGPT, GPT-4, Vicuna, and Llama2 
 | 
 
 
 FaVIQ, ComQA, GSM-8K, SVAMP, ARCChallenge, and CommonsenseQA 
 | 
 
 
 The findings of their research reveal that even GPT-4 can exhibit an inconsistency rate of 32% in FaVIQ. 
 | 

 Table 2: 
Comparison of quantitative results on the impact of various types of knowledge conflicts.
 
 
 

## Appendix A Appendix

 

### A.1 Quantitative Analysis and Comparison

 
 In the context of a survey paper, while it is beneficial to include quantitative results and analyses concerning the impact of knowledge conflicts across various types of conflicts and the performance comparison of different mitigation strategies, it is not a strict requirement.
We acknowledge the complexity and impracticality involved in conducting such quantitative experiments, particularly due to the use of disparate datasets in behavioral analyses, as well as the variance in the inherent knowledge of LLMs across different knowledge cut-off snapshots, as detailed in  Table 1 .

 
 
 Moreover, establishing a “fair” comparison within the mitigation strategies segment poses its own set of challenges, given the diversity in objectives influenced by various assumed priors, such as the perceived accuracy of context or inherent knowledge, as discussed in the main text.
Despite these intricacies, we opt to present quantitative results by compiling existing evaluations from a range of papers. It is imperative, however, to approach this analysis with caution , recognizing that original authors may have employed different datasets, LLMs variants, or even pursued contrasting objectives. 

 
 
 

### A.2 Quantitative Results on the Impact of Knowledge Conflicts

 
 The comparison of quantitative results on the impact of the three types of knowledge conflicts is shown in  Table 2 . We pick the results of representative behavior analysis literature for comparison.

 
 
 

### A.3 Quantitative Results on the Effectiveness of Mitigation Strategies

 
 
 
 
 
 Reference 
 | 
 
 
 Model 
 | 
 
 
 Dataset 
 | 
 
 
 Quantitative Results 
 | 

 
 Faithful to context | 

 
 
 
 Shi et al. (2023a) 
 | 
 
 
 Llama, OPT, GPT-Neo, and FLAN 
 | 
 
 
 NQ-SWAP, MemoTrap, and NQ 
 | 
 
 
 Their method improves GPT-Neo 20B by 54.4% on Memotrap and by 128% on NQ-SWAP where LLMs need to adhere to the given context. 
 | 

 
 
 
 Zhou et al. (2023d) 
 | 
 
 
 ChatGPT and Llama2 
 | 
 
 
 MRC and Re-TACRED 
 | 
 
 
 Compared to the zero-shot base prompts, their prompting method leads to a reduction of 32.2% for maintaining parametric knowledge for MRC and a 10.9% reduction
for Re-TACRED on GPT-3.5. Similarly, on Llama2, there is a 39.4% reduction for MRC and a 57.3% reduction for Re-TACRED. 
 | 

 
 Discriminating misinformation | 

 
 
 
 Hong et al. (2023) 
 | 
 
 
 ChatGPT and FiD 
 | 
 
 
 NQ and TQA 
 | 
 
 
 The authors train a discriminator with about 80% F1 score and use it to improve models performance above 5%. 
 | 

 
 
 
 Pan et al. (2023b) 
 | 
 
 
 ChatGPT 
 | 
 
 
 NQ-1500 and CovidNews 
 | 
 
 
 The author’s mitigation method improves the accuracy by more than 10%. 
 | 

 
 Disentangling sources | 

 
 
 
 Wang et al. (2023h) 
 | 
 
 
 ChatGPT 
 | 
 
 
 KNOWLEDGE CONFLICT 
 | 
 
 
 The authors’ method achieved over 80% F1 score on contextual knowledge conflict detection. 
 | 

 Table 3: 
Comparison of quantitative results on the effectiveness of various mitigation strategies w.r.t. their objectives.
 
 
 
 The effectiveness of various mitigation strategies is quantitatively compared in  Table 3 . It is important to note that our analysis is limited to works addressing three predominant types of mitigating objectives within the context of memory conflicts. This selection is deliberate, as other types of mitigating objectives in different conflict categories do not yet have a substantial body of work that would allow for a meaningful cross-method comparison.