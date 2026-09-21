# MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering

Shayne Longpre

Apple Inc.

slongpre@mit.edu

Yi Lu

Apple Inc.

ylu7@apple.com

Joachim Daiber

Apple Inc.

jodaiber@apple.com

# Abstract

Progress in cross-lingual modeling depends on challenging, realistic, and diverse evaluation sets. We introduce Multilingual Knowledge Questions and Answers (MKQA), an open-domain question answering evaluation set comprising 10k question-answer pairs aligned across 26 typologically diverse languages (260k question-answer pairs in total). Answers are based on heavily curated, language-independent data representation, making results comparable across languages and independent of language-specific passages.

With 26 languages, this dataset supplies the widest range of languages to date for evaluating question answering. We benchmark a variety of state-of-the-art methods and baselines for generative and extractive question answering, trained on Natural Questions, in zero shot and translation settings. Results indicate this dataset is challenging even in English, but especially in low-resource languages. $^{1}$

# 1 Introduction

Training and evaluation data for question answering (QA) is severely lacking outside of high-resource languages like English. As unsupervised, transfer learning and zero/few-shot methods narrow the multilingual performance gap with English (Conneau et al., 2020; Lee and Lee, 2019; Cui et al., 2019a; Lewis et al., 2020), their real progress is hard to measure without challenging, realistic, and linguistically diverse evaluation sets. Existing multilingual QA datasets are realistic and challenging, but they lack linguistic diversity, comparable evaluation between languages, and are often limited to passages provided with the dataset (see Table 2).

We introduce Multilingual Knowledge Questions and Answers (MKQA) for evaluation of open

domain question answering. MKQA selects 10k realistic English queries from the Natural Questions dataset (NQ, Kwiatkowski et al., 2019) and human translates them into 25 additional languages and dialects. Accompanying these query translations we replace NQ's passage embedded answer spans with high-quality, language- and retrieval-independent answer annotations, linked directly against Wiki-data entities and a limited set of well-defined value types (numbers, dates, strings, etc.).<sup>2</sup>

See one full example in Table 1. More flexible than existing multilingual datasets, MKQA's grading procedure ensures these labels are sufficient to evaluate any QA method, including knowledge graph and generative approaches. The objective of this evaluation set is to facilitate fair comparison between languages, without imposing assumptions on the underlying QA approach. We see MKQA as a useful tool enabling practitioners to benchmark a variety of multilingual open domain question answering methods against the widest range of available languages yet. Below, we discuss its central properties as an evaluation benchmark.

Realistic & Reliable Annotations Of crucial importance to any evaluation set is (a) how well it reflects realistic, real-world settings, and (b) the reliability of its annotations. To ensure the English queries, which form the basis of our dataset, are realistic, we use Natural Questions, formulated by real users, independent of passages or answers. To ensure these queries are realistic in other languages we employ expert bilingual translators, guided by strict localization criteria. We confirm that a large majority of these queries are geographically invariant, meaning that their answer is not culturally or geographically dependent (we found that less than $4\%$ of answers are rendered incorrect by geographical and cultural context, for more details

$^{2}$ Wikidata is a collaboratively edited open knowledge graph: https://www.wikidata.org/.

arXiv:2007.15207v2 [cs.CL] 17 Aug 2021

$^{1}$ MKQA data and evaluation scripts are available at https://github.com/apple/ml-mkqa.

Table 1: Questions and answers in all supported languages for one instance in MKQA. The IETF BCP47 language codes specify the language and locale. The Entity ID corresponds to Wikidata (see for instance https://www.wikidata.org/wiki/Q794).

![](dt=2026-05-29/ht=08/b75d05f60fcdc3530c04310b7199d2821ed213199dd1b4c4cfef25da5b519dbe.jpg)

<table><tr><td>Language</td><td>Language Code</td><td>Question</td><td>Answer and Aliases</td><td>Answer Type</td><td>Entity ID</td></tr><tr><td>English</td><td></td><td>who produces the most pistachios in the world</td><td>Iran, Republic of Iran, ir, Islamic Republic of Iran, Persia, Islamic Rep. Iran</td><td>Entity</td><td>Q794</td></tr><tr><td>Arabic</td><td>ar-SA</td><td></td><td>الله صIENT貅 الحرفicit</td><td>الله</td><td>الله</td></tr><tr><td>Danish</td><td>da-DK</td><td>hvm producere flect pistacienodder i verden</td><td>Iran</td><td></td><td></td></tr><tr><td>German</td><td>de-(DE,AT,CH)</td><td>Wer produziert weltweit die meisten Pistazien</td><td>Iran, Islamische Republik Iran</td><td></td><td></td></tr><tr><td>Spanish</td><td>es-ES</td><td>¿ Quién produce la mayor candida de pistachos del mundo?</td><td>Irán, Républca Islamica de Irán, Iran, Republica Islamica de Irán</td><td></td><td></td></tr><tr><td>Finnish</td><td>fi-FI</td><td>Kuka tuottaa eniten pistaasipakhinòita maiimassa</td><td>Iran</td><td></td><td></td></tr><tr><td>French</td><td>fr-FR</td><td>Qui produit le plus de pistaches dans le monde</td><td>Iran, Perse, République islamique d&#x27;Iran</td><td></td><td></td></tr><tr><td>Hebrew</td><td>he-IL</td><td></td><td>הכלה��ienne��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��ية��oya</td><td></td><td></td></tr><tr><td>Hungarian</td><td>hu-HU</td><td>Ki termeli a legtobb pistáciáti a vilagón?</td><td>Irán, Irání Iszlám Köktársáság, Perzsiá</td><td></td><td></td></tr><tr><td>Italian</td><td>it-IT</td><td>Chi produce più pistacchi al mondo</td><td>Iran, Repubblica Islamica dell&#x27;Iran</td><td></td><td></td></tr><tr><td>Japanese</td><td>ja-JP</td><td>世界に一番ビ斯塔チを産産してるは誰だけか</td><td>イラún,イラún イスム共和国</td><td></td><td></td></tr><tr><td>Khmer</td><td>km-KH</td><td>ýnandaibn pistachios yñsésaúntyúngistudttuynon?</td><td>Jaδ</td><td></td><td></td></tr><tr><td>Korean</td><td>ko-KR</td><td>선 전쟁yne에서 충가가상품을 썼고다이를 썼선 Hatha</td><td>이본</td><td></td><td></td></tr><tr><td>Malay</td><td>ms-MY</td><td>siapa menghasilkan pistachios paling banyak di dunia</td><td>Iran</td><td></td><td></td></tr><tr><td>Dutch</td><td>nl-NL</td><td>wie producêt de meeste pistachio nootjes ter wergeld</td><td>Iran, Islamitische Republik Iran, IR, IRN</td><td></td><td></td></tr><tr><td>Norwegian</td><td>nb-NO</td><td>hvm lager mest pistasjønfter i verden</td><td>Iran</td><td></td><td></td></tr><tr><td>Polish</td><td>pl-PL</td><td>kto produkuje najwiecj e pistacji na swiecie</td><td>Iran</td><td></td><td></td></tr><tr><td>Portuguese</td><td>pt-BR</td><td>quem produz mais pistaches no mundo</td><td>Irão, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,</td><td>Irián, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,</td><td>(Иreland, Илармская Рéc�ынka), Иозанiche Ирana</td></tr><tr><td>Russian</td><td>ru-RU</td><td>Кто в міре поиздоть сожисяп ecero Фichteшиек</td><td>Iran, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , .</td><td></td><td></td></tr><tr><td>Swedish</td><td>sv-SE</td><td>Vem pr
oducrer mest pistagenöter i världen?</td><td>Iran</td><td></td><td></td></tr><tr><td>Thai</td><td>th-TH</td><td>ythmänhötnaiaiauynunvua</td><td>Işerbıncıya</td><td></td><td></td></tr><tr><td>Turkish</td><td>tr-TR</td><td>Dünyannin en qok antepe fişüzmi kim üretiyor?</td><td>Iran, Iran Islam Cumhuriyeti</td><td></td><td></td></tr><tr><td>Vietnamese</td><td>vi-VN</td><td>nuoc não sàn xuát khác quà h/to nhat thé,gioi</td><td>Iran</td><td></td><td></td></tr><tr><td>Chinese (Simplified)</td><td>zh-CN</td><td>谁生产了世界上最多的开心果</td><td>伊朗</td><td></td><td></td></tr><tr><td>Chinese (Hong Kong)</td><td>zh-HK</td><td>誰是世界上開心果產量最高的人</td><td>伊朗,伊朗伊斯蘭共和國</td><td></td><td></td></tr><tr><td>Chinese (Traditional)</td><td>zh-TW</td><td>全世界開心果的主要生產者是誰</td><td>伊朗</td><td></td><td></td></tr></table>

see Section 4.2). To ensure annotation reliability, we enforce minimum inter-grader agreement, conduct quality checks, and re-annotation from expert graders where necessary. Further, the Wikidata entity identifiers (QIDs) ground the answer annotations in structured data. This can be used for other knowledge graph-specific metrics, to retrieve other valid answer strings, and trivial entity translation into hundreds of languages beyond the scope of MKQA.

Parallel Questions Our evaluation set is fully aligned, or "parallel", across all available languages, meaning the same examples exist in all languages. This is accomplished by a mixture of expert human translation and using multilingual data from Wikidata. This property enables direct comparison between all 26 languages for fully cross-lingual or zero-shot systems. While Clark et al. (2020) point out the natural query distribution varies by language and geography, we reserve our assessment to geographically invariant queries for the purpose of more fair comparison between methods.

Retrieval-Independent Annotations Existing training and evaluation sets are oriented to "extractive" QA, providing specific passages and passagedependent answer annotations (Clark et al., 2020; Lewis et al., 2020; Artetxe et al., 2020b; Liu et al., 2019a). These types of annotations are of limited use with varying retrieval systems, knowledge graph approaches, and even generative ap

proaches because the answers are tied to the particular phrasing of their passage. Translating annotations from English passages may also introduce "translationese artifacts" as the translation is implicitly influenced by the original English structure (Artetxe et al., 2020a). These artifacts render the task easier for methods relying on English supervision or machine translation techniques. As we shall discuss in Section 3, the MKQA collection procedure yields primarily entity and structured "atomic" answer types. We contend retrieval-independent (and particularly entity-oriented) annotations minimize the risk of translation artifacts, and remove limitations on the underlying QA approach.

Linguistic Diversity Lastly, MKQA has broad linguistic diversity, covering 26 languages and dialects from 14 language family branches. Languages from MKQA cover half of the world populations' native language, and more than $90\%$ of the world population lives in a country where one of these languages is an official language (see Section 4.1 for more details). It is to our knowledge both the largest and most linguistically diverse open-domain QA evaluation set currently available (see Tables 2 and 3).

MKQA makes two important contributions to the field of multilingual question answering:

- Our answer collection procedure renders the evaluation set highly reliable, independent, and unbiased towards the QA technique used.

Table 2: Comparison of multilingual QA evaluation sets. Answer independence indicates whether the gold answer is independent of a retrieved document, and parallel questions indicate whether examples are the same across languages.

![](dt=2026-05-29/ht=08/e3695bd38bc83d05f19650dcbae523c55bcf890944f9bd986a4fd58dbd4a2076.jpg)

<table><tr><td>Multilingual QA Evaluation Set</td><td>Answer Independence</td><td>Parallel Questions</td><td>Language Fam. Branches</td><td>Languages</td><td>Total Examples</td></tr><tr><td>XQA (Liu et al., 2019a)</td><td>✓</td><td>×</td><td>5</td><td>9</td><td>28k</td></tr><tr><td>MLQA (Lewis et al., 2020)</td><td>×</td><td>✓</td><td>6</td><td>7</td><td>46k</td></tr><tr><td>XQuAD (Artetxe et al., 2020b)</td><td>×</td><td>✓</td><td>11</td><td>11</td><td>13k</td></tr><tr><td>TyDi (Clark et al., 2020)</td><td>×</td><td>×</td><td>11</td><td>11</td><td>204k</td></tr><tr><td>Xor-QA (Asai et al., 2021)</td><td>×</td><td>×</td><td>7</td><td>7</td><td>40k</td></tr><tr><td>MKQA (This work)</td><td>✓</td><td>✓</td><td>14</td><td>26</td><td>260k</td></tr></table>

This unique setup allows us to fairly compare the performance of techniques as distinct as knowledge graph-based, dense and sparse retrieval and generative QA techniques on a large number of languages (see Section 5).

- Our dataset provides fully aligned examples in the largest yet number of typologically diverse languages, enabling comparable evaluation across many languages.

We find MKQA is innately more challenging than Natural Questions from which it was derived, due to the multi-stage re-annotation process. The best model obtains only $52.3\%$ F1 in English, and only $5.7\%$ above a naive baseline on the lowest resource language. Given these qualities, our dataset facilitates broad and reliable evaluation of multilingual, open-domain question answering.

# 2 Related Work

Cross-Lingual Modeling Recent work trains cross-lingual representations with unsupervised language modeling over many languages, including Multilingual BERT (Devlin et al., 2019), XLM-R (Conneau et al., 2020), and Multilingual T5 (Xue et al., 2021). Transfer learning techniques are often applied to these cross-lingual representations to overcome the dearth of non-English data (Cui et al., 2019a; Hsu et al., 2019; Lee and Lee, 2019; Kumar et al., 2019).

Recent investigations into cross-lingual modeling have revealed "translation artifacts" in datasets where machine translation systems are used, or human translation tasks are not carefully curated (Artetxe et al., 2020a; Wintner, 2016; Rabinovich and Wintner, 2015). "Translationese" results in hidden linguistic cues in translated text that render the task easier than a natural translation.

English QA Resources A majority of question answering research focuses on English, which of

fers ample selection of evaluation datasets, including SQuAD (Rajpurkar et al., 2016), TriviaQA (Joshi et al., 2017), and Natural Questions (Kwiatkowski et al., 2019). Open Domain QA, pioneered by Green et al. (1986), is the task of answering open questions using external knowledge sources. A common approach is to combine retrieval and extractive techniques (Chen et al., 2016, 2017; Dhingra et al., 2017; Cui et al., 2017).

Monolingual QA Resources Non-English question answering resource options remain comparatively rare, with most options spanning only one other language, and rarely low-resource languages. DuReader (He et al., 2018), CMRC (Cui et al., 2019b), and DRCD (Shao et al., 2018) all offer high-quality Chinese QA datasets. Similarly, XCMRC (Liu et al., 2019b) and BiPar (Jing et al., 2019) present parallel, cross-lingual QA dataset between English and Chinese. Exploring slightly less resource-rich languages, numerous works have derived new datasets from SQuAD, employing varying degrees of human or semi-automatic translation techniques to non-English target languages: ARCD for Arabic (Mozannar et al., 2019), KorQuAD-1.0 for Korean (Lim et al., 2019), and MMQA for Hindi (Gupta et al., 2018).

Multilingual QA Resources Table 2 compares the largest publicly available multilingual question answering evaluation sets. The table highlights the following properties of each dataset: whether the available gold answers
are independent of retrieved documents, whether examples are aligned across languages, and the number of languages and examples provided. MLQA (Lewis et al., 2020) and XQuAD (Artetxe et al., 2020b) are examples of SQuAD-style extractive datasets, employing human translators to create parallel examples. Both MLQA and XQuAD ensure that all answers are answerable (discarding "No Answer" exam

ples), and derive answers from provided documents. XQA (Liu et al., 2019a), one of the few retrieval-independent QA datasets, offers cloze-style questions, leveraging Wikipedia's daily questions and entity answers to populate document-independent answers. TyDi (Clark et al., 2020), like MKQA, focuses on typological diversity in its wide language selection.

While TyDi offers a more natural distribution of questions, its annotations are based on the retrieval system used by the authors (Google search); hence their answers are actually start and end indices for spans of text within a given passage. XOR-QA (Asai et al., 2021) explores cross-lingual subtasks by re-annotating 40k TyDi examples, over 7 languages, sourcing answers from English documents and translating them back to the target language. Many of these multilingual resources have been bundled into cross-lingual benchmarks, such as XTREME (Hu et al., 2020) and XGLUE (Liang et al., 2020).

# 2.1 Comparison to Native Speaker Datasets

There are key advantages to datasets such as TyDi (Clark et al., 2020) and Xor-QA (Asai et al., 2021) which use native speakers questions, particularly in the naturalness and cultural authenticity of the corpora. However, there are also key disadvantages to these datasets which MKQA circumvents with language alignment, to provide more challenging and fair model evaluations across languages.

TyDi (Clark et al., 2020) and MKQA both target high typological diversity, highlight the importance of sourcing realistic questions (with answers unseen), and incorporate a broader distribution of question types than competing datasets (including "No Answer" and "Yes"/"No" answers). There are three main differences between MKQA and TyDi: (a) question alignment across languages, (b) answer distribution, and (c) annotation retrieval independence (closely tied with the notions of "open" and "closed" domain).

TyDi provides a different set of natural questions per language, at the expense of direct comparability across languages. Not only are the TyDi questions different between languages, but the percentage of answerable passages varies dramatically, from $22\%$ in Korean to $69\%$ in Arabic. XorQA-TyDi (Asai et al., 2021) partially resolves this issue by sourcing answers from English documents, but this may in turn re-introduce cultural biases. This suggests that the conceptual difficulty of these questions may also vary dramatically, as consumers from different locales cater their ques

tions based on their existing beliefs of the quality of the virtual assistants in their language. As a result, it is difficult to interpret the core reasons why multilingual system's performance varies between languages. To ensure this property, MKQA verifies its questions are predominantly geographically invariant, and thus the answers will not change due to geographical or cultural factors.

The second difference between datasets is the answer distribution. MKQA answers (a) are predominantly entities $(42.2\%)$ or atomic answers such as dates, binary, or numbers with units, and (b) use a different definition of "Unanswerable". Xor-QA focuses only on answerable queries, TyDi's definition conditions on the presence of the answer in the passage, whereas MKQA's definition is based on the ability of a human to find a succinct answer to a question on the web, i.e. whether it is human answerable. As a result, our annotations are not limited by the quality of selected passages, and provide higher answer coverage $(67.58\%)$ as opposed to the TyDi language average of $38\%$ .

Finally, while MKQA does not expect an answer to be derived from a single source document, TyDi is an extractive QA dataset. Consequently, its answer annotations are defined as spans, tied directly to particular Wikipedia documents and fixed index from which they were retrieved. As an evaluation set we contend the flexibility of document-independent answers is critical to not restrain what approaches can be evaluated in future research.

# 3 Dataset Collection

We aim for certain properties of our evaluation set: (i) realistic questions, (ii) reliable annotations (e.g. via inter-annotator agreement), and (iii) a flexible task setup that makes as few assumptions as possible about the underlying modeling techniques, enabling fair comparison between any approach.

# 3.1 Query Selection

Our evaluation set collection pipeline begins with the Answer Curation steps outlined in Figure 1. These are designed to yield high-concensus answer labels, with normalized textual formats, expressive alias sets for robust comparison, and grounding in structured information for entity disambiguation or more informative analysis. For the first step, we sample 10,000 queries from Natural Questions (NQ) (Kwiatkowski et al., 2019), as this is one of the few QA datasets based on realistic queries, generated by information seeking users.

![](dt=2026-05-29/ht=08/b4db7379ec11413edcdd0b5afd706bdb09ded0388b83c58a7ef08a2a5141eb22.jpg)

![](dt=2026-05-29/ht=08/ab89ed5d83c27782a16c9a04a500c0a220180abeac324b0d0bbf6e1393173b50.jpg)

Sample 10k queries, discarding passage-embedded annotations.

![](dt=2026-05-29/ht=08/ec25fcf1f58a2df3ffd23445c1cd6a0530218a425bf800107b78fdc27b91b608.jpg)

Annotators research queries to manually generate or select answers.

![](dt=2026-05-29/ht=08/69b5edd44ceb3cae275f25c1856f04b2bc06b2651da86eb17bb3fe4c7f253069.jpg)

Resolve answers by linking to Wikidata KB and normalizing formats.

![](dt=2026-05-29/ht=08/ad8a39e72ecd70e33524ccc4e1db42396a62c800202940319b5f33126cd1ae16.jpg)

Answers + Wikidata aliases are verified when $\geq 2$ resolved annotations agree, or a domain expert resolves disagreement.

![](dt=2026-05-29/ht=08/0907b8b608e6206f156f11f883676994345f2b9bd1b6ed49c832de2cbde4895a.jpg)

Answers are localized to 25 (other) languages by Wikidata QIDs or bilingual translators.

![](dt=2026-05-29/ht=08/101f83fa171f488803d7f7f09b0643331a0bc33f197ad031ce4c12bbfe727d24.jpg)

Queries are localized to 25 (other) languages by bilingual translators.

# 3.2 Raw Answer Collection

At the raw answer collection stage, 5 annotators are independently shown the query and asked to search the web to either copy or generate an ideal answer. They are asked to select an answer type (radio buttons) from the options shown below, and input the answer (text box) according to format instructions per answer type. The formatting constraints allow us to automatically link WikiData entities for the units in "number with units" and to gather well-structured data for answers such as dates, to save annotator time.

For each query, the graders select a typed answer from the following taxonomy:

# 3.3 Answer Resolution

Given the query and a candidate answer from the previous stage, annotators are next asked to normalize date/number formats and resolve the answer text against Wikidata entities, where feasible. To resolve short textual answers against Wikidata entities, we apply an internal entity linking system to
the answer string to generate Wikidata candidate entities.<sup>3</sup> The top 10 entity suggestions and their descriptions, along with the original query and short answer are then presented to 3 graders, who are asked to pick the correct reference entity or "None of the above." In cases where graders

3This step can be replicated using an off-the-shelf entity linker such as spaCy available at https://spacy.io/api-entitylinker.

do not achieve sufficient agreement or where the correct entity is not in the list, a domain expert (one of the MKQA authors/designers) provides the correct reference. Overall, this step enables us to disambiguate homonyms and collect valid answer synonyms/aliases, for more robustly measuring annotator agreement and prediction accuracy.

# 3.4 Answer Verification

Up until this stage, 5 raw answers were collected per query, and subsequently format normalized and resolved against Wikidata. In the fourth stage of Answer Curation (in Figure 1) any normalized answer given by at least 2 annotators is admitted to the final set as a gold answer. For those annotations that did not achieve the required agreement from at least two annotators, a domain expert (one of the MKQA authors/designers) with access to all 5 preliminary annotations is tasked to provide a final decision. This second manual round was afforded as much time per decision as necessary to obtain a satisfactory answer. The instructions permit the selection of existing normalized answer(s), modifying them slightly, or overriding them if necessary.

# 3.5 Answer Localization

In the last two stages of MKQA curation shown in Figure 1 we translate, or "localize" the English queries and answers into the target languages. Given the special care we took to avoid them in our methodology, and since we only localize short answers and queries (no context passages), we believe translation artifacts are likely to be minimal in MKQA.

Verified answers are localized into the target language by a combination of methods. For Wikidata-resolved answers, we leverage Wikidata's names and aliases for the target language. These names and aliases are transcribed in the native alphabet where appropriate, reflecting the expected answer in each language. Atomic answer types, including numeric, number with entity, and date types were also translated by this method, maintaining Arabic numerals for all languages, but naturalizing unit terms such as "November", "century", "b.c", "acres", and "light years". For date types specifically, for every combination of year, month, and day, we generate template answers in each language, accommodating both American and European date formats, as well as numeric and written out versions for months.

In cases where a Wikidata link could not be

found, or where answers were not available for a given language code, professional bilingual human translators were used to provide the native equivalent. For this task, human translators are given access to the English query, the English answer, and where available the Wikidata link and Wikipedia page for the entity. We found localization quality improved when bilingual translators are shown several examples prior to grading, covering each of the localization options:

# Localization Options:

# 3.6 Query Localization

The final stage of MKQA construction, as shown in Figure 1, is query localization. As with answer localization, bilingual translators were asked to translate each query ensuring the query's meaning is maximally preserved, while naturally phrased. Translators were further instructed to use localized names of named entities if they exist in the target language and to transliterate names otherwise. Our translators, who are native speakers of the target language, are verified to live in the targeted region and are required to pass an entrance exam to verify a high level of fluency in English.

Translators received a standard hourly wage varying with the target region and were not compensated per completed task, as is usual with alternative public services such as Mechanical Turk. On average, around 16 translators participated in the translation of the 10k source queries from English into each target language.

# 4 Dataset Quality and Analysis

Given our dataset collection and methodology, we evaluate the effect of our choices, and the properties

Table 3: Languages with their corresponding language families and speakers. Reach indicates the combined number of first-language (L1) and second-language (L2) speakers as a percentage of the world population (Ethnologue, Simons and Fennig, 2018).

![](dt=2026-05-29/ht=08/3d25da03161c65464dc4583cf89ff17374fa51e6a4940f8bb5fb0e671371eddc.jpg)

<table><tr><td>Family</td><td>Branch</td><td>Language</td><td>Reach</td></tr><tr><td rowspan="12">Indo-European</td><td rowspan="6">Germanic</td><td>English</td><td>16.46%</td></tr><tr><td>German</td><td>1.70%</td></tr><tr><td>Dutch</td><td>0.38%</td></tr><tr><td>Swedish</td><td>0.17%</td></tr><tr><td>Danish</td><td>0.08%</td></tr><tr><td>Norwegian</td><td>0.07%</td></tr><tr><td rowspan="4">Italic</td><td>Spanish</td><td>6.99%</td></tr><tr><td>French</td><td>3.59%</td></tr><tr><td>Portuguese</td><td>3.28%</td></tr><tr><td>Italian</td><td>0.87%</td></tr><tr><td rowspan="2">Balto-Slavic</td><td>Russian</td><td>3.35%</td></tr><tr><td>Polish</td><td>0.58%</td></tr><tr><td rowspan="2">Sino-Tibetan</td><td rowspan="2">Sinitic</td><td>Mandarin</td><td>14.54%</td></tr><tr><td>Cantonese</td><td>1.10%</td></tr><tr><td rowspan="2">Afro-Asiatic</td><td rowspan="2">Semitic</td><td>Arabic</td><td>4.44%</td></tr><tr><td>Hebrew</td><td>0.12%</td></tr><tr><td>Austronesian</td><td>Malayo-Poly.</td><td>Malay</td><td>3.47%</td></tr><tr><td>Japonic</td><td>Japonic</td><td>Japanese</td><td>1.64%</td></tr><tr><td rowspan="2">Austroasiatic</td><td>Vietic</td><td>Vietnamese</td><td>1.00%</td></tr><tr><td>Khmer</td><td>Khmer</td><td>0.21%</td></tr><tr><td>Turkic</td><td>Com. Turkic</td><td>Turkish</td><td>1.10%</td></tr><tr><td>Kra–Dai</td><td>Tai</td><td>Thai</td><td>0.78%</td></tr><tr><td>Koreanic</td><td>Han</td><td>Korean</td><td>1.03%</td></tr><tr><td rowspan="2">Uralic</td><td>Finnic</td><td>Finnish</td><td>0.07%</td></tr><tr><td>Ugric</td><td>Hungarian</td><td>0.17%</td></tr></table>

of the final set, including the selected languages, annotation quality, geographical invariance, and answer type distribution as compared to NQ.

# 4.1 Language Selection

We select a set of languages meeting both academic and practical considerations, by maximizing typological diversity as well as the share of the world population that understand at least one of the languages in the set. Table 3 shows the languages selected for our dataset with the corresponding branch of their language family. We also show the language's reach, i.e. the percentage of the world population that speaks the language either as a first or second language (based on Ethnologue data, Simons and Fennig, 2018). Since combined first- and second-language speaker statistics are not readily available, it is not straightforward to accurately determine what share of the world population can be covered by the languages in this set (e.g. a native

speaker of German may also be fluent in English). A practical option is to calculate the share of the world population that lives in a country where one of the languages in our set is recognized as an official language. By this measure, $90.62\%$ of the world population live in a country with an official language covered by the languages in our set. With the large number of diverse language families covered and the reach of the selected languages, MKQA addresses both academic and practical requirements for a wide and diverse question answering benchmark. Finally, we note that the Wikidata IDs provided for a large portion of our gold answers allow these answers to be further local
ized into Wikipedia languages beyond those in MKQA, should practitioners wish to expand their analysis.

# 4.2 Translation and Answer Quality

The quality and reliability of our dataset is highly dependent on two factors: (a) How well our professional translators were able to translate the English queries into each target language, and (b) how well our language-independent answer representations transfer to each target language.

We run a small-scale grading experiment, grading just above $1\%$ of the total data, to estimate the quality of the query translations and how well the meaning of our language-independent answer annotation is preserved across languages (geographical invariance). We present graders with the localized query and its answer annotations and ask them to judge whether (a) the localized query is an acceptable translation of the original English query, and (b) whether the provided answer (entities are shown with their QID and description, and a short explanation is added to each other answer type) is acceptable for the translated target-language query. In addition, we also ask graders to judge the answer quality for the original English queries as a baseline.

Table 4 shows the acceptance rates for query translations and answers for a small selection of languages. The table shows that query translations are consistently judged as acceptable in German, Spanish and Thai, while the quality for Chinese translations was judged as lower in comparison. Most translation issues are related to the localiza

<sup>4</sup>We determine this percentage based on Wikidata as the combined population (Wikidata property “P1082”) of all countries that have an official language (Wikidata property “P37”) in our dataset divided by the combined population of all countries in Wikidata.

![](dt=2026-05-29/ht=08/c1c395a2a11693e04d08eea67048bef643335418dd93e08f5a18e12427349af4.jpg)

Table 4: Query translation and retrieval-agnostic answer quality in various languages. Query translation acceptance rate is the percentage of query translations judged as acceptable. Answer acceptance rates is the percentage of answers graders found acceptable in response to the translated target-language query.

![](dt=2026-05-29/ht=08/9d099d52e9b5011edd5996f247b6e7a8f81503cb6729212ce3fb5cccd91e9627.jpg)

<table><tr><td rowspan="2">Language</td><td colspan="2">Acceptance Rate</td></tr><tr><td>Query Translation</td><td>Answer</td></tr><tr><td>English</td><td>-</td><td>97.03%</td></tr><tr><td>German</td><td>99.01%</td><td>91.08%</td></tr><tr><td>Spanish</td><td>99.01%</td><td>92.07%</td></tr><tr><td>Thai</td><td>96.04%</td><td>91.09%</td></tr><tr><td>Chinese (simpl.)</td><td>92.24%</td><td>89.32%</td></tr></table>

tion of entities and to domain-specific terms (e.g. sports terminology such as "receptions" in football). As expected, the acceptability of answers is judged to be higher for English than other languages but it is still at or above $90\%$ even for languages as linguistically distant from English as Thai. Note that errors in answer acceptance rate and query translation acceptance rate heavily overlap since incorrect query translations will most likely mean that the existing language-independent answer will not match. Answer quality issues fall into the following categories (illustrated with German examples):

in the world born” and questions with ambiguous answers in the data such as “is northern ireland a part of great britain.”

Translation artifacts are a recognized problem in multilingual datasets and manual grading of the data in Table 4 shows that the human translation step may introduce more or less query-answer discrepancies depending on the target language. In an alternative scenario, annotation could be performed directly on native queries from each language; however, such data is not readily available and might additionally suffer from other downsides such as relatively small user bases in less frequently spoken languages (see Section 2.1 for further discussion).

Similar to our evaluation, the authors of NQ perform a manual precision grading of their data and find an overall data precision of $84\%$ for short answers. While we hope that future work can improve on data quality further, comparatively even for the language with the most severe translation artifacts in our evaluation, Simplified Chinese, the resulting data quality (answer acceptance rate of $89\%$ ) is still within an acceptable range. In addition, our dataset provides the only available source of question answering evaluation in many languages.

We encourage authors of future multilingual datasets that use any translation methods to report and detail their geographical invariance, as we have

done, and to benchmark the reliability of examples and presence of translation artifacts.

# 4.3 Annotation Breakdowns

Next, we compare the distribution of answer types between the original NQ dataset, with those newly assigned in MKQA. As Figure 2 shows, $50\%$ of NQ are completely "Unanswerable" by retrieved passages and another $13\%$ require long passage answers. In the short answer setup for NQ both of these are considered unanswerable, amounting to $63\%$ of all questions. In comparison, only $32.4\%$ of examples are "Unanswerable" or "Long" answer type in MKQA. This is due to a shift in definition from whether a passage contains an answer, to whether a question is (succinctly) answerable by a human, with full web access. Given that the answer types in MKQA are not dependent on a learned retrieval system, they reflect the properties of the question only.

We later show that this "unanswerable" definition yields more challenging evaluation because (i) correctly answering questions is on average harder than learning when to abstain, and (ii) many of the most difficult questions were unanswerable in NQ but are answerable in MKQA. This suggests the property of "retrieval independent annotations", currently not used in any other multilingual QA benchmarks except XQA, is highly desirable for (a) constructing more challenging QA evaluation sets, and (b) yielding annotations useful to evaluate any QA approach, not just extractive QA models.

We also encourage future QA benchmarks to mimic our multi-stage data collection framework in providing supplementary metadata per example (answer type and Wikidata QIDs). Beyond basic comparison of systems, our evaluation tools allow practitioners to perform further error analysis with more interpretable metrics.

# 5 Experiments

# 5.1 Task Definition

Given a question $q^l$ in language $l$ , the task is to produce a prediction $p^l \in \{\text{No Answer}, \text{Text Answer}\}$ , where a Text Answer is a sequence of tokens in the corresponding language. $p^l$ can be obtained by any method, extracted from a document, generated, or derived from a knowledge graph.

For evaluation using MKQA gold answers, every question $q_{i}^{l}$ from $i \in [1, 10000]$ is accompanied

by a set of valid annotations $a_i^l$ per language. Every prediction $p_i^l$ is scored based on exact match (EM) and token overlap F1, as with previous open-retrieval QA datasets. The official evaluation script also ingests a "No Answer probability" for each example. If the probability is above a chosen threshold value then the prediction defaults to No Answer instead of the provided Textual Answer. As this threshold varies from 0 to 1 the predictions shift from entirely No Answer to all textual answers.

We follow NQ in reporting the best F1 over the range of thresholds, to remove threshold tuning as a factor in evaluation. A best threshold is computed and applied per language, where each example receives a "textual" (token overlap) F1 after language-specific normalization (removing whitespace, punctuation, and articles) is applied to both the pre
diction and gold answers. Finally, the official per-language F1 is computed as the mean of example F1s, and the official Macro Average F1 is the mean of per-language F1 scores.

# 5.2 Baseline Approaches

To benchmark our evaluation set, we combine state-of-the-art approaches in retrieval, machine translation, extractive QA and generative QA. All retriever models are off-the-shelf, and all reader models are finetuned on Natural Questions, including XLM-ROBERTA LARGE (Conneau et al., 2020) and M-BERT (Devlin et al., 2019) for extractive QA, and MT5-LARGE (Xue et al., 2021) for generative QA.

In each case, tokenization is handled by the multilingual model used — sentencepiece for XLM-R and MT5-LARGE, WordPiece for M-BERT, each with vocabularies initialized from their specific pre-training implementations. Further, all query and prediction translations in our approaches use Zhang et al. (2020)'s open source many-to-many, encoder-decoder machine translation system, trained on the OPUS multilingual corpus, covering 100 languages.

Retrieval Corpora Our baselines operate on a Wikipedia document corpus from Dec. 07, 2020 following previous work in open-domain question answering (Kwiatkowski et al., 2019; Asai et al., 2021; Clark et al., 2020). We use the language-specific Wikipedia corpora for Elucidsearch and the English versions for other baselines. Using Wikipedia as this base corpus is a pragmatic choice

<sup>5</sup>Note that we exclude the $10k$ examples used in our evaluation set from this training set.

based on several aspects: 1) It provides comparability across baselines and previous work, and 2) compared to large web document corpora, such as Commoncrawl, it requires less data cleaning and is computationally more tractable, which improves the replicability of our results and helps to ensure that the major variable being evaluated is model performance (rather than engineering effort). Hence, while we believe that using a web-scale corpus, such as Commoncrawl, would potentially enable even stronger baselines, we leave such experiments to future work.

Elasticsearch $\rightarrow$ XLM-R We benchmark a fully multilingual retriever approach using Elasticsearch followed by XLM-R as the extractive reader. Elasticsearch leverages language-specific tokenizers and analyzers with BM25 to search for native passages in the target language's Wikipedia dump. We used their built in language specific analyzers which include stopwords and stemmer in each language. We took the Wikipedia dump from Dec. 7, 2020 for each language as source documents. The languages Hebrew, Khmer, Korean, Malay, and Vietnamese are not part of the Elasticsearch baseline as they are not natively supported by Elasticsearch.

$\mathbf{DPR} \rightarrow \mathbf{RoBERTa}$ We benchmark an approach that utilizes state-of-the-art English retrieval and reader systems, enabled by translating the incoming query into English, and the outgoing prediction into the target language. We use off-the-shelf Dense Passage Retrieval (DPR, Karpukhin et al., 2020), followed by RoBERTA (Liu et al., 2019c) to extract a prediction.<sup>7</sup>

Gold NQ $\rightarrow$ Extractive QA For this set of baselines, optimal English retrieval is simulated via the passages provided with NQ. We illustrate baselines that leverage these provided "Gold" English documents, machine translation, and extractive QA models. We vary the type of QA model (M-BERT vs. XLM-R) and the train/test approach, comparing common Zero Shot, Translate Test, and Translate Train approaches.

In zero shot transfer each multilingual model is finetuned with NQs' default English questions $Q_{en}$ and passages $P_{en}$ . At test time the model receives

MKQA questions $Q_{xx}$ in language $xx$ , paired with English passages $P_{en}$ .

For translate test, at train time the model uses NQ's default English. At test time, MKQA questions are translated into English $Q_{xx\rightarrow en}$ , and the passage remains in English $P_{en}$ . Passages remain in English for both training and inference.

For translate train, at train time, questions are translated into the target language $Q_{en \to xx}$ . At test time the model is given queries in the target language $Q_{xx}$ and passages $P_{en}$ in the default English from NQ. Passages are always in English.

Query-only mT5 We benchmark a "closed-book", query-only generative QA approach, based on Roberts et al. (2020). This approach allows us to circumvent retrieval and machine translation entirely, using parametric knowledge within MT5 LARGE. Simply, the query is fed to the model, which is trained to generate the localized answer directly.

Gold NQ $\rightarrow$ mT5 We benchmark a stronger generative QA approach, that also has access to the English Gold NQ passages. Based on open-source implementations for MLQA and XQuAD datasets, the model is fed the non-English query, with (in this case) the English gold passage, and generates the predicted answer. $^{8}$

# 5.3 Results

Table 5 presents retrieval and end-to-end metrics for each baseline, as the mean across all 26 languages. Retrieval metrics include recall at K, measuring if the correct answer appears anywhere in the top K retrieved passages, as traditionally used in information retrieval settings. Note that these metrics are computed by looking for an exact match of the text-normalized gold answer in the text-normalized passage. We find that translation followed by English DPR outperforms the Elastic-search multilingual sparse retrievers. This is consistent with results observed in XOR-QA (Asai et al.

, 2021) which shows the surprising underperformance of multilingual retrievers. Errors are likely a combination of no answer being present in smaller non-English Wikipedia indexes, and the weak performance of sparse retrieval. The Gold NQ documents contain a valid answer $80.22\%$ of the time. However, this is likely an upper bound,

Implementation and hyperparameters based on https://github.com/google-research/multilingual-t5.

$^{6}$ https://www/elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html#arabic-analyzer

<sup>7</sup>We use the trained "Multiset" DPR model available in https://github.com/facebookresearch/DPR.

Table 5: Results for each baseline, broken down by retrieval metrics (Recall @ K passages), answerable question metrics (F1 at the best confidence threshold) and end-to-end metrics (F1 at the best confidence threshold). A naive approach, predicting exclusively No ANSWER, achieves a lower bound score of $32.42\%$ F1. Translate-Train using NQs Gold passages and an XLM-R reader outperforms all alternate settings.

![](dt=2026-05-29/ht=08/11b5637db942d65c695e7d8c1a7b4a5c316bf778873222d9be68bb80d6ef8f32.jpg)

<table><tr><td rowspan="2">Retriever</td><td rowspan="2">Reader</td><td rowspan="2">Translation 
Query</td><td rowspan="2">Answer</td><td rowspan="2">Retrieval Metrics 
R@1</td><td colspan="2">Answerable Metrics</td><td colspan="2">End-to-End Metrics</td></tr><tr><td>Mean A ∈ D F1</td><td>Mean A∉D F1</td><td>En F1</td><td>Mean F1</td></tr><tr><td>NO ANSWER</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>32.4</td><td>32.4</td></tr><tr><td colspan="9">MULTILINGUAL RETRIEVER</td></tr><tr><td>ELASTICSEARCH*</td><td>XLM-R</td><td>-</td><td>-</td><td>42.57 ±1.2</td><td>25.18 ±3.8</td><td>7.24 ±2.5</td><td>34.99</td><td>34.13 ±0.4</td></tr><tr><td colspan="9">TRANSLATE-TEST ENGLISH RETRIEVER</td></tr><tr><td>DPR</td><td>ROBERTA</td><td>Test</td><td>Test</td><td>53.62 ±2.2</td><td>20.33 ±4.1</td><td>10.24 ±1.8</td><td>45.19</td><td>36.81 ±1.2</td></tr><tr><td colspan="9">GOLD NQ PASSAGES</td></tr><tr><td>GOLD NQ</td><td>M-BERT</td><td>-</td><td rowspan="6">Test</td><td rowspan="6">80.22</td><td>20.13 ±5.5</td><td>7.56 ±1.7</td><td rowspan="3">51.97</td><td>37.8 ±2.0</td></tr><tr><td>GOLD NQ</td><td>M-BERT</td><td>Test</td><td>
28.10 ±6.5</td><td>12.1 ±2.1</td><td>41.4 ±2.2</td></tr><tr><td>GOLD NQ</td><td>M-BERT</td><td>Train</td><td>32.21 ±6.0</td><td>14.8 ±1.9</td><td>44.1 ±1.8</td></tr><tr><td>GOLD NQ</td><td>XLM-R</td><td>-</td><td>38.81 ±3.2</td><td>20.05 ±2.6</td><td rowspan="3">52.27</td><td>45.5 ±1.4</td></tr><tr><td>GOLD NQ</td><td>XLM-R</td><td>Test</td><td>34.23 ±5.0</td><td>16.38 ±2.6</td><td>42.9 ±2.1</td></tr><tr><td>GOLD NQ</td><td>XLM-R</td><td>Train</td><td>40.28 ±3.1</td><td>20.93 ±2.7</td><td>46.0 ±1.4</td></tr><tr><td colspan="9">GENERATIVE MODELS</td></tr><tr><td>QUERY-ONLY</td><td>MT5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>43.8</td><td>35.0 ±1.2</td></tr><tr><td>GOLD NQ</td><td>MT5</td><td>-</td><td>-</td><td>80.22</td><td>36.8 ±6.2</td><td>17.07 ±2.6</td><td>47.6</td><td>38.5 ±2.2</td></tr></table>

$\mathbf{A} \in \mathbf{D}$ denotes metrics for where the answer A exists in the top retrieved document D (exact match). $\mathbf{A} \notin \mathbf{D}$ denotes metrics for where the answer A does not exist in top retrieved document D (exact match).  
* Elasticsearch benchmark does not include Hebrew, Khmer, Korean, Malay, and Vietnamese.

as these documents are often very long and noisy, such that NQ annotators often marked them as not containing an answer to the question, even though we find the gold answer string is present.

For end-to-end metrics, we measure F1 just for English ("EN F1"), which omits the impact of machine translation, and mean F1 over all 26 languages. The naive baseline of only predicting No Answer achieves a lower bound score of $32.42\%$ . We chose to combine both Unanswerable and Long Answers into the No Answer category for evaluation to focus MKQA on short, factoid answers that can be evaluated automatically and robustly. Unsurprisingly, we observe models with access to NQ gold documents achieve the best results, with Translate Train XLM-R achieving the best mean F1 of $46.0 \pm 1.4$ .

Among these methods, XLM-R outperforms M-Bert, and Translate-Train outperforms Translate-Test and Zero Shot. Generative approaches using MT5 perform fairly well, even under zero shot conditions (trained only on English), or without any passage provided (query-only).

We also measure the F1 scores for the subset of answerable questions to measure the ability of the retrievers and readers to find the right answer. We separately report the average all-language F1 for (i) questions in which a gold answer appears in the top retrieved document, and (ii) questions in which none are found. As expected, performance is much

higher for both extractive and generative models where the retriever has succeeded. Translate Train with XLM-R still achieves the best performance. XLM-R also performs well on the correct outputs $(\mathbf{A} \in \mathbf{D})$ of the weakest retriever, Elastischearch, though there are fewer of them. Comparing with end-to-end metrics, which includes unanswerable questions, answerable questions are more difficult to answer.

Overall, these results show how collecting relevant passages remains a challenging bottleneck in multilingual open-retrieval QA. Multilingual retrievers, English state-of-the-art retrievers, and generative QA models all fail to overcome this problem, and even when gold passages are provided, multilingual readers and machine translation still fail to consistently produce localized answers (with generous evaluation settings).

In Figure 3 we compare cross-lingual performance between languages, ranked by F1 score. We plot XLM-R Zero Shot to minimize the noise from machine translation. As expected, the XLM-R model performs fairly well on English (52.3), and common non-English languages, including the most common Indo-European Germanic and Italic languages, but poorly on languages from lower-resourced families. Note that the minimum F1 score is $32.42\%$ , where a threshold of 0 predicts No Answer to every question. Interestingly, as

![](dt=2026-05-29/ht=08/dc8fc7e5f55d70ddc140b63147f98a9df573f27b79494a65a9334b1dc90cb66c.jpg)

the Aggregate F1 decreases, the Unanswerable F1 rises on average from $\sim 27\%$ to $\sim 29\%$ , abstaining from an answer more often. Given the parallel questions property of MKQA, these metrics allow a practitioner to specifically identify languages with weak model performance, and answer abstinence behaviour for commonly used reader models, such as XLM-R. Even before considering a cultural shift in query distribution, these metrics allow us to isolate performance on geographically invariant queries, and general effectiveness of transfer learning for particular languages and training regimes.

# 5.4 Unanswerable vs. Long Answers

As discussed in Section 4.3, following the Short Answer setup for Natural Questions (Kwiatkowski et al., 2019) we define Unanswerable as a query without a short answer (i.e. examples with long or unanswerable answer types) — for our task. Although evaluating long answers is important, it is out of the scope of MKQA. The primary benefit of this decision is that it enforces the retrieval-independent annotations property of MKQA, since long answers have an unbounded number of correct answer strings. Here we investigate whether long and “truly” unanswerable examples in MKQA are treated differently by our baseline models.

To answer this question, we break down the larger Unanswerable set into the long and 'truly' unanswerable examples, comprising $56\%$ and $44\%$ respectively. We then compute the final performance (F1) by model type and by language for each of these two categories. We find the results vary ac

cording to the quality of the model and the language (as do performance on answerable queries), but the difference between the long answer and truly unanswerable scores are marginal. For instance, XLM-R Translate Train, using Gold NQ passages, achieves $84.2\%$ F1 on long, and $84.7\%$ on truly unanswerable examples, with a mean difference over all 26 languages of only $0.5\%$ . These differences are similarly negligible across other baselines. This finding suggests standard open-domain QA systems, trained on short answer datasets like Natural Questions, have learned to consider long answers as unanswerable, and do not appear to find one set more challenging than the other.

# 6 Discussion

Difficulty of MKQA Our baselines represent a strong and diverse set of methods, that score competitively with state-of-the-art on similar open domain question answering datasets. Nonetheless, on English alone, the best system recieves an F1 score of only $52.3\%$ , less than the same methods achieve on the open datasets Natural Questions and TriviaQA, or other standard benchmarks for this task. These comparative results demonstrate MKQA is highly challenging and leaves ample room for improvement in both English and the long tail of natural languages. In this section we explain why, with a detailed comparison to its closest set, Natural Questions.

Why is MKQA so challenging for state-of-the-art approaches even for English open-domain QA? To shed light on this, we compare the difficulty of

![](dt=2026-05-29/ht=08/1cf169d7e956098207fd595638be9b9ead6fb31cf734b170ba8aeb0214e3a6bd.jpg)

![](dt=2026-05-29/ht=08/595f9d9a5e65ffb99317429d9834be0c280afe3e380af580318992fbdaf2b26a.jpg)

![](dt=2026-05-29/ht=08/7e01eaa0c0b41836e339a3c5251fa7c33ecb895ba37c585889db081cde7549a3.jpg)

English-only annotations between Natural Questions (NQ) and MKQA. In Figure 4 we use the same BERT-LARGE English model (trained on NQ, using Gold NQ passages) and evaluate it on both sets of annotations. The "F1 by Answer Type" diagram shows unanswerable examples in MKQA (re
d line) are easier than the unanswerable examples in NQ (red dashed line), as the model maintains higher performance at all No Answer confidence thresholds. The opposite relationship is observed for answerable examples.

We hypothesize that this is due to the Retrieval-Independence property and high coverage of our re-annotation process (described in Section 3). Due to the annotation procedures NQ uses, there are several cases that can lead to a potential answer missing from the dataset: (a) the initial retrieval may have not produced a candidate, (b) the answer may have not been in Wikipedia, or (c) NQ graders may have missed a valid answer. MKQA annotations are not susceptible to (a) and (b) and likely less impacted by (c). Consequently, the most challenging questions migrated from unanswerable in NQ to answerable in MKQA, shifting the unanswerable distribution from $63\%$ to $32\%$ (as shown in Figure 2). Consider the following examples.

(a) NQ retrieval failure In this example, the NQ retrieved document does not contain an answer to the question, causing no long or short answer (No Answer) in NQ. There exists a better Wikipedia document (Wheel of Fortune) that does contain the MKQA answer "Autumn Erhard".

- Q: Who won the most money on wheel of fortune?

(b) No Wikipedia answer This is also an answerable query, labelled as no answer by NQ, because the answer is not found on Wikipedia (either by NQ or our best efforts). However, an answer can be found by MKQA graders from other websites and sources.

(c) Annotator misses valid answer For this query, the answer is clearly visible in the provided Wikipedia article, but NQ's annotation process yields no answer.

Given the answer to these queries are not easily found in the corpus, by retrieval, or by human annotators, they are likely more challenging on average. As such, their label shift from no answer in NQ to answerable in MKQA likely explains why there is higher mean difficulty of answerable questions in

MKQA, as observed in Figure 4. To understand the prevalence of each error type, we compute how often any MKQA answer appears in the retrieved document for which the NQ label says no answer exists. We find a valid answer appears in $70.4\%$ of these documents, suggesting category (c), annotator error, is the largest source of such unanswerable queries in NQ (and the largest source of improvement in label quality for MKQA).

The middle and right diagrams in Figure 4 normalize the answer types by their proportion within the dataset, so we can compare their relative contributions to the aggregate F1 (the sum of answerable and unanswerable). NQ labels enable a much higher aggregate F1 score (69.38% at the best threshold) than MKQA (52.08% at the best threshold) primarily due to the higher proportion of unanswerable examples — which are easier on average than answerable examples. By comparing the ratio of unanswerable to answerable examples attempted at the best thresholds in each of the middle and right diagrams (the blue regions vs. the red regions) we see that the MKQA task is more oriented to answering questions rather than abstaining.

Due to the Parallel Question property of MKQA, the dataset is similarly challenging in all 26 languages. There is also a noticeable gap between the performance on English and on lower-resourced languages (Figure 3). For Korean and Arabic the best F1 score is only $6\%$ higher than the lower bound score of $32.42\%$ obtained from predicting exclusively "unanswerable." This demonstrates that existing transfer learning methods have significant deficits to overcome for low-resource multilingual QA to match English performance. MKQA offers a challenging benchmark to measure this cross-language progress specifically.

Future Work The parallel questions property of MKQA offers alternative task setups in addition to typical open domain question answering. Lewis et al. (2020) suggests a generalized cross-lingual transfer task (G-XLT) where the question and answer languages are intentionally different. Alternatively, future work might assume we are given the English question-answer pairs, and attempt to propagate these answers into other languages by localizing the questions and answers.

We anticipate that this dataset will enable industry practitioners and researchers to rapidly test and compare novel cutting-edge techniques for QA against existing techniques in a more fair, compara

ble, and precise manner than previous benchmarks. Additionally, we hope that the linguistic diversity and large number of languages will inspire more researchers to treat model performance across many (partially less-resourced) languages as an important and worthy goal in itself. As MKQA offers the only open-QA option for many of these languages, we also hope to spark important research in these monolingual, non-English settings.

# 7 Conclusion

In this work, we introduce a multilingual open domain question answering evaluation set. Its properties, including geographical invariance, language-parallel questions, retrieval-independent annotations, and linguistic diversity, set it apart from existing resources in terms of annotation quality, difficulty, and flexibility to evaluate new approaches. We encourage future multilingual benchmarks to adopt data collection and annotation principles to promote higher-quality, and informative evaluation practices.

We evaluate several baselines, based on state-of-the-art methods, and demonstrate ample room for improvement both in English and in the tail of lower-resourced languages. We hope that this evaluation set enables wider exploration of cross-lingual and monolingual methods in non-English QA.

# Acknowledgments

We would like to thank Chris DuBois, who has been instrumental to releasing this data. Ilya Chatsviorkin, Xiao Ling, Nikhil Ramesh, Ni Lao, Agatha Downey, Silviana Ciurea-Ilcus, Anthony Chen, and Russ Webb have provided invaluable feedback on early versions of this paper. Thanks to Ivan Montero for testing out early versions of the data. Thanks to Pablo N. Mendes and Charles Srisuwananukorn for guidance and support, as well as to Noriyo Sakamoto for help in data collection. This work would not have been possible without the TryRating annotation platform.

# References