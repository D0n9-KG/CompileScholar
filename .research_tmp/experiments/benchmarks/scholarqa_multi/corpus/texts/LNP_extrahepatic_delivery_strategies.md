# DECODING EXTRAHEPATIC TARGETING OF LIPID NANOPARTICLES WITH INTERPRETABLE MACHINE LEARNING

Asal Mehradfar $^{1,\dagger,*}$ , Mohammad Shahab Sepehri $^{1,\dagger,*}$ , Owen Antholine $^{2}$ , Varun Shankar $^{2}$ , Glen S. Kwon $^{2}$ , Salman Avestimehr $^{1}$ , and Morteza Rasoulianboroujeni $^{2,3,*}$

$^{1}$ Department of Electrical and Computer Engineering, University of Southern California, Los Angeles, CA

$^{2}$ Pharmaceutical Sciences Division, School of Pharmacy, University of Wisconsin-Madison, Madison, WI

$^{3}$ Department of Pharmaceutical Sciences, Gatton College of Pharmacy, East Tennessee State University, Johnson City, TN

$^{\dagger}$ These authors contributed equally to this work.

\*Corresponding authors: mehradfa@usc.edu, sepehri@usc.edu, rasoulianbor@etsu.edu

# ABSTRACT

Lipid nanoparticles (LNPs) have transformed RNA medicine, yet their clinical utility remains constrained by predominant hepatic accumulation after systemic administration. Redirecting LNPs to extrahepatic tissues requires a deeper understanding of how lipid chemistry and formulation composition jointly govern in vivo biodistribution. Here, we develop an interpretable machine learning framework to predict hepatic versus extrahepatic LNP accumulation and identify molecular design rules for extrahepatic RNA delivery. A literature-derived dataset of 476 intravenously administered LNP formulations was curated from 81 studies, integrating formulation composition, lipid chemical structures, and experimentally reported IVIS-based biodistribution profiles. Standardized SMILES representations of ionizable lipids, helper lipids, sterols, PEGylated or polymer-conjugated lipids, additional lipids, and polymer repeat units were converted into RDKit Expert descriptors and combined with formulation-level variables to generate an 808-dimensional feature representation. Logistic regression, random forest, and XGBoost classifiers achieved strong predictive performance on a held-out test set, with ROC–AUC values of 0.839, 0.866, and 0.874, respectively. SHAP-based model interpretation and consensus feature ranking revealed that ionizable-lipid descriptors dominate biodistribution prediction, while formulation composition, particularly ionizable lipid, sterol, and PEGylated/polymer-conjugated lipid fractions, also contributes substantially. Notably, the top 20 consensus features retained nearly all predictive information in tree-based models, indicating that a compact set of molecular determinants is sufficient to classify LNP tropism. The most informative features implicated electrotopological surface properties, charge- and hydrophobicity-weighted surface areas, molecular topology, and amide/alkyl structural motifs as key drivers of extrahepatic accumulation. This study establishes an interpretable, data-driven strategy for decoding LNP biodistribution and provides actionable design principles for engineering next-generation LNPs beyond the liver. Our code and dataset are publicly available at https://github.com/AsalMehradfar/lnp-extrahepatic-targeting/.

# 1 Introduction

RNA-based therapeutics, including messenger RNA (mRNA), small interfering RNA (siRNA), microRNA (miRNA), and antisense oligonucleotides (ASOs), have emerged as powerful tools for modulating gene expression and targeting previously "undruggable" proteins, transcripts, and genes $[1]$ . Despite their potential, the clinical application of RNA is limited by its inherent instability, as RNA molecules are highly susceptible to degradation by nucleases in the reticuloendothelial system and within endosomal compartments. Additionally, their inability to passively diffuse across lipid bilayers further hinders their efficacy. Therefore, the development of efficient delivery systems is essential to protect RNA from degradation, enable targeted transport to specific tissues and cell types, and ensure effective cytoplasmic release for optimal therapeutic outcomes $[2]$ .

In 2018, the United States Food and Drug Administration (FDA) approved the first therapeutic small interfering RNA (siRNA), ONPATTRO™, which utilizes lipid nanoparticles (LNPs) as the delivery system for the treatment of hereditary transthyretin-mediated amyloidosis. Similarly, LNP-mediated delivery was successfully employed to transport mRNA encoding the spike protein in COVID-19 vaccines, demonstrating the efficacy of LNPs in RNA drug delivery [3]. The successful application of LNPs in these RNA-based products underscores their effectiveness as non-viral delivery systems. LNPs are now considered a superior alternative to viral vectors, which face limitations in clinical applications due to high immunogenicity, narrow tissue specificity, costly manufacturing processes, and biosafety concerns [4]. The commercial viability and clinical significance of RNA-LNP formulations have been well established, fueling extensive research into their application for diverse purposes, including cancer immunotherapy, vaccines, and gene-editing therapeutics [5].

Despite their notable advantages, LNPs face challenges in achieving broader therapeutic applications, primarily due to their tendency to accumulate in hepatocytes following intravenous administration, a process mediated by endogenous transport mechanisms involving lipoprotein receptors and apolipoproteins $[6, 7]$ , leading to rapid clearance from the bloodstream and limiting their use predominantly to liver-related diseases $[8, 9]$ . However, recent preclinical studies have demonstrated that modifications in lipid structures and LNP composition can facilitate the delivery of RNA to extrahepatic organs, such as the lungs and spleen, thereby broadening their therapeutic potential beyond hepatic applications. For instance, Cheng et al. $[10]$ developed Selective Organ Targeting (SORT) technology, which incorporates a supplemental lipid, termed a SORT molecule, into the LNP formulation to achieve tissue-specific mRNA delivery to the lungs or spleen. Additionally, Ni et al. $[11]$ and Gan et al. $[12]$ engineered novel piperazine-containing lipids and adamantyl-containing phospholipids, respectively, that preferentially deliver mRNA to immune cells in vivo without the need for targeting ligands. Radmand et al. $[13]$ demonstrated that incorporating cationic cholesterol with a cationic helper lipid enhances mRNA delivery to the heart and various lung cell types, including stem cell-like populations. Furthermore, Qiu et al. $[14]$ synthesized a new ionizable lipid that alters LNP tropism, favoring mRNA delivery to the lungs. These advancements underscore the potential of next-generation LNP platforms to overcome current limitations in extrahepatic delivery and broaden the scope of RNA-based therapeutics.

While these advancements suggest that it is possible to achieve extrahepatic tropism by altering the chemical identity of LNPs, specifically through modifications in lipid structures and compositions, the precise relationship between chemical identity and biological function remains poorly understood. As a result, the development of new LNP formulations often relies on screening large libraries of candidates to evaluate their in vivo biodistribution in animal models, such as mice $[15, 16]$ . Although emerging technologies, including DNA/RNA barcoding coupled with sequencing $[13, 17, 18, 19]$ , have facilitated in vivo high-throughput screening, exclusive dependence on experimental approaches presents significant limitations: (i) high costs associated with scaling up throughput and (ii) the immensity of the chemical space, which encompasses billions of potential lipid structures and compositions. This complexity arises from the four key components of LNPs—ionizable lipids, helper lipids, sterols, and PEGylated lipids—each playing a critical role in determining the properties and functions of the LNPs $[20]$ . Given the impracticality of exhaustive experimental screening, particularly for extrahepatic delivery where the likelihood of success is low, the development of new, more efficient strategies for LNP discovery is critically needed.

Machine learning (ML) has emerged as a powerful approach for accelerating materials discovery and rational design by enabling accurate prediction of material properties from compositional and structural data. In the field of LNPs, ML models have recently demonstrated considerable success in correlating the chemical identity of LNPs with their biological activity. These approaches include supervised binary classification models for predicting high versus low transfection efficiency $[21, 22]$ , multiclass classification models for predicting different levels of transfection performance $[23]$ , and deep learning frameworks integrated with combinatorial chemistry for the design and optimization of ionizable lipids $[24, 25]$ . More recently, benchmarking studies have systematically compared different molecular representations and ML algorithms for predicting LNP transfection performance, providing valuable guidance for future model development $[26, 27]$ .

Despite recent advances, existing ML models for LNP design have important limitations that restrict their translational utility. First, many studies focus primarily on ionizable lipid structure while treating other formulation variables—including helper lipids, sterols, PEGylated lipids, additional lipid components, and overall lipid composition—as fixed. As a result, these models do not fully capture the combinatorial effects of formulation components that collectively determine LNP biological activity. Second, many current models offer limited mechanistic interpretability, making it difficult to extract actionable design principles for developing new lipid structures or optimized LNP formulations. Third, most ML approaches remain focused on predicting in vitro transfection efficiency. Although useful for high-throughput screening, such predictions may have limited translational relevance because the relationship between in vitro transfection, in vivo biodistribution, and therapeutic efficacy remains incompletely understood.

Emerging data-driven efforts in in vivo LNP biodistribution are primarily focused on high-throughput platforms, such as barcoded nanoparticle libraries. While these approaches generate large-scale datasets, they primarily quantify nanoparticle accumulation or cellular uptake in specific organs or cell populations and do not necessarily measure functional RNA delivery, transfection, or protein expression. In contrast, in vivo imaging-based biodistribution studies provide imaging readouts linked to reporter expression and therefore offer a functional measure of delivery efficiency. However, the lower throughput of imaging-based studies, combined with the difficulty of extracting standardized quantitative data from the literature, has limited their use in predictive modeling. Thus, there remains a critical need for interpretable, data-driven frameworks that leverage functional biodistribution data to predict tissue-specific LNP delivery and elucidate how lipid chemistry and formulation composition jointly govern extrahepatic accumulation, functional transfection, and protein expression. Such approaches are essential for the rational engineering of therapeutically relevant RNA delivery systems and for expanding the clinical potential of LNP-based nanomedicines beyond the liver.

In this study, we present an interpretable, data-driven framework for predicting the functional in vivo biodistribution of LNPs while simultaneously identifying actionable molecular and formulation design principles for the rational development of next-generation LNPs. We developed supervised machine learning models using a curated dataset comprising 476 LNP formulations extracted from the published literature and performed a comprehensive meta-analysis through feature importance analysis and consensus feature ranking. The dataset integrates lipid chemical structures, formulation compositions, and corresponding in vivo biodistribution profiles, enabling the classification of formulations as either hepatic- or extrahepatic-accumulating. By combining predictive modeling with model interpretability, this framework reveals how molecular- and formulation-level features collectively influence tissue tropism, providing mechanistic insights and practical design rules for engineering LNPs with enhanced extrahepatic delivery for RNA therapeutics.

# 2 Dataset

To investigate the molecular determinants governing hepatic and extrahepatic biodistribution of LNPs, we constructed a literature-derived dataset by systematically curating experimentally validated LNP formulations reported in published in vivo studies. For each formulation, the dataset includes formulation composition, biodistribution outcomes, and the chemical structures of the constituent lipid components. The curated chemical structures were subsequently used to generate molecular descriptors, enabling a unified molecular representation for subsequent computational analysis. The following subsections describe the publication-based data curation process and the construction of the molecular feature representation used throughout this study.

# 2.1 Publication Selection and Data Extraction

The following inclusion criteria were used to select LNP formulations for this study: (i) formulations were administered intravenously; (ii) biodistribution data obtained by in vivo imaging system (IVIS) imaging were available for multiple organs; (iii) formulations did not contain active targeting ligands; and (iv) the complete chemical structures of all lipid components could be extracted from published sources.

A Google Scholar search covering publications from January 2011 to January 2025 was conducted using the keywords “in vivo biodistribution,” “lipid nanoparticle,” “IVIS,” “mRNA,” and “ionizable.” A total of 126 publications were identified, of which 81 satisfied the inclusion criteria and were included in the final analysis. The complete list of selected publications is provided in Table S1 in the Supplementary Information.

Each study was systematically reviewed to extract the composition of the LNP formulations, including the identities and molar ratios of ionizable lipids, helper lipids, PEGylated lipids or alternative polymer-conjugated lipids, and sterols. For formulations containing an additional lipid component, its identity and molar ratio were also recorded. For polymer-conjugated lipids, whether PEGylated or conjugated to an alternative polymer, the molecular weight and monomer structure of the corresponding polymer were recorded to ensure consistent structural representation.

Experimental biodistribution profiles were used to assign binary classification labels. Formulations exhibiting the highest IVIS signal intensity in the liver were classified as liver-accumulating, whereas formulations showing maximal signal intensity in any other organ were classified as non-liver-accumulating. This curation process resulted in a dataset comprising 476 distinct LNP formulations, including 307 unique ionizable lipids, 29 helper lipids, 2 sterols, 34 additional lipids, and 24 PEGylated or polymer-conjugated lipids. Of the 476 formulations, 237 were classified as non-liver-accumulating and 239 as liver-accumulating.

Additional details regarding the distributions of formulation compositions between the two classes, as well as the numbers of unique lipid structures across lipid categories, are provided in Figure 1.

![](images/6c2fa46e23d3d8e80d45ed3c48f9a7f257d007c3b3e07eb04bc7bd7b40243b83.jpg)

<details>
<summary>boxplot</summary>

| Lipid Type       | Liver Composition (%) | Non-Liver Composition (%) |
| ---------------- | --------------------- | ------------------------- |
| Ionizable Lipid   | 50                    | 50                        |
| Helper Lipid     | 25                    | 25                        |
| Sterol           | 45                    | 45                        |
| PEGylated Lipid  | 5                     | 5                         |
</details>

(a)

![](images/487c4d3ee0ae252a2ead582e4419b37bd47f8ce586e419c07a58b6e74ea2faa4.jpg)

<details>
<summary>bar</summary>

| Lipid Type        | Liver | Non-Liver | All  |
| ----------------- | ----- | --------- | ---- |
| Ionizable Lipid    | 155   | 170       | 310  |
| Helper Lipid      | 20    | 15        | 30   |
| Sterol            | 1     | 2         | 1    |
| PEGylated Lipid   | 12    | 15        | 20   |
| Additional Lipid  | 15    | 20        | 35   |
</details>

(b)   
Figure 1: Overview of the curated LNP dataset. (a) Distribution of lipid composition across liver and non-liver LNP formulations. Box plots show the mole fraction (%) of the major lipid components, including ionizable lipids, helper lipids, sterols, and PEGylated lipids. The center line represents the median, the box denotes the interquartile range, and whiskers indicate the data spread. (b) Chemical diversity of lipid components in the dataset measured by the number of unique molecular structures (unique SMILES) observed for each lipid category. Counts are shown separately for liver and non-liver formulations, together with the total number of unique structures across the dataset.

# 2.2 Molecular Descriptor Extraction and Feature Construction

Canonical SMILES representations were curated for each molecular component present in the LNP formulations, including ionizable lipids, helper lipids, sterols, PEGylated lipids, additional lipids, and the corresponding polymer component. Molecular descriptors were computed from the standardized SMILES representations using the RDKit cheminformatics toolkit $[28, 29]$ with the RDKit Expert descriptor set. This descriptor collection encompasses a broad range of physicochemical, constitutional, topological, and electronic properties derived directly from molecular graph representations.

Descriptors were calculated independently for each molecular component to preserve their individual structural characteristics. For formulations lacking one or more molecular components, the corresponding descriptor fields were encoded in a consistent manner to maintain a uniform feature representation across all formulations. Descriptor columns exhibiting no variation across the curated dataset were subsequently removed. Following this preprocessing step, the final molecular descriptor set comprised 802 numeric features, including 158 descriptors derived from ionizable lipid SMILES, 147 from helper lipid SMILES, 102 from sterol SMILES, 147 from PEGylated lipid SMILES, 162 from additional lipid SMILES, and 86 from the corresponding polymer SMILES. For clarity, molecular descriptor names are prefixed according to the LNP component from which they are derived, including ionizable lipid (IL), helper lipid (HL), sterol (SL), PEGylated lipid (PL), and additional lipid (AL). For example, IL-BalabanJ denotes the BalabanJ descriptor computed for the ionizable lipid, whereas PL-VSA EState7 denotes the corresponding descriptor computed for the PEGylated lipid.

To construct the final feature representation, the filtered molecular descriptors were combined with formulation-level compositional variables, including the mole percentages of ionizable lipids, helper lipids, sterols, PEGylated lipids, and additional lipids, together with the molecular weight of the corresponding polymer. This integration resulted in a comprehensive 808-dimensional feature representation for each LNP formulation. Figure 2 schematically illustrates the workflow for constructing the final feature representation used in the subsequent machine learning analyses.

# 3 Machine Learning Framework

This section describes the machine learning framework used to classify LNP formulations and identify molecular determinants associated with extrahepatic targeting. Supervised learning models were trained using the molecular feature representation described in Section 2 and evaluated using standard classification metrics. Model interpretability was subsequently investigated using SHAP analysis, and feature rankings from multiple models were integrated to identify consensus molecular determinants.

![](images/52a8d13d719988af3bf5bed958b3392b17e05188a77aefae4520c016bfeaf262.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Lipid Nanoparticle (LNP)"] --> B["RDKit Expert Descriptor Calculation"]
    B --> C["Remove Constant Descriptors"]
    C --> D["Formulation Composition"]
    D --> E["Feature Integration"]
    E --> F["808-Dimensional Feature Representation"]

    subgraph A
        G["Sterol"] --> H["Hydroxy-Lipid"]
        I["Ionizable Lipid"] --> J["N-CH2-CH2-O-C(=O)-CH2-CH3"]
        K["Polymer Repeat Unit"] --> L["Polymer"]
        M["Ionizable Lipid"] --> N["N-CH2-CH2-O-C(=O)-CH2-CH3"]
        O["Helper Lipid"] --> P["N-CH2-CH2-O-C(=O)-CH2-CH3"]
    end

    subgraph B
        Q["Sterol"] --> R["217 descriptors"]
        S["PEGylated Lipid"] --> T["217 descriptors"]
        U["Additional Lipid"] --> V["217 descriptors"]
        W["Polymer"] --> X["217 descriptors"]
    end

    subgraph C
        Y["Ionizable Lipid"] --> Z["158 descriptors"]
        AA["Helper Lipid"] --> AB["147 descriptors"]
        AC["Sterol"] --> AD["102 descriptors"]
        AE["PEGylated Lipid"] --> AF["147 descriptors"]
        AG["Additional Lipid"] --> AH["162 descriptors"]
        AI["Polymer"] --> AJ["86 descriptors"]
    end

    subgraph D
        AK["Ionizable Lipid %"] & AL["Helper Lipid %"] & AM["PEGylated Lipid %"] & AN["Additional Lipid %"] & AO["Polymer Molecular Weight"]
    end

    subgraph E
        AP["Ionizable Lipid %"] & AQ["Helper Lipid %"] & AR["PEGylated Lipid %"] & AS["Additional Lipid %"] & AT["Polymer"]
    end

    subgraph F
        AU["Ionizable Lipid %"] & AV["Helper Lipid %"] & AW["PEGylated Lipid %"] & AX["Additional Lipid %"] & AY["Polymer"]
    end

    subgraph F
        AZ["Ionizable Lipid %"] & BA["Helper Lipid %"] & BB["PEGylated Lipid %"] & BC["Additional Lipid %"]
        BD["Polymer"] & BE["Composition"]
    end
```
</details>

Figure 2: Construction of the molecular feature representation. Representative LNP components, including the ionizable lipid, helper lipid, sterol, PEGylated lipid, and polymer repeat unit, are first represented by their molecular structures. RDKit expert descriptors are calculated independently for each molecular component, followed by the removal of descriptors with constant values across the dataset. The remaining descriptors are combined with formulation composition variables, including the mole percentages of each lipid component and the polymer molecular weight. Finally, all molecular descriptors and formulation variables are concatenated through feature integration to produce the final 808-dimensional feature representation used for downstream machine learning analyses.

# 3.1 Machine Learning Models

Three supervised machine learning models were employed to classify LNP formulations as liver- or non-liver-accumulating: logistic regression (LR) [30, 31], random forest (RF) [32], and extreme gradient boosting (XGBoost) [33]. Logistic regression was selected as a regularized linear baseline, whereas random forest and XGBoost were chosen as complementary tree-based ensemble methods capable of capturing complex nonlinear relationships between molecular descriptors, formulation composition, and biodistribution outcomes. Together, these models provide a balance between interpretability and predictive performance while representing diverse machine learning paradigms commonly used for molecular property prediction.

# 3.2 Model Training and Evaluation

Prior to model training, missing feature values were imputed using the median value of each feature, and missing-value indicators were incorporated to preserve information associated with absent molecular components. Logistic regression was implemented within a preprocessing pipeline including median imputation, missing-value indicators, and feature standardization, followed by elastic-net regularization $[34]$ , whereas random forest and XGBoost were trained using median-imputed features with missing-value indicators but without feature standardization. Hyperparameter values were selected empirically based on preliminary experiments and subsequently fixed for all analyses. Logistic regression employed the SAGA solver with elastic-net regularization ( $L_{1}$ ratio = 0.5, regularization strength C = 1.0), whereas random forest was trained using 500 decision trees with balanced class weights. XGBoost employed 500 boosting iterations with a learning rate of 0.05, a maximum tree depth of 3, subsampling ratio of 0.8, column sampling ratio of 0.8, and an $L_{2}$ regularization coefficient of 1.0. All models were trained using a fixed random seed (42) to ensure reproducibility.

The curated dataset was divided into training (80%) and held-out test (20%) subsets using stratified random sampling to preserve the class distribution. Model development was assessed using five-fold stratified cross-validation on the training set, with the mean and standard deviation of the ROC-AUC reported across folds. Final model performance was evaluated on the held-out test set using ROC-AUC, accuracy, recall, and F1-score. ROC curves were generated to compare the classification performance of the three models. Model calibration was additionally assessed using calibration curves and the Brier score, which quantifies the agreement between predicted probabilities and observed outcomes, with lower values indicating better calibration.

# 3.3 Feature Importance Analysis

Model interpretability was investigated using SHapley Additive exPlanations (SHAP) $[35]$ . SHAP values were computed for each trained model using the held-out test set to quantify the contribution of individual features to the predicted class. A linear SHAP explainer was employed for logistic regression, whereas tree-based SHAP explainers were used for the random forest and XGBoost models.

Global feature importance was quantified by computing the mean absolute SHAP value of each feature across all held-out test samples. This aggregation produced a single global importance score for each feature, representing its average contribution to model predictions across the held-out test set. Because missing-value indicators were incorporated during data preprocessing, the corresponding SHAP contributions were merged with those of their associated original features to obtain a single importance score for each molecular or compositional feature. These global feature importance scores served as the basis for constructing a consensus feature ranking across the three machine learning models, as described in the following subsection.

# 3.4 Consensus Feature Ranking

The global feature importance scores obtained from the three machine learning models were first normalized within each model to enable meaningful comparison of feature importance across models. For each model, the normalized feature importance of feature i was computed as

$$
\tilde {I} _ {i} ^ {(m)} = \frac {I _ {i} ^ {(m)}}{\sum_ {k = 1} ^ {P} I _ {k} ^ {(m)}}, \tag {1}
$$

where $I_{i}^{(m)}$ denotes the global feature importance score of feature $i$ obtained from model $m$ , computed as the mean absolute SHAP value across the held-out test samples, and $P$ is the total number of features.

Features within each model were subsequently ranked in descending order according to their normalized feature importance scores, with larger normalized importance corresponding to higher feature importance. When multiple features had identical normalized importance values within an individual model, tied features were assigned the average of their corresponding ranks, resulting in one ranked feature list for each model.

The consensus rank of each feature was then calculated as the median of its ranks across the three machine learning models,

$$
R _ {i} ^ {\text { cons }} = \text { median } \left(R _ {i} ^ {\text { LR }}, R _ {i} ^ {\text { RF }}, R _ {i} ^ {\text { XGB }}\right), \tag {2}
$$

where $R_{i}^{LR}$ , $R_{i}^{RF}$ , and $R_{i}^{XGB}$ denote the ranks assigned to feature i by the logistic regression, random forest, and XGBoost models, respectively. Features were subsequently sorted according to their consensus ranks, with lower consensus ranks indicating features that were consistently identified as more important across different machine learning models.

When multiple features shared the same consensus rank after median aggregation, ties were resolved using the average normalized feature importance across the three models,

$$
\bar {I} _ {i} = \frac {\tilde {I} _ {i} ^ {\mathrm{LR}} + \tilde {I} _ {i} ^ {\mathrm{RF}} + \tilde {I} _ {i} ^ {\mathrm{XGB}}}{3}, \tag {3}
$$

where features with larger average normalized importance were assigned higher priority. The resulting consensus ranking was used to identify subsets of the highest-ranked features for downstream analyses and reduced-feature model training.

# 4 Results and Discussion

# 4.1 Prediction of Extrahepatic Targeting

The ability of the proposed molecular feature representation to distinguish liver-targeting from extrahepatic-targeting LNP formulations was first evaluated using logistic regression, random forest, and XGBoost classifiers. Receiver operating characteristic (ROC) curves for the three models are presented in Figure 3, and the corresponding quantitative performance metrics are summarized in Table 1.

All three models achieved strong predictive performance, with test ROC–AUC values exceeding 0.83. Among the evaluated models, XGBoost achieved the highest test ROC–AUC of 0.874, followed closely by random forest (0.866), whereas logistic regression achieved a test ROC–AUC of 0.839. Similar trends were observed for accuracy, recall, and F1-score, with XGBoost consistently providing the strongest overall performance on the held-out test set. Five-fold cross-validation demonstrated comparable performance across training folds, indicating good generalization of the proposed models.

Compared with logistic regression, the superior performance of the tree-based models suggests that the relationship between molecular descriptors and LNP biodistribution is governed by nonlinear interactions that are not fully captured by a linear decision boundary. Nevertheless, the competitive performance achieved by logistic regression indicates that the proposed molecular descriptor representation itself contains substantial predictive information. Calibration analysis further demonstrated good agreement between predicted probabilities and observed outcomes for all three models (Figure S1). Consistent with the ROC–AUC results, XGBoost achieved the lowest Brier score, followed closely by random forest (Table S2), indicating reliable probability estimates across the evaluated models.

![](images/2a1fd06a161a8b22ec1a07edd8d8957dc63cb15aae7c70fc5358fbf04edb97ce.jpg)

<details>
<summary>line</summary>

| Model               | AUC    |
| ------------------- | ------ |
| Logistic Regression | 0.839  |
| Random Forest       | 0.865  |
| XGBoost             | 0.874  |
</details>

Figure 3: Performance comparison of ML models for LNP classification. ROC curves for logistic regression, random forest, and XGBoost models evaluated on the test set. The curves illustrate the trade-off between true positive rate and false positive rate for predicting liver versus non-liver LNP formulations.

Table 1: Performance of ML models across different feature-set sizes for LNP classification. Models were evaluated using the full feature set (All) and reduced feature sets containing the top 20 and top 10 features. Cross-validation performance is reported as the mean AUC ± standard deviation across folds, and held-out test performance is reported using AUC, accuracy, recall, and F1-score. Within each feature-set block, the best-performing values are shown in bold. 

<table><tr><td>Feature Set</td><td>Model</td><td>CV AUC</td><td>Test AUC</td><td>Accuracy</td><td>Recall</td><td>F1-score</td></tr><tr><td rowspan="3">All</td><td>logistic regression</td><td>0.826±0.054</td><td>0.839</td><td>0.792</td><td>0.792</td><td>0.792</td></tr><tr><td>random forest</td><td>0.882±0.036</td><td>0.866</td><td>0.802</td><td>0.792</td><td>0.800</td></tr><tr><td>XGBoost</td><td>0.875±0.034</td><td>0.874</td><td>0.854</td><td>0.854</td><td>0.854</td></tr><tr><td rowspan="3">20</td><td>logistic regression</td><td>0.759±0.056</td><td>0.724</td><td>0.615</td><td>0.583</td><td>0.602</td></tr><tr><td>random forest</td><td>0.854±0.050</td><td>0.866</td><td>0.781</td><td>0.812</td><td>0.788</td></tr><tr><td>XGBoost</td><td>0.852±0.048</td><td>0.851</td><td>0.781</td><td>0.812</td><td>0.788</td></tr><tr><td rowspan="3">10</td><td>logistic regression</td><td>0.707±0.053</td><td>0.714</td><td>0.729</td><td>0.750</td><td>0.735</td></tr><tr><td>random forest</td><td>0.842±0.054</td><td>0.859</td><td>0.760</td><td>0.792</td><td>0.768</td></tr><tr><td>XGBoost</td><td>0.852±0.048</td><td>0.847</td><td>0.771</td><td>0.771</td><td>0.771</td></tr></table>

# 4.2 Consensus Molecular Determinants of Extrahepatic Targeting

To identify molecular descriptors that were consistently informative across different machine learning models, a consensus feature ranking was constructed from the SHAP-based importance rankings of logistic regression, random forest, and XGBoost (Figure 4). Despite differences in the underlying learning algorithms, the three models exhibited substantial agreement in their highest-ranked features, indicating that the identified molecular determinants were robust to the choice of machine learning model. The complete consensus ranking of the top 50 features is provided in Table S3.

Several molecular descriptors were consistently identified among the highest-ranked features across the three models, including IL-VSA EState5, IL-VSA EState7, IL-MinEStateIndex, and IL-BalabanJ. Although the relative ordering of individual descriptors varied between models, these descriptors repeatedly appeared among the most influential predictors, highlighting their importance in distinguishing liver-targeting and extrahepatic-targeting LNP formulations.

A notable observation is that the majority of the highest-ranked descriptors originated from the ionizable lipid (IL) component of the LNP formulation, suggesting that this component plays a dominant role in distinguishing liver-targeting and extrahepatic-targeting formulations. In addition to molecular descriptors, formulation composition also emerged as an important predictor, with the ionizable lipid percentage (IL %), sterol percentage (SL %), and PEGylated lipid percentage (PL %) consistently appearing among the highest-ranked features across the three models. Together, these findings suggest that both the molecular properties of the ionizable lipid and the relative composition of the LNP formulation play important roles in distinguishing liver-targeting and extrahepatic-targeting formulations.

# 4.3 Predictive Power of Consensus Molecular Determinants

To assess whether the consensus molecular determinants identified in the previous subsection captured the majority of the predictive information, model performance was re-evaluated using reduced feature sets derived from the consensus feature ranking. Specifically, the highest-ranked 20 and 10 consensus features were selected, and the three machine learning models were retrained using only these descriptors. Their predictive performance was subsequently compared with that of models trained using the complete molecular descriptor set (Table 1). Performance results for additional feature subsets are provided in Table S4. The influence of the number of selected features on model performance is further illustrated in Figure S2.

Reducing the feature set from 808 descriptors to the top 20 consensus-ranked features resulted in only a modest decrease in predictive performance for the tree-based models. Notably, random forest achieved the same test ROC–AUC (0.866) using only the top 20 features as with the complete feature set, while XGBoost exhibited only a slight reduction in test ROC–AUC from 0.874 to 0.851. Although logistic regression showed a larger decrease in predictive performance, the tree-based models consistently maintained superior classification accuracy. These results indicate that the highest-ranked consensus molecular determinants retained nearly all of the predictive information contained in the complete molecular descriptor representation.

Further reducing the feature set to the top 10 consensus-ranked features resulted in a moderate decline in predictive performance for all three models, although random forest and XGBoost continued to outperform logistic regression. As shown in Figure S2, predictive performance improved substantially as additional consensus-ranked features were incorporated and approached a plateau after approximately 20 features. Collectively, these findings demonstrate that a

![](images/d21c8d408ccf0a59f67e7ecb66a993075550372d3004c2dd1d87b951e6e8a3db.jpg)

<details>
<summary>bar</summary>

| Category | Mean | SHAP value |
| :--- | :--- | :--- |
| SL % | 0.53 | 0.54 |
| IL-VSA EState7 | 0.36 | 0.36 |
| IL-BalabanJ | 0.34 | 0.34 |
| IL-PEOE VSA12 | 0.32 | 0.32 |
| IL-fr unbrch alkane | 0.30 | 0.30 |
| IL-NumAmideBonds | 0.29 | 0.29 |
| IL-fr amide | 0.28 | 0.28 |
| IL-VSA EState5 | 0.28 | 0.28 |
| PL-SlogP VSA10 | 0.20 | 0.20 |
| IL-FractionCSP3 | 0.19 | 0.19 |
| IL-PEOE VSA8 | 0.19 | 0.19 |
| PL % | 0.15 | 0.15 |
| IL-PEOE VSA1 | 0.14 | 0.14 |
| IL % | 0.11 | 0.11 |
| IL-MinEStateIndex | 0.04 | 0.04 |
| IL-AvgIpc | 0.03 | 0.03 |
| IL-HallKierAlpha | 0.03 | 0.03 |
| IL-BertzCT | 0.03 | 0.03 |
| IL-EState VSA4 | 0.03 | 0.03 |
| IL-VSA EState8 | 0.03 | 0.03 |
</details>

(a)

![](images/a37eeac3ca84eb5c25d4ffbc3e1a8999f1dc1acbc7a3e8dcbbc0053e1bd5932b.jpg)  
(b)

![](images/b0a5edb676a2b790fbabb67acb778012d492b84e7f765b9222157acd880cd3d1.jpg)

<details>
<summary>bar</summary>

| Category | Mean [SHAP value] |
| :--- | :--- |
| IL-VSA EState7 | 0.5 |
| PL % | 0.36 |
| IL-VSA EState5 | 0.34 |
| IL % | 0.33 |
| IL-BertzCT | 0.32 |
| IL-MinEStateIndex | 0.31 |
| SL % | 0.28 |
| IL-ESstate VSA4 | 0.17 |
| IL-FractionCSP3 | 0.16 |
| IL-BalabanJ | 0.15 |
| PL-SlogP VSA10 | 0.14 |
| IL-fr unbrch alkane | 0.12 |
| IL-PEOE VSA1 | 0.11 |
| IL-Avglpc | 0.11 |
| IL-HallKierAlpha | 0.10 |
| IL-VSA EState8 | 0.10 |
| IL-NumAmideBonds | 0.08 |
| IL-PEOE VSA8 | 0.04 |
| IL-fr amide | 0.03 |
| IL-PEOE VSA12 | 0.02 |
</details>

(c)   
Figure 4: SHAP-based feature importance analysis of ML models predicting LNP biodistribution. Mean absolute SHAP values for the top 20 consensus features are shown for (a) logistic regression, (b) random forest, and (c) XGBoost models. Bars are color-coded according to feature categories. Feature names are prefixed by the lipid component from which the descriptor is derived (e.g., IL–), where the suffix denotes the corresponding molecular descriptor or formulation variable. Features are ranked independently within each model, and the relative importance of these consensus features across models highlights key predictors of liver-targeting behavior in LNP formulations.

relatively small subset of consensus molecular determinants is sufficient to accurately distinguish liver-targeting and extrahepatic-targeting LNP formulations while substantially reducing the dimensionality of the molecular descriptor space.

# 4.4 Predictive Contributions of Consensus Molecular Determinants

To further investigate how the consensus molecular determinants influenced model predictions, SHAP summary plots were generated for the top 20 consensus-ranked features identified across the three machine learning models (Figure 5). Whereas the previous analyses established which molecular descriptors were consistently important, the SHAP summary plots provide additional insight into how variations in these descriptors contributed to the classification of individual LNP formulations as liver-targeting or extrahepatic-targeting.

![](images/ea770419bffdcc887e2badd5750f0518a0a19a07c592a856098b11e05ac66e11.jpg)

<details>
<summary>scatter</summary>

| Drug | Feature Value |
| --- | --- |
| SL % | High |
| IL-VSA EState7 | High |
| IL-BalabanJ | High |
| IL-PEOE VSA12 | High |
| IL-fr unbrch alkane | High |
| IL-fr amide | High |
| IL-NumAmideBonds | High |
| IL-VSA EState5 | High |
| PL-SlogP VSA10 | High |
| IL-FractionCSP3 | High |
| IL-PEOE VSA8 | High |
| PL % | High |
| IL-PEOE VSA1 | High |
| IL % | High |
| IL-MinEStateIndex | High |
| IL-Avglpc | High |
| IL-EState VSA4 | High |
| IL-VSA EState8 | High |
| IL-HallKierAlpha | High |
| IL-BertzCT | High |
</details>

(a)

![](images/bc701d8c571856cff66a6f386ea33e9261554d5534cd0eb78e01261b9df938fb.jpg)

<details>
<summary>scatter</summary>

| Feature | SHAP Value |
| --- | --- |
| IL-MinEStateIndex | ~0.01 |
| IL-NumAmideBonds | ~-0.03 |
| IL % | ~0.02 |
| IL-VSA EState5 | ~0.01 |
| PL % | ~0.04 |
| SL % | ~0.06 |
| IL-PEOE VSA1 | ~0.02 |
| IL-BertzCT | ~0.01 |
| IL-fr amide | ~-0.02 |
| IL-HallKierAlpha | ~0.01 |
| IL-PEOE VSA12 | ~0.01 |
| IL-VSA EState7 | ~0.01 |
| IL-EState VSA4 | ~0.01 |
| IL-VSA EState8 | ~0.01 |
| IL-PEOE VSA8 | ~0.01 |
| IL-AvgIpc | ~0.01 |
| IL-BalabanJ | ~0.01 |
| IL-fr unbrch alkane | ~0.01 |
| IL-FractionCSP3 | ~0.01 |
| PL-SlogP VSA10 | ~0.01 |
</details>

(b)

![](images/b70aaf9e56b730eca266d6c62585106ebd53ffdf96500a097db21459bd5ead46.jpg)

<details>
<summary>scatter</summary>

| Gene | SHAP Value | Feature Value |
| --- | --- | --- |
| IL-VSA EState7 | ~0.8 | High |
| PL % | ~0.6 | High |
| IL-VSA EState5 | ~0.4 | High |
| IL % | ~0.2 | High |
| IL-BertzCT | ~0.0 | High |
| IL-MinEStateIndex | ~-0.2 | High |
| SL % | ~-0.4 | High |
| IL-EState VSA4 | ~-0.6 | High |
| IL-FractionCSP3 | ~-0.8 | High |
| IL-BalabanJ | ~-0.9 | High |
| PL-SlogP VSA10 | ~-0.7 | High |
| IL-fr unbrch alkane | ~-0.5 | High |
| IL-PEOE VSA1 | ~-0.3 | High |
| IL-AvgIpc | ~-0.1 | High |
| IL-HallKierAlpha | ~0.1 | High |
| IL-VSA EState8 | ~0.3 | High |
| IL-NumAmideBonds | ~0.5 | High |
| IL-PEOE VSA8 | ~0.7 | High |
| IL-fr amide | ~0.9 | High |
| IL-PEOE VSA12 | ~1.0 | High |
</details>

(c)   
Figure 5: SHAP summary plots illustrating feature effects in ML models predicting LNP biodistribution. SHAP values for the top 20 consensus features are shown for (a) logistic regression, (b) random forest, and (c) XGBoost models. Each point represents a single LNP formulation in the dataset. The horizontal position of a point indicates the SHAP value, which quantifies the contribution of that feature to the model prediction for that sample; positive values increase and negative values decrease the predicted probability of liver targeting. Feature values are color-coded from low (blue) to high (pink). Feature names are prefixed by the lipid component from which the descriptor is derived (e.g., IL–), where the suffix denotes the corresponding molecular descriptor or formulation variable. The spread of points along the horizontal axis reflects how variations in each feature influence model predictions across samples, highlighting features that consistently contribute to predictions of liver-targeting behavior.

Despite differences in the underlying learning algorithms, the three models exhibited broadly consistent patterns in the contributions of the highest-ranked molecular determinants. In particular, random forest and XGBoost displayed remarkably similar SHAP distributions, while logistic regression identified many of the same influential descriptors despite its linear decision boundary. This agreement further supports the robustness of the consensus feature ranking and indicates that the identified molecular determinants contribute consistently to model predictions across different machine learning algorithms.

Several of the highest-ranked descriptors, including IL-VSA EState7, IL-MinEStateIndex, and the formulation composition variables IL %, SL %, and PL %, exhibited broad SHAP value distributions, indicating that variations in these features substantially influenced the classification of LNP formulations as liver-targeting or non-liver-targeting. For many of these descriptors, higher and lower feature values produced systematically different SHAP contributions, demonstrating that changes in their values were consistently associated with changes in model predictions. The broader SHAP distributions observed for these descriptors further emphasize their dominant influence on prediction relative to lower-ranked features, which generally exhibited SHAP values concentrated near zero.

The top-ranked features can be grouped into six major categories: formulation composition, electrotopological state and surface-area descriptors, charge-weighted surface-area descriptors, hydrophobicity-weighted surface-area descriptors, molecular topology and complexity descriptors, and functional group/structural motif descriptors. This categorization provides a chemically interpretable framework for understanding how formulation-level variables and molecular-level features collectively contribute to hepatic versus extrahepatic LNP accumulation and protein expression.

The mole percentages of ionizable lipid (IL%), PEGylated/polymer-conjugated lipid (PL%), and sterol (SL%) were among the top-ranked features for determining biodistribution across all three models. IL% showed the clearest extrahepatic trend in this family, with higher values associated with extrahepatic accumulation and protein expression across logistic regression, random forest, and XGBoost models. In contrast, PL% showed the opposite trend, with higher values consistently associated with hepatic accumulation across all three models. SL% showed weaker and more model-dependent behavior, although higher values were more hepatic in logistic regression and random forest, while XGBoost showed limited separation. These findings suggest that a higher ionizable lipid fraction, together with lower PEGylated/polymer-conjugated lipid and sterol fractions, represents a compositionally relevant signature associated with extrahepatic accumulation. The study by Zhu et al. [36] provides independent experimental support for the importance of lipid molar composition in determining functional LNP biodistribution. Because this study was not included in our training dataset, it did not influence model learning and can therefore serve as external validation for the compositional trends identified by our ML analysis. In their multi-step high-throughput screening study, 1080 LNP formulations were initially evaluated in vitro, after which 32 top-performing formulations for each lipid category were clustered and tested in vivo. Among formulations composed of the same components, Dlin-MC3/DOPE/cholesterol/DMG-PEG, clusters with average molar ratios of 35/35/29.74/0.26 and 32.84/24.66/41.94/0.56 showed distinct organ-level protein expression profiles. The formulation cluster with higher ionizable lipid content and lower cholesterol and PEG-lipid fractions showed reduced hepatic protein expression (65.9% to 49.8%) and increased splenic expression (30.0% to 45.2%), consistent with our model-derived association between higher IL%, lower PL%, lower cholesterol, and enhanced extrahepatic delivery. This trend was further supported by individual formulations containing the same component set, Dlin-MC3/DSPC/cholesterol/DMG-PEG, but different molar ratios. Increasing the ionizable lipid fraction while decreasing cholesterol and PEG-lipid content from 36.3/3.63/59.9/0.12 to 54.5/5.45/39.9/0.08 decreased hepatic protein expression from 60.8% to 47.9% and increased lung expression from 2.3% to 21.4%. These results demonstrate that, even when lipid identities are held constant, changes in formulation composition can substantially alter functional organ-level delivery, supporting the compositional trends identified by our ML analysis.

The electrotopological surface descriptor family included IL-VSA EState5, IL-VSA EState7, IL-VSA EState8, IL-EState VSA4, and IL-MinEStateIndex. These descriptors integrate atom-level electrotopological state information with molecular surface-area contributions, capturing how distinct electronic environments are distributed across the surface of the ionizable lipid. Within this family, IL-VSA EState7 showed the strongest and most consistent class separation, with higher values associated with extrahepatic accumulation. In contrast, lower IL-VSA EState7 values were associated with hepatic accumulation in logistic regression. IL-VSA EState5 and IL-MinEStateIndex also exhibited meaningful but more model-dependent trends, with higher values more frequently associated with hepatic accumulation, whereas IL-VSA EState8 and IL-EState VSA4 showed weaker or less consistent separation between classes. Collectively, these findings suggest that electrotopological surface environments do not contribute equally to LNP biodistribution. Rather, the combination of higher IL-VSA EState7 with lower IL-VSA EState5 and IL-MinEStateIndex may define an ionizable-lipid surface profile associated with improved extrahepatic accumulation and functional protein expression. Chemically, this descriptor pattern may reflect a heteroatom-rich headgroup or linker region in which tertiary amine, ester, amide, or ether functionalities create a distinct surface-exposed electronic environment. For example, atoms adjacent to an ionizable tertiary amine and nearby carbonyl or ether oxygens may exhibit altered electrotopological states due to differences in electronegativity, bonding pattern, and molecular connectivity. Thus, high IL-VSA EState7 likely

reflects a specific arrangement of heteroatom-proximal surface atoms rather than the presence of a single functional group, while lower IL-VSA EState5 and IL-MinEStateIndex suggest limited contribution from alternative lower-EState surface environments or localized electronically perturbed regions within the ionizable lipid structure.

Charge-weighted surface-area descriptors included IL-PEOE VSA1, IL-PEOE VSA8, and IL-PEOE VSA12. These descriptors group atoms according to partial charge ranges and sum their approximate surface-area contributions, thereby capturing how charged or weakly charged atomic environments are distributed across the ionizable lipid surface. Within this family, IL-PEOE VSA1 showed the most informative and consistent pattern. Higher IL-PEOE VSA1 values were associated with extrahepatic accumulation in logistic regression, whereas lower values were associated with hepatic accumulation in random forest and XGBoost. In contrast, IL-PEOE VSA8 and IL-PEOE VSA12 showed weaker and more model-dependent separation between hepatic and extrahepatic classes. These findings suggest that the distribution of partial charge across the ionizable lipid surface may contribute to LNP biodistribution. In particular, higher IL-PEOE VSA1 may reflect greater surface contribution from strongly electronegative atomic environments, such as oxygen- or nitrogen-containing regions associated with ester, ether, amide, carbonyl, or tertiary amine-containing motifs. Chemically, this descriptor may capture ionizable lipids in which heteroatom-rich headgroup or linker regions create localized charged or polar surface domains that influence interactions with serum proteins, cell membranes, or endosomal interfaces. The weaker trends observed for IL-PEOE VSA8 and IL-PEOE VSA12 indicate that not all charge-defined surface regions are equally informative. Rather, specific partial-charge bins, particularly those captured by IL-PEOE VSA1, may be more closely associated with extrahepatic accumulation and functional protein expression.

The hydrophobicity-weighted surface-area category was represented by PL-SlogP VSA10, a descriptor derived from the PEGylated/polymer-conjugated lipid. SlogP VSA descriptors group atoms according to atomic logP contributions and sum their corresponding surface-area contributions. PL-SlogP VSA10 was one of the clearest non-ionizable lipid features associated with extrahepatic accumulation: higher values were associated with extrahepatic accumulation across logistic regression, random forest, and XGBoost, while lower values were more hepatic in XGBoost. This suggests that the hydrophobic surface contribution of the PEGylated/polymer-conjugated lipid, likely related to the lipid anchor or polymer-lipid interface, may be an important determinant of LNP biodistribution. This is consistent with the known role of PEG/polymer-lipid hydrophobic anchoring in controlling surface presentation and PEG/polymer retention within the nanoparticle [37, 38]. Mui et al. [39] demonstrated that approximately $80\%$ of the initially incorporated ${}^{3}\mathrm{H}$ -labeled PEG-C14 lipid dissociated from LNPs within $2\mathrm{h}$ in mouse plasma, whereas PEG-C16 and PEG-C18 lipids were retained for substantially longer periods. This difference was attributed to the greater energetic barrier required for longer hydrophobic chains to desorb from the lipid membrane. Consistent with this mechanism, the estimated desorption rates were $45\%$ , $1.3\%$ , and $0.2\%$ per hour for PEG-C14, PEG-C16, and PEG-C18 lipids, respectively. The authors further showed that PEG-lipid anchor length influenced hepatic accumulation. PEG-C14-containing LNPs accumulated rapidly in the liver, reaching approximately $55\%$ of the injected dose at $4\mathrm{h}$ . In contrast, PEG-C16- and PEG-C18-containing LNPs accumulated more slowly and reached lower maximum hepatic levels of approximately $35\%$ and $25\%$ of the injected dose, respectively.

Molecular topology and complexity descriptors included IL-BertzCT, IL-BalabanJ, IL-HallKierAlpha, IL-FractionCSP3, and IL-AvgIpc. These descriptors capture global structural characteristics of the ionizable lipid, including molecular complexity, graph compactness, molecular shape, degree of saturation, and graph-based information content. Within this family, IL-FractionCSP3 showed the clearest and most consistent hepatic trend, with higher values associated with hepatic accumulation across all three models and lower values associated with extrahepatic accumulation in logistic regression. Because FractionCSP3 reflects the proportion of $sp^{3}$ -hybridized carbons, higher values generally indicate a more saturated, aliphatic, and three-dimensional carbon framework, whereas lower values suggest greater contribution from unsaturated, $sp^{2}$ -rich, or more conformationally restricted structural elements. IL-BalabanJ and IL-HallKierAlpha also showed hepatic trends at higher values, particularly in logistic regression/XGBoost and random forest/XGBoost, respectively. These descriptors reflect aspects of molecular connectivity, compactness, branching, atom-type composition, and shape. Thus, higher values may capture ionizable lipids with more compact or topology-dependent structural arrangements that favor hepatic accumulation. In contrast, IL-BertzCT showed an opposite pattern in the tree-based models, where lower values were associated with hepatic accumulation, suggesting that greater molecular complexity may be more favorable for extrahepatic accumulation in nonlinear models. IL-AvgIpc showed limited class separation, indicating that not all graph-information descriptors contribute equally to biodistribution prediction. Overall, the strongest conclusion from this descriptor family is that ionizable lipids with high IL-FractionCSP3, and to a lesser extent high IL-BalabanJ and IL-HallKierAlpha, are more consistently associated with hepatic accumulation than extrahepatic delivery. Chemically, this suggests that highly saturated, aliphatic, and topology-defined ionizable lipid frameworks may favor liver accumulation, whereas lower saturation and/or greater structural complexity may contribute to extrahepatic accumulation and functional protein expression. However, because these descriptors capture global molecular topology rather than discrete functional groups, their interpretation should be viewed as structural guidance rather than a direct mechanistic assignment.

Functional group and structural motif descriptors included IL-NumAmideBonds, IL-fr amide, and IL-fr unbrch alkane. IL-NumAmideBonds and IL-fr amide quantify amide-containing motifs within the ionizable lipid, whereas IL-fr unbrch alkane captures the presence of linear hydrophobic alkyl segments. Within this family, the amide-related descriptors showed a consistent pattern in which low amide content was associated with hepatic accumulation, while higher IL-fr amide values were associated with extrahepatic accumulation in random forest and XGBoost. This suggests that amide-containing structural motifs may contribute favorably to extrahepatic delivery, potentially by altering hydrogen-bonding capacity, polarity, lipid packing, degradability, or interactions with proteins and biological membranes. IL-fr unbrch alkane also showed an extrahepatic trend, with higher values associated with extrahepatic accumulation in logistic regression and random forest, and lower values associated with hepatic accumulation in XGBoost. Chemically, this descriptor may reflect defined linear hydrophobic domains within the ionizable lipid tails or linker regions. Such unbranched alkyl segments can influence lipid packing, membrane insertion, nanoparticle stability, and interactions with biological interfaces. Together, these findings suggest that ionizable lipids containing both amide motifs and well-defined linear hydrophobic alkyl segments may be more likely to support extrahepatic accumulation and functional protein expression. However, because these descriptors capture structural motifs rather than complete mechanistic pathways, their interpretation should be viewed as design guidance for prioritizing ionizable lipid chemistries rather than as direct evidence of causality.

Taken together, the most differentiating features associated with extrahepatic accumulation were higher IL%, lower PL%, higher IL-VSA EState7, higher PL-SlogP VSA10, higher IL-fr amide, and higher IL-fr unbrch alkane. In contrast, higher PL%, higher IL-FractionCSP3, higher IL-BalabanJ, higher IL-HallKierAlpha, and higher IL-MinEStateIndex were more frequently associated with hepatic accumulation. Based on these trends, an LNP formulation designed for extrahepatic accumulation could prioritize a relatively high ionizable lipid fraction, avoid excessive PEGylated/polymer-conjugated lipid content, incorporate PEG/polymer-lipid structures with favorable hydrophobic anchor-associated surface features, and use ionizable lipids enriched in the molecular surface/electronic environments captured by IL-VSA EState7. At the ionizable lipid level, candidate structures could preferentially include amide-containing motifs and appropriate unbranched alkyl segments while avoiding structural profiles associated with high FractionCSP3, high BalabanJ, high HallKierAlpha, or high MinEStateIndex. These recommendations provide interpretable, descriptor-guided design principles for prioritizing LNP formulations with a higher probability of extrahepatic accumulation.

Collectively, these results demonstrate that the consensus molecular determinants not only contain strong predictive information but also contribute to model predictions in a consistent and interpretable manner across different machine learning algorithms. The close agreement between the SHAP analyses of logistic regression, random forest, and XGBoost provides additional confidence that the identified molecular determinants represent robust predictors of LNP biodistribution rather than artifacts of a specific machine learning model.

# 5 Conclusion

This work demonstrates that interpretable machine learning can bridge a critical knowledge gap in LNP design by linking molecular structure and formulation composition to in vivo biodistribution. By curating a diverse literature-derived dataset of 476 LNP formulations and constructing an 808-dimensional molecular and compositional feature representation, we show that hepatic versus extrahepatic accumulation can be predicted with strong accuracy across distinct learning paradigms. The superior performance of random forest and XGBoost relative to logistic regression highlights the nonlinear nature of LNP biodistribution, while the competitive performance of all three models confirms that chemically meaningful descriptors contain substantial predictive information.

Beyond prediction, the central contribution of this study is interpretability. SHAP analysis and consensus feature ranking identified a relatively small set of molecular determinants that accounted for much of the model's predictive behavior. These features reveal that extrahepatic accumulation is not governed by ionizable lipid chemistry alone, but by the combined influence of ionizable-lipid surface electronics, charge distribution, molecular topology, amide and alkyl motifs, PEG/polymer-lipid hydrophobic surface features, and formulation composition. In particular, higher ionizable lipid fraction, lower PEGylated/polymer-conjugated lipid fraction, favorable IL-VSA EState7 values, PL-SlogP VSA10, and ionizable-lipid amide and unbranched alkane motifs emerged as practical design-relevant features.

These findings provide a rational framework for prioritizing LNP formulations with increased probability of extrahepatic accumulation before resource-intensive in vivo screening. More broadly, this study illustrates how interpretable machine learning can convert heterogeneous literature data into actionable nanomedicine design rules. Expanding this framework with larger organ-resolved datasets, standardized biodistribution measurements, and broader chemical diversity across helper, sterol, PEGylated, and polymer-conjugated lipids will further improve generalizability and accelerate the design of tissue-selective RNA delivery systems.

# References

[1] Young-Kook Kim. Rna therapy: rich history, various applications and unlimited future prospects. Experimental & Molecular Medicine, 54(4):455–465, 2022.   
[2] Yiran Zhu, Liyuan Zhu, Xian Wang, and Hongchuan Jin. Rna-based therapeutics: an overview and prospectus. Cell death & disease, 13(7):644, 2022.   
[3] Yi Yan, Xiao-Yu Liu, An Lu, Xiang-Yu Wang, Lin-Xia Jiang, and Jian-Cheng Wang. Non-viral vectors for rna delivery. Journal of Controlled Release, 342:241–279, 2022.   
[4] Maria L Ibba, Giuseppe Ciccone, Carla L Esposito, Silvia Catuogno, and Paloma H Giangrande. Advances in mrna non-viral delivery approaches. Advanced drug delivery reviews, 177:113930, 2021.   
[5] Khalid A Hajj, Jilian R Melamed, Namit Chaudhary, Nicholas G Lamson, Rebecca L Ball, Saigopalakrishna S Yerneni, and Kathryn A Whitehead. A potent branched-tail lipid nanoparticle enables multiplexed mrna delivery and gene editing in vivo. Nano letters, 20(7):5167–5175, 2020.   
[6] Cheng Lin, Alexander Jans, Justina Clarinda Wolters, Mohamed Ramadan Mohamed, Emiel PC Van der Vorst, Christian Trautwein, and Matthias Bartneck. Targeting ligand independent tropism of sirna-lnp by small molecules for directed therapy of liver or myeloid immune cells. Advanced healthcare materials, 13(26):2202670, 2024.   
[7] Yusuke Sato, Yoshiyuki Kinami, Kazuki Hashiba, and Hideyoshi Harashima. Different kinetics for the hepatic uptake of lipid nanoparticles between the apolipoprotein e/low density lipoprotein receptor and the n-acetyl-d-galactosamine/asialoglycoprotein receptor pathway. Journal of Controlled Release, 322:217–226, 2020.   
[8] Marius Maximilian Woitok, Miguel Eugenio Zoubek, Dennis Doleschel, Matthias Bartneck, Mohamed Ramadan Mohamed, Fabian Kießling, Wiltrud Lederle, Christian Trautwein, and Francisco Javier Cubero. Lipid-encapsulated sirna for hepatocyte-directed treatment of advanced liver disease. Cell Death & Disease, 11(5):343, 2020.   
[9] Brian Truong, Gabriella Allegri, Xiao-Bo Liu, Kristine E Burke, Xuling Zhu, Stephen D Cederbaum, Johannes Häberle, Paolo GV Martini, and Gerald S Lipshutz. Lipid nanoparticle-targeted mrna therapy as a treatment for the inherited metabolic liver disorder arginase deficiency. Proceedings of the National Academy of Sciences, 116(42):21150–21159, 2019.   
[10] Qiang Cheng, Tuo Wei, Lukas Farbiak, Lindsay T Johnson, Sean A Dilliard, and Daniel J Siegwart. Selective organ targeting (sort) nanoparticles for tissue-specific mrna delivery and crispr–cas gene editing. Nature nanotechnology, 15(4):313–320, 2020.   
[11] Huanzhen Ni, Marine ZC Hatit, Kun Zhao, David Loughrey, Melissa P Lokugamage, Hannah E Peck, Ada Del Cid, Abinaya Muralidharan, YongTae Kim, Philip J Santangelo, et al. Piperazine-derived lipid nanoparticles deliver mrna to immune cells in vivo. Nature Communications, 13(1):4766, 2022.   
[12] Zubao Gan, Melissa P Lokugamage, Marine ZC Hatit, David Loughrey, Kalina Paunovska, Manaka Sato, Ana Cristian, and James E Dahlman. Nanoparticles containing constrained phospholipids deliver mrna to liver immune cells in vivo without targeting ligands. Bioengineering & translational medicine, 5(3):e10161, 2020.   
[13] Afsane Radmand, Hyejin Kim, Jared Beyersdorf, Curtis N Dobrowolski, Ryan Zenhausern, Kalina Paunovska, Sebastian G Huayamares, Xuanwen Hua, Keyi Han, David Loughrey, et al. Cationic cholesterol-dependent lnp delivery to lung stem cells, the liver, and heart. Proceedings of the National Academy of Sciences, 121(11):e2307801120, 2024.   
[14] Min Qiu, Yan Tang, Jinjin Chen, Rachel Muriph, Zhongfeng Ye, Changfeng Huang, Jason Evans, Elizabeth P Henske, and Qiaobing Xu. Lung-selective mrna delivery of synthetic lipid nanoparticles for the treatment of pulmonary lymphangioleiomyomatosis. Proceedings of the national academy of sciences, 119(8):e2116271119, 2022.   
[15] Shuai Liu, Qiang Cheng, Tuo Wei, Xueliang Yu, Lindsay T Johnson, Lukas Farbiak, and Daniel J Siegwart. Membrane-destabilizing ionizable phospholipids for organ-selective mrna delivery and crispr–cas gene editing. Nature materials, 20(5):701–710, 2021.   
[16] Shuai Liu, Xu Wang, Xueliang Yu, Qiang Cheng, Lindsay T Johnson, Sumanta Chatterjee, Di Zhang, Sang M Lee, Yehui Sun, Ting-Chih Lin, et al. Zwitterionic phospholipidation of cationic polymers facilitates systemic mrna delivery to spleen and lymph nodes. Journal of the American Chemical Society, 143(50):21321–21330, 2021.   
[17] Afsane Radmand, Melissa P Lokugamage, Hyejin Kim, Curtis Dobrowolski, Ryan Zenhausern, David Loughrey, Sebastian G Huayamares, Marine ZC Hatit, Huanzhen Ni, Ada Del Cid, et al. The transcriptional response to lung-targeting lipid nanoparticles in vivo. Nano letters, 23(3):993–1002, 2023.   
[18] Pedro PG Guimaraes, Rui Zhang, Roman Spektor, Mingchee Tan, Amanda Chung, Margaret M Billingsley, Rakan El-Mayta, Rachel S Riley, Lili Wang, James M Wilson, et al. Ionizable lipid nanoparticles encapsulating barcoded mrna for accelerated in vivo delivery screening. Journal of Controlled Release, 316:404–417, 2019.   
[19] James E Dahlman, Kevin J Kauffman, Yiping Xing, Taylor E Shaw, Faryal F Mir, Chloe C Dlott, Robert Langer, Daniel G Anderson, and Eric T Wang. Barcoded nanoparticles for high throughput in vivo discovery of targeted therapeutics. Proceedings of the National Academy of Sciences, 114(8):2060–2065, 2017.   
[20] Camilla Hald Albertsen, Jayesh A Kulkarni, Dominik Witzigmann, Marianne Lind, Karsten Petersson, and Jens B Simonsen. The role of lipid components in lipid nanoparticles for vaccines and gene therapy. Advanced drug delivery reviews, 188:114416, 2022.

[21] Daisy Yi Ding, Yuhui Zhang, Yuan Jia, and Jiuzhi Sun. Machine learning-guided lipid nanoparticle design for mrna delivery. arXiv preprint arXiv:2308.01402, 2023.   
[22] Bowen Li, Idris O Raji, Akiva GR Gordon, Lizhuang Sun, Theresa M Raimondo, Favour A Oladimeji, Allen Y Jiang, Andrew Varley, Robert S Langer, and Daniel G Anderson. Accelerating ionizable lipid discovery for mrna delivery using machine learning and combinatorial chemistry. Nature Materials, 23(7):1002–1008, 2024.   
[23] Saeed Moayedpour, Jonathan Broadbent, Saleh Riahi, Michael Bailey, Hoa V. Thu, Dimitar Dobchev, Akshay Balsubramani, Ricardo ND Santos, Lorenzo Kogler-Anele, Alejandro Corrochano-Navarro, et al. Representations of lipid nanoparticles using large language models for transfection efficiency prediction. Bioinformatics, 40(7):btae342, 2024.   
[24] Yue Xu, Shihao Ma, Haotian Cui, Jingan Chen, Shufen Xu, Fanglin Gong, Alex Golubovic, Muye Zhou, Kevin Chang Wang, Andrew Varley, et al. Agile platform: a deep learning powered approach to accelerate lnp development for mrna delivery. Nature communications, 15(1):6305, 2024.   
[25] Yue Xu, Haotian Cui, Kuan Pang, Gen Li, Fanglin Gong, Songtao Dong, Bo Wang, and Bowen Li. Lumi-lab: A foundation model-driven autonomous platform enabling discovery of ionizable lipid designs for mrna delivery. Cell, 189(6):1620–1635, 2026.   
[26] Asal Mehradfar, Mohammad Shahab Sepehri, Jose Miguel Hernandez-Lobato, Glen S Kwon, Mahdi Soltanolkotabi, Salman Avestimehr, and Morteza Rasoulianboroujeni. Lantern: A machine learning framework for lipid nanoparticle transfection efficiency prediction. arXiv preprint arXiv:2507.03209, 2025.   
[27] Mostafa Zahed, Maryam Skafyan, and Morteza Rasoulianboroujeni. Machine learning and ranking-based evaluation for prioritizing high-potency ionizable lipids in lnp-mediated rna delivery. Algorithms, 19(5):353, 2026.   
[28] Gregory Landrum. Rdkit: Open-source cheminformatics. Online, 1, 2013.   
[29] S. Riniker and G. A. Landrum. Better informed distance geometry: Using what we know to improve conformation generation. Journal of Chemical Information and Modeling, 55(12):2562–2574, 2015.   
[30] Jerome H Friedman, Trevor Hastie, and Rob Tibshirani. Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33:1–22, 2010.   
[31] Trevor Hastie, Robert Tibshirani, Jerome Friedman, et al. The elements of statistical learning, 2009.   
[32] Leo Breiman. Random forests. Machine learning, 45(1):5–32, 2001.   
[33] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794, 2016.   
[34] Hui Zou and Trevor Hastie. Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society Series B: Statistical Methodology, 67(2):301–320, 2005.   
[35] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.   
[36] Yining Zhu, Ruochen Shen, Ivan Vuong, Rebekah A Reynolds, Melanie J Shears, Zhi-Cheng Yao, Yizong Hu, Won June Cho, Jiayuan Kong, Sashank K Reddy, et al. Multi-step screening of dna/lipid nanoparticles and co-delivery with sirna to enhance and prolong gene expression. Nature communications, 13(1):4282, 2022.   
[37] John R Silvius and Martin J Zuckermann. Interbilayer transfer of phospholipid-anchored macromolecules via monomer diffusion. Biochemistry, 32(12):3153–3161, 1993.   
[38] Sean C Semple, Troy O Harasym, Kathy A Clow, Steven M Ansell, Sandra K Klimuk, and Michael J Hope. Immunogenicity and rapid blood clearance of liposomes containing polyethylene glycol-lipid conjugates and nucleic acid. The Journal of pharmacology and experimental therapeutics, 312(3):1020–1026, 2005.   
[39] Barbara L Mui, Ying K Tam, Muthusamy Jayaraman, Steven M Ansell, Xinyao Du, Yuen Yi C Tam, Paulo JC Lin, Sam Chen, Jayaprakash K Narayanannair, Kallanthottathil G Rajeev, et al. Influence of polyethylene glycol lipid desorption rates on pharmacokinetics and pharmacodynamics of sirna lipid nanoparticles. Molecular Therapy Nucleic Acids, 2, 2013.

# Supplementary Information

Table S1: Literature sources used to construct the curated LNP dataset. The table lists the publications from which LNP formulations were extracted during dataset curation. The numbers indicate how many LNPs were extracted from each study. The final row reports the total number of LNPs included in the dataset. 

<table><tr><td>DOI</td><td>LNPs</td><td>DOI</td><td>LNPs</td></tr><tr><td>10.1039/D1BM00866H</td><td>4</td><td>10.1016/j.jconrel.2019.10.028</td><td>6</td></tr><tr><td>10.1039/D2BM00168C</td><td>3</td><td>10.1038/s42003-021-02441-2</td><td>4</td></tr><tr><td>10.1016/j.jconrel.2016.05.059</td><td>3</td><td>10.1021/acs.nanolett.5b02497</td><td>1</td></tr><tr><td>10.1016/j.ijpharm.2016.06.124</td><td>1</td><td>10.1021/acs.nanolett.0c00596</td><td>5</td></tr><tr><td>10.1002/smll.201805097</td><td>11</td><td>10.1016/j.jconrel.2013.09.027</td><td>5</td></tr><tr><td>10.1126/sciadv.abf4398</td><td>5</td><td>10.1016/j.biomaterials.2012.05.002</td><td>1</td></tr><tr><td>10.1039/C9NR05788A</td><td>11</td><td>10.1002/smll.202304378</td><td>2</td></tr><tr><td>10.1002/anie.201809055</td><td>2</td><td>10.1021/acsanm.0c01834</td><td>6</td></tr><tr><td>10.1039/D3TB00516J</td><td>3</td><td>10.1016/j.ijpharm.2022.122489</td><td>1</td></tr><tr><td>10.1002/ange.202013927</td><td>10</td><td>10.1126/sciadv.aba1028</td><td>14</td></tr><tr><td>10.1038/s41565-020-0669-6</td><td>8</td><td>10.1016/j.ymthe.2021.06.004</td><td>1</td></tr><tr><td>10.1073/pnas.2307813120</td><td>6</td><td>10.1021/jacs.2c12893</td><td>3</td></tr><tr><td>10.1073/pnas.2311276120</td><td>1</td><td>10.1126/sciadv.ade1444</td><td>2</td></tr><tr><td>10.1002/anie.202310401</td><td>65</td><td>10.1016/j.jconrel.2024.05.015</td><td>5</td></tr><tr><td>10.1016/j.jconrel.2021.01.005</td><td>3</td><td>10.1002/adfm.202303795</td><td>29</td></tr><tr><td>10.1038/s41467-020-17029-3</td><td>4</td><td>10.1021/acs.nanolett.3c03509</td><td>5</td></tr><tr><td>10.1021/acs.nanolett.3c05031</td><td>2</td><td>10.1073/pnas.2307809121</td><td>1</td></tr><tr><td>10.1038/s41467-022-35637-z</td><td>2</td><td>10.1038/s41598-024-57997-w</td><td>1</td></tr><tr><td>10.1016/j.colsurfb.2024.113980</td><td>2</td><td>10.1002/adhm.201901487</td><td>21</td></tr><tr><td>10.1016/j.jconrel.2021.11.022</td><td>4</td><td>10.1073/pnas.2020401118</td><td>3</td></tr><tr><td>10.1002/adfm.202312038</td><td>6</td><td>10.1016/j.nantod.2024.102325</td><td>2</td></tr><tr><td>10.1002/adma.201606944</td><td>1</td><td>10.34133/bmr.0017</td><td>2</td></tr><tr><td>10.1016/j.jconrel.2020.06.030</td><td>10</td><td>10.1016/j.jconrel.2024.04.018</td><td>3</td></tr><tr><td>10.1016/j.jconrel.2022.03.046</td><td>18</td><td>10.1016/j.omtn.2020.01.018</td><td>3</td></tr><tr><td>10.1021/acs.nanolett.2c03234</td><td>3</td><td>10.1002/anie.202302676</td><td>1</td></tr><tr><td>10.1038/s41563-020-00886-0</td><td>31</td><td>10.1016/j.jconrel.2024.05.036</td><td>1</td></tr><tr><td>10.1021/acsnano.4c01171</td><td>3</td><td>10.1021/jacs.3c09143</td><td>16</td></tr><tr><td>10.1002/adma.202302901</td><td>12</td><td>10.1038/s41467-024-45422-9</td><td>4</td></tr><tr><td>10.1002/smll.202303568</td><td>3</td><td>10.1039/D1TB00736J</td><td>1</td></tr><tr><td>10.1021/jacs.3c05574</td><td>4</td><td>10.1021/jacs.4c04565</td><td>4</td></tr><tr><td>10.1021/acs.nanolett.4c01235</td><td>5</td><td>10.1002/adma.202303614</td><td>1</td></tr><tr><td>10.1039/D1BM01454D</td><td>14</td><td>10.1002/anie.202013927</td><td>20</td></tr><tr><td>10.1002/adma.201805308</td><td>4</td><td>10.1002/jbm.a.37705</td><td>5</td></tr><tr><td>10.1038/s41565-023-01404-4</td><td>1</td><td>10.1002/advs.202202556</td><td>1</td></tr><tr><td>10.1016/j.ymthe.2019.03.001</td><td>1</td><td>10.1021/acsnano.3c06225</td><td>1</td></tr><tr><td>10.1039/D2BM01846B</td><td>3</td><td>10.1038/mt.2011.141</td><td>3</td></tr><tr><td>10.1093/nsr/nwae135</td><td>3</td><td>10.1002/anie.202008082</td><td>4</td></tr><tr><td>10.1101/2022.12.22.521490</td><td>3</td><td>10.1021/acscentsci.4c00071</td><td>1</td></tr><tr><td>10.3390/ijms25031388</td><td>3</td><td>10.1007/s12274-018-2082-0</td><td>1</td></tr><tr><td>10.1021/acsnano.1c04456</td><td>1</td><td>10.1002/smll.202105832</td><td>1</td></tr><tr><td>10.1038/s41557-024-01557-2</td><td>6</td><td></td><td></td></tr></table>

Total 476

![](images/61d266978f2ac72ce0dc77ef8f21540c0fde7a2ada1759171e6c9ef2fbebbbaa.jpg)

<details>
<summary>line</summary>

| Predicted Probability | Observed Positive Rate | Count |
| --------------------- | ---------------------- | ----- |
| 0.0                   | 0.0                    | 20    |
| 0.1                   | 0.2                    | 8     |
| 0.2                   | 0.1                    | 6     |
| 0.3                   | 0.5                    | 4     |
| 0.4                   | 0.7                    | 3     |
| 0.5                   | 0.8                    | 2     |
| 0.6                   | 0.9                    | 6     |
| 0.7                   | 0.9                    | 4     |
| 0.8                   | 0.9                    | 8     |
| 0.9                   | 0.9                    | 2     |
| 1.0                   | 0.9                    | 14    |
</details>

(a)

![](images/7b47663c2969c611c5ca663da1aec17e2bd9de83968131db9dcf9ffebf1b9d1d.jpg)

<details>
<summary>bar_line</summary>

| Predicted Probability | Observed Positive Rate (Ideal) | Observed Positive Rate (Random Forest) | Count |
| --------------------- | ------------------------------ | -------------------------------------- | ----- |
| 0.0                   | 0.0                            | 0.1                                    | 15    |
| 0.2                   | 0.2                            | 0.3                                    | 7     |
| 0.4                   | 0.4                            | 0.6                                    | 5     |
| 0.6                   | 0.6                            | 0.45                                   | 3     |
| 0.8                   | 0.8                            | 1.0                                    | 6     |
| 1.0                   | 1.0                            | 0.9                                    | 15    |
</details>

(b)

![](images/6b46628cf3875b99049716663e6bb47c8c6fe56ed3119820e3a3083b3360a7a6.jpg)

<details>
<summary>bar_line</summary>

| Predicted Probability | Observed Positive Rate (Ideal) | Observed Positive Rate (XGBoost) | Count |
| --------------------- | ------------------------------ | -------------------------------- | ----- |
| 0.0                   | 0.0                            | 0.2                              | 25    |
| 0.2                   | 0.2                            | 0.2                              | 5     |
| 0.4                   | 0.4                            | 0.1                              | 2     |
| 0.6                   | 0.6                            | 0.9                              | 3     |
| 0.8                   | 0.8                            | 0.8                              | 5     |
| 1.0                   | 1.0                            | 1.0                              | 20    |
</details>

(c)   
Figure S1: Calibration analysis of ML models predicting LNP biodistribution. Calibration curves for (a) logistic regression, (b) random forest, and (c) XGBoost models are shown. The upper panels compare observed positive rates with predicted probabilities across probability bins; the dashed diagonal line indicates ideal calibration where predicted probabilities perfectly match observed frequencies. The lower panels show histograms of predicted probabilities for the corresponding models, illustrating the distribution of model confidence across the test samples.

Table S2: Calibration performance of the evaluated machine learning models trained using the full feature set. Lower Brier scores indicate better calibration, corresponding to better agreement between predicted probabilities and observed outcomes. 

<table><tr><td>Model</td><td>Brier Score</td></tr><tr><td>logistic regression</td><td>0.160</td></tr><tr><td>random forest</td><td>0.147</td></tr><tr><td>XGBoost</td><td>0.144</td></tr></table>

![](images/13be688baf49e666d78fc7d75585931983858e83f9450f4aa5d07bb67647df15.jpg)

<details>
<summary>line</summary>

| Number of Selected Features (K) | Logistic Regression (CV) | Random Forest (CV) | XGBoost (CV) | Logistic Regression (Test) | Random Forest (Test) | XGBoost (Test) |
| ------------------------------- | ------------------------ | ------------------ | ------------ | -------------------------- | -------------------- | -------------- |
| 5                               | 0.67                     | 0.79               | 0.80         | 0.68                       | 0.82                 | 0.81           |
| 10                              | 0.71                     | 0.85               | 0.84         | 0.71                       | 0.86                 | 0.85           |
| 20                              | 0.76                     | 0.86               | 0.85         | 0.72                       | 0.87                 | 0.86           |
| 40                              | 0.79                     | 0.85               | 0.86         | 0.78                       | 0.87                 | 0.86           |
| All                             | 0.83                     | 0.88               | 0.87         | 0.84                       | 0.87                 | 0.87           |
</details>

Figure S2: Model performance as a function of the number of selected features. Model performance (ROC–AUC) is shown for logistic regression, random forest, and XGBoost models using different numbers of selected features (K = 5, 10, 20, 40, and all features). Solid lines represent mean cross-validation performance, while dashed lines denote test performance. Error bars indicate the standard deviation across cross-validation folds. Model performance generally improves as additional features are incorporated and stabilizes when larger feature sets are used.

Table S3: Top 50 consensus-ranked features identified across ML models. Feature rankings from logistic regression (LR), random forest (RF), and XGBoost (XGB) are shown along with the median rank used to aggregate model-specific rankings. The consensus rank represents the final ordering obtained after sorting features by median rank and resolving ties using the average normalized SHAP importance values. 

<table><tr><td>Consensus Rank</td><td>Feature</td><td>LR Rank</td><td>RF Rank</td><td>XGB Rank</td><td>Median Rank</td></tr><tr><td>1</td><td>IL-VSA EState5</td><td>17</td><td>5</td><td>4</td><td>5</td></tr><tr><td>2</td><td>IL %</td><td>59</td><td>4</td><td>5</td><td>5</td></tr><tr><td>3</td><td>IL-VSA EState7</td><td>6</td><td>20</td><td>1</td><td>6</td></tr><tr><td>4</td><td>PL %</td><td>36</td><td>6</td><td>3</td><td>6</td></tr><tr><td>5</td><td>SL %</td><td>1</td><td>7</td><td>9</td><td>7</td></tr><tr><td>6</td><td>IL-MinEStateIndex</td><td>148</td><td>1</td><td>7</td><td>7</td></tr><tr><td>7</td><td>IL-BertzCT</td><td>735</td><td>9</td><td>6</td><td>9</td></tr><tr><td>8</td><td>IL-BalabanJ</td><td>7</td><td>35</td><td>15</td><td>15</td></tr><tr><td>9</td><td>IL-NumAmideBonds</td><td>15.5</td><td>2</td><td>36</td><td>15.5</td></tr><tr><td>10</td><td>IL-fr amide</td><td>15.5</td><td>11</td><td>92</td><td>15.5</td></tr><tr><td>11</td><td>IL-PEOE VSA12</td><td>8</td><td>18</td><td>525.5</td><td>18</td></tr><tr><td>12</td><td>IL-fr unbrch alkane</td><td>12</td><td>69</td><td>22</td><td>22</td></tr><tr><td>13</td><td>IL-PEOE VSA1</td><td>38</td><td>8</td><td>23</td><td>23</td></tr><tr><td>14</td><td>IL-EState VSA4</td><td>735</td><td>24</td><td>12</td><td>24</td></tr><tr><td>15</td><td>PL-SlogP VSA10</td><td>26</td><td>117</td><td>17</td><td>26</td></tr><tr><td>16</td><td>IL-HallKierAlpha</td><td>735</td><td>14</td><td>26</td><td>26</td></tr><tr><td>17</td><td>IL-FractionCSP3</td><td>27</td><td>84</td><td>13</td><td>27</td></tr><tr><td>18</td><td>IL-AvgIpc</td><td>161.5</td><td>27</td><td>24</td><td>27</td></tr><tr><td>19</td><td>IL-VSA EState8</td><td>735</td><td>25</td><td>27</td><td>27</td></tr><tr><td>20</td><td>IL-PEOE VSA8</td><td>28</td><td>26</td><td>87</td><td>28</td></tr><tr><td>21</td><td>IL-SPS</td><td>30</td><td>36</td><td>14</td><td>30</td></tr><tr><td>22</td><td>IL-EState VSA2</td><td>80</td><td>30</td><td>21</td><td>30</td></tr><tr><td>23</td><td>IL-BCUT2D MRHI</td><td>5</td><td>31</td><td>74</td><td>31</td></tr><tr><td>24</td><td>IL-SMR VSA3</td><td>89</td><td>3</td><td>31</td><td>31</td></tr><tr><td>25</td><td>IL-BCUT2D MWHI</td><td>2</td><td>33</td><td>69</td><td>33</td></tr><tr><td>26</td><td>IL-EState VSA5</td><td>175</td><td>34</td><td>28</td><td>34</td></tr><tr><td>27</td><td>IL-SlogP VSA1</td><td>117</td><td>13</td><td>35</td><td>35</td></tr><tr><td>28</td><td>IL-PEOE VSA6</td><td>35</td><td>101</td><td>34</td><td>35</td></tr><tr><td>29</td><td>IL-SlogP VSA3</td><td>735</td><td>16</td><td>37</td><td>37</td></tr><tr><td>30</td><td>IL-BCUT2D MRLOW</td><td>64</td><td>39</td><td>19</td><td>39</td></tr><tr><td>31</td><td>IL-VSA EState2</td><td>67</td><td>12</td><td>41</td><td>41</td></tr><tr><td>32</td><td>IL-MaxPartialCharge</td><td>77.5</td><td>42</td><td>40</td><td>42</td></tr><tr><td>33</td><td>IL-BCUT2D MWLOW</td><td>68</td><td>44</td><td>11</td><td>44</td></tr><tr><td>34</td><td>IL-Phi</td><td>103</td><td>15</td><td>44</td><td>44</td></tr><tr><td>35</td><td>IL-NOCount</td><td>46</td><td>40</td><td>525.5</td><td>46</td></tr><tr><td>36</td><td>IL-Chi4n</td><td>735</td><td>41</td><td>46</td><td>46</td></tr><tr><td>37</td><td>IL-BCUT2D LOGPLOW</td><td>138</td><td>22</td><td>47</td><td>47</td></tr><tr><td>38</td><td>IL-BCUT2D CHGLO</td><td>735</td><td>48</td><td>10</td><td>48</td></tr><tr><td>39</td><td>IL-Kappa3</td><td>735</td><td>51</td><td>20</td><td>51</td></tr><tr><td>40</td><td>IL-SMR VSA5</td><td>735</td><td>52</td><td>50</td><td>52</td></tr><tr><td>41</td><td>IL-MaxAbsEStateIndex</td><td>164.5</td><td>54</td><td>33</td><td>54</td></tr><tr><td>42</td><td>IL-EState VSA7</td><td>324</td><td>55</td><td>25</td><td>55</td></tr><tr><td>43</td><td>IL-BCUT2D LOGPHI</td><td>735</td><td>19</td><td>56</td><td>56</td></tr><tr><td>44</td><td>IL-EState VSA3</td><td>735</td><td>56</td><td>45</td><td>56</td></tr><tr><td>45</td><td>IL-EState VSA1</td><td>735</td><td>10</td><td>57</td><td>57</td></tr><tr><td>46</td><td>IL-VSA EState9</td><td>84</td><td>58</td><td>18</td><td>58</td></tr><tr><td>47</td><td>IL-BCUT2D CHGHI</td><td>75</td><td>17</td><td>60</td><td>60</td></tr><tr><td>48</td><td>IL-FpDensityMorgan3</td><td>63</td><td>60</td><td>68</td><td>63</td></tr><tr><td>49</td><td>IL-SlogP VSA5</td><td>735</td><td>63</td><td>39</td><td>63</td></tr><tr><td>50</td><td>HL %</td><td>735</td><td>64</td><td>53</td><td>64</td></tr></table>

Table S4: Extended performance comparison of ML models across different feature-set sizes for LNP classification. This table extends Table 1 by including additional reduced feature sets containing the top 40 and top 5 features. Models were evaluated using the full feature set (All) and reduced feature sets containing the top 40, 20, 10, and 5 features. Cross-validation performance is reported as the mean AUC ± standard deviation across folds, and held-out test performance is reported using AUC, accuracy, recall, and F1-score. The best AUC values within each feature-set block are highlighted in bold. 

<table><tr><td>Feature Set</td><td>Model</td><td>CV AUC</td><td>Test AUC</td><td>Accuracy</td><td>Recall</td><td>F1-score</td></tr><tr><td rowspan="3">All</td><td>logistic regression</td><td>0.826±0.054</td><td>0.839</td><td>0.792</td><td>0.792</td><td>0.792</td></tr><tr><td>random forest</td><td>0.882±0.036</td><td>0.866</td><td>0.802</td><td>0.792</td><td>0.800</td></tr><tr><td>XGBoost</td><td>0.875±0.034</td><td>0.874</td><td>0.854</td><td>0.854</td><td>0.854</td></tr><tr><td rowspan="3">40</td><td>logistic regression</td><td>0.787±0.055</td><td>0.775</td><td>0.698</td><td>0.667</td><td>0.688</td></tr><tr><td>random forest</td><td>0.846±0.047</td><td>0.872</td><td>0.771</td><td>0.792</td><td>0.776</td></tr><tr><td>XGBoost</td><td>0.848±0.048</td><td>0.858</td><td>0.781</td><td>0.812</td><td>0.788</td></tr><tr><td rowspan="3">20</td><td>logistic regression</td><td>0.759±0.056</td><td>0.724</td><td>0.615</td><td>0.583</td><td>0.602</td></tr><tr><td>random forest</td><td>0.854±0.050</td><td>0.866</td><td>0.781</td><td>0.812</td><td>0.788</td></tr><tr><td>XGBoost</td><td>0.852±0.048</td><td>0.851</td><td>0.781</td><td>0.812</td><td>0.788</td></tr><tr><td rowspan="3">10</td><td>logistic regression</td><td>0.707±0.053</td><td>0.714</td><td>0.729</td><td>0.750</td><td>0.735</td></tr><tr><td>random forest</td><td>0.842±0.054</td><td>0.859</td><td>0.760</td><td>0.792</td><td>0.768</td></tr><tr><td>XGBoost</td><td>0.852±0.048</td><td>0.847</td><td>0.771</td><td>0.771</td><td>0.771</td></tr><tr><td rowspan="3">5</td><td>logistic regression</td><td>0.670±0.046</td><td>0.677</td><td>0.615</td><td>0.500</td><td>0.565</td></tr><tr><td>random forest</td><td>0.795±0.070</td><td>0.819</td><td>0.740</td><td>0.771</td><td>0.747</td></tr><tr><td>XGBoost</td><td>0.807±0.063</td><td>0.805</td><td>0.719</td><td>0.771</td><td>0.733</td></tr></table>