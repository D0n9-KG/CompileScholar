# OPEN

# Bioinformatic analysis of peripheral blood RNA-sequencing sensitively detects the cause of late graft loss following overt hyperglycemia in pig-to-nonhuman primate islet xenotransplantation

Hyun-Je Kim $^{1,2,3,12,14}$ , Ji Hwan Moon $^{4,13,14}$ , Hyunwoo Chung $^{1,2,3,14}$ , Jun-Seop Shin $^{1}$ , Bongi Kim $^{2}$ , Jong-Min Kim $^{1}$ , Jung-Sik Kim $^{1}$ , Il-Hee Yoon $^{1}$ , Byoung-Hoon Min $^{1}$ , Seong-Jun Kang $^{1,2,3}$ , Yong-Hee Kim $^{1}$ , Kyuri Jo $^{5}$ , Joungmin Choi $^{6}$ , Heejoon Chae $^{6}$ , Won-Woo Lee $^{1,2,3}$ , Sun Kim $^{4,7,8*}$ & Chung-Gyu Park $^{1,2,3,9,10,11*}$

Clinical islet transplantation has recently been a promising treatment option for intractable type 1 diabetes patients. Although early graft loss has been well studied and controlled, the mechanisms of late graft loss largely remains obscure. Since long-term islet graft survival had not been achieved in islet xenotransplantation, it has been impossible to explore the mechanism of late islet graft loss. Fortunately, recent advances where consistent long-term survival ( $\geq$ 6 months) of adult porcine islet grafts was achieved in five independent, diabetic nonhuman primates (NHPs) enabled us to investigate on the late graft loss. Regardless of the conventional immune monitoring methods applied in the post-transplant period, the initiation of late graft loss could rarely be detected before the overt graft loss observed via uncontrolled blood glucose level. Thus, we retrospectively analyzed the gene expression profiles in 2 rhesus monkey recipients using peripheral blood RNA-sequencing (RNA-seq) data to find out the potential cause(s) of late graft loss. Bioinformatic analyses showed that highly relevant immunological pathways were activated in the animal which experienced late graft failure. Further connectivity analyses revealed that the activation of T cell signaling pathways was the most prominent, suggesting that T cell-mediated graft rejection could be the cause of the late-phase islet loss. Indeed, the porcine islets in the biopsied monkey liver samples were heavily infiltrated with CD3 $^{+}$ T cells. Furthermore, hypothesis test using a computational experiment reinforced our conclusion. Taken

$^{1}$ Xenotransplantation Research Center, Seoul National University College of Medicine, Seoul, 03080, Republic of Korea. $^{2}$ Department of Microbiology and Immunology, Seoul National University College of Medicine, Seoul, 03080, Republic of Korea. $^{3}$ Department of Biomedical Sciences, Seoul National University Graduate School, Seoul, 03080, Republic of Korea. $^{4}$ Interdisciplinary Program in Bioinformatics, Seoul National University, Seoul, 08826, Republic of Korea. $^{5}$ Department of Computer Engineering, Chungbuk National University, Cheongju, 28644, Republic of Korea. $^{6}$ Division of Computer Science, Sookmyung Women's University, Seoul, 04310, Republic of Korea. $^{7}$ Bioinformatics Institute, Department of Computer Science and Engineering, Seoul National University, Seoul, 08826, Republic of Korea. $^{8}$ Department of Computer Science & Engineering, Seoul National University, Seoul, 08826, Republic of Korea. $^{9}$ Cancer Research Institute, Seoul National University College of Medicine, Seoul, 03080, Republic of Korea. $^{10}$ Institute of Endemic Diseases, Seoul National University College of Medicine, Seoul, 03080, Republic of Korea. $^{11}$ Biomedical Research Institute, Seoul National University Hospital, Seoul, 03080, Republic of Korea. $^{12}$ Present address: Department of Dermatology and the Laboratory of Inflammatory Skin Diseases, Icahn School of Medicine at Mount Sinai, New York, NY, 10029, USA. $^{13}$ Present address: Department of Biological Sciences, University at Buffalo, Buffalo, NY, 14260, USA. $^{14}$ These authors contributed equally: Hyun-Je Kim, Ji Hwan Moon and Hyunwoo Chung. \*email: sunkim.bioinfo@snu.ac.kr; chgpark@snu.ac.kr

together, we suggest that bioinformatics analyses with peripheral blood RNA-seq could unveil the cause of insidious late islet graft loss.

The Edmonton protocol was introduced in 2000 $^{1}$ , and since then human pancreatic islet transplantation has become an established treatment option for type 1 diabetic patients who frequently experienced fatal hypoglycemic unawareness $^{2}$ . However, over half of the patients transplanted with human islets returned to the insulin-dependent, diabetic state within 5 years $^{3,4}$ . The causes for this late islet graft loss are still controversial. Previous reports encompass a higher rate of islet apoptosis due to endoplasmic reticulum (ER) stress $^{5,6}$ , hypoxia $^{7,8}$ in end-portal venules within the liver, and recurrent autoimmunity $^{9}$ . In addition, there were evidences of metabolic deterioration due to lipid accumulated around the islets (lipotoxicity) $^{10,11}$ and toxicity of immunosuppressive drugs $^{12,13}$ , which could all result in graft loss. Last but not least, insufficient immunosuppression could also be an important cause of chronic islet loss especially due to antibody-mediated rejection processes $^{14}$ . However, none of the above could clearly and single-handedly explain the exact causes of chronic islet graft loss.

Recently, we reported consistent long-term ( $\geq$ 6 months) porcine islet graft survivals in five independent monkeys $^{15}$ . This unique opportunity allowed us to examine how the porcine islets are lost in the late phase of islet xenotransplantation. Here, we selected two monkeys with the same immunosuppressive regimen to analyze the cause of late graft loss in islet xenotransplantation: one (R051) had stable graft function for the entire follow-up periods and the other (R080) lost graft function around 160 days post-transplantation (DPT). Peripheral blood RNA-seq and subsequent bioinformatics analyses using Time-series RNA-seq analysis package (TRAP) $^{16}$ hinted on the possibility of immune rejection in R080. Further in silico analyses focused on the interactions of graft loss period-related activated pathways (GLPAPs) proposed that lymphocytes- or platelet-mediated rejection might have been the cause of late graft loss.

# Results

Peripheral blood RNA sequencing. R051 had shown complete normoglycemia and glucose disposal capacity for the entire follow-up periods, whereas the other (R080) exhibited relatively early hyperglycemia around DPT160, suggesting a graft failure. Intravenous glucose tolerance test (IVGTT) had shown that the porcine islet graft loss in R080 was in progress between DPT120 and 180 (Fig. 1a\~d, processed from published data $^{15}$ ). However, vital signs and routine laboratory examinations including complete blood cell count (CBC), liver function test (LFT), C-reactive protein (CRP), kidney function test (blood urine nitrogen/creatinine), electrolyte panel (sodium, potassium and chloride), lipase and amylase had shown no abnormal findings in both of the monkeys (data not shown). Also, monitoring of peripheral blood lymphocyte subsets by flow cytometry and ELISPOT (Supplementary Fig. S1), and titer of donor-specific antibody by enzyme-linked immunosorbent assay (ELISA) $^{15}$ had not revealed any marked changes. Since recent report showed gene expression perturbation in peripheral blood could reflect graft site event $^{17,18}$ , we performed RNA sequencing with the archives of whole blood samples taken at four different time points from graft-losing R080 vs. graft-stable R051 (Fig. 1e) to explore the cause of chronic islet loss happened in R080.

Graft loss period-related activated pathways (GLPAPs) defined by TRAP. After confirming the validity of RNA-seq data, we used TRAP $^{16}$ to determine which pathways had played roles in the graft loss process. Because $t_{2}$ and $t_{3}$ represent the graft-maintaining and the graft-losing period, respectively in R080, we focused on these time points and selected pathways as follows: i) select up-regulated pathways with p-value under 0.05 from the results yielded by TRAP comparing $t_{3}$ and $t_{2}$ in R080, ii) select up-regulated pathways with p-value under 0.05 from the results yielded by comparing R080 and R051 at $t_{3}$ and then take the pathways belonging to the intersection of those sets (Supplementary Fig. S2). As a result, we could obtain 59 pathways among 287 pathways in Kyoto Encyclopedia of Genes and Genomes (KEGG) $^{19}$ Rhesus database and these pathways were named GLPAPs as can be seen in Table 1.

After obtaining 59 of GLPAPs, we were able to calculate p-values for each 'category' of the pathways using Fisher's exact test to determine how significantly GLPAPs were enriched in each category. To calculate p-values, we constructed a contingency table with two variables: GLPAP and category. Each cell of the table was filled by the number of the pathways according to the standard if the pathway belongs to the category or not and if the pathway is GLPAP or not (Supplementary Fig. S3). The p-values for each category are shown in Table 2. The most enriched category was found to be 'immune system,' although we had not found significant perturbation of immunological parameters in routine immune monitoring system. This finding strongly implied that immunological responses were somehow activated and ongoing during $t_3$ in R080 compared to $t_2$ in R080 and corresponding time points in R051.

Pathway interaction network analisys. Even though we found that the pathways categorized as 'immune system' were enriched mostly after GLPAP filtering, we were not able to specify pathways which had been potentially responsible for late graft loss. Because biological pathways usually function in a cooperative manner by constituting a network, understanding the network of pathways can provide the insight about which pathways are important in a given condition. Therefore, it would be desirable to analyze the network of the pathways to find out the most interacting pathways to induce the late graft loss among GLPAPs. To this end, we constructed a pathway interaction network of GLPAPs using PINTnet $^{20}$ . There were 52 pathways out of 59 GLPAPs connected by 225 edges in the network (Fig. 2). We calculated closeness centrality for every node and used degree information to analyze which pathways had played a central role in the network to induce the biological response at $t_{3}$ of R080. We focused only on the pathways belonging to the 'immune system' because 'immune system' was

![](images/7ad46faba7489c056085956cd84704861f606028af18c2a13e473ec6202fdabe.jpg)

<details>
<summary>line</summary>

| Time (day) | Blood glucose levels (mg/dl) |
| ---------- | ---------------------------- |
| 0          | 600                          |
| 50         | 100                          |
| 100        | 100                          |
| 150        | 100                          |
| 200        | 100                          |
| 250        | 100                          |
| 300        | 100                          |
| 350        | 100                          |
</details>

![](images/4e55ceb2214cdbecf937a3d75d038e015c166b86f7c8a474e9e8028e7655f335.jpg)

<details>
<summary>line</summary>

| Time (day) | Blood glucose levels (mg/dl) |
| ---------- | ---------------------------- |
| 0          | 400                          |
| 50         | 100                          |
| 100        | 100                          |
| 150        | 100                          |
| 200        | 400                          |
</details>

![](images/23e95835500d6583142bfcdfd2db45435460e598023266212535afde974c9b7a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["RNA seq"] --> B["R080 (case)"]
    B --> C["t1: D-13"]
    B --> D["t2: D+27"]
    C --> E["t1: D-8"]
    D --> F["t3: D+168"]
    E --> G["t4: D+205"]
    F --> H["t3: D+166"]
    G --> I["t4: D+221"]
    J["Islet Loss (IVGTT)"] --> K["D+120"]
    J --> L["D+180"]
```
</details>

![](images/ee7db9513b1977e758510856ca6ed6f4b74f932b7e497e5a0495270b50f9b7c8.jpg)

<details>
<summary>line</summary>

| Time(min) | Non-DM | DM   | D+29 | D+62 | D+120 | D+180 |
| --------- | ------ | ---- | ---- | ---- | ----- | ----- |
| 0         | 400    | 400  | 400  | 400  | 400   | 400   |
| 50        | 200    | 350  | 300  | 250  | 200   | 150   |
| 100       | 100    | 350  | 250  | 200  | 150   | 100   |
| 150       | 50     | 350  | 200  | 150  | 100   | 50    |
</details>

![](images/51cf78a963dffc66d16b4cc746b6016754f4a497e31cb7014ecf55a99614614c.jpg)

<details>
<summary>line</summary>

| Time(min) | Non-DM | DM   | D+34 | D+61 | D+120 | D+180 |
| --------- | ------ | ---- | ---- | ---- | ----- | ----- |
| 0         | 200    | 550  | 300  | 250  | 200   | 450   |
| 50        | 100    | 400  | 150  | 100  | 100   | 350   |
| 100       | 50     | 350  | 100  | 50   | 50    | 300   |
| 150       | 50     | 300  | 100  | 50   | 50    | 250   |
</details>

Figure 1. Graft function and experimental scheme. (a,b) Blood glucose levels of R051 and R080. R080 showed gradual increase of blood glucose level around DPT 150. (c,d) IVGTT results of R051 and R080. Between DPT 120 and 180, R080 showed prominent glucose intolerance. (e) Sampling time point for RNA-seq. Whole blood archives were used for RNA-seq. ( $t_{1}$ : before transplantation, $t_{2}$ : one month after transplantation, $t_{3}$ : immediate after increase of blood glucose in R080 and corresponding time point for R051, $t_{4}$ : after overt hyperglycemia in R080 and corresponding time point for R051).

the most enriched category as mentioned in the previous section. The pathways with the closeness centrality value and degree higher than the average closeness centrality value and the average degree of all the nodes in the network were considered meaningful. Among eight pathways of 'immune system,' three met the criteria and these pathways were T cell receptor signaling pathway, B cell receptor signaling pathway, and platelet activation. The pathways are shown in Table 3.

T cell-mediated immune rejection confirmed by biopsy. The results from bioinformatics analyses on RNA-seq suggested that T cell-mediated immune rejection toward the porcine islets had been in progress at $t_{3}$ of R080. We wanted to confirm whether our analysis reflected real biological processes. Thus, we collected liver biopsy samples at DPT184 from R080 from the archives and examined graft histology by immunohistochemistry. Indeed, we found that insulin-positive islet grafts were heavily infiltrated by mostly CD3 $^{+}$ T cells (Fig. 3a). Moreover, immunofluorescence staining also showed that porcine islet grafts had been positive for fibrinogen and monkey IgG (Fig. 3b), the result which was in parallel with the pathway interaction network analysis. Because we did not find any noticeable change in peripheral blood immune cell phenotyping, antibody titers, ELISPOT analysis, and other routine laboratory tests, we concluded that bioinformatics analyses on peripheral blood RNA-seq could only gave us information on whether immunological reaction in response to the graft was in progress or not and furthermore, which biological pathways would be activated during this process in the transplant recipient.

Hypothesis evaluation using network propagation. To reconfirm our findings, we sought to computationally test each hypothesis which could explain the islet graft loss. We selected five hypotheses that are known to cause the late graft loss $^{14}$ . Those were ER stress $^{5,21-23}$ , islet exhaustion $^{24}$ , lipotoxicity $^{10,25-27}$ , chronic graft rejection $^{28,29}$ , and toxicity of immunosuppressants $^{12}$ . To evaluate the five hypotheses, we designed and performed a computational experiment: the rationale behind the experiment was that if a hypothesis had been the cause of the late graft loss and the genes related to the hypothesis had been important, the global effects of the genes of the hypothesis should have been similar to the gene expression profile that we measured. To measure the global effects of the genes, we used the network propagation algorithm $^{30}$ . The evaluation process was as follows: The five hypotheses-related seed genes were collected, and a differentially expressed gene (DEG) profile was established. Then, a protein-protein interaction (PPI) network was mapped with the seed genes and the global effects were measured using network propagation. After the genes were ranked under each hypothesis, Pearson's correlation analysis was performed between the DEG profile ranks and the ranks calculated by the network propagation. This was to test which of the five hypotheses represented by the seed genes produced the gene expression profile similar to the DEG profile we measured. In other words, we tried to see how much the participation of genes in the actual biological process that had induced the graft loss coincided with the perturbation in the expression of genes in the given condition. Furthermore, we performed random simulations for 1000 times and calculated empirical p-values to test the significance of the coefficient as shown in Methods. The result is shown in Table 4 and we were able to see that the correlation coefficients of 'chronic graft rejection' was the highest and most significant. In addition, we carried out the same process for top 100 genes of network propagation results for each hypothesis. As shown in Table 5, chronic graft rejection was the highest in terms of the coefficient. This result suggested and supported that late-phase graft loss reflected by the condition-specific changes in gene expression of R080 was explained the best by chronic graft rejection.

<table><tr><td>Pathway</td><td>Name</td><td>Category</td></tr><tr><td>mcc04062</td><td>Chemokine signaling pathway</td><td rowspan="10">Immune system</td></tr><tr><td>mcc04611</td><td>Platelet activation</td></tr><tr><td>mcc04620</td><td>Toll-like receptor signaling pathway</td></tr><tr><td>mcc04621</td><td>NOD-like receptor signaling pathway</td></tr><tr><td>mcc04623</td><td>Cytosolic DNA-sensing pathway</td></tr><tr><td>mcc04650</td><td>Natural killer cell mediated cytotoxicity</td></tr><tr><td>mcc04660</td><td>T cell receptor signaling pathway</td></tr><tr><td>mcc04662</td><td>B cell receptor signaling pathway</td></tr><tr><td>mcc04664</td><td>Fc epsilon RI signaling pathway</td></tr><tr><td>mcc04670</td><td>Leukocyte transendothelial migration</td></tr><tr><td>mcc04010</td><td>MAPK signaling pathway</td><td rowspan="10">Signal transduction</td></tr><tr><td>mcc04012</td><td>ErbB signaling pathway</td></tr><tr><td>mcc04022</td><td>cGMP-PKG signaling pathway</td></tr><tr><td>mcc04064</td><td>NF-kappa B signaling pathway</td></tr><tr><td>mcc04068</td><td>FoxO signaling pathway</td></tr><tr><td>mcc04070</td><td>Phosphatidylinositol signaling system</td></tr><tr><td>mcc04152</td><td>AMPK signaling pathway</td></tr><tr><td>mcc04370</td><td>VEGF signaling pathway</td></tr><tr><td>mcc04630</td><td>Jak-STAT signaling pathway</td></tr><tr><td>mcc04668</td><td>TNF signaling pathway</td></tr><tr><td>mcc04910</td><td>Insulin signaling pathway</td><td rowspan="6">Endocrine system</td></tr><tr><td>mcc04915</td><td>Estrogen signaling pathway</td></tr><tr><td>mcc04917</td><td>Prolactin signaling pathway</td></tr><tr><td>mcc04918</td><td>Thyroid hormone synthesis</td></tr><tr><td>mcc04919</td><td>Thyroid hormone signaling pathway</td></tr><tr><td>mcc04921</td><td>Oxytocin signaling pathway</td></tr><tr><td>mcc03013</td><td>RNA transport</td><td>Translation</td></tr><tr><td>mcc04210</td><td>Apoptosis</td><td>Cell growth and death</td></tr><tr><td>mcc05211</td><td>Renal cell carcinoma</td><td rowspan="9">Cancers: Specific types</td></tr><tr><td>mcc05212</td><td>Pancreatic cancer</td></tr><tr><td>mcc05213</td><td>Endometrial cancer</td></tr><tr><td>mcc05214</td><td>Glioma</td></tr><tr><td>mcc05215</td><td>Prostate cancer</td></tr><tr><td>mcc05219</td><td>Bladder cancer</td></tr><tr><td>mcc05220</td><td>Chronic myeloid leukemia</td></tr><tr><td>mcc05221</td><td>Acute myeloid leukemia</td></tr><tr><td>mcc05223</td><td>Non-small cell lung cancer</td></tr><tr><td>mcc04141</td><td>Protein processing in endoplasmic reticulum</td><td>Folding, sorting and degradation</td></tr><tr><td>mcc04320</td><td>Dorso-ventral axis formation</td><td rowspan="2">Development</td></tr><tr><td>mcc04380</td><td>Osteoclast differentiation</td></tr><tr><td>mcc04540</td><td>Gap junction</td><td>Cellular communication</td></tr><tr><td>mcc04810</td><td>Regulation of actin cytoskeleton</td><td>Cell motility</td></tr><tr><td>mcc04961</td><td>Endocrine and other factor-regulated calcium reabsorption</td><td>Excretory system</td></tr><tr><td>mcc04722</td><td>Neurotrophin signaling pathway</td><td rowspan="2">Nervous system</td></tr><tr><td>mcc04725</td><td>Cholinergic synapse</td></tr><tr><td>mcc04060</td><td>Cytokine-cytokine receptor interaction</td><td>Signaling molecules and interaction</td></tr><tr><td>mcc05142</td><td>Chagas disease (American trypanosomiasis)</td><td rowspan="3">Infectious diseases: Parasitic</td></tr><tr><td>mcc05143</td><td>African trypanosomiasis</td></tr><tr><td>mcc05144</td><td>Malaria</td></tr><tr><td>mcc05161</td><td>Hepatitis B</td><td rowspan="6">Infectious diseases: Viral</td></tr><tr><td>mcc05162</td><td>Measles</td></tr><tr><td>mcc05164</td><td>Influenza A</td></tr><tr><td>mcc05166</td><td>HTLV-I infection</td></tr><tr><td>mcc05168</td><td>Herpes simplex infection</td></tr><tr><td>mcc05169</td><td>Epstein-Barr virus infection</td></tr><tr><td>mcc04970</td><td>Salivary secretion</td><td>Digestive system</td></tr><tr><td colspan="3">Continued</td></tr><tr><td>mcc05200</td><td>Pathways in cancer</td><td rowspan="3">Cancers: Overview</td></tr><tr><td>mcc05203</td><td>Viral carcinogenesis</td></tr><tr><td>mcc05205</td><td>Proteoglycans in cancer</td></tr></table>

Table 1. Graft losing period-related activated pathways (GLPAPs). 59 out of 287 pathways in Rhesus KEGG database were selected after applying of TRAP algorithm.

<table><tr><td>Category</td><td>P-value</td></tr><tr><td>Immune system</td><td>0.0001962</td></tr><tr><td>Cancers: Specific types</td><td>0.0003236</td></tr><tr><td>Infectious diseases: Viral</td><td>0.0003591</td></tr><tr><td>Signal transduction</td><td>0.0120207</td></tr><tr><td>Infectious diseases: Parasitic</td><td>0.1036661</td></tr><tr><td>Endocrine system</td><td>0.1076348</td></tr><tr><td>Development</td><td>0.1083940</td></tr><tr><td>Cell motility</td><td>0.2055749</td></tr><tr><td>Cancers: Overview</td><td>0.2132333</td></tr><tr><td>Digestive system</td><td>0.6910951</td></tr><tr><td>Nervous system</td><td>1.0000000</td></tr><tr><td>Cell growth and death</td><td>1.0000000</td></tr><tr><td>Cellular communication</td><td>1.0000000</td></tr><tr><td>Excretory system</td><td>1.0000000</td></tr><tr><td>Folding, sorting and degradation</td><td>1.0000000</td></tr><tr><td>Signaling molecules and interaction</td><td>1.0000000</td></tr><tr><td>Translation</td><td>1.0000000</td></tr></table>

Table 2. Significantly enriched categories of GLPAPs. Categories are listed in ascending order of p-values calculated by Fisher's exact test. 'Immune system' category pathways were highly enriched.

# Discussion

Pancreatic islet transplantation is currently one of the best treatment options for end-stage type 1 diabetes patients $^{31}$ . Although the engraftment of islets has been successful short-term, it relatively lacked long-term durability, resulting in late-phase graft failure in some islet transplant recipients $^{2}$ . Likewise, we and others have found that the porcine islet grafts were also lost in the transplant recipient monkeys within 6\~30 months after transplantation in pig-to-NHP islet xenotransplantation $^{32}$ . Luckily, we were able to experience consistent, prolonged graft survival in rhesus monkeys $^{15}$ , and thus this retrospective analysis was performed in hopes to unearth the cause of late graft loss in pig-to-NHP islet xenotransplantation. To our knowledge, there have not been studies that were focused on the cause of late graft loss nor the mechanism of immune rejection in islet transplantation models of higher mammals.

Because we were not able to find any definitive explanation of islet loss with routine laboratory tests including biochemical and immunological assays in peripheral blood from R080, we used RNA-seq technology to quantify the amount of transcript in the samples obtained from whole blood taken at various time points after transplantation. Then, we carried out pathway analysis and pathway interaction network analysis based on TRAP and PINTnet, respectively. By performing pathway analyses, we found 59 activated pathways and named those GLPAPs (Table 1). Furthermore, we categorized GLPAPs to retrieve meaningful information. Indeed, mostly enriched category was revealed as 'immune system' (Table 2). This highly suggested that cause of graft loss in chronic phase in R080 is due to insufficient immune suppression, i.e. immune rejection. Subsequently, we constructed a pathway interaction network using GLPAPs as nodes to reveal which pathways played a central role in the given condition and found that 'T cell receptor signaling pathway,' 'B cell signaling pathway,' and 'platelet activation' were the most interconnected pathways (Fig. 2 & Table 3). This information suggested that our immunosuppressive regimen in the maintenance period should be revised and fortified to overcome the late graft loss.

Computationally, the above findings were reinforced by hypothesis testing. We can measure the influence of some nodes of interests to other nodes on a network using the network propagation. Likewise, we can measure the influence of genes on a biological network or a gene regulatory network. If we map genes on a curated biological network, select some genes as seed genes and run network propagation, we can measure the influence of the seed genes to other genes and rank the genes by the influence they received. Top 100 genes, for example, are the most influenced 100 genes. In our study, network propagation and subsequent correlation analysis revealed that chronic rejection-related genes had been the most related to the late graft loss (Table 4). It is noteworthy that our computational analyses are fairly relevant to the previous findings concerning the chronic rejection in transplantation. The B cells and antibodies have been known as the culprit of chronic rejection in solid organ transplantation $^{29}$ . There also have been reports indicative of platelet's contribution in the chronic rejection of transplanted

![](images/9cc341995fc1b1664e25850048ebc76c1e0cd3ede0488959933c6c05b6ccc87f.jpg)

<details>
<summary>network</summary>

| Node | Label | Color | Category |
|---|---|---|---|
| mcc04810 | | Blue | Cell motility |
| mcc04540 | | Cyan | Development |
| mcc04921 | mcc04921 | Yellow | Endocrine system |
| mcc04910 | mcc04910 | Yellow | Endocrine system |
| mcc04917 | mcc04917 | Yellow | Endocrine system |
| mcc04380 | mcc04380 | Cyan | Development |
| mcc04915 | mcc04915 | Yellow | Endocrine system |
| mcc04918 | mcc04918 | Yellow | Endocrine system |
| mcc03013 | | Red | Signaling molecules and interaction |
| mcc05166 | | Purple | Infectious diseases: Viral |
| mcc05169 | | Purple | Infectious diseases: Viral |
| mcc05161 | | Purple | Excretory system |
| mcc04210 | mcc04210 | Cyan | Cell motility |
| mcc04662 | mcc04662 | Red | Immune System |
| mcc04660 | mcc04660 | Red | Immune System |
| mcc04611 | mcc04611 | Red | Immune System |
| mcc04650 | mcc04650 | Red | Immune System |
| mcc04062 | mcc04062 | Red | Immune System |
| mcc04664 | mcc04664 | Red | Immune System |
| mcc04670 | mcc04670 | Red | Immune System |
| mcc05220 | mcc05220 | Green | Cancers: Specific types |
| mcc05212 | mcc05212 | Green | Cancers: Specific types |
| mcc05211 | mcc05211 | Green | Cancers: Specific types |
| mcc05219 | mcc05219 | Green | Cancers: Specific types |
| mcc05223 | mcc05223 | Green | Cancers: Specific types |
| mcc05215 | mcc05215 | Green | Cancers: Specific types |
| mcc05213 | mcc05213 | Green | Cancers: Specific types |
| mcc04722 | mcc04722 | Light Blue | Nervous system |
| mcc05200 | mcc05200 | Light Blue | Nervous system |
| mcc05203 | mcc05203 | Orange | Signaling transduction |
| mcc05205 | mcc05205 | Light Blue | Nervous system |
| mcc04141 | mcc04141 | Purple | Infectious diseases: Parasitic |
| mcc04630 | mcc04630 | Orange | Signaling transduction |
| mcc04668 | mcc04668 | Orange | Signaling transduction |
| mcc04725 | mcc04725 | Light Blue | Cancers: Overview |
| mcc05142 | mcc05142 | Purple | Infectious diseases: Parasitic |
| mcc04688 | mcc04688 | Green | Cell growth and death |
| mcc04370 | mcc04370 | Orange | Signaling transduction |
| mcc04370 | mcc04370 | Orange | Signaling transduction |
| mcc04688 | mcc04688 | Green | Cell growth and death |
| mcc04668 | mcc04668 | Orange | Signaling transduction |
| mcc04668 | mcc04668 | Orange | Signaling transduction |
| mcc04630 | mcc04630 | Green | Cell growth and death |
| mcc04630 | mcc04630 | Orange | Signaling transduction |
| mcc041411 | mcc041411 | Light Blue | Nervous system |
| mcc05169 | mcc05169 | Purple | Infectious diseases: Viral |
| mcc05162 | mcc05162 | Purple | Excretory system |
| mcc04961 | mcc04961 | Purple | Excretory system |
| mcc05223 | mcc05223 | Green | Cell growth and death |
| mcc05221 | mcc05221 | Green | Cell growth and death |
| mcc05219 | mcc05219 | Green | Cell growth and death |
| mcc05213 | mcc05213 | Green | Cell growth and death |
| mcc05219 (Green) vs. Negative (Cancers) vs. Negative (Nervous) (Cancers) vs. Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) (Cancers) vs. Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Negative (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive (Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecules and interaction) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Viral) < Positive(Signaling molecule: Vira/ Viral), < Positive(Signaling molecule: Vira/ Viral), < Positive(Signaling molecule: Vira/ Viral), < Positive(Signaling molecule: Vira/ Viral), < Positive(Signaling molecule: Vira/ Viral), < Positive(Signaling molecule: Vira/ Viral), < Positive(Signaling molecule: Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Vira/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Varea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Virea/ Vilea/ Vilea<fcel>Left
Right
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Left
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Right
Rice 1 Rice 2 Rice 3 Rice 4 Rice 5 Rice 6 Rice 7 Rice 8 Rice 9 Rice 1 Rice 2 Rice 3 Rice 4 Rice 5 Rice 6 Rice 7 Rice 8 Rice 9 Rice 1 Rice 3 Rice 4 Rice 5 Rice 6 Rice 7 Rice 8 Rice 9 Rice 1 Rice 3 Rice 4 Rice 5 Rice 6 Rice 7 Rice 8 Rice 9 Rice 1 Rice 3 Rice 4 Rice 5 Rice 6 Rice 7 Rice 8 Rice 9 Rice 1 Rice 2 Rice 3 Rice 4 Rice 5 Rice 6 Rice 7 Rice 8 Rice 9 Rice 1 Rice 3 Rice 4 Rice 5 Rice 6 Rice 7 Rice 8 Rice 9 Rice 9 Rice 1 Rice 3 Rice 4 Rice 5 Rice 6 Rice 7 Rice 8 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 9 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 8 Rice 7 Lrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rrrm rllrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmslrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrmlllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lrrn lttlslrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrmlrsmlslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrslslrsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknskntssrknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsrknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknsknskntssrknnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm<nl>
</details>

Figure 2. Pathway interaction network. Blue dotted rectangle represents T cell receptor signaling pathway (mcc04660), B cell receptor signaling pathway (mcc04662), and Platelet activation (mcc04611). The size of the nodes reflects the closeness centrality of each node. The network was visualized by Cytoscape $^{40}$ .

organs $^{33}$ . T cells have been mentioned very little regarding the chronic rejection of transplanted organs, but our results might have discovered that T cells may partake significantly in the chronic rejection of transplanted islets, at least in pig-to-NHP settings.

It would be of dire importance to show that the RNA-seq bioinformatics data truly represented the immunobiology of the recipient which experienced late graft loss. We retrieved the archives of liver biopsy materials of R080, and performed IHC that could prove the pertinence of the three GLPAPs ('T cell receptor signaling pathway,' 'B cell signaling pathway,' 'platelet activation') to the rejection. Accordingly, we could observe the infiltration of T cells to the graft site as well as antibody and fibrinogen accumulation (Fig. 3). Since these immune responses had not been conspicuous via routine immune monitoring (Supplementary Fig. S1) as mentioned before, our result calls for further studies on the monitoring of graft-proximal immunology in islet transplantation.

It is intriguing that only R080 experience immune system activation compared with R051 despite the usage of the same immunosuppressive regimen. To find out putative reason(s), we carefully reviewed pre-clinical symptoms, signs and laboratory data. Interestingly, we noticed that R080 had experienced two times of severe giardiasis-induced diarrhea around DPT90 and DPT120. This finding indicated that intestinal infections preceded the sampling for the RNA-seq about one month. There is increasing evidence that microbiota products could activate the innate immune system of the liver $^{34}$ , and it was reported by Xie et al. that microbiota alteration could result in acute rejection of rat liver allografts $^{35}$ . Because intestinal blood is drained to the liver via portal vein, we hypothesized that intestinal infection might have activated strong innate immunity and this in turn triggered adaptive immune response to the graft through heterologous immunity. In line with this notion, the

<table><tr><td>ID</td><td>Name</td><td>Closeness centrality</td><td>Degree</td></tr><tr><td>mcc04660</td><td>T cell receptor signaling pathway</td><td>0.6</td><td>21</td></tr><tr><td>mcc04662</td><td>B cell receptor signaling pathway</td><td>0.59302326</td><td>20</td></tr><tr><td>mcc04611</td><td>Platelet activation</td><td>0.46788991</td><td>11</td></tr><tr><td>mcc04664</td><td>Fc epsilon RI signaling pathway</td><td>0.49514563</td><td>8</td></tr><tr><td>mcc04650</td><td>Natural killer cell mediated cytotoxicity</td><td>0.45945946</td><td>7</td></tr><tr><td>mcc04062</td><td>Chemokine signaling pathway</td><td>0.43589744</td><td>7</td></tr><tr><td>mcc04670</td><td>Leukocyte transendothelial migration</td><td>0.43220339</td><td>5</td></tr><tr><td>mcc04621</td><td>NOD-like receptor signaling pathway</td><td>0.33774834</td><td>1</td></tr></table>

Table 3. The closeness centrality and the degree of GLPAPs in immune system. The average closeness centrality and the average degree of all GLPAPs in the network were 0.4669 and 8.65 respectively and the values were used as the cutoff values to determine if a GLPAP was meaningful in the pathway interaction network. Only T cell receptor signaling pathway, B cell receptor signaling pathway, and Platelet activation satisfied the cutoff values. The pathways are highlighted by underlines.

immunologically hostile effects of infection on the transplanted allografts were reported $^{36}$ and infection even might break down established tolerance to the graft in murine heart transplantation models $^{37}$ .

There are some limitations in our work. First, we only had one monkey which experienced relative early graft loss within the chronic phase after transplantation. Thus, our study cannot give a definite conclusion, but rather an intriguing insight to our field. We are planning to scale up our study using more animals to validate our bioinformatic analysis method. Second, although we suggested that three pathways might have been involved in immune rejection in the chronic phase, we only found the presence of the effectors in each pathway and could not present their actual involvement. Third, because rhesus pathways in KEGG were relatively insufficient, we were not able to analyze our data in high-resolution maps. For example, even though we were interested in CD40L or IL-6 signaling pathways in our setting, we were not able to analyze them because KEGG did not support those pathways. Lastly, we were not able to pinpoint single candidate molecule or a set of molecules which could be critically responsible for graft rejection. Our next works will focus on these questions.

# Methods

Animals. Rhesus monkeys (Macaca mulatta), 3–4 years of age were used in our study. All animal experiment procedures were performed in compliance with the Guide for the Care and Use of Laboratory Animals prepared by the Institute of Laboratory Animal Resources and published by the National Institutes of Health (NIH Publication No. 86–23, revised 2011). The experiments were approved by Seoul National University (SNU) Institutional Animal Care and Use Committee (IACUC no. 15-0297-S1A0). Islet donor pigs, the Seoul National University (SNU) miniature pigs were bred as in designated pathogen-free (DPF) grade $^{38}$ .

Porcine islet isolation transplantation into the monkey recipients. Adult porcine islet were isolated from pig and transplanted into the liver of rhesus monkey as described $^{15}$ . In brief, chemically diabetic induced rhesus monkeys were transplanted with porcine islet via jejunal vein after a laparotomy was performed.

Immunosuppression. Induction immunosuppression included a regimen with anti-human CD154 monoclonal antibody, sirolimus (Rapamune $^{®}$ , Wyeth), and anti-thymocyte globulin (ATG, Thymoglobulin $^{®}$ , Genzyme). Sirolimus was administered daily to achieve stable trough levels between 3 and 8 ng/ml. CVF (100 U/kg, Quidel) was administered on day -1 of the transplant to prevent complement activation. TNF- $\alpha$ neutralizing monoclonal antibody, adalimumab (Humira $^{®}$ , Abbott Laboratories Ltd., Queenborough, UK) was administered subcutaneously 2\~3 hrs before islet infusion with dose of 5 mg/kg. 10 $^{6}$ to 10 $^{7}$ ex-vivo expanded regulatory T cells were adoptively transferred after ATG depletion.

IVGTT. After an overnight fasting without insulin, 0.5 g/kg of 50% dextrose solution with same volume of normal saline was infused i.v. for 1 min. Blood glucose levels were measured in monkeys before and 2, 5, 15, 30, 60, 90, and 120 min after infusion. Insulin and c-peptide levels were measured at the same time intervals as described previously $^{15}$ .

Hematology

Enzyme-linked immunosorbent spot (ELISPOT) assay. ELISPOT analysis was performed by the method previously described $^{39}$ . $2.5 \times 10^{5}$ monkey peripheral blood mononuclear cells were cultured with $5 \times 10^{5}$ porcine splenocytes (30 Gy irradiation). The resulting spots were counted on a computer-assisted ELISpot Reader System (AID, Germany).

Flow cytometry. Flow cytometry analyses of peripheral blood leukocytes were performed using the following monoclonal antibodies: CD3-FITC (FN-18; U-CyTech biosciences, Utrecht, The Netherlands), CD4-APC-Cy7 (OKT; BioLegend, San Diego, CA), CD8-PE-Cy7 (SK1; eBioscience, San Diego, CA), FoxP3-PE (PCH101; eBioscience), HLA-DR-PerCP-eFluor710 (L243; eBioscience), CD20-PE (2H7; Thermo Fisher

a   
![](images/06b3c61821fff6c939b0d72d21d8cf96273ec5e353fa24690202c3ee2d179ebd.jpg)

<details>
<summary>natural_image</summary>

Microscopic tissue image showing cellular structures with red and blue staining, labeled 'Insulin / CD3 / CD68' and a black arrow pointing to a specific region (no text or symbols beyond labels)
</details>

![](images/5d12cfd30d68d14060b287c7735f56123f333cc6a8c4f95cbbadbfd2e30f9f2e.jpg)

<details>
<summary>natural_image</summary>

Microscopic tissue image showing cellular structures with red and brown staining, labeled 'Insulin / CD4 / CD8' (no additional text or symbols)
</details>

![](images/9605b0f2d1cc6820257e399028774af5434dc6f21c39acd9665cde2c079b024e.jpg)

<details>
<summary>text_image</summary>

Insulin / CD20 / CD3
</details>

b   
![](images/117bbe70949eccebc33614f7168c6d20e6c6ebff8e1510b7ce8f9141d5028573.jpg)

<details>
<summary>natural_image</summary>

Fluorescence microscopy image showing cellular structures with red and green staining, labeled with insulin/fibrinogen/DAPI markers (no text or symbols beyond labels)
</details>

![](images/68f0593762a8a789b02d1222680efc1598b4e41b0f59b9d3d23275fcdd30b529.jpg)

<details>
<summary>natural_image</summary>

Fluorescence microscopy image showing cellular staining with blue, red, and green channels (no text or symbols)
</details>

Figure 3. Histology of islet xenografts on DPT184. (a) The islet graft was heavily infiltrated by several types of immune cells in R080. Immune cells largely consisted of CD3 $^{+}$ T cells. Both CD4 $^{+}$ and CD8 $^{+}$ cells infiltrated the graft. CD68 $^{+}$ cells were also observed. Black arrowheads indicate intra-graft infiltrating T cells. The scale bar applies universally. (b) Immunofluorescence staining against insulin (red), fibrinogen, and monkey IgG (left and right, respectively; both green). Blue denotes nuclei stained by DAPI. The scale bar applies universally.

<table><tr><td>Hypothesis</td><td>Coefficient</td><td>p-value</td><td>Empirical p-value</td></tr><tr><td>ERstress</td><td>0.031115500</td><td>0.001246017</td><td>0.048</td></tr><tr><td>IsletExh</td><td>0.049513322</td><td>0.000048063</td><td>0.292</td></tr><tr><td>Lipotoxicity</td><td>0.051612597</td><td>0.000022611</td><td>0.251</td></tr><tr><td>CGR</td><td>0.087461960</td><td>0.000000000</td><td>0.010</td></tr><tr><td>ToxImmDrug</td><td>0.050939480</td><td>0.000028885</td><td>0.275</td></tr></table>

Table 4. Ranking comparison between network propagation results and differential expression. Pearson's correlation coefficients of each hypothesis. IsletExh, CGR, and ToxImmDrug stand for islet exhaustion, chronic graft rejection, and toxicity of immunosuppressant, respectively. The coefficient of chronic graft rejection was the highest.

<table><tr><td>Hypothesis</td><td>Coefficient</td><td>p-value</td></tr><tr><td>ERstress</td><td>0.235867693</td><td>0.018154182</td></tr><tr><td>IsletExh</td><td>0.154287565</td><td>0.125356981</td></tr><tr><td>Lipotoxicity</td><td>0.188772704</td><td>0.059979769</td></tr><tr><td>CGR</td><td>0.556480178</td><td>0.000000001</td></tr><tr><td>ToxImmDrug</td><td>0.536431327</td><td>0.000000008</td></tr></table>

Table 5. Ranking comparison between network propagation results and differential expression. Correlation coefficients of ranking comparison for the 100 most-influenced genes from the network propagation results. Chronic graft rejection showed the highest coefficient.

Scientific, Waltham, MA), CD14-Alexa488 (M5E2; Biolegend), CD16-APC (3G8; BD Biosciences). Absolute counts of leukocytes were measured with 123count eBeadsTM (Thermo Fisher Scientific). FACSCanto II flow cytometer (BD Biosciences; San Jose, CA) and FACSDiva software (BD Biosciences) were used for analyses.

RNA sequencing. Total RNA from peripheral blood of rhesus monkey was extracted using the Ambion Ribopure $^{TM}$ -Blood kit (Thermo Fisher Scientific) as recommended by the manufacturer. 500 $\mu$ l of peripheral blood from the recipient monkeys was used for each sample. Eluted total RNA was stored in $-80^{\circ}$ C. Next, RNA sequencing was performed. Purified total RNA was sequenced by Theragen Etex (Korea).

Biopsy and immunohistochemistry. Biopsy and immunohistochemical staining was conducted as described previously $^{[15]}$ . Briefly, liver biopsy samples from the recipient monkeys were fixed in 10% neutral buffered formalin, and embedded in paraffin. Paraffin-embedded tissues were 4- $\mu$ m sectioned using a microtome. Each de-paraffinized and hydrated section was incubated for 30 min with primary antibody cocktails for insulin (Santa Cruz Biotechnologies, Dallas, TX), CD3 (DAKO, Agilent, Santa Clara, CA), CD4 (Santa cruz Biotechnologies), CD8, CD20, and CD68 (all from Abcam, UK), and then washed four times in TBST. After staining procedure, all of the stained slides were dried at 60°C, and mounted with aqueous mounting medium (Thermo Fisher Scientific). For immunofluorescence staining, anti-fibrinogen antibody and anti-insulin antibody (both from Abcam) were used as primary antibodies and, Alexa 488-conjugated anti-rabbit IgG and Alexa 647-conjugated anti-mouse IgG were used as secondary antibodies, respectively. FITC-conjugated anti-monkey IgG (Acris, Germany) was used to detect antibody involvement at the graft site. The stained sample was observed by Carl Zeiss Axio Imager A1 microscope and images were taken with a micrograph with AxioVision software (Carl Zeiss, Germany).

Bioinformatic analyses. TRAP $^{16}$ and pathway interaction network $^{20}$ were performed as described previously. Network propagation was performed as follows: The genes related to the five hypotheses were collected as seed genes by the literature search and domain knowledge, and each hypothesis was represented by a set of genes. Next, DEG profile was established by measuring the expression change of each gene with calculating the log2 fold change between R080 and R051 at time point 3 and ranking the genes. At that time, we removed the genes of which the expression value was smaller than 1 in either R080 or R051 to prevent extremely high or low fold change yielded by the comparison between small numbers. Then, we mapped the seed genes on a PPI network. The number of nodes and edges in the network are 6,780 and 117,963 respectively. The number of seed genes are ten, nine, eight, ten, and nine for ER stress, islet exhaustion, lipotoxicity, chronic graft rejection, and toxicity of immunosuppressant, respectively. After that, the global effect of the seed genes was measured using network propagation and the genes in the PPI network were ranked for each hypothesis. Then, Pearson's correlation coefficients were calculated between the ranking in the DEG profile and the ranking by the network propagation for each hypothesis. One thousand random simulations were run and empirical p-values were calculated to test the significance of the coefficient, as shown in the equation below:

$$
p ^ {i} = \frac {1}{N} \sum_ {j = 1} ^ {N} \left\{ \begin{array}{l} 1 \text {   if   } c _ {i j} > c _ {i} ^ {R} \\ 0 \text {   otherwise } \end{array} \right.
$$

where i indicates each hypothesis and it ranges from 1 to 5. $p^{i}$ indicates the empirical p-value of i-th hypothesis. N is the number of random simulation and it is 1,000. j indicates the j-th random simulation. $c_{ij}$ is the coefficient of j-th random simulation of i-th hypothesis. $c_{i}^{R}$ is the reference coefficient of i-th hypothesis.

Received: 14 March 2019; Accepted: 12 November 2019;

Published online: 11 December 2019

# References

1. Shapiro, A. J. et al. Islet transplantation in seven patients with type 1 diabetes mellitus using a glucocorticoid-free immunosuppressive regimen. New Engl. J. Med. 343, 230–238 (2000).   
2. Shapiro, A. J., Pokrywczynska, M. & Ricordi, C. Clinical pancreatic islet transplantation. Nature Reviews. Endocrinology 13, 268 (2017).   
3. Ryan, E. A. et al. Five-year follow-up after clinical islet transplantation. Diabetes 54, 2060–2069 (2005).   
4. Barton, F. B. et al. Improvement in outcomes of clinical islet transplantation: 1999–2010. Diabetes Care 35, 1436–1445 (2012).   
5. Fonseca, S. G., Gromada, J. & Urano, F. Endoplasmic reticulum stress and pancreatic $\beta$ -cell death. Trends in Endocrinology & Metabolism 22, 266–274 (2011).   
6. Negi, S. et al. Evidence of endoplasmic reticulum stress mediating cell death in transplanted human islets. Cell transplantation 21, 889–900 (2012).   
7. Lau, J., Henriksnäs, J., Svensson, J. & Carlsson, P.-O. Oxygenation of islets and its role in transplantation. Current opinion in organ transplantation 14, 688–693 (2009).   
8. Zheng, X. et al. Acute hypoxia induces apoptosis of pancreatic $\beta$ -cell by activation of the unfolded protein response and upregulation of CHOP. Cell death & disease 3, e322 (2012).   
9. Pugliese, A., Reijonen, H. K., Nepom, J. & Burke, G. W. Recurrence of autoimmunity in pancreas transplant patients: research update. Diabetes management 1, 229–238 (2011).   
10. Lee, Y. et al. Metabolic mechanisms of failure of intraportally transplanted pancreatic $\beta$ -cells in rats: role of lipotoxicity and prevention by leptin. Diabetes 56, 2295–2301 (2007).   
11. Leitão, C. B. et al. Lipotoxicity and decreased islet graft survival. Diabetes care 33, 658–660 (2010).   
12. Barlow, A. D., Nicholson, M. L. & Herbert, T. P. Evidence for rapamycin toxicity in pancreatic $\beta$ -cells and a review of the underlying molecular mechanisms. Diabetes 62, 2674–2682 (2013).   
13. Drachenberg, C. B. et al. islet cell damage associated with tacrolimus and cyclosporine: morphological features in pancreas allograft biopsies and clinical correlation $^{1}$ . Transplantation 68, 396–402 (1999).   
14. Becker, L. E., Morath, C. & Suesal, C. Immune mechanisms of acute and chronic rejection. Clin. Biochem. 49, 320–323 (2016).   
15. Shin, J. et al. Long-Term Control of Diabetes in Immunosuppressed Nonhuman Primates (NHP) by the Transplantation of Adult Porcine Islets. Am. J. Transplant. 15, 2837–2850 (2015).

16. Jo, K., Kwon, H.-B. & Kim, S. Time-series RNA-seq analysis package (TRAP) and its application to the analysis of rice, Oryza sativa L. ssp. Japonica, upon drought stress. Methods 67, 364–372 (2014).   
17. Chen, Y. et al. Peripheral blood transcriptome sequencing reveals rejection-relevant genes in long-term heart transplantation. International journal of cardiology 168, 2726–2733 (2013).   
18. Dorr, C. et al. Differentially expressed gene transcripts using RNA sequencing from the blood of immunosuppressed kidney allograft recipients. PloS one 10, e0125045 (2015).   
19. Kanehisa, M. & Goto, S. KEGG: kyoto encyclopedia of genes and genomes. Nucleic acids research 28, 27–30 (2000).   
20. Moon, J. H. et al. PINTnet: construction of condition-specific pathway interaction network by computing shortest paths on weighted PPI. BMC systems biology 11, 15 (2017).   
21. Rickels, M. R., Collins, H. W. & Naji, A. Amyloid and transplanted islets. The New England journal of medicine 359, 2729 (2008).   
22. Potter, K. et al. Islet amyloid deposition limits the viability of human islet grafts but not porcine islet grafts. Proc. Natl. Acad. Sci., 200909024 (2010).   
23. Westermark, G. T., Westermark, P., Berne, C. & Korsgren, O. Widespread amyloid deposition in transplanted human pancreatic islets. New England Journal of Medicine 359, 977–979 (2008).   
24. Kim, J.-W. & Yoon, K.-H. Glucolipotoxicity in pancreatic β-cells. Diabetes & metabolism journal 35, 444–450 (2011).   
25. Brown, M. S. & Goldstein, J. L. The SREBP pathway: regulation of cholesterol metabolism by proteolysis of a membrane-bound transcription factor. Cell 89, 331–340 (1997).   
26. Brown, M. S. & Goldstein, J. L. Sterol regulatory element binding proteins (SREBPs): controllers of lipid synthesis and cellular uptake. Nutrition reviews 56, S1–S3 (1998).   
27. Kakuma, T. et al. Leptin, troglitazone, and the expression of sterol regulatory element binding proteins in liver and pancreatic islets. Proceedings of the National Academy of Sciences 97, 8536–8541 (2000).   
28. Libby, P. & Pober, J. S. Chronic rejection. Immunity 14, 387–397 (2001).   
29. Valenzuela, N. M. & Reed, E. F. Antibody-mediated rejection across solid organ transplants: manifestations, mechanisms, and therapies. The Journal of clinical investigation 127, 2492–2504 (2017).   
30. Cowen, L., Ideker, T., Raphael, B. J. & Sharan, R. Network propagation: a universal amplifier of genetic associations. Nature Reviews Genetics 18, 551 (2017).   
31. McCall, M. & Shapiro, A. J. Update on islet transplantation. Cold Spring Harbor perspectives in medicine 2, a007823 (2012).   
32. Park, C.-G., Bottino, R. & Hawthorne, W. J. Current status of islet xenotransplantation. Int. J. Surg. 23, 261–266 (2015).   
33. Morrell, C., Sun, H., Swaim, A. & Baldwin, W. III Platelets an inflammatory force in transplantation. Am. J. Transplant. 7, 2447–2454 (2007).   
34. Chassaing, B., Etienne-Mesmin, L. & Gewirtz, A. T. Microbiota-liver axis in hepatic disease. Hepatology 59, 328–339 (2014).   
35. Xie, Y. et al. Effect of intestinal microbiota alteration on hepatic damage in rats with acute rejection after liver transplantation. Microb. Ecol. 68, 871–880 (2014).   
36. Chong, A. S. & Alegre, M.-L. The impact of infection and tissue damage in solid-organ transplantation. Nature Reviews Immunology 12, 459–471 (2012).   
37. Wang, T. et al. Infection with the intracellular bacterium, Listeria monocytogenes, overrides established tolerance in a mouse cardiac allograft model. American Journal of Transplantation 10, 1524–1533 (2010).   
38. Jin, S. M. et al. Islet isolation from adult designated pathogen-free pigs: use of the newer bovine nervous tissue-free enzymes and a revised donor selection strategy would improve the islet graft function. Xenotransplantation 18, 369–379 (2011).   
39. Kim, H. J. et al. Porcine antigen-specific IFN- $\gamma$ ELISpot as a potentially valuable tool for monitoring cellular immune responses in pig-to-non-human primate islet xenotransplantation. Xenotransplantation 23, 310–319 (2016).   
40. Shannon, P. et al. Cytoscape: a software environment for integrated models of biomolecular interaction networks. Genome research 13, 2498–2504 (2003).

# Acknowledgements

This work was supported by a grant from the Korea Healthcare Technology R&D Project through the Korea Health Industry Development Institute (KHIDI) and funding from the Ministry for Health and Welfare, Republic of Korea (Grant No. HI13C0954). This work was partly supported by the interdisciplinary Research Initiatives Program from College of Engineering and College of Medicine, Seoul National University (Grant No. 800-20130070) and by a grant from Seoul National University Hospital (2019). Anti-CD154 antibody used in these studies was provided by the Nonhuman Primate Reagent Resource supported by U.S. National Institutes of Health NIAID contract HHSN 272201300031C.

# Author contributions

Kim H.-J. and Moon J.H. conceptualized and designed the work. Kim H.-J., Moon J.H. and Chung H. analyzed and interpreted the data. Kim H.-J., Moon J.H. and Chung H. drafted the manuscript. Kim H.-J., Moon J.H. and Chung H. revised the manuscript. Kim H.-J., Chung H., Shin J.-S., Kim B., Kim J.-M., Kim J.-S., Yoon I.-H., Min B.-H., Kang S.-J. and Kim Y.-H. participated in NHP data acquisition. Moon J.H. and Jo K. conducted bioinformatics analyses, Choi J. and Chae H. conducted differentially expressed miRNA analyses. Lee W.-W., Kim S. and Park C.-G. designed the experiment. Kim S. and Park C.-G. drafted the manuscript, and supervised overall project.

# Competing interests

The authors declare no competing interests.

# Additional information

Supplementary information is available for this paper at https://doi.org/10.1038/s41598-019-55417-y.

Correspondence and requests for materials should be addressed to S.K. or C.-G.P.

Reprints and permissions information is available at www.nature.com/reprints.

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

![](images/01041e73a1c8b82c4c15e37b9dacee404c9e866882ba48b0fd556fd68690ff17.jpg)

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or

format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2019