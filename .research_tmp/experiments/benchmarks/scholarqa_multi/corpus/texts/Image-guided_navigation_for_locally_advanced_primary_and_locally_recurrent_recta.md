# RESEARCH ARTICLE

# Open Access

# Image-guided navigation for locally advanced primary and locally recurrent rectal cancer: evaluation of its early cost-effectiveness

![](images/c86cddcc7f5ceaec8fa8ea6f733fae042a7d84883b41da8284bd53cad811d740.jpg)

Melanie Lindenberg $^{1,2\dagger}$ , Astrid Kramer $^{1\dagger}$ , Esther Kok $^{3}$ , Valesca Retèl $^{1,2}$ , Geerard Beets $^{3}$ , Theo Ruers $^{3,4}$ and Wim van Harten $^{1,2*}$

# Abstract

Background: A first pilot study showed that an image-guided navigation system could improve resection margin rates in locally advanced (LARC) and locally recurrent rectal cancer (LRRC) patients. Incremental surgical innovation is often implemented without reimbursement consequences, health economic aspects should however also be taken into account. This study evaluates the early cost-effectiveness of navigated surgery compared to standard surgery in LARC and LRRC.

Methods: A Markov decision model was constructed to estimate the expected costs and outcomes for navigated and standard surgery. The input parameters were based on pilot data from a prospective (navigation cohort n = 33) and retrospective (control group n = 142) data. Utility values were measured in a comparable group (n = 63) through the EQ5D-5L. Additionally, sensitivity and value of information analyses were performed.

Results: Based on this early evaluation, navigated surgery showed incremental costs of €3141 and €2896 in LARC and LRRC. In LARC, navigated surgery resulted in 2.05 Quality-Adjusted Life Years (QALYs) vs 2.02 QALYs for standard surgery. For LRRC, we found 1.73 vs 1.67 QALYs respectively. This showed an Incremental Cost-Effectiveness Ratio (ICER) of €136.604 for LARC and €52.510 for LRRC per QALY gained. In scenario analyses, optimal utilization rates of the navigation technology lowered the ICER to €61.817 and €21.334 for LARC and LRRC. The ICERs of both indications were most sensitive to uncertainty surrounding the risk of progression in the first year after surgery, the risk of having a positive surgical margin, and the costs of the navigation system.

Conclusion: Adding navigation system use is expected to be cost-effective in LRRC and has the potential to become cost-effective in LARC. To increase the probability of being cost-effective, it is crucial to optimize efficient use of both the hybrid OR and the navigation system and identify subgroups where navigation is expected to show higher effectiveness.

Keywords: Early cost-effectiveness analysis, Navigation technology, Surgery, Early health technology assessment, Locally advanced rectal cancer, Local recurrent rectal cancer

# Background

Rectal cancer is mainly treated by surgical resection, often complemented with pre- and/or postoperative (chemo) radiotherapy in stage II-IV tumors $[1-3]$ , showing a 5-year survival rate of $\sim45\%$ for stage III and $\sim20\%$ for stage IV tumors $[4]$ . Surgical resection of both locally

Decision tree   
![](images/0cfcb8a81bd94aeaea38d502d26271b42ea9036f5cbad7831ede3a29c29e78d0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["LARC/LRRC"] --> B["Standard surgery"]
    A --> C["Navigated surgery"]
    B --> D["Negative surgical margin (R0)"]
    B --> E["Positive surgical margin (R1)"]
    C --> F["Negative surgical margin (R0)"]
    C --> G["Positive surgical margin (R1)"]
```
</details>

Markov model   
![](images/39e2ad1687d02447f8ce63b278117b164aa9c4b0f10eb3168dd3db855fdbca83.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Disease Free"] --> B["Progress of disease"]
    B --> C["Death"]
    C --> D["Progression of disease in the 1st year"]
    C --> E["Progression of disease in the 2nd and 3rd year"]
    A --> F["Loop back to Disease Free"]
```
</details>

Fig. 1 Overview of the model. On the left, the decision tree is visualized in which the margin status after navigated and standard surgery is incorporated. On the right, the Markov model is shown which is used to model the costs and effects after having a negative or positive surgical margin. It also shows the tunnel states used to incorporate time effects on the transition from progression to death due to progression

advanced (LARC) and locally recurrent rectal cancer (LRRC) requires special consideration because (1) the disruption of normal anatomical planes and (2) radiotherapy-induced fibrosis can lead to a higher risk of a tumor positive involved circumferential resection margin [1, 5]. In this setting, LARC was defined as T3 or T4 tumors extending close to (<2 mm) or invading the mesorectal fascia, as shown on rectal magnetic resonance imaging. LRRC was defined as rectal cancer that recurred in the pelvic area after earlier treatment. In 10–15% of rectal cancer patients, positive surgical margins are found [6, 7] which negatively affects the prognosis [8–10]. Local recurrence can cause debilitating symptoms, and often requires additional treatment, such as chemoradiotherapy and radiotherapy. Optimizing surgical practice and decreasing the risk of positive resection margins is therefore of great clinical and financial importance.

Multiple technologies have emerged to improve the quality of surgery and surgical outcomes $[11]$ . The Netherlands Cancer Institute (NKI-AVL) has developed an image-guided navigation system to improve tumor localization during the operative procedure and prevent damage to surrounding vital structures $[12]$ . Recently, this navigation system has been evaluated in the first series of LARC and LRRC patients, showing substantially improved negative surgical margin rates compared to standard surgery in a historical control group $[12]$ . Since the use of a navigation system is associated with extra costs (e.g. due to extra imaging, the navigation system, and personnel), and hospital budgets are limited, new surgical technologies have to prove themselves in terms of cost-effectiveness to have a chance of reimbursement.

To evaluate the potential value of this navigation system, to inform policymakers, and to guide subsequent decisions on further research and development $[13]$ , early cost-effectiveness analyses can be performed. This study evaluates the early cost-effectiveness of the image-guided navigation system used during surgery for LARC and LRRC patients compared to standard surgery based on the first clinical data sampled in the Netherlands Cancer Institute $[12]$ .

# Methods

# Study design and model structure

To evaluate the early cost-effectiveness of navigated surgery we used a combination of a decision tree and a Markov model. The decision tree showed the possibility of having a positive (R1) or negative (R0) resection margin after standard and navigated surgery $[12]$ . The Markov model comprised the mutually exclusive health states: “disease-free”, “progression of disease” and the absorbing state “death” (Fig. 1). Whether a patient moves is partly explained by the outcome of the decision tree (R1 or R0). In the Markov model, all patients start in “disease free” and could either remain in “disease free” or transfer to “progression of the disease” or “death”. Since the course of disease for LARC and LRRC is different, two separate models were constructed with a similar

design. The time horizon was set at 3 years because most recurrences develop in the first 3 years after (curative) resection [14]. Besides, recent literature reported a median survival time of 37 [15] and 30 months [16] for LARC and LRRC, respectively. A cycle time of 3 months was chosen according to guidelines for follow-up visits [17]. The early cost-effectiveness analysis was performed from a Dutch healthcare perspective, using the Dutch guideline for health economic costing studies [18]. This means that we evaluate all relevant costs and effects part of the healthcare system, e.g. productivity losses or travel expenses of patients were not included. The primary outcome of this analysis is the incremental cost-effectiveness ratio (ICER).

# Standard and navigated surgery

Standard treatment of LARC and LRRC consists of a rectal resection performed with an Abdomioperineal Resection (APR) or a Low Anterior Resection (LAR), with or without resection of the surrounding organs (exenterative procedures, sacral bone etc.) and intraoperative radiotherapy, depending on the patient and tumor characteristics (e.g. tumor location, previous surgeries, etc.). Procedures can be performed open or laparoscopically.

The addition of the navigation system for rectal surgery in patients with LARC or LRRC changed the regular workflow before and during surgery. One day before surgery, a multiphase contrast-enhanced CT scan (with early arterial and excretion phase) was acquired. Based on this preoperative imaging a digital 3-dimensional anatomical model was made, including the most important anatomical structures (blood vessels, ureters, bones and targets). Before surgery, in a hybrid operating room, three patient trackers (electromagnetic) were taped to the skin of the patient, and a cone-beam CT scan was performed. The acquired intraoperative images were matched with the preoperative images and the 3-dimensional anatomical model. During surgery, the patient lies on a specific imaging bed including an electromagnetic field generator. The location of the patient trackers was matched with the preoperative imaging and 3-dimensional anatomical model. By using an electromagnetic pointer, the surgeon could navigate towards the tumor in the 3D anatomical model on a separate screen. A more detailed description of the navigation system can be found in the article of Nijkamp et al., 2018 [19].

# Input parameters

The input parameters are presented in Tables 1, 2, 3. Supplement 1 shows a schematic overview of the data sources used for the input parameters.

# Clinical effectiveness

The effectiveness of navigated surgery compared to standard surgery in terms of R0 or R1 were obtained from the patient population of the study of Kok et al. [12]. They prospectively included 33 patients who received navigated surgery for either LARC (n = 14) or LRRC (n = 19) between 2016 and 2019 in the Netherlands Cancer Institute (NKI-AVL). As a control group, Kok et al. included 142 patients having standard surgery for LARC (n = 101) and LRRC (n = 41) as a retrospective cohort. These patients had a similar indication and type of surgery at the NKI-AVL [12]. Supplement 2 shows the characteristics of these patient populations (prospective and retrospective group) [12]. The Institutional Review Board of the NKI-AVL approved data extraction for the included patients.

Among LARC patients, 93% R0 resections were achieved after navigated- and 84% after standard surgery. Among LRRC patients, 79% had an R0 resection after navigated- and 49% after standard surgery [12]. These values were incorporated in the decision tree.

To calculate the transitions between the health states in the Markov model, progression of disease was evaluated in the retrospective control group $n = 142$ . Based on literature, we assumed that (1) progression of disease was affected by the resection margin status [5, 26] and (2) that death due to colorectal cancer (CRC) was affected by progression status. Information on progression of disease stratified by margin status and mortality data stratified by progression status were retrieved from medical records. Progression of disease was defined as “local recurrence or distant metastasis after surgery”, as the sample size was too small to stratify for local and distant recurrence. Among these patients, some had limited metastatic disease prior to surgery (e.g. liver metastasis). To prevent overestimating the risk of progression in the whole population, these patients were incorporated in the progression of disease state in the first cycle after surgery.

The probabilities to experience events (progression or death) per 3 months were calculated linearly using the number of events and the total number of patients at risk with the following formula: $1-\exp(-r^{*}t)$ . Where 'r' stands for the rate per 3 months calculated by -(ln (1-observed chance)/time of the observation), and 't' stands for the time [27]. To incorporate time or disease history in the model, two tunnel states were incorporated in the model: "1 $^{st}$ -year progression of disease" and "2 $^{nd}$ or 3 $^{rd}$ -year progression of disease" [28, 29]. The risk of dying due to progression within 3 years was evaluated separately for patients having progression in the first year and separately for patients having progression in the second and third year. The second and third year were combined because of a limited number of cases. Mortality due to all

Table 1 Input parameters for the decision tree and the Markov model on clinical effectiveness 

<table><tr><td>The observed number of patients</td><td colspan="2">LARC</td><td colspan="3">LRRC</td><td>Source</td></tr><tr><td>Having a negative surgical margin (R0)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>After navigated surgery</td><td colspan="2">13 (n = 14)</td><td colspan="3">15 (n = 19)</td><td>[A]</td></tr><tr><td>After standard surgery</td><td colspan="2">85 (n = 101)</td><td colspan="3">20 (n = 41)</td><td>[B]</td></tr><tr><td>Having progression per year</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>after R0 1st year</td><td colspan="2">29 (n = 85)</td><td colspan="3"> $9 (n = 20)^c$ </td><td>[B]</td></tr><tr><td>after R0 2nd (for LRRC: and 3rd) year</td><td colspan="2">9 (n = 85)</td><td colspan="3"> $4 (n = 20)^c$ </td><td>[B]</td></tr><tr><td>after R0  $3rd year^a$ </td><td colspan="2"> $4^a (n = 85)$ </td><td colspan="3">-</td><td>[B]</td></tr><tr><td>after R1 1st year</td><td colspan="2">11 (n = 16)</td><td colspan="3">11 (n = 20)</td><td>[B]</td></tr><tr><td>after R1 2nd and 3rd year</td><td colspan="2">1 (n = 16)</td><td colspan="3">3 (n = 20)</td><td>[B]</td></tr><tr><td>Died due to CRC after progression; over a timeframe of 3 years</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>after progression in the 1st year</td><td colspan="2"> $25 (n = 41)^d$ </td><td colspan="3">13 (n = 20)</td><td>[B]</td></tr><tr><td>after progression in the 2nd and 3rd year</td><td colspan="2"> $2 (n = 13)^d$ </td><td colspan="3">3 (n = 7)</td><td>[B]</td></tr><tr><td rowspan="2">Parameters used in the decision model</td><td colspan="2">LARC</td><td colspan="2">LRRC</td><td rowspan="2">Distribution</td><td rowspan="2">Source</td></tr><tr><td>Mean</td><td>SE</td><td>Mean</td><td>SE</td></tr><tr><td>Negative surgical margin rate (R0)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Navigated surgery</td><td>0.93</td><td>0.0665</td><td>0.79</td><td>0.0911</td><td>Beta</td><td>[A]</td></tr><tr><td>Standard surgery</td><td>0.84</td><td>0.0362</td><td>0.49</td><td>0.0771</td><td>Beta</td><td>[B]</td></tr><tr><td colspan="7">Transition probability for Disease-free to Progression after a negative surgical margin (R0)</td></tr><tr><td>from DF to PD in the 1st year</td><td>0.103</td><td>0.0328</td><td>0.159</td><td>0.0798</td><td>Beta</td><td>[B]</td></tr><tr><td>from DF to PD in the 2nd year</td><td>0.047</td><td>0.0229</td><td>0.100</td><td>0.0656</td><td>Beta</td><td>[B]</td></tr><tr><td>from DF to PD in the 3rd year</td><td>0.013</td><td>0.0121</td><td> $0.100^b$ </td><td>0.0656</td><td>Beta</td><td>[B]</td></tr><tr><td colspan="7">Transition probability for Disease-free to Progression after a positive surgical margin (R1)</td></tr><tr><td>from DF to PD in the 1st year</td><td>0.252</td><td>0.105</td><td>0.252</td><td>0.0926</td><td>Beta</td><td>[B]</td></tr><tr><td>from DF to PD in the 2nd year</td><td>0.0275</td><td>0.0397</td><td>0.159</td><td>0.0780</td><td>Beta</td><td>[B]</td></tr><tr><td>from DF to PD in the 3rd year</td><td> $0.0275^b$ </td><td>0.0397</td><td> $0.159^b$ </td><td>0.0780</td><td>Beta</td><td>[B]</td></tr><tr><td>Transition probability for Progressive Disease to Death</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>from PD in the 1st year to Death</td><td>0.090</td><td>0.0665</td><td>0.135</td><td>0.0745</td><td>Beta</td><td>[B]</td></tr><tr><td>from PD in the 2nd and 3rd year to Death</td><td>0.030</td><td>0.0362</td><td>0.089</td><td>0.1007</td><td>Beta</td><td>[B]</td></tr><tr><td>Transition probability for Disease-free to Death</td><td>0.0028</td><td>-</td><td>0.0044</td><td>-</td><td>-</td><td>[20] back-ground mortality</td></tr></table>

SE Standard error, DF Disease Free, PD progression of disease, CRC ColoRectal Cancer; [A] Prospective data collection within the navigated group at the NKI-AVL [12]; [B] Retrospective data collection within the control group at the NKI-AVL [12]   
$^{a}$ Only in the LARC group, among patients showing a negative surgical margin enough events were found in both the 2nd and 3rd year to calculate probabilities for both years. In the other groups, we found limited events and decided to calculate a combined probability for the 2nd and 3rd year   
$^{b}$ shows the transitions that were similar for the 2nd and 3rd years. This probability was based on the sum of events occurring in the 2nd and 3rd years   
$^{c}$ 1 of the LRRC patients received two surgeries and were both included in the analysis by Kok et al. For evaluating progression of disease this does not make sense, therefore this patient was excluded. Therefore the sum is 40 instead of 41   
$^{d}$ The total number of patients having progression in the 1st, and 2nd and 3rd year is different from the number presented between brackets in the lines for died due to progression. After R1 in the 1st year, all 12 events occurred in the 1st year and none in the 2nd and 3rd year. To incorporate uncertainty surrounding the chance on having progression in the 2nd and 3rd year we moved 1 event to the second year to calculate the transitions from disease-free to progression. Therefore, the number of patients progressed in the row for patients died due to progression shows one person more for the 1st year, and one person less for the 2nd and 3rd year

causes was based on data from the Dutch Central Bureau for Statistics, mirroring the average age of the two patient populations (LARC:60 years and LRRC:65 years) [12, 20]. All observed events and transition probabilities incorporated in the Markov model are listed in Table 1. Figure 2 shows the number of patients in the stable disease state over time for both interventions and patient groups.

# Health-related quality of life

Since no representative Quality of Life (QoL) data was available in literature $[30]$ , we used data from 63 patients of an ongoing prospective cohort study of patients with colorectal cancer who undergo standard and navigated surgery within the NKI-AVL with similar inclusion criteria as the patients enrolled in the study of Kok et al. $[12]$ . The clinical characteristics of these patients are presented

Table 2 Intervention and state costs and utilities used in the Markov model 

<table><tr><td>State costs</td><td>LARC</td><td>SE</td><td>LRRC</td><td>SE</td><td>Distribution</td><td>Source</td></tr><tr><td>Disease-free</td><td>€ 492</td><td>€ 63</td><td>€ 492</td><td>€ 63</td><td>Gamma</td><td>Expert</td></tr><tr><td>Transition from DF to PD</td><td>€ 14.883</td><td>€ 1.898</td><td>€ 13.107</td><td>€ 1.672</td><td>Gamma</td><td>Expert</td></tr><tr><td>Progressive disease</td><td>€ 585</td><td>€ 75</td><td>€ 585</td><td>€ 75</td><td>Gamma</td><td>Expert</td></tr><tr><td>Intervention costs</td><td>Combined LARC &amp; LRRC</td><td></td><td></td><td>SE</td><td>Distribution</td><td>Source</td></tr><tr><td>Surgery</td><td>€10.970</td><td></td><td></td><td>€1.399</td><td>Gamma</td><td>[21]</td></tr><tr><td>Addition of navigation</td><td>€3.388</td><td></td><td></td><td>€432</td><td>Gamma</td><td>Expert</td></tr><tr><td>Developing 3D model and preoperative CT scan</td><td>€ 269</td><td></td><td></td><td></td><td></td><td>[22–24]</td></tr><tr><td>Additional personnel during OR</td><td>€ 197</td><td></td><td></td><td></td><td></td><td>[22, 24]</td></tr><tr><td>Navigation system</td><td>€ 2.745</td><td></td><td></td><td></td><td></td><td>List prices; expert</td></tr><tr><td>Overhead</td><td>€ 177</td><td></td><td></td><td></td><td></td><td>[22]</td></tr><tr><td>Utilities</td><td>Combined LARC &amp; LRRC</td><td></td><td></td><td>SE</td><td>Distribution</td><td>Source</td></tr><tr><td>First cycle</td><td>0.70 (n = 63)</td><td></td><td></td><td>0.029</td><td>Beta</td><td>[C] (1mo survey)</td></tr><tr><td>Disease free (subsequent cycles)</td><td>0.85 (n = 44)</td><td></td><td></td><td>0.022</td><td>Beta</td><td>[C] (6mo survey)</td></tr><tr><td>Progressive disease (subsequent cycles)</td><td>0.77 (n = 14)</td><td></td><td></td><td>0.050</td><td>Beta</td><td>[C] (6mo survey)</td></tr></table>

SE Standard error, mo month, [C] Prospective observational cohort study

Table 3 Parameters used in scenario analysis 

<table><tr><td></td><td>Combined LARC &amp; LRRC</td><td>SE</td><td>Distribution</td><td>Source</td></tr><tr><td colspan="5">Scenario 1: Including the costs of constructing a hybrid OR when no hybrid OR is available in the hospital</td></tr><tr><td>Additional costs to use a hybrid OR and the C-arm CBCT in hybrid OR</td><td>€2.975</td><td></td><td></td><td>[25]; Supplement 4</td></tr><tr><td>Total costs of the addition of navigation</td><td>€6.363</td><td>€812</td><td>Gamma</td><td></td></tr><tr><td colspan="5">Scenario 2: Using the navigation system for 50% instead of 12%</td></tr><tr><td>Navigation system</td><td>€1.027</td><td></td><td></td><td>Supplement 4</td></tr><tr><td>Addition of navigation costs per patient</td><td>€1.670</td><td>€213</td><td>Gamma</td><td>Expert; supplement 4</td></tr></table>

SE Standard error, CBCT Cone-Beam CT

in Supplement 3. Based on these clinical characteristics, the group was judged sufficiently comparable to the control group to be used in our analysis.

Utilities were measured among these patients using the EQ5D-5L [31] to incorporate Quality Adjusted Life Years (QALYs) in the model. Utilities are values between 0 and 1 where a higher value indicates a better health status. Patients in the ongoing cohort study complete questionnaires before surgery and after 1, 3, 6, 12, and 24 months. The utility value for the first cycle in the model was based on the first-month questionnaire, incorporating the impact of the surgery itself. The utility value of the subsequent cycles (progression of disease or disease-free) was based on the questionnaire completed at 6 months after surgery stratified for the health status at 6 months. We only included patients from this ongoing cohort study when they had returned the follow-up questionnaires after 1 and 6 months. Due to the limited number of observations per indication (LARC and LRRC), we did not stratify for LARC or LRRC, assuming similar QoL when having progression or being disease-free (Table 2). Furthermore, we did not stratify for navigation or standard surgery as we hypothesize that the effect of using navigation is found in the number of patients showing progression of disease as a consequence of a higher positive resection margin rate.

# Intervention costs

For the costs of surgery, the formally average registered tariff (DRG) for open and laparoscopic low anterior resection (LAR) and abdominoperineal resection (APR) in the Netherlands were used. Open and laparoscopic APR and LRP showed the same tariff $[21]$ . The additional costs for using the image-guided navigation system were estimated using a bottom-up costing methodology, taking into account additional activities, additional required time, and personnel. We assumed implementing navigation in an already existing hybrid OR and, for the base case, exclusive use of the navigation system

![](images/99bc0d75fb720d1fa2b0d67d768dea0542cd67bff78b03ace91fac8d9a90277b.jpg)

<details>
<summary>line</summary>

| Years | Navigation group | Control group |
|-------|------------------|---------------|
| 0     | 1000             | 1000          |
| 1     | 500              | 450           |
| 2     | 250              | 200           |
| 3     | 150              | 100           |
</details>

![](images/6271ff1904f9034cd4cf389c9ee88d7d9ad6a4f4dbb571779d4ff571d30b350f.jpg)

<details>
<summary>line</summary>

| Years | Navigation group | Control group |
|-------|------------------|---------------|
| 0     | 1000             | 1000          |
| 1     | ~700             | ~650          |
| 2     | ~500             | ~480          |
| 3     | ~450             | ~430          |
</details>

Fig. 2 Graphical representation of patients in the stable disease state over time. These graphs show the number of patients in the model (cohort of 1000 patients) that stay in the stable disease state over time for the navigated and standard surgery group. A shows the patient flow for LRRC and B shows the patient flow for LARC

for this indication (12%). This resulted in an additional cost of €3.388 per patient. Details on the calculation are described in Supplement 4.

# State and transition costs

The health state costs and transition costs were based on the care delivered per state and transition. Care consumed per health state was based on the Dutch guideline on follow-up care for colorectal cancer $[32]$ . Expert elicitation was used to estimate a weighted average of care used in case of an event (local recurrence, distant metastasis) for both LARC and LRRC, such as radiotherapy and chemoradiation. To calculate the transition costs, the identified consumed care was linked to tariffs for DRGs, health activities, and medications $[22, 23, 33]$ (Table 2). Details on these costs are listed in Supplement 5.

# Model analysis and probabilistic sensitivity analysis

The models were built in Microsoft Excel, 2010. Costs were discounted at a rate of 4% and effects at a rate of 1.5% according to Dutch guidelines $[18]$ . The primary outcome of the models was the ICER, which is calculated by dividing the incremental costs by the incremental QALYs. The involved experts of the NKI-AVL (TR, EK, GB) collaborated to validate the model, input parameters, and assumptions. Because this analysis evaluates an innovation early in its development process, the input parameters are subject to uncertainty. This uncertainty in the data and its effect on the ICER was evaluated using a probabilistic sensitivity analysis. Tables 1, 2 and 3 show the distributions surrounding the parameter values used in the Monte Carlo simulations (2000 random samples) for this analysis. The results of the probabilistic analysis are shown in a cost-effectiveness (CE-)plane. Furthermore, cost-effectiveness acceptability curves (CEAC) were generated, indicating the probability that an intervention is cost-effective, given a certain Willingness To Pay (WTP). The informal WTP ratio for diseases with a high symptom burden is €80.000 per QALY $[34]$ in the Netherlands.

# Sensitivity analyses

In addition to the probabilistic sensitivity analysis, a deterministic one-way sensitivity analysis was performed, evaluating the influence of the uncertainty surrounding each of the input parameters. All parameters were varied over their upper and lower limits. The outcomes were plotted in a tornado diagram. Besides, two scenarios were evaluated: 1) Inclusion of construction costs for a hybrid OR to use the navigation system (in case a hospital does not have this yet), 2) Utilization of the navigation system was set at 50%, as it is assumed that the system is valuable in other indications as well. The input parameters for these scenarios are presented in Table 3 and detailed information is listed in Supplement 6. Finally, since the costs of the navigation system are still uncertain, a threshold analysis was performed assuming a WTP of €80.000 per QALY to identify the maximum incremental costs per patient [35].

# Value of information analysis

As this analysis was based on the first clinical data available for navigated surgery, the results are surrounded by a degree of uncertainty and therefore there is a chance that the ‘wrong’ policy decision is made. A value of information analysis provides insight in the worth of performing additional research, assuming that additional research would provide more certain estimates of the effects. The expected value of perfect information (EVPI), indicating the value of improved decision making by removing all uncertainty (i.e. by obtaining perfect information on all model parameters), was estimated $[36]$ . Additionally, the expected value of partial perfect information (EVPPI) was calculated. This analysis shows the expected value of eliminating uncertainty on (a group of) specific input parameters. Both these analyses can thus be used to support decisions on further research in early stages of technology development. The EVPI was calculated by taking the difference between the expected net monetary benefit - obtained under perfect information - and the expected net monetary benefit obtained based on the current data. To evaluate the EVPI and EVPPI for the beneficial population, we used the yearly incidence numbers of LARC $(n=1384)$ and LRRC $(n=250)$ based on the Dutch situation $[37, 38]$ . The population EVPI was evaluated for the coming 10 years and we discounted this population at a rate of 4%. The EVPPI was calculated for a willingness to pay threshold of €80.000.

# Results

# Base case results

Using the input parameters such as quality of life scores, risk of progression and chances of survival shown in Table 1 the incremental cost-effectiveness ratios for navigation use in LARC and LRRC were calculated. In this paragraph, the base case results – using the point estimates presented in Tables 1 and 2 – are presented. For LARC, we found 2.50 total Life Years (LY) after standard surgery versus 2.53 LY for navigated surgery. Total QALYs were 2.02 for standard and 2.05 for navigated surgery. Total costs for standard surgery were €23.238 compared to €26.379 for navigated surgery this amount includes the follow-up costs and costs for treating progression of disease. Dividing the incremental costs by the incremental effects resulted in an ICER of €136.604/QALY for LARC. Assuming a WTP ratio of €80.000 navigated surgery for LARC is judged not cost-effective.

For LRRC, we found 2.11 LYs after standard surgery compared to 2.17 LYs after navigated surgery. Total QALYs were 1.67 and 1.73 for standard and navigated surgery, respectively. Total costs of standard surgery were €25.862 and €28.719 for navigated surgery, including follow-up care and treating progression of disease. This early analysis resulted in an ICER of €51.802 per QALY gained (Table 4A). Assuming a WTP ratio of €80.000, navigated surgery for LRRC could be judged cost-effective.

# Probabilistic sensitivity analysis

The CE-plane in Fig. 3 shows that most observations for LARC (84%) indicated that navigated surgery resulted in better outcomes at higher costs. The Cost-Effectiveness Acceptability Curve (CEAC) shows that standard surgery in LARC has the highest probability of being cost-effective (78%) at a WTP of €80.000.

For LRRC, also most of the observations (79%) indicated improved outcomes at higher costs. The CE-plane shows more uncertainty compared to LARC, which corresponds to the smaller sample size in this study group. At a WTP threshold of €80.000, navigated surgery has a probability of 52% to be cost-effective compared to standard surgery for LRRC.

# Sensitivity analyses

Figure 4 shows that the results are mostly influenced by the uncertainty surrounding the transition probabilities for the first year, the surgical margin rate, and the costs of the navigation system in both groups. For example, when the maximum value for the transition from disease-free to progression after an R0 resection was used - showing similar or even worse progression than after R1 resection - LRRC and LARC show ICERs around €200.000. Contrary, when the maximum value from disease-free to progression after an R1 resection was used, the ICERs for LARC and LRRC decreased substantially. For LRRC, this resulted for example in a QALY difference of 0.11 compared to 0.06 in the base case.

Table 4 Deterministic outcomes of the cost-utility analysis on navigated surgery compared to standard surgery: base case and scenarios 

<table><tr><td colspan="8">A. Base case results</td></tr><tr><td></td><td>Treatment costs</td><td>QALYs</td><td>LYs</td><td>iCosts</td><td>iQALYs</td><td>ICER</td><td>Conclusion</td></tr><tr><td colspan="8">Base case results LARC</td></tr><tr><td>Navigated surgery</td><td>€26.379</td><td>2.05</td><td>2.53</td><td></td><td></td><td></td><td>Navigated surgery is more effective, more costly.</td></tr><tr><td>Standard surgery</td><td>€23.238</td><td>2.02</td><td>2.50</td><td></td><td></td><td></td><td rowspan="2">Costs are above the WTP of €80.000/QALY</td></tr><tr><td></td><td></td><td></td><td></td><td>€3.141</td><td>0.02</td><td>€136.604</td></tr><tr><td colspan="8">Base case results LRRC</td></tr><tr><td>Navigated surgery</td><td>€28.060</td><td>1.73</td><td>2.17</td><td></td><td></td><td></td><td>Navigated surgery is more effective, more costly.</td></tr><tr><td>Standard surgery</td><td>€25.164</td><td>1.67</td><td>2.11</td><td></td><td></td><td></td><td rowspan="2">Costs are below the WTP of €80.000/QALY</td></tr><tr><td></td><td></td><td></td><td></td><td>€2.896</td><td>0.06</td><td>€52.510</td></tr><tr><td colspan="8">B. Results from the scenario analysis</td></tr><tr><td></td><td></td><td></td><td>Intervention</td><td></td><td>ICER scenario</td><td>Conclusion scenario</td><td></td></tr><tr><td colspan="3">Scenario 1: A hybrid OR has to be constructed before the navigation system can be used</td><td>LARC</td><td></td><td>€266.019</td><td>Navigated surgery is more effective, more costly. Costs are above the WTP of €80.000/QALY</td><td></td></tr><tr><td></td><td></td><td></td><td>LRRC</td><td></td><td>€106.458</td><td>Navigated surgery is more effective, more costly. Costs are above the WTP of €80.000/QALY</td><td></td></tr><tr><td colspan="3">Scenario 2: Increase in utilization of the navigation system to 50%</td><td>LARC</td><td></td><td>€61.817</td><td>Navigation is more effective, more costly. Costs are below the WTP of €80.000/QALY</td><td></td></tr><tr><td></td><td></td><td></td><td>LRRC</td><td></td><td>€21.334</td><td>Navigation is more effective, more costly. Costs are below the WTP of €80.000/QALY</td><td></td></tr><tr><td colspan="3">Combination of 1 and 2: increased use of the navigation system and including the costs of constructing a hybrid OR to use the navigation system#</td><td>LARC</td><td></td><td>€191.232</td><td>Navigated surgery is more effective, more costly. Costs are above the WTP of €80.000/QALY</td><td></td></tr><tr><td></td><td></td><td></td><td>LRRC</td><td></td><td>€75.282</td><td>Navigation is more effective, more costly. Costs are below the WTP of €80.000/QALY</td><td></td></tr></table>

The WTP threshold used was €80.000. QALYs Quality of life years, Lys Life years, iCosts incremental costs, iQALYs incremental Quality of life years, ICER Incremental cost-effectiveness ratio, WTP Willingness To Pay threshold. # = total costs for the use of navigation including the hybrid OR costs assuming a use of 50% = €4.644,28

Table 4B presents the results of the scenario analysis. When a hospital has to construct a hybrid OR before navigation can be used, navigated surgery is not cost-effective in LARC or LRRC (scenario 1). Increasing the utilization of the navigation system (scenario 2) results in navigated surgery being cost-effective at a WTP threshold of €80.000 for LARC. For LRRC, navigated surgery is then cost-effective at most commonly used WTP thresholds. Since this is a realistic scenario, supplement 7 shows the probabilistic results for this scenario, showing that navigated surgery has a probability of 67% to be cost-effective for LRRC patients. Figure 5 shows the effect of various utilization ratios on the ICER for scenario 2 and the combination of Scenario 1 and 2.

Based on the threshold analysis, we found that the navigation system may have a maximum cost per patient of €1.839 in LARC and €4.412 in LRRC.

# Value of information analysis

The EVPI was almost €3.7M in LRRC, which indicates the value of reducing the risk of making the wrong decision (e.g. reimbursing navigated surgery when actually not cost-effective) by performing further research (Supplement 8). The results of the EVPPI are presented in Fig. 6. This graph shows that obtaining more certain estimates on having a positive or negative resection margin after standard- and navigated surgery would be most valuable. Other valuable topics for further research were the utility values and treatment costs (including the navigation costs). Since standard surgery was preferred at a WTP threshold of €80.000 in LARC, estimating the EVPI was not considered relevant for LARC.

# Discussion

This early evaluation indicates that navigated surgery is expected to be cost-effective in LRRC patients (ICER: €52.510). Furthermore, it has the potential to become cost-effective for LARC when the costs of the navigation system would decrease by for example increased use or price negotiations. These results are in line with the promising clinical results of Kok et al. [12].

Based on Fig. 4, a strong relationship between an R0 resection and a reduced risk of recurrence seems crucial for navigated surgery to become cost-effective. Although we concluded that navigation is cost-effective in LRRC, our data for LRRC showed no clear relationship between an R0 resection and a reduced risk of progression, e.g. reflected by the limited QALY gain found. Based on the

![](images/16a6508b0b46ccfa5e964ae565c72ed604db5b474251b904eab3435fe9c50567.jpg)

<details>
<summary>scatter</summary>

| Incremental QALYs | Incremental Costs (€) |
| ----------------- | --------------------- |
| 0.0               | €3000                 |
</details>

![](images/63b4f7b38787804697c3c43226871e5abcec17fa0cb1bd5aba50a4e7539d463f.jpg)

<details>
<summary>scatter</summary>

| Incremental QALYs | Incremental Costs (€) |
| ----------------- | --------------------- |
| 0.0               | €3000                 |
</details>

![](images/ef18b3f1e8c2b170bcfc094f46dbf294bf4bc84dd8a7da2c598f0419f5f8ff07.jpg)

<details>
<summary>line</summary>

| Willingness to pay (€) | Standard surgery | Navigated surgery |
| ---------------------- | ---------------- | ----------------- |
| 0                      | 1.0              | 0.0               |
| 20000                  | 1.0              | 0.0               |
| 40000                  | 0.95             | 0.1               |
| 60000                  | 0.85             | 0.2               |
| 80000                  | 0.75             | 0.25              |
</details>

![](images/c02ed758a376e4c0be425e8e8e6edd678aeffed33adbdf03e21011b30d7c490b.jpg)

<details>
<summary>line</summary>

| Willingness to pay (€) | Standard surgery | Navigated surgery |
| ---------------------- | ---------------- | ----------------- |
| 0                      | 1.0              | 0.0               |
| 20000                  | 0.9              | 0.2               |
| 40000                  | 0.7              | 0.4               |
| 60000                  | 0.5              | 0.5               |
| 80000                  | 0.5              | 0.5               |
</details>

Fig. 3 A and C show Cost-effectiveness planes for LARC (A) and LRRC (C) showing the incremental Quality Adjusted Life Years (QALYs) per incremental costs for navigated surgery versus standard surgery. The scatterplots show the mean differences in costs and outcomes from the data using 2000 bootstrap replicates. In both indications, most of the observations are in the North-East quadrant which indicates improved outcomes at higher costs. B and D show Cost-Effectiveness Acceptability Curves for LARC (B) and LRRC (D) presenting the probability of the cost-effectiveness of navigated surgery and standard surgery for a range of willingness to pay thresholds

significantly higher chance to achieve an R0 resection (79% vs 49% (p = 0.047)) [12] we expected a larger QALY difference. In a best-case situation, having a lower risk of progression with an R0 resection, the ICER could drop to €23.648 (Fig. 3). Based on the current evidence base [16, 39, 40], it could be expected that a stronger relation between R0 and reduced risk on progression is found when the analysis is based on a larger dataset, and progression of disease is stratified in local recurrence and distant metastasis. This would result in a higher chance for navigated surgery to become cost-effective. It should, however, be noted that resection margin status is also influenced by tumor biology.

Although a strong relationship between an R0 resection and a reduced risk of progression was found in LARC, navigated surgery was not cost-effective in the base case analysis, since the difference in having an R0 resection between standard and navigated surgery was small [12]. Identification of clinical subtypes that would especially benefit from navigated surgery would be of interest to become cost-effective in LARC.

The navigation system costs seem another crucial aspect that influenced the results. One could consider pricing the navigation system related to its cost-effectiveness, as in value-based pricing. The threshold analysis showed that the maximum per patient cost may be €1.839 in LARC and €4.412 in LRRC. When the navigation system is used in multiple indications, the value base in each of these indications should be taken into account.

A. LARC   
![](images/d9accace03d1b224e59f6d861d8dbbb8c566886fa1b652e67f3c3d2368140db1.jpg)

<details>
<summary>bar</summary>

A. LARC
| Category | Min (€) | Max (€) |
| :--- | :--- | :--- |
| %R0 resection navigated surgery [86%-99%] | 240 | 350 |
| Transition from DF to PD after R1 1st year [0.147-0.358] | 240 | 340 |
| %R0 resection standard surgery [81%-88%] | 120 | 220 |
| % of death due to progression 1st year [0.042-0.13] | 210 | 110 |
| Transition from DF to PD after R0 1st year [0.071-0.136] | 120 | 190 |
| Costs of navigation system [€2541-€4235] | 120 | 170 |
| Utility score of the Disease Free state [0.76-0.91] | 150 | 130 |
| Utility score of the Progression state [0.67-0.87] | 120 | 160 |
| % of death due to progression 2nd and 3rd year [0.001-0.091] | 80 | 60 |
| Transition from DF to PD after R0 2nd year [0.024-0.070] | 80 | 50 |
| Transition from DF to PD after R1 3rd year [0.001-0.067] | 80 | 50 |
| Transition costs stable to progression [€11162-€18604] | 80 | 50 |
| Transition from DF to PD after R1 2nd year [0.001-0.067] | 80 | 50 |
| Transition from DF to PD after R0 3rd year [0.001-0.025] | 80 | 50 |
| State costs of DF [€369-€614] | 80 | 50 |
| State costs of PD [€439-€732] | 80 | 50 |
</details>

B. LRRC   
![](images/5f8684e871d1469f43a1d1b7e6728d1358833881890eb0b8d25df80f7fe4a8db.jpg)

<details>
<summary>bar</summary>

B. ERK
| Category | Min (€) | Max (€) |
| :--- | :--- | :--- |
| Transition from DF to PD after R1 1st year [0.16-0.34] | 105 | 350 |
| Transition from DF to PD after R0 1st year [0.08-0.24] | 35 | 280 |
| % of death due to progression 1st year [0.06-0.21] | 90 | 45 |
| %R0 resection navigated surgery [70%-88%] | 75 | 45 |
| Transition from DF to PD after R0 2nd year [0.03-0.17] | 50 | 75 |
| %R0 resection standard surgery [41%-57%] | 50 | 75 |
| Costs of navigation system [€2541-€4235] | 60 | 70 |
| Transition from DF to PD after R1 2nd year [0.08-0.24] | 65 | 55 |
| Utility score of the Disease Free state [0.76-0.91] | 65 | 55 |
| Utility score of the Progression state [0.67-0.87] | 55 | 60 |
| Transition costs stable to progression [€9830-€16384] | 45 | 35 |
| % of death due to progression 2nd and 3rd year [0.001-0.19] | 45 | 35 |
| State costs of DF [€369-€614] | 45 | 35 |
| State costs of PD [€439-€732] | 45 | 35 |
</details>

Fig. 4 Sensitivity analyses. Tornado diagram showing the results of the one-way sensitivity analysis. A shows the results for the LARC group with a deterministic ICER of €136.604. B shows the results for the LRRC group with a deterministic ICER of €52.510. The scales of both figures are different and the gap on x-axis shows that a different scale is used after the gap. A dotted line is placed at the willingness to pay threshold of €80.000 which is used in the Netherlands. DF = disease free, PD = progression of disease, R0 = radical resection, R1 = a positive surgical margin

A final important finding is that, when investing in a navigation system, users should aim for optimal capacity use. For instance by organizing centralization of care or identification of multiple indications where navigation could be of added value (Table 4). This is especially the case when a hospital has to invest in a hybrid OR. Currently, the use of navigation is piloted in multiple oncologic indications [19, 41, 42], which could facilitate the future adoption of navigated surgery.

This study presents results from the first cost-effectiveness analysis for navigated surgery based on the first clinical data available. The results could inform its further development and the start of subsequent clinical or pilot studies. Further strengths include, (i) the inclusion of

A. Scenario 1 and 2 combined   
![](images/35ce9f8e30e56b7d86c78bd27418439df04d3876d707109110367cbd69b8a245.jpg)

<details>
<summary>line</summary>

| Utilization rate navigation | ICER LARC | ICER LRRC |
| --------------------------- | --------- | --------- |
| 10%                         | 270000    | 105000    |
| 20%                         | 225000    | 90000     |
| 30%                         | 205000    | 85000     |
| 40%                         | 195000    | 80000     |
| 50%                         | 190000    | 75000     |
</details>

B. Scenario 2   
![](images/940012cf43831f37441a2f7aa07cf210a7e2b991f9561a0abb8fdd65b0be3362.jpg)

<details>
<summary>line</summary>

| Utilization rate navigation | LARC     | LRRC     |
| --------------------------- | -------- | -------- |
| 10%                         | €140000  | €55000   |
| 20%                         | €95000   | €35000   |
| 30%                         | €75000   | €25000   |
| 40%                         | €65000   | €22000   |
| 50%                         | €60000   | €20000   |
</details>

Fig. 5 Graphical illustration of the scenario analysis. Shows the impact of varying the utilization rate of the navigation technology on the ICER. A shows the ICER for multiple utilization rates of navigation for the combination of scenario 1 and 2. Scenario 1 includes the construction costs for a hybrid OR when a hospital does not have this yet. In Scenario 2 the navigation system was used for 50%. B shows the ICER for multiple utilization rates of navigation of Scenario 2

tunnel states to incorporate time-to-event information, and (ii) the utility values that were based on prospective data from a relatively large $n=63$ and similar patient group, that showed utility values that were in line with literature [43].

This study has several limitations that should be acknowledged. In general, early cost-effectiveness analyses are associated with uncertainty in the input parameters, for example, because of small sample sizes and suboptimal study designs. Therefore, the outcomes could be debatable. This is also shown by the CEAC (using the uncertainty surrounding the input parameters) for LRRC, showing a probability of only 52% that navigated surgery is cost-effective at a WTP threshold of €80.000. The value of information analysis shows that it is worthwhile to perform further research especially focusing on the risk of having an R0 or R1 resection after navigated- and standard surgery. Furthermore, based on the available dataset it was not possible to test the influence of a longer follow-up period.

More specifically, our analysis is limited because we did not incorporate patient characteristics (e.g. tumor stage, tumor location) since treatment history could affect margin status and progression of disease $[44]$ . Besides, by not stratifying for local recurrence and distant metastasis, evaluation of the relationship between R0 and a reduced risk of progression was challenging, because achieving an R0 resection has a limited influence on reducing the risk of metastases.

Related to the utility values, the utility-scores for LARC and LRRC were assumed to be equal, although LRRC patients are expected to receive multiple chemotherapy lines which is expected to result in lower utility values $[45]$ . Furthermore, we should note that we could not include all patients included in the ongoing trial because of missing data. Another limitation is that we used the 6-month questionnaire to base our utility value on for the disease-free and progression state, which is likely to overestimate the utility-score since patients experience more complaints as the disease progresses. As the utility scores

![](images/03c8602c77e6244580828cc2a33cbf77db5614a6e41599694d7bbcf605149cf3.jpg)

<details>
<summary>bar</summary>

| Category | Expected value of perfect information (in millions) |
| :--- | :--- |
| Surgical outcome NAV | 8.84 |
| Surgical outcome CON | 6.46 |
| Progression of disease | 0 |
| Utilities | 0.02 |
| Treatment costs | 0.02 |
| Progression and follow-up costs | 0 |
</details>

Fig. 6 The expected value of perfect information for parameter groups for LRRC

show a large impact on the cost-effectiveness results, utility values for 1 and 2 years after surgery should be incorporated. Furthermore, as curative surgery (R0 resection) is expected to decrease pain complaints, we expect to underestimate the QoL benefit of navigated surgery leading to rather negative ICER estimates. Another issue is related to the costs used in the analysis. The costs were calculated from a Dutch healthcare perspective while the costs of the navigation system were based on list prices and expert elicitation. Finally, since the navigation system is a new surgical tool, a learning curve may be present which potentially underestimates the performance of navigated surgery.

This analysis should be seen as a first step in evaluating the added value of navigated surgery for LRRC and LARC. Although there is a tendency in surgery to not formally evaluate incremental improvements in technology, we recommend comparing navigated and standard surgery prospectively, preferably multi-center, in terms of resection margin rate (R0, R1, and R2), complication rate, QoL and utility values. Based on this data, the cost-effectiveness should be updated using the large (inter) national studies presenting survival after R0 and R1/R2 margins $[15, 16, 39, 40]$ to inform adoption and reimbursement decisions, and validate the results of this study. Furthermore, we suggest validating the mapping study of Wong et al. when QoL and utilities are measured at several moments in time since the EQ5D-5L seems not sensitive to capture CRC specific complaints $[46]$ . Subsequently, to inform decision-makers with the best available evidence, also on potential unforeseen effects e.g. learning curve, the cost-effectiveness analysis should be updated (iterative approach $[47]$ ) when more robust survival data is available.

# Conclusion

Based on this early cost-effectiveness analysis, navigated surgery is expected to be cost-effective in LRRC patients and it has the potential to become cost-effective for LARC patients. Further research, preferably prospective studies, is necessary to validate these results and conclude on routine use of navigation and reimbursement of navigated surgery. Since the navigation system seems to be associated with high costs per patient, it is crucial to, when hospitals invest in such an innovative medical device, use it optimally (centralization of care) and seek other indications where it could be of additional value.

# Abbreviations

APR: Abdominoperineal resection; CE: Cost-effectiveness; CEAC: Cost-effectiveness acceptability curves; CRC: Colorectal cancer; DRG: Diagnosis-related group; EVPI: Expected value of perfect information; ICER: Incremental cost-effectiveness ratio; LAR: Los anterior resection; LARC: Locally advanced rectal cancer; LRRC: Locally recurrent rectal cancer; LY: Life-years; NKI – AVL: Netherlands cancer institute – Antoni van leeuwenhoek; QALY: Quality adjusted life years; QoL: Quality of Life; R0: Negative resection margin; R1: Positive resection margin; WTP: Willingness to pay.

# Supplementary Information

The online version contains supplementary material available at https://doi.org/10.1186/s12885-022-09561-w.

Additional file 1.: Supplement 1. Patient characteristics of patients included in the study of Kok et al. 2020. Supplement 2. Characteristics of patients included in the prospective study evaluating quality of life.

Supplement 3. Overview of sources for input of the model. Supplement

4. Details on the additional costs for using the navigation system during surgery. Supplement 5. Details of state costs. Supplement 6. Detailed information on the scenario input parameters. Supplement 7. Probabilistic results for LARC and LRRC when Scenario 2 is present. Supplement 8. Graphical visualization of Expected Value of Perfect Information.

# Acknowledgments

We want to thank the research group of Theo Ruers and Esther Kok for providing the clinical data for this analysis. Furthermore, we want to thank all the patients that participated in the prospective cohort study evaluation quality of life after colorectal surgery.

# Authors' contributions

AK and ML drafted the manuscript. WvH, VR and EK were involved in reviewing the data, analysis and critically reviewed the manuscript. GB and TR critically reviewed the manuscript and verified the clinical data incorporated in the analysis. All authors read and approved the final manuscript.

# Funding

There was no specific funding for the research, the research was financially supported by the Netherlands Cancer Institute. The funding body played no role in the design of the study and collection, analysis, and interpretation of data and in writing the manuscript.

# Availability of data and materials

The datasets used and/or analysed during the current study are stored in the repository of the Netherlands Cancer Institute and are available from the corresponding author on reasonable request. All input parameters are presented in the manuscript or supplementary files and can be used to reproduce our analysis and results.

# Declarations

# Ethics approval and consent to participate

The data collection of the retrospective data collected in this study and the pervious analysis by Kok et al. 2020 was approved by the IRB of the Netherlands Cancer Institute. Patients in this retrospective analysis provided written informed consent. The medical ethical committee approved the prospective cohort study that sampled quality of life questionnaires among patients undergoing surgery for colorectal cancer. For their participation, written informed consent was required.

# Consent for publication

Not applicable.

# Competing interests

The authors declare that they have no competing interests.

# Author details

$^{1}$ Health Technology and Services Research, University of Twente, Enschede, The Netherlands. $^{2}$ Division of Psychosocial Research and Epidemiology Netherlands Cancer Institute, Antoni van Leeuwenhoek, Amsterdam, The Netherlands. $^{3}$ Department of Surgical Oncology, Netherlands Cancer Institute – Antoni van Leeuwenhoek, Amsterdam, The Netherlands. $^{4}$ Faculty TNW, Group Nanobiophysics, Twente University, Enschede, The Netherlands.

# Received: 30 September 2020 Accepted: 17 April 2022

Published online: 06 May 2022

# References

1. Glynne-Jones R, Wyrwicz L, Tiret E, Brown G, Rödel C, Cervantes A, et al. Rectal cancer: ESMO Clinical Practice Guidelines for diagnosis, treatment and follow-up†. Ann Oncol. 2017;28(suppl\_4):iv22–40.   
2. Ned Tijdschr Geneeskd. Behandeling van lokaal recidiverend rectumcarcinoom. 2015;159;A8199.   
3. Integraal Kankercentrum Nederland. Darmkanker in Nederland: cijfers uit de Nederlandse Kankerregistratie. https://www iknl.nl/nieuws/2019/darmkanker-in-nederland-cijfers-uit-de-nederlandse. Accessed 4 Feb 2020.   
4. Lee Y-C, Lee Y-L, Chuang J-P, Lee J-C. Differences in survival between Colon and Rectal Cancer from SEER data. PLoS One. 2013;8:e78709.   
5. Bhangu A, Ali SM, Darzi A, Brown G, Tekkis P. Meta-analysis of survival based on resection margin status following surgery for recurrent rectal cancer. Color Dis. 2012;14:1457–66.   
6. Bonjer HJ, Deijen CL, Abis GA, Cuesta MA, van der Pas MHGM, de Lange-de Klerk ESM, et al. A randomized trial of laparoscopic versus open surgery for rectal Cancer. N Engl J Med 2015;372:1324–1332.   
7. Rickles AS, Dietz DW, Chang GJ, Wexner SD, Berho ME, Remzi FH, et al. High rate of positive circumferential resection margins following rectal Cancer surgery. Ann Surg. 2015;262:891–8.   
8. Yang HY, Park SC, Hyun JH, Seo HK, Oh JH. Outcomes of pelvic exenteration for recurrent or primary locally advanced colorectal cancer. Ann Surg Treat Res. 2015;89:131.   
9. Palmer G, Martling A, Cedermark B, Holm T. A population-based study on the management and outcome in patients with locally recurrent rectal Cancer. Ann Surg Oncol. 2007;14:447–54.   
10. Vermaas M, Ferenschild FTJ, Verhoef C, Nuyttens JJME, Marinelli AWKS, Wiggers T, et al. Total pelvic exenteration for primary locally advanced and locally recurrent rectal cancer. Eur J Surg Oncol. 2007;33:452–8.   
11. Tejedor P, Khan J. Surgical trends in the management of rectal cancer. Clin Oncol. 2018;3:1500.

12. Kok END, van Veen R, Groen HC, Heerink WJ, Hoetjes NJ, van Werkhoven E, et al. Association of image-guided navigation with complete resection in patients with locally advanced primary and recurrent rectal cancer: a nonrandomized trial. JAMA Netw Open. 2020;3:e208522–2.   
13. IJzerman MJ, Steuten LM. Early assessment of medical technologies to inform product development and market access: a review of methods and applications. Appl Health Econ Health Policy. 2011;9:331–47.   
14. Vera R, Aparicio J, Carballo F, Esteva M, González-Flores E, Santianes J, et al. Recommendations for follow-up of colorectal cancer survivors. Clin Transl Oncol. 2019;21:1302–11.   
15. Collaborative PE. Surgical and survival outcomes following pelvic Exenteration for locally advanced primary rectal Cancer. Ann Surg. 2019;269:315–21.   
16. Kelly ME, Glynn R, Aalbers AGJ, Abraham-Nordling M, Alberda W, Antoniou A, et al. Factors affecting outcomes following pelvic exenteration for locally recurrent rectal cancer. Br J Surg. 2018;105:650–7.   
17. NCCN. NCCN guidelines for patients colon cancer. 2018.   
18. Zorginstituut Nederland (Dutch institute of healthcare). Richtlijn voor het uitvoeren van economische evaluaties in de gezondheidszorg. 2016.   
19. Nijkamp J, Kuhlmann KFDKFDD, Ivashchenko O, Pouw B, Hoetjes N, Lindenberg MAMA, et al. Prospective study on image-guided navigation surgery for pelvic malignancies. J Surg Oncol. 2019;119:510–7.   
20. Centraal Bureau van Statistiek. Prognose periode-levensverwachting; geslacht en leeftijd. 2019.   
21. Dutch Healthcare Authority (NZa). DBC-zorgproduct 029199032 (DRG open tariff). 2018. https://www.opendisdata.nl/msz/zorgproduct/029199032. Accessed 2 Feb 2020.   
22. Hakkaart-van Roijen L, van der Linden N, Bouwmans C, Kanters T, Swan TS. Manual for cost research: methods and standard cost prices for economic evaluations in health care. Diemen: Institute for Medical Technology Assessment; 2015.   
23. Dutch Healthcare Authority (NZa). DBC product finder for tariffs. 2019. http://dbc-zorgproducten-tarieven.nza.nl. Accessed 2 Feb 2017.   
24. Dutch Federation of Academic Medical Centers. Collective labor agreement 2018–2020 for academic medical centers. Utrecht; 2018.   
25. Patel S, Lindenberg M, Rovers MM, van Harten WH, Ruers TJM, Poot L, et al. Understanding the costs of surgery: a bottom-up cost analysis of both a hybrid operating room and conventional operating room. Int J Health Policy Manag. 2020. https://doi.org/10.34172/ijhpm.2020.119.   
26. Renehan AG. Techniques and outcome of surgery for locally advanced and local recurrent rectal Cancer. Clin Oncol. 2016;28:103–15.   
27. Briggs A, Claxton K, Sculpher M. Decision Modelling for health economic evaluation. New York: Oxford university press; 2006.   
28. Hawkins N, Sculpher M, Epstein D. Cost-effectiveness analysis of treatments for chronic disease: using R to incorporate time dependency of treatment response. Med Decis Mak. 2005;25:511–9.   
29. Sonnenberg FA, Beck JR. Markov models in medical decision making. Med Decis Mak. 1993;13:322–38.   
30. Jeong K, Cairns J. Systematic review of health state utility values for economic evaluation of colorectal cancer. Heal Econ Rev. 2016;6:36.   
31. Ramos-Goni JM, Rivero-Arias O. Eq 5d: a command to calculate index values for the EQ-5D quality-of-life instrument. Stata J. 2011;11:120–5.   
32. Integraal Kankercentrum Nederland. Colorectaal carcinoom. 2014.   
33. Zorginstituut Nederland (Dutch institute of healthcare). Medicijnkosten (costs of pharmaceuticals). 2019. http://www.medicijnkosten.nl/. Accessed 2 Apr 2020.   
34. Versteegh MM, Ramos IC, Buyukkaramikli NC, Ansaripour A, Reckers-Droog VT, Brouwer WBF. Severity-adjusted probability of being cost effective. Pharmacoeconomics. 2019;37:1155–63.   
35. Girling A, Lilford R, Cole A, Young T. Headroom approach to device development: current and future directions. Int J Technol Assess Health Care. 2015;31:331–8.   
36. Willan AR, Pinto EM. The value of information and optimal clinical trial design. Stat Med. 2005;24:1791–806.   
37. Klaver CEL, Gietelink L, Bemelman WA, Wouters MWJM, Wiggers T, Tollenaar RAEM, et al. Locally advanced Colon Cancer: evaluation of current clinical practice and treatment outcomes at the population level. J Natl Compr Cancer Netw. 2017;15:181–90.   
38. IKNL. Oncoline guideline on colorectal cancer. https://www.oncoline.nl/colorectaalcarcinoom. Accessed 20 Dec 2019.

39. Westberg K, Palmer G, Hjern F, Johansson H, Holm T, Martling A. Management and prognosis of locally recurrent rectal cancer – a national population-based study. Eur J Surg Oncol. 2018;44:100–7.   
40. Harris CA, Solomon MJ, Heriot AG, Sagar PM, Tekkis PP, Dixon L, et al. The outcomes and patterns of treatment failure after surgery for locally recurrent rectal Cancer. Ann Surg. 2016;264:323–9.   
41. Ivashchenko O, Pouw B, Van Veen R, Kuhlmann KF, Kok NF, Klompenhouwer EG, et al. Intraoperative electromagnetic navigation towards liver tumors. Int J Comput Assist Radiol Surg. 2018;13:1–273.   
42. Janssen N, Eppenga R, Peeters MJV, Van Duijnhoven F, Oldenburg H, Van Der Hage J, et al. The use of real-time tumor tracking to perform navigation guided breast conserving surgery: feasibility of the Calypso systemTM in breast phantoms. Int J Comput Assist Radiol Surg. 2017;12:1–286.   
43. Ness RM, Holmes AM, Klein R, Dittus R. Utility valuations for outcome states of colorectal cancer. Am J Gastroenterol. 1999;94:1650–7.   
44. Bhangu A, Mohammed Ali S, Brown G, Nicholls RJ, Tekkis P. Indications and outcome of pelvic exenteration for locally advanced primary and recurrent rectal cancer. Ann Surg. 2014;259:315–22.   
45. Mayrbäurl B, Giesinger JM, Burgstaller S, Piringer G, Holzner B, Thaler J. Quality of life across chemotherapy lines in patients with advanced colorectal cancer: a prospective single-center observational study. Support Care Cancer. 2016;24:667–74.   
46. Wong CKH, Lam CLK, Wan YF, Rowen D. Predicting SF-6D from the European organization for treatment and research of cancer quality of life questionnaire scores in patients with colorectal cancer. Value Health. 2013. https://doi.org/10.1016/j.jval.2012.12.004.   
47. Vallejo-Torres L, Steuten LMG, Buxton MJ, Girling AJ, Lilford RJ, Young T. Integrating health economics modeling in the product development cycle of medical devices: a Bayesian approach. Int J Technol Assess Health Care. 2008;24:459–64.

# Publisher's Note

Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

# Ready to submit your research? Choose BMC and benefit from:

• fast, convenient online submission   
• thorough peer review by experienced researchers in your field   
• rapid publication on acceptance   
- support for research data, including large and complex data types   
- gold Open Access which fosters wider collaboration and increased citations   
• maximum visibility for your research: over 100M website views per year

At BMC, research is always in progress.

Learn more biomedcentral.com/submissions

![](images/bc04d6b2ec953a3856d3cb9a048e5128dda49f15f90ef7e1bbc0ac7f40c4cafa.jpg)

BMC