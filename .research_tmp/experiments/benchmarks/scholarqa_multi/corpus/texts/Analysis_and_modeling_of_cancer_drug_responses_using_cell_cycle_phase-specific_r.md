# Analysis and modeling of cancer drug responses using cell cycle phase-specific rate effects

Received: 18 October 2021

Accepted: 29 May 2023

Published online: 10 June 2023

Check for updates

Sean M. Gross<sup>1,3</sup>, Farnaz Mohammadi <sup>2,3</sup>, Crystal Sanchez-Aguila<sup>1</sup>, Paulina J. Zhan<sup>1</sup>, Tiera A. Liby <sup>1</sup>, Mark A. Dane <sup>1</sup>, Aaron S. Meyer <sup>2,4</sup> & Laura M. Heiser <sup>1,4</sup>

Identifying effective therapeutic treatment strategies is a major challenge to improving outcomes for patients with breast cancer. To gain a comprehensive understanding of how clinically relevant anti-cancer agents modulate cell cycle progression, here we use genetically engineered breast cancer cell lines to track drug-induced changes in cell number and cell cycle phase to reveal drug-specific cell cycle effects that vary across time.

We use a linear chain trick (LCT) computational model, which faithfully captures drug-induced dynamic responses, correctly infers drug effects, and reproduces influences on specific cell cycle phases. We use the LCT model to predict the effects of unseen drug combinations and confirm these in independent validation experiments. Our integrated experimental and modeling approach opens avenues to assess drug responses, predict effective drug combinations, and identify optimal drug sequencing strategies.

Effective treatment of cancer requires drug combinations; however, rational identification of effective combination therapy regimens remains challenging $^{2-5}$ . Many anti-cancer agents are designed to impact cell proliferation and viability, which suggests that incorporating information about how individual drugs modulate cell cycle behavior can lead to improved predictions about drug combination effects.

The mammalian cell cycle is typically separated into four linked phases $(\mathrm{G}_1, \mathrm{S}, \mathrm{G}_2, \text{and M})$ with multiple checkpoints (restriction point, DNA damage checkpoint, and the spindle assembly checkpoint) $^{6-9}$ , each relying on distinct molecular programs which results in minimal correlation between cell cycle phase durations in individual cells $^{10}$ .

This independence between phases and checkpoints has implications for cancer treatment because many cancer drugs directly target different aspects of the cell cycle; for example, CDK4/6 inhibitors block progression out of $\mathrm{G}_1$ phase $^{11}$ , while the nucleoside analog gemcitabine activates the DNA damage checkpoint by targeting DNA synthesis during Sphase $^{12}$ . Together, these findings imply that drug-induced

changes to cell numbers can be achieved through distinct cell cycle-dependent molecular mechanisms. For example, these observations suggest that combing two drugs that each reduce the rate of $G_{1}$ progression will lead to deeper reductions in the rate of $G_{1}$ progression, rather than an increase in cell death. Further, this framework also predicts dose-dependent impacts, for example: at sub-saturating doses, $G_{1}$ effects will add together to reduce cell numbers while at higher saturating doses the cell number will peak at the maximum cytostatic effect.

This general idea of drug combination efficacy was recently explored in a study of the multi-drug CHOP protocol, which showed that the effectiveness of this drug combination for treatment of non-Hodgkin Lymphoma could be attributed to the fact that each agent had non-overlapping cytotoxic effects<sup>13</sup>. The effectiveness of the CHOP protocol also demonstrates the benefit of drug combinations to improve patient outcomes. Considering both cell cycle and cell death effects in greater detail, therefore, has the potential to significantly improve drug combination predictions.

nature communications

#

Article

https://doi.org/10.1038/s41467-023-39122-z

<sup>1</sup>Department of Biomedical Engineering, Knight Cancer Institute, Oregon Health & Science University, Portland, OR, USA. <sup>2</sup>Department of Bioengineering, University of California, Los Angeles; Jonsson Comprehensive Cancer Center, University of California at Los Angeles, Los Angeles, CA, USA. <sup>3</sup>These authors contributed equally: Sean M. Gross, Farnaz Mohammadi. <sup>4</sup>These authors jointly supervised this work: Aaron S. Meyer, Laura M. Heiser.

e-mail: heiserl@ohsu.edu

Nature Communications| (2023)14:3450

1

The classic approach to quantifying drug response assumes that cells are undergoing exponential growth at the time of drug treatment and then calculates the number of cells $72\mathrm{H}$ after drug addition[6,14-16], however more recent approaches have used multiple markers to gain more detailed insights into drug effects[17,18]. Other approaches to quantify drug response include compartmental models such as pharmacokinetic and pharmacodynamic (PK-PD) models that consider drug uptake and population dynamics[19].

Recent advances in methodological and quantitative approaches enable assessment of the impact of therapies on cell growth rates, rather than static cell counts[20], which yields more robust correlations between molecular features and drug sensitivity[20,21]. While growth rate approaches significantly improve quantification, they provide limited information about cell cycle effects.

A related approach, fractional proliferation, which models the number of cycling, quiescent, and dying cells in a drug-treated population, incorporates growth rates and assumes that cells irreversibly exit the cell cycle into quiescence[22]; however, recent studies demonstrate that cells may not irreversibly exit the cell cycle and instead may extend the duration of a specific cell cycle phase before restarting progression through the cell cycle[23].

These prior findings motivate our interest to deeply assess the influence of drugs on specific cell cycle phases and progression through the cell cycle.

In this work, we quantify and incorporate cell cycle phase effects in an analysis of drug responses. We use live-cell imaging of a panel of molecularly diverse breast cancer cells engineered to express a cell cycle reporter and track the dynamics of cell number and cell cycle phase in response to single drugs and drug combinations. Across single drugs, we observe distinct cell cycle effects that lead to similar final cell numbers, with phase-specific responses that are oscillatory over time due to their temporal impacts on the cell cycle.

To describe these responses, we develop a computational model that uses a linear chain trick (LCT) to account for the delay from cell cycle phase transit time upon drug treatment. The LCT model correctly infers single drug responses across time as well as the drug-induced oscillatory cell cycle dynamics. We use this model to predict the effect of unseen combinations of drugs that impact different aspects of the cell cycle. Experimentally testing several drug combinations validates that responses are primarily determined by the specific cell cycle effects of each drug pair.

These studies reveal the complexity of cell behavior underlying drug responses, provide mechanistic insights into how individual drugs modulate cell numbers, and yield a framework to rationally model and predict drug combinations.

# Results

# Drug treatments induce distinct changes in cell number and cell cycle phasing

To track drug responses in individual cells, we genetically engineered HER2+ AU565 breast cancer cells to stably express the HDHB cell cycle reporter $^{23}$ and a nuclear-localized red fluorescent protein (Fig. 1A, B). Cells were treated with escalating doses of five clinically relevant breast cancer drugs, each targeting different cell cycle phases (Fig. 1C). Cells were imaged every 30 minutes for $96\mathrm{H}$ and the number of cells in each cell cycle phase and total cell numbers were quantified. Each drug effectively reduced cell numbers in a dose-dependent manner (Fig.

1D, Supplementary Fig. 1). As expected, p
aclitaxel, gemcitabine, and doxorubicin led to cytotoxic effects indicated by the final cell numbers dropping below the starting cell numbers (Fig. 1D) $^{24,25}$ . In contrast, at the highest doses of palbociclib and lapatinib, final cell numbers were approximately equal to the starting cell numbers, suggesting cytostatic effects.

For each drug, the pattern of cell counts varied across time; at high doses responses tended to reach a peak and then decline as the duration of drug exposure increased—an effect most marked for $30\mathrm{nM}$ gemcitabine where the relative cell number declined from 1.1 at 48H to 0.5 at 96H (Fig. 1D) $^{20,21}$ . The fraction of $\mathrm{S - G_2}$ cells varied over time and showed both drug- and dose-specific effects (Fig. 1E, F).

For example, lapatinib and palbociclib initially reduced the fraction of cells in S-G $_2$ phase in a dose-dependent manner, whereas gemcitabine and doxorubicin initially increased this fraction. Of note, intermediate doses of lapatinib (50 nM) and paclitaxel (3 nM) induced oscillating cell cycle responses, with an initial S-G $_2$ reduction near 30H, followed by a second S-G $_2$ reduction at 84H. In sum, this approach revealed drug-specific cell cycle changes across time, which confirms that these drugs yield similar final numbers through distinct impacts on the cell cycle.

# A dynamical model captures drug-induced changes to cell cycle behavior

A common approach to model drug effects assumes exponential growth that varies as a function of drug dose $^{26}$ . This approach, although informative, cannot explain the cell cycle dynamics described above and motivated the development of a dynamical model to capture the observed behavior. As an initial model, we defined a system of ordinary differential equations (ODEs) with transitions between $G_{1}$ and $S - G_{2}$ .

The parameters of the ODE model were the cell cycle phase progression and death rates, which were assumed to follow a Hill function with respect to drug concentration (Supplementary Table 1). This model failed to fit the experimental data of $G_{1}$ and $S - G_{2}$ cell numbers (Supplementary Fig. 2); furthermore, dynamical systems theory dictates that this model is unable to oscillate under any reasonable parameterization $^{27}$ .

To address these limitations and capture the observed oscillatory temporal dynamics, we incorporated into the model the observations that phase durations follow a gamma distribution and are uncorrelated $^{10}$ (Supplementary Fig. 3A). Gamma and related distributions model each cell cycle phase as a series of steps, with the key feature that they can model processes wherein there is always some measurable duration before a system can move to the next state (e.g., a cell progressing through the cell cycle).

The number of steps in each phase were determined by estimating the shape parameter of the gamma distributions fitted to single cell measurements of $\mathrm{G}_1$ and $\mathrm{S - G_2}$ phase durations measured in the untreated control condition $^{28}$ . This resulted in partitioning the $\mathrm{G}_1$ phase into 8 steps and $\mathrm{S - G_2}$ phase into 20 steps (Fig. 2A). We incorporated a linear chain trick (LCT) into our model, which creates similarly-distributed time delays in the cell cycle phase durations through a mean-field system of ODEs $^{29}$ .

The model was further simplified by sharing parameters that were not drug specific, such as the number of cell cycle subphases and the initial fraction of cells in $\mathrm{G}_1$ phase. We then fit all five drug dose responses, varying the drug-specific and shared parameters, simultaneously. Incorporation of this component enabled the model to capture the experimentally observed oscillatory cell cycle behavior and cell cycle phase-specific drug effects.

We computed the fitting error of the two modeling frameworks by calculating the sum of squared error of the difference between the data and model predictions across all concentrations and observed that the LCT model had lower error terms (Fig. 2B). The fits to lapatinib and palbociclib were particularly improved by the model refinement. Examples of dose-response curves and model fits for lapatinib and gemcitabine are shown in Fig. 2C-H. Importantly, the model captured the dose-dependent changes to G1 and S-G2 populations as well as the oscillatory dynamics.

Estimating the cell cycle phase progression and death rates also enabled calculation of the accumulated amount of cell death across time using inferred cell counts at each phase (Fig. 2E,H). The LCT model also performed well for each of the other drugs (Supplementary Fig. 4A-F).

The phase durations and cell death probabilities inferred from the LCT model varied with drug treatment (Fig. 2I-L). Comparison of inferred effects at the half maximum concentration (EC50) revealed that lapatinib and palbociclib treatments lead to longer average $\mathrm{G}_{1}$ phase durations compared to untreated cells (Fig. 2I, J), a $10\%$ higher chance of cell death in $\mathrm{G}_{1}$ phase for lapatinib treated cells, and a slight

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

2

![](dt=2026-03-17/ht=18/8f9035bb5e6f2f663c2049a8630945bff2def37792901c37941cb97090f12690.jpg)

![](dt=2026-03-17/ht=18/d6acdf975aeeef37d3128e1d5bd48095bc766cdc29947853d9215da4946c5911.jpg)

![](dt=2026-03-17/ht=18/28b27dda0deea395a320e4b65a40672c7f9247b214ba50bbee896ccd47b8f05d.jpg)

![](dt=2026-03-17/ht=18/5d0baa23a0136c069c574593c128154831075f8ae23d44db012113894bc8bdea.jpg)

![](dt=2026-03-17/ht=18/da688a7a13dc9e108d9c964ca078f680b6ed0aa2fe85d974c6042023c55c6b0f.jpg)

![](dt=2026-03-17/ht=18/85d4625f5ef6b739762f9ed19513af90afa5a97f4d409b20adada0c83f575edd.jpg)

![](dt=2026-03-17/ht=18/2542f01685eaf8e4bee6ce902fc05cae0defb8a927fc6e8ace9834660284a82f.jpg)

![](dt=2026-03-17/ht=18/2d587d2114bb49a584cfa7064684170cc41e586e8b08e271fa4d448e142d6947.jpg)

![](dt=2026-03-17/ht=18/61fad58a11dceea3f77d99fe2a19840723e85a1b6848e813f95b54014bbf5b0c.jpg)

![](dt=2026-03-17/ht=18/f74fb4cb1fa89adfe93703e281920615e264a0990c41455f46fe7d482db82dc0.jpg)

![](dt=2026-03-17/ht=18/2c60443702b0879eac619e9cc6752e5074316d1fb75a87ad7e3f49c61749ce7f.jpg)

![](dt=2026-03-17/ht=18/26c8e9f6d0b386b71497b0ffbd2013c4b78980cbea83f7bc4917ba806feae5c2.jpg)

![](dt=2026-03-17/ht=18/b95f66079fd8e42196fbc0ac1581735181925b8bdd06ef5e33ddcd7e345fa7f4.jpg)

![](dt=2026-03-17/ht=18/b31dc68969fc5e09c7e78434205c6ce4ad909edc48f755add8c3388c72e099bb.jpg)

![](dt=2026-03-17/ht=18/82f19365a9291fa1f7f86b8813381f2083873cf965d126ebdc0c8c0b3560ebce.jpg)

![](image)
3614ce42a12fbed4abd4069ab3646ba95998a7f437ba.jpg)

![](dt=2026-03-17/ht=18/0f9e2976a55021c7a633eb33cc288b2a78a40d9977c9a67ea70df31a3390eb2a.jpg)

chance of cell death in $\mathrm{S - G_2}$ after palbociclib treatment (Fig. 2K, L). The model also inferred that gemcitabine induces an increase in $\mathrm{S - G_2}$ durations and greater chance of cell death in $\mathrm{S - G_2}$ phase as compared to untreated cells (Fig. 2G, H). Finally, a $10\%$ chance of cell death at the EC50 concentration (2.4 nM) was inferred in late $\mathrm{G_2}$ phase for cells treated with paclitaxel as compared to untreated controls (Fig. 2J and Supplementary Fig. 3J).

# Analysis of single cell responses confirms model inferences and reveals drug-specific cell cycle phase effects

We developed model parameters from the average population response at each timepoint, which facilitates robust model development by leveraging information from a large number of cells. Importantly, as described above, the LCT model infers aspects of drug responses that can be quantified at the individual cell level—including cell cycle phase duration and cell cycle-specific death. We therefore tracked single cells in the image time course data to quantify cell cycle

phase durations and also cell death events associated with specific drug treatments and concentrations (Supplementary Fig. 3B). Quantification of cell death events also enables direct assessment of whether drug effects are cytotoxic or cytostatic. The first complete cell cycle was analyzed to examine early drug effects. We also quantified the relative fate outcomes for the progeny of cells observed at time OH (relative to drug addition) that later underwent division, which provides insights into the influence of drug treatment on G1 and S-G2 durations (Supplementary Fig. 3C).

As expected, in the untreated condition, most cells $(93\%)$ present at OH underwent cell division. In contrast, at the highest lapatinib and gemcitabine doses, $32\%$ and $61\%$ of the cells present at time OH failed to divide. Additionally, of the cells that did divide in these two conditions, only $10\%$ underwent a second division. For both drugs, lower doses showed more modest changes in the fraction of cells that divided as compared to untreated. As described below, we compared these experimentally-observed drug-induced cell cycle effects to those inferred by the LCT model.

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

3

![](dt=2026-03-17/ht=18/0091161476ce15a373c84b0df5c78f1e02236585328711395c5150e11bb0bd56.jpg)

![](dt=2026-03-17/ht=18/ea64f1acd8b5e5b6aa0fa1f3c5d0f0cd2fd5e271a08cb86df59055da620d1eb1.jpg)

![](dt=2026-03-17/ht=18/e7c0e8cfcefe5d2d848cc6040068c632cf42748e88faaf88befc0fccbe6e0aa8.jpg)

![](dt=2026-03-17/ht=18/1d58793a8607d7aac3143b85fc66afe35e98c6cc8d9fdd3a0290e4bd4d87bfeb.jpg)

![](dt=2026-03-17/ht=18/07442c102c80e2de47e9de8559e19c8d37087cbef94b3fb078da9d1f7873a030.jpg)

![](dt=2026-03-17/ht=18/5484977926ee1e8e8d2f764c8ed2bcaa049f3154e24bacea8b1394fa3df1ec39.jpg)

![](dt=2026-03-17/ht=18/cb01145ae5ae45f3411241c32d0abfe02a8861a88fa9c896eec4fdc12b18a1d8.jpg)

![](dt=2026-03-17/ht=18/cc4a225372321864fa03f0bf6862e21c78139705354349525fc7a13c662191cb.jpg)

![](dt=2026-03-17/ht=18/39f600008d21da3ae89f7aecc357ee74f2eaad64916434cdb42df06aa91639b5.jpg)

![](dt=2026-03-17/ht=18/4129467254165ff8db9173fd1eec0cb48693bd93b9d34b0db7aa3f4bd82926b5.jpg)

![](dt=2026-03-17/ht=18/980a106a0f9f12b069b00370493363138d6f9046a4c6fbda5a11b84fbc07d559.jpg)

![](dt=2026-03-17/ht=18/75ea48a3525ff4d5bf29ee4caa7a24fd0b8f772af978b1bea6dd9fdec2a15c72.jpg)

The model inferred that the predominant lapatinib effect was to extend $G_{1}$ durations from 22.3H in the untreated condition to 33.6H and 47.4H for 25 nM and 50 nM lapatinib, respectively (Fig. 3A). Experimentally, $G_{1}$ durations increased after lapatinib (mean 26.2H and 32.5H with 25 nM and 50 nM lapatinib, respectively) (Fig. 3A, B). We also observed that as the dose of lapatinib was increased, there was a corresponding increase in the variance of $G_{1}$ duration, indicating that cells showed greater variability in their response to lapatinib at higher doses (Fig. 3B). The model inferred only modest changes to S-G $_{2}$ durations or cell death, which was consistent with experimentally observed S-G $_{2}$ durations and cell death associated with lapatinib treatment (Fig. 3C,D).

The model inferred that oscillations in the percentage of $G_{1}$ cells after lapatinib treatment arise from the durations it takes to pass

through the cell cycle (see Fig. 2). To explore the mechanism underlying this behavior, we quantified cell cycle dynamics in individual cells. The relative fraction of cells undergoing cell division was reduced after lapatinib treatment, with effects emerging $\sim 24\mathrm{H}$ after treatment (Fig. 3E). This observation, together with the lengthening of the subsequent $G_{1}$ duration following cell division (Fig. 3B), can explain the cell cycle synchronization observed in the experimental data (see Fig. 1) and in the LCT model.

At the start of the assay, cells in $G_{1}$ are delayed in their time to division, while cells in S-G2 only become delayed at the onset of $G_{1}$ following division. In effect, this creates two populations of cells with distinct timing in the induction of drug effects. We observed a similar effect after treatment with palbociclib (Supplementary Fig. 3D).

For gemcitabine, the model inferred a slight acceleration of $G_{1}$ phases, which was recapitulated experimentally (Fig. 3A,F). The model

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

4

A

![](dt=2026-03-17/ht=18/a93ffbb88267b0acff323bc80e5a24c62b00d6c7395c4c71787f757dbcc1bd07.jpg)

<table><tr><td rowspan="2">Treatment</td><td colspan="2">G1 Duration</td><td colspan="2">S/G2 Duration</td><td colspan="2">Cell Death</td></tr><tr><td>Predicted (Hr)</td><td>Observed (Hr)</td><td>Predicted (Hr)</td><td>Observed (Hr)</td><td>Predicted (Relative Number)</td><td>Observed (Relative Number)</td></tr><tr><td>Control</td><td>22.7</td><td>19.4</td><td>32.9</td><td>22.3</td><td>0.00</td><td>0.14</td></tr><tr><td>Lapatinib 25nM</td><td>33.6</td><td>26.2</td><td>33.2</td><td>24.0</td><td>0.25</td><td>0.06</td></tr><tr><td>Lapatinib 50nM</td><td>47.4</td><td>32.5</td><td>36.1</td><td>25.2</td><td>0.37</td><td>0.20</td></tr><tr><td>Gemcitabine 5 nM</td><td>15.1</td><td>18.0</td><td>33.3</td><td>34.5</td><td>0.57</td><td>0.14</td></tr><tr><td>Gemcitabine 10nM</td><td>15.1</td><td>20.5</td><td>37.8</td><td>38.0</td><td>1.00</td><td>0.41</td></tr><tr><td>Paclitaxel 2nM</td><td>26.7</td><td>21.8</td><td>31.4</td><td>24.6</td><td>0.56</td><td>0.54</td></tr><tr><td>Paclitaxel 3nM</td><td>32.0</td><td>22.9</td><td>33.7</td><td>
26.9</td><td>0.90</td><td>1.00</td></tr></table>

![](dt=2026-03-17/ht=18/b81773502d6132becc29fc37862b200a202910ff4e6d6ed4ee368321623b3b92.jpg)

![](dt=2026-03-17/ht=18/429eb7832c5e0df30fde1e1cc909545464f0b33c745a8cfcd5237eab5c164040.jpg)

![](dt=2026-03-17/ht=18/f3d9d6e84ac7f1f532bce2ca6bc023cf026ff2a97b751e50e93eedf51313057c.jpg)

![](dt=2026-03-17/ht=18/5540e0613b8655799eac5ca4eab9f67fcd5c6abeaad3aee89695290e71e3eea1.jpg)

![](dt=2026-03-17/ht=18/7ca75c72a26cceeb3bcaab6ec3d3d4b9da91f7ad4c220db135368019072a5696.jpg)

![](dt=2026-03-17/ht=18/a8d95b158f426f14f6990972df24ad61045f8965b6ec55efe41afd65bd0b1594.jpg)

![](dt=2026-03-17/ht=18/75652a9855c499fcac6789bdc8130989caaf913f6fabadf99486100e5cadc8d7.jpg)

![](dt=2026-03-17/ht=18/0a5d2ed6cea7ae4b19e3f560305303a856dc7be446ac5a402ebf5bb1774df084.jpg)

![](dt=2026-03-17/ht=18/f308159ca18ba8152a1727b53d74e47aa538e9b631b6ad1da4ecea8d5e25bb00.jpg)

![](dt=2026-03-17/ht=18/328b508316ddee27bd96cd469b4afb9323c5440c0a8692ff8cc6992f751a5884.jpg)

![](dt=2026-03-17/ht=18/26bfc68ab08c8b116863e9f89e509fc43c0e4aeb912025df51122e405fb7d5a0.jpg)

![](dt=2026-03-17/ht=18/971fc185338c9a36a767b75f50863dd72c4ebe7745b04e44a7b807bf8a244892.jpg)

inferred that $\mathrm{S - G_2}$ durations were extended following gemcitabine treatment, which we confirmed experimentally: $\mathrm{S - G_2}$ durations were extended from 22.3H to 34.5H with $5\mathrm{nM}$ and to 38H with $10\mathrm{nM}$ gemcitabine (Fig. 3G). Lastly, the model inferred an increase in the number of cell death events relative to the starting cell number, from 0 in control to 0.57 with $5\mathrm{nM}$ gemcitabine.

At $10\mathrm{nM}$ gemcitabine, the model predicted 1.0 relative cell death events such that the number of cell death events across 96H was the same as the initial starting cell number (Fig. 3A). The experimentally observed values showed similar trends, though with more modest changes in cell numbers (0.14 and 0.41 relative cell numbers for 5 and $10\mathrm{nM}$ gemcitabine, respectively) (Fig. 3H).

Overall, we observed similar trends in each of the parameters for gemcitabine treated cells as inferred by the model; modest differences were that the model inferred higher cell death and shorter extensions to $\mathrm{S - G_2}$ than we observed experimentally.

We also tested an assumption of the model that $\mathrm{G}_1$ and $\mathrm{S - G_2}$ phases are independent variables, which captures the idea that these cell cycle phases are independently regulated at the molecular level. We analyzed $\mathrm{G}_1$ versus $\mathrm{S - G_2}$ durations for individual cells in the untreated, $10\mathrm{nM}$ gemcitabine, and $50\mathrm{nM}$ lapatinib conditions, and found a minimal correlation between $\mathrm{G}_1$ and $\mathrm{S - G_2}$ durations (Fig. 3l). These experimental observations confirm the implicit assumption of the model that $\mathrm{G}_1$ and $\mathrm{S - G_2}$ durations are uncorrelated.

Lastly, we evaluated model inferences for paclitaxel treatment. Consistent with our experimental observations, the model inferred minimal changes to $\mathrm{G}_1$ and $\mathrm{S - G_2}$ durations following treatment (Fig. 3A, J, K). At $2\mathrm{nM}$ paclitaxel, the model inferred 0.56 cell deaths relative to the starting cell numbers, and at $3\mathrm{nM}$ inferred 0.90 relative cell deaths (Fig. 3A, Methods). Experimentally, our observations were consistent with the values inferred by the model: we observed 0.54 and 1.00

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

5

![](dt=2026-03-17/ht=18/c4d25a9fe271f0f97d928f6fabc0831d82a1b8877af23970fb180b67b7eb7113.jpg)

![](dt=2026-03-17/ht=18/859f1c75613dc40db2ce870193637ca522f9147f923ad09ddb8e91eb516304d0.jpg)

![](dt=2026-03-17/ht=18/ecf064d820fdddbafdb1d5a75fbf37c2a49000cf10a6f32b9208c966bf1dfad3.jpg)

![](dt=2026-03-17/ht=18/0b0ef001e36284a1a1a8895d236d1c00806c5d393a45588869b7719b76eabb72.jpg)

![](dt=2026-03-17/ht=18/e39090080dbd5f5534b71bb1865967430d4087903cbf1869520e692c8698c957.jpg)

![](dt=2026-03-17/ht=18/56641cc649b6d86b380e4c63385c1de2a8162360b06f7ebeae3caeea60d8cda9.jpg)

![](dt=2026-03-17/ht=18/df621d8214e59b45041158d2c72a394d048c9dd986037e6c50aeeec5d29ba835.jpg)

![](dt=2026-03-17/ht=18/0cfbe51b0f786b1392ddd985594a3a41045c5309be5abb0635e347e454674b73.jpg)

![](dt=2026-03-17/ht=18/f8bb8f82abac003766592ba5cde897072c8da90b6dd676b2f098e518628b82a6.jpg)

![](dt=2026-03-17/ht=18/0ed341943ac46c73a0b2c7873b2d6633d541aacf65b151c6fd3cb94486cbb399.jpg)

relative cell deaths for $2\mathrm{nM}$ and $3\mathrm{nM}$ paclitaxel (Fig. 3L). To summarize the mechanisms that account for the observed changes in cell numbers due to paclitaxel treatment, we compared the number of cell death events against final cell counts for each of the other drugs. These data show the relative bias of paclitaxel toward inducing cell death, especially at $2\mathrm{nM}$ , as compared to $5\mathrm{nM}$ gemcitabine and $50\mathrm{nM}$ lapatinib, which both result in similar final cell numbers (Fig. 3M). Overall, the LCT model captures key observations about the cell cycle effects of each drug, which were confirmed by in-depth single-cell tracking of the experimental data.

# Drug-induced changes to cell cycle behavior generalize across a molecularly diverse panel of breast cancer cell lines

To assess the generalizability of our computational framework and experimental observations, we generated and tested three additional breast cancer cell lines from diverse molecular backgrounds $^{30}$ : 21MT1 (Basal subtype, HER2+), HCC1143 (Basal subtype, HER2-) and MDAMB157 (Claudin-low subtype, HER2-) (Supplementary Figs. 5-7). Because these cell lines do not uniformly overexpress HER2, we additionally tested BEZ235 and trametinib, which respectively target PI3K and MEK, two growth factor pathways downstream from HER2.

We observed dose-dependent reductions in cell numbers and also modulation of the percent of $G_{1}$ cells following drug treatment. Importantly, similar to our findings for AU565 cells, we observed dynamic responses not captured by terminal endpoint readouts of cell viability (Supplementary Figs. 5-7, panels A-B). Unique respons
e patterns observed include: a delayed $G_{1}$ enrichment from trametinib in 21MT1 cells (Supplementary Fig. 6), a lack of $G_{1}$ enrichment from palbociclib and BEZ235 in MDAMB157 cells (Supplementary Fig.

7), and a dose-dependent bifurcation in $G_{1}$ enrichment for doxorubicin in all three of the cell lines (Supplementary Figs. 5-7).

Next, we tested our LCT model on each of the new cell lines. Comparison of model fits to experimental observations confirmed that our model could capture the dynamic responses observed across this panel of molecularly distinct cell lines, indicating the generalizability of our computational framework (Supplementary Figs. 5-7, panels C-E). We analyzed the output of the LCT model, which inferred changes to cell cycle phase durations and cell death probabilities for drug-cell

line pairs at the EC50 concentration (Sup. Fig. 8). The model inferred cell-line-specific changes to both $G_{1}$ and $G_{2}$ phases (Supplementary Fig. 8A, B). For instance, 21MT1 cells were inferred to preferentially undergo $G_{1}$ cell death after doxorubicin and paclitaxel treatments, at probabilities of $60\%$ and $15\%$ , respectively (Supplementary Fig. 8C). The model inferred that HCC1143 cells arrest and die in $S - G_{2}$ following paclitaxel or palbociclib treatment (Supplementary Fig. 8B, D).

MDAMB157 cells were inferred to become growth-arrested by drug treatment and to preferentially die in $G_{1}$ phase (Supplementary Fig. 8C, D). Overall, these findings confirm that our computational framework generalizes across several drugs and cell lines and can infer a range of drug treatment response behaviors.

# Responses to drug combinations are dependent on drug specific cell cycle and cell death effects

Durable and effective cancer treatments frequently require administration of multiple drugs; however, identification of the principles underlying optimal drug combinations have been challenging<sup>31</sup>. Here, we tested the idea that our LCT model, which incorporates cell cycle effects, can be used to predict the impact of different drug combinations on cell cycle behavior and final cell numbers.

We compared two strategies in accounting for drug combination effects: first, we combined drug effects by additive combination of the drugs' rate effects on $G_{1}$ and $S-G_{2}$ progression using Bliss additivity; second, we applied Bliss additivity directly using the cell numbers at the end of each experiment.

To explore these predictions, we varied the dose of one drug in the two drug combination pair and analyzed responses to drug combinations that targeted either the same cell cycle phase ( $G_{1}$ and $G_{1}$ , or $S-G_{2}$ and $S-G_{2}$ ) or different cell cycle phases ( $G_{1}$ and $S-G_{2}$ ).

We tested combining the rates for two $\mathrm{G}_{1}$ targeted drugs, in this case lapatinib and palbociclib. The model predicted that effects on cell number would saturate around the initial starting cell number, indicating cytostatic effects of this drug combination (Fig. 4A). In contrast, drug combination effects based on cell numbers alone predicted a cytotoxic effect at higher drug concentrations, resulting in a reduction in cell numbers relative to the starting cell numbers. We tested these drug combinations experimentally and found a cytostatic

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

6

effect at higher doses, which matches the model prediction based on combining rates of cell cycle progression (Fig. 4A). Bliss additivity fails to account for cytostatic effects, and so deviation in the palbociclib/ lapatinib combination is to be expected $^{32}$ . We also analyzed predictions of gemcitabine combined with doxorubicin, which both extend S-G $_2$ durations and induce cell death. Here, both methods of predicting combination effects qualitatively agreed, and we again saw agreement with experimentally measured combination effects (Fig. 4B). Overall, we were encouraged to observe that our cell cycle model could accurately account for these two distinct combinations.

Finally, we used the LCT model to examine the impact of combining two drugs that target different cell cycle phases, which mimics lapatinib $\left(\mathrm{G}_{1}\right.$ effect) combined with gemcitabine (S phase effect). The cell cycle model predicted an antagonistic effect at higher doses, such that $30\mathrm{nM}$ gemcitabine combined with $100\mathrm{nM}$ lapatinib is expected to yield similar final cell numbers as $30\mathrm{nM}$ gemcitabine on its own (Fig. 4C).

Experimentally, we observed that three of the four lapatinib and gemcitabine combination doses show an antagonistic impact on cell number as compared to gemcitabine alone, indicating that combining these two drugs is counterproductive. These antagonistic effects of the combination held when lapatinib was replaced by palbociclib, which also impacts $\mathrm{G}_{1}$ durations (Fig. 4D). We examined the model predictions in more detail to gain insights into the underlying biological mechanisms driving drug combination responses.

The LCT model predicted that the $\mathrm{G}_{1}$ effect of lapatinib initially dominates over the S-phase effects of gemcitabine, leading to an increased $\mathrm{G}_{1}$ proportion for the population. We confirmed this prediction experimentally, indicating that lapatinib co-treatment can mitigate the S-G2 effects of gemcitabine (Fig. 4E).

In summary, these data indicate that the cell cycle phase and cell death impacts of each drug in a pair are critical for determining the influence of single drugs on cell cycle behavior and that this information can be used for rational identification of drug combinations likely to be therapeutically beneficial.

# Discussion

In this report, we link cell cycle regulatory mechanisms with drug-specific cell cycle effects to gain insights into cancer cell responses to individual drugs and drug combinations. To meld these ideas, we developed a combined experimental and modeling approach to measure cell dynamics and infer cell behavior. This approach revealed that assessment of temporal dynamics and cell behavior is critical to interpret and model drug-induced effects. Importantly, assessment of the impacts of single agents on cell cycle behavior could be used to identify drug combinations likely to yield therapeutic benefits.

Recently, an in-depth analysis revealed that cell cycle phases in individual cells are uncorrelated and have durations that can be accurately modeled as an Erlang distribution, which is a special case of a gamma distribution $^{10}$ . This observation indicates that the cell cycle can be viewed as a series of uncoupled, memoryless phases rather than a single process $^{10,33}$ . In this work, we found similar uncorrelated patterns in cell cycle phase responses after treatment with different anticancer drugs. This revealed multiple implications for assessing and modeling drug responses.

First, viewing the cell cycle as a single process implies that cell behavior is immediately impacted upon drug treatment; however, we and others have reported that drug effects are often not observed until individual cells enter or approach a specific cell cycle phase or checkpoint $^{34,35}$ . For instance, we found that cells were initially distributed across all phases of the cell cycle and that the addition of lapatinib, a $\mathrm{G}_{1}$ -targeting drug, did not initially affect cells in S-G $_2$ phase.

This led to a partial cell cycle synchronization across the population and required the incorporation of a linear chain trick into our model to account for this dwell time. Our studies also revealed a delayed induction of apoptosis after doxorubicin or gemcitabine

treatment, which we attribute to accumulation of DNA damage over the course of the assay. The temporal dynamics of therapeutic responses were also an important consideration for co-treatment with gemcitabine and lapatinib. If
both drugs had immediate effects on cell behavior, we would expect that the $\mathrm{G}_1$ and S-G $_2$ effects of each drug would counteract each other and lead to a constant ratio of cells in $\mathrm{G}_1$ phase. Instead, both experimentally and through model predictions, we found an initial $\mathrm{G}_1$ enrichment.

This likely induced a secondary effect of reducing the relative time that each cell spent in S-phase, which further reduced gemcitabine sensitivity. This finding could explain the antagonistic effects on cell numbers that we and others have observed when combining gemcitabine with lapatinib or palbociclib $^{36,37}$ . We speculate that a synergistic effect on cell numbers could also arise by combining two drugs that target S-G $_2$ phases, where each drug acts to extend the relative duration in which the other is effective.

This general strategy could be used to identify optimal temporal scheduling of other drug combinations that induce different effects on the cell cycle.

A second implication of multiple independently regulated cell cycle processes relates to the concept of effect equivalence in drug combinations. This concept—that two drugs with independent targets can be used to identify drug synergy or drug antagonism—has predominantly focused on the cell number effect of each $\mathrm{drug}^{2-5}$ . Our current work suggests that equivalence in effect may be better applied to rates of cell cycle phase progression and cell death. In our work, we found that lapatinib and palbociclib primarily impacted $G_{1}$ phase with limited effects on cell death.

In contrast, doxorubicin and gemcitabine both extended $\mathrm{S-G}_{2}$ durations and induced cell death. These cell cycle and cell death effects were critical for gaining insights into the effect of drug combinations. For example, two cytostatic drugs, lapatinib and palbociclib, were additive up to doses that reached the maximum cytostatic effect, with further dose increases leading to only minor effects on cell numbers. In contrast, combining the two cytotoxic drugs led to increasingly cytotoxic responses across the full dose range.

These results suggest that considering the cell cycle and cell death impacts of each drug is necessary to make predictions about the effects of their combinations and implies that this information could be used for the rational identification of effective drug combinations[38,39]. Further, the LCT computational model could be used to explore the impact of different drug sequences or schedules, thereby increasing the efficiency of laboratory experiments by identifying drug regimens most likely to yield therapeutic benefit.

Future studies that examine additional drug combinations and schedules across a larger panel of cell lines would further extend our findings and help to prioritize therapeutic strategies that may warrant further preclinical testing in more complex model systems.

Drug response measurements evaluated in the context of a mechanistic cell cycle model can reveal insights about the nature of drug response and resistance not immediately apparent from purely data-driven analyses. For instance, a model for the proliferation dynamics of cancer cells can separate the contribution of dividing, non-dividing, and dying cells[22], revealing that the rates of cell death and entry into quiescence change with drug treatment.

Previous computational models of cell cycle behavior have explored various ways in which cell cycle behavior might impact drug response but have struggled to identify experimental data amenable for model fitting and evaluation. For instance, others have appreciated that drugs do not affect the cell cycle uniformly and have therefore proposed computational models that partition the cell cycle into several independent steps, both with[10] and without[33] cell death effects.

Modeling cell lifetimes as being hypo-exponentially distributed helps to explain the distribution of cell lifetimes within a population but does not connect these observations to known cell cycle stages[40]. In this report, we demonstrate that partitioning known cell cycle phases to account for

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

7

their dwell time effects—and including experimentally observed drug effects like cell death—results in a modeling framework that can faithfully and mechanistically capture experimentally observed anticancer drug effects.

We applied our experimental approach and computational framework to examine dynamic drug-induced responses in a molecularly diverse set of breast cancer cell lines. In all cases, we observed that therapeutic inhibition induces a wide array of responses, indicating that the influence of therapies on cell cycle dynamics is a generalizable mechanism operable in a wide array of molecular backgrounds.

Cancer cells treated with therapies may adopt new molecular programs associated with adaptive and acquired resistance, and indeed previous studies have demonstrated this principle in both model systems and patient samples $^{41}$ . We hypothesize that cells with acquired resistance may show distinct drug-induced cell cycle programs as compared to naïve cells and that our approach could be used to uncover the molecular mechanisms associated with adaptive resistance.

The approach outlined here is built around the concept that therapies perturb cell cycle behavior and are agnostic of the exact type of cellular perturbation. Our study therefore provides a blueprint for studying responses of diverse cell types—both normal and diseased—to a wide array of perturbations, including diverse panels of therapeutic inhibitors, growth factors, or genetic manipulation with CRISPR $_i$ a.

The resultant data could be used to adapt our computational framework to identify mechanisms of cell cycle control in different cellular contexts, microenvironmental conditions, or disease states.

While our model could explain many of the key observations in our experimental data, extensions of the model could further improve its generalizability and robustness. We partitioned the cell cycle into two observed phases, $\mathrm{G}_1$ and $\mathrm{S - G_2}$ , which were further subdivided to explain the dwell time behavior of each phase. With improved reporter strategies[42], we may be able to further subdivide these phases into constituent parts, which could help to localize the effect of a drug to a more specific portion of one cell cycle phase.

Generalizations of the linear chain trick could be used to account for both subphases of varying passage rates, as well as heterogeneity in the rates of passage, such as would arise through cell-to-cell heterogeneity[29]. While the subdivisions within each cell cycle phase are phenomenological, it is tempting to imagine that they represent mechanistic steps within each phase. Identifying how effects connect to actual biological events in the cell cycle would help identify opportunities for drug combinations.

A practical challenge when using the model for drug combinations has been normalization between experiments. While cell number measurements are routinely normalized by dividing by a control, experiment-to-experiment variation in inferred rates requires additional consideration. A wider panel of experiments, across multiple cell lines, may help to tease apart variations associated with drugs, cell lines, or experiments. A final potential extension is considering the existence of phenotypically diverse subpopulations[43].

At the cost of additional complexity, one could employ several instances of the current model with transition probabilities between these states when the cells divide to simulate a heterogeneous population of cells.

We observed that five commonly used cancer drugs each modulated cell numbers through distinct routes and with different temporal dynamics. By revealing how these drugs uniquely impacted cell fate, our model and
analyses have implications for how different cancer drugs can be combined to maximize therapeutic impact. For instance, our results can identify drug combinations that modulate cell cycle effects in orthogonal ways or drug schedules that take advantage of the shift in cell cycle state of the overall population. In summary, these studies provide a map for understanding how cancer cells respond to treatment and how drugs may be combined and timed for maximal effect.

# Methods

# Creation of stable cell lines

AU565 (ATCC CRL 2351) and MDAMB157 (ATCC HTB 24) cells were grown in DMEM supplemented with $10\%$ FBS, HCC1143 (ATCC CRL 2321) cells were grown in RPMI supplemented with $10\%$ FBS, and 21MT1 (generous gift from Kornelia Polyak) cells were grown in DMEM/F12 supplemented with $5\%$ horse serum, $20\mathrm{ng / ml}$ rhEGF, $0.5\mu \mathrm{g / ml}$ hydrocortisone, $100\mathrm{ng / ml}$ cholera toxin, and $10\mu \mathrm{g / ml}$ insulin. Cells were genetically engineered using plasmids available from Addgene.

The coding fragment for clover-HDHB (Addgene plasmid 136461)[23] was cloned in frame into a transposase expression plasmid modified to also express a nuclear localized mCherry and stable cell lines were created as previously described[44]. In brief, the clover-HDHB-NLS-mCherry expression plasmid was co-transfected $24 - 48\mathrm{h}$ with the pSB100X transposase plasmid (Addgene plasmid 34879) at a ratio of 4:1 using Lipofectamine 3000 (AU565, HCC1143, 21MT1) or LTX (MDAMB157) and selected for 3-7 days with $0.5 - 2\mu \mathrm{g / ml}$ puromycin.

To ensure uniform fluorescence across the transfected population, HCC1143 and 21MT1 cells were sorted at OHSU's Flow Cytometry Core and cells with a medium intensity clover-HDHB signal and a high intensity NLS-mCherry signal were selected for drug dose response experiments. In all cases, cells were validated by STR profiling (LabCorp) and tested negative for mycoplasma.

# Drug dose response protocol

In initial experiments, AU565 cells were plated at a density of 25,000 cells per well into 24-well Falcon plates (Corning #353047). 24H after plating the media was exchanged with Fluorobrite media supplemented with $10\%$ FBS, glutamine, and penicillin-streptomycin. Cells were then treated with dose-escalation: lapatinib (Selleckchem #S1028), gemcitabine (#S1149), paclitaxel (#S1150), doxorubicin (#S1208), palbociclib (#S1116), BEZ235 (#S1009), or trametinib (#S2673).

Starting immediately after drug addition (T0), plates were imaged every $30\mathrm{min}$ for $96\mathrm{H}$ using phase, GFP, and RFP imaging channels with an IncuCyte S3. For single drug treatments of AU565 cells only, at $48\mathrm{H}$ the media was replaced in all wells including the control wells, and fresh media and drug were added. Four equally-spaced image locations per well and three biological replicates were collected.

In subsequent experiments, we tested additional cell lines. MDAMB157, HCC1143, and 21MT1 cell lines were transferred to and maintained in a base of either Fluorobrite media and $1 \times$ GlutaMAX or mixed Fluorobrite/F12 media and $0.5 \times$ GlutaMAX along with their corresponding supplements for no less than one week before performing the drug dose response protocol. MDAMB157 and HCC1143 cells were plated at a density of 25,000 cells per well, while the larger 21MT1 cells were plated at a density of 5000 cells per well into 24-well Falcon plates (Corning #353047).

24H after plating the media was exchanged with fresh Fluorobrite media as indicated per cell line. Cells were then treated with dose-escalation: BEZ235, gemcitabine, paclitaxel, doxorubicin, palbociclib, or trametinib. After drug addition (TO), plates were imaged every 2 h for 96H using phase, GFP, and RFP imaging channels with the IncuCyte S3. Four equally-spaced image locations per well and three biological replicates were collected. In pilot experiments, we confirmed that drugs were stable of the duration of the assay, so excluded the 48H media refresh in this set of experiments.

# Image analysis

To analyze AUS65 image data, phase, GFP, and RFP images were overlaid and collated into single files using FIJI $^{45}$ (version 2.3.0), then segmented into three classes (nuclei, background, debris) using a manually trained classifier in Ilastik $^{46}$ (version 1.3.3b2). The segmented nuclear masks from Ilastik and the IncuCyte GFP images were used to count the number of nuclei in each image with Cell

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

8

$\text{Profiler}^{47}$ . Additionally, using the same images (nuclear masks from Ilastik and GFP cell cycle reporter images) cell cycle phase was determined by taking the mean fluorescence in the nucleus compared to the mean fluorescence in a 5-pixel ring surrounding the nucleus, excluding background pixels. A threshold was then manually set for the ratio of nuclear fluorescence to cytoplasmic fluorescence and cells with values above the threshold were defined as being in G1 and cells with values below the threshold were defined as being in S-G2 phase $^{47}$ .

To manually track AU565 cells and identify drug-induced changes operable in single cells, GFP image sequences were registered using the FIJI plug-in StackReg (version 2.0). Individual cells present in the first image and their progeny were followed to identify the time of $G_{1}$ transition, cell death, and cell division using the plug-in mTrackJ $^{48}$ (version 1.5.1). We excluded cells that were binucleated, had abnormally large nuclei, or were near the image border where complete lineages could not be tracked. The $G_{1}$ transition was defined as the last frame before the nuclear intensity of the cell cycle reporter was above the level of the cytoplasm. Assessment of cell death enabled disentangling of cytostatic and cytotoxic drug effects.

We used the following approach for automated analysis of HCC1143, 21MT1 and MDAMB157 cell lines. Image registration was performed on the red channel nuclear marker image stack using the python skimage phase CROSS_CORREATION function to correct translations. Image stacks were cropped to their common areas and individual cells were segmented with the Cellpose LC2 model trained on phase and nuclear images from the untreated and highest drug concentration treatments[49]. Nuclei were segmented with the Cellpose cyto2 model on the nuclear channel.

To associate nuclei across the image stack, to identify progeny after mitosis, and to identify cell death events we used Loeffler tracking[50] with the default parameters of delta_t = 3 and roi_size = 2. We created cytoplasm masks by subtracting the nuclear masks from the cell masks and applied these masks to the green channel cell cycle reporter images using the python skimage function regionprops_table. To assign cells to G1 or S-G2 states, we computed the ratios between the cytoplasm and nuclear cell cycle reporter.

k-means clustering of the ratios observed in cells in the untreated condition was used to establish a per-plate threshold between cell cycle states.

The quantitated cell-level data was mean summarized to the population level for each image and to assess cell counts and G1 cell cycle state proportion. The cell counts were normalized to the mean of the counts of the first three images. The cell count dose response curves were normalized to the control by dividing each drug cell count by the control cell count at the same time slice.

# Core model

To identify the dynamics of the AU565 cancer cell population in response to compounds, we built a system of ordinary differential equations (ODEs) with two states: $\mathrm{G}_1$ , and $\mathrm{S - G_2}$ . Cells transition from $\mathrm{G}_1$ to $\mathrm{S - G_2}$ phase, and then vice versa when doubling. Cell death can occur in either phase with phase-specific death rates. S and $\mathrm{G}_2$ phases are combined as our reporter cannot distinguish them. From s
ingle-cell tracking, we identified that $\mathrm{G}_1$ and $\mathrm{S - G_2}$ phase time-intervals are gamma-distributed.

Based on this observation, we employed the linear chain trick (LCT)[28] to capture gamma-distributed cell cycle phase durations. We broke down each phase into a series of sequential subphases and derived the system of mean-field ordinary differential equations. Each sub-phase is represented as a single state variable within the differential equation system. The total number of cells in each phase is the sum of the cell numbers in each sub-phase.

Furthermore, to account for the non-uniform effect of the drugs over each cell cycle phase, we divided $\mathrm{G}_1$ and $\mathrm{S - G_2}$ into 4 parts each, such that the effect of a drug can be distinguishable at the beginning, middle, or the end of the phases.

The mean-field system of ODEs is:

$$
\frac {d G _ {1 1 , 1}}{d t} = + 2 \beta_ {4} G _ {2 4, 5} - (\alpha_ {1} + \gamma_ {1, 1}) G _ {1 1, 1} \tag {1}
$$

$$
\frac {d G _ {1 k , 1}}{d t} = + \alpha_ {k - 1} G _ {1 k - 1, 2} - (\alpha_ {k} + \gamma_ {1, k}) G _ {1 k, 1} \tag {2}
$$

$$
\frac {d G _ {1 k , 2}}{d t} = + \alpha_ {k} G _ {1 k, 1} - \left(\alpha_ {k} + \gamma_ {1, k}\right) G _ {1 k, 2} 1 \leq k \leq 4 \tag {3}
$$

$$
\frac {d G _ {2 1 , 1}}{d t} = + \alpha_ {4} G _ {1 4, 2} - (\beta_ {1} + \gamma_ {2, 1}) G _ {2 1, 1} \tag {4}
$$

$$
\frac {d G _ {2 i , j}}{d t} = + \beta_ {i} G _ {2 i, j - 1} - \left(\beta_ {i} + \gamma_ {2, i}\right) G _ {2 i, j} 2 \leq j \leq 5, 1 \leq i \leq 4 \tag {5}
$$

The parameters of the model include progression rates through $\mathrm{G}_1$ phase, $\alpha$ , and $\mathrm{S - G_2}$ phase, $\beta$ , and death rates in each of the $\mathrm{G}_1$ phase, $\gamma_{1}$ , and $\mathrm{S - G_2}$ phase, $\gamma_{2}$ . Cells at the end of the $\mathrm{S - G_2}$ phase divide and give birth to two cells at $\mathrm{G}_1$ phase. Because each phase is divided into 4 parts, each part of $\mathrm{G}_1$ contains 2 sub-phases, and each part of $\mathrm{S - G_2}$ contains 5 sub-phases.

The model was implemented in Julia v1.5.3. The differential equations were solved by the matrix exponential. As the data was measured with equal spacing, we pre-calculated the transition matrix between timesteps.

# Dose response relationship

We assumed that the progression and death rates in $\mathrm{G}_1$ and $\mathrm{S - G_2}$ that form the quantified drug effects on the population follow a Hill function:

$$
H i l l (C) = E _ {\min } + \frac {E _ {\max } - E _ {\min }}{1 + \left(\frac {E C 5 0}{C}\right) ^ {k}} \tag {6}
$$

where the $EC_{50}$ indicates the half-maximal drug effect concentration, $E_{\min}$ the value of the rate parameter in the absence of drug, $E_{\max}$ the rate parameter at infinite concentration, and $k$ the steepness of the dose-response curve. Given these parameters and the drug concentration $(C)$ we then calculated the specific rate parameters for that treatment.

# Exponential model

To show the benefit of our LCT model, we employed a commonly-used exponential model to fit to the $G_{1}$ and $S - G_{2}$ cell numbers and showed that the exponential model cannot capture the dynamics of the data. The parameters were the same as the mean-field model.

$$
\frac {d G _ {1}}{d t} = + 2 \beta G _ {2} - (\alpha + \gamma_ {1}) G _ {1} \tag {7}
$$

$$
\frac {d G _ {2}}{d t} = + \alpha G _ {1} - (\beta + \gamma_ {1}) G _ {2} \tag {8}
$$

# Model fitting

The data included the percentage of cells in $G_{1}$ phase and the total number of cells normalized to the cell numbers at the initial time point. We assumed 1 starting cell at $0\mathrm{H}$ and calculated the number of cells in $G_{1}$ and $S - G_{2}$ phase over time. The Savitzky-Golay filter was used to smooth the data. Three replicates of each experiment were averaged, and the average was used for the purpose of fitting.

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

9

The number of $\mathrm{G}_1$ and $\mathrm{S - G_2}$ subphases, and the parameters in the absence of drug were shared across all drugs and concentrations. The sum of squared error was used as the cost function value and was calculated between the cell numbers predicted by the model and the average cell numbers of three replicates, over all time points, concentrations, and drugs tested. This cost function was then minimized using the default adaptive differential evolution optimizer from the BlackBoxOptim.jl Julia package, version 0.5.0.

To characterize the identifiability of our fit parameters we conducted a local sensitivity analysis. To do so, we calculated the cost function while varying each parameter from 0.1 to 10 times the optimal value, holding all the other parameters at their optimum. All parameters were identifiably constrained according to this analysis.

# Calculating relative number of cell deaths and average phase durations

We evaluated the number of dead cells at 96H relative to the starting cell number at OH. This formed the observed relative cell death numbers reported in Fig. 3A. To calculate the corresponding cell death values inferred from the model, we calculated the predicted number of cells at each phase part $(\mathrm{G}_{11}, \mathrm{G}_{12}, \mathrm{G}_{13}, \mathrm{G}_{14}, \mathrm{G}_{21}, \mathrm{G}_{22}, \mathrm{G}_{23}, \mathrm{G}_{24})$ separately, and multiplied them by their individual death rates at all time points. This provides the number of dead cells at each phase part at each time point.

The sum of cell numbers died in each phase part provides the total cell death counts at each time point, $n(t)$ . Figure 2C, D shows the accumulated dead cells across time for lapatinib and gemcitabine treatments which was calculated by summing over the cell death counts, $n(t)$ , across time from 0 to each timepoint, $T$ . Calculating for 96H results in the total cell death normalized by the initial cell numbers, 1, this value refers to the relative predicted cell death number reported in Fig. 3A.

$$
n (t) = \sum_ {i = 1} ^ {2} \sum_ {j = 1} ^ {4} G _ {i j} (t) \times \gamma_ {i j} \tag {9}
$$

$$
N (T) = \sum_ {t = 0} ^ {T} n (t) \tag {10}
$$

The average phase durations $(\bar{G}_1, \overline{S - G_2})$ from the model were calculated using the progression rates. The $G_1$ phase has 8 subphases which is divided into 4 parts that results in 2 phases per part. S-G2 phase has 20 subphases divided into 4 parts that results in 5 subphases in each part. Each phase part has a unique parameter for cell death and phase progression rate. The average phase duration will be given by the following expressions, derived by recognizing that the time in each part is gamma-distributed with a shape parameter equal to the number of subphases.

$$
\bar {G} _ {1} = \sum_ {j = 1} ^ {4} \frac {2}{\alpha_ {1 j}} \tag {11}
$$

$$
\overline {{S - G _ {2}}} = \sum_ {j = 1} ^ {4} \frac {5}{\beta_ {2 j}} \tag {12}
$$

# Predicting drug combinations

Bliss independence was used to calculate the predicted effect of drug combinations. Assuming $E_{a}$ and $E_{b}$ to be the saturable, quantified effects of drugs $a$ and $b$ , the expected combined effect would be:

$$
E _ {a b} = E _ {a} + E _ {b} - E _ {a} \cdot E _ {b} \tag {13}
$$

For death effects, we added the effects of each drug to find the death effect of the drug combination:

$$
D _ {a b} = D _ {a} + D _ {b} \tag {14}
$$

The Bliss relationship requires that data first be scaled to be between 0 and 1, and then scaled back after the interaction calculation:

$$
\hat {X} = \frac {\left(X _ {\text {c o n t r o l}} - X\right)}{X _ {\text {c o n t r o l}}} \tag {15}
$$

This measure is usually used as a baseline to decide whether the combination of two drugs is synergistic or antagonistic. Here we used Bliss in two ways: (1) on the progression rate parameters to simulate the model predictions of drug combinations; and (2) on cell numbers to serve as a baseline approach to calculate dru
g combination effects, as is commonly used.

In the first case, we use Bliss additivity on the cell cycle progression rates $(^{*})$ to find the set of progression parameters representing the combined treatment and assume that the death effects are only additive because the cell death process is not saturable $(^{**})$ . The combination parameters for all the eight concentrations for all pairs of drugs were calculated and then converted back to their original units. Next, we simulated the cell numbers using these parameters.

In the baseline case, we used the cell numbers in the control condition to normalize the cell number measurements and then converted the cell numbers back to their original scale. This was used as a benchmark reference.

# Data availability

All data and code as well as additional relevant resources are available under Synapse ID syn51239112. Specifically, the AU565 single drug treatment data corresponding to Figs. 1, 2, 3, Supplementary Figs. S1, S2, S3, and S4 are available under Synapse ID syn51239569. Cell cycle durations in Supplementary Fig. S3 are available under Synapse ID syn51359628. AU565 drug combination results corresponding to Fig. 4 are available under Synapse ID syn51239601. HCC1143 single drug treatment data corresponding to Supplementary Fig. S5 are available under Synapse ID syn51239126.

21MT1 single drug treatment data corresponding to Supplementary Fig. S6 are available under Synapse ID syn51239113. MDAMB157 single drug treatment data corresponding to Supplementary Fig. S7 are available under Synapse ID syn51239135. The live-cell image data generated in this study have been deposited in the Image Data Resource database under accession code idr0119. Source data are provided with this paper.

# Code availability

The code and analysis<sup>51</sup> is available at https://github.com/meyer-lab/DrugResponseModel.jl. Code for the automated segmentation and tracking<sup>52</sup> is on Zenodo at https://zenodo.org/badge/latestdoi/612325635.

# References

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

10

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

11

# Acknowledgements

These studies were supported by NIH research grants U54-CA209988 (L.M.H.), U54-HG008100 (L.M.H.), U01-CA215709 (A.S.M.), Jayne Koskinas Ted Giovanis Foundation for Health and Policy (L.M.H. and A.S.M.). Cell sorting was performed by the OHSU Flow Cytometry Core, which is supported in part by the OHSU Knight Cancer Center Support Grant (P30CA069533) and an S10 instrument award (1S10OD028512). The Jayne Koskinas Ted Giovanis Foundation for Health and Policy is a private foundation committed to critical funding of cancer research. The opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and not necessarily those of the Jayne Koskinas Ted Giovanis Foundation for Health and Policy or its respective directors, officers, or staff.

# Author contributions

S.M.G. performed experiments and analyzed data. F.M. constructed the computational model and performed simulation studies. C.S-A. and T.A.L. performed experiments. C.S-A. and P.J.Z. analyzed live-cell imaging data. M.A.D. developed automated pipelines to analyze live-cell imaging data. A.S.M. supervised the computational modeling. L.M.H. supervised the experimental data generation and image analysis. S.M.G., L.M.H. and A.S.M. conceived the study. S.M.G., F.M., A.S.M. and L.M.H. wrote the paper. All authors edited the paper.

# Competing interests

The authors declare no competing interests.

# Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41467-023-39122-z.

Correspondence and requests for materials should be addressed to Laura M. Heiser.

Peer review information Nature Communications thanks Jeremie Roux, Anna Sorace and the other, anonymous, reviewer(s) for their contribution to the peer review of this work. This article has been peer reviewed as part of Springer Nature's Guided Open Access initiative.

Reprints and permissions information is available at http://www.nature.com/reprints

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material.

If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2023

Article

https://doi.org/10.1038/s41467-023-39122-z

Nature Communications| (2023)14:3450

12