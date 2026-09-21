# Dynamic heterogeneity as a strategy of stem cell self-renewal

Philip Greulich $^{a,b,1}$ and Benjamin D. Simons $^{a,b,c,1}$

$^{a}$ Cavendish Laboratory, University of Cambridge, Cambridge CB3 0HE, United Kingdom; $^{b}$ Wellcome Trust/Cancer Research UK Gurdon Institute, Cambridge CB2 1QN, United Kingdom; and $^{c}$ Wellcome Trust/Medical Research Council Stem Cell Institute, Cambridge CB2 1QR, United Kingdom

Edited by Roeland Nusse, Stanford University School of Medicine, Stanford, CA, and approved May 9, 2016 (received for review February 23, 2016)

To maintain cycling adult tissue in homeostasis the balance between proliferation and differentiation of stem cells needs to be precisely regulated. To investigate how stem cells achieve perfect self-renewal, emphasis has been placed on models in which stem cells progress sequentially through a one-way proliferative hierarchy. However, investigations of tissue regeneration have revealed a surprising degree of flexibility, with cells normally committed to differentiation able to recover stem cell competence following injury.

Here, we investigate whether the reversible transfer of cells between states poised for proliferation or differentiation may provide a viable mechanism for a heterogeneous stem cell population to maintain homeostasis even under normal physiological conditions. By addressing the clonal dynamics, we show that such models of "dynamic heterogeneity" may be equally capable of describing the results of recent lineage tracing assays involving epithelial tissues. Moreover, together with competition for limited niche access, such models may provide a mechanism to render tissue homeostasis robust.

In particular, in 2D epithelial layers, we show that the mechanism of dynamic heterogeneity avoids some pathological dependencies that undermine models based on a hierarchical stem/progenitor organization.

stem cell fate | stem cell heterogeneity | tissue homeostasis | clonal dynamics

Stem cells are defined by their capacity to self-renew long-term while giving rise to more differentiated cell types. To achieve homeostasis, the choice between proliferation and differentiation—stem cell fate—needs to be perfectly balanced (1-4). In recent years, experimental techniques for tracing cell lineages in vivo have enabled quantitative information to be gathered on the proliferative potential of stem cells. In particular, inducible genetic labeling methods have allowed the fate behavior of individual cells and their progeny—clones—to be traced over time (5-7).

Using these approaches, quantitative analyses have shown that the cell lineage data in many mammalian tissues conform to a pattern by which stem cell fate is chosen stochastically in a balanced way, such that the average number of stem cells in a tissue is conserved (8-11).

In defining the quantitative fate behavior of stem cells and their differentiating progeny, most studies begin with a model in which stem and progenitor cells are organized in a one-way proliferative hierarchy. In this paradigm, the loss of stem cell competence and entry into a differentiation pathway is an irreversible process. However, in recent years it has been shown that, under conditions of stress or injury, cells normally committed to differentiation may reacquire stem cell competence and contribute to the long-term maintenance of tissue (12-18).

Moreover, intravital imaging studies of mouse testis and intestine suggest that, even under normal physiological conditions, adult stem cells are not homogeneous but may transfer reversibly between states primed for proliferation and differentiation (19-21). Finally, evidence for stem cell flexibility has also been observed in the context of cell competition, induced through the activation of oncogenic mutation, with potential implications for the initiation of cancer (11, 22, 23).

Taken together, these observations suggest that transition through a differentiation pathway may not involve the sequential one-way progression through a hierarchy of functionally equivalent states but

may instead be dynamic and heterogeneous, allowing cells to move reversibly between states primed for duplication or differentiation (24-26). The genetic basis of such behavior, which we term "dynamic heterogeneity," has been considered in a variety of contexts (21, 27, 28).

In this study, we address theoretically the question of whether tissue maintenance can be sustained by a process of dynamic stem cell heterogeneity, whether it can be discriminated from hierarchical self-renewal strategies by cell lineage tracing assays, and whether such a mechanism offers advantages in promoting robustness. Specifically, we consider a paradigm in which a (stem) cell may switch stochastically between internal "states" that may differ in their potential to enter into a differentiation pathway or to proliferate. As well as investigating the clonal fate behavior of the model system we also explore how the spatial distribution of cell types is affected by dynamic heterogeneity.

# Results

Dynamic Heterogeneity as a Model of Tissue Maintenance. As a starting point, we consider a model of a cycling adult tissue that comprises a heterogeneous population of self-renewing progenitor cells that give rise to more differentiated progeny. However, the fate potential of individual progenitors is not invariant but conforms to a process of dynamic heterogeneity in which progenitors transit reversibly between states primed for proliferation or biased toward differentiation and loss.

Whether these states represent defined cell types marked by signature expression of molecular markers, or whether they are simply primed by location within a niche environment, we do not distinguish. Both scenarios will lead to the same long-term clonal dynamics, the focus of the present study. For simplicity, and to illustrate the paradigmatic features of the model, we consider only two progenitor cell states, termed $A$ and $B$ type. State $B$ is primed but not yet committed to terminal differentiation and loss, and is nonproliferative, whereas

# Significance

In many tissues, such as intestine and skin, cells are constantly turned over throughout life. To replenish cells that are lost, new cells are generated by stem cells, which divide and differentiate to maintain tissue in a steady state. The mechanisms that allow stem cells to achieve perfect self-renewal promise fundamental insights into processes leading to diseased states. Efforts to define strategies of stem cell self-renewal have placed emphasis on models in which stem cells progress one way through a differentiation hierarchy. Here, we show that a different paradigm, in which stem cells transfer reversibly between states primed for renewal or poised for differentiation, offers a viable and robust mechanism of tissue self-renewal.

Author contributions: P.G. and B.D.S. designed research; P.G. performed research; P.G. contributed new reagents/analytic tools; and P.G. and B.D.S. wrote the paper.

The authors declare no conflict of interest.

This article is a PNAS Direct Submission.

<sup>1</sup>To whom correspondence may be addressed. Email: bds10@cam.ac.uk or pg409@cam.ac.uk.

This article contains supporting information online at www.pnas.org/lookup/suppl/doi:10. 1073/pnas.1602779113/-/DCSupplemental.

CrossMark $\Lleftarrow$ click for updates

PNAS

www.pnas.org/cgi/doi/10.1073/pnas.1602779113

PNAS Early Edition

1 of 6

BIOPHYSICS AND COMPUTATIONAL BIOLOGY

an $A$ -type cell remains in cycle. However, this bias is temporary, and cells can switch reversibly and stochastically between the two states; an $A$ -type cell can transit into a $B$ -type cell, and a primed $B$ -type cell can also return to the proliferative state $A$ . From a biological perspective, the $B$ -type cell may represent some transition state, poised in $\mathrm{G}_0$ , and deciding stochastically between reentry into cell cycle or commitment to terminal d
ifferentiation. Furthermore, for convenience, we assume that cell division leads to asymmetric fate outcome, $A \rightarrow A + B$ , noting that other "channels" of division—cell duplication or terminal division—can be captured through a combination of division and "reversion" $B \leftrightarrow A$ .

Within this framework, the model dynamics is specified by the "zero-dimensional" nonequilibrium process:

$$
A \xrightarrow {\lambda} A + B, \quad A \xrightarrow {\omega_ {B}} B, \quad B \xrightarrow {\gamma} \emptyset \tag {1}
$$

where $\lambda$ denotes the cell division rate and $\omega_{B,A}$ represent the transition rates from $A\rightarrow B$ , respectively $B\to A$ . The last process denotes the rate, $\gamma$ , at which $B$ -type cells commit to terminal differentiation and loss. In the following we refer to this dynamics as the dynamic heterogeneity (DH) model. Finally, for simplicity, we suppose that all processes are stochastic and Markovian, with the defined average rates. Although periodicity in the timing of cell division and differentiation would affect the short-term dynamics, the long-term behavior would be unaffected.

In the following, we will compare the kinetics of the DH model with that of a more orthodox hierarchical scheme (termed the H model) in which progenitor cell fate is assigned irreversibly following cell division, with only $A$ -type cells retaining stem-like renewal potential (cf. refs. 8 and 11):

$$
A \xrightarrow {\lambda} \left\{ \begin{array}{l l} A + A & \Pr . r + \Delta / 2 \\ A + B & \Pr . 1 - 2 r, \\ B + B & \Pr . r - \Delta / 2 \end{array} , \quad B \xrightarrow {\gamma} \emptyset . \right. \tag {2}
$$

Here $0 \leq r \leq 1/2$ determines the relative frequency of symmetric $(A \to A + A, B + B)$ vs. asymmetric cell division $(A \to A + B)$ , and $\Delta$ parameterizes the potential bias in cell fate toward cell proliferation or differentiation. The parameters $r$ and $\Delta$ may be constant and set intrinsically, or be moderated by extrinsic cues from their niche environment (9, 10, 29). Moreover, one can conceive of further adaptations of the model in which the $B$ -type cell population has a limited renewal potential (viz.

transit amplification) or represents just one cell type in a longer hierarchy (30). Crucially, however, the H model, as defined by Eq. 2, is paradigmatic of all models that involve a one-way proliferative hierarchy in which the differentiating progeny of $A$ -type cells are irreversibly committed to differentiation.

In recent years, lineage tracing studies of stratified epithelial tissues (including interfollicular epidermis, esophagus, and trachea), have found that the dynamics of epithelial cell populations are consistent with models based on the hierarchical scheme (8, 9, 11). In the following, we will investigate whether a model of dynamic heterogeneity can also provide a basis for long-term tissue maintenance, and whether its dynamics can be discriminated from that of a hierarchical model through clone size statistics alone.

Robustness of Homeostasis. If the rates of cell division and cell fate ratios of the H and DH model are fixed, for a given arbitrary choice of parameters, the average size of the cell population is not stationary and the system is not homeostatic. Instead, the average number of $A$ - and $B$ -type cells would expand or contract over time. To achieve homeostasis in the H model, the net rate of progenitor cell duplication must perfectly balance differentiation and loss, requiring that the degree of imbalance, $\Delta$ , must be tuned to zero. In ref. 31 it was suggested that the dynamics of the H model can be rendered stable by imposing a feedback mechanism in which $\Delta$ depends on the total size of the $A$ -cell pool. Indeed, it is known

that the cell division rate can be correlated with local cell density (32) (contact inhibition) as well as the cell loss rate (33, 34), a phenomenon that we call crowding feedback (for a discussion of potential feedback mechanisms, see refs. 35-39). For the H model, such crowding feedback in the cell division rate or loss rate is not sufficient to confer stability. Without spatial regulation, only by controlling the cell fate bias $\Delta$ may stability be imposed (Supporting Information).

In the DH model, the time evolution of the average densities of cell type $A$ , $n_A$ , and of cell type $B$ , $n_B$ —where $n_{B,A}$ are cell numbers normalized by volume—is given by (Supporting Information)

$$
\begin{array}{l} \partial_ {t} n _ {A} = \omega_ {A} n _ {B} - \omega_ {B} n _ {A} \\ \partial_ {t} n _ {P} = (\lambda + \omega_ {P}) n _ {A} = (\gamma + \omega_ {A}) n _ {P} \end{array} \tag {3}
$$

Therefore, to achieve homeostasis, viz. $\partial_t n_{A,B} = 0$ , the corresponding rates must also be fine-tuned such that

$$
\frac {\gamma}{\lambda} = \frac {\omega_ {A}}{\omega_ {B}}. \tag {4}
$$

Under these conditions, with constant rates (i.e., no feedback), the average density of $A$ - and $B$ -type cells remains constant with $n_B = (\lambda/\gamma)n_A$ . When seeded away from these values, $n_A$ and $n_B$ will converge back to their steady-state values, as depicted in the flow diagram (Fig. 1A). If, however, the balance condition is not met, the populations of $A$ - and $B$ -type cells either decay to zero $(n_{A,B} = 0)$ or grow indefinitely (cf. Fig. 1B).

Even when the systems are fine-tuned to conditions of homeostasis, without further regulation, both models are unstable toward fluctuations. In particular, in a closed system in which the size of the cell population is only finite, statistical fluctuations due to stochastic dynamics will inevitably lead to the chance extinction of the population.

The mechanism of crowding feedback can be incorporated in the models by imposing a dependence of the parameters on the average total density of cells $n = n_{A} + n_{B}$ . To illustrate this, we consider the case where the cell division rate depends on $n$ , viz. $\lambda = \lambda (n)$ , and decreases monotonically with $n$ , $\partial_n\lambda < 0$ , as would be the case for contact inhibition (32). With this implementation, the DH model, Eq. 3, acquire a single stable fixed point with

$$
n ^ {*} = \lambda^ {- 1} \left(\frac {\omega_ {B} \gamma}{\omega_ {A}}\right), \quad n _ {A} ^ {*} = \rho n ^ {*}, \tag {5}
$$

where $\lambda^{-1}(\dots)$ denotes the inverse of the function $\lambda(n)$ and $\rho \coloneqq \omega_A / (\omega_A + \omega_B)$ is an effective parameter, equivalent to the steady-state fraction of $A$ -type cells (Supporting Information). In particular, for a linear response $\lambda(n) = \lambda_0 - k n$ , where $\lambda_0$ is the unconstrained cell division rate without crowding and $k = -\partial_n \lambda > 0$ parameterizes the strength of the feedback, the stationary point is

![](dt=2026-03-07/ht=00/782c15e2c8de3e912d8a98f42199536121ae23cc7776bf87168eb9adb28680b5.jpg)

![](dt=2026-03-07/ht=00/0ab6ac4d2e6b017b44d1199bc0c5cd87869219e1b854ebbcd42ca731ed06fd4a.jpg)

![](dt=2026-03-07/ht=00/33bc22762e9b063d8a010fa854c4d2eb8c17976b5bc536fc0cf7695d0d5e1cb5.jpg)

2 of 6

www.pnas.org/cgi/doi/10.1073/pnas.1602779113

Greulich and Simons

$n^{*} = (\lambda_{0} - \omega_{B}\gamma /\omega_{A}) / k$ . Inspection of the flow diagram (Fig. 1C) shows that in this case the stationary point is indeed globally stable. In Supporting Information we show that, in fact, the system achieves a stable homeostatic state for any monotonically decreasing function, $\lambda (n)$ , that achieves the point $\lambda = \omega_B\gamma /\omega_A$ at some given value of $n$ . Since there is only one stable point, the system will eventually attain this homeostatic steady state.

Importantly, ref
erring to Supporting Information for details, a stable homeostatic state is also attained when any of the other parameters, $\gamma$ , $\omega_{A}$ , or $\omega_{B}$ , are subject to negative feedback from the total cell density $n$ . In each case, the parameters self-adjust to attain the balance condition 4. It therefore follows that, in the case of dynamic heterogeneity, the system is robust, meaning that the failure of one feedback pathway can be compensated by another to maintain homeostasis. By contrast, for the H model, a stable homeostatic state is only attained if the cell fate bias $\Delta$ is a function of cell density $n$ (Supporting Information). Crowding feedback in the parameters $\lambda$ , $\gamma$ , and $r$ is not sufficient to confer stability.

Clonal Dynamics. So far we have discussed the average behavior of the DH model and its stability, but we have not addressed the dynamics of clones. Because the dynamics of the model is stochastic, the time evolution and survival of individual clones is variable and unpredictable. However, the dynamics of the statistical ensemble of clones can be determined. In the following we will consider the time-evolution of the clone size distribution in the balanced case (fulfilling Eq. 4), defined as the probability $P_{N_A,N_B}(t)$ to find a clone with $N_{A}A$ cells and $N_B$ cells at time $t$ when starting with a single labeled cell at $t = 0$ (clonal induction).

Assuming a representative labeling of cell types, starting with a single cell means that we have initially a cell of type $A$ with probability $\rho = \omega_A / (\omega_A + \omega_B)$ and of type $B$ with probability $(1 - \rho)$ . For the H model, Eq. 2, it was shown that, over time, the distribution of total clone sizes $N$ converges onto the form (8, 29) (for $N \gg 1$ )

$$
P _ {N} ^ {H} (t) = \frac {1}{(\Omega t) ^ {2}} \exp \left(- \frac {N}{\Omega t}\right), \tag {6}
$$

where $\Omega = \lambda r / \rho$ defines the growth rate of the average size of surviving clones, that is, clones that retain at least one cell, $\langle N\rangle_{s} = (1 - P_{0}^{H}(t))^{-1}\sum_{N > 0}NP_{N}^{H}(t) = \Omega t$ , and $P_0^H (t) = \Omega t / (1 + \Omega t)$ is the extinction probability (40). In turn, the survival probability (norm of $P_N^H$ in Eq. 6) diminishes as $1 / (\Omega t)$ at large times, so that the total cell number remains on average constant, consistent with homeostasis.

Formally, the dynamics of the clone size distribution for the DH model can be obtained from the master equation for the probability $P_{N_A,N_B}^{DH}(t)$ , as given in Eq. S2 of the Supporting Information. In general, a full analytic solution to the master equation is unavailable. However, to address the long-time dependence of the probability distribution in the balanced case (fullfilling Eq.

4), we proceed by a van Kampen system size expansion (41) to transform the master equation into the Fokker-Planck equation (FPE) involving a continuous function $P(N_A,N_B)$ that interpolates $P_{N_A,N_B}^{DH}$ (Supporting Information). The latter can be solved by an adequate variable substitution, and for large times the solution can be expressed as a function of $N = N_A + N_B$ only (Supporting Information),

$$
P (N, t) = \frac {1}{(\Omega t) ^ {2}} \exp \left(- \frac {N}{\Omega t}\right), \tag {7}
$$

with

$$
\Omega = \frac {\omega_ {B}}{\rho} \frac {1 + \frac {\omega_ {B}}{\lambda}}{\left(1 + \frac {\omega_ {B}}{\lambda \rho}\right) ^ {2}}, \tag {8}
$$

where we chose to eliminate the parameter $\gamma$ through Eq. 4. From these results it follows that the long-term clone size distribution of the DH model is identical to that of the hierarchical

model. It also follows that for a slow rate of cell type conversion, $\omega_{A,B} \ll \lambda$ , the clonal growth rate $\Omega \approx \omega_B / \rho$ is proportional to the switching rate, whereas for fast conversion, $\omega_{A,B} \gg \lambda$ , we have $\Omega \approx \lambda \rho$ , proportional to the cell division rate. In the latter case, each cell loses memory about its priming quickly and the two cell types behave just like a single proliferating cell population with both cell division rate and terminal differentiation rate $\lambda \rho$ . This means that, between two divisions, the probability of each cell to differentiate is $1/2$ , which corresponds to the cell fate model suggested by Marques-Pereira and Leblond (42) in their study of esophageal maintenance.

To assess how rapidly the system converges onto the limiting size dependences defined above, the full solution of the master equation can be determined at arbitrary times from numerical integration (Materials and Methods). In Fig. 2 and Fig. S1, the numerical solutions for the distribution of clone sizes for the DH and H model are shown as a function of the clone size, scaled by the average clone size. This comparison shows that H and DH models are not distinguishable, both at short times $(t = 1 / \Omega)$ and at longer times $(t = 10 / \Omega)$ .

Moreover, at $t = 10 / \Omega$ , the distributions have already converged onto the predicted long-term scaling form, Eq. 7. In the Supporting Information and Fig. S2 it is also confirmed that Eq. 8 for $\Omega$ agrees well with the numerical solution of the master equation.

Thus, when expressed in terms of the dimensionless rescaled variable $X = N / \langle N\rangle_{s}$ , the clone size distribution of the DH model and of the H model cannot be discriminated, based only on static clonal data.

Spatial Regulation. Although both the H model and DH model can be rendered stable in the zero-dimensional system through feedback mechanisms, the advantage of the latter theory, in terms of homeostatic control, becomes apparent when the system is embedded into a defined stem cell niche geometry (see refs. 43 and 44 for niche-based regulation mechanisms). To illustrate this point, in the following we consider two concrete examples: $(i)$ one-dimensional layers of progenitor cells, where the lowest layer is stem-like $(A$ cells) and higher layers are prone to differentiation $(B$ cells) and $(ii)$ a 2D epithelial sheet, as originally conceived for the H model in its application to basal interfollicular epidermis (45).

One-dimensional layers. As a starting point, we consider a quasi one-dimensional organization of cells in which a chain of $B$ -type cells is maintained by an adjacent population of $A$ -type cells, as illustrated in Fig. 3A. In the course of turnover, following the rules of the DH model (Eq. 1) we suppose that $A$ -type cells divide asymmetrically so that the daughter $B$ -type cell replaces

![](dt=2026-03-07/ht=00/e7a7c446e70f2455b6934bd3abafdbbf37a972c4a1406c3719062ea92e2d2f1f.jpg)

![](dt=2026-03-07/ht=00/d02947aff3a1e73f28397c589f3206a74d3254ab2d2ec4a40976773fb53b8581.jpg)

PNAAS

A090108 10N0110nDnNO 00N S0Hd0B

Greulich and Simons

PNAS Early Edition

3 of 6

![](dt=2026-03-07/ht=00/ce7890f1f4925ebe665b19ad7041bd619767d1a6896cedf3fb28dd7759eff379.jpg)

![](dt=2026-03-07/ht=00/99d8bf5ce3677cdd87b6e88f193fbe73280f4078dd61cf13d26885bdb2c785ce.jpg)

another $B$ -type cell in the upper layer, which is presumed to be lost. Alternatively, an $A$ -type cell (respectively $B$ -type cell) can "switch" into a $B$ -type cell (respectively $A$ -type cell). However, to maintain the architecture of the tissue, this change of identity is accompanied by a switch of a neighboring $B$ -type cell (respectively $A$ -type cell) and an exchange of their positions.

Such dynamics mimics the process of niche-based regulation in which stem cell competence relies only on the proximity of progenitors t
o a localized niche environment. A similar dynamics has been conjectured to define the maintenance of the intestinal epithelium, where stem cell competence is linked to the proximity of cells to Paneth cells, which are restricted to the crypt base (19).

Effectively, the model can be reorganized as a single one-dimensional chain of alternating cell types, with each cell initially belonging to a different clone:

$$
A _ {i} B _ {j} A _ {k} B _ {l} A _ {m} \dots , \tag {9}
$$

where the indices denote the clone affiliation. The model dynamics then read

$$
A _ {i} B _ {j} \xrightarrow {\lambda} A _ {i} B _ {i} \tag {10}
$$

$$
A _ {i} B _ {j} \xrightarrow {\omega} A _ {j} B _ {i}. \tag {11}
$$

Eq. 10 represents asymmetric division wherein a neighboring $B$ -type cell of clone $j$ is replaced by one of clone $i$ , and Eq. 11 represents a reversion of cell type followed by an exchange of position, with rate $\omega = \omega_{A} = \omega_{B}$ , according to the DH model. (Note that such exchanges can occur in both directions.) With these rules, the configuration of $A$ - and $B$ -type cells remains by definition unchanged, ensuring homeostasis.

To explore the clonal dynamics implied by this model, we made use of a Monte Carlo simulation (Materials and Methods). The resulting clone size distribution is shown in Fig. 3B together with the theoretical prediction for the analogous one-dimensional hierarchical model, a one-dimensional voter model in which stem cell loss through differentiation is compensated by the replacement of a neighbor through duplication (29). At long times, both models converge onto the same distribution. Thus, we conclude that the process of dynamic heterogeneity can provide a viable means to ensure long-term homeostasis and cannot be discriminated from a hierarchical model by static clonal fate data alone.

Two-dimensional cell sheet. In the following we now turn to address the implementation of cell fate models in a 2D geometry, like the arrangement of cells in an epithelial basal layer. More precisely, we consider a 2D lattice of cells illustrated in Fig. 4, in which all sites play host to either an $A$ - or $B$ -type cell. Keeping the total

number of cells on the lattice fixed (viz. uniform cell density), cell division of an $A$ -type cell only occurs when a neighboring $B$ -type cell commits to terminal differentiation and leaves the cell sheet (stratification). Effectively, the constraint of fixed cell number implements (local) crowding feedback, because cell division becomes licensed only when another cell leaves the cell layer, or vice versa. For the hierarchical scheme this dynamics is captured by the following process:

$$
A B \xrightarrow {\lambda} \left\{ \begin{array}{l l} A A & \Pr . r + \Delta / 2 \\ B A & \Pr . (1 - 2 r) / 2 \\ A B & \Pr . (1 - 2 r) / 2, \\ B B & \Pr . r - \Delta / 2 \end{array} \right. \tag {12}
$$

where only neighboring lattice sites are depicted (lattice H model). Formally, as the differentiating cell $B$ leaves the cell layer, the neighboring proliferative $A$ -type cell may divide to replenish the vacated site, with each progeny occupying either of the two sites with equal probability. However, this process is licensed to occur only on the condition that there is an $A$ -type progenitor bordering the differentiated $B$ type cell. In cases where $A$ cells border $A$ cells, or $B$ cells border $B$ cells, the system is "blocked" from cell division or stratification.

The process defined by Eq. 12 has been studied for the balanced case $\Delta = 0$ in ref. 45. There it was shown that the system coarsens over time and becomes increasingly inhomogeneous: The layer phase separates into $A$ - and $B$ -cell-rich domains that grow over time (see also Fig. 5, top row). This process of coarsening is accompanied by the gradual cessation of tissue turnover because only cells on the boundary of clusters can divide. Thus, the system remains in a nonhomeostatic state until the lattice consists of $A$ cells or $B$ cells only (fixation).

Therefore, to achieve steady-state turnover, further steps must be taken to regulate proliferative activity and/or fate behavior to ensure tissue maintenance in the hierarchical model. This lack of a homeostatic state also renders the definition of a clone size distribution problematic, because it depends on the specific initial condition of the configuration of cell types.

By contrast, in the paradigm of dynamic heterogeneity, steady-state behavior of the 2D system is ensured when the dynamics are implemented through the following process (Fig. 4):

$$
A B \xrightarrow {\lambda} \left\{ \begin{array}{l l} A B & \Pr . 1 / 2 \\ B A & \Pr . 1 / 2, \end{array} \quad A \xrightarrow {\omega_ {B}} B \right. \tag {13}
$$

(lattice DH model). Formally, although the chance development of $A$ - or $B$ -cell-rich clusters would, respectively, inhibit cell division and differentiation, the reversible transition of $A$ and $B$ -type cells will always serve to release the deadlock, allowing the system to achieve long-term steady-state behavior.

To illustrate the process by which the system evolves spatially, we implemented the model by Monte Carlo simulation

![](dt=2026-03-07/ht=00/f9b4933c14a95240fae16a9ad05ff7591ab1def61592b222077f9c454cbf8ccc.jpg)

PNAS

4 of 6

www.pnas.org/cgi/doi/10.1073/pnas.1602779113

Greulich and Simons

![](dt=2026-03-07/ht=00/65c8c7f7083e458d9028e4c53b702862887e143651fce98cf8aadab6499034dc.jpg)

(Materials and Methods) to determine the lattice configurations and clonal distributions. Fig. 5 shows the time course of spatial configurations of cell types. For the lattice DH model (bottom row) the system remains homeostatic and homogeneous on large length scales. In contrast, the lattice H model (Fig. 5, top row) shows a persistent, nonhomeostatic coarsening over time. Furthermore, the constraint of fixed cell number leads to a stable ratio of cell types in the lattice DH model, for any choice of parameters, while in the lattice H model any choice of $\Delta \neq 0$ leads to an imbalanced growth of one cell type population (see Supporting Information and Fig. S3).

Superficially, the "pathological" coarsening behavior of the lattice H model emerged through the rigid constraint on local cell density, which appeared in the model through the tight correlation of cell division with differentiation of a neighboring cell. This constraint may be relaxed by accommodating a degree of compressibility in the model. Formally, this can be achieved by accommodating vacancies or holes allowing for stratification of $B$ -type cells uncompensated by the division of neighboring $A$ -type cells. However, even under these conditions (considered in ref. 45), the dynamics of the lattice H model are qualitatively the same and phase-separated domains of cell types grow (see Supporting Information and Fig. S4).

Further insight into the steady-state properties of the system can be obtained by mapping these lattice models onto a corresponding "kinetic spin model," as encountered in statistical physics. By interpreting an $A$ -type cell as a spin $\uparrow$ degree of freedom and a $B$ -type cell as spin $\downarrow$ , the lattice DH model translates to a kinetic Ising model at infinite temperature, featuring a combination of Glauber dynamics (random spin flips) and Kawasaki dynamics (spin exchange) (46) (Supporting Information).

Starting from any initial condition, the system flows to a homogeneous equilibrium distribution of spins, with the ratio of $\uparrow$ , $\rho = \omega_{A} / (\omega_{B} + \omega_{A})$ , corresponding to a homeostatic state. By contrast, the lattice H model translates to a voter model, in which cells are stochastically and irreversible replaced by neighboring cells, following the transitions $\uparrow \d
ownarrow \rightarrow \uparrow \uparrow$ or $\uparrow \downarrow \rightarrow \downarrow \downarrow$ , respectively, with equal probability (47).

The latter model is nonergodic, exhibiting coarsening and phase separation into large irregular domains of cell states that grow over time (45, 47). This dynamics does not support a homeostatic state.

Tracing the clonal dynamics in the lattice DH model, we also compared the clone size distribution with results of a voter model, shown in Fig. 6 as the result of Monte Carlo simulations (Materials and Methods). According to Fig. 6 the rescaled clone size distributions of the lattice DH model cannot be distinguished from the voter model. Because it has been rigorously shown that the voter model clone size distribution converges onto an exponential distribution, we expect this also to be the case for the 2D lattice DH model (47).

# Discussion

Our study shows that dynamic heterogeneity in stem cell populations of cells reversibly switching between states that differ in their proliferative potential and their propensity toward differentiation yields a viable mechanism to maintain homeostatic tissues. Moreover, considering the long-term clone size dependences, a model based on dynamic heterogeneity cannot be discriminated from a hierarchical model.

Because hierarchical schemes involving intrinsic or extrinsic (niche-based) regulation have been used to infer self-renewal strategies in epithelial tissues such as mouse epidermis, esophagus, germline, and intestine (8-11), it follows that both dynamic heterogeneity and hierarchical fate may be equally capable of describing the results of recent lineage tracing assays. It is important to find further short-term characteristics that can help to discriminate these models. This remains true when the model is embedded in specific spatial niche architectures that resemble tissues.

Nonetheless, both dynamic heterogeneity and hierarchical balanced fate belong to the class of population asymmetry (4), in which stem cells are lost and replaced, with equal probability.

Through sensing of the cellular environment, cells may respond to variations in cell density and adjust cell division (contact inhibition) (32) and loss rate (33). We show that with this crowding feedback a dynamically heterogeneous system adjusts to attain a stable homeostatic state. The biological background of crowding feedback, by which the cells can measure local cell density, may find a basis in the mechanisms of mechanosensing (48), the limited exposure to diffusible molecules released by the niche environment (20), or biochemical communication between cells (44).

Importantly, for dynamic heterogeneity, homeostasis is robust toward disruption of some crowding feedback pathways. In contrast, for hierarchical models, involving intrinsic or cell-autonomous regulation of fate, it is essential that cell fate outcomes are specified at cell division, whereas the control of cell division or loss rates are not sufficient to maintain homeostasis.

In a 2D epithelium, however, the spatial dynamics of cell types show fundamental differences between dynamic heterogeneity and hierarchical cell fate, if left unchecked. The dynamics of the hierarchical model lead to a nonhomeostatic coarsening of the tissue structure, in that regions enriched with one cell type emerge so that the cell types phase-separate over time. In contrast, for dynamic heterogeneity, reversible switching between states homogenizes the tissue structure, leading to a macroscopically homogeneous, homeostatic cell population. The constraint of fixed total cell number in the lattice is sufficient to

![](dt=2026-03-07/ht=00/87d3786cce876eb781b3868887b5132ba6e33026e847f8b0bd3cb6f56fd5f4d3.jpg)

PNAS

Greulich and Simons

PNAS Early Edition

5 of 6

BIOPHYSICS AND COMPUTATIONAL BIOLOGY

confer cell fate balance in the case of dynamic heterogeneity, whereas for the hierarchical model the ratios of symmetric cell divisions need to be fine-tuned to assure balance.

To summarize, dynamic heterogeneity provides an alternative paradigm for cell fate dynamics in homeostatic tissues, in accordance with cell lineage data. It provides a simple mechanism to balance tissue homeostasis and to homogenize the distribution of cell types in epithelial sheets. The two cell types in our model may be considered as a caricature of a single progenitor cell population carrying some memory over cell generations that renders cells primed toward proliferation or differentiation. In the case of fast cell type conversion, this priming would be lost.

In this limit the system can be considered as a homogeneous cell population following the random differentiation dynamics of the model introduced by Marques-Pereira and Leblond (42). To determine whether cell fate behavior in epithelial tissues may involve dynamic heterogeneity, further detailed studies will be required that track cell lineages and distinguish cell states over time.

# Materials and Methods

Numerical Solution of Master Equation. The master equation, Eq. S2, is solved by numerical integration of the constituent ordinary differential equations, when a cutoff in $N_{A}$ and $N_{B}$ is applied. Specifically, we considered only terms with $N_{A,B} \leq 50$ and used an adaptive Runge-Kutta method via Mathematica to solve the resulting $50 \times 50$ ordinary differential equations to obtain $P_{N_A,N_B}$ . We then determined the rescaled clone size distribution $P(X = N / \langle N\rangle_s)$ , where $\langle N\rangle_s$ denotes the average size of surviving clones $(N > 0)$ .

Monte Carlo Simulations of Lattice Models. Time is subdivided in discrete time steps $\Delta t = 1 / \phi_{\mathrm{max}}$ , where $\phi_{\mathrm{max}}$ is the largest transition rate out of any system configuration. At each time step $\Delta t$ , a lattice site and one of its neighbors are $N$ times randomly chosen ( $N$ is the number of lattice sites). Then a random variable is generated and any possible transition, as defined by the models 10, 12, and 13, with given rate $\phi$ , is chosen to be updated with probability $\phi \Delta t$ , according to the Gillespie algorithm (49).

ACKNOWLEDGMENTS. We thank Steffen Rulands for insightful discussions. This work was supported by a Research Fellowship of the German Research Foundation and an Engineering and Physical Sciences Research Council Critical Mass Grant (EP/J017639/1).

PNS

6 of 6

www.pnas.org/cgi/doi/10.1073/pnas.1602779113

Greulich and Simons