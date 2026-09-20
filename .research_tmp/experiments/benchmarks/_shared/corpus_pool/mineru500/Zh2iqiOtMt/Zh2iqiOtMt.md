# Towards the Fundamental Limits of Knowledge Transfer over Finite Domains

Qingyue Zhao $^{*}$ and Banghua Zhu $^{\dagger}$

# Abstract

We characterize the statistical efficiency of knowledge transfer through n samples from a teacher to a probabilistic student classifier with input space S over labels A. We show that privileged information at three progressive levels accelerates the transfer. At the first level, only samples with hard labels are known, via which the maximum likelihood estimator attains the minimax rate $\sqrt{|S||A|/n}$ . The second level has the teacher probabilities of sampled labels available in addition, which turns out to boost the convergence rate lower bound to $|S||A|/n$ . However, under this second data acquisition protocol, minimizing a naive adaptation of the cross-entropy loss results in an asymptotically biased student. We overcome this limitation and achieve the fundamental limit by using a novel empirical variant of the squared error logit loss. The third level further equips the student with the soft labels (complete logits) on A given every sampled input, thereby provably enables the student to enjoy a rate $|S|/n$ free of $|A|$ . We find any Kullback-Leibler divergence minimizer to be optimal in the last case. Numerical simulations distinguish the four learners and corroborate our theory.

# 1 Introduction

It has become common sense that transferring intrinsic information from teachers to the greatest extent can expedite a student's learning progress, especially in machine learning given versatile and powerful teacher models. Learning with their assistance has been coined knowledge distillation (KD) (Hinton et al., 2015; Lopez-Paz et al., 2015), a famous paradigm of knowledge transfer leading to remarkable empirical effectiveness in classification tasks across various downstream applications (Gou et al., 2021; Wang and Yoon, 2021; Gu et al., 2023b). The term distillation implies a belief that the inscrutable teacher(s) may possess useful yet complicated structural information, which we should be able to compress and inject into a compact one, i.e., the student model (Breiman and Shang, 1996; Buciluă et al., 2006; Li et al., 2014; Ba and Caruana, 2014; Allen-Zhu and Li, 2020). This has guided the community towards a line of knowledge transfer methods featuring the awareness of teacher training details or snapshots, such as the original training set, the intermediate activations, the last-layer logits (for a probabilistic classifier), the first- or second-order derivative or statistical information, and even task-specific knowledge (Hinton et al., 2015; Furlanello et al., 2018; Cho and Hariharan, 2019; Zhao et al., 2022; Romero et al., 2014; Zagoruyko and Komodakis, 2016;

Yim et al., 2017; Huang and Wang, 2017; Park et al., 2019; Tian et al., 2019; Tung and Mori, 2019; Qiu et al., 2022; Srinivas and Fleuret, 2018; Cheng et al., 2020; Liang et al., 2023).

However, in more general modes of knowledge transfer, the procedure- or architecture-specific information can be unavailable, irrelevant, or ill-defined. Modern proprietary large language models (LLMs) like GPT-4 (OpenAI, 2023b) and Claude 2 (Anthropic, 2023) strictly confine the data returned through their APIs except the generated tokens, not to mention their training sets. Therefore, there has been an effort towards distilling LLMs only through oracle queries via locally crafted prompts (Zheng et al., 2023; Peng et al., 2023). In denoising (Tsybakov, 2003), boosting (Freund and Schapire, 1995), and prediction with expert advice (Cesa-Bianchi et al., 1997), the teacher group itself is the classes of interest and the student only gets positive-valued feedback from the teacher group, so the teacher implementation is immaterial. In certain circumstances of robot learning (Abbeel and Ng, 2004; Ross and Bagnell, 2010), the teacher demonstration may come from threshold-based classifiers or even human experts, where no teacher probability is defined.

To unify different natures of teacher information acquisition in knowledge transfer and assess the technical barriers in a principled way, we decompose the transfer set (the information a student has access to in total) into a generative model $\pi^{\star}(\cdot|\cdot)$ giving one a in the label space A given a query s in the input space S and the additional information provided by the teacher for each sample pair $(s,a)$ . This informative decomposition is motivated by a more general learning using privileged information (LUPI) framework (Vapnik et al., 2015; Lopez-Paz et al., 2015). Take several LLMs as examples. In OpenAI's chat completion API $^{1}$ for GPT-4, only responses are returned given a prompt s, in which no privileged information is accessible. An early established version $^{2}$ for GPT-3 (Brown et al., 2020), however, also returns the log-likelihood of every token in the generated response. If further with open-sourced models like GPT-2 (Radford et al., 2019) in hand, practitioners can extract the last-layer logits of each position in the sampled sequence as the privileged information. These typical examples reveal a trend in the development of language models $^{3}$ that the more powerful a foundation model (Bommasani et al., 2021) is, the less likely it is to release privileged information as a teacher model, which raises a grandiose question beyond learning from neural teacher networks:

What is the fundamental limit of knowledge transfer given limited privileged information in general?

In this work, we answer this question with minimal inductive bias through the analyses of three protocols of information extraction from the teacher with easier difficulties in order in terms of the (offline) accessibility of the reference policy $\pi^{\star}$ over finite S and A. Concretely, for any learner $\widehat{\pi}$ , we study the total variation between it and the reference policy $\pi^{\star}$ conditioned on an input distribution $\rho$ , i.e.,

$$
\sum_ {s \in \mathcal {S}} \rho (s) \mathsf {T V} (\widehat {\pi} (\cdot | s), \pi^ {\star} (\cdot | s)) =: \mathsf {T V} (\widehat {\pi}, \pi^ {\star} | \rho),
$$

where $\widehat{\pi}$ can utilize n samples $\{(s_{i},a_{i})\}_{i=1}^{n}$ and optionally certain fraction of $\pi^{\star}(\cdot|s_{i}),\forall i\in[n]$ .

# 1.1 Main Contributions and Clarifications

Table 1 gives an overview of rate bounds in distinct data acquisition protocols. Hard Labels indicates $\pi^{\star}$ to be a black-box for $\widehat{\pi}$ throughout the learning process. Partial SLs stands for partial soft labels,

which means the teacher provides the student with an extra (partial) ground truth $\pi^{\star}(a|s)$ for every sample $(s,a)$ . Soft Labels is a synonym of the logits on the entire A exposed to $\widehat{\pi}$ given each s in $\{s_{i}\}_{i=1}^{n}$ . Intuitively, each of the three levels discloses richer information than previous ones, which at least cannot make it harder to learn. Rigorously, progressively faster minimax rates are matching exactly at all levels $^{4}$ and every high-probability upper bound has its upper confidence radius at most the same order of the corresponding expectation lower bound up to polylog factors.

Table 1: Major theoretical findings of knowledge transfer with different acquired data. The second column shows one out of n data points available to the student. Worst-case bounds in the last column hold with high probability and $\widetilde{O}(\cdot)$ hides polylog factors. 

<table><tr><td>Transfer via</td><td>Data Format</td><td>Lower Bound</td><td>Loss</td><td>Performance</td></tr><tr><td>Hard Labels</td><td> $(s, a)$ </td><td> $\Omega\left(\sqrt{|\mathcal{S}||\mathcal{A}|/n}\right)$ Theorem 3.1</td><td> $\text{CE}_{\text{sgl}}$ </td><td> $\widetilde{\mathcal{O}}\left(\sqrt{|\mathcal{S}||\mathcal{A}|/n}\right)$ Theorem 3.2</td></tr><tr><td rowspan="2">Partial SLs</td><td rowspan="2"> $(s, a, \pi^{\star}(a|s))$ </td><td rowspan="2"> $\Omega\left(|\mathcal{S}||\mathcal{A}|/n\right)$ Theorem 4.1</td><td> $\text{CE}_{\text{ptl}}$ </td><td rowspan="2"> $\Omega(1)$ Theorem 4.3 $\widetilde{\mathcal{O}}\left(|\mathcal{S}||\mathcal{A}|/n\right)$ Theorem 4.4</td></tr><tr><td> $\text{SEL}_{\text{ptl}}$ </td></tr><tr><td>Soft Labels</td><td> $(s, a, \pi^{\star}(\cdot|s))$ </td><td> $\Omega\left(|\mathcal{S}|/n\right)$ Theorem 5.1</td><td> $\text{CE}_{\text{ful}}$ </td><td> $\widetilde{\mathcal{O}}\left(|\mathcal{S}|/n\right)$ Theorem 5.2</td></tr></table>

Technically, the later two protocols are nonstandard in density estimation, especially in terms of the derivation of minimax lower bounds in that we assume the data generating process to be $(\rho \times \pi^{\star})^{n}$ . The constructive proof (Appendix C.3) of Theorem 4.1 may be of independent interest. The performances at the second level are also more tricky. Theorem 4.3 sentences a naive adaptation of the cross-entropy loss $(CE_{\mathrm{ptl}})$ to be a misfit with probability 1 when the privileged information is at a modest level.

We do not require $\pi^{\star}$ to be trained in any sense and assume $\widehat{\pi}$ to have no awareness of the teacher implementation, inspired by the aforementioned practical trend. Consequently, we do no distinguish the teacher probability from the Bayes probability (Menon et al., 2020; Dao et al., 2021) and $\rho$ can have no relation with teacher training. We idealize the samples to be i.i.d., while all the results already extend to the setting where different $(s,a)$ pairs are mildly dependent, of which we refer the readers to Appendix G for detailed discussions.

We give more discussions on the connections between our framework and previous formulations in Appendix A.2 and review key related works aiming at the principles of knowledge transfer in the next subsection.

# 1.2 Related Works on Understanding Knowledge Transfer

Hinton et al. (2015) refers to the logits generated on a dataset by a teacher model, who has been trained using the same dataset, as soft labels; and refers to $\{a_{i}\}_{i=1}^{n}$ in this dataset as hard labels.

Our terms, however, pay no attention to whether $\rho$ matches the original dataset; and our Soft Labels strictly carry more information (of $\rho \times \pi^{\star}$ ) than our Hard Labels, both of whose inputs $\{s_{i}\}_{i=1}^{n}$ are sampled from $\rho$ . So the third column of Table 1 does not account for the class similarities argument (Furlanello et al., 2018), the regularization effect via label smoothing (LS) argument (Yuan et al., 2020; Tang et al., 2020), or the bias-variance trade-off argument (Zhou et al., 2021; Menon et al., 2020); all of which are classical wisdom explaining the benefit of soft labels in KD (Hinton et al., 2015).

These previous views are neither consistent nor complete for justifying why soft labels improves student training. (Müller et al., 2020) undermines the hypothesis that class similarities in soft labels are vital by the effectiveness of KD in binary classification. Han et al. (2023) challenges the regularization via LS thesis through better interpretability-lifting effect of KD than LS. Dao et al. (2021) develops the bias-variance trade-off perspective in more complex scenarios. In contrast, we define Soft Labels (similarly Partial SLs) and tackle its boost over Hard Labels rigorously following the information-theoretic direction of the data processing inequality (Polyanskiy and Wu, 2022). We survey more works unraveling knowledge transfer. Other empirical and enlightening paradigms are deferred to Appendix A.

Most analyses of KD with trained deep nets as the teacher (Phuong and Lampert, 2019; Ji and Zhu, 2020; Panahi et al., 2022; Harutyunyan et al., 2023) lie in the linear or kernel regime, notably except Hsu et al. (2021), which finds the student network to have fundamentally tighter generalization bound than the teacher network under several nonlinear function approximation schemes. There are also works analyzing knowledge transfer between neural networks of the same architecture (Mobahi et al., 2020; Allen-Zhu and Li, 2020). Our framework goes exactly in the reverse direction: our analysis is not restricted to parsimonious students, over-parameterized teachers; or the compression subconcept of knowledge transfer (Buciluă et al., 2006; Bu et al., 2020). For example, a human demonstrator, who does not learn only from data and is able to output probabilistic belief, can also fit into the first column of Table 1 as a kind of teacher under our specification of $\pi^{\star}$ .

# 2 Preliminaries

Notation. For two nonnegative sequences $\{a_{n}\}$ and $\{b_{n}\}$ , we write $a_{n}=O(b_{n})$ or $a_{n}\lesssim b_{n}$ if $\limsup a_{n}/b_{n}<\infty$ ; equivalently, $b_{n}=\Omega(a_{n})$ or $b_{n}\gtrsim a_{n}$ . We write $c_{n}=\Theta(d_{n})$ or $c_{n}\asymp d_{n}$ if $c_{n}=O(d_{n})$ and $c_{n}=\Omega(d_{n})$ . For a metric space $(\mathcal{M},d)$ , $\text{dist}_{d}(\cdot,\mathcal{N}):=\inf_{y\in\mathcal{N}}d(\cdot,y)$ for any $N\subset M$ . The term alphabet is a synonym of finite set. For a set or multiset C, let $|C|$ denote its cardinality. For any alphabet X, on which given two distributions p, q, the total variation between them is $\mathsf{TV}(p,q):=0.5\left\|\boldsymbol{p}-q\right\|_{1}$ , their Kullback-Leibler (KL) divergence is $\mathsf{KL}(p\|q):=\mathbb{E}_{p}[\log p-\log q]$ ; and we denote all $|X|$ Dirac distributions on X by $\text{Dirac}(\mathcal{X})$ , where $\text{Dirac}(\mathcal{X},x)$ stands for the one concentrated at x. For two alphabets X and Y, we denote by $\Delta(\mathcal{Y})$ the probability simplex on Y and define $\Delta(\mathcal{Y}|\mathcal{X}):=\{\pi(\cdot|\cdot):\mathcal{X}\to\Delta(\mathcal{Y})\}\Rightarrow\pi(\cdot|x)\in\Delta(\mathcal{Y}),\forall x\in\mathcal{X}$ .

# 2.1 Common Setup

The teacher always exposes to the student a multiset $\mathcal{D} = \{(s_i, a_i) \in \mathcal{S} \times \mathcal{A}\}_{i=1}^n$ consisting of $n$ i.i.d. input-label tuples. To analyze $\mathcal{D}$ in a fine-grained way we introduce for $\mathcal{X}^n \ni X^n \stackrel{i.i.d.}{\sim} \nu$ the number of occurrences $\mathfrak{n}_x(X^n)$ of $x$ and the missing mass $\mathfrak{m}_0(\nu, X^n)$ , which measures the portion of $\mathcal{X}$ never observed in $X^n$ (McAllester and Ortiz, 2003). We refer to the input (resp. label) component $\{s_i\}_{i=1}^n$

$(\text{resp. } \{a_i\}_{i=1}^n)$ of $\mathcal{D}$ as $\mathcal{S}(\mathcal{D})$ (resp. $\mathcal{A}(\mathcal{D})$ ) and also define a multiset $\mathcal{A}(\mathcal{D}, s)$ for every $s \in \mathcal{S}$ to denote the $a_i$ 's in $\mathcal{D}$ that are associated with the visitations of $s$ , taking into account multiplicity. $^5$

# 2.2 Quantity of Interest

We assume $\mathcal{D}\sim(\rho\times\pi^{\star})^{n}$ follows a product measure, where $\pi^{\star}$ is the ground truth distribution over A given every $s\in S$ and $\rho$ is some underlying input generating distribution. No assumption is imposed on the data generating process $\rho\times\pi^{\star}$ except for belonging to

$$
\mathcal {P} := \left\{\rho \times \pi^ {\star}: \rho \in \Delta (\mathcal {S}), \pi^ {\star} \in \Delta (\mathcal {A} | \mathcal {S}) \right\}.
$$

In this work, we evaluate the performance of a student $\widehat{\pi}$ based on the TV between $\widehat{\pi}$ and $\pi^{\star}$ conditioned on $\rho$ defined as

$$
\mathsf {T V} (\widehat {\pi}, \pi^ {\star} | \rho) := \mathbb {E} _ {s \sim \rho} \left[ \mathsf {T V} (\widehat {\pi}, \pi^ {\star}) \right], \tag {2.1}
$$

though the student is never allowed to access $\rho$ directly. We investigate the convergence rate of (2.1) among three categories of students told by the teacher's degree of openness in the tabular setting. Intuitively speaking, all the learning procedures of interest try to match the log-probability kernel $\log\pi$ , a notion of normalized logits, between the student and the teacher, especially via variants of the cross-entropy loss, which is standard in the study of classification both theoretically and practically (Paszke et al., 2019). Besides the universal definition $\mathsf{CE}_{\mathsf{ful}}(p||q):=-\mathbb{E}_{p}[\log q]$ of the cross-entropy between $p\ll q$ , a popular counterpart for hard labels specialized to classifiers is commonly defined as $\mathsf{CE}_{\mathsf{sgl}}(s,a;\pi):=\mathsf{CE}_{\mathsf{ful}}(\mathsf{Dirac}(\mathcal{A},a)||\pi(\cdot|s))=-\log\pi(a|s)$ .

# 3 Transfer via Hard Labels

This is equivalent to the standard setting for estimating the conditional density $\pi^{\star}(\cdot|\cdot)$ .

# 3.1 Hardness of Estimation

We first generalize the idea of constructing hard instances for learning discrete distributions on $\mathcal{A}$ (Paninski, 2008) to our nonsingleton $\mathcal{S}$ to understand the difficulty when only $(s,a)$ paris are available. We remark that the proof of Theorem 3.1 (in Appendix C.2) is the only one in this work that utilizes Assouad's method (Yu, 1997) directly.

Theorem 3.1. For nonempty S, A with $|A| > 1$ , and $n \geq |S||A|/4$ ,

$$
\inf _ {\widehat {\pi} \in \widehat {\Pi} (\mathcal {D})} \sup _ {\rho \times \pi^ {\star} \in \mathcal {P}} \mathbb {E} _ {(\rho \times \pi^ {\star}) ^ {n}} \mathrm{TV} (\widehat {\pi}, \pi^ {\star} | \rho) \gtrsim \sqrt {\frac {| \mathcal {S} | | \mathcal {A} |}{n}}, \tag {3.1}
$$

where $\mathcal{D} \sim (\rho \times \pi^{\star})^{n}$ , $\widehat{\Pi}(\mathcal{D})$ denotes all (possibly random) estimators mapping $\mathcal{D}$ to $\Delta(\mathcal{A}|\mathcal{S})$ .

The $\sqrt{|\mathcal{S}|}$ dependence in this lower bound intuitively makes sense because the classic lower bound for $|S|=1$ is $\Omega(\sqrt{|A|/n})$ and each input roughly get $n/|S|$ samples when $\rho$ is distributed evenly.

# 3.2 Maximum Likelihood Estimation

We approximate the teacher $\pi^{\star}$ via minimizing the following negative log-likelihood loss:

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} \in \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} \mathsf {C E} _ {\mathsf {s g l}} (\mathcal {D}) := \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} - \sum_ {i = 1} ^ {n} \log \pi (a _ {i} | s _ {i}). \tag {3.2}
$$

It is possible to exactly attain the minimum 0 in (3.2). A refactoring detailed in Appendix B.1 indicates a natural relation between the hard version $CE_{sgl}$ and the soft version $CE_{ful}$ , which leads to a neat closed-form solution for the optimization problem.

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (a | s) \left\{ \begin{array}{l l} = \mathfrak {n} _ {(s, a)} (\mathcal {D}) / \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})), & s \in \mathcal {S} (\mathcal {D}), \\ \in \Delta (\mathcal {A}) \text {arbitrarily,} & \text {otherwise.} \end{array} \right. \tag {3.3}
$$

We study the convergence behavior of $\widehat{\pi}_{CE,sgl}$ in a fine-grained way with its proof detailed in Appendix D.2:

Theorem 3.2. For any $\delta\in(0,1)$ , with probability at least $1-\delta$ ,

$$
\mathrm{TV} (\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}}, \pi^ {\star} | \rho) \lesssim \sqrt {\frac {| \mathcal {S} | (| \mathcal {A} | + \log (| \mathcal {S} | / \delta))}{n}}. \tag {3.4}
$$

The upper bound in expectation $\mathbb{E}\left[\mathsf{TV}(\widehat{\pi}_{\mathsf{CE},\mathsf{sgl}},\pi^{\star}|\rho)\right]\lesssim\sqrt{|S||\mathcal{A}|/n}$ is no better than (3.4) up to log factors. An instance-dependent version in expectation is

$$
\mathbb {E} \left[ \mathrm{TV} (\widehat {\pi} _ {\mathrm{CE}, \mathrm{sgl}}, \pi^ {\star} | \rho) \right] \lesssim \sqrt {\xi (\pi^ {\star}) \frac {| \mathcal {S} | | \mathcal {A} |}{n}} + \frac {| \mathcal {S} |}{n}, \tag {3.5}
$$

where $\xi (\pi^{\star})\coloneqq \max_{s\in \mathcal{S}}\mathrm{dist}_{\mathsf{TV}}(\pi^{\star}(\cdot |s),\mathrm{Dirac}(\mathcal{A})) = 2\max_{s\in \mathcal{S}}\min_{a\in \mathcal{A}}(1 - \pi^{\star}(a|s)).$

Thus, $\widehat{\pi}_{\mathsf{CE},\mathsf{sg}!}$ is worst-case optimal and may even have a $n^{-1}$ rate in some benign cases with $\pi^{\star}$ close enough to vertices of the simplex $\Delta(\mathcal{A})$ .

# 4 Transfer via Partial SLs

Besides D, we can also access $\mathcal{R} := \{(s_i, a_i, \pi^\star(a_i | s_i))\}_{i=1}^n$ as Partial SLs. The introduction of R leads to a quadratic reduction of the learning difficulty.

# 4.1 Blessing of Ground Truth

Theorem 4.1. For nonempty S, A with $|A| > 2$ , and $n \geq |\mathcal{S}|(|\mathcal{A}| - 1)/2 - 1$ ,

$$
\inf _ {\widehat {\pi} \in \widehat {\Pi} (\mathcal {D}, \mathcal {R})} \sup _ {\rho \times \pi^ {\star} \in \mathcal {P}} \mathbb {E} _ {(\rho \times \pi^ {\star}) ^ {n}} \mathsf {T V} (\widehat {\pi}, \pi^ {\star} | \rho) \gtrsim \frac {| \mathcal {S} | | \mathcal {A} |}{n}, \tag {4.1}
$$

where $\mathcal{D} \sim (\rho \times \pi^{\star})^{n}$ , $\mathcal{R} = \{(s, a, \pi^{\star}(a|s)): (s, a) \in \mathcal{D}\}$ , and $\widehat{\Pi}(\mathcal{D}, \mathcal{R})$ denotes all (possibly random) learners mapping $(\mathcal{D}, \mathcal{R})$ to $\Delta(\mathcal{A}|\mathcal{S})$ .

We provide a constructive proof of Theorem 4.1 in Appendix C.3, in which we resort to the power of randomized policy so as to reveal the linear in $|A|$ dependence.

# 4.2 Failure of Empirical Cross-Entropy Loss

The Partial SLs R motivates us to define a loss $CE_{ptl}$ interpolating between $CE_{sgl}$ and $CE_{ful}$ .

$$
\widehat {\pi} _ {\mathsf {C E}, \mathrm{ptl}} \in \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} \mathsf {C E} _ {\mathrm{ptl}} (\mathcal {D}, \mathcal {R}) := \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} - \sum_ {i = 1} ^ {n} \pi^ {\star} (a _ {i} | s _ {i}) \log \pi (a _ {i} | s _ {i}). \tag {4.2}
$$

We can obtain the following exact solution to $(4.2)$ by another technique detailed in Appendix B.2.

$$
\widehat {\pi} _ {\mathsf {C E}, \mathrm{ptl}} (a | s) \left\{ \begin{array}{l l} \propto \mathfrak {n} _ {(s, a)} (\mathcal {D}) \pi^ {\star} (a | s), & s \in \mathcal {S} (\mathcal {D}), \\ \in \Delta (\mathcal {A}) \text {   arbitrarily }, & s \notin \mathcal {S} (\mathcal {D}). \end{array} \right. \tag {4.3}
$$

The convergence analysis of $\widehat{\pi}_{CE,ptl}$ crucially relies on its relationship with $\widehat{\pi}_{CE,sgl}$ . For any $s \in \mathcal{S}(\mathcal{D})$ , $\widehat{\pi}_{CE,ptl}$ can be reformulated as

$$
\widehat {\pi} _ {\mathsf {C E}, \mathrm{ptl}} (a | s) = \frac {\pi^ {\star} (a | s) \widehat {\pi} _ {\mathsf {C E} , \mathrm{sgl}} (a | s)}{\sum_ {b \in \mathcal {A}} \pi^ {\star} (b | s) \widehat {\pi} _ {\mathsf {C E} , \mathrm{sgl}} (b | s)}. \tag {4.4}
$$

Lemma 4.2. For $|\mathcal{S}| = 1$ ,

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {p t l}} (\cdot | s) \xrightarrow {a . s .} [ \pi^ {\star} (\cdot | s) ] ^ {2} / \sum_ {a \in \mathcal {A}} [ \pi^ {\star} (a | s) ] ^ {2},
$$

under $\ell_{\infty}$ in $\mathbb{R}^{|\mathcal{A}|}$ if $\rho \times \pi^{\star}$ is independent of $n$ .

Lemma 4.2 roughly means $\widehat{\pi}_{\mathrm{CE,ptl}} \propto (\pi^{\star})^{2}$ approximately as $n$ goes to infinity, which implies that small parts of $\pi^{\star}$ are underestimated and large parts of $\pi^{\star}$ are overestimated. We make this intuition technically right in Theorem 4.3, whose rigorous statement, mechanism, and proof are deferred to Appendix D.4.

Theorem 4.3. If $\rho \times \pi^{\star}$ does not vary with $n$ , $\widehat{\pi}_{\mathsf{CE,ptl}}$ coincides with $\widehat{\pi}_{\mathsf{CE,sgl}}$ (and thus asymptotically unbiased $^6$ ) only if $\pi^{\star}(\cdot |s) = \operatorname{Uniform}(\mathcal{A})$ or $\pi^{\star}(\cdot |s)\in \operatorname{Dirac}(\mathcal{A})$ for all $s\in \mathcal{S}$ . Even for $|\mathcal{S}| = 1$ , $\widehat{\pi}_{\mathsf{CE,ptl}}$ is asymptotically biased in general.

# 4.3 Empirical Squared Error Logit Loss

Ba and Caruana (2014) suggests the practically promising SEL loss $^{7}$ :

$$
\mathcal {L} (\pi , \pi^ {\star}) = \sum_ {i = 1} ^ {n} \frac {1}{2} \sum_ {a \in \mathcal {A}} [ \log \pi (a | s _ {i}) - \log \pi^ {\star} (a | s _ {i}) ] ^ {2}. \tag {4.5}
$$

Here we analyze the minimization of its empirical variant with normalized logits under the second data acquisition protocol for simplicity:

$$
\mathrm{SEL} _ {\mathrm{ptl}} (\mathcal {D}, \mathcal {R}) := \sum_ {i = 1} ^ {n} \frac {1}{2} \left[ \log \pi (a _ {i} | s _ {i}) - \log \pi^ {\star} (a _ {i} | s _ {i}) \right] ^ {2}. \tag {4.6}
$$

Exact matching on the seen samples in $(\mathcal{D},\mathcal{R})$ shows that $\widehat{\pi}_{SEL,ptl} \in \arg\min_{\pi \in \Delta(\mathcal{A}|S)} SEL_{ptl}(\mathcal{D},\mathcal{R})$

$$
\Leftrightarrow \widehat {\pi} _ {\mathsf {S E L}, \mathsf {p t l}} (\cdot | s) \in \left\{ \begin{array}{l l} \{p \in \Delta (\mathcal {A}): p (a) = \pi^ {\star} (a | s), \forall a \in \mathcal {A} (\mathcal {D}, s) \}, & s \in \mathcal {S} (\mathcal {D}), \\ \Delta (\mathcal {A}) \text {   arbitrarily }, & \text { otherwise }. \end{array} \right. \tag {4.7}
$$

The following three-fold Theorem 4.4 with its proof in Appendix D.4 indicates that $\widehat{\pi}_{\mathrm{SEL,ptl}}$ converges faster than $\widehat{\pi}_{\mathrm{CE,sgl}}$ by a factor of $\sqrt{n}$ though its performance upper bound has worse dependence on $|\mathcal{S}||\mathcal{A}|$ compared with that of $\widehat{\pi}_{\mathrm{CE,sgl}}$ .

Theorem 4.4. If $|S| > 1$ , for any $\delta \in (0, \min(1, (|S| + 2)/10))$ , with probability at least $1 - \delta$ ,

$$
\mathrm{TV} (\widehat {\pi} _ {\mathrm{SEL}, \mathrm{ptl}}, \pi^ {\star} | \rho) \lesssim \frac {| \mathcal {S} | (| \mathcal {A} | + \sqrt {| \mathcal {A} | \log (| \mathcal {S} | / \delta)})}{n} \log \frac {| \mathcal {S} |}{\delta}. \tag {4.8}
$$

If $|\mathcal{S}| = 1$ , for any $\delta \in (0,1/10]$ , with probability at least $1 - \delta$ ,

$$
\mathrm{TV} \left(\widehat {\pi} _ {\mathrm{SEL}, \mathrm{ptl}}, \pi^ {\star} | \rho\right) = \mathrm{TV} \left(\widehat {\pi} _ {\mathrm{SEL}, \mathrm{ptl}}, \pi^ {\star}\right) \lesssim \frac {| \mathcal {A} |}{n} + \frac {\sqrt {| \mathcal {A} |}}{n} \log \frac {1}{\delta}. \tag {4.9}
$$

The expected risk $\mathbb{ETV}(\widehat{\pi}_{\mathsf{SEL},\mathsf{ptl}},\pi^{\star}|\rho)\lesssim^{|\mathcal{S}||\mathcal{A}|/n}$ is not polynomially tighter than (4.8) or (4.9).

Remark 4.5. Theorem 4.3 together with Theorem 4.4 manifests the advantage of employing the empirical SEL loss, which induces an alignment between the normalized logits of the learner and those of $\pi^{\star}$ under squared loss, over the empirical CE loss in offline distillation when the teacher is moderately reserved. A similar observation between these two style of empirical surrogate losses in online policy optimization is verified in practice (Zhu et al., 2023b).

# 5 Transfer via Soft Labels

At the lightest level, the student has the extra information $\mathcal{Q}=\{(s,\pi^{\star}(\cdot|s):s\in\mathcal{S}(\mathcal{D})\}$ . The availability of Q apparently eases the transfer process, especially when the support size $|A|$ of the teacher classifier is huge. Such an intuition can be precisely depicted by a $|A|$ -free minimax lower bound.

# 5.1 |A|-Free Lower Bound

The following lower bound roughly requires $\Omega(|\mathcal{S}|)$ burn-in cost, whose constructive proof is deferred to Appendix C.4.

Theorem 5.1. For S with $|S| > 1$ , A with $|A| > 1$ , and $n > |S| - 1$ ,

$$
\inf _ {\widehat {\pi} \in \widehat {\Pi} (\mathcal {D}, \mathcal {Q})} \sup _ {\rho \times \pi^ {\star} \in \mathcal {P}} \mathbb {E} _ {(\rho \times \pi^ {\star}) ^ {n}} \mathrm{TV} (\widehat {\pi}, \pi^ {\star} | \rho) \gtrsim \frac {| \mathcal {S} |}{n}, \tag {5.1}
$$

where $\mathcal{D}\sim(\rho\times\pi^{\star})^{n},\mathcal{Q}=\{(s,\pi^{\star}(\cdot|s)):s\in\mathcal{S}(\mathcal{D})\}$ , and $\widehat{\Pi}(\mathcal{D},\mathcal{Q})$ denotes all (possibly random) learners mapping $(\mathcal{D},\mathcal{Q})$ to $\Delta(\mathcal{A}|\mathcal{S})$ .

This setting cannot have a rate better than $n^{-1}$ , which is consistent with the $n^{-1}$ rate in Theorem 4.1 since the difficulties of the later two settings (intuitively, the information provided at these two levels) should be the same when $\pi^{\star}(\cdot|s)\in\mathsf{Dirac}(\mathcal{A})$ for any $s\in S$ .

# 5.2 Kullback-Leibler Divergence Minimization

Cross-entropy loss minimization under full observation is equivalent to

$$
\widehat {\pi} _ {\mathsf {C E} _ {\text {ful}}} \in \underset {\pi} {\operatorname{argmin}} \sum_ {i = 1} ^ {n} \mathrm{KL} \left(\pi^ {\star} (\cdot | s _ {i}) \| \pi (\cdot | s _ {i})\right) \Rightarrow \widehat {\pi} _ {\mathsf {C E} _ {\text {ful}}} (\cdot | s) \left\{ \begin{array}{l l} = \pi^ {\star} (\cdot | s), & \text {if} s \in \mathcal {S} (\mathcal {D}), \\ \in \Delta (\mathcal {A}) \text {arbitrarily,} & \text {otherwise.} \end{array} \right. \tag {5.2}
$$

We give the missing-mass-based proof of the matching upper bounds for $\widehat{\pi}_{CE_{ful}}$ in Theorem 5.2 in Appendix D.6.

Theorem 5.2. For any $\delta \in (0,1/10]$ , with probability at least $1 - \delta$ ,

$$
\mathrm{TV} (\widehat {\pi} _ {C E _ {\text { ful }}}, \pi^ {\star} | \rho) \lesssim \frac {| S |}{n} + \frac {\sqrt {| S |}}{n} \log \frac {1}{\delta}. \tag {5.3}
$$

The upper bound $\mathbb{ETV}(\widehat{\pi}_{\mathsf{CE}_{\mathrm{ful}}},\pi^{\star}|\rho)\lesssim|S|/n$ on the expected risk for $\widehat{\pi}_{CE_{ful}}$ nearly matches (5.3).

Theorem 5.2 guarantees the optimality (not only in expectation but also in expectation) of $\widehat{\pi}_{CE_{ful}}$ in that it maximally utilizes the given logits.

# 6 Experiments

We conduct simulations to verify the intuitive performance rankings $\widehat{\pi}_{CE,sgl} \preceq \widehat{\pi}_{SEL,ptl} \preceq \widehat{\pi}_{CE_{ful}}$ given moderately large sample sizes and also numerically provide the asymptotical biasedness of $\widehat{\pi}_{CE,ptl}$ with a finite-sample counterpart. Moreover, we design adversarial data generating distributions inspired by the information-theoretic arguments (Appendix C) for the three types of reserved teachers respectively in the non-asymptotic regime, thereby accurately exhibiting the matching convergence rates of $\widehat{\pi}_{CE,sgl}$ , $\widehat{\pi}_{SEL,ptl}$ , and $\widehat{\pi}_{CE_{ful}}$ in terms of n.

In this section, we specify a fair inductive bias due to the tabular nature: if $s \notin \mathcal{S}(\mathcal{D})$ , $\widehat{\pi}(\cdot|s)$ is set to $\text{Uniform}(\mathcal{A})$ for all learners; for $\widehat{\pi}_{\text{SEL,ptl}}(\cdot|s)$ , the missing mass is amortized uniformly among $\mathcal{A}\backslash\mathcal{A}(\mathcal{D}, s)$ if $s \in \mathcal{S}(\mathcal{D})$ .

# 6.1 Classic Regime: Telling Learners Apart

In the classic regime, $\rho \times \pi^{\star}$ stays invariant no matter whether n tends to infinity or not. An instance in this sense should not only expose the inferior of $\widehat{\pi}_{CE,ptl}$ but also showcase the hardness of $|S| > 1$ , i.e., $\rho$ should be strictly bounded away from zero in $\Omega(|S|)$ inputs. To these ends, we specify Instance 0 in Appendix F. We simulate four typical realizations of it, whose estimated risks is presented in Figure 1. Each marker in Figure 1 represents the empirical mean of $TV(\widehat{\pi}, \pi^{\star}|\rho)$ in 100 independent repeats given the corresponding sample size n. Any broken line in Figure 1 has nothing to do with sequential design and our experiment is purely offline. As shown in Figure 1 (a, b), either in a general case with $(|S|, |A|) = (100, 25)$ or for a 100-armed rewardless bandit, $\widehat{\mathbb{E}}TV(\widehat{\pi}_{CE,ptl}, \pi^{\star}|\rho)$ fails to converge but all other learners do, corroborating the asymptotically constant bias of $\widehat{\pi}_{CE,ptl}$ and the consistency of the others.

Though Instance 0 is effectively so easy to learn for any of $\widehat{\pi}_{\mathrm{CE,sgl}}$ , $\widehat{\pi}_{\mathrm{SEL,ptl}}$ , and $\widehat{\pi}_{\mathrm{CE_{ful}}}$ that none of the three worst-case upper bounds is tightly attained, the numerical performance rankings among them in Figure 1 coincide with our intuition and theoretical analysis. The “benign”-case

![](images/5a308bd3627d85f9a2281b528d706698cb4b5cbc5cf41e9c0c4a049c8e958af4.jpg)

<details>
<summary>line</summary>

| n    | CE_sgl | CE_ptl | SEL_ptl | CE_ful |
| ---- | ------ | ------ | ------- | ------ |
| 0    | 0.6    | 0.5    | 0.45    | 0.43   |
| 100  | 0.6    | 0.6    | 0.37    | 0.29   |
| 200  | 0.58   | 0.58   | 0.17    | 0.18   |
| 300  | 0.55   | 0.55   | 0.1     | 0.07   |
| 400  | 0.52   | 0.52   | 0.05    | 0.03   |
| 500  | 0.5    | 0.48   | 0.03    | 0.02   |
| 600  | 0.48   | 0.46   | 0.02    | 0.01   |
| 700  | 0.46   | 0.45   | 0.01    | 0.01   |
| 800  | 0.45   | 0.44   | 0.01    | 0.01   |
| 900  | 0.44   | 0.44   | 0.01    | 0.01   |
| 1000 | 0.43   | 0.44   | 0.01    | 0.01   |
| 1100 | 0.42   | 0.44   | 0.01    | 0.01   |
| 1200 | 0.41   | 0.44   | 0.01    | 0.01   |
| 1300 | 0.4    | 0.44   | 0.01    | 0.01   |
| 1400 | 0.39   | 0.44   | 0.01    | 0.01   |
| 1500 | 0.38   | 0.44   | 0.01    | 0.01   |
| 1600 | 0.37   | 0.44   | 0.01    | 0.01   |
| 1700 | 0.36   | 0.44   | 0.01    | 0.01   |
| 1800 | 0.35   | 0.44   | 0.01    | 0.01   |
| 1900 | 0.34   | 0.44   | 0.01    | 0.01   |
| 2000 | 0.33   | 0.44   | 0.01    | 0.01   |
| 2100 | 0.32   | 0.44   | 0.01    | 0.01   |
| 2200 | 0.31   | 0.44   | 0.01    | 0.01   |
| 2300 | 0.3    | 0.44   | 0.01    | 0.01   |
| 2400 | 0.29   | 0.44   | 0.01    | 0.01   |
| 2500 | 0.28   | 0.44   | 0.01    | 0.01   |
</details>

![](images/375ef2a428d4ad3980c08ff0c98c664785eae9e9fdff47fe9072ac310815cd4b.jpg)  
Figure 1: Estimated risks in expectation. $\rho \times \pi^{\star}$ does not vary with n.

comparison in Figure 1 (c), where the sample sizes are small enough to make the worst-case lower bounds vacuous, still favors $\widehat{\pi}_{\mathsf{CE}_{\mathsf{ful}}}$ over $\widehat{\pi}_{\mathsf{SEL},\mathsf{ptl}}$ in the large- $|\mathcal{A}|$ regime. Figure 1 (d) manifests that the $\sqrt{|\mathcal{S}||\mathcal{A}|}$ gap between the worst-case upper bounds for $\widehat{\pi}_{\mathsf{CE},\mathsf{sgl}}$ and $\widehat{\pi}_{\mathsf{SEL},\mathsf{ptl}}$ , which is reversely dominated by the exponential rate $^8$ of $\widehat{\pi}_{\mathsf{SEL},\mathsf{ptl}}$ in this Instance 0, may not be observed in general even for large $|\mathcal{S}||\mathcal{A}|$ and small $n$ .

Remark 6.1. Direct calculations imply that if $\rho \geq c_{\mathcal{S}} > 0$ for all inputs with $c_{\mathcal{S}}$ irrespective of $n$ when $n$ is large enough, $\mathbb{ETV}(\widehat{\pi}_{\mathsf{CE}_{\mathsf{ful}}},\pi^{\star}|\rho)$ can decay exponentially fast, exemplified by Figure 1 (a). $\mathbb{ETV}(\widehat{\pi}_{\mathsf{SEL,ptl}},\pi^{\star}|\rho)$ will enjoy a similar linear convergence so long as we additionally require $\pi^{\star}(\cdot |s)\geq c_{\mathcal{A}} > 0$ for all $s\in \mathcal{S}$ and all labels with $c_{\mathcal{A}}$ independent of $n$ for sufficiently large $n$ , again exemplified by Figure 1 (a).

# 6.2 Non-Asymptotic Regime: Illustration of Matching Rates

Instance 0 serves as an intriguing average case, but we need to design worst-case instances that may vary with n (Wainwright, 2019) in the non-asymptotic regime for different data acquisition settings in order to verify the minimax optimalities. The adversarial Instance 1, 2, and 3 with their design insights are detailed in order in Appendix F for verification of matching rates at all three levels.

Since $\widehat{\pi}_{\mathsf{CE,sgl}}$ , $\widehat{\pi}_{\mathsf{SEL,ptl}}$ , and $\widehat{\pi}_{\mathsf{CE_{ful}}}$ enjoy optimal rates of order $n^{-1}$ or $n^{-0.5}$ , we can manifest them using lines in a log-log plot. More generally, if some notion of risk has risk = $\Theta(n^{\beta^{\star}})$ for some $\beta^{\star} < 0$ , log risk - $\beta^{\star}\log n$ will be at least bounded by two straight lines on a log-log scale. We instantiate the above idea for Instance 1, Instance 2, and Instance 3 in Figure 2, in which each marker represents the average of 64000 independent repeats. We also conduct linear regressions

over log risk $\sim \log n$ for corresponding minimax learners and report the slope $\widehat{\beta}$ as estimated $\beta^{\star}$ in each subfigure of Figure 2.

![](images/e9975c29e822bdffc49dd71aa836b28d0bfa379ca42ef917cfb134188c8e7499.jpg)

<details>
<summary>line</summary>

| n       | CE_sgl: β̂ ≈ -0.532 (for n ≥ 10³) | CE_ptl | SEL_ptl | CE_ful |
| ------- | ------------------------------- | ------ | ------- | ------ |
| 10^0    | ~1.0                            | ~1.0   | ~0.1    | ~0.0001 |
| 10^1    | ~1.0                            | ~1.0   | ~0.1    | ~0.00001 |
| 10^2    | ~1.0                            | ~1.0   | ~0.05   | ~0.000001 |
| 10^3    | ~1.0                            | ~1.0   | ~0.02   | ~0.0000001 |
| 10^4    | ~1.0                            | ~1.0   | ~0.01   | ~0.00000001 |
| 10^5    | ~1.0                            | ~1.0   | ~0.005  | ~0.000000001 |
| 10^6    | ~1.0                            | ~1.0   | ~0.002  | ~0.0000000001 |
| 10^7    | ~1.0                            | ~1.0   | ~0.001  | ~0.00000000001 |
| 10^8    | ~1.0                            | ~1.0   | ~0.0005 | ~0.000000000001 |
| 10^9    | ~1.0                            | ~1.0   | ~0.0002 | ~0.0000000000001 |
| 10^10   | ~1.0                            | ~1.0   | ~0.0001 | ~0.00000000000001 |
| 10^11   | ~1.0                            | ~1.0   | ~0.0      | ~0.0      |
| 10^12   | ~1.0                            | ~1.0   | ~0.      | ~*       |
| 10^13   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^14   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^15   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^16   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^17   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^18   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^19   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^20   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^21   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^22   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^23   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^24   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^25   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^26   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^27   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^28   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^29   | ~1.0                            | ~1.0   | ~*      | ~*       |
| 10^3    | ~1.0                            | ~1.0   | ~*      | ~*       |
| 1        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 2        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 3        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 4        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 5        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 6        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 7        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 8        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 9        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 1         | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 2        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 3        | 5e-3                            | 5e-3   | 5e-3    | 5e-4     |
| 4        + N=2 to N=4, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=8, N=9, N=6, N=<ecel><ecel><ecel><nl>
</details>

![](images/0182c47e151fba558bafe183b802da1e5cedaebd94eac63785dbb93bc1e15661.jpg)

<details>
<summary>line</summary>

| n       | CE_sgl     | CE_ptl     | SEL_ptl: β̂ ≈ -1.004 (for n ≥ 10³) | CE_ful     |
| ------- | ---------- | ---------- | ---------------------------------- | ---------- |
| 10²     | ~10⁰       | ~10⁰       | ~10⁰                               | ~10⁰       |
| 10³     | ~10⁻²      | ~10⁻²      | ~10⁻²                              | ~10⁻⁴      |
</details>

![](images/dac6f0a96ae9d062e93b6f0ba888c0723240b773ab063f24aac81a934fc54f84.jpg)

<details>
<summary>line</summary>

| n     | CEsgl  | CEptl  | SELptl | CEful  |
|-------|--------|--------|--------|--------|
| 100   | 0.5    | 0.4    | 0.3    | 0.2    |
| 200   | 0.3    | 0.25   | 0.2    | 0.15   |
| 500   | 0.15   | 0.12   | 0.1    | 0.08   |
| 1000  | 0.1    | 0.08   | 0.06   | 0.05   |
| 2000  | 0.06   | 0.05   | 0.04   | 0.03   |
| 5000  | 0.03   | 0.025  | 0.02   | 0.015  |
| 10000 | 0.02   | 0.015  | 0.01   | 0.01   |
</details>

![](images/f438edb9ffb4a99bebad942ae8cee9f1a82d022fa0c55aa8ab7228369796a980.jpg)

<details>
<summary>line</summary>

| n     | CE_sgi: β̂ ≈ -0.525 | CE_ptl | SEL_ptl | CE_ful |
|-------|-------------------|--------|---------|--------|
| 10^1  | ~10^-3            | ~10^-1 | ~10^-1  | ~10^-3 |
| 10^2  | ~10^-4            | ~10^-1 | ~10^-2  | ~10^-4 |
| 10^3  | ~10^-5            | ~10^-1 | ~10^-6  | ~10^-7 |
</details>

![](images/275de9ed2a93be0eb54229cdb3fbac7927a7c60c4834946c25bc729bf54c8652.jpg)

<details>
<summary>line</summary>

| n     | CE_sgl | CE_ptl | SEL_ptl: β̂ ≈ -0.998 | CE_ful |
|-------|--------|--------|---------------------|--------|
| 100   | 0.1    | 0.08   | 0.03                | 0.0001 |
| 200   | 0.05   | 0.04   | 0.015               | 0.00005|
| 500   | 0.02   | 0.015  | 0.005               | 0.00002|
| 1000  | 0.01   | 0.008  | 0.002               | 0.00001|
| 2000  | 0.005  | 0.004  | 0.001               | 0.000005|
| 5000  | 0.002  | 0.0015 | 0.0005              | 0.000002|
| 10000 | 0.001  | 0.0008 | 0.0002              | 0.000001|
</details>

![](images/66b74346ac770a4f987035a946e0d9a0f880105ab33590452dd9a16a1d0a5f08.jpg)

<details>
<summary>line</summary>

| n     | CE_sgl | CE_ptl | SEL_ptl | CE_ful: β̂ ≈ -0.977 |
|-------|--------|--------|---------|-------------------|
| 100   | 0.1    | 0.1    | 0.05    | 0.05              |
| 1000  | 0.01   | 0.01   | 0.005   | 0.005             |
</details>

Figure 2: Estimated risks in expectation. $\rho \times \pi^{\star}$ scales with n in setting-specific ways.

We elaborate on several interesting phenomena of Figure 2.

- Regarding parameters other than $n$ as constants, the $\widehat{\beta}$ 's precisely verify the worst-case rates $\mathbb{ETV}(\widehat{\pi}_{\mathsf{CE},\mathsf{sgl}},\pi^{\star}|\rho)\asymp n^{-0.5}$ (Figure 2 (a, b)), $\mathbb{ETV}(\widehat{\pi}_{\mathsf{SEL},\mathsf{ptl}},\pi^{\star}|\rho)\asymp n^{-1}$ (Figure 2 (c, d)), and $\mathbb{ETV}(\widehat{\pi}_{\mathsf{CE}_{\mathsf{ful}}},\pi^{\star}|\rho)\asymp n^{-1}$ (Figure 2 (e, f)).   
- All Instance 1, 2, and 3 happen to be too easy for $\widehat{\pi}_{\mathsf{CE,ptl}}$ to be obviously biased. This phenomenon is actually predictable because $\pi^{\star}(\cdot |s)$ becomes very close to $\operatorname{Uniform}(\mathcal{A})$ in Instance 1, and to some one in $\operatorname{Dirac}(\mathcal{A})$ in both Instance 2 or 3 for all inputs when $n$ is relatively large. In these cases, $\widehat{\pi}_{\mathsf{CE,ptl}}$ largely coincides with $\widehat{\pi}_{\mathsf{CE,sgl}}$ as predicted by Theorem 4.3.   
- The risks of $\widehat{\pi}_{\mathsf{SEL,ptl}}$ in Instance 1 (Figure 2 (a, b)) and the risks of $\widehat{\pi}_{\mathsf{CE_{ful}}}$ in Instance 1 (Figure 2 (a, b)) and Instance 2 (Figure 2 (c, d)) decay exponentially fast, which is consistent with the arguments in Remark 6.1, i.e., only vanishing schemes like $\pi^{\star}$ in Instance 2 for $\widehat{\pi}_{\mathsf{SEL,ptl}}$ and $\rho$ in Instance 3 for $\widehat{\pi}_{\mathsf{CE_{ful}}}$ can force their risks to have a polynomial decay.   
- We are able to provably explain the good performance (beyond worst cases) of $\widehat{\pi}_{\mathsf{CE,sgl}}$ in Instance 2 (Figure 2 (c, d)) and Instance 3 (Figure 2 (e, f)), in which $\xi(\pi^{\star}) \lesssim n^{-1}$ . (Recall

the definition of $\xi (\cdot)$ in Theorem 3.2.) Thus, the instance-dependent bound in Theorem 3.2 indicates a benign-case rate of $\mathbb{ETV}(\widehat{\pi}_{\mathsf{CE},\mathsf{sgl}},\pi^{\star}|\rho)\lesssim n^{-1}$ .

# 7 Conclusion

We embark on investigating knowledge transfer beyond pure black- or white-box regimes and settle its sample complexity, respectively; provided that the teacher can afford (1) only to act as a generative model, or (2) additionally the probabilities at sampled classes, or (3) additionally the logits conditioned on each sampled input. The theoretical analysis unveils a crucial insight that tailoring the idea of minimizing CE to new information acquisition scenarios may be sub-optimal in general, provably in knowledge transfer via Partial SLs. Several avenues remain to be further explored.

- Huge S and A in practice (Zeng et al., 2022; Almazrouei et al., 2023) necessitate function approximation, which is also important to our analysis itself. $^{9}$ For example, the equivalent effectiveness of minimizing the CE or SEL loss (with Soft Labels) from an alphabet-matching perspective may be incomplete after incorporating the approximation error, which may consequently corroborate the empirical superior of the vanilla SEL loss (Ba and Caruana, 2014).   
- All our upper bounds do not adapt to the student initialization strategy for unseen inputs or labels; while the student may be pre-trained in practice (Gu et al., 2023b; Jiang et al., 2023). Thus it is vital to incorporate the skillfulness of the student before transfer to the convergence analysis.

# References

ABBEEL, P. and NG, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning.   
AGARWAL, R., VIEILLARD, N., STANCZYK, P., RAMOS, S., GEIST, M. and BACHEM, O. (2023). Gkd: Generalized knowledge distillation for auto-regressive sequence models. arXiv preprint arXiv:2306.13649.   
ALLEN-ZHU, Z. and LI, Y. (2020). Towards understanding ensemble, knowledge distillation and self-distillation in deep learning. arXiv preprint arXiv:2012.09816.   
ALMAZROUEI, E., ALOBEIDLI, H., ALSHAMSI, A., CAPPELLI, A., COJOCARU, R., ALHAMMADI, M., DANIELE, M., HESLOW, D., LAUNAY, J., MALARTIC, Q., NOUNE, B., PANNIER, B. and PENEDO, G. (2023). The falcon series of language models: Towards open frontier models.   
ALYAFEAI, Z., ALSHAIBANI, M. S. and AHMAD, I. (2020). A survey on transfer learning in natural language processing. arXiv preprint arXiv:2007.04239.   
ANTHROPIC (2023). Model card and evaluations for claude models. Accessed: Sep. 27, 2023.   
BA, J. and CARUANA, R. (2014). Do deep nets really need to be deep? Advances in neural information processing systems 27.   
BOMMASANI, R., HUDSON, D. A., ADELI, E., ALTMAN, R., ARORA, S., VON ARX, S., BERNSTEIN, M. S., BOHG, J., BOSSELUT, A., BRUNSKILL, E. ET AL. (2021). On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.   
BREIMAN, L. and SHANG, N. (1996). Born again trees. University of California, Berkeley, Berkeley, CA, Technical Report 1 4.   
BRETAGNOLLE, J. and HUBER, C. (1979). Estimation des densités: risque minimax. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete 47 119–137.   
BROWN, D., Goo, W., NAGARAJAN, P. and NIEKUM, S. (2019). Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations. In International conference on machine learning. PMLR.   
BROWN, T., MANN, B., RYDER, N., SUBBIAH, M., KAPLAN, J. D., DHARIWAL, P., NEELAKANTAN, A., SHYAM, P., SASTRY, G., ASKELL, A. ET AL. (2020). Language models are few-shot learners. Advances in neural information processing systems 33 1877–1901.   
BU, Y., GAO, W., ZOU, S. and VEERAVALLI, V. (2020). Information-theoretic understanding of population risk improvement with model compression. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34.   
BUCILUĂ, C., CARUANA, R. and NICULESCU-MIZIL, A. (2006). Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining.   
BURGES, C. J. and SCHÖLKOPF, B. (1996). Improving the accuracy and speed of support vector machines. Advances in neural information processing systems 9.

CANONNE, C. L. (2020). A short note on learning discrete distributions. arXiv preprint arXiv:2002.11457.   
CESA-BIANCHI, N., FREUND, Y., HAUSSLER, D., HELMBOLD, D. P., SCHAPIRE, R. E. and WARMUTH, M. K. (1997). How to use expert advice. Journal of the ACM (JACM) 44 427–485.   
CHEN, L., PALEJA, R. and GOMBOLAY, M. (2021). Learning from suboptimal demonstration via self-supervised reward regression. In Conference on robot learning. PMLR.   
CHENG, X., RAO, Z., CHEN, Y. and ZHANG, Q. (2020). Explaining knowledge distillation by quantifying the knowledge. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.   
CHO, J. H. and HARIHARAN, B. (2019). On the efficacy of knowledge distillation. In Proceedings of the IEEE/CVF international conference on computer vision.   
DAO, T., KAMATH, G. M., SYRGKANIS, V. and MACKEY, L. (2021). Knowledge distillation as semiparametric inference. arXiv preprint arXiv:2104.09732.   
DURRETT, R. (2019). Probability: theory and examples, vol. 49. Cambridge university press.   
FANG, G., MO, K., WANG, X., SONG, J., BEI, S., ZHANG, H. and SONG, M. (2022). Up to 100x faster data-free knowledge distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36.   
FREUND, Y. and SCHAPIRE, R. E. (1995). A desicion-theoretic generalization of on-line learning and an application to boosting. In European conference on computational learning theory. Springer.   
FURLANELLO, T., LIPTON, Z., TSCHANNEN, M., ITTI, L. and ANANDKUMAR, A. (2018). Born again neural networks. In International Conference on Machine Learning. PMLR.   
GAO, L., SCHULMAN, J. and HILTON, J. (2023). Scaling laws for reward model overoptimization. In International Conference on Machine Learning. PMLR.   
GAO, L., TOW, J., BIDERMAN, S., BLACK, S., DIPOFI, A., FOSTER, C., GOLDING, L., HSU, J., MCDONELL, K., MUENNIGHOFF, N., PHANG, J., REYNOLDS, L., TANG, E., THITE, A., WANG, B., WANG, K. and ZOU, A. (2021). A framework for few-shot language model evaluation.   
GOU, J., YU, B., MAYBANK, S. J. and TAO, D. (2021). Knowledge distillation: A survey. International Journal of Computer Vision 129 1789–1819.   
GU, J., ZHAI, S., ZHANG, Y., LIU, L. and SUSSKIND, J. M. (2023a). Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference { $\&$ } Generative Modeling.   
Gu, Y., Dong, L., Wei, F. and Huang, M. (2023b). Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543.   
HAN, H., KIM, S., CHOI, H.-S. and YOON, S. (2023). On the impact of knowledge distillation for model interpretability. arXiv preprint arXiv:2305.15734.

HARUTYUNYAN, H., RAWAT, A. S., MENON, A. K., KIM, S. and KUMAR, S. (2023). Supervision complexity and its role in knowledge distillation. arXiv preprint arXiv:2301.12245.   
HINTON, G., VINYALS, O. and DEAN, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.   
HSU, D., JI, Z., TELGARSKY, M. and WANG, L. (2021). Generalization bounds via distillation. arXiv preprint arXiv:2104.05641.   
HUANG, Z. and WANG, N. (2017). Like what you like: Knowledge distill via neuron selectivity transfer. arXiv preprint arXiv:1707.01219.   
Ji, G. and ZHU, Z. (2020). Knowledge distillation in wide neural networks: Risk bound, data efficiency and imperfect teacher. Advances in Neural Information Processing Systems 33 20823-20833.   
JIANG, H., HE, P., CHEN, W., LIU, X., GAO, J. and ZHAO, T. (2019). Smart: Robust and efficient fine-tuning for pre-trained natural language models through principled regularized optimization. arXiv preprint arXiv:1911.03437.   
JIANG, Y., CHAN, C., CHEN, M. and WANG, W. (2023). Lion: Adversarial distillation of closed-source large language model. arXiv preprint arXiv:2305.12870.   
Kim, T., Oh, J., Kim, N., Cho, S. and Yun, S.-Y. (2021). Comparing kullback-leibler divergence and mean squared error loss in knowledge distillation. arXiv preprint arXiv:2105.08919.   
LI, J., ZHAO, R., HUANG, J.-T. and GONG, Y. (2014). Learning small-size dnn with output-distribution-based criteria. In Fifteenth annual conference of the international speech communication association.   
LI, X., ZHANG, T., DUBOIS, Y., TAORI, R., GULRAJANI, I., GUESTRIN, C., LIANG, P. and HASHIMOTO, T. B. (2023). Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca\_eval.   
LIANG, C., ZUO, S., ZHANG, Q., HE, P., CHEN, W. and ZHAO, T. (2023). Less is more: Task-aware layer-wise distillation for language model compression. In International Conference on Machine Learning. PMLR.   
LIU, H., LI, C., WU, Q. and LEE, Y. J. (2023). Visual instruction tuning. arXiv preprint arXiv:2304.08485.   
LOPES, R. G., FENU, S. and STARNER, T. (2017). Data-free knowledge distillation for deep neural networks. arXiv preprint arXiv:1710.07535.   
LOPEZ-PAZ, D., BOTTOU, L., SCHÖLKOPF, B. and VAPNIK, V. (2015). Unifying distillation and privileged information. arXiv preprint arXiv:1511.03643.   
McALLESTER, D. and ORTIZ, L. (2003). Concentration inequalities for the missing mass and for histogram rule error. Journal of Machine Learning Research 4 895–911.

MENON, A. K., RAWAT, A. S., REDDI, S. J., KIM, S. and KUMAR, S. (2020). Why distillation helps: a statistical perspective. arXiv preprint arXiv:2005.10419.   
MITZENMACHER, M. and UPFAL, E. (2017). Probability and computing: Randomization and probabilistic techniques in algorithms and data analysis. Cambridge university press.   
MOBAHI, H., FARAJTABAR, M. and BARTLETT, P. (2020). Self-distillation amplifies regularization in hilbert space. Advances in Neural Information Processing Systems 33 3351–3361.   
MÜLLER, R., KORNBLITH, S. and HINTON, G. (2020). Subclass distillation. arXiv preprint arXiv:2002.03936.   
NAYAK, G. K., MOPURI, K. R., SHAJ, V., RADHAKRISHNAN, V. B. and CHAKRABORTY, A. (2019). Zero-shot knowledge distillation in deep networks. In International Conference on Machine Learning. PMLR.   
NG, A. Y., RUSSELL, S. ET AL. (2000). Algorithms for inverse reinforcement learning. In Icml, vol. 1.   
NGUYEN, D., GUPTA, S., DO, K. and VENKATESH, S. (2022). Black-box few-shot knowledge distillation. In European Conference on Computer Vision. Springer.   
OPENAI (2023a). Accessed: Sep. 28,2023.   
OPENAI (2023b). Gpt-4 technical report.   
OREKONDY, T., SCHIELE, B. and FRITZ, M. (2019). Knockoff nets: Stealing functionality of black-box models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.   
OUYANG, L., WU, J., JIANG, X., ALMEIDA, D., WAINWRIGHT, C., MISHKIN, P., ZHANG, C., AGARWAL, S., SLAMA, K., RAY, A. ET AL. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35 27730–27744.   
PANAHI, A., RAHBAR, A., BHATTACHARYYA, C., DUBHASHI, D. and HAGHIR CHEHREGHANI, M. (2022). Analysis of knowledge transfer in kernel regime. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management.   
PANINSKI, L. (2008). A coincidence-based test for uniformity given very sparsely sampled discrete data. IEEE Transactions on Information Theory 54 4750–4755.   
PARK, W., KIM, D., LU, Y. and CHO, M. (2019). Relational knowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.   
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., Devito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. and Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché Buc, E. Fox and R. Garnett, eds.). Curran Associates, Inc.

PENG, B., LI, C., HE, P., GALLEY, M. and GAO, J. (2023). Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277.   
PHUONG, M. and LAMPERT, C. (2019). Towards understanding knowledge distillation. In International conference on machine learning. PMLR.   
POLYANSKIY, Y. and WU, Y. (2022). Information theory: From coding to learning. Book draft .   
QIU, Z., MA, X., YANG, K., LIU, C., HOU, J., YI, S. and OUYANG, W. (2022). Better teacher better student: Dynamic prior knowledge for knowledge distillation. arXiv preprint arXiv:2206.06067.   
RADFORD, A., WU, J., CHILD, R., LUAN, D., AMODEI, D., SUTSKEVER, I. ET AL. (2019). Language models are unsupervised multitask learners. OpenAI blog 1 9.   
RAJARAMAN, N., YANG, L. F., JIAO, J. and RAMACHANDRAN, K. (2020). Toward the fundamental limits of imitation learning. arXiv preprint arXiv:2009.05990.   
RAMESH, A., DHARIWAL, P., NICHOL, A., CHU, C. and CHEN, M. (2022). Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1 3.   
RAMESH, A., PAVLOV, M., GOH, G., GRAY, S., VOSS, C., RADFORD, A., CHEN, M. and SUTSKEVER, I. (2021). Zero-shot text-to-image generation. In International Conference on Machine Learning. PMLR.   
RASHIDINEJAD, P., ZHU, B., MA, C., JIAO, J. and RUSSELL, S. (2021). Bridging offline reinforcement learning and imitation learning: A tale of pessimism. Advances in Neural Information Processing Systems 34 11702–11716.   
ROMERO, A., BALLAS, N., KAHOU, S. E., CHASSANG, A., GATTA, C. and BENGIO, Y. (2014). Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550.   
ROSS, S. and BAGNELL, D. (2010). Efficient reductions for imitation learning. In Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings.   
SHALEV-SHWARTZ, S. and BEN-DAVID, S. (2014). Understanding machine learning: From theory to algorithms. Cambridge university press.   
SRINIVAS, S. and FLEURET, F. (2018). Knowledge transfer with jacobian matching. In International Conference on Machine Learning. PMLR.   
TANG, J., SHIVANNA, R., ZHAO, Z., LIN, D., SINGH, A., CHI, E. H. and JAIN, S. (2020). Understanding and improving knowledge distillation. arXiv preprint arXiv:2002.03532.   
TIAN, Y., KRISHNAN, D. and ISOLA, P. (2019). Contrastive representation distillation. arXiv preprint arXiv:1910.10699.   
TIAN, Y., PEI, S., ZHANG, X., ZHANG, C. and CHAWLA, N. V. (2023). Knowledge distillation on graphs: A survey. arXiv preprint arXiv:2302.00219.

TOUVRON, H., LAVRIL, T., IZACARD, G., MARTINET, X., LACHAUX, M.-A., LACROIX, T., ROZIÈRE, B., GOYAL, N., HAMBRO, E., AZHAR, F. ET AL. (2023a). Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.   
TOUVRON, H., MARTIN, L., STONE, K., ALBERT, P., ALMAHAIRI, A., BABAEI, Y., BASHLYKOV, N., BATRA, S., BHARGAVA, P., BHOSALE, S. ET AL. (2023b). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.   
TSYBAKOV, A. B. (2003). Optimal rates of aggregation. In Learning Theory and Kernel Machines: 16th Annual Conference on Learning Theory and 7th Kernel Workshop, COLT/Kernel 2003, Washington, DC, USA, August 24-27, 2003. Proceedings. Springer.   
TSYBAKOV, A. B. (2009). Introduction to Nonparametric Estimation. Springer series in statistics, Springer.   
TUNG, F. and MORI, G. (2019). Similarity-preserving knowledge distillation. In Proceedings of the IEEE/CVF international conference on computer vision.   
VAN DER VAART, A. W. (2000). Asymptotic statistics, vol. 3. Cambridge university press.   
VAN DER VAART, A. W. and WELLNER, J. A. (1996). Springer series in statistics. Weak convergence and empirical processesSpringer, New York.   
VAPNIK, V., IZMAILOV, R. ET AL. (2015). Learning using privileged information: similarity control and knowledge transfer. J. Mach. Learn. Res. 16 2023–2049.   
WAINWRIGHT, M. J. (2019). High-dimensional statistics: A non-asymptotic viewpoint, vol. 48. Cambridge university press.   
WANG, D., LI, Y., WANG, L. and GONG, B. (2020). Neural networks are more productive teachers than human raters: Active mixup for data-efficient knowledge distillation from a blackbox model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.   
WANG, G., CHENG, S., YU, Q. and LIU, C. (2023a). OpenChat: Advancing Open-source Language Models with Imperfect Data.   
WANG, L. and YOON, K.-J. (2021). Knowledge distillation and student-teacher learning for visual intelligence: A review and new outlooks. IEEE transactions on pattern analysis and machine intelligence 44 3048–3068.   
WANG, Y., IVISON, H., DASIGI, P., HESSEL, J., KHOT, T., CHANDU, K. R., WADDEN, D., MACMILLAN, K., SMITH, N. A., BELTAGY, I. and HAJISHIRZI, H. (2023b). How far can camels go? exploring the state of instruction tuning on open resources.   
WANG, Z. (2021). Zero-shot knowledge distillation from a decision-based black-box model. In International Conference on Machine Learning. PMLR.   
WEISS, K., KHOSHGOFTAAR, T. M. and WANG, D. (2016). A survey of transfer learning. Journal of Big data 3 1–40.

YIM, J., JOO, D., BAE, J. and KIM, J. (2017). A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In Proceedings of the IEEE conference on computer vision and pattern recognition.   
YIN, H., MOLCHANOV, P., ALVAREZ, J. M., LI, Z., MALLYA, A., HOIEM, D., JHA, N. K. and KAUTZ, J. (2020). Dreaming to distill: Data-free knowledge transfer via deepinversion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.   
YIN, M., BAI, Y. and WANG, Y.-X. (2021). Near-optimal provable uniform convergence in offline policy evaluation for reinforcement learning. In International Conference on Artificial Intelligence and Statistics. PMLR.   
Yu, B. (1997). Assouad, fano, and le cam. In Festschrift for Lucien Le Cam: research papers in probability and statistics. Springer, 423–435.   
YUAN, L., TAY, F. E., LI, G., WANG, T. and FENG, J. (2020). Revisiting knowledge distillation via label smoothing regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.   
ZAGORUYKO, S. and KOMODAKIS, N. (2016). Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928   
ZENG, A., LIU, X., DU, Z., WANG, Z., LAI, H., DING, M., YANG, Z., XU, Y., ZHENG, W., XIA, X. ET AL. (2022). Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414   
ZHAO, B., CUI, Q., SONG, R., QIU, Y. and LIANG, J. (2022). Decoupled knowledge distillation. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition.   
ZHENG, L., CHIANG, W.-L., SHENG, Y., ZHUANG, S., WU, Z., ZHUANG, Y., LIN, Z., LI, Z., LI, D., XING, E. P., ZHANG, H., GONZALEZ, J. E. and STOICA, I. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena.   
ZHOU, H., SONG, L., CHEN, J., ZHOU, Y., WANG, G., YUAN, J. and ZHANG, Q. (2021). Rethinking soft labels for knowledge distillation: A bias-variance tradeoff perspective. arXiv preprint arXiv:2102.00650.   
ZHU, B., JIAO, J. and JORDAN, M. I. (2023a). Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. arXiv preprint arXiv:2301.11270.   
ZHU, B., SHARMA, H., FRUJERI, F. V., DONG, S., ZHU, C., JORDAN, M. I. and JIAO, J. (2023b). Fine-tuning language models with advantage-induced policy alignment. arXiv preprint arXiv:2306.02231.   
ZHU, Z., LIN, K., JAIN, A. K. and ZHOU, J. (2023c). Transfer learning in deep reinforcement learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.

# A Additional Related Works

We review key paradigms closely related to our formulation, especially KD; and point the reader to Weiss et al. (2016); Gou et al. (2021); Alyafeai et al. (2020); Zhu et al. (2023c); Wang and Yoon (2021); Tian et al. (2023) for numerous algorithms of knowledge transfer on modalities like sequences, images, graphs, etc. We restrict our survey to the single task of interest: classification. $^{10}$

# A.1 Inspiring Paradigms of Knowledge Transfer in Machine Learning

Paradigms of Knowledge Transfer. Knowledge transfer is not a patent of neural nets with softmax as their last layers. Similar ideas have realizations for ensemble of classification trees (Breiman and Shang, 1996) and even margin-based classifiers (Burges and Schölkopf, 1996). Since 2010s, there has been a line of work concentrating on the utilization of a trained teacher model and its original training set in a totally white-box manner with the purpose of student accuracy improvement (Hinton et al., 2015; Furlanello et al., 2018; Cho and Hariharan, 2019; Zhao et al., 2022; Romero et al., 2014; Yim et al., 2017; Huang and Wang, 2017; Park et al., 2019; Tian et al., 2019; Tung and Mori, 2019; Qiu et al., 2022). As the computation budget becomes relatively tight with respect to the scale of datasets, another line of works propose to avoid using of the full dataset for teacher pre-training during KD, and resort to architecture-specific metadata (Lopes et al., 2017), synthetic data (Nayak et al., 2019; Yin et al., 2020; Fang et al., 2022), or bootstrapping (Gu et al., 2023a) instead; which is dubbed the data-free approach. The sagacious vision that the teacher architecture may be agnostic or the teacher (log-)probability output may at least go through certain censorship does not receive enough attention from the community in the era of open sourcing (Orekondy et al., 2019; Wang et al., 2020; Wang, 2021; Nguyen et al., 2022). However, after the debut of closed-source and game-changing foundation models, practitioners find it plausibly nice to train their own models to purely mimic the response of these strong teachers. For example, though OpenAI only exposes transparent APIs of ChatGPT & GPT-4 (OpenAI, 2023b) to customers, there has been a line of efforts towards distilling black-box language models without even accessing the last-layer logits (Zheng et al., 2023; Wang et al., 2023b,a). Some primary results (Wang et al., 2023a) show that only letting LLaMA (Touvron et al., 2023a) mimic about 6000 carefully chosen trajectories generated by human-GPT-4 interactions can drastically boost the performance of these open-source autoregressive language models on common evaluation benchmarks (Gao et al., 2021; Li et al., 2023).

# A.2 Our Formulation versus Previous Paradigms

Motivated by the progress trend of knowledge transfer, we decouple the formulation of the input distribution $\rho$ from teacher training details and view the reference policy $\pi^{\star}$ as the gold standard (ground truth), beyond just a proxy of it. In the following two paragraphs, we would like to also emphasize the distinctions between our formulations and 1) imitation learning (whose performance is measured by reward sub-optimality), or 2) the LUPI framework (Vapnik et al., 2015; Lopez-Paz et al., 2015), respectively.

Optimality is not explicitly defined for $\pi^{\star}$ in our formulation, yet we solely want to mimic (the conditional density of) the teacher, so $\pi^{\star}$ is dubbed the reference policy. This view aligns with a recent belief that foundation models are by definition “good” and in effect black-box teachers,

judges, and raters of the student ones (Peng et al., 2023; Liu et al., 2023; Zheng et al., 2023). Generally speaking, optimal reward maximization is neither sufficient nor necessary for efficient knowledge transfer ( $\Leftrightarrow$ accurate behavioral imitation), because an optimal policy can differ from teacher demonstrations (Ng et al., 2000), which may even be sub-optimal itself (Brown et al., 2019; Chen et al., 2021), and pure behavioral imitation can result in constant sub-optimality (Gao et al., 2023; Zhu et al., 2023a).

Remark A.1. This work departs from Lopez-Paz et al. (2015, Section 4.1) essentially in several ways. First, our results hold for any legal data generating distribution $\rho \times \pi^{\star}$ and do not need to assume data→teacher, data→student, teacher→student transfer rates manually. Second, their result crucially hinges on the assumption that the teacher learns from data faster than the student, while we have clarified our formulation of $\pi^{\star}$ that differs completely. Finally, their transfer speed from the teacher to the student relies on the hardness of classification of each data point, e.g., the notion of linear separability (Shalev-Shwartz and Ben-David, 2014), yet we consider the product space of finite domains $S \times \mathcal{A}$ , which does not have these issues.

# B Missing Notation, Definitions, and Derivations

Additional Notation in Appendix. For any event $\mathcal{I}$ , $\mathcal{I}^c$ denotes its complementary event. $[K] := \{1, \ldots, K\}$ and $[\overline{K}] := \{0, \ldots, K - 1\}$ for any positive integer $K$ . We set $(S, A) = (|\mathcal{S}|, |\mathcal{A}|)$ and index the input and label spaces by $(\mathcal{S}, \mathcal{A}) = (\overline{[S]}, [A])$ when necessary. We also use a general notion of Dirac distribution in proving the lower bounds: given any measurable space $(\mathcal{Z}, \Sigma)$ , we define $\mathrm{Dirac}(\mathcal{Z}, z)(D) := \mathbb{1}\{z \in D\}, \forall D \in \Sigma$ . We denote by log the natural logarithm and adopt the convention $0 \log 0 = 0$ .

Definition B.1. For n i.i.d. samples $X^{n}$ drawn from a distribution $\nu$ over an alphabet X, the number of occurrences of x is denoted by $\mathfrak{n}_{x}(X^{n}) := \sum_{i=1}^{n} \mathbb{1}\{X_{i} = x\}$ , upon which we further measure the portion of X never observed in $X^{n}$ by the missing mass

$$
\mathfrak {m} _ {0} \left(\nu , X ^ {n}\right) := \sum_ {x \in \mathcal {S}} \nu (x)   \mathbb {1} \{\mathfrak {n} _ {x} \left(X ^ {n}\right) = 0 \}. \tag {B.1}
$$

It is worth mentioning that, $\mathcal{D},\mathcal{S}(\mathcal{D}),\mathcal{A}(\mathcal{D})$ , and $\mathcal{A}(\mathcal{D},s),\forall s\in \mathcal{S}$ are treated as multisets when fed into functionals like $\mathfrak{m}_0(\nu ,\cdot),\mathfrak{n}_x(\cdot)$ , or the cardinality operator $|\cdot |,$ for example, $|\mathcal{A}(\mathcal{D},s)| = \mathfrak{n}_s(\mathcal{S}(\mathcal{D}))$ ; while in set operations like $\mathcal{S}\backslash \mathcal{S}(\mathcal{D})$ or under the summation sign like $\sum_{s\in \mathcal{S}(\mathcal{D})}$ , where they act as ranges of enumeration, we slightly abuse the notations for simplicity to refer to their deduplicated counterparts.

Definition B.2. All the $\mathfrak{n}_s(S(\mathcal{D}))$ labels in $\mathcal{D}$ mapped from $s_i = s$ form a multiset

$$
\mathcal {A} (\mathcal {D}, s) := \left\{a \in \mathcal {A}: \text { for   all } (x, a) \in \mathcal {D} \text { s.t. } x = s \right\}.
$$

# B.1 Maximum Likelihood Estimation

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} \in \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} \mathsf {C E} _ {\mathsf {s g l}} (\mathcal {D})
$$

$$
\begin{array}{l} = \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} \sum_ {i = 1} ^ {n} \mathrm{CE} _ {\mathsf {s g l}} (s _ {i}, a _ {i}; \pi) = \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} - \sum_ {s \in \mathcal {S}} \sum_ {a \in \mathcal {A}} \mathfrak {n} _ {(s, a)} (\mathcal {D}) \log (a | s) \\ = \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} - \sum_ {s \in \mathcal {S} (\mathcal {D})} \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \underbrace {\sum_ {a \in \mathcal {A}} \frac {\mathfrak {n} _ {(s , a)} (\mathcal {D})}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))} \log \pi (a | s)} _ {- C E _ {\text {ful}} \left(\mathfrak {n} _ {(s, \cdot)} (\mathcal {D}) / \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \| \pi (\cdot | s)\right)}. \tag {B.2} \\ \end{array}
$$

Noticing that (B.2) is the summation of the cross-entropy between $\mathfrak{n}_{(s,\cdot)}(\mathcal{D})/\mathfrak{n}_{s}(\mathcal{S}(\mathcal{D}))$ and $\pi(\cdot|s)$ weighted by $\mathfrak{n}_{s}\left(\mathcal{S}(\mathcal{D})\right)$ over $\mathcal{S}(\mathcal{D})$ , we figure out the explicit solution of $\widehat{\pi}_{CE,sgl}$ as

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (a | s) \left\{ \begin{array}{l l} = \mathfrak {n} _ {(s, a)} (\mathcal {D}) / \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})), & s \in \mathcal {S} (\mathcal {D}), \\ \in \Delta (\mathcal {A}) \text {arbitrarily,} & \text {otherwise.} \end{array} \right.
$$

# B.2 Empirical Cross-Entropy Loss

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {p t l}} \in \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}}   \mathsf {C E} _ {\mathsf {p t l}} (\mathcal {D}, \mathcal {R}) = \underset {\pi \in \Delta (\mathcal {A} | \mathcal {S})} {\operatorname{argmin}} - \sum_ {s \in \mathcal {S} (\mathcal {D})} Z _ {s} \sum_ {a \in \mathcal {A}} \frac {\mathfrak {n} _ {(s , a)} (\mathcal {D}) \pi^ {\star} (a | s)}{Z _ {s}} \log \pi (a | s),
$$

where $Z_{s} := \sum_{a \in \mathcal{A}} \mathfrak{n}_{(s,a)}(\mathcal{D})\pi^{\star}(a|s)$ . Therefore, by the same cross-entropy minimization argument, the explicit solution is

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {p t l}} (a | s) \left\{ \begin{array}{l l} = \left\{ \begin{array}{l l} \mathfrak {n} _ {(s, a)} (\mathcal {D}) \pi^ {\star} (a | s) / Z _ {s}, & (s, a) \in \mathcal {D}, \\ 0, & s \in \mathcal {S} (\mathcal {D}), a \notin \mathcal {A} (\mathcal {D}, s), \end{array} \right. \\ \in \Delta (\mathcal {A}) \text {arbitrarily,} & s \notin \mathcal {S} (\mathcal {D}). \end{array} \right. \propto \mathfrak {n} _ {(s, a)} (\mathcal {D}) \pi^ {\star} (a | s);
$$

# C Information-Theoretic Arguments

# C.1 Additional Notation in Appendix C

We write $\mathsf{KL}(\pi\|\pi'|\lambda):=\mathbb{E}_{x\sim\lambda}\left[\mathsf{KL}\left(\pi(\cdot|x)\|\pi'(\cdot|x)\right)\right]$ for $\lambda\in\Delta(\mathcal{S})$ and $\pi,\pi'\in\Delta(\mathcal{A}|\mathcal{S})$ likewise for alphabets S, A to have notations concise. The values of $\triangle$ in the proofs of Theorem 3.1, Theorem 4.1, and Theorem 5.1 are different under the same notation. A similar logic applies to the values of $\pi_{\tau}$ in the proofs of Theorem 3.1 and Theorem 4.1.

# C.2 Proof of Theorem 3.1

Proof. We fix $\rho$ to be $\operatorname{Uniform}(\mathcal{S})$ and define a loss function over $\Delta(\mathcal{A}|\mathcal{S}) \times \Delta(\mathcal{A}|\mathcal{S})$ as

$$
l (\pi , \pi^ {\prime}) := \mathsf {T V} (\pi , \pi^ {\prime} | \rho) = \frac {1}{S} \sum_ {s = 0} ^ {S - 1} \mathsf {T V} \left(\pi (\cdot | s), \pi^ {\prime} (\cdot | s)\right). \tag {C.1}
$$

Obviously, $(\Delta(\mathcal{A}|\mathcal{S}), l)$ becomes a metric space. Without loss of generality, suppose A is even, we decompose l as

$$
d _ {s A / 2 + j} (\pi , \pi^ {\prime}) := l _ {s, j} (\pi , \pi^ {\prime}) := \frac {| \pi (2 j - 1 | s) - \pi^ {\prime} (2 j - 1 | s) | + | \pi (2 j | s) - \pi^ {\prime} (2 j | s) |}{2 S}, \tag {C.2}
$$

$$
l (\pi , \pi^ {\prime}) = \sum_ {s = 0} ^ {S - 1} \sum_ {j = 1} ^ {A / 2} l _ {s, j} (\pi , \pi^ {\prime}) = \sum_ {i = 1} ^ {S A / 2} d _ {i} (\pi , \pi^ {\prime}). \tag {C.3}
$$

Inspired by Paninski's construction (Paninski, 2008), we define $\pi_{\tau}$ as

$$
\pi_ {\tau} (2 j - 1 | s) = \frac {1 + \tau_ {s A / 2 + j} \triangle}{A}, \pi_ {\tau} (2 j | s) = \frac {1 - \tau_ {s A / 2 + j} \triangle}{A}, \forall (s, j) \in \overline {{[ S ]}} \times \left[ \frac {A}{2} \right]; \qquad (\mathrm{C.4})
$$

where $\tau \in \{-1, + 1\}^{AS / 2}$ and $\triangle$ is to be specified later. For any $\tau \sim_i\tau '$ , i.e., any pair in $\{-1, + 1\}^{AS / 2}$ that differs only in the $i$ -th coordinate, the construction of $\pi_{\tau}$ leads to

$$
d _ {i} (\pi_ {\tau}, \pi_ {\tau^ {\prime}}) = \frac {2 \triangle}{S A}. \tag {C.5}
$$

We thereby refer $\tau \sim \tau'$ to any pair in $\{-1, + 1\}^{AS / 2}$ that differs only in one coordinate and obtain

$$
\begin{array}{l} \text{LHS of (3.1)}\geq \inf_{\widehat{\pi}\in \widehat{\Pi} (\mathcal{D})}\sup_{\substack{\tau \in \{-1, + 1\}^{AS / 2}\\ \rho = \text{Uniform} (\mathcal{S})}}\mathbb{E}_{(\rho \times \pi_{\tau})^{n}}l(\widehat{\pi},\pi_{\tau}) \\ \geq \frac {A S}{2} \cdot \frac {2 \triangle / (S A)}{2} \min _ {\tau \sim \tau^ {\prime}} (1 - \mathsf {T V} ((\rho \times \pi_ {\tau}) ^ {n}, (\rho \times \pi_ {\tau^ {\prime}}) ^ {n})) \\ \geq \frac {\triangle}{4} \min _ {\tau \sim \tau^ {\prime}} \exp \left(- K L \left((\rho \times \pi_ {\tau}) ^ {n} \| (\rho \times \pi_ {\tau^ {\prime}}) ^ {n}\right)\right) \\ = \frac {\triangle}{4} \min _ {\tau \sim \tau^ {\prime}} \exp \left(- n \mathsf {K L} \left(\rho \times \pi_ {\tau} \| \rho \times \pi_ {\tau^ {\prime}}\right)\right) = \frac {\triangle}{4} \min _ {\tau \sim \tau^ {\prime}} \exp \left(- n \mathsf {K L} (\pi_ {\tau} \| \pi_ {\tau^ {\prime}} | \rho)\right) \\ = \frac {\triangle}{4} \exp \left(- \frac {n}{S A} \cdot 2 \triangle \log \frac {1 + \triangle}{1 - \triangle}\right) \\ \geq \frac {\triangle}{4} \exp \left(- 8 \frac {n}{S A} \triangle^ {2}\right), \tag {C.6} \\ \end{array}
$$

where

- the second inequality is by Assouad's lemma (Yu, 1997, Lemma 2),   
- the third inequality holds due to a variant (Lemma E.2) of the Bretagnolle–Huber inequality,   
- the first equality holds due to the decomposable property of KL (Tsybakov, 2009, Section 2.4),   
- the second equality follows from a basic property of $f$ -divergence (Polyanskiy and Wu, 2022, Proposition 7.2.4),   
- the last equality is by the definition of $\pi_{\tau}$ in (C.4) and $\rho = \text{Uniform}(\mathcal{S})$ ,   
- the last inequality derives from $\log (1 + x)\leq x,x > 0$ and an additional constraint $\triangle \leq 0.5$ we impose.

The assignment $\triangle = 0.25\sqrt{SA / n}$ with $n\geq SA / 4$ is a feasible choice for inequality (C.6) to hold, whose RHS further equals to

$$
\frac {\exp (- 0 . 5)}{1 6} \sqrt {\frac {S A}{n}}.
$$

□

# C.3 Proof of Theorem 4.1

Proof. Without loss of generality, we assume A > 2 is odd, then for a fixed $\triangle := 1$ , which does NOT vary with S, A, or n in THIS proof, we define $\Pi := \left\{\pi_{\tau} : \tau \in \{-1, +1\}^{AS/2}\right\}$ , where

$$
\left. \begin{array}{c} \pi_ {\tau} (2 j - 1 | s) := \frac {S \left(1 + \tau_ {s A / 2 + j} \triangle\right)}{2 (n + 1)}, \\ \pi_ {\tau} (2 j | s) := \frac {S \left(1 - \tau_ {s A / 2 + j} \triangle\right)}{2 (n + 1)}; \\ \pi_ {\tau} (A | s) := 1 - \frac {S}{2} \cdot \frac {A - 1}{n + 1}. \end{array} \right\} (s, j) \in \overline {{[ S ]}} \times \left[ \frac {A - 1}{2} \right]. \tag {C.7}
$$

To get a lower bound on a Bayes risk, we design a prior over P as

$$
\Lambda := \operatorname{Dirac} (\Delta (\mathcal {S}), \operatorname{Uniform} (\mathcal {S})) \times \Gamma , \tag {C.8}
$$

where

$$
\Gamma := \operatorname{Uniform} (\Pi).
$$

Intuitively speaking, for any $\rho \times \pi$ sampled from $\Lambda$ , $\rho$ must be uniform over $S$ and $\pi$ must be some $\pi_{\tau}$ with $\tau$ uniformly distributed over $\{-1, +1\}^{SA/2}$ . Therefore, if we let $\Lambda(\mathcal{D}, \mathcal{R})$ be the corresponding posterior over $\mathcal{P}$ conditioned on $(\mathcal{D}, \mathcal{R})$ and $\Gamma(\mathcal{D}, \mathcal{R})$ be the marginal posterior over $\Delta(\mathcal{A}|S)$ , we can by the definition of $\Gamma$ and $\pi_{\tau}$ obtain $\Lambda(\mathcal{D}, \mathcal{R}) = \text{Dirac}(\Delta(S), \text{Uniform}(S)) \times \Gamma(\mathcal{D}, \mathcal{R})$ where for any $(s, a) \in \{0, \ldots, S - 1\} \times \{1, \ldots, A - 1\}$ and $\pi \sim \Gamma(\mathcal{D}, \mathcal{R})$ , by the Bayes rule,

$$
\left\{ \begin{array}{l l} & \Gamma (\mathcal {D}, \mathcal {R}) \left[ \pi (a | s) = \pi^ {\star} (a | s) \right] = 1, (s, a) \in \mathcal {D} \text {or} (s, \mathrm{Buddy} (a)) \in \mathcal {D}; \\ & \Gamma (\mathcal {D}, \mathcal {R}) \left[ \pi (a | s) = \frac {S (1 + \triangle)}{2 (n + 1)} \right] = \Gamma (\mathcal {D}, \mathcal {R}) \left[ \pi (a | s) = \frac {S (1 - \triangle)}{2 (n + 1)} \right] = \frac {1}{2}, \text {otherwise}; \end{array} \right. \tag {C.9}
$$

where (recall that A > 2 is assumed to be odd) we define the “Buddy” for $a \in [A - 1] = \{1, \ldots, A - 1\}$ as

$$
\text { Buddy } (a) := \left\{ \begin{array}{l l} a - 1, & a \text {   is   even }; \\ a + 1, & a \text {   is   odd }. \end{array} \right. \tag {C.10}
$$

The intuition behind (C.9) is that if $\pi\sim\Gamma(\mathcal{D},\mathcal{R})$ , the marginal posterior of $\pi(a|s)$ for any seen $(s,a)$ in D must be a Dirac concentrated at $\pi^{\star}(a|s)$ and by the design of $\Pi=\{\pi_{\tau}\}$ , $\pi(a|s)$ is also determined if $(s,\text{Buddy}(a))\in\mathcal{D}$ . Note that the last label, A, is designed to be ignored, which we do not consider in both (C.9) and the following argument driven by Fubini's theorem. We define a event $\mathcal{E}_{\mathcal{D}}(s,a)$ for every $(s,a)$ with a<A as

$$
\mathcal {E} _ {\mathcal {D}} (s, a) = (s, a) \in \mathcal {D} \text {   or   } (s, \text { Buddy } (a)) \in \mathcal {D}.
$$

Next, we can apply Fubini's theorem to the Bayes risk with $\Lambda$ as its prior.

$$
\mathbb {E} _ {\rho \times \pi^ {\star} \sim \Lambda} \left[ \mathbb {E} _ {(\rho \times \pi^ {\star}) ^ {n}} \mathsf {T V} (\widehat {\pi}, \pi^ {\star} | \rho) \right]
$$

$$
\begin{array}{l} = \frac {1}{S} \sum_ {s \in \mathcal {S}} \mathbb {E} _ {\pi^ {\star} \sim \Gamma} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \mathbb {E} _ {\pi \sim \Gamma (\mathcal {D}, \mathcal {Q})} \operatorname{TV} \left(\widehat {\pi} (\cdot | s), \pi (\cdot | s)\right) \\ \geq \frac {1}{2 S} \sum_ {s \in \mathcal {S}} \mathbb {E} _ {\pi^ {\star} \sim \Gamma} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \sum_ {a <   A} \mathbb {E} _ {\pi \sim \Gamma (\mathcal {D}, \mathcal {Q})} | \widehat {\pi} (a | s) - \pi (a | s) | \\ = \frac {1}{2 S} \sum_ {s \in \mathcal {S}} \mathbb {E} _ {\pi^ {\star} \sim \Gamma} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \sum_ {a <   A} \left\{\right. \\ \mathbb {E} _ {\pi \sim \Gamma (\mathcal {D}, \mathcal {Q})} \left[ | \widehat {\pi} (a | s) - \pi (a | s) | \left| \mathcal {E} _ {\mathcal {D}} (s, a) \right] \mathbb {E} [ \mathbb {1} \{\mathcal {E} _ {\mathcal {D}} (s, a) \} | \mathcal {D} ] \right. \\ + \mathbb {E} _ {\pi \sim \Gamma (\mathcal {D}, \mathcal {Q})} \left[ | \widehat {\pi} (a | s) - \pi (a | s) |   | \mathcal {E} _ {\mathcal {D}} ^ {c} (s, a) \right] \mathbb {E} [ \mathbb {1} \{\mathcal {E} _ {\mathcal {D}} ^ {c} (s, a) \} | \mathcal {D} ] \\ \} \\ \geq \frac {1}{2 S} \sum_ {s \in \mathcal {S}} \mathbb {E} _ {\pi^ {\star} \sim \Gamma} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \sum_ {a <   A} \mathbb {E} _ {\pi \sim \Gamma (\mathcal {D}, \mathcal {Q})} \left[ | \widehat {\pi} (a | s) - \pi (a | s) | \mid \mathcal {E} _ {\mathcal {D}} ^ {c} (s, a) \right] \mathbb {E} [ \mathbb {1} \{\mathcal {E} _ {\mathcal {D}} ^ {c} (s, a) \} | \mathcal {D} ] \\ = \frac {1}{2 S} \sum_ {s \in \mathcal {S}} \mathbb {E} _ {\pi^ {\star} \sim \Gamma} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \sum_ {a <   A} \mathbb {E} [ \mathbb {1} \{\mathcal {E} _ {\mathcal {D}} ^ {c} (s, a) \} | \mathcal {D} ] \cdot \{ \\ \frac {1}{2} \left| \widehat {\pi} (a | s) - \frac {S (1 + \triangle)}{2 (n + 1)} \right| + \frac {1}{2} \left| \widehat {\pi} (a | s) - \frac {S (1 - \triangle)}{2 (n + 1)} \right| \\ \} \\ \geq \frac {1}{2 S} \sum_ {s \in \mathcal {S}} \mathbb {E} _ {\pi^ {\star} \sim \Gamma} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \sum_ {a <   A} \mathbb {E} [ \mathbb {1} \{\mathcal {E} _ {\mathcal {D}} ^ {c} (s, a) \} | \mathcal {D} ] \frac {1}{2} \cdot \frac {S}{n + 1} (C.11) \\ = \frac {1}{4 (n + 1)} \sum_ {s, a: a <   A} \mathbb {P} \{(s, a) \notin \mathcal {D} \text {   and   } (s, \text { Buddy } (a)) \notin \mathcal {D} \} (C.12) \\ = \frac {1}{4 (n + 1)} \sum_ {s, a: a <   A} \left(1 - \rho (s) \cdot \frac {S}{n + 1}\right) ^ {n} = \frac {1}{4 (n + 1)} \sum_ {s, a: a <   A} \left(1 - \frac {1}{S} \cdot \frac {S}{n + 1}\right) ^ {n} \\ = \frac {1}{4 (n + 1)} \sum_ {s, a: a <   A} \left(1 - \frac {1}{n + 1}\right) ^ {n} \\ \geq \frac {S (A - 1)}{4 e (n + 1)} \gtrsim \frac {S A}{n}, (C.13) \\ \end{array}
$$

where the inequality in (C.11) holds because of the triangle inequality and $\triangle = 1$ by design, and the equality in (C.12) holds because for any possible value of $\pi^{\star}$ a priori, i.e., any $\pi_{\tau}$ , $\mathbb{P}\{\mathcal{E}_{\mathcal{D}}^{c}(s,a)\}$ is always $(1 - \frac{1}{(n+1)})^{n}$ by the design of $\Pi = \{\pi_{\tau}\}$ ; the penultimate and the last equality hold due to the fact that the distribution of $\rho$ is always $\text{Dirac}(\Delta(\mathcal{S}), \text{Uniform}(\mathcal{S}))$ no matter whether a priori or a posteriori (conditioned on $(\mathcal{D}, \mathcal{R})$ ). Since the minimax risk is bounded from below by the worst-case Bayes risk (Polyanskiy and Wu, 2022, Theorem 28.1), the proof is completed. ☐

# C.4 Proof of Theorem 5.1

Proof. We assign $\xi = 1 / (n + 1)$ and design a $p\in \Delta (\mathcal{S})$ as $p(0) = 1 - S^{-1} / (n + 1)$ and $p = 1 / (n + 1)$ for all other inputs following Rajaraman et al. (2020, Figure 1 (b)). Then it suffices to get a lower bound for a Bayes risk given a prior over $\mathcal{P}$ , which we design as

$$
\Lambda_ {3} := \operatorname{Dirac} (\Delta (\mathcal {S}), p) \times \Gamma_ {3}, \tag {C.14}
$$

where

$$
\Gamma_ {3} := \operatorname{Uniform} (\Pi_ {\det}), \Pi_ {\det} := \left\{\pi \in \Delta (\mathcal {A} | \mathcal {S}): \pi (\cdot | s) \in \operatorname{Dirac} (\mathcal {A}), \forall s \in \mathcal {S} \right\}.
$$

Intuitively speaking, $\pi^{\star}$ is uniformly distributed over all deterministic policies, which indicates the marginal prior distribution of $\pi^{\star}(\cdot |s)$ for any $s\in S$ is

$$
\Gamma_ {3} \left[ \pi (\cdot | s) = \operatorname{Dirac} (\mathcal {A}, a) \right] = \frac {1}{A}, \forall a \in \mathcal {A}.
$$

We abbreviate the corresponding posterior of $\Lambda_3$ (resp. $\Gamma_3$ ) conditioned on $(\mathcal{D},\mathcal{Q})$ as $\Lambda_3(\mathcal{D},\mathcal{Q})$ (resp. $\Gamma_3(\mathcal{D},\mathcal{Q})$ ), which by definition implies $\Lambda_3(\mathcal{D},\mathcal{Q}) = \mathrm{Dirac}(\Delta (\mathcal{S}),p)\times \Gamma_3(\mathcal{D},\mathcal{Q})$ and by the Bayes formula implies that for any $s\in \mathcal{S}$ and $\pi \sim \Gamma_3(\mathcal{D},\mathcal{Q})$ ,

$$
\left\{ \begin{array}{l l} \Gamma_ {3} (\mathcal {D}, \mathcal {Q}) [ \pi (\cdot | s) = \pi^ {\star} (\cdot | s) ] & = 1, s \in \mathcal {S} (\mathcal {D}); \\ \Gamma_ {3} (\mathcal {D}, \mathcal {Q}) [ \pi (\cdot | s) = \operatorname{Dirac} (\mathcal {A}, a) ] & = 1 / A, s \in \mathcal {S} \backslash \mathcal {S} (\mathcal {D}). \end{array} \right. \tag {C.15}
$$

Without loss of generality, we assume $A > 1$ is even and then by Fubini's theorem,

$$
\begin{array}{l} \mathbb {E} _ {\rho \times \pi^ {\star} \sim \Lambda_ {3}} \left[ \mathbb {E} _ {(\rho \times \pi^ {\star}) ^ {n}} \mathsf {T V} (\widehat {\pi}, \pi^ {\star} | \rho) \right] \\ = \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {E} _ {\pi^ {\star} \sim \Gamma_ {3}} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \mathbb {E} _ {\pi \sim \Gamma_ {3} (\mathcal {D}, \mathcal {Q})} \mathsf {T V} (\widehat {\pi} (\cdot | s), \pi (\cdot | s)) \\ = \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {E} _ {\pi^ {\star} \sim \Gamma_ {3}} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \left\{\mathbb {E} _ {\pi \sim \Gamma_ {3} (\mathcal {D}, \mathcal {Q})} \left[ \mathrm{TV} (\widehat {\pi} (\cdot | s), \pi (\cdot | s)) \mid s \in \mathcal {S} (\mathcal {D}) \right] \mathbb {E} [ 1 (s \in \mathcal {S} (\mathcal {D})) | \mathcal {D} ] \right. \\ + \mathbb {E} _ {\pi \sim \Gamma_ {3} (\mathcal {D}, \mathcal {Q})} \left[ \mathrm{TV} \left(\widehat {\pi} (\cdot | s), \pi (\cdot | s)\right) \mid s \notin \mathcal {S} (\mathcal {D}) \right] \mathbb {E} \left[ \mathbb {1} \left(s \notin \mathcal {S} (\mathcal {D})\right) | \mathcal {D} \right] \rbrace \\ \geq \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {E} _ {\pi^ {\star} \sim \Gamma_ {3}} \mathbb {E} _ {\mathcal {D} \sim (\rho \times \pi^ {\star}) ^ {n}} \left\{\mathbb {E} \left[ \mathbb {1} \left(s \notin \mathcal {S} (\mathcal {D})\right) | \mathcal {D} \right] \mathbb {E} _ {\pi \sim \Gamma_ {3} (\mathcal {D}, \mathcal {Q})} \left[ \mathrm{TV} \left(\widehat {\pi} (\cdot | s), \pi (\cdot | s)\right) | s \notin \mathcal {S} (\mathcal {D}) \right] \right\} \\ = \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {E} _ {\pi^ {\star} \sim \Gamma_ {3}} \mathbb {E} _ {(\rho \times \pi^ {\star}) ^ {n}} \left\{\mathbb {E} \left[ \mathbb {1} \left(s \notin \mathcal {S} (\mathcal {D})\right) | \mathcal {D} \right] \cdot \frac {1}{A} \sum_ {a \in \mathcal {A}} \operatorname{TV} \left(\widehat {\pi} (\cdot | s), \operatorname{Dirac} (\mathcal {A}, a)\right) \right\}, \\ = \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {E} _ {\pi^ {\star} \sim \Gamma_ {3}} \mathbb {E} _ {(\rho \times \pi^ {\star}) ^ {n}} \left\{\right. \\ \mathbb {E} \left[ \mathbb {1} (s \notin \mathcal {S} (\mathcal {D})) | \mathcal {D} \right] \cdot \frac {1}{A} \sum_ {a = 0} ^ {A / 2 - 1} \left[ \operatorname{TV} \left(\widehat {\pi} (\cdot | s), \operatorname{Dirac} (\mathcal {A}, a)\right) + \operatorname{TV} \left(\widehat {\pi} (\cdot | s), \operatorname{Dirac} (\mathcal {A}, a + A / 2)\right) \right] \\ \} \\ \geq \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {P} (s \notin \mathcal {S} (\mathcal {D})) \frac {1}{A} \sum_ {a = 0} ^ {A / 2 - 1} \operatorname{TV} (\operatorname{Dirac} (\mathcal {A}, a), \operatorname{Dirac} (\mathcal {A}, a + A / 2)) \\ = \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {P} (s \notin \mathcal {S} (\mathcal {D})) \frac {1}{A} \cdot \frac {A}{2} = \frac {1}{2} \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {P} (s \notin \mathcal {S} (\mathcal {D})), \tag {C.16} \\ \end{array}
$$

where the last inequality holds due to the triangle inequality of TV. Therefore,

$$
\begin{array}{l} \text { LHS   of   (C.16) } \geq 0. 5 \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {P} \left(s \notin \mathcal {S} (\mathcal {D})\right) = 0. 5 \sum_ {s \in \mathcal {S}} \rho (s) \left(1 - \rho (s)\right) ^ {n} \\ \geq \frac {S - 1}{2 (n + 1)} \left(1 - \frac {1}{n + 1}\right) ^ {n} \geq \frac {S - 1}{2 e (n + 1)} \gtrsim \frac {S}{n}, \tag {C.17} \\ \end{array}
$$

where the second inequality is by only considering the S-1 inputs with mass $1/(n+1)$ . Since the minimax risk is bounded from below by the worst-case Bayes risk, the proof is completed. □

# D Arguments for Specific Learners

# D.1 Additional Definitions in Appendix D

In this section we denote the MLE of $\rho$ by

$$
\widehat {\rho} (\cdot) := \frac {\mathfrak {n} _ {(\cdot)} (\mathcal {S} (\mathcal {D}))}{n}. \tag {D.1}
$$

The event $B_{s,i}$ defined as follows will be used in the proofs of Theorem 3.2 and Theorem 4.4.

$$
B _ {s, i} := \left\{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) = i \right\}, \forall (s, i) \in \mathcal {S} \times \overline {{[ n + 1 ]}}. \tag {D.2}
$$

# D.2 Proof of Theorem 3.2

# D.2.1 Proof of the High-Probability Bound

Proof. For $|\mathcal{S}| > 1$ , we define

$$
u _ {s} := \mathfrak {n} _ {s} \left(\mathcal {S} (\mathcal {D})\right) \mathrm{TV} \left(\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (\cdot | s), \pi^ {\star} (\cdot | s)\right), \forall s \in \mathcal {S}.
$$

We decompose the LHS of $(3.4)$ as

$$
\begin{array}{l} \mathrm{LHS} = \sum_ {s \in \mathcal {S}} \left(\frac {\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}{n} + \rho (s) - \widehat {\rho} (s)\right) \mathrm{TV} \left(\widehat {\pi} _ {\mathrm{CE}, \mathrm{sgl}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) \\ \leq \sum_ {s \in \mathcal {S}} \frac {u _ {s}}{n} + \sum_ {s \in \mathcal {S}} | \rho (s) - \widehat {\rho} (s) |   \mathsf {T V} \left(\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) \\ \leq \underbrace {\frac {1}{n} \sum_ {s \in \mathcal {S}} u _ {s} + 2 \mathrm{TV} (\rho , \widehat {\rho})} _ {(i)}, \tag {D.3} \\ \end{array}
$$

where the first inequality is by triangle inequality and the second one holds due to the boundedness of TV. We define another two types of events to bound (i) in (D.3):

$$
D _ {s} := \left\{u _ {s} \leq \sqrt {\frac {\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}{2} \left(| \mathcal {A} | \log 2 + \log \frac {| \mathcal {S} | + 1}{\delta}\right)} \right\}, \forall s \in \mathcal {S}; \tag {D.4}
$$

$$
E := \left\{2 \mathrm{TV} (\rho , \widehat {\rho}) \leq \sqrt {\frac {2}{n} \left(| \mathcal {S} | \log 2 + \log \frac {| \mathcal {S} | + 1}{\delta}\right)} \right\}. \tag {D.5}
$$

Notice that $\mathbb{P}(D_s^c |B_{s,0}) = 0$ by the definition of $B_{s,i}$ in (D.2) and for any $i > 0$ , $\mathbb{P}(D_s^c |B_{s,i})\leq \delta /(|\mathcal{S}| + 1),\forall s\in \mathcal{S}$ by Lemma E.4; thus by the law of total probability,

$$
\mathbb {P} (D _ {s}) = \sum_ {i = 0} ^ {n} \mathbb {P} (D _ {s} | B _ {s, i}) \mathbb {P} (B _ {s, i}) \geq \left(1 - \frac {\delta}{| \mathcal {S} | + 1}\right) \sum_ {i = 0} ^ {n} \mathbb {P} (B _ {s, i}) = 1 - \frac {\delta}{| \mathcal {S} | + 1}, \forall s \in \mathcal {S}. \tag {D.6}
$$

Also noticing that $\mathbb{P}(E^c) \leq \delta / (|\mathcal{S}| + 1)$ by Lemma E.4, we apply a union bound over $E^c$ and $\{D_s^c\}_{s \in \mathcal{S}}$ for (i) in (D.3) to conclude that with probability at least $1 - \delta$ ,

$$
(i) \leq \sqrt {\frac {| \mathcal {A} | \log 2 + \log ((| \mathcal {S} | + 1) / \delta)}{2}} \cdot \underbrace {\frac {\sum_ {s \in \mathcal {S}} \sqrt {\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}}{n}} _ {\heartsuit} + \sqrt {\frac {2}{n} \left(| \mathcal {S} | \log 2 + \log \frac {| \mathcal {S} | + 1}{\delta}\right)}. \tag {D.7}
$$

By the Cauchy-Schwarz inequality,

$$
\heartsuit \text {   in   (D.7)   } \leq \frac {1}{n} \sqrt {| \mathcal {S} | \sum_ {s \in \mathcal {S}} \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))} = \sqrt {\frac {| \mathcal {S} |}{n}}. \tag {D.8}
$$

Substituting (D.8) back to the RHS of (D.7) yields the conclusion. The case of $|\mathcal{S}| = 1$ follows from Lemma E.4.

# D.2.2 Proof of the Worst-Case Upper Bound in Expectation

Proof. Taking expectation on both sides of (D.3) yields

$$
\mathbb {E} \left[ \mathrm{TV} (\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}}, \pi^ {\star} | \rho) \right] \leq \frac {1}{n} \sum_ {s \in \mathcal {S}} \mathbb {E} u _ {s} + \sqrt {\frac {| \mathcal {S} |}{n}}
$$

$$
= \frac {1}{n} \sum_ {s \in \mathcal {S}} \mathbb {E} \left[ \underbrace {\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \mathbb {E} [ \mathrm{TV} (\widehat {\pi} _ {\mathrm{CE} , \mathrm{sgl}} (\cdot | s) , \pi^ {\star} (\cdot | s)) | \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) ]} _ {=: \widetilde {u} _ {s}} \right] + \sqrt {\frac {| \mathcal {S} |}{n}}, \tag {D.9}
$$

where the inequality holds due to Lemma E.3. For every s, we trivially have

$$
\widetilde {u} _ {s} \leq \frac {\sqrt {| \mathcal {A} | \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}}{2} \tag {D.10}
$$

if conditioned on $B_{s,0}$ . If otherwise conditioned on $B_{s,0}^c$ , we still have

$$
\widetilde {u} _ {s} \leq \frac {\sqrt {| \mathcal {A} | \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}}{2}, \tag {D.11}
$$

where the inequality follows from Lemma E.3. Therefore, substituting (D.10) and (D.11) back to (D.9) gives

$$
\text {LHS of (D.9)} \leq \frac {\sqrt {| \mathcal {A} |}}{2 n} \sum_ {s \in \mathcal {S}} \mathbb {E} \sqrt {\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))} + \sqrt {\frac {| \mathcal {S} |}{n}} \leq \sqrt {\frac {| \mathcal {A} |}{2 n}} \sum_ {s \in \mathcal {S}} \sqrt {\rho (s)} + \sqrt {\frac {| \mathcal {S} |}{n}}
$$

$$
\lesssim \sqrt {\frac {| \mathcal {S} | | \mathcal {A} |}{n}},
$$

where the first inequality follows from the law of total expectation with respect to $B_{s,0}$ and $B_{s,0}^{c}$ , the second inequality follows from Jensen's inequality together with the definition of $\mathfrak{n}_s(\mathcal{S}(\mathcal{D}))$ , and the last inequality is by the Cauchy-Schwarz inequality.

# D.2.3 Proof of the Instance-Depedent Upper Bound in Expectation

Proof. We define the set of the numbers of occurrences of all inputs as

$$
N _ {\mathcal {S}} := \left\{\mathfrak {n} _ {s} \left(\mathcal {S} (\mathcal {D})\right): s \in \mathcal {S} \right\}. \tag {D.12}
$$

Then we decompose the LHS of $(3.5)$ as

$$
\begin{array}{l} \sum_ {s \in \mathcal {S}} \rho (s) \mathbb {E} \left[ \mathrm{TV} \left(\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) \right] \\ = \mathbb {E} [ \\ \sum_ {s \in \mathcal {S} (\mathcal {D})} \rho (s) \mathbb {E} \left[ \mathsf {T V} \left(\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) \bigg | N _ {\mathcal {S}} \right] \\ \begin{array}{l} + \sum_ {s \in \mathcal {S} \backslash \mathcal {S} (\mathcal {D})} \rho (s) \mathbb {E} \left[ \mathrm{TV} \left(\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (\cdot | s), \pi^ {\star} (\cdot | s)\right)   \bigg | N _ {\mathcal {S}} \right] \\ ] \end{array} \\ \leq \underbrace {\mathbb {E} \left[ \sum_ {s \in \mathcal {S} (\mathcal {D})} \rho (s) \mathbb {E} \left[ \mathrm{TV} \left(\widehat {\pi} _ {\mathrm{CE} , \mathrm{sgl}} (\cdot | s) , \pi^ {\star} (\cdot | s)\right) \mid \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \right] \right]} _ {I _ {1}} + \underbrace {\mathbb {E} \mathfrak {m} _ {0} (\rho , \mathcal {S} (\mathcal {D}))} _ {I _ {2}}, \tag {D.13} \\ \end{array}
$$

where the inequality is by the definition and boundedness of $\mathsf{TV}\left(\widehat{\pi}_{\mathsf{CE},\mathsf{sgl}}(\cdot|s),\pi^{\star}(\cdot|s)\right)$ . We divide $I_{1}$ and $I_{2}$ so at to conquer them as follows.

Bounding $I_{1}$ . For every $s \in \mathcal{S}$ , we define

$$
I _ {1} (s) := \mathbb {E} \left[ \mathrm{TV} \left(\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) \Bigg | \mathfrak {n} _ {s} \left(\mathcal {S} (\mathcal {D})\right) \right].
$$

Then we can bound $I_1(s)$ by Jensen's inequality for $s \in \mathcal{S}(\mathcal{D})$ :

$$
\begin{array}{l} I _ {1} (s) = \frac {1}{2} \sum_ {a \in \mathcal {A}} \mathbb {E} \left[ | \widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} (a | s) - \pi^ {\star} (a | s) |   \bigg |   \mathfrak {n} _ {s} \left(\mathcal {S} (\mathcal {D})\right) \right] \\ \leq \frac {1}{2} \sum_ {a \in \mathcal {A}} \sqrt {\mathbb {E} \left[ \frac {(\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \widehat {\pi} _ {\mathrm{CE,sgl}} (a | s) - \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \pi^ {\star} (a | s)) ^ {2}}{[ \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) ] ^ {2}} \bigg | \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \right]} \\ = \frac {1}{2} \sum_ {a \in \mathcal {A}} \sqrt {\frac {\pi^ {\star} (a | s) (1 - \pi^ {\star} (a | s))}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}}, \tag {D.14} \\ \end{array}
$$

where the last equality holds due to the observation that

$$
\mathfrak {n} _ {s} \left(\mathcal {S} (\mathcal {D})\right) \widehat {\pi} _ {\mathrm{CE}, \mathrm{sgl}} (a | s) \big | \mathfrak {n} _ {s} \left(\mathcal {S} (\mathcal {D})\right) \sim \text { Binomial } \left(\mathfrak {n} _ {s} \left(\mathcal {S} (\mathcal {D})\right), \pi^ {\star} (a | s)\right).
$$

Therefore, we can bound the summation inside the expectation of $I_{1}$ as

$$
\begin{array}{l} \sum_ {s \in \mathcal {S} (\mathcal {D})} \rho (s) I _ {1} (s) = \frac {1}{2} \sum_ {a \in \mathcal {A}} \sum_ {s \in \mathcal {S} (\mathcal {D})} \sqrt {\rho (s)} \sqrt {\rho (s) \frac {\pi^ {\star} (a | s) (1 - \pi^ {\star} (a | s))}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}} \\ \leq \frac {1}{\sqrt {2}} \sum_ {a \in \mathcal {A}} \sum_ {s \in \mathcal {S}} \sqrt {\rho (s)} \sqrt {\rho (s) \frac {\pi^ {\star} (a | s) (1 - \pi^ {\star} (a | s))}{1 + \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}} \\ \leq \frac {1}{\sqrt {2}} \sum_ {a \in \mathcal {A}} \sqrt {\sum_ {s \in \mathcal {S}} \rho (s) \frac {\pi^ {\star} (a | s) (1 - \pi^ {\star} (a | s))}{1 + \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}}, \tag {D.15} \\ \end{array}
$$

where the first inequality holds due to $\mathfrak{n}_{s}\left(\mathcal{S}(\mathcal{D})\right)\geq1,\forall s\in\mathcal{S}(\mathcal{D})$ and the last inequality is by the Cauchy-Schwarz inequality. Substituting (D.15) into $I_{1}=\mathbb{E}[\sum_{s\in\mathcal{S}(\mathcal{D})}\rho(s)I_{1}(s)]$ gives

$$
\begin{array}{l} I _ {1} \leq \frac {1}{\sqrt {2}} \sum_ {a \in \mathcal {A}} \sqrt {\sum_ {s \in \mathcal {S}} \pi^ {\star} (a | s) \left(1 - \pi^ {\star} (a | s)\right) \mathbb {E} \frac {\rho (s)}{1 + \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))}} \\ \leq \frac {1}{\sqrt {2}} \sum_ {a \in \mathcal {A}} \sqrt {\sum_ {s \in \mathcal {S}} \frac {\pi^ {\star} (a | s) (1 - \pi^ {\star} (a | s))}{n + 1}} \\ \leq \frac {1}{\sqrt {2 (n + 1)}} \sum_ {a \in \mathcal {A}} \sqrt {\sum_ {s \in \mathcal {S}} \min \left(\pi^ {\star} (a | s) , 1 - \pi^ {\star} (a | s)\right)} \\ \leq \frac {1}{\sqrt {2 (n + 1)}} \sqrt {| \mathcal {A} | \sum_ {a \in \mathcal {A}} \sum_ {s \in \mathcal {S}} \min \left(\pi^ {\star} (a | s) , 1 - \pi^ {\star} (a | s)\right)} \\ \leq \sqrt {\frac {| \mathcal {S} | | \mathcal {A} |}{2 (n + 1)}} \cdot \sqrt {\underbrace {\max _ {s \in \mathcal {S}} \sum_ {a \in \mathcal {A}} \min \left(\pi^ {\star} (a | s) , 1 - \pi^ {\star} (a | s)\right)} _ {\widetilde {\xi} (\pi^ {\star})}}, \tag {D.16} \\ \end{array}
$$

where the first inequality is by Jensen's inequality, the second inequality derives from Lemma E.8, and the penultimate inequality holds due to the Cauchy-Schwarz inequality. $\widetilde{\xi} (\pi^{\star})$ in (D.16) can be further bounded from above by

$$
\begin{array}{l} \max _ {s \in \mathcal {S}} \min _ {b \in \mathcal {A}} \left(1 - \pi^ {\star} (b | s) + \sum_ {a: a \neq b} \pi^ {\star} (a | s)\right) = \max _ {s \in \mathcal {S}} \min _ {b \in \mathcal {A}} \operatorname{TV} \left(\pi^ {\star} (\cdot | s), \operatorname{Dirac} (\mathcal {A}, b)\right) \\ = \max _ {s \in \mathcal {S}} \operatorname{dist} _ {\mathrm{TV}} \left(\pi^ {\star} (\cdot | s), \operatorname{Dirac} (\mathcal {A})\right) = \xi \left(\pi^ {\star}\right). \\ \end{array}
$$

To sum up, $I_1 \lesssim \sqrt{\xi(\pi^\star) |\mathcal{S}| |\mathcal{A}| n^{-1}}$ .

Bounding $I_{2}$ . Explicit calculation yields

$$
I _ {2} = \sum_ {s \in \mathcal {S}} \rho (s) (1 - \rho (s)) ^ {n} \leq \frac {4 | \mathcal {S} |}{9 n} \lesssim \frac {| \mathcal {S} |}{n},
$$

where the inequality follows from Lemma E.5.

![](images/f805166383aa8b21d2b7a14d7e31cb4b765ecd0acd2ecdd13c4959dfd01f1c1f.jpg)

# D.3 Proof of Lemma 4.2

Proof. Since $|\mathcal{S}| = 1$ , we omit the conditioning on $s \in \mathcal{S}$ for brevity in this proof. By Lemma E.1,

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {s g l}} \xrightarrow {a . s .} \pi^ {\star}.
$$

Therefore, applying the continuous mapping theorem (Durrett, 2019, Theorem 3.2.10) to (4.4) gives

$$
\widehat {\pi} _ {\mathsf {C E}, \mathsf {p t l}} (a) \xrightarrow {a . s .} \frac {[ \pi^ {\star} (a) ] ^ {2}}{\sum_ {b \in \mathcal {A}} [ \pi^ {\star} (b) ] ^ {2}}
$$

uniformly for every $a \in A$ .

![](images/930f93914c7ada87ae44663221bce3f5eeba6402e2d14efee0123b3426e02a75.jpg)

# D.4 Proof of Theorem 4.3

Proof. By (3.3) and (4.3), $\widehat{\pi}_{\mathsf{CE,sgl}}(\cdot |s)\propto \mathfrak{n}_s(\mathcal{S}(\mathcal{D}))$ and $\widehat{\pi}_{\mathsf{CE,ptl}}(\cdot |s)\propto \mathfrak{n}_s(\mathcal{S}(\mathcal{D}))\pi^{\star}(\cdot |s)$ for any $s\in$ $\mathcal{S}(\mathcal{D})$ . Therefore, the solution set of $\widehat{\pi}_{\mathsf{CE,ptl}}$ coincides with that of $\widehat{\pi}_{\mathsf{CE,sgl}}$ only if $\pi^{\star} = \mathrm{Uniform}(\mathcal{A})$ or $\pi^{\star}\in \mathrm{Dirac}(\mathcal{A})$ ; otherwise Lemma 4.2 implies that

$$
\liminf _ {n \to \infty} \mathsf {T V} (\widehat {\pi} _ {\mathsf {C E}, \mathsf {p t l}}, \pi^ {\star} | \rho) > 0 \text { almost   surely },
$$

and we thus rigorously justify the $\Omega(1)$ in Table 1 with probability one.

![](images/9396effb2c46e9b5c5f7ebb7c52dac97d1af033c765e952e0254911756cdbf0a.jpg)

# D.5 Proof of Theorem 4.4

# D.5.1 Proof of the High-Probability Bounds

Proof. For $|\mathcal{S}| > 1$ , we define

$$
v _ {s} := \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \mathrm{TV} \left(\widehat {\pi} _ {\text { SEL,ptl }} (\cdot | s), \pi^ {\star} (\cdot | s)\right)
$$

and decompose LHS of (4.8) into three terms as

$$
\mathrm{LHS} \leq \sum_ {s \in \mathcal {S}} \widehat {\rho} (s) \mathrm{TV} \left(\widehat {\pi} _ {\mathrm{SEL}, \mathrm{ptl}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) + \sum_ {s \in \mathcal {S}} | \rho (s) - \widehat {\rho} (s) | \mathrm{TV} \left(\widehat {\pi} _ {\mathrm{SEL}, \mathrm{ptl}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) \tag {D.17}
$$

$$
\leq \underbrace {\frac {1}{n} \sum_ {s \in \mathcal {S}} v _ {s} + \frac {1}{n} \sum_ {s \in \mathcal {S} (\mathcal {D})} \left| \frac {\rho (s)}{\widehat {\rho} (s)} - 1 \right| v _ {s} + \overbrace {\sum_ {s \in \mathcal {S} \backslash \mathcal {S} (\mathcal {D})} \rho (s)} ^ {\mathfrak {m} _ {0} (\rho , \mathcal {S} (\mathcal {D})) , \text { matches   Definition   B.1}}} _ {(i i)}, \tag {D.18}
$$

where $\widehat{\rho}$ is defined in (D.1) and the second inequality follows from $\widehat{\rho}(s)=0,\forall s\in\mathcal{S}\backslash\mathcal{S}(\mathcal{D})$ along with the boundedness of TV. We additionally define two kinds of events to bound (ii) in (D.18).

$$
\check {D} _ {s} = \left\{v _ {s} \leq \frac {4}{9} | \mathcal {A} | + 3 \sqrt {| \mathcal {A} | \log \frac {| \mathcal {S} | + 2}{\delta}} \right\}, \forall s \in \mathcal {S};
$$

$$
\check {E} = \left\{\mathfrak {m} _ {0} (\rho , \mathcal {S} (\mathcal {D})) \leq \frac {4 | \mathcal {S} |}{9 n} + \frac {3 \sqrt {| \mathcal {S} |}}{n} \log \frac {| \mathcal {S} | + 2}{\delta} \right\}.
$$

Recall Definition B.2 for $\mathcal{A}(\mathcal{D},s)$ , if $\mathfrak{n}_s(\mathcal{S}(\mathcal{D})) > 0$ , by (4.7),

$$
2 \mathrm{TV} \left(\widehat {\pi} _ {\mathrm{SEL}, \mathrm{ptl}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) = \sum_ {a \in \mathcal {A}} | \widehat {\pi} _ {\mathrm{SEL}, \mathrm{ptl}} (a | s) - \pi^ {\star} (a | s) |
$$

$$
= \sum_ {a \in \mathcal {A} \backslash \mathcal {A} (\mathcal {D}, s)} | \widehat {\pi} _ {\text { SEL,ptl }} (a | s) - \pi^ {\star} (a | s) | \leq 2 \mathfrak {m} _ {0} \left(\pi^ {\star} (\cdot | s), \mathcal {A} (\mathcal {D}, s)\right), \tag {D.19}
$$

where the inequality holds due to triangle inequality. Consequently,

$$
v _ {s} \leq \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \mathfrak {m} _ {0} \left(\pi^ {\star} (\cdot | s), \mathcal {A} (\mathcal {D}, s)\right), \forall s \in \mathcal {S} (\mathcal {D});
$$

to which we apply Lemma E.7 to obtain

$$
\mathbb {P} (\check {D} _ {s} ^ {c} | B _ {s, i}) \leq \frac {\delta}{| \mathcal {S} | + 2}, \forall i > 0;
$$

where $B_{s,i}$ is defined in (D.2). Also noticing that $\mathbb{P}(\check{D}_s^c | B_{s,0}) = 0$ by definition, we can control $\mathbb{P}(\check{D}_s)$ by

$$
\mathbb {P} (\check {D} _ {s}) = \sum_ {i} ^ {n} \mathbb {P} (\check {D} _ {s} | B _ {s, i}) \mathbb {P} (B _ {s, i}) \geq \left(1 - \frac {\delta}{| \mathcal {S} | + 2}\right) \sum_ {i} ^ {n} \mathbb {P} (B _ {s, i}) = 1 - \frac {\delta}{| \mathcal {S} | + 2}, \forall s \in \mathcal {S}.
$$

Since $\mathbb{P}(\check{E})\geq 1 - \delta /(|\mathcal{S}| + 2)$ by Lemma E.7, we apply a union bound over $\check{E}^c$ and $\{\check{D}_s^c\}_{s\in \mathcal{S}}$ for (ii) in (D.18) to conclude that with probability at least $1 - (|\mathcal{S}| + 1)\delta /(|\mathcal{S}| + 2)$ ,

$$
(i i) \leq \frac {4 | \mathcal {A} | + 3 \sqrt {| \mathcal {A} | \log ((| \mathcal {S} | + 2) / \delta)}}{9 n} \cdot \left(| \mathcal {S} | + \overbrace {\sum_ {s \in \mathcal {S} (\mathcal {D})} \underbrace {\left| \frac {\rho (s)}{\widehat {\rho} (s)} - 1 \right|} _ {=: o _ {s}}}\right) \tag {D.20}
$$

$$
+ \frac {4 | \mathcal {S} |}{9 n} + \frac {3 \sqrt {| \mathcal {S} |}}{n} \log \frac {| \mathcal {S} | + 2}{\delta}.
$$

We further decompose ▲ in (D.20) in a pragmatically tight enough way as

$$
\begin{array}{l} \blacktriangle \leq | \mathcal {S} | + \sum_ {s \in \mathcal {S} (\mathcal {D})} \frac {\rho (s)}{\widehat {\rho} (s)} = | \mathcal {S} | + \sum_ {s \in \mathcal {S} (\mathcal {D})} \frac {n \rho (s)}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))} \\ = | \mathcal {S} | + \sum_ {s \in \mathcal {S} (\mathcal {D})} \frac {2 n \rho (s)}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) + \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D}))} \leq | \mathcal {S} | + 2 \sum_ {s \in \mathcal {S} (\mathcal {D})} \frac {n \rho (s)}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) + 1} \\ \leq | \mathcal {S} | + 2 \sum_ {s \in \mathcal {S}} \frac {n \rho (s)}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) + 1} \\ = | \mathcal {S} | + \underbrace {\sum_ {s \in \bar {\mathcal {S}}} \frac {n \rho (s)}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) + 1}} _ {\tilde {\mathbf {A}}} + \overbrace {\sum_ {s \in \tilde {\mathcal {S}}} \underbrace {\frac {n \rho (s)}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) + 1}} _ {r _ {s}}} ^ {\tilde {\mathbf {A}}}, \tag {D.21} \\ \end{array}
$$

where $\bar{\mathcal{S}}$ and $\widetilde{\mathcal{S}}$ are defined as

$$
\bar {\mathcal {S}} := \left\{s \in \mathcal {S}: 0 <   \rho (s) <   \frac {\log \left(| \mathcal {S} | (| \mathcal {S} | + 2) / \delta\right)}{n} \cdot \frac {2 0 0}{9 9} \right\}, \tag {D.22}
$$

$$
\widetilde {\mathcal {S}} := \left\{s \in \mathcal {S}: \rho (s) \geq \frac {\log \left(| \mathcal {S} | (| \mathcal {S} | + 2) / \delta\right)}{n} \cdot \frac {2 0 0}{9 9} \right\}. \tag {D.23}
$$

By the definition of $\bar{\mathcal{S}}$ , all $s$ 's in $\bar{\mathcal{S}}$ have small enough $\rho(s)$ , and thus $\triangleleft$ in (D.21) can be trivially bounded from above, i.e.,

$$
\bar {\Delta} \leq \frac {2 0 0}{9 9} | \mathcal {S} | \log \frac {| \mathcal {S} | (| \mathcal {S} | + 2)}{\delta}. \tag {D.24}
$$

For each $s\in \widetilde{\mathcal{S}}$ , we define

$$
\eta_ {s} := \sqrt {\frac {2 \log (| \mathcal {S} | (| \mathcal {S} | + 2) / \delta)}{n \rho (s)}},
$$

then by the definition of $\widetilde{S}$ , $1 - \eta_{s} \geq 0.1$ . Therefore, noticing that $\mathfrak{n}_{s}\left(\mathcal{S}(\mathcal{D})\right) \sim \text{Binomial}\left(n, \rho(s)\right)$ , we can apply Corollary E.10 to each $r_{s}$ in (D.21) to conclude that for every $s \in \widetilde{S}$ , with probability at least $1 - \delta / (|\mathcal{S}|(|\mathcal{S}| + 2))$ ,

$$
\frac {r _ {s}}{n \rho (s)} \leq \frac {1}{(1 - \eta_ {s}) n \rho (s) + 1} \leq \frac {1}{0 . 1 n \rho (s) + 1} \leq \frac {1 0}{n \rho (s)}, \tag {D.25}
$$

which followed by a union bound argument within $\widetilde{\mathcal{S}}$ yields that with probability at least $1 - \delta / (|\mathcal{S}| + 2)$ ,

$$
\widetilde {\blacktriangle} \text {   in   (D.21)   } \leq 1 0 | \mathcal {S} |. \tag {D.26}
$$

We then denote by $\widetilde{E}$ the event conditioned on which (D.26) holds and denote by $\dot{E}$ the event conditioned on which (D.20) holds. A union bound argument over $\widetilde{E}^{c}$ and $\dot{E}^{c}$ shows that with probability at least $1 - \delta$ , the LHS of (4.8) is bounded from above by

$$
\frac {4 | \mathcal {A} | + 3 \sqrt {| \mathcal {A} | \log ((| \mathcal {S} | + 2) / \delta)}}{9 n} \cdot \left(1 2 | \mathcal {S} | + \frac {2 0 0}{9 9} | \mathcal {S} | \log \frac {| \mathcal {S} | (| \mathcal {S} | + 2)}{\delta}\right) + \frac {4 | \mathcal {S} |}{9 n} + \frac {3 \sqrt {| \mathcal {S} |}}{n} \log \frac {| \mathcal {S} | + 2}{\delta}.
$$

For $|\mathcal{S}| = 1$ , invoking Lemma E.7 for (D.19) to draw the conclusion.

![](images/be68eb5d64cae52c8f5ada71f42952b967744eaebc732b989f3bd99d71ce0164.jpg)

# D.5.2 Proof of the Upper Bound in Expectation

Proof. Substituting (D.19) into (D.17) yields an upper bound for $\mathbb{ETV}(\widehat{\pi}_{\mathrm{SEL,ptl}},\pi^{\star}|\rho)$ as

$$
\frac {1}{n} \sum_ {s \in \mathcal {S}} \mathbb {E} v _ {s} + \sum_ {s \in \mathcal {S}} \mathbb {E} \left[ \underbrace {| \rho (s) - \widehat {\rho} (s) | \mathbb {E} [ \mathfrak {m} _ {0} (\pi^ {\star} (\cdot | s) , \mathcal {A} (\mathcal {D} , s)) | \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) ]} _ {\widetilde {v} _ {s}} \right]. \tag {D.27}
$$

Each $\mathbb{E}v_{s}$ in (D.27) can be bounded from above via (D.19) by

$$
\mathbb {E} \left[ \underbrace {\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) \mathbb {E} [ \mathfrak {m} _ {0} (\pi^ {\star} (\cdot | s) , \mathcal {A} (\mathcal {D} , s)) | \mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) ]} _ {\tilde {v} _ {s}} \right]. \tag {D.28}
$$

Conditioned on $B_{s,0}^{c}$ , invoking Lemma E.5 to conclude

$$
\bar {v} _ {s} \leq \frac {4 | \mathcal {A} |}{9}. \tag {D.29}
$$

The above inequality trivially holds if otherwise conditioned on $B_{s,0}$ . Similarly, we always have

$$
\widetilde {v} _ {s} \leq \rho (s) \frac {| \mathcal {A} |}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) + 1} + \frac {4 | \mathcal {A} |}{9 n}. \tag {D.30}
$$

Substituting (D.29) back to (D.28) gives

$$
\frac {1}{n} \sum_ {s \in \mathcal {S}} \mathbb {E} v _ {s} \lesssim \frac {| \mathcal {S} | | \mathcal {A} |}{n}. \tag {D.31}
$$

By Lemma E.8, substituing (D.30) and (D.31) back to (D.27) yields

$$
\mathbb {E} \mathrm{TV} (\widehat {\pi} _ {\mathrm{SEL}, \mathrm{ptl}}, \pi^ {\star} | \rho) \lesssim \frac {| \mathcal {S} | | \mathcal {A} |}{n} + | \mathcal {A} | \sum_ {s \in \mathcal {S}} \mathbb {E} \frac {\rho (s)}{\mathfrak {n} _ {s} (\mathcal {S} (\mathcal {D})) + 1} \lesssim \frac {| \mathcal {S} | | \mathcal {A} |}{n}. \tag {D.32}
$$

![](images/59d255c36110144a3c130d770928b157e2e8cde93c809b17b88532cbc09f5543.jpg)

Remark D.1. The empirical variant of vanilla SEL in our second case actually match the log probability, which is by definition normalized. Analyzing an unnormalized version, which is more relevant to the matching the logits in practice (Ba and Caruana, 2014; Kim et al., 2021), in the second setting may call for new techniques. Also, some preliminary results on the empirical side manifest the difference between minimizing forward KL and reverse KL in scenarios related to our last setting (Jiang et al., 2019; Gu et al., 2023b; Agarwal et al., 2023), whose analysis are left are future work.

# D.6 Proof of Theorem 5.2

Proof. Since TV is bounded from above by 1 and TV $(\widehat{\pi}_{\mathsf{CE}_{\mathrm{ful}}}(\cdot |s),\pi^{\star}(\cdot |s)) = 0,\forall s\in \mathcal{S}(\mathcal{D}),$

$$
\mathrm{LHS} = \sum_ {s \in \mathcal {S} \backslash \mathcal {S} (\mathcal {D})} \rho (s) \mathrm{TV} \left(\widehat {\pi} _ {\mathsf {C E} _ {\text {ful}}} (\cdot | s), \pi^ {\star} (\cdot | s)\right) \leq \sum_ {s \in \mathcal {S}} \rho (s)   1 1 \{s \notin \mathcal {S} (\mathcal {D}) \} =: M. \tag {D.33}
$$

Noticing that M realizes Definition B.1 to $\mathfrak{m}_{0}\left(\rho,\mathcal{S}(\mathcal{D})\right)$ , we invoke Lemma E.7 to get

$$
M \leq \mathbb {E} M + \frac {3 \sqrt {| \mathcal {S} |} \log (1 / \delta)}{n} \leq \frac {4 | \mathcal {S} |}{9 n} + \frac {3 \sqrt {| \mathcal {S} |} \log (1 / \delta)}{n}. \tag {D.34}
$$

Substituting (D.34) back to (D.33) finishes the proof.

![](images/7a1c70dd607c558d8591c6beca9415b36b678c83426228f35db11a5632c8d2bb.jpg)

# E Auxiliary Lemmas

In contrast with other non-asymptotic tools below, we must assume the mass p does not vary with n in the asymptotic guarantee Lemma E.1.

Lemma E.1. Let $p$ be a probability mass function over an alphabet $S$ , whose empirical estimation from $X_1, \ldots, X_n \stackrel{i.i.d.}{\sim} p$ is $p_n(\cdot) := \sum_{i=1}^{n} 1\{X_i = \cdot\} / n$ , then

$$
p _ {n} \xrightarrow {a . s .} p,
$$

where the almost surely convergence is defined under the $\ell_{\infty}$ metric in $\mathbb{R}^{|\mathcal{S}|}$ .

Proof. Without loss of generality, we assume $\mathcal{S} = [|S|]$ ; thereby inducing $p(x) = F(x) - F(x - 1)$ for $x \in [|S|]$ , where $F(x) = \mathbb{P}(X \leq x)$ is the distribution function of $X \sim p$ . Similarly, $p_n(x) = F_n(x) - F_n(x - 1)$ for

$$
F _ {n} (\cdot) = \sum_ {i = 1} ^ {n} \frac {\mathbb {P} (X _ {i} \leq \cdot)}{n}.
$$

Therefore,

$$
\begin{array}{l} \max _ {x \in [ | \mathcal {S} | ]} | p _ {n} (x) - p (x) | \leq \sup _ {x \in \mathbb {R}} | F _ {n} (x) - F (x) - (F _ {n} (x - 1) - F (x - 1)) | \\ \leq 2 \sup _ {x \in \mathbb {R}} | F _ {n} (x) - F (x) | =: 2 \| F _ {n} - F \| _ {\infty}. \\ \end{array}
$$

The proof is thus completed by invoking the Glivenko-Cantelli Theorem (Van der Vaart, 2000, Theorem 19.1).

# E.1 Bounding TV from Above

Lemma E.2 (Bretagnolle–Huber inequality (Bretagnolle and Huber, 1979)). If P and Q are two probability measures on the same measurable space, then

$$
\operatorname{TV} (P, Q) \leq 1 - \frac {1}{2} \exp (- \mathrm{KL} (P \| Q)).
$$

Lemma E.3. If $a_1, \ldots, a_n \stackrel{i.i.d.}{\sim} \pi \in \Delta(\mathcal{A})$ , whose MLE is $\widehat{\pi} = \widehat{\pi}(a_1, \ldots, a_n)$ ; and $|\mathcal{A}| < \infty$ , then

$$
\mathbb {E} \mathrm{TV} (\widehat {\pi}, \pi) \leq \frac {1}{2} \sqrt {\frac {| \mathcal {A} |}{n}}.
$$

Proof. We reproduce the proof of this standard result here for completeness.

$$
\begin{array}{l} \mathrm{LHS} = \frac {1}{2} \sum_ {a \in \mathcal {A}} \mathbb {E} | \widehat {\pi} (a) - \pi (a) | \leq \frac {1}{2} \sum_ {a \in \mathcal {A}} \sqrt {\mathbb {E} (\widehat {\pi} (a) - \pi (a)) ^ {2}} \\ = \frac {1}{2} \sum_ {a \in \mathcal {A}} \sqrt {\frac {1}{n ^ {2}} \operatorname{Var} (n \widehat {\pi} (a))} = \frac {1}{2 \sqrt {n}} \sum_ {a \in \mathcal {A}} \sqrt {\pi (a) (1 - \pi (a))} \\ \leq \frac {1}{2 \sqrt {n}} \sum_ {a \in \mathcal {A}} \sqrt {\pi (a)} \leq \frac {1}{2} \sqrt {\frac {| \mathcal {A} |}{n}}, \\ \end{array}
$$

where the first inequality is by Jensen's inequality, the third equality holds due to $n\widehat{\pi}(a) \sim \text{Binomail}(n, \pi(a))$ , and the last inequality is by the Cauchy-Schwarz inequality.

Lemma E.4. Under the same setting as Lemma E.3, for any $\delta \in (0,1)$ , with probability at least $1 - \delta$ ,

$$
\mathsf {T V} (\widehat {\pi}, \pi) \leq \sqrt {\frac {| \mathcal {A} | \log 2 + \log (1 / \delta)}{2 n}}.
$$

Proof. This is a straightforward corollary of the Bretagnolle-Huber-Carol inequality (van der Vaart and Wellner, 1996, Proposition A.6.6) based on the relationship between TV and $\ell_1$ .

# E.2 Missing Mass Analysis

Observations like Lemma E.5 are key and common in the analysis of learning from finite and static datasets (Rajaraman et al., 2020; Rashidinejad et al., 2021).

Lemma E.5. For all $x \in [0,1], n > 0$ , $x(1 - x)^n \leq (4/9)n$ .

Proof. By taking the derivative w.r.t. $x$ ,

$$
\max _ {x \in [ 0, 1 ]} \mathrm{LHS} = \left(\frac {n}{n + 1}\right) ^ {n + 1} \frac {1}{n} \leq \frac {1}{n} \lim _ {n \rightarrow \infty} (1 - \frac {1}{n + 1}) ^ {n + 1} = \frac {1}{e n} \leq \frac {4}{9 n}.
$$

![](images/f7e3affc01718784057993c25918b1907601e932ecfd69ca914fc896d9b19101.jpg)

Lemma E.6 (Rajaraman et al. 2020, Theorem A.2). Given a distribution $\nu$ on an alphabet $S$ and $n$ i.i.d. samples $X^n \stackrel{i.i.d.}{\sim} \nu$ , then for any $\delta \in (0,1/10]$ , with probability at least $1 - \delta$ ,

$$
\mathfrak {m} _ {0} \left(\nu , X ^ {n}\right) - \mathbb {E} \left[ \mathfrak {m} _ {0} \left(\nu , X ^ {n}\right) \right] \leq \frac {3 \sqrt {| \mathcal {S} |} \log (1 / \delta)}{n}.
$$

Lemma E.7. Under the same setting as Lemma E.6, for any $\delta \in (0,1/10]$ , with probability at least $1 - \delta$ ,

$$
\mathfrak {m} _ {0} \left(\nu , X ^ {n}\right) \leq \frac {4 | \mathcal {S} |}{9 n} + \frac {3 \sqrt {| \mathcal {S} |} \log (1 / \delta)}{n}.
$$

Proof.

$$
\begin{array}{l} \mathbb {E} \mathfrak {m} _ {0} (\nu , X ^ {n}) = \sum_ {x \in \mathcal {S}} \nu (x) \mathbb {E} \mathbb {1} \{x \notin X ^ {n} \} = \sum_ {x \in \mathcal {S}} \nu (x) \mathbb {P} (x \notin X ^ {n}) \\ = \sum_ {x \in \mathcal {S}} \nu (x) (1 - \nu (x)) ^ {n} \leq \frac {4 | \mathcal {S} |}{9 n}, \tag {E.1} \\ \end{array}
$$

where the inequality holds due to Lemma E.5; we conclude that $\forall \delta \in (0,1/10]$ , with probability at least $1 - \delta$ ,

$$
\mathfrak {m} _ {0} \left(\nu , X ^ {n}\right) \leq \frac {4 | \mathcal {S} |}{9 n} + \frac {3 \sqrt {| \mathcal {S} |} \log (1 / \delta)}{n}
$$

by substituting (E.1) into Lemma E.6.

![](images/d7b60e6f5e4052e4f1ba618a70ce38c484eb6d46626782c4ee9796df269e51df.jpg)

# E.3 Upper Bounds for Binomial(n, p)

The following two bounds for $X \sim \operatorname{Binomial}(n, p)$ both follow from $\mathbb{E}[z^X] = (1 - p + pz)^n$ , $\forall z \in \mathbb{R}$ .

Lemma E.8. Let $X \sim \text{Binomial}(n, p)$ . If $p \in (0, 1]$ ,

$$
\mathbb {E} \frac {1}{X + 1} \leq \frac {1}{p (n + 1)}.
$$

Proof. This folklore (Canonne, 2020) derives from an observation that by Fubini's Theorem,

$$
\mathbb {E} \frac {1}{X + 1} = \int_ {0} ^ {1} \mathbb {E} [ z ^ {X} ] d z,
$$

whose RHS is

$$
\int_ {0} ^ {1} (1 - p + p z) ^ {n} d z = \frac {(1 - p + p z) ^ {n + 1}}{p (n + 1)} \bigg | _ {0} ^ {1} = \frac {1 - (1 - p) ^ {n + 1}}{p (n + 1)} \leq \frac {1}{p (n + 1)}.
$$

![](images/7c1707de38a51e4c39715b0e74a597902f62b7809af653e01e1afb9d829139d2.jpg)

Lemma E.9. Let $X \sim \text{Binomial}(n, p)$ . For any $\eta \in (0,1)$ ,

$$
\mathbb {P} (X \leq (1 - \eta) n p) \leq \exp \left(- \frac {\eta^ {2} n p}{2}\right).
$$

Proof. A combination of Mitzenmacher and Upfal (2017, Exercise 4.7) and the proof of Mitzenmacher and Upfal (2017, Theorem 4.5) yields the upper bound, which we provide here for completeness. For any $t < 0$ , by Markov's inequality,

$$
\begin{array}{l} \mathbb {P} (X \leq (1 - \eta) n p) = \mathbb {P} \left(e ^ {t X} \leq e ^ {t (1 - \eta) n p}\right) \leq \frac {\mathbb {E} [ e ^ {t X} ]}{e ^ {t (1 - \eta) n p}} = \frac {(1 + p (e ^ {t} - 1)) ^ {n}}{e ^ {t (1 - \eta) n p}} \\ \leq \frac {\exp (n p (e ^ {t} - 1))}{e ^ {t (1 - \eta) n p}} = \left(\frac {\exp (e ^ {t} - 1)}{e ^ {t (1 - \eta)}}\right) ^ {n p} = \left(\frac {e ^ {- \eta}}{(1 - \eta) ^ {1 - \eta}}\right) ^ {n p}, \\ \end{array}
$$

where the last inequality follows from $1 + x \leq e^x$ , $\forall x \in \mathbb{R}$ and in the last equality we set $t = \log(1 - \eta)$ . It remains to show

$$
- \eta - (1 - \eta) \log (1 - \eta) \leq - \frac {\eta^ {2}}{2}, \forall \eta \in (0, 1). \tag {E.2}
$$

We thereby define $f(\eta) := -\eta - (1 - \eta) \log(1 - \eta) + 0.5\eta^{2}$ . A direct calculation gives

$$
\begin{array}{l} f ^ {\prime} (\eta) = \log (1 - \eta) + \eta , f ^ {\prime} (0) = 0; \\ f ^ {\prime \prime} (\eta) = - \frac {1}{1 - \eta} + 1 <   0, \forall \eta \in (0, 1); \\ \end{array}
$$

So $f$ is nonincreasing in [0, 1) and thus (E.2) holds.

![](images/937e9d7e815d8c46fa409c84556dc46ef409430cc7ebc084dd4d1852a474c358.jpg)

Lemma E.9 helps us obtain a high-probability counterpart of Lemma E.8.

Corollary E.10. Let $X \sim \operatorname{Binomial}(n, p)$ and $p > 0$ . For any $\delta \in (0,1)$ , if

$$
\eta = \sqrt {\frac {2 \log (1 / \delta)}{n p}} <   1,
$$

then with probability at least $1 - \delta$ ,

$$
{\frac {1}{X + 1}} \leq {\frac {1}{(1 - \eta) n p + 1}}.
$$

Proof. By Lemma E.9,

$$
\mathbb {P} \left(\frac {1}{X + 1} > \frac {1}{(1 - \eta) n p + 1}\right) \leq \mathbb {P} (X \leq (1 - \eta) n p) \leq \exp \left(- \frac {\eta^ {2} n p}{2}\right) = \delta .
$$

![](images/6f8d24e51d32e7b25f6b17775cfb244fa791c8f848411de89214f3f7afed6931.jpg)

# F Hard-to-Learn Instances for Experiments

Instance 0 For every $s \in S$ , $\pi^{\star}(\cdot | s) := 0.5\text{Uniform}(\mathcal{A}) + 0.5\text{Dirac}(\mathcal{A}, s \mod A + 1)$ because any reference policy far away from both $\text{Uniform}(\mathcal{A})$ and any one in $\text{Dirac}(\mathcal{A})$ is sufficient to reveal the disadvantage of $\widehat{\pi}_{\text{CE,ptl}}$ according to Theorem 4.3. $\rho := \text{Uniform}(\mathcal{S})$ is enough to ensure about $n / s$ visitations of each input.

Interestingly, we conjecture there does not exist a worst-of-three-worlds instance that can simultaneously expose the fundamental limits of $\widehat{\pi}_{CE,sgl}$ , $\widehat{\pi}_{SEL,ptl}$ , and $\widehat{\pi}_{CE_{ful}}$ , in that the constructive proofs (in Appendix C) of Theorem 3.1, Theorem 4.1, and Theorem 5.1 since the progressively richer information are substantially different from each other. Since our learners in this section is uniformly initialized over unseen labels, any single instance covered by the Bayes prior in the lower bound arguments of a setting $^{13}$ is sufficient to numerically illustrate the corresponding difficulty of estimation (learning).

Instance 1 To verify the minimax optimality of $\widehat{\pi}_{\mathrm{CE,sgl}}$ with only samples available, we adapt the proof of Theorem 3.1 (Appendix C.2), which is based on Assouad's hypercube reduction (Yu, 1997). In numerical simulations, any vertex of the hypercube is applicable since we have already enforced an uniform initialization of any $\widehat{\pi}$ in unseen inputs. We choose the teacher policy

$$
\pi^ {\star} (2 j - 1 | s) = \frac {1 + 0 . 2 5 \sqrt {S A / n}}{A}, \pi^ {\star} (2 j | s) = \frac {1 - 0 . 2 5 \sqrt {S A / n}}{A}, \forall (s, j) \in [ \overline {{S}} ] \times \left[ \frac {A}{2} \right];
$$

and $\rho=\operatorname{Uniform}(\mathcal{S})$ for simplicity. The two key insights behind the design of Instance 1 are (1) $\rho$ must be nonvanishing in $\Omega(|\mathcal{S}|)$ inputs to manifest the hardness of $|S|>0$ , (2) $\operatorname{TV}\left(\pi^{\star}(\cdot|s),\operatorname{Uniform}(\mathcal{A})\right)=\Theta(n^{-0.5})$ is crucial for Instance 1 to be hard enough for any minimax optimal learner. (If $|A|$ is odd, simply let the last label A to have zero mass and replace A with A-1 here.)

Instance 2 To verify the minimax optimality of $\widehat{\pi}_{SEL,ptl}$ with sampled odds available, we adapt the proof of Theorem 4.1 (Appendix C.3), which is based on a carefully designd Bayes prior. Similarly, we can use any single instance covered by the support of the Bayes prior. We choose the teacher policy

$$
\pi^ {\star} (2 j - 1 | s) = \frac {S}{n + 1}, \pi^ {\star} (2 j | s) = 0, \pi^ {\star} (A | s) = 1 - \frac {S}{2} \cdot \frac {A - 1}{n + 1}, \forall (s, j) \in \overline {{[ S ]}} \times \left[ \frac {A - 1}{2} \right];
$$

and $\rho=\operatorname{Uniform}(\mathcal{S})$ for simplicity. (If $|A|$ is even, simply let the last label A to have zero mass and replace A with A-1 here.)

Instance 3 To verify the minimax optimality of $\widehat{\pi}_{\mathsf{CE}_{\mathsf{ful}}}$ with complete logits available, we adapt the proof of Theorem 5.1 (Appendix C.4), which includes a specialized $\rho$ to slow down the convergence of $\widehat{\pi}_{\mathsf{CE}_{\mathsf{ful}}}$ . Specifically, $\rho = (n + 1)^{-1}$ for all inputs except the last one. Theoretically, the assignment of $\pi^{\star}$ will not affect the convergence of $\widehat{\pi}_{\mathsf{CE}_{\mathsf{ful}}}$ , so we use a $\pi^{\star}$ same as that in Instance 3 only to ensure that $\widehat{\pi}_{\mathsf{SEL},\mathsf{ptl}}$ is not able to converge too fast.

# G Discussions: Dependent Samples in Rewardless MDPs

Besides the popular approach of fine-tuning LLMs (Ouyang et al., 2022; Touvron et al., 2023b) that interprets instructions as inputs and the entire response as a label, there is a more granular perspective where each token is considered a label $a_i$ (See, e.g., the logprobs option in the OpenAI completion API $^{14}$ .) and $s_{i+1}$ is simply the concatenation of $s_i$ and $a_i$ , i.e., $s_{i+1} \sim \mathbb{P}(\cdot | s_i, a_i)$ , where $\mathbb{P}$ is the deterministic transition kernel induced by concatenation. Our bounds for i.i.d. samples already subsume this plausible more involved case through lack of reward. The following reductions hold for any $\mathbb{P} \in \Delta(\mathcal{A}|\mathcal{S} \times \mathcal{A})$ including the aforementioned concatenation kernel.

The proof of any lower bound remains valid so long as $\mathbb{P}(\cdot|s,a):=\rho(\cdot),\forall(s,a)\in\mathcal{S}\times\mathcal{A}$ in the constructed hard-to-learn instance, making the samples i.i.d. Our upper bounds allow $\rho$ to explicitly depend on $\pi^{\star}$ and even P, so the samples can be viewed as i.i.d. samples from $d_{\pi^{\star}}^{P}\times\pi^{\star}$ given the input occupancy measure $d_{\pi^{\star}}^{P}\in\Delta(\mathcal{S})$ is well-defined. Hence, replacing $\rho$ with $d_{\pi^{\star}}^{P}$ validates all arguments for upper bounds. Intuitively, $d_{\pi^{\star}}^{P}(s)$ is the probability of visiting s in a trajectory induced by the transition kernel P and reference policy $\pi^{\star}$ . It is well-developed in either episodic MDPs (Yin et al., 2021) or discounted MDPs (Rashidinejad et al., 2021). These seamless reductions crucially hinge on the absence of value functions and any notion of reward signal in our theoretical framework.

Remark G.1. Our analysis covers but is not specialized to the case where $\rho$ depends on $\pi^{\star}$ or vice versa. Therefore, the result remains unchanged regardless of the relation between $\rho$ and the original training set for training the teacher $\pi^{\star}$ . (For example, $\rho$ may be the distribution of instructions selected by maintainers on the student side (Peng et al., 2023).) It will be intriguing if some further analysis can show any impact of teacher training or data quality on the students' statistical rate.

# H Dicussions on Function Approximation

# H.1 Log-linear and General Softmax Conditional Densities

We discuss potential ways and obstacles of generalizing the results above to large or even uncountable (continuous) state space. First, we extend the concept of conditional probability space $\Delta(\cdot|\cdot)$ to general spaces rigorously. In the following discussions, we assume the notation Y and A refer to finite sets for simplicity. $\Delta(\cdot)$ in this section receives any standard Borel space as input and returns the set of probability measures on it.

Definition H.1 (Polyanskiy and Wu 2022, Definition 2.8). Given two standard Borel spaces $\mathcal{X},\mathcal{Y}$ , a conditional probability (the teacher/student we considered in this paper) $\pi : \mathcal{X} \to \mathcal{Y}$ is a bivariate function $\pi(\cdot|\cdot)$ , whose first argument is a measurable subset of $\mathcal{Y}$ and the second is an element of $\mathcal{X}$ , such that:

- $\forall x \in \mathcal{X}$ , $\pi(\cdot|x)$ is a probability measure on $\mathcal{Y}$ , and   
- $\forall$ measurable $A \subset \mathcal{Y}$ , $x \to \pi(A|x)$ is a measurable function on $\mathcal{X}$ .

The following preliminary result (whose proof is deferred to Section H.3) may shed light on prospective approaches to the analysis of function approximation.

Proposition H.2. For standard Borel spaces X and Y, if both $\dot{\pi},\dot{\pi}\in\Delta(\mathcal{Y}|\mathcal{X})$ are log-linear, i.e., $\dot{\pi}=\Pi(\phi,\dot{\theta}),\dot{\pi}=\Pi(\phi,\dot{\theta})$ , where

$$
\Delta (\mathcal {Y} | \mathcal {X}) \ni \Pi (\boldsymbol {\phi}, \boldsymbol {\theta}) \propto \exp \left(\langle \boldsymbol {\phi}, \boldsymbol {\theta} \rangle\right), \boldsymbol {\phi}: \mathcal {X} \times \mathcal {Y} \to \mathbb {R} ^ {d};
$$

and $\sup_{(x,y)\in\mathcal{X}\times\mathcal{Y}}\|\phi(x,y)\|_2\leq M$ ; then for any $\nu\in\Delta(\mathcal{X})$ ,

$$
\mathrm{TV} (\acute {\pi}, \dot {\pi} | \nu) \leq M \left\| \dot {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\| _ {2}. \tag {H.1}
$$

# H.2 Take-Home Messages and Conjectures

Obviously, any analysis leveraging Proposition H.2 can potentially generalize our results, since log-linear $\pi^{\star}$ subsumes tabular $\pi^{\star}$ . Technically, (H.1) mainly hinges on the (uniform) M-Lipschitz continuity of $\langle\phi,\theta\rangle$ with respect to $\theta$ for any $\theta\in R^{d}$ , therefore, it is also conceptually straightforward to extend Proposition H.2 to general Sofxmax $\pi^{\star}$ , which we omit here for brevity.

Based on (H.1), we conjecture that a fine-grained analysis of the $\ell_{2}$ -norm of $\theta - \theta^{\star}$ may be the key to bound $TV(\widehat{\pi}, \pi^{\star} | \rho)$ from above for any $\widehat{\pi}, \pi^{\star} \in \Delta(\mathcal{A} | \mathcal{S})$ and $\rho \in \Delta(\mathcal{S})$ . Since the tabular setting is a special case of the log-linear setting, we also conjecture that $\left\|\widehat{\theta} - \theta^{\star}\right\|_{2} \gtrsim \sqrt{d/n}$ via Hard Labels and $\left\|\widehat{\theta} - \theta^{\star}\right\|_{2} \gtrsim d/n$ via Partial SLs.

# H.3 Proof of Proposition H.2

Proof of Proposition H.2. For any $x \in X$ ,

$$
\begin{array}{l} \operatorname{KL} (\acute {\pi} (\cdot | x) \| \dot {\pi} (\cdot | x)) = \sum_ {y \in \mathcal {Y}} \acute {\pi} (y | x) \log \frac {\acute {\pi} (y | x)}{\dot {\pi} (y | x)} \\ = \sum_ {i \in \mathcal {Y}} \acute {\pi} (i | x) \left[ \left\langle \boldsymbol {\phi} (x, i), \dot {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\rangle + \log \frac {\sum_ {k \in \mathcal {Y}} \exp \left(\left\langle \boldsymbol {\phi} (x , k) , \dot {\boldsymbol {\theta}} \right\rangle\right)}{\sum_ {j \in \mathcal {Y}} \exp \left(\left\langle \boldsymbol {\phi} (x , j) , \dot {\boldsymbol {\theta}} \right\rangle\right)} \right] \\ \leq \sum_ {i \in \mathcal {Y}} \acute {\pi} (i | x) \left\langle \phi (x, i), \dot {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\rangle + \sum_ {j \in \mathcal {Y}} \frac {\exp \left(\left\langle \phi (x , j) , \dot {\boldsymbol {\theta}} \right\rangle\right)}{\sum_ {k \in \mathcal {Y}} \exp \left(\left\langle \phi (x , k) , \dot {\boldsymbol {\theta}} \right\rangle\right)} \left\langle \phi (x, j), \dot {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\rangle \\ = \sum_ {i \in \mathcal {Y}} (\acute {\pi} (i | x) - \dot {\pi} (i | x)) \left\langle \phi (x, i), \acute {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\rangle \\ \leq \sum_ {i \in \mathcal {Y}} | \dot {\pi} (i | x) - \dot {\pi} (i | x) | \| \phi (x, i) \| _ {2} \left\| \dot {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\| _ {2} \\ \leq 2 \mathrm{TV} (\dot {\pi} (\cdot | x), \dot {\pi} (\cdot | x)) \cdot M \cdot \left\| \dot {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\| _ {2}, \tag {H.2} \\ \end{array}
$$

where the first inequality holds due to the log-sum inequality, the second inequality is a combination of triangle inequality and Cauchy–Schwarz inequality, and the last inequality is by the boundedness of $\phi$ together with the well-known $2TV = \ell_{1}$ relation. We plug (H.2) into Pinsker's inequality to obtain

$$
\left[ \mathsf {T V} \left(\acute {\pi} (\cdot | x), \check {\pi} (\cdot | x)\right) \right] ^ {2} \leq \frac {1}{2} \mathsf {K L} \left(\acute {\pi} (\cdot | x) \| \check {\pi} (\cdot | x)\right) \leq M \mathsf {T V} \left(\acute {\pi} (\cdot | x), \check {\pi} (\cdot | x)\right) \left\| \dot {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\| _ {2}.
$$

$$
\text { So } \mathsf {T V} (\acute {\pi}, \dot {\pi} | \nu) = \int_ {\mathcal {X}} \mathsf {T V} \left(\acute {\pi} (\cdot | x), \dot {\pi} (\cdot | x)\right) \mathrm{d} \nu \leq M \left\| \dot {\boldsymbol {\theta}} - \dot {\boldsymbol {\theta}} \right\| _ {2}.
$$