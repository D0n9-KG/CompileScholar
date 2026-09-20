# Follow-ups Also Matter: Improving Contextual Bandits via Post-serving Contexts

Chaoqi Wang $^{1}$ Ziyu Ye $^{1}$ Zhe Feng $^{2}$ Ashwinkumar Badanidiyuru $^{3}$ Haifeng Xu $^{1}$

University of Chicago $^{1}$ Google Research $^{2}$ Google $^{3}$

{chaoqi, ziyuye, haifengxu}@uchicago.edu

{zhef, ashwinkumarbv}@google.com

# Abstract

Standard contextual bandit problem assumes that all the relevant contexts are observed before the algorithm chooses an arm. This modeling paradigm, while useful, often falls short when dealing with problems in which valuable additional context can be observed after arm selection. For example, content recommendation platforms like Youtube, Instagram, Tiktok also observe valuable follow-up information pertinent to the user's reward after recommendation (e.g., how long the user stayed, what is the user's watch speed, etc.). To improve online learning efficiency in these applications, we study a novel contextual bandit problem with post-serving contexts and design a new algorithm, poLinUCB, that achieves tight regret under standard assumptions. Core to our technical proof is a robustified and generalized version of the well-known Elliptical Potential Lemma (EPL), which can accommodate noise in data. Such robustification is necessary for tackling our problem, and we believe it could also be of general interest. Extensive empirical tests on both synthetic and real-world datasets demonstrate the significant benefit of utilizing post-serving contexts as well as the superior performance of our algorithm over the state-of-the-art approaches.

# 1 Introduction

Contextual bandits represent a fundamental mathematical model that is employed across a variety of applications, such as personalized recommendations (Li et al., 2010; Wu et al., 2016) and online advertising (Schwartz et al., 2017; Nuara et al., 2018). In their conventional setup, at each round t, a learner observes the context $x_{t}$ , selects an arm $a_{t} \in A$ , and subsequently, observes its associated reward $r_{t,a_{t}}$ . Despite being a basic and influential framework, it may not always capture the complexity of real-world scenarios (Wang et al., 2016; Yang et al., 2020). Specifically, the learner often observes valuable follow-up information pertinent to the payoff post arm selection (henceforth, the post-serving context). Standard contextual bandits framework that neglects such post-serving contexts may result in significantly suboptimal performance due to model misspecification.

Consider an algorithm designed to recommend educational resources to a user by utilizing the user's partially completed coursework, interests, and proficiency as pre-serving context (exemplified in platforms such as Coursera). After completing the recommendation, the system can refine the user's profile by incorporating many post-serving context features such as course completion status, how much time spent on different educational resources, performances, etc. This transition naturally delineates a mapping from user attributes (i.e., the pre-serving context) to user's learning experiences and outcomes (i.e., the post-serving context). It is not difficult to see that similar scenarios happen in many other recommender system applications. For instance, in e-commerce platforms (e.g., Amazon, Etsy or any retailing website), the system will first recommend products based on the user's profile information, purchasing pattern and browsing history, etc.; post recommendations, the system can

then update these information by integrating post-serving contexts such as this recent purchase behaviour and product reviews. Similarly, media content recommendation platforms like Youtube, Instagram and Tiktok, also observe many post-serving features (e.g., how long the user stayed) that can refine the system's estimation about users' interaction behavior as well as the rewards.

A common salient point in all the aforementioned scenarios is that the post-serving context are prevalent in many recommender systems; moreover, despite being unseen during the recommendation/serving phase, they can be estimated from the pre-serving context given enough past data. More formally, we assume that there exists a learnable mapping $\phi^{\star}(\cdot):\mathbb{R}^{d_{x}}\to \mathbb{R}^{d_{z}}$ that maps pre-serving feature $\pmb {x}\in \mathbb{R}^{d_{x}}$ to the expectation of the post-serving feature $\pmb {z}\in \mathbb{R}^{d_z}$ , i.e., $\mathbb{E}[\pmb {z}|\pmb {x}] = \phi^{\star}(\pmb {x})$ .

Unsurprisingly, and as we will also show, integrating the estimation of such post-serving features can significantly help to enhance the performance of contextual bandits. However, most of the existing contextual bandit algorithms, e.g., (Auer, 2002; Li et al., 2010; Chu et al., 2011; Agarwal et al., 2014; Tewari and Murphy, 2017), are not designed to accommodate the situations with post-serving contexts. We observe that directly applying these algorithms by ignoring post-serving contexts may lead to linear regret, whereas simple modification of these algorithms will also be sub-optimal. To address these shortcomings, this work introduces a novel algorithm, poLinUCB. Our algorithm leverages historical data to simultaneously estimate reward parameters and the functional mapping from the pre- to post-serving contexts so to optimize arm selection and achieves sublinear regret. En route to analyzing our algorithm, we also developed new technique tools that may be of independent interest.

# Main Contributions.

- First, we introduce a new family of contextual linear bandit problems. In this framework, the decision-making process can effectively integrate post-serving contexts, premised on the assumption that the expectation of post-serving context as a function of the pre-serving context can be gradually learned from historical data. This new model allows us to develop more effective learning algorithms in many natural applications with post-serving contexts.   
- Second, to study this new model, we developed a robustified and generalized version of the well-regarded elliptical potential lemma (EPL) in order to accommodate random noise in the post-serving contexts. While this generalized EPL is an instrumental tool in our algorithmic study, we believe it is also of independent interest due to the broad applicability of EPL in online learning.   
- Third, building upon the generalized EPL, we design a new algorithm poLinUCB and prove that it enjoys a regret bound $\widetilde{\mathcal{O}}(T^{1 - \alpha}d_u^\alpha +d_u\sqrt{TK})$ , where $T$ denotes the time horizon and $\alpha \in [0,1 / 2]$ is the learning speed of the pre- to post-context mapping function $\phi^{\star}(\cdot)$ , whereas $d_{u} = d_{x} + d_{z}$ and $K$ denote the parameter dimension and number of arms. When $\phi^{\star}(\cdot)$ is easy to learn, e.g., $\alpha = 1 / 2$ , the regret bound becomes $\widetilde{\mathcal{O}} (\sqrt{Td_u} +d_u\sqrt{TK})$ and is tight. For general functions $\phi^{\star}(\cdot)$ that satisfy $\alpha \leq 1 / 2$ , this regret bound degrades gracefully as the function becomes more difficult to learn, i.e., as $\alpha$ decreases.   
- Lastly, we empirically validate our proposed algorithm through thorough numerical experiments on both simulated benchmarks and real-world datasets. The results demonstrate that our algorithm surpasses existing state-of-the-art solutions. Furthermore, they highlight the tangible benefits of incorporating the functional relationship between pre- and post-serving contexts into the model, thereby affirming the effectiveness of our modeling.

# 2 Related Works

Contextual bandits. The literature on linear (contextual) bandits is extensive, with a rich body of works (Abe et al., 2003; Auer, 2002; Dani et al., 2008; Rusmevichientong and Tsitsiklis, 2010; Lu et al., 2010; Filippi et al., 2010; Li et al., 2010; Chu et al., 2011; Abbasi-Yadkori et al., 2011; Li et al., 2017; Jun et al., 2017). One of the leading design paradigm is to employ upper confidence bounds as a means of balancing exploration and exploitation, leading to the attainment of minimax optimal regret bounds. The derivation of these regret bounds principally hinges on the utilization of confidence ellipsoids and the elliptical potential lemma. Almost all these works assume that the contextual information governing the payoff is fully observable. In contrast, our work focuses on scenarios where the context is not completely observable during arm selection, thereby presenting new challenges in addressing partially available information.

Contextual bandits with partial information. Contextual bandits with partial information has been relatively limited in the literature. Initial progress in this area was made by Wang et al. (2016), who studied settings with hidden contexts. In their setup there is some context (the post-serving context in our model) that can never be observed by the learner, whereas in our setup the learner can observe post-serving context but only after pulling the arm. Under the assumption that if the parameter initialization is extremely close to the true optimal parameter, then they develop a sub-linear regret algorithm. Our algorithm does not need such strong assumption on parameter initialization. Moreover, we show that their approach may perform poorly in our setup. Subsequent research by Qi et al. (2018); Yang et al. (2020); Park and Faradonbeh (2021); Yang and Ren (2021); Zhu and Kveton (2022) investigated scenarios with noisy or unobservable contexts. In these studies, the learning algorithm was designed to predict context information online through context history analysis, or selectively request context data from an external expert. Our work, on the other hand, introduces a novel problem setting that separates contexts into pre-serving and post-serving categories, enabling the exploration of a wide range of problems with varying learnability. Additionally, we also need to employ new techniques for analyzing our problem to get a near-optimal regret bound.

Generalizations of the elliptical potential lemma (EPL). The EPL, introduced in the seminal work of Lai and Wei (1982), is arguably a cornerstone in analyzing how fast stochastic uncertainty decreases with the observations of new sampled directions. Initially being employed in the analysis of stochastic linear regression, the EPL has since been extensively utilized in stochastic linear bandit problems (Auer, 2002; Dani et al., 2008; Chu et al., 2011; Abbasi-Yadkori et al., 2011; Li et al., 2019; Zhou et al., 2020; Wang et al., 2022). Researchers have also proposed various generalizations of the EPL to accommodate diverse assumptions and problems. For example, Carpentier et al. (2020) extended the EPL by allowing for the use of the $X_{t}^{-p}$ -norm, as opposed to the traditional $X_{t}^{-1}$ -norm. Meanwhile, Hamidi and Bayati (2022) investigated a generalized form of the $1 \wedge \| \varphi(\boldsymbol{x}_t) \|_{\boldsymbol{X}_{t-1}^{-1}}^2$ term, which was inspired by the pursuit of variance reduction in non-Gaussian linear regression models. However, existing (generalized) EPLs are inadequate for the analysis of our new problem setup. Towards that end, we develop a new generalization of the EPL in this work to accommodate noisy feature vectors.

# 3 Linear Bandits with Post-Serving Contexts

Basic setup. We hereby delineate a basic setup of linear contextual bandits within the scope of the partial information setting, whereas multiple generalizations of our framework can be found in Section 6. This setting involves a finite and discrete action space, represented as $\mathcal{A} = [K]$ . Departing from the classic contextual bandit setup, the context in our model is bifurcated into two distinct components: the pre-serving context, denoted as $\boldsymbol{x} \in \mathbb{R}^{d_x}$ , and the post-serving context, signified as $\boldsymbol{z} \in \mathbb{R}^{d_z}$ . When it is clear from context, we sometimes refer to pre-serving context simply as context as in classic setup, but always retain the post-serving context notion to emphasize its difference. We will denote $X_t = \sum_{s=1}^t x_s x_s^\top + \lambda I$ and $Z_t = \sum_{s=1}^t z_s z_s^\top + \lambda I$ . For the sake of brevity, we employ $\boldsymbol{u} = (\boldsymbol{x}, \boldsymbol{z})$ to symbolize the stacked vector of $\boldsymbol{x}$ and $\boldsymbol{z}$ , with $d_u = d_x + d_z$ and $\| \boldsymbol{u} \|_2 \leq L_u$ . The pre-serving context is available during arm selection, while the post-serving context is disclosed post the arm selection. For each arm $a \in \mathcal{A}$ , the payoff, $r_a(x, z)$ , is delineated as follows:

$$
r _ {a} (\boldsymbol {x}, \boldsymbol {z}) = \boldsymbol {x} ^ {\top} \boldsymbol {\theta} _ {a} ^ {\star} + \boldsymbol {z} ^ {\top} \boldsymbol {\beta} _ {a} ^ {\star} + \eta ,
$$

where $\boldsymbol{\theta}_{a}^{\star}$ and $\beta_{a}^{\star}$ represent the parameters associated with the arm, unknown to the learner, whereas $\eta$ is a random noise sampled from an $R_{\eta}$ -sub-Gaussian distribution. We use $\| \pmb{x}\| _p$ to denote the $p$ -norm of a vector $\pmb{x}$ , and $\| \pmb{x}\|_{\pmb{A}}:= \sqrt{\pmb{x}^{\top}\pmb{A}\pmb{x}}$ is the matrix norm. For convenience, we assume $\| \pmb{\theta}_{a}^{\star}\|_{2}\leq 1$ and $\| \beta_{a}^{\star}\|_{2}\leq 1$ for all $a\in \mathcal{A}$ . Additionally, we posit that the norm of the pre-serving and post-serving contexts satisfies $\| \pmb{x}\|_{2}\leq L_{x}$ and $\| \pmb {z}\| _2\leq L_z$ , respectively, and $\max_{t\in [T]}\sup_{a,b\in \mathcal{A}}\langle \pmb{\theta}_{a}^{\star} - \pmb{\theta}_{b}^{\star},\pmb{x}_{t}\rangle \leq 1$ and $\max_{t\in [T]}\sup_{a,b\in \mathcal{A}}\langle \pmb{\beta}_{a}^{\star} - \pmb{\beta}_{b}^{\star},\pmb{z}_{t}\rangle \leq 1$ , same as in (Lattimore and Szepesvári, 2020).

# 3.1 Problem Settings and Assumptions.

The learning process proceeds as follows at each time step $t = 1, 2, \cdots, T$ :

1. The learner observes the context $x_{t}$ .   
2. An arm $a_{t} \in [K]$ is selected by the learner.

# 3. The learner observes the realized reward $r_{t,a_{t}}$ and the post-serving context, $z_{t}$ .

Without incorporating the post-serving context, one may incur linear regret as a result of model misspecification, as illustrated in the following observation. To see this, consider a setup with two arms, $a_1$ and $a_2$ , and a context $x \in \mathbb{R}$ drawn uniformly from the set $\{-3, -1, 1\}$ with $\phi^{\star}(x) = x^{2}$ . The reward functions for the arms are noiseless and determined as $r_{a_1}(x) = x + x^2 / 2$ and $r_{a_2}(x) = -x - x^2 / 2$ . It can be observed that $r_{a_1}(x) > r_{a_2}(x)$ when $x \in \{-3, 1\}$ and $r_{a_1}(x) < r_{a_2}(x)$ when $x = -1$ . Any linear bandit algorithm that solely dependent on the context $x$ (ignoring $\phi^{\star}(x)$ ) will inevitably suffer from linear regret, since it is impossible to have a linear function (i.e., $r(x) = \theta x$ ) that satisfies the above two inequalities simultaneously.

Observation 1. There exists linear bandit environments in which any online algorithm without using post-serving context information will have $\Omega(T)$ regret.

Therefore, it is imperative that an effective learning algorithm must leverage the post-serving context, denoted as $z_{t}$ . As one might anticipate, in the absence of any relationship between $z_{t}$ and $x_{t}$ , it would be unfeasible to extrapolate any information regarding $z_{t}$ while deciding which arm to pull, a point at which only $x_{t}$ is known. Consequently, it is reasonable to hypothesize a correlation between $z_{t}$ and $x_{t}$ . This relationship is codified in the subsequent learnability assumption.

Specifically, we make the following natural assumption — there exists an algorithm that can learn the mean of the post-serving context $z_{t}$ , conditioned on $x_{t}$ . Our analysis will be general enough to accommodate different convergence rates of the learning algorithm, as one would naturally expect, the corresponding regret will degrade as this learning algorithm's convergence rate becomes worse. More specifically, we posit that, given the context $x_{t}$ , the post-serving context $z_{t}$ is generated as $^{1}$

$$
\text { post - serving   context   generation   process: } \quad z _ {t} = \phi^ {\star} (x _ {t}) + \epsilon_ {t}, i. e., \phi^ {\star} (x) = \mathbb {E} [ z | x ].
$$

Here, $\epsilon_t$ is a zero-mean noise vector in $\mathbb{R}^{d_z}$ , and $\phi^\star : \mathbb{R}^{d_x} \to \mathbb{R}^{d_z}$ can be viewed as the post-serving context generating function, which is unknown to the learner. However, we assume $\phi^\star$ is learnable in the following sense.

Assumption 1 (Generalized learnability of $\phi^{*}$ ). There exists an algorithm that, given $t$ pairs of examples $\{(\boldsymbol{x}_s,\boldsymbol{z}_s)\}_{s = 1}^t$ with arbitrarily chosen $\boldsymbol{x}_s$ 's, outputs an estimated function of $\phi^{\star}:\mathbb{R}^{d_{x}}\to \mathbb{R}^{d_{z}}$ such that for any $\boldsymbol {x}\in \mathbb{R}^{d_x}$ , the following holds with probability at least $1 - \delta$ ,

$$
e _ {t} ^ {\delta} := \left\| \widehat {\phi} _ {t} (\boldsymbol {x}) - \phi^ {\star} (\boldsymbol {x}) \right\| _ {2} \leq C _ {0} \cdot \left(\| \boldsymbol {x} \| _ {\boldsymbol {X} _ {t} ^ {- 1}} ^ {2}\right) ^ {\alpha} \cdot \log (t / \delta),
$$

where $\alpha \in (0,1 / 2]$ and $C_0$ is some universal constant.

The aforementioned assumption encompasses a wide range of learning scenarios, each with different rates of convergence. Generally, the value of $\alpha$ is directly proportional to the speed of learning; the larger the value of $\alpha$ , the quicker the learning rate. Later, we will demonstrate that the regret of our algorithm is proportional to $O(T^{1-\alpha})$ , exhibiting a graceful degradation as $\alpha$ decreases. The ensuing proposition demonstrates that for linear functions, $\alpha = 1/2$ . This represents the best learning rate that can be accommodated $^{2}$ . In this scenario, the regret of our algorithm is $O(\sqrt{T})$ , aligning with the situation devoid of post-serving contexts (Li et al., 2010; Abbasi-Yadkori et al., 2011).

Observation 2. Suppose $\phi (\cdot)$ is a linear function, i.e., $\phi (\pmb {x}) = \Phi^{\top}\pmb{x}$ for some $\Phi \in \mathbb{R}^{d_x\times d_z}$ , then $e_t^\delta = \mathcal{O}\left(\| \pmb {x}\|_{\pmb{X}_t^{-1}}\cdot \log (t / \delta)\right)$ .

This observation follows from the following inequalities $\|\phi_{t}(\boldsymbol{x})-\phi^{\star}(\boldsymbol{x})\|=\|\widehat{\Phi}_{t}^{\top}\boldsymbol{x}-\Phi^{\star^{\top}}\boldsymbol{x}\|\leq\|\widehat{\Phi}_{t}-\Phi^{\star}\|_{\boldsymbol{X}_{t}}\cdot\|x\|_{\boldsymbol{X}_{t}^{-1}}=\mathcal{O}(\|x\|_{\boldsymbol{X}_{t}^{-1}}\cdot\log\left(\frac{t}{\delta}\right))$ , where the last equation is due to the confidence ellipsoid bound (Abbasi-Yadkori et al., 2011). We refer curious readers to Appendix D.2 for a discussion on other more challenging $\phi(\cdot)$ functions with possibly worse learning rates $\alpha$ .

# 3.2 Warm-up: Why Natural Attempts May Be Inadequate?

Given the learnability assumption of $\phi^{\star}$ , one natural idea for solving the above problem is to estimate $\phi^{\star}$ , and then run the standard LinUCB algorithm to estimate $(\theta_{a},\beta_{a})$ together by treating $(\boldsymbol{x}_t,\widehat{\phi}_t(\boldsymbol{x}_t))$ as the true contexts. Indeed, this is the approach adopted by Wang et al. (2016) for addressing a similar problem of missing contexts $z_{t}$ , except that they used a different unsurprised-learning-based approach to estimate the context $z_{t}$ due to not being able to observing any data about $z_{t}$ . Given the estimation of $\widehat{\phi}$ , their algorithm — which we term it as LinUCB $(\widehat{\phi})$ — iteratively carries out the steps below at each iteration $t$ (see Algorithm 2 for additional details): 1) Estimation of the context-generating function $\widehat{\phi}_t(\cdot)$ from historical data; 2) Solve of the following regularized least square problem for each arm $a\in \mathcal{A}$ , with regularization coefficient $\lambda \geq 0$ :

$$
\ell_ {t} \left(\boldsymbol {\theta} _ {a}, \boldsymbol {\beta} _ {a}\right) = \sum_ {s \in [ t ]: a _ {s} = a} \left(r _ {s, a} - \boldsymbol {x} _ {t} ^ {\top} \boldsymbol {\theta} _ {a} - \widehat {\phi} _ {s} \left(\boldsymbol {x} _ {s}\right) ^ {\top} \boldsymbol {\beta} _ {a}\right) ^ {2} + \lambda \left(\| \boldsymbol {\theta} _ {a} \| _ {2} ^ {2} + \| \boldsymbol {\beta} _ {a} \| _ {2} ^ {2}\right), \tag {1}
$$

Under the assumption that the initialized parameters in their estimations are very close to the global optimum, Wang et al. (2016) were able to show the $O(\sqrt{T})$ regret of this algorithm. However, it turns out that this algorithm will fail to yield an satisfying regret bound without their strong assumption on very close parameter initialization, because the errors arising from $\widehat{\phi}(\cdot)$ will significantly enlarge the confidence set of $\widehat{\theta}_a$ and $\widehat{\beta}_a$ . Thus after removing their initialization assumption, the best possible regret bound we can possibly achieve is of order $\widetilde{\mathcal{O}}(T^{3/4})$ , as illustrated in the subsequent proposition.

Proposition 1 (Regret of LinUCB-( $\widehat{\phi}$ )). The regret of LinUCB-( $\widehat{\phi}$ ) in Algorithm 2 is upper bounded by $\widetilde{\mathcal{O}}\left(T^{1-\alpha}d_{u}^{\alpha}+T^{1-\alpha/2}\sqrt{Kd_{u}^{1+\alpha}}\right)$ with probability at least $1-\delta$ , by carefully setting the regularization coefficient $\lambda=\Theta(L_{u}d_{u}^{\alpha}T^{1-\alpha}\log(T/\delta))$ in Equation 1.

Since $\alpha \in [0,1/2]$ , the best possible regret upper bound above is $\widetilde{\mathcal{O}}(T^{3/4})$ , which is considerably inferior to the sought-after regret bound of $\widetilde{\mathcal{O}}(\sqrt{T})$ . Such deficiency of LinUCB- $(\widehat{\phi})$ is further observed in all our experiments in Section 7 as well. These motivate our following design of a new online learning algorithm to address the challenge of post-serving context, during which we also developed a new technical tool which may be of independent interest to the research community.

# 4 A Robustified and Generalized Elliptical Potential Lemma

It turns out that solving the learning problem above requires some novel designs; core to these novelties is a robustified and generalized version of the well-known elliptical potential lemma (EPL), which may be of independent interest. This widely used lemma states a fact about a sequence of vectors $x_{1},\cdots,x_{T}\in R^{d}$ . Intuitively, it captures the rate of the sum of additional information contained in each $x_{t}$ , relative to its predecessors $x_{1},\cdots,x_{t-1}$ . Formally,

Lemma (Original Elliptical Potential Lemma). Suppose (1) $X_{0} \in R^{d \times d}$ is any positive definite matrix; (2) $x_{1}, \ldots, x_{T} \in R^{d}$ is any sequence of vectors; and (3) $X_{t} = X_{0} + \sum_{s=1}^{t} x_{s} x_{s}^{\top}$ . Then the following inequality holds

$$
\sum_ {t = 1} ^ {T} 1 \wedge \| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2} \leq 2 \log \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right),
$$

where $a \wedge b = \min\{a, b\}$ is the min among $a, b \in \mathbb{R}$ .

To address our new contextual bandit setup with post-serving contexts, it turns out that we will need to robustify and generalize the above lemma to accommodate noises in $x_{t}$ vectors and slower learning rates. Specifically, we present the following variant of the EPL lemma.

Lemma 1 (Generalized Elliptical Potential Lemma). Suppose (1) $\mathbf{X}_0 \in \mathbb{R}^{d \times d}$ is any positive definite matrix; (2) $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \mathbb{R}^d$ is a sequence of vectors with bounded $l_2$ norm $\max_t \| \mathbf{x}_t \| \leq L_x$ ; (3) $\epsilon_1, \ldots, \epsilon_T \in \mathbb{R}^d$ is a sequence of independent (not necessarily identical) bounded zero-mean noises satisfying $\max_t \| \epsilon_t \| \leq L_\epsilon$ and $\mathbb{E}[\epsilon_t \epsilon_t^\top] \succcurlyeq \sigma_\epsilon^2 I$ for any $t$ ; and (4) $\widetilde{\mathbf{X}}_t$ is defined as follows:

$$
\widetilde {\boldsymbol {X}} _ {t} = \boldsymbol {X} _ {0} + \sum_ {s = 1} ^ {t} (\boldsymbol {x} _ {s} + \boldsymbol {\epsilon} _ {s}) (\boldsymbol {x} _ {s} + \boldsymbol {\epsilon} _ {s}) ^ {\top} \in \mathbb {R} ^ {d \times d}.
$$

Then, for any $p \in [0,1]$ , the following inequality holds with probability at least $1 - \delta$ ,

$$
\sum_ {t = 1} ^ {T} \left(1 \wedge \| \boldsymbol {x} _ {t} \| _ {\widetilde {\boldsymbol {X}} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \leq 2 ^ {p} T ^ {1 - p} \log^ {p} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right) + \frac {8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\sigma_ {\epsilon} ^ {4}} \log \left(\frac {3 2 d L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\delta \sigma_ {\epsilon} ^ {4}}\right) \tag {2}
$$

Note that the second term is independent of time horizon T and only depends on the setup parameters. Generally, this can be treated as a constant. Before describing main proof idea of the lemma, we make a few remarks regarding Lemma 1 to highlight the significance of these generalizations.

1. The original Elliptical Potential Lemma (EPL) corresponds to the specific case of $p = 1$ , while Lemma 1 is applicable for any $p \in [0,1]$ . Notably, the $(1 - p)$ rate in the $T^{1 - p}$ term of Inequality 2 is tight for every $p$ . In fact, this rate is tight even for $\pmb{x}_t = 1 \in \mathbb{R}, \forall t$ and $\pmb{X}_0 = 1 \in \mathbb{R}$ since, under these conditions, $\| \pmb{x}_t\|_{\pmb{X}_{t - 1}^{-1}}^2 = 1 / t$ and, consequently, $\sum_{t = 1}^{T}\left(1\wedge \| \pmb{x}_t\|_{\pmb{X}_{t - 1}^{-1}}^2\right)^p = \sum_{t = 1}^{T}t^{-p}$ , yielding a rate of $T^{1 - p}$ . This additional flexibility gained by allowing a general $p\in [0,1]$ (with the original EPL corresponding to $p = 1$ ) helps us to accommodate slower convergence rates when learning the mean context from observed noisy contexts, as formalized in Assumption 1.   
2. A crucial distinction between Lemma 1 and the original EPL lies in the definition of the noisy data matrix $\widetilde{\mathbf{X}}_t$ in Equation 1, which permits noise. However, the measured context vector $x_{t}$ does not have noise. This is beneficial in scenarios where a learner observes noisy contexts but seeks to establish an upper bound on the prediction error based on the underlying noise-free context or the mean context. Such situations are not rare in real applications; our problem of contextual bandits with post-serving contexts is precisely one of such case — while choosing an arm, we can estimate the mean post-serving context conditioned on the observable pre-serving context but are only able to observe the noisy realization of post-serving contexts after acting.   
3. Other generalized variants of the EPL have been recently proposed and found to be useful in different contexts. For instance, Carpentier et al. (2020) extends the EPL to allow for the $X_{t}^{-p}$ -norm, as opposed to the $X_{t}^{-1}$ -norm, while Hamidi and Bayati (2022) explores a generalized form of the $1 \wedge \| \varphi(\boldsymbol{x}_t)\|_{\boldsymbol{X}_{t-1}^{-1}}^2$ term, which is motivated by variance reduction in non-Gaussian linear regression models. Nevertheless, to the best of our knowledge, our generalized version is novel and has not been identified in prior works.

Proof Sketch of Lemma 1. The formal proof of this lemma is involved and deferred to Appendix B.1. At a high level, our proof follows procedure for proving the original EPL. However, to accommodate the noises in the data matrix, we have to introduce new matrix concentration tools to the original (primarily algebraic) proof, and also identify the right conditions for the argument to go through. A key lemma to our proof is a high probability bound regarding the constructed noisy data matrix $\widetilde{\mathbf{X}}_t$ (Lemma 2 in Appendix B.1) that we derive based on Bernstein's Inequality for matrices under spectral norm (Tropp et al., 2015). We prove that, under mild assumptions on the noise, $\| \pmb{x}_t\|_{\widetilde{\mathbf{X}}_{t - 1}^{-1}}^2\leq \| \pmb{x}_t\|_{\mathbf{X}_{t - 1}^{-1}}^2$ with high probability for any $t$ . Next, we have to apply the union bound and this lemma to show that the above matrix inequality holds for every $t\geq 1$ with high probability. Unfortunately, this turns out to not be true because when $t$ is very small (e.g., $t = 1$ ), the above inequality cannot hold with high probability. Therefore, we have to use the union bound in a carefully tailored way by excluding all $t$ 's that are smaller than a certain threshold (chosen optimally by solving certain inequalities) and handling these terms with small $t$ separately (which is the reason of the second $\mathcal{O}(\log (1 / \delta))$ term in Inequality 2). Finally, we refine the analysis of sthe standard EPL by allowing the exponent $p$ in

Algorithm 1 poLinUCB (Linear UCB with post-serving contexts)   
1: for $t = 0, 1, \ldots, T$ do
2: Receive the pre-serving context $x_{t}$ 3: Compute the optimistic parameters by maximizing the UCB objective $\left(a_{t},\widetilde{\phi}_{t}(\boldsymbol{x}_{t}),\widetilde{\boldsymbol{w}}_{t}\right)=\operatorname*{arg max}_{(a,\phi,\boldsymbol{w}_{a})\in[K]\times\mathcal{C}_{t-1}\left(\widehat{\phi}_{t-1},\boldsymbol{x}_{t}\right)\times\mathcal{C}_{t-1}\left(\widehat{\boldsymbol{w}}_{t-1,a}\right)}\left[\begin{matrix}\boldsymbol{x}_{t}\\ \phi(\boldsymbol{x}_{t})\end{matrix}\right]^{\top}\boldsymbol{w}_{a}.$ 4: Play the arm $a_{t}$ and receive the realized post-serving context as $z_{t}$ and the real-valued reward $r_{t,a_{t}}=\begin{bmatrix}\boldsymbol{x}_{t}\\ \boldsymbol{z}_{t}\end{bmatrix}^{\top}\boldsymbol{w}_{a_{t}}^{\star}+\eta_{t}.$ 5: Compute $\widehat{w}_{t,a}$ using Equation 3 for each $a\in A$ .
6: Compute the estimated post-serving context generating function $\widehat{\phi}_{t}(\cdot)$ using ERM.
7: Update confidence sets $\mathcal{C}_{t}(\widehat{\boldsymbol{w}}_{t,a})$ and $\mathcal{C}_{t}(\widehat{\phi}_{t},\boldsymbol{x}_{t})$ for each a based on Equations 6 and 5.
8: end for

$(1 \wedge \|x_{t}\|_{\widetilde{X}_{t-1}^{-1}}^{2})^{p}$ and derive an upper bound on the sum $\sum_{t=1}^{T}(1 \wedge \|x_{t}\|_{\widetilde{X}_{t-1}^{-1}}^{2})^{p}$ with high probability. These together yeilds a robustified and generalized version of EPL as in Lemma 1. □

# 5 No Regret Learning in Linear Bandits with Post-Serving Contexts

# 5.1 The Main Algorithm

In the ensuing section, we introduce our algorithm, poLinUCB, designed to enhance linear contextual bandit learning through the incorporation of post-serving contexts and address the issue arose from the algorithm introduced in Section 3.2. The corresponding pseudo-code is delineated in Algorithm 1. Unlike the traditional LinUCB algorithm, which solely learns and sustains confidence sets for parameters (i.e., $\widehat{\beta}_a$ and $\widehat{\theta}_a$ for each $a$ ), our algorithm also simultaneously manages the same for the post-serving context generating function, $\widehat{\phi} (\cdot)$ . Below, we expound on our methodology for parameter learning and confidence set construction.

Parameter learning. During each iteration t, we fit the function $\widehat{\phi}_{t}(\cdot)$ and the parameters $\{\widehat{\theta}_{t,a}\}_{a\in\mathcal{A}}$ and $\{\widehat{\beta}_{t,a}\}_{a\in\mathcal{A}}$ . To fit $\widehat{\phi}_{t}(\cdot)$ , resort to the conventional empirical risk minimization (ERM) framework. As for $\{\widehat{\theta}_{t,a}\}_{a\in\mathcal{A}}$ and $\{\widehat{\beta}_{t,a}\}_{a\in\mathcal{A}}$ , we solve the following least squared problem for each arm a,

$$
\ell_ {t} \left(\boldsymbol {\theta} _ {a}, \boldsymbol {\beta} _ {a}\right) = \sum_ {s \in [ t ]: a _ {s} = a} \left(r _ {s, a} - \boldsymbol {x} _ {s} ^ {\top} \boldsymbol {\theta} _ {a} - \boldsymbol {z} _ {s} ^ {\top} \boldsymbol {\beta} _ {a}\right) ^ {2} + \lambda \left(\| \boldsymbol {\theta} _ {a} \| _ {2} ^ {2} + \| \boldsymbol {\beta} _ {a} \| _ {2} ^ {2}\right). \tag {3}
$$

For convenience, we use $\pmb{w}$ and $\pmb{u}$ to denote $(\pmb{\theta},\beta)$ and $(\pmb{x},\pmb{z})$ respectively. The closed-form solutions to $\widehat{\pmb{\theta}}_{t,a}$ and $\widehat{\beta}_{t,a}$ for each arm $a\in \mathcal{A}$ are

$$
\widehat {\boldsymbol {w}} _ {t, a} := \left[ \begin{array}{l} \widehat {\boldsymbol {\theta}} _ {t, a} \\ \widehat {\boldsymbol {\beta}} _ {t, a} \end{array} \right] = \boldsymbol {A} _ {t, a} ^ {- 1} \boldsymbol {b} _ {t, a}, \text {   where   } \boldsymbol {A} _ {t, a} = \lambda \boldsymbol {I} + \sum_ {s: a _ {s} = a} ^ {t} \boldsymbol {u} _ {s} \boldsymbol {u} _ {s} ^ {\top} \quad \text { and } \quad \boldsymbol {b} _ {t, a} = \sum_ {s: a _ {s} = a} ^ {t} r _ {s, a} \boldsymbol {u} _ {s}. \tag {4}
$$

Confidence set construction. At iteration t, we construct the confidence set for $\widehat{\phi}_{t}(\boldsymbol{x}_{t})$ by

$$
\mathcal {C} _ {t} \left(\widehat {\phi} _ {t}, \boldsymbol {x} _ {t}\right) := \left\{\boldsymbol {z} \in \mathbb {R} ^ {d}: \left\| \widehat {\phi} _ {t} (\boldsymbol {x} _ {t}) - \boldsymbol {z} \right\| _ {2} \leq e _ {t} ^ {\delta} \right\}. \tag {5}
$$

Similarly, we can construct the confidence set for the parameters $\widehat{\pmb{w}}_{t,a}$ for each arm $a\in \mathcal{A}$ by

$$
\mathcal {C} _ {t} \left(\widehat {\boldsymbol {w}} _ {t, a}\right) := \left\{\boldsymbol {w} \in \mathbb {R} ^ {d _ {x} + d _ {z}}: \| \boldsymbol {w} - \widehat {\boldsymbol {w}} _ {t, a} \| _ {\boldsymbol {A} _ {t, a}} \leq \zeta_ {t, a} \right\}, \tag {6}
$$

where $\zeta_{t,a}=2\sqrt{\lambda}+R_{\eta}\sqrt{d_{u}\log((1+n_{t}(a)L_{u}^{2}/\lambda)/\delta)}$ and $n_{t}(a)=\sum_{s=1}^{t}\mathbb{1}[a_{s}=a]$ . Additionally, we further define $\zeta_{t}:=\max_{a\in\mathcal{A}}\zeta_{t,a}$ . By the assumption 1 and Lemma 3, we have the followings hold with probability at least $1-\delta$ for each of the following events,

$$
\phi^ {\star} (\boldsymbol {x} _ {t}) \in \mathcal {C} _ {t} \left(\widehat {\phi} _ {t}, \boldsymbol {x} _ {t}\right) \quad \text { and } \quad \boldsymbol {w} ^ {\star} \in \mathcal {C} _ {t} \left(\widehat {\boldsymbol {w}} _ {t, a}\right). \tag {7}
$$

# 5.2 Regret Analysis

In the forthcoming section, we establish the regret bound. Our proof is predicated upon the conventional proof of LinUCB (Li et al., 2010) in conjunction with our robust elliptical potential lemma. The pseudo-regret (Audibert et al., 2009) within this partial contextual bandit problem is defined as,

$$
R _ {T} = \text { Regret } (T) = \sum_ {t = 1} ^ {T} \left(r _ {t, a _ {t} ^ {*}} - r _ {t, a _ {t}}\right), \tag {8}
$$

in which we reload the notation of reward by ignoring the noise,

$$
r _ {t, a} = \left\langle \boldsymbol {\theta} _ {a} ^ {\star}, \boldsymbol {x} _ {t} \right\rangle + \left\langle \boldsymbol {\beta} _ {a} ^ {\star}, \phi^ {\star} (\boldsymbol {x} _ {t}) \right\rangle \quad \text { and } \quad a _ {t} ^ {\star} = \underset {a \in \mathcal {A}} {\arg \max} \left\langle \boldsymbol {\theta} _ {a} ^ {\star}, \boldsymbol {x} _ {t} \right\rangle + \left\langle \boldsymbol {\beta} _ {a} ^ {\star}, \phi^ {\star} (\boldsymbol {x} _ {t}) \right\rangle . \tag {9}
$$

It is crucial to note that our definition of the optimal action, $a_{t}^{\star}$ , in Eq. 9 depends on $\phi^{\star}(\boldsymbol{x}_{t})$ as opposed to $z_{t}$ . This dependency ensures a more pragmatic benchmark, as otherwise, the noise present in z would invariably lead to a linear regret, regardless of the algorithm implemented. In the ensuing section, we present our principal theoretical outcomes, which provide an upper bound on the regret of our poLinUCB algorithm.

Theorem 1 (Regret of poLinUCB). The regret of poLinUCB in Algorithm 1 is upper bounded by $\widetilde{\mathcal{O}}\left(T^{1 - \alpha}d_u^\alpha +d_u\sqrt{TK}\right)$ with probability at least $1 - \delta$ , if $T = \Omega (\log (1 / \delta))$ .

The first term in the bound is implicated by learning the function $\phi^{\star}(\cdot)$ . Conversely, the second term resembles the one derived in conventional contextual linear bandits, with the exception that our dependency on $d_{u}$ is linear. This linear dependency is a direct consequence of our generalized robust elliptical potential lemma. The proof is deferred in Appendix B.2.

# 6 Generalizations

So far we have focused on a basic linear bandit setup with post-serving features. Our results and analysis can be easily generalized to other variants of linear bandits, including those with feature mappings, and below we highlight some of these generalizations. They use similar proof ideas, up to some technical modifications; we thus defer all their formal proofs to Appendix B.3.

# 6.1 Generalization to Action-Dependent Contexts

Our basic setup in Section 3 has a single context $x_{t}$ at any time step t. This can be generalized to action-dependent contexts settings as studied in previous works (e.g., Li et al. (2010)). That is, during each iteration indexed by t, the learning algorithm observes a context $x_{t,a}$ for each individual arm $a \in A$ . Upon executing the action of pulling arm $a_{t}$ , the corresponding post-serving context $z_{t,a_{t}}$ is subsequently revealed. Notwithstanding, the post-serving context for all alternative arms remains unobserved. The entire procedure is the same as that of Section 3.

In extending this framework, we persist in our assumption that for each arm $a \in A$ , there exists a specific function $\phi_{a}^{\star}(\cdot): \mathbb{R}^{d_{x}} \to \mathbb{R}^{d_{z}}$ that generates the post-serving context z upon receiving x associated with arm $a \in A$ . The primary deviation from our preliminary setup lies in the fact that we now require the function $\phi_{a}^{\star}(\cdot)$ to be learned for each arm independently. The reward is generated as

$$
r _ {t, a _ {t}} = \left\langle \boldsymbol {\theta} _ {a _ {t}} ^ {\star}, \boldsymbol {x} _ {t, a _ {t}} \right\rangle + \left\langle \boldsymbol {\beta} _ {a _ {t}} ^ {\star}, \boldsymbol {z} _ {t, a _ {t}} \right\rangle + \eta_ {t}.
$$

The following proposition shows our regret bound for this action-dependent context case. Its proof largely draws upon the proof idea of Theorem 1 and also relies on the generalized EPL Lemma 1.

Proposition 2. The regret of poLinUCB in Algorithm 1 for action-dependent contexts is upper bounded by $\widetilde{\mathcal{O}}\left(T^{1-\alpha}d_{u}^{\alpha}\sqrt{K}+d_{u}\sqrt{TK}\right)$ with probability at least $1-\delta$ if $T=\Omega(\log(1/\delta))$ .

The main difference with the bound in Theorem 1 is the additional $\sqrt{K}$ appeared in the first term, which is caused by learning multiple $\phi_a^\star (\cdot)$ functions with $a\in \mathcal{A}$ .

![](images/077168743e5c9987eed3c8a6a6f6ed0981e4c8dd26c3f7aeb231ab83f18f125f.jpg)

<details>
<summary>line</summary>

| Time steps | Random | LinUCB (x only) | LinUCB (x and z) | LinUCB (φ̂) | poLinUCB (ours) |
| ---------- | ------ | --------------- | ---------------- | ---------- | --------------- |
| 0          | 0      | 0               | 0                | 0          | 0               |
| 1000       | 3e5    | 4.5e5           | 1.5e5            | 1.5e5      | 1.5e5           |
| 2000       | 4e5    | 6.0e5           | 2.0e5            | 2.0e5      | 2.0e5           |
| 3000       | 4.5e5  | 7.0e5           | 2.2e5            | 2.2e5      | 2.2e5           |
| 4000       | 5e5    | 8.0e5           | 2.4e5            | 2.4e5      | 2.4e5           |
| 5000       | 5.5e5  | 9.0e5           | 2.6e5            | 2.6e5      | 2.6e5           |
| 6000       | 6e5    | 1.0e6           | 2.8e5            | 2.8e5      | 2.8e5           |
| 7000       | 6.5e5  | 1.1e6           | 3.0e5            | 3.0e5      | 3.0e5           |
| 8000       | 7e5    | 1.2e6           | 3.2e5            | 3.2e5      | 3.2e5           |
| 9000       | 7.5e5  | 1.3e6           | 3.4e5            | 3.4e5      | 3.4e5           |
| 10000      | 8e5    | 1.4e6           | 3.6e5            | 3.6e5      | 3.6e5           |
| 11000      | 8.5e5  | 1.5e6           | 3.8e5            | 3.8e5      | 3.8e5           |
| 12000      | 9e5    | 1.6e6           | 4.0e5            | 4.0e5      | 4.0e5           |
| 13000      | 9.5e5  | 1.7e6           | 4.2e5            | 4.2e5      | 4.2e5           |
| 14000      | 1e6    | 1.8e6           | 4.4e5            | 4.4e5      | 4.4e5           |
| 15000      | 1.1e6  | 1.9e6           | 4.6e5            | 4.6e5      | 4.6e5           |
| 16000      | 1.2e6  | 2.0e6           | 4.8e5            | 4.8e5      | 4.8e5           |
| 17000      | 1.3e6  | 2.1e6           | 5.0e5            | 5.0e5      | 5.0e5           |
| 18000      | 1.4e6  | 2.2e6           | 5.2e5            | 5.2e5      | 5.2e5           |
| 19000      | 1.5e6  | 2.3e6           | 5.4e5            | 5.4e5      | 5.4e5           |
| 20000      | 1.6e6  | 2.4e6           | 5.6e5            | 5.6e5      | 5.6e5           |
| 21000      | 1.7e6  | 2.5e6           | 5.8e5            | 5.8e5      | 5.8e5           |
| 22000      | 1.8e6  | 2.6e6           | 6.0e5            | 6.0e5      | 6.0e5           |
| 23000      | 1.9e6  | 2.7e6           | 6.2e5            | 6.2e5      | 6.2e5           |
| 24000      | 2e6    | 2.8e6           | 6.4e5            | 6.4e5      | 6.4e5           |
| 25000      | 2.1e6  | 2.9e6           | 6.6e5            | 6.6e5      | 6.6e5           |
| 26000      | 2.2e6  | 3.0e6           | 6.8e5            | 6.8e5      | 6.8e5           |
| 27000      | 2.3e6  | 3.1e6           | 7.0e5            | 7.0e5      | 7.0e5           |
| 28000      | 2.4e6  | 3.2e6           | 7.2e5            | 7.2e5      | 7.2e5           |
| 29000      | 2.5e6  | 3.3e6           | 7.4e5            | 7.4e5      | 7.4e5           |
| 30000      | 2.6e6  | 3.4e6           | 7.6e5            | 7.6e5      | 7.6e5           |
| ...        | ...    | ...             | ...              | ...        | ...             |
| ...        | ...    | ...             | ...              | ...        | ...             |
| ...        | ...    | ...             | ...              | ...        | ...             |
| ...        | ...    | ...             | ...              | ...        | ...             |
| ...        | ...    | ...             | ...              | ...        | ...             |
| ...        | ...    | ...             | ...              | ...        | ...             |
| ...        = ?         : φ(x) = W^T(x^2) + b; linUCB(x only); linUCB(x and z); linUCB(φ̂); linUCB(φ); poLinUCB(x only); linUCB(x only); linUCB(x and z); linUCB(φ̂); poLinUCB(x only); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(φ̂); linUCB(x and z); linUCB(x and z); linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linUCB(\phi) ; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; Linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucb x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x + b; linucs x+ b; linucs x+ b; linucs x+ b; linucs x+ b; linucs x+ b; linucs x+ b; linucs x+ b; linucs x+ b; linucs x+ b; LinuCB (x only) ; LinuCB (x and z) ; LinuCB (φ̂) ; no data series for this series are not provided in the code snippet in the provided code.
</details>

Figure 1: Cumulative Regret in three synthetic environments. Comparisons of different algorithms in terms of cumulative regret across the three synthetic environments. Our proposed poLinUCB (ours) consistently outperforms other strategies (except for LinUCB which has access to the post-serving context during arm selection), showcasing its effectiveness in utilizing post-serving contexts. The shaded area denotes the standard error computed using 10 different random seeds.

# 6.2 Generalization to Linear Stochastic Bandits

Another variant of linear bandits is the linear stochastic bandits setup (see, e.g., (Abbasi-Yadkori et al., 2011)). This model allows infinitely many arms, which consists of a decision set $D_{t} \subseteq \mathbb{R}^{d}$ at time $t$ , and the learner picks an action $\boldsymbol{x}_{t} \in D_{t}$ . This setup naturally generalizes to our problem with post-serving contexts. That is, at iteration $t$ , the learner selects an arm $\boldsymbol{x}_{t} \in D_{t}$ first, receives reward $r_{t,\boldsymbol{x}_t}$ , and then observe the post-serving feature $\boldsymbol{z}_t$ conditioned on $\boldsymbol{x}_t$ . Similarly, we assume the existence of a mapping $\phi^{\star}(\boldsymbol{x}_t) = \mathbb{E}[\boldsymbol{z}_t|\boldsymbol{x}_z]$ that satisfies the Assumption 1. Consequently, the realized reward is generated as follows where $\theta^{*}, \beta^{*}$ are unknown parameters:

$$
r _ {t, \boldsymbol {x} _ {t}} = \left\langle \boldsymbol {x} _ {t}, \boldsymbol {\theta} ^ {\star} \right\rangle + \left\langle \boldsymbol {z} _ {t}, \boldsymbol {\beta} ^ {\star} \right\rangle + \eta_ {t}.
$$

Therefore, the learner needs to estimate the linear parameters $\widehat{\pmb{\theta}}$ and $\widehat{\pmb{\beta}}$ , as well as the function $\widehat{\phi} (\cdot)$ . We obtain the following proposition for the this setup.

Proposition 3. The regret of poLinUCB in Algorithm 1 for the above setting is upper bounded by $\widetilde{\mathcal{O}}\left(T^{1 - \alpha}d_u^\alpha +d_u\sqrt{T}\right)$ with probability at least $1 - \delta$ if $T = \Omega (\log (1 / \delta))$ .

# 6.3 Generalization to Linear Bandits with Feature Mappings

Finally, we briefly remark that while we have so far assumed that the arm parameters are directly linear in the context $x_{t}, z_{t}$ , just like classic linear bandits our analysis can be easily generalized to accommodate feature mapping $\pi^{x}(\boldsymbol{x}_{t})$ and $\pi^{z}(\boldsymbol{z}_{t}) = \pi^{z}(\phi(\boldsymbol{x}_{t}) + \varepsilon_{t})$ . Specifically, if the reward generation process is $r_{a} = \langle \boldsymbol{\theta}_{a}^{\star}, \pi^{x}(\boldsymbol{x}_{t}) \rangle + \langle \boldsymbol{\beta}_{a}^{\star}, \pi^{z}(\boldsymbol{z}_{t}) \rangle + \eta_{t}$ instead, then we can simply view $\tilde{\boldsymbol{x}}_{t} = \pi^{x}(\boldsymbol{x}_{t})$ and $\tilde{\boldsymbol{z}}_{t} = \pi^{z}(\boldsymbol{z}_{t})$ as the new features, with $\tilde{\phi}(\boldsymbol{x}_{t}) = \mathbb{E}_{\boldsymbol{\epsilon}_{t}}[\pi^{z}(\phi(\boldsymbol{x}_{t}) + \boldsymbol{\epsilon}_{t})]$ . By working with $\tilde{x}_{t}, \tilde{z}_{t}, \tilde{\phi}$ , we shall obtain the same guarantees as Theorem 1.

# 7 Experiments

This section presents a comprehensive evaluation of our proposed poLinUCB algorithm on both synthetic and real-world data, demonstrating its effectiveness in incorporating follow-up information and outperforming the LinUCB( $\widehat{\phi}$ ) variant. More empirical results can be found in Appendix D.1.

# 7.1 Synthetic Data with Ground Truth Models

Evaluation Setup. We adopt three different synthetic environments that are representative of a range of mappings from the pre-serving context to the post-serving context: polynomial, periodicical and linear functions. The pre-serving contexts are sampled from a uniform noise in the range $[-10, 10]^{d_{x}}$ , and Gaussian noise is employed for both the post-serving contexts and the rewards. In each environment, the dimensions of the pre-serving context $(d_{x})$ and the post-serving context $(d_{z})$ are of 100 and 5, respectively with 10 arms $(K)$ . The evaluation spans T = 1000 or 5000 time steps, and each experiment is repeated with 10 different seeds. The cumulative regret for each policy in each environment is then calculated to provide a compararison.

Results and Discussion. Our experimental results, which are presented graphically in Figures 1, provide strong evidence of the superiority of our proposed poLinUCB algorithm. Across all setups, we

observe that the LinUCB $(x$ and $z)$ strategy, which has access to the post-serving context during arm selection, consistently delivers the best performance, thus serving as the upper bound for comparison. On the other hand, the Random policy, which does not exploit any environment information, performs the worst, serving as the lower bound. Our proposed poLinUCB (ours) outperforms all the other strategies, including the LinUCB $(\hat{\phi})$ variant, in all three setups, showcasing its effectiveness in adaptively handling various mappings from the pre-serving context to the post-serving context. Importantly, poLinUCB delivers significantly superior performance to LinUCB $(x$ only), which operates solely based on the pre-serving context.

# 7.2 Real World Data without Ground Truth

Evaluation Setup. The evaluation was conducted on a real-world dataset, MovieLens (Harper and Konstan, 2015), where the task is to recommend movies (arms) to a incoming user (context). Following Yao et al. (2023), we first map both movies and users to 32-dimensional real vectors using a neural network trained for predicting the rating. Initially, K = 5 movies were randomly sampled to serve as our arms and were held fixed throughout the experiment. The user feature vectors were divided into two parts serving as the pre-serving context ( $d_{x} = 25$ ) and the post-serving context ( $d_{z} = 7$ ). We fit the function $\phi(\boldsymbol{x})$ using a two-layer neural network with 64 hidden units and ReLU activation. The network was trained using the Adam optimizer with a learning rate of 1e-3. At each iteration, we randomly sampled a user from the dataset and exposed only the pre-serving context x to our algorithm. The reward was computed as the dot product of the user's feature vector and the selected movie's feature vector and was revealed post the movie selection. The evaluation spanned T = 500 iterations and repeated with 10 seeds.

Results and Discussion. The experimental results, presented in Figure 2, demonstrate the effectiveness of our proposed algorithm. The overall pattern is similar to it observed in our synthetic experiments. Our proposed policy consistently outperforms the other strategies (except for Lin-UCB with both pre-serving and post-serving features). Significantly, our algorithm yields superior performance compared to policies operating solely on the pre-serving context, thereby demonstrating its effectiveness in leveraging the post-serving information.

![](images/da624b3159ac11b8b8af31a81c58c0912d4d568b967ae9e19c1b7d6bba2519f0.jpg)

<details>
<summary>line</summary>

| Time steps | Random | LinUCB (x only) | LinUCB (x and z) | LinUCB (φ̂) | poLinUCB (ours) |
| ---------- | ------ | --------------- | ---------------- | ---------- | --------------- |
| 0          | 0      | 0               | 0                | 0          | 0               |
| 100        | 100    | 80              | 40               | 60         | 30              |
| 200        | 200    | 160             | 80               | 120        | 50              |
| 300        | 300    | 240             | 120              | 180        | 70              |
| 400        | 400    | 320             | 160              | 240        | 90              |
| 500        | 500    | 400             | 200              | 300        | 110             |
</details>

Figure 2: Results on MovieLens.

# 8 Conclusions and Limitations

Conclusions. In this work, we have introduced a novel contextual bandit framework that incorporates post-serving contexts, thereby widening the range of complex real-world challenges it can address. By leveraging historical data, our proposed algorithm, poLinUCB, estimates the functional mapping from pre-serving to post-serving contexts, leading to improved online learning efficiency. For the purpose of theoretical analysis, the elliptical potential lemma has been expanded to manage noise within post-serving contexts, a development which may have wider applicability beyond this particular framework. Extensive empirical tests on synthetic and real-world datasets have demonstrated the significant benefits of utilizing post-serving contexts and the superior performance of our algorithm compared to state-of-the-art approaches.

Limitations. Our theoretical analysis hinges on a crucial assumption that the function $\phi^{\star}(\cdot)$ is learnable, which may not always be satisfied. This is particularly a concern when the post-serving contexts may hold additional information that cannot be deduced from the pre-serving context, irrespective of the amount of data collected. In such scenarios, the function mapping from the preserving context to the post-serving context may be much more difficult to learn, or even not learnable. Consequently, a linear regret may be inevitable due to model misspecification. However, from a practical point of view, our empirical findings from the real-world MovieLens dataset demonstrate that modeling the functional relationship between the pre-serving and post-serving contexts can still significantly enhance the learning efficiency. We hope our approaches can inform the design of practical algorithms that more effectively utilizes post-serving data in recommendations.

Acknowledgement. This work is supported by an NSF Award CCF-2132506, an Army Research Office Award W911NF-23-1-0030, and an Office of Naval Research Award N00014-23-1-2802.

# References

Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24, 2011.   
Naoki Abe, Alan W Biermann, and Philip M Long. Reinforcement learning with immediate rewards and linear hypotheses. Algorithmica, 37:263–293, 2003.   
Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In International Conference on Machine Learning, pages 1638–1646. PMLR, 2014.   
Jean-Yves Audibert, Rémi Munos, and Csaba Szepesvári. Exploration–exploitation tradeoff using variance estimates in multi-armed bandits. Theoretical Computer Science, 410(19):1876–1902, 2009.   
Peter Auer. Using confidence bounds for exploitation-exploration trade-offs. Journal of Machine Learning Research, 3(Nov):397–422, 2002.   
Alexandra Carpentier, Claire Vernade, and Yasin Abbasi-Yadkori. The elliptical potential lemma revisited. arXiv preprint arXiv:2010.10182, 2020.   
Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 208–214. JMLR Workshop and Conference Proceedings, 2011.   
Varsha Dani, Thomas P Hayes, and Sham M Kakade. Stochastic linear optimization under bandit feedback. 2008.   
Sarah Filippi, Olivier Cappe, Aurélien Garivier, and Csaba Szepesvári. Parametric bandits: The generalized linear case. Advances in Neural Information Processing Systems, 23, 2010.   
Nima Hamidi and Mohsen Bayati. The elliptical potential lemma for general distributions with an application to linear thompson sampling. Operations Research, 2022.   
F Maxwell Harper and Joseph A Konstan. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis), 5(4):1–19, 2015.   
Kwang-Sung Jun, Aniruddha Bhargava, Robert Nowak, and Rebecca Willett. Scalable generalized linear bandits: Online computation and hashing. Advances in Neural Information Processing Systems, 30, 2017.   
Tze Leung Lai and Ching Zong Wei. Least squares estimates in stochastic regression models with applications to identification and control of dynamic systems. The Annals of Statistics, 10(1):154–166, 1982.   
Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.   
Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pages 661–670, 2010.   
Lihong Li, Yu Lu, and Dengyong Zhou. Provably optimal algorithms for generalized linear contextual bandits. In International Conference on Machine Learning, pages 2071–2080. PMLR, 2017.   
Yingkai Li, Yining Wang, and Yuan Zhou. Nearly minimax-optimal regret for linearly parameterized bandits. In Conference on Learning Theory, pages 2173–2174. PMLR, 2019.   
Tyler Lu, Dávid Pál, and Martin Pál. Contextual multi-armed bandits. In Proceedings of the Thirteenth international conference on Artificial Intelligence and Statistics, pages 485–492. JMLR Workshop and Conference Proceedings, 2010.   
Alessandro Nuara, Francesco Trovo, Nicola Gatti, and Marcello Restelli. A combinatorial-bandit algorithm for the online joint bid/budget optimization of pay-per-click advertising campaigns. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

Hongju Park and Mohamad Kazem Shirani Faradonbeh. Analysis of thompson sampling for partially observable contextual multi-armed bandits. IEEE Control Systems Letters, 6:2150–2155, 2021.   
Yi Qi, Qingyun Wu, Hongning Wang, Jie Tang, and Maosong Sun. Bandit learning with implicit feedback. Advances in Neural Information Processing Systems, 31, 2018.   
Paat Rusmevichientong and John N Tsitsiklis. Linearly parameterized bandits. Mathematics of Operations Research, 35(2):395–411, 2010.   
Eric M Schwartz, Eric T Bradlow, and Peter S Fader. Customer acquisition via display advertising using multi-armed bandit experiments. Marketing Science, 36(4):500–522, 2017.   
Ambuj Tewari and Susan A Murphy. From ads to interventions: Contextual bandits in mobile health. Mobile Health: Sensors, Analytic Methods, and Applications, pages 495–517, 2017.   
Ryan Tibshirani. Nonparametric regression (and classification). 2017.   
Joel A Tropp et al. An introduction to matrix concentration inequalities. Foundations and Trends® in Machine Learning, 8(1-2):1–230, 2015.   
André Uschmajew. Local convergence of the alternating least squares algorithm for canonical tensor approximation. SIAM Journal on Matrix Analysis and Applications, 33(2):639–652, 2012.   
Huazheng Wang, Qingyun Wu, and Hongning Wang. Learning hidden features for contextual bandits. In Proceedings of the 25th ACM international on conference on information and knowledge management, pages 1633–1642, 2016.   
Huazheng Wang, Haifeng Xu, and Hongning Wang. When are linear stochastic bandits attackable? In International Conference on Machine Learning, pages 23254–23273. PMLR, 2022.   
Qingyun Wu, Huazheng Wang, Quanquan Gu, and Hongning Wang. Contextual bandits in a collaborative environment. In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval, pages 529–538, 2016.   
Jianyi Yang and Shaolei Ren. Robust bandit learning with imperfect context. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10594–10602, 2021.   
Shangdong Yang, Hao Wang, Chenyu Zhang, and Yang Gao. Contextual bandits with hidden features to online recommendation via sparse interactions. IEEE Intelligent Systems, 35(5):62–72, 2020.   
Yun Yang and David B Dunson. Bayesian manifold regression. 2016.   
Fan Yao, Chuanhao Li, Denis Nekipelov, Hongning Wang, and Haifeng Xu. How bad is top-k recommendation under competing content creators? International Conference on Machine Learning, 2023.   
Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural contextual bandits with ucb-based exploration. In International Conference on Machine Learning, pages 11492–11502. PMLR, 2020.   
Rong Zhu and Branislav Kveton. Robust contextual linear bandits. arXiv preprint arXiv:2210.14483, 2022.

# A Algorithm and Regret Analysis of LinUCB( $\widehat{\phi}$ )

We present the details of the algorithm described in Section 3.2 and the proof of the regret bound.

# A.1 Main Algorithm

Parameter learning. We consider solving the following regularized least squared problem for estimating $\{\widehat{\theta}_{t,a}\}_{a\in\mathcal{A}}$ and $\{\widehat{\beta}_{t,a}\}_{a\in\mathcal{A}}$ for each arm a:

$$
\ell_ {t} (\boldsymbol {\theta} _ {a}, \boldsymbol {\beta} _ {a}) = \sum_ {s: a _ {s} = a} ^ {t} \left(r _ {s, a} - \boldsymbol {x} _ {t} ^ {\top} \boldsymbol {\theta} _ {a} - \widehat {\phi} _ {s} (\boldsymbol {x} _ {s}) ^ {\top} \boldsymbol {\beta} _ {a}\right) ^ {2} + \lambda \left(\| \boldsymbol {\theta} _ {a} \| _ {2} ^ {2} + \| \boldsymbol {\beta} _ {a} \| _ {2} ^ {2}\right), \tag {10}
$$

where $\lambda \geq 0$ are penalty factors ensuring the uniqueness of minimizers $\widehat{\pmb{\theta}}_{t,a}$ and $\widehat{\beta}_{t,a}$ .

In the same convention, we use $\pmb{w}$ to denote $(\pmb{\theta},\beta)$ , and $\pmb{u}$ to denote $\left(\pmb{x},\widehat{\phi} (\pmb{x})\right)$ . The closed-form solutions for $\widehat{\pmb{\theta}}_{t,a}$ and $\widehat{\beta}_{t,a}$ in this least squared problem then become:

$$
\widehat {\boldsymbol {w}} _ {t, a} := \left[ \begin{array}{c} \widehat {\boldsymbol {\theta}} _ {t, a} \\ \widehat {\boldsymbol {\beta}} _ {t, a} \end{array} \right] = \boldsymbol {A} _ {t, a} ^ {- 1} \boldsymbol {b} _ {t, a},
$$

where we reload the notations of $A_{t,a}$ and $b_{t,a}$ ,

$$
\boldsymbol {A} _ {t, a} = \lambda \boldsymbol {I} + \sum_ {s: a _ {s} = a} ^ {t} \boldsymbol {u} _ {s} \boldsymbol {u} _ {s} ^ {\top} \quad \text { and } \quad \boldsymbol {b} _ {t, a} = \sum_ {s: a _ {s} = a} ^ {t} r _ {s, a} \boldsymbol {u} _ {s}. \tag {11}
$$

Confidence set construction. At iteration t, we construct the confidence set for $\widehat{\phi}_{t}(\boldsymbol{x}_{t})$ by

$$
\mathcal {C} _ {t} \left(\widehat {\phi} _ {t}, \boldsymbol {x} _ {t}\right) := \left\{\boldsymbol {z} \in \mathbb {R} ^ {d}: \left\| \widehat {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| _ {2} \leq e _ {t} ^ {\delta} \right\}. \tag {12}
$$

The construction of the confidence set for $\widehat{\pmb{w}}_{t,a}$ will be different, as we are using the predicted value $\widehat{\phi}_t(\cdot)$ for linear regression. Consider the following,

$$
\begin{array}{l} \boldsymbol {A} _ {t, a} \left(\left[ \begin{array}{c} \widehat {\boldsymbol {\theta}} _ {t, a} \\ \widehat {\boldsymbol {\beta}} _ {t, a} \end{array} \right] - \left[ \begin{array}{c} \boldsymbol {\theta} _ {a} ^ {\star} \\ \boldsymbol {\beta} _ {a} ^ {\star} \end{array} \right]\right) = \underbrace {\sum_ {s : a _ {s} = a} ^ {t} \left(\boldsymbol {\epsilon} _ {s} ^ {\top} \boldsymbol {\beta} _ {a} ^ {\star} \left[ \begin{array}{c} \boldsymbol {x} _ {s} \\ \widehat {\phi} _ {s} (\boldsymbol {x} _ {s}) \end{array} \right]\right)} _ {(1)} + \underbrace {\sum_ {s : a _ {s} = a} ^ {t} \left(\left(\phi^ {\star} (\boldsymbol {x} _ {s}) - \widehat {\phi} _ {s} (\boldsymbol {x} _ {s})\right) ^ {\top} \boldsymbol {\beta} _ {a} ^ {\star} \left[ \begin{array}{c} \boldsymbol {x} _ {s} \\ \widehat {\phi} _ {s} (\boldsymbol {x} _ {s}) \end{array} \right]\right)} _ {(2)} \\ + \underbrace {\sum_ {s : a _ {s} = a} ^ {t} \eta_ {s} \left[ \widehat {\phi} _ {s} (\boldsymbol {x} _ {s}) \right]} _ {(3)} - \underbrace {\lambda \left[ \begin{array}{c} \boldsymbol {\theta} _ {a} ^ {\star} \\ \boldsymbol {\beta} _ {a} ^ {\star} \end{array} \right]} _ {(4)} \\ \end{array}
$$

Therefore, the confidence set will be enlarged due to the error introduced by $\widehat{\phi}_t(\cdot)$ . In the below, we derive the confidence set. To build the confidence set, we need to bound

$$
\| \widehat {\boldsymbol {w}} _ {t, a} - \boldsymbol {w} _ {t, a} ^ {\star} \| _ {\boldsymbol {A} _ {t, a}} = \left\| \left[ \begin{array}{c} \widehat {\boldsymbol {\theta}} _ {t, a} \\ \widehat {\boldsymbol {\beta}} _ {t, a} \end{array} \right] - \left[ \begin{array}{c} \boldsymbol {\theta} _ {a} ^ {\star} \\ \boldsymbol {\beta} _ {a} ^ {\star} \end{array} \right] \right\| _ {\boldsymbol {A} _ {t, a}} = \left\| ① + ② + ③ + ④ \right\| _ {\boldsymbol {A} _ {t, a} ^ {- 1}}
$$

Since both $\{\epsilon_s\}_{s=1}^t$ and $\{\eta_s\}_{s=1}^t$ are i.i.d sub-Gaussian random variables, respectively, we can use the self-normalized inequality to bound the corresponding terms, i.e., the followings hold with probability at least $1 - 2\delta$ ,

$$
\begin{array}{l} \| \widehat {\pmb {w}} _ {t, a} - \pmb {w} _ {t, a} ^ {\star} \| _ {\pmb {A} _ {t, a}} \leq \sqrt {2 (L _ {\epsilon} ^ {2} + R _ {\eta} ^ {2}) \log \left(\frac {\det (\pmb {A} _ {t , a}) ^ {1 / 2} \det (\lambda \pmb {I}) ^ {- 1 / 2}}{\delta / 2}\right)} + \frac {L _ {u}}{\sqrt {\lambda}} \left(\sum_ {s = 1} ^ {t} e _ {s} ^ {\delta / t}\right) + 2 \sqrt {\lambda} \\ \leq \sqrt {2 (L _ {\epsilon} ^ {2} + R _ {\eta} ^ {2}) \log \left(\frac {1 + n _ {t} (a) L _ {u} ^ {2} / \lambda}{\delta / 2}\right)} + \frac {L _ {u}}{\sqrt {\lambda}} \left(\sum_ {s = 1} ^ {t} e _ {s} ^ {\delta / t}\right) + 2 \sqrt {\lambda} \\ \end{array}
$$

Algorithm 2 LinUCB- $(\widehat{\phi})$ (Linear UCB adapted from Wang et al. (2016) with post-serving contexts; The differences with Algorithm 1 are highlighted in blue color.)

1: for $t = 0,1,\ldots ,T$ do   
2: Receive the pre-serving context $\pmb{x}_t$   
3: Compute the optimistic parameters by maximizing the UCB objective

$$
\Big (a _ {t}, \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}), \widetilde {\boldsymbol {w}} _ {t} \Big) = \underset {(a, \phi , \boldsymbol {w} _ {a}) \in [ K ] \times \mathcal {C} _ {t - 1} (\widehat {\phi} _ {t - 1}, \boldsymbol {x} _ {t}) \times \mathcal {C} _ {t - 1} (\widehat {\boldsymbol {w}} _ {t - 1, a})} {\arg \max} \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi (\boldsymbol {x} _ {t}) \end{array} \right] ^ {\top} \boldsymbol {w} _ {a}.
$$

4: Play the arm $a_{t}$ and receive the realized post-serving context as $z_{t}$ and the real-valued reward

$$
r _ {t, a _ {t}} = \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \boldsymbol {z} _ {t} \end{array} \right] ^ {\top} \boldsymbol {w} _ {a _ {t}} ^ {\star} + \eta_ {t}.
$$

5: Compute the estimated post-serving context generating function $\widehat{\phi}_t(\cdot)$ using ERM.

6: Compute $\widehat{w}_{t,a}$ by solving Equation 10 for each a.

7: Update confidence sets $\mathcal{C}_t(\widehat{\boldsymbol{w}}_{t,a})$ and $\mathcal{C}_t(\widehat{\phi}_t,\boldsymbol{x}_t)$ for each $a$ based on Equations 12 and 5.

8: end for

Therefore, the confidence set is

$$
\mathcal {C} _ {t, a} (\widehat {\boldsymbol {w}} _ {t, a}) = \left\{\boldsymbol {w} \in \mathbb {R} ^ {d _ {u}}: \| \boldsymbol {w} - \widehat {\boldsymbol {w}} _ {t, a} \| _ {\boldsymbol {A} _ {t, a}} \leq \zeta_ {t, a} \right\}, \tag {13}
$$

where $\zeta_{t,a} = \sqrt{2(L_{\epsilon}^{2} + R_{\eta}^{2})\log\left((1 + n_{t}(a)L_{u}^{2}/\lambda)/(\delta/2)\right)} + L_{u}\left(\sum_{s=1}^{t}e_{s}^{\delta/t}\right)/\sqrt{\lambda} + 2\sqrt{\lambda}$ . In comparison to the original confidence set, there is one additional term due to the generalization error introduced from $\widehat{\phi}_{s}(\cdot)$ . In the next section, we will provide a regret analysis, which following from the proof of LinUCB (Li et al., 2010). We simply have the following

$$
\begin{array}{l} R _ {T} = \sum_ {t = 1} ^ {T} \left(r _ {t, a _ {t} ^ {\star}} - r _ {t, a _ {t}}\right) = \sum_ {t = 1} ^ {T} \Delta_ {t} \leq \sqrt {T \sum_ {t = 1} ^ {T} \Delta_ {t} ^ {2}} \\ \leq \sqrt {T \sum_ {t = 1} ^ {T} \left(\left\| \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| + \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1 , a _ {t}} ^ {- 1}} \left\| \left[ \begin{array}{c} \widetilde {\boldsymbol {\theta}} _ {a _ {t}} - \boldsymbol {\theta} _ {a _ {t}} \\ \widetilde {\boldsymbol {\beta}} _ {a _ {t}} - \boldsymbol {\beta} _ {a _ {t}} \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1 , a _ {t}}}\right) ^ {2}} \\ \leq \sqrt {T \left(\sum_ {t = 1} ^ {T} 2 \left\| \widetilde {\phi_ {t}} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| ^ {2} \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| ^ {2} + 2 \zeta_ {T , a} ^ {2} \left(1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1 , a _ {t}} ^ {- 1}} ^ {2}\right)\right)} \\ \leq \sqrt {T \left(\sum_ {t = 1} ^ {T} 2 \left\| \widetilde {\phi_ {t}} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| ^ {2} \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| ^ {2} + 2 \zeta_ {T , a} ^ {2} \left(1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1 , a _ {t}} ^ {- 1}} ^ {2}\right)\right)} \\ \leq \sqrt {T \cdot \left(8 C _ {0} T ^ {1 - 2 \alpha} \log^ {2 \alpha} \left(\frac {\det \boldsymbol {X} _ {t}}{\det \boldsymbol {X} _ {0}}\right) \log^ {2} \left(\frac {T}{\delta}\right) + 2 K \zeta_ {T} ^ {2} d _ {u} \log \left(1 + \frac {T L _ {u} ^ {2}}{\lambda d _ {u}}\right)\right)} \\ \end{array}
$$

In the next, we expand the term $\zeta_T^2$ ,

$$
\begin{array}{l} \zeta_ {T} ^ {2} \leq \left(\sqrt {2 (L _ {\epsilon} ^ {2} + R _ {\eta} ^ {2}) \log \left(\frac {1 + T L _ {u} ^ {2} / \lambda}{\delta / 2}\right)} + \frac {L _ {u} \left(\sum_ {s = 1} ^ {T} e _ {s} ^ {\delta / T}\right)}{\sqrt {\lambda}} + 2 \sqrt {\lambda}\right) ^ {2} \\ \leq 6 (L _ {\epsilon} ^ {2} + R _ {\eta} ^ {2}) \log \left(\frac {1 + T L _ {u} ^ {2} / \lambda}{\delta / 2}\right) + \frac {3 L _ {u} ^ {2}}{\lambda} \left(\sum_ {s = 1} ^ {T} e _ {s} ^ {\delta / T}\right) ^ {2} + 1 2 \lambda \\ \end{array}
$$

In the next, we bound the second term in the above equation under the learnability assumption 1,

$$
\left(\sum_ {s = 1} ^ {T} e _ {s} ^ {\delta / T}\right) ^ {2} \leq 1 6 T ^ {2 - 2 \alpha} \log^ {2 \alpha} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right) \log^ {2} \left(\frac {T}{\delta}\right).
$$

Therefore, naively choosing the value of $\lambda$ will lead to a linear regret due to the term $T^{3/2 - \alpha}$ in the equation. To minimize the upper bound, we can choose the value of $\lambda$ to be

$$
\lambda = 2 L _ {u} T ^ {1 - \alpha} \log^ {\alpha} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right) \log \left(\frac {T}{\delta}\right).
$$

Then, we can bound $\zeta_T^2$ by

$$
\zeta_ {T} ^ {2} \leq 6 (L _ {\epsilon} ^ {2} + R _ {\eta} ^ {2}) \log \left(\frac {1 + T L _ {u} ^ {2} / \lambda}{\delta / 2}\right) + 4 8 L _ {u} T ^ {1 - \alpha} \log^ {\alpha} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right) \log \left(\frac {T}{\delta}\right).
$$

By plugging it in and following the simplication as in the proof of Theorem 1, we can get the regret is upper bounded by

$$
\widetilde {\mathcal {O}} \left(T ^ {1 - \alpha} d _ {u} ^ {\alpha} + T ^ {1 - \alpha / 2} \sqrt {K d _ {u} ^ {1 + \alpha}}\right).
$$

The above result is summarized as the following proposition.

Proposition 1 (Regret of LinUCB-( $\widehat{\phi}$ )). The regret of LinUCB-( $\widehat{\phi}$ ) in Algorithm 2 is upper bounded by $\widetilde{\mathcal{O}}\left(T^{1-\alpha}d_{u}^{\alpha}+T^{1-\alpha/2}\sqrt{Kd_{u}^{1+\alpha}}\right)$ with probability at least $1-\delta$ , by carefully setting the regularization coefficient $\lambda=\Theta(L_{u}d_{u}^{\alpha}T^{1-\alpha}\log(T/\delta))$ in Equation 1.

# B Missing Proofs

# B.1 Missing Proofs in the Generalized Elliptical Potential Lemma

Lemma 1 (Generalized Elliptical Potential Lemma). Suppose (1) $\mathbf{X}_0 \in \mathbb{R}^{d \times d}$ is any positive definite matrix; (2) $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \mathbb{R}^d$ is a sequence of vectors with bounded $l_2$ norm $\max_t \| \mathbf{x}_t \| \leq L_x$ ; (3) $\epsilon_1, \ldots, \epsilon_T \in \mathbb{R}^d$ is a sequence of independent (not necessarily identical) bounded zero-mean noises satisfying $\max_t \| \epsilon_t \| \leq L_\epsilon$ and $\mathbb{E}[\epsilon_t \epsilon_t^\top] \succcurlyeq \sigma_\epsilon^2 I$ for any $t$ ; and (4) $\widetilde{\mathbf{X}}_t$ is defined as follows:

$$
\widetilde {\boldsymbol {X}} _ {t} = \boldsymbol {X} _ {0} + \sum_ {s = 1} ^ {t} (\boldsymbol {x} _ {s} + \boldsymbol {\epsilon} _ {s}) (\boldsymbol {x} _ {s} + \boldsymbol {\epsilon} _ {s}) ^ {\top} \in \mathbb {R} ^ {d \times d}.
$$

Then, for any $p \in [0,1]$ , the following inequality holds with probability at least $1 - \delta$ ,

$$
\sum_ {t = 1} ^ {T} \left(1 \wedge \| \boldsymbol {x} _ {t} \| _ {\widetilde {\boldsymbol {X}} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \leq 2 ^ {p} T ^ {1 - p} \log^ {p} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right) + \frac {8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\sigma_ {\epsilon} ^ {4}} \log \left(\frac {3 2 d L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\delta \sigma_ {\epsilon} ^ {4}}\right) \tag {2}
$$

Proof. Our proof follows the high level idea for proving the original EPL. However, to accommodate the noises in the data matrix, we have to introduce new matrix concentration tools to the original (primarily algebraic) proof, and also identify the right conditions for the argument to go through. A key lemma to our proof is the following high probability bound regarding the noisy data matrix:

Lemma 2. Let $\pmb{x}_1, \dots, \pmb{x}_T \in \mathbb{R}^d$ be a fixed sequence of vectors, and $\epsilon_1, \dots, \epsilon_T \in \mathbb{R}^d$ are independent random variables satisfying $\max_t \| \pmb{x}_t\| _2 \leq L_x$ , $\max_t \| \pmb{\epsilon}_t\| _2 \leq L_\epsilon$ , and $\mathbb{E}[\pmb{\epsilon}_t\pmb{\epsilon}_t^\top] \succcurlyeq \sigma_\epsilon^2\pmb{I}$ . Then the following hold with probability at least $1 - 2d\exp \left(\frac{-T\sigma_\epsilon^4}{8L_\epsilon^2(L_\epsilon + L_x)^2}\right)$ ,

$$
\sum_ {t = 1} ^ {T} \left(\boldsymbol {x} _ {t} + \boldsymbol {\epsilon} _ {t}\right) \left(\boldsymbol {x} _ {t} + \boldsymbol {\epsilon} _ {t}\right) ^ {\top} \succcurlyeq \sum_ {t = 1} ^ {T} \boldsymbol {x} _ {t} \boldsymbol {x} _ {t} ^ {\top}.
$$

The Proof of Lemma 2 employs the Bernstein's Inequality for matrices (Tropp et al., 2015), which is technical; for ease of presentation, we defer its proof of Appendix B.1. By Lemma 2, we have that, for every $t \in [T]$ , the following inequality holds with probability at least $1 - 2d\exp \left(\frac{-t\sigma_{\epsilon}^{4}}{8L_{\epsilon}^{2}(L_{\epsilon} + L_{x})^{2}}\right)$ :

$$
\widetilde {\boldsymbol {X}} _ {t} = \boldsymbol {X} _ {0} + \sum_ {s = 1} ^ {t} (\boldsymbol {x} _ {s} + \boldsymbol {\epsilon} _ {s}) (\boldsymbol {x} _ {s} + \boldsymbol {\epsilon} _ {s}) ^ {\top} \succcurlyeq \boldsymbol {X} _ {0} + \sum_ {s = 1} ^ {t} \boldsymbol {x} _ {s} \boldsymbol {x} _ {s} ^ {\top} := \boldsymbol {X} _ {t},
$$

under which we have

$$
\left\| \boldsymbol {x} _ {t + 1} \right\| _ {\widetilde {\boldsymbol {X}} _ {t} ^ {- 1}} ^ {2} \leq \left\| \boldsymbol {x} _ {t + 1} \right\| _ {\boldsymbol {X} _ {t} ^ {- 1}} ^ {2}.
$$

To prove our Lemma 1, we need to apply union bound to guarantee the above hold simultaneously for every $t \geq 1$ with high probability. Unfortunately, this turns out to not be true because when $t$ is very small (e.g., $t = 1$ ), the above inequality cannot hold with high probability. Therefore, to obtain high-probability guarantee by the union bound, we will have to exclude these small $t$ 's and apply the union bound for only the events from some $t' \in [T]$ , as follows

$$
\begin{array}{l} \mathbb {P} \left[ \forall t \in [ t ^ {\prime}, T ], \| \pmb {x} _ {t} \| _ {\widetilde {\pmb {X}} _ {t - 1} ^ {- 1}} ^ {2} \leq \| \pmb {x} _ {t} \| _ {\pmb {X} _ {t - 1} ^ {- 1}} ^ {2} \right] \\ \geq 1 - \sum_ {t = t ^ {\prime} - 1} ^ {T - 1} 2 d \exp \left(\frac {- t \sigma_ {\epsilon} ^ {4}}{(8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2})}\right) \\ \geq 1 - \sum_ {t = t ^ {\prime} - 1} ^ {\infty} 2 d \exp \left(\frac {- t \sigma_ {\epsilon} ^ {4}}{(8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2})}\right) \\ = 1 - 2 d \left(\frac {\exp \big (- (t ^ {\prime} - 1) \sigma_ {\epsilon} ^ {4} / (8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}) \big)}{1 - \exp \big (- \sigma_ {\epsilon} ^ {4} / (8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}) \big)}\right). \\ \geq 1 - 2 d \times \exp \big (- (t ^ {\prime} - 1) \sigma_ {\epsilon} ^ {4} / (8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}) \big) \times \frac {1 6 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\sigma_ {\epsilon} ^ {4}}, \\ \end{array}
$$

where the last inequality uses the fact that $\sigma_{\epsilon}^{4} / (L_{\epsilon}^{2}(L_{\epsilon} + L_{x})^{2})\leq (\sigma_{\epsilon} / L_{\epsilon})^{4}\leq 1$ and $1 - e^{-x}\geq x / 2$ for any $x\in [0,1]$ . By solving the following inequality,

$$
\exp \big (- (t ^ {\prime} - 1) \sigma_ {\epsilon} ^ {4} / (8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}) \big) \times \frac {1 6 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\sigma_ {\epsilon} ^ {4}} \leq \frac {\delta}{2 d},
$$

we have,

$$
t ^ {\prime} \geq 1 + \frac {8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\sigma_ {\epsilon} ^ {4}} \log \left(\frac {3 2 d L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\delta \sigma_ {\epsilon} ^ {4}}\right)
$$

Let $T_0$ denote the ceiling of the right-hand-side of the above term. Therefore, we have the following hold with high probability at least $1 - \delta$ :

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} \left(1 \wedge \| \boldsymbol {x} _ {t} \| _ {\tilde {\boldsymbol {X}} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \leq (T _ {0} - 1) + \sum_ {t = T _ {0}} ^ {T} \left(1 \wedge \| \boldsymbol {x} _ {t} \| _ {\tilde {\boldsymbol {X}} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \\ \leq (T _ {0} - 1) + \sum_ {t = T _ {0}} ^ {T} \left(1 \wedge \| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \\ \leq \left(T _ {0} - 1\right) + \sum_ {t = T _ {0}} ^ {T} \left(1 \wedge \| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \\ \end{array}
$$

In the next, we are going to bound the second term in the above equation, whose proof can be adapted from the proof of the original elliptical potential lemma. Using the fact that for any $z \in [0, +\infty]$ , $z \wedge 1 \leq 2 \ln(1 + z)$ , we get

$$
\sum_ {t = 1} ^ {T} 1 \wedge \left(\| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \leq \sum_ {t = 1} ^ {T} \left(2 \log \left(1 + \| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right)\right) ^ {p}.
$$

Additionally, by definition, we have

$$
\boldsymbol {X} _ {t} = \boldsymbol {X} _ {t - 1} + \boldsymbol {x} _ {t} \boldsymbol {x} _ {t} ^ {\top} = \boldsymbol {X} _ {t - 1} ^ {1 / 2} \left(\boldsymbol {I} + \boldsymbol {X} _ {t - 1} ^ {- 1 / 2} \boldsymbol {x} _ {t} \boldsymbol {x} _ {t} ^ {\top} \boldsymbol {X} _ {t - 1} ^ {- 1 / 2}\right) \boldsymbol {X} _ {t - 1} ^ {1 / 2}.
$$

This implies the following relationship between the determinant,

$$
\det \boldsymbol {X} _ {t} = \det \left(\boldsymbol {X} _ {t - 1}\right) \det \left(\boldsymbol {I} + \boldsymbol {X} _ {t - 1} ^ {- 1 / 2} \boldsymbol {x} _ {t} \boldsymbol {x} _ {t} ^ {\top} \boldsymbol {X} _ {t - 1} ^ {- 1 / 2}\right).
$$

Since the only eigenvalues of a matrix of the form $I + yy^{\top}$ are $1 + \|y\|_{2}$ and 1, we have

$$
\log \left(1 + \| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right) = \log \det \boldsymbol {X} _ {t} - \log \det \boldsymbol {X} _ {t - 1}.
$$

By taking the power $p$ for both sides and taking the sum, we have

$$
\sum_ {t = 1} ^ {T} \left(\log \left(1 + \| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right)\right) ^ {p} = \sum_ {t = 1} ^ {T} (\log \det \boldsymbol {X} _ {t} - \log \det \boldsymbol {X} _ {t - 1}) ^ {p}.
$$

Since $p \in [0,1]$ , the function $g(x) = x^{p}$ is a concave function. Thus, we have

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} (\log \det \boldsymbol {X} _ {t} - \log \det \boldsymbol {X} _ {t - 1}) ^ {p} \leq \left(\frac {1}{T} \sum_ {t = 1} ^ {T} \log \det \boldsymbol {X} _ {t} - \log \det \boldsymbol {X} _ {t - 1}\right) ^ {p} = \frac {1}{T ^ {p}} \log^ {p} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right).
$$

Therefore, we can conclude that

$$
\sum_ {t = 1} ^ {T} 1 \wedge \left(\| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \leq 2 ^ {p} T ^ {1 - p} \log^ {p} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right).
$$

By combining the above results, we have the following hold with probability at least $1 - \delta$ :

$$
\sum_ {t = 1} ^ {T} \left(1 \wedge \| \boldsymbol {x} _ {t} \| _ {\widetilde {\boldsymbol {X}} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \leq T _ {0} - 1 + \left(\sum_ {t = 1} ^ {T} 1 \wedge \| \boldsymbol {x} _ {t} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \leq T _ {0} - 1 + 2 ^ {p} T ^ {1 - p} \log^ {p} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right).
$$

Invoking

$$
T _ {0} - 1 \leq \frac {8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\sigma_ {\epsilon} ^ {4}} \log \left(\frac {3 2 d L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\delta \sigma_ {\epsilon} ^ {4}}\right),
$$

we obtained the desired inequality with probability at least $1 - \delta$ :

$$
\sum_ {t = 1} ^ {T} \left(1 \wedge \| \pmb {x} _ {t} \| _ {\widetilde {\pmb {X}} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {p} \leq 2 ^ {p} T ^ {1 - p} \log^ {p} \left(\frac {\det \pmb {X} _ {T}}{\det \pmb {X} _ {0}}\right) + \frac {8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\sigma_ {\epsilon} ^ {4}} \log \left(\frac {3 2 d L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\delta \sigma_ {\epsilon} ^ {4}}\right).
$$

![](images/ac9d929236cf39bbe7c47368343b7510b9fb97c90401052d1486c88feea0338a.jpg)

Lemma 2. Let $\pmb{x}_1, \dots, \pmb{x}_T \in \mathbb{R}^d$ be a fixed sequence of vectors, and $\epsilon_1, \dots, \epsilon_T \in \mathbb{R}^d$ are independent random variables satisfying $\max_t \| \pmb{x}_t\| _2 \leq L_x$ , $\max_t \| \pmb{\epsilon}_t\| _2 \leq L_\epsilon$ , and $\mathbb{E}[\pmb{\epsilon}_t\pmb{\epsilon}_t^\top] \succcurlyeq \sigma_\epsilon^2\pmb{I}$ . Then the following hold with probability at least $1 - 2d\exp \left(\frac{-T\sigma_\epsilon^4}{8L_\epsilon^2(L_\epsilon + L_x)^2}\right)$ ,

$$
\sum_ {t = 1} ^ {T} \left(\boldsymbol {x} _ {t} + \boldsymbol {\epsilon} _ {t}\right) \left(\boldsymbol {x} _ {t} + \boldsymbol {\epsilon} _ {t}\right) ^ {\top} \succcurlyeq \sum_ {t = 1} ^ {T} \boldsymbol {x} _ {t} \boldsymbol {x} _ {t} ^ {\top}.
$$

Proof. We will analyze the term-wise difference, denoted as

$$
\boldsymbol {S} _ {t} := \left(\boldsymbol {x} _ {t} + \boldsymbol {\epsilon} _ {t}\right) \left(\boldsymbol {x} _ {t} + \boldsymbol {\epsilon} _ {t}\right) ^ {\top} - \boldsymbol {x} _ {t} \boldsymbol {x} _ {t} ^ {\top} = \boldsymbol {\epsilon} _ {t} \boldsymbol {x} _ {t} ^ {\top} + \boldsymbol {x} _ {t} \boldsymbol {\epsilon} _ {t} ^ {\top} + \boldsymbol {\epsilon} _ {t} \boldsymbol {\epsilon} _ {t} ^ {\top} \tag {14}
$$

Moreover, since $\mathbb{E}[\pmb{\epsilon}_t] = 0$ for any $t\in [T]$ , the expectation of $S_{t}$ (over randomness of noise) can be lower bounded as

$$
\mathbb {E} [ \boldsymbol {S} _ {t} ] = \mathbb {E} [ \boldsymbol {\epsilon} _ {t} ] \boldsymbol {x} _ {t} ^ {\top} + \boldsymbol {x} _ {t} \mathbb {E} [ \boldsymbol {\epsilon} _ {t} ^ {\top} ] + \mathbb {E} [ \boldsymbol {\epsilon} _ {t} \boldsymbol {\epsilon} _ {t} ^ {\top} ] = \mathbb {E} [ \boldsymbol {\epsilon} _ {t} \boldsymbol {\epsilon} _ {t} ^ {\top} ] \succcurlyeq \sigma_ {\epsilon} ^ {2} \boldsymbol {I}.
$$

Since $\|x_{t}\|_{2}\leq L_{x}$ and $\|\epsilon_{t}\|_{2}\leq L_{\epsilon}$ , we know that $S_{t}\preccurlyeq(2L_{\epsilon}L_{x}+L_{\epsilon}^{2})I$ with probability 1. Thus $S_{t}$ is uniformly upper bounded under the spectral-norm denoted by $\|\cdot\|$ , or formally

$$
\| \boldsymbol {S} _ {t} \| \leq 2 L _ {\epsilon} L _ {x} + L _ {\epsilon} ^ {2}.
$$

Consider the “centered” matrix sum $Z_{T} = \sum_{t=1}^{T} \left[ S_{t} - \mathbb{E}[\epsilon_{t}\epsilon_{t}^{\top}] \right]$ , with mean 0. Since the spectral-norm of $S_{t}$ is upper bounded by $2L_{\epsilon}L_{x} + L_{\epsilon}^{2}$ , its variance $\mathbb{V}(S_{t}) = \left\| \mathbb{E}\left[S_{t}S_{t}^{\top}\right] \right\|_{2}$ is upper bounded by $(2L_{\epsilon}L_{x} + L_{\epsilon}^{2})^{2}$ . Thus, the variance of $Z_{T}$ equals the variance of sum $\sum_{t=1}^{T} S_{t}$ , which is then upper bounded by $T(2L_{\epsilon}L_{x} + L_{\epsilon}^{2})^{2}$ . By the Bernstein’s Inequality for random matrices (Tropp et al., 2015), we have the following high probability upper bound for the spectral norm $\|\cdot\|$ of $Z_{T}$ :

$$
\mathbb {P} \left[ \| \boldsymbol {Z} _ {T} \| \geq \iota \right] \leq 2 d \exp \left(\frac {- \iota^ {2} / 2}{\mathbb {V} (\boldsymbol {Z} _ {T}) + (2 L _ {\epsilon} L _ {x} + L _ {\epsilon} ^ {2}) \iota / 3}\right). \tag {15}
$$

Since $Z_{T}$ is a symmetric matrix, its spectral norm upper bounds the absolute value of any eigenvalue. Thus we can lower bound the smallest eigenvalues as follows:

$$
\begin{array}{l} \mathbb {P} \left[ \| \boldsymbol {Z} _ {T} \| \geq \iota \right] \geq \mathbb {P} \left[ \lambda_ {\min} \left(\sum_ {t = 1} ^ {T} \boldsymbol {S} _ {t} - \mathbb {E} [ \boldsymbol {\epsilon} _ {t} \boldsymbol {\epsilon} _ {t} ^ {\top} ]\right) \leq - \iota \right] \\ \geq \mathbb {P} \left[ \lambda_ {\min} \left(\sum_ {t = 1} ^ {T} \boldsymbol {S} _ {t} - \sigma_ {\epsilon} ^ {2} \boldsymbol {I}\right) \leq - \iota \right] \\ = \mathbb {P} \left[ \lambda_ {\min} \left(\sum_ {t = 1} ^ {T} \boldsymbol {S} _ {t}\right) \leq T \sigma_ {\epsilon} ^ {2} - \iota \right]. \\ \end{array}
$$

where the second inequality is due to two facts: (1) $A \succcurlyeq B$ implies $\lambda_{\min}(A) \geq \lambda_{\min}(B)$ where $A = \left(\sum_{t=1}^{T} S_t - \sigma_\epsilon^2 I\right)$ and $B = \left(\sum_{t=1}^{T} S_t - \mathbb{E}[\epsilon_t \epsilon_t^\top]\right)$ ; and (2) the event $\lambda_{\min}(A) \leq -\iota$ thus is included within the event $\lambda_{\min}(B) \leq -\iota$ . Consequently, we have

$$
\mathbb {P} \left[ \lambda_ {\min} \left(\sum_ {t = 1} ^ {T} \boldsymbol {S} _ {t}\right) \leq T \sigma_ {\epsilon} ^ {2} - \iota \right] \leq 2 d \exp \left(\frac {- \iota^ {2} / 2}{\mathbb {V} (\boldsymbol {Z} _ {T}) + (2 L _ {\epsilon} L _ {x} + L _ {\epsilon} ^ {2}) \iota / 3}\right),
$$

or equivalently,

$$
\mathbb {P} \left[ \lambda_ {\min} \left(\sum_ {t = 1} ^ {T} \boldsymbol {S} _ {t}\right) \geq T \sigma_ {\epsilon} ^ {2} - \iota \right] \geq 1 - 2 d \exp \left(\frac {- \iota^ {2} / 2}{\mathbb {V} (\boldsymbol {Z} _ {T}) + (2 L _ {\epsilon} L _ {x} + L _ {\epsilon} ^ {2}) \iota / 3}\right),
$$

By choosing the value of $\iota = T\sigma_{\epsilon}^{2}$ , we get

$$
\begin{array}{l} \mathbb {P} \left[ \lambda_ {\min} \left(\sum_ {t = 1} ^ {T} \boldsymbol {S} _ {t}\right) \geq 0 \right] \\ \geq 1 - 2 d \exp \left(\frac {- T ^ {2} \sigma_ {\epsilon} ^ {4} / 2}{\mathbb {V} (\boldsymbol {Z} _ {T}) + (2 L _ {\epsilon} L _ {x} + L _ {\epsilon} ^ {2}) T \sigma_ {\epsilon} ^ {2} / 3}\right) \\ \geq 1 - 2 d \exp \left(\frac {- T ^ {2} \sigma_ {\epsilon} ^ {4} / 2}{T (2 L _ {\epsilon} L _ {x} + L _ {\epsilon} ^ {2}) ^ {2} + (2 L _ {\epsilon} L _ {x} + L _ {\epsilon} ^ {2}) T \sigma_ {\epsilon} ^ {2} / 3}\right). \\ \end{array}
$$

Using the fact that $\sigma_{\epsilon} \leq L_{\epsilon}$ , we can further simplify the above equation by

$$
\mathbb {P} \left[ \lambda_ {\min} \left(\sum_ {t = 1} ^ {T} \boldsymbol {S} _ {t}\right) \geq 0 \right] \geq 1 - 2 d \exp \left(\frac {- T \sigma_ {\epsilon} ^ {4}}{8 L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}\right).
$$

□

# B.2 Missing Proofs in the Regret Analysis

Theorem 1 (Regret of poLinUCB). The regret of poLinUCB in Algorithm 1 is upper bounded by $\widetilde{\mathcal{O}}\left(T^{1 - \alpha}d_u^\alpha +d_u\sqrt{TK}\right)$ with probability at least $1 - \delta$ , if $T = \Omega (\log (1 / \delta))$ .

Proof. In the next, we prove the regret bound. For each time step t, the immediate regret is

$$
\begin{array}{l} \Delta_ {t} = r _ {t, a _ {t} ^ {\star}} - r _ {t, a _ {t}} \\ = \left\langle \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right], \left[ \begin{array}{c} \boldsymbol {\theta} _ {a _ {t} ^ {\star}} - \boldsymbol {\theta} _ {a _ {t}} ^ {\star} \\ \boldsymbol {\beta} _ {a _ {t} ^ {\star}} - \boldsymbol {\beta} _ {a _ {t}} ^ {\star} \end{array} \right] \right\rangle \\ \stackrel {\text {(a)}} {\leq} \left\langle \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) \end{array} \right], \left[ \begin{array}{c} \widetilde {\boldsymbol {\theta}} _ {a _ {t}} - \boldsymbol {\theta} _ {a _ {t}} ^ {\star} \\ \widetilde {\boldsymbol {\beta}} _ {a _ {t}} - \boldsymbol {\beta} _ {a _ {t}} ^ {\star} \end{array} \right] \right\rangle + \left\langle \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}), \boldsymbol {\beta} _ {a _ {t}} ^ {\star} \right\rangle \\ = \left\langle \left[ \begin{array}{c} \mathbf {0} \\ \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] + \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right], \left[ \begin{array}{c} \widetilde {\boldsymbol {\theta}} _ {a _ {t}} - \boldsymbol {\theta} _ {a _ {t}} ^ {\star} \\ \widetilde {\boldsymbol {\beta}} _ {a _ {t}} - \boldsymbol {\beta} _ {a _ {t}} ^ {\star} \end{array} \right] \right\rangle + \left\langle \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}), \boldsymbol {\beta} _ {a _ {t}} ^ {\star} \right\rangle \\ = \left\langle \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}), \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\rangle + \left\langle \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right], \left[ \begin{array}{c} \widetilde {\boldsymbol {\theta}} _ {a _ {t}} - \boldsymbol {\theta} _ {a _ {t}} ^ {\star} \\ \widetilde {\boldsymbol {\beta}} _ {a _ {t}} - \boldsymbol {\beta} _ {a _ {t}} ^ {\star} \end{array} \right] \right\rangle \\ \stackrel {{(\mathrm{b})}} {{\leq}} \left\| \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| \cdot \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| + \left\langle \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right], \left[ \begin{array}{c} \widetilde {\boldsymbol {\theta}} _ {a _ {t}} - \boldsymbol {\theta} _ {a _ {t}} ^ {\star} \\ \widetilde {\boldsymbol {\beta}} _ {a _ {t}} - \boldsymbol {\beta} _ {a _ {t}} ^ {\star} \end{array} \right] \right\rangle \\ \stackrel {{(\mathrm{c})}} {{\leq}} \left\| \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| \cdot \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| + \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1, a _ {t}} ^ {- 1}} \left\| \left[ \begin{array}{c} \widetilde {\boldsymbol {\theta}} _ {a _ {t}} - \boldsymbol {\theta} _ {a _ {t}} ^ {\star} \\ \widetilde {\boldsymbol {\beta}} _ {a _ {t}} - \boldsymbol {\beta} _ {a _ {t}} ^ {\star} \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1, a _ {t}}}. \\ \end{array}
$$

where the inequality (a) is due to the definition of UCB, and (b) and (c) are obtained using the Cauchy-Schwarz inequality. Therefore, the cumulative regret can be further upper bounded by

$$
\begin{array}{l} R _ {T} = \sum_ {t = 1} ^ {T} \Delta_ {t} \leq \sqrt {T \sum_ {t = 1} ^ {T} \Delta_ {t} ^ {2}} \\ \leq \sqrt {T \sum_ {t = 1} ^ {T} \left(\left\| \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| + \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1 , a _ {t}} ^ {- 1}} \left\| \left[ \begin{array}{c} \widetilde {\boldsymbol {\theta}} _ {a _ {t}} - \boldsymbol {\theta} _ {a _ {t}} \\ \widetilde {\boldsymbol {\beta}} _ {a _ {t}} - \boldsymbol {\beta} _ {a _ {t}} \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1 , a _ {t}}}\right) ^ {2}} \\ \leq \sqrt {T \left(\sum_ {t = 1} ^ {T} 2 \left\| \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| ^ {2} \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| ^ {2} + 2 \zeta_ {T} ^ {2} \left(1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1 , a _ {t}} ^ {- 1}} ^ {2}\right)\right)} \\ \leq \sqrt {T \left(\sum_ {t = 1} ^ {T} 2 \left\| \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| ^ {2} \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| ^ {2} + 2 \zeta_ {T} ^ {2} \left(1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1 , a _ {t}} ^ {- 1}} ^ {2}\right)\right)} \\ \end{array}
$$

In the next, we bound each term separately. Firstly, we have the following hold with probability at least $1 - \delta$ by using the union bound,

$$
\sum_ {t = 1} ^ {T} 2 \left\| \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\| ^ {2} \cdot \left\| \widetilde {\boldsymbol {\beta}} _ {a _ {t}} \right\| ^ {2} \leq 8 \sum_ {t = 1} ^ {T} \left(e _ {t} ^ {\delta / T}\right) ^ {2}.
$$

In the next, to bound the remaining term, we use the result from Lemma 1. We first group the sums based the arm,

$$
\sum_ {t = 1} ^ {T} 1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1, a _ {t}} ^ {- 1}} ^ {2} = \sum_ {a \in \mathcal {A}} \sum_ {t \in [ T ]: a _ {t} = a} 1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1, a} ^ {- 1}} ^ {2} \tag {16}
$$

By denoting $n_T(a)$ as the number of times that arm $a$ is pulled, we can divide the arms into two groups,

$$
\mathcal {G} _ {0} := \{a \in \mathcal {A}: n _ {T} (a) = \Omega (\log (1 / \delta)) \} \quad \text { and } \quad \mathcal {G} _ {1} := \mathcal {A} \setminus \mathcal {G} _ {0}.
$$

Then, we can further decompose the r.h.s term of Equation 16 based on if the corresponding arm is in $\mathcal{G}_0$ or $\mathcal{G}_1$ . Then, by applying Lemma 1, we have the following holds, with probability at least $1 - \delta$ ,

$$
\begin{array}{l} \sum_ {a \in \mathcal {A}} \sum_ {t \in [ T ]: a _ {t} = a} 1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1, a} ^ {- 1}} ^ {2} \\ = \sum_ {a \in \mathcal {G} _ {0}} \sum_ {t \in [ T ]: a _ {t} = a} 1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1, a} ^ {- 1}} ^ {2} + \sum_ {a \in \mathcal {G} _ {1}} \sum_ {t \in [ T ]: a _ {t} = a} 1 \wedge \left\| \left[ \begin{array}{c} \boldsymbol {x} _ {t} \\ \phi^ {\star} (\boldsymbol {x} _ {t}) \end{array} \right] \right\| _ {\boldsymbol {A} _ {t - 1, a} ^ {- 1}} \\ \leq 2 K d _ {u} \log \left(1 + \frac {T L _ {u} ^ {2}}{\lambda d _ {u}}\right) + \frac {8 K L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\sigma_ {\epsilon} ^ {4}} \log \left(\frac {3 2 K d _ {u} L _ {\epsilon} ^ {2} (L _ {\epsilon} + L _ {x}) ^ {2}}{\delta \sigma_ {\epsilon} ^ {4}}\right). \\ \end{array}
$$

where the last inequality is due to Lemma 1 and apply the union bound on the K arms. To bound the remainder term, since $\alpha \in [0, 1/2]$ , by the learnability assumption as stated in Assumption 1 and Lemma 1, we have

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} \left(e _ {t} ^ {\delta / t}\right) ^ {2} \leq \sum_ {t = 1} ^ {T} C _ {0} ^ {2} \cdot \left(1 \wedge \| \boldsymbol {x} \| _ {\boldsymbol {X} _ {t - 1} ^ {- 1}} ^ {2}\right) ^ {2 \alpha} \cdot \log^ {2} \left(\frac {t T}{\delta}\right) \\ \leq 4 C _ {0} ^ {2} T ^ {1 - 2 \alpha} \log^ {2 \alpha} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right) \log^ {2} \left(\frac {T}{\delta}\right) \\ \leq 4 C _ {0} ^ {2} T ^ {1 - 2 \alpha} d _ {u} ^ {2 \alpha} \log^ {2 \alpha} \left(\frac {T L _ {u} ^ {2} / d + \lambda}{\lambda}\right) \log^ {2} \left(\frac {T}{\delta}\right) \\ \end{array}
$$

Therefore, the total regret bound is bounded by the following term with probability at least $1 - 2\delta$ ,

$$
\sqrt {T \cdot \left(8 C _ {0} T ^ {1 - 2 \alpha} \log^ {2 \alpha} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right) \log^ {2} \left(\frac {T}{\delta}\right) + K \zeta_ {T} ^ {2} \left(d _ {u} \log \left(1 + \frac {T L _ {u} ^ {2}}{\lambda d _ {u}}\right) + \frac {4 8 L _ {\epsilon} ^ {4} L _ {u} ^ {2}}{\sigma_ {\epsilon} ^ {4}} \log \left(\frac {1 9 2 K d _ {u} L _ {\epsilon} ^ {4} L _ {u} ^ {2}}{\delta \sigma_ {\epsilon} ^ {4}}\right)\right)\right)}
$$

By hiding the logarithmic terms, we can further simplify it to be

$$
\widetilde {\mathcal {O}} \left(T ^ {1 - \alpha} d _ {u} ^ {\alpha} + d _ {u} \sqrt {T K}\right)
$$

![](images/6f5e54eb608ad8e81aba2c20a0d262ce2c4dd27254f64860fb012cff7a470810.jpg)

# B.3 Missing Proofs in Generalizations

Proposition 2. The regret of poLinUCB in Algorithm 1 for action-dependent contexts is upper bounded by $\widetilde{\mathcal{O}}\left(T^{1 - \alpha}d_u^\alpha \sqrt{K} +d_u\sqrt{TK}\right)$ with probability at least $1 - \delta$ if $T = \Omega (\log (1 / \delta))$ .

Proof. Our proof follows from the proof of Theorem 1. The immediate regret at each time step t is

$$
\Delta_ {t} = r _ {t, a _ {t} ^ {\star}} - r _ {t, a _ {t}} \tag {17}
$$

Recall that the definition of $a_{t}^{\star}$ ,

$$
a _ {t} ^ {\star} := \underset {a \in \mathcal {A}} {\arg \max} \left\langle \boldsymbol {\theta} _ {a} ^ {\star}, \boldsymbol {x} _ {t, a} \right\rangle + \left\langle \boldsymbol {\beta} _ {a} ^ {\star}, \phi_ {a} ^ {\star} \left(\boldsymbol {x} _ {t, a}\right) \right\rangle . \tag {18}
$$

To bound the immediate regret, we have

$$
\Delta_ {t} = \left\langle \boldsymbol {\theta} _ {a _ {t} ^ {\star}} ^ {\star}, \boldsymbol {x} _ {t, a _ {t} ^ {\star}} \right\rangle + \left\langle \boldsymbol {\beta} _ {a _ {t} ^ {\star}} ^ {\star}, \phi_ {a _ {t} ^ {\star}} ^ {\star} \left(\boldsymbol {x} _ {t, a _ {t} ^ {\star}}\right) \right\rangle - \left\langle \boldsymbol {\theta} _ {a _ {t}} ^ {\star}, \boldsymbol {x} _ {t, a _ {t}} \right\rangle - \left\langle \boldsymbol {\beta} _ {a _ {t}} ^ {\star}, \phi_ {a _ {t}} ^ {\star} \left(\boldsymbol {x} _ {t, a _ {t}}\right) \right\rangle . \tag {19}
$$

By the definition of UCB, we further have

$$
\Delta_ {t} \leq \left\langle \widetilde {\boldsymbol {\theta}} _ {t, a _ {t}}, \boldsymbol {x} _ {t, a _ {t}} \right\rangle + \left\langle \widetilde {\boldsymbol {\beta}} _ {t, a _ {t}}, \widetilde {\phi} _ {t, a _ {t}} ^ {\star} \left(\boldsymbol {x} _ {t, a _ {t}}\right) \right\rangle - \left\langle \boldsymbol {\theta} _ {a _ {t}} ^ {\star}, \boldsymbol {x} _ {t, a _ {t}} \right\rangle - \left\langle \boldsymbol {\beta} _ {a _ {t}} ^ {\star}, \phi_ {a _ {t}} ^ {\star} \left(\boldsymbol {x} _ {t, a _ {t}}\right) \right\rangle . \tag {20}
$$

By rearranging the terms, we get

$$
\Delta_ {t} \leq \underbrace {\left\langle \widetilde {\boldsymbol {\theta}} _ {t , a _ {t}} - \boldsymbol {\theta} _ {a _ {t}} ^ {\star} , \boldsymbol {x} _ {t , a _ {t}} \right\rangle + \left\langle \widetilde {\boldsymbol {\beta}} _ {t , a _ {t}} - \boldsymbol {\beta} _ {a _ {t}} ^ {\star} , \phi_ {a _ {t}} ^ {\star} (\boldsymbol {x} _ {t , a _ {t}}) \right\rangle} _ {(1)} + \underbrace {\left\langle \widetilde {\boldsymbol {\beta}} _ {t , a _ {t}} , \widetilde {\phi} _ {t , a _ {t}} (\boldsymbol {x} _ {t , a _ {t}}) - \phi_ {a _ {t}} ^ {\star} (\boldsymbol {x} _ {t , a _ {t}}) \right\rangle} _ {(2)} \tag {21}
$$

Bounding the first term ① is the same as the proof in Theorem 1, while bounding the second term ② will be slightly different, as we now have K such functions of $\phi_{a}^{\star}(\cdot)$ for $a \in A$ to learn. By denoting the error for each estimate of $\phi_{a}^{\star}(\cdot)$ at iteration t as $e_{t,a}^{\delta}$ . Therefore, the contribution from the second term to the total regret can be bounded by

$$
\sum_ {t = 1} ^ {T} \left(e _ {t, a _ {t}} ^ {\delta / t}\right) ^ {2} = \sum_ {a \in \mathcal {A}} \sum_ {t \in [ T ]: a _ {t} = a} \left(e _ {t, a} ^ {t / \delta}\right) ^ {2} \tag {22}
$$

$$
\leq 4 K C _ {0} T ^ {1 - 2 \alpha} \log^ {2 \alpha} \left(\frac {\det \boldsymbol {X} _ {T}}{\det \boldsymbol {X} _ {0}}\right) \log^ {2} \left(\frac {T}{\delta}\right). \tag {23}
$$

Hence, by following the remaining steps in the proof of Theorem 1, we can conclude that the regret is upper bounded by

$$
\widetilde {\mathcal {O}} \left(T ^ {1 - \alpha} d _ {u} ^ {\alpha} \sqrt {K} + d _ {u} \sqrt {T K}\right), \tag {24}
$$

where the only difference is the additional $\sqrt{K}$ appeared in the first term.

![](images/a8a276b32f6ac244da8cd6fc30d76442993a35f6fcb213086b143a7decb754ef.jpg)

Proposition 3. The regret of poLinUCB in Algorithm 1 for the above setting is upper bounded by $\widetilde{\mathcal{O}}\left(T^{1 - \alpha}d_u^\alpha +d_u\sqrt{T}\right)$ with probability at least $1 - \delta$ if $T = \Omega (\log (1 / \delta))$ .

Proof. This proof also follows from the proof of Theorem 1. The immediate regret at each time step $t$ is

$$
\Delta_ {t} = r _ {t, \boldsymbol {x} _ {t} ^ {\star}} - r _ {t, \boldsymbol {x} _ {t}} \tag {25}
$$

Recall that the definition of $x_{t}^{\star}$ ,

$$
\boldsymbol {x} _ {t} ^ {\star} := \underset {\boldsymbol {x} \in D _ {t}} {\arg \max} \left\langle \boldsymbol {\theta} ^ {\star}, \boldsymbol {x} \right\rangle + \left\langle \boldsymbol {\beta} ^ {\star}, \phi^ {\star} (\boldsymbol {x}) \right\rangle . \tag {26}
$$

To bound the immediate regret, we have

$$
\Delta_ {t} = \left\langle \boldsymbol {\theta} ^ {\star}, \boldsymbol {x} _ {t} ^ {\star} \right\rangle + \left\langle \boldsymbol {\beta} ^ {\star}, \phi^ {\star} \left(\boldsymbol {x} _ {t} ^ {\star}\right) \right\rangle - \left\langle \boldsymbol {\theta} ^ {\star}, \boldsymbol {x} _ {t} \right\rangle - \left\langle \boldsymbol {\beta} ^ {\star}, \phi^ {\star} \left(\boldsymbol {x} _ {t}\right) \right\rangle \tag {27}
$$

By the definition of UCB, we further have

$$
\Delta_ {t} \leq \left\langle \widetilde {\boldsymbol {\theta}} _ {t}, \boldsymbol {x} _ {t} \right\rangle + \left\langle \widetilde {\boldsymbol {\beta}} _ {t}, \widetilde {\phi} _ {t} ^ {\star} (\boldsymbol {x} _ {t}) \right\rangle - \left\langle \boldsymbol {\theta} ^ {\star}, \boldsymbol {x} _ {t} \right\rangle - \left\langle \boldsymbol {\beta} ^ {\star}, \phi^ {\star} (\boldsymbol {x} _ {t}) \right\rangle . \tag {28}
$$

By rearranging the terms, we get

$$
\Delta_ {t} \leq \underbrace {\left\langle \widetilde {\boldsymbol {\theta}} _ {t} - \boldsymbol {\theta} ^ {\star} , \boldsymbol {x} _ {t} \right\rangle + \left\langle \widetilde {\boldsymbol {\beta}} _ {t} - \boldsymbol {\beta} ^ {\star} , \phi^ {\star} (\boldsymbol {x} _ {t}) \right\rangle} _ {(1)} + \underbrace {\left\langle \widetilde {\boldsymbol {\beta}} _ {t} , \widetilde {\phi} _ {t} (\boldsymbol {x} _ {t}) - \phi^ {\star} (\boldsymbol {x} _ {t}) \right\rangle} _ {(2)} \tag {29}
$$

Since we only need to fit a single $\theta^{\star}$ , $\beta^{\star}$ and $\phi^{\star}(\cdot)$ . We thus have the following bound for the total regret,

$$
\widetilde {\mathcal {O}} \left(T ^ {1 - \alpha} d _ {u} ^ {\alpha} + d _ {u} \sqrt {T}\right). \tag {30}
$$

![](images/3ff9378c254e3954dc596595a3fbb416869dc3980b8de58d28f8cc5bc3460e06.jpg)

# C Technical Lemmas

Lemma 3 (Confidence Ellipsoid, based on Theorem 2 of (Abbasi-Yadkori et al., 2011)). Let $\pmb{w}^{\star}\in\mathbb{R}^{d}$ , $\pmb{V}_{0}=\lambda\pmb{I}$ , $\lambda>0$ . For any $t\geq0$ , let $\pmb{u}_{1},\cdots,\pmb{u}_{t}\in\mathbb{R}^{d}$ , define $r_{t}=\langle\pmb{u}_{t},\pmb{w}^{\star}\rangle+\eta_{t}$ where $\eta_{t}$ is $R_{\eta}$ -sub-Gaussian and assume that $\|\pmb{w}^{\star}\|_{2}\leq L_{\pmb{w}}$ ; let $\pmb{V}_{t}=\pmb{V}_{0}+\sum_{s=1}^{t}\pmb{u}_{s}\pmb{u}_{s}^{\top}$ and $\hat{\pmb{w}}_{t}$ be the corresponding regularised least-square estimator. Then, for any $\delta>0$ and $t\geq0$ , with probability at least $1-\delta$ , $\pmb{w}^{\star}$ lies in the set:

$$
\mathcal {C} _ {t} = \left\{\boldsymbol {w} \in \mathbb {R} ^ {d}: \| \hat {\boldsymbol {w}} _ {t} - \boldsymbol {w} \| _ {\boldsymbol {V} _ {t}} \leq \sqrt {\lambda} L _ {\boldsymbol {w}} + R _ {\eta} \sqrt {2 \log \left(\frac {\det (\boldsymbol {V} _ {t}) ^ {1 / 2} \det (\lambda \boldsymbol {I}) ^ {- 1 / 2}}{\delta}\right)} \right\}. \tag {31}
$$

Furthermore, if for all $t \geq 1$ , $\| \mathbf{u}_t \| \leq L_{\mathbf{u}}$ , then for any $\delta > 0$ and $t \geq 0$ , with probability at least $1 - \delta$ , $\mathbf{w}^{\star}$ lies in the set:

$$
\mathcal {C} _ {t} = \left\{\boldsymbol {w} \in \mathbb {R} ^ {d}: \| \hat {\boldsymbol {w}} _ {t} - \boldsymbol {w} \| _ {\boldsymbol {V} _ {t}} \leq \sqrt {\lambda} L _ {\boldsymbol {w}} + R _ {\eta} \sqrt {d \log \left(\frac {1 + t L _ {\boldsymbol {u}} ^ {2} / \lambda}{\delta}\right)} \right\}. \tag {32}
$$

Lemma 4 (Bernstein's Inequality for Matrices, Theorem 6.1.1 of (Tropp et al., 2015)). Let $\mathbf{X}_1, \cdots, \mathbf{X}_n \in \mathbb{R}^{d_1 \times d_2}$ be independent and centered random matrices. Assume that for each $i \in [n]$ , $\mathbf{X}_i$ is uniformly bounded, that is:

$$
\mathbb {E} \left[ \boldsymbol {X} _ {i} \right] = \mathbf {0} \quad a n d \quad \| \boldsymbol {X} _ {i} \| \leq B, \tag {33}
$$

where $\|\cdot\|$ denotes the spectral-norm distance here. Introduce the sum

$$
\boldsymbol {Z} = \sum_ {i = 1} ^ {n} \boldsymbol {X} _ {i}, \tag {34}
$$

and let $\mathbb{V}(\mathbf{Z})$ denote the matrix variance statistics of the sum Z:

$$
\mathbb {V} (\boldsymbol {Z}) = \max \left\{\| \mathbb {E} [ \boldsymbol {Z} \boldsymbol {Z} ^ {*} ] \|, \| \mathbb {E} [ \boldsymbol {Z} ^ {*} \boldsymbol {Z} ] \| \right\} \tag {35}
$$

$$
= \max \left\{\left\| \sum_ {i = 1} ^ {n} \boldsymbol {X} _ {i} \boldsymbol {X} _ {i} ^ {*} \right\|, \left\| \sum_ {i = 1} ^ {n} \boldsymbol {X} _ {i} ^ {*} \boldsymbol {X} _ {i} \right\| \right\}, \tag {36}
$$

where the asterisk $^{*}$ denotes the conjugate transpose operation. Then, for every $\epsilon \geq 0$ , we have,

$$
\mathbb {P} \left(\| \boldsymbol {Z} \| \geq \epsilon\right) \leq \left(d _ {1} + d _ {2}\right) \cdot \exp \left(\frac {- \epsilon^ {2} / 2}{\mathbb {V} (\boldsymbol {Z}) + B \epsilon / 3}\right). \tag {37}
$$

![](images/0309d2faacef7049b7cb9a2ff9e425cd1047235917aec255e7f9ee26c887e560.jpg)

<details>
<summary>line</summary>

| Number of Examples (T) | Linear (α = 0.5): T^(-0.5) | Hölder (β = 0.25, d = 2): T^(-19/2) | Hölder (β = 0.5, d = 3): T^(-19/2) | Hölder (β = 0.75, d = 3): T^(-19/2) | Hölder (β = 1, d = 3): T^(-19/2) |
| ---------------------- | -------------------------- | ----------------------------------- | ---------------------------------- | ---------------------------------- | ------------------------------- |
| 1                      | 1.0                        | 1.0                                 | 1.0                                | 1.0                                | 1.0                             |
| 10                     | ~0.1                       | ~0.3                                | ~0.4                               | ~0.5                               | ~0.6                            |
| 100                    | ~0.01                      | ~0.1                                | ~0.2                               | ~0.3                               | ~0.4                            |
| 1000                   | ~0.001                     | ~0.03                               | ~0.1                               | ~0.2                               | ~0.3                            |
</details>

Figure 3: Comparison of convergence under different $\phi$ functions (e.g., linear and those in Holder space).

![](images/c3ca973ef21d6711f8a35899ae76cc6187b1386be502d4db6b3d8c21d7b9b63c.jpg)

<details>
<summary>line</summary>

| Time steps | Random | LinUCB (φ̂) | poLinUCB (ours) | LinUCB (x and z) | LinUCB (x only) |
| ---------- | ------ | ---------- | --------------- | ---------------- | --------------- |
| 0          | 0      | 0          | 0               | 0                | 0               |
| 500        | 2.5e6  | 0.1e6      | 0.1e6           | 0.1e6            | 0.3e6           |
| 1000       | 5.0e6  | 0.15e6     | 0.15e6          | 0.15e6           | 0.45e6          |
| 1500       | 7.5e6  | 0.18e6     | 0.18e6          | 0.18e6           | 0.55e6          |
| 2000       | 9.0e6  | 0.2e6      | 0.2e6           | 0.2e6            | 0.65e6          |
| 2500       | 9.5e6  | 0.22e6     | 0.22e6          | 0.22e6           | 0.75e6          |
| 3000       | 9.8e6  | 0.23e6     | 0.23e6          | 0.23e6           | 0.8e6           |
| 3500       | 9.9e6  | 0.24e6     | 0.24e6          | 0.24e6           | 0.85e6          |
| 4000       | 9.95e6 | 0.25e6     | 0.25e6          | 0.25e6           | 0.9e6           |
| 4500       | 9.98e6 | 0.255e6    | 0.255e6         | 0.255e6          | 0.95e6          |
| 5000       | 1.0e6  | 0.26e6     | 0.26e6          | 0.26e6           | 1.0e6           |
</details>

![](images/5168363c920eee8f547fe2e73335dad16a3daafe4ca4b8fd5d06f7cd18a99804.jpg)

<details>
<summary>line</summary>

| Time steps | Random | LinUCB (φ̂) | polLinUCB (ours) | LinUCB (x and z) | LinUCB (x only) |
| ---------- | ------ | ---------- | ---------------- | ---------------- | --------------- |
| 0          | 0      | 0          | 0                | 0                | 0               |
| 200        | ~1.5e6 | ~0.3e6     | ~0.1e6           | ~0.1e6           | ~0.6e6          |
| 400        | ~2.5e6 | ~0.4e6     | ~0.1e6           | ~0.1e6           | ~1.2e6          |
| 600        | ~3.0e6 | ~0.5e6     | ~0.1e6           | ~0.1e6           | ~1.7e6          |
| 800        | ~3.5e6 | ~0.55e6    | ~0.1e6           | ~0.1e6           | ~2.3e6          |
| 1000       | ~3.8e6 | ~0.6e6     | ~0.1e6           | ~0.1e6           | ~2.9e6          |
</details>

Figure 4: Comparison under the setup where the reward's dependency on the post-serving context is noiseless.

# D Additional Results and Discussions

# D.1 Additional Experiments

We further investigate the case where the true reward is $\langle \phi^{\star}(\pmb {x}),\beta \rangle$ . For this case, the reward's dependency on the post-serving context is noiseless. The results are presented in Figure 4. We observe that the algorithm adapted from Wang et al. (2016) still performs worse than our method, though the gap seems become smaller.

# D.2 Additional Discussions on Assumption 1

Our regret analysis can accommodate different values of $\alpha$ in Assumption 1. The rate of $1/\sqrt{T}$ (when $\alpha = 0.5$ ) is a commonly observed rate for many classical machine learning algorithms, including linear regression, logistic regression, and SVM with a linear kernel. This rate is rooted in the law of large numbers and the central limit theorem. For smaller $\alpha$ values, the generalization error will converge more slowly than $1/\sqrt{T}$ , indicating that the learning problem becomes increasingly difficult. In the extreme case when $\alpha = 0$ , $\phi^{\star}$ function cannot be learned accurately, thus we will inevitably suffer a linear regret (simply due to model misspecification).

For standard linear functions, $\alpha = 0.5$ and our regret bound in such situations is tight w.r.t. $\alpha$ . However, we intentionally made assumption 1 to be more general in order to accommodate other much more complex ways of estimating the $\phi$ functions such as manifold regression (see, e.g., Yang and Dunson (2016)), non-parametric ways like k-nearest-neighbors to estimate phi, or to estimate complex non-smooth function (e.g., functions in Holder spaces). For example, when $\phi$ is a function in Holder space $H(\beta)$ , then the learning rate is $T^{-2\beta/(2\beta+d)}$ , which is generally slower than $T^{-0.5}$ and depends on $\beta$ as well as the data dimension d (see, e.g., the note of Tibshirani (2017)). A visual illustration can be found in Figure 3.