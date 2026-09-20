# FASTER APPROXIMATION OF PROBABILISTIC AND DISTRIBUTIONAL VALUES VIA LEAST SQUARES

Weida Li

vidaslee@gmail.com

Yaoliang Yu

School of Computer Science

University of Waterloo

Vector Institute

yaoliang.yu@uwaterloo.ca

# ABSTRACT

The family of probabilistic values, axiomatically-grounded in cooperative game theory, has recently received much attention in data valuation. However, it is often computationally expensive to compute exactly (exponential w.r.t. the number of data to valuate denoted by n). The existing generic estimator costs $O(n^{2} \log n)$ utility evaluations to achieve an $(\epsilon, \delta)$ -approximation under the 2-norm, while faster estimators have been developed recently for special cases (e.g., empirically for the Shapley value and theoretically for the Banzhaf value). In this work, starting from the discovered connection between probabilistic values and least square regressions, we propose a Generic Estimator based on Least Squares (GELS) along with its variants that cost $O(n \log n)$ utility evaluations for many probabilistic values, largely extending the scope of this currently best complexity bound. Moreover, we show that each distributional value, proposed by Ghorbani et al. (2020) to alleviate the inconsistency of probabilistic values induced by using distinct databases, can also be cast as optimizing a similar least square regression. This observation leads to a theoretically-grounded framework TrELS (Training Estimators based on Least Squares) that can train estimators towards the specified distributional values without requiring any supervised signals. Particularly, the trained estimators are capable of predicting the corresponding distributional values for unseen data, largely saving the budgets required for running Monte-Carlo methods otherwise. Our experiments verify the faster convergence of GELS, and demonstrate the effectiveness of TrELS in learning distributional values. Our code is available at https://github.com/watml/fastpvalue.

# 1 INTRODUCTION

In cooperative game theory, the family of probabilistic values, to which the Shapley value (Shapley 1953) and the Banzhaf value (Banzhaf 1965) belong, is uniquely characterized by the axioms of linearity, dummy, monotonicity and symmetry (Weber 1977, Theorems 5 and 10). The induced formula of probabilistic values is often deemed essential for data valuation methods (Kwon and Zou 2022a; Lin et al. 2022; Wang and Jia 2023), and is also proved effective in feature attribution (Jethani et al. 2022; Kwon and Zou 2022b; Lundberg and Lee 2017). Specifically, data valuation aims to impute an importance value to each data point z in the training dataset $D_{tr}$ that represents its contribution to the performance of a model trained on $D_{tr}$ , and it is empirically expected that models retrained without the “less valuable” data (e.g., data that are assigned with importance values lower than a specified threshold) in $D_{tr}$ may achieve performance gains (Ghorbani and Zou 2019).

Throughout, we identify the dataset $D_{tr}$ with $[n] = \{1, 2, \ldots, n\}$ , where $n = |D_{tr}|$ is its size. Without ambiguity, for each subset $S$ , its lower-case $s$ is used to denote its cardinality $|S|$ , and we write $S \cup i$ and $S \setminus i$ instead of $S \cup \{i\}$ and $S \setminus \{i\}$ , respectively. Each probabilistic value can be parameterized by a list of non-negative vectors $\mathcal{P} = \{\mathbf{p}^n \in \mathbb{R}_+^n\}_{n \geq 1}$ such that $\sum_{s=1}^{n} \binom{n-1}{s-1} p_s^n = 1$ for every $n \geq 1$ , and the importance value assigned to the $i$ -th data point is computed by

$$
\phi_ {i} (U ^ {n}) = \phi_ {i} (U ^ {n}; \mathcal {P}) = \sum_ {S \subseteq [ n ] \backslash i} p _ {s + 1} ^ {n} \left(U ^ {n} (S \cup i) - U ^ {n} (S)\right) \tag {1}
$$

where $U^{n}:2^{[n]}\to R$ is a user-specified utility function and the superscript n is to indicate its domain $2^{[n]}$ . Typically, $U^{n}(S)$ outputs the performance of a chosen model trained on $S\subseteq[n]$ . In this work, $U^{n}(\emptyset)$ denotes the performance of initialized models. Take classification tasks as an example, $U^{n}(S)$ could be the accuracy or the cross-entropy loss reported on a held-out dataset $D_{perf}$ . It is obvious that computing $\phi(U^{n})$ exactly requires $2^{n}$ times of evaluating the utility function $U^{n}$ , and hence is intractable. Therefore, there has been much research devoted to developing efficient estimators. To our best knowledge, there is only one family of generic estimators designed for all probabilistic values: sampling lift and its weighted variant (Kwon and Zou 2022a). Specifically, the sampling lift estimator requires $O\left(\frac{n^{2}}{\epsilon^{2}}\log\frac{n}{\delta}\right)$ utility evaluations to achieve an $(\epsilon,\delta)$ -approximation (see Proposition 5 in the Appendix). We also note that Zhou et al. (2023, Lemma 1 and Proposition 1) analyzed convergence using $(\epsilon_{1},\epsilon_{2},\delta)$ -approximation instead where $\epsilon_{1}$ and $\epsilon_{2}$ account for the multiplicative and additive error, respectively.

So far, many faster estimators have been proposed for specific probabilistic values. For instance, the advent of faster estimators designed specifically for the Shapley value was witnessed (Covert and Lee 2021; Kolpaczki et al. 2023; Lundberg and Lee 2017; Zhang et al. 2023b); Wang and Jia (2023, Theorem 4.9) proved that the estimator based on their proposed maximum sample reuse (MSR) principle only requires $O\left(\frac{n}{\epsilon^2}\log \frac{n}{\delta}\right)$ utility evaluations for the Banzhaf value, but they also demonstrated in Appendix C.2 therein that the MSR estimator does not extend to many other probabilistic values, e.g., the family of Beta Shapley values (Kwon and Zou 2022a), in which the Shapley value resides. We also notice that Lin et al. (2022) discovered a framework of unconstrained least square regressions that leads to an estimator for a subfamily of probabilistic values (which, however, does not include the Shapley value). All in all, it is still an open question on how to efficiently approximate other probabilistic values, e.g., the Beta Shapley values.

A potential drawback of the probabilistic values is that they depend on the underlying database $D_{tr}$ . Using another database, the recalculated importance values could be very inconsistent with the previous ones. To overcome this issue, Ghorbani et al. (2020) proposed the framework of distributional values, in which the importance value of any (unseen) data point z is

$$
\psi (z; \mathcal {D}, \mathbf {w}, U) = \underset {s \sim [ m ]} {\mathbb {E}} \underset {S \sim \mathcal {D} ^ {s - 1}} {\mathbb {E}} [ U (S \cup z) - U (S) ] \tag {2}
$$

where $\mathcal{D}$ is a data distribution, $\mathbf{w} \in \mathbb{R}^m$ is a probability vector and the domain of the utility function $U$ is $\bigcup_{n \geq 1} \{S \sim \mathcal{D}^n\}$ . Empirically, $\mathcal{D}$ is replaced by $\frac{1}{|B|} \sum_{z \in B} \delta_z$ where $B$ is a (large) dataset sampled from $\mathcal{D}$ (Ghorbani et al. 2020). Clearly, the computational cost for the distributional values is a big hurdle for its practical deployment.

In this paper, starting from the discovered connection between probabilistic values and least squares (see Proposition 2), we develop a Generic Estimator based on Least Squares (GELS) along with its variants: GELS-R and GELS-Shapley. Precisely, GELS-R is to approximate the ranking of probabilistic values, while GELS-Shapley is specific to the Shapley value. In addition, we demonstrate that each distributional value can also be cast into a similar least square regression, which serves as the theoretical ground for formulating the framework TrELS (Training Estimators based Least Squares) that can train estimators towards the specified distributional value without requiring any supervised signals. In other words, unlike Monte-Carlo methods, TrELS allows the use of trained estimators for prediction. Our main contributions are summarized as follows:

1. We propose GELS and GELS-R for all probabilistic values, and prove that both of them require $O\left(\frac{n}{\epsilon^2}\log \frac{n}{\delta}\right)$ utility evaluations to achieve an $(\epsilon ,\delta)$ -approximation for many probabilistic values, matching the currently best bound for special cases; see Algorithms 1 and 2. In addition, we also develop GELS-Shapley in Algorithm 3 specific to the Shapely value.   
2. By casting the distributional values into optimizing least square regressions, we design TrELS for training estimators towards distributional values without supervision. See Algorithm 4 and Theorem 1.   
3. Our experiments verify the faster convergence of GELS and GELS-R, and we also show that the distributional values can be well-learned using TrELS.   
4. As a minor side note, we also extend the approximation-without-requiring-marginal (ARM) estimator, designed specifically for the Shapley value (Kolpaczki et al. 2023), to the whole family of probabilistic values. See Proposition 8 in the Appendix.

# 2 BACKGROUND

The sampling lift estimator (Moehle et al. 2022) refers to any approximation algorithm designed according to

$$
\phi_ {i} (U ^ {n}) = \sum_ {S \subseteq [ n ] \setminus i} p _ {s + 1} ^ {n} \left(U ^ {n} (S \cup i) - U ^ {n} (S)\right) = \mathbb {E} _ {S \subseteq [ n ] \setminus i} [ U ^ {n} (S \cup i) - U ^ {n} (S) ].
$$

Its weighted variant is to use $n\binom{n-1}{s}p_{s+1}^{n}(U^{n}(S \cup i) - U^{n}(S))$ with $P(S) = \frac{1}{n}\binom{n-1}{s}^{-1}$ instead (Kwon and Zou 2022a). Particularly, they are the same for the Shapley value. As summarized in Proposition 5, the sampling lift requires $O(\frac{n^{2}}{\epsilon^{2}} \log \frac{n}{\delta})$ utility evaluations to achieve an $(\epsilon, \delta)$ -approximation. Recently, Kolpaczki et al. (2023) proposed an ARM estimator specifically for the Shapley value, but we notice that it can be easily generalized for all probabilistic values. Therefore, we present the generic formula of ARM in the main paper while leaving its justification to Proposition 8 in the Appendix. The ARM estimator is to approximate by using

$$
\phi_ {i} (U ^ {n}) = \mathbb {E} _ {S \sim P _ {A R M} ^ {+} | i \in S} [ U ^ {n} (S) ] - \mathbb {E} _ {S \sim P _ {A R M} ^ {-} | i \not \in S} [ U ^ {n} (S) ]
$$

where $P_{ARM}^{+}(S) \propto p_{s}^{n}$ for all non-empty subsets $S \subseteq [n]$ and $P_{ARM}^{-}(S) \propto p_{s+1}^{n}$ for all $S \subsetneq [n]$ .

In addition, Lin et al. (2022) developed the average marginal effect (AME) that casts a subfamily of probabilistic values as the uniquely optimal solution to

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n}} {\arg \min} \mathbb {E} [ (Y - \boldsymbol {X} ^ {\top} \mathbf {v}) ^ {2} ],
$$

where $X$ and $Y$ are random variables whose distribution is induced by i) sampling $t \sim P$ where $P$ is any probability distribution on the open interval (0, 1), ii) sampling a subset $S \subseteq [n]$ by including each data point in [n] with probability $t$ , and iii) setting $Y = U^n(S)$ and $X_i = \frac{1}{t \cdot M_P}$ if $i \in S$ and $\frac{-1}{(1-t)M_P}$ otherwise where $M_P = \mathbb{E}_{t \sim P}[\frac{1}{t(1-t)}]$ . Particularly, the uniquely optimal solution is just a probabilistic value parameterized by $p_s^n = \int_0^1 t^{s-1}(1-t)^{n-s} dP(t)$ . If $P$ is uniform, the corresponding $\mathbf{p}^n$ is the one defining the Shapley value, but $M_P = \infty$ , and thus AME does not work for the Shapley value.

On the other hand, Wang and Jia (2023) proposed the maximum sample reuse (MSR) principle which aims to find a distribution $P_{MSR}$ on $2^{[n]}$ such that for every $i \in [n]$ ,

$$
P _ {M S R} (S \mid i \in S) = p _ {s} ^ {n} \text {   and   } P _ {M S R} (S \mid i \not \in S) = p _ {s + 1} ^ {n},
$$

with which Eq. (1) can be rewritten as $\phi_i(U^n) = \mathbb{E}_{S|i \in S}[U^n(S)] - \mathbb{E}_{S|i \notin S}[U^n(S)]$ . Specifically, $P_{MSR}$ is uniform over $2^{[n]}$ for the Banzhaf value, and Wang and Jia (2023, Theorem 4.9) showed that the resulting estimator only requires $O\left(\frac{n}{\epsilon^2} \log \frac{n}{\delta}\right)$ utility evaluations. However, they also demonstrated in Appendix C.2 therein that such $P_{MSR}$ exists if and only if $p_{s+1}^n = \eta(n) \cdot p_s^n$ for every $1 \leq s \leq n-1$ where $\eta(n) \in \mathbb{R}$ , which excludes all Beta Shapley values.

A recently-proposed faster algorithm specific to the Shapley value is the complement estimator (Zhang et al. 2023b) using the complement formula

$$
\phi_ {i} ^ {S h} (U ^ {n}) = \sum_ {S \subseteq [ n ]: i \in S} \frac {(s - 1) ! (n - s) !}{n !} \left(U ^ {n} (S) - U ^ {n} ([ n ] \backslash S)\right),
$$

which was discovered half a century ago by Harsanyi (1963, Eq. (4.1)). Earlier, Lundberg and Lee (2017) proposed the kernelSHAP estimator that exploits the fact that the Shapley value $\phi^{Sh}(U^{n})$ is the uniquely optimal solution to

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n}} {\arg \min} \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} \binom {n - 2} {s - 1} ^ {- 1} \left(U ^ {n} (S) - U ^ {n} (\emptyset) - \sum_ {i \in S} v _ {i}\right) ^ {2} \tag {3}
$$

$$
\text { s.t. } \sum_ {i \in [ n ]} v _ {i} = U ^ {n} ([ n ]) - U ^ {n} (\emptyset),
$$

which was first discovered by Charnes et al. (1988, Theorem 4). Since it is difficult to analyze whether the kernelSHAP is unbiased or not, Covert and Lee (2021, Eq. (9)) later proposed a (provably) unbiased variant, and suggested that the paired sampling technique can enhance both. Very recently, Fumagalli et al. (2023, Theorem 4.5) and Zhang et al. (2023a, Eqs. (11) and (12)) simplified the formula of the unbiased kernelSHAP.

# 3 MAIN RESULTS

In this section, we first show how to efficiently estimate the ranking induced by any probabilistic value. Then, by introducing a null data point we show how to turn our ranking estimator into a bona fide estimator, while retaining the same efficiency for many probabilistic values. Lastly, we extend our theory to an unsupervised framework that trains estimators towards distributional values.

# 3.1 WHEN RANKING SUFFICES

Practitioners often need to screen training data before feeding them to a model, to remove outliers, low-quality data, or even adversarial examples that are deemed harmful to training. Data valuation is a natural way to achieve this goal, i.e., only data assigned with high importance values could be considered potentially “valuable.” If one has a good estimate of the proportion of “valuable” data, then the relative ranking, instead of the more precise and demanding importance values, suffices. Our first results below pave the way to efficiently estimate the relative ranking underlying any probabilistic value. Particularly, $G^{n} = \{U^{n} : 2^{[n]} \to R\}$ is the set that contains all possible utility functions provided there are n data to valuate.

Proposition 1. Define, for every $n \geq 1$ , $U^{n} \in G^{n}$ and $i \in [n]$ ,

$$
\mathcal {R} _ {i} (U ^ {n}) = \mathcal {R} _ {i} (U ^ {n}; \mathcal {P}) = \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} m _ {s} ^ {n} \cdot \mathbb {1} _ {i \in S} \cdot U ^ {n} (S) \tag {4}
$$

where $m_{s}^{n}=m_{s}^{n}(\mathcal{P})=p_{s}^{n}+p_{s+1}^{n}$ for every $s\in[n-1]$ . Then, for every $n\geq1$ and $U^{n}\in G^{n}$ , $\mathcal{R}(U^{n})$ and $\phi(U^{n})$ produce the same ranking. Precisely, $\mathcal{R}(U^{n})=\phi(U^{n})+g(U^{n})\mathbf{1}_{n}$ where $g(U^{n})=g(U^{n};\mathcal{P})\in\mathbb{R}$ . As a side note, it holds for any $p^{n}\in R^{n}$ .

We emphasize that Proposition 1 is closely related to Proposition 2 below in the sense that they immediately imply each other; we first noticed a more general version of proposition 2 that applies to all least square values (Ruiz et al. 1998, Definition 5), a broader family that includes all the additive-efficient-normalized probabilistic values; see Appendix D for more details.

Proposition 2. Consider the problem

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n}} {\arg \min} \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} m _ {s} ^ {n} \cdot \left(U ^ {n} (S) - \sum_ {i \in S} v _ {i}\right) ^ {2}, \tag {5}
$$

its uniquely optimal solution shares the same ranking as $\phi(U^{n})$ . As a side note, it holds true for any $p^{n} \in R^{n}$ that produces a non-negative weight vector $m^{n}$ with $\sum_{s=1}^{n-1} m_{s}^{n} > 0$ .

Algorithm 1 summarizes the estimator induced by Propositions 1 and 2. Note that for GELS-R each utility evaluation $U^n(S)$ can be used for updating $s$ estimate of the specified probabilistic value. By contrast, the (weighted) sampling lift estimator spends two utility evaluations to update the estimate of only one data point.

# 3.2 WHEN PROBABILISTIC VALUES ARE DESIRED

Remark 1. To recover $\phi(U^n)$ from $\mathcal{R}(U^n)$ , we introduce a null data point labeled as $n+1$ to extend each $U^n$ into $\overline{U}^{n+1}$ such that $\overline{U}^{n+1}(S) = U^n(S \cap [n])$ for every $S \subseteq [n+1]$ . Meanwhile, we construct $\mathbf{p}^{n+1} \in \mathbb{R}^{n+1}$ such that $p_s^n = p_s^{n+1} + p_{s+1}^{n+1}$ for every $s \in [n]$ , and note that $\mathbf{p}^{n+1}$ may contain negative weights. Proposition 1 demonstrates that $\mathcal{R}(\overline{U}^{n+1}) = \phi(\overline{U}^{n+1}) + g(\overline{U}^{n+1})\mathbf{1}_{n+1}$ . Particularly, the definition of Eq. (1) makes that $\phi_{n+1}(\overline{U}^{n+1}) = 0$ and the structure $p_s^n = p_s^{n+1} + p_{s+1}^{n+1}$ leads to $\phi_i(U^n) = \phi_i(\overline{U}^{n+1})$ for every $i \in [n]$ . Therefore, $g(\overline{U}^{n+1}) = \mathcal{R}_{n+1}(\overline{U}^{n+1})$ , and we have the recovery formula $\phi_i(U^n) = \phi_i(\overline{U}^{n+1}) = \mathcal{R}_i(\overline{U}^{n+1}) - \mathcal{R}_{n+1}(\overline{U}^{n+1})$ . This idea is summarized in Proposition 3 and Algorithm 2.

Proposition 3. The uniquely optimal solution $\mathbf{v}^*$ to the problem

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n + 1}} {\arg \min} \sum_ {\emptyset \subsetneq S \subsetneq [ n + 1 ]} p _ {s} ^ {n} \cdot \left(U ^ {n} (S \cap [ n ]) - \sum_ {i \in S} v _ {i}\right) ^ {2} \tag {6}
$$

satisfies that $\phi_i(U^n) = v_i^* - v_{n+1}^*$ for every $i \in [n]$ .

Algorithm 1: GELS-R (Generic Estimator based on Least Squares for Rankings)   
Input: A dataset $D_{tr} \equiv [n]$ to be valuated, a utility function $U^{n} \in G^{n}$ , a weight vector $q \in R^{n-1}$ defined by $q_{s} = \binom{n}{s}(p_{s}^{n} + p_{s+1}^{n})$ , and a total number T of samples

Output: An unbiased estimate $\hat{r}$ to $\mathcal{R}(U^{n})$ up to some scalar

1 Normalize q into a probability vector $q \leftarrow q / \sum_{s=1}^{n-1} q_{s}$ 2 $\hat{r} \leftarrow 0_{n}, t \leftarrow 0_{n}$ 3 for $k = 1, 2, \ldots, T$ do

4 Sample $s_{k} \in [n-1]$ using the probability vector q

5 Uniformly sample $S_{k}$ from $\{R \subseteq [n] \mid |R| = s_{k}\}$ 6 for $i \in S_{k}$ do

7 $t_{i} \leftarrow t_{i} + 1$ and $\hat{r}_{i} \leftarrow (1 - \frac{1}{t_{i}})\hat{r}_{i} + \frac{1}{t_{i}} U^{n}(S_{k})$

Algorithm 2: GELS (Generic Estimator based on Least Squares)   
Input: A $D_{tr} \equiv [N]$ to be valuated, a utility function $U^{n} \in G^{n}$ , a weight vector $q \in R^{n}$ defined by $q_{s} = \binom{n+1}{s} p_{s}^{n}$ , and a total number T of samples

Output: An unbiased estimate $\hat{\phi}$ to $\phi(U^{n})$ 1 Introduce a null datum labeled by $n+1$ , and extend $U^{n}$ to $\overline{U}^{n+1} // \overline{U}^{n+1}(S) = U^{n}(S \cap [n])$ 2 Obtain $\hat{r} \in R^{n+1}$ from Algorithm 1 using $[n+1]$ , $\overline{U}^{n+1}$ , q and T

3 $\hat{\phi}_{i} \leftarrow (\sum_{s=1}^{n} \frac{s}{n+1} q_{s})(\hat{r}_{i} - \hat{r}_{n+1})$ for each $i \in [n]$ // using unnormalized q

Remark 2. For the Shapley value, since $\sum_{i\in [n]}\phi_i^{Sh}(U^n) = U^n ([n]) - U^n (\emptyset)$ and by Proposition 1 $\phi (U^n) = \mathcal{R}(U^n) + g(U^n)\mathbf{1}_n$ , we have $U^n ([n]) - U^n (\emptyset) = \sum_{i\in [n]}\mathcal{R}_i(U^n) + g(U^n)\cdot n$ , and thus $g(U^n) = \frac{1}{n} (U^n ([n]) - U^n (\emptyset) - \sum_{i\in [n]}\mathcal{R}_i(U^n))$ . Therefore, we can recover $\phi^{Sh}(U^n)$ without introducing a null data point, which is summarized in Algorithm 3.

Proposition 4. Assume that $\| U^n\|_{\infty}\leq u$ for every $U^n\in \bigcup_{n\geq 1}\mathcal{G}^n$ . We have the following results: i) GELS requires $O\left(\frac{\tau(n)n}{\epsilon^2}\log \frac{n}{\delta}\right)$ utility evaluations to achieve an $(\epsilon ,\delta)$ -approximation, i.e., $P(\| \hat{\phi} (U^n) - \phi (U^n)\| _2\geq \epsilon)\leq \delta$ ; ii) for GELS-R estimator, it requires $O\left(\frac{\kappa(n)n}{\epsilon^2}\log \frac{n}{\delta}\right)$ utility evaluations instead; iii) plus, the corresponding convergence of GELS-Shapley is $O\left(\frac{n}{\epsilon^2}\log (n)^2\log \frac{n}{\delta}\right)$ .

Remark 3. Appendix H presents that both $\tau(n)$ and $\kappa(n)$ are proportional to the inverse square of the average reuse rate of utility evaluations. In other words, the more estimates on average in $\hat{\phi}$ each utility evaluation are used to update, the more efficient GELS and GELS-R will be. Precisely, we study the asymptotic behaviors of $\tau(n)$ and $\kappa(n)$ for semi-values, which include all probabilistic values mentioned and whose $\mathcal{P}$ can be summarized by a probability measure $\mu$ over the closed interval [0,1] by $p_s^n = \int_0^1 t^{s-1}(1-t)^{n-s} \mathrm{d}\mu(t)$ . To sum, i) the Banzhaf value and the Beta Shapley values with $\alpha, \beta > 1$ corresponds to $\tau(n), \kappa(n) \in \Theta(1)$ ; ii) for the Beta Shapley values with $\alpha, \beta \geq 1$ , it becomes $\tau(n), \kappa(n) \in O(\log(n)^2)$ instead; iii) particularly, $\kappa(n) \in \Theta(1)$ for the Shapley value (which is the Beta Shapley value with $\alpha = \beta = 1$ ), which indicates GELS-R requires $O\left(\frac{n}{\epsilon^2} \log \frac{n}{\delta}\right)$ utility evaluations to achieve an $(\epsilon, \delta)$ -approximation; however, we emphasize that the convergence of GELS-R and GELS-Shapley cannot be compared directly as the definition of $(\epsilon, \delta)$ -approximation does not account for multiplicative error; iv) additionally, while using the paired sampling technique, GELS-Shapley exactly recovers unbiased KernelSHAP, and thus Proposition 4 also proves that the convergence of unbiased KernelSHAP with the paired sampling technique is $O\left(\frac{n}{\epsilon^2} \log(n)^2 \log \frac{n}{\delta}\right)$ . We refer the reader to Appendices F, I and J for more details.

# 3.3 DISTRIBUTIONAL VALUE

We are now ready to combine the previous results with the notion of distributional values Eq. (2) to establish an unsupervised framework for training estimators towards distributional values. We refer the reader to Appendix A for more details of distributional values. Such trained estimators can valuate any unseen data point sampled from the same (or close-enough) data distribution in a single forward pass, which largely saves the budgets required for running Monte-Carlo methods otherwise.

Algorithm 3: GELS-Shapley (Generic Estimator based on Least Squares for the Shapley value)

Input: A $D_{tr} \equiv [N]$ to be valuated, a utility function $U^{n} \in G^{n}$ , a weight vector $q \in R^{n-1}$ defined by $q_{s} = \frac{n}{s(n-s)}$ , and a total number T of samples

Output: An unbiased estimate $\hat{\phi}$ to the Shpaley value of $U^{n}$

1 Obtain $\hat{r} \in R^{n}$ from Algorithm 1 using [n], $U^{n}$ , q and T

2 $\hat{r}_i\gets \hat{r}_i\cdot H_{n - 1}$ for each $i\in [n]$

3 $\hat{\phi}_i\gets \hat{r}_i + O$ for each $i\in [n]$

$$
\begin{array}{r l r} & & {/ / H _ {n - 1} = \sum_ {s = 1} ^ {n - 1} \frac {1}{s}} \\ & / / O = (U ^ {n} ([ n ]) - U ^ {n} (\emptyset) - \sum_ {i = 1} ^ {n} \hat {r} _ {i}) / n \end{array}
$$

As counterparts in feature attribution, Covert et al. (2023) and Jethani et al. (2022) exploited the least square regression (3) together with the additive efficient normalization proposed by Ruiz et al. (1998, Definition 11) to train neural networks that can then predict the Shapley value of any unseen instance in a single forward pass.

In practice, the data distribution $\mathcal{D}$ in Eq. (2) is replaced by an empirical distribution $\mathcal{B} = \frac{1}{|B|}\sum_{z\in B}\delta_z$ where $B$ is a sufficiently large dataset sampled by $\mathcal{D}$ . Substituting $\mathcal{B}$ for $\mathcal{D}$ in Eq. (2), observe that there exists another probability vector $\omega \in \mathbb{R}^m$ such that

$$
\psi (z; \mathcal {B}, \mathbf {w}, U) = \varphi (z; \mathcal {B}, \boldsymbol {\omega}, U) := \underset {s \sim [ m ]} {\mathbb {E}} \underset {S \sim \mathcal {B} _ {s - 1}} {\mathbb {E}} [ U (S \cup z) - U (S) ] \tag {7}
$$

where $\mathcal{U}$ represents the uniform sampling and $\mathcal{B}_{s-1} = \{R \subseteq B \mid |R| = s - 1\}$ . We point out $\varphi$ is more natural to analyze. If $z \notin B$ , Eq. (7) is equal to calculating some probabilistic value for $z$ with the domain of utility function being $2^{B \cup z}$ ; if $z \in B$ , it produces some probabilistic value of $z$ up to some scalar, with the domain of utility function being $2^B$ .

Theorem 1. Consider the case when $B = [n]$ (recall that $D_{tr} \equiv [n]$ ) for the empirical distributional values $\varphi$ defined in Eq. (7). Let $\mathbf{d} \in \mathbb{R}^m$ be a probability vector that satisfies $d_s \propto \frac{n - s + 1}{s}\omega_s$ for every $s \in [m]$ , and $\mathbf{v}^*$ be the uniquely optimal solution to

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n + 1}} {\arg \min} \mathbb {E} _ {s \stackrel {\mathrm{d}} {\sim} [ m ]} \mathbb {E} _ {S \stackrel {\mathcal {U}} {\sim} \overline {{\mathcal {B}}} _ {s}} \left(U ^ {n} (S \cap [ n ]) - \sum_ {i \in S} v _ {i}\right) ^ {2}, \tag {8}
$$

where $\overline{\mathcal{B}}_s = \{R\subseteq [n + 1]\mid |R| = s\}$ . There is, for every $i\in [n]$ ,

$$
C \cdot (v _ {i} ^ {*} - v _ {n + 1} ^ {*}) = \varphi (i; \mathcal {B}, \boldsymbol {\omega}, U ^ {n}) \text {   where   } C = \sum_ {s = 1} ^ {m} \frac {n - s + 1}{n} \omega_ {s}.
$$

Additionally, setting $p_s^n \propto \binom{n}{s-1}^{-1} \omega_s$ if $s \in [m]$ and 0 otherwise such that $\mathbf{p}^n$ defines a probabilistic value, $\hat{\mathbf{r}}$ obtained in Algorithm 2 meets that $\mathbb{E}[\hat{r}_i] - \mathbb{E}[\hat{r}_{n+1}] = \varphi(i; \mathcal{B}, \boldsymbol{\omega}, U^n)$ for every $i \in [n]$ .

Theorem 1 constitutes a theoretically unsupervised ground for training estimators towards distributional values. Precisely, let $\phi_{\theta}(\mathbf{x},y)$ be a trainable model parameterized by $\theta$ ; $\mathbf{x}$ and $y$ stands for features and label, respectively; we propose training the estimator $\phi_{\theta}$ based on the specified utility function $U^n\in \mathcal{G}^n$ through optimizing

$$
\underset {\boldsymbol {\theta}, \phi_ {o}} {\arg \min} \underset {s \stackrel {{\mathbf {d}}} {{\sim}} [ m ]} {\mathbb {E}} \underset {S \stackrel {{\mathcal {U}}} {{\sim}} \overline {{\mathcal {B}}} _ {s}} {\mathbb {E}} \left(U ^ {n} (S \cap [ n ]) - \phi_ {o} \mathbb {1} _ {n + 1 \in S} - \sum_ {i \in S \cap [ n ]} \phi_ {\boldsymbol {\theta}} (\mathbf {x} _ {i}, y _ {i})\right) ^ {2}. \tag {9}
$$

As will be seen in the experiments, for any data point $z \notin B$ , the trained $\phi_{\theta}$ indeed predicts quite accurately the corresponding probabilistic value of $z$ with the domain of utility function being $2^{B \cup z}!$

Generally, it is impractical to train estimators using the exact distributional values, or regressing with a reasonable approximation as supervised signals, since it is extremely expensive to calculate them exactly. For example, in our experiments where N = 10,000 and m = 1,000, each unseen data point requires $2 \sum_{s=0}^{999} \binom{10,000}{s}$ utility evaluations to compute exactly. Therefore, the merits of training estimators through optimizing the problem (9) are clear: i) we do not have to be concerned with what the exact distributional values are, and ii) Theorem 1 guarantees that each estimator will be trained towards the exact distributional values. The outline of such an unsupervised framework is summarized in Algorithm 4.

Algorithm 4: TrELS (Training Estimators based on Least Squares)   
Input: A database $D_{tr} \equiv (\mathbf{x}_i, y_i)_{1 \leq i \leq n}$ , a probability vector $\omega \in R^m$ with m < n, a utility function $U^n \in G^n$ , a trainable model $\phi_\theta$ with an additional trainable parameter $\phi_o$ , a batch size Z and a total number T of batches used for training.

1 Compute $d \in R^m$ by letting $d_s = \frac{n-s+1}{s} \omega_s$ for every $s \in [m]$ 2 Normalize: $d \leftarrow d / \sum_{s=1}^m d_s$ 3 for $t = 1, 2, \ldots, T$ do

4 loss $\leftarrow 0$ 5 for $j = 1, 2, \ldots, Z$ do

6 Sample $s_j$ from [m] according to the distribution vector d

7 Sample $S_j$ uniformly from $\{R \subseteq [n+1] \mid |R| = s_j\}$ 8 loss $\leftarrow loss + (U^n(S_j \cap [n]) - \phi_o \mathbb{1}_{n+1 \in S_j} - \sum_{i \in S_j \cap [n]} \phi_\theta(\mathbf{x}_i, y_i))^2$ 9 loss $\leftarrow loss / Z$ 10 Update $\theta$ and $\phi_o$ using loss

Evaluation Phase: $C \cdot (\phi_\theta(\mathbf{x}, y) - \phi_o)$ // $C = \sum_{s=1}^m \frac{n-s+1}{n} \omega_s$

# 4 EVALUATION

In this section we perform experiments to i) verify the faster convergence of Algorithms 1 and 2, and ii) demonstrate that distributional values can be effectively predicted by trained estimators using TrELS. The classification datasets employed are from open resources, which are iris, wind (both are from OpenML), FMNIST (Xiao et al. 2017) and MNIST. For all utility functions, their outputs are set to be the performance of the trained models reported on a dataset $D_{perf}$ disjoint from $D_{tr}$ . Precisely, $U^n(S)$ reports the performance of the specified model trained on $S \subseteq D_{tr} \equiv [n]$ , and we leave the specified model untrained for evaluating $U^n(\emptyset)$ . Without being stated explicitly, the performance of trained models is measured by classification accuracy. In addition, we fix the random seed to be 2024 to make all utility functions deterministic. To be computationally efficient, we adopt one-mini-batch one-epoch learning for evaluating utility functions (Ghorbani and Zou 2019). Logistic regression is implemented for iris and wind, while LeNet (LeCun et al. 1998) is employed for FMNIST and MNIST.

# 4.1 FASTER CONVERGENCE OF THE PROPOSED ESTIMATORS

This experiment is performed on all the above-mentioned datasets. The SGD optimizer with a learning rate 1.0 is employed for iris and wind, whereas we set the learning rate to be 0.1 instead for MNIST and FMNIST. To evaluate the specified probabilistic values exactly, we set $|D_{tr}| = |D_{perf}| = 24$ . All estimators are run 30 times with different random seeds, which range from 0 to 29. We compare to the following benchmarks:

- those designed specifically for the Shapley value, including the complement (Zhang et al. 2023b), kernelSHAP (Lundberg and Lee 2017), unbiased KernelSHAP (Covert and Lee 2021), group testing (Jia et al. 2019), and simSHAP (Zhang et al. 2023a);   
- those work for all probabilistic values, including the sampling lift (see Proposition 5) and its weighted variant (Kwon and Zou 2022a), and the generalized ARM (see Proposition 8);   
- AME (Lin et al. 2022), restricted to a subfamily of probabilistic values;   
- MSR (Wang and Jia 2023), proposed specifically for the Banzhaf value.

The probabilistic values we consider are: i) Beta(1,1), which equals to the Shapley value, ii) Beta(2,2), $^{1}$ iii) Beta(4,1), and iv) the Banzhaf value. Particularly, AME does not apply to Beta(4,1) and the Shapley since both of them lead to $M_{P}=\infty$ .

![](images/b1089e275a4a1bfe3cef59ce41c4fd8354fb2b0dce284220d2c1148c525829ab.jpg)  
Figure 1: Comparison of different estimators using the dataset wind. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ .

Remark 4. Covert and Lee (2021) proposed the paired sampling technique to enhance (unbiased) kernelSHAP, and we point out that this technique can also be employed for our methods, (weighted) sampling lift, group testing, AME, MSR and simSHAP. We only compare the plain estimators in the main paper, and experiments including the paired sampling are provided in Appendix J.

We present some of the results in Figure 1, while deferring others to Appendix J. These results support that GELS, GELS-R are currently among the (empirically) fastest tier in the group of generic estimators. For the Shapley value, we omit the results of GELS-R since GELS-R and GELS-Shapley share the same ranking; GELS-Shapley converges significantly faster than GELS; however, GELS-Shapley may converge slower compared with the complement and kernelSHPA; notice that their faster convergence comes at the cost of using $\Theta(n^2)$ memory storage instead of $\Theta(n)$ ; precisely, the complement requires another $O(n^2)$ time complexity to aggregate all $n^2$ estimates, whereas kernelSHAP needs an extra $O(n^3)$ time complexity for calculating the inverse of matrices. For the Banzhaf value, though GELS converges slower than MSR in terms of relative error, the performance of GELS and GELS-ranking in terms of Spearman correlation is not worse than MSR, and sometimes better. For the Beta(2, 2), GELS and GELS-R achieve the best. We notice that the generalized ARM almost enjoys the same empirical convergence across different experiment settings, except that it is slower than GELS-Shapley in terms of relative error.

# 4.2 TRAINING ESTIMATORS

In this experiment, we demonstrate the effectiveness of training estimators using Algorithm 4 on FMNIST and MNIST. The corresponding results for MNIST are included in Appendix J. For the utility functions, the SGD optimizer with a learning rate 0.01 and one-mini-batch one-epoch learning is adopted, and we take $|D_{tr}| = 10,000$ and $|D_{perf}| = 500$ from the training set. LeNet is the architecture we employ as a trainable estimator $\phi_{\theta}$ , and the input of the softmax layer is taken as the output of $\phi_{\theta}$ . For training estimators, we employ the Adam optimizer (Kingma and Ba 2014) with a learning rate 0.001. The batch size $Z$ is set to be 10,000, and we randomly generate 1,000 batches. After these batches have all been fed into the estimators, we permute and reuse them to continue training. Therefore, we have in total 1,000 utility evaluations per data point for training estimators. Lastly, the probability vector $\omega \in \mathbb{R}^{1,000}$ we employ is $\omega_s \propto s^{-\frac{1}{2}}$ . Moreover, 200 data, denoted by $D_{val}$ , are taken from the training dataset for selecting the best trained models. For the results in the second row of Figure 2, we report on another 200 data extracted from the test dataset, for which we refer to as $D_{test}$ . To sum, $D_{tr}, D_{perf}, D_{val}$ and $D_{test}$ are all disjoint, which means $D_{val}$ and $D_{test}$ are composed of unseen data. For $D_{val}$ and $D_{test}$ , we randomly run 600,000 utility evaluations for each data point using Eq. (7) to generate estimates of the specified distributional value, which are

![](images/acc9d5013bde5a3ccfb8a8d3ec833440ad7d97cf2ffa7fdd0002bad8daeb09b4.jpg)

<details>
<summary>line</summary>

| #batches (k) | empirical loss |
| ------------ | -------------- |
| 0            | ~1.0           |
| 5            | ~0.4           |
| 10           | ~0.3           |
| 15           | ~0.25          |
| 20           | ~0.2           |
| 25           | ~0.18          |
| 30           | ~0.17          |
</details>

![](images/6f9243e2a00a54cdc3cb2dc214b7b8678323aedd272432d6a0a7526e2ced1881.jpg)

<details>
<summary>line</summary>

| #batches (k) | specific | average | best |
| ------------ | -------- | ------- | ---- |
| 0            | 10^1     | 10^1    | 10^1 |
| 5            | ~10^0.5  | ~10^0.5 | ~10^0.5 |
| 10           | ~10^0.5  | ~10^0.5 | ~10^0.2 |
| 15           | ~10^0.5  | ~10^0.5 | ~10^0.2 |
| 20           | ~10^0.5  | ~10^0.5 | ~10^0.2 |
| 25           | ~10^0.5  | ~10^0.5 | ~10^0.2 |
| 30           | ~10^0.5  | ~10^0.5 | ~10^0.2 |
</details>

![](images/224743a7371ff910a58b98c2cdf2a60c9d941217e38957dffbc852908c68837a.jpg)

<details>
<summary>line</summary>

| #batches (k) | specific | average | best |
| ------------ | -------- | ------- | ---- |
| 0            | 0.0      | 0.0     | 0.0  |
| 5            | 0.6      | 0.5     | 0.7  |
| 10           | 0.8      | 0.7     | 0.9  |
| 15           | 0.85     | 0.75    | 0.95 |
| 20           | 0.85     | 0.75    | 0.95 |
| 25           | 0.85     | 0.75    | 0.95 |
| 30           | 0.85     | 0.75    | 0.95 |
</details>

![](images/158088d1277129c2dc1f033a6d40d09c1fcc958292389e09983989f0181123ee.jpg)

<details>
<summary>line</summary>

| #utility evaluations per datum (k) | Monte Carlo | TrELS-SC (ours) | TrELS-RD (ours) |
| ---------------------------------- | ----------- | --------------- | --------------- |
| 0                                  | ~10^1       | ~10^1           | ~10^0           |
| 5                                  | ~10^0.5     | ~10^1           | ~10^0           |
| 10                                 | ~10^0.3     | ~10^1           | ~10^0           |
| 15                                 | ~10^0.2     | ~10^1           | ~10^0           |
| 20                                 | ~10^0.1     | ~10^1           | ~10^0           |
</details>

![](images/a4141160d55ecb20cc0d02a4ac9d6be0fb646fef0c151a1efb7ceea8064738c0.jpg)

<details>
<summary>line</summary>

| #utility evaluations per datum (k) | Monte Carlo | TrELS-SC (ours) | TrELS-RD (ours) |
| ---------------------------------- | ----------- | --------------- | --------------- |
| 0                                  | 0.2         | 0.85            | 0.85            |
| 5                                  | 0.6         | 0.85            | 0.85            |
| 10                                 | 0.75        | 0.85            | 0.85            |
| 15                                 | 0.8         | 0.85            | 0.85            |
| 20                                 | 0.85        | 0.85            | 0.85            |
</details>

Figure 2: i) The first row: The first plot is the empirical loss of the sampled batch, while the others present the relative distance $\|\hat{\varphi}-\varphi\|_{2}/\|\varphi\|_{2}$ and the Spearman correlation between $\varphi$ and $\hat{\varphi}$ . $\hat{\varphi}$ denotes the one predicted by the estimators trained on FMNIST. The first two plots are in log scale. The label “best” reports the best one achieved by the trained estimators at each time point, whereas “specific” is the one with the random seed being 0. ii) The second row: the two plots compare the performance of the estimators trained on FMNIST with the Monte-Carlo method using Eq (2). Specifically, TrELS-SC uses trained estimators that achieve the best Spearman correlation on $D_{val}$ , whereas RD stands for relative difference.

regarded as the ground-truths $\varphi$ . All results are reported with mean and standard deviation using 30 different random seeds ranging from 0 to 29.

The performance curves of the estimators during training are shown in Figure 2. First of all, the increasing curve in terms of the Spearman correlation indicates that the estimators trained under TrELS are able to gradually learn the exact ranking. Secondly, the decreasing curve (the best one) of the relative difference suggests that the transform $C \cdot (\phi_{\theta}(\mathbf{x}, y) - \phi_o)$ is essential for training!

Next, we examine the efficiency of the trained estimators by comparing them with the Monte-Carlo method based on Eq. (7). Specifically, for each random seed, the trained estimators that achieve the best performance on $D_{val}$ are selected. The comparison is shown in the second row of Figure 2. As clearly shown, the Monte-Carlo methods using 20,000 utility evaluations for each data point are still inferior to the estimators trained under our TrELS. Observe that TrELS-SC performs poorly in terms of relative difference, which is expected since the reported relative differences during training are significantly unstable.

# 5 CONCLUSION

In this work, we start from the least square regression to develop GELS with its variants for all probabilistic values. GELS-R is to approximate the ranking of probabilistic values, whereas GELS-Shapley is specifically designed for the Shapley value. The faster convergence of the proposed estimators is theoretically guaranteed, and is also verified empirically in our experiments. Besides, we also demonstrate how to cast each distributional value into a least square problem, making it the first-time theoretically-grounded to train estimators towards distributional values in an unsupervised manner, the framework of which is introduced as TrELS. Notably, our experiments show that the estimators trained under TrELS learn the specified distributional values quite well in terms of both relative difference and Spearman correlation. Our work significantly broadens the practicality of deploying value-based data valuation methods on rather large datasets.

# ACKNOWLEDGEMENTS

We thank the reviewers and the area chair for thoughtful comments that have improved our final presentation. YY gratefully acknowledges NSERC and CIFAR for funding support.

# REFERENCES

Banzhaf III, J. F. (1965). “Weighted Voting Doesn’t Work: A Mathematical Analysis”. Rutgers Law Review, vol. 19, pp. 317–343.   
Charnes, A., B. Golany, M. S. Keane, and J. J. Rousseau (1988). “Extremal Principle Solutions of Games in Characteristic Function Form: Core, Chebychev and Shapley Value Generalizations”. In: Econometrics of Planning and Efficiency, pp. 123–133.   
Covert, I. and S.-I. Lee (2021). “Improving KernelSHAP: Practical Shapley Value Estimation Using Linear Regression”. In: Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, pp. 3457–3465.   
Covert, I. C., C. Kim, and S.-I. Lee (2023). “Learning to Estimate Shapley Values with Vision Transformers”. In: The Eleventh International Conference on Learning Representations.   
Dubey, P., A. Neyman, and R. J. Weber (1981). “Value Theory without Efficiency”. Mathematics of Operations Research, vol. 6, no. 1, pp. 122–128.   
Fumagalli, F., M. Muschalik, P. Kolpaczki, E. Hüllermeier, and B. E. Hammer (2023). “SHAP-IQ: Unified Approximation of any-order Shapley Interactions”. In: Thirty-seventh Conference on Neural Information Processing Systems.   
Ghorbani, A., M. Kim, and J. Zou (2020). “A Distributional Framework for Data Valuation”. In: International Conference on Machine Learning, pp. 3535–3544.   
Ghorbani, A. and J. Zou (2019). “Data Shapley: Equitable Valuation of Data for Machine Learning”. In: International Conference on Machine Learning, pp. 2242–2251.   
Grabisch, M. and C. Labreuche (2001). “How to Improve Acts: An Alternative Representation of the Importance of Criteria in MCDC”. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, vol. 9, no. 02, pp. 145–157.   
Harsanyi, J. C. (1963). “A Simplified Bargaining Model for the N-Person Cooperative Game”. International Economic Review, vol. 4, no. 2, pp. 194–220.   
Jethani, N., M. Sudarshan, I. C. Covert, S.-I. Lee, and R. Ranganath (2022). “FastSHAP: Real-Time Shapley Value Estimation”. In: International Conference on Learning Representations.   
Jia, R. et al. (2019). “Towards Efficient Data Valuation Based on the Shapley Value”. In: The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1167–1176.   
Kingma, D. P. and J. Ba (2014). “Adam: A Method for Stochastic Optimization”. In: ICLR.   
Kolpaczki, P., V. Bengs, and E. Hüllermeier (2023). “Approximating the Shapley Value without Marginal Contributions”. arXiv preprint arXiv:2302.00736.   
Kwon, Y. and J. Zou (2022a). “Beta Shapley: A Unified and Noise-reduced Data Valuation Framework for Machine Learning”. In: International Conference on Artificial Intelligence and Statistics, pp. 8780–8802.   
Kwon, Y. and J. Y. Zou (2022b). “WeightedSHAP: Analyzing and Improving Shapley Based Feature Attributions”. In: Advances in Neural Information Processing Systems, pp. 34363–34376.   
LeCun, Y., L. Bottou, Y. Bengio, and P. Haffner (1998). “Gradient-Based Learning Applied to Document Recognition”. Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324.   
Lin, J., A. Zhang, M. Lécuyer, J. Li, A. Panda, and S. Sen (2022). “Measuring the Effect of Training Data on Deep Learning Predictions via Randomized Experiments”. In: International Conference on Machine Learning, pp. 13468–13504.

Lundberg, S. M. and S.-I. Lee (2017). “A Unified Approach to Interpreting Model Predictions”. In: Advances in Neural Information Processing Systems 30.   
Marichal, J.-L. and P. Mathonet (2008). “Approximations of Lovász Extensions and Their Induced Interaction Index”. Discrete Applied Mathematics, vol. 156, no. 1, pp. 11–24.   
Moehle, N., S. Boyd, and A. Ang (2022). “Portfolio Performance Attribution via Shapley Value”. Journal of Investment Management, vol. 20, no. 3.   
Rudin, W. (1953). “Principles of Mathematical Analysis”. McGraw-Hill.   
Ruiz, L. M., F. Valenciano, and J. M. Zarzuelo (1998). “The Family of Least Square Values for Transferable Utility Games”. Games and Economic Behavior, vol. 24, no. 1-2, pp. 109–130.   
Shapley, L. S. (1953). “A Value for N-Person Games”. Annals of Mathematics Studies, vol. 28, pp. 307–317.   
Wang, J. and R. Jia (2023). “Data Banzhaf: A Robust Data Valuation Framework for Machine Learning”. In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics.   
Weber, R. J. (1977). “Probabilistic Values for Games”. In: The Shapley Value. Essays in Honor of Lloyd S. Shapley, pp. 101–119.   
Xiao, H., K. Rasul, and R. Vollgraf (2017). “Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms”. arXiv preprint arXiv:1708.07747.   
Zhang, B., B. Tian, W. Zheng, J. Zhou, and J. Lu (2023a). “Exploring Unified Perspective For Fast Shapley Value Estimation”. arXiv preprint arXiv:2311.01010.   
Zhang, J., Q. Sun, J. Liu, L. Xiong, J. Pei, and K. Ren (2023b). “Efficient Sampling Approaches to Shapley Value Approximation”. In: ACM SIGMOD International Conference on Management of Data.   
Zhou, Z., X. Xu, R. H. L. Sim, C. S. Foo, and B. K. H. Low (2023). “Probably Approximate Shapley Fairness with Applications in Machine Learning”. In: Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. 5, pp. 5910–5918.

Table of Contents 

<table><tr><td>A</td><td>Distributional Values</td><td>12</td></tr><tr><td>B</td><td>Proofs for the Proposed Algorithms</td><td>13</td></tr><tr><td>C</td><td>Proofs for Convergences</td><td>15</td></tr><tr><td>D</td><td>Least Square Values</td><td>18</td></tr><tr><td>E</td><td>Generalized ARM</td><td>19</td></tr><tr><td>F</td><td>Practical Probabilistic Values</td><td>20</td></tr><tr><td>G</td><td>Practical Aspect of Faster Estimators for Probabilistic Values</td><td>20</td></tr><tr><td>H</td><td>Interpretation of κ(N) and τ(N) in Proposition 4</td><td>22</td></tr><tr><td>I</td><td>Asymptotic Analysis</td><td>23</td></tr><tr><td>J</td><td>More Experiment Results</td><td>25</td></tr></table>

# A DISTRIBUTIONAL VALUES

Originally, (Ghorbani et al. 2020, Definition 2.2) proposed the Distributional Shapley value defined by

$$
\nu (z; \mathcal {D}, m, U) = \underset {S \sim \mathcal {D} ^ {m - 1}} {\mathbb {E}} [ \phi (z; U, S \cup z) ]
$$

$$
\text { where } \phi (z; U, S \cup z) = \sum_ {T \subseteq S} \frac {t ! (m - 1 - t) !}{m !} (U (T \cup z) - U (T)).
$$

Note that $\phi$ is supposed to be the Shapley value if $z \notin S$ and $|S| = m - 1$ . Then, Ghorbani et al. (2020, Theorem 2.3) proved that

$$
\nu (z; \mathcal {D}, m, U) = \underset {k \sim [ m ]} {\mathbb {E}} \underset {S \sim \mathcal {D} ^ {k - 1}} {\mathbb {E}} [ U (S \cup z) - U (S) ] \tag {10}
$$

where $\mathcal{U}$ refers to the uniform distribution. To improve the overall running time, they suggested using a weighted sampling $k\stackrel {\mathbf{w}}{\sim}[m]$ instead.

However, Eq. (10) holds only with an abuse of set operations. Precisely, repeated items are counted in their proof, e.g., $\{x,y,x\} = \{x,x,y\} \neq \{x,y\}$ and $\{z,x,z\} \cup z = \{z,x,z,z\} \neq \{z,x,z\}$ . In this work, we adhere to set operations. Regarding Eq. (2), samples from $\mathcal{D}^{s-1}$ are tuples that may contain repeated items, but they are reduced to be sets.

Adhering to set operations, Eq. (10) is false. Take $\mathcal{D} \leftarrow \frac{1}{3}\delta_x + \frac{1}{3}\delta_y + \frac{1}{3}\delta_z$ , $m \leftarrow 3$ as an example and assume $U(\emptyset) = 0$ , one can verify that

$$
\begin{array}{l} \nu (z; \mathcal {D}, m, U) = \frac {1}{3} U (\{z \}) + \frac {5}{5 4} (U (\{x, z \}) + U (\{y, z \}) - U (\{x \}) - U (\{y \})) \\ + \frac {2}{2 7} (U (\{x, y, z \}) - U (\{x, y \})), \\ \end{array}
$$

$$
\begin{array}{l} \underset {k \sim [ m ]} {\mathbb {E}} \underset {S \sim \mathcal {D} ^ {k - 1}} {\mathbb {E}} [ U (S \cup z) - U (S) ] = \frac {1}{3} U (\{z \}) + \frac {4}{2 7} (U (\{x, z \}) + U (\{y, z \}) - U (\{x \}) - U (\{y \})) \\ + \frac {2}{2 7} (U (\{x, y, z \}) - U (\{x, y \})). \\ \end{array}
$$

Therefore, we refer to Eq. (2) in accordance with set operations as distributional values.

# B PROOFS FOR THE PROPOSED ALGORITHMS

Lemma 1. Let $\mathbf{J}_n\in \mathbb{R}^{n\times n}$ be the all-one matrix. The matrix $\mathbf{A} = a\mathbf{J}_n + b\mathbf{I}_n$ is invertible if and only if $na + b\neq 0$ and $b\neq 0$ . Particularly, $\mathbf{A}^{-1} = -\frac{a}{b(na + b)}\mathbf{J}_n + \frac{1}{b}\mathbf{I}_n$ .

Proof. Observe that the eigenvalues of $J_{n}$ are 0 (with n - 1 independent eigenvectors) and n, and thus the eigenvalues of $aJ_{n} + bI_{n}$ are b and $na + b$ . Therefore, $aJ_{n} + bI_{n}$ is invertible if and only if $b \neq 0$ and $na + b \neq 0$ .

A priori is that $A^{-1}$ admits the form of $xJ_{n} + yI_{n}$ . Using $AA^{-1} = I_{n}$ , we have the equation

$$
(n a + b) x + (a + b) y = 1,
$$

$$
(n a + b) x + a y = 0,
$$

which leads to that $x = -\frac{a}{b(na + b)}$ and $y = \frac{1}{b}$ .

![](images/441e6f3c1d69e60aff1f1296a044b7d14d6e97748c4e4df775b2a26f0d0302b8.jpg)

Proposition 1. Define, for every $n \geq 1$ , $U^{n} \in G^{n}$ and $i \in [n]$ ,

$$
\mathcal {R} _ {i} (U ^ {n}) = \mathcal {R} _ {i} (U ^ {n}; \mathcal {P}) = \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} m _ {s} ^ {n} \cdot \mathbb {1} _ {i \in S} \cdot U ^ {n} (S) \tag {4}
$$

where $m_s^n = m_s^n(\mathcal{P}) = p_s^n + p_{s+1}^n$ for every $s \in [n-1]$ . Then, for every $n \geq 1$ and $U^n \in \mathcal{G}^n$ , $\mathcal{R}(U^n)$ and $\phi(U^n)$ produce the same ranking. Precisely, $\mathcal{R}(U^n) = \phi(U^n) + g(U^n)\mathbf{1}_n$ where $g(U^n) = g(U^n; \mathcal{P}) \in \mathbb{R}$ . As a side note, it holds for any $\mathbf{p}^n \in \mathbb{R}^n$ .

Proof.

$$
\begin{array}{l} \phi_ {i} (U ^ {n}) = \sum_ {S \subseteq [ n ] \backslash i} p _ {s + 1} ^ {n} (U ^ {n} (S \cup i) - U ^ {n} (S)) \\ = \sum_ {S \subsetneq [ n ]: i \in S} m _ {s} ^ {n} U ^ {n} (S) + \left[ p _ {n} ^ {n} U ^ {n} ([ n ]) - \sum_ {S \subsetneq [ n ]} p _ {s + 1} ^ {n} U ^ {n} (S) \right] \tag {11} \\ = \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} m _ {s} ^ {n} \cdot \mathbb {1} _ {i \in S} \cdot U ^ {n} (S) + g (U ^ {n}). \\ \end{array}
$$

![](images/e0eb246e23ea05e19be7902a7ac1c1fc09448a5c864440c1c50fed52bf507bb4.jpg)

Proposition 2. Consider the problem

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n}} {\arg \min} \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} m _ {s} ^ {n} \cdot \left(U ^ {n} (S) - \sum_ {i \in S} v _ {i}\right) ^ {2}, \tag {5}
$$

its uniquely optimal solution shares the same ranking as $\phi(U^{n})$ . As a side note, it holds true for any $p^{n} \in R^{n}$ that produces a non-negative weight vector $m^{n}$ with $\sum_{s=1}^{n-1} m_{s}^{n} > 0$ .

Proof. Let $\mathbf{J}_n\in \mathbb{R}^{n\times n}$ be the all-one matrix, and $\mathbf{I}_n\in \mathbb{R}^{n\times n}$ be the identity matrix. Since the problem (5) is convex, its optimal solution can be obtained by letting its derivative equal 0, which yields

$$
\mathbf {A} \mathbf {v} ^ {*} = \mathbf {b}
$$

$$
\text { where } \mathbf {A} = e \mathbf {J} _ {n} + d \mathbf {I} _ {n} \text { with } e = \sum_ {s = 2} ^ {n - 1} \binom {n - 2} {s - 2} m _ {s} ^ {n} \text { and } d = \left(\sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s - 1} m _ {s} ^ {n}\right) - e, \tag {12}
$$

$$
\text { and } \mathbf {b} _ {i} = \sum_ {S \subsetneq [ n ]: i \in S} m _ {s} ^ {n} \cdot U ^ {n} (S) \text { for   every } i \in [ n ].
$$

Specifically,

$$
\begin{array}{l} d = m _ {1} ^ {n} + \sum_ {s = 2} ^ {n - 1} \left(\binom{n - 2}{s - 1} + \binom{n - 2}{s - 2}\right) m _ {s} ^ {n} - \sum_ {s = 2} ^ {n - 1} \binom{n - 2}{s - 2} m _ {s} ^ {n} = \sum_ {s = 1} ^ {n - 1} \binom{n - 2}{s - 1} m _ {s} ^ {n} \\ = \sum_ {s = 1} ^ {n - 1} \binom {n - 2} {s - 1} p _ {s} ^ {n} + \sum_ {s = 1} ^ {n - 1} \binom {n - 2} {s - 1} p _ {s + 1} ^ {n} = p _ {1} ^ {n} + \sum_ {s = 2} ^ {n - 1} \binom {n - 2} {s - 1} p _ {s} ^ {n} + \sum_ {s = 2} ^ {n - 1} \binom {n - 2} {s - 2} p _ {s} ^ {n} + p _ {n} ^ {n} \\ = \sum_ {s = 1} ^ {n} \binom {n - 1} {s - 1} p _ {s} ^ {n} = 1. \\ \end{array}
$$

By Lemma 1, $\mathbf{A}^{-1} = -\frac{e}{e\cdot n + 1}\mathbf{J}_n + \mathbf{I}_n$ , which implies the uniqueness of the optimal solution. Therefore,

$$
\mathbf {v} ^ {*} = \mathbf {A} ^ {- 1} \mathbf {b} = \mathbf {b} + c \mathbf {1} _ {n} \tag {13}
$$

where $c = \frac{-e}{e\cdot n + 1}\sum_{i = 1}^{n}b_{i}$ . According to Eq. (4), we have $\mathbf{b} = \mathcal{R}(U^n)$ . Therefore, $\mathbf{v}^*$ and $\phi (U^n)$ share the same ranking. Suppose we have $\mathbf{p}^n\in \mathbb{R}^n$ with $\sum_{s = 1}^{n - 1}m_s^n >0$ , then there is $\mathbf{v}^* = \frac{1}{d}\mathbf{b} + \hat{c}\mathbf{1}_n$ instead where $\hat{c} = \frac{-e}{e\cdot n + d}\sum_{i = 1}^{n}b_{i}$ . Since $d = \sum_{s = 1}^{n - 1}\binom{n - 2}{s - 1}m_s^n >0$ , the factor $\frac{1}{d}$ does not change the ranking of $\mathbf{b}$ .

Proposition 3. The uniquely optimal solution $\mathbf{v}^*$ to the problem

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n + 1}} {\arg \min} \sum_ {\emptyset \subsetneq S \subsetneq [ n + 1 ]} p _ {s} ^ {n} \cdot \left(U ^ {n} (S \cap [ n ]) - \sum_ {i \in S} v _ {i}\right) ^ {2} \tag {6}
$$

satisfies that $\phi_i(U^n) = v_i^* - v_{n+1}^*$ for every $i \in [n]$ .

Proof. For each $U^n \in \mathcal{G}^n$ , define $\overline{U}^{n+1} \in \mathcal{G}^{n+1}$ by letting $\overline{U}^{n+1}(S) = U^n (S \cap [n])$ . Substituting $\overline{U}^{n+1}(S)$ for $U^n (S \cap [n])$ in the problem (6), and reusing the Eq. (12) accordingly, we have

$$
\begin{array}{l} \mathbf {v} ^ {*} = \mathbf {A} ^ {- 1} \mathbf {b} \\ \text { where } b _ {i} = p _ {n} ^ {n} \cdot U ^ {n} ([ n ]) + \sum_ {S \subsetneq [ n ]: i \in S} (p _ {s} ^ {n} + p _ {s + 1} ^ {n}) \cdot U ^ {n} (S) \quad \forall i \in [ n ], \\ \text { and } b _ {n + 1} = \sum_ {S \subsetneq [ n ]} p _ {s + 1} ^ {n} \cdot U ^ {n} (S). \\ \end{array}
$$

Using Eq. (11), we obtain $\phi_i(U^n) = b_i - b_{n+1}$ for every $i \in [n]$ . Meanwhile, by Eq. (13), we have $\mathbf{v}^* = \mathbf{b} + c\mathbf{1}_{n+1}$ for some $c \in \mathbb{R}$ , and thus $v_i - v_{n+1} = b_i - b_{n+1} = \phi_i(U^n)$ for every $i \in [n]$ .

Theorem 1. Consider the case when $B = [n]$ (recall that $D_{tr} \equiv [n]$ ) for the empirical distributional values $\varphi$ defined in Eq. (7). Let $\mathbf{d} \in \mathbb{R}^m$ be a probability vector that satisfies $d_s \propto \frac{n - s + 1}{s}\omega_s$ for every $s \in [m]$ , and $\mathbf{v}^*$ be the uniquely optimal solution to

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n + 1}} {\arg \min} \mathbb {E} _ {s \stackrel {\mathrm{d}} {\sim} [ m ]} \mathbb {E} _ {S \stackrel {\mathcal {U}} {\sim} \overline {{\mathcal {B}}} _ {s}} \left(U ^ {n} (S \cap [ n ]) - \sum_ {i \in S} v _ {i}\right) ^ {2}, \tag {8}
$$

where $\overline{\mathcal{B}}_s = \{R\subseteq [n + 1]\mid |R| = s\}$ . There is, for every $i\in [n]$ ,

$$
C \cdot (v _ {i} ^ {*} - v _ {n + 1} ^ {*}) = \varphi (i; \mathcal {B}, \boldsymbol {\omega}, U ^ {n}) \text {   where   } C = \sum_ {s = 1} ^ {m} \frac {n - s + 1}{n} \omega_ {s}.
$$

Additionally, setting $p_s^n \propto \binom{n}{s-1}^{-1} \omega_s$ if $s \in [m]$ and 0 otherwise such that $\mathbf{p}^n$ defines a probabilistic value, $\hat{\mathbf{r}}$ obtained in Algorithm 2 meets that $\mathbb{E}[\hat{r}_i] - \mathbb{E}[\hat{r}_{n+1}] = \varphi(i; \mathcal{B}, \omega, U^n)$ for every $i \in [n]$ .

Proof. Since $U^n(S \cup i) - U^n(S) = 0$ if $i \in S$ , for each $i \in [n]$ ,

$$
\begin{array}{l} \varphi (i; \mathcal {B}, \boldsymbol {\omega}, U ^ {n}) = \sum_ {S \subseteq [ n ] \backslash i: s <   m} \omega_ {s + 1} \binom {n} {s} ^ {- 1} (U ^ {n} (S \cup i) - U ^ {n} (S)) \\ = C \cdot \sum_ {S \subseteq [ n ] \backslash i} q _ {s + 1} \cdot (U ^ {n} (S \cup i) - U ^ {n} (S)) \\ \end{array}
$$

where $q_{s} = \binom{n}{s-1}^{-1}\frac{\omega_{s}}{C}$ if $s \in [m]$ and 0 otherwise, and $C = \sum_{s=1}^{m} \binom{n-1}{s-1} \binom{n}{s-1}^{-1} \omega_{s} = \sum_{s=1}^{m} \frac{n-s+1}{n} \omega_{s}$ . On the other hand, the objective in the problem (8) can be rewritten as

$$
\sum_ {\emptyset \subsetneq S \subsetneq [ n + 1 ]: s \leq m} d _ {s} \binom {n} {s} ^ {- 1} \left(\overline {{U}} ^ {n + 1} (S) - \sum_ {i \in S} v _ {i}\right) ^ {2}
$$

where $\overline{U}^{n + 1}\in \mathcal{G}^{n + 1}$ is defined by letting $\overline{U}^{n + 1}(S) = U^n (S\cap [n])$ for every $S\subseteq [n + 1]$ . Note that $d_{s} = \alpha^{\frac{n - s + 1}{s}}\omega_{s}$ for every $s\in [m]$ where $\alpha$ is a scalar that makes $\mathbf{d}$ a probability vector. By Proposition 3, for every $i\in [n]$ ,

$$
v _ {i} ^ {*} - v _ {n + 1} ^ {*} = \sum_ {S \subseteq [ n ] \backslash i} \hat {q} _ {s + 1} \cdot (U ^ {n} (S \cup i) - U ^ {n} (S))
$$

where $\hat{q}_s = \beta \cdot d_s\binom{n}{s}^{-1}$ for every $s\in [m]$ and 0 otherwise. Suffice it to show that $\hat{q}_s = q_s$ for every $s\in [m]$ . Note that

$$
1 = \sum_ {s = 1} ^ {m} \binom {n - 1} {s - 1} \hat {q _ {s}} = \sum_ {s = 1} ^ {m} \binom {n - 1} {s - 1} \beta \cdot \alpha \frac {n - s + 1}{s} \omega_ {s} \binom {n} {s} ^ {- 1} = \alpha \cdot \beta \cdot C,
$$

and thus

$$
q _ {s} = \beta \cdot \alpha \binom{n}{s - 1} ^ {- 1} \omega_ {s} = \beta \binom{n}{s - 1} ^ {- 1} \frac {s}{n - s + 1} d _ {s} = \beta \binom{n}{s} ^ {- 1} d _ {s} = \hat {q} _ {s}.
$$

For the side note, note that

$$
L \cdot (\mathbb {E} [ \hat {r} _ {i} ] - \mathbb {E} [ \hat {r} _ {n + 1} ]) = v _ {i} ^ {*} - v _ {n + 1} ^ {*}
$$

where

$$
L = \sum_ {s = 1} ^ {n} \frac {s}{n + 1} \binom {n + 1} {s} q _ {s} = \sum_ {s = 1} ^ {m} \frac {s}{n + 1} \binom {n + 1} {s} \binom {n} {s - 1} ^ {- 1} \frac {\omega_ {s}}{C} = \frac {1}{C}.
$$

![](images/5bb9b3cc2dac592233a4fbecf8d5ae198e2ac6255017555e37fab15aa86b3caf.jpg)

# C PROOFS FOR CONVERGENCES

Proposition 5. Assume that $\| U^n\|_{\infty}\leq u$ for every $U^n\in \bigcup_{n\geq 1}\mathcal{G}^n$ , any estimator based on the sampling lift strategy requires $O\left(\frac{n^2}{\epsilon^2}\log \frac{n}{\delta}\right)$ utility evaluations to achieve $P(\| \hat{\phi} (U^n) - \phi (U^n)\| _2\geq$ $\epsilon)\leq \delta$ .

The sampling lift strategy refers to any estimator designed according to

$$
\phi_ {i} (U ^ {n}) = \sum_ {S \subseteq [ n ] \setminus i} p _ {s + 1} ^ {n} \left(U ^ {n} (S \cup i) - U ^ {n} (S)\right) = \mathbb {E} _ {S} [ U ^ {n} (S \cup i) - U ^ {n} (S) ]
$$

for every $i \in [n]$ as $\sum_{s=1}^{n} \binom{n-1}{s-1} p_{s}^{n} = 1$ . Precisely, $U^{n}(S \cup i) - U^{n}(S)$ is a random variable following a certain probability distribution, denoted by $P_{i}$ , on all subsets $S \subseteq [n] \setminus i$ . Let T be the total number of samples and $\mathcal{S} = \{(S_{1}^{1}, S_{1}^{2}, \ldots, S_{1}^{n}), (S_{2}^{1}, S_{2}^{2}, \ldots, S_{2}^{n}), \ldots, (S_{T}^{\perp}, S_{T}^{2}, \ldots, S_{T}^{n})\}$ be a list of T samples where each $S^{i} = \{S_{1}^{i}, S_{2}^{i}, \ldots, S_{T}^{i}\}$ are sampled from $P_{i}$ . Then, the estimator based on the sampling lift strategy is $\hat{\phi}_{i}(U^{n}) = \frac{1}{T} \sum_{k=1}^{T} (U^{n}(S_{k}^{i} \cup i) - U^{n}(S_{k}^{i}))$ . There are two slightly different implementations: i) sampling each $S^{i}$ independently, and ii) sampling $S^{i}$ for some $i \in [n]$ , and then reusing all samples in $S^{i}$ for every other $j \in [n]$ by swapping i and j, the latter of which is our implementation. Nevertheless, the proof below works for both of them. Besides, the proof is adapted from (Wang and Jia 2023, Theorem 4.8) where it is stated for the Banzhaf value. Nevertheless, we point out that it applies to all probabilistic values.

Proof. For simplicity, we write $\phi = \phi(U^n)$ and $\hat{\phi} = \hat{\phi}(U^n)$ . Fix an $i \in [n]$ , since $\mathbb{E}[\hat{\phi}_i] = \phi_i$ , by the Hoeffding's inequality,

$$
P (| \hat {\phi} _ {i} - \phi_ {i} | \geq \epsilon) \leq 2 \exp (- \frac {T \epsilon^ {2}}{2 u ^ {2}}).
$$

Then,

$$
P (\| \hat {\phi} - \phi \| _ {2} \geq \epsilon) \leq P (\bigcup_ {i \in [ n ]} | \hat {\phi} _ {i} - \phi_ {i} | \geq \frac {\epsilon}{\sqrt {n}}) \leq 2 n \exp (- \frac {T \epsilon^ {2}}{2 n u ^ {2}})
$$

Letting $\delta \geq 2n\exp \left(-\frac{T\epsilon^2}{2nu^2}\right)$ leads to $T \geq \frac{2nu^2}{\epsilon^2} \log \frac{2n}{\delta}$ . Since each sample $(S_k^1, S_k^2, \ldots, S_k^n)$ invokes $2n$ utility evaluations, it requires $O\left(\frac{n^2}{\epsilon^2} \log \frac{n}{\delta}\right)$ utility evaluations to achieve an $(\epsilon, \delta)$ -approximation.

Proposition 4. Assume that $\| U^n\|_{\infty}\leq u$ for every $U^n\in \bigcup_{n\geq 1}\mathcal{G}^n$ . We have the following results: i) GELS requires $O(\frac{\tau(n)n}{\epsilon^2}\log \frac{n}{\delta})$ utility evaluations to achieve an $(\epsilon ,\delta)$ -approximation, i.e., $P(\| \hat{\phi} (U^n) - \phi (U^n)\| _2\geq \epsilon)\leq \delta$ ; ii) for GELS-R estimator, it requires $O(\frac{\kappa(n)n}{\epsilon^2}\log \frac{n}{\delta})$ utility evaluations instead; iii) plus, the corresponding convergence of GELS-Shapley is $O(\frac{n}{\epsilon^2}\log (n)^2\log \frac{n}{\delta})$ .

Proof. The proof is adapted from (Wang and Jia 2023, Theorem 4.9). First, we prove the convergence for GELS-R. Note that it is an unbiased estimator of

$$
\mathbf {r} = \left(\sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s - 1} (p _ {s} ^ {n} + p _ {s + 1} ^ {n})\right) ^ {- 1} \mathcal {R} (U ^ {n}). \tag {14}
$$

Let $\gamma(n) = \frac{\sum_{s=1}^{n-1}\binom{n-1}{s-1}(p_s^n + p_{s+1}^n)}{\sum_{s=1}^{n-1}\binom{n}{s}(p_s^n + p_{s+1}^n)}$ . For convenience, we write $S = \{S_1, S_2, \ldots, S_T\}$ that contains all sampled subsets, and $T_i = |\{S \in S \mid i \in S\}|$ for every $i \in [n]$ . For every $i \in [n]$ , define

$$
\overline {{r}} _ {i} = \frac {1}{\gamma (n) T} \sum_ {S \in \mathcal {S}: i \in S} U ^ {n} (S).
$$

Then, we have

$$
| \hat {r} _ {i} - \overline {{r}} _ {i} | = \left| \left(\frac {1}{T _ {i}} - \frac {1}{\gamma (n) T}\right) \sum_ {S \in \mathcal {S}: i \in S} U ^ {n} (S) \right| \leq \frac {u}{\gamma (n) T} | \gamma (n) T - T _ {i} |.
$$

Note that this inequality $|\hat{r}_i - \overline{r}_i| \leq \frac{u}{\gamma(n)T} |\gamma(n)T - T_i|$ still holds when $T_i = 0$ . Since $T_i \sim \text{binomial}(T, \gamma(n))$ , by the Hoeffding's inequality, there is

$$
P (| T _ {i} - \gamma (n) T | \geq \Delta) \leq 2 \exp (- \frac {2 \Delta^ {2}}{T}).
$$

Therefore, $|\hat{r}_i - \overline{r}_i| < \frac{u\Delta}{\gamma(n)T}$ provided that $|T_i - \gamma(n)T| < \Delta$ . Since $\overline{r}_i = \frac{1}{T}\sum_{S\in S}\beta(S,n)U^n(S)$ where $\beta(S,n) = \gamma(n)^{-1}$ if $i\in S$ and 0 otherwise, and $\mathbb{E}[\overline{r}_i] = r_i$ , using the Heoffding's inequality again yields

$$
P (| \overline {{r}} _ {i} - r _ {i} | \geq \sigma) \leq 2 \exp (- \frac {\gamma (n) ^ {2} T \sigma^ {2}}{2 u ^ {2}})
$$

Therefore,

$$
\begin{array}{l} P (| \hat {r} _ {i} - r _ {i} | \geq \epsilon) = P (| \hat {r} _ {i} - r _ {i} | \geq \epsilon \cap | T _ {i} - \gamma (n) T | <   \Delta) + P (| \hat {r} _ {i} - r _ {i} | \geq \epsilon \cap | T _ {i} - \gamma (n) T | \geq \Delta) \\ \leq P (| \hat {r} _ {i} - r _ {i} | \geq \epsilon | | T _ {i} - \gamma (n) T | <   \Delta) + 2 \exp (- \frac {2 \Delta^ {2}}{T}) \\ \leq P (| \bar {r} _ {i} - r _ {i} | \geq \epsilon - \frac {u \Delta}{\gamma (n) T} | | T _ {i} - \gamma (n) T | <   \Delta) + 2 \exp (- \frac {2 \Delta^ {2}}{T}) \\ \leq \frac {P (| \overline {{r}} _ {i} - r _ {i} | \geq \epsilon - \frac {u \Delta}{\gamma (n) T})}{1 - 2 \exp (- \frac {2 \Delta^ {2}}{T})} + 2 \exp (- \frac {2 \Delta^ {2}}{T}) \\ \leq \frac {2 \exp (- \frac {\gamma (n) ^ {2} T (\epsilon - \frac {u \Delta}{\gamma (n) T}) ^ {2}}{2 u ^ {2}})}{1 - 2 \exp (- \frac {2 \Delta^ {2}}{T})} + 2 \exp (- \frac {2 \Delta^ {2}}{T}) \\ \leq 3 \exp (- \frac {\gamma (n) ^ {2} T \left(\epsilon - \frac {u \Delta}{\gamma (n) T}\right) ^ {2}}{2 u ^ {2}}) + 2 \exp (- \frac {2 \Delta^ {2}}{T}) \\ \end{array}
$$

where $1 - 2\exp \left(-\frac{2\Delta^2}{T}\right)\geq \frac{2}{3}$ provided that $T$ is sufficiently large. The next step is to determine $\Delta$ by solving the equation $-\frac{\gamma(n)^2T\left(\epsilon -\frac{u\Delta}{\gamma(n)T}\right)^2}{2u^2} = -\frac{2\Delta^2}{T}$ , which yields $\Delta = \frac{\gamma(n)T\epsilon}{3u}$ . Note that this solution gives $\epsilon -\frac{u\Delta}{\gamma(n)T} = \frac{2\epsilon}{3} >0$ and $\frac{2\Delta^2}{T} = \frac{2\gamma(n)^2T\epsilon^2}{9u^2}$ . Besides, the inequality $1 - 2\exp \left(-\frac{2\Delta^2}{T}\right)\geq \frac{2}{3}$ leads to $T\geq \frac{9\log(6)u^2}{2\gamma(n)^2\epsilon^2}$ . Eventually, we have

$$
P (| \hat {r} _ {i} - r _ {i} | \geq \epsilon) \leq 5 \exp (- \frac {2 \gamma (n) ^ {2} T \epsilon^ {2}}{9 u ^ {2}}) \tag {15}
$$

provided that $T \geq \frac{9\log(6)u^2}{2\gamma(n)^2\epsilon^2}$ . Then,

$$
P (\| \hat {\mathbf {r}} - \mathbf {r} \| _ {2} \geq \epsilon) \leq P (\bigcup_ {i \in [ n ]} | \hat {r} _ {i} - r _ {i} | \geq \frac {\epsilon}{\sqrt {n}}) \leq 5 n \exp (- \frac {2 \gamma (n) ^ {2} T \epsilon^ {2}}{9 u ^ {2} n}). \tag {16}
$$

Solving $5n\exp \left(-\frac{2\gamma(n)^2T\epsilon^2}{9u^2n}\right)\leq \delta$ leads to $T\geq \frac{9u^2n}{2\gamma(n)^2\epsilon^2}\log \frac{5n}{\delta}$ , and thus GELS-R requires

$$
\max (\frac {9 \log (6) u ^ {2}}{2 \gamma (n) ^ {2} \epsilon^ {2}}, \frac {9 u ^ {2} n}{2 \gamma (n) ^ {2} \epsilon^ {2}} \log \frac {5 n}{\delta}) = O (\frac {\kappa (n) n}{\epsilon^ {2}} \log \frac {n}{\delta})
$$

utility evaluations to achieve an $(\epsilon,\delta)$ -approximation where $\kappa(n)=\gamma(n)^{-2}$ .

Next, we prove the convergence for GELS. For simplicity, we write $\phi = \phi(U^n)$ and $\mathbf{h} = \left(\sum_{s=1}^{n}\binom{n}{s-1}p_s^n\right)^{-1}\mathcal{R}(\overline{U}^{n+1})$ ; see Eq. (14). By Eq. (15), there is, for every $i \in [n+1]$ ,

$$
P (| \hat {h} _ {i} - h _ {i} | \geq \epsilon) \leq 5 \exp (- \frac {2 \widetilde {\gamma} (n) ^ {2} T \epsilon^ {2}}{9 u ^ {2}})
$$

provided that $T \geq \frac{9\log(6)u^2}{2\widetilde{\gamma}(n)^2\epsilon^2}$ where $\widetilde{\gamma}(n) = \frac{\sum_{s=1}^{n}\binom{n}{s-1}p_s^n}{\sum_{s=1}^{n}\binom{n+1}{s}p_s^n}$ . Therefore, for every $i \in [n]$ ,

$$
\begin{array}{l} P (| (\hat {h} _ {i} - \hat {h} _ {n + 1}) - (h _ {i} - h _ {n + 1}) | \geq \epsilon) \leq P (| \hat {h} _ {i} - h _ {i} | \geq \frac {\epsilon}{2} \cup | \hat {h} _ {n + 1} - h _ {n + 1} | \geq \frac {\epsilon}{2}) \\ \leq 1 0 \exp (- \frac {\widetilde {\gamma} (n) ^ {2} T \epsilon^ {2}}{1 8 u ^ {2}}). \\ \end{array}
$$

let $\chi(n) = \sum_{s=1}^{n}\binom{n}{s-1}p_s^n$ and $\eta(n) = \sum_{s=1}^{n}\binom{n+1}{s}p_s^n$ , and note that $\widetilde{\gamma}(n) = \frac{\chi(n)}{\eta(n)}$ . As argued in Remark 1, $\mathcal{R}_i(\overline{U}^{n+1}) - \mathcal{R}_{n+1}(\overline{U}^{n+1}) = \phi_i(U^n)$ for every $i \in [n]$ , and thus $\chi(n)(h_i - h_{n+1}) = \phi_i(U^n)$ for every $i \in [n]$ . So, for every $i \in [n]$ ,

$$
P (| \hat {\phi} _ {i} - \phi_ {i} | \geq \epsilon) = P (| (\hat {h} _ {i} - \hat {h} _ {n + 1}) - (h _ {i} - h _ {n + 1}) | \geq \frac {\epsilon}{\chi (n)}) \leq 1 0 \exp (- \frac {T \epsilon^ {2}}{1 8 u ^ {2} \eta (n) ^ {2}}).
$$

Therefore,

$$
P (\| \hat {\phi} - \phi \| _ {2} \geq \epsilon) \leq P (\bigcup_ {i \in [ n ]} | \hat {\phi} _ {i} - \phi_ {i} | \geq \frac {\epsilon}{\sqrt {n}}) \leq 1 0 n \exp (- \frac {T \epsilon^ {2}}{1 8 u ^ {2} n \eta (n) ^ {2}}).
$$

Solving $\delta \geq 10n\exp \left(-\frac{T\epsilon^2}{18u^2n\eta(n)^2}\right)$ yields $T \geq \frac{18u^2n\eta(n)^2}{\epsilon^2}\log \frac{10n}{\delta}$ . To conclude, GELS requires

$$
\max (\frac {9 \log (6) u ^ {2}}{2 \widetilde {\gamma} (n) ^ {2} \epsilon^ {2}}, \frac {1 8 u ^ {2} n \eta (n) ^ {2}}{\epsilon^ {2}} \log \frac {1 0 n}{\delta}) = O (\frac {\tau (n) n}{\epsilon^ {2}} \log \frac {n}{\delta})
$$

utility evaluations to achieve an $(\epsilon, \delta)$ -approximation where $\tau(n) = \eta(n)^2$ . Note that $\widetilde{\gamma}(n)^{-1} \leq \eta(n)$ as $\chi(n) = \sum_{s=1}^{n} \binom{n}{s-1} \binom{n-1}{s-1}^{-1} \binom{n-1}{s-1} p_s^n = \sum_{s=1}^{n} \frac{n}{n-s+1} \binom{n-1}{s-1} p_s^n \geq 1$ .

Last, we prove the convergence for GELS-Shapley. Reusing Eq. (14), since $p_s^n = \frac{(s - 1)!(n - s)!}{n!}$ for the Shapley value, it is $H_{n - 1}\cdot \mathbf{r} = \mathcal{R}(U^n)$ where $H_{n - 1} = \sum_{s = 1}^{n - 1}\binom{n - 1}{s - 1}(p_s^n +p_{s + 1}^n) = \sum_{s = 1}^{n - 1}\frac{1}{s}$ . By Eq. (16), there is

$$
P (H _ {n - 1} \cdot \| \hat {\mathbf {r}} - \mathbf {r} \| _ {2} \geq \epsilon) \leq 5 n \exp \left(- \frac {2 T \epsilon^ {2}}{9 u ^ {2} n \eta (n - 1) ^ {2}}\right)
$$

provided that $T \geq \frac{9\log(6)u^2}{2\gamma(n)^2\epsilon^2}$ . Note that $\sum_{s=1}^{n-1}\binom{n}{s}(p_s^n + p_{s+1}^n) = \sum_{s=1}^{n-1}\binom{n}{s}p_s^{n-1} = \eta (n-1)$ .

For convenience, let $\phi^{Sh}$ be the corresponding Shapley value. As discussed in Remark 2,

$$
\phi^ {S h} = H _ {n - 1} \cdot \mathbf {r} + \frac {U ^ {n} ([ n ]) - U ^ {n} (\emptyset) - \sum_ {s = 1} ^ {n} H _ {n - 1} \cdot r _ {i}}{n}.
$$

The procedure of casting $H_{n-1} \cdot \mathbf{r}$ into $\phi^{Sh}$ is called additive-efficient-normalization (Ruiz et al. 1998, Definition 11). Particularly, we write

$$
\hat {\phi} ^ {S h} = H _ {n - 1} \cdot \hat {\mathbf {r}} + \frac {U ^ {n} ([ n ]) - U ^ {n} (\emptyset) - \sum_ {s = 1} ^ {n} H _ {n - 1} \cdot \hat {r} _ {i}}{n}.
$$

As pointed out by Jethani et al. (2022, Appendix B), the additive-efficient-normalization is just an orthogonal projection, and therefore we have

$$
\| \hat {\phi} ^ {S h} - \phi^ {S h} \| _ {2} \leq \| H _ {n - 1} \cdot \hat {\mathbf {r}} - H _ {n - 1} \cdot \mathbf {r} \| _ {2},
$$

which suggest

$$
P (\| \hat {\phi} ^ {S h} - \phi^ {S h} \| _ {2} \geq \epsilon) \leq P (\| H _ {n - 1} \cdot \hat {\mathbf {r}} - H _ {n - 1} \cdot \mathbf {r} \| _ {2} \geq \epsilon) \leq 5 n \exp \left(- \frac {2 T \epsilon^ {2}}{9 u ^ {2} n \eta (n - 1) ^ {2}}\right).
$$

Solving $\delta \geq 5n\exp \left(-\frac{2T\epsilon^2}{9u^2n\eta(n - 1)^2}\right)$ yields $T\geq \frac{9u^2n\eta(n - 1)^2}{2\epsilon^2}\log \frac{5n}{\delta}$ . Since $\gamma (n)^{-1}\leq \eta (n - 1)$ , we eventually have that GELS-Shapley requires $O(\frac{n\tau(n)}{\epsilon^2}\log \frac{n}{\delta})$ utility evaluations to achieve an $(\epsilon ,\delta)$ -approximation where $\tau (n) = \eta (n)^2 >\eta (n - 1)^2$ . Note that i) for the Shapley value, $\eta (n) = \sum_{s = 1}^{n}\binom {n + 1}{s}\binom {n - 1}{s - 1}^{-1}\frac{1}{n} = \sum_{s = 1}^{n}\frac{n + 1}{s(n + 1 - s)} = \sum_{s = 1}^{n}\left(\frac{1}{s} +\frac{1}{n + 1 - s}\right) = 2H_n\in \log n,$ and thus $\eta (n - 1) <   \eta (n)$ ; and ii) $\gamma (n)^{-1}\leq \eta (n - 1)$ as $\gamma (n)\cdot \eta (n - 1) = H_{n - 1}\geq 1$ .

# D LEAST SQUARE VALUES

In this appendix, we demonstrate how we obtained proposition 2 at the very beginning without knowing proposition 1.

Definition 1 (Least Square Values (Ruiz et al. 1998, Definition 5)). Suppose a non-negative nonzero vector $\mathbf{m} \in \mathbb{R}^{n-1}$ is given, a least square value $\xi(U^n; \mathbf{m}) \in \mathbb{R}^n$ is defined to be the uniquely optimal solution to the problem

$$
\underset {\mathbf {v} \in \mathbb {R} ^ {n}} {\arg \min} \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} m _ {s} \left(U ^ {n} (S) - U ^ {n} (\emptyset) - \sum_ {i \in S} v _ {i}\right) ^ {2} \tag {17}
$$

$$
s. t. \sum_ {i = 1} ^ {n} v _ {i} = U ^ {n} ([ n ]) - U ^ {n} (\emptyset).
$$

Specifically, Ruiz et al. (1998, Theorem 8) developed a system of axioms that uniquely characterizes the family of least square values. Moreover, its relationship with the family of probabilistic values was also revealed.

Proposition 6 (Ruiz et al. 1998, Theorem 12). For each vector of weights $\mathbf{p}^n\in \mathbb{R}^n$ that satisfies $\sum_{s = 1}^{n}\binom{n - 1}{s - 1}p_s^n = 1$ , define $\mathbf{m}\in \mathbb{R}^{n - 1}$ by letting $m_{s} = p_{s}^{n} + p_{s + 1}^{n}$ , there exists some function $f:\mathcal{G}^n\to \mathbb{R}$ such that

$$
\xi_ {i} (U ^ {n}; \mathbf {m}) = \sum_ {S \subseteq [ n ] \setminus i} p _ {s} ^ {n} \cdot (U ^ {n} (S \cup i) - U ^ {n} (\emptyset)) + f (U ^ {n})
$$

for every $U^n \in \mathcal{G}^n$ and $i \in [n]$ .

As proposed by Ruiz et al. (1998, Definition 11), for each probabilistic value $\phi(U^n)$ , its additive-efficient-normalization $\overline{\phi}$ is defined to be

$$
\overline {{\phi}} (U ^ {n}) = \phi (U ^ {n}) + \frac {1}{n} (U ^ {n} ([ n ]) - U ^ {n} (\emptyset) - \sum_ {i \in [ n ]} \phi_ {i} (U ^ {n})).
$$

Using Proposition 6, one can verify that $\xi(U^n; \mathbf{m}) = \overline{\phi}(U^n)$ and $f(U^n) = \frac{1}{n}(U^n([n]) - U^n(\emptyset) - \sum_{i \in [n]} \phi_i(U^n))$ . Particularly, we notice that the constraint of the problem (17) and the term $-U^n(\emptyset)$ in the objective can be removed if it is the ranking that matters.

Proposition 7. Suppose a utility function $U^n \in \mathcal{G}^n$ and a non-negative non-zero vector $\mathbf{m} \in \mathbb{R}^{n-1}$ are given, the uniquely optimal solution $\mathbf{u}^*$ to the problem

$$
\underset {\mathbf {u} \in \mathbb {R} ^ {n}} {\arg \min} \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} m _ {s} \left(U ^ {n} (S) - \sum_ {i \in S} u _ {i}\right) ^ {2} \tag {18}
$$

satisfies that there exists some $c \in \mathbb{R}$ such that

$$
\mathbf {u} ^ {*} = \xi (U ^ {n}; \mathbf {m}) + c \mathbf {1} _ {n}.
$$

Proof. Since the optimization problem (17) is convex, it can be solved using the KKT condition. Specifically, by introducing a dual variable $\lambda \in \mathbb{R}$ , there is

$$
\mathbf {v} ^ {\top} \mathbf {A} \mathbf {v} - 2 \mathbf {b} ^ {\top} \mathbf {v} + \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} m _ {s} (U ^ {n} (S) - U ^ {n} (\emptyset)) ^ {2} - 2 \lambda (U ^ {n} ([ n ]) - U ^ {n} (\emptyset) - \sum_ {i = 1} ^ {n} \phi_ {i})
$$

where $A_{ij} = \left\{ \begin{array}{ll} \sum_{s=1}^{n-1} \binom{n-1}{s-1} m_s, & i = j \\ \sum_{s=2}^{n-1} \binom{n-2}{s-2} m_s, & i \neq j \end{array} \right.$ , and $b_i = \sum_{S \subsetneq [n] : i \in S} m_s (U^n(S) - U^n(\emptyset)) \quad \forall i \in [n]$ .

Letting its derivative equal 0 yields

$$
\mathbf {A} \mathbf {v} - \mathbf {b} + \lambda \mathbf {1} _ {n} = 0 \text {   and   } \mathbf {1} _ {n} ^ {\top} \mathbf {v} = U ^ {n} ([ n ]) - U ^ {n} (\emptyset),
$$

which leads to the uniquely optimal solution

$$
\mathbf {v} ^ {*} = \mathbf {A} ^ {- 1} \left(\mathbf {b} - \frac {\mathbf {1} _ {n} ^ {\top} \mathbf {A} ^ {- 1} \mathbf {b} - U ^ {n} ([ n ]) + U ^ {n} (\emptyset)}{\mathbf {1} _ {n} ^ {\top} \mathbf {A} ^ {- 1} \mathbf {1} _ {n}} \mathbf {1} _ {n}\right). \tag {19}
$$

Specifically,

$$
\sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s - 1} m _ {s} - \sum_ {s = 2} ^ {n - 1} \binom {n - 2} {s - 2} m _ {s}
$$

$$
= m _ {1} + \sum_ {s = 2} ^ {n - 1} \left(\binom{n - 2}{s - 1} + \binom{n - 2}{s - 2}\right) m _ {s} - \sum_ {s = 2} ^ {n - 1} \binom{n - 2}{s - 2} m _ {s} = \sum_ {s = 1} ^ {n - 1} \binom{n - 2}{s - 1} m _ {s} > 0,
$$

by lemma 1, the existence of $\mathbf{A}^{-1}$ is guaranteed. On the other hand, letting the derivative of the objective in the problem (18) equal 0 gives

$$
\mathbf {u} ^ {*} = \mathbf {A} ^ {- 1} (\mathbf {b} + t \mathbf {1} _ {n}) \text {   where   } t = U ^ {n} (\emptyset) \cdot A _ {1 1}.
$$

Therefore, we eventually have, for some $c \in \mathbb{R}$ ,

$$
\mathbf {u} ^ {*} = \mathbf {v} ^ {*} + c \mathbf {1} _ {n} = \xi (U ^ {n}; \mathbf {m}) + c \mathbf {1}.
$$

for some $c\in \mathbb{R}$

![](images/da11cb6e14e0f5de02b71a7c7d4ef33ada3016f6c92cada003f54befe8c3c631.jpg)

Remark 5. To conclude, Proposition 2 is derived from Propositions 6 and 7.

# E GENERALIZED ARM

As proposed by Kolpaczki et al. (2023), the Shapley value can be rewritten as, for every $U^n \in \mathcal{G}^n$ and $i \in [n]$ ,

$$
\phi_ {i} ^ {S h} (U ^ {n}) = \mathbb {E} _ {S \sim P _ {A R M} ^ {+} | i \in S} [ U ^ {n} (S) ] - \mathbb {E} _ {S \sim P _ {A R M} ^ {-} | i \not \in S} [ U ^ {n} (S) ]
$$

where $P_{ARM}^{+}(S) = \frac{1}{s \cdot H}\binom{n}{s}^{-1}$ for every $S \in S^{+} = \{\emptyset \subsetneq S \subseteq [n]\}$ and $P_{ARM}^{-}(S) = \frac{1}{(n-s) \cdot H}\binom{n}{s}^{-1}$ for every $S \in S^{-} = \{S \subsetneq [n]\}$ ; $H = \sum_{s=1}^{n} \frac{1}{s}$ .

The estimator based on this formula is called the approximation-without-requiring-marginal estimator. We found that this methodology can be easily adapted for every other probabilistic value, which is summarized in the below.

Proposition 8. For every $U^n \in \mathcal{G}^n$ and $i \in [n]$ ,

$$
\phi_ {i} (U ^ {n}) = \mathbb {E} _ {S \sim P _ {A R M} ^ {+} | i \in S} [ U ^ {n} (S) ] - \mathbb {E} _ {S \sim P _ {A R M} ^ {-} | i \not \in S} [ U ^ {n} (S) ]
$$

where $P_{ARM}^{+}(S)\propto p_{s}$ for every $S\in \mathcal{S}^{+}$ and $P_{ARM}^{-}(S)\propto p_{s + 1}$ for every $S\in \mathcal{S}^{-}$ .

Proof. Precisely, $p_{s}^{+} = \alpha \cdot p_{s}$ and $p_{s}^{-} = \beta \cdot p_{s+1}$ such that $\sum_{s=1}^{n} \binom{n}{s} p_{s}^{+} = 1$ and $\sum_{s=0}^{n-1} \binom{n}{s} p_{s+1}^{-} = 1$ . Let $S^{+}$ and $S^{-}$ be the corresponding random samples from $S^{+}$ and $S^{-}$ , respectively. Fix an $i \in [n]$ , observe that,

$$
\begin{array}{l} P (i \in S ^ {+}) = \sum_ {s = 1} ^ {n} P (i \in S ^ {+} \mid | S ^ {+} | = s) \cdot P (| S ^ {+} | = s) = \sum_ {s = 1} ^ {n} {\binom {n - 1} {s - 1}} {\binom {n} {s}} ^ {- 1} \cdot {\binom {n} {s}} p _ {s} ^ {+} \\ = \alpha \sum_ {s = 1} ^ {n} {\binom {n - 1} {s - 1}} p _ {s} = \alpha . \\ \end{array}
$$

Therefore,

$$
P (S ^ {+} \mid i \in S ^ {+}) = \frac {P (S ^ {+} , i \in S ^ {+})}{P (i \in S ^ {+})} = \frac {\alpha \cdot p _ {s}}{\alpha} = p _ {s}.
$$

Similarly,

$$
\begin{array}{l} P (i \not \in S ^ {-}) = \sum_ {s = 0} ^ {N - 1} P (i \not \in S ^ {-} | | S ^ {-} | = s) \cdot P (| S ^ {-} | = s) = \sum_ {s = 0} ^ {n - 1} \binom {n - 1} {s} \binom {n} {s} ^ {- 1} \cdot \binom {n} {s} p _ {s} ^ {-} \\ = \beta \sum_ {s = 0} ^ {n - 1} \binom {n - 1} {s} p _ {s + 1} = \beta , \\ \end{array}
$$

which leads to

$$
P (S ^ {-} \mid i \not \in S ^ {-}) = \frac {P (S ^ {-} , i \not \in S ^ {-})}{P (i \not \in S ^ {-})} = \frac {\beta \cdot p _ {s + 1}}{\beta} = p _ {s + 1}.
$$

![](images/ca18de3e4aa1cd7b1e8827a5be62518525af0f37ace22886a79ca87637e8e23b.jpg)

# F PRACTICAL PROBABILISTIC VALUES

As provided by Dubey et al. (1981), each semi-value, which is a subfamily of probabilistic values, can be expressed as, for every $U^n \in \mathcal{G}^n$ and $i \in [n]$ ,

$$
\phi_ {i} (U ^ {n}) = \phi_ {i} (U ^ {n}; \mu) = \sum_ {S \subseteq [ n ] \setminus i} \left(\int_ {0} ^ {1} t ^ {s} (1 - t) ^ {n - 1 - s} \mathrm{d} \mu (t)\right) (U ^ {n} (S \cup i) - U ^ {n} (S))
$$

where $\mu$ is any probability measure on the closed interval [0, 1]. In other words, in view of Eq. (1), there is $p_s^n = \int_0^1 t^{s-1}(1 - t)^{n-s} \mathrm{d}\mu(t)$ .

To the best of our knowledge, practical semi-values, i.e., those ever studied in the previous references, include the Banzhaf value (Wang and Jia 2023) and the Beta Shapley values with $\alpha, \beta \geq 1$ (Kwon and Zou 2022a; Kwon and Zou 2022b). Note that $\mathrm{Beta}(1,1)$ is exactly the Shapley value. For the Banzhaf value, the corresponding $\mu$ is the Dirac delta distribution $\delta_{0.5}$ , which leads to $p_s^n = 2^{-(n-1)}$ . For $\mathrm{Beta}(\alpha, \beta)$ , the corresponding probability density function for $\mu$ is $\propto t^{\beta-1}(1-t)^{\alpha-1}$ , and thus $p_s^n = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \cdot \frac{\Gamma(\beta+s-1)\Gamma(\alpha+n-s)}{\Gamma(\alpha+\beta+n-1)}$ . Specifically, it is $p_s^n = \frac{(s-1)!(n-s)!}{n!}$ for $\mathrm{Beta}(1,1)$ , i.e., the Shapley value.

# G PRACTICAL ASPECT OF FASTER ESTIMATORS FOR PROBABILISTIC VALUES

As far as we know, in feature attribution, FastSHAP is the only framework for training semi-value-based explainers (Jethani et al. 2022), which is based on the least square regression Eq. (3) specific

to the Shapley value. Recently, Kwon and Zou (2022b) showed that other candidates of the Beta Shapley values tend to perform better than the Shapley value in feature attribution. Therefore, one may ask how to cast other probabilistic values into optimization, which is answered by Propositions 1 and 3. Though AME proposed by Lin et al. (2022) provides an alternative way to achieve this goal, it is restricted to a subfamily of semi-values, which does not include, e.g., the Shapley value and Beta(4, 1) used in our experiments. Besides, as shown in our experiments of comparing convergences, on Beta(2, 2), the induced estimator of AME does not rival our GELS-R and GELS estimators, which are derived from solving the least square regressions (5) and (6).

Recall that the optimization problem used by FastSHAP is

$$
\mathbb {E} _ {z \in \mathcal {Z}} \left[ \sum_ {\emptyset \subsetneq S \subsetneq [ d ]} \frac {d - 1}{\binom {d} {s} s (d - s)} \left(U _ {z} (S) - U _ {z} (\emptyset) - \mathbf {1} _ {S} ^ {\top} \phi (z; \pmb {\theta})\right) ^ {2} \right]
$$

where d is the number of features (a counterpart of n), $\phi(z;\boldsymbol{\theta})\in\mathbb{R}^{d}$ is a trainable explainer parameterized by $\theta$ , $U_{z}$ is a utility function based on the data point $z=(\mathbf{x},y)$ where x and y represent the features and label, respectively, and $\mathbf{1}_{S}\in\mathbb{R}^{d}$ is defined by $\mathbf{1}_{S}(i)=1$ if $i\in S$ and 0 otherwise.

Assume $\{S\}_{\emptyset \subsetneq S\subsetneq [d]}$ is ordered so that each utility function $U_{z}$ can be treated as a vector $\mathbf{U}_z\in \mathbb{R}^{2^d -2}$ . Besides, let $\mathbf{W}\in \mathbb{R}^{(2^d -2)\times (2^d -2)}$ be a diagonal matrix such that $W(S,S) = \frac{d - 1}{\binom{d}{s}s(d - s)}$ ,

and define $\mathbf{X} \in \mathbb{R}^{(2^d - 2) \times d}$ by letting the $S$ -th row of $\mathbf{X}$ is $\mathbf{1}_S$ . Very recently, Zhang et al. (2023a) discovered that

$$
\mathbb {E} _ {z \in \mathcal {Z}} \left[ \| \phi (z; \boldsymbol {\theta}) - (\mathbf {X} ^ {\top} \mathbf {W X}) ^ {- 1} \mathbf {X} ^ {\top} \mathbf {W} \left(\mathbf {U} _ {z} - U _ {z} (\emptyset) \mathbf {1} _ {2 ^ {d} - 2}\right) \| _ {\mathbf {X} ^ {\top} \mathbf {W X}} ^ {2} \right]
$$

$$
= \mathbb {E} _ {z \in \mathcal {Z}} \left[ \| \mathbf {U} _ {z} - U _ {z} (\emptyset) \mathbf {1} _ {2 ^ {d} - 2} - \mathbf {X} \phi (z; \boldsymbol {\theta}) \| _ {\mathbf {W}} ^ {2} + C _ {z} \right]
$$

$$
= \mathbb {E} _ {z \in \mathcal {Z}} \left[ \sum_ {\emptyset \subsetneq S \subsetneq [ d ]} \frac {d - 1}{\binom {d} {s} s (d - s)} \left(U _ {z} (S) - U _ {z} (\emptyset) - \mathbf {1} _ {S} ^ {\top} \phi (z; \boldsymbol {\theta})\right) ^ {2} + C _ {z} \right]
$$

where $C_z = \| (\mathbf{X}^\top \mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top \mathbf{W}(\mathbf{U}_z - U_z(\emptyset)\mathbf{1}_{2^d -2})\|_{\mathbf{X}^\top \mathbf{W}\mathbf{X}}^2 -\| U_z(\emptyset)\mathbf{1}_{2^d -2} - \mathbf{U}_z\|_{\mathbf{W}}^2.$

The convention is $\|z\|_{A}^{2}:=z^{\top}Az$ . This relationship reveals that FastSHAP is just to train explainers towards a (potentially) biased target $(\mathbf{X}^{\top}\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{W}\left(\mathbf{U}_{z}-U_{z}(\emptyset)\mathbf{1}_{2^{d}-2}\right)$ under the metric induced by $X^{\top}WX$ ; see Proposition 9. Therefore, they instead proposed to train explainers based on

$$
\mathbb {E} _ {z \sim \mathcal {Z}} [ \| \phi (z; \theta) - \phi \| ^ {2} ] \tag {20}
$$

where $\phi$ is any unbiased estimator for the Shapley value. They demonstrated in the experiments that this framework of training explainers is not just simple but effective. Note that the target $(\mathbf{X}^{\top}\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{W}\left(\mathbf{U}_{z}-U_{z}(\emptyset)\mathbf{1}_{2^{d}-2}\right)$ is expected to be biased because compared with Eq. (3), the efficiency constraint, which is supposed to be $\sum_{i=1}^{d}\phi(z;\pmb{\theta}) = U_z([d]) - U_z(\emptyset)$ , has been removed. Nevertheless, to overcome this issue, the authors of FastSHAP added an additive-efficient-normalization layer on top during training and inference (Jethani et al. 2022, Table 3), which is $\phi(z;\pmb{\theta}) \leftarrow \phi(z;\pmb{\theta}) + \frac{1}{d}(U_z([d]) - U_z(\emptyset) - \sum_{i=1}^{d}\phi_i(z;\pmb{\theta}))\mathbf{1}_d$ .

Remark 6. Note that the framework (20) can be used for training explainers towards any probabilistic value. Intuitively, a faster unbiased estimator substituted in the framework (20) would lead to better training of explainers. All in all, faster estimators could possibly benefit the research line of training probabilistic-value-based explainers in feature attribution.

Proposition 9. Let $\hat{\phi}^{Sh} = (\mathbf{X}^{\top}\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{W}\left(\mathbf{U}_z - U_z(\emptyset)\mathbf{1}_{2^d -2}\right)$ and $\phi^{Sh}$ be the Shapley value using the utility function $U_{z}$ . Then, there is

$$
\boldsymbol {\phi} ^ {S h} = \hat {\boldsymbol {\phi}} ^ {S h} + \rho \mathbf {1} _ {d}
$$

$$
w h e r e \rho = - \frac {\mathbf {1} _ {d} ^ {\top} (\mathbf {X} ^ {\top} \mathbf {W X}) ^ {- 1} \mathbf {X} ^ {\top} \mathbf {W} (\mathbf {U} _ {z} - U _ {z} (\emptyset) \mathbf {1} _ {2 ^ {d} - 2}) - U _ {z} ([ d ]) + U _ {z} (\emptyset)}{\mathbf {1} _ {d} ^ {\top} (\mathbf {X} ^ {\top} \mathbf {W X}) ^ {- 1} \mathbf {1} _ {d}}.
$$

In other words, $(\mathbf{X}^{\top}\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{W}\left(\mathbf{U}_{z}-U_{z}(\emptyset)\mathbf{1}_{2^{d}-2}\right)$ is potentially biased as $\rho$ is not necessarily zero.

Proof. In view of the optimization problem (17), let $m_{s} \leftarrow \frac{d-1}{\binom{d}{s}s(d-s)}$ , $n \leftarrow d$ and $U^{n} \leftarrow U_{z}$ . By (Charnes et al. 1988, Theorem 4), the induced uniquely optimal solution $v^{*}$ is exactly $\phi^{Sh}$ . According to Eq. (19), we have

$$
\mathbf {v} ^ {*} = \mathbf {A} ^ {- 1} \left(\mathbf {b} - \frac {\mathbf {1} _ {d} ^ {\top} \mathbf {A} ^ {- 1} \mathbf {b} - U _ {z} ([ d ]) + U _ {z} (\emptyset)}{\mathbf {1} _ {d} ^ {\top} \mathbf {A} ^ {- 1} \mathbf {1} _ {d}} \mathbf {1} _ {d}\right)
$$

where $\mathbf{A} = \mathbf{X}^{\top}\mathbf{W}\mathbf{X}$ and $\mathbf{b} = \mathbf{X}^{\top}\mathbf{W}(\mathbf{U}_z - U_z(\emptyset)\mathbf{1}_{2^d -2})$ ,

from which we can deduce

$$
\phi^ {S h} = \mathbf {v} ^ {*} = \mathbf {A} ^ {- 1} \mathbf {b} + \rho \mathbf {1} _ {d} = \hat {\phi} ^ {S h} + \rho \mathbf {1} _ {d}
$$

where $\rho = -\frac{\mathbf{1}_d^\top\mathbf{A}^{-1}\mathbf{b} - U_z([d]) + U_z(\emptyset)}{\mathbf{1}_d^\top\mathbf{A}^{-1}\mathbf{1}_d}$ .

![](images/ada1c6e7536fbd6c6f1cab8bd3fae8b5bce0850a12efbc12655246a74fbd50bb.jpg)

# H INTERPRETATION OF $\kappa(N)$ AND $\tau(N)$ IN PROPOSITION 4

Recall that $\kappa(n) = \gamma(n)^{-2}$ where $\gamma(n) = \frac{\sum_{s=1}^{n-1}\binom{n-1}{s-1}(p_s^n + p_{s+1}^n)}{\sum_{s=1}^{n-1}\binom{n}{s}(p_s^n + p_{s+1}^n)}$ . Our interpretation of $\kappa(n)$ is based on $\gamma(n)$ . In Algorithm 1, for each non-empty proper subset $S$ , its probability of being sampled is

$$
P (S) = \frac {p _ {s} ^ {n} + p _ {s + 1} ^ {n}}{\sum_ {s = 1} ^ {n - 1} {\binom {n} {s}} (p _ {s} ^ {n} + p _ {s + 1} ^ {n})}.
$$

Since $\gamma(n) = \sum_{S \subsetneq [n]: i \in S} P(S)$ , $\gamma(n)$ is the probability that the $i$ -th data point appears in a random sample $S$ . Thanks to symmetry, the choice of $i$ is immaterial. Thus,

$$
n \gamma (n) = \sum_ {i = 1} ^ {n} \sum_ {S \subsetneq [ n ]: i \in S} P (S) = \sum_ {S \subsetneq [ n ]} \sum_ {i = 1} ^ {n} \mathbb {1} _ {i \in S} P (S) = \sum_ {\emptyset \subsetneq S \subsetneq [ n ]} s P (S) = \mathbb {E} _ {S} [ s ],
$$

whence follows

$$
\gamma (n) = \frac {\mathbb {E} _ {S} [ s ]}{n} \geq \frac {1}{n}.
$$

The lower bound is achieved when $p_1^n = 1$ and $p_i^n = 0$ for all $i \geq 2$ , corresponding to the semivalue with $\mu = \delta_0$ , i.e., leave everything else out.

Looking into Algorithm 1, for each sampled utility evaluation $U^n(S)$ , it is used to update the estimates of $s$ data. Therefore, $\gamma(n)$ is just the average rate of reusing utility evaluations, and $\kappa(n)$ is the inverse square of this average rate. The higher $\gamma(n)$ is, the more efficient GELS-R is. On the flip side, reusing utility evaluations also creates correlation among the estimates. We conclude that for the benefit to outweigh the cost, $\gamma(n)$ needs to be on the order of $\omega(\frac{1}{\sqrt{n}})$ , which is easily satisfied as long as the sequence $\{p_s^n\}$ is not exclusively concentrated around small $s$ . All practical probabilistic values used in the literature, including the ones in our experiments, meet this condition.

Recall that $\tau(n) = \left(\sum_{s=1}^{n}\binom{n+1}{s}p_s^n\right)^2$ and we write $\zeta(n) = \tau(n)^{-\frac{1}{2}}$ . Observe that, for every $i \in [n]$ ,

$$
2 \zeta (n) = \lambda \cdot \frac {1}{\sum_ {s = 1} ^ {n} \binom {n} {s} p _ {s} ^ {n}} + (1 - \lambda) \cdot \frac {1}{\sum_ {s = 1} ^ {n} \binom {n} {s - 1} p _ {s} ^ {n}}
$$

where $\lambda = \frac{\sum_{s=1}^{n}\binom{n}{s}p_s^n}{\sum_{s=1}^{n}\binom{n+1}{s}p_s^n}$ is the probability of subsets sampled by Algorithm 2 not containing the introduced null data point (equivalently, the $(n + 1)$ -th data point). Then, for any $i \in [n]$ ,

$$
\frac {1}{\sum_ {s = 1} ^ {n} \binom {n} {s} p _ {s} ^ {n}} = \frac {\sum_ {s = 1} ^ {n} \binom {n - 1} {s - 1} p _ {s} ^ {n}}{\sum_ {s = 1} ^ {n} \binom {n} {s} p _ {s} ^ {n}} = \sum_ {\emptyset \subsetneq S \subseteq [ n ]: i \in S} P _ {\not \ni n + 1} (S)
$$

where $P_{\not\ni n + 1}(S)$ is the probability of the set $S$ sampled by Algorithm 2 conditioned on that the samples do not contain the $(n + 1)$ -th data point. Therefore,

$$
\frac {n}{\sum_ {s = 1} ^ {n} \binom {n} {s} p _ {s} ^ {n}} = \sum_ {i = 1} ^ {n} \sum_ {\emptyset \subsetneq S \subseteq [ n ]} \mathbb {1} _ {i \in S} P _ {\not \ni n + 1} (S) = \sum_ {\emptyset \subsetneq S \subseteq [ n ]} s P _ {\not \ni n + 1} (S) = \mathbb {E} _ {S | S \not \ni n + 1} [ s ].
$$

On the other hand, for every $i \in [n]$ ,

$$
\frac {1}{\sum_ {s = 1} ^ {n} \binom {n} {s - 1} p _ {s} ^ {n}} = \frac {\sum_ {s = 1} ^ {n} \binom {n - 1} {s - 1} p _ {s} ^ {n}}{\sum_ {s = 1} ^ {n} \binom {n} {s - 1} p _ {s} ^ {n}} = \sum_ {\emptyset \subsetneq S \subsetneq [ n + 1 ]: i \not \in S, n + 1 \in S} P _ {\ni n + 1} (S),
$$

which leads to

$$
\begin{array}{l} \frac {n}{\sum_ {s = 1} ^ {n} \binom {n} {s - 1} p _ {s} ^ {n}} = \sum_ {i = 1} ^ {n} \sum_ {\emptyset \subsetneq S \subsetneq [ n + 1 ]: n + 1 \in S} \mathbb {1} _ {i \not \in S} P _ {\ni n + 1} (S) \\ = \sum_ {\emptyset \subsetneq S \subsetneq [ n + 1 ]: n + 1 \in S} (n + 1 - s) P _ {\ni n + 1} (S) = \mathbb {E} _ {S | S \ni n + 1} [ n + 1 - s ]. \\ \end{array}
$$

Eventually, we have

$$
\begin{array}{l} \zeta (n) = \frac {1}{2 n} \left(\lambda \cdot \mathbb {E} _ {S | S \ni n + 1} [ s ] + (1 - \lambda) \cdot \mathbb {E} _ {S | S \ni n + 1} [ n + 1 - s ]\right) \\ = \frac {1}{2 n} \mathbb {E} _ {S} [ s \mathbb {1} _ {n + 1 \not \in S} + (n + 1 - s) \mathbb {1} _ {n + 1 \in S} ]. \\ \end{array}
$$

Looking into Algorithm 2, i) if the sampled subset S does not contain the null data, there are s of $\{\hat{r}_{i}-\hat{r}_{n+1}\}_{1\leq i\leq n}$ receiving updates from $U^{n}(S)$ ; ii) for the other way, there are $n+1-s$ of them receiving updates from $U^{n}(S)$ . To conclude, $\frac{\tau(n)}{4}$ is the square inverse of the average rate of reusing utility evaluations while running Algorithm 2.

# I ASYMPTOTIC ANALYSIS

This appendix is mainly to analyze the asymptotic behavior (as $n \to \infty$ ) of $\kappa(n)$ and $\tau(n)$ that appear in Proposition 4. For probabilistic values, generally, there is no restriction on how $\{\mathbf{p}^n \in \mathbb{R}^n\}_{n \geq 1}$ are organized. Therefore, to analyze the convergence of our proposed estimators, we focus on semi-values instead. According to Dubey et al. (1981), each semi-value corresponds to a probability measure $\mu$ on the interval [0, 1] such that

$$
p _ {s} ^ {n} = \int_ {0} ^ {1} t ^ {s - 1} (1 - t) ^ {n - s} \mathrm{d} \mu (t) \text {   for   every   } 1 \leq s \leq n. \tag {21}
$$

Lemma 2. Suppose $n > 1$ , and for each probability measure $\mu$ on the closed interval [0, 1], define

$$
m _ {k} ^ {\mu} = \int_ {0} ^ {1} t ^ {k} \mathrm{d} \mu (t) a n d w _ {k} ^ {\mu} = \int_ {0} ^ {1} (1 - t) ^ {k} \mathrm{d} \mu (t) f o r e v e r y k \geq 0,
$$

$$
M _ {k} ^ {\mu} = \sum_ {j = 0} ^ {k} m _ {k} ^ {\mu} a n d W _ {k} ^ {\mu} = \sum_ {j = 0} ^ {k} w _ {k} ^ {\mu} f o r e v e r y k \geq 0.
$$

Then, there is

$$
\kappa (n) ^ {\frac {1}{2}} = \frac {M _ {n - 2} ^ {\mu} + W _ {n - 2} ^ {\mu}}{M _ {n - 2} ^ {\mu}} a n d \tau (n) ^ {\frac {1}{2}} = M _ {n - 1} ^ {\mu} + W _ {n - 1} ^ {\mu}.
$$

Proof. Recall that in Proposition 4 $\kappa(n)^{\frac{1}{2}} = \frac{\sum_{s=1}^{n-1}\binom{n}{s}(p_s^n + p_{s+1}^n)}{\sum_{s=1}^{n-1}\binom{n-1}{s-1}(p_s^n + p_{s+1}^n)}$ . By Eq. (21), there is $p_s^n + p_{s+1}^n = p_s^{n-1}$ for every $1 \leq s \leq n-1$ . Notice that

$$
\sum_ {s = 1} ^ {n - 1} \binom {n} {s} p _ {s} ^ {n - 1} = \sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s} p _ {s} ^ {n - 1} + \sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s - 1} p _ {s} ^ {n - 1}.
$$

By Eq. (21),

$$
\begin{array}{l} \sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s} p _ {s} ^ {n - 1} = \int_ {0} ^ {1} \sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s} t ^ {s - 1} (1 - t) ^ {n - 1 - s} \mathrm{d} \mu (t) \\ = \int_ {0} ^ {1} \frac {1}{t} \sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s} t ^ {s} (1 - t) ^ {n - 1 - s} \mathrm{d} \mu (t) \\ = \int_ {0} ^ {1} \frac {1 - (1 - t) ^ {n - 1}}{t} \mathrm{d} \mu (t) = \int_ {0} ^ {1} \frac {(1 - (1 - t)) (\sum_ {j = 0} ^ {n - 2} (1 - t) ^ {j})}{t} \mathrm{d} \mu (t) \\ = W _ {n - 2} ^ {\mu} \\ \end{array}
$$

Note $\sum_{s=1}^{n-1}\binom{n-1}{s}t^{s-1}(1-t)^{n-1-s} = \sum_{j=0}^{n-2}(1-t)^j$ still holds for $t = 0$ . Similarly, one can get

$$
\sum_ {s = 1} ^ {n - 1} \binom {n - 1} {s - 1} p _ {s} ^ {n - 1} = M _ {n - 2} ^ {\mu}.
$$

Recall that in Proposition 4 $\tau(n)^{\frac{1}{2}} = \sum_{s=1}^{n}\binom{n+1}{s}p_s^n$ , and thus one can get $\tau(n)^{\frac{1}{2}} = M_{n-1}^{\mu} + W_{n-1}^{\mu}$ in a similar fashion.

Using the monotone convergence theorem, we have

$$
\lim _ {n \rightarrow \infty} M _ {n} ^ {\mu} = \lim _ {n \rightarrow \infty} \int_ {0} ^ {1} \frac {1 - t ^ {n + 1}}{1 - t} \mathrm{d} \mu (t) = \int_ {0} ^ {1} \lim _ {n \rightarrow \infty} \frac {1 - t ^ {n + 1}}{1 - t} \mathrm{d} \mu (t) = \int_ {0} ^ {1} \frac {1}{1 - t} \mathrm{d} \mu (t),
$$

$$
\lim _ {n \to \infty} W _ {n} ^ {\mu} = \lim _ {n \to \infty} \int_ {0} ^ {1} \frac {1 - (1 - t) ^ {n + 1}}{t} \mathrm{d} \mu (t) = \int_ {0} ^ {1} \frac {1}{t} \mathrm{d} \mu (t). \tag {22}
$$

Note that the extreme cases, e.g., $\lim_{n\to \infty}M_n^{\delta_1} = \int_0^1\frac{1}{1 - t}\mathrm{d}\delta_1(t)$ is trivially true where the convention is $\frac{1}{0} = \infty$ . Therefore, $\tau (n)\in \Theta (1)$ if and only if the two integrals in the above are finite. In particular, it holds if the probability density function $p$ of $\mu$ satisfies $\lim_{t\to 0}\frac{p(t)}{t^a} = \lim_{t\to 1}\frac{p(t)}{(1 - t)^b} = 0$ for some $a,b > 0$ ; see Proposition 10. Interestingly, we have

$$
\kappa (n) ^ {\frac {1}{2}} = 1 + \frac {W _ {n - 2} ^ {\mu}}{M _ {n - 2} ^ {\mu}} \leq 1 + W _ {n - 2} ^ {\mu}.
$$

Thus, $\kappa(n)=\Theta(1)$ if the integral Eq. (22) is finite, meaning that $\mu$ cannot put “large” mass around t=0. However, this is not necessary as any symmetric probability measure $\mu$ , e.g., the uniform distribution that leads to the Shapley value, will have $\kappa(n)=4$ .

Remark 7. For extreme cases $\mu = \delta_{0}$ (leave everything else out) and $\mu = \delta_{1}$ (leave one out), it is clear that $\kappa(n) \in \Theta(1)$ for the latter but it becomes $\Theta(n^{2})$ for the former, in which case our GELS-R (Algorithm 1) in fact only samples subsets of size 1, and therefore each utility evaluation is used to update the estimate of one data point. By contrast, the (weighted) sampling lift estimator always spends two utility evaluations to update the estimate of one data point.

Corollary 1. For the Banzhaf value, $\kappa(n), \tau(n) \in \Theta(1)$ . In other words, GELS, GELS-R and GELS-Shapley require $O\left(\frac{n}{\epsilon^2} \log \frac{n}{\delta}\right)$ utility evaluations to achieve an $(\epsilon, \delta)$ -approximation for the Banzhaf value.

Proof. For the Banzhaf value, $\mu = \delta_{0.5}$ (Dirac delta distribution), and thus $\lim_{n\to \infty}M_n^{\delta_{0.5}} = \lim_{n\to \infty}W_n^{\delta_{0.5}} = 2$ .

We now provide some easily verifiable conditions that determine the growth of $\kappa(n)$ and $\tau(n)$ . Recall that as long as $\kappa(n), \tau(n) = o(n)$ , our estimators are more efficient than the sampling lift estimator.

Proposition 10. Assume the probability measure $\mu$ admits a density function $p$ such that $\mu(S) = \int_{S} p(t) \, \mathrm{d}t$ for every Borel-measurable subset $S \subseteq [0,1]$ , then,

1. $\kappa(n), \tau(n) \in \Theta(1)$ if there exist $a, b > 0$ such that $\lim_{t \to 0} \frac{p(t)}{t^a} = \lim_{t \to 1} \frac{p(t)}{(1-t)^b} = 0$ .   
2. $\kappa(n), \tau(n) \in O(\log(n)^2)$ if $\limsup_{t \to 0} p(t) < \infty$ and $\limsup_{t \to 1} p(t) < \infty$ . Examples include if $p$ is bounded (the so-called continuous semivalves used by Dubey et al. (1981)).

Proof. We first show that there exist counterexamples if these conditions are violated. For the Shapley value with $\mu = U$ (the uniform measure), $m_{k}^{U} = w_{k}^{U} = \frac{1}{k+1}$ , and thus $\tau(n) = \Theta(\log(n)^{2})$ . For the second, looking into Lemma 3, consider $\alpha = \beta < 1$ , $\tau(n) \in \Theta(n^{2-2\alpha})$ .

Suppose there exist $a, b > 0$ such that $\lim_{t \to 0} \frac{p(t)}{t^a} = \lim_{t \to 1} \frac{p(t)}{(1 - t)^b} = 0$ . Then, there exists some $\epsilon, C > 0$ such that $p(t) \leq Ct^a(1 - t)^b$ if $t < \epsilon$ and $t > 1 - \epsilon$ . Define a positive measure $\mu_\epsilon$ by letting $\mu_\epsilon(S) = \mu(S \cap [\epsilon, 1 - \epsilon])$ for every Borel-measurable subset $S \subseteq [0, 1]$ . Besides, define another positive measure $\lambda$ by letting $\lambda(S) = \int_S C t^a(1 - t)^b \mathrm{d}t$ for every Borel-measurable subset $S \subseteq [0, 1]$ . Therefore, we have $\mu \leq \mu_\epsilon + \lambda$ , which indicates that $m_k^\mu \leq m_k^{\mu_\epsilon} + m_k^\lambda$ , and thus $M_k^\mu \leq M_k^{\mu_\epsilon} + M_k^\lambda$ . By Lemma 3, $M_k^\lambda \in \Theta(1)$ . For the other, observe that $m_k^{\mu_\epsilon} \leq m_k^{\delta_{1-\epsilon}}$ , and therefore $M_k^{\mu_\epsilon} \leq M_k^{\delta_{1-\epsilon}} \in \Theta(1)$ . The remaining case $W_k^\mu$ can be tackled similarly. To conclude, $\kappa(n), \tau(n) \in \Theta(1)$ .

Suppose $\limsup_{t\to 0}p(t) < \infty$ and $\limsup_{t\to 1}p(t) < \infty$ , then there exist $\epsilon, C > 0$ such that $p(t)\leq C$ for every $t\in [0,\epsilon)\cup (1 - \epsilon ,1]$ . Define $\mu_{\epsilon}$ as the one in the above. Then, $\mu \leq \mu_{\epsilon} + CU$ where $\mathcal{U}$ denotes the uniform measure. Therefore, $w_{k}^{\mu}\leq w_{k}^{\mu_{\epsilon}} + w_{k}^{CU}$ . Since $w_{k}^{CU} = \frac{C}{k + 1}$ , there is $W_{k}^{CU}\in \Theta (\log n)$ . Besides, $W_{k}^{\mu_{\epsilon}}\leq W_{k}^{\delta_{\epsilon}}\in \Theta (1)$ . The remaining case $M_k^\mu$ can be dealt with similarly. Therefore, the conclusion follows by using Lemma 2.

Next, we derive a more precise estimate for the Beta Shapley values.

Lemma 3. Let $B(\alpha, \beta)$ be the beta distribution with probability density function $\propto t^{\alpha - 1}(1 - t)^{\beta - 1}$ . Note that if $\mu = B(\alpha, \beta)$ , it yields the $\text{Beta}(\beta, \alpha)$ (a parameterized Beta Shapley value).

$$
M _ {n} ^ {B (\alpha , \beta)} \in \left\{ \begin{array}{l l} \Theta (\log n) & \beta = 1 \\ \Theta (1) & \beta > 1 \\ \Theta (n ^ {1 - \beta}) & 0 <   \beta <   1 \end{array} \right. a n d W _ {n} ^ {B (\alpha , \beta)} \in \left\{ \begin{array}{l l} \Theta (\log n) & \alpha = 1 \\ \Theta (1) & \alpha > 1 \\ \Theta (n ^ {1 - \alpha}) & 0 <   \alpha <   1 \end{array} \right..
$$

Proof. Let $\Gamma$ denote the Gamma function.

$$
m _ {k} ^ {\mathrm{B} (\alpha , \beta)} = \frac {\Gamma (\alpha + \beta)}{\Gamma (\alpha) \Gamma (\beta)} \cdot \frac {\Gamma (\alpha + k) \Gamma (\beta)}{\Gamma (\alpha + \beta + k)} = \frac {\prod_ {j = 0} ^ {k - 1} (\alpha + j)}{\prod_ {j = 0} ^ {k - 1} (\alpha + \beta + j)}.
$$

We write $a_{k} \sim b_{k}$ if $\lim_{k \to \infty} \frac{a_{k}}{b_{k}}$ converges. Since $\Gamma(x) = \lim_{n \to \infty} \frac{n! n^{x}}{x(x+1) \cdots (x+n)}$ , e.g., see (Rudin 1953, Eq. (95) in Chapter 8), we have $m_{k}^{\mathrm{B}(\alpha,\beta)} \sim \frac{k! k^{\alpha}}{k! k^{\alpha+\beta}} = k^{-\beta}$ , and thus the conclusion follows. The remaining case can be derived similarly.

Remark 8. Lemma 3 is useful for obtaining the time complexity of the proposed estimators for the Beta Shapley values parameterized by $\alpha, \beta > 0$ . If $\alpha, \beta > 1$ , the two proposed estimators require $O\left(\frac{n}{\epsilon^2} \log \frac{n}{\delta}\right)$ utility evaluations to achieve an $(\epsilon, \delta)$ -approximation. Note that this is the currently best time complexity. For the Shapley value, i.e., Beta(1,1), it is $O\left(\frac{n}{\epsilon^2} \log \frac{n}{\delta}\right)$ for GELS-R, and $O\left(\frac{n}{\epsilon^2} \log \left(\frac{n}{\delta}\right) \log (n)^2\right)$ for GELS and GELS-Shapley. To our knowledge, in terms of $(\epsilon, \delta)$ -approximation, the previously best time complexity for the Shapley value is $O\left(\frac{n}{\epsilon^2} \log \left(\frac{n}{\delta}\right) \log n\right)$ achieved by the group testing estimator (Wang and Jia 2023, Theorem C.7).

# J MORE EXPERIMENT RESULTS

For training estimators on MNIST, all performance curves are provided in Figure 11. It can be seen that the conclusions we have in the main paper still hold on MNIST.

The paired sampling technique was proposed by Covert and Lee (2021) to enhance (unbiased) KernelSHAP, but we notice that it can also be employed for GELS, GELS-R, GELS-Shapley, (weighted)

sampling lift, group testing, AME, MSR and simSHAP. Precisely, suppose the sampled subsets is $\{S_{k}\}_{k\geq1}$ , the paired sampling employs sampled subsets $\{T_{j}\}_{j\geq1}$ such that $T_{2k-1}=S_{k}$ and $T_{2k}=[n]\backslash S_{k}$ for every $k\geq1$ . Roughly speaking, symmetric probabilistic values, i.e., $p_{s}^{n}=p_{n-s}^{n}$ for every $s\in[n]$ could possibly take advantage of this technique. Examples include the Shapley value, the Banzhaf value and $\operatorname{Beta}(\gamma,\gamma)$ . An exception is that weighted sampling lift can be coupled with the paired sampling for any probabilistic value. In this appendix, we implement the paired sampling technique if possible. The results with utility functions reporting the classification accuracy on $D_{perf}$ are shown in Figures 3, 4, 5 and 6. Moreover, we also set the utility functions to report the cross-entropy loss instead, and the corresponding results are presented in Figures 7, 8, 9 and 10.

Remark 9. Observe that AME, simSHAP and group testing gain significant performance boosts using the paired sampling technique, while (unbiased) KernelSHAP takes advantage of this technique occasionally. For other estimators, the paired sampling does not play a noticeable role. Interestingly, while using the paired sampling technique, group testing is exactly equal to GELS, whereas GELS-Shapley, unbiased KernelSHAP and simSHAP are all equal.

To see the equality between GELS and group testing while using the paired sampling technique, notice that they share the same way of sampling subsets. Therefore, suppose we have one pair of samples $(S_{1}, S_{2})$ where $S_{2} = [n + 1] \backslash S_{1}$ . According to Algorithm 2, the corresponding $i$ -th estimate of GELS is

$$
\hat {\phi} _ {i} ^ {\text { GELS }} = \left\{ \begin{array}{l l} H _ {n} \cdot (U ^ {n} (S _ {1} \cap [ n ]) - U ^ {n} (S _ {2} \cap [ n ])), & i \in S _ {1} \text {   and   } n + 1 \not \in S _ {1} \\ H _ {n} \cdot (U ^ {n} (S _ {2} \cap [ n ]) - U ^ {n} (S _ {1} \cap [ n ])), & i \not \in S _ {1} \text {   and   } n + 1 \in S _ {1} \\ 0, & \text { otherwise } \end{array} \right.
$$

where $H_{n} = \sum_{s=1}^{n} \frac{1}{s}$ . Looking into the procedure of including a dummy player for group testing (Wang and Jia 2023, Appendix C.3), the reader can verify that group testing produces the same i-th estimate using the paired samples $(S, [n+1] \backslash S)$ .

For the remaining equality while employing the paired sampling technique, observe that they also have the same way of sampling subsets. Again, suppose we have one pair of samples $(S_{1}, S_{2})$ where $S_{2} = [n] \backslash S_{1}$ . By Algorithm 3, the corresponding i-th estimate of GELS-Shapley is

$$
\hat {\phi} _ {i} ^ {\mathrm{GELS-Shapley}} = \left\{ \begin{array}{l l} \frac {U ^ {n} ([ n ]) - U ^ {n} (\emptyset)}{n} + \frac {H _ {n - 1} \cdot s _ {2}}{n} (U ^ {n} (S _ {1}) - U ^ {n} (S _ {2}))  , & i \in S _ {1} \\ \frac {U ^ {n} ([ n ]) - U ^ {n} (\emptyset)}{n} + \frac {H _ {n - 1} \cdot s _ {1}}{n} (U ^ {n} (S _ {2}) - U ^ {n} (S _ {1}))  , & i \in S _ {2} \end{array} \right.
$$

where $H_{n - 1} = \sum_{s = 1}^{n - 1}\frac{1}{s}$ . As proved by Fumagalli et al. (2023, Theorem 4.5), the corresponding $i$ -th estimate of unbiased KernelSHAP is

$$
\hat {\phi} _ {i} ^ {\text { unbiased   KernelSHAP }} = \frac {\hat {U} ^ {n} ([ n ]) - \hat {U} ^ {n} (\emptyset)}{n} + \frac {2 H _ {n - 1}}{T} \sum_ {t = 1} ^ {T} \hat {U} ^ {n} (S _ {t}) \cdot \left(\mathbb {1} _ {i \in S _ {t}} - \frac {s _ {t}}{n}\right)
$$

where $\hat{U}^n (S) = U^n (S) - U^n (\emptyset)$ . On the other hand, the corresponding $i$ -th estimate of simSHAP can be expressed as

$$
\hat {\phi} _ {i} ^ {\text { simSHAP }} = \frac {U ^ {n} ([ n ]) - U ^ {n} (\emptyset)}{n} + \frac {2 H _ {n - 1}}{T} \sum_ {t = 1} ^ {T} U ^ {n} (S _ {t}) \cdot \left(\mathbb {1} _ {i \in S _ {t}} - \frac {s _ {t}}{n}\right). \tag {23}
$$

The reader can verify that $\hat{\phi}_i^{\mathrm{GELS - Shapley}} = \hat{\phi}_i^{\mathrm{unbiased KernelSHAP}} = \hat{\phi}_i^{\mathrm{simSHAP}}$ using $T = 2$ and $S_{2} = [n]\backslash S_{1}$ . We point out that Eq. (23) is not the original formula of simSHAP but an equivalent one that has been implicitly mentioned in (Fumagalli et al. 2023, Remark B.1 and Appendix B.4) where they argued that the choice of $\hat{U}^n$ is better than $U^n$ . Our experiments confirm that unbiased KernelSHAP converges significantly faster than simSHAP while not using the paired sampling technique.

![](images/12814f3997f122f1ac101b49f33665804bccdaf95f6d869f61032ceaff8ee5fd.jpg)  
GELS (ours) sampling lift group testing permutation AME GELS-Shapley (ours) kernelSHAP ARM weighted sampling lift MSR GELS-R (ours) unbiased kernelSHAP complement simSHAP

Figure 3: Comparison of different estimators on four probabilistic values using the dataset wind. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ . The utility functions report the classification accuracy on $D_{perf}$ . The dashed lines correspond to the use of the paired sampling technique.

![](images/1397fc6e528cbd56c9210480bab407b3f503833c3bda02e87781f4ef6b6bde18.jpg)  
GELS (ours) sampling lift group testing permutation AME GELS-Shapley (ours) kernelSHAP ARM weighted sampling lift MSR GELS-R (ours) unbiased kernelSHAP complement simSHAP

Figure 4: Comparison of different estimators on four probabilistic values using the dataset iris. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ . The utility functions report the classification accuracy on $D_{perf}$ . The dashed lines correspond to the use of the paired sampling technique.

![](images/208d0b252f119a38fb84f8436a65454325d3396bd4328b4b4890f93e7836b5da.jpg)  
GELS (ours) sampling lift group testing permutation AME GELS-Shapley (ours) kernelSHAP ARM weighted sampling lift MSR GELS-R (ours) unbiased kernelSHAP complement simSHAP

Figure 5: Comparison of different estimators on four probabilistic values using the dataset MNIST. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ . The utility functions report the classification accuracy on $D_{perf}$ . The dashed lines correspond to the use of the paired sampling technique.

![](images/b8fcfa36b235d2ef450b3558a9048777cf88b38f67a45c0656f1ed679d489116.jpg)  
GELS (ours) sampling lift group testing permutation AME GELS-Shapley (ours) kernelSHAP ARM weighted sampling lift MSR GELS-R (ours) unbiased kernelSHAP complement simSHAP

Figure 6: Comparison of different estimators on four probabilistic values using the dataset FMNIST. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ . The utility functions report the classification accuracy on $D_{perf}$ . The dashed lines correspond to the use of the paired sampling technique.

![](images/1e0b5a0b09a744348aa97701034adca3e2912371b7d6a1b23005b57f58da86b8.jpg)  
GELS (ours) sampling lift group testing permutation AME GELS-Shapley (ours) kernelSHAP ARM weighted sampling lift MSR GELS-R (ours) unbiased kernelSHAP complement simSHAP

Figure 7: Comparison of different estimators on four probabilistic values using the dataset wind. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ . The utility functions report the cross-entropy loss on $D_{perf}$ . The dashed lines correspond to the use of the paired sampling technique.

![](images/20fe38d92ea36447622be07c2a482ea504b01d98c30940a3fe4af827c0e813c5.jpg)  
GELS (ours) sampling lift group testing permutation AME GELS-Shapley (ours) kernelSHAP ARM weighted sampling lift MSR GELS-R (ours) unbiased kernelSHAP complement simSHAP

Figure 8: Comparison of different estimators on four probabilistic values using the dataset iris. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ . The utility functions report the cross-entropy loss on $D_{perf}$ . The dashed lines correspond to the use of the paired sampling technique.

![](images/babb566ad2735f7fbf4b8cab2888dd9cc87c5a579cbd11a0b7d632e9b8949ff9.jpg)  
GELS (ours) sampling lift group testing permutation AME GELS-Shapley (ours) kernelSHAP ARM weighted sampling lift MSR GELS-R (ours) unbiased kernelSHAP complement simSHAP

Figure 9: Comparison of different estimators on four probabilistic values using the dataset MNIST. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ . The utility functions report the cross-entropy loss on $D_{perf}$ . The dashed lines correspond to the use of the paired sampling technique.

![](images/ed1f874e19c0113b16c507a8e60399f4c65ca6f7b445543d6d595cc9a1145475.jpg)  
GELS (ours) sampling lift group testing permutation AME GELS-Shapley (ours) kernelSHAP ARM weighted sampling lift MSR GELS-R (ours) unbiased kernelSHAP complement simSHAP

Figure 10: Comparison of different estimators on four probabilistic values using the dataset FM-NIST. The relative difference is plotted in log scale while the Spearman correlation is in logit scale. In addition, $\bar{r} = 1 - r$ . The utility functions report the cross-entropy loss on $D_{perf}$ . The dashed lines correspond to the use of the paired sampling technique.

![](images/21f5e38554dbe799008655bdef23709706afe4aadea90ec83db2ed56c07389ad.jpg)

<details>
<summary>line</summary>

| #batches (k) | empirical loss |
| ------------ | -------------- |
| 0            | ~0.01          |
| 5            | ~0.005         |
| 10           | ~0.003         |
| 15           | ~0.002         |
| 20           | ~0.0015        |
| 25           | ~0.001         |
| 30           | ~0.001         |
</details>

![](images/81d71ccd443280b4d63c9d3829e3d8d6d7a3e36c3b811675d277d16ee2cde3fd.jpg)

<details>
<summary>line</summary>

| #batches (k) | specific | average | best |
| ------------ | -------- | ------- | ---- |
| 0            | 10^1     | 10^1    | 10^1 |
| 5            | ~10^0.5  | ~10^0.8 | ~10^0.3 |
| 10           | ~10^0.3  | ~10^0.6 | ~10^0.2 |
| 15           | ~10^0.2  | ~10^0.5 | ~10^0.1 |
| 20           | ~10^0.2  | ~10^0.5 | ~10^0.1 |
| 25           | ~10^0.2  | ~10^0.5 | ~10^0.1 |
| 30           | ~10^0.2  | ~10^0.5 | ~10^0.1 |
</details>

![](images/0f105c7901a54759d882dd06137255ca3131cf7d7905631f16971f4f36e0a0c4.jpg)

<details>
<summary>line</summary>

| #batches (k) | specific | average | best |
| ------------ | -------- | ------- | ---- |
| 0            | 0.0      | 0.0     | 0.0  |
| 5            | 0.6      | 0.6     | 0.7  |
| 10           | 0.8      | 0.8     | 0.9  |
| 15           | 0.85     | 0.85    | 0.9  |
| 20           | 0.85     | 0.85    | 0.9  |
| 25           | 0.85     | 0.85    | 0.9  |
| 30           | 0.85     | 0.85    | 0.9  |
</details>

![](images/44fee1f8e7c9cf632f1210d89a4a44af474fd92af0d37e0c8f5a53746b0125fc.jpg)

<details>
<summary>line</summary>

| #utility evaluations per datum (k) | Monte Carlo | TrELS-SC (ours) | TrELS-RD (ours) |
| ---------------------------------- | ----------- | --------------- | --------------- |
| 0                                  | 10.0        | 10.0            | 1.0             |
| 5                                  | 2.0         | 10.0            | 1.0             |
| 10                                 | 1.5         | 10.0            | 1.0             |
| 15                                 | 1.2         | 10.0            | 1.0             |
| 20                                 | 1.0         | 10.0            | 1.0             |
</details>

![](images/e7f7d2ffb43f6984e42565a57b2aa6064c83c61c362f527da596d6c909e25b75.jpg)

<details>
<summary>line</summary>

| #utility evaluations per datum (k) | Monte Carlo | TrELS-SC (ours) | TrELS-RD (ours) |
| ---------------------------------- | ----------- | --------------- | --------------- |
| 0                                  | 0.1         | 0.9             | 0.9             |
| 5                                  | 0.6         | 0.9             | 0.9             |
| 10                                 | 0.75        | 0.9             | 0.9             |
| 15                                 | 0.8         | 0.9             | 0.9             |
| 20                                 | 0.85        | 0.9             | 0.9             |
</details>

Figure 11: i) The first row: The first plot is the empirical loss of the sampled batch, while the others present the relative distance $\|\hat{\varphi}-\varphi\|_{2}/\|\varphi\|_{2}$ and the Spearman correlation between $\varphi$ and $\hat{\varphi}$ . $\hat{\varphi}$ denotes the one predicted by the estimators trained on MNIST. The first two plots are in log scale. The label “best” reports the best one achieved by the trained estimators at each time point, whereas “specific” is the one with the random seed being 0. ii) The second row: the two plots compare the performance of the estimators trained on MNIST with the Monte-Carlo method using Eq (2). Specifically, TrELS-SC uses trained estimators that achieve the best Spearman correlation on $D_{val}$ , whereas RD stands for relative difference.