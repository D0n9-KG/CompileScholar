# Self-Calibrating Conformal Prediction

Lars van der Laan

University of Washington

lvdlaan@uw.edu

Ahmed M. Alaa

UC Berkeley and UCSF

amalaa@berkeley.edu

# Abstract

In machine learning, model calibration and predictive inference are essential for producing reliable predictions and quantifying uncertainty to support decision-making. Recognizing the complementary roles of point and interval predictions, we introduce Self-Calibrating Conformal Prediction, a method that combines Venn-Abers calibration and conformal prediction to deliver calibrated point predictions alongside prediction intervals with finite-sample validity conditional on these predictions. To achieve this, we extend the original Venn-Abers procedure from binary classification to regression. Our theoretical framework supports analyzing conformal prediction methods that involve calibrating model predictions and subsequently constructing conditionally valid prediction intervals on the same data, where the conditioning set or conformity scores may depend on the calibrated predictions. Real-data experiments show that our method improves interval efficiency through model calibration and offers a practical alternative to feature-conditional validity.

# 1 Introduction

Particularly in safety-critical sectors, such as healthcare, it is important to ensure decisions inferred from machine learning models are reliable, under minimal assumptions (Mandinach et al., 2006; Veale et al., 2018; Char et al., 2018; Quest et al., 2018; Vazquez and Facelli, 2022). As a response, there is growing interest in predictive inference methods that quantify uncertainty in model predictions via prediction intervals (Patel, 1989; Heskes, 1996). Conformal prediction (CP) is a popular, model-agnostic, and distribution-free framework for predictive inference, which can be applied post-hoc to any prediction pipeline (Vovk et al., 2005; Shafer and Vovk, 2008; Balasubramanian et al., 2014; Lei et al., 2018). Given a prediction issued by a black-box model, CP outputs a prediction interval that is guaranteed to contain the unseen outcome with a user-specified probability (Lei et al., 2018). However, a limitation of CP is that this prediction interval only provides valid coverage marginally, when averaged across all possible contexts – with ‘context’ referring to the information available for decision-making. Constructing informative prediction intervals that offer context-conditional coverage is generally unattainable without making additional distributional assumptions (Vovk, 2012; Lei and Wasserman, 2014; Barber et al., 2021). Consequently, there has been an upsurge in research developing CP methods that offer weaker, yet practically useful, notions of conditional validity; see, e.g., (Papadopoulos et al., 2008; Johansson et al., 2014; Romano et al., 2020; Jung et al., 2022; Guan, 2023; Gibbs et al., 2023).

In prediction settings, model calibration is a desirable property of machine learning predictors that ensures that the predicted outcomes accurately reflect the true outcomes (Mincer and Zarnowitz, 1969; Zadrozny and Elkan, 2001, 2002; Gneiting et al., 2007). Specifically, a predictor is calibrated for the outcome if the average outcome among individuals with identical predictions is close to their shared prediction value (Gupta et al., 2020). Such a predictor is more robust against the over-or-under estimation of the outcome in extremes of predicted values. It also has the property that the best prediction of the outcome conditional on the model's prediction is the prediction itself, which facilitates transparent decision-making (van der Laan et al., 2023). There is a rich literature studying

post-hoc calibration of prediction algorithms using techniques such as Platt's scaling (Platt et al., 1999; Cox, 1958), histogram binning (Zadrozny and Elkan, 2001; Gupta et al., 2020; Gupta and Ramdas, 2021), isotonic calibration (Zadrozny and Elkan, 2002; Niculescu-Mizil and Caruana, 2005; van der Laan et al., 2023), and Venn-Abers calibration (Vovk and Petej, 2012).

Given the roles of both point and interval predictions in decision-making, we introduce a dual calibration objective that aims to construct (i) calibrated point predictions and (ii) associated prediction intervals with valid coverage conditional on these point predictions. Marrying model calibration and predictive inference, we propose a solution to this objective that combines two post-hoc approaches — Venn-Abers calibration (Vovk et al., 2003; Vovk and Petej, 2012) and CP (Vovk et al., 2005) — to simultaneously provide point predictions and prediction intervals that achieve our dual objective in finite samples. In doing so, we extend the original Venn-Abers procedure from binary classification to the regression setting. Our theoretical and experimental results support the integration of model calibration into predictive inference methods to improve interval efficiency and interpretability.

# 2 Problem setup

# 2.1 Notation

We consider a standard regression setup in which the input $X \in X \subset R^{d}$ corresponds to contextual information available for decision-making, and the output $Y \in Y \subset R$ is an outcome of interest. We assume that we have access to a calibration dataset $\mathcal{C}_{n} = \{(X_{i}, Y_{i})\}_{i=1}^{n}$ comprising n i.i.d. data points drawn from an unknown distribution $P := P_{X}P_{Y|X}$ . We assume access to a black-box predictor $f : X \mapsto Y$ , obtained by training an ML model on a dataset that is independent of $C_{n}$ . Throughout this paper, we do not make any assumptions on the model f or the distribution P. For a quantile level $\alpha \in (0,1)$ , we denote the “pinball” quantile loss function $\ell_{\alpha}$ by $\ell_{\alpha}(f(x), y) := 1(y \geq f(x)) \cdot \alpha(y - f(x)) + 1(y < f(x)) \cdot (1 - \alpha)(f(x) - y)$ .

# 2.2 Conditional predictive inference and a curse of dimensionality

Let $(X_{n+1}, Y_{n+1})$ be a new data point drawn from P independently of the calibration data $C_n$ . Our high-level aim is to develop a predictive inference algorithm that constructs a prediction interval $\widehat{C}_n(X_{n+1})$ around the point prediction issued by the black-box model, i.e., $f(X_{n+1})$ . For this prediction interval to be deemed valid, it should cover the true outcome $Y_{n+1}$ with a probability $1 - \alpha$ . Conformal prediction (CP) is a method for predictive inference that can be applied in a post-hoc fashion to any black-box model (Vovk et al., 2005). The vanilla CP procedure issues prediction intervals that satisfy the marginal coverage condition:

$$
\mathbb {P} (Y _ {n + 1} \in \widehat {C} _ {n} (X _ {n + 1})) \geq 1 - \alpha , \tag {1}
$$

where the probability P is taken with respect to the randomness in $C_{n}$ and $(X_{n+1}, Y_{n+1})$ . However, marginal coverage might lack utility in decision-making scenarios where decisions are context-dependent. A prediction band $\widehat{C}_{n}(x)$ achieving 95% coverage may exhibit arbitrarily poor coverage for specific contexts x. Ideally, we would like this coverage condition to hold for each context $x \in X$ , i.e., the conventional notion of “conditional validity” requires

$$
\mathbb {P} (Y _ {n + 1} \in \widehat {C} _ {n} (X _ {n + 1}) | X _ {n + 1} = x) \geq 1 - \alpha , \tag {2}
$$

for all $x \in \mathcal{X}$ . However, previous work has shown that it is impossible to achieve (2) without distributional assumptions (Vovk, 2012; Lei and Wasserman, 2014; Barber et al., 2021).

While context-conditional validity as in (2) is generally unachievable, it is feasible to attain weaker forms of conditional validity. Given any finite set of groups $\mathcal{G}$ and a grouping function $G:\mathcal{X}\times \mathcal{Y}\to \mathcal{G}$ , Mondrian-CP offers coverage conditional on group membership, that is, $\mathbb{P}(Y_{n + 1}\in \widehat{C}_n(X_{n + 1})|G(X_{n + 1},Y_{n + 1}) = g)\geq 1 - \alpha$ , $\forall g\in \mathcal{G}$ (Vovk et al., 2005; Romano et al., 2020). Expanding upon group- and context-conditional coverage, A multicalibration objective was introduced in Deng et al. (2023) that seeks to satisfy, for all $h$ in an (infinite-dimensional) class $\mathcal{F}$ of weighting functions (i.e., 'covariate shifts'), the property:

$$
\mathbb {E} \left[ h (X _ {n + 1}) \{(1 - \alpha) - 1 \{Y _ {n + 1} \in \widehat {C} _ {n} (X _ {n + 1}) \} \} \right] = 0. \tag {3}
$$

Gibbs et al. (2023) proposed a regularized CP framework for (approximately) achieving (3) that provides a means to trade off the efficiency (i.e., width) of prediction intervals and the degree of conditional coverage achieved. However, Barber et al. (2021) and Gibbs et al. (2023) establish the existence of a “curse of dimensionality”: as the dimension of the context increases, smaller classes of weighting functions must be considered to retain the same level of efficiency. For group-conditional coverage, this curse of dimensionality manifests in the size of the subgroup class G (Barber et al., 2021) via its VC dimension (Vapnik et al., 1994). Thus, especially in data-rich contexts, prediction intervals with meaningful multicalibration guarantees over the context space may be too wide for decision-making.

# 2.3 A dual calibration objective

In decision-making, both point predictions and prediction intervals play a role. For example, in scenarios with a low signal-to-noise ratio, prediction intervals may be too wide to directly inform decision-making, as their width is typically of the order of the standard deviation of the outcome. Point predictions might be used to guide decisions, while prediction intervals help quantify deviations of point predictions from unseen outcomes and assess the risk associated with these decisions.

Viewing the black-box model f as a scalar dimension reduction of the context x, a natural relaxation of the infeasible objective of context-conditional validity in (2) is prediction-conditional validity, i.e., $P(Y_{n+1} \in \widehat{C}_{n}(X_{n+1})|f(X_{n+1})) \geq 1 - \alpha$ . Prediction-conditional validity ensures that the interval widths adapt to the outputs of the model $f(\cdot)$ , so that the intervals can be reliably used to quantify the deviation of model predictions from unseen outcomes. Since prediction-conditional validity only requires coverage conditional on a one-dimensional random variable, it avoids the curse of dimensionality associated with context-conditional validity. In addition, as illustrated in our experiments in Section 5 and Appendix C, when the heteroscedasticity (e.g., variance) in the outcome is a function of its conditional mean, prediction-conditional validity can closely approximate context-conditional validity, so long as the predictor estimates the conditional mean of the outcome sufficiently well.

Given the roles of both point and interval predictions in decision-making, we introduce a novel dual calibration objective, self-calibration, that aims to construct (i) calibrated point predictions and (ii) associated prediction intervals with valid coverage conditional on these point predictions. Formally, given the model $f$ and calibration data $\mathcal{C}_n \cup \{X_{n+1}\}$ , our objective is to post-hoc construct a calibrated point prediction $f_{n+1}(X_{n+1})$ and a compatible prediction interval $\widehat{C}_{n+1}(X_{n+1})$ centered around $f_{n+1}(X_{n+1})$ that satisfies the following desiderata:

(i) Perfectly Calibrated Point Prediction: $f_{n + 1}(X_{n + 1}) = \mathbb{E}[Y_{n + 1} \mid f_{n + 1}(X_{n + 1})]$ .   
(ii) Prediction-Conditional Validity: $\mathbb{P}(Y_{n + 1}\in \widehat{C}_{n + 1}(X_{n + 1})|f_{n + 1}(X_{n + 1}))\geq 1 - \alpha .$

Desideratum (i) states that the point prediction $f_{n+1}(X_{n+1})$ should be perfectly calibrated — or self-consistent (Flury and Tarpey, 1996) — for the true outcome $Y_{n+1}$ (Lichtenstein et al., 1977; Gupta et al., 2020; van der Laan et al., 2023). It is widely recognized that the calibration of model predictions is important to ensure their reliability, trustworthiness, and interpretability in decision-making (Lichtenstein et al., 1977; Zadrozny and Elkan, 2001; Bella et al., 2010; Guo et al., 2017; Davis et al., 2017). Desideratum (i) also improves interval efficiency by ensuring $\widehat{C}_{n+1}(X_{n+1})$ is centered around an unbiased prediction, meaning the interval's width is driven by outcome variation rather than by prediction bias. Desideratum (ii) is a prediction interval variant of (i) that ensures the prediction interval $\widehat{C}_{n+1}(X_{n+1})$ is calibrated with respect to the model $f_{n+1}$ , providing valid coverage for $Y_{n+1}$ within contexts with the same calibrated point prediction. We refer to a predictive inference algorithm simultaneously satisfying (i) and (ii) as self-calibrating, as such a procedure is automatically able to adapt to miscalibration in the model $f(\cdot)$ due to, e.g., model misspecification or distribution shifts, ensuring that the interval is constructed from a calibrated predictive model.

Self-calibration can also be motivated by a decision-making scenario where point predictions determine actions and prediction intervals are used to apply these actions selectively. When point predictions are sufficient statistics for actions, self-calibration implies that the point and interval predictions are accurate, on average, within the subset of all contexts receiving the same prediction and, therefore, the same action.

# 3 Self-Calibrating Conformal Prediction

A key advantage of CP is that it can be applied post-hoc to any black-box model f without disrupting its point predictions. However, desideratum (i) introduces a perfect calibration requirement for the point predictions of f, thereby interfering with the underlying model specification. In this section, we introduce Self-Calibrating CP (SC-CP), a modified version of CP that is self-calibrating in that it satisfies (i) and (ii), while preserving all the favorable properties of CP, including its finite-sample validity and post-hoc applicability. Before describing our complete procedure in Section 3.3, we provide background on point calibration and propose Venn-Abers calibration for regression.

# 3.1 Preliminaries on point calibration

Following the framing of van der Laan et al. (2023) (see also Gupta et al. (2020)), a point calibrator is a post-hoc procedure that aims to learn a transformation $\theta_{n}: R \to R$ of the black-box model f such that: (1) $\theta_{n}(f(X_{n+1}))$ is well-calibrated for $Y_{n+1}$ in the sense of Desideratum (i); and (2) $\theta_{n} \circ f$ is comparably predictive to f. Condition (2) ensures that in the process of achieving (1), the quality of the model f is not compromised, and excludes trivial calibrators such as $\theta_{n}(f(\cdot)) := \frac{1}{n} \sum_{i=1}^{n} Y_{i}$ (Gupta and Ramdas, 2021). To our knowledge, this notion of calibration traces back to Mincer and Zarnowitz (1969), who introduced the idea of regressing outcomes on predictions to achieve calibration in forecasting.

Commonly-employed point calibrators include Platt's scaling (Platt et al., 1999; Cox, 1958), histogram (or quantile) binning (Zadrozny and Elkan, 2001), and isotonic calibration (Zadrozny and Elkan, 2002; Niculescu-Mizil and Caruana, 2005). Mechanistically, these point calibrators learn $\theta_{n}$ by regressing the outcomes $\{Y_i\}_{i=1}^n$ on the model predictions $\{f(X_i)\}_{i=1}^n$ . Importantly, however, point calibration fundamentally differs from the regression task of learning $E_P[Y|f(X)]$ , as calibration can be achieved without smoothness assumptions, allowing for misspecification of the regression task (Gupta et al., 2020). Histogram binning is a simple and distribution-free calibration procedure (Gupta et al., 2020; Gupta and Ramdas, 2021) that learns $\theta_{n}$ via a histogram regression over a finite (outcome-agnostic) binning of the output space $f(\mathcal{X})$ . Isotonic calibration is an outcome-adaptive binning method that uses isotonic regression (Barlow and Brunk, 1972) to learn $\theta_{n}$ by minimizing the empirical mean square error over all 1D piece-wise constant, monotone nondecreasing transformations. Isotonic calibration is distribution-free — it does not rely on monotonicity assumptions — and, in contrast with histogram binning, it is tuning parameter-free and naturally preserves the mean-square error of the original predictor (as the identity transform is monotonic) (van der Laan et al., 2023). A key limitation of histogram binning and isotonic calibration is that their calibration guarantees are only approximate, and desideratum (i) only holds asymptotically.

# 3.2 Venn-Abers calibration

For binary classification, Vovk and Petej (2012) proposed Venn-Abers calibration, which iterates isotonic calibration over imputations $y \in Y$ of the unseen outcome $Y_{n+1}$ to provide calibrated multi-probabilistic predictions in finite samples. In this section, we generalize the Venn-Abers calibration procedure to regression, offering finite-sample calibration guarantees for non-binary outcomes.

Let $\Theta_{\mathrm{iso}}$ consist of all univariate, piecewise constant functions that are monotonically nondecreasing. Our Venn-Abers calibration procedure, outlined in Alg. 1, is derived from an oracle variant of isotonic calibration that provides a perfectly calibrated point prediction in finite samples, but requires knowledge of the true outcome $Y_{n + 1}$ . Specifically, the Venn-Abers calibration algorithm iterates over imputed outcomes $y\in \mathcal{Y}$ for $Y_{n + 1}$ and applies isotonic calibration to the augmented dataset $\mathcal{C}_n\cup \{(X_{n + 1},y)\}$ to produce a set of point predictions $f_{n,X_{n + 1}}(X_{n + 1}):= \{f_n^{(X_{n + 1},y)}(X_{n + 1}):y\in \mathcal{Y}\} \}$ . When the outcome space $\mathcal{Y}$ is non-discrete, Alg. 1 may be infeasible to compute exactly and can be approximated by discretizing $\mathcal{Y}$ . Nonetheless, the range of the Venn-Abers multi-prediction can be feasibly computed as $[f_n^{(x,y_\min)}(x),f_n^{(x,y_\max)}(x)]$ where $[y_{\min},y_{\max}] := \mathrm{range}(\mathcal{Y})$ , in light of the min-max representation of isotonic regression (Lee, 1983).

Unlike point calibrators, Venn-Abers calibration generates a set of calibrated predictions for each context $X_{n+1}$ , indexed by $y \in Y$ . As we demonstrate later, this set prediction is guaranteed in finite samples to include a perfectly calibrated point prediction, namely, the oracle prediction $f_{n}^{(X_{n+1}, Y_{n+1})}(X_{n+1})$ corresponding to the true outcome $Y_{n+1}$ . Moreover, each prediction in the set,

being obtained via isotonic calibration, still enjoys the same large-sample calibration guarantees as isotonic calibration (van der Laan et al., 2023). By the stability of isotonic regression, as the size of the calibration set n increases, the width of this set of predictions rapidly narrows, eventually converging to a single, perfectly calibrated point prediction (Vovk and Petej, 2012). Venn-Abers calibration thus provides a measure of epistemic uncertainty by producing a range of values for a perfectly calibrated point prediction. In cases with small sample sizes, standard isotonic calibration can overfit, leading to poorly calibrated point predictions. When this overfitting occurs, the Venn-Abers set prediction widens, reflecting greater uncertainty in the perfectly calibrated point prediction within the set (Johansson et al., 2023).

For the binary classification case, Vovk and Petej (2012) derived a (large-sample) calibrated point prediction from the Venn-Abers multi-prediction using a shrinkage approach. We can similarly construct, for each $x \in X$ , a point prediction as follows:

$$
\widetilde {f} _ {n + 1, x} (x) := f _ {n + 1, x} ^ {\text { mid }} (x) + \frac {f _ {n} ^ {(x , y _ {\max})} (x) - f _ {n} ^ {(x , y _ {\min})} (x)}{y _ {\max} - y _ {\min}} \left\{\overline {{y}} _ {n} - f _ {n + 1, x} ^ {\text { mid }} (x) \right\}, \tag {4}
$$

where $f_{n+1,x}^{\mathrm{mid}}(x) := \frac{1}{2}\{f_n^{(x,y_\mathrm{max})}(x) + f_n^{(x,y_\mathrm{min})}(x)\}$ is the midpoint of the multi-prediction and $\overline{y}_n := \frac{1}{n} \sum_{i=1}^{n} Y_i$ . The behavior of $\widetilde{f}_{n+1,x}(x)$ is natural; it shrinks the point prediction $f_{n+1,x}^{\mathrm{mid}}(x)$ towards the average outcome $\overline{y}_n$ (a well-calibrated prediction) proportional to how uncertain we are in the calibration of $f_{n+1,x}^{\mathrm{mid}}(x)$ . The ratio $\frac{1}{y_{\mathrm{max}} - y_{\mathrm{min}}} (f_n^{(x,y_{\mathrm{max}})}(x) - f_n^{(x,y_{\mathrm{min}})}(x))$ measures the sensitivity of isotonic regression to the addition of a single data point to $C_n$ , and a value closer to 1 corresponds to a higher degree of overfitting. In the extreme case where the calibration dataset is very large, we have $f_n^{(x,y_{\mathrm{max}})}(x) \approx f_n^{(x,y_{\mathrm{min}})}(x)$ , implying that $\widetilde{f}_{n+1,x}(x) \approx f_{n+1,x}^{\mathrm{mid}}(x)$ . Conversely, in the opposite extreme where the calibration dataset is very small and isotonic regression overfits, we have $f_n^{(x,y_{\mathrm{max}})}(x) \approx y_{\mathrm{max}}$ and $f_n^{(x,y_{\mathrm{min}})}(x) \approx y_{\mathrm{min}}$ , in which case $\widetilde{f}_{n+1,x}(x) \approx \overline{y}_n$ . We could replace $\overline{y}_n$ in (4) with any reference predictor, such as one calibrated using Platt's scaling or quantile-binning.

# 3.3 Conformalizing Venn-Abers Calibration

In this section, we propose Self-Calibrating Conformal Prediction, which conformalizes the Venn-Abers calibration procedure to provide prediction intervals centered around the Venn-Abers multi-prediction that are self-calibrated in the sense of desiderata (i) and (ii).

A simple, albeit naive, strategy for achieving (i) and (ii) without finite-sample guarantees involves using the dataset $C_{n}$ to calibrate point predictions of $f(\cdot)$ through isotonic regression, and then constructing prediction intervals from the $1 - \alpha$ empirical quantiles of prediction errors within subgroups defined by unique values of the calibrated point predictions. To motivate our SC-CP algorithm, we introduce an infeasible variant of this seemingly naive procedure that is valid in finite samples, but can only be computed by an oracle that knows the unseen outcome $Y_{n+1}$ . In this oracle procedure, we compute a perfectly calibrated prediction $f_{n+1}^{*}(X_{n+1}) := \theta_{n+1}^{*}(f(X_{n+1}))$ by isotonic calibrating f using the oracle-augmented calibration set $\{(X_{i}, Y_{i})\}_{i=1}^{n+1}$ , where $\theta_{n+1}^{*} \in \argmin_{\theta \in \Theta_{\mathrm{iso}}} \sum_{i=1}^{n+1} \{Y_{i} - \theta(f(X_{i}))\}^{2}$ . Next, we compute the conformity scores $S_{i}^{*} := |Y_{i} - f_{n+1}^{*}(X_{i})|$ as the absolute residuals of the calibrated predictions. An oracle prediction interval is then given by $C_{n+1}^{*}(X_{n+1}) := f_{n+1}^{*}(X_{n+1}) \pm \rho_{n+1}^{*}(X_{n+1})$ , where $\rho_{n+1}^{*}(X_{n+1})$ is the empirical $1 - \alpha$ quantile of conformity scores with calibrated predictions identical to $X_{n+1}$ , that is, scores in the set $\{S_{i}^{*} : f_{n+1}^{*}(X_{i}) = f_{n+1}^{*}(X_{n+1}), i \in [n+1]\}$ . Importantly, isotonic regression, which is an outcome-adaptive histogram binning method, ensures that the calibrated model $f_{n+1}^{*}$ is piece-wise constant, with a sufficiently large number of observations averaged within each constant segment—typically on the order of $n^{2/3}$ (Deng et al., 2021). Consequently, the empirical quantile $\rho_{n+1}^{*}(X_{n+1})$ is generally stable with relatively low variability across realizations of $C_{n}$ . In our proofs, using the first-order conditions characterizing the optimizer $\theta_{n+1}$ and exchangeability, we show that $f_{n+1}^{*}(X_{n+1}) = \mathbb{E}[Y_{n+1}|f_{n+1}^{*}(X_{n+1})]$ , so that desideratum (i) is satisfied. Furthermore, we establish that the interval $C_{n+1}^{*}(X_{n+1})$ achieves desideratum (ii), i.e., $\mathbb{P}(Y_{n+1} \in C_{n+1}^{*}(X_{n+1}) \mid f_{n+1}^{*}(X_{n+1})) \geq 1 - \alpha$ . To do so, our key insight is that $\rho_{n+1}^{*}(X_{n+1})$ corresponds to the evaluation of the function $\rho_{n+1}^{*}$ computed via prediction-conditional quantile regression as:

$$
\rho_ {n + 1} ^ {*} \in \underset {\theta \circ f _ {n + 1} ^ {*}; \theta : \mathbb {R} \to \mathbb {R}} {\operatorname{argmin}} \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} \ell_ {\alpha} \left(\theta \circ f _ {n + 1} ^ {*} (X _ {i}), S _ {i}\right).
$$

# Algorithm 1 Venn-Abers Calibration

Input: Calibration data $\mathcal{C}_n = \{(X_i,Y_i)\}_{i=1}^n$ , model $f$ , context $x \in \mathcal{X}$ .

1: for each $y \in \mathcal{Y}$ do   
2: Set augmented dataset $\mathcal{C}_n^{(x,y)} := \mathcal{C}_n \cup \{(x,y)\}$ ;   
3: Apply isotonic calibration to $f$ using $\mathcal{C}_n^{(x,y)}$ :

$$
\theta_ {n} ^ {(x, y)} := \operatorname{argmin} _ {\theta \in \Theta_ {\mathrm{iso}}} \sum_ {i \in \mathcal {C} _ {n} ^ {(x, y)}} \left\{Y _ {i} - \theta \circ f (X _ {i}) \right\} ^ {2}.
$$

$$
f _ {n} ^ {(x, y)} := \theta_ {n} ^ {(x, y)} \circ f.
$$

4: end for

Output: Multi-prediction $\{f_n^{(x,y)}(x):y\in \mathcal{Y}\}$ .

![](images/f76f5ec3b89fe9a28efd15832aa8987bb75149555280e644e43adc7df265f3e5.jpg)

<details>
<summary>line</summary>

| Original Prediction (uncalibrated) | Predicted Outcome (Prediction Interval) | Venn-Abers Multi-Prediction (Predicted Interval) | Calibrated Prediction (Predicted Interval) | Original Prediction (Original Interval) | Outcome (Predicted Outcome) |
| ----------------------------------- | ---------------------------------------- | -------------------------------------------------- | ------------------------------------------ | --------------------------------------- | --------------------------- |
| 0.5                                 | 1.0                                      | 0.5                                                | 0.5                                        | 0.5                                     | 0.0                         |
| 1.0                                 | 2.0                                      | 1.0                                                | 1.0                                        | 1.0                                     | 0.0                         |
| 1.5                                 | 3.0                                      | 1.5                                                | 1.5                                        | 1.5                                     | 0.0                         |
| 2.0                                 | 4.0                                      | 2.0                                                | 2.0                                        | 2.0                                     | 0.0                         |
| 2.5                                 | 5.0                                      | 2.5                                                | 2.5                                        | 2.5                                     | 0.0                         |
| 3.0                                 | 6.0                                      | 3.0                                                | 3.0                                        | 3.0                                     | 0.0                         |
| 3.5                                 | 7.0                                      | 3.5                                                | 3.5                                        | 3.5                                     | 0.0                         |
</details>

Figure 1: Example SC-CP output with small $\mathcal{C}_n$ ( $n = 200$ ).

# Algorithm 2 Self-Calibrating Conformal Prediction

Input: Calibration data $\mathcal{C}_n = \{(X_i,Y_i)\}_{i = 1}^n$ , model $f$ , context $x\in \mathcal{X}$ , miscoverage level $\alpha \in (0,1)$

1: for each $y \in Y$ do   
2: Obtain calibrated model $f_{n}^{(x,y)}$ by isotonic calibrating $f$ on $\mathcal{C}_n \cup \{(x,y)\}$ as in Alg. 1;   
3: Set self-calibrating conformity scores $S_{i}^{(x,y)} = |Y_{i} - f_{n}^{(x,y)}(X_{i})|, \forall i \in [n]$ and $S_{n + 1}^{(x,y)} = |y - f_{n}^{(x,y)}(x)|$ ;   
4: Calculate $1 - \alpha$ empirical quantile $\rho_n^{(x,y)}(x)$ of conformity scores with same calibrated prediction as $x$ :

$$
\underset {q \in \mathbb {R}} {\operatorname{argmin}} \sum_ {i = 1} ^ {n} \mathbf {1} \left\{f _ {n} ^ {(x, y)} (X _ {i}) = f _ {n} ^ {(x, y)} (x) \right\} \cdot \ell_ {\alpha} (q, S _ {i} ^ {(x, y)}) + \ell_ {\alpha} (q, S _ {n + 1} ^ {(x, y)});
$$

5: end for

6: Set $f_{n+1}(x) := \{f_n^{(x,y)}(x) : y \in \mathcal{Y}\}$ .   
7: Set $\widehat{C}_{n+1}(x):=\{y\in\mathcal{Y}:|y-f_{n}^{(x,y)}(x)|\leq\rho_{n}^{(x,y)}(x)]\}$ .

Output: $f_{n+1}(x) \subset \text{conv}(\mathcal{Y}), \widehat{C}_{n+1}(x) \subset \mathcal{Y}$

The first-order conditions characterizing the optimizer $\rho_{n+1}^*$ combined with the exchangeability between $\mathcal{C}_n$ and $(X_{n+1}, Y_{n+1})$ can be used to show that $C_{n+1}^*(X_{n+1})$ is multi-calibrated against the class of weighting functions $\mathcal{F}_{n+1} := \{\theta \circ f_{n+1}^*; \theta : \mathbb{R} \to \mathbb{R}\}$ in the sense of (3). Using first-order conditions to establish the theoretical validity of conformal prediction was also applied by Gibbs et al. (2023) to demonstrate the multi-calibration of oracle prediction intervals obtained from quantile regression over a fixed class $\mathcal{F}$ . In our case, quantile regression is performed over a data-dependent function class $\mathcal{F}_{n+1}$ , learned from the calibration data, which introduces additional challenges in our proofs.

Our SC-CP method, which is outlined in Alg. 2, follows a similar procedure to the above oracle procedure. Since the new outcome $Y_{n+1}$ is unobserved, we instead iterate the oracle procedure over all possible imputed values $y \in \mathcal{Y}$ for $Y_{n+1}$ . As in Alg. 1., this yields a set of isotonic calibrated models $f_{n,X_{n+1}} := \{f_n^{(X_{n+1},y)} : y \in \mathcal{Y}\}$ , where $f_{n+1}(X_{n+1})$ is the Venn-Abers multi-prediction of $Y_{n+1}$ . Then, for each $y \in \mathcal{Y}$ and $i \in [n]$ , we define the self-calibrating conformity scores $S_i^{(X_{n+1},y)} := |Y_i - f_n^{(X_{n+1},y)}(X_i)|$ and $S_{n+1}^{(X_{n+1},y)} := |y - f_n^{(X_{n+1},y)}(X_{n+1})|$ , where the dependency of our scores on the imputed outcome $y \in \mathcal{Y}$ is akin to Full (or transductive) CP (Vovk et al., 2005). Our SC-CP interval is then given by $\widehat{C}_{n+1}(X_{n+1}) := \{y \in \mathcal{Y} : S_{n+1}^{(X_{n+1},y)} \leq \rho_n^{(X_{n+1},y)}(X_{n+1})\}$ , where $\rho_n^{(X_{n+1},y)}(X_{n+1})$ is the empirical $1 - \alpha$ quantile of the level set $\{S_i^{(X_{n+1},y)} : f_n^{(X_{n+1},y)}(X_i) = f_n^{(X_{n+1},y)}(X_{n+1}), i \in [n+1]\}$ . By definition, $\widehat{C}_{n+1}(X_{n+1})$ covers $Y_{n+1}$ if, and only if, the oracle interval $C_{n+1}^*(X_{n+1})$ covers $Y_{n+1}$ , thereby inheriting the self-calibration of $C_{n+1}^*(X_{n+1})$ . Formally, $\widehat{C}_{n+1}(X_{n+1})$ is a set, but it can be converted to an interval by taking its range, with little efficiency loss.

# 3.4 Computational considerations

The main computational cost of Alg. 1 and Alg. 2 is in the isotonic calibration step, executed for each $y \in \mathcal{Y}$ . Isotonic regression (Barlow and Brunk, 1972) can be scalably and efficiently computed using implementations of xgboost (Chen and Guestrin, 2016) for univariate regression trees with monotonicity constraints. Similar to Full (or transductive) CP (Vovk et al., 2005), Alg. 2 may be computationally infeasible for non-discrete outcomes, and can be approximated by iterating over a finite subset of $\mathcal{Y}$ . In our implementation, we iterate over a grid of $\mathcal{Y}$ and use linear interpolation to impute the threshold $\rho_n^{(x,y)}(x)$ and score $S_{n+1}^{(x,y)}$ for each $y \in \mathcal{Y}$ . As with Full CP and multicalibrated CP (Gibbs et al., 2023), Alg. 1 and Alg. 2 must be separately applied for each context $x \in \mathcal{X}$ . The algorithms depend solely on $x \in \mathcal{X}$ through its prediction $f(x)$ , so we can approximate the outputs for all $x \in \mathcal{X}$ by running each algorithm for a finite number of $x \in \mathcal{X}$ corresponding to a finite grid of the 1D output space $f(\mathcal{X}) = \{f(x): x \in \mathcal{X}\} \subset \mathbb{R}$ . In addition, both algorithms are fully parallelizable across both the input context $x \in \mathcal{X}$ and the imputed outcome $y \in \mathcal{Y}$ . In our implementation, we use nearest neighbor interpolation in the prediction space to impute outputs for each $x \in \mathcal{X}$ . In our experiments with sample sizes ranging from $n = 5000$ to 40000, quantile binning of both $f(\mathcal{X})$ and $\mathcal{Y}$ into 200 equal-frequency bins enables execution of Alg. 1 and Alg. 2 across all contexts in minutes with negligible approximation error.

# 4 Theoretical guarantees

In this section, under exchangeability of the data, we establish that the Venn-Abers multi-prediction $f_{n,X_{n+1}}(X_{n+1}):=\{f_{n}^{(X_{n+1},y)}(X_{n+1}):y\in\mathcal{Y}\}$ and SC-CP interval $\widehat{C}_{n+1}(X_{n+1})$ output by Alg. 2 satisfy desiderata (i) and (ii) in finite samples and without distributional assumptions. Under an iid condition, we further establish that, asymptotically, the Venn-Abers calibration step within the SC-CP algorithm results in better point predictions and, consequently, more efficient prediction intervals.

The following theorem establishes that the Venn-Abers multi-prediction is perfectly calibrated in the sense of Vovk et al. (2003), containing a perfectly calibrated point prediction of $Y_{n+1}$ in finite samples.

C1) Exchangeability: $\{(X_i,Y_i)\}_{i = 1}^{n + 1}$ are exchangeable.   
C2) Finite second moment: $E_P[Y^2] < \infty$ .

Theorem 4.1 (Perfect calibration of Venn-Abers multi-prediction). Under Conditions C1 and C2, the Venn-Abers multi-prediction $f_{n,X_{n + 1}}(X_{n + 1})$ almost surely satisfies the condition $f_{n}^{(X_{n + 1},Y_{n + 1})}(X_{n + 1}) = \mathbb{E}[Y_{n + 1}\mid f_n^{(X_{n + 1},Y_{n + 1})}(X_{n + 1})]$ .

Theorem 4.1 generalizes an analogous result by Vovk and Petej (2012) for the special case of binary classification. Even in this special case, our proof is novel and elucidates how Venn-Abers calibration uses exchangeability with the least-squares loss in a manner analogous to how CP uses exchangeability with the quantile loss (Gibbs et al., 2023).

The following theorem establishes desideratum (ii) for the interval $\widehat{C}_{n+1}(X_{n+1})$ with respect to the oracle prediction $f_n^{(X_{n+1},Y_{n+1})}(X_{n+1})$ of Theorem 4.1. In what follows, let $\mathrm{polylog}(n)$ be a given sequence that grows polynomially logarithmically in $n$ .

C3) The conformity scores $|Y_i - f_n^{(X_{n+1}, Y_{n+1})}(X_i)|, \forall i \in [n+1]$ , are almost surely distinct.   
C4) The number of constant segments for $f_{n}^{(X_{n + 1},Y_{n + 1})}$ is at most $n^{1 / 3}$ polylog(n).

Theorem 4.2 (Self-calibration of prediction interval). Under C1, it holds almost surely that

$$
\mathbb {P} \left(Y _ {n + 1} \in \widehat {C} _ {n + 1} (X _ {n + 1}) \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right) \geq 1 - \alpha .
$$

If also C3 and C4 hold, then $\mathbb{E}\left|\alpha-\mathbb{P}\left(Y_{n+1}\notin\widehat{C}_{n+1}(X_{n+1})\mid f_{n}^{(X_{n+1},Y_{n+1})}(X_{n+1})\right)\right|\leq\frac{\text{polylog}(n)}{n^{2/3}}.$

Theorem 4.2 says that $\widehat{C}_{n + 1}(X_{n + 1})$ satisfies desideratum (ii) with coverage that is, on average, nearly exact up to a factor $\frac{\mathrm{polylog}(n)}{n^{2 / 3}}$ . Notably, the deviation error from exact coverage tends to zero at a fast

dimensionless rate and, therefore, does not suffer from a “curse of dimensionality”. Condition C3 is only required to establish the upper coverage bound and is standard in CP - see, e.g., (Li et al., 2018; Gibbs et al., 2023). Although it may fail for non-continuous outcomes, this condition can be avoided by adding a small amount of noise to all outcomes (Li et al., 2018). The constant segment number of $n^{1/3}$ polylog(n) in C4 is motivated by the theoretical properties of isotonic regression; Assuming C2 and continuous differentiability of $t \mapsto E_{P}[Y \mid f(X) = t]$ , it is shown in Deng et al. (2021) that the number of observations in a given constant segment of an isotonic regression solution concentrates in probability around $n^{2/3}$ . In general, without C4, our proof establishes a miscoverage upper bound of $\frac{1}{n+1}E[N_{n+1}]$ , where $E[N_{n+1}]$ is the expected number of constant segments of $f_{n}^{(X_{n+1}, Y_{n+1})}$ .

The next theorem examines the interaction between calibration and CP within SC-CP in terms of efficiency of the self-calibrating conformity scores. In the following, let $x \in \mathcal{X}, y \in \mathcal{Y}$ be arbitrary. For each $\theta \in \Theta_{iso}$ , define the $\theta$ -transformed conformity scoring function $S_{\theta}: (x', y') \mapsto |y' - \theta \circ f(x')|$ . Let $\theta_0 := \argmin_{\theta \in \Theta_{iso}} \int \{S_{\theta}(x', y')\}^2 dP(x', y')$ be the optimal isotonic transformation of $f(\cdot)$ that minimizes the population mean-square error. Define the self-calibrating conformity scoring function as $S_n^{(x,y)}(x', y') := |y' - f_n^{(x,y)}(x')|$ , where $f_n^{(x,y)}$ is obtained as in Alg. 1.

C5) Independent data: $\{(X_{i}, Y_{i})\}_{i=1}^{n+1}$ are iid.   
C6) Bounded outcomes: Y is a uniformly bounded set.

Theorem 4.3. Under C5 and C6, we have $\int\{S_{n}^{(x,y)}(x',y')-S_{\theta_{0}}(x',y')\}^{2}dP(x',y')=O_{p}(n^{-2/3})$ .

The above theorem indicates that the self-calibrating scoring function $S_{n}^{(x,y)}$ used in Alg. 2 asymptotically converges in mean-square error to the oracle scoring function $S_{\theta_{0}}$ at a rate of $n^{-2/3}$ . Since the oracle scoring function $S_{\theta_{0}}$ corresponds to a model $\theta_{0} \circ f$ with better mean square error than f, we heuristically expect that the Venn-Abers scoring function $S_{n}^{(x,y)}$ will translate to narrower CP intervals, at least asymptotically. We provide experimental evidence for this heuristic in Section 5.

Limitations. The perfectly calibrated prediction $f_{n}^{(X_{n+1},Y_{n+1})}(X_{n+1})$ , guaranteed to lie by Theorem 4.1 in the Venn-Abers multi-prediction, typically cannot be determined precisely without knowledge of $Y_{n+1}$ . However, the stability of isotonic regression implies that the width of multi-prediction $f_{n+1}(X_{n+1})$ shrinks towards zero very quickly as the size of the calibration set increases (Caponnetto and Rakhlin, 2006). Moreover, the large-sample theory for isotonic calibration in van der Laan et al. (2023) demonstrates that the $\ell^{2}$ -calibration error of each model $f_{n}^{(X_{n+1},y)}$ with $y \in Y$ is $O_{p}(n^{-2/3})$ . One caveat of SC-CP intervals is that desideratum (ii) is satisfied with respect to the unknown, oracle point prediction $f_{n}^{(X_{n+1},Y_{n+1})}(X_{n+1})$ . However, we know that this oracle prediction lies within the Venn-Abers multi-prediction by Theorem 4.1, and its value can be determined with high precision with relatively small calibration sets (Vovk and Petej, 2012) — see, e.g., Figure 1. These limitations appear to be unavoidable as perfectly calibrated point predictions can generally not be constructed in finite samples without oracle knowledge (Vovk et al., 2003; Vovk and Petej, 2012).

# 4.1 Related work

The work of Nouretdinov et al. (2018) proposes a regression extension of Venn-Abers calibration that differs from ours, both algorithmically and in its objective. While our extension constructs a calibrated point prediction $f(X)$ of $Y$ such that $f(X) = \mathbb{E}[Y \mid f(X)]$ , their approach uses the original Venn-Abers calibration procedure of Vovk and Petej (2012) to construct a distributional prediction $f_{t}(X)$ of $1(Y \leq t)$ that satisfies $f_{t}(X) = \mathbb{P}(Y \leq t \mid f_{t}(X))$ for $t \in \mathcal{Y}$ .

The impossibility results of Gupta et al. (2020) imply that any universal procedure providing prediction-conditionally calibrated intervals must explicitly or implicitly discretize the output of the model $f(\cdot)$ . The works of Johansson et al. (2014) and Johansson et al. (2018) apply Mondrian CP (Vovk et al., 2005) within leaves of a regression tree f to construct prediction intervals with prediction-conditional validity. However, this approach is restricted to tree-based predictors and does not guarantee calibrated point predictions and self-calibrated intervals. Mondrian conformal predictive distributions were applied within bins of model predictions in Boström et al. (2021) to satisfy a coarser, distributional form of prediction-conditional validity. A limitation of Mondrian-CP approaches to prediction-conditional validity is that they require pre-specification of a binning scheme

for the predictor $f(\cdot)$ , which introduces a trade-off between model performance and the width of prediction intervals, and they do not perform point calibration (desideratum (i)) and, thereby, do not guarantee self-calibration. In contrast, SC-CP data-adaptively discretizes the predictor $f(\cdot)$ using isotonic calibration and, in doing so, provides calibrated predictions, improved conformity scores, and self-calibrated intervals.

Other notions of conditional validity have been proposed that, like prediction-conditional validity and self-calibration, avoid the curse of dimensionality of context-conditional validity. In the multiclassification setup, Shi et al. (2013) and Ding et al. (2023) use Mondrian CP to provide prediction intervals with valid coverage conditional on the class label (i.e., outcome). In Boström and Johansson (2020), Mondrian CP is applied within bins categorized by context-specific difficulty estimates, such as conditional variance estimates. Multivalid-CP (Jung et al., 2022; Bastani et al., 2022) offers coverage based on a threshold defining the prediction interval. For multiclassification, Noarov et al. (2023) propose a procedure for attaining valid coverage conditional on the prediction set size (Angelopoulos et al., 2020).

# 5 Real-Data Experiments: predicting utilization of medical services

# 5.1 Experimental setup

In this experiment, we illustrate how prediction-conditional validity can approximate context-conditional validity when the heteroscedasticity in outcomes is strongly associated with model predictions, thereby ensuring validity across critical subgroups without their pre-specification. We analyze the Medical Expenditure Panel Survey (MEPS) dataset (abc), supplied by the Agency for Healthcare Research and Quality (Cohen et al., 2009), which was used in Romano et al. (2020) for Mondrian CP with fairness applications. We use the preprocessed dataset acquired using the Python package cqr, also associated with (Romano et al., 2020). This dataset contains n = 15, 656 observations and d = 139 features, and includes information such as age, marital status, race, and poverty status, alongside medical service utilization. Our objective is to predict each individual's healthcare system utilization, represented by a score that reflects visits to doctors' offices, hospital visits, etc. Following (Romano et al., 2020), we designate race as the sensitive attribute A, aiming for equalized coverage, where A = 0 represents non-white individuals ( $n_{0} = 9640$ ) and A = 1 represents white individuals ( $n_{1} = 6016$ ). The outcome variable Y is transformed by $Y = \log(1 + \text{utilization score})$ to address the skewness of the raw score. In Appendix B, we present additional experimental results for the Concrete, Community, STAR, Bike, and Bio datasets used in Romano et al. (2019) and publicly available in the Python package cqr, associated with Romano et al. (2019) and (Romano et al., 2020).

We randomly partition the dataset into three segments: a training set (50%) for model training, a calibration set (30%) for CP, and a test set (20%) for evaluation. For training the initial model $f(\cdot)$ , we use the xgboost (Chen and Guestrin, 2016) implementation of gradient boosted regression trees (Freund and Schapire, 1997), where maximum tree depth, boosting rounds, and learning rate are tuned using 5-fold cross-validation. We consider two settings for training the model. In Setting A, we train the initial model on the untransformed outcomes and then transform the predictions as $\hat{y} \mapsto \log(1 + \hat{y})$ , which makes the model predictive but poorly calibrated because it overestimates the true outcomes, in light of Jensen's inequality. In Setting B, we train the initial model on the transformed outcomes, leading to fairly well-calibrated predictions. In both settings, calibration and evaluation are applied to the transformed outcomes.

For direct comparison, we compare SC-CP with baselines that leverage the standard absolute residual scoring function $S(x,y):=|y-f(x)|$ and target either marginal validity or prediction-conditional validity. The baselines are: Marginal CP Lei et al. (2018), Mondrian CP with categories defined by bins of model predictions Vovk et al. (2005); Boström et al. (2021), CQR (Romano et al., 2019) with model predictions used as features, and the Kernel-smoothed conditional CP approach of Gibbs et al. (2023) with model predictions $\{f(X_{i})\}_{i=1}^{n}$ used as features and bandwidth tuned with cross-validation. Due to the slow computing time of the implementation provided by Gibbs et al. (2023), we apply Kernel on a subset of the calibration data of size $n_{cal}=500$ . SC-CP is implemented as described in Alg. 2, using isotonic regression constrained to have at least 20 observations averaged within each constant segment to mitigate overfitting (via the minimum child weight argument of xgboost). The miscoverage level is taken to be $\alpha=0.1$ . SC-CP provides calibrated point predictions and self-calibrated intervals, while the Mondrian and Kernel baselines

offer approximate prediction-conditional validity, and Marginal and CQR only guarantee marginal coverage. We report empirical coverage, average interval width, and calibration error of model predictions in the test set within the sensitive attribute. Calibration error is defined as the mean error of the point predictions, $E[\widehat{Y}-Y \mid A]$ within the sensitive attribute A, which measures model over- or under-confidence. For SC-CP, we use the calibrated point predictions from (4), while the original point predictions are used for Marginal, Mondrian, and Kernel. For CQR, we use an estimate of the conditional median, obtained from a separate xgboost quantile regression model, as the point prediction. We note that, since quantiles are preserved under monotone transformations of the outcome, we expect the conditional median model of CQR to be well-calibrated, at least in a median sense, in both Setting A and Setting B. We include the baseline Mondrian\* for direct comparison with SC-CP, in which Mondrian is applied with the same number of prediction bins as data-adaptively selected by SC-CP.

# 5.2 Results and discussion

![](images/3fafb9252e74e2aae8a710bb38603c90bce707444750ac9487f8259266298588.jpg)

<details>
<summary>line</summary>

| Original Prediction (uncalibrated) | Predicted Outcome (Prediction Interval) | Predicted Outcome (Venn-Abers Multi-Prediction) | Predicted Outcome (Calibrated Prediction) | Predicted Outcome (Original Prediction) |
| ---------------------------------- | ---------------------------------------- | ------------------------------------------------ | ------------------------------------------ | ----------------------------------------- |
| 1.5                                | 0.0                                      | 0.0                                              | 0.0                                        | 0.0                                       |
| 2.0                                | 1.0                                      | 1.0                                              | 1.0                                        | 1.0                                       |
| 2.5                                | 2.0                                      | 2.0                                              | 2.0                                        | 2.0                                       |
| 3.0                                | 3.0                                      | 3.0                                              | 3.0                                        | 3.0                                       |
| 3.5                                | 4.0                                      | 4.0                                              | 4.0                                        | 4.0                                       |
| 4.0                                | 5.0                                      | 5.0                                              | 5.0                                        | 5.0                                       |
| 4.5                                | 6.0                                      | 6.0                                              | 6.0                                        | 6.0                                       |
| 5.0                                | 7.0                                      | 7.0                                              | 7.0                                        | 7.0                                       |
</details>

![](images/871cebf5706bf7e08e48349316ed884f92d72102a9580c47a5b314fed7de61cd.jpg)

<details>
<summary>line</summary>

| Method | Original Prediction (uncalibrated) | Predicted Outcome |
| --- | --- | --- |
| Marginal | 1.0 | 0.5 |
| Marginal | 2.0 | 0.8 |
| Marginal | 3.0 | 1.0 |
| Marginal | 4.0 | 1.2 |
| Marginal | 5.0 | 1.5 |
| CQR (marginal) | 1.0 | 0.3 |
| CQR (marginal) | 2.0 | 0.6 |
| CQR (marginal) | 3.0 | 0.9 |
| CQR (marginal) | 4.0 | 1.1 |
| CQR (marginal) | 5.0 | 1.3 |
| Mondrian (10 bins) | 1.0 | 0.2 |
| Mondrian (10 bins) | 2.0 | 0.5 |
| Mondrian (10 bins) | 3.0 | 0.7 |
| Mondrian (10 bins) | 4.0 | 0.9 |
| Mondrian (10 bins) | 5.0 | 1.1 |
| Kernel | 1.0 | 0.4 |
| Kernel | 2.0 | 0.7 |
| Kernel | 3.0 | 0.9 |
| Kernel | 4.0 | 1.1 |
| Kernel | 5.0 | 1.3 |
| Prediction bands (Marginal) vs. CQR (marginal) vs. Mondrian (10 bins) vs. Kernel (Marginal) vs. CQR (marginal) vs. Mondrian (10 bins) vs. Kernel (Marginal) vs. CQR (marginal) vs. Kernel (Marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (marginal) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant) vs. CQR (magnant)
</details>

![](images/0c01a238f7c7bd4e4ee9e96f41eb90fb0cd4868c12232e28303d90957dffe6d4.jpg)

<details>
<summary>line</summary>

| Original Prediction (uncalibrated) | Predicted Outcome |
| ---------------------------------- | ----------------- |
| 0.0                                | 0.0               |
| 0.5                                | 1.0               |
| 1.0                                | 2.0               |
| 1.5                                | 3.0               |
| 2.0                                | 4.0               |
| 2.5                                | 4.5               |
| 3.0                                | 4.8               |
| 3.5                                | 4.9               |
| 4.0                                | 5.0               |
</details>

![](images/fe0d3c64d3c200c633addfe3fc829e0640a5863954f105e40d5fe95ed688095b.jpg)

<details>
<summary>line</summary>

| Method | Original Prediction (uncalibration) | Predicted Outcome (uncalibration) |
| --- | --- | --- |
| Marginal | 0.5 | 0.3 |
| CQR (marginal) | 1.2 | 0.8 |
| Mondrian (10 bins) | 1.8 | 1.5 |
| Kernel | 2.5 | 2.0 |
</details>

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage ( $\alpha = 0.1$ )</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td>Difference</td></tr><tr><td>Marginal</td><td>0.933</td><td>0.865</td><td>3.40</td><td>3.40</td><td>0.690</td><td>0.513</td><td>0.177</td></tr><tr><td>CQR (marginal)</td><td>0.908</td><td>0.881</td><td>2.25</td><td>2.68</td><td>-0.0952</td><td>-0.0854</td><td>-0.0098</td></tr><tr><td>Mondrian (5 bins)</td><td>0.888</td><td>0.893</td><td>3.24</td><td>3.63</td><td>0.690</td><td>0.513</td><td>0.177</td></tr><tr><td>Mondrian (10 bins)</td><td>0.913</td><td>0.886</td><td>3.21</td><td>3.54</td><td>0.690</td><td>0.513</td><td>0.177</td></tr><tr><td>Mondrian (83 bins)</td><td>0.932</td><td>0.925</td><td>3.32</td><td>3.74</td><td>0.690</td><td>0.513</td><td>0.177</td></tr><tr><td>Kernel</td><td>0.895</td><td>0.913</td><td>3.38</td><td>3.93</td><td>0.690</td><td>0.513</td><td>0.177</td></tr><tr><td>SC-CP</td><td>0.902</td><td>0.911</td><td>2.20</td><td>2.91</td><td>-0.0119</td><td>0.000931</td><td>-0.01283</td></tr></table>

(a) Setting A (poorly-calibrated $f(\cdot)$ )

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage ( $\alpha = 0.1$ )</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td>Difference</td></tr><tr><td>Marginal</td><td>0.918</td><td>0.862</td><td>2.82</td><td>2.82</td><td>-0.0136</td><td>-0.0427</td><td>0.0291</td></tr><tr><td>CQR (marginal)</td><td>0.908</td><td>0.884</td><td>2.28</td><td>2.71</td><td>-0.108</td><td>-0.0862</td><td>-0.0218</td></tr><tr><td>Mondrian (5 bins)</td><td>0.889</td><td>0.885</td><td>2.26</td><td>2.79</td><td>-0.0136</td><td>-0.0427</td><td>0.0291</td></tr><tr><td>Mondrian (10 bins)</td><td>0.893</td><td>0.896</td><td>2.22</td><td>2.87</td><td>-0.0136</td><td>-0.0427</td><td>0.0291</td></tr><tr><td>Mondrian (101 bins)</td><td>0.910</td><td>0.918</td><td>2.56</td><td>3.26</td><td>-0.0136</td><td>-0.0427</td><td>0.0291</td></tr><tr><td>Kernel</td><td>0.893</td><td>0.901</td><td>2.26</td><td>2.94</td><td>-0.0136</td><td>-0.0427</td><td>0.0291</td></tr><tr><td>SC-CP</td><td>0.909</td><td>0.924</td><td>2.14</td><td>2.86</td><td>-0.0275</td><td>0.0231</td><td>-0.0506</td></tr></table>

(b) Setting B (well-calibrated $f(\cdot)$ )   
Figure 2: MEPS-21 dataset: Calibration plot for SC-CP, prediction bands for SC-CP and baselines, and empirical coverage, width, and calibration error within sensitive subgroup.

The experimental results for each setting are depicted in Figure 2. Each panel's left-most plot showcases a calibration plot (Vuk and Curk, 2006) for SC-CP, illustrating original and calibrated predictions alongside prediction bands. On the right, the panels display prediction bands of our baselines as a function of the original model predictions. Visually, as expected by Theorem 4.2, the SC-CP bands adapt to outcome heteroscedasticity within model predictions, while Marginal lacks adaptation, Mondrian under-adapts due to insufficient bins, and Kernel adapts but offers wider intervals for large predictions where observations are sparse. The bands of CQR appear adaptive and similar to those of SC-CP, however, do not gaurnatee finite-sample prediction-conditional validity. The calibration plots reveal that heteroscedasticity in outcomes is primarily driven by their mean, suggesting that prediction-conditional validity may approximate context-conditional validity. This heuristic is supported by the tables in Figure 2, which display empirical coverage, average interval width, and calibration error within the sensitive attribute $(A)$ for all methods. In Setting A, the base regression model $f(\cdot)$ are poorly calibrated, i.e., $\mathbb{E}[f(X) - Y|A]$ is not close to 0, resulting in wider intervals, overconfidence in point predictions, and decreased interpretability for the baselines, as their intervals center around biased point predictions. In contrast, being self-calibrated, SC-CP corrects the calibration error in $f$ , achieving the smallest interval widths and well-calibrated point predictions in both settings, as guaranteed by Theorem 4.1. In both settings, the quantile regression model of CQR appears to have worse calibration than SC-CP, which may be due to the median differing from the mean because of the skewness of the outcomes. Additionally, SC-CP predictions achieve a smaller difference in calibration error between the two subgroups than Marginal, Mondrian, and Kernel, suggesting they are less discriminatory and more fair (Romano et al., 2020). SC-CP and Kernel achieve the desired coverage level of $1 - \alpha = 0.9$ in each subgroup and setting, whereas Marginal exhibits over- or under-coverage in each subgroup. Mondrian tends to under-cover with 5 and 10 bins and only attains good coverage when using the same binning number data-adaptively selected by SC-CP, highlighting its sensitivity to the pre-specified binning scheme. CQR attains good coverage in the $A = 0$ group but undercovers in the $A = 1$ group, which may be explained by CQR only guaranteeing marginal coverage in finite samples. Even with SC-CP having higher

coverage, the intervals of SC-CP are narrower than those of Kernel and Mondrian\*. This provides experimental evidence that calibration improves conformity scores and translates into greater interval efficiency, as suggested by Theorem 4.3.

# 6 Extensions

Our theoretical techniques can be used to analyze conformal prediction methods that involve the calibration of model predictions followed by the construction of conditionally valid prediction intervals. Our analysis can be adapted to the general case where either the conformity score or the conditioning variable depends on the calibrated model prediction. While we use the absolute residual conformity score in our work, SC-CP can be applied to other conformity scores, such as the normalized absolute residual scoring function (Papadopoulos et al., 2008), allowing for the inclusion of context-specific difficulty estimates in the SC-CP procedure. Although we use Venn-Abers calibration in SC-CP, our analysis also applies to other binning calibration methods, such as Venn calibration (Vovk et al., 2003; Vovk and Petej, 2012). Thus, we can replace the isotonic calibration step in Alg. 1 and 2, for example, with histogram binning (Gupta et al., 2020). Additionally, a group-valid form of SC-CP can be achieved by applying Alg. 2 separately within subgroups, similar to Multivalid CP (Jung et al., 2022). Interesting areas for future work involve integrating point calibration with conformal prediction methods for predictive models beyond regression, such as the isotonic calibration of quantile predictions in conformal quantile regression (Romano et al., 2019).

# References

abc. Medical expenditure panel survey, panel 21, 2021. URL https://meps.ahrq.gov/mepsweb/data\_stats/download\_data\_files\_detail.jsp?cboPufNumber=HC-192. Accessed: May, 2024.   
A. Angelopoulos, S. Bates, J. Malik, and M. I. Jordan. Uncertainty sets for image classifiers using conformal prediction. arXiv preprint arXiv:2009.14193, 2020.   
V. Balasubramanian, S.-S. Ho, and V. Vovk. Conformal prediction for reliable machine learning: theory, adaptations and applications. Newnes, 2014.   
R. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani. The limits of distribution-free conditional predictive inference. Information and Inference: A Journal of the IMA, 10(2):455–482, 2021.   
R. E. Barlow and H. D. Brunk. The isotonic regression problem and its dual. Journal of the American Statistical Association, 67(337):140–147, 1972.   
O. Bastani, V. Gupta, C. Jung, G. Noarov, R. Ramalingam, and A. Roth. Practical adversarial multivalid conformal prediction. Advances in Neural Information Processing Systems, 35:29362–29373, 2022.   
A. Bella, C. Ferri, J. Hernández-Orallo, and M. J. Ramírez-Quintana. Calibration of machine learning models. In Handbook of Research on Machine Learning Applications and Trends: Algorithms, Methods, and Techniques, pages 128–146. IGI Global, 2010.   
H. Boström and U. Johansson. Mondrian conformal regressors. In Conformal and Probabilistic Prediction and Applications, pages 114–133. PMLR, 2020.   
H. Boström, U. Johansson, and T. Löfström. Mondrian conformal predictive distributions. In Conformal and Probabilistic Prediction and Applications, pages 24–38. PMLR, 2021.   
A. Caponnetto and A. Rakhlin. Stability properties of empirical risk minimization over donsker classes. Journal of Machine Learning Research, 7(12), 2006.   
D. S. Char, N. H. Shah, and D. Magnus. Implementing machine learning in health care—addressing ethical challenges. The New England journal of medicine, 378(11):981, 2018.   
T. Chen and C. Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, pages 785–794, New York, NY, USA, 2016. ACM. ISBN 978-1-4503-4232-2. doi: 10.1145/2939672.2939785. URL http://doi.acm.org/10.1145/2939672.2939785.   
J. W. Cohen, S. B. Cohen, and J. S. Banthin. The medical expenditure panel survey: a national information resource to support healthcare cost research and inform policy and practice. Medical care, 47(7\_Supplement\_1):S44–S50, 2009.   
D. R. Cox. Two further applications of a model for binary regression. Biometrika, 45(3/4):562–565, 1958.   
S. E. Davis, T. A. Lasko, G. Chen, E. D. Siew, and M. E. Matheny. Calibration drift in regression and machine learning models for acute kidney injury. Journal of the American Medical Informatics Association, 24(6):1052–1061, 2017.   
H. Deng, Q. Han, and C.-H. Zhang. Confidence intervals for multiple isotonic regression and other monotone models. The Annals of Statistics, 49(4):2021–2052, 2021.   
Z. Deng, C. Dwork, and L. Zhang. Happymap: A generalized multi-calibration method. arXiv preprint arXiv:2303.04379, 2023.   
T. Ding, A. N. Angelopoulos, S. Bates, M. I. Jordan, and R. J. Tibshirani. Class-conditional conformal prediction with many classes. arXiv preprint arXiv:2306.09335, 2023.   
B. Flury and T. Tarpey. Self-consistency: A fundamental concept in statistics. Statistical Science, 11(3):229–243, 1996.

Y. Freund and R. E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997. ISSN 0022-0000. doi: https://doi.org/10.1006/jcss.1997.1504. URL https://www.sciencedirect.com/science/article/pii/S002200009791504X.   
I. Gibbs, J. J. Cherian, and E. J. Candès. Conformal prediction with conditional guarantees. arXiv preprint arXiv:2305.12616, 2023.   
T. Gneiting, F. Balabdaoui, and A. E. Raftery. Probabilistic forecasts, calibration and sharpness. Journal of the Royal Statistical Society Series B: Statistical Methodology, 69(2):243–268, 2007.   
L. Guan. Localized conformal prediction: A generalized inference framework for conformal prediction. Biometrika, 110(1):33–50, 2023.   
C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.   
C. Gupta and A. Ramdas. Distribution-free calibration guarantees for histogram binning without sample splitting. In International Conference on Machine Learning, pages 3942–3952. PMLR, 2021.   
C. Gupta, A. Podkopaev, and A. Ramdas. Distribution-free binary classification: prediction sets, confidence intervals and calibration. Advances in Neural Information Processing Systems, 33:3711–3723, 2020.   
T. Hastie and R. Tibshirani. Generalized additive models: some applications. Journal of the American Statistical Association, 82(398):371–386, 1987.   
T. Heskes. Practical confidence and prediction intervals. Advances in neural information processing systems, 9, 1996.   
U. Johansson, C. Sönströd, H. Linusson, and H. Boström. Regression trees for streaming data with local performance guarantees. In 2014 IEEE International Conference on Big Data (Big Data), pages 461–470. IEEE, 2014.   
U. Johansson, H. Linusson, T. Löfström, and H. Boström. Interpretable regression trees using conformal prediction. Expert systems with applications, 97:394–404, 2018.   
U. Johansson, T. Löfström, and C. Sönströd. Well-calibrated probabilistic predictive maintenance using venn-abers. arXiv preprint arXiv:2306.06642, 2023.   
C. Jung, G. Noarov, R. Ramalingam, and A. Roth. Batch multivalid conformal prediction. arXiv preprint arXiv:2209.15145, 2022.   
C.-I. C. Lee. The min-max algorithm and isotonic regression. The Annals of Statistics, 11(2):467–477, 1983.   
J. Lei and L. Wasserman. Distribution-free prediction bands for non-parametric regression. Journal of the Royal Statistical Society Series B: Statistical Methodology, 76(1):71–96, 2014.   
J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman. Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523):1094–1111, 2018.   
Y. Li, L. Qi, and Y. Sun. Semiparametric varying-coefficient regression analysis of recurrent events with applications to treatment switching. Statistics in Medicine, 37:3959–3974, 2018. doi:10.1002/sim.7856. PubMed PMID: 29992591. NIHMSID: NIHMS1033642.   
S. Lichtenstein, B. Fischhoff, and L. D. Phillips. Calibration of probabilities: The state of the art. In Decision Making and Change in Human Affairs: Proceedings of the Fifth Research Conference on Subjective Probability, Utility, and Decision Making, Darmstadt, 1–4 September, 1975, pages 275–324. Springer, 1977.   
E. B. Mandinach, M. Honey, and D. Light. A theoretical framework for data-driven decision making. In annual meeting of the American Educational Research Association, San Francisco, CA, 2006.

J. A. Mincer and V. Zarnowitz. The evaluation of economic forecasts. In Economic forecasts and expectations: Analysis of forecasting behavior and performance, pages 3–46. NBER, 1969.   
A. Niculescu-Mizil and R. Caruana. Obtaining calibrated probabilities from boosting. In UAI, volume 5, pages 413–20, 2005.   
G. Noarov, R. Ramalingam, A. Roth, and S. Xie. High-dimensional prediction for sequential decision making. arXiv preprint arXiv:2310.17651, 2023.   
I. Nouretdinov, D. Volkhonskiy, P. Lim, P. Toccaceli, and A. Gammerman. Inductive venn-abers predictive distribution. In Conformal and Probabilistic Prediction and Applications, pages 15–36. PMLR, 2018.   
H. Papadopoulos, A. Gammerman, and V. Vovk. Normalized nonconformity measures for regression conformal prediction. In Proceedings of the IASTED International Conference on Artificial Intelligence and Applications (AIA 2008), pages 64–69, 2008.   
J. Patel. Prediction intervals-a review. Communications in Statistics-Theory and Methods, 18(7):2393–2465, 1989.   
J. Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61–74, 1999.   
L. Quest, A. Charrie, L. du Croo de Jongh, and S. Roy. The risks and benefits of using ai to detect crime. Harv. Bus. Rev. Digit. Artic, 8:2–5, 2018.   
Y. Romano, E. Patterson, and E. Candes. Conformalized quantile regression. Advances in neural information processing systems, 32, 2019.   
Y. Romano, R. F. Barber, C. Sabatti, and E. Candès. With malice toward none: Assessing uncertainty via equalized coverage. Harvard Data Science Review, 2(2):4, 2020.   
G. Shafer and V. Vovk. A tutorial on conformal prediction. Journal of Machine Learning Research, 9(3), 2008.   
F. Shi, C. S. Ong, and C. Leckie. Applications of class-conditional conformal predictor in multi-class classification. In 2013 12th International Conference on Machine Learning and Applications, volume 1, pages 235–239. IEEE, 2013.   
L. van der Laan, E. Ulloa-Pérez, M. Carone, and A. Luedtke. Causal isotonic calibration for heterogeneous treatment effects. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202, Honolulu, Hawaii, USA, 2023. PMLR.   
L. van der Laan, E. Ulloa-Pérez, M. Carone, and A. Luedtke. Causal isotonic calibration for heterogeneous treatment effects. arXiv preprint arXiv:2302.14011, 2023.   
A. van der Vaart and J. Wellner. Weak Convergence and Empirical Processes. Springer, New York, 1996.   
A. Van Der Vaart and J. A. Wellner. A local maximal inequality under uniform entropy. Electronic Journal of Statistics, 5(2011):192, 2011.   
V. Vapnik, E. Levin, and Y. Le Cun. Measuring the vc-dimension of a learning machine. Neural computation, 6(5):851–876, 1994.   
J. Vazquez and J. C. Facelli. Conformal prediction in clinical medical sciences. Journal of Healthcare Informatics Research, 6(3):241–252, 2022.   
M. Veale, M. Van Kleek, and R. Binns. Fairness and accountability design needs for algorithmic support in high-stakes public sector decision-making. In Proceedings of the 2018 chi conference on human factors in computing systems, pages 1–14, 2018.   
V. Vovk. Conditional validity of inductive conformal predictors. In Asian conference on machine learning, pages 475–490. PMLR, 2012.

V. Vovk and I. Petej. Venn-abers predictors. arXiv preprint arXiv:1211.0025, 2012.   
V. Vovk, G. Shafer, and I. Nouretdinov. Self-calibrating probability forecasting. Advances in neural information processing systems, 16, 2003.   
V. Vovk, A. Gammerman, and G. Shafer. Algorithmic learning in a random world, volume 29. Springer, 2005.   
M. Vuk and T. Curk. Roc curve, lift chart and calibration plot. Metodoloski zvezki, 3(1):89, 2006.   
M. N. Wright and A. Ziegler. ranger: A fast implementation of random forests for high dimensional data in C++ and R. Journal of Statistical Software, 77(1):1–17, 2017. doi: 10.18637/jss.v077.i01.   
Y. Xu and S. Yadlowsky. Calibration error for heterogeneous treatment effects. In International Conference on Artificial Intelligence and Statistics, pages 9280–9303. PMLR, 2022.   
B. Zadrozny and C. Elkan. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In IcmI, volume 1, pages 609–616. Citeseer, 2001.   
B. Zadrozny and C. Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 694–699, 2002.

# A Code

The methods implemented in this paper are not computationally intensive and were run in a Jupyter notebook environment on a MacBook Pro with 16GB RAM and an M1 chip. A Python implementation of SC-CP is provided in the package SelfCalibratingConformal, available via pip. Code implementing SC-CP and reproducing our experiments is available in the GitHub repository SelfCalibratingConformal, which can be accessed at the following link: https://github.com/Larsvanderlaan/SelfCalibratingConformal.

# B Supplementary real data experiments

# B.1 Additional results

In this section, we present the experimental results for the concrete, STAR, bike, community, and bio datasets used in Romano et al. (2019) and publicly available in the Python package cqr, associated with Romano et al. (2019) and (Romano et al., 2020). For the STAR dataset, the sensitive attribute A was set to “gender.” For the Bike dataset, the sensitive attribute A was set to “workingday,” and for Community, it was set to “race\_binary.” For the remaining datasets, the sensitive attribute A was set to a dichotomization of the final column in the feature matrix, as above or below its median value.

![](images/2391da649e5f252b6a8d1786e204f73ed85e46e1c236ed65f95d3b62e3a78ca7.jpg)

<details>
<summary>line</summary>

| Original Prediction | Predicted Outcome |
| ------------------- | ----------------- |
| 8.34                | 8.35              |
| 8.36                | 8.37              |
| 8.38                | 8.39              |
| 8.40                | 8.41              |
| 8.42                | 8.43              |
| 8.44                | 8.45              |
| 8.46                | 8.47              |
</details>

![](images/8d23f1f1dc65a03162b29bcc03a9c528d23464e401939a61adb0fdbc8578fa0f.jpg)

<details>
<summary>line</summary>

| Method | Prediction Bands (Uncalibrated) |
| --- | --- |
| Marginal | 0.50 |
| CQR (marginal) | 0.45 |
| Mondrian (10 bins) | 0.35 |
| Kernel | 0.25 |
</details>

![](images/41d4233a20d6fdc7af6ed4c217ded3a70f30e9f158c9107c92e47d324e41a923.jpg)

<details>
<summary>line</summary>

| Original Prediction | Predicted Outcome |
| ------------------- | ----------------- |
| 8.34                | 8.25              |
| 8.36                | 8.30              |
| 8.38                | 8.35              |
| 8.40                | 8.40              |
| 8.42                | 8.45              |
| 8.44                | 8.47              |
| 8.46                | 8.48              |
</details>

![](images/e2d93e685116c6094771393c5435a26da2fa7a29ba1f8bc1308393de62663ec6.jpg)

<details>
<summary>line</summary>

| Model | Prediction Bands | Predicted Outcome |
|-------|------------------|-------------------|
| Marginal | 0.30 | 0.35 |
| Marginal | 0.40 | 0.40 |
| Marginal | 0.50 | 0.45 |
| Marginal | 0.60 | 0.50 |
| Marginal | 0.70 | 0.55 |
| Marginal | 0.80 | 0.60 |
| Marginal | 0.90 | 0.65 |
| Marginal | 1.00 | 0.70 |
| CQR (marginal) | 0.30 | 0.35 |
| CQR (marginal) | 0.40 | 0.40 |
| CQR (marginal) | 0.50 | 0.45 |
| CQR (marginal) | 0.60 | 0.50 |
| CQR (marginal) | 0.70 | 0.55 |
| CQR (marginal) | 0.80 | 0.60 |
| CQR (marginal) | 0.90 | 0.65 |
| CQR (marginal) | 1.00 | 0.70 |
| Mondrian (10 bins) | 0.30 | 0.35 |
| Mondrian (10 bins) | 0.40 | 0.40 |
| Mondrian (10 bins) | 0.50 | 0.45 |
| Mondrian (10 bins) | 0.60 | 0.50 |
| Mondrian (10 bins) | 0.70 | 0.55 |
| Mondrian (10 bins) | 0.80 | 0.60 |
| Mondrian (10 bins) | 0.90 | 0.65 |
| Mondrian (10 bins) | 1.00 | 0.70 |
| Kernel | 0.30 | 0.35 |
| Kernel | 0.40 | 0.40 |
| Kernel | 0.50 | 0.45 |
| Kernel | 0.60 | 0.50 |
| Kernel | 0.70 | 0.55 |
| Kernel | 0.80 | 0.60 |
| Kernel | 0.90 | 0.65 |
| Kernel | 1.00 | 0.70 |
</details>

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage ( $\alpha = 0.1$ )</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td>Difference</td></tr><tr><td>Marginal</td><td>0.892</td><td>0.896</td><td>0.174</td><td>0.174</td><td>0.00482</td><td>0.00961</td><td>-0.00479</td></tr><tr><td>CQR (marginal)</td><td>0.892</td><td>0.872</td><td>0.173</td><td>0.177</td><td>0.00302</td><td>0.00597</td><td>-0.00295</td></tr><tr><td>Mondrian (5 bins)</td><td>0.887</td><td>0.882</td><td>0.171</td><td>0.171</td><td>0.00482</td><td>0.00961</td><td>-0.00479</td></tr><tr><td>Mondrian (10 bins)</td><td>0.901</td><td>0.886</td><td>0.176</td><td>0.173</td><td>0.00482</td><td>0.00961</td><td>-0.00479</td></tr><tr><td>Mondrian (99 bins)</td><td>0.860</td><td>0.872</td><td>0.184</td><td>0.179</td><td>0.00482</td><td>0.00961</td><td>-0.00479</td></tr><tr><td>Kernel</td><td>0.887</td><td>0.891</td><td>0.170</td><td>0.170</td><td>0.00482</td><td>0.00961</td><td>-0.00479</td></tr><tr><td>SC-CP</td><td>0.905</td><td>0.919</td><td>0.180</td><td>0.178</td><td>0.00607</td><td>0.00983</td><td>-0.00376</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage ( $\alpha = 0.1$ )</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td>Difference</td></tr><tr><td>Marginal</td><td>0.896</td><td>0.896</td><td>0.175</td><td>0.175</td><td>0.00338</td><td>0.00796</td><td>-0.00458</td></tr><tr><td>CQR (marginal)</td><td>0.896</td><td>0.910</td><td>0.180</td><td>0.182</td><td>0.00292</td><td>0.00454</td><td>-0.00162</td></tr><tr><td>Mondrian (5 bins)</td><td>0.892</td><td>0.900</td><td>0.174</td><td>0.174</td><td>0.00338</td><td>0.00796</td><td>-0.00458</td></tr><tr><td>Mondrian (10 bins)</td><td>0.892</td><td>0.891</td><td>0.178</td><td>0.176</td><td>0.00338</td><td>0.00796</td><td>-0.00458</td></tr><tr><td>Mondrian (98 bins)</td><td>0.860</td><td>0.863</td><td>0.176</td><td>0.175</td><td>0.00338</td><td>0.00796</td><td>-0.00458</td></tr><tr><td>Kernel</td><td>0.892</td><td>0.900</td><td>0.171</td><td>0.171</td><td>0.00338</td><td>0.00796</td><td>-0.00458</td></tr><tr><td>SC-CP</td><td>0.892</td><td>0.905</td><td>0.177</td><td>0.177</td><td>0.00624</td><td>0.01000</td><td>-0.00376</td></tr></table>

(a) Setting A (poorly-calibrated $f(\cdot)$ )   
(b) Setting B (well-calibrated $f(\cdot)$ )   
Figure 3: STAR dataset: Calibration plot for SC-CP, prediction bands for SC-CP and baselines, and empirical coverage, width, and calibration error within sensitive subgroup.

![](images/b8043f36841a72a64c489521287695a4e97602d83e788f9a28b06e8c59f1d353.jpg)

<details>
<summary>line</summary>

| Original Prediction (uncalibrated) | Predicted Outcome |
| ---------------------------------- | ----------------- |
| 0                                  | 0                 |
| 1                                  | 1                 |
| 2                                  | 2                 |
| 3                                  | 3                 |
| 4                                  | 4                 |
| 5                                  | 5                 |
| 6                                  | 6                 |
| 7                                  | 7                 |
</details>

![](images/18c35be2bc96030b0aee5b881021974b5da80d30ed618a234e8bf7f403c49c75.jpg)

<details>
<summary>line</summary>

| Method          | Original Prediction (uncalibrated) | Predicted Outcome |
| --------------- | ----------------------------------- | ----------------- |
| Marginal        | 0                                   | 0                 |
| CQR (marginal)  | 0                                   | 0                 |
| Mondrian (10 bins) | 0                                 | 0                 |
| Kernel          | 0                                   | 0                 |
</details>

![](images/653297ab45e935e0e656b1be51c83c4647250e5ccd863fa7cabc8c1be63b30a8.jpg)

<details>
<summary>line</summary>

| Original Prediction (uncalibrated) | Predicted Outcome |
| ---------------------------------- | ----------------- |
| 1                                  | 1.0               |
| 2                                  | 2.0               |
| 3                                  | 3.0               |
| 4                                  | 4.0               |
| 5                                  | 5.0               |
| 6                                  | 6.0               |
| 7                                  | 7.0               |
</details>

![](images/6d52865bccc65a48adc5b4bed5a6524bff9d23a33e02c7076377f3ff4b03c5b5.jpg)

<details>
<summary>line</summary>

| Model Type | Original Prediction (uncalibrated) | Predicted Outcome |
|------------|------------------------------------|-------------------|
| Marginal   | 0                                  | 0                 |
| CQR (marginal) | 0                                | 0                 |
| Mondrian (10 bins) | 0                          | 0                 |
| Kernel     | 0                                  | 0                 |
</details>

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage (α = 0.1)</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td>A=0</td><td>A=1</td><td>A=0</td><td>A=1</td><td>A=0</td><td>A=1</td><td>Difference</td></tr><tr><td>Marginal</td><td>0.909</td><td>0.914</td><td>1.46</td><td>1.46</td><td>0.155</td><td>0.147</td><td>0.008</td></tr><tr><td>CQR (marginal)</td><td>0.884</td><td>0.894</td><td>1.31</td><td>1.10</td><td>0.055</td><td>0.0238</td><td>0.0312</td></tr><tr><td>Mondrian (5 bins)</td><td>0.878</td><td>0.923</td><td>1.20</td><td>1.17</td><td>0.155</td><td>0.147</td><td>0.008</td></tr><tr><td>Mondrian (10 bins)</td><td>0.863</td><td>0.927</td><td>1.15</td><td>1.15</td><td>0.155</td><td>0.147</td><td>0.008</td></tr><tr><td>Mondrian (101 bins)</td><td>0.872</td><td>0.935</td><td>1.23</td><td>1.22</td><td>0.155</td><td>0.147</td><td>0.008</td></tr><tr><td>Kernel</td><td>0.865</td><td>0.927</td><td>1.12</td><td>1.16</td><td>0.155</td><td>0.147</td><td>0.008</td></tr><tr><td>SC-CP</td><td>0.863</td><td>0.933</td><td>0.966</td><td>0.935</td><td>0.00833</td><td>-0.01100</td><td>0.01933</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage ( $\alpha = 0.1$ )</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td>Difference</td></tr><tr><td>Marginal</td><td>0.885</td><td>0.923</td><td>1.03</td><td>1.03</td><td>0.0215</td><td>0.00754</td><td>0.01396</td></tr><tr><td>CQR (marginal)</td><td>0.885</td><td>0.900</td><td>1.31</td><td>1.16</td><td>0.0328</td><td>0.0292</td><td>0.0036</td></tr><tr><td>Mondrian (5 bins)</td><td>0.872</td><td>0.925</td><td>0.992</td><td>0.970</td><td>0.0215</td><td>0.00754</td><td>0.01396</td></tr><tr><td>Mondrian (10 bins)</td><td>0.863</td><td>0.929</td><td>0.978</td><td>0.955</td><td>0.0215</td><td>0.00754</td><td>0.01396</td></tr><tr><td>Mondrian (101 bins)</td><td>0.883</td><td>0.937</td><td>1.12</td><td>1.07</td><td>0.0215</td><td>0.00754</td><td>0.01396</td></tr><tr><td>Kernel</td><td>0.860</td><td>0.926</td><td>0.958</td><td>0.966</td><td>0.0215</td><td>0.00754</td><td>0.01396</td></tr><tr><td>SC-CP</td><td>0.878</td><td>0.929</td><td>0.995</td><td>0.945</td><td>0.00797</td><td>-0.00647</td><td>0.01444</td></tr></table>

(a) Setting A (poorly-calibrated $f(\cdot)$ )   
(b) Setting B (well-calibrated $f(\cdot)$ )   
Figure 4: Bike dataset: Calibration plot for SC-CP, prediction bands for SC-CP and baselines, and empirical coverage, width, and calibration error within sensitive subgroup.

![](images/cceec5f651cb16238bb4770113acb52f835830f2fb4a65eb8f6cd816e54bb9d7.jpg)

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage (α = 0.1)</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td>A=0</td><td>A=1</td><td>A=0</td><td>A=1</td><td>A=0</td><td>A=1</td><td>Difference</td></tr><tr><td>Marginal</td><td>0.870</td><td>0.949</td><td>0.321</td><td>0.321</td><td>-0.00871</td><td>0.0126</td><td>-0.02131</td></tr><tr><td>CQR (marginal)</td><td>0.919</td><td>0.943</td><td>0.366</td><td>0.245</td><td>-0.0181</td><td>-0.00647</td><td>-0.01163</td></tr><tr><td>Mondrian (5 bins)</td><td>0.825</td><td>0.847</td><td>0.275</td><td>0.186</td><td>-0.00871</td><td>0.0126</td><td>-0.02131</td></tr><tr><td>Mondrian (10 bins)</td><td>0.928</td><td>0.864</td><td>0.372</td><td>0.204</td><td>-0.00871</td><td>0.0126</td><td>-0.02131</td></tr><tr><td>Mondrian (98 bins)</td><td>0.843</td><td>0.852</td><td>0.340</td><td>0.219</td><td>-0.00871</td><td>0.0126</td><td>-0.02131</td></tr><tr><td>Kernel</td><td>0.901</td><td>0.886</td><td>0.337</td><td>0.189</td><td>-0.00871</td><td>0.0126</td><td>-0.02131</td></tr><tr><td>SC-CP</td><td>0.901</td><td>0.892</td><td>0.362</td><td>0.187</td><td>-0.01090</td><td>0.00859</td><td>-0.01949</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage (α = 0.1)</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td>A=0</td><td>A=1</td><td>A=0</td><td>A=1</td><td>A=0</td><td>A=1</td><td>Difference</td></tr><tr><td>Marginal</td><td>0.861</td><td>0.949</td><td>0.321</td><td>0.321</td><td>-0.0153</td><td>0.00739</td><td>-0.02269</td></tr><tr><td>CQR (marginal)</td><td>0.888</td><td>0.915</td><td>0.333</td><td>0.227</td><td>-0.0346</td><td>-0.00193</td><td>-0.03267</td></tr><tr><td>Mondrian (5 bins)</td><td>0.852</td><td>0.875</td><td>0.293</td><td>0.188</td><td>-0.0153</td><td>0.00739</td><td>-0.02269</td></tr><tr><td>Mondrian (10 bins)</td><td>0.906</td><td>0.903</td><td>0.367</td><td>0.201</td><td>-0.0153</td><td>0.00739</td><td>-0.02269</td></tr><tr><td>Mondrian (97 bins)</td><td>0.821</td><td>0.858</td><td>0.366</td><td>0.214</td><td>-0.0153</td><td>0.00739</td><td>-0.02269</td></tr><tr><td>Kernel</td><td>0.888</td><td>0.881</td><td>0.347</td><td>0.185</td><td>-0.0153</td><td>0.00739</td><td>-0.02269</td></tr><tr><td>SC-CP</td><td>0.901</td><td>0.892</td><td>0.359</td><td>0.191</td><td>-0.00977</td><td>0.00804</td><td>-0.01781</td></tr></table>

(a) Setting A (poorly-calibrated $f(\cdot)$ )   
(b) Setting B (well-calibrated $f(\cdot)$ )   
Figure 5: Community dataset: Calibration plot for SC-CP, prediction bands for SC-CP and baselines, and empirical coverage, width, and calibration error within sensitive subgroup.

![](images/8364fa729eeb2511a4c7e2c52ea22ec83d9f300b1167a4afb42b0038acd7697a.jpg)

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage ( $\alpha = 0.1$ )</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td>Difference</td></tr><tr><td>Marginal</td><td>0.839</td><td>0.903</td><td>0.529</td><td>0.529</td><td>-0.0120</td><td>-0.00892</td><td>-0.00308</td></tr><tr><td>CQR (marginal)</td><td>0.887</td><td>0.861</td><td>0.918</td><td>0.629</td><td>-0.0143</td><td>0.00432</td><td>-0.01862</td></tr><tr><td>Mondrian (5 bins)</td><td>0.968</td><td>0.889</td><td>0.816</td><td>0.551</td><td>-0.0120</td><td>-0.00892</td><td>-0.00308</td></tr><tr><td>Mondrian (10 bins)</td><td>0.952</td><td>0.910</td><td>0.783</td><td>0.601</td><td>-0.0120</td><td>-0.00892</td><td>-0.00308</td></tr><tr><td>Mondrian (87 bins)</td><td>0.774</td><td>0.806</td><td>0.652</td><td>0.481</td><td>-0.0120</td><td>-0.00892</td><td>-0.00308</td></tr><tr><td>Kernel</td><td>0.952</td><td>0.917</td><td>0.791</td><td>0.532</td><td>-0.0120</td><td>-0.00892</td><td>-0.00308</td></tr><tr><td>SC-CP</td><td>0.887</td><td>0.861</td><td>0.879</td><td>0.563</td><td>-0.0356</td><td>-0.0462</td><td>0.0106</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage ( $\alpha = 0.1$ )</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td>Difference</td></tr><tr><td>Marginal</td><td>0.839</td><td>0.903</td><td>0.525</td><td>0.525</td><td>-0.0540</td><td>-0.0150</td><td>-0.0390</td></tr><tr><td>CQR (marginal)</td><td>0.871</td><td>0.889</td><td>0.901</td><td>0.617</td><td>-0.00706</td><td>0.0119</td><td>-0.01896</td></tr><tr><td>Mondrian (5 bins)</td><td>0.935</td><td>0.903</td><td>0.698</td><td>0.534</td><td>-0.0540</td><td>-0.0150</td><td>-0.0390</td></tr><tr><td>Mondrian (10 bins)</td><td>0.935</td><td>0.903</td><td>0.699</td><td>0.504</td><td>-0.0540</td><td>-0.0150</td><td>-0.0390</td></tr><tr><td>Mondrian (86 bins)</td><td>0.726</td><td>0.792</td><td>0.602</td><td>0.453</td><td>-0.0540</td><td>-0.0150</td><td>-0.0390</td></tr><tr><td>Kernel</td><td>0.952</td><td>0.875</td><td>0.695</td><td>0.482</td><td>-0.0540</td><td>-0.0150</td><td>-0.0390</td></tr><tr><td>SC-CP</td><td>0.935</td><td>0.903</td><td>0.859</td><td>0.545</td><td>-0.0587</td><td>-0.0265</td><td>-0.0322</td></tr></table>

(a) Setting A (poorly-calibrated $f(\cdot)$ )   
(b) Setting B (well-calibrated $f(\cdot)$ )   
Figure 6: Concrete dataset: Calibration plot for SC-CP, prediction bands for SC-CP and baselines, and empirical coverage, width, and calibration error within sensitive subgroup.

![](images/89a9ed224066335e45c4aa02d26557632d070a3a26a816e7064ba06342c6a5bd.jpg)

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage ( $\alpha = 0.1$ )</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td> $A = 0$ </td><td> $A = 1$ </td><td>Difference</td></tr><tr><td>Marginal</td><td>0.907</td><td>0.904</td><td>1.81</td><td>1.81</td><td>0.168</td><td>0.153</td><td>0.015</td></tr><tr><td>CQR (marginal)</td><td>0.895</td><td>0.897</td><td>1.64</td><td>1.73</td><td>-0.0223</td><td>-0.0062</td><td>-0.0161</td></tr><tr><td>Mondrian (5 bins)</td><td>0.908</td><td>0.915</td><td>1.73</td><td>1.84</td><td>0.168</td><td>0.153</td><td>0.015</td></tr><tr><td>Mondrian (10 bins)</td><td>0.907</td><td>0.910</td><td>1.69</td><td>1.79</td><td>0.168</td><td>0.153</td><td>0.015</td></tr><tr><td>Mondrian (101 bins)</td><td>0.903</td><td>0.914</td><td>1.62</td><td>1.79</td><td>0.168</td><td>0.153</td><td>0.015</td></tr><tr><td>Kernel</td><td>0.892</td><td>0.901</td><td>1.52</td><td>1.67</td><td>0.168</td><td>0.153</td><td>0.015</td></tr><tr><td>SC-CP</td><td>0.882</td><td>0.908</td><td>1.36</td><td>1.56</td><td>-0.0191</td><td>0.003</td><td>-0.0221</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="2">Coverage (α = 0.1)</td><td colspan="2">Average Width</td><td colspan="3">Cal. Error</td></tr><tr><td>A=0</td><td>A=1</td><td>A=0</td><td>A=1</td><td>A=0</td><td>A=1</td><td>Difference</td></tr><tr><td>Marginal</td><td>0.898</td><td>0.904</td><td>1.67</td><td>1.67</td><td>0.00227</td><td>-0.0191</td><td>0.02137</td></tr><tr><td>CQR (marginal)</td><td>0.891</td><td>0.889</td><td>1.66</td><td>1.70</td><td>-0.0209</td><td>-0.0124</td><td>-0.0085</td></tr><tr><td>Mondrian (5 bins)</td><td>0.902</td><td>0.922</td><td>1.58</td><td>1.68</td><td>0.00227</td><td>-0.0191</td><td>0.02137</td></tr><tr><td>Mondrian (10 bins)</td><td>0.893</td><td>0.923</td><td>1.49</td><td>1.59</td><td>0.00227</td><td>-0.0191</td><td>0.02137</td></tr><tr><td>Mondrian (101 bins)</td><td>0.894</td><td>0.925</td><td>1.49</td><td>1.61</td><td>0.00227</td><td>-0.0191</td><td>0.02137</td></tr><tr><td>Kernel</td><td>0.880</td><td>0.895</td><td>1.37</td><td>1.48</td><td>0.00227</td><td>-0.0191</td><td>0.02137</td></tr><tr><td>SC-CP</td><td>0.893</td><td>0.916</td><td>1.38</td><td>1.52</td><td>-0.0174</td><td>0.00773</td><td>-0.02513</td></tr></table>

(a) Setting A (poorly-calibrated $f(\cdot)$ )   
(b) Setting B (well-calibrated $f(\cdot)$ )   
Figure 7: Bio dataset: Calibration plot for SC-CP, prediction bands for SC-CP and baselines, and empirical coverage, width, and calibration error within sensitive subgroup.

# C Supplementary synthetic data experiments

# C.1 Experimental setup

In this appendix, we perform additional synthetic experiments to evaluate the prediction-conditional validity of our method and how it translates to approximate context-conditional validity in certain cases.

Synthetic datasets. We construct synthetic training, calibration, and test datasets $D_{train}$ , $D_{cal}$ , $D_{test}$ of sizes $n_{train}$ , $n_{cal}$ , $n_{test}$ , which are respectively used to train f, apply CP, and evaluate performance. For parameters $d \in N$ , $\kappa > 0$ , $a \geq 0$ , $b \geq 0$ , each dataset consists of iid observations of $(X, Y)$ drawn as follows. The covariate vector $X := (X_1, \ldots, X_d) \in [0, 1]^d$ is coordinate-wise independently drawn from a Beta(1, $\kappa$ ) distribution with shape parameter $\kappa$ . Then, conditionally on X = x, the outcome Y is drawn normally distributed with conditional mean $\mu(x) := d^{-1/2} \sum_{j=1}^{d} \{x_j + \sin(4x_j)\}$ and conditional variance $\sigma^2(x) := \{0.035 - a \log(0.5 + 0.5x_1)/8 + b \left( |\mu_0(x)|^6/20 - 0.02 \right)/2\}^2$ . Here, a and b control the heteroscedasticity and mean-variance relationship for the outcomes. For $D_{cal}$ and $D_{test}$ , we set $\kappa_{cal} = \kappa_{test} = 1$ and, for $D_{train}$ , we vary $\kappa_{train}$ to introduce distribution shift and, thereby, calibration error in f. The parameters d, a, and b are fixed across the datasets.

Baselines. To mitigate overfitting, we implement SC-CP so that each function in $\Theta_{iso}$ has at least 20 observations averaged within each constant segment (implemented using the minimum leaf node size of argument xgboost). When appropriate, we will consider the following baseline CP algorithms for comparison. Unless stated otherwise, for all baselines, we use the scoring function $S(x,y):=|y-f(x)|$ . The first baseline, uncond-CP, is split-CP (Li et al., 2018), which provides only unconditional coverage guarantees. The second baseline, cond-CP, is adapted from Gibbs et al. (2023) and provides conditional coverage over distribution shifts within a specified reproducing kernel Hilbert space. Following Section 5.1 of Gibbs et al. (2023), we use the Gaussian kernel $K(X_{i},X_{j}):=\exp\left(-4\|X_{i}-X_{j}\|_{2}^{2}\right)$ with euclidean norm $\|\cdot\|_{2}$ and select the regularization parameter $\lambda$ using 5-fold cross-validation. The third baseline, Mondrian CP, applies the Mondrian CP method (Vovk et al., 2005; Boström and Johansson, 2020) to categories formed by dividing f's predictions into 20 equal-frequency bins based on $D_{cal}$ . As an optimal benchmark, we consider the oracle satisfying (2).

# C.2 Experiment 1: Calibration and efficiency

In this experiment, we illustrate how calibration of the predictor $f$ can improve the efficiency (i.e., width) of the resulting prediction intervals. We consider the data-generating process of the previous section, with $n_{train} = n_{cal} = n_{test} = 1000$ , $d = 5$ , $a = 0$ , and $b = 0.6$ . The predictor $f$ is trained on $\mathcal{D}_{train}$ using the ranger (Wright and Ziegler, 2017) implementation of random forests with default settings. To control the calibration error in $f$ , we vary the distribution shift parameter $\kappa_{train}$ for $\mathcal{D}_{train}$ over $\{1, 1.5, 2, 2.5, 3\}$ .

Results. Figure 8a compares the average interval width across $D_{test}$ for SC-CP and baselines as the $\ell^{2}$ -calibration error in f increases. Here, we estimate the calibration error using the approach of Xu and Yadlowsky (2022). As calibration error increases, the average interval width for SC-SP appears smaller than those of uncond-CP, Mondrian-CP, and cond-CP, especially in comparison to uncond-CP. The observed efficiency gains in SC-CP are consistent with Theorem 4.3 and provides empirical evidence that self-calibrated conformity scores translate to tighter prediction intervals, given sufficient data. To test this hypothesis under controlled conditions, we compare the widths of prediction intervals obtained using vanilla (unconditional) CP for two scoring functions: $S : (x, y) \mapsto |y - f(x)|$ and the Venn-Abers (worst-case) scoring function $S_{\mathrm{cal}} : (x, y) \mapsto \max_{y \in \mathcal{Y}} |y - f_{n}^{(x,y)}(x)|$ . For miscoverage levels $\alpha \in \{0.05, 0.1, 0.2\}$ , the left panel of Figure 8b illustrates the relative efficiency gain achieved by using $S_{cal}$ , which we define as the ratio of the average interval widths for $S_{cal}$ relative to S. The widths and calibration errors in Figure 8b are averaged across 100 data replicates.

Role of calibration set size. With too small calibration sets, the isotonic calibration step in Alg. 2 can lead to overfitting. In such cases, the self-calibrated conformity scores could be larger than their uncalibrated counterparts, potentially resulting in less efficient prediction intervals. In our experiments, overfitting is mitigated by constraining the minimum size of the leaf node in the isotonic regression tree to 20. For $\alpha \in \{0.05, 0.1, 0.2\}$ , the right panel of Figure 8b displays the relationship

between $n_{cal}$ and the relative efficiency gain achieved by using $S_{cal}$ , holding calibration error fixed ( $\kappa_{train} = 3$ ). We find that calibration leads to a noticeable reduction in interval width as soon as $n_{cal} \geq 50$ .

![](images/e54573076ff1a6d6b3947e98843fc0e169d979a5cd289e7948ef988ddd1625b0.jpg)

<details>
<summary>scatter</summary>

| Calibration error in f(.) | SC-CP (ours) | cond-CP | uncond-CP | Mondrian-CP | Oracle |
| ------------------------- | ------------ | ------- | --------- | ----------- | ------ |
| 0.050                     | 0.28         | 0.34    | 0.36      | 0.30        | 0.27   |
| 0.075                     | 0.36         | 0.40    | 0.45      | 0.39        | 0.33   |
| 0.100                     | 0.43         | 0.46    | 0.53      | 0.43        | 0.38   |
| 0.125                     | 0.48         | 0.54    | 0.68      | 0.52        | 0.45   |
</details>

(a) Avg. interval width with varying calibration error.

![](images/5fb704a4397ceef6a6af4f415a87a953a3b0f4d72400eb9924c521e3264681b3.jpg)

<details>
<summary>line</summary>

| Calibration error in f | Relative Width (cal/uncal) - Level 0.05 | Relative Width (cal/uncal) - Level 0.1 | Relative Width (cal/uncal) - Level 0.2 |
| ---------------------- | ---------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0.050                  | 0.95                                     | 1.00                                   | 1.00                                   |
| 0.075                  | 0.85                                     | 0.90                                   | 0.95                                   |
| 0.100                  | 0.80                                     | 0.85                                   | 0.90                                   |
| 0.125                  | 0.75                                     | 0.80                                   | 0.85                                   |
| 30                     | 1.15                                     | 1.05                                   | 1.00                                   |
| 100                    | 0.85                                     | 0.80                                   | 0.90                                   |
| 300                    | 0.75                                     | 0.75                                   | 0.75                                   |
</details>

(b) Relative efficiency change from calibration.   
Figure 8: Figure 8a shows the average interval widths for varying $\ell^{2}$ -calibration errors in f. Figure 8b shows the relative change in average interval width using Venn-Abers calibrated versus uncalibrated predictions, with varying $\ell^{2}$ -calibration error in f (left), and calibration set size (right). Below the horizontal lines signifies efficiency gains for SC-CP.

# C.3 Experiment 2: Coverage and Adaptivity

In this experiment, we illustrate how self-calibration of $\widehat{C}_{n+1}(X_{n+1})$ can, in some cases, translate to stronger conditional coverage guarantees. Here we take $n_{train} = n_{cal} = 1000$ and $n_{test} = 2500$ , and no distribution shift ( $\kappa_{train} = 1$ ) in $D_{train}$ . We consider three setups: Setup A (d = 5, a = 0, b = 0.6) and Setup B (d = 5, a = 0, b = 0.6) which have a strong mean-variance relationship in the outcome process; and Setup C (d = 5, a = 0.6, b = 0) which has no such relationship. We obtain the predictor f from $D_{train}$ using a generalized additive model (Hastie and Tibshirani, 1987), so that it is well calibrated and accurately estimates $\mu$ . To assess the adaptivity of SC-CP to heteroscedasticity, we report the coverage and the average interval width within subgroups of $D_{test}$ defined by quintiles of the conditional variances $\{\sigma^{2}(X_{i}) : (X_{i}, Y_{i}) \in \mathcal{D}_{test}\}$ .

Results. The top two panels in Figure 9a displays the coverage and average interval width results for Setup A and Setup B. In both setups, SC-CP, cond-CP, and Mondrian-CP exhibit satisfactory coverage both marginally and within the quintile subgroups. In contrast, while uncond-CP attains the nominal level of marginal coverage, it exhibits noticeable overcoverage within the first three quintiles and significant undercoverage within the fifth quintile. The satisfactory performance of SC-CP with respect to heteroscedasticity in Setups A and B can be attributed to the strong mean-variance relationship in the outcome process. Regarding efficiency, the average interval widths of SC-CP are competitive with those of prediction-binned Mondrian-CP and the oracle intervals. The interval widths for cond-CP are wider than those for SC-CP and Mondrian-CP, especially for Setup B. This difference can be explained by cond-CP aiming for conditional validity in a 5D and 20D space, whereas SC-CP and Mondrian-CP target the 1D output space.

Limitation. If there is no mean-variance relationship in the outcomes, SC-CP is generally not expected to adapt to heteroscedasticity. The third (bottom) panel of Figure 9a displays SC-CP's performance in Setup C, where there is no such relationship. In this scenario, it is evident that the conditional coverage of both uncond-CP and SC-CP are poor, while cond-CP maintains adequate coverage.

Adaptivity and calibration set size. SC-CP can be derived by applying CP within subgroups defined by a data-dependent binning of the output space $f(\mathcal{X})$ , learned via Venn-Abers calibration, where the number of bins can grow with $n_{cal}$ . In particular, if $t \mapsto E[Y \mid f(X) = t]$ is asymptotically monotone, which is plausible when $f$ consistently estimates the outcome regression, then the number of bins selected by isotonic calibration will generally increase with $n_{cal}$ , and the width of these bins will tend to zero. As a consequence, in such cases, the self-calibration result in Theorem 4.2 translates to conditional guarantees over finer partitions of $f(\mathcal{X})$ as $n_{\mathrm{cal}}$ increases. For $d = 1$ and

![](images/6e6f5a9a1b4eec2eb7e54c86839a71ecd669023b97e62eb90631a5454adf32ed.jpg)  
SC-CP (ours) ▲ Cond-CP ■ Uncond-CP + Mondrian-CP ✉ Oracle

Figure 9: Figure 9a displays the coverage and average interval width of SC-CP, marginally and within quintiles of the conditional outcome variance for setups A, B, and C. For a d = 1 example, Figure 9b shows the adaptivity of SC-CP prediction bands ( $\alpha = 0.1$ ) across various calibration set sizes.

a strong mean-variance relationship $(a = 0, b = 0.6)$ , Figure 9b demonstrates how the adaptivity of SC-CP bands improves as $n_{cal}$ increases. In this case, for $n_{cal}$ sufficiently large, we find that the SC-CP bands closely match those of cond-CP and the oracle.

![](images/c6b5854ec5b920a3c09a2ce7c9d4d46aa61d4445e1d5a8129debcea2b38f10c3.jpg)

<details>
<summary>scatter</summary>

| Method         | Calibration error in f( ) | Coverage (α=0.1) |
| -------------- | ------------------------- | ---------------- |
| SC-CP (ours)   | 0.050                     | 0.90             |
| SC-CP (ours)   | 0.055                     | 0.90             |
| SC-CP (ours)   | 0.075                     | 0.88             |
| SC-CP (ours)   | 0.100                     | 0.90             |
| SC-CP (ours)   | 0.125                     | 0.89             |
| cond-CP        | 0.050                     | 0.93             |
| cond-CP        | 0.055                     | 0.90             |
| cond-CP        | 0.075                     | 0.91             |
| cond-CP        | 0.100                     | 0.88             |
| cond-CP        | 0.125                     | 0.89             |
| uncond-CP      | 0.050                     | 0.92             |
| uncond-CP      | 0.055                     | 0.89             |
| uncond-CP      | 0.075                     | 0.89             |
| uncond-CP      | 0.100                     | 0.89             |
| uncond-CP      | 0.125                     | 0.90             |
| Mondrian-CP    | 0.050                     | 0.91             |
| Mondrian-CP    | 0.075                     | 0.89             |
| Mondrian-CP    | 0.100                     | 0.88             |
| Mondrian-CP    | 0.125                     | 0.88             |
| Oracle         | 0.050                     | 0.91             |
| Oracle         | 0.075                     | 0.91             |
| Oracle         | 0.100                     | 0.91             |
| Oracle         | 0.125                     | 0.91             |
</details>

(a) Avg. interval width with varying calibration error.   
Figure 10: Figure 10a shows the marginal coverage corresponding to the interval widths of Figure 8a for varying $\ell^{2}$ -calibration errors in f.

# D Proofs

Proof of Theorem 4.1. It can be verified that an isotonic calibrator class $\Theta_{\mathrm{iso}}$ satisfies the following properties: (a) It consists of univariate regression trees (i.e., piecewise constant functions) that are monotonically nondecreasing; (b) For all elements $\theta \in \Theta_{\mathrm{iso}}$ and transformations $g:\mathbb{R}\to \mathbb{R}$ , it holds that $g\circ \theta \in \Theta_{\mathrm{iso}}$ . Property (a) holds by definition. Property (b) is satisfied since constraining the maximum number of constant segments by $k(n)$ does not constrain the possible values that the function may take in a given constant region. We note that the result of this theorem holds in general for any function class $\Theta_{\mathrm{iso}}$ that satisfies (a) and (b).

Recall from Alg. 1 that $f_{n}^{(X_{n + 1},Y_{n + 1})} = \theta_{n}^{(X_{n + 1},Y_{n + 1})}\circ f$ , where $\theta_{n}^{(X_{n + 1},Y_{n + 1})}\in \Theta_{iso}$ is an isotonic calibrator satisfying $f_{n}^{(X_{n + 1},Y_{n + 1})} = \theta_{n}^{(X_{n + 1},Y_{n + 1})}\circ f$ . By definition, recall that the isotonic calibrator class $\Theta_{iso}$ satisfies the invariance property that, for all $g:\mathbb{R}\to \mathbb{R}$ , the inclusion $\theta \in \Theta_{iso}$ implies $g\circ \theta \in \Theta_{iso}$ . Hence, for all $g:\mathbb{R}\rightarrow \mathbb{R}$ and $\varepsilon >0$ , it also holds that $(1 + \varepsilon g)\circ \theta_{n}^{(X_{n + 1},Y_{n + 1})}$ lies in $\Theta_{iso}$ . Now, we use that $(1 + \varepsilon g)\circ \theta_{n}^{(X_{n + 1},Y_{n + 1})}\circ f = (1 + \varepsilon g)\circ f_n^{(X_{n + 1},Y_{n + 1})}$ and that $f_{n}^{(X_{n + 1},Y_{n + 1})}$ is an empirical risk minimizer over the class $\{g\circ f:g\in \Theta_{iso}\}$ . Using these two observations, the first order derivative equations characterizing the empirical risk minimizer $f_{n}^{(X_{n + 1},Y_{n + 1})}$ imply, for all $g:\mathbb{R}\rightarrow \mathbb{R}$ , that

$$
\begin{array}{l} \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} (g \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) \left\{Y _ {i} - f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \right\} = \frac {d}{d \varepsilon} \left[ \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} \left\{Y _ {i} - (1 + \varepsilon g) \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} \right\} ^ {2} \right] \Bigg | _ {\varepsilon = 0} \\ = 0. \tag {5} \\ \end{array}
$$

Taking expectations of both sides of the above display, we conclude

$$
\frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} \mathbb {E} \left[ (g \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) \left\{Y _ {i} - f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \right\} \right] = 0. \tag {6}
$$

We now use the fact that $\{(X_i, Y_i, f_n^{(X_{n+1}, Y_{n+1})}(X_i)) : i \in [n + 1]\}$ are exchangeable, since $\{(X_i, Y_i)\}_{i=1}^{n+1}$ are exchangeable by C1 and the function $f_n^{(X_{n+1}, Y_{n+1})}$ is invariant under permutations of $\{(X_i, Y_i)\}_{i=1}^{n+1}$ . Consequently, Equation (6) remains true if we replace each $(X_i, Y_i, f_n^{(X_{n+1}, Y_{n+1})}(X_i))$ with $i \in [n]$ by $(X_{n+1}, Y_{n+1}, f_n^{(X_{n+1}, Y_{n+1})}(X_{n+1}))$ . That is,

$$
\begin{array}{l} \mathbb {E} \left[ (g \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {n + 1}) \left\{Y _ {n + 1} - f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \right\} \right] \\ = \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} \mathbb {E} \left[ (g \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {n + 1}) \left\{Y _ {n + 1} - f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \right\} \right] \\ = \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} \mathbb {E} \left[ (g \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) \left\{Y _ {i} - f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \right\} \right] \\ = 0. \\ \end{array}
$$

By the law of iterated conditional expectations, the preceding display further implies

$$
\mathbb {E} \left[ (g \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {n + 1}) \left\{\mathbb {E} [ Y _ {n + 1} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) ] - f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \right\} \right] = 0.
$$

Taking $g: \mathbb{R} \to \mathbb{R}$ to be defined by $(g \circ f_n^{(X_{n+1}, Y_{n+1})})(X_{n+1}) := \mathbb{E}[Y_{n+1} \mid f_n^{(X_{n+1}, Y_{n+1})}(X_{n+1})] - f_n^{(X_{n+1}, Y_{n+1})}(X_{n+1})$ , we find

$$
\mathbb {E} \left[ \left\{\mathbb {E} [ Y _ {n + 1} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) ] - f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \right\} ^ {2} \right] = 0.
$$

The above equality implies $\mathbb{E}[Y_{n + 1}\mid f_n^{(X_{n + 1},Y_{n + 1})}(X_{n + 1})] = f_n^{(X_{n + 1},Y_{n + 1})}(X_{n + 1})$ almost surely, as desired.

![](images/56f2b10952b9f219451ba1e367f755bf359d2021c0653cea40fa819780f590f5.jpg)

Proof of Theorem 4.2. Recall, for a quantile level $\alpha \in (0,1)$ , the "pinball" quantile loss function $\ell_{\alpha}$ is given by

$$
\ell_ {\alpha} (f (x), s) := \left\{ \begin{array}{l l} \alpha (s - f (x)) & \text { if } s \geq f (x), \\ (1 - \alpha) (f (x) - s) & \text { if } s <   f (x). \end{array} \right.
$$

As established in Gibbs et al. (2023), each subgradient of $\ell_{\alpha}(\cdot, x)$ at $f$ in the direction $g$ , for some $\beta \in [\alpha - 1, \alpha]$ , given by:

$$
\partial_ {\varepsilon [ \beta ]} \left\{\ell_ {\alpha} (f (x) + \varepsilon g (x), s) \right\} \big | _ {\varepsilon = 0} := 1 (f (x) \neq s) g (x) \{\alpha - 1 (f <   s) \} + 1 (f (x) = s) \beta g (x).
$$

For $(x,y)\in\mathcal{X}\times\mathcal{Y}$ , define the empirical risk minimizer:

$$
\rho_ {n} ^ {(x, y)} \in \underset {\theta \circ f _ {n} ^ {(x, y)}; \theta : \mathbb {R} \to \mathbb {R}} {\operatorname{argmin}} \sum_ {i = 1} ^ {n} \ell_ {\alpha} \left(\theta \circ f _ {n} ^ {(x, y)} (X _ {i}), S _ {i} ^ {(x, y)}\right) + \ell_ {\alpha} \left(\theta \circ f _ {n} ^ {(x, y)} (x), S _ {n + 1} ^ {(x, y)}\right).
$$

Then, since the isotonic calibrated predictor $f_{n}^{(x,y)}$ is piece-wise constant and the above optimization problem is unconstrained in the map $\theta : \mathbb{R} \to \mathbb{R}$ , it holds that the evaluation $\rho_{n}^{(x,y)}(x)$ lies in the solution set:

$$
\underset {q \in \mathbb {R}} {\operatorname{argmin}} \sum_ {i = 1} ^ {n} K _ {i} (f _ {n} ^ {(x, y)}, x) \cdot \ell_ {\alpha} (q, S _ {i} ^ {(x, y)}) + \ell_ {\alpha} (q, S _ {n + 1} ^ {(x, y)}),
$$

where $K_{i}(f_{n}^{(x,y)},x) = \mathbf{1}\Big\{f_{n}^{(x,y)}(X_{i}) = f_{n}^{(x,y)}(x)\Big\}$ . Consequently, we see that the empirical quantile $\rho_n^{(x,y)}(x)$ defined in Alg. 2 coincides with the evaluation of the empirical risk minimizer $\rho_n^{(x,y)}(\cdot)$ at $x$ , as suggested by our notation.

We will now theoretically analyze Alg. 2 by studying the empirical risk minimizer $x' \mapsto \rho_n^{(X_{n+1}, Y_{n+1})}(x')$ . To do so, we modify the arguments used to establish Theorem 2 of Gibbs et al. (2023). As in Gibbs et al. (2023), we begin by examining the first-order equations of the convex optimization problem defining $x' \mapsto \rho_n^{(X_{n+1}, Y_{n+1})}(x')$ .

Studying first-order equations of convex problem: Given any transformation $\theta : \mathbb{R} \to \mathbb{R}$ , each subgradient of the map

$$
\varepsilon \mapsto \ell_ {\alpha} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) + \varepsilon \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}), S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})}\right)
$$

is, for some $\beta \in [\alpha - 1, \alpha]^{n + 1}$ , of the following form:

$$
\begin{array}{l} \left. \partial_ {\varepsilon [ \beta ]} \left\{\ell_ {\alpha} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) + \varepsilon \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}), S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})}\right) \right\} \right| _ {\varepsilon = 0} \\ := \left\{ \begin{array}{l l} \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) <   S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} & \text {if} S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} \neq \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}), \\ \beta_ {i} (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) & \text {if} S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} = \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}). \end{array} \right. \\ \end{array}
$$

Now, since $\rho_n^{(X_{n+1},Y_{n+1})}$ is an empirical risk minimizer of the quantile loss over the class $\{\theta \circ f_n^{(X_{n+1},Y_{n+1})}; \theta : \mathbb{R} \to \mathbb{R}\}$ , there exists some vector $\beta^* = (\beta_1^*, \ldots, \beta_{n+1}^*) \in [\alpha - 1, \alpha]^{n+1}$ such that

$$
\begin{array}{l} 0 = \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} 1 \{S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} \neq \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) <   S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} \right] \\ + \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} 1 \{S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} = \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \} \left[ \beta_ {i} ^ {*} (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) \right] \\ \end{array}
$$

The above display can be rewritten as:

$$
\begin{array}{l} \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) <   S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} \right] \\ = \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} 1 \left\{S _ {i} ^ {\left(X _ {n + 1}, Y _ {n + 1}\right)} = \rho_ {n} ^ {\left(X _ {n + 1}, Y _ {n + 1}\right)} \left(X _ {i}\right) \right\} \left[ \left(1 - \beta_ {i} ^ {*}\right) \left(\theta \circ f _ {n} ^ {\left(X _ {n + 1}, Y _ {n + 1}\right)}\right) \left(X _ {i}\right) \right]. \tag {7} \\ \end{array}
$$

Now, observe that the collection of random variables

$$
\left\{(f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}), S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})}, \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i})): i \in [ n + 1 ] \right\}
$$

are exchangeable, since $\{((X_i,Y_i):i\in [n + 1]\}$ are exchangeable by C1 and the functions $f_n^{(X_{n + 1},Y_{n + 1})}(\cdot)$ and $\rho_n^{(X_{n + 1},Y_{n + 1})}(\cdot)$ are unchanged under permutations of the training data $\{(X_i,Y_i)\}_{i\in [n + 1]}$ . Thus, the expectation of the left-hand side of Equation (7) can be expressed as:

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) <   S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} \right] \right] \\ = \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} \mathbb {E} \left[ \theta \circ f _ {n} ^ {\left(X _ {n + 1}, Y _ {n + 1}\right)} \left(X _ {i}\right) \left\{\alpha - 1 \left(\rho_ {n} ^ {\left(X _ {n + 1}, Y _ {n + 1}\right)} \left(X _ {i}\right) <   S _ {i} ^ {\left(X _ {n + 1}, Y _ {n + 1}\right)}\right) \right\} \right] \\ = \mathbb {E} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} \right], \\ \end{array}
$$

where the final inequality follows from exchangeability. Combining this with Equation (7), we find

$$
\mathbb {E} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} \right] \tag {8}
$$

$$
= \mathbb {E} \left[ 1 \{S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} = \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \} \left[ (1 - \beta_ {i} ^ {*}) (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) \right] \right| \tag {9}
$$

Lower bound on coverage: We first obtain the lower coverage bound in the theorem statement. Note, for any nonnegative $\theta : R \to R$ , that (9) implies:

$$
\mathbb {E} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} \right] \geq 0.
$$

This inequality holds since $(1 - \beta_{i}^{*})\geq 0$ and $(\theta \circ f_n^{(X_{n + 1},Y_{n + 1})})(X_i)\geq 0$ almost surely, for each $i\in [n + 1]$ .

By the law of iterated expectations, we then have

$$
\mathbb {E} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \left\{\alpha - \mathbb {P} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right) \right\} \right] \geq 0.
$$

Taking $\theta : \mathbb{R} \to \mathbb{R}$ as a nonnegative map that almost surely satisfies

$$
\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) = 1 \left\{\alpha \leq \mathbb {P} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right) \right\},
$$

we find

$$
\left. \right. - \mathbb {E} \left[\left\{\alpha - \mathbb {P} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right)\right\} _ {-} \right] \geq 0,
$$

where the map $t \mapsto \{t\}_{-} := |t|1(t \leq 0)$ extracts the negative part of its input. Multiplying both sides of the previous inequality by -1, we obtain

$$
0 \leq \mathbb {E} \left[ \left\{\alpha - \mathbb {P} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right) \right\} _ {-} \right] \leq 0.
$$

We conclude that the negative part of $\left\{\alpha -\mathbb{P}\left(\rho_n^{(X_{n + 1},Y_{n + 1})}(X_{n + 1}) <   S_{n + 1}^{(X_{n + 1},Y_{n + 1})}\mid f_n^{(X_{n + 1},Y_{n + 1})}(X_{n + 1})\right)\right\}$ is almost surely zero. Thus, it must be almost surely true that

$$
\alpha \geq \mathbb {P} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right).
$$

Note that the event $Y_{n+1} \in \widehat{C}_{n+1}(X_{n+1}) = \{y \in \mathcal{Y} : S_{n+1}^{(X_{n+1}, y)} \leq \rho_n^{(X_{n+1}, y)}(X_{n+1})\}$ occurs if, and only if, $\rho_n^{(X_{n+1}, Y_{n+1})}(X_{n+1}) \geq S_{n+1}^{(X_{n+1}, Y_{n+1})}$ . As a result, we obtain the desired lower coverage bound:

$$
1 - \alpha \leq \mathbb {P} \left(Y _ {n + 1} \in \widehat {C} _ {n + 1} (X _ {n + 1}) \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right).
$$

Deviation bound for the coverage: We now bound the deviation of the coverage of SC-CP from the lower bound. Note, for any $f : R \to R$ , that (9) implies:

$$
\begin{array}{l} \mathbb {E} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} \right] \\ \leq \left| \mathbb {E} \left[ 1 \{S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} = \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \} \left[ (1 - \beta_ {i} ^ {*}) (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) \right] \right| \right|. \tag {10} \\ \end{array}
$$

Using that $(1 - \beta_{i}^{*})\in [0,1]$ and exchangeability, we can bound the right-hand side of the above as

$$
\begin{array}{l} \left| \mathbb {E} \left[ \frac {1}{n + 1} \sum_ {i = 1} ^ {n + 1} 1 \{S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} = \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \} \left[ (1 - \beta_ {i} ^ {*}) (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) \right] \right] \right| \\ \leq \frac {1}{n + 1} \mathbb {E} \left[ \left\{\max _ {i \in [ n + 1 ]} \left| (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) \right| \right\} \sum_ {i = 1} ^ {n + 1} 1 \{S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} = \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \} \right], \\ \end{array}
$$

Next, since there are no ties by C3, the event $1\{S_i^{(X_{n+1},Y_{n+1})} = \rho_n^{(X_{n+1},Y_{n+1})}(X_i)\}$ for some index $i \in [n+1]$ can only occur once per piecewise constant segment of $\rho_n^{(X_{n+1},Y_{n+1})}$ , since, otherwise, $S_i^{(X_{n+1},Y_{n+1})} = \rho_n^{(X_{n+1},Y_{n+1})}(X_i) = \rho_n^{(X_{n+1},Y_{n+1})}(X_j) = S_j^{(X_{n+1},Y_{n+1})}$ for some $i \neq j$ . However, $\rho_n^{(X_{n+1},Y_{n+1})}$ is a transformation of $f_n^{(X_{n+1},Y_{n+1})}$ and, therefore, has the same number of constant segments as $f_n^{(X_{n+1},Y_{n+1})}$ . Thus, it holds that

$$
\sum_ {i = 1} ^ {n + 1} 1 \{S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} = \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \} \leq N ^ {(X _ {n + 1}, Y _ {n + 1})},
$$

where $N^{(X_{n + 1},Y_{n + 1})}$ is the (random) number of constant segments of $f_{n}^{(X_{n + 1},Y_{n + 1})}$ . This implies that

$$
\begin{array}{l} \frac {1}{n + 1} \mathbb {E} \left[ \max _ {i \in [ n + 1 ]} | (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) | \sum_ {i = 1} ^ {n + 1} 1 \{S _ {i} ^ {(X _ {n + 1}, Y _ {n + 1})} = \rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {i}) \} \right] \\ \leq \frac {1}{n + 1} \mathbb {E} \left[ N ^ {(X _ {n + 1}, Y _ {n + 1})} \max _ {i \in [ n + 1 ]} | (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) | \right]. \\ \end{array}
$$

Combining this bound with (10), we find

$$
\begin{array}{l} \mathbb {E} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \{\alpha - 1 (\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})}) \} \right] \\ \leq \frac {1}{n + 1} \mathbb {E} \left[ N ^ {(X _ {n + 1}, Y _ {n + 1})} \max _ {i \in [ n + 1 ]} | (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) | \right]. \\ \end{array}
$$

By the law of iterated expectations, we then have

$$
\begin{array}{l} \mathbb {E} \left[ \theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) \left\{\alpha - \mathbb {P} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right) \right\} \right] \\ \leq \frac {1}{n + 1} \mathbb {E} \left[ N ^ {(X _ {n + 1}, Y _ {n + 1})} \max _ {i \in [ n + 1 ]} | (\theta \circ f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})}) (X _ {i}) | \right]. \\ \end{array}
$$

Next, let $\mathcal{V} \subset \mathbb{R}$ denote the support of the random variable $f_n^{(X_{n+1}, Y_{n+1})}(X_{n+1})$ . Then, taking $\theta$ to be $t \mapsto 1(t \in \mathcal{V})\mathrm{sign}\left\{\alpha - \mathbb{P}\left(\rho_n^{(X_{n+1}, Y_{n+1})}(X_{n+1}) < S_{n+1}^{(X_{n+1}, Y_{n+1})} \mid f_n^{(X_{n+1}, Y_{n+1})}(X_{n+1}) = t\right)\right\}$ , which falls almost surely in $\{-1, 1\}$ , we obtain the mean absolute error bound:

$$
\mathbb {E} \left| \alpha - \mathbb {P} \left(\rho_ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1}) <   S _ {n + 1} ^ {(X _ {n + 1}, Y _ {n + 1})} \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right) \right| \leq \frac {1}{n + 1} \mathbb {E} \left[ N ^ {(X _ {n + 1}, Y _ {n + 1})} \right].
$$

Since the event $Y_{n+1} \notin \widehat{C}_{n+1}(X_{n+1}) = \{y \in \mathcal{Y} : S_{n+1}^{(X_{n+1},y)} \leq \rho_n^{(X_{n+1},y)}(X_{n+1})\}$ occurs if, and only if, $\rho_n^{(X_{n+1},Y_{n+1})}(X_{n+1}) < S_{n+1}^{(X_{n+1},Y_{n+1})}$ , we conclude that

$$
\mathbb {E} \left| \alpha - \mathbb {P} \left(Y _ {n + 1} \not \in \widehat {C} _ {n + 1} (X _ {n + 1}) \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right) \right| \leq \frac {\mathbb {E} [ N ^ {(X _ {n + 1} , Y _ {n + 1})} ]}{n + 1}.
$$

Under C4, $\mathbb{E}[N^{(X_{n + 1},Y_{n + 1})}]\leq n^{1 / 3}$ polylog $n$ , such that

$$
\mathbb {E} \left| \alpha - \mathbb {P} \left(Y _ {n + 1} \not \in \widehat {C} _ {n + 1} (X _ {n + 1}) \mid f _ {n} ^ {(X _ {n + 1}, Y _ {n + 1})} (X _ {n + 1})\right) \right| \leq \frac {n ^ {1 / 3} \mathrm{polylog} n}{n + 1} \leq \frac {\mathrm{polylog} n}{n ^ {2 / 3}},
$$

as desired.

![](images/1ea19426d56fc15f354af8a35e931c122eae86dcc17f894bc726f7b5c960dabc.jpg)

Proof of Theorem 4.3. Let $P_{n+1}$ denote the empirical distribution of $\{(X_i, Y_i)\}_{i=1}^{n+1}$ and let $P_n$ denote the empirical distribution of $\{(X_i, Y_i)\}_{i=1}^n$ . For any function $g : X \times Y \to R$ : we use the following empirical process notation: $Pg := \int g(x, y) dP(x, y)$ , $P_{n+1}g := \int g(x, y) dP_{n+1}(x, y)$ , and $P_n g := \int g(x, y) dP_n(x, y)$ .

Define the risk functions $R_{n}^{(x,y)}(\theta) := \frac{1}{n+1} \sum_{i=1}^{n} \{S_{\theta}(X_i, Y_i)\}^2 + \frac{1}{n+1} \{S_{\theta}(x, y)\}^2$ , $R_{n+1}(\theta) := \frac{1}{n+1} \sum_{i=1}^{n+1} \{S_{\theta}(X_i, Y_i)\}^2$ , and $R_0(\theta) := \int \{S_{\theta}(x, y)\}^2 dP(x, y)$ . Moreover, define the risk minimizers as $\theta_n^{(x,y)} := \arg\min_{\theta \in \Theta_{iso}} R_n^{(x,y)}(f)$ and $\theta_0 := \arg\min_{\theta \in \Theta_{iso}} R_0(\theta)$ . Observe that $R_n^{(x,y)}(\theta_n^{(x,y)}) - R_n^{(x,y)}(\theta_0) \leq 0$ since $f_n$ minimizes $R_n$ over $\Theta_{iso}$ . Using this inequality, it follows that

$$
\begin{array}{l} R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {0} (\theta_ {0}) = R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {n} ^ {(x, y)} (\theta_ {n} ^ {(x, y)}) \\ + R _ {n} ^ {(x, y)} (\theta_ {n} ^ {(x, y)}) - R _ {n} ^ {(x, y)} (\theta_ {0}) + R _ {n} ^ {(x, y)} (\theta_ {0}) - R _ {0} (\theta_ {0}) \\ \leq R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {n} ^ {(x, y)} (\theta_ {n} ^ {(x, y)}) - \left\{R _ {n} ^ {(x, y)} (\theta_ {0}) - R _ {0} (\theta_ {0}) \right\} \\ \leq R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {n + 1} (\theta_ {n} ^ {(x, y)}) - \{R _ {n + 1} (\theta_ {0}) - R _ {0} (\theta_ {0}) \} \\ + R _ {n} ^ {(x, y)} (\theta_ {n} ^ {(x, y)}) - R _ {n + 1} (\theta_ {n} ^ {(x, y)}) - \left\{R _ {n} ^ {(x, y)} (\theta_ {0}) - R _ {n + 1} (\theta_ {0}) \right\}. \\ \end{array}
$$

The first term on the right-hand side of the above display can written as

$$
R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {n + 1} (\theta_ {n} ^ {(x, y)}) - \{R _ {n + 1} (\theta_ {0}) - R _ {0} (\theta_ {0}) \} = (P _ {n + 1} - P) \left[ \{S _ {\theta_ {n} ^ {(x, y)}} \} ^ {2} - \{S _ {\theta_ {0}} \} ^ {2} \right].
$$

We now bound the second term, $R_{n}^{(x,y)}(\theta_{n}^{(x,y)}) - R_{n + 1}(\theta_{n}^{(x,y)}) - \left\{R_{n}^{(x,y)}(\theta_{0}) - R_{n + 1}(\theta_{0})\right\}$ . For any $\theta \in \Theta_{iso}$ , observe that

$$
R _ {n} ^ {(x, y)} (\theta) - R _ {n + 1} (\theta) = \frac {1}{n + 1} \left[ \{y - \theta \circ f (x) \} ^ {2} - \{Y _ {n + 1} - \theta \circ f (X _ {n + 1}) \} ^ {2} \right].
$$

We know that $\theta_{n}^{(x,y)}$ and $\theta_{0}$ , being defined via isotonic regression, are uniformly bounded by $B := \sup_{y \in Y} |y|$ , which is finite by C6. Therefore,

$$
\left| R _ {n} ^ {(x, y)} (\theta_ {n} ^ {(x, y)}) - R _ {n + 1} (\theta_ {n} ^ {(x, y)}) - \left\{R _ {n} ^ {(x, y)} (\theta_ {0}) - R _ {n + 1} (\theta_ {0}) \right\} \right| \leq \frac {8 B ^ {2}}{n + 1} = O (n ^ {- 1}).
$$

Combining the previous displays, we obtain the excess risk bound

$$
R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {0} (\theta_ {0}) \leq (P _ {n + 1} - P) \left[ \{S _ {\theta_ {n} ^ {(x, y)}} \} ^ {2} - \{S _ {\theta_ {0}} \} ^ {2} \right] + O (n ^ {- 1}). \tag {11}
$$

Next, we claim that $R_0(\theta_n^{(x,y)}) - R_0(\theta_0) \geq \| (\theta_n^{(x,y)} \circ f) - (\theta_0 \circ f) \|_P^2$ . To show this, expanding the squares, note, pointwise for each $x \in \mathcal{X}$ and $y \in \mathcal{Y}$ , that

$$
\begin{array}{l} \{S _ {\theta_ {n} ^ {(x, y)}} (x, y) \} ^ {2} - \{S _ {\theta_ {0}} (x, y) \} ^ {2} = \{\theta_ {n} ^ {(x, y)} \circ f (x) \} ^ {2} - \{\theta_ {0} \circ f (x) \} ^ {2} - 2 y \left\{(\theta_ {n} ^ {(x, y)} \circ f) (x) - (\theta_ {0} \circ f) (x) \right\} \\ = \left\{(\theta_ {n} ^ {(x, y)} \circ f) (x) + (\theta_ {0} \circ f) (x) - 2 y \right\} \left\{(\theta_ {n} ^ {(x, y)} \circ f) (x) - (\theta_ {0} \circ f) (x) \right\}. \\ \end{array}
$$

Consequently,

$$
R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {0} (\theta_ {0}) = \int \left\{(\theta_ {n} ^ {(x, y)} \circ f) (x) + (\theta_ {0} \circ f) (x) - 2 y \right\} \left\{(\theta_ {n} ^ {(x, y)} \circ f) (x) - (\theta_ {0} \circ f) (x) \right\} d P (x). \tag {12}
$$

The class $\Theta_{iso}$ consists of all isotonic functions and is, therefore, a convex space. Thus, the first-order derivative equations defining the population minimizer $\theta_0$ imply that

$$
\int \left\{\theta_ {n} ^ {(x, y)} \circ f (x) - \theta_ {0} \circ f (x) \right\} \{y - \theta_ {0} \circ f (x) \} d P (x, y) \leq 0. \tag {13}
$$

Combining (12) and (13), we find

$$
\begin{array}{l} R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {0} (\theta_ {0}) = \int \left\{(\theta_ {n} ^ {(x, y)} \circ f) (x) - (\theta_ {0} \circ f) (x) \right\} ^ {2} d P (x) \\ + 2 \int \left\{\left(\theta_ {0} \circ f\right) (x) - y \right\} \left\{\left(\theta_ {n} ^ {(x, y)} \circ f\right) (x) - \left(\theta_ {0} \circ f\right) (x) \right\} d P (x) \\ \geq \int \left\{(\theta_ {n} ^ {(x, y)} \circ f) (x) - (\theta_ {0} \circ f) (x) \right\} ^ {2} d P (x), \\ \end{array}
$$

as desired. Combining this lower bound with (11), we obtain the inequality

$$
\int \left\{(\theta_ {n} ^ {(x, y)} \circ f) (x) - (\theta_ {0} \circ f) (x) \right\} ^ {2} d P (x) \leq R _ {0} (\theta_ {n} ^ {(x, y)}) - R _ {0} (\theta_ {0}) \leq (P _ {n} - P) \left[ \{S _ {\theta_ {n} ^ {(x, y)}} \} ^ {2} - \{S _ {\theta_ {0}} \} ^ {2} \right] + O (1 / n) \tag {14}
$$

Define $\delta_{n}:=\sqrt{\int\left\{(\theta_{n}^{(x,y)}\circ f)(x)-(\theta_{0}\circ f)(x)\right\}^{2}dP(x)}$ , the bound $B:=\sup_{y\in\mathcal{Y}}|y|$ , and the function class,

$$
\Theta_ {1, n} := \{(x, y) \mapsto \{(\theta_ {1} + \theta_ {2}) \circ f - 2 y \} \{(\theta_ {1} - \theta_ {2}) \circ f \} \}.
$$

Using this notation, (14) implies

$$
\begin{array}{l} \delta_ {n} ^ {2} \leq \sup _ {\theta_ {1}, \theta_ {2} \in \Theta_ {i s o}: \| \theta_ {1} - \theta_ {2} \| \leq \delta_ {n}} \int \{(\theta_ {1} + \theta_ {2}) \circ f (x) - 2 y \} \{(\theta_ {1} - \theta_ {2}) \circ f (x) \} d (P _ {n} - P) (x, y) + O (1 / n) \\ \leq \sup _ {h \in \Theta_ {1, n}: \| h \| \leq 4 B \delta_ {n}} (P _ {n} - P) h + O (1 / n) \\ \end{array}
$$

Using the above inequality and C5, we will use an argument similar to the proof of Theorem 3 in van der Laan et al. (2023) to establish that $\delta_n = O_p(n^{-1/3})$ . This rate then implies the result of the theorem. To see this, note, by the reverse triangle inequality,

$$
\left| S _ {n} ^ {(x, y)} (y ^ {\prime}, x ^ {\prime}) - S _ {0} (x ^ {\prime}, y ^ {\prime}) \right| = \left| | y ^ {\prime} - \theta_ {n} ^ {(x, y)} (x ^ {\prime}) | - | y ^ {\prime} - \theta_ {0} (x ^ {\prime}) | \right|
$$

$$
\leq \left| \theta_ {0} (x ^ {\prime}) - \theta_ {n} ^ {(x, y)} (x ^ {\prime}) \right|.
$$

Squaring and integrating the left- and right-hand sides, we find

$$
\int \left| S _ {n} ^ {(x, y)} (y ^ {\prime}, x ^ {\prime}) - S _ {0} (x ^ {\prime}, y ^ {\prime}) \right| ^ {2} d P (x ^ {\prime}, y ^ {\prime}) \leq \int \left| \theta_ {0} (x ^ {\prime}) - \theta_ {n} ^ {(x, y)} (x ^ {\prime}) \right| ^ {2} d P (x ^ {\prime}) = \delta_ {n} ^ {2},
$$

as desired.

We now establish that $\delta_n = O_p(n^{-1/3})$ . For a function class $\mathcal{F}$ , let $N(\epsilon, \mathcal{F}, L_2(P))$ denote the $\epsilon$ -covering number (van der Vaart and Wellner, 1996) of $\mathcal{F}$ and define the uniform entropy integral of $\mathcal{F}$ by

$$
\mathcal {J} (\delta , \mathcal {F}) := \int_ {0} ^ {\delta} \sup _ {Q} \sqrt {\log N (\epsilon , \mathcal {F} , L _ {2} (Q))} d \epsilon ,
$$

where the supremum is taken over all discrete probability distributions $Q$ . We note that

$$
\mathcal {J} (\delta , \Theta_ {1, n}) = \int_ {0} ^ {\delta} \sup _ {Q} \sqrt {N (\varepsilon , \Theta_ {1 , n} , \| \cdot \| _ {Q})} d \varepsilon = \int_ {0} ^ {\delta} \sup _ {Q} \sqrt {N (\varepsilon , \Theta_ {i s o} , \| \cdot \| _ {Q \circ f ^ {- 1}})} d \varepsilon = \mathcal {J} (\delta , \Theta_ {i s o}),
$$

where $Q \circ f^{-1}$ is the push-forward probability measure for the random variable $f(W)$ . Additionally, the covering number bound for bounded monotone functions given in Theorem 2.7.5 of van der Vaart and Wellner (1996) implies that

$$
\mathcal {J} (\delta , \Theta_ {1, n}) = \mathcal {J} (\delta , \Theta_ {i s o}) \lesssim \sqrt {\delta}.
$$

Recall that f is obtained from an external dataset, say $E_{n}$ , independent of the calibration data. Noting that f is deterministic conditional on a training dataset $E_{n}$ . Applying Theorem 2.1 of Van Der Vaart and Wellner (2011) conditional on $E_{n}$ , we obtain, for any $\delta > 0$ , that

$$
E \left[ \sup _ {h \in \Theta_ {1, n}: \| h \| \leq 4 B \delta} (P _ {n} - P) h   |   \mathcal {E} _ {n} \right] \lesssim n ^ {- 1 / 2} \mathcal {J} (\delta , \Theta_ {1, n}) \left(1 + \frac {\mathcal {J} (\delta , \Theta_ {1 , n})}{\sqrt {n} \delta^ {2}}\right).
$$

$$
\lesssim n ^ {- 1 / 2} \mathcal {J} (\delta , \Theta_ {i s o}) \left(1 + \frac {\mathcal {J} (\delta , \Theta_ {i s o})}{\sqrt {n} \delta^ {2}}\right).
$$

Noting that the right-hand side of the above bound is deterministic, we conclude that

$$
E \left[ \sup _ {h \in \Theta_ {1, n}: \| h \| \leq 4 B \delta} (P _ {n} - P) h \right] \lesssim n ^ {- 1 / 2} \mathcal {J} (\delta , \Theta_ {i s o}) \left(1 + \frac {\mathcal {J} (\delta , \Theta_ {i s o})}{\sqrt {n} \delta^ {2}}\right).
$$

We use the so-called “peeling” argument (van der Vaart and Wellner, 1996) to obtain our bound for $\delta_{n}$ . Note

$$
\begin{array}{l} P \left(\delta_ {n} ^ {2} \geq n ^ {- 2 / 3} 2 ^ {M}\right) = \sum_ {m = M} ^ {\infty} P \left(2 ^ {m + 1} \geq n ^ {2 / 3} \delta_ {n} ^ {2} \geq 2 ^ {m}\right) \\ = \sum_ {m = M} ^ {\infty} P \left(2 ^ {m + 1} \geq n ^ {2 / 3} \delta_ {n} ^ {2} \geq 2 ^ {m}, \delta_ {n} ^ {2} \leq \sup _ {h \in \Theta_ {1, n}: \| h \| \leq 4 B \delta_ {n}} (P _ {n} - P) h + O (1 / n)\right) \\ = \sum_ {m = M} ^ {\infty} P \left(2 ^ {m + 1} \geq n ^ {2 / 3} \delta_ {n} ^ {2} \geq 2 ^ {m}, 2 ^ {2 m} n ^ {- 2 / 3} \leq \sup _ {h \in \Theta_ {1, n}: \| h \| \leq 4 B 2 ^ {m + 1} n ^ {- 1 / 3}} (P _ {n} - P) h + O (1 / n)\right) \\ \leq \sum_ {m = M} ^ {\infty} P \left(2 ^ {2 m} n ^ {- 2 / 3} \leq \sup _ {h \in \Theta_ {1, n}: \| h \| \leq 4 B 2 ^ {m + 1} n ^ {- 1 / 3}} (P _ {n} - P) h + O (1 / n)\right) \\ \leq \sum_ {m = M} ^ {\infty} \frac {E \left[ \sup _ {h \in \Theta_ {1 , n} : \| h \| \leq 4 B 2 ^ {m + 1} n ^ {- 1 / 3}} (P _ {n} - P) h \right] + O (1 / n)}{2 ^ {2 m} n ^ {- 2 / 3}} \\ \leq \sum_ {m = M} ^ {\infty} \frac {\mathcal {J} (2 ^ {m + 1} n ^ {- 1 / 3} , \Theta_ {i s o}) \left(1 + \frac {\mathcal {J} (2 ^ {2 m + 2} n ^ {- 2 / 3} , \Theta_ {i s o})}{\sqrt {n} 2 ^ {m + 1} n ^ {- 1 / 3}}\right)}{\sqrt {n} 2 ^ {2 m} n ^ {- 2 / 3}} + \sum_ {m = M} ^ {\infty} \frac {O (1 / n)}{2 ^ {2 m} n ^ {- 2 / 3}} \\ \leq \sum_ {m = M} ^ {\infty} \frac {2 ^ {(m + 1) / 2} n ^ {- 1 / 6}}{2 ^ {2 m} n ^ {- 1 / 6}} + \sum_ {m = M} ^ {\infty} \frac {o (1)}{2 ^ {2 m}} \\ \lesssim \sum_ {m = M} ^ {\infty} \frac {2 ^ {(m + 1) / 2}}{2 ^ {2 m}}. \\ \end{array}
$$

Since $\sum_{m=1}^{\infty} \frac{2^{(m+1)/2}}{2^{2m}} < \infty$ , we have that $\sum_{m=M}^{\infty} \frac{2^{(m+1)/2}}{2^{2m}} \to 0$ as $M \to \infty$ . Therefore, for all $\varepsilon > 0$ , we can choose $M > 0$ large enough so that

$$
P \left(\delta_ {n} ^ {2} \geq n ^ {- 2 / 3} 2 ^ {M}\right) \leq \varepsilon .
$$

We conclude that $\delta_n = O_p(n^{-1/3})$ as desired.

![](images/6bebafb407a64766bfc2e703651c697170b05eb28577a29eb5e83af64e62ab8c.jpg)