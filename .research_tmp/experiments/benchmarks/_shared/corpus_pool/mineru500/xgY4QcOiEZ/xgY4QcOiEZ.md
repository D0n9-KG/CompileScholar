# Learning a Neuron by a Shallow ReLU Network: Dynamics and Implicit Bias for Correlated Inputs

Dmitry Chistikov\*

University of Warwick

d.chistikov@warwick.ac.uk

Matthias Englert\*

University of Warwick

m.englert@warwick.ac.uk

Ranko Lazić\*

University of Warwick

r.s.lazic@warwick.ac.uk

# Abstract

We prove that, for the fundamental regression task of learning a single neuron, training a one-hidden layer ReLU network of any width by gradient flow from a small initialisation converges to zero loss and is implicitly biased to minimise the rank of network parameters. By assuming that the training points are correlated with the teacher neuron, we complement previous work that considered orthogonal datasets. Our results are based on a detailed non-asymptotic analysis of the dynamics of each hidden neuron throughout the training. We also show and characterise a surprising distinction in this setting between interpolator networks of minimal rank and those of minimal Euclidean norm. Finally we perform a range of numerical experiments, which corroborate our theoretical findings.

# 1 Introduction

One of the grand challenges for machine learning research is to understand how overparameterised neural networks are able to fit perfectly the training examples and simultaneously to generalise well to unseen data [Zhang, Bengio, Hardt, Recht, and Vinyals, 2021]. The double-descent phenomenon [Belkin, Hsu, Ma, and Mandal, 2019], where increasing the neural network capacity beyond the interpolation threshold can eventually reduce the test loss much further than could be achieved around the underparameterised “sweet spot”, is a mystery from the standpoint of classical machine learning theory. This has been observed to happen even for training without explicit regularisers.

Implicit bias of gradient-based algorithms. A key hypothesis towards explaining the double-descent phenomenon is that the gradient-based algorithms that are used for training are implicitly biased (or implicitly regularised) [Neyshabur, Bhojanapalli, McAllester, and Srebro, 2017] to converge to solutions that in addition to fitting the training examples have certain properties which cause them to generalise well. It has attracted much attention in recent years from the research community, which has made substantial progress in uncovering implicit biases of training algorithms in many important settings [Vardi, 2023]. For example, for classification tasks, and for homogeneous networks (which is a wide class that includes ReLU networks provided they contain neither biases at levels deeper than the first nor residual connections), Lyu and Li [2020] and Ji and Telgarsky [2020] established that gradient flow is biased towards maximising the classification margin in parameter space, in the sense that once the training loss

gets sufficiently small, the direction of the parameters subsequently converges to a Karush-Kuhn-Tucker point of the margin maximisation problem.

Insights gained in this foundational research direction have not only shed light on overparameterised generalisation, but have been applied to tackle other central problems, such as the susceptibility of networks trained by gradient-based algorithms to adversarial examples [Vardi, Yehudai, and Shamir, 2022] and the possibility of extracting training data from network parameters [Haim, Vardi, Yehudai, Shamir, and Irani, 2022].

Regression tasks and initialisation scale. Showing the implicit bias for regression tasks, where the loss function is commonly mean square, has turned out to be more challenging than for classification tasks, where loss functions typically have exponential tails. A major difference is that, whereas most of the results for classification do not depend on how the network parameters are initialised, the scale of the initialisation has been observed to affect decisively the implicit bias of gradient-based algorithms for regression [Woodworth, Gunasekar, Lee, Moroshko, Savarese, Golan, Soudry, and Srebro, 2020]. When it is large so that the training follows the lazy regime, we tend to have fast convergence to a global minimum of the loss, however without an implicit bias towards sparsity and with limited generalisation [Jacot, Ged, Şimşek, Hongler, and Gabriel, 2021]. The focus, albeit at the price of uncertain convergence and lengthier training, has therefore been on the rich regime where the initialisation scale is small.

Considerable advances have been achieved for linear networks. For example, Azulay, Moroshko, Nacson, Woodworth, Srebro, Globerson, and Soudry [2021] and Yun, Krishnan, and Mobahi [2021] proved that gradient flow is biased to minimise the Euclidean norm of the predictor for one-hidden layer linear networks with infinitesimally small initialisation, and that the same holds also for deeper linear networks under an additional assumption on their initialisation. A related extensive line of work is on implicit bias of gradient-based algorithms for matrix factorisation and reconstruction, which has been a fruitful test-bed for regression using multi-layer networks. For example, Gunasekar, Woodworth, Bhojanapalli, Neyshabur, and Srebro [2017] proved that, under a commutativity restriction and starting from a small initialisation, gradient flow is biased to minimise the nuclear norm of the solution matrix; they also conjectured that the restriction can be dropped, which after a number of subsequent works was refuted by Li, Luo, and Lyu [2021], leading to a detailed analysis of both underparameterised and overparameterised regimes by Jin, Li, Lyu, Du, and Lee [2023].

For non-linear networks, such as those with the popular ReLU activation, progress has been difficult. Indeed, Vardi and Shamir [2021] showed that precisely characterising the implicit bias via a non-trivial regularisation function is impossible already for single-neuron one-hidden layer ReLU networks, and Timor, Vardi, and Shamir [2023] showed that gradient flow is not biased towards low-rank parameter matrices for multiple-output ReLU networks already with one hidden layer and small training datasets.

ReLU networks and training dynamics. We suggest that, in order to further substantially our knowledge of convergence, implicit bias, and generalisation for regression tasks using non-linear networks, we need to understand more thoroughly the dynamics throughout the gradient-based training. This is because of the observed strong influence that initialisation has on solutions, but is challenging due to the highly non-convex optimisation landscape. To this end, evidence and intuition were provided by Maennel, Bousquet, and Gelly [2018], Li et al. [2021], and Jacot et al. [2021], who conjectured that, from sufficiently small initialisations, after an initial phase where the neurons get aligned to a number of directions that depend only on the dataset, training causes the parameters to pass close to a sequence of saddle points, during which their rank increases gradually but stays low.

The first comprehensive analysis in this vein was accomplished by Boursier, Pillaud-Vivien, and Flammarion [2022], who focused on orthogonal datasets (which are therefore of cardinality less than or equal to the input dimension), and established that, for one-hidden layer ReLU networks, gradient flow from an infinitesimal initialisation converges to zero loss and is implicitly biased to minimise

the Euclidean norm of the network parameters. They also showed that, per sign class of the training labels (positive or negative), minimising the Euclidean norm of the interpolator networks coincides with minimising their rank.

Our contributions. We tackle the main challenge posed by Boursier et al. [2022], namely handling datasets that are not orthogonal. A major obstacle to doing so is that, whereas the analysis of the training dynamics in the orthogonal case made extensive use of an almost complete separation between a turning phase and a growth phase for all hidden neurons, non-orthogonal datasets cause considerably more complex dynamics in which hidden neurons follow training trajectories that simultaneously evolve their directions and norms [Boursier et al., 2022, Appendix A].

To analyse this involved dynamics in a reasonably clean setting, we consider the training of one-hidden layer ReLU networks by gradient flow from a small balanced initialisation on datasets that are labelled by a teacher ReLU neuron with which all the training points are correlated. More precisely, we assume that the angles between the training points and the teacher neuron are less than $\pi/4$ , which implies that all angles between training points are less than $\pi/2$ . The latter restriction has featured per label class in many works in the literature (such as by Phuong and Lampert [2021] and Wang and Pilanci [2022]), and the former is satisfied for example if the training points can be obtained by summing the teacher neuron $v^{*}$ with arbitrary vectors of length less than $\|v^{*}\|/\sqrt{2}$ . All our other assumptions are very mild, either satisfied with probability exponentially close to 1 by any standard random initialisation, or excluding corner cases of Lebesgue measure zero.

Our contributions can be summarised as follows.

- We provide a detailed non-asymptotic analysis of the dynamics of each hidden neuron throughout the training, and show that it applies whenever the initialisation scale $\lambda$ is below a precise bound which is polynomial in the network width $m$ and exponential in the training dataset cardinality $n$ . Moreover, our analysis applies for any input dimension $d > 1$ , for any $n \geq d$ (otherwise exact learning of the teacher neuron may not be possible), for any $m$ , and without assuming any specific random distribution for the initialisation. In particular, we demonstrate that the role of the overparameterisation in this setting is to ensure that initially at least one hidden neuron with a positive last-layer weight has in its active half-space at least one training point.   
- We show that, during a first phase of the training, all active hidden neurons with a positive last-layer weight get aligned to a single direction which is positively correlated with all training points, whereas all active hidden neurons with a negative last-layer weight get turned away from all training points so that they deactivate. In contrast to the orthogonal dataset case where the sets of training points that are in the active half-spaces of the neurons are essentially constant during the training, in our correlated setting this first phase in general consists, for each neuron, of a different sequence of stages during which the cardinality of the set of training points in its active half-space gradually increases or decreases, respectively.   
- We show that, during the rest of the training, the bundle of aligned hidden neurons with their last-layer weights, formed by the end of the first phase, grows and turns as it travels from near the origin to near the teacher neuron, and does not separate. To establish the latter property, which is the most involved part of this work, we identify a set in predictor space that depends only on $\lambda$ and the training dataset, and prove: first, that the trajectory of the bundle stays inside the set; and second, that this implies that the directional gradients of the individual neurons are such that the angles between them are non-increasing.   
- We prove that, after the training departs from the initial saddle, which takes time logarithmic in $\lambda$ and linear in $d$ , the gradient satisfies a Polyak-Łojasiewicz inequality and consequently the loss converges to zero exponentially fast.   
- We prove that, although for any fixed $\lambda$ the angles in the bundle of active hidden neurons do not

in general converge to zero as the training time tends to infinity, if we let $\lambda$ tend to zero then the networks to which the training converges have a limit: a network of rank 1, in which all non-zero hidden neurons are positive scalings of the teacher neuron and have positive last-layer weights. This establishes that gradient flow from an infinitesimal initialisation is implicitly biased to select interpolator networks of minimal rank. Note also that the limit network is identical in predictor space to the teacher neuron.

- We show that, surprisingly, among all networks with zero loss, there may exist some whose Euclidean norm is smaller than that of any network of rank 1. Moreover, we prove that this is the case if and only if a certain condition on angles determined by the training dataset is satisfied. This result might be seen as refuting the conjecture of Boursier et al. [2022, section 3.2] that the implicit bias to minimise Euclidean parameter norm holds beyond the orthogonal setting, and adding some weight to the hypothesis of Razin and Cohen [2020]. The counterexample networks in our proof have rank 2 and make essential use of the ReLU non-linearity.   
- We perform numerical experiments that indicate that the training dynamics and the implicit bias we theoretically established occur in practical settings in which some of our assumptions are relaxed. In particular, gradient flow is replaced by gradient descent with a realistic learning rate, the initialisation scales are small but not nearly as small as in the theory, and the angles between the teacher neuron and the training points are distributed around $\pi/4$ .

We further discuss related work, prove all theoretical results, and provide additional material on our experiments, in the appendix.

# 2 Preliminaries

Notation. We write: $[n]$ for the set $\{1,\ldots ,n\}$ , $\| \pmb {v}\|$ for the Euclidean length of a vector $\pmb {v},\overline{\pmb{v}}\coloneqq \pmb {v} / \| \pmb {v}\|$ for the normalised vector, $\angle (\pmb {v},\pmb {v}^{\prime})\coloneqq \arccos (\overline{\pmb{v}}^{\top}\overline{\pmb{v}}^{\prime})$ for the angle between $\pmb{v}$ and $\pmb {v}^{\prime}$ , and cone $\{\pmb {v}_1,\dots ,\pmb {v}_n\} \coloneqq \{\sum_{i = 1}^{n}\beta_{i}\pmb{v}_{i}\mid \beta_{1},\dots ,\beta_{n}\geq 0\}$ for the cone generated by vectors $\pmb {v}_1,\dots ,\pmb {v}_n$ .

One-hidden layer ReLU network. For an input $x \in R^{d}$ , the output of the network is

$$
h _ {\boldsymbol {\theta}} (\boldsymbol {x}) := \sum_ {j = 1} ^ {m} a _ {j}   \sigma (\boldsymbol {w} _ {j} ^ {\top} \boldsymbol {x})  ,
$$

where m is the width, the parameters $\boldsymbol{\theta} = (\boldsymbol{a}, \boldsymbol{W}) \in \mathbb{R}^{m} \times \mathbb{R}^{m \times d}$ consist of last-layer weights $a = [a_{1}, \ldots, a_{m}]$ and hidden-layer weights $W^{\top} = [w_{1}, \ldots, w_{m}]$ , and $\sigma(u) := \max\{u, 0\}$ is the ReLU function.

Balanced initialisation. For all $j \in [m]$ let

$$
\boldsymbol {w} _ {j} ^ {0} := \lambda   \boldsymbol {z} _ {j} \quad a _ {j} ^ {0} := s _ {j} \| \boldsymbol {w} _ {j} ^ {0} \|
$$

where $\lambda > 0$ is the initialisation scale, $z_j \in \mathbb{R}^d \setminus \{\mathbf{0}\}$ , and $s_j \in \{\pm 1\}$ .

A precise upper bound on $\lambda$ will be stated in Assumption 2.

We regard the initial unscaled hidden-layer weights $z_{j}$ and last-layer signs $s_{j}$ as given, without assuming any specific random distributions for them. For example, we might have that each $z_{j}$ consists of d independent centred Gaussians with variance $\frac{1}{dm}$ and each $s_{j}$ is uniform over $\{\pm1\}$ .

We consider only initialisations for which the layers are balanced, i.e. $|a_{j}^{0}| = \|w_{j}^{0}\|$ for all $j \in [m]$ . Since more generally each difference $(a_{j}^{t})^{2} - \|w_{j}^{t}\|^{2}$ is constant throughout training [Du, Hu, and Lee, 2018, Theorem 2.1] and we focus on small initialisation scales that tend to zero, this restriction (which is also present in Boursier et al. [2022]) is minor but simplifies our analysis.

Neuron-labelled correlated inputs. The teacher neuron $v^{*} \in R^{d}$ and the training dataset $\{(x_{i}, y_{i})\}_{i=1}^{n} \subseteq (\mathbb{R}^{d} \setminus \{\mathbf{0}\}) \times \mathbb{R}$ are such that for all i we have

$$
y _ {i} = \sigma (\pmb {v} ^ {* ^ {\top}} \pmb {x} _ {i}) \qquad \angle (\pmb {v} ^ {*}, \pmb {x} _ {i}) <   \pi / 4.
$$

In particular, since the angles between $v^{*}$ and the training points $x_{i}$ are acute, each label $y_{i}$ is positive.

To apply our results to a network with biases in the hidden layer and to a teacher neuron with a bias, one can work in dimension $d + 1$ and extend the training points to $\begin{bmatrix} x_{i} \\ 1 \end{bmatrix}$ .

Mean square loss gradient flow. For the regression task of learning the teacher neuron by the one-hidden layer ReLU network, we use the standard mean square empirical loss

$$
L (\pmb {\theta}) := \frac {1}{2 n} \sum_ {i = 1} ^ {n} (y _ {i} - h _ {\pmb {\theta}} (\pmb {x} _ {i})) ^ {2}.
$$

Our theoretical analysis concentrates on training by gradient flow, which from an initialisation as above evolves the network parameters by descending along the gradient of the loss by infinitesimal steps in continuous time [Li, Tai, and E, 2019]. Formally, we consider any parameter trajectory $\theta^t: [0, \infty) \to \mathbb{R}^m \times \mathbb{R}^{m \times d}$ that is absolutely continuous on every compact subinterval, and that satisfies the differential inclusion

$$
\mathrm{d} \boldsymbol {\theta} ^ {t} / \mathrm{d} t \in - \partial L (\boldsymbol {\theta} ^ {t}) \quad \text { for   almost   all } t \in [ 0, \infty) ,
$$

where $\partial L$ denotes the Clarke [1975] subdifferential of the loss function (which is locally Lipschitz).

We work with the Clarke subdifferential, which is a generalisation of the gradient, because the ReLU activation is not differentiable at 0, which causes non-differentiability of the loss function [Bolte, Daniilidis, Ley, and Mazet, 2010]. Although it follows from our results that, in our setting, the derivative of the ReLU can be fixed as $\sigma'(0) := 0$ like in the orthogonal case [Boursier et al., 2022, Appendix D], and the gradient flow trajectories are uniquely defined, that is not a priori clear; hence we work with the unrestricted Clarke subdifferential of the ReLU. We also remark that, in other settings, $\sigma'(0)$ cannot be fixed in this way due to gradient flow subtrajectories that correspond to gradient descent zig-zagging along a ReLU boundary (cf. e.g. Maennel et al. [2018, section 9.4]).

Basic observations. We establish the formulas for the derivatives of the last-layer weights and the hidden neurons; and that throughout the training, the signs of the last-layer weights do not change, and their absolute values track the norms of the corresponding hidden neurons. The latter property holds for all times t by continuity and enables us to focus the analysis on the hidden neurons.

Proposition 1. For all $j \in [m]$ and almost all $t \in [0, \infty)$ we have:

(i) $\mathrm{da}_j^t /\mathrm{dt} = \pmb{w}_j^t^\top \pmb{g}_j^t$ and $\mathrm{d}\pmb {w}_j^t /\mathrm{dt} = a_j^t\pmb {g}_j^t$ , where $\pmb {g}_j^t\in \frac{1}{n}\sum_{i = 1}^{n}(y_i - h_{\pmb{\theta}^t}(\pmb {x}_i))\partial \sigma (\pmb {w}_j^t^\top \pmb {x}_i)\pmb{x}_i;$   
(ii) $a_{j}^{t} = s_{j}\| \pmb{w}_{j}^{t}\| \neq 0.$

The definition in part (i) of the vectors $g_{j}^{t}$ that govern the dynamics is a membership because the subdifferential of the ReLU at 0 is the set of all values between 0 and 1, i.e. $\partial\sigma(0)=[0,1]$ .

# 3 Assumptions

To state our assumptions precisely, we introduce some additional notation. Let

$$
I _ {+} (\boldsymbol {v}) := \{i \in [ n ] | \boldsymbol {v} ^ {\top} \boldsymbol {x} _ {i} > 0 \} \quad I _ {0} (\boldsymbol {v}) := \{i \in [ n ] | \boldsymbol {v} ^ {\top} \boldsymbol {x} _ {i} = 0 \} \quad I _ {-} (\boldsymbol {v}) := \{i \in [ n ] | \boldsymbol {v} ^ {\top} \boldsymbol {x} _ {i} <   0 \}
$$

denote the sets of indices of training points that are, respectively, either inside or on the boundary or outside of the non-negative half-space of a vector v. Then let

$$
J _ {+} := \{j \in [ m ] \mid I _ {+} (\boldsymbol {z} _ {j}) \neq \emptyset \land s _ {j} = + 1 \} \quad J _ {-} := \{j \in [ m ] \mid I _ {+} (\boldsymbol {z} _ {j}) \neq \emptyset \land s _ {j} = - 1 \}
$$

be the sets of indices of hidden neurons that are initially active on at least one training point and whose last-layer signs are, respectively, positive or negative. Also let

$$
\boldsymbol {X} := \left[ \boldsymbol {x} _ {1}, \dots , \boldsymbol {x} _ {n} \right] \quad \boldsymbol {\gamma} _ {I} := \frac {1}{n} \sum_ {i \in I} y _ {i} \boldsymbol {x} _ {i}
$$

denote the matrix whose columns are all the training points, and the sum of all training points whose indices are in a set I, weighted by the corresponding labels and divided by n.

Moreover we define, for each $j \in J_{+} \cup J_{-}$ , a continuous trajectory $\alpha_{j}^{t}$ in $R^{d}$ by

$$
\boldsymbol {\alpha} _ {j} ^ {0} := \boldsymbol {z} _ {j} \quad \mathrm{d} \boldsymbol {\alpha} _ {j} ^ {t} / \mathrm{d} t := s _ {j} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \gamma_ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} \quad \text { for   all } t \in (0, \infty)  .
$$

Thus, starting from the unscaled initialisation $z_{j}$ of the corresponding hidden neuron, $\alpha_{j}^{t}$ follows a dynamics obtained from that of $w_{j}^{t}$ in Proposition 1 (i) and (ii) by replacing the vector $g_{j}^{t}$ by $\gamma_{I_{+}(\alpha_{j}^{t})}$ , which amounts to removing from $g_{j}^{t}$ the network output terms and the activation boundary summands. These trajectories will be useful as yardsticks in our analysis of the first phase of the training.

Assumption 1. (i) $d > 1$ , $\operatorname{span}\{\pmb{x}_1, \dots, \pmb{x}_n\} = \mathbb{R}^d$ , and $\| \pmb{v}^* \| = 1$ .

(ii) $J_{+} \neq \emptyset, I_{0}(\boldsymbol{z}_{j}) = \emptyset$ for all $j \in [m]$ , and $\angle (\boldsymbol{z}_{j}, \boldsymbol{\gamma}_{[n]}) > 0$ for all $j \in J_{-}$ .   
(iii) $\overline{\mathbf{x}}_1, \ldots, \overline{\mathbf{x}}_n$ are distinct, the eigenvalues of $\frac{1}{n} \mathbf{X} \mathbf{X}^\top$ are distinct, and $\mathbf{v}^*$ does not belong to a span of fewer than $d$ eigenvectors of $\frac{1}{n} \mathbf{X} \mathbf{X}^\top$ .   
(iv) $|I_0(\boldsymbol{\alpha}_j^t)| \leq 1$ for all $j \in J_+ \cup J_-$ and all $t \in [0, \infty)$ .   
(v) For all $j \in [m]$ and all $0 \leq T < T'$ , if for all $t \in (T, T')$ we have $I_{+}(\boldsymbol{w}_{j}^{t}) = I_{0}(\boldsymbol{w}_{j}^{T'}) \neq \emptyset$ and $I_{0}(\boldsymbol{w}_{j}^{t}) = I_{+}(\boldsymbol{w}_{j}^{T'}) = \emptyset$ , then for all $t \geq T'$ we have $w_{j}^{t} = w_{j}^{T'}$ .

This assumption is very mild. Part (i) excludes the trivial univariate case without biases (for univariate inputs with biases one needs d = 2), ensures that exact learning is possible, and fixes the length of the teacher neuron to streamline the presentation. Part (ii) assumes that, initially: at least one hidden neuron with a positive last-layer weight has in its active half-space at least one training point, no training point is at a ReLU boundary, and no hidden neuron with a negative last-layer weight is perfectly aligned with the $\gamma_{[n]}$ vector; this holds with probability at least $1 - (3/4)^{m}$ for any continuous symmetric distribution of the unscaled hidden-neuron initialisations, e.g. $z_{j} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \frac{1}{dm} I_{d})$ , and the uniform distribution of the last-layer signs $s_{j} \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}\{\pm 1\}$ . Parts (iii) and (iv) exclude corner cases of Lebesgue measure zero; observe that $\frac{1}{n} X X^{\top}$ is positive-definite, and that (iv) rules out a yardstick trajectory encountering two or more training points in its half-space boundary at exactly the same time. Part (v) excludes some unrealistic gradient flows that might otherwise be possible due to the use of the subdifferential: it specifies that, whenever a neuron deactivates (i.e. all training points exit its positive half-space), then it stays deactivated for the remainder of the training.

Before our next assumption, we define several further quantities. Let $\eta_{1} > \cdots > \eta_{d} > 0$ denote the eigenvalues of $\frac{1}{n} X X^{\top}$ , and let $u_{1}, \ldots, u_{d}$ denote the corresponding unit-length eigenvectors such that $v^{*} = \sum_{k=1}^{d} \nu_{k}^{*} u_{k}$ for some $\nu_{1}^{*}, \ldots, \nu_{d}^{*} > 0$ . Also, for each $j \in J_{+} \cup J_{-}$ , let $n_{j} := |I_{-s_{j}}(z_{j})|$ be the number of training points that should enter into or exit from the non-negative half-space along the trajectory $\alpha_{j}^{t}$ depending on whether the sign $s_{j}$ is positive or negative (respectively), and let

$$
\varphi_ {j} ^ {t} := \angle (\boldsymbol {\alpha} _ {j} ^ {t}, \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})}) \quad \text {   for   all   } t \in [ 0, \infty) \text {   such   that   } I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t}) \neq \emptyset
$$

be the evolving angle between $\alpha_{j}^{t}$ and the vector governing its dynamics (if any). Then the existence of the times at which the entries or the exits occur is confirmed in the following.

Proposition 2. For all $j \in J_{+} \cup J_{-}$ there exist a unique enumeration $i_{j}^{1}, \ldots, i_{j}^{n_{j}}$ of $I_{-s_{j}}(z_{j})$ and unique $0 = \tau_{j}^{0} < \tau_{j}^{1} < \cdots < \tau_{j}^{n_{j}}$ such that for all $\ell \in [n_{j}]$ :

(i) $I_{s_j}(\pmb{\alpha}_j^t) = I_{s_j}(\pmb{z}_j)\cup \{i_j^1,\dots ,i_j^{\ell -1}\}$ for all $t\in (\tau_j^{\ell -1},\tau_j^\ell)$ ;

(ii) $I_0(\pmb{\alpha}_j^t) = \emptyset$ for all $t\in (\tau_j^{\ell -1},\tau_j^\ell)$ , and $I_0\left(\pmb{\alpha}_j^{\tau_j^\ell}\right) = \{i_j^\ell \}$ .

Finally we define two measurements of the unscaled initialisation and the training dataset, which are positive thanks to Assumption 1, and which will simplify the presentation of our results.

$$
\delta := \min \left\{ \begin{array}{c} \min _ {i \in [ n ]} \| \boldsymbol {x} _ {i} \|,   \min _ {i, i ^ {\prime} \in [ n ]} \overline {{\boldsymbol {x}}} _ {i} ^ {\top}   \overline {{\boldsymbol {x}}} _ {i ^ {\prime}},   \min _ {k \in [ d - 1 ]} (\sqrt {\eta_ {k}} - \sqrt {\eta_ {k + 1}}) (d - 1),   \sqrt {\eta_ {d}}, \\ \min _ {k \in [ d ]} \nu_ {k} ^ {*} \sqrt {d},   \min _ {j \in [ m ]} \| \boldsymbol {z} _ {j} \|,   \min _ {j \in J _ {+}} \cos \varphi_ {j} ^ {0},   \min _ {j \in J _ {-}} \sin \varphi_ {j} ^ {0}, \\ \min \left\{| \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top}   \overline {{\boldsymbol {x}}} _ {i} |   \left| \begin{array}{c} j \in J _ {+} \cup J _ {-}   \wedge   \ell \in [ n _ {j} ] \\ \wedge   t \in [ \tau_ {j} ^ {\ell - 1}, \tau_ {j} ^ {\ell} ]   \wedge   i \in [ n ] \\ \wedge   i \neq i _ {j} ^ {\ell}   \wedge   (\ell \neq 1 \Rightarrow i \neq i _ {j} ^ {\ell - 1}) \end{array} \right. \right\},   \min _ {j \in J _ {-}} \overline {{\boldsymbol {\alpha}}} _ {j} ^ {0} ^ {\top}   \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {1}}, \\ \min \{\tau_ {j} ^ {\ell} - \tau_ {j} ^ {\ell - 1} | j \in J _ {+} \cup J _ {-}   \wedge   \ell \in [ n _ {j} ] \} \\ \end{array} \right\}
$$

$$
\Delta := \max \{\max _ {i \in [ n ]} \| \pmb {x} _ {i} \|, \max _ {j \in [ m ]} \| \pmb {z} _ {j} \|, 1 \}.
$$

Assumption 2. $0 < \varepsilon \leq \frac{1}{4}$ and $\lambda \leq \left(m n^{9n\Delta^2/\delta^3}\right)^{-3/\varepsilon}$ .

The quantity $\varepsilon$ introduced here has no effect on the network training, but is a parameter of our analysis, so that varying it within the assumed range tightens some of the resulting bounds while loosening others. The assumed bound on the initialisation scale $\lambda$ is polynomial in the network width m and exponential in the dataset cardinality n. The latter is also the case in Boursier et al. [2022], where the bound was stated informally and without its dependence on parameters other than m and n.

# 4 First phase: alignment or deactivation

We show that, for each initially active hidden neuron, if its last-layer sign is positive then it turns to include in its active half-space all training points that were initially outside, whereas if its last-layer sign is negative then it turns to remove from its active half-space all training points that were initially inside. Moreover, those training points cross the activation boundary in the same order as they cross the half-space boundary of the corresponding yardstick trajectory $\alpha_{j}^{t}$ , and at approximately the same times (cf. Proposition 2).

Lemma 3. For all $j \in J_{+} \cup J_{-}$ there exist unique $0 = t_j^0 < t_j^1 < \ldots < t_j^{n_j}$ such that for all $\ell \in [n_j]$ :

(i) $I_{s_j}(\pmb{w}_j^t) = I_{s_j}(\pmb{z}_j)\cup \{i_j^1,\dots ,i_j^{\ell -1}\}$ for all $t\in (t_j^{\ell -1},t_j^\ell)$ ;

(ii) $I_0(\boldsymbol{w}_j^t) = \emptyset$ for all $t\in (t_{j}^{\ell -1},t_{j}^{\ell})$ , and $I_0\Big(\boldsymbol{w}_j^{t_j^\ell}\Big) = \{i_j^\ell \}$

(iii) $|\tau_j^\ell - t_j^\ell| \leq \lambda^{1 - \left(1 + \frac{3\ell - 1}{3n_j}\right)\varepsilon}$ .

The preceding lemma is proved by establishing, for this first phase of the training, non-asymptotic upper bounds on the Euclidean norms of the hidden neurons and hence on the absolute values of the network outputs, and inductively over the stage index $\ell$ , on the distances between the unit-sphere normalisations of $\alpha_{j}^{t}$ and $w_{j}^{t}$ . Based on that analysis, we then obtain that each negative-sign hidden neuron does not grow from its initial length and deactivates by time $T_{0} := \max_{j \in J_{+} \cup J_{-}} \tau_{j}^{n_{j}} + 1$ .

Lemma 4. For all $j \in J_{-}$ we have:

$$
\| \boldsymbol {w} _ {j} ^ {T _ {0}} \| \leq \lambda \| \boldsymbol {z} _ {j} \| \quad \boldsymbol {w} _ {j} ^ {t} = \boldsymbol {w} _ {j} ^ {T _ {0}} \quad \text { for   all   } t \geq T _ {0}.
$$

We also obtain that, up to a later time $T_{1} := \varepsilon \ln(1/\lambda)/\|\gamma_{[n]}\|$ , each positive-sign hidden neuron: grows but keeps its length below $2\|z_{j}\|\lambda^{1-\varepsilon}$ , continues to align to the vector $\gamma_{[n]}$ up to a cosine of at least $1 - \lambda^{\varepsilon}$ , and maintains bounded by $\lambda^{1-3\varepsilon}$ the difference between the logarithm of its length divided by the initialisation scale and the logarithm of the corresponding yardstick vector length.

Lemma 5. For all $j \in J_{+}$ we have:

$$
\| \boldsymbol {w} _ {j} ^ {T _ {1}} \| <   2 \| \boldsymbol {z} _ {j} \| \lambda^ {1 - \varepsilon} \qquad \overline {{\boldsymbol {w}}} _ {j} ^ {T _ {1}} ^ {\top} \overline {{\boldsymbol {\gamma}}} _ {[ n ]} \geq 1 - \lambda^ {\varepsilon} \qquad | \ln \| \boldsymbol {\alpha} _ {j} ^ {T _ {1}} \| - \ln \| \boldsymbol {w} _ {j} ^ {T _ {1}} / \lambda \| | \leq \lambda^ {1 - 3 \varepsilon}.
$$

# 5 Second phase: growth and convergence

We next analyse the gradient flow subsequent to the deactivation of the negative-sign hidden neurons by time $T_{0}$ and the alignment of the positive-sign ones up to time $T_{1}$ , and establish that the loss converges to zero at a rate which is exponential and does not depend on the initialisation scale $\lambda$ .

Theorem 6. Under Assumptions 1 and 2, there exists a time $T_{2} < \ln (1 / \lambda)(4 + \varepsilon)d\Delta^{2} / \delta^{6}$ such that for all $t\geq 0$ we have $L(\pmb {\theta}^{T_2 + t}) < 0.5\Delta^2\mathrm{e}^{-t\cdot 0.4\delta^4 /\Delta^2}$ .

In particular, for $\varepsilon = 1/4$ and $\lambda = \left((mn^n)^{9\Delta^2/\delta^3}\right)^{-3/\varepsilon}$ (cf. Assumption 2), the first bound in Theorem 6 becomes $T_2 < (\ln m + n\ln n)d\cdot 17\cdot 27\Delta^4/\delta^9$ .

The proof of Theorem 6 is in large part geometric, with a key role played by a set $S := S_{1} \cup \cdots \cup S_{d}$ in predictor space, whose constituent subsets are defined as

$$
\mathcal {S} _ {\ell} := \left\{\boldsymbol {v} = \sum_ {k = 1} ^ {d} \nu_ {k} \boldsymbol {u} _ {k} \left| \bigwedge_ {1 \leq k <   \ell} \Omega_ {k} \wedge \Phi_ {\ell} \wedge \bigwedge_ {\ell \leq k <   k ^ {\prime} \leq d} \left(\Psi_ {k, k ^ {\prime}} ^ {\downarrow} \wedge \Psi_ {k, k ^ {\prime}} ^ {\uparrow}\right) \wedge \Xi \right. \right\},
$$

where the individual constraints are as follows (here $\eta_{0} := \infty$ so that e.g. $\frac{\eta_{1}}{2\eta_{0}} = 0$ ):

$$
\begin{array}{l} \Omega_ {k} \colon 1 <   \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \qquad \qquad \Phi_ {\ell} \colon \frac {\eta_ {\ell}}{2 \eta_ {\ell - 1}} <   \frac {\nu_ {\ell}}{\nu_ {\ell} ^ {*}} \leq 1 \qquad \qquad \Psi_ {k, k ^ {\prime}} ^ {\downarrow} \colon \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} <   \frac {\nu_ {k ^ {\prime}}}{\nu_ {k ^ {\prime}} ^ {*}} \\ \Xi \colon \overline {{\boldsymbol {v}}} ^ {\top} \overline {{\boldsymbol {X} \boldsymbol {X} ^ {\top} (\boldsymbol {v} ^ {*} - \boldsymbol {v})}} > \lambda^ {\varepsilon / 3} \quad \Psi_ {k, k ^ {\prime}} ^ {\uparrow} \colon \frac {\nu_ {k ^ {\prime}}}{\nu_ {k ^ {\prime}} ^ {*}} <   1 - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {\frac {1}{2} + \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}}. \\ \end{array}
$$

Thus S is connected, open, and constrained by $\Xi$ to be within the ellipsoid $\boldsymbol{v}^{\top}\boldsymbol{X}\boldsymbol{X}^{\top}(\boldsymbol{v}^{*}-\boldsymbol{v})=0$ which is centred at $\frac{v^{*}}{2}$ , with the remaining constraints slicing off further regions by straight or curved boundary surfaces.

In the most complex component of this work, we show that, for all $t \geq T_{1}$ , the trajectory of the sum $v^{t} := \sum_{j \in J_{+}} a_{j}^{t} w_{j}^{t}$ of the active hidden neurons weighted by the last layer stays inside S, and the cosines of the angles between the neurons remain above $1 - 4\lambda^{\varepsilon}$ . This involves proving that each face of the boundary of S is repelling for the training dynamics when approached from the inside; we remark that, although that is in general false for the entire boundary of the constraint $\Xi$ , it is in particular true for its remainder after the slicing off by the other constraints. We also show that all points in S are positively correlated with all training points, which together with the preceding facts implies that, during this second phase of the training, the network behaves approximately like a linear one-hidden layer one-neuron network. Then, as the cornerstone of the rest of the proof, we show that, for all $t \geq T_{2}$ ,

the gradient of the loss satisfies a Polyak-Lojasiewicz inequality $\|\nabla L(\boldsymbol{\theta}^{t})\|^{2} > \frac{2\eta_{d}\|\gamma_{[n]}\|}{5\eta_{1}}L(\boldsymbol{\theta}^{t})$ . Here $T_{2} := \inf\{t \geq T_{1} \mid \nu_{1}^{t}/\nu_{1}^{*} \geq 1/2\}$ is a time by which the network has departed from the initial saddle, more precisely when the first coordinate $\nu_{1}^{t}$ of the bundle vector $v^{t}$ with respect to the basis consisting of the eigenvectors of the matrix $\frac{1}{n}XX^{\top}$ crosses the half-way threshold to the first coordinate $\nu_{1}^{*}$ of the teacher neuron.

The interior of the ellipsoid in the constraint $\Xi$ actually consists of all vectors that have an acute angle with the derivative of the training dynamics in predictor space, and the “padding” of $\lambda^{\varepsilon/3}$ is present because the derivative of the bundle vector $v^{t}$ is “noisy” due to the latter being made up of the approximately aligned neurons. The remaining constraints delimit the subsets $S_{1},\ldots,S_{d}$ of the set S, through which the bundle vector $v_{t}$ passes in that order, with each unique “handover” from $S_{\ell}$ to $S_{\ell+1}$ happening exactly when the corresponding coordinate $\nu_{\ell}^{t}$ exceeds its target $\nu_{\ell}^{*}$ . The non-linearity of the constraints $\Psi_{k,k'}^{\uparrow}$ is needed to ensure the repelling for the training dynamics.

# 6 Implicit bias of gradient flow

Let us denote the set of all balanced networks by

$$
\Theta := \left\{\left(\boldsymbol {a}, \boldsymbol {W}\right) \in \mathbb {R} ^ {m} \times \mathbb {R} ^ {m \times d} \mid \forall j \in [ m ]: | a _ {j} | = \| \boldsymbol {w} _ {j} \| \right\}
$$

and the subset in which all non-zero hidden neurons are positive scalings of $v^{*}$ , have positive last-layer weights, and have lengths whose squares sum up to $\|v^{*}\|=1$ , by

$$
\Theta_ {\boldsymbol {v} ^ {*}} := \left\{(\boldsymbol {a}, \boldsymbol {W}) \in \Theta \mid \sum_ {j = 1} ^ {m} \| \boldsymbol {w} _ {j} \| ^ {2} = 1 \land \forall j \in [ m ]: \boldsymbol {w} _ {j} \neq \mathbf {0} \Rightarrow (\overline {{\boldsymbol {w}}} _ {j} = \boldsymbol {v} ^ {*} \land a _ {j} > 0) \right\}.
$$

Our main result establishes that, as the initialisation scale $\lambda$ tends to zero, the networks with zero loss to which the gradient flow converges tend to a network in $\Theta_{v^{*}}$ . The explicit subscripts indicate the dependence on $\lambda$ of the parameter vectors. The proof builds on the preceding results and involves a careful control of accumulations of approximation errors over lengthy time intervals.

Theorem 7. Under Assumptions 1 and 2, $L\left(\lim_{t \to \infty} \boldsymbol{\theta}_{\lambda}^{t}\right) = 0$ and $\lim_{\lambda \to 0^{+}} \lim_{t \to \infty} \boldsymbol{\theta}_{\lambda}^{t} \in \Theta_{\boldsymbol{v}^{*}}$ .

# 7 Interpolators with minimum norm

To compare the set $\Theta_{v^{*}}$ of balanced rank-1 interpolator networks with the set of all minimum-norm interpolator networks, in this section we focus on training datasets of cardinality d, we assume the network width is greater than 1 (otherwise the rank is necessarily 1), and we exclude the threshold case of Lebesgue measure zero where M = 0. The latter measurement of the training dataset is defined below in terms of angles between the teacher neuron and vectors in any two cones generated by different generators of the dual of the cone of all training points.

Let $[\chi_1, \ldots, \chi_d]^{\top} := X^{-1}$ and

$$
\mathcal {M} := \max \left\{\cos \angle (\boldsymbol {p}, \boldsymbol {q}) - \sin \angle (\boldsymbol {p}, \boldsymbol {v} ^ {*}) \left| \begin{array}{c} \emptyset \subsetneq K \subsetneq [ d ] \\ \wedge \boldsymbol {0} \neq \boldsymbol {p} \in \operatorname{cone} \{\boldsymbol {\chi} _ {k} \mid k \in K \} \\ \wedge \boldsymbol {0} \neq \boldsymbol {q} \in \operatorname{cone} \{\boldsymbol {\chi} _ {k} \mid k \notin K \} \end{array} \right. \right\}.
$$

Assumption 3. n = d, m > 1, and $M \neq 0$ .

We obtain that, surprisingly, $\Theta_{v^{*}}$ equals the set of all interpolators with minimum Euclidean norm if M < 0, but otherwise they are disjoint.

# Theorem 8. Under Assumptions 1 and 3:

(i) if $\mathcal{M} < 0$ then $\Theta_{\pmb{v}^*}$ is the set of all global minimisers of $\| \pmb{\theta}\|^2$ subject to $L(\pmb{\theta}) = 0$ ;   
(ii) if $\mathcal{M} > 0$ then no point in $\Theta_{\boldsymbol{v}^*}$ is a global minimiser of $\| \pmb{\theta}\|^2$ subject to $L(\pmb{\theta}) = 0$ .

For each of the two cases, we provide a family of example datasets in Appendix G. We remark that a sufficient condition for M < 0 to hold is that the inner product of any two distinct rows $\chi_{k}$ of the inverse of the dataset matrix X is non-positive, i.e. that the inverse of the Gram matrix of the dataset (in our setting this Gram matrix is positive) is a Z-matrix (cf. e.g. Fiedler and Pták [1962]). Also, if the training points were orthogonal then all the $\cos\angle(p,q)$ terms in the definition of M would be zero and consequently we would have M < 0; this is consistent with the result that, per sign class of the training labels in the orthogonal setting, minimising the Euclidean norm of interpolators coincides with minimising their rank [Boursier et al., 2022, Appendix C].

# 8 Experiments

We consider two schemes for generating the training dataset, where $S^{d-1}$ is the unit sphere in $R^{d}$ .

Centred: We sample $\mu$ from $\mathcal{U}(\mathbb{S}^{d-1})$ , then sample $x_{1},\ldots,x_{d}$ from $\mathcal{N}(\boldsymbol{\mu},\frac{\rho}{d}\boldsymbol{I}_{d})$ where $\rho=1$ , and finally set $v^{*}=\mu$ . This distribution has the property that, in high dimensions, the angles between the teacher neuron $v^{*}$ and the training points $x_{i}$ concentrate around $\pi/4$ . We exclude rare cases where some of these angles exceed $\pi/2$ .

Uncentred: This is the same, except that we use $\rho = \sqrt{2} - 1$ , sample one extra point $x_{0}$ , and finally set $v^{*} = \overline{x}_{0}$ . Here the angles between $v^{*}$ and $x_{i}$ also concentrate around $\pi/4$ in high dimensions, but the expected distance between $v^{*}$ and $\mu$ is $\sqrt{\rho}$ .

For each of the two dataset schemes, we train a one-hidden layer ReLU network of width m = 200 by gradient descent with learning rate 0.01, from a balanced initialisation such that $z_{j} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mathbf{0}, \frac{1}{dm} \mathbf{I}_{d})$ and $s_{j} \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}\{\pm 1\}$ , and for a range of initialisation scales $\lambda$ and input dimensions d.

We present in Figure 1 some results from considering initialisation scales $\lambda = 4^{2}, 4^{1}, \ldots, 4^{-12}, 4^{-13}$ and input dimensions $d = 4, 16, 64, 256, 1024$ , where we train until the number of iterations reaches $2 \cdot 10^{7}$ or the loss drops below $10^{-9}$ . The plots are in line with Theorem 7, showing how the maximum angle between active hidden neurons at the end of the training decreases with $\lambda$ .

Figure 2 on the left illustrates the exponential convergence of the training loss (cf. Theorem 6), and on the right how the implicit bias can result in good generalisation. The test loss is computed over an input distribution which is different from that of the training points, namely we sample 64 test inputs from the standard multivariate $\mathcal{N}(\mathbf{0},\mathbf{I}_{d})$ . These plots are for initialisation scales $\lambda=4^{-2},4^{-3},\ldots,4^{-7},4^{-8}$ .

# 9 Conclusion

We provided a detailed analysis of the dynamics of training a shallow ReLU network by gradient flow from a small initialisation for learning a single neuron which is correlated with the training points, establishing convergence to zero loss and implicit bias to rank minimisation in parameter space. We believe that in particular the geometric insights we obtained in order to deal with the complexities of the multi-stage alignment of hidden neurons followed by the simultaneous evolution of their norms and directions, will be useful to the community in the ongoing quest to understand implicit bias of gradient-based algorithms for regression tasks using non-linear networks.

A major direction for future work is to bridge the gap between, on one hand, our assumption that the angles between the teacher neuron and the training points are less than $\pi/4$ , and the other, the

![](images/dc6f14f75e013e023a82796d9729a94d29bfad7f05ea27648e5a3bfd33e670dc.jpg)  
- $d = 4$ - $d = 16$ - $d = 64$ - $\star$ - $d = 256$ - $\diamond$ - $d = 1024$

Figure 1: Dependence of the maximum angle between active hidden neurons on the initialisation scale $\lambda$ , for two generation schemes of the training dataset and a range of input dimensions, at the end of the training. Both axes are logarithmic, and each point plotted shows the median over five trials.   
![](images/eae456b80e02f21a7106d151d5963dd896acadc19c32bccada612cc2723c5459.jpg)  
$\lambda = 4^{-2}$ $\lambda = 4^{-3}$ $\lambda = 4^{-4}$ $\lambda = 4^{-5}$ $\lambda = 4^{-6}$ $\lambda = 4^{-7}$ $\lambda = 4^{-8}$

Figure 2: Evolution of the training loss, and of an outside distribution test loss, during training for an example centred training dataset in dimension 16 and width 200. The horizontal axes, logarithmic for the training loss and linear for the test loss, show iterations. The vertical axes are logarithmic.

assumption of Boursier et al. [2022] that the training points are orthogonal, while keeping a fine granularity of description. We expect this to be difficult because it seems to require handling bundles of approximately aligned neurons which may have changing sets of training points in their active half-spaces and which may separate during the training. However, it should be straightforward to extend our results to orthogonally separable datasets and two teacher ReLU neurons, where each of the latter has an arbitrary sign, labels one of the two classes of training points, and has angles less than $\pi/4$ with them; the gradient flow would then pass close to a second saddle point, where the labels of one of the classes have been nearly fitted but the hidden neurons that will fit the labels of the other class are still small. We report on related numerical experiments in Appendix I.

We also obtained a condition on the dataset that determines whether rank minimisation and Euclidean norm minimisation for interpolator networks coincide or are distinct. Although this dichotomy remains true if the $\pi/4$ correlation bound is relaxed to $\pi/2$ , the implicit bias of gradient flow in that extended setting is an open question.

Other directions for future work include considering multi-neuron teacher networks, student networks with more than one hidden layer, further non-linear activation functions, and gradient descent instead of gradient flow; also refining the bounds on the initialisation scale and the convergence time.

# Acknowledgments and Disclosure of Funding

We acknowledge the Centre for Discrete Mathematics and Its Applications at the University of Warwick for partial support, and the Scientific Computing Research Technology Platform at the University of Warwick for providing the compute cluster on which the experiments presented in this paper were run.

# References

Shahar Azulay, Edward Moroshko, Mor Shpigel Nacson, Blake E. Woodworth, Nathan Srebro, Amir Globerson, and Daniel Soudry. On the Implicit Bias of Initialization Shape: Beyond Infinitesimal Mirror Descent. In ICML, pages 468–477, 2021. 2   
Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine-learning practice and the classical bias-variance trade-off. Proc. Natl. Acad. Sci., 116(32):15849–15854, 2019. 1   
Jérôme Bolte, Aris Daniilidis, Olivier Ley, and Laurent Mazet. Characterizations of Łojasiewicz inequalities: subgradient flows, talweg, convexity. Trans. Amer. Math. Soc., 362(6):3319–3363, 2010. 5   
Etienne Boursier and Nicolas Flammarion. Penalising the biases in norm regularisation enforces sparsity. CoRR, abs/2303.01353, 2023. Accepted to NeurIPS 2023. 16   
Etienne Boursier, Loucas Pillaud-Vivien, and Nicolas Flammarion. Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs. In NeurIPS, 2022. 2, 3, 4, 5, 7, 10, 12, 15   
Frank H. Clarke. Generalized gradients and applications. Trans. Amer. Math. Soc., 205:247-262, 1975. 5   
Damek Davis, Dmitriy Drusvyatskiy, Sham M. Kakade, and Jason D. Lee. Stochastic Subgradient Method Converges on Tame Functions. Found. Comput. Math., 20(1):119–154, 2020. 17   
Simon S. Du, Wei Hu, and Jason D. Lee. Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are Automatically Balanced. In NeurIPS, pages 382–393, 2018. 4   
Matthias Englert and Ranko Lazić. Adversarial Reprogramming Revisited. In NeurIPS, 2022. 16   
Tolga Ergen and Mert Pilanci. Convex Geometry and Duality of Over-parameterized Neural Networks. J. Mach. Learn. Res., 22(212):1–63, 2021. 16

Mathieu Even, Scott Pesme, Suriya Gunasekar, and Nicolas Flammarion. (S)GD over Diagonal Linear Networks: Implicit Regularisation, Large Stepsizes and Edge of Stability. CoRR, abs/2302.08982, 2023. Accepted to NeurIPS 2023. 16   
Miroslav Fiedler and Vlastimil Pták. On matrices with non-positive off-diagonal elements and positive principal minors. Czechoslovak Mathematical Journal, 12(3):382–400, 1962. 10   
Spencer Frei, Yuan Cao, and Quanquan Gu. Agnostic Learning of a Single Neuron with Gradient Descent. In NeurIPS, 2020. 15   
Spencer Frei, Gal Vardi, Peter L. Bartlett, and Nathan Srebro. Benign Overfitting in Linear Classifiers and Leaky ReLU Networks from KKT Conditions for Margin Maximization. In COLT, pages 3173-3228, 2023a. 16   
Spencer Frei, Gal Vardi, Peter L. Bartlett, and Nathan Srebro. The Double-Edged Sword of Implicit Bias: Generalization vs. Robustness in ReLU Networks. CoRR, abs/2303.01456, 2023b. Accepted to NeurIPS 2023. 16   
Spencer Frei, Gal Vardi, Peter L. Bartlett, Nathan Srebro, and Wei Hu. Implicit Bias in Leaky ReLU Networks Trained on High-Dimensional Data. In ICLR, 2023c. 16   
Suriya Gunasekar, Blake E. Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Implicit Regularization in Matrix Factorization. In NeurIPS, pages 6151–6159, 2017. 2   
Niv Haim, Gal Vardi, Gilad Yehudai, Ohad Shamir, and Michal Irani. Reconstructing Training Data From Trained Neural Networks. In NeurIPS, 2022. 2   
Arthur Jacot, François Ged, Berfin Şimşek, Clément Hongler, and Franck Gabriel. Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity. CoRR, abs/2106.15933, 2021. 2   
Arnulf Jentzen and Adrian Riekert. Convergence analysis for gradient flows in the training of artificial neural networks with ReLU activation. J. Math. Anal. Appl., 517(2):126601, 2023. 16   
Ziwei Ji and Matus Telgarsky. Directional convergence and alignment in deep learning. In NeurIPS, 2020. 1, 16   
Jikai Jin, Zhiyuan Li, Kaifeng Lyu, Simon S. Du, and Jason D. Lee. Understanding Incremental Learning of Gradient Descent: A Fine-grained Analysis of Matrix Sensing. In ICML, pages 15200–15238, 2023. 2   
Sangmin Lee, Byeongsu Sim, and Jong Chul Ye. Magnitude and Angle Dynamics in Training Single ReLU Neurons. CoRR, abs/2209.13394, 2022. 16   
Qianxiao Li, Cheng Tai, and Weinan E. Stochastic Modified Equations and Dynamics of Stochastic Gradient Algorithms I: Mathematical Foundations. J. Mach. Learn. Res., 20(40):1–47, 2019. 5   
Zhiyuan Li, Yuping Luo, and Kaifeng Lyu. Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning. In ICLR, 2021. 2   
Kaifeng Lyu and Jian Li. Gradient Descent Maximizes the Margin of Homogeneous Neural Networks. In ICLR, 2020. 1, 16   
Kaifeng Lyu, Zhiyuan Li, Runzhe Wang, and Sanjeev Arora. Gradient Descent on Two-layer Nets: Margin Maximization and Simplicity Bias. In NeurIPS, pages 12978–12991, 2021. 15, 16   
Hartmut Maennel, Olivier Bousquet, and Sylvain Gelly. Gradient Descent Quantizes ReLU Network Features. CoRR, abs/1803.08367, 2018. 2, 5, 15   
Odelia Melamed, Gilad Yehudai, and Gal Vardi. Adversarial Examples Exist in Two-Layer ReLU Networks for Low Dimensional Data Manifolds. CoRR, abs/2303.00783, 2023. Accepted to NeurIPS 2023. 17   
Hancheng Min, René Vidal, and Enrique Mallada. Early Neuron Alignment in Two-layer ReLU Networks with Small Initialization. CoRR, abs/2307.12851, 2023. 15   
Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro. Exploring Generalization in Deep Learning. In NeurIPS, pages 5947–5956, 2017. 1

Greg Ongie, Rebecca Willett, Daniel Soudry, and Nathan Srebro. A Function Space View of Bounded Norm Infinite Width ReLU Nets: The Multivariate Case. In ICLR, 2020. 16   
Scott Pesme and Nicolas Flammarion. Saddle-to-Saddle Dynamics in Diagonal Linear Networks. CoRR, abs/2304.00488, 2023. Accepted to NeurIPS 2023. 16   
Mary Phuong and Christoph H. Lampert. The inductive bias of ReLU networks on orthogonally separable data. In ICLR, 2021. 3   
B.T. Polyak. Gradient methods for the minimisation of functionals. USSR Computational Mathematics and Mathematical Physics, 3(4):864–878, 1963. 9   
Noam Razin and Nadav Cohen. Implicit Regularization in Deep Learning May Not Be Explainable by Norms. In NeurIPS, 2020. 4   
Roei Sarussi, Alon Brutzkus, and Amir Globerson. Towards Understanding Learning in Neural Networks with Linear Teachers. In ICML, pages 9313–9322, 2021. 16   
Pedro Savarese, Itay Evron, Daniel Soudry, and Nathan Srebro. How do infinite width bounded norm networks look in function space? In COLT, pages 2667-2690, 2019. 16   
Lawrence Stewart, Francis Bach, Quentin Berthet, and Jean-Philippe Vert. Regression as Classification: Influence of Task Formulation on Neural Network Features. In AISTATS, pages 11563–11582, 2023. 16   
Nadav Timor, Gal Vardi, and Ohad Shamir. Implicit Regularization Towards Rank Minimization in ReLU Networks. In ALT, pages 1429–1459, 2023. 2   
Gal Vardi. On the Implicit Bias in Deep-Learning Algorithms. Commun. ACM, 66(6):86–93, 2023. 1   
Gal Vardi and Ohad Shamir. Implicit Regularization in ReLU Networks with the Square Loss. In COLT, pages 4224-4258, 2021. 2   
Gal Vardi, Gilad Yehudai, and Ohad Shamir. Learning a Single Neuron with Bias Using Gradient Descent. In NeurIPS, pages 28690–28700, 2021. 15, 16   
Gal Vardi, Gilad Yehudai, and Ohad Shamir. Gradient Methods Provably Converge to Non-Robust Networks. In NeurIPS, 2022. 2   
Mingze Wang and Chao Ma. Early Stage Convergence and Global Convergence of Training Mildly Parameterized Neural Networks. In NeurIPS, 2022. 15   
Yifei Wang and Mert Pilanci. The Convex Geometry of Backpropagation: Neural Network Gradient Flows Converge to Extreme Points of the Dual Convex Program. In ICLR, 2022. 3   
Blake E. Woodworth, Suriya Gunasekar, Jason D. Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and Rich Regimes in Overparametrized Models. In COLT, pages 3635–3673, 2020. 2   
Weihang Xu and Simon Du. Over-Parameterization Exponentially Slows Down Gradient Descent for Learning a Single Neuron. In COLT, pages 1155–1198, 2023. 15, 16   
Gilad Yehudai and Ohad Shamir. Learning a Single Neuron with Gradient Methods. In COLT, pages 3756-3786, 2020. 15   
Chulhee Yun, Shankar Krishnan, and Hossein Mobahi. A unifying view on implicit bias in training linear neural networks. In ICLR, 2021. 2   
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Commun. ACM, 64(3):107–115, 2021. 1

# Contents

1 Introduction 1   
2 Preliminaries 4   
3 Assumptions 5   
4 First phase: alignment or deactivation 7   
5 Second phase: growth and convergence 8   
6 Implicit bias of gradient flow 9   
7 Interpolators with minimum norm 9   
8 Experiments 10   
9 Conclusion 10

Acknowledgments and Disclosure of Funding 12

References 12

A Related work 15   
B Proof for the preliminaries 17   
C Proofs for the assumptions 17   
D Proofs for the first phase 27   
E Proofs for the second phase 37   
F Proofs for the implicit bias 49   
G Proofs and examples for the interpolators 51   
H Additional information about the experiments 56   
I Further experiments 69

# A Related work

Here we further discuss a selection of related work.

Early training phase. Our analysis of the first phase of the training builds on those of Maennel et al. [2018], who considered unrestricted datasets but focused on asymptotic results; and of Boursier et al. [2022], who restricted the datasets to orthogonal but provided detailed non-asymptotic bounds. In particular, our normalised yardstick trajectories $\overline{\alpha}_{j}^{t}$ are analogous to $\vec{U}_{i}(t)$ in Maennel et al. [2018, section 9.6] and $\widetilde{w}_{j}^{t}$ in Boursier et al. [2022, Appendix B.6]. Our contribution in this part, in relation to these two works, is to extend the fine-grained description for the orthogonal case to the more involved correlated case, which exhibits a sequence of intermediate stages, obtaining detailed non-asymptotic bounds including for the initialisation scale $\lambda$ ; the latter are essential for our analysis of the subsequent second phase of the training and our proof of the implicit bias when $\lambda$ tends to zero.

Non-asymptotic bounds for early training of one-hidden layer networks were shown by Lyu, Li, Wang, and Arora [2021] with the Leaky-ReLU non-linearity, logistic loss, and linearly separable data which are either symmetric or have a principal direction and uniformly labelled support vectors; and by Min, Vidal, and Mallada [2023] with the ReLU non-linearity, exponential loss, and orthogonally separable data. Wang and Ma [2022] studied early training by gradient descent of one-hidden layer ReLU networks, focusing on a non-balanced random initialisation and on obtaining a lower bound for the Euclidean norm of the gradient. Another related work is by Xu and Du [2023], where the hidden layer is trained, every last-layer weight is fixed to 1, and the loss is over a Gaussian data population; consequently some aspects of the training dynamics are simpler, and already the first phase aligns the neurons to the teacher.

Learning a single neuron. A number of previous works studied learning a single neuron by gradient-based algorithms, in settings including realisable without bias [Yehudai and Shamir, 2020], agnostic and noisy [Frei, Cao, and Gu, 2020], realisable with bias [Vardi, Yehudai, and Shamir, 2021], multi

layer [Lee, Sim, and Ye, 2022], and overparameterised [Xu and Du, 2023]. In particular, Vardi et al. [2021] proved exponentially fast convergence of gradient descent to the global minimum by geometric and algebraic arguments, for two complementary sets of assumptions on the data distribution and the student initialisation; Xu and Du [2023] determined that, when the student network has width at least 2 and only its hidden layer is trained, the speed of convergence drops to cubic; and Lee et al. [2022] studied how neuron depth and initialisation scale affect the speed of convergence. In contrast to the settings in those works, our student network has arbitrary width and both its layers are trained.

Convergence for one-hidden layer ReLU networks and mean square loss. Further related convergence results were obtained by Jentzen and Riekert [2023], who proved that, for one-hidden layer ReLU networks trained by gradient flow with respect to a mean square population loss, if the data is one-dimensional, the target function is affine, the initial loss is sufficiently small, and the training trajectory is bounded, then it converges to zero loss; and that if moreover the network is width-one then the boundedness assumption is not needed.

Also with univariate data, Stewart, Bach, Berthet, and Vert [2023] compared the features learnt by one-hidden layer ReLU networks for the square loss and the cross-entropy loss, postulating that sparseness in the regression case may cause optimisation difficulties, and reporting synthetic experiments that support the claim.

Properties transferred from parameter space to function space. The implicit bias in parameter space that we established, namely to interpolator networks of rank 1, has a clear implication in function space, namely the resulting function is identical to that defined by the teacher neuron. However, we also compared that set of interpolator networks with the one obtained by minimising the Euclidean norm. The question of what functions the latter networks define is in general non-trivial; for one-hidden layer ReLU networks, it was studied in the univariate case by Savarese, Evron, Soudry, and Srebro [2019] and Ergen and Pilanci [2021], and in the multivariate case by Ongie, Willett, Soudry, and Srebro [2020]. More recently, Boursier and Flammarion [2023] investigated further the univariate case, elucidating the consequences of whether the norm takes into account the bias terms.

Regression using diagonal linear networks. Even, Pesme, Gunasekar, and Flammarion [2023] and Pesme and Flammarion [2023] considered implicit bias for regression tasks, focusing on diagonal networks with linear activation. The former proved convergence and compared implicit biases of gradient descent and stochastic gradient descent with large learning rates. The latter studied gradient flow from a vanishing initialisation and provided a full description of training trajectories, showing that they jump from saddle to saddle until reaching the minimum $\ell_{1}$ -norm solution.

Classification using Leaky-ReLU networks. Building on the implicit bias to margin maximisation in parameter space for homogenous networks [Lyu and Li, 2020, Ji and Telgarsky, 2020], convergence to a linear classifier was shown by Lyu et al. [2021], Sarussi, Brutzkus, and Globerson [2021], and Frei, Vardi, Bartlett, Srebro, and Hu [2023c] for one-hidden layer networks with the Leaky-ReLU activation and several sets of assumptions that include linear separability of the data. Frei, Vardi, Bartlett, and Srebro [2023a] subsequently established that in two kinds of distributional settings benign overfitting occurs, namely the predictors interpolate noisy training data and simultaneously generalise well to unseen test data.

Implicit bias and adversarial examples. In addition to Vardi et al. [2021], likewise in the context of one-hidden layer ReLU networks and exponentially-tailed loss functions, consequences for adversarial examples of the implicit bias to margin maximisation in parameter space were investigated by Englert and Lazić [2022], who showed that for orthogonally separable training datasets it may prevent adversarial reprogrammability; and by Frei, Vardi, Bartlett, and Srebro [2023b], who showed that for

clustered data it leads to non-robust solutions even though robust networks that fit the data exist. In contrast to those works, which focus on gradient flow, Melamed, Yehudai, and Vardi [2023] considered possibly stochastic gradient descent and established that, when data belongs to a low-dimensional linear subspace, the training produces non-robust solutions, but decreasing the initialisation scale or adding a Euclidean norm regulariser increases robustness to orthogonal adversarial perturbations.

# B Proof for the preliminaries

First we note that the gradient flow ensures that the loss monotonically decreases, at the rate equal to the square of the Euclidean norm of the gradient.

Proposition 9 (by Davis, Drusvyatskiy, Kakade, and Lee [2020, Lemma 5.2]). For almost all $t \in [0, \infty)$ we have $\mathrm{d}L(\pmb{\theta}^t) / \mathrm{d}t = -\| \mathrm{d}\pmb{\theta}^t / \mathrm{d}t\|^2$ .

Then we show the following proposition from section 2.

Proposition 1. For all $j \in [m]$ and almost all $t \in [0, \infty)$ we have:

(i) $\mathrm{da}_j^t /\mathrm{dt} = \pmb{w}_j^t^\top \pmb{g}_j^t$ and $\mathrm{d}\pmb {w}_j^t /\mathrm{dt} = a_j^t\pmb {g}_j^t$ , where $\pmb {g}_j^t\in \frac{1}{n}\sum_{i = 1}^{n}(y_i - h_{\pmb{\theta}^t}(\pmb {x}_i))\partial \sigma (\pmb {w}_j^t^\top \pmb {x}_i)\pmb{x}_i;$   
(ii) $a_{j}^{t} = s_{j} \|w_{j}^{t}\| \neq 0.$

Proof. Part (i) follows by straightforward calculations. For part (ii), by (i), for all $j \in [m]$ and almost all $t \in [0, \infty)$ we have

$$
\mathrm{d} (a _ {j} ^ {t}) ^ {2} / \mathrm{d} t = 2 a _ {j} ^ {t} \boldsymbol {w} _ {j} ^ {\top} \boldsymbol {g} _ {j} ^ {t} = \mathrm{d} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} / \mathrm{d} t,
$$

and so $(a_j^t)^2 - \| w_j^t\|^2$ is constant and therefore zero for all $t$ by the initialisation and continuity. It remains to show that $a_j^t \neq 0$ for all $t$ . Observe that, by Proposition 9, for almost all $t$ , provided $a_j^t \neq 0$ we have

$$
\mathrm{d} \ln (a _ {j} ^ {t}) ^ {2} / \mathrm{d} t \geq - 2 \| \boldsymbol {g} _ {j} ^ {t} \| \geq - 2 \sqrt {2 L (\boldsymbol {\theta} ^ {t})} \max _ {i = 1} ^ {n} \| \boldsymbol {x} _ {i} \| \geq - 2 \sqrt {2 L (\boldsymbol {\theta} ^ {0})} \max _ {i = 1} ^ {n} \| \boldsymbol {x} _ {i} \|.
$$

Hence, by continuity, for all $t$ we have

$$
(a _ {j} ^ {t}) ^ {2} \geq (a _ {j} ^ {0}) ^ {2} \exp \left(- 2 t \sqrt {2 L (\boldsymbol {\theta} ^ {0})} \max _ {i = 1} ^ {n} \| \boldsymbol {x} _ {i} \|\right).
$$

Also from Proposition 1 we obtain the formulas for the derivatives of the spherical coordinates of the hidden neurons, i.e. of their logarithmic Euclidean norms and their unit normalisations.

Corollary 10. For all $j \in [m]$ and almost all $t \in [0, \infty)$ we have

$$
\mathrm{d} \ln \| \boldsymbol {w} _ {j} ^ {t} \| / \mathrm{d} t = s _ {j} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t}
$$

$$
\mathrm{d} \overline {{\boldsymbol {w}}} _ {j} ^ {t} / \mathrm{d} t = s _ {j} (\boldsymbol {g} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t})  .
$$

# C Proofs for the assumptions

In this section we prove Proposition 2 from section 3, which is subsumed by Proposition 14 below, and then show as Corollary 15 that the cases excluded by Assumption 1 (iv) have Lebesgue measure zero, as claimed in section 3.

We first establish several elementary properties of the yardstick trajectories $\alpha_{j}^{t}$ .

Proposition 11. For all $j \in J_{+} \cup J_{-}$ , the following statements hold.

(i) $\gamma_{I_+(\boldsymbol{\alpha}_j^t)} = \mathbf{0}$ if and only if $I_+(\boldsymbol{\alpha}_j^t) = \emptyset$ .

(ii) $(\pmb{\alpha}_j^t)^\top \pmb{\gamma}_{I_+(\pmb{\alpha}_j^t)} \geq 0$ . Moreover, $(\pmb{\alpha}_j^t)^\top \pmb{\gamma}_{I_+(\pmb{\alpha}_j^t)} = 0$ if and only if $(\pmb{\alpha}_j^t)^\top \pmb{x}_i \leq 0$ for all $i \in [n]$ .   
(iii) If $(\boldsymbol{\alpha}_j^t)^\top \boldsymbol{\gamma}_{I_+(\boldsymbol{\alpha}_j^t)} = \| \boldsymbol{\alpha}_j^t\| \| \boldsymbol{\gamma}_{I_+(\boldsymbol{\alpha}_j^t)}\|$ for some $t$ , then $I_+(\boldsymbol{\alpha}_j^t) = \emptyset$ or $I_+(\boldsymbol{\alpha}_j^t) = [n]$ .   
(iv) $\alpha_{j}^{t}\neq \mathbf{0}$ for all $t\in [0,\infty)$ .

Proof. For part (i), only the left-to-right implication requires a proof. Let $\boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t})} = \frac{1}{n} \sum_{i} y_{i} \boldsymbol{x}_{i}$ , where the summation is over $i \in I_{+}(\boldsymbol{\alpha}_{j}^{t})$ ; then $\|\boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}\|^{2} = \frac{1}{n^{2}} \sum_{i,k} y_{i} y_{k} \boldsymbol{x}_{i}^{\top} \boldsymbol{x}_{k}$ . Since all training points are positively correlated, and all coefficients $y_{i}$ that are present in the sum are positive, $\|\boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}\|^{2} > 0$ unless the sum contains no terms; that is, unless $I_{+}(\boldsymbol{\alpha}_{j}^{t}) = \emptyset$ .

For part (ii), observe that $\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}$ is a positive linear combination of $x_{k}$ for $k \in I_{+}(\boldsymbol{\alpha}_{j}^{t})$ , and every such $x_{k}$ forms an acute angle with $\alpha_{j}^{t}$ ; thus, $(\boldsymbol{\alpha}_{j}^{t})^{\top}\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})} > 0$ unless $\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})} = \mathbf{0}$ .

For part (iii), we again use the definition of $\gamma_{I_+(\alpha_j^t)}$ and the fact that $x_i^\top x_k > 0$ for all $i, k \in [n]$ . If $I_+(\alpha_j^t) \neq \emptyset$ , then $\alpha_j^t \neq 0$ and $x_i^\top \alpha_j^t = x_i^\top \gamma_{I_+(\alpha_j^t)} \cdot \| \alpha_j^t \| / \| \gamma_{I_+(\alpha_j^t)} \| > 0$ for all $i$ , and thus $I_+(\alpha_j^t) = [n]$ .

For part (iv), note that

$$
\frac {\mathrm{d} \| \boldsymbol {\alpha} _ {j} ^ {t} \| ^ {2}}{\mathrm{d} t} = 2 (\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \cdot s _ {j} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} = \| \boldsymbol {\alpha} _ {j} ^ {t} \| ^ {2} \| \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} \| \cdot 2 s _ {j} \cos \angle (\boldsymbol {\alpha} _ {j} ^ {t}, \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})}),
$$

where the final product is 0 if $\alpha_{j}^{t} = 0$ . It follows that $\mathrm{d}\| \pmb{\alpha}_{j}^{t}\|^ {2} / \mathrm{d}t\geq -2\| \pmb{\alpha}_{j}^{t}\|^ {2}\| \pmb{\gamma}_{[n]}\|$ for all $t\in [0,\infty)$ . By Gronwall's inequality, $\| \pmb{\alpha}_j^t\| ^2\geq \| \pmb{\alpha}_j^0\| ^2\mathrm{e}^{-2\| \pmb{\gamma}_{[n]}\| t} > 0$ for all $t$ , and it remains to recall that $\pmb{\alpha}_{j}^{0} = z_{j}$ was initially chosen to be non-zero.

The next proposition establishes continuity and monotonicity properties, assuming that the trajectory $\alpha_{j}^{t}$ crosses between regions of continuity of the right-hand side of the ODE finitely many times.

Proposition 12. Let $j \in J_{+} \cup J_{-}$ . Let $T > 0$ be such that $I_0(\boldsymbol{\alpha}_j^t)$ is only non-empty for finitely many $t \in [0, T]$ . Then the following statements hold.

(i) The map $t \mapsto (\pmb{\alpha}_j^t)^\top \pmb{\gamma}_{I_+(\pmb{\alpha}_j^t)}$ from $[0, T]$ to $\mathbb{R}$ is continuous.   
(ii) If $\pmb{\gamma}_{I_+(\pmb{\alpha}_j^t)} = \mathbf{0}$ for some $t\in [0,T]$ , then $t = T$ .

$$
(i i i) \gamma_ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} = \left\{ \begin{array}{l l} \lim _ {\xi \to t ^ {-}} \gamma_ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {\xi})} & \text {if s_{j} = +1 and t\in(0,T]}, \\ \lim _ {\xi \to t ^ {+}} \gamma_ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {\xi})} & \text {if s_{j} = -1 and t\in[0,T)}. \end{array} \right.
$$

(iv) For each $i \in [n]$ , $s_j(\boldsymbol{\alpha}_j^t)^\top \boldsymbol{x}_i$ is a strictly increasing function of $t$ on $[0, T]$ . Furthermore, for all $i \in [n]$ , whenever $0 \leq t_1 < t_2 \leq T$ :

- if $s_j = +1$ and $i \in I_0(\boldsymbol{\alpha}_j^{t_1}) \cup I_+(\boldsymbol{\alpha}_j^{t_1})$ , then $i \in I_+(\boldsymbol{\alpha}_j^{t_2})$ ;   
- if $s_j = -1$ and $i \notin I_+(\alpha_j^{t_1})$ , then $i \notin I_0(\alpha_j^{t_2}) \cup I_+(\alpha_j^{t_2})$ .

(v) Let $t_0 \in [0, T)$ be either 0 or a point of discontinuity of $\gamma_{I_+(\alpha_j^t)}$ . If $s_j = -1$ , then $\lim_{t \to t_0^+} \cos \varphi_j^t \neq 1$ .

We remark that the assumption of Proposition 12 that the set $\{t \geq 0 \mid I_0(\boldsymbol{\alpha}_j^t) \neq \emptyset\}$ has a finite intersection with $[0, T]$ will be justified in Proposition 14, in the following sense: we will show that the assumption holds for every segment $[0, T]$ such that $\gamma_{I_+(\boldsymbol{\alpha}_j^t)} \neq \mathbf{0}$ for all $t \in [0, T)$ .

Proof. For part (i), since the map $t \mapsto \alpha_j^t$ from $[0, \infty)$ to $\mathbb{R}^d$ is continuous by definition, it suffices to consider points of discontinuity of $\gamma_{I_+(\alpha_j^t)}$ . The set $I_0(\alpha_j^{t_0})$ is necessarily non-empty at every such point

$t_{0} \in [0, T]$ . Let us fix such a $t_{0}$ ; then one-sided limits of $\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}$ as $t \to t_{0}^{+}$ and $t \to t_{0}^{-}$ exist (excepting $t \to 0^{-}$ and $t \to T^{+}$ , which we do not consider). Here we used the assumption from the statement of the proposition: in a small enough neighbourhood of $t_{0}$ there are no other points t for which $I_{0}(\boldsymbol{\alpha}_{j}^{t})$ is non-empty; this assumption could have been avoided if necessary. The value of $(\boldsymbol{\alpha}_{j}^{t_{0}})^{\top}\boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t_{0}})}$ may only differ from (either of) the one-sided limits of $(\boldsymbol{\alpha}_{j}^{t})^{\top}\boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}$ as $t \to t_{0}^{\pm}$ by the summands $(\boldsymbol{\alpha}_{j}^{t_{0}})^{\top} \cdot \frac{1}{n}y_{i}\boldsymbol{x}_{i}$ with $i \in I_{0}(\boldsymbol{\alpha}_{j}^{t_{0}})$ . But every such summand is equal to 0 anyway by the definition of $I_{0}$ .

Before establishing part (ii), we first prove a weaker version of part (iv), namely non-strict monotonicity: for each $i \in [n]$ , $s_j(\boldsymbol{\alpha}_j^t)^\top \boldsymbol{x}_i$ is a non-decreasing function of $t$ on $[0, T]$ . To this end, we consider the derivatives

$$
\frac {\mathrm{d} (\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {x} _ {i}}{\mathrm{d} t} = s _ {j} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \cdot (\boldsymbol {\gamma} _ {I + (\boldsymbol {\alpha} _ {j} ^ {t})}) ^ {\top} \boldsymbol {x} _ {i}
$$

inside each interval $(t_{1}, t_{2})$ on which $\boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}$ is continuous. Observe that $(\boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t})})^{\top}\boldsymbol{x}_{i} \geq 0$ for each $i \in [n]$ , since training points are pairwise positively correlated. Therefore, $s_{j}(\boldsymbol{\alpha}_{j}^{t})^{\top}\boldsymbol{x}_{i}$ is non-decreasing on $(t_{1}, t_{2})$ . Since $\alpha_{j}^{t}$ is continuous, and there are only finitely many points at which $\boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}$ is discontinuous, $s_{j}(\boldsymbol{\alpha}_{j}^{t})^{\top}\boldsymbol{x}_{i}$ is also non-decreasing on $[0, T]$ .

We now establish part (ii). If $\gamma_{I_+(\alpha_j^t)} = 0$ for some $t \in [0, T]$ , then

$$
t _ {0} := \inf \{t \in [ 0, T ] \mid I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t}) = \emptyset \}
$$

is well-defined by Proposition 11, part (i). Observe that if $I_{+}(\boldsymbol{\alpha}_{j}^{t_0})$ is non-empty, then $I_{+}(\boldsymbol{\alpha}_{j}^{t})$ is non-empty for all $t$ in a small neighbourhood of $t_0$ by the continuity of $\boldsymbol{\alpha}_{j}^{t}$ . Therefore, $I_{+}(\boldsymbol{\alpha}_{j}^{t_0}) = \emptyset$ by our choice of $t_0$ . At the same time, notice that $0 < t_0$ because we assume $j \in J_{+} \cup J_{-}$ . Furthermore, for all $t' < t_0$ there is some $i \in [n]$ with $(\boldsymbol{\alpha}_{j}^{t'})^\top \boldsymbol{x}_i > 0$ . By compactness and by the (non-strict) monotonicity property proved above, there exists a single $i \in [n]$ such that $(\boldsymbol{\alpha}_{j}^{t'})^\top \boldsymbol{x}_i > 0$ for all $t' < t_0$ . Once again by continuity, we have $(\boldsymbol{\alpha}_{j}^{t_0})^\top \boldsymbol{x}_i \geq 0$ . Since $I_{+}(\boldsymbol{\alpha}_{j}^{t_0}) = \emptyset$ , we conclude that $i \in I_0(\boldsymbol{\alpha}_{j}^{t_0})$ .

We have shown that, assuming $\gamma_{I_+(\alpha_j^t)} = \mathbf{0}$ for some $t \in [0,T]$ , the existence of $t_0 \in (0,t]$ such that $I_+(\alpha_j^{t_0}) = \emptyset$ and $I_0(\alpha_j^{t_0}) \neq \emptyset$ . It follows that $\alpha_j^t = \alpha_j^{t_0}$ and $I_0(\alpha_j^t) = I_0(\alpha_j^{t_0})$ for all $t \geq t_0$ . Under the assumptions of the proposition, this means $t_0 = t = T$ . This completes the proof of part (ii).

We now proceed to part (iv), proving strict monotonicity of each $s_j(\boldsymbol{\alpha}_j^t)^\top \boldsymbol{x}_i$ . By part (ii), for every interval $(t_1, t_2)$ on which $\gamma_{I_+(\boldsymbol{\alpha}_j^t)}$ is continuous, we have in fact $\gamma_{I_+(\boldsymbol{\alpha}_j^t)} \neq \mathbf{0}$ . Therefore, $(\gamma_{I_+(\boldsymbol{\alpha}_j^t)})^\top \boldsymbol{x}_i > 0$ , because training points are pairwise positively correlated, and each $s_j(\boldsymbol{\alpha}_j^t)^\top \boldsymbol{x}_i$ strictly increases on $(t_1, t_2)$ . Since $\boldsymbol{\alpha}_j^t$ is continuous, and there are only finitely many points at which $\gamma_{I_+(\boldsymbol{\alpha}_j^t)}$ is discontinuous, $s_j(\boldsymbol{\alpha}_j^t)^\top \boldsymbol{x}_i$ is also strictly increasing on $[0, T]$ . The two remaining implications in the statement of part (iv) follow.

Part (iii) is a consequence of part (iv).

To establish part (v), firstly observe that, by part (iii) and by the continuity of $\alpha_{j}^{t}$ , the function $\cos\varphi_{j}^{t}$ is in fact right-continuous at $t_{0}$ : the one-sided limit in question exists and is equal to $\cos\varphi_{j}^{t_{0}}$ . By Proposition 11, part (iii), if $\cos\varphi_{j}^{t_{0}}=1$ , then $I_{+}(\boldsymbol{\alpha}_{j}^{t_{0}})=[n]$ . We now consider two cases. If $t_{0}=0$ , then $\alpha_{j}^{t_{0}}=z_{j}$ , but this is ruled out by Assumption 1, part (ii). Otherwise $t_{0}$ is a point of discontinuity of $\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}$ . Since $\alpha_{j}^{t}$ is a continuous function of t, the set $I_{0}(\boldsymbol{\alpha}_{j}^{t_{0}})$ is necessarily non-empty, but this is also a contradiction because this set must be disjoint from $I_{+}(\boldsymbol{\alpha}_{j}^{t_{0}})$ .

The following proposition strengthens the previously proved statement (Proposition 11, part (iv)) that $\alpha_{j}^{t} \neq 0$ , bounding $\| \alpha_{j}^{t} \|$ from below. In the sequel, we will require analytic expressions for two related

quantities:

$$
\frac {\mathrm{d} \| \boldsymbol {\alpha} _ {j} ^ {t} \|}{\mathrm{d} t} = \frac {\mathrm{d} \sqrt {\| \boldsymbol {\alpha} _ {j} ^ {t} \| ^ {2}}}{\mathrm{d} t} = \frac {1}{2 \| \boldsymbol {\alpha} _ {j} ^ {t} \|} \cdot \frac {\mathrm{d} \| \boldsymbol {\alpha} _ {j} ^ {t} \| ^ {2}}{\mathrm{d} t} = \frac {2 (\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \cdot s _ {j} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})}}{2 \| \boldsymbol {\alpha} _ {j} ^ {t} \|} = s _ {j} (\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})},
$$

$$
\frac {\mathrm{d}}{\mathrm{d} t} \binom{\boldsymbol {\alpha} _ {j} ^ {t}}{\| \boldsymbol {\alpha} _ {j} ^ {t} \|} = \frac {\frac {\mathrm{d} \boldsymbol {\alpha} _ {j} ^ {t}}{\mathrm{d} t} \cdot \| \boldsymbol {\alpha} _ {j} ^ {t} \| - \frac {\mathrm{d} \| \boldsymbol {\alpha} _ {j} ^ {t} \|}{\mathrm{d} t} \cdot \boldsymbol {\alpha} _ {j} ^ {t}}{\| \boldsymbol {\alpha} _ {j} ^ {t} \| ^ {2}} = \frac {s _ {j} \| \boldsymbol {\alpha} _ {j} ^ {t} \| ^ {2} \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} - s _ {j} \cdot (\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} \cdot \boldsymbol {\alpha} _ {j} ^ {t}}{\| \boldsymbol {\alpha} _ {j} ^ {t} \| ^ {2}}.
$$

Proposition 13. Let $0 \leq t_1 < t_2$ be such that $I_0(\boldsymbol{\alpha}_j^t) = \emptyset$ for all $t \in (t_1, t_2)$ and $I_0(\boldsymbol{\alpha}_j^t) \neq \emptyset$ for at most finitely many $t \in [0, t_1]$ .

(i) If $t_1$ is either 0 or a point of discontinuity of $\gamma_{I_+(\alpha_j^t)}$ , then there is a constant $\mu > 0$ , only dependent on $t_1$ but not on $t$ or $t_2$ , such that $\| \boldsymbol{\alpha}_j^t \| \geq \mu$ for all $t \in (t_1, t_2)$ .   
(ii) Suppose $\gamma_{I_+(\alpha_j^t)} = \gamma$ for all $t \in (t_1, t_2)$ , and denote $\varphi_j^{t_1^+} := \angle(\alpha_j^{t_1}, \gamma) = \arccos\big((\overline{\alpha}_j^{t_1})^\top \overline{\gamma}\big)$ . Then for all $t \in (t_1, t_2)$ we have

$$
\| \pmb {\alpha} _ {j} ^ {t} \| = \frac {1}{2} \cdot (1 + s _ {j} \cos \varphi_ {j} ^ {t _ {1} ^ {+}}) \cdot \| \pmb {\alpha} _ {j} ^ {t _ {1}} \| \cdot \mathrm{e} ^ {\| \pmb {\gamma} \| (t - t _ {1})} +
$$

$$
\frac {1}{2} \cdot (1 - s _ {j} \cos \varphi_ {j} ^ {t _ {1} ^ {+}}) \cdot \| \boldsymbol {\alpha} _ {j} ^ {t _ {1}} \| \cdot \mathrm{e} ^ {- \| \boldsymbol {\gamma} \| (t - t _ {1})}.
$$

Proof. We establish part (ii) first. The functions $\| \pmb{\alpha}_j^t\|$ and $(\pmb{\alpha}_j^t)^\top \pmb{\gamma}$ satisfy, for all $t\in (t_1,t_2)$ , the following system of ordinary differential equations:

$$
\frac {\mathrm{d} \| \boldsymbol {\alpha} _ {j} ^ {t} \|}{\mathrm{d} t} = s _ {j} \cdot (\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma}, \quad \frac {\mathrm{d} \big ((\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} \big)}{\mathrm{d} t} = s _ {j} \| \boldsymbol {\gamma} \| ^ {2} \| \boldsymbol {\alpha} _ {j} ^ {t} \|.
$$

By the standard theory of linear ODEs, the solution can be sought in the form

$$
\left\| \boldsymbol {\alpha} _ {j} ^ {t} \right\| = c _ {1} \mathrm{e} ^ {\left\| \boldsymbol {\gamma} \right\| t} + c _ {2} \mathrm{e} ^ {- \left\| \boldsymbol {\gamma} \right\| t},
$$

$$
(\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} = (c _ {1} \mathrm{e} ^ {\| \boldsymbol {\gamma} \| t} - c _ {2} \mathrm{e} ^ {- \| \boldsymbol {\gamma} \| t}) \cdot s _ {j} \| \boldsymbol {\gamma} \|.
$$

The constants $c_{1}$ and $c_{2}$ are chosen based on the initial conditions as $t \to t_1^+$ , i.e., they should satisfy the following system of linear equations:

$$
\left[ \begin{array}{c c} \mathrm{e} ^ {\| \boldsymbol {\gamma} \| t _ {1}} & \mathrm{e} ^ {- \| \boldsymbol {\gamma} \| t _ {1}} \\ \mathrm{e} ^ {\| \boldsymbol {\gamma} \| t _ {1}} & - \mathrm{e} ^ {- \| \boldsymbol {\gamma} \| t _ {1}} \end{array} \right] \left[ \begin{array}{c} c _ {1} \\ c _ {2} \end{array} \right] = \left[ \begin{array}{c} \| \pmb {\alpha} _ {j} ^ {t _ {1}} \| \\ s _ {j} (\pmb {\alpha} _ {j} ^ {t _ {1}}) ^ {\top} \overline {{\pmb {\gamma}}} \end{array} \right].
$$

Here we rely on the continuity of $t \mapsto \alpha_j^t$ . We obtain

$$
\left\| \boldsymbol {\alpha} _ {j} ^ {t} \right\| = \frac {1}{2} \cdot \left(\left\| \boldsymbol {\alpha} _ {j} ^ {t _ {1}} \right\| + s _ {j} \left(\boldsymbol {\alpha} _ {j} ^ {t _ {1}}\right) ^ {\top} \overline {{\boldsymbol {\gamma}}}\right) \cdot \mathrm{e} ^ {\left\| \boldsymbol {\gamma} \right\| \left(t - t _ {1}\right)} +
$$

$$
\frac {1}{2} \cdot \left(\| \boldsymbol {\alpha} _ {j} ^ {t _ {1}} \| - s _ {j} (\boldsymbol {\alpha} _ {j} ^ {t _ {1}}) ^ {\top} \overline {{\boldsymbol {\gamma}}}\right) \cdot \mathrm{e} ^ {- \| \boldsymbol {\gamma} \| (t - t _ {1})},
$$

which can then be rewritten in the required form.

We now establish part (i). By the continuity of dot products $(\boldsymbol{\alpha}_{j}^{t})^{\top}\boldsymbol{x}_{i}$ , there exists a vector $\gamma\in R^{d}$ such that $\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}=\gamma$ for all $t\in(t_{1},t_{2})$ . In the degenerate case, $\gamma=0$ , we have $\alpha_{j}^{t}=\alpha_{j}^{t_{1}}$ for all $t\in(t_{1},t_{2})$ . Hence, we can choose $\mu:=\|\alpha_{j}^{t_{1}}\|$ , which is positive by Proposition 11, part (iv). We will therefore assume $\gamma\neq0$ . The idea is to rely on Proposition 13, part (ii), noting that $\cos\varphi_{j}^{t_{1}^{+}}=\lim_{t\to t_{1}^{+}}(\overline{\boldsymbol{\alpha}}_{j}^{t})^{\top}\overline{\boldsymbol{\gamma}}=(\overline{\boldsymbol{\alpha}}_{j}^{t_{1}})^{\top}\overline{\boldsymbol{\gamma}}\geq0$ , by Proposition 11, part (ii), and by continuity of $\alpha_{j}^{t}$ . So, if $s_{j}=+1$ ,

then clearly $\|\alpha_{j}^{t}\|\geq\frac{1}{2}\|\alpha_{j}^{t_{1}}\|=: \mu$ . If $s_{j}=-1$ , then, again dropping the second term in the closed-form expression for $\|\alpha_{j}^{t}\|$ , we obtain $\|\alpha_{j}^{t}\|\geq\frac{1}{2}\|\alpha_{j}^{t_{1}}\|\cdot(1-\lim_{t\to t_{1}^{+}}\cos\varphi_{j}^{t})\cdot1$ . By Proposition 12, part (v), $\lim_{t\to t_{1}^{+}}\cos\varphi_{j}^{t}<1$ , which completes the proof.

Proposition 14. For all $j \in J_{+} \cup J_{-}$ there exist a unique enumeration $i_j^1, \ldots, i_j^{n_j}$ of $I_{-s_j}(z_j)$ and unique $\tau_j^1, \ldots, \tau_j^{n_j} \in [0, \infty)$ such that for all $\ell \in [n_j]$ the following hold, where $\tau_j^0 := 0$ , $\varphi_j^{(\ell - 1)^+} := \lim_{t \to (\tau_j^{\ell - 1})^+} \varphi_j^t$ , $\varphi_j^{\ell^-} := \lim_{t \to (\tau_j^{\ell})^-} \varphi_j^t$ , and

$$
I _ {j} ^ {\ell} := \left\{ \begin{array}{l l} I _ {+} (\boldsymbol {z} _ {j}) \cup \{i _ {j} ^ {1}, \ldots , i _ {j} ^ {\ell - 1} \} & \text {if s_{j} = 1}, \\ I _ {+} (\boldsymbol {z} _ {j}) \setminus \{i _ {j} ^ {1}, \ldots , i _ {j} ^ {\ell - 1} \} & \text {if s_{j} = -1}: \end{array} \right.
$$

(i) $i_{j}^{\ell} = \arg\min\left\{-s_{j}\left(\overline{\boldsymbol{\alpha}}_{j}^{\tau_{j}^{\ell-1}}\right)^{\top}\overline{\boldsymbol{x}}_{i}\Big/ \overline{\boldsymbol{\gamma}}_{I_{j}^{\ell}}^{\top}\overline{\boldsymbol{x}}_{i}\middle| i \in I_{-s_{j}}(\boldsymbol{z}_{j}) \setminus \{i_{j}^{1},\ldots,i_{j}^{\ell-1}\}\right\};$

(ii) $\sin \left( \varphi_j^{(\ell - 1)^+} - \varphi_j^{\ell^-} \right) / \sin \varphi_j^{\ell^-} = -\left( \overline{\alpha}_j^{\tau_j^{\ell - 1}} \right)^{\top} \overline{\boldsymbol{x}}_{i_j^\ell} / \overline{\gamma}_{I_j^\ell}^\top \overline{\boldsymbol{x}}_{i_j^\ell};$

(iii) $\tau_{j}^{\ell -1} <   \tau_{j}^{\ell};$

(iv) $I_{+}(\pmb{\alpha}_{j}^{t}) = I_{j}^{\ell}$ for all $t\in (\tau_j^{\ell -1},\tau_j^\ell)$ ;

(v) $I_0(\pmb{\alpha}_j^t) = \emptyset$ for all $t\in (\tau_j^{\ell -1},\tau_j^\ell)$ , and $I_0\left(\pmb{\alpha}_j^{\tau_j^\ell}\right) = \{i_j^\ell \}$ ;

(vi) $\cos \varphi_{j}^{t} = \tanh\left(\operatorname{artanh}\cos \varphi_{j}^{(\ell -1)^{+}} + s_{j}\left\| \boldsymbol{\gamma}_{I_{j}^{\ell}}\right\| (t - \tau_{j}^{\ell -1})\right)$ for all $t\in (\tau_j^{\ell -1},\tau_j^\ell)$ ;

(vii) if $\ell < n_j$ then $\left\| \boldsymbol{\gamma}_{I_j^\ell} \right\| \cos \varphi_j^{\ell^-} = \left\| \boldsymbol{\gamma}_{I_j^{\ell + 1}} \right\| \cos \varphi_j^{\ell^+}$ ;

(viii) if $\ell = n_j$ and $s_j = 1$ then $\left\| \boldsymbol{\gamma}_{I_j^\ell} \right\| \cos \varphi_j^{\ell^-} = \| \boldsymbol{\gamma}_{[n]} \| \cos \lim_{t \to (\tau_j^\ell)^+} \varphi_j^t$ ;

(ix) if $\ell = n_j$ and $s_j = -1$ then $\varphi_j^{\ell^-} = \pi /2$ ;

$(x)\overline{\boldsymbol{\alpha}}_{j}^{t} = \left(\sin (\varphi_{j}^{t})\overline{\boldsymbol{\alpha}}_{j}^{\tau_{j}^{\ell -1}} + \sin \left(\varphi_{j}^{(\ell -1)^{+}} - \varphi_{j}^{t}\right)\overline{\boldsymbol{\gamma}}_{I_{j}^{\ell}}\right)\Big/ \sin \varphi_{j}^{(\ell -1)^{+}}$ for all $t\in (\tau_j^{\ell -1},\tau_j^\ell)$ ;

$(xi)\ s_j \mathrm{d}\overline{\alpha}_j^{\top} \overline{\boldsymbol{x}}_i / \mathrm{d}t \geq \boldsymbol{\gamma}_{I_j^\ell}^\top \overline{\boldsymbol{x}}_i$ for all $i \notin I_j^\ell$ and all $t \in (\tau_j^{\ell-1}, \tau_j^\ell)$ ;

$(xii)$ if $s_j = -1$ , then $\mathrm{d}\overline{\alpha}_j^{t^\top}\overline{\mathbf{x}}_{i_j^\ell} / \mathrm{d}t < 0$ and $\mathrm{d}^2\overline{\alpha}_j^{t^\top}\overline{\mathbf{x}}_{i_j^\ell} / \mathrm{d}t^2 < 0$ for all $t\in (\tau_j^{\ell -1},\tau_j^\ell)$ .

Proof. Throughout, we let $j \in J_{+} \cup J_{-}$ stay fixed but arbitrary.

Parts (i), (iii), (iv), and (v). We establish these parts by a common inductive argument. The induction is on $\ell$ , ranging from 1 to $n_j$ . We do not separate the base case. We first notice that $\mathrm{d}((\boldsymbol{\alpha}_j^t)^\top \overline{\boldsymbol{x}}_i) / \mathrm{d}t = s_j \| \boldsymbol{\alpha}_j^t \| \cdot (\gamma_{I_+(\boldsymbol{\alpha}_j^t)})^\top \overline{\boldsymbol{x}}_i$ . In particular, for any fixed $i \in [n]$ we can write

$$
s _ {j} \left(\boldsymbol {\alpha} _ {j} ^ {\xi}\right) ^ {\top} \overline {{\boldsymbol {x}}} _ {i} = s _ {j} \left(\boldsymbol {\alpha} _ {j} ^ {\tau_ {j} ^ {\ell - 1}}\right) ^ {\top} \overline {{\boldsymbol {x}}} _ {i} + \int_ {\tau_ {j} ^ {\ell - 1}} ^ {\xi} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \cdot \left(\boldsymbol {\gamma} _ {I + \left(\boldsymbol {\alpha} _ {j} ^ {t}\right)}\right) ^ {\top} \overline {{\boldsymbol {x}}} _ {i} d t. \tag {1}
$$

This equality holds for any $\xi > \tau_{j}^{\ell-1}$ as long as the integral on the right-hand side is well-defined; we first need to justify the existence of an appropriate $\xi > \tau_{j}^{\ell-1}$ . This is not automatic because the function $\gamma_{I_{+}(\alpha_{j}^{t})}$ is not assumed continuous. We consider two cases.

If $\ell = 1$ , then $I_0\left(\boldsymbol{\alpha}_j^{\tau_j^{\ell - 1}}\right) = \emptyset$ by Assumption 1, part (ii), and thus $\left(\boldsymbol{\alpha}_j^{\tau_j^{\ell - 1}}\right)^\top \overline{\boldsymbol{x}}_i$ are all non-zero; by continuity of $\boldsymbol{\alpha}_j^t$ , this holds in a sufficiently small right-neighbourhood of $\tau_j^{\ell - 1}$ . Thus, if the set $\{t > \tau_j^{\ell - 1} \mid I_0(\boldsymbol{\alpha}_j^t) \neq \emptyset\}$ is non-empty, its infimum is strictly greater than $\tau_j^{\ell - 1}$ , and we can pick this infimum as $\xi$ . In fact, we will show below that the set cannot be empty, but for now let us say that it is safe to pick any $\xi > \tau_j^{\ell - 1}$ in this hypothetical situation.

Now suppose $\ell > 1$ , then by the inductive hypothesis $I_0\left(\boldsymbol{\alpha}_j^{\tau_j^{\ell - 1}}\right) = \{i_j^{\ell - 1}\}$ . For all $i \neq i_j^{\ell - 1}$ , we have $\left(\boldsymbol{\alpha}_j^{\tau_j^{\ell - 1}}\right)^\top \overline{\boldsymbol{x}}_i \neq 0$ and thus each $(\boldsymbol{\alpha}_j^t)^\top \overline{\boldsymbol{x}}_i$ maintains the sign in some right-neighbourhood of $\tau_j^{\ell - 1}$ . For $i = i_j^{\ell - 1}$ , rewrite Equation 1 as

$$
s _ {j} (\boldsymbol {\alpha} _ {j} ^ {\xi}) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} = \int_ {\tau_ {j} ^ {\ell - 1}} ^ {\xi} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \cdot (\boldsymbol {\gamma} _ {I + (\boldsymbol {\alpha} _ {j} ^ {t})}) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} \mathrm{d} t, \tag {2}
$$

where the integrand is non-negative. Therefore, the function $s_{j}(\boldsymbol{\alpha}_{j}^{\xi})^{\top}\overline{\boldsymbol{x}}_{i_{j}^{\ell-1}}$ is non-negative for all $\xi > \tau_{j}^{\ell-1}$ and moreover is non-decreasing. Consider the set $I_{j}^{\ell}$ defined in the statement of the proposition; we have $\emptyset \neq I_{j}^{\ell} = I_{+}(\boldsymbol{\alpha}_{j}^{t})$ and $(\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})})^{\top}\overline{\boldsymbol{x}}_{i_{j}^{\ell-1}} > 0$ for all t greater than $\tau_{j}^{\ell-1}$ in a small neighbourhood of $\tau_{j}^{\ell-1}$ . (Note that $\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})} \neq \mathbf{0}$ by Proposition 11, part (i).) So we can replace $\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}$ with $\gamma_{I_{j}^{\ell}}$ in Equation 2 if $\xi$ is close enough to $\tau_{j}^{\ell-1}$ ; we have now shown the existence of a $\xi > \tau_{j}^{\ell-1}$ such that $I_{+}(\boldsymbol{\alpha}_{j}^{t}) = \gamma_{I_{j}^{\ell}}$ for all $t \in (\tau_{j}^{\ell-1}, \xi)$ . (In fact, we can again choose $\xi := \inf\{t > \tau_{j}^{\ell-1} \mid I_{0}(\boldsymbol{\alpha}_{j}^{t}) \neq \emptyset\}$ .)

Having found an appropriate $\xi$ , let us observe that, by the inductive hypothesis (part (v)) and by the choice of $\xi$ , the set $I_{0}(\alpha_{j}^{t})$ is only non-empty for finitely many time points $t \in [0, \xi]$ . Let us consider

$$
I ^ {\prime} := \left\{i \in [ n ]   \middle |   s _ {j} \biggl (\boldsymbol {\alpha} _ {j} ^ {\tau_ {j} ^ {\ell - 1}} \biggr) ^ {\top} \overline {{\boldsymbol {x}}} _ {i} <   0 \right\} = I _ {- s _ {j}} (\boldsymbol {z} _ {j}) \setminus \{i _ {j} ^ {1}, \ldots , i _ {j} ^ {\ell - 1} \}  .
$$

Recalling that all training points are positively correlated, we see that $(\boldsymbol{\gamma}_{I_{j}^{\ell}})^{\top}\overline{\boldsymbol{x}}_{i}>0$ . By Proposition 13, part (i), the integrand in Equation 1 is lower-bounded by $\mu\cdot(\boldsymbol{\gamma}_{I_{j}^{\ell}})^{\top}\overline{\boldsymbol{x}}_{i}$ for all t. Therefore, for each $i\in I'$ the expression on the right-hand side of Equation 1 tends to $+\infty$ if we let, formally, $\xi\to+\infty$ . Since for $\xi=\tau_{j}^{\ell-1}$ each of the right-hand sides is negative if $i\in I'$ , there exists some $\xi>\tau_{j}^{\ell-1}$ and an $i\in I'$ for which the left-hand side, $(\boldsymbol{\alpha}_{j}^{\xi})^{\top}\overline{\boldsymbol{x}}_{i}$ , becomes 0. Rewriting Equation 1 as $\int_{\tau_{j}^{\ell-1}}^{\xi}\|\boldsymbol{\alpha}_{j}^{t}\|dt=r_{i}$ , where $r_{i}:=-s_{j}\left(\boldsymbol{\alpha}_{j}^{\tau_{j}^{\ell-1}}\right)^{\top}\overline{\boldsymbol{x}}_{i}/(\boldsymbol{\gamma}_{I_{j}^{\ell}})^{\top}\overline{\boldsymbol{x}}_{i}>0$ , we observe that the integral on the left-hand side does not depend on i. Thus, the smallest $\xi>\tau_{j}^{\ell-1}$ for which the integral is equal to $r_{i}$ for some $i\in I'$ is the earliest time point after $\tau_{j}^{\ell-1}$ at which the set $I_{0}(\boldsymbol{\alpha}_{j}^{\xi})$ becomes non-empty. This value of $\xi$ is then, by definition, $\tau_{j}^{\ell}$ . Since for $\xi=\tau_{j}^{\ell-1}$ the integral is zero and since $r_{i}>0$ for all $i\in I'$ , we also have $I_{0}(\boldsymbol{\alpha}_{j}^{\xi})=\{i_{j}^{\ell}\}$ where $i=i_{j}^{\ell}$ is the index of the smallest $r_{i}$ among $i\in I'$ ; this i is unique by Assumption 1, part (iv).

To complete the proof of the inductive step for part (i), it remains to note that rescaling each $r_{i}$ by a factor of $\left\|\gamma_{I_{j}^{\ell}}\right\|/\left\|\alpha_{j}^{\tau_{j}^{\ell-1}}\right\|$ does not change the arg min.

Notice that, for the current value of $\ell$ we have also justified the inequality $\tau_{j}^{\ell-1} < \tau_{j}^{\ell}$ of part (iii), as well as equalities $I_{+}(\boldsymbol{\alpha}_{j}^{t}) = I_{j}^{\ell}$ and $I_{0}(\boldsymbol{\alpha}_{j}^{t}) = \emptyset$ for all $t \in (\tau_{j}^{\ell-1}, \tau_{j}^{\ell})$ , and $I_{0}\left(\boldsymbol{\alpha}_{j}^{\tau_{j}^{\ell}}\right) = \{i_{j}^{\ell}\}$ , which together comprise parts (iv) and (v). This completes the inductive argument, proving parts (i), (iii), (iv), and (v).

Intermediate summary. We have already established uniqueness of the enumeration $i_{j}^{1},\ldots,i_{j}^{n_{j}}$ and time points $\tau_{j}^{1},\ldots,\tau_{j}^{n_{j}}$ : part (iii) requires that the latter be sorted in the ascending order, and our argument for the choice of $\tau_{j}^{\ell}$ makes it clear that there is always only one possibility, if we want to require (as parts (iv) and (v) do) that $I_{+}(\boldsymbol{\alpha}_{j}^{t})$ remain constant in between $\tau_{j}^{\ell-1}$ and $\tau_{j}^{\ell}$ , and $I_{0}\left(\boldsymbol{\alpha}_{j}^{\tau_{j}^{\ell}}\right)$ be non-empty. In addition, we now know that the assumption of Proposition 12 holds for every time segment [0,T] such that $\gamma_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}\neq\mathbf{0}$ for all $t\in[0,T)$ ; and in particular up to $T=\tau_{j}^{n_{j}}$ .

Parts (vii), (viii), and (ix). For part (vii), suppose $\ell < n_j$ . By Proposition 12, part (iii), both one-sided limits of $\cos \varphi_j^t$ as $t \to (\tau_j^\ell)^\pm$ exist. By part (iv) of the current proposition,

$$
\lim _ {t \to (\tau_ {j} ^ {\ell}) ^ {-}} \cos \varphi_ {j} ^ {t} = \frac {\left(\boldsymbol {\alpha} _ {j} ^ {\tau_ {j} ^ {\ell}}\right) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}}}{\left\| \boldsymbol {\alpha} _ {j} ^ {\tau_ {j} ^ {\ell}} \right\| \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\|} \quad \text {and} \quad \lim _ {t \to (\tau_ {j} ^ {\ell}) ^ {+}} \cos \varphi_ {j} ^ {t} = \frac {\left(\boldsymbol {\alpha} _ {j} ^ {\tau_ {j} ^ {\ell}}\right) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell + 1}}}{\left\| \boldsymbol {\alpha} _ {j} ^ {\tau_ {j} ^ {\ell}} \right\| \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell + 1}} \right\|},
$$

and we notice that the numerators are equal by Proposition 12, part (i). Multiplying each limit by the norm of the corresponding $\gamma$ , we obtain the desired equation.

Part (viii) follows from the same calculations in the case $\ell = n_j$ , where instead of $I_j^{\ell + 1}$ we use $I_j^{n_j} \cup \{i_j^{n_j}\} = [n]$ .

For part (ix) we observe that $I_j^{n_j} = \{i_j^{n_j}\}$ , so we have $I_+(\alpha_j^{\tau_j^{n_j}}) = \emptyset$ and $I_0(\alpha_j^{\tau_j^{n_j}}) = \{i_j^{n_j}\}$ , so indeed $\cos \varphi_j^t \to 0$ as $t \to (\tau_j^{n_j})^-$ .

Part (vi). We rely on the facts that $I_0(\boldsymbol{\alpha}_j^t) = \emptyset$ and that $\gamma_{I_+(\boldsymbol{\alpha}_j^t)} \equiv \gamma_{I_j^\ell}$ for all $t \in (\tau_j^{\ell-1}, \tau_j^\ell)$ , proved in parts (iv) and (v). Notice that

$$
\frac {\mathrm{d} \big ((\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \big)}{\mathrm{d} t} = s _ {j} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \big \| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \big \| ^ {2}
$$

and, using a previously obtained formula for $d\|\alpha_{j}^{t}\|/dt$ (just before Proposition 13),

$$
\begin{array}{l} \frac {\mathrm{d} \cos \varphi_ {j} ^ {t}}{\mathrm{d} t} = \frac {\mathrm{d}}{\mathrm{d} t} \left(\frac {(\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}}}{\| \boldsymbol {\alpha} _ {j} ^ {t} \| \cdot \| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \|}\right) \\ = \frac {\frac {\mathrm{d}}{\mathrm{d} t} \left(\left(\boldsymbol {\alpha} _ {j} ^ {t}\right) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}}\right) \cdot \left\| \boldsymbol {\alpha} _ {j} ^ {t} \right\| \cdot \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| - \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| \cdot \frac {\mathrm{d} \left\| \boldsymbol {\alpha} _ {j} ^ {t} \right\|}{\mathrm{d} t} \cdot \left(\boldsymbol {\alpha} _ {j} ^ {t}\right) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}}}{\left\| \boldsymbol {\alpha} _ {j} ^ {t} \right\| ^ {2} \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| ^ {2}} \\ = \frac {s _ {j} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \| ^ {2} \| \boldsymbol {\alpha} _ {j} ^ {t} \| \| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \| - \| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \| (\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \cdot s _ {j} (\boldsymbol {\alpha} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}}}{\| \boldsymbol {\alpha} _ {j} ^ {t} \| ^ {2} \| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \| ^ {2}} \\ = s _ {j} \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| \left(1 - \cos^ {2} \varphi_ {j} ^ {t}\right). \\ \end{array}
$$

Separating variables, we obtain

$$
\frac {\mathrm{d} \cos \varphi_ {j} ^ {t}}{1 - \cos^ {2} \varphi_ {j} ^ {t}} = s _ {j} \| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \| \mathrm{d} t,
$$

and so $\operatorname{artanh}\cos\varphi_{j}^{t}=s_{j}\left\|\gamma_{I_{j}^{\ell}}\right\|t+C$ for $t\in(\tau_{j}^{\ell-1},\tau_{j}^{\ell})$ , where the constant C is determined from the initial condition $\lim_{t\to(\tau_{j}^{\ell-1})^{+}}\operatorname{artanh}\cos\varphi_{j}^{t}=s_{j}\left\|\gamma_{I_{j}^{\ell}}\right\|\tau_{j}^{\ell-1}+C$ . The left-hand side is well-defined, since

$\cos \varphi_{j}^{(\ell -1)^{+}} \notin \{-1,1\}$ . Indeed, $\cos \varphi_{j}^{t} \geq 0$ for all $t$ by Proposition 11, part (ii), so the limit cannot be negative; it thus suffices to rule out the value 1. The case $s_{j} = -1$ is already handled in Proposition 12, part (v). For $s_{j} = +1$ , we observe that, if $\ell > 1$ , then $\cos \varphi_{j}^{(\ell -1)^{-}} > \cos \varphi_{j}^{(\ell -1)^{+}}$ by part (vii) of the current proposition, since $\| \boldsymbol{\gamma}_{I_j^{\ell -1}} \|^2 < \| \boldsymbol{\gamma}_{I_j^{\ell -1} \cup \{i_j^{\ell -1}\}} \|^2 = \| \boldsymbol{\gamma}_{I_j^\ell} \|^2$ thanks to the positive correlation between training points; hence $\cos \varphi_{j}^{(\ell -1)^{+}} < 1$ . Finally, $\ell = 1$ implies $\cos \varphi_{j}^{(\ell -1)^{+}} = \lim_{t \to 0^{+}} \cos \varphi_{j}^{t} = \cos \angle (\mathbf{z}_j, \gamma_{I_+(\alpha_j^0)}) = 1$ , and in this case $I_+(\alpha_j^0) = [n]$ by Proposition 11, part (iii). Hence, $I_{-s_j}(\mathbf{z}_j) = \emptyset$ and $n_j = 0$ , a contradiction. In conclusion, we have thus argued that $\cos \varphi_{j}^{(\ell -1)^{+}} \notin \{-1,1\}$ in all cases, so artanh $\cos \varphi_{j}^{t} = s_{j} \| \boldsymbol{\gamma}_{I_j^\ell} \| (t - \tau_j^{\ell -1}) + \operatorname{artanh} \cos \varphi_{j}^{(\ell -1)^{+}}$ and it remains to take the hyperbolic tangent on both sides of this equation to prove part (vi) for $t \in (\tau_j^{\ell -1}, \tau_j^\ell)$ .

Part (x). We rely on the result of part (vi). Recall the analytic expression for the derivative $d\overline{\alpha}_{j}^{t}/dt$ , obtained just before Proposition 13. Notice that, for all $t \in (\tau_{j}^{\ell-1}, \tau_{j}^{\ell})$ , the derivative $d\overline{\alpha}_{j}^{t}/dt$ belongs to the linear subspace spanned by vectors $\overline{\alpha}_{j}^{t}$ and $\overline{\gamma}_{I_{j}^{\ell}}$ . It follows that $\overline{\alpha}_{j}^{t}$ can be expressed as a linear combination of two fixed vectors, $\overline{\alpha}_{j}^{\tau_{j}^{\ell-1}}$ and $\overline{\gamma}_{I_{j}^{\ell}}$ . It is thus sufficient to check that the vector

$$
\boldsymbol {f} := \left(\sin (\varphi_ {j} ^ {t})   \overline {{\boldsymbol {\alpha}}} _ {j} ^ {\tau_ {j} ^ {\ell - 1}} + \sin \left(\varphi_ {j} ^ {(\ell - 1) ^ {+}} - \varphi_ {j} ^ {t}\right)   \overline {{\boldsymbol {\gamma}}} _ {I _ {j} ^ {\ell}}\right) \bigg / \sin \varphi_ {j} ^ {(\ell - 1) ^ {+}}
$$

has norm 1 and forms an angle of $\varphi_j^t$ with $\gamma_{I_j^\ell}$ . We have

$$
\begin{array}{l} \| \boldsymbol {f} \| ^ {2} = \left(\frac {\sin \varphi_ {j} ^ {t}}{\sin \varphi_ {j} ^ {(\ell - 1) ^ {+}}}\right) ^ {2} + \left(\frac {\sin \left(\varphi_ {j} ^ {(\ell - 1) ^ {+}} - \varphi_ {j} ^ {t}\right)}{\sin \varphi_ {j} ^ {(\ell - 1) ^ {+}}}\right) ^ {2} \\ + 2 \cdot \frac {\sin \varphi_ {j} ^ {t} \sin (\varphi_ {j} ^ {(\ell - 1) ^ {+}} - \varphi_ {j} ^ {t})}{\sin^ {2} \varphi_ {j} ^ {(\ell - 1) ^ {+}}} \cdot \cos \varphi_ {j} ^ {(\ell - 1) ^ {+}}. \\ \end{array}
$$

Denote $a = \varphi_j^t$ and $b = \varphi_j^{(\ell -1)^+} - \varphi_j^t$ , then

$$
\begin{array}{l} \| \boldsymbol {f} \| ^ {2} = \frac {\sin^ {2} a + \sin^ {2} b + 2 \sin a \sin b \cos (a + b)}{\sin^ {2} (a + b)} \\ = \frac {\sin^ {2} a + \sin^ {2} b + 2 \sin a \sin b (\cos a \cos b - \sin a \sin b)}{(\sin a \cos b + \cos a \sin b) ^ {2}} \\ = \frac {\sin^ {2} a + \sin^ {2} b + 2 \sin a \sin b (\cos a \cos b - \sin a \sin b)}{\sin^ {2} a \cos^ {2} b + 2 \sin a \cos b \cos a \sin b + \cos^ {2} a \sin^ {2} b} \\ = \frac {\sin^ {2} a + \sin^ {2} b + 2 \sin a \sin b \cos a \cos b - 2 \sin^ {2} a \sin^ {2} b}{\sin^ {2} a (1 - \sin^ {2} b) + 2 \sin a \cos b \cos a \sin b + (1 - \sin^ {2} a) \sin^ {2} b} \\ = 1. \\ \end{array}
$$

To verify the second claim, observe that

$$
\begin{array}{l} \boldsymbol {f} ^ {\top} \overline {{\boldsymbol {\gamma}}} _ {I _ {j} ^ {\ell}} = \frac {\sin \varphi_ {j} ^ {t}}{\sin \varphi_ {j} ^ {(\ell - 1) ^ {+}}} \cdot \left(\overline {{\boldsymbol {\alpha}}} _ {j} ^ {\tau_ {j} ^ {\ell - 1}}\right) ^ {\top} \overline {{\boldsymbol {\gamma}}} _ {I _ {j} ^ {\ell}} + \frac {\sin \left(\varphi_ {j} ^ {(\ell - 1) ^ {+}} - \varphi_ {j} ^ {t}\right)}{\sin \varphi_ {j} ^ {(\ell - 1) ^ {+}}} \cdot \left\| \overline {{\boldsymbol {\gamma}}} _ {I _ {j} ^ {\ell}} \right\| ^ {2} \\ = \frac {\sin \varphi_ {j} ^ {t} \cos \varphi_ {j} ^ {(\ell - 1) ^ {+}} + \sin \varphi_ {j} ^ {(\ell - 1) ^ {+}} \cos \varphi_ {j} ^ {t} - \cos \varphi_ {j} ^ {(\ell - 1) ^ {+}} \sin \varphi_ {j} ^ {t}}{\sin \varphi_ {j} ^ {(\ell - 1) ^ {+}}} \\ \end{array}
$$

$$
= \cos \varphi_ {j} ^ {t}.
$$

We must still check still that the vector f is on the correct side of $\overline{\gamma}_{I_{j}^{\ell}}$ : indeed, there are two arcs on the unit circle that connect the endpoint of vector $\overline{\gamma}_{I_{j}^{\ell}}$ with a point at arc length $\varphi_{j}^{t}$ away from it. However, this check is easy: for $t = \tau_{j}^{\ell-1}$ , only one of these arcs connects $\overline{\gamma}_{I_{j}^{\ell}}$ to $\overline{\alpha}_{j}^{\tau_{j}^{\ell-1}}$ , and we can see that $f \to \overline{\alpha}_{j}^{\tau_{j}^{\ell-1}}$ as $\varphi_{j}^{t} \to \varphi_{j}^{(\ell-1)^{+}}$ .

Part (ii). Let $t \to (\tau_j^\ell)^-$ in the equation of part (x), and take the dot product of each side with $\overline{\boldsymbol{x}}_{i_j^\ell}$ . Observe that $\left(\overline{\boldsymbol{\alpha}}_j^{\tau_j^\ell}\right)^\top \overline{\boldsymbol{x}}_{i_\ell} = 0$ , because $I_0\left(\boldsymbol{\alpha}_j^{\tau_j^\ell}\right) = \{i_j^\ell\}$ by part (v). We obtain

$$
0 = \frac {\sin \varphi_ {j} ^ {\ell^ {-}} \cdot \left(\overline {{\boldsymbol {\alpha}}} _ {j} ^ {\tau_ {j} ^ {\ell - 1}}\right) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} + \sin \left(\varphi_ {j} ^ {(\ell - 1) ^ {+}} - \varphi_ {j} ^ {\ell^ {-}}\right) \cdot \left(\overline {{\boldsymbol {\gamma}}} _ {I _ {j} ^ {\ell}}\right) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}}}{\sin \varphi_ {j} ^ {(\ell - 1) ^ {+}}}
$$

and the required equation follows. It remains to note that $\sin\varphi_{j}^{\ell^{-}}\neq0$ because otherwise either $\varphi_{j}^{(\ell-1)^{+}}-\varphi_{j}^{\ell^{-}}\in\{-\pi,0,\pi\}$ or $(\overline{\gamma}_{I_{j}^{\ell}})^{\top}\overline{x}_{i_{j}^{\ell}}=0$ . The former is impossible because we know already from part (vi) that $\cos\varphi_{j}^{t}\in(0,1)$ when $t\in(\tau_{j}^{\ell-1},\tau_{j}^{\ell})$ , and $\varphi_{j}^{t}\geq0$ by definition, so $\varphi_{j}^{(\ell-1)^{+}}=\varphi_{j}^{\ell^{-}}$ but this would still contradict part (vi). The latter is impossible because $I_{j}^{\ell}\neq\emptyset$ and, by Proposition 11, part (i), the dot product must be positive due to positive correlation between training points.

Part (xi). Consider any interval $(t_{1}, t_{2})$ such that $I_{0}(\boldsymbol{\alpha}_{j}^{t}) = \emptyset$ for all $t \in (t_{1}, t_{2})$ , and let $\boldsymbol{\gamma} := \boldsymbol{\gamma}_{I_{+}(\boldsymbol{\alpha}_{j}^{t})}$ ; the choice of t in the interval is immaterial by the continuity of the map $t \mapsto \alpha_{j}^{t}$ and of the dot product function with a fixed vector $x_{i}$ . We have $\boldsymbol{\gamma} = \boldsymbol{\gamma}_{I_{j}^{\ell}}$ when $t \in (\tau_{j}^{\ell-1}, \tau_{j}^{\ell})$ by part (iv). Recall our calculations for the derivative of $\|\alpha_{j}^{t}\|$ and of $\overline{\alpha}_{j}^{t}$ (before Proposition 13). We have $d\overline{\alpha}_{j}^{t}/dt = s_{j} p$ , where $p := \boldsymbol{\gamma} - \overline{\alpha}_{j}^{t} \cdot (\overline{\alpha}_{j}^{t})^{\top} \boldsymbol{\gamma}$ is the vector obtained by subtracting from $\boldsymbol{\gamma}$ its orthogonal projection onto the line with direction $\alpha_{j}^{t}$ . Then

$$
s _ {j} \frac {\mathrm{d} (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \overline {{\boldsymbol {x}}} _ {i}}{\mathrm{d} t} = \boldsymbol {\gamma} ^ {\top} \overline {{\boldsymbol {x}}} _ {i} - (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \overline {{\boldsymbol {x}}} _ {i} \cdot (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma}.
$$

For $t \in (t_1, t_2)$ , we have $(\overline{\alpha}_j^t)^\top \overline{x}_i \leq 0$ because $i \notin I_+(\alpha_j^t)$ . Recall that $(\overline{\alpha}_j^t)^\top \gamma \geq 0$ by Proposition 11, part (ii). We have shown that $-(\overline{\alpha}_j^t)^\top \overline{x}_i \cdot (\overline{\alpha}_j^t)^\top \gamma \geq 0$ , completing the proof of part (xi).

Part (xii). We continue the calculation from part (xi) assuming that $s_j = -1$ and $i = i_j^\ell$ . For the function $g(t) := (\overline{\alpha}_j^t)^\top \overline{x}_{i_j^\ell}$ , we have

$$
\begin{array}{l} \frac {\mathrm{d} g}{\mathrm{d} t} = - \boldsymbol {\gamma} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} + (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \cdot (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma}, \\ \frac {1}{\| \boldsymbol {\gamma} \|} \frac {\mathrm{d} ^ {2} g}{\mathrm{d} t ^ {2}} = \frac {\mathrm{d}}{\mathrm{d} t} \left((\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \overline {{\boldsymbol {\gamma}}}\right) \cdot (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} + (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \overline {{\boldsymbol {\gamma}}} \cdot \frac {\mathrm{d}}{\mathrm{d} t} \left((\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t}) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}}\right) \\ = \frac {\mathrm{d} \cos \varphi_ {j} ^ {t}}{\mathrm{d} t} \cdot g + \cos \varphi_ {j} ^ {t} \cdot \frac {\mathrm{d} g}{\mathrm{d} t}. \\ \end{array}
$$

We will show the following two properties:

- $\mathrm{d}g / \mathrm{d}t < 0$ as $t \to (\tau_j^{\ell - 1})^{+}$ ;   
- if $\mathrm{dg} / \mathrm{dt} = 0$ for some $t$ , then $\mathrm{d}^2 g / \mathrm{dt}^2 < 0$ for the same $t$ .

Together, these properties imply that $dg/dt < 0$ throughout the interval $(\tau_{j}^{\ell-1}, \tau_{j}^{\ell})$ . Indeed, assume otherwise for the sake of contradiction, then $dg/dt = 0$ at some point $t_{0} \in (\tau_{j}^{\ell-1}, \tau_{j}^{\ell})$ . By the second property, g must have a local maximum at $t_{0}$ , and in particular $dg/dt > 0$ for all $t < t_{0}$ close enough to $t_{0}$ . By the first property, the minimum of $dg/dt$ on $(\tau_{j}^{\ell-1}, t_{0})$ exists and is attained at an interior point of the interval. But this contradicts the second property.

Let us now justify the properties. For the first property, notice that

$$
\begin{array}{l} \left. \frac {\mathrm{d} g}{\mathrm{d} t} \right| _ {t \rightarrow (\tau_ {j} ^ {\ell - 1}) ^ {+}} = - \| \boldsymbol {\gamma} \| \cdot \left(\overline {{\boldsymbol {\gamma}}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} - \left(\overline {{\boldsymbol {\alpha}}} _ {j} ^ {\tau_ {j} ^ {\ell - 1}}\right) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \cdot \left(\overline {{\boldsymbol {\alpha}}} _ {j} ^ {\tau_ {j} ^ {\ell - 1}}\right) ^ {\top} \overline {{\boldsymbol {\gamma}}}\right) \\ = - \| \boldsymbol {\gamma} \| \cdot (\overline {{\boldsymbol {\gamma}}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}}) \cdot \left(1 - \frac {\left(\overline {{\boldsymbol {\alpha}}} _ {j} ^ {\tau_ {j} ^ {\ell - 1}}\right) ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}}}{\overline {{\boldsymbol {\gamma}}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}}} \cdot \left(\overline {{\boldsymbol {\alpha}}} _ {j} ^ {\tau_ {j} ^ {\ell - 1}}\right) ^ {\top} \overline {{\boldsymbol {\gamma}}}\right). \\ \end{array}
$$

Here, $\overline{\gamma}^{\top}\overline{x}_{i_j^\ell} > 0$ since $i_j^\ell \in I_j^\ell$ . The value of the ratio $\left(\overline{\alpha}_j^{\tau_j^{\ell - 1}}\right)^\top \overline{x}_{i_j^\ell} / \overline{\gamma}^\top \overline{x}_{i_j^\ell}$ appears in the statement of part (i), and in particular replacing the index $i_j^\ell$ with any other $i \in I_j^\ell$ would result in a higher (positive) value. Therefore, if we assume for the sake of contradiction that the right-hand side in the last equation is non-negative, then it will remain non-negative if $i_j^\ell$ is replaced with every other $i \in I_j^\ell$ . In other words, if $g(t) = (\overline{\alpha}_j^t)^\top \overline{x}_{i_j^\ell}$ is non-decreasing in a right-neighbourhood of $\tau_j^{\ell - 1}$ , so is every dot product $(\overline{\alpha}_j^t)^\top \overline{x}_i$ with $i \in I_j^\ell$ . But then their linear combination with positive coefficients $y_i \| x_i\| / n$ is also non-decreasing. This, however, is not possible because this linear combination is $(\overline{\alpha}_j^t)^\top \gamma_{I_+(\alpha_j^t)}$ , and we already saw in the proof of part (xi) that $\mathrm{d}\overline{\alpha}_j^t/\mathrm{d}t = s_j p$ , where $p$ is an orthogonal projection of $\gamma_{I_+(\alpha_j^t)}$ onto a proper subspace. By standard properties of projections we must have $p^\top \gamma_{I_+(\alpha_j^t)} > 0$ and, since $s_j = -1$ , $\mathrm{d}\big((\overline{\alpha}_j^t)^\top \gamma_{I_+(\alpha_j^t)}\big)/\mathrm{d}t < 0$ , which is a contradiction. (The case $p^\top \gamma_{I_+(\alpha_j^t)} = 0$ is impossible by Proposition 12, part (v), as we would then have $\cos\varphi_j^{(\ell -1)^+} = 1$ .) This concludes the proof of the first property.

The second property follows directly from the equation for $d^{2}g/dt^{2}$ , because $\cos\varphi_{j}^{t}$ decreases by part (vi) and because g > 0.

For the sign of second derivative in general, it remains to consider the second term. The first factor is positive by Proposition 11, part (ii); and we just proved above that $dg/dt < 0$ . (Note that $\gamma \neq 0$ by Proposition 11, part (i), because $i_{j}^{\ell} \in L_{+}(\alpha_{j}^{t})$ .) This completes the proof of part (xii). ☐

Corollary 15. For all $j \in J_{+} \cup J_{-}$ , the set of all $z_{j} \in \mathbb{R}^{d}$ such that $|I_0(\alpha_j^t)| > 1$ for some $t \in [0, \infty)$ has Lebesgue measure zero.

Proof. A single yardstick trajectory at any time t follows a direction $\gamma_{S}$ for some $S \subseteq [n]$ . The set S changes at most n times, namely at the crossing of $\bigcup_{i \in [n]} H_{i}$ , where $H_{i}$ is the set of vectors orthogonal to the training point $x_{i}$ . (The proof of this fact does not rely on Assumption 1, part (iv). It is a consequence of Proposition 12, part (iv). We note that the assumption of Proposition 12 is shown to be valid in the proof of Proposition 14, under “Intermediate summary” on page 23.)

The union U of all $H_{i} \cap H_{k}, i < k$ , is a union of finitely many subspaces of dimension d - 2 (because no two training points are collinear by Assumption 1, part (iii)). Consider all the vectors u such that the yardstick trajectory starting at u passes through U. We claim that this is a set of zero measure. Indeed:

\- Every convex polyhedron $P$ of dimension $d - 2$ , for example $H_{i} \cap H_{k}$ , can be reached by a straight-line trajectory (without change of direction) from a convex polyhedron $P'$ of dimension at most $d - 1$ , i.e., of co-dimension at least 1.

- The previous change of direction occurs at the intersection of the polyhedron $P'$ and the union of all $H_i$ . This intersection is a finite union of convex polyhedra of co-dimension at least 2, because, for all nonempty subsets $S \subseteq [n]$ , the vector $\gamma_S$ cannot belong to any $H_i$ , thanks to the 45-degree condition. To each of these polyhedra, the previous bullet point applies.   
- No more than $n$ changes of direction may take place along a single trajectory.

Thus, all vectors from which a point in U can be reached along a yardstick trajectory belong to a finite union of affine subspaces of co-dimension 1. Thus, they form a measure zero set. □

# D Proofs for the first phase

Here we prove Lemma 3, Lemma 4, and Lemma 5, as well as a number of related results. The former are subsumed by Lemma 19, Lemma 21, and Lemma 23 below.

Recall the definitions of $\delta$ and $\Delta$ in section 3:

$$
\begin{array}{l} \delta := \min \left\{ \begin{array}{c} \min _ {i \in [ n ]} \| \boldsymbol {x} _ {i} \|,   \min _ {i, i ^ {\prime} \in [ n ]} \overline {{\boldsymbol {x}}} _ {i} ^ {\top} \overline {{\boldsymbol {x}}} _ {i ^ {\prime}},   \min _ {k \in [ d - 1 ]} (\sqrt {\eta_ {k}} - \sqrt {\eta_ {k + 1}}) (d - 1),   \sqrt {\eta_ {d}}, \\ \min _ {k \in [ d ]} \nu_ {k} ^ {*} \sqrt {d},   \min _ {j \in [ m ]} \| \boldsymbol {z} _ {j} \|,   \min _ {j \in J _ {+}} \cos \varphi_ {j} ^ {0},   \min _ {j \in J _ {-}} \sin \varphi_ {j} ^ {0}, \\ \min \left\{| \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i} |   \left| \begin{array}{c} j \in J _ {+} \cup J _ {-} \wedge \ell \in [ n _ {j} ] \\ \wedge t \in [ \tau_ {j} ^ {\ell - 1}, \tau_ {j} ^ {\ell} ] \wedge i \in [ n ] \\ \wedge i \neq i _ {j} ^ {\ell} \wedge (\ell \neq 1 \Rightarrow i \neq i _ {j} ^ {\ell - 1}) \end{array} \right. \right\},   \min _ {j \in J _ {-}} \overline {{\boldsymbol {\alpha}}} _ {j} ^ {0} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {1}}, \\ \min \{\tau_ {j} ^ {\ell} - \tau_ {j} ^ {\ell - 1} | j \in J _ {+} \cup J _ {-} \wedge \ell \in [ n _ {j} ] \} \\ \end{array} \right\} \\ \Delta := \max \left\{\max _ {i \in [ n ]} \| \boldsymbol {x} _ {i} \|, \max _ {j \in [ m ]} \| \boldsymbol {z} _ {j} \|, 1 \right\}. \\ \end{array}
$$

Thus $\delta$ is the minimum of: the length of any training point, the cosine of the angle between any two training points, the difference between the square roots of any consecutive eigenvalues adjusted by the dimension, the square root of the smallest eigenvalue, the smallest eigenvector coordinate of the teacher neuron adjusted by the square root of the dimension, the length of any unscaled hidden-neuron initialisation, the cosine or sine of the angle between it (if active) and the corresponding vector $\gamma_{I}$ depending on whether the last-layer sign is positive or negative (respectively), the absolute cosine of any angle between a trajectory point $\alpha_{j}^{t}$ and a training point which is neither the previous nor the next to cross the half-space boundary, the cosine of the angle between any initial negative-sign active hidden neuron and the first data point to cross the boundary, and the time between any two consecutive crossings; and $\Delta \geq 1$ is the maximum length of any training point or unscaled hidden-neuron initialisation.

First we observe that, immediately from the definitions in section 3 of the vectors $\gamma_{I}$ , the matrix $X$ , the eigenvalues $\eta_{k}$ , the eigenvectors $u_{k}$ , and the coordinates $\nu_{k}^{*}$ of the teacher neuron with respect to the basis consisting of the eigenvectors, we have the following two alternative expressions for the vector $\gamma_{[n]}$ .

Proposition 16. $\gamma_{[n]} = \frac{1}{n} X X^{\top} v^{*} = \sum_{k=1}^{d} \eta_{k} \nu_{k}^{*} u_{k}$ .

Then we establish upper bounds on: the largest eigenvalue of the matrix $\frac{1}{n}XX^{\top}$ , the ratio of any two consecutive eigenvalues in their decreasing ordering, the Euclidean lengths of the vectors $\gamma_{I}$ , the cosines of the angles $\varphi_{j}^{t}$ that measure alignment of the yardstick trajectories $\alpha_{j}^{t}$ (both defined in section 3) mapped backwards through the hyperbolic tangent sigmoid, and the finish time of the last intermediate alignment stage of a yardstick trajectory; and lower bounds on: the Euclidean lengths of the vectors $\gamma_{I}$ , and the cosines of the angles between a yardstick trajectory and the training point that is the next to enter or exit its active half-space.

Proposition 17. (i) $\eta_{1} \leq \Delta^{2}$ .

(ii) $\frac{\eta_{k + 1}}{\eta_k}\leq \left(1 - \frac{\delta}{(d - 1)\Delta}\right)^2$ for all $k\in [d - 1]$ .

(iii) $\frac{\delta^{5/2}|I|}{\sqrt{2}n} \leq \| \boldsymbol{\gamma}_I \| \leq \frac{\Delta^2|I|}{n}$ for all $I \subseteq [n]$ , and $\delta^2 \leq \| \boldsymbol{\gamma}_{[n]} \|$ .

(iv) $\max_{j\in J_+}^{\ell \in [n_j]}\operatorname{artanh}\cos \varphi_j^{\ell^-} < \ln \left(\frac{2}{\delta}\right)$ .

(v) $\max_{j\in J_{-}}^{\ell\in[n_{j}]}$ artanh $\cos\varphi_{j}^{(\ell-1)^{+}}<\ln\left(\frac{2}{\delta}\right)$ .

(vi) $\max_{j\in J_+ \cup J_-}\tau_j^{n_j} < \frac{4n\ln n}{\delta^3}.$

(vii) $\left|\overline{\alpha}_{j}^{t^{\top}}\overline{\boldsymbol{x}}_{i_{j}^{\ell}}\right| \geq \frac{2\delta^{4}}{3n}(\tau_{j}^{\ell}-t)$ for all $j \in J_{+} \cup J_{-}, \ell \in [n_{j}]$ , and $t \in [\tau_{j}^{\ell-1}, \tau_{j}^{\ell}]$ .

Proof. For part (i), we have

$$
\eta_ {1} = \left\| \frac {1}{n} \boldsymbol {X} \boldsymbol {X} ^ {\top} \boldsymbol {u} _ {1} \right\| = \frac {1}{n} \left\| \sum_ {i \in [ n ]} \boldsymbol {x} _ {i} \boldsymbol {x} _ {i} ^ {\top} \boldsymbol {u} _ {1} \right\| \leq \frac {1}{n} \sum_ {i \in [ n ]} \| \boldsymbol {x} _ {i} \boldsymbol {x} _ {i} ^ {\top} \boldsymbol {u} _ {1} \| \leq \max _ {i \in [ n ]} \| \boldsymbol {x} _ {i} \| ^ {2} = \Delta^ {2}.
$$

For part (ii), supposing $k \in [d - 1]$ , by part (i) we have

$$
\frac {\sqrt {\eta_ {k + 1}}}{\sqrt {\eta_ {k}}} = 1 - \frac {\sqrt {\eta_ {k}} - \sqrt {\eta_ {k + 1}}}{\sqrt {\eta_ {k}}} \leq 1 - \frac {\sqrt {\eta_ {k}} - \sqrt {\eta_ {k + 1}}}{\sqrt {\eta_ {1}}} \leq 1 - \frac {\delta}{(d - 1) \Delta}.
$$

For part (iii), supposing $I \subseteq [n]$ , recalling that $\angle (\pmb{v}^*, \pmb{x}_i) < \pi / 4$ for all $i \in [n]$ we have

$$
\begin{array}{l} \| \boldsymbol {\gamma} _ {I} \| = \frac {1}{n} \left\| \sum_ {i \in I} y _ {i} \boldsymbol {x} _ {i} \right\| = \frac {1}{n} \sqrt {\sum_ {i , i ^ {\prime} \in I} y _ {i} y _ {i ^ {\prime}} \boldsymbol {x} _ {i} ^ {\top} \boldsymbol {x} _ {i ^ {\prime}}} \geq \frac {| I |}{n} \min _ {i, i ^ {\prime} \in I} \sqrt {y _ {i} y _ {i ^ {\prime}} \boldsymbol {x} _ {i} ^ {\top} \boldsymbol {x} _ {i ^ {\prime}}} \\ = \frac {| I |}{n} \min _ {i, i ^ {\prime} \in I} \sqrt {\pmb {v} ^ {* ^ {\top}} \pmb {x} _ {i} \cdot \pmb {v} ^ {* ^ {\top}} \pmb {x} _ {i ^ {\prime}} \cdot \pmb {x} _ {i} ^ {\top} \pmb {x} _ {i ^ {\prime}}} \geq \frac {| I |}{n} \sqrt {\left(\frac {\delta}{\sqrt {2}}\right) ^ {2} \delta^ {3}} = \frac {\delta^ {5 / 2} | I |}{\sqrt {2} n}, \\ \end{array}
$$

and we have

$$
| \boldsymbol {\gamma} _ {I} \| \leq \frac {1}{n} \sum_ {i \in I} y _ {i} \| \boldsymbol {x} _ {i} \| \leq \frac {| I |}{n} \max _ {i \in I} \boldsymbol {v} ^ {* ^ {\top}} \boldsymbol {x} _ {i} \cdot \| \boldsymbol {x} _ {i} \| \leq \frac {| I |}{n} \max _ {i \in I} \| \boldsymbol {x} _ {i} \| ^ {2} \leq \frac {\Delta^ {2} | I |}{n}.
$$

Also, recalling Proposition 16 we have $\| \boldsymbol{\gamma}_{[n]}\| = \| \frac{1}{n}\boldsymbol {X}\boldsymbol {X}^{\top}\boldsymbol{v}^{*}\| \geq \eta_{d}\geq \delta^{2}$ .

For part (iv), supposing $j \in J_{+}$ and $\ell \in [n_{j}]$ , and observing that $\operatorname{artanh} q = \frac{1}{2} \ln\left(\frac{1+q}{1-q}\right) < \frac{1}{2} \ln\left(\frac{2}{1-q}\right)$ for all $|q| < 1$ , by Proposition 14 (iv) and (v) we have

$$
\begin{array}{l} \operatorname{artanh} \cos \varphi_ {j} ^ {\ell^ {-}} = \operatorname{artanh} \lim _ {t \rightarrow (\tau_ {j} ^ {\ell}) ^ {-}} \cos \angle \left(\boldsymbol {\alpha} _ {j} ^ {t}, \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}}\right) \\ <   \operatorname{artanh} \sin \angle \left(\boldsymbol {x} _ {i _ {j} ^ {\ell}}, \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}}\right) = \operatorname{artanh} \sqrt {1 - \cos^ {2} \angle \left(\boldsymbol {x} _ {i _ {j} ^ {\ell}} , \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}}\right)} \leq \operatorname{artanh} \sqrt {1 - \delta^ {2}} \\ <   \operatorname{artanh} \left(1 - \frac {\delta^ {2}}{2}\right) <   \frac {1}{2} \ln \left(\frac {4}{\delta^ {2}}\right) = \ln \left(\frac {2}{\delta}\right). \\ \end{array}
$$

Part (v) follows analogously, once we recall that, for all $j \in J_{-}$ , by Assumption 1 (ii) we have $\cos\varphi_{j}^{0^{+}} = \cos\varphi_{j}^{0} = \sqrt{1 - \sin^{2}\varphi_{j}^{0}} \leq \sqrt{1 - \delta^{2}}$ .

For part (vi), supposing $j \in J_{+} \cup J_{-}$ we have

$$
\begin{array}{l} \tau_ {j} ^ {n _ {j}} = \sum_ {\ell \in [ n _ {j} ]} \tau_ {j} ^ {\ell} - \tau_ {j} ^ {\ell - 1} \quad \text { since } \tau_ {j} ^ {0} := 0 \text { in   Proposition   14 } \\ \leq \sum_ {\ell \in [ n _ {j} ]} \frac {\ln \left(\frac {2}{\delta}\right)}{\left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\|} \quad \text { by   parts   (iv)   and   (v),   and   Proposition   14   (vi) } \\ \leq \frac {\sqrt {2} n}{\delta^ {5 / 2}} \ln \left(\frac {2}{\delta}\right) \sum_ {\ell \in [ n _ {j} ]} \frac {1}{| I _ {j} ^ {\ell} |} \quad \text { by   part   (iii) } \\ \leq \frac {\sqrt {2} n}{\delta^ {5 / 2}} \ln \left(\frac {2}{\delta}\right) \sum_ {i \in [ n ]} \frac {1}{i} \quad \text { by   the   definition   of } I _ {j} ^ {\ell} \text { in   Proposition   14 } \\ <   \frac {\sqrt {2} n (1 + \ln n)}{\delta^ {5 / 2}} \ln \left(\frac {2}{\delta}\right) \quad \text { by   properties   of   the   harmonic   series } \\ <   \frac {7 n \ln n}{2 \delta^ {5 / 2}} \ln \left(\frac {2}{\delta}\right) \quad \text { since } n \geq 2 \text { by   Assumption1(i)} \\ <   \frac {4 n \ln n}{\delta^ {3}} \quad \text { since } \ln \left(\frac {2}{\delta}\right) <   \frac {8}{7 \sqrt {\delta}}. \\ \end{array}
$$

For part (vii), if $s_j = 1$ then by Proposition 14 (xi) and by part (iii) we have

$$
\inf _ {t \in (\tau_ {j} ^ {\ell - 1}, \tau_ {j} ^ {\ell})} \frac {\mathrm{d} \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} {} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}}}{\mathrm{d} t} \geq \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \geq \frac {\delta^ {7 / 2}}{\sqrt {2} n}  .
$$

If $s_{j} = -1$ then by Proposition 14 (xii) we have that $\overline{\alpha}_{j}^{t^{\top}}\overline{x}_{i_{j}^{\ell}}$ is concave on $[\tau_{j}^{\ell-1}, \tau_{j}^{\ell}]$ , so by part (v), by Proposition 14 (vi), by part (iii), and since $\ln\left(\frac{2}{\delta}\right) < \frac{3}{2\sqrt{2\delta}}$ , for all $t \in [\tau_{j}^{\ell-1}, \tau_{j}^{\ell}]$ we have

$$
\frac {\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}}}{\tau_ {j} ^ {\ell} - t} \geq \frac {\overline {{\boldsymbol {\alpha}}} _ {j} ^ {\tau_ {j} ^ {\ell - 1} ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}}}{\tau_ {j} ^ {\ell} - \tau_ {j} ^ {\ell - 1}} \geq \frac {\delta}{\frac {\sqrt {2} n}{\delta^ {5 / 2}} \ln \left(\frac {2}{\delta}\right)} = \frac {\delta^ {7 / 2}}{\sqrt {2} n} \frac {1}{\ln \left(\frac {2}{\delta}\right)} > \frac {2 \delta^ {4}}{3 n}.
$$

Recall from section 4 that $T_0 = \max_{j \in J_+ \cup J_-} \tau_j^{n_j} + 1$ and $T_1 = \varepsilon \ln(1/\lambda) / \| \boldsymbol{\gamma}_{[n]} \|$ .

Proposition 18. $T_0 < T_1 / 3$ .

Proof. We have

$$
\begin{array}{l} T _ {1} / 3 \geq 9 n \ln n \frac {\Delta^ {2}}{\delta^ {3}} \bigg / \| \boldsymbol {\gamma} _ {[ n ]} \| \quad \text { by   Assumption   2 } \\ \geq \frac {9 n \ln n}{\delta^ {3}} \quad \text { by   Proposition   17   (iii) } \\ > \frac {4 n \ln n}{\delta^ {3}} + 1 \quad \text { since } n \geq 2 \text { by   Assumption } 1 (\mathrm{i}) \\ > T _ {0} \quad \text { by   Proposition   17(vi). } \\ \end{array}
$$

The following lemma states that, throughout the first phase of the training, the lengths of the positive-sign initially active hidden neurons are non-decreasing, and the lengths of the negative-sign initially active hidden neurons are non-increasing; and it provides a time-sensitive upper bound for the former.

As for the remaining hidden neurons, i.e. those whose indices are not in the sets $J_{+}$ and $J_{-}$ , they are by definition inactive at initialisation, and by Assumption 1 (ii) have no training points in their activation boundaries, so they do not change throughout the training.

Lemma 19. (i) $\| \pmb{w}_j^t\| < 2\| \pmb{z}_j\| \lambda^{1 - \varepsilon t / T_1}$ for all $j\in J_{+}$ and all $t\in [0,T_1]$ .

(ii) $s_j \mathrm{d}\| \boldsymbol{w}_j^t\| /\mathrm{d}t \geq 0$ for all $j \in J_{+} \cup J_{-}$ and almost all $t \in [0,T_1]$ .

Proof. First we establish the following, which implies (i).

Claim 20. $\| \pmb{w}_j^t\| < 2\| \pmb {z}_j\| \lambda^{1 - \varepsilon t / T_1}$ for all $j\in [m]$ and all $t\in [0,T_1]$ .

Proof of claim. Assume for a contradiction that this fails, and let $t \in [0, T_1]$ be the smallest such that $\| \pmb{w}_j^t \| \geq 2 \| \pmb{z}_j \| \lambda^{1 - \varepsilon t / T_1}$ for some $j \in [m]$ . Then we have

$$
\begin{array}{l} \| \pmb {w} _ {j} ^ {t} \| \leq \lambda \| \pmb {z} _ {j} \| \mathrm{e} ^ {t \max _ {t ^ {\prime} \in [ 0, t ]}} \big \| \pmb {g} _ {j} ^ {t ^ {\prime}} \big \| \\ \leq \lambda \| \pmb {z} _ {j} \| \mathrm{e} ^ {t \left(\| \pmb {\gamma} _ {[ n ]} \| + \max _ {t ^ {\prime} \in [ 0, t ]} ^ {i \in [ n ]} \big | h _ {\pmb {\theta} ^ {t ^ {\prime}}} (\pmb {x} _ {i}) \big | \| \pmb {x} _ {i} \|\right)} \\ \leq \lambda \| \pmb {z} _ {j} \| \mathrm{e} ^ {t \left(\| \pmb {\gamma} _ {[ n ]} \| + m \max _ {t ^ {\prime} \in [ 0, t ]} ^ {j ^ {\prime} \in [ m ], i \in [ n ]} \left\| \pmb {w} _ {j ^ {\prime}} ^ {t ^ {\prime}} \right\| ^ {2} \| \pmb {x} _ {i} \| ^ {2}\right)} \\ \leq \lambda \| \pmb {z} _ {j} \| \mathrm{e} ^ {t (\| \pmb {\gamma} _ {[ n ]} \| + \lambda^ {2 - 2 \varepsilon} 4 m \Delta^ {4})} \\ \leq \lambda \| \boldsymbol {z} _ {j} \| \mathrm{e} ^ {t \| \boldsymbol {\gamma} _ {[ n ]} \|} \mathrm{e} ^ {T _ {1} \lambda^ {2 - 2 \varepsilon} 4 m \Delta^ {4}} \\ = \| \pmb {z} _ {j} \| \lambda^ {1 - \varepsilon t / T _ {1}} \lambda^ {- \varepsilon \lambda^ {2 - 2 \varepsilon} 4 m \Delta^ {4} / \| \pmb {\gamma} _ {[ n ]} \|} \\ \leq \| \boldsymbol {z} _ {j} \| \lambda^ {1 - \varepsilon t / T _ {1}} \lambda^ {- \varepsilon \lambda^ {2 - 2 \varepsilon} 4 m \Delta^ {4} / \delta^ {2}} \\ = \| \pmb {z} _ {j} \| \lambda^ {1 - \varepsilon t / T _ {1}} \mathrm{e} ^ {\ln (\lambda^ {- \varepsilon}) \lambda^ {2 - 2 \varepsilon} 4 m \Delta^ {4} / \delta^ {2}} \\ <   \| \boldsymbol {z} _ {j} \| \lambda^ {1 - \varepsilon t / T _ {1}} \mathrm{e} ^ {\lambda^ {2 - 3 \varepsilon} 4 m \Delta^ {4} / \delta^ {2}} \\ <   \| \boldsymbol {z} _ {j} \| \lambda^ {1 - \varepsilon t / T _ {1}} \mathrm{e} ^ {\lambda^ {2 - 4 \varepsilon}} \\ <   2 \| \boldsymbol {z} _ {j} \| \lambda^ {1 - \varepsilon t / T _ {1}} \\ \end{array}
$$

by Proposition 1 and Grönwall's inequality

by the definition of $g_{j}^{t'}$ in Proposition 1 (i)

by the definition of $h_{\theta^{t'}}$ in section 2

since $\left\| \boldsymbol{w}_{j'}^{t'}\right\| \leq 2\left\| \boldsymbol{z}_{j'}\right\| \lambda^{1 - \varepsilon t' / T_1}$

since $t\leq T_1$

since $\mathrm{e}^{T_1\| \boldsymbol{\gamma}_{[n]}\|} = \lambda^{-\varepsilon}$

by Proposition 17 (iii)

since exp and ln are inverses

since $\lambda^{-\varepsilon} > \ln (\lambda^{-\varepsilon})$

since $\lambda^{-\varepsilon}\geq m^3 n^{9\cdot 3n\Delta^2 /\delta^3} > 4m\Delta^4 /\delta^2$

since $\lambda^{2 - 4\varepsilon}\leq \lambda \leq 2^{-9\cdot 2\cdot 3\cdot 4} <   \ln 2.$

To prove (ii), observing that by Proposition 1 for all $j \in [m]$ and almost all $t \in [0, \infty)$ we have

$$
s _ {j} \mathrm{d} \| \boldsymbol {w} _ {j} ^ {t} \| / \mathrm{d} t = \boldsymbol {w} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t} \in \frac {1}{n} \sum_ {i = 1} ^ {n} (y _ {i} - h _ {\boldsymbol {\theta} ^ {t}} (\boldsymbol {x} _ {i}))   \partial \sigma (\boldsymbol {w} _ {j} ^ {t} ^ {\top} \boldsymbol {x} _ {i})   \boldsymbol {w} _ {j} ^ {t} ^ {\top} \boldsymbol {x} _ {i} \subseteq \sum_ {i = 1} ^ {n} (y _ {i} - h _ {\boldsymbol {\theta} ^ {t}} (\boldsymbol {x} _ {i}))   [ 0, \infty) ,
$$

it suffices to show that for all $t \in [0, T_1]$ and all $i \in [n]$ we have $|h_{\boldsymbol{\theta}^t}(\boldsymbol{x}_i)| \leq y_i$ . Indeed

$|h_{\boldsymbol{\theta}^t}(\boldsymbol{x}_i)| \leq m \max_{j=1}^{m} \| \boldsymbol{w}_j^t\|^2 \| \boldsymbol{x}_i\|$ by the definition of $h_{\boldsymbol{\theta}^t'}$ in section 2

$$
<   4 m \Delta^ {3} \lambda^ {2 - 2 \varepsilon} \quad \text { by   Claim   20 }
$$

$$
<   \frac {\delta}{\sqrt {2}} \quad \text { since } \lambda^ {2 \varepsilon - 2} \geq m ^ {3 \cdot 6} n ^ {9 \cdot 3 \cdot 6 n \Delta^ {2} / \delta^ {3}} > 4 \sqrt {2} m \Delta^ {3} / \delta
$$

$$
<   y _ {i} \quad \text { since } \angle (\boldsymbol {v} ^ {*}, \boldsymbol {x} _ {i}) <   \pi / 4.
$$

The next lemma provides a detailed description of the intermediate alignment stages for each hidden neuron. It states that the training points enter or exit the active half-space of each positive-sign or negative-sign (respectively) hidden neuron in the same order as they do for the positive half-space of the corresponding yardstick trajectory, and it provides non-asymptotic bounds for each difference: between a dynamics-governing vector $g_{j}^{t}$ (defined in Proposition 1 (i)) and the corresponding intermediate

yardstick target $\gamma_{I_{j}^{\ell}}$ (the sets $I_{j}^{\ell}$ of indices of training points that are in the active half-space of hidden neuron j during stage $\ell$ are defined in Proposition 14), between the unit-sphere normalisations of a hidden neuron and the corresponding yardstick vector, and between the corresponding boundary crossing times for a hidden neuron and its yardstick vector. In particular, it shows that each negative-sign initially active hidden neuron j deactivates at time $t_{j}^{n_{j}}$ , and hence before time $T_{0}$ . The proof of the lemma is inductive over the stage index $\ell$ , and involves carefully controlling the differences between the trajectories on the unit sphere of the hidden neurons and their yardstick vectors; this is non-trivial because, in contrast to the latter which have separate individual dynamics, the dynamics of the former are joint since each governing vector $g_{j}^{t}$ depends on the outputs of the whole network.

Lemma 21. For all $j \in J_{+} \cup J_{-}$ there exist unique $t_j^1, \ldots, t_j^{n_j} \in [0, \infty)$ such that for all $\ell \in [n_j]$ the following hold, where $t_j^0 := 0$ :

(i) $I_{+}(\boldsymbol{w}_{j}^{t}) = I_{j}^{\ell}$ for all $t \in (t_{j}^{\ell-1}, t_{j}^{\ell})$ ;   
(ii) $I_0(\pmb{w}_j^t) = \emptyset$ for all $t\in (t_j^{\ell -1},t_j^\ell)$ , and $I_0\left(\pmb{w}_j^{t_j^\ell}\right) = \{i_j^\ell \}$ ;   
(iii) $\| \pmb{\gamma}_{I_j^\ell} - \pmb{g}_j^t\| \leq \lambda^{2 - \varepsilon}$ for all $t\in (t_j^{\ell -1},t_j^\ell)$ ;   
(iv) $\| \overline{\boldsymbol{\alpha}}_j^t -\overline{\boldsymbol{w}}_j^t\| \leq \lambda^{1 - \left(1 + \frac{3\ell}{3n_j}\right)\varepsilon}$ for all $t\in (t_j^{\ell -1},\max \{t_j^\ell ,\tau_j^\ell \} ];$   
(v) $|\tau_j^\ell - t_j^\ell| \leq \lambda^{1 - \left(1 + \frac{3\ell - 1}{3n_j}\right)\varepsilon}$ .

Proof. For all $j \in J_{+} \cup J_{-}$ and all $\ell \in [n_j]$ let

$$
\mathfrak {t} _ {j} ^ {0} := 0 \qquad \qquad \mathfrak {t} _ {j} ^ {- \ell} := \tau_ {j} ^ {\ell} - \lambda^ {1 - \left(1 + \frac {3 \ell - 1}{3 n _ {j}}\right) \varepsilon} \qquad \qquad \mathfrak {t} _ {j} ^ {\ell} := \tau_ {j} ^ {\ell} + \lambda^ {1 - \left(1 + \frac {3 \ell - 1}{3 n _ {j}}\right) \varepsilon}.
$$

For all $j \in J_{+} \cup J_{-}$ and all $\ell \in [n_j]$ , define a continuous $w_j^t$ by

$$
\mathsf {w} _ {j} ^ {\mathfrak {t} _ {j} ^ {\ell - 1}} := \boldsymbol {\alpha} _ {j} ^ {\mathfrak {t} _ {j} ^ {\ell - 1}} \quad \mathrm{dw} _ {j} ^ {t} / \mathrm{d} t := s _ {j} \| \mathsf {w} _ {j} ^ {t} \|   \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \quad \text { for   all } t \in (\mathfrak {t} _ {j} ^ {\ell - 1}, \mathfrak {t} _ {j} ^ {\ell} ]  .
$$

Thus $w_{j}^{t}$ equals $\alpha_{j}^{t}$ on $[t_{j}^{\ell-1}, \tau_{j}^{\ell}]$ , and $w_{j}^{t}$ continues with the same dynamics on $(\tau_{j}^{\ell}, t_{j}^{\ell}]$ .

We first observe that, until the largest of the times $t_{j}^{\ell}$ , the network outputs at all the training points remain small. Specifically, for all $t \in [0, \max_{j \in J_{+} \cup J_{-}} t_{j}^{n_{j}}]$ we have

$$
\begin{array}{l} \frac {1}{n} \sum_ {i = 1} ^ {n} | h _ {\boldsymbol {\theta} ^ {t}} (\boldsymbol {x} _ {i}) | \| \boldsymbol {x} _ {i} \| \leq m \left(\max _ {j = 1} ^ {m} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2}\right) \left(\max _ {i = 1} ^ {n} \| \boldsymbol {x} _ {i} \| ^ {2}\right) \\ <   4 m \Delta^ {4} \lambda^ {2 - 2 \varepsilon / 3} \\ <   \lambda^ {2 - \varepsilon} \\ \end{array}
$$

by the definition of $h_{\pmb{\theta}^t}$ in section 2

by Proposition 18 and Lemma 19

since $\lambda^{-\varepsilon /3}\geq mn^{9n\Delta^2 /\delta^3} > 4m\Delta^4$

That shows that part (iii) of the lemma is implied by parts (i), (ii), and (v).

Let $j \in J_{+} \cup J_{-}$ be fixed for the remainder of the proof.

We proceed to show the lemma by induction on $\ell\in[n_{j}]$ , where we make use of the following inductive hypothesis:

$$
\text { at } t = \mathfrak {t} _ {j} ^ {\ell - 1} \text { we   have } I _ {+} (\boldsymbol {w} _ {j} ^ {t}) = I _ {j} ^ {\ell}, I _ {0} (\boldsymbol {w} _ {j} ^ {t}) = \emptyset , \text { and } \| \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} \| \leq \lambda^ {1 - \left(1 + \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon}.
$$

That holds for $\ell = 1$ by Assumption 1 (ii) and since $\overline{\alpha}_j^0 = \overline{z_j} = \overline{\lambda z_j} = \overline{w}_j^0$ .

Consider $\ell \in [n_j]$ . By the inductive hypothesis, at $t = \mathfrak{t}_j^{\ell - 1}$ we have $1 - \overline{\mathsf{w}}_j^t^\top \overline{\mathsf{w}}_j^t \leq \frac{1}{2} \lambda^{2 - 2\left(1 + \frac{3\ell - 3}{3n_j}\right)\varepsilon}$ .

Let $T := \min\{t > t_{j}^{\ell-1} \mid I_{0}(w_{j}^{t}) \neq \emptyset\}$ .

If $s_j = 1$ , then for all $t \in (\mathfrak{t}_j^{\ell - 1}, \min \{\mathfrak{t}_j^\ell, \mathsf{T}\})$ we have

$$
\begin{array}{l} \mathrm{d} (1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\mathbf {w}}} _ {j} ^ {t}) / \mathrm{d} t \\ = - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} + \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t} (\overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} + \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t}) \\ <   2 \lambda^ {2 - \varepsilon} - (1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) (\overline {{\mathbf {w}}} _ {j} ^ {t} + \overline {{\boldsymbol {w}}} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \\ \leq 2 \lambda^ {2 - \varepsilon} - (1 - \overline {{\mathsf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t})   \overline {{\mathsf {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \\ \leq 2 \lambda^ {2 - \varepsilon} - (1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {1}} \right\| \cos \varphi_ {j} ^ {0} \\ \leq 2 \lambda^ {2 - \varepsilon} - (1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) \frac {\delta^ {7 / 2}}{\sqrt {2} n} \\ \leq 4 \lambda^ {\left(1 + 2 \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon} \left(\frac {1}{2} \lambda^ {2 - 2 \left(1 + \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon} - (1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t})\right) \\ \end{array}
$$

by Corollary 10 and since $I_0(\boldsymbol{w}_j^t) = \emptyset$

since $\| \pmb{\gamma}_{I_j^\ell} - \pmb{g}_j^t\| <  \lambda^{2 - \varepsilon}$

since $I_{+}(\pmb{w}_{j}^{t}) = I_{j}^{\ell}$

by Proposition 14 (vi) and (vii)

by Proposition 17 (iii)

since $\lambda^{\varepsilon}\leq n^{-27n / \delta^{3}} <   \left(\frac{\delta^{3}}{n}\right)^{27} <   \frac{\delta^{7 / 2}}{4\sqrt{2}n},$

so for all $t \in (\mathfrak{t}_{j}^{\ell-1}, \min\{\mathfrak{t}_{j}^{\ell}, \mathsf{T}\}]$ we have $1 - \overline{w}_{j}^{t}^{\top} \overline{w}_{j}^{t} \leq \frac{1}{2} \lambda^{2-2\left(1 + \frac{3\ell-3}{3n_{j}}\right)\varepsilon}$ and thus $\|\overline{w}_{j}^{t} - \overline{w}_{j}^{t}\| \leq \lambda^{1-\left(1 + \frac{3\ell-3}{3n_{j}}\right)\varepsilon}$ . If $s_{j} = -1$ then for all $t \in (\mathfrak{t}_{j}^{\ell-1}, \min\{\mathfrak{t}_{j}^{\ell}, \mathsf{T}\})$ we have

$$
\begin{array}{l} \mathrm{d} \left(\frac {1}{2} \lambda^ {2 - 2 \left(1 + \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon} + (1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t})\right) / \mathrm{d} t \\ = \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t} + \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t} (\overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} + \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t}) \\ <   2 \lambda^ {2 - \varepsilon} + (1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) (\overline {{\mathbf {w}}} _ {j} ^ {t} + \overline {{\boldsymbol {w}}} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \\ \leq 2 \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| \left(\frac {1}{2} \lambda^ {2 - 2 \left(1 + \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon} + (1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t})\right) \\ \end{array}
$$

by Corollary 10 and since $I_0(\pmb{w}_j^t) = \emptyset$

since $\| \pmb{\gamma}_{I_j^\ell} - \pmb{g}_j^t\| <  \lambda^{2 - \varepsilon}$

since $\lambda^{\varepsilon}\leq n^{-27n / \delta^{3}} <   \frac{\delta^{5 / 2}}{2\sqrt{2}n},$

so for all $t \in (\mathfrak{t}_j^{\ell - 1}, \min \{\mathfrak{t}_j^\ell, \mathsf{T}\}]$ we have

$$
\begin{array}{l} 1 - \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\mathbf {w}}} _ {j} ^ {t} \\ \leq \lambda^ {2 - 2 \left(1 + \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon} \left(\exp \left(2 \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| (t - t _ {j} ^ {\ell - 1})\right) - \frac {1}{2}\right) \\ <   \lambda^ {2 - 2 \left(1 + \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon} \left(\exp \left(2 \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| \left(\tau_ {j} ^ {\ell} - \tau_ {j} ^ {\ell - 1} + \lambda^ {1 - 2 \varepsilon}\right)\right) - \frac {1}{2}\right) \\ <   \lambda^ {2 - 2 \left(1 + \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon} \Bigl (\exp \Bigl (3 \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| \left(\tau_ {j} ^ {\ell} - \tau_ {j} ^ {\ell - 1}\right) \Bigr) - \frac {1}{2} \Bigr) \\ <   \lambda^ {2 - 2 \left(1 + \frac {3 \ell - 3}{3 n _ {j}}\right) \varepsilon} \left(\frac {8}{\delta^ {3}} - \frac {1}{2}\right) \\ <   \frac {1}{2} \lambda^ {2 - 2 \left(1 + \frac {3 \ell - 2}{3 n _ {j}}\right) \varepsilon} \\ \end{array}
$$

by Grönwall's inequality

by the definitions of $\mathfrak{t}_j^0$ and $\mathfrak{t}_j^\ell$

since $\lambda^{1 - 2\varepsilon}\leq n^{-9\cdot 6n / \delta^3} <   \frac{\delta}{2}$

by Proposition 14 (vi)

and Proposition 17 (v)

since $\lambda^{-\frac{2\varepsilon}{3n}}\geq n^{9\cdot 2 / \delta^3} > \frac{16}{\delta^3}$

and thus $\| \overline{\mathbf{w}}_j^t -\overline{\mathbf{w}}_j^t\| \leq \lambda^{1 - \left(1 + \frac{3\ell - 2}{3n_j}\right)\varepsilon}.$

For all $i \in [n]$ such that $i \neq i_j^\ell$ and if $\ell \neq 1$ then $i \neq i_j^{\ell - 1}$ , for all $t \in [\tau_j^\ell, \mathfrak{t}_j^\ell]$ we have

$$
\begin{array}{l} \left| \overline {{\mathrm{w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i} \right| \geq \delta - \left(\max _ {t ^ {\prime} \in \left[ \tau_ {j} ^ {\ell}, t \right]} \left\| \mathrm{d} \overline {{\mathrm{w}}} _ {j} ^ {t ^ {\prime}} / \mathrm{d} t ^ {\prime} \right\|\right) (t - \tau_ {j} ^ {\ell}) \quad \text { since } | \overline {{\mathrm{w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i} | \geq \delta \text { for } t = \tau_ {j} ^ {\ell} \\ \geq \delta - \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| (t - \tau_ {j} ^ {\ell}) \\ \geq \delta - \Delta^ {2} (t - \tau_ {j} ^ {\ell}) \\ > \delta - \Delta^ {2} \lambda^ {1 - 2 \varepsilon} \\ > \delta / 2 \\ \text { since } | \overline {{w}} _ {j} ^ {t} ^ {\top} \overline {{x}} _ {i} | \geq \delta \text { for } t = \tau_ {j} ^ {\ell} \\ \end{array}
$$

by properties of projection

by Proposition 17 (iii)

by the definition of $\mathbf{t}_j^\ell$

since $\lambda^{1 - 2\varepsilon}\leq n^{-9\cdot 6n\Delta^2 /\delta^3} <   \frac{\delta}{2\Delta^2},$

so for all $t \in (\mathfrak{t}_j^{\ell-1}, \min\{\mathfrak{t}_j^\ell, \mathsf{T}\}]$ we have $|\overline{\boldsymbol{w}}_j^t^\top \overline{\boldsymbol{x}}_i| > \delta/2 - \lambda^{1-(1+\frac{3\ell-2}{3n_j})\varepsilon} > \delta/4$ .

If $\ell \neq 1$ then for all $t\in (\mathfrak{t}_j^{\ell -1},\min \{\mathfrak{t}_j^\ell ,\mathsf{T}\})$ we have

$$
\begin{array}{l} \mathrm{d} s _ {j} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} / \mathrm{d} t \\ = \boldsymbol {g} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} - \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \boldsymbol {g} _ {j} ^ {t} \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} \quad \text { by   Corollary   10   and   since } I _ {0} (\boldsymbol {w} _ {j} ^ {t}) = \emptyset \\ > \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} - \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} \right| - 2 \lambda^ {2 - \varepsilon} \quad \text { since } \| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} - \boldsymbol {g} _ {j} ^ {t} \| <   \lambda^ {2 - \varepsilon} \\ > \frac {\delta^ {3}}{\sqrt {2} n} - \left\| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \right\| \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} \right| - 2 \lambda^ {2 - \varepsilon} \quad \text { recalling } \forall i \in [ n ]: \angle (\boldsymbol {v} ^ {*}, \boldsymbol {x} _ {i}) <   \pi / 4 \\ \geq \frac {\delta^ {3}}{\sqrt {2} n} - \Delta^ {2} \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} \right| - 2 \lambda^ {2 - \varepsilon} \quad \text { by   Proposition   17   (iii) } \\ > \frac {\delta^ {3}}{2 \sqrt {2} n} - \Delta^ {2} \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell - 1}} \right| \qquad \text { since } \lambda^ {2 - \varepsilon} \leq n ^ {- 9 \cdot 3 \cdot 7 n / \delta^ {3}} <   \frac {\delta^ {3}}{4 \sqrt {2} n}, \\ \end{array}
$$

so $i_j^{\ell - 1} \notin I_0(\boldsymbol{w}_j^t)$ at $t = \min \{\mathfrak{t}_j^\ell, \mathsf{T}\}$ since otherwise the continuous curve $\overline{\boldsymbol{w}}_j^{t^\top} \overline{\boldsymbol{x}}_{i_j^{\ell - 1}}$ would have different signs to the right of $\mathfrak{t}_j^{\ell - 1}$ and to the left of $\min \{\mathfrak{t}_j^\ell, \mathsf{T}\}$ without crossing zero in between.

Assume for a contradiction that $\mathsf{T} < \mathfrak{t}_j^{-\ell}$ . Then $I_0(\boldsymbol{w}_j^{\mathsf{T}}) = \{i_j^\ell\}$ . But also

$$
\begin{array}{l} \left| \overline {{\boldsymbol {w}}} _ {j} ^ {\mathsf {T}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| \geq \left| \overline {{\boldsymbol {w}}} _ {j} ^ {\mathsf {T}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| - \lambda^ {1 - \left(1 + \frac {3 \ell - 2}{3 n _ {j}}\right) \varepsilon} \quad \text { since } \| \overline {{\boldsymbol {w}}} _ {j} ^ {\mathsf {T}} - \overline {{\boldsymbol {w}}} _ {j} ^ {\mathsf {T}} \| \leq \lambda^ {1 - \left(1 + \frac {3 \ell - 2}{3 n _ {j}}\right) \varepsilon} \\ = \left| \overline {{{\boldsymbol {\alpha}}}} _ {j} ^ {\top} \overline {{{\boldsymbol {x}}}} _ {i _ {j} ^ {\ell}} \right| - \lambda^ {1 - \left(1 + \frac {3 \ell - 2}{3 n _ {j}}\right) \varepsilon} \quad \text { since } w _ {j} ^ {\top} = \boldsymbol {\alpha} _ {j} ^ {\top} \\ \geq \frac {2 \delta^ {4}}{3 n} (\tau_ {j} ^ {\ell} - \mathsf {T}) - \lambda^ {1 - \left(1 + \frac {3 \ell - 2}{3 n _ {j}}\right) \varepsilon} \quad \text { by   Proposition   17(vii)} \\ > \frac {2 \delta^ {4}}{3 n} \lambda^ {1 - \left(1 + \frac {3 \ell - 1}{3 n _ {j}}\right) \varepsilon} - \lambda^ {1 - \left(1 + \frac {3 \ell - 2}{3 n _ {j}}\right) \varepsilon} \quad \text { by   the   definition   of } t _ {j} ^ {- \ell} \\ = \lambda^ {1 - \left(1 + \frac {3 \ell - 1}{3 n _ {j}}\right) \varepsilon} \left(\frac {2 \delta^ {4}}{3 n} - \lambda^ {\frac {\varepsilon}{3 n _ {j}}}\right) \quad \text {   calculation   } \\ \end{array}
$$

$$
> 0 \quad \text { since } \lambda^ {\frac {\varepsilon}{3 n}} \leq n ^ {- 9 / \delta^ {3}} <   \frac {\delta^ {3 \cdot 7}}{n ^ {2}}.
$$

For all $t \in [\mathbf{t}_j^{-\ell}, \mathbf{t}_j^\ell]$ we have

$$
\begin{array}{l} \left| \mathrm{d} \overline {{\mathbf {w}}} _ {j} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \Big / \mathrm{d} t \right| \\ = \left| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} - \overline {{\mathsf {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \overline {{\mathsf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| \quad \text { by   the   definition   of } \mathsf {w} _ {j} ^ {t} \\ \geq \left| \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| - \left| \overline {{\mathrm{w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \overline {{\mathrm{w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| \quad \text { by   properties   of   absolute   value } \\ \end{array}
$$

$$
\begin{array}{l} > \frac {\delta^ {3}}{\sqrt {2} n} - \left| \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {j} ^ {\ell}} \overline {{\mathbf {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| \quad \text { recalling }   \forall i \in [ n ]: \angle (\boldsymbol {v} ^ {*}, \boldsymbol {x} _ {i}) <   \pi / 4 \\ \geq \frac {\delta^ {3}}{\sqrt {2} n} - \Delta^ {2} \left| \overline {{w}} _ {j} ^ {t ^ {\top}} \overline {{x}} _ {i _ {j} ^ {\ell}} \right| \quad \text { by   Proposition   17   (iii) } \\ \geq \frac {\delta^ {3}}{\sqrt {2} n} - \Delta^ {2} \left(\max _ {t ^ {\prime} \in [ \mathfrak {t} _ {j} ^ {- \ell}, \mathfrak {t} _ {j} ^ {\ell} ]} \left\| \mathrm{d} \overline {{\mathbf {w}}} _ {j} ^ {t ^ {\prime}} / \mathrm{d} t ^ {\prime} \right\|\right) | t - \tau_ {j} ^ {\ell} | \quad \text { since } \overline {{\mathbf {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} = 0 \text { at } t = \tau_ {j} ^ {\ell} \\ \geq \frac {\delta^ {3}}{\sqrt {2} n} - \Delta^ {4} | t - \tau_ {j} ^ {\ell} | \quad \text { by   properties   of   projection } \\ > \frac {\delta^ {3}}{\sqrt {2 n}} - \Delta^ {4} \lambda^ {1 - 2 \varepsilon} \quad \text { by   the   definitions   of } t _ {j} ^ {- \ell} \text { and } t _ {j} ^ {\ell} \\ > \frac {\delta^ {3}}{2 \sqrt {2} n} \quad \text { since } \lambda^ {1 - 2 \varepsilon} \leq n ^ {- 9 \cdot 6 n \Delta^ {2} / \delta^ {3}} <   \frac {\delta^ {3}}{2 \sqrt {2} n \Delta^ {4}} \\ > \lambda^ {\frac {\varepsilon}{3 n}} \quad \text { since } \lambda^ {\frac {\varepsilon}{3 n}} \leq n ^ {- 9 / \delta^ {3}}, \\ \end{array}
$$

so at $t = \mathfrak{t}_j^{-\ell}$ and at $t = \mathfrak{t}_j^\ell$ it holds that $\overline{\mathsf{w}}_j^{t^\top} \overline{\mathsf{x}}_{i_j^\ell}$ has different signs and absolute values greater than $\lambda^{\frac{\varepsilon}{3n}}\lambda^{1 - \left(1 + \frac{3\ell - 1}{3n_j}\right)\varepsilon} \geq \lambda^{1 - \left(1 + \frac{3\ell - 2}{3n_j}\right)\varepsilon}$ .

Assume for a contradiction that $T > t_{j}^{\ell}$ . Then at $t = t_{j}^{-\ell}$ and at $t = t_{j}^{\ell}$ it holds that $\|w_{j}^{t} - w_{j}^{t}\| \leq \lambda^{1 - \left(1 + \frac{3\ell - 2}{3n_{j}}\right)\varepsilon}$ , so $w_{j}^{t}^{\top} \overline{x}_{i_{j}}$ has different signs.

Therefore $\mathsf{T} \in [\mathfrak{t}_j^{-\ell}, \mathfrak{t}_j^\ell]$ and $I_0(\boldsymbol{w}_j^\mathsf{T}) = \{i_j^\ell\}$ . Let $t_j^\ell := \mathsf{T}$ .

To complete the proof, it suffices to show that, for all $t \in (t_j^\ell, \mathfrak{t}_j^\ell]$ , we have $\| \overline{\alpha}_j^t - \overline{\boldsymbol{w}}_j^t \| \leq \lambda^{1 - \left(1 + \frac{3\ell}{3n_j}\right)\varepsilon}$ , and if $\ell \neq n_j$ then $I_+(\boldsymbol{w}_j^t) = I_j^{\ell+1}$ and $I_0(\boldsymbol{w}_j^t) = \emptyset$ .

Since at $t = \min\{\tau_j^\ell, t_j^\ell\}$ we have $\| \overline{\alpha}_j^t - \overline{w}_j^t \| = \| \overline{w}_j^t - \overline{w}_j^t \| \leq \lambda^{1 - \left(1 + \frac{3\ell - 2}{3n_j}\right)\varepsilon}$ , and for almost all $t \in (\min\{\tau_j^\ell, t_j^\ell\}, t_j^\ell)$ we have

$$
\begin{array}{l} | \mathrm{d} \| \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} \| / \mathrm{d} t | \leq \| \mathrm{d} (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t}) / \mathrm{d} t \| \quad \text { by   properties   of   projection } \\ \leq \| \mathrm{d} \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} / \mathrm{d} t \| + \| \mathrm{d} \overline {{\boldsymbol {w}}} _ {j} ^ {t} / \mathrm{d} t \| \quad \text { by   the   triangle   inequality } \\ \leq \left(\left\| \boldsymbol {\gamma} _ {I + (\boldsymbol {\alpha} _ {j} ^ {t})} \right\| + \| \boldsymbol {g} _ {j} ^ {t} \|\right) \quad \text { by   properties   of   projection } \\ \leq \left(\left\| \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} \right\| + 2 \left\| \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {w} _ {j} ^ {t}) \cup I _ {0} (\boldsymbol {w} _ {j} ^ {t})} \right\|\right) \quad \text { since } \forall i \in [ n ]: | h _ {\boldsymbol {\theta} ^ {t}} (\boldsymbol {x} _ {i}) | \leq y _ {i} \\ \text { by   the   proof   of   Lemma   19(ii) } \\ \leq 3 \Delta^ {2} \quad \text { by   Proposition   17   (iii) }, \\ \end{array}
$$

it follows that for all $t \in [\min\{\tau_j^\ell, t_j^\ell\}, \mathfrak{t}_j^\ell]$ we have

$$
\begin{array}{l} \| \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} \| \leq \lambda^ {1 - \left(1 + \frac {3 \ell - 2}{3 n _ {j}}\right) \varepsilon} + 6 \Delta^ {2} \lambda^ {1 - \left(1 + \frac {3 \ell - 1}{3 n _ {j}}\right) \varepsilon} \quad \text { by   the   definitions   of } t _ {j} ^ {- \ell} \text { and } t _ {j} ^ {\ell} \\ <   7 \Delta^ {2} \lambda^ {1 - \left(1 + \frac {3 \ell - 1}{3 n _ {j}}\right) \varepsilon} \quad \text { since } \lambda^ {1 - \left(1 + \frac {3 \ell - 2}{3 n _ {j}}\right) \varepsilon} <   \lambda^ {1 - \left(1 + \frac {3 \ell - 1}{3 n _ {j}}\right) \varepsilon} \\ <   \lambda^ {1 - \left(1 + \frac {3 \ell}{3 n _ {j}}\right) \varepsilon} \quad \text { since } \lambda^ {- \frac {\varepsilon}{3 n}} \geq n ^ {9 \Delta^ {2}} > n ^ {3} \Delta^ {2 \cdot 6}. \\ \end{array}
$$

If $\ell \neq n_j$ then for all $i \neq i_j^\ell$ and all $t \in [\mathfrak{t}_j^{-\ell}, \mathfrak{t}_j^\ell]$ we have

$$
| \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i} | \geq \delta - \Delta^ {2} | t - \tau_ {j} ^ {\ell} | > \delta - \Delta^ {2} \lambda^ {1 - 2 \varepsilon} > \delta / 2 ,
$$

so for all $t \in (t_j^\ell, \mathfrak{t}_j^\ell]$ we have $|\overline{\boldsymbol{w}}_j^t^\top \overline{\boldsymbol{x}}_i| > \delta/2 - \lambda^{1 - \left(1 + \frac{3\ell}{3n_j}\right)\varepsilon} > \delta/4$ . Also, for almost all $t \in (t_j^\ell, \mathfrak{t}_j^\ell)$ the following holds, where $I := I_j^\ell \cap I_j^{\ell+1}$ :

$$
\begin{array}{l} \mathrm{d} s _ {j} \overline {{\boldsymbol {w}}} _ {j} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} / \mathrm{d} t \\ = \boldsymbol {g} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} - \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \boldsymbol {g} _ {j} ^ {t} \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \\ > \mathbf {g} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} - \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \mathbf {g} _ {j} ^ {t} \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| - 2 \lambda^ {2 - \varepsilon} \\ \geq \boldsymbol {\gamma} _ {I} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I} \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| - 2 \lambda^ {2 - \varepsilon} \\ \geq \boldsymbol {\gamma} _ {I} ^ {\top} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} - \| \boldsymbol {\gamma} _ {I} \| \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| - 2 \lambda^ {2 - \varepsilon} \\ > \frac {\delta^ {3}}{\sqrt {2} n} - \| \boldsymbol {\gamma} _ {I} \| \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| - 2 \lambda^ {2 - \varepsilon} \\ \geq \frac {\delta^ {3}}{\sqrt {2} n} - \Delta^ {2} \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| - 2 \lambda^ {2 - \varepsilon} \\ > \frac {\delta^ {3}}{2 \sqrt {2} n} - \Delta^ {2} \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i _ {j} ^ {\ell}} \right| \\ \end{array}
$$

by Corollary 10

since $\frac{1}{n}\sum_{i=1}^{n}|h_{\boldsymbol{\theta}^t}(\boldsymbol{x}_i)|\|\boldsymbol{x}_i\| < \lambda^{2-\varepsilon}$ ,

where $\exists \varsigma_j^t\in [0,1]\colon \mathbf{g}_j^t = \boldsymbol{\gamma}_I + \frac{1}{n} y_{i_j^\ell}\varsigma_j^t\boldsymbol{x}_{i_j^\ell}$

since $\boldsymbol{x}_{i_j^{\ell}}^\top \overline{\boldsymbol{x}}_{i_j^{\ell}}\geq \overline{\boldsymbol{w}}_j^{t^\top}\boldsymbol{x}_{i_j^{\ell}}\left|\overline{\boldsymbol{w}}_j^{t^\top}\overline{\boldsymbol{x}}_{i_j^{\ell}}\right|$

since $\overline{\boldsymbol{w}}_j^t^\top \gamma_I\leq \| \gamma_I\|$

recalling $\forall i\in [n]\colon \angle (\pmb {v}^{*},\pmb{x}_{i}) <   \pi /4$

by Proposition 17 (iii)

since $\lambda^{2 - \varepsilon}\leq n^{-9\cdot 3\cdot 7n / \delta^3} <   \frac{\delta^3}{4\sqrt{2}n}.$

Hence $i_{j}^{\ell} \notin I_{0}\left(\boldsymbol{w}_{j}^{t'}\right)$ for all $t' \in (t_{j}^{\ell}, t_{j}^{\ell}]$ since otherwise the continuous curve $\overline{w}_{j}^{t}^{\top} \overline{x}_{i_{j}^{\ell}}$ would have different signs to the right of $t_{j}^{\ell}$ and to the left of the smallest such $t'$ without crossing zero in between. ☐

For each positive-sign hidden neuron, after the completion of all of the intermediate alignment stages (if any), its yardstick vector proceeds to align to the vector $\gamma_{[n]}$ . For the cosine of the angle between them, whose starting point is the angle $\varphi_j^{n_j^+}$ defined below, from the proof of Proposition 14 (vi) we obtain the following expression, which will be useful in the proof of the next lemma. We also remark that the angle $\varphi_j^{n_j^+}$ already featured in Proposition 14 (viii).

Proposition 22. For all $j \in J_{+}$ and all $t > \tau_{j}^{n_{j}}$ we have $I_{+}(\boldsymbol{\alpha}_{j}^{t}) = [n]$ and

$$
\cos \varphi_ {j} ^ {t} = \tanh \left(\operatorname{artanh} \cos \varphi_ {j} ^ {n _ {j} ^ {+}} + \| \boldsymbol {\gamma} _ {[ n ]} \| (t - \tau_ {j} ^ {n _ {j}})\right),
$$

where $\varphi_j^{n_j^+} := \lim_{t \to (\tau_j^{n_j})^+} \varphi_j^t$ .

The final lemma in this section establishes non-asymptotic bounds for the final alignment stage in the first phase of the training, which we consider to end at time $T_{1}$ . During it, each positive-sign hidden neuron: continues to grow but keeps its length below $2\|z_{j}\|\lambda^{1-\varepsilon}$ , aligns to the vector $\gamma_{[n]}$ up to a cosine of at least $1-\lambda^{\varepsilon}$ , and maintains bounded by $\lambda^{1-3\varepsilon}$ the difference between the logarithm of its length divided by the initialisation scale and the logarithm of the corresponding yardstick vector length. Establishing the latter bound, which is stated in part (iv) and will be instrumental in the proof of the implicit bias (cf. Lemma 34), involves putting together the bounds in Lemma 21 (iii), (iv), and (v), and Lemma 23 (i) and (ii) on the dynamics-governing vectors, the unit-sphere normalisations, and the boundary crossing times, over the lengthy time period up to $T_{1}$ which depends on the initialisation scale $\lambda$ .

Lemma 23. For all $j \in J_{+}$ we have:

(i) $\| \pmb{\gamma}_{[n]} - \pmb{g}_j^t\| \leq \lambda^{2 - 3\varepsilon}$ for all $t\in (t_j^{n_j},T_1];$

(ii) $\| \overline{\boldsymbol{\alpha}}_j^t -\overline{\boldsymbol{w}}_j^t\| \leq \lambda^{1 - 2\varepsilon}$ for all $t\in (t_j^{n_j},T_1];$

(iii) $\overline{\boldsymbol{w}}_j^{T_1^\top}\overline{\boldsymbol{\gamma}}_{[n]}\geq 1 - \lambda^{\varepsilon};$

(iv) $|\ln \| \pmb{\alpha}_j^{T_1}\| -\ln \| \pmb{w}_j^{T_1} / \lambda \| |\leq \lambda^{1 - 3\varepsilon}.$

Proof. For all $t \in [0, T_1]$ we have

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} | h _ {\boldsymbol {\theta} ^ {t}} (\boldsymbol {x} _ {i}) | \| \boldsymbol {x} _ {i} \| \leq m \left(\max _ {j = 1} ^ {m} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2}\right) \left(\max _ {i = 1} ^ {n} \| \boldsymbol {x} _ {i} \| ^ {2}\right) \quad \text { by   the   definition   of } h _ {\boldsymbol {\theta} ^ {t}} \text { in   section } 2
$$

$$
<   4 m \Delta^ {4} \lambda^ {2 - 2 \varepsilon} \quad \text { by   Lemma } 1 9
$$

$$
<   \lambda^ {2 - 3 \varepsilon} \quad \text { since } \lambda^ {- \varepsilon} \geq m ^ {3} n ^ {9 \cdot 3 n \Delta^ {2} / \delta^ {3}} > 4 m \Delta^ {4}.
$$

Suppose $j \in J_{+}$ .

Assume for a contradiction that $I_0(\boldsymbol{w}_j^t) \neq \emptyset$ for some $t \in (t_j^{n_j}, T_1]$ , and let $\mathsf{T} > t_j^{n_j}$ be the smallest such that $\boldsymbol{w}_j^\top \boldsymbol{x}_i = 0$ for some $i \in [n]$ . Then for all $t \in (t_j^{n_j}, \mathsf{T})$ we have

$$
\begin{array}{l} \mathrm{d} \overline {{{\boldsymbol {w}}}} _ {j} ^ {t ^ {\top}} \overline {{{\boldsymbol {x}}}} _ {i} / \mathrm{d} t = \boldsymbol {g} _ {j} ^ {t ^ {\top}} \overline {{{\boldsymbol {x}}}} _ {i} - \overline {{{\boldsymbol {w}}}} _ {j} ^ {t ^ {\top}} \boldsymbol {g} _ {j} ^ {t} \overline {{{\boldsymbol {w}}}} _ {j} ^ {t ^ {\top}} \overline {{{\boldsymbol {x}}}} _ {i} \quad \text { by   Corollary   10   and   since } I _ {0} (\boldsymbol {w} _ {j} ^ {t}) = \emptyset \\ > \boldsymbol {\gamma} _ {[ n ]} ^ {\top} \overline {{\boldsymbol {x}}} _ {i} - \| \boldsymbol {\gamma} _ {[ n ]} \| \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i} - 2 \lambda^ {2 - 3 \varepsilon} \quad \text { since } \| \boldsymbol {\gamma} _ {[ n ]} - \boldsymbol {g} _ {j} ^ {t} \| <   \lambda^ {2 - 3 \varepsilon} \\ \geq \delta^ {3} - \Delta^ {2} \overline {{\boldsymbol {w}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {x}}} _ {i} - 2 \lambda^ {2 - 3 \varepsilon} \quad \text { by   Proposition   17   (iii) } \\ > \frac {\delta^ {3}}{2} - \Delta^ {2} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {x}}} _ {i} \quad \text { since } \lambda^ {2 - 3 \varepsilon} \leq n ^ {- 9 \cdot 3 \cdot 5 n / \delta^ {3}} <   \frac {\delta^ {3}}{4}, \\ \end{array}
$$

so the continuous curve $\overline{\boldsymbol{w}}_j^t^\top \overline{\boldsymbol{x}}_i$ is positive to the right of $t_j^{n_j}$ , negative to the left of $\mathsf{T}$ , and does not cross zero in between.

Hence $I_0(\boldsymbol{w}_j^t) = \emptyset$ for all $t \in (t_j^{n_j}, T_1]$ , which together with the inequality $\frac{1}{n} \sum_{i=1}^{n} |h_{\boldsymbol{\theta}^t}(\boldsymbol{x}_i)| \| \boldsymbol{x}_i \| < \lambda^{2-3\varepsilon}$ establishes part (i).

By Lemma 21 (iv), for all $t \in [t_j^{n_j}, \max\{t_j^{n_j}, \tau_j^{n_j}\}]$ we have $1 - \overline{\alpha}_j^t^\top \overline{w}_j^t \leq \frac{1}{2} \lambda^{2-4\varepsilon}$ .

Moreover, for all $t \in (\max \{t_j^{n_j}, \tau_j^{n_j}\}, T_1)$ we have

$$
\begin{array}{l} \mathrm{d} (1 - \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) / \mathrm{d} t \\ = - \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {[ n ]} + \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t} (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {[ n ]} + \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t}) \quad \text { by   Corollary   10   and   Proposition   22 } \\ <   2 \lambda^ {2 - 3 \varepsilon} - (1 - \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} + \overline {{\boldsymbol {w}}} _ {j} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {[ n ]} \quad \text { since } \| \boldsymbol {\gamma} _ {[ n ]} - \boldsymbol {g} _ {j} ^ {t} \| <   \lambda^ {2 - 3 \varepsilon} \\ \leq 2 \lambda^ {2 - 3 \varepsilon} - (1 - \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) \frac {\delta^ {7 / 2}}{\sqrt {2 n}} \quad \text { by   Proposition   14   (vi),   (vii),   and   (viii), }   \text { and   Proposition   17   (iii) } \\ \leq 4 \lambda^ {\varepsilon} \left(\frac {1}{2} \lambda^ {2 - 4 \varepsilon} - \left(1 - \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t ^ {\top}} \overline {{\boldsymbol {w}}} _ {j} ^ {t}\right)\right) \quad \text { since } \lambda^ {- \varepsilon} \geq n ^ {9 \cdot 3 n / \delta^ {3}} > \frac {4 \sqrt {2} n}{\delta^ {7 / 2}}. \\ \end{array}
$$

Hence for all $t \in (t_{j}^{n_{j}}, T_{1}]$ we have $1 - \overline{\alpha}_{j}^{t^{\top}} \overline{w}_{j}^{t} \leq \frac{1}{2} \lambda^{2-4\varepsilon}$ and thus $\|\overline{\alpha}_{j}^{t} - \overline{w}_{j}^{t}\| \leq \lambda^{1-2\varepsilon}$ , establishing part (ii).

By Proposition 18 and Proposition 22 we have

$$
\overline {{{{\boldsymbol {\alpha}}}}} _ {j} ^ {T _ {1} \top} \overline {{{{\boldsymbol {\gamma}}}}} _ {[ n ]} > \tanh \left(\frac {2}{3} \| \boldsymbol {\gamma} _ {[ n ]} \| T _ {1}\right) = \tanh \left(\frac {2 \varepsilon}{3} \ln \left(\frac {1}{\lambda}\right)\right) > 1 - 2 \lambda^ {\frac {4 \varepsilon}{3}},
$$

so $\overline{\boldsymbol{w}}_{j}^{T_1^\top}\overline{\boldsymbol{\gamma}}_{[n]} > 1 - 2\lambda^{\frac{4\varepsilon}{3}} - \lambda^{1 - 2\varepsilon} = 1 - \lambda^{\varepsilon}\big(2\lambda^{\frac{\varepsilon}{3}} + \lambda^{1 - 3\varepsilon}\big) > 1 - \lambda^{\varepsilon}$ , establishing part (iii).

Recalling Lemma 21, for all $t \in [0, T_1]$ we have

$$
\left\| \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} - \boldsymbol {g} _ {j} ^ {t} \right\| \leq \left\{ \begin{array}{l l} \frac {\Delta^ {2}}{n} + \lambda^ {2 - 3 \varepsilon} <   \frac {2 \Delta^ {2}}{n} & \text { if } \exists \ell \in [ n _ {j} ]: | t - \tau_ {j} ^ {\ell} | \leq \lambda^ {1 - 2 \varepsilon}, \\ \lambda^ {2 - 3 \varepsilon} & \text { otherwise } \end{array} \right.
$$

$$
\| \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} \| \leq \lambda^ {1 - 2 \varepsilon},
$$

so for almost all $t \in [0, T_1]$ we have

$$
\begin{array}{l} \left| \frac {\mathrm{d} \ln \| \boldsymbol {\alpha} _ {j} ^ {t} \|}{\mathrm{d} t} - \frac {\mathrm{d} \ln \| \boldsymbol {w} _ {j} ^ {t} / \lambda \|}{\mathrm{d} t} \right| \\ = \left| \overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} ^ {\top} \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} _ {j} ^ {t} \right| \quad \text { by   Corollary   10 } \\ \leq \left| \left(\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t}\right) ^ {\top} \boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} \right| + \left| \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \left(\boldsymbol {\gamma} _ {I _ {+} (\boldsymbol {\alpha} _ {j} ^ {t})} - \boldsymbol {g} _ {j} ^ {t}\right) \right| \quad \text { by   properties   of   absolute   value } \\ \leq \lambda^ {1 - 2 \varepsilon} \Delta^ {2} + \left\{ \begin{array}{l l} \frac {2 \Delta^ {2}}{n} & \text { if } \exists \ell \in [ n _ {j} ]: | t - \tau_ {j} ^ {\ell} | \leq \lambda^ {1 - 2 \varepsilon}, \\ \lambda^ {2 - 3 \varepsilon} & \text { otherwise } \end{array} \right. \quad \text { by   Proposition   17   (iii) } \\ <   \left\{ \begin{array}{l l} \frac {3 \Delta^ {2}}{n} & \text { if } \exists \ell \in [ n _ {j} ] \colon | t - \tau_ {j} ^ {\ell} | \leq \lambda^ {1 - 2 \varepsilon}, \\ \lambda^ {1 - \frac {5 \varepsilon}{2}} & \text { otherwise } \end{array} \right. \quad \text { since } \lambda^ {- \frac {\varepsilon}{2}} \geq n ^ {\frac {9 \cdot 3}{2} n \Delta^ {2}} > 2 \Delta^ {2}, \\ \end{array}
$$

and therefore

$$
\begin{array}{l} \left| \ln \left\| \boldsymbol {\alpha} _ {j} ^ {T _ {1}} \right\| - \ln \left\| \boldsymbol {w} _ {j} ^ {T _ {1}} / \lambda \right\| \right| \\ \leq 2 n \lambda^ {1 - 2 \varepsilon} \frac {3 \Delta^ {2}}{n} + (T _ {1} - 2 n \lambda^ {1 - 2 \varepsilon}) \lambda^ {1 - \frac {5 \varepsilon}{2}} \quad \text { by   the   previous   inequality } \\ <   6 \Delta^ {2} \lambda^ {1 - 2 \varepsilon} + T _ {1} \lambda^ {1 - \frac {5 \varepsilon}{2}} \quad \text {   omitting   the   negative   term   } \\ <   \frac {1}{2} \lambda^ {1 - 3 \varepsilon} + \frac {1}{\delta^ {2}} \lambda^ {1 - \frac {5 \varepsilon}{2}} \ln (1 / \lambda^ {\varepsilon}) \quad \text { since } \lambda^ {- \varepsilon} > 1 2 \Delta^ {2} \text { and   by   Proposition   17   (iii) } \\ <   \frac {1}{2} \lambda^ {1 - 3 \varepsilon} + \frac {3}{\delta^ {2}} \lambda^ {1 - \frac {8 \varepsilon}{3}} \quad \text { since } 3 \lambda^ {- \frac {\varepsilon}{6}} = 3 \sqrt [ 6 ]{1 / \lambda^ {\varepsilon}} > \ln (1 / \lambda^ {\varepsilon}) \\ <   \lambda^ {1 - 3 \varepsilon} \quad \text { since } \lambda^ {- \frac {\varepsilon}{3}} \geq n ^ {9 n / \delta^ {3}} > \frac {6}{\delta^ {2}}, \\ \end{array}
$$

establishing part (iv).

![](images/c334f1e9b17d006cfe46d48baa0a077171345b91f22b4e95b5b3a5387b7e22ea.jpg)

# E Proofs for the second phase

Here we prove a number of results which culminate in Lemma 32 below, whose parts (ii) and (v) establish Theorem 6.

We begin by observing that the eigenvector of the largest eigenvalue of the matrix $\frac{1}{n}XX^{\top}$ is in the interior of the cone spanned by the training points.

Proposition 24. $u_{1} \in \text{int}(\text{cone}\{x_{1}, \ldots, x_{n}\})$ .

Proof. If $\boldsymbol{v} \in \text{cone}\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\} \setminus \{\boldsymbol{0}\}$ , i.e. $\boldsymbol{v} = \sum_{i=1}^{n} \beta_i \boldsymbol{x}_i$ for some $\beta_1, \ldots, \beta_n \geq 0$ that are not all zero, then

$$
\frac {1}{n} \boldsymbol {X} \boldsymbol {X} ^ {\top} \boldsymbol {v} = \frac {1}{n} \sum_ {i ^ {\prime} = 1} ^ {n} \left(\sum_ {i = 1} ^ {n} \beta_ {i} \boldsymbol {x} _ {i ^ {\prime}} ^ {\top} \boldsymbol {x} _ {i}\right) \boldsymbol {x} _ {i ^ {\prime}} \in \operatorname{int} (\operatorname{cone} \{\boldsymbol {x} _ {1}, \dots , \boldsymbol {x} _ {n} \}).
$$

Thus $\frac{1}{n} X X^{\top}$ maps cone $\{x_1, \ldots, x_n\} \setminus \{0\}$ into int(cone $\{x_1, \ldots, x_n\}$ ).

Let $v_{0} := v^{*}$ , and $v_{\ell+1} := \frac{1}{n} X X^{\top} v_{\ell}$ for all $\ell \in N$ .

Recalling Proposition 16, we have $\boldsymbol{v}_1 = \gamma_{[n]} \in \mathrm{int}(\mathrm{cone}\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\}) \subseteq \mathrm{cone}\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\} \setminus \{\boldsymbol{0}\}$ , and so $\boldsymbol{v}_{\ell} \in \mathrm{int}(\mathrm{cone}\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\})$ for all $\ell \geq 1$ .

Since $\boldsymbol{v}_{\ell} = \sum_{k=1}^{d} \eta_k^\ell \nu_k \boldsymbol{u}_k$ and $\eta_1$ is strictly the largest eigenvalue, we have that $\angle (\boldsymbol{u}_1, \boldsymbol{v}_{\ell}) \to 0$ as $\ell \to \infty$ . Therefore $\boldsymbol{u}_1 \in \mathrm{cone}\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\} \setminus \{\mathbf{0}\}$ , but since $\frac{1}{n} \boldsymbol{X} \boldsymbol{X}^\top \boldsymbol{u}_1 = \eta_1 \boldsymbol{u}_1$ , in fact $\boldsymbol{u}_1 \in \mathrm{int}(\mathrm{cone}\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\})$ .

Our next observation is that the key set S defined in section 5 is strictly contained in the ball with centre $v^{*}/2$ and radius $\|v^{*}\|/2 = 1/2$ , i.e. that passes through the origin and the teacher neuron, and is centred half-way between them.

Proposition 25. For all $v \in S$ we have $\boldsymbol{v}^{\top}(\boldsymbol{v}^{*} - \boldsymbol{v}) > 0$ .

Proof. Suppose $\pmb{v} = \sum_{k=1}^{d} \nu_k \pmb{u}_k \in S_\ell$ for some $\ell \in [d]$ . Then

$$
\boldsymbol {v} ^ {\top} \left(\boldsymbol {v} ^ {*} - \boldsymbol {v}\right) = \sum_ {k = 1} ^ {d} \nu_ {k} \left(\nu_ {k} ^ {*} - \nu_ {k}\right) > \sum_ {k = 1} ^ {d} \frac {\eta_ {k}}{\eta_ {\ell}} \nu_ {k} \left(\nu_ {k} ^ {*} - \nu_ {k}\right) = \frac {1}{\eta_ {\ell}} \boldsymbol {v} ^ {\top} \frac {1}{n} \boldsymbol {X} \boldsymbol {X} ^ {\top} \left(\boldsymbol {v} ^ {*} - \boldsymbol {v}\right) > 0.
$$

![](images/fa8f5fb7b1ffd24366d092e33889933c00d8b3b68477fea52ab3f795d512b6e5.jpg)

The angles between a vector v in S and the vector obtained by applying the operator $\frac{1}{n}XX^{\top}$ to the vector $v^{*}-v$ will be important in what follows. We now show that, if v is in the subset $S_{1}$ (also defined in section 5), then the cosine of that angle has a positive lower bound that does not depend on the initialisation scale $\lambda$ .

Proposition 26. For all $\pmb{v} = \sum_{k=1}^{d} \nu_k \pmb{u}_k \in S_1$ we have

$$
\overline {{\boldsymbol {v}}} ^ {\top} \overline {{\boldsymbol {X} \boldsymbol {X} ^ {\top} (\boldsymbol {v} ^ {*} - \boldsymbol {v})}} > \frac {1}{2} \left(\frac {\eta_ {d} \nu_ {d} ^ {*}}{\| \boldsymbol {\gamma} _ {[ n ]} \|}\right) ^ {2}.
$$

Proof. Observe that

$$
\begin{array}{l} \overline {{\boldsymbol {v}}} ^ {\top} \overline {{\boldsymbol {X} \boldsymbol {X} ^ {\top} (\boldsymbol {v} ^ {*} - \boldsymbol {v})}} = \overline {{\frac {\nu_ {1} ^ {*}}{\nu_ {1}} \boldsymbol {v}}} ^ {\top} \overline {{\frac {1}{n} \boldsymbol {X} \boldsymbol {X} ^ {\top} \frac {\nu_ {d} ^ {*}}{\nu_ {d} ^ {*} - \nu_ {d}} (\boldsymbol {v} ^ {*} - \boldsymbol {v})}} \\ > \frac {\eta_ {1} \sum_ {k = 1} ^ {d} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right)}{\frac {\nu_ {1}}{\nu_ {1} ^ {*}} \left(1 - \frac {\nu_ {d}}{\nu_ {d} ^ {*}}\right) \| \boldsymbol {\gamma} _ {[ n ]} \| ^ {2}} \\ > \eta_ {1} \eta_ {d} \frac {\frac {\nu_ {d}}{\nu_ {d} ^ {*}}}{\frac {\nu_ {1}}{\nu_ {1} ^ {*}}} \frac {\nu_ {d} ^ {* 2}}{\| \boldsymbol {\gamma} _ {[ n ]} \| ^ {2}} \\ > \frac {1}{2} \left(\frac {\eta_ {d} \nu_ {d} ^ {*}}{\| \boldsymbol {\gamma} _ {[ n ]} \|}\right) ^ {2}. \\ \end{array}
$$

![](images/ac6f136463dc8c4daaac1b530660fe52deb506570af221d26c648f0de043f628.jpg)

Our final preparatory result is a positive lower bound, however depending on $\lambda$ , on the cosine of every angle between a vector v in S and a training point. The proof relies on the correlation property of our datasets, i.e. that the angles between the training points and the teacher neuron are less than $\pi/4$ .

Proposition 27. $\overline{v}^{\top}\overline{x}_{i} > \sqrt{8}\lambda^{\varepsilon/2}$ for all $v \in S$ and all $i \in [n]$ .

Proof. Suppose $v = \sum_{k=1}^{d} \nu_k u_k \in S$ .

It suffices to establish that $\cos\angle(\boldsymbol{v}^{*},\boldsymbol{v})\geq\frac{1}{\sqrt{2}}+2\lambda^{\varepsilon/2}$ , because it implies that for all $i\in[n]$ we have

$$
\begin{array}{l} \cos \angle (\boldsymbol {v}, \boldsymbol {x} _ {i}) \geq \cos \angle (\boldsymbol {v} ^ {*}, \boldsymbol {v}) \cos \angle (\boldsymbol {v} ^ {*}, \boldsymbol {x} _ {i}) - \sin \angle (\boldsymbol {v} ^ {*}, \boldsymbol {v}) \sin \angle (\boldsymbol {v} ^ {*}, \boldsymbol {x} _ {i}) \\ > \frac {1}{\sqrt {2}} \left(\frac {1}{\sqrt {2}} + 2 \lambda^ {\varepsilon / 2}\right) - \frac {1}{\sqrt {2}} \sqrt {1 - \left(\frac {1}{\sqrt {2}} + 2 \lambda^ {\varepsilon / 2}\right) ^ {2}} \\ = \frac {1}{2} + \sqrt {2} \lambda^ {\varepsilon / 2} - \sqrt {\frac {1}{2} - \left(\frac {1}{2} + \sqrt {2} \lambda^ {\varepsilon / 2}\right) ^ {2}} \\ > \frac {1}{2} + \sqrt {2} \lambda^ {\varepsilon / 2} - \sqrt {\frac {1}{4} - \sqrt {2} \lambda^ {\varepsilon / 2}} \\ > \frac {1}{2} + \sqrt {2} \lambda^ {\varepsilon / 2} - \left(\frac {1}{2} - \sqrt {2} \lambda^ {\varepsilon / 2}\right) \\ = \sqrt {8} \lambda^ {\varepsilon / 2}. \\ \end{array}
$$

By Proposition 24, we have $\pmb{u}_1^\top \pmb{v}^* > 1 / \sqrt{2}$ .

If $\pmb{v}\in S_1$ then

$$
\left\| \boldsymbol {v} ^ {*} - \frac {\nu_ {1} ^ {*}}{\nu_ {1}} \boldsymbol {v} \right\| ^ {2} = \sum_ {k = 2} ^ {d} \left(1 - \frac {\nu_ {1} ^ {*}}{\nu_ {1}} \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {2} \nu_ {k} ^ {* 2} <   \sum_ {k = 2} ^ {d} \left(1 - \frac {\eta_ {k}}{2 \eta_ {1}}\right) ^ {2} \nu_ {k} ^ {* 2} \leq \left(1 - \frac {\eta_ {d}}{2 \eta_ {1}}\right) ^ {2} \| \boldsymbol {v} ^ {*} - \nu_ {1} ^ {*} \boldsymbol {u} _ {1} \| ^ {2},
$$

so we have

$$
\begin{array}{l} \cos \angle (\pmb {v} ^ {*}, \pmb {v}) > \sqrt {1 - \frac {1}{2} \left(1 - \frac {\eta_ {d}}{2 \eta_ {1}}\right) ^ {2}} = \sqrt {\frac {1}{2} + \frac {\eta_ {d}}{2 \eta_ {1}} - \frac {\eta_ {d} ^ {2}}{8 \eta_ {1} ^ {2}}} > \sqrt {\frac {1}{2} + \frac {3 \eta_ {d}}{8 \eta_ {1}}} \\ > \frac {1}{\sqrt {2}} + (2 - \sqrt {2}) \frac {3 \eta_ {d}}{8 \eta_ {1}} \geq \frac {1}{\sqrt {2}} + (2 - \sqrt {2}) \frac {3 \delta^ {2}}{8 \Delta^ {2}} > \frac {1}{\sqrt {2}} + 2 \lambda^ {\varepsilon / 2}. \\ \end{array}
$$

Otherwise $\boldsymbol{v} \in S_{\ell}$ for some $\ell \neq 1$ . Then $(\boldsymbol{v} - \nu_1^*\boldsymbol{u}_1)^\top (\boldsymbol{v}^* - \boldsymbol{v}) > \boldsymbol{v}^\top (\boldsymbol{v}^* - \boldsymbol{v}) > 0$ by Proposition 25. Also $(\boldsymbol{v} - \nu_1^*\boldsymbol{u}_1)^\top (\boldsymbol{v}^* - \nu_1^*\boldsymbol{u}_1) = \sum_{k=2}^{d} \nu_k \nu_k^* > \sum_{k=2}^{d} \frac{\eta_k}{2\eta_1} \nu_k^{*2} \geq \frac{\eta_d}{2\eta_1} \| \boldsymbol{v}^* - \nu_1^*\boldsymbol{u}_1\|^2$ . Hence

$$
\| \pmb {v} ^ {*} - \pmb {v} \| ^ {2} \leq \| \pmb {v} ^ {*} - \nu_ {1} ^ {*} \pmb {u} _ {1} \| ^ {2} - \| \pmb {v} - \nu_ {1} ^ {*} \pmb {u} _ {1} \| ^ {2} <   \left(1 - \left(\frac {\eta_ {d}}{2 \eta_ {1}}\right) ^ {2}\right) \| \pmb {v} ^ {*} - \nu_ {1} ^ {*} \pmb {u} _ {1} \| ^ {2},
$$

so we have

$$
\begin{array}{l} \cos \angle (\boldsymbol {v} ^ {*}, \boldsymbol {v}) > \sqrt {\frac {1}{2} + \frac {1}{2} \left(\frac {\eta_ {d}}{2 \eta_ {1}}\right) ^ {2}} > \frac {1}{\sqrt {2}} + \frac {2 - \sqrt {2}}{2} \left(\frac {\eta_ {d}}{2 \eta_ {1}}\right) ^ {2} \\ \geq \frac {1}{\sqrt {2}} + \frac {2 - \sqrt {2}}{2} \bigg (\frac {\delta^ {2}}{2 \Delta^ {2}} \bigg) ^ {2} > \frac {1}{\sqrt {2}} + 2 \lambda^ {\varepsilon / 2}. \\ \end{array}
$$

For all $t \geq T_1$ , recall from section 5 that $\boldsymbol{v}^t = \sum_{j \in J_+} a_j^t \boldsymbol{w}_j^t$ , and let

$$
\boldsymbol {g} ^ {t} := \frac {1}{n} \boldsymbol {X} \boldsymbol {X} ^ {\top} (\boldsymbol {v} ^ {*} - \boldsymbol {v} ^ {t})
$$

$$
\boldsymbol {f} ^ {t} := \| \boldsymbol {v} ^ {t} \| (\boldsymbol {g} ^ {t} + \overline {{\boldsymbol {v}}} ^ {t} \overline {{\boldsymbol {v}}} ^ {t ^ {\top}} \boldsymbol {g} ^ {t}).
$$

The next lemma is at the heart of our analysis of the training dynamics. It establishes several key facts that hold at all times t from the start $T_{1}$ of the second phase, and which form the statement of the lemma as follows.

Parts (i) and (ii). The cosines of all angles between hidden neurons that form the aligned bundle remain above $1 - 4\lambda^{\varepsilon}$ , and the bundle vector $v^{t}$ which was defined as the sum of the constituent hidden neurons multiplied by their last-layer weights stays in the set S. These two properties support each other, e.g. we show that the containment in S implies that the gradients of the individual hidden neurons are such that the bundle keeps together rather than breaks apart.

Parts (iii) and (iv). The network acts linearly on the training points, namely its outputs for the training points equal their inner products with the bundle vector. Moreover, the vectors $g_{j}^{t}$ (defined in Proposition 1 (i)) that govern the dynamics are all equal to the vector $g^{t}$ which is obtained by applying the operator $\frac{1}{n}X X^{\top}$ to the vector $v^{*} - v^{t}$ .

Parts (v) and (vi). The derivative of the bundle vector $v^{t}$ with respect to the time t exists, i.e. the issue of the non-differentiability of the ReLU activation at 0 does not arise in this respect. However, the two-layer dynamics is such that this derivative is in general only approximated by the vector $f^{t}$ defined above, and we bound that error by a ball centred at $f^{t}$ whose radius depends on the initialisation scale $\lambda$ .

Parts (vii), (viii) and (ix). We show that the squared norm of $v^{t}$ grows exponentially fast as it moves away from the saddle at the origin, obtaining a lower bound on the speed of its increase that does not depend on $\lambda$ as long as $v^{t}$ is in the subset $S_{1}$ , and a lower bound that depends on $\lambda$ subsequently. Also we show an upper bound on the speed of decrease of the squared norm of $v^{*} - v$ , i.e. the square of the distance between the bundle vector and the teacher neuron.

Perhaps the most involved segment of the proof proceeds by showing that each face of the boundary of the set S is repelling towards the interior of S with respect to the dynamics of the bundle vector $v^{t}$ , whose derivative is approximately $f^{t}$ . A major complication is that this is in general not true for the entire boundary of the “padded ellipsoid” constraint $\Xi$ , but holds for its remainder after the slicing off by the other constraints that define S.

Lemma 28. For all $t \geq T_{1}$ we have:

(i) $1 - \overline{\boldsymbol{w}}_j^t^\top \overline{\boldsymbol{w}}_{j'}^t < 4\lambda^\varepsilon$ for all $j, j' \in J_+$ ;

(ii) $v^{t} \in S;$

(iii) $h_{\pmb{\theta}^t}(\pmb{x}_i) = \pmb{v}^{t^\top}\pmb{x}_i$ for all $i\in [n]$ ;

(iv) $\pmb{g}_j^t = \pmb{g}^t$ for all $j\in J_{+}$ ;

(v) $v^{t}$ is differentiable at t;

(vi) $\| \mathrm{d}\pmb {v}^t /\mathrm{d}t - \pmb {f}^t\| \leq 3\lambda^{\varepsilon /2}\| \pmb {f}^{t}\|$ ;

(vii) $\mathrm{d}\| \pmb{v}^t\|^2/\mathrm{d}t \geq (\eta_d\nu_d^*/\|\pmb{\gamma}_{[n]}\|)^2\|\pmb{v}^t\|^2\|\pmb{g}^t\|$ if $\pmb{v}^t \in S_1$ ;

(viii) $\mathrm{d}\| \pmb{v}^t\|^2/\mathrm{d}t \geq 3\lambda^{\varepsilon/3}\|\pmb{v}^t\|^2\|\pmb{g}^t\|$ ;

(ix) $\mathrm{d}\| \pmb{v}^{*} - \pmb{v}^{t}\|^{2} / \mathrm{d}t\geq -5\eta_{1}\| \pmb{v}^{t}\| \| \pmb{v}^{*} - \pmb{v}^{t}\|^{2}.$

Proof. First we establish the following.

Claim 29. For all $t \geq T_1$ , assertions (i)-(ii) imply assertions (iii)-(ix).

Proof of claim. Suppose $t \geq T_{1}$ , and (i) and (ii) are true.

By Lemma 21 and Proposition 27, we have (iii), (iv), and (v).

For (vi), we have

$$
\begin{array}{l} \left\| \frac {\mathrm{d} \boldsymbol {v} ^ {t}}{\mathrm{d} t} - \boldsymbol {f} ^ {t} \right\| = \left\| \sum_ {j \in J _ {+}} \frac {\mathrm{d}}{\mathrm{d} t} (\| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) - \| \boldsymbol {v} ^ {t} \| (\boldsymbol {g} ^ {t} + \overline {{\boldsymbol {v}}} ^ {t} \overline {{\boldsymbol {v}}} ^ {t ^ {\top}} \boldsymbol {g} ^ {t}) \right\| \\ = \left\| \sum_ {j \in J _ {+}} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} (\boldsymbol {g} ^ {t} + \overline {{\boldsymbol {w}}} _ {j} ^ {t} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} ^ {t}) - \| \boldsymbol {v} ^ {t} \| (\boldsymbol {g} ^ {t} + \overline {{\boldsymbol {v}}} ^ {t} \overline {{\boldsymbol {v}}} ^ {t} ^ {\top} \boldsymbol {g} ^ {t}) \right\| \\ = \left\| \sum_ {j \in J _ {+}} \left(\| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \boldsymbol {g} ^ {t} + \overline {{\boldsymbol {w}}} _ {j} ^ {t} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} ^ {t} - \overline {{\boldsymbol {v}}} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \boldsymbol {g} ^ {t} - \overline {{\boldsymbol {v}}} ^ {t} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} ^ {t}\right) \right\| \\ = \left\| \sum_ {j \in J _ {+}} (1 - \overline {{\boldsymbol {v}}} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j} ^ {t}) \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \boldsymbol {g} ^ {t} + \sum_ {j \in J _ {+}} (\overline {{\boldsymbol {w}}} _ {j} ^ {t} - \overline {{\boldsymbol {v}}} ^ {t}) \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} ^ {t} \right\| \\ <   4 \lambda^ {\varepsilon} \sum_ {j \in J _ {+}} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \| \boldsymbol {g} ^ {t} \| + \sqrt {8} \lambda^ {\varepsilon / 2} \sum_ {j \in J _ {+}} \| \boldsymbol {w} _ {j} ^ {t} \| ^ {2} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} ^ {t} \\ <   \frac {4 \lambda^ {\varepsilon}}{1 - 4 \lambda^ {\varepsilon}} \| \boldsymbol {v} ^ {t} \| \| \boldsymbol {g} ^ {t} \| + \sqrt {8} \lambda^ {\varepsilon / 2} \boldsymbol {v} ^ {t ^ {\top}} \boldsymbol {g} ^ {t} \\ <   \left(5 \lambda^ {\varepsilon} + \sqrt {8} \lambda^ {\varepsilon / 2}\right) \| \boldsymbol {f} ^ {t} \| \\ <   3 \lambda^ {\varepsilon / 2} \| \boldsymbol {f} ^ {t} \|. \\ \end{array}
$$

Now

$$
\begin{array}{l} \mathrm{d} \| \boldsymbol {v} ^ {t} \| ^ {2} / \mathrm{d} t \geq 2 (\boldsymbol {v} ^ {t ^ {\top}} \boldsymbol {f} ^ {t} - 3 \lambda^ {\varepsilon / 2} \| \boldsymbol {v} ^ {t} \| \| \boldsymbol {f} ^ {t} \|) \\ = 2 \| \boldsymbol {v} ^ {t} \| (2 \boldsymbol {v} ^ {t ^ {\top}} \boldsymbol {g} ^ {t} - 3 \lambda^ {\varepsilon / 2} \| \boldsymbol {f} ^ {t} \|), \\ \end{array}
$$

so for (vii), if $\pmb{v}^t\in S_1$ then by Proposition 26 we have

$$
\begin{array}{l} 2 \| \boldsymbol {v} ^ {t} \| (2 \boldsymbol {v} ^ {t ^ {\top}} \boldsymbol {g} ^ {t} - 3 \lambda^ {\varepsilon / 2} \| \boldsymbol {f} ^ {t} \|) > (2 (\eta_ {d} \nu_ {d} ^ {*} / \| \boldsymbol {\gamma} _ {[ n ]} \|) ^ {2} - 1 2 \lambda^ {\varepsilon / 2}) \| \boldsymbol {v} ^ {t} \| ^ {2} \| \boldsymbol {g} ^ {t} \| \\ > (\eta_ {d} \nu_ {d} ^ {*} / \| \boldsymbol {\gamma} _ {[ n ]} \|) ^ {2} \| \boldsymbol {v} ^ {t} \| ^ {2} \| \boldsymbol {g} ^ {t} \| \\ \end{array}
$$

since

$$
\lambda^ {\varepsilon / 2} \leq n ^ {- \frac {9 \cdot 3 n \Delta^ {2}}{2 \delta^ {3}}} <   \left(\frac {4 \delta^ {3}}{9 \cdot 3 n \Delta^ {2}}\right) ^ {2} <   \frac {\delta^ {6}}{1 2 d \Delta^ {4}} \leq (\eta_ {d} \nu_ {d} ^ {*} / \| \boldsymbol {\gamma} _ {[ n ]} \|) ^ {2} / 1 2,
$$

and for (viii), in general we have

$$
\begin{array}{l} 2 \| \boldsymbol {v} ^ {t} \| (2 \boldsymbol {v} ^ {t ^ {\top}} \boldsymbol {g} ^ {t} - 3 \lambda^ {\varepsilon / 2} \| \boldsymbol {f} ^ {t} \|) > (4 \lambda^ {\varepsilon / 3} - 1 2 \lambda^ {\varepsilon / 2}) \| \boldsymbol {v} ^ {t} \| ^ {2} \| \boldsymbol {g} ^ {t} \| \\ > 3 \lambda^ {\varepsilon / 3} \| \boldsymbol {v} ^ {t} \| ^ {2} \| \boldsymbol {g} ^ {t} \|. \\ \end{array}
$$

For (ix), we have

$$
\begin{array}{l} \mathrm{d} \| \boldsymbol {v} ^ {*} - \boldsymbol {v} ^ {t} \| ^ {2} / \mathrm{d} t \geq - 2 ((\boldsymbol {v} ^ {*} - \boldsymbol {v} ^ {t}) ^ {\top} \boldsymbol {f} ^ {t} + 3 \lambda^ {\varepsilon / 2} \| \boldsymbol {v} ^ {*} - \boldsymbol {v} ^ {t} \| \| \boldsymbol {f} ^ {t} \|) \\ \geq - 4 (1 + 3 \lambda^ {\varepsilon / 2}) \| \boldsymbol {v} ^ {t} \| \| \boldsymbol {g} ^ {t} \| \| \boldsymbol {v} ^ {*} - \boldsymbol {v} ^ {t} \| \\ > - 5 \| \boldsymbol {v} ^ {t} \| \| \boldsymbol {g} ^ {t} \| \| \boldsymbol {v} ^ {*} - \boldsymbol {v} ^ {t} \| \\ > - 5 \eta_ {1} \| \boldsymbol {v} ^ {t} \| \| \boldsymbol {v} ^ {*} - \boldsymbol {v} ^ {t} \| \\ \end{array}
$$

since $\| \pmb{g}^t\| = \| \frac{1}{n}\pmb {X}\pmb {X}^\top (\pmb {v}^{*} - \pmb{v}^{t})\| \leq \eta_{1}\| \pmb{v}^{*} - \pmb{v}^{t}\| <  \eta_{1}\| \pmb{v}^{*}\| = \eta_{1}$ by Proposition 25.

![](images/5fdf860895e2ee288a7ddb903fe7894c3392a1d80d41a90ac50490920f8701c6.jpg)

Second we show the following.

Claim 30. Assertions (i) and (ii) are true for $t = T_{1}$ .

Proof of claim. By Lemma 23 (iii), for all $j, j' \in J_{+}$ we have

$$
\begin{array}{l} 1 - \overline {{\boldsymbol {w}}} _ {j} ^ {T _ {1} \top} \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {T _ {1}} = \| \overline {{\boldsymbol {w}}} _ {j} ^ {T _ {1}} - \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {T _ {1}} \| ^ {2} / 2 \\ <   \| \overline {{\boldsymbol {w}}} _ {j} ^ {T _ {1}} - \overline {{\boldsymbol {\gamma}}} _ {[ n ]} \| ^ {2} + \| \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {T _ {1}} - \overline {{\boldsymbol {\gamma}}} _ {[ n ]} \| ^ {2} \\ \leq 4 \lambda^ {\varepsilon} \\ \end{array}
$$

where the first inequality is strict unless $\overline{\boldsymbol{w}}_j^{T_1} = \overline{\gamma}_{[n]} = \overline{\boldsymbol{w}}_{j'}^{T_1}$ , but in that case $1 - \overline{\boldsymbol{w}}_{j}^{T_1^\top}\overline{\boldsymbol{w}}_{j'}^{T_1} = 0$ .

Writing $\pmb{v}^{T_1} = \sum_{k=1}^{d} \nu_k \pmb{u}_k$ , and recalling Proposition 16, Lemma 19 (i), and Lemma 23 (i) and (iii), we obtain that $\pmb{v}^{T_1} \in S_1$ because:

$\Phi_{1}$ : we have

$$
\frac {\nu_ {1}}{\| \boldsymbol {v} ^ {T _ {1}} \|} \geq \frac {\eta_ {1} \nu_ {1} ^ {*}}{\| \boldsymbol {\gamma} _ {[ n ]} \|} - \sqrt {2} \lambda^ {\varepsilon / 2} \geq \frac {4 \delta^ {3}}{\sqrt {d} \Delta^ {2}} - \sqrt {2} \lambda^ {\varepsilon / 2} > 0
$$

and

$$
\frac {\nu_ {1}}{\nu_ {1} ^ {*}} \leq \frac {4 m \sqrt {d} \Delta^ {2}}{\delta} \lambda^ {2 - 2 \varepsilon} <   \frac {1}{2};
$$

$\Psi_{k,k'}^{\downarrow}$ for all $1 \leq k < k' \leq d$ : we have

$$
\begin{array}{l} \frac {\nu_ {k}}{\| \boldsymbol {v} ^ {T _ {1}} \| \eta_ {k} \nu_ {k} ^ {*}} \leq \frac {1}{\| \boldsymbol {\gamma} _ {[ n ]} \|} + \frac {\sqrt {2} \lambda^ {\varepsilon / 2}}{\eta_ {k} \nu_ {k} ^ {*}} \\ \leq \frac {1}{\| \boldsymbol {\gamma} _ {[ n ]} \|} + \frac {\sqrt {2 d} \lambda^ {\varepsilon / 2}}{\delta^ {3}} \\ <   \frac {2}{\| \boldsymbol {\gamma} _ {[ n ]} \|} - \frac {2 \sqrt {2 d} \lambda^ {\varepsilon / 2}}{\delta^ {3}} \\ \leq \frac {2}{\| \boldsymbol {\gamma} _ {[ n ]} \|} - \frac {2 \sqrt {2} \lambda^ {\varepsilon / 2}}{\eta_ {k ^ {\prime}} \nu_ {k ^ {\prime}} ^ {*}} \\ \leq \frac {2 \nu_ {k ^ {\prime}}}{\| \boldsymbol {v} ^ {T _ {1}} \| \eta_ {k ^ {\prime}} \nu_ {k ^ {\prime}} ^ {*}}; \\ \end{array}
$$

$\Psi_{k,k'}^{\uparrow}$ for all $1 \leq k < k' \leq d$ : by Bernoulli's inequality we have

$$
\begin{array}{l} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {\frac {\eta_ {k} + \eta_ {k ^ {\prime}}}{2 \eta_ {k}}} <   1 - \frac {\eta_ {k} + \eta_ {k ^ {\prime}}}{2 \eta_ {k}} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \\ \leq 1 - \| \boldsymbol {v} ^ {T _ {1}} \| \frac {\eta_ {k} + \eta_ {k ^ {\prime}}}{2} \left(\frac {1}{\| \boldsymbol {\gamma} _ {[ n ]} \|} - \frac {\sqrt {2} \lambda^ {\varepsilon / 2}}{\eta_ {k} \nu_ {k} ^ {*}}\right) \\ = 1 - \| \boldsymbol {v} ^ {T _ {1}} \| \left(\frac {\eta_ {k ^ {\prime}}}{\| \boldsymbol {\gamma} _ {[ n ]} \|} + \frac {\eta_ {k} - \eta_ {k ^ {\prime}}}{2 \| \boldsymbol {\gamma} _ {[ n ]} \|} - \frac {\eta_ {k} + \eta_ {k ^ {\prime}}}{2} \frac {\sqrt {2} \lambda^ {\varepsilon / 2}}{\eta_ {k} \nu_ {k} ^ {*}}\right) \\ \leq 1 - \| \boldsymbol {v} ^ {T _ {1}} \| \left(\frac {\eta_ {k ^ {\prime}}}{\| \boldsymbol {\gamma} _ {[ n ]} \|} + \frac {\delta^ {2}}{d \Delta^ {2}} - \Delta^ {2} \frac {\sqrt {2 d} \lambda^ {\varepsilon / 2}}{\delta^ {3}}\right) \\ \end{array}
$$

$$
\begin{array}{l} <   1 - \| \boldsymbol {v} ^ {T _ {1}} \| \left(\frac {\eta_ {k ^ {\prime}}}{\| \boldsymbol {\gamma} _ {[ n ]} \|} + \Delta^ {2} \frac {\sqrt {2 d} \lambda^ {\varepsilon / 2}}{\delta^ {3}}\right) \\ \leq 1 - \| \boldsymbol {v} ^ {T _ {1}} \| \eta_ {k ^ {\prime}} \left(\frac {1}{\| \boldsymbol {\gamma} _ {[ n ]} \|} + \frac {\sqrt {2} \lambda^ {\varepsilon / 2}}{\eta_ {k ^ {\prime}} \nu_ {k ^ {\prime}} ^ {*}}\right) \\ \leq 1 - \frac {\nu_ {k ^ {\prime}}}{\nu_ {k ^ {\prime}} ^ {*}}; \\ \end{array}
$$

$\Xi$ : we have

$$
\begin{array}{l} \overline {{\boldsymbol {v}}} ^ {T _ {1} \top} \boldsymbol {g} ^ {T _ {1}} \geq \overline {{\boldsymbol {v}}} ^ {T _ {1} \top} \boldsymbol {\gamma} _ {[ n ]} - \lambda^ {2 - 3 \varepsilon} \\ \geq (1 - \lambda^ {\varepsilon}) \| \boldsymbol {\gamma} _ {[ n ]} \| - \lambda^ {2 - 3 \varepsilon} \\ > 3 \lambda^ {\varepsilon / 3} \| \boldsymbol {\gamma} _ {[ n ]} \| - \lambda^ {2 - 3 \varepsilon} \\ > 2 \lambda^ {\varepsilon / 3} \| \boldsymbol {\gamma} _ {[ n ]} \| \\ > \lambda^ {\varepsilon / 3} (\| \boldsymbol {\gamma} _ {[ n ]} \| + \lambda^ {2 - 3 \varepsilon}) \\ \geq \lambda^ {\varepsilon / 3} \| \boldsymbol {g} ^ {T _ {1}} \|. \\ \end{array}
$$

![](images/2a24260d2caa2854d92737a6835e14cf284b89634c6d692beb6890d75d16a51d.jpg)

Assume for a contradiction that there exists $t > T_{1}$ such that either (i) or (ii) is false, and let $t$ be the smallest such.

For all $j, j' \in J_{+}$ we have

$$
\begin{array}{l} \mathrm{d} (1 - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {t}) / \mathrm{d} t = - \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {t} ^ {\top} (\boldsymbol {g} ^ {t} - \overline {{\boldsymbol {w}}} _ {j} ^ {t} \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \boldsymbol {g} ^ {t}) - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} (\boldsymbol {g} ^ {t} - \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {t} \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {t} ^ {\top} \boldsymbol {g} ^ {t}) \\ = - (1 - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {t}) (\overline {{\boldsymbol {w}}} _ {j} ^ {t} + \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {t}) ^ {\top} \boldsymbol {g} ^ {t} \\ \leq - 2 (\lambda^ {\varepsilon / 3} - \sqrt {8} \lambda^ {\varepsilon / 2}) \| \boldsymbol {g} ^ {t} \| (1 - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {t}) \\ \leq - \lambda^ {\varepsilon / 3} \| \boldsymbol {g} ^ {t} \| (1 - \overline {{\boldsymbol {w}}} _ {j} ^ {t} ^ {\top} \overline {{\boldsymbol {w}}} _ {j ^ {\prime}} ^ {t})  . \\ \end{array}
$$

Therefore (ii) is false. Hence $v^{t}$ is in at least one possibly curved face of the boundary of S, and is distinct from 0 and $v^{*}$ . We consider those faces in the following cases, where we omit the superscripts t, assume $v = \sum_{k=1}^{d} \nu_{k} u_{k} \in \mathrm{cl}(S_{\ell})$ for some $\ell \in [d]$ , write e.g. $\widehat{\Omega}_{k}$ for the constraint obtained by replacing the unique strict inequality in $\Omega_{k}$ by equality, and denote by p a normal vector to the respective face that is at v and on the side of the interior of S. To get a contradiction, it suffices to show in each of the cases that $\overline{p}^{\top} \overline{f} > 3\lambda^{\varepsilon/2}$ , because by Claim 29 and continuity we have that (vi) is true at t, and so $\overline{p}^{\top}(\mathrm{d}\boldsymbol{v}/\mathrm{d}t) \geq \overline{p}^{\top} f - 3\lambda^{\varepsilon/2} \|f\| > 0$ .

Case $\widehat{\Omega}_k$ for some $1\leq k < \ell$ . Picking $\pmb {p}:= \pmb{u}_{k}$ , we have

$$
\overline {{\boldsymbol {p}}} ^ {\top} \overline {{\boldsymbol {f}}} = \nu_ {k} ^ {*} \overline {{\boldsymbol {v}}} ^ {\top} \boldsymbol {g} / \| \boldsymbol {f} \| > \frac {\nu_ {k} ^ {*}}{2} \overline {{\boldsymbol {v}}} ^ {\top} \overline {{\boldsymbol {g}}} \geq \frac {\delta}{2 \sqrt {d}} \lambda^ {\varepsilon / 3} > 3 \lambda^ {\varepsilon / 2}
$$

since $\lambda^{-\varepsilon /6}\geq n^{\frac{9}{2} n / \delta^3}\geq \frac{9}{\sqrt{2}}\mathrm{e}(\ln 2)\sqrt{d} /\delta^3 >\frac{6\sqrt{d}}{\delta}.$

Case $\widehat{\Phi}_{\ell}$ . Necessarily $\ell \neq 1$ . Picking $p := u_{\ell}$ , we have

$$
\begin{array}{l} \overline {{\boldsymbol {p}}} ^ {\top} \overline {{\boldsymbol {f}}} \geq \left(\eta_ {\ell} \left(1 - \frac {\eta_ {\ell}}{2 \eta_ {\ell - 1}}\right) \nu_ {\ell} ^ {*} + \frac {1}{\| \boldsymbol {v} \|} \frac {\eta_ {\ell}}{2 \eta_ {\ell - 1}} \nu_ {\ell} ^ {*} \overline {{\boldsymbol {v}}} ^ {\top} \boldsymbol {g}\right) / (2 \| \boldsymbol {g} \|) \\ > \frac {\eta_ {\ell}}{2 \eta_ {1}} \bigg (1 - \frac {\eta_ {\ell}}{2 \eta_ {\ell - 1}} \bigg) \nu_ {\ell} ^ {*} > \frac {\delta^ {3}}{4 \sqrt {d} \Delta^ {2}} > 3 \lambda^ {\varepsilon / 2}. \\ \end{array}
$$

Case $\widehat{\Psi}_{k,k^{\prime}}^{\downarrow}$ for some $\ell \leq k < k^{\prime} \leq d$ . Picking $p := -\eta_{k^{\prime}} \nu_{k^{\prime}}^{*} u_{k} + 2 \eta_{k} \nu_{k}^{*} u_{k^{\prime}}$ , we have

$$
\begin{array}{l} \overline {{\boldsymbol {p}}} ^ {\top} \overline {{\boldsymbol {f}}} = \eta_ {k} \eta_ {k ^ {\prime}} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} \left(- \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) + 2 \left(1 - \frac {\nu_ {k ^ {\prime}}}{\nu_ {k ^ {\prime}} ^ {*}}\right)\right) \| \boldsymbol {v} \| / (\| \boldsymbol {p} \| \| \boldsymbol {f} \|) \\ = \eta_ {k} \eta_ {k ^ {\prime}} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} \left(1 + \left(1 - \frac {\eta_ {k ^ {\prime}}}{\eta_ {k}}\right) \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) \| \boldsymbol {v} \| / (\| \boldsymbol {p} \| \| \boldsymbol {f} \|) \\ > \eta_ {k} \eta_ {k ^ {\prime}} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} / (2 \| \boldsymbol {p} \| \| \boldsymbol {g} \|) \\ > \frac {\eta_ {k} \eta_ {k ^ {\prime}} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*}}{2 \eta_ {1} \sqrt {(2 \eta_ {k} \nu_ {k} ^ {*}) ^ {2} + (\eta_ {k ^ {\prime}} \nu_ {k ^ {\prime}} ^ {*}) ^ {2}}} \\ = \left(\left(\frac {2 \eta_ {1}}{\eta_ {k}} \frac {1}{\nu_ {k} ^ {*}}\right) ^ {2} + \left(\frac {2 \eta_ {1}}{\eta_ {k ^ {\prime}}} \frac {2}{\nu_ {k ^ {\prime}} ^ {*}}\right) ^ {2}\right) ^ {- 1 / 2} \\ \geq \frac {\delta^ {3}}{5 \sqrt {2 d} \Delta^ {2}} \\ > 3 \lambda^ {\varepsilon / 2}. \\ \end{array}
$$

Case $\widehat{\Psi}_{k,k'}^{\uparrow}$ for some $\ell \leq k < k' \leq d$ . Picking

$$
\boldsymbol {p} := \frac {\eta_ {k} + \eta_ {k ^ {\prime}}}{2} \nu_ {k ^ {\prime}} ^ {*} \boldsymbol {u} _ {k} - \eta_ {k} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {\frac {1}{2} - \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}} \nu_ {k} ^ {*} \boldsymbol {u} _ {k ^ {\prime}},
$$

we have

$$
\begin{array}{l} \boldsymbol {p} ^ {\top} \boldsymbol {g} = \eta_ {k} \eta_ {k ^ {\prime}} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} \left(\left(\frac {1}{2} + \frac {\eta_ {k}}{2 \eta_ {k ^ {\prime}}}\right) \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {\frac {1}{2} - \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}} \left(1 - \frac {\nu_ {k ^ {\prime}}}{\nu_ {k ^ {\prime}} ^ {*}}\right)\right) \\ = \eta_ {k} \eta_ {k ^ {\prime}} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} \left(\left(\frac {1}{2} + \frac {\eta_ {k}}{2 \eta_ {k ^ {\prime}}}\right) \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right)\right) \\ = \frac {\eta_ {k} ^ {2} - \eta_ {k} \eta_ {k ^ {\prime}}}{2} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} \\ \end{array}
$$

and

$$
\begin{array}{l} \boldsymbol {p} ^ {\top} \boldsymbol {v} = \frac {\eta_ {k} + \eta_ {k ^ {\prime}}}{2} \nu_ {k} \nu_ {k ^ {\prime}} ^ {*} - \eta_ {k} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {\frac {1}{2} - \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}} \nu_ {k ^ {\prime}} \nu_ {k} ^ {*} \\ = \eta_ {k} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} \left(\left(\frac {1}{2} + \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}\right) \frac {\nu_ {k}}{\nu_ {k} ^ {*}} - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {\frac {1}{2} - \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}} \left(1 - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {\frac {1}{2} + \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}}\right)\right) \\ = \eta_ {k} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} \left(\left(1 - \left(\frac {1}{2} - \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}\right) \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {\frac {1}{2} - \frac {\eta_ {k ^ {\prime}}}{2 \eta_ {k}}}\right) \\ > \frac {\eta_ {k} ^ {2} - \eta_ {k ^ {\prime}} ^ {2}}{8 \eta_ {k}} \left(\frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {2} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} . \\ \end{array}
$$

Hence if $\nu_{k} / \nu_{k}^{*}\leq 1 / 2$ then

$$
\begin{array}{l} \overline {{\boldsymbol {p}}} ^ {\top} \overline {{\boldsymbol {f}}} > \frac {\eta_ {k} ^ {2} - \eta_ {k} \eta_ {k ^ {\prime}}}{4} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} / (2 \| \boldsymbol {p} \| \| \boldsymbol {g} \|) \\ > \frac {(\eta_ {k} ^ {2} - \eta_ {k} \eta_ {k ^ {\prime}}) \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*}}{\eta_ {1} \sqrt {(\eta_ {k} + \eta_ {k ^ {\prime}}) ^ {2} \nu_ {k ^ {\prime}} ^ {* 2} + 4 \eta_ {k} ^ {2} \nu_ {k} ^ {* 2}}} \\ \end{array}
$$

$$
\geq \frac {\delta^ {6}}{\sqrt {2} d ^ {2} \Delta^ {4}}
$$

$$
> 3 \lambda^ {\varepsilon / 2},
$$

else

$$
\overline {{\boldsymbol {p}}} ^ {\top} \overline {{\boldsymbol {f}}} > \frac {\eta_ {k} ^ {2} - \eta_ {k ^ {\prime}} ^ {2}}{3 2 \eta_ {k}} \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*} \overline {{\boldsymbol {v}}} ^ {\top} \boldsymbol {g} / (\| \boldsymbol {p} \| \| \boldsymbol {f} \|)
$$

$$
> \frac {(\eta_ {k} ^ {2} - \eta_ {k ^ {\prime}} ^ {2}) \nu_ {k} ^ {*} \nu_ {k ^ {\prime}} ^ {*}}{3 2 \eta_ {k} \sqrt {(\eta_ {k} + \eta_ {k ^ {\prime}}) ^ {2} \nu_ {k ^ {\prime}} ^ {* 2} + 4 \eta_ {k} ^ {2} \nu_ {k} ^ {* 2}}} \overline {{\boldsymbol {v}}} ^ {\top} \overline {{\boldsymbol {g}}}
$$

$$
\geq \frac {\delta^ {6}}{1 6 \sqrt {2} d ^ {2} \Delta^ {4}} \lambda^ {\varepsilon / 3}
$$

$$
> 3 \lambda^ {\varepsilon / 2}
$$

since $\lambda^{-\varepsilon /6}\geq n^{\frac{9}{2} n\Delta^{2} / \delta^{3}}\geq \left(2^{\frac{9}{4} n\Delta^{2} / \delta^{3}}\right)^{2} > \left(\frac{11n\Delta^{2}}{\delta^{3}}\right)^{2} > \frac{48\sqrt{2}d^{2}\Delta^{4}}{\delta^{6}}.$

Case $\widehat{\Xi}$ . Here $\overline{v}^{\top}\overline{g}=\lambda^{\varepsilon/3}$ .

Hence, by Proposition 26, since $\lambda^{\varepsilon /3} < \frac{1}{2}\frac{\delta^6}{d\Delta^4}\leq \frac{1}{2}\left(\frac{\eta_d\nu_d^*}{\|\gamma_{[n]}\|}\right)^2$ , necessarily $\ell \neq 1$ .

Picking

$$
\boldsymbol {p} := \sum_ {k = 1} ^ {d} \eta_ {k} \nu_ {k} ^ {*} \left(1 - \frac {2 \nu_ {k}}{\nu_ {k} ^ {*}} - \lambda^ {\varepsilon / 3} \left(\frac {\nu_ {k}}{\nu_ {k} ^ {*}} \frac {\| \boldsymbol {g} \|}{\eta_ {k} \| \boldsymbol {v} \|} - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) \frac {\eta_ {k} \| \boldsymbol {v} \|}{\| \boldsymbol {g} \|}\right)\right) \boldsymbol {u} _ {k},
$$

we have

$$
\boldsymbol {p} ^ {\top} \boldsymbol {g} = \sum_ {k = 1} ^ {d} \eta_ {k} ^ {2} \nu_ {k} ^ {* 2} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) \left(1 - \frac {2 \nu_ {k}}{\nu_ {k} ^ {*}} - \lambda^ {\varepsilon / 3} \left(\frac {\nu_ {k}}{\nu_ {k} ^ {*}} \frac {\| \boldsymbol {g} \|}{\eta_ {k} \| \boldsymbol {v} \|} - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) \frac {\eta_ {k} \| \boldsymbol {v} \|}{\| \boldsymbol {g} \|}\right)\right)
$$

$$
= \sum_ {k = 1} ^ {d} \eta_ {k} ^ {2} \nu_ {k} ^ {* 2} \bigg (1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \bigg) \bigg (1 - \frac {2 \nu_ {k}}{\nu_ {k} ^ {*}} \bigg) + \lambda^ {\varepsilon / 3} \sum_ {k = 1} ^ {d} \eta_ {k} ^ {3} \nu_ {k} ^ {* 2} \bigg (1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \bigg) ^ {2} \frac {\| \boldsymbol {v} \|}{\| \boldsymbol {g} \|} - \lambda^ {2 \varepsilon / 3} \| \boldsymbol {g} \| ^ {2}
$$

$$
= \| \boldsymbol {g} \| ^ {2} + \sum_ {k = 1} ^ {\ell - 1} \eta_ {k} ^ {2} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(\frac {\nu_ {k}}{\nu_ {k} ^ {*}} - 1\right) - \sum_ {k = \ell} ^ {d} \eta_ {k} ^ {2} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right)
$$

$$
+ \lambda^ {\varepsilon / 3} \sum_ {k = 1} ^ {d} \eta_ {k} ^ {3} \nu_ {k} ^ {* 2} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) ^ {2} \frac {\| \boldsymbol {v} \|}{\| \boldsymbol {g} \|} - \lambda^ {2 \varepsilon / 3} \| \boldsymbol {g} \| ^ {2}
$$

$$
\geq \| \boldsymbol {g} \| ^ {2} + \eta_ {\ell - 1} \sum_ {k = 1} ^ {\ell - 1} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(\frac {\nu_ {k}}{\nu_ {k} ^ {*}} - 1\right) - \eta_ {\ell} \sum_ {k = \ell} ^ {d} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right)
$$

$$
+ \lambda^ {\varepsilon / 3} \eta_ {d} \| \boldsymbol {v} \| \| \boldsymbol {g} \| - \lambda^ {2 \varepsilon / 3} \| \boldsymbol {g} \| ^ {2}
$$

$$
= \| \boldsymbol {g} \| ^ {2} + \frac {\eta_ {\ell - 1} - \eta_ {\ell}}{2} \left(\sum_ {k = 1} ^ {\ell - 1} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(\frac {\nu_ {k}}{\nu_ {k} ^ {*}} - 1\right) + \sum_ {k = \ell} ^ {d} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right)\right)
$$

$$
- \lambda^ {\varepsilon / 3} \left(\frac {\eta_ {\ell - 1} + \eta_ {\ell}}{2} - \eta_ {d}\right) \| \boldsymbol {v} \| \| \boldsymbol {g} \| - \lambda^ {2 \varepsilon / 3} \| \boldsymbol {g} \| ^ {2}
$$

$$
\geq \left(\frac {1}{8} \left(1 - \frac {\eta_ {\ell}}{\eta_ {\ell - 1}}\right) \eta_ {d} \left(\min _ {k = 1} ^ {\ell - 1} \nu_ {k} ^ {*}\right) + \| \boldsymbol {g} \| - \lambda^ {\varepsilon / 3} \left(\frac {\eta_ {\ell - 1} + \eta_ {\ell}}{2} - \eta_ {d}\right) \| \boldsymbol {v} \| - \lambda^ {2 \varepsilon / 3} \| \boldsymbol {g} \|\right) \| \boldsymbol {g} \|
$$

and

$$
\begin{array}{l} \boldsymbol {p} ^ {\top} \boldsymbol {v} = \sum_ {k = 1} ^ {d} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(1 - \frac {2 \nu_ {k}}{\nu_ {k} ^ {*}} - \lambda^ {\varepsilon / 3} \left(\frac {\nu_ {k}}{\nu_ {k} ^ {*}} \frac {\| \boldsymbol {g} \|}{\eta_ {k} \| \boldsymbol {v} \|} - \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) \frac {\eta_ {k} \| \boldsymbol {v} \|}{\| \boldsymbol {g} \|}\right)\right) \\ > - \eta_ {1} \| \boldsymbol {v} \| ^ {2} + \lambda^ {\varepsilon / 3} \frac {\| \boldsymbol {v} \|}{\| \boldsymbol {g} \|} \sum_ {k = 1} ^ {d} \eta_ {k} ^ {2} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) \\ \geq - \eta_ {1} \| \boldsymbol {v} \| ^ {2} - \frac {\lambda^ {\varepsilon / 3}}{n} \frac {\| \boldsymbol {v} \|}{\| \boldsymbol {g} \|} \left(\eta_ {1} \sum_ {k = 1} ^ {\ell - 1} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(\frac {\nu_ {k}}{\nu_ {k} ^ {*}} - 1\right) - \eta_ {d} \sum_ {k = \ell} ^ {d} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right)\right) \\ = - \eta_ {1} \| \boldsymbol {v} \| ^ {2} (1 - \lambda^ {2 \varepsilon / 3}) - \lambda^ {\varepsilon / 3} \frac {\| \boldsymbol {v} \|}{\| \boldsymbol {g} \|} (\eta_ {1} - \eta_ {d}) \sum_ {k = \ell} ^ {d} \eta_ {k} \nu_ {k} ^ {* 2} \frac {\nu_ {k}}{\nu_ {k} ^ {*}} \left(1 - \frac {\nu_ {k}}{\nu_ {k} ^ {*}}\right) \\ \geq - \eta_ {1} \left(1 + \lambda^ {\varepsilon / 3} \left(1 - \frac {\eta_ {d}}{\eta_ {1}}\right) - \lambda^ {2 \varepsilon / 3}\right) \| \boldsymbol {v} \| ^ {2}, \\ \end{array}
$$

also

$$
\begin{array}{l} \| \boldsymbol {p} \| <   2 \sqrt {\sum_ {k = 1} ^ {d} \left((\eta_ {1} \nu_ {k}) ^ {2} + (\eta_ {k} (\nu_ {k} ^ {*} - \nu_ {k})) ^ {2} + \left(\lambda^ {\varepsilon / 3} \nu_ {k} \frac {\| \boldsymbol {g} \|}{\| \boldsymbol {v} \|}\right) ^ {2} + \left(\lambda^ {\varepsilon / 3} \eta_ {1} \eta_ {k} (\nu_ {k} ^ {*} - \nu_ {k}) \frac {\| \boldsymbol {v} \|}{\| \boldsymbol {g} \|}\right) ^ {2}\right)} \\ = 2 \sqrt {1 + \lambda^ {2 \varepsilon / 3}} \sqrt {\eta_ {1} ^ {2} \| \boldsymbol {v} \| ^ {2} + \| \boldsymbol {g} \| ^ {2}} \\ <   2 \left(1 + \lambda^ {\varepsilon / 3}\right) \left(\eta_ {1} \| \boldsymbol {v} \| + \| \boldsymbol {g} \|\right) \\ <   4 (1 + \lambda^ {\varepsilon / 3}) \eta_ {1} \\ \end{array}
$$

and

$$
\| \boldsymbol {f} \| = (1 + \lambda^ {\varepsilon / 3}) \| \boldsymbol {v} \| \| \boldsymbol {g} \|.
$$

Therefore

$$
\begin{array}{l} \overline {{\boldsymbol {p}}} ^ {\top} \overline {{\boldsymbol {f}}} > \left[ \frac {1}{8} \left(1 - \frac {\eta_ {\ell}}{\eta_ {\ell - 1}}\right) \eta_ {d} \left(\min _ {k = 1} ^ {\ell - 1} \nu_ {k} ^ {*}\right) \right. \\ - \lambda^ {\varepsilon / 3} \left(\frac {\eta_ {\ell - 1} + \eta_ {\ell}}{2} + \eta_ {1} - \eta_ {d}\right) \| \boldsymbol {v} \| \\ \left. - \lambda^ {2 \varepsilon / 3} ((\eta_ {1} - \eta_ {d}) \| \boldsymbol {v} \| + \| \boldsymbol {g} \|) \right] \\ \end{array}
$$

$$
\Big / 4 (1 + \lambda^ {\varepsilon / 3}) ^ {2} \eta_ {1}
$$

$$
> \frac {1}{4 0} \left(1 - \frac {\eta_ {\ell}}{\eta_ {\ell - 1}}\right) \frac {\eta_ {d}}{\eta_ {1}} \binom{\ell - 1}{\min _ {k = 1} \nu_ {k} ^ {*}}
$$

$$
- \frac {\lambda^ {\varepsilon / 3}}{5} \left(\frac {\eta_ {\ell - 1} + \eta_ {\ell}}{2 \eta_ {1}} + 1 - \frac {\eta_ {d}}{\eta_ {1}}\right) \| \boldsymbol {v} \|
$$

$$
- \frac {\lambda^ {2 \varepsilon / 3}}{5} \left(\left(1 - \frac {\eta_ {d}}{\eta_ {1}}\right) \| \boldsymbol {v} \| + \frac {\| \boldsymbol {g} \|}{\eta_ {1}}\right)
$$

$$
> \frac {\delta^ {4}}{4 0 d \sqrt {d} \Delta^ {3}} - \frac {2}{5} \lambda^ {\varepsilon / 3} - \frac {2}{5} \lambda^ {2 \varepsilon / 3}
$$

$$
> \frac {1}{4 0} \left(\frac {\delta^ {4}}{d \sqrt {d} \Delta^ {3}} - 1 7 \lambda^ {\varepsilon / 3}\right)
$$

calculation

since $\lambda^{\varepsilon /3}\leq n^{-9n}\leq 2^{-9\cdot 2} <   \sqrt{5 / 4} -1$

by the definitions of $\delta$ and $\Delta$ ,

Proposition 17 (ii), and Proposition 25

since $\lambda^{\varepsilon /3}\leq n^{-9n}\leq 2^{-9\cdot 2} <   1 / 16$

![](images/980fd539e05282d45a8bf9ef5184cccf7bb304eb7569bed24e29a9b01de83cc1.jpg)

<details>
<summary>line</summary>

| iteration | arsinh((ν*_k - ν_k)/10^-5) |
| --------- | -------------------------- |
| 10^0      | ~10                        |
| 10^1      | ~10                        |
| 10^2      | ~10                        |
| 10^3      | ~-5                        |
| 10^4      | ~0                         |
| 10^5      | ~0                         |
| 10^6      | ~0                         |
</details>

Figure 3: The coordinates in the eigenvectors basis of the difference between the teacher neuron and the weighted sum of the hidden neurons crossing zero in the decreasing eigenvalue order. The horizontal axis is logarithmic. The vertical axis shows the values mapped using the inverse of the hyperbolic sine in order to be able to visualise numbers at different scales on both sides around zero. The colours are picked from a colourmap based on the corresponding eigenvalue.

$$
\begin{array}{l} > \frac {\delta^ {4}}{8 0 d \sqrt {d} \Delta^ {3}} \\ \text { since } \lambda^ {- \varepsilon / 3} \geq n ^ {9 n \Delta^ {2} / \delta^ {3}} \geq \left(2 ^ {6 n \Delta^ {2} / \delta^ {3}}\right) ^ {3 / 2} \\ \geq (2 ^ {1 1} n \Delta^ {2} / \delta^ {3}) ^ {3 / 2} > \frac {1 7 \cdot 8 0 d \sqrt {d} \Delta^ {3}}{\delta^ {4}} \\ > 3 \lambda^ {\varepsilon / 2} \\ \text { since } \lambda^ {- \varepsilon / 2} \geq n ^ {\frac {9 \cdot 3}{2} n \Delta^ {2} / \delta^ {3}} \geq \left(2 ^ {9 n \Delta^ {2} / \delta^ {3}}\right) ^ {3 / 2} \\ \geq (2 ^ {1 7} n \Delta^ {2} / \delta^ {3}) ^ {3 / 2} > \frac {3 \cdot 8 0 d \sqrt {d} \Delta^ {3}}{\delta^ {4}}, \\ \end{array}
$$

completing the proof.

![](images/9c99da05984134831e4f3e60e81e5819a52a3d90261129701b5ebb5a6c9d3594.jpg)

Example 31. To illustrate some aspects of the training dynamics, let us consider a single run of gradient descent with learning rate 0.01, for a network of width m = 25 initialised using $z_{j} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mathbf{0}, \frac{1}{dm} \mathbf{I}_{d})$ and $s_{j} \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}\{\pm 1\}$ with scale $\lambda = 4^{-7}$ , and on a synthetic uncentred training dataset in dimension d = 16 as described in section 8.

Figure 3 shows the coordinates of the vector $v^{*} - v^{t}$ in the eigenvectors basis crossing zero one by one exactly in the order of their indices, i.e. in the decreasing order of the corresponding eigenvalues of the matrix $\frac{1}{n} X X^{\top}$ . Thus the bundle vector $v^{t}$ travels through the subsets $S_{1}, S_{2}, \ldots$ of the set S exactly in their order, and in line with what we established in the proof of Lemma 28, passing through each $S_{k}$ at most once.

Let us write $v^{t}=\sum_{k=1}^{d}\nu_{k}^{t}u_{k}$ .

Recall from section 5 that $T_{2}=\inf\{t\geq T_{1}\mid\nu_{1}^{t}/\nu_{1}^{*}\geq1/2\}$ .

Building on the preceding results, the final lemma in this section establishes that the loss converges to zero at an exponential rate. To show it, we partition the second phase of the training into two, namely before and after the time $T_{2}$ which is when the coordinate of the bundle vector $v^{t}$ with respect to the largest-eigenvalue eigenvector of the matrix $\frac{1}{n}X X^{\top}$ crosses the half-way threshold to the corresponding coordinate of the teacher neuron $v^{*}$ . The period before $T_{2}$ (and after the start $T_{1}$ of the second phase) consists of an exponentially fast departure of $v^{t}$ from near the saddle at the origin, whereas the period after $T_{2}$ obeys a Polyak-Łojasiewicz inequality which implies exponentially fast convergence.

Lemma 32. (i) $\| \pmb{v}^t\| < \frac{1}{2}$ for all $t\in [T_1,T_2]$ .

(ii) $T_{2} - T_{1} < \ln \left(\frac{1}{\lambda}\right)\frac{(4 + \varepsilon / 2)d\Delta^{2}}{\delta^{6}}$ and $T_{2} < \ln \left(\frac{1}{\lambda}\right)\frac{(4 + \varepsilon)d\Delta^{2}}{\delta^{6}}$ .

(iii) $\| \pmb{v}^t\| >\frac{\|\pmb{\gamma}_{[n]}\|}{4\eta_1}$ for all $t\geq T_2$

(iv) $\| \nabla L(\pmb{\theta}^t)\|^2 >\frac{2\eta_d\|\pmb{\gamma}_{[n]}\|}{5\eta_1} L(\pmb{\theta}^t)$ for all $t\geq T_2$

(v) $L(\boldsymbol{\theta}^{t}) < \frac{\Delta^{2}}{2} \exp\left(-(t - T_{2})\frac{2\delta^{4}}{5\Delta^{2}}\right)$ for all $t \geq T_{2}$ .

Proof. By Lemma 19 (ii), we have $\| \boldsymbol{v}^{T_1} \| > |J_+|(1 - 4\lambda^\varepsilon)\lambda \min_{j \in J_+} \| \boldsymbol{z}_j \|$ .

For all $t \in [T_{1}, T_{2}]$ , by Lemma 28 (ii) we have $\|v^{t}\| < \frac{\|v^{*}\|}{2}$ , and by Lemma 28 (vii) we have $\frac{d}{dt}\|v^{t}\|^{2} > \frac{\eta_{d}^{2}\nu_{d}^{*2}}{2\|\gamma_{[n]}\|}\|v^{t}\|^{2}$ .

Hence

$$
\begin{array}{l} T _ {2} - T _ {1} <   \left(\ln \left(\frac {1}{\lambda}\right) + \ln \left(\frac {\| \boldsymbol {v} ^ {*} \|}{\min _ {j \in J _ {+}} \| \boldsymbol {z} _ {j} \|}\right)\right) \frac {4 \| \boldsymbol {\gamma} _ {[ n ]} \|}{\eta_ {d} ^ {2} \nu_ {d} ^ {* ^ {2}}} \\ \leq \left(\ln \left(\frac {1}{\lambda}\right) + \ln \left(\frac {1}{\delta}\right)\right) \frac {4 d \Delta^ {2}}{\delta^ {6}} \\ <   \ln \left(\frac {1}{\lambda}\right) \frac {(4 + \varepsilon / 2) d \Delta^ {2}}{\delta^ {6}} \\ \end{array}
$$

since $\ln (1 / \lambda)\varepsilon /2\geq \frac{9.3}{2} n(\ln n) / \delta^3\geq 9\cdot 3(\ln 2) / \delta^3 >4\ln (1 / \delta)$ , and so

$$
T _ {2} <   \ln \left(\frac {1}{\lambda}\right) \frac {\varepsilon}{\| \pmb {\gamma} _ {[ n ]} \|} + \ln \left(\frac {1}{\lambda}\right) \frac {(4 + \varepsilon / 2) d \Delta^ {2}}{\delta^ {6}} \leq \ln \left(\frac {1}{\lambda}\right) \frac {(4 + \varepsilon) d \Delta^ {2}}{\delta^ {6}}.
$$

By Lemma 28 (ii) we have $L(\boldsymbol{\theta}^{T_2}) < \frac{1}{2}\left(1 - \frac{\eta_d}{4\eta_1}\right)^2 \| \boldsymbol{v}^* \| \| \boldsymbol{\gamma}_{[n]} \| < \frac{\|\boldsymbol{\gamma}_{[n]}\|}{2}$ .

For all $t \geq T_2$ , by Lemma 28 (ii) we have $\| \boldsymbol{v}^t \| > \frac{\|\boldsymbol{\gamma}_{[n]}\|}{4\eta_1}$ , and so

$$
\begin{array}{l} \left\| \nabla L (\boldsymbol {\theta} ^ {t}) \right\| ^ {2} = - \mathrm{d} L (\boldsymbol {\theta} ^ {t}) / \mathrm{d} t \\ = \boldsymbol {g} ^ {t ^ {\top}} \mathrm{d} \boldsymbol {v} ^ {t} / \mathrm{d} t \\ \geq \boldsymbol {g} ^ {t ^ {\top}} \boldsymbol {f} ^ {t} - 3 \lambda^ {\varepsilon / 2} \| \boldsymbol {g} ^ {t} \| \| \boldsymbol {f} ^ {t} \|) \\ > \left((1 + \lambda^ {2 \varepsilon / 3}) \frac {\| \boldsymbol {\gamma} _ {[ n ]} \|}{4 \eta_ {1}} - 6 \lambda^ {\varepsilon / 2} \| \boldsymbol {v} ^ {*} \|\right) \| \boldsymbol {g} ^ {t} \| ^ {2} \\ > \frac {\| \pmb {\gamma} _ {[ n ]} \|}{5 \eta_ {1}} \| \pmb {g} ^ {t} \| ^ {2} \\ \geq \frac {\eta_ {d}}{5 \eta_ {1}} \| \boldsymbol {\gamma} _ {[ n ]} \| \| \boldsymbol {v} ^ {*} - \boldsymbol {v} ^ {t} \| \| \boldsymbol {g} ^ {t} \| \\ \geq \frac {2 \eta_ {d}}{5 \eta_ {1}} \| \boldsymbol {\gamma} _ {[ n ]} \| L (\boldsymbol {\theta} ^ {t}) \\ \end{array}
$$

since $\lambda^{\varepsilon /2}\leq n^{-\frac{9\cdot 3}{2} n\Delta^{2} / \delta^{3}} <   \frac{\delta^{2}}{120\Delta^{2}}\leq \frac{\|\pmb{\gamma}_{[n]}\|}{120\eta_{1}}.$

Hence for all $t \geq T_2$ we have

$$
L (\pmb {\theta} ^ {t}) <   \frac {\| \pmb {\gamma} _ {[ n ]} \|}{2} \exp \left(- (t - T _ {2}) \frac {2 \eta_ {d} \| \pmb {\gamma} _ {[ n ]} \|}{5 \eta_ {1}}\right) \leq \frac {\Delta^ {2}}{2} \exp \left(- (t - T _ {2}) \frac {2 \delta^ {4}}{5 \Delta^ {2}}\right).
$$

![](images/727259ba2223b0edc61e26707fa3104edff0a80ea67ea0023781fd7c32b56ca7.jpg)

<details>
<summary>line</summary>

| iteration | norm       | distance to v* | angle to γ[n] | angle to v* |
| --------- | ---------- | -------------- | ------------- | ----------- |
| 1         | 1e-10      | 1e-4           | 1e-2          | 1e-2        |
| 10        | 1e-10      | 1e-4           | 1e-2          | 1e-2        |
| 100       | 1e-8       | 1e-4           | 1e-2          | 1e-2        |
| 1000      | 1e-4       | 1e-4           | 1e-2          | 1e-2        |
| 10000     | 1e-3       | 1e-4           | 1e-2          | 1e-2        |
| 100000    | 1e-3       | 1e-4           | 1e-2          | 1e-2        |
| 1000000   | 1e-3       | 1e-4           | 1e-2          | 1e-2        |
</details>

Figure 4: The evolution of several measures of the weighted sum of the hidden neurons during the training. Both axes are logarithmic, and the angles are in degrees.

Example 33. Using the single run from Example 31 again, we illustrate in Figure 4 the progression of several significant measures of the sum $\pmb{v}^t$ of the hidden neurons multiplied by their last-layer weights during the training. In particular, we can see that the alignment with the vector $\gamma_{[n]}$ reaches its maximum around iteration 500, after which the distance and the angle to the teacher neuron $\pmb{v}^*$ starts to decrease.

# F Proofs for the implicit bias

In this section, we include an explicit subscript $\lambda$ for quantities that depend on the initialisation scale.

A key part of showing that the networks to which the training converges as time tends to infinity themselves have a limit in parameter space as the initialisation scale tends to zero is to establish the existence of that double limit for every ratio between the Euclidean norms of two hidden neurons. The following lemma does that, and provides two alternative expressions for each such double-limit ratio: the limit of the same ratio at time $T_{1}$ as $\lambda$ tends to zero, and the corresponding limit for the yardstick trajectories as t tends to infinity.

Lemma 34. For all $j, j' \in J_{+}$ we have

$$
\lim _ {\lambda \rightarrow 0 ^ {+}} \lim _ {t \rightarrow \infty} \frac {\| \boldsymbol {w} _ {\lambda , j} ^ {t} \|}{\| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {t} \|} = \lim _ {\lambda \rightarrow 0 ^ {+}} \frac {\| \boldsymbol {w} _ {\lambda , j} ^ {T _ {\lambda , 1}} \|}{\| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {T _ {\lambda , 1}} \|} = \lim _ {t \rightarrow \infty} \frac {\| \boldsymbol {\alpha} _ {j} ^ {t} \|}{\| \boldsymbol {\alpha} _ {j ^ {\prime}} ^ {t} \|}.
$$

Proof. Suppose $j, j' \in J_+$ .

Recalling Proposition 22 and letting $u_{j} := \operatorname{artanh}\cos \varphi_{j}^{T_{0}}$ , for all $t \geq T_{0}$ we have

$$
\begin{array}{l} \left| \mathrm{d} \ln \frac {\| \boldsymbol {\alpha} _ {j} ^ {t} \|}{\| \boldsymbol {\alpha} _ {j ^ {\prime}} ^ {t} \|} / \mathrm{d} t \right| = | (\overline {{\boldsymbol {\alpha}}} _ {j} ^ {t} - \overline {{\boldsymbol {\alpha}}} _ {j ^ {\prime}} ^ {t}) ^ {\top} \boldsymbol {\gamma} _ {[ n ]} | \\ = \left| \tanh (u _ {j} + \| \boldsymbol {\gamma} _ {[ n ]} \| (t - T _ {0})) - \tanh (u _ {j ^ {\prime}} + \| \boldsymbol {\gamma} _ {[ n ]} \| (t - T _ {0})) \right| \| \boldsymbol {\gamma} _ {[ n ]} \| \\ = | \tanh (u _ {j} - u _ {j ^ {\prime}}) | \| \boldsymbol {\gamma} _ {[ n ]} \| \\ \left(1 - \tanh (u _ {j} + \| \boldsymbol {\gamma} _ {[ n ]} \| (t - T _ {0})) \tanh (u _ {j ^ {\prime}} + \| \boldsymbol {\gamma} _ {[ n ]} \| (t - T _ {0}))\right) \\ <   | \mathrm{tanh} (u _ {j} - u _ {j ^ {\prime}}) | \| \boldsymbol {\gamma} _ {[ n ]} \| \big (1 - \mathrm{tanh} ^ {2} (\| \boldsymbol {\gamma} _ {[ n ]} \| (t - T _ {0})) \big) \\ <   \left. | \tanh (u _ {j} - u _ {j ^ {\prime}}) | \| \boldsymbol {\gamma} _ {[ n ]} \| \left(1 - \left(1 - 2 / \exp (2 \| \boldsymbol {\gamma} _ {[ n ]} \| (t - T _ {0}))\right) ^ {2}\right) \right. \\ \end{array}
$$

$$
<   4 | \tanh (u _ {j} - u _ {j ^ {\prime}}) | \| \boldsymbol {\gamma} _ {[ n ]} \| \exp (- 2 \| \boldsymbol {\gamma} _ {[ n ]} \| (t - T _ {0})) ,
$$

so $\lim_{t\to\infty}\frac{\|\boldsymbol{\alpha}_{j}^{t}\|}{\|\boldsymbol{\alpha}_{j'}^{t}\|}$ exists.

By Lemma 23 (iv), we have

$$
\begin{array}{l} \left| \ln \frac {\| \boldsymbol {w} _ {\lambda , j} ^ {T _ {\lambda , 1}} \|}{\| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {T _ {\lambda , 1}} \|} - \ln \frac {\| \boldsymbol {\alpha} _ {j} ^ {T _ {\lambda , 1}} \|}{\| \boldsymbol {\alpha} _ {j ^ {\prime}} ^ {T _ {\lambda , 1}} \|} \right| \\ = \left| \left(\ln \| \boldsymbol {w} _ {\lambda , j} ^ {T _ {\lambda , 1}} / \lambda \| - \ln \| \boldsymbol {\alpha} _ {j} ^ {T _ {\lambda , 1}} \|\right) - \left(\ln \| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {T _ {\lambda , 1}} / \lambda \| - \ln \| \boldsymbol {\alpha} _ {j ^ {\prime}} ^ {T _ {\lambda , 1}} \|\right) \right| \leq 2 \lambda^ {1 - 3 \varepsilon}, \\ \end{array}
$$

so $\lim_{\lambda \to 0^{+}}\frac{\left|\left|\boldsymbol{w}_{\lambda,j}^{T_{\lambda,1}}\right|\right|}{\left|\left|\boldsymbol{w}_{\lambda,j^{\prime}}^{T_{\lambda,1}}\right|\right|} = \lim_{t\to \infty}\frac{\|\boldsymbol{\alpha}_{j}^{t}\|}{\|\boldsymbol{\alpha}_{j^{\prime}}^{t}\|}.$

By Lemma 28 (i), (iv), and (v), for all $t \geq T_{\lambda,1}$ we have

$$
\left| \mathrm{d} \ln \frac {\| \boldsymbol {w} _ {\lambda , j} ^ {t} \|}{\| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {t} \|} / \mathrm{d} t \right| = | (\overline {{\boldsymbol {w}}} _ {\lambda , j} ^ {t} - \overline {{\boldsymbol {w}}} _ {\lambda , j ^ {\prime}} ^ {t}) ^ {\top} \boldsymbol {g} _ {\lambda} ^ {t} | <   \sqrt {8} \lambda^ {\varepsilon / 2} \| \boldsymbol {g} _ {\lambda} ^ {t} \|.
$$

By Lemma 28 (ix) and Lemma 32 (iii), for all $t \geq T_{\lambda,2}$ we have

$$
\| \pmb {g} _ {\lambda} ^ {t} \| \leq \eta_ {1} \| \pmb {v} ^ {*} - \pmb {v} _ {\lambda} ^ {t} \| <   \eta_ {1} \exp (- 5 \| \pmb {\gamma} _ {[ n ]} \| (t - T _ {\lambda , 2}) / 8) .
$$

Hence $\lim_{t\to \infty}\frac{\|\boldsymbol{w}_{\lambda,j}^t\|}{\|\boldsymbol{w}_{\lambda,j'}^t\|}$ exists. Moreover, by Lemma 32 (ii), for all $t\geq T_{\lambda ,2}$ we have

$$
\begin{array}{l} \left| \ln \frac {\| \boldsymbol {w} _ {\lambda , j} ^ {t} \|}{\| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {t} \|} - \ln \frac {\| \boldsymbol {w} _ {\lambda , j} ^ {T _ {\lambda , 1}} \|}{\| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {T _ {\lambda , 1}} \|} \right| <   \sqrt {8} \lambda^ {\varepsilon / 2} \Delta^ {2} \left(\ln \left(\frac {1}{\lambda}\right) \frac {(4 + \varepsilon / 2) d \Delta^ {2}}{\delta^ {6}} + \int_ {0} ^ {t - T _ {\lambda , 2}} \exp (- 5 \delta^ {2} t ^ {\prime} / 8) d t ^ {\prime}\right) \\ <   \sqrt {8} \lambda^ {\varepsilon / 2} \Delta^ {2} \left(\ln \left(\frac {1}{\lambda}\right) \frac {(4 + \varepsilon / 2) d \Delta^ {2}}{\delta^ {6}} + \frac {8}{5 \delta^ {2}}\right) \\ <   \sqrt {8} \lambda^ {\varepsilon / 2} \ln \left(\frac {1}{\lambda}\right) \frac {(4 + \varepsilon) d \Delta^ {4}}{\delta^ {6}} \\ <   \lambda^ {\varepsilon / 3} \ln \left(\frac {1}{\lambda}\right) \\ \end{array}
$$

since $\lambda^{-\varepsilon /6}\geq n^{\frac{9}{2} n\Delta^2 /\delta^3}\geq \left(2^{\frac{9}{4} n\Delta^2 /\delta^3}\right)^2 >\left(\frac{11n\Delta^2}{\delta^3}\right)^2 >\frac{\sqrt{8}(4 + \varepsilon)d\Delta^4}{\delta^6}$ . Therefore

$$
\left| \lim _ {t \to \infty} \frac {\| \boldsymbol {w} _ {\lambda , j} ^ {t} \|}{\| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {t} \|} - \frac {\| \boldsymbol {w} _ {\lambda , j} ^ {T _ {\lambda , 1}} \|}{\| \boldsymbol {w} _ {\lambda , j ^ {\prime}} ^ {T _ {\lambda , 1}} \|} \right| <   \lambda^ {\varepsilon / 3} \ln \left(\frac {1}{\lambda}\right),
$$

so $\lim_{\lambda \to 0^{+}}\lim_{t\to \infty}\frac{\|\boldsymbol{w}_{\lambda,j}^{t}\|}{\|\boldsymbol{w}_{\lambda,j^{\prime}}^{t}\|} = \lim_{\lambda \to 0^{+}}\frac{\left\|\boldsymbol{w}_{\lambda,j}^{T_{\lambda,1}}\right\|}{\left\|\boldsymbol{w}_{\lambda,j^{\prime}}^{T_{\lambda,1}}\right\|}.$

We are now in a position to prove the main theorem, restated from section 6. It establishes that, as the initialisation scale $\lambda$ tends to zero, the networks with zero loss to which the gradient flow converges tend to a network in the set $\Theta_{v^{*}}$ (defined in section 6) of balanced interpolators of rank 1.

Theorem 7. Under Assumptions 1 and 2, $L\left(\lim_{t \to \infty} \boldsymbol{\theta}_{\lambda}^{t}\right) = 0$ and $\lim_{\lambda \to 0^{+}} \lim_{t \to \infty} \boldsymbol{\theta}_{\lambda}^{t} \in \Theta_{\boldsymbol{v}^{*}}$ .

Proof. By Lemma 32 (v), we have $\lim_{t\to \infty}L(\pmb{\theta}_{\lambda}^{t}) = 0$ .

By Proposition 9, for all $t \in [0, \infty)$ we have

$$
\int_ {t} ^ {\infty} \left\| \frac {\mathrm{d}}{\mathrm{d} t ^ {\prime}} \boldsymbol {\theta} _ {\lambda} ^ {t ^ {\prime}} \right\| ^ {2} \mathrm{d} t ^ {\prime} = - \int_ {t} ^ {\infty} \frac {\mathrm{d}}{\mathrm{d} t ^ {\prime}} L \Big (\boldsymbol {\theta} _ {\lambda} ^ {t ^ {\prime}} \Big) \mathrm{d} t ^ {\prime} = L (\boldsymbol {\theta} _ {\lambda} ^ {t}).
$$

Hence $\theta_{\lambda}^{\infty} := \lim_{t \to \infty} \theta_{\lambda}^{t}$ exists, and since the loss function is continuous, we have $L(\theta_{\lambda}^{\infty}) = 0$ .

Let us write $\boldsymbol{\theta}_{\lambda}^{\infty} = ([a_{\lambda,1}^{\infty},\ldots ,a_{\lambda,m}^{\infty}],[\boldsymbol{w}_{\lambda ,1}^{\infty},\ldots ,\boldsymbol{w}_{\lambda ,m}^{\infty}]^{\top})$ , and let $\boldsymbol{v}_{\lambda}^{\infty}:= \sum_{j\in J_{+}}a_{\lambda ,j}^{\infty}\boldsymbol{w}_{\lambda ,j}^{\infty}$

Since $\Theta$ is closed, for all $j\in [m]$ we have $a_{\lambda ,j}^{\infty} = \| w_{\lambda ,j}^{\infty}\|$ .

Recalling span $\{\pmb{x}_1, \dots, \pmb{x}_n\} = \mathbb{R}^d$ and Lemma 28 (iii), we have $v_{\lambda}^{\infty} = v^{*}$ .

By Lemma 28 (i), for all $j \in J_{+}$ we have $\overline{\boldsymbol{w}}_{\lambda,j}^{\infty^{\top}} \boldsymbol{v}^{*} > 1 - 4\lambda^{\varepsilon}$ , so

$$
1 \leq \sum_ {j \in J _ {+}} \| \boldsymbol {w} _ {\lambda , j} ^ {\infty} \| ^ {2} <   \frac {1}{1 - 4 \lambda^ {\varepsilon}},
$$

and thus $\lim_{\lambda \to 0^{+}}\sum_{j\in J_{+}}\| \pmb{w}_{\lambda ,j}^{\infty}\| ^2 = 1.$

By Lemma 34, for all $j, j' \in J_{+}$ , $\lim_{\lambda \to 0^{+}} \frac{\|\boldsymbol{w}_{\lambda,j}^{\infty}\|}{\|\boldsymbol{w}_{\lambda,j'}^{\infty}\|}$ exists.

Hence, for all $j \in J_{+}$ , we have that $a_{j}^{\infty} := \lim_{\lambda \to 0^{+}} \| \boldsymbol{w}_{\lambda,j}^{\infty} \|$ and $\lim_{\lambda \to 0^{+}} \overline{\boldsymbol{w}}_{\lambda,j}^{\infty}$ exist, and so also $\boldsymbol{w}_{j}^{\infty} := \lim_{\lambda \to 0^{+}} \boldsymbol{w}_{\lambda,j}^{\infty}$ exists. Moreover, we have $a_{j}^{\infty} = \| \boldsymbol{w}_{j}^{\infty} \|$ and $\overline{\boldsymbol{w}}_{j}^{\infty} = \boldsymbol{v}^{*}$ for all $j \in J_{+}$ , and we have $\sum_{j \in J_{+}} \| \boldsymbol{w}_{j}^{\infty} \|^{2} = 1$ .

By Proposition 18, Lemma 19 (ii), Lemma 21, and Assumption 1 (v), for all $j \notin J_{+}$ , we have $\| \boldsymbol{w}_{\lambda,j}^{t} \| \leq \lambda \| \boldsymbol{z}_{j} \|$ for all $t \in [0,\infty)$ , so $a_{j}^{\infty} := \lim_{\lambda \to 0^{+}} a_{\lambda,j}^{\infty} = 0$ and $\boldsymbol{w}_{j}^{\infty} := \lim_{\lambda \to 0^{+}} \boldsymbol{w}_{\lambda,j}^{\infty} = \boldsymbol{0}$ .

Therefore $\boldsymbol{\theta}^{\infty} := ([a_{1}^{\infty}, \ldots, a_{m}^{\infty}], [w_{1}^{\infty}, \ldots, w_{m}^{\infty}]^{\top}) \in \Theta_{\boldsymbol{v}^{*}}$ .

Example 35. Continuing with the single run from Example 31, we illustrate in Figure 5 how, although the loss converges to zero exponentially fast, for a fixed positive initialisation scale the angles between the hidden neurons in the aligned bundle do not in general decrease to zero.

Figure 6 shows the course of the training from the point of view of the two measures of network complexity, namely the nuclear and square Euclidean norms: during the alignment phase they are both close to zero, they grow rapidly as the network departs from the saddle at the origin, and they converge towards 1 and 2 respectively as the loss converges to zero.

# G Proofs and examples for the interpolators

First we prove the following theorem, restated from section 7. For case $\mathcal{M} < 0$ , the main part of our argument shows that, if a global minimiser of $\| \pmb{\theta}\|^2$ was not a member of $\Theta_{\pmb{v}^*}$ , then we could obtain from each hidden neuron $\pmb{w}_j$ a vector $\pmb{p}_j$ such that the inner products of the inputs $\pmb{x}_i$ with the vector $\sum_{j\in [m]}a_j\pmb{p}_j$ coincide with the network outputs, the projection of each vector $\pmb{p}_j$ onto the teacher neuron has length at most $\| \pmb{w}_j\|$ , and at least one of those inequalities is strict, leading to a contradiction. For case $\mathcal{M} > 0$ , we provide counterexample interpolator networks, the Euclidean norm of whose parameters is smaller than the Euclidean norm of the networks in $\Theta_{\pmb{v}^*}$ .

Theorem 8. Under Assumptions 1 and 3:

![](images/ea2b1d2a1d982c2b4d3109e261773a08b07bdb67916f9e6b5285606be18d251a.jpg)

<details>
<summary>line</summary>

| iteration | Red Line Value | Blue Line Value |
| --------- | -------------- | --------------- |
| 10^0      | 10^3           | 10^0            |
| 10^1      | 10^3           | 10^0            |
| 10^2      | 10^3           | 10^0            |
| 10^3      | 10^3           | 10^-9           |
| 10^4      | 10^3           | 10^-9           |
| 10^5      | 10^3           | 10^-9           |
| 10^6      | 10^3           | 10^-21          |
</details>

![](images/17c20be52fcea766b8ae0dcc4c58bb108c9dff5c250fe9b660a65195ca680acf.jpg)

<details>
<summary>line</summary>

| iteration (x10^6) | training loss | max. angle b. active neurons |
| ----------------- | ------------- | ---------------------------- |
| 0                 | 10^3          | 10^3                         |
| 0.2               | ~10^-9        | 10^3                         |
| 0.4               | ~10^-9        | 10^3                         |
| 0.6               | ~10^-9        | 10^3                         |
| 0.8               | ~10^-9        | 10^3                         |
| 1.0               | ~10^-9        | 10^3                         |
| 1.2               | ~10^-9        | 10^3                         |
| 1.4               | ~10^-9        | 10^3                         |
| 1.6               | ~10^-9        | 10^3                         |
| 1.8               | ~10^-9        | 10^3                         |
| 2.0               | ~10^-9        | 10^3                         |
| 2.2               | ~10^-9        | 10^3                         |
| 2.4               | ~10^-9        | 10^3                         |
| 2.6               | ~10^-9        | 10^3                         |
| 2.8               | ~10^-9        | 10^3                         |
</details>

Figure 5: The evolution of the training loss and the maximum angle between active hidden neurons during the training. The two plots are of the same data, the vertical axes are logarithmic, the horizontal axis is logarithmic in the top plot and linear in the bottom plot, and the angles are in degrees.

![](images/4c1c1fea021990b0c06ae1c38509b2a57674f824bf863c8b2f135825b6109bf5.jpg)

<details>
<summary>line</summary>

| iteration | nuclear norm | squared Euclidean norm |
| --------- | ------------ | ---------------------- |
| 1         | 0.0          | 0.0                    |
| 10        | 0.0          | 0.0                    |
| 100       | 0.0          | 0.0                    |
| 1000      | 0.9          | 1.5                    |
| 10000     | 1.0          | 1.8                    |
| 100000    | 1.0          | 1.9                    |
| 1000000   | 1.0          | 2.0                    |
</details>

Figure 6: The evolution of the nuclear and square Euclidean norms during the training. The horizontal axis is logarithmic, and the vertical axis is linear.

(i) if $\mathcal{M} < 0$ then $\Theta_{\boldsymbol{v}^*}$ is the set of all global minimisers of $\| \pmb{\theta}\|^2$ subject to $L(\pmb{\theta}) = 0$ ;   
(ii) if $\mathcal{M} > 0$ then no point in $\Theta_{\boldsymbol{v}^*}$ is a global minimiser of $\| \pmb{\theta}\|^2$ subject to $L(\pmb{\theta}) = 0$ .

Proof. For all $\boldsymbol{\theta} = (\boldsymbol{a},\boldsymbol{W})\in \Theta_{\boldsymbol{v}^{*}}$ we have $L(\boldsymbol {\theta}) = 0$ and $\| \pmb {\theta}\| ^2 = \sum_{j\in [m]}(a_j^2 +\| \boldsymbol {w}_j\| ^2) = 2.$

To establish the case when $\mathcal{M} < 0$ , supposing $\boldsymbol{\theta} = (\boldsymbol{a}, \boldsymbol{W}) \in \mathbb{R}^m \times \mathbb{R}^{m \times d}$ is a global minimiser of $\|\boldsymbol{\theta}\|^2$ subject to $L(\boldsymbol{\theta}) = 0$ , it suffices to show $\boldsymbol{\theta} \in \Theta_{\boldsymbol{v}^*}$ .

By the minimality of $\|\theta\|^{2}$ subject to $L(\boldsymbol{\theta})=0$ , for all $j\in[m]$ , if $a_{j}=0$ then $w_{j}=0$ , and also if $\forall i\in[d]\colon\sigma(\boldsymbol{w}_{j}^{\top}\boldsymbol{x}_{i})=0$ then $a_{j}=0$ .

For all $j \in [m]$ , if $a_j = 0$ and $w_j = 0$ , then removing $a_j$ and $w_j$ from $\theta$ preserves the values of $\| \theta \|^2$ and $L(\theta)$ , and the truth or falsity of $\theta \in \Theta_{v^*}$ . Hence we may assume for all $j \in [m]$ that $a_j \neq 0$ and $\exists i \in [d]$ : $\sigma(\boldsymbol{w}_j^\top \boldsymbol{x}_i) \neq 0$ .

For all $j \in [m]$ , replacing $a_j$ by $\sqrt{\|\boldsymbol{w}_j\| / |a_j|} a_j$ and $\boldsymbol{w}_j$ by $\sqrt{|a_j| / \| \boldsymbol{w}_j\|} \boldsymbol{w}_j$ preserves $L(\boldsymbol{\theta})$ , and decreases $\| \boldsymbol{\theta}\|^2$ unless $|a_j| = \| \boldsymbol{w}_j\|$ . Hence $\boldsymbol{\theta} \in \Theta$ .

For all $j \in [m]$ , let $K_{j} := \{k \in [d] \mid \boldsymbol{w}_{j}^{\top} \boldsymbol{x}_{k} \geq 0\}$ and

$$
\boldsymbol {p} _ {j} := \sum_ {k \in K _ {j}} (\boldsymbol {w} _ {j} ^ {\top} \boldsymbol {x} _ {k}) \chi_ {k} \quad \boldsymbol {q} _ {j} := \sum_ {k \notin K _ {j}} - (\boldsymbol {w} _ {j} ^ {\top} \boldsymbol {x} _ {k}) \chi_ {k},
$$

so that $w_{j} = p_{j} - q_{j}$ . Observe also that since $\exists i \in [d]$ : $\sigma(\boldsymbol{w}_{j}^{\top} \boldsymbol{x}_{i}) \neq 0$ , we have $p_{j} \neq 0$ .

Claim 36. For all $j \in [m]$ we have $|p_{j}^{\top}v^{*}| \leq \|w_{j}\|$ , and if $q_{j} \neq 0$ then the inequality is strict.

Proof of claim. Suppose $j \in [m]$ . If $\mathbf{q}_j = \mathbf{0}$ then $w_j = p_j$ . If $q_j \neq 0$ then

$$
\begin{array}{l} \left\| \boldsymbol {w} _ {j} \right\| ^ {2} = \left\| \boldsymbol {p} _ {j} \right\| ^ {2} + \left\| \boldsymbol {q} _ {j} \right\| ^ {2} - 2 \left\| \boldsymbol {p} _ {j} \right\| \left\| \boldsymbol {q} _ {j} \right\| \cos \angle (\boldsymbol {p} _ {j}, \boldsymbol {q} _ {j}) \\ > \| \boldsymbol {p} _ {j} \| ^ {2} + \| \boldsymbol {q} _ {j} \| ^ {2} - 2 \| \boldsymbol {p} _ {j} \| \| \boldsymbol {q} _ {j} \| \sin \angle (\boldsymbol {p} _ {j}, \boldsymbol {v} ^ {*}) \\ = \left\| \boldsymbol {p} _ {j} \right\| ^ {2} \cos^ {2} \angle (\boldsymbol {p} _ {j}, \boldsymbol {v} ^ {*}) + \left(\left\| \boldsymbol {p} _ {j} \right\| \sin \angle (\boldsymbol {p} _ {j}, \boldsymbol {v} ^ {*}) - \left\| \boldsymbol {q} _ {j} \right\|\right) ^ {2} \\ \geq \left\| \boldsymbol {p} _ {j} \right\| ^ {2} \cos^ {2} \angle (\boldsymbol {p} _ {j}, \boldsymbol {v} ^ {*}) \\ = \left(\boldsymbol {p} _ {j} ^ {\top} \boldsymbol {v} ^ {*}\right) ^ {2}. \\ \end{array}
$$

![](images/587dd0e8d8d026b9662f408cd4c4218961f9a1af7023d852cde0673b9e9f4bd1.jpg)

Now for all $i\in [d]$ we have

$$
\left(\sum_ {j \in [ m ]} a _ {j} \boldsymbol {p} _ {j}\right) ^ {\top} \boldsymbol {x} _ {i} = \sum_ {j \in [ m ]} a _ {j} \boldsymbol {p} _ {j} ^ {\top} \boldsymbol {x} _ {i} = \sum_ {j \in [ m ]} a _ {j} \sigma (\boldsymbol {w} _ {j} ^ {\top} \boldsymbol {x} _ {i}) = \boldsymbol {v} ^ {* ^ {\top}} \boldsymbol {x} _ {i}.
$$

Since $\operatorname{span}\{\pmb{x}_1, \ldots, \pmb{x}_d\} = \mathbb{R}^d$ , we infer $\sum_{j \in [m]} a_j \pmb{p}_j = \pmb{v}^*$ , so by Claim 36 we have

$$
1 = \sum_ {j \in [ m ]} a _ {j} \boldsymbol {p} _ {j} ^ {\top} \boldsymbol {v} ^ {*} \leq \sum_ {j \in [ m ]} | a _ {j} \boldsymbol {p} _ {j} ^ {\top} \boldsymbol {v} ^ {*} | \leq \sum_ {j \in [ m ]} \| a _ {j} \boldsymbol {w} _ {j} \| = \frac {1}{2} \sum_ {j \in [ m ]} (a _ {j} ^ {2} + \| \boldsymbol {w} _ {j} \| ^ {2}) = \frac {1}{2} \| \boldsymbol {\theta} \| ^ {2} \leq 1,
$$

and if $q_{j} \neq 0$ for some $j \in [m]$ then the second of the three inequalities is strict. However, all three inequalities must be equalities, so also for all $j \in [m]$ we have $q_{j} = 0$ . Hence $a_{j} w_{j}^{\top} v^{*} = a_{j} p_{j}^{\top} v^{*} = \|a_{j} w_{j}\|$ , and thus $\overline{a_{j} w_{j}} = v^{*}$ . Since $a_{j} < 0$ would imply $\overline{w}_{j} = -v^{*}$ , which would contradict $q_{j} = 0$ , we have $a_{j} > 0$ and $\overline{w}_{j} = v^{*}$ . Therefore $\theta \in \Theta_{v^{*}}$ .

To establish the case when $\mathcal{M} > 0$ , it suffices to exhibit $\boldsymbol{\theta} = (\boldsymbol{a}, \boldsymbol{W}) \in \mathbb{R}^m \times \mathbb{R}^{m \times d}$ such that $L(\boldsymbol{\theta}) = 0$ and $\|\boldsymbol{\theta}\|^2 < 2$ .

Let $\emptyset \subsetneq K \subsetneq [d]$ , $\mathbf{0} \neq \mathbf{p} \in \mathrm{cone}\{\chi_k \mid k \in K\}$ , and $\mathbf{0} \neq \mathbf{q} \in \mathrm{cone}\{\chi_k \mid k \notin K\}$ be such that $\cos \angle (\mathbf{p}, \mathbf{q}) > \sin \angle (\mathbf{p}, \mathbf{v}^*)$ . We have $\overline{\mathbf{p}} = \sum_{k \in K} b_k \chi_k$ for some $b_k \geq 0$ , and $\overline{\mathbf{q}} = \sum_{k \notin K} c_k \chi_k$ for some $c_k \geq 0$ . Since $\operatorname{span}\{\chi_1, \ldots, \chi_d\} = \mathbb{R}^d$ , we have $\cos \angle (\mathbf{p}, \mathbf{q}) < 1$ .

Case $\angle (\pmb {p},\pmb{v}^{*})\leq \pi /2.$ Then $\cos \angle (\pmb {p},\pmb{v}^{*}) > \sin \angle (\pmb {p},\pmb {q}).$

Let $\xi := \min \left\{ \min \{ y_k / b_k \mid k \in K \land b_k \neq 0 \}, \cos \angle (\boldsymbol{p}, \boldsymbol{v}^*) - \sin \angle (\boldsymbol{p}, \boldsymbol{q}) \right\}$ , $r := \overline{\boldsymbol{p}} - \overline{\boldsymbol{q}} \overline{\boldsymbol{q}}^\top \overline{\boldsymbol{p}}$ ,

$$
a _ {1} := 1 \quad \boldsymbol {w} _ {1} := \boldsymbol {v} ^ {*} - \xi \overline {{\boldsymbol {p}}}
$$

$$
a _ {2} := \sqrt {\xi \| \boldsymbol {r} \|} \quad \boldsymbol {w} _ {2} := \sqrt {\xi / \| \boldsymbol {r} \|} \boldsymbol {r},
$$

and $a_{j} := 0$ and $w_{j} := 0$ for all j > 2.

From

$$
\boldsymbol {r} ^ {\top} \boldsymbol {x} _ {i} = \left\{ \begin{array}{l l} b _ {i} & \text {if} i \in K, \\ - c _ {i} \cos \angle (\boldsymbol {p}, \boldsymbol {q}) & \text {if} i \notin K, \end{array} \right.
$$

it follows that $h_{\boldsymbol{\theta}}(\boldsymbol{x}_{i}) = y_{i}$ for all $i \in [d]$ , i.e. $L(\boldsymbol{\theta}) = 0$ .

We have

$$
\begin{array}{l} \| \boldsymbol {\theta} \| ^ {2} = a _ {1} ^ {2} + \| \boldsymbol {w} _ {1} \| ^ {2} + a _ {2} ^ {2} + \| \boldsymbol {w} _ {2} \| ^ {2} \\ = 2 + \xi^ {2} - 2 \xi \overline {{{\boldsymbol {p}}}} ^ {\top} \boldsymbol {v} ^ {*} + 2 \xi \| \boldsymbol {r} \| \\ = 2 - \xi \left[ 2 (\cos \angle (\boldsymbol {p}, \boldsymbol {v} ^ {*}) - \sin \angle (\boldsymbol {p}, \boldsymbol {q})) - \xi \right] \\ \leq 2 - \xi^ {2}. \\ \end{array}
$$

Case $\angle(\boldsymbol{p},\boldsymbol{v}^{*})>\pi/2$ . Then $-\cos\angle(\boldsymbol{p},\boldsymbol{v}^{*})>\sin\angle(\boldsymbol{p},\boldsymbol{q})$ .

Let $\xi := -\cos \angle (\pmb {p},\pmb{v}^{*}) - \sin \angle (\pmb {p},\pmb {q}),\pmb {r} := \overline{\pmb{p}} -\overline{\pmb{q}}\overline{\pmb{q}}^{\top}\overline{\pmb{p}},$

$$
\begin{array}{l} a _ {1} := 1 \quad \boldsymbol {w} _ {1} := \boldsymbol {v} ^ {*} + \xi \overline {{\boldsymbol {p}}} \\ a _ {2} := - \sqrt {\xi \| \boldsymbol {r} \|} \quad \boldsymbol {w} _ {2} := \sqrt {\xi / \| \boldsymbol {r} \|} \boldsymbol {r}, \\ \end{array}
$$

and $a_{j} := 0$ and $w_{j} := 0$ for all j > 2.

From

$$
\boldsymbol {r} ^ {\top} \boldsymbol {x} _ {i} = \left\{ \begin{array}{l l} b _ {i} & \text { if } i \in K, \\ - c _ {i} \cos \angle (\boldsymbol {p}, \boldsymbol {q}) & \text { if } i \notin K, \end{array} \right.
$$

it follows that $h_{\boldsymbol{\theta}}(\boldsymbol{x}_{i}) = y_{i}$ for all $i \in [d]$ , i.e. $L(\boldsymbol{\theta}) = 0$ .

We have

$$
\begin{array}{l} \left\| \boldsymbol {\theta} \right\| ^ {2} = a _ {1} ^ {2} + \left\| \boldsymbol {w} _ {1} \right\| ^ {2} + a _ {2} ^ {2} + \left\| \boldsymbol {w} _ {2} \right\| ^ {2} \\ = 2 + \xi^ {2} + 2 \xi \overline {{\boldsymbol {p}}} ^ {\top} \boldsymbol {v} ^ {*} + 2 \xi \| \boldsymbol {r} \| \\ = 2 - \xi \left[ 2 (- \cos \angle (\boldsymbol {p}, \boldsymbol {v} ^ {*}) - \sin \angle (\boldsymbol {p}, \boldsymbol {q})) - \xi \right] \\ = 2 - \xi^ {2}. \\ \end{array}
$$

Example 37. Now we present two families of examples of a teacher neuron and training points that respectively satisfy: $\mathcal{M} < 0$ for any $d > 1$ , and $\mathcal{M} > 0$ for any $d > 2$ .

Let $\{e_i\}_{i=1}^d$ denote the standard basis of $\mathbb{R}^d$ .

$\mathcal{M} < 0$ . Let $\xi \in (0,1)$ and consider, for all $i \in [d]$ , vectors

$$
\boldsymbol {x} _ {i} := \left(1 - \frac {d - 1}{d} (1 - \xi)\right) \boldsymbol {e} _ {i} + \frac {1 - \xi}{d} \sum_ {k \neq i} \boldsymbol {e} _ {k}.
$$

Take $s := (1, \ldots, 1) \in \mathbb{R}^{d}$ and $v^{*} := \overline{s}$ .

It can be checked that, for vectors $\chi_{1},\ldots,\chi_{d}$ defined by

$$
\boldsymbol {\chi} _ {k} := \frac {1}{\xi} \left(\boldsymbol {e} _ {k} - \frac {1 - \xi}{d} \boldsymbol {s}\right),
$$

we have $[\chi_{1},\ldots,\chi_{d}]^{\top}=X^{-1}$ .

Notice that whenever $k \neq i$ we have $\boldsymbol{\chi}_k^\top \boldsymbol{\chi}_i = \frac{1}{\xi^2} \left( -2 \frac{1 - \xi}{d} + \left( \frac{1 - \xi}{d} \right)^2 d \right) = \frac{1 - \xi}{\xi^2 d} (-2 + 1 - \xi) < 0$ . Hence for all $\emptyset \subsetneq K \subsetneq [d]$ , all $\mathbf{0} \neq \mathbf{p} \in \mathrm{cone}\{\boldsymbol{\chi}_k \mid k \in K\}$ , and all $\mathbf{0} \neq \mathbf{q} \in \mathrm{cone}\{\boldsymbol{\chi}_i \mid i \notin K\}$ we have $\cos \angle (\mathbf{p}, \mathbf{q}) < 0$ . Thus $\mathcal{M} < 0$ .

It remains to verify that $\angle (\pmb{v}^{*},\pmb{x}_{i}) < \pi /4$ for all $i$ . Indeed

$$
\begin{array}{l} \left\| \boldsymbol {x} _ {i} \right\| ^ {2} = \left(1 - \frac {d - 1}{d} (1 - \xi)\right) ^ {2} + (d - 1) \left(\frac {1 - \xi}{d}\right) ^ {2} \\ = \left(\frac {1}{d} + \xi \left(1 - \frac {1}{d}\right)\right) ^ {2} + \frac {d - 1}{d ^ {2}} - \frac {d - 1}{d ^ {2}} 2 \xi + \frac {d - 1}{d ^ {2}} \xi^ {2} \\ = \frac {1}{d ^ {2}} + 2 \xi \frac {1}{d} \left(1 - \frac {1}{d}\right) + \left(1 - \frac {1}{d}\right) ^ {2} \xi^ {2} + \frac {d - 1}{d ^ {2}} - \frac {d - 1}{d ^ {2}} 2 \xi + \frac {d - 1}{d ^ {2}} \xi^ {2} \\ = \frac {1}{d} + \frac {(d - 1) ^ {2} + (d - 1)}{d ^ {2}} \xi^ {2} \\ = \frac {1}{d} + \frac {d - 1}{d} \xi^ {2}, \\ \end{array}
$$

so in particular $\| s\|^2 \| x_i\|^2 = 1 + (d - 1)\xi^2$ . Therefore

$$
\begin{array}{l} \cos \angle (\boldsymbol {v} ^ {*}, \boldsymbol {x} _ {i}) = \frac {\boldsymbol {s} ^ {\top} \boldsymbol {x} _ {i}}{\| \boldsymbol {s} \| \| \boldsymbol {x} _ {i} \|} \\ > \frac {\left(1 - \frac {d - 1}{d} (1 - \xi)\right) + (d - 1) \frac {1 - \xi}{d}}{1 + \frac {1}{2} (d - 1) \xi^ {2}} \\ = \frac {1}{1 + \frac {1}{2} (d - 1) \xi^ {2}}, \\ \end{array}
$$

so it suffices to take $\xi \leq \sqrt{\frac{2(\sqrt{2} - 1)}{d - 1}}$ .

$\mathcal{M} > 0$ . For $d > 2$ , let $b \geq 11$ , and consider the data points

$$
\begin{array}{l} \boldsymbol {x} _ {1} := b \boldsymbol {e} _ {1} \\ \boldsymbol {x} _ {2} := b \boldsymbol {e} _ {1} - \sqrt {b} \boldsymbol {e} _ {2} + \boldsymbol {e} _ {3} \\ \boldsymbol {x} _ {3} := b \boldsymbol {e} _ {1} + \sqrt {b} \boldsymbol {e} _ {2} + \boldsymbol {e} _ {3} \\ \boldsymbol {x} _ {i} := b \boldsymbol {e} _ {1} + \boldsymbol {e} _ {i} \quad \text { for   all } 4 \leq i \leq d \\ \end{array}
$$

and the teacher neuron $v^{*} := \frac{4}{5}e_{1} + \frac{3}{5}e_{3}$ .

For all $i$ we have

$$
\cos \angle (\boldsymbol {v} ^ {*}, \boldsymbol {x} _ {i}) > \frac {4 b}{5 \sqrt {b ^ {2} + b + 1}} > \frac {4}{5} \frac {b}{b + 1} \geq \frac {4}{5} \frac {1 1}{1 2} = \frac {1 1}{1 5} > \frac {1}{\sqrt {2}}.
$$

Straightforward calculation shows that, for $[\chi_{1},\ldots,\chi_{d}]^{\top}:=X^{-1}$ , we have

$$
\chi_ {2} = - \frac {1}{2 \sqrt {b}} e _ {2} + \frac {1}{2} e _ {3}
$$

$$
\chi_ {3} = \frac {1}{2 \sqrt {b}} e _ {2} + \frac {1}{2} e _ {3}
$$

and hence

$$
\begin{array}{l} \cos \angle (\boldsymbol {\chi} _ {2}, \boldsymbol {\chi} _ {3}) - \sin \angle (\boldsymbol {\chi} _ {2}, \boldsymbol {v} ^ {*}) = \frac {\frac {1}{4} - \frac {1}{4 b}}{\frac {1}{4} + \frac {1}{4 b}} - \sqrt {1 - \frac {9}{1 0 0 \left(\frac {1}{4} + \frac {1}{4 b}\right)}} \\ = \frac {b - 1}{b + 1} - \sqrt {1 - \frac {9}{2 5} \frac {b}{b + 1}} \\ \geq \frac {5}{6} - \sqrt {\frac {6 7}{1 0 0}} \\ > 0. \\ \end{array}
$$

![](images/c50d89294501271ac3b5671cc4bd2568eb7bf210aeaf969f45d5a7573505f51e.jpg)

Remark 38. (i) For any $\boldsymbol{\theta}$ such that $L(\boldsymbol{\theta}) = 0$ , we have $\| \boldsymbol{\theta} \|^2 \geq 2h_{\boldsymbol{\theta}}(\boldsymbol{x}_1) / \| \boldsymbol{x}_1 \| = 2\cos \angle (\boldsymbol{v}^*, \boldsymbol{x}_1) > \sqrt{2}$ .

(ii) For $d = 2$ , since $\angle (\pmb{x}_1, \pmb{x}_2) < \pi / 2$ , we have $\angle (\chi_1, \chi_2) > \pi / 2$ , so necessarily $\mathcal{M} < 0$ .

(iii) As its proof above shows, Theorem 8 remains true if we relax the correlation between the teacher neuron and the training points to $\angle (\pmb{v}^{*},\pmb{x}_{i}) < \pi /2$ for all $i$ .

# H Additional information about the experiments

For both the centred and the uncentred schemes of generating the training dataset (defined in section 8), we train a one-hidden layer ReLU network by gradient descent with learning rate 0.01, from a balanced initialisation such that $z_j \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \frac{1}{dm} I_d)$ and $s_j \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}\{\pm 1\}$ , for a range of initialisation scales $\lambda$ , and for several combinations of input dimensions $d$ and network widths $m$ .

The plots in Figure 7, which extends Figure 1 in the main, are obtained by varying the input dimension as d = 4, 16, 64, 256, 1024 while keeping the network width at m = 200. The plots in Figure 8 are obtained with input dimension d = 1024 by varying the network width as m = 25, 50, 200. For all twelve plots, we vary the initialisation scale as $\lambda = 4^{2}, 4^{1}, \ldots, 4^{-12}, 4^{-13}$ , and we train the network until the number of iterations reaches $2 \cdot 10^{7}$ or the loss drops below $10^{-9}$ . The plots are in line with Theorem 7, showing how the three different proxies of rank decrease as $\lambda$ decreases.

Figure 9 complements Figure 2 in the main, illustrating exponential convergence of the training loss (cf. Theorem 6) and reduction of the outside distribution test loss as $\lambda$ decreases, for the uncentred scheme of generating the training dataset.

The medians plotted in Figure 7 and Figure 8, as well as the corresponding standard deviations, can be found in Tables 1–6 and Tables 7–12 respectively.

The experiments were run using Python 3.10.4 and Pytorch 1.12.1 with CUDA 11.7 on a cluster utilising Intel Xeon Platinum 8268 processors. Some experiments for dimension 1024 also used NVIDIA RTX 6000 GPUs. The time taken per iteration greatly depends on the dimension and the width. For dimension 1024 and width 200, about 300 iterations per second could be performed on the CPU. The GPU was about $20\%$ faster in this setting. The total number of iterations performed for dimension 1024 was about 1.6 billion. Experiments for lower dimensions or smaller widths are less demanding.

Overall, these numerical results correspond to our theoretical predictions, and suggest that the training dynamics and the implicit bias we established theoretically occur in practical settings in which some of our assumptions are relaxed.

![](images/f6e8d520ac4b7cf52cfcb360aa027096a61dbefb347556eee169f893e9f08536.jpg)  
d = 4 d = 16 d = 64 \* d = 256 d = 1024

Figure 7: Dependence of the maximum angle between active hidden neurons, of the average angle between active hidden neurons, and of the nuclear norm of the hidden-layer weights on the initialisation scale $\lambda$ , for the two generation schemes of the training dataset, the five different input dimensions, and network width 200, at the end of the training. Both axes are logarithmic, and each point plotted shows the median over five trials.

![](images/1c60cd703e476a36189dc201728bbc970c94d11160ce92bee1b5876ecc3e533f.jpg)  
Figure 8: Dependence of the maximum angle between active hidden neurons, of the average angle between active hidden neurons, and of the nuclear norm of the hidden-layer weights on the initialisation scale $\lambda$ , for the two generation schemes of the training dataset, the three different network widths, and input dimension 1024, at the end of the training. Both axes are logarithmic, and each point plotted shows the median over five trials.

![](images/1f8128ee1138e9e7bc32144d2dd703777aa5c120f744a5b95832b6c54424cf91.jpg)  
$\text{— } \lambda = 4^{-2} \text{ — } \lambda = 4^{-3} \text{ — } \lambda = 4^{-4} \text{ — } \lambda = 4^{-5} \text{ — } \lambda = 4^{-6} \text{ — } \lambda = 4^{-7} \text{ — } \lambda = 4^{-8}$

Figure 9: Evolution of the training loss, and of an outside distribution test loss, during training for an example uncentred training dataset in dimension 16 and with m = 25. The horizontal axes show iterations; they are logarithmic for the training loss, and linear for the test loss. The vertical axes are logarithmic.

Table 1: The medians over five trials plotted in Figure 7 on the top left, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td> $d = 4$ </td><td></td><td> $d = 16$ </td><td></td><td> $d = 64$ </td><td></td></tr><tr><td> $4^{2}$ </td><td> $1.665 \cdot 10^{2}$ </td><td> $(1.045 \cdot 10^{1})$ </td><td> $1.405 \cdot 10^{2}$ </td><td> $(3.256 \cdot 10^{0})$ </td><td> $1.136 \cdot 10^{2}$ </td><td> $(1.466 \cdot 10^{0})$ </td></tr><tr><td> $4^{1}$ </td><td> $1.743 \cdot 10^{2}$ </td><td> $(3.577 \cdot 10^{0})$ </td><td> $1.399 \cdot 10^{2}$ </td><td> $(3.818 \cdot 10^{0})$ </td><td> $1.140 \cdot 10^{2}$ </td><td> $(1.252 \cdot 10^{0})$ </td></tr><tr><td> $4^{0}$ </td><td> $1.644 \cdot 10^{2}$ </td><td> $(7.951 \cdot 10^{0})$ </td><td> $1.336 \cdot 10^{2}$ </td><td> $(2.045 \cdot 10^{0})$ </td><td> $1.122 \cdot 10^{2}$ </td><td> $(3.772 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.592 \cdot 10^{2}$ </td><td> $(2.641 \cdot 10^{1})$ </td><td> $1.246 \cdot 10^{2}$ </td><td> $(8.589 \cdot 10^{0})$ </td><td> $1.089 \cdot 10^{2}$ </td><td> $(2.359 \cdot 10^{1})$ </td></tr><tr><td> $4^{-2}$ </td><td> $9.943 \cdot 10^{1}$ </td><td> $(5.393 \cdot 10^{1})$ </td><td> $1.114 \cdot 10^{2}$ </td><td> $(1.892 \cdot 10^{1})$ </td><td> $9.976 \cdot 10^{1}$ </td><td> $(3.900 \cdot 10^{1})$ </td></tr><tr><td> $4^{-3}$ </td><td> $7.037 \cdot 10^{1}$ </td><td> $(5.924 \cdot 10^{1})$ </td><td> $1.003 \cdot 10^{2}$ </td><td> $(3.897 \cdot 10^{1})$ </td><td> $5.205 \cdot 10^{1}$ </td><td> $(4.263 \cdot 10^{1})$ </td></tr><tr><td> $4^{-4}$ </td><td> $3.106 \cdot 10^{1}$ </td><td> $(6.113 \cdot 10^{1})$ </td><td> $7.797 \cdot 10^{1}$ </td><td> $(4.700 \cdot 10^{1})$ </td><td> $1.391 \cdot 10^{1}$ </td><td> $(4.877 \cdot 10^{1})$ </td></tr><tr><td> $4^{-5}$ </td><td> $8.538 \cdot 10^{0}$ </td><td> $(6.251 \cdot 10^{1})$ </td><td> $2.931 \cdot 10^{1}$ </td><td> $(4.519 \cdot 10^{1})$ </td><td> $3.500 \cdot 10^{0}$ </td><td> $(4.279 \cdot 10^{1})$ </td></tr><tr><td> $4^{-6}$ </td><td> $2.180 \cdot 10^{0}$ </td><td> $(4.159 \cdot 10^{1})$ </td><td> $7.605 \cdot 10^{0}$ </td><td> $(3.021 \cdot 10^{1})$ </td><td> $8.756 \cdot 10^{-1}$ </td><td> $(2.551 \cdot 10^{1})$ </td></tr><tr><td> $4^{-7}$ </td><td> $5.476 \cdot 10^{-1}$ </td><td> $(2.987 \cdot 10^{1})$ </td><td> $1.915 \cdot 10^{0}$ </td><td> $(1.084 \cdot 10^{1})$ </td><td> $2.189 \cdot 10^{-1}$ </td><td> $(6.959 \cdot 10^{0})$ </td></tr><tr><td> $4^{-8}$ </td><td> $1.371 \cdot 10^{-1}$ </td><td> $(6.403 \cdot 10^{0})$ </td><td> $4.793 \cdot 10^{-1}$ </td><td> $(2.760 \cdot 10^{0})$ </td><td> $5.473 \cdot 10^{-2}$ </td><td> $(1.754 \cdot 10^{0})$ </td></tr><tr><td> $4^{-9}$ </td><td> $3.428 \cdot 10^{-2}$ </td><td> $(1.585 \cdot 10^{0})$ </td><td> $1.198 \cdot 10^{-1}$ </td><td> $(6.903 \cdot 10^{-1})$ </td><td> $1.368 \cdot 10^{-2}$ </td><td> $(4.390 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-10}$ </td><td> $8.570 \cdot 10^{-3}$ </td><td> $(3.956 \cdot 10^{-1})$ </td><td> $2.996 \cdot 10^{-2}$ </td><td> $(1.728 \cdot 10^{-1})$ </td><td> $3.421 \cdot 10^{-3}$ </td><td> $(1.098 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-11}$ </td><td> $2.142 \cdot 10^{-3}$ </td><td> $(9.884 \cdot 10^{-2})$ </td><td> $7.491 \cdot 10^{-3}$ </td><td> $(4.321 \cdot 10^{-2})$ </td><td> $8.552 \cdot 10^{-4}$ </td><td> $(2.744 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-12}$ </td><td> $5.356 \cdot 10^{-4}$ </td><td> $(2.471 \cdot 10^{-2})$ </td><td> $1.873 \cdot 10^{-3}$ </td><td> $(1.080 \cdot 10^{-2})$ </td><td> $2.138 \cdot 10^{-4}$ </td><td> $(6.860 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-13}$ </td><td> $1.339 \cdot 10^{-4}$ </td><td> $(6.176 \cdot 10^{-3})$ </td><td> $4.682 \cdot 10^{-4}$ </td><td> $(2.701 \cdot 10^{-3})$ </td><td> $5.343 \cdot 10^{-5}$ </td><td> $(1.715 \cdot 10^{-3})$ </td></tr></table>

<table><tr><td> $\lambda$ </td><td colspan="2"> $d = 256$ </td><td colspan="2"> $d = 1024$ </td></tr><tr><td> $4^{2}$ </td><td> $1.030 \cdot 10^{2}$ </td><td> $(7.357 \cdot 10^{-1})$ </td><td> $9.654 \cdot 10^{1}$ </td><td> $(2.816 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $1.028 \cdot 10^{2}$ </td><td> $(6.489 \cdot 10^{-1})$ </td><td> $9.650 \cdot 10^{1}$ </td><td> $(4.394 \cdot 10^{-1})$ </td></tr><tr><td> $4^{0}$ </td><td> $1.037 \cdot 10^{2}$ </td><td> $(1.092 \cdot 10^{0})$ </td><td> $8.695 \cdot 10^{1}$ </td><td> $(5.651 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $9.990 \cdot 10^{1}$ </td><td> $(4.216 \cdot 10^{0})$ </td><td> $2.997 \cdot 10^{1}$ </td><td> $(1.003 \cdot 10^{1})$ </td></tr><tr><td> $4^{-2}$ </td><td> $9.841 \cdot 10^{1}$ </td><td> $(2.837 \cdot 10^{1})$ </td><td> $7.662 \cdot 10^{0}$ </td><td> $(2.820 \cdot 10^{0})$ </td></tr><tr><td> $4^{-3}$ </td><td> $9.404 \cdot 10^{1}$ </td><td> $(4.252 \cdot 10^{1})$ </td><td> $1.918 \cdot 10^{0}$ </td><td> $(7.110 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-4}$ </td><td> $5.304 \cdot 10^{1}$ </td><td> $(4.524 \cdot 10^{1})$ </td><td> $4.796 \cdot 10^{-1}$ </td><td> $(1.779 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.427 \cdot 10^{1}$ </td><td> $(3.982 \cdot 10^{1})$ </td><td> $1.199 \cdot 10^{-1}$ </td><td> $(4.447 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-6}$ </td><td> $3.583 \cdot 10^{0}$ </td><td> $(1.824 \cdot 10^{1})$ </td><td> $2.998 \cdot 10^{-2}$ </td><td> $(1.112 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-7}$ </td><td> $8.963 \cdot 10^{-1}$ </td><td> $(4.782 \cdot 10^{0})$ </td><td> $7.494 \cdot 10^{-3}$ </td><td> $(2.779 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-8}$ </td><td> $2.241 \cdot 10^{-1}$ </td><td> $(1.199 \cdot 10^{0})$ </td><td> $1.874 \cdot 10^{-3}$ </td><td> $(6.948 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-9}$ </td><td> $5.602 \cdot 10^{-2}$ </td><td> $(2.998 \cdot 10^{-1})$ </td><td> $4.684 \cdot 10^{-4}$ </td><td> $(1.737 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-10}$ </td><td> $1.401 \cdot 10^{-2}$ </td><td> $(7.495 \cdot 10^{-2})$ </td><td> $1.171 \cdot 10^{-4}$ </td><td> $(4.342 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-11}$ </td><td> $3.501 \cdot 10^{-3}$ </td><td> $(1.874 \cdot 10^{-2})$ </td><td> $2.930 \cdot 10^{-5}$ </td><td> $(1.086 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-12}$ </td><td> $8.753 \cdot 10^{-4}$ </td><td> $(4.685 \cdot 10^{-3})$ </td><td> $7.636 \cdot 10^{-6}$ </td><td> $(2.675 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-13}$ </td><td> $2.188 \cdot 10^{-4}$ </td><td> $(1.171 \cdot 10^{-3})$ </td><td> $3.912 \cdot 10^{-6}$ </td><td> $(4.332 \cdot 10^{-7})$ </td></tr></table>

Table 2: The medians over five trials plotted in Figure 7 on the top right, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td> $d = 4$ </td><td></td><td> $d = 16$ </td><td></td><td> $d = 64$ </td><td></td></tr><tr><td> $4^{2}$ </td><td> $1.728 \cdot 10^{2}$ </td><td> $(8.402 \cdot 10^{0})$ </td><td> $1.372 \cdot 10^{2}$ </td><td> $(4.366 \cdot 10^{0})$ </td><td> $1.154 \cdot 10^{2}$ </td><td> $(1.454 \cdot 10^{0})$ </td></tr><tr><td> $4^{1}$ </td><td> $1.704 \cdot 10^{2}$ </td><td> $(5.072 \cdot 10^{0})$ </td><td> $1.381 \cdot 10^{2}$ </td><td> $(3.195 \cdot 10^{0})$ </td><td> $1.152 \cdot 10^{2}$ </td><td> $(1.564 \cdot 10^{0})$ </td></tr><tr><td> $4^{0}$ </td><td> $1.641 \cdot 10^{2}$ </td><td> $(2.687 \cdot 10^{1})$ </td><td> $1.324 \cdot 10^{2}$ </td><td> $(6.702 \cdot 10^{0})$ </td><td> $1.101 \cdot 10^{2}$ </td><td> $(3.607 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $9.916 \cdot 10^{1}$ </td><td> $(6.601 \cdot 10^{1})$ </td><td> $1.287 \cdot 10^{2}$ </td><td> $(1.970 \cdot 10^{1})$ </td><td> $1.056 \cdot 10^{2}$ </td><td> $(8.201 \cdot 10^{0})$ </td></tr><tr><td> $4^{-2}$ </td><td> $2.210 \cdot 10^{1}$ </td><td> $(6.712 \cdot 10^{1})$ </td><td> $9.046 \cdot 10^{1}$ </td><td> $(2.885 \cdot 10^{1})$ </td><td> $1.028 \cdot 10^{2}$ </td><td> $(3.421 \cdot 10^{1})$ </td></tr><tr><td> $4^{-3}$ </td><td> $5.421 \cdot 10^{0}$ </td><td> $(6.697 \cdot 10^{1})$ </td><td> $3.727 \cdot 10^{1}$ </td><td> $(4.346 \cdot 10^{1})$ </td><td> $7.944 \cdot 10^{1}$ </td><td> $(4.106 \cdot 10^{1})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.350 \cdot 10^{0}$ </td><td> $(6.609 \cdot 10^{1})$ </td><td> $9.250 \cdot 10^{0}$ </td><td> $(5.094 \cdot 10^{1})$ </td><td> $2.129 \cdot 10^{1}$ </td><td> $(2.172 \cdot 10^{1})$ </td></tr><tr><td> $4^{-5}$ </td><td> $3.370 \cdot 10^{-1}$ </td><td> $(6.163 \cdot 10^{1})$ </td><td> $2.287 \cdot 10^{0}$ </td><td> $(3.956 \cdot 10^{1})$ </td><td> $5.368 \cdot 10^{0}$ </td><td> $(6.186 \cdot 10^{0})$ </td></tr><tr><td> $4^{-6}$ </td><td> $8.423 \cdot 10^{-2}$ </td><td> $(6.122 \cdot 10^{1})$ </td><td> $5.699 \cdot 10^{-1}$ </td><td> $(3.996 \cdot 10^{1})$ </td><td> $1.340 \cdot 10^{0}$ </td><td> $(1.562 \cdot 10^{0})$ </td></tr><tr><td> $4^{-7}$ </td><td> $2.106 \cdot 10^{-2}$ </td><td> $(6.065 \cdot 10^{1})$ </td><td> $1.423 \cdot 10^{-1}$ </td><td> $(4.068 \cdot 10^{1})$ </td><td> $3.348 \cdot 10^{-1}$ </td><td> $(3.912 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-8}$ </td><td> $5.264 \cdot 10^{-3}$ </td><td> $(5.959 \cdot 10^{1})$ </td><td> $3.557 \cdot 10^{-2}$ </td><td> $(4.091 \cdot 10^{1})$ </td><td> $8.370 \cdot 10^{-2}$ </td><td> $(9.783 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-9}$ </td><td> $1.316 \cdot 10^{-3}$ </td><td> $(5.841 \cdot 10^{1})$ </td><td> $8.893 \cdot 10^{-3}$ </td><td> $(4.100 \cdot 10^{1})$ </td><td> $2.092 \cdot 10^{-2}$ </td><td> $(2.446 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-10}$ </td><td> $3.290 \cdot 10^{-4}$ </td><td> $(5.722 \cdot 10^{1})$ </td><td> $2.223 \cdot 10^{-3}$ </td><td> $(1.845 \cdot 10^{1})$ </td><td> $5.231 \cdot 10^{-3}$ </td><td> $(6.115 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-11}$ </td><td> $8.225 \cdot 10^{-5}$ </td><td> $(5.606 \cdot 10^{1})$ </td><td> $5.558 \cdot 10^{-4}$ </td><td> $(5.136 \cdot 10^{0})$ </td><td> $1.308 \cdot 10^{-3}$ </td><td> $(1.529 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-12}$ </td><td> $2.056 \cdot 10^{-5}$ </td><td> $(5.495 \cdot 10^{1})$ </td><td> $1.389 \cdot 10^{-4}$ </td><td> $(1.309 \cdot 10^{0})$ </td><td> $3.269 \cdot 10^{-4}$ </td><td> $(3.822 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-13}$ </td><td> $5.123 \cdot 10^{-6}$ </td><td> $(5.389 \cdot 10^{1})$ </td><td> $3.473 \cdot 10^{-5}$ </td><td> $(3.285 \cdot 10^{-1})$ </td><td> $8.173 \cdot 10^{-5}$ </td><td> $(9.554 \cdot 10^{-5})$ </td></tr></table>

<table><tr><td> $\lambda$ </td><td colspan="2"> $d = 256$ </td><td colspan="2"> $d = 1024$ </td></tr><tr><td> $4^{2}$ </td><td> $1.025 \cdot 10^{2}$ </td><td> $(1.676 \cdot 10^{0})$ </td><td> $9.693 \cdot 10^{1}$ </td><td> $(1.513 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $1.022 \cdot 10^{2}$ </td><td> $(1.122 \cdot 10^{0})$ </td><td> $9.623 \cdot 10^{1}$ </td><td> $(4.889 \cdot 10^{-1})$ </td></tr><tr><td> $4^{0}$ </td><td> $1.004 \cdot 10^{2}$ </td><td> $(2.333 \cdot 10^{0})$ </td><td> $9.507 \cdot 10^{1}$ </td><td> $(6.628 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-1}$ </td><td> $9.800 \cdot 10^{1}$ </td><td> $(1.214 \cdot 10^{1})$ </td><td> $9.253 \cdot 10^{1}$ </td><td> $(6.296 \cdot 10^{0})$ </td></tr><tr><td> $4^{-2}$ </td><td> $9.213 \cdot 10^{1}$ </td><td> $(3.434 \cdot 10^{1})$ </td><td> $9.143 \cdot 10^{1}$ </td><td> $(2.830 \cdot 10^{1})$ </td></tr><tr><td> $4^{-3}$ </td><td> $7.404 \cdot 10^{1}$ </td><td> $(4.140 \cdot 10^{1})$ </td><td> $9.113 \cdot 10^{1}$ </td><td> $(4.401 \cdot 10^{1})$ </td></tr><tr><td> $4^{-4}$ </td><td> $2.102 \cdot 10^{1}$ </td><td> $(3.818 \cdot 10^{1})$ </td><td> $9.031 \cdot 10^{1}$ </td><td> $(4.837 \cdot 10^{1})$ </td></tr><tr><td> $4^{-5}$ </td><td> $5.305 \cdot 10^{0}$ </td><td> $(4.104 \cdot 10^{1})$ </td><td> $4.268 \cdot 10^{1}$ </td><td> $(4.573 \cdot 10^{1})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.327 \cdot 10^{0}$ </td><td> $(4.139 \cdot 10^{1})$ </td><td> $1.146 \cdot 10^{1}$ </td><td> $(3.987 \cdot 10^{1})$ </td></tr><tr><td> $4^{-7}$ </td><td> $3.316 \cdot 10^{-1}$ </td><td> $(4.075 \cdot 10^{1})$ </td><td> $2.875 \cdot 10^{0}$ </td><td> $(4.045 \cdot 10^{1})$ </td></tr><tr><td> $4^{-8}$ </td><td> $8.290 \cdot 10^{-2}$ </td><td> $(4.008 \cdot 10^{1})$ </td><td> $7.186 \cdot 10^{-1}$ </td><td> $(4.141 \cdot 10^{1})$ </td></tr><tr><td> $4^{-9}$ </td><td> $2.072 \cdot 10^{-2}$ </td><td> $(2.250 \cdot 10^{1})$ </td><td> $1.796 \cdot 10^{-1}$ </td><td> $(4.170 \cdot 10^{1})$ </td></tr><tr><td> $4^{-10}$ </td><td> $5.181 \cdot 10^{-3}$ </td><td> $(6.373 \cdot 10^{0})$ </td><td> $4.491 \cdot 10^{-2}$ </td><td> $(4.178 \cdot 10^{1})$ </td></tr><tr><td> $4^{-11}$ </td><td> $1.295 \cdot 10^{-3}$ </td><td> $(1.614 \cdot 10^{0})$ </td><td> $1.123 \cdot 10^{-2}$ </td><td> $(4.180 \cdot 10^{1})$ </td></tr><tr><td> $4^{-12}$ </td><td> $3.238 \cdot 10^{-4}$ </td><td> $(4.042 \cdot 10^{-1})$ </td><td> $2.807 \cdot 10^{-3}$ </td><td> $(4.172 \cdot 10^{1})$ </td></tr><tr><td> $4^{-13}$ </td><td> $8.099 \cdot 10^{-5}$ </td><td> $(1.011 \cdot 10^{-1})$ </td><td> $7.016 \cdot 10^{-4}$ </td><td> $(2.239 \cdot 10^{1})$ </td></tr></table>

Table 3: The medians over five trials plotted in Figure 7 on the middle left, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td> $d = 4$ </td><td></td><td> $d = 16$ </td><td></td><td> $d = 64$ </td><td></td></tr><tr><td> $4^{2}$ </td><td> $7.683 \cdot 10^{1}$ </td><td> $(9.817 \cdot 10^{0})$ </td><td> $9.016 \cdot 10^{1}$ </td><td> $(5.402 \cdot 10^{-1})$ </td><td> $9.001 \cdot 10^{1}$ </td><td> $(1.067 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $8.311 \cdot 10^{1}$ </td><td> $(3.064 \cdot 10^{0})$ </td><td> $8.978 \cdot 10^{1}$ </td><td> $(3.437 \cdot 10^{-1})$ </td><td> $8.954 \cdot 10^{1}$ </td><td> $(1.429 \cdot 10^{-1})$ </td></tr><tr><td> $4^{0}$ </td><td> $6.196 \cdot 10^{1}$ </td><td> $(8.026 \cdot 10^{0})$ </td><td> $6.779 \cdot 10^{1}$ </td><td> $(3.166 \cdot 10^{0})$ </td><td> $6.012 \cdot 10^{1}$ </td><td> $(1.436 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $2.831 \cdot 10^{1}$ </td><td> $(8.054 \cdot 10^{0})$ </td><td> $2.917 \cdot 10^{1}$ </td><td> $(2.550 \cdot 10^{0})$ </td><td> $2.170 \cdot 10^{1}$ </td><td> $(2.631 \cdot 10^{0})$ </td></tr><tr><td> $4^{-2}$ </td><td> $1.555 \cdot 10^{1}$ </td><td> $(5.922 \cdot 10^{0})$ </td><td> $1.125 \cdot 10^{1}$ </td><td> $(1.727 \cdot 10^{0})$ </td><td> $7.769 \cdot 10^{0}$ </td><td> $(2.285 \cdot 10^{0})$ </td></tr><tr><td> $4^{-3}$ </td><td> $6.157 \cdot 10^{0}$ </td><td> $(4.770 \cdot 10^{0})$ </td><td> $4.808 \cdot 10^{0}$ </td><td> $(1.412 \cdot 10^{0})$ </td><td> $2.509 \cdot 10^{0}$ </td><td> $(1.719 \cdot 10^{0})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.966 \cdot 10^{0}$ </td><td> $(3.649 \cdot 10^{0})$ </td><td> $2.203 \cdot 10^{0}$ </td><td> $(1.188 \cdot 10^{0})$ </td><td> $6.430 \cdot 10^{-1}$ </td><td> $(1.313 \cdot 10^{0})$ </td></tr><tr><td> $4^{-5}$ </td><td> $5.171 \cdot 10^{-1}$ </td><td> $(3.181 \cdot 10^{0})$ </td><td> $7.332 \cdot 10^{-1}$ </td><td> $(1.057 \cdot 10^{0})$ </td><td> $1.612 \cdot 10^{-1}$ </td><td> $(1.013 \cdot 10^{0})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.308 \cdot 10^{-1}$ </td><td> $(2.053 \cdot 10^{0})$ </td><td> $1.886 \cdot 10^{-1}$ </td><td> $(6.984 \cdot 10^{-1})$ </td><td> $4.031 \cdot 10^{-2}$ </td><td> $(5.756 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-7}$ </td><td> $3.278 \cdot 10^{-2}$ </td><td> $(1.152 \cdot 10^{0})$ </td><td> $4.741 \cdot 10^{-2}$ </td><td> $(2.505 \cdot 10^{-1})$ </td><td> $1.008 \cdot 10^{-2}$ </td><td> $(1.566 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-8}$ </td><td> $8.199 \cdot 10^{-3}$ </td><td> $(2.638 \cdot 10^{-1})$ </td><td> $1.186 \cdot 10^{-2}$ </td><td> $(6.378 \cdot 10^{-2})$ </td><td> $2.520 \cdot 10^{-3}$ </td><td> $(3.948 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-9}$ </td><td> $2.050 \cdot 10^{-3}$ </td><td> $(6.571 \cdot 10^{-2})$ </td><td> $2.967 \cdot 10^{-3}$ </td><td> $(1.595 \cdot 10^{-2})$ </td><td> $6.299 \cdot 10^{-4}$ </td><td> $(9.880 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-10}$ </td><td> $5.126 \cdot 10^{-4}$ </td><td> $(1.641 \cdot 10^{-2})$ </td><td> $7.417 \cdot 10^{-4}$ </td><td> $(3.992 \cdot 10^{-3})$ </td><td> $1.575 \cdot 10^{-4}$ </td><td> $(2.470 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-11}$ </td><td> $1.282 \cdot 10^{-4}$ </td><td> $(4.101 \cdot 10^{-3})$ </td><td> $1.854 \cdot 10^{-4}$ </td><td> $(9.984 \cdot 10^{-4})$ </td><td> $3.938 \cdot 10^{-5}$ </td><td> $(6.175 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-12}$ </td><td> $3.204 \cdot 10^{-5}$ </td><td> $(1.025 \cdot 10^{-3})$ </td><td> $4.635 \cdot 10^{-5}$ </td><td> $(2.496 \cdot 10^{-4})$ </td><td> $9.849 \cdot 10^{-6}$ </td><td> $(1.544 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-13}$ </td><td> $7.973 \cdot 10^{-6}$ </td><td> $(2.562 \cdot 10^{-4})$ </td><td> $1.161 \cdot 10^{-5}$ </td><td> $(6.245 \cdot 10^{-5})$ </td><td> $2.332 \cdot 10^{-6}$ </td><td> $(3.861 \cdot 10^{-5})$ </td></tr></table>

<table><tr><td> $\lambda$ </td><td colspan="2"> $d = 256$ </td><td colspan="2"> $d = 1024$ </td></tr><tr><td> $4^{2}$ </td><td> $9.011 \cdot 10^{1}$ </td><td> $(3.716 \cdot 10^{-2})$ </td><td> $9.013 \cdot 10^{1}$ </td><td> $(1.563 \cdot 10^{-2})$ </td></tr><tr><td> $4^{1}$ </td><td> $8.965 \cdot 10^{1}$ </td><td> $(7.506 \cdot 10^{-2})$ </td><td> $8.960 \cdot 10^{1}$ </td><td> $(9.857 \cdot 10^{-2})$ </td></tr><tr><td> $4^{0}$ </td><td> $5.760 \cdot 10^{1}$ </td><td> $(1.833 \cdot 10^{0})$ </td><td> $5.427 \cdot 10^{1}$ </td><td> $(1.127 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.952 \cdot 10^{1}$ </td><td> $(2.337 \cdot 10^{0})$ </td><td> $1.494 \cdot 10^{1}$ </td><td> $(4.246 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-2}$ </td><td> $7.718 \cdot 10^{0}$ </td><td> $(2.352 \cdot 10^{0})$ </td><td> $3.758 \cdot 10^{0}$ </td><td> $(1.110 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-3}$ </td><td> $3.520 \cdot 10^{0}$ </td><td> $(2.141 \cdot 10^{0})$ </td><td> $9.400 \cdot 10^{-1}$ </td><td> $(2.784 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.450 \cdot 10^{0}$ </td><td> $(1.730 \cdot 10^{0})$ </td><td> $2.350 \cdot 10^{-1}$ </td><td> $(6.962 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-5}$ </td><td> $3.816 \cdot 10^{-1}$ </td><td> $(1.407 \cdot 10^{0})$ </td><td> $5.875 \cdot 10^{-2}$ </td><td> $(1.740 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-6}$ </td><td> $9.573 \cdot 10^{-2}$ </td><td> $(4.887 \cdot 10^{-1})$ </td><td> $1.469 \cdot 10^{-2}$ </td><td> $(4.351 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-7}$ </td><td> $2.394 \cdot 10^{-2}$ </td><td> $(1.254 \cdot 10^{-1})$ </td><td> $3.672 \cdot 10^{-3}$ </td><td> $(1.088 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-8}$ </td><td> $5.985 \cdot 10^{-3}$ </td><td> $(3.140 \cdot 10^{-2})$ </td><td> $9.180 \cdot 10^{-4}$ </td><td> $(2.719 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-9}$ </td><td> $1.496 \cdot 10^{-3}$ </td><td> $(7.851 \cdot 10^{-3})$ </td><td> $2.295 \cdot 10^{-4}$ </td><td> $(6.797 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-10}$ </td><td> $3.741 \cdot 10^{-4}$ </td><td> $(1.963 \cdot 10^{-3})$ </td><td> $5.739 \cdot 10^{-5}$ </td><td> $(1.696 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-11}$ </td><td> $9.352 \cdot 10^{-5}$ </td><td> $(4.907 \cdot 10^{-4})$ </td><td> $1.434 \cdot 10^{-5}$ </td><td> $(4.200 \cdot 10^{-7})$ </td></tr><tr><td> $4^{-12}$ </td><td> $2.343 \cdot 10^{-5}$ </td><td> $(1.227 \cdot 10^{-4})$ </td><td> $3.686 \cdot 10^{-6}$ </td><td> $(1.467 \cdot 10^{-7})$ </td></tr><tr><td> $4^{-13}$ </td><td> $5.816 \cdot 10^{-6}$ </td><td> $(3.058 \cdot 10^{-5})$ </td><td> $9.589 \cdot 10^{-7}$ </td><td> $(9.762 \cdot 10^{-8})$ </td></tr></table>

Table 4: The medians over five trials plotted in Figure 7 on the middle right, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td> $d = 4$ </td><td></td><td> $d = 16$ </td><td></td><td> $d = 64$ </td><td></td></tr><tr><td> $4^{2}$ </td><td> $8.693 \cdot 10^{1}$ </td><td> $(6.307 \cdot 10^{0})$ </td><td> $9.007 \cdot 10^{1}$ </td><td> $(1.912 \cdot 10^{-1})$ </td><td> $9.020 \cdot 10^{1}$ </td><td> $(1.051 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $8.332 \cdot 10^{1}$ </td><td> $(3.578 \cdot 10^{0})$ </td><td> $8.919 \cdot 10^{1}$ </td><td> $(3.286 \cdot 10^{-1})$ </td><td> $8.965 \cdot 10^{1}$ </td><td> $(1.669 \cdot 10^{-1})$ </td></tr><tr><td> $4^{0}$ </td><td> $5.427 \cdot 10^{1}$ </td><td> $(1.395 \cdot 10^{1})$ </td><td> $6.270 \cdot 10^{1}$ </td><td> $(2.967 \cdot 10^{0})$ </td><td> $5.755 \cdot 10^{1}$ </td><td> $(2.259 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.690 \cdot 10^{1}$ </td><td> $(1.879 \cdot 10^{1})$ </td><td> $2.528 \cdot 10^{1}$ </td><td> $(3.054 \cdot 10^{0})$ </td><td> $1.834 \cdot 10^{1}$ </td><td> $(2.899 \cdot 10^{0})$ </td></tr><tr><td> $4^{-2}$ </td><td> $4.138 \cdot 10^{0}$ </td><td> $(1.715 \cdot 10^{1})$ </td><td> $8.632 \cdot 10^{0}$ </td><td> $(1.621 \cdot 10^{0})$ </td><td> $6.326 \cdot 10^{0}$ </td><td> $(2.343 \cdot 10^{0})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.041 \cdot 10^{0}$ </td><td> $(1.471 \cdot 10^{1})$ </td><td> $2.621 \cdot 10^{0}$ </td><td> $(1.206 \cdot 10^{0})$ </td><td> $2.697 \cdot 10^{0}$ </td><td> $(1.608 \cdot 10^{0})$ </td></tr><tr><td> $4^{-4}$ </td><td> $2.606 \cdot 10^{-1}$ </td><td> $(1.387 \cdot 10^{1})$ </td><td> $6.547 \cdot 10^{-1}$ </td><td> $(1.258 \cdot 10^{0})$ </td><td> $7.072 \cdot 10^{-1}$ </td><td> $(6.715 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-5}$ </td><td> $6.517 \cdot 10^{-2}$ </td><td> $(1.357 \cdot 10^{1})$ </td><td> $1.632 \cdot 10^{-1}$ </td><td> $(8.705 \cdot 10^{-1})$ </td><td> $1.778 \cdot 10^{-1}$ </td><td> $(1.836 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.630 \cdot 10^{-2}$ </td><td> $(1.337 \cdot 10^{1})$ </td><td> $4.075 \cdot 10^{-2}$ </td><td> $(8.647 \cdot 10^{-1})$ </td><td> $4.439 \cdot 10^{-2}$ </td><td> $(4.626 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-7}$ </td><td> $4.074 \cdot 10^{-3}$ </td><td> $(1.314 \cdot 10^{1})$ </td><td> $1.018 \cdot 10^{-2}$ </td><td> $(8.829 \cdot 10^{-1})$ </td><td> $1.110 \cdot 10^{-2}$ </td><td> $(1.158 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-8}$ </td><td> $1.018 \cdot 10^{-3}$ </td><td> $(1.291 \cdot 10^{1})$ </td><td> $2.546 \cdot 10^{-3}$ </td><td> $(8.890 \cdot 10^{-1})$ </td><td> $2.774 \cdot 10^{-3}$ </td><td> $(2.897 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-9}$ </td><td> $2.546 \cdot 10^{-4}$ </td><td> $(1.191 \cdot 10^{1})$ </td><td> $6.364 \cdot 10^{-4}$ </td><td> $(8.913 \cdot 10^{-1})$ </td><td> $6.935 \cdot 10^{-4}$ </td><td> $(7.243 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-10}$ </td><td> $6.366 \cdot 10^{-5}$ </td><td> $(1.093 \cdot 10^{1})$ </td><td> $1.591 \cdot 10^{-4}$ </td><td> $(4.011 \cdot 10^{-1})$ </td><td> $1.734 \cdot 10^{-4}$ </td><td> $(1.811 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-11}$ </td><td> $1.592 \cdot 10^{-5}$ </td><td> $(9.224 \cdot 10^{0})$ </td><td> $3.978 \cdot 10^{-5}$ </td><td> $(1.116 \cdot 10^{-1})$ </td><td> $4.334 \cdot 10^{-5}$ </td><td> $(4.527 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-12}$ </td><td> $3.972 \cdot 10^{-6}$ </td><td> $(8.167 \cdot 10^{0})$ </td><td> $9.970 \cdot 10^{-6}$ </td><td> $(2.845 \cdot 10^{-2})$ </td><td> $1.083 \cdot 10^{-5}$ </td><td> $(1.132 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-13}$ </td><td> $1.002 \cdot 10^{-6}$ </td><td> $(7.210 \cdot 10^{0})$ </td><td> $2.308 \cdot 10^{-6}$ </td><td> $(7.141 \cdot 10^{-3})$ </td><td> $2.737 \cdot 10^{-6}$ </td><td> $(2.805 \cdot 10^{-6})$ </td></tr></table>

<table><tr><td> $\lambda$ </td><td colspan="2"> $d = 256$ </td><td colspan="2"> $d = 1024$ </td></tr><tr><td> $4^{2}$ </td><td> $9.008 \cdot 10^{1}$ </td><td> $(5.085 \cdot 10^{-2})$ </td><td> $9.012 \cdot 10^{1}$ </td><td> $(2.708 \cdot 10^{-2})$ </td></tr><tr><td> $4^{1}$ </td><td> $8.965 \cdot 10^{1}$ </td><td> $(9.344 \cdot 10^{-2})$ </td><td> $8.973 \cdot 10^{1}$ </td><td> $(8.384 \cdot 10^{-2})$ </td></tr><tr><td> $4^{0}$ </td><td> $5.698 \cdot 10^{1}$ </td><td> $(1.686 \cdot 10^{0})$ </td><td> $5.645 \cdot 10^{1}$ </td><td> $(1.837 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.935 \cdot 10^{1}$ </td><td> $(3.007 \cdot 10^{0})$ </td><td> $1.855 \cdot 10^{1}$ </td><td> $(2.099 \cdot 10^{0})$ </td></tr><tr><td> $4^{-2}$ </td><td> $6.903 \cdot 10^{0}$ </td><td> $(2.854 \cdot 10^{0})$ </td><td> $6.111 \cdot 10^{0}$ </td><td> $(2.381 \cdot 10^{0})$ </td></tr><tr><td> $4^{-3}$ </td><td> $2.820 \cdot 10^{0}$ </td><td> $(1.859 \cdot 10^{0})$ </td><td> $2.766 \cdot 10^{0}$ </td><td> $(2.318 \cdot 10^{0})$ </td></tr><tr><td> $4^{-4}$ </td><td> $7.530 \cdot 10^{-1}$ </td><td> $(1.521 \cdot 10^{0})$ </td><td> $1.922 \cdot 10^{0}$ </td><td> $(2.149 \cdot 10^{0})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.893 \cdot 10^{-1}$ </td><td> $(1.465 \cdot 10^{0})$ </td><td> $1.066 \cdot 10^{0}$ </td><td> $(1.672 \cdot 10^{0})$ </td></tr><tr><td> $4^{-6}$ </td><td> $4.733 \cdot 10^{-2}$ </td><td> $(1.442 \cdot 10^{0})$ </td><td> $2.823 \cdot 10^{-1}$ </td><td> $(1.218 \cdot 10^{0})$ </td></tr><tr><td> $4^{-7}$ </td><td> $1.183 \cdot 10^{-2}$ </td><td> $(1.117 \cdot 10^{0})$ </td><td> $7.077 \cdot 10^{-2}$ </td><td> $(9.110 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-8}$ </td><td> $2.958 \cdot 10^{-3}$ </td><td> $(8.501 \cdot 10^{-1})$ </td><td> $1.769 \cdot 10^{-2}$ </td><td> $(8.302 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-9}$ </td><td> $7.394 \cdot 10^{-4}$ </td><td> $(4.485 \cdot 10^{-1})$ </td><td> $4.422 \cdot 10^{-3}$ </td><td> $(8.104 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-10}$ </td><td> $1.849 \cdot 10^{-4}$ </td><td> $(1.261 \cdot 10^{-1})$ </td><td> $1.106 \cdot 10^{-3}$ </td><td> $(8.055 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-11}$ </td><td> $4.622 \cdot 10^{-5}$ </td><td> $(3.192 \cdot 10^{-2})$ </td><td> $2.764 \cdot 10^{-4}$ </td><td> $(8.043 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-12}$ </td><td> $1.156 \cdot 10^{-5}$ </td><td> $(7.995 \cdot 10^{-3})$ </td><td> $6.910 \cdot 10^{-5}$ </td><td> $(8.025 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-13}$ </td><td> $2.938 \cdot 10^{-6}$ </td><td> $(1.999 \cdot 10^{-3})$ </td><td> $1.718 \cdot 10^{-5}$ </td><td> $(4.307 \cdot 10^{-1})$ </td></tr></table>

Table 5: The medians over five trials plotted in Figure 7 on the bottom left, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td> $d = 4$ </td><td></td><td> $d = 16$ </td><td></td><td> $d = 64$ </td><td></td></tr><tr><td> $4^{2}$ </td><td> $3.288 \cdot 10^{1}$ </td><td> $(1.412 \cdot 10^{0})$ </td><td> $6.314 \cdot 10^{1}$ </td><td> $(5.680 \cdot 10^{-1})$ </td><td> $1.216 \cdot 10^{2}$ </td><td> $(1.296 \cdot 10^{0})$ </td></tr><tr><td> $4^{1}$ </td><td> $7.962 \cdot 10^{0}$ </td><td> $(2.012 \cdot 10^{-1})$ </td><td> $1.588 \cdot 10^{1}$ </td><td> $(1.277 \cdot 10^{-1})$ </td><td> $3.056 \cdot 10^{1}$ </td><td> $(3.182 \cdot 10^{-1})$ </td></tr><tr><td> $4^{0}$ </td><td> $2.512 \cdot 10^{0}$ </td><td> $(7.491 \cdot 10^{-2})$ </td><td> $4.628 \cdot 10^{0}$ </td><td> $(4.922 \cdot 10^{-2})$ </td><td> $8.342 \cdot 10^{0}$ </td><td> $(7.887 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.380 \cdot 10^{0}$ </td><td> $(2.005 \cdot 10^{-2})$ </td><td> $1.929 \cdot 10^{0}$ </td><td> $(1.723 \cdot 10^{-2})$ </td><td> $2.849 \cdot 10^{0}$ </td><td> $(2.055 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-2}$ </td><td> $1.094 \cdot 10^{0}$ </td><td> $(7.294 \cdot 10^{-3})$ </td><td> $1.231 \cdot 10^{0}$ </td><td> $(7.694 \cdot 10^{-3})$ </td><td> $1.463 \cdot 10^{0}$ </td><td> $(5.370 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.023 \cdot 10^{0}$ </td><td> $(2.200 \cdot 10^{-3})$ </td><td> $1.057 \cdot 10^{0}$ </td><td> $(1.075 \cdot 10^{-3})$ </td><td> $1.116 \cdot 10^{0}$ </td><td> $(1.341 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.006 \cdot 10^{0}$ </td><td> $(9.145 \cdot 10^{-4})$ </td><td> $1.014 \cdot 10^{0}$ </td><td> $(5.712 \cdot 10^{-4})$ </td><td> $1.029 \cdot 10^{0}$ </td><td> $(3.568 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(6.534 \cdot 10^{-4})$ </td><td> $1.003 \cdot 10^{0}$ </td><td> $(5.127 \cdot 10^{-4})$ </td><td> $1.007 \cdot 10^{0}$ </td><td> $(1.379 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(6.009 \cdot 10^{-4})$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(5.074 \cdot 10^{-4})$ </td><td> $1.002 \cdot 10^{0}$ </td><td> $(9.865 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-7}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.887 \cdot 10^{-4})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.070 \cdot 10^{-4})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(9.271 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-8}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.857 \cdot 10^{-4})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.070 \cdot 10^{-4})$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(9.154 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-9}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.850 \cdot 10^{-4})$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(5.070 \cdot 10^{-4})$ </td><td> $9.998 \cdot 10^{-1}$ </td><td> $(9.127 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-10}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.848 \cdot 10^{-4})$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(5.070 \cdot 10^{-4})$ </td><td> $9.998 \cdot 10^{-1}$ </td><td> $(9.121 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-11}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.848 \cdot 10^{-4})$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(5.070 \cdot 10^{-4})$ </td><td> $9.998 \cdot 10^{-1}$ </td><td> $(9.119 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-12}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.848 \cdot 10^{-4})$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(5.070 \cdot 10^{-4})$ </td><td> $9.998 \cdot 10^{-1}$ </td><td> $(9.138 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-13}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(5.848 \cdot 10^{-4})$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(5.070 \cdot 10^{-4})$ </td><td> $9.998 \cdot 10^{-1}$ </td><td> $(9.18 \cdot 10^{-5})$ </td></tr></table>

<table><tr><td> $\lambda$ </td><td> $d = 256$ </td><td></td><td> $d = 1024$ </td><td></td></tr><tr><td> $4^{2}$ </td><td> $2.007 \cdot 10^{2}$ </td><td> $(7.876 \cdot 10^{-1})$ </td><td> $2.205 \cdot 10^{2}$ </td><td> $(2.389 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $5.038 \cdot 10^{1}$ </td><td> $(2.024 \cdot 10^{-1})$ </td><td> $5.534 \cdot 10^{1}$ </td><td> $(5.103 \cdot 10^{-2})$ </td></tr><tr><td> $4^{0}$ </td><td> $1.334 \cdot 10^{1}$ </td><td> $(5.002 \cdot 10^{-2})$ </td><td> $1.461 \cdot 10^{1}$ </td><td> $(1.094 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-1}$ </td><td> $4.098 \cdot 10^{0}$ </td><td> $(1.253 \cdot 10^{-2})$ </td><td> $4.421 \cdot 10^{0}$ </td><td> $(4.102 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-2}$ </td><td> $1.776 \cdot 10^{0}$ </td><td> $(3.138 \cdot 10^{-3})$ </td><td> $1.856 \cdot 10^{0}$ </td><td> $(1.132 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.194 \cdot 10^{0}$ </td><td> $(7.825 \cdot 10^{-4})$ </td><td> $1.214 \cdot 10^{0}$ </td><td> $(2.891 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.048 \cdot 10^{0}$ </td><td> $(1.923 \cdot 10^{-4})$ </td><td> $1.054 \cdot 10^{0}$ </td><td> $(7.164 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.012 \cdot 10^{0}$ </td><td> $(4.655 \cdot 10^{-5})$ </td><td> $1.013 \cdot 10^{0}$ </td><td> $(1.663 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.003 \cdot 10^{0}$ </td><td> $(1.634 \cdot 10^{-5})$ </td><td> $1.003 \cdot 10^{0}$ </td><td> $(2.939 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-7}$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(1.497 \cdot 10^{-5})$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(1.031 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-8}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.550 \cdot 10^{-5})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.699 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-9}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.568 \cdot 10^{-5})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.891 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-10}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(1.573 \cdot 10^{-5})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.940 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-11}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(1.574 \cdot 10^{-5})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.952 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-12}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(1.574 \cdot 10^{-5})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.955 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-13}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(1.574 \cdot 10^{-5})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.956 \cdot 10^{-6})$ </td></tr></table>

Table 6: The medians over five trials plotted in Figure 7 on the bottom right, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td> $d = 4$ </td><td></td><td> $d = 16$ </td><td></td><td> $d = 64$ </td><td></td></tr><tr><td> $4^{2}$ </td><td> $3.215 \cdot 10^{1}$ </td><td> $(1.355 \cdot 10^{0})$ </td><td> $6.310 \cdot 10^{1}$ </td><td> $(5.531 \cdot 10^{-1})$ </td><td> $1.228 \cdot 10^{2}$ </td><td> $(7.578 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $7.710 \cdot 10^{0}$ </td><td> $(2.875 \cdot 10^{-1})$ </td><td> $1.585 \cdot 10^{1}$ </td><td> $(1.381 \cdot 10^{-1})$ </td><td> $3.081 \cdot 10^{1}$ </td><td> $(1.894 \cdot 10^{-1})$ </td></tr><tr><td> $4^{0}$ </td><td> $2.495 \cdot 10^{0}$ </td><td> $(6.191 \cdot 10^{-2})$ </td><td> $4.659 \cdot 10^{0}$ </td><td> $(5.141 \cdot 10^{-2})$ </td><td> $8.493 \cdot 10^{0}$ </td><td> $(4.228 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.375 \cdot 10^{0}$ </td><td> $(2.960 \cdot 10^{-2})$ </td><td> $1.941 \cdot 10^{0}$ </td><td> $(2.509 \cdot 10^{-2})$ </td><td> $2.907 \cdot 10^{0}$ </td><td> $(3.309 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-2}$ </td><td> $1.093 \cdot 10^{0}$ </td><td> $(4.173 \cdot 10^{-3})$ </td><td> $1.237 \cdot 10^{0}$ </td><td> $(8.799 \cdot 10^{-3})$ </td><td> $1.484 \cdot 10^{0}$ </td><td> $(5.647 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.023 \cdot 10^{0}$ </td><td> $(1.006 \cdot 10^{-3})$ </td><td> $1.058 \cdot 10^{0}$ </td><td> $(1.431 \cdot 10^{-3})$ </td><td> $1.118 \cdot 10^{0}$ </td><td> $(4.962 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.005 \cdot 10^{0}$ </td><td> $(5.226 \cdot 10^{-4})$ </td><td> $1.014 \cdot 10^{0}$ </td><td> $(8.382 \cdot 10^{-4})$ </td><td> $1.028 \cdot 10^{0}$ </td><td> $(5.080 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(4.275 \cdot 10^{-4})$ </td><td> $1.003 \cdot 10^{0}$ </td><td> $(7.240 \cdot 10^{-4})$ </td><td> $1.006 \cdot 10^{0}$ </td><td> $(5.213 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(4.136 \cdot 10^{-4})$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(7.028 \cdot 10^{-4})$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(5.204 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-7}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(4.106 \cdot 10^{-4})$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(6.978 \cdot 10^{-4})$ </td><td> $9.991 \cdot 10^{-1}$ </td><td> $(5.201 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-8}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(4.099 \cdot 10^{-4})$ </td><td> $9.998 \cdot 10^{-1}$ </td><td> $(6.965 \cdot 10^{-4})$ </td><td> $9.988 \cdot 10^{-1}$ </td><td> $(5.201 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-9}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(4.097 \cdot 10^{-4})$ </td><td> $9.997 \cdot 10^{-1}$ </td><td> $(6.962 \cdot 10^{-4})$ </td><td> $9.987 \cdot 10^{-1}$ </td><td> $(5.201 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-10}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(4.097 \cdot 10^{-4})$ </td><td> $9.997 \cdot 10^{-1}$ </td><td> $(6.961 \cdot 10^{-4})$ </td><td> $9.987 \cdot 10^{-1}$ </td><td> $(5.201 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-11}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(4.097 \cdot 10^{-4})$ </td><td> $9.997 \cdot 10^{-1}$ </td><td> $(6.961 \cdot 10^{-4})$ </td><td> $9.987 \cdot 10^{-1}$ </td><td> $(5.2O1 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-12}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(4.096 \cdot 10^{-4})$ </td><td> $9.997 \cdot 10^{-1}$ </td><td> $(6.961 \cdot 10^{-4})$ </td><td> $9.987 \cdot 10^{-1}$ </td><td> $(5.2O1 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-13}$ </td><td> $9.999 \cdot 10^{-1}$ </td><td> $(4.096 \cdot 10^{-4})$ </td><td> $9.997 \cdot 10^{-1}$ </td><td> $(6.961 \cdot 10^{-4})$ </td><td> $9.987 \cdot 10^{-1}$ </td><td> $(5.201 \cdot 10^{-3})$ </td></tr></table>

<table><tr><td> $\lambda$ </td><td colspan="2"> $d = 256$ </td><td colspan="2"> $d = 1024$ </td></tr><tr><td> $4^{2}$ </td><td> $1.999 \cdot 10^{2}$ </td><td> $(9.674 \cdot 10^{-1})$ </td><td> $2.203 \cdot 10^{2}$ </td><td> $(2.081 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $5.019 \cdot 10^{1}$ </td><td> $(2.479 \cdot 10^{-1})$ </td><td> $5.530 \cdot 10^{1}$ </td><td> $(5.434 \cdot 10^{-2})$ </td></tr><tr><td> $4^{0}$ </td><td> $1.340 \cdot 10^{1}$ </td><td> $(7.287 \cdot 10^{-2})$ </td><td> $1.469 \cdot 10^{1}$ </td><td> $(2.460 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-1}$ </td><td> $4.185 \cdot 10^{0}$ </td><td> $(3.805 \cdot 10^{-2})$ </td><td> $4.507 \cdot 10^{0}$ </td><td> $(2.189 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-2}$ </td><td> $1.800 \cdot 10^{0}$ </td><td> $(1.273 \cdot 10^{-2})$ </td><td> $1.872 \cdot 10^{0}$ </td><td> $(5.661 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.195 \cdot 10^{0}$ </td><td> $(2.320 \cdot 10^{-3})$ </td><td> $1.213 \cdot 10^{0}$ </td><td> $(5.792 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.047 \cdot 10^{0}$ </td><td> $(8.068 \cdot 10^{-4})$ </td><td> $1.049 \cdot 10^{0}$ </td><td> $(7.904 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.010 \cdot 10^{0}$ </td><td> $(7.644 \cdot 10^{-4})$ </td><td> $1.009 \cdot 10^{0}$ </td><td> $(9.884 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(7.698 \cdot 10^{-4})$ </td><td> $9.984 \cdot 10^{-1}$ </td><td> $(1.029 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-7}$ </td><td> $9.981 \cdot 10^{-1}$ </td><td> $(7.690 \cdot 10^{-4})$ </td><td> $9.959 \cdot 10^{-1}$ </td><td> $(1.036 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-8}$ </td><td> $9.975 \cdot 10^{-1}$ </td><td> $(7.696 \cdot 10^{-4})$ </td><td> $9.952 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-9}$ </td><td> $9.974 \cdot 10^{-1}$ </td><td> $(7.696 \cdot 10^{-4})$ </td><td> $9.951 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-10}$ </td><td> $9.973 \cdot 10^{-1}$ </td><td> $(7.696 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-11}$ </td><td> $9.973 \cdot 10^{-1}$ </td><td> $(7.696 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-12}$ </td><td> $9.973 \cdot 10^{-1}$ </td><td> $(7.696 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-13}$ </td><td> $9.973 \cdot 10^{-1}$ </td><td> $(7.696 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr></table>

Table 7: The medians over five trials plotted in Figure 8 on the top left, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td colspan="2"> $m = 25$ </td><td colspan="2"> $m = 50$ </td><td colspan="2"> $m = 200$ </td></tr><tr><td> $4^{2}$ </td><td> $9.575 \cdot 10^{1}$ </td><td> $(1.097 \cdot 10^{0})$ </td><td> $9.718 \cdot 10^{1}$ </td><td> $(1.086 \cdot 10^{0})$ </td><td> $9.654 \cdot 10^{1}$ </td><td> $(2.816 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $9.713 \cdot 10^{1}$ </td><td> $(1.851 \cdot 10^{0})$ </td><td> $9.604 \cdot 10^{1}$ </td><td> $(7.676 \cdot 10^{-1})$ </td><td> $9.650 \cdot 10^{1}$ </td><td> $(4.394 \cdot 10^{-1})$ </td></tr><tr><td> $4^{0}$ </td><td> $5.524 \cdot 10^{1}$ </td><td> $(1.869 \cdot 10^{1})$ </td><td> $7.969 \cdot 10^{1}$ </td><td> $(1.272 \cdot 10^{1})$ </td><td> $8.695 \cdot 10^{1}$ </td><td> $(5.651 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.491 \cdot 10^{1}$ </td><td> $(8.641 \cdot 10^{0})$ </td><td> $2.501 \cdot 10^{1}$ </td><td> $(1.831 \cdot 10^{1})$ </td><td> $2.997 \cdot 10^{1}$ </td><td> $(1.003 \cdot 10^{1})$ </td></tr><tr><td> $4^{-2}$ </td><td> $3.746 \cdot 10^{0}$ </td><td> $(2.267 \cdot 10^{0})$ </td><td> $6.351 \cdot 10^{0}$ </td><td> $(5.363 \cdot 10^{0})$ </td><td> $7.662 \cdot 10^{0}$ </td><td> $(2.820 \cdot 10^{0})$ </td></tr><tr><td> $4^{-3}$ </td><td> $9.367 \cdot 10^{-1}$ </td><td> $(5.693 \cdot 10^{-1})$ </td><td> $1.590 \cdot 10^{0}$ </td><td> $(1.361 \cdot 10^{0})$ </td><td> $1.918 \cdot 10^{0}$ </td><td> $(7.110 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-4}$ </td><td> $2.342 \cdot 10^{-1}$ </td><td> $(1.424 \cdot 10^{-1})$ </td><td> $3.974 \cdot 10^{-1}$ </td><td> $(3.405 \cdot 10^{-1})$ </td><td> $4.796 \cdot 10^{-1}$ </td><td> $(1.779 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-5}$ </td><td> $5.854 \cdot 10^{-2}$ </td><td> $(3.559 \cdot 10^{-2})$ </td><td> $9.935 \cdot 10^{-2}$ </td><td> $(8.513 \cdot 10^{-2})$ </td><td> $1.199 \cdot 10^{-1}$ </td><td> $(4.447 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.464 \cdot 10^{-2}$ </td><td> $(8.898 \cdot 10^{-3})$ </td><td> $2.484 \cdot 10^{-2}$ </td><td> $(2.128 \cdot 10^{-2})$ </td><td> $2.998 \cdot 10^{-2}$ </td><td> $(1.112 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-7}$ </td><td> $3.659 \cdot 10^{-3}$ </td><td> $(2.225 \cdot 10^{-3})$ </td><td> $6.210 \cdot 10^{-3}$ </td><td> $(5.321 \cdot 10^{-3})$ </td><td> $7.494 \cdot 10^{-3}$ </td><td> $(2.779 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-8}$ </td><td> $9.147 \cdot 10^{-4}$ </td><td> $(5.561 \cdot 10^{-4})$ </td><td> $1.552 \cdot 10^{-3}$ </td><td> $(1.330 \cdot 10^{-3})$ </td><td> $1.874 \cdot 10^{-3}$ </td><td> $(6.948 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-9}$ </td><td> $2.287 \cdot 10^{-4}$ </td><td> $(1.390 \cdot 10^{-4})$ </td><td> $3.881 \cdot 10^{-4}$ </td><td> $(3.325 \cdot 10^{-4})$ </td><td> $4.684 \cdot 10^{-4}$ </td><td> $(1.737 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-10}$ </td><td> $5.721 \cdot 10^{-5}$ </td><td> $(3.476 \cdot 10^{-5})$ </td><td> $9.702 \cdot 10^{-5}$ </td><td> $(8.313 \cdot 10^{-5})$ </td><td> $1.171 \cdot 10^{-4}$ </td><td> $(4.342 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-11}$ </td><td> $1.439 \cdot 10^{-5}$ </td><td> $(8.669 \cdot 10^{-6})$ </td><td> $2.427 \cdot 10^{-5}$ </td><td> $(2.074 \cdot 10^{-5})$ </td><td> $2.930 \cdot 10^{-5}$ </td><td> $(1.086 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-12}$ </td><td> $4.005 \cdot 10^{-6}$ </td><td> $(2.205 \cdot 10^{-6})$ </td><td> $5.533 \cdot 10^{-6}$ </td><td> $(5.167 \cdot 10^{-6})$ </td><td> $7.636 \cdot 10^{-6}$ </td><td> $(2.675 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-13}$ </td><td> $2.958 \cdot 10^{-6}$ </td><td> $(9.800 \cdot 10^{-7})$ </td><td> $3.520 \cdot 10^{-6}$ </td><td> $(8.499 \cdot 10^{-7})$ </td><td> $3.912 \cdot 10^{-6}$ </td><td> $(4.332 \cdot 10^{-7})$ </td></tr></table>

Table 8: The medians over five trials plotted in Figure 8 on the top right, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td colspan="2"> $m = 25$ </td><td colspan="2"> $m = 50$ </td><td colspan="2"> $m = 200$ </td></tr><tr><td> $4^{2}$ </td><td> $9.529 \cdot 10^{1}$ </td><td> $(3.501 \cdot 10^{0})$ </td><td> $1.036 \cdot 10^{2}$ </td><td> $(9.435 \cdot 10^{0})$ </td><td> $9.693 \cdot 10^{1}$ </td><td> $(1.513 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $9.730 \cdot 10^{1}$ </td><td> $(1.043 \cdot 10^{0})$ </td><td> $9.607 \cdot 10^{1}$ </td><td> $(9.753 \cdot 10^{-1})$ </td><td> $9.623 \cdot 10^{1}$ </td><td> $(4.889 \cdot 10^{-1})$ </td></tr><tr><td> $4^{0}$ </td><td> $7.921 \cdot 10^{1}$ </td><td> $(1.510 \cdot 10^{1})$ </td><td> $9.347 \cdot 10^{1}$ </td><td> $(7.892 \cdot 10^{0})$ </td><td> $9.507 \cdot 10^{1}$ </td><td> $(6.628 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-1}$ </td><td> $2.949 \cdot 10^{1}$ </td><td> $(3.891 \cdot 10^{1})$ </td><td> $4.184 \cdot 10^{1}$ </td><td> $(3.133 \cdot 10^{1})$ </td><td> $9.253 \cdot 10^{1}$ </td><td> $(6.296 \cdot 10^{0})$ </td></tr><tr><td> $4^{-2}$ </td><td> $7.645 \cdot 10^{0}$ </td><td> $(3.979 \cdot 10^{1})$ </td><td> $1.114 \cdot 10^{1}$ </td><td> $(2.911 \cdot 10^{1})$ </td><td> $9.143 \cdot 10^{1}$ </td><td> $(2.830 \cdot 10^{1})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.917 \cdot 10^{0}$ </td><td> $(4.028 \cdot 10^{1})$ </td><td> $2.798 \cdot 10^{0}$ </td><td> $(1.139 \cdot 10^{1})$ </td><td> $9.113 \cdot 10^{1}$ </td><td> $(4.401 \cdot 10^{1})$ </td></tr><tr><td> $4^{-4}$ </td><td> $4.796 \cdot 10^{-1}$ </td><td> $(4.148 \cdot 10^{1})$ </td><td> $6.999 \cdot 10^{-1}$ </td><td> $(2.957 \cdot 10^{0})$ </td><td> $9.031 \cdot 10^{1}$ </td><td> $(4.837 \cdot 10^{1})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.199 \cdot 10^{-1}$ </td><td> $(4.186 \cdot 10^{1})$ </td><td> $1.750 \cdot 10^{-1}$ </td><td> $(7.414 \cdot 10^{-1})$ </td><td> $4.268 \cdot 10^{1}$ </td><td> $(4.573 \cdot 10^{1})$ </td></tr><tr><td> $4^{-6}$ </td><td> $2.998 \cdot 10^{-2}$ </td><td> $(4.052 \cdot 10^{1})$ </td><td> $4.375 \cdot 10^{-2}$ </td><td> $(1.854 \cdot 10^{-1})$ </td><td> $1.146 \cdot 10^{1}$ </td><td> $(3.987 \cdot 10^{1})$ </td></tr><tr><td> $4^{-7}$ </td><td> $7.494 \cdot 10^{-3}$ </td><td> $(1.466 \cdot 10^{1})$ </td><td> $1.094 \cdot 10^{-2}$ </td><td> $(4.636 \cdot 10^{-2})$ </td><td> $2.875 \cdot 10^{0}$ </td><td> $(4.045 \cdot 10^{1})$ </td></tr><tr><td> $4^{-8}$ </td><td> $1.874 \cdot 10^{-3}$ </td><td> $(3.832 \cdot 10^{0})$ </td><td> $2.734 \cdot 10^{-3}$ </td><td> $(1.159 \cdot 10^{-2})$ </td><td> $7.186 \cdot 10^{-1}$ </td><td> $(4.141 \cdot 10^{1})$ </td></tr><tr><td> $4^{-9}$ </td><td> $4.684 \cdot 10^{-4}$ </td><td> $(9.530 \cdot 10^{-1})$ </td><td> $6.835 \cdot 10^{-4}$ </td><td> $(2.898 \cdot 10^{-3})$ </td><td> $1.796 \cdot 10^{-1}$ </td><td> $(4.170 \cdot 10^{1})$ </td></tr><tr><td> $4^{-10}$ </td><td> $1.171 \cdot 10^{-4}$ </td><td> $(2.383 \cdot 10^{-1})$ </td><td> $1.709 \cdot 10^{-4}$ </td><td> $(7.244 \cdot 10^{-4})$ </td><td> $4.491 \cdot 10^{-2}$ </td><td> $(4.178 \cdot 10^{1})$ </td></tr><tr><td> $4^{-11}$ </td><td> $2.935 \cdot 10^{-5}$ </td><td> $(5.957 \cdot 10^{-2})$ </td><td> $4.270 \cdot 10^{-5}$ </td><td> $(1.811 \cdot 10^{-4})$ </td><td> $1.123 \cdot 10^{-2}$ </td><td> $(4.180 \cdot 10^{1})$ </td></tr><tr><td> $4^{-12}$ </td><td> $7.295 \cdot 10^{-6}$ </td><td> $(1.489 \cdot 10^{-2})$ </td><td> $1.063 \cdot 10^{-5}$ </td><td> $(4.519 \cdot 10^{-5})$ </td><td> $2.807 \cdot 10^{-3}$ </td><td> $(4.172 \cdot 10^{1})$ </td></tr><tr><td> $4^{-13}$ </td><td> $3.415 \cdot 10^{-6}$ </td><td> $(3.722 \cdot 10^{-3})$ </td><td> $2.700 \cdot 10^{-6}$ </td><td> $(1.115 \cdot 10^{-5})$ </td><td> $7.016 \cdot 10^{-4}$ </td><td> $(2.239 \cdot 10^{1})$ </td></tr></table>

Table 9: The medians over five trials plotted in Figure 8 on the middle left, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td colspan="2">m=25</td><td colspan="2">m=50</td><td colspan="2">m=200</td></tr><tr><td> $4^{2}$ </td><td> $9.013 \cdot 10^{1}$ </td><td> $(2.193 \cdot 10^{-1})$ </td><td> $9.019 \cdot 10^{1}$ </td><td> $(5.887 \cdot 10^{-2})$ </td><td> $9.013 \cdot 10^{1}$ </td><td> $(1.563 \cdot 10^{-2})$ </td></tr><tr><td> $4^{1}$ </td><td> $9.292 \cdot 10^{1}$ </td><td> $(5.592 \cdot 10^{-1})$ </td><td> $9.081 \cdot 10^{1}$ </td><td> $(6.437 \cdot 10^{-2})$ </td><td> $8.960 \cdot 10^{1}$ </td><td> $(9.857 \cdot 10^{-2})$ </td></tr><tr><td> $4^{0}$ </td><td> $4.898 \cdot 10^{1}$ </td><td> $(3.903 \cdot 10^{0})$ </td><td> $5.533 \cdot 10^{1}$ </td><td> $(2.030 \cdot 10^{0})$ </td><td> $5.427 \cdot 10^{1}$ </td><td> $(1.127 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.301 \cdot 10^{1}$ </td><td> $(1.671 \cdot 10^{0})$ </td><td> $1.521 \cdot 10^{1}$ </td><td> $(1.672 \cdot 10^{0})$ </td><td> $1.494 \cdot 10^{1}$ </td><td> $(4.246 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-2}$ </td><td> $3.265 \cdot 10^{0}$ </td><td> $(4.356 \cdot 10^{-1})$ </td><td> $3.826 \cdot 10^{0}$ </td><td> $(4.739 \cdot 10^{-1})$ </td><td> $3.758 \cdot 10^{0}$ </td><td> $(1.110 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-3}$ </td><td> $8.164 \cdot 10^{-1}$ </td><td> $(1.093 \cdot 10^{-1})$ </td><td> $9.569 \cdot 10^{-1}$ </td><td> $(1.199 \cdot 10^{-1})$ </td><td> $9.400 \cdot 10^{-1}$ </td><td> $(2.784 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-4}$ </td><td> $2.041 \cdot 10^{-1}$ </td><td> $(2.734 \cdot 10^{-2})$ </td><td> $2.392 \cdot 10^{-1}$ </td><td> $(3.000 \cdot 10^{-2})$ </td><td> $2.350 \cdot 10^{-1}$ </td><td> $(6.962 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-5}$ </td><td> $5.103 \cdot 10^{-2}$ </td><td> $(6.836 \cdot 10^{-3})$ </td><td> $5.981 \cdot 10^{-2}$ </td><td> $(7.501 \cdot 10^{-3})$ </td><td> $5.875 \cdot 10^{-2}$ </td><td> $(1.740 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.276 \cdot 10^{-2}$ </td><td> $(1.709 \cdot 10^{-3})$ </td><td> $1.495 \cdot 10^{-2}$ </td><td> $(1.875 \cdot 10^{-3})$ </td><td> $1.469 \cdot 10^{-2}$ </td><td> $(4.351 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-7}$ </td><td> $3.189 \cdot 10^{-3}$ </td><td> $(4.272 \cdot 10^{-4})$ </td><td> $3.738 \cdot 10^{-3}$ </td><td> $(4.688 \cdot 10^{-4})$ </td><td> $3.672 \cdot 10^{-3}$ </td><td> $(1.088 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-8}$ </td><td> $7.974 \cdot 10^{-4}$ </td><td> $(1.068 \cdot 10^{-4})$ </td><td> $9.345 \cdot 10^{-4}$ </td><td> $(1.172 \cdot 10^{-4})$ </td><td> $9.180 \cdot 10^{-4}$ </td><td> $(2.719 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-9}$ </td><td> $1.994 \cdot 10^{-4}$ </td><td> $(2.675 \cdot 10^{-5})$ </td><td> $2.336 \cdot 10^{-4}$ </td><td> $(2.929 \cdot 10^{-5})$ </td><td> $2.295 \cdot 10^{-4}$ </td><td> $(6.797 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-10}$ </td><td> $4.998 \cdot 10^{-5}$ </td><td> $(6.686 \cdot 10^{-6})$ </td><td> $5.842 \cdot 10^{-5}$ </td><td> $(7.310 \cdot 10^{-6})$ </td><td> $5.739 \cdot 10^{-5}$ </td><td> $(1.696 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-11}$ </td><td> $1.258 \cdot 10^{-5}$ </td><td> $(1.667 \cdot 10^{-6})$ </td><td> $1.462 \cdot 10^{-5}$ </td><td> $(1.808 \cdot 10^{-6})$ </td><td> $1.434 \cdot 10^{-5}$ </td><td> $(4.200 \cdot 10^{-7})$ </td></tr><tr><td> $4^{-12}$ </td><td> $3.306 \cdot 10^{-6}$ </td><td> $(5.100 \cdot 10^{-7})$ </td><td> $3.694 \cdot 10^{-6}$ </td><td> $(4.799 \cdot 10^{-7})$ </td><td> $3.686 \cdot 10^{-6}$ </td><td> $(1.467 \cdot 10^{-7})$ </td></tr><tr><td> $4^{-13}$ </td><td> $8.538 \cdot 10^{-7}$ </td><td> $(1.928 \cdot 10^{-7})$ </td><td> $1.034 \cdot 10^{-6}$ </td><td> $(3.022 \cdot 10^{-7})$ </td><td> $9.589 \cdot 10^{-7}$ </td><td> $(9.762 \cdot 10^{-8})$ </td></tr></table>

Table 10: The medians over five trials plotted in Figure 8 on the middle right, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td colspan="2">m=25</td><td colspan="2">m=50</td><td colspan="2">m=200</td></tr><tr><td> $4^{2}$ </td><td> $9.000 \cdot 10^{1}$ </td><td> $(2.752 \cdot 10^{-1})$ </td><td> $9.023 \cdot 10^{1}$ </td><td> $(2.088 \cdot 10^{-1})$ </td><td> $9.012 \cdot 10^{1}$ </td><td> $(2.708 \cdot 10^{-2})$ </td></tr><tr><td> $4^{1}$ </td><td> $9.220 \cdot 10^{1}$ </td><td> $(4.438 \cdot 10^{-1})$ </td><td> $9.081 \cdot 10^{1}$ </td><td> $(6.729 \cdot 10^{-2})$ </td><td> $8.973 \cdot 10^{1}$ </td><td> $(8.384 \cdot 10^{-2})$ </td></tr><tr><td> $4^{0}$ </td><td> $6.040 \cdot 10^{1}$ </td><td> $(5.291 \cdot 10^{0})$ </td><td> $5.576 \cdot 10^{1}$ </td><td> $(2.495 \cdot 10^{0})$ </td><td> $5.645 \cdot 10^{1}$ </td><td> $(1.837 \cdot 10^{0})$ </td></tr><tr><td> $4^{-1}$ </td><td> $1.729 \cdot 10^{1}$ </td><td> $(9.637 \cdot 10^{0})$ </td><td> $1.701 \cdot 10^{1}$ </td><td> $(3.629 \cdot 10^{0})$ </td><td> $1.855 \cdot 10^{1}$ </td><td> $(2.099 \cdot 10^{0})$ </td></tr><tr><td> $4^{-2}$ </td><td> $4.364 \cdot 10^{0}$ </td><td> $(6.059 \cdot 10^{0})$ </td><td> $4.306 \cdot 10^{0}$ </td><td> $(3.047 \cdot 10^{0})$ </td><td> $6.111 \cdot 10^{0}$ </td><td> $(2.381 \cdot 10^{0})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.092 \cdot 10^{0}$ </td><td> $(4.761 \cdot 10^{0})$ </td><td> $1.078 \cdot 10^{0}$ </td><td> $(1.174 \cdot 10^{0})$ </td><td> $2.766 \cdot 10^{0}$ </td><td> $(2.318 \cdot 10^{0})$ </td></tr><tr><td> $4^{-4}$ </td><td> $2.730 \cdot 10^{-1}$ </td><td> $(4.860 \cdot 10^{0})$ </td><td> $2.694 \cdot 10^{-1}$ </td><td> $(3.045 \cdot 10^{-1})$ </td><td> $1.922 \cdot 10^{0}$ </td><td> $(2.149 \cdot 10^{0})$ </td></tr><tr><td> $4^{-5}$ </td><td> $6.825 \cdot 10^{-2}$ </td><td> $(4.918 \cdot 10^{0})$ </td><td> $6.735 \cdot 10^{-2}$ </td><td> $(7.634 \cdot 10^{-2})$ </td><td> $1.066 \cdot 10^{0}$ </td><td> $(1.672 \cdot 10^{0})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.706 \cdot 10^{-2}$ </td><td> $(4.765 \cdot 10^{0})$ </td><td> $1.684 \cdot 10^{-2}$ </td><td> $(1.909 \cdot 10^{-2})$ </td><td> $2.823 \cdot 10^{-1}$ </td><td> $(1.218 \cdot 10^{0})$ </td></tr><tr><td> $4^{-7}$ </td><td> $4.266 \cdot 10^{-3}$ </td><td> $(1.724 \cdot 10^{0})$ </td><td> $4.209 \cdot 10^{-3}$ </td><td> $(4.774 \cdot 10^{-3})$ </td><td> $7.077 \cdot 10^{-2}$ </td><td> $(9.110 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-8}$ </td><td> $1.067 \cdot 10^{-3}$ </td><td> $(4.507 \cdot 10^{-1})$ </td><td> $1.052 \cdot 10^{-3}$ </td><td> $(1.194 \cdot 10^{-3})$ </td><td> $1.769 \cdot 10^{-2}$ </td><td> $(8.302 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-9}$ </td><td> $2.666 \cdot 10^{-4}$ </td><td> $(1.121 \cdot 10^{-1})$ </td><td> $2.631 \cdot 10^{-4}$ </td><td> $(2.984 \cdot 10^{-4})$ </td><td> $4.422 \cdot 10^{-3}$ </td><td> $(8.104 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-10}$ </td><td> $6.671 \cdot 10^{-5}$ </td><td> $(2.803 \cdot 10^{-2})$ </td><td> $6.579 \cdot 10^{-5}$ </td><td> $(7.462 \cdot 10^{-5})$ </td><td> $1.106 \cdot 10^{-3}$ </td><td> $(8.055 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-11}$ </td><td> $1.665 \cdot 10^{-5}$ </td><td> $(7.007 \cdot 10^{-3})$ </td><td> $1.645 \cdot 10^{-5}$ </td><td> $(1.865 \cdot 10^{-5})$ </td><td> $2.764 \cdot 10^{-4}$ </td><td> $(8.043 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-12}$ </td><td> $4.401 \cdot 10^{-6}$ </td><td> $(1.752 \cdot 10^{-3})$ </td><td> $4.109 \cdot 10^{-6}$ </td><td> $(4.718 \cdot 10^{-6})$ </td><td> $6.910 \cdot 10^{-5}$ </td><td> $(8.025 \cdot 10^{-1})$ </td></tr><tr><td> $4^{-13}$ </td><td> $1.507 \cdot 10^{-6}$ </td><td> $(4.378 \cdot 10^{-4})$ </td><td> $9.764 \cdot 10^{-7}$ </td><td> $(1.226 \cdot 10^{-6})$ </td><td> $1.718 \cdot 10^{-5}$ </td><td> $(4.307 \cdot 10^{-1})$ </td></tr></table>

Table 11: The medians over five trials plotted in Figure 8 on the bottom left, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td colspan="2">m=25</td><td colspan="2">m=50</td><td colspan="2">m=200</td></tr><tr><td> $4^{2}$ </td><td> $7.823 \cdot 10^{1}$ </td><td> $(2.780 \cdot 10^{-1})$ </td><td> $1.118 \cdot 10^{2}$ </td><td> $(4.888 \cdot 10^{-1})$ </td><td> $2.205 \cdot 10^{2}$ </td><td> $(2.389 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $1.974 \cdot 10^{1}$ </td><td> $(8.401 \cdot 10^{-2})$ </td><td> $2.803 \cdot 10^{1}$ </td><td> $(1.204 \cdot 10^{-1})$ </td><td> $5.534 \cdot 10^{1}$ </td><td> $(5.103 \cdot 10^{-2})$ </td></tr><tr><td> $4^{0}$ </td><td> $5.695 \cdot 10^{0}$ </td><td> $(1.629 \cdot 10^{-2})$ </td><td> $7.771 \cdot 10^{0}$ </td><td> $(2.592 \cdot 10^{-2})$ </td><td> $1.461 \cdot 10^{1}$ </td><td> $(1.094 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-1}$ </td><td> $2.189 \cdot 10^{0}$ </td><td> $(4.216 \cdot 10^{-3})$ </td><td> $2.712 \cdot 10^{0}$ </td><td> $(6.911 \cdot 10^{-3})$ </td><td> $4.421 \cdot 10^{0}$ </td><td> $(4.102 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-2}$ </td><td> $1.298 \cdot 10^{0}$ </td><td> $(1.043 \cdot 10^{-3})$ </td><td> $1.429 \cdot 10^{0}$ </td><td> $(1.780 \cdot 10^{-3})$ </td><td> $1.856 \cdot 10^{0}$ </td><td> $(1.132 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.075 \cdot 10^{0}$ </td><td> $(2.595 \cdot 10^{-4})$ </td><td> $1.107 \cdot 10^{0}$ </td><td> $(4.487 \cdot 10^{-4})$ </td><td> $1.214 \cdot 10^{0}$ </td><td> $(2.891 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.019 \cdot 10^{0}$ </td><td> $(6.386 \cdot 10^{-5})$ </td><td> $1.027 \cdot 10^{0}$ </td><td> $(1.122 \cdot 10^{-4})$ </td><td> $1.054 \cdot 10^{0}$ </td><td> $(7.164 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.005 \cdot 10^{0}$ </td><td> $(1.513 \cdot 10^{-5})$ </td><td> $1.007 \cdot 10^{0}$ </td><td> $(2.796 \cdot 10^{-5})$ </td><td> $1.013 \cdot 10^{0}$ </td><td> $(1.663 \cdot 10^{-5})$ </td></tr><tr><td> $4^{-6}$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(3.763 \cdot 10^{-6})$ </td><td> $1.002 \cdot 10^{0}$ </td><td> $(7.153 \cdot 10^{-6})$ </td><td> $1.003 \cdot 10^{0}$ </td><td> $(2.939 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-7}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.674 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.684 \cdot 10^{-6})$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(1.031 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-8}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.889 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.209 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.699 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-9}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.972 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.199 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.891 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-10}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.994 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.205 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.940 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-11}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(3.000 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.207 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.952 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-12}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(3.001 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.207 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.955 \cdot 10^{-6})$ </td></tr><tr><td> $4^{-13}$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(3.002 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(2.207 \cdot 10^{-6})$ </td><td> $1.000 \cdot 10^{0}$ </td><td> $(1.956 \cdot 10^{-6})$ </td></tr></table>

Table 12: The medians over five trials plotted in Figure 8 on the bottom right, with the standard deviations shown in parentheses, both rounded to four-digit mantissas. 

<table><tr><td> $\lambda$ </td><td colspan="2"> $m = 25$ </td><td colspan="2"> $m = 50$ </td><td colspan="2"> $m = 200$ </td></tr><tr><td> $4^{2}$ </td><td> $7.790 \cdot 10^{1}$ </td><td> $(4.055 \cdot 10^{-1})$ </td><td> $1.114 \cdot 10^{2}$ </td><td> $(1.844 \cdot 10^{-1})$ </td><td> $2.203 \cdot 10^{2}$ </td><td> $(2.081 \cdot 10^{-1})$ </td></tr><tr><td> $4^{1}$ </td><td> $1.960 \cdot 10^{1}$ </td><td> $(9.365 \cdot 10^{-2})$ </td><td> $2.799 \cdot 10^{1}$ </td><td> $(5.122 \cdot 10^{-2})$ </td><td> $5.530 \cdot 10^{1}$ </td><td> $(5.434 \cdot 10^{-2})$ </td></tr><tr><td> $4^{0}$ </td><td> $5.687 \cdot 10^{0}$ </td><td> $(2.398 \cdot 10^{-2})$ </td><td> $7.818 \cdot 10^{0}$ </td><td> $(2.092 \cdot 10^{-2})$ </td><td> $1.469 \cdot 10^{1}$ </td><td> $(2.460 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-1}$ </td><td> $2.209 \cdot 10^{0}$ </td><td> $(6.457 \cdot 10^{-3})$ </td><td> $2.758 \cdot 10^{0}$ </td><td> $(1.127 \cdot 10^{-2})$ </td><td> $4.507 \cdot 10^{0}$ </td><td> $(2.189 \cdot 10^{-2})$ </td></tr><tr><td> $4^{-2}$ </td><td> $1.300 \cdot 10^{0}$ </td><td> $(3.430 \cdot 10^{-3})$ </td><td> $1.436 \cdot 10^{0}$ </td><td> $(2.735 \cdot 10^{-3})$ </td><td> $1.872 \cdot 10^{0}$ </td><td> $(5.661 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-3}$ </td><td> $1.071 \cdot 10^{0}$ </td><td> $(9.842 \cdot 10^{-4})$ </td><td> $1.105 \cdot 10^{0}$ </td><td> $(6.528 \cdot 10^{-4})$ </td><td> $1.213 \cdot 10^{0}$ </td><td> $(5.792 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-4}$ </td><td> $1.015 \cdot 10^{0}$ </td><td> $(6.093 \cdot 10^{-4})$ </td><td> $1.022 \cdot 10^{0}$ </td><td> $(6.194 \cdot 10^{-4})$ </td><td> $1.049 \cdot 10^{0}$ </td><td> $(7.904 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-5}$ </td><td> $1.001 \cdot 10^{0}$ </td><td> $(5.900 \cdot 10^{-4})$ </td><td> $1.002 \cdot 10^{0}$ </td><td> $(6.740 \cdot 10^{-4})$ </td><td> $1.009 \cdot 10^{0}$ </td><td> $(9.884 \cdot 10^{-4})$ </td></tr><tr><td> $4^{-6}$ </td><td> $9.972 \cdot 10^{-1}$ </td><td> $(5.898 \cdot 10^{-4})$ </td><td> $9.967 \cdot 10^{-1}$ </td><td> $(6.937 \cdot 10^{-4})$ </td><td> $9.984 \cdot 10^{-1}$ </td><td> $(1.029 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-7}$ </td><td> $9.963 \cdot 10^{-1}$ </td><td> $(5.864 \cdot 10^{-4})$ </td><td> $9.954 \cdot 10^{-1}$ </td><td> $(6.985 \cdot 10^{-4})$ </td><td> $9.959 \cdot 10^{-1}$ </td><td> $(1.036 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-8}$ </td><td> $9.961 \cdot 10^{-1}$ </td><td> $(5.857 \cdot 10^{-4})$ </td><td> $9.951 \cdot 10^{-1}$ </td><td> $(6.994 \cdot 10^{-4})$ </td><td> $9.952 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-9}$ </td><td> $9.961 \cdot 10^{-1}$ </td><td> $(5.856 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(6.996 \cdot 10^{-4})$ </td><td> $9.951 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-10}$ </td><td> $9.961 \cdot 10^{-1}$ </td><td> $(5.855 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(6.996 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-11}$ </td><td> $9.960 \cdot 10^{-1}$ </td><td> $(5.855 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(6.996 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(1.037 \cdot 10^{-3})$ )</td></tr><tr><td> $4^{-12}$ </td><td> $9.960 \cdot 10^{-1}$ </td><td> $(5.855 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(6.996 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(1.027 \cdot 10^{-3})$ </td></tr><tr><td> $4^{-13}$ </td><td> $9.960 \cdot 10^{-1}$ </td><td> $(5.855 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(6.996 \cdot 10^{-4})$ </td><td> $9.950 \cdot 10^{-1}$ </td><td> $(1.007 \cdot 10^{-3})$ </td></tr></table>

![](images/c80f1410f670b71731f72ada04648ae2fcfce695f418dbe9e49eb47e5e0c8a8a.jpg)  
- max. angle b. positive neurons - max. angle b. negative neurons

Figure 10: The maximum angle between hidden neurons that start with a positive (blue) and negative (red) inner product with the first teacher neuron. The first teacher neuron has norm 1 and the second teacher neuron has norm 3. The vertical axes are logarithmic and the angles are in degrees. The horizontal axes show different multipliers $\rho$ for the variance of the distribution of the data points (cf. section 8). The input dimension is d = 16 and, for each teacher neuron, we sample d data points from the distribution specified in the main. Each point in the plot shows the median over 15 trials of the angle in degrees at the end of training. The training runs for $2 \cdot 10^{7}$ iterations or until the loss reaches $10^{-9}$ . The width of the network is m = 25, and the initialisation scale is $\lambda = 4^{-7}$ .

# I Further experiments

Here we report on experiments in which we explore the effects of adding a second teacher neuron whose direction is opposite to that of the first, and of increasing the scale $\rho$ of the noise used to generate the synthetic datasets (cf. section 8) so that quickly most of the data points exceed the $\pi/4$ angle with their corresponding teacher neuron.

In Figure 10, the growing maximum angles between neurons at the end of the training indicate that we no longer have a single (or one per teacher neuron) aligned bundle of neurons forming and sticking together for the rest of the training.

In the bottom two plots of Figure 10 and in Figure 11, for small scales $\rho$ (where the smallest values are such that the angles between the data points and the corresponding teacher neuron concentrate around $\pi/4$ ), the phenomena we identified theoretically still seem to hold, where the training passes near a second saddle point as we outlined in section 9.

![](images/80f4732a3c3291aeed680c5b44c4970a7bdeedba0b960ae0ca107bca5a70cc80.jpg)

<details>
<summary>line</summary>

| iteration | max. angle b. positive neurons | max. angle b. negative neurons | training loss |
| --------- | ------------------------------ | ------------------------------ | ------------- |
| 10^0      | ~1.5                           | ~1.5                           | ~1.0          |
| 10^1      | ~1.5                           | ~1.5                           | ~1.0          |
| 10^2      | ~1.0                           | ~1.0                           | ~1.0          |
| 10^3      | ~0.1                           | ~0.1                           | ~0.1          |
| 10^4      | ~0.1                           | ~0.1                           | ~0.01         |
| 10^5      | ~0.1                           | ~0.1                           | ~0.001        |
| 10^6      | ~0.1                           | ~0.1                           | ~0.0001       |
</details>

Figure 11: The evolution of the training loss and the maximum angle between positive and negative hidden neurons during the training in dimension 16. The vertical axes are logarithmic and the angles are in degrees. This is one example of a run contributing to Figure 10. Specifically, in this run the training dataset is uncentered and $\rho = \sqrt{2} - 1$ . The two fast drops in loss (after passing of the first and then the second saddle point) coincide with the times at which the respective group of hidden neurons aligns.