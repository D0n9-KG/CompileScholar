# RAMP: Boosting Adversarial Robustness Against Multiple $l_{p}$ Perturbations for Universal Robustness

Enyi Jiang, Gagandeep Singh

University of Illinois Urbana-Champaign

{enyij2, ggnds}@illinois.edu

# Abstract

Most existing works focus on improving robustness against adversarial attacks bounded by a single $l_{p}$ norm using adversarial training (AT). However, these AT models' multiple-norm robustness (union accuracy) is still low, which is crucial since in the real-world an adversary is not necessarily bounded by a single norm. The tradeoffs among robustness against multiple $l_{p}$ perturbations and accuracy/robustness make obtaining good union and clean accuracy challenging. We design a logit pairing loss to improve the union accuracy by analyzing the tradeoffs from the lens of distribution shifts. We connect natural training (NT) with AT via gradient projection, to incorporate useful information from NT into AT, where we empirically and theoretically show it moderates the accuracy/robustness tradeoff. We propose a novel training framework RAMP, to boost the robustness against multiple $l_{p}$ perturbations. RAMP can be easily adapted for robust fine-tuning and full AT. For robust fine-tuning, RAMP obtains a union accuracy up to 53.3% on CIFAR-10, and 29.1% on ImageNet. For training from scratch, RAMP achieves a union accuracy of 44.6% and good clean accuracy of 81.2% on ResNet-18 against AutoAttack on CIFAR-10. Beyond multi-norm robustness RAMP-trained models achieve superior universal robustness, effectively generalizing against a range of unseen adversaries and natural corruptions.

# 1 Introduction

Though deep neural networks (DNNs) demonstrate superior performance in various vision applications, they are vulnerable against adversarial examples [Goodfellow et al., 2014, Kurakin et al., 2018]. Adversarial training (AT) [Tramèr et al., 2017, Madry et al., 2017] which works by injecting adversarial examples into training for enhanced robustness, is currently the most popular defense. However, most AT methods address only a single type of perturbation [Wang et al., 2020, Wu et al., 2020, Carmon et al., 2019, Gowal et al., 2020, Raghunathan et al., 2020, Zhang et al., 2021, Debenedetti and Troncoso—EPFL, 2022, Peng et al., 2023, Wang et al., 2023]. An $l_{\infty}$ robust model may not be robust against $l_{p}(p \neq \infty)$ attacks. Also, enhancing robustness against one perturbation type can sometimes increase vulnerability to others [Engstrom et al., 2017, Schott et al., 2018]. On the contrary, training a model to be robust against multiple $l_{p}$ perturbations is crucial as it reflects real-world scenarios [Sharif et al., 2016, Eykholt et al., 2018, Song et al., 2018, Athalye et al., 2018] where adversaries can use multiple $l_{p}$ perturbations. We show that multi-norm robustness is the key to improving generalization against other threat models [Croce and Hein, 2022]. For instance, we show it enables robustness against perturbations not easily defined mathematically, such as image corruptions and unseen adversaries [Wong and Kolter, 2020].

Two main challenges exist for training models robust against multiple perturbations: (i) tradeoff among robustness against different perturbation models [Tramer and Boneh, 2019] and (ii) tradeoff between accuracy and robustness [Zhang et al., 2019, Raghunathan et al., 2020]. Adversarial examples

induce a shift from the original distribution, causing a drop in clean accuracy with AT [Xie et al., 2020, Benz et al., 2021]. The distinct distributions created by $l_{1}, l_{2}, l_{\infty}$ adversarial examples make the problem even more challenging. Through a finer analysis of the distribution shifts caused by these adversaries, we propose the RAMP framework to efficiently boost the Robustness Against Multiple Perturbations. RAMP can be used for both fine-tuning and training from scratch. It utilizes a novel logit pairing loss on a certain pair and connects NT with AT via gradient projection [Jiang et al., 2023] to improve union accuracy while maintaining good clean accuracy and training efficiency.

Logit pairing loss. We visualize the changing of $l_{1}, l_{2}, l_{\infty}$ robustness when fine-tuning a $l_{\infty}$ -AT pre-trained model in Figure 1 using the CIFAR-10 training dataset. The DNN loses substantial robustness against $l_{\infty}$ attack after only 1 epoch of fine-tuning: $l_{1}$ fine-tuning and E-AT [Croce and Hein, 2022] (red and yellow histograms under Linf category) both lose significant $l_{\infty}$ robustness (compared with blue histogram under Linf category). Inspired by this observation, we devise a new logit pairing loss for a $l_{q} - l_{r}$ tradeoff pair to attain better union accuracy, which enforces the logit distributions of $l_{q}$ and $l_{r}$ adversarial examples to be close, specifically on the correctly classified $l_{q}$ subsets. In comparison, our method (green histogram under Linf and union categories) preserves more $l_{\infty}$ and union robustness than others after 1 epoch. We show this technique works on larger models and datasets (Section 5.1).

Connect natural training (NT) with AT. We explore the connections between NT and AT to obtain a better accuracy/robustness trade-off. We find that NT can help with adversarial robustness: useful information in natural distribution can be extracted and leveraged to achieve better robustness. To this end, we compare the similarities of model updates of NT and AT layer-wise for each epoch, where we find and incorporate useful NT components into AT via gradient projection (GP), as outlined in Algorithm 2. In Figure 2 and Section 5.1, we empirically and theoretically show this technique strikes a better balance between accuracy and robustness, for both single and multiple $l_{p}$ perturbations. We provide a theoretical analysis of why GP works for adversarial robustness in Theorem A.2 & 4.5.

![](images/a7e77c30a0ada0428f35494e2ffad1efb91d371efe774d4633eb72f8beb4abff.jpg)

<details>
<summary>bar</summary>

| Model | Epoch 0 | Epoch 1 - Fintune L1 | Epoch 1 - EAT | Epoch 1 - RAMP |
|---|---|---|---|---|
| Linf | 0.63 | 0.30 | 0.45 | 0.58 |
| L1 | 0.49 | 0.56 | 0.58 | 0.69 |
| L2 | 0.73 | 0.63 | 0.67 | 0.78 |
| Union | 0.48 | 0.30 | 0.44 | 0.58 |
</details>

Figure 1: Multiple-norm tradeoff with robust fine-tuning: We observe that fine-tuning on $l_{\infty}$ -AT model using $l_{1}$ examples drastically reduces $l_{\infty}$ robustness. RAMP preserves more $l_{\infty}$ and union robustness.

# Main contributions:

- We design a new logit pairing loss to mitigate the $l_{q} - l_{r}$ tradeoff for better union accuracy, by enforcing the logit distributions of $l_{q}$ and $l_{r}$ adversarial examples to be close.   
- We empirically and theoretically show that connecting NT with AT via gradient projection better balances the accuracy/robustness tradeoff for $l_{p}$ perturbations, compared with standard AT.   
- RAMP achieves good union accuracy, accuracy-robustness tradeoff, and generalizes better to diverse perturbations and corruptions (Section 5.1) achieving superior universal robustness (75.5% for common corruption and 26.1% union accuracy against unseen adversaries). RAMP fine-tuned DNNs achieve union accuracy up to 53.3% on CIFAR-10, and 29.1% on ImageNet. RAMP achieves a 44.6% union accuracy and good clean accuracy on ResNet-18 against AutoAttack on CIFAR-10.

Our code is available at https://github.com/uiuc-focal-lab/RAMP.

# 2 Related Work

Adversarial training (AT). Adversarial Training (AT) usually employs gradient descent to discover adversarial examples, incorporating them into training for enhanced adversarial robustness [Tramèr et al., 2017, Madry et al., 2017]. Numerous works focus on improving robustness by exploring the trade-off between robustness and accuracy [Zhang et al., 2019, Wang et al., 2020], instance reweighting [Zhang et al., 2021], loss landscapes [Wu et al., 2020], wider/larger architectures [Gowal et al., 2020, Debenedetti and Troncoso—EPFL, 2022], data augmentation [Carmon et al., 2019,

Raghunathan et al., 2020], and using synthetic data [Peng et al., 2023, Wang et al., 2023]. However, these methods often yield DNNs robust against a single perturbation type while remaining vulnerable to other types.

Robustness against multiple perturbations. Tramer and Boneh [2019], Kang et al. [2019] observe that robustness against $l_{p}$ attacks does not necessarily transfer to other $l_{q}$ attacks ( $q \neq p$ ). Previous studies [Tramer and Boneh, 2019, Maini et al., 2020, Madaan et al., 2021, Croce and Hein, 2022] modified Adversarial Training (AT) to enhance robustness against multiple $l_{p}$ attacks, employing average-case [Tramer and Boneh, 2019], worst-case [Tramer and Boneh, 2019, Maini et al., 2020], and random-sampled [Madaan et al., 2021, Croce and Hein, 2022] defenses. There are also works [Nandy et al., 2020, Liu et al., 2020, Xu et al., 2021, Xiao et al., 2022, Maini et al., 2022] using preprocessing, ensemble methods, mixture of experts, and stability analysis to solve this problem. Ensemble models and preprocessing methods are weakened since their performance heavily relies on correctly classifying or detecting various types of adversarial examples. Also, prior works are hard to scale to larger models and datasets, e.g. ImageNet, due to the efficiency issue. Furthermore, Croce and Hein [2022] devise Extreme norm Adversarial Training (E-AT) and fine-tune a $l_{p}$ robust model on another $l_{q}$ perturbation to quickly make a DNN robust against multiple $l_{p}$ attacks. However, E-AT does not adapt to varying epsilon values. Our work demonstrates that the suboptimal tradeoff observed in prior studies can be improved with our proposed framework.

Logit pairing in adversarial training. Adversarial logit pairing methods encourage logits for pairs of examples to be similar [Kannan et al., 2018, Engstrom et al., 2018]. People apply this technique to both clean images and their adversarial counterparts, to devise a stronger form of adversarial training. In our work, we devise a novel logit pairing loss to train a DNN originally robust against $l_{p}$ attack to become robust against another $l_{q}(q \neq p)$ attack on the correctly predicted $l_{p}$ subsets, which helps gain better union accuracy.

Adversarial versus distributional robustness. Sinha et al. [2018] theoretically studies the AT problem through distributional robust optimization. Mehrabi et al. [2021] establishes a pareto-optimal tradeoff between standard and adversarial risks by perturbing the test distribution. Other works explore the connection between natural and adversarial distribution shifts [Moayeri et al., 2022, Alhamoud et al., 2023], assessing transferability and generalizability of adversarial robustness across datasets. However, little research delves into distribution shifts induced by $l_1, l_2, l_\infty$ adversarial examples and their interplay with the robustness-accuracy tradeoff [Zhang et al., 2019, Yang et al., 2020, Rade and Moosavi-Dezfooli, 2021]. Our work, inspired by recent domain adaptation techniques [Jiang, 2023, Jiang et al., 2023], designs a logit pairing loss and utilizes model updates from NT via GP to enhance adversarial robustness. We show that GP adapts to both single and multi-norm scenarios.

# 3 AT against Multiple Perturbations

We consider a standard classification task with samples $\{(x_{i},y_{i})\}_{i=0}^{N}$ from an empirical data distribution $\widehat{D}_{n}$ ; we have input images $x\in R^{d}$ and corresponding labels $y\in R^{k}$ . Standard training aims to obtain a classifier f parameterized by $\theta$ to minimize a loss function $L:R^{k}\times R^{k}\to R$ on $\widehat{D}_{n}$ . Adversarial training (AT) [Madry et al., 2017, Tramèr et al., 2017] aims to find a DNN robust against adversarial examples. It is framed as a min-max problem where a DNN is optimized using the worst-case examples within an adversarial region around each $x_{i}$ . Different types of adversarial regions $B_{p}(x,\epsilon_{p})=\{x^{\prime}\in\mathbb{R}^{d}:\|x^{\prime}-x\|_{p}\leq\epsilon_{p}\}$ can be defined around a given image x using various $l_{p}$ -based perturbations. Formally, we can write the optimization problem of AT against a certain $l_{p}$ attack as follows:

$$
\min _ {\theta} \mathbb {E} _ {(x, y) \sim \widehat {\mathcal {D}} _ {n}} \left[ \max _ {x ^ {\prime} \in B _ {p} (x, \epsilon_ {p})} \mathcal {L} (f (x ^ {\prime}), y) \right]
$$

The above optimization is only for certain $p$ values and is usually vulnerable to other perturbation types. To this end, prior works have proposed several approaches to train the network robust against multiple perturbations $(l_1, l_2, l_\infty)$ at the same time. We focus on the union threat model $\Delta = B_1(x, \epsilon_1) \cup B_2(x, \epsilon_2) \cup B_\infty(x, \epsilon_\infty)$ which requires the DNN to be robust within the $l_1, l_2, l_\infty$ adversarial regions simultaneously [Croce and Hein, 2022]. Union accuracy is then defined as the robustness against $\Delta_{(i)}$ for each $x_i$ sampled from $\mathcal{D}$ . In this paper, similar to the prior works, we use

union accuracy as the main metric to evaluate the multiple-norm robustness. Apart from that, we define universal robustness as the generalization ability against a range of unseen adversaries and common corruptions. Specifically, we have average accuracy across five severity levels for common corruption and union accuracy against a range of unseen adversaries used in Laidlaw et al. [2020].

Worst-case defense follows the following min-max optimization problem to train DNNs using the worst-case example from the $l_{1}, l_{2}, l_{\infty}$ adversarial regions:

$$
\min _ {\theta} \mathbb {E} _ {(x, y) \sim \widehat {\mathcal {D}} _ {n}} \left[ \max _ {p \in \{1, 2, \infty \}} \max _ {x ^ {\prime} \in B _ {p} (x, \epsilon_ {p})} \mathcal {L} (f (x ^ {\prime}), y) \right]
$$

MAX [Tramer and Boneh, 2019] and MSD [Maini et al., 2020] fall into this category. Finding worst-case examples yields a good union accuracy but results in a loss of clean accuracy as the distribution of generated examples is different from the clean data distribution.

Average-case defense train DNNs using the average of the $l_{1}, l_{2}, l_{\infty}$ worst-case examples:

$$
\min _ {\theta} \mathbb {E} _ {(x, y) \sim \widehat {\mathcal {D}} _ {n}} \left[ \mathbb {E} _ {p \in \{1, 2, \infty \}} \max _ {x ^ {\prime} \in B _ {p} (x, \epsilon_ {p})} \mathcal {L} (f (x ^ {\prime}), y) \right]
$$

AVG [Tramer and Boneh, 2019] is of this type. This method generally leads to good clean accuracy but suboptimal union accuracy as it does not penalize worst-case behavior within the $l_{1}, l_{2}, l_{\infty}$ regions.

Random-sampled defense. The defenses mentioned above lead to a high training cost as they compute multiple attacks for each sample. SAT [Madaan et al., 2021] and E-AT [Croce and Hein, 2022] randomly sample one attack out of each type at a time, contributing to a similar computational cost as standard AT on a single perturbation model. They achieve a slightly better union accuracy compared with AVG and relatively good clean accuracy. However, they are not better than worst-case defenses for multiple-norm robustness, since they do not consider the strongest attack within the union region all the time.

# 4 RAMP

There are two main tradeoffs in achieving better union accuracy while maintaining good accuracy: 1. Among perturbations: there is a tradeoff among different attacks, e.g., a $l_{\infty}$ pre-trained AT DNN is not robust against $l_{1}, l_{2}$ perturbations, which makes the union accuracy harder to attain. Also, we observe there exists a main tradeoff pair of two attacks among the union over $l_{1}, l_{2}, l_{\infty}$ attacks. 2. Accuracy and robustness: all defenses lead to degraded clean accuracy. To address these tradeoffs, we study the problem from the lens of distribution shifts.

Interpreting tradeoffs from the lens of distribution shifts. The adversarial examples with respect to an empirical data distribution $\widehat{D}_{n}$ , adversarial region $B_{p}(x,\epsilon_{p})$ , and DNN $f_{\theta}$ generate a new adversarial distribution $\widehat{D}_{a}$ with samples $\{(x_{i}^{\prime},y_{i})\}_{i=0}^{N}$ , that are correlated by adding certain perturbations but different from the original $\widehat{D}_{n}$ . Because of the shifts between $\widehat{D}_{n}$ and $\widehat{D}_{a}$ , DNN decreases performance on $\widehat{D}_{n}$ when we move away from it and towards $\widehat{D}_{a}$ . Also, the distinct distributions created by multiple perturbations, $\widehat{D}_{a}^{l_{1}}$ , $\widehat{D}_{a}^{l_{2}}$ , $\widehat{D}_{a}^{l_{\infty}}$ , contribute to the tradeoff among $l_{1}, l_{2}, l_{\infty}$ attacks. To address the tradeoff among perturbations while maintaining good efficiency, we focus on the distributional interconnections between $\widehat{D}_{n}$ and $\widehat{D}_{a}^{l_{1}}$ , $\widehat{D}_{a}^{l_{2}}$ , $\widehat{D}_{a}^{l_{\infty}}$ . From the insights we get from above, we propose our framework RAMP, which includes (i) logit pairing to improve tradeoffs among multiple perturbations, and (ii) identifying and combining the useful DNN components using the model updates from NT and AT, to obtain a better robustness/accuracy tradeoff.

Identify the Key Tradeoff Pair. We study the common case with $l_{p}$ norms $\epsilon_{1}=12,\epsilon_{2}=0.5,\epsilon_{\infty}=\frac{8}{255}$ on CIFAR-10 [Tramer and Boneh, 2019]. The distributions generated by the two strongest attacks show the largest shifts from $\widehat{D}_{n}$ ; also, they have the largest distribution shifts between each other because of larger and most distinct search areas. Thus, by calculating the ball volume [Wikipedia contributors, 2024] for each attack, we select the two with the largest volumes as the key tradeoff pair. They refer to the strongest attack as the attacker has more search area. The attack with the smallest

ball volume is mostly included by the convex hull of the other two stronger attacks [Croce and Hein, 2022]. Here we identify $l_{\infty} - l_1$ as the key tradeoff pair.

# 4.1 Logit Pairing for Multiple Perturbations

Figure 1: Finetuning a $l_q$ -AT model on $l_r$ examples reduces $l_q$ robustness. To get a finer analysis of the $l_\infty - l_1$ tradeoff mentioned above, we visualize the changing of $l_1, l_2, l_\infty$ robustness of the training dataset when we fine-tune a $l_\infty$ pre-trained model with $l_1$ examples for 1 epochs, as shown in Figure 1: x-axis represents the robustness against different attacks and y-axis is the accuracy. After 1 epoch of finetuning on $l_1$ examples or performing E-AT, we lose much $l_\infty$ robustness since blue/yellow histograms are much lower than the red histogram under the Linf category. RAMP preserves both $l_\infty$ and union robustness more effectively: the green histogram is higher than the red/yellow histogram under Linf and Union categories. Specifically, RAMP maintains 14%, 28% more union robustness than E-AT and $l_1$ fine-tuning. The above observations indicate the necessity of preserving more $l_q$ robustness as we adversarially fine-tune with $l_r$ adversarial examples on a $l_q$ pre-trained AT model, with $l_q - l_r$ as the key tradeoff pair, which inspires us to design our loss design with logit pairing. We want to enforce the union predictions between $l_q$ and $l_r (q \neq r)$ attacks: bringing the predictions of $l_q$ and $l_r (q \neq r)$ close to each other, specifically on the correctly predicted $l_q$ subsets. Based on our observations, we design a new logit pairing loss to enforce a DNN robust against one $l_q$ attack to be robust against another $l_r (q \neq r)$ attack.

Enforcing the Union Prediction via Logit Pairing. The $l_{q} - l_{r}(q \neq r)$ tradeoff leads us to the following principle to improve union accuracy: for a given set of images, when we have a DNN robust against some $l_{q}$ examples, we want it to be robust against $l_{r}$ examples as well. This serves as the main insight for our loss design: we want to enforce the logits predicted by $l_{q}$ and $l_{r}$ adversarial examples to be close, specifically on the correctly predicted $l_{q}$ subsets. To accomplish this, we design a KL-divergence (KL) loss between the predictions from $l_{q}$ and $l_{r}$ perturbations. For each batch of data $(x, y) \sim \mathcal{D}$ , we generate $l_{q}$ and $l_{r}$ adversarial examples $x_{q}^{\prime}, x_{r}^{\prime}$ and their predictions $p_{q}, p_{r}$ using APGD [Croce and Hein, 2020]. Then, we select indices $\gamma$ , which part elements of $p_{q}$ correctly predicts the ground truth y. We denote the size of the indices as $n_{c}$ , and the batch size as N. We compute a KL-divergence loss over this set of samples using $KL(p_{q}[\gamma] \| p_{r}[\gamma])$ (Eq. 1). For the subset indexed by $\gamma$ , we want to push its $l_{r}$ logit distribution towards its $l_{q}$ logit distribution, such that we prevent losing more $l_{q}$ robustness when training with $l_{r}$ adversarial examples.

$$
\mathcal {L} _ {K L} = \frac {1}{n _ {c}} \cdot \sum_ {i = 1} ^ {n _ {c}} \sum_ {j = 0} ^ {k} p _ {q} [ \gamma [ i ] ] [ j ] \cdot \log \left(\frac {p _ {q} [ \gamma [ i ] ] [ j ]}{p _ {r} [ \gamma [ i ] ] [ j ]}\right) \tag {1}
$$

To further boost the union accuracy, apart from the KL loss, we add another loss term using a MAX-style approach in Eq. 2: we find the worst-case example between $l_{q}$ and $l_{r}$ adversarial regions by selecting the example with the higher loss. $\mathcal{L}_{max}$ is a cross-entropy loss over the approximated worst-case adversarial examples. Here, we use $\mathcal{L}_{ce}$ to represent the cross-entropy loss. Our final loss $\mathcal{L}$ combines $\mathcal{L}_{KL}$ and $\mathcal{L}_{max}$ , via a hyper-parameter $\lambda$ in Eq. 3.

$$
\mathcal {L} _ {\max} = \frac {1}{N} \sum_ {i = 0} ^ {N} \left[ \max _ {p \in \{q, r \}} \max _ {x _ {i} ^ {\prime} \in B _ {p} (x, \epsilon_ {p})} \mathcal {L} _ {c e} (f (x _ {i} ^ {\prime}), y _ {i}) \right] \tag {2}
$$

Algorithm 1 shows the pseudocode of robust fine-tuning with RAMP that leverages logit pairing.

# 4.2 Connecting Natural Training with AT

To improve the robustness and accuracy tradeoff against multiple perturbations, we explore the connections between AT and NT. Since extracting valuable information in NT aids in improving robustness (Section 4.2), we use gradient projection [Jiang et al., 2023] to compare and integrate natural and adversarial model updates, which yields an improved tradeoff between robustness and accuracy.

NT can help adversarial robustness. Let us consider two models $f_{1}$ and $f_{2}$ , where $f_{1}$ is randomly initialized and $f_{2}$ undergoes NT on $\widehat{D}_{n}$ for k epochs: $f_{2}$ results in a better decision boundary and

higher clean accuracy. Performing AT on $f_{1}$ and $f_{2}$ subsequently, intuitively, $f_{2}$ becomes more robust than $f_{1}$ due to its improved decision boundary, leading to fewer misclassifications of adversarial examples. This effect is empirically shown in Figure 2. For AT (blue), standard AT against $l_{\infty}$ attack [Madry et al., 2017] is performed, while for AT-pre (red), 50 epochs of pre-training precede the standard AT procedure. AT-pre shows superior clean and robust accuracy on CIFAR-10 against $l_{\infty}$ PGD-20 attack with $\epsilon_{\infty}=0.031$ . Despite $\widehat{D}_{n}$ and $\widehat{D}_{a}$ are different, Figure 2 suggests valuable information in $\widehat{D}_{n}$ that potentially enhances performance on $\widehat{D}_{a}$ .

AT with Gradient Projection. To connect NT with AT more effectively, we analyze the training procedures on $\widehat{D}_{n}$ and $\widehat{D}_{a}$ . We consider model updates over all samples from $\widehat{D}_{n}$ and $\widehat{D}_{a}$ , with the initial model $f^{(r)}$ at epoch r, and models $f_{n}^{(r)}$ and $f_{a}^{(r)}$ after 1 epoch of natural and adversarial training from the same starting point $f^{(r)}$ , respectively. Here, we compare the natural updates $\widehat{g}_{n}=f_{n}^{(r)}-f^{(r)}$ and adversarial updates $\widehat{g}_{a}=f_{a}^{(r)}-f^{(r)}$ . Due to distribution shift, an angle exists between them. Our goal is to identify useful components from $g_{n}$ and incorporate them into $g_{a}$ for increased robustness in $\widehat{D}_{a}$ while maintaining accuracy in $\widehat{D}_{n}$ . Inspired by Jiang et al. [2023], we layer-wisely compute the cosine similarity between $\widehat{g}_{n}$ and $\widehat{g}_{a}$ . For a specific layer l of $\widehat{g}_{n}^{l}$ and $\widehat{g}_{a}^{l}$ , we preserve a portion of $\widehat{g}_{n}^{l}$ based on their cosine similarity score (Eq.4). Negative scores indicate that $\widehat{g}_{n}^{l}$ is not beneficial for robustness in $\widehat{D}_{a}$ . Therefore, we filter components with similarity score $\leq0$ . We define the GP (Gradient Projection) operation in Eq.5 by projecting $\widehat{g}_{a}^{l}$ towards $\widehat{g}_{n}^{l}$ .

$$
\cos (\widehat {g} _ {n} ^ {l}, \widehat {g} _ {a} ^ {l}) = \frac {\widehat {g} _ {n} ^ {l} \cdot \widehat {g} _ {a} ^ {l}}{\| \widehat {g} _ {n} ^ {l} \| \| \widehat {g} _ {a} ^ {l} \|} \tag {4}
$$

$$
\mathbf {G P} (\widehat {g} _ {n} ^ {l}, \widehat {g} _ {a} ^ {l}) = \left\{ \begin{array}{l l} \cos (\widehat {g} _ {n} ^ {l}, \widehat {g} _ {a} ^ {l}) \cdot \widehat {g} _ {n} ^ {l}, & \cos (\widehat {g} _ {n} ^ {l}, \widehat {g} _ {a} ^ {l}) > 0 \\ 0, & \cos (\widehat {g} _ {n} ^ {l}, \widehat {g} _ {a} ^ {l}) \leq 0 \end{array} \right. \tag {5}
$$

Therefore, the total projected (useful) model updates $g_{p}$ coming from $\widehat{g}_{n}$ could be computed as Eq. 6. We use M to denote all layers of the current model update. Note that $\bigcup_{l\in M}$ concatenates all layers' useful natural model update components. A hyper-parameter $\beta$ is used to balance the contributions of $g_{GP}$ and $\widehat{g}_{a}$ , as shown in Eq. 7. By finding a proper $\beta$ (0.5 as in Figure 4c), we can obtain better robustness on $\widehat{D}_{a}$ , as shown in Figure 2 and Figure 3. In Figure 2, with $\beta = 0.5$ , AT-GP refers to AT with GP; for AT-GP-pre, we perform 50 epochs of NT before doing AT-GP. We see AT-GP obtains a better accuracy/robustness tradeoff than AT. We observe a similar trend for AT-GP-pre vs. AT-pre. Further, in Figure 3, RN-18 $l_{\infty}$ -GP achieves good clean accuracy and better robustness than RN-18 $l_{\infty}$ against AutoAttack [Croce and Hein, 2020].

$$
g _ {p} = \bigcup_ {l \in \mathcal {M}} \mathbf {G P} (\widehat {g} _ {n} ^ {l}, \widehat {g} _ {a} ^ {l}) \tag {6}
$$

$$
f ^ {(r + 1)} = f ^ {(r)} + \beta \cdot g _ {p} + (1 - \beta) \cdot \widehat {g} _ {a} \tag {7}
$$

Algorithm 1 Fine-tuning via Logit Pairing   
1: Input: model f, input samples $(x,y)$ from distribution $\widehat{D}_{n}$ , fine-tuning rounds R, hyper-parameter $\lambda$ , adversarial regions $B_{q}, B_{r}$ with size $\epsilon_{q}$ and $\epsilon_{r}$ , APGD attack.
2: for $r = 1, 2, \ldots, R$ do
3:    for $(x, y) \sim \text{training set } \mathcal{D} \text{ do}$ 4: $x_{q}^{\prime}, p_{q} \leftarrow \text{APGD}(B_{q}(x, \epsilon_{q}), y)$ 5: $x_{r}^{\prime}, p_{r} \leftarrow \text{APGD}(B_{r}(x, \epsilon_{r}), y)$ 6: $\gamma \leftarrow where(argmax p_{q} = y)$ 7: $n_{c} \leftarrow \gamma.size()$ 8:    calculate L using Eq. 3 and update f
9:    end for
10: end for
11: Output: model f.

Algorithm 2 Connect AT with NT via GP   
1: Input: model f, input images with distribution $\widehat{D}_{n}$ , training rounds R, adversarial region $B_{p}$ and its size $\epsilon_{p}$ , $\beta$ , natural training NT and adversarial training AT.
2: for $r = 1, 2, \ldots, R$ do
3: $f_{n} \leftarrow \mathbf{NT}(f^{(r)}, \mathcal{D})$ 4: $f_{a} \leftarrow \mathbf{AT}(f^{(r)}, B_{p}, \epsilon_{p}, \mathcal{D})$ 5: compute $\widehat{g}_{n} \leftarrow f_{n} - f^{(r)}$ , $\widehat{g}_{a} \leftarrow f_{a} - f^{(r)}$ 6: compute $g_{p}$ using Eq. 6
7: update $f^{(r+1)}$ using Eq. 7 with $\beta$ and $\widehat{g}_{a}$ 8: end for
9: Output: model f.

# 4.3 Theoretical Analysis of GP for Adversarial Robustness

We define $\mathcal{D}_{n}=\{(x_{i},y_{i})\}_{i=0}^{\infty}$ as the ideal data distribution with an infinite cardinality. Here, we consider a classifier $f_{\theta}$ at epoch t. We define $D_{a}$ as the distribution created by $\{(x_{i}+\epsilon(f_{\theta},x_{i},y_{i}),y_{i})\}_{i=0}^{\infty}$ where $(x_{i},y_{i})\sim\mathcal{D}_{n}$ . $x_{i}+\epsilon(f_{\theta},x_{i},y_{i})$ denotes the perturbed image, which could be both single and multiple perturbations based on $f_{\theta}$ itself.

Assumption 4.1. We assume $\widehat{\mathcal{D}}_n$ consists of $N$ i.i.d. samples from the ideal distribution $\mathcal{D}_n$ and $\widehat{\mathcal{D}}_a = \{(x_i + \epsilon(f^\theta, x_i, y_i), y_i)\}_{i=0}^N$ where $(x_i, y_i) \sim \widehat{\mathcal{D}}_n$ consists of $N$ i.i.d. samples from $\mathcal{D}_a$ .

We define the population loss as $\mathcal{L}_{\mathcal{D}}(\theta):=\mathbb{E}_{(x,y)\sim\mathcal{D}}\mathcal{L}(f(x),y)$ , and let $g_{\mathcal{D}}(\theta):=\nabla\mathcal{L}_{\mathcal{D}}(\theta)$ . For simplification, we use $g_{a}:=\nabla\mathcal{L}_{\mathcal{D}_{a}}(\theta)$ , $\widehat{g}_{a}:=\nabla\mathcal{L}_{\widehat{\mathcal{D}}_{a}}(\theta)$ , and $\widehat{g}_{n}:=\nabla\mathcal{L}_{\widehat{\mathcal{D}}_{n}}(\theta)$ . $g_{GP}=\beta\cdot g_{p}+(1-\beta)\cdot\widehat{g}_{a}$ (Definition A.3) is the aggregation using GP. We define the following optimization problem.

Definition 4.2 (Aggregation for NT and AT). $f_{\theta}$ is trained by iteratively updating the parameter

$$
\theta \leftarrow \theta - \mu \cdot A g g r (\widehat {g} _ {a}, \widehat {g} _ {n}),
$$

where $\mu$ is the step size. We seek an aggregation rule $Aggr(\cdot)=\widehat{g}_{Aggr}$ such that after training, $f_{\theta}$ minimizes the population loss function $\mathcal{L}_{\mathcal{D}_{a}}(\theta)$ .

We need $\widehat{g}_{\mathrm{Aggr}}$ to be close to $g_{a}$ for each iteration, since $g_{a}$ is the optimal update on $\mathcal{D}_a$ . Thus, we define $L^{\pi}$ -Norm and delta error to indicate the performance of different aggregation rules.

Definition 4.3 ( $L^{\pi}$ -Norm [Enyi Jiang, 2024]). Given a distribution $\pi$ on the parameter space $\theta$ , we define an inner product $\langle g_{\mathcal{D}}, g_{\mathcal{D}'} \rangle_{\pi} = \mathbb{E}_{\theta \sim \pi}[\langle g_{\mathcal{D}}(\theta), g_{\mathcal{D}'}(\theta) \rangle]$ . The inner product induces the $L^{\pi}$ -norm on $g_{\mathcal{D}}$ as $\|g_{\mathcal{D}}\|_{\pi} := \sqrt{\mathbb{E}_{\theta \sim \pi} \|g_{\mathcal{D}}(\theta)\|^2}$ . We use $L^{\pi}$ -norm to measure the gradient differences under certain $\mathcal{D}$ .

Definition 4.4 (Delta Error of an aggregation rule $\mathsf{Aggr}(\cdot)$ ). We define the following squared error term to measure the closeness between $\widehat{g}_{\mathsf{Aggr}}$ and $g_{a}$ under $\widehat{\mathcal{D}}_a^t$ (distribution at time step t), i.e.,

$$
\Delta_ {A g g r} ^ {2} := \mathbb {E} _ {\widehat {\mathcal {D}} _ {a} ^ {t}} \| g _ {a} - \widehat {g} _ {A g g r} \| _ {\pi} ^ {2}.
$$

Delta errors $\Delta_{AT}^{2}$ and $\Delta_{GP}^{2}$ measure the closeness of $g_{GP}, \widehat{g}_{a}$ from $g_{a}$ in $\widehat{D}_{a}$ at each iteration.

Theorem 4.5 (Error Analysis of GP). When the model dimension $m \to \infty$ , for an epoch $t$ , we have an approximation of the error difference $\Delta_{AT}^2 - \Delta_{GP}^2$ as follows

$$
\Delta_ {A T} ^ {2} - \Delta_ {G P} ^ {2} \approx \beta (2 - \beta) \mathbb {E} _ {\widehat {\mathcal {D}} _ {a} ^ {t}} \| g _ {a} - \widehat {g _ {a}} \| _ {\pi} ^ {2} - \beta^ {2} \bar {\tau} ^ {2} \| g _ {a} - \widehat {g _ {n}} \| _ {\pi} ^ {2}
$$

$\bar{\tau}^2 = \mathbb{E}_\pi [\tau^2 ]\in [0,1],$ where $\tau (\theta)$ is the $\sin (\cdot)$ value of the angle between $\widehat{g}_n$ and $g_{a} - \widehat{g}_{n}$ .

Theorem 4.5 shows $\Delta_{GP}^2$ is generally smaller than $\Delta_{AT}^2$ for a large model dimension during each iteration, as is the case for the models in our evaluation, with $\beta = 0.5$ , since $\beta(1 - \beta) > \beta^2(0.75 > 0.25)$ and the small value of $\bar{\tau}$ in practice (see Interpretation of Theorem A.2 in Appendix A, where we show the order of difference is between $1e^{-8}$ and $1e^{-12}$ ). Thus, GP achieves better robust accuracy than AT by achieving a smaller delta error; GP also obtains good clean accuracy by combining parts of the model updates from the clean distribution $\widehat{\mathcal{D}}_n$ . Further, we provide an error analysis of a single gradient step in Theorem A.1 and convergence analysis in Theorem A.2, showing that a smaller Delta error results in better convergence. The full proof of all theorems is in Appendix A.

We outline the AT-GP method in Algorithm 2 and it can be extended to the multiple-norm scenario. The overhead of this algorithm comes from natural training and GP operation. Their costs are small, and we discuss this more in Section 5.2. Combining logit pairing and gradient projection methods, we provide the RAMP framework which is similar to Algorithm 2, except that we replace line 4 of Algorithm 2 as Algorithm 1 line 3-9.

# 5 Experiment

Datasets, baselines, and models. CIFAR-10 [Krizhevsky et al., 2009] includes 60K images with 50K and 10K images for training and testing respectively. ImageNet has $\approx$ 14.2M images and 1K classes, containing $\approx$ 1.3M training, 50K validation, and 100K test images [Russakovsky et al.,

![](images/1737de90e2475f43cbb0394d6dfbfe0d2164c149fb5185b09239d76c17985c15.jpg)

<details>
<summary>line</summary>

| Epochs | AT     | AT-GP  | AT-GP-pre | AT-pre |
| ------ | ------ | ------ | --------- | ------ |
| 0      | 0.700  | 0.700  | 0.700     | 0.700  |
| 20     | 0.750  | 0.740  | 0.760     | 0.730  |
| 40     | 0.800  | 0.790  | 0.810     | 0.780  |
| 60     | 0.850  | 0.840  | 0.860     | 0.830  |
| 80     | 0.860  | 0.850  | 0.870     | 0.840  |
| 100    | 0.865  | 0.855  | 0.875     | 0.845  |
| 120    | 0.870  | 0.860  | 0.880     | 0.850  |
</details>

(a) Clean Accuracy

![](images/65b599355069b83cdfae87cee983896a1aae85209210c8e11a16aa72035e9147.jpg)

<details>
<summary>line</summary>

| Epochs | AT    | AT-GP | AT-GP-pre | AT-pre |
| ------ | ----- | ----- | --------- | ------ |
| 0      | 0.20  | 0.20  | 0.20      | 0.20   |
| 20     | 0.45  | 0.48  | 0.49      | 0.47   |
| 40     | 0.48  | 0.50  | 0.51      | 0.49   |
| 60     | 0.47  | 0.51  | 0.52      | 0.48   |
| 80     | 0.46  | 0.50  | 0.51      | 0.47   |
| 100    | 0.45  | 0.49  | 0.50      | 0.46   |
| 120    | 0.44  | 0.48  | 0.49      | 0.45   |
</details>

(b) Robust Accuracy: PGD-20   
Figure 2: $l_{\infty}$ AT-GP with PGD [Madry et al., 2017] with $\epsilon = 0.031$ on CIFAR-10 improves accuracy and robustness. Pre-training on $\widehat{D}_{n}$ for 50 epochs further boosts the performance.

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td></tr><tr><td>RN-18  $l_{\infty}$ </td><td>84.2</td><td>47.4</td></tr><tr><td>RN-18  $l_{\infty}$ -GP</td><td>84.5</td><td>48.3</td></tr><tr><td>RN-18  $l_{\infty}$ -GP-pre</td><td>84.9</td><td>48.3</td></tr></table>

Figure 3: $l_{\infty}$ AT-GP with APGD [Croce and Hein, 2020] improves robustness against $l_{\infty}$ AutoAttack [Croce and Hein, 2020] with $\epsilon = \frac{8}{255}$ . RN-18 $l_{\infty}$ -GP uses AT-GP; RN-18 $l_{\infty}$ -GP-pre pre-trains 40 epochs on $\widehat{D}_{n}$ before AT-GP is applied.

2015]. We compare RAMP with following baselines: 1. SAT [Madaan et al., 2021]: randomly sample one of the $l_{1}$ , $l_{2}$ , $l_{\infty}$ attacks. 2. AVG [Tramer and Boneh, 2019]: take the average of $l_{1}$ , $l_{2}$ , $l_{\infty}$ examples. 3. MAX [Tramer and Boneh, 2019]: take the worst of $l_{1}$ , $l_{2}$ , $l_{\infty}$ attacks. 4. MSD [Maini et al., 2020]: find the worst-case examples over $l_{1}$ , $l_{2}$ , $l_{\infty}$ steepest descent directions during each step of inner maximization. 5. E-AT [Croce and Hein, 2022]: randomly sample between $l_{1}$ , $l_{\infty}$ attacks. For models, we use PreAct-ResNet-18, ResNet-50, WideResNet-34-20, and WideResNet-70-16 for CIFAR-10, as well as ResNet-50 and XCiT-S transformer for ImageNet.

Implementations and Evaluation. For AT from scratch for CIFAR-10, we train PreAct ResNet-18 [He et al., 2016] with a lr = 0.05 for 70 epochs and 0.005 for 10 more epochs. We set $\lambda = 2$ , $\beta = 0.5$ for training from scratch, and $\lambda = 0.5$ for robust fine-tuning. For all methods, we use 10 steps for the inner maximization in AT. For ImageNet, we perform 1 epoch of fine-tuning and use a learning rate lr = 0.005, $\lambda = 0.5$ for ResNet-50 and $lr = 1e^{-4}$ , $\lambda = 0.5$ for XCiT-S models. We reduce the rate by a factor of 10 every $\frac{1}{3}$ of the training epoch and set the weight decay to $1e^{-4}$ . We use APGD with 5 steps for $l_{\infty}$ and $l_{2}$ , 15 steps for $l_{1}$ . Settings are similar to [Croce and Hein, 2022]. We use the standard values of $\epsilon_{1} = 12$ , $\epsilon_{2} = 0.5$ , $\epsilon_{\infty} = \frac{8}{255}$ for CIFAR-10 and $\epsilon_{1} = 255$ , $\epsilon_{2} = 2$ , $\epsilon_{\infty} = \frac{4}{255}$ for ImageNet. We focus on $l_{\infty}$ -AT models for fine-tuning, as Croce and Hein [2022] shows their higher union accuracy for the $\epsilon$ values in our evaluation. We report the clean accuracy, robust accuracy against $\{l_{1}, l_{2}, l_{\infty}\}$ attacks, union accuracy, universal robustness against common corruptions and unseen adversaries, as well as runtime for RAMP. The robust accuracy is evaluated using Autoattack [Croce and Hein, 2020]. More implementation details are in Appendix B.

# 5.1 Main Results

Table 1: Different epsilon values: RAMP consistently outperforms E-AT and MAX for both training from scratch and robust fine-tuning when the key tradeoff pair changes. 

<table><tr><td rowspan="2" colspan="2"></td><td colspan="5">(12, 0.5,  $\frac{2}{255}$ )</td><td colspan="5">(12, 1.5,  $\frac{8}{255}$ )</td></tr><tr><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td rowspan="3">Training from Scratch</td><td>E-AT</td><td>87.2</td><td>73.3</td><td>64.1</td><td>55.4</td><td>55.4</td><td>83.5</td><td>41.0</td><td>25.5</td><td>52.9</td><td>25.5</td></tr><tr><td>MAX</td><td>85.6</td><td>72.1</td><td>63.6</td><td>56.4</td><td>56.4</td><td>74.6</td><td>42.9</td><td>35.7</td><td>50.3</td><td>35.6</td></tr><tr><td>RAMP</td><td>86.3</td><td>73.3</td><td>64.9</td><td>59.1</td><td>59.1</td><td>74.4</td><td>43.4</td><td>37.2</td><td>51.1</td><td>37.1</td></tr><tr><td rowspan="3">Robust Fine-tuning</td><td>E-AT</td><td>86.5</td><td>74.8</td><td>66.7</td><td>57.9</td><td>57.9</td><td>80.2</td><td>42.8</td><td>31.5</td><td>52.4</td><td>31.5</td></tr><tr><td>MAX</td><td>85.7</td><td>74.0</td><td>66.2</td><td>60.0</td><td>60.0</td><td>74.8</td><td>43.8</td><td>36.7</td><td>50.2</td><td>36.6</td></tr><tr><td>RAMP</td><td>85.8</td><td>74.0</td><td>66.2</td><td>60.1</td><td>60.1</td><td>74.9</td><td>43.7</td><td>37.0</td><td>50.2</td><td>36.9</td></tr></table>

Robust fine-tuning. In Table 2, we apply RAMP to larger models and datasets (ImageNet). However, the implementation of other baselines is not publicly available and Croce and Hein [2022] do not report other baseline results except E-AT on larger models and datasets, so we only compare against E-AT in Table 2, which shows RAMP consistently obtains better union accuracy and accuracy-robustness tradeoff than E-AT. We observe that RAMP improves the performance more as the model becomes larger. We obtain the SOTA union accuracy of 53.3% on CIFAR-10 and 29.1% on ImageNet.

RAMP with varying $\epsilon_{1},\epsilon_{2},\epsilon_{\infty}$ values. We provide results with 1. ( $\epsilon_{1}=12,\epsilon_{2}=0.5,\epsilon_{\infty}=\frac{2}{255}$ ) where $\epsilon_{\infty}$ size is small and 2. ( $\epsilon_{1}=12,\epsilon_{2}=1.5,\epsilon_{\infty}=\frac{8}{255}$ ) where $\epsilon_{2}$ size is large, using PreAct ResNet-18 model for CIFAR-10 dataset: these cases have different tradeoff pair compared to

Table 2: Robust fine-tuning on larger models and datasets (\* uses extra data for pre-training). We evaluate all CIFAR-10 and Imagenet test points. RAMP consistently achieves better union accuracy with significant margins and good accuracy-robustness tradeoff. 

<table><tr><td></td><td>Models</td><td>Methods</td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td rowspan="10">CIFAR-10</td><td rowspan="2">WRN-70-16- $l_{\infty}$ (*) [Gowal et al., 2020]</td><td>E-AT</td><td>89.6</td><td>54.4</td><td>76.7</td><td>58.0</td><td>51.6</td></tr><tr><td>RAMP</td><td>90.6</td><td>54.7</td><td>74.6</td><td>57.9</td><td>53.3</td></tr><tr><td rowspan="2">WRN-34-20- $l_{\infty}$ [Gowal et al., 2020]</td><td>E-AT</td><td>87.8</td><td>49.0</td><td>71.6</td><td>49.8</td><td>45.1</td></tr><tr><td>RAMP</td><td>87.1</td><td>49.7</td><td>70.8</td><td>50.4</td><td>46.9</td></tr><tr><td rowspan="2">WRN-28-10- $l_{\infty}$ (*) [Carmon et al., 2019]</td><td>E-AT</td><td>89.3</td><td>51.8</td><td>74.6</td><td>53.3</td><td>47.9</td></tr><tr><td>RAMP</td><td>89.2</td><td>55.9</td><td>74.7</td><td>55.7</td><td>52.7</td></tr><tr><td rowspan="2">WRN-28-10- $l_{\infty}$ (*) [Gowal et al., 2020]</td><td>E-AT</td><td>89.8</td><td>54.4</td><td>76.1</td><td>56.0</td><td>50.5</td></tr><tr><td>RAMP</td><td>89.4</td><td>55.9</td><td>74.7</td><td>56.0</td><td>52.9</td></tr><tr><td rowspan="2">RN-50- $l_{\infty}$ [Engstrom et al., 2019]</td><td>E-AT</td><td>85.3</td><td>46.5</td><td>68.3</td><td>45.3</td><td>41.6</td></tr><tr><td>RAMP</td><td>84.3</td><td>47.0</td><td>67.7</td><td>46.5</td><td>43.3</td></tr><tr><td rowspan="4">ImageNet</td><td rowspan="2">XCiT-S- $l_{\infty}$ [Debenedetti and Troncoso—EPFL, 2022]</td><td>E-AT</td><td>68.4</td><td>38.1</td><td>51.8</td><td>23.8</td><td>23.4</td></tr><tr><td>RAMP</td><td>66.0</td><td>35.7</td><td>50.2</td><td>30.0</td><td>29.1</td></tr><tr><td rowspan="2">RN-50- $l_{\infty}$ [Engstrom et al., 2019]</td><td>E-AT</td><td>58.2</td><td>26.9</td><td>39.5</td><td>18.8</td><td>17.8</td></tr><tr><td>RAMP</td><td>55.6</td><td>25.1</td><td>38.3</td><td>22.4</td><td>20.9</td></tr></table>

Figure 1. The pair identified using our heuristic are $l_{1} - l_{2}$ and $l_{2} - l_{\infty}$ . In Table 1, we observe that RAMP consistently outperforms E-AT and MAX with significant margins in union accuracy, when training from scratch and performing robust fine-tuning. In Table 1, when $l_{2}$ is the bottleneck, E-AT obtains a lower union accuracy as it does not leverage $l_{2}$ examples. Similar observations are made across various epsilon values, with RAMP consistently outperforming other baselines, as detailed in Appendix B.4. Appendix B includes more training details/results, and ablation studies. Results for applying the trades loss to RAMP outperforming E-AT are detailed in Appendix B.6. Appendix B.7 presents robust fine-tuning using ResNet-18, where RAMP achieves the highest union accuracy.

Adversarial training from random initialization. Table 3 presents the results of AT from random initialization on CIFAR-10 with PreAct ResNet-18. RAMP has the highest union accuracy with good clean accuracy, which indicates that RAMP can mitigate the tradeoffs among perturbations and robustness/accuracy in this setting. The results for all baselines are from Croce and Hein [2022].

Table 3: RN-18 model trained from random initialization on CIFAR-10 over 5 trials: RAMP achieves the best union robustness and good clean accuracy compared with other baselines. Baseline results are from Croce and Hein [2022].

<table><tr><td>Methods</td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>SAT</td><td>83.9±0.8</td><td>40.7±0.7</td><td>68.0±0.4</td><td>54.0±1.2</td><td>40.4±0.7</td></tr><tr><td>AVG</td><td>84.6±0.3</td><td>40.8±0.7</td><td>68.4±0.7</td><td>52.1±0.4</td><td>40.1±0.8</td></tr><tr><td>MAX</td><td>80.4±0.5</td><td>45.7±0.9</td><td>66.0±0.4</td><td>48.6±0.8</td><td>44.0±0.7</td></tr><tr><td>MSD</td><td>81.1±1.1</td><td>44.9±0.6</td><td>65.9±0.6</td><td>49.5±1.2</td><td>43.9±0.8</td></tr><tr><td>E-AT</td><td>82.2±1.8</td><td>42.7±0.7</td><td>67.5±0.5</td><td>53.6±0.1</td><td>42.4±0.6</td></tr><tr><td>RAMP (λ=5)</td><td>81.2±0.3</td><td>46.0±0.5</td><td>65.8±0.2</td><td>48.3±0.6</td><td>44.6±0.6</td></tr><tr><td>RAMP (λ=2)</td><td>82.1±0.3</td><td>45.5±0.3</td><td>66.6±0.3</td><td>48.4±0.2</td><td>44.0±0.2</td></tr></table>

Table 4: Individual, average, and union accuracy against common corruptions (averaged across five levels) and unseen adversaries using WideResNet-28-10 on CIFAR-10 dataset. 

<table><tr><td>Models</td><td>Common Corruptions</td><td> $l_0$ </td><td>fog</td><td>snow</td><td>gabor</td><td>elastic</td><td>jpeginf</td><td>Avg</td><td>Union</td></tr><tr><td> $l_{1}$ -AT</td><td>78.2</td><td>79.0</td><td>41.4</td><td>22.9</td><td>40.5</td><td>48.9</td><td>48.4</td><td>46.9</td><td>12.8</td></tr><tr><td> $l_{2}$ -AT</td><td>77.2</td><td>67.5</td><td>48.7</td><td>26.1</td><td>44.1</td><td>53.2</td><td>45.4</td><td>47.5</td><td>16.2</td></tr><tr><td> $l_\infty$ -AT</td><td>73.4</td><td>55.5</td><td>44.7</td><td>32.9</td><td>53.8</td><td>56.6</td><td>33.4</td><td>46.2</td><td>19.1</td></tr><tr><td>Winninghand [Diffenderfer et al., 2021]</td><td>91.1</td><td>74.1</td><td>74.5</td><td>18.3</td><td>76.5</td><td>12.6</td><td>0.0</td><td>42.7</td><td>0.0</td></tr><tr><td>E-AT</td><td>71.5</td><td>58.5</td><td>35.9</td><td>35.3</td><td>50.7</td><td>55.7</td><td>60.3</td><td>49.4</td><td>21.9</td></tr><tr><td>MAX</td><td>71.0</td><td>56.2</td><td>42.9</td><td>35.4</td><td>49.8</td><td>57.8</td><td>55.7</td><td>49.6</td><td>24.4</td></tr><tr><td>RAMP</td><td>75.5</td><td>55.5</td><td>40.5</td><td>40.2</td><td>52.9</td><td>60.3</td><td>56.1</td><td>50.9</td><td>26.1</td></tr></table>

# Universal Robustness. In Table 4, we

report average accuracy against common corruptions and union accuracy against unseen adversaries from Laidlaw et al. [2020] (implementation details are in Appendix B.3). We compare against $l_{p}$ pretrained models, E-AT, MAX, winninghand [Diffenderfer et al., 2021] (a SOTA method for natural corruptions) using WideResNet-28-10 architecture on the CIFAR-10 dataset. Compared to E-AT and MAX, RAMP achieves 4% higher accuracy for common corruptions with five severity levels and 2-4% better union accuracy against multiple unseen adversaries. Winninghand has high corruption robustness but no adversarial robustness. The results show that RAMP obtains a better robustness and accuracy tradeoff with stronger universal robustness. In Appendix B.3, we evaluate on ResNet-18 to support this fact further.

# 5.2 Ablation Study and Discussion

Sensitivities of $\lambda$ . We perform experiments with different $\lambda$ values in [0.1, 0.5, 1.0, 1.5, 2, 3, 4, 5] for robust fine-tuning and [1.5, 2, 3, 4, 5, 6] for AT from scratch using PreAct-ResNet-18 model for CIFAR-10 dataset. In Figure 4, we observe a decreased clean accuracy when $\lambda$ becomes larger. We

![](images/2d82c8545991b8dda623482fd5b2498c26af88e330c9296ea8333db54af65fb2.jpg)

<details>
<summary>line</summary>

| λ    | Clean | Linf  | L2    | L1    | Union |
| ---- | ----- | ----- | ----- | ----- | ----- |
| 0    | 81.7  | 43.4  | 65.0  | 43.4  | 43.4  |
| 1    | 81.2  | 43.4  | 65.0  | 43.4  | 43.4  |
| 2    | 80.8  | 43.4  | 65.0  | 43.4  | 43.4  |
| 3    | 80.2  | 43.4  | 65.0  | 43.4  | 43.4  |
| 4    | 79.9  | 43.4  | 65.0  | 43.4  | 43.4  |
| 5    | 78.9  | 43.4  | 65.0  | 43.4  | 43.4  |
</details>

(a) $\lambda$ : Robust fine-tuning.

![](images/326dd84b1fb8d75dffca425ac9fe125ed39a18c7a08fd97eaf4501ff9987234c.jpg)

<details>
<summary>line</summary>

| λ | Clean | Linf | L2 | L1 | Union |
| --- | --- | --- | --- | --- | --- |
| 2 | 82.3 | 44.4 | 67.0 | 44.7 | 44.5 |
| 3 | 81.2 | 44.4 | 67.0 | 45.0 | 44.5 |
| 4 | 81.3 | 44.6 | 67.0 | 45.0 | 44.5 |
| 5 | 81.2 | 44.6 | 67.0 | 45.0 | 44.5 |
| 6 | 80.8 | 44.5 | 67.0 | 45.0 | 44.5 |
</details>

(b) $\lambda$ : Train from scratch.

![](images/9b274999199d53d00b0c1b892a28585f0a8c1a2365efe37adcb132dc6afa980e.jpg)

<details>
<summary>line</summary>

| β    | Clean | Linf  | L2   | L1   | Union |
| ---- | ----- | ----- | ---- | ---- | ----- |
| 0.2  | 82.1  | 43.0  | 67.0 | 50.0 | 43.0  |
| 0.4  | 82.0  | 44.4  | 67.0 | 50.0 | 44.4  |
| 0.6  | 81.4  | 44.4  | 67.0 | 50.0 | 44.4  |
| 0.8  | 79.7  | 42.9  | 65.0 | 50.0 | 42.9  |
</details>

(c) $\beta$ : Train from scratch.   
Figure 4: Alabtion studies on $\lambda$ and $\beta$ hyper-parameters.

pick $\lambda = 2.0$ for training from scratch (Figure 4a) and $\lambda = 0.5$ for robust fine-tuning (Figure 4b) in our main experiments, as these values of $\lambda$ yield both good clean and union accuracy.

Choices of $\beta$ . Figure 4c shows the performance of RAMP with varying $\beta$ values on CIFAR-10 ResNet-18 experiments. We pick $\beta = 0.5$ for combining natural training and AT via GP, which achieves comparatively good robustness and clean accuracy. This choice is also based on Theorem 4.5 when $\beta(2 - \beta)$ has the largest difference from $\beta^{2}$ (0.75 vs 0.25).

Fine-tune $l_{p}$ AT models with RAMP. Table 5 shows the robust fine-tuning results using RAMP with $l_{\infty}$ -AT ( $q = \infty, r = 1$ ), $l_{1}$ -AT ( $q = 1, r = \infty$ ), $l_{2}$ -AT ( $q = \infty, r = 1$ ) RN-18 models for CIFAR-10 dataset. For $l_{\infty} - l_{1}$ tradeoffs, RAMP on $l_{\infty}$ -AT pre-trained model achieves the best union accuracy.

Computational analysis and Limitations. The extra training costs of AT-GP are small, e.g. for each epoch on ResNet-18, the extra NT takes 6 seconds and the standard AT takes 78 seconds using a single NVIDIA A100 GPU, and the GP operation only takes 0.04 seconds on average. RAMP is more expensive than E-AT and less expensive than MAX. We have a complete runtime analysis in Appendix B.2. We notice occasional drops in clean accuracy during fine-tuning with RAMP. In some cases, union accuracy improves slightly but clean accuracy and single $l_{p}$ robustness reduce. Further, we find no negative societal impact from this work.

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td></tr><tr><td>RN-18  $l_{\infty}$ -AT</td><td>81.5</td><td>45.5</td><td>66.4</td><td>47.0</td><td>42.9</td></tr><tr><td>RN-18  $l_{1}$ -AT</td><td>81.0</td><td>42.6</td><td>66.0</td><td>48.1</td><td>41.5</td></tr><tr><td>RN-18  $l_{2}$ -AT</td><td>84.1</td><td>41.6</td><td>69.1</td><td>45.4</td><td>39.4</td></tr></table>

Table 5: RAMP with $l_{\infty}$ , $l_{1}$ , $l_{2}$ -RN-18-AT models on CIFAR-10 with standard epsilons.

# 6 Conclusion

We introduce RAMP, a framework enhancing multiple-norm robustness and achieving superior universal robustness against corruptions and perturbations by addressing tradeoffs among $l_{p}$ perturbations and accuracy/robustness. We apply a new logit pairing loss and use gradient projection to obtain SOTA union accuracy with favorable accuracy/robustness tradeoffs against common corruptions and other unseen adversaries. Results demonstrate that RAMP surpasses SOTA methods in union accuracy across model architectures on CIFAR-10 and ImageNet.

# References

Kumail Alhamoud, Hasan Abed Al Kader Hammoud, Motasem Alfarra, and Bernard Ghanem. Generalizability of adversarial robustness under distribution shifts. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=XNFo3dQiCJ. Featured Certification.   
Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok. Synthesizing robust adversarial examples. In International conference on machine learning, pages 284–293. PMLR, 2018.   
Philipp Benz, Chaoning Zhang, and In So Kweon. Batch normalization increases adversarial vulnerability and decreases adversarial transferability: A non-robust feature perspective. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7818–7827, 2021.

Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, John C Duchi, and Percy S Liang. Unlabeled data improves adversarial robustness. Advances in neural information processing systems, 32, 2019.   
Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning, pages 2206–2216. PMLR, 2020.   
Francesco Croce and Matthias Hein. Adversarial robustness against multiple and single $l_{p}$ -threat models via quick fine-tuning of robust classifiers. In International Conference on Machine Learning, pages 4436–4454. PMLR, 2022.   
Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. arXiv preprint arXiv:2010.09670, 2020.   
Francesco Croce, Maksym Andriushchenko, Naman D Singh, Nicolas Flammarion, and Matthias Hein. Sparse-rs: a versatile framework for query-efficient sparse black-box adversarial attacks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 6437–6445, 2022.   
Edoardo Debenedetti and Carmela Troncoso—EPFL. Adversarially robust vision transformers, 2022.   
James Diffenderfer, Brian Bartoldson, Shreya Chaganti, Jize Zhang, and Bhavya Kailkhura. A winning hand: Compressing deep networks can improve out-of-distribution robustness. Advances in neural information processing systems, 34:664–676, 2021.   
Logan Engstrom, Brandon Tran, Dimitris Tsipras, Ludwig Schmidt, and Aleksander Madry. A rotation and a translation suffice: Fooling cnns with simple transformations. 2017.   
Logan Engstrom, Andrew Ilyas, and Anish Athalye. Evaluating and understanding the robustness of adversarial logit pairing. arXiv preprint arXiv:1807.10272, 2018.   
Logan Engstrom, Andrew Ilyas, Hadi Salman, Shibani Santurkar, and Dimitris Tsipras. Robustness (python library), 2019. URL https://github.com/MadryLab/robustness.   
Sanmi Koyejo Enyi Jiang, Yibo Jacky Zhang. Principled federated domain adaptation: Gradient projection and auto-weighting. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=6J3ehSUrMU.   
Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1625–1634, 2018.   
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.   
Sven Gowal, Chongli Qin, Jonathan Uesato, Timothy Mann, and Pushmeet Kohli. Uncovering the limits of adversarial training against norm-bounded adversarial examples. arXiv preprint arXiv:2010.03593, 2020.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.   
Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019.   
Enyi Jiang. Federated domain adaptation for healthcare, 2023.   
Enyi Jiang, Yibo Jacky Zhang, and Oluwasanmi Koyejo. Federated domain adaptation via gradient projection. arXiv preprint arXiv:2302.05049, 2023.   
Daniel Kang, Yi Sun, Tom Brown, Dan Hendrycks, and Jacob Steinhardt. Transfer of adversarial robustness between perturbation types. arXiv preprint arXiv:1905.01034, 2019.

Harini Kannan, Alexey Kurakin, and Ian Goodfellow. Adversarial logit pairing. arXiv preprint arXiv:1803.06373, 2018.   
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.   
Alexey Kurakin, Ian J Goodfellow, and Samy Bengio. Adversarial examples in the physical world. In Artificial intelligence safety and security, pages 99–112. Chapman and Hall/CRC, 2018.   
Cassidy Laidlaw, Sahil Singla, and Soheil Feizi. Perceptual adversarial robustness: Defense against unseen threat models. arXiv preprint arXiv:2006.12655, 2020.   
Aishan Liu, Shiyu Tang, Xianglong Liu, Xinyun Chen, Lei Huang, Zhuozhuo Tu, Dawn Song, and Dacheng Tao. Towards defending multiple adversarial perturbations via gated batch normalization. arXiv preprint arXiv:2012.01654, 2020.   
Divyam Madaan, Jinwoo Shin, and Sung Ju Hwang. Learning to generate noise for multi-attack robustness. In International Conference on Machine Learning, pages 7279–7289. PMLR, 2021.   
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.   
Pratyush Maini, Eric Wong, and Zico Kolter. Adversarial robustness against the union of multiple perturbation models. In International Conference on Machine Learning, pages 6640–6650. PMLR, 2020.   
Pratyush Maini, Xinyun Chen, Bo Li, and Dawn Song. Perturbation type categorization for multiple adversarial perturbation robustness. In Uncertainty in Artificial Intelligence, pages 1317–1327. PMLR, 2022.   
Mohammad Mehrabi, Adel Javanmard, Ryan A Rossi, Anup Rao, and Tung Mai. Fundamental tradeoffs in distributionally adversarial training. In International Conference on Machine Learning, pages 7544–7554. PMLR, 2021.   
Mazda Moayeri, Kiarash Banihashem, and Soheil Feizi. Explicit tradeoffs between adversarial and natural distributional robustness. Advances in Neural Information Processing Systems, 35:38761–38774, 2022.   
Jay Nandy, Wynne Hsu, and Mong Li Lee. Approximate manifold defense against multiple adversarial perturbations. In 2020 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2020.   
Sheng Yun Peng, Weilin Xu, Cory Cornelius, Matthew Hull, Kevin Li, Rahul Duggal, Mansi Phute, Jason Martin, and Duen Horng Chau. Robust principles: Architectural design principles for adversarially robust cnns. arXiv preprint arXiv:2308.16258, 2023.   
Rahul Rade and Seyed-Mohsen Moosavi-Dezfooli. Reducing excessive margin to achieve a better accuracy vs. robustness trade-off. In International Conference on Learning Representations, 2021.   
Aditi Raghunathan, Sang Michael Xie, Fanny Yang, John Duchi, and Percy Liang. Understanding and mitigating the tradeoff between robustness and accuracy. arXiv preprint arXiv:2002.10716, 2020.   
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015.   
Lukas Schott, Jonas Rauber, Matthias Bethge, and Wieland Brendel. Towards the first adversarially robust neural network model on mnist. arXiv preprint arXiv:1805.09190, 2018.   
Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer, and Michael K Reiter. Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition. In Proceedings of the 2016 acm sigsac conference on computer and communications security, pages 1528–1540, 2016.

Aman Sinha, Hongseok Namkoong, and John Duchi. Certifiable distributional robustness with principled adversarial training. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=Hk6kPgZA-.   
Dawn Song, Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Florian Tramer, Atul Prakash, and Tadayoshi Kohno. Physical adversarial examples for object detectors. In 12th USENIX workshop on offensive technologies (WOOT 18), 2018.   
Florian Tramer and Dan Boneh. Adversarial training and robustness for multiple perturbations. Advances in neural information processing systems, 32, 2019.   
Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. Ensemble adversarial training: Attacks and defenses. arXiv preprint arXiv:1705.07204, 2017.   
Yisen Wang, Difan Zou, Jinfeng Yi, James Bailey, Xingjun Ma, and Quanquan Gu. Improving adversarial robustness requires revisiting misclassified examples. In ICLR, 2020.   
Zekai Wang, Tianyu Pang, Chao Du, Min Lin, Weiwei Liu, and Shuicheng Yan. Better diffusion models further improve adversarial training. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 36246–36263. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/wang23ad.html.   
Wikipedia contributors. Volume of an n-ball — Wikipedia, the free encyclopedia, 2024. URL https://en.wikipedia.org/w/index.php?title=Volume\_of\_an\_n-ball&oldid=1216556293. [Online; accessed 11-May-2024].   
Eric Wong and J Zico Kolter. Learning perturbation sets for robust machine learning. arXiv preprint arXiv:2007.08450, 2020.   
Dongxian Wu, Shu-Tao Xia, and Yisen Wang. Adversarial weight perturbation helps robust generalization. Advances in Neural Information Processing Systems, 33:2958–2969, 2020.   
Jiancong Xiao, Zeyu Qin, Yanbo Fan, Baoyuan Wu, Jue Wang, and Zhi-Quan Luo. Adaptive smoothness-weighted adversarial training for multiple perturbations with its stability analysis. arXiv preprint arXiv:2210.00557, 2022.   
Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang, Alan L Yuille, and Quoc V Le. Adversarial examples improve image recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 819–828, 2020.   
Kaidi Xu, Chenan Wang, Hao Cheng, Bhavya Kailkhura, Xue Lin, and Ryan Goldhahn. Mixture of robust experts (more): A robust denoising method towards multiple perturbations. arXiv preprint arXiv:2104.10586, 2021.   
Yao-Yuan Yang, Cyrus Rashtchian, Hongyang Zhang, Russ R Salakhutdinov, and Kamalika Chaudhuri. A closer look at accuracy vs. robustness. Advances in neural information processing systems, 33:8588–8601, 2020.   
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In Proceedings of the British Machine Vision Conference 2016. British Machine Vision Association, 2016.   
Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael Jordan. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pages 7472–7482. PMLR, 2019.   
Jingfeng Zhang, Jianing Zhu, Gang Niu, Bo Han, Masashi Sugiyama, and Mohan Kankanhalli. Geometry-aware instance-reweighted adversarial training. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=iAX016Cz8ub.

# A Proof of Theorems

# A.1 Proof of Theorem A.2

We first show what happens during one step of optimization, where we highlight the importance of analyzing delta error.

Theorem A.1. Consider model parameter $\theta \sim \pi$ and an aggregation rule $Aggr(\cdot)$ with step size $\mu > 0$ . Define the updated parameter as

$$
\theta^ {+} := \theta - \mu \widehat {g} _ {A g g r} (\theta).
$$

Assuming the gradient $\nabla \mathcal{L}(\theta)$ is $\gamma$ -Lipschitz in $\theta$ for any input, and let the step size $\mu \leq \frac{1}{\gamma}$ , we have

$$
\mathbb {E} _ {\widehat {\mathcal {D}} _ {a}, \theta} [ \mathcal {L} _ {\mathcal {D} _ {a}} (\theta^ {+}) - \mathcal {L} _ {\mathcal {D} _ {a}} (\theta) ] \leq - \frac {\mu}{2} (\| g _ {a} \| _ {\pi} ^ {2} - \Delta_ {A g g r} ^ {2}).
$$

Proof. The proof is the same as Theorem A.1 in [Enyi Jiang, 2024].

![](images/2f54005476742e0a4c3898c117f5775640c6d7998933726b5409e2effb08f1f7.jpg)

Theorem A.2 (Convergence of $\mathrm{Aggr}(\cdot)$ ). For any probability measure $\pi$ over the parameter space, and an aggregation rule $\mathrm{Aggr}(\cdot)$ with step size $\mu > 0$ . We update the parameter for $T$ steps by $\theta^{t+1} := \theta^t - \mu \widehat{g}_{\mathrm{Aggr}}(\theta^t)$ . Assume the gradient $\nabla \mathcal{L}(\theta)$ and $\widehat{g}_{\mathrm{Aggr}}(\theta)$ are $\frac{\gamma}{2}$ -Lipschitz in $\theta$ such that $\theta^t \to \widehat{\theta}_{\mathrm{Aggr}}$ . $\Delta_{\mathrm{Aggr\_max}}$ is the Delta error at time $t'$ when $\| \widehat{g}_{\mathrm{Aggr}}(\widehat{\theta}_{\mathrm{Aggr}}) - \nabla \mathcal{L}_{\mathcal{D}_a^{t'}}(\widehat{\theta}_{\mathrm{Aggr}}) \|^2$ is maximized. Then, given step size $\mu \leq \frac{1}{\gamma}$ and a small enough $\epsilon > 0$ , with probability at least $1 - \delta$ we have

$$
\| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {T}} (\theta^ {T}) \| ^ {2} \leq \frac {1}{\delta^ {2}} \left(\sqrt {C _ {\epsilon} \cdot \Delta_ {A g g r \_ m a x} ^ {2}} + \mathcal {O} (\epsilon)\right) ^ {2} + \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon),
$$

where $C_{\epsilon} = \mathbb{E}_{\widehat{\mathcal{D}}_a^{t'}}[1 / \pi (B_\epsilon (\widehat{\theta}_{Aggr}))]^2$ and $B_{\epsilon}(\widehat{\theta}_{Aggr})\subset \mathbb{R}^m$ is the ball with radius $\epsilon$ centered at $\widehat{\theta}_{Aggr}$ . The $C_{\epsilon}$ measures how well $\pi$ covers where the optimization goes.

Proof. Denote random function $\widehat{f}:\mathbb{R}^m\to \mathbb{R}_+$ as

$$
\widehat {f} (\theta) = \left\| \widehat {g} _ {\mathrm{Aggr}} (\theta) - \nabla \mathcal {L} _ {\mathcal {D} _ {a}} (\theta) \right\|, \tag {8}
$$

where the randomness comes from $\widehat{D}_{a}$ . Note that $\widehat{f}$ is $\gamma$ -Lipschitz by assumption. Now we consider $B_{\epsilon}(\widehat{\theta}_{\mathrm{Aggr}}) \subset \mathbb{R}^{m}$ , i.e., the ball with radius $\epsilon$ centered at $\widehat{\theta}_{Aggr}$ . Then, by $\gamma$ -Lipschitzness we have

$$
\mathbb {E} _ {\theta \sim \pi} \widehat {f} (\theta) = \int \widehat {f} (\theta) \mathrm{d} \pi (\theta)
$$

$$
\geq \int_ {B _ {\epsilon} (\widehat {\theta} _ {\mathrm{Aggr}})} (\widehat {f} (\widehat {\theta} _ {\mathrm{Aggr}}) - \gamma \epsilon) \mathrm{d} \pi (\theta)
$$

$$
= (\widehat {f} (\widehat {\theta} _ {\mathrm{Aggr}}) - \gamma \epsilon) \pi (B _ {\epsilon} (\widehat {\theta} _ {\mathrm{Aggr}}))
$$

Therefore,

$$
\widehat {f} (\widehat {\theta} _ {\mathrm{Aggr}}) \leq \frac {1}{\pi (B _ {\epsilon} (\widehat {\theta} _ {\mathrm{Aggr}}))} \cdot \mathbb {E} _ {\theta \sim \pi} \widehat {f} (\theta) + \mathcal {O} (\epsilon).
$$

Taking expectation w.r.t. $\widehat{\mathcal{D}}_a$ on both sides, we have

$$
\begin{array}{l} \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \widehat {f} (\widehat {\theta} _ {\mathrm{Aggr}}) \leq \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \left[ \frac {1}{\pi (B _ {\epsilon} (\widehat {\theta} _ {\mathrm{Aggr}}))} \cdot \mathbb {E} _ {\theta \sim \pi} \widehat {f} (\theta) \right] + \mathcal {O} (\epsilon) \\ \leq \sqrt {\mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \left[ \frac {1}{\pi (B _ {\epsilon} (\widehat {\theta} _ {\mathrm{Aggr}}))} \right] ^ {2} \cdot \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \left[ \mathbb {E} _ {\theta \sim \pi} \widehat {f} (\theta) \right] ^ {2}} + \mathcal {O} (\epsilon) \quad \text {(Cauchy - Schwarz)} \\ = \sqrt {C _ {\epsilon} \cdot \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \left[ \mathbb {E} _ {\theta \sim \pi} \widehat {f} (\theta) \right] ^ {2}} + \mathcal {O} (\epsilon) \quad \text {(by definition of C_{\epsilon})} \\ \leq \sqrt {C _ {\epsilon} \cdot \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \mathbb {E} _ {\theta \sim \pi} \left[ \widehat {f} (\theta) \right] ^ {2}} + \mathcal {O} (\epsilon) \tag {$Jensen^{\prime$} s i n e q u a l i t y} \\ = \sqrt {C _ {\epsilon} \cdot \Delta_ {\mathrm{Aggr}} ^ {2}} + \mathcal {O} (\epsilon) \\ \end{array}
$$

By Markov's inequality, with probability at least $1 - \delta$ we have a sampled dataset $\widehat{\mathcal{D}}_a$ such that

$$
\widehat {f} \left(\widehat {\theta} _ {\mathrm{Aggr}}\right) \leq \frac {1}{\delta} \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \widehat {f} \left(\widehat {\theta} _ {\mathrm{Aggr}}\right) \leq \frac {1}{\delta} \sqrt {C _ {\epsilon} \cdot \Delta_ {\mathrm{Aggr}} ^ {2}} + \mathcal {O} (\epsilon / \delta) \tag {9}
$$

Conditioned on such event, we proceed on to the optimization part.

Note that Theorem A.1 characterizes how the optimization works for one gradient update. We denote $D_{a}^{t}$ as the data distribution $D_{a}$ at time step t. Therefore, for any time step $t = 0, \ldots, T - 1$ , we can apply Theorem A.1 which only requires the Lipschitz assumption:

$$
\mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t + 1}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \leq - \frac {\mu}{2} \left(\| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2} - \| \widehat {g} _ {\mathrm{Aggr}} (\theta^ {t}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2}\right).
$$

We notice that $\mathcal{D}_a^t$ changes based on $\theta$ s of different time steps. On both sides, to sum over $t = 0, \ldots, T - 1$ , we first consider two terms:

$$
\big (\mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} \big (\theta^ {t + 1} \big) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} \big (\theta^ {t} \big) \big) + \big (\mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} \big (\theta^ {t} \big) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} \big (\theta^ {t - 1} \big) \big)
$$

To compare $\mathcal{L}_{\mathcal{D}_a^t}(\theta^t)$ and $\mathcal{L}_{\mathcal{D}_a^{t - 1}}(\theta^t)$ , since $\mathcal{D}_a^t$ optimizes one more step than $\mathcal{D}_a^{t - 1}$ , we assume $\mathcal{L}_{\mathcal{D}_a^t}(\theta^t)\leq \mathcal{L}_{\mathcal{D}_a^{t - 1}}(\theta^t)$ for $\forall t$ . Therefore, we have:

$$
(\mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t + 1}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t})) + (\mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t - 1})) \geq (\mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t + 1}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t})) + (\mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t - 1}))
$$

Summing up all time steps,

$$
\begin{array}{l} \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t + 1}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {0}} (\theta^ {0}) \leq \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t + 1}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) + \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 2}} (\theta^ {t - 1}) + \dots - \mathcal {L} _ {\mathcal {D} _ {a} ^ {0}} (\theta^ {0}) \\ \leq - \frac {\mu}{2} \left(\sum_ {t = 0} ^ {T - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) \| ^ {2} - \sum_ {t = 0} ^ {T - 1} \| \widehat {g} _ {\mathbb {A g g r}} (\theta^ {t}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2}\right). \\ \end{array}
$$

Dividing both sides by $T$ , and with regular algebraic manipulation we derive

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) \| ^ {2} \leq \frac {2}{\mu T} (\mathcal {L} _ {\mathcal {D} _ {a} ^ {0}} (\theta^ {0}) - \mathcal {L} _ {\mathcal {D} _ {a} ^ {T - 1}} (\theta^ {T})) + \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \| \widehat {g} _ {\mathrm{Aggr}} (\theta^ {t}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2}.
$$

Note that we assume the loss function $\mathcal{L}_{\mathcal{D}}(\theta):=\mathbb{E}_{(x,y)\sim\mathcal{D}}\mathcal{L}(f(x),y)$ , is non-negative. Thus, we have

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) \| ^ {2} \leq \frac {2 \mathcal {L} _ {\mathcal {D} _ {a} ^ {0}} (\theta^ {0})}{\mu T} + \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \| \widehat {g} _ {\mathrm{Aggr}} (\theta^ {t}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2}. \tag {10}
$$

Note that we assume given $\widehat{\mathcal{D}}_a$ we have $\theta^t\to \widehat{\theta}_{\mathrm{Aggr}}$ . Therefore, for any $\epsilon >0$ there exist $T_{\epsilon}$ such that

$$
\forall t > T _ {\epsilon}: \| \theta^ {t} - \widehat {\theta} _ {\mathrm{Aggr}} \| <   \epsilon . \tag {11}
$$

This implies that $\forall t > T_{\epsilon}$ :

$$
\mu \| \widehat {g} _ {\mathrm{Aggr}} (\theta^ {t}) \| = \| \theta^ {t + 1} - \widehat {\theta} _ {\mathrm{Aggr}} + \widehat {\theta} _ {\mathrm{Aggr}} - \theta^ {t} \| \leq \| \theta^ {t + 1} - \widehat {\theta} _ {\mathrm{Aggr}} \| + \| \widehat {\theta} _ {\mathrm{Aggr}} - \theta^ {t} \| <   2 \epsilon . \tag {12}
$$

Moreover, (11) also implies $\forall t_1, t_2 > T_\epsilon$ :

$$
\begin{array}{l} \left\| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t _ {1}}} (\theta^ {t _ {1}}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t _ {2}}} (\theta^ {t _ {2}}) \right\| \leq \gamma \| \theta^ {t _ {1}} - \theta^ {t _ {2}} \| ($\gamma$-Lipschitzness) \\ <   2 \epsilon . (13) \\ \end{array}
$$

Now, let's get back to (10). For $\forall T > T_{\epsilon}$ we have

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) \| ^ {2} \leq \frac {2 \mathcal {L} _ {\mathcal {D} _ {a} ^ {0}} (\theta^ {0})}{\mu T} + \frac {1}{T} \sum_ {t = 0} ^ {T _ {\epsilon} - 1} \| \widehat {g} _ {\mathbb {A g g r}} (\theta^ {t}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2} + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \| \widehat {g} _ {\mathbb {A g g r}} (\theta^ {t}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2}
$$

$$
= \mathcal {O} \left(\frac {1}{T}\right) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \| \widehat {g} _ {\mathrm{Aggr}} (\theta^ {t}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2}
$$

$$
= \mathcal {O} \left(\frac {1}{T}\right) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \| \widehat {g} _ {\mathrm{Aggr}} (\theta^ {t}) - \widehat {g} _ {\mathrm{Aggr}} (\widehat {\theta} _ {\mathrm{Aggr}}) + \widehat {g} _ {\mathrm{Aggr}} (\widehat {\theta} _ {\mathrm{Aggr}}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2}
$$

$$
\leq \mathcal {O} \left(\frac {1}{T}\right) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \left(\| \widehat {g} _ {\mathrm{Aggr}} (\theta^ {t}) - \widehat {g} _ {\mathrm{Aggr}} (\widehat {\theta} _ {\mathrm{Aggr}}) \| + \| \widehat {g} _ {\mathrm{Aggr}} (\widehat {\theta} _ {\mathrm{Aggr}}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \|\right) ^ {2}
$$

(triangle inequality)

$$
= \mathcal {O} \left(\frac {1}{T}\right) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \left(\mathcal {O} (\epsilon) + \| \widehat {g} _ {\mathrm{Aggr}} \left(\widehat {\theta} _ {\mathrm{Aggr}}\right) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} \left(\theta^ {t}\right) \|\right) ^ {2} \quad (\text {by (12)})
$$

$$
= \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \left(\| \widehat {g} _ {\mathbb {A g g r}} (\widehat {\theta} _ {\mathbb {A g g r}}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\widehat {\theta} _ {\mathbb {A g g r}}) + \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\widehat {\theta} _ {\mathbb {A g g r}}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \|\right) ^ {2}
$$

$$
\leq \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \left(\| \widehat {g} _ {\text {Aggr}} \left(\widehat {\theta} _ {\text {Aggr}}\right) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} \left(\widehat {\theta} _ {\text {Aggr}}\right) \| + \mathcal {O} (\epsilon)\right) ^ {2} \tag {by(13)}
$$

$$
\leq \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon) + \| \widehat {g} _ {\mathrm{Aggr}} \left(\widehat {\theta} _ {\mathrm{Aggr}}\right) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t ^ {\prime}}} \left(\widehat {\theta} _ {\mathrm{Aggr}}\right) \| ^ {2} \tag {14}
$$

Equation 14 bounds the left hand side with the maximum $\| \widehat{g}_{\mathrm{Aggr}}(\widehat{\theta}_{\mathrm{Aggr}}) - \nabla \mathcal{L}_{\mathcal{D}_a^t}(\widehat{\theta}_{\mathrm{Aggr}})\|^2$ one can get during the optimization steps. Here, we assume at time $t'$ , the largest value is attained. We denote $\Delta_{\mathrm{Aggr\_max}}^2$ as the delta error at time step $t'$ .

Then, we can continue with what we have done at the beginning of the proof of this theorem:

$$
(1 4) = \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon) + f (\widehat {\theta} _ {\mathrm{Aggr}}) ^ {2} \tag {by(8)}
$$

$$
\leq \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon) + \left(\frac {1}{\delta} \sqrt {C _ {\epsilon} \cdot \Delta_ {\text {Aggr\_max}} ^ {2}} + \mathcal {O} (\epsilon / \delta)\right) ^ {2} \tag {by(9))}
$$

Therefore, combining the above we finally have: for $\forall T > T_{\epsilon}$ with probability at least $1 - \delta$ ,

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) \| ^ {2} \leq \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon) + \frac {1}{\delta^ {2}} \left(\sqrt {C _ {\epsilon} \cdot \Delta_ {\mathrm{Aggr} _ {-} \max} ^ {2}} + \mathcal {O} (\epsilon)\right) ^ {2} \tag {15}
$$

To complete the proof, let us investigate the left-hand side.

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t - 1}} (\theta^ {t}) \| ^ {2} = \frac {1}{T} \sum_ {t = 0} ^ {T _ {\epsilon} - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2} + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2} \\ = \mathcal {O} \left(\frac {1}{T}\right) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) \| ^ {2} \\ \geq \mathcal {O} \left(\frac {1}{T}\right) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \left(\| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {t}} (\theta^ {t}) - \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {T}} (\theta^ {T}) \| - \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {T}} (\theta^ {T}) \|\right) ^ {2} \\ \end{array}
$$

(triangle inequality)

$$
= \mathcal {O} \left(\frac {1}{T}\right) + \frac {1}{T} \sum_ {t = T _ {\epsilon}} ^ {T - 1} \left(\mathcal {O} (\epsilon) + \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {T}} (\theta^ {T}) \| ^ {2}\right) \tag {by(13)}
$$

$$
= \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon) + \| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {T}} (\theta^ {T}) \| ^ {2}. \tag {16}
$$

Combining (15) and (16), we finally have

$$
\| \nabla \mathcal {L} _ {\mathcal {D} _ {a} ^ {T}} (\theta^ {T}) \| ^ {2} \leq \mathcal {O} \left(\frac {1}{T}\right) + \mathcal {O} (\epsilon) + \frac {1}{\delta^ {2}} \left(\sqrt {C _ {\epsilon} \cdot \Delta_ {\mathrm{Aggr} _ {-} \max} ^ {2}} + \mathcal {O} (\epsilon)\right) ^ {2},
$$

which completes the proof.

![](images/eda85c0d181b8ee7fb2ba9eb48611375c3c310eeeb513f2517ea518907718e41.jpg)

# A.2 Proof of Theorem 4.5

To prove Theorem 4.5, we first use the following definitions and lemmas from [Enyi Jiang, 2024], to get the delta errors of Gradient Projection (GP) and standard adversarial training (AT):

Definition A.3 (GP Aggregation). Let $\beta \in [0,1]$ be the weight that balances between $\widehat{g}_a$ and $\widehat{g}_n$ . The GP aggregation operation is

$$
G P (\widehat {g} _ {a}, \widehat {g} _ {n}) = \left((1 - \beta) \widehat {g} _ {a} + \beta P r o j _ {+} (\widehat {g} _ {a} | \widehat {g} _ {n})\right).
$$

where $\mathsf{Proj}_{\pm}(\widehat{g}_{a}|\widehat{g}_{n})=\max\{\langle\widehat{g}_{a},\widehat{g}_{n}\rangle,0\}\widehat{g}_{n}/\|\widehat{g}_{n}\|^{2}$ is the operation that projects $\widehat{g}_{a}$ to the positive direction of $\widehat{g}_{n}$ .

Definition A.4 (AT Aggregation). The AT aggregation operation is

$$
A T (\widehat {g} _ {a}) = \widehat {g} _ {a}.
$$

standard AT only leverages the gradient update on $\widehat{\mathcal{D}}_a$ .

Lemma A.5 (Delta Error of GP). Given distributions $\widehat{D}_{a}$ , $D_{a}$ and $\widehat{D}_{n}$ , as well as the model updates $\widehat{g}_{a}$ , $g_{a}$ , $\widehat{g}_{n}$ on these distributions per epoch, we have $\Delta_{GP}^{2}$ as follows

$$
\Delta_ {G P} ^ {2} \approx \left((1 - \beta) ^ {2} + \frac {2 \beta - \beta^ {2}}{m}\right) \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \| g _ {a} - \widehat {g} _ {a} \| _ {\pi} ^ {2} + \beta^ {2} \bar {\tau} ^ {2} \| g _ {a} - \widehat {g} _ {n} \| _ {\pi} ^ {2},
$$

In the above equation, m is the model dimension and $\bar{\tau}^{2} = E_{\pi}[\tau^{2}] \in [0,1]$ where $\tau(\theta)$ is the $\sin(\cdot)$ value of the angle between $\widehat{g}_{n}$ and $g_{a} - \widehat{g}_{n}$ . $\|\cdot\|_{\pi}$ is the $\pi$ -norm over the model parameter space.

Proof. The proof is the same as Theorem 4.4 in Enyi Jiang [2024].

![](images/9d72778969e39b30f174fc8c0db70b6f818ced3aeb3f6d51955b5ce2f777d79f.jpg)

Lemma A.6 (Delta Error of AT). Given distributions $\widehat{D}_{a}$ , $D_{a}$ and $\widehat{D}_{n}$ , as well as the model updates $\widehat{g}_{a}, g_{a}, \widehat{g}_{n}$ on these distributions per epoch, we have $\Delta_{AT}^{2}$ as follows

$$
\Delta_ {A T} ^ {2} = \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \| g _ {a} - \widehat {g} _ {a} \| _ {\pi} ^ {2},
$$

where $\| \cdot \|_{\pi}$ is the $\pi$ -norm over the model parameter space.

Then, we prove Theorem 4.5.

Theorem A.7 (Error Analysis of GP). When the model dimension is large $(m \to \infty)$ at time step t, we have

$$
\Delta_ {A T} ^ {2} - \Delta_ {G P} ^ {2} \approx \beta (2 - \beta) \mathbb {E} _ {\widehat {\mathcal {D}} _ {a} ^ {t}} \| g _ {a} - \widehat {g _ {a}} \| _ {\pi} ^ {2} - \beta^ {2} \bar {\tau} ^ {2} \| g _ {a} - \widehat {g _ {n}} \| _ {\pi} ^ {2}.
$$

$\bar{\tau}^2 = \mathbb{E}_\pi [\tau^2 ]\in [0,1]$ where $\tau$ is the $\sin (\cdot)$ value of the angle between $\widehat{g}_n$ and $g_{a} - \widehat{g}_{n}$ , $\| \cdot \|_{\pi}$ is the $\pi$ -norm over the model parameter space.

Proof. $\Delta_{\mathrm{AT}}^2 -\Delta_{\mathrm{GP}}^2$

$$
\approx \mathbb {E} _ {\widehat {\mathcal {D}} _ {a} ^ {t}} \| g _ {a} - \widehat {g _ {a}} \| _ {\pi} ^ {2} - \left((1 - \beta) ^ {2} + \frac {2 \beta - \beta^ {2}}{m}\right) \mathbb {E} _ {\widehat {\mathcal {D}} _ {a}} \| g _ {a} - \widehat {g _ {a}} \| _ {\pi} ^ {2} - \beta^ {2} \bar {\tau} ^ {2} \| g _ {a} - \widehat {g _ {n}} \| _ {\pi} ^ {2}
$$

$$
= \left(1 - ((1 - \beta) ^ {2} + \frac {2 \beta - \beta^ {2}}{m})\right) \mathbb {E} _ {\widehat {\mathcal {D}} _ {a} ^ {t}} \| g _ {a} - \widehat {g} _ {a} \| _ {\pi} ^ {2} - \beta^ {2} \bar {\tau} ^ {2} \| g _ {a} - \widehat {g} _ {n} \| _ {\pi} ^ {2}
$$

$$
= (1 + \frac {1}{m}) \beta (2 - \beta) \mathbb {E} _ {\widehat {\mathcal {D}} _ {a} ^ {t}} \| g _ {a} - \widehat {g _ {a}} \| _ {\pi} ^ {2} - \beta^ {2} \bar {\tau} ^ {2} \| g _ {a} - \widehat {g _ {n}} \| _ {\pi} ^ {2}
$$

When $m \to \infty$ , we have a simplified version of the error difference as follows

$$
\Delta_ {A T} ^ {2} - \Delta_ {G P} ^ {2} \approx \beta (2 - \beta) \mathbb {E} _ {\widehat {\mathcal {D}} _ {a} ^ {t}} \| g _ {a} - \widehat {g _ {a}} \| _ {\pi} ^ {2} - \beta^ {2} \bar {\tau} ^ {2} \| g _ {a} - \widehat {g _ {n}} \| _ {\pi} ^ {2}
$$

![](images/40f032ebe5ab0ff334d3d5bcb5e86d4275d7b135318ab2329c8e9eea12b35fb3.jpg)

Interpretation. When $\beta = 0.5$ , we can usually show $\Delta_{AT}^{2} > \Delta_{GP}^{2}$ , because $\beta(2 - \beta) > \beta^{2}\bar{\tau}^{2}(0.75 > 0.25)$ for the coefficients of two terms. We estimate the actual values of terms $E_{\widehat{D}_{a,t}}\|g_{a} - \widehat{g}_{a}\|_{\pi}^{2}$ (variance), $\|g_{a} - \widehat{g}_{n}\|_{\pi}^{2}$ (bias), and $\bar{\tau}$ using the estimation methods in Enyi Jiang [2024]. Table 6 displays the values of those terms as well as the error differences on ResNet18 experiments at epoch 5, 10, 15, 20, 60. We plot the changing of these terms on the ResNet18 experiment in Figure 5. The order of difference is always positive and usually smaller than $1e^{-08}$ and approaches the order of $1e^{-12}$ in the end.

Table 6: Estimations the actual values of terms $E_{\widehat{D_{a^t}}}\| g_a - \widehat{g}_a\|_\pi^2$ (variance), $\| g_a - \widehat{g}_n\|_\pi^2$ (bias), $\bar{\tau}$ , and $\Delta_{AT}^2 - \Delta_{GP}^2$ (error differences) across different epochs. 

<table><tr><td>Terms / epochs</td><td>5</td><td>10</td><td>15</td><td>20</td><td>60</td></tr><tr><td> $E_{\widehat{D}_{a^t}} \| g_a - \widehat{g}_a \|_\pi^2$ </td><td>4.6017e-08</td><td>2.0448e-09</td><td>6.9623e-10</td><td>6.4329e-10</td><td>2.3849e-11</td></tr><tr><td> $\| g_a - \widehat{g}_n \|_\pi^2$ </td><td>0.0007</td><td>9.9098e-05</td><td>4.4932e-05</td><td>3.7930e-05</td><td>2.8391e-06</td></tr><tr><td> $\bar{\tau}$ </td><td>0.0071</td><td>0.0052</td><td>0.0036</td><td>0.0038</td><td>0.0030</td></tr><tr><td> $\Delta_{AT}^2 - \Delta_{GP}^2$ </td><td>2.5335e-08</td><td>8.5709e-10</td><td>3.7609e-10</td><td>3.4487e-10</td><td>1.1574e-11</td></tr></table>

# B Additional Experiment Information

In this section, we provide more training details, additional experiment results on the universal robustness of RAMP to common corruptions and unseen adversaries, runtime analysis of RAMP, additional ablation studies on different logit pairing losses, and AT from random initialization results on CIFAR-10 using WideResNet-28-10.

# B.1 More Training Details

We set the batch size to 128 for the experiments on ResNet-18 and WideResNet-28-10 architectures. We use an SGD optimizer with 0.9 momentum and $5e^{-4}$ weight decay. For other experiments on ImageNet, we use a batch size of 64 to fit into the GPU memory for larger models. For all training procedures, we select the last checkpoint for the comparison. When the pre-trained model was originally trained with extra data beyond the CIFAR-10 dataset, similar to Croce and Hein [2022], we use the extra 500k images introduced by Carmon et al. [2019] for fine-tuning, and each batch contains the same amount of standard and extra images. An epoch is completed when the whole standard training set has been used.

![](images/bf652444acf55cf5552abbc44baf3b80ebd4bac4e7ff50823719112f1c0de60c.jpg)  
Figure 5: Plot of values of terms $E_{\widehat{D}_{a^t}} \| g_a - \widehat{g}_a \|_\pi^2$ (variance), $\| g_a - \widehat{g}_n \|_\pi^2$ (bias), $\bar{\tau}$ , and $\Delta_{AT}^2 - \Delta_{GP}^2$ (error differences).

# B.2 Runtime Analysis of RAMP

We present runtime analysis results demonstrating the fact that RAMP is more expensive than E-AT and less expensive than MAX in Table 7. These results, recorded in seconds per epoch, were obtained using a single A100 40GB GPU. RAMP consistently supports that fact in all experiments.

Table 7: Analysis of time per epoch for RAMP and related baselines. RAMP is more expensive than E-AT and less expensive than MAX. 

<table><tr><td>Models \Methods</td><td>E-AT [Croce and Hein, 2022]</td><td>MAX</td><td>RAMP</td></tr><tr><td>CIFAR-10 RN-18 scratch</td><td>78</td><td>219</td><td>157</td></tr><tr><td>CIFAR-10 WRN-28-10 scratch</td><td>334</td><td>1048</td><td>660</td></tr><tr><td>CIFAR-10 RN-50</td><td>188</td><td>510</td><td>388</td></tr><tr><td>CIFAR-10 WRN-34-20</td><td>1094</td><td>2986</td><td>2264</td></tr><tr><td>CIFAR-10 WRN-28-10 carmon</td><td>546</td><td>1420</td><td>1110</td></tr><tr><td>CIFAR-10 WRN-28-10 gowal</td><td>698</td><td>1895</td><td>1456</td></tr><tr><td>CIFAR-10 WRN-70-16</td><td>3486</td><td>10330</td><td>7258</td></tr><tr><td>ImageNet ResNet50</td><td>15656</td><td>41689</td><td>35038</td></tr><tr><td>ImageNet Transformer</td><td>38003</td><td>101646</td><td>81279</td></tr></table>

# B.3 Additional Results on RAMP Generalizing to Common Corruptions and Unseen Adversaries for Universal Robustness

In this section, we show RAMP can generalize better to other corruptions and unseen adversaries on union accuracy for stronger universal robustness.

Implementations. For the $l_{0}$ attack, we use Croce et al. [2022] with an epsilon of 9 pixels and 5k query points. For common corruptions, we directly use the implementation of RobustBench [Croce et al., 2020] for evaluation across 5 severity levels on all corruption types used in Hendrycks and Dietterich [2019]. For other unseen adversaries, we follow the implementation of Laidlaw et al. [2020], where we set eps = 12 for the fog attack, eps = 0.5 for the snow attack, eps = 60 for the gabor attack, eps = 0.125 for the elastic attack, and eps = 0.125 for the jpeglinf attack with 100 iterations. For ResNet-18 experiments, we do not compare with Winninghand [Diffenderfer et al.,

2021] since it uses a Wide-ResNet architecture. Also, we select the strongest baselines (E-AT and MAX) from the Wide-ResNet experiment results to compare for ResNet-18 experiments on universal robustness.

Results. For the ResNet-18 training from scratch experiment on CIFAR-10, in Table 8 and 9, we also show RAMP generally outperforms by 0.5% on common corruptions and 7% on union accuracy against unseen adversaries compared with E-AT.

Table 8: Accuracy against common corruptions using ResNet-18 on CIFAR-10 dataset. 

<table><tr><td>Models</td><td>common corruptions</td></tr><tr><td>E-AT</td><td>73.8</td></tr><tr><td>MAX</td><td>75.1</td></tr><tr><td>RAMP</td><td>74.3</td></tr></table>

Table 9: Individual, average, and union accuracy against unseen adversaries using ResNet-18 on CIFAR-10 dataset. 

<table><tr><td>Models</td><td> $l_0$ </td><td>fog</td><td>snow</td><td>gabor</td><td>elastic</td><td>jpeglinf</td><td>Avg</td><td>Union</td></tr><tr><td>E-AT</td><td>58.5</td><td>41.8</td><td>30.8</td><td>45.9</td><td>55.0</td><td>59.1</td><td>48.5</td><td>18.8</td></tr><tr><td>MAX</td><td>70.8</td><td>40.0</td><td>34.4</td><td>45.1</td><td>54.8</td><td>56.8</td><td>50.3</td><td>20.6</td></tr><tr><td>RAMP</td><td>56.8</td><td>40.5</td><td>40.5</td><td>50.0</td><td>59.2</td><td>56.2</td><td>50.5</td><td>25.9</td></tr></table>

# B.4 Additional Experiments with Different Epsilon Values

In this section, we provide additional results with different $\epsilon_{1}, \epsilon_{2}, \epsilon_{\infty}$ values. We select $\epsilon_{\infty} = [\frac{2}{255}, \frac{4}{255}, \frac{12}{255}, \frac{16}{255}]$ , $\epsilon_{1} = [6, 9, 12, 15]$ , and $\epsilon_{2} = [0.25, 0.75, 1.0, 1.5]$ . We provide additional RAMP results compared with related baselines with training from scratch and performing robust fine-tuning in Section B.4.1 and Section B.4.2, respectively. We observe that RAMP can surpass E-AT with significant margins as well as a better accuracy-robustness tradeoff for both training from scratch and robust fine-tuning with $\lambda = 2.0$ for training from scratch and $\lambda = 0.5$ for robust fine-tuning in most cases.

# B.4.1 Additional Results with Training from Scratch

Changing $l_{\infty}$ perturbations with $\epsilon_{\infty} = [\frac{2}{255}, \frac{4}{255}, \frac{12}{255}, \frac{16}{255}]$ . Table 10 and Table 11 show that RAMP consistently outperforms E-AT [Croce and Hein, 2022] on union accuracy when training from scratch.

Table 10: $(\epsilon_{\infty} = \frac{2}{255}, \epsilon_{1} = 12, \epsilon_{2} = 0.5)$ and $(\epsilon_{\infty} = \frac{4}{255}, \epsilon_{1} = 12, \epsilon_{2} = 0.5)$ with random initializations. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td></tr><tr><td>E-AT</td><td>87.2</td><td>73.3</td><td>64.1</td><td>55.4</td><td>55.4</td><td>E-AT</td><td>86.8</td><td>58.9</td><td>66.4</td><td>54.6</td><td>53.7</td></tr><tr><td>RAMP</td><td>86.3</td><td>73.3</td><td>64.9</td><td>59.1</td><td>59.1</td><td>RAMP</td><td>86.1</td><td>60.0</td><td>67.4</td><td>58.5</td><td>57.4</td></tr></table>

Table 11: $(\epsilon_{\infty} = \frac{12}{255}, \epsilon_1 = 12, \epsilon_2 = 0.5)$ and $(\epsilon_{\infty} = \frac{16}{255}, \epsilon_1 = 12, \epsilon_2 = 0.5)$ with random initializations. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT</td><td>77.5</td><td>28.8</td><td>64.0</td><td>50.1</td><td>28.7</td><td>E-AT</td><td>69.4</td><td>18.8</td><td>58.7</td><td>47.7</td><td>18.7</td></tr><tr><td>RAMP</td><td>73.7</td><td>34.6</td><td>59.1</td><td>38.9</td><td>33.3</td><td>RAMP</td><td>65.0</td><td>25.7</td><td>49.8</td><td>32.6</td><td>25.0</td></tr></table>

Changing $l_{1}$ perturbations with $\epsilon_{1} = [6, 9, 12, 15]$ . Table 12 and Table 13 show that RAMP consistently outperforms E-AT [Croce and Hein, 2022] on union accuracy when training from scratch.

Changing $l_{2}$ perturbations with $\epsilon_{2} = [0.25, 0.75, 1.0, 1.5]$ . Table 14 and Table 15 show that RAMP consistently outperforms E-AT [Croce and Hein, 2022] on union accuracy when training from scratch.

Table 12: $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{\mathbf{1}} = \mathbf{6}, \epsilon_{2} = 0.5)$ and $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{\mathbf{1}} = \mathbf{9}, \epsilon_{2} = 0.5)$ with random initializations. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT</td><td>85.5</td><td>43.1</td><td>67.9</td><td>63.9</td><td>42.8</td><td>E-AT</td><td>84.6</td><td>41.8</td><td>67.7</td><td>57.6</td><td>41.4</td></tr><tr><td>RAMP</td><td>83.8</td><td>48.1</td><td>63.0</td><td>51.2</td><td>46.0</td><td>RAMP</td><td>82.6</td><td>47.5</td><td>65.7</td><td>50.8</td><td>45.9</td></tr></table>

Table 13: $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{1} = 15, \epsilon_{2} = 0.5)$ and $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{1} = 18, \epsilon_{2} = 0.5)$ with random initializations. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT</td><td>81.9</td><td>40.2</td><td>66.9</td><td>48.7</td><td>39.2</td><td>E-AT</td><td>81.0</td><td>39.8</td><td>65.8</td><td>44.3</td><td>38.0</td></tr><tr><td>RAMP</td><td>80.9</td><td>45.0</td><td>66.4</td><td>46.7</td><td>43.3</td><td>RAMP</td><td>79.9</td><td>43.5</td><td>65.7</td><td>45.0</td><td>41.9</td></tr></table>

Table 14: $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_1 = 12, \epsilon_2 = 0.25)$ and $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_1 = 12, \epsilon_2 = 0.75)$ with random initializations. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT</td><td>82.8</td><td>41.3</td><td>75.6</td><td>52.9</td><td>40.5</td><td>E-AT</td><td>83.0</td><td>41.2</td><td>57.6</td><td>53.0</td><td>40.5</td></tr><tr><td>RAMP</td><td>81.8</td><td>46.0</td><td>74.7</td><td>48.8</td><td>44.5</td><td>RAMP</td><td>81.9</td><td>46.1</td><td>56.9</td><td>48.7</td><td>44.5</td></tr></table>

Table 15: $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_1 = 12, \epsilon_{\mathbf{2}} = \mathbf{1.0})$ and $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_1 = 12, \epsilon_{\mathbf{2}} = \mathbf{1.5})$ with random initializations. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td></tr><tr><td>E-AT</td><td>83.4</td><td>41.0</td><td>47.3</td><td>52.8</td><td>40.3</td><td>E-AT</td><td>83.5</td><td>41.0</td><td>25.5</td><td>52.9</td><td>25.5</td></tr><tr><td>RAMP( $\lambda=5$ )</td><td>81.5</td><td>46.0</td><td>46.5</td><td>48.1</td><td>44.1</td><td>RAMP</td><td>74.4</td><td>43.4</td><td>37.2</td><td>51.1</td><td>37.1</td></tr></table>

# B.4.2 Additional Results with Robust Fine-tuning

Changing $l_{\infty}$ perturbations with $\epsilon_{\infty} = [\frac{2}{255}, \frac{4}{255}, \frac{12}{255}, \frac{16}{255}]$ . Table 16 and Table 17 show that RAMP consistently outperforms E-AT [Croce and Hein, 2022] on union accuracy when performing robust fine-tuning.

Table 16: $(\epsilon_{\infty} = \frac{2}{255}, \epsilon_{1} = 12, \epsilon_{2} = 0.5)$ and $(\epsilon_{\infty} = \frac{4}{255}, \epsilon_{1} = 12, \epsilon_{2} = 0.5)$ with robust fine-tuning. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT</td><td>86.5</td><td>74.8</td><td>66.7</td><td>57.9</td><td>57.9</td><td>E-AT</td><td>85.9</td><td>61.4</td><td>67.9</td><td>57.6</td><td>56.8</td></tr><tr><td>RAMP</td><td>85.8</td><td>74.0</td><td>66.2</td><td>60.1</td><td>60.1</td><td>RAMP</td><td>85.7</td><td>60.9</td><td>67.6</td><td>59.3</td><td>58.1</td></tr></table>

Table 17: $(\epsilon_{\infty} = \frac{12}{255}, \epsilon_1 = 12, \epsilon_2 = 0.5)$ and $(\epsilon_{\infty} = \frac{16}{255}, \epsilon_1 = 12, \epsilon_2 = 0.5)$ with robust fine-tuning. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT</td><td>75.5</td><td>30.8</td><td>62.4</td><td>44.6</td><td>30.0</td><td>E-AT</td><td>68.7</td><td>20.7</td><td>56.1</td><td>42.1</td><td>20.5</td></tr><tr><td>RAMP</td><td>74.0</td><td>33.6</td><td>59.7</td><td>38.5</td><td>31.9</td><td>RAMP</td><td>65.6</td><td>25.0</td><td>51.5</td><td>31.2</td><td>23.8</td></tr></table>

Changing $l_{1}$ perturbations with $\epsilon_{1} = [6, 9, 12, 15]$ . Table 12 and Table 13 show that RAMP consistently outperforms E-AT [Croce and Hein, 2022] on union accuracy when performing robust fine-tuning.

Changing $l_{2}$ perturbations with $\epsilon_{2} = [0.25, 0.75, 1.0, 1.5]$ . Table 14 and Table 15 show that RAMP consistently outperforms E-AT [Croce and Hein, 2022] on union accuracy when performing robust fine-tuning.

Table 18: $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{1} = 6, \epsilon_{2} = 0.5)$ and $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{1} = 9, \epsilon_{2} = 0.5)$ with robust fine-tuning. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT</td><td>84.2</td><td>45.8</td><td>66.8</td><td>59.0</td><td>45.0</td><td>E-AT</td><td>83.1</td><td>44.9</td><td>67.2</td><td>52.6</td><td>43.2</td></tr><tr><td>RAMP( $\lambda=1.5$ )</td><td>83.0</td><td>48.7</td><td>63.5</td><td>51.7</td><td>46.4</td><td>RAMP</td><td>82.5</td><td>47.1</td><td>66.0</td><td>49.9</td><td>44.8</td></tr></table>

Table 19: $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{1} = 15, \epsilon_{2} = 0.5)$ and $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{1} = 18, \epsilon_{2} = 0.5)$ with robust fine-tuning. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT</td><td>81.3</td><td>43.5</td><td>66.6</td><td>42.8</td><td>39.0</td><td>E-AT</td><td>81.3</td><td>38.9</td><td>66.6</td><td>45.0</td><td>37.5</td></tr><tr><td>RAMP</td><td>80.4</td><td>44.2</td><td>66.1</td><td>44.4</td><td>41.2</td><td>RAMP</td><td>80.7</td><td>40.6</td><td>66.3</td><td>43.5</td><td>38.8</td></tr></table>

Table 20: $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{1} = 12, \epsilon_{2} = 0.25)$ and $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_{1} = 12, \epsilon_{2} = 0.75)$ with robust fine-tuning. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td></tr><tr><td>E-AT</td><td>82.3</td><td>44.2</td><td>75.3</td><td>47.2</td><td>41.4</td><td>E-AT</td><td>83.0</td><td>43.5</td><td>58.1</td><td>46.5</td><td>40.4</td></tr><tr><td>RAMP</td><td>81.5</td><td>45.6</td><td>74.4</td><td>47.1</td><td>43.1</td><td>RAMP</td><td>81.4</td><td>45.6</td><td>57.4</td><td>47.2</td><td>42.9</td></tr></table>

Table 21: $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_1 = 12, \epsilon_2 = 1.0)$ and $(\epsilon_{\infty} = \frac{8}{255}, \epsilon_1 = 12, \epsilon_2 = 1.5)$ with robust fine-tuning. 

<table><tr><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td><td></td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_{2}$ </td><td> $l_{1}$ </td><td>Union</td></tr><tr><td>E-AT</td><td>82.3</td><td>41.0</td><td>49.0</td><td>51.6</td><td>40.2</td><td>E-AT</td><td>80.2</td><td>42.8</td><td>31.5</td><td>52.4</td><td>31.5</td></tr><tr><td>RAMP</td><td>81.4</td><td>45.6</td><td>47.8</td><td>47.1</td><td>42.9</td><td>RAMP</td><td>74.9</td><td>43.7</td><td>37.0</td><td>50.2</td><td>36.9</td></tr></table>

# B.5 Different Logit Pairing Methods

In this section, we test RAMP with robust fine-tuning using two more different logit pairing losses: (1) Mean Squared Error Loss $(\mathcal{L}_{mse})$ (Eq. 17), (2) Cosine-Similarity Loss $(\mathcal{L}_{cos})$ (Eq. 18). We replace the KL loss we used in the paper using the following losses. We use the same lambda value $\lambda = 1.5$ for both cases.

$$
\mathcal {L} _ {m s e} = \frac {1}{n _ {c}} \cdot \sum_ {i = 0} ^ {n _ {c}} \frac {1}{2} \left(p _ {q} [ \gamma [ i ] ] - p _ {r} [ \gamma [ i ] ]\right) ^ {2} \tag {17}
$$

$$
\mathcal {L} _ {c o s} = \frac {1}{n _ {c}} \cdot \sum_ {i = 0} ^ {n _ {c}} (1 - \cos (p _ {q} [ \gamma [ i ] ], p _ {r} [ \gamma [ i ] ])) \tag {18}
$$

Table 22 displays RAMP robust fine-tuning results of different logit pairing losses using PreAct-ResNet-18 on CIFAR-10 with $\lambda = 1.5$ . We see those losses generally improve union accuracy compared with baselines in Table 24. $L_{cos}$ has a better clean accuracy yet slightly worsened union accuracy. $L_{mse}$ has the best union accuracy and the worst clean accuracy. $L_{KL}$ is in the middle of the two others. However, we acknowledge the possibility that each logit pairing loss may have its own best-tuned $\lambda$ value.

Table 22: RAMP fine-tuning results of different logit pairing losses using PreAct-ResNet-18 on CIFAR-10. 

<table><tr><td>Losses</td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>KL</td><td>80.9</td><td>45.5</td><td>66.2</td><td>47.3</td><td>43.1</td></tr><tr><td>MSE</td><td>80.4</td><td>45.6</td><td>65.8</td><td>47.6</td><td>43.5</td></tr><tr><td>Cosine</td><td>81.6</td><td>45.4</td><td>66.7</td><td>47.0</td><td>42.9</td></tr></table>

# B.6 AT from Scratch Using WideResNet-28-10

Implementations. We use a cyclic learning rate with a maximum rate of 0.1 for 30 epochs and adopt the outer minimization trades loss from Zhang et al. [2019] with the default hyperparameters, same as Croce and Hein [2022]; also, we set $\lambda = 2.0$ and $\beta = 0.5$ for training RAMP. Additionally, we use the WideResNet-28-10 architecture same as Zagoruyko and Komodakis [2016] for our reimplementations on CIFAR-10.

Results. Since the implementation of experiments on WideResNet-28-10 in Croce and Hein [2022] paper is not public at present, we report our implementation results on E-AT, where our results show that RAMP outperforms E-AT in union accuracy with a significant margin, as shown in Table 23. Also, we experiment with using the trade loss (RAMP w trades) for the outer minimization, we observe that RAMP w trades achieves a better union accuracy at the loss of some clean accuracy.

Table 23: WideResNet-28-10 trained from random initialization on CIFAR-10. RAMP outperforms E-AT on union accuracy with our implementation. 

<table><tr><td>Methods</td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>E-AT w trades (reported in Croce and Hein [2022])</td><td>79.9</td><td>46.6</td><td>66.2</td><td>56.0</td><td>46.4</td></tr><tr><td>E-AT w trades (ours)</td><td>79.2</td><td>44.2</td><td>64.9</td><td>54.9</td><td>44.0</td></tr><tr><td>RAMP w/o trades (ours)</td><td>81.1</td><td>46.6</td><td>65.9</td><td>48.1</td><td>44.6</td></tr><tr><td>RAMP w trades (ours)</td><td>79.9</td><td>47.1</td><td>65.1</td><td>49.0</td><td>45.8</td></tr></table>

# B.7 Robust Fine-tuning Using PreAct-ResNet-18

Implementations. For robust fine-tuning with ResNet-18, we perform 3 epochs on CIFAR-10. We set the learning rate as 0.05 for PreAct-ResNet-18 and 0.01 for other models. We set $\lambda = 0.5$ in this case. Also, we reduce the learning rate by a factor of 10 after completing each epoch.

Result. Table 24 shows the robust fine-tuning results using PreAct ResNet-18 model on the CIFAR-10 dataset with different methods. The results for all baselines are directly from the E-AT paper [Croce and Hein, 2022] where the authors reimplemented other baselines (e.g., MSD, MAX) to achieve better union accuracy than presented in the original works. RAMP surpasses all other methods on union accuracy.

Table 24: RN-18 $l_{\infty}$ -AT model fine-tuned for 3 epochs (repeated for 5 seeds). RAMP has the highest union accuracy. Baseline results are from Croce and Hein [2022]. 

<table><tr><td>Methods</td><td>Clean</td><td> $l_{\infty}$ </td><td> $l_2$ </td><td> $l_1$ </td><td>Union</td></tr><tr><td>RN-18- $l_{\infty}$ -AT</td><td>83.7</td><td>48.1</td><td>59.8</td><td>7.7</td><td>38.5</td></tr><tr><td>+ SAT</td><td>83.5 ± 0.2</td><td>43.5 ± 0.2</td><td>68.0 ± 0.4</td><td>47.4 ± 0.5</td><td>41.0 ± 0.3</td></tr><tr><td>+ AVG</td><td>84.2 ± 0.4</td><td>43.3 ± 0.4</td><td>68.4 ± 0.6</td><td>46.9 ± 0.6</td><td>40.6 ± 0.4</td></tr><tr><td>+ MAX</td><td>82.2 ± 0.3</td><td>45.2 ± 0.4</td><td>67.0 ± 0.7</td><td>46.1 ± 0.4</td><td>42.2 ± 0.6</td></tr><tr><td>+ MSD</td><td>82.2 ± 0.4</td><td>44.9 ± 0.3</td><td>67.1 ± 0.6</td><td>47.2 ± 0.6</td><td>42.6 ± 0.2</td></tr><tr><td>+ E-AT</td><td>82.7 ± 0.4</td><td>44.3 ± 0.6</td><td>68.1 ± 0.5</td><td>48.7 ± 0.5</td><td>42.2 ± 0.8</td></tr><tr><td>+ RAMP ( $\lambda$ =1.5)</td><td>81.1 ± 0.2</td><td>45.4 ± 0.3</td><td>66.1 ± 0.2</td><td>47.2 ± 0.1</td><td>43.1 ± 0.2</td></tr><tr><td>+ RAMP ( $\lambda$  = 0.5)</td><td>81.5 ± 0.1</td><td>45.5 ± 0.2</td><td>66.4 ± 0.2</td><td>47.0 ± 0.1</td><td>42.9 ± 0.2</td></tr></table>

# B.8 Robust Fine-tuning with More Epochs

In Table 25, we apply robust fine-tuning on the PreAct ResNet-18 model for the CIFAR-10 dataset with 5, 7, 10, 15 epochs, and compare it with E-AT. RAMP consistently outperforms the baseline on union accuracy, with a larger improvement when we increase the number of epochs.

# C Additional Visualization Results

In this section, we provide additional t-SNE visualizations of the multiple-norm tradeoff and robust fine-tuning procedures using different methods.

Table 25: Fine-tuning with more epochs: RAMP consistently outperforms E-AT on union accuracy. E-AT results are from Croce and Hein [2022]. 

<table><tr><td rowspan="2"></td><td colspan="2">5 epochs</td><td colspan="2">7 epochs</td><td colspan="2">10 epochs</td><td colspan="2">15 epochs</td></tr><tr><td>Clean</td><td>Union</td><td>Clean</td><td>Union</td><td>Clean</td><td>Union</td><td>Clean</td><td>Union</td></tr><tr><td>E-AT</td><td>83.0</td><td>43.1</td><td>83.1</td><td>42.6</td><td>84.0</td><td>42.8</td><td>84.6</td><td>43.2</td></tr><tr><td>RAMP</td><td>81.7</td><td>43.6</td><td>82.1</td><td>43.8</td><td>82.5</td><td>44.6</td><td>83.0</td><td>44.9</td></tr></table>

# C.1 Pre-trained $l_{1}, l_{2}, l_{\infty}$ AT models

Figure 6 shows the robust accuracy of $l_1, l_2, l_\infty$ AT models against their respect $l_1, l_2, l_\infty$ perturbations, on CIFAR-10 using PreAct-ResNet-18 architecture. Similar to Figure ??, $l_\infty$ -AT model has a low $l_1$ robustness and vice versa. In this common choice of epsilons, we further confirm that $l_\infty - l_1$ is the key trade-off pair.

# C.2 Robust Fine-tuning for all Epochs

We provide the complete visualizations of robust fine-tuning for 3 epochs on CIFAR-10 using $l_{1}$ examples, E-AT, and RAMP. Rows in $l_{1}$ fine-tuning (Figure 7), E-AT fine-tuning (Figure 8), and RAMP fine-tuning (Figure 9) show the robust accuracy against $l_{\infty}, l_{1}, l_{2}$ attacks individually, of epoch 0, 1, 2, 3, respectively. We observe that throughout the procedure, RAMP manages to maintain more $l_{\infty}$ robustness during the fine-tuning with more points colored in cyan, in comparison with two other methods. This visualization confirms that after we identify a $l_{p} - l_{r} (p \neq r)$ key tradeoff pair, RAMP successfully preserves more $l_{p}$ robustness when training with some $l_{r}$ examples via enforcing union predictions with the logit pairing loss.

![](images/225a4edb4f0e02aee2f94a33a84cd337eb3eef286d0acfcba27836c195b7e224.jpg)  
Figure 6: $l_{1}, l_{2}, l_{\infty}$ pre-trained RN18 $l_{\infty}$ -AT models with correct/incorrect predictions against $l_{1}, l_{2}, l_{\infty}$ attacks. Correct predictions are colored with cyan and incorrect with magenta. Each row represents $l_{\infty}, l_{1}, l_{2}$ AT models, respectively. Each column shows the accuracy concerning a certain $l_{p}$ attack.

![](images/90e8d53818919bc0d488474b66f9c48c8d3fee3a38c37a2699b1f65a84a1b0c9.jpg)

<details>
<summary>scatter</summary>

| Group | X Range | Y Range | Color  |
|-------|---------|---------|--------|
| Linf  | -0.5 to 0.5 | -0.5 to 0.5 | Purple |
| L1    | -0.5 to 0.5 | -0.5 to 0.5 | Purple |
| L2    | -0.5 to 0.5 | -0.5 to 0.5 | Purple |
</details>

Figure 7: Finetune RN18 $l_{\infty}$ -AT model on $l_{1}$ examples for 3 epochs. Each row represents the prediction results of epoch 0, 1, 2, 3 respectively.

![](images/20418fd147346cb0d9034ccec31dd6425e3ed103176271be6e46ce1000a7e3b3.jpg)  
Figure 8: Finetune RN18 $l_{\infty}$ -AT model with E-AT for 3 epochs. Each row represents the prediction results of epoch 0, 1, 2, 3 respectively.

![](images/c41d12225a7af44009633721d9cd0a74ab10233c0639f30603ec5d91fc310410.jpg)  
Figure 9: Finetune RN18 $l_{\infty}$ -AT model with RAMP for 3 epochs. Each row represents the prediction results of epoch 0, 1, 2, 3 respectively.