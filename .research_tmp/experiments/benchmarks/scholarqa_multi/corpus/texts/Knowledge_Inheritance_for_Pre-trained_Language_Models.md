# Knowledge Inheritance for Pre-trained Language Models

Yujia Qin $^{1,2,3}$ , Yankai Lin $^{4}$ , Jing Yi $^{1,2,3}$ , Jiajie Zhang $^{1,2,3}$ , Xu Han $^{1,2,3}$ , Zhengyan Zhang $^{1,2,3}$ , Yusheng Su $^{1,2,3}$ , Zhiyuan Liu $^{1,2,3,5,6,*}$ , Peng Li $^{7\dagger}$ , Maosong Sun $^{1,2,3,5,*}$ , Jie Zhou $^{4}$

$^{1}$ Department of Computer Science and Technology, Tsinghua University, Beijing, China

$^{2}$ Beijing National Research Center for Information Science and Technology $^{3}$ Institute for Artificial Intelligence, Tsinghua University, Beijing, China $^{4}$ Pattern Recognition Center, WeChat AI, Tencent Inc.

$^{5}$ International Innovation Center of Tsinghua University, Shanghai, China $^{6}$ Quan Cheng Laboratory

$^{7}$ Institute for AI Industry Research (AIR), Tsinghua University, China.
qyj20@mails.tsinghua.edu.cn

# Abstract

Recent explorations of large-scale pre-trained language models (PLMs) have revealed the power of PLMs with huge amounts of parameters, setting off a wave of training ever-larger PLMs. However, it requires tremendous computational resources to train a large-scale PLM, which may be practically unaffordable. In addition, existing large-scale PLMs are mainly trained from scratch individually, ignoring that many well-trained PLMs are available. To this end, we explore the question how could existing PLMs benefit training large-scale PLMs in future. Specifically, we introduce a pre-training framework named “knowledge inheritance” (KI) and explore how could knowledge distillation serve as auxiliary supervision during pre-training to efficiently learn larger PLMs. Experimental results demonstrate the superiority of KI in training efficiency. We also conduct empirical analyses to explore the effects of teacher PLMs’ pre-training settings, including model architecture, pre-training data, etc. Finally, we show that KI could be applied to domain adaptation and knowledge transfer. The implementation is publicly available at https://github.com/thunlp/Knowledge-Inheritance.

# 1 Introduction

Recently, it has become a consensus in the NLP community to use pre-trained language models (PLMs) as the backbone for various downstream tasks (Han et al., 2021; Min et al., 2021). Despite the great follow-up efforts of exploring various pre-training techniques and model architectures, researchers find that simply enlarging the model capacity, data size and training steps can further improve the performance of PLMs (Kaplan et al., 2020; Li et al., 2020b). This discovery sets off a wave of training large-scale PLMs (Raffel et al., 2019; Brown et al., 2020; Fedus et al., 2021).

Although huge PLMs have shown awesome performance (Bommasani et al., 2021), it requires tremendous computational resources to train large-scale PLMs (Schwartz et al., 2019), raising severe environmental concerns on the prohibitive computational costs. Moreover, existing PLMs are generally trained from scratch individually, ignoring that many well-trained PLMs are available. This leaves us an important question: how could existing PLMs benefit training larger PLMs in future?

Considering that humans can leverage the knowledge summarized by their predecessors to learn new tasks, so that the learning process could become efficient; similarly, it is worth inheriting the implicit knowledge distributed in existing PLMs. In this sense, we could distill the knowledge summarized by an existing small PLM during pretraining to efficiently learn larger PLMs. We dub the above process as knowledge inheritance (KI). This intuition is similar to reversed KD (Yuan et al., 2020) in the field of computer vision. They indicate that a delicate student model could still benefit from a teacher with an inferior architecture for a specific downstream task.

However, the success of reversed KD in supervised downstream tasks does not guarantee its feasibility under the scenario of large-scale self-supervised pre-training. Therefore, in this paper,

we strive to answer the following research questions: (RQ1) could distilling knowledge from an existing trained PLM benefit large PLMs' training from scratch? (RQ2) Considering human beings are able to hand down knowledge from generation to generation, could KI similarly be sequentially performed among a series of PLMs with growing sizes? (RQ3) As more and more PLMs with different pre-training settings (model architectures, training data, training strategies, etc) emerge, how would different settings affect the performance of KI? (RQ4) Besides training a large PLM from scratch, when adapting an already trained large PLM to a new domain, how could smaller domain teachers benefit such a process?

In conclusion, the contributions of this paper are summarized as follows: (1) we are the first to formulate the problem of knowledge inheritance, and demonstrate the feasibility of inheriting the knowledge from previously trained PLMs for efficiently training larger ones; (2) we show that the learned knowledge in PLMs could accumulate and further be passed down from generation to generation; (3) we systematically conduct empirical analyses to show the effects of various teacher pre-training settings, which may indicate how to select the most appropriate PLM as the teacher for KI; (4) we further show that during domain adaptation, an already trained large PLM could benefit from multiple small PLMs of different domains under the KI framework. The above empirical studies indicate that KI can well support cross-model knowledge transfer, providing a promising direction to share the knowledge learned by different PLMs and continuously promote their performance.

# 2 Related Work

Efficient Pre-training for NLP. Recently, researchers find that the performance of PLMs can be simply improved by increasing the model size, data size and training steps (Liu et al., 2019; Raffel et al., 2019; Kaplan et al., 2020), sparking a wave of training ever-larger PLMs. For instance, the revolutionary GPT-3 (Brown et al., 2020), which contains 175 billion parameters, shows strong capabilities for language understanding and generation. This means that utilizing PLMs with huge parameters for downstream tasks may greatly relieve the cost of manual labeling and model training for new tasks. However, larger models require greater computational demands (Patterson et al., 2021). To this end, researchers propose to accelerate pre-training by mixed-precision training (Shoeybi et al., 2019), distributed training (Shoeybi et al., 2019), large batch optimization (You et al., 2020), etc.

Another line of methods (Gong et al., 2019; Gu et al., 2021; Chen et al., 2022; Qin et al., 2022) proposes to pre-train larger PLMs progressively. They first train a small PLM, and then gradually increase the depth or width of the network based on parameter recycling (PR). Although PR could be used for the goal of KI, these methods typically have strict requirements on the architectures of both models, which is not flexible for practical uses; instead, we resort to KD as the solution for KI without architecture constraints. In addition, different from KI, PR is not applicable for absorbing knowledge from multiple teacher models and domain adaptation. More detailed comparisons between KI and PR are discussed in appendix E.

Knowledge Distillation for PLMs. Knowledge Distillation (KD) (Hinton et al., 2015) aims to compress a large model into a fast-to-execute one. KD has renewed a surge of interest in PLMs recently. Some explore KD at different training phases, e.g., pre-training (Sanh et al., 2019), downstream fine-tuning (Sun et al., 2019; Krishna et al., 2020), or both of them (Jiao et al., 2020); others explore distilling not only the final logits output by the large PLM, but also the intermediate hidden representations (Sanh et al., 2019; Jiao et al., 2020; Sun et al., 2020). Conventional KD presumes that teacher models play pivotal roles in mastering knowledge, and student models generally cannot match their teachers in performance. When it comes to the scenario of KI, since student models have larger capacities, the performance of teacher models is no longer an “upper bound” of student models. Outside NLP, researchers recently demonstrate that a student model could also benefit from a poor teacher for a specific downstream task (Yuan et al., 2020) (reversed KD). Based on the prior explorations, in this paper, we investigate the application of reversed KD in pre-training.

# 3 Knowledge Inheritance

Task Formulation. Given a textual input $x = \{x^{1}, \ldots, x^{n}\}$ and the corresponding label $y \in R^{K}$ , where K is the number of classes for the specific pre-training task, e.g., the vocabulary size for masked language modeling (MLM) (Devlin et al., 2019), a PLM M converts each token

$x^{j} \in x$ to task-specific logits $z^{j} = [z_{1}^{j}, ..., z_{K}^{j}]$ . $z^{j}$ is then converted to a probability distribution $\mathcal{P}(x^{j}; \tau) = [p_{1}(x^{j}; \tau), ..., p_{K}(x^{j}; \tau)]$ using a softmax function with temperature $\tau$ . $\mathcal{M}$ is pre-trained with the objective $\mathcal{L}_{\text{SELF}}(\mathbf{x}, \mathbf{y}) = \mathcal{H}(\mathbf{y}, \mathcal{P}(\mathbf{x}; \tau))$ , where $\mathcal{H}$ is the loss function, e.g., cross-entropy for MLM. Assume that we have a well-trained small PLM $\mathcal{M}_{S}$ optimized with the self learning objective $\mathcal{L}_{\text{SELF}}$ (such as MLM), our goal is leveraging $\mathcal{M}_{S}$ 's knowledge to efficiently train a larger PLM $\mathcal{M}_{L}$ on the corpora $\mathcal{D}_{L} = \{(\mathbf{x}_{i}, \mathbf{y}_{i})\}_{i=1}^{\left|\mathcal{D}_{L}\right|}$ .

Investigated Methodology. Specifically, imparting $M_{S}$ 's knowledge to $M_{L}$ on $D_{L}$ is implemented by minimizing the Kullback-Leibler (KL) divergence between two probability distributions output by $M_{S}$ and $M_{L}$ on the same input $x_{i} \in D_{L}$ , i.e., $\mathcal{L}_{\mathrm{KI}}(\mathbf{x}_{i}; \mathcal{M}_{S}) = \tau^{2}\mathrm{KL}(\mathcal{P}_{\mathcal{M}_{S}}(\mathbf{x}_{i}; \tau) || \mathcal{P}_{\mathcal{M}_{L}}(\mathbf{x}_{i}; \tau))$ . In addition, $M_{L}$ is also encouraged to conduct self-learning by optimizing $\mathcal{L}_{\mathrm{SELF}}(\mathbf{x}_{i}, \mathbf{y}_{i})$ . Both $L_{SELF}$ and $L_{KI}$ are balanced with an inheritance rate $\alpha$ :

$$
\begin{array}{l} \mathcal {L} (\mathcal {D} _ {L}; \mathcal {M} _ {S}) = \sum_ {(\mathbf {x} _ {i}, \mathbf {y} _ {i}) \in \mathcal {D} _ {L}} (1 - \alpha) \mathcal {L} _ {\mathrm{SELF}} (\mathbf {x} _ {i}, \mathbf {y} _ {i}) + \alpha \mathcal {L} _ {\mathrm{KI}} (\mathbf {x} _ {i}; \mathcal {M} _ {S}) \\ = \sum_ {(\mathbf {x} _ {i}, \mathbf {y} _ {i}) \in \mathcal {D} _ {L}} (1 - \alpha) \mathcal {H} (\mathbf {y} _ {i}, \mathcal {P} _ {\mathcal {M} _ {L}} (\mathbf {x} _ {i}; 1)) \tag {1} \\ + \alpha \tau^ {2} \mathrm{KL} (\mathcal {P} _ {\mathcal {M} _ {S}} (\mathbf {x} _ {i}; \tau) | | \mathcal {P} _ {\mathcal {M} _ {L}} (\mathbf {x} _ {i}; \tau))). \\ \end{array}
$$

Since larger models generally converge faster and can achieve better final performance (Li et al., 2020b), $M_{L}$ becomes more and more knowledgeable during the learning process, and would surpass the teacher eventually. Thus, it is necessary to encourage $M_{L}$ increasingly learning knowledge on its own, not only following the teacher's instructions. Additionally, after $M_{L}$ has surpassed its teacher, it no longer needs the guidance from $M_{S}$ and should conduct pure self-learning from then on. Therefore, different from reversed KD, we dynamically change the inheritance rate $\alpha$ . Specifically, for a total training steps of T, we linearly decay $\alpha_{t}$ with a slope of $\frac{\alpha_{T}}{T}$ . The student only inherits knowledge from the teacher for $\frac{T}{\alpha_{T}}$ steps, and then conducts pure self-learning, i.e., $\alpha_{t} = \max(1 - \alpha_{T} \times \frac{t}{T}, 0)$ . Formally, at step t, the loss function for inheriting knowledge of $M_{S}$ on $D_{L}$ is formulated as:

$$
\mathcal {L} \left(\mathcal {D} _ {L}; \mathcal {M} _ {S}\right) = \sum_ {\left(\mathbf {x} _ {i}, \mathbf {y} _ {i}\right) \in \mathcal {D} _ {L}} \left(1 - \alpha_ {t}\right) \mathcal {L} _ {\text { SELF }} \left(\mathbf {x} _ {i}, \mathbf {y} _ {i}\right) + \alpha_ {t} \mathcal {L} _ {\mathrm{KI}} \left(\mathbf {x} _ {i}; \mathcal {M} _ {S}\right). \tag {2}
$$

Note the logits of $M_{S}$ on $D_{L}$ can be pre-computed and saved offline so that we do not need to re-compute the logits of $\mathcal{M}_S$ when training $\mathcal{M}_L$ . This process is done once and for all.

# 4 Empirical Analysis

In this section, we answer our research questions proposed before. Specifically, (1) we first demonstrate the effectiveness of KI in § 4.1. (2) Then we show PLMs can accumulate knowledge over generations in § 4.2. (3) We also investigate the effects of different pre-training settings of the teacher models in § 4.3. (4) Finally, we show that KI could benefit domain adaptation, and a trained PLM can learn more efficiently with the help of multiple domain teachers in § 4.4. Detailed pre-training hyper-parameters are listed in appendix B.

# 4.1 RQ1: How Could Knowledge Inheritance Benefit Large PLMs' Training?

Setting. Our KI framework is agnostic to the specific self-supervised pre-training task and the PLM architecture. Without loss of generality, we mainly focus on the representative MLM task and use the model architecture of RoBERTa (Liu et al., 2019). Specifically, we first choose RoBERTa $_{BASE}$ (denoted as BASE) as the teacher ( $M_{S}$ ) and RoBERTa $_{LARGE}$ (denoted as LARGE) as the student ( $M_{L}$ ). We also experiment on auto-regressive language modeling using GPT (Radford et al., 2018) to show KI is model-agnostic.

For pre-training data, we use the concatenation of Wikipedia and BookCorpus (Zhu et al., 2015) same as BERT (Devlin et al., 2019), with roughly 3,400M tokens in total. All models ( $\mathcal{M}_S$ and $\mathcal{M}_L$ ) are trained for 125k steps, with a batch size of 2,048 and a sequence length of 512. Note the whole training computations are comparable with those of BERT. We pre-train $\mathcal{M}_L$ by inheriting $\mathcal{M}_S$ 's knowledge under KI (denoted as “BASE → LARGE”). We compare it with “LARGE” that only conducts self-learning from beginning to end.

For performance evaluation, we report the validation perplexity (PPL) during pre-training and the downstream performance on development sets of eight GLUE (Wang et al., 2019) tasks. Note compared with the self-learning baseline, in KI, the logits output by $M_{L}$ are additionally used to calculate $L_{KI}$ , we empirically find that the additional computations caused by it are almost negligible compared with the cumbersome computations in Transformer blocks. Therefore, it requires almost the same computational cost between KI and the baseline for

![](images/eccb246876dbbaed7d8de2feadaa23f27898fd7070ca30ea72ccd2b42fa3b57f.jpg)

<details>
<summary>line</summary>

| Steps | LARGE | BASE → LARGE | BASE(final) |
| ----- | ----- | ------------ | ----------- |
| 0     | 8.0   | 8.0          | 4.18        |
| 20k   | 6.0   | 5.5          | 4.18        |
| 40k   | 4.8   | 4.5          | 4.18        |
| 60k   | 4.2   | 4.0          | 4.18        |
| 80k   | 3.8   | 3.7          | 4.18        |
| 100k  | 3.6   | 3.5          | 4.18        |
| 120k  | 3.5   | 3.4          | 4.18        |
</details>

(a)

![](images/9af12844850edb768c52b10a25abf443b583dbe16b14bcd531b36b7105930144.jpg)

<details>
<summary>line</summary>

| Steps | BASE   | MEDIUM → BASE_Heviside | MEDIUM → BASE_Constant | MEDIUM → BASE_Linear |
|-------|--------|------------------------|-------------------------|----------------------|
| 20k   | 6.0    | 5.8                    | 5.7                     | 5.7                  |
| 30k   | 5.6    | 5.5                    | 5.4                     | 5.3                  |
| 40k   | 5.2    | 5.4                    | 5.1                     | 5.0                  |
| 50k   | 4.9    | 5.1                    | 4.9                     | 4.8                  |
| 60k   | 4.8    | 4.8                    | 4.8                     | 4.7                  |
</details>

(b)

![](images/ab085684b68a72194be82cacafae3b6ae7fb52dc11ca34efad0d438c02043760.jpg)

<details>
<summary>line</summary>

| Steps | BASE | MEDIUM → BASE | MEDIUM → BASE_Top = 10 | MEDIUM → BASE_Top = 50 | MEDIUM → BASE_Top = 100 | MEDIUM → BASE_Top = 1000 |
| ----- | ---- | ------------- | ---------------------- | ---------------------- | ----------------------- | ------------------------ |
| 10k   | 7.5  | 7.5           | 7.5                    | 7.5                    | 7.5                     | 7.5                      |
| 20k   | 6.2  | 6.0           | 6.1                    | 6.0                    | 6.1                     | 6.0                      |
| 30k   | 5.5  | 5.4           | 5.5                    | 5.4                    | 5.5                     | 5.4                      |
| 40k   | 5.2  | 5.1           | 5.2                    | 5.1                    | 5.2                     | 5.1                      |
| 50k   | 5.0  | 4.9           | 5.0                    | 4.9                    | 5.0                     | 4.9                      |
</details>

(c)   
Figure 1: (a) The validation PPL curve for pre-training $\mathcal{M}_L$ under KI framework (BASE $\rightarrow$ LARGE) and the self-learning baseline (LARGE). The teacher's (BASE) performance is 4.18. (b) Pre-training BASE under KI with three strategies for the inheritance rate $\alpha_t$ : Linear, Heviside and Constant. The teacher's (MEDIUM) performance is 4.95. (c) Pre-training BASE under KI with top- $K$ logits, we vary $K$ in $\{10, 50, 100, 1000\}$ , respectively.

each step. Hence, we report the performance w.r.t training step (Li et al., 2020a), while the performance w.r.t. FLOPs (Schwartz et al., 2019) and wall-clock time (Li et al., 2020b) can be roughly obtained by stretching the figure horizontally.

Overall Results. As shown in Figure 1 (a), we conclude that: (1) training $M_{L}$ under KI converges faster than the self-learning baseline, indicating that inheriting the knowledge from an existing teacher is far more efficient than solely learning such knowledge. That is, to achieve the same level of validation PPL, KI requires fewer computational costs. Specifically, under the guidance of $M_{S}$ , whose validation PPL is 4.18, BASE → LARGE achieves a validation PPL of 3.41 at the end of pre-training, compared with baseline (LARGE) 3.58. After BASE → LARGE stops learning from the teacher at the 40k-th step, it improves the validation PPL from 4.60 (LARGE) to 4.28, which is almost the performance when the baseline LARGE conducts self-learning for 55k steps, thus saving roughly 27.3% computational costs $^{1}$ . The results in Table 1 show that (2) $M_{L}$ trained under KI achieves better performance than the baseline on downstream tasks at each step. We also found empirically that, under the same setting (e.g., data, hyper-parameters and model architectures), lower validation PPL generally indicates better downstream performance. Since the performance gain in downstream tasks is consistent with that reflected in PPL, we only show the latter for the remaining experiments. Concerning the energy cost, for the remaining experiments, unless otherwise specified, we choose MEDIUM (9 layers, 576 hidden size) as $\mathcal{M}_S$ and BASE as $\mathcal{M}_L$ .

Effects of Inheritance Rate. We set $\alpha_{t}$ in Eq. (2) to be linearly decayed (denoted as Linear) to gradually encourage $\mathcal{M}_L$ exploring knowledge on its own. We analyze whether this design is necessary by comparing it with two other strategies: the first is to only learn from the teacher at first and change to pure self-learning (denoted as Heviside) at the 35k-th step; the second is to use a constant ratio (1:1) between $\mathcal{L}_{\mathrm{SELF}}$ and $\mathcal{L}_{\mathrm{KI}}$ throughout the whole training process (denoted as Constant). We can conclude from Figure 1 (b) that: (1) annealing at first is necessary. The validation PPL curve of Linear converges the fastest, while Heviside tends to increase after $\mathcal{M}_L$ stops learning from the teacher, indicating that, due to the difference between learning from the teacher and self-learning, annealing at first is necessary so that the performance won't decay at the transition point (the 35k-th step). (2) Supervision from the teacher is redundant after $\mathcal{M}_L$ surpasses $\mathcal{M}_S$ . Although Constant performs well in the beginning, its PPL gradually becomes even worse than the other two strategies. This indicates that after $\mathcal{M}_L$ has already surpassed $\mathcal{M}_S$ , it will be encumbered by keeping following guidance from $\mathcal{M}_S$ .

Saving Storage Space with Top-K Logits. Loading the teacher $M_{S}$ repeatedly for KI is cumbersome, and an alternative way is to pre-compute and save the predictions of $M_{S}$ offline once and for all. We show that using the information of top-K logits (Tan et al., 2019) can reduce the memory footprint without much performance decrease. Specifically, we save only top-K probabilities of $\mathcal{P}_{S}(x^{j};\tau)$ followed by re-normalization, instead of the full distribution over all tokens. For RoBERTa,

<table><tr><td>Step</td><td>Model</td><td>CoLA</td><td>MNLI</td><td>QNLI</td><td>RTE</td><td>SST-2</td><td>STS-B</td><td>MRPC</td><td>QQP</td><td>Avg</td></tr><tr><td rowspan="2">5k</td><td>LARGE</td><td>0.0</td><td>73.5</td><td>81.7</td><td>53.0</td><td>81.7</td><td>45.8</td><td>71.4</td><td>87.5</td><td>61.8</td></tr><tr><td>BASE → LARGE</td><td>17.4</td><td>75.8</td><td>83.4</td><td>54.7</td><td>85.7</td><td>72.0</td><td>72.6</td><td>88.6</td><td>68.8</td></tr><tr><td rowspan="2">45k</td><td>LARGE</td><td>61.8</td><td>84.9</td><td>91.7</td><td>63.4</td><td>92.9</td><td>88.6</td><td>87.7</td><td>91.5</td><td>82.8</td></tr><tr><td>BASE → LARGE</td><td>64.3</td><td>85.9</td><td>92.2</td><td>75.3</td><td>93.2</td><td>89.3</td><td>89.4</td><td>91.5</td><td>85.2</td></tr><tr><td rowspan="2">85k</td><td>LARGE</td><td>64.5</td><td>86.8</td><td>92.7</td><td>69.7</td><td>93.5</td><td>89.9</td><td>89.7</td><td>91.7</td><td>84.8</td></tr><tr><td>BASE → LARGE</td><td>65.7</td><td>87.2</td><td>93.0</td><td>77.0</td><td>94.3</td><td>90.0</td><td>90.4</td><td>91.8</td><td>86.2</td></tr><tr><td rowspan="2">125k</td><td>LARGE</td><td>64.3</td><td>87.1</td><td>93.2</td><td>73.4</td><td>94.1</td><td>90.3</td><td>90.1</td><td>91.8</td><td>85.5</td></tr><tr><td>BASE → LARGE</td><td>67.7</td><td>87.7</td><td>93.1</td><td>74.9</td><td>94.8</td><td>90.6</td><td>88.2</td><td>91.9</td><td>86.1</td></tr></table>

Table 1: Downstream performances on GLUE tasks (dev). KI requires fewer pre-training steps to get a high score after fine-tuning. Detailed results at different training steps are illustrated in appendix A.6.

the dimension of $\mathcal{P}_{S}(x^{j};\tau)$ is decided by its vocabulary size, which is around 50,000. We thus vary K in $\{10,50,100,1000\}$ to see its effects in Figure 1 (c), from which we observe that: top-K logits contain the vast majority of information. Choosing a relatively small K (e.g., 10) is already good enough for inheriting knowledge from the teacher without much performance decrease. Previous work also indicates the relation between KD and label smoothing (Shen et al., 2021), however, we show in appendix F that the improvements of KI are not because of benefiting from optimizing smoothed targets, which impose regularization.

Experiments on GPT. To demonstrate that KI is model-agnostic, we conduct experiments on auto-regressive language modeling and choose GPT (Radford et al., 2018) architecture with growing sizes of $\{73M, 124M, 209M, 354M, 773M, 1B\}$ parameters in total, respectively. The detailed architectures are specified in Table 6. All the teacher models are pre-trained for 62.5k steps with a batch size of 2,048. As reflected in Figure 2 (a), training larger GPTs under our KI framework converges faster than the self-learning baseline, which demonstrates KI is agnostic to the specific pre-training objective and PLM architecture.

# 4.2 RQ2: Could Knowledge Inheritance be Performed over Generations?

Human beings can inherit the knowledge from their antecedents, refine it and pass it down to their offsprings, so that knowledge can gradually accumulate over generations. Inspired by this, we investigate whether PLMs also have this kind of pattern. Specifically, we experiment with the knowledge inheritance among three generations of RoBERTa with roughly 1.7x growth in model size: $G_{1}$ (BASE, 125M), $G_{2}$ (BASE\_PLUS, 211M) and $G_{3}$ (LARGE, 355M), whose architectures are listed in Table 6. All models are trained from scratch for 125k steps with a batch size of 2,048 on the same corpus. We compare the differences among (1) self-learning for each generation (denoted as $G_{1}$ , $G_{2}$ and $G_{3}$ ), (2) KI over two generations (denoted as $G_{1} \rightarrow G_{2}$ , $G_{1} \rightarrow G_{3}$ and $G_{2} \rightarrow G_{3}$ ), and (3) KI over three generations (denoted as $G_{1} \rightarrow G_{2} \rightarrow G_{3}$ ), where $G_{2}$ first inherits the knowledge from $G_{1}$ , refines it by additional self-exploring and passes its knowledge down to $G_{3}$ . The results are drawn in Figure 2 (b). Comparing the performance of $G_{2}$ and $G_{1} \rightarrow G_{2}$ , $G_{3}$ and $G_{1} \rightarrow G_{3}$ , or $G_{3}$ and $G_{2} \rightarrow G_{3}$ , we can again demonstrate the superiority of KI over self-training as concluded before. Comparing the performance of $G_{1} \rightarrow G_{3}$ and $G_{1} \rightarrow G_{2} \rightarrow G_{3}$ , or $G_{2} \rightarrow G_{3}$ and $G_{1} \rightarrow G_{2} \rightarrow G_{3}$ , it is observed that the performance of $G_{3}$ benefits from the involvements of both $G_{1}$ and $G_{2}$ , which means knowledge could be accumulated through more generations' involvements.

# 4.3 RQ3: How Could $\mathcal{M}_S$ 's Pre-training Setting Affect Knowledge Inheritance?

Existing PLMs are typically trained under quite different settings, and it is unclear how these different settings will affect the performance of KI. Formally, we have a series of well-trained smaller PLMs $\overline{\mathcal{M}_S} = \{\mathcal{M}_S^1,\dots,\mathcal{M}_S^{N_S}\}$ , each having been optimized on $\overline{\mathcal{D}_S} = \{\mathcal{D}_S^1,\dots,\mathcal{D}_S^{N_S}\}$ , respectively. Considering that the PLMs in $\overline{\mathcal{M}_S}$ , consisting of varied model architectures, are pre-trained on different corpora of various sizes and domains with arbitrary strategies, thus the knowledge they master is also manifold. In addition, $\mathcal{M}_L$ 's pre-training data $\overline{\mathcal{D}_L}$ may also consist of massive, heterogeneous corpora from multiple sources, i.e., $\overline{\mathcal{D}_L} = \{\mathcal{D}_L^1,\dots,\mathcal{D}_L^{N_L}\}$ . Due to the difference between $\overline{\mathcal{D}_L}$ and $\overline{\mathcal{D}_S}$ , $\mathcal{M}_S$

![](images/be0f8a0eaf7a8757b026a93514c5f04bfbcd83675d903b54ea16d4ea83f157b0.jpg)  
(a)

![](images/7b09a9888a50522139fdb60dd7dfdacd4e14343356e4c3f3e6a2637f050f8845.jpg)

<details>
<summary>line</summary>

| Steps | G1    | G2    | G3    | G1 → G2 | G2 → G3 | G1 → G2 → G3 |
|-------|-------|-------|-------|---------|---------|--------------|
| 10k   | 6.5   | 6.5   | 6.5   | 6.5     | 6.5     | 6.5          |
| 20k   | 5.8   | 5.7   | 5.6   | 5.5     | 5.4     | 5.3          |
| 30k   | 5.2   | 5.0   | 4.9   | 4.8     | 4.7     | 4.6          |
| 40k   | 4.8   | 4.6   | 4.5   | 4.4     | 4.3     | 4.2          |
| 50k   | 4.5   | 4.3   | 4.2   | 4.1     | 4.0     | 3.9          |
| 60k   | 4.2   | 4.0   | 3.9   | 3.8     | 3.7     | 3.6          |
</details>

(b)

![](images/716d206ddc0f22998db97034057b039862d04ac20d23a95cf8ea37a662b29aee.jpg)

<details>
<summary>line</summary>

| Steps | BASE  | H_4 → BASE | H_6 → BASE | H_8 → BASE | H_10 → BASE |
|-------|-------|------------|------------|------------|-------------|
| 20k   | 6.00  | 5.90       | 5.85       | 5.80       | 5.75        |
| 40k   | 5.30  | 5.25       | 5.20       | 5.15       | 5.10        |
| 60k   | 4.80  | 4.75       | 4.70       | 4.65       | 4.60        |
| 80k   | 4.50  | 4.45       | 4.40       | 4.35       | 4.30        |
| 100k  | 4.30  | 4.25       | 4.20       | 4.15       | 4.10        |
| 120k  | 4.10  | 4.05       | 4.00       | 3.95       | 3.90        |
</details>

(c)   
Figure 2: (a) Experiments on GPT. (b) KI over generations. (c) Effects of $\mathcal{M}_S$ 's architecture (depth).

may be required to transfer its knowledge on instances unseen during its pre-training. Ideally, we want $M_{S}$ to teach the courses it is skilled in. Therefore, it is essential to choose the most appropriate teacher for each composition $D_{L}^{*} \in \overline{D_{L}}$ . To this end, we conduct thorough experiments to analyze the effects of several representative factors: model architecture, pre-training data, $M_{S}$ 's pre-training step (appendix A.2) and batch size (appendix A.3).

Effects of Model Architecture. Large PLMs generally converge faster and achieve lower PPL, thus serving as more competent teachers. We experiment with two widely chosen architecture variations, i.e., depth (number of layers) and width (hidden size), to explore the effects of $M_{S}$ 's model architectures. We choose BASE (12 layer, 768 hidden size) as $M_{L}$ 's architecture, and choose the architecture of $M_{S}$ to differ from $M_{L}$ in either depth or width. Specifically, for $M_{S}$ , we vary the depth in $\{4,6,8,10\}$ , and the width in $\{384,480,576,672\}$ , respectively, and pre-train $M_{S}$ under the same setting as $M_{L}$ . The PPL curve for each teacher model is shown in appendix A.7, from which we observe that deeper / wider teachers with more parameters converge faster and achieve lower PPL. After that, we pre-train $M_{L}$ under KI leveraging these teacher models. As shown in Figure 2 (c) and appendix A.4, choosing a deeper / wider teacher accelerates $M_{L}$ 's convergence, demonstrating the benefits of learning from a more knowledgeable teacher. Since the performance of PLMs is weakly related to the model shape but highly related to the model size (Li et al., 2020b), it is always a better strategy to choose the larger teacher if other settings are kept the same. In experiments, we also find empirically that, the optimal duration of learning from the teacher is longer for larger teachers, which means it takes more time to learn from a more knowledgeable teacher.

Effects of Pre-training Data. In previous experiments, we assume $M_{L}$ is pre-trained on the same corpus as $M_{S}$ , i.e., $D_{L} = D_{S}$ . However, in real-world scenarios, it may occur that the pre-training corpus used by both $M_{L}$ and $M_{S}$ is mismatched, due to three main factors: (1) data size. When training larger models, the pre-training corpus is often enlarged to improve downstream performance, i.e., $|D_{S}| \ll |D_{L}|$ ; (2) data domain. PLMs are trained on heterogeneous corpora from various sources (e.g., news articles, literary works, etc.), i.e., $P_{D_{S}} \neq P_{D_{L}}$ . The different knowledge contained in each domain may affect PLMs' generalization in downstream tasks; (3) data privacy. Even if both size and domain of $D_{S}$ and $D_{L}$ are ensured to be the same, it may be hard to retrieve the pre-training corpus used by $M_{S}$ due to privacy concerns, with an extreme case: $D_{L} \cap D_{S} = \emptyset$ . The gap between $D_{S}$ and $D_{L}$ may hinder $M_{S}$ 's successful knowledge transfer. We thus design experiments to analyze the effects of these factors, with three observations concluded:

\- Obs. 1: PLMs can image the big from the small for in-domain data. To evaluate the effects of data size, we first pre-train teacher models on different partitions of the original training corpus under the same setting by randomly sampling $\{\frac{1}{16}, \frac{1}{8}, \frac{1}{4}, \frac{1}{2}, \frac{1}{1}\}$ of it, resulting in teacher models with final validation PPL of $\{5.43, 5.15, 5.04, 4.98, 4.92\}$ , respectively. The final validation PPL increases as we shrink the size of $\mathcal{M}_S$ 's pre-training corpus, which implies that training with less data weakens the teacher's ability. Next, we compare the differences when their knowledge is inherited by $\mathcal{M}_L$ . As reflected in Figure 3 (a), however, the performance of KI is not substantially undermined until only $\frac{1}{16}$ of the original data is leveraged by the teacher. This indicates that PLMs can well image the overall data distribution even if it only sees a small part. Hence,

![](images/8cd50f5685c0008da4a8ce3b2795f23cb179182bfec9081e15ecd46e917e41d8.jpg)

<details>
<summary>line</summary>

| Steps | BASE | MEDIUM_1/16→BASE | MEDIUM_1/4→BASE | MEDIUM_1/2→BASE | MEDIUM→BASE |
|-------|------|------------------|-----------------|-----------------|-------------|
| 10k   | 7.5  | 7.5              | 7.5             | 7.5             | 7.5         |
| 20k   | 6.0  | 6.0              | 6.0             | 6.0             | 6.0         |
| 30k   | 5.5  | 5.5              | 5.5             | 5.5             | 5.5         |
| 40k   | 5.2  | 5.2              | 5.2             | 5.2             | 5.2         |
| 50k   | 5.0  | 5.0              | 5.0             | 5.0             | 5.0         |
</details>

(a)

![](images/b0e5348f5792582b7826227a27f3722b1b570275e3b2880dfc9984be7b92e408.jpg)

<details>
<summary>line</summary>

| Steps | BASE | MEDIUM_WB : CS = 1 : 2 → BASE | MEDIUM_WB : CS = 2 : 1 → BASE | MEDIUM_WB : CS = 3 : 1 → BASE | MEDIUM_WB : CS = 4 : 1 → BASE | MEDIUM_WB → BASE |
| ----- | ---- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ---------------- |
| 10k   | 7.5  | 7.5                           | 7.5                           | 7.5                           | 7.5                           | 7.5              |
| 20k   | 6.5  | 6.0                           | 5.8                           | 5.6                           | 5.4                           | 5.2              |
| 30k   | 5.8  | 5.3                           | 5.1                           | 4.9                           | 4.7                           | 4.5              |
| 40k   | 5.3  | 4.8                           | 4.6                           | 4.4                           | 4.2                           | 4.0              |
| 50k   | 5.0  | 4.5                           | 4.3                           | 4.1                           | 3.9                           | 3.7              |
</details>

(b)

![](images/b15064ff9c935f0dc44681072e224c721d300fd3252c0f0e24896e96b7d5d84e.jpg)

<details>
<summary>line</summary>

| Steps | BASE_B | MEDIUM_A→BASE_B | MEDIUM_B→BASE_B |
| ----- | ------ | --------------- | --------------- |
| 10k   | 7.5    | 6.4             | 7.4             |
| 20k   | 6.2    | 5.8             | 5.9             |
| 30k   | 5.6    | 5.4             | 5.5             |
| 40k   | 5.3    | 5.2             | 5.2             |
| 50k   | 5.0    | 5.0             | 5.0             |
</details>

(c)   
Figure 3: Effects of $\mathcal{M}_S$ 's pre-training (a) data size, (b) data domain and (c) data privacy for KI.

when training larger PLMs, unless the data size is extensively enlarged, its impact can be ignored.

\- Obs. 2: Inheriting on similar domain improves performance. To evaluate the effects of data domain, we experiment on the cases where $\mathcal{D}_S$ and $\mathcal{D}_L$ have domain mismatch. Specifically, keeping data size the same, we mix Wikipedia and BookCorpus (WB) used before with computer science (CS) papers from S2ORC (Lo et al., 2020), whose domain is distinct from WB, using different proportions, i.e., WB : CS = {1 : 2, 2 : 1, 3 : 1, 4 : 1}, respectively. We pre-train $\mathcal{M}_S$ on the constructed corpora, then test the performance when $\mathcal{M}_L$ inherits these teachers' knowledge on the WB domain data. As shown in Figure 3 (b), with the domain of the constructed corpus $\mathcal{M}_S$ is trained on becoming gradually similar to WB, the benefits from KI become more obvious, which means it is essential that both $\mathcal{M}_S$ and $\mathcal{M}_L$ are trained on similar domain of data, so that $\mathcal{M}_S$ can successfully impart knowledge to $\mathcal{M}_L$ by teaching the "right" course.

\- Obs. 3: Data privacy is not that important if the same domain is ensured. To evaluate the effects of data privacy, we experiment in an extreme case where $\mathcal{D}_S$ and $\mathcal{D}_L$ have no overlap at all. To avoid the influences of size and domain, we randomly split the WB domain training corpus $\mathcal{D}$ into two halves ( $\mathcal{D}_A$ and $\mathcal{D}_B$ ) and pre-train two teacher models (denoted as MEDIUM $_A$ and MEDIUM $_B$ ) on them. After pre-training, both of them achieve almost the same final PPL (4.99) on the same validation set. They are then inherited by the student model BASE on $\mathcal{D}_B$ (denoted as MEDIUM $_A$ → BASE $_B$ and MEDIUM $_B$ → BASE $_B$ ), which is exactly the pre-training corpus of MEDIUM $_B$ and has no overlap with that of MEDIUM $_A$ . We also choose $\mathcal{M}_L$ that conducts pure self-learning on $\mathcal{D}_B$ as the baseline (denoted as BASE $_B$ ). It is observed from Figure 3 (c) that, there is little difference between the validation PPL curves of MEDIUM $_A$ → BASE $_B$

and $MEDIUM_{B} \rightarrow BASE_{B}$ , indicating that whether the pre-training corpus of $M_{S}$ and $M_{L}$ has data overlap or not is not a serious issue as long as they share the same domain. This is meaningful when organizations aim to share the knowledge of their PLMs without exposing either the pre-training data or the model parameters due to privacy concerns.

# 4.4 RQ4: How Could Knowledge Inheritance Benefit Domain Adaptation?

With streaming data of various domains continuously emerging, training domain-specific PLMs and storing the model parameters for each domain can be prohibitively expensive. To this end, researchers recently demonstrated the feasibility of adapting PLMs to the target domain through continual pre-training (Gururangan et al., 2020). In this section, we further extend KI and demonstrate that domain adaptation for PLM can benefit from inheriting knowledge of existing domain experts.

Specifically, instead of training large PLMs from scratch, which is the setting used before, we focus on adapting $BASE_{WB}$ , which has been well-trained on the WB domain for 125k steps, to two target domains, i.e., computer science (CS) and biomedical (BIO) papers from S2ORC (Lo et al., 2020). The proximity (vocabulary overlap) of three domains is listed in appendix D. We assume there exist two domain experts, i.e., $MEDIUM_{CS}$ and $MEDIUM_{BIO}$ . Each model has been trained on CS / BIO domain for 125k steps. Note their training computation is far less than $BASE_{WB}$ due to fewer model parameters. Hence, either $MEDIUM_{CS}$ or $MEDIUM_{BIO}$ is no match for $BASE_{WB}$ in WB domain but has richer knowledge in CS / BIO domain. For evaluation, we compare both (1) the validation PPL on the target domain and (2) the performance (test F1) on downstream tasks, i.e. ACL-ARC (Jurgens et al., 2018) for CS domain and CHEMPROT (Kringelum et al., 2016) for BIO domain. Before adaptation,

<table><tr><td colspan="2"> $N_{tokens}$ </td><td colspan="2">3,400M</td><td colspan="2">200M</td><td colspan="2">100M</td><td colspan="2">40M</td><td colspan="2">20M</td></tr><tr><td colspan="2">Metrics</td><td>F1</td><td>PPL</td><td>F1</td><td>PPL</td><td>F1</td><td>PPL</td><td>F1</td><td>PPL</td><td>F1</td><td>PPL</td></tr><tr><td rowspan="2">CS</td><td>SL</td><td>69.8</td><td>3.12</td><td>71.7</td><td>3.17</td><td>71.4</td><td>3.24</td><td>68.3</td><td>3.51</td><td>67.5</td><td>4.07</td></tr><tr><td>KI</td><td>72.9</td><td>3.06</td><td>72.6</td><td>3.09</td><td>71.9</td><td>3.11</td><td>71.1</td><td>3.21</td><td>70.8</td><td>3.37</td></tr><tr><td rowspan="2">BIO</td><td>SL</td><td>84.0</td><td>2.67</td><td>82.8</td><td>2.72</td><td>83.2</td><td>2.83</td><td>83.3</td><td>3.16</td><td>82.7</td><td>3.81</td></tr><tr><td>KI</td><td>84.5</td><td>2.65</td><td>83.4</td><td>2.66</td><td>83.9</td><td>2.69</td><td>83.6</td><td>2.82</td><td>83.5</td><td>3.01</td></tr></table>

Table 2: The validation PPL (PPL) and downstream performance (F1) on the target domain (CS / BIO) after $BASE_{WB}$ is post-trained for 4k steps with self-learning (SL) or knowledge inheritance (KI). We experiment with different sizes of domain corpus. All downstream experiments are repeated 10 times with different seeds. 

<table><tr><td> $N_{tokens}$ </td><td colspan="4">3,400M</td><td colspan="4">200M</td><td colspan="4">100M</td><td colspan="4">40M</td></tr><tr><td>Metrics</td><td> $F1_C$ </td><td> $PPL_C$ </td><td> $F1_B$ </td><td> $PPL_B$ </td><td> $F1_C$ </td><td> $PPL_C$ </td><td> $F1_B$ </td><td> $PPL_B$ </td><td> $F1_C$ </td><td> $PPL_C$ </td><td> $F1_B$ </td><td> $PPL_B$ </td><td> $F1_C$ </td><td> $PPL_C$ </td><td> $F1_B$ </td><td> $PPL_B$ </td></tr><tr><td>SL</td><td>71.7</td><td>3.15</td><td>83.7</td><td>2.71</td><td>70.5</td><td>3.97</td><td>82.7</td><td>3.36</td><td>67.7</td><td>5.95</td><td>81.7</td><td>4.84</td><td>68.3</td><td>11.7</td><td>81.1</td><td>10.5</td></tr><tr><td>KI</td><td>72.2</td><td>3.15</td><td>83.9</td><td>2.70</td><td>71.8</td><td>3.42</td><td>83.1</td><td>2.92</td><td>69.8</td><td>3.90</td><td>82.6</td><td>3.32</td><td>69.1</td><td>5.70</td><td>81.3</td><td>4.64</td></tr></table>

Table 3: The results when $BASE_{WB}$ is post-trained on two new domains simultaneously with self-learning (SL) or knowledge inheritance (KI). We report both validation PPL ( $PPL_{B} / PPL_{C}$ ) and downstream performance ( $F1_{B} / F1_{C}$ ) for BIO / CS domain. We observe that SL exhibits severe overfitting when data is relatively scarce.

BASE $_{WB}$ achieves a PPL of 5.41 / 4.86 and F1 of 68.5 / 81.6 on CS / BIO domain, while MEDIUM $_{CS}$ achieves 2.95 (PPL) and 69.4 (F1) on CS domain, MEDIUM $_{BIO}$ achieves 2.55 (PPL) and 83.6 (F1) on BIO domain. This demonstrates the superiority of two teachers over the student in their own domain despite their smaller model capacity.

We compare two strategies for domain adaptation: (1) only conducting self-learning on the target domain and (2) inheriting knowledge from well-trained domain teachers. Specifically, $BASE_{WB}$ is post-trained for additional 4k steps on either CS or BIO domain to learn new knowledge. In addition, considering that in real-world scenarios, it can be hard to retrieve enough pre-training data for a specific domain, due to some privacy issues. Hence, we conduct experiments with different sizes of domain corpus. In Table 2, $BASE_{WB}$ is post-trained on either CS or BIO domain while in Table 3, it is trained on synthetic domain data (BIO : CS = 1 : 1) to absorb knowledge from two domains simultaneously (we assume $M_{L}$ is trained with the optimal teacher selection strategy, i.e., each teacher imparts the knowledge on its own domain data). It can be concluded from Table 2 and Table 3 that:

(1) KI is more training-efficient. Compared with self-learning, inheriting knowledge from domain teachers achieves lower final PPL and improved performance in domain-specific downstream tasks, indicating that, for domain adaptation, KI is more training-efficient so that an already trained large PLM could absorb more knowledge

from new domain with the same training budget. (2) KI is more data-efficient. The PPL gap between KI and SL is further enlarged when there is less domain-specific data available for adaptation, which means KI is more data-efficient especially under the low-resource setting, where domain data is scarce. In other words, only providing a small portion of domain-specific data is enough for satisfactory adaptation performance under KI, while self-learning exhibits overfitting to some extent. (3) Large PLMs can simultaneously absorb knowledge from multiple domains and thus become omnipotent. From Table 3, we observe BASE $_{WB}$ achieves improved performance on both domains after being taught by two teachers simultaneously. KI shows superiority over self-learning. However, simultaneous learning overfits training data more easily and its performance on either domain is no match for learning only one domain at a time.

# 5 Conclusion and Future Work

In this work, we propose a general knowledge inheritance (KI) framework that leverages previously trained PLMs for training larger ones. We conduct sufficient empirical studies to demonstrate its feasibility. In addition, we show that KI could well support knowledge transfer over a series of PLMs with growing sizes. We also comprehensively analyze various pre-training settings of the teacher model that may affect KI's performance, the results shed light on how to choose the most appropriate teacher PLM for KI. Finally, we ex-

tend KI and show that, during domain adaptation, an already trained large PLM could benefit from smaller domain teachers. In general, we provide a promising direction to share and exchange the knowledge learned by different models and continuously promote their performance.

In future, we aim to explore the following directions: (1) the efficiency of KI, i.e., given limited computational budget and pre-training corpus, how to more efficiently absorb knowledge from teacher models. Potential solutions include denoising teacher models' predictions and utilizing more information from the teacher. How to select the most representative data points for KI is also an interesting topic; (2) the effectiveness of KI under different settings, i.e., how can KI be applied if the teachers and the students are pre-trained on different vocabularies, languages, pre-training objectives and modalities.

Finally, we believe it is vital to use fair benchmarking that can accurately and reliably judge each KI algorithm. Thus, we suggest future work to: (1) conduct all experiments under the same computation environment and report the pre-training hyper-parameters and hardware deployments in detail, (2) evaluate the downstream tasks with multiple different random seeds and choose tasks that give relatively stable and consistent results, which could serve as better indicators for PLMs' effectiveness. In addition, it is also essential that PLMs are tested on diverse downstream tasks which evaluate PLMs' different abilities, (3) save the checkpoint more frequently during pre-training and evaluate the downstream performance, which can better indicate the trend of PLMs' effectiveness, and (4) open-source all the codes and model parameters for future comparisons.

# Acknowledgments

This work is supported by the National Key R&D Program of China (No. 2020AAA0106502), Institute Guo Qiang at Tsinghua University, NExT++ project from the National Research Foundation, Prime Minister's Office, Singapore under its IRC@Singapore Funding Initiative, and Beijing Academy of Artificial Intelligence (BAAI). This work is also supported by the Pattern Recognition Center, WeChat AI, Tencent Inc. Yujia Qin and Yankai Lin designed the methods and the experiments. Yujia Qin, Jing Yi and Jiajie Zhang conducted the experiments. Yujia Qin, Yankai Lin and Xu Han wrote the paper. Zhiyuan Liu, Peng Li, Maosong Sun and Jie Zhou advised the project. All the authors participated in the discussion.

# References

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.   
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.   
Cheng Chen, Yichun Yin, Lifeng Shang, Xin Jiang, Yujia Qin, Fengyu Wang, Zhi Wang, Xiao Chen, Zhiyuan Liu, and Qun Liu. 2022. bert2bert: Towards reusable pretrained language models. In Association for Computational Linguistics: ACL 2022, Online.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.   
William Fedus, Barret Zoph, and Noam Shazeer. 2021. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. ArXiv preprint, abs/2101.03961.   
Tommaso Furlanello, Zachary Chase Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar. 2018. Born-again neural networks. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 1602–1611. PMLR.   
Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tie-Yan Liu. 2019. Efficient training of BERT by progressively stacking. In Proceedings

of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 2337–2346. PMLR.   
Xiaotao Gu, Liyuan Liu, Hongkun Yu, Jing Li, Chen Chen, and Jiawei Han. 2021. On the transformer growth for progressive BERT training. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5174–5180, Online. Association for Computational Linguistics.   
Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. 2020. Don’t stop pretraining: Adapt language models to domains and tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342–8360, Online. Association for Computational Linguistics.   
Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong Qiu, Liang Zhang, Wentao Han, Minlie Huang, Qin Jin, Yanyan Lan, Yang Liu, Zhiyuan Liu, Zhiwu Lu, Xipeng Qiu, Ruihua Song, Jie Tang, Ji-Rong Wen, Jinhui Yuan, Wayne Xin Zhao, and Jun Zhu. 2021. Pre-trained models: Past, present and future. ArXiv preprint, abs/2106.07139.   
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. ArXiv preprint, abs/1503.02531.   
Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2020. TinyBERT: Distilling BERT for natural language understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4163–4174, Online. Association for Computational Linguistics.   
David Jurgens, Srijan Kumar, Raine Hoover, Dan McFarland, and Dan Jurafsky. 2018. Measuring the evolution of a scientific field through citation frames. Transactions of the Association for Computational Linguistics, 6:391–406.   
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. ArXiv preprint, abs/2001.08361.   
Jens Kringelum, Sonny Kim Kjaerulff, Søren Brunak, Ole Lund, Tudor I Oprea, and Olivier Taboureau. 2016. Chemprot-3.0: a global chemical biology diseases mapping. Database, 2016.   
Kalpesh Krishna, Gaurav Singh Tomar, Ankur P. Parikh, Nicolas Papernot, and Mohit Iyyer. 2020. Thieves on sesame street! model extraction of bert-based apis. In 8th International Conference on

Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.   
Mengtian Li, Ersin Yumer, and Deva Ramanan. 2020a. Budgeted training: Rethinking deep neural network training under resource constraints. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.   
Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, and Joey Gonzalez. 2020b. Train big, then compress: Rethinking model size for efficient training and inference of transformers. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 5958–5968. PMLR.   
Tianyang Lin, Yuxin Wang, Xiangyang Liu, and Xipeng Qiu. 2021. A survey of transformers. ArXiv preprint, abs/2106.04554.   
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. ArXiv preprint, abs/1907.11692.   
Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel Weld. 2020. S2ORC: The semantic scholar open research corpus. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4969–4983, Online. Association for Computational Linguistics.   
Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heinz, and Dan Roth. 2021. Recent advances in natural language processing via large pre-trained language models: A survey. arXiv preprint arXiv:2111.01243.   
David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. 2021. Carbon emissions and large neural network training. ArXiv preprint, abs/2104.10350.   
Yujia Qin, Jiajie Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. 2022. Elle: Efficient lifelong pre-training for emerging data. In Findings of the Association for Computational Linguistics: ACL 2022, Online.   
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding by generative pre-training.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. ArXiv preprint, abs/1910.10683.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. ArXiv preprint, abs/1910.01108.   
Roy Schwartz, Jesse Dodge, Noah A Smith, and Oren Etzioni. 2019. Green ai. ArXiv preprint, abs/1907.10597.   
Zhiqiang Shen, Zechun Liu, Dejia Xu, Zitian Chen, Kwang-Ting Cheng, and Marios Savvides. 2021. Is label smoothing truly incompatible with knowledge distillation: An empirical study. arXiv preprint arXiv:2104.00676.   
Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multi-billion parameter language models using model parallelism. ArXiv preprint, abs/1909.08053.   
Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. 2019. Patient knowledge distillation for BERT model compression. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4323–4332, Hong Kong, China. Association for Computational Linguistics.   
Siqi Sun, Zhe Gan, Yuwei Fang, Yu Cheng, Shuohang Wang, and Jingjing Liu. 2020. Contrastive distillation on intermediate representations for language model compression. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 498–508, Online Association for Computational Linguistics.   
Xu Tan, Yi Ren, Di He, Tao Qin, Zhou Zhao, and Tie-Yan Liu. 2019. Multilingual neural machine translation with knowledge distillation. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.   
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.   
Chenglin Yang, Lingxi Xie, Siyuan Qiao, and Alan L Yuille. 2019. Training deep neural networks in generations: A more tolerant teacher educates better students. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 5628–5635.   
Yang You, Jing Li, Jonathan Hseu, Xiaodan Song, James Demmel, and Cho-Jui Hsieh. 2019. Reducing bert pre-training time from 3 days to 76 minutes. ArXiv preprint, abs/1904.00962.

Yang You, Jing Li, Sashank J. Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. 2020. Large batch optimization for deep learning: Training BERT in 76 minutes. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.   
Li Yuan, Francis EH Tay, Guilin Li, Tao Wang, and Jiashi Feng. 2020. Revisiting knowledge distillation via label smoothing regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3903–3911.   
Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pages 19–27. IEEE Computer Society.

# Appendices

# A Additional Experiments and Analysis

# A.1 Effects of Model Size

We experiment on four PLMs with roughly 1.7x growth in model size: $\mathcal{M}_1$ (RoBERTa $_{\text{MEDIUM}}$ , 73.5M), $\mathcal{M}_2$ (RoBERTa $_{\text{BASE}}$ , 125M), $\mathcal{M}_3$ (RoBERTa $_{\text{BASE\_PLUS}}$ , 211M) and $\mathcal{M}_4$ (RoBERTa $_{\text{LARGE}}$ , 355M), whose architectures are listed in Table 6. We first pre-train a teacher PLM $\mathcal{M}_i$ ( $\mathcal{M}_S$ ) for 125k steps with a batch size of 2,048 under the same setting then train a larger one $\mathcal{M}_{i+1}$ ( $\mathcal{M}_L$ ) by inheriting $\mathcal{M}_i$ 's knowledge under KI framework (denoted as $\mathcal{M}_i \to \mathcal{M}_{i+1}, i \in \{1,2,3\}$ ). We compare $\mathcal{M}_i \to \mathcal{M}_{i+1}$ with $\mathcal{M}_{i+1}$ that conducts self-learning from beginning to end. As shown in Figure 4, the superiority of KI is observed across all models. In addition, with the overall model size of $\mathcal{M}_S$ and $\mathcal{M}_L$ gradually increasing, the benefits of KI become more evident, reflected in the broader absolute gap between the PPL curve of $\mathcal{M}_i \to \mathcal{M}_{i+1}$ and $\mathcal{M}_{i+1}$ when $i$ gradually grows. This implies that with the advance of computing power in future, training larger PLMs will benefit more and more from our KI framework.

# A.2 Effects of $\mathcal{M}_S$ 's Pre-training Steps

Longer pre-training has been demonstrated as an effective way for PLMs to achieve better performance (Liu et al., 2019) and thus become more knowledgeable. To evaluate the benefits of more pre-training steps for $\mathcal{M}_S$ , we first vary RoBERTa $_{\text{MEDIUM}}$ ’s pre-training steps in {62.5k, 125k, 250k, 500k}, and keep all other settings the same. After pre-training, these teacher models achieve the final validation PPL of {5.25, 4.92, 4.72, 4.51}, respectively. Then we compare the performances when RoBERTa $_{\text{BASE}}$ learn from these teacher models and visualize the results in Figure 4, from which we can conclude that, inheriting knowledge from teachers with longer pre-training time (steps) helps $\mathcal{M}_L$ converge faster. However, such a benefit is less and less obvious as $\mathcal{M}_S$ ’s pre-training steps increase, which means after enough training computations are invested, the teacher model enters a plateau of convergence in validation PPL, and digging deeper in knowledge becomes even harder. The bottleneck lies in other factors, e.g., the size and diversity of pre-training data, which hinder $\mathcal{M}_S$ from becoming more knowledgeable. We also found empirically that, after being pre-trained for 125k steps on the corpus with a batch size of 2,048, all the models used in this paper have well converged, and longer pre-training only results in limited performance gain in either PPL or downstream performance.

# A.3 Effects of $\mathcal{M}_L$ 's Batch Size

Batch size is highly related to PLM's training efficiency, and previous work (Liu et al., 2019; Li et al., 2020b; You et al., 2019) found that slow-but-accurate large batch sizes can bring improvements to model training, although the improvements become marginal after increasing the batch size beyond a certain point (around 2, 048). BERT (Devlin et al., 2019) is pre-trained for 1,000k steps with a batch size of 256, and the computational cost is equivalent to training for 125k steps with a batch size of 2,048 (Liu et al., 2019), which is the pre-training setting chosen in our main paper. Choosing RoBERTa $_{\text{MEDIUM}}$ as the teacher model and RoBERTa $_{\text{BASE}}$ as the student model, in Figure 4 we compare the validation PPL as we vary the batch size in {256, 512, 1024, 2, 048}, controlling for the number of passes through the pre-training corpus. We also vary the peak learning rate in {1.0 × 10 $^{-4}$ , 2.5 × 10 $^{-4}$ , 3.8 × 10 $^{-4}$ , 5.0 × 10 $^{-4}$ } and pre-train for {1,000k, 500k, 250k, 125k} steps, respectively, when increasing the batch size. We observe that increasing the batch size results in improved final validation PPL, which is aligned with previous findings (Liu et al., 2019). When adjusting batch size, KI accelerates the convergence unanimously, and its benefits become more evident when training with a smaller batch size, reflected in the absolute improvement in final validation PPL. We hypothesize that this is because learning from the smoothed target probability of KI, containing rich secondary information (Yang et al., 2019) or dark knowledge (Furlanello et al., 2018), makes the pre-training process more stable. The student PLM is prevented from fitting to unnecessarily strict distributions and can thus learn faster.

# A.4 Additional Experiments of the Effects of Teacher Model $\mathcal{M}_S$ 's architecture (width)

We show in Figure 5 the validation PPL of $\mathcal{M}_L$ when choosing the teacher PLM $\mathcal{M}_S$ with different hidden sizes ( $\{384, 480, 576, 672\}$ ). As mentioned in our main paper, choosing a wider teacher

![](images/f208e70b2d60cb7ca040fc3272b6136e170ef91b2314dafc0151dae24934b0d2.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTaBASE | RoBERTaBASE_PULS | RoBERTaLARGE | RoBERTaMEDIUM → RoBERTaBASE | RoBERTaBASE → RoBERTaBASE_PULS | RoBERTaBASE_PULS → RoBERTaLARGE |
| ------------------------ | ----------- | ---------------- | ------------ | --------------------------- | ----------------------------- | ------------------------------ |
| 20k                      | 8.0         | 7.5              | 7.0          | 7.8                         | 7.6                           | 7.4                            |
| 40k                      | 6.5         | 6.0              | 5.5          | 6.8                         | 6.3                           | 6.0                            |
| 60k                      | 5.5         | 5.0              | 4.5          | 5.8                         | 5.2                           | 4.8                            |
| 80k                      | 5.0         | 4.5              | 4.0          | 5.2                         | 4.6                           | 4.2                            |
| 100k                     | 4.5         | 4.0              | 3.5          | 4.8                         | 4.0                           | 3.6                            |
| 120k                     | 4.0         | 3.5              | 3.0          | 4.2                         | 3.4                           | 3.0                            |
</details>

![](images/150ad79120c3ac4a24b001563cb8fac51481cc6bf4099a2965782def323b2887.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTaBASE | RoBERTaMEDIUM_62.5k | RoBERTaMEDIUM_125k | RoBERTaBASE | RoBERTaMEDIUM_250k | RoBERTaBASE | RoBERTaMEDIUM_500k |
| ------------------------ | ----------- | ------------------ | ----------------- | ----------- | ----------------- | ----------- | ----------------- |
| 10k                      | 7.5         | 7.5                | 7.5               | 7.5         | 7.5               | 7.5         | 7.5               |
| 20k                      | 6.0         | 6.0                | 6.0               | 6.0         | 6.0               | 6.0         | 6.0               |
| 30k                      | 5.5         | 5.5                | 5.5               | 5.5         | 5.5               | 5.5         | 5.5               |
| 40k                      | 5.2         | 5.2                | 5.2               | 5.2         | 5.2               | 5.2         | 5.2               |
| 50k                      | 5.0         | 5.0                | 5.0               | 5.0         | 5.0               | 5.0         | 5.0               |
</details>

![](images/5e5c1fbf9484261fce87516a6c522c29a948cddcbb10819430e2c9ddafc0a7d5.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTaBASE_2048 | RoBERTaBASE_1024 | RoBERTaBASE_512 | RoBERTaBASE_256 | RoBERTaMEDIUM → RoBERTaBASE_2048 | RoBERTaMEDIUM → RoBERTaBASE_1024 | RoBERTaMEDIUM → RoBERTaBASE_512 | RoBERTaMEDIUM → RoBERTaBASE_256 |
| ------------------------ | ---------------- | ---------------- | --------------- | --------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| 0                        | 7.5              | 7.5              | 7.5             | 7.5             | 7.5                               | 7.5                               | 7.5                               | 7.5                               |
| 2k                       | 6.0              | 6.0              | 6.0             | 6.0             | 6.0                               | 6.0                               | 6.0                               | 6.0                               |
| 4k                       | 5.0              | 5.0              | 5.0             | 5.0             | 5.0                               | 5.0                               | 5.0                               | 5.0                               |
| 6k                       | 4.5              | 4.5              | 4.5             | 4.5             | 4.5                               | 4.5                               | 4.5                               | 4.5                               |
| 8k                       | 4.3              | 4.3              | 4.3             | 4.3             | 4.3                               | 4.3                               | 4.3                               | 4.3                               |
| 10k                      | 4.2              | 4.2              | 4.2             | 4.2             | 4.2                               | 4.2                               | 4.2                               | 4.2                               |
</details>

Figure 4: Left: effects of $\mathcal{M}_L$ 's model size. Middle: effects of $\mathcal{M}_S$ 's number of pre-training steps. Right: effects of $\mathcal{M}_L$ 's batch size.

![](images/32e48e215cc4c8010f8a926cb8fbb52028e00f7e681869338d47bcdaf8b0b3be.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTaBASE | RoBERTaD_384 → RoBERTaBASE | RoBERTaD_480 → RoBERTaBASE | RoBERTaD_576 → RoBERTaBASE | RoBERTaD_672 → RoBERTaBASE |
| ------------------------ | ----------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| 20k                      | 6.0         | 6.0                        | 6.0                        | 6.0                        | 6.0                        |
| 40k                      | 5.5         | 5.5                        | 5.5                        | 5.5                        | 5.5                        |
| 60k                      | 5.0         | 5.0                        | 5.0                        | 5.0                        | 5.0                        |
| 80k                      | 4.75        | 4.75                       | 4.75                       | 4.75                       | 4.75                       |
| 100k                     | 4.5         | 4.5                        | 4.5                        | 4.5                        | 4.5                        |
| 120k                     | 4.25        | 4.25                       | 4.25                       | 4.25                       | 4.25                       |
</details>

![](images/06bc2ef32b33a3388fc85f44e97003f48c2ca24b8e5b9ad9fadebb264d0529da.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTa_CS_40M | RoBERTaBASE_WB→CS_40M | RoBERTa_CS_100M | RoBERTaBASE_WB→CS_100M | RoBERTa_CS_200M | RoBERTaBASE_WB→CS_200M | RoBERTa_CS_340M | RoBERTaBASE_WB→CS_340M |
| ------------------------ | -------------- | ---------------------- | --------------- | ----------------------- | --------------- | ----------------------- | --------------- | ----------------------- |
| 1k                       | 3.6            | 3.5                    | 3.5             | 3.5                     | 3.5             | 3.5                     | 3.5             | 3.5                     |
| 2k                       | 3.5            | 3.4                    | 3.4             | 3.4                     | 3.4             | 3.4                     | 3.4             | 3.4                     |
| 3k                       | 3.4            | 3.3                    | 3.3             | 3.3                     | 3.3             | 3.3                     | 3.3             | 3.3                     |
| 4k                       | 3.3            | 3.2                    | 3.2             | 3.2                     | 3.2             | 3.2                     | 3.2             | 3.2                     |
| 5k                       | 3.2            | 3.1                    | 3.1             | 3.1                     | 3.1             | 3.1                     | 3.1             | 3.1                     |
</details>

![](images/4142740a3e9b28e41723c3e2fd7ae0f32a84dd3fa59217bb22c4f30f650c9482.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTaBIO_40M | RoBERTaBASE_WB → BIO_40M | RoBERTaBIO_100M | RoBERTaBASE_WB → BIO_100M | RoBERTaBIO_200M | RoBERTaBASE_WB → BIO_200M | RoBERTaBIO_3400M | RoBERTaBASE_WB → BIO_3400M |
| ------------------------ | -------------- | ------------------------ | --------------- | ------------------------- | --------------- | ------------------------- | ---------------- | -------------------------- |
| 1k                       | 3.18           | 3.05                     | 3.02            | 3.08                      | 3.06            | 3.04                      | 3.01             | 2.99                       |
| 2k                       | 3.15           | 2.98                     | 2.95            | 2.99                      | 2.97            | 2.95                      | 2.92             | 2.89                       |
| 3k                       | 3.12           | 2.95                     | 2.92            | 2.96                      | 2.94            | 2.92                      | 2.89             | 2.86                       |
| 4k                       | 3.10           | 2.92                     | 2.89            | 2.93                      | 2.91            | 2.89                      | 2.86             | 2.83                       |
| 5k                       | 3.08           | 2.90                     | 2.87            | 2.91                      | 2.89            | 2.87                      | 2.84             | 2.81                       |
</details>

Figure 5: Left: the PPL curve when choosing the teacher PLM with different hidden sizes. Middle & Right: adapting RoBERTa $_{BASE\_WB}$ to CS (middle) / BIO (right) domain with different number of training steps on different sizes of domain data. We compare two strategies: self-learning and KI. For example, RoBERTa $_{CS\_3400M}$ denotes post-training RoBERTa $_{BASE\_WB}$ with the self-learning strategy on the 3,400M token CS domain corpus. RoBERTa $_{BASE\_WB}\rightarrow CS\_3400M$ denotes post-training RoBERTa $_{BASE\_WB}$ with the KI strategy on the 3,400M token CS domain corpus.

model improves the training efficiency of the student PLM.

A.5 Additional Experiments of Knowledge Inheritance for Domain Adaptation 

<table><tr><td>Domain</td><td>Strategy</td><td>3,400M</td><td>200M</td><td>100M</td><td>40M</td></tr><tr><td rowspan="2">CS</td><td>SL</td><td>6.71</td><td>7.01</td><td>7.39</td><td>8.77</td></tr><tr><td>KI</td><td>8.63</td><td>9.39</td><td>9.48</td><td>9.87</td></tr><tr><td rowspan="2">BIO</td><td>SL</td><td>7.29</td><td>6.61</td><td>8.16</td><td>10.34</td></tr><tr><td>KI</td><td>10.74</td><td>10.78</td><td>10.93</td><td>11.66</td></tr></table>

Table 4: The validation PPL on the source domain (WB) after RoBERTa $_{BASE\_WB}$ is post-trained on the target domain (CS / BIO) with self-learning (SL) and knowledge inheritance (KI).

Different Number of Post-training Steps. In the main paper, we adapt RoBERTa $_{BASE\_WB}$ to either CS or BIO domain by post-training it for 4k steps. We further vary the number of training steps in {1k, 2k, 3k, 4k, 5k} and visualize the validation PPL in Figure 5. We also experiment on different sizes of domain corpus, i.e., 3, 400M, 200M, 100M, 40M tokens, respectively, as done in the main paper. We observe that generally the validation PPL on each domain decreases with the training step growing, and the performance of KI is always better than self-learning. The improvement of KI over self-learning is further enlarged when there is less target domain data available, demonstrating that KI is more data-efficient and can work well in low-resource settings. In addition, self-learning exhibits overfitting problems when the data size of the target domain is relatively small, which is not observed under our KI framework, which means KI can mitigate overfitting under low-resource settings.

Catastrophic Forgetting on the Source Domain. Table 4 lists the validation PPL on the source domain (WB) after RoBERTa $_{BASE\_WB}$ is post-trained on the target domain (CS / BIO) with self-learning (SL) and knowledge inheritance (KI) for 4k steps. We show the results w.r.t. different sizes of domain corpus (3, 400M, 200M, 100M and 40M tokens). We observe that after domain adaptation, the validation PPL on the source domain increases, which means PLMs may forget some key knowledge on the source domain when learning new knowledge

![](images/733686e39935d7aa1255768e45167f32de173ef18a5639bf0b74f44d4356a732.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTa LARGE | RoBERTa_BASE → RoBERTa LARGE |
| ------------------------ | ------------- | --------------------------- |
| 0k                       | 40            | 40                          |
| 20k                      | 58            | 58                          |
| 40k                      | 62            | 62                          |
| 60k                      | 64            | 67                          |
| 80k                      | 65            | 66                          |
| 100k                     | 64            | 67                          |
| 120k                     | 64            | 68                          |
</details>

![](images/8247c74844fff60ffc683fd30ec5ac93c7f8109f9dda8c04ccb29f4874e3d4f3.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTa LARGE | RoBERTaBASE → RoBERTa LARGE |
| ------------------------ | ------------- | -------------------------- |
| 0                        | 74            | 76                         |
| 20k                      | 80            | 84                         |
| 40k                      | 83            | 85                         |
| 60k                      | 85            | 86                         |
| 80k                      | 86            | 87                         |
| 100k                     | 87            | 87                         |
| 120k                     | 87            | 87                         |
</details>

![](images/7c4055a4882deecfb1102c8c57cea5df7c92aa3c6198c6a127ae0160d166126b.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTa LARGE | RoBERTaBASE → RoBERTa LARGE |
| ------------------------ | ------------- | -------------------------- |
| 0                        | 82.0          | 83.5                       |
| 20k                      | 90.5          | 91.0                       |
| 40k                      | 91.5          | 92.0                       |
| 60k                      | 92.0          | 92.5                       |
| 80k                      | 92.5          | 93.0                       |
| 100k                     | 92.8          | 93.2                       |
| 120k                     | 93.0          | 93.3                       |
</details>

![](images/567b9b87195d37e8eb820865e92f8e2293b1dbd658c09779b2de5bafdd11bb65.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTa LARGE | RoBERTa BASE | RoBERTa LARGE → RoBERTa LARGE |
| ------------------------ | ------------- | ------------ | ----------------------------- |
| 0                        | 53            | 54           | 54                            |
| 20k                      | 65            | 70           | 68                            |
| 40k                      | 63            | 75           | 72                            |
| 60k                      | 70            | 76           | 75                            |
| 80k                      | 71            | 77           | 76                            |
| 100k                     | 75            | 76           | 77                            |
| 120k                     | 73            | 75           | 76                            |
</details>

![](images/c46b0af869624d683cfdf4c31826cb842dbaacd7dcb28c9ef1543201b2d78dab.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTa LARGE | RoBERTa BASE | RoBERTa LARGE |
| ------------------------ | ------------- | ------------ | ------------- |
| 0k                       | 82.0          | 86.0         | 86.0          |
| 20k                      | 90.0          | 92.0         | 92.0          |
| 40k                      | 92.0          | 93.0         | 93.0          |
| 60k                      | 93.0          | 94.0         | 94.0          |
| 80k                      | 93.5          | 94.5         | 94.5          |
| 100k                     | 94.0          | 95.0         | 95.0          |
| 120k                     | 94.5          | 95.5         | 95.5          |
</details>

![](images/349b091766c7a630421092c9948523d2895dd3848ee789a502cb6e65d474bba1.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTa LARGE | RoBERTa_BASE → RoBERTa LARGE |
| ------------------------ | ------------- | --------------------------- |
| 0                        | 80            | 80                          |
| 20k                      | 87            | 88                          |
| 40k                      | 88            | 89                          |
| 60k                      | 89            | 90                          |
| 80k                      | 90            | 90                          |
| 100k                     | 90            | 90                          |
| 120k                     | 91            | 91                          |
</details>

Figure 6: Downstream performance visualization on six GLUE tasks comparing RoBERTa $_{LARGE}$ and RoBERTa $_{BASE} \rightarrow$ RoBERTa $_{LARGE}$ . For CoLA, RTE, SST-2 and STS-B, we repeat fine-tuning for 5 times; for MNLI and QNLI, we repeat fine-tuning for 3 times.

in the target domain, i.e., the catastrophic forgetting problem. In addition, we find that the problem is more evident for KI than self-learning. We expect future work to further explore how to mitigate the catastrophic forgetting.

# A.6 Detailed Downstream Performances on GLUE Tasks

Figure 6 visualizes the downstream performance of RoBERTa $_{LARGE}$ and RoBERTa $_{BASE} \rightarrow$ RoBERTa $_{LARGE}$ on the dev sets of six GLUE tasks at different pre-training steps with an interval of 5k. It can be observed that the downstream performance of RoBERTa $_{BASE} \rightarrow$ RoBERTa $_{LARGE}$ rises faster than the baseline, which means it takes fewer pre-training steps for our KI framework to get a high score in downstream tasks. Aligned with previous findings (Li et al., 2020b), we found MNLI and SST-2 to be the most stable tasks in GLUE, whose variances are lower.

We also list the average GLUE performance for RoBERTa $_{BASE}$ → RoBERTa $_{LARGE}$ and the baseline RoBERTa $_{LARGE}$ in Table 5, from which we observe that the baseline at 70k-th step achieves almost the same GLUE performance as our method at 40k-th step, which means our framework saves around 42.9% FLOPs, much higher than the reported 27.3% FLOPs saved based on the pre-training PPL metric in the main paper. In addition, our method achieves almost the same GLUE performance as the baseline at the final step (125k) with only 70k steps, which means our framework saves 44% FLOPs in total. Both the perplexity in the pre-training stage and performance in downstream tasks can be chosen as the evaluation metric for measuring the computational cost savings. However, in this paper, we choose the former because it is more stable and accurate than the latter. We find empirically that some GLUE tasks like CoLA have higher variances than others, which might make the measurement inaccurate.

Besides, when discussing the effects of model architectures in the main paper, we only show the validation PPL of each model during pre-training, we visualize the corresponding downstream performance (MNLI) in Figure 7, from which it can be observed that learning from teacher models with more parameters helps achieve better downstream performance at the same pre-training step. In general, we observe that, under our setting, the performance gain in downstream tasks is aligned with that reflected in validation PPL during pre-training.

# A.7 Teacher Models' Validation PPL Curves during Pre-training for "Effects of Model Architecture"

Figure 7 visualizes the validation PPL curves for all the teacher models used in the experiments

![](images/047db1ac09cf74e05b66fd238776b72a872c7dcbd07f88c5d8feec886b581d4d.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTaBASE | RoBERTaD_384 → RoBERTaBASE | RoBERTaD_480 → RoBERTaBASE | RoBERTaD_576 → RoBERTaBASE | RoBERTaD_672 → RoBERTaBASE |
| ------------------------ | ----------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| 0                        | 70.0        | 70.0                       | 70.0                       | 70.0                       | 70.0                       |
| 20k                      | 82.0        | 82.5                       | 82.3                       | 82.4                       | 82.6                       |
| 60k                      | 84.5        | 84.8                       | 84.7                       | 84.9                       | 85.0                       |
| 100k                     | 85.0        | 85.1                       | 85.0                       | 85.2                       | 85.3                       |
| 120k                     | 85.1        | 85.2                       | 85.1                       | 85.3                       | 85.4                       |
</details>

![](images/3541dfa565b4e4f4b3f3a25b88a7bf8fe067d63f2028d5e33348691a7fb50724.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTaBASE | RoBERTaH_4 → RoBERTaBASE | RoBERTaH_6 → RoBERTaBASE | RoBERTaH_8 → RoBERTaBASE | RoBERTaH_10 → RoBERTaBASE |
| ------------------------ | ----------- | ------------------------ | ------------------------ | ------------------------ | ------------------------- |
| 0                        | 70.0        | 70.0                     | 70.0                     | 70.0                     | 70.0                      |
| 20k                      | 82.5        | 83.0                     | 82.8                     | 83.2                     | 83.5                      |
| 40k                      | 84.0        | 84.5                     | 84.3                     | 84.7                     | 85.0                      |
| 60k                      | 84.5        | 85.0                     | 84.8                     | 85.2                     | 85.5                      |
| 80k                      | 84.8        | 85.2                     | 85.0                     | 85.4                     | 85.6                      |
| 100k                     | 85.0        | 85.3                     | 85.1                     | 85.5                     | 85.7                      |
| 120k                     | 85.2        | 85.4                     | 85.2                     | 85.6                     | 85.8                      |
</details>

![](images/6acea8b9beae04e042e5d0aedc72cef2b18738ee17844aae72df5e08d31ef3e2.jpg)

<details>
<summary>line</summary>

| Number of gradient steps | RoBERTa_d4.5E | RoBERTa_d4 | RoBERTa_d3.6 | RoBERTa_d3.8 | RoBERTa_d3.10 | RoBERTa_d3.84 | RoBERTa_d4.80 | RoBERTa_d5.76 | RoBERTa_d5.72 |
| ------------------------ | ------------- | ---------- | ------------ | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- |
| 20k                      | 7.5           | 7.5        | 7.5          | 7.5          | 7.5           | 7.5           | 7.5           | 7.5           | 7.5           |
| 40k                      | 6.5           | 6.8        | 6.7          | 6.6          | 6.9           | 6.8           | 6.7           | 6.6           | 6.5           |
| 60k                      | 5.5           | 6.2        | 6.0          | 5.9          | 6.3           | 6.1           | 5.9           | 5.8           | 5.7           |
| 80k                      | 4.8           | 5.8        | 5.6          | 5.5          | 5.9           | 5.7           | 5.5           | 5.4           | 5.3           |
| 100k                     | 4.3           | 5.3        | 5.1          | 5.0          | 5.4           | 5.2           | 5.0           | 4.9           | 4.8           |
| 120k                     | 4.0           | 4.9        | 4.8          | 4.7          | 4.9           | 4.7           | 4.6           | 4.5           | 4.4           |
</details>

Figure 7: Left & Middle: downstream performances corresponding to the experiments on effects of $\mathcal{M}_S$ 's model architecture (width (left) & depth (middle)). Right: validation PPL during pre-training for the teacher models used in experiments of effects of teacher model architecture.

<table><tr><td>Step</td><td>RoBERTaBASE</td><td>RoBERTaBASE → RoBERTaLARGE</td></tr><tr><td>5k</td><td>61.8</td><td>68.8</td></tr><tr><td>10k</td><td>75.6</td><td>78.1</td></tr><tr><td>15k</td><td>79.3</td><td>81.5</td></tr><tr><td>20k</td><td>80.4</td><td>82.8</td></tr><tr><td>25k</td><td>81.7</td><td>83.6</td></tr><tr><td>30k</td><td>82.4</td><td>83.9</td></tr><tr><td>35k</td><td>83.1</td><td>84.1</td></tr><tr><td>40k</td><td>83.6</td><td>84.5</td></tr><tr><td>45k</td><td>82.8</td><td>85.2</td></tr><tr><td>50k</td><td>83.9</td><td>84.6</td></tr><tr><td>55k</td><td>83.4</td><td>85.2</td></tr><tr><td>60k</td><td>84.0</td><td>85.7</td></tr><tr><td>65k</td><td>84.1</td><td>85.3</td></tr><tr><td>70k</td><td>84.3</td><td>85.5</td></tr><tr><td>75k</td><td>85.0</td><td>85.8</td></tr><tr><td>80k</td><td>84.7</td><td>85.8</td></tr><tr><td>85k</td><td>84.8</td><td>86.2</td></tr><tr><td>...</td><td>...</td><td>...</td></tr><tr><td>125k</td><td>85.5</td><td>86.1</td></tr></table>

Table 5: Average GLUE performance comparing both RoBERTa $_{BASE}$ and RoBERTa $_{BASE} \rightarrow$ RoBERTa $_{LARGE}$ at different pre-training steps.

on the effects of model architecture. The teacher models differ from RoBERTa $_{\text{BASE}}$ in either the depth or width. Specifically, we vary the depth in $\{4,6,8,10\}$ (denoted as $\{\text{RoBERTa}_{\text{H\_4}}, \text{RoBERTa}_{\text{H\_6}}, \text{RoBERTa}_{\text{H\_8}}, \text{RoBERTa}_{\text{H\_10}}\}$ ), and the width in $\{384,480,576,672\}$ (denoted as $\{\text{RoBERTa}_{\text{D\_384}}, \text{RoBERTa}_{\text{D\_480}}, \text{RoBERTa}_{\text{D\_576}}, \text{RoBERTa}_{\text{D\_672}}\}$ ). Generally, PLMs with larger model parameters converge faster and achieve better final performance.

# B Pre-training Hyper-parameters

In Table 6, we list the architectures we used for all models, covering the details for the total number of trainable parameters ( $n_{params}$ ), the total number of layers ( $n_{layers}$ ), the number of units in each bottleneck layer ( $d_{model}$ ), the total number of attention heads ( $n_{heads}$ ), the inner hidden size of FFN layer ( $d_{FFN}$ ) and the learning rate when batch size is set to 2,048 (lr). The training-validation ratio of pre-training data is set to 199:1. We set the weight decay to 0.01, dropout rate to 0.1, and use linear learning rate decay. Adam is chosen as the optimizer. The learning rate is warmed up for the first 10% steps. The hyper-parameters for Adam optimizer is set to $1 \times 10^{-6}$ , 0.9, 0.98 for $\epsilon$ , $\beta_{1}$ , $\beta_{2}$ , respectively. For a fair comparison, all experiments are done in the same computation environment with 8 NVIDIA 32GB V100 GPUs. Table 7 describes the total number of pre-training steps for each ( $M_{L}$ , $M_{S}$ ) pair chosen in our experiments.

# C Fine-tuning Hyper-parameters

Table 8 describes the hyper-parameters for ACLARC, CHEMPROT and GLUE tasks. The selection of these hyper-parameters closely follows (Liu et al., 2019) and (Gururangan et al., 2020).

# D Domain Proximity of WB, CS and BIO

Table 9 lists the domain proximity (vocabulary overlap) of WB, CS and BIO used in this paper.

# E Comparison between Knowledge Inheritance and Parameter Recycling

Parameter recycling (i.e., progressive training) first trains a small PLM, and then gradually increases the depth or width of the network based on parameter initialization. It is an orthogonal research direction against our KI, and has many limitations as follows:

Architecture Mismatch. Existing parameter recycling methods (Gong et al., 2019; Gu et al., 2021) require that the architectures of both small PLMs

<table><tr><td>Model Name</td><td> $n_{params}$ </td><td> $n_{layers}$ </td><td> $d_{model}$ </td><td> $n_{heads}$ </td><td> $d_{FFN}$ </td><td>lr (bs = 2,048)</td></tr><tr><td>RoBERTaMEDIUM</td><td>74M</td><td>9</td><td>576</td><td>12</td><td>3072</td><td> $5.0 \times 10^{-4}$ </td></tr><tr><td>RoBERTaD_d</td><td>-</td><td>12</td><td>d</td><td>12</td><td>3072</td><td> $5.0 \times 10^{-4}$ </td></tr><tr><td>RoBERTaH_h</td><td>-</td><td>h</td><td>768</td><td>12</td><td>3072</td><td> $5.0 \times 10^{-4}$ </td></tr><tr><td>RoBERTaBASE</td><td>125M</td><td>12</td><td>768</td><td>12</td><td>3072</td><td> $5.0 \times 10^{-4}$ </td></tr><tr><td>RoBERTaBASE_PLUS</td><td>211M</td><td>18</td><td>864</td><td>12</td><td>3600</td><td> $3.5 \times 10^{-4}$ </td></tr><tr><td>RoBERTaLARGE</td><td>355M</td><td>24</td><td>1024</td><td>16</td><td>4096</td><td> $2.5 \times 10^{-4}$ </td></tr><tr><td>GPT73M</td><td>73M</td><td>9</td><td>576</td><td>12</td><td>3072</td><td> $5.0 \times 10^{-4}$ </td></tr><tr><td>GPT124M</td><td>124M</td><td>12</td><td>768</td><td>12</td><td>3072</td><td> $5.0 \times 10^{-4}$ </td></tr><tr><td>GPT209M</td><td>209M</td><td>18</td><td>864</td><td>12</td><td>3600</td><td> $4.0 \times 10^{-4}$ </td></tr><tr><td>GPT354M</td><td>354M</td><td>24</td><td>1024</td><td>16</td><td>4096</td><td> $3.5 \times 10^{-4}$ </td></tr><tr><td>GPT773M</td><td>773M</td><td>36</td><td>1280</td><td>20</td><td>5120</td><td> $3.0 \times 10^{-4}$ </td></tr><tr><td>GPT1B</td><td>1068M</td><td>40</td><td>1440</td><td>20</td><td>5760</td><td> $2.5 \times 10^{-4}$ </td></tr></table>

Table 6: Model architectures for all the models we used in this paper.

<table><tr><td> $\mathcal{M}_{L}$ </td><td> $\mathcal{M}_{S}$ </td><td>Steps of teacher-guided learning</td></tr><tr><td rowspan="9">RoBERTaBASE</td><td>RoBERTaMEDIUM</td><td>35k</td></tr><tr><td>RoBERTaD_384</td><td>28k</td></tr><tr><td>RoBERTaD_480</td><td>40k</td></tr><tr><td>RoBERTaD_576</td><td>70k</td></tr><tr><td>RoBERTaD_672</td><td>85k</td></tr><tr><td>RoBERTaH_4</td><td>22k</td></tr><tr><td>RoBERTaH_6</td><td>35k</td></tr><tr><td>RoBERTaH_8</td><td>55k</td></tr><tr><td>RoBERTaH_10</td><td>65k</td></tr><tr><td>RoBERTaBASE_PLUS</td><td>RoBERTaBASE</td><td>55k</td></tr><tr><td rowspan="3">RoBERTaLARGE</td><td>RoBERTaBASE</td><td>40k</td></tr><tr><td>RoBERTaBASE_PLUS</td><td>65k</td></tr><tr><td>RoBERTaBASE → RoBERTaBASE_PLUS</td><td>75k</td></tr><tr><td>GPT124M</td><td>GPT73M</td><td>10k</td></tr><tr><td>GPT209M</td><td>GPT124M</td><td>15k</td></tr><tr><td>GPT354M</td><td>GPT209M</td><td>18k</td></tr><tr><td>GPT773M</td><td>GPT354M</td><td>16k</td></tr><tr><td>GPT1B</td><td>GPT773M</td><td>20k</td></tr></table>

Table 7: The total number of steps for teacher-guided learning for different $(\mathcal{M}_{L}, \mathcal{M}_{S})$ pairs.

and large PLMs are matched to some extent, however, our KI does not have such a requirement. For example, Gong et al. (2019); Gu et al. (2021) either requires the number of layers, or the hidden size/embedding size of a large PLM to be the integer multiples of that of a small PLM. Hence, it is not flexible to train larger PLMs with arbitrary architectures, making parameter recycling hard to be implemented practically. Besides, there are more and more advanced non-trivial Transformer modifications appearing (we refer to Lin et al. (2021) for details), e.g., pre-normalization, relative embedding, sparse attention, etc. It is non-trivial to directly transfer the parameters between two PLMs if they have different inner structures. Nevertheless, our KI framework will not be influenced by such architectural mismatches.

Inability for Multi-to-one Knowledge Inheritance. It is non-trivial to support absorbing knowledge from multiple teacher models by jointly recycling their model parameters. Instead, it is easy to implement for KI. As shown in our experiments, we demonstrate that under our framework, large PLMs can simultaneously absorb knowledge from multiple teachers.

Inability of Knowledge Inheritance for Domain Adaptation. Parameter recycling is hard to support continual learning, which makes large PLMs absorb knowledge from small ones in a lifelong manner. In real-world scenarios, numerous PLMs of different architectures are trained locally with

<table><tr><td>HyperParam</td><td>ACL-ARC &amp; CHEMPROT</td><td>GLUE</td></tr><tr><td>Learning Rate</td><td> $2 \times 10^{-5}$ </td><td> $\{1 \times 10^{-5}, 2 \times 10^{-5}, 3 \times 10^{-5}\}$ </td></tr><tr><td>Batch Size</td><td>256</td><td> $\{16, 32\}$ </td></tr><tr><td>Weight Decay</td><td>0.1</td><td>0.1</td></tr><tr><td>Max Epochs</td><td>10</td><td>10</td></tr><tr><td>Learning Rate Decay</td><td>Linear</td><td>Linear</td></tr><tr><td>Warmup Ratio</td><td>0.06</td><td>0.06</td></tr></table>

Table 8: Hyper-parameters for fine-tuning RoBERTa on ACL-ARC, CHEMPROT and GLUE.

<table><tr><td></td><td>WB</td><td>CS</td><td>BIO</td></tr><tr><td>WB</td><td>100%</td><td>19.1%</td><td>25.6%</td></tr><tr><td>CS</td><td>19.1%</td><td>100%</td><td>22.5%</td></tr><tr><td>BIO</td><td>25.6%</td><td>22.5%</td><td>100%</td></tr></table>

Table 9: Domain proximity (vocabulary overlap) among three domains (WB, CS, BIO) discussed in this paper. Following (Gururangan et al., 2020), we create the vocabulary for each domain by considering the top 10k most frequent words (excluding stopwords).

different data. These small PLMs can be seen as domain experts, and it is essential that larger PLMs can continuously benefit from these existing PLMs efficiently by incorporating their knowledge so that larger PLMs can become omnipotent. As described before, it is easy to implement for our framework and we have demonstrated the effectiveness.

Model Privacy. Parameter recycling requires the availability of the parameters of an existing PLM, which may be impractical due to some privacy issues, e.g., GPT-3 only provides API access for prediction instead of the model parameters. Instead, our KI framework does not presume access to an existing model parameter since the predictions of the small model can be pre-computed and saved offline. This superiority will further make it possible for API-based online knowledge transfer.

# F Comparing Label Smoothing and Knowledge Inheritance

Previous work shows the relation between label smoothing and knowledge distillation to some extent (Shen et al., 2021). To demonstrate that the success of our KI is not because of learning from a more smoothed target, we conduct experiments comparing both label smoothing and our KI in Table 10. Specifically, for label smoothing, PLMs optimize a smoothed target $\mathbf{y}_{i}^{S} = (1 - \alpha) * \mathbf{y}_{i} + \alpha * \vec{\mathbf{1}} / (K - 1)$ , where $\alpha = 0$ denotes learning from scratch with no label smoothing, larger $\alpha$ means a more smoothed target for PLMs to learn

<table><tr><td>Step</td><td>20k</td><td>40k</td><td>60k</td><td>80k</td><td>100k</td></tr><tr><td> $\alpha = 0.3$ </td><td>8.68</td><td>7.29</td><td>6.90</td><td>6.57</td><td>6.26</td></tr><tr><td> $\alpha = 0.2$ </td><td>7.27</td><td>6.47</td><td>5.95</td><td>5.68</td><td>5.46</td></tr><tr><td> $\alpha = 0.1$ </td><td>6.71</td><td>5.74</td><td>5.35</td><td>5.06</td><td>4.86</td></tr><tr><td> $\alpha = 0$ </td><td>6.13</td><td>5.21</td><td>4.83</td><td>4.57</td><td>4.36</td></tr><tr><td>KI</td><td>5.69</td><td>5.17</td><td>4.78</td><td>4.52</td><td>4.32</td></tr></table>

Table 10: Validation PPL for training RoBERTa $_{BASE}$ with different strategies. KI denotes our knowledge inheritance framework, where RoBERTa $_{MEDIUM}$ is chosen as the teacher.

from, K denotes the vocabulary size. Specifically, we choose $\alpha$ from $\{0.1, 0.2, 0.3\}$ . It can be concluded from the results in Table 10 that adding label smoothing into the pre-training objectives of PLMs leads to far worse performance than the vanilla baseline, which shows that the improvements of our knowledge inheritance framework are non-trivial: larger PLMs are indeed inheriting the “knowledge” from smaller ones, instead of benefiting from optimizing a smoothed target, which imposes regularization.