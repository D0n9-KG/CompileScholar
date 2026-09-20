# Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models

1 $^{st}$ Qingyuan Zeng

Institute of Artificial Intelligence

Xiamen University

Fujian, China

36920221153145@stu.xmu.edu.cn

$2^{nd}$ Zhenzhong Wang

Department of Computing

The Hong Kong Polytechnic University

Hongkong, China

zhenzhong16.wang@connect.polyu.hk

$3^{rd}$ Yiu-ming Cheung

Department of Computer Science

Hong Kong Baptist University

Hongkong, China

ymc@comp.hkbu.edu.hk

$4^{\mathrm{th}}$ Min Jiang\*

School of Informatics

Xiamen University

Fujian, China

minjiang@xmu.edu.cn

Abstract—While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process Ask, Attend, Attack, called AAA, is proposed to coordinate with the solver. Ask guides attackers to create target texts that satisfy the specific semantics. Attend identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent Attack. Attack uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of Ask, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed AAA.

# I. INTRODUCTION

Image-to-text models, referring to generating descriptive and accurate textual descriptions of images, have received increasing attention in various applications, including image-captioning [1], [2], visual-question-answering [3], [4], and image-retrieval [5], [6]. Despite the remarkable progress, they are vulnerable to deliberate attacks, giving rise to concerns about the reliability and trustworthiness of these models in real-world scenarios. For example, one may mislead models

![](images/1bc952574a45e8f695f8a881a0c15068b6540e3743b0887307d66d230b6e15d2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph "(a) existing work"
        A1["clean image"] --> B1["adversarial perturbation"]
        B1 --> C1["target model"]
        C1 --> D1["image-to-text"]
        D1 --> E1["text-to-image"]
        E1 --> F1["target image"]
        F1 --> G1["semantic mismatch"]
        G1 --> H1["target text &quot;a boy and a girl are looking at a camera in the mirror&quot;"]
    end

    subgraph "(b) ours"
        I1["clean image"] --> J1["adversarial perturbation"]
        J1 --> K1["target model"]
        K1 --> L1["image-to-text"]
        L1 --> M1["out text &quot;there are two boys brushing their teeth in the bathroom/m&quot;"]
        M1 --> N1["output text &quot;a man and a girl are looking at the camera in the mirror&quot;"]
        N1 --> O1["semantic mismatch"]
        O1 --> P1["target text &quot;a boy and a girl are looking at a camera in the mirror&quot;"]
    end

    A1 --> B1 --> C1 --> D1 --> E1 --> F1 --> G1 --> H1 --> I1 --> J1 --> K1 --> L1 --> M1 --> N1 --> O1 --> P1 --> Q["ours"]
    style "(a)" fill:#f9f,stroke:#333
    style "(b)" fill:#bbf,stroke:#333
```
</details>

Fig. 1. The semantic loss problem existing in existing gray-box targeted attack methods.

to output harmful content such as political slogans and hate speech by making imperceptible perturbations to images [7]–[9].

To gain insight into the reliability and trustworthiness of the image-to-text models, a series of adversarial attack methods have been proposed to poison the outputted textual descriptions of given images [7]–[10]. Specifically, based on the attacker's level of access to information about the target model, they can be divided into three categories: white-box attacks [7], [10], [11], gray-box attacks [8], [9], and black-box attacks [12], [13]. The white-box attacks can obtain target models' information including the entire architecture, parameters, gradients of both the image encoder and text decoder, and probability of each word of the output text. Gray-box attacks can only access the architecture, parameters, and gradients of the image encoder, while black-box attacks cannot access any internal information of the target model, but only the output text of the model. Furthermore, black-box attacks can be divided into score-based and decision-based attacks. Score-based black-box attacks can access the probability of each word of the output text [13], while decision-based black-box attacks can only access the output text [12], [14], [15].

Because less information about the target models is provided, decision-based black-box attacks are more challenging than other categories $[15]$ . Additionally, these attack methods can be categorized based on whether the attacker is able to specify the incorrect output text, dividing them into two types: targeted and untargeted attacks $[8]$ , $[16]$ .

Although numerous adversarial attack methods for image-to-text models have been proposed, to our best knowledge, the study on black-box attacks is under-explored, especially decision-based black-box targeted attacks. This kind of attack is more challenging due to the following reasons. Firstly, less information on the target model can be accessed. Specifically, only the output text instead of gradients, architectures, parameters, and the probability of each word in the output text is available. Secondly, the attackers not only cause the target model to output incorrect text, but also outputs the specified target text. Existing attacks easily suffer from the loss of semantics, resulting in the inability to effectively output the specified target text. Figure 1 (a) show that transfer+query [9] fabricates one target text to poison the target image-to-text model, leading to this model outputting an incorrect text. However, the output text could mismatch the original semantics of the target text, as the target image-to-text model may focus on secondary information while ignoring the crucial semantics of the target text behind the target image, resulting in semantic loss. More examples are in Appendix ??.

To narrow the research gap, we propose a decision-based black-box targeted attack approach for image-to-text models. In our work, only the output text of the target model can be accessed, which is closer to the real-world cases $[12]$ . Additionally, Figure 1 (b) demonstrates our targeted attack method, which optimizes against the target text directly under the decision-based black-box conditions, preventing semantic loss and maintaining semantic consistency with the target text.

Perturbing pixels in the image can change the output text. Therefore, the objective of the targeted attack can be considered to find the imperceptible pixel modification to make the output text similar to the target text. In this manner, the targeted attack can be formulated as a large-scale optimization problem, where pixels are decision variables and the optimization objective is to poison the output text. Inspired by the distinctive competency of evolutionary algorithms for solving large-scale optimization problems $[17]$ – $[19]$ , we develop a dedicated evolutionary algorithm-based framework for decision-based black-box targeted attacks on image-to-text models. However, directly applying evolutionary algorithms to solve this large-scale optimization problem could suffer from low search efficiency, due to the numerous pixels and their wide range of values. To address the issue, we embed three-step processes, i.e., Ask, Attend, Attack, into the proposed evolutionary algorithm-based attacks. As shown in Figure 2, during the Ask stage, attackers can arbitrarily specify words related to certain semantics, such as photograph. Then, candidate words (e.g., camera, scenic, and phone) that are related to certain semantics are searched. Meanwhile, these words are close to the clean image in the feature space of the target image-to-text model. By selecting words from the candidate words, the target text (e.g., a cute girl using a phone to take pictures of the fantastic TV) related to the attacker's specified semantics can be formed to poison the target model. Subsequently, based on the attention mechanism, Attend identifies the crucial regions of the clean image (e.g. attention heatmap), thus reducing the search space for the subsequent Attack. Lastly, Attack uses a differential evolution strategy to impose imperceptible adversarial perturbations to the crucial regions, where the optimization objective is to minimize the discrepancy between the target text in Ask stage and the output text of the target model. Our contributions can be summarized as follows:

![](images/e41acc8edd4441192b5a8a7ee497f7989d2a03a0ba8582bdcfec7f0789f6e7c5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["clean image"] --> B["Ask"]
    B --> C["target semantics photograph"]
    C --> D["take / using / hold / -- / cutting / walk / put"]
    D --> E["picture / photo / camera / -- / phone / TV / girl"]
    E --> F["stunning / photogenics / cute / -- / fantastic / clever / scenic"]
    F --> G["target semantics directory"]
    H["attention heatmap"] --> I["Attend"]
    I --> J["a cute girl using phone to take pictures of the fantastic TV"]
    J --> K["target text"]
    L["Attack"] --> M["adversarial image"]
    M --> N["target model"]
    N --> O["the cute girl is using a cell phone to take a picture of fantastic TV"]
    O --> P "% output text"]
```
</details>

Fig. 2. Diagram of our decision-based black-box targeted attack method Ask, Attend, Attack.

1) We first propose a decision-based black-box targeted attack Ask, Attend, Attack (AAA) for image-to-text models. Specifically, our method achieves targeted attacks without losing semantics while only the model's output text can be accessed.   
2) We designed a target semantic directory to guide attackers in creating target text and utilized attention heatmaps to significantly reduce search space. This improves the search efficiency of evolutionary algorithms in adversarial attacks and makes attacks difficult to perceive.   
3) We conducted extensive experiments on the Transformer-based VIT-GPT2 model and CNN+RNN-based Show-Attend-Tell model, which are the two most-used image-to-text models in HuggingFace, and surprisingly found that our decision-based black-box method has stronger attack performance than existing gray-box methods.

# II. RELATED WORK

# A. White-box Attack

In white-box attacks, the attacker has full access to all parameters, gradients, architecture of the target model, and the probability of each word of the output text. The authors in $[7]$ add invisible perturbations to the image to make the image-to-text model produce wrong or targeted text outputs. The authors in $[20]$ add global or local perturbations to the image to make the vision and language models unable to correctly locate and describe the content of the image. The authors in $[11]$ modify the content of the image at the semantic level to make the image-to-text model output text that is inconsistent with the original image. The authors in

[21] crafts adversarial examples with semantic embedding of targeted captions as perturbation in the complex domain. The authors in [22] preserves the accuracy of non-target words while effectively removing target words from the generated captions. The authors in [23] generate coherent and contextually rich story endings by integrating textual narratives with relevant visual cues. The authors in [10] add limited-area perturbations to the image to make the image-to-text model fail to correctly describe the content of the perturbed area. The above methods require complete information of the image-to-text target model, including architecture, gradients, parameters, and probability distribution of the output text, which limits their practicality.

# B. Gray-box Attack

To improve the practicality of adversarial attacks for image-to-text models, recent research explores how to attack with partial knowledge of the target model. All existing gray-box attack studies $[8]$ , $[9]$ , $[16]$ , $[24]$ assume full access to the image encoder of the image-to-text model. The basic idea of gray-box targeted attacks is to reduce the distance between the adversarial image and the target image generated based on the target text in the image encoder's feature space. The authors in $[24]$ generates adversarial images to mimic the feature representation of original images. The authors in $[16]$ use a generative model to destroy the image encoder's features, achieving the untargeted attack. The authors in $[8]$ minimize the feature distance in the image encoder between the adversarial image and the target image, thereby using gradient back-propagation to optimize the adversarial image and achieve the targeted attack. The authors in $[9]$ combine existing gray-box method $[8]$ with pseudo gradient estimation method $[25]$ to achieve better performance in targeted attack. It is worth noting that they $[9]$ call their method a black-box attack, but since they use the image encoder of the target model as the surrogate model, we classify their method as a gray-box attack. These gray-box attacks on image-to-text models are more practical than white-box attacks, but it is still unrealistic to assume that attackers can access the image encoder of the image-to-text model. Moreover, existing gray-box methods may have poor targeted attack performance due to the semantic loss mentioned above.

# III. METHODOLOGY

# A. Problem Formulation

The image-to-text model $\mathcal{G}:\mathcal{X}\to\mathcal{Y}$ maps the image domain $\mathcal{X}$ to the text domain $\mathcal{Y}$ . A well-trained model should be able to accurately describe the content of the image using grammatically correct and contextually coherent text. Given a target text $y_{t}$ , the attacker's goal is to find an adversarial image $\mathbf{x}_{\mathrm{adv}}$ that is visually similar to clean image $\mathbf{x}$ and can generate an adversarial text $y_{adv}$ that is semantically similar to $y_{t}$ . We formalize the optimization problem for black-box targeted attack as:

$$
\arg \max _ {\mathbf {x} _ {a d v}} S (\mathcal {G} (\mathbf {x} _ {\mathbf {a d v}}), y _ {t}) \text {   s.t.   } \frac {1}{n} \sum_ {i = 1} ^ {n} \| \mathbf {x} _ {\mathbf {a d v}} (i) - \mathbf {x} (i) \| \leq \epsilon , \tag {1}
$$

where $S(\cdot,\cdot)$ represents the semantic similarity function between two texts, $\epsilon$ is the threshold for the average perturbation size per pixel, $\mathbf{x}_{\mathbf{adv}}(i)$ and $\mathbf{x}(i)$ represents the value of the i-th pixel in the adversarial and clean images. n is the total number of pixels in all channels of the image.

# B. Overview

To enhance the efficiency and stealth of decision-based black-box attacks, we propose the Ask, Attend, Attack (AAA) framework as shown in Figure 2. Ask: We compile a semantic dictionary from words within the input image's search space that align with the attacker's specified semantics. This facilitates targeted text generation, meeting the attacker's target semantics while simplifying the search process. Attend: We employ attention visualization and a surrogate model to generate an attention heatmap for the target text on the image, narrowing the search to significant decision variables and enhancing perturbation stealth. Attack: We use the differential evolution in the reduced search space to find the optimal solution that can mislead the target model to output target text. The framework's pseudo-code is detailed in Appendix ??

# C. Ask Stage

According to the target semantics, the goal of Ask is to find words in the feature space of the target model to form a target semantic dictionary. These words should be closer to the input image. Firstly, we treat each pixel in each channel of image x as a variable, which means the search space size is the product of length, width, and number of channels. And then generate NP (number of population) individuals to form a population based on the following formula:

$$
\mathbf {x} _ {j} (i) = \mathbf {x} (i) + r a n d (- 1, 1) \cdot \eta , \tag {2}
$$

where $\mathbf{x}(i)$ is the i-th variable of clean image x, $\mathbf{x}_{j}(i)$ is the i-th variable of the j-th individual in the population, $\eta$ is a hyperparameter about the maximum search range, $rand(-1,1)$ is a random number from the range of -1 to 1.

Secondly, for each variable, random mutation occurs between different individuals. The mutation for the i-th variable of the j-th individual $\mathbf{x}_{j}(i)$ is as follows:

$$
\mathbf {v} _ {j} ^ {g} (i) = \mathbf {x} _ {r 1} ^ {g} (i) + F * (\mathbf {x} _ {r 2} ^ {g} (i) - \mathbf {x} _ {r 3} ^ {g} (i)), \tag {3}
$$

where $\mathbf{v}_{j}^{g}(i)$ is the mutated variable for mutation in the g-th generation of $\mathbf{x}_{j}(i)$ . $\mathbf{x}_{r1}^{g}(i)$ , $\mathbf{x}_{r2}^{g}(i)$ , and $\mathbf{x}_{r3}^{g}(i)$ are three randomly selected individuals from the current population who are different from each other, F is the scaling factor.

Thirdly, each individual crossovers with the mutated individuals with a certain probability of generating candidate individuals. The formula is as follows:

$$
\mathbf {u} _ {j} ^ {g} (i) = \left\{ \begin{array}{l l} \mathbf {v} _ {j} ^ {g} (i), & \text { if   } \operatorname{rand} (0, 1) \leq C R, \\ \mathbf {x} _ {j} ^ {g} (i), & \text { otherwise }, \end{array} \right. \tag {4}
$$

where CR is crossover probability factor, $\mathbf{u}_{j}^{g}(i)$ is the i-th variable of the candidate individual in the g-th generation of the j-th individual in the population.

Fourthly, we use WordNet [26], a synonym dictionary, to measure the similarity between the target semantics and each individual's output text. WordNet groups words with the same semantics into synonyms, each representing a basic concept. We use WordNet to count the same semantic words m between each individual's output text and the target semantics. We calculate the Precision=(m/t) and Recall=(m/r), where t is the output text word count and r is the target semantics word count. Then, we calculate semantic similarity using the following formula:

$$
S _ {s e m} = \frac {(1 - \gamma (\frac {c h}{m}) ^ {\theta}) (\alpha^ {2} + 1) \cdot P r e c i s i o n \cdot R e c a l l}{\alpha^ {2} \cdot P r e c i s i o n + R e c a l l}, \tag {5}
$$

where $S_{seg}$ is the semantic similarity between the individual's output text and the target semantics [26], $\alpha$ balances the precision and recall weights, $\gamma$ and $\theta$ control the penalty factor strength, $ch$ is the number of consecutive word sets that match between the output text and the target semantics, with fewer chunks meaning more consistent word order.

Ultimately, we select offspring based on $S_{sem}$ , choosing the current and candidate individuals that match the target semantics better as the next generation:

$$
\mathbf {x} _ {j} ^ {g + 1} = \left\{ \begin{array}{l l} \mathbf {u} _ {j} ^ {g}, & S _ {\text {sem}} (\mathcal {G} (\mathbf {u} _ {j} ^ {g}), T S) \geq S _ {\text {sem}} (\mathcal {G} (\mathbf {x} _ {j} ^ {g}), T S), \\ \mathbf {x} _ {j} ^ {g}, & \text {otherwise}, \end{array} \right. \tag {6}
$$

where $TS$ is the attacker's target semantics. We extract nouns, adjectives, and verbs from the output texts of all the more semantically relevant and preserved individuals of each generation, expanding the target semantic dictionary. We use $\mathcal{G}(\mathbf{x}_j^{g + 1}) = \{w_1,w_2,\dots ,w_n\}$ to represent the target model $\mathcal{G}$ 's output text for the next generation of individuals, where $w_{i}$ is the $i$ -th word of the text and $n$ is the word count. Then we use the following formula to extract important words and make a dictionary:

$$
\mathbf {D} _ {j} ^ {g + 1} = \{w \in \mathcal {G} (\mathbf {x} _ {j} ^ {g + 1}) \mid w \text {   is   noun,   adjective   or   verb } \}, \tag {7}
$$

where $D_{j}^{g+1}$ is the dictionary for the preserved individual. We combine the dictionaries of each preserved individual in each generation to get the target semantic dictionary $D = D_{1}^{2} \cup D_{2}^{2} \cup \cdots \cup D_{NP}^{m}$ , where m is the total number of generation. The attacker selects words from dictionary D that match the specified semantics to make the target text $y_{t}$ . Words in dictionary D near input image x in feature space enhance searchability, enabling more efficient targeted attacks.

# D. Attend Stage

The goal of $Attend$ is to calculate the target text's attention area on the image $\mathbf{x}$ . Because we do not have access to the internal information of the target model, we can only calculate the Grad-CAM attention heatmap [27] with the help of surrogate model $f$ (such as ResNet trained in ImageNet). The surrogate model's sole purpose is to the compute attention heatmap. Since different models produce similar heatmaps for the same target text and input image, selecting a well-established visual model suffices [28]. The calculation formula of attention heatmap $\mathbf{A}$ is as follows:

$$
\mathbf {A} (i, j) = \operatorname{MAX} \left(0, \frac {1}{Z} \sum_ {k} \sum_ {i} \sum_ {j} \cdot \frac {\partial y ^ {c ^ {*}}}{\partial \mathcal {F} _ {k} (i , j)} \cdot \mathcal {F} _ {k} (i, j)\right), \tag {8}
$$

where $\mathbf{A}(i,j)$ is the decision-making contribution of the image to the target text at pixel $(i,j)$ , $\mathcal{F}_{k}(i,j)$ is the pixel $(i,j)$ of the feature map of the k-th convolution kernel of the last convolutional layer of the surrogate model f, Z is the feature map's pixel count, $y^{c^{*}}$ is the probability that f predicts that the image x belongs to class $c^{*}$ . We use $C=\{c_{1},c_{2},\cdots,c_{1000}\}$ for the ImageNet category names, where $c_{i}$ is the i-th category name. We make the category text $y_{c_{i}}=$ “a photo of” + $c_{i}$ from the category name $c_{i}$ . We calculate the category $c^{*}$ as:

$$
c ^ {*} = \underset {c _ {i} \in \mathbf {C}} {\operatorname{argmax}} \frac {E (y _ {t}) \cdot E (y _ {c _ {i}})}{\| E (y _ {t}) \| _ {2} \| E (y _ {c _ {i}}) \| _ {2}}, \tag {9}
$$

where $E$ is the text encoder of the pre-trained CLIP model, and $c^*$ is the closest category to the target text. We substitute $c^*$ into Formula 8 to get the target text's attention heatmap A. A(i,j) is the pixel (i,j)'s contribution to the target text. 1 means more contribution, and 0 means less contribution.

# E. Attack Stage

The goal of Attack is to search for the best individual (adversarial sample) that outputs the target text $y_{t}$ in the smaller search space reduced by the attention heatmap. Firstly, we copy the attention heatmap A three times in the channel dimension to match the shape of the image x. We generated NP (number of population) individuals as a population with this formula:

$$
\mathbf {x} _ {j} (i) = \mathbf {x} (i) + r a n d (- \mathbf {A} (i), \mathbf {A} (i)) \cdot \eta , \tag {10}
$$

where $\mathbf{x}(i)$ is the i-th variable of clean image x, $\mathbf{x}_{j}(i)$ is the i-th variable of the j-th individual in the population, $\mathbf{A}(i)$ is the contribution of the i-th variable to the target text, and $rand(-\mathbf{A}(i), \mathbf{A}(i))$ is a random number in the range from $-\mathbf{A}(i)$ to $\mathbf{A}(i)$ . The value of A is less than 1, and its mean and median are about [0.3,0.4]. The search space volume from the attention heatmap is much smaller than a hypersphere with radius $\eta$ , because the radius and volume have an exponential relationship. This improves the search efficiency and concealment of adversarial perturbation.

TABLE I PERFORMANCE COMPARISON (%) OF DIFFERENT ATTACK METHODS. 

<table><tr><td rowspan="2">ε</td><td rowspan="2">Attack Methods</td><td colspan="4">VIT-GPT2</td><td colspan="4">Show-Attend-Tell</td></tr><tr><td>METEOR</td><td>BLEU</td><td>CLIP</td><td>SPICE</td><td>METEOR</td><td>BLEU</td><td>CLIP</td><td>SPICE</td></tr><tr><td></td><td>Clean Sample</td><td>0.201±0.11</td><td>0.24±0.11</td><td>0.64±0.07</td><td>0.156±0.07</td><td>0.21±0.11</td><td>0.229±0.13</td><td>0.646±0.09</td><td>0.179±0.08</td></tr><tr><td rowspan="7">25</td><td>transfer (black)</td><td>0.206±0.11</td><td>0.246±0.11</td><td>0.639±0.07</td><td>0.165±0.07</td><td>0.211±0.12</td><td>0.225±0.14</td><td>0.648±0.09</td><td>0.185±0.11</td></tr><tr><td>transfer+query (black)</td><td>0.221±0.16</td><td>0.264±0.15</td><td>0.651±0.18</td><td>0.167±0.07</td><td>0.219±0.11</td><td>0.231±0.14</td><td>0.654±0.05</td><td>0.187±0.14</td></tr><tr><td>transfer (gray)</td><td>0.414±0.23</td><td>0.396±0.14</td><td>0.821±0.09</td><td>0.32±0.16</td><td>0.382±0.26</td><td>0.348±0.17</td><td>0.782±0.11</td><td>0.299±0.17</td></tr><tr><td>transfer+query (gray)</td><td>0.433±0.21</td><td>0.411±0.12</td><td>0.832±0.13</td><td>0.35±0.09</td><td>0.401±0.21</td><td>0.355±0.15</td><td>0.794±0.11</td><td>0.311±0.13</td></tr><tr><td>AAA (w/o Attend)</td><td>0.541±0.25</td><td>0.519±0.19</td><td>0.854±0.24</td><td>0.477±0.11</td><td>0.642±0.19</td><td>0.564±0.19</td><td>0.841±0.06</td><td>0.455±0.14</td></tr><tr><td>AAA (w/o Ask)</td><td>0.398±0.21</td><td>0.384±0.18</td><td>0.795±0.25</td><td>0.412±0.13</td><td>0.364±0.21</td><td>0.322±0.19</td><td>0.754±0.08</td><td>0.376±0.13</td></tr><tr><td>AAA</td><td>0.696±0.21</td><td>0.658±0.22</td><td>0.952±0.29</td><td>0.634±0.15</td><td>0.855±0.15</td><td>0.799±0.21</td><td>0.964±0.04</td><td>0.786±0.14</td></tr><tr><td rowspan="7">15</td><td>transfer (black)</td><td>0.204±0.09</td><td>0.241±0.15</td><td>0.627±0.18</td><td>0.164±0.07</td><td>0.232±0.13</td><td>0.236±0.14</td><td>0.643±0.08</td><td>0.187±0.09</td></tr><tr><td>transfer+query (black)</td><td>0.211±0.14</td><td>0.256±0.15</td><td>0.644±0.15</td><td>0.181±0.09</td><td>0.245±0.13</td><td>0.246±0.11</td><td>0.656±0.06</td><td>0.203±0.09</td></tr><tr><td>transfer (gray)</td><td>0.398±0.24</td><td>0.381±0.15</td><td>0.816±0.11</td><td>0.325±0.16</td><td>0.361±0.24</td><td>0.359±0.17</td><td>0.778±0.11</td><td>0.296±0.16</td></tr><tr><td>transfer+query (gray)</td><td>0.408±0.19</td><td>0.399±0.11</td><td>0.824±0.15</td><td>0.341±0.13</td><td>0.375±0.19</td><td>0.368±0.15</td><td>0.784±0.11</td><td>0.311±0.13</td></tr><tr><td>AAA (w/o Attend)</td><td>0.461±0.21</td><td>0.423±0.15</td><td>0.808±0.11</td><td>0.375±0.09</td><td>0.438±0.15</td><td>0.434±0.16</td><td>0.827±0.04</td><td>0.422±0.14</td></tr><tr><td>AAA (w/o Ask)</td><td>0.378±0.25</td><td>0.361±0.17</td><td>0.768±0.15</td><td>0.356±0.15</td><td>0.341±0.15</td><td>0.337±0.18</td><td>0.749±0.07</td><td>0.365±0.13</td></tr><tr><td>AAA</td><td>0.556±0.31</td><td>0.504±0.26</td><td>0.851±0.12</td><td>0.44±0.17</td><td>0.617±0.25</td><td>0.574±0.22</td><td>0.913±0.05</td><td>0.553±0.14</td></tr></table>

Secondly, in order to accelerate convergence and better find the global optimal solution, we use the following Current-ToBest mutation [29]:

$$
\mathbf {v} _ {j} ^ {g} (i) = \mathbf {x} _ {j} ^ {g} (i) + F * \left(\mathbf {x} _ {r 1} ^ {g} (i) - \mathbf {x} _ {r 2} ^ {g} (i)\right) \tag {11}
$$

$$
+ F * (\mathbf {x} _ {b e s t} ^ {g} (i) - \mathbf {x} _ {j} ^ {g} (i)),
$$

where $\mathbf{x}_{j}^{g}(i)$ is the i-th variable of the j-th individual in the g-th generation, $\mathbf{v}_{j}^{g}(i)$ is the mutated variable, $x_{best}^{g}$ is the best fitness individual in the g-th generation population, $x_{r1}^{g}$ and $x_{r2}^{g}$ are two randomly selected individuals in the g-th generation population, and F is the scaling factor. The main advantage of this mutation strategy is that it combines the information of the current individual $x_{j}$ and the best fitness individual $x_{best}$ , which can better guide the search process towards the direction of the optimal solution.

Thirdly, we use Formula 4 to calculate the candidate individual $\mathbf{u}_{j}^{g}(i)$ . We design the following formula to calculate the deep feature similarity $S_{clip}$ between two texts (u and v):

$$
S _ {c l i p} = 1 - \frac {E (u) \cdot E (v)}{\| E (u) \| _ {2} \| E (v) \| _ {2}}, \tag {12}
$$

where E is the text encoder of the pre-trained CLIP model. Text is discrete and complex, so it cannot calculate the distance directly [30]. Therefore, we use the CLIP text encoder E to extract the deep features of the texts, and then calculate the feature distance to obtain the similarity $S_{clip}$ between the texts. The closer $S_{clip}$ is to 0, the higher the similarity between the two texts u and v.

Ultimately, we select offspring using the following formula:

$$
\mathbf {x} _ {j} ^ {g + 1} = \left\{ \begin{array}{l l} \mathbf {u} _ {j} ^ {g}, & S _ {\text { clip }} (\mathcal {G} (\mathbf {u} _ {j} ^ {g}), y _ {t}) \leq S _ {\text { clip }} (\mathcal {G} (\mathbf {x} _ {j} ^ {g}), y _ {t}), \\ \mathbf {x} _ {j} ^ {g}, & \text { otherwise }, \end{array} \right. \tag {13}
$$

where $x_{j}^{g+1}$ is the next individual with closer feature distance between the output text and the target text $y_{t}$ . After performing the above evolutionary calculations multiple times, the optimal solution (adversarial sample) for outputting the target text is found.

# IV. EVALUATION AND RESULTS

# A. Experiment setups

a) Model and dataset: We experimented with the two most-used image-to-text models on HuggingFace: VIT-GPT2

![](images/37208aeaa641c90e5dd01ebd29a311e43f2501b85d925a4d72783098dfc47c3f.jpg)

Fig. 3. We compared the convergence curves of populations with and without Attend under the same perturbation size $\epsilon$ in (a-b). The fitness function is $S_{clip}$ in Formula 12, where lower values mean stronger attacks. The dashed line is the average fitness value, and the solid line is the best fitness value. The green line is AAA and the red line is AAA w/o Attend. (c) shows the attention heatmap. (d) and (e) show the visual effects of adversarial image with and without Attend, with minimal perturbation of 100% attack success rate.   
![](images/60b3bce13b33da3cd43c2130468e364f1706de3e54782f5490160a6e4c441716.jpg)  
Fig. 4. Grad-CAM attention heatmaps of different surrogate models for the same target text a woman is holding a pair of shoes. M is METEOR, B is BLEU, C is CLIP, S is SPICE.

(Transformer-based) [31] and Show-Attend-Tell (CNN+RNN-based) [32]. VIT-GPT2 was trained on ImageNet-21k. Show-Attend-Tell was trained on MSCOCO-2014. We only used the target model's output text, not its internal information like gradients, parameters, or word probability. Following this work [8], we used Flick30k as our dataset, which has 31783 images and 5 caption texts each. We removed samples with less than 0.7 similarities between predicted text and truth text to ensure the target model's accuracy on clean images.

b) Evaluation metrics: We used these evaluation metrics in our experiments: (1) BLEU(#4), an early machine translation metric that measures text precision [33]. 1 means similar, and 0 means dissimilar. (2) METEOR, a more comprehensive metric that considers synonyms, stems, word order, etc [26]. 1 means similar, and 0 means dissimilar. (3) CLIP, the distance between the CLIP text encoder's deep features for two texts [34]. 1 means similar, and 0 means dissimilar. (4) SPICE, an evaluation metric tailored for image-to-text models [35]. 1 means similar, and 0 means dissimilar. (5) $\epsilon$ , the mean perturbation size of each pixel of the adversarial sample [8].

# B. Experiment results

a) Comparison experiment of existing gray-box attacks.: We evaluate state-of-the-art gray-box attacks [8], [9] on image-to-text models. We designate the gray-box attack [8] as transfer (gray) and the one [9] as transfer+query (gray). To simulate a black-box environment, we adapted these gray-box attacks by employing the CLIP model's image encoder in lieu of the target model's encoder, resulting in "transfer (black)" and "transfer+query (black)" variants. As depicted in Table I, adversarial samples generated by the original gray-box attacks exhibit a marked increase in textual similarity to the target text when compared to clean samples. Conversely, the black-box adaptations maintain a similarity level akin to that of clean samples, indicating a significant loss of attack capability upon changing the image encoder. This underscores the dependency of gray-box attacks on the target model's image encoder. Our proposed method AAA demonstrates superior attack performance in black-box scenarios compared to the existing methods in their native gray-box settings. This is attributed to the semantic loss inherent in existing gray-box attacks, which constrains their attacking potential. It is noteworthy that our work represents the first black-box attack on image-to-text models. So we can only compare our approach with existing gray-box attacks. We have adapted these gray-box attacks into a black-box version solely to demonstrate their ineffectiveness in a black-box scenario.

b) Ablation experiment of our black-box attack.: We conducted ablation experiments on our AAA method. AAA (w/o Attend) means no attention heatmap to reduce the search space, but the proportional reduction of the search range. AAA (w/o Ask) means the target text is not from the target semantic dictionary, but random words. Table I shows that losing any module decreases our attack performance. In addition, Ask performs worse than AAA (w/o Attend), indicating that finding a target text with lower search difficulty contributes relatively more to the performance of our targeted attack.

c) Qualitative experiment of attention.: We presented the optimization curves of AAA and AAA (w/o Attend) in Figure 3. Figure 3 (a) and (b) illustrate the best and average fitness values during AAA and AAA (w/o Attend) optimization of VIT-GPT2 and Show-Attend-Tell. It is evident that the inclusion of Attend expedites and enhances the convergence of the population, with an equivalent perturbation size. Consequently, AAA exhibits more effective concealment in adversarial perturbations, maintaining the same level of attack efficacy, as depicted in Figures 3 (d) and (e). Furthermore, we evaluated the impact of selecting different surrogate models during Attend. Notably, the sole function of the surrogate model is to compute the attention heatmap. Figure 4 demonstrates that, despite significant structural variances among several surrogate models, they produce strikingly similar attention heatmaps for the same target text and input images. This similarity arises from mapping the target text to the most pertinent category within the surrogate model's label space (as Formula 9). The position of the same category of objects on the same picture

is constant, and the model needs to focus on the object first, no matter what structure it is [28]. Performance comparisons, as shown in Figure 4, indicate that the similarity in attention heatmaps across different surrogate models leads to similar final attack performances. Therefore, we opted for a stable, well-established, pre-trained model, such as ResNet-50, to serve as our surrogate model.

d) Qualitative experiment of different perturbation sizes.: We used the words mirror, cell phone, man, looking at from the target semantic dictionary (as shown in Appendix ??) to make the target text a man is looking at a cell phone in a mirror. We compared output texts of our black-box method AAA and the existing gray-box method [9] for adversarial samples with different $\epsilon$ , the average pixel perturbation size, in Figure 5. The same conclusion drawn from both methods is that bigger perturbation causes worse concealment and better attack performance; too small perturbation causes attack failure. Moreover, (f) and (j) in Figure 5 show that the existing methods have a semantic loss that limits their attack performance. Subjectively, target image (j) accurately draws the semantics of the target text, and the output text of adversarial image (f) perfectly describes the content of the target image (j). However the adversarial sample (f)'s output text does not have the semantics of the target text. Our method does not have semantic loss, so our black-box method AAA does a better targeted attack than the existing gray-box method. More examples of semantic loss are in Appendix ??.

e) Comparison experiment on computation time.: We evaluated the computational efficiency of various attack methodologies for generating adversarial samples in image-to-text models. As depicted in Figure 6, our black-box attack method AAA, demonstrates a longer computation time to reach an optimal solution compared to existing gray-box attacks. For instance, the transfer approach [8] illustrated in Figure 6 (a) produces an adversarial sample with a CLIP score of 0.82 within a mere 29 seconds, while the transfer+query approach [9] achieves a CLIP score of 0.85 in just 97 seconds. Conversely, our AAA method requires 151 seconds to generate an adversarial sample with a superior CLIP score of 0.951. The shorter computation times of the existing gray-box methods are expected due to their ability to access real gradients, which significantly expedites the optimization process. Given that adversarial attacks are not time-sensitive operations and considering that our AAA method delivers a more potent attack capability and is applicable in a broader range of realistic black-box scenarios, the trade-off for a higher computational cost is deemed acceptable. Additional experiments on similarity measurements are included in the Appendix ??.

f) Further analyses.: Firstly, we show the impact of different forms of target semantics TS in Ask on the target semantic dictionary, as shown in Appendix ??. More ambiguous target semantics can enrich the target semantic dictionary, which also means that the attacker has more choices when designing $y_{t}$ . Secondly, we show the effect of different word selection strategies of $y_{t}$ based on target semantic dictionary on the final attack effect, as shown in Appendix ??. Thirdly,

![](images/fc61d0af51439456b6cfea3868af97c428fdbb0fc55bba6f77e92f796999ee42.jpg)

<details>
<summary>text_image</summary>

M:0.99 B:1.0 C:1.0
a man is looking at a cell phone in a mirror
M:0.81 B:0.82 C:0.97
a man is looking at his phone in a mirror
M:0.52 B:0.53 C:0.8
a man is standing in the street with a cell phone
M:0.14 B:0.17 C:0.67
a young boy is playing with a toilet
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(h)
(i)
(j)
(k)
(l)
(m)
n
ours
existing work
a man standing in a living room holding a wii remote
M:0.34 B:0.32 C:0.64
a man in a suit is looking at a remote in a room
M:0.67 B:0.59 C:0.77
a man is standing in a room with a chair
M:0.47 B:0.43 C:0.72
a man in a suit and tie
M:0.29 B:0.21 C:0.69
a man standing in a living room holding a wii remote
</details>

Fig. 5. Performance of adversarial image attacks varies with perturbation size $\epsilon$ . The $\epsilon$ of (a) and (f) is 25, $\epsilon$ of (b) and (g) is 15, $\epsilon$ of (c) and (h) is 10, $\epsilon$ of (d) and (i) is 5. (e) is our attention heatmap of the target text on the image. (j) is the target image generated based on the target text used in existing works. M is METEOR score, B is BLEU score, and C is CLIP score.

![](images/4914bd72ce421e718bcc1e1d81528bd512d6c8a1c34b8980ab9203da99f41794.jpg)

<details>
<summary>line</summary>

| Model | Time (second) | CLIP Score |
|-------|---------------|------------|
| VIT-GPT2 | 0 | 0.65 |
| VIT-GPT2 | 100 | 0.75 |
| VIT-GPT2 | 200 | 0.85 |
| VIT-GPT2 | 300 | 0.90 |
| VIT-GPT2 | 400 | 0.92 |
| VIT-GPT2 | 500 | 0.93 |
| VIT-GPT2 | 600 | 0.94 |
| VIT-GPT2 | 700 | 0.95 |
| VIT-GPT2 | 800 | 0.96 |
| VIT-GPT2 | 900 | 0.97 |
| VIT-GPT2 | 1000 | 0.98 |
| VIT-GPT2 | 1100 | 0.99 |
| VIT-GPT2 | 1200 | 1.00 |
| VIT-GPT2 | 1300 | 1.01 |
| VIT-GPT2 | 1400 | 1.02 |
| VIT-GPT2 | 1500 | 1.03 |
| VIT-GPT2 | 1600 | 1.04 |
| VIT-GPT2 | 1700 | 1.05 |
| VIT-GPT2 | 1800 | 1.06 |
| VIT-GPT2 | 1900 | 1.07 |
| VIT-GPT2 | 2000 | 1.08 |
| VIT-GPT2 | 2100 | 1.09 |
| VIT-GPT2 | 2200 | 1.10 |
| VIT-GPT2 | 2300 | 1.11 |
| VIT-GPT2 | 2400 | 1.12 |
| VIT-GPT2 | 2500 | 1.13 |
| VIT-GPT2 | 2600 | 1.14 |
| VIT-GPT2 | 2700 | 1.15 |
| VIT-GPT2 | 2800 | 1.16 |
| VIT-GPT2 | 2900 | 1.17 |
| VIT-GPT2 | 3000 | 1.18 |
| VIT-GPT2 | 3100 | 1.19 |
| VIT-GPT2 | 3200 | 1.20 |
| VIT-GPT2 | 3300 | 1.21 |
| VIT-GPT2 | 3400 | 1.22 |
| VIT-GPT2 | 3500 | 1.23 |
| VIT-GPT2 | 3600 | 1.24 |
| VIT-GPT2 | 3700 | 1.25 |
| VIT-GPT2 | 3800 | 1.26 |
| VIT-GPT2 | 3900 | 1.27 |
| VIT-GPT2 | 4000 | 1.28 |
| VIT-GPT2 | 4100 | 1.29 |
| VIT-GPT2 | 4200 | 1.30 |
| VIT-GPT2 | 4300 | 1.31 |
| VIT-GPT2 | 4400 | 1.32 |
| VIT-GPT2 | 4500 | 1.33 |
| VIT-GPT2 | 4600 | 1.34 |
| VIT-GPT2 | 4700 | 1.35 |
| VIT-GPT2 | 4800 | 1.36 |
| VIT-GPT2 | 4900 | 1.37 |
| VIT-GPT2 | 5000 | 1.38 |
| VIT-GPT2 | 5100 | 1.39 |
| VIT-GPT2 | 5200 | 1.40 |
| VIT-GPT2 | 5300 | 1.41 |
| VIT-GPT2 | 5400 | 1.42 |
| VIT-GPT2 | 5500 | 1.43 |
| VIT-GPT2 | 5600 | 1.44 |
| VIT-GPT2 | 5700 | 1.45 |
| VIT-GPT2 | 5800 | 1.46 |
| VIT-GPT2 | 5900 | 1.47 |
| VIT-GPT2 | 6000 | 1.48 |
| VIT-GPT2 | 6100 | 1.49 |
| VIT-GPT2 | 6200 | 1.50 |
| VIT-GPT2 | 6300 | 1.51 |
| VIT-GPT2 | 6400 | 1.52 |
| VIT-GPT2 | 6500 | 1.53 |
| VIT-GPT2 | 6600 | 1.54 |
| VIT-GPT2 | 6700 | 1.55 |
| VIT-GPT2 | 6800 | 1.56 |
| VIT-GPT2 | 6900 | 1.57 |
| VIT-GPT2 | 7000 | 1.58 |
| VIT-GPT2 | 7100 | 1.59 |
| VIT-GPT2 | 7200 | 1.60 |
| VIT-GPT2 | 7300 | 1.61 |
| VIT-GPT2 | 7400 | 1.62 |
| VIT-GPT2 | 7500 | 1.63 |
| VIT-GPT2 | 7600 | 1.64 |
| VIT-GPT2 | 7700 | 1.65 |
| VIT-GPT2 | 7800 | 1.66 |
| VIT-GPT2 | 7900 | 1.67 |
| VIT-GPT2 | 8000 | 1.68 |
| VIT-GPT2 + Transfer+query (c) - CLIP score vs Computation Time (seconds) - [a] - [b] - [c] - [d] - [e] - [f] - [g] - [h] - [i] - [j] - [k] - [l] - [m] - [n] - [o] - [p] - [q] - [r] - [s] - [t] - [u] - [v] - [w] - [x] - [y] - [z] - [t] - [u] - [v] - [w] - [x] - [y] - [z] - [u] - [v] - [w] - [x] - [u] - [v] - [w] - [x] - [u] - [v] - [w] - [x] - [u] - [v] - [w] - [x] - [u] - [v] - [w] - [x] - [u] - [v] - [w] - [x] - [u] - [v] - [w] - [x] - [v] - [u] - [w] - [w] - [x] - [u] - [w] - [w] - [x] - [u] - [w] - [w] - [x] - [u] - [w] - [w] - [x] - [u] - [w] - [w] - [x] - [u] - [w] - [w] - [x] - [u] - [w] - [w] - [x] - [U] - [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U], [U],[U], [U], [U], [U], [U], [U], [U], [U], [U], [U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[U],[O]
</details>

Fig. 6. Comparison of computation time for generating a single adversarial sample using different adversarial attack methods. The y-axis is a measure of similarity between the generated text and the target text, with higher values indicating better target attack performance. The x-axis represents the computation time, and the shorter the time required to find a stable solution, the better.

we compare the convergence curves of different population sizes and choose a population size of 40 based on the trade-off of attack performance and convergence efficiency, as shown in Appendix ??. Furthermore, we compare the effects of different evolutionary algorithms on attack performance and convergence efficiency, as shown in Appendix ??. Additionally, to better observe the attack effect of our framework, we show more examples of attention heatmaps A, optimization convergence curves, target text $y_t$ , and output text, as shown in Appendix ??. Lastly, we discuss the limitations of our framework, defense strategies, and future work in Appendix ??.

# V. CONCLUSION

In our research, we introduce a novel and practical approach for adversarial attacks on image-to-text models. We propose the Ask, Attend, Attack (AAA) framework, a decision-based black-box attack method that achieves targeted attacks without semantic loss, even with access limited to the target model's output text. Our framework uses the target semantic directory to guide the creation of target text and attention heatmap to reduce the search space, thereby improving the efficiency of evolutionary algorithms and making our attack harder to detect. Our extensive experiments on the Transformer-based VIT-GPT2 model and the CNN+RNN-based Show-Attend-Tell model demonstrate that our decision-based black-box method outperforms existing gray-box methods in targeted attack performance. These findings highlight the vulnerabilities in current image-to-text models and underscore the need for more robust defense mechanisms, significantly contributing to the field of adversarial machine learning and enhancing the security of vision-language systems.

# REFERENCES

[1] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the International Conference on Machine Learning (ICML), pages 12888–12900. PMLR, 2022.   
[2] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.   
[3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2425–2433. IEEE, 2015.   
[4] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In Proceedings of the International Conference on Machine Learning (ICML), pages 5583–5594. PMLR, 2021.   
[5] Qiong Cao, Li Shen, Weidi Xie, Omkar M Parkhi, and Andrew Zisserman. Vggface2: A dataset for recognising faces across pose and age. In Proceedings of the IEEE International Conference on Automatic Face & Gesture Recognition (FG), pages 67–74. IEEE, 2018.   
[6] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Proceedings of the Neural Information Processing Systems (NIPS), 32, 2019.   
[7] Hongge Chen, Huan Zhang, Pinyu Chen, Jinfeng Yi, and Cho-Jui Hsieh. Attacking visual language grounding with adversarial examples: A case study on neural image captioning. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pages 2587–2597. Association for Computational Linguistics, 2018.   
[8] Raz Lapid and Moshe Sipper. I see dead people: Gray-box adversarial attack on image-to-text models. In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD), 2023.   
[9] Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan Li, Ngai-Man Cheung, and Min Lin. On evaluating adversarial robustness of large vision-language models. In Proceedings of the Neural Information Processing Systems (NIPS), 2023.   
[10] Hyun Kwon and SungHwan Kim. Restricted-area adversarial example attack for image captioning model. In Wireless Communications and Mobile Computing (WCMC). Hindawi, 2022.   
[11] Anand Bhattad, Minjin Chong, Kaizhao Liang, Bo Li, and D. A. Forsyth. Unrestricted adversarial examples via semantic manipulation. In Proceedings of the International Conference on Learning Representations (ICLR). ICLR, 2020.   
[12] Yinpeng Dong, Hang Su, Baoyuan Wu, Zhifeng Li, Wei Liu, Tong Zhang, and Jun Zhu. Efficient decision-based black-box adversarial attacks on face recognition. 2019.

[13] Yucheng Shi, Yahong Han, Qinghua Hu, Yi Yang, and Qi Tian. Query-efficient black-box adversarial attack with customized iteration and sampling. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2), 2022.   
[14] Shuai Jia, Yibing Song, Chao Ma, and Xiaokang Yang. Iou attack: Towards temporally coherent black-box adversarial attack for visual object tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.   
[15] Kaixun Jiang, Zhaoyu Chen, Hao Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, and Wenqiang Zhang. Efficient decision-based black-box patch attacks on video recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4379–4389, 2023.   
[16] Hanjie Wu, Yongtuo Liu, Hongmin Cai, and Shengfeng He. Learning transferable perturbations for image captioning. ACM Transactions on Multimedia Computing, Communications and Applications, 18(2), 2022.   
[17] Mohammad Nabi Omidvar, Xiaodong Li, and Yi Mei. Cooperative co-evolution with differential grouping for large scale optimization. IEEE Transactions on evolutionary computation, 18(3):378–393, 2013.   
[18] Zhenzhong Wang, Haokai Hong, Kai Ye, Guangen Zhang, Min Jiang, and Kay Chen Tan. Manifold interpolation for large-scale multiobjective optimization via generative adversarial networks. IEEE Transactions on Neural Networks and Learning Systems, 34(8):4631–4645, 2023.   
[19] Haokai Hong, Min Jiang, and Gary G Yen. Improving performance insensitivity of large-scale multiobjective optimization via monte carlo tree search. IEEE Transactions on Cybernetics, 2023.   
[20] Xiaojun Xu, Xinyun Chen, Chang Liu, Anna Rohrbach, Trevor Darrell, and Dawn Song. Fooling vision and language models despite localization and attention mechanism. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2018.   
[21] Shaofeng Zhang, Zheng Wang, Xing Xu, Xiang Guan, and Yang Yang. Fooled by imagination: Adversarial attack to image captioning via perturbation in complex domain. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2020.   
[22] Jiayi Ji, Xiaoshuai Sun, Yiyi Zhou, Rongrong Ji, and Fuhai Chen. Attacking image captioning towards accuracy-preserving target words removal. In Proceedings of the ACM International Conference on Multimedia (ACM MM). ACM, 2020.   
[23] Qingbao Huang, Chuan Huang, Linzhang Mo, Jielong Wei, Yi Cai, Hofung Leung, and Qing Li. Igseg: Image-guided story ending generation. In Findings of the ACL: International Journal of Conference on Natural Language Processing, 2021.   
[24] Akshay Chaturvedi and Utpal Garain. Mimic and fool: A task-agnostic adversarial attack. IEEE Transactions on Neural Networks and Learning Systems, 32(4):1801–1808, 2020.   
[25] Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17:527-566, 2017.   
[26] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization (ACL WIEEMMTS), pages 65–72, 2005.   
[27] Ramprasaath R. Selvaraju, Michael Cogswell, and Abhishek Das. Gradcam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 618–626, 2017.   
[28] Jiakai Wang, Aishan Liu, Zixin Yin, Shunchang Liu, Shiyu Tang, and Xianglong Liu. Dual attention suppression attack: Generate adversarial camouflage in physical world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8565–8574, 2021.   
[29] Jingqiao Zhang and Arthur C Sanderson. Jade: adaptive differential evolution with optional external archive. IEEE Transactions on evolutionary computation, 13(5):945–958, 2009.   
[30] Jyun-Yu Jiang, Mingyang Zhang, Cheng Li, Michael Bendersky, Nadav Golbandi, and Marc Najork. Semantic text matching for long-form documents. In Proceedings of the World Wide Web Conference (WWW), pages 795–806, 2019.   
[31] NLP Connect. vit-gpt2-image-captioning (revision 0e334c7). https://huggingface.co/nlpconnect/vit-gpt2-image-captioning, 2022.   
[32] Kelvin Xu, Jimmy Ba, and Kiros Jamie. Show, attend and tell: Neural image caption generation with visual attention. In Proceedings of the

International Conference on Machine Learning (ICML), pages 2048-2057. PMLR, 2015.   
[33] Kishore Papineni and Salim Roukos. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pages 311–318, 2002.   
[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning (ICML), pages 8748–8763. PMLR, 2021.   
[35] P. Anderson, B. Fernando, M. Johnson, and S. Gould. Spice: Semantic propositional image caption evaluation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 382–398. Springer, 2016.   
[36] Zhenzhong Wang, Qingyuan Zeng, Wanyu Lin, Min Jiang, and Kaychen Tan. Generating diagnostic and actionable explanations for fair graph neural networks. In Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI), 2024.   
[37] Rainer Storn and Kenneth Price. Differential evolution—a simple and efficient heuristic for global optimization over continuous spaces. Journal of global optimization, 11:341–359, 1997.   
[38] Wael Khatib and Peter J. Fleming. The stud ga: a mini revolution? In Proceedings of the International Conference on Parallel Problem Solving from Nature (PPSN), pages 683–691. Springer Berlin Heidelberg, 1998.