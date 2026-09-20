# LoRA Unlearns More and Retains More (Student Abstract)

Atharv Mittal

Vision and Language Group, Indian Institute of Technology Roorkee, Roorkee, Uttarakhand, India - 247667 atharv\_m@mfs.iitr.ac.in

# Abstract

Due to increasing privacy regulations and regulatory compliance, Machine Unlearning (MU) has become essential. The goal of unlearning is to remove information related to a specific class from a model. Traditional approaches achieve exact unlearning by retraining the model on the remaining dataset, but incur high computational costs. This has driven the development of more efficient unlearning techniques, including model sparsification techniques, which boost computational efficiency, but degrade the model's performance on the remaining classes. To mitigate these issues, we propose a novel method, PruneLoRA which introduces a new MU paradigm, termed prune first, then adapt, then unlearn. LoRA (Hu et al. 2022) reduces the need for large-scale parameter updates by applying low-rank updates to the model. We leverage LoRA to selectively modify a subset of the pruned model's parameters, thereby reducing the computational cost, memory requirements and improving the model's ability to retain performance on the remaining classes. Experimental Results across various metrics showcase that our method outperforms other approximate MU methods and bridges the gap between exact and approximate unlearning. Our code is available at https://github.com/vlgiitr/LoRA-Unlearn.

# Introduction

The process of removing specific data points or classes from trained machine learning models is known as machine unlearning. Its importance has intensified due to growing privacy concerns and the need to comply with evolving regulations, which enable users to request the removal of their personal data from models as part of the “right to be forgotten” in General Data Protection Regulation (GDPR).

Machine unlearning techniques can be classified into two broad categories: exact and approximate unlearning. The exact approach to machine unlearning typically involves retraining the entire model on a modified dataset, excluding the data to be forgotten. While this method guarantees the removal of the influence of a data instance from a model, it is highly computationally intensive for larger models. Approximate unlearning focuses on reducing the influence of targeted data points through efficient parameter updates. However, these methods often struggle to balance unlearning effectiveness with performance and computational efficiency. Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

One common approximate unlearning method is simple fine-tuning (FT), which fine-tunes the pre-trained model on the remaining dataset for a few training epochs, but presents its own challenges. When a model is fine-tuned to forget specific information, it often suffers from catastrophic forgetting, i.e it loses the ability to perform well on previously learned tasks, thus degrading its performance on the remaining classes. Fine-tuning can also be computationally expensive on very large models.

To address these limitations, (Liu et al. 2024) explored model sparsification techniques, where they selectively removed specific weights or neurons within the model, rather than updating the entire network before fine-tuning. By focusing on a subset of the model's parameters, they reduced overfitting and computational cost. However, despite offering improvements over standard fine-tuning, these methods still encounter challenges in effectively balancing the trade-offs between unlearning efficiency, computational cost, and maintaining overall model performance. Low-Rank Adaptation (LoRA), offers a solution that builds upon the principles of model sparsity while addressing its limitations by updating only a small subset of model parameters through low-rank matrix decomposition. Since (Biderman et al. 2024) shows that in the context of LLMs, LoRA provides a form of regularization that mitigates “forgetting” of the source domain, through rigorous experimentation, we prove using LoRA to update model parameters preserves model performance and lowers computational costs exponentially.

# Methodology

We evaluated four paradigms for machine unlearning:

1. Fine-tuning: The model is fine-tuned on the remaining dataset, using standard gradient descent techniques.   
2. Pruning + Fine-tuning: First, we apply model pruning to reduce the number of parameters. Then, the pruned model is fine-tuned on the remaining dataset (Liu et al. 2024)   
3. LoRA: Apply LoRA to selectively modify a subset of the model's parameters.   
4. Pruning + LoRA: First prune the model, then add LoRA Adapters and fine-tune.

For our experiments, we employed a ResNet50 and a Vision Transformer (ViT) and trained both on the CIFAR-10

<table><tr><td rowspan="2">Model</td><td colspan="2">UA</td><td colspan="2">MIA-Efficacy</td><td colspan="2">RA</td><td colspan="2">TA</td><td>RTE</td><td>GPU</td></tr><tr><td>5 Epochs</td><td>10 Epochs</td><td>5 Epochs</td><td>10 Epochs</td><td>5 Epochs</td><td>10 Epochs</td><td>5 Epochs</td><td>10 Epochs</td><td>(secs/epoch)</td><td>GB</td></tr><tr><td colspan="11">ResNet-50</td></tr><tr><td>Retrain</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>98.02</td><td>98.02</td><td>96.70</td><td>96.70</td><td>-</td><td>-</td></tr><tr><td>Finetune</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>92.52</td><td>96.27</td><td>88.29</td><td>91.44</td><td>137</td><td>6.9</td></tr><tr><td>Pruned</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>94.92</td><td>95.68</td><td>90.79</td><td>90.72</td><td>137</td><td>4.2</td></tr><tr><td>LoRA</td><td>97.22</td><td>100.00</td><td>100.00</td><td>100.00</td><td>96.90</td><td>97.19</td><td>95.03</td><td>93.49</td><td>122</td><td>5.8</td></tr><tr><td>Pruned LoRA</td><td>99.78</td><td>99.98</td><td>97.68</td><td>97.89</td><td>97.96</td><td>98.00</td><td>95.18</td><td>95.41</td><td>122</td><td>5.5</td></tr><tr><td colspan="11">ViT</td></tr><tr><td>Retrain</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>96.90</td><td>96.90</td><td>84.92</td><td>84.92</td><td>-</td><td>-</td></tr><tr><td>Finetune</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>87.64</td><td>86.80</td><td>79.79</td><td>79.94</td><td>132</td><td>3.2</td></tr><tr><td>Pruned</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>84.76</td><td>86.14</td><td>78.81</td><td>79.53</td><td>132</td><td>3.2</td></tr><tr><td>LoRA</td><td>87.66</td><td>95.58</td><td>98.14</td><td>99.58</td><td>97.72</td><td>97.81</td><td>85.33</td><td>85.16</td><td>48</td><td>0.7</td></tr><tr><td>Pruned LoRA</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>97.39</td><td>97.63</td><td>85.53</td><td>85.34</td><td>48</td><td>0.7</td></tr></table>

Figure 1: Results of ResNet-50 and ViT When Tested Using Various Unlearning Approaches (in percent accuracy)

dataset. Our unlearning task focused on removing the influence of the forget class while maintaining performance on the remaining classes. To establish an exact unlearning baseline, we retrain both the models on the remaining dataset for 200 and 90 epochs respectively. We used L2 Pruning to prune 50% of the specific layers in each model, Convolutional layers were pruned in ResNet50 and linear and attention layers were pruned in ViT. After final finetuning on the remaining dataset for 5/10 epochs, we evaluate the models based on the following metrics:

- Unlearning accuracy (UA): 1-Acc(Df), where Acc(Df) is the accuracy of the unlearned model on the forget dataset.   
- Membership inference attack (MIA-Efficacy): Applying the confidence-based MIA predictor to the unlearned model on the forgetting dataset (Df). A higher MIA-Efficacy implies less information about Df in the model.   
- Remaining accuracy (RA): This refers to the accuracy of the unlearned model on the retain dataset.   
- Testing accuracy (TA): This refers to the accuracy of the unlearned model on the testing dataset of the remaining classes.   
- Run-time efficiency (RTE): This measures the computation efficiency of the MU method (run time cost).   
- GPU Memory (GPU): This measures the memory requirements of the MU method for a model.

# Results

Table 1 presents the accuracy metrics for both model under the given five paradigms: It is observed that all methods achieved perfect or near-perfect Unlearning Accuracy (UA) and Membership Inference Attack (MIA) efficacy, indicating successful removal of the target class information. For ResNet-50, PruneLoRA outperformed all methods, achieving the highest Remaining (RA) and Testing accuracy (TA), while experiencing near-perfect UA. For the ViT model, PruneLoRA significantly outperformed other methods (except LoRA) in terms of RA and TA. Moreover, while LoRA demonstrated a drastically low UA, PruneLoRA achieved perfect UA. These results suggest that PruneLoRA offers a balance between effective unlearning, retained model performance, and computational efficiency.

# Future Scope

There is significant potential for further research and experimentation to strengthen and validate our hypothesis. A promising avenue for future research is the application of this method to Large Language Models (LLMs) and Vision-Language Models (VLMs). These models, with their vast parameter spaces, emphasize the need for efficient unlearning techniques. Although computational constraints limited our ability to explore this direction, scaling our approach to these larger models could help develop adaptable and privacy-preserving AI systems.

# Conclusion

This study addresses the challenge of machine unlearning in light of growing privacy regulations and the need for adaptable AI systems. We present a novel approach, PruneLoRA to LoRA to fine-tune sparse models. Our findings highlight the efficacy of LoRA, especially when combined with pruning, in achieving high unlearning performance with minimal computational cost and memory requirements while maintaining general accuracy on remaining classes. These results advances the research in exploring parameter efficient machine approximate unlearning techniques, thus laying the groundwork for applying these methods to complex models such as Large Language Models and Vision-Language Models.

# References

Biderman, D.; Portes, J.; Ortiz, J. J. G.; Paul, M.; Greengard, P.; Jennings, C.; King, D.; Havens, S.; Chiley, V.; Frankle, J.; Blakeney, C.; and Cunningham, J. P. 2024. LoRA Learns Less and Forgets Less. Transactions on Machine Learning Research. Featured Certification.

Hu, E. J.; yelong shen; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

Liu, J.; Ram, P.; Yao, Y.; Liu, G.; Liu, Y.; SHARMA, P.; Liu, S.; et al. 2024. Model sparsity can simplify machine unlearning. Advances in Neural Information Processing Systems, 36.

# Appendix

# Experiment Details

We trained ResNet-50 and Vision Transformer (ViT) on CIFAR10, using custom implementations. The ResNet-50 model was trained for 200 epochs, and the ViT model was trained for 90 epochs, both on a P100 GPU. They achieved a test accuracy of 95.56% and 83.77% respectively.

All further experiments were conducted on a T4 GPU. To allow for meaningful comparison between the various fine-tuning techniques employed, we consistently used the Adam optimizer with a learning rate of $10^{-3}$ , along with cross-entropy loss.

In the case of ResNet-50, we applied Structured L2 pruning with a sparsity level of 0.5 across all convolutional layers. Additionally, LoRA was applied to these layers to enable efficient fine-tuning. It is worth noting that future work could explore reducing the number of layers to which LoRA is applied, potentially leading to further computational gains without sacrificing model performance.

For the ViT model, Structured L2 pruning with 0.5 sparsity was applied to the last linear layer and the last attention layer. While initial experiments involved applying LoRA to multiple attention layers, we found that restricting LoRA to the last attention layer yielded the best results. This insight highlights the importance of targeted layer modification in enhancing the model's efficiency.

The exact architecture and implementation details for these experiments can be found in the public repository at https://github.com/vlgiitr/LoRA-Unlearn.

# Detailed metric settings

Details of MIA implementation: MIA is implemented using the prediction confidence-based attack method. There are mainly two phases during its computation: (1) training phase, and (2) testing phase. To train an MIA model, we first sample a balanced dataset from the remaining dataset (Dr) and the test dataset (different from the forgetting dataset Df) to train the MIA predictor. The learned MIA is then used for MU evaluation in its testing phase. To evaluate the performance of MU, MIA-Efficacy is obtained by applying the learned MIA predictor to the unlearned model on the forgetting dataset (Df). Our objective is to find out how many samples in Df can be correctly predicted as non-training samples by the MIA model.

$$
\text { MIA - Efficacy } = \frac {T N}{| \mathcal {D} _ {\mathrm{f}} |}
$$

where TN refers to the true negatives predicted by our MIA predictor, i.e., the number of the forgetting samples predicted as non-training examples, and $|Df|$ refers to the size of the forgetting dataset.

# Future Scope

Due to lack of computational resources and funding, we were only able to perform a limited number of experiments,

![](images/4da58a597efdcdf2f9f72b10b877f5a70f32e0b818561b7c7b5d2e62f6d4eedf.jpg)

<details>
<summary>bar</summary>

| Loss Range | Test Losses set Frequency | Forget Losses set Frequency |
|------------|---------------------------|-----------------------------|
| 0-5        | ~10^1                     | 0                           |
| 5-10       | ~10^1                     | 0                           |
| 10-15      | ~10^1                     | 0                           |
| 15-20      | ~10^1                     | ~10^1                       |
| 20-25      | ~10^1                     | ~10^1                       |
| 25-30      | ~10^1                     | ~10^1                       |
| 30-35      | ~10^1                     | ~10^1                       |
| 35-40      | ~10^1                     | ~10^1                       |
| 40-45      | ~10^1                     | ~10^1                       |
| 45-50      | ~10^1                     | ~10^1                       |
</details>

(a) Fine-tuned R

![](images/0a685826012c3f00e5cf3fa885d6869dec220bca60e51549e598086aa78918b4.jpg)

<details>
<summary>histogram</summary>

| Loss Range | Test Losses set Frequency | Forget Losses set Frequency |
|------------|---------------------------|-----------------------------|
| 0-1        | ~10^1                     | 0                           |
| 1-2        | ~10^0.5                   | 0                           |
| 2-3        | ~10^0.3                   | 0                           |
| 3-4        | ~10^0.2                   | 0                           |
| 4-5        | ~10^0.1                   | 0                           |
| 5-6        | ~10^0                    | 0                           |
| 6-7        | ~10^0                    | 0                           |
| 7-8        | ~10^0                    | 0                           |
| 8-9        | ~10^0                    | 0                           |
| 9-10       | ~10^0                    | ~10^0                       |
| 10-11      | ~10^0                    | ~10^0                       |
| 11-12      | ~10^0                    | ~10^0                       |
| 12-13      | ~10^0                    | ~10^0                       |
| 13-14      | ~10^0                    | ~10^0                       |
| 14-15      | ~10^0                    | ~10^0                       |
| 15-16      | ~10^0                    | ~10^0                       |
| 16-17      | ~10^0                    | ~10^0                       |
| 17-18      | ~10^0                    | ~10^0                       |
| 18-19      | ~10^0                    | ~10^0                       |
| 19-20      | ~10^0                    | ~10^0                       |
| 20-21      | ~10^0                    | ~10^0                       |
| 21-22      | ~10^0                    | ~10^0                       |
| 22-23      | ~10^0                    | ~10^0                       |
| 23-24      | ~10^0                    | ~10^0                       |
| 24-25      | ~10^0                    | ~10^0                       |
| 25+        | ~10^0                    | ~10^0                       |
</details>

(b) PruneFT R

![](images/659cc48512149e1d3f8e8ccd096ebdca5f70c4cad04674e42031f6b705a74940.jpg)  
(c) LoRA R

![](images/a8264a775bbb632c2b3d55e8b296eb2d772e1b3918fb2a701ba4105b4504c389.jpg)  
(d) PruneLoRA R

![](images/948d46a9f2d8e92e77cf94b6717cd8ead0eced60cc4c3a8be129537692ad4461.jpg)

<details>
<summary>bar</summary>

| Loss Range | Test Losses set Frequency | Forget Losses set Frequency |
|------------|----------------------------|-----------------------------|
| 0-1        | ~10^1                      | ~0                          |
| 1-2        | ~10^1                      | ~0                          |
| 2-3        | ~10^1                      | ~0                          |
| 3-4        | ~10^1                      | ~0                          |
| 4-5        | ~10^1                      | ~0                          |
| 5-6        | ~10^1                      | ~0                          |
| 6-7        | ~10^1                      | ~0                          |
| 7-8        | ~10^1                      | ~0                          |
| 8-9        | ~10^1                      | ~0                          |
| 9-10       | ~10^1                      | ~0                          |
| 10-11      | ~10^1                      | ~0                          |
| 11-12      | ~10^1                      | ~0                          |
| 12-13      | ~10^1                      | ~0                          |
| 13-14      | ~10^1                      | ~0                          |
| 14-15      | ~10^1                      | ~0                          |
| 15-16      | ~10^1                      | ~0                          |
| 16-17      | ~10^1                      | ~0                          |
| 17-18      | ~10^1                      | ~0                          |
| 18-19      | ~10^1                      | ~0                          |
| 19-20      | ~10^1                      | ~0                          |
| 20-21      | ~10^1                      | ~0                          |
| 21-22      | ~10^1                      | ~0                          |
| 22-23      | ~10^1                      | ~0                          |
| 23-24      | ~10^1                      | ~0                          |
| 24-25      | ~10^1                      | ~0                          |
</details>

(e) Fine-tuned V

![](images/2b76b86d4e3cf691629d8ad9bd0c5cf38c0793fbd8013672e96457cc338ae9f1.jpg)

<details>
<summary>histogram</summary>

| Loss Range | Test Losses set Frequency | Forget Losses set Frequency |
|------------|----------------------------|-----------------------------|
| 0-1        | ~10^3                      | ~0                          |
| 1-2        | ~10^2                      | ~0                          |
| 2-3        | ~10^1                      | ~0                          |
| 3-4        | ~10^0                      | ~0                          |
| 4-5        | ~10^-1                     | ~0                          |
| 5-6        | ~10^-2                     | ~0                          |
| 6-7        | ~10^-3                     | ~0                          |
| 7-8        | ~10^-4                     | ~0                          |
| 8-9        | ~10^-5                     | ~0                          |
| 9-10       | ~10^-6                     | ~0                          |
| 10-11      | ~10^-7                     | ~0                          |
| 11-12      | ~10^-8                     | ~0                          |
| 12-13      | ~10^-9                     | ~0                          |
| 13-14      | ~10^-10                    | ~0                          |
| 14-15      | ~10^-11                    | ~0                          |
| 15-16      | ~10^-12                    | ~0                          |
| 16-17      | ~10^-13                    | ~0                          |
| 17-18      | ~10^-14                    | ~0                          |
| 18-19      | ~10^-15                    | ~0                          |
| 19-20      | ~10^-16                    | ~0                          |
| 20-21      | ~10^-17                    | ~0                          |
| 21-22      | ~10^-18                    | ~0                          |
| 22-23      | ~10^-19                    | ~0                          |
| 23-24      | ~10^-20                    | ~0                          |
| 24-25      | ~10^-21                    | ~0                          |
</details>

(f) PruneFT V

![](images/66f8b184ab7a0c309f6b696a8154b077bdeb693892dd51d3f93d08e0dadf8f41.jpg)

<details>
<summary>bar</summary>

| Loss | Test Losses set | Forget Losses set |
|------|-----------------|-------------------|
| 0    | 10^2            | 10^2              |
| 5    | 10^1            | 10^1              |
| 10   | 10^1            | 10^1              |
| 15   | 10^1            | 10^1              |
| 20   | 10^1            | 10^1              |
| 25   | 10^1            | 10^1              |
| 30   | 10^1            | 10^1              |
</details>

(g) LoRA V

![](images/b052f1b405494a551a043b89f09d242979506f41d91c886813cf74a7100ea2a8.jpg)

<details>
<summary>bar</summary>

| Loss Range | Test Losses set Frequency | Forget Losses set Frequency |
|------------|----------------------------|-----------------------------|
| 0-1        | ~10^1                      | ~10^1                       |
| 1-2        | ~10^1                      | ~10^1                       |
| 2-3        | ~10^1                      | ~10^1                       |
| 3-4        | ~10^1                      | ~10^1                       |
| 4-5        | ~10^1                      | ~10^1                       |
| 5-6        | ~10^1                      | ~10^1                       |
| 6-7        | ~10^1                      | ~10^1                       |
| 7-8        | ~10^1                      | ~10^1                       |
| 8-9        | ~10^1                      | ~10^1                       |
| 9-10       | ~10^1                      | ~10^1                       |
| 10-11      | ~10^1                      | ~10^1                       |
| 11-12      | ~10^1                      | ~10^1                       |
| 12-13      | ~10^1                      | ~10^1                       |
| 13-14      | ~10^1                      | ~10^1                       |
| 14-15      | ~10^1                      | ~10^1                       |
| 15-16      | ~10^1                      | ~10^1                       |
| 16-17      | ~10^1                      | ~10^1                       |
| 17-18      | ~10^1                      | ~10^1                       |
| 18-19      | ~10^1                      | ~10^1                       |
| 19-20      | ~10^1                      | ~10^1                       |
| 20-21      | ~10^1                      | ~10^1                       |
| 21-22      | ~10^1                      | ~10^1                       |
| 22-23      | ~10^1                      | ~10^1                       |
| 23-24      | ~10^1                      | ~10^1                       |
| 24-25      | ~10^1                      | ~10^1                       |
</details>

(h) PruneLoRA V   
Figure 2: Visual Comparison of Cross-Entropy Losses of Test and Forget Set (R: ResNet-50 V: ViT) Forget Losses must be higher than Test losses for Unlearning Accuracy

without such constraints, another approach is to include layer-specific adaptation strategies where different layers or components of the model are subject to distinct unlearning approaches after studying optimal unlearning strategies for the respective layers. Models can also be studied under continual learning contexts, where models repeatedly learn and unlearn information over time.