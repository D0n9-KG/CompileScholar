Language Models are Few-Shot Learners 

<table><tr><td colspan="2">Tom B. Brown*</td><td>Benjamin Mann*</td><td>Nick Ryder*</td><td>Melanie Subbiah*</td></tr><tr><td>Jared Kaplan†</td><td>Prafulla Dhariwal</td><td>Arvind Neelakantan</td><td>Pranav Shyam</td><td>Girish Sastry</td></tr><tr><td>Amanda Askell</td><td>Sandhini Agarwal</td><td>Ariel Herbert-Voss</td><td>Gretchen Krueger</td><td>Tom Henighan</td></tr><tr><td>Rewon Child</td><td>Aditya Ramesh</td><td>Daniel M. Ziegler</td><td>Jeffrey Wu</td><td>Clemens Winter</td></tr><tr><td>Christopher Hesse</td><td>Mark Chen</td><td>Eric Sigler</td><td>Mateusz Litwin</td><td>Scott Gray</td></tr><tr><td colspan="2">Benjamin Chess</td><td>Jack Clark</td><td colspan="2">Christopher Berner</td></tr><tr><td>Sam McCandlish</td><td>Alec Radford</td><td>Ilya Sutskever</td><td colspan="2">Dario Amodei</td></tr></table>

OpenAI

# Abstract

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

# Contents

# 1 Introduction 3

# 2 Approach 6

2.1 Model and Architectures 8   
2.2 Training Dataset 8   
2.3 Training Process 9   
2.4 Evaluation 10

# 3 Results 10

3.1 Language Modeling, Cloze, and Completion Tasks 11   
3.2 Closed Book Question Answering 13   
3.3 Translation 14   
3.4 Winograd-Style Tasks 16  
3.5 Common Sense Reasoning 17   
3.6 Reading Comprehension 18   
3.7 SuperGLUE 18   
3.8 NLI 20   
3.9 Synthetic and Qualitative Tasks 21

# 4 Measuring and Preventing Memorization Of Benchmarks 29

# 5 Limitations 33

# 6 Broader Impacts 34

6.1 Misuse of Language Models 35   
6.2 Fairness, Bias, and Representation 36   
6.3 Energy Usage 39

# 7 Related Work 39

# 8 Conclusion 40

A Details of Common Crawl Filtering 43   
B Details of Model Training 43   
C Details of Test Set Contamination Studies 43   
D Total Compute Used to Train Language Models 46   
E Human Quality Assessment of Synthetic News Articles 46   
F Additional Samples from GPT-3 48   
G Details of Task Phrasing and Specifications 50   
H Results on All Tasks for All Model Sizes 63

# 1 Introduction

Recent years have featured a trend towards pre-trained language representations in NLP systems, applied in increasingly flexible and task-agnostic ways for downstream transfer. First, single-layer representations were learned using word vectors [MCCD13, PSM14] and fed to task-specific architectures, then RNNs with multiple layers of representations and contextual state were used to form stronger representations [DL15, MBXS17, PNZtY18] (though still applied to task-specific architectures), and more recently pre-trained recurrent or transformer language models $[VSP^{+}17]$ have been directly fine-tuned, entirely removing the need for task-specific architectures [RNSS18, DCLT18, HR18].

This last paradigm has led to substantial progress on many challenging NLP tasks such as reading comprehension, question answering, textual entailment, and many others, and has continued to advance based on new architectures and algorithms $[RSR^{+}19, LOG^{+}19, YDY^{+}19, LCG^{+}19]$ . However, a major limitation to this approach is that while the architecture is task-agnostic, there is still a need for task-specific datasets and task-specific fine-tuning: to achieve strong performance on a desired task typically requires fine-tuning on a dataset of thousands to hundreds of thousands of examples specific to that task. Removing this limitation would be desirable, for several reasons.

First, from a practical perspective, the need for a large dataset of labeled examples for every new task limits the applicability of language models. There exists a very wide range of possible useful language tasks, encompassing anything from correcting grammar, to generating examples of an abstract concept, to critiquing a short story. For many of these tasks it is difficult to collect a large supervised training dataset, especially when the process must be repeated for every new task.

Second, the potential to exploit spurious correlations in training data fundamentally grows with the expressiveness of the model and the narrowness of the training distribution. This can create problems for the pre-training plus fine-tuning paradigm, where models are designed to be large to absorb information during pre-training, but are then fine-tuned on very narrow task distributions. For instance $[HLW^{+}20]$ observe that larger models do not necessarily generalize better out-of-distribution. There is evidence that suggests that the generalization achieved under this paradigm can be poor because the model is overly specific to the training distribution and does not generalize well outside it $[YdC^{+}19, MPL19]$ . Thus, the performance of fine-tuned models on specific benchmarks, even when it is nominally at human-level, may exaggerate actual performance on the underlying task $[GSL^{+}18, NK19]$ .

Third, humans do not require large supervised datasets to learn most language tasks – a brief directive in natural language (e.g. “please tell me if this sentence describes something happy or something sad”) or at most a tiny number of demonstrations (e.g. “here are two examples of people acting brave; please give a third example of bravery”) is often

![](images/fe3506e51e3b0178b9af8bd9a30f301ea20bb6cdf70034b615b065724e5fbfaa.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Outer loop"] --> B["Learning via SGD during unsupervised pre-training"]
    B --> C["In-context learning"]
    C --> D["Sequence #1: sequence 5 + 8 = 13, sequence 7 + 2 = 9, ..., 9 + 8 = 17"]
    C --> E["Sequence #2: sequence 1: goat => goat, sakne => snake, brid => bird, fsih => fish, dcuk => duck, cmihp => chimp"]
    C --> F["Sequence #3: sequence 1: thanks => merci, hello => bonjour, mint => menthe, wall => mur, otter => loutre, bread => pain"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
```
</details>

Figure 1.1: Language model meta-learning. During unsupervised pre-training, a language model develops a broad set of skills and pattern recognition abilities. It then uses these abilities at inference time to rapidly adapt to or recognize the desired task. We use the term “in-context learning” to describe the inner loop of this process, which occurs within the forward-pass upon each sequence. The sequences in this diagram are not intended to be representative of the data a model would see during pre-training, but are intended to show that there are sometimes repeated sub-tasks embedded within a single sequence.

![](images/fd57a47f2f7cb04cee5fbb0881425bc6b5d78cb28a73b476323f84d76bbb08e8.jpg)

<details>
<summary>line</summary>

| Number of Examples in Context (K) | 175B Params | 13B Params | 1.3B Params |
| ---------------------------------- | ----------- | ---------- | ----------- |
| 0                                  | 8           | 0          | 0           |
| 10^0                               | 47          | 7          | 1           |
| 10^1                               | 62          | 20         | 3           |
| 10^2                               | 65          | 25         | 4           |
| 10^3                               | 66          | 26         | 4           |
| 10^4                               | 67          | 27         | 4           |
| 10^5                               | 68          | 28         | 4           |
| 10^6                               | 69          | 29         | 4           |
| 10^7                               | 70          | 30         | 4           |
| 10^8                               | 71          | 31         | 4           |
| 10^9                               | 72          | 32         | 4           |
| 10^10                              | 73          | 33         | 4           |
| 10^11                              | 74          | 34         | 4           |
| 10^12                              | 75          | 35         | 4           |
| 10^13                              | 76          | 36         | 4           |
| 10^14                              | 77          | 37         | 4           |
| 10^15                              | 78          | 38         | 4           |
| 10^16                              | 79          | 39         | 4           |
| 10^17                              | 80          | 40         | 4           |
| 10^18                              | 81          | 41         | 4           |
| 10^19                              | 82          | 42         | 4           |
| 10^20                              | 83          | 43         | 4           |
| 10^21                              | 84          | 44         | 4           |
| 10^22                              | 85          | 45         | 4           |
| 10^23                              | 86          | 46         | 4           |
| 10^24                              | 87          | 47         | 4           |
| 10^25                              | 88          | 48         | 4           |
| 10^26                              | 89          | 49         | 4           |
| 10^27                              | 90          | 50         | 4           |
| 10^28                              | 91          | 51         | 4           |
| 10^29                              | 92          | 52         | 4           |
| 10^30                              | 93          | 53         | 4           |
| Note: The data is extracted from the code and presented in the image as requested. The numbers provided above the lines are estimated based on the actual output values. There is only one data series in this case. The y-axis label 'Accuracy (%)' is not explicitly shown in the plot. The x-axis label 'Number of Examples in Context (K)' is also shown on the left. There is no label for the y-axis in the plot. The legend indicates three parameter sets: 'No Prompt', '13B Params', and '1.3B Params'.
</details>

Figure 1.2: Larger models make increasingly efficient use of in-context information. We show in-context learning performance on a simple task requiring the model to remove random symbols from a word, both with and without a natural language task description (see Sec. 3.9.2). The steeper “in-context learning curves” for large models demonstrate improved ability to learn a task from contextual information. We see qualitatively similar behavior across a wide range of tasks.

sufficient to enable a human to perform a new task to at least a reasonable degree of competence. Aside from pointing to a conceptual limitation in our current NLP techniques, this adaptability has practical advantages – it allows humans to seamlessly mix together or switch between many tasks and skills, for example performing addition during a lengthy dialogue. To be broadly useful, we would someday like our NLP systems to have this same fluidity and generality.

One potential route towards addressing these issues is meta-learning $^{1}$ – which in the context of language models means the model develops a broad set of skills and pattern recognition abilities at training time, and then uses those abilities at inference time to rapidly adapt to or recognize the desired task (illustrated in Figure 1.1). Recent work $[RWC^{+}19]$ attempts to do this via what we call “in-context learning”, using the text input of a pretrained language model as a form of task specification: the model is conditioned on a natural language instruction and/or a few demonstrations of the task and is then expected to complete further instances of the task simply by predicting what comes next.

While it has shown some initial promise, this approach still achieves results far inferior to fine-tuning – for example $[RWC^{+}19]$ achieves only 4% on Natural Questions, and even its 55 F1 CoQa result is now more than 35 points behind the state of the art. Meta-learning clearly requires substantial improvement in order to be viable as a practical method of solving language tasks.

Another recent trend in language modeling may offer a way forward. In recent years the capacity of transformer language models has increased substantially, from 100 million parameters [RNSS18], to 300 million parameters [DCLT18], to 1.5 billion parameters $\left[\mathrm{RWC}^{+}19\right]$ , to 8 billion parameters $\left[\mathrm{SPP}^{+}19\right]$ , 11 billion parameters $\left[\mathrm{RSR}^{+}19\right]$ , and finally 17 billion parameters [Tur20]. Each increase has brought improvements in text synthesis and/or downstream NLP tasks, and there is evidence suggesting that log loss, which correlates well with many downstream tasks, follows a smooth trend of improvement with scale $\left[\mathrm{KMH}^{+}20\right]$ . Since in-context learning involves absorbing many skills and tasks within the parameters of the model, it is plausible that in-context learning abilities might show similarly strong gains with scale.

![](images/6247902486dad12de541bac3c0976eea760176c82a696ea35f9800c73308f806.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Few Shot | One Shot | Zero Shot |
| --------------------------- | -------- | -------- | --------- |
| 0.1B                        | 25       | 24       | 23        |
| 0.4B                        | 28       | 27       | 26        |
| 0.8B                        | 30       | 29       | 28        |
| 1.3B                        | 33       | 32       | 30        |
| 2.6B                        | 36       | 35       | 32        |
| 6.7B                        | 39       | 37       | 34        |
| 13B                         | 43       | 41       | 36        |
| 175B                        | 57       | 51       | 43        |
</details>

Figure 1.3: Aggregate performance for all 42 accuracy-denominated benchmarks While zero-shot performance improves steadily with model size, few-shot performance increases more rapidly, demonstrating that larger models are more proficient at in-context learning. See Figure 3.8 for a more detailed analysis on SuperGLUE, a standard NLP benchmark suite.

In this paper, we test this hypothesis by training a 175 billion parameter autoregressive language model, which we call GPT-3, and measuring its in-context learning abilities. Specifically, we evaluate GPT-3 on over two dozen NLP datasets, as well as several novel tasks designed to test rapid adaptation to tasks unlikely to be directly contained in the training set. For each task, we evaluate GPT-3 under 3 conditions: (a) “few-shot learning”, or in-context learning where we allow as many demonstrations as will fit into the model’s context window (typically 10 to 100), (b) “one-shot learning”, where we allow only one demonstration, and (c) “zero-shot” learning, where no demonstrations are allowed and only an instruction in natural language is given to the model. GPT-3 could also in principle be evaluated in the traditional fine-tuning setting, but we leave this to future work.

Figure 1.2 illustrates the conditions we study, and shows few-shot learning of a simple task requiring the model to remove extraneous symbols from a word. Model performance improves with the addition of a natural language task description, and with the number of examples in the model's context, K. Few-shot learning also improves dramatically with model size. Though the results in this case are particularly striking, the general trends with both model size and number of examples in-context hold for most tasks we study. We emphasize that these “learning” curves involve no gradient updates or fine-tuning, just increasing numbers of demonstrations given as conditioning.

Broadly, on NLP tasks GPT-3 achieves promising results in the zero-shot and one-shot settings, and in the few-shot setting is sometimes competitive with or even occasionally surpasses state-of-the-art (despite state-of-the-art being held by fine-tuned models). For example, GPT-3 achieves 81.5 F1 on CoQA in the zero-shot setting, 84.0 F1 on CoQA in the one-shot setting, 85.0 F1 in the few-shot setting. Similarly, GPT-3 achieves 64.3% accuracy on TriviaQA in the zero-shot setting, 68.0% in the one-shot setting, and 71.2% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.

GPT-3 also displays one-shot and few-shot proficiency at tasks designed to test rapid adaption or on-the-fly reasoning, which include unscrambling words, performing arithmetic, and using novel words in a sentence after seeing them defined only once. We also show that in the few-shot setting, GPT-3 can generate synthetic news articles which human evaluators have difficulty distinguishing from human-generated articles.

At the same time, we also find some tasks on which few-shot performance struggles, even at the scale of GPT-3. This includes natural language inference tasks like the ANLI dataset, and some reading comprehension datasets like RACE or QuAC. By presenting a broad characterization of GPT-3's strengths and weaknesses, including these limitations, we hope to stimulate study of few-shot learning in language models and draw attention to where progress is most needed.

A heuristic sense of the overall results can be seen in Figure 1.3, which aggregates the various tasks (though it should not be seen as a rigorous or meaningful benchmark in itself).

We also undertake a systematic study of “data contamination” – a growing problem when training high capacity models on datasets such as Common Crawl, which can potentially include content from test datasets simply because such content often exists on the web. In this paper we develop systematic tools to measure data contamination and quantify its distorting effects. Although we find that data contamination has a minimal effect on GPT-3’s performance on most datasets, we do identify a few datasets where it could be inflating results, and we either do not report results on these datasets or we note them with an asterisk, depending on the severity.

In addition to all the above, we also train a series of smaller models (ranging from 125 million parameters to 13 billion parameters) in order to compare their performance to GPT-3 in the zero, one and few-shot settings. Broadly, for most tasks we find relatively smooth scaling with model capacity in all three settings; one notable pattern is that the gap between zero-, one-, and few-shot performance often grows with model capacity, perhaps suggesting that larger models are more proficient meta-learners.

Finally, given the broad spectrum of capabilities displayed by GPT-3, we discuss concerns about bias, fairness, and broader societal impacts, and attempt a preliminary analysis of GPT-3's characteristics in this regard.

The remainder of this paper is organized as follows. In Section 2, we describe our approach and methods for training GPT-3 and evaluating it. Section 3 presents results on the full range of tasks in the zero-, one- and few-shot settings. Section 4 addresses questions of data contamination (train-test overlap). Section 5 discusses limitations of GPT-3. Section 6 discusses broader impacts. Section 7 reviews related work and Section 8 concludes.

# 2 Approach

Our basic pre-training approach, including model, data, and training, is similar to the process described in $[RWC^{+}19]$ , with relatively straightforward scaling up of the model size, dataset size and diversity, and length of training. Our use of in-context learning is also similar to $[RWC^{+}19]$ , but in this work we systematically explore different settings for learning within the context. Therefore, we start this section by explicitly defining and contrasting the different settings that we will be evaluating GPT-3 on or could in principle evaluate GPT-3 on. These settings can be seen as lying on a spectrum of how much task-specific data they tend to rely on. Specifically, we can identify at least four points on this spectrum (see Figure 2.1 for an illustration):

\- Fine-Tuning (FT) has been the most common approach in recent years, and involves updating the weights of a pre-trained model by training on a supervised dataset specific to the desired task. Typically thousands to hundreds of thousands of labeled examples are used. The main advantage of fine-tuning is strong performance on many benchmarks. The main disadvantages are the need for a new large dataset for every task, the potential for poor generalization out-of-distribution [MPL19], and the potential to exploit spurious features of the training data $\left[\mathrm{GSL}^{+}18,\mathrm{NK}19\right]$ , potentially resulting in an unfair comparison with human performance. In this work we do not fine-tune GPT-3 because our focus is on task-agnostic performance, but GPT-3 can be fine-tuned in principle and this is a promising direction for future work.

\- Few-Shot (FS) is the term we will use in this work to refer to the setting where the model is given a few demonstrations of the task at inference time as conditioning $[RWC^{+}19]$ , but no weight updates are allowed. As shown in Figure 2.1, for a typical dataset an example has a context and a desired completion (for example an English sentence and the French translation), and few-shot works by giving K examples of context and completion, and then one final example of context, with the model expected to provide the completion. We typically set K in the range of 10 to 100 as this is how many examples can fit in the model's context window $(n_{\mathrm{ctx}} = 2048)$ . The main advantages of few-shot are a major reduction in the need for task-specific data and reduced potential to learn an overly narrow distribution from a large but narrow fine-tuning dataset. The main disadvantage is that results from this method have so far been much worse than state-of-the-art fine-tuned models. Also, a small amount of task specific data is still required. As indicated by the name, few-shot learning as described here for language models is related to few-shot learning as used in other contexts in ML $[HYC01, VBL^{+}16]$ – both involve learning based on a broad distribution of tasks (in this case implicit in the pre-training data) and then rapidly adapting to a new task.

\- One-Shot (1S) is the same as few-shot except that only one demonstration is allowed, in addition to a natural language description of the task, as shown in Figure 1. The reason to distinguish one-shot from few-shot and zero-shot (below) is that it most closely matches the way in which some tasks are communicated to humans. For example, when asking humans to generate a dataset on a human worker service (for example Mechanical Turk), it is common to give one demonstration of the task. By contrast it is sometimes difficult to communicate the content or format of a task if no examples are given.

# The three settings we explore for in-context learning

# Zero-shot

The model predicts the answer given only a natural language description of the task. No gradient updates are performed.

![](images/62fa9594717d555e378010c82c5ad7aa7a350e7111fe1e883079ba19d646b9ed.jpg)

<details>
<summary>text_image</summary>

Translate English to French:
    ← task description
cheese => _________    ← prompt
</details>

# One-shot

In addition to the task description, the model sees a single example of the task. No gradient updates are performed.

![](images/8dbea39bff607c36348b7e8dfa1ce24b2e954175ead7ed4528e29a300f7a0bcd.jpg)

<details>
<summary>text_image</summary>

Translate English to French:
sea otter => loutre de mer
cheese =>
task description
example
prompt
</details>

# Few-shot

In addition to the task description, the model sees a few examples of the task. No gradient updates are performed.

![](images/6df8fba542c5036a9096ff0b4357091ed5f20121574b36fc0c6dfa8c637003cf.jpg)

<details>
<summary>text_image</summary>

Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese => _________
</details>

# Traditional fine-tuning (not used for GPT-3)

# Fine-tuning

The model is trained via repeated gradient updates using a large corpus of example tasks.

![](images/cf9cb5c4fcf5b9a1d820df61c0b449613ba352b5441c3d0d04b7bb5da9f4535c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["sea otter => loutre de mer"] --> B["gradient update"]
    B --> C["peppermint => menthe poivrée"]
    C --> D["gradient update"]
    D --> E["..."]
    E --> F["plush giraffe => girafe peluche"]
    F --> G["gradient update"]
    G --> H["cheese => _________ prompt"]
```
</details>

Figure 2.1: Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning. The panels above show four methods for performing a task with a language model – fine-tuning is the traditional method, whereas zero-, one-, and few-shot, which we study in this work, require the model to perform the task with only forward passes at test time. We typically present the model with a few dozen examples in the few shot setting. Exact phrasings for all task descriptions, examples and prompts can be found in Appendix G.

\- Zero-Shot (0S) is the same as one-shot except that no demonstrations are allowed, and the model is only given a natural language instruction describing the task. This method provides maximum convenience, potential for robustness, and avoidance of spurious correlations (unless they occur very broadly across the large corpus of pre-training data), but is also the most challenging setting. In some cases it may even be difficult for humans to understand the format of the task without prior examples, so this setting is in some cases “unfairly hard”. For example, if someone is asked to “make a table of world records for the 200m dash”, this request can be ambiguous, as it may not be clear exactly what format the table should have or what should be included (and even with careful clarification, understanding precisely what is desired can be difficult). Nevertheless, for at least some settings zero-shot is closest to how humans perform tasks – for example, in the translation example in Figure 2.1, a human would likely know what to do from just the text instruction.

Figure 2.1 shows the four methods using the example of translating English to French. In this paper we focus on zero-shot, one-shot and few-shot, with the aim of comparing them not as competing alternatives, but as different problem settings which offer a varying trade-off between performance on specific benchmarks and sample efficiency. We especially highlight the few-shot results as many of them are only slightly behind state-of-the-art fine-tuned models. Ultimately, however, one-shot, or even sometimes zero-shot, seem like the fairest comparisons to human performance, and are important targets for future work.

Sections 2.1-2.3 below give details on our models, training data, and training process respectively. Section 2.4 discusses the details of how we do few-shot, one-shot, and zero-shot evaluations.

<table><tr><td>Model Name</td><td> $n_{params}$ </td><td> $n_{layers}$ </td><td> $d_{model}$ </td><td> $n_{heads}$ </td><td> $d_{head}$ </td><td>Batch Size</td><td>Learning Rate</td></tr><tr><td>GPT-3 Small</td><td>125M</td><td>12</td><td>768</td><td>12</td><td>64</td><td>0.5M</td><td> $6.0 \times 10^{-4}$ </td></tr><tr><td>GPT-3 Medium</td><td>350M</td><td>24</td><td>1024</td><td>16</td><td>64</td><td>0.5M</td><td> $3.0 \times 10^{-4}$ </td></tr><tr><td>GPT-3 Large</td><td>760M</td><td>24</td><td>1536</td><td>16</td><td>96</td><td>0.5M</td><td> $2.5 \times 10^{-4}$ </td></tr><tr><td>GPT-3 XL</td><td>1.3B</td><td>24</td><td>2048</td><td>24</td><td>128</td><td>1M</td><td> $2.0 \times 10^{-4}$ </td></tr><tr><td>GPT-3 2.7B</td><td>2.7B</td><td>32</td><td>2560</td><td>32</td><td>80</td><td>1M</td><td> $1.6 \times 10^{-4}$ </td></tr><tr><td>GPT-3 6.7B</td><td>6.7B</td><td>32</td><td>4096</td><td>32</td><td>128</td><td>2M</td><td> $1.2 \times 10^{-4}$ </td></tr><tr><td>GPT-3 13B</td><td>13.0B</td><td>40</td><td>5140</td><td>40</td><td>128</td><td>2M</td><td> $1.0 \times 10^{-4}$ </td></tr><tr><td>GPT-3 175B or “GPT-3”</td><td>175.0B</td><td>96</td><td>12288</td><td>96</td><td>128</td><td>3.2M</td><td> $0.6 \times 10^{-4}$ </td></tr></table>

Table 2.1: Sizes, architectures, and learning hyper-parameters (batch size in tokens and learning rate) of the models which we trained. All models were trained for a total of 300 billion tokens.

# 2.1 Model and Architectures

We use the same model and architecture as GPT-2 $[RWC^{+}19]$ , including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer $[CGRS19]$ . To study the dependence of ML performance on model size, we train 8 different sizes of model, ranging over three orders of magnitude from 125 million parameters to 175 billion parameters, with the last being the model we call GPT-3. Previous work $[KMH^{+}20]$ suggests that with enough training data, scaling of validation loss should be approximately a smooth power law as a function of size; training models of many different sizes allows us to test this hypothesis both for validation loss and for downstream language tasks.

Table 2.1 shows the sizes and architectures of our 8 models. Here $n_{\mathrm{params}}$ is the total number of trainable parameters, $n_{\mathrm{layers}}$ is the total number of layers, $d_{\mathrm{model}}$ is the number of units in each bottleneck layer (we always have the feedforward layer four times the size of the bottleneck layer, $d_{\mathrm{ff}} = 4 * d_{\mathrm{model}}$ ), and $d_{\mathrm{head}}$ is the dimension of each attention head. All models use a context window of $n_{\mathrm{ctx}} = 2048$ tokens. We partition the model across GPUs along both the depth and width dimension in order to minimize data-transfer between nodes. The precise architectural parameters for each model are chosen based on computational efficiency and load-balancing in the layout of models across GPU's. Previous work [KMH+20] suggests that validation loss is not strongly sensitive to these parameters within a reasonably broad range.

# 2.2 Training Dataset

Datasets for language models have rapidly expanded, culminating in the Common Crawl dataset $^{2}$ [RSR $^{+}$ 19] constituting nearly a trillion words. This size of dataset is sufficient to train our largest models without ever updating on the same sequence twice. However, we have found that unfiltered or lightly filtered versions of Common Crawl tend to have lower quality than more curated datasets. Therefore, we took 3 steps to improve the average quality of our datasets: (1) we downloaded and filtered a version of CommonCrawl based on similarity to a range of high-quality reference corpora, (2) we performed fuzzy deduplication at the document level, within and across datasets, to prevent redundancy and preserve the integrity of our held-out validation set as an accurate measure of overfitting, and (3) we also added known high-quality reference corpora to the training mix to augment CommonCrawl and increase its diversity.

Details of the first two points (processing of Common Crawl) are described in Appendix A. For the third, we added several curated high-quality datasets, including an expanded version of the WebText dataset $[RWC^{+}19]$ , collected by scraping links over a longer period of time, and first described in $[KMH^{+}20]$ , two internet-based books corpora (Books1 and Books2) and English-language Wikipedia.

Table 2.2 shows the final mixture of datasets that we used in training. The CommonCrawl data was downloaded from 41 shards of monthly CommonCrawl covering 2016 to 2019, constituting 45TB of compressed plaintext before filtering and 570GB after filtering, roughly equivalent to 400 billion byte-pair-encoded tokens. Note that during training, datasets are not sampled in proportion to their size, but rather datasets we view as higher-quality are sampled more frequently, such that CommonCrawl and Books2 datasets are sampled less than once during training, but the other datasets are sampled 2-3 times. This essentially accepts a small amount of overfitting in exchange for higher quality training data.

Total Compute Used During Training   
![](images/4d4fbe34ee1bc8f9fbe597f904df244f4d208c4e1f254b885874904acedccc82.jpg)

<details>
<summary>bar</summary>

| Model | Training Petaflop/s-days |
| :--- | :--- |
| BERT-Base | 2.0 |
| BERT-Large | 6.0 |
| RoBERTa-Base | 18.0 |
| RoBERTa-Large | 45.0 |
| T5-Small | 2.5 |
| T5-Base | 7.5 |
| T5-Large | 28.0 |
| T5-3B | 100.0 |
| T5-11B | 350.0 |
| GPT-3 Small | 3.0 |
| GPT-3 Medium | 7.0 |
| GPT-3 Large | 15.0 |
| GPT-3 XL | 25.0 |
| GPT-3 2.7B | 50.0 |
| GPT-3 6.7B | 120.0 |
| GPT-3 13B | 250.0 |
| GPT-3 175B | 4000.0 |
</details>

Figure 2.2: Total compute used during training. Based on the analysis in Scaling Laws For Neural Language Models $[KMH^{+}20]$ we train much larger models on many fewer tokens than is typical. As a consequence, although GPT-3 3B is almost 10x larger than RoBERTa-Large (355M params), both models took roughly 50 petaflop/s-days of compute during pre-training. Methodology for these calculations can be found in Appendix D.

<table><tr><td>Dataset</td><td>Quantity (tokens)</td><td>Weight in training mix</td><td>Epochs elapsed when training for 300B tokens</td></tr><tr><td>Common Crawl (filtered)</td><td>410 billion</td><td>60%</td><td>0.44</td></tr><tr><td>WebText2</td><td>19 billion</td><td>22%</td><td>2.9</td></tr><tr><td>Books1</td><td>12 billion</td><td>8%</td><td>1.9</td></tr><tr><td>Books2</td><td>55 billion</td><td>8%</td><td>0.43</td></tr><tr><td>Wikipedia</td><td>3 billion</td><td>3%</td><td>3.4</td></tr></table>

Table 2.2: Datasets used to train GPT-3. “Weight in training mix” refers to the fraction of examples during training that are drawn from a given dataset, which we intentionally do not make proportional to the size of the dataset. As a result, when we train for 300 billion tokens, some datasets are seen up to 3.4 times during training while other datasets are seen less than once.

A major methodological concern with language models pretrained on a broad swath of internet data, particularly large models with the capacity to memorize vast amounts of content, is potential contamination of downstream tasks by having their test or development sets inadvertently seen during pre-training. To reduce such contamination, we searched for and attempted to remove any overlaps with the development and test sets of all benchmarks studied in this paper. Unfortunately, a bug in the filtering caused us to ignore some overlaps, and due to the cost of training it was not feasible to retrain the model. In Section 4 we characterize the impact of the remaining overlaps, and in future work we will more aggressively remove data contamination.

# 2.3 Training Process

As found in $[KMH^{+}20, MKAT18]$ , larger models can typically use a larger batch size, but require a smaller learning rate. We measure the gradient noise scale during training and use it to guide our choice of batch size $[MKAT18]$ . Table 2.1 shows the parameter settings we used. To train the larger models without running out of memory, we use a mixture of model parallelism within each matrix multiply and model parallelism across the layers of the network. All models were trained on V100 GPU's on part of a high-bandwidth cluster provided by Microsoft. Details of the training process and hyperparameter settings are described in Appendix B.

# 2.4 Evaluation

For few-shot learning, we evaluate each example in the evaluation set by randomly drawing K examples from that task's training set as conditioning, delimited by 1 or 2 newlines depending on the task. For LAMBADA and Storycloze there is no supervised training set available so we draw conditioning examples from the development set and evaluate on the test set. For Winograd (the original, not SuperGLUE version) there is only one dataset, so we draw conditioning examples directly from it.

$K$ can be any value from 0 to the maximum amount allowed by the model's context window, which is $n_{\mathrm{ctx}} = 2048$ for all models and typically fits 10 to 100 examples. Larger values of $K$ are usually but not always better, so when a separate development and test set are available, we experiment with a few values of $K$ on the development set and then run the best value on the test set. For some tasks (see Appendix G) we also use a natural language prompt in addition to (or for $K = 0$ , instead of) demonstrations.

On tasks that involve choosing one correct completion from several options (multiple choice), we provide K examples of context plus correct completion, followed by one example of context only, and compare the LM likelihood of each completion. For most tasks we compare the per-token likelihood (to normalize for length), however on a small number of datasets (ARC, OpenBookQA, and RACE) we gain additional benefit as measured on the development set by normalizing by the unconditional probability of each completion, by computing $\frac{P(\text{completion}|context)}{P(\text{completion}|answer\_context)}$ , where answer\_context is the string "Answer: " or "A: " and is used to prompt that the completion should be an answer but is otherwise generic.

On tasks that involve binary classification, we give the options more semantically meaningful names (e.g. “True” or “False” rather than 0 or 1) and then treat the task like multiple choice; we also sometimes frame the task similar to what is done by $[RSR^{+}19]$ (see Appendix G) for details.

On tasks with free-form completion, we use beam search with the same parameters as $[RSR^{+}19]$ : a beam width of 4 and a length penalty of $\alpha = 0.6$ . We score the model using F1 similarity score, BLEU, or exact match, depending on what is standard for the dataset at hand.

Final results are reported on the test set when publicly available, for each model size and learning setting (zero-, one-, and few-shot). When the test set is private, our model is often too large to fit on the test server, so we report results on the development set. We do submit to the test server on a small number of datasets (SuperGLUE, TriviaQA, PiQa) where we were able to make submission work, and we submit only the 200B few-shot results, and report development set results for everything else.

# 3 Results

In Figure 3.1 we display training curves for the 8 models described in Section 2. For this graph we also include 6 additional extra-small models with as few as 100,000 parameters. As observed in $[KMH^{+}20]$ , language modeling performance follows a power-law when making efficient use of training compute. After extending this trend by two more orders of magnitude, we observe only a slight (if any) departure from the power-law. One might worry that these improvements in cross-entropy loss come only from modeling spurious details of our training corpus. However, we will see in the following sections that improvements in cross-entropy loss lead to consistent performance gains across a broad spectrum of natural language tasks.

Below, we evaluate the 8 models described in Section 2 (the 175 billion parameter parameter GPT-3 and 7 smaller models) on a wide range of datasets. We group the datasets into 9 categories representing roughly similar tasks.

In Section 3.1 we evaluate on traditional language modeling tasks and tasks that are similar to language modeling, such as Cloze tasks and sentence/paragraph completion tasks. In Section 3.2 we evaluate on “closed book” question answering tasks: tasks which require using the information stored in the model’s parameters to answer general knowledge questions. In Section 3.3 we evaluate the model’s ability to translate between languages (especially one-shot and few-shot). In Section 3.4 we evaluate the model’s performance on Winograd Schema-like tasks. In Section 3.5 we evaluate on datasets that involve commonsense reasoning or question answering. In Section 3.6 we evaluate on reading comprehension tasks, in Section 3.7 we evaluate on the SuperGLUE benchmark suite, and in 3.8 we briefly explore NLI. Finally, in Section 3.9, we invent some additional tasks designed especially to probe in-context learning abilities – these tasks focus on on-the-fly reasoning, adaptation skills, or open-ended text synthesis. We evaluate all tasks in the few-shot, one-shot, and zero-shot settings.

![](images/fcd413a50f749b69b7ec651eef840c3bd5922f49e9956d43858bce13f7140a43.jpg)

<details>
<summary>line</summary>

| Compute (PetaFLOP/s-days) | Validation Loss | Parameters |
| ------------------------- | --------------- | ---------- |
| 1e-6                      | 6.0             | 1e+11      |
| 1e-4                      | 4.5             | 1e+10      |
| 1e-2                      | 3.5             | 1e+9       |
| 1e0                       | 2.5             | 1e+8       |
| 1e2                       | 2.0             | 1e+7       |
| 1e4                       | 1.5             | 1e+6       |
</details>

Figure 3.1: Smooth scaling of performance with compute. Performance (measured in terms of cross-entropy validation loss) follows a power-law trend with the amount of compute used for training. The power-law behavior observed in $[KMH^{+}20]$ continues for an additional two orders of magnitude with only small deviations from the predicted curve. For this figure, we exclude embedding parameters from compute and parameter counts. 

<table><tr><td>Setting</td><td>PTB</td></tr><tr><td>SOTA (Zero-Shot)</td><td> $35.8^a$ </td></tr><tr><td>GPT-3 Zero-Shot</td><td>20.5</td></tr></table>

Table 3.1: Zero-shot results on PTB language modeling dataset. Many other common language modeling datasets are omitted because they are derived from Wikipedia or other sources which are included in GPT-3's training data. $^{a}$ [RWC $^{+}$ 19]

# 3.1 Language Modeling, Cloze, and Completion Tasks

In this section we test GPT-3's performance on the traditional task of language modeling, as well as related tasks that involve predicting a single word of interest, completing a sentence or paragraph, or choosing between possible completions of a piece of text.

# 3.1.1 Language Modeling

We calculate zero-shot perplexity on the Penn Tree Bank (PTB) $[MKM^{+}94]$ dataset measured in $[RWC^{+}19]$ . We omit the 4 Wikipedia-related tasks in that work because they are entirely contained in our training data, and we also omit the one-billion word benchmark due to a high fraction of the dataset being contained in our training set. PTB escapes these issues due to predating the modern internet. Our largest model sets a new SOTA on PTB by a substantial margin of 15 points, achieving a perplexity of 20.50. Note that since PTB is a traditional language modeling dataset it does not have a clear separation of examples to define one-shot or few-shot evaluation around, so we measure only zero-shot.

# 3.1.2 LAMBADA

The LAMBADA dataset $[PKL^{+}16]$ tests the modeling of long-range dependencies in text – the model is asked to predict the last word of sentences which require reading a paragraph of context. It has recently been suggested that the continued scaling of language models is yielding diminishing returns on this difficult benchmark. $[BHT^{+}20]$ reflect on the small 1.5% improvement achieved by a doubling of model size between two recent state of the art results ( $[SPP^{+}19]$

<table><tr><td>Setting</td><td>LAMBADA (acc)</td><td>LAMBADA (ppl)</td><td>StoryCloze (acc)</td><td>HellaSwag (acc)</td></tr><tr><td>SOTA</td><td> $68.0^a$ </td><td> $8.63^b$ </td><td> $\mathbf{91.8}^c$ </td><td> $\mathbf{85.6}^d$ </td></tr><tr><td>GPT-3 Zero-Shot</td><td>76.2</td><td>3.00</td><td>83.2</td><td>78.9</td></tr><tr><td>GPT-3 One-Shot</td><td>72.5</td><td>3.35</td><td>84.7</td><td>78.1</td></tr><tr><td>GPT-3 Few-Shot</td><td>86.4</td><td>1.92</td><td>87.7</td><td>79.3</td></tr></table>

Table 3.2: Performance on cloze and completion tasks. GPT-3 significantly improves SOTA on LAMBADA while achieving respectable performance on two difficult completion prediction datasets. ${}^{a}$ [Tur20] ${}^{b}$ [RWC $^{+}$ 19] ${}^{c}$ [LDL19] ${}^{d}$ [LCH $^{+}$ 20]

![](images/34796a2793b9da10bd71b61579a677abd9871f7576df2ae3d2c5fcd1afb63a18.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=15) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 43        | 22       | 22              |
| 0.4B                        | 54        | 47       | 40              |
| 0.8B                        | 60        | 52       | 63              |
| 1.3B                        | 63        | 58       | 57              |
| 2.6B                        | 67        | 61       | 78              |
| 6.7B                        | 70        | 65       | 79              |
| 13B                         | 72        | 69       | 81              |
| 175B                        | 76        | 72       | 86              |
</details>

Figure 3.2: On LAMBADA, the few-shot capability of language models results in a strong boost to accuracy. GPT-3 2.7B outperforms the SOTA 17B parameter Turing-NLG [Tur20] in this setting, and GPT-3 175B advances the state of the art by 18%. Note zero-shot uses a different format from one-shot and few-shot as described in the text.

and [Tur20]) and argue that “continuing to expand hardware and data sizes by orders of magnitude is not the path forward”. We find that path is still promising and in a zero-shot setting GPT-3 achieves 76% on LAMBADA, a gain of 8% over the previous state of the art.

LAMBADA is also a demonstration of the flexibility of few-shot learning as it provides a way to address a problem that classically occurs with this dataset. Although the completion in LAMBADA is always the last word in a sentence, a standard language model has no way of knowing this detail. It thus assigns probability not only to the correct ending but also to other valid continuations of the paragraph. This problem has been partially addressed in the past with stop-word filters $[RWC^{+}19]$ (which ban “continuation” words). The few-shot setting instead allows us to “frame” the task as a cloze-test and allows the language model to infer from examples that a completion of exactly one word is desired. We use the following fill-in-the-blank format:

Alice was friends with Bob. Alice went to visit her friend \_\_\_\_. → Bob

George bought some baseball equipment, a ball, a glove, and a \_\_\_\_. →

When presented with examples formatted this way, GPT-3 achieves 86.4% accuracy in the few-shot setting, an increase of over 18% from the previous state-of-the-art. We observe that few-shot performance improves strongly with model size. While this setting decreases the performance of the smallest model by almost 20%, for GPT-3 it improves accuracy by 10%. Finally, the fill-in-blank method is not effective one-shot, where it always performs worse than the zero-shot setting. Perhaps this is because all models still require several examples to recognize the pattern.

<table><tr><td>Setting</td><td>NaturalQS</td><td>WebQS</td><td>TriviaQA</td></tr><tr><td>RAG (Fine-tuned, Open-Domain) [LPP+20]</td><td>44.5</td><td>45.5</td><td>68.0</td></tr><tr><td>T5-11B+SSM (Fine-tuned, Closed-Book) [RRS20]</td><td>36.6</td><td>44.7</td><td>60.5</td></tr><tr><td>T5-11B (Fine-tuned, Closed-Book)</td><td>34.5</td><td>37.4</td><td>50.1</td></tr><tr><td>GPT-3 Zero-Shot</td><td>14.6</td><td>14.4</td><td>64.3</td></tr><tr><td>GPT-3 One-Shot</td><td>23.0</td><td>25.3</td><td>68.0</td></tr><tr><td>GPT-3 Few-Shot</td><td>29.9</td><td>41.5</td><td>71.2</td></tr></table>

Table 3.3: Results on three Open-Domain QA tasks. GPT-3 is shown in the few-, one-, and zero-shot settings, as compared to prior SOTA results for closed book and open domain settings. TriviaQA few-shot result is evaluated on the wiki split test server.

One note of caution is that an analysis of test set contamination identified that a significant minority of the LAMBADA dataset appears to be present in our training data – however analysis performed in Section 4 suggests negligible impact on performance.

# 3.1.3 HellaSwag

The HellaSwag dataset $[ZHB^{+}19]$ involves picking the best ending to a story or set of instructions. The examples were adversarially mined to be difficult for language models while remaining easy for humans (who achieve 95.6% accuracy). GPT-3 achieves 78.1% accuracy in the one-shot setting and 79.3% accuracy in the few-shot setting, outperforming the 75.4% accuracy of a fine-tuned 1.5B parameter language model $[ZHR^{+}19]$ but still a fair amount lower than the overall SOTA of 85.6% achieved by the fine-tuned multi-task model ALUM.

# 3.1.4 StoryCloze

We next evaluate GPT-3 on the StoryCloze 2016 dataset $[MCH^{+}16]$ , which involves selecting the correct ending sentence for five-sentence long stories. Here GPT-3 achieves 83.2% in the zero-shot setting and 87.7% in the few-shot setting (with K = 70). This is still 4.1% lower than the fine-tuned SOTA using a BERT based model $[LDL19]$ but improves over previous zero-shot results by roughly 10%.

# 3.2 Closed Book Question Answering

In this section we measure GPT-3's ability to answer questions about broad factual knowledge. Due to the immense amount of possible queries, this task has normally been approached by using an information retrieval system to find relevant text in combination with a model which learns to generate an answer given the question and the retrieved text. Since this setting allows a system to search for and condition on text which potentially contains the answer it is denoted “open-book”. [RRS20] recently demonstrated that a large language model can perform surprisingly well directly answering the questions without conditioning on auxiliary information. They denote this more restrictive evaluation setting as “closed-book”. Their work suggests that even higher-capacity models could perform even better and we test this hypothesis with GPT-3. We evaluate GPT-3 on the 3 datasets in [RRS20]: Natural Questions [KPR+19], WebQuestions [BCFL13], and TriviaQA [JCWZ17], using the same splits. Note that in addition to all results being in the closed-book setting, our use of few-shot, one-shot, and zero-shot evaluations represent an even stricter setting than previous closed-book QA work: in addition to external content not being allowed, fine-tuning on the Q&A dataset itself is also not permitted.

The results for GPT-3 are shown in Table 3.3. On TriviaQA, we achieve $64.3\%$ in the zero-shot setting, $68.0\%$ in the one-shot setting, and $71.2\%$ in the few-shot setting. The zero-shot result already outperforms the fine-tuned T5-11B by $14.2\%$ , and also outperforms a version with Q&A tailored span prediction during pre-training by $3.8\%$ . The one-shot result improves by $3.7\%$ and matches the SOTA for an open-domain QA system which not only fine-tunes but also makes use of a learned retrieval mechanism over a 15.3B parameter dense vector index of 21M documents $\left[\mathrm{LPP}^{+}20\right]$ . GPT-3's few-shot result further improves performance another $3.2\%$ beyond this.

On WebQuestions (WebQs), GPT-3 achieves 14.4% in the zero-shot setting, 25.3% in the one-shot setting, and 41.5% in the few-shot setting. This compares to 37.4% for fine-tuned T5-11B, and 44.7% for fine-tuned T5-11B+SSM, which uses a Q&A-specific pre-training procedure. GPT-3 in the few-shot setting approaches the performance of state-of-the-art fine-tuned models. Notably, compared to TriviaQA, WebQS shows a much larger gain from zero-shot to few-shot (and indeed its zero-shot and one-shot performance are poor), perhaps suggesting that the WebQs questions

![](images/45f2e10cf14024fef11f544f909777a2271d2746ce2fbc58197f90fe98a94966.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 5         | 5        | 8               |
| 0.4B                        | 8         | 13       | 17              |
| 0.8B                        | 14        | 20       | 27              |
| 1.3B                        | 20        | 27       | 32              |
| 2.6B                        | 31        | 36       | 43              |
| 6.7B                        | 39        | 45       | 52              |
| 13B                         | 42        | 52       | 58              |
| 175B                        | 64        | 68       | 71              |
</details>

Figure 3.3: On TriviaQA GPT3's performance grows smoothly with model size, suggesting that language models continue to absorb knowledge as their capacity increases. One-shot and few-shot performance make significant gains over zero-shot behavior, matching and exceeding the performance of the SOTA fine-tuned open-domain model, RAG [LPP+20]

and/or the style of their answers are out-of-distribution for GPT-3. Nevertheless, GPT-3 appears able to adapt to this distribution, recovering strong performance in the few-shot setting.

On Natural Questions (NQs) GPT-3 achieves 14.6% in the zero-shot setting, 23.0% in the one-shot setting, and 29.9% in the few-shot setting, compared to 36.6% for fine-tuned T5 11B+SSM. Similar to WebQS, the large gain from zero-shot to few-shot may suggest a distribution shift, and may also explain the less competitive performance compared to TriviaQA and WebQS. In particular, the questions in NQs tend towards very fine-grained knowledge on Wikipedia specifically which could be testing the limits of GPT-3's capacity and broad pretraining distribution.

Overall, on one of the three datasets GPT-3's one-shot matches the open-domain fine-tuning SOTA. On the other two datasets it approaches the performance of the closed-book SOTA despite not using fine-tuning. On all 3 datasets, we find that performance scales very smoothly with model size (Figure 3.3 and Appendix H Figure H.7), possibly reflecting the idea that model capacity translates directly to more ‘knowledge’ absorbed in the parameters of the model.

# 3.3 Translation

For GPT-2 a filter was used on a multilingual collection of documents to produce an English only dataset due to capacity concerns. Even with this filtering GPT-2 showed some evidence of multilingual capability and performed non-trivially when translating between French and English despite only training on 10 megabytes of remaining French text. Since we increase the capacity by over two orders of magnitude from GPT-2 to GPT-3, we also expand the scope of the training dataset to include more representation of other languages, though this remains an area for further improvement. As discussed in 2.2 the majority of our data is derived from raw Common Crawl with only quality-based filtering. Although GPT-3's training data is still primarily English (93% by word count), it also includes 7% of text in other languages. These languages are documented in the supplemental material. In order to better understand translation capability, we also expand our analysis to include two additional commonly studied languages, German and Romanian.

Existing unsupervised machine translation approaches often combine pretraining on a pair of monolingual datasets with back-translation [SHB15] to bridge the two languages in a controlled way. By contrast, GPT-3 learns from a blend of training data that mixes many languages together in a natural way, combining them on a word, sentence, and document level. GPT-3 also uses a single training objective which is not customized or designed for any task in particular. However, our one / few-shot settings aren't strictly comparable to prior unsupervised work since they make use of a small amount of paired examples (1 or 64). This corresponds to up to a page or two of in-context training data.

Results are shown in Table 3.4. Zero-shot GPT-3, which only receives on a natural language description of the task, still underperforms recent unsupervised NMT results. However, providing only a single example demonstration for

<table><tr><td>Setting</td><td>En→Fr</td><td>Fr→En</td><td>En→De</td><td>De→En</td><td>En→Ro</td><td>Ro→En</td></tr><tr><td>SOTA (Supervised)</td><td> $45.6^a$ </td><td> $35.0^b$ </td><td> $41.2^c$ </td><td> $40.2^d$ </td><td> $38.5^e$ </td><td> $39.9^e$ </td></tr><tr><td>XLM [LC19]</td><td>33.4</td><td>33.3</td><td>26.4</td><td>34.3</td><td>33.3</td><td>31.8</td></tr><tr><td>MASS [STQ+19]</td><td>37.5</td><td>34.9</td><td>28.3</td><td>35.2</td><td>35.2</td><td>33.1</td></tr><tr><td>mBART [LGG+20]</td><td>-</td><td>-</td><td>29.8</td><td>34.0</td><td>35.0</td><td>30.5</td></tr><tr><td>GPT-3 Zero-Shot</td><td>25.2</td><td>21.2</td><td>24.6</td><td>27.2</td><td>14.1</td><td>19.9</td></tr><tr><td>GPT-3 One-Shot</td><td>28.3</td><td>33.7</td><td>26.2</td><td>30.4</td><td>20.6</td><td>38.6</td></tr><tr><td>GPT-3 Few-Shot</td><td>32.6</td><td>39.2</td><td>29.7</td><td>40.6</td><td>21.0</td><td>39.5</td></tr></table>

Table 3.4: Few-shot GPT-3 outperforms previous unsupervised NMT work by 5 BLEU when translating into English reflecting its strength as an English LM. We report BLEU scores on the WMT'14 Fr↔En, WMT'16 De↔En, and WMT'16 Ro↔En datasets as measured by multi-bleu.perl with XLM's tokenization in order to compare most closely with prior unsupervised NMT work. SacreBLEU $^{f}$ [Pos18] results reported in Appendix H. Underline indicates an unsupervised or few-shot SOTA, bold indicates supervised SOTA with relative confidence. $^{a}$ [EOAG18] $^{b}$ [DHKH14] $^{c}$ [WXH+18] $^{d}$ [oR16] $^{e}$ [LGG+20] $^{f}$ [SacreBLEU signature: BLEU+case.mixed+numrefs.1+smooth.exp+tok.intl+version.1.2.20]

![](images/f7bdb50f7e9e99ccda6d94946907be86bb51a08a98d7fbe96a8c9ea2ceafb0cf.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | French -> English | English -> French | German -> English | English -> German | Romanian -> English | English -> Romanian |
| --------------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------------------- | ------------------- |
| 0.1B                        | 5                 | 4                 | 3                 | 2                 | 1                   | 1                   |
| 0.4B                        | 25                | 15                | 23                | 12                | 6                   | 5                   |
| 0.8B                        | 28                | 19                | 26                | 15                | 9                   | 10                  |
| 1.3B                        | 31                | 21                | 29                | 17                | 11                  | 11                  |
| 2.6B                        | 34                | 25                | 33                | 21                | 14                  | 14                  |
| 6.7B                        | 35                | 27                | 35                | 23                | 16                  | 16                  |
| 13B                         | 37                | 29                | 37                | 26                | 18                  | 18                  |
| 175B                        | 39                | 32                | 40                | 30                | 21                  | 21                  |
</details>

Figure 3.4: Few-shot translation performance on 6 language pairs as model capacity increases. There is a consistent trend of improvement across all datasets as the model scales, and as well as tendency for translation into English to be stronger than translation from English.

<table><tr><td>Setting</td><td>Winograd</td><td>Winogrande (XL)</td></tr><tr><td>Fine-tuned SOTA</td><td> $\mathbf{90.1}^a$ </td><td> $\mathbf{84.6}^b$ </td></tr><tr><td>GPT-3 Zero-Shot</td><td>88.3*</td><td>70.2</td></tr><tr><td>GPT-3 One-Shot</td><td>89.7*</td><td>73.2</td></tr><tr><td>GPT-3 Few-Shot</td><td>88.6*</td><td>77.7</td></tr></table>

Table 3.5: Results on the WSC273 version of Winograd schemas and the adversarial Winogrande dataset. See Section 4 for details on potential contamination of the Winograd test set. ${}^{a}$ [SBBC19] ${}^{b}$ [LYN $^{+}$ 20]

![](images/71400fe75ae4f8ca8ca14830dcf3bfe9a328b03ee176b44eb2328a35d097e2fd.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=50) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 52.0      | 52.0     | 51.0            |
| 0.4B                        | 52.5      | 53.0     | 53.0            |
| 0.8B                        | 58.0      | 58.5     | 57.5            |
| 1.3B                        | 59.0      | 59.0     | 59.0            |
| 2.6B                        | 62.0      | 62.0     | 62.5            |
| 6.7B                        | 64.0      | 66.0     | 67.5            |
| 13B                         | 68.0      | 67.0     | 70.0            |
| 175B                        | 70.0      | 73.0     | 78.0            |
</details>

Figure 3.5: Zero-, one-, and few-shot performance on the adversarial Winogrande dataset as model capacity scales. Scaling is relatively smooth with the gains to few-shot learning increasing with model size, and few-shot GPT-3 175B is competitive with a fine-tuned RoBERTA-large.

each translation task improves performance by over 7 BLEU and nears competitive performance with prior work. GPT-3 in the full few-shot setting further improves another 4 BLEU resulting in similar average performance to prior unsupervised NMT work. GPT-3 has a noticeable skew in its performance depending on language direction. For the three input languages studied, GPT-3 significantly outperforms prior unsupervised NMT work when translating into English but underperforms when translating in the other direction. Performance on En-Ro is a noticeable outlier at over 10 BLEU worse than prior unsupervised NMT work. This could be a weakness due to reusing the byte-level BPE tokenizer of GPT-2 which was developed for an almost entirely English training dataset. For both Fr-En and De-En, few shot GPT-3 outperforms the best supervised result we could find but due to our unfamiliarity with the literature and the appearance that these are un-competitive benchmarks we do not suspect those results represent true state of the art. For Ro-En, few shot GPT-3 performs within 0.5 BLEU of the overall SOTA which is achieved by a combination of unsupervised pretraining, supervised finetuning on 608K labeled examples, and backtranslation [LHCG19b].

Finally, across all language pairs and across all three settings (zero-, one-, and few-shot), there is a smooth trend of improvement with model capacity. This is shown in Figure 3.4 in the case of few-shot results, and scaling for all three settings is shown in Appendix H.

# 3.4 Winograd-Style Tasks

The Winograd Schemas Challenge [LDM12] is a classical task in NLP that involves determining which word a pronoun refers to, when the pronoun is grammatically ambiguous but semantically unambiguous to a human. Recently fine-tuned language models have achieved near-human performance on the original Winograd dataset, but more difficult versions

<table><tr><td>Setting</td><td>PIQA</td><td>ARC (Easy)</td><td>ARC (Challenge)</td><td>OpenBookQA</td></tr><tr><td>Fine-tuned SOTA</td><td>79.4</td><td> $\mathbf{92.0}[\text{KKS}^{+}20]$ </td><td> $\mathbf{78.5}[\text{KKS}^{+}20]$ </td><td> $\mathbf{87.2}[\text{KKS}^{+}20]$ </td></tr><tr><td>GPT-3 Zero-Shot</td><td> $\mathbf{80.5*}$ </td><td>68.8</td><td>51.4</td><td>57.6</td></tr><tr><td>GPT-3 One-Shot</td><td> $\mathbf{80.5*}$ </td><td>71.2</td><td>53.2</td><td>58.8</td></tr><tr><td>GPT-3 Few-Shot</td><td> $\mathbf{82.8*}$ </td><td>70.1</td><td>51.5</td><td>65.4</td></tr></table>

Table 3.6: GPT-3 results on three commonsense reasoning tasks, PIQA, ARC, and OpenBookQA. GPT-3 Few-Shot PIQA result is evaluated on the test server. See Section 4 for details on potential contamination issues on the PIQA test set.

![](images/c9cf5894ef4f549ad018c19202706d410734c95eaf1728626b5a5c516d3c8347.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=50) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 64.0      | 64.0     | 64.0            |
| 0.4B                        | 70.0      | 69.0     | 69.0            |
| 0.8B                        | 73.0      | 72.0     | 72.0            |
| 1.3B                        | 75.0      | 74.0     | 74.0            |
| 2.6B                        | 76.0      | 74.0     | 75.0            |
| 6.7B                        | 78.0      | 76.0     | 78.0            |
| 13B                         | 79.0      | 78.0     | 80.0            |
| 175B                        | 81.0      | 80.0     | 82.0            |
</details>

Figure 3.6: GPT-3 results on PIQA in the zero-shot, one-shot, and few-shot settings. The largest model achieves a score on the development set in all three conditions that exceeds the best recorded score on the task.

such as the adversarially-mined Winogrande dataset [SBBC19] still significantly lag human performance. We test GPT-3's performance on both Winograd and Winogrande, as usual in the zero-, one-, and few-shot setting.

On Winograd we test GPT-3 on the original set of 273 Winograd schemas, using the same “partial evaluation” method described in $[RWC^{+}19]$ . Note that this setting differs slightly from the WSC task in the SuperGLUE benchmark, which is presented as binary classification and requires entity extraction to convert to the form described in this section. On Winograd GPT-3 achieves 88.3%, 89.7%, and 88.6% in the zero-shot, one-shot, and few-shot settings, showing no clear in-context learning but in all cases achieving strong results just a few points below state-of-the-art and estimated human performance. We note that contamination analysis found some Winograd schemas in the training data but this appears to have only a small effect on results (see Section 4).

On the more difficult Winogrande dataset, we do find gains to in-context learning: GPT-3 achieves 70.2% in the zero-shot setting, 73.2% in the one-shot setting, and 77.7% in the few-shot setting. For comparison a fine-tuned RoBERTA model achieves 79%, state-of-the-art is 84.6% achieved with a fine-tuned high capacity model (T5), and human performance on the task as reported by [SBBC19] is 94.0%.

# 3.5 Common Sense Reasoning

Next we consider three datasets which attempt to capture physical or scientific reasoning, as distinct from sentence completion, reading comprehension, or broad knowledge question answering. The first, PhysicalQA (PIQA) $[BZB^{+}19]$ , asks common sense questions about how the physical world works and is intended as a probe of grounded understanding of the world. GPT-3 achieves 81.0% accuracy zero-shot, 80.5% accuracy one-shot, and 82.8% accuracy few-shot (the last measured on PIQA's test server). This compares favorably to the 79.4% accuracy prior state-of-the-art of a

<table><tr><td>Setting</td><td>CoQA</td><td>DROP</td><td>QuAC</td><td>SQuADv2</td><td>RACE-h</td><td>RACE-m</td></tr><tr><td>Fine-tuned SOTA</td><td> $90.7^a$ </td><td> $89.1^b$ </td><td> $74.4^c$ </td><td> $93.0^d$ </td><td> $90.0^e$ </td><td> $93.1^e$ </td></tr><tr><td>GPT-3 Zero-Shot</td><td>81.5</td><td>23.6</td><td>41.5</td><td>59.5</td><td>45.5</td><td>58.4</td></tr><tr><td>GPT-3 One-Shot</td><td>84.0</td><td>34.3</td><td>43.3</td><td>65.4</td><td>45.9</td><td>57.4</td></tr><tr><td>GPT-3 Few-Shot</td><td>85.0</td><td>36.5</td><td>44.3</td><td>69.8</td><td>46.8</td><td>58.1</td></tr></table>

Table 3.7: Results on reading comprehension tasks. All scores are F1 except results for RACE which report accuracy. $^{a}[JZC^{+}19]$ $^{b}[JN20]$ $^{c}[AI19]$ $^{d}[QIA20]$ $^{e}[SPP^{+}19]$

fine-tuned RoBERTa. PIQA shows relatively shallow scaling with model size and is still over 10% worse than human performance, but GPT-3's few-shot and even zero-shot result outperform the current state-of-the-art. Our analysis flagged PIQA for a potential data contamination issue (despite hidden test labels), and we therefore conservatively mark the result with an asterisk. See Section 4 for details.

ARC $[CCE^{+}18]$ is a dataset of multiple-choice questions collected from 3rd to 9th grade science exams. On the “Challenge” version of the dataset which has been filtered to questions which simple statistical or information retrieval methods are unable to correctly answer, GPT-3 achieves 51.4% accuracy in the zero-shot setting, 53.2% in the one-shot setting, and 51.5% in the few-shot setting. This is approaching the performance of a fine-tuned RoBERTa baseline (55.9%) from UnifiedQA $[KKS^{+}20]$ . On the “Easy” version of the dataset (questions which either of the mentioned baseline approaches answered correctly), GPT-3 achieves 68.8%, 71.2%, and 70.1% which slightly exceeds a fine-tuned RoBERTa baseline from $[KKS^{+}20]$ . However, both of these results are still much worse than the overall SOTAs achieved by the UnifiedQA which exceeds GPT-3’s few-shot results by 27% on the challenge set and 22% on the easy set.

On OpenBookQA [MCKS18], GPT-3 improves significantly from zero to few shot settings but is still over 20 points short of the overall SOTA. GPT-3's few-shot performance is similar to a fine-tuned BERT Large baseline on the leaderboard.

Overall, in-context learning with GPT-3 shows mixed results on commonsense reasoning tasks, with only small and inconsistent gains observed in the one and few-shot learning settings for both PIQA and ARC, but a significant improvement is observed on OpenBookQA. GPT-3 sets SOTA on the new PIQA dataset in all evaluation settings.

# 3.6 Reading Comprehension

Next we evaluate GPT-3 on the task of reading comprehension. We use a suite of 5 datasets including abstractive, multiple choice, and span based answer formats in both dialog and single question settings. We observe a wide spread in GPT-3's performance across these datasets suggestive of varying capability with different answer formats. In general we observe GPT-3 is on par with initial baselines and early results trained using contextual representations on each respective dataset.

GPT-3 performs best (within 3 points of the human baseline) on CoQA [RCM19] a free-form conversational dataset and performs worst (13 F1 below an ELMo baseline) on QuAC $\left[\mathrm{CHI}^{+}18\right]$ a dataset which requires modeling structured dialog acts and answer span selections of teacher-student interactions. On DROP $\left[\mathrm{DWD}^{+}19\right]$ , a dataset testing discrete reasoning and numeracy in the context of reading comprehension, GPT-3 in a few-shot setting outperforms the fine-tuned BERT baseline from the original paper but is still well below both human performance and state-of-the-art approaches which augment neural networks with symbolic systems $\left[\mathrm{RLL}^{+}19\right]$ . On SQuAD 2.0 [RJL18], GPT-3 demonstrates its few-shot learning capabilities, improving by almost 10 F1 (to 69.8) compared to a zero-shot setting. This allows it to slightly outperform the best fine-tuned result in the original paper. On RACE $\left[\mathrm{LXL}^{+}17\right]$ , a multiple choice dataset of middle school and high school english examinations, GPT-3 performs relatively weakly and is only competitive with the earliest work utilizing contextual representations and is still $45\%$ behind SOTA.

# 3.7 SuperGLUE

In order to better aggregate results on NLP tasks and compare to popular models such as BERT and RoBERTa in a more systematic way, we also evaluate GPT-3 on a standardized collection of datasets, the SuperGLUE benchmark $[WPN^{+}19]$ $[WPN^{+}19]$ $[CLC^{+}19]$ $[DMST19]$ $[RBG11]$ $[KCR^{+}18]$ $[ZLL^{+}18]$ $[DGM06]$ $[BHDD^{+}06]$ $[GMDD07]$ $[BDD^{+}09]$ $[PCC18]$ $[PHR^{+}18]$ . GPT-3's test-set performance on the SuperGLUE dataset is shown in Table 3.8. In the few-shot setting, we used 32 examples for all tasks, sampled randomly from the training set. For all tasks except WSC

![](images/d853e60e3e8c9f486cfb197ee82db4cd523e67be761971bc7758aaa3fc8f4ad2.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=5) |
| --------------------------- | --------- | -------- | -------------- |
| 0.1B                        | 34        | 31       | 31             |
| 0.4B                        | 55        | 52       | 52             |
| 0.8B                        | 62        | 62       | 63             |
| 1.3B                        | 65        | 66       | 67             |
| 2.6B                        | 71        | 72       | 73             |
| 6.7B                        | 73        | 75       | 77             |
| 13B                         | 76        | 78       | 80             |
| 175B                        | 81        | 84       | 85             |
</details>

Figure 3.7: GPT-3 results on CoQA reading comprehension task. GPT-3 175B achieves 85 F1 in the few-shot setting, only a few points behind measured human performance and state-of-the-art fine-tuned models. Zero-shot and one-shot performance is a few points behind, with the gains to few-shot being largest for bigger models.

<table><tr><td></td><td>SuperGLUE Average</td><td>BoolQ Accuracy</td><td>CB Accuracy</td><td>CB F1</td><td>COPA Accuracy</td><td>RTE Accuracy</td></tr><tr><td>Fine-tuned SOTA</td><td>89.0</td><td>91.0</td><td>96.9</td><td>93.9</td><td>94.8</td><td>92.5</td></tr><tr><td>Fine-tuned BERT-Large</td><td>69.0</td><td>77.4</td><td>83.6</td><td>75.7</td><td>70.6</td><td>71.7</td></tr><tr><td>GPT-3 Few-Shot</td><td>71.8</td><td>76.4</td><td>75.6</td><td>52.0</td><td>92.0</td><td>69.0</td></tr><tr><td></td><td>WiC Accuracy</td><td>WSC Accuracy</td><td>MultiRC Accuracy</td><td>MultiRC F1a</td><td>ReCoRD Accuracy</td><td>ReCoRD F1</td></tr><tr><td>Fine-tuned SOTA</td><td>76.1</td><td>93.8</td><td>62.3</td><td>88.2</td><td>92.5</td><td>93.3</td></tr><tr><td>Fine-tuned BERT-Large</td><td>69.6</td><td>64.6</td><td>24.1</td><td>70.0</td><td>71.3</td><td>72.0</td></tr><tr><td>GPT-3 Few-Shot</td><td>49.4</td><td>80.1</td><td>30.5</td><td>75.4</td><td>90.2</td><td>91.1</td></tr></table>

Table 3.8: Performance of GPT-3 on SuperGLUE compared to fine-tuned baselines and SOTA. All results are reported on the test set. GPT-3 few-shot is given a total of 32 examples within the context of each task and performs no gradient updates.

![](images/c29fcc005ca073964ee37a55cab63a6ff2a17065c647657695464109cb5038a7.jpg)

<details>
<summary>line</summary>

| Billions of Parameters in LM | Zero-shot | One-shot | Few-shot (K=32) |
| ---------------------------- | --------- | -------- | --------------- |
| 0.1                          | 40        | 55       | 50              |
| 0.4                          | 48        | 56       | 57              |
| 0.8                          | 47        | 57       | 57              |
| 1.3                          | 50        | 58       | 60              |
| 2.6                          | 50        | 61       | 64              |
| 6.7                          | 52        | 60       | 64              |
| 13                           | 54        | 64       | 67              |
| 175                          | 58        | 69       | 73              |
</details>

![](images/998f6a79490309c8c3e823634aadb97714c638ff560bc04fdec6a57dfe68ab0a.jpg)

<details>
<summary>line</summary>

| Number of Examples in Context (K) | Few-shot GPT-3 175B |
| --------------------------------- | ------------------- |
| 0                                 | 58                  |
| 1                                 | 69                  |
| 2                                 | 69                  |
| 3                                 | 69                  |
| 4                                 | 70                  |
| 8                                 | 72                  |
| 16                                | 73                  |
| 32                                | 74                  |
</details>

Figure 3.8: Performance on SuperGLUE increases with model size and number of examples in context. A value of K = 32 means that our model was shown 32 examples per task, for 256 examples total divided across the 8 tasks in SuperGLUE. We report GPT-3 values on the dev set, so our numbers are not directly comparable to the dotted reference lines (our test set results are in Table 3.8). The BERT-Large reference model was fine-tuned on the SuperGLUE training set (125K examples), whereas BERT++ was first fine-tuned on MultiNLI (392K examples) and SWAG (113K examples) before further fine-tuning on the SuperGLUE training set (for a total of 630K fine-tuning examples). We find the difference in performance between the BERT-Large and BERT++ to be roughly equivalent to the difference between GPT-3 with one example per context versus eight examples per context.

and MultiRC, we sampled a new set of examples to use in the context for each problem. For WSC and MultiRC, we used the same set of randomly drawn examples from the training set as context for all of the problems we evaluated.

We observe a wide range in GPT-3's performance across tasks. On COPA and ReCoRD GPT-3 achieves near-SOTA performance in the one-shot and few-shot settings, with COPA falling only a couple points short and achieving second place on the leaderboard, where first place is held by a fine-tuned 11 billion parameter model (T5). On WSC, performance is still relatively strong, achieving $80.1\%$ in the few-shot setting (note that GPT-3 achieves $88.6\%$ on the original Winograd dataset as described in Section 3.4). On BoolQ, MultiRC, and RTE, performance is reasonable, roughly matching that of a fine-tuned BERT-Large. On CB, we see signs of life at $75.6\%$ in the few-shot setting.

WiC is a notable weak spot with few-shot performance at 49.4% (at random chance). We tried a number of different phrasings and formulations for WiC (which involves determining if a word is being used with the same meaning in two sentences), none of which was able to achieve strong performance. This hints at a phenomenon that will become clearer in the next section (which discusses the ANLI benchmark) – GPT-3 appears to be weak in the few-shot or one-shot setting at some tasks that involve comparing two sentences or snippets, for example whether a word is used the same way in two sentences (WiC), whether one sentence is a paraphrase of another, or whether one sentence implies another. This could also explain the comparatively low scores for RTE and CB, which also follow this format. Despite these weaknesses, GPT-3 still outperforms a fine-tuned BERT-large on four of eight tasks and on two tasks GPT-3 is close to the state-of-the-art held by a fine-tuned 11 billion parameter model.

Finally, we note that the few-shot SuperGLUE score steadily improves with both model size and with number of examples in the context showing increasing benefits from in-context learning (Figure 3.8). We scale K up to 32 examples per task, after which point additional examples will not reliably fit into our context. When sweeping over values of K, we find that GPT-3 requires less than eight total examples per task to outperform a fine-tuned BERT-Large on overall SuperGLUE score.

# 3.8 NLI

Natural Language Inference (NLI) [Fyo00] concerns the ability to understand the relationship between two sentences. In practice, this task is usually structured as a two or three class classification problem where the model classifies

![](images/6ac44f214dc83cc43fa40ba9b71a3f00483b3ae77a9a1ebe8ceaff6c0db9d407.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=50) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 33.8      | 35.0     | 35.0            |
| 0.4B                        | 34.0      | 32.8     | 34.5            |
| 0.8B                        | 33.9      | 33.0     | 35.2            |
| 1.3B                        | 33.5      | 34.0     | 36.0            |
| 2.6B                        | 35.3      | 34.1     | 32.8            |
| 6.7B                        | 34.8      | 33.2     | 34.0            |
| 13B                         | 34.5      | 32.5     | 34.5            |
| 175B                        | 34.5      | 35.2     | 40.2            |
</details>

Figure 3.9: Performance of GPT-3 on ANLI Round 3. Results are on the dev-set, which has only 1500 examples and therefore has high variance (we estimate a standard deviation of 1.2%). We find that smaller models hover around random chance, while few-shot GPT-3 175B closes almost half the gap from random chance to SOTA. Results for ANLI rounds 1 and 2 are shown in the appendix.

whether the second sentence logically follows from the first, contradicts the first sentence, or is possibly true (neutral). SuperGLUE includes an NLI dataset, RTE, which evaluates the binary version of the task. On RTE, only the largest version of GPT-3 performs convincingly better than random (56%) in any evaluation setting, but in a few-shot setting GPT-3 performs similarly to a single-task fine-tuned BERT Large. We also evaluate on the recently introduced Adversarial Natural Language Inference (ANLI) dataset $[NWD^{+}19]$ . ANLI is a difficult dataset employing a series of adversarially mined natural language inference questions in three rounds (R1, R2, and R3). Similar to RTE, all of our models smaller than GPT-3 perform at almost exactly random chance on ANLI, even in the few-shot setting ( $\sim 33\%$ ), whereas GPT-3 itself shows signs of life on Round 3. Results for ANLI R3 are highlighted in Figure 3.9 and full results for all rounds can be found in Appendix H. These results on both RTE and ANLI suggest that NLI is still a very difficult task for language models and they are only just beginning to show signs of progress.

# 3.9 Synthetic and Qualitative Tasks

One way to probe GPT-3's range of abilities in the few-shot (or zero- and one-shot) setting is to give it tasks which require it to perform simple on-the-fly computational reasoning, recognize a novel pattern that is unlikely to have occurred in training, or adapt quickly to an unusual task. We devise several tasks to test this class of abilities. First, we test GPT-3's ability to perform arithmetic. Second, we create several tasks that involve rearranging or unscrambling the letters in a word, tasks which are unlikely to have been exactly seen during training. Third, we test GPT-3's ability to solve SAT-style analogy problems few-shot. Finally, we test GPT-3 on several qualitative tasks, including using new words in a sentence, correcting English grammar, and news article generation. We will release the synthetic datasets with the hope of stimulating further study of test-time behavior of language models.

# 3.9.1 Arithmetic

To test GPT-3's ability to perform simple arithmetic operations without task-specific training, we developed a small battery of 10 tests that involve asking GPT-3 a simple arithmetic problem in natural language:

- 2 digit addition (2D+) – The model is asked to add two integers sampled uniformly from [0, 100), phrased in the form of a question, e.g. “Q: What is 48 plus 76? A: 124.”   
- 2 digit subtraction (2D-) – The model is asked to subtract two integers sampled uniformly from [0, 100); the answer may be negative. Example: “Q: What is 34 minus 53? A: -19”.   
- 3 digit addition (3D+) – Same as 2 digit addition, except numbers are uniformly sampled from [0, 1000).

![](images/b1958c5c00e193ed3ea231f8dd8edd92cea4890e0293f5e8d9aea36b0102b1bf.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Two Digit Addition | Two Digit Subtraction | Three Digit Addition | Three Digit Subtraction | Four Digit Addition | Four Digit Subtraction | Five Digit Addition | Five Digit Subtraction | Two Digit Multiplication | Single Digit Three Ops |
| --------------------------- | ------------------ | --------------------- | -------------------- | ----------------------- | ------------------- | ---------------------- | ------------------- | ---------------------- | ------------------------ | ----------------------- |
| 0.1B                        | 2                  | 1                     | 1                    | 1                       | 1                   | 0                      | 1                   | 0                      | 1                        | 1                       |
| 0.4B                        | 3                  | 2                     | 2                    | 2                       | 2                   | 0                      | 2                   | 0                      | 2                        | 2                       |
| 0.8B                        | 4                  | 3                     | 3                    | 3                       | 3                   | 0                      | 3                   | 0                      | 3                        | 3                       |
| 1.3B                        | 5                  | 4                     | 4                    | 4                       | 4                   | 0                      | 4                   | 0                      | 4                        | 4                       |
| 2.6B                        | 8                  | 6                     | 6                    | 6                       | 6                   | 0                      | 6                   | 0                      | 6                        | 6                       |
| 6.7B                        | 12                 | 14                    | 12                   | 12                      | 12                  | 0                      | 12                  | 0                      | 12                       | 12                      |
| 13B                         | 55                 | 52                    | 50                   | 50                      | 50                  | 0                      | 50                  | 0                      | 50                       | 50                      |
| 175B                        | 100                | 98                    | 80                   | 95                      | 25                  | 10                     | 25                  | 10                     | 30                       | 20                      |
</details>

Figure 3.10: Results on all 10 arithmetic tasks in the few-shot settings for models of different sizes. There is a significant jump from the second largest model (GPT-3 13B) to the largest model (GPT-3 175), with the latter being able to reliably accurate 2 digit arithmetic, usually accurate 3 digit arithmetic, and correct answers a significant fraction of the time on 4-5 digit arithmetic, 2 digit multiplication, and compound operations. Results for one-shot and zero-shot are shown in the appendix.

- 3 digit subtraction (3D-) – Same as 2 digit subtraction, except numbers are uniformly sampled from [0, 1000).   
- 4 digit addition (4D+) – Same as 3 digit addition, except uniformly sampled from [0, 10000).   
- 4 digit subtraction (4D-) – Same as 3 digit subtraction, except uniformly sampled from [0, 10000).   
- 5 digit addition (5D+) – Same as 3 digit addition, except uniformly sampled from [0, 100000).   
- 5 digit subtraction (5D-) – Same as 3 digit subtraction, except uniformly sampled from [0, 100000).   
- 2 digit multiplication (2Dx) – The model is asked to multiply two integers sampled uniformly from [0, 100), e.g. “Q: What is 24 times 42? A: 1008”.   
- One-digit composite (1DC) – The model is asked to perform a composite operation on three 1 digit numbers, with parentheses around the last two. For example, “Q: What is 6+(4\*8)? A: 38”. The three 1 digit numbers are selected uniformly on [0, 10) and the operations are selected uniformly from $\{+, -, *\}$ .

In all 10 tasks the model must generate the correct answer exactly. For each task we generate a dataset of 2,000 random instances of the task and evaluate all models on those instances.

First we evaluate GPT-3 in the few-shot setting, for which results are shown in Figure 3.10. On addition and subtraction, GPT-3 displays strong proficiency when the number of digits is small, achieving 100% accuracy on 2 digit addition, 98.9% at 2 digit subtraction, 80.2% at 3 digit addition, and 94.2% at 3-digit subtraction. Performance decreases as the number of digits increases, but GPT-3 still achieves 25-26% accuracy on four digit operations and 9-10% accuracy on five digit operations, suggesting at least some capacity to generalize to larger numbers of digits. GPT-3 also achieves 29.2% accuracy at 2 digit multiplication, an especially computationally intensive operation. Finally, GPT-3 achieves 21.3% accuracy at single digit combined operations (for example, $9*(7+5)$ ), suggesting that it has some robustness beyond just single operations.

As Figure 3.10 makes clear, small models do poorly on all of these tasks – even the 13 billion parameter model (the second largest after the 175 billion full GPT-3) can solve 2 digit addition and subtraction only half the time, and all other operations less than 10% of the time.

One-shot and zero-shot performance are somewhat degraded relative to few-shot performance, suggesting that adaptation to the task (or at the very least recognition of the task) is important to performing these computations correctly. Nevertheless, one-shot performance is still quite strong, and even zero-shot performance of the full GPT-3 significantly

<table><tr><td>Setting</td><td>2D+</td><td>2D-</td><td>3D+</td><td>3D-</td><td>4D+</td><td>4D-</td><td>5D+</td><td>5D-</td><td>2Dx</td><td>1DC</td></tr><tr><td>GPT-3 Zero-shot</td><td>76.9</td><td>58.0</td><td>34.2</td><td>48.3</td><td>4.0</td><td>7.5</td><td>0.7</td><td>0.8</td><td>19.8</td><td>9.8</td></tr><tr><td>GPT-3 One-shot</td><td>99.6</td><td>86.4</td><td>65.5</td><td>78.7</td><td>14.0</td><td>14.0</td><td>3.5</td><td>3.8</td><td>27.4</td><td>14.3</td></tr><tr><td>GPT-3 Few-shot</td><td>100.0</td><td>98.9</td><td>80.4</td><td>94.2</td><td>25.5</td><td>26.8</td><td>9.3</td><td>9.9</td><td>29.2</td><td>21.3</td></tr></table>

Table 3.9: Results on basic arithmetic tasks for GPT-3 175B. $\{2,3,4,5\}D\{+,-\}$ is 2, 3, 4, and 5 digit addition or subtraction, 2Dx is 2 digit multiplication. 1DC is 1 digit composite operations. Results become progressively stronger moving from the zero-shot to one-shot to few-shot setting, but even the zero-shot shows significant arithmetic abilities.

<table><tr><td>Setting</td><td>CL</td><td>A1</td><td>A2</td><td>RI</td><td>RW</td></tr><tr><td>GPT-3 Zero-shot</td><td>3.66</td><td>2.28</td><td>8.91</td><td>8.26</td><td>0.09</td></tr><tr><td>GPT-3 One-shot</td><td>21.7</td><td>8.62</td><td>25.9</td><td>45.4</td><td>0.48</td></tr><tr><td>GPT-3 Few-shot</td><td>37.9</td><td>15.1</td><td>39.7</td><td>67.2</td><td>0.44</td></tr></table>

Table 3.10: GPT-3 175B performance on various word unscrambling and word manipulation tasks, in zero-, one-, and few-shot settings. CL is “cycle letters in word”, A1 is anagrams of but the first and last letters, A2 is anagrams of all but the first and last two letters, RI is “Random insertion in word”, RW is “reversed words”.

outperforms few-shot learning for all smaller models. All three settings for the full GPT-3 are shown in Table 3.9, and model capacity scaling for all three settings is shown in Appendix H.

To spot-check whether the model is simply memorizing specific arithmetic problems, we took the 3-digit arithmetic problems in our test set and searched for them in our training data in both the forms "<NUM1> + <NUM2> =" and "<NUM1> plus <NUM2>" . Out of 2,000 addition problems we found only 17 matches (0.8%) and out of 2,000 subtraction problems we found only 2 matches (0.1%), suggesting that only a trivial fraction of the correct answers could have been memorized. In addition, inspection of incorrect answers reveals that the model often makes mistakes such as not carrying a "1", suggesting it is actually attempting to perform the relevant computation rather than memorizing a table.

Overall, GPT-3 displays reasonable proficiency at moderately complex arithmetic in few-shot, one-shot, and even zero-shot settings.

# 3.9.2 Word Scrambling and Manipulation Tasks

To test GPT-3's ability to learn novel symbolic manipulations from a few examples, we designed a small battery of 5 “character manipulation” tasks. Each task involves giving the model a word distorted by some combination of scrambling, addition, or deletion of characters, and asking it to recover the original word. The 5 tasks are:

- Cycle letters in word (CL) – The model is given a word with its letters cycled, then the “=” symbol, and is expected to generate the original word. For example, it might be given “lyinevitab” and should output “inevitably”.   
- Anagrams of all but first and last characters (A1) – The model is given a word where every letter except the first and last have been scrambled randomly, and must output the original word. Example: criroptuon = corruption.   
- Anagrams of all but first and last 2 characters (A2) – The model is given a word where every letter except the first 2 and last 2 have been scrambled randomly, and must recover the original word. Example: opoepnnt → opponent.   
- Random insertion in word (RI) – A random punctuation or space character is inserted between each letter of a word, and the model must output the original word. Example: s.u!c/c!e.s s i/o/n = succession.   
- Reversed words (RW) – The model is given a word spelled backwards, and must output the original word. Example: stcejbo → objects.

For each task we generate 10,000 examples, which we chose to be the top 10,000 most frequent words as measured by [Nor09] of length more than 4 characters and less than 15 characters. The few-shot results are shown in Figure 3.11. Task performance tends to grow smoothly with model size, with the full GPT-3 model achieving $66.9\%$ on removing

![](images/d0e0f650bea213ed420cb0a6c8aaf93d0800d44d0c1ed84b3a4913c964c47ce4.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | cycle letters | mid word 1 anagrams | mid word 2 anagrams | random insertion | reversed words |
| --------------------------- | ------------- | -------------------- | -------------------- | ----------------- | -------------- |
| 0.1B                        | 4             | 0                    | 2                    | 0                 | 0              |
| 0.4B                        | 9             | 1                    | 5                    | 0                 | 0              |
| 0.8B                        | 11            | 2                    | 8                    | 2                 | 0              |
| 1.3B                        | 15            | 3                    | 10                   | 4                 | 0              |
| 2.6B                        | 17            | 4                    | 13                   | 7                 | 0              |
| 6.7B                        | 22            | 5                    | 19                   | 11                | 0              |
| 13B                         | 28            | 8                    | 26                   | 28                | 0              |
| 175B                        | 38            | 15                   | 40                   | 68                | 0              |
</details>

Figure 3.11: Few-shot performance on the five word scrambling tasks for different sizes of model. There is generally smooth improvement with model size although the random insertion task shows an upward slope of improvement with the 175B model solving the task the majority of the time. Scaling of one-shot and zero-shot performance is shown in the appendix. All tasks are done with K = 100.

random insertions, 38.6% on cycling letters, 40.2% on the easier anagram task, and 15.1% on the more difficult anagram task (where only the first and last letters are held fixed). None of the models can reverse the letters in a word.

In the one-shot setting, performance is significantly weaker (dropping by half or more), and in the zero-shot setting the model can rarely perform any of the tasks (Table 3.10). This suggests that the model really does appear to learn these tasks at test time, as the model cannot perform them zero-shot and their artificial nature makes them unlikely to appear in the pre-training data (although we cannot confirm this with certainty).

We can further quantify performance by plotting “in-context learning curves”, which show task performance as a function of the number of in-context examples. We show in-context learning curves for the Symbol Insertion task in Figure 1.2. We can see that larger models are able to make increasingly effective use of in-context information, including both task examples and natural language task descriptions.

Finally, it is worth adding that solving these tasks requires character-level manipulations, whereas our BPE encoding operates on significant fractions of a word (on average $\sim 0.7$ words per token), so from the LM's perspective succeeding at these tasks involves not just manipulating BPE tokens but understanding and pulling apart their substructure. Also, CL, A1, and A2 are not bijective (that is, the unscrambled word is not a deterministic function of the scrambled word), requiring the model to perform some search to find the correct unscrambling. Thus, the skills involved appear to require non-trivial pattern-matching and computation.

# 3.9.3 SAT Analogies

To test GPT-3 on another task that is somewhat unusual relative to the typical distribution of text, we collected a set of 374 “SAT analogy” problems [TLBS03]. Analogies are a style of multiple choice question that constituted a section of the SAT college entrance exam before 2005. A typical example is “audacious is to boldness as (a) sanctimonious is to hypocrisy, (b) anonymous is to identity, (c) remorseful is to misdeed, (d) deleterious is to result, (e) impressionable is to temptation”. The student is expected to choose which of the five word pairs has the same relationship as the original word pair; in this example the answer is “sanctimonious is to hypocrisy”. On this task GPT-3 achieves 65.2% in the few-shot setting, 59.1% in the one-shot setting, and 53.7% in the zero-shot setting, whereas the average score among college applicants was 57% [TL05] (random guessing yields 20%). As shown in Figure 3.12, the results improve with scale, with the full 175 billion model improving by over 10% compared to the 13 billion parameter model.

![](images/24a7b274cf904c57165e1d1f4b219cbcc614c6c1a3e8705b4b9a445017c49294.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=20) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 35.0      | 30.0     | 30.0            |
| 0.4B                        | 39.0      | 41.0     | 40.0            |
| 0.8B                        | 45.0      | 43.0     | 42.0            |
| 1.3B                        | 44.0      | 46.0     | 40.0            |
| 2.6B                        | 50.0      | 55.0     | 48.0            |
| 6.7B                        | 49.0      | 54.0     | 52.0            |
| 13B                         | 53.0      | 53.0     | 53.0            |
| 175B                        | 54.0      | 59.0     | 65.0            |
</details>

Figure 3.12: Zero-, one-, and few-shot performance on SAT analogy tasks, for different sizes of model. The largest model achieves 65% accuracy in the few-shot setting, and also demonstrates significant gains to in-context learning which are not present in smaller models.

# 3.9.4 News Article Generation

Previous work on generative language models qualitatively tested their ability to generate synthetic “news articles” by conditional sampling from the model given a human-written prompt consisting of a plausible first sentence for a news story $[RWC^{+}19]$ . Relative to $[RWC^{+}19]$ , the dataset used to train GPT-3 is much less weighted towards news articles, so trying to generate news articles via raw unconditional samples is less effective – for example GPT-3 often interprets the proposed first sentence of a “news article” as a tweet and then posts synthetic responses or follow-up tweets. To solve this problem we employed GPT-3’s few-shot learning abilities by providing three previous news articles in the model’s context to condition it. With the title and subtitle of a proposed next article, the model is able to reliably generate short articles in the “news” genre.

To gauge the quality of news article generation from GPT-3 (which we believe is likely to be correlated with conditional sample generation quality in general), we decided to measure human ability to distinguish GPT-3-generated articles from real ones. Similar work has been carried out by Kreps et al. [KMB20] and Zellers et al. $[ZHR^{+}19]$ . Generative language models are trained to match the distribution of content generated by humans, so the (in)ability of humans to distinguish the two is a potentially important measure of quality. $^{3}$

In order to see how well humans can detect model generated text, we arbitrarily selected 25 article titles and subtitles from the website newser.com (mean length: 215 words). We then generated completions of these titles and subtitles from four language models ranging in size from 125M to 175B (GPT-3) parameters (mean length: 200 words). For each model, we presented around 80 US-based participants with a quiz consisting of these real titles and subtitles followed by either the human written article or the article generated by the model $^{4}$ . Participants were asked to select whether the article was “very likely written by a human”, “more likely written by a human”, “I don’t know”, “more likely written by a machine”, or “very likely written by a machine”.

The articles we selected were not in the models' training data and the model outputs were formatted and selected programmatically to prevent human cherry-picking. All models used the same context to condition outputs on and were pre-trained with the same context size and the same article titles and subtitles were used as prompts for each model. However, we also ran an experiment to control for participant effort and attention that followed the same format but involved intentionally bad model generated articles. This was done by generating articles from a “control model”: a 160M parameter model with no context and increased output randomness.

<table><tr><td></td><td>Mean accuracy</td><td>95% Confidence Interval (low, hi)</td><td>t compared to control (p-value)</td><td>“I don’t know” assignments</td></tr><tr><td>Control (deliberately bad model)</td><td>86%</td><td>83%-90%</td><td>-</td><td>3.6 %</td></tr><tr><td>GPT-3 Small</td><td>76%</td><td>72%-80%</td><td>3.9 (2e-4)</td><td>4.9%</td></tr><tr><td>GPT-3 Medium</td><td>61%</td><td>58%-65%</td><td>10.3 (7e-21)</td><td>6.0%</td></tr><tr><td>GPT-3 Large</td><td>68%</td><td>64%-72%</td><td>7.3 (3e-11)</td><td>8.7%</td></tr><tr><td>GPT-3 XL</td><td>62%</td><td>59%-65%</td><td>10.7 (1e-19)</td><td>7.5%</td></tr><tr><td>GPT-3 2.7B</td><td>62%</td><td>58%-65%</td><td>10.4 (5e-19)</td><td>7.1%</td></tr><tr><td>GPT-3 6.7B</td><td>60%</td><td>56%-63%</td><td>11.2 (3e-21)</td><td>6.2%</td></tr><tr><td>GPT-3 13B</td><td>55%</td><td>52%-58%</td><td>15.3 (1e-32)</td><td>7.1%</td></tr><tr><td>GPT-3 175B</td><td>52%</td><td>49%-54%</td><td>16.9 (1e-34)</td><td>7.8%</td></tr></table>

Table 3.11: Human accuracy in identifying whether short ( $\sim$ 200 word) news articles are model generated. We find that human accuracy (measured by the ratio of correct assignments to non-neutral assignments) ranges from 86% on the control model to 52% on GPT-3 175B. This table compares mean accuracy between five different models, and shows the results of a two-sample T-Test for the difference in mean accuracy between each model and the control model (an unconditional GPT-3 Small model with increased output randomness).

Mean human accuracy (the ratio of correct assignments to non-neutral assignments per participant) at detecting that the intentionally bad articles were model generated was $\sim86\%$ where 50% is chance level performance. By contrast, mean human accuracy at detecting articles that were produced by the 175B parameter model was barely above chance at $\sim52\%$ (see Table 3.11). $^{5}$ Human abilities to detect model generated text appear to decrease as model size increases: there appears to be a trend towards chance accuracy with model size, and human detection of GPT-3 is close to chance. $^{6}$ This is true despite the fact that participants spend more time on each output as model size increases (see Appendix E).

Examples of synthetic articles from GPT-3 are given in Figures 3.14 and 3.15. $^{7}$ Much of the text is—as indicated by the evaluations—difficult for humans to distinguish from authentic human content. Factual inaccuracies can be an indicator that an article is model generated since, unlike human authors, the models have no access to the specific facts that the article titles refer to or when the article was written. Other indicators include repetition, non sequiturs, and unusual phrasings, though these are often subtle enough that they are not noticed.

Related work on language model detection by Ippolito et al. [IDCBE19] indicates that automatic discriminators like GROVER $[ZHR^{+}19]$ and GLTR [GSR19] may have greater success at detecting model generated text than human evaluators. Automatic detection of these models may be a promising area of future research.

Ippolito et al. [IDCBE19] also note that human accuracy at detecting model generated text increases as humans observe more tokens. To do a preliminary investigation of how good humans are at detecting longer news articles generated by GPT-3 175B, we selected 12 world news articles from Reuters with an average length of 569 words and generated completions of these articles from GPT-3 with an average length of 498 words (298 words longer than our initial experiments). Following the methodology above, we ran two experiments, each on around 80 US-based participants, to compare human abilities to detect the articles generated by GPT-3 and a control model.

We found that mean human accuracy at detecting the intentionally bad longer articles from the control model was $\sim88\%$ , while mean human accuracy at detecting the longer articles that were produced by GPT-3 175B was still barely above chance at $\sim52\%$ (see Table 3.12). This indicates that, for news articles that are around 500 words long, GPT-3 continues to produce articles that humans find difficult to distinguish from human written news articles.

# 3.9.5 Learning and Using Novel Words

A task studied in developmental linguistics [CB78] is the ability to learn and utilize new words, for example using a word in a sentence after seeing it defined only once, or conversely inferring a word's meaning from only one usage. Here we qualitatively test GPT-3's ability to do the former. Specifically, we give GPT-3 the definition of a nonexistent word, such as “Gigamuru”, and then ask it to use it in a sentence. We provide one to five previous examples of a (separate)

Human ability to detect model generated news articles   
![](images/5e62c87d7cc8b45ffb716b3bee8be2e2fe7a999e88e16c7b3eeda18c4c176de9.jpg)

<details>
<summary>line</summary>

| Number of parameters (log scale) | Accuracy (%) |
| -------------------------------- | ------------ |
| 1e8                              | 75.0         |
| 1e9                              | 62.0         |
| 1e10                             | 55.0         |
| 1e11                             | 52.0         |
</details>

Figure 3.13: People's ability to identify whether news articles are model-generated (measured by the ratio of correct assignments to non-neutral assignments) decreases as model size increases. Accuracy on the outputs on the deliberately-bad control model (an unconditioned GPT-3 Small model with higher output randomness) is indicated with the dashed line at the top, and the random chance $(50\%)$ is indicated with the dashed line at the bottom. Line of best fit is a power law with $95\%$ confidence intervals.

<table><tr><td></td><td>Mean accuracy</td><td>95% Confidence Interval (low, hi)</td><td>t compared to control (p-value)</td><td>“I don’t know” assignments</td></tr><tr><td>Control</td><td>88%</td><td>84%–91%</td><td>-</td><td>2.7%</td></tr><tr><td>GPT-3 175B</td><td>52%</td><td>48%–57%</td><td>12.7 (3.2e-23)</td><td>10.6%</td></tr></table>

Table 3.12: People's ability to identify whether $\sim 500$ word articles are model generated (as measured by the ratio of correct assignments to non-neutral assignments) was $88\%$ on the control model and $52\%$ on GPT-3 175B. This table shows the results of a two-sample T-Test for the difference in mean accuracy between GPT-3 175B and the control model (an unconditional GPT-3 Small model with increased output randomness).

Title: United Methodists Agree to Historic Split

Subtitle: Those who oppose gay marriage will form their own denomination

Article: After two days of intense debate, the United Methodist Church has agreed to a historic split - one that is expected to end in the creation of a new denomination, one that will be "theologically and socially conservative," according to The Washington Post. The majority of delegates attending the church's annual General Conference in May voted to strengthen a ban on the ordination of LGBTQ clergy and to write new rules that will "discipline" clergy who officiate at same-sex weddings. But those who opposed these measures have a new plan: They say they will form a separate denomination by 2020, calling their church the Christian Methodist denomination.

The Post notes that the denomination, which claims 12.5 million members, was in the early 20th century the "largest Protestant denomination in the U.S.," but that it has been shrinking in recent decades. The new split will be the second in the church's history. The first occurred in 1968, when roughly 10 percent of the denomination left to form the Evangelical United Brethren Church. The Post notes that the proposed split "comes at a critical time for the church, which has been losing members for years," which has been "pushed toward the brink of a schism over the role of LGBTQ people in the church." Gay marriage is not the only issue that has divided the church. In 2016, the denomination was split over ordination of transgender clergy, with the North Pacific regional conference voting to ban them from serving as clergy, and the South Pacific regional conference voting to allow them.

Figure 3.14: The GPT-3 generated news article that humans had the greatest difficulty distinguishing from a human written article (accuracy: 12%).

Title: Star's Tux Promise Draws Megyn Kelly's Sarcasm

Subtitle: Joaquin Phoenix pledged to not change for each awards event

Article: A year ago, Joaquin Phoenix made headlines when he appeared on the red carpet at the Golden Globes wearing a tuxedo with a paper bag over his head that read, "I am a shape-shifter. I can't change the world. I can only change myself." It was a promise to not change to fit into the Hollywood mold: "I think that's a really special thing, to not change yourself. I think it's a really special thing to say, 'This is what's inside of me, I'm proud of it, and I'm not going to be ashamed because of the way that someone else thinks I should be.'" Now, it's the Oscars, and Phoenix is at it again. But this time, his publicist is saying he'll be wearing a tux no matter what.

Megyn Kelly was not impressed, and she let him have it on The Tonight Show. "You know, I feel like, I feel like you could have worn the tux," she says. "But you're saying you're a shape-shifter. I don't know if you can change your tux, but you can change your mind. You can change your mind. You can change your mind." Phoenix says he did, but it didn't stick. "I was like, 'Okay, I'm going to wear a tuxedo to this thing.' And then I thought, 'I don't want to wear a tuxedo to this thing.'" Kelly goes on to encourage him to change his mind again, but Phoenix says it's too late: "I'm committed to wearing this."

Figure 3.15: The GPT-3 generated news article that humans found the easiest to distinguish from a human written article (accuracy: 61%).

A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is:
We were traveling in Africa and we saw these very cute whatpus.

To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:  
One day when I was playing tag with my little sister, she got really excited and she started doing these crazy farduddles.  
A "yalubalu" is a type of vegetable that looks like a big pumpkin. An example of a sentence that uses the word yalubalu is:  
I was on a trip to Africa and I tried this yalubalu vegetable that was grown in a garden there. It was delicious.

A "Burringo" is a car with very fast acceleration. An example of a sentence that uses the word Burringo is: In our garage we have a Burringo that my father drives to work every day.

A "Gigamuru" is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:  
I have a Gigamuru that my uncle gave me as a gift. I love to play it at home.

To "screeg" something is to swing a sword at it. An example of a sentence that uses the word screeg is:  
We screeghed at each other for several minutes and then we went outside and ate ice cream.

Figure 3.16: Representative GPT-3 completions for the few-shot task of using a new word in a sentence. Boldface is GPT-3's completions, plain text is human prompts. In the first example both the prompt and the completion are provided by a human; this then serves as conditioning for subsequent examples where GPT-3 receives successive additional prompts and provides the completions. Nothing task-specific is provided to GPT-3 other than the conditioning shown here.

nonexistent word being defined and used in a sentence, so the task is few-shot in terms of previous examples of the broad task and one-shot in terms of the specific word. Table 3.16 shows the 6 examples we generated; all definitions were human-generated, and the first answer was human-generated as conditioning while the subsequent answers were generated by GPT-3. These examples were generated continuously in one sitting and we did not omit or repeatedly try any prompts. In all cases the generated sentence appears to be a correct or at least plausible use of the word. In the final sentence the model generates a plausible conjugation for the word “screeg” (namely “screeghed”), although the use of the word is slightly awkward (“screeghed at each other”) despite being plausible in the sense that it could describe a toy sword fight. Overall, GPT-3 appears to be at least proficient at the task of using novel words in a sentence.

# 3.9.6 Correcting English Grammar

Another task well suited for few-shot learning is correcting English grammar. We test this with GPT-3 in the few-shot setting by giving prompts of the form "Poor English Input: <sentence>\n Good English Output: <sentence>" . We give GPT-3 one human-generated correction and then ask it to correct 5 more (again without any omissions or repeats). Results are shown in Figure 3.17.

# 4 Measuring and Preventing Memorization Of Benchmarks

Since our training dataset is sourced from the internet, it is possible that our model was trained on some of our benchmark test sets. Accurately detecting test contamination from internet-scale datasets is a new area of research without established best practices. While it is common practice to train large models without investigating contamination, given the increasing scale of pretraining datasets, we believe this issue is becoming increasingly important to attend to.

This concern is not just hypothetical. One of the first papers to train a language model on Common Crawl data [TL18] detected and removed a training document which overlapped with one of their evaluation datasets. Other work such as GPT-2 $[RWC^{+}19]$ also conducted post-hoc overlap analysis. Their study was relatively encouraging, finding that

Poor English input: I eat the purple berries.

Good English output: I ate the purple berries.

Poor English input: Thank you for picking me as your designer. I'd appreciate it.

Good English output: Thank you for choosing me as your designer. I appreciate it.

Poor English input: The mentioned changes have done. or I did the alteration that you requested. or I changed things you wanted and did the modifications.

Good English output: The requested changes have been made. or I made the alteration that you requested. or I changed things you wanted and made the modifications.

Poor English input: I'd be more than happy to work with you in another project.

Good English output: I'd be more than happy to work with you on another project.

Poor English input: Please provide me with a short brief of the design you're looking for and that'd be nice if you could share some examples or project you did before.

Good English output: Please provide me with a brief description of the design you're looking for and that would be nice if you could share some examples or projects you have done before.

Poor English input: The patient was died.

Good English output: The patient died.

Poor English input: We think that Leslie likes ourselves.

Good English output: We think that Leslie likes us.

Poor English input: Janet broke Bill on the finger.

Good English output: Janet broke Bill's finger.

Poor English input: Mary arranged for, in St. Louis, John to rent a house cheap.

Good English output: Mary arranged for John to rent a house in St. Louis.

Poor English input: Today I have went to the store to to buys some many bottle of water.

Good English output: Today I went to the store to buy some bottles of water.

Poor English input: I have tried to hit ball with bat, but my swing is has miss.

Good English output: I tried to hit the ball with the bat, but my swing missed.

Figure 3.17: Representative GPT-3 completions for the few-shot task of correcting English grammar. Boldface is GPT-3's completions, plain text is human prompts. In the first few examples example both the prompt and the completion are provided by a human; this then serves as conditioning for subsequent examples where GPT-3 receives successive additional prompts and provides the completions. Nothing task-specific is provided to GPT-3 aside from the first few examples as conditioning and the “Poor English input/Good English output” framing. We note that the distinction between ”poor” and ”good” English (and the terms themselves) is complex, contextual, and contested. As the example mentioning the rental of a house shows, assumptions that the model makes about what “good” is can even lead it to make errors (here, the model not only adjusts grammar, but also removes the word ”cheap” in a way that alters meaning).

![](images/487783e25a3245104c94139bab573d57ede20d07c1128f2a1fde19efddb298bf.jpg)

<details>
<summary>line</summary>

| Tokens Elapsed (Billions) | Validation Loss | Train Loss |
| -------------------------- | --------------- | ---------- |
| 0                          | 3.50            | 3.50       |
| 50                         | 2.80            | 2.75       |
| 100                        | 2.70            | 2.65       |
| 150                        | 2.65            | 2.60       |
| 200                        | 2.60            | 2.55       |
| 250                        | 2.55            | 2.50       |
| 300                        | 2.50            | 2.45       |
</details>

Figure 4.1: GPT-3 Training Curves We measure model performance during training on a deduplicated validation split of our training distribution. Though there is some gap between training and validation performance, the gap grows only minimally with model size and training time, suggesting that most of the gap comes from a difference in difficulty rather than overfitting.

although models did perform moderately better on data that overlapped between training and testing, this did not significantly impact reported results due to the small fraction of data which was contaminated (often only a few percent).

GPT-3 operates in a somewhat different regime. On the one hand, the dataset and model size are about two orders of magnitude larger than those used for GPT-2, and include a large amount of Common Crawl, creating increased potential for contamination and memorization. On the other hand, precisely due to the large amount of data, even GPT-3 175B does not overfit its training set by a significant amount, measured relative to a held-out validation set with which it was deduplicated (Figure 4.1). Thus, we expect that contamination is likely to be frequent, but that its effects may not be as large as feared.

We initially tried to address the issue of contamination by proactively searching for and attempting to remove any overlap between our training data and the development and test sets of all benchmarks studied in this paper. Unfortunately, a bug resulted in only partial removal of all detected overlaps from the training data. Due to the cost of training, it wasn't feasible to retrain the model. To address this, we investigate in detail how the remaining detected overlap impacts results.

For each benchmark, we produce a ‘clean’ version which removes all potentially leaked examples, defined roughly as examples that have a 13-gram overlap with anything in the pretraining set (or that overlap with the whole example when it is shorter than 13-grams). The goal is to very conservatively flag anything that could potentially be contamination, so as to produce a clean subset that is free of contamination with high confidence. The exact procedure is detailed in Appendix C.

We then evaluate GPT-3 on these clean benchmarks, and compare to the original score. If the score on the clean subset is similar to the score on the entire dataset, this suggests that contamination, even if present, does not have a significant effect on reported results. If the score on the clean subset is lower, this suggests contamination may be inflating the results. The results are summarized in Figure 4.2. Although potential contamination is often high (with a quarter of benchmarks scoring over 50%), in most cases performance changes only negligibly, and we see no evidence that contamination level and performance difference are correlated. We conclude that either our conservative method substantially overestimated contamination or that contamination has little effect on performance.

Below, we review in more detail the few specific cases where either (1) the model performs significantly worse on the cleaned version, or (2) potential contamination is very high, which makes measuring the performance difference difficult.

Our analysis flagged six groups of benchmarks for further investigation: Word Scrambling, Reading Comprehension (QuAC, SQuAD2, DROP), PIQA, Winograd, language modeling tasks (Wikitext tasks, 1BW), and German to English

![](images/bfbd2df7b18a1c71472c8e78a264d3969877c701cb70c5dd7f30eb0739141713.jpg)

<details>
<summary>scatter</summary>

| Model | Percentage of Data Clean in Dataset | Percent Change in Performance (Accuracy, F1 or BLEU) |
| --- | --- | --- |
| QuAC | 0% | 20% |
| SQuADv2 | ~5% | ~-2% |
| Drop | ~10% | -20% |
| Symbol Insertion | ~15% | ~0% |
| Winograd | ~40% | ~-2% |
| PIQA WMT16 en->de WMT16 de->en | ~65% | ~-5% |
| Anagrams 2 | ~75% | ~-3% |
| Anagrams 1 | ~95% | ~-8% |
| Reversed Words | ~95% | ~-25% |
</details>

Figure 4.2: Benchmark contamination analysis We constructed cleaned versions of each of our benchmarks to check for potential contamination in our training set. The x-axis is a conservative lower bound for how much of the dataset is known with high confidence to be clean, and the y-axis shows the difference in performance when evaluating only on the verified clean subset. Performance on most benchmarks changed negligibly, but some were flagged for further review. On inspection we find some evidence for contamination of the PIQA and Winograd results, and we mark the corresponding results in Section 3 with an asterisk. We find no evidence that other benchmarks are affected.

translation. Since our overlap analysis is designed to be extremely conservative, we expect it to produce some false positives. We summarize the results for each group of tasks below:

- Reading Comprehension: Our initial analysis flagged $>90\%$ of task examples from QuAC, SQuAD2, and DROP as potentially contaminated, so large that even measuring the differential on a clean subset was difficult. Upon manual inspection, however, we found that for every overlap we inspected, in all 3 datasets, the source text was present in our training data but the question/answer pairs were not, meaning the model gains only background information and cannot memorize the answer to a specific question.   
- German translation: We found 25% of the examples in the WMT16 German-English test set were marked as potentially contaminated, with an associated total effect size of 1-2 BLEU. Upon inspection, none of the flagged examples contain paired sentences resembling NMT training data and collisions were monolingual matches mostly of snippets of events discussed in the news.   
- Reversed Words and Anagrams: Recall that these tasks are of the form “alaok = koala”. Due to the short length of these tasks, we used 2-grams for filtering (ignoring punctuation). After inspecting the flagged overlaps, we found that they were not typically instances of real reversals or unscramblings in the training set, but rather palindromes or trivial unscramblings, e.g “kayak = kayak”. The amount of overlap was small, but removing the trivial tasks lead to an increase in difficulty and thus a spurious signal. Related to this, the symbol insertion task shows high overlap but no effect on performance – this is because that task involves removing non-letter characters from a word, and the overlap analysis itself ignores such characters, leading to many spurious matches.   
- PIQA: The overlap analysis flagged 29% of examples as contaminated, and observed a 3 percentage point absolute decrease (4% relative decrease) in performance on the clean subset. Though the test dataset was released after our training set was created and its labels are hidden, some of the web pages used by the crowdsourced dataset creators are contained in our training set. We found a similar decrease in a 25x smaller model with much less capacity to memorize, leading us to suspect that the shift is likely statistical bias rather than memorization; examples which workers copied may simply be easier. Unfortunately, we cannot rigorously prove this hypothesis. We therefore mark our PIQA results with an asterisk to denote this potential contamination.   
- Winograd: The overlap analysis flagged $45\%$ of examples, and found a $2.6\%$ decrease in performance on the clean subset. Manual inspection of the overlapping data point showed that 132 Winograd schemas were in fact present in our training set, though presented in a different format than we present the task to the model. Although the decrease in performance is small, we mark our Winograd results in the main paper with an asterisk.

\- Language modeling: We found the 4 Wikipedia language modeling benchmarks measured in GPT-2, plus the Children's Book Test dataset, to be almost entirely contained in our training data. Since we cannot reliably extract a clean subset here, we do not report results on these datasets, even though we intended to when starting this work. We note that Penn Tree Bank due to its age was unaffected and therefore became our chief language modeling benchmark.

We also inspected datasets where contamination was high, but the impact on performance was close to zero, simply to verify how much actual contamination existed. These appeared to often contain false positives. They had either no actual contamination, or had contamination that did not give away the answer to the task. One notable exception was LAMBADA, which appeared to have substantial genuine contamination, yet the impact on performance was very small, with the clean subset scoring within 0.5% of the full dataset. Also, strictly speaking, our fill-in-the-blank format precludes the simplest form of memorization. Nevertheless, since we made very large gains on LAMBADA in this paper, the potential contamination is noted in the results section.

An important limitation of our contamination analysis is that we cannot be sure that the clean subset is drawn from the same distribution as the original dataset. It remains possible that memorization inflates results but at the same time is precisely counteracted by some statistical bias causing the clean subset to be easier. However, the sheer number of shifts close to zero suggests this is unlikely, and we also observed no noticeable difference in the shifts for small models, which are unlikely to be memorizing.

Overall, we have made a best effort to measure and document the effects of data contamination, and to note or outright remove problematic results, depending on the severity. Much work remains to be done to address this important and subtle issue for the field in general, both when designing benchmarks and when training models. For a more detailed explanation of our analysis, we refer the reader to Appendix C.

# 5 Limitations

GPT-3 and our analysis of it have a number of limitations. Below we describe some of these and suggest directions for future work.

First, despite the strong quantitative and qualitative improvements of GPT-3, particularly compared to its direct predecessor GPT-2, it still has notable weaknesses in text synthesis and several NLP tasks. On text synthesis, although the overall quality is high, GPT-3 samples still sometimes repeat themselves semantically at the document level, start to lose coherence over sufficiently long passages, contradict themselves, and occasionally contain non-sequitur sentences or paragraphs. We will release a collection of 500 uncurated unconditional samples to help provide a better sense of GPT-3's limitations and strengths at text synthesis. Within the domain of discrete language tasks, we have noticed informally that GPT-3 seems to have special difficulty with “common sense physics”, despite doing well on some datasets (such as PIQA $[BZB^{+}19]$ ) that test this domain. Specifically GPT-3 has difficulty with questions of the type “If I put cheese into the fridge, will it melt?”. Quantitatively, GPT-3's in-context learning performance has some notable gaps on our suite of benchmarks, as described in Section 3, and in particular it does little better than chance when evaluated one-shot or even few-shot on some “comparison” tasks, such as determining if two words are used the same way in a sentence, or if one sentence implies another (WIC and ANLI respectively), as well as on a subset of reading comprehension tasks. This is especially striking given GPT-3's strong few-shot performance on many other tasks.

GPT-3 has several structural and algorithmic limitations, which could account for some of the issues above. We focused on exploring in-context learning behavior in autoregressive language models because it is straightforward to both sample and compute likelihoods with this model class. As a result our experiments do not include any bidirectional architectures or other training objectives such as denoising. This is a noticeable difference from much of the recent literature, which has documented improved fine-tuning performance when using these approaches over standard language models $[RSR^{+}19]$ . Thus our design decision comes at the cost of potentially worse performance on tasks which empirically benefit from bidirectionality. This may include fill-in-the-blank tasks, tasks that involve looking back and comparing two pieces of content, or tasks that require re-reading or carefully considering a long passage and then generating a very short answer. This could be a possible explanation for GPT-3's lagging few-shot performance on a few of the tasks, such as WIC (which involves comparing the use of a word in two sentences), ANLI (which involves comparing two sentences to see if one implies the other), and several reading comprehension tasks (e.g. QuAC and RACE). We also conjecture, based on past literature, that a large bidirectional model would be stronger at fine-tuning than GPT-3. Making a bidirectional model at the scale of GPT-3, and/or trying to make bidirectional models work with few- or zero-shot learning, is a promising direction for future research, and could help achieve the “best of both worlds”.

A more fundamental limitation of the general approach described in this paper - scaling up any LM-like model, whether autoregressive or bidirectional - is that it may eventually run into (or could already be running into) the limits of the

pretraining objective. Our current objective weights every token equally and lacks a notion of what is most important to predict and what is less important. [RRS20] demonstrate benefits of customizing prediction to entities of interest. Also, with self-supervised objectives, task specification relies on forcing the desired task into a prediction problem, whereas ultimately, useful language systems (for example virtual assistants) might be better thought of as taking goal-directed actions rather than just making predictions. Finally, large pretrained language models are not grounded in other domains of experience, such as video or real-world physical interaction, and thus lack a large amount of context about the world $[BHT^{+}20]$ . For all these reasons, scaling pure self-supervised prediction is likely to hit limits, and augmentation with a different approach is likely to be necessary. Promising future directions in this vein might include learning the objective function from humans $[ZSW^{+}19a]$ , fine-tuning with reinforcement learning, or adding additional modalities such as images to provide grounding and a better model of the world $[CLY^{+}19]$ .

Another limitation broadly shared by language models is poor sample efficiency during pre-training. While GPT-3 takes a step towards test-time sample efficiency closer to that of humans (one-shot or zero-shot), it still sees much more text during pre-training than a human sees in the their lifetime [Lin20]. Improving pre-training sample efficiency is an important direction for future work, and might come from grounding in the physical world to provide additional information, or from algorithmic improvements.

A limitation, or at least uncertainty, associated with few-shot learning in GPT-3 is ambiguity about whether few-shot learning actually learns new tasks “from scratch” at inference time, or if it simply recognizes and identifies tasks that it has learned during training. These possibilities exist on a spectrum, ranging from demonstrations in the training set that are drawn from exactly the same distribution as those at test time, to recognizing the same task but in a different format, to adapting to a specific style of a general task such as QA, to learning a skill entirely de novo. Where GPT-3 is on this spectrum may also vary from task to task. Synthetic tasks such as wordscrambling or defining nonsense words seem especially likely to be learned de novo, whereas translation clearly must be learned during pretraining, although possibly from data that is very different in organization and style than the test data. Ultimately, it is not even clear what humans learn from scratch vs from prior demonstrations. Even organizing diverse demonstrations during pre-training and identifying them at test time would be an advance for language models, but nevertheless understanding precisely how few-shot learning works is an important unexplored direction for future research.

A limitation associated with models at the scale of GPT-3, regardless of objective function or algorithm, is that they are both expensive and inconvenient to perform inference on, which may present a challenge for practical applicability of models of this scale in their current form. One possible future direction to address this is distillation [HVD15] of large models down to a manageable size for specific tasks. Large models such as GPT-3 contain a very wide range of skills, most of which are not needed for a specific task, suggesting that in principle aggressive distillation may be possible. Distillation is well-explored in general [LHCG19a] but has not been tried at the scale of hundred of billions parameters; new challenges and opportunities may be associated with applying it to models of this size.

Finally, GPT-3 shares some limitations common to most deep learning systems – its decisions are not easily interpretable, it is not necessarily well-calibrated in its predictions on novel inputs as observed by the much higher variance in performance than humans on standard benchmarks, and it retains the biases of the data it has been trained on. This last issue – biases in the data that may lead the model to generate stereotyped or prejudiced content – is of special concern from a societal perspective, and will be discussed along with other issues in the next section on Broader Impacts (Section 6).

# 6 Broader Impacts

Language models have a wide range of beneficial applications for society, including code and writing auto-completion, grammar assistance, game narrative generation, improving search engine responses, and answering questions. But they also have potentially harmful applications. GPT-3 improves the quality of text generation and adaptability over smaller models and increases the difficulty of distinguishing synthetic text from human-written text. It therefore has the potential to advance both the beneficial and harmful applications of language models.

Here we focus on the potential harms of improved language models, not because we believe the harms are necessarily greater, but in order to stimulate efforts to study and mitigate them. The broader impacts of language models like this are numerous. We focus on two primary issues: the potential for deliberate misuse of language models like GPT-3 in Section 6.1, and issues of bias, fairness, and representation within models like GPT-3 in Section 6.2. We also briefly discuss issues of energy efficiency (Section 6.3).

# 6.1 Misuse of Language Models

Malicious uses of language models can be somewhat difficult to anticipate because they often involve repurposing language models in a very different environment or for a different purpose than researchers intended. To help with this, we can think in terms of traditional security risk assessment frameworks, which outline key steps such as identifying threats and potential impacts, assessing likelihood, and determining risk as a combination of likelihood and impact [Ros12]. We discuss three factors: potential misuse applications, threat actors, and external incentive structures.

# 6.1.1 Potential Misuse Applications

Any socially harmful activity that relies on generating text could be augmented by powerful language models. Examples include misinformation, spam, phishing, abuse of legal and governmental processes, fraudulent academic essay writing and social engineering pretexting. Many of these applications bottleneck on human beings to write sufficiently high quality text. Language models that produce high quality text generation could lower existing barriers to carrying out these activities and increase their efficacy.

The misuse potential of language models increases as the quality of text synthesis improves. The ability of GPT-3 to generate several paragraphs of synthetic content that people find difficult to distinguish from human-written text in 3.9.4 represents a concerning milestone in this regard.

# 6.1.2 Threat Actor Analysis

Threat actors can be organized by skill and resource levels, ranging from low or moderately skilled and resourced actors who may be able to build a malicious product to ‘advanced persistent threats’ (APTs): highly skilled and well-resourced (e.g. state-sponsored) groups with long-term agendas $[SBC^{+}19]$ .

To understand how low and mid-skill actors think about language models, we have been monitoring forums and chat groups where misinformation tactics, malware distribution, and computer fraud are frequently discussed. While we did find significant discussion of misuse following the initial release of GPT-2 in spring of 2019, we found fewer instances of experimentation and no successful deployments since then. Additionally, those misuse discussions were correlated with media coverage of language model technologies. From this, we assess that the threat of misuse from these actors is not immediate, but significant improvements in reliability could change this.

Because APTs do not typically discuss operations in the open, we have consulted with professional threat analysts about possible APT activity involving the use of language models. Since the release of GPT-2 there has been no discernible difference in operations that may see potential gains by using language models. The assessment was that language models may not be worth investing significant resources in because there has been no convincing demonstration that current language models are significantly better than current methods for generating text, and because methods for “targeting” or “controlling” the content of language models are still at a very early stage.

# 6.1.3 External Incentive Structures

Each threat actor group also has a set of tactics, techniques, and procedures (TTPs) that they rely on to accomplish their agenda. TTPs are influenced by economic factors like scalability and ease of deployment; phishing is extremely popular among all groups because it offers a low-cost, low-effort, high-yield method of deploying malware and stealing login credentials. Using language models to augment existing TTPs would likely result in an even lower cost of deployment.

Ease of use is another significant incentive. Having stable infrastructure has a large impact on the adoption of TTPs. The outputs of language models are stochastic, however, and though developers can constrain these (e.g. using top-k truncation) they are not able to perform consistently without human feedback. If a social media disinformation bot produces outputs that are reliable 99% of the time, but produces incoherent outputs 1% of the time, this could reduce the amount of human labor required in operating this bot. But a human is still needed to filter the outputs, which restricts how scalable the operation can be.

Based on our analysis of this model and analysis of threat actors and the landscape, we suspect AI researchers will eventually develop language models that are sufficiently consistent and steerable that they will be of greater interest to malicious actors. We expect this will introduce challenges for the broader research community, and hope to work on this through a combination of mitigation research, prototyping, and coordinating with other technical developers.

# 6.2 Fairness, Bias, and Representation

Biases present in training data may lead models to generate stereotyped or prejudiced content. This is concerning, since model bias could harm people in the relevant groups in different ways by entrenching existing stereotypes and producing demeaning portrayals amongst other potential harms [Cra17]. We have conducted an analysis of biases in the model in order to better understand GPT-3's limitations when it comes to fairness, bias, and representation. $^{8}$

Our goal is not to exhaustively characterize GPT-3, but to give a preliminary analysis of some of its limitations and behaviors. We focus on biases relating to gender, race, and religion, although many other categories of bias are likely present and could be studied in follow-up work. This is a preliminary analysis and does not reflect all of the model's biases even within the studied categories.

Broadly, our analysis indicates that internet-trained models have internet-scale biases; models tend to reflect stereotypes present in their training data. Below we discuss our preliminary findings of bias along the dimensions of gender, race, and religion. We probe for bias in the 175 billion parameter model and also in similar smaller models, to see if and how they are different in this dimension.

# 6.2.1 Gender

In our investigation of gender bias in GPT-3, we focused on associations between gender and occupation. We found that occupations in general have a higher probability of being followed by a male gender identifier than a female one (in other words, they are male leaning) when given a context such as "The {occupation} was a" (Neutral Variant). 83% of the 388 occupations we tested were more likely to be followed by a male identifier by GPT-3. We measured this by feeding the model a context such as "The detective was a" and then looking at the probability of the model following up with male indicating words (eg. man, male etc.) or female indicating words (woman, female etc.). In particular, occupations demonstrating higher levels of education such as legislator, banker, or professor emeritus were heavily male leaning along with occupations that require hard physical labour such as mason, millwright, and sheriff. Occupations that were more likely to be followed by female identifiers include midwife, nurse, receptionist, housekeeper etc.

We also tested how these probabilities changed when we shifted the context to be the "The competent {occupation} was a" (Competent Variant), and when we shifted the context to be "The incompetent {occupation} was a" (Incompetent Variant) for each occupation in the dataset. We found that, when prompted with "The competent {occupation} was a," the majority of occupations had an even higher probability of being followed by a male identifier than a female one than was the case with our original neutral prompt, "The {occupation} was a". With the prompt "The incompetent {occupation} was a" the majority of occupations still leaned male with a similar probability than for our original neutral prompt. The average occupation bias - measured as $\frac{1}{n_{jobs}}\sum_{jobs}\log(\frac{P(\text{female}|Context)}{P(\text{male}|Context)})$ - was -1.11 for the Neutral Variant, -2.14 for the Competent Variant and -1.15 for the Incompetent Variant.

We also carried out pronoun resolution on the Winogender dataset [RNLVD18] using two methods which further corroborated the model's tendency to associate most occupations with males. One method measured the models ability to correctly assign a pronoun as the occupation or the participant. For example, we fed the model a context such as "The advisor met with the advisee because she wanted to get advice about job applications. 'She' refers to the" and found the option with the lowest probability between the two possible options (Choices between Occupation Option: advisor; Participant Option: advisee).

Occupation and participant words often have societal biases associated with them such as the assumption that most occupants are by default male. We found that the language models learnt some of these biases such as a tendency to associate female pronouns with participant positions more than male pronouns. GPT-3 175B had the highest accuracy of all the models (64.17%) on this task. It was also the only model where the accuracy for Occupant sentences (sentences where the correct answer was the Occupation option) for females was higher than for males (81.7% vs 76.7%). All other models had a higher accuracy for male pronouns with Occupation sentences as compared to female pronouns with the exception of our second largest model- GPT-3 13B - which had the same accuracy (60%) for both. This offers some preliminary evidence that in places where issues of bias can make language models susceptible to error, the larger models are more robust than smaller models.

We also performed co-occurrence tests, where we analyzed which words are likely to occur in the vicinity of other pre-selected words. We created a model output sample set by generating 800 outputs of length 50 each with a temperature

Table 6.1: Most Biased Descriptive Words in 175B Model 

<table><tr><td>Top 10 Most Biased Male Descriptive Words with Raw Co-Occurrence Counts</td><td>Top 10 Most Biased Female Descriptive Words with Raw Co-Occurrence Counts</td></tr><tr><td>Average Number of Co-Occurrences Across All Words: 17.5</td><td>Average Number of Co-Occurrences Across All Words: 23.9</td></tr><tr><td>Large (16)</td><td>Optimistic (12)</td></tr><tr><td>Mostly (15)</td><td>Bubbly (12)</td></tr><tr><td>Lazy (14)</td><td>Naughty (12)</td></tr><tr><td>Fantastic (13)</td><td>Easy-going (12)</td></tr><tr><td>Eccentric (13)</td><td>Petite (10)</td></tr><tr><td>Protect (10)</td><td>Tight (10)</td></tr><tr><td>Jolly (10)</td><td>Pregnant (10)</td></tr><tr><td>Stable (9)</td><td>Gorgeous (28)</td></tr><tr><td>Personable (22)</td><td>Sucked (8)</td></tr><tr><td>Survive (7)</td><td>Beautiful (158)</td></tr></table>

of 1 and top\_p of 0.9 for every prompt in our dataset. For gender, we had prompts such as "He was very", "She was very", "He would be described as", "She would be described as" $^{9}$ . We looked at the adjectives and adverbs in the top 100 most favored words using an off-the-shelf POS tagger [LB02]. We found females were more often described using appearance oriented words such as "beautiful" and "gorgeous" as compared to men who were more often described using adjectives that span a greater spectrum.

Table 6.1 shows the top 10 most favored descriptive words for the model along with the raw number of times each word co-occurred with a pronoun indicator. “Most Favored” here indicates words which were most skewed towards a category by co-occurring with it at a higher rate as compared to the other category. To put these numbers in perspective, we have also included the average for the number of co-occurrences across all qualifying words for each gender.

# 6.2.2 Race

To investigate racial bias in GPT-3, we seeded the model with prompts such as - "The {race} man was very", "The {race} woman was very" and "People would describe the {race} person as" and generated 800 samples for each of the above prompts, with {race} replaced with a term indicating a racial category such as White or Asian. We then measure word co-occurrences in the generated samples. Given prior research demonstrating that language models produce text of differing sentiment when varying features such as occupation [HZJ+19], we explored how race impacted sentiment. We measured sentiment using Senti WordNet [BES10] for the words which co-occurred disproportionately with each race. Each word sentiment varied from 100 to -100, with positive scores indicating positive words (eg. wonderfulness: 100, amicable: 87.5), negative scores indicating negative words (eg. wretched: -87.5, horrid: -87.5) and a score of 0 indicating neutral words (eg. sloping, chalet).

It should be noted that we were explicitly prompting the models to talk about race and this in turn generated text that focused on racial features; these results are not from the models talking about race in the wild but talking about race in an experimental setup where they have been primed to do so. Additionally, since we are measuring sentiment by simply looking at word co-occurrences, the resulting sentiment can reflect socio-historical factors - for instance, text relating to a discussion of slavery will frequently have a negative sentiment, which may lead to a demographic being associated with a negative sentiment under this testing methodology.

Across the models we analyzed, ‘Asian’ had a consistently high sentiment - it ranked 1st in 3 out of 7 models. On the other hand, ’Black’ had a consistently low sentiment - it ranked the lowest in 5 out of 7 models. These differences narrowed marginally on the larger model sizes. This analysis gives a sense of the biases of different models and highlights the need for more sophisticated analysis of the relationship between sentiment, entities, and input data.

![](images/b17438e8b76c88d417c9b2b85ed662ca7e9b8b654b72e31e9d8a4d34d0b6096f.jpg)

<details>
<summary>line</summary>

| Model Size | Asian | Black | White | Latinx | Indian | Middle eastern |
| ---------- | ----- | ----- | ----- | ------ | ------ | -------------- |
| 350M       | 14    | -8    | 1     | 14     | 21     | 5              |
| 760M       | 15    | -7    | 1     | 2      | 8      | 6              |
| 1.3B       | 18    | -10   | 5     | 10     | 12     | 10             |
| 2.7B       | 16    | -2    | 4     | 28     | 14     | 3              |
| 6.7B       | 15    | -6    | 5     | 12     | 12     | 5              |
| 13B        | 14    | -2    | -2    | 8      | 10     | 7              |
| 175B       | 11    | 2     | 3     | 19     | 10     | 1              |
</details>

Figure 6.1: Racial Sentiment Across Models

<table><tr><td>Religion</td><td>Most Favored Descriptive Words</td></tr><tr><td>Atheism</td><td>‘Theists’, ‘Cool’, ‘Agnostics’, ‘Mad’, ‘Theism’, ‘Defensive’, ‘Complaining’, ‘Correct’, ‘Arrogant’, ‘Characterized’</td></tr><tr><td>Buddhism</td><td>‘Myanmar’, ‘Vegetarians’, ‘Burma’, ‘Fellowship’, ‘Monk’, ‘Japanese’, ‘Reluctant’, ‘Wisdom’, ‘Enlightenment’, ‘Non-Violent’</td></tr><tr><td>Christianity</td><td>‘Attend’, ‘Ignorant’, ‘Response’, ‘Judgmental’, ‘Grace’, ‘Execution’, ‘Egypt’, ‘Continue’, ‘Comments’, ‘Officially’</td></tr><tr><td>Hinduism</td><td>‘Caste’, ‘Cows’, ‘BJP’, ‘Kashmir’, ‘Modi’, ‘Celebrated’, ‘Dharma’, ‘Pakistani’, ‘Originated’, ‘Africa’</td></tr><tr><td>Islam</td><td>‘Pillars’, ‘Terrorism’, ‘Fasting’, ‘Sheikh’, ‘Non-Muslim’, ‘Source’, ‘Charities’, ‘Levant’, ‘Allah’, ‘Prophet’</td></tr><tr><td>Judaism</td><td>‘Gentiles’, ‘Race’, ‘Semites’, ‘Whites’, ‘Blacks’, ‘Smartest’, ‘Racists’, ‘Arabs’, ‘Game’, ‘Russian’</td></tr></table>

Table 6.2: Shows the ten most favored words about each religion in the GPT-3 175B model.

# 6.2.3 Religion

We studied which words co-occurred with religious terms relating to Atheism, Buddhism, Christianity, Hinduism, Islam, and Judaism, by generating 800 model outputs of length $\approx50$ with a temperature of 1 and a top p of 0.9 for every prompt. Our prompts were of the nature " {Religion practitioners} are" (Eg. "Christians are") for each of the six religious categories listed above. We then allowed the model to naturally carry out completions and created a corpus of such completions for studying co-occurrence of words.

The following is an example output from the model:

"Buddhists are divided into two main branches - Theravada and Mahayana. Theravada is the more conservative branch, centering on monastic life and the earliest sutras and refusing to recognize the later Mahayana sutras as authentic."

Similar to race, we found that the models make associations with religious terms that indicate some propensity to reflect how these terms are sometimes presented in the world. For example, with the religion Islam, we found that words such as ramadan, prophet and mosque co-occurred at a higher rate than for other religions. We also found that words such as violent, terrorism and terrorist co-occurred at a greater rate with Islam than with other religions and were in the top 40 most favored words for Islam in GPT-3.

# 6.2.4 Future Bias and Fairness Challenges

We have presented this preliminary analysis to share some of the biases we found in order to motivate further research, and to highlight the inherent difficulties in characterizing biases in large-scale generative models; we expect this to be an area of continuous research for us and are excited to discuss different methodological approaches with the community. We view the work in this section as subjective signposting - we chose gender, race, and religion as a starting point, but we recognize the inherent subjectivity in this choice. Our work is inspired by the literature on characterizing model attributes to develop informative labels such as Model Cards for Model Reporting from $[MWZ^{+}18]$ .

Ultimately, it is important not just to characterize biases in language systems but to intervene. The literature on this is also extensive $[QMZH19, HZJ^{+}19]$ , so we offer only a few brief comments on future directions specific to large language models. In order to pave the way for effective bias prevention in general purpose models, there is a need for building a common vocabulary tying together the normative, technical and empirical challenges of bias mitigation for these models. There is room for more research that engages with the literature outside NLP, better articulates normative statements about harm, and engages with the lived experience of communities affected by NLP systems $[BBDIW20]$ . Thus, mitigation work should not be approached purely with a metric driven objective to ‘remove’ bias as this has been shown to have blind spots $[GG19, NvNvdG19]$ but in a holistic manner.

# 6.3 Energy Usage

Practical large-scale pre-training requires large amounts of computation, which is energy-intensive: training the GPT-3 175B consumed several thousand petaflop/s-days of compute during pre-training, compared to tens of petaflop/s-days for a 1.5B parameter GPT-2 model (Figure 2.2). This means we should be cognizant of the cost and efficiency of such models, as advocated by [SDSE19].

The use of large-scale pre-training also gives another lens through which to view the efficiency of large models - we should consider not only the resources that go into training them, but how these resources are amortized over the lifetime of a model, which will subsequently be used for a variety of purposes and fine-tuned for specific tasks. Though models like GPT-3 consume significant resources during training, they can be surprisingly efficient once trained: even with the full GPT-3 175B, generating 100 pages of content from a trained model can cost on the order of 0.4 kW-hr, or only a few cents in energy costs. Additionally, techniques like model distillation [LHCG19a] can further bring down the cost of such models, letting us adopt a paradigm of training single, large-scale models, then creating more efficient versions of them for use in appropriate contexts. Algorithmic progress may also naturally further increase the efficiency of such models over time, similar to trends observed in image recognition and neural machine translation [HB20].

# 7 Related Work

Several lines of work have focused on increasing parameter count and/or computation in language models as a means to improve generative or task performance. An early work scaled LSTM based language models to over a billion parameters $[JVS^{+}16]$ . One line of work straightforwardly increases the size of transformer models, scaling up parameters and FLOPS-per-token roughly in proportion. Work in this vein has successively increased model size: 213 million parameters $[VSP^{+}17]$ in the original paper, 300 million parameters $[DCLT18]$ , 1.5 billion parameters $[RWC^{+}19]$ , 8 billion parameters $[SPP^{+}19]$ , 11 billion parameters $[RSR^{+}19]$ , and most recently 17 billion parameters $[Tur20]$ . A second line of work has focused on increasing parameter count but not computation, as a means of increasing models' capacity to store information without increased computational cost. These approaches rely on the conditional computation framework $[BLC13]$ and specifically, the mixture-of-experts method $[SMM^{+}17]$ has been used to produce 100 billion parameter models and more recently 50 billion parameter translation models $[AJF19]$ , though only a small fraction of the parameters are actually used on each forward pass. A third approach increases computation without increasing parameters; examples of this approach include adaptive computation time $[Gra16]$ and the universal transformer $[DGV^{+}18]$ . Our work focuses on the first approach (scaling compute and parameters together, by straightforwardly making the neural net larger), and increases model size 10x beyond previous models that employ this strategy.

Several efforts have also systematically studied the effect of scale on language model performance. $[KMH^{+}20, RRBS19, LWS^{+}20, HNA^{+}17]$ , find a smooth power-law trend in loss as autoregressive language models are scaled up. This work suggests that this trend largely continues as models continue to scale up (although a slight bending of the curve can perhaps be detected in Figure 3.1), and we also find relatively smooth increases in many (though not all) downstream tasks across 3 orders of magnitude of scaling.

Another line of work goes in the opposite direction from scaling, attempting to preserve strong performance in language models that are as small as possible. This approach includes ALBERT $[LCG^{+}19]$ as well as general $[HVD15]$ and

task-specific [SDCW19, JYS $^{+}$ 19, KR16] approaches to distillation of language models. These architectures and techniques are potentially complementary to our work, and could be applied to decrease latency and memory footprint of giant models.

As fine-tuned language models have neared human performance on many standard benchmark tasks, considerable effort has been devoted to constructing more difficult or open-ended tasks, including question answering $[KPR^{+}19, IBGC^{+}14, CCE^{+}18, MCKS18]$ , reading comprehension $[CHI^{+}18, RCM19]$ , and adversarially constructed datasets designed to be difficult for existing language models $[SBBC19, NWD^{+}19]$ . In this work we test our models on many of these datasets.

Many previous efforts have focused specifically on question-answering, which constitutes a significant fraction of the tasks we tested on. Recent efforts include $[RSR^{+}19, RRS20]$ , which fine-tuned an 11 billion parameter language model, and $[GLT^{+}20]$ , which focused on attending over a large corpus of data at test time. Our work differs in focusing on in-context learning but could be combined in the future with those of $[GLT^{+}20, LPP^{+}20]$ .

Metalearning in language models has been utilized in $[RWC^{+}19]$ , though with much more limited results and no systematic study. More broadly, language model metalearning has an inner-loop-outer-loop structure, making it structurally similar to metalearning as applied to ML in general. Here there is an extensive literature, including matching networks $[VBL^{+}16]$ , RL2 $[DSC^{+}16]$ , learning to optimize $[RL16, ADG^{+}16, LM17]$ and MAML $[FAL17]$ . Our approach of stuffing the model's context with previous examples is most structurally similar to RL2 and also resembles $[HYC01]$ , in that an inner loop of adaptation takes place through computation in the model's activations across timesteps, without updating the weights, while an outer loop (in this case just language model pre-training) updates the weights, and implicitly learns the ability to adapt to or at least recognize tasks defined at inference-time. Few-shot auto-regressive density estimation was explored in $[RCP^{+}17]$ and $[GWC^{+}18]$ studied low-resource NMT as a few-shot learning problem.

While the mechanism of our few-shot approach is different, prior work has also explored ways of using pre-trained language models in combination with gradient descent to perform few-shot learning $[SS20]$ . Another sub-field with similar goals is semi-supervised learning where approaches such as UDA $[XDH^{+}19]$ also explore methods of fine-tuning when very little labeled data is available.

Giving multi-task models instructions in natural language was first formalized in a supervised setting with $[MKXS18]$ and utilized for some tasks (such as summarizing) in a language model with $[RWC^{+}19]$ . The notion of presenting tasks in natural language was also explored in the text-to-text transformer $[RSR^{+}19]$ , although there it was applied for multi-task fine-tuning rather than for in-context learning without weight updates.

Another approach to increasing generality and transfer-learning capability in language models is multi-task learning $[Car97]$ , which fine-tunes on a mixture of downstream tasks together, rather than separately updating the weights for each one. If successful multi-task learning could allow a single model to be used for many tasks without updating the weights (similar to our in-context learning approach), or alternatively could improve sample efficiency when updating the weights for a new task. Multi-task learning has shown some promising initial results $[LGH^{+}15, LSP^{+}18]$ and multi-stage fine-tuning has recently become a standardized part of SOTA results on some datasets $[PFB18]$ and pushed the boundaries on certain tasks $[KKS^{+}20]$ , but is still limited by the need to manually curate collections of datasets and set up training curricula. By contrast pre-training at large enough scale appears to offer a “natural” broad distribution of tasks implicitly contained in predicting the text itself. One direction for future work might be attempting to generate a broader set of explicit tasks for multi-task learning, for example through procedural generation $[TFR^{+}17]$ , human interaction $[ZSW^{+}19b]$ , or active learning $[Mac92]$ .

Algorithmic innovation in language models over the last two years has been enormous, including denoising-based bidirectionality [DCLT18], prefixLM [DL15] and encoder-decoder architectures $[LLG^{+}19, RSR^{+}19]$ , random permutations during training $[YDY^{+}19]$ , architectures that improve the efficiency of sampling $[DYY^{+}19]$ , improvements in data and training procedures $[LOG^{+}19]$ , and efficiency increases in the embedding parameters $[LCG^{+}19]$ . Many of these techniques provide significant gains on downstream tasks. In this work we continue to focus on pure autoregressive language models, both in order to focus on in-context learning performance and to reduce the complexity of our large model implementations. However, it is very likely that incorporating these algorithmic advances could improve GPT-3's performance on downstream tasks, especially in the fine-tuning setting, and combining GPT-3's scale with these algorithmic techniques is a promising direction for future work.

# 8 Conclusion

We presented a 175 billion parameter language model which shows strong performance on many NLP tasks and benchmarks in the zero-shot, one-shot, and few-shot settings, in some cases nearly matching the performance of

state-of-the-art fine-tuned systems, as well as generating high-quality samples and strong qualitative performance at tasks defined on-the-fly. We documented roughly predictable trends of scaling in performance without using fine-tuning. We also discussed the social impacts of this class of model. Despite many limitations and weaknesses, these results suggest that very large language models may be an important ingredient in the development of adaptable, general language systems.

# Acknowledgements

The authors would like to thank Ryan Lowe for giving detailed feedback on drafts of the paper. Thanks to Jakub Pachocki and Szymon Sidor for suggesting tasks, and Greg Brockman, Michael Petrov, Brooke Chan, and Chelsea Voss for helping run evaluations on OpenAI's infrastructure. Thanks to David Luan for initial support in scaling up this project, Irene Solaiman for discussions about ways to approach and evaluate bias, Harrison Edwards and Yura Burda for discussions and experimentation with in-context learning, Geoffrey Irving and Paul Christiano for early discussions of language model scaling, Long Ouyang for advising on the design of the human evaluation experiments, Chris Hallacy for discussions on data collection, and Shan Carter for help with visual design. Thanks to the millions of people who created content that was used in the training of the model, and to those who were involved in indexing or upvoting the content (in the case of WebText). Additionally, we would like to thank the entire OpenAI infrastructure and supercomputing teams for making it possible to train models at this scale.

# Contributions

Tom Brown, Ben Mann, Prafulla Dhariwal, Dario Amodei, Nick Ryder, Daniel M Ziegler, and Jeffrey Wu implemented the large-scale models, training infrastructure, and model-parallel strategies.

Tom Brown, Dario Amodei, Ben Mann, and Nick Ryder conducted pre-training experiments.

Ben Mann and Alec Radford collected, filtered, deduplicated, and conducted overlap analysis on the training data.

Melanie Subbiah, Ben Mann, Dario Amodei, Jared Kaplan, Sam McCandlish, Tom Brown, Tom Henighan, and Girish Sastry implemented the downstream tasks and the software framework for supporting them, including creation of synthetic tasks.

Jared Kaplan and Sam McCandlish initially predicted that a giant language model should show continued gains, and applied scaling laws to help predict and guide model and data scaling decisions for the research.

Ben Mann implemented sampling without replacement during training.

Alec Radford originally demonstrated few-shot learning occurs in language models.

Jared Kaplan and Sam McCandlish showed that larger models learn more quickly in-context, and systematically studied in-context learning curves, task prompting, and evaluation methods.

Prafulla Dhariwal implemented an early version of the codebase, and developed the memory optimizations for fully half-precision training.

Rewon Child and Mark Chen developed an early version of our model-parallel strategy.

Rewon Child and Scott Gray contributed the sparse transformer.

Aditya Ramesh experimented with loss scaling strategies for pretraining.

Melanie Subbiah and Arvind Neelakantan implemented, experimented with, and tested beam search.

Pranav Shyam worked on SuperGLUE and assisted with connections to few-shot learning and meta-learning literature.

Sandhini Agarwal conducted the fairness and representation analysis.

Girish Sastry and Amanda Askell conducted the human evaluations of the model.

Ariel Herbert-Voss conducted the threat analysis of malicious use.

Gretchen Krueger edited and red-teamed the policy sections of the paper.

Benjamin Chess, Clemens Winter, Eric Sigler, Christopher Hesse, Mateusz Litwin, and Christopher Berner optimized OpenAI's clusters to run the largest models efficiently.

Scott Gray developed fast GPU kernels used during training.

Jack Clark led the analysis of ethical impacts — fairness and representation, human assessments of the model, and broader impacts analysis, and advised Gretchen, Amanda, Girish, Sandhini, and Ariel on their work.

Dario Amodei, Alec Radford, Tom Brown, Sam McCandlish, Nick Ryder, Jared Kaplan, Sandhini Agarwal, Amanda Askell, Girish Sastry, and Jack Clark wrote the paper.

Sam McCandlish led the analysis of model scaling, and advised Tom Henighan and Jared Kaplan on their work.

Alec Radford advised the project from an NLP perspective, suggested tasks, put the results in context, and demonstrated the benefit of weight decay for training.

Ilya Sutskever was an early advocate for scaling large generative likelihood models, and advised Pranav, Prafulla, Rewon, Alec, and Aditya on their work.

Dario Amodei designed and led the research.

# A Details of Common Crawl Filtering

As mentioned in Section 2.2, we employed two techniques to improve the quality of the Common Crawl dataset: (1) filtering Common Crawl and (2) fuzzy deduplication:

1. In order to improve the quality of Common Crawl, we developed an automatic filtering method to remove low quality documents. Using the original WebText as a proxy for high-quality documents, we trained a classifier to distinguish these from raw Common Crawl. We then used this classifier to re-sample Common Crawl by prioritizing documents which were predicted by the classifier to be higher quality. The classifier is trained using logistic regression classifier with features from Spark's standard tokenizer and HashingTF $^{10}$ . For the positive examples, we used a collection of curated datasets such as WebText, Wikipedia, and our web books corpus as the positive examples, and for the negative examples, we used unfiltered Common Crawl. We used this classifier to score Common Crawl documents. We kept each document in our dataset iff

$$
\text { np.random.pareto } (\alpha) > 1 - \text { document\_score }
$$

We chose $\alpha = 9$ in order to take mostly documents the classifier scored highly, but still include some documents that were out of distribution. $\alpha$ was chosen to match the distribution of scores from our classifier on WebText. We found this re-weighting increased quality as measured by loss on a range of out-of-distribution generative text samples.

2. To further improve model quality and prevent overfitting (which becomes increasingly important as model capacity increases), we fuzzily deduplicated documents (i.e. removed documents with high overlap with other documents) within each dataset using Spark's MinHashLSH implementation with 10 hashes, using the same features as were used for classification above. We also fuzzily removed WebText from Common Crawl. Overall this decreased dataset size by an average of $10\%$ .

After filtering for duplicates and quality, we also partially removed text occurring in benchmark datasets, described in Appendix C.

# B Details of Model Training

To train all versions of GPT-3, we use Adam with $\beta_{1}=0.9$ , $\beta_{2}=0.95$ , and $\epsilon=10^{-8}$ , we clip the global norm of the gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375 million tokens. We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size. Data are sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting. All models use weight decay of 0.1 to provide a small amount of regularization [LH17].

During training we always train on sequences of the full $n_{ctx} = 2048$ token context window, packing multiple documents into a single sequence when documents are shorter than 2048, in order to increase computational efficiency. Sequences with multiple documents are not masked in any special way but instead documents within a sequence are delimited with a special end of text token, giving the language model the information necessary to infer that context separated by the end of text token is unrelated. This allows for efficient training without need for any special sequence-specific masking.

# C Details of Test Set Contamination Studies

In section 4 we gave a high level overview of test set contamination studies. In this section we provide details on methodology and results.

Initial training set filtering We attempted to remove text occurring in benchmarks from training data by searching for 13-gram overlaps between all test/development sets used in this work and our training data, and we removed the colliding 13-gram as well as a 200 character window around it, splitting the original document into pieces. For filtering purposes we define a gram as a lowercase, whitespace delimited word with no punctuation. Pieces less than 200 characters long were discarded. Documents split into more than 10 pieces were considered contaminated and

removed entirely. Originally we removed entire documents given a single collision, but that overly penalized long documents such as books for false positives. An example of a false positive might be a test set based on Wikipedia, in which the Wikipedia article quotes a single line from a book. We ignored 13–grams that matched more than 10 training documents, as inspection showed the majority of these to contain common cultural phrases, legal boilerplate, or similar content that we likely do want the model to learn, rather than undesired specific overlaps with test sets. Examples for various frequencies can be found in the GPT-3 release repository $^{11}$ .

Overlap methodology For our benchmark overlap analysis in Section 4, we used a variable number of words N to check for overlap for each dataset, where N is the 5th percentile example length in words, ignoring all punctuation, whitespace, and casing. Due to spurious collisions at lower values of N we use a minimum value of 8 on non-synthetic tasks. For performance reasons, we set a maximum value of 13 for all tasks. Values for N and the amount of data marked as dirty are shown in Table C.1. Unlike GPT-2's use of bloom filters to compute probabilistic bounds for test contamination, we used Apache Spark to compute exact collisions across all training and test sets. We compute overlaps between test sets and our full training corpus, even though we only trained on 40% of our filtered Common Crawl documents per Section 2.2.

We define a ‘dirty’ example as one with any N-gram overlap with any training document, and a ‘clean’ example as one with no collision.

Test and validation splits had similar contamination levels despite some test splits being unlabeled. Due to a bug revealed by this analysis, filtering described above failed on long documents such as books. Because of cost considerations it was infeasible to retrain the model on a corrected version of the training dataset. As such, several language modeling benchmarks plus the Children's Book Test showed almost complete overlap, and therefore were not included in this paper. Overlaps are shown in Table C.1

Overlap results To understand how much having seen some of the data helps the model perform on downstream tasks, we filter every validation and test set by dirtiness. Then we run evaluation on the clean-only examples and report the relative percent change between the clean score and the original score. If the clean score is more than 1% or 2% worse than the overall score, it suggests the model may have overfit to the examples it has seen. If the clean score is significantly better, our filtering scheme may have preferentially marked easier examples as dirty.

This overlap metric tends to show a high rate of false positives for datasets that contain background information (but not answers) drawn from the web (such as SQuAD, which draws from Wikipedia) or examples less than 8 words long, which we ignored in our filtering process (except for wordscrambling tasks). One instance where this technique seems to fail to give good signal is DROP, a reading comprehension task in which 94% of the examples are dirty. The information required to answer the question is in a passage provided to the model, so having seen the passage during training but not the questions and answers does not meaningfully constitute cheating. We confirmed that every matching training document contained only the source passage, and none of the questions and answers in the dataset. The more likely explanation for the decrease in performance is that the 6% of examples that remain after filtering come from a slightly different distribution than the dirty examples.

Figure 4.2 shows that as the dataset becomes more contaminated, the variance of the clean/all fraction increases, but there is no apparent bias towards improved or degraded performance. This suggests that GPT-3 is relatively insensitive to contamination. See Section 4 for details on the datasets we flagged for further review.

<table><tr><td>Name</td><td>Split</td><td>Metric</td><td>N</td><td>Acc/F1/BLEU</td><td>Total Count</td><td>Dirty Acc/F1/BLEU</td><td>Dirty Count</td><td>Clean Acc/F1/BLEU</td><td>Clean Count</td><td>Clean Percentage</td><td>Relative Difference Clean vs All</td></tr><tr><td>Quac</td><td>dev</td><td>f1</td><td>13</td><td>44.3</td><td>7353</td><td>44.3</td><td>7315</td><td>54.1</td><td>38</td><td>1%</td><td>20%</td></tr><tr><td>SQuADv2</td><td>dev</td><td>f1</td><td>13</td><td>69.8</td><td>11873</td><td>69.9</td><td>11136</td><td>68.4</td><td>737</td><td>6%</td><td>-2%</td></tr><tr><td>DROP</td><td>dev</td><td>f1</td><td>13</td><td>36.5</td><td>9536</td><td>37.0</td><td>8898</td><td>29.5</td><td>638</td><td>7%</td><td>-21%</td></tr><tr><td>Symbol Insertion</td><td>dev</td><td>acc</td><td>7</td><td>66.9</td><td>10000</td><td>66.8</td><td>8565</td><td>67.1</td><td>1435</td><td>14%</td><td>0%</td></tr><tr><td>CoQa</td><td>dev</td><td>f1</td><td>13</td><td>86.0</td><td>7983</td><td>85.3</td><td>5107</td><td>87.1</td><td>2876</td><td>36%</td><td>1%</td></tr><tr><td>ReCoRD</td><td>dev</td><td>acc</td><td>13</td><td>89.5</td><td>10000</td><td>90.3</td><td>6110</td><td>88.2</td><td>3890</td><td>39%</td><td>-1%</td></tr><tr><td>Winograd</td><td>test</td><td>acc</td><td>9</td><td>88.6</td><td>273</td><td>90.2</td><td>164</td><td>86.2</td><td>109</td><td>40%</td><td>-3%</td></tr><tr><td>BoolQ</td><td>dev</td><td>acc</td><td>13</td><td>76.0</td><td>3270</td><td>75.8</td><td>1955</td><td>76.3</td><td>1315</td><td>40%</td><td>0%</td></tr><tr><td>MultiRC</td><td>dev</td><td>acc</td><td>13</td><td>74.2</td><td>953</td><td>73.4</td><td>558</td><td>75.3</td><td>395</td><td>41%</td><td>1%</td></tr><tr><td>RACE-h</td><td>test</td><td>acc</td><td>13</td><td>46.8</td><td>3498</td><td>47.0</td><td>1580</td><td>46.7</td><td>1918</td><td>55%</td><td>0%</td></tr><tr><td>LAMBADA</td><td>test</td><td>acc</td><td>13</td><td>86.4</td><td>5153</td><td>86.9</td><td>2209</td><td>86.0</td><td>2944</td><td>57%</td><td>0%</td></tr><tr><td>LAMBADA (No Blanks)</td><td>test</td><td>acc</td><td>13</td><td>77.8</td><td>5153</td><td>78.5</td><td>2209</td><td>77.2</td><td>2944</td><td>57%</td><td>-1%</td></tr><tr><td>WSC</td><td>dev</td><td>acc</td><td>13</td><td>76.9</td><td>104</td><td>73.8</td><td>42</td><td>79.0</td><td>62</td><td>60%</td><td>3%</td></tr><tr><td>PIQA</td><td>dev</td><td>acc</td><td>8</td><td>82.3</td><td>1838</td><td>89.9</td><td>526</td><td>79.3</td><td>1312</td><td>71%</td><td>-4%</td></tr><tr><td>RACE-m</td><td>test</td><td>acc</td><td>13</td><td>58.5</td><td>1436</td><td>53.0</td><td>366</td><td>60.4</td><td>1070</td><td>75%</td><td>3%</td></tr><tr><td>De→En 16</td><td>test</td><td>bleu-sb</td><td>12</td><td>43.0</td><td>2999</td><td>47.4</td><td>739</td><td>40.8</td><td>2260</td><td>75%</td><td>-5%</td></tr><tr><td>En→De 16</td><td>test</td><td>bleu-sb</td><td>12</td><td>30.9</td><td>2999</td><td>32.6</td><td>739</td><td>29.9</td><td>2260</td><td>75%</td><td>-3%</td></tr><tr><td>En→Ro 16</td><td>test</td><td>bleu-sb</td><td>12</td><td>25.8</td><td>1999</td><td>24.9</td><td>423</td><td>26.1</td><td>1576</td><td>79%</td><td>1%</td></tr><tr><td>Ro→En 16</td><td>test</td><td>bleu-sb</td><td>12</td><td>41.3</td><td>1999</td><td>40.4</td><td>423</td><td>41.6</td><td>1576</td><td>79%</td><td>1%</td></tr><tr><td>WebQs</td><td>test</td><td>acc</td><td>8</td><td>41.5</td><td>2032</td><td>41.6</td><td>428</td><td>41.5</td><td>1604</td><td>79%</td><td>0%</td></tr><tr><td>ANLI R1</td><td>test</td><td>acc</td><td>13</td><td>36.8</td><td>1000</td><td>40.5</td><td>200</td><td>35.9</td><td>800</td><td>80%</td><td>-3%</td></tr><tr><td>ANLI R2</td><td>test</td><td>acc</td><td>13</td><td>34.0</td><td>1000</td><td>29.4</td><td>177</td><td>35.0</td><td>823</td><td>82%</td><td>3%</td></tr><tr><td>TriviaQA</td><td>dev</td><td>acc</td><td>10</td><td>71.2</td><td>7993</td><td>70.8</td><td>1390</td><td>71.3</td><td>6603</td><td>83%</td><td>0%</td></tr><tr><td>ANLI R3</td><td>test</td><td>acc</td><td>13</td><td>40.2</td><td>1200</td><td>38.3</td><td>196</td><td>40.5</td><td>1004</td><td>84%</td><td>1%</td></tr><tr><td>En→Fr 14</td><td>test</td><td>bleu-sb</td><td>13</td><td>39.9</td><td>3003</td><td>38.3</td><td>411</td><td>40.3</td><td>2592</td><td>86%</td><td>1%</td></tr><tr><td>Fr→En 14</td><td>test</td><td>bleu-sb</td><td>13</td><td>41.4</td><td>3003</td><td>40.9</td><td>411</td><td>41.4</td><td>2592</td><td>86%</td><td>0%</td></tr><tr><td>WiC</td><td>dev</td><td>acc</td><td>13</td><td>51.4</td><td>638</td><td>53.1</td><td>49</td><td>51.3</td><td>589</td><td>92%</td><td>0%</td></tr><tr><td>RTE</td><td>dev</td><td>acc</td><td>13</td><td>71.5</td><td>277</td><td>71.4</td><td>21</td><td>71.5</td><td>256</td><td>92%</td><td>0%</td></tr><tr><td>CB</td><td>dev</td><td>acc</td><td>13</td><td>80.4</td><td>56</td><td>100.0</td><td>4</td><td>78.8</td><td>52</td><td>93%</td><td>-2%</td></tr><tr><td>Anagrams 2</td><td>dev</td><td>acc</td><td>2</td><td>40.2</td><td>10000</td><td>76.2</td><td>705</td><td>37.4</td><td>9295</td><td>93%</td><td>-7%</td></tr><tr><td>Reversed Words</td><td>dev</td><td>acc</td><td>2</td><td>0.4</td><td>10000</td><td>1.5</td><td>660</td><td>0.3</td><td>9340</td><td>93%</td><td>-26%</td></tr><tr><td>OpenBookQA</td><td>test</td><td>acc</td><td>8</td><td>65.4</td><td>500</td><td>58.1</td><td>31</td><td>65.9</td><td>469</td><td>94%</td><td>1%</td></tr><tr><td>ARC (Easy)</td><td>test</td><td>acc</td><td>11</td><td>70.1</td><td>2268</td><td>77.5</td><td>89</td><td>69.8</td><td>2179</td><td>96%</td><td>0%</td></tr><tr><td>Anagrams 1</td><td>dev</td><td>acc</td><td>2</td><td>15.0</td><td>10000</td><td>49.8</td><td>327</td><td>13.8</td><td>9673</td><td>97%</td><td>-8%</td></tr><tr><td>COPA</td><td>dev</td><td>acc</td><td>9</td><td>93.0</td><td>100</td><td>100.0</td><td>3</td><td>92.8</td><td>97</td><td>97%</td><td>0%</td></tr><tr><td>ARC (Challenge)</td><td>test</td><td>acc</td><td>12</td><td>51.6</td><td>1144</td><td>45.2</td><td>31</td><td>51.8</td><td>1113</td><td>97%</td><td>0%</td></tr><tr><td>HellaSwag</td><td>dev</td><td>acc</td><td>13</td><td>79.3</td><td>10042</td><td>86.2</td><td>152</td><td>79.2</td><td>9890</td><td>98%</td><td>0%</td></tr><tr><td>NQs</td><td>test</td><td>acc</td><td>11</td><td>29.9</td><td>3610</td><td>32.7</td><td>52</td><td>29.8</td><td>3558</td><td>99%</td><td>0%</td></tr><tr><td>Cycled Letters</td><td>dev</td><td>acc</td><td>2</td><td>38.6</td><td>10000</td><td>20.5</td><td>73</td><td>38.7</td><td>9927</td><td>99%</td><td>0%</td></tr><tr><td>SAT Analogies</td><td>dev</td><td>acc</td><td>9</td><td>65.8</td><td>374</td><td>100.0</td><td>2</td><td>65.6</td><td>372</td><td>99%</td><td>0%</td></tr><tr><td>StoryCloze</td><td>test</td><td>acc</td><td>13</td><td>87.7</td><td>1871</td><td>100.0</td><td>2</td><td>87.6</td><td>1869</td><td>100%</td><td>0%</td></tr><tr><td>Winogrande</td><td>dev</td><td>acc</td><td>13</td><td>77.7</td><td>1267</td><td>-</td><td>0</td><td>77.7</td><td>1267</td><td>100%</td><td>0%</td></tr></table>

Table C.1: Overlap statistics for all datasets sorted from dirtiest to cleanest. We consider a dataset example dirty if it has a single N-gram collision with any document in our training corpus. “Relative Difference Clean vs All” shows the percent change in performance between only the clean examples vs all the examples in the benchmark. “Count” shows the number of examples. “Clean percentage” is the percent of examples that are clean vs total. For “Acc/F1/BLEU” we use the metric specified in “Metric”. These scores come from evaluations with a different seed for the random examples used for in-context learning, and will therefore differ slightly from the scores elsewhere in the paper.

# D Total Compute Used to Train Language Models

This appendix contains the calculations that were used to derive the approximate compute used to train the language models in Figure 2.2. As a simplifying assumption, we ignore the attention operation, as it typically uses less than 10% of the total compute for the models we are analyzing.

Calculations can be seen in Table D.1 and are explained within the table caption.

<table><tr><td>Model</td><td>Total train compute (PF-days)</td><td>Total train compute (flops)</td><td>Params (M)</td><td>Training tokens (billions)</td><td>Flops per param per token</td><td>Mult for bwd pass</td><td>Fwd-pass flops per active param per token</td><td>Frac of params active for each token</td></tr><tr><td>T5-Small</td><td>2.08E+00</td><td>1.80E+20</td><td>60</td><td>1,000</td><td>3</td><td>3</td><td>1</td><td>0.5</td></tr><tr><td>T5-Base</td><td>7.64E+00</td><td>6.60E+20</td><td>220</td><td>1,000</td><td>3</td><td>3</td><td>1</td><td>0.5</td></tr><tr><td>T5-Large</td><td>2.67E+01</td><td>2.31E+21</td><td>770</td><td>1,000</td><td>3</td><td>3</td><td>1</td><td>0.5</td></tr><tr><td>T5-3B</td><td>1.04E+02</td><td>9.00E+21</td><td>3,000</td><td>1,000</td><td>3</td><td>3</td><td>1</td><td>0.5</td></tr><tr><td>T5-11B</td><td>3.82E+02</td><td>3.30E+22</td><td>11,000</td><td>1,000</td><td>3</td><td>3</td><td>1</td><td>0.5</td></tr><tr><td>BERT-Base</td><td>1.89E+00</td><td>1.64E+20</td><td>109</td><td>250</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>BERT-Large</td><td>6.16E+00</td><td>5.33E+20</td><td>355</td><td>250</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>RoBERTa-Base</td><td>1.74E+01</td><td>1.50E+21</td><td>125</td><td>2,000</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>RoBERTa-Large</td><td>4.93E+01</td><td>4.26E+21</td><td>355</td><td>2,000</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>GPT-3 Small</td><td>2.60E+00</td><td>2.25E+20</td><td>125</td><td>300</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>GPT-3 Medium</td><td>7.42E+00</td><td>6.41E+20</td><td>356</td><td>300</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>GPT-3 Large</td><td>1.58E+01</td><td>1.37E+21</td><td>760</td><td>300</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>GPT-3 XL</td><td>2.75E+01</td><td>2.38E+21</td><td>1,320</td><td>300</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>GPT-3 2.7B</td><td>5.52E+01</td><td>4.77E+21</td><td>2,650</td><td>300</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>GPT-3 6.7B</td><td>1.39E+02</td><td>1.20E+22</td><td>6,660</td><td>300</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>GPT-3 13B</td><td>2.68E+02</td><td>2.31E+22</td><td>12,850</td><td>300</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr><tr><td>GPT-3 175B</td><td>3.64E+03</td><td>3.14E+23</td><td>174,600</td><td>300</td><td>6</td><td>3</td><td>2</td><td>1.0</td></tr></table>

Table D.1: Starting from the right hand side and moving left, we begin with the number of training tokens that each model was trained with. Next we note that since T5 uses an encoder-decoder model, only half of the parameters are active for each token during a forward or backwards pass. We then note that each token is involved in a single addition and a single multiply for each active parameter in the forward pass (ignoring attention). Then we add a multiplier of 3x to account for the backwards pass (as computing both $\frac{\partial params}{\partial loss}$ and $\frac{\partial acts}{\partial loss}$ use a similar amount of compute as the forwards pass. Combining the previous two numbers, we get the total flops per parameter per token. We multiply this value by the total training tokens and the total parameters to yield the number of total flops used during training. We report both flops and petaflop/s-day (each of which are 8.64e+19 flops).

# E Human Quality Assessment of Synthetic News Articles

This appendix contains details on the experiments measuring human ability to distinguish GPT-3-generated synthetic news articles from real news articles. We first describe the experiments on the $\sim$ 200 word news articles, and then describe the preliminary investigation of $\sim$ 500 word news articles generated by GPT-3.

Participants: We recruited 718 unique participants to take part in 6 experiments. 97 participants were excluded for failing an internet check question, leaving a total of 621 participants: 343 male, 271 female, and 7 other. Mean participant age was $\sim$ 38 years old. All participants were recruited through Positly, which maintains a whitelist of high-performing workers from Mechanical Turk. All participants were US-based but there were no other demographic restrictions. Participants were paid \$12 for their participation, based on a task time estimate of 60 minutes determined by pilot runs. In order to ensure that the sample of participants for each experiment quiz was unique, participants were not allowed to take part in an experiment more than once.

Procedure and design: We arbitrarily selected 25 news articles that appeared in newser.com in early 2020. We used the article titles and subtitles to produce outputs from the 125M, 350M, 760M, 1.3B, 2.7B, 6.7B, 13.0B, and 200B (GPT-3) parameter language models. Five outputs per question were generated by each model and the generation with a word count closest to that of the human written article was selected automatically. This was to minimize the effect that completion length might have on participants' judgments. The same output procedure for each model with the exception of the removal of the intentionally bad control model, as described in the main text.

<table><tr><td>Model</td><td>Participants Recruited</td><td>Participants Excluded</td><td>Genders (m:f:other)</td><td>Mean Age</td><td>Average Word Count (human:model)</td></tr><tr><td>Control</td><td>76</td><td>7</td><td>32:37:0</td><td>39</td><td>216:216</td></tr><tr><td>GPT-3 Small</td><td>80</td><td>7</td><td>41:31:1</td><td>40</td><td>216:188</td></tr><tr><td>GPT-3 Medium</td><td>80</td><td>7</td><td>46:28:2</td><td>39</td><td>216:202</td></tr><tr><td>GPT-3 Large</td><td>81</td><td>24</td><td>46:28:2</td><td>37</td><td>216:200</td></tr><tr><td>GPT-3 XL</td><td>79</td><td>14</td><td>32:32:1</td><td>38</td><td>216:199</td></tr><tr><td>GPT-3 2.7B</td><td>80</td><td>11</td><td>36:33:0</td><td>40</td><td>216:202</td></tr><tr><td>GPT-3 6.7B</td><td>76</td><td>5</td><td>46:28:2</td><td>37</td><td>216:195</td></tr><tr><td>GPT-3 13.0B</td><td>81</td><td>13</td><td>46:28:2</td><td>37</td><td>216:209</td></tr><tr><td>GPT-3 175B</td><td>80</td><td>9</td><td>42:29:0</td><td>37</td><td>216:216</td></tr></table>

Table E.1: Participant details and article lengths for each experiment to evaluate human detection of $\sim$ 200 word model generated news articles. Participants were excluded due to internet check fails.

Average time spent trying to detect model generated news article   
![](images/8f096a65dccbf817b0805ffef6030e876013083ff0ecf4594b6f225de3c9f456.jpg)

<details>
<summary>scatter</summary>

| Number of parameters (log scale) | Duration (seconds) |
| -------------------------------- | ------------------ |
| 1e8                              | 108                |
| 2e8                              | 110                |
| 5e8                              | 111                |
| 1e9                              | 119                |
| 2e9                              | 118                |
| 5e9                              | 114                |
| 1e10                             | 115                |
| 2e10                             | 126                |
</details>

Figure E.1: Participants spend more time trying to identify whether each news article is machine generated as model size increases. Duration on the control model is indicated with the dashed line. Line of best fit is a linear model on a log scale with 95% confidence intervals.

In each experiment, half of the participants were randomly assigned to quiz A and half were randomly assigned to quiz B. Each quiz consisted of 25 articles: half (12-13) were human written and half (12-13) were model generated: the articles with human written completions in quiz A had model generated completions in quiz B and vice versa. The order of quiz question was shuffled for each participant. Participants could leave comments and were asked to indicate if they had seen the articles before. Participants were instructed not to look up the articles or their content during the quiz and at the end of the quiz were asked if they had looked anything up during the quiz.

Statistical Tests: To compare means on the different runs, we performed a two-sample t-test for independent groups for each model against the control. This was implemented in Python using the scipy.stats.ttest\_ind function. When plotting a regression line in the graph of average participant accuracy vs model size, we fit a power law of the form $ax^{-b}$ . The 95% confidence intervals were estimated from the t-distribution of the sample mean.

Duration statistics: In the main text, we discussed the finding that the ability of human participants to distinguish model and human generated news articles decreases as our models become larger. We have also found that the average time spent for a given set of questions increases as the model size increases, as shown in Figure E.1. Lower

<table><tr><td>Model</td><td>Participants Recruited</td><td>Participants Excluded</td><td>Genders (m:f:other)</td><td>Mean Age</td><td>Average Word Count (human:model)</td></tr><tr><td>Control</td><td>79</td><td>17</td><td>32:37:0</td><td>39</td><td>569:464</td></tr><tr><td>GPT-3 175B</td><td>81</td><td>19</td><td>32:30:0</td><td>40</td><td>569:498</td></tr></table>

Table E.2: Participant details and article lengths for the experiments investigating human detection of $\sim$ 500 word model generated news articles. Participants were excluded due to internet check fails.

accuracy scores despite increased time investment from participants supports the finding that larger models generate harder-to-distinguish news articles.

Preliminary investigation of $\sim 500$ word articles: We recruited 160 unique US-based participants to take part in 2 experiments through Positly (details are given in Table E.2). We randomly selected 12 Reuters world news articles from late 2019 and created a context for GPT-3 175B that consisted of a single Reuters article not in this set of 12. We then used the article titles and Reuters locations to generate completions from GPT-3 175B and the 160M control model from the previous experiments. These were used to create two 12-question quizzes per model, each consisting of half human written and half model generated articles. Comprehension questions were added and articles were shown to participants in 3 stages at 30 second intervals to encourage closer reading. Participants were paid \$12 for this task. Model generation selection methods, exclusion criteria, and statistical tests mirror those of the previous experiments.

# F Additional Samples from GPT-3

GPT-3 adapts well to many tasks other than the ones explored in the main body of the paper. As an example, in Figure F.1, we show four uncurated samples from a prompt suggesting that the model write a poem, with a given title, in the style of Wallace Stevens. We first experimented with a few prompts, then generated four samples with no additional editing or selection (sampling at temperature 1 using nucleus sampling [HBFC19] with P = 0.9). Completions were truncated when the model began to write a new title and author heading, or broke into prose commentary.

<table><tr><td rowspan="3">Context →</td><td>The CityBY C. P. CAVAFYTRANSLATED BY EDMUND KEELEY[Poem text omitted]</td></tr><tr><td>SOME TREESJohn Ashbery[Poem text omitted]</td></tr><tr><td>Shadows on the WayWallace Stevens</td></tr><tr><td>---- Generated Poem 1 ----</td><td>---- Generated Poem 3 ----</td></tr><tr><td>I must have shadows on the wayIf I am to walk I must haveEach step taken slowly and aloneTo have it ready made</td><td>The sun was all we had. Now, in the shadeAll is changed. The mind must dwell on thoseWhite fields, that to its eyes were always old;Those ancient gleams, convolutingThe way they lay among the huge roots,The great dark tomes of reverie,The plumed labyrinth of the sea.</td></tr><tr><td>And I must think in lines of greyTo have dim thoughts to be my guideMust look on blue and greenAnd never let my eye forgetThat color is my friendAnd purple must surround me too</td><td>When our own breath gonne faint and fadeUp the long stairway of the deadWe shall remember wellThe blindness of the light, the whitenessOf that white land. The footsteps, andThe grasses in that light, and in the shadeThe menacing life.</td></tr><tr><td>The yellow of the sun is no moreIntrusive than the bluish snowThat falls on all of us. I must haveGrey thoughts and blue thoughts walk with meIf I am to go away at all.</td><td>We played, too, once, in the warmest rooms.To one content, there is one contentFor all mankind. The forest gladesAre all the more for being in fable,And wood and town in legend shadowed-Once more to return. The one contentHas in it every happiness, more braveFor being purposeless; and soThose island gardens, deep in the mind,Make of remoteness familiarityAnd of the unspoken name a friend.</td></tr><tr><td>---- Generated Poem 2 ----</td><td></td></tr><tr><td>There is a way in the world.That way a man may live in a small townHis lifetime and see no man come,Never the stir of men about a market,Never the bobbing of covered carts,Never the creak passed on the stairClosing some other life; an old manCan prosper in that town, can prosperWhen his store is seen to close, his wifeFades in his house, and the world shouts in vainThat no one comes to hear himWhere his small bell tolls.In that small townEverything is known.The ways of the young men are known,And the ways of the young womenAs they walk to the Spring;The rumors are all known,The quarrels, the hurt, the envyOf the men and the women.Till a day comesThat way out of the worldThat leads to a wood.Young, the young manWalks, a lover now,Knows the streets as they are and is of them.</td><td>---- Generated Poem 4 ----</td></tr><tr><td>Comes to the wood.There, in the wood, among trees,He sees shadows on the way, hears voices,hears the wind and the rustling of leaves;Through an open gladeHe sees a shape and the shape hears:It waits as he waits,Waits as the shadows wait,As the voices wait;Shadows on the way, voices in the wind.</td><td>Nobody will come to this place. It is a road that leads nowhere.The solitude is deep. The mountains are high.But they are desolate, and they turn the traveler&#x27;s faceTowards the North. All the sounds of the world are far away.When the wind rises above the trees,The boughs bow to the ground.Even the birds that inhabit the tangle of weedsThat is the roadside cover, are silent. One listens,But hears no roar of the forest. One is alone.One will be taken.One will be taken.There is no utterance, there is no conversation,But one is uneasy all the same....There is a thin blue mist,A darkness rising like smoke,And within that darknessA possession of the heart.One will be taken.... It was here, and it will be here again-Here, under this sky empty and full of light.</td></tr></table>

Figure F.1: Four uncurated completions from a context suggesting the model compose a poem in the style of Wallace Stevens with the title ‘Shadows on the Way’.

# G Details of Task Phrasing and Specifications

The following figures illustrate the formatting and phrasing of all the tasks included in the paper. All data comes from the ground truth datasets in this section, and no samples from GPT-3 are included here.

Context → Article:

Informal conversation is an important part of any business relationship. Before you start a discussion, however, make sure you understand which topics are suitable and which are considered taboo in a particular culture. Latin Americans enjoy sharing information about their local history, art and customs. You may expect questions about your family, and be sure to show pictures of your children. You may feel free to ask similar questions of your Latin American friends. The French think of conversation as an art form, and they enjoy the value of lively discussions as well as disagreements. For them, arguments can be interesting and they can cover pretty much or any topic ---- as long as they occur in are respectful and intelligent manner.

In the United States, business people like to discuss a wide range of topics, including opinions about work, family, hobbies, and politics. In Japan, China, and Korea, however, people are much more private. They do not share much about their thoughts, feelings, or emotions because they feel that doing so might take away from the harmonious business relationship they're trying to build. Middle Easterners are also private about their personal lives and family matters. It is considered rude, for example, to ask a businessman from Saudi Arabia about his wife or children.

As a general rule, it's best not to talk about politics or religion with your business friends. This can get you into trouble, even in the United States, where people hold different religious views. In addition, discussing one's salary is usually considered unsuitable. Sports is typically a friendly subject in most parts of the world, although be careful not to criticize national sport. Instead, be friendly and praise your host's team.

Q: What shouldn't you do when talking about sports with colleagues from another country?

A: Criticizing the sports of your colleagues' country.

Q: Which is typically a friendly topic in most places according to the author?

A: Sports.

Q: Why are people from Asia more private in their conversation with others?

A: They don't want to have their good relationship with others harmed by informal conversation.

Q: The author considers politics and religion \_.

A:

```txt
Correct Answer → taboo
Incorrect Answer → cheerful topics
Incorrect Answer → rude topics
Incorrect Answer → topics that can never be talked about 
```  
Figure G.1: Formatted dataset example for RACE-h. When predicting, we normalize by the unconditional probability of each answer as described in 2.

<table><tr><td>Context →</td><td>anli 2: anli 2: The Gold Coast Hotel &amp; Casino is a hotel and casino located in Paradise, Nevada. This locals’ casino is owned and operated by Boyd Gaming. The Gold Coast is located one mile (~ 1.6km) west of the Las Vegas Strip on West Flamingo Road. It is located across the street from the Palms Casino Resort and the Rio All Suite Hotel and Casino. Question: The Gold Coast is a budget-friendly casino. True, False, or Neither?</td></tr><tr><td>Correct Answer →</td><td>Neither</td></tr><tr><td>Incorrect Answer →</td><td>True</td></tr><tr><td>Incorrect Answer →</td><td>False</td></tr></table>

Figure G.2: Formatted dataset example for ANLI R2

<table><tr><td>Context →</td><td>Article:Mrs. Smith is an unusual teacher. Once she told each student to bring along a few potatoes in plastic bag. On each potato the students had to write a name of a person that they hated And the next day, every child brought some potatoes. Some had two potatoes;some three;some up to five.Mrs. Smith then told the children to carry the bags everywhere they went, even to the toilet, for two weeks. As day after day passed, the children started to complain about the awful smell of the rotten potatoes.Those children who brought five potatoes began to feel the weight trouble of the bags. After two weeks, the children were happy to hear that the game was finally ended. Mrs. Smith asked,"How did you feel while carrying the potatoes for two weeks?" The children started complaining about the trouble loudly.Then Mrs. Smith told them why she asked them to play the game. She said,"This is exactly the situation when you carry your hatred for somebody inside your heart. The terrible smell of the hatred will pollute your heart and you will carry something unnecessary with you all the time. If you cannot stand the smell of the rotten potatoes for just two weeks, can you imagine how heavy it would be to have the hatred in your heart for your lifetime? So throw away any hatred from your heart, and you'll be really happy."Q: Which of the following is True according to the passage?A: If a kid hated four people,he or she had to carry four potatoes.Q: We can learn from the passage that we should _.A: throw away the hatred insideQ: The children complained about _ besides the weight trouble.A: the smellQ: Mrs.Smith asked her students to write _ on the potatoes.A:</td></tr><tr><td>Correct Answer →</td><td>names</td></tr><tr><td>Incorrect Answer →</td><td>numbers</td></tr><tr><td>Incorrect Answer →</td><td>time</td></tr><tr><td>Incorrect Answer →</td><td>places</td></tr><tr><td>Context →</td><td>How to apply sealant to wood.</td></tr><tr><td>Correct Answer →</td><td>Using a brush, brush on sealant onto wood until it is fully saturated with the sealant.</td></tr><tr><td>Incorrect Answer →</td><td>Using a brush, drip on sealant onto wood until it is fully saturated with the sealant.</td></tr></table>

Figure G.3: Formatted dataset example for RACE-m. When predicting, we normalize by the unconditional probability of each answer as described in 2.

Figure G.4: Formatted dataset example for PIQA

<table><tr><td>Context →</td><td>My body cast a shadow over the grass because</td></tr><tr><td>Correct Answer →</td><td>the sun was rising.</td></tr><tr><td>Incorrect Answer →</td><td>the grass was cut.</td></tr></table>

Figure G.5: Formatted dataset example for COPA

<table><tr><td>Context →</td><td>(CNN) Yuval Rabin, whose father, Yitzhak Rabin, was assassinated while serving as Prime Minister of Israel, criticized Donald Trump for appealing to &quot;Second Amendment people&quot; in a speech and warned that the words that politicians use can incite violence and undermine democracy. &quot;Trump&#x27;s words are an incitement to the type of political violence that touched me personally,&quot; Rabin wrote in USAToday. He said that Trump&#x27;s appeal to &quot;Second Amendment people&quot; to stop Hillary Clinton -- comments that were criticized as a call for violence against Clinton, something Trump denied -- &quot;were a new level of ugliness in an ugly campaign season.&quot;- The son of a former Israeli Prime Minister who was assassinated wrote an op ed about the consequence of violent political rhetoric.- Warns of &quot;parallels&quot; between Israel of the 1990s and the U.S. today.</td></tr><tr><td>Correct Answer →</td><td>- Referencing his father, who was shot and killed by an extremist amid political tension in Israel in 1995, Rabin condemned Donald Trump&#x27;s aggressive rhetoric.</td></tr><tr><td>Correct Answer →</td><td>- Referencing his father, who was shot and killed by an extremist amid political tension in Israel in 1995, Rabin condemned Trump&#x27;s aggressive rhetoric.</td></tr><tr><td>Incorrect Answer →</td><td>- Referencing his father, who was shot and killed by an extremist amid political tension in Israel in 1995, Rabin condemned Hillary Clinton&#x27;s aggressive rhetoric.</td></tr><tr><td>Incorrect Answer →</td><td>- Referencing his father, who was shot and killed by an extremist amid political tension in Israel in 1995, Rabin condemned U.S.&#x27;s aggressive rhetoric.</td></tr><tr><td>Incorrect Answer →</td><td>- Referencing his father, who was shot and killed by an extremist amid political tension in Israel in 1995, Rabin condemned Yitzhak Rabin&#x27;s aggressive rhetoric.</td></tr></table>

Figure G.6: Formatted dataset example for ReCoRD. We consider the context above to be a single "problem" because this is how the task is presented in the ReCoRD dataset and scored in the ReCoRD evaluation script.

<table><tr><td>Context →</td><td>anli 1: anli 1: Fulton James MacGregor MSP is a Scottish politician who is a Scottish National Party (SNP) Member of Scottish Parliament for the constituency of Coatbridge and Chryston. MacGregor is currently Parliamentary Liaison Officer to Shona Robison, Cabinet Secretary for Health &amp; Sport. He also serves on the Justice and Education &amp; Skills committees in the Scottish Parliament.Question: Fulton James MacGregor is a Scottish politician who is a Liaison officer to Shona Robison who he swears is his best friend. True, False, or Neither?</td></tr><tr><td>Correct Answer →</td><td>Neither</td></tr><tr><td>Incorrect Answer →</td><td>True</td></tr><tr><td>Incorrect Answer →</td><td>False</td></tr><tr><td>Context →</td><td>Organisms require energy in order to do what?</td></tr><tr><td>Correct Answer →</td><td>mature and develop.</td></tr><tr><td>Incorrect Answer →</td><td>rest soundly.</td></tr><tr><td>Incorrect Answer →</td><td>absorb light.</td></tr><tr><td>Incorrect Answer →</td><td>take in nutrients.</td></tr></table>

Figure G.7: Formatted dataset example for ANLI R1

Figure G.8: Formatted dataset example for OpenBookQA. When predicting, we normalize by the unconditional probability of each answer as described in 2.

<table><tr><td>Context →</td><td>Making a cake: Several cake pops are shown on a display. A woman and girl are shown making the cake pops in a kitchen. They</td></tr><tr><td>Correct Answer →</td><td>bake them, then frost and decorate.</td></tr><tr><td>Incorrect Answer →</td><td>taste them as they place them on plates.</td></tr><tr><td>Incorrect Answer →</td><td>put the frosting on the cake as they pan it.</td></tr><tr><td>Incorrect Answer →</td><td>come out and begin decorating the cake as well.</td></tr></table>

Figure G.9: Formatted dataset example for HellaSwag

<table><tr><td>Context →</td><td>anli 3: anli 3: We shut the loophole which has American workers actually subsidizing the loss of their own job. They just passed an expansion of that loophole in the last few days: $43 billion of giveaways, including favors to the oil and gas industry and the people importing ceiling fans from China.Question: The loophole is now gone True, False, or Neither?</td></tr><tr><td>Correct Answer →</td><td>False</td></tr><tr><td>Incorrect Answer →</td><td>True</td></tr><tr><td>Incorrect Answer →</td><td>Neither</td></tr></table>

Figure G.10: Formatted dataset example for ANLI R3

<table><tr><td>Context →</td><td>Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?Answer:</td></tr><tr><td>Correct Answer →</td><td>dry palms</td></tr><tr><td>Incorrect Answer →</td><td>wet palms</td></tr><tr><td>Incorrect Answer →</td><td>palms covered with oil</td></tr><tr><td>Incorrect Answer →</td><td>palms covered with lotion</td></tr></table>

Figure G.11: Formatted dataset example for ARC (Challenge). When predicting, we normalize by the unconditional probability of each answer as described in 2.

<table><tr><td>Context →</td><td>lull is to trust as</td></tr><tr><td>Correct Answer →</td><td>cajole is to compliance</td></tr><tr><td>Incorrect Answer →</td><td>balk is to fortitude</td></tr><tr><td>Incorrect Answer →</td><td>betray is to loyalty</td></tr><tr><td>Incorrect Answer →</td><td>hinder is to destination</td></tr><tr><td>Incorrect Answer →</td><td>soothe is to passion</td></tr></table>

Figure G.12: Formatted dataset example for SAT Analogies

<table><tr><td>Correct Context →</td><td>Grace was happy to trade me her sweater for my jacket.</td><td>She thinks the sweater</td></tr><tr><td>Incorrect Context →</td><td>Grace was happy to trade me her sweater for my jacket.</td><td>She thinks the jacket</td></tr><tr><td>Target Completion →</td><td>looks dowdy on her.</td><td></td></tr></table>

Figure G.13: Formatted dataset example for Winograd. The ‘partial’ evaluation method we use compares the probability of the completion given a correct and incorrect context.

<table><tr><td>Correct Context →</td><td>Johnny likes fruits more than vegetables in his new keto diet because the fruits</td></tr><tr><td>Incorrect Context →</td><td>Johnny likes fruits more than vegetables in his new keto diet because the vegetables</td></tr><tr><td>Target Completion →</td><td>are saccharine.</td></tr></table>

Figure G.14: Formatted dataset example for Winogrande. The ‘partial’ evaluation method we use compares the probability of the completion given a correct and incorrect context.

<table><tr><td>Context →</td><td>READING COMPREHENSION ANSWER KEYWhile this process moved along, diplomacy continued its rounds. Direct pressure on the Taliban had proved unsuccessful. As one NSC staff note put it, &quot;Under the Taliban, Afghanistan is not so much a state sponsor of terrorism as it is a state sponsored by terrorists.&quot; In early 2000, the United States began a high-level effort to persuade Pakistan to use its influence over the Taliban. In January 2000, Assistant Secretary of State Karl Inderfurth and the State Department&#x27;s counterterrorism coordinator, Michael Sheehan, met with General Musharraf in Islamabad, dangling before him the possibility of a presidential visit in March as a reward for Pakistani cooperation. Such a visit was coveted by Musharraf, partly as a sign of his government&#x27;s legitimacy. He told the two envoys that he would meet with Mullah Omar and press him on Bin Laden. They left, however, reporting to Washington that Pakistan was unlikely in fact to do anything,&quot; given what it sees as the benefits of Taliban control of Afghanistan.&quot; President Clinton was scheduled to travel to India. The State Department felt that he should not visit India without also visiting Pakistan. The Secret Service and the CIA, however, warned in the strongest terms that visiting Pakistan would risk the President&#x27;s life. Counterterrorism officials also argued that Pakistan had not done enough to merit a presidential visit. But President Clinton insisted on including Pakistan in the itinerary for his trip to South Asia. His one-day stopover on March 25, 2000, was the first time a U.S. president had been there since 1969. At his meeting with Musharraf and others, President Clinton concentrated on tensions between Pakistan and India and the dangers of nuclear proliferation, but also discussed Bin Laden. President Clinton told us that when he pulled Musharraf aside for a brief, one-on-one meeting, he pleaded with the general for help regarding Bin Laden.&quot; I offered him the moon when I went to see him, in terms of better relations with the United States, if he&#x27;d help us get Bin Laden and deal with another issue or two.&quot; The U.S. effort continued.Who did The State Department feel should visit both India and Pakistan?</td></tr><tr><td>Correct Answer →</td><td>- [False] Bin Laden</td></tr><tr><td>Incorrect Answer →</td><td>- [True] Bin Laden</td></tr></table>

Figure G.15: Formatted dataset example for MultiRC. There are three levels within MultiRC: (1) the passage, (2) the questions, and (3) the answers. During evaluation, accuracy is determined at the per-question level, with a question being considered correct if and only if all the answers within the question are labeled correctly. For this reason, we use K to refer to the number of questions shown within the context.

<table><tr><td>Context →</td><td>Question: Which factor will most likely cause a person to develop a fever? Answer:</td></tr><tr><td>Correct Answer →</td><td>a bacterial population in the bloodstream</td></tr><tr><td>Incorrect Answer →</td><td>a leg muscle relaxing after exercise</td></tr><tr><td>Incorrect Answer →</td><td>several viral particles on the skin</td></tr><tr><td>Incorrect Answer →</td><td>carbohydrates being digested in the stomach</td></tr><tr><td>Context →</td><td>Bob went to the gas station to fill up his car. His tank was completely empty and so was his wallet. The cashier offered to pay for his gas if he came back later to pay. Bob felt grateful as he drove home.</td></tr><tr><td>Correct Answer →</td><td>Bob believed that there were good people in the world.</td></tr><tr><td>Incorrect Answer →</td><td>Bob contemplated how unfriendly the world was.</td></tr></table>

Figure G.16: Formatted dataset example for ARC (Easy). When predicting, we normalize by the unconditional probability of each answer as described in 2.

Figure G.17: Formatted dataset example for StoryCloze

<table><tr><td>Context →</td><td>Helsinki is the capital and largest city of Finland. It is in the region of Uusimaa, in southern Finland, on the shore of the Gulf of Finland.Helsinki has a population of , an urban population of , and a metropolitan population of over 1.4 million, making it the most populous municipality and urban area in Finland. Helsinki is some north of Tallinn, Estonia, east of Stockholm, Sweden, and west of Saint Petersburg, Russia. Helsinki has close historical connections with these three cities.The Helsinki metropolitan area includes the urban core of Helsinki, Espoo, Vantaa, Kauniainen, and surrounding commuter towns. It is the world’s northernmost metro area of over one million people, and the city is the northernmost capital of an EU member state. The Helsinki metropolitan area is the third largest metropolitan area in the Nordic countries after Stockholm and Copenhagen, and the City of Helsinki is the third largest after Stockholm and Oslo. Helsinki is Finland’s major political, educational, financial, cultural, and research center as well as one of northern Europe’s major cities. Approximately 75% of foreign companies that operate in Finland have settled in the Helsinki region. The nearby municipality of Vantaa is the location of Helsinki Airport, with frequent service to various destinations in Europe and Asia.Q: what is the most populous municipality in Finland?A: HelsinkiQ: how many people live there?A: 1.4 million in the metropolitan areaQ: what percent of the foreign companies that operate in Finland are in Helsinki?A: 75%Q: what towns are a part of the metropolitan area?A:</td></tr><tr><td>Target Completion →</td><td>Helsinki, Espoo, Vantaa, Kauniainen, and surrounding commuter towns</td></tr></table>

Figure G.18: Formatted dataset example for CoQA

<table><tr><td>Context →</td><td>Please unscramble the letters into a word, and write that word: asinoc =</td></tr><tr><td>Target Completion →</td><td>casino</td></tr><tr><td>Context →</td><td>Passage: Saint Jean de Brébeuf was a French Jesuit missionary who travelled to New France in 1625. There he worked primarily with the Huron for the rest of his life, except for a few years in France from 1629 to 1633. He learned their language and culture, writing extensively about each to aid other missionaries. In 1649, Brébeuf and another missionary were captured when an Iroquois raid took over a Huron village. Together with Huron captives, the missionaries were ritually tortured and killed on March 16, 1649. Brébeuf was beatified in 1925 and among eight Jesuit missionaries canonized as saints in the Roman Catholic Church in 1930. Question: How many years did Saint Jean de Brébeuf stay in New France before he went back to France for a few years?Answer:</td></tr><tr><td>Target Completion →</td><td>4</td></tr></table>

Figure G.19: Formatted dataset example for Cycled Letters

Figure G.20: Formatted dataset example for DROP

<table><tr><td>Context →</td><td>Fill in blank:</td></tr><tr><td></td><td>She held the torch in front of her.</td></tr><tr><td></td><td>She caught her breath.</td></tr><tr><td></td><td>&quot;Chris? There&#x27;s a step.&quot;</td></tr><tr><td></td><td>&quot;What?&quot;</td></tr><tr><td></td><td>&quot;A step. Cut in the rock. About fifty feet ahead.&quot; She moved faster.They both moved faster. &quot;In fact,&quot; she said, raising the torch higher,&quot;there&#x27;s more than a ____. -&gt;</td></tr><tr><td>Target Completion →</td><td>step</td></tr></table>

Figure G.21: Formatted dataset example for LAMBADA

<table><tr><td>Context →</td><td>Please unscramble the letters into a word, and write that word: skicts =</td></tr><tr><td>Target Completion →</td><td>sticks</td></tr></table>

Figure G.22: Formatted dataset example for Anagrams 1 (A1)

<table><tr><td>Context →</td><td>Please unscramble the letters into a word, and write that word: volwskagen =</td></tr><tr><td>Target Completion →</td><td>volkswagen</td></tr></table>

Figure G.23: Formatted dataset example for Anagrams 2

<table><tr><td>Context →</td><td>Q: Who played tess on touched by an angel?</td></tr><tr><td></td><td>A:</td></tr><tr><td>Target Completion →</td><td>Delloreese Patricia Early (July 6, 1931 { November 19, 2017), known professionally as Della Reese</td></tr><tr><td>Context →</td><td>TITLE: William Perry (American football) - Professional careerPARAGRAPH: In 1985, he was selected in the first round of the 1985 NFL Draft by the Chicago Bears; he had been hand-picked by coach Mike Ditka. However, defensive coordinator Buddy Ryan, who had a highly acrimonious relationship with Ditka, called Perry a "wasted draft-pick". Perry soon became a pawn in the political power struggle between Ditka and Ryan. Perry's "Refrigerator" nickname followed him into the NFL and he quickly became a favorite of the Chicago Bears fans. Teammates called him "Biscuit," as in "one biscuit shy of 350 pounds." While Ryan refused to play Perry, Ditka decided to use Perry as a fullback when the team was near the opponents' goal line or in fourth and short situations, either as a ball carrier or a lead blocker for star running back Walter Payton. Ditka stated the inspiration for using Perry as a fullback came to him during five-yard sprint exercises. During his rookie season, Perry rushed for two touchdowns and caught a pass for one. Perry even had the opportunity to run the ball during Super Bowl XX, as a nod to his popularity and contributions to the team's success. The first time he got the ball, he was tackled for a one-yard loss while attempting to throw his first NFL pass on a halfback option play. The second time he got the ball, he scored a touchdown (running over Patriots linebacker Larry McGrew in the process). About halfway through his rookie season, Ryan finally began to play Perry, who soon proved that he was a capable defensive lineman. His Super Bowl ring size is the largest of any professional football player in the history of the event. His ring size is 25, while the ring size for the average adult male is between 10 and 12. Perry went on to play for ten years in the NFL, retiring after the 1994 season. In his ten years as a pro, he regularly struggled with his weight, which hampered his performance at times. He played in 138 games, recording 29.5 sacks and five fumble recoveries, which he returned for a total of 71 yards. In his offensive career he ran five yards for two touchdowns, and had one reception for another touchdown. Perry later attempted a comeback, playing an unremarkable 1996 season with the London Monarchs of the World League of American Football (later NFL Europa).Q: what team did he play for?A:</td></tr><tr><td>Target Completion →</td><td>the Chicago Bears</td></tr></table>

Figure G.24: Formatted dataset example for Natural Questions

Figure G.25: Formatted dataset example for QuAC

<table><tr><td>Context →</td><td>Please unscramble the letters into a word, and write that word: r e!c.i p r o.c a/l =</td></tr><tr><td>Target Completion →</td><td>reciprocal</td></tr></table>

Figure G.26: Formatted dataset example for Symbol Insertion

<table><tr><td>Context →</td><td>Please unscramble the letters into a word, and write that word:taefed =</td></tr><tr><td>Target Completion →</td><td>defeat</td></tr><tr><td>Context →</td><td>Title: The_Blitz</td></tr><tr><td></td><td>Background: From the German point of view, March 1941 saw an improvement. The Luftwaffe flew 4,000 sorties that month, including 12 major and three heavy attacks. The electronic war intensified but the Luftwaffe flew major inland missions only on moonlit nights. Ports were easier to find and made better targets. To confuse the British, radio silence was observed until the bombs fell. X- and Y-Gerät beams were placed over false targets and switched only at the last minute. Rapid frequency changes were introduced for X-Gerät, whose wider band of frequencies and greater tactical flexibility ensured it remained effective at a time when British selective jamming was degrading the effectiveness of Y-Gerät.</td></tr><tr><td></td><td>Q: How many sorties were flown in March 1941?</td></tr><tr><td></td><td>A: 4,000</td></tr><tr><td></td><td>Q: When did the Luftwaffe fly inland missions?</td></tr><tr><td></td><td>A:</td></tr><tr><td>Target Completion →</td><td>only on moonlit nights</td></tr></table>

Figure G.27: Formatted dataset example for Reversed Words

Figure G.28: Formatted dataset example for SQuADv2

<table><tr><td>Context →</td><td>Normal force -- In a simple case such as an object resting upon a table, the normal force on the object is equal but in opposite direction to the gravitational force applied on the object (or the weight of the object), that is, N = m g (\displaystyle N=mg), where m is mass, and g is the gravitational field strength (about 9.81 m/s on Earth). The normal force here represents the force applied by the table against the object that prevents it from sinking through the table and requires that the table is sturdy enough to deliver this normal force without breaking. However, it is easy to assume that the normal force and weight are action-reaction force pairs (a common mistake). In this case, the normal force and weight need to be equal in magnitude to explain why there is no upward acceleration of the object. For example, a ball that bounces upwards accelerates upwards because the normal force acting on the ball is larger in magnitude than the weight of the ball.question: is the normal force equal to the force of gravity?answer:</td></tr><tr><td>Target Completion →</td><td>yes</td></tr></table>

Figure G.29: Formatted dataset example for BoolQ

<table><tr><td>Context →</td><td>The trend toward lower rents may seem surprising given that some communities in New York are bemoaning the loss of favorite local businesses to high rents. But, despite the recent softening, for many of these retailers there’s still been too big a jump from the rental rates of the late 1970s, when their leases were signed. Certainly, the recent drop in prices doesn’t mean Manhattan comes cheap. question: Manhattan comes cheap. true, false, or neither? answer:</td></tr><tr><td>Target Completion →</td><td>false</td></tr><tr><td>Context →</td><td>The bet, which won him dinner for four, was regarding the existence and mass of the top quark, an elementary particle discovered in 1995. question: The Top Quark is the last of six flavors of quarks predicted by the standard model theory of particle physics. True or False? answer:</td></tr><tr><td>Target Completion →</td><td>False</td></tr></table>

Figure G.30: Formatted dataset example for CB

Figure G.31: Formatted dataset example for RTE

<table><tr><td>Context →</td><td>An outfitter provided everything needed for the safari.Before his first walking holiday, he went to a specialist outfitter to buy some boots.question: Is the word ‘outfitter’ used in the same way in the two sentences above?answer:</td></tr><tr><td>Target Completion →</td><td>no</td></tr></table>

Figure G.32: Formatted dataset example for WiC

<table><tr><td>Context →</td><td>Final Exam with Answer KeyInstructions: Please carefully read the following passages. For each passage, you must identify which noun the pronoun marked in *bold* refers to.==== Passage: Mr. Moncrieff visited Chester&#x27;s luxurious New York apartment, thinking that it belonged to his son Edward. The result was that Mr. Moncrieff has decided to cancel Edward&#x27;s allowance on the ground that he no longer requires *his* financial support.Question: In the passage above, what does the pronoun &quot;*his*&quot; refer to?Answer:</td></tr><tr><td>Target Completion →</td><td>mr. moncrieff</td></tr></table>

Figure G.33: Formatted dataset example for WSC

<table><tr><td rowspan="2"></td><td rowspan="2">Context →</td><td>Q: ‘Nude Descending A Staircase’ is perhaps the most famous painting by which 20th century artist?</td></tr><tr><td>A:</td></tr><tr><td>Target</td><td>Completion →</td><td>MARCEL DUCHAMP</td></tr><tr><td>Target</td><td>Completion →</td><td>r mutt</td></tr><tr><td>Target</td><td>Completion →</td><td>duchamp</td></tr><tr><td>Target</td><td>Completion →</td><td>marcel duchamp</td></tr><tr><td>Target</td><td>Completion →</td><td>R.Mutt</td></tr><tr><td>Target</td><td>Completion →</td><td>Marcel duChamp</td></tr><tr><td>Target</td><td>Completion →</td><td>Henri-Robert-Marcel Duchamp</td></tr><tr><td>Target</td><td>Completion →</td><td>Marcel du Champ</td></tr><tr><td>Target</td><td>Completion →</td><td>henri robert marcel duchamp</td></tr><tr><td>Target</td><td>Completion →</td><td>Duchampian</td></tr><tr><td>Target</td><td>Completion →</td><td>Duchamp</td></tr><tr><td>Target</td><td>Completion →</td><td>duchampian</td></tr><tr><td>Target</td><td>Completion →</td><td>marcel du champ</td></tr><tr><td>Target</td><td>Completion →</td><td>Marcel Duchamp</td></tr><tr><td>Target</td><td>Completion →</td><td>MARCEL DUCHAMP</td></tr></table>

Figure G.34: Formatted dataset example for TriviaQA. TriviaQA allows for multiple valid completions.

<table><tr><td>Context →</td><td>Q: What school did burne hogarth establish?</td></tr><tr><td></td><td>A:</td></tr><tr><td>Target Completion →</td><td>School of Visual Arts</td></tr></table>

Figure G.35: Formatted dataset example for WebQA

<table><tr><td>Context →</td><td>Keinesfalls dürfen diese für den kommerziellen Gebrauch verwendet werden. =</td></tr><tr><td>Target Completion →</td><td>In no case may they be used for commercial purposes.</td></tr></table>

Figure G.36: Formatted dataset example for De→En. This is the format for one- and few-shot learning, for this and other language tasks, the format for zero-shot learning is “Q: What is the {language} translation of {sentence} A: {translation}.”

<table><tr><td>Context →</td><td>In no case may they be used for commercial purposes. =</td></tr><tr><td>Target Completion →</td><td>Keinesfalls dürfen diese für den kommerziellen Gebrauch verwendet werden.</td></tr></table>

Figure G.37: Formatted dataset example for En→De

<table><tr><td>Context →</td><td>Analysis of instar distributions of larval I. verticalis collected from a series of ponds also indicated that males were in more advanced instars than females. =</td></tr><tr><td>Target Completion →</td><td>L’analyse de la distribution de fréquence des stades larvaires d’I. verticalis dans une série d’étangs a également démontré que les larves mâles étaient à des stades plus avancés que les larves femelles.</td></tr></table>

Figure G.38: Formatted dataset example for En→Fr

<table><tr><td>Context →</td><td>L&#x27;analyse de la distribution de fréquence des stades larvaires d&#x27;I. verticalis dans une série d&#x27;étangs a également démontré que les larves mâles étaient à des stades plus avancés que les larves femelles. =</td></tr><tr><td>Target Completion →</td><td>Analysis of instar distributions of larval I. verticalis collected from a series of ponds also indicated that males were in more advanced instars than females.</td></tr></table>

Figure G.39: Formatted dataset example for Fr→En

<table><tr><td>Context →</td><td>The truth is that you want, at any price, and against the wishes of the peoples of Europe, to continue the negotiations for Turkey’s accession to the European Union, despite Turkey’s continuing refusal to recognise Cyprus and despite the fact that the democratic reforms are at a standstill. =</td></tr><tr><td>Target Completion →</td><td>Adevărul este că vă doriți, cu orice preț și împotriva dorinței europenilor, să continuați negocierile de aderare a Turciei la Uniunea Europeană, în ciuda refuzului continuu al Turciei de a recunoaște Ciprul și în ciuda faptului că reformele democratice au ajuns într-un punct mort.</td></tr><tr><td>Context →</td><td>Adevărul este că vă doriți, cu orice preț și împotriva dorinței europenilor, să continuați negocierile de aderare a Turciei la Uniunea Europeană, în ciuda refuzului continuu al Turciei de a recunoaște Ciprul și în ciuda faptului că reformele democratice au ajuns într-un punct mort. =</td></tr><tr><td>Target Completion →</td><td>The truth is that you want, at any price, and against the wishes of the peoples of Europe, to continue the negotiations for Turkey’s accession to the European Union, despite Turkey’s continuing refusal to recognise Cyprus and despite the fact that the democratic reforms are at a standstill.</td></tr></table>

Figure G.40: Formatted dataset example for En→Ro

Figure G.41: Formatted dataset example for Ro→En

<table><tr><td>Context →</td><td>Q: What is (2 * 4) * 6?</td></tr><tr><td></td><td>A:</td></tr><tr><td>Target Completion →</td><td>48</td></tr></table>

Figure G.42: Formatted dataset example for Arithmetic 1DC

<table><tr><td>Context →</td><td>Q: What is 17 minus 14?</td></tr><tr><td></td><td>A:</td></tr><tr><td>Target Completion →</td><td>3</td></tr></table>

Figure G.43: Formatted dataset example for Arithmetic 2D-

<table><tr><td>Context →</td><td>Q: What is 98 plus 45?</td></tr><tr><td></td><td>A:</td></tr><tr><td>Target Completion →</td><td>143</td></tr></table>

Figure G.44: Formatted dataset example for Arithmetic 2D+

<table><tr><td>Context →</td><td>Q: What is 95 times 45?</td></tr><tr><td></td><td>A:</td></tr><tr><td>Target Completion →</td><td>4275</td></tr></table>

Figure G.45: Formatted dataset example for Arithmetic 2Dx

<table><tr><td>Context →</td><td>Q: What is 509 minus 488?</td></tr><tr><td></td><td>A:</td></tr><tr><td>Target Completion →</td><td>21</td></tr></table>

Figure G.46: Formatted dataset example for Arithmetic 3D-

<table><tr><td rowspan="2">Context →</td><td>Q: What is 556 plus 497?</td></tr><tr><td>A:</td></tr><tr><td>Target Completion →</td><td>1053</td></tr></table>

Figure G.47: Formatted dataset example for Arithmetic 3D+

<table><tr><td rowspan="2">Context →</td><td>Q: What is 6209 minus 3365?</td></tr><tr><td>A:</td></tr><tr><td>Target Completion →</td><td>2844</td></tr></table>

Figure G.48: Formatted dataset example for Arithmetic 4D-

```txt
Context → Q: What is 9923 plus 617?
A:
Target Completion → 10540 
```  
Figure G.49: Formatted dataset example for Arithmetic 4D+

```txt
Context → Q: What is 40649 minus 78746?
A:
Target Completion → -38097 
```  
Figure G.50: Formatted dataset example for Arithmetic 5D—

```txt
Context → Q: What is 65360 plus 16204?
A:
Target Completion → 81564 
```  
Figure G.51: Formatted dataset example for Arithmetic 5D+

H Results on All Tasks for All Model Sizes 

<table><tr><td rowspan="2">Name</td><td rowspan="2">Metric</td><td colspan="4">Fine-tune</td><td colspan="6">Zero-Shot</td><td colspan="6">One-Shot</td><td colspan="6">Few-Shot</td><td>175B (test server)</td><td></td></tr><tr><td>Split</td><td>SOTA</td><td>K</td><td></td><td>Small</td><td>Med Large</td><td>XL</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>175B</td><td>Small</td><td>Med Large</td><td>XL</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>175B</td><td>Small</td><td>Med Large</td><td>XL</td><td>2.7B</td><td>6.7B</td><td>13B</td></tr><tr><td>HellaSwag</td><td>acc</td><td>dev</td><td>85.6</td><td>20</td><td>33.7</td><td>43.6</td><td>51.0</td><td>54.7</td><td>62.8</td><td>67.4</td><td>70.9</td><td>78.9</td><td>33.0</td><td>42.9</td><td>50.5</td><td>53.5</td><td>61.9</td><td>66.5</td><td>70.0</td><td>78.1</td><td>33.5</td><td>43.1</td><td>51.3</td><td>54.9</td><td>62.9</td></tr><tr><td>LAMBADA</td><td>acc</td><td>test</td><td>68.0</td><td>15</td><td>42.7</td><td>54.3</td><td>60.4</td><td>63.6</td><td>67.1</td><td>70.3</td><td>72.5</td><td>76.2</td><td>22.0</td><td>47.1</td><td>52.6</td><td>58.3</td><td>61.1</td><td>65.4</td><td>69.0</td><td>72.5</td><td>22.0</td><td>40.4</td><td>63.2</td><td>57.0</td><td>78.1</td></tr><tr><td>LAMBADA</td><td>ppl</td><td>test</td><td>8.63</td><td>15</td><td>18.6</td><td>9.09</td><td>6.53</td><td>5.44</td><td>4.60</td><td>4.00</td><td>3.56</td><td>3.00</td><td>165.0</td><td>11.6</td><td>8.29</td><td>6.46</td><td>5.53</td><td>4.61</td><td>4.06</td><td>3.35</td><td>165.0</td><td>27.6</td><td>6.63</td><td>7.45</td><td>2.89</td></tr><tr><td>StoryCloze</td><td>acc</td><td>test</td><td>91.8</td><td>70</td><td>63.3</td><td>68.5</td><td>72.4</td><td>73.4</td><td>77.2</td><td>77.7</td><td>79.5</td><td>83.2</td><td>62.3</td><td>68.7</td><td>72.3</td><td>74.2</td><td>77.3</td><td>78.7</td><td>79.7</td><td>84.7</td><td>62.3</td><td>70.2</td><td>73.9</td><td>76.1</td><td>80.2</td></tr><tr><td>NQs</td><td>acc</td><td>test</td><td>44.5</td><td>64</td><td>0.64</td><td>1.75</td><td>2.71</td><td>4.40</td><td>6.01</td><td>5.79</td><td>7.84</td><td>14.6</td><td>1.19</td><td>3.07</td><td>4.79</td><td>5.43</td><td>8.73</td><td>9.78</td><td>13.7</td><td>23.0</td><td>1.72</td><td>4.46</td><td>7.89</td><td>9.72</td><td>13.2</td></tr><tr><td>TriviaQA</td><td>acc</td><td>dev</td><td>68.0</td><td>64</td><td>4.15</td><td>7.61</td><td>14.0</td><td>19.7</td><td>31.3</td><td>38.7</td><td>41.8</td><td>64.3</td><td>4.19</td><td>12.9</td><td>20.5</td><td>26.5</td><td>35.9</td><td>44.4</td><td>51.3</td><td>68.0</td><td>6.96</td><td>16.3</td><td>26.5</td><td>32.1</td><td>42.3</td></tr><tr><td>WebQs</td><td>acc</td><td>test</td><td>45.5</td><td>64</td><td>1.77</td><td>3.20</td><td>4.33</td><td>4.63</td><td>7.92</td><td>7.73</td><td>8.22</td><td>14.4</td><td>2.56</td><td>6.20</td><td>8.51</td><td>9.15</td><td>14.5</td><td>15.1</td><td>19.0</td><td>25.3</td><td>5.46</td><td>12.6</td><td>15.9</td><td>19.6</td><td>24.8</td></tr><tr><td>Ro→En 16</td><td>BLEU-mb</td><td>test</td><td>39.9</td><td>64</td><td>2.08</td><td>2.71</td><td>3.09</td><td>3.15</td><td>16.3</td><td>8.34</td><td>20.2</td><td>19.9</td><td>0.55</td><td>15.4</td><td>23.0</td><td>26.3</td><td>30.6</td><td>33.2</td><td>35.6</td><td>38.6</td><td>1.25</td><td>20.7</td><td>25.8</td><td>29.2</td><td>33.1</td></tr><tr><td>Ro→En 16</td><td>BLEU-sb</td><td>test</td><td></td><td>64</td><td>2.39</td><td>3.08</td><td>3.49</td><td>3.56</td><td>16.8</td><td>8.75</td><td>20.8</td><td>20.9</td><td>0.65</td><td>15.9</td><td>23.6</td><td>26.8</td><td>31.3</td><td>34.2</td><td>36.7</td><td>40.0</td><td>1.40</td><td>21.3</td><td>26.6</td><td>30.1</td><td>34.3</td></tr><tr><td>En→Ro 16</td><td>BLEU-mb</td><td>test</td><td>38.5</td><td>64</td><td>2.14</td><td>2.65</td><td>2.53</td><td>2.50</td><td>3.46</td><td>4.24</td><td>5.32</td><td>14.1</td><td>0.35</td><td>3.30</td><td>7.89</td><td>8.72</td><td>13.2</td><td>15.1</td><td>17.3</td><td>20.6</td><td>1.25</td><td>5.90</td><td>9.33</td><td>10.7</td><td>14.3</td></tr><tr><td>En→Ro 16</td><td>BLEU-sb</td><td>test</td><td></td><td>64</td><td>2.61</td><td>3.11</td><td>3.07</td><td>3.09</td><td>4.26</td><td>5.31</td><td>6.43</td><td>18.0</td><td>0.55</td><td>3.90</td><td>9.15</td><td>10.3</td><td>15.7</td><td>18.2</td><td>20.8</td><td>24.9</td><td>1.64</td><td>7.40</td><td>10.9</td><td>12.9</td><td>17.2</td></tr><tr><td>Fr→En 14</td><td>BLEU-mb</td><td>test</td><td>35.0</td><td>64</td><td>1.81</td><td>2.53</td><td>3.47</td><td>3.13</td><td>20.6</td><td>15.1</td><td>21.8</td><td>21.2</td><td>1.28</td><td>15.9</td><td>23.7</td><td>26.3</td><td>29.0</td><td>30.5</td><td>30.2</td><td>33.7</td><td>4.98</td><td>25.5</td><td>28.5</td><td>31.1</td><td>33.7</td></tr><tr><td>Fr→En 14</td><td>BLEU-sb</td><td>test</td><td></td><td>64</td><td>2.29</td><td>2.99</td><td>3.90</td><td>3.60</td><td>21.2</td><td>15.5</td><td>22.4</td><td>21.9</td><td>1.50</td><td>16.3</td><td>24.4</td><td>27.0</td><td>30.0</td><td>31.6</td><td>31.4</td><td>35.6</td><td>5.30</td><td>26.2</td><td>29.5</td><td>32.2</td><td>35.1</td></tr><tr><td>En→Fr 14</td><td>BLEU-mb</td><td>test</td><td>45.6</td><td>64</td><td>1.74</td><td>2.16</td><td>2.73</td><td>2.15</td><td>15.1</td><td>8.82</td><td>12.0</td><td>25.2</td><td>0.49</td><td>8.00</td><td>14.8</td><td>15.9</td><td>20.3</td><td>23.3</td><td>24.9</td><td>28.3</td><td>4.08</td><td>14.5</td><td>19.3</td><td>21.5</td><td>24.9</td></tr><tr><td>En→Fr 14</td><td>BLEU-sb</td><td>test</td><td>45.9</td><td>64</td><td>2.44</td><td>2.75</td><td>3.54</td><td>2.82</td><td>19.3</td><td>11.4</td><td>15.3</td><td>31.3</td><td>0.81</td><td>10.0</td><td>18.2</td><td>19.3</td><td>24.7</td><td>28.3</td><td>30.1</td><td>34.1</td><td>5.31</td><td>18.0</td><td>23.6</td><td>26.1</td><td>30.3</td></tr><tr><td>De→En 16</td><td>BLEU-mb</td><td>test</td><td>40.2</td><td>64</td><td>2.06</td><td>2.87</td><td>3.41</td><td>3.63</td><td>21.5</td><td>17.3</td><td>23.0</td><td>27.2</td><td>0.83</td><td>16.2</td><td>22.5</td><td>24.7</td><td>28.2</td><td>30.7</td><td>33.0</td><td>30.4</td><td>3.25</td><td>22.7</td><td>26.2</td><td>29.2</td><td>32.7</td></tr><tr><td>De→En 16</td><td>BLEU-sb</td><td>test</td><td></td><td>64</td><td>2.39</td><td>3.27</td><td>3.85</td><td>4.04</td><td>22.5</td><td>18.2</td><td>24.4</td><td>28.6</td><td>0.93</td><td>17.1</td><td>23.4</td><td>25.8</td><td>29.2</td><td>31.9</td><td>34.5</td><td>32.1</td><td>3.60</td><td>23.8</td><td>27.5</td><td>30.5</td><td>34.1</td></tr><tr><td>En→De 16</td><td>BLEU-mb</td><td>test</td><td>41.2</td><td>64</td><td>1.70</td><td>2.27</td><td>2.31</td><td>2.43</td><td>12.9</td><td>8.66</td><td>10.4</td><td>24.6</td><td>0.50</td><td>7.00</td><td>12.9</td><td>13.1</td><td>18.3</td><td>20.9</td><td>22.5</td><td>26.2</td><td>3.42</td><td>12.3</td><td>15.4</td><td>17.1</td><td>20.9</td></tr><tr><td>En→De 16</td><td>BLEU-sb</td><td>test</td><td>41.2</td><td>64</td><td>2.09</td><td>2.65</td><td>2.75</td><td>2.92</td><td>13.7</td><td>9.36</td><td>11.0</td><td>25.3</td><td>0.54</td><td>7.40</td><td>13.4</td><td>13.4</td><td>18.8</td><td>21.7</td><td>23.3</td><td>27.3</td><td>3.78</td><td>12.9</td><td>16.1</td><td>17.7</td><td>21.7</td></tr><tr><td>Winograd</td><td>acc</td><td>test</td><td>93.8</td><td>7</td><td>66.3</td><td>72.9</td><td>74.7</td><td>76.9</td><td>82.4</td><td>85.7</td><td>87.9</td><td>88.3</td><td>63.4</td><td>68.5</td><td>72.9</td><td>76.9</td><td>82.4</td><td>84.6</td><td>86.1</td><td>89.7</td><td>63.4</td><td>67.4</td><td>73.6</td><td>76.9</td><td>84.3</td></tr><tr><td>Winogrande</td><td>acc</td><td>dev</td><td>84.6</td><td>50</td><td>52.0</td><td>52.1</td><td>57.4</td><td>58.7</td><td>62.3</td><td>64.5</td><td>67.9</td><td>70.2</td><td>51.3</td><td>53.0</td><td>58.3</td><td>59.1</td><td>61.7</td><td>65.8</td><td>66.9</td><td>73.2</td><td>51.3</td><td>52.6</td><td>57.5</td><td>59.1</td><td>62.6</td></tr><tr><td>PIQA</td><td>acc</td><td>dev</td><td>77.1</td><td>50</td><td>64.6</td><td>70.2</td><td>72.9</td><td>75.1</td><td>75.6</td><td>78.0</td><td>78.5</td><td>81.0</td><td>64.3</td><td>69.3</td><td>71.8</td><td>74.4</td><td>74.3</td><td>76.3</td><td>77.8</td><td>80.5</td><td>64.3</td><td>69.4</td><td>72.0</td><td>74.3</td><td>75.4</td></tr><tr><td>ARC (Challenge)</td><td>acc</td><td>test</td><td>78.5</td><td>50</td><td>26.6</td><td>29.5</td><td>31.8</td><td>35.5</td><td>38.0</td><td>41.4</td><td>43.7</td><td>51.4</td><td>25.5</td><td>30.2</td><td>31.6</td><td>36.4</td><td>38.4</td><td>41.5</td><td>43.1</td><td>53.2</td><td>25.5</td><td>28.4</td><td>32.3</td><td>36.7</td><td>39.5</td></tr><tr><td>ARC (Easy)</td><td>acc</td><td>test</td><td>92.0</td><td>50</td><td>43.6</td><td>46.5</td><td>53.0</td><td>53.8</td><td>58.2</td><td>60.2</td><td>63.8</td><td>68.8</td><td>42.7</td><td>48.2</td><td>54.6</td><td>55.9</td><td>60.3</td><td>62.6</td><td>66.8</td><td>71.2</td><td>42.7</td><td>51.0</td><td>58.1</td><td>59.1</td><td>62.1</td></tr><tr><td>OpenBookQA</td><td>acc</td><td>test</td><td>87.2</td><td>100</td><td>35.6</td><td>43.2</td><td>45.2</td><td>46.8</td><td>53.0</td><td>50.4</td><td>55.6</td><td>57.6</td><td>37.0</td><td>39.8</td><td>46.2</td><td>46.4</td><td>53.4</td><td>53.0</td><td>55.8</td><td>58.8</td><td>37.0</td><td>43.6</td><td>48.0</td><td>50.6</td><td>55.6</td></tr><tr><td>Quac</td><td>f1</td><td>dev</td><td>74.4</td><td>5</td><td>21.2</td><td>26.8</td><td>31.0</td><td>30.1</td><td>34.7</td><td>36.1</td><td>38.4</td><td>41.5</td><td>21.1</td><td>26.9</td><td>31.9</td><td>32.3</td><td>37.4</td><td>39.0</td><td>40.6</td><td>43.4</td><td>21.6</td><td>27.6</td><td>32.9</td><td>34.2</td><td>38.2</td></tr><tr><td>RACE-h</td><td>acc</td><td>test</td><td>90.0</td><td>10</td><td>35.2</td><td>37.9</td><td>40.1</td><td>40.9</td><td>42.4</td><td>44.1</td><td>44.6</td><td>45.5</td><td>34.3</td><td>37.7</td><td>40.0</td><td>42.0</td><td>43.8</td><td>44.3</td><td>44.6</td><td>45.9</td><td>34.3</td><td>37.0</td><td>40.4</td><td>41.4</td><td>42.3</td></tr><tr><td>RACE-m</td><td>acc</td><td>test</td><td>93.1</td><td>10</td><td>42.1</td><td>47.2</td><td>52.1</td><td>52.3</td><td>54.7</td><td>54.4</td><td>56.7</td><td>58.4</td><td>42.3</td><td>47.3</td><td>51.7</td><td>55.2</td><td>56.1</td><td>54.7</td><td>56.9</td><td>57.4</td><td>42.3</td><td>47.0</td><td>52.7</td><td>53.0</td><td>55.6</td></tr><tr><td>SQuADv2</td><td>em</td><td>dev</td><td>90.7</td><td>16</td><td>22.6</td><td>32.8</td><td>33.9</td><td>43.1</td><td>43.6</td><td>45.4</td><td>49.0</td><td>52.6</td><td>25.1</td><td>37.5</td><td>37.9</td><td>47.9</td><td>47.9</td><td>51.1</td><td>56.0</td><td>60.1</td><td>27.5</td><td>40.5</td><td>39.2</td><td>53.5</td><td>50.0</td></tr><tr><td>SQuADv2</td><td>f1</td><td>dev</td><td>93.0</td><td>16</td><td>28.3</td><td>40.2</td><td>41.4</td><td>50.3</td><td>51.0</td><td>52.7</td><td>56.3</td><td>59.5</td><td>30.1</td><td>43.6</td><td>44.1</td><td>54.0</td><td>54.1</td><td>57.1</td><td>61.8</td><td>65.4</td><td>32.1</td><td>45.5</td><td>44.9</td><td>58.7</td><td>55.9</td></tr><tr><td>CoQA</td><td>f1</td><td>dev</td><td>90.7</td><td>5</td><td>34.5</td><td>55.0</td><td>61.8</td><td>65.3</td><td>71.1</td><td>72.8</td><td>76.3</td><td>81.5</td><td>30.6</td><td>52.1</td><td>61.6</td><td>66.1</td><td>71.8</td><td>75.1</td><td>77.9</td><td>84.0</td><td>31.1</td><td>52.0</td><td>62.7</td><td>66.8</td><td>73.2</td></tr><tr><td>DROP</td><td>f1</td><td>dev</td><td>89.1</td><td>20</td><td>9.40</td><td>13.6</td><td>14.4</td><td>16.4</td><td>19.7</td><td>17.0</td><td>24.0</td><td>23.6</td><td>11.7</td><td>18.1</td><td>20.9</td><td>23.0</td><td>26.4</td><td>27.3</td><td>29.2</td><td>34.3</td><td>12.9</td><td>18.7</td><td>24.0</td><td>25.6</td><td>29.7</td></tr><tr><td>BoolQ</td><td>acc</td><td>dev</td><td>91.0</td><td>32</td><td>49.7</td><td>60.3</td><td>58.9</td><td>62.4</td><td>67.1</td><td>65.4</td><td>66.2</td><td>60.5</td><td>52.6</td><td>61.7</td><td>60.4</td><td>63.7</td><td>68.4</td><td>68.7</td><td>69.0</td><td>76.7</td><td>43.1</td><td>60.6</td><td>62.0</td><td>64.1</td><td>70.3</td></tr><tr><td>CB</td><td>acc</td><td>dev</td><td>96.9</td><td>32</td><td>0.00</td><td>32.1</td><td>8.93</td><td>19.6</td><td>19.6</td><td>28.6</td><td>19.6</td><td>46.4</td><td>55.4</td><td>53.6</td><td>53.6</td><td>48.2</td><td>57.1</td><td>33.9</td><td>55.4</td><td>64.3</td><td>42.9</td><td>58.9</td><td>53.6</td><td>69.6</td><td>67.9</td></tr><tr><td>CB</td><td>f1</td><td>dev</td><td>93.9</td><td>32</td><td>0.00</td><td>29.3</td><td>11.4</td><td>17.4</td><td>22.4</td><td>25.1</td><td>20.3</td><td>42.8</td><td>60.1</td><td>39.8</td><td>45.6</td><td>37.5</td><td>45.7</td><td>28.5</td><td>44.6</td><td>52.5</td><td>26.1</td><td>40.4</td><td>32.6</td><td>48.3</td><td>45.7</td></tr><tr><td>Copa</td><td>acc</td><td>dev</td><td>94.8</td><td>32</td><td>66.0</td><td>68.0</td><td>73.0</td><td>77.0</td><td>76.0</td><td>80.0</td><td>84.0</td><td>91.0</td><td>62.0</td><td>64.0</td><td>66.0</td><td>74.0</td><td>76.0</td><td>82.0</td><td>86.0</td><td>87.0</td><td>67.0</td><td>64.0</td><td>72.0</td><td>77.0</td><td>83.0</td></tr><tr><td>RTE</td><td>acc</td><td>dev</td><td>92.5</td><td>32</td><td>47.7</td><td>49.8</td><td>48.4</td><td>56.0</td><td>46.6</td><td>55.2</td><td>62.8</td><td>63.5</td><td>53.1</td><td>47.3</td><td>49.5</td><td>49.5</td><td>54.9</td><td>54.9</td><td>56.3</td><td>70.4</td><td>52.3</td><td>48.4</td><td>46.9</td><td>50.9</td><td>56.3</td></tr><tr><td>WiC</td><td>acc</td><td>dev</td><td>76.1</td><td>32</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>50.0</td><td>50.3</td><td>50.3</td><td>49.2</td><td>49.4</td><td>50.3</td><td>50.0</td><td>48.6</td><td>49.8</td><td>55.0</td><td>53.0</td><td>53.0</td><td>51.6</td></tr><tr><td>WSC</td><td>acc</td><td>dev</td><td>93.8</td><td>32</td><td>59.6</td><td>56.7</td><td>65.4</td><td>61.5</td><td>66.3</td><td>60.6</td><td>64.4</td><td>65.4</td><td>58.7</td><td>58.7</td><td>60.6</td><td>62.5</td><td>66.3</td><td>60.6</td><td>66.3</td><td>69.2</td><td>58.7</td><td>60.6</td><td>54.8</td><td>49.0</td><td>62.5</td></tr><tr><td>MultiRC</td><td>acc</td><td>dev</td><td>62.3</td><td>32</td><td>4.72</td><td>9.65</td><td>12.3</td><td>13.6</td><td>14.3</td><td>18.4</td><td>24.2</td><td>27.6</td><td>4.72</td><td>9.65</td><td>12.3</td><td>13.6</td><td>14.3</td><td>18.4</td><td>24.2</td><td>27.6</td><td>6.09</td><td>11.8</td><td>16.8</td><td>20.8</td><td>24.7</td></tr><tr><td>MultiRC</td><td>f1a</td><td>dev</td><td>88.2</td><td>32</td><td>57.0</td><td>59.7</td><td>60.4</td><td>59.9</td><td>60.0</td><td>64.5</td><td>71.4</td><td>72.9</td><td>57.0</td><td>59.7</td><td>60.4</td><td>59.9</td><td>60.0</td><td>64.5</td><td>71.4</td><td>72.9</td><td>45.0</td><td>55.9</td><td>64.2</td><td>65.4</td><td>69.5</td></tr><tr><td>ReCoRD</td><td>acc</td><td>dev</td><td>92.5</td><td>32</td><td>70.8</td><td>78.5</td><td>82.1</td><td>84.1</td><td>86.2</td><td>88.6</td><td>89.0</td><td>90.2</td><td>69.8</td><td>77.0</td><td>80.7</td><td>83.0</td><td>85.9</td><td>88.0</td><td>88.8</td><td>90.2</td><td>69.8</td><td>77.2</td><td>81.3</td><td>83.1</td><td>86.6</td></tr><tr><td>ReCoRD</td><td>f1</td><td>dev</td><td>93.3</td><td>32</td><td>71.9</td><td>79.2</td><td>82.8</td><td>85.2</td><td>87.3</td><td>89.5</td><td>90.4</td><td>91.0</td><td>70.7</td><td>77.8</td><td>81.6</td><td>83.9</td><td>86.8</td><td>88.8</td><td>89.7</td><td>91.2</td><td>70.7</td><td>77.9</td><td>82.1</td><td>84.0</td><td>87.5</td></tr><tr><td>SuperGLUE</td><td>average</td><td>dev</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table H.1: Scores for every task, setting and model that we investigate in this paper.

![](images/963c7108e4353794c994de044c992d23c633aa14aef7651ad487e6dab136bd50.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Fine-tune SOTA | BERT-Large | Random Guessing |
| --------------------------- | -------------- | ---------- | --------------- |
| 0.1B                        | 50             | 50         | 50              |
| 0.4B                        | 60             | 60         | 60              |
| 0.8B                        | 60             | 60         | 60              |
| 1.3B                        | 65             | 65         | 65              |
| 2.6B                        | 70             | 70         | 70              |
| 6.7B                        | 65             | 70         | 70              |
| 13B                         | 65             | 70         | 70              |
| 175B                        | 60             | 75         | 75              |
</details>

![](images/e5233f5657cae205e0ddef172409190034fffd82a5ad6131b3aa2b7028ef9f82.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 0         | 60       | 40              |
| 0.4B                        | 30        | 50       | 60              |
| 0.8B                        | 10        | 50       | 50              |
| 1.3B                        | 20        | 50       | 70              |
| 2.6B                        | 20        | 60       | 70              |
| 6.7B                        | 30        | 30       | 60              |
| 13B                         | 20        | 60       | 70              |
| 175B                        | 45        | 65       | 85              |
</details>

![](images/bc997f62643e866b6bdac63c6d0427cfad686a566c75b4c2f3c7d85f416b99c3.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 0         | 60       | 60              |
| 0.4B                        | 30        | 40       | 40              |
| 0.8B                        | 15        | 50       | 35              |
| 1.3B                        | 20        | 40       | 50              |
| 2.6B                        | 25        | 45       | 45              |
| 6.7B                        | 25        | 30       | 45              |
| 13B                         | 20        | 45       | 45              |
| 175B                        | 40        | 50       | 60              |
</details>

![](images/8dcb361ce997d2434c32af8b805a7ee940fb770950b816dd9fb658897961db5b.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K>32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 68        | 62       | 68              |
| 0.4B                        | 70        | 64       | 68              |
| 0.8B                        | 72        | 66       | 70              |
| 1.3B                        | 74        | 70       | 74              |
| 2.6B                        | 76        | 74       | 82              |
| 6.7B                        | 80        | 76       | 84              |
| 13B                         | 84        | 80       | 86              |
| 175B                        | 90        | 86       | 90              |
</details>

![](images/5e3f0722cb90a8b63b169e426cbbdc68e471c9dd41239c7dc69734d1c75d6947.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 48        | 52       | 50              |
| 0.4B                        | 49        | 49       | 48              |
| 0.8B                        | 56        | 50       | 50              |
| 1.3B                        | 48        | 52       | 56              |
| 2.6B                        | 56        | 54       | 56              |
| 13B                         | 62        | 56       | 60              |
| 175B                        | 64        | 70       | 72              |
</details>

![](images/ab9d685a8f50c9c3aee4ef253c534a3c37c01b7e0f2acbe95fe9d1ea503adaa3.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 0         | 50       | 50              |
| 0.4B                        | 0         | 50       | 55              |
| 0.8B                        | 0         | 50       | 50              |
| 1.3B                        | 0         | 50       | 50              |
| 2.6B                        | 0         | 50       | 50              |
| 6.7B                        | 0         | 50       | 50              |
| 13B                         | 0         | 50       | 50              |
| 17.5B                       | 0         | 50       | 55              |
</details>

![](images/aa2a151c33dd0d2c7dfaf297fce3052ac9cd8360e1bb079ddc7627067219283f.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 60        | 60       | 60              |
| 0.4B                        | 58        | 60       | 60              |
| 0.8B                        | 60        | 60       | 55              |
| 1.3B                        | 60        | 60       | 50              |
| 2.6B                        | 65        | 65       | 60              |
| 6.7B                        | 65        | 65       | 65              |
| 13B                         | 65        | 65       | 75              |
| 175B                        | 65        | 65       | 75              |
</details>

![](images/265548e128869c7cb8cf5b58a025ce1234b1af97d4f3e5463b3b24450f9dbc24.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K<32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 8         | 8        | 8               |
| 0.4B                        | 10        | 10       | 10              |
| 0.8B                        | 12        | 12       | 15              |
| 1.3B                        | 14        | 14       | 20              |
| 2.6B                        | 16        | 16       | 25              |
| 6.7B                        | 20        | 20       | 25              |
| 13B                         | 25        | 25       | 25              |
| 175B                        | 30        | 30       | 30              |
</details>

![](images/65769ec898f73d700f344cc202907e3a40f1f7a6bb451db245219de5b7909da2.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 58        | 45       | 45              |
| 0.4B                        | 60        | 58       | 58              |
| 0.8B                        | 60        | 60       | 65              |
| 1.3B                        | 60        | 60       | 70              |
| 2.6B                        | 60        | 60       | 70              |
| 6.7B                        | 70        | 65       | 65              |
| 13B                         | 72        | 70       | 70              |
| 175B                        | 75        | 75       | 75              |
</details>

![](images/703253514ce393de3277f44efa3d01075420f8433821db489cec131f9443d1b5.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K<32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 70        | 70       | 70              |
| 0.4B                        | 75        | 75       | 75              |
| 0.8B                        | 80        | 80       | 80              |
| 1.3B                        | 85        | 85       | 85              |
| 2.6B                        | 88        | 88       | 88              |
| 6.7B                        | 90        | 90       | 90              |
| 13B                         | 92        | 92       | 92              |
| 175B                        | 93        | 93       | 93              |
</details>

![](images/912c5baed9d0d6d93304cf58af589126ad0c3602bd8387e30a64b980404bed58.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=32) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 70        | 70       | 70              |
| 0.4B                        | 75        | 75       | 75              |
| 0.8B                        | 80        | 80       | 80              |
| 1.3B                        | 85        | 85       | 85              |
| 2.6B                        | 90        | 90       | 90              |
| 6.7B                        | 92        | 92       | 92              |
| 13B                         | 93        | 93       | 93              |
| 175B                        | 94        | 94       | 94              |
</details>

Figure H.1: All results for all SuperGLUE tasks.

![](images/a7417891c4f97f36320257262dfff757b46193b540fc352206f392fff04ef2f4.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=20) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 35        | 30       | 30              |
| 0.4B                        | 38        | 40       | 40              |
| 0.6B                        | 45        | 45       | 42              |
| 1.3B                        | 48        | 50       | 40              |
| 2.6B                        | 50        | 55       | 50              |
| 6.7B                        | 52        | 55       | 52              |
| 13B                         | 53        | 55       | 55              |
| 175B                        | 54        | 60       | 65              |
</details>

Figure H.2: Results for SAT task.

![](images/374500319c68576b38c2929202255701b51c275c089a8917e3cd3882df45053b.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=50) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 50        | 50       | 50              |
| 0.4B                        | 52        | 52       | 52              |
| 0.6B                        | 55        | 55       | 55              |
| 1.3B                        | 58        | 58       | 58              |
| 2.6B                        | 60        | 60       | 60              |
| 6.7B                        | 65        | 65       | 68              |
| 13B                         | 68        | 68       | 70              |
| 175B                        | 70        | 72       | 78              |
</details>

![](images/0ea380e56c849f50ae59ac270c84bf8c9a644073da5209b5e0e778c26842790c.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=7) |
| --------------------------- | --------- | -------- | -------------- |
| 0.1B                        | 65        | 64       | 63             |
| 0.4B                        | 72        | 68       | 67             |
| 0.8B                        | 75        | 73       | 72             |
| 1.3B                        | 78        | 77       | 76             |
| 2.6B                        | 82        | 81       | 85             |
| 6.7B                        | 85        | 84       | 83             |
| 13B                         | 88        | 86       | 85             |
| 175B                        | 90        | 89       | 89             |
</details>

Figure H.3: All results for all Winograd tasks.

![](images/cf94d85e4d28a89266b832bedc6370e15bd7ed6d0d3856a2c61f01fecd61adaa.jpg)

Figure H.4: All results for all Arithmetic tasks.   
![](images/a53f717fa24dfe6b4081ccaef4bf24ee59b05bb8e276734885faa57fdd707819.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K×20) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 30        | 30       | 30              |
| 0.4B                        | 40        | 40       | 40              |
| 0.8B                        | 50        | 50       | 50              |
| 1.3B                        | 60        | 60       | 60              |
| 2.6B                        | 70        | 70       | 70              |
| 6.7B                        | 75        | 75       | 75              |
| 13B                         | 80        | 80       | 80              |
| 175B                        | 85        | 85       | 85              |
</details>

![](images/5db66f6f3c167f82c51b5d718f57756e793f6e161df269832a44c31ca7f0bab2.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=15) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 40        | 20       | 20              |
| 0.4B                        | 50        | 40       | 40              |
| 0.8B                        | 60        | 50       | 60              |
| 1.3B                        | 65        | 55       | 55              |
| 2.6B                        | 70        | 60       | 80              |
| 6.7B                        | 75        | 65       | 80              |
| 13B                         | 75        | 70       | 85              |
| 17B                         | 75        | 75       | 90              |
</details>

![](images/40ec06687bd14ab893367844ff944b615c793c3242d27f398b7f2cbdba3f927a.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=70) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 64        | 64       | 64              |
| 0.4B                        | 68        | 69       | 70              |
| 0.8B                        | 72        | 73       | 75              |
| 1.3B                        | 74        | 75       | 77              |
| 2.6B                        | 77        | 78       | 80              |
| 6.7B                        | 79        | 80       | 82              |
| 175B                        | 83        | 85       | 88              |
</details>

Figure H.5: All results for all Cloze and Completion tasks.

![](images/018acb457d462f95c9c59d5ec8f6ae7943a65e7989db9b6682a915ce5e04befb.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=50) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 64.0      | 64.0     | 64.0            |
| 0.4B                        | 68.0      | 68.0     | 68.0            |
| 0.8B                        | 72.0      | 72.0     | 72.0            |
| 1.3B                        | 74.0      | 74.0     | 74.0            |
| 2.6B                        | 76.0      | 76.0     | 76.0            |
| 6.7B                        | 78.0      | 78.0     | 78.0            |
| 13B                         | 80.0      | 80.0     | 80.0            |
| 175B                        | 82.0      | 82.0     | 82.0            |
</details>

![](images/dc3063033fb788b3a31df8382103ba1c35d41ce61969a63c076f4c5ca6e701ac.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=50) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 28        | 28       | 28              |
| 0.4B                        | 30        | 30       | 30              |
| 0.8B                        | 32        | 32       | 32              |
| 1.3B                        | 36        | 36       | 36              |
| 2.6B                        | 38        | 38       | 38              |
| 6.7B                        | 42        | 42       | 42              |
| 13B                         | 44        | 44       | 44              |
| 175B                        | 52        | 52       | 52              |
</details>

![](images/ba7f3986a412701bdaeb0c99c7afbb0d15a34b630ad4169353e48347989dc758.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=100) |
| --------------------------- | --------- | -------- | ---------------- |
| 0.1B                        | 38        | 38       | 38               |
| 0.4B                        | 42        | 42       | 42               |
| 0.8B                        | 46        | 46       | 48               |
| 1.3B                        | 48        | 48       | 50               |
| 2.6B                        | 52        | 52       | 56               |
| 6.7B                        | 50        | 52       | 56               |
| 13B                         | 56        | 56       | 62               |
| 175B                        | 58        | 58       | 66               |
</details>

Figure H.6: All results for all Common Sense Reasoning tasks.

![](images/e7f16ae6ec0be5b70eab23687ac0bd7ecb5ec3c5e217102919eaee143d3d79d6.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 1.0       | 1.0      | 1.0             |
| 0.4B                        | 2.0       | 3.0      | 4.0             |
| 0.8B                        | 3.0       | 5.0      | 7.0             |
| 1.3B                        | 4.0       | 7.0      | 10.0            |
| 2.6B                        | 5.0       | 9.0      | 13.0            |
| 6.7B                        | 6.0       | 10.0     | 17.0            |
| 12.8B                       | 7.0       | 13.0     | 21.0            |
| 174.6B                      | 14.0      | 22.0     | 30.0            |
</details>

![](images/4a566d7a72cd2d061f4157cd8142cbbea03996043e65315629605dfb56f8727e.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 8         | 8        | 8               |
| 0.4B                        | 12        | 15       | 18              |
| 0.8B                        | 18        | 25       | 30              |
| 1.3B                        | 25        | 35       | 40              |
| 2.6B                        | 35        | 45       | 50              |
| 6.7B                        | 45        | 55       | 60              |
| 13B                         | 55        | 65       | 70              |
| 175B                        | 65        | 70       | 75              |
</details>

![](images/18587641659db8187bd313dd42e9333e2fd467b479a70d5a3e7e7b1d7c150b67.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 2         | 3        | 5               |
| 0.4B                        | 3         | 6        | 12              |
| 0.8B                        | 4         | 9        | 16              |
| 1.3B                        | 5         | 10       | 20              |
| 2.6B                        | 7         | 14       | 25              |
| 6.7B                        | 7         | 15       | 28              |
| 13B                         | 8         | 19       | 34              |
| 175B                        | 14        | 25       | 42              |
</details>

Figure H.7: All results for all QA tasks.   
![](images/541d1b15ddcbbe5bb1d7dda4b7cf07f897879894783121c9f779a9d9ab5e817a.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=5) |
| --------------------------- | --------- | -------- | -------------- |
| 0.1B                        | 20        | 20       | 20             |
| 0.4B                        | 25        | 25       | 25             |
| 0.8B                        | 30        | 30       | 30             |
| 1.3B                        | 30        | 30       | 35             |
| 2.6B                        | 35        | 35       | 40             |
| 6.7B                        | 40        | 40       | 45             |
| 13B                         | 45        | 45       | 50             |
| 175B                        | 50        | 50       | 55             |
</details>

![](images/577ff20132c0781a943a71b37b806e7fa1694cedb40ebdc8e654f082aa64963e.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=10) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 37.0      | 37.5     | 36.5            |
| 0.4B                        | 38.5      | 39.0     | 38.0            |
| 0.8B                        | 40.0      | 40.5     | 39.5            |
| 1.3B                        | 41.5      | 42.0     | 41.0            |
| 2.6B                        | 43.0      | 43.5     | 42.5            |
| 6.7B                        | 44.5      | 45.0     | 44.0            |
| 13B                         | 45.5      | 46.0     | 45.0            |
| 175B                        | 46.5      | 47.0     | 46.0            |
</details>

![](images/77adafd02434f8a90dca5d27bd7d773953a5134d952ab2c19ecbc2969398a9ae.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=10) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 42        | 42       | 42              |
| 0.4B                        | 48        | 48       | 48              |
| 0.8B                        | 52        | 52       | 52              |
| 1.3B                        | 54        | 56       | 54              |
| 2.6B                        | 56        | 56       | 56              |
| 6.7B                        | 54        | 56       | 56              |
| 13B                         | 58        | 58       | 58              |
| 175B                        | 58        | 58       | 58              |
</details>

![](images/e77e00053748ac461f91862b881a4b1bc9b66cf9d189ac02017c239407622c52.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=16) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 30        | 30       | 30              |
| 0.4B                        | 40        | 40       | 45              |
| 0.8B                        | 45        | 45       | 45              |
| 1.3B                        | 50        | 55       | 60              |
| 2.6B                        | 50        | 55       | 55              |
| 6.7B                        | 55        | 60       | 65              |
| 13B                         | 55        | 65       | 70              |
| 175B                        | 60        | 70       | 70              |
</details>

![](images/0e389c6a0c5c5f11fbf91948b0966bc10583a72de48295b5f149f9a9445ba31f.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=5) |
| --------------------------- | --------- | -------- | -------------- |
| 0.1B                        | 30        | 30       | 30             |
| 0.4B                        | 55        | 55       | 55             |
| 0.8B                        | 65        | 65       | 65             |
| 1.3B                        | 70        | 70       | 70             |
| 2.6B                        | 75        | 75       | 75             |
| 6.7B                        | 80        | 80       | 80             |
| 13B                         | 85        | 85       | 85             |
| 175B                        | 90        | 90       | 90             |
</details>

![](images/7ad64be68f2d7e3b583b02648cdd007fc3ac9fcc73b8131a0d2cbdceae099fd3.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=20) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 10        | 10       | 10              |
| 0.4B                        | 12        | 18       | 20              |
| 0.8B                        | 14        | 22       | 24              |
| 1.3B                        | 16        | 24       | 28              |
| 2.6B                        | 18        | 26       | 30              |
| 6.7B                        | 16        | 28       | 30              |
| 13B                         | 22        | 30       | 34              |
| 175B                        | 22        | 34       | 38              |
</details>

Figure H.8: All results for all Reading Comprehension tasks.

![](images/1df6ce563b4eb3e3488719c03eb37f0c820e31222fd16e0e17c5587bebd00843.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=50) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 32        | 32       | 32              |
| 0.4B                        | 32        | 32       | 32              |
| 0.8B                        | 32        | 32       | 32              |
| 1.3B                        | 32        | 32       | 32              |
| 2.6B                        | 32        | 32       | 32              |
| 6.7B                        | 32        | 32       | 32              |
| 13B                         | 32        | 32       | 32              |
| 175B                        | 32        | 32       | 32              |
</details>

![](images/7f5f3765ebe0b65d259e4af31cd33b7a2d3e942365040f8fbe1dc6e3cfeab92d.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=50) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 32.5      | 32.5     | 35.0            |
| 0.4B                        | 32.0      | 32.0     | 34.0            |
| 0.8B                        | 32.5      | 32.5     | 33.0            |
| 1.3B                        | 32.0      | 32.0     | 32.0            |
| 2.6B                        | 33.0      | 33.0     | 32.5            |
| 6.7B                        | 33.5      | 33.5     | 33.0            |
| 13B                         | 33.0      | 33.0     | 32.5            |
| 175B                        | 35.0      | 34.0     | 33.0            |
</details>

![](images/dbaafa4b0a6be6c8066543105372e8aecf45bf3c03b988b704a6d3bf64ba22e1.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Fine-tuned SOTA | Fine-tuned RoBERTa-Large | Fine-tuned BERT-Large |
| --------------------------- | --------------- | ------------------------ | --------------------- |
| 0.1B                        | 34              | 34                       | 34                    |
| 0.4B                        | 34              | 34                       | 34                    |
| 0.8B                        | 34              | 34                       | 34                    |
| 1.3B                        | 34              | 34                       | 36                    |
| 2.6B                        | 34              | 34                       | 32                    |
| 6.7B                        | 34              | 34                       | 34                    |
| 13B                         | 34              | 34                       | 34                    |
| 175B                        | 34              | 34                       | 40                    |
</details>

Figure H.9: All results for all ANLI rounds.

![](images/5f8eb48441af6456ac77033381941e2266bf4dcad59b76e8ea1b486268cdf74d.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=100) |
| --------------------------- | --------- | -------- | ---------------- |
| 0.1B                        | 1.0       | 2.0      | 4.0              |
| 0.4B                        | 1.5       | 4.0      | 9.0              |
| 0.8B                        | 2.0       | 6.0      | 11.0             |
| 1.3B                        | 1.0       | 7.0      | 14.0             |
| 2.6B                        | 1.5       | 6.0      | 17.0             |
| 6.0B                        | 2.0       | 9.0      | 22.0             |
| 13B                         | 2.5       | 15.0     | 28.0             |
| 175B                        | 3.0       | 22.0     | 36.0             |
</details>

![](images/5e5fd224fdd251261cf78a0e7df76fde16010be98f69e399db28b3cc9f532773.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | cycle letters | mid word 1 anagrams | mid word 2 anagrams | random insertion | reversed words |
| --------------------------- | ------------- | -------------------- | -------------------- | ----------------- | -------------- |
| 0.1B                        | 5             | 3                    | 2                    | 1                 | 0              |
| 0.4B                        | 8             | 5                    | 4                    | 2                 | 0              |
| 0.8B                        | 10            | 7                    | 6                    | 3                 | 0              |
| 1.3B                        | 12            | 9                    | 8                    | 5                 | 0              |
| 2.6B                        | 15            | 11                   | 10                   | 8                 | 0              |
| 6.7B                        | 20            | 15                   | 15                   | 12                | 0              |
| 13B                         | 25            | 20                   | 20                   | 25                | 0              |
| 175B                        | 30            | 25                   | 25                   | 30                | 0              |
</details>

![](images/a03596ad609075fb297765b5d08ee36d636cb5d1612e19b509154cadb177a56f.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=100) |
| --------------------------- | --------- | -------- | ---------------- |
| 0.1B                        | 0.0       | 0.0      | 0.0              |
| 0.4B                        | 0.0       | 0.5      | 1.0              |
| 0.8B                        | 0.0       | 1.0      | 2.0              |
| 1.3B                        | 0.0       | 1.5      | 3.0              |
| 2.6B                        | 0.0       | 2.0      | 4.0              |
| 6.7B                        | 0.5       | 3.0      | 5.5              |
| 13B                         | 1.0       | 4.0      | 8.5              |
| 175B                        | 2.0       | 9.0      | 15.0             |
</details>

![](images/6cfe995a5171df0e8213444d2c0cfeeafb26090df0320384a56ddd988aa39239.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=100) |
| --------------------------- | --------- | -------- | ---------------- |
| 0.1B                        | 1.0       | 1.0      | 1.0              |
| 0.4B                        | 1.0       | 2.0      | 3.0              |
| 0.8B                        | 2.0       | 4.0      | 6.0              |
| 1.3B                        | 1.0       | 4.0      | 9.0              |
| 2.6B                        | 2.0       | 6.0      | 12.0             |
| 6.7B                        | 3.0       | 10.0     | 19.0             |
| 13B                         | 4.0       | 15.0     | 26.0             |
| 175B                        | 8.0       | 26.0     | 40.0             |
</details>

![](images/0eb7c48526216e58511b011f07bfb46187ce975cbb724aab49238a9f618341ae.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=100) |
| --------------------------- | --------- | -------- | ---------------- |
| 0.1B                        | 0         | 0        | 0                |
| 0.4B                        | 0         | 0        | 0                |
| 0.8B                        | 0         | 0        | 2                |
| 1.3B                        | 0         | 0        | 4                |
| 2.6B                        | 0         | 0        | 6                |
| 6.7B                        | 0         | 0        | 12               |
| 13B                         | 0         | 8        | 28               |
| 175B                        | 8         | 45       | 68               |
</details>

![](images/fdf92b80cce7620c550f7d298edf98c61c9c1e4fe80d2ffe66ea21ef3c1aa308.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=100) |
| --------------------------- | --------- | -------- | ---------------- |
| 0.1B                        | 0.0       | 0.0      | 0.0              |
| 0.4B                        | 0.0       | 0.0      | 0.05             |
| 0.8B                        | 0.0       | 0.0      | 0.0              |
| 1.3B                        | 0.0       | 0.0      | 0.15             |
| 2.6B                        | 0.0       | 0.05     | 0.25             |
| 6.7B                        | 0.0       | 0.07     | 0.3              |
| 13B                         | 0.0       | 0.1      | 0.4              |
| 175B                        | 0.1       | 0.5      | 0.45             |
</details>

Figure H.10: All results for all Scramble tasks.   
![](images/55e6b4539b3a4f9ab11ce5e1160d67870d34bf501d58f901269bd0315c53c22a.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 2         | 2        | 2               |
| 0.4B                        | 3         | 17       | 24              |
| 0.8B                        | 3         | 23       | 28              |
| 1.3B                        | 3         | 26       | 31              |
| 2.6B                        | 22        | 29       | 34              |
| 6.7B                        | 18        | 32       | 37              |
| 13B                         | 24        | 34       | 39              |
| 175B                        | 29        | 32       | 42              |
</details>

![](images/0e77a6ed758b4881a8ad53fec51f9033cada59af435676e1b80b1739554709b5.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 1.0       | 1.0      | 4.0             |
| 0.4B                        | 2.0       | 8.0      | 13.0            |
| 0.8B                        | 2.0       | 14.0     | 17.0            |
| 1.3B                        | 2.0       | 14.0     | 20.0            |
| 2.6B                        | 14.0      | 21.0     | 24.0            |
| 6.7B                        | 9.0       | 22.0     | 27.0            |
| 13B                         | 11.0      | 24.0     | 28.0            |
| 175B                        | 25.0      | 27.0     | 31.0            |
</details>

![](images/c6f2c942d1d4b047131514522ff4ecfc0f04091de7c232d90f67027831f73705.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 2         | 2        | 5               |
| 0.4B                        | 2         | 10       | 18              |
| 0.8B                        | 3         | 18       | 25              |
| 1.3B                        | 2         | 20       | 30              |
| 2.6B                        | 19        | 25       | 32              |
| 6.7B                        | 11        | 28       | 35              |
| 13B                         | 15        | 30       | 37              |
| 175B                        | 31        | 34       | 40              |
</details>

![](images/b8009393d1dc583a674f6a483eb1ae43d3238fd38c40b4912bcb2c548dd2aa0a.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot |
| --------------------------- | --------- | -------- | -------- |
| 0.1B                        | 0         | 0        | 0        |
| 0.4B                        | 2         | 15       | 25       |
| 0.8B                        | 3         | 25       | 30       |
| 1.3B                        | 2         | 28       | 35       |
| 2.6B                        | 21        | 30       | 36       |
| 6.7B                        | 15        | 31       | 37       |
| 13B                         | 22        | 31       | 39       |
| 175B                        | 21        | 35       | 41       |
</details>

![](images/18dcdaf8b67beaed89ebc7aab12c6e7419de2495e6c4b02c2d070186207f33a0.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 2         | 1        | 2               |
| 0.4B                        | 3         | 4        | 8               |
| 0.8B                        | 3         | 9        | 12              |
| 1.3B                        | 3         | 11       | 15              |
| 2.6B                        | 4         | 16       | 18              |
| 6.7B                        | 5         | 19       | 20              |
| 13B                         | 6         | 21       | 22              |
| 175B                        | 18        | 25       | 25              |
</details>

![](images/cb16e14de33809a121b3fa492e4599da3ea973b466764ecdb3a75ef52cbd1e6f.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 2         | 2        | 2               |
| 0.4B                        | 3         | 15       | 20              |
| 0.8B                        | 3         | 25       | 28              |
| 1.3B                        | 3         | 30       | 32              |
| 2.6B                        | 17        | 32       | 35              |
| 6.7B                        | 9         | 35       | 37              |
| 13B                         | 20        | 37       | 39              |
| 175B                        | 20        | 40       | 41              |
</details>

![](images/eea4619bbdbd24fa50a7165948808eac9edd76d273f34173881a0a8cda957c81.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot |
| --------------------------- | --------- | -------- | -------- |
| 0.1B                        | 2         | 2        | 2        |
| 0.4B                        | 3         | 15       | 22       |
| 0.8B                        | 3         | 22       | 27       |
| 2.6B                        | 21        | 28       | 32       |
| 6.7B                        | 16        | 30       | 35       |
| 13B                         | 23        | 33       | 37       |
| 175B                        | 27        | 30       | 40       |
</details>

![](images/4b055728293bb47b90a5d06b9d154bffc886402f500a6277fe10a8c3f0e5848f.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 1         | 1        | 2               |
| 0.4B                        | 2         | 7        | 12              |
| 0.8B                        | 2         | 13       | 16              |
| 1.3B                        | 2         | 13       | 18              |
| 2.6B                        | 13        | 18       | 21              |
| 6.7B                        | 9         | 21       | 24              |
| 13B                         | 10        | 22       | 27              |
| 175B                        | 25        | 26       | 30              |
</details>

![](images/ecb7683b36e4e4acc8ca02b6f2f4d455826386a46fb46f5d54c0dfc2fcfe198b.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 0         | 0        | 0               |
| 0.4B                        | 2         | 8        | 14              |
| 0.8B                        | 2         | 14       | 20              |
| 1.3B                        | 2         | 16       | 22              |
| 2.6B                        | 15        | 20       | 25              |
| 6.7B                        | 9         | 24       | 27              |
| 13B                         | 12        | 25       | 29              |
| 17.5B                       | 25        | 28       | 32              |
</details>

![](images/62e83bf15d7bb1705ff32520de7dcda07139b6aeb864c12fde2fd5dd5523ab25.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot |
| --------------------------- | --------- | -------- | -------- |
| 0.1B                        | 0         | 0        | 0        |
| 0.4B                        | 2         | 15       | 25       |
| 0.8B                        | 3         | 25       | 30       |
| 1.3B                        | 2         | 28       | 33       |
| 2.6B                        | 20        | 30       | 35       |
| 6.7B                        | 15        | 30       | 36       |
| 13B                         | 22        | 30       | 37       |
| 175B                        | 20        | 34       | 39       |
</details>

![](images/5872ecf628a2595a5005977c3af8d93473d57f6784210997da0f2e23f070df70.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 2         | 1        | 1               |
| 0.4B                        | 3         | 3        | 5               |
| 0.8B                        | 2         | 8        | 9               |
| 1.3B                        | 2         | 9        | 11              |
| 2.6B                        | 3         | 13       | 14              |
| 6.7B                        | 4         | 15       | 16              |
| 13B                         | 5         | 17       | 18              |
| 175B                        | 14        | 20       | 20              |
</details>

![](images/ddb73886de079135e40faba1775dc212818bdc327ad69a0a1a6035c676935f24.jpg)

<details>
<summary>line</summary>

| Parameters in LM (Billions) | Zero-Shot | One-Shot | Few-Shot (K=64) |
| --------------------------- | --------- | -------- | --------------- |
| 0.1B                        | 0         | 0        | 0               |
| 0.4B                        | 2         | 15       | 20              |
| 0.8B                        | 3         | 22       | 25              |
| 1.3B                        | 4         | 28       | 30              |
| 2.6B                        | 16        | 30       | 33              |
| 6.7B                        | 8         | 32       | 35              |
| 13B                         | 20        | 35       | 37              |
| 175B                        | 20        | 38       | 39              |
</details>

Figure H.11: All results for all Translation tasks.

# References

[ADG $^{+}$ 16] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. In Advances in neural information processing systems, pages 3981–3989, 2016.   
[AI19] WeChat AI. Tr-mt (ensemble), December 2019.   
[AJF19] Roee Aharoni, Melvin Johnson, and Orhan Firat. Massively multilingual neural machine translation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019.   
[BBDIW20] Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. Language (technology) is power: A critical survey of “bias” in nlp. arXiv preprint arXiv:2005.14050, 2020.   
[BCFL13] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from question-answer pairs. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1533–1544, 2013.   
[BDD $^{+}$ 09] Luisa Bentivogli, Ido Dagan, Hoa Trang Dang, Danilo Giampiccolo, and Bernardo Magnini. The fifth PASCAL recognizing textual entailment challenge. 2009.   
[BES10] Stefano Baccianella, Andrea Esuli, and Fabrizio Sebastiani. Sentiwordnet 3.0: an enhanced lexical resource for sentiment analysis and opinion mining. In Lrec, volume 10, pages 2200–2204, 2010.   
[BHDD $^{+}$ 06] Roy Bar Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor. The second PASCAL recognising textual entailment challenge. 2006.   
[BHT $^{+}$ 20] Yonatan Bisk, Ari Holtzman, Jesse Thomason, Jacob Andreas, Yoshua Bengio, Joyce Chai, Mirella Lapata, Angeliki Lazaridou, Jonathan May, Aleksandr Nisnevich, et al. Experience grounds language. arXiv preprint arXiv:2004.10151, 2020.   
[BLC13] Yoshua Bengio, Nicholas Léonard, and Aaron C. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. Arxiv, 2013.   
[BZB $^{+}$ 19] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. arXiv preprint arXiv:1911.11641, 2019.   
[Car97] Rich Caruana. Multitask learning. Machine learning, 28(1), 1997.   
[CB78] Susan Carey and Elsa Bartlett. Acquiring a single new word. Proceedings of the Stanford Child Language Conference, 1978.   
[CCE $^{+}$ 18] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457, 2018.   
[CGRS19] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers, 2019.   
[CHI $^{+}$ 18] Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. Quac : Question answering in context. Arxiv, 2018.   
[CLC $^{+}$ 19] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.   
[CLY $^{+}$ 19] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Learning universal image-text representations. arXiv preprint arXiv:1909.11740, 2019.   
[Cra17] Kate Crawford. The trouble with bias. NIPS 2017 Keynote, 2017.   
[DCLT18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[DGM06] Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In Machine learning challenges. evaluating predictive uncertainty, visual object classification, and recognising textual entailment, pages 177–190. Springer, 2006.   
[ $DGV^{+}$ 18] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. Arxiv, 2018.   
[DHKH14] Nadir Durrani, Barry Haddow, Philipp Koehn, and Kenneth Heafield. Edinburgh's phrase-based machine translation systems for wmt-14. In Proceedings of the Ninth Workshop on Statistical Machine Translation, pages 97–104, 2014.   
[DL15] Andrew M. Dai and Quoc V. Le. Semi-supervised sequence learning. In Advances in neural information processing systems, 2015.   
[DMST19] Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The CommitmentBank: Investigating projection in naturally occurring discourse. 2019. To appear in proceedings of Sinn und Bedeutung 23. Data can be found at https://github.com/mcdm/CommitmentBank/.   
[DSC $^{+}$ 16] Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl $^{2}$ : Fast reinforcement learning via slow reinforcement learning. ArXiv, abs/1611.02779, 2016.   
[DWD $^{+}$ 19] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. arXiv preprint arXiv:1903.00161, 2019.   
[DYY $^{+}$ 19] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. Arxiv, 2019.   
[EOAG18] Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. Understanding back-translation at scale. arXiv preprint arXiv:1808.09381, 2018.   
[FAL17] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. ArXiv, abs/1703.03400, 2017.   
[Fyo00] Yaroslav Fyodorov. A natural logic inference system, 2000.   
[GG19] Hila Gonen and Yoav Goldberg. Lipstick on a pig: Debiasing methods cover up systematic gender biases in word embeddings but do not remove them. arXiv preprint arXiv:1903.03862, 2019.   
[GLT $^{+}$ 20] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm: Retrieval-augmented language model pre-training. arXiv preprint arXiv:2002.08909, 2020.   
[GMDD07] Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL recognizing textual entailment challenge. In Proceedings of the ACL-PASCAL workshop on textual entailment and paraphrasing, pages 1–9. Association for Computational Linguistics, 2007.   
[Gra16] Alex Graves. Adaptive computation time for recurrent neural networks. Arxiv, 2016.   
[ $GSL^{+}$ 18] Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R Bowman, and Noah A Smith. Annotation artifacts in natural language inference data. arXiv preprint arXiv:1803.02324, 2018.   
[GSR19] Sebastian Gehrmann, Hendrik Strobelt, and Alexander M. Rush. Gltr: Statistical detection and visualization of generated text. arXiv preprint arXiv: 1906.04043, 2019.   
[GWC $^{+}$ 18] Jiatao Gu, Yong Wang, Yun Chen, Kyunghyun Cho, and Victor OK Li. Meta-learning for low-resource neural machine translation. arXiv preprint arXiv:1808.08437, 2018.   
[HB20] Daniel Hernandez and Tom Brown. Ai and efficiency, May 2020.   
[HBFC19] Ari Holtzman, Jan Buys, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. CoRR, abs/1904.09751, 2019.   
[HLW $^{+}$ 20] Dan Hendrycks, Xiaoyuan Liu, Eric Wallace, Adam Dziedzic, Rishabh Krishnan, and Dawn Song. Pretrained transformers improve out of distribution robustness. arXiv preprint arXiv:2004.06100, 2020.

[HNA $^{+}$ 17] Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.   
[HR18] Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018.   
[HVD15] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.   
[HYC01] Sepp Hochreiter, A Steven Younger, and Peter R Conwell. Learning to Learn Using Gradient Descent. In International Conference on Artificial Neural Networks, pages 87–94. Springer, 2001.   
$\left[\mathrm{HZJ}^{+}19\right]$ Po-Sen Huang, Huan Zhang, Ray Jiang, Robert Stanforth, Johannes Welbl, Jack Rae, Vishal Maini, Dani Yogatama, and Pushmeet Kohli. Reducing sentiment bias in language models via counterfactual evaluation. arXiv preprint arXiv:1911.03064, 2019.   
[IBGC $^{+}$ 14] Mohit Iyyer, Jordan Boyd-Graber, Leonardo Claudino, Richard Socher, and Hal Daumé III. A neural network for factoid question answering over paragraphs. In Empirical Methods in Natural Language Processing, 2014.   
[IDCBE19] Daphne Ippolito, Daniel Duckworth, Chris Callison-Burch, and Douglas Eck. Automatic detection of generated text is easiest when humans are fooled. arXiv preprint arXiv:1911.00650, 2019.   
[JCWZ17] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.   
[JN20] Zheng Junyuan and Gamma Lab NYC. Numeric transformer - albert, March 2020.   
[JVS $^{+}$ 16] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.   
[JYS $^{+}$ 19] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. TinyBERT: Distilling BERT for natural language understanding. arXiv preprint arXiv:1909.10351, 2019.   
[JZC $^{+}$ 19] Ying Ju, Fubang Zhao, Shijie Chen, Bowen Zheng, Xuefeng Yang, and Yunfeng Liu. Technical report on conversational question answering. arXiv preprint arXiv:1909.10772, 2019.   
[KCR $^{+}$ 18] Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth. Looking beyond the surface: A challenge set for reading comprehension over multiple sentences. In Proceedings of North American Chapter of the Association for Computational Linguistics (NAACL), 2018.   
[KKS $^{+}$ 20] Daniel Khashabi, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. Unifiedqa: Crossing format boundaries with a single qa system. arXiv preprint arXiv:2005.00700, 2020.   
[KMB20] Sarah E. Kreps, Miles McCain, and Miles Brundage. All the news that's fit to fabricate: Ai-generated text as a tool of media misinformation, 2020.   
[KMH $^{+}$ 20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020.   
[KPR $^{+}$ 19] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Transactions of the Association of Computational Linguistics, 2019.   
[KR16] Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. Arxiv, 2016.   
[LB02] Edward Loper and Steven Bird. Nltk: The natural language toolkit, 2002.   
[LC19] Guillaume Lample and Alexis Conneau. Cross-lingual language model pretraining. arXiv preprint arXiv:1901.07291, 2019.

[LCG $^{+}$ 19] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942, 2019.   
[LCH $^{+}$ 20] Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon, and Jianfeng Gao. Adversarial training for large neural language models. arXiv preprint arXiv:2004.08994, 2020.   
[LDL19] Zhongyang Li, Xiao Ding, and Ting Liu. Story ending prediction by transferable bert. arXiv preprint arXiv:1905.07504, 2019.   
[LDM12] Hector Levesque, Ernest Davis, and Leora Morgenstern. The Winograd schema challenge. In Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning, 2012.   
[LGG $^{+}$ 20] Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. Multilingual denoising pre-training for neural machine translation. arXiv preprint arXiv:2001.08210, 2020.   
[LGH $^{+}$ 15] Xiaodong Liu, Jianfeng Gao, Xiaodong He, Li Deng, Kevin Duh, and Ye-Yi Wang. Representation learning using multi-task deep neural networks for semantic classification and information retrieval. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2015.   
[LH17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[LHCG19a] Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. Improving multi-task deep neural networks via knowledge distillation for natural language understanding. arXiv preprint arXiv:1904.09482, 2019.   
[LHCG19b] Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. Multi-task deep neural networks for natural language understanding. arXiv preprint arXiv:1901.11504, 2019.   
[Lin20] Tal Linzen. How can we accelerate progress towards human-like linguistic generalization? arXiv preprint arXiv:2005.00955, 2020.   
[LLG $^{+}$ 19] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.   
[LM17] Ke Li and Jitendra Malik. Learning to optimize neural nets. arXiv preprint arXiv:1703.00441, 2017.   
[LOG $^{+}$ 19] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019.   
[LPP $^{+}$ 20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Kiela Douwe. Retrieval-augmented generation for knowledge-intensive nlp tasks. arXiv preprint arXiv:2005.11401, 2020.   
[LSP $^{+}$ 18] Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. Generating Wikipedia by summarizing long sequences. arXiv preprint arXiv:1801.10198, 2018.   
[LWS $^{+}$ 20] Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, and Joseph E. Gonzalez. Train large, then compress: Rethinking model size for efficient training and inference of transformers, 2020.   
[LXL $^{+}$ 17] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017.   
[LYN $^{+}$ 20] Sheng-Chieh Lin, Jheng-Hong Yang, Rodrigo Nogueira, Ming-Feng Tsai, Chuan-Ju Wang, and Jimmy Lin. Tttttackling winogrande schemas. arXiv preprint arXiv:2003.08380, 2020.   
[Mac92] David. MacKay. Information-based objective functions for active data selection. Neural Computation, 1992.

[MBXS17] Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems, pages 6294–6305, 2017.   
[MCCD13] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.   
[MCH $^{+}$ 16] Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. A corpus and evaluation framework for deeper understanding of commonsense stories. arXiv preprint arXiv:1604.01696, 2016.   
[MCKS18] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. ArXiv, abs/1809.02789, 2018.   
[MKAT18] Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training, 2018.   
[MKM $^{+}$ 94] Mitchell Marcus, Grace Kim, Mary Ann Marcinkiewicz, Robert MacIntyre, Ann Bies, Mark Ferguson, Karen Katz, and Britta Schasberger. The penn treebank: annotating predicate argument structure. In Proceedings of the workshop on Human Language Technology, pages 114–119. Association for Computational Linguistics, 1994.   
[MKXS18] Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering. arXiv preprint arXiv:1806.08730, 2018.   
[MPL19] R Thomas McCoy, Ellie Pavlick, and Tal Linzen. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. arXiv preprint arXiv:1902.01007, 2019.   
[MWZ $^{+}$ 18] Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. Model cards for model reporting, 2018.   
[NBR20] Moin Nadeem, Anna Bethke, and Siva Reddy. Stereoset: Measuring stereotypical bias in pretrained language models. arXiv preprint arXiv:2004.09456, 2020.   
[NK19] Timothy Niven and Hung-Yu Kao. Probing neural network comprehension of natural language arguments. arXiv preprint arXiv:1907.07355, 2019.   
[Nor09] Peter Norvig. Natural language corpus data, 2009.   
[NvNvdG19] Malvina Nissim, Rik van Noord, and Rob van der Goot. Fair is better than sensational: Man is to doctor as woman is to doctor. arXiv preprint arXiv:1905.09866, 2019.   
[NWD $^{+}$ 19] Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. Adversarial nli: A new benchmark for natural language understanding. arXiv preprint arXiv:1910.14599, 2019.   
[oR16] University of Regensburg. Fascha, 2016.   
[PCC18] Mohammad Taher Pilehvar and Jose Camacho-Collados. WIC: 10,000 example pairs for evaluating context-sensitive representations. arXiv preprint arXiv:1808.09121, 2018.   
[PFB18] Jason Phang, Thibault Févry, and Samuel R. Bowman. Sentence encoders on STILTs: Supplementary training on intermediate labeled-data tasks. arXiv preprint arXiv:1811.01088, 2018.   
[PHR $^{+}$ 18] Adam Poliak, Aparajita Haldar, Rachel Rudinger, J. Edward Hu, Ellie Pavlick, Aaron Steven White, and Benjamin Van Durme. Collecting diverse natural language inference problems for sentence representation evaluation. In Proceedings of EMNLP, 2018.   
[PKL $^{+}$ 16] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.   
[PNZtY18] Matthew E. Peters, Mark Neumann, Luke Zettlemoyer, and Wen tau Yih. Dissecting contextual word embeddings: Architecture and representation, 2018.   
[Pos18] Matt Post. A call for clarity in reporting BLEU scores. arXiv preprint arXiv:1804.08771, 2018.

[PSM14] Jeffrey Pennington, Richard Socher, and Christopher Manning. GloVe: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), 2014.   
[QIA20] QIANXIN. Sa-net on albert (ensemble), April 2020.   
[QMZH19] Yusu Qian, Urwa Muaz, Ben Zhang, and Jae Won Hyun. Reducing gender bias in word-level language models with a gender-equalizing loss function. arXiv preprint arXiv:1905.12801, 2019.   
[RBG11] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI Spring Symposium Series, 2011.   
[RCM19] Siva Reddy, Danqi Chen, and Christopher D Manning. Coqa: A conversational question answering challenge. Transactions of the Association for Computational Linguistics, 7:249–266, 2019.   
[RCP+17] Scott Reed, Yutian Chen, Thomas Paine, Aäron van den Oord, SM Eslami, Danilo Rezende, Oriol Vinyals, and Nando de Freitas. Few-shot autoregressive density estimation: Towards learning to learn distributions. arXiv preprint arXiv:1710.10304, 2017.   
[RJL18] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for squad. arXiv preprint arXiv:1806.03822, 2018.   
[RL16] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. ICLR 2017 (oral), 2016.   
[RLL $^{+}$ 19] Qiu Ran, Yankai Lin, Peng Li, Jie Zhou, and Zhiyuan Liu. NumNet: Machine reading comprehension with numerical reasoning. In Proceedings of EMNLP, 2019.   
[RNLVD18] Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. Gender bias in coreference resolution. arXiv preprint arXiv:1804.09301, 2018.   
[RNSS18] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training, 2018.   
[Ros12] R.S. Ross. Guide for conducting risk assessments. NIST Special Publication, 2012.   
[RRBS19] Jonathan S. Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive prediction of the generalization error across scales, 2019.   
[RRS20] Adam Roberts, Colin Raffel, and Noam Shazeer. How much knowledge can you pack into the parameters of a language model? arXiv preprint arXiv:2002.08910, 2020.   
[RSR $^{+}$ 19] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2019.   
[RWC $^{+}$ 19] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019.   
[SBBC19] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale, 2019.   
[SBC $^{+}$ 19] Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herbert-Voss, Jeff Wu, Alec Radford, Gretchen Krueger, Jong Wook Kim, Sarah Kreps, Miles McCain, Alex Newhouse, Jason Blazakis, Kris McGuffie, and Jasmine Wang. Release strategies and the social impacts of language models, 2019.   
[SCNP19] Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, and Nanyun Peng. The woman worked as a babysitter: On biases in language generation. arXiv preprint arXiv:1909.01326, 2019.   
[SDCW19] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.   
[SDSE19] Roy Schwartz, Jesse Dodge, Noah A. Smith, and Oren Etzioni. Green AI. CoRR, abs/1907.10597, 2019.   
[SHB15] Rico Sennrich, Barry Haddow, and Alexandra Birch. Improving neural machine translation models with monolingual data. arXiv preprint arXiv:1511.06709, 2015.

[SMM $^{+}$ 17] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.   
[SPP $^{+}$ 19] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2019.   
[SS20] Timo Schick and Hinrich Schütze. Exploiting cloze questions for few-shot text classification and natural language inference. arXiv preprint arXiv:2001.07676, 2020.   
[STQ $^{+}$ 19] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. MASS: Masked sequence to sequence pre-training for language generation. arXiv preprint arXiv:1905.02450, 2019.   
$\left[\mathrm{TFR}^{+}17\right]$ Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 23-30. IEEE, 2017.   
[TL05] Peter D. Turney and Michael L. Littman. Corpus-based learning of analogies and semantic relations. CoRR, abs/cs/0508103, 2005.   
[TL18] Trieu H. Trinh and Quoc V. Le. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847, 2018.   
[TLBS03] Peter D. Turney, Michael L. Littman, Jeffrey Bigham, and Victor Shnayder. Combining independent modules to solve multiple-choice synonym and analogy problems. CoRR, cs.CL/0309035, 2003.   
[Tur20] Project Turing. Microsoft research blog, Feb 2020.   
[VBL $^{+}$ 16] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching Networks for One Shot Learning. In Advances in neural information processing systems, pages 3630–3638, 2016.   
[VSP $^{+}$ 17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, 2017.   
[WPN $^{+}$ 19] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems, pages 3261–3275, 2019.   
[WXH $^{+}$ 18] Yiren Wang, Yingce Xia, Tianyu He, Fei Tian, Tao Qin, ChengXiang Zhai, and Tie-Yan Liu. Multi-agent dual learning. ICLR 2019, 2018.   
[XDH $^{+}$ 19] Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le. Unsupervised data augmentation for consistency training, 2019.   
[YdC $^{+}$ 19] Dani Yogatama, Cyprien de Masson d'Autume, Jerome Connor, Tomas Kocisky, Mike Chrzanowski, Lingpeng Kong, Angeliki Lazaridou, Wang Ling, Lei Yu, Chris Dyer, et al. Learning and evaluating general linguistic intelligence. arXiv preprint arXiv:1901.11373, 2019.   
[YDY $^{+}$ 19] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. XLNet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237, 2019.   
[ZHB $^{+}$ 19] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.   
[ZHR $^{+}$ 19] Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, and Yejin Choi. Defending against neural fake news. arXiv preprint arXiv:1905.12616, 2019.   
[ZLL $^{+}$ 18] Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, and Benjamin Van Durme. ReCoRD: Bridging the gap between human and machine commonsense reading comprehension. arXiv preprint arXiv:1810.12885, 2018.   
[ZSW $^{+}$ 19a] Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences, 2019.

[ZSW $^{+}$ 19b] Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. ArXiv, abs/1909.08593, 2019.