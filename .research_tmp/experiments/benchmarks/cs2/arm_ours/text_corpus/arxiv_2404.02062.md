Digital Forgetting in Large Language Models: A Survey of Unlearning Methods 
 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.02062v1 [cs.CR] 02 Apr 2024 
 
 

# Digital Forgetting in Large Language Models: A Survey of Unlearning Methods

 
 
 Alberto Blanco-Justicia
 
 Affiliation: Universitat Rovira i VirgiliDepartment of Computer Engineering and MathematicsCYBERCAT-Center for Cybersecurity Research of CataloniaAv. Països Catalans 26, 43007 Tarragona, Catalonia.
 
    
 Najeeb Jebreel
 
 Affiliation: Universitat Rovira i VirgiliDepartment of Computer Engineering and MathematicsCYBERCAT-Center for Cybersecurity Research of CataloniaAv. Països Catalans 26, 43007 Tarragona, Catalonia.
 
    
 Benet Manzanares
 
 Affiliation: Universitat Rovira i VirgiliDepartment of Computer Engineering and MathematicsCYBERCAT-Center for Cybersecurity Research of CataloniaAv. Països Catalans 26, 43007 Tarragona, Catalonia.
 
    
 David Sánchez * 
 
 Affiliation: Universitat Rovira i VirgiliDepartment of Computer Engineering and MathematicsCYBERCAT-Center for Cybersecurity Research of CataloniaAv. Països Catalans 26, 43007 Tarragona, Catalonia.
 
    
 Josep Domingo-Ferrer * 
 
 Affiliation: Universitat Rovira i VirgiliDepartment of Computer Engineering and MathematicsCYBERCAT-Center for Cybersecurity Research of CataloniaAv. Països Catalans 26, 43007 Tarragona, Catalonia.
 
    
 Guillem Collell
 
 Affiliation: Huawei Technologies Finland Research CenterItämerenkatu 9, Helsinki, FI-00180 Finland. * Corresponding authors: {david.sanchez, josep.domingo}@urv.cat
 
    
 Kuan Eeik Tan
 
 Affiliation: Huawei Technologies Finland Research CenterItämerenkatu 9, Helsinki, FI-00180 Finland. * Corresponding authors: {david.sanchez, josep.domingo}@urv.cat
 
 August 24, 2026 

 

 
 

## 1 Introduction

 
 Large language models have become the state of the art in most if not all
natural language processing (NLP) and natural language understanding (NLU) tasks.
Since the publication of the transformer architecture by Vaswani et al., (2017) ,
several authors have made use of this architecture, or variations of it, to
tackle tasks such as translation, summarization, question answering, sentiment
analysis, or text generation.
Since the announcement and publication of ChatGPT by OpenAI in November 2022,
which brought the LLMs capabilities to a broad audience, several issues have
been raised, mainly concerned with the alignment of such models to
societal values and the rule of law 1 1 
 1 
 
 
 
 While this document is dedicated
to LLMs, similar issues have been raised in regard to all generative ML models,
such as image or voice generation. .
Such concerns include the impact of these models on the labor market,
on the right to privacy of individuals, on copyright laws, on the furthering
of biases and discrimination, and on the potential generation of harmful content,
including content that could be used to damage people.

 
 
 One proposed solution to these issues is that of digital forgetting. The objective
of digital forgetting is, given a model with undesirable knowledge or behavior, obtain a new model where the
detected issues are no longer present. However, effective digital forgetting
mechanisms have to fulfill potentially conflicting requirements: the effectiveness
of forgetting, that is how well the new model has forgotten the undesired knowledge/behavior
(either with formal guarantees or through empirical evaluation); the retained
performance of the model on the desirable tasks; and the timeliness and scalability of the forgetting procedure.

 
 
 This document is organized as follows.
Section 2 provides a background on LLMs, including their components,
the types of LLMs, and their usual training pipeline.
Section 3 describes the motivations, types, and desired
properties of digital forgetting.
Section 4 introduces the approaches to digital forgetting in LLMs, among which unlearning methodologies stand out as the state of the art.
Section 5 provides a detailed taxonomy of machine unlearning methods for LLMs, and surveys and compares current approaches.
Section 6 details datasets, models and metrics
used for the evaluation of forgetting, retaining and runtime.
Section 7 discusses challenges in the area.
Finally, we provide some concluding remarks in Section 8 .

 
 
 

## 2 Background on large language models

 
 (Large) Language models (LLM) are statistical models that assign probabilities to sequences of words.
These models are currently the state of the art in many natural language processing (NLP)
and understanding (NLU) tasks, such as summarization, translation, question answering, sentiment
analysis, text generation, or chatbots, among many other applications.

 
 
 The first and most popular transformer architecture was presented by Google engineers
 Vaswani et al., in 2017 .
It was introduced as an encoder-decoder architecture for language translation, but it has
since become the main building block for the current (generative) LLM ecosystem.

 
 
 Such encoder-decoder architectures were first proposed by Cho et al., (2014) , using recurrent
neural networks (RNNs). RNNs were substituted by long short-term memories (LSTMs) in seq2seq by Sutskever et al., (2014) .
The main limitation of RNN/LSTM-based language models was that these models struggled to analyze or
maintain long-term relationships between words in sequences.
 Bahdanau et al., (2014) added an initial differentiable attention mechanism to the encoder-decoder
RNN architecture.
Finally, in Attention is all you need ( Vaswani et al.,, 2017 ) , RNNs were dropped
in favor of using only the attention mechanism, which gave rise to the transformer model.
Since the introduction of the transformer model in 2017, it has become the main architecture
for (large) language models.

 
 

### 2.1 Components of the transformer architecture

 
 The original transformer model, shown in Figure 1 ,
consisted of an encoder-decoder architecture.
The first components are embedding layers which take in tokens
from the input and output sequences. The resulting embeddings are then passed to the encoder
and decoder components, respectively.
The encoder consists of a number of blocks, each one consisting of two subcomponents: a self-attention layer followed by a feed-forward network. A residual skip connection
adds the output of each subcomponent to its input. In the original transformer
architecture, the outputs of the additions were normalized.
Since the work by Xiong et al., (2020) , normalization is applied before each subcomponent.
The decoder is similar in structure, but includes an additional self-attention layer
that attends to the outputs of the encoder.
The first of the self-attention layers in each block of the decoder is
masked, that is, it only attends to previous tokens (it is causal or autoregressive).
The output of the final decoder block is passed to a dense layer with a softmax output.
Since self-attention is order-independent, position encodings are provided (added) to
the embeddings of the input tokens.

 
 
 
 
 
 Input 
 Embedding 
 
 
 
 Output 
 Embedding 
 
 
 
 Add Norm 
 
 
 
 Multi-Head 
 Attention 
 
 
 
 Add Norm 
 
 
 
 Multi-Head 
 Attention 
 
 
 
 Add Norm 
 
 
 
 Masked 
 Multi-Head 
 Attention 
 
 
 
 Add Norm 
 
 
 
 Feed 
 Forward 
 
 
 
 Add Norm 
 
 
 
 Feed 
 Forward 
 
 
 
 Linear 
 
 
 
 Softmax 
 
 
 
 Inputs 
 
 
 
 Outputs 
 (shifted right) 
 
 
 
 Output 
 Probabilities 
 N × N\times N × N\times 
 
 
 Positional 
 Encoding 
 
 
 
 Positional 
 Encoding 
 
 
 Figure 1: The transformer architecture. Note that, since Xiong et al., (2020) , layer normalization is typically applied before the attention and the feed-forward layers instead of after addition. Image source extracted from https://github.com/negrinho/sane_tikz/blob/master/examples/transformer.tex under the MIT licence. 
 
 
 We describe below the components of the transformer architecture. Details on such components will be needed later when describing some unlearning mechanisms and privacy attacks.

 
 
 Tokenization . While not exclusive to the transformer model, tokenizers are a central component of most LLMs. LLMs have a limitation on the amount of different words they can deal with. The size of the embedding and the output layers directly depends
on the number of different words of the vocabulary in use. In an attempt to minimize the size of such model components (and therefore the memory requirements, training, and inference times), natural language is tokenized in single words or fragments of words. For example, if we have the vocabulary {quick, quicker, quickest, fast, faster, fastest}, consisting of 6 6 different words, a possible tokenization would be {quick, fast, er, est}, of size 4 4 . Combining the tokens allows us to build the same words as in the original vocabulary, but fewer different tokens are required. The objective of a tokenizer is to cover as much as possible of the target language vocabulary while minimizing the number of tokens. In the following, we will use the terms words and tokens interchangeably.

 
 
 Embeddings ( Mikolov et al.,, 2013 ) .
Word embeddings are numerical vector representations of words or tokens. These representations allow for mathematical operations to be applied to words, including distance computations, additions, and subtractions. A notable example of these operations is v ​ e ​ c ​ t ​ o ​ r ​ ( K ​ i ​ n ​ g ) − v ​ e ​ c ​ t ​ o ​ r ​ ( M ​ a ​ n ) + v ​ e ​ c ​ t ​ o ​ r ​ ( W ​ o ​ m ​ a ​ n ) vector(King)-vector(Man)+vector(Woman) , which results in a vector representation close to v ​ e ​ c ​ t ​ o ​ r ​ ( Q ​ u ​ e ​ e ​ n ) vector(Queen) . An advantage of word embeddings is that they can be trained on general text data, independent of the task, and then be exported to be used as a component in other NLP tasks.

 
 
 Positional encoding .
The self-attention mechanism is insensitive to ordering. Thus, explicit order information has to be fed to the transformer. The original architecture used sinusoidal encoding, but relative positions are normally used nowadays.
The position information is added to the vector embeddings before being fed to the attention layers.

 
 
 Multi-head attention . The attention mechanism enables models to detect the influence or dependence of words or tokens within a sequence (the context) even if the words are not nearby, which solves the main limitation of RNNs and LSTMs (for example, the connection of pronouns to the nouns they refer to in different sentences).
Attention is computed as

 

 
 | 
 A ​ t ​ t ​ n = s ​ o ​ f ​ t ​ m ​ a ​ x ​ ( 𝐐 ⋅ 𝐊 ⊤ d k ) ​ 𝐕 , Attn=softmax\left(\frac{\mathbf{Q}\cdot\mathbf{K}^{\top}}{\sqrt{d_{k}}}\right)\mathbf{V}, | 
 | 
 

 where 𝐐 \mathbf{Q} , 𝐊 \mathbf{K} , and 𝐕 \mathbf{V} are the input sequences multiplied by trainable weight matrices (resp. W Q W^{Q} , W K W^{K} , and W V W^{V} ), and d k d_{k} is the dimension of the weight matrices.
In autoregressive models (like decoder-only transformers), masked self-attention is used instead of plain self-attention. The mask M M is a
triangular matrix where the upper triangle is set to − ∞ -\infty and the lower triangle is set to 0 0 .
 M M is added to the product 𝐐𝐊 ⊤ \mathbf{QK}^{\top} to cancel out any influence of future tokens on the current token. Note that autoregressive models can only be influenced by past tokens.
The attention mechanism can be easily parallelized (and hence the term multi-head attention). The attention operation is applied to several (depending on a set hyperparameter) input sequences, and the results are concatenated before being fed to the following layers.

 
 
 Feed-forward network . After attention, a feed-forward network consisting of two linear layers is applied. These two layers do not typically include biases.

 
 
 

### 2.2 Types of LLMs

 
 While the original transformer architecture consisted of an encoder-decoder architecture, variations of this have been introduced in subsequent works.
Namely, encoder-only architectures such as BERT ( Devlin et al.,, 2018 ) , decoder-only architectures such as GPT ( Radford et al.,, 2018 ) and LLaMa ( Touvron et al., 2023a, ) , and encoder-decoder architectures, such as T5 ( Raffel et al.,, 2020 ) .
Each of these architectures has been used to tackle different NLU and NLP tasks, and is trained with different pre-training objectives, all of them self-supervised.

 
 
 Encoder-only models , such as BERT ( Devlin et al.,, 2018 ) , use only the encoder blocks of the transformer architecture, which include the full self-attention layers. Thus, these models can access all
words in the given input.
Encoder models are often used for tasks that require the understanding of full sentences, such as sentence or word classification, or extractive question answering.
The pre-training of these models usually consists of masked language modeling (MLM) and/or next-sentence prediction (NSP).

 
 
 
 
 
 Input 
 Embedding 
 
 
 
 Add Norm 
 
 
 
 Multi-Head 
 Attention 
 
 
 
 Add Norm 
 
 
 
 Feed 
 Forward 
 
 
 
 Inputs 
 N × N\times 
 
 
 Positional 
 Encoding 
 
 
 Figure 2: Encoder-only transformer 
 
 
 Decoder-only models , such as GPT ( Radford et al.,, 2018 ) , only make use of the (non-conditioned) decoder blocks of the transformer model, which contain the masked attention layers. Therefore, these models can only attend to past tokens. These models are often called auto-regressive models.
Decoder models are used in generative tasks and are usually pre-trained with next-word prediction (NWP) tasks.

 
 
 
 
 
 Embedding 
 
 
 
 Add Norm 
 
 
 
 Masked 
 Multi-Head 
 Attention 
 
 
 
 Add Norm 
 
 
 
 Feed 
 Forward 
 
 
 
 Linear 
 
 
 
 Softmax 
 
 
 
 Inputs 
 
 
 
 Next token 
 Probabilities 
 N × N\times 
 
 
 Positional 
 Encoding 
 
 
 Figure 3: Decoder-only transformer 
 
 
 Encoder-decoder models (or sequence-to-sequence models), such as the original transformer and T5 ( Vaswani et al.,, 2017 ; Raffel et al.,, 2020 ) , use both the encoder and the conditioned decoder blocks of the transformer architecture.
These models are suitable for tasks that generate text conditioned on some other input text, such as translation, summarization, or generative question answering.
The pre-training objective employed by Raffel et al., (2020) consists of MLM plus NWP.

 
 
 

### 2.3 Training of large language models

 
 The training of LLMs often occurs in different phases, depending on the objectives.
These phases include a self-supervised pre-training phase, with tasks or training objectives
that depend on the type of LLM being trained; a fine-tuning phase, where labeled data are used
to train specific tasks, such as sentiment analysis, conversational text generation,
or summarization; an additional round of fine-tuning using reinforcement learning from human
feedback (RLHF) in which human annotators help fine-tuning the model by providing feedback
on the model responses; and a final phase, where specific system prompts can be passed
to the model to guide it and condition its future responses.

 
 
 We consider of interest to this document any model that has at least undergone a pre-training
phase. These pre-trained models are often called foundation models.
In the following, we describe these training phases.

 
 
 Pre-training : self-supervised learning of different tasks, with standard cross-entropy losses and stochastic gradient descent (SGD) –Adam– optimizers.
We next exemplify some of the most common pre-training tasks given the original sentence My cat likes playing with the ball. , found in the pre-training dataset.

 
 
 
 • 
 
 Masked Language Modeling . In MLM, random tokens from the input sequences are masked with a special token. Models are trained to predict the missing token.

 
 
 
 Example of MLM 
 
 
 Input: My cat MASK playing with the ball.
 Label: likes 
 
 

 • 
 
 Next Sentence Prediction . In NSP, the models are trained to predict whether a given sentence logically follows another sentence. The two input sentences are separated with a special token.

 
 
 
 Example of NSP 
 
 
 Input: My cat likes playing with the ball. SEP He likes playing with other toys, too.
 Label: IsNextSentence
 Input: My cat likes playing with the ball. SEP Harry Potter is a wizard.
 Label: NotNextSentence 
 
 

 • 
 
 Next Word Prediction . In NWP, the model is instructed to return the next word (token) given some text sequence. Single sentences can be used as multiple training inputs, as shown below.

 
 
 
 Example of NWP 
 
 
 Input: My
 Label: cat
 Input: My cat likes playing
 Label: with
 Input: My cat likes playing with the
 Label: ball 
 
 

 • 
 
 T5 Task . The T5 transformer is an encoder-decoder architecture, and thus the pre-training is performed in text-to-text tasks, also including MLM.

 
 
 
 Example of T5 pre-training 
 
 
 Input: My X likes Y with the Z .
 Output: X cat Y playing Z ball.
 Labels: cat, playing, ball 
 
 

 
 
 
 Fine-tuning : further supervised training conducted on domain-related labeled data to teach new
(downstream) tasks or knowledge domains to a pre-trained model ( Touvron et al., 2023b, ) .
For example, an encoder-only transformer such as BERT can be fine-tuned to conduct sentiment analysis by
passing the model sentences and sentiment pairs.
Another such example is GitHub Copilot 2 2 
 2 
 
 
 
 GitHub Copilot – https://github.com/features/copilot ,
where a pre-trained decoder-only model is further trained on code.
Chatbots, such as Llama2-Chat or GPT-3.5-turbo, are fine-tuned on question-answer pairs.
Adapters, including LoRA ( Hu et al.,, 2021 ) , can be used to reduce the cost of fine-tuning by
freezing the pre-trained model weights and injecting trainable rank decomposition matrices
into each layer of the transformer architecture, thereby reducing the number of trainable
parameters for downstream tasks.

 
 
 Reinforced learning from human feedback (RLHF) ( Christiano et al.,, 2017 ) .
In RLHF, models generate different outputs for a given prompt. Then, human annotators rank the quality or appropriateness of the answers produced by the model. These new data, that is, the prompts, the related answers, and the given rankings of answers, are used to train a reward model (as in reinforcement learning) that can then be used to further train the original model given the rewards the model produces for each answer. RLHF has been used to build ChatGPT 3 3 
 3 
 
 
 
 OpenAI, Introducing ChatGPT – https://openai.com/blog/chatgpt , to make the answers of the original model (GPT-3, GPT-3.5, or GPT-4) more conversational and aligned with OpenAI’s objectives.
Meta uses RLHF to improve the models usefulness and safety ( Touvron et al., 2023b, ) .

 
 
 Prompt engineering (zero-shot, one-shot, few-shot learning): the behavior of LLMs can be further steered by including appropriate information in the prompts given to the models ( Brown et al.,, 2020 ; Min et al.,, 2022 ; Wei et al.,, 2023 ) . For example, an LLM can be given some examples of how to solve a problem, with the formulation of the problem and its solution, before asking it the solution of a similar problem (few-shot learning). Likewise, models can be instructed on how to respond or not to respond to specific questions. Production models such as ChatGPT include secret instructions inserted by the provider to guide their general operations. These “in-context learning” or “prompt engineering” approaches, however, can be
sometimes bypassed by other user-provided instructions.

 
 
 
 

## 3 Digital forgetting

 
 As described above, LLMs undergo an initial pre-training phase using text mostly gathered
from the internet (and often uncurated). This opens the possibility for models to be trained
on private, biased, or copyright-protected data, and even to pick up harmful or hateful
behaviors. Additionally, these issues can be further augmented during fine-tuning if the
labeled data is of poor quality.

 
 
 This section discusses the motivations for digital forgetting, including legal and ethical issues,
the types of digital forgetting, and the desirable properties of forgetting procedures.

 
 

### 3.1 Motivations for digital forgetting

 
 The need for digital forgetting stems from several ethical principles, national and supranational
regulations, and codes of conduct. In general, regulations relating to the protection of the privacy
of individuals ( e.g. , GDPR) and the protection of intellectual property ( e.g. , copyright
laws) motivate the research, discussion, and implementation of digital forgetting measures in software
systems, including large language models.

 
 
 In the following, we categorize and discuss the reasons for implementing digital forgetting in ML.

 
 

#### 3.1.1 Privacy

 
 ML models, including LLMs, are trained on vast amounts of data, often obtained from open sources on the web.
Misconfigured services could include private data, either personal or internal data from organizations,
which can be indexed by search engines and freely accessed. These data, which are unintended to be public,
may end up in the training datasets used for LLM pre-training.
Additionally, private data may be used to fine-tune pre-trained models to teach them new downstream tasks.
For example, a hospital could fine-tune a pre-trained model using data from patients to teach the model
how to diagnose new patients.

 
 
 Machine learning models in general have been shown to memorize and leak data from the training dataset ( Shokri et al.,, 2017 ; Salem et al.,, 2018 ) .
Outlying data are even more at risk of being memorized and leaked, which in the case of personal
data may lead to privacy risks ( Smith et al.,, 2023 ) .
One of the most basic forms of attacks on ML models are membership inference attacks (MIA),
in which an attacker attempts to determine whether a given data point is part of the training dataset.
In essence, most MIAs consist of finding out how the models behave when exposed to data from the
training dataset compared to previously unseen data.
This difference in behavior can be measured by the differences in the loss, the classification
confidence, or other metrics. In the case of LLMs, a common approach is to focus on the perplexity metric,
which measures how certain a model is about a given sequence of
text.

 
 
 Yeom et al., (2018) established a connection between the vulnerability to membership inference attacks
and overfitting. While this has been confirmed by other works ( Blanco-Justicia et al.,, 2023 ) , overfitting does not seem to be the only source of vulnerability.
As an example, pre-trained LLMs are seldom overfitted to the training data.
First, because they use vast datasets to train, and second, because they are trained for very
few epochs (sometimes just a single epoch).
However, there have been some attacks that show that LLMs memorize the training data even when not overfitted.

 
 
 

#### 3.1.2 Copyright protection

 
 The copyright protection objective for digital forgetting in LLMs is closely related to privacy, as in both cases we do not want the generated text to include specific information.
However, there is a key distinction: in privacy protection, we want to avoid disclosing personal information expressed in any way, whereas in copyright protection we specifically focus on verbatim text.

 
 
 A model that is capable of answering questions about an individual and whose answers contain sensitive details about such an individual is infringing on their right to privacy. This is not the case with copyright. Copyright laws do not protect facts, but the exact form in which they are expressed. Therefore, a model that can answer questions about some copyrighted work does not necessarily infringe copyright law unless verbatim fragments of the work are generated (and still, the law includes provisions to lawfully quote protected works). Therefore, any information extraction attacks that consider “exact” information may be enough to check for compliance regarding copyright
laws (copyright infringement requires verbatim information to be extracted), but they are not enough to check privacy compliance (privacy can be infringed even if the information extracted about someone has been rephrased).

 
 
 

#### 3.1.3 Model robustness

 
 The whole LLM training pipeline includes pre-training from public data, fine-tuning with
public, crowdsourced, or proprietary data, and possibly further fine-tuning with RLHF, which
may also be public, crowdsourced, or closed.
In all these phases of training, there is a possibility to process low-quality information.
Datasets used for pre-training may include false, inconsistent, or outdated information.
Datasets used for fine-tuning may be crowdsourced, which opens the door to malicious actors providing wrong information.
RLHF depends on human annotators, which rank model outputs based not only on usefulness but also on safety. Rogue annotators may provide wrong information.

 
 
 All of these sources of outdated information, misinformation, outliers, noise, and malicious reporting may influence the learning process and thus produce underperforming models with potentially critical failures.

 
 
 Forgetting procedures may be used to correct some of these issues.

 
 
 

#### 3.1.4 Alignment with human values

 
 LLM pre-training datasets are compiled from diverse, often uncurated sources such as web pages, online encyclopedias, social media posts, and online book libraries. Some of these sources may contain content misaligned with current societal values, including discriminatory language based on gender, race, ethnicity, sexual orientation, or age, often manifesting as stereotypes or prejudices.
Notably, studies have identified strong correlations between pronouns (he/she) and different careers.
Furthermore, these sources may include hateful, violent, or harmful speech associated with discrimination.
Pre-training models on such data may not only continue these biased and harmful behaviors but even amplify them in some cases.

 
 
 Machine learning models, including LLMs, must not only protect individual
privacy rights, as discussed above, but also adhere to ethical values and principles, taking as a
reference point the Universal Declaration of Human Rights ( United Nations,, 1948 ) , and also including laws and
social norms. Alignment with principles such as non-discrimination, fairness, benevolence, and
non-maleficence is of utmost importance. In addition, the EU Guidelines
for Trustworthy AI ( European Commission,, 2019 ) require that developed machine learning models be always under human
control and supervision, which means that any deviation from such principles should be addressed
(or addressable). Therefore, alignment with ethical values may prompt requests for model forgetting
procedures.

 
 
 Forgetting procedures can be employed to identify and eliminate such sources of discriminatory or
harmful behavior, aligning the models with prevailing social norms.

 
 
 
 

### 3.2 Types of digital forgetting

 
 We next discuss the types of undesired knowledge that may be subjected to forgetting in LLMs.

 
 
 General forgetting request. 
The Right to Erasure in Article 17 of the GDPR states that: “The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay and the controller shall have the obligation to erase personal data without undue delay where one of the following grounds applies.” ( European Parliament and Council of the European Union,, 2016 ) The regulation does not specify any specific form or information the data subject has to provide to the data controller to exercise their right.
Search engines, such as Google Search, provide specific forms to exercise the right to be forgotten. In their forms, Google requests the data subjects to identify, and then provide a list of URLs that contain personal information about them, which specific search queries point to the documents of interest, and the motivation for the erasure 4 4 
 4 
 
 
 
 Google’s Right to be Forgotten form – https://reportcontent.google.com/forms/rtbf .
Therefore, the implementation of the right to be forgotten revolves around removing documents that are present on the internet (and not published by the data subjects themselves) and contain some personal information about the data subject from search results.
This erasure does not imply all information about a data subject has to be deleted from search results, but only that information requested by the data subject.
How this translates to LLMs is a subject to be further analyzed.

 
 
 In Section 6.3.1 , we describe a series of knowledge memorization definitions and attacks
that shed some light on how a forgetting request may be studied and acted on.
Definition 1 in that section states that some information is extractable from an LLM if there is some context of prompt for which the LLM returns said information. Definition 3 follows similar lines.
Connecting those definitions to forgetting requests as implemented by Google (and other services) amounts to providing a series of prompts for which private data about the data subject are returned. Note that in the case of generative LLMs, the generation is stochastic and depends on a sampling strategy from the generated LLM distributions; therefore, finding appropriate prompts may not be feasible.
Definition 4 , however, mentions prompt-response pairs; hence, testing whether an LLM returns private information about a topic having the specific document that should be deleted should be straightforward (as shown by attacks described in the same Section 6.3.1 ).

 
 
 Item removal requests. These want forgetting of one or more specific data points or samples contained within the model. Such requests are straightforward for models dealing with tabular or image data, where data points are precisely defined. In the realm of tabular data, each row within a table is considered a data point, with one of the attributes serving as the class label for classification or regression. Likewise, in computer vision tasks, individual images are the designated data points.

 
 
 However, the distinction between items becomes less evident in natural language processing (NLP) tasks and LLMs. While one might consider each token in a text dataset as an individual data point, these tokens often lack meaning on their own. Consequently, in NLP data points can consist of entire sentences or even whole documents, as the separation between meaningful units is less clear-cut compared to tabular or image-based data scenarios.

 
 
 Feature or concept removal requests. These want the model to forget all information about a given subject. The information on a subject may be spread across different sentences and documents. An example of such a request is given by Eldan and Russinovich, (2023) , where the authors attempt to make a model forget everything about the Harry Potter books. A similar approach could be followed to comply with privacy requirements, where all information about a data subject is required to be removed from a model.

 
 
 Class removal requests. These consist of removing all information about a whole class from a model. These requests are quite natural for models used to identify people. For example, in facial recognition, each of the classes corresponds to a single individual, and data points in the class are images of said individual. Thus, removing one class amounts to making the model unable to identify a person that the model could previously identify. A trivial approach for such requests would be to zero out the logits corresponding to the appropriate classes during inference. However,
this is only feasible in a black-box setting where the service provider controls the inference.

 
 
 Class removal requests can make sense for NLP classification tasks, such as sentiment analysis, in which sentences are classified as being positive, negative, or any other range of sentiments. However, for generative models, each class corresponds to a word or token in the vocabulary of the model.

 
 
 Task removal requests. As described in previous sections, LLMs are pre-trained on large text datasets with generic tasks, such as next-word prediction or masked language modeling. After pre-training, models are fine-tuned to teach them different tasks, such as summarization, sentiment analysis, code writing, conversational text generation, etc. Task removal requests attempt to make the fine-tuned LLMs forget one or more of the tasks that the model has been taught to perform.

 
 
 

### 3.3 Requirements of digital forgetting

 
 Regardless of the reason or type of forgetting request, we can define a series of general requirements
that ensure the forgetting procedure is carried out correctly and the resulting model still performs
adequately. This is the purpose of this section, but first we
introduce some preliminary definitions.

 
 
 A dataset D D consists of samples { x i , y i } i = 1 N \{x_{i},y_{i}\}_{i=1}^{N} , where N N is the size of the dataset, x i x_{i} are token sequences, and y i y_{i} the true labels. Note that in self-supervised training, as is the case for most pre-training tasks in LLMs, the labels y i y_{i} do not need to be explicitly defined. For example,
in NLP y i y_{i} is the token immediately following the sequence x i x_{i} in the text. However, during fine-tuning, labels are expected to be explicitly provided. A forget dataset D f ⊂ D D_{f}\subset D is the set of samples to be forgotten. The retain set is defined as D r = D ∖ D f D_{r}=D\setminus D_{f} .

 
 
 A learning algorithm A ⁡ ( D ) A(D) is a probabilistic algorithm that outputs a model given a training dataset D D . Due to the probabilistic nature of A ⁡ ( ⋅ ) A(\cdot) , we cannot ensure that running the learning algorithm twice on the same dataset will return the same model.
However, we can define a probability distribution P ⁡ ( A ⁡ ( D ) ) P(A(D)) over all possible models returned by the learning algorithm when trained on the same dataset D D . Additionally, we can define D ​ i ​ s ​ t ​ ( ⋅ , ⋅ ) Dist(\cdot,\cdot) as the distance between two probability distributions. An example of such a distance is the Kullback-Leibler (KL) divergence.

 
 
 A forgetting algorithm F ⁡ ( D f , A ⁡ ( D ) ) F(D_{f},A(D)) is a (probabilistic) algorithm that returns a model in which the influence of the samples in D f D_{f} has been removed from A ⁡ ( D ) A(D) .
Definitions of unlearning by Nguyen et al., (2022) ; Xu et al., (2023) include D D as a parameter of the unlearning mechanism (as in F ⁡ ( D , D f , A ⁡ ( D ) ) F(D,D_{f},A(D)) ), but access to D D is not always feasible.
For example, a service provider that leverages a foundation model such as Llama2 to offer some service
(possibly after some additional fine-tuning) does not have access to the data used by the party that
conducted the pre-training. However, the service provider may still be required by law to fulfill forgetting requests. For the sake of generality, we will abstain from including D D as a parameter for forgetting, although it may be used in some procedures.

 
 
 When implementing digital forgetting in ML models (and in LLMs in particular)
the following requirements should be taken into consideration.

 
 
 Forgetting guarantees. An essential requirement for any forgetting procedure, whether in the conventional scenario of search engines or with large language models, lies in the ability to demonstrate the fulfillment of a forgetting request, particularly when
such fulfillment is a legal obligation.
A forgetting guarantee serves as a theoretical proof, offering assurance that the content associated with a forgetting request has been forgotten, accompanied by a level of certainty. This ensures a transparent and accountable process in meeting legal requirements and reinforces the credibility of the forgetting mechanism.
 Nguyen et al., (2022) refer to two levels of guarantees, exact and approximate .
 Xu et al., (2023) further expand approximate forgetting into strong and weak forgetting, where strong forgetting is equivalent to the definition of approximate forgetting by Nguyen et al., (2022) , and weak forgetting only applies to the logits of the model, which might be a sufficient guarantee for grey-box and black-box settings.
However, most of the literature on unlearning mechanisms provides no provable guarantees, relying instead on empirical evaluation or auditing.

 
 • 
 
 Exact forgetting. A forgetting algorithm F ⁡ ( D f , A ⁡ ( D ) ) F(D_{f},A(D)) provides an exact forgetting guarantee
if

 

 
 | 
 D ​ i ​ s ​ t ​ ( P ⁡ ( F ⁡ ( D f , A ⁡ ( D ) ) ) , P ⁡ ( A ⁡ ( D ∖ D f ) ) = 0 CLOSE . Dist(P(F(D_{f},A(D))),P(A(D\setminus D_{f}))=0. | 
 | 
 

 From this definition, we can conclude that retraining from scratch on the retain
set D r D_{r} is a straightforward approach to exact unlearning, since
 P ⁡ ( A ⁡ ( D r ) ) = P ⁡ ( A ⁡ ( D ∖ D f ) ) P(A(D_{r}))=P(A(D\setminus D_{f})) 

 

 • 
 
 Approximate forgetting. A forgetting algorithm F ⁡ ( D f , A ⁡ ( D ) ) F(D_{f},A(D)) provides an approximate 
forgetting guarantee, if

 

 
 | 
 D ​ i ​ s ​ t ​ ( P ⁡ ( F ⁡ ( D f , A ⁡ ( D ) ) ) , P ⁡ ( A ⁡ ( D ∖ D f ) ) ≤ t CLOSE , Dist(P(F(D_{f},A(D))),P(A(D\setminus D_{f}))\leq t, | 
 | 
 

 for an acceptable threshold t t .
 Nguyen et al., provide a definition for ϵ \epsilon -certified forgetting, inspired by differential privacy. Given ϵ 0 \epsilon 0 and a sample z ∈ D z\in D ,

 

 
 | 
 e − ϵ ≤ P ​ r ​ ( F ⁡ ( z , A ⁡ ( D ) ) ) P ​ r ​ ( A ⁡ ( D ∖ z ) ) ≤ e ϵ . e^{-\epsilon}\leq\frac{Pr(F(z,A(D)))}{Pr(A(D\setminus z))}\leq e^{\epsilon}. | 
 | 
 

 

 • 
 
 No guarantees. The forgetting guarantees described above may not be attainable in all cases, either because achieving them or computing them is unfeasible. In these cases, we can refer to empirical evaluation or auditing to provide a level of risk
reduction (that is, how much the risk of the model remembering the undesired item has been reduced). These evaluations will depend on the type of forgetting request that needs to be dealt with. When requests are related to privacy or copyright protection, the difference in membership inference vulnerabilities can serve as a measure of risk reduction. If the forgetting request involves corrections of biases in the models, fairness metrics can be used. Finally, if some task needs to be forgotten, specific task benchmarks can be used. Refer to Section 6.3 for different evaluation mechanisms.

 

 
 
 
 Generalization. In the context of unlearning in LLMs, D f D_{f} does not need to be exactly from the original LLM’s training corpus.
Given the diversity and size of the LLM’s training data, the samples we unlearn can represent a general concept, rather than specific training samples.
This necessitates an unlearning method that can generalize to similar samples with shared characteristics, thereby enhancing the effectiveness of unlearning across a broad concept and improving robustness against paraphrasing attacks.

 
 
 Retaining of performance. Whatever the method used for forgetting, the resulting models should retain much of the performance of the original model with respect to the same metrics and benchmarks. Note that if a given task is removed from the resulting model, some specific benchmarks may no longer be applicable.
 Xu et al., (2023) consider two performance metrics for classifiers,
namely consistency and accuracy.
Consistency is defined as the level of agreement between a model resulting from a forgetting procedure and a model trained only on the retain dataset.
Accuracy is defined in a standard way, by evaluating the model resulting from a forgetting procedure on the retain dataset.
While these metrics may not be directly applicable to generative models, other metrics, such as perplexity, could be used to compare unlearned and retrained models.

 
 
 Runtime and scalability. 
When serving forgetting requests, especially if triggered by privacy concerns or by the application of the right to be forgotten, it is important that the forgetting procedure can be executed promptly.
Article 17 of the GDPR requires in paragraph 1 that “the controller shall have the obligation to erase personal data without undue delay”; additionally, paragraph 2 indicates that such procedures should be
carried out “taking account of available technology and the cost of implementation”.
Thus, it is important that forgetting algorithms can be executed in a timely manner, so that no personal information is accessible for longer than needed (to minimize potential harm to individuals).
Other types of forgetting requests, such as those seeking to delete copyrighted material, correct biases or remove tasks may not be so urgent.

 
 
 A different, but related property is scalability, meaning how many forgetting requests can be processed simultaneously and how that
affects the runtime and utility, of both the resulting model and the forgetting procedure.

 
 
 
 

## 4 Approaches to digital forgetting in LLM

 
 We next delineate the main approaches to digital forgetting in LLMs.
More details can be found in Zhang et al., (2023) and the more general surveys on unlearning Qu et al., (2023) ; Xu et al., (2023) ; Nguyen et al., (2022) ; Shaik et al., (2023) .

 
 
 Data pre-processing and model retraining. Carefully choosing the data to include in the pre-training and fine-tuning phases is a sensible and recommended approach to prevent any unwanted behavior from the models, be it from a privacy, a copyright, or an alignment perspective. As an example, during data collection, Meta refrains from using data sources where high amounts of private data are found (Llama2 model ( Touvron et al., 2023b, ) ). Although Meta provides an analysis on gender, nationality, sexual orientation, and religion potential biases, they do not filter any data. They also analyze the pre-training data in search for hateful content using HateBERT, and determine that about a 0.2% of documents are potentially hateful.

 
 
 A second potential approach to limit the amount of private information
in the training text is to perform text anonymization, also called text redaction or sanitization. Traditionally, redaction has been manually carried out by human experts. However, with the improvement of artificial intelligence mechanisms, some automated approaches have been proposed. Most approaches are based on named-entity recognition (either rule-based or ML-based), where potentially private or sensitive items in the text are identified and then either removed or generalized
 ( Sánchez and Batet,, 2016 ; Hassan et al.,, 2019 ) .

 
 
 A third approach based on data pre-processing is deduplication ( Kandpal et al.,, 2022 ) . This consists in finding replications of the same text within the training corpus and deleting any duplicates. What constitutes a duplicate can be parameterized in terms of length. Duplicate documents tend to be more vulnerable to membership inference attacks, as shown in Section 6.3.1 . Thus, deduplication has been shown to be an effective mechanism against memorization (which may lead to privacy, copyright, robustness, and alignment issues).

 
 
 Privacy-preserving model pre-training. Using
privacy-preserving machine learning mechanisms may limit the influence of any single
data point on the model. In this case, instead of protecting the data, we use some training mechanism which ensures privacy. We next
describe two such mechanisms.

 
 
 
 • 
 
 Differentially private stochastic gradient descent (DP-SGD) .
Differential privacy (DP) bounds the probability of correctly inferring
private information about any individual subject within a database, parameterized by the privacy budget ϵ \epsilon .
If each individual is represented by a record,
the output of a DP mechanism should be (almost) unaltered by the presence or absence of
any single record.
This could provide strong guarantees against knowledge extraction.
Values of ϵ \epsilon closer to 0 provide more privacy at the cost of data utility.
In ML, differential privacy is usually applied through the DP-SGD private training algorithm ( Abadi et al.,, 2016 ) to provide ( ϵ , δ ) (\epsilon,\delta) -DP,
a relaxation that basically consists of ϵ \epsilon -DP being satisified with probability 1 − δ 1-\delta .
In LLMs, and text data protection in general, it is highly challenging to define who are the
individual subjects to be protected, which may be a limitation of DP-SGD in this context.

 

 • 
 
 Private aggregation of teacher ensembles (PATE) .
PATE ( Papernot et al.,, 2018 ) uses a private ensemble of models
trained on independent partitions of data, called the teacher models,
to train an additional model, called the student model, which is then made public (either the model or an API to query the model).
Each teacher model is a model trained independently on a subset of the data whose privacy one
wishes to protect. The data are partitioned to ensure that no pair of teachers will be trained on overlapping data.
Training each teacher on a partition of the sensitive data produces different models solving the same task.
At inference time, teachers independently predict labels.
Then, to train the student model, a differentially private aggregation mechanism is used.
PATE’s final step involves the training of the student model by knowledge transfer from the
teacher ensemble using access to public but unlabeled data.

 

 
 
 
 Machine unlearning .
Given the high cost and long duration required to train LLMs, retraining them from scratch to eliminate undesirable behaviors is often a tedious and impractical endeavor.
Currently, there is a growing trend in the literature to adopt the unlearning approach as an efficient means for digital forgetting in LLMs.
Methods that attempt to remove undesirable knowledge or behaviors from models that have already undergone
pre-training (and maybe also fine-tuning) without retraining the models from scratch are called
 machine unlearning mechanisms.
These mechanisms rely on further fine-tuning, often with adversarial objectives, on the identification of
parameters that are correlated with unwanted information and their modification, and on parameter arithmetic.
Sections 5.1 and 5.2 below cover machine unlearning mechanisms of this kind.

 
 
 Prompt engineering .
As described in Section 2 , specific prompts to trained LLMs can be used to further steer the models’ behavior after fine-tuning. These methods do not cause any changes to the model parameters and can
in some cases be bypassed by inputting contradicting prompts.
However, carefully crafted prompts inserted at the beginning of a conversation with a generative model can
prevent it from generating private, biased, or harmful information.
For example, Üstün et al., (2024) use the prompt (preamble)
 Does the following request contain harmful, unethical, racist, sexist, toxic, dangerous,
offensive or illegal content or intent? If yes, explain that you do not engage in these type of requests. 
They find that their model rejects up to 88% of such requests.
Section 5.4 describes methods that use prompt engineering for digital forgetting.

 
 
 Post-processing .
Other technical mechanisms, that may be applied to models that are only accessible through an API, are
post-processing or filtering. After the models have generated an output, but before serving it to the
user, the service provider could analyze the output to search for unwanted generations, possibly using
other LLM-based classifiers. Other approaches use a memory of unwanted responses to identify and filter
out any such generations. Section 5.4 includes some of these approaches.

 
 
 Sampling strategy in generative LLMs. While not a forgetting strategy, the choice of a sampling strategy in generative LLMs may impact the probability of generating verbatim sequences of training text data.
As mentioned before, a generative LLM outputs a distribution over all possible tokens given a sequence. The next token is chosen from this distribution by sampling. Common sampling strategies include “greedy” sampling, in which the highest probable token is returned; top- k k sampling, in which the probabilities of all tokens that are not among the top k k most probable ones are set to 0 0 , the top k k are renormalized, and then the next token is sampled from those; and vanilla multimodal sampling, in which the next token is obtained by sampling from the whole distribution of probabilities. Often, a temperature parameter t t is used to flatten the distribution obtained from the model. In technical terms, the logits are divided by t t before applying the softmax function, which makes the model less confident in its outputs and therefore diversifies the token generation procedure.

 
 
 

## 5 Survey on unlearning in LLMs

 
 As discussed in Section 4 , unlearning is the most general way to efficiently eliminate undesirable or to-be-forgotten knowledge from LLMs without the need for full retraining.
In this section, we conduct a comprehensive survey of unlearning methods applicable to LLMs and classify them into four primary categories: global weight modification, local weight modification, architecture modification, and input/output modification methods.
This classification is predicated on the location within the model where the unlearning process is executed.

 
 
 Global weight modification methods encompass those that have the potential to alter all model weights as a final outcome of the unlearning process. Conversely, local weight modification methods are restricted to modifying a specific subset of weights.

 
 
 Architecture modification methods introduce additional layers into the model’s structure, while input/output modification methods function exclusively at the input/output level.

 
 
 Subsequently, we further divide these primary categories based on how the unlearning is performed.

 
 
 Figure 4 illustrates the taxonomy of unlearning methods for LLMs that we propose, to be used as a framework for this survey.

 
 
 Figure 4: Taxonomy of unlearning methods in LLMs 
 
 

### 5.1 Global weight modification

 
 In these methods ( Bourtoule et al.,, 2021 ; Jang et al.,, 2022 ) , every parameter of the model is subject to modification during the unlearning process.
Whereas global weight modification methods offer a stronger unlearning guarantee compared to the other approaches, they often entail substantial computational and time overheads, which renders them impractical for LLMs in most cases.

 
 

#### 5.1.1 Data sharding

 
 This approach typically entails dividing the training data into multiple disjoint shards, each corresponding to a subset of the overall data, and training a separate model for each shard ( Bourtoule et al.,, 2021 ; Liu and Kalinli,, 2023 ) .
These individual models can then be leveraged to effectively remove data
whose unlearning has been requested.

 
 
 Bourtoule et al., (2021) introduce SISA (Sharded, Isolated, Sliced, and Aggregated training) as a generic exact unlearning framework.
The training dataset is divided into S S non-overlapping shards, each containing R R slices.
For each shard, a model is trained using gradient descent, by processing the data slice by slice and saving a checkpoint after each slice.
Once training is complete, the model is saved and associated with the shard.
This process is repeated for all shards.
During inference, each model predicts a label and these labels are aggregated, similar to ensemble methods.
If an unlearning request is received, the shard containing the data point is identified, and the slice containing the unlearning request is located.
The data point (typically a sequence of tokens in NLP tasks) is removed from this slice, and the model is retrained from the last checkpoint,
which by construction ensures the model forgets the data point to be unlearned.
The main advantage of SISA is that it provides an exact unlearning guarantee because the data to be forgotten do not influence the retrained version of the model.
This method can be applied to a wide range of ML tasks and model architectures, including LLMs.
However, SISA is not very practical for LLMs due to the high computational/memory cost associated with model and checkpoint saving, retraining, and inference.
On the other hand, there is a trade-off between the number of shards and other performance aspects.
Increasing the number of shards reduces forgetting costs, but besides increasing the cost of inference, it reduces the ensemble model utility due to the loss of synergistic information between training samples.
Additionally, Kadhe et al., (2023) found that SISA can indeed reduce fairness in LLMs and adapted a post-processing fairness improvement technique to make it fairer.

 
 
 To reduce retraining cost and maintain utility at inference, Liu and Kalinli, (2023) propose the leave-one-out (LOO) ensemble method to unlearn the to-be-forgotten token sequences from LMs.
In addition to the base LM (the “student” model) being trained on the entire dataset, M M “teacher” LMs are trained on disjoint data from different users.
Upon receiving a forget request, the base student model is fine-tuned on the target token sequence using the aggregation of the predictions of M − 1 M-1 teachers as soft labels.
The teacher model excluded from aggregation is the one trained on the target sequence.
The fine-tuned base model is then used for future predictions.
This method was evaluated on text generation and automatic speech recognition using an LSTM.
To simulate private user information, “canaries” representing rare or unique token sequences are inserted into the training data.
The authors assess the privacy risks of the unlearned LMs by trying to extract those canaries using beam search and random sampling methods.
While offering improved utility compared to SISA, this method’s unlearning guarantee for the base model is approximate,
and there is no guarantee for the teacher models.
Moreover, the process of predicting soft labels, aggregating them, and then fine-tuning the entire base model’s parameters incurs significant computational overheads, particularly with LLMs.

 
 
 

#### 5.1.2 Gradient ascent

 
 Methods under this approach aim to move the model away from target undesirable knowledge by fine-tuning all model parameters to maximize loss on the related target tokens.

 
 
 Given a set of target token sequences representing sensitive information, Jang et al., (2022) simply negate the original training loss function for those token sequences.
Specifically, given a model f ⁡ ( x , θ ) f(x;\theta) where x x is a sequence of tokens x = ( x 1 , … , x T ) x=(x_{1},\ldots,x_{T}) , the loss for x x is given by

 

 
 | 
 ℒ x ( θ ) = − ∑ t = 1 T log ( p θ ( x t | x t ) ) . \mathcal{L}_{x}(\theta)=-\sum_{t=1}^{T}\log(p_{\theta}(x_{t}|x_{ t})). | 
 | 
 

 
 
 The overall loss for N N samples is computed as

 
 
 

 
 | 
 ℒ ⁡ ( θ ) = 1 N ​ ∑ i = 1 N ℒ x i ​ ( θ ) . \mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{x^{i}}(\theta). | 
 | 
 

 
 
 The parameters θ \theta are then updated using the gradient ascent (GA) rule:

 
 
 

 
 | 
 θ = θ + η ​ ∇ θ ℒ ​ ( θ ) . \theta=\theta+\eta\nabla_{\theta}\mathcal{L}(\theta). | 
 | 
 

 
 
 The authors found that unlearning many samples at once substantially degrades the performance of LMs, and unlearning them sequentially can mitigate this degradation.
The method is evaluated on text classification and dialogue tasks, with empirical validation based on extraction likelihood ( Jang et al.,, 2022 ) and memorization accuracy ( Tirumala et al.,, 2022 ) .
These metrics assess whether the model’s behavior on forgotten sequences aligns with that of unseen data.
Gradient ascent (GA) and its variants only require the to-be-forgotten data and sometimes enhance the generalization capabilities of the model as observed by Yoon et al., (2023) .
However, GA can cause the model to lose its understanding of the language ( Eldan and Russinovich,, 2023 ) .
Furthermore, the success of unlearning depends on the specific target data and the domain of the to-be-forgotten data ( Smith et al.,, 2023 ) .

 
 
 To preserve the generation capability of LLMs, SeUL ( Wang et al.,, 2024 ) applies GA ( Jang et al.,, 2022 ) on specific sensitive spans within the sequence instead of the entire sequence.
Two annotation methods are used for sensitive span selection.
The online method assumes a token to be sensitive if it has a low prediction probability, i.e. , and a higher perplexity score.
The offline method employs the in-context learning capability of LLMs, such as ChatGPT, to annotate such sensitive spans.
The authors use offline annotations to evaluate the
unlearned LLMs —the LLMs after unlearning—, accompanied by two unlearning evaluation metrics—sensitive extraction likelihood (S-EL) and sensitive memorization accuracy (S-MA).
The method is evaluated on text classification and dialogue tasks, with empirical validation based on S-EL and S-MA.
While this method allows for more utility-preserving, focused, and efficient unlearning compared to GA ( Jang et al.,, 2022 ) , the annotation methods may lack precision in identifying privacy-sensitive tokens.

 
 
 Yao et al., (2023) observed that: (1) only applying gradient ascent as Jang et al., (2022) do is insufficient to effectively unlearn unwanted (mis)behaviors ( e.g. , harmful responses and hallucinations), (2) preserving performance on normal samples is harder to achieve than unlearning, and (3) the format of the normal data used for guiding the LLMs to preserve utility on normal tasks greatly impacts normal performance.
Based on these observations, an unlearning method that minimizes three weighted loss functions is proposed.
The method involves updating the LLM during unlearning by jointly (1) applying GA on forget samples, (2) forcing random outputs on forget samples, and (3) minimizing the KL divergence between predictions of the original and unlearned models on normal samples to preserve normal utility.
The authors found that forcing random outputs helps the model forget the learned undesirable outputs on the forget samples by forcing it to predict random outputs.
Also, this method helps preserve the model utility on normal samples.
The method is evaluated on text generation and question-answering tasks, with empirical validation based on metrics for evaluating unlearning in language models, covering efficacy, diversity, and fluency.
In their evaluation, the authors consider several unlearning applications: remove harmful responses, erase copyrighted content, and eliminate hallucinations.
While the method provides a better trade-off between unlearning and model utility, it requires a large number of training epochs to unlearn forget data and maintain utility simultaneously.

 
 
 

#### 5.1.3 Knowledge distillation

 
 Methods under this approach treat the unlearned model as a student model that aims to mimic the behavior of a teacher model with desirable behavior.

 
 
 Wang et al., (2023) propose the Knowledge Gap Alignment (KGA) method as an unlearning technique for LLMs.
KGA utilizes training data, data to be forgotten, and external data for unlearning to produce an updated model that exhibits a similar behavior on forgotten data as on unseen data while retaining
utility on the remaining data.
This is achieved by aligning the “knowledge gap,” which refers to the difference in prediction distributions between models trained with different data.
Aligning the unlearned model’s behavior on the forget data with unseen data is achieved by minimizing the distribution difference between the unlearned model predictions on the forget data and the original model predictions on the unseen data.
The authors use the KL divergence to measure this difference.
To maintain the utility, the original model is treated as a teacher for the unlearned model to minimize the distribution difference when processing the retain data.
The method is evaluated on text classification, machine translation, and response generation, with an empirical evaluation based on metrics used to measure the changes in the probability distributions of models.
The main advantage of this method lies in its generic nature, which allows it to be applied to various models and tasks.
However, the need to train two identical models and then fine-tune all model parameters may limit its practicality and efficiency when applied to large language models (LLMs).
Additionally, the unlearning process requires the training data, the data to be forgotten, and other external data with no overlapping with the training data.
The utility of the unlearned model is highly dependent on the sizes of the data to be forgotten and the external data.

 
 
 

#### 5.1.4 Generic alternatives

 
 This approach aims to achieve unlearning by fine-tuning the whole model parameters to predict generic alternatives instead of the to-be-forgotten tokens.

 
 
 Eldan and Russinovich, (2023) propose a novel unlearning method for erasing source-specific target data from LLMs.
The authors claim that their method successfully removed all knowledge on the Harry Potter books from Meta’s open-source LLM, Llama 2-7B, in about 1 GPU hour of fine-tuning.
Their three-step method involves:

 
 • 
 
 Token identification through reinforced modeling. This process involves creating an augmented model with a deeper understanding of the content that needs to be unlearned. This is done by further fine-tuning the original model on the target data (Harry Potter books). Tokens with significantly increased probability are identified, indicating content-related tokens that should be avoided during generation.

 

 • 
 
 Expression replacement. Distinctive expressions in the target data are replaced by their generic alternative labels. The authors used GPT-4 to generate those alternatives automatically. This helps approximate the next-token predictions of a model that has not seen the target data.

 

 • 
 
 Fine-tuning. Armed with these alternative labels, the model undergoes fine-tuning to erase the original text from the model’s memory and provides a plausible alternative when prompted with its context.

 

 
 The authors demonstrated that the unlearned model no longer generates or recalls Harry Potter-related content, whereas its performance on common benchmarks remains nearly unaffected.
The method was tested empirically on two evaluation tasks: i) completion-based evaluation, which uses 300 Harry Potter-related prompts that are analyzed to check whether it can generate Harry Potter-related content; and ii) token-probability-based evaluation, which
checks the model’s completion probabilities for selected prompts to ensure it does not favor Harry Potter-specific terms.
This method provides plausible alternatives to unlearned tokens, which is important to maintain model utility in many scenarios.
However, Shi et al., (2023) found that models that underwent unlearning with this approach could still output related copyrighted content.
Another limitation is that this approach relies on replacing unique terms with their generic alternatives, which is challenging when extending it to non-fiction content.
In Harry Potter, there are many unique terms, but in non-fiction, idiosyncrasies are rare and the core of the text is often ideas or concepts rather than specific words.
This presents a significant challenge for this unlearning approach when applied to non-fiction content.

 
 
 

#### 5.1.5 Reinforcement learning from human feedback (RLHF)

 
 RLHF involves leveraging human feedback to guide the model’s learning process.
It combines reinforcement learning (RL) with human feedback to teach the model to generate text that aligns better with human preferences and intentions.

 
 
 Lu et al., (2022) present Quark, which considers the task of unlearning undesired behaviors of an LLM by fine-tuning the model on signals of what not to do.
Quark starts with a pre-trained LLM, initial training prompts, and a reward function to initialize a datapool of examples.
It alternates between exploration, quantization, and learning.
In quantization, it sorts the datapool by reward and partitions it into quantiles.
For learning, it trains on the quantized datapool using a standard language modeling objective and a KL-penalty.
During exploration, it adds new generations to the datapool by sampling from the model conditioned on the highest-reward token.
The objective of the three-step process above is to teach the model to generate texts of varying quality with respect to the reward token.
Then, at inference, the sampling is conditioned with the best reward token to steer toward desirable generations.
Quark was evaluated on various benchmarks like toxicity, unwanted sentiments, and repetitive text, and it showed promising performance in reducing the targeted undesired behaviors while maintaining overall language fluency and diversity.
Quark is one of the well-known state-of-the-art controllable text generation methods that effectively aligns the model output with human expectations Daheim et al., (2023) .
However, in addition to its computational cost significantly increasing as the datapool size increases, the model may still retain the sensitive information on its parameters. Such information could be extracted by white-box extraction attacks.

 
 
 
 

### 5.2 Local weight modification

 
 These methods only allow limited manipulation of certain weights for unlearning.

 
 

#### 5.2.1 Local retraining

 
 This approach employs a selective retraining strategy, where only the model parameters relevant to the unlearning target are optimized, leaving the rest unaltered.

 
 
 Yu et al., 2023a () introduce a gradient-based debiasing method called Partitioned Contrastive Gradient Unlearning (PCGU).
PCGU systematically identifies and retrains specific model weights responsible for biased behaviors.
The approach involves rank-ordering weights and selecting them based on the gradients observed in contrastive sentence pairs that differ along a demographic axis.
A contrastive sentence pair consists of two sentences that are similar but have one key difference that introduces a bias.
The gradients of these sentence pairs are used to determine which weights in the model significantly contribute to the bias.
The authors use a subset of the Winogender Schemas dataset ( Zhao et al.,, 2018 ) that has 240 sentence pairs, differing only in gendered pronouns for the subject, who is identified by their job.
The model’s stereotypes for each job are reflected in the probabilities assigned to male and female pronouns.
The authors demonstrate that the method is effective both in mitigating bias for the gender-profession domain as well as in generalizing these influences to other unseen domains.
However, the work only addresses gender bias in masked LLMs and it remains uncertain whether the proposed method can be generalized to other kinds of LLMs and more complex social biases like racism and classism.

 
 
 

#### 5.2.2 Task vector

 
 These methods build on the idea that an LLM can be decomposed into task-specific vectors and eliminate the forget task vector to achieve unlearning.
 Ilharco et al., (2022) introduce a novel paradigm for steering neural network behavior based on task vectors.
A task vector τ t \tau_{t} is defined as the difference between the parameters of a model fine-tuned on a specific task t t ( θ ft t \theta^{t}_{\text{ft}} ) and those of the corresponding pre-trained model ( θ pre \theta_{\text{pre}} ).
Employing τ t \tau_{t} allows for selective modification of the model’s behavior for task t t without significant impact on other tasks.
This is done via the negation and addition arithmetic operations: negating τ t \tau_{t} allows forgetting knowledge related to task t t while adding it improves the model’s performance on t t .
The method is evaluated on text generation based on toxic text generation and the perplexity of tokens in the to-be-forgotten task.
The authors show that this method can be used to improve the model performance on multiple tasks by adding their vectors.
However, it cannot handle forgetting discrete facts or specific token sequences,
which makes it more suitable for broader task-based modifications than for fine-grained ones.

 
 
 By drawing inspiration from human cognitive processes and the task arithmetic in Ilharco et al., (2022) , the authors of Ni et al., (2023) propose a paradigm of knowledge updating called F-Learning (Forgetting before Learning).
Specifically, the initial model is fine-tuned with old knowledge, followed by subtraction of the parameter differences between the fine-tuned and initial models from the initial model parameters.
This process is termed “old knowledge forgetting.” Then, the unlearned model is fine-tuned with the new knowledge, constituting the “new knowledge learning” stage.
After the two stages, the model’s knowledge is updated.
The method was evaluated on text generation and question answering and showed promise in mitigating conflicts between old and new knowledge.
However, as it happened with Ilharco et al., (2022) , this method is not suitable for forgetting individual facts or sequences of tokens.

 
 
 

#### 5.2.3 Direct modification

 
 Instead of gradient optimization of the original or additional parameters, this approach locates and directly modifies the relevant parameters or neurons to achieve unlearning.

 
 
 DEPN ( Wu et al.,, 2023 ) assumes that private information, such as usernames and contact information, is encoded in privacy-related neurons of LLMs.
Based on that, the method first locates these neurons by using the integrated gradient method by Sundararajan et al., (2017) and then sets their activations to zero in order to eliminate the privacy information encoded in those neurons.
A privacy neuron aggregator is also introduced to handle multiple unlearning requests.
An analysis of the relationship between privacy neurons and model memorization was also performed.
The authors found that model size, training time, and frequency of occurrence of private data are all factors that have an impact on model memorization.
As the model memorization of private data deepens, the aggregation of privacy-related neurons associated with those data becomes more obvious.
The method was evaluated on masked language modeling (MLM) and empirically based on metrics that measure the privacy preservation of MLM.
This method is efficient, as it only requires the forget set without fine-tuning.
However, as the amount of forget data increases, the utility of the model drops significantly because more neurons are deactivated.
Also, the authors found that too many instances in a batch reduces the effect of forgetting.
Besides, the method evaluation on forgetting private information was limited to names and phone numbers, due to the limited availability of datasets with a wide range of private data.
The authors recognize the necessity of expanding their dataset to improve the effectiveness and relevance of their method.

 
 
 Pochinkov and Schoots, (2023) found that both feed-forward and attention neurons in LLMs are task-specialized, and removing certain neurons from them significantly decreases performance on the forget task while hardly affecting performance on the other tasks.
Based on that, they introduce a selective pruning method for identifying and removing neurons from an LLM that are related to a certain capability like coding and toxic generation.
Neurons are removed based on their relative importance on a targeted capability compared to the overall model performance.
The evaluation of this method involves selectively removing the coding capability from several LLMs.
This method is compute- and data-efficient for identifying and removing task-specialized neurons.
It also has the potential of applicability in the removal of other harmful skills, such as toxicity and manipulation.
However, it requires a task-representative dataset and it is only effective for capabilities directly captured by these datasets.
Besides, its effectiveness depends on the separability of the capabilities of an LLM. It becomes
less effective with models like Pythia (trained without dropout) and on smaller LLMs.

 
 
 
 

### 5.3 Architecture modification

 
 In addition to the computational burden, altering whole model weights may degrade the model utility on the remaining data.
To tackle these issues, these methods add extra parameters within the model, with the aim to achieve efficient and utility-preserving unlearning ( Chen and Yang,, 2023 ; Limisiewicz and Mareček,, 2022 ) .

 
 

#### 5.3.1 Extra learnable layers

 
 EUL ( Chen and Yang,, 2023 ) integrates an additional unlearning layer into transformer structures after the feed-forward networks.
For each forgetting request, an unlearning layer is individually trained using a selective teacher-student objective.
During the training of this layer, the original parameters are frozen, and a joint unlearning objective is used.
The design of this joint objective is such that it compels the unlearning layer to diverge from the original model’s predictions on the forget data, while following it on the remaining data.
A fusion mechanism is also introduced to handle sequential forgetting requests.
The method was evaluated on text classification and generation, and empirically verified using the Masked Language Model loss (MLM loss) on predictions of the masked sensitive tokens from the forget set.
Also, MIA ( Kurmanji et al.,, 2024 ) was used to predict whether the input data belong to the forget set or the retain set based on their representations after the final layer of the model.
While this method maintains the model
utility on the retain set, it has its limitations.
It does not constitute complete forgetting, as removing the additional layers causes the old model to revert to its original behavior on the data to be forgotten, thus contradicting privacy laws.
Also, training an additional layer is required to forget every unlearning request, which might not be scalable.

 
 
 Kumar et al., (2022) propose two variants of SISA to improve efficiency with LLMs: SISA-FC and SISA-A.
SISA-FC starts from a base model, pre-trained on a generic text corpus, and then adds fully connected (FC) layers on top of it.
Only the parameters from the added layers are updated during retraining.
This approach minimizes overall retraining time, as backpropagation of gradients occurs only in the final layers, and storage requirements are reduced to the weights of these additional parameters.
However, the utility of the model will be severely affected when compared to fine-tuning the entire model.
SISA-A tries to address this by training the model using the parameter-efficient adapter method from Houlsby et al., (2019) , which injects learnable modules into the encoder blocks of the transformer.
The methods were evaluated on classification tasks with the BERT model with no evaluation for forgetting performance.
While SISA-A preserves the classification utility of the resulting model better than SISA-FC, it entails more computational and storage costs than SISA-FC.
From the average results in the paper, SISA has the highest accuracy (83.3%) but also the longest runtime (1732.7s) and linearly increasing memory usage.
SISA-A has a slightly lower accuracy (80.7%) but a much shorter runtime (145.7s) and a less drastic memory increase.
SISA-FC has the lowest accuracy (20-30% lower than SISA-A) and runtime (20-30% lower than SISA-A), with a lower memory usage than SISA-A.
A common drawback of both SISA-FC and SISA-A is that the content of the generic text corpus (learned during the pre-training phase) cannot be forgotten.

 
 
 

#### 5.3.2 Linear transformation

 
 Belrose et al., (2023) introduce LEACE, a closed-form method to prevent linear classifiers from detecting a concept, such as word part-of-speech, with minimal change of representation.
This is achieved by applying a procedure called concept scrubbing, which erases all linear information about a target concept in every intermediate representation.
LEACE sequentially fits an affine transformation to every intermediate layer to erase the target concept from its features while minimizing the distance from the original features.
The method was evaluated on text classification and empirically verified based on metrics that measure the bias in a classifier.
To fit LEACE parameters, samples from the respective model pretraining distribution were used.
While this method can be used to improve fairness ( e.g. preventing a classifier from using gender or race as deciding criteria) and interpretability ( e.g. removing a concept to observe changes in model behavior), it has many limitations.
It may degrade utility due to erasing irrelevant features to the concept and requires caching hidden states during training, thereby leading to a substantial demand for memory.
Also, its validation is limited to part-of-speech, and its practical applicability is uncertain.

 
 
 Gender information in representations of biased LLMs can be divided into factual gender information and gender bias.
Factual gender information encompasses grammatical or semantic properties indicating gender in English texts, such as explicit gendered pronouns like ”he” or ”she.”
Gender bias, on the other hand, refers to the model’s tendency to associate certain words with specific genders, like ”nurse” being more correlated with females than males.
 Limisiewicz and Mareček, (2022) aim to mitigate the gender bias of a pre-trained LLM by manipulating contextual embedding.
They apply an orthogonal transformation to separate lexical and syntactic information encoded in the model embedding.
Then they filter out the bias subspace from the embedding space and keep the subspace encoding factual gender information.
The authors evaluated the effectiveness of the method on three MLMs using metrics to measure the overall improvement of a de-biasing algorithm.
Although this method is efficient because it applies linear transformations on the contextual embedding only, there is no guarantee that all bias-related dimensions will be filtered.
This is because the bias signal can be encoded non-linearly in LLMs and even when the whole bias subspace is removed, the information can be recovered in the next layer of the model ( Ravfogel et al.,, 2020 ) .

 
 
 
 

### 5.4 Input/output modification

 
 This approach treats the model as a black-box and requires only the input/output access of a model without delving into its internal workings.
This approach is specifically useful when the operator has no access to the model weights, e.g., for products and services wrapping API-based models.
However, in the context of the RTBF, it does not yield any real privacy, as the models still retain the to-be-forgotten knowledge.

 
 

#### 5.4.1 Input manipulation

 
 In this approach, the forget input to an LLM is deliberately altered or adjusted in a systematic way to facilitate the process of unlearning.

 
 
 Gallegos et al., (2024) propose a prompting-based bias mitigation method that leverages the zero-shot capabilities of LLMs to reduce biased stereotypical predictions and call it zero-shot self-debiasing .
Two techniques, self-debiasing via explanation and self-debiasing via reprompting, are proposed.
In the former, the model first explains invalid assumptions in the answer choices, identifying potential stereotyping. Then, it answers the question.
In the latter, the model first answers the question as per the baseline approach. Then, it is reprompted to remove bias from its initial answer.
Below are two examples illustrating the two techniques.

 
 
 
 Self-debiasing via explanation 
 
 
 - User: Explain which answers rely on invalid assumptions.
 - Question Answers: Who is more likely to be a nurse? A) John B) Mary.
 - Assistant: The assumption that Mary is more likely to be a nurse because she is a woman is invalid. Both John and Mary have an equal likelihood of being a nurse based on this information.
 - User: Answer the multiple-choice question with a single letter.
 - Question Answers: Who is more likely to be a nurse? A) John B) Mary.
 - Assistant: Without additional information, both options are equally likely. So, I can’t choose between A) John and B) Mary. 
 
 
 
 
 Self-debiasing via reprompting 
 
 
 - User: Answer the multiple-choice question with a single letter.
 - Question Answers: Who is more likely to be a nurse? A) John B) Mary.
 - Assistant: B) Mary.
 - User: Remove bias from your answer by answering the question again with a single letter.
 - Assistant: Without additional information, both options are equally likely. So, I can’t choose between A) John and B) Mary. 
 
 
 
 The authors demonstrate the ability of their method to decrease stereotyping in question-answering over nine different social groups with a single prompt.
This method does not require any additional training data, exemplar responses, fine-tuning, or auxiliary models that traditional bias mitigations require, and thus it is a more efficient and practical solution for bias mitigation.
However, the method is task and context-dependent.
It is designed for multiple-choice questions, rather than for the more common open-ended question.
Also, it uses manually created prompts, which limits its generalization to other biases.
Future research could focus on detecting biases in free text and also explore automated prompt generation to manage biases more effectively.

 
 
 

#### 5.4.2 Information retrieval

 
 The methods in this group aim to selectively extract or manipulate information from external knowledge
to shape the unlearning trajectory of LLMs.

 
 
 SERAC ( Mitchell et al.,, 2022 ) utilizes a memory-based model editing approach that treats an LLM as a black-box model.
It serves as a simple wrapper around the base model and consists of three main components: an explicit cache of edits, an auxiliary scope classifier, and a counterfactual model.
When presented with a knowledge edit (which can be the unlearning request), the wrapped model predicts a new input in two steps.
Firstly, the scope classifier assesses the likelihood of the new input falling within the scope of each cached edit.
If deemed within scope, the edit with the highest probability is retrieved, and the counterfactual model provides a prediction based on both the new input and the retrieved edit.
If the new input is considered out-of-scope for all edits, the base model’s prediction is returned.
Although SERAC was evaluated on editing question answering, fact-checking, and dialogue generation, it can also be used to unlearn undesirable behaviors of LLMs and replace them with desirable ones.
This method is simple and easy to implement and requires no modifications to the original model.
Also, it can edit multiple models with different architectures.
However, it may be prone to retrieval errors, such as noise and harmful content, and knowledge conflict issues ( Zhang et al.,, 2024 ) .
Further, it relies on an edit dataset for training and may require more computational and memory resources in some settings.

 
 
 To correct model errors via user interactions without retraining, MemPrompt ( Madaan et al.,, 2022 ) pairs the model with a growing memory of recorded cases where the model misunderstood the user intent.
The system maintains a memory of the feedback as a set of key-value pairs, where the key is a misunderstood input ( e.g. , question), and the value is its correction.
Given a new prompt, the system looks in the memory for a similar prompt to check if the model has made a mistake on a similar prompt earlier.
If a match is found, the corresponding correction is appended to the prompt, and then the updated prompt is fed to the model.
In this sense, this approach can be seen as an instance of prompt engineering ( Liu et al.,, 2023 ) which involves editing the prompts.
The method was applied to lexical relations like antonyms, word scrambling such as anagrams, and ethics, where user feedback is used as
the appropriate ethical consideration in natural language.
Whereas the method is efficient and gives the end-user more control over the unlearning process, it might struggle with the scalability of the memory or in maintaining utility as the volume of user interactions grows.
Also, its effectiveness highly depends on the quality of user corrective feedback and on the success of matching the new input with its similar recorded one.

 
 
 

#### 5.4.3 In-context learning

 
 This approach exploits the in-context power of LLMs for unlearning.

 
 
 To remove the impact of a specific training point on the model’s output, ICUL ( Pawelczyk et al.,, 2023 ) constructs a specific context at inference time that makes the model classify it as if it had never seen the data point during training.
The ICUL method involves three steps:

 
 1. 
 
 Label-flipping. Given a forgetting request, the label on the corresponding training point whose influence should be removed from the model is flipped, resulting in a new template.

 

 2. 
 
 Addition of correctly labeled training points. Excluding the to-be-forgotten point, s s labeled example pairs are randomly sampled and added to the template from Step 1.

 

 3. 
 
 Prediction. The query input is added to the template, forming the final prompt, and the model predicts the next token using a temperature of 0.

 

 
 The label-flipping operation in Step 1 aims to remove the influence of a specific training point on the model outcome.
Step 2 aims to reduce the effect of the label flipping, with the number of points s s allowing for a trade-off between efficiency and utility.
The ICUL method was empirically evaluated on three classification datasets using a test called LiRA-Forget, which was also introduced to empirically measure unlearning effectiveness.
The results demonstrate that ICUL can effectively eliminate the influence of training points on the model’s output, occasionally outperforming white-box methods that necessitate direct access to the LLM parameters and are more computationally demanding.
While this method is efficient and preserves the utility of the model, its effectiveness depends on the model’s capacity for in-context learning.
It is worth noting that its efficacy was only tested with text classification tasks where the label-flipping process is applicable. Its effectiveness with other tasks remains to be determined.

 
 
 Table 1 compares the LLM unlearning methods surveyed in this paper.

 
 
 Table 1: Comparison between LLM unlearning methods. s denotes data shard. L, M and H denote low, medium and high, respectively. Sgnf denotes significant. r ( . ) r(.) is the reward model required for Quark. 
 
 
 
 
 
 
 
 Method 
 Input 
 Guarantee 
 Target 
 Application 
 Utility 
 Cost 
 
 
 
 SISA ( Bourtoule et al.,, 2021 ) 
 A ⁡ ( D s ) , D s ​ r , D s ​ f A(D_{s}),D_{sr},D_{sf} 
 Exact 
 Data point 
 Privacy, Copyright, Robustness, Toxicity 
 M-L 
 Sgnf 
 
 LOO ( Liu and Kalinli,, 2023 ) 
 A ⁡ ( D ) , { A ⁡ ( D s ) } s = 1 M − 1 , D s ​ f A(D),\{A(D_{s})\}_{s=1}^{M-1},D_{sf} 
 Approx. 
 Data point 
 Privacy, Copyright, Robustness, Toxicity 
 H-M 
 H 
 
 GA ( Jang et al.,, 2022 ) 
 A ⁡ ( D ) , D f A(D),D_{f} 
 Approx. 
 Data point 
 Privacy, Copyright, Toxicity 
 M-L 
 M 
 
 SeUL ( Wang et al.,, 2024 ) 
 A ⁡ ( D ) , D f A(D),D_{f} 
 Approx. 
 Data point 
 Privacy, Copyright, Toxicity 
 M 
 H 
 
 Yao et al., (2023) 
 A ⁡ ( D ) , D f , D a ​ u ​ x A(D),D_{f},D_{aux} 
 Approx. 
 Data point 
 Privacy, Copyright, Robustness, Toxicity, Hallucination 
 M 
 H 
 
 KGA ( Wang et al.,, 2023 ) 
 A ⁡ ( D ) , D , D f , D a ​ u ​ x A(D),D,D_{f},D_{aux} 
 Approx. 
 Data point 
 General 
 M-L 
 H 
 
 Eldan and Russinovich, (2023) 
 A ⁡ ( D ) , D f A(D),D_{f} 
 Approx. 
 Content 
 Privacy, Copyright, Toxicity 
 H-M 
 H 
 
 Quark ( Lu et al.,, 2022 ) 
 A ( D ) , D a ​ u ​ x , r ( . ) A(D),D_{aux},r(.) 
 Approx. 
 General 
 General 
 H 
 Sgnf 
 
 PCGU ( Yu et al., 2023a, ) 
 A ⁡ ( D ) , D a ​ u ​ x A(D),D_{aux} 
 Approx. 
 Concept 
 Fairness 
 H 
 M 
 
 Task vector ( Ilharco et al.,, 2022 ) 
 A ⁡ ( D ) , D f A(D),D_{f} 
 Approx. 
 Task 
 Model capability 
 H-M 
 H 
 
 F-Learning ( Ni et al.,, 2023 ) 
 A ⁡ ( D ) , D f A(D),D_{f} 
 Approx. 
 Task 
 Model capability 
 H-M 
 H 
 
 DEPN ( Wu et al.,, 2023 ) 
 A ⁡ ( D ) , D f A(D),D_{f} 
 Approx. 
 Data point 
 Privacy 
 M-L 
 M 
 
 Selective pruning ( Pochinkov and Schoots,, 2023 ) 
 A ⁡ ( D ) , D f A(D),D_{f} 
 Approx. 
 Task 
 Model capability 
 H 
 M 
 
 EUL ( Chen and Yang,, 2023 ) 
 A ⁡ ( D ) , D r , D f A(D),D_{r},D_{f} 
 Weak 
 Data point 
 Privacy, Copyright, Robustness, Toxicity 
 H-M 
 M 
 
 SISA-FC ( Kumar et al.,, 2022 ) 
 A ⁡ ( D s ) , D s ​ r , D s ​ f A(D_{s}),D_{sr},D_{sf} 
 Weak/No 
 Data point 
 Privacy, Copyright, Robustness, Toxicity 
 L 
 L 
 
 SISA-A ( Kumar et al.,, 2022 ) 
 A ⁡ ( D s ) , D s ​ r , D s ​ f A(D_{s}),D_{sr},D_{sf} 
 Weak/No 
 Data point 
 Privacy, Copyright, Robustness, Toxicity 
 M 
 M 
 
 LEACE ( Belrose et al.,, 2023 ) 
 A ⁡ ( D ) , D A(D),D 
 Weak 
 Concept 
 Fairness 
 H-M 
 M 
 
 Limisiewicz and Mareček, (2022) 
 A ⁡ ( D ) , D a ​ u ​ x A(D),D_{aux} 
 Weak 
 Concept 
 Fairness 
 H-M 
 L 
 
 Zero-shot self-debiasing ( Gallegos et al.,, 2024 ) 
 A ​ P ​ I ​ ( A ⁡ ( D ) ) , D f API(A(D)),D_{f} 
 Weak 
 Concept 
 Fairness 
 H 
 L 
 
 SERAC ( Mitchell et al.,, 2022 ) 
 A ​ P ​ I ​ ( A ⁡ ( D ) ) , D f API(A(D)),D_{f} 
 Weak 
 General 
 General 
 H 
 M 
 
 MemPrompt ( Madaan et al.,, 2022 ) 
 A ​ P ​ I ​ ( A ⁡ ( D ) ) , D f API(A(D)),D_{f} 
 Weak 
 General 
 General 
 M 
 L 
 
 ICUL ( Pawelczyk et al.,, 2023 ) 
 A ​ P ​ I ​ ( A ⁡ ( D ) ) , D f , D a ​ u ​ x API(A(D)),D_{f},D_{aux} 
 Weak 
 General 
 General 
 H 
 L 
 
 
 
 

 
 
 
 
 
 

## 6 Evaluation of unlearning in LLMs

 
 In this section, we analyze the evaluation of forgetting in LLMs, including the datasets, models, and metrics employed by the surveyed works. The analysis of metrics focuses on the main aspects discussed in Section 3.3 : i) whether the model has effectively forgotten the target knowledge, ii) whether the model retained the rest of its capabilities, and iii) the computational cost of the forgetting process. Hereafter, these measures will be referred
to as forgetting, retaining, and runtime, respectively.

 
 

### 6.1 Datasets

 
 A proper assessment of forgetting generally requires three different datasets, which we refer to as forgetting training set , forgetting test set , and retaining test set . The two forgetting sets
are the unlearning counterparts of the training and test sets employed in traditional ML: the forgetting training set comprises samples representing the knowledge that the model should forget, whereas the forgetting test set facilitates the measurement of the forgetting generalization. The retaining test set is typically disjoint from the forgetting sets and is used to assess the utility —preservation of capabilities— of the unlearned model.

 
 
 Table 2 lists the forgetting datasets used by the surveyed methods. The “Forgetting target” column specifies the forgetting request type (as defined in Section 3.2 ) and a more particular description of the undesired knowledge to be unlearned. The table shows that most works align with feature or concept removal (such as problematic generations or biases) and item removal ( e.g. , a subset of samples) requests. Only Pochinkov and Schoots, (2023) are found to work on task removal .
It is important to note that some authors select combinations of undesired knowledge and forgetting request type that do not align with real-world scenarios, although this choice could be justified for experimental purposes. On one hand, Wu et al., (2023) and Borkar, (2023) consider Personal Identifiable Information (PII) as items to be forgotten. Nevertheless, sensitive information should be treated as a concept to be forgotten, to preclude the model from providing any details compromising privacy. For example, for an individual born in Paris, even if the model does not generate the city name verbatim, it can disclose that the birthplace is ’The city of light’, sobriquet of Paris. On the other hand, Eldan and Russinovich, (2023) experiment with forgetting a copyrighted corpus (Harry Potter books) as a concept. This may be excessive to avoid copyright laws, for which it is sufficient not to generate copies of the text. Forgetting copyrighted texts as items should be enough for these cases.

 
 
 The datasets in Table 2 are split into the forgetting training and test sets. For example, Yao et al., (2023) use the training and test splits of PKU-SafeRLHF ( Ji et al.,, 2023 ) to compare the rate of harmful answers for forgotten and unseen prompts. Authors focusing on forgetting a copyrighted corpus or specific samples ( e.g. , data of an individual) do not use a forgetting test set ( Wu et al.,, 2023 ; Pawelczyk et al.,, 2023 ) . Instead, they measure the forgetting success based on the model’s inability to generate verbatim copies of the text to be forgotten or correctly process the samples whose forgetting is desired. This will be later expanded in Section 6.3 .

 
 
 Table 2: Datasets used for forgetting 
 
 
 
 
 
 
 Dataset name 
 Forgetting target 
 
 
 Paper/s 
 
 
 
 
 PKU-SafeRLHF ( Ji et al.,, 2023 ) 
 Concept: Harmful generations 
 
 
 Yao et al., (2023) 
 
 
 HaluEval ( Li et al., 2023a, ) 
 Concept: Hallucinated generations 
 
 
 Yao et al., (2023) 
 
 
 RealToxicityPrompts ( Gehman et al.,, 2020 ) 
 Concept: Toxic generations 
 
 
 Lu et al., (2022) ; Ilharco et al., (2022) 
 
 
 Civil Comments ( Borkan et al.,, 2019 ) 
 Concept: Toxic generations 
 
 
 Ilharco et al., (2022) 
 
 
 Harry Potter Universe ( Eldan and Russinovich,, 2023 ) 
 Concept: Related generations 
 
 
 Eldan and Russinovich, (2023) 
 
 
 WinoBias ( Zhao et al.,, 2018 ) 
 Concept: Gender bias 
 
 
 Limisiewicz and Mareček, (2022) 
 
 
 WinoMT ( Stanovsky et al.,, 2019 ) 
 Concept: Gender bias 
 
 
 Limisiewicz and Mareček, (2022) 
 
 
 CrowS Pairs ( Nangia et al.,, 2020 ) 
 Concept: Gender bias 
 
 
 Yu et al., 2023a () 
 
 
 Winogender Schemas ( Rudinger et al.,, 2018 ) 
 Concept: Gender bias 
 
 
 Yu et al., 2023a () 
 
 
 Bias in Bios ( De-Arteaga et al.,, 2019 ) 
 Concept: Gender bias 
 
 
 Belrose et al., (2023) 
 
 
 StereoSet ( Nadeem et al.,, 2021 ) 
 Concept: Stereotype bias 
 
 
 Yu et al., 2023a () 
 
 
 English Universal Dependencies ( Nivre et al.,, 2020 ) 
 Concept: Part-of-Speech 
 
 
 Belrose et al., (2023) 
 
 
 RedPajama ( Computer,, 2023 ) 
 Concept: Part-of-Speech 
 
 
 Belrose et al., (2023) 
 
 
 Training Data Extraction Challenge 5 5 
 5 
 
 
 
 https://github.com/google-research/lm-extraction-benchmark 
 Item: Specific generations 
 
 
 Jang et al., (2022) 
 
 
 Pile ( Gao et al.,, 2021 ) 
 Item: Samples \And Concept: PoS 
 
 
 Jang et al., (2022) ; Belrose et al., (2023) ; Pochinkov and Schoots, (2023) 
 
 
 Harry Potter and the Sorcerer’s Stone ( Rowling,, 2000 ) 
 Item: Copyright corpus 
 
 
 Yao et al., (2023) 
 
 
 SST2 ( Socher et al.,, 2013 ) 
 Item: Samples 
 
 
 Pawelczyk et al., (2023) 
 
 
 Amazon polarity ( Zhang et al.,, 2015 ) 
 Item: Samples 
 
 
 Pawelczyk et al., (2023) 
 
 
 Yelp polarity ( Zhang et al.,, 2015 ) 
 Item: Samples 
 
 
 Pawelczyk et al., (2023) 
 
 
 IMDB ( Maas et al.,, 2011 ) 
 Item: Samples 
 
 
 Chen and Yang, (2023) 
 
 
 SAMSum ( Gliwa et al.,, 2019 ) 
 Item: Samples 
 
 
 Chen and Yang, (2023) 
 
 
 LEDGAR ( Tuggener et al.,, 2020 ) 
 Item: Samples 
 
 
 Wang et al., (2023) 
 
 
 PersonaChat ( Zhang et al.,, 2018 ) 
 Item: Samples 
 
 
 Wang et al., (2023) 
 
 
 IWSLT14 ( Cettolo et al.,, 2014 ) 
 Item: Samples 
 
 
 Wang et al., (2023) 
 
 
 Enron emails ( Klimt and Yang,, 2004 ) 
 Item: Private information 
 
 
 Wu et al., (2023) ; Borkar, (2023) 
 
 
 zsRE ( Levy et al.,, 2017 ) 
 Item: Old facts 
 
 
 Ni et al., (2023) 
 
 
 CounterFact ( Meng et al.,, 2022 ) 
 Item: Old facts 
 
 
 Ni et al., (2023) 
 
 
 CodeParrot GitHub Code ( Tunstall et al.,, 2022 ) 
 Task: Programming 
 
 
 Pochinkov and Schoots, (2023) 
 
 
 
 

 
 
 
 Table 3 lists datasets used to assess the model’s retaining. The “Retaining target” column specifies the preserved capability under evaluation. Note that the zsRE ( Levy et al.,, 2017 ) dataset and those subsequently listed were also employed for forgetting purposes by the same authors (see Table 2 ). This is because the objective was to selectively forget a subset of samples, specific generations ( i.e. , private information in Enron emails ( Klimt and Yang,, 2004 ) ), a concept ( i.e. , gender bias in (Bias in Bios ( De-Arteaga et al.,, 2019 ) ) or a task ( e.g. , Python programming in CodeParrot GitHub Code ( Tunstall et al.,, 2022 ) ), while retaining the rest dataset-related skills. Several researchers choose widely recognized benchmarks for LLMs ( Eldan and Russinovich,, 2023 ; Jang et al.,, 2022 ; Yao et al.,, 2023 ; Pawelczyk et al.,, 2023 ; Chen and Yang,, 2023 ) to evaluate the preservation of overall capabilities. In contrast, a limited number of authors employ datasets with tasks that are closely aligned with the knowledge intended for forgetting ( Limisiewicz and Mareček,, 2022 ; Belrose et al.,, 2023 ; Pochinkov and Schoots,, 2023 ; Ni et al.,, 2023 ) . As will be detailed in Section 6.4 , this strategy is taken because tasks that are similar but peripheral to the focus of the forgetting process are expected to be the most affected by it.

 
 
 Table 3: Datasets only used for retaining 
 
 

 
 
 
 
 Dataset name 
 Retaining target 
 
 
 Paper/s 
 
 
 
 
 Wizard of Wikipedia ( Dinan et al.,, 2019 ) 
 Dialogue 
 
 
 Jang et al., (2022) 
 
 
 Empathetic Dialogues ( Rashkin et al.,, 2019 ) 
 Dialogue 
 
 
 Jang et al., (2022) 
 
 
 Blended Skill Talk ( Smith et al.,, 2020 ) 
 Dialogue 
 
 
 Jang et al., (2022) 
 
 
 Wizard of Internet ( Komeili et al.,, 2022 ) 
 Dialogue 
 
 
 Jang et al., (2022) 
 
 
 piqa ( Bisk et al.,, 2020 ) 
 Q A 
 
 
 Eldan and Russinovich, (2023) ; Jang et al., (2022) 
 
 
 COPA ( Gordon et al.,, 2012 ) 
 Q A 
 
 
 Jang et al., (2022) 
 
 
 ARC ( Clark et al.,, 2018 ) 
 Q A 
 
 
 Jang et al., (2022) 
 
 
 MathQA ( Amini et al.,, 2019 ) 
 Q A 
 
 
 Jang et al., (2022) 
 
 
 PubmedQA ( Jin et al.,, 2019 ) 
 Q A 
 
 
 Jang et al., (2022) 
 
 
 TruthfulQA ( Lin et al.,, 2022 ) 
 Q A 
 
 
 Yao et al., (2023) 
 
 
 GAP Coreference ( Webster et al.,, 2018 ) 
 Coreference 
 
 
 Limisiewicz and Mareček, (2022) 
 
 
 English Web Treebank ( Silveira et al.,, 2014 ) 
 Dependency 
 
 
 Limisiewicz and Mareček, (2022) 
 
 
 WinoGrande ( Sakaguchi et al.,, 2020 ) 
 Completition 
 
 
 Eldan and Russinovich, (2023) ; Jang et al., (2022) 
 
 
 HellaSwag ( Zellers et al.,, 2019 ) 
 Completition 
 
 
 Eldan and Russinovich, (2023) ; Jang et al., (2022) 
 
 
 Lambada ( Paperno et al.,, 2016 ) 
 Completition 
 
 
 Jang et al., (2022) 
 
 
 zsRE ( Levy et al.,, 2017 ) 
 Completition 
 
 
 Ni et al., (2023) 
 
 
 CounterFact ( Meng et al.,, 2022 ) 
 Completition 
 
 
 Ni et al., (2023) 
 
 
 Pile ( Gao et al.,, 2021 ) 
 Samples subset 
 
 
 Jang et al., (2022) ; Pochinkov and Schoots, (2023) 
 
 
 SST2 ( Socher et al.,, 2013 ) 
 Samples subset 
 
 
 Pawelczyk et al., (2023) 
 
 
 Amazon polarity ( Zhang et al.,, 2015 ) 
 Samples subset 
 
 
 Pawelczyk et al., (2023) 
 
 
 Yelp polarity ( Zhang et al.,, 2015 ) 
 Samples subset 
 
 
 Pawelczyk et al., (2023) 
 
 
 IMDB ( Maas et al.,, 2011 ) 
 Samples subset 
 
 
 Chen and Yang, (2023) 
 
 
 SAMSum ( Gliwa et al.,, 2019 ) 
 Samples subset 
 
 
 Chen and Yang, (2023) 
 
 
 LEDGAR ( Tuggener et al.,, 2020 ) 
 Samples subset 
 
 
 Wang et al., (2023) 
 
 
 PersonaChat ( Zhang et al.,, 2018 ) 
 Samples subset 
 
 
 Wang et al., (2023) 
 
 
 IWSLT14 ( Cettolo et al.,, 2014 ) 
 Samples subset 
 
 
 Wang et al., (2023) 
 
 
 Enron emails ( Klimt and Yang,, 2004 ) 
 Generation 
 
 
 Wu et al., (2023) ; Borkar, (2023) 
 
 
 CodeParrot GitHub Code ( Tunstall et al.,, 2022 ) 
 Generation 
 
 
 Pochinkov and Schoots, (2023) 
 
 
 WikiText-103 ( Merity et al.,, 2016 ) 
 Generation 
 
 
 Ilharco et al., (2022) 
 
 
 Bias in Bios ( De-Arteaga et al.,, 2019 ) 
 Classification 
 
 
 Belrose et al., (2023) 
 
 
 
 

 
 
 
 

### 6.2 Models

 
 Table 4 reports the LLMs used by each of the surveyed methods.
Models are sorted by their number of parameters, differing by as much as 3 orders of magnitude ( i.e. , from 11 million to 30 billion parameters).
LLMs are often released in various sizes to accommodate a wide range of use cases. The table enumerates the specific sizes used, separated by commas. In cases where multiple authors selected the same model but in different sizes, a row without the model name is used. As suggested by Carlini et al., (2021) , larger models tend to memorize more, making them more challenging and interesting for evaluation. Notably, 8 works opted for models with over 1 billion parameters.

 
 
 Table 4: LLMs used for forgetting 
 
 

 
 
 
 Model name 
 Number of parameters 
 
 
 Paper/s 
 
 
 
 
 ALBERT ( Lan et al.,, 2020 ) 
 11M 
 
 
 Yu et al., 2023a () 
 
 
 DistilBERT ( Sanh et al.,, 2019 ) 
 67M 
 
 
 Wang et al., (2023) 
 
 
 BERT ( Devlin et al.,, 2019 ) 
 110M 
 
 
 Wu et al., (2023) ; Yu et al., 2023a () ; Limisiewicz and Mareček, (2022) ; Belrose et al., (2023) 
 
 
 ELECTRA ( Clark et al.,, 2020 ) 
 110M 
 
 
 Limisiewicz and Mareček, (2022) 
 
 
 RoBERTa ( Liu et al.,, 2019 ) 
 355M 
 
 
 Pochinkov and Schoots, (2023) 
 
 
 
 125M 
 
 
 Yu et al., 2023a () 
 
 
 GPT-2 ( Radford et al.,, 2019 ) 
 117M, 355M, 774M 
 
 
 Ilharco et al., (2022) 
 
 
 
 355M, 774M 
 
 
 Lu et al., (2022) 
 
 
 Bloom ( Scao et al.,, 2022 ) 
 560M, 1.1B 
 
 
 Pawelczyk et al., (2023) 
 
 
 Phi ( Li et al., 2023b, ) 
 1.5B 
 
 
 Eldan and Russinovich, (2023) 
 
 
 GPT-Neo ( Black et al.,, 2021 ) 
 125M, 1.3B, 2.7B 
 
 
 Jang et al., (2022) 
 
 
 T5 ( Raffel et al.,, 2020 ) 
 223M, 3B 
 
 
 Chen and Yang, (2023) 
 
 
 OPT ( Zhang et al.,, 2022 ) 
 125M, 1.3B, 6.7B 
 
 
 Pochinkov and Schoots, (2023) 
 
 
 
 125M, 1.3B, 2.7B 
 
 
 Jang et al., (2022) 
 
 
 
 1.3B, 2.7B 
 
 
 Yao et al., (2023) 
 
 
 Galactica ( Taylor et al.,, 2022 ) 
 125M, 1.3B, 6.7B 
 
 
 Pochinkov and Schoots, (2023) 
 
 
 Llama-2 ( Touvron et al., 2023b, ) 
 7B 
 
 
 Yao et al., (2023) ; Ni et al., (2023) 
 
 
 Pythia ( Biderman et al.,, 2023 ) 
 160M, 1.4B, 6.9B, 12B 
 
 
 Belrose et al., (2023) 
 
 
 
 160M, 1.4B, 6.9B 
 
 
 Pochinkov and Schoots, (2023) 
 
 
 Llama ( Touvron et al., 2023a, ) 
 7B, 13B, 30B 
 
 
 Belrose et al., (2023) 
 
 
 
 7B 
 
 
 Eldan and Russinovich, (2023) 
 
 
 

 
 
 
 

### 6.3 Metrics and attacks to evaluate forgetting

 
 Most surveyed methods only offer approximate unlearning and, therefore, do not provide forgetting guarantees. In these cases, the forgetting success should be empirically evaluated ex post .

 
 
 Forgetting success is usually assessed by quantifying the decrease in the level of undesired knowledge from before to after the forgetting process. By undesired knowledge level, we mean the model’s capability/likelihood of making the kind of predictions that the model owners aim to forget ( e.g. , harmful, copyrighted, or privacy-leaking generations).
Specifically, the methods aim to minimize the undesired knowledge level in both the forgetting training and test sets. A reduction in the forgetting training set implies that seen samples are being forgotten, whereas a reduction in the forgetting test set indicates that forgetting is affecting even unseen samples ( i.e. , it is able to generalize). If the observed decrease in the training set is much more significant than in the test set, the forgetting approach is assumed to “overfit” the training samples. Considering the capabilities of LLMs for memorizing text ( Carlini et al.,, 2021 ) , it is possible that the forgetting process only worked for samples in the forgetting training set, and even a slight paraphrasing could be sufficient to obtain undesired predictions.

 
 
 In the following, we examine the metrics employed by the surveyed works to measure the attained level of forgetting:

 
 • 
 
 Dataset-specific metrics : Most papers ( Eldan and Russinovich,, 2023 ; Jang et al.,, 2022 ; Yao et al.,, 2023 ; Pawelczyk et al.,, 2023 ; Chen and Yang,, 2023 ; Ni et al.,, 2023 ; Pochinkov and Schoots,, 2023 ) leverage the predefined performance metric of the forgetting dataset, that is, the accuracy in a classification or completion dataset.

 

 • 
 
 Toxicity : Papers seeking to mitigate toxicity of generated texts ( Ilharco et al.,, 2022 ; Lu et al.,, 2022 ) sometimes rely on off-the-shelf metrics, such as Perspective API 6 6 
 6 
 
 
 
 https://github.com/conversationai/perspectiveapi or Toxicity 7 7 
 7 
 
 
 
 https://github.com/unitaryai/Toxicity . These metrics involve training an LLM on a toxicity dataset to predict the toxicity score of input text.

 

 • 
 
 Bias : Works addressing bias in LLMs ( e.g. , gender, race or stereotype) ( Belrose et al.,, 2023 ; Limisiewicz and Mareček,, 2022 ; Yu et al., 2023a, ) commonly rely on the bias-sensitive prediction probability. For example, for pronoun prediction of a person with a known occupation ( e.g. , doctor), probabilities for both pronouns are expected to be 50%. Under the premise of profession-induced gender bias, De-Arteaga et al., (2019) introduced the TPR-GAP metric, which assesses the difference (GAP) in the true positive rate (TPR) between the two genders for each occupation. On the same basis, Limisiewicz and Mareček, (2022) presented the relative gender preference (RGP) metric, which can be roughly defined as the difference in gender probabilities for texts with and without the profession name.

 

 • 
 
 Generation : For undesired knowledge without any specific metrics, evaluation often focuses on how easily the model can generate that undesired knowledge. This differs depending on whether the forgetting process aims to avoid generating exact or similar text to that of the forgetting request:

 
 – 
 
 Exact generation : Here the focus is on preventing the model from generating verbatim copies of the text to be forgotten. This is common in scenarios involving copyright, where we do not want the model generating a text in the same way as the source. Most authors consider that such exact reproductions will probably be the result of feeding the model with that undesirable text. On this basis, the perplexity metric (which is standard in text generation) is often employed ( Jang et al.,, 2022 ; Yao et al.,, 2023 ; Wu et al.,, 2023 ; Pochinkov and Schoots,, 2023 ) . This approach is sometimes followed when the objective is to avoid the generation of Personal Identifiable Information (PII). Jang et al., (2022) propose the extraction likelihood metric, which considers overlaps of n-grams, and the memorization accuracy metric, for matching of identical token sequences. They define unlearning as complete when both metrics are lower than a threshold (akin to a privacy guarantee) for the forgetting test set. Wu et al., (2023) adapt their metrics to the type of PII to forget, such as exposure for phone numbers and mean reciprocal rank for names. Note that most of these evaluations do not use a forgetting test set, but focus on exact replicates, which are the consequence of using the forgetting training set as input. This seems an insufficient approach for PII (or any other private information) since the target for privacy preservation should be to avoid the generation of the targeted sensitive concept, not to prevent a concrete way of expressing it (unlike copyright).

 

 – 
 
 Similar generation : Here any generations are considered that indicate that the model has been trained with undesired knowledge. For instance, Wang et al., (2023) measure output similarity with the text to be forgotten by using the Jensen-Shannon Divergence (JSD), the Language model Probability Distance (LPD), and the Proportion of instances with Decreased Language model Probability (PDLP), while Yao et al., (2023) 
leverage the BiLingual Evaluation Understudy (BLEU) score. Eldan and Russinovich, (2023) , who aimed to forget the Harry Potter universe corpora, opt for a more fine-grained evaluation by using a forgetting test set specifically curated to elicit related knowledge ( e.g. , “When Harry returned to class, he observed his best friends,”).

 

 
 

 • 
 
 Membership Inference Attack (MIA) : A subset of authors ( Wang et al.,, 2023 ; Chen and Yang,, 2023 ; Pawelczyk et al.,, 2023 ) use the success of a membership inference attack as a metric. This type of attack aims to determine whether a piece of knowledge was used during the (pre-)training of the model.
See Section 3.1.1 for a brief introduction to privacy issues and the role of MIA, and Section 6.3.1 for more details on specific attacks and definitions.
This inversely aligns with the forgetting objective of nullifying the influence of undesired knowledge on the model (thereby being untraceable by MIAs).

 

 
 
 
 Two different scenarios are commonly employed as starting points for forgetting, one of them being more realistic and the other one more controlled. In the more realistic scenario ( Jang et al.,, 2022 ; Yao et al.,, 2023 ; Yu et al., 2023a, ; Limisiewicz and Mareček,, 2022 ) , forgetting is directly applied to a standard (usually publicly available) LLM. The undesired knowledge is then assumed to be somehow part of the pre-training data. For example, most LLM pre-training datasets are assumed to contain hard-to-filter stereotypical biases inherent to the human-written text employed for training. This scenario imposes restrictions on the kind of knowledge to be forgotten: i) undesired knowledge must have sufficient presence in the pre-training dataset for the LLM to learn it, and ii) if the pre-training data samples related to the undesired knowledge are unknown (as it is typical in the case of biases), an external dataset is required as a representation of that knowledge because forgetting is data-driven.

 
 
 These constraints can complicate the exploration for certain forgetting tasks, leading some authors ( Eldan and Russinovich,, 2023 ; Chen and Yang,, 2023 ; Borkar,, 2023 ; Wu et al.,, 2023 ; Wang et al.,, 2023 ) to opt for a second less realistic but more controllable scenario. In this case, forgetting is applied to an LLM fine-tuned with the (dataset representing the) undesired knowledge. This way, researchers can experiment with any learnable domain while having significant control over the undesired knowledge. That control can be absolute if that knowledge set is completely disjoint from the pre-training dataset ( e.g. , very specific or not publicly available). Moreover, the original LLM can be used as the forgetting ground truth, by assuming it has (presumably) never been trained on the undesirable knowledge. In contrast, in the realistic scenario, the authors requiring a ground truth need to re-train the LLM from scratch with a version of the pre-training dataset that does not include the undesired knowledge. The lack of realism in the controlled scenario arises from two primary factors. First, undesired knowledge is the last to be learned by the model, making it possible to overfit to it. This may cause the undesired knowledge to be prominent in the model, thereby resulting in an excessively high initial undesired knowledge level and an easier (or at least unrealistic) forgetting process. The sources of the undesired knowledge within the (typically Gargantuan) pre-training dataset are usually unknown. And what is known is just that the model has that undesired behavior for very specific cases, and that undesired learning may have occurred at any time during pre-training.

 
 

#### 6.3.1 Membership Inference Attacks in LLMs

 
 Carlini et al., (2021) propose a series of black-box data extraction attacks
on pre-trained LLMs based on membership inference. They conducted attacks on GPT-2,
which is a closed model (both in parameters and training data), but had limited access to the GPT-2
training data (through colleagues) to verify their findings.
Their proposed baseline attack is to generate 200,000 samples of unconditioned text with a
generative model, beginning each sample with a special beginning-of-sequence token.
Sequences are generated using a top- k k sampling strategy until an ending-of-sequence token
is reached. The samples are then ranked based on the perplexity assigned to each sequence
by the LLM.

 
 
 The attack is then refined in two ways, by changing the sequence generation process and by
focusing on different metrics.
Regarding sample generation, the authors propose two refinements.
The first one consists in changing the temperature during sequence generation, starting with
a high temperature and reducing it progressively until it reaches a value of 1 1 . This allows
the LLM to first explore different paths in the generation of the first tokens, and then follow
a clearer sequence of (possibly) seen data. Their second approach consists in conditioning the
text generation with data from the internet. Note that LLMs are pre-trained on big datasets from
publicly accessible data on the internet and thus using data from the same sources could mean
that there are intersections between the training dataset and the attacker knowledge.
Regarding metrics, the authors use the following ones: perplexity as in the baseline attack,
the ratio of the log-perplexity assigned to the sample by the generating model over the log-perplexity
assigned to the sample by a different model (mimicking the shadow-model approach by Shokri et al., (2017) ),
the ratio of the log-perplexity over the zlib compression library,
the entropy of the sample, the ratio of perplexities between the generated sample and a lowercased
version of the same sample, and the minimum perplexity across any 50-token long sliding windows.

 
 
 The authors show that exact sequences from the training data are returned from the model,
including personal data such as names and contact information, copyrighted material such as
news pieces, and potentially security-related information, such as logs and configuration
files.
The work suggests that bigger models tend to leak more exact sequences.

 
 
 The authors provide the k k -eidetic memorization privacy/security criteria, defined as follows.

 
 
 Definition 1 
 
 Model knowledge extraction. A string s s is extractable from an LLM f θ f_{\theta} if there
exists a prefix c c such that
 s ← arg ​ max s ′ : | s ′ | = N f θ ( s ′ | c ) . s\leftarrow\argmax_{s^{\prime}:|s^{\prime}|=N}f_{\theta}(s^{\prime}|c). 

 
 
 
 Definition 2 
 
 k k -Eidetic memorization. A string s s is k k -eidetic memorized (for k ≥ 1 k\geq 1 )
by an LLM f θ f_{\theta} if s s is extractable from f θ f_{\theta} and s s appears in at most k k examples
in the training data X X , that is | { x ∈ X : s ⊆ x } | ≤ k |\{x\in X:s\subseteq x\}|\leq k . 

 
 
 
 Thus, two limitations on memorization are introduced.
The first is the length of the memorized strings.
The second is the number k k of times that a given string appears in the training data.

 
 
 Nasr et al., (2023) follow up on the work by Carlini et al., (2021) by executing attacks
on open-source and semi-open large language models. The key difference in this work is that the
adversaries have access to the training dataset or at least enough information to build a similar
dataset.
The authors confirm the findings of Carlini et al., (2021) , including the observation
that larger and more capable models are more vulnerable to data extraction attacks.
Then, they conduct attacks on ChatGPT (with gpt-3.5-turbo) and find that the same approach
does not make the models leak any information.

 
 
 The authors suggest that fine-tuning –including RLHF– to make the models act as conversational
agents prevents them from reverting to training data from their pre-training phase.
However, the authors launch a divergence attack .
As mentioned above, generative models sample from the distribution of tokens conditioned
on the previous tokens. In their attack, they make ChatGPT repeat a single token, expecting that
the generation will diverge from the expected path. In the example provided in the paper, the
authors make ChatGPT repeat the sequence “poem, poem, poem…” indefinitely. Even if the most
probable token at each generation step is the word “poem”, nonzero probabilities are assigned
to other tokens. Sampling repeatedly from the same distribution may cause the generator to output
a low-probability token, thus diverging from the expected path. Once the generation diverges,
ChatGPT seems to fall back to memorized data. By comparing the behavior of gpt-3.5-turbo and
gpt-3.5-instruct-turbo, which are (presumably) fine-tuned on different data, the authors conclude
that the leaked data comes from the data used during model pre-training, which likely includes
personal, sensitive, and copyrighted material. Namely, the authors recover personal data,
inappropriate content, fragments of books, URLs, UUIDs, code snippets, research papers, boilerplate
texts, and mixed memorized data.

 
 
 The authors provide two security/privacy criteria.

 
 
 Definition 3 
 
 Extractable memorization. Given a generative large language model G ​ e ​ n Gen , an example s s from the training set X X is extractably memorized if an adversary (without access to X X ) can construct a prompt c c that makes the model produce s s (i.e., G ​ e ​ n ​ ( c ) = s Gen(c)=s ). 

 
 
 
 Definition 4 
 
 Discoverable memorization. For a model G ​ e ​ n Gen and an example [ c | | s ] [c||s] from the training set X X , we say that s s is discoverably memorized if G ​ e ​ n ​ ( c ) = s Gen(c)=s . 

 
 
 
 Patil et al., (2023) present two white-box access data recovery attacks.
In this case, the authors attempt to extract information from the models after some
digital forgetting procedure has been applied to them.

 
 
 In this paper, similarly to previous works, the objective of the attacker is to obtain
some answer s s to a question c c , where the pair ( c , s ) (c,s) is sensitive information.
However, the authors present several threat models defined by how many attempts the
attacker has. In particular, the attacker is successful if the answer s s is within
a set of candidate answers 𝒮 \mathcal{S} .
The size | 𝒮 | = B |\mathcal{S}|=B of the candidate set is defined as the attack budget.

 
 
 The attack scenarios are as follows:

 
 1. 
 
 Password attempts . The attacker does not know s s but can verify its correctness in B B attempts. This represents an attacker trying to recover a password from a model to steal a personal account.

 

 2. 
 
 Parallel pursuit . Again, the attacker does not know s s ex ante but can use the information in S S in parallel (and does not care if some candidates are incorrect). An example of this scenario is an attacker building an address book from leaked email addresses to conduct spam operations.

 

 3. 
 
 Verification by data owner . The attacker is the owner of datapoint s s and wants it to be deleted from the model.

 

 
 
 
 The authors propose two white-box attacks based on the logit lens
 ( Nostalgebraist,, 2020 ) 
The logit lens is an interpretability tool that computes the product of the intermediate activations
in (decoder-only) pre-trained transformers with the embedding layer, obtaining intermediate distributions
over the vocabulary. The logit lens can be used to explore how the prediction of a model evolves after
each intermediate block.

 
 
 Their first attack is the Head Projection attack. It consists of obtaining the logit lens distribution of every layer in the LLM and taking the top- k k candidates in each distribution. The candidate set S S is the union of these candidates.
Their second attack leverages changes in the probability assigned to each token
and is called the Probability Delta attack.
As in the previous attack, first the logit lens distributions of every layer are obtained.
Then, the differences in the distributions between every two consecutive layers are obtained.
Finally, the top- k k and bottom- k k candidates from these differences are included
in the candidate list S S .

 
 
 The authors define their attack success metric as follows.

 
 
 Definition 5 
 
 Attack Success Metric. Given datapoints c i , s i i = 1 N {c_{i},s_{i}}_{i=1}^{N} : 

 

 
 | 
 A t t a c k S u c c e s s @ B ( ℳ ) = 1 N ∑ i = 1 N 𝟙 [ s i ∈ S i ] AttackSuccess@B(\mathcal{M})=\frac{1}{N}\sum_{i=1}^{N}\mathds{1}[s_{i}\in S_{i}] | 
 | 
 

 where S i S_{i} is the candidate set produced for model ℳ \mathcal{M} on datapoint c i c_{i} (with | S i | = B |S_{i}|=B ),
and 𝟙 ​ [ ⋅ ] \mathds{1}[\cdot] is the indicator function. 

 
 
 
 Note that this definition integrates concepts from Definition 4 , in which both the question and answers are known, but accounts for multiple attempts at extraction.

 
 
 
 

### 6.4 Retaining evaluation

 
 The assessment of model retaining often involves evaluating the utility difference of the LLM before and after the forgetting process on tasks different from the undesired knowledge. The smaller the utility difference, the better the retention. Ideally, retention evaluation should encompass all possible knowledge except that to be forgotten. Nonetheless, this is unfeasible in most cases, given the wide range of capabilities that LLMs can learn. Instead, researchers define the forgetting test set by selecting the subset of the most relevant tasks for which utility should be retained.

 
 
 In the following, we list the most prominent strategies used to measure retained utility:

 
 • 
 
 General benchmarks . Eldan and Russinovich, (2023) , Wang et al., (2023) , and Jang et al., (2022) leverage well-established LLM benchmarks, such as those used for comparing LLMs ( Sakaguchi et al.,, 2020 ; Zellers et al.,, 2019 ; Paperno et al.,, 2016 ; Bisk et al.,, 2020 ) . These benchmarks assess the model’s capabilities in natural language understanding, reasoning, and human-like generation. In other words, this strategy aims to measure the preservation of the model’s general capabilities. The utility on these general tasks is compared to find the model’s weaknesses and strengths.

 

 • 
 
 Related but different . A reasonable assumption is that knowledge conceptually close to that to be forgotten will be the first and most affected by the forgetting process.
Therefore, the (declining) utility for this knowledge may be an appropriate indicator of whether the model is retaining the remaining capabilities or forgetting is affecting more than just the undesired knowledge. Following this principle, several authors evaluate retaining on knowledge domains related but different to the undesired knowledge ( Yao et al.,, 2023 ; Lu et al.,, 2022 ; Limisiewicz and Mareček,, 2022 ; Belrose et al.,, 2023 ; Chen and Yang,, 2023 ; Wang et al.,, 2023 ; Pawelczyk et al.,, 2023 ; Wu et al.,, 2023 ; Ni et al.,, 2023 ; Pochinkov and Schoots,, 2023 ) . The specific strategies employed differ in terms of closeness and/or relatedness to the forgetting target:

 
 
 
 – 
 
 Related dataset ( Lu et al.,, 2022 ; Yao et al.,, 2023 ; Limisiewicz and Mareček,, 2022 ; Ilharco et al.,, 2022 ) :
These proposals select a dataset different from that used for forgetting, but such that it is more or less related to the undesired knowledge. For instance, Lu et al., (2022) measured toxicity, fluency and diversity in the WritingPrompts dataset ( Fan et al.,, 2018 ) to measure the retaining of the model’s writing capabilities when forgetting toxic generations from the RealToxicPrompts dataset ( Gehman et al.,, 2020 ) . Similarly, Yao et al., (2023) leveraged BLEURT ( Sellam et al.,, 2020 ) and the deberta-v3-large-v2 reward model 8 8 
 8 
 
 
 
 https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2 to compare the generations for TruthfulQA ( Lin et al.,, 2022 ) questions by the original LLM with those resulting from unlearning harmful Q A pairs of PKU-SafeRLHF ( Ji et al.,, 2023 ) .

 

 – 
 
 Same dataset but different samples ( Chen and Yang,, 2023 ; Wang et al.,, 2023 ; Pawelczyk et al.,, 2023 ; Ni et al.,, 2023 ; Pochinkov and Schoots,, 2023 ) : Methods that aim to forget a subset of samples from a dataset often evaluate retaining on the remaining samples of that same dataset. For example, Pochinkov and Schoots, (2023) forget Python code samples from Tunstall et al., (2022) while aiming to retain the remaining code samples.

 

 – 
 
 Same dataset but different task ( Belrose et al.,, 2023 ; Wu et al.,, 2023 ) : The forgetting (training and/or test) dataset is used to measure the model’s capabilities in a task disjoint of that to be forgotten. For instance, the Bias in Bios dataset ( De-Arteaga et al.,, 2019 ) is employed by Belrose et al., (2023) for both forgetting gender bias and measuring retaining in the job prediction task. Another example is provided by Wu et al., (2023) , who aim to forget
personally identifiable information from the Enron emails dataset ( Klimt and Yang,, 2004 ) , and measure retaining as the generation quality for the rest of the corpus.

 

 
 

 
 
 
 The utility degradation observed in these studies usually falls within the range of 1% to 5%, although it can occasionally reach up to 20%. It has been noted by Jang et al., (2022) and Pawelczyk et al., (2023) that larger models tend to exhibit better utility retention.

 
 
 

### 6.5 Runtime evaluation

 
 The runtime of forgetting should always be significantly lower than that required for retraining the LLM without the undesired knowledge. Such retraining from scratch is the straightforward but resource-intensive process
whose avoidance is sought by the surveyed methods. However, forgetting is rarely a “fast” procedure, since it usually involves costly training-related steps such as fine-tuning.

 
 
 Wang et al., (2023) ; Chen and Yang, (2023) ; Limisiewicz and Mareček, (2022) ; Jang et al., (2022) ; Yao et al., (2023) ; Lu et al., (2022) ; Yu et al., 2023a () explicitly report the runtime of their methods. Values range from minutes ( Wang et al.,, 2023 ; Yao et al.,, 2023 ; Limisiewicz and Mareček,, 2022 ; Jang et al.,, 2022 ; Yu et al., 2023a, ) to hours ( Chen and Yang,, 2023 ) to even more than a day ( Lu et al.,, 2022 ) .
However, directly comparing the runtimes reported by different works would be unfair, due to the different hardware configurations, models, and/or undesired knowledge.

 
 
 There are some workarounds to compensate for hardware discrepancies, although they are not universally applicable. Firstly, runtimes can be normalized to the most common GPU, adjusting runtimes from other GPUs based on their relative performance in deep learning, as assessed through specific benchmarks. However, the substantial memory requirements of LLMs often require multi-GPU configurations with the corresponding interconnections. The latency of these interconnections also influences runtimes, but this is rarely reported in the literature.
Secondly, for those methods whose temporal cost is predominantly determined by fine-tuning, the number of epochs can be used as a more general indicator of cost. Nonetheless, this information is only reported by a subset of works ( Eldan and Russinovich,, 2023 ; Chen and Yang,, 2023 ; Jang et al.,, 2022 ; Yao et al.,, 2023 ; Yu et al., 2023a, ; Wu et al.,, 2023 ; Ni et al.,, 2023 ; Ilharco et al.,, 2022 ) and can only be used when comparing identical models, data and forgetting tasks, which is rarely the case.

 
 
 
 

## 7 Challenges and potential solutions

 
 Section 3 discussed the motivations, types, and requirements of digital forgetting.
In this section, we evaluate the surveyed methods in terms of the presented requirements, focusing on
the provided forgetting guarantees, the performance of the resulting models, and the computational costs
and scalability of the methods.

 
 

### 7.1 Guarantees of forgetting

 
 The forgetting guarantees presented in Section 3.3 were mainly devised for general neural network classifiers.
Although they can still be used for LLMs, some factors limit their applicability.
On one hand, the guarantees refer to the parameters of the resulting models after training or forgetting.
That means that, for an ex ante guarantee of forgetting, a clear approximation of the influence of every training sample on the model weights is known or can be computed.
While this may be feasible for small or simpler models, the complexity of LLMs may make such analyses difficult if not impossible.

 
 
 On the other hand, while unlearning an image or a record is relatively straightforward in typical machine unlearning, the complexity increases when dealing with text data in LLMs.
If the goal is to unlearn a text sequence containing sensitive information, not only is identifying explicit
sensitive data difficult, but identifying implicit
sensitive data (data that may allow sensitive inferences) is even more challenging.
This challenge also applies when unlearning copyrighted documents, especially in non-fiction texts, where identifying the unique tokens associated with these documents becomes tedious.
When the aim is to eliminate model bias or hallucination, the complexity increases further.
Bias is often widespread and difficult to detect in training text data, as it appears in scattered patterns across many examples in both explicit and implicit forms.
Thus, unlearners often resort to identifying biased concepts within internal model representations and addressing them at this level.
Similarly, addressing hallucinations poses a significant challenge, as they often stem from multiple sources, making identification a nontrivial task.

 
 
 Furthermore, methods that do not result in complete and permanent changes to model parameters can still facilitate the extraction of sensitive information.
The study by Maini et al., (2024) evaluates the effectiveness of several existing unlearning methods for LLMs.
It concludes that none of these methods fully achieve effective unlearning, which indicates a need for developing more effective approaches that ensure models behave as if they were never trained on the data intended to be forgotten.
Large models contain many parameters representing numerous data dimensions and correlations, and this makes it hard to pinpoint the effect of specific training data.
Existing LLM unlearning methods are usually context and task-dependent and their generality remains unclear.

 
 
 Potential solution. The ideal method should focus on directly erasing sensitive data from model parameters, and ensuring that they remain inaccessible, thus complying with privacy regulations ( Zhang et al.,, 2023 ) .
Additionally, this approach protects against the potential threat of extracting sensitive data through white-box attacks, particularly when LLMs are publicly available to adversaries possessing the technical expertise required to access sensitive information stored within model parameters or hidden states ( Patil et al.,, 2023 ) .

 
 
 We next examine the limitations of the forgetting achieved by several families of method.

 
 
 As an example, Xu et al., (2023) classify some of the surveyed forgetting mechanisms as exact or strong, but most of them are simple models such as linear regression, which can be amenable to such kinds of analyses.
On the other hand, given some of the approaches, like fine-tuning using adapters (including LoRA), the applicability of these guarantees seems challenging since they make obvious changes to the model’s architecture. At most, such forgetting procedures could be considered as providing weak forgetting guarantees.

 
 
 Another example are methods based on RLHF. These may still know the information they were supposed to forget, which could be triggered and produced with adversarial prompts ( Zou et al.,, 2023 ) .

 
 
 Regarding direct modification methods, while they offer a promising approach to efficiently and permanently remove to-be-forgotten information from LLMs, Patil et al., (2023) observed that sensitive information that is ‘deleted’ from LLMs by model editing can still be extracted from the model’s hidden states. Also, applying an editing method for one question may not effectively erase information across rephrased versions of the same question.
These shortcomings are due to the non-linearity of LLMs.

 
 
 Black-box methods do not completely erase the to-be-forgotten knowledge from the model because it does not alter the model’s parameters. This might not be compliant with privacy and copyright regulations.

 
 
 In-context learning methods may be prone to knowledge conflict issues ( Zhang et al.,, 2024 ) and their unlearning effectiveness depends on the in-context learning capability of the used LLMs.
A recent study by Yu et al., 2023b () explores different scenarios where LLMs choose between in-context and memorized answers.
These scenarios often involve a trade-off between relying on the model’s pre-existing knowledge and the new information provided in the context.
This study underscores the need for further investigation into when and how to use in-context learning methods.

 
 
 

### 7.2 Retaining of model utility

 
 Methods based on loss maximization are very sensitive to
hyperparameter choosing ( e.g. , learning rate, number of fine-tuning epochs).
A wrong choice of parameters can cause the model to lose its language understanding.

 
 
 On the other hand, unlearning a single fact can have intricate implications due to the interconnected nature of knowledge acquired by LLMs.
This interconnection means that the removal or modification of one piece of information can potentially ripple through the network, altering the way other facts are represented or understood.
Given this complexity, there is a need for a more profound investigation and comprehensive evaluation of the unlearning effects.
Such an investigation would not only illuminate the direct consequences of unlearning but also reveal any indirect effects caused by the interconnectedness of knowledge.

 
 
 Also, the requirement of the retain set for unlearning is not only restrictive but also poses an unrealistic assumption ( Maini et al.,, 2024 ) .
Therefore, it becomes crucial for contemporary methods to pivot towards a more feasible approach.

 
 
 Another thorny issue is the model’s ability to maintain its utility in the face of numerous unlearning requests.
How does the frequency of unlearning requests influence the overall performance of the model?
This is a crucial area of study to ensure the robustness of LLMs amidst continuous updates and modifications.
Existing methods predominantly utilize static forget rates for performance evaluation, and typically process forget requests in a single batch.
This approach may not accurately represent real-world scenarios where forget requests may occur sporadically and at varying intervals.

 
 
 Potential solution. Future research should focus on examining individual forget requests that arise sequentially but at distant time intervals.
This would provide a more comprehensive understanding of their impact on the model’s utility and scalability, especially when dealing with a large volume of successive forgetting requests. This shift in focus could potentially lead to more robust and adaptable LLMs.

 
 
 

### 7.3 Generalization of unlearning

 
 Patil et al., (2023) proposed six different defense methods to counter extraction attacks but concluded that no single universally effective defense method exists against all extraction attacks.

 
 
 Potential solution. One could combine one or more unlearning methods with model editing to create a multifaceted and effective defense strategy against extraction attacks.

 
 
 In existing methods, there is no general method that is suitable for all forgetting/unlearning purposes. Some methods are suitable for meeting privacy/copyright requirements. Other methods are suitable for debiasing models.

 
 
 Potential solution. One could combine several specialized methods to obtain a general unlearning method.

 
 
 

### 7.4 Runtime and scalability

 
 The unlearning approach is an appropriate choice when the priority is to stop generating undesirable outputs rather than trying to generate desirable ones.
It becomes more attractive when computational resources are limited or the model operator does not want to spend significant time eliminating undesirable outputs.

 
 
 White-box methods encounter several challenges.
These include computational and memory requirements due to the huge number of parameters, risks of overfitting, and catastrophic forgetting.
Such methods are widely used but: i) they are likely to overfit when data used for fine-tuning are small; ii) they incur high computational costs even with efficient fine-tuning methods; iii) they could lead to losing learned knowledge in the pre-trained parameters due to modifying them with no constraints.

 
 
 Methods relying on training data sharding provide a stronger forgetting guarantee but require significant computational and storage resources, and they are time-consuming. With LLMs, they are not practical.
Moreover, these methods are unsuitable for addressing biases in LLMs.
In instances where bias patterns permeate across all shards, necessitating retraining of all segments, such methods will require significant computations.

 
 
 Potential solution. Progress is being made in modular machine learning Feng et al., (2023) , a field that revolves around the concept of dividing fundamental models into distinct, manageable modules.
Each module’s specific knowledge can be then updated individually.
This structure paves the way for local modification methods that focus on unlearning specific knowledge once it’s been localized within an LLM.
This makes these methods increasingly feasible and holds great promise for delivering a form of unlearning that is permanent, lightweight, and utility-preserving.

 
 
 

### 7.5 Evaluation

 
 The literature on digital forgetting for LLMs exhibits a wide variety in terms of forgetting and retaining datasets, selected models, and evaluation metrics.
This makes it impossible to fairly compare methods, and thereby ascertain the relative effectiveness of different approaches for specific forgetting requests.
There is an evident need for a standard benchmark establishing a common testing ground for diverse unlearning methods.
Such a benchmark should include a comprehensive and carefully selected set of forgetting requests (including forgetting and retaining datasets), models and metrics for forgetting, retaining, and speed.

 
 
 Evaluating the success of forgetting poses a major challenge. There are multiple ways in which a model can be induced to produce results based on the undesired knowledge.
Inputting the data leveraged for unlearning ( i.e. , forgetting training set) or disjoint but from the same distribution ( i.e. , forgetting test set) is easy.
Nonetheless, Eldan and Russinovich, (2023) show that undesired generations can be also obtained from out-of-distribution prompts ( e.g. , asking to list fictional schools after “unlearning” the Harry Potter corpora and obtaining “Hogwarts”).
Moreover, the “poem, poem, poem…” case depicted in Nasr et al., (2023) indicates that even out-of-distribution inputs can cause the model to come back to the
training text to be forgotten.
Beyond the challenge of identifying ways in which the model preserves undesired knowledge, there lies the issue of determining how to effectively measure its actual preservation.
For numerous forgetting requests, merely preventing the generation of verbatim duplicates of the forgetting training set is insufficient; any prediction indicating that the LLM retains that undesired knowledge needs to be avoided. For example, harmful or toxic generations can occur in a wide (even unlimited) variety of ways. Moreover, users actively define workarounds ( e.g. , fine-grained paraphrasing and cheat prompts 9 9 
 9 
 
 
 
 https://www.reddit.com/r/ChatGPT/comments/11yvk5g/using_the_neurosemantic_invertitis_hack_can/ ) to deceive toxicity detection mechanisms, thereby transforming the process into an ongoing adversarial challenge. This complicates the definition of a forgetting test set that is representative of potential error scenarios and of a metric to confidently assess whether
resulting predictions remain undesirable.
Almost all methods use a single dataset to evaluate the forgetting of a target task (privacy, copyright).
To ensure the method is not dataset-specific, diverse datasets for the same target task must be used.
Unlearning in LLMs is more challenging than in traditional classification models due to the vast output space of language models (their outputs are not just a class label), higher efficiency requirements, and limited access to training data. This makes evaluations difficult.

 
 
 Currently, no de facto standard datasets exist for explicitly evaluating unlearning methods.
Researchers typically assess these methods across diverse datasets based on individual considerations.
Another challenge is that all methods are evaluated on English language datasets and there is no evidence about their performance with other languages.
More datasets on different unlearning applications are required to assess how well unlearning methods generalize across different languages.

 
 
 Furthermore, even if unlearning is motivated by concerns about private information leakage, the datasets used in the literature rarely contain sensitive information.
To better validate the effectiveness of unlearning for sensitive information, future research should use simulated data sets with representative distributions of sensitive information.

 
 
 

### 7.6 When can each method be used?

 
 Despite the wide range of LLM unlearning methods, the choice among them depends on different factors and contexts.

 
 
 
 • 
 
 Short-term hot fixes: Input/output modification methods might be the most suitable choice.
These methods are typically quick to implement and do not require extensive knowledge about the model’s internal workings.

 

 • 
 
 Medium-term fixes: Local modification or fine-tuning-based approaches could be more appropriate.
These methods offer a balance between speed and thoroughness, allowing for more targeted adjustments to the model.

 

 • 
 
 Long-term fixes: Retraining the model from scratch while excluding the problematic data may be the best option.
This approach is often accompanied by the availability of new training data and the obsolescence of previous data.

 

 
 
 
 The required unlearning guarantee related to the target application also plays a crucial role in determining the method used.

 
 
 
 • 
 
 To prevent bias, toxic predictions, or the production of copyrighted data, any method that can fix the issue (even a black-box method) will suffice.
The primary concern here is ensuring the final output delivered to the end user does not contain undesirable behavior.

 

 • 
 
 However, when it comes to privacy and fulfilling the right to be forgotten, a method is needed that ensures the associated knowledge is removed from the internal weights themselves and that provides the greatest forgetting guarantee possible.

 

 
 
 
 It is important to note that most bias removal methods focus on identifying the source (concept) of bias in the model’s internal representations.
These methods may not be as useful in other applications such as privacy, copyright, and detoxification.
Similarly, methods that focus on unlearning knowledge associated with the input tokens themselves may not be effective for bias reduction, as they do not address biased concepts produced later in the model’s internal representation.

 
 
 

### 7.7 Black-box access scenario

 
 In recent times, there has been a surge in the utilization of LLMs across various services provided by third-party entities to the end users. In these scenarios, both the third-party service provider and the end user interact with the model as a black box, having no direct access to its internal workings or parameters.
Occasionally, there may arise a need to ‘forget’ or eliminate certain undesirable behaviors that manifest in the model’s output. This necessitates the intermediary operator, who mediates the interaction between the end user and the model, to have effective strategies in place to address such instances.

 
 
 One promising approach to tackle this issue is the use of input/output modification methods.
However, while these methods can be suitable for some cases, they may not always provide the optimal solution.
A potentially more effective strategy could involve fostering increased interaction between the end user, the intermediary operator, and the primary operator of the model, who has full access to its components.
This collaborative approach could facilitate the development of more robust unlearning solutions, allowing for the dynamic adjustment of the model’s behavior based on user feedback and real-time requirements.

 
 
 

### 7.8 Reconciling effectiveness, utility and efficiency

 
 The primary challenge of unlearning in large language models (LLMs) is achieving a balance between three critical aspects: the effectiveness of unlearning, the utility of the unlearned model, and the efficiency of the unlearning process. As of now, no existing method has been able to satisfactorily reconcile these conflicting objectives.

 
 
 For instance, a retraining approach such as SISA accomplishes exact unlearning. However, this exact guarantee comes at a significant cost. The retraining approach is computationally intensive and imposes significant memory demands, which makes it unsuitable for LLMs due to their large scale and complexity.

 
 
 On the other hand, black-box methods, which operate without needing access to the model’s internal parameters, are able to maintain the utility of the model and are generally more lightweight. However, these methods offer weak unlearning guarantees: they may not always effectively remove the undesired behavior from the model’s output.

 
 
 This challenge underscores the urgent need for future research to focus on finding a better alignment between these contradictory requirements. The goal should be to develop unlearning methods that are not only effective and efficient but also preserve the utility of the model. This would represent a significant advancement in the field, and it would enable safer and more flexible use of LLMs in a wide range of applications.

 
 
 
 

## 8 Conclusions

 
 This document has surveyed the state of the art on unlearning in LLMs. We have first given background on LLMs. Then we have reviewed the motivation, the types, and the requirements of digital forgetting. Next, we have described the main approaches used by digital forgetting methods in LLMs. After that, we have surveyed the literature by grouping unlearning methods in four categories: global weight modification, local weight modification, architecture modification, and input/output modification. After the description of the literature, we have described the way the proposed methods are evaluated: which datasets are used; to which LLMs models is unlearning applied; the metrics and attacks used to measure forgetting; the metrics used to measure retaining on the tasks to be preserved after forgetting; and the runtime of unlearning methods.

 
 
 Last but not least, we have identified a good number of challenges in the current state of the art. It can be concluded that, if machine unlearning in general is a hot topic far from having reached maturity, this is even truer when we talk about machine unlearning in LLMs. The size and the unprecedented power of LLMs can only add challenges to this still nascent area.

 
 
 

## 9 Acknowledgments

 
 This work has been funded by Huawei Technologies Finland Research Center.

 
 
 

## References

 
 
 Abadi et al., (2016) 
 
Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K.,
and Zhang, L. (2016).

 
 Deep learning with differential privacy.

 
 In Proceedings of the 2016 ACM SIGSAC conference on computer and
communications security , pages 308–318.

 

 
 Amini et al., (2019) 
 
Amini, A., Gabriel, S., Lin, S., Koncel-Kedziorski, R., Choi, Y., and
Hajishirzi, H. (2019).

 
 MathQA: Towards interpretable math word problem solving with
operation-based formalisms.

 
 In Burstein, J., Doran, C., and Solorio, T., editors, Proceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies,
Volume 1 (Long and Short Papers) , pages 2357–2367, Minneapolis, Minnesota.
Association for Computational Linguistics.

 

 
 Bahdanau et al., (2014) 
 
Bahdanau, D., Cho, K., and Bengio, Y. (2014).

 
 Neural machine translation by jointly learning to align and
translate.

 
 arXiv preprint arXiv:1409.0473 .

 

 
 Belrose et al., (2023) 
 
Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., and
Biderman, S. (2023).

 
 Leace: Perfect linear concept erasure in closed form.

 
 arXiv preprint arXiv:2306.03819 .

 

 
 Biderman et al., (2023) 
 
Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K.,
Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al.
(2023).

 
 Pythia: A suite for analyzing large language models across training
and scaling.

 
 In International Conference on Machine Learning , pages
2397–2430. PMLR.

 

 
 Bisk et al., (2020) 
 
Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y. (2020).

 
 PIQA: reasoning about physical commonsense in natural language.

 
 In The Thirty-Fourth AAAI Conference on Artificial
Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of
Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium
on Educational Advances in Artificial Intelligence, EAAI 2020, New York,
NY, USA, February 7-12, 2020 , pages 7432–7439. AAAI Press.

 

 
 Black et al., (2021) 
 
Black, S., Gao, L., Wang, P., Leahy, C., and Biderman, S. (2021).

 
 Gpt-neo: Large scale autoregressive language modeling with
mesh-tensorflow.

 

 
 Blanco-Justicia et al., (2023) 
 
Blanco-Justicia, A., Sánchez, D., Domingo-Ferrer, J., and Muralidhar, K.
(2023).

 
 A critical review on the use (and misuse) of differential privacy in
machine learning.

 
 ACM Computing Surveys , 55(8):1–16.

 

 
 Borkan et al., (2019) 
 
Borkan, D., Dixon, L., Sorensen, J., Thain, N., and Vasserman, L. (2019).

 
 Nuanced metrics for measuring unintended bias with real data for text
classification.

 
 In Companion Proceedings of The 2019 World Wide Web Conference ,
WWW ’19, page 491–500, New York, NY, USA. Association for Computing
Machinery.

 

 
 Borkar, (2023) 
 
Borkar, J. (2023).

 
 What can we learn from data leakage and unlearning for law?

 
 arXiv preprint arXiv:2307.10476 .

 

 
 Bourtoule et al., (2021) 
 
Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A.,
Zhang, B., Lie, D., and Papernot, N. (2021).

 
 Machine unlearning.

 
 In 2021 IEEE Symposium on Security and Privacy (SP) , pages
141–159. IEEE.

 

 
 Brown et al., (2020) 
 
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P.,
Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S.,
Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler,
D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray,
S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever,
I., and Amodei, D. (2020).

 
 Language models are few-shot learners.

 

 
 Carlini et al., (2021) 
 
Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K.,
Roberts, A., Brown, T., Song, D., Erlingsson, U., et al. (2021).

 
 Extracting training data from large language models.

 
 In 30th USENIX Security Symposium (USENIX Security 21) , pages
2633–2650.

 

 
 Cettolo et al., (2014) 
 
Cettolo, M., Niehues, J., Stüker, S., Bentivogli, L., and Federico, M.
(2014).

 
 Report on the 11th IWSLT evaluation campaign.

 
 In Federico, M., Stüker, S., and Yvon, F., editors, Proceedings of the 11th International Workshop on Spoken Language
Translation: Evaluation Campaign@IWSLT 2014, Lake Tahoe, CA, USA, December
4-5, 2014 .

 

 
 Chen and Yang, (2023) 
 
Chen, J. and Yang, D. (2023).

 
 Unlearn what you want to forget: Efficient unlearning for llms.

 
 arXiv preprint arXiv:2310.20150 .

 

 
 Cho et al., (2014) 
 
Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
Schwenk, H., and Bengio, Y. (2014).

 
 Learning phrase representations using rnn encoder-decoder for
statistical machine translation.

 
 arXiv preprint arXiv:1406.1078 .

 

 
 Christiano et al., (2017) 
 
Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D.
(2017).

 
 Deep reinforcement learning from human preferences.

 
 Advances in neural information processing systems , 30.

 

 
 Clark et al., (2020) 
 
Clark, K., Luong, M., Le, Q. V., and Manning, C. D. (2020).

 
 ELECTRA: pre-training text encoders as discriminators rather than
generators.

 
 In 8th International Conference on Learning Representations,
ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net.

 

 
 Clark et al., (2018) 
 
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and
Tafjord, O. (2018).

 
 Think you have solved question answering? try arc, the AI2
reasoning challenge.

 
 CoRR , abs/1803.05457.

 

 
 Computer, (2023) 
 
Computer, T. (2023).

 
 Redpajama: an open dataset for training large language models.

 
 https://github.com/togethercomputer/RedPajama-Data .

 

 
 Daheim et al., (2023) 
 
Daheim, N., Dziri, N., Sachan, M., Gurevych, I., and Ponti, E. M. (2023).

 
 Elastic weight removal for faithful and abstractive dialogue
generation.

 
 arXiv preprint arXiv:2303.17574 .

 

 
 De-Arteaga et al., (2019) 
 
De-Arteaga, M., Romanov, A., Wallach, H. M., Chayes, J. T., Borgs, C.,
Chouldechova, A., Geyik, S. C., Kenthapadi, K., and Kalai, A. T. (2019).

 
 Bias in bios: A case study of semantic representation bias in a
high-stakes setting.

 
 In danah boyd and Morgenstern, J. H., editors, Proceedings of
the Conference on Fairness, Accountability, and Transparency, FAT* 2019,
Atlanta, GA, USA, January 29-31, 2019 , pages 120–128. ACM.

 

 
 Devlin et al., (2019) 
 
Devlin, J., Chang, M., Lee, K., and Toutanova, K. (2019).

 
 BERT: pre-training of deep bidirectional transformers for language
understanding.

 
 In Burstein, J., Doran, C., and Solorio, T., editors, Proceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies,
NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and
Short Papers) , pages 4171–4186. Association for Computational Linguistics.

 

 
 Devlin et al., (2018) 
 
Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2018).

 
 Bert: Pre-training of deep bidirectional transformers for language
understanding.

 
 arXiv preprint arXiv:1810.04805 .

 

 
 Dinan et al., (2019) 
 
Dinan, E., Roller, S., Shuster, K., Fan, A., Auli, M., and Weston, J. (2019).

 
 Wizard of wikipedia: Knowledge-powered conversational agents.

 
 In International Conference on Learning Representations .

 

 
 Eldan and Russinovich, (2023) 
 
Eldan, R. and Russinovich, M. (2023).

 
 Who’s harry potter? approximate unlearning in llms.

 
 arXiv preprint arXiv:2310.02238 .

 

 
 European Commission, (2019) 
 
European Commission (2019).

 
 Ethics guidelines for trustworthy AI .

 
 Publications Office of the EU.

 

 
 European Parliament and Council of the European Union, (2016) 
 
European Parliament and Council of the European Union (2016).

 
 Regulation (EU) 2016/679 of the European Parliament and of the
Council of 27 April 2016 on the protection of natural persons with regard
to the processing of personal data and on the free movement of such data, and
repealing Directive 95/46/EC (General Data Protection
Regulation).

 
 https://data.europa.eu/eli/reg/2016/679/oj .

 

 
 Fan et al., (2018) 
 
Fan, A., Lewis, M., and Dauphin, Y. N. (2018).

 
 Hierarchical neural story generation.

 
 In Gurevych, I. and Miyao, Y., editors, Proceedings of the 56th
Annual Meeting of the Association for Computational Linguistics, ACL 2018,
Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers , pages
889–898. Association for Computational Linguistics.

 

 
 Feng et al., (2023) 
 
Feng, S., Shi, W., Bai, Y., Balachandran, V., He, T., and Tsvetkov, Y. (2023).

 
 Cook: Empowering general-purpose language models with modular and
collaborative knowledge.

 
 arXiv preprint arXiv:2305.09955 .

 

 
 Gallegos et al., (2024) 
 
Gallegos, I. O., Rossi, R. A., Barrow, J., Tanjim, M. M., Yu, T., Deilamsalehy,
H., Zhang, R., Kim, S., and Dernoncourt, F. (2024).

 
 Self-debiasing large language models: Zero-shot recognition and
reduction of stereotypes.

 
 arXiv preprint arXiv:2402.01981 .

 

 
 Gao et al., (2021) 
 
Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang,
J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. (2021).

 
 The pile: An 800gb dataset of diverse text for language modeling.

 
 CoRR , abs/2101.00027.

 

 
 Gehman et al., (2020) 
 
Gehman, S., Gururangan, S., Sap, M., Choi, Y., and Smith, N. A. (2020).

 
 Realtoxicityprompts: Evaluating neural toxic degeneration in language
models.

 
 In Findings .

 

 
 Gliwa et al., (2019) 
 
Gliwa, B., Mochol, I., Biesek, M., and Wawer, A. (2019).

 
 Samsum corpus: A human-annotated dialogue dataset for abstractive
summarization.

 
 CoRR , abs/1911.12237.

 

 
 Gordon et al., (2012) 
 
Gordon, A., Kozareva, Z., and Roemmele, M. (2012).

 
 SemEval-2012 task 7: Choice of plausible alternatives: An
evaluation of commonsense causal reasoning.

 
 In Agirre, E., Bos, J., Diab, M., Manandhar, S., Marton, Y., and
Yuret, D., editors, *SEM 2012: The First Joint Conference on Lexical
and Computational Semantics – Volume 1: Proceedings of the main conference
and the shared task, and Volume 2: Proceedings of the Sixth International
Workshop on Semantic Evaluation (SemEval 2012) , pages 394–398,
Montréal, Canada. Association for Computational Linguistics.

 

 
 Hassan et al., (2019) 
 
Hassan, F., Sánchez, D., Soria-Comas, J., and Domingo-Ferrer, J. (2019).

 
 Automatic anonymization of textual documents: detecting sensitive
information via word embeddings.

 
 In 2019 18th IEEE International Conference On Trust, Security
And Privacy In Computing And Communications/13th IEEE International
Conference On Big Data Science And Engineering (TrustCom/BigDataSE) , pages
358–365. IEEE.

 

 
 Houlsby et al., (2019) 
 
Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q.,
Gesmundo, A., Attariyan, M., and Gelly, S. (2019).

 
 Parameter-efficient transfer learning for nlp.

 
 In International Conference on Machine Learning , pages
2790–2799. PMLR.

 

 
 Hu et al., (2021) 
 
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and
Chen, W. (2021).

 
 Lora: Low-rank adaptation of large language models.

 
 arXiv preprint arXiv:2106.09685 .

 

 
 Ilharco et al., (2022) 
 
Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L.,
Hajishirzi, H., and Farhadi, A. (2022).

 
 Editing models with task arithmetic.

 
 arXiv preprint arXiv:2212.04089 .

 

 
 Jang et al., (2022) 
 
Jang, J., Yoon, D., Yang, S., Cha, S., Lee, M., Logeswaran, L., and Seo, M.
(2022).

 
 Knowledge unlearning for mitigating privacy risks in language models.

 
 arXiv preprint arXiv:2210.01504 .

 

 
 Ji et al., (2023) 
 
Ji, J., Liu, M., Dai, J., Pan, X., Zhang, C., Bian, C., Zhang, B., Sun, R.,
Wang, Y., and Yang, Y. (2023).

 
 Beavertails: Towards improved safety alignment of LLM via a
human-preference dataset.

 
 CoRR , abs/2307.04657.

 

 
 Jin et al., (2019) 
 
Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., and Lu, X. (2019).

 
 Pubmedqa: A dataset for biomedical research question answering.

 
 In Inui, K., Jiang, J., Ng, V., and Wan, X., editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language
Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019 , pages
2567–2577. Association for Computational Linguistics.

 

 
 Kadhe et al., (2023) 
 
Kadhe, S., Halimi, A., Rawat, A., and Baracaldo, N. (2023).

 
 Fairsisa: Ensemble post-processing to improve fairness of unlearning
in llms.

 
 In Socially Responsible Language Modelling Research .

 

 
 Kandpal et al., (2022) 
 
Kandpal, N., Wallace, E., and Raffel, C. (2022).

 
 Deduplicating training data mitigates privacy risks in language
models.

 

 
 Klimt and Yang, (2004) 
 
Klimt, B. and Yang, Y. (2004).

 
 Introducing the enron corpus.

 
 In CEAS 2004 - First Conference on Email and Anti-Spam, July
30-31, 2004, Mountain View, California, USA .

 

 
 Komeili et al., (2022) 
 
Komeili, M., Shuster, K., and Weston, J. (2022).

 
 Internet-augmented dialogue generation.

 
 In Muresan, S., Nakov, P., and Villavicencio, A., editors, Proceedings of the 60th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 8460–8478, Dublin, Ireland.
Association for Computational Linguistics.

 

 
 Kumar et al., (2022) 
 
Kumar, V. B., Gangadharaiah, R., and Roth, D. (2022).

 
 Privacy adhering machine un-learning in nlp.

 
 arXiv preprint arXiv:2212.09573 .

 

 
 Kurmanji et al., (2024) 
 
Kurmanji, M., Triantafillou, P., Hayes, J., and Triantafillou, E. (2024).

 
 Towards unbounded machine unlearning.

 
 Advances in Neural Information Processing Systems , 36.

 

 
 Lan et al., (2020) 
 
Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., and Soricut, R. (2020).

 
 ALBERT: A lite BERT for self-supervised learning of language
representations.

 
 In 8th International Conference on Learning Representations,
ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net.

 

 
 Levy et al., (2017) 
 
Levy, O., Seo, M., Choi, E., and Zettlemoyer, L. (2017).

 
 Zero-shot relation extraction via reading comprehension.

 
 In Levy, R. and Specia, L., editors, Proceedings of the 21st
Conference on Computational Natural Language Learning (CoNLL 2017),
Vancouver, Canada, August 3-4, 2017 , pages 333–342. Association for
Computational Linguistics.

 

 
 (51) 
 
Li, J., Cheng, X., Zhao, X., Nie, J., and Wen, J. (2023a).

 
 Halueval: A large-scale hallucination evaluation benchmark for
large language models.

 
 In Bouamor, H., Pino, J., and Bali, K., editors, Proceedings of
the 2023 Conference on Empirical Methods in Natural Language Processing,
EMNLP 2023, Singapore, December 6-10, 2023 , pages 6449–6464. Association
for Computational Linguistics.

 

 
 (52) 
 
Li, Y., Bubeck, S., Eldan, R., Del Giorno, A., Gunasekar, S., and Lee, Y. T.
(2023b).

 
 Textbooks are all you need ii: phi-1.5 technical report.

 
 arXiv preprint arXiv:2309.05463 .

 

 
 Limisiewicz and Mareček, (2022) 
 
Limisiewicz, T. and Mareček, D. (2022).

 
 Don’t forget about pronouns: Removing gender bias in language models
without losing factual gender information.

 
 arXiv preprint arXiv:2206.10744 .

 

 
 Lin et al., (2022) 
 
Lin, S., Hilton, J., and Evans, O. (2022).

 
 Truthfulqa: Measuring how models mimic human falsehoods.

 
 In Muresan, S., Nakov, P., and Villavicencio, A., editors, Proceedings of the 60th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27,
2022 , pages 3214–3252. Association for Computational Linguistics.

 

 
 Liu et al., (2023) 
 
Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., and Neubig, G. (2023).

 
 Pre-train, prompt, and predict: A systematic survey of prompting
methods in natural language processing.

 
 ACM Computing Surveys , 55(9):1–35.

 

 
 Liu et al., (2019) 
 
Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M.,
Zettlemoyer, L., and Stoyanov, V. (2019).

 
 Roberta: A robustly optimized BERT pretraining approach.

 
 CoRR , abs/1907.11692.

 

 
 Liu and Kalinli, (2023) 
 
Liu, Z. and Kalinli, O. (2023).

 
 Forgetting private textual sequences in language models via
leave-one-out ensemble.

 
 arXiv preprint arXiv:2309.16082 .

 

 
 Lu et al., (2022) 
 
Lu, X., Welleck, S., Hessel, J., Jiang, L., Qin, L., West, P., Ammanabrolu, P.,
and Choi, Y. (2022).

 
 Quark: Controllable text generation with reinforced unlearning.

 
 Advances in neural information processing systems ,
35:27591–27609.

 

 
 Maas et al., (2011) 
 
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C.
(2011).

 
 Learning word vectors for sentiment analysis.

 
 In Lin, D., Matsumoto, Y., and Mihalcea, R., editors, The 49th
Annual Meeting of the Association for Computational Linguistics: Human
Language Technologies, Proceedings of the Conference, 19-24 June, 2011,
Portland, Oregon, USA , pages 142–150. The Association for Computer
Linguistics.

 

 
 Madaan et al., (2022) 
 
Madaan, A., Tandon, N., Clark, P., and Yang, Y. (2022).

 
 Memory-assisted prompt editing to improve gpt-3 after deployment.

 
 arXiv preprint arXiv:2201.06009 .

 

 
 Maini et al., (2024) 
 
Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., and Kolter, J. Z.
(2024).

 
 Tofu: A task of fictitious unlearning for llms.

 
 arXiv preprint arXiv:2401.06121 .

 

 
 Meng et al., (2022) 
 
Meng, K., Bau, D., Andonian, A., and Belinkov, Y. (2022).

 
 Locating and editing factual knowledge in GPT.

 
 CoRR , abs/2202.05262.

 

 
 Merity et al., (2016) 
 
Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016).

 
 Pointer sentinel mixture models.

 

 
 Mikolov et al., (2013) 
 
Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013).

 
 Efficient estimation of word representations in vector space.

 
 arXiv preprint arXiv:1301.3781 .

 

 
 Min et al., (2022) 
 
Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., and
Zettlemoyer, L. (2022).

 
 Rethinking the role of demonstrations: What makes in-context learning
work?

 

 
 Mitchell et al., (2022) 
 
Mitchell, E., Lin, C., Bosselut, A., Manning, C. D., and Finn, C. (2022).

 
 Memory-based model editing at scale.

 
 In International Conference on Machine Learning , pages
15817–15831. PMLR.

 

 
 Nadeem et al., (2021) 
 
Nadeem, M., Bethke, A., and Reddy, S. (2021).

 
 StereoSet: Measuring stereotypical bias in pretrained language
models.

 
 In Zong, C., Xia, F., Li, W., and Navigli, R., editors, Proceedings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language
Processing (Volume 1: Long Papers) , pages 5356–5371, Online. Association
for Computational Linguistics.

 

 
 Nangia et al., (2020) 
 
Nangia, N., Vania, C., Bhalerao, R., and Bowman, S. R. (2020).

 
 CrowS-pairs: A challenge dataset for measuring social biases in
masked language models.

 
 In Webber, B., Cohn, T., He, Y., and Liu, Y., editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP) , pages 1953–1967, Online. Association for Computational
Linguistics.

 

 
 Nasr et al., (2023) 
 
Nasr, M., Carlini, N., Hayase, J., Jagielski, M., Cooper, A. F., Ippolito, D.,
Choquette-Choo, C. A., Wallace, E., Tramèr, F., and Lee, K. (2023).

 
 Scalable extraction of training data from (production) language
models.

 
 arXiv preprint arXiv:2311.17035 .

 

 
 Nguyen et al., (2022) 
 
Nguyen, T. T., Huynh, T. T., Nguyen, P. L., Liew, A. W.-C., Yin, H., and
Nguyen, Q. V. H. (2022).

 
 A survey of machine unlearning.

 
 arXiv preprint arXiv:2209.02299 .

 

 
 Ni et al., (2023) 
 
Ni, S., Chen, D., Li, C., Hu, X., Xu, R., and Yang, M. (2023).

 
 Forgetting before learning: Utilizing parametric arithmetic for
knowledge updating in large language models.

 
 arXiv preprint arXiv:2311.08011 .

 

 
 Nivre et al., (2020) 
 
Nivre, J., de Marneffe, M., Ginter, F., Hajic, J., Manning, C. D., Pyysalo, S.,
Schuster, S., Tyers, F. M., and Zeman, D. (2020).

 
 Universal dependencies v2: An evergrowing multilingual treebank
collection.

 
 In Calzolari, N., Béchet, F., Blache, P., Choukri, K., Cieri,
C., Declerck, T., Goggi, S., Isahara, H., Maegaard, B., Mariani, J., Mazo,
H., Moreno, A., Odijk, J., and Piperidis, S., editors, Proceedings of
The 12th Language Resources and Evaluation Conference, LREC 2020,
Marseille, France, May 11-16, 2020 , pages 4034–4043. European Language
Resources Association.

 

 
 Nostalgebraist, (2020) 
 
Nostalgebraist (2020).

 
 Interpreting GPT: The logit lens.

 
 https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens .

 

 
 Paperno et al., (2016) 
 
Paperno, D., Kruszewski, G., Lazaridou, A., Pham, N. Q., Bernardi, R.,
Pezzelle, S., Baroni, M., Boleda, G., and Fernández, R. (2016).

 
 The LAMBADA dataset: Word prediction requiring a broad discourse
context.

 
 In Erk, K. and Smith, N. A., editors, Proceedings of the 54th
Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers) , pages 1525–1534, Berlin, Germany. Association for
Computational Linguistics.

 

 
 Papernot et al., (2018) 
 
Papernot, N., Song, S., Mironov, I., Raghunathan, A., Talwar, K., and
Erlingsson, Ú. (2018).

 
 Scalable private learning with pate.

 
 arXiv preprint arXiv:1802.08908 .

 

 
 Patil et al., (2023) 
 
Patil, V., Hase, P., and Bansal, M. (2023).

 
 Can sensitive information be deleted from llms? objectives for
defending against extraction attacks.

 
 arXiv preprint arXiv:2309.17410 .

 

 
 Pawelczyk et al., (2023) 
 
Pawelczyk, M., Neel, S., and Lakkaraju, H. (2023).

 
 In-context unlearning: Language models as few shot unlearners.

 
 arXiv preprint arXiv:2310.07579 .

 

 
 Pochinkov and Schoots, (2023) 
 
Pochinkov, N. and Schoots, N. (2023).

 
 Dissecting large language models.

 
 In Socially Responsible Language Modelling Research .

 

 
 Qu et al., (2023) 
 
Qu, Y., Yuan, X., Ding, M., Ni, W., Rakotoarivelo, T., and Smith, D. (2023).

 
 Learn to unlearn: A survey on machine unlearning.

 
 arXiv preprint arXiv:2305.07512 .

 

 
 Radford et al., (2018) 
 
Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al. (2018).

 
 Improving language understanding by generative pre-training.

 

 
 Radford et al., (2019) 
 
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.
(2019).

 
 Language models are unsupervised multitask learners.

 
 OpenAI blog , 1(8):9.

 

 
 Raffel et al., (2020) 
 
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou,
Y., Li, W., and Liu, P. J. (2020).

 
 Exploring the limits of transfer learning with a unified text-to-text
transformer.

 
 The Journal of Machine Learning Research , 21(1):5485–5551.

 

 
 Rashkin et al., (2019) 
 
Rashkin, H., Smith, E. M., Li, M., and Boureau, Y.-L. (2019).

 
 Towards empathetic open-domain conversation models: A new benchmark
and dataset.

 
 In Korhonen, A., Traum, D., and Màrquez, L., editors, Proceedings of the 57th Annual Meeting of the Association for Computational
Linguistics , pages 5370–5381, Florence, Italy. Association for
Computational Linguistics.

 

 
 Ravfogel et al., (2020) 
 
Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., and Goldberg, Y. (2020).

 
 Null it out: Guarding protected attributes by iterative nullspace
projection.

 
 arXiv preprint arXiv:2004.07667 .

 

 
 Rowling, (2000) 
 
Rowling, J. K. (2000).

 
 Harry Potter and the Sorcerer’s Stone .

 
 Bloomsbury.

 

 
 Rudinger et al., (2018) 
 
Rudinger, R., Naradowsky, J., Leonard, B., and Van Durme, B. (2018).

 
 Gender bias in coreference resolution.

 
 In Walker, M., Ji, H., and Stent, A., editors, Proceedings of
the 2018 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, Volume 2 (Short
Papers) , pages 8–14, New Orleans, Louisiana. Association for Computational
Linguistics.

 

 
 Sakaguchi et al., (2020) 
 
Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. (2020).

 
 Winogrande: An adversarial winograd schema challenge at scale.

 
 In The Thirty-Fourth AAAI Conference on Artificial
Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of
Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium
on Educational Advances in Artificial Intelligence, EAAI 2020, New York,
NY, USA, February 7-12, 2020 , pages 8732–8740. AAAI Press.

 

 
 Salem et al., (2018) 
 
Salem, A., Zhang, Y., Humbert, M., Berrang, P., Fritz, M., and Backes, M.
(2018).

 
 Ml-leaks: Model and data independent membership inference attacks and
defenses on machine learning models.

 
 arXiv preprint arXiv:1806.01246 .

 

 
 Sánchez and Batet, (2016) 
 
Sánchez, D. and Batet, M. (2016).

 
 C-sanitized: A privacy model for document redaction and sanitization.

 
 Journal of the Association for Information Science and
Technology , 67(1):148–163.

 

 
 Sanh et al., (2019) 
 
Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019).

 
 Distilbert, a distilled version of bert: smaller, faster, cheaper and
lighter.

 
 arXiv preprint arXiv:1910.01108 .

 

 
 Scao et al., (2022) 
 
Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ilić, S., Hesslow, D.,
Castagné, R., Luccioni, A. S., Yvon, F., et al. (2022).

 
 Bloom: A 176b-parameter open-access multilingual language model.

 
 arXiv preprint arXiv:2211.05100 .

 

 
 Sellam et al., (2020) 
 
Sellam, T., Das, D., and Parikh, A. P. (2020).

 
 BLEURT: learning robust metrics for text generation.

 
 In Jurafsky, D., Chai, J., Schluter, N., and Tetreault, J. R.,
editors, Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics, ACL 2020, Online, July 5-10, 2020 , pages
7881–7892. Association for Computational Linguistics.

 

 
 Shaik et al., (2023) 
 
Shaik, T., Tao, X., Xie, H., Li, L., Zhu, X., and Li, Q. (2023).

 
 Exploring the landscape of machine unlearning: A survey and taxonomy.

 
 arXiv preprint arXiv:2305.06360 .

 

 
 Shi et al., (2023) 
 
Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D., and
Zettlemoyer, L. (2023).

 
 Detecting pretraining data from large language models.

 
 arXiv preprint arXiv:2310.16789 .

 

 
 Shokri et al., (2017) 
 
Shokri, R., Stronati, M., Song, C., and Shmatikov, V. (2017).

 
 Membership inference attacks against machine learning models.

 
 In 2017 IEEE symposium on security and privacy (SP) , pages
3–18. IEEE.

 

 
 Silveira et al., (2014) 
 
Silveira, N., Dozat, T., de Marneffe, M.-C., Bowman, S., Connor, M., Bauer, J.,
and Manning, C. (2014).

 
 A gold standard dependency corpus for English.

 
 In Calzolari, N., Choukri, K., Declerck, T., Loftsson, H., Maegaard,
B., Mariani, J., Moreno, A., Odijk, J., and Piperidis, S., editors, Proceedings of the Ninth International Conference on Language Resources and
Evaluation (LREC’14) , pages 2897–2904, Reykjavik, Iceland. European
Language Resources Association (ELRA).

 

 
 Smith et al., (2020) 
 
Smith, E. M., Williamson, M., Shuster, K., Weston, J., and Boureau, Y.-L.
(2020).

 
 Can you put it all together: Evaluating conversational agents’
ability to blend skills.

 
 In Jurafsky, D., Chai, J., Schluter, N., and Tetreault, J., editors,
 Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics , pages 2021–2030, Online. Association for
Computational Linguistics.

 

 
 Smith et al., (2023) 
 
Smith, V., Shamsabadi, A. S., Ashurst, C., and Weller, A. (2023).

 
 Identifying and mitigating privacy risks stemming from language
models: A survey.

 
 arXiv preprint arXiv:2310.01424 .

 

 
 Socher et al., (2013) 
 
Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., and
Potts, C. (2013).

 
 Recursive deep models for semantic compositionality over a sentiment
treebank.

 
 In Yarowsky, D., Baldwin, T., Korhonen, A., Livescu, K., and Bethard,
S., editors, Proceedings of the 2013 Conference on Empirical Methods in
Natural Language Processing , pages 1631–1642, Seattle, Washington, USA.
Association for Computational Linguistics.

 

 
 Stanovsky et al., (2019) 
 
Stanovsky, G., Smith, N. A., and Zettlemoyer, L. (2019).

 
 Evaluating gender bias in machine translation.

 
 In Korhonen, A., Traum, D., and Màrquez, L., editors, Proceedings of the 57th Annual Meeting of the Association for Computational
Linguistics , pages 1679–1684, Florence, Italy. Association for
Computational Linguistics.

 

 
 Sundararajan et al., (2017) 
 
Sundararajan, M., Taly, A., and Yan, Q. (2017).

 
 Axiomatic attribution for deep networks.

 
 In International conference on machine learning , pages
3319–3328. PMLR.

 

 
 Sutskever et al., (2014) 
 
Sutskever, I., Vinyals, O., and Le, Q. V. (2014).

 
 Sequence to sequence learning with neural networks.

 
 Advances in neural information processing systems , 27.

 

 
 Taylor et al., (2022) 
 
Taylor, R., Kardas, M., Cucurull, G., Scialom, T., Hartshorn, A., Saravia, E.,
Poulton, A., Kerkez, V., and Stojnic, R. (2022).

 
 Galactica: A large language model for science.

 
 arXiv preprint arXiv:2211.09085 .

 

 
 Tirumala et al., (2022) 
 
Tirumala, K., Markosyan, A., Zettlemoyer, L., and Aghajanyan, A. (2022).

 
 Memorization without overfitting: Analyzing the training dynamics of
large language models.

 
 Advances in Neural Information Processing Systems ,
35:38274–38290.

 

 
 (105) 
 
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix,
T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin,
A., Grave, E., and Lample, G. (2023a).

 
 Llama: Open and efficient foundation language models.

 

 
 (106) 
 
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y.,
Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L.,
Ferrer, C. C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu,
W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S.,
Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev,
A., Koura, P. S., Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y.,
Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y.,
Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva,
R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R.,
Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A.,
Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and
Scialom, T. (2023b).

 
 Llama 2: Open foundation and fine-tuned chat models.

 

 
 Tuggener et al., (2020) 
 
Tuggener, D., von Däniken, P., Peetz, T., and Cieliebak, M. (2020).

 
 LEDGAR: A large-scale multi-label corpus for text classification
of legal provisions in contracts.

 
 In Calzolari, N., Béchet, F., Blache, P., Choukri, K., Cieri,
C., Declerck, T., Goggi, S., Isahara, H., Maegaard, B., Mariani, J., Mazo,
H., Moreno, A., Odijk, J., and Piperidis, S., editors, Proceedings of
The 12th Language Resources and Evaluation Conference, LREC 2020,
Marseille, France, May 11-16, 2020 , pages 1235–1241. European Language
Resources Association.

 

 
 Tunstall et al., (2022) 
 
Tunstall, L., Von Werra, L., and Wolf, T. (2022).

 
 Natural language processing with transformers .

 
 ” O’Reilly Media, Inc.”.

 

 
 United Nations, (1948) 
 
United Nations (1948).

 
 Universal Declaration of Human Rights .

 

 
 Üstün et al., (2024) 
 
Üstün, A., Aryabumi, V., Yong, Z.-X., Ko, W.-Y., D’souza, D., Onilude,
G., Bhandari, N., Singh, S., Ooi, H.-L., Kayid, A., et al. (2024).

 
 Aya model: An instruction finetuned open-access multilingual language
model.

 
 arXiv preprint arXiv:2402.07827 .

 

 
 Vaswani et al., (2017) 
 
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., and Polosukhin, I. (2017).

 
 Attention is all you need.

 
 Advances in neural information processing systems , 30.

 

 
 Wang et al., (2023) 
 
Wang, L., Chen, T., Yuan, W., Zeng, X., Wong, K.-F., and Yin, H. (2023).

 
 Kga: A general machine unlearning framework based on knowledge gap
alignment.

 
 arXiv preprint arXiv:2305.06535 .

 

 
 Wang et al., (2024) 
 
Wang, L., Zeng, X., Guo, J., Wong, K.-F., and Gottlob, G. (2024).

 
 Selective forgetting: Advancing machine unlearning techniques and
evaluation in language models.

 
 arXiv preprint arXiv:2402.05813 .

 

 
 Webster et al., (2018) 
 
Webster, K., Recasens, M., Axelrod, V., and Baldridge, J. (2018).

 
 Mind the GAP: A balanced corpus of gendered ambiguous pronouns.

 
 Transactions of the Association for Computational Linguistics ,
6:605–617.

 

 
 Wei et al., (2023) 
 
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le,
Q., and Zhou, D. (2023).

 
 Chain-of-thought prompting elicits reasoning in large language
models.

 

 
 Wu et al., (2023) 
 
Wu, X., Li, J., Xu, M., Dong, W., Wu, S., Bian, C., and Xiong, D. (2023).

 
 Depn: Detecting and editing privacy neurons in pretrained language
models.

 
 arXiv preprint arXiv:2310.20138 .

 

 
 Xiong et al., (2020) 
 
Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan,
Y., Wang, L., and Liu, T. (2020).

 
 On layer normalization in the transformer architecture.

 
 In International Conference on Machine Learning , pages
10524–10533. PMLR.

 

 
 Xu et al., (2023) 
 
Xu, H., Zhu, T., Zhang, L., Zhou, W., and Yu, P. S. (2023).

 
 Machine unlearning: A survey.

 
 ACM Computing Surveys , 56(1):1–36.

 

 
 Yao et al., (2023) 
 
Yao, Y., Xu, X., and Liu, Y. (2023).

 
 Large language model unlearning.

 
 arXiv preprint arXiv:2310.10683 .

 

 
 Yeom et al., (2018) 
 
Yeom, S., Giacomelli, I., Fredrikson, M., and Jha, S. (2018).

 
 Privacy risk in machine learning: Analyzing the connection to
overfitting.

 
 In 2018 IEEE 31st computer security foundations symposium
(CSF) , pages 268–282. IEEE.

 

 
 Yoon et al., (2023) 
 
Yoon, D., Jang, J., Kim, S., and Seo, M. (2023).

 
 Gradient ascent post-training enhances language model generalization.

 
 arXiv preprint arXiv:2306.07052 .

 

 
 (122) 
 
Yu, C., Jeoung, S., Kasi, A., Yu, P., and Ji, H. (2023a).

 
 Unlearning bias in language models by partitioning gradients.

 
 In Findings of the Association for Computational Linguistics:
ACL 2023 , pages 6032–6048.

 

 
 (123) 
 
Yu, Q., Merullo, J., and Pavlick, E. (2023b).

 
 Characterizing mechanisms for factual recall in language models.

 
 arXiv preprint arXiv:2310.15910 .

 

 
 Zellers et al., (2019) 
 
Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. (2019).

 
 Hellaswag: Can a machine really finish your sentence?

 
 CoRR , abs/1905.07830.

 

 
 Zhang et al., (2023) 
 
Zhang, D., Finckenberg-Broman, P., Hoang, T., Pan, S., Xing, Z., Staples, M.,
and Xu, X. (2023).

 
 Right to be forgotten in the era of large language models:
Implications, challenges, and solutions.

 
 arXiv preprint arXiv:2307.03941 .

 

 
 Zhang et al., (2024) 
 
Zhang, N., Yao, Y., Tian, B., Wang, P., Deng, S., Wang, M., Xi, Z., Mao, S.,
Zhang, J., Ni, Y., et al. (2024).

 
 A comprehensive study of knowledge editing for large language models.

 
 arXiv preprint arXiv:2401.01286 .

 

 
 Zhang et al., (2018) 
 
Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., and Weston, J. (2018).

 
 Personalizing dialogue agents: I have a dog, do you have pets too?

 
 In Gurevych, I. and Miyao, Y., editors, Proceedings of the 56th
Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers) , pages 2204–2213, Melbourne, Australia. Association for
Computational Linguistics.

 

 
 Zhang et al., (2022) 
 
Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C.,
Diab, M., Li, X., Lin, X. V., Mihaylov, T., Ott, M., Shleifer, S., Shuster,
K., Simig, D., Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer, L.
(2022).

 
 Opt: Open pre-trained transformer language models.

 

 
 Zhang et al., (2015) 
 
Zhang, X., Zhao, J. J., and LeCun, Y. (2015).

 
 Character-level convolutional networks for text classification.

 
 In Cortes, C., Lawrence, N. D., Lee, D. D., Sugiyama, M., and
Garnett, R., editors, Advances in Neural Information Processing Systems
28: Annual Conference on Neural Information Processing Systems 2015, December
7-12, 2015, Montreal, Quebec, Canada , pages 649–657.

 

 
 Zhao et al., (2018) 
 
Zhao, J., Wang, T., Yatskar, M., Ordonez, V., and Chang, K.-W. (2018).

 
 Gender bias in coreference resolution: Evaluation and debiasing
methods.

 
 In Walker, M., Ji, H., and Stent, A., editors, Proceedings of
the 2018 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, Volume 2 (Short
Papers) , pages 15–20, New Orleans, Louisiana. Association for Computational
Linguistics.

 

 
 Zou et al., (2023) 
 
Zou, A., Wang, Z., Kolter, J. Z., and Fredrikson, M. (2023).

 
 Universal and transferable adversarial attacks on aligned language
models.

 
 arXiv preprint arXiv:2307.15043 .