Efficient Prompting Methods for Large Language Models: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.01077v2 [cs.CL] 02 Dec 2024 
 
 

# Efficient Prompting Methods for Large Language Models: A Survey

 CCS:  General and reference Surveys and overviews CCS:  Computing methodologies Natural language processing 
 
 
 Kaiyan Chang
 
 email: changkaiyan_neu@outlook.com 
 
 Affiliation:  Northeastern University , NO. 195, Chuangxin Road, Hunnan District , Shenyang , Liaoning , China , 110819 
 
 , 
 Songcheng Xu
 
 email: winsome.xsc@gmail.com 
 
 Affiliation:  Northeastern University , NO. 195, Chuangxin Road, Hunnan District , Shenyang , Liaoning , China , 110819 
 
 , 
 Chenglong Wang
 
 Affiliation:  Northeastern University , NO. 195, Chuangxin Road, Hunnan District , Shenyang , Liaoning , China , 110819 
 
 , 
 Yingfeng Luo
 
 Affiliation:  Northeastern University , NO. 195, Chuangxin Road, Hunnan District , Shenyang , Liaoning , China , 110819 
 
 , 
 Xiaoqian Liu
 
 Affiliation:  Northeastern University , NO. 195, Chuangxin Road, Hunnan District , Shenyang , Liaoning , China , 110819 
 
 , 
 Tong Xiao
 
 Note:  Corresponding Author.
 
 email: xiaotong@mail.neu.edu.cn 
 
 Affiliation:  Northeastern University NiuTrans Research , NO. 195, Chuangxin Road, Hunnan District , Shenyang , Liaoning , China , 110819 
 
 and 
 Jingbo Zhu
 
 email: zhujingbo@mail.neu.edu.cn 
 
 Affiliation:  Northeastern University NiuTrans Research , NO. 195, Chuangxin Road, Hunnan District , Shenyang , Liaoning , China , 110819 
 

 Abstract. 
 
 Prompting is a mainstream paradigm for adapting large language models to specific natural language processing tasks without modifying internal parameters.
Therefore, detailed supplementary knowledge needs to be integrated into external prompts, which inevitably brings extra human efforts and computational burdens for practical applications.
As an effective solution to mitigate resource consumption, Efficient Prompting Methods have attracted a wide range of attention.
We provide mathematical expressions at a high level to deeply discuss Automatic Prompt Engineering for different prompt components and Prompt Compression in continuous and discrete spaces.
Finally, we highlight promising future directions to inspire researchers interested in this field.

 
 
 
 Keywords:  Large Language Model, Efficient Prompting Method, Automatic Prompt Engineering, Prompt Compression
 
 

## 1. Introduction

 
 {forest} 
 
 Figure 1. Taxonomy of efficient prompting methods. 
 
 
 As hundreds of billions of breakthroughs on the parameter scale, Large Language Models (LLMs) acquire emergent abilities  ( Wei et al., 2022a ) , especially in-context learning ability  ( Brown et al., 2020 ) that promote rapid advancement in prompting techniques.
Prompting stands out as a lightweight promising solution for controlling LLMs without tuning parameters, having received widespread attention within the Natural Language Processing (NLP) community.
There are two types of prompts, where the hard prompt is discrete natural language descriptions and the soft prompt is continuous vector representations.
In particular, the hard prompt has become a crucial bridge for human-machine interaction relying on its improved interpretability, controllability and flexibility compared to the soft prompt.
Therefore, we mainly focus on the hard prompt that covers all the components of LLM input scaling from concise instructions to long context with demonstrations (Chain-of-Thought (CoT)  ( Wei et al., 2022b ) , role-playing system prompts, etc. ).

 
 
 At present, it is common to unlock the potential of LLMs in specific domains by prompting. For example, the CoT series of studies  ( Yao et al., 2023a ; Besta et al., 2024 ; Chen et al., 2022 ) have progressively enhanced LLM reasoning capability by thinking aloud. Furthermore, OpenAI recently introduced the reasoning LLM o1  ( OpenAI, 2024a ; OpenAI, 2024b ) trained with reinforcement learning to break down more difficult problems and produce a long internal CoT before responding. The excellent performance of these methods promotes increasing research on optimized prompting methods, which has gradually formed a brand new area in the NLP landscape. Meanwhile, several challenges related to application efficiency come one after another: more and more complex prompt design makes manual prompt optimization time-consuming and labor-intensive; more and more detailed prompt content inevitably consumes significant computational resources when applied to large-scale models. Such prohibitive overheads have become a major barrier to the practical deployment of LLMs, so we define “Efficient Prompting Methods” as prompting language models to achieve comparable or even better performance with fewer human or computational resources in this paper.

 
 
 We narrow this survey to efficient prompting methods in the era of LLMs. To the best of our knowledge, this is the first survey to summarize LLM prompting methods from the point of “Efficient”. It is remarkable that we model the core concepts of each category of resource-saving methods from a mathematical perspective in § 2.3 . Following this, we propose a novel taxonomy as shown in Fig. 1 to comprehensively review efficient prompting methods based on their consistent optimization strategies. To avoid human resources, we introduce automatic prompt engineering efforts based on LLM empowerment in § 3 , including iterative design and optimization of different prompt components. To save computational resources, we organize prompt compression efforts into two categories based on prompt types in § 4 : Text-to-Vector (T2V) compression in continuous space and Text-to-Text (T2T) compression in discrete space. Additionally, we provide sufficient schematic diagrams depicting the basic pipeline of each category of methods, as well as tables representing differences in the details of the same category of methods. There is also a list of open-source projects in Appendix A.1 as a quick access for NLP practitioners efficiently prompting LLMs in both scientific research and commercial deployment. Finally, we analyze the challenges of existing methods and discuss promising future directions in § 5 . We hope this survey can provide a clear picture of the efficient prompting topic and contribute to convenient human-machine interaction in the progress of Artificial General Intelligence (AGI).

 
 
 

## 2. Preliminary

 

### 2.1. Origin of Prompting

 
 There have been two significant paradigm shifts from supervised learning to fine-tuning to prompting in the field of NLP, where the “paradigm” refers to the way to train a language model (LM) and apply it to specific downstream tasks. Currently, prompting has become the dominant paradigm for LLMs. In the following, we elicit the origin of prompting in the context of the development of language models.

 
 
 With the rise of Pre-trained Language Models (PLMs) based on the Transformer  ( Vaswani et al., 2017 ) architecture, the first major shift transformed from a one-stage (fully supervised learning) to a two-stage process (pre-train + fine-tune). We have embraced the era of PLM that avoids the computational cost of training from scratch. The first stage is pre-training an LM with a wealth of unlabeled data, and the second stage is fine-tuning the PLM with a small amount of labeled data for specific tasks.

 
 
 Notable generative PLM GPT-3  ( Brown et al., 2020 ) with 175B parameters has presented impressive in-context learning ability to generate expected responses from few-shot demonstrations, which triggered the second major shift (pre-train, prompt and predict). It is feasible for prompting to guide a single generalist model to perform multiple tasks through appropriate prompts rather than updating parameters, saving substantial computational costs required to fine-tune separate models for each task. As a result, the parameter scale of language models continues to expand to a ten-billion level, opening the era of LLM that have set a solid stage for further development of prompting. In conclusion, prompting has dominated the interaction way between humans and AI systems, with great potential to open the door to AGI.

 
 
 

### 2.2. Prompt Type

 
 Prompt as the model input to describe human intention can be various modalities such as text, images, audio, etc. , its quality directly determines the accuracy of specific tasks. This paper only discusses the prompt in text form in the field of NLP, referred to as “hard prompt”. While the text is processed into vectors by neural networks, the prompt in vector form is also known as “soft prompt”. Considering the close relationship between prompt type and model architecture, the following introduction will be presented from the perspectives of PLM and LLM eras, respectively.

 
 

#### 2.2.1. Hard Prompt

 
 Table 1. Three shapes of the hard prompt, where the green represents hard prompts and the highlighted represents CoT. (a) Cloze prompt is usually a prompt template with an empty space to be filled. (b) Prefix prompt uses concise instructions and input-output examples to activate the memory of LLMs during pre-training. (c) Detailed description contains task-specific instruction, context, input, and output indicators, which makes better use of LLM emergent abilities in various downstream tasks. 
 
 
 
 
 Prompt Shape 
 Task 
 Illustration 
 Instantiation 
 
 a. Cloze 
 NLU 
 
 
 
 How was the movie? [MASK] , I was surprised. 
 
 b. Prefix 
 NLG 
 
 
 
 
 
 
 Translate English to French: 
 
 sea otter = loutre de mer 
 
 cheese = 
 
 
 
 
 
 c. Detailed 
 
 description 
 
 NLP 
 
 
 
 
 
 
 Assuming you are a mathematician, help me solve the 
 
 following math problem. 
 
 Question: John writes 20 pages a day. How long will it 
 
 take him to write 3 books that are 400 pages each? 
 
 Answer: He wants to write 3*400= 3*400=1200 1200 pages. 
 
 So it will take him 1200/20= 1200/20=60 60 days. #### 60. 
 
 Question: Joy can read 8 pages of a book in 20 minutes. 
 
 How many hours will it take her to read 120 pages? 
 
 Answer: 
 
 

 
 
 
 
 • 
 
 The PLM Era .
Transformer-based PLMs can be categorized into three types according to their architecture: Encoder-only, Decoder-only, and Encoder-Decoder. The input formats suitable for each model largely depend on the choice of pre-training tasks, so different prompt shapes are designed for specific architectures. For example, Encoder-only models  ( Devlin et al., 2019 ; Liu et al., 2019 ; Lan et al., 2020 ) pre-trained with Masked Language Modeling (MLM) task are good at predicting the [MASK] token in the prompt template, as shown in Table 1 -a. While Decoder-only models  ( Radford and Narasimhan, 2018 ; Radford et al., 2019 ; Brown et al., 2020 ) pre-trained with auto-regressive language modeling task do well in generating the next token after the prefix prompt from left to right, as depicted in Table 1 -b.

 

 • 
 
 The LLM Era .
At present, most LLMs adopt a Decoder-only architecture that is fine-tuned with instructions and aligned with human preferences, so they can generate desired responses instructed by natural language prompts. Furthermore, as scaling up LLMs perform excellently across a variety of general tasks, prompts are designed to be more detailed and comprehensive. The basic components of a prompt can be summarized into four parts, as illustrated in Table 1 -c.

 

 
 
 
 

#### 2.2.2. Soft Prompt

 
 Table 2. Three shapes of the soft prompt, where circles represent continuous vectors and squares represent discrete tokens, the orange represents soft prompts, the gray represents frozen parameters and the green represents hard prompts. (a) Embeddings refer to trainable parameters added to the prompt template. (b) Prefixes are prepended to each layer of the neural network to indicate the downstream task types. (c) Compact vectors are hard prompts compressed by open-source LLMs. 
 
 
 
 
 
 
 
 Prompt 
 
 Shape 
 
 a. Embedding 
 b. Prefix 
 c. Compact vector 
 
 Task 
 NLU 
 NLG 
 NLP 
 
 Illustration 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 • 
 
 The PLM Era .
Early work has investigated parameter-efficient fine-tuning using soft prompts, including prompt tuning  ( Lester et al., 2021 ) , prefix tuning  ( Li and Liang, 2021 ) , etc. As can be seen in Table 2 -a b, these methods freeze original parameters and fine-tune only a small number of additional soft prompts to achieve comparable performance to full-parameter fine-tuning. This phenomenon becomes more evident as the model scale increases. However, with the advent of the LLM era, many LLMs are closed-source with inaccessible parameters, leading to a relative stagnation in follow-up research on soft prompts.

 

 • 
 
 The LLM Era .
In this survey, we refer to the vector representations of hard prompts inside language models as soft prompts as well. The reason soft prompts are still critical is that they can be trained together with the language model to achieve effective model adjustments. Soft prompts in the LLM era as shown in Table 2 -c are typically more compact than hard prompts, storing valid information from hard prompts with less space to accelerate inference.

 

 
 
 
 
 

### 2.3. Mathematical Modeling

 
 To facilitate the broad advantages of LLMs across various research and applications, we review efficient prompting methods from the point of reducing resource consumption in real-world applications. We divide the primary sources into two main categories: human resources and computational resources. Our analysis reveals that although there are many commonalities in the optimization objectives within each category, there is a lack of connections between them. To bridge this gap, we attempt to model the optimization process of each category of efficient prompting methods into mathematical formulas and shed light on how they serve as the theoretical foundation for this survey. In this way, not only can beginners clarify the essence of efficient prompting methods at a glance, but researchers also can be inspired to rationally integrate the two strategies into win-win solutions that simultaneously reduce resources in both areas.

 
 
 Automatic Prompt Engineering (§ 3 ): The optimization object is mainly oriented to the instruction p i ​ n ​ s p_{ins} and demonstration p d ​ e ​ m p_{dem} of the prompt. The optimization objective is to search for the optimal natural language prompt p ∗ p^{*} that maximizes the performance of the target model M in the discrete prompt space P Hard P_{\text{Hard}} :

 
 
 

 
 (1) | 
 | 
 p ∗ = arg ​ max p ∈ P Hard ⁡ 𝒫 ​ [ f ⁡ ( p ins , p dem , M ) ] p^{*}=\argmax_{p\in P_{\text{Hard}}}\mathcal{P}[f(p_{ins},p_{dem};\text{M})] | 
 | 
 

 
 
 where f ⁡ ( ⋅ , M ) f(\cdot;\text{M}) denotes the output of the target model and 𝒫 \mathcal{P} denotes the performance measured by an evaluation metric comparing the output with the ground truth label, e.g., accuracy in NLU or math reasoning tasks.

 
 
 Prompt Compression (§ 4 ): There are two prompt compression methods performed in different optimization spaces. Text-to-Vector (T2V) compression in continuous space compresses long text p o h p^{h}_{o} (original hard prompt) into short vectors p c s p^{s}_{c} (compressed soft prompt) through fine-tuning the LLM. Text-to-Text (T2T) compression in discrete space compresses long text p o h p^{h}_{o} into short text p c h p^{h}_{c} (compressed hard prompt) by filtering redundant information, or summarizing until information is sufficient. The informativeness I ⁡ ( p ) I(p) (such as self-information  ( Li et al., 2023a ) , perplexity  ( Jiang et al., 2023a ; Jiang et al., 2023b ) , etc. ) can be measured by a Small Language Model (SLM) or an LLM. The optimization objective is to minimize the difference in model output distributions before and after the prompt compression:

 
 
 

 
 (2) | 
 | 
 p c s ∗ = arg ​ min p s ∼ { θ M , θ Soft } 𝒟 [ q ( y M | p o h ) ∥ q ( y M | p c s ) ] p^{s*}_{c}=\argmin_{p^{s}\sim\{\theta_{\text{M}},\;\theta_{\text{Soft}}\}}\mathcal{D}[q(y_{\text{M}}\;|\;p^{h}_{o})\parallel q(y_{\text{M}}\;|\;p^{s}_{c})] | 
 | 
 

 
 
 

 
 (3) | 
 | 
 p c h ∗ = arg ​ min I ⁡ ( p h ) ≥ λ 𝒟 [ q ( y M | p o h ) ∥ q ( y M | p c h ) ] p^{h*}_{c}=\argmin_{I(p^{h})\geq\lambda}\mathcal{D}[q(y_{\text{M}}\;|\;p^{h}_{o})\parallel q(y_{\text{M}}\;|\;p^{h}_{c})] | 
 | 
 

 
 
 where θ \theta denotes trainable parameters, λ \lambda denotes minimum threshold of informativeness, q ( y M | ⋅ ) q(y_{\text{M}}\;|\;\cdot) denotes the output distribution of the target model M and 𝒟 \mathcal{D} denotes the distance metric between the output distributions, e.g., Kullback-Leibler divergence.

 
 
 Figure 2. An overview of efficient prompting methods. 
 
 
 Based on the above theoretical foundation, we provide a detailed presentation of the representative efficient prompting methods from two categories in the following, which will further enhance the understanding of these mathematical formulas. Figure 2 shows an overview of this survey.
The left side expresses that automatic prompt engineering (§ 3 ) takes advantage of LLMs’ powerful generative capabilities through meta-prompts to iteratively exploit high-quality instructions (§ 3.1 ) and optimize CoT prompting frameworks (§ 3.2 ) for better performance, which substitutes repetitive labor of human prompt engineers.
The right side discusses prompt compression (§ 4 ) in both continuous and discrete spaces to reduce prompt length for more input space and conserve computational resources for inference acceleration. T2V compression (§ 4.1 ) utilizes vectors to cache key information of hard prompts by training open-source LLMs. T2T compression (§ 4.2 ) preserves only valid information from hard prompts without performance loss.

 
 
 
 

## 3. Automatic Prompt Engineering

 
 Figure 3. The basic pipeline of automatic prompt engineering. Step 1: The discrete prompt space is expanded according to the customized optimization direction. Step 2: Candidate prompts are reasonably evaluated based on target model performance. Step 3: The optimal prompts are selected from the prompt pool using appropriate sampling strategies. 
 
 
 Prompt engineering refers to designing effective prompts to make better use of specific language models in complex scenarios.

 
 
 Challenges .
It is intuitive for prompt engineering to manually design elaborate prompts that make language models fully understand human intentions and mimic human behaviors. However, due to the limited comprehension competency of early language models, prompts were usually concise instructions and left little room for manual improvement. Later on, since the revolutionary breakthrough in the general capabilities of scaling language models, humans began to frequently use more detailed natural language prompts to interact with LLMs. A large number of experiments have found that the upper boundary of prompting LLMs becomes ambiguous, so human expertise cannot quickly and accurately navigate novel tasks. We summarize several challenges of manual-designed prompts for LLMs: (1) Sensitivity : Especially in zero-shot settings, subtle differences in prompt content can lead to significant differences in model performance  ( Jiang et al., 2020 ; Zhao et al., 2021 ; Sclar et al., 2023 ) , so customized prompts are needed for different models and tasks. (2) Suboptimality : Although well-crafted prompts effectively improve task accuracy, the internal compatibility of LLMs with particular prompts remains uncertain. The missing optimization direction means that human engineers can only rely on trial and error to explore a wider prompt space in search of relatively optimal solutions. (3) Discrepancy : LLMs may interpret natural language differently from humans  ( Deng et al., 2022 ) , sometimes gibberish prompts inconsistent in grammar and syntax may be more effective, which is obviously beyond the scope of human-designed prompts. To conclude, designing natural language prompts is an art that demands a lot of time and experience in the LLM era, which inevitably over-consumes human resources.

 
 
 Solutions .
LLMs may exactly know what kinds of prompts they want, so a wide range of LLM-driven prompt optimization methods have been proposed to mitigate human biases and adaptively search out high-quality prompts to maximize LLM performance by Eq. 1 . This line of work is collectively referred to as automatic prompt engineering, which essentially mimics search algorithms in discrete space consisting of three iterative steps, as shown in Fig. 3 .

 
 
 According to the different components of prompts, automatic prompt engineering is divided into two parts: Instruction Design in § 3.1 and CoT Optimization in § 3.2 . Both approaches primarily employ search algorithm principles to identify optimal solutions, with meta-prompts playing a crucial role in guiding LLMs (refer to as optimizers) to perform generation, evaluation, and selection instead of human engineers, effectively automating prompt design in a simple and efficient way. The difference between them is that the former focuses on expanding the instruction space, while the latter is more inclined to construct comprehensive problem-solving frameworks.

 
 

### 3.1. Instruction Design

 
 Instructions as the most direct way of human-computer interaction are the first choice for in-context learning. Typically, instructions are concise and simple task descriptions that guide LLMs in performing specific tasks, similar to activating certain knowledge in a specific domain. As the core component of prompts, instruction design has been the most prevalent subject of numerous discussions in the field of automatic prompt engineering. The premise for imitating search algorithms is to create a sufficient search space, which is perhaps a big challenge for human engineers. However, it is a piece of cake for LLMs with extensive language knowledge accumulated from pre-training and excellent generative capability mastered during supervised fine-tuning, leading to creativity superior to that of humans. In the following, we will organize existing works from three perspectives.

 
 

#### 3.1.1. Sampling-based methods

 
 Self-instruct  ( Wang et al., 2023b ) has indicated that LLMs can generate applicable instructions from a small set of seed human-written instructions automatically. To fully leverage the generative capability of LLMs in constructing a diverse prompt space, the most straightforward approach is iteratively sampling during inference. But how does the LLM generate instructions? It can either match the most suitable instructions based on input-output pairs in the prompt or imitate a well-crafted prompt. Subsequently, the meta-prompt steers the LLM to perform each step in automatic prompt engineering.

 
 
 APE (Automatic Prompt Engineer)  ( Zhou et al., 2023b ) is a pioneering work that treats the human-intractable question as a black-box optimization process guided by LLMs. ALL prompting optimization processes can be executed by ONE human-aligned instruction-generation LLM. The LLM respectively acts as different prompt engineers to optimize the instruction in the prompt. Step 1: The inference LLM generates instructions according to demonstrations (input-output pairs). Step 2: The scoring LLM evaluates these candidate instructions by predicting the probability of the next token and selects high-quality instructions whose scores exceed a certain threshold. Step 3 (Optional): The resampling LLM performs Monte Carlo search for semantically similar instructions around the best candidates. Such an optimization pipeline is equivalent to compressing the demonstrations into a single instruction, which greatly outperforms the prior LLM baseline by a large margin in zero/few-shot learning and zero-shot CoT settings. Moreover, APE achieves better or comparable performance to the instructions generated by human annotators on 24/24 Instruction Induction tasks and 17/21 curated Big-Bench  ( Suzgun et al., 2023 ) tasks.
However, prompt optimization methods face challenges of significant performance gaps under data distribution shifts. GPO framework  ( Li et al., 2023c ) first utilizes a meta-prompt to ask the LLM to generate multiple corresponding outputs for a single unlabeled input. Then, the prompt ensemble labeling strategy selects the output with the highest consistency as the label. Finally, demonstrations among different distributions are mixed to run APE for joint prompt optimization and obtain the final optimized prompt with high robustness and generalizability.

 
 
 Based on the exploration and exploitation concept, OPRO  ( Yang et al., 2023 ) uses meta-prompt to instruct the LLM respectively as an optimizer to exploit plenty of instructions and a scorer to evaluate their scores. Then, instructions paired with their scores iteratively form an optimization trajectory to guide the optimizer to explore new instructions for higher task accuracy.
Due to the lack of human trial-and-error processes and deeper domain expertise in the aforementioned methods, PromptAgent  ( Wang et al., 2023c ) automatically design expert prompts rich in domain-specific details and structured guidance through Monte Carlo Tree Search (MCTS) based on the self-reflection ability of LLMs. It systematically augments the prompt space in a tree structure and opts for the best node in the optimal path with the highest reward.
Inspired by Mixture of Experts (MoE), Mixture-of-Prompt (MoP)  ( Wang et al., 2024a ) expands the instruction optimization space by clustering demonstrations for different experts and then identifies the best instruction for each cluster by a Region-Based Joint Search (RBJS) algorithm.

 
 
 GPS  ( Xu et al., 2022 ) has verified the feasibility of automatically searching for high-performing prompts based on the Evolutionary Algorithm (EA). EA is a highly robust and widely applicable cluster of optimization algorithms that simulate the principle “survival of the fittest” of natural selection, including Genetic Algorithm (GA), Differential Evolution (DE), etc. In the field of prompt optimization, some efforts adhere to the EA concept to enhance the diversity of the discrete prompt space. The LLM serves as the evolutionary operator to perform four critical operations guided by meta-prompts. Step 1: Initializing the prompt population with appropriate seeds. Step 2: Evolving initial prompts into candidate prompts by selection, mutation, and crossover. Step 3: Evaluating candidates based on a proper metric. Step 4: Updating the population with the fittest samples. Population initialization may vary from different methods while the iterative evolution process is basically the same.

 
 
 EvoPrompting  ( Chen et al., 2023a ) introduces LLMs to be evolutionary operators for the first time, which avoids resource wastage and human bias of manual-designed discrete search space. The code-pretrained LLM is used to perform the end-to-end meta-learning algorithm for Neural Architecture Search (NAS) task. Parent samples are initialized by a handful of well-designed program seeds to warm-start the iteration. EvoPrompting outperforms both manual-design and naive few-shot prompting methods in terms of model performance and computational efficiency.
EvoPrompt  ( Guo et al., 2023 ) is the first attempt to synergize LLM and EA algorithms through natural language descriptions, which achieves an exploration and exploitation trade-off in discrete prompt optimization. The initial population includes both manual prompts to leverage human prior knowledge and diverse generated prompts selected by the roulette wheel method to avoid local optimum. Experiments suggest choosing GA when several high-quality prompts already exist, otherwise DE.
Promptbreeder  ( Fernando et al., 2023 ) is a general-purpose self-referential self-improvement mechanism to evolve task-prompts (origin-prompts) and mutation-prompts (meta-prompts) simultaneously for a given domain.
AELP  ( Hsieh et al., 2023b ) optimizes long prompts by a simple greedy algorithm with beam search and utilizes search history to enhance the effectiveness of LLM-based mutation.
 Cui et al. (2024) introduce a multi-phase exploration-exploitation strategy PhaseEvo to achieve joint optimization of instructions and demonstrations. Firstly, both available input-output pairs and human-designed prompts are used to explore a vast joint initialization population. Then, an LLM Improver generates candidates based on the feedback offered by an LLM Examiner. Notably, global and local LLM-based mutations are applied for evolution diversity and rapid convergence in succession.

 
 
 

#### 3.1.2. Feedback-based methods

 
 Accurate feedback is a key factor in supervising iterative updates of the optimization object to better versions, which can take various forms as shown in Fig. 4 , such as reward signals in reinforcement learning (RL), gradients in gradient descent, or even human preference. The difference between this line of work and the previous one is that the feedback signal indicates a more clear optimization direction so that the search space is relatively small.

 
 
 Discrete prompt optimization for black-box LLMs can be formulated as the RL problem at an early stage. Typically, the policy combines a frozen PLM with a trainable neural network with a relatively small amount of parameters, and the reward reflects comprehensive feedback comparing the prediction of the target LLM with the ground truth.

 
 
 RLPrompt  ( Deng et al., 2022 ) consists of a frozen PLM (RoBERTa-large  ( Liu, 2019 ) ) and a learnable task-specific MLP (Multilayer Perceptron) to generate tokens of prefix prompt one by one to maximize the reward, leading to a semantically incoherent prompt.
DSP  ( Li et al., 2023b ) fine-tunes a small policy LM (T5  ( Raffel et al., 2020 ) ) using only minimal labeled data to generate nuanced, instance-specific directional stimulus prompts for original inputs, such as keywords added after summarization instructions.
PACE  ( Dong et al., 2023 ) leverages LLMs as the dual roles of actors and critics for iterative prompt refinement.
PRewrite  ( Kong et al., 2024 ) trains a prompt rewriter LLM (PaLM  ( Anil et al., 2023 ) ) to optimize input-independent instructions by RL and designs two rewrite strategies PRewrite-I with greedy decoding and PRewrite-S with search algorithm.
SCULPT  ( Kumar et al., 2024 ) is a novel framework that systematically refines long prompts in the form of a hierarchical tree through an actor-critic mechanism of RL. There are two complementary feedback mechanisms: Preliminary Assessment evaluates the prompt structure before execution and Error Assessment diagnoses model errors after execution.

 
 
 Figure 4. Various feedback signals contribute to specifying the optimization space of automatic prompt engineering, where the orange dashed line indicates the original prompt search space and the gray solid line represents the reduced search space that benefits from a more clear optimization direction. 
 
 
 In fact, gradient descent is the most common algorithm for exploring optimal solutions in continuous space, where the gradient is a form of negative feedback. In the context of the widespread prevalence of close-resource LLMs, meta-prompts are commonly used to guide LLMs to implement critical steps of automatic prompting optimization by themselves, especially gradient generation can be seen as a kind of self-feedback.

 
 
 ProTeGi  ( Pryzant et al., 2023 ) proposes a simple and nonparametric prompt optimization solution with the help of API-based LLMs. The non-parametric nature is reflected in the following aspects: (1) Textual gradient: LLM summarizes the flaws of the initial prompt as the textual gradient based on mini-batches of training data. (2) Backpropagation: Another LLM fixes flaws by editing the initial prompt into candidate prompts in the opposite semantic direction of the textual gradient. (3) Expansion: The paraphrasing LLM performs Monte Carlo search to explore the prompt space within a scope semantically similar to the candidates. (4) Beam search: The best arm identification algorithm in bandit optimization chooses the most promising candidates for the next iteration.
Interestingly, as the meta-prompt plays an important role in manipulating LLM engineers, PE2  ( Ye et al., 2023 ) focuses on prompt engineering the meta-prompt following concepts commonly used in gradient-based optimization.
To optimize search space for ProTeGi, PREFER  ( Zhang et al., 2024b ) designs a feedback-reflect-refine framework for prompt ensemble learning, where the feedback reflects on the inadequacies of each weak prompt and works in tandem to cover multiple sub-spaces during boosting. There is also an effective bagging method to enhance the stability of the prompt effect evaluation, which is superior to regular beam search or Monte Carlo search.
Based on gradient accumulation, AutoHint  ( Sun et al., 2023c ) deduces a hint for each sample (prompt-input pair) with incorrect output and summarizes the gradient from per-sample hints. UniPrompt  ( Juneja et al., 2024 ) generates optimized prompts that capture multiple facets of a task by clustering samples with similar task facets and combining feedback from each cluster like mini-batch.
Without summarizing gradient information, GPO  ( Tang et al., 2024 ) directly guides the LLM to realize the update direction from the meta-prompt including task prompts, wrong demonstrations and prompt-score trajectory, and perform the update method with edit distance.
To effectively adapt to diverse data distributions, AMPO  ( Yang et al., 2024 ) automatically constructs a multi-branched prompt to efficiently handle multiple patterns in complex tasks. LLM-Analyzer identifies the cause for each failed case and LLM-Summarizer summarizes all the causes into different patterns iteratively to reduce repetitive and redundant branches. LLM-Revisor finally optimizes the prompt either in depth (by adding more details) or in breadth (by adding more branches).

 
 
 Human preference alignment benefits LLMs in learning human intelligence, where the effect of alignment depends largely on the quality of the preference data.

 
 
 Similar to RLHF  ( Ouyang et al., 2022 ) , APOHF  ( Lin et al., 2024 ) automates prompt optimization by training a neural network to predict the best prompt based on human preference.
BPO  ( Cheng et al., 2023 ) constructs paired original and optimized prompts based on human preference feedback to train a 7B seq2seq model as the prompt preference optimizer. The optimized prompts are accommodated LLMs’ understanding and applicable to various LLMs, which further eliminate the Human-LLM alignment gap compared with RL based on PPO  ( Schulman et al., 2017 ) and DPO  ( Rafailov et al., 2023 ) .
APEER  ( Jin et al., 2024 ) employs the LLM to iteratively refine prompts, especially for passage relevance ranking tasks based on self-feedback and high-quality prompt preferences.
FIPO  ( Lu et al., 2024 ) introduces the first large-scale Prompt Optimization Preference dataset (POP) that collects rejected data from a suboptimal GPT-3.5 and chosen data from an optimal GPT-4 and undergoes rigorous cross-validation by human experts and analytical models. POP is used to fine-tune offline local LLM-based optimizers to improve free-form instruction-oriented prompts.

 
 
 However, whether LLM is a good prompt optimizer is a question worth exploring  ( Ma et al., 2024 ) .

 
 
 

#### 3.1.3. Editing-based methods

 
 In addition to the aforementioned methods, it is also feasible to expand the prompt space without deviating from the original prompt by editing the prompt directly with delete, swap, paraphrase, add, etc. operations at specific lexical unit levels.

 
 
 GrIPS  ( Prasad et al., 2023 ) is a gradient-free, edit-based search method consisting of three steps. Step 1 involves two operations: slicing and editing. Firstly, base instructions are split into shorter lexical units ( i.e., word, phrase, or sentence) using CRF-based constituency parser  ( Zhang et al., 2020 ) , with phrase-level slices proven to be the most helpful. Then, the certain slice is edited by one of four operations (delete, swap, paraphrase, addition) to expand the instruction space. Step 2 evaluates candidate instructions on a score set and selects the best instructions based on greedy search, beam search, and simulated annealing (SA) algorithms. Step 3 conducts multiple search iterations until the score no longer improves or a maximum threshold is reached. GrIPS has been shown to work for many prompt modes (with/without demonstrations) especially with instruction fine-tuned LLMs.
Plum  ( Pan et al., 2023 ) as a general prompt learning method that satisfies automatic, discrete, black-box, gradient-free, and interpretable all at once combines GrIPS editing operations with various metaheuristic algorithms (including hill climbing, simulated annealing, genetic algorithms with/without crossover, tabu search, and harmony search) to expand the discrete prompt space.
SPRIG  ( Zhang et al., 2024a ) especially optimizes system prompts with GrIPS editing operations.

 
 
 Editing operations can also be used to explore optimization space in the RL process. TEMPERA  ( Zhang et al., 2022 ) sequentially edits query-dependent prompts during test time. Editing actions for prompts include Swap, Add and Delete for instructions; Permute and Swap for examples; Change for verbalizers. According to the step reward proposed in RLPrompt  ( Deng et al., 2022 ) , TEMPERA uses the score difference between successive edits as the immediate reward and attention-based policy architecture to choose possible actions. Distinct from prior work, this formulation strikes a good balance between human prior knowledge and prompt performance.

 
 
 
 

### 3.2. CoT Optimization

 
 Given the brevity of instructions, the information they contain is predictably insufficient. To better inspire the potential of LLMs in solving complex tasks, task-related demonstrations can be added to enrich the context of the prompt. Among these techniques, Chain-of-Thought (CoT)  ( Wei et al., 2022b ) was introduced as a pivotal prompting strategy that significantly enhances LLMs’ reasoning capability. The core idea of CoT is “divide and conquer” which breaks down complex problems into finer-grained subproblems and explicitly deduces step by step in a complete framework of mind. As reasoning is one of the most meaningful tasks for AI systems, numerous optimization efforts for prompting frameworks represented by CoT have emerged. We incorporate them into CoT optimization in automatic prompt engineering and provide three categories similar to the instruction design classification criteria in § 3.1 . The final category involves the use of external tools to interact with broader knowledge bases.

 
 

#### 3.2.1. Sampling-based methods

 
 Zero-Shot-CoT  ( Kojima et al., 2022 ) has first validated LLMs can sample diverse CoT demonstrations (rationale-answer pairs) with a piece of instructive prompt “Let’s think step by step”, which greatly reduces the human resource requirements of manual-designed few-shot CoTs.
Following this, a new decoding strategy, self-consistency  ( Wang et al., 2023d ) , was proposed to replace the naive greedy decoding used in CoT sampling. Specifically, it first samples diverse rationale-answer pairs by adjusting the temperature coefficient and then selects the most consistent answer by majority voting.
These two studies significantly unleash the reasoning ability of LLMs and establish a reliable foundation for CoT prompting optimization.

 
 
 At first, LMSI  ( Huang et al., 2023a ) has demonstrated that LLMs can self-improve by training with mixed formats of rationale-augmented answers using CoT prompting and self-consistency.
Instead of augmenting training data, a lot of work has begun to focus on improving prompting input for LLMs based on their own output. To mitigate LLM hallucinations in Zero-Shot-CoT prompting, Auto-CoT  ( Zhang et al., 2023 ) utilizes Sentence-BERT  ( Reimers and Gurevych, 2019 ) to cluster questions with diversity and generate corresponding CoT demonstrations for more flexible and task-adaptive few-shot prompting.
Boosted Prompting  ( Pitis et al., 2023 ) encourages the LLM to sample multiple rationales for different problems. The problems with minimal accuracy or consistency and their correct rationales are considered as hard examples to form informative prompts that iteratively comprise a boosted ensemble, increasing overall coverage of the problem space.
COSP  ( Wan et al., 2023a ) specially designs a scoring function that incorporates consistency, diversity and repetitiveness to select outstanding outputs generated by Zero-Shot-CoT prompting LLMs in stage 1, and self-adaptively determines the number of demonstrations through outcome entropy in stage 2. Finally, majority voting over outputs from both stages forms the final prediction.
USP  ( Wan et al., 2023b ) is an improved, universal version of COSP  ( Wan et al., 2023a ) not limited to reasoning tasks. It employs the same candidate prompt generation method but differs in using a Task-Specific Selector, which designs scoring functions tailored to three classic NLP task types (classification, short-form generation, and long-form generation) to select the appropriate pseudo-demos. Finally, greedy decoding is used to replace majority voting.
Facing mixed-task scenarios, Meta-CoT  ( Zou et al., 2023 ) employs a routing mechanism to match the question type with an off-the-shelf Demo Pool. If unmatched, Zero-Shot-CoT prompting is used to generate CoT demonstrations that automatically update the data cache with similar-question clusters based on density like Auto-CoT  ( Zhang et al., 2023 ) .
Reprompting  ( Xu et al., 2023a ) generates step-by-step CoT demonstrations by Gibbs Sampling. It deduces the joint distribution of CoT demonstrations by sampling from the conditional distribution of the training data, which can also be viewed as a variant of evolutionary algorithms that iteratively finds effective CoT for each model given a few question-answer pairs without human intervention.

 
 
 

#### 3.2.2. Feedback-based methods

 
 Self-refine  ( Madaan et al., 2023 ) has verified that LLMs possess the competency for iterative refinement across various tasks based on self-feedback, where meta-prompts instruct the same LLM for initial generation, feedback, and refinement. As a result, many optimized prompting frameworks include iterative feedback, which is usually the LLM’s deep reflection on its own behavior that may be completely wrong or suboptimal. We summarize related work involving the nature of step-by-step reasoning in this category.

 
 
 Reinforcement learning still works in reasoning tasks, PromptPG  ( Lu et al., 2023 ) trains a small linear layer on top of the frozen PLM (BERT  ( Reimers and Gurevych, 2019 ) ) to select performing in-context examples based on the consistency between LLM predictions and labels, ensuring the integrity of prompts.
Prompt-OIRL  ( Sun et al., 2023b ) optimizes query-dependent prompts on an instance level rather than a distributional level by offline inverse RL based on a proxy reward model (XGBoost  ( Chen, 2015 ) ). The proxy fed with query-prompt pairs is trained to approximate the true online reward calculated by black-box LLMs and golden labels.
Reflexion framework  ( Shinn et al., 2023 ) designs a special memory module to transform scalar rewards into natural language feedback. The interaction history (call it trajectory) between the actor and the environment serves as the short-term memory while a verbal experience feedback summarized from the trajectory with its reward scores is stored in long-term memory. The Reflexion agent outperforms other LLMs based on the synergy of the two memory components.
In contrast to Reflexion, which executes a task multiple times for online dynamic feedback, TaskLLM in the PROMST framework  ( Chen et al., 2024a ) executes a task at once to automatically synthesize offline feedback about errors based on human-designed feedback rules. SumLLM summarizes feedback as context for GenLLM to generate new prompt candidates. To mitigate the evaluation cost, the score prediction model is fine-tuned to heuristically sample a subset of candidates for evaluation.

 
 
 Negative feedback may be more helpful for LLMs to deliberate. DTG  ( Li et al., 2023d ) designs templates to prompt LLMs to detect error types in irrelevant system output and triggers the deliberation ability of LLMs from the negative feedback. The prompt templates require only minimal adjustments to guide the LLM in performing a simple single-step inference applicable to a wide range of text generation tasks.
Reprompt  ( Chen et al., 2024b ) optimizes the step-by-step instructions of CoT across various reasoning tasks. If there is no such part, the additional checker will convert the CoT into one with a step-by-step instruction. Inspired by gradient descent, the loss is the focus point summarized from a batch of interaction history between the target LLM and the feedback generator. Then, the prompt optimizer mainly updates the common prompt part based on the loss. This act loop iterates until the prompt has converged.

 
 
 

#### 3.2.3. Interaction-based methods

 
 LLMs have been shown to often experience issues like hallucinations and error propagation when addressing reasoning tasks. To this end, it is increasingly common for LLMs to interact with external resources to refine their internal capabilities. For example, intervening with external tools to check for errors at intermediate steps in a prompting framework has proven to be highly effective.

 
 
 Improved ReAct  ( Yao et al., 2023b ) prompt trajectories are formed by sparsely synergizing CoT (reason-only) prompts and action-only prompts, which guide the LLM to generate reasoning traces and task-specific actions in an interleaved manner and overcome prevalent reasoning issues by interacting with a simple Wikipedia API.
It is more natural and conversational for Verify-and-Edit framework  ( Zhao et al., 2023 ) to produce verifying questions for less consistent CoT demonstrations and post-edit rationales with external knowledge for more factually aligned predictions.
ART framework  ( Paranjape et al., 2023 ) retrieves similar tasks from the task library to craft a few-shot prompt that decomposes the task into sub-steps and calls tools provided in the tool library such as search engine and code generation when necessary.
In self-ask framework  ( Press et al., 2023 ) , the LLM continues to explicitly ask follow-up questions and answer with a search engine until there is sufficient information to solve the original task.
ToolLLM  ( Qin et al., 2023 ) is a general tool-use framework that includes ToolBench constructed with API collection, instruction generation and solution annotation, ToolLLaMA supervised fine-tuned by ToolBench, and ToolEval for multi-round interaction evaluation.
ATC  ( Shi et al., 2024 ) shows that LLMs can not only utilize a chain of tools through programming until solving the complex tasks (multi-tool user) but also master fast-paced new tools automatically by black-box probing (multi-tool learner).

 
 
 
 
 

## 4. Prompt Compression

 
 Figure 5. Prompt compression in continuous and discrete space. T2V compression includes Internalizing system or user prompts into model parameters based on KD, and Encoding key information of hard prompts into soft prompts in an iterative or one-off way. T2T compression contains extractive and abstractive methods respectively are Pruning in various granularities, and Summarization for sufficient informativeness. 
 
 
 Prompt compression refers to distilling long text prompts into the shortest possible text or vectors without sacrificing LLM performance.

 
 
 Challenges .
With the increasing generative capability of LLMs, researchers have found that prompting LLMs with more specific contexts in few-shot settings can improve task accuracy. Meanwhile, the applications of LLMs in long-context scenarios, such as multi-turn dialogue, Retrieval-Augmented Generation (RAG), multi-document summarization, etc. , are being actively examined. Theoretically, the more comprehensive task-relevant knowledge covered in the context, the better LLMs perform. However, the unlimited input length will bring difficulties for the practical deployment: (1) Limited Context Window : Each LLM has a fixed input length identified during pre-training, any text exceeding this length will be truncated so that losing this part of contextual semantics. (2) Catastrophic Forgetting : When there is no sufficient cache space, the LLM may forget previously learned knowledge when modeling long sequences. (3) Slow Inference Speed : The large scale of LLMs is a double-edged sword, while LLMs with extensive parameters can store rich knowledge to be generalist, the over-consumption of computational resources is inevitable during inference.

 
 
 Solutions .
To alleviate limitations of the context window, reusable system prompts can be encapsulated within the model itself, thereby conserving more adequate space for user inputs. Following earlier parameter-efficient prompt tuning efforts ( Lester et al., 2021 ; Li and Liang, 2021 ) , a small number of learnable parameters (soft prompts) can carry task-specific information so that task descriptions in lengthy hard prompts can be transformed into more compact soft prompts to address catastrophic forgetting. Both approaches leverage open-source LLMs for training by Eq. 2 , in the case of closed-source LLMs, the information density of hard prompts can be increased with the help of the SLM for information measurement or the LLM for summarization as described in Eq. 3 to speed up inference.

 
 
 According to the type of compressed prompts, we divide prompt compression into Text-to-Vector (T2V) compression in continuous space and Text-to-Text (T2T) compression in discrete space. Figure 5 depicts the four main types of methods, T2V compression considers the LLM as a compressor that compresses essential text information into more compact vector representations. T2T compression directly measures the token informativeness and compresses as much as possible redundant information with acceptable performance loss.

 
 

### 4.1. Text-to-Vector Compression

 
 When a language model processes text prompts, converting them into vectors is a necessary step. Since vectors can be denser representations than text, it is natural to consider compressing discrete text prompts outside the model into continuous vectors inside the model. In this context, continuous vectors can be either internal parameters of the model referred to as internalization, or additional soft prompts known as encoding. This kind of prompt compression not only expands the context window but also accelerates inference speed especially when the same prompts are frequently reused.

 
 

#### 4.1.1. Internalization

 
 Here, we introduce the concept of Knowledge Distillation (KD)  ( Hinton et al., 2015 ) , where the knowledge from a larger teacher model is transferred into a smaller student model. Inspired by this, the language model can be fine-tuned with Kullback-Leibler (KL) divergence of output distribution between original and compressed prompts to distill the knowledge from text prompts into model parameters, then achieve the internalization of prompt functionality.

 
 
 The initial research efforts attempted to internalize the system prompts. To develop a versatile AI system aligned with human values, Context Distillation  ( Askell et al., 2021 ) internalizes an HHH (helpful, honest, and harmless) prompt within the language model by KD between the original model with the context and input and the distilled model with the input alone.
PI (Prompt Injection)  ( Choi et al., 2022 ) proposes the Pseudo-INput Generation (PING) method to internalize system prompts, which avoids the repeated computation of reused prompts during inference. The first stage generates pseudo-inputs based on prompts using an input generator trained on task-specific data. The second stage performs KD between the teacher model with the prompt and pseudo-input and the student model only with the pseudo-input.

 
 
 Later on, researchers start compressing the context of the user prompts inside the model. Snell et al. (2022) propose to distill three specific types of context in the prompt: abstract task instructions, complex reasoning, and concrete training examples. Based on teacher and student prompt templates, the original input is transformed into two kinds of prompts with significant distribution differences. The language model is trained with different prompts to produce the same response for internalization. Three distillation methods are tested to demonstrate that once the model learns the knowledge of task-specific prompts, it can perform the corresponding task without explicit prompts.
Instruction Distillation  ( Sun et al., 2023a ) internalizes three complex instruction techniques: pointwise ranking, pairwise ranking, and listwise ranking. The goal is to address the inefficiency of zero-shot relevance ranking by enabling LLMs to rank efficiently with simpler instructions.
Especially for reasoning, Distilling Step-by-Step  ( Hsieh et al., 2023a ) not only enables student SLM to learn labels from the teacher LLM but also extracts rationales as additional supervisory signal for internalizing LLM’s reasoning capability inside the SLM under a multi-task framework, which achieves better performance than standard fine-tuning with smaller model sizes and less training data.
As for retrieval-augmented generation task, inspired by the modality encoder CLIP  ( Radford et al., 2021 ) extracting modality features, xRAG  ( Cheng et al., 2024 ) redefines document embeddings in dense retrieval as features from the retrieval modality, which are seamlessly integrated into the language model representation space using a plug-and-play projector (modality fusion bridge). There are two training phases for xRAG: first, the LLM is internally compatible with compressed representations provided by the projector with paraphrase instructions, and then the frozen LLM with trainable projector is instruction fine-tuned to perform specific downstream tasks based on compressed representations. Self-distillation is also used to enhance xRAG’s resilience in noisy contexts.

 
 
 

#### 4.1.2. Encoding

 
 In the above internalization methods, the compressed hard prompts lack explicit carriers, while encoding methods predefine LLM-adapted soft prompts to represent hard prompts such as multi-task instructions, lengthy context, etc. Normally, there is a compressor encoding hard prompts into soft prompts that can be applied for various downstream tasks efficiently.

 
 
 Wingate et al. (2022) first proposed to compress hard prompts into soft prompts, where hard prompts are prepended to the input sequence while soft prompts are prepended to the input embeddings. By minimizing both output distributions, important information from complex hard prompts is distilled into concise soft prompts. Although this training process is costly, it reduces inference costs when the same prompts are reused.

 
 
 There are a series of works derived from Gisting  ( Mu et al., 2023 ) as shown in Fig. 6 with the goal of better generalization among diverse prompts.

 
 
 Figure 6. Gisting series work compresses key information of hard prompts into gist tokens based on an encoder or a decoder trained with special attention mechanisms. The matrices in the upper half represent masking strategies, where the gray box indicates the standard mask and the yellow box indicates the gist mask. 
 
 
 The goal of Gisting  ( Mu et al., 2023 ) is encoding multi-task instructions into gist tokens that can be cached and reused for computation efficiency. Notably, it leverages meta-learning in HyperTuning  ( Phang et al., 2023 ) to predict gist tokens instead of gradient descent to update soft prompts. Specifically, gist tokens are first added to the vocabulary and embedding matrix, then concatenated after the instructions, and finally fine-tuned based on masking as depicted in Fig. 6 -(a). Gisting can be applied to both decoder-only and encoder-decoder architecture, enabling up to 26x compression of prompts, a 40% reduction in FLOPs, 4.2% wall time speedups, and storage savings, all with minimal loss in output quality.
Inspired by the Minimum Description Length (MDL) principle  ( Grünwald, 2007 ) from information theory, Gist-COCO (Gist COnditioned COding)  ( Li et al., 2024a ) compresses original prompts into shorter gist prompts with an encoder-decoder architecture. The encoder is fine-tuned to compress gist tokens into gist representations to simulate the output distribution of the original prompts via KL divergence. The decoder verbalizes these gist representations into gist prompts to generalize across different LLMs with high compression rates.
In addition to compressing short instructions, Gisting can also compress long contexts. As the name suggests, UltraGist  ( Zhang et al., 2024c ) compresses ultra-long context with gist tokens. It employs a decoder-only architecture with an optimized cross-attention mechanism as shown in Fig. 6 -(c) to progressively compress fine-grained context segments into UltraGist tokens that follows the next segment for subsequent compression. It distinctly allows randomly sampled compression ratios to deal with dynamic context while maintaining near-lossless performance.

 
 
 In the following, we introduce some research as shown in Fig. 7 that specializes in compressing long context in the prompt, which realizes advantages in both inference acceleration and GPU memory reduction.

 
 
 Figure 7. Differences of representative encoding methods specially for long context. AutoCompressor iteratively compresses context segments with summary tokens. ICAE compresses the complete context all at once with memory tokens. SelfCP only compresses unlimited context segments based on a connector. 
 
 
 AutoCompressor  ( Chevalier et al., 2023 ) employs a Recurrent Memory Transformer (RMT)  ( Bulatov et al., 2022 ) architecture to iteratively compress arbitrary-length context segments with summary tokens into summary vectors. In subsequent iterations, summary vectors from the previous segment are concatenated in front of the next segment and summary tokens are concatenated at the end. Additionally, the backpropagation through time (BPTT) method  ( Chen et al., 2016 ) stops gradient updates on cached summary vectors after two steps of compression, further reducing computational load and GPU memory requirements.
In terms of the RAG system, LLoCO (Learning Long Contexts Offline)  ( Tan et al., 2024 ) employs the AutoCompressor to compress long documents into multiple summary embeddings stored in the vector database for retrieval.

 
 
 Another line of work compresses the complete context all at once instead of iterating in segments. There is usually a trainable LLM for encoding (refer to as compressor) and a frozen LLM for decoding (refer to as generator). ICAE (In-context Autoencoder)  ( Ge et al., 2023 ) fine-tunes a LoRA-adapted LLM to encode the context with memory tokens into memory slots based on the PWC (Prompt-with-Context) dataset. The generator with memory slots is pre-trained with autoencoding and text continuation tasks to reconstruct the original context and apply it to various downstream tasks.
500xCompressor  ( Li et al., 2024b ) similarly fine-tunes LoRA-adapted LLM to compress prompts. The difference is that the generator learns to adapt KV values of compressed prompts as inputs and responses based on them by pre-training with the regeneration task and fine-tuning with the question-answering task. 500xCompressor can compress any text, answer various types of questions and achieve compression ratios ranging from 6x to 480x.
As for LLM-based recommendation, POD (Prompt Distillation)  ( Li et al., 2023e ) is primarily used for three typical tasks: sequential recommendation, Top-N recommendation and explainable recommendation. The backbone model follows an encoder-decoder architecture where the encoder distills the discrete prompt templates into multiple continuous prompt vectors with an additional whole-word embedding to ensure the integrity of the item ID, the decoder generates recommendations based on these prompt vectors.
Following this, RDRec  ( Wang et al., 2024b ) framework consists of two stages to synthesize training data and internalize rationales into a smaller model. In the interaction rationale distillation, the LLM generates rationales including user preferences and item attributes as training labels. In the rationale-aware recommendation, prompt template vectors compressed by POD  ( Li et al., 2023e ) are concatenated with user and item IDs as training inputs. RDRec achieves SOTA performance in both sequential and Top-N recommendations.

 
 
 In order to balance training cost, inference efficiency, and generation quality, SelfCP (Self-Compressor)  ( Gao et al., 2024 ) only compresses over-limit prompts to reduce the compression and generation difficulty. There are three components in SelfCP: the compressor and the generator are frozen LLMs, and a trainable connector between them is a 17M linear layer. Different from AutoCompressor with an inefficiency training process and ICAE with limited input length, SelfCP employs decoder-only compressors to compress each over-limit segment asynchronously and concatenates them as the AutoCompressor. Then, the connector is fine-tuned to project the hidden states on top of memory tags into LLM-acceptable memory tokens. As for in-context learning, it is more efficient to directly withdraw memory tokens compressed from target demonstrations.

 
 
 Table 3. Comparisons between different encoding methods. 
 
 
 
 Encoding Methods | 
 Target Model | 
 Compressor Model | 
 Soft prompt location | 
 Hard prompt | 
 Soft prompt | 

 
 
 
 
 Prompt 
 
 Compression 
 | 
 Decoder-only | 
 
 
 
 Bayesian attribute 
 
 classifier framework 
 | 
 Before context | 
 Context | 
 Embeddings | 
 Vectors | 

 
 Gisting | 
 
 
 
 Encoder-Decoder 
 
 Decoder-only 
 | 
 With gist masking | 
 
 
 
 Between instruction 
 
 and context 
 | 
 Instruction | 
 Gist tokens | 
 Vectors | 

 
 Gist-COCO | 
 Encoder-Decoder | 
 Encoder | 
 Before prompt | 
 Prompt | 
 Gist tokens | 
 Gist representations | 

 
 UltraGist | 
 Decoder-only | 
 
 
 
 With optimized 
 
 cross-attention 
 | 
 After context segment | 
 Context | 
 Gist tokens | 
 Vectors | 

 
 AutoCompressor | 
 RMT | 
 RMT | 
 After context segment | 
 Context | 
 Summary tokens | 
 Summary vectors | 

 
 ICAE | 
 Encoder-Decoder | 
 Encoder (LoRA) | 
 After context | 
 Context | 
 Memory tokens | 
 Memory Slots | 

 
 500xCompressor | 
 Encoder-Decoder | 
 Encoder (LoRA) | 
 After context | 
 Context | 
 Compressed tokens | 
 K V values | 

 
 POD | 
 Encoder-Decoder | 
 Encoder | 
 Before context | 
 Context | 
 Embeddings | 
 Vectors | 

 
 RDRec | 
 Encoder-Decoder | 
 Encoder | 
 Before context | 
 Rationale | 
 Embeddings | 
 Vectors | 

 
 SelfCP | 
 Decoder-only | 
 Decoder-only | 
 After context segment | 
 
 
 
 Over-limit 
 
 context 
 | 
 Memory tokens | 
 Vectors | 

 
 
 
 Here, we provide comparisons between different encoding methods, as shown in Table 3 . We find that soft prompts are usually prepended to the input sequence in the compressor with Decoder-only architecture while appended to that of Encoder architecture, which is related to the training task mentioned in § 2.2 .

 
 
 
 

### 4.2. Text-to-Text Compression

 
 Text-to-Vector compression in continuous space focuses on the perspective of adapting language models to natural language. Nevertheless, the compressed soft prompts usually lack human readability and interpretability. To further facilitate seamless interaction between LLMs and humans, researchers consider compressing prompts only in discrete space from the perspective of human understanding natural language. The common practice is to utilize accessible language models to measure the information density of hard prompts and then shorten the prompt length as much as possible with stable LLM performance. We conclude two categories of methods as shown in the right half of Fig. 5 , pruning can be defined as an extractive compression method while summarization is an abstractive compression method.

 
 

#### 4.2.1. Pruning

 
 Pruning indicates the direct removal of less informative lexical units in different granularity ( i.e., token, phrase, sentence) without changing the linguistic expression of the original prompt. It measures the amount of information by leveraging smaller language models to calculate information entropy or training discriminator models to determine whether to discard certain tokens in the prompt. According to different filter units, there are three kinds of pruning schemes: coarse-grained, coarse-to-fine grained and fine-grained.

 
 
 Coarse-grained . Demonstration-level : In order to achieve performance-efficiency trade-off of black-box LLMs, DynaICL  ( Zhou et al., 2023a ) trains a meta controller to dynamically allocate the number of in-context demonstrations according to the input complexity and the computational budget.
 Sentence-level : FilCo  ( Wang et al., 2023a ) employs three lexical and information-theoretic methods—String Inclusion, Lexical Overlap, and Conditional Cross-Mutual Information (CXMI)—to distill retrieved documents into a useful context that trains a context filtering model as well as a generation model for RAG tasks. To maintain the semantic integrity of the context, CPC  ( Liskavets et al., 2024 ) trains a context-aware sentence encoder to measure the embedding similarity (cosine distance) between the given question and each sentence in the context, and then removes irrelevant sentences for compression.
 Document-level : AdaComp  ( Zhang et al., 2024e ) trains a compression-rate predictor to dynamically select optimal documents based on query complexity and retrieval quality.

 
 
 Coarse-to-fine grained . Microsoft proposes the LLMLingua series of research to compress prompts for accelerating LLM inference. Instead of considering the prompt as a whole, LLMLingua  ( Jiang et al., 2023a ) is the first to separately compress different components ( i.e., instruction, question, and demonstration) of the prompt in a coarse-to-fine way. First of all, an SLM aligned with the LLM by instruction fine-tuning is prepared to measure token informativeness based on perplexity (PPL). There is a budget controller dynamically allocating compression ratios for different components according to their importance, where the instruction and question are predefined a lower ratio while the higher ratios are left for demonstrations. Coarse-grained compression reduces the number of demonstrations and fine-grained compression iteratively filters tokens with lower PPL in the remained demonstrations.
As the name suggests, LongLLMLingua  ( Jiang et al., 2023b ) further compress long documents on the basis of the LLMLingua framework. There is a linear scheduler adaptively controlling the fine-grained compression ratio based on the coarse-grained compression ratio. The significant difference is that LongLLMLingua measures the amount of information relevant to the given question, where a reordering mechanism is used to identify more question-aware documents and contrastive perplexity is proposed to retain more question-related tokens. Finally, the subsequence recovery method ensures the integrity of the key information (such as time and location) in the LLM response. With above tricks, compressed prompts can derive higher performance at lower costs while reducing end-to-end system latency.
CoT-Influx  ( Huang et al., 2023b ) is a plug-and-play coarse-to-fine pruner to specially compress CoT prompts generated by GPT-4  ( Achiam et al., 2023 ) . The shot-pruner reduces useless CoT examples and the token-pruner filters redundant tokens. Both pruners are a two-layer feed-forward network (MLP) with GELU activation to be trained by RL with multi-objective rewards instead of backpropagation.

 
 
 Fine-grained . Selective Context  ( Li et al., 2023a ) 
evaluates informativeness of lexical units with self- information  ( Shannon, 1948 ) computed by a causal language model and filters redundant content by a percentile-based approach. It is worth mentioning that token self-information is calculated by predicting the next token probability, which is accumulated in the range of lexical units and then averaged to obtain phrase and sentence level self-information. Phrase has been proven to be the most effective filtering unit that can reduce inference memory usage by 36% and inference time by 32%, with negligible performance drop, striking a good balance between efficiency and performance.
In order to retain syntactic and semantic structure, PROMPT-SAW  ( Ali et al., 2024 ) extracts tokens with key information via relation-aware graphs and reinstates them to compressed prompts that achieve better readability and interpretability.
In addition to pruning redundant tokens by explicitly calculating the amount of information, the language model can be trained to determine whether to prune certain tokens or not by themselves. PCRL  ( Jung and Kim, 2023 ) considers prompt compression as a binary classification task, which is performed by a frozen pre-trained policy LM with trainable MLP layers based on RL. The compression policy assigns an include or exclude label for each token in the prompt and the reward function considers both faithfulness and reduced length of the compressed prompt.
Instead of calculating information entropy by a unidirectional decoder-only model, LLMLinga-2  ( Pan et al., 2024 ) employs a bidirectional encoder-only model with a linear classification layer as a compressor to decide whether each token preserve or discard . The small but general compressor is trained 10 epochs on distilled data synthesized by GPT-4  ( Achiam et al., 2023 ) to significantly enhance performance, with a 3x-6x improvement in compression speedup, and a 1.6x-2.9x acceleration in end-to-end latency.

 
 
 Table 4. The LLM performance and prompt compression ratio ( shot is the number of demonstrations and × is the ratio of the original prompt to the compressed prompt) of various pruning methods in commonly used reasoning tasks and long-context tasks. It is worth noting that, due to variations in the experimental setups of different methods, their performances may be not directly comparable. 
 
 
 
 Methods | 
 
 
 
 Compression 
 
 Granularity 
 | 
 NaturalQuestions | 
 GSM8K | 
 BBH | 
 ZeroSCROLLS | 
 LongBench | 

 
 F1 | 
 Ratio | 
 EM | 
 Ratio | 
 EM | 
 Ratio | 
 Acc | 
 Ratio | 
 Acc | 
 Ratio | 
 Latency | 

 
 DynalCL | 
 demonstration | 
 42.40(EM) | 
 10-shot | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 

 
 40.20(EM) | 
 5-shot | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 

 
 FliCo | 
 sentence | 
 61.80 | 
 5-shot | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 

 
 CPC | 
 sentence | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 34.90 | 
 3× | 
 50.00 | 
 3× | 
 1× | 

 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 33.80 | 
 5× | 
 49.50 | 
 5× | 
 - | 

 
 AdaComp | 
 document | 
 70.96 | 
 3.66-shot | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 

 
 LLMLingua | 
 demonstration - token | 
 - | 
 - | 
 79.08 | 
 5× | 
 70.11 | 
 3× | 
 30.70 | 
 3× | 
 37.40 | 
 3× | 
 9.8× | 

 
 30.00 | 
 3.8× | 
 77.41 | 
 14× | 
 61.60 | 
 5× | 
 27.20 | 
 5× | 
 34.60 | 
 5× | 
 - | 

 
 LongLLMLingua | 
 document - token | 
 75.50 | 
 3.9× | 
 - | 
 - | 
 - | 
 - | 
 32.80 | 
 3× | 
 48.80 | 
 3× | 
 10.93× | 

 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 32.50 | 
 6× | 
 48.00 | 
 6× | 
 - | 

 
 CoT-Influx | 
 CoT - token | 
 - | 
 - | 
 73.31 | 
 7.7× | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 

 
 Selective Context | 
 token, phrase, sentence | 
 43.80 | 
 3.7× | 
 53.98 | 
 5× | 
 54.27 | 
 3× | 
 20.70 | 
 3× | 
 32.00 | 
 3× | 
 - | 

 
 - | 
 - | 
 52.99 | 
 11× | 
 54.02 | 
 5× | 
 19.40 | 
 5× | 
 24.80 | 
 5× | 
 - | 

 
 PROMPT-SAW | 
 entity, relation | 
 73.22(EM) | 
 3.86× | 
 72.12 | 
 1.49× | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 
 - | 

 
 LLMLingua-2 | 
 token | 
 71.90 | 
 3.9× | 
 79.08 | 
 5× | 
 70.02 | 
 3× | 
 33.50 | 
 3× | 
 42.20 | 
 3× | 
 0.67× | 

 
 - | 
 - | 
 77.79 | 
 14× | 
 61.94 | 
 5× | 
 33.40 | 
 5× | 
 39.10 | 
 5× | 
 - | 

 
 
 
 Here, we provide an overall empirical insight into pruning methods across different compression granularities on classical benchmarks, as shown in Table 4 . We observe that coarse-to-fine compression seems to be more beneficial for complex reasoning tasks while fine-grained compression is more suitable for long context tasks.

 
 
 

#### 4.2.2. Summarization

 
 In essence, summarization is a semantic-level compression that may change linguistic expressions while retaining the original idea. To ensure that the LLM performance of the compressed prompt does not deviate significantly from the original prompt, there are two types of methods with and without training will be introduced in the following.

 
 
 The first type of methods typically take the original output as a supervised signal to train the summarizer.
In RAG scenarios, RECOMP (Retrieve, Compress, Prepend)  ( Xu et al., 2023b ) compresses the retrieved documents into textual summaries prior to in-context integration. There are two query-focus compressors summarizing multi-documents to improve LLM performance. The extractive compressor is a dual encoder model (110M) trained to select sentences with high semantic similarity to the input query by contrastive learning. The abstractive compressor is an encoder-decoder model (775M) to learn the summarization ability of the LLM by knowledge distillation, which realizes selective augmentation instead of prepending irrelevant documents.
Nano-Capsulator  ( Chuang et al., 2024 ) is a compressor LLM to semantically compress prompts in natural language formats via summarization instructions. The response difference between the original and compressed prompt can be viewed as the reward feedback to monitor the optimization of Nano-Capsulator. There is also a strict cut-off mechanism to restrict the length of the compressed prompt.

 
 
 The second type of methods monitor whether the information of the summarized prompt is sufficient to correctly respond in real-time.
MEMWALKER  ( Chen et al., 2023b ) compresses long contexts by interactively prompting the LLM instead of fine-tuning. It consists of two stages where the first stage iteratively summarizes context segments to construct a memory tree and the second stage navigates from the root node to search for sufficient relevant information to respond the given query.
CompAct  ( Yoon et al., 2024 ) mainly addresses question-answering tasks in long context scenarios. The compressor LLM sequentially summarizes each segment of retrieved documents into a compressed prompt, whose information is evaluated to determine whether it is complete to answer the given question. If incomplete, the compressed prompt will be concatenated with the subsequent segment for next-iteration summarization. CompAct can serve as a cost-efficient and flexible plug-in compressor between off-the-shelf retrievers and readers, achieving exceptionally high compression rates (47x).
Style-Compress  ( Pu et al., 2024 ) can be seen as an efficient few-shot prompting method combining automatic prompt engineering and prompt compression tricks. A smaller LLM (LLaMA-2-7B  ( Touvron et al., 2023 ) ) iteratively compresses the original prompt by prompting with diverse human-written styles and in-context learning with task-specific high-performing examples in turn. Then, the best compressed prompt evaluated by a larger LLM (LLaMA-2-13B / GPT-3.5) and its comparative advantage are added to the demonstrations pool that instructs the compressor to conduct task-specific prompt compression.

 
 
 
 
 

## 5. Future Directions

 
 Despite the remarkable advances, the area of LLM prompting still faces several major challenges, which also provide exciting opportunities for future research. We conclude our review with a conjecture of promising areas of future work, specifically highlighting the intersection with two resource-saving prompting strategies. Meanwhile, we point out some open problems in the future research direction.

 
 
 
 • 
 
 Combining two resource-saving prompting strategies as a Win-Win solution: Actually, there is a gap between optimization objectives in prompt engineering and prompt compression. The former tends to optimize shorter instructions, while the latter mainly deals with longer demonstrations. Chain-of-Thought (CoT) represents the intersection of the interests of both areas, particularly as it plays a pivotal role in reasoning, which is a key research focus in the pursuit of Artificial General Intelligence (AGI). Future research could improve the CoT prompting framework based on the Eq. 1 and 3 . One feasible option is to consider straightforward nesting two strategies. For instance, iteratively self-improving the thought process first, and then compressing useless information from the lengthy CoT. Alternatively, both strategies could be performed synchronously in a single iteration, possibly using reinforcement learning where LLM performance serves as a reward signal to supervise compression. In addition, designing balancing factors to integrate the objective functions Eq. 1 and 3 could facilitate analytical work, such as discussing the mutual constraints of optimization and compression levels, as well as how to balance various metrics such as model performance, computational costs, and inference speed.

 

 • 
 
 Discussion of differences between LLMs and Humans in understanding natural language: While prompt compression accelerates LLM inference and generally maintains or slightly improves performance, it often struggles to ensure the readability and interpretability of the compressed prompts. So, there is an opportunity to explore the significance of gibberish versus human-interpretable compressed prompts in the context of AGI. Is the information defined from the perspective of human comprehension a reasonable standard for measuring the effective information provided for LLMs? Investigating this distinction could yield valuable insights into optimizing prompts for better alignment with humans.

 

 • 
 
 Robustness of efficient prompting: Optimized instructions typically cater to specific downstream tasks, so it is worth investigating how to enhance the transferability of prompts across different data distributions during the automatic optimization process. For instance, mixing labeled and unlabeled data might facilitate the learning of a more universal prompt. Similarly, T2V-compressed prompts are typically tailored to a specific language model, making it challenging to adapt to other models without fine-tuning. One potential solution is to encapsulate soft prompts within a plug-and-play adapter, enhancing the transferability of encoded representations and achieving both high performance and general capability.

 

 • 
 
 Open questions for future reflection: Limited by the reasoning ability of SLMs, automatic prompt engineering heavily relies on LLMs to optimize prompts  ( Zhang et al., 2024d ) . Hence, both model capabilities and computational costs need to be considered. While automatic prompt engineering significantly alleviates human labor, it still cannot completely avoid human intervention. Specifically, during the optimization of instructions, the initial iteration requires users to provide input-output examples or high-quality initial prompts; during the optimization of demonstrations (especially CoT), careful design of meta-prompt content or frameworks is necessary to correctly guide LLMs in generating appropriate responses.

 

 
 
 
 

## 6. Conclusion

 
 This survey provides an extensive tour of recent advances in efficient prompting methods for LLMs through the lens of reducing resource consumption.
We comprehensively review two families of existing efficient prompting methods: automatic prompt engineering to avoid human resources and prompt compression to save computational resources. Notably, we distill the optimization objectives of each category from a mathematical perspective and expect to combine both resource-saving strategies for lightweight LLM applications in the future. We further summarize representative methods for each category with a particular emphasis on efficiency. Finally, we outline potential directions for future research and provide a list of open-source efficient prompting projects as shown in Appendix A.1 .

 
 
 Acknowledgements. 
This work was supported in part by the National Science Foundation of China (No.62276056), the Natural Science Foundation of Liaoning Province of China (2022-KF-16-01), the Fundamental Research Funds for the Central Universities (Nos. N2216016 and N2316002), the Yunnan Fundamental Research Projects (No. 202401BC070021), and the Program of Introducing Talents of Discipline to Universities, Plan 111 (No.B16009).

 
 
 

## References

 
 
 Achiam et al . (2023) 
 
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al . 2023.

 
 Gpt-4 technical report.

 
 ArXiv preprint abs/2303.08774 (2023).

 
 
 https://arxiv.org/abs/2303.08774 

 

 
 Ali et al . (2024) 
 
Muhammad Asif Ali, Zhengping Li, Shu Yang, Keyuan Cheng, Yang Cao, Tianhao Huang, Lijie Hu, Lu Yu, and Di Wang. 2024.

 
 PROMPT-SAW: Leveraging Relation-Aware Graphs for Textual Prompt Compression.

 
 ArXiv preprint abs/2404.00489 (2024).

 
 
 https://arxiv.org/abs/2404.00489 

 

 
 Anil et al . (2023) 
 
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al . 2023.

 
 Palm 2 technical report.

 
 ArXiv preprint abs/2305.10403 (2023).

 
 
 https://arxiv.org/abs/2305.10403 

 

 
 Askell et al . (2021) 
 
Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Benjamin Mann, Nova Dassarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, John Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Christopher Olah, and Jared Kaplan. 2021.

 
 A General Language Assistant as a Laboratory for Alignment.

 
 ArXiv preprint abs/2112.00861 (2021).

 
 
 https://arxiv.org/abs/2112.00861 

 

 
 Besta et al . (2024) 
 
Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. 2024.

 
 Graph of Thoughts: Solving Elaborate Problems with Large Language Models. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada , Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (Eds.). AAAI Press, 17682–17690.

 
 
 https://doi.org/10.1609/AAAI.V38I16.29720 

 

 
 Brown et al . (2020) 
 
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever,
and Dario Amodei. 2020.

 
 Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.).

 
 
 https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html 

 

 
 Bulatov et al . (2022) 
 
Aydar Bulatov, Yuri Kuratov, and Mikhail Burtsev. 2022.

 
 Recurrent Memory Transformer. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2022/hash/47e288629a6996a17ce50b90a056a0e1-Abstract-Conference.html 

 

 
 Chen et al . (2023a) 
 
Angelica Chen, David Dohan, and David R. So. 2023a.

 
 EvoPrompting: Language Models for Code-Level Neural Architecture Search. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html 

 

 
 Chen et al . (2023b) 
 
Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. 2023b.

 
 Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading.

 
 ArXiv preprint abs/2310.05029 (2023).

 
 
 https://arxiv.org/abs/2310.05029 

 

 
 Chen (2015) 
 
T Chen. 2015.

 
 Xgboost: extreme gradient boosting.

 
 R package version 0.4-2 1, 4 (2015).

 
 
 

 
 Chen et al . (2016) 
 
Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016.

 
 Training Deep Nets with Sublinear Memory Cost.

 
 ArXiv preprint abs/1604.06174 (2016).

 
 
 https://arxiv.org/abs/1604.06174 

 

 
 Chen et al . (2024b) 
 
Weizhe Chen, Sven Koenig, and Bistra N. Dilkina. 2024b.

 
 RePrompt: Planning by Automatic Prompt Engineering for Large Language Models Agents.

 
 ArXiv preprint abs/2406.11132 (2024).

 
 
 https://arxiv.org/abs/2406.11132 

 

 
 Chen et al . (2022) 
 
Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2022.

 
 Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.

 
 Trans. Mach. Learn. Res. 2023 (2022).

 
 
 https://api.semanticscholar.org/CorpusID:253801709 

 

 
 Chen et al . (2024a) 
 
Yongchao Chen, Jacob Arkin, Yilun Hao, Yang Zhang, Nicholas Roy, and Chuchu Fan. 2024a.

 
 PRompt Optimization in Multi-Step Tasks (PROMST): Integrating Human Feedback and Heuristic-based Sampling.

 
 
 https://api.semanticscholar.org/CorpusID:270559211 

 

 
 Cheng et al . (2023) 
 
Jiale Cheng, Xiao Liu, Kehan Zheng, Pei Ke, Hongning Wang, Yuxiao Dong, Jie Tang, and Minlie Huang. 2023.

 
 Black-Box Prompt Optimization: Aligning Large Language Models without Model Training.

 
 ArXiv preprint abs/2311.04155 (2023).

 
 
 https://arxiv.org/abs/2311.04155 

 

 
 Cheng et al . (2024) 
 
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao. 2024.

 
 xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token.

 
 ArXiv preprint abs/2405.13792 (2024).

 
 
 https://arxiv.org/abs/2405.13792 

 

 
 Chevalier et al . (2023) 
 
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023.

 
 Adapting Language Models to Compress Contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 3829–3846.

 
 
 https://doi.org/10.18653/v1/2023.emnlp-main.232 

 

 
 Choi et al . (2022) 
 
Eunbi Choi, Yongrae Jo, Joel Jang, and Minjoon Seo. 2022.

 
 Prompt Injection: Parameterization of Fixed Inputs.

 
 ArXiv preprint abs/2206.11349 (2022).

 
 
 https://arxiv.org/abs/2206.11349 

 

 
 Chuang et al . (2024) 
 
Yu-Neng Chuang, Tianwei Xing, Chia-Yuan Chang, Zirui Liu, Xun Chen, and Xia Hu. 2024.

 
 Learning to Compress Prompt in Natural Language Formats. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , Kevin Duh, Helena Gomez, and Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City, Mexico, 7756–7767.

 
 
 https://aclanthology.org/2024.naacl-long.429 

 

 
 Cui et al . (2024) 
 
Wendi Cui, Jiaxin Zhang, Zhuohang Li, Hao Sun, Damien Lopez, Kamalika Das, Bradley Malin, and Sricharan Kumar. 2024.

 
 PhaseEvo: Towards Unified In-Context Prompt Optimization for Large Language Models.

 
 ArXiv preprint abs/2402.11347 (2024).

 
 
 https://arxiv.org/abs/2402.11347 

 

 
 Deng et al . (2022) 
 
Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric Xing, and Zhiting Hu. 2022.

 
 RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 3369–3391.

 
 
 https://doi.org/10.18653/v1/2022.emnlp-main.222 

 

 
 Devlin et al . (2019) 
 
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.

 
 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , Jill Burstein, Christy Doran, and Thamar Solorio (Eds.). Association for Computational Linguistics, Minneapolis, Minnesota, 4171–4186.

 
 
 https://doi.org/10.18653/v1/N19-1423 

 

 
 Dong et al . (2023) 
 
Yihong Dong, Kangcheng Luo, Xue Jiang, Zhi Jin, and Ge Li. 2023.

 
 PACE: Improving Prompt with Actor-Critic Editing for Large Language Model.

 
 ArXiv preprint abs/2308.10088 (2023).

 
 
 https://arxiv.org/abs/2308.10088 

 

 
 Fernando et al . (2023) 
 
Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. 2023.

 
 Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution.

 
 ArXiv preprint abs/2309.16797 (2023).

 
 
 https://arxiv.org/abs/2309.16797 

 

 
 Gao et al . (2024) 
 
Jun Gao, Ziqiang Cao, and Wenjie Li. 2024.

 
 SelfCP: Compressing over-limit prompt via the frozen large language model itself.

 
 Information Processing Management (2024).

 
 
 https://api.semanticscholar.org/CorpusID:270063106 

 

 
 Ge et al . (2023) 
 
Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, and Furu Wei. 2023.

 
 In-context Autoencoder for Context Compression in a Large Language Model.

 
 ArXiv preprint abs/2307.06945 (2023).

 
 
 https://arxiv.org/abs/2307.06945 

 

 
 Grünwald (2007) 
 
Peter D. Grünwald. 2007.

 
 The Minimum Description Length Principle (Adaptive Computation and Machine Learning).

 
 
 https://api.semanticscholar.org/CorpusID:119390683 

 

 
 Guo et al . (2023) 
 
Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang, Tsinghua University, and Microsoft Research. 2023.

 
 Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers.

 
 ArXiv preprint abs/2309.08532 (2023).

 
 
 https://arxiv.org/abs/2309.08532 

 

 
 Hinton et al . (2015) 
 
Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015.

 
 Distilling the Knowledge in a Neural Network.

 
 ArXiv preprint abs/1503.02531 (2015).

 
 
 https://arxiv.org/abs/1503.02531 

 

 
 Hsieh et al . (2023b) 
 
Cho-Jui Hsieh, Si Si, Felix X Yu, and Inderjit S Dhillon. 2023b.

 
 Automatic engineering of long prompts.

 
 ArXiv preprint abs/2311.10117 (2023).

 
 
 https://arxiv.org/abs/2311.10117 

 

 
 Hsieh et al . (2023a) 
 
Cheng-Yu Hsieh, Chun-Liang Li, Chih-kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023a.

 
 Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. In Findings of the Association for Computational Linguistics: ACL 2023 , Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 8003–8017.

 
 
 https://doi.org/10.18653/v1/2023.findings-acl.507 

 

 
 Huang et al . (2023a) 
 
Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2023a.

 
 Large Language Models Can Self-Improve. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 1051–1068.

 
 
 https://doi.org/10.18653/v1/2023.emnlp-main.67 

 

 
 Huang et al . (2023b) 
 
Xijie Huang, Li Lyna Zhang, Kwang-Ting Cheng, Fan Yang, and Mao Yang. 2023b.

 
 Fewer is More: Boosting LLM Reasoning with Reinforced Context Pruning.

 
 
 https://api.semanticscholar.org/CorpusID:266210460 

 

 
 Jiang et al . (2023a) 
 
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023a.

 
 LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 13358–13376.

 
 
 https://doi.org/10.18653/v1/2023.emnlp-main.825 

 

 
 Jiang et al . (2023b) 
 
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023b.

 
 LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.

 
 ArXiv preprint abs/2310.06839 (2023).

 
 
 https://arxiv.org/abs/2310.06839 

 

 
 Jiang et al . (2020) 
 
Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. 2020.

 
 How Can We Know What Language Models Know?

 
 Transactions of the Association for Computational Linguistics 8 (2020), 423–438.

 
 
 https://doi.org/10.1162/tacl_a_00324 

 

 
 Jin et al . (2024) 
 
Can Jin, Hongwu Peng, Shiyu Zhao, Zhenting Wang, Wujiang Xu, Ligong Han, Jiahui Zhao, Kai Zhong, Sanguthevar Rajasekaran, and Dimitris N Metaxas. 2024.

 
 APEER: Automatic Prompt Engineering Enhances Large Language Model Reranking.

 
 ArXiv preprint abs/2406.14449 (2024).

 
 
 https://arxiv.org/abs/2406.14449 

 

 
 Juneja et al . (2024) 
 
Gurusha Juneja, Nagarajan Natarajan, Hua Li, Jian Jiao, and Amit Sharma. 2024.

 
 Task Facet Learning: A Structured Approach to Prompt Optimization.

 
 ArXiv preprint abs/2406.10504 (2024).

 
 
 https://arxiv.org/abs/2406.10504 

 

 
 Jung and Kim (2023) 
 
Hoyoun Jung and Kyung-Joong Kim. 2023.

 
 Discrete Prompt Compression With Reinforcement Learning.

 
 IEEE Access 12 (2023), 72578–72587.

 
 
 https://api.semanticscholar.org/CorpusID:261030884 

 

 
 Kojima et al . (2022) 
 
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022.

 
 Large Language Models are Zero-Shot Reasoners. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2022/hash/8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html 

 

 
 Kong et al . (2024) 
 
Weize Kong, Spurthi Amba Hombaiah, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. 2024.

 
 PRewrite: Prompt Rewriting with Reinforcement Learning.

 
 ArXiv preprint abs/2401.08189 (2024).

 
 
 https://arxiv.org/abs/2401.08189 

 

 
 Kumar et al . (2024) 
 
Shanu Kumar, Akhila Yesantarao Venkata, Shubhanshu Khandelwal, Bishal Santra, Parag Agrawal, and Manish Gupta. 2024.

 
 SCULPT: Systematic Tuning of Long Prompts.

 
 ArXiv preprint abs/2410.20788 (2024).

 
 
 https://arxiv.org/abs/2410.20788 

 

 
 Lan et al . (2020) 
 
Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020.

 
 ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net.

 
 
 https://openreview.net/forum?id=H1eA7AEtvS 

 

 
 Lester et al . (2021) 
 
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.

 
 The Power of Scale for Parameter-Efficient Prompt Tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 3045–3059.

 
 
 https://doi.org/10.18653/v1/2021.emnlp-main.243 

 

 
 Li et al . (2023d) 
 
Bei Li, Rui Wang, Junliang Guo, Kaitao Song, Xuejiao Tan, Hany Hassan, Arul Menezes, Tong Xiao, Jiang Bian, and Jingbo Zhu. 2023d.

 
 Deliberate then Generate: Enhanced Prompting Framework for Text Generation.

 
 ArXiv preprint abs/2305.19835 (2023).

 
 
 https://arxiv.org/abs/2305.19835 

 

 
 Li et al . (2023e) 
 
Lei Li, Yongfeng Zhang, and Li Chen. 2023e.

 
 Prompt Distillation for Efficient LLM-based Recommendation.

 
 Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (2023).

 
 
 https://api.semanticscholar.org/CorpusID:264350121 

 

 
 Li et al . (2023c) 
 
Moxin Li, Wenjie Wang, Fuli Feng, Yixin Cao, Jizhi Zhang, and Tat-Seng Chua. 2023c.

 
 Robust Prompt Optimization for Large Language Models Against Distribution Shifts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 1539–1554.

 
 
 https://doi.org/10.18653/v1/2023.emnlp-main.95 

 

 
 Li et al . (2024a) 
 
Xinze Li, Zhenghao Liu, Chenyan Xiong, Shi Yu, Yukun Yan, Shuo Wang, and Ge Yu. 2024a.

 
 Say More with Less: Understanding Prompt Learning Behaviors through Gist Compression.

 
 ArXiv preprint abs/2402.16058 (2024).

 
 
 https://arxiv.org/abs/2402.16058 

 

 
 Li and Liang (2021) 
 
Xiang Lisa Li and Percy Liang. 2021.

 
 Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (Eds.). Association for Computational Linguistics, Online, 4582–4597.

 
 
 https://doi.org/10.18653/v1/2021.acl-long.353 

 

 
 Li et al . (2023a) 
 
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023a.

 
 Compressing Context to Enhance Inference Efficiency of Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 6342–6353.

 
 
 https://doi.org/10.18653/v1/2023.emnlp-main.391 

 

 
 Li et al . (2023b) 
 
Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, and Xifeng Yan. 2023b.

 
 Guiding Large Language Models via Directional Stimulus Prompting. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2023/hash/c5601d99ed028448f29d1dae2e4a926d-Abstract-Conference.html 

 

 
 Li et al . (2024b) 
 
Zongqian Li, Yixuan Su, and Nigel Collier. 2024b.

 
 500xCompressor: Generalized Prompt Compression for Large Language Models.

 
 ArXiv preprint abs/2408.03094 (2024).

 
 
 https://arxiv.org/abs/2408.03094 

 

 
 Lin et al . (2024) 
 
Xiaoqiang Lin, Zhongxiang Dai, Arun Verma, See-Kiong Ng, Patrick Jaillet, and Bryan Kian Hsiang Low. 2024.

 
 Prompt Optimization with Human Feedback.

 
 ArXiv preprint abs/2405.17346 (2024).

 
 
 https://arxiv.org/abs/2405.17346 

 

 
 Liskavets et al . (2024) 
 
Barys Liskavets, Maxim Ushakov, Shuvendu Roy, Mark Klibanov, Ali Etemad, and Shane Luke. 2024.

 
 Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference.

 
 
 https://api.semanticscholar.org/CorpusID:272367247 

 

 
 Liu (2019) 
 
Yinhan Liu. 2019.

 
 Roberta: A robustly optimized bert pretraining approach.

 
 ArXiv preprint abs/1907.11692 (2019).

 
 
 https://arxiv.org/abs/1907.11692 

 

 
 Liu et al . (2019) 
 
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019.

 
 RoBERTa: A Robustly Optimized BERT Pretraining Approach.

 
 ArXiv preprint abs/1907.11692 (2019).

 
 
 https://arxiv.org/abs/1907.11692 

 

 
 Lu et al . (2024) 
 
Junru Lu, Siyu An, Min Zhang, Yulan He, Di Yin, and Xing Sun. 2024.

 
 FIPO: Free-form Instruction-oriented Prompt Optimization with Preference Dataset and Modular Fine-tuning Schema.

 
 ArXiv preprint abs/2402.11811 (2024).

 
 
 https://arxiv.org/abs/2402.11811 

 

 
 Lu et al . (2023) 
 
Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. 2023.

 
 Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net.

 
 
 https://openreview.net/pdf?id=DHyHRBwJUTN 

 

 
 Ma et al . (2024) 
 
Ruotian Ma, Xiaolei Wang, Xin Zhou, Jian Li, Nan Du, Tao Gui, Qi Zhang, and Xuanjing Huang. 2024.

 
 Are Large Language Models Good Prompt Optimizers?

 
 ArXiv preprint abs/2402.02101 (2024).

 
 
 https://arxiv.org/abs/2402.02101 

 

 
 Madaan et al . (2023) 
 
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023.

 
 Self-Refine: Iterative Refinement with Self-Feedback. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html 

 

 
 Mu et al . (2023) 
 
Jesse Mu, Xiang Li, and Noah D. Goodman. 2023.

 
 Learning to Compress Prompts with Gist Tokens. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html 

 

 
 OpenAI (2024a) 
 
OpenAI. 2024a.

 
 Introducing OpenAI o1-preview.

 
 
 https://openai.com/index/introducing-openai-o1-preview/ 

 

 
 OpenAI (2024b) 
 
OpenAI. 2024b.

 
 OpenAI o1-mini.

 
 
 https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/ 

 

 
 Ouyang et al . (2022) 
 
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022.

 
 Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html 

 

 
 Pan et al . (2023) 
 
Rui Pan, Shuo Xing, Shizhe Diao, Xiang Liu, Kashun Shum, Jipeng Zhang, and Tong Zhang. 2023.

 
 Plum: Prompt Learning using Metaheuristic.

 
 ArXiv preprint abs/2311.08364 (2023).

 
 
 https://arxiv.org/abs/2311.08364 

 

 
 Pan et al . (2024) 
 
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang, Karl Cobbe, Vineet Kosaraju, Mo Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, and Reiichiro Nakano. 2024.

 
 LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. In Annual Meeting of the Association for Computational Linguistics .

 
 
 https://api.semanticscholar.org/CorpusID:268531237 

 

 
 Paranjape et al . (2023) 
 
Bhargavi Paranjape, Scott M. Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, and Marco Tulio Ribeiro. 2023.

 
 ART: Automatic multi-step reasoning and tool-use for large language models.

 
 ArXiv preprint abs/2303.09014 (2023).

 
 
 https://arxiv.org/abs/2303.09014 

 

 
 Phang et al . (2023) 
 
Jason Phang, Yi Mao, Pengcheng He, and Weizhu Chen. 2023.

 
 HyperTuning: Toward Adapting Large Language Models without Back-propagation. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA (Proceedings of Machine Learning Research, Vol. 202) , Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 27854–27875.

 
 
 https://proceedings.mlr.press/v202/phang23a.html 

 

 
 Pitis et al . (2023) 
 
Silviu Pitis, Michael R Zhang, Andrew Wang, and Jimmy Ba. 2023.

 
 Boosted prompt ensembles for large language models.

 
 ArXiv preprint abs/2304.05970 (2023).

 
 
 https://arxiv.org/abs/2304.05970 

 

 
 Prasad et al . (2023) 
 
Archiki Prasad, Peter Hase, Xiang Zhou, and Mohit Bansal. 2023.

 
 GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics , Andreas Vlachos and Isabelle Augenstein (Eds.). Association for Computational Linguistics, Dubrovnik, Croatia, 3845–3864.

 
 
 https://doi.org/10.18653/v1/2023.eacl-main.277 

 

 
 Press et al . (2023) 
 
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. 2023.

 
 Measuring and Narrowing the Compositionality Gap in Language Models. In Findings of the Association for Computational Linguistics: EMNLP 2023 , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 5687–5711.

 
 
 https://doi.org/10.18653/v1/2023.findings-emnlp.378 

 

 
 Pryzant et al . (2023) 
 
Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. 2023.

 
 Automatic Prompt Optimization with “Gradient Descent” and Beam Search. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 7957–7968.

 
 
 https://doi.org/10.18653/v1/2023.emnlp-main.494 

 

 
 Pu et al . (2024) 
 
Xiao Pu, Tianxing He, and Xiaojun Wan. 2024.

 
 Style-Compress: An LLM-Based Prompt Compression Framework Considering Task-Specific Styles. In Findings of the Association for Computational Linguistics: EMNLP 2024 . 14533–14549.

 
 
 

 
 Qin et al . (2023) 
 
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al . 2023.

 
 Toolllm: Facilitating large language models to master 16000+ real-world apis.

 
 ArXiv preprint abs/2307.16789 (2023).

 
 
 https://arxiv.org/abs/2307.16789 

 

 
 Radford et al . (2021) 
 
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021.

 
 Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (Proceedings of Machine Learning Research, Vol. 139) , Marina Meila and Tong Zhang (Eds.). PMLR, 8748–8763.

 
 
 http://proceedings.mlr.press/v139/radford21a.html 

 

 
 Radford and Narasimhan (2018) 
 
Alec Radford and Karthik Narasimhan. 2018.

 
 Improving Language Understanding by Generative Pre-Training.

 
 
 https://api.semanticscholar.org/CorpusID:49313245 

 

 
 Radford et al . (2019) 
 
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019.

 
 Language Models are Unsupervised Multitask Learners.

 
 
 https://api.semanticscholar.org/CorpusID:160025533 

 

 
 Rafailov et al . (2023) 
 
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023.

 
 Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html 

 

 
 Raffel et al . (2020) 
 
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020.

 
 Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.

 
 J. Mach. Learn. Res. 21 (2020), 140:1–140:67.

 
 
 http://jmlr.org/papers/v21/20-074.html 

 

 
 Reimers and Gurevych (2019) 
 
Nils Reimers and Iryna Gurevych. 2019.

 
 Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 3982–3992.

 
 
 https://doi.org/10.18653/v1/D19-1410 

 

 
 Schulman et al . (2017) 
 
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017.

 
 Proximal Policy Optimization Algorithms.

 
 ArXiv preprint abs/1707.06347 (2017).

 
 
 https://arxiv.org/abs/1707.06347 

 

 
 Sclar et al . (2023) 
 
Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2023.

 
 Quantifying Language Models’ Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting.

 
 ArXiv preprint abs/2310.11324 (2023).

 
 
 https://arxiv.org/abs/2310.11324 

 

 
 Shannon (1948) 
 
Claude E. Shannon. 1948.

 
 A mathematical theory of communication.

 
 Bell Syst. Tech. J. 27 (1948), 623–656.

 
 
 https://api.semanticscholar.org/CorpusID:55379485 

 

 
 Shi et al . (2024) 
 
Zhengliang Shi, Shen Gao, Xiuyi Chen, Yue Feng, Lingyong Yan, Haibo Shi, Dawei Yin, Zhumin Chen, Suzan Verberne, and Zhaochun Ren. 2024.

 
 Chain of Tools: Large Language Model is an Automatic Multi-tool Learner.

 
 ArXiv preprint abs/2405.16533 (2024).

 
 
 https://arxiv.org/abs/2405.16533 

 

 
 Shinn et al . (2023) 
 
Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023.

 
 Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html 

 

 
 Snell et al . (2022) 
 
Charles Burton Snell, Dan Klein, and Ruiqi Zhong. 2022.

 
 Learning by Distilling Context.

 
 ArXiv preprint abs/2209.15189 (2022).

 
 
 https://arxiv.org/abs/2209.15189 

 

 
 Sun et al . (2023b) 
 
Hao Sun, Alihan Hüyük, and Mihaela van der Schaar. 2023b.

 
 Query-dependent prompt evaluation and optimization with offline inverse RL. In The Twelfth International Conference on Learning Representations .

 
 
 

 
 Sun et al . (2023c) 
 
Hong Sun, Xue Li, Yinchuan Xu, Youkow Homma, Qi Cao, Min Wu, Jian Jiao, and Denis Charles. 2023c.

 
 Autohint: Automatic prompt optimization with hint generation.

 
 ArXiv preprint abs/2307.07415 (2023).

 
 
 https://arxiv.org/abs/2307.07415 

 

 
 Sun et al . (2023a) 
 
Weiwei Sun, Zheng Chen, Xinyu Ma, Lingyong Yan, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023a.

 
 Instruction Distillation Makes Large Language Models Efficient Zero-shot Rankers.

 
 ArXiv preprint abs/2311.01555 (2023).

 
 
 https://arxiv.org/abs/2311.01555 

 

 
 Suzgun et al . (2023) 
 
Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023.

 
 Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. In Findings of the Association for Computational Linguistics: ACL 2023 , Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 13003–13051.

 
 
 https://doi.org/10.18653/v1/2023.findings-acl.824 

 

 
 Tan et al . (2024) 
 
Sijun Tan, Xiuyu Li, Shishir G. Patil, Ziyang Wu, Tianjun Zhang, Kurt Keutzer, Joseph E. Gonzalez, and Raluca A. Popa. 2024.

 
 LLoCO: Learning Long Contexts Offline.

 
 ArXiv preprint abs/2404.07979 (2024).

 
 
 https://arxiv.org/abs/2404.07979 

 

 
 Tang et al . (2024) 
 
Xinyu Tang, Xiaolei Wang, Wayne Xin Zhao, Siyuan Lu, Yaliang Li, and Ji-Rong Wen. 2024.

 
 Unleashing the Potential of Large Language Models as Prompt Optimizers: An Analogical Analysis with Gradient-based Model Optimizers.

 
 ArXiv preprint abs/2402.17564 (2024).

 
 
 https://arxiv.org/abs/2402.17564 

 

 
 Touvron et al . (2023) 
 
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al . 2023.

 
 Llama 2: Open foundation and fine-tuned chat models.

 
 ArXiv preprint abs/2307.09288 (2023).

 
 
 https://arxiv.org/abs/2307.09288 

 

 
 Vaswani et al . (2017) 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.

 
 Attention is All you Need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA , Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (Eds.). 5998–6008.

 
 
 https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html 

 

 
 Wan et al . (2023a) 
 
Xingchen Wan, Ruoxi Sun, Hanjun Dai, Sercan Arik, and Tomas Pfister. 2023a.

 
 Better Zero-Shot Reasoning with Self-Adaptive Prompting. In Findings of the Association for Computational Linguistics: ACL 2023 , Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 3493–3514.

 
 
 https://doi.org/10.18653/v1/2023.findings-acl.216 

 

 
 Wan et al . (2023b) 
 
Xingchen Wan, Ruoxi Sun, Hootan Nakhost, Hanjun Dai, Julian Eisenschlos, Sercan Arik, and Tomas Pfister. 2023b.

 
 Universal Self-Adaptive Prompting. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 7437–7462.

 
 
 https://doi.org/10.18653/v1/2023.emnlp-main.461 

 

 
 Wang et al . (2024a) 
 
Ruochen Wang, Sohyun An, Minhao Cheng, Tianyi Zhou, Sung Ju Hwang, and Cho-Jui Hsieh. 2024a.

 
 One Prompt is not Enough: Automated Construction of a Mixture-of-Expert Prompts.

 
 ArXiv preprint abs/2407.00256 (2024).

 
 
 https://arxiv.org/abs/2407.00256 

 

 
 Wang et al . (2024b) 
 
Xinfeng Wang, Jin Cui, Yoshimi Suzuki, and Fumiyo Fukumoto. 2024b.

 
 RDRec: Rationale Distillation for LLM-based Recommendation. In Annual Meeting of the Association for Computational Linguistics .

 
 
 https://api.semanticscholar.org/CorpusID:269899491 

 

 
 Wang et al . (2023c) 
 
Xinyuan Wang, Chenxi Li, Zhen Wang, Fan Bai, Haotian Luo, Jiayou Zhang, Nebojsa Jojic, Eric P. Xing, and Zhiting Hu. 2023c.

 
 PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization.

 
 ArXiv preprint abs/2310.16427 (2023).

 
 
 https://arxiv.org/abs/2310.16427 

 

 
 Wang et al . (2023d) 
 
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023d.

 
 Self-Consistency Improves Chain of Thought Reasoning in Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net.

 
 
 https://openreview.net/pdf?id=1PL1NIMMrw 

 

 
 Wang et al . (2023b) 
 
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023b.

 
 Self-Instruct: Aligning Language Models with Self-Generated Instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 13484–13508.

 
 
 https://doi.org/10.18653/v1/2023.acl-long.754 

 

 
 Wang et al . (2023a) 
 
Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md. Rizwan Parvez, and Graham Neubig. 2023a.

 
 Learning to Filter Context for Retrieval-Augmented Generation.

 
 ArXiv preprint abs/2311.08377 (2023).

 
 
 https://arxiv.org/abs/2311.08377 

 

 
 Wei et al . (2022a) 
 
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed Huai hsin Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022a.

 
 Emergent Abilities of Large Language Models.

 
 ArXiv preprint abs/2206.07682 (2022).

 
 
 https://arxiv.org/abs/2206.07682 

 

 
 Wei et al . (2022b) 
 
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022b.

 
 Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html 

 

 
 Wingate et al . (2022) 
 
David Wingate, Mohammad Shoeybi, and Taylor Sorensen. 2022.

 
 Prompt Compression and Contrastive Conditioning for Controllability and Toxicity Reduction in Language Models. In Findings of the Association for Computational Linguistics: EMNLP 2022 , Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 5621–5634.

 
 
 https://doi.org/10.18653/v1/2022.findings-emnlp.412 

 

 
 Xu et al . (2023b) 
 
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023b.

 
 RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation.

 
 ArXiv preprint abs/2310.04408 (2023).

 
 
 https://arxiv.org/abs/2310.04408 

 

 
 Xu et al . (2022) 
 
Hanwei Xu, Yujun Chen, Yulun Du, Nan Shao, Wang Yanggang, Haiyu Li, and Zhilin Yang. 2022.

 
 GPS: Genetic Prompt Search for Efficient Few-Shot Learning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 8162–8171.

 
 
 https://doi.org/10.18653/v1/2022.emnlp-main.559 

 

 
 Xu et al . (2023a) 
 
Weijia Xu, Andrzej Banburski-Fahey, and Nebojsa Jojic. 2023a.

 
 Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling.

 
 ArXiv preprint abs/2305.09993 (2023).

 
 
 https://arxiv.org/abs/2305.09993 

 

 
 Yang et al . (2023) 
 
Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. 2023.

 
 Large Language Models as Optimizers.

 
 ArXiv preprint abs/2309.03409 (2023).

 
 
 https://arxiv.org/abs/2309.03409 

 

 
 Yang et al . (2024) 
 
Sheng Yang, Yurong Wu, Yan Gao, Zineng Zhou, Bin Benjamin Zhu, Xiaodi Sun, Jian-Guang Lou, Zhiming Ding, Anbang Hu, Yuan Fang, et al . 2024.

 
 AMPO: Automatic Multi-Branched Prompt Optimization.

 
 ArXiv preprint abs/2410.08696 (2024).

 
 
 https://arxiv.org/abs/2410.08696 

 

 
 Yao et al . (2023a) 
 
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a.

 
 Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).

 
 
 http://papers.nips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html 

 

 
 Yao et al . (2023b) 
 
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023b.

 
 ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net.

 
 
 https://openreview.net/pdf?id=WE_vluYUL-X 

 

 
 Ye et al . (2023) 
 
Qinyuan Ye, Maxamed Axmed, Reid Pryzant, and Fereshte Khani. 2023.

 
 Prompt engineering a prompt engineer.

 
 ArXiv preprint abs/2311.05661 (2023).

 
 
 https://arxiv.org/abs/2311.05661 

 

 
 Yoon et al . (2024) 
 
Chanwoong Yoon, Taewhoo Lee, Hyeon Hwang, Minbyul Jeong, and Jaewoo Kang. 2024.

 
 CompAct: Compressing Retrieved Documents Actively for Question Answering.

 
 ArXiv preprint abs/2407.09014 (2024).

 
 
 https://arxiv.org/abs/2407.09014 

 

 
 Zhang et al . (2020) 
 
Biao Zhang, Ivan Titov, and Rico Sennrich. 2020.

 
 Fast Interleaved Bidirectional Sequence Generation. In Proceedings of the Fifth Conference on Machine Translation , Loïc Barrault, Ondřej Bojar, Fethi Bougares, Rajen Chatterjee, Marta R. Costa-jussà, Christian Federmann, Mark Fishel, Alexander Fraser, Yvette Graham, Paco Guzman, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, André Martins, Makoto Morishita, Christof Monz, Masaaki Nagata, Toshiaki Nakazawa, and Matteo Negri (Eds.). Association for Computational Linguistics, Online,
503–515.

 
 
 https://aclanthology.org/2020.wmt-1.62 

 

 
 Zhang et al . (2024b) 
 
Chenrui Zhang, Lin Liu, Chuyuan Wang, Xiao Sun, Hongyu Wang, Jinpeng Wang, and Mingchen Cai. 2024b.

 
 PREFER: Prompt Ensemble Learning via Feedback-Reflect-Refine. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada , Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (Eds.). AAAI Press, 19525–19532.

 
 
 https://doi.org/10.1609/AAAI.V38I17.29924 

 

 
 Zhang et al . (2024a) 
 
Lechen Zhang, Tolga Ergen, Lajanugen Logeswaran, Moontae Lee, and David Jurgens. 2024a.

 
 SPRIG: Improving Large Language Model Performance by System Prompt Optimization.

 
 ArXiv preprint abs/2410.14826 (2024).

 
 
 https://arxiv.org/abs/2410.14826 

 

 
 Zhang et al . (2024c) 
 
Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. 2024c.

 
 Compressing Lengthy Context With UltraGist.

 
 ArXiv preprint abs/2405.16635 (2024).

 
 
 https://arxiv.org/abs/2405.16635 

 

 
 Zhang et al . (2024e) 
 
Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng. 2024e.

 
 AdaComp: Extractive Context Compression with Adaptive Predictor for Retrieval-Augmented Large Language Models.

 
 ArXiv preprint abs/2409.01579 (2024).

 
 
 https://arxiv.org/abs/2409.01579 

 

 
 Zhang et al . (2022) 
 
Tianjun Zhang, Xuezhi Wang, Denny Zhou, Dale Schuurmans, and Joseph E. Gonzalez. 2022.

 
 TEMPERA: Test-Time Prompting via Reinforcement Learning.

 
 ArXiv preprint abs/2211.11890 (2022).

 
 
 https://arxiv.org/abs/2211.11890 

 

 
 Zhang et al . (2024d) 
 
Tuo Zhang, Jinyue Yuan, and Amir Salman Avestimehr. 2024d.

 
 Revisiting OPRO: The Limitations of Small-Scale LLMs as Optimizers. In Annual Meeting of the Association for Computational Linguistics .

 
 
 https://api.semanticscholar.org/CorpusID:269791383 

 

 
 Zhang et al . (2023) 
 
Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2023.

 
 Automatic Chain of Thought Prompting in Large Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net.

 
 
 https://openreview.net/pdf?id=5NTt8GFjUHkr 

 

 
 Zhao et al . (2023) 
 
Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei Qin, and Lidong Bing. 2023.

 
 Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 5823–5840.

 
 
 https://doi.org/10.18653/v1/2023.acl-long.320 

 

 
 Zhao et al . (2021) 
 
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021.

 
 Calibrate Before Use: Improving Few-shot Performance of Language Models. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (Proceedings of Machine Learning Research, Vol. 139) , Marina Meila and Tong Zhang (Eds.). PMLR, 12697–12706.

 
 
 http://proceedings.mlr.press/v139/zhao21c.html 

 

 
 Zhou et al . (2023a) 
 
Wangchunshu Zhou, Yuchen Jiang, Ryan Cotterell, and Mrinmaya Sachan. 2023a.

 
 Efficient Prompting via Dynamic In-Context Learning.

 
 ArXiv preprint abs/2305.11170 (2023).

 
 
 https://arxiv.org/abs/2305.11170 

 

 
 Zhou et al . (2023b) 
 
Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2023b.

 
 Large Language Models are Human-Level Prompt Engineers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net.

 
 
 https://openreview.net/pdf?id=92gvk82DE- 

 

 
 Zou et al . (2023) 
 
Anni Zou, Zhuosheng Zhang, Hai Zhao, and Xiangru Tang. 2023.

 
 Meta-CoT: Generalizable Chain-of-Thought Prompting in Mixed-task Scenarios with Large Language Models.

 
 ArXiv preprint abs/2310.06692 (2023).

 
 
 https://arxiv.org/abs/2310.06692 

 

 
 
 
 

## Appendix A Appendix

 

### A.1. Open Resources

 
 Table 5. Open resources of efficient prompting methods in Prompt Compression 
 
 
 Method | 
 
 
 Link 
 | 

 
 Prompt Injection | 
 
 
 https://github.com/unbiarirang/Fixed-Input-Parameterization 
 | 

 
 Distilling Context | 
 
 
 https://en.wikipedia.org/wiki/Declarative_learning 
 | 

 
 Instrcution Distillation | 
 
 
 www.github.com/sunnweiwei/RankGPT 
 | 

 
 xRAG | 
 
 
 https://github.com/Hannibal046/xRAG 
 | 

 
 Gisting | 
 
 
 https://github.com/jayelm/gisting 
 | 

 
 UltraGist | 
 
 
 https://github.com/namespace-Pt/UltraGist 
 | 

 
 AutoCompressor | 
 
 
 https://github.com/princeton-nlp/AutoCompressors 
 | 

 
 LLoCO | 
 
 
 https://github.com/jeffreysijuntan/lloco 
 | 

 
 ICAE | 
 
 
 https://github.com/getao/icae 
 | 

 
 SelfCP | 
 
 
 https://github.com/jungao1106/SelfCP 
 | 

 
 500xCompressor | 
 
 
 https://github.com/ZongqianLi/500xCompressor 
 | 

 
 POD | 
 
 
 https://github.com/lileipisces/POD 
 | 

 
 RDRec | 
 
 
 https://github.com/WangXFng/RDRec 
 | 

 
 FilCo | 
 
 
 https://github.com/zorazrw/filco 
 | 

 
 CPC | 
 
 
 https://github.com/Workday/cpc 
 | 

 
 AdaComp | 
 
 
 https://anonymous.4open.science/r/AdaComp-8C0C/ 
 | 

 
 LLMLingua | 
 
 
 https://huggingface.co/spaces/microsoft/LLMLingua 
 | 

 
 LongLLMLingua | 
 
 
 https://aka.ms/LongLLMLingua 
 | 

 
 Selective Context | 
 
 
 https://github.com/liyucheng09/Selective_Context 
 | 

 
 PCRL | 
 
 
 https://github.com/nenomigami/PromptCompressor 
 | 

 
 LLMLingua-2 | 
 
 
 https://aka.ms/LLMLingua-2 
 | 

 
 RECOMP | 
 
 
 https://github.com/carriex/recomp 
 | 

 
 CompAct | 
 
 
 https://github.com/dmis-lab/CompAct 
 | 

 
 
 Table 6. Open resources of efficient prompting methods in Automatic Prompt Engineering 
 
 
 Method | 
 
 
 Link 
 | 

 
 APE | 
 
 
 https://github.com/keirp/automatic_prompt_engineer 
 | 

 
 OPRO | 
 
 
 https://github.com/google-deepmind/opro 
 | 

 
 PromptAgent | 
 
 
 https://github.com/XinyuanWangCS/PromptAgent 
 | 

 
 EvoPrompt | 
 
 
 https://github.com/beeevita/EvoPrompt 
 | 

 
 RLPrompt | 
 
 
 https://github.com/mingkaid/rl-prompt 
 | 

 
 DSP | 
 
 
 https://github.com/Leezekun/Directional-Stimulus-Prompting 
 | 

 
 ProTeGi | 
 
 
 https://github.com/microsoft/LMOps/tree/main/prompt_optimization 
 | 

 
 PE2 | 
 
 
 https://www.promptingguide.ai/introduction 
 | 

 
 PREFER | 
 
 
 https://github.com/zcrwind/PREFER 
 | 

 
 UniPrompt | 
 
 
 https://aka.ms/uniprompt 
 | 

 
 APOHF | 
 
 
 https://github.com/xqlin98/APOHF 
 | 

 
 BPO | 
 
 
 https://github.com/thu-coai/BPO 
 | 

 
 APEER | 
 
 
 https://github.com/jincan333/APEER 
 | 

 
 FIPO | 
 
 
 https://github.com/LuJunru/FIPO_Project 
 | 

 
 GrIPS | 
 
 
 https://github.com/archiki/GrIPS 
 | 

 
 Plum | 
 
 
 https://github.com/research4pan/Plum 
 | 

 
 TEMPERA | 
 
 
 https://github.com/tianjunz/TEMPERA 
 | 

 
 Auto-CoT | 
 
 
 https://github.com/amazon-research/auto-cot 
 | 

 
 Boosted Prompting | 
 
 
 https://github.com/awwang10/llmpromptboosting 
 | 

 
 Self-refine | 
 
 
 https://selfrefine.info/ 
 | 

 
 PromptPG | 
 
 
 https://promptpg.github.io/ 
 | 

 
 Prompt-OIRL | 
 
 
 https://github.com/holarissun/Prompt-OIRL 
 | 

 
 Reflection | 
 
 
 https://github.com/noahshinn024/reflexion 
 | 

 
 PROMST | 
 
 
 https://yongchao98.github.io/MIT-REALM-PROMST/ 
 | 

 
 ReAct | 
 
 
 https://react-lm.github.io/ 
 | 

 
 Verify-and-Edit | 
 
 
 https://github.com/RuochenZhao/Verify-and-Edit 
 | 

 
 ART | 
 
 
 https://github.com/bhargaviparanjape/language-programmes/ 
 | 

 
 Self-ask | 
 
 
 https://github.com/ofirpress/self-ask 
 | 

 
 ToolLLM | 
 
 
 https://github.com/OpenBMB/ToolBench 
 |