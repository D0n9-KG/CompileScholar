# LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning

$^{\dagger}$ Bo Liu, $^{*}$ $^{\dagger}$ Yifeng Zhu $^{*}$ , $^{\ddagger}$ Chongkai Gao $^{*}$ , $^{\dagger}$ Yihao Feng $^{\dagger}$ Qiang Liu, $^{\dagger}$ Yuke Zhu, $^{\dagger,\S}$ Peter Stone

$^{\dagger}$ The University of Texas at Austin, $^{\S}$ Sony AI, $^{\ddagger}$ Tsinghua University {bliu,yifengz,lqiang,yukez,pstone}@cs.utexas.edu yihao.ac@gmail.com, gck20@mails.tsinghua.edu.cn

# Abstract

Lifelong learning offers a promising paradigm of building a generalist agent that learns and adapts over its lifespan. Unlike traditional lifelong learning problems in image and text domains, which primarily involve the transfer of declarative knowledge of entities and concepts, lifelong learning in decision-making (LLDM) also necessitates the transfer of procedural knowledge, such as actions and behaviors. To advance research in LLDM, we introduce LIBERO, a novel benchmark of lifelong learning for robot manipulation. Specifically, LIBERO highlights five key research topics in LLDM: 1) how to efficiently transfer declarative knowledge, procedural knowledge, or the mixture of both; 2) how to design effective policy architectures and 3) effective algorithms for LLDM; 4) the robustness of a lifelong learner with respect to task ordering; and 5) the effect of model pretraining for LLDM. We develop an extendible procedural generation pipeline that can in principle generate infinitely many tasks. For benchmarking purpose, we create four task suites (130 tasks in total) that we use to investigate the above-mentioned research topics. To support sample-efficient learning, we provide high-quality human-teleoperated demonstration data for all tasks. Our extensive experiments present several insightful or even unexpected discoveries: sequential finetuning outperforms existing lifelong learning methods in forward transfer, no single visual encoder architecture excels at all types of knowledge transfer, and naive supervised pretraining can hinder agents' performance in the subsequent LLDM. $^{2}$

# 1 Introduction

A longstanding goal in machine learning is to develop a generalist agent that can perform a wide range of tasks. While multitask learning [10] is one approach, it is computationally demanding and not adaptable to ongoing changes. Lifelong learning [65], however, offers a practical solution by amortizing the learning process over the agent's lifespan. Its goal is to leverage prior knowledge to facilitate learning new tasks (forward transfer) and use the newly acquired knowledge to enhance performance on prior tasks (backward transfer).

The main body of the lifelong learning literature has focused on how agents transfer declarative knowledge in visual or language tasks, which pertains to declarative knowledge about entities and concepts $[7, 40]$ . Yet it is understudied how agents transfer knowledge in decision-making tasks, which involves a mixture of both declarative and procedural knowledge (knowledge about how to do something). Consider a scenario where a robot, initially trained to retrieve juice from a fridge, fails

![](images/09f16f44844b8839ce34c0555f2b1a7ef52d321d094b451316ad16f32fd005c4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Procedural Generation"] --> B["LIBERO-Object"]
    A --> C["LIBERO-Goal"]
    A --> D["LIBERO-Spatial"]
    A --> E["LIBERO-100"]
    B --> F["Different layouts, same objects"]
    C --> G["Different goals, same objects & layout"]
    D --> H["Diverse objects, layouts, backgrounds"]
    E --> I["Distribution Shifts"]
    E --> J["Algorithmic Designs"]
    E --> K["Neural Architectures"]
    E --> L["Task Orderings"]
    E --> M["Pretraining Effects"]
```
</details>

Figure 1: Top: LIBERO has four procedurally-generated task suites: LIBERO-SPATIAL, LIBERO-OBJECT, and LIBERO-GOAL have 10 tasks each and require transferring knowledge about spatial relationships, objects, and task goals; LIBERO-100 has 100 tasks and requires the transfer of entangled knowledge. Bottom: we investigate five key research topics in LLDM on LIBERO.

after learning new tasks. This could be due to forgetting the juice or fridge's location (declarative knowledge) or how to open the fridge or grasp the juice (procedural knowledge). So far, we lack methods to systematically and quantitatively analyze this complex knowledge transfer.

To bridge this research gap, this paper introduces a new simulation benchmark, LIfelong learning BEchmark on RObot manipulation tasks, LIBERO, to facilitate the systematic study of lifelong learning in decision making (LLDM). An ideal LLDM testbed should enable continuous learning across an expanding set of diverse tasks that share concepts and actions. LIBERO supports this through a procedural generation pipeline for endless task creation, based on robot manipulation tasks with shared visual concepts (declarative knowledge) and interactions (procedural knowledge).

For benchmarking purpose, LIBERO generates 130 language-conditioned robot manipulation tasks inspired by human activities $[22]$ and, grouped into four suites. The four task suites are designed to examine distribution shifts in the object types, the spatial arrangement of objects, the task goals, or the mixture of the previous three (top row of Figure 1). LIBERO is scalable, extendable, and designed explicitly for studying lifelong learning in robot manipulation. To support efficient learning, we provide high-quality, human-teleoperated demonstration data for all 130 tasks.

We present an initial study using LIBERO to investigate five major research topics in LLDM (Figure 1): 1) knowledge transfer with different types of distribution shift; 2) neural architecture design; 3) lifelong learning algorithm design; 4) robustness of the learner to task ordering; and 5) how to leverage pre-trained models in LLDM (bottom row of Figure 1). We perform extensive experiments across different policy architectures and different lifelong learning algorithms. Based on our experiments, we make several insightful or even unexpected observations:

1. Policy architecture design is as crucial as lifelong learning algorithms. The transformer architecture is better at abstracting temporal information than a recurrent neural network. Vision transformers work well on tasks with rich visual information (e.g., a variety of objects). Convolution networks work well when tasks primarily need procedural knowledge.   
2. While the lifelong learning algorithms we evaluated are effective at preventing forgetting, they generally perform worse than sequential finetuning in terms of forward transfer.   
3. Our experiment shows that using pretrained language embeddings of semantically-rich task descriptions yields performance no better than using those of the task IDs.

4. Basic supervised pretraining on a large-scale offline dataset can have a negative impact on the learner's downstream performance in LLDM.

# 2 Background

This section introduces the problem formulation and defines key terms used throughout the paper.

# 2.1 Markov Decision Process for Robot Learning

A robot learning problem can be formulated as a finite-horizon Markov Decision Process: $\mathcal{M} = (\mathcal{S},\mathcal{A},\mathcal{T},H,\mu_0,R)$ . Here, $\mathcal{S}$ and $\mathcal{A}$ are the state and action spaces of the robot. $\mu_0$ is the initial state distribution, $R:\mathcal{S}\times \mathcal{A}\to \mathbb{R}$ is the reward function, and $\mathcal{T}:\mathcal{S}\times \mathcal{A}\rightarrow \mathcal{S}$ is the transition function. In this work, we assume a sparse-reward setting and replace $R$ with a goal predicate $g:\mathcal{S}\to \{0,1\}$ . The robot's objective is to learn a policy $\pi$ that maximizes the expected return: $\max_{\pi}J(\pi) = \mathbb{E}_{s_t,a_t\sim \pi ,\mu_0}[\sum_{t = 1}^H g(s_t)]$ .

# 2.2 Lifelong Robot Learning Problem

In a lifelong robot learning problem, a robot sequentially learns over K tasks $\{T^{1},\ldots,T^{K}\}$ with a single policy $\pi$ . We assume $\pi$ is conditioned on the task, i.e., $\pi(\cdot\mid s;T)$ . For each task, $T^{k}\equiv(\mu_{0}^{k},g^{k})$ is defined by the initial state distribution $\mu_{0}^{k}$ and the goal predicate $g^{k}$ . We assume S, A, T, H are the same for all tasks. Up to the k-th task $T^{k}$ , the robot aims to optimize

$$
\max _ {\pi} J _ {\mathrm{LRL}} (\pi) = \frac {1}{k} \sum_ {p = 1} ^ {k} \left[ \underset {s _ {t} ^ {p}, a _ {t} ^ {p} \sim \pi (\cdot ; T ^ {p}), \mu_ {0} ^ {p}} {\mathbb {E}} \left[ \sum_ {t = 1} ^ {L} g ^ {p} (s _ {t} ^ {p}) \right] \right]. \tag {1}
$$

An important feature of the lifelong setting is that the agent loses access to the previous $k - 1$ tasks when it learns on task $T^k$ .

Lifelong Imitation Learning Due to the challenge of sparse-reward reinforcement learning, we consider a practical alternative setting where a user would provide a small demonstration dataset for each task in the sequence. Denote $D^k = \{\tau_i^k\}_{i=1}^N$ as $N$ demonstrations for task $T^k$ . Each $\tau_i^k = (o_0, a_0, o_1, a_1, \ldots, o_{l^k})$ where $l^k \leq H$ . Here, $o_t$ is the robot's sensory input, including the perceptual observation and the information about the robot's joints and gripper. In practice, the observation $o_t$ is often non-Markovian. Therefore, following works in partially observable MDPs [25], we represent $s_t$ by the aggregated history of observations, i.e. $s_t \equiv o_{\leq t} \triangleq (o_0, o_1, \ldots, o_t)$ . This results in the lifelong imitation learning problem with the same objective as in Eq. (1). But during training, we perform behavioral cloning [4] with the following surrogate objective function:

$$
\min _ {\pi} J _ {\mathrm{BC}} (\pi) = \frac {1}{k} \sum_ {p = 1} ^ {k} \underset {o _ {t}, a _ {t} \sim D ^ {p}} {\mathbb {E}} \left[ \sum_ {t = 0} ^ {l ^ {p}} \mathcal {L} \left(\pi (o _ {\leq t}; T ^ {p}), a _ {t} ^ {p}\right) \right], \tag {2}
$$

where L is a supervised learning loss, e.g., the negative log-likelihood loss, and $\pi$ is a Gaussian mixture model. Similarly, we assume $\{D^{p}: p < k\}$ are not fully available when learning $T^{k}$ .

# 3 Research Topics in LLDM

We outline five major research topics in LLDM that motivate the design of LIBERO and our study.

(T1) Transfer of Different Types of Knowledge In order to accomplish a task such as put the ketchup next to the plate in the basket, a robot must understand the concept ketchup, the location of the plate/basket, and how to put the ketchup in the basket. Indeed, robot manipulation tasks in general necessitate different types of knowledge, making it hard to determine the cause of failure. We present four task suites in Section 4.2: three task suites for studying the transfer of knowledge about spatial relationships, object concepts, and task goals in a disentangled manner, and one suite for studying the transfer of mixed types of knowledge.

![](images/3ea419387486297242db321096afb1b5f2ee4603dcf95920febcf533d032431a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Scenes"] --> B["Human activity dataset"]
    B --> C["(1) open... put... in... turn on..."]
    C --> D["(2) &quot;Open the top drawer of the cabinet and put the bowl in it&quot;"]
    D --> E["(3) *.pddl"]
    E --> F["Specify Objects And Layouts"]
    F --> G["(:object black_bowl_1 - black_bowl plate_1 - plate)<br>    (:fixture kitchen_table - kitchen_table cabinet_1 - wooden_cabinet)<br>    (:regions bowl_region<br>    (:target kitchen_table)<br>    (-0.02 -0.02 0.02 0.02)<br>    )"]
    F --> H["(A) Init Configuration"]
    H --> I["(B) Specify Init Configuration"]
    I --> J["(:init (On cabinet_1 cabinet_region)<br>    (On plate_1 plate_init_region)<br>    (On bowl_1 bowl_init_region)"]
    I --> K["(C) Specify Task Goals"]
```
</details>

Figure 2: LIBERO's procedural generation pipeline: Extracting behavioral templates from a large-scale human activity dataset (1), Ego4D, for generating task instructions (2); Based on the task description, selecting the scene and generating the PDDL description file (3) that specifies the objects and layouts (A), the initial object configurations (B), and the task goal (C).

(T2) Neural Architecture Design An important research question in LLDM is how to design effective neural architectures to abstract the multi-modal observations (images, language descriptions, and robot states) and transfer only relevant knowledge when learning new tasks.   
(T3) Lifelong Learning Algorithm Design Given a policy architecture, it is crucial to determine what learning algorithms to apply for LLDM. Specifically, the sequential nature of LLDM suggests that even minor forgetting over successive steps can potentially lead to a total failure in execution. As such, we consider the design of lifelong learning algorithms to be an open area of research in LLDM.   
(T4) Robustness to Task Ordering It is well-known that task curriculum influences policy learning [6, 48]. A robot in the real world, however, often cannot choose which task to encounter first. Therefore, a good lifelong learning algorithm should be robust to different task orderings.   
(T5) Usage of Pretrained Models In practice, robots will be most likely pretrained on large datasets in factories before deployment [28]. However, it is not well-understood whether or how pretraining could benefit subsequent LLDM.

# 4 LIBERO

This section introduces the components in LIBERO: the procedural generation pipeline that allows the never-ending creation of tasks (Section 4.1), the four task suites we generate for benchmarking (Section 4.2), five algorithms (Section 4.3), and three neural architectures (Section 4.4).

# 4.1 Procedural Generation of Tasks

Research in LLVM requires a systematic way to create new tasks while maintaining task diversity and relevance to existing tasks. LIBERO procedurally generates new tasks in three steps: 1) extract behavioral templates from language annotations of human activities and generate sampled tasks described in natural language based on such templates; 2) specify an initial object distribution given a task description; and 3) specify task goals using a propositional formula that aligns with the language instructions. Our generation pipeline is built on top of Robosuite [76], a modular manipulation simulator that offers seamless integration. Figure 2 illustrates an example of task creation using this pipeline, and each component is expanded upon below.

Behavioral Templates and Instruction Generation Human activities serve as a fertile source of tasks that can inspire and generate a vast number of manipulation tasks. We choose a large-scale activity dataset, Ego4D [22], which includes a large variety of everyday activities with language annotations. We pre-process the dataset by extracting the language descriptions and then summarize them into a large set of commonly used language templates. After this pre-processing step, we use the templates and select objects available in the simulator to generate a set of task descriptions in the

form of language instructions. For example, we can generate an instruction “Open the drawer of the cabinet” from the template “Open ...”.

Initial State Distribution $(\mu_{0})$ To specify $\mu_{0}$ , we first sample a scene layout that matches the objects/behaviors in a provided instruction. For instance, a kitchen scene is selected for an instruction Open the top drawer of the cabinet and put the bowl in it. Then, the details about $\mu_{0}$ are generated in the PDDL language [43, 63]. Concretely, $\mu_{0}$ contains information about object categories and their placement (Figure 2-(A)), and their initial status (Figure 2-(B)).

Goal Specifications (g) Based on $\mu_{0}$ and the language instruction, we specify the task goal using a conjunction of predicates. Predicates include unary predicates that describe the properties of an object, such as Open(X) or TurnOff(X), and binary predicates that describe spatial relations between objects, such as On(A, B) or In(A, B). An example of the goal specification using PDDL language can be found in Figure 2-(C). The simulation terminates when all predicates are verified true.

# 4.2 Task Suites

While the pipeline in Section 4.1 supports the generation of an unlimited number of tasks, we offer fixed sets of tasks for benchmarking purposes. LIBERO has four task suites: LIBERO-SPATIAL, LIBERO-OBJECT, LIBERO-GOAL, and LIBERO-100. The first three task suites are curated to disentangle the transfer of declarative and procedural knowledge (as mentioned in (T1)), while LIBERO-100 is a suite of 100 tasks with entangled knowledge transfer.

LIBERO-X LIBERO-SPATIAL, LIBERO-OBJECT, and LIBERO-GOAL all have 10 tasks $^{4}$ and are designed to investigate the controlled transfer of knowledge about spatial information (declarative), objects (declarative), and task goals (procedural). Specifically, all tasks in LIBERO-SPATIAL request the robot to place a bowl, among the same set of objects, on a plate. But there are two identical bowls that differ only in their location or spatial relationship to other objects. Hence, to successfully complete LIBERO-SPATIAL, the robot needs to continually learn and memorize new spatial relationships. All tasks in LIBERO-OBJECT request the robot to pick-place a unique object. Hence, to accomplish LIBERO-OBJECT, the robot needs to continually learn and memorize new object types. All tasks in LIBERO-GOAL share the same objects with fixed spatial relationships but differ only in the task goal. Hence, to accomplish LIBERO-GOAL, the robot needs to continually learn new knowledge about motions and behaviors. More details are in Appendix C.

LIBERO-100 LIBERO-100 contains 100 tasks that entail diverse object interactions and versatile motor skills. In this paper, we split LIBERO-100 into 90 short-horizon tasks (LIBERO-90) and 10 long-horizon tasks (LIBERO-LONG). LIBERO-90 serves as the data source for pretraining (T5) and LIBERO-LONG for downstream evaluation of lifelong learning algorithms.

# 4.3 Lifelong Learning Algorithms

We implement three representative lifelong learning algorithms to facilitate research in algorithmic design for LLDM. Specifically, we implement Experience Replay (ER) [13], Elastic Weight Consolidation (EWC) [33], and PACKNET [41]. We pick ER, EWC, and PACKNET because they correspond to the memory-based, regularization-based, and dynamic-architecture-based methods for lifelong learning. In addition, prior research [69] has discovered that they are state-of-the-art methods. Besides these three methods, we also implement sequential finetuning (SEQL) and multitask learning (MTL), which serve as a lower bound and upper bound for lifelong learning algorithms, respectively. More details about the algorithms are in Appendix B.1.

# 4.4 Neural Network Architectures

We implement three vision-language policy networks, RESNET-RNN, RESNET-T, and VIT-T, that integrate visual, temporal, and linguistic information for LLDM. Language instructions of tasks are encoded using pretrained BERT embeddings [19]. The RESNET-RNN [42] uses a ResNet as the visual backbone that encodes per-step visual observations and an LSTM as the temporal backbone to process a sequence of encoded visual information. The language instruction is incorporated into the ResNet features using the FiLM method [50] and added to the LSTM inputs, respectively. RESNET-T

architecture [75] uses a similar ResNet-based visual backbone, but a transformer decoder [66] as the temporal backbone to process outputs from ResNet, which are a temporal sequence of visual tokens. The language embedding is treated as a separate token in inputs to the transformer alongside the visual tokens. The ViT-T architecture [31], which is widely used in visual-language tasks, uses a Vision Transformer (ViT) as the visual backbone and a transformer decoder as the temporal backbone. The language embedding is treated as a separate token in inputs of both ViT and the transformer decoder. All the temporal backbones output a latent vector for every decision-making step. We compute the multi-modal distribution over manipulation actions using a Gaussian-Mixture-Model (GMM) based output head [8, 42, 68]. In the end, a robot executes a policy by sampling a continuous value for end-effector action from the output distribution. Figure 6 visualizes the three architectures.

For all the lifelong learning algorithms and neural architectures, we use behavioral cloning (BC) [4] to train policies for individual tasks (See (2)). BC allows for efficient policy learning such that we can study lifelong learning algorithms with limited computational resources. To train BC, we provide 50 trajectories of high-quality demonstrations for every single task in the generated task suites. The demonstrations are collected by human experts through teleoperation with 3Dconnexion Spacemouse.

# 5 Experiments

Experiments are conducted as an initial study for the five research topics mentioned in Section 3. We first introduce the evaluation metric used in experiments, and present analysis of empirical results in LIBERO. The detailed experimental setup is in Appendix D. Our experiments focus on addressing the following research questions:

Q1: How do different architectures/LL algorithms perform under specific distribution shifts?   
Q2: To what extent does neural architecture impact knowledge transfer in LLDM, and are there any discernible patterns in the specialized capabilities of each architecture?   
Q3: How do existing algorithms from lifelong supervised learning perform on LLDM tasks?   
Q4: To what extent does language embedding affect knowledge transfer in LLDM?   
Q5: How robust are different LL algorithms to task ordering in LLDM?   
Q6: Can supervised pretraining improve downstream lifelong learning performance in LLDM?

# 5.1 Evaluation Metrics

We report three metrics: FWT (forward transfer) [20], NBT (negative backward transfer), and AUC (area under the success rate curve). All metrics are computed in terms of success rate, as previous literature has shown that the success rate is a more reliable metric than training loss for manipulation policies [42] (Detailed explanation in Appendix E.2). Lower NBT means a policy has better performance in the previously seen tasks, higher FWT means a policy learns faster on a new task, and higher AUC means an overall better performance considering both NBT and FWT. Specifically, denote $c_{i,j,e}$ as the agent's success rate on task j when it learned over i-1 previous tasks and has just learned e epochs ( $e \in \{0, 5, \ldots, 50\}$ ) on task i. Let $c_{i,i}$ be the best success rate over all evaluated epochs e for the current task i (i.e., $c_{i,i} = \max_{e} c_{i,i,e}$ ). Then, we find the earliest epoch $e_{i}^{*}$ in which the agent achieves the best performance on task i (i.e., $e_{i}^{*} = \arg \min_{e} c_{i,i,e_{i}} = c_{i,i}$ ), and assume for all $e \geq e_{i}^{*}$ , $c_{i,i,e} = c_{i,i}$ .⁵ Given a different task $j \neq i$ , we define $c_{i,j} = c_{i,j,e_{i}^{*}}$ . Then the three metrics are defined:

$$
\mathrm{FWT} = \sum_ {k \in [ K ]} \frac {\mathrm{FWT} _ {k}}{K}, \quad \mathrm{FWT} _ {k} = \frac {1}{1 1} \sum_ {e \in \{0 \dots 5 0 \}} c _ {k, k, e}
$$

$$
\mathrm{NBT} = \sum_ {k \in [ K ]} \frac {\mathrm{NBT} _ {k}}{K}, \quad \mathrm{NBT} _ {k} = \frac {1}{K - k} \sum_ {\tau = k + 1} ^ {K} \left(c _ {k, k} - c _ {\tau , k}\right) \tag {3}
$$

$$
\mathrm{AUC} = \sum_ {k \in [ K ]} \frac {\mathrm{AUC} _ {k}}{K}, \quad \mathrm{AUC} _ {k} = \frac {1}{K - k + 1} \left(\mathrm{FWT} _ {k} + \sum_ {\tau = k + 1} ^ {K} c _ {\tau , k}\right)
$$

A visualization of these metrics is provided in Figure 3.

![](images/7a602a79a51f4b15d543f04318ceab4a38c688e733c9b8898f8afababdf25995.jpg)

<details>
<summary>line</summary>

| Task | Success Rate |
| ---- | ------------ |
| k-1  | Low          |
| k    | High         |
| K    | Low          |
</details>

Figure 3: Metrics for LLDM.

# 5.2 Experimental Results

We present empirical results to address the research questions. Please refer to Appendix E.1 for the full results across all algorithms, policy architectures, and task suites.

Study on the Policy's Neural Architectures (Q1, Q2) Table 1 reports the agent's lifelong learning performance using the three different neural architectures on the four task suites. Results are reported when ER and PACKNET are used as they demonstrate the best lifelong learning performance across all task suites.

<table><tr><td rowspan="2">Policy Arch.</td><td colspan="3">ER</td><td colspan="3">PACKNET</td></tr><tr><td>FWT(↑)</td><td>NBT(↓)</td><td>AUC(↑)</td><td>FWT(↑)</td><td>NBT(↓)</td><td>AUC(↑)</td></tr><tr><td colspan="7">LIBERO-LONG</td></tr><tr><td>RESNet-RNN</td><td>0.16 ± 0.02</td><td>0.16 ± 0.02</td><td>0.08 ± 0.01</td><td>0.13 ± 0.00</td><td>0.21 ± 0.01</td><td>0.03 ± 0.00</td></tr><tr><td>RESNet-T</td><td>0.48 ± 0.02</td><td>0.32 ± 0.04</td><td>0.32 ± 0.01</td><td>0.22 ± 0.01</td><td>0.08 ± 0.01</td><td>0.25 ± 0.00</td></tr><tr><td>ViT-T</td><td>0.38 ± 0.05</td><td>0.29 ± 0.06</td><td>0.25 ± 0.02</td><td>0.36 ± 0.01</td><td>0.14 ± 0.01</td><td>0.34 ± 0.01</td></tr><tr><td colspan="7">LIBERO-SPATIAL</td></tr><tr><td>RESNet-RNN</td><td>0.40 ± 0.02</td><td>0.29 ± 0.02</td><td>0.29 ± 0.01</td><td>0.27 ± 0.03</td><td>0.38 ± 0.03</td><td>0.06 ± 0.01</td></tr><tr><td>RESNet-T</td><td>0.65 ± 0.03</td><td>0.27 ± 0.03</td><td>0.56 ± 0.01</td><td>0.55 ± 0.01</td><td>0.07 ± 0.02</td><td>0.63 ± 0.00</td></tr><tr><td>ViT-T</td><td>0.63 ± 0.01</td><td>0.29 ± 0.02</td><td>0.50 ± 0.02</td><td>0.57 ± 0.04</td><td>0.15 ± 0.00</td><td>0.59 ± 0.03</td></tr><tr><td colspan="7">LIBERO-OBJECT</td></tr><tr><td>RESNet-RNN</td><td>0.30 ± 0.01</td><td>0.27 ± 0.05</td><td>0.17 ± 0.05</td><td>0.29 ± 0.02</td><td>0.35 ± 0.02</td><td>0.13 ± 0.01</td></tr><tr><td>RESNet-T</td><td>0.67 ± 0.07</td><td>0.43 ± 0.04</td><td>0.44 ± 0.06</td><td>0.60 ± 0.07</td><td>0.17 ± 0.05</td><td>0.60 ± 0.05</td></tr><tr><td>ViT-T</td><td>0.70 ± 0.02</td><td>0.28 ± 0.01</td><td>0.57 ± 0.01</td><td>0.58 ± 0.03</td><td>0.18 ± 0.02</td><td>0.56 ± 0.04</td></tr><tr><td colspan="7">LIBERO-GOAL</td></tr><tr><td>RESNet-RNN</td><td>0.41 ± 0.00</td><td>0.35 ± 0.01</td><td>0.26 ± 0.01</td><td>0.32 ± 0.03</td><td>0.37 ± 0.04</td><td>0.11 ± 0.01</td></tr><tr><td>RESNet-T</td><td>0.64 ± 0.01</td><td>0.34 ± 0.02</td><td>0.49 ± 0.02</td><td>0.63 ± 0.02</td><td>0.06 ± 0.01</td><td>0.75 ± 0.01</td></tr><tr><td>ViT-T</td><td>0.57 ± 0.00</td><td>0.40 ± 0.02</td><td>0.38 ± 0.01</td><td>0.69 ± 0.02</td><td>0.08 ± 0.01</td><td>0.76 ± 0.02</td></tr></table>

Table 1: Performance of the three neural architectures using ER and PACKNET on the four task suites. Results are averaged over three seeds and we report the mean and standard error. The best performance is bolded, and colored in purple if the improvement is statistically significant over other neural architectures, when a two-tailed, Student's t-test under equal sample sizes and unequal variance is applied with a p-value of 0.05.

Findings: First, we observe that RESNET-T and ViT-T work much better than RESNET-RNN on average, indicating that using a transformer on the “temporal” level could be a better option than using an RNN model. Second, the performance difference among different architectures depends on the underlying lifelong learning algorithm. If PACKNET (a dynamic architecture approach) is

used, we observe no significant performance difference between RESNET-T and ViT-T except on the LIBERO-LONG task suite where ViT-T performs much better than RESNET-T. In contrast, if ER is used, we observe that RESNET-T performs better than ViT-T on all task suites except LIBERO-OBJECT. This potentially indicates that the ViT architecture is better at processing visual information with more object varieties than the ResNet architecture when the network capacity is sufficiently large (See the MTL results in Table 8 on LIBERO-OBJECT as the supporting evidence). The above findings shed light on how one can improve architecture design for better processing of spatial and temporal information in LLDM.

Study on Lifelong Learning Algorithms (Q1, Q3) Table 2 reports the lifelong learning performance of the three lifelong learning algorithms, together with the SEQL and MTL baselines. All experiments use the same RESNET-T architecture as it performs the best across all policy architectures.

<table><tr><td>Lifelong Algo.</td><td>FWT(↑)</td><td>NBT(↓)</td><td>AUC(↑)</td><td>FWT(↑)</td><td>NBT(↓)</td><td>AUC(↑)</td></tr><tr><td></td><td colspan="3">LIBERO-LONG</td><td colspan="3">LIBERO-SPATIAL</td></tr><tr><td>SEQL</td><td>0.54 ± 0.01</td><td>0.63 ± 0.01</td><td>0.15 ± 0.00</td><td>0.72 ± 0.01</td><td>0.81 ± 0.01</td><td>0.20 ± 0.01</td></tr><tr><td>ER</td><td>0.48 ± 0.02</td><td>0.32 ± 0.04</td><td>0.32 ± 0.01</td><td>0.65 ± 0.03</td><td>0.27 ± 0.03</td><td>0.56 ± 0.01</td></tr><tr><td>EWC</td><td>0.13 ± 0.02</td><td>0.22 ± 0.03</td><td>0.02 ± 0.00</td><td>0.23 ± 0.01</td><td>0.33 ± 0.01</td><td>0.06 ± 0.01</td></tr><tr><td>PACKNET</td><td>0.22 ± 0.01</td><td>0.08 ± 0.01</td><td>0.25 ± 0.00</td><td>0.55 ± 0.01</td><td>0.07 ± 0.02</td><td>0.63 ± 0.00</td></tr><tr><td>MTL</td><td></td><td></td><td>0.48 ± 0.01</td><td></td><td></td><td>0.83 ± 0.00</td></tr><tr><td></td><td colspan="3">LIBERO-OBJECT</td><td colspan="3">LIBERO-GOAL</td></tr><tr><td>SEQL</td><td>0.78 ± 0.04</td><td>0.76 ± 0.04</td><td>0.26 ± 0.02</td><td>0.77 ± 0.01</td><td>0.82 ± 0.01</td><td>0.22 ± 0.00</td></tr><tr><td>ER</td><td>0.67 ± 0.07</td><td>0.43 ± 0.04</td><td>0.44 ± 0.06</td><td>0.64 ± 0.01</td><td>0.34 ± 0.02</td><td>0.49 ± 0.02</td></tr><tr><td>EWC</td><td>0.56 ± 0.03</td><td>0.69 ± 0.02</td><td>0.16 ± 0.02</td><td>0.32 ± 0.02</td><td>0.48 ± 0.03</td><td>0.06 ± 0.00</td></tr><tr><td>PACKNET</td><td>0.60 ± 0.07</td><td>0.17 ± 0.05</td><td>0.60 ± 0.05</td><td>0.63 ± 0.02</td><td>0.06 ± 0.01</td><td>0.75 ± 0.01</td></tr><tr><td>MTL</td><td></td><td></td><td>0.54 ± 0.02</td><td></td><td></td><td>0.80 ± 0.01</td></tr></table>

Table 2: Performance of three lifelong algorithms and the SEQL and MTL baselines on the four task suites, where the policy is fixed to be RESNET-T. Results are averaged over three seeds and we report the mean and standard error. The best performance is bolded, and colored in purple if the improvement is statistically significant over other algorithms, when a two-tailed, Student's t-test under equal sample sizes and unequal variance is applied with a p-value of 0.05.

Findings: We observed a series of interesting findings that could potentially benefit future research on algorithm design for LLDM: 1) SEQL shows the best FWT over all task suites. This is surprising since it indicates all lifelong learning algorithms we consider actually hurt forward transfer; 2) PACKNET outperforms other lifelong learning algorithms on LIBERO-X but is outperformed by ER significantly on LIBERO-LONG, mainly because of low forward transfer. This confirms that the dynamic architecture approach is good at preventing forgetting. But since PACKNET splits the network into different sub-networks, the essential capacity of the network for learning any individual task is smaller. Therefore, we conjecture that PACKNET is not rich enough to learn on LIBERO-LONG; 3) EWC works worse than SEQL, showing that the regularization on the loss term can actually impede the agent's performance on LLDM problems (See Appendix E.2); and 4) ER, the rehearsal method, is robust across all task suites.

Study on Language Embeddings as the Task Identifier (Q4) To investigate to what extent language embedding play a role in LLDM, we compare the performance of the same lifelong learner using four different pretrained language embeddings. Namely, we choose BERT [19], CLIP [52], GPT-2 [53] and the Task-ID embedding. Task-ID embeddings are produced by feeding a string such as “Task 5” into a pretrained BERT model.

Findings: From Table 3, we observe no statistically significant difference among various language embeddings, including the Task-ID embedding. This, we believe, is due to sentence embeddings functioning as bag-of-words that differentiates different tasks. This insight calls for better language encoding to harness the semantic information in task descriptions. Despite the similar performance, we opt for BERT embeddings as our default task embedding.

<table><tr><td>Embedding Type</td><td>Dimension</td><td>FWT(↑)</td><td>NBT(↓)</td><td>AUC(↑)</td></tr><tr><td>BERT</td><td>768</td><td>0.48 ± 0.02</td><td>0.32 ± 0.04</td><td>0.32 ± 0.01</td></tr><tr><td>CLIP</td><td>512</td><td>0.52 ± 0.00</td><td>0.34 ± 0.01</td><td>0.35 ± 0.01</td></tr><tr><td>GPT-2</td><td>768</td><td>0.46 ± 0.01</td><td>0.34 ± 0.02</td><td>0.30 ± 0.01</td></tr><tr><td>Task-ID</td><td>768</td><td>0.50 ± 0.01</td><td>0.37 ± 0.01</td><td>0.33 ± 0.01</td></tr></table>

Table 3: Performance of a lifelong learner using four different language embeddings on LIBERO-LONG, where we fix the policy architecture to RESNET-T and the lifelong learning algorithm to ER. The Task-ID embeddings are retrieved by feeding “Task + ID” into a pretrained BERT model. Results are averaged over three seeds and we report the mean and standard error. The best performance is bolded. No statistically significant difference is observed among the different language embeddings.

Study on task ordering (Q5) Figure 4 shows the result of the study on Q4. For all experiments in this study, we used RESNET-T as the neural architecture and evaluated both ER and PACKNET. As the figure illustrates, the performance of both algorithms varies across different task orderings. This finding highlights an important direction for future research: developing algorithms or architectures that are robust to varying task orderings.

![](images/839a26f5353fb376f5ec3ec38c6064262b117b8712920fdb090f890aff3adfea.jpg)

<details>
<summary>bar</summary>

| Model | AUC (5 orderings) | Success Rate |
| :--- | :--- | :--- |
| ER + ResNet-T | 0.32 | 0.30 |
| ER + ResNet-T | 0.26 | 0.30 |
| ER + ResNet-T | 0.24 | 0.30 |
| PackNet + ResNet-T | 0.25 | 0.33 |
| PackNet + ResNet-T | 0.28 | 0.31 |
| PackNet + ResNet-T | 0.27 | 0.33 |
</details>

Figure 4: Performance of ER and PACKNET using RESNET-T on five different task orderings. An error bar shows the performance standard deviation for a fixed ordering.

Findings: From Figure 4, we observe that indeed different task ordering could result in very different performances for the same algorithm. Specifically, such difference is statistically significant for PACKNET.

Study on How Pretraining Affects Downstream LLDM (Q6) Fig 5 reports the results on LIBERO-LONG of five combinations of algorithms and policy architectures, when the underlying model is pretrained on the 90 short-horizon tasks in LIBERO-100 or learned from scratch. For pretraining, we apply behavioral cloning on the 90 tasks using the three policy architectures for 50 epochs. We save a checkpoint every 5 epochs of training and then pick the checkpoint for each architecture that has the best performance as the pretrained model for downstream LLDM.

![](images/22793841e240bd3c054205b568bdb287b4249b4b4902f202c77154379a78ebb4.jpg)

<details>
<summary>bar</summary>

| Model | w/ pretraining | w/o pretraining | multitask |
| :--- | :--- | :--- | :--- |
| ER ResNet-RNN | 0.04 | 0.08 | 0.20 |
| ER ResNet-T | 0.22 | 0.32 | 0.48 |
| ER ViT-T | 0.10 | 0.25 | 0.46 |
| EWC ResNet-T | 0.01 | 0.02 | 0.48 |
| PackNet ResNet-T | 0.17 | 0.25 | 0.48 |
</details>

Figure 5: Performance of different combinations of algorithms and architectures without pretraining or with pretraining. The multi-task learning performance is also included for reference.

Findings: We observe that the basic supervised pretraining can hurt the model's downstream lifelong learning performance. This, together with the results seen in Table 2 (e.g., naive sequential fine-tuning

has better forward transfer than when lifelong learning algorithms are applied), indicates that better pretraining techniques are needed.

Attention Visualization: To better understand what type of knowledge the agent forgets during the lifelong learning process, we visualize the agent's attention map on each observed image input. The visualized saliency maps and the discussion can be found in Appendix E.4.

# 6 Related Work

This section provides an overview of existing benchmarks for lifelong learning and robot learning. We refer the reader to Appendix B.1 for a detailed review of lifelong learning algorithms.

Lifelong Learning Benchmarks Pioneering work has adapted standard vision or language datasets for studying LL. This line of work includes image classification datasets like MNIST [18], CIFAR [34], and ImageNet [17]; segmentation datasets like Core50 [38]; and natural language understanding datasets like GLUE [67] and SuperGLUE [59]. Besides supervised learning datasets, video game benchmarks (e.g., Atari [46], XLand [64], and VisDoom [30]) in reinforcement learning (RL) have also been used for studying LL. However, LL in standard supervised learning does not involve procedural knowledge transfer, while RL problems in games do not represent human activities. ContinualWorld [69] modifies the 50 manipulation tasks in MetaWorld for LL. CORA [51] builds four lifelong RL benchmarks based on Atari, Procgen [15], MiniHack [58], and ALFRED [62]. F-SIOL-310 [3] and OpenLORIS [61] are challenging real-world lifelong object learning datasets that are captured from robotic vision systems. Prior works have also analyzed different components in a LL agent [45, 70, 21], but they do not focus on robot manipulation problems.

Robot Learning Benchmarks A variety of robot learning benchmarks have been proposed to address challenges in meta learning (MetaWorld [73]), causality learning (CausalWorld [1]), multi-task learning [27, 35], policy generalization to unseen objects [47, 24], and compositional learning [44]. Compared to existing benchmarks in lifelong learning and robot learning, the task suites in LIBERO are curated to address the research topics of LLDM. The benchmark includes a large number of tasks based on everyday human activities that feature rich interactive behaviors with a diverse range of objects. Additionally, the tasks in LIBERO are procedurally generated, making the benchmark scalable and adaptable. Moreover, the provided high-quality human demonstration dataset in LIBERO supports and encourages learning efficiency.

# 7 Conclusion and Limitations

This paper introduces LIBERO, a new benchmark in the robot manipulation domain for supporting research in LLDM. LIBERO includes a procedural generation pipeline that can create an infinite number of manipulation tasks in the simulator. We use this pipeline to create 130 standardized tasks and conduct a comprehensive set of experiments on policy and algorithm designs. The empirical results suggest several future research directions: 1) how to design a better neural architecture to better process spatial information or temporal information; 2) how to design a better algorithm to improve forward transfer ability; and 3) how to use pretraining to help improve lifelong learning performance. In the short term, we do not envision any negative societal impacts triggered by LIBERO. But as the lifelong learner mainly learns from humans, studying how to preserve user privacy within LLDM [36] is crucial in the long run.

# References

[1] Ossama Ahmed, Frederik Träuble, Anirudh Goyal, Alexander Neitz, Yoshua Bengio, Bernhard Schölkopf, Manuel Wüthrich, and Stefan Bauer. Causalworld: A robotic manipulation benchmark for causal structure and transfer learning. arXiv preprint arXiv:2010.04296, 2020.   
[2] Ali Ayub and Carter Fendley. Few-shot continual active learning by a robot. arXiv preprint arXiv:2210.04137, 2022.   
[3] Ali Ayub and Alan R Wagner. F-siol-310: A robotic dataset and benchmark for few-shot incremental object learning. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 13496–13502. IEEE, 2021.

[4] Michael Bain and Claude Sammut. A framework for behavioural cloning. In Machine Intelligence 15, pages 103–129, 1995.   
[5] Eseoghene Ben-Iwhiwhu, Saptarshi Nath, Praveen K Pilly, Soheil Kolouri, and Andrea Soltoggio. Lifelong reinforcement learning with modulating masks. arXiv preprint arXiv:2212.11110, 2022.   
[6] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.   
[7] Magdalena Biesialska, Katarzyna Biesialska, and Marta R Costa-Jussa. Continual lifelong learning in natural language processing: A survey. arXiv preprint arXiv:2012.09823, 2020.   
[8] Christopher M Bishop. Mixture density networks. 1994.   
[9] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920–15930, 2020.   
[10] Rich Caruana. Multitask learning. Machine learning, 28(1):41-75, 1997.   
[11] Arslan Chaudhry, Puneet K Dokania, Thalaiyasingam Ajanthan, and Philip HS Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Proceedings of the European Conference on Computer Vision (ECCV), pages 532–547, 2018.   
[12] Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. arXiv preprint arXiv:1812.00420, 2018.   
[13] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc'Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486, 2019.   
[14] Brian Cheung, Alexander Terekhov, Yubei Chen, Pulkit Agrawal, and Bruno Olshausen. Superposition of many models into one. Advances in neural information processing systems, 32, 2019.   
[15] Karl Cobbe, Chris Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. In International conference on machine learning, pages 2048–2056. PMLR, 2020.   
[16] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7):3366-3385, 2021.   
[17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.   
[18] Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.   
[19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.   
[20] Natalia Díaz-Rodríguez, Vincenzo Lomonaco, David Filliat, and Davide Maltoni. Don't forget, there is more than forgetting: new metrics for continual learning. arXiv preprint arXiv:1810.13166, 2018.   
[21] Beyza Ermis, Giovanni Zappella, Martin Wistuba, and Cédric Archambeau. Memory efficient continual learning with transformers. 2022.

[22] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022.   
[23] Sam Greydanus, Anurag Koul, Jonathan Dodge, and Alan Fern. Visualizing and understanding atari agents. ArXiv, abs/1711.00138, 2017.   
[24] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, et al. Maniskill2: A unified benchmark for generalizable manipulation skills. arXiv preprint arXiv:2302.04659, 2023.   
[25] Matthew Hausknecht and Peter Stone. Deep recurrent q-learning for partially observable mdps. In 2015 aaai fall symposium series, 2015.   
[26] Ching-Yi Hung, Cheng-Hao Tu, Cheng-En Wu, Chien-Hung Chen, Yi-Ming Chan, and Chu-Song Chen. Compacting, picking and growing for unforgettable continual learning. Advances in Neural Information Processing Systems, 32, 2019.   
[27] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2):3019–3026, 2020.   
[28] Leslie Pack Kaelbling. The foundation of efficient robot learning. Science, 369(6506):915–916, 2020.   
[29] Minsoo Kang, Jaeyoo Park, and Bohyung Han. Class-incremental learning by knowledge distillation with adaptive feature consolidation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16071–16080, 2022.   
[30] Michał Kempka, Marek Wydmuch, Grzegorz Runc, Jakub Toczek, and Wojciech Jaśkowski. Vizdoom: A doom-based ai research platform for visual reinforcement learning. In 2016 IEEE conference on computational intelligence and games (CIG), pages 1–8. IEEE, 2016.   
[31] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In International Conference on Machine Learning, pages 5583–5594. PMLR, 2021.   
[32] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
[33] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.   
[34] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.   
[35] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pages 80–93. PMLR, 2023.   
[36] B. Liu, Qian Liu, and Peter Stone. Continual learning and private unlearning. In CoLLAs, 2022.   
[37] Hao Liu and Huaping Liu. Continual learning with recursive gradient optimization. arXiv preprint arXiv:2201.12522, 2022.   
[38] Vincenzo Lomonaco and Davide Maltoni. Core50: a new dataset and benchmark for continuous object recognition. In Conference on Robot Learning, pages 17–26. PMLR, 2017.   
[39] David Lopez-Paz and Marc'Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017.

[40] Zheda Mai, Ruiwen Li, Jihwan Jeong, David Quispe, Hyunwoo Kim, and Scott Sanner. Online continual learning in image classification: An empirical survey. Neurocomputing, 469:28–51, 2022.   
[41] Arun Mallya and Svetlana Lazebnik. Packnet: Adding multiple tasks to a single network by iterative pruning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 7765–7773, 2018.   
[42] Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298, 2021.   
[43] Drew McDermott, Malik Ghallab, Adele Howe, Craig Knoblock, Ashwin Ram, Manuela Veloso, Daniel Weld, and David Wilkins. Pddl-the planning domain definition language. 1998.   
[44] Jorge A Mendez, Marcel Hussing, Meghna Gummadi, and Eric Eaton. Composite: A compositional reinforcement learning benchmark. arXiv preprint arXiv:2207.04136, 2022.   
[45] Seyed Iman Mirzadeh, Arslan Chaudhry, Dong Yin, Timothy Nguyen, Razvan Pascanu, Dilan Gorur, and Mehrdad Farajtabar. Architecture matters in continual learning. arXiv preprint arXiv:2202.00275, 2022.   
[46] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.   
[47] Tongzhou Mu, Zhan Ling, Fanbo Xiang, Derek Yang, Xuanlin Li, Stone Tao, Zhiao Huang, Zhiwei Jia, and Hao Su. Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations. arXiv preprint arXiv:2107.14483, 2021.   
[48] Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E Taylor, and Peter Stone. Curriculum learning for reinforcement learning domains: A framework and survey. arXiv preprint arXiv:2003.04960, 2020.   
[49] German I Parisi, Ronald Kemker, Jose L Part, Christopher Kanan, and Stefan Wermter. Continual lifelong learning with neural networks: A review. Neural Networks, 113:54–71, 2019.   
[50] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.   
[51] Sam Powers, Eliot Xing, Eric Kolve, Roozbeh Mottaghi, and Abhinav Gupta. Cora: Benchmarks, baselines, and metrics as a platform for continual reinforcement learning agents. arXiv preprint arXiv:2110.10067, 2021.   
[52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.   
[53] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
[54] Amanda Rios and Laurent Itti. Lifelong learning without a task oracle. In 2020 IEEE 32nd International Conference on Tools with Artificial Intelligence (ICTAI), pages 255–263. IEEE, 2020.   
[55] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

[56] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.   
[57] Gobinda Saha, Isha Garg, Aayush Ankit, and Kaushik Roy. Space: Structured compression and sharing of representational space for continual learning. IEEE Access, 9:150480–150494, 2021.   
[58] Mikayel Samvelyan, Robert Kirk, Vitaly Kurin, Jack Parker-Holder, Minqi Jiang, Eric Hambro, Fabio Petroni, Heinrich Küttler, Edward Grefenstette, and Tim Rocktäschel. Minihack the planet: A sandbox for open-ended reinforcement learning research. arXiv preprint arXiv:2109.13202, 2021.   
[59] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4938–4947, 2020.   
[60] Jonathan Schwarz, Wojciech Czarnecki, Jelena Luketina, Agnieszka Grabska-Barwinska, Yee Whye Teh, Razvan Pascanu, and Raia Hadsell. Progress & compress: A scalable framework for continual learning. In International Conference on Machine Learning, pages 4528–4537. PMLR, 2018.   
[61] Qi She, Fan Feng, Xinyue Hao, Qihan Yang, Chuanlin Lan, Vincenzo Lomonaco, Xuesong Shi, Zhengwei Wang, Yao Guo, Yimin Zhang, et al. Openloris-object: A robotic vision dataset and benchmark for lifelong deep learning. In 2020 IEEE international conference on robotics and automation (ICRA), pages 4767–4773. IEEE, 2020.   
[62] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10740–10749, 2020.   
[63] Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Martín-Martín, Fei Xia, Kent Elliott Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, Karen Liu, et al. Behavior: Benchmark for everyday household activities in virtual, interactive, and ecological environments. In Conference on Robot Learning, pages 477–490. PMLR, 2022.   
[64] Open Ended Learning Team, Adam Stoke, Anuj Mahajan, Catarina Barros, Charlie Deck, Jakob Bauer, Jakub Sygnowski, Maja Trebacz, Max Jaderberg, Michael Mathieu, et al. Open-ended learning leads to generally capable agents. arXiv preprint arXiv:2107.12808, 2021.   
[65] Sebastian Thrun and Tom M Mitchell. Lifelong robot learning. Robotics and autonomous systems, 15(1-2):25–46, 1995.   
[66] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[67] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.   
[68] Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, and Anima Anandkumar. Mimicplay: Long-horizon imitation learning by watching human play. arXiv preprint arXiv:2302.12422, 2023.   
[69] Maciej Wołczyk, Michal Zajkac, Razvan Pascanu, Lukasz Kuci’nski, and Piotr Milo’s. Continual world: A robotic benchmark for continual reinforcement learning. In Neural Information Processing Systems, 2021.   
[70] Maciej Wołczyk, Michal Zajkac, Razvan Pascanu, Lukasz Kuci'nski, and Piotr Milo's. Disentangling transfer in continual reinforcement learning. ArXiv, abs/2209.13900, 2022.

[71] Lemeng Wu, Bo Liu, Peter Stone, and Qiang Liu. Firefly neural architecture descent: a general approach for growing neural networks. Advances in Neural Information Processing Systems, 33:22373–22383, 2020.   
[72] Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. Lifelong learning with dynamically expandable networks. arXiv preprint arXiv:1708.01547, 2017.   
[73] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pages 1094–1100. PMLR, 2020.   
[74] Da-Wei Zhou, Fu-Yun Wang, Han-Jia Ye, Liang Ma, Shiliang Pu, and De-Chuan Zhan. Forward compatible few-shot class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9046–9056, 2022.   
[75] Yifeng Zhu, Abhishek Joshi, Peter Stone, and Yuke Zhu. Viola: Imitation learning for vision-based manipulation with object proposal priors. arXiv preprint arXiv:2210.11339, 2022.   
[76] Yuke Zhu, Josiah Wong, Ajay Mandlekar, and Roberto Martín-Martín. robosuite: A modular simulation framework and benchmark for robot learning. arXiv preprint arXiv:2009.12293, 2020.

# Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]   
(b) Did you describe the limitations of your work? [Yes]   
(c) Did you discuss any potential negative societal impacts of your work? [Yes]   
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [N/A]   
(b) Did you include complete proofs of all theoretical results? [N/A]

3. If you ran experiments (e.g. for benchmarks)...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes]   
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes]   
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes]   
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]   
(b) Did you mention the license of the assets? [N/A]   
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes]   
(d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [N/A]   
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]   
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]   
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

# A Implemented Neural Architectures and Lifelong Learning Algorithms

<table><tr><td rowspan="3">Neural Policy Arch.</td><td>RESNET-RNN</td></tr><tr><td>RESNET-T</td></tr><tr><td>VIT-T</td></tr><tr><td rowspan="5">Lifelong Learning Algo.</td><td>SEQL</td></tr><tr><td>EWC [33]</td></tr><tr><td>ER [13]</td></tr><tr><td>PACKNET [41]</td></tr><tr><td>MTL</td></tr></table>

Table 4: The implemented neural policy architectures and the lifelong learning algorithms in LIBERO.

# A.1 Neural Architectures

In Section 4.4, we outlined the neural network architectures utilized in our experiments, namely RESNET-RNN, RESNET-T, and V1T-T. The specifics of each architecture are illustrated in Figure 6. Furthermore, Table 5, 6, and 7 display the hyperparameters for the architectures used throughout all of our experiments.

![](images/95ec179220e03eafd5928b13bd548b3217d0c1b8e2644948d7dee0f481f432b3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["GMM Policy Head"] --> B["LSTM"]
    B --> C["Joint/Gripper"]
    C --> D["ResNet Encoder"]
    D --> E["AgentView"]
    D --> F["ResNet Encoder"]
    E --> G["Language Embedding"]
    F --> H["Eye-in-Hand"]
    G --> I["⊕"]
    H --> J["⊕"]
    I --> K["⊕"]
    J --> L["⊕"]
```
</details>

![](images/222130da1b72c61d4661eac5a0d5e539d4eb7485300c52bf24f4f8ce08ea72dd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["GMM Policy Head"] --> B["Temporal Transformer"]
    B --> C["Token"]
    B --> D["Token"]
    B --> E["Token"]
    B --> F["Token"]
    C --> G["Language Embedding"]
    D --> H["ResNet Encoder"]
    E --> I["ResNet Encoder"]
    F --> J["Joint/Gripper"]
    G --> K["AgentView"]
    H --> L["Eye-in-Hand"]
    I --> L
    J --> L
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#ccf,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#cff,stroke:#333
    style H fill:#cff,stroke:#333
    style I fill:#cff,stroke:#333
    style J fill:#cff,stroke:#333
```
</details>

![](images/bed396fd489e2817fb77001dc610ae118373c4e496f649f459711bb7acb3023e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["BC-Transformer2"] --> B["GMM Policy Head"]
    A --> C["Temporal Transformer"]
    C --> D["Token"]
    C --> E["Token"]
    C --> F["Token"]
    C --> G["Token"]
    D --> H["Language Embedding"]
    E --> I["VIT Encoder"]
    F --> J["VIT Encoder"]
    G --> K["Joint/Gripper"]
    H --> L["AgentView"]
    I --> M["Eye-in-Hand"]
    J --> N["Joint/Gripper"]
```
</details>

Figure 6: We provide visualizations of the architectures for RESNET-RNN, RESNET-T, and VIT-T, respectively. It is worth noting that each model architecture incorporates language embedding in distinct ways.

![](images/559e947ddb2710b19ca80f6f51a019414284e282c315df1f0ef004c62bb0a3cb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["ResNet Encoder"] --> B["Linear"]
    A --> C["Spatial SoftMax"]
    C --> D["(1 + γ) ⊙ ⊕ β"]
    C --> E["Residual Block"]
    F["Language Embedding"] --> G["(1 + γ) ⊙ ⊕ β"]
    F --> H["Residual Block"]
    F --> I["(1 + γ) ⊙ ⊕ β"]
    F --> J["Residual Block"]
    F --> K["Convolution"]
    L["ViT Encoder"] --> M["Linear"]
    L --> N["Self Attention"]
    N --> O["..."]
    N --> P["Self Attention"]
    P --> Q["Self Attention"]
    Q --> R["Convolution"]
    S["Language Token"] --> T["Image Patch Tokens"]
    T --> U["..."]
    T --> V["..."]
```
</details>

Figure 7: The image encoders: ResNet-based encoder and the vision transformer-based encoder.

# B Computation

For all experiments, we use a single Nvidia A100 GPU or a single Nvidia A40 GPU (CUDA 11.7) with 8 16 CPUs for training and evaluation.

<table><tr><td>Variable</td><td>Value</td></tr><tr><td>resnet_image_embed_size</td><td>64</td></tr><tr><td>text_embed_size</td><td>32</td></tr><tr><td>rnn_hidden_size</td><td>1024</td></tr><tr><td>rnn_layer_num</td><td>2</td></tr><tr><td>rnn_dropout</td><td>0.0</td></tr></table>

Table 5: Hyper parameters of RESNET-RNN.

<table><tr><td>Variable</td><td>Value</td></tr><tr><td>extra_info_hidden_size</td><td>128</td></tr><tr><td>img_embed_size</td><td>64</td></tr><tr><td>transformer_num_layers</td><td>4</td></tr><tr><td>transformer_num_heads</td><td>6</td></tr><tr><td>transformer_head_output_size</td><td>64</td></tr><tr><td>transformer_mlp_hidden_size</td><td>256</td></tr><tr><td>transformer_dropout</td><td>0.1</td></tr><tr><td>transformer_max_seq_len</td><td>10</td></tr></table>

Table 6: Hyper parameters of RESNET-T.

<table><tr><td>Variable</td><td>Value</td></tr><tr><td>extra_info_hidden_size</td><td>128</td></tr><tr><td>img_embed_size</td><td>128</td></tr><tr><td>spatial_transformer_num_layers</td><td>7</td></tr><tr><td>spatial_transformer_num_heads</td><td>8</td></tr><tr><td>spatial_transformer_head_output_size</td><td>120</td></tr><tr><td>spatial_transformer_mlp_hidden_size</td><td>256</td></tr><tr><td>spatial_transformer_dropout</td><td>0.1</td></tr><tr><td>spatial_down_sample_embed_size</td><td>64</td></tr><tr><td>temporal_transformer_input_size</td><td>null</td></tr><tr><td>temporal_transformer_num_layers</td><td>4</td></tr><tr><td>temporal_transformer_num_heads</td><td>6</td></tr><tr><td>temporal_transformer_head_output_size</td><td>64</td></tr><tr><td>temporal_transformer_mlp_hidden_size</td><td>256</td></tr><tr><td>temporal_transformer_dropout</td><td>0.1</td></tr><tr><td>temporal_transformer_max_seq_len</td><td>10</td></tr></table>

Table 7: Hyper parameters of ViT-T.

# B.1 Lifelong Learning Algorithms

Lifelong learning (LL) is a field of study that aims to understand how an agent can continually acquire and retain knowledge over an infinite sequence of tasks without catastrophically forgetting previous knowledge. Recent literature proposes three main approaches to address the problem of catastrophic forgetting in deep learning: Dynamic Architecture approaches, Regularization-Based approaches, and Rehearsal approaches. Although some recent works explore the combination of different approaches $[2, 29, 54]$ or new strategies $[74, 57, 14]$ , our benchmark aims to provide an in-depth analysis of these three basic lifelong learning directions to reveal their pros and cons on robot learning tasks.

The dynamic architecture approach gradually expands the learning model to incorporate new knowledge $[56, 72, 41, 26, 71, 5]$ . Regularization-based methods, on the other hand, regularize the learner to a previous checkpoint when it learns a new task $[33, 11, 60, 37]$ . Rehearsal methods save exemplar data from prior tasks and replay them with new data to consolidate the agent's memory $[13, 39, 12, 9]$ . For a comprehensive review of LL methods, we refer readers to surveys $[16, 49]$ .

The following paragraphs provide details on the three lifelong learning algorithms that we have implemented.

ER Experience Replay (ER) [13] is a rehearsal-based approach that maintains a memory buffer of samples from previous tasks and leverages it to learn new tasks. After the completion of policy learning for a task, ER stores a portion of the data into a storage memory. When training a new task, ER samples data from the memory and combines it with the training data from the current task so that the training data approximately represents the empirical distribution of all-task data. In our implementation, we use a replay buffer to store a portion of the training data (up to 1000 trajectories) after training each task. For every training iteration during the training of a new task, we uniformly

sample a fixed number of replay data from the memory (32 trajectories) along with each batch of training data from the new task.

EWC Elastic Weight Consolidation(EWC) [33] is a regularization-based approach that add a regularization term that constraints neural network update to the original single-task learning objective. Specifically, EWC uses the Fisher information matrix that quantify the importance of every neural network parameter. The loss function for task k is:

$$
\mathcal {L} _ {k} ^ {E W C} (\theta) = \mathcal {L} _ {K} ^ {B C} (\theta) + \sum_ {i} \frac {\lambda}{2} F _ {i} \left(\theta_ {i} - \theta_ {k - 1, i} ^ {*}\right) ^ {2},
$$

where $\lambda$ is a penalty hyperparameter, and the coefficient $F_{i}$ is the diagonal of the Fisher information matrix: $F_{k} = \mathbb{E}_{s \sim \mathcal{D}_{k}} \mathbb{E}_{a \sim p_{\theta}(\cdot|s)} (\nabla_{\theta_{k}} \log p_{\theta_{k}}(a|s))^{2}$ . In this work, we use the online update version of EWC that updates the Fisher information matrix using exponential moving average along the lifelong learning process, and use the empirical estimation of above Fisher information matrix to stabilize the estimation. Formally, the actually used estimation of Fisher Information Matrix is $\tilde{F}_{k} = \gamma F_{k-1} + (1 - \gamma) F_{k}$ , where $F_{k} = \mathbb{E}_{(s,a) \sim \mathcal{D}_{k}} (\nabla_{\theta_{k}} \log p_{\theta_{k}}(a|s))^{2}$ and k is the task number. We set $\gamma = 0.9$ and $\lambda = 5 \cdot 10^{4}$ .

PACKNET PACKNET [41] is a dynamic architecture-based approach that aims to prevent changes to parameters that are important for previous tasks in lifelong learning. To achieve this, PACKNET iteratively trains, prunes, fine-tunes, and freezes parts of the network. The method theoretically completely avoids catastrophic forgetting, but for each new task, the number of available parameters shrinks. The pruning process in PACKNET involves two stages. First, the network is trained, and at the end of the training, a fixed proportion of the most important parameters (25% in our implementation) are chosen, and the rest are pruned. Second, the selected part of the network is fine-tuned and then frozen. In our implementation, we follow the original paper [41] and do not train all biases and normalization layers. We perform the same number of fine-tuning epochs as for training (50 epochs in our implementation). Note that all evaluation metrics are calculated before the fine-tuning stage.

# C LIBERO Task Suite Designs

# C.1 Task Suites

We visualize all the tasks from the four task suites in Figure 8-11. Figure 8 visualizes the initial states since the task goals are always the same. All the figures visualize the goal states of tasks except for Figure 8, which visualizes the initial states since the task goals are always the same.

![](images/b088d7e520008b19059191a9babbf9d0c1c2bea9cb32da566ab87bf436161554.jpg)

<details>
<summary>natural_image</summary>

Sequence of 3D-rendered images showing a robotic arm interacting with various small objects on a wooden surface (no text or symbols visible)
</details>

Figure 8: LIBERO-SPATIAL

![](images/00de7e762a3396a313d9ca820b323eeea9c69cec089fa012f1dab057d5140100.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a robotic arm in a basket on tiled floor, surrounded by scattered bottles and containers (no text or symbols visible)
</details>

Figure 9: LIBERO-OBJECT

![](images/5bd4ef6c4da95bbe86c237e1bf09caf0f2585b870b6c49cc206ccfbcb2118ecc.jpg)

<details>
<summary>natural_image</summary>

Sequence of 3D robotic arm diagrams showing mechanical assembly and component placement on a wooden surface (no text or symbols)
</details>

Figure 10: LIBERO-GOAL

![](images/fe43f99d943697b3a4f7bfe8ef8388156bbe81529f2e551a9938b7326769fe13.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a modern office interior with robotic arm and coffee cups, no visible text or symbols
</details>

Figure 11: LIBERO-100

# C.2 PDDL-based Scene Description File

Here we visualize the whole content of an example scene description file based on PDDL. This file corresponds to the task shown in Figure 2.

Example task: Open the top drawer of the cabinet and put the bowl in it.   
```lisp
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language open the top drawer of the cabinet and put the bowl in it)
    (:regions
    (wooden_cabinet_init_region
    (:target kitchen_table)
    (:ranges (
    (-0.01 -0.31 0.01 -0.29)
    )
    )
    (:yaw_rotation (
    (3.141592653589793 3.141592653589793)
    )
    )
    )
    (akita_black_bowl_init_region
    (:target kitchen_table)
    (:ranges (
    (-0.025 -0.025 0.025 0.025)
    )
    )
    (:yaw_rotation (
    (0.0 0.0)
    )
    )
    )
    (plate_init_region
    (:target kitchen_table)
    (:ranges (
    (-0.025 0.225 0.025 0.275)
    )
    )
    (:yaw_rotation (
    (0.0 0.0)
    )
    )
    )
    (top_side
    (:target wooden_cabinet_1)
    )
    (top_region
    (:target wooden_cabinet_1)
    )
    (middle_region
    (:target wooden_cabinet_1)
    )
    (bottom_region
    (:target wooden_cabinet_1)
    )
)

(:fixtureses
kitchen_table - kitchen_table 
```

```lisp
wooden_cabinet_1 - wooden_cabinet
)
(: objects
akita_black_bowl_1 - akita_black_bowl
plate_1 - plate
)
(: obj_of_interest
wooden_cabinet_1
akita_black_bowl_1
)
(: init
(On akita_black_bowl_1 kitchen_table_akita_black_bowl_init_region)
(On plate_1 kitchen_table_plate_init_region)
(On wooden_cabinet_1 kitchen_table_wooden_cabinet_init_region)
)
(: goal
(And (Open wooden_cabinet_1_top_region)
(In akita_black_bowl_1 wooden_cabinet_1_top_region)
)
) 
```

# D Experimental Setup

We consider five lifelong learning algorithms: SEQL the sequential learning baseline where the agent learns each task in the sequence directly without any further consideration, MTL the multitask learning baseline where the agent learns all tasks in the sequence simultaneously, the regularization-based method EWC [33], the replay-based method ER [13], and the dynamic architecture-based method PACKNET [41]. SEQL and MTL can be seen as approximations of the lower and upper bounds respectively for any lifelong learning algorithm. The other three methods represent the three primary categories of lifelong learning algorithms. For the neural architectures, we consider three vision-language policy architectures: RESNET-RNN, RESNET-T, VIT-T, which differ in how spatial or temporal information is aggregated (See Appendix A.1 for more details). For each task, the agent is trained over 50 epochs on the 50 demonstration trajectories. We evaluate the agent's average success rate over 20 test rollout trajectories of a maximum length of 600 every 5 epochs. We use Adam optimizer [32] with a batch size of 32, and a cosine scheduled learning rate from 0.0001 to 0.00001 for each task. Following the convention of Robomimic [42], we pick the model checkpoint that achieves the best success rate as the final policy for a given task. After 50 epochs of training, the agent with the best checkpoint is then evaluated on all previously learned tasks, with 20 test rollout trajectories for each task. All policy networks are matched in Floating Point Operations Per Second (FLOPS): all policy architectures have $\sim$ 13.5G FLOPS. For each combination of algorithm, policy architecture, and task suite, we run the lifelong learning method 3 times with random seeds {100, 200, 300} (180 experiments in total). See Table 4 for the implemented algorithms and architectures.

# E Additional Experiment Results

# E.1 Full Results

We provide the full results across three different lifelong learning algorithms (e.g., EWC, ER, PACKNET) and three different policy architectures (e.g., RESNET-RNN, RESNET-T, VIT-T) on the four task suites in Table 8.

To better illustrate the performance of each lifelong learning agent throughout the learning process, we present plots that show how the agent's performance evolves over the stream of tasks. Firstly, we provide plots that compare the performance of the agent using different lifelong learning algorithms while fixing the policy architecture (refer to Figure 12,13, and 14). Next, we provide plots that compare the performance of the agent using different policy architectures while fixing the lifelong learning algorithm (refer to Figure15, 16, and 17)

<table><tr><td>Algo.</td><td>Policy Arch.</td><td>FWT(↑)</td><td>NBT(↓)</td><td>AUC(↑)</td><td>FWT(↑)</td><td>NBT(↓)</td><td>AUC(↑)</td></tr><tr><td></td><td></td><td colspan="3">LIBERO-LONG</td><td colspan="3">LIBERO-SPATIAL</td></tr><tr><td rowspan="3">SeQL</td><td>RESNET-RNN</td><td>0.24 ± 0.02</td><td>0.28 ± 0.01</td><td>0.07 ± 0.01</td><td>0.50 ± 0.01</td><td>0.61 ± 0.01</td><td>0.14 ± 0.01</td></tr><tr><td>RESNET-T</td><td>0.54 ± 0.01</td><td>0.63 ± 0.01</td><td>0.15 ± 0.00</td><td>0.72 ± 0.01</td><td>0.81 ± 0.01</td><td>0.20 ± 0.01</td></tr><tr><td>ViT-T</td><td>0.44 ± 0.04</td><td>0.50 ± 0.05</td><td>0.13 ± 0.01</td><td>0.63 ± 0.02</td><td>0.76 ± 0.01</td><td>0.16 ± 0.01</td></tr><tr><td rowspan="3">ER</td><td>RESNET-RNN</td><td>0.16 ± 0.02</td><td>0.16 ± 0.02</td><td>0.08 ± 0.01</td><td>0.40 ± 0.02</td><td>0.29 ± 0.02</td><td>0.29 ± 0.01</td></tr><tr><td>RESNET-T</td><td>0.48 ± 0.02</td><td>0.32 ± 0.04</td><td>0.32 ± 0.01</td><td>0.65 ± 0.03</td><td>0.27 ± 0.03</td><td>0.56 ± 0.01</td></tr><tr><td>ViT-T</td><td>0.38 ± 0.05</td><td>0.29 ± 0.06</td><td>0.25 ± 0.02</td><td>0.63 ± 0.01</td><td>0.29 ± 0.02</td><td>0.50 ± 0.02</td></tr><tr><td rowspan="3">EWC</td><td>RESNET-RNN</td><td>0.02 ± 0.00</td><td>0.04 ± 0.01</td><td>0.00 ± 0.00</td><td>0.14 ± 0.02</td><td>0.23 ± 0.02</td><td>0.03 ± 0.00</td></tr><tr><td>RESNET-T</td><td>0.13 ± 0.02</td><td>0.22 ± 0.03</td><td>0.02 ± 0.00</td><td>0.23 ± 0.01</td><td>0.33 ± 0.01</td><td>0.06 ± 0.01</td></tr><tr><td>ViT-T</td><td>0.05 ± 0.02</td><td>0.09 ± 0.03</td><td>0.01 ± 0.00</td><td>0.32 ± 0.03</td><td>0.48 ± 0.03</td><td>0.06 ± 0.01</td></tr><tr><td rowspan="3">PACKNET</td><td>RESNET-RNN</td><td>0.13 ± 0.00</td><td>0.21 ± 0.01</td><td>0.03 ± 0.00</td><td>0.27 ± 0.03</td><td>0.38 ± 0.03</td><td>0.06 ± 0.01</td></tr><tr><td>RESNET-T</td><td>0.22 ± 0.01</td><td>0.08 ± 0.01</td><td>0.25 ± 0.00</td><td>0.55 ± 0.01</td><td>0.07 ± 0.02</td><td>0.63 ± 0.00</td></tr><tr><td>ViT-T</td><td>0.36 ± 0.01</td><td>0.14 ± 0.01</td><td>0.34 ± 0.01</td><td>0.57 ± 0.04</td><td>0.15 ± 0.00</td><td>0.59 ± 0.03</td></tr><tr><td rowspan="3">MTL</td><td>RESNET-RNN</td><td></td><td></td><td>0.20 ± 0.01</td><td></td><td></td><td>0.61 ± 0.00</td></tr><tr><td>RESNET-T</td><td></td><td></td><td>0.48 ± 0.01</td><td></td><td></td><td>0.83 ± 0.00</td></tr><tr><td>ViT-T</td><td></td><td></td><td>0.46 ± 0.00</td><td></td><td></td><td>0.79 ± 0.01</td></tr><tr><td></td><td></td><td colspan="3">LIBERO-OBJECT</td><td colspan="3">LIBERO-GOAL</td></tr><tr><td rowspan="3">SeQL</td><td>RESNET-RNN</td><td>0.48 ± 0.03</td><td>0.53 ± 0.04</td><td>0.15 ± 0.01</td><td>0.61 ± 0.01</td><td>0.73 ± 0.01</td><td>0.16 ± 0.00</td></tr><tr><td>RESNET-T</td><td>0.78 ± 0.04</td><td>0.76 ± 0.04</td><td>0.26 ± 0.02</td><td>0.77 ± 0.01</td><td>0.82 ± 0.01</td><td>0.22 ± 0.00</td></tr><tr><td>ViT-T</td><td>0.76 ± 0.03</td><td>0.73 ± 0.03</td><td>0.27 ± 0.02</td><td>0.75 ± 0.01</td><td>0.85 ± 0.01</td><td>0.20 ± 0.01</td></tr><tr><td rowspan="3">ER</td><td>RESNET-RNN</td><td>0.30 ± 0.01</td><td>0.27 ± 0.05</td><td>0.17 ± 0.05</td><td>0.41 ± 0.00</td><td>0.35 ± 0.01</td><td>0.26 ± 0.01</td></tr><tr><td>RESNET-T</td><td>0.67 ± 0.07</td><td>0.43 ± 0.04</td><td>0.44 ± 0.06</td><td>0.64 ± 0.01</td><td>0.34 ± 0.02</td><td>0.49 ± 0.02</td></tr><tr><td>ViT-T</td><td>0.70 ± 0.02</td><td>0.28 ± 0.01</td><td>0.57 ± 0.01</td><td>0.57 ± 0.00</td><td>0.40 ± 0.02</td><td>0.38 ± 0.01</td></tr><tr><td rowspan="3">EWC</td><td>RESNET-RNN</td><td>0.17 ± 0.04</td><td>0.23 ± 0.04</td><td>0.06 ± 0.01</td><td>0.16 ± 0.01</td><td>0.22 ± 0.01</td><td>0.06 ± 0.01</td></tr><tr><td>RESNET-T</td><td>0.56 ± 0.03</td><td>0.69 ± 0.02</td><td>0.16 ± 0.02</td><td>0.32 ± 0.02</td><td>0.48 ± 0.03</td><td>0.06 ± 0.00</td></tr><tr><td>ViT-T</td><td>0.57 ± 0.03</td><td>0.64 ± 0.03</td><td>0.23 ± 0.00</td><td>0.32 ± 0.04</td><td>0.45 ± 0.04</td><td>0.07 ± 0.01</td></tr><tr><td rowspan="3">PACKNET</td><td>RESNET-RNN</td><td>0.29 ± 0.02</td><td>0.35 ± 0.02</td><td>0.13 ± 0.01</td><td>0.32 ± 0.03</td><td>0.37 ± 0.04</td><td>0.11 ± 0.01</td></tr><tr><td>RESNET-T</td><td>0.60 ± 0.07</td><td>0.17 ± 0.05</td><td>0.60 ± 0.05</td><td>0.63 ± 0.02</td><td>0.06 ± 0.01</td><td>0.75 ± 0.01</td></tr><tr><td>ViT-T</td><td>0.58 ± 0.03</td><td>0.18 ± 0.02</td><td>0.56 ± 0.04</td><td>0.69 ± 0.02</td><td>0.08 ± 0.01</td><td>0.76 ± 0.02</td></tr><tr><td rowspan="3">MTL</td><td>RESNET-RNN</td><td></td><td></td><td>0.10 ± 0.03</td><td></td><td></td><td>0.59 ± 0.00</td></tr><tr><td>RESNET-T</td><td></td><td></td><td>0.54 ± 0.02</td><td></td><td></td><td>0.80 ± 0.01</td></tr><tr><td>ViT-T</td><td></td><td></td><td>0.78 ± 0.02</td><td></td><td></td><td>0.82 ± 0.01</td></tr></table>

Table 8: We present the full results of all networks and algorithms on all four task suites. For each task suite, we highlight the top three AUC scores among the combinations of the three lifelong learning algorithms and the three neural architectures. The best three results are highlighted in magenta (the best), light magenta (the second best), and super light magenta (the third best), respectively.

![](images/3d81be65a0e949bb38415ba092d650f204afceeaced8c1332c7df3b45c971d1f.jpg)

<details>
<summary>line</summary>

| Task    | Multitask | Sequential | ER   | EWC  | PackNet |
|---------|-----------|------------|------|------|---------|
| Task 1  | 0.2       | 0.3        | 0.4  | 0.5  | 0.6     |
| Task 2  | 0.1       | 0.2        | 0.3  | 0.4  | 0.5     |
| Task 3  | 0.3       | 0.4        | 0.5  | 0.6  | 0.7     |
| Task 4  | 0.4       | 0.5        | 0.6  | 0.7  | 0.8     |
| Task 5  | 0.5       | 0.6        | 0.7  | 0.8  | 0.9     |
| Task 6  | 0.6       | 0.7        | 0.8  | 0.9  | 1.0     |
| Task 7  | 0.7       | 0.8        | 0.9  | 1.0  | 1.1     |
| Task 8  | 0.8       | 0.9        | 1.0  | 1.1  | 1.2     |
| Task 9  | 0.9       | 1.0        | 1.1  | 1.2  | 1.3     |
| Task 10 | 1.0       | 1.1        | 1.2  | 1.3  | 1.4     |
</details>

Figure 12: We compare the performance of different algorithms using the RESNET-RNN policy architecture in Figure 12. The y-axis represents the success rate, and the x-axis shows the agent's performance on each of the 10 tasks in a specific task suite over the course of learning. For example, the upper-left plot in the figure displays the agent's performance on the first task as it learns the 10 tasks sequentially.

![](images/2d7be9ac46c765ab09063fbf079686b0e3f5e7ef9699d4c7457d9a0bfa10e9d3.jpg)

<details>
<summary>line</summary>

| Task    | Multitask | Sequential | ER   | EWC  | PackNet |
|---------|-----------|------------|------|------|---------|
| Task 1  | 0.8       | 0.9        | 0.7  | 0.6  | 0.5     |
| Task 2  | 0.7       | 0.8        | 0.6  | 0.5  | 0.4     |
| Task 3  | 0.9       | 0.7        | 0.5  | 0.4  | 0.3     |
| Task 4  | 0.8       | 0.6        | 0.4  | 0.3  | 0.2     |
| Task 5  | 0.9       | 0.8        | 0.6  | 0.5  | 0.4     |
| Task 6  | 0.7       | 0.6        | 0.4  | 0.3  | 0.2     |
| Task 7  | 0.8       | 0.7        | 0.5  | 0.4  | 0.3     |
| Task 8  | 0.9       | 0.8        | 0.6  | 0.5  | 0.4     |
| Task 9  | 0.7       | 0.6        | 0.4  | 0.3  | 0.2     |
| Task 10 | 0.8       | 0.9        | 0.7  | 0.6  | 0.5     |
</details>

Figure 13: Comparison of different algorithms using the RESNET-T policy architecture. The y-axis represents the success rate, while the x-axis shows the agent's performance on each of the 10 tasks in a given task suite during the course of learning. For example, the plot in the upper-left corner depicts the agent's performance on the first task as it learns the 10 tasks sequentially.

![](images/1db97ce502be03de281475b6743577732b588e7f7364a3af82bdef34e287da32.jpg)

<details>
<summary>line</summary>

| Task    | Multitask | Sequential | ER   | EWC  | PackNet |
|---------|-----------|------------|------|------|---------|
| Task 1  | 0.4       | 0.3        | 0.2  | 0.1  | 0.3     |
| Task 2  | 0.8       | 0.7        | 0.6  | 0.5  | 0.6     |
| Task 3  | 0.9       | 0.8        | 0.7  | 0.6  | 0.7     |
| Task 4  | 0.7       | 0.6        | 0.5  | 0.4  | 0.5     |
| Task 5  | 0.8       | 0.7        | 0.6  | 0.5  | 0.6     |
| Task 6  | 0.9       | 0.8        | 0.7  | 0.6  | 0.7     |
| Task 7  | 0.7       | 0.6        | 0.5  | 0.4  | 0.5     |
| Task 8  | 0.8       | 0.7        | 0.6  | 0.5  | 0.6     |
| Task 9  | 0.9       | 0.8        | 0.7  | 0.6  | 0.7     |
| Task 10 | 0.7       | 0.6        | 0.5  | 0.4  | 0.5     |
</details>

Figure 14: Comparison of different algorithms using the VIT-T policy architecture. The success rate is represented on the $y$ -axis, while the $x$ -axis shows the agent's performance on the 10 tasks in a given task suite over the course of learning. For instance, the plot in the upper-left corner illustrates the agent's performance on the first task when learning the 10 tasks sequentially.

![](images/26610b4ff898e05b8572f3b0b6010760f24e4d6d7afcff94309823b24e934282.jpg)

<details>
<summary>line</summary>

| Task | Multitask | ResNet-Transformer | ViT-Transformer | ResNet-RNN |
|------|-----------|---------------------|-----------------|------------|
| Task 1 | 0.85 | 0.95 | 0.75 | 0.65 |
| Task 2 | 0.70 | 0.85 | 0.60 | 0.50 |
| Task 3 | 0.65 | 0.80 | 0.55 | 0.45 |
| Task 4 | 0.55 | 0.75 | 0.45 | 0.35 |
| Task 5 | 0.45 | 0.65 | 0.35 | 0.25 |
| Task 6 | 0.35 | 0.55 | 0.25 | 0.15 |
| Task 7 | 0.25 | 0.45 | 0.15 | 0.05 |
| Task 8 | 0.15 | 0.35 | 0.05 | 0.02 |
| Task 9 | 0.05 | 0.25 | 0.02 | 0.01 |
| Task 10 | 0.02 | 0.15 | 0.01 | 0.01 |
</details>

Figure 15: Comparison of different architectures with the EWC algorithm. The y-axis is the success rate, while the x-axis shows the agent's performance on the 10 tasks in a given task suite over the course of learning. For instance, the upper-left plot shows the agent's performance on the first task when learning the 10 tasks sequentially.

![](images/f292326cb852f63dab8bfdf6624b060b100b2a978f2cfb9ceed79bc8fea6f6cc.jpg)

<details>
<summary>line</summary>

| Model          | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
|----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| Multitask      | 0.8    | 0.6    | 0.7    | 0.5    | 0.9    | 0.4    | 0.6    | 0.7    | 0.8    | 0.9     |
| ResNet-Transformer | 0.7   | 0.5    | 0.6    | 0.4    | 0.8    | 0.3    | 0.5    | 0.6    | 0.7    | 0.8     |
| ViT-Transformer  | 0.6    | 0.4    | 0.5    | 0.3    | 0.7    | 0.2    | 0.4    | 0.5    | 0.6    | 0.7     |
| ResNet-RNN     | 0.5    | 0.3    | 0.4    | 0.2    | 0.6    | 0.1    | 0.3    | 0.4    | 0.5    | 0.6     |
</details>

Figure 16: Comparison of different architectures with the ER algorithm. The y-axis is the success rate, while the x-axis shows the agent's performance on the 10 tasks in a given task suite ver the course of learning. For instance, the upper-left plot shows the agent's performance on the first task when learning the 10 tasks sequentially.

![](images/dd1950b855822da549cf31d76a3554b4cacd8d8e588507f8bd60e35cbfac247e.jpg)

<details>
<summary>line</summary>

| Task       | Multitask | ResNet-Transformer | ViT-Transformer | ResNet-RNN |
|------------|-----------|--------------------|-----------------|------------|
| Task 1     | 0.3       | 0.2                | 0.4             | 0.1        |
| Task 2     | 0.4       | 0.3                | 0.5             | 0.2        |
| Task 3     | 0.6       | 0.7                | 0.8             | 0.9        |
| Task 4     | 0.8       | 0.9                | 1.0             | 1.1        |
| Task 5     | 0.7       | 0.8                | 0.9             | 1.0        |
| Task 6     | 0.6       | 0.7                | 0.8             | 0.9        |
| Task 7     | 0.5       | 0.6                | 0.7             | 0.8        |
| Task 8     | 0.4       | 0.5                | 0.6             | 0.7        |
| Task 9     | 0.3       | 0.4                | 0.5             | 0.6        |
| Task 10    | 0.2       | 0.3                | 0.4             | 0.5        |
</details>

Figure 17: Comparison of different architectures with the PACKNET algorithm. The y-axis is the success rate, while the x-axis shows the agent's performance on the 10 tasks in a given task suite over the course of learning. For instance, the upper-left plot shows the agent's performance on the first task when learning the 10 tasks sequentially.

# E.2 Loss v.s. Success Rates

We demonstrate that behavioral cloning loss can be a misleading indicator of task success rate in this section. In supervised learning tasks like image classifications, lower loss often indicates better prediction accuracy. However, this is not, in general, true for decision-making tasks. This is because errors can compound until failures during executing a robot $[55]$ . Figure 18, 13 and 14 plots the training loss and success rates of three lifelong learning methods (ER, EWC, and PACKNET) for comparison. We evaluate the three algorithms on four task suites using three different neural architectures.

Findings: We observe that though sometimes EWC has the lowest loss, it did not achieve good success rate. ER, on the other hand, can have the highest loss but perform better than EWC. In conclusion, success rates, instead of behavioral cloning loss, should be the right metric to evaluate whether a model checkpoint is good or not.

![](images/9a1217b08cbdc222a227991417e84dac49f32febe8ea68fb9e6ebb76a51615f8.jpg)  
Figure 18: Losses and success rates of ER (violet), EWC (grey), and PACKNET (blue) on four task suites with RESNET-RNN policy. The first (second) row shows the loss (success rate) of the agent on task i throughout the LLDM procedure.

![](images/ac0abf292fe06ad21782674d525a7926a3003e8a90783495273707cd1473712f.jpg)

<details>
<summary>line</summary>

| Task   | ER Loss | EWC Loss | PackNet Loss | ER Success Rate | EWC Success Rate | PackNet Success Rate |
|--------|---------|----------|--------------|-----------------|------------------|----------------------|
| Task 1 | -10     | -20      | -10          | 0.8             | 0.6              | 0.4                  |
| Task 2 | -5      | -15      | -5           | 0.6             | 0.4              | 0.2                  |
| Task 3 | 0       | -10      | 0            | 0.9             | 0.7              | 0.5                  |
| Task 4 | 5       | -5       | 5            | 1.0             | 0.8              | 0.6                  |
| Task 5 | 0       | -10      | 0            | 0.4             | 0.2              | 0.1                  |
| Task 6 | -5      | -15      | -5           | 0.8             | 0.6              | 0.4                  |
| Task 7 | 0       | -10      | 0            | 0.6             | 0.4              | 0.2                  |
| Task 8 | -5      | -15      | -5           | 0.6             | 0.4              | 0.2                  |
| Task 9 | 0       | -10      | 0            | 0.4             | 0.2              | 0.1                  |
| Task 10| -5      | -15      | -5           | 0.8             | 0.6              | 0.4                  |
</details>

![](images/3e8501f7d9abcc2f6f4471af7df0f68ef071d0f0856488809635ac046da2dcd4.jpg)

<details>
<summary>line</summary>

| Task   | Loss  | Success Rate |
|--------|-------|--------------|
| Task 1 | -20   | 1.0          |
| Task 2 | -10   | 0.8          |
| Task 3 | -5    | 1.0          |
| Task 4 | 0     | 1.0          |
| Task 5 | 5     | 0.8          |
| Task 6 | 10    | 1.0          |
| Task 7 | 5     | 1.0          |
| Task 8 | 0     | 1.0          |
| Task 9 | -5    | 0.8          |
| Task 10| -20   | 1.0          |
</details>

![](images/3ecbd61f1505531eabda5d01e2707e02038cf095f68733fe2ef0b439527b522f.jpg)

<details>
<summary>line</summary>

| Task   | Loss  | Success Rate |
|--------|-------|--------------|
| Task 1 | -20   | 0.8          |
| Task 2 | -15   | 0.6          |
| Task 3 | -10   | 0.4          |
| Task 4 | -5    | 0.2          |
| Task 5 | 0     | 0.0          |
| Task 6 | 5     | -0.2         |
| Task 7 | 10    | -0.4         |
| Task 8 | 15    | -0.6         |
| Task 9 | 20    | -0.8         |
| Task 10| 25    | -1.0         |
</details>

![](images/29c184a5827227ee9c9a9f58ae4631711a5f05181ddb77783d8055a6cdbfb19a.jpg)

<details>
<summary>line</summary>

| Task   | Loss  | Success Rate |
|--------|-------|--------------|
| Task 1 | -20   | 0.8          |
| Task 2 | -5    | 0.6          |
| Task 3 | -10   | 0.4          |
| Task 4 | -15   | 0.2          |
| Task 5 | -20   | 0.0          |
| Task 6 | -25   | -0.2         |
| Task 7 | -30   | -0.4         |
| Task 8 | -35   | -0.6         |
| Task 9 | -40   | -0.8         |
| Task 10| -45   | -1.0         |
</details>

Figure 19: Losses and success rates of ER (violet), EWC (grey), and PACKNET (blue) on four task suites with RESNET-T policy. The first (second) row shows the loss (success rate) of the agent on task i throughout the LLDM procedure.

![](images/e7a398ed2189562da72661a86be8285b20a9e5a8f453284facec793d65b9c44f.jpg)

<details>
<summary>line</summary>

| Task   | ER Loss | EWC Loss | PackNet Loss | ER Success Rate | EWC Success Rate | PackNet Success Rate |
|--------|---------|----------|--------------|-----------------|------------------|----------------------|
| Task 1 | -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 2 | -10     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 3 | -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 4 | -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 5 | -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 6 | -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 7 | -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 8 | -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 9 | -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
| Task 10| -20     | -10      | -10          | 0.4             | 0.6              | 0.8                  |
</details>

![](images/2a6b714a2d8bde3dd88aafb3a1f2f4ace8a464810167078ce3500985047a1f27.jpg)

<details>
<summary>line</summary>

| Task   | Loss  | Success Rate |
|--------|-------|--------------|
| Task 1 | -10.0 | 0.8          |
| Task 2 | -5.0  | 0.6          |
| Task 3 | -15.0 | 0.4          |
| Task 4 | -20.0 | 0.2          |
| Task 5 | -10.0 | 0.6          |
| Task 6 | -5.0  | 0.4          |
| Task 7 | 0.0   | 0.8          |
| Task 8 | 5.0   | 0.6          |
| Task 9 | -10.0 | 0.4          |
| Task 10| -5.0  | 0.2          |
</details>

![](images/935d4dcaf9bd0748adcfb1bfe955762c7619ba0f47c09e60a069bd59daf56bd0.jpg)

<details>
<summary>line</summary>

| Task   | Loss  | Success Rate |
|--------|-------|--------------|
| Task 1 | -5    | 1.0          |
| Task 2 | -10   | 0.8          |
| Task 3 | -15   | 1.0          |
| Task 4 | -20   | 0.6          |
| Task 5 | -25   | 1.0          |
| Task 6 | -30   | 0.8          |
| Task 7 | -35   | 1.0          |
| Task 8 | -40   | 0.6          |
| Task 9 | -45   | 1.0          |
| Task 10| -50   | 0.8          |
</details>

![](images/22f3ecf5f9b24ac19252ea2c329a5575d4452a4ca401794515bda5565f341966.jpg)

<details>
<summary>line</summary>

| Task   | Loss  | Success Rate |
|--------|-------|--------------|
| Task 1 | -5    | 1.0          |
| Task 2 | -10   | 0.8          |
| Task 3 | -15   | 0.6          |
| Task 4 | -20   | 0.4          |
| Task 5 | -25   | 0.2          |
| Task 6 | -30   | 0.0          |
| Task 7 | -35   | -0.2         |
| Task 8 | -40   | -0.4         |
| Task 9 | -45   | -0.6         |
| Task 10| -50   | -0.8         |
</details>

Figure 20: Losses and success rates of ER (violet), EWC (grey), and PACKNET (blue) on four task suites with V1T-T policy. The first (second) row shows the loss (success rate) of the agent on task i throughout the LLDM procedure.

# E.3 Multitask Success Rate

<table><tr><td>Env Name</td><td>RESNET-RNN</td><td>RESNET-T</td><td>V1T-T</td></tr><tr><td>KITCHEN SCENE10 close the top drawer of the cabinet</td><td>0.45</td><td>0.45</td><td>0.6</td></tr><tr><td>KITCHEN SCENE10 close the top drawer of the cabinet and put the black bowl on top of it</td><td>0.1</td><td>0.1</td><td>0.0</td></tr><tr><td>KITCHEN SCENE10 put the black bowl in the top drawer of the cabinet</td><td>0.0</td><td>0.25</td><td>0.05</td></tr><tr><td>KITCHEN SCENE10 put the butter at the back in the top drawer of the cabinet and close it</td><td>0.15</td><td>0.0</td><td>0.0</td></tr><tr><td>KITCHEN SCENE10 put the butter at the front in the top drawer of the cabinet and close it</td><td>0.2</td><td>0.0</td><td>0.0</td></tr><tr><td>KITCHEN SCENE10 put the chocolate pudding in the top drawer of the cabinet and close it</td><td>0.25</td><td>0.0</td><td>0.0</td></tr><tr><td>KITCHEN SCENE1 open the bottom drawer of the cabinet</td><td>0.05</td><td>0.0</td><td>0.15</td></tr><tr><td>KITCHEN SCENE1 open the top drawer of the cabinet</td><td>0.3</td><td>0.45</td><td>0.25</td></tr><tr><td>KITCHEN SCENE1 open the top drawer of the cabinet and put the bowl in it</td><td>0.05</td><td>0.0</td><td>0.0</td></tr><tr><td>KITCHEN SCENE1 put the black bowl on the plate</td><td>0.35</td><td>0.3</td><td>0.0</td></tr><tr><td>KITCHEN SCENE1 put the black bowl on top of the cabinet</td><td>0.2</td><td>0.7</td><td>0.15</td></tr><tr><td>KITCHEN SCENE2 open the top drawer of the cabinet</td><td>0.45</td><td>0.4</td><td>0.65</td></tr><tr><td>KITCHEN SCENE2 put the black bowl at the back on the plate</td><td>0.2</td><td>0.05</td><td>0.35</td></tr><tr><td>KITCHEN SCENE2 put the black bowl at the front on the plate</td><td>0.1</td><td>0.35</td><td>0.0</td></tr><tr><td>KITCHEN SCENE2 put the middle black bowl on the plate</td><td>0.35</td><td>0.1</td><td>0.0</td></tr><tr><td>KITCHEN SCENE2 put the middle black bowl on top of the cabinet</td><td>0.5</td><td>0.1</td><td>0.6</td></tr><tr><td>KITCHEN SCENE2 stack the black bowl at the front on the black bowl in the middle</td><td>0.25</td><td>0.05</td><td>0.25</td></tr><tr><td>KITCHEN SCENE2 stack the middle black bowl on the back black bowl</td><td>0.05</td><td>0.05</td><td>0.0</td></tr><tr><td>KITCHEN SCENE3 put the frying pan on the stove</td><td>0.4</td><td>0.3</td><td>0.35</td></tr><tr><td>KITCHEN SCENE3 put the moka pot on the stove</td><td>0.15</td><td>0.0</td><td>0.4</td></tr><tr><td>KITCHEN SCENE3 turn on the stove</td><td>0.6</td><td>0.7</td><td>1.0</td></tr><tr><td>KITCHEN SCENE3 turn on the stove and put the frying pan on it</td><td>0.15</td><td>0.0</td><td>0.35</td></tr><tr><td>KITCHEN SCENE4 close the bottom drawer of the cabinet</td><td>0.75</td><td>0.4</td><td>0.55</td></tr><tr><td>KITCHEN SCENE4 close the bottom drawer of the cabinet and open the top drawer</td><td>0.2</td><td>0.0</td><td>0.05</td></tr><tr><td>KITCHEN SCENE4 put the black bowl in the bottom drawer of the cabinet</td><td>0.25</td><td>0.3</td><td>0.25</td></tr><tr><td>KITCHEN SCENE4 put the black bowl on top of the cabinet</td><td>0.8</td><td>0.6</td><td>0.85</td></tr><tr><td>KITCHEN SCENE4 put the wine bottle in the bottom drawer of the cabinet</td><td>0.0</td><td>0.0</td><td>0.35</td></tr><tr><td>KITCHEN SCENE4 put the wine bottle on the wine rack</td><td>0.05</td><td>0.0</td><td>0.2</td></tr><tr><td>KITCHEN SCENE5 close the top drawer of the cabinet</td><td>0.05</td><td>0.8</td><td>0.9</td></tr><tr><td>KITCHEN SCENE5 put the black bowl in the top drawer of the cabinet</td><td>0.2</td><td>0.1</td><td>0.15</td></tr><tr><td>KITCHEN SCENE5 put the black bowl on the plate</td><td>0.05</td><td>0.05</td><td>0.1</td></tr><tr><td>KITCHEN SCENE5 put the black bowl on top of the cabinet</td><td>0.5</td><td>0.25</td><td>0.2</td></tr><tr><td>KITCHEN SCENE5 put the ketchup in the top drawer of the cabinet</td><td>0.0</td><td>0.0</td><td>0.05</td></tr><tr><td>KITCHEN SCENE6 close the microwave</td><td>0.1</td><td>0.1</td><td>0.2</td></tr><tr><td>KITCHEN SCENE6 put the yellow and white mug to the front of the white mug</td><td>0.2</td><td>0.05</td><td>0.1</td></tr><tr><td>KITCHEN SCENE7 open the microwave</td><td>0.7</td><td>0.1</td><td>0.45</td></tr><tr><td>KITCHEN SCENE7 put the white bowl on the plate</td><td>0.05</td><td>0.0</td><td>0.05</td></tr><tr><td>KITCHEN SCENE7 put the white bowl to the right of the plate</td><td>0.05</td><td>0.05</td><td>0.15</td></tr><tr><td>KITCHEN SCENE8 put the right moka pot on the stove</td><td>0.15</td><td>0.0</td><td>0.1</td></tr><tr><td>KITCHEN SCENE8 turn off the stove</td><td>0.2</td><td>0.25</td><td>0.75</td></tr><tr><td>KITCHEN SCENE9 put the frying pan on the cabinet shelf</td><td>0.45</td><td>0.15</td><td>0.05</td></tr><tr><td>KITCHEN SCENE9 put the frying pan on top of the cabinet</td><td>0.25</td><td>0.4</td><td>0.3</td></tr><tr><td>KITCHEN SCENE9 put the frying pan under the cabinet shelf</td><td>0.15</td><td>0.45</td><td>0.15</td></tr><tr><td>KITCHEN SCENE9 put the white bowl on top of the cabinet</td><td>0.1</td><td>0.1</td><td>0.15</td></tr><tr><td>KITCHEN SCENE9 turn on the stove</td><td>0.5</td><td>0.4</td><td>0.95</td></tr><tr><td>KITCHEN SCENE9 turn on the stove and put the frying pan on it</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE1 pick up the alphabet soup and put it in the basket</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE1 pick up the cream cheese box and put it in the basket</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE1 pick up the ketchup and put it in the basket</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE1 pick up the tomato sauce and put it in the basket</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE2 pick up the alphabet soup and put it in the basket</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE2 pick up the butter and put it in the basket</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE2 pick up the milk and put it in the basket</td><td>0.0</td><td>0.05</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE2 pick up the orange juice and put it in the basket</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE2 pick up the tomato sauce and put it in the basket</td><td>0.0</td><td>0.05</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE3 pick up the alphabet soup and put it in the tray</td><td>0.0</td><td>0.05</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE3 pick up the butter and put it in the tray</td><td>0.0</td><td>0.3</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE3 pick up the cream cheese and put it in the tray</td><td>0.0</td><td>0.25</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE3 pick up the ketchup and put it in the tray</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE3 pick up the tomato sauce and put it in the tray</td><td>0.0</td><td>0.25</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE4 pick up the black bowl on the left and put it in the tray</td><td>0.0</td><td>0.4</td><td>0.2</td></tr><tr><td>LIVING ROOM SCENE4 pick up the chocolate pudding and put it in the tray</td><td>0.0</td><td>0.2</td><td>0.25</td></tr><tr><td>LIVING ROOM SCENE4 pick up the salad dressing and put it in the tray</td><td>0.0</td><td>0.0</td><td>0.1</td></tr><tr><td>LIVING ROOM SCENE4 stack the left bowl on the right bowl and place them in the tray</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE4 stack the right bowl on the left bowl and place them in the tray</td><td>0.0</td><td>0.0</td><td>0.05</td></tr><tr><td>LIVING ROOM SCENE5 put the red mug on the left plate</td><td>0.0</td><td>0.0</td><td>0.05</td></tr><tr><td>LIVING ROOM SCENE5 put the red mug on the right plate</td><td>0.15</td><td>0.0</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE5 put the white mug on the left plate</td><td>0.1</td><td>0.15</td><td>0.05</td></tr><tr><td>LIVING ROOM SCENE5 put the yellow and white mug on the right plate</td><td>0.35</td><td>0.05</td><td>0.05</td></tr><tr><td>LIVING ROOM SCENE6 put the chocolate pudding to the left of the plate</td><td>0.1</td><td>0.65</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE6 put the chocolate pudding to the right of the plate</td><td>0.05</td><td>0.55</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE6 put the red mug on the plate</td><td>0.0</td><td>0.2</td><td>0.0</td></tr><tr><td>LIVING ROOM SCENE6 put the white mug on the plate</td><td>0.0</td><td>0.2</td><td>0.0</td></tr><tr><td>STUDY SCENE1 pick up the book and place it in the front compartment of the caddy</td><td>0.0</td><td>0.0</td><td>0.05</td></tr><tr><td>STUDY SCENE1 pick up the book and place it in the left compartment of the caddy</td><td>0.2</td><td>0.05</td><td>0.0</td></tr><tr><td>STUDY SCENE1 pick up the book and place it in the right compartment of the caddy</td><td>0.0</td><td>0.1</td><td>0.0</td></tr><tr><td>STUDY SCENE1 pick up the yellow and white mug and place it to the right of the caddy</td><td>0.0</td><td>0.3</td><td>0.35</td></tr><tr><td>STUDY SCENE2 pick up the book and place it in the back compartment of the caddy</td><td>0.35</td><td>0.3</td><td>0.0</td></tr><tr><td>STUDY SCENE2 pick up the book and place it in the front compartment of the caddy</td><td>0.25</td><td>0.1</td><td>0.0</td></tr><tr><td>STUDY SCENE2 pick up the book and place it in the left compartment of the caddy</td><td>0.25</td><td>0.45</td><td>0.05</td></tr><tr><td>STUDY SCENE2 pick up the book and place it in the right compartment of the caddy</td><td>0.0</td><td>0.2</td><td>0.0</td></tr><tr><td>STUDY SCENE3 pick up the book and place it in the front compartment of the caddy</td><td>0.2</td><td>0.0</td><td>0.0</td></tr><tr><td>STUDY SCENE3 pick up the book and place it in the left compartment of the caddy</td><td>0.4</td><td>0.45</td><td>0.15</td></tr><tr><td>STUDY SCENE3 pick up the book and place it in the right compartment of the caddy</td><td>0.0</td><td>0.05</td><td>0.05</td></tr><tr><td>STUDY SCENE3 pick up the red mug and place it to the right of the caddy</td><td>0.0</td><td>0.2</td><td>0.05</td></tr><tr><td>STUDY SCENE3 pick up the white mug and place it to the right of the caddy</td><td>0.0</td><td>0.0</td><td>0.05</td></tr><tr><td>STUDY SCENE4 pick up the book in the middle and place it on the cabinet shelf</td><td>0.05</td><td>0.2</td><td>0.05</td></tr><tr><td>STUDY SCENE4 pick up the book on the left and place it on top of the shelf</td><td>0.2</td><td>0.1</td><td>0.15</td></tr><tr><td>STUDY SCENE4 pick up the book on the right and place it on the cabinet shelf</td><td>0.0</td><td>0.25</td><td>0.05</td></tr><tr><td>STUDY SCENE4 pick up the book on the right and place it under the cabinet shelf</td><td>0.15</td><td>0.1</td><td>0.05</td></tr></table>

# E.4 Attention Visualization

It is also important to visualize the behavior of the robot and its attention maps during the completion of tasks in the lifelong learning process to give us intuition and qualitative feedback on the performance of different algorithms and architectures. We visualize the attention maps of learned policies with (author?) [23] and compare them in different studies as in 5.2 to see if the robot correctly pays attention to the right regions of interest in each task.

Perturbation-based attention visualization: We use a perturbation-based method [23] to extract attention maps from agents. Given an input image I, the method applies a Gaussian filter to a pixel location $(i,j)$ to blur the image partially, and produces the perturbed image $\Phi(I,i,j)$ . Denote the learned policy as $\pi$ and the inputs to the spatial module (e.g., the last latent representation of resnet or ViT encoder) $\pi_{u}(I)$ for image I. Then we define the saliency score as the Euclidean distance between the latent representations of the original and the blurred images:

$$
S _ {\pi} (i, j) = \frac {1}{2} \left| \left| \pi_ {u} (I) - \pi_ {u} (\Phi (I, i, j)) \right| \right| ^ {2}. \tag {4}
$$

Intuitively, $S_{\pi}(i,j)$ describes how much removing information from the region around location $(i,j)$ changes the policy. In other words, a large $S_{\pi}(i,j)$ indicates that the information around pixel $(i,j)$ is important for the learning agent's decision-making. Instead of calculating the score for every pixel, [23] found that computing a saliency score for pixel $i \mod 5$ and $j \mod 5$ produced good saliency maps at lower computational costs for Atari games. The final saliency map $P$ is normalized as $P(i,j) = \frac{S_{\pi}(i,j)}{\sum_{i,j} S_{\pi}(i,j)}$ .

We provide the visualization and our analysis on the following pages.

# Different Task Suites

# LIBERO-SPATIAL

![](images/9d83c3bf60f05f0280d936fb37253baa739750d68651d93a3c8b81de79b625a0.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a laboratory setup with microscope, glassware, and a circular object (no visible text or symbols)
</details>

pick up the black
bowl on the wooden
cabinet and place it
on the plate

![](images/d6ed25364a6f10578992138114ed594d011e5af604ad8f4af311f29616f24abc.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a microfluidic device with heat signatures (no text or symbols visible)
</details>

pick up the black
bowl in the top
drawer of the
cabinet and place it
on the plate

![](images/213800c1ddd33ef0807063e1eec4d0e1bb8702dbd3ec14a12784b9d36176afdb.jpg)

<details>
<summary>natural_image</summary>

Experimental setup with microscope, sample holder, and glowing color gradient (no visible text or symbols)
</details>

pick up the black
bowl between the
plate and the
ramekin and place it
on the plate

# LIBERO-OBJECT

![](images/739c9c100cac46c4d1fd742c5c31f0ad0321f79e62e69e2f46a3578feb2aa4f0.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a room with objects and a basket, showing heat signatures (no text or symbols)
</details>

pick up the orange juice and place it in the basket

![](images/d45b96b26daf57988655535cb6a2bae100c5191ad921e3b9f425f46193d29202.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a tiled room with objects and a basket, no visible text or symbols
</details>

pick up the ketchup and place it in the basket

![](images/295f4a6c586257c65ef5b091c6efc4eb2c3fdf5bb12b9387ed7e82d9a0ee5d1b.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a room with lighting sources and equipment (no visible text or symbols)
</details>

pick up the alphabet soup and place it in the basket

# LIBERO-GOAL

![](images/cd9612a2cdffa8b3e7b5faf5ef1e635d5d87da3ed47665be6a188ad40243b2b6.jpg)

<details>
<summary>natural_image</summary>

Laboratory setup with microscope, gas bath, and microscope equipment (no visible text or labels)
</details>

put the wine bottle on the rack

![](images/59846b776d289ddb8279918a8f7c5bf7f2d8cbfa461768889ff5cf3c2b1b4f9f.jpg)

<details>
<summary>natural_image</summary>

Laboratory setup with microscope, beakers, and a color-coded thermal image (no visible text or symbols)
</details>

put the bowl on top of the cabinet

![](images/6df7f40d9c44b576bb18ad438b81a7159994323e86d39a0b2583ffc2869a437d.jpg)

<details>
<summary>natural_image</summary>

Laboratory microscope setup with microscope, beakers, and a thermal imaging display (no visible text or symbols)
</details>

open the middle
drawer of the
cabinet

# LIBERO-LONG

![](images/485f2c9208dc1a3543a9a9fc65474d42e0d1e130bb9244e10ba9c1560b4e3a08.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a coffee machine setup with two mugs and a cup on a dark surface (no text or symbols visible)
</details>

put the white mug on
the left plate and
put the yellow and
white mug on the
right plate

![](images/78e81589465a8138e9a74d8ab65f12e129f3951b02645c557b27dceac4aaf2ad.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view showing heat distribution over a surface with multiple rectangular objects (no text or symbols visible)
</details>

put both the alphabet soup and the tomato sauce in the basket

![](images/259159555a131eccd720a35086a62ee9c53ef47256e774a2d35ca59ef66457c7.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a laboratory setup with three mugs and a glassware unit (no visible text or symbols)
</details>

put the yellow and white mug in the microwave and close it

Figure 21: Attention map comparison among different task suites with ER and RESNET-T. Each row corresponds to a task suite.

Findings: Figure 21 shows attention visualization for 12 tasks across 4 task suites (e.g., 3 tasks per suite). We observe that:

1. policies pay more attention to the robot arm and the target placement area than the target object.   
2. sometimes the policy pays attention to task-irrelevant areas, such as the blank area on the table.

These observations demonstrate that the learned policy use perceptual data for decision-making in a very different way from how humans do. The robot policies tends to spuriously correlate task-irrelevant features with actions, a major reason why the policies overfit to the tasks and do not generalize well across tasks.

# The Same Task over the Course of Lifelong Learning

put both the alphabet soup and the tomato sauce in the basket

![](images/d6035c89bbb31bb33963bfdf496e312029c35783132ca993b115309098f3cdda.jpg)

<details>
<summary>text_image</summary>

ER
ResNet-T
ER
ViT-T
PackNet
ResNet-T
PackNet
ViT-T
Initial Policy
After 3rd Task
After 5th Task
After 10th Task
</details>

Figure 22: Attention map of the same state of the task put both the alphabet soup and the tomato sauce in the basket from LIBERO-LONG during lifelong learning. Each row visualizes how the attention maps change on the first task with one of the LL algorithms (ER and PACKNET) and one of the neural architectures (RESNET-T and VIT-T). Initial policy is the policy that is trained on the first task. And all the following attention maps correspond to policies after training on the third, fifth, and the tenth tasks.

Findings: Figure 22 shows attention visualizations from policies trained with ER and PACKNET using the architectures RESNET-T and ViT-T respectively. We observe that:

1. The ViT visual encoder's attention is more consistent over time, while the ResNet encoder's attention map gradually dilutes.   
2. PackNet, as it splits the model capacity for different tasks, shows a more consistent attention map over the course of learning.

# Different Lifelong Learning Algorithms

Policy after training task 5 on task 5   
![](images/3fc444f8acaf22ce55dfdb95ea6a92039f1038c557530dc33ad11415f5428936.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of three mugs on a dark surface with blurred background (no text or symbols)
</details>

![](images/68b837beb1e091b6306809012b4ff1a95a507ef863961473bb4dc8e73b2c8007.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view showing heat distribution with a yellow-orange hot spot on a surface, surrounded by three mugs (no text or symbols visible)
</details>

![](images/565c2644927fc5d64352b1b08f04953cb7a6870585582ef24feeba44e4d2efae.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a coffee machine with three mugs on a dark surface (no text or symbols visible)
</details>

Policy after training task 10 on task 5   
![](images/d837d71af55374a188b628b1c0928a134488a9f73d6493a81521e26d061b9cab.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of two mugs on a dark surface with blurred background (no text or symbols)
</details>

![](images/59a9fba719f7361c2b06e9dbfd5637fd0030e95ab450c2116563d9bffdabe74e.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a robotic arm with yellow-orange heat source, surrounded by white cups on a dark surface (no text or symbols)
</details>

![](images/7823e7080083f45f17b706d932e6ebe8d00b0e750e34c99916dcc9a4111accd4.jpg)

<details>
<summary>natural_image</summary>

Thermal image of coffee cups and a coffee cup on a dark surface, with no visible text or symbols
</details>

Policy after training task 10 on task 10   
![](images/13218f4c02dea38134bed96a6699860ffa5f46eb7705f9d0224ab7852b0738bb.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a kitchen setup with three mugs and a refrigerator (no text or symbols visible)
</details>

ER

![](images/3f0ef9f1e57b1ab569a6a0b256ab80e99efa7c3b3d3ece2b981c85352bc09c6d.jpg)

<details>
<summary>natural_image</summary>

3D-rendered scene with a robot, two mugs, and a monitor on a table (no visible text or symbols)
</details>

PackNet

![](images/a11af016a7c4b435018fc31e07b8f909b2c4df967055e452b70dd46002d4c096.jpg)

<details>
<summary>natural_image</summary>

Interior scene with two mugs and a monitor, no visible text or symbols
</details>

EWC   
Figure 23: Comparison of attention maps of different lifelong learning algorithms with RESNET-T on LIBERO-LONG. Each row shows the same state of a task with different neural architectures. "Task 5" refers to the task put the white mug on the left plate and put the yellow and white mug on the right plate. "Task 10" refers to the task put the yellow and white mug in the microwave and close it. The second row shows the policy that is trained on task 10 and gets evaluated on task 5, showing the attention map differences in backward transfer.

Findings: Figure 23 shows the attention visualization of three lifelong learning algorithms on LIBERO-LONG with RESNET-T on two tasks (task 5 and task 10). The first and third rows show the attention of the policy on the same task it has just learned. While the second row shows the attention of the policy on the task it learned in the past. We observe that:

1. PACKNET shows more concentrated attention compared against ER and EWC (usually just a single mode).   
2. ER shares similar attention map with EWC, but EWC performs much worse than ER. Therefore, attention can only assist the analysis but cannot be treated as a criterion for performance prediction.

# Different Neural Architectures

Policy after training task 5 on task 5   
![](images/de0e4ace0cd94ecd2b2050c75dd881aa906d280d8c46f88f282ae852df8d0822.jpg)

<details>
<summary>natural_image</summary>

Three mugs on a dark surface with blurred background lights (no text or symbols visible)
</details>

![](images/4fdb8703513f5ee9ae501d3134fb8004b11cd99bd3b1cc080158f4d599b982b9.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of three mugs on a dark surface with blurred background elements (no text or symbols)
</details>

![](images/c384d098e1bd162ae2df3513a360aa948fe74c7984e9ff89d9b28dcd8af11733.jpg)

<details>
<summary>natural_image</summary>

Interior scene with three mugs and a robot arm, featuring a thermal or heat map overlay (no text or symbols)
</details>

Policy after training task 10 on task 5   
![](images/39ce6509c1a5ed27188ff728fbfe28a3dd54fe6dd22b37b6081478f5a1fd832b.jpg)

<details>
<summary>natural_image</summary>

Three mugs on a dark surface under a blue-tinted light source, with no visible text or symbols.
</details>

![](images/b102679d894f0c787a6fa85d008811a8e7f4250bdd685b41b0dde0e1ead78adb.jpg)

<details>
<summary>natural_image</summary>

Thermal image showing three mugs on a dark surface with warm lighting (no text or symbols visible)
</details>

![](images/a2bc6b163334869d5dbeac232c71fdf634ac09e3da514463babd2924d5b869ea.jpg)

<details>
<summary>natural_image</summary>

Three white mugs on a dark surface under a blue-lit background, with a glowing orange-yellow spot in the corner (no text or symbols visible)
</details>

Policy after training task 10 on task 10   
![](images/3dda8f5d05a0a92347ef99a536620af40a148e9d007dd52b840063497e97864a.jpg)

<details>
<summary>natural_image</summary>

Interior scene with three mugs and a server unit, no visible text or symbols
</details>

ResNet-RNN

![](images/79d65ebffc027ab3b50c495589e443f1482231c659ac12f353e31c1e568c57f2.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a kitchen with three items (masks, a refrigerator, and a monitor) on a surface, no visible text or symbols
</details>

ResNet-T

![](images/c25f91a9122a89f9f6639bda1e16610ba3750b53d4e8a73d63f074e80fe39d9e.jpg)

<details>
<summary>natural_image</summary>

Interior scene with two mugs and a robotic arm, no visible text or symbols
</details>

ViT-T   
Figure 24: Comparison of attention maps of different neural architectures with ER on LIBERO-LONG. Each row shows the same state of a task with different neural architectures. “Task 5” refers to the task put the white mug on the left plate and put the yellow and white mug on the right plate. “Task 10” refers to the task put the yellow and white mug in the microwave and close it. The second row shows the policy that is trained on task 10 and gets evaluated on task 5, showing the attention map differences in backward transfer.

Findings: Figure 24 shows attention map comparisons of the three neural architectures on LIBERO-LONG with ER on two tasks (task 5 and task 10). We observe that:

1. ViT has more concentrated attention than policies using ResNet.   
2. When ResNet forgets, the attention is changing smoothly (more diluted). But for ViT, when it forgets, the attention can completely shift to a different location.   
3. When ResNet is combined with LSTM or a temporal transformer, the attention hints at the "course of future trajectory". But we do not observe that when ViT is used as the encoder.

# Different Task Ordering

![](images/fe1ffbc67e0e36d389ced3164bb80016d3b8dd772d86ee277b1c39308637a498.jpg)  
Figure 25: Attention map comparison among different orderings with ER and RESNET-T on three selected tasks from LIBERO-LONG: put both the alphabet soup and the tomato sauce in the basket, put the white mug on the left plate and put the yellow and white mug on the right plate, and put the yellow and white mug in the microwave and close it. Each row corresponds to a specific sequence of task ordering, and the caption of each attention map indicates the order of the task in that sequence.

Findings: Figure 25 shows attention map comparisons of three different task orderings. We show two immediately learned tasks from LIBERO-LONG trained with ER and RESNET-T. We observe that:

1. As expected, learning the same task at different positions in the task stream results in different attention visualization.   
2. There seems to be a trend that the policy has a more spread-out attention when it learns on tasks that are later in the sequence.

# With or Without Pretraining

![](images/ceb8edf265d20398685aac9e9db89e25f7bb822496d79fa30ce5c8fb66e45bf7.jpg)

<details>
<summary>text_image</summary>

w/o pretrained
ER
w/ pretrained
ER
w/o pretrained
PackNet
w/ pretrained
PackNet
w/o pretrained
EWC
w/ pretrained
EWC
put both the
alphabet soup and
the tomato sauce in
the basket
put the white mug on
the left plate and
put the yellow and
white mug on the
right plate
put the yellow and
white mug in the
microwave and close
it
</details>

Figure 26: Attention map comparison between models without/with pretrained models using RESNET-T and different lifelong learning algorithms on three selected tasks from LIBERO-LONG.

Findings: Figure 26 shows attention map comparisons between models with/without pretrained models on LIBERO-LONG with RESNET-T and all three LL algorithms. We observe that:

1. With pretraining, the policies attend to task-irrelevant regions more easily than those without pretraining.   
2. Some of the policies with pretraining have better attention to the task-relevant features than their counterparts without pertaining, but their performance remains lower (the last in the second row and the second in the fourth row). This observation, again, shows that there is no positive correlation between semantically meaningful attention maps and the policy's performance.