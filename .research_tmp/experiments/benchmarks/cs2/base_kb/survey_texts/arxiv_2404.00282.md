Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.00282v3 [cs.LG] 30 Oct 2024 
 
 

# Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods

 
 
 Yuji Cao
 
    
 Huan Zhao
 
    
 Yuheng Cheng
 
    
 Ting Shu
 
    
 Yue Chen
 
    
 Guolong Liu
 
    
 Gaoqi Liang
 
    
 Junhua Zhao
 
    
 Jinyue Yan
 
    
 Yun Li
 † † thanks: Corresponding author: Huan Zhao, Junhua Zhao. † † thanks: Yuji Cao and Yue Chen are with the Department of Mechanical and Automation Engineering, The Chinese University of Hong Kong, Hong Kong SAR, 999077, China (email: yjcao@mae.cuhk.edu.hk, yuechen@mae.cuhk.edu.hk) † † thanks: Huan Zhao and Jinyue Yan are with the Department of Building Environment and Energy Engineering, The Hong Kong Polytechnic University, Hong Kong, 100872, China (email: huan-paul.zhao@polyu.edu.hk, j-jerry.yan@polyu.edu.hk) † † thanks: Yuheng Cheng, Guolong Liu, and Junhua Zhao are with the School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen, 518172, China, and also with the Center for Crowd Intelligence, Shenzhen Institute of Artificial Intelligence and Robotics for Society (AIRS), Shenzhen, 518129, China (email: yuhengcheng@link.cuhk.edu.cn, liuguolong@cuhk.edu.cn, zhaojunhua@cuhk.edu.cn) † † thanks: Ting Shu is with Guangdong-Hongkong-Macao Greater Bay Area Weather Research Center for Monitoring Warning and Forecasting (Shenzhen Institute of Meteorological Innovation), Shenzhen, 518125, China. (email: shuting@gbamwf.com) † † thanks: Gaoqi Liang is with the School of Mechanical Engineering and Automation,
Harbin Institute of Technology, Shenzhen, 518055, China (e-mail:
lianggaoqi@hit.edu.cn) † † thanks: Yun Li is with the Shenzhen Institute for Advanced Study, University of Electronic Science and Technology of China, Shenzhen, 518110, China, and also with i4AI Ltd., WC1N 3AX London, U.K (email: Yun.Li@ieee.org) 

 Abstract 
 
 With extensive pre-trained knowledge and high-level general capabilities, large language models (LLMs) emerge as a promising avenue to augment reinforcement learning (RL) in aspects such as multi-task learning, sample efficiency, and high-level task planning. In this survey, we provide a comprehensive review of the existing literature in LLM-enhanced RL and summarize its characteristics compared to conventional RL methods, aiming to clarify the research scope and directions for future studies. Utilizing the classical agent-environment interaction paradigm, we propose a structured taxonomy to systematically categorize LLMs’ functionalities in RL, including four roles: information processor, reward designer, decision-maker, and generator. For each role, we summarize the methodologies, analyze the specific RL challenges that are mitigated, and provide insights into future directions. Lastly, comparative analysis of each role, potential applications, prospective opportunities and challenges of the LLM-enhanced RL are discussed. By proposing this taxonomy, we aim to provide a framework for researchers to effectively leverage LLMs in the RL field, potentially accelerating RL applications in complex applications such as robotics, autonomous driving, and energy systems.

 
 
 
 Index Terms:  Reinforcement learning (RL), large language models (LLM), vision-language models (VLM), multimodal RL, LLM-enhanced RL.

 
 

## I Introduction 

 
 Reinforcement learning (RL) is a powerful learning paradigm that focuses on control and decision-making, where an agent learns to optimize a specified target through trial and error interactions with the environment. Traditional RL methods, however, often struggled with high-dimensional state spaces and complex environments  [ 1 ] . The integration of deep learning techniques with RL, known as deep RL, has led to significant breakthroughs. In 2015, Deep Q-Networks (DQN)  [ 2 ] marked a turning point, demonstrating human-level performance on Atari games using raw pixel inputs. Subsequent innovations such as proximal policy optimization  [ 3 ] and soft actor-critic  [ 4 ] have further expanded the capabilities of deep RL. In different realms, deep RL algorithms have achieved promising performance, such as real-time strategy games  [ 5 ] ,  [ 6 ] , board games  [ 7 ] ,  [ 8 ] , energy management  [ 9 ] and imperfect information games  [ 10 ] ,  [ 11 ] . Concurrent advancements in natural language processing (NLP) and computer vision (CV)  [ 12 , 13 ] , have fostered new RL paradigms, such as language-conditional RL  [ 14 ] , which uses natural language to instruct agents, and vision-based RL  [ 15 ] , where agents learn from high-dimensional visual inputs.

 
 
 The integration of language and vision capabilities into deep RL has introduced new challenges, as the agent has to learn the high-dimensional features and a control policy jointly. To reduce the burden of visual feature learning, reference  [ 16 ] decoupled representation learning from RL. To handle language-involved tasks, a survey  [ 17 ] called for potential uses of NLP techniques in RL. Nevertheless, the capabilities of language models were limited at that time and the following four challenges still have not been addressed: 1) Sample inefficiency : Language and vision tasks involve large, complex state-action spaces, making it challenging for RL agents to learn effective policies. Moreover, agents must understand tasks and connect them to corresponding states, necessitating even more extensive interactions  [ 18 ] ,  [ 19 ] ,  [ 20 ] . 2) Reward function design : In language and vision tasks, designing effective reward functions is particularly challenging. These functions must capture subtle linguistic nuances and complex visual features, significantly increasing the complexity of an already difficult process. Moreover, aligning rewards with high-level task objectives in these domains often requires domain expertise and extensive trial-and-error  [ 21 ] ,  [ 22 ] ,  [ 1 ] ,  [ 23 ] . 3) Generalization : RL agents often overfit to training data, especially in vision-based environments, leading to poor performance when deployed in states with interventions (e.g., added noise). Agents must learn invariant features robust to such interventions, enabling generalization across varied linguistic contexts and visual scenes. However, the complexity of these domains makes extracting such features and adapting to new environments particularly challenging  [ 24 ] ,  [ 25 ] . 4) Natural language understanding : deep RL faces difficulties in natural language processing and understanding scenarios, where the nuances and complexities of human language present unique challenges that are not adequately addressed by current RL methodologies  [ 26 ] .

 
 
 The field of NLP has undergone a revolutionary transformation since the introduction of the Transformer architecture in 2017  [ 12 ] . This breakthrough paved the way for the development of Large Language Models (LLMs), with landmark models such as BERT  [ 27 ] , GPT  [ 28 ] , and more recent iterations such as GPT-3  [ 29 ] and PaLM  [ 30 ] marking significant milestones. The emergence of these LLMs has demonstrated powerful capabilities across various real-world applications, including medicine  [ 31 ] , chemical research  [ 32 ] , energy system  [ 33 ] ,  [ 34 ] ,  [ 35 ] and embodied control in robotics  [ 36 ] . These models have not only advanced the field of NLP but also shown remarkable potential in tackling complex, multi-disciplinary challenges. Compared to small language models, LLMs have emergent capabilities that are not present in small language models  [ 37 ] , such as in-context learning  [ 38 ] , reasoning ability  [ 39 ] etc. Additionally, leveraging the vast amounts of training data, pre-trained LLMs are equipped with a broad spectrum of world knowledge  [ 40 ] . Benefiting from these capabilities, the applications of language models have been shifted from language modeling to task-solving, ranging from basic text classification and sentiment analysis to complex high-level task planning  [ 41 ] and decision-making  [ 42 ] ,  [ 43 ] .

 
 
 With emergent capabilities, the potential of LLMs to address the inherent challenges of RL has recently gained popularity  [ 44 ] ,  [ 45 ] . The capabilities of LLMs, particularly in natural language understanding, reasoning, and task planning, provide a unique approach to solving the above-mentioned RL issues. For sample inefficiency, reference  [ 46 ] proposed a framework where LLMs can be employed to improve the sample efficiency of RL agents by providing rich, contextually informed predictions or suggestions, thereby reducing the need for extensive environment interactions. For reward function design, LLMs can aid in constructing more nuanced and effective reward functions, enhancing the learning process by offering a deeper understanding of complex scenarios  [ 47 ] . For generalization, reference  [ 48 ] proposed a framework that leverages language-based feedback for improving the generalization of RL policy in unseen environments. For natural language understanding, Pang et al.   [ 49 ] used LLMs to translate complex natural language-based instructions to simple task-specified languages for RL agents. These works show that LLM is a promising and powerful role that can contribute to the longstanding RL challenges.

 
 
 Despite the advancements in the domain of integrating LLMs into the RL paradigm, there is currently a notable absence of comprehensive review in this rapidly evolving area. Additionally, though various methods are proposed to integrate LLMs into the RL paradigm, there is no unified framework for such integration. Our survey paper seeks to fill these gaps by providing an extensive review of the related literature, defining the scope of the novel paradigm called LLM-enhanced RL , and further proposing a taxonomy to categorize the functionalities of LLMs in the proposed paradigm.

 
 

### I-A Contributions 

 
 This survey makes the following contributions:

 
 
 
 • 
 
 LLM-enhanced RL paradigm : This paper presents the first comprehensive review in the emerging field of integrating LLM into the RL paradigm. To clarify the research scope and direction for future works, we define the term LLM-enhanced RL to encapsulate this class of methodologies, summarize the characteristics and provide a corresponding framework that clearly illustrates 1) how to integrate LLMs in classical agent-environment interaction and 2) the multifaceted enhancements that LLMs offer to the conventional RL paradigm.

 

 • 
 
 Unified taxonomy : Further classifying the functionalities of LLMs in the LLM-enhanced RL paradigm, we propose a structured taxonomy to systematically categorize LLMs within the classical agent-environment paradigm, where LLMs are classified as information processors, reward designers, decision-makers, and generators. By such a categorization, a clear view of how LLMs integrate into the classical RL paradigm is offered.

 

 • 
 
 Algorithmic review : For each role of LLM, we review emerging works in this direction and discuss different algorithmic characteristics from the perspective of capabilities. Based on this foundation, future applications, opportunities, and challenges of LLM-enhanced RL are analyzed to provide a potential roadmap for advancing this interdisciplinary field.

 

 
 
 
 

### I-B Text Organization 

 
 The remaining sections are organized as follows. Section  II provides foundational knowledge of both RL and LLM. Section  III presents the concept of LLM-enhanced RL and provides its overall framework. Following this, Sections  IV ,  V ,  VI , and  6 offer an in-depth analysis of LLMs within the RL context, exploring their roles as information processor, reward designer, decision-maker, and generator, respectively. Last, Section  VIII discusses the application, opportunities and challenges of LLM-enhanced RL. Finally, Section  IX concludes the survey.

 
 
 
 

## II Background 

 
 In this section, we provide a concise overview of the classical RL paradigm and related challenges. Next, we explore the prevailing trend in RL—specifically, the fusion of multimodal data sources, including language and visual information. Following this, we offer an introductory background on LLMs and outline the key capabilities that can enhance the RL learning paradigm.

 
 

### II-A Background of Reinforcement Learning 

 

#### II-A 1 Classical Reinforcement Learning

 
 In a classical RL paradigm shown in Fig.  1 , the agent interacts with an environment through a trial-and-error process to maximize the specified rewards in the trajectory. In each step, the agent takes an action a a based on the observed state s s from the environment. By optimizing the policy π \pi (action controller), the agent maximizes the cumulative rewards. Such an optimization problem is usually formalized through the concept of Markov Decision Process (MDP), defined by the quintuple ⟨ 𝒮 , 𝒜 , 𝒯 , ℛ , γ ⟩ \langle\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma\rangle . Here, 𝒮 \mathcal{S} denotes a set comprising all possible states, 𝒜 \mathcal{A} denotes a set of all possible actions, 𝒯 \mathcal{T} represents the state transition probability function 𝒯 : 𝒮 × 𝒜 × 𝒮 → [ 0 , 1 ] \mathcal{T}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow[0,1] , ℛ \mathcal{R} is a reward function ℛ : 𝒮 × 𝒜 × 𝒮 → ℝ \mathcal{R}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow\mathbb{R} , and γ \gamma (with 0 ≤ γ ≤ 1 0\leq\gamma\leq 1 ) is the discount factor. The objective in RL is to optimize the policy π ⁡ ( a | s ) \pi(a|s) such that the cumulative returns ∑ k = 0 ∞ γ k ​ r k + 1 \sum_{k=0}^{\infty}\gamma^{k}r_{k+1} is maximized.

 
 
 Fig. 1: Classical reinforcement learning paradigm. 
 
 
 

#### II-A 2 Challenges of Reinforcement Learning

 
 While RL algorithms have made remarkable performance in recent years  [ 2 ] ,  [ 7 ] ,  [ 50 ] , there are still several longstanding challenges that limit the real-world applicability of RL:

 
 • 
 
 Generalization in Unseen Environment : Generalization in unseen environments remains a significant challenge in the field of RL  [ 51 ] . The core issue lies in the ability of RL algorithms to transfer learned knowledge or behaviors to new, previously unseen environments. RL models are often trained in simulated or specific settings, excelling in those scenarios but struggling to maintain performance when faced with novel or dynamic conditions. This limitation hinders the practical application of RL in real-world situations, where environments are rarely static or perfectly predictable. Achieving generalization requires models to not only learn specific task solutions but also to understand underlying principles that can be adapted to a range of situations.

 

 • 
 
 Reward Function Design : Reward function is the principal contributing factor to the performance of an RL agent. Despite their fundamental importance, reward functions are known to be notoriously difficult to design, especially in contexts involving sparse reward environments and complex scenarios  [ 1 ] . In sparse reward settings, where feedback is limited, reward shaping becomes essential to guide agents toward meaningful behaviors; however, this introduces the risk of inadvertently biasing the agent towards sub-optimal policies or overfitting to specific scenarios  [ 52 ] ,  [ 53 ] . Conversely, for complex tasks, high-performance reward functions usually require massive manual trial-and-error since most designed rewards are sub-optimal  [ 23 ] or lead to unintended behavior  [ 54 ] .

 

 • 
 
 Compounding Error in Model-based Planning : Model-based RL is prone to the issue of compounding errors during planning. As the prediction horizon extends, errors in the model’s predictions accumulate, leading to significant deviations from optimal trajectories  [ 55 ] ,  [ 56 ] . This problem is particularly acute in complex environments with high-dimensional state spaces. With their advanced predictive capabilities and understanding of sequential dependencies, LLMs could help mitigate these errors, leading to more accurate and reliable planning in model-based RL.

 

 • 
 
 Multi-task Learning: Multi-task RL faces several key challenges that limit its effectiveness. One major issue is managing varying task difficulties, where simpler tasks can overshadow learning of more complex ones, leading to negative transfer  [ 57 ] . Task interference is another critical problem, as shared parameters or data between tasks can result in suboptimal performance on individual tasks  [ 58 ] . Determining optimal parameter-sharing strategies is complex, as it must balance learning efficiency with task-specific requirements  [ 59 ] . Sample efficiency remains a significant hurdle, with traditional data-sharing approaches not fully leveraging learned behaviors across tasks  [ 60 ] . Finally, effectively transferring knowledge between tasks without negative interference is an ongoing challenge that impacts the agent’s ability to accelerate learning across multiple objectives  [ 61 ] .

 

 
 
 
 

#### II-A 3 Multimodal Reinforcement Learning

 
 With the advances in both CV and NLP, pattern recognition in vision and natural language has become increasingly powerful, and multimodal data has been involved in the RL paradigm recently. Visual data is commonly involved in the observation space of RL when agents receive image-based information from the environment, e.g. in applications such as robots  [ 62 ] , video game control  [ 2 ] etc. Compared to visual data, natural languages are usually included when RL agents are given specific tasks when interacting with the environments. The use of natural languages in RL can be divided into the following two categories  [ 17 ] :

 
 • 
 
 Language-conditional RL : In language-conditional RL, the problem itself requires the agent to interact with the environment through language. Specifically, there are two ways to integrate natural language in RL: 1) task description : the task or instruction is described in natural languages, e.g. instruction following, where the agents learn to interpret the instructions first and then execute actions; 2) action space or observation space : natural language is part of the state and action space, e.g. . text games, dialogue systems, and question answering (Q A). This class of RL leverages natural language as a direct component of the RL process, guiding the agent’s actions and decisions within the language environment.

 

 • 
 
 Language-assisted RL : In language-assisted RL, natural language is used to facilitate learning but not as a part of problem formulation. Two usages of language-assisted RL are: 1) communicating domain knowledge : the text containing task-related information can be helpful for agents. Therefore, wikis and manuals related to the environment can potentially assist the agents in such cases; 2) structuring policies : structuring policies is to communicate information about the state or dynamics of the environment based on language instead of representations of the environment or models. In such cases, language can be leveraged to shape representations towards a generalizable abstraction, such as using “avoid hitting the wall” instead of representations of a policy. This approach represents a more indirect use of natural language, serving as a guide or enhancer to the primary RL tasks.

 

 
 
 
 The integration of multimodal data challenges the RL paradigm since the agent has to simultaneously learn how to process complex multimodal data and optimize the control policy in the environment  [ 16 ] . Issues such as natural language understanding  [ 63 ] and visual-based reward function design  [ 64 ] require to be addressed.

 
 
 
 

### II-B Background of Large Language Models 

 
 LLMs typically refer to the Transformer-based language models  [ 65 ] containing billions of parameters and being trained on massive text data (i.e., several terabytes (TB) scale)  [ 66 ] , such as GPT-3  [ 29 ] and LLaMA  [ 67 ] . The extensive number of parameters and internet-scale training data enable LLMs to master a diverse array of tasks, resulting in enhanced capabilities in language generation, knowledge representation, and logical reasoning, as well as improved generalization to novel tasks.

 
 
 The development and effectiveness of LLMs are largely driven by Scaling Laws , i.e., as these models grow in size – both in terms of their parameter count and the data they are trained on – they tend to exhibit emergent abilities that are not present in small models  [ 68 ] ,  [ 69 ] ,  [ 70 ] , such as in-context learning, reasoning, and generalization. Here, we briefly introduce such capabilities of LLMs in detail:

 
 
 
 • 
 
 In-context Learning : In-context learning capability eliminates the need for explicit model retraining or gradient update  [ 29 ] , as it can generate better responses or perform tasks by inputs cueing examples or related knowledge. Specifically, task-related texts are included in the prompts as context information, helping the LLMs to understand the situations and execute instructions.

 

 • 
 
 Instruction Following : Leveraging diverse task-specific datasets formatted with natural language descriptions (also called instruction tuning ), LLMs are shown to perform well on unseen tasks that are also described in the form of natural language  [ 71 ] ,  [ 72 ] ,  [ 73 ] . Therefore, this capability equips LLMs with the ability to comprehend instructions for new tasks and effectively generalize across tasks not previously encountered, even in the absence of explicit examples.

 

 • 
 
 Step-by-step Reasoning : For smaller models, tackling multi-step tasks, such as solving math word problems, often proves to be challenging. However, large language models can address the complex task effectively with sophisticated prompting strategies such as Chain of Thought (CoT)  [ 74 ] , Tree of Thought (ToT)  [ 75 ] , and Graph of Thought (GoT)  [ 76 ] . These strategies structure the problem-solving process into sequential or hierarchical steps, facilitating a more articulated and understandable reasoning pathway. Additionally, prompts designed for planning enable LLMs to output sequences that reflect a progression of thoughts or actions, proving invaluable for tasks demanding logical sequence or decision-making outputs.

 

 
 
 
 
 

## III Large Language Model-Enhanced Reinforcement Learning 

 
 Fig. 2: Framework of LLM-enhanced RL in classical Agent-Environment interactions, where LLM plays different roles in enhancing RL. 
 
 

### III-A Definition 

 
 RL agents are often tasked with making robust and deliberate decisions using multimodal information in real-real applications, whether in the MDP setting or within the context of specific task descriptions. Examples include robots designed to follow natural language instructions while navigating physical environments or visual games with tasks described in natural language  [ 77 ] ,  [ 78 ] ,  [ 79 ] . However, it is challenging for conventional RL methods as the agent is required to simultaneously interpret complex multimodal data and optimize control policies amidst ever-changing environments  [ 80 ] . Compounding these challenges are issues like sample inefficiency, the difficulty of crafting reward functions that accurately reflect multimodal inputs, and the need for robust generalization across varied tasks and settings.

 
 
 The rapid advancements in LLMs present a viable solution to these challenges, thanks to their potent natural language understanding and reasoning abilities, coupled with recent progress in incorporating visual data processing  [ 81 ] . This dual capability enables LLMs to interpret and act upon complex multimodal information effectively, serving as a robust helper for enhancing the RL paradigm for real-world applications.

 
 
 Nevertheless, despite the powerful functionalities of LLMs, the current studies are varied and lack a standard concept correctly specifying the systematic methodology, which impedes the advancement of research in this area. Therefore, we introduce the concept called LLM-enhanced RL as follows:

 
 
 LLM-enhanced RL refers to the methods that utilize the multimodal information processing, generating, reasoning, and other high-level cognitive capabilities of pre-trained, knowledge-inherent LLM models to assist the RL paradigm. 

 
 
 LLM-enhanced RL differs from traditional model-based RL by leveraging knowledge-rich LLM models. This approach provides two key advantages: First, LLM equips the agent with substantial pre-trained capabilities at the beginning of the learning process, such as reasoning and high-level planning, etc. Second, it offers superior generalization capabilities. Pre-trained on diverse data, LLMs can effectively transfer knowledge across domains, enabling better adaptation to unseen environments than conventional data-driven models. Last, LLM-enhanced RL addresses a key limitation of pre-trained models: their inability to interact with environments to expand their knowledge and ground themselves in specific domains. Through environmental interactions, this approach generates task-specific data, grounds the LLM in particular domains by in-context learning, and eventually helps them adapt to dynamic changes with continuous learning.

 
 
 

### III-B Framework 

 
 The framework of LLM-enhanced RL is illustrated in the center of Fig.  2 , which is founded on the classical agent-environment interaction paradigm. Along with the trial-and-error learning process, LLM processes the state information, redesigns the reward, assists in action selection, and interprets the policy after the action selection.

 
 
 Specifically, on the one hand, when the agent receives the state and reward information from the environment, LLM is able to process or modify the information to either filter unnecessary natural language-based information or design appropriate rewards to accelerate the learning process, based on the natural language understanding and reasoning capabilities. On the other hand, when the agent is about to choose an action based on the observation, LLM can assist the action selection process by either simulating a world model or serving as the policy network to generate reasonable actions based on the modeling capability and common-sense knowledge. Additionally, after the action selection process, integrating state, reward, and action information, LLM can interpret the underlying possible reasons behind the policy selection, which helps human supervisors understand the scenarios for further system optimization.

 
 
 Based on the functions of LLM in the framework, we extract characteristics of LLM-enhanced RL and further divide four different LLM roles in LLM-enhanced RL, including information processor, reward designer, generator, and decision-maker, which will be elaborated in the next subsections.

 
 
 

### III-C Characteristics 

 
 The LLM-enhanced RL paradigm enhances the vanilla RL paradigm with the following characteristics:

 
 • 
 
 Multimodal Information Understanding : LLMs enhance RL agents’ comprehension of scenarios involving multimodal information, enabling them to learn from tasks or environments described in natural language and vision data more effectively.

 

 • 
 
 Multi-task Learning and Generalization : Benefiting from the multi-disciplinary pre-trained knowledge and powerful sequence modeling capability, LLMs empower RL agents by providing a high-capacity model capable of accommodating task variances and transferring knowledge across multiple tasks, assisting in handling multiple tasks.

 

 • 
 
 Improved Sample Efficiency : Given the inherent exploratory nature, the RL paradigm demands significant samples to learn. Pre-trained LLM can enhance data generation by simulation or leverage the prior knowledge to improve the sample efficiency of RL.

 

 • 
 
 Long-Horizon Handling : RL becomes more challenging as the length of trajectory increases, due to the credit assignment problem. LLMs can decompose complex tasks down into sub-tasks to assist RL agents in planning over longer temporal horizons, aiding in the decision-making process for complex, multi-step tasks such as the Minecraft game.

 

 • 
 
 Reward Signal Generation : Based on the context understanding and domain knowledge, LLMs contribute to the reward shaping and reward function designing, which help guide the RL towards effective policy learning in sparse-reward environments.

 

 
 
 
 

### III-D Taxonomy 

 
 In this subsection, we illustrate the different roles of LLMs within the above framework, by detailing their functions and corresponding issues of RL they address:

 
 
 
 • 
 
 Information Processor : When observation or task description involves language or visual features, it is challenging for the agent to comprehend the complex information and optimize the control policy simultaneously. To release the agent from the burden of understanding the multimodal data, LLM can serve as an information processor for environment information or task instruction information by 1): extracting meaningful feature representations for speeding up network learning; 2) translating natural language-based environment information or task instruction information into formal specific task languages to reduce learning complexity.

 
 
 Application example : In instruction-following RL for robots, tasks can have unbounded natural language forms due to users’ diverse speaking habits, which can impede RL learning performance. LLMs transform these varied natural language instructions into a unique task language, enabling more robust RL performance  [ 49 ] .

 

 • 
 
 Reward Designer : In complex task environments where the reward is sparse or a high-performance reward function is hard to define, using the prior world knowledge, reasoning abilities, and code generation ability, LLM can serve as two roles: 1) an implicit reward model to provide reward values based on the environment information, either by training or prompting; 2) an explicit reward model that generates executable codes of reward functions that transparently specifies the logical calculation process of reward scalars based on the environment specifications and language-based instructions or goals.

 
 
 Application example : For complex robotic control problems, such as dexterous manipulation, reward design requires both expertise and trial and error. LLM designs reward functions based on knowledge and iteratively improves it based on the performance  [ 82 ] .

 

 • 
 
 Decision-maker : RL faces challenges such as sample inefficiency and exploration inefficiency. To address these challenges, LLMs can be leveraged as decision-makers in RL, offering promising solutions through two main approaches: 1) action-making: LLMs treat offline RL as a sequence modeling problem, using rewards for the conditional generation of actions. Pretrained on diverse, internet-scale data, LLMs possess advanced semantic understanding capabilities, which can be exploited to accelerate offline RL learning. 2) action-guiding: LLMs act as expert instructors to produce a reduced set of action candidates or expert actions. The action candidates constrain the original action space, thereby enhancing exploration efficiency. Expert actions encapsulate prior knowledge from LLMs. When incorporated to regularize the policy learning, this expert knowledge is distilled into an RL agent, resulting in better sample efficiency.

 
 
 Application example : In embodied robot, given human-instruction and language-based description, LLM generates potential actions for the robot to choose from  [ 83 ] .

 

 • 
 
 Generator : Model-based RL hinges on precise world models to learn accurate environment dynamics and simulate high-fidelity trajectories. Additionally, interpretability remains another important issue in RL. Using the multimodal information understanding capability and prior common-sense reasoning ability, LLMs can be 1) a generator to generate accurate trajectories in model-based RL; 2) generate policy explanations with the prompts of related information in explainable RL.
 Application example : In Minecraft item crafting, LLMs generate Abstract World Models—hypothesized sequences of subgoals for a given task. These LLM-generated world models guide RL agents’ exploration and learning and the RL agent verifies and corrects the world model through gameplay, combining LLM knowledge with grounded experience to achieve an order of magnitude improvement in sample efficiency over traditional methods  [ 84 ] .

 

 
 
 
 
 

## IV LLM as Information Processor 

 
 The normal way for deep RL with language or visual information is to jointly process the information and learn a control policy, end-to-end. However, this demands the RL agent to learn to comprehend the information and manage the task simultaneously. Additionally, learning the language or visual features by simply relying on the reward function is challenging and may narrow the learned features to a narrow utility, hampering the generalization ability of the agent  [ 16 ] .

 
 
 With the advances in unsupervised techniques and large-scale pre-trained models for CV and NLP, the decoupled structure, where the encoders are separately trained, has gained popularity  [ 16 ] ,  [ 85 ] ,  [ 86 ] . Utilizing the powerful representation ability and prior knowledge, the pre-trained LLM or vision-language model (VLM) model can serve as an information processor for RL. They can extract observation representations for downstream networks or translate natural language into formal specifications, enabling the execution of multiple tasks. This multi-task capability improves sample efficiency and zero-shot performance, allowing agents to generalize effectively across diverse and sparse-reward environments.

 
 

### IV-A Feature Representation Extractor 

 
 Adopting the large pre-trained models in CV and NLP, the learned feature representation can be a scaffold embedding for downstream network learning and increase the sample efficiency. As illustrated in Fig.  3 (i), the usages can be further divided into two categories according to whether the model is trained simultaneously. One way is to directly use the frozen pre-trained model to extract embeddings from the observation 𝒪 t \mathcal{O}_{t} and another way is to further fine-tune the pre-trained model using contrastive learning with a contrastive loss ℒ t c \mathcal{L}_{t}^{c} to achieve better adaptation in new environments.

 
 
 Fig. 3: LLM as an information processor. (i) Feature Representation Extractor: frozen/fine-tuned LLM extracts meaningful representations for downstream RL networks. In the fine-tuning process, given observation ( 𝒪 t \mathcal{O}_{t} ), invariant feature abstraction ( S ~ t \tilde{S}_{t} ) is learned with the contrastive loss ( ℒ t c \mathcal{L}^{c}_{t} ). Then, the invariant is fed into the actor-critic network. After fine-tuning, given different observations ( O t O_{t} ) and ( O t ′ O^{\prime}_{t} ) with appearance variation, the extracted representation is invariant, leading to robust RL performance. (ii) Language Translator: LLM interprets diverse natural language inputs, converting them into a standardized, task-specific format that the RL agent can efficiently process and act upon. 
 
 

#### IV-A 1 Frozen Pre-trained Model

 
 Using the frozen large-scale pre-trained model is the straightforward way. In reference  [ 87 ] , the author proposed History Compression via Language Models (HELM) to utilize a frozen pre-trained Language Transformer to extract history representation and compression and thus addresses the problem of partially observed MDP by approximating the underlying MDP with the past representation. Specifically, the framework first uses FrozenHopfield, a frozen associative memory to map observations [ o t − 2 , o t − 1 , o t ] [o_{t-2},o_{t-1},o_{t}] to a compressed representation h t h_{t} and then concatenate it with a learned encoding of the current observation via a convolutional neural network. Such a method solves the problem of how to effectively utilize the compressed history for policy optimization. After that, Semantic HELM  [ 88 ] proposed a human-readable memory mechanism that summarizes past visual observations in human language and uses multimodal models to associate visual inputs with language tokens. The memory is a semantic database 𝒮 \mathcal{S} constructed by encoding prompt-augmented tokens from the vocabularies of Contrastive Language-Image Pre-training (CLIP)  [ 89 ] and the pre-trained language models. Given an observation o t o_{t} , the agent retrieves top- k k embeddings from 𝒮 \mathcal{S} as actor-critic input to assist the policy optimization. Such a memory mechanism provides a human-readable representation of the past and helps the agent to cope with partially observable environments. In the experiment, they used the proximal policy optimization (PPO)  [ 3 ] algorithm and pretrained Transformer XL  [ 90 ] model. When testing on the partially observable environments, they found the extracted semantics help the memory-less agent obtain comparable scores with memory-based methods trained on long trajectories. However, one limitation of such frozen pre-trained models is that the representations cannot dynamically adjust according to the task and environment.

 
 
 

#### IV-A 2 Fine-tuning Pre-trained Model

 
 When trained RL agents are deployed in real-world applications, their performance often deteriorates under significant appearance variations (out-of-distribution data) due to overfitting to training scenarios. For example, robots with vision-based navigation tasks may fail when the environment color changes. Invariant feature representations serve as a form of state abstraction that remains consistent across out-of-distribution appearance variations such as added noise, brightness changes, or slight rotations. When encountering appearance changes or out-of-distribution data, though the observation changed, the representation (feature embedding) that fed into the policy/value network is nearly unchanged, leading to robust RL performance and increased generalization in unseen environments.

 
 
 Contrastive learning is a common way to learn the invariant feature representation. It learns representations from high-dimensional data by contrasting positive examples against negatives. Given a query q q , the goal is to match query q q more closely to a positive key k + k_{+} than to any negative keys 𝕂 ∖ { k + } \mathbb{K}\setminus\{k_{+}\} in a set 𝕂 \mathbb{K} . This process is modeled using similarity measures, such as the dot product ( q T ​ k ) (q^{T}k) or the bilinear product ( q T ​ W ​ k ) (q^{T}Wk) , where W W is a weight matrix. To effectively learn these representations, a contrastive loss function such as InfoNCE  [ 91 ] is used:

 

 
 | 
 ℒ c = log ⁡ exp ⁡ ( q T ​ W ​ k + ) exp ⁡ ( q T ​ W ​ k + ) + ∑ k K − 1 exp ⁡ ( q T ​ W ​ k i ) \mathcal{L}^{c}=\log\frac{\exp(q^{T}Wk_{+})}{\exp(q^{T}Wk_{+})+\sum_{k}^{K-1}\exp(q^{T}Wk_{i})} | 
 | 
 (1) | 
 

 where exp \exp is the exponential symbol and the whole ℒ c \mathcal{L}^{c} can be viewed as the log-loss of a softmax classifier, treating the matching of q q to k + k_{+} as a multi-class classification problem among K K classes. By maximizing alignment between different changes of the same observation via the above loss, the model can learn the invariant representations.

 
 
 When combining with RL, given different RL tasks, the required invariant feature representations should be adjusted accordingly. Therefore, researchers have explored different ways to improve contrastive learning. In reference  [ 92 ] , to achieve the zero-shot capability of embodied agents, the author devised a visual prompt-based contrastive learning framework that uses a pre-trained VLM to learn the visual state representations. The visual prompts are learned on expert demonstrations from domain factors such as camera settings and stride length. By contrastively training the VLM on a pool of visual prompts along with the RL policy learning process, the learned representations are robust to the variations of environments, leading an increase of 18-20% success rates on unseen scenarios and improved generalization capability. Based on the contrastive learning, another method ReCoRe  [ 93 ] added an intervention-invariant regularizer in the form of an auxiliary task such as depth prediction and image denoising to explicitly enforce invariance of learned representations to environment changes.

 
 
 
 

### IV-B Language Translator 

 
 The unbounded and diverse representation of natural languages in both human instruction and environmental information impedes policy learning. LLM can be leveraged as a language translator to reduce the additional burden of comprehending natural language for RL agents and increase sample efficiency. As illustrated in Fig.  3 (ii), LLM transforms the diverse and informal natural language information into formal task-specific information, such as feature representation or task-specific languages, thus assisting the learning process of the RL agent.

 
 

#### IV-B 1 Instruction Information Translation

 
 One application of LLM is to translate natural language-based instructions for instruction-following applications. In reference  [ 49 ] , the author investigated an inside-out scheme for natural language-conditioned RL by training an LLM that translates the natural language to a task-related unique language. Such an inside-out scheme prevents the policy from being directly exposed to natural language instructions and helps efficient policy learning. Another literature STARLING   [ 94 ] used LLM to translate natural language-based instructions to game information and example game metadata. The translated information is then fed forward to Inform7 , an interactive fiction game engine, to develop a large amount of text-based games for RL agents to master the desired skills.

 
 
 

#### IV-B 2 Environment Information Translation

 
 On the other hand, LLM can also be used to translate the natural language environment information into formal domain-specific language that can specify MDP information, which converts natural language sentences into grounded usable knowledge for the agent. Previous works generally ground natural language into individual task components such as task objectives description  [ 95 ] , rewards  [ 96 ] and polices  [ 97 ] ,  [ 98 ] . To unify the information about all the components of a task,  [ 99 ] introduces RLang , a grounded formal language capable of expressing information about every element of an MDP and solutions such as policies, plans, reward functions and transition functions. By using LLM to translate natural language to RLang and train RL agents upon RLang, the agents are capable of leveraging information and avoid having to learn tabula rasa .

 
 
 
 

### IV-C Summarization and Outlook 

 
 For information processing, LLM is used to accelerate RL learning processing by decoupling the information processing task and the controlling task, where LLM extracts feature representations or handles the natural language-based information.

 
 
 When multimodal data are involved in the environment, e.g., robot manipulation tasks, the information processing task becomes more challenging. For one thing, the misalignment or contradiction between different modalities may exist  [ 100 ] . Modality weighting techniques that adaptively learn the importance of different modalities by attention mechanisms provide a potential solution for this problem  [ 101 ] . In addition, CLIP-based multimodal foundation models using large-scale image-text pairs have shown outstanding zero-shot ability in various multimodal tasks  [ 89 ] . By learning cross-modal and task semantics, the misaligned modality can be replaced with a virtually generated modality  [ 102 ] . For another, how to effectively combine the information between different modalities into a unified representation, i.e., multimodal fusion, is also an important issue. In reference  [ 103 ] , a multimodal contrastive learning with attention mechanisms, which learns the intra and inter-modal representations, was proposed. The different attention heads align the agreement from one modality to another and vice versa, providing an informative representation for the RL agent. However, most multimodal learning methods do not consider the task information and only focus on the modality alignment, remaining an area to be explored.

 
 
 In the following, we list potential directions for future research.

 
 • 
 
 Feature Representation Extractor : Although the use of LLMs as feature representation extractors has shown promise in enhancing RL, several challenges persist in future research. Short-term goals include developing a computationally efficient feature extractor and improving the generalization of LLM-derived representations. In the long term, researchers should focus on exploiting task compositionality for better generalization and creating adaptive extraction methods for diverse control tasks.

 

 • 
 
 Language Translator : Current existing works are still limited. Short-term objectives involve exploring LLMs’ ability to handle more tasks and improving translation efficiency and accuracy in RL contexts. Long-term goals include developing multimodal translation capabilities and integrating these with RL algorithms, achieving a more general translator, and helping agent learning.

 

 
 
 
 
 

## V LLM as Reward Designer 

 
 The reward signal is the most important information to instruct agent learning in RL  [ 1 ] . However, despite the fundamental importance, high-performing reward functions are known to be notoriously difficult to design  [ 104 ] . First, specifying human notions of desired behavior is difficult via designed reward functions or requires huge expert demonstrations. Moreover, dense rewards that accurately provide learning signals require either manually decomposing the general goal into sub-goals  [ 105 ] or rewarding interesting auxiliary objectives  [ 106 ] . Nevertheless, both of these methods suffer from the need for expert input and meticulous manual crafting  [ 23 ] .

 
 
 Benefiting from pre-trained common-sense knowledge, code generation, and in-context learning ability, LLM has the potential to design or shape reward functions for DRL by leveraging natural language-based instructions and environment information. In this section, we review recent literature in which LLMs act as reward models that implicitly provide reward values or explicitly write executable reward function codes detailing the calculation process of reward scalars.

 
 
 Fig. 4: LLM as a reward designer. (i) Implicit Reward Model: LLMs provide rewards through direct prompting or alignment scoring between language instructions and visual observations. (ii) Explicit Reward Model: LLMs generate executable code for reward functions, with potential for self-refinement through evaluation loops. 
 
 

### V-A Implicit Reward Model 

 
 A large pre-trained model can be an implicit reward model that directly provides auxiliary or overall reward value based on the understanding of task objectives and observations. The methods are illustrated in Fig.  4 (i). One way is by directly prompting with language descriptions and another way is by scoring the alignment between the feature representation of the visual observations and language-based instructions.

 
 

#### V-A 1 Direct Prompting

 
 Reference  [ 107 ] simplified the manual reward design by prompting an LLM as a proxy reward function with examples of desirable behaviors and the preferences description of the desired behaviors. Reference  [ 108 ] proposed a Read and Reward framework that utilizes LLMs to read instruction manuals to boost the learning policies of specific tasks. The framework includes a Question Answer (QA) extraction module for information retrieval and summarization and a Reasoning module for evaluation. Experimentally, they show RL algorithms, by their design, can obtain significant improvement in performance and training speed. In reference  [ 45 ] , Carta et al. proposed an automated reward shaping method where the agent extracts auxiliary objectives from the general language goal. Using a question generation and QA system, the framework guides the agent in reconstructing partial information about the global goal and provides an intrinsic reward signal for the agent. This intrinsic reward incentivizes the agent to produce trajectories that help them reconstruct the partial information about the general language goal. To acquire a generalizable policy by continually learning a set of tasks is challenging for RL agents since the agent is required to retain the previous knowledge while quickly adapting to new tasks. Reference  [ 109 ] introduced the Lafite-RL (Language agent feedback interactive RL) framework that provides interactive rewards mimicking human feedback based on LLMs’ real-time understanding of the agent’s behavior. By designing two prompts, one to let LLM understand the scenario and the other to instruct it about the evaluation criterion, their framework can accelerate the RL process while freeing up human effort during the interaction between agent and environment.

 
 
 Algorithm 1 Language Reward Modulated Pretraining (LAMP) in  [ 110 ] 
 
 
 1: 
 
 
  Initialize parameters for Masked World Models (MWM)

 
 
 2: 
 
 
  Load pretrained DistilBERT  [ 111 ] for language preprocessing

 
 
 3: 
 
 
  Load pretrained R3M visual encoder and score predictor

 
 
 4: 
 
 
  Initialize empty replay buffer

 
 
 5: 
 
 
  Populate language prompt buffer and synonym buffers with predefined samples

 
 
 6: 
 
 
   for each training episode do 

 
 
 7: 
 
 
   Randomize scene textures from Ego4D and RLBench datasets

 
 
 8: 
 
 
   Sample ShapeNet objects and language prompts

 
 
 9: 
 
 
   Insert ShapeNet objects into the scene

 
 
 10: 
 
 
   Generate and process language embeddings using DistilBERT

 
 
 11: 
 
 
   Execute policy to collect transitions

 
 
 12: 
 
 
   Assign LAMP rewards using R3M score predictos

 
 
 13: 
 
 
   Update buffers and train MWM with augmented rewards

 
 
 14: 
 
 
   end for 
 
 
 
 
 

#### V-A 2 Alignment Scoring

 
 For visual RL, some literature utilizes vision-language models as reward models to align multimodal data and calculate the similarity score using metrics such as cosine similarity. Rocamonde et al. employs the CLIP model as a zero-shot reward model to specify tasks via natural language  [ 64 ] . They first compute the probability p o t , l p_{o_{t},l} that the agent achieves a goal given by language description l l out of a set of potential goals l ′ ∈ ℒ l^{\prime}\in\mathcal{L} in the task set ℒ \mathcal{L} using the softmax computation with temperature τ \tau over the cosine similarity between visual state embeddings f θ ​ ( o t ) f_{\theta}(o_{t}) and language description embeddings g θ ​ ( l ) g_{\theta}(l) across the set of potential goals l ′ l^{\prime} :

 

 
 | 
 p o t , l = exp ⁡ ( f θ ​ ( o t ) ⋅ g ϕ ​ ( l ) / τ ) ∑ l ′ exp ⁡ ( f θ ​ ( o t ) ⋅ g ϕ ​ ( l ′ ) / τ ) . p_{o_{t},l}=\frac{\exp(f_{\theta}(o_{t})\cdot g_{\phi}(l)/\tau)}{\sum_{l^{\prime}}\exp(f_{\theta}(o_{t})\cdot g_{\phi}(l^{\prime})/\tau)}. | 
 | 
 (2) | 
 

 Then the reward is obtained by a binary reward function r t = r ( o t + 1 , l ) = 𝕀 [ p o t + 1 , l ≥ β ] r_{t}=r(o_{t+1},l)=\mathbb{I}[p_{o_{t+1},l}\geq\beta] , which thresholding the probability. Their framework only requires a single-sentence text prompt description of the desired task with minimal prompt engineering. Another work  [ 112 ] constructed reward signals based on the similarity between natural language-based description and embeddings from the pre-trained VLM encoder. By labeling the expert demonstration with the reward signals, the framework effectively mitigates the problem of mis-generalization. In reference  [ 110 ] , the authors proposed the Language Reward Modulated Pretraining (LAMP) framework as a pertaining utility for RL as opposed to a downstream task reward to warm-start sample-efficient learning. The framework leverages frozen, pre-trained VLMs such as R3M  [ 113 ] to generate noisy, albeit shaped exploration rewards by computing the alignment score between instructions and image observations. The algorithm is presented in Algorithm  1 . They used Masked World Model  [ 114 ] , a visual model-based RL algorithm for robot manipulation based on image and instructions. The images were downloaded from Ego4D  [ 115 ] and the language instructions were obtained by querying ChatGPT. The reward is then calculated from the R3M alignment score. After that, the generated rewards are optimized with standard novelty-seeking exploration rewards for language-conditioned policy. Reference  [ 116 ] explored preference-based RL, where the agent learns a reward function from preference labels over the behaviors. A VLM is leveraged to generate preference labels given visual observations and a text description of the task goal. The evaluation is based on a series of vision-based manipulation tasks. Results suggested that prompting VLMs to produce preference labels for reward learning leads to better performance, in contrast to treating them as reward functions to produce raw reward scores.

 
 
 
 

### V-B Explicit Reward Model 

 
 Another way to design reward functions is by generating executable codes that explicitly specify the details of the calculation process, as illustrated in Fig.  4 (ii). Compared to the implicit reward value provision, this explicit way transparently reflects the reasoning and logical process of LLMs and thus is readable for humans to further evaluate and optimize.

 
 
 To help robots learn low-level actions, reference  [ 117 ] harnessed the code generation ability of LLMs to define the lower-level reward parameters based on high-level instructions. Using such a reward design paradigm, their work bridges the gap between high-level language instructions to low-level robot actions and can reliably tackle 90% of the designed tasks compared to the 50% of the baseline. Motivated by the capability of LLM for self-refinement  [ 118 ] , reference  [ 119 ] proposed a framework with a self-refinement mechanism for automated reward function design, including initial design, evaluation and self-refinement loop. Their results indicate that the LLM-designed reward functions are able to rival or surpass manually designed reward functions. Similarly, Eureka   [ 82 ] developed a reward optimization algorithm with self-reflection. The algorithm is outlined in Algorithm  2 . In each iteration, it uses an environment source code and task description to sample different reward function candidates from a coding LLM. Then the candidates are used to instruct RL training. After training, the results are used to calculate the scores of the reward candidates. Then the best reward function code are selected for reflection, where LLM uses the reasoning capability to progressively improve the reward code. In the experiment, results show that their proposed method can achieve human-level performance on reward design and solve dexterous manipulation tasks that were previously infeasible by manual reward engineering. Another work Text2Reward  [ 120 ] generated shaped dense reward functions as executable programs based on the environment description. Given the sensitivity of RL training and the ambiguity of language, the RL policy may fail to achieve the goal. Text2Reward addresses the problem by executing the learned policy in the environment, requesting human feedback and refining the reward accordingly.

 
 
 Algorithm 2 Eureka  [ 82 ] 
 
 
 0: 
 
 
  Task description l l , environment code C C , coding LLM M c M_{c} , evaluation function E E , initial prompt prompt , optimization iterations N N , iteration batch size K K 

 
 
 1: 
 
 
   for i ← 1 i\leftarrow 1 to N N do 

 
 
 2: 
 
 
    // Sample K K reward code candidates from coding LLM M c M_{c} 

 
 
 3: 
 
 
    R 1 , … , R K ← sample ​ ( l , M c , C , prompt ) R_{1},\dots,R_{K}\leftarrow\texttt{sample}(l,M_{c},C,\texttt{prompt}) 

 
 
 4: 
 
 
    // Evaluate reward candidates 

 
 
 5: 
 
 
    s 1 i = E ⁡ ( R 1 i ) s_{1}^{i}=E(R_{1}^{i}) , s 2 i = E ⁡ ( R 2 i ) s_{2}^{i}=E(R_{2}^{i}) , …, s K i = E ⁡ ( R K i ) s_{K}^{i}=E(R_{K}^{i}) 

 
 
 6: 
 
 
    // Select the best reward code 

 
 
 7: 
 
 
    best = arg ⁡ max k ⁡ ( s 1 i , … , s K i ) {\mathrm{best}}=\arg\max_{k}(s_{1}^{i},\dots,s_{K}^{i}) 

 
 
 8: 
 
 
    // Reward reflection 

 
 
 9: 
 
 
    prompt ← \leftarrow prompt:Reflection ( R best i , s best i R^{i}_{\mathrm{best}},s^{i}_{\mathrm{best}} )

 
 
 10: 
 
 
    // Optimize Eureka reward code 

 
 
 11: 
 
 
    if s best i s Eureka s^{i}_{\mathrm{best}} s_{\mathrm{Eureka}} then 

 
 
 12: 
 
 
     R Eureka , s Eureka ← ( R best i , s best i ) R_{\mathrm{Eureka}},s_{\mathrm{Eureka}}\leftarrow(R^{i}_{\mathrm{best}},s^{i}_{\mathrm{best}}) 

 
 
 13: 
 
 
    end if 
 
 
 14: 
 
 
   end for 
 
 
 15: 
 
 
   return R Eureka R_{\mathrm{Eureka}} 

 
 
 
 
 

### V-C Summarization and Outlook 

 
 The use of LLMs as reward designers in RL offers a more natural and efficient way to design complex reward functions. By leveraging natural language processing capabilities, LLMs simplify the traditionally challenging task of reward function design, enhancing both the efficiency and effectiveness of RL algorithms.

 
 
 However, the inherent biases in LLMs may transfer to the designed reward functions, potentially resulting in suboptimal or harmful behaviors  [ 121 ] . This presents a potential risk requiring careful consideration. While existing mitigation efforts primarily address biases in demographic, cultural, and political beliefs  [ 122 ] , task-specific biases remain understudied, thus limiting the applicability of LLM-designed reward functions. Toward this end, we list some potential solutions. First, in preference-based RL, when an RL agent optimizes against biased reward models generated by LLMs to predict human preferences, overoptimization and overfitting may occur  [ 123 ] , impeding the learning of the true reward function. Reward function regularization  [ 124 ] includes the agent preference generated by the value function as a regularization term to mitigate the risk, which helps recover the true underlying reward function. Secondly, human-in-the-loop approaches are another direction to prevent harmful behaviors from designed reward functions. One viable solution is to design an evaluation module where humans can intervene and correct undesired behaviors and reward functions when the agent’s action violates a predefined set of rules. Finally, principles from ensemble learning suggest that combining the reward functions from different LLM models mitigates bias from individual LLMs, leading to improved unbiased performance compared to a single LLM  [ 125 ] .

 
 
 In the future, we expect advancements in the following areas:

 
 • 
 
 Implicit Reward Model : An immediate focus may consider improving how well LLM-generated rewards align with human intentions. This involves refining the quality of language instructions to reduce ambiguities and inaccuracies, ensuring that the reward functions align precisely with human notions of desired behavior. In the longer term, generalization and transferability of LLM-generated rewards across different tasks and environments, especially in complex, high-dimensional visual environments, is also an important direction to explore.

 

 • 
 
 Explicit Reward Model : A key limitation in reward code generation is its dependency on pre-trained common-sense knowledge, which can be restrictive for highly specialized tasks not covered in training data. Therefore, short-term goals may include enhancing prompts with detailed, task-specific information and external knowledge. Additionally, the reliance on manually designed templates for motion descriptions limits the adaptability. Looking further ahead, researchers might develop automated or unified processes for designing templates, moving beyond the current limitations of manual motion description templates.

 

 
 
 
 
 

## VI LLM as Decision-Maker 

 
 LLMs trained on a massive amount of data show impressive results in language understanding tasks  [ 29 ] , instruction-following  [ 126 ] , vision-language navigation  [ 127 ] and tasks requiring planning and sequential reasoning  [ 83 ] . Such success motivates researchers to explore the potential of LLMs for decision-making problems. In this section, we divide the role of LLM as 1) the action-maker that generates actions; and 2) the action-guider that instructs the actions.

 
 
 Fig. 5: LLM as a decision-maker. (i) Action-Making: given a T T -length trajectory τ = ( R ^ 1 , s 1 , a 1 , … , R ^ T , s T , a T ) \tau=(\hat{R}_{1},s_{1},a_{1},\dots,\hat{R}_{T},s_{T},a_{T}) as a sequence of ordered return-to-go R ^ \hat{R} , action a a , and states s s , LLM learns to predict future action a t ′ a^{\prime}_{t} by minimizing the mean squared error loss ℒ = ∑ t ‖ a t − a t ′ ‖ 2 2 \mathcal{L}=\sum_{t}\left\|a_{t}-a_{t}^{\prime}\right\|_{2}^{2} . (ii) Action-Guiding: LLM generates a reduced set of action candidates for agents or generates expert actions to regularize RL learning. 
 
 

### VI-A Action-Making 

 
 Transformer-based models such as Decision Transformer (DT)  [ 128 ] have shown great potential in offline RL domain. Instead of using the traditional trial-and-error way, these models treat offline RL as a sequence modeling problem, yielding promising results. As LLM itself is a large-scale Transformer-based model, a natural thought is to leverage the pre-trained power of LLM within this paradigm.

 
 
 We term this function of LLM as action-making, and identify two typical approaches, as illustrated in Fig.  5 (i). In the first approach (left figure), pre-trained LLM is fine-tuned and then employed for action generation. In the second approach (right figure), goal/instruction along with trajectory are fed to pre-trained LLM. Moreover, a task-specific smaller model is appended after the fine-tuned LLM to facilitate rapid adaptation to diverse tasks.

 
 
 Pre-trained LLMs outperform basic DT in generalization and sample efficiency, especially for sparse-reward and long-horizon tasks. LLMs’ latent representations, learned from diverse linguistic data, provide valuable prior knowledge for new tasks. This knowledge enables LLMs to solve unseen tasks with less training data, by transferring knowledge from similar tasks and predicting high-reward actions even with sparse feedback. For instance, comparing pre-trained LLM with basic DT, Li et al.  [ 129 ] reported a 43.6% improvement in out-of-distribution (novel) task completion rates while requiring less training data (e.g., 500 vs 10K). Additionally, Shi et al.  [ 130 ] demonstrated a 50% performance gain in sparse-reward environments like Kitchen and Reacher2d. For long-horizon tasks, pre-trained representations encode future information, guiding decision-making over extended sequences. For instance, in AntMaze, a long-horizon navigation environment, pre-trained representations yield five times higher scores compared to non-pre-trained counterparts  [ 131 ] .

 
 
 Several studies have further explored the application of LLMs in offline RL, demonstrating their versatility and effectiveness across various tasks and benchmarks. Reference  [ 132 ] investigated the transferability of general language models on specific RL tasks. Fined-tuned on offline RL tasks (control, games), these general language models outperform Decision Transformer and reduce training time by 3-6x on D4RL benchmark  [ 133 ] . Reference  [ 129 ] used pre-trained LLM as a general scaffold for task-specific model learning, where goals were added along with observations as the input for LLM. Results on embodied decision-making tasks demonstrate that their proposed method outperforms others with less training data, especially when generalizing to novel tasks. In addition, they found representations in pre-trained language models can aid learning and generalization even outside of language. To unify language reasoning with actions in a single policy, reference  [ 134 ] generated textual captions interleaved with actions when training the Transformer-based policy. Results show that by using captions describing the next subgoals, the reasoning policy can consistently outperform the caption-free baseline. For scenarios where data collection is costly and risky, reference  [ 130 ] proposed a general framework to effectively use pre-trained LLM for offline RL. The pre-trained LLM are based on DT. To combine the pre-trained knowledge and task-related domain knowledge, they fine-tuned pre-trained LLM with Low-Rank Adaptation (LoRA) method. The architecture of Transformer is based on GPT-2 model with 12 layers and 12 attention heads. To fine-tune the LLM, they obtained the trajectory data from the D4RL dataset. Results show their method achieves state-of-the-art performance in sparse-reward tasks with limited data samples. To integrate multimodal data, e.g., vision and language, into the offline RL, reference  [ 135 ] co-fine-tuned vision-language models on both robotic trajectory data and Internet-scale vision-language tasks, e.g., visual question answering. In the framework, they incorporate the actions as natural language tokens and co-fine-tune models on both vision and language datasets. Results show that such co-fine-tune methods can increase generalization performance and the chain of thought reasoning can help the agent perform multi-stage semantic reasoning and solve complex tasks.

 
 
 

### VI-B Action-Guiding 

 
 The action-guider role is illustrated in Fig.  5 (ii). As an action-guider, LLM guides the action selection by either generating reasonable action candidates or expert actions. By instructing the action selection, LLM improves sample efficiency and exploration efficiency posed by enormous action spaces and natural language.

 
 

#### VI-B 1 Action Candidates

 
 In environments such as text-based games, action spaces are large and only a tiny fraction of actions are accessible. Although RL agents can learn through extensive trials, they often face exploration efficiency issues, especially in multi-task settings where agents must manage various tasks simultaneously. LLMs address this challenge by generating a reduced set of action candidates based on task understanding. These candidates are likely to yield high rewards and are applicable across multiple tasks, enhancing exploration efficiency and reducing the need for ineffective exploration. Reference  [ 136 ] trained a GPT-2 model to generate the candidates. To maximize long-term rewards, another neural network is used to calculate the Q-values of these candidates. When tested in 28 man-made text games from Jericho framework  [ 137 ] , they found the method excludes non-useful actions, speeds up the exploration and consistently achieves higher scores by more than 20%. Following this work, another study  [ 83 ] proposed the SayCan framework, where instruction-following robots are integrated with an embodied LLM to understand tasks. When receiving instructions, the embodied LLM generates a high-level step-by-step plan. When acting, LLM produces action candidates based on task prompts and then the candidate with the largest critic value is executed. Being evaluated in an office kitchen with real-world robotic tasks from 101 instructions, their method can complete long-horizon, abstract, natural language instructions on a mobile manipulator.

 
 
 

#### VI-B 2 Expert Actions

 
 Traditional RL agents cannot converge to desirable equilibrium in human-AI collaboration or learn efficiently for complex tasks, due to the lack of expert demonstrations. With the understanding of human behavior and general knowledge, LLM solves the issues by producing high-quality expert actions to regularize RL agents. In human-AI collaboration, instructRL   [ 138 ] used LLM to generate prior policy based on human instructions and uses this prior to regularizing the RL objective. Specifically, instructRL augments the policy update function with an auxiliary term p LLM [ lang ( a t ) | lang ( τ t i ) , inst ] p_{\mathrm{LLM}}[\mathrm{lang(}a_{t}\mathrm{)|lang(}\tau_{t}^{i}\mathrm{),inst}] , the probability of choosing an action based on the trajectory and instructions. Experiments show instructRL converges to policies aligned with human preferences. Similarly, to address the sample inefficiency issue of RL, Zhou et al.   [ 139 ] included the policy difference between the student model and LLM-based teacher into RL learning loss. Their experiments on simulation platforms demonstrate that the method reduced training iterations by a factor of 1 to 9. In reference  [ 140 ] , LLM was leveraged to generate a high-level expert motion plan of robotics tasks, guiding RL policies to efficiently solve robotics control tasks. The high-level language plan breaks long-horizon tasks into stages to execute. Then, a single RL policy was trained across all states and stepped through the language plan. Their results show the proposed method solves long-horizon tasks from raw visual input spanning different benchmarks at success rates of over 85%, out-performing classical, language-based, and end-to-end approaches.

 
 
 
 

### VI-C Summarization and Outlook 

 
 Sample inefficiency and exploration inefficiency remain long-standing challenges for deep RL, particularly in environments with sparse rewards or where data collection is expensive or risky. LLM provides three ways to solve the problems. First, LLM as action-makers treat RL as a conditional sequence modeling problem. The supervised way fine-tunes pre-trained LLMs to predict future actions. Benefiting from learned prior knowledge, LLM can perform well even in out-of-distribution, sparse-reward, and long-horizon tasks. Second, LLM as action-guiders generates potential action candidates for RL. With task comprehension, these action candidates promote agents to explore potentially high task-value states, thus increasing exploration efficiency. Last, LLM generates expert actions to help RL learn from demonstrations. By incorporating demonstrations from LLM, and RL learns specific prior knowledge and improves sample efficiency.

 
 
 Safety issues are another important topic when using LLMs as decision-makers in RL, particularly for costly and risky tasks  [ 141 ] . For action-making, DT-based offline RL learns the optimal policy from pre-collected datasets. Recently, integrating safety constraints has been explored by some works using methods such as pessimistic estimations  [ 142 ] and stationary distribution correction  [ 143 ] . These methods set a constant constraint threshold before training. To dynamically adjust the threshold during deployment, a constrained decision transformer that dynamically relabels the reward based on safety-reward trade-offs was proposed  [ 144 ] . For action-guiding, LLMs are viewed as instructors to provide expert actions. Ensuring the safety of actions generated by LLMs differs from that in action-making. Leveraging ideas from LLM-based agent research, we propose several potential solutions. First, LLMs can be equipped with testing modules that execute code to evaluate action safety and feasibility  [ 145 ] . Second, implementing a carefully designed human-in-the-loop framework enables safety intervention through human oversight  [ 146 ] . Finally, developing memory modules with reasoning mechanisms that adaptively learn safety boundaries from past experiences provides a continual learning-based approach to ensuring long-term safety  [ 147 ] .

 
 
 In the following, we list the challenges and future directions of the two roles as below:

 
 • 
 
 Action-making : Directly employing pre-trained large-scale LLM to generate actions demands huge computational resources and huge data for fine-tuning it in specific tasks. In the short term, future work may consider more cost-effective methods such as the Low-Rank Adaptation (LoRA)  [ 148 ] to exploit the power of LLM in direct decision-making. Long-term goals involve developing innovative techniques to efficiently exploit the power of LLMs in direct decision-making, potentially creating hybrid models that combine the strengths of LLMs with more lightweight, task-specific architectures.

 

 • 
 
 Action-guiding : Since the LLM acts as an instructor to provide expert actions, the bias and limitations are also inherited by the agent. In addition, LLM itself cannot inherently interrogate or intervene in the environment, which limits LLMs’ capabilities to intentionally aggregate information. Short-term goal is to address the inherited biases and limitations when LLMs act as instructors providing expert actions, focusing on methods to filter or correct biased information. In the long term, it is crucial to develop mechanisms for LLMs to actively interrogate and intervene in the environment, enabling them to intentionally aggregate information and improve their capabilities through real-world interactions. Therefore, how to use the information gained from real-world interactions to improve the LLM itself in terms of actuality and reasoning is another important problem.

 

 
 
 
 
 

## VII LLM as Generator 

 
 The generative capability of LLMs can be applied to environmental simulation and behavior explanation. On the one hand, growing interests in applying RL to the real world and risky data-collection process suggests a need for imaginary rollouts  [ 149 ] ,  [ 150 ] . Possessed with a powerful modeling capability and world knowledge, LLMs can serve as a world model simulator to learn complex environmental dynamics with high fidelity by iteratively predicting the next state and reward, thus increasing the sample efficiency in model-based RL  [ 151 ] ,  [ 152 ] . On the other hand, in RL, interpretability remains an important security issue in current black-box AI systems as they are increasingly being deployed to help end-users with everyday tasks. Explanations of policy can improve the end-user understanding of the agent’s decision-making and inform the reward function design for agent learning. In such aspects, LLM can act as a policy interpreter based on their knowledge and reasoning ability. In this section, we classify the above two roles of LLM as a generator and review recent related works.

 
 
 Fig. 6: LLM as a generator. (i) World Model Simulator: LLM uses real-world data and knowledge to model dynamics, generate simulated worlds, and assist policy learning. (ii) Policy Interpreter: LLM generates interpretations of agent behavior based on state-action history and prompts, potentially leading to explainable RL. 
 
 

### VII-A World Model Simulator 

 
 Serving as a world model simulator, LLM is trained as a 1) trajectory rolloutor , which auto-regressively generates accurate trajectories for the agent to learn and plan; 2) dynamics representation learner , which predicts the latent representation of the world using representation learning. A flow chart of the world model is illustrated in Fig.  6 (i). From the real world, knowledge and real data can be collected to construct a world model simulator, which further models the dynamics representation of the world, generates trajectories and helps the policy learning of the agent in the real world.

 
 

#### VII-A 1 Trajectory Rolloutor

 
 Similar to the decision transformer  [ 153 ] in offline RL, pre-trained large-scale models were used in model-based to synthesize trajectories. In 2022, Micheli et al. proposed IRIS, an agent that employs a discrete autoencoder and an autoregressive Transformer to learn the world model for Atari games  [ 154 ] . With the equivalent of two hours of gameplay in the Atari 100k benchmark, the proposed method outperforms humans in 10 out of 26 games. Similarly, reference  [ 155 ] applied a transformer to build a sample-efficient world model for Atari games. Utilizing such a Transformer-based world model (TWM), the RL agent can solve the long-term dependency and train a state-of-the-art policy on the Atari 100k benchmark based on generated meaningful experiences from TWM. Visual RL enables the RL agent to learn from visual observations effectively. Reference  [ 156 ] proposed TransDreamer, a Transformer-based model-based RL agent that leverages a Transformer for dynamics predictions in 2D and 3D visual RL tasks. Experiments showed the TransDreamer agent can outperform Dreamer with long-range memory access and memory-based reasoning. Based on the development of supervised pre-training methods, Seo et al. proposed a framework that learns world model dynamics from action-free video representations  [ 157 ] . The framework added a video-based intrinsic bonus for better-guiding exploration to effectively encourage agents to learn diverse behaviors. The experimental results demonstrated their proposed method can improve the performances of vision-based RL on manipulation and locomotion tasks by transferring the pre-trained representations from unseen domains.

 
 
 

#### VII-A 2 Dynamics Representation Learner

 
 Using representation learning techniques, the latent representation of the future can be learned to assist decision-making. Reference  [ 114 ] introduced a visual model-based RL framework that decouples visual representation learning and dynamic learning by training an autoencoder with a vision Transformer to reconstruct pixel-given masked observations and learn the dynamics from the latent space. By such a decoupling approach, their proposed method achieved state-of-the-art performance on visual robotic tasks from Meta-world and RLBench. Utilizing the fact that language contains rich information signals and can help agents to predict the future, Lin et al. proposed Dynalang  [ 46 ] , where an agent that learns a multimodal world model to predict future text and image representations and thereby instruct the decision-making process. The algorithm is presented in Algorithm  3 . In the training part, a LLM-based world model implemented with Recurrent State Space Model (RSSM)  [ 158 ] computes the state representation z t z_{t} based on the collected transitions. After that, the representation z t z_{t} and the action a t a_{t} are fed into the RSSM to predict the future representations z t + 1 z_{t+1} . In addition, the language l ^ t \hat{l}_{t} and images x ^ t \hat{x}_{t} are predicted to reconstruct the original state. Then, the world model is trained to learn the next representation and reconstruct observations from the representations. After updating, the world model imagined (sampled) rollouts and the policy is trained to maximize the imagined rewards. Compared to other works that use language only for predicting actions, multimodal information enables Dynalang to handle tasks that require grounded language generation, obtaining higher score than methods provided with only task descriptions. To solve the out-of-distribution generalization problem in visual control tasks of RL, reference  [ 159 ] proposed the Language Grounded World Model (LanGWM), which focuses on learning language-grounded visual features to enhance the world model learning. To improve the generalization of the learned visual features, they masked the bounding boxes and predicted them with given language descriptions. Utilizing the expressing ability of language in higher-level concepts and global contexts, the proposed LanGWM method yields state-of-the-art results on out-of-distribution tests.

 
 
 Algorithm 3 Training part of Dynalang  [ 46 ] 
 
 
 0: 
 
 
  Rewards r t r_{t} , episode continue flag c t c_{t} , images x t x_{t} ,
language tokens l t l_{t} , actions a t a_{t} , model state ( h t , z t ) (h_{t},z_{t}) .

 
 
 1: 
 
 
   while training do 

 
 
 2: 
 
 
   Draw batch of transitions { ( r t , c t , x t , l t , a t ) } \{(r_{t},c_{t},x_{t},l_{t},a_{t})\} from replay buffer.

 
 
 3: 
 
 
   Use world model to compute multimodal representations z t z_{t} , future predictions z ^ t + 1 \hat{z}_{t+1} , and decode x ^ t \hat{x}_{t} , l ^ t \hat{l}_{t} , r ^ t \hat{r}_{t} , c ^ t \hat{c}_{t} .

 
 
 4: 
 
 
   Update world model to minimize ℒ pred + ℒ repr \mathcal{L}_{\text{pred}}+\mathcal{L}_{\text{repr}} .

 
 
 5: 
 
 
   Imagine rollouts from all z t z_{t} using π \pi .

 
 
 6: 
 
 
   Update actor to minimize ℒ π \mathcal{L}_{\pi} .

 
 
 7: 
 
 
   Update critic to minimize ℒ V \mathcal{L}_{V} .

 
 
 8: 
 
 
   end while 
 
 
 
 
 
 

### VII-B Policy Interpreter 

 
 Explainable RL (XRL) is an emerging subfield of both explainable machine learning and RL that has attracted considerable attention recently. XRL aims to elucidate the decision-making process of learning agents. According to a survey in explainable RL  [ 160 ] , the categories of XRL include the feature importance, learning process and MDP, and policy level. Currently, the usage of LLMs in XRL has only been limited to the policy level, i.e. , as the policy interpreter. Therefore, though there is limited literature related to this area, we think this field is important and requires more attention in the future. The following will introduce the role of LLMs as policy interpreters and an outlook regarding other categories of XRL will be provided in the last subsection.

 
 
 As the policy interpreter, LLM generates explanations with the prompts of state and action description or trajectory information. An illustration is depicted in Fig.  6 . Using the trajectory history of states and actions as context information, LLMs can be prompts to generate readable interpretations of current policies or situations for humans.

 
 
 Das et al.   [ 161 ] proposed a unified framework called State2Explanation (S2E), that learns a joint embedding model between state-action pairs and concept-based explanation. Based on the learned models, the explanation can help inform reward shaping during an agent’s training and provide insights to end-users at deployment. Another work   [ 162 ] first distilled the policy into a decision tree, derives the decision path, and then prompts an LLM to generate a natural language explanation based on the decision path. Additionally, Lu et al.   [ 163 ] introduced a framework that decomposes the overall reward into multiple sub-rewards based on specific object properties, defines actions as high-level motion primitives executed at precise 3D positions to simplify decision-making, and integrates LLMs to enable interactive and flexible querying of explanations.

 
 
 

### VII-C Summarization and Outlook 

 
 As a generator, LLMs can be integrated into model-based RL or explainable RL, i.e. , serving as world model simulators or policy interpreters, respectively. As world model simulators, LLMs enhance model-based RL by auto-regressively generating accurate trajectories (trajectory rollout) and by predicting latent world representations (world representation learners), significantly improving sample efficiency and decision-making accuracy. In the realm of explainable RL, LLMs provide valuable insights for both end-user understanding and reward shaping by generating explanations based on trajectory information. However, the current usages of LLMs in explainable RL are rather limited and have great potential for future work. Below are discussions on the key limitations and future work directions of the two roles:

 
 • 
 
 World Model Simulator : LLMs encounter challenges in aligning their abstract knowledge with the specific requirements of different environments, leading to limitations in functional competence and grounding. This misalignment affects their effectiveness in generating trajectories and interacting with the environment. Additionally, the current model-based agents, which rely on purely observational world models, present difficulties for human adaptation. These models are typically modifiable only through observational data, which is not an effective means for humans to communicate complex intentions or adjustments. Furthermore, while LLMs hold promise for enhancing multi-task learning by generating trajectories and dynamic representations applicable to multiple tasks, there remains a scarcity of research exploring this potential.Looking ahead, in the short term, researchers might prioritize improving LLMs’ alignment with specific environment requirements. For the long term, a promising direction would be to integrate language instructions or an adapter into the world model. This approach could lead to a more flexible and adaptive world model that can better accommodate human intentions and adjustments, as well as support more effective multi-task learning through enhanced trajectory generation and dynamic representations.

 

 • 
 
 Policy Interpreter : As policy interpreters, the quality of explanations depends on the LLM’s understanding of feature representations and implicit logic of policy. How to utilize domain knowledge or examples to improve the understanding of a complex policy is still a major issue. In the short term, researchers could focus on enhancing LLMs’ ability to interpret the correlation between observations and policy selection, providing insights into the decision-making process of RL agents. This could involve developing techniques to better leverage domain knowledge and examples for improving LLMs’ understanding of complex policies. In the long-term, the field could explore advanced applications of LLMs in explainable RL, such as analyzing the learning process and MDP to reveal influences on agent behavior. Researchers could develop sophisticated prompting techniques for LLMs to answer nuanced “why” and “why-not” questions with MDP context, providing deeper explanations of agent decision-making.

 

 
 
 
 
 

## VIII Discussion 

 
 In previous sections, we introduced the concept “LLM-enhanced RL” and developed a corresponding framework, which we extended to the integration of multimodal AI models such as visual-language models. We then discussed the different LLM-enhanced RL approaches.
This section provides a comprehensive analysis of the LLM-enhanced RL approach. We begin with a comparative analysis of the different LLM roles, highlighting their advantages and limitations. Following this, we explore potential real-world applications of the LLM-enhanced RL paradigm. Finally, we discuss future opportunities and challenges, taking into account both the untapped capabilities and inherent limitations of LLMs, with a particular focus on their application in multimodal information environments.

 
 

### VIII-A Comparison of Different LLM-Enhanced RL Approaches 

 
 This section provides a comparative analysis of the four LLM-enhanced RL approaches, helping researchers understand the strengths and limitations of each approach.

 
 • 
 
 Information Processor : As an information processor, LLM excels in handling complex, multimodal inputs, particularly in translating natural language instructions or environment information into a format more readily usable by RL agents. This role significantly enhances the agent’s ability to understand and interact with complex environments. However, it faces challenges in computational efficiency and may struggle with highly specialized domain knowledge not covered in its pre-training.

 

 • 
 
 Reward Designer : LLMs serving as reward designers offer a more intuitive and flexible approach to defining reward functions, especially in complex or sparse-reward environments. This role can significantly improve the alignment of RL objectives with human intentions. The main limitation lies in ensuring that generated rewards accurately reflect task-specific nuances and long-term goals, particularly in highly specialized domains.

 

 • 
 
 Decision-Maker : As decision-makers, LLMs can either directly generate actions or guide action selection, leveraging their vast knowledge base to improve sample efficiency and exploration in RL. This role is particularly effective in tasks requiring complex reasoning or long-term planning. However, it may face challenges in real-time decision-making scenarios due to computational overhead and may inherit biases present in the LLM’s training data.

 

 • 
 
 Generator : In the generator role, LLMs can simulate complex environments for model-based RL and provide interpretable explanations of RL policies. This capability is invaluable for improving sample efficiency and making RL more transparent and understandable. The main challenges include aligning generated simulations with real-world dynamics and ensuring the relevance and accuracy of policy explanations.

 

 
 
 
 

### VIII-B Applications of LLM-Enhanced RL 

 
 Based on the characteristics of LLM-enhanced RL, such as multimodal information understanding and multi-task learning and generation, we believe that LLM-enhanced RL opens up a wide array of potential applications. Here we list several applications to inspire researchers.

 
 • 
 
 Robotics : RL is widely used in robots to learn how to make decisions and execute actions to achieve goals. Utilizing natural language understanding and general logical reasoning abilities, LLM-enhanced RL can 1) improve the efficiency of human-robot interaction, 2) help robots understand human needs and behavioral logic, and 3) enhance decision-making and planning capabilities.

 

 • 
 
 Autonomous Driving : Autonomous driving uses RL to make decisions in complex, ever-changing environments that involve understanding both sensor data (visual, lidar, radar) and contextual information (traffic laws, human behavior). LLM-enhanced RL could employ LLMs to 1) process this multimodal information and natural language instructions; or 2) design comprehensive rewards based on multi-disciplinary metrics such as safety, efficiency, and passenger comfort.

 

 • 
 
 Energy Management : In the energy system, operators or users apply RL to efficiently manage the usage, transportation, conversion and storage of multiple energy with high uncertainty brought by renewable resources. LLM-enhanced RL in such cases can 1) improve the RL agents’ ability to handle multi-objective tasks, such as economy, safety, and low carbon, by reward function designing, and 2) increase the sample efficiency for the new energy system.

 

 • 
 
 Healthcare Recommendation : RL is used to learn recommendations or suggestions in healthcare  [ 164 ] . LLMs can utilize the domain knowledge to analyze the vast amount of patient data and medical histories, therefore, 1) accelerating the learning process of RL recommendation and 2) providing more accurate diagnostic and treatment recommendations in healthcare.

 

 
 
 
 

### VIII-C Opportunities for LLM-Enhanced RL 

 
 Although current works in LLM-enhanced RL have already shown better performance in several aspects, more unexplored areas remain to be explored and may lead to significant improvement. Below, we summarize the potential opportunities of LLM-enhanced RL from both the perspectives of RL and LLM capabilities, respectively.

 
 • 
 
 RL : Existing work such as  [ 138 , 134 , 129 ] mainly focus on the general RL while various specialized branches of RL are still under-exploited, e.g. multi-agent RL, safe RL, transfer RL, explainable RL, multi-task RL, in-context RL and human-centric RL. In the multi-agent area, compared to various works about multi-LLM-agent collaboration  [ 165 ] ,  [ 166 ] , LLM-based multi-agent RL remains largely unexplored  [ 167 ] . A recent survey discussed some open research problems within this field  [ 168 ] . LLM-enhanced strategies could be employed to facilitate communication and collaboration among RL agents. The natural language understanding capabilities of LLMs can be used to interpret and generate instructions or strategies among agents, enhancing cooperative behaviors or competition strategies; Safe RL could benefit from the reasoning and predictive capabilities of LLMs to design cost functions that encourage safety criteria compliance, reducing the risks of dangerous exploration; In transfer RL, LLMs can assist in identifying similarities between tasks or environments, leveraging their vast knowledge base to facilitate knowledge transfer and thus improve learning efficiency and adaptability across different tasks; For multi-task RL, LLMs enhance agents by processing diverse entity representations and aligning instructions (Information Processor), generating reduced action sets and expert actions for various tasks (Decision-Maker), and creating task-specific trajectories and dynamic representations (Generator). Recently, in-context RL has emerged as a promising field by leveraging the in-context learning capability of LLMs to solve decision-making problems  [ 169 ] . Given a query state and an in-context offline dataset, a trained LLM exhibits both online exploration and offline conservation while avoiding extensive trial and error and is sample-efficient. Additionally, human-centric RL is another prominent field in healthcare and robotics, where AI and humans learn and communicate with each other for collaboration  [ 170 ] . In this setting, an LLM naturally serves as a perfect mediator to facilitate bidirectional communication through natural language interactions.

 

 • 
 
 LLM : While LLMs have been integrated with RL in various methods, several promising directions remain to be explored to further enhance LLM capabilities for RL. It can be divided into the language model view and the language agent view as follows:

 
 – 
 
 From the model perspective : LLM could be enhanced by an external knowledge base and continual learning. Retrieval-augmented generation (RAG) techniques help LLM to retrieve the most relevant data in an external database, which could be used to attach knowledge in the task domains  [ 171 ] . Continual learning techniques are also a hot field that enables models to continuously acquire new knowledge while retaining previously learned capabilities  [ 172 ] . This technique allows both LLMs and RL agents to continuously evolve and adapt to new tasks or environments while maintaining their fundamental pre-trained abilities, leading to more flexible and robust learning systems.

 

 – 
 
 From the agent perspective : Equipping LLM-based agents with specialized modules—namely, planning modules, memory modules, and action modules—can significantly enhance the capabilities of LLMs  [ 173 , 174 ] . In the planning module, multi-step reasoning techniques such as CoT-based reasoning  [ 74 ] and Monte Carlo tree search (MCTS)  [ 175 ] can be used to enhance LLM’s long-term planning ability, improving the long-term task decomposition ability of RL agents; for the memory module, long- and short-term human-memories and corresponding memory retrieval mechanisms provide a way to store previous experiences and learn adaptively along with the RL agents  [ 176 ] ; for the action module, tool integration presents another promising direction to unlock new possibilities for LLMs. External tools such as mathematical solvers  [ 177 ] and internet browsers  [ 178 ] can augment LLMs’ capabilities in RL tasks requiring complex mathematical computations and real-time information processing, respectively. Furthermore, collaboration of multiple LLM agents  [ 179 ] is another promising direction. With each LLM playing different roles and acquiring different information, multiple distinctive LLM agents solve complex problems by communicating and exchanging information. In the context of LLM-enhanced RL, different LLMs could potential serves different roles and work together to guiding RL. For example, in a group of three LLM agents, one LLM focuses on guiding short-term or immediate problem-solving for RL; another LLM is responsible for long-term planning for RL; the last LLM serves as a coordinator responsible for overseeing the group and adjusting horizon consideration accordingly.

 

 
 

 
 
 
 

### VIII-D Challenges of LLM-Enhanced RL 

 
 While LLM improves different issues in the RL paradigm, the success of LLM-enhanced RL is inherently tied to the capabilities and limitations of the underlying LLM. The primary concerns revolve around the inherent limitations of LLMs, adaptability of LLMs in RL environments, computational demands, and the broader implications of deploying such systems in real-world scenarios.

 
 
 
 • 
 
 Inherent Limitations of LLMs : The effectiveness of LLM-enhanced RL systems is fundamentally constrained by the inherent capabilities and limitations of the underlying language models. The presence of systematic biases and potential hallucinations in pre-trained LLMs can significantly impact the reliability of multimodal input interpretation, potentially compromising the overall performance of the RL agent. This necessitates the development of robust evaluation frameworks to systematically characterize and delineate the capability boundaries of LLMs given specified RL contexts. Furthermore, incorporating uncertainty quantification methods aids in identifying unreliable answers, increasing the trustworthiness of LLMs’ responses  [ 180 ] .

 

 • 
 
 Adaptability of LLMs in RL Environments : Despite the vast knowledge base of LLMs, they may struggle to adapt to specific or novel RL task environments that are not well-represented in their training data. This calls for methods to expand the task-related knowledge for LLM and ground LLMs’ inherent knowledge in specific domains. In terms of expanding the task-related knowledge, RAG-related techniques attach LLMs with the external domain-knowledge without further training  [ 171 ] . When fine-tuning LLMs, employing data augmentation techniques such as generating synthetic data is another way to expand the diversity of training scenarios and improve generalization to novel environments  [ 181 ] . Additionally, continual learning mechanisms help preserve previously acquired knowledge while adapting to new information  [ 182 ] . For domain-specific knowledge grounding, approaches include training value functions to evaluate the long-term utility of LLM instructions  [ 83 ] . Expert trajectories provide another valuable source of information, allowing LLMs to learn optimal decision-making patterns through in-context learning.

 

 • 
 
 Computational Demands : The integration of LLMs into the RL learning process introduces complexities in terms of computational overhead and inference times, which slow down the learning process of RL. To address this challenge, several potential solutions have been proposed across different levels. At the data level, input compression techniques such as prompt pruning  [ 183 ] can be employed to directly shorten the model input, significantly reducing inference time without substantial loss in performance. At the model level, efficient architecture designs like mixture-of-experts (MoE)  [ 184 ] enable conditional computation, activating only relevant parts of the model for each input, thus reducing computational costs. Similarly, structured state space models (SSM)  [ 185 ] offer linear-time complexity for sequence modeling, providing a more efficient alternative to traditional attention mechanisms. At the system level, advanced caching strategies  [ 186 ] and asynchronous processing techniques  [ 187 ] can be implemented to reuse intermediate computations and parallelize the execution, respectively.

 

 • 
 
 Ethical, Legal, and Safety Concerns : In practical usages, the use of LLMs involves complex ethical, legal, and safety concerns. Data privacy, intellectual property, and accountability for AI decisions should be carefully discussed. To address these issues, researchers are developing robust frameworks for responsible AI deployment. At the privacy level, differential privacy techniques  [ 188 ] are being implemented to protect individual data during training and inference. For transparency, efforts focus on developing interpretable AI systems, allowing stakeholders to audit AI-driven decisions  [ 189 ] . To enhance system robustness, adversarial training methods are being explored to strengthen LLM-RL systems against potential attacks  [ 190 ] . Regarding ethical and legal issues, a recent survey also analyzed potential solutions to integrate ethical standards and societal values into LLM   [ 191 ] .

 

 
 
 
 
 

## IX Conclusion 

 
 LLMs, with their pre-trained knowledge bases and powerful capabilities such as reasoning and in-context learning, present as a viable solution to enhance RL in terms of natural language understanding, multi-task generalization, task planning, and sample efficiency. In this survey, we have defined this paradigm as LLM-enhanced RL and summarized its characteristics, together with opportunities and challenges. To formalize the research scope and methodology of LLM-enhanced RL, we propose a structured framework to systematically categorize the roles of LLM based on the functionalities within the classical agent-environment interaction paradigm. According to functionalities, we categorize the roles of LLMs into information processor, reward designer, decision-maker, and generator. For each category, we review current literature based on their methods and applications, outline the methodologies, discuss the addressed issues, and provide insights into future directions. These are listed below:

 
 • 
 
 Information Processor : LLMs extract observational representations and formal specification languages for RL agents, thus increasing the sample efficiency. Future directions include incorporating goal-based information and integrating multimodal environmental information to obtain stronger information extraction ability.

 

 • 
 
 Reward Designer : For intricate or un-quantifiable tasks, LLMs can leverage pre-trained knowledge to generate high-performing rewards that are notoriously difficult for humans to design. Current methods still require consistently modifying the prompts and instructions of LLMs, calling for future work on automated self-evolving frameworks without human intervention.

 

 • 
 
 Decision-Maker : LLMs generate direct actions or indirect advice for the agent to improve exploration efficiency. Huge computational overhead is a major issue in online interaction. Cost-effective methods to reduce the huge computational overhead of LLM in online RL is an important direction.

 

 • 
 
 Generator : With the generative capability and world knowledge, LLMs are used as 1) a high-fidelity world model to reduce real-world learning cost; and 2) a language-based policy interpreter to explain the agent policy. Leveraging human instructions to improve the accuracy and generalizability of the world model and policy interpreter would be a crucial direction.

 

 
 
 
 

## References

 
 
 [1] 
 
R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction . MIT press, 2018.

 

 
 [2] 
 
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller et al. , “Human-level control through deep reinforcement learning,” Nature , vol. 518, no. 7540, pp. 529–533, 2015.

 

 
 [3] 
 
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347 , 2017.

 

 
 [4] 
 
T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in International conference on machine learning . PMLR, 2018, pp. 1861–1870.

 

 
 [5] 
 
O. Vinyals, I. Babuschkin, W. M. Czarnecki, M. Mathieu, A. Dudzik, J. Chung, D. H. Choi, R. Powell et al. , “Grandmaster level in starcraft ii using multi-agent reinforcement learning,” Nature , vol. 575, no. 7782, pp. 350–354, 2019.

 

 
 [6] 
 
C. Berner, G. Brockman, B. Chan, V. Cheung, P. Déak, C. Dennison, D. Farhi, Q. Fischer et al. , “Dota 2 with large scale deep reinforcement learning,” arXiv preprint arXiv:1912.06680 , 2019.

 

 
 [7] 
 
D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou et al. , “Mastering the game of go with deep neural networks and tree search,” Nature , vol. 529, no. 7587, pp. 484–489, 2016.

 

 
 [8] 
 
J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart et al. , “Mastering atari, go, chess and shogi by planning with a learned model,” arXiv preprint arXiv:1911.08265 , 2020.

 

 
 [9] 
 
H. Zhao, Z. Liu, X. Mai, J. Zhao, J. Qiu, G. Liu, Z. Y. Dong, and A. M. Ghias, “Mobile battery energy storage system control with knowledge-assisted deep reinforcement learning,” Energy Conversion and Economics , vol. 3, no. 6, pp. 381–391, 2022.

 

 
 [10] 
 
N. B. Schmid, A. Botev, A. Hennig, A. Lerer, Q. Wu, D. Yarats, J. Foerster, T. Rocktäschel et al. , “Rebel: A general game playing ai,” Science , vol. 373, no. 6556, pp. 664–670, 2021.

 

 
 [11] 
 
N. Brown, A. Lerer, S. Gross, and T. Sandholm, “Superhuman ai for multiplayer poker,” Science , vol. 365, no. 6456, pp. 885–890, 2020.

 

 
 [12] 
 
A. Vaswani, “Attention is all you need,” Advances in Neural Information Processing Systems , 2017.

 

 
 [13] 
 
A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems , vol. 25, 2012.

 

 
 [14] 
 
Y. Jiang, S. S. Gu, K. P. Murphy, and C. Finn, “Language as an abstraction for hierarchical deep reinforcement learning,” Advances in Neural Information Processing Systems , vol. 32, 2019.

 

 
 [15] 
 
P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid, S. Gould et al. , “Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,” in Proceedings of the IEEE conference on computer vision and pattern recognition , 2018, pp. 3674–3683.

 

 
 [16] 
 
A. Stooke, K. Lee, P. Abbeel, and M. Laskin, “Decoupling representation learning from reinforcement learning,” in International Conference on Machine Learning . PMLR, 2021, pp. 9870–9879.

 

 
 [17] 
 
J. Luketina, N. Nardelli, G. Farquhar, J. Foerster, J. Andreas, E. Grefenstette, S. Whiteson, and T. Rocktäschel, “A Survey of Reinforcement Learning Informed by Natural Language,” Jun. 2019.

 

 
 [18] 
 
A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese et al. , “What matters in learning from offline human demonstrations for robot manipulation,” 2021.

 

 
 [19] 
 
C. Lynch, A. Wahid, J. Tompson, T. Ding, J. Betker, R. Baruch, T. Armstrong, and P. Florence, “Interactive language: Talking to robots in real time,” 2022.

 

 
 [20] 
 
Z. Yang, K. Ren, X. Luo, M. Liu, W. Liu, J. Bian, W. Zhang, and D. Li, “Towards applicable reinforcement learning: Improving the generalization and sample efficiency with policy ensemble,” in International Joint Conference on Artificial Intelligence , 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:248887230 

 

 
 [21] 
 
W. B. Knox, A. Allievi, H. Banzhaf, F. Schmitt, and P. Stone, “Reward (mis) design for autonomous driving,” Artificial Intelligence , vol. 316, p. 103829, 2023.

 

 
 [22] 
 
F. Dworschak, S. Dietze, M. Wittmann, B. Schleich, and S. Wartzack, “Reinforcement learning for engineering design automation,” Advanced Engineering Informatics , vol. 52, p. 101612, 2022.

 

 
 [23] 
 
S. Booth, W. B. Knox, J. Shah, S. Niekum, P. Stone, and A. Allievi, “The perils of trial-and-error reward design: misdesign through overfitting and invalid task specifications,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 37, no. 5, 2023, pp. 5920–5929.

 

 
 [24] 
 
L. L. Di Langosco, J. Koch, L. D. Sharkey, J. Pfau, and D. Krueger, “Goal misgeneralization in deep reinforcement learning,” in International Conference on Machine Learning . PMLR, 2022, pp. 12 004–12 019.

 

 
 [25] 
 
R. Yang, X. Sun, and K. Narasimhan, “A generalized algorithm for multi-objective reinforcement learning and policy adaptation,” Advances in neural information processing systems , vol. 32, 2019.

 

 
 [26] 
 
J. Luketina, N. Nardelli, G. Farquhar, J. Foerster, J. Andreas, E. Grefenstette, S. Whiteson, and T. Rocktäschel, “A survey of reinforcement learning informed by natural language,” arXiv preprint arXiv:1906.03926 , 2019.

 

 
 [27] 
 
J. Devlin, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805 , 2018.

 

 
 [28] 
 
A. Radford and K. Narasimhan, “Improving language understanding by generative pre-training,” 2018. [Online]. Available: https://api.semanticscholar.org/CorpusID:49313245 

 

 
 [29] 
 
T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam et al. , “Language models are few-shot learners,” Advances in neural information processing systems , vol. 33, pp. 1877–1901, 2020.

 

 
 [30] 
 
A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung et al. , “Palm: Scaling language modeling with pathways,” Journal of Machine Learning Research , vol. 24, no. 240, pp. 1–113, 2023.

 

 
 [31] 
 
A. J. Thirunavukarasu, D. S. J. Ting, K. Elangovan, L. Gutierrez, T. F. Tan, and D. S. W. Ting, “Large language models in medicine,” Nature medicine , vol. 29, no. 8, pp. 1930–1940, 2023.

 

 
 [32] 
 
D. A. Boiko, R. MacKnight, B. Kline, and G. Gomes, “Autonomous chemical research with large language models,” Nature , vol. 624, no. 7992, pp. 570–578, 2023.

 

 
 [33] 
 
G. Jiang, Z. Ma, L. Zhang, and J. Chen, “Eplus-llm: A large language model-based computing platform for automated building energy modeling,” Applied Energy , vol. 367, p. 123431, 2024.

 

 
 [34] 
 
H. Tan, Z. Guo, Z. Lin, Y. Chen, D. Huang, W. Yuan, H. Zhang, and J. Yan, “General generative ai-based image augmentation method for robust rooftop pv segmentation,” Applied Energy , vol. 368, p. 123554, 2024.

 

 
 [35] 
 
X. Zhou, H. Zhao, Y. Cheng, Y. Cao, G. Liang, G. Liu, and J. Zhao, “Elecbench: a power dispatch evaluation benchmark for large language models,” arXiv preprint arXiv:2407.05365 , 2024.

 

 
 [36] 
 
J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, “Code as policies: Language model programs for embodied control,” in 2023 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2023, pp. 9493–9500.

 

 
 [37] 
 
T. Webb, K. J. Holyoak, and H. Lu, “Emergent analogical reasoning in large language models,” Nature Human Behaviour , vol. 7, no. 9, pp. 1526–1541, 2023.

 

 
 [38] 
 
J. Wei, J. Wei, Y. Tay, D. Tran, A. Webson, Y. Lu, X. Chen, H. Liu et al. , “Larger language models do in-context learning differently,” arXiv preprint arXiv:2303.03846 , 2023.

 

 
 [39] 
 
J. Huang and K. C.-C. Chang, “Towards reasoning in large language models: A survey,” arXiv preprint arXiv:2212.10403 , 2022.

 

 
 [40] 
 
J. Yu, X. Wang, S. Tu, S. Cao, D. Zhang-Li, X. Lv, H. Peng, Z. Yao et al. , “Kola: Carefully benchmarking world knowledge of large language models,” 2023.

 

 
 [41] 
 
I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason et al. , “Progprompt: program generation for situated robot task planning using large language models,” Autonomous Robots , pp. 1–14, 2023.

 

 
 [42] 
 
N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei et al. , “Learning to summarize from human feedback,” 2022.

 

 
 [43] 
 
E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou, “What learning algorithm is in-context learning? investigations with linear models,” 2023.

 

 
 [44] 
 
Y. Du, O. Watkins, Z. Wang, C. Colas, T. Darrell, P. Abbeel, A. Gupta, and J. Andreas, “Guiding Pretraining in Reinforcement Learning with Large Language Models,” Sep. 2023.

 

 
 [45] 
 
T. Carta, C. Romac, T. Wolf, S. Lamprier, O. Sigaud, and P.-Y. Oudeyer, “Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning,” Sep. 2023.

 

 
 [46] 
 
J. Lin, Y. Du, O. Watkins, D. Hafner, P. Abbeel, D. Klein, and A. Dragan, “Learning to Model the World with Language,” Jul. 2023.

 

 
 [47] 
 
H. Li, X. Yang, Z. Wang, X. Zhu, J. Zhou, Y. Qiao, X. Wang, H. Li et al. , “Auto mc-reward: Automated dense reward design with large language models for minecraft,” 2023.

 

 
 [48] 
 
S. Chakraborty, K. Weerakoon, P. Poddar, M. Elnoor, P. Narayanan, C. Busart, P. Tokekar, A. S. Bedi et al. , “RE-MOVE: An Adaptive Policy Design for Robotic Navigation Tasks in Dynamic Environments via Language-Based Feedback,” Sep. 2023.

 

 
 [49] 
 
J.-C. Pang, X.-Y. Yang, S.-H. Yang, and Y. Yu, “Natural language-conditioned reinforcement learning with inside-out task language development and translation,” 2023. [Online]. Available: https://arxiv.org/abs/2302.09368 

 

 
 [50] 
 
D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly et al. , “Qt-opt: Scalable deep reinforcement learning for vision-based robotic manipulation,” 2018.

 

 
 [51] 
 
K. Wang, B. Kang, J. Shao, and J. Feng, “Improving generalization in reinforcement learning with mixture regularization,” Advances in Neural Information Processing Systems , vol. 33, pp. 7968–7978, 2020.

 

 
 [52] 
 
R. Devidze, P. Kamalaruban, and A. Singla, “Exploration-guided reward shaping for reinforcement learning under sparse rewards,” Advances in Neural Information Processing Systems , vol. 35, pp. 5829–5842, 2022.

 

 
 [53] 
 
A. Singh, L. Yang, K. Hartikainen, C. Finn, and S. Levine, “End-to-end robotic reinforcement learning without reward engineering,” arXiv preprint arXiv:1904.07854 , 2019.

 

 
 [54] 
 
D. Hadfield-Menell, S. Milli, P. Abbeel, S. J. Russell, and A. Dragan, “Inverse reward design,” Advances in neural information processing systems , vol. 30, 2017.

 

 
 [55] 
 
C. Xiao, Y. Wu, C. Ma, D. Schuurmans, and M. Müller, “Learning to combat compounding-error in model-based reinforcement learning,” arXiv preprint arXiv:1912.11206 , 2019.

 

 
 [56] 
 
T. M. Moerland, J. Broekens, A. Plaat, C. M. Jonker et al. , “Model-based reinforcement learning: A survey,” Foundations and Trends® in Machine Learning , vol. 16, no. 1, pp. 1–118, 2023.

 

 
 [57] 
 
M. Cho, J. Park, S. Lee, and Y. Sung, “Hard tasks first: Multi-task reinforcement learning through task scheduling,” in Forty-first International Conference on Machine Learning , 2024. [Online]. Available: https://openreview.net/forum?id=haUOhXo70o 

 

 
 [58] 
 
J. Feng, M. Chen, Z. Pu, T. Qiu, and J. Yi, “Efficient multi-task reinforcement learning via task-specific action correction,” 2024. [Online]. Available: https://arxiv.org/abs/2404.05950 

 

 
 [59] 
 
L. Sun, H. Zhang, W. Xu, and M. Tomizuka, “Paco: Parameter-compositional multi-task reinforcement learning,” Advances in Neural Information Processing Systems , vol. 35, pp. 21 495–21 507, 2022.

 

 
 [60] 
 
G. Zhang, A. Jain, I. Hwang, S.-H. Sun, and J. J. Lim, “Efficient multi-task reinforcement learning via selective behavior sharing,” 2024. [Online]. Available: https://openreview.net/forum?id=LYGHdwyXUb 

 

 
 [61] 
 
N. Vithayathil Varghese and Q. H. Mahmoud, “A survey of multi-task deep reinforcement learning,” Electronics , vol. 9, no. 9, 2020. [Online]. Available: https://www.mdpi.com/2079-9292/9/9/1363 

 

 
 [62] 
 
T. Xiao, H. Chan, P. Sermanet, A. Wahid, A. Brohan, K. Hausman, S. Levine, and J. Tompson, “Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models,” Jul. 2023.

 

 
 [63] 
 
H. Yuan, C. Zhang, H. Wang, F. Xie, P. Cai, H. Dong, and Z. Lu, “Plan4mc: Skill reinforcement learning and planning for open-world minecraft tasks,” arXiv preprint arXiv:2303.16563 , 2023.

 

 
 [64] 
 
J. Rocamonde, V. Montesinos, E. Nava, E. Perez, and D. Lindner, “Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning,” Oct. 2023.

 

 
 [65] 
 
A. Vaswani, N. M. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Neural Information Processing Systems , 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:13756489 

 

 
 [66] 
 
M. Shanahan, “Talking about large language models,” ArXiv , vol. abs/2212.03551, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:254366666 

 

 
 [67] 
 
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal et al. , “Llama: Open and efficient foundation language models,” ArXiv , vol. abs/2302.13971, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:257219404 

 

 
 [68] 
 
J. Kaplan, S. McCandlish, T. J. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford et al. , “Scaling laws for neural language models,” ArXiv , vol. abs/2001.08361, 2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:210861095 

 

 
 [69] 
 
J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks et al. , “Training compute-optimal large language models,” ArXiv , vol. abs/2203.15556, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:247778764 

 

 
 [70] 
 
J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma et al. , “Emergent abilities of large language models,” Trans. Mach. Learn. Res. , vol. 2022, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:249674500 

 

 
 [71] 
 
V. Sanh, A. Webson, C. Raffel, S. H. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler et al. , “Multitask prompted training enables zero-shot task generalization,” arXiv preprint arXiv:2110.08207 , 2021.

 

 
 [72] 
 
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal et al. , “Training language models to follow instructions with human feedback,” 2022.

 

 
 [73] 
 
Y. Cheng, H. Zhao, X. Zhou, J. Zhao, Y. Cao, and C. Yang, “Gaia–a large language model for advanced power dispatch,” arXiv preprint arXiv:2408.03847 , 2024.

 

 
 [74] 
 
J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le et al. , “Chain-of-thought prompting elicits reasoning in large language models,” 2023.

 

 
 [75] 
 
S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan, “Tree of thoughts: Deliberate problem solving with large language models,” 2023.

 

 
 [76] 
 
M. Besta, N. Blach, A. Kubicek, R. Gerstenberger, L. Gianinazzi, J. Gajda, T. Lehmann, M. Podstawski et al. , “Graph of thoughts: Solving elaborate problems with large language models,” 2023.

 

 
 [77] 
 
Z. Rao, Y. Wu, Z. Yang, W. Zhang, S. Lu, W. Lu, and Z. Zha, “Visual navigation with multiple goals based on deep reinforcement learning,” IEEE Transactions on Neural Networks and Learning Systems , vol. 32, no. 12, pp. 5445–5455, 2021.

 

 
 [78] 
 
C. Huang, R. Zhang, M. Ouyang, P. Wei, J. Lin, J. Su, and L. Lin, “Deductive reinforcement learning for visual autonomous urban driving navigation,” IEEE Transactions on Neural Networks and Learning Systems , vol. 32, no. 12, pp. 5379–5391, 2021.

 

 
 [79] 
 
M. Yang, W. Huang, W. Tu, Q. Qu, Y. Shen, and K. Lei, “Multitask learning and reinforcement learning for personalized dialog generation: An empirical study,” IEEE transactions on neural networks and learning systems , vol. 32, no. 1, pp. 49–62, 2020.

 

 
 [80] 
 
Z. He, J. Li, F. Wu, H. Shi, and K.-S. Hwang, “Derl: Coupling decomposition in action space for reinforcement learning task,” IEEE Transactions on Emerging Topics in Computational Intelligence , 2023.

 

 
 [81] 
 
Y. Liu, Y. Zhang, Y. Wang, F. Hou, J. Yuan, J. Tian, Y. Zhang, Z. Shi et al. , “A survey of visual transformers,” IEEE Transactions on Neural Networks and Learning Systems , 2023.

 

 
 [82] 
 
Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan et al. , “Eureka: Human-Level Reward Design via Coding Large Language Models,” Oct. 2023.

 

 
 [83] 
 
M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu et al. , “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances,” Aug. 2022.

 

 
 [84] 
 
K. Nottingham, P. Ammanabrolu, A. Suhr, Y. Choi, H. Hajishirzi, S. Singh, and R. Fox, “Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling,” Apr. 2023.

 

 
 [85] 
 
A. Srinivas, M. Laskin, and P. Abbeel, “CURL: Contrastive Unsupervised Representations for Reinforcement Learning,” Sep. 2020.

 

 
 [86] 
 
M. Schwarzer, A. Anand, R. Goel, R. D. Hjelm, A. Courville, and P. Bachman, “Data-efficient reinforcement learning with self-predictive representations,” arXiv preprint arXiv:2007.05929 , 2020.

 

 
 [87] 
 
F. Paischer, T. Adler, V. Patil, A. Bitto-Nemling, M. Holzleitner, S. Lehner, H. Eghbal-zadeh, and S. Hochreiter, “History Compression via Language Models in Reinforcement Learning,” Feb. 2023.

 

 
 [88] 
 
F. Paischer, T. Adler, M. Hofmarcher, and S. Hochreiter, “Semantic helm: A human-readable memory for reinforcement learning,” Advances in Neural Information Processing Systems , vol. 36, 2024.

 

 
 [89] 
 
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell et al. , “Learning transferable visual models from natural language supervision,” in International conference on machine learning . PMLR, 2021, pp. 8748–8763.

 

 
 [90] 
 
Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdinov, “Transformer-xl: Attentive language models beyond a fixed-length context,” 2019. [Online]. Available: https://arxiv.org/abs/1901.02860 

 

 
 [91] 
 
A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748 , 2018.

 

 
 [92] 
 
W. K. Kim, S. Kim, H. Woo et al. , “Efficient policy adaptation with contrastive prompt ensemble for embodied agents,” Advances in Neural Information Processing Systems , vol. 36, 2024.

 

 
 [93] 
 
R. P. Poudel, H. Pandya, S. Liwicki, and R. Cipolla, “Recore: Regularized contrastive representation learning of world model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024, pp. 22 904–22 913.

 

 
 [94] 
 
S. Basavatia, K. Murugesan, and S. Ratnakar, “Starling: Self-supervised training of text-based reinforcement learning agent with large language models,” arXiv preprint arXiv:2406.05872 , 2024.

 

 
 [95] 
 
R. Patel, E. Pavlick, and S. Tellex, “Grounding language to non-markovian tasks with no supervision of task specifications.” in Robotics: Science and Systems , vol. 2020, 2020.

 

 
 [96] 
 
T. R. Sumers, M. K. Ho, R. D. Hawkins, K. Narasimhan, and T. L. Griffiths, “Learning rewards from linguistic feedback,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 35, no. 7, 2021, pp. 6002–6010.

 

 
 [97] 
 
J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, “Code as Policies: Language Model Programs for Embodied Control,” May 2023.

 

 
 [98] 
 
C. H. Song, J. Wu, C. Washington, B. M. Sadler, W.-L. Chao, and Y. Su, “Llm-planner: Few-shot grounded planning for embodied agents with large language models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 2998–3009.

 

 
 [99] 
 
B. A. Spiegel, Z. Yang, W. Jurayj, B. Bachmann, S. Tellex, and G. Konidaris, “Informing reinforcement learning agents by grounding language to markov decision processes,” in Workshop on Training Agents with Foundation Models at RLC 2024 , 2024. [Online]. Available: https://openreview.net/forum?id=uFm9e4Ly26 

 

 
 [100] 
 
B. Gordon, Y. Bitton, Y. Shafir, R. Garg, X. Chen, D. Lischinski, D. Cohen-Or, and I. Szpektor, “Mismatch quest: Visual and textual feedback for image-text misalignment,” in European Conference on Computer Vision . Springer, 2025, pp. 310–328.

 

 
 [101] 
 
Y. Wang, J. He, D. Wang, Q. Wang, B. Wan, and X. Luo, “Multimodal transformer with adaptive modality weighting for multimodal sentiment analysis,” Neurocomputing , vol. 572, p. 127181, 2024.

 

 
 [102] 
 
X. Zhao, S. Poria, X. Li, Y. Chen, and B. Tang, “Toward robust multimodal learning using multimodal foundational models,” arXiv preprint arXiv:2401.13697 , 2024.

 

 
 [103] 
 
F. Lygerakis, V. Dave, and E. Rueckert, “M2curl: Sample-efficient multimodal reinforcement learning via self-supervised representation learning for robotic manipulation,” arXiv preprint arXiv:2401.17032 , 2024.

 

 
 [104] 
 
J. Eschmann, “Reward function design in reinforcement learning,” Reinforcement Learning Algorithms: Analysis and Applications , pp. 25–33, 2021.

 

 
 [105] 
 
J. Andreas, D. Klein, and S. Levine, “Modular multitask reinforcement learning with policy sketches,” in International conference on machine learning . PMLR, 2017, pp. 166–175.

 

 
 [106] 
 
S. Mirchandani, S. Karamcheti, and D. Sadigh, “Ella: Exploration through learned language abstraction,” Advances in Neural Information Processing Systems , vol. 34, pp. 29 529–29 540, 2021.

 

 
 [107] 
 
M. Kwon, S. M. Xie, K. Bullard, and D. Sadigh, “Reward Design with Language Models,” in The Eleventh International Conference on Learning Representations , Sep. 2022.

 

 
 [108] 
 
Y. Wu, Y. Fan, P. P. Liang, A. Azaria, Y. Li, and T. M. Mitchell, “Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals,” Oct. 2023.

 

 
 [109] 
 
K. Chu, X. Zhao, C. Weber, M. Li, and S. Wermter, “Accelerating Reinforcement Learning of Robotic Manipulations via Feedback from Large Language Models,” Nov. 2023.

 

 
 [110] 
 
A. Adeniji, A. Xie, C. Sferrazza, Y. Seo, S. James, and P. Abbeel, “Language reward modulation for pretraining reinforcement learning,” arXiv preprint arXiv:2308.12270 , 2023.

 

 
 [111] 
 
V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter,” 2020. [Online]. Available: https://arxiv.org/abs/1910.01108 

 

 
 [112] 
 
C. Kim, Y. Seo, H. Liu, L. Lee, J. Shin, H. Lee, and K. Lee, “Guide Your Agent with Adaptive Multimodal Rewards,” Oct. 2023.

 

 
 [113] 
 
S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, “R3m: A universal visual representation for robot manipulation,” 2022. [Online]. Available: https://arxiv.org/abs/2203.12601 

 

 
 [114] 
 
Y. Seo, D. Hafner, H. Liu, F. Liu, S. James, K. Lee, and P. Abbeel, “Masked World Models for Visual Control,” May 2023.

 

 
 [115] 
 
K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang et al. , “Ego4d: Around the world in 3,000 hours of egocentric video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp. 18 995–19 012.

 

 
 [116] 
 
Y. Wang, Z. Sun, J. Zhang, Z. Xian, E. Biyik, D. Held, and Z. Erickson, “Rl-vlm-f: Reinforcement learning from vision language foundation model feedback,” 2024. [Online]. Available: https://arxiv.org/abs/2402.03681 

 

 
 [117] 
 
W. Yu, N. Gileadi, C. Fu, S. Kirmani, K.-H. Lee, M. G. Arenas, H.-T. L. Chiang, T. Erez et al. , “Language to rewards for robotic skill synthesis,” arXiv preprint arXiv:2306.08647 , 2023.

 

 
 [118] 
 
A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri et al. , “Self-refine: Iterative refinement with self-feedback,” 2023.

 

 
 [119] 
 
J. Song, Z. Zhou, J. Liu, C. Fang, Z. Shu, and L. Ma, “Self-Refined Large Language Model as Automated Reward Function Designer for Deep Reinforcement Learning in Robotics,” Oct. 2023.

 

 
 [120] 
 
T. Xie, S. Zhao, C. H. Wu, Y. Liu, Q. Luo, V. Zhong, Y. Yang, and T. Yu, “Text2reward: Reward shaping with language models for reinforcement learning,” in The Twelfth International Conference on Learning Representations , 2024. [Online]. Available: https://openreview.net/forum?id=tUM39YTRxH 

 

 
 [121] 
 
E. Ferrara, “Should chatgpt be biased? challenges and risks of bias in large language models,” arXiv preprint arXiv:2304.03738 , 2023.

 

 
 [122] 
 
I. O. Gallegos, R. A. Rossi, J. Barrow, M. M. Tanjim, S. Kim, F. Dernoncourt, T. Yu, R. Zhang et al. , “Bias and fairness in large language models: A survey,” Computational Linguistics , pp. 1–79, 2024.

 

 
 [123] 
 
L. Gao, J. Schulman, and J. Hilton, “Scaling laws for reward model overoptimization,” in Proceedings of the 40th International Conference on Machine Learning , ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 10 835–10 866.

 

 
 [124] 
 
S. Chakraborty, A. Bhaskar, A. Singh, P. Tokekar, D. Manocha, and A. S. Bedi, “Rebel: A regularization-based solution for reward overoptimization in reinforcement learning from human feedback,” arXiv preprint arXiv:2312.14436 , 2023.

 

 
 [125] 
 
A. Radwan, L. Zaafarani, J. Abudawood, F. AlZahrani, and F. Fourat, “Addressing bias through ensemble learning and regularized fine-tuning,” arXiv preprint arXiv:2402.00910 , 2024.

 

 
 [126] 
 
Y. Inoue and H. Ohashi, “Prompter: Utilizing large language model prompting for a data efficient embodied instruction following,” arXiv preprint arXiv:2211.03267 , 2022.

 

 
 [127] 
 
A. Majumdar, A. Shrivastava, S. Lee, P. Anderson, D. Parikh, and D. Batra, “Improving vision-and-language navigation with image-text pairs from the web,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VI 16 . Springer, 2020, pp. 259–274.

 

 
 [128] 
 
M. Janner, Q. Li, and S. Levine, “Offline reinforcement learning as one big sequence modeling problem,” Advances in neural information processing systems , vol. 34, pp. 1273–1286, 2021.

 

 
 [129] 
 
S. Li, X. Puig, C. Paxton, Y. Du, C. Wang, L. Fan, T. Chen, D.-A. Huang et al. , “Pre-trained language models for interactive decision-making,” Advances in Neural Information Processing Systems , vol. 35, pp. 31 199–31 212, 2022.

 

 
 [130] 
 
R. Shi, Y. Liu, Y. Ze, S. S. Du, and H. Xu, “Unleashing the power of pre-trained language models for offline reinforcement learning,” arXiv preprint arXiv:2310.20587 , 2023.

 

 
 [131] 
 
Z. Zeng, C. Zhang, S. Wang, and C. Sun, “Goal-conditioned predictive coding for offline reinforcement learning,” Advances in Neural Information Processing Systems , vol. 36, 2024.

 

 
 [132] 
 
M. Reid, Y. Yamada, and S. S. Gu, “Can Wikipedia Help Offline Reinforcement Learning?” Jul. 2022.

 

 
 [133] 
 
J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine, “D4rl: Datasets for deep data-driven reinforcement learning,” arXiv preprint arXiv:2004.07219 , 2020.

 

 
 [134] 
 
L. Mezghani, P. Bojanowski, K. Alahari, and S. Sukhbaatar, “Think before you act: Unified policy for interleaving language reasoning with actions,” arXiv preprint arXiv:2304.11063 , 2023.

 

 
 [135] 
 
B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart et al. , “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning . PMLR, 2023, pp. 2165–2183.

 

 
 [136] 
 
S. Yao, R. Rao, M. Hausknecht, and K. Narasimhan, “Keep calm and explore: Language models for action generation in text-based games,” arXiv preprint arXiv:2010.02903 , 2020.

 

 
 [137] 
 
M. Hausknecht, P. Ammanabrolu, M.-A. Côté, and X. Yuan, “Interactive fiction games: A colossal adventure,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 34, no. 05, 2020, pp. 7903–7910.

 

 
 [138] 
 
H. Hu and D. Sadigh, “Language Instructed Reinforcement Learning for Human-AI Coordination,” Jun. 2023.

 

 
 [139] 
 
Z. Zhou, B. Hu, P. Zhang, C. Zhao, and B. Liu, “Large language model is a good policy teacher for training reinforcement learning agents,” arXiv preprint arXiv:2311.13373 , 2023.

 

 
 [140] 
 
M. Dalal, T. Chiruvolu, D. S. Chaplot, and R. Salakhutdinov, “Plan-seq-learn: Language model guided RL for solving long horizon robotics tasks,” in The Twelfth International Conference on Learning Representations , 2024. [Online]. Available: https://openreview.net/forum?id=hQVCCxQrYN 

 

 
 [141] 
 
J. Garcıa and F. Fernández, “A comprehensive survey on safe reinforcement learning,” Journal of Machine Learning Research , vol. 16, no. 1, pp. 1437–1480, 2015.

 

 
 [142] 
 
H. Xu, X. Zhan, and X. Zhu, “Constraints penalized q-learning for safe offline reinforcement learning,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 36, no. 8, 2022, pp. 8753–8760.

 

 
 [143] 
 
J. Lee, C. Paduraru, D. J. Mankowitz, N. Heess, D. Precup, K.-E. Kim, and A. Guez, “Coptidice: Offline constrained reinforcement learning via stationary distribution correction estimation,” arXiv preprint arXiv:2204.08957 , 2022.

 

 
 [144] 
 
Z. Liu, Z. Guo, Y. Yao, Z. Cen, W. Yu, T. Zhang, and D. Zhao, “Constrained decision transformer for offline safe reinforcement learning,” in International Conference on Machine Learning . PMLR, 2023, pp. 21 611–21 630.

 

 
 [145] 
 
S. Naihin, D. Atkinson, M. Green, M. Hamadi, C. Swift, D. Schonholtz, A. T. Kalai, and D. Bau, “Testing language model agents safely in the wild,” arXiv preprint arXiv:2311.10538 , 2023.

 

 
 [146] 
 
W. Huang, H. Liu, Z. Huang, and C. Lv, “Safety-aware human-in-the-loop reinforcement learning with shared control for autonomous driving,” IEEE Transactions on Intelligent Transportation Systems , pp. 1–12, 2024.

 

 
 [147] 
 
N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: Language Agents with Verbal Reinforcement Learning,” Oct. 2023.

 

 
 [148] 
 
E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” 2021.

 

 
 [149] 
 
D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to Control: Learning Behaviors by Latent Imagination,” Mar. 2020.

 

 
 [150] 
 
D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba, “Mastering Atari with Discrete World Models,” Feb. 2022.

 

 
 [151] 
 
D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” 2020.

 

 
 [152] 
 
Y. Matsuo, Y. LeCun, M. Sahani, D. Precup, D. Silver, M. Sugiyama, E. Uchibe, and J. Morimoto, “Deep learning, reinforcement learning, and world models,” Neural Networks , vol. 152, pp. 267–275, 2022.

 

 
 [153] 
 
L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas et al. , “Decision transformer: Reinforcement learning via sequence modeling,” Advances in neural information processing systems , vol. 34, pp. 15 084–15 097, 2021.

 

 
 [154] 
 
V. Micheli, E. Alonso, and F. Fleuret, “Transformers are Sample-Efficient World Models,” in The Eleventh International Conference on Learning Representations , Sep. 2022.

 

 
 [155] 
 
J. Robine, M. Höftmann, T. Uelwer, and S. Harmeling, “Transformer-based World Models Are Happy With 100k Interactions,” Mar. 2023.

 

 
 [156] 
 
C. Chen, Y.-F. Wu, J. Yoon, and S. Ahn, “TransDreamer: Reinforcement Learning with Transformer World Models,” Feb. 2022.

 

 
 [157] 
 
Y. Seo, K. Lee, S. James, and P. Abbeel, “Reinforcement Learning with Action-Free Pre-Training from Videos,” Jun. 2022.

 

 
 [158] 
 
D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in International conference on machine learning . PMLR, 2019, pp. 2555–2565.

 

 
 [159] 
 
R. P. K. Poudel, H. Pandya, C. Zhang, and R. Cipolla, “LanGWM: Language Grounded World Model,” Nov. 2023.

 

 
 [160] 
 
S. Milani, N. Topin, M. Veloso, and F. Fang, “A survey of explainable reinforcement learning,” 2022.

 

 
 [161] 
 
D. Das, S. Chernova, and B. Kim, “State2explanation: Concept-based explanations to benefit agent learning and user understanding,” in Thirty-seventh Conference on Neural Information Processing Systems , 2023. [Online]. Available: https://openreview.net/forum?id=xGz0wAIJrS 

 

 
 [162] 
 
J. Lin, Y. Du, O. Watkins, D. Hafner, P. Abbeel, D. Klein, and A. Dragan, “Learning to model the world with language,” arXiv preprint arXiv:2308.01399 , 2023.

 

 
 [163] 
 
W. Lu, X. Zhao, S. Magg, M. Gromniak, M. Li, and S. Wermterl, “A closer look at reward decomposition for high-level robotic explanations,” in 2023 IEEE International Conference on Development and Learning (ICDL) . IEEE, 2023, pp. 429–436.

 

 
 [164] 
 
C. Yu, J. Liu, S. Nemati, and G. Yin, “Reinforcement learning in healthcare: A survey,” ACM Computing Surveys (CSUR) , vol. 55, no. 1, pp. 1–36, 2021.

 

 
 [165] 
 
S. Agashe, Y. Fan, and X. E. Wang, “Evaluating multi-agent coordination abilities in large language models,” arXiv preprint arXiv:2310.03903 , 2023.

 

 
 [166] 
 
S. S. Kannan, V. L. Venkatesh, and B.-C. Min, “Smart-llm: Smart multi-agent robot task planning using large language models,” arXiv preprint arXiv:2309.10062 , 2023.

 

 
 [167] 
 
O. Slumbers, D. H. Mguni, K. Shao, and J. Wang, “Leveraging large language models for optimised coordination in textual multi-agent reinforcement learning,” 2023.

 

 
 [168] 
 
C. Sun, S. Huang, and D. Pompili, “Llm-based multi-agent reinforcement learning: Current and future directions,” arXiv preprint arXiv:2405.11106 , 2024.

 

 
 [169] 
 
J. Lee, A. Xie, A. Pacchiano, Y. Chandak, C. Finn, O. Nachum, and E. Brunskill, “Supervised pretraining can learn in-context reinforcement learning,” Advances in Neural Information Processing Systems , vol. 36, 2024.

 

 
 [170] 
 
Z. Buçinca, S. Swaroop, A. E. Paluch, S. A. Murphy, and K. Z. Gajos, “Towards optimizing human-centric objectives in ai-assisted decision-making with offline reinforcement learning,” arXiv preprint arXiv:2403.05911 , 2024.

 

 
 [171] 
 
P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis et al. , “Retrieval-augmented generation for knowledge-intensive nlp tasks,” 2021. [Online]. Available: https://arxiv.org/abs/2005.11401 

 

 
 [172] 
 
L. Wang, X. Zhang, H. Su, and J. Zhu, “A comprehensive survey of continual learning: theory, method and application,” IEEE Transactions on Pattern Analysis and Machine Intelligence , 2024.

 

 
 [173] 
 
L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang et al. , “A survey on large language model based autonomous agents,” Frontiers of Computer Science , vol. 18, no. 6, p. 186345, 2024.

 

 
 [174] 
 
Y. Cheng, C. Zhang, Z. Zhang, X. Meng, S. Hong, W. Li, Z. Wang, Z. Wang et al. , “Exploring large language model based intelligent agents: Definitions, methods, and prospects,” arXiv preprint arXiv:2401.03428 , 2024.

 

 
 [175] 
 
C. B. Browne, E. Powley, D. Whitehouse, S. M. Lucas, P. I. Cowling, P. Rohlfshagen, S. Tavener, D. Perez et al. , “A survey of monte carlo tree search methods,” IEEE Transactions on Computational Intelligence and AI in games , vol. 4, no. 1, pp. 1–43, 2012.

 

 
 [176] 
 
Z. Zhang, X. Bo, C. Ma, R. Li, X. Chen, Q. Dai, J. Zhu, Z. Dong et al. , “A survey on the memory mechanism of large language model based agents,” 2024. [Online]. Available: https://arxiv.org/abs/2404.13501 

 

 
 [177] 
 
S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, “Gorilla: Large language model connected with massive apis,” arXiv preprint arXiv:2305.15334 , 2023.

 

 
 [178] 
 
R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain et al. , “Webgpt: Browser-assisted question-answering with human feedback,” arXiv preprint arXiv:2112.09332 , 2021.

 

 
 [179] 
 
Y. Talebirad and A. Nadiri, “Multi-agent collaboration: Harnessing the power of intelligent llm agents,” arXiv preprint arXiv:2306.03314 , 2023.

 

 
 [180] 
 
Z. Lin, S. Trivedi, and J. Sun, “Generating with confidence: Uncertainty quantification for black-box large language models,” 2024. [Online]. Available: https://arxiv.org/abs/2305.19187 

 

 
 [181] 
 
R. S. Y. C. Tan, Q. Lin, G. H. Low, R. Lin, T. C. Goh, C. C. E. Chang, F. F. Lee, W. Y. Chan et al. , “Inferring cancer disease response from radiology reports using large language models with data augmentation and prompting,” Journal of the American Medical Informatics Association , vol. 30, no. 10, pp. 1657–1664, 2023.

 

 
 [182] 
 
Q. Gao, C. Zhao, Y. Sun, T. Xi, G. Zhang, B. Ghanem, and J. Zhang, “A unified continual learning framework with general parameter-efficient tuning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 11 483–11 493.

 

 
 [183] 
 
W. Zhou, Y. E. Jiang, R. Cotterell, and M. Sachan, “Efficient prompting via dynamic in-context learning,” 2023. [Online]. Available: https://arxiv.org/abs/2305.11170 

 

 
 [184] 
 
Z.-F. Gao, P. Liu, W. X. Zhao, Z.-Y. Lu, and J.-R. Wen, “Parameter-efficient mixture-of-experts architecture for pre-trained language models,” 2022. [Online]. Available: https://arxiv.org/abs/2203.01104 

 

 
 [185] 
 
A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” 2024. [Online]. Available: https://arxiv.org/abs/2312.00752 

 

 
 [186] 
 
L. D. Corro, A. D. Giorno, S. Agarwal, B. Yu, A. Awadallah, and S. Mukherjee, “Skipdecode: Autoregressive skip decoding with batching and caching for efficient llm inference,” 2023. [Online]. Available: https://arxiv.org/abs/2307.02628 

 

 
 [187] 
 
Y. Chen, Z. han Ding, Z. Wang, Y. Wang, L. Zhang, and S. Liu, “Asynchronous large language model enhanced planner for autonomous driving,” 2024. [Online]. Available: https://arxiv.org/abs/2406.14556 

 

 
 [188] 
 
Z. Charles, A. Ganesh, R. McKenna, H. B. McMahan, N. Mitchell, K. Pillutla, and K. Rush, “Fine-tuning large language models with user-level differential privacy,” 2024. [Online]. Available: https://arxiv.org/abs/2407.07737 

 

 
 [189] 
 
E. Cambria, L. Malandri, F. Mercorio, N. Nobani, and A. Seveso, “Xai meets llms: A survey of the relation between explainable ai and large language models,” 2024. [Online]. Available: https://arxiv.org/abs/2407.15248 

 

 
 [190] 
 
S. Xhonneux, A. Sordoni, S. Günnemann, G. Gidel, and L. Schwinn, “Efficient adversarial training in llms with continuous attacks,” 2024. [Online]. Available: https://arxiv.org/abs/2405.15589 

 

 
 [191] 
 
C. Deng, Y. Duan, X. Jin, H. Chang, Y. Tian, H. Liu, H. P. Zou, Y. Jin et al. , “Deconstructing the ethics of large language models from long-standing issues to new-emerging dilemmas,” 2024. [Online]. Available: https://arxiv.org/abs/2406.05392