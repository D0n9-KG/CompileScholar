A Survey on Game Playing Agents and Large Models: Methods, Applications, and Challenges 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.10249v1 [cs.AI] 15 Mar 2024 
 
 

# A Survey on Game Playing Agents and Large Models: 
 Methods, Applications, and Challenges

 
 
 Xinrun Xu
 
 Affiliation: Beijing Academy of Artificial Intelligence (BAAI)
Institute of Software, Chinese Academy of Science
 
 Email: xuxinrun20@mails.ucas.ac.cn 
 
    
 Yuxin Wang
 
 Affiliation: University of Chinese Academy of Science
Dartmouth College
 
 Email: borje@baai.ac.cn 
 
    
 Chaoyi Xu
 
    
 Ziluo Ding
 
    
 Jiechuan Jiang
 
 Affiliation: Beijing University of Posts and Telecommunications
Peking University
 
    
 Zhiming Ding
 
 Affiliation: Beijing Academy of Artificial Intelligence (BAAI)
Institute of Software, Chinese Academy of Science
 
    
 Börje F. Karlsson
 
 Note: Corresponding Author
 

 Abstract 
 
 The swift evolution of Large-scale Models (LMs), either language-focused or multi-modal, has garnered extensive attention in both academy and industry. But despite the surge in interest in this rapidly evolving area, there are scarce systematic reviews on their capabilities and potential in distinct impactful scenarios. This paper endeavours to help bridge this gap, offering a thorough examination of the current landscape of LM usage in regards to complex game playing scenarios and the challenges still open. Here, we seek to systematically review the existing architectures of LM-based Agents (LMAs) for games and summarize their commonalities, challenges, and any other insights. Furthermore, we present our perspective on promising future research avenues for the advancement of LMs in games.
We hope to assist researchers in gaining a clear understanding of the field and to generate more interest in this highly impactful research direction.
A corresponding resource, continuously updated, can be found in our GitHub repository 1 1 
 1 
 
 
 
 https://github.com/BAAI-Agents/GPA-LM .

 
 
 

## 1 Introduction

 
 Figure 1: Target areas and Challenges leveraging Large Model capabilities in Game Playing Agents. 
 
 {forest} 
 Figure 2: Taxonomy of research on complex digital game playing agents and large models. 
 
 
 The development of large-scale models (LMs), including both language and multi-modal models, has been a significant advancement in the fields of natural language processing and computer vision. Recent advancements in LMs have led to notable achievements across various applications, including text generation [ 93 ] , image understanding [ 91 ] , and robotics [ 16 ] .
This progress has led researchers to explore the application of LMs "as" agents to perform complex tasks, where LM-based Agents (LMAs) in many cases have exhibited interesting generalization capabilities compared to their traditionally trained counterparts [ 61 ; 70 ] .
The demonstrated capabilities of LMs have led to considerable interest in their application to game playing. This interest is particularly evident in popular games such as Minecraft [ 37 ] ,
where LMs’ potential to handle complex, dynamic environments is being actively explored.

 
 
 In the context of pursuing Artificial General Intelligence (AGI) research, digital games are recognized as significant for their provision of complex challenges that necessitate advanced reasoning and cognitive abilities, serving as an ideal benchmark for assessing agent and system capabilities [ 79 ; 6 ] . The process of data acquisition in gaming contexts offers advantages in terms of cost-effectiveness, controllability, safety, and diversity in comparison to real-world experiments, while preserving significant challenges.
Although attempting to analyze or formalize game AI agents and their components is hardly a recent phenomena even outside academia [ 42 ; 43 ; 44 ] , investigating the performance of LMAs within complex gaming environments is crucial for delineating their current limitations and assessing the progress towards autonomy, generalizability, informing the design of new architectures, and moving closer to a potential AGI.
Furthermore, this survey represents an inaugural comprehensive examination of Game Playing Agents and Large Models, aiming to catalyze subsequent research in this area, by providing an overview of recent endeavors that integrate LMAs with gaming playing applications (e.g., as main player, assisting human players, controlling NPCs), contextualizing and contrasting them, and identifying remaining open challenges.

 
 
 The human brain functions as a sophisticated information processing system that first transforms sensory information into perceptual representations, then uses these to construct knowledge about the world and make decisions, and finally implements decisions through action [ 13 ] . As this abstract sequence mirrors the typical iteration cycle observed in game-playing agents, which consists of perception (§ 2 ), inference (§ 3 ), and action (§ 4 ), we follow a similar organization in this survey. Figure 1 illustrates the core survey structure, covering the essence of how sensory information is converted into actions and how LMs can play a role in each step.

 
 
 Perception involves transforming raw observation information during game play into actionable insights, supporting subsequent interactions. Initial studies focused on understanding semantic information through text [ 64 ; 83 ] , while more recent work focuses on integrating visual information (e.g., [ 91 ] ).

 
 
 Inference encompasses key abilities of game agents including memorization, learning, reasoning, reflection, and decision-making; usually built upon a comprehensive cognitive framework.
The general framework is adaptable to various application contexts, meaning not all individual components are necessary in every scenario. 
Specifically, memorization is about efficient storage and retrieval of learnt knowledge to enhance common sense and game-specific insights [ 94 ; 24 ] .
Learning usually concerns skill acquisition and strategic adaptation through experiences and collaborative efforts in multi-agent systems [ 14 ] .
Reasoning is the process of processing and synthesizing information for problem-solving [ 48 ] .
Decision-making in complex games demands multi-hop inference [ 34 ] and long-term planning [ 24 ] , incorporating sequential task decomposition and collaborative decision-making to effectively respond to dynamic games.
And reflection means the process of self-improvement, where agents evaluate and adjust their strategies based on feedback [ 64 ] .
These components enable LMs-empowered agents to act effectively within the dynamic and evolving environments of modern digital games.

 
 
 Action covers the interaction back with a game environment, i.e., operations executed by the agent as response to the game state and environment feedback.
Behaviors are performed using generative coding, with techniques like iterative prompting [ 64 ] , role-specific engineering [ 14 ] , or code generations [ 64 ; 62 ] .
Dialogue interactions span agent-agent and human-agent communications, employing collaborative frameworks [ 12 ] , and conversation-driven controls for dynamic interactions [ 77 ] .
Behavioral consistency in agents can be emphasized through, e.g., structural methods like Directed Acyclic Graph (DAG) for logical action progression [ 78 ] , coupled with feedback mechanisms for environmental adaptation [ 94 ] , and reinforced by strategies like Reinforcement Learning (RL) for coherent action selection [ 84 ] .
These methodologies enable LMAs to not only handle complex tasks but also adapt behaviors to maintain consistency and alignment with game objectives in dynamic gaming contexts.

 
 
 Nonetheless, challenges remain across all stages (and in other game scenarios). Four of them being especially significant in LMAs: i) addressing the problem of hallucinations in both critic agents and structured reasoning [ 24 ; 78 ] ; ii) correcting errors through iterative learning or feedback (e.g., [ 32 ] ); iii) generalizing learnt knowledge to unseen tasks, potentially using zero-shot learning or structured adaptability [ 64 ; 90 ] ; and iv) interpretability, which requires transparent decision-making processes.
While these are demonstrated in a diversity of AI systems, they also highlight the impacts of inherent limitations of LMs against the specific demands of gaming environments.

 
 
 An overview of this survey structure is illustrated in Figure 2 . In § 2 , § 3 , and § 4 , we review how existing LM-based game agents deal with perception, inference, and behavior. In § 5 , we analyze the common challenges encountered throughout these three stages. Finally, in § 6 we discuss future open research directions for furthering the development of general game-playing agents.

 
 
 

## 2 Perception

 
 Perceiving raw observations from the gaming world and extracting information are crucial for subsequent reflection and action selection.
The primary function of perception is to transform the multi-modal space, encompassing vision (§ 2.2 ), semantics (§ 2.1 ), and audio, into inputs for the agent.
Text-based games primarily focus on communication and interaction among players, often through text or spoken words, to solve puzzles, reveal hidden information, or identify character roles, with games like Werewolf [ 83 ; 84 ] and Avalon [ 66 ; 38 ; 31 ] being quintessential examples.
Digital games [ 64 ; 7 ] , by integrating multiple modality perception channels, offer richer and more immersive experiences, enabling deeper engagement with the game world.
However, unfortunately, not much effort in the existing literature has been dedicated to the integration of audio data into the training of LMs or to optimize game agents. This remains a topic for future exploration (§ 6 ).

 
 

### 2.1 Semantics

 
 Traditionally, semantics-based perception of environmental elements (from locations to objects, characters to actions) in games relied mostly on text items and textual descriptions, encompassing a broad spectrum of descriptions, to natural language instructions and dialogues.
In categorizing the text perception methods used by LMAs in gaming, we can distinguish them by the nature and complexity of textual inputs, ranging from basic text to advanced multi-modal input. This analysis delineates four primary classes:
Basic text input, which involves simple user ideas or descriptions [ 32 ] and basic game state variables and dialogues [ 3 ; 47 ] .
Structured or role-based inputs, characterized by the inclusion of character, story, role-related information [ 48 ; 24 ; 69 ] , and skills [ 21 ] .
Environmental and context-specific inputs, including cases such as detailed game descriptors and contextual information [ 78 ; 89 ] , and natural language instructions for tasks [ 90 ; 83 ] .
Advanced multi-modal inputs, which integrate visual, audible, and textual data for decision-making [ 11 ; 85 ] , and the combination of dialogue, game state, and scripts for richer interaction [ 2 ] .
This categorization underscores the adaptive capabilities of LMAs in processing a wide spectrum of semantic information, pivotal for enhancing interaction within gaming environments.

 
 
 

### 2.2 Vision

 
 The initial efforts in LMAs showed promise, but primarily focused on translating the entire environmental experience into text by converting multimodal perception data into textual information through various models and APIs (e.g., [ 74 ] ).
These methods, however, tend to ignore or lose complex visual information in this translation , thus failing to accurately and intuitively reflecting the real environment (e.g., [ 91 ] ).
As such process can lose important information during the conversion from multi-modal to textual formats [ 79 ; 64 ] , it is likely to cause difficulty for agents to understand their situation or navigate effectively in target games, which emphasizes the need for more comprehensive sensory data processing in LMAs.
Addressing these impediments by leveraging the latest developments in Multi-modal Large Language Models (MLLMs), like [ 46 ] , becomes then a natural potential approach. By utilizing MLLMs, LMAs can obtain a richer perception of their surroundings, which facilitates more intricate cognitive processes and decision-making capabilities in multifaceted, real-world-like settings.

 
 
 The performance of various visual encoders, including both end-to-end trained and pre-trained models, in accurately capturing and learning from the unique visual aspects and gameplay dynamics of each game is evaluated in [ 54 ] .
A video game bug detecting method [ 60 ] , leverages CLIP for zero-shot learning, processes data by encoding video frames and text queries into embeddings, and calculates similarity scores to identify relevant problematic video sequences.
A benchmark called GlitchBench [ 59 ] evaluates LMM’s ability to detect and interpret video game glitches by interpreting single-frame images.
Vision-based perception is also used for Success Detection and Scene Description [ 26 ] .
Octopus [ 85 ] utilizes images as key training inputs for action planning.
Steve-Eye [ 91 ] combines a visual encoder with a pre-trained LM for effective multi-modal interactions in open-world scenarios.
Palm-e [ 16 ] , while not directly game-focused, also showcases the significance of visual inputs, using multi-modal sequences for tasks ranging from robotic manipulation to visual question answering.
Regarding this visual observation space, there are three distinct approaches for perception worth highlighting: i) Utilization of internal data representations via game-specific APIs to feed game-related information into LMAs (e.g., [ 64 ] ); ii) Model pre-training, based on the acquisition of image and action data (e.g., VPT [ 5 ] ); and iii) Direct use of pure image information for processing and interpretation (e.g., Cradle [ 62 ] ).

 
 
 
 

## 3 Inference

 
 Owing to its potential in supporting autonomy, reactivity, initiative, and sociability functionality [ 81 ] , LMs are well positioned as core components of an intelligent agent’s cognitive framework.

 
 
 Distinct requirements present themselves in different stages of gameplay though.

 
 
 In the initial phase of a game, an agent is required to absorb essential common sense and game-specific background knowledge (either through pre-training or on the fly perception).
In the course of the game, the agent’s role extends to synthesizing past game events, managing knowledge storage and retrieval (§ 3.1 ), and undertaking core cognitive functions such as information learning (§ 3.2 ), reasoning (§ 3.3 ), decision-making (§ 3.4 ), and reflection (§ 3.5 ). Moreover, the agent continually updates or improves its knowledge base for future endeavors.

 
 

### 3.1 Memory

 
 In order to properly represent learned knowledge or past event and use this information in inference, an agent need to effectively manipulate such memories .
How to devise a high-quality memory mechanism that enables efficient retrieval and storage of memory by agents, while both making full use of backing LMs and respecting their constraints, remains a not-settled question.

 
 

#### 3.1.1 Common Sense.

 
 Commonsense knowledge, typically acquired early in life and often unstated in a given context, is crucial for reasoning and avoiding misinterpretations.
LMs, pre-trained on diverse internet corpora, exhibit advanced language comprehension and reasoning skills, using this implicit knowledge to adapt to new challenges.
However, this knowledge might not always align with specific game scenarios.
So current research aims to enhance LMs with this essential commonsense understanding in the following ways.
Embedding common sense using a structured format of goals and instructions is a key feature in [ 94 ; 90 ] .
A Standardized Operating Procedure (SOP) is utilized to integrate common sense into specific roles and tasks, showcasing an innovative approach to contextual understanding [ 24 ] .
Design thinking [ 14 ] and interpretation of robot action language instructions [ 82 ] expand the common sense of LMAs.
Retrieval enhancement method [ 65 ] helps LMAs utilize external knowledge to break cognitive limitations.
These methods show that, with refined settings, LMs can grasp and apply common sense knowledge as intuitively as human understanding.

 
 
 

#### 3.1.2 Game Background Knowledge.

 
 Game Background Knowledge in LMs refers to the understanding of various aspects of gaming environments.
However integrating this knowledge with actual gameplay remains challenging [ 63 ; 40 ] .
Current techniques enhance multi-agent systems with this knowledge by incorporating basic game rules, fundamental processes, reasoning skills, and one-shot demonstrations into prompts, ensuring a more comprehensive understanding of the game context.
This includes:
A benchmark called GameBugDescriptions [ 58 ] prompts LMs using zero-shot for background knowledge about a specific game by inputting game names.
The practice of inputting detailed character profiles into LMs to aid in simulating actions and dialogues is exemplified in [ 32 ; 48 ; 56 ; 69 ] .
Knowledge graphs are used to align player input with entities and relationships to provide game background knowledge [ 4 ] .
Furthermore, the incorporation of game-specific knowledge, such as poker strategies, enhances strategic decision-making capabilities [ 23 ] , indicating a focused approach towards improving models’ understanding and functionality within specific contexts.
Integrating game background knowledge from datasets, such as Disco Elysium which includes dialogue and game state variables, enables LMs to understand and respond within the narrative context [ 2 ] .
Incorporating narrative elements and structured game mechanics significantly improves LMs’ contextual understanding [ 3 ; 93 ] .
The ability of LMs to generate social behaviors based on community descriptions is illustrated by [ 47 ] .
Using detailed scenarios from simulated environments and Role-Playing Game (RPG) logs significantly enhances language models’ understanding of game dynamics [ 35 ; 36 ] .
Additionally, building an external knowledge base with documentation from the Minecraft Wiki and item crafting/smelting recipes creates an exhaustive knowledge source about the Minecraft world [ 94 ] .
In summary, Game Background Knowledge in LMs encapsulates a broad spectrum of gaming-related information, ranging from basic rules and mechanics to complex narratives and role-specific knowledge, enabling these models to interact and perform more effectively in gaming environments.

 
 
 

#### 3.1.3 Retrieval.

 
 Memory Retrieval in LMAs is a multifaceted process that enhances inference and actions by recalling learned information.
In categorizing the memory retrieval methods used by LMAs in gaming, we can distinguish between structured memory systems, dynamic and adaptive retrieval processes, and advanced memory retrieval techniques. Structured memory systems, such as the skill library [ 64 ] , shared message pools [ 24 ] , hierarchical structure [ 39 ] , and structured communication [ 51 ] , focus on efficiently organizing and indexing information for quick retrieval.
These systems allow agents to access previously acquired skills and knowledge, facilitating effective memory management.
The "Summarize-and-Forget" memory mechanism efficiently prioritizes key memory items by summarizing important information and discarding the rest, reducing computational costs [ 30 ] .
The dynamic and adaptive retrieval process in [ 48 ] focuses on combining relevance, recency, and importance for memory retrieval, while strategic collection in [ 83 ] emphasizes the relevance and freshness of historical game information. This approach is enhanced with continuous feedback mechanisms for plan adjustments [ 26 ; 70 ] .
These methods demonstrate the ability of agents to dynamically adapt their memory retrieval to the context and ongoing inputs.
Advanced memory retrieval techniques, such as the long-short term memory [ 92 ] and subtask re-labeling [ 19 ] , generating concise state descriptions [ 45 ] , compressing historical text sequences [ 50 ] , and the specialized memory retrieval modules for storing and accessing historical records [ 75 ] , highlight the sophisticated approaches employed to manage memory in complex gaming scenarios.
The use of a video-language model for task-relevant information encoding and retrieval further underscores the advanced capabilities of LMAs [ 18 ] .
Collectively, these diverse methods illustrate the range and sophistication of memory retrieval strategies in LM-based game agents, showcasing their ability to efficiently store, access, and utilize information for effective and informed gameplay.

 
 
 
 

### 3.2 Learning

 
 Learning in the context of LMs involves interpreting and integrating information from various sources, including training data, user interactions, and environmental feedback.
The learning process can be explicit, such as through fine-tuning on specific datasets or task-focused training [ 19 ] , or implicit, as seen in models leveraging pre-existing knowledge to adapt to new scenarios [ 26 ; 83 ] .
Iterative and Feedback-Based Learning [ 64 ; 51 ; 54 ] merges iterative prompting, role specialization, feedback mechanisms, and behavior cloning, enhancing their capabilities through continuous feedback and adaptation.
Reinforcement and Offline Learning encompasses methods like RL with environmental feedback, offline learning, and language-instruction-conditioned RL, focusing on training LMs to make informed decisions based on environmental interactions and predefined instructions [ 85 ; 90 ; 31 ] .
Experience-Based and Memory Utilization Learning [ 83 ; 40 ] focuses on leveraging past experiences, memory retrieval, and experience pools to inform current and future actions.
Collaborative and Adaptive Learning [ 14 ; 10 ] emphasizes learning through multi-agent collaboration, adaptive strategies, and interactive learning processes, underlining the importance of collaborative problem-solving and adapting to new information or tasks.
Simulated and Contextual Learning [ 47 ; 76 ] pertains to learning within simulated environments or through context-based training, with models adapting to and learning from virtual environments using contextually rich scenarios.
In summary, the diverse approaches to learning underscore a multi-faceted strategy, including iterative feedback, reinforcement techniques, experience-based strategies, collaborative processes simulated contexts, and so on.
These learning methodologies are increasingly tailored to leverage complex interactions and data integrations for enhanced cognitive and perceptual abilities in varied environments.

 
 
 

### 3.3 Reasoning

 
 Reasoning involves abstracting the foundational knowledge acquired from the learning phase into more advanced forms of understanding through deductive, inductive, or abductive reasoning. This process encompasses the ability of LMs and AI agents to process information, draw conclusions, and make predictions; which can be involved in synthesizing data, understanding contexts, making inferences, and applying logical thinking to navigate complex situations or tasks.
A strategic reasoning approach, engaging in abstract thinking and strategic planning, synthesizes complex information required for decision-making and extends reasoning beyond direct data [ 48 ; 66 ] .
Contextual and adaptive reasoning methods allow models to adjust their reasoning based on feedback, environmental factors, and changing scenarios, showcasing their adaptability through the process of model interpretation and adaptation to specific environments [ 64 ; 26 ; 62 ] .
The utilization of multi-agent collaboration and interactive reasoning, where reasoning is a collective process involving various agents or factors working together to obtain a unified decision or solution [ 24 ; 51 ] .
Task-oriented reasoning that focuses on specific difficult problems and demonstrates directly achieving established goals [ 90 ; 94 ] .
The exploration of reasoning within LMAs reflects a sophisticated blend of strategies that encompasses strategic, contextual, collaborative, and task-oriented approaches. These illustrate the capacity of LMAs to not only interpret and engage with complex data but also to dynamically adapt and respond to an ever-changing array of challenges and environments.

 
 
 

### 3.4 Decision-making

 
 In the expansive and rich worlds of digital games, especially within the realms of open-world and immersive environments, players and AI agents are presented with complex decision and action spaces. Such spaces encapsulate the vast array of possible actions, interactions, and decisions available, characterized by an extensive variety of roles, item usage, and skill combinations.
Moreover, multi-agent planning can add another layer of complexity, necessitating coordination and collaboration among multiple agents in environments with incomplete information, like in strategy games. Therefore, adaptation is key for agents to effectively interact in these environments, requiring a blend of strategic contextual problem-solving reasoning skills to effectively respond to dynamic and complex game worlds.
Such environments demand not only fast, reactive decision-making but also the capability to engage in multi-hop inference and long-term planning.

 
 

#### 3.4.1 Multi-Hop Inference.

 
 Multi-hop inference refers to the cognitive process of considering multiple layers of information and dependencies before making a decision. This demands a high level of cognitive sophistication, where the agent must assess, predict, and integrate multiple strands of information to formulate strategies that are both effective and adaptable. For instance, in a strategy game, an agent must consider the current state of play, anticipate opponent moves, evaluate the potential outcomes of various actions, and then decide on the most advantageous course of action.
Although LMs like GPT-4 have foundational decision-making abilities, they struggle with coherent multi-hop logical reasoning such as Minesweeper [ 34 ] .
The involvement of multiple agents working together enhances multi-step decision-making through knowledge sharing [ 24 ; 51 ; 12 ] .
Multi-level decision-making in a hierarchical structure for detailed and strategic choices is adopted by [ 86 ; 88 ; 8 ] .
Automated process and SOP management [ 92 ; 70 ] use standardized procedures to make consistent decisions.
Although these strategies underscore the ongoing efforts to enhance LMAs’ multi-hop inference capabilities, the transition to coherent multi-hop inference in complex scenarios remains a work in progress.

 
 
 

#### 3.4.2 Long-Term Planning.

 
 Long-term planning is the strategic process of setting goals and determining actions to achieve these goals over an extended period, involving multiple steps or stages.
In gaming, this could translate to developing a strategy, requiring the player or LMA to make decisions that not only have immediate benefits but also contribute to their overarching objectives.
In the realm of LMAs tackling long-term planning challenges in gaming contexts, a broad categorization reveals several key approaches.
Chain-of-Thought (COT) [ 73 ] planning generates intermediate steps for a final decision, mimicking human reasoning where each step builds on the previous for a conclusive outcome [ 74 ] .
Complex tasks are decomposed into manageable sub-tasks, emphasizing step-by-step decision-making [ 14 ; 29 ] .
Structured planning and iterative processes [ 24 ; 51 ; 12 ] emphasize the importance of organized workflows and the iterative refinement of strategies.
Skill pre-training and adaptive learning [ 90 ; 87 ] underscores the necessity for agents to be equipped with a diverse set of pre-trained skills that enable them to adapt to new scenarios efficiently.
Finally, these approaches to long-term planning in the context of LMAs within gaming illustrate a strategic convergence of methodologies aimed at enhancing decision-making and problem-solving capabilities.
Through CoT planning, structured and iterative processes, as well as skill pre-training and adaptive learning, LMAs are being progressively refined to tackle complex, long-term planning challenges.

 
 
 
 

### 3.5 Reflection.

 
 Reflection refers to the capacity of LMAs to evaluate, assess, and potentially adjust their own processes, decisions, or outputs as tasks progress.
It implies a level of meta-cognition where the system considers its actions, their outcomes, and the context in which they occur.
Reflection can lead to improvements in performance, the refinement of strategies, or the reassessment of goals.
Feedback-Based Self-Improvement [ 64 ; 94 ; 12 ; 24 ; 31 ] includes processes that utilize external feedback, such as environmental responses or code execution outcomes, for self-verification, analysis, and iterative enhancement.
Iterative Optimization of Plans and Actions [ 26 ; 87 ; 8 ] focuses on methods that continuously refine plans or actions based on feedback and new information to better achieve or adapt to goals.
Collaboration and Multi-Agent Interaction [ 51 ; 66 ] emphasizes collaborative reassessment and adjustment of decisions in multi-agent systems.
Debate and Theory of Mind-Based Reflection [ 17 ; 22 ] focus on models that employ debate or exchange formats as reflective processes.
Thus, reflection embodies a critical aspect of LMAs, enabling them to evaluate and refine their processes through feedback-based self-improvement, iterative optimization, collaborative interactions, and debate-driven reflection. This capacity for self-assessment and adjustment is instrumental in enhancing decision-making, strategy development, and goal reassessment, ultimately leading to more sophisticated, self-aware, and adaptable solutions.

 
 
 
 

## 4 Action

 
 In this section, we will explore how LMs exhibit human-like action in-game environments, including specific behavior execution (§ 4.1 ), communication with humans or other agents (§ 4.2 ), and how to ensure that these actions are consistent (§ 4.3 ).
These agents utilize generative programming techniques, interactive feedback with the environment, and complex conversational exchanges with other agents or human players to perform tasks and solve in-game challenges.

 
 
 The action spaces within which LMAs operate in-game can be broadly categorized into three distinct types, each with its unique challenges and opportunities for interaction and control. These categories range from purely linguistic engagements to direct manipulation of game controls, not only defining the scope of actions available to LMAs but also shaping the strategies and technologies employed to navigate them. i) Text-Based Interaction. 
The first category encompasses pure linguistic interaction, which primarily focuses on linguistic communication and interaction among players. These games, such as Werewolf (e.g., [ 83 ] ) and Avalon (e.g., [ 66 ] ), revolve around dialogue, decision-making, and the interpretation of textual information. In these environments, LMAs are required to understand and generate natural language, engaging with players and the game’s narrative through text. This demands a deep understanding of language nuances, player intentions, and the ability to craft responses that can influence the game’s outcome.
 ii) API or Predefined Actions. 
The second category involves manipulating the game environment through APIs or pre-defined actions, offering a more structured approach to game mechanics.
Examples include the use of the Mineflayer JavaScript APIs in Voyager for action controls [ 64 ] and GITM opting for structured actions implemented via hand-written scripts [ 94 ] . This method requires understanding of the game’s mechanics and the ability to strategically select and sequence actions to achieve desired outcomes, but benefits from the extra semantics and control provided by game-specific APIs.
 iii) Direct Control via IO Operations. 
The third category represents the most immersive form of interaction: direct control only over input devices, like mouse and keyboard. VPT [ 5 ] and Cradle [ 62 ] operate using the same IO devices available to users, e.g., mouse and keyboard, and their input space is discussed in visual observations (§ 2.2 ).
This approach simulates the human gaming experience most closely, with LMAs performing actions at a higher abstraction level, navigating menus, and manipulating items just as a human player would.
This represents a more general form of interaction, necessitating a significant integration of cognitive processing and motor skills.
Each of these categories showcases the versatility and potential of LMAs in gaming, from the purely cognitive and linguistic challenges of text-based games to the physical and tactical demands of direct control.

 
 

### 4.1 Behavior

 
 LMAs perform specific actions through generative programming techniques.
Iterative prompting that incorporates environmental feedback, execution errors, and self-verification to continuously refine generated programs, ensuring their effectiveness and relevance to tasks at hand [ 64 ] .
Programs are generated by interpreting existing instructions or employing role-specific prompt engineering [ 90 ; 14 ] , to enhance the specificity of tasks addressable by LMAs.
Generating programs (task plans) by interpreting natural language instructions within the context of a 3D Scene Graph, which maps instructions to actionable steps for robot execution [ 53 ; 80 ] .
By continuously refining their outputs through feedback loops and leveraging detailed environmental data, LMAs demonstrate an evolving capacity for precise and contextually relevant action planning.

 
 
 

### 4.2 Dialogue

 
 LMAs interact with humans or other agents by taking into account the current environment and past experiences.
To categorize the described interaction methods, we can distinguish them based on their primary focus: agent-agent interactions, and human-agent interactions.
Some methods are versatile, covering both categories.
The collaboration framework of LMAs is divided into two methods: horizontal communication (mutual equality) and vertical communication (there is a superior-subordinate relationship between agents or between humans and agents) [ 12 ] .
Agent-agent interactions include collaborative frameworks [ 10 ] , conversation-driven control in multi-agent systems [ 77 ; 92 ; 20 ] , and game-based strategic behaviors [ 83 ; 38 ; 1 ] .
On the other hand, human-agent interactions are characterized by instruction-following and collaborative task solving [ 32 ; 48 ; 51 ] , gameplay enhancement and creative assistance [ 3 ; 93 ; 14 ] , natural and flexible communication [ 85 ] .
Additionally, Lyfe Agents [ 30 ] exemplify autonomy within human-agent and agent-agent interactions by navigating to various locations, forming groups, and engaging in proximity-based conversations, which includes autonomously choosing to ignore or leave conversations, showcasing a practical application of agent autonomy and social behaviors.
Additionally, LMAs exhibit human-like social behaviors in collaboration, such as Teamwork, Leadership, Persuasion, Deception, and Confrontation [ 31 ] .
This multifaceted approach facilitates a spectrum of interactions, from cooperative frameworks to strategic gameplay and creative task-solving, underpinning the importance of versatile, strategic, and socially aware communication methods in the advancement of LMAs.

 
 
 

### 4.3 Consistency

 
 Action consistency ensures that agent actions remain coherent and aligned with their objectives.
This consistency is maintained through consideration of past experiences and current context, utilization of feedback mechanisms, and modification of actions and plans when necessary.
Specific strategies for ensuring consistent actions include look-ahead checking mechanisms to assess action feasibility [ 21 ] , leveraging visual feedback mechanisms for task-specific responses [ 74 ] , using RL to train policies that select the most appropriate actions [ 84 ] , and employing the Self-Consistency (SC) [ 67 ] such as PokéLLMon [ 25 ] which generates multiple action predictions and selects the most common one for consistent decision-making.
Structural approaches include using DAG and game-related structures for logical action progression [ 78 ] , employing skill libraries for consistent behavior execution [ 64 ] , and defining structured actions with clear semantics [ 94 ] .
Furthermore, assigning specific roles to agents and following structured workflows helps maintain consistency by aligning outputs with defined roles and tasks [ 24 ] , while SOP offers fine-grained control over behavior [ 92 ] .
Feedback and adaptation mechanisms play a crucial role, with techniques ranging from revising plans based on environmental feedback [ 19 ] to employing unified interfaces for automatic responses [ 77 ] .
Neuro-Symbolic Approach (e.g. COTTAGE [ 15 ] ) introduced addresses the consistency issue in text-based adventure games generated by LMs. This approach combines reasoning and code generation for dynamic game state tracking and improved player interaction, ensuring narrative coherence and continuity.
Through these mechanisms, large model agents can not only perform complex and diverse tasks in the game, but also engage in highly interactive dialogue exchanges with human players or other agents, while ensuring the consistency and adaptability of their actions.

 
 
 Figure 3: Overview of Representative Works on Game-Playing Agents and Large Models. 
 
 
 
 

## 5 Challenges Throughout the Playing Cycle

 
 In exploring the gameplay of LMAs, several key challenges emerge, including resolving hallucinations (§ 5.1 ), error correction (§ 5.2 ), generalizing to unseen tasks (§ 5.3 ), and enhancing interpretability (§ 5.4 ). While existing research proposes some solutions, these challenges remain significant obstacles for LMAs in gaming contexts.

 
 

### 5.1 Hallucination

 
 LMs can produce outputs that diverge from original or accurate information, a situation commonly described as hallucinations.
Addressing hallucinations is complex, with LMAs sometimes sending messages with grounding errors, contradictions, or strategic subpar content [ 41 ] .
To classify the various methods employed for solving hallucinations in LMAs, we can look at strategies ranging from structured reasoning and contextual awareness to interactive and collaborative methods, as well as specific prompt or feedback mechanisms.
Studies introduce critic agents, SOP, or role-specific prompts to reduce logical hallucinations, as shown in MetaGPT and JARVIS-1 [ 24 ; 71 ] .
Structured reasoning frameworks like DAG used in [ 78 ] provide a solid foundation for maintaining coherence.
A multi-faceted approach [ 9 ] is employed that encompasses dual collaboration, code decoupling, the development of in-house lexicons, and direct interaction with users for plan rectification.
Additionally, breaking down tasks into smaller components and employing cross-examination in [ 51 ] , or using role-specific cues and character context as seen in [ 83 ] , effectively mitigate hallucinations.
The interactive and collaborative aspect includes allowing for human editing of AI-generated content [ 93 ] , and utilizing structured multi-agent collaboration and feedback mechanisms [ 12 ] .
PokéLLMon [ 25 ] addresses the hallucination problem by integrating In-context Reinforcement Learning (ICRL) and Knowledge-Augmented Generation (KAG) to improve decision-making accuracy and leverage external knowledge for informed actions.
Visual fidelity to outputs from Visual Foundation Models (VFMs) is another approach [ 74 ] .
The incorporation of external knowledge sources in [ 65 ] further enhances the accuracy of LMAs-generated content.
Specific prompt or feedback mechanisms like reducing contradictory messages [ 41 ] , prompt reconstruction [ 14 ] , confidence scoring [ 69 ] , and regular expression matching in MindAgent [ 21 ] contribute to reducing inaccuracies.
Debates are used to refine responses [ 17 ] , and task-specific fine-tuning is employed in [ 29 ] .
Finally, TaPA [ 80 ] ensures that generated action plans only involve objects present in the scene to maintain accuracy.
While diverse methods like structured reasoning, interactive editing, and specific mechanisms play a crucial role in mitigating hallucinations in AI systems and enhancing the coherence and accuracy of outputs, they do not entirely eliminate the issue of hallucinations.

 
 
 

### 5.2 Error Correction

 
 In terms of error correction, various studies employ iterative prompting mechanisms, executable feedback mechanisms, or chat chains among models to identify and correct errors.
The use of critic agents for selection and feedback [ 32 ] , iterative prompting with environment feedback [ 64 ] , and executable feedback for debugging [ 24 ] highlights the importance of iterative learning and feedback mechanisms in refining actions and decisions.
Iterative re-planning based on real-time feedback is employed in [ 53 ; 71 ] while focusing on learning from mistakes to improve game strategies is adapted by [ 31 ] .
MindAgent [ 21 ] through an Action Validation mechanism that employs a look-ahead checking system to parse and assess the feasibility of proposed actions, returning an error message if an action is found to be inexecutable.
Auto MC-Reward [ 33 ] evaluates and corrects reward functions for accuracy, consistency, and relevance, including managing runtime errors with iterative feedback up to three times.
Peer review and system testing [ 51 ] , evaluation stage feedback [ 12 ] , and the use of an additional model, MarioBert [ 57 ] , represent typical approaches to error resolution.
Structured systems and processes, such as the historical experiences and reflections in [ 83 ] , debate structures and evidence pools in [ 65 ] , and the debate process for refining responses in [ 17 ] , contribute to minimizing errors.
Specific strategy implementations like the trial-and-error approach in [ 85 ] and the closed-loop approach with visual feedback in [ 29 ] are vital for correcting errors and enhancing accuracy.
While diverse methods such as iterative feedback, specialized tools, structured processes, and strategic implementations are crucial for effective error correction in LMAs and improve agent adaptability, there is still a need to enhance the efficiency and accuracy of these processes.

 
 
 

### 5.3 Generalization

 
 The ability to generalize involves agents applying the knowledge, skills, and abilities gained in one context to achieve goals and perform tasks in new, previously unencountered environments, allowing for ongoing adaptation and development.
The method of COT reasoning guided by context [ 78 ] , along with the GITM’s utilization of long-term planning and external knowledge in Minecraft [ 94 ] , and the emphasis on information extraction and collaborative planning in multi-agent settings [ 89 ] , highlight contributions to generalizing to unseen tasks in LMAs.
VOYAGER [ 64 ] and the DEPS [ 70 ] showcase the ability to handle new tasks without prior specific training, demonstrating zero-shot generalization, while RoleLLM exhibits robust generalization to unseen roles with minimal input [ 69 ] .
ChatDev’s discussion of adaptive software development using the chat chain framework [ 51 ] and AlphaBlock’s structured framework for adaptability in creating subtask plans for various robot operation scenarios [ 29 ] is highlighted.
Pre-training based on language instructions [ 90 ] demonstrates effective fine-tuning and the ability to transfer skills to new environments, outperforming other methods in learning tasks in new environments.
Cradle [ 62 ] and SIMA [ 52 ] take different approaches to extend LMA generalization to easily adapt to new environments. Both use a common human-like interface to interact with the game environment, where input consists of image observations and language instructions, and output takes the form of keyboard and mouse operations. While SIMA focuses on pre-training a new model, Cradle focuses on leveraging LMs capabilities into a cognitive agent framework.
These methods showcase the capabilities of LMAs in adapting to unfamiliar tasks while acknowledging that generalization capabilities in more complex or variable environments continue to present significant challenges.

 
 
 

### 5.4 Interpretability

 
 Regarding interpretability, some research has made strides in enhancing the explainability of agent actions through COT processes, modular decision-making processes, or clear conversational programming paradigms. However, the logic behind agents’ decisions and behaviors remains challenging to fully comprehend, especially in complex gaming environments.
For reasoning and decision-making process transparency, SPRING [ 78 ] and LLaMA Rider [ 19 ] employ COT reasoning to make actions understandable, while Generative Agents [ 48 ] bases explainability on a combination of past experiences and reflections, and AlphaBlock [ 29 ] derives explainability from logical sub-task plans based on instructions and observations.
For structured and modular approaches, Voyager [ 64 ] utilizes a skill library and iterative prompting for transparent decision-making, GITM [ 94 ] focuses on structured actions with clear semantics for explainability, and a modular approach in planning and decision-making enhances understandability [ 89 ] .
Structured communication and role-based task management MetaGPT [ 24 ] and Agentverse [ 12 ] further aid in making agent actions interpretable.
Action explanation [ 51 ] includes a chat chain offering a transparent view of the software development process, enhancing explainability through structured conversation programming [ 77 ] , and integrating cognitive processes for explainability in deceptive environments [ 66 ] .
Additionally, DEPS [ 70 ] uses an LLM-based explainer for plan failures, Grad-CAM is employed for visual attention interpretability [ 54 ] , and the decision-making process is made more transparent through detailed prompts guiding the use of VFMs [ 74 ] .
The clear role definition in debates [ 65 ] , RL and diverse action candidates [ 84 ] , imitation learning and expert behavior mimicking [ 27 ] , and behavior cloning and networks for explicit action choices [ 8 ] , all contribute to making agent actions more explainable.
These diverse methods make the decision-making of LMAs more understandable and traceable.

 
 
 
 

## 6 Conclusion and Future Direction

 
 In this paper, we survey the literature on combining agents and LMs towards playing complex digital games ( Figure 3 ), which to our knowledge is the first review to thoroughly investigate game-playing by LMAs.
We provide a detailed overview of the various challenges LMAs face in open gaming worlds.
By emphasizing the typical agent workflow - perception, reasoning, and behavior - we can compare and contrast how LMs can be applied to different problems at different stages and elaborate on specific solutions of individual LM-empowered game agent approaches.

 
 
 Besides providing an overview of current approaches and common challenges, based on the executed literature review this work also identifies future research directions for LMAs and digital games.

 
 
 Multi-modal Perception: 
Despite advancements, LMAs like VOYAGER still lack visual perception due to the unavailability of models until recently, suggesting a significant area for improvement [ 64 ] .
Furthermore, sound effects in games, are crucial for perceiving context, understanding tasks, and receiving feedback—key components for the successful completion of game objectives. Despite their significance, there is a scarcity of literature exploring how auditory information can be leveraged to enhance game agents’ performance.
Enhancing multi-modal capabilities, including visual and auditory perceptions, could lead to more sophisticated task handling and immersive game playing experiences.

 
 
 Authenticity in Gaming Experience: 
The evolution towards authenticity in gaming scenarios also merits further exploration.
Studies have shown a preference for human-written content over LLM-generated dialogue and ideas, indicating the need for better grounding of LLM generations in the game’s narrative and state [ 48 ; 3 ; 93 ; 68 ] . Enhancements in narrative generation and character simulation, as explored in RoleLLM, could lead to more realistic and immersive interactions [ 69 ] . Furthermore, the behaviour itself of LMAs is oftentimes artificial and making it more fluid/realistic opens new possibilities.

 
 
 Use of External Tools: 
LMA’s inability to master leveraging external tools to enhance gameplay represents another significant gap in achieving AGI.
Current LMs struggle with selecting and employing tools efficiently, resulting in limited access to real-time information.
Enabling the ability to dynamically access and interpret online game guides and generalize external game-specific knowledge to new games will represent significant progress, and this will require enhanced model capabilities to interact with external APIs and documentation.
Works such as Toolformer [ 55 ] and Gorilla [ 49 ] address some of these challenges by focusing on re-representing API call primitives and fine-tuning models to better understand and use documents. However, a full implementation of LM game agents proficient in the use of external tools remains open.

 
 
 Real-Time Gaming: 
Considering the inherent reasoning process and computational demands of LMs, mastering real-time, high-paced gaming presents a formidable challenge. The demanding nature of gaming environments necessitates adherence to rigorous real-time performance and inference speed criteria. While LMs offer considerable advantages, they are not a substitute for RL in contexts requiring immediate, time-critical decision-making [ 28 ; 72 ; 7 ] . Looking ahead to the integration of LMAs in live gaming sessions alongside human participants, these agents must achieve latencies aligned with the required frame rates to guarantee real-time interaction. Therefore, the pursuit of greatly improving the efficiency and reaction times of LMAs in dynamic and swiftly evolving gaming landscapes is essential for forthcoming progress.

 
 
 In conclusion, while LMAs have made remarkable strides in gaming, their full potential is yet to be realized. Future work should focus on improving multi-modal perception, achieving greater authenticity in gaming experiences, effectively integrating external tools, and excelling in real-time gaming environments; among other unexplored fronts (e.g., LMAs as automatic game testers). These advancements will not only enhance the capabilities of LMAs, lead to more engaging and realistic gaming experiences but also lead to significant impact on real-world scenarios.
Moreover LMs can also be applied in diverse other digital game-related scenarios, e.g., level and story generation, or game balancing and mechanics tuning; which were outside the scope of this survey and deserve their own in-depth research.

 
 
 

## References

 
 
 Abdelnabi and et al [2023] 
 
Sahar Abdelnabi and Amr Gomaa et al.

 
 LLM-Deliberation: Evaluating LLMs with Interactive Multi-Agent Negotiation Games.

 
 arXiv:2309.17234 , 2023.

 

 
 Akoury and et al. [2023] 
 
Nader Akoury and Ronan Salz et al.

 
 Towards grounded dialogue generation in video game environments.

 
 2023.

 

 
 [3] 
 
Nader Akoury, Qian Yang, and Mohit Iyyer.

 
 A framework for exploring player perceptions of llm-generated dialogue in commercial video games.

 
 In Findings of EMNLP 2023 .

 

 
 Ashby and et al [2023] 
 
Trevor Ashby and Braden K Webb et al.

 
 Personalized quest and dialogue generation in role-playing games: A knowledge graph-and language model-based approach.

 
 In CHI , 2023.

 

 
 Baker and et al. [2022] 
 
Bowen Baker and Ilge Akkaya et al.

 
 Video pretraining (vpt): Learning to act by watching unlabeled online videos.

 
 NeurIPS , 2022.

 

 
 Bellemare and et al. [2013] 
 
Marc G Bellemare and Yavar Naddaf et al.

 
 The arcade learning environment: An evaluation platform for general agents.

 
 Journal of Artificial Intelligence Research , 2013.

 

 
 Berner and et al. [2019] 
 
Christopher Berner and Greg Brockman et al.

 
 Dota 2 with large scale deep reinforcement learning.

 
 arXiv:1912.06680 , 2019.

 

 
 Cai and et al. [2023] 
 
Shaofei Cai and Bowei Zhang et al.

 
 Groot: Learning to follow instructions by watching gameplay videos.

 
 arXiv:2310.08235 , 2023.

 

 
 Chen and et al [2023a] 
 
Dake Chen and Hanbin Wang et al.

 
 GameGPT: Multi-agent collaborative framework for game development.

 
 arXiv:2310.08067 , 2023.

 

 
 Chen and et al. [2023b] 
 
Guangyao Chen and Siwei Dong et al.

 
 AutoAgents: A framework for Automatic Agent Generation.

 
 arXiv:2309.17288 , 2023.

 

 
 Chen and et al. [2023c] 
 
Liang Chen and Yichi Zhang et al.

 
 Towards end-to-end embodied decision making with multi-modal large language model.

 
 In NeurIPS 2023 Workshop , 2023.

 

 
 Chen and et al. [2023d] 
 
Weize Chen and Yusheng Su et al.

 
 Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents.

 
 arXiv:2308.10848 , 2023.

 

 
 Cisek and Kalaska [2010] 
 
Paul Cisek and John F Kalaska.

 
 Neural mechanisms for interacting with a world full of action choices.

 
 Annual review of neuroscience , 2010.

 

 
 Ding and et al. [2023] 
 
Shiying Ding and Xinyi Chen et al.

 
 DesignGPT: Multi-agent collaboration in design.

 
 arXiv:2311.11591 , 2023.

 

 
 Dong [2023] 
 
Yijiang River Dong.

 
 COTTAGE: Coherent Text Adventure Games Generation .

 
 PhD thesis, University of Pennsylvania, 2023.

 

 
 Driess and et al. [2023] 
 
Danny Driess and Fei Xia et al.

 
 Palm-e: An embodied multimodal language model.

 
 arXiv:2303.03378 , 2023.

 

 
 Du and et al. [2023] 
 
Yilun Du and Shuang Li et al.

 
 Improving factuality and reasoning in language models through multiagent debate.

 
 arXiv:2305.14325 , 2023.

 

 
 Fan and et al. [2022] 
 
Linxi Fan and Guanzhi Wang et al.

 
 Minedojo: Building open-ended embodied agents with internet-scale knowledge.

 
 NeurIPS , 2022.

 

 
 Feng and et al. [2023] 
 
Yicheng Feng and Yuxuan Wang et al.

 
 Llama rider: Spurring large language models to explore the open world.

 
 arXiv:2310.08922 , 2023.

 

 
 Fu and et al. [2023] 
 
Yao Fu and Hao Peng et al.

 
 Improving language model negotiation with self-play and in-context learning from ai feedback.

 
 arXiv:2305.10142 , 2023.

 

 
 Gong and et al. [2023] 
 
Ran Gong and Qiuyuan Huang et al.

 
 Mindagent: Emergent gaming interaction.

 
 arXiv:2309.09971 , 2023.

 

 
 Guo and et al. [2023] 
 
Jiaxian Guo and Bo Yang et al.

 
 Suspicion-agent: Playing imperfect information games with theory of mind-aware GPT-4.

 
 arXiv:2309.17277 , 2023.

 

 
 Gupta [2023] 
 
Akshat Gupta.

 
 Are ChatGPT and GPT-4 Good Poker Players? – A Pre-Flop Analysis.

 
 arXiv:2308.12466 , 2023.

 

 
 Hong and et al. [2023] 
 
Sirui Hong and Mingchen Zhuge et al.

 
 MetaGPT: Meta programming for a multi-agent collaborative framework.

 
 arXiv:2308.00352 , 2023.

 

 
 Hu et al. [2024] 
 
Sihao Hu, Tiansheng Huang, and Ling Liu.

 
 Pokéllmon: A human-parity agent for pokémon battles with large language models.

 
 arXiv:2402.01118 , 2024.

 

 
 Huang and et al. [2022] 
 
Wenlong Huang and Fei Xia et al.

 
 Inner monologue: Embodied reasoning through planning with language models.

 
 arXiv:2207.05608 , 2022.

 

 
 Huang and et al. [2023] 
 
Qiuyuan Huang and Jae Sung Park et al.

 
 Ark: Augmented reality with knowledge interactive emergent ability.

 
 arXiv:2305.00970 , 2023.

 

 
 Iskander and et al. [2020] 
 
Nancy Iskander and Aurelien Simoni et al.

 
 Reinforcement learning agents for ubisoft’s roller champions.

 
 arXiv:2012.06031 , 2020.

 

 
 Jin and et al. [2023] 
 
Chuhao Jin and Wenhui Tan et al.

 
 Alphablock: Embodied finetuning for vision-language reasoning in robot manipulation.

 
 arXiv:2305.18898 , 2023.

 

 
 Kaiya and et al [2023] 
 
Zhao Kaiya and Michelangelo Naim et al.

 
 Lyfe agents: Generative agents for low-cost real-time social interactions.

 
 arXiv:2310.02172 , 2023.

 

 
 Lan and et al. [2023] 
 
Yihuai Lan and Zhiqiang Hu et al.

 
 Llm-based agent society investigation: Collaboration and confrontation in avalon gameplay.

 
 arXiv:2310.14985 , 2023.

 

 
 Li and et al. [2023a] 
 
Guohao Li and Hasan Abed Al Kader Hammoud et al.

 
 CAMEL: Communicative agents for ’mind’ exploration of large language model society.

 
 In The 37th NeurIPS , 2023.

 

 
 Li and et al. [2023b] 
 
Hao Li and Xue Yang et al.

 
 Auto mc-reward: Automated dense reward design with large language models for minecraft.

 
 arXiv:2312.09238 , 2023.

 

 
 Li et al. [2023a] 
 
Yinghao Li, Haorui Wang, and Chao Zhang.

 
 Assessing logical puzzle solving in large language models: Insights from a minesweeper case study.

 
 arXiv:2311.07387 , 2023.

 

 
 Li et al. [2023b] 
 
Yuan Li, Yixuan Zhang, and Lichao Sun.

 
 Metaagents: Simulating interactions of human behaviors for llm-based task-oriented coordination via collaborative generative agents.

 
 arXiv:2310.06500 , 2023.

 

 
 Liang et al. [2023] 
 
Yuanzhi Liang, Linchao Zhu, and Yi Yang.

 
 Tachikuma: Understading complex interactions with multi-character and novel objects by large language models.

 
 arXiv:2307.12573 , 2023.

 

 
 Lifshitz and et al. [2023] 
 
Shalev Lifshitz and Keiran Paster et al.

 
 Steve-1: A generative model for text-to-behavior in minecraft.

 
 In NeurIPS 2023 Workshop , 2023.

 

 
 Light and et al. [2023] 
 
Jonathan Light and Min Cai et al.

 
 Avalonbench: Evaluating llms playing the game of avalon.

 
 In NeurIPS 2023 Workshop , 2023.

 

 
 Liu and et al [2024] 
 
Jijia Liu and Chao Yu et al.

 
 Llm-powered hierarchical language agent for real-time human-ai coordination.

 
 arXiv:2312.15224 , 2024.

 

 
 Ma and et al. [2023] 
 
Weiyu Ma and Qirui Mi et al.

 
 Large language models play starcraft ii: Benchmarks and a chain of summarization approach.

 
 arXiv:2312.11865 , 2023.

 

 
 Meta’s Fundamental AI Research Diplomacy Team [2022] 
 
Meta’s Fundamental AI Research Diplomacy Team.

 
 Human-level play in the game of Diplomacy by combining language models with strategic reasoning.

 
 Science , 2022.

 

 
 Nareyek and et al. [2003] 
 
Alexander Nareyek and Bjoern Knafla et al.

 
 The 2003 Report of the IGDA’s Artificial Intelligence Interface Standards Committee.

 
 International Game Developers Association (IGDA) Technical Report , 2003.

 

 
 Nareyek and et al. [2004] 
 
Alexander Nareyek and Börje F. Karlsson et al.

 
 The 2004 Report of the IGDA’s Artificial Intelligence Interface Standards Committee.

 
 IGDA Technical Report , 2004.

 

 
 Nareyek and et al. [2005] 
 
Alexander Nareyek and Nathan Combs et al.

 
 The 2005 Report of the IGDA’s Artificial Intelligence Interface Standards Committee.

 
 IGDA Technical Report , 2005.

 

 
 Nottingham and et al [2023] 
 
Kolby Nottingham and Yasaman Razeghi et al.

 
 Selective perception: Optimizing state descriptions with reinforcement learning for language model actors.

 
 arXiv:2307.11922 , 2023.

 

 
 OpenAI [2023] 
 
OpenAI.

 
 ChatGPT can now see, hear, and speak, 2023.

 

 
 Park and et al. [2022] 
 
Joon Sung Park and Lindsay Popowski et al.

 
 Social simulacra: Creating populated prototypes for social computing systems.

 
 In ACM UIST , 2022.

 

 
 Park and et al. [2023] 
 
Joon Sung Park and Joseph O’Brien et al.

 
 Generative agents: Interactive simulacra of human behavior.

 
 In The 36th UIST , 2023.

 

 
 Patil and et al. [2023] 
 
Shishir G Patil and Tianjun Zhang et al.

 
 Gorilla: Large language model connected with massive apis.

 
 arXiv:2305.15334 , 2023.

 

 
 Piterbarg et al. [2023] 
 
Ulyana Piterbarg, Lerrel Pinto, and Rob Fergus.

 
 diff history for neural language agents.

 
 arXiv:2312.07540 , 2023.

 

 
 Qian and et al. [2023] 
 
Chen Qian and Xin Cong et al.

 
 Communicative agents for software development.

 
 arXiv:2307.07924 , 2023.

 

 
 Raad and et al [2024] 
 
Maria Abi Raad and Arun Ahuja et al.

 
 Scaling instructable agents across many simulated worlds.

 
 Technical Report , 2024.

 

 
 Rana and et al. [2023] 
 
Krishan Rana and Jesse Havil et al.

 
 Sayplan: Grounding large language models using 3d scene graphs for scalable task planning.

 
 arXiv:2307.06135 , 2023.

 

 
 Schäfer and et al. [2023] 
 
Lukas Schäfer and Logan Jones et al.

 
 Visual encoders for data-efficient imitation learning in modern video games.

 
 arXiv:2312.02312 , 2023.

 

 
 Schick and et al. [2023] 
 
Timo Schick and Jane Dwivedi-Yu et al.

 
 Toolformer: Language models can teach themselves to use tools.

 
 arXiv:2302.04761 , 2023.

 

 
 Shao and et al. [2023] 
 
Yunfan Shao and Linyang Li et al.

 
 Character-LLM: A trainable agent for role-playing.

 
 In EMNLP , 2023.

 

 
 Sudhakaran and et al. [2023] 
 
Shyam Sudhakaran and Miguel González-Duque et al.

 
 MarioGPT: Open-Ended Text2Level Generation through Large Language Models.

 
 arXiv:2302.05981 , 2023.

 

 
 Taesiri and et al [2022] 
 
Mohammad Reza Taesiri and Finlay Macklon et al.

 
 Large language models are pretty good zero-shot video game bug detectors.

 
 arXiv:2210.02506 , 2022.

 

 
 Taesiri and et al [2023] 
 
Mohammad Reza Taesiri and Tianjun Feng et al.

 
 Glitchbench: Can large multimodal models detect video game glitches?

 
 arXiv:2312.05291 , 2023.

 

 
 Taesiri and et al [2024] 
 
Mohammad Reza Taesiri and Finlay Macklon et al.

 
 Searching bug instances in gameplay video repositories.

 
 IEEE Transactions on Games , 2024.

 

 
 Tan and et al. [2024a] 
 
Weihao Tan and Wentao Zhang et al.

 
 True knowledge comes from practice: Aligning large language models with embodied environments via reinforcement learning.

 
 In ICLR , 2024.

 

 
 Tan and et al. [2024b] 
 
Weihao Tan and Ziluo Ding et al.

 
 Towards General Computer Control: A Multimodal Agent for Red Dead Redemption II as a Case Study.

 
 arXiv:2403.03186 , 2024.

 

 
 Tsai and et al. [2023] 
 
Chen Feng Tsai and Xiaochen Zhou et al.

 
 Can large language models play text games well? Current state-of-the-art and open questions.

 
 arXiv:2304.02868 , 2023.

 

 
 Wang and et al. [2023a] 
 
Guanzhi Wang and Yuqi Xie et al.

 
 Voyager: An open-ended embodied agent with large language models.

 
 arXiv:2305.16291 , 2023.

 

 
 Wang and et al. [2023b] 
 
Haotian Wang and Xiyuan Du et al.

 
 Apollo’s oracle: Retrieval-augmented reasoning in multi-agent debates.

 
 arXiv:2312.04854 , 2023.

 

 
 Wang and et al. [2023c] 
 
Shenzhi Wang and Chang Liu et al.

 
 Avalon’s game of thoughts: Battle against deception through recursive contemplation.

 
 arXiv:2310.01320 , 2023.

 

 
 Wang and et al. [2023d] 
 
Xuezhi Wang and Jason Wei et al.

 
 Self-consistency improves chain of thought reasoning in language models.

 
 arXiv:2203.11171 , 2023.

 

 
 Wang and et al. [2023e] 
 
Yuxin Wang and Jieru Lin et al.

 
 Open-world story generation with structured knowledge enhancement: A comprehensive survey.

 
 Neurocomputing , 2023.

 

 
 Wang and et al. [2023f] 
 
Zekun Moore Wang and Zhongyuan Peng et al.

 
 Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models.

 
 arXiv:2310.00746 , 2023.

 

 
 Wang and et al. [2023g] 
 
Zihao Wang and Shaofei Cai et al.

 
 Describe, explain, plan and select: Interactive planning with llms enables open-world multi-task agents.

 
 In The 37th NeurIPS , 2023.

 

 
 Wang and et al. [2023h] 
 
Zihao Wang and Shaofei Cai et al.

 
 Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models.

 
 arXiv:2311.05997 , 2023.

 

 
 [72] 
 
Hua Wei and Jingxiao Chen et al.

 
 Honor of kings arena: An environment for generalization in competitive reinforcement learning.

 
 The 35th NeurIPS .

 

 
 Wei and et al. [2022] 
 
Jason Wei and Xuezhi Wang et al.

 
 Chain-of-thought prompting elicits reasoning in large language models.

 
 NeurIPS , 35:24824–24837, 2022.

 

 
 Wu and et al. [2023a] 
 
Chenfei Wu and Shengming Yin et al.

 
 Visual ChatGPT: Talking, drawing and editing with visual foundation models.

 
 arXiv:2303.04671 , 2023.

 

 
 Wu and et al. [2023b] 
 
Dekun Wu and Haochen Shi et al.

 
 Deciphering digital detectives: Understanding llm behaviors and capabilities in multi-agent mystery games.

 
 arXiv:2312.00746 , 2023.

 

 
 Wu and et al. [2023c] 
 
Jimmy Wu and Rika Antonova et al.

 
 Tidybot: Personalized robot assistance with large language models.

 
 arXiv:2305.05658 , 2023.

 

 
 Wu and et al. [2023d] 
 
Qingyun Wu and Gagan Bansal et al.

 
 Autogen: Enabling next-gen llm applications via multi-agent conversation framework.

 
 arXiv:2308.08155 , 2023.

 

 
 Wu and et al. [2023e] 
 
Yue Wu and So Yeon Min et al.

 
 SPRING: Studying papers and reasoning to play games.

 
 In The 37th NeurIPS , 2023.

 

 
 Wu and et al. [2023f] 
 
Yue Wu and Xuan Tang et al.

 
 Smartplay: A benchmark for llms as intelligent agents.

 
 2023.

 

 
 Wu and et al. [2023g] 
 
Zhenyu Wu and Ziwei Wang et al.

 
 Embodied task planning with large language models.

 
 arXiv:2307.01848 , 2023.

 

 
 Xi and et al. [2023] 
 
Zhiheng Xi and Wenxiang Chen et al.

 
 The rise and potential of large language model based agents: A survey.

 
 arXiv:2309.07864 , 2023.

 

 
 Xiao and et al. [2023] 
 
Ted Xiao and Harris Chan et al.

 
 Robotic skill acquisition via instruction augmentation with vision-language models.

 
 RSS , 2023.

 

 
 Xu and et al. [2023a] 
 
Yuzhuang Xu and Shuo Wang et al.

 
 Exploring large language models for communication games: An empirical study on werewolf.

 
 arXiv:2309.04658 , 2023.

 

 
 Xu and et al. [2023b] 
 
Zelai Xu and Chao Yu et al.

 
 Language agents with reinforcement learning for strategic play in the werewolf game.

 
 arXiv:2310.18940 , 2023.

 

 
 Yang and et al. [2023] 
 
Jingkang Yang and Yuhao Dong et al.

 
 Octopus: Embodied vision-language programmer from environmental feedback.

 
 arXiv:2310.08588 , 2023.

 

 
 Yuan and et al. [2023] 
 
Haoqi Yuan and Chi Zhang et al.

 
 Skill reinforcement learning and planning for open-world long-horizon tasks.

 
 In NeurIPS 2023 Foundation Models for Decision Making Workshop , 2023.

 

 
 Zhai and et al. [2023] 
 
Shaopeng Zhai and Jie Wang et al.

 
 Building open-ended embodied agent via language-policy bidirectional adaptation.

 
 arXiv:2401.00006 , 2023.

 

 
 Zhang and et al. [a] 
 
Ceyao Zhang and Kaijie Yang et al.

 
 Proagent: Building proactive cooperative ai with large language models.

 
 AAAI 2024 .

 

 
 Zhang and et al. [b] 
 
Hongxin Zhang and Weihua Du et al.

 
 Building cooperative embodied agents modularly with large language models.

 
 In NeurIPS 2023 Foundation Models for Decision Making Workshop .

 

 
 Zhang and et al. [2023] 
 
Jesse Zhang and Karl Pertsch et al.

 
 Sprint: Scalable policy pre-training via language instruction relabeling.

 
 arXiv:2306.11886 , 2023.

 

 
 Zheng and et al. [2023] 
 
Sipeng Zheng and Jiazheng Liu et al.

 
 Steve-eye: Equipping llm-based embodied agents with visual perception in open worlds.

 
 arXiv:2310.13255 , 2023.

 

 
 Zhou and et al. [2023] 
 
Wangchunshu Zhou and Yuchen Eleanor Jiang et al.

 
 Agents: An open-source framework for autonomous language agents.

 
 arXiv:2309.07870 , 2023.

 

 
 Zhu and et al. [2023a] 
 
Andrew Zhu and Lara Martin et al.

 
 Calypso: Llms as dungeon master’s assistants.

 
 In AIIDE 2023 , 2023.

 

 
 Zhu and et al. [2023b] 
 
Xizhou Zhu and Yuntao Chen et al.

 
 Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory.

 
 2023.