A Survey on Large Language Model-Based Game Agents 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2404.02039v5 [cs.AI] 08 Jun 2026 
 
 

# A Survey on Large Language Model-Based Game Agents

 DOI: XXXXXXX.XXXXXXX Journal: CSUR CCS: Computing methodologies Artificial intelligence CCS: Computing methodologies Natural language processing CCS: Computing methodologies Intelligent agents CCS: Software and its engineering Interactive games 
 
 
 Sihao Hu
 
 email: sihaohu@gatech.edu 
 
 Affiliation: Georgia Institute of Technology , USA 
 
 , 
 Tiansheng Huang
 
 email: thuang@gatech.edu 
 
 Affiliation: Georgia Institute of Technology , USA 
 
 , 
 Gaowen Liu
 
 email: gaoliu@cisco.com 
 
 Affiliation: Cisco Research , USA 
 
 , 
 Ramana Rao Kompella
 
 email: rkompell@cisco.com 
 
 Affiliation: Cisco Research , USA 
 
 , 
 Fatih Ilhan
 
 email: filhan@gatech.edu 
 
 Affiliation: Georgia Institute of Technology , USA 
 
 , 
 Selim Furkan Tekin
 
 email: stekin6@gatech.edu 
 
 Affiliation: Georgia Institute of Technology , USA 
 
 , 
 Yichang Xu
 
 email: xuyichang@gatech.edu 
 
 Affiliation: Georgia Institute of Technology , USA 
 
 , 
 Zachary Yahn
 
 email: zachary.yahn@gatech.edu 
 
 Affiliation: Georgia Institute of Technology , USA 
 
 and 
 Ling Liu
 
 email: ling.liu@cc.gatech.edu 
 
 Affiliation: Georgia Institute of Technology , USA 
 
 2026 

 Abstract. 
 
 Game environments provide rich, controllable settings that simulate many aspects of real-world complexity. As such, game agents offer a valuable testbed for exploring capabilities relevant to Artificial General Intelligence ( Yannakakis and Togelius, 2018 ) . Recently, the emergence of Large Language Models (LLMs) provides new opportunities to endow these agents with generalizable reasoning, memory, and adaptability in complex game environments.
This survey offers an up-to-date review of LLM-based game agents (LLMGAs) through a unified reference architecture. At the single-agent level, we synthesize existing studies around three core components: memory, reasoning, and perception–action interfaces, which jointly characterize how language enables agents to perceive, think, and act.
At the multi-agent level, we outline how communication protocols and organizational models support coordination, role differentiation, and large-scale social behaviors.
To contextualize these designs, we introduce a challenge-centered taxonomy linking six major game genres to their dominant agent requirements, from low-latency response in action games to open-ended goal formation in sandbox worlds. We will continuously update the survey to track new developments, and we maintain a continuously updated list of related papers at: https://github.com/git-disl/awesome-LLM-game-agent-papers .

 
 
 

## 1. Introduction

 
 By scaling model capacity and training on massive, diverse text corpora, large language models (LLMs) have demonstrated strong capabilities in language understanding, knowledge generalization, and conversational dialogue ( Ouyang et al., 2022 ; Brown et al., 2020 ; Achiam et al., 2023 ) . Despite these advances, current LLMs are primarily optimized on fixed, static text corpora. Human intelligence, in contrast, develops through continuous sensorimotor engagement with the environment ( Smith and Gasser, 2005 ) ,
for example, by forming perceptual representations from repeated interactions that capture the structure and dynamics of the world ( Barsalou, 1999 ) ,
and by adjusting behavior in response to feedback from action outcomes that gradually improves performance ( Clark, 2013 ) . In general, the literature on embodied cognition emphasizes that human intelligence arises from situated interaction with the environment rather than from disembodied symbol manipulation ( Clark, 1998 ; Varela et al., 2017 ; Smith and Gasser, 2005 ) .

 
 
 Unlike humans, LLM-based agents lack a physical body, making deep participation in real-world interactions difficult and costly. In contrast, game environments provide a natural testbed for realizing the coupling between agent and environment, and offer a richer, more embodied alternative compared to typical settings of current LLM-based agents, such as dialogue, web navigation, or API tool use ( Wang et al., 2024b ) . By granting avatars to agents in the interactive world with perception and action modules, digital games approximate aspects of real-world while remaining safe, controllable, and cost-effective. In addition, they are reproducible and span a wide range of complexity, making them an effective platform for advancing LLMs toward interactive intelligence.

 
 
 Traditional game agents follow a control-based paradigm, where decision-making is coupled through predefined or learned state–action mappings ( Yannakakis and Togelius, 2018 ) . Finite state machines, behavior trees, and reinforcement learning agents ( Iovino et al., 2022 ; Sutton et al., 1998 ) exemplify this design. In contrast, language serves as a unified medium for LLM-based agents to represent goals, contexts, and interactions, enabling explicit reasoning, reflection, and communication beyond traditional systems.

 
 
 Existing surveys ( Wang et al., 2024b ; Gao et al., 2024 ) touch on the topic from different angles yet largely treat games as one of downstream applications alongside dialogue, tool use, or web automation. However, the complexity and openness of game environments distinguish them from narrowly defined tasks. For instance, while a web-based agent may complete a query or transaction through a handful of API calls, a sandbox game enables researchers to cultivate entire agent societies and allows agents to freely explore, interact, and build within physics-driven worlds. These game environments afford a degree of freedom that enables emergent behaviors far beyond constrained, task-oriented interactions. On the other hand, game-focused surveys ( Gallotta et al., 2024 ; Sweetser, 2024 ) emphasize areas such as game development, educational applications, or content generation, leaving the field of LLM-based game agents (LLMGAs) underexplored. As a result, a dedicated survey of LLMGAs as a distinct research area is in strong demand.

 
 
 To bridge this gap, this survey focuses exclusively on LLM-based
game agents. Our contributions include (i) a reference architecture for analyzing LLMGAs at both single-agent and multi-agent levels, and (ii) a challenge-centered game taxonomy linking game genres to agent design requirements. Through these two lenses, we review existing studies and identify open challenges and future directions. The remainder of this paper is organized as follows. Section 2 describes our research methodology. Section 3 provides an overview of the LLMGA framework and game taxonomy. Section 4 , Section 5 , and Section 6 detail the three core single-agent components: memory system, reasoning mechanism, and perception-action interface. Section 7 covers the multi-agent framework. Section 8 applies the challenge-centered taxonomy to analyze design challenges and methods. Section 9 synthesizes the reviewed literature and identifies the open challenges. Section 10 concludes the survey.

 
 
 

## 2. Research Methodology

 

### 2.1. Research Objectives and Questions

 
 This survey is guided by two main objectives. The first objective is to clarify the architectural landscape of LLM-based agents in games and identify common design patterns across existing studies. The second objective is to analyze how contextual characteristics of games, such as game genre, relate to architectural design decisions in LLMGAs. Based on these objectives, this survey addresses the following research questions:

 
 
 (Q1) How are the core architectural components of LLMGA frameworks designed and implemented in existing studies? 

 
 
 (Q2) How do different game genres influence architectural design requirements in LLMGAs? 

 
 
 To answer Q1, we categorize existing LLMGA studies under a unified reference architecture that integrates two complementary perspectives. The first, the LLMGA framework, enables component-level analysis of a single agent. It abstracts common design choices into three modules: a memory system that stores and retrieves past experience, a reasoning mechanism that plans and makes decisions, and a perception action interface that connects the agent to the game environment. The second, the multi-LLMGA framework, examines how populations of agents interact and self-organize. It distinguishes two levels: agent-level communication, which governs how agents exchange messages and align their beliefs, and organization-level structure, which shapes how agents are coordinated and organized collectively.

 
 
 To answer Q2, we introduce a challenge-centered taxonomy that maps six representative game genres ( SteamDB, 2025 ; Lee et al., 2014 ) to the distinct demands they impose on agent design. For example, role-playing games center on the problem of role fidelity, i.e. , how to encode and maintain consistent personas in memory so that dialogue and actions remain aligned with character identity over extended interactions. These genre–challenge mappings offer a structured lens on prior work and practical guidance for developing future LLMGAs. The broader aim of this survey is to position game environments as experimental grounds for examining whether sustained interaction between agents and their environments can foster more general and adaptive forms of intelligence.

 
 
 

### 2.2. Source Retrieval Process

 
 We include papers that employ an LLM or vision language model (VLM) as a central decision-making component and involve interaction with a game or game-like environment. The source retrieval process can be divided into three steps:
In the first step, we conducted keyword searches with the Boolean query (“large language model” OR LLM) AND (game OR gaming) AND (agent) across four sources, including ACM Digital Library, IEEE Xplore, Google Scholar, and arXiv, covering publications from January 1, 2022 to June 1, 2026. In the second step, we used Rayyan 1 1 
 1 
 
 
 
 Rayyan is a web-based tool for collaborative screening in systematic reviews, supporting blinded inclusion/exclusion decisions. https://www.rayyan.ai/ to facilitate collaborative screening by six co-authors. Each co-author independently reviewed titles and abstracts in blind mode and labeled each record as included or excluded. Papers receiving at least two independent include votes were retained. We then conducted full-text screening by evenly assigning the papers to the same six co-authors, who labeled each paper as included or excluded based on its relevance to the survey scope. In the third step, we conducted backward snowballing by reviewing the recent references on LLMGAs. Identified candidates were subjected to the same collaborative full-text screening described in the second step, resulting in a final corpus analyzed in this survey.

 
 
 
 

## 3. Overview

 

### 3.1. LLM-based Game Agent (LLMGA) Framework

 
 Cognitive science views intelligence as an integrated system in which perception–action, memory, and reasoning processes interact to produce adaptive behavior ( Newell, 1994 ; Kotseruba and Tsotsos, 2020 ) . In line with this view, we find that existing studies on LLMGAs primarily introduce techniques that fall into three components: memory, reasoning, and perception–action ( Park et al., 2023 ; Yao et al., 2023 ; Hu et al., 2024 ) . Building on this perspective, we categorize existing LLMGA studies under a unified framework that instantiates these cognitive principles through the three components. Figure 1 (a) illustrates the overall architecture: a central LLM connects the three components in continuous interaction with the game environment. At each step of gameplay, the environment evolves and produces new observations, which the agent perceives, interprets, and acts upon, completing a closed perception-action loop.

 
 
 The perception interface transforms these observations into representations that the LLM can interpret ( Ma et al., 2024 ) . In Section 6 , we discuss how different modalities of observations, including textual, symbolic, and visual inputs, are handled by the agent.

 
 
 The memory system provides a temporal mechanism that links past, present, and future, allowing information to persist across time and guide ongoing decisions. Following classic distinctions in cognitive psychology ( Baddeley and Hitch, 1974 ; Baddeley, 2012 ) , we divide it into working memory and long-term memory. Working memory offers a short-term buffer that supports immediate processing and coordination across steps, with technical considerations centered on extending its capacity and maintaining consistency over time. Long-term memory, by contrast, accumulates knowledge and experience across episodes. In Section 4 , we will focus on how to decide when and what to consolidate from transient experiences into long-term memory, and how stored content can be structured and retrieved.

 
 
 Building on observations and memories, the reasoning mechanism defines how the LLM generates reasoning traces, such as plans, explanations, or self-critiques, that guide action proposals ( Wei et al., 2022 ; Yao et al., 2023 ; Shinn et al., 2023 ) . In cognitive science, reasoning is understood as constructing and operating on internal representations to draw inferences beyond the given information ( Johnson-Laird, 2010 ; Evans, 2008 ) . In Section 5 , we outline two complementary approaches: prompting strategies, which elicit diverse reasoning paths at inference time, ranging from single linear chains to multiple parallel explorations and iterative refinements; Training paradigms, which improve reasoning ability by learning from expert demonstrations and from trial-and-error interaction with the environment.

 
 
 Finally, the action interface functions as the agent’s hand and foot, translating language-based action proposals into concrete interactions with the environment ( Wang et al., 2024a ) . In Section 6 , we discuss how high-level, free-form language decisions are transformed into executable behaviors, including constrained natural language commands, symbolic actions, and sequences of low-level controls. These actions in turn alter the game state, producing new observations and completing the cycle of interaction.

 
 
 Figure 1. 
(a) Single-agent framework for LLMGAs, consisting of a memory system, a reasoning mechanism, and interfaces for perception and action. These modules are connected through the central LLM, driving a continuous gameplay loop where the agent perceives the evolving environment and acts in response.
(b) Multi-LLMGA framework that extends the architecture to populations of agents, including the communication protocol that governs message exchange and the organizational structure that determines topology, task allocation, and role differentiation.
 
 
 
 

### 3.2. Multi-LLMGA Framework

 
 In games, agents interact not only with the environment but also with one another, which naturally calls for explicit mechanisms for coordination and communication. Game environments impose realistic constraints on information sharing: observations are distributed across agents, communication channels are often bandwidth-limited, and direct sharing of internal states or memories is typically disallowed or time-constrained. In such environments, communication between agents needs to be explicitly designed to operate under constrained channels ( Zhang et al., 2024d ) . Communication protocols therefore formalize what information is shared, when it is shared, and how it is interpreted by the receiver ( Qian et al., 2025a ) . Without such protocols, directly transmitting raw observations is often inefficient, noisy, and prone to conflict with an agent’s local beliefs.

 
 
 When multiple agents act in a shared environment, decision conflicts, redundant actions, and inconsistent plans become unavoidable. We need explicit mechanisms to manage these challenges, such as introducing decision authority and responsibilities. Organizational structures are therefore necessary to manage, track and constrain how decisions are made and combined ( Li et al., 2024b ; Qian et al., 2025a ) . By defining role assignments and coordination pathways, organizational structures reduce coordination complexity and prevent uncontrolled interaction among agents. Such organizational structures help prevent situations in which coordination overhead grows combinatorially with the number of agents, thereby reducing inefficiency in cooperative behavior.

 
 
 

### 3.3. Game Taxonomy for LLMGA Design

 
 The way a game agent is designed cannot be isolated from the environment in which it operates: Different game genres foreground distinct capabilities and place different challenges on agent design. For example, action games like Street Fighter demand far quicker reactions than strategy games like Poker, while requiring much less reasoning. Therefore, a taxonomy that captures how these characteristics shape agent design is essential.

 
 
 Clarke et al. ( Clarke et al., 2017 ) critically examine how conventional video game genre classifications often mix orthogonal dimensions such as mechanics and player structures, thereby lacking conceptual clarity. Building on this insight, we ground our taxonomy in established game studies literature through a gameplay-oriented perspective, drawing on the top-level groupings from SteamDB ( SteamDB, 2025 ) and the classification proposed by et al. ( Lee et al., 2014 ) . To maintain coherence with existing LLMGA studies, we merge narrower categories (e.g., driving/racing, fighting) and additionally include sandbox games, resulting in six major genres as depicted in Table 1 . Building on this categorization, we further introduce a challenge-centered view, where each genre is linked to the core design challenge that most strongly drives agent development.

 
 
 As shown in Table 1 , we identify six representative game genres, each posing distinct design challenges for LLM-based agents. (1) Action games ( , 2013 ; OpenGenerativeAI, 2024 ) unfold in real time and emphasize reflexive control, such as aiming, dodging, or chaining combos under tight temporal constraints. The core challenge is low-latency response, which shapes agent design by requiring fast action and hybrid architectures that reconcile LLM reasoning with frame-level responsiveness; (2) Adventure games ( Wang et al., 2022 ; Hausknecht et al., 2020 ) emphasize exploration and long-horizon quests, where progress depends on remembering locations, items, and unresolved preconditions. The challenge is stateful world modeling, pushing agents to develop memory structures that maintain coherent records of evolving environments and dependencies; (3) Role-playing games ( Xu et al., 2024 ; (FAIR)† et al., 2022 ) center on character customization, where players assume predefined roles with distinct traits and narrative trajectories ( Klevjer, 2012 ) . The key challenge is role fidelity, shaping agent design toward embedding role profiles into memory and reasoning so that dialogue and actions remain persona-consistent over extended horizons; (4) Strategy games ( Hu et al., 2025b ; Ma et al., 2024 ) involve multi-step planning against adaptive adversaries, ranging from fully observable board games to imperfect-information settings with hidden states. Their central challenge is opponent-aware planning, which requires agents to integrate multi-step reasoning with theory-of-mind style opponent modeling; (5) Simulation games ( Park et al., 2023 ) approximate real-world or systemic processes, from individual social life to the evolution of societies. The challenge is dynamics fidelity, shaping agent design to ensure that behaviors remain credible and human-like rather than drifting into unrealistic patterns; (6) Sandbox games ( Mojang Studios, ; Hafner, 2022 ) offer open-ended environments where players set their own objectives, explore, and build. The challenge is open-ended goal progression, which drives designs where agents can generate self-directed goals, decompose them hierarchically, and accumulate reusable skills to sustain long-term play.

 
 
 Table 1. Gameplay taxonomy: game genres, core challenges, and representative environments. 
 
 
 Genre | 
 Core Challenge | 
 
 
 Representative Environments 
 | 

 
 Action games | 
 Low-latency response | 
 
 
 
 
 
 Atari 2600 games ( , 2013 ) ; Procgen ( Cobbe et al., 2020 ) ;
ViZDoom ( Kempka et al., 2016 ) ; 
 
 DeepMind Lab ( Beattie et al., 2016 ) ;
Street Fighter ( OpenGenerativeAI, 2024 ) 
 
 | 

 
 Adventure games | 
 Stateful world modeling | 
 
 
 
 
 
 TextWorld ( Côté et al., 2019 ) ; Jericho ( Hausknecht et al., 2020 ) ; ALFWorld ( Shridhar et al., 2021 ) ; 
 
 ScienceWorld ( Wang et al., 2022 ) ; Red Dead Redemption II ( Tan et al., 2024a ) 
 
 STARLING ( Basavatia et al., 2024 ) 
 
 | 

 
 Role-playing games | 
 Role fidelity | 
 
 
 
 
 
 AvalonBench ( Light et al., 2023 ) ; Werewolf ( Xu et al., 2023 ) ; Diplomacy ( (FAIR)† et al., 2022 ) ; 
 
 Among Us ( Milkowski and Weninger, 2026 ) ; SOTOPIA ( Zhou et al., 2024 ) ; clembench ( Chalamalasetti et al., 2023 ) 
 
 | 

 
 Strategy games | 
 Opponent-aware planning | 
 
 
 
 
 
 Chess/Go ( Feng et al., 2024a ; Toshniwal et al., 2022 ) ; Poker ( Gupta, 2023 ; Huang et al., 2024 ) ; 
 
 Pokémon Battles ( Hu et al., 2024 ) ; StarCraft II ( Ma et al., 2024 ) 
 
 Card Games ( Wang et al., 2025d ) ; LLM-PySC2 ( Li et al., 2025c ) 
 
 | 

 
 Simulation games | 
 Dynamics fidelity | 
 
 
 
 
 
 Generative Agents ( Park et al., 2023 ) ; Humanoid Agents ( Wang et al., 2023b ) ; 
 
 AgentSims ( Lin et al., 2023 ) ;
LyfeGame ( Kaiya et al., 2023 ) ; CivRealm ( Qi et al., 2024 ) ; 
 
 Artificial Leviathan ( Dai et al., 2024 ) 
 
 IndoorWorld ( Wu et al., 2025 ) ; Moltbook ( Li et al., 2026 ) 
 
 | 

 
 Sandbox games | 
 Open-ended goal progression | 
 
 
 
 
 
 Minecraft ( Mojang Studios, ) ; MineDojo ( Fan et al., 2022 ) ; Crafter ( Hafner, 2022 ) 
 
 Plancraft ( Dagan et al., 2025 ) ; UnrealZoo ( Zhong et al., 2025 ) 
 
 | 

 
 
 
 

## 4. Memory System of LLMGA

 
 LLMGAs require memory systems that encode and retain prior experience to ensure coherent and efficient interaction. Following classic distinctions in cognitive psychology ( Baddeley and Hitch, 1974 ; Baddeley, 2012 ) , we conceptualize an agent’s memory as working memory and long-term memory.

 
 
 In cognitive psychology, working memory functions as a transient and limited-capacity buffer that temporarily stores and manipulates information needed for ongoing cognitive processing ( Baddeley and Hitch, 1974 ; Baddeley, 2012 ) . In LLMGAs, this role is fulfilled by the model’s short context window and auxiliary mechanisms that keep recent observations “in mind”. For working memory, we examine three key mechanisms. The first is context extension , which enlarges the effective context window so that recent events can be accommodated within short-term processing. The second is memory compression , which condenses lengthy inputs into compact representations, reducing capacity limits while preserving essential content. The third is active maintenance , which explicitly preserves recent bindings, plans, and intermediate states, preventing short-term drift and inconsistency caused by temporal decay.

 
 
 In contrast, long-term memory refers to the durable store of information that persists over extended periods beyond the limited span of working memory ( Tulving and others, 1972 ; Squire, 2004 ) . An LLMGA operating over long horizons faces three fundamental challenges naturally. As the storage is limited and most observations are transient, noisy, or low-value, the agent needs mechanisms to decide not only what information should persist beyond the immediate context, but also when transient traces should be committed to durable storage, so that memory remains useful and tractable. This motivates the role of memory consolidation . Once memories are retained, raw observations alone are insufficient to support abstraction, generalization, or efficient access. This creates the need to organize experiences into structured representations, motivating memory structuring . Finally, because decision-making in a given situation typically depends on only a small subset of relevant past memories, the agent needs to selectively reactivate critical information into working memory, motivating memory retrieval . Figure 2 presents the structure of this section of different components within the memory system.

 
 
 Figure 2. Overview of the memory system of LLMGAs. 
 
 

### 4.1. Working Memory

 
 Recent studies can be grouped into three categories. First, capacity extension enlarges the effective span of working memory by expanding positional encodings or restructuring attention. Second, memory compression distills lengthy or redundant input into more salient representations, mirroring the cognitive process of recoding multiple stimuli into higher-order units to overcome capacity limits ( Cowan, 2001 ) . Finally, active maintenance explicitly preserves variable bindings and states over short time scales, mirroring the human use of rehearsal to prevent rapid forgetting and inconsistency due to temporal decay ( Baddeley and Hitch, 1974 ; Cowan, 2001 ) .

 
 
 Context Extension. Context refers to the input tokens that the LLM can access when generating a new token, which is bounded by its context length ( Brown et al., 2020 ) . To overcome this, recent research focuses on extending the effective scope of the context window without full retraining. In LLM, position refers to the relative order of tokens within this context, typically represented through positional encodings that allow the model to distinguish token order in a sequence ( Vaswani et al., 2017 ) . Position-based approaches modify positional encodings to enable length extrapolation.

 
 
 Position Interpolation (PI) rescales Rotary Position Embeddings (RoPE) to support longer sequences with minimal fine-tuning. YaRN observes that uniform scaling distorts high-frequency positional components and instead applies non-uniform, frequency-aware
interpolation that better preserves local token
relationships ( Peng et al., 2024 ) . LongRoPE further identifies that different RoPE dimensions tolerate different amounts of extension and searches for per-dimension rescaling factors, scaling context
to over 2M tokens ( Ding et al., 2024 ) . Beyond positional scaling, several methods restructure attention over long inputs. For example, Parallel Context Windows (PCW) processes long sequences by dividing them into coordinated segments without retraining ( Ratner et al., 2023 ) , while PoSE enables generalization to longer contexts via sparse positional encoding ( Zhu et al., 2024 ) .

 
 
 Memory Compression. As input length grows, LLM performance often degrades due to limited capacity to maintain and manipulate multiple information items simultaneously ( Gong et al., 2024a ) . To alleviate this bottleneck, memory compression techniques aim to condense long contexts into compact representations that preserve salient information. One line of work introduces soft or virtual tokens that summarize longer text spans, allowing the model to condition on compressed representations rather than the full input ( Lester et al., 2021 ; Ge et al., 2024 ; Mu et al., 2023 ) . Representative examples include AutoCompressor, which learns segment-level summary vectors ( Chevalier et al., 2023 ) , and GIST, which trains virtual tokens to encode the essential content of a prompt for reuse ( Mu et al., 2023 ) .

 
 
 Another direction focuses on hierarchical summarization, where long sequences are recursively condensed into higher-level abstractions. Methods such as chain-of-summarization ( Ma et al., 2024 ) incrementally condense game-state trajectories by segmenting the temporal sequence into short windows and recursively summarizing them into higher-level representations. Similarly, NUGGET ( Qin and Van Durme, 2023 ) organizes long contexts into structured summaries, enabling efficient retrieval and reasoning while keeping inputs within the context window.

 
 
 Active Maintenance. Active maintenance refers to preserving the contents of working memory over short intervals to ensure continuity in reasoning and action. In LLMGAs, failure to maintain such short-term state often leads to inconsistent decisions, even when recent events remain within the context window. A representative example is shown in PokéLLMon ( Hu et al., 2025b ) (Figure 3 ), where agents repeatedly switch Pokémon in consecutive turns instead of attacking, and the issue is further exacerbated when chain-of-thought reasoning is adopted.

 
 
 From the perspective of generation, reasoning introduces cumulative stochasticity that can lead to divergent decisions. Self-Consistency CoT (SC-CoT) ( Wang et al., 2023a ) attempts to mitigate this inconsistency by applying majority voting across reasoning paths in every step. One effective solution is Last-Thoughts ( Hu et al., 2025b ) , which explicitly carries the reasoning trace from the previous step into the next prompt, anchoring decisions to prior deliberation and substantially improving consistency. Related approaches maintain short-term state by explicitly carrying compact summaries across steps.
Belief-state maintenance summarizes the agent’s current understanding for reuse ( Li et al., 2023a ) ,
while MEM1 ( Zhou et al., ) and HiAgent ( Hu et al., 2025a ) update concise shared or subgoal-level memory states to retain salient information and improve short-horizon consistency. In a similar spirit, StateFlow conceptualizes an agent’s task-solving process as a state machine, explicitly tracking task state through states and transitions while delegating sub-tasks to actions ( Wu et al., 2024b ) .

 
 
 Figure 3. Illustration of temporal inconsistency ( Hu et al., 2025b ) : When facing a powerful opponent, the agent tends to switch different Pokémon in consecutive steps rather than taking attack. 
 
 
 

### 4.2. Long-Term Memory

 
 Recent agent architectures emphasize three fundamental processes in the design of long-term memory systems. First, memory consolidation determines when and what to commit from transient buffers to durable storage. Second, memory structuring addresses how stored content is organized. Finally, memory retrieval specifies how past knowledge is re-activated to guide ongoing decision-making. These components together ensure that long-term memory effectively archives past experience and supports future behavior.

 
 
 Consolidation. In cognitive psychology, the transfer of information from working memory into long-term memory is termed consolidation ( Baddeley, 2012 ; Squire, 2004 ) .
For LLMGAs, the analogous process is to decide when and what to commit from transient buffers to durable storage so that memory remains useful and tractable. A common paradigm is signal-triggered consolidation, where specific signals determine whether new information should be committed. In Generative Agents ( Park et al., 2023 ) , each incoming observation is assigned an importance score, and once the cumulative importance of recent events exceeds a threshold, the agent pauses to reflect, producing a summary that is then written into long-term memory. MemoryBank ( Zhong et al., 2024 ) applies a similar principle, committing experiences when their relevance to the goal surpasses a salience threshold. Voyager ( Wang et al., 2024a ) instead uses task outcomes as signals: successful code executions are committed into a skill library, while failed attempts are excluded or down-weighted.

 
 
 More recent works extend write-back into more flexible learning-based schemes. For instance, CoALA ( Sumers et al., 2024 ) models “learning” as an explicit internal action within the agent’s action space, leaving it to the control policy (e.g., LLM) to decide when to encode new information into long-term memory. Self-Controlled Memory ( Wang et al., 2025a ) introduces a trainable memory controller that adaptively decides whether to write or use memory at each step.
The controller is optimized jointly with the LLM through task-level supervision, such that memory updates are triggered only when they improve downstream performance. Continual-learning agents extend this idea: NeSyC pairs an LLM with symbolic reasoning and a contrastive generality-improvement scheme that iteratively hypothesizes, validates, and refines reusable action knowledge, with a memory-based monitor triggering updates as embodied agents transition across ALFWorld, VirtualHome, and Minecraft ( Choi et al., 2025 ) . O3D mines large offline interaction logs to automatically discover reusable skills and distill generalizable knowledge, improving long-horizon decision-making ( Xiao et al., 2024 ) .

 
 
 Figure 4. Illustration of representative memory structuring approaches. 
 
 
 Structuring. Representative structures range from simple text fragments to highly organized graphs and implicit parametric storage, as shown in Figure 4 . The most direct approach is to store observations as chunks ( Park et al., 2023 ) . To facilitate later retrieval, each chunk can be augmented with metadata such as timestamps, importance scores ( Park et al., 2023 ) , or Q-values ( Zhang et al., 2024c ) . For example, MrSteve specializes such episodic chunks into a Place Event Memory that records what, where, and when each event occurred, enabling selective recall during Minecraft play ( Park et al., 2025 ) . Beyond raw fragments, many systems adopt a key–value representation, where keys encode identifiers or semantic descriptors, and values store the corresponding content. This allows fast lookups and supports multimodal inputs: for example, Voyager represents keys as program descriptions paired with code snippets as values ( Wang et al., 2024a ) , while JARVIS-1 stores visual observations as keys and successful execution plans as values ( Wang et al., 2025h ) .

 
 
 To capture hierarchical relations, memories can be recursively clustered into a tree structure. Generative Agents ( Park et al., 2023 ) , RAPTOR ( Sarthi et al., ) , and MemTree ( Rezazadeh et al., 2025 ) all build memory trees where raw chunks form the leaves, and higher layers summarize increasingly abstract topics. Although the update mechanism differs, the underlying idea is to let new experiences traverse the tree, merging with existing nodes or forming new branches, while recursively updating parent summaries. Game agents adopt similar hierarchies: GLoW maintains a dual-scale world memory pairing a global trajectory frontier of high-value discoveries with local multi-path reflection for hard-exploration text-adventure games ( Kim and Hwang, 2026 ) , while MACLA distills trajectories into an external hierarchical procedural memory whose entries are tracked by Bayesian reliability estimates and contrastively refined ( Forouzandeh et al., 2026 ) . An alternative design is to use graph -structured memory. In knowledge graph approaches, nodes correspond to entities and edges correspond to semantic relations, typically extracted as triplets from text chunks, emphasizing fact representation ( Li et al., 2024a ; Anokhin et al., 2025 ) . A-MeM ( Xu et al., 2025c ) organizes memory into a network of atomic notes enriched with tags and context, and edges represent semantic links between related notes, emphasizing interlinked note-taking and allowing updates to existing nodes. KLPEG structures game elements, task dependencies, and causal relations into such a graph for multi-hop reasoning over update logs ( Mu et al., 2025 ) , and Optimus-1 pairs a hierarchical directed knowledge graph of world knowledge with a multimodal experience pool for long-horizon Minecraft tasks ( Li et al., 2024c ) .

 
 
 Finally, some work explores parametric storage , where memory is encoded implicitly in the model’s parameters rather than as external data. This perspective aligns with human cognition, which does not store verbatim text but instead internalizes experience. Fine-tuning on domain knowledge or episodic data can thus endow LLMs with embedded semantic or procedural memory ( Weber et al., 2024 ; Feng et al., 2024b ) . For instance, CharacterLLM fine-tuned on synthetic character experiences can recall detailed knowledge of people, events, and objects in a role-consistent manner ( Shao et al., 2023 ) .

 
 
 Retrieval. Memory retrieval is the process of reactivating stored information to guide current reasoning and action. Human studies also highlight that retrieval is selective and subject to recency, salience, and interference effects ( Ebbinghaus, 1913 ; Cowan, 2001 ) , which resonate with the design of LLMGAs. One common strategy is metadata-based retrieval, where memories are annotated with attributes such as timestamps, importance scores, or Q-values and ranked accordingly during retrieval ( Park et al., 2023 ; Zhong et al., 2024 ; Zhang et al., 2024c ) .

 
 
 A second approach is semantic retrieval, where queries are embedded into a vector space and compared with stored representations. Generative Agents, for instance, compute cosine similarity between a self-instructed query and stored text memories ( Park et al., 2023 ) . In key–value settings, similarity is measured between the query and the key, with the associated value returned. This design allows flexibility across modalities: Voyager retrieves executable code by comparing program descriptions ( Wang et al., 2024a ) , while JARVIS-1 retrieves action plans from multimodal keys that combine task descriptions and visual observations ( Wang et al., 2025h ) . Building on example-based retrieval, agents can curate a database of their own successful trajectories and retrieve them as in-context exemplars, improving sequential decision-making on the ALFWorld and Wordcraft games without manual prompt engineering ( Sarukkai et al., 2025 ) .

 
 
 For more structured memories, retrieval can exploit graph or tree topologies. Graph-based retrieval begins by identifying relevant nodes using semantic or lexical cues, then traverses edges to explore multi-hop neighborhoods, finally synthesizing the resulting subgraph into a coherent narrative for the LLM to consume ( Li et al., 2024a ; Anokhin et al., 2025 ) . For Minecraft, goal-oriented graphs apply this graph traversal to objectives, so that retrieving a goal recursively surfaces its prerequisites for coherent multi-step plans ( Leung et al., 2026 ) . Tree-based retrieval instead performs hierarchical traversal: starting from the root, the agent selects top- k k relevant nodes at each level based on similarity, gradually descending to finer-grained leaves. Some variants collapse the hierarchy into a flat pool of summaries and retrieve based purely on semantic similarity ( Sarthi et al., ; Rezazadeh et al., 2025 ) .

 
 
 Finally, for parametric storage , knowledge is embedded implicitly in model weights rather than explicit structures. Such retrieval resembles implicit or procedural memory in humans, in which skills and habits are expressed without deliberate recall ( Shao et al., 2023 ) . Table 2 summarizes representative LLMGAs by their memory design, showing the diversity of memory mechanisms across different game environments.

 
 
 Table 2. Summary of representative LLMGAs in terms of memory design. 
 
 
 
 LLMGA | 
 Environment | 
 Working Memory | 
 Long-Term Memory | 

 
 Reflexion ( Shinn et al., 2023 ) | 
 ALFWorld | 
 In-episode experience | 
 Reflection on previous episodes | 

 
 PokéLLMon ( Hu et al., 2025b ) | 
 Pokémon Battles | 
 Active maintenance (last-step thoughts) | 
 External game knowledge for retrieval | 

 
 TextStarCraft ( Ma et al., 2024 ) | 
 StarCraft II | 
 Memory compression (chain-of-summarization) | 
 | 

 
 SuspicionAgent ( Guo et al., 2024 ) | 
 Leduc Hold’em | 
 In-episode experience | 
 Reflection on previous episodes | 

 
 ProAgent ( Zhang et al., 2024a ) | 
 Overcooked-AI | 
 Active Maintenance (Intention and belief) | 
 Past experience for retrieval | 

 
 Voyager ( Wang et al., 2024a ) | 
 Minecraft | 
 Short-term code feedback | 
 Successful code for retrieval | 

 
 GTIM ( Zhu et al., 2023 ) | 
 Minecraft | 
 Short-term action feedback | 
 Successful plan for retrieval | 

 
 JARVIS-1 ( Wang et al., 2025h ) | 
 Minecraft | 
 Short-term situational context | 
 Successful multimodal plan for retrieval | 

 
 GenerativeAgents ( Park et al., 2023 ) | 
 Small Village | 
 Memory compression (tree-based reflection) | 
 Streaming memory with metadata | 

 
 E2WM ( Xiang et al., 2024 ) | 
 VirtualHome | 
 In-context dialogue | 
 Exploration experience for fine-tuning | 

 
 LLMPlanner ( Song et al., 2023 ) | 
 ALFRED | 
 In-episode experience | 
 Exemplar plan for retrieval | 

 
 CharacterLLM ( Shao et al., 2023 ) | 
 Role-playing QA | 
 In-context dialogue | 
 Synthetic experience for fine-tuning | 

 
 Optimus-1 ( Li et al., 2024c ) | 
 Minecraft | 
 Short-term observation | 
 Hybrid knowledge graph + experience pool | 

 
 MrSteve ( Park et al., 2025 ) | 
 Minecraft | 
 In-episode observation | 
 Place Event Memory (what-where-when) | 

 
 GLoW ( Kim and Hwang, 2026 ) | 
 Jericho | 
 Local multi-path reflection | 
 Dual-scale world memory | 

 
 MACLA ( Forouzandeh et al., 2026 ) | 
 ALFWorld | 
 In-episode experience | 
 Hierarchical procedural memory | 

 
 KLPEG ( Mu et al., 2025 ) | 
 Overcooked, Minecraft | 
 | 
 Knowledge-graph memory | 

 
 GoG ( Leung et al., 2026 ) | 
 Minecraft | 
 | 
 Goal-oriented graph for retrieval | 

 
 
 
 
 

## 5. Reasoning of LLMGA

 
 In cognitive science, reasoning is understood as the process of constructing and manipulating internal representations of known information to uncover implicit relations and abstract structures, thereby enabling conclusions that extend beyond what is explicitly given ( Johnson-Laird, 2010 ; Evans, 2008 ) . In LLMGAs, reasoning serves as the central mechanism that transforms perceived and retrieved information into coherent plans, decisions, and explanations. It unfolds through language, by generating intermediate thought sequences that externalize internal deliberation and guide subsequent actions ( Wei et al., 2022 ; Kojima et al., 2022 ) .

 
 
 For instruction-guided reasoning, designed prompts elicit reasoning behavior directly at inference time. The first mechanism is chain-of-thought , which guides the model to articulate intermediate steps before arriving at an answer. The second is search-based reasoning , which explores multiple reasoning paths in parallel and selects among them to ensure consistency. The third is reflective reasoning , which iteratively improves reasoning across steps by incorporating internal self-critique or external signals.

 
 
 For fine-tuning paradigms, reasoning abilities are improved through optimization on data or experience interacted with the game environments. The first mechanism is supervised fine-tuning , where agents imitate expert demonstrations to acquire reasoning behaviors. The second is reinforcement learning , which updates policies or value models to optimize reasoning with task rewards. The third is preference optimization , which contrasts preferred and dispreferred generations to bias reasoning toward desirable outcomes. Figure 5 presents the structure of this section of different components within the reasoning mechanism.

 
 
 Figure 5. Categorization of reasoning mechanisms of LLMGAs. 
 
 

### 5.1. Instruction-Guided Reasoning

 
 Prior studies have demonstrated that reasoning abilities can be elicited and amplified by deliberate prompting strategies at inference time, which guide models to externalize intermediate steps rather than relying solely on direct answer generation ( Wei et al., 2022 ; Kojima et al., 2022 ) . We categorize existing methods into three groups. Chain-of-thought prompting elicits a single linear reasoning path, but is prone to error propagation. Search-based reasoning mitigates this by generating and organizing multiple trajectories to enhance robustness. Reflective reasoning emphasizes temporal refinement, where reasoning is iteratively improved using signals from prior experience or the environment.

 
 
 Chain-of-Thought. CoT ( Wei et al., 2022 ) is the basic approach that prompts LLMs to conduct intermediate reasoning before generating the answers, as shown in Figure 6 . Since generation can be seen as an auto-regressive process of searching the next token in the latent space, the introduction of intermediate reasoning enhances the ability to traverse greater distances in that latent space, making LLMs capable of addressing more complex tasks. The ReAct ( Yao et al., 2023 ) agent interleaves CoT reasoning and actions using few-shot prompting in text-based games. In their approach, reasoning acts as a mechanism for the agent to periodically check its task progress and plan its next steps.

 
 
 Intermediate reasoning introduces additional stochasticity, which can lead to inconsistent outputs. For instance, in Pokémon Battles, CoT may cause agents to panic-switch Pokémon in consecutive turns ( Hu et al., 2025b ) , as shown in Figure 3 .
Moreover, once an early step deviates, subsequent tokens may inherit and magnify the error ( Madaan et al., 2024 ) . Self-Refine ( Madaan et al., 2024 ) , GPTLens ( Hu et al., 2023 ) and RCI ( Kim et al., 2024 ) aim to mitigate error propagation through self-criticism, first generating reasoning thoughts and then evaluating and refining them to improve the reasoning generation.

 
 
 Search-based Reasoning. A major limitation of single-path chain-of-thought is fragility: randomness in sampling may yield inconsistent outputs, and early errors can propagate through the chain ( Wang et al., 2023a ; Madaan et al., 2024 ) . Search-based methods mitigate this by generating multiple intermediate reasoning candidates and then selecting, aggregating, or revising them. As shown in Figure 6 , Self-Consistency ( Wang et al., 2023a ) alleviates inconsistency by prompting LLMs to generate multiple chains of thoughts independently, and conduct majority voting on the final answer to find the most consistent reasoning path. Tree-of-Thoughts ( , 2023 ) focuses on preventing error propagation by proposing multiple intermediate thoughts and selecting the correct one. Specifically, it decomposes a task into multiple steps, generates candidate thoughts for each step, and selects the most promising one, making the reasoning process resemble traversing a tree of thoughts. THREAD frames generation as a thread of execution that dynamically spawns child threads for intermediate reasoning, improving adaptive decomposition ( Schroeder et al., 2025 ) . For graph-based reasoning, Graph-of-Thoughts ( Besta et al., 2024 ) aggregates thoughts across different reasoning paths, converting a tree structure into a directed acyclic graph (DAG). SPRING ( Wu et al., 2024d ) constructs a template DAG in which each node corresponds to a question or instruction used to prompt LLMs for progressive reasoning. In their study, the authors prompt LLMs to summarize the Crafter paper ( Hafner, 2022 ) into a DAG and then progressively traverse the DAG to answer these questions, thereby guiding the model through a step-by-step reasoning process. AgentKit composes agent reasoning as a dynamic graph of modular natural-language nodes that support hierarchical planning and reflection in the Crafter sandbox game ( Wu et al., 2024c ) .

 
 
 Figure 6. Illustration of representative instruction-guided reasoning approaches. O t O_{t} denotes the observation at step t t , a t a_{t} denotes the action output at step t t , and thg denotes an intermediate reasoning step. In Reflexion, f t f_{t} represents feedback and r ​ e ​ f t ref_{t} denotes the resulting reflection at step t t . 
 
 
 Reflective Reasoning. Unlike generic LLM agents often evaluated on single-turn tasks, game agents operate within an observation–action–feedback loop, continuously perceiving the environment, taking actions, and adjusting decisions based on the resulting outcomes. Reflective reasoning builds on this loop by allowing agents to analyze the outcomes of their own actions and incorporate these reflections into future reasoning and behavior, as Reflexion ( Shinn et al., 2023 ) shown in Figure 6 . This introduces a temporal dimension to reasoning, enabling the integration of experience over time.

 
 
 Studies have shown that such temporal interaction enables LLMGAs to evolve over time by integrating feedback from past trajectories. The most direct form is reflection on failure: when an action fails, the agent can reuse the error signal to avoid repeating the same mistake. For instance, environments may provide explicit feedback such as “I cannot make a stone shovel because I need 2 more sticks” in MineCraft, which agents like Voyager ( Wang et al., 2024a ) and GTIM ( Zhu et al., 2023 ) exploit to iteratively refine their plans. Beyond explicit signals, reflective mechanisms such as Reflexion ( Shinn et al., 2023 ) , DEPS ( Wang et al., 2023c ) , and ProAgent ( Zhang et al., 2024a ) guide agents to analyze their own chain-of-thought traces, identify where reasoning went wrong, and incorporate these insights into subsequent decisions. Even in environments with sparse feedback, agents can still benefit from heuristic signals ( Shinn et al., 2023 ) . Recent game agents broaden the corrective signal beyond self-generated critiques. DGAP trains a discriminator from a few demonstrations to score how well each action aligns with optimal ones, then prompts the LLM to refine toward higher critic scores ( Qian et al., 2025b ) . LEAP corrects a student agent with critiques from a teacher granted privileged state information only at training time, letting it surpass the teacher ( Choudhury and Sodhi, 2025 ) . Framing the LLM as an in-context reinforcement learner, scalar-reward feedback can be optimized over successive inference-time rounds ( Song et al., 2026 ) . ReCAPA further mitigates cascading failures by predicting and contrasting deviations across actions, subgoals, and trajectories on benchmarks such as MineDojo ( Zeng et al., 2026 ) .

 
 
 In addition to learning from failures, reflective reasoning can also benefit from reflecting on successes. Successful trajectories not only consolidate effective strategies but also provide contrastive signals when compared against failures. ExpeL ( Zhao et al., 2024a ) leverages this idea by retrieving the most relevant successful experiences, summarizing common patterns, and deriving insights through success–failure comparisons. Similarly, KWM ( Qiao et al., 2024 ) extracts task knowledge from expert-demonstrated trajectories and distills it into a dedicated world knowledge model, which is then used to guide the agent’s planning in future episodes. AutoManual has a builder agent refine interaction experience into human-readable rule manuals that curb hallucination, reaching high success on ALFWorld from a single demonstration ( Chen et al., 2024c ) , while DiVE discovers, verifies, and evolves world-dynamics rules from a handful of demonstrations to reach human-level reward in Crafter and MiniHack ( Sun et al., 2024 ) . In summary, reflective reasoning shares the basic idea of reinforcement learning that uses feedback to correct mistakes and reinforce successful strategies, embodying the principle of learning through interaction with the environment.

 
 
 

### 5.2. Fine-tuning for Improving Reasoning

 
 In this subsection, we examine fine-tuning techniques for optimizing reasoning and action generation. Based on the training strategy, existing methods can be grouped into three categories. Supervised fine-tuning learns from expert trajectories to imitate reasoning and action generation. Reinforcement learning updates policies with reward feedback, reinforcing reasoning and actions that lead to favorable outcomes. Preference optimization leverages comparisons between better and worse trajectories to align models without the need for explicit reward models. It is worth noting that some methods mentioned below optimize only the final action without explicit reasoning, however, they can be extended to improve reasoning by eliciting chain-of-thought, allowing reasoning to be shaped through its effect on action outcomes ( Kahneman, 2011 ) .

 
 
 Supervised Fine-Tuning. Supervised fine-tuning trains LLM agents on collected trajectories to maximize the likelihood of reproducing demonstrated reasoning and actions. The most common approach is behavior cloning, where agents directly imitate expert demonstrations. Such trajectories may come from human experts ( Reed et al., 2022 ) , from state-of-the-art agents ( Lin et al., 2024 ) , or from teacher LLMs that generate rollouts for training student models ( Zeng et al., 2024 ) . Behavior cloning is widely adopted as an initialization strategy, providing a strong prior policy that can later be refined by reinforcement learning ( , 2024 ; Song et al., 2024 ) .

 
 
 Building on this idea, rejection sampling fine-tuning introduces a selection stage before training. Instead of imitating all trajectories, the model generates multiple candidates and filters them according to predefined criteria, such as binary success/failure signals or reward estimates. RFT ( Liu et al., 2024c ) , for example, fine-tunes models only on successful trajectories, while other works employ environment-provided or model-estimated rewards to guide sample selection ( Touvron et al., 2023 ) . Although this improves data quality, it can be inefficient when the agent initially produces few successful rollouts. Wang et al. show that supervised fine-tuning on high-quality gameplay data lets LLMs approach strong game AIs across eight complex card games and acquire proficiency in several games at once ( Wang et al., 2025d ) . ChessLLM supervised-fine-tunes a language model on complete chess games trajectories to reach a 1788 Elo against Stockfish, showing that training on longer game sequences yields a large rating advantage ( Zhang et al., 2025b ) . For socially intelligent agents, SOTOPIA- π \pi combines behavior cloning with self-reinforcement on social-interaction episodes filtered by LLM ratings, letting a 7B model match GPT-4’s social-goal completion ( Wang et al., 2024d ) .

 
 
 Reinforcement Learning. Reinforcement learning (RL) provides another major paradigm for improving reasoning and action generation in LLM agents. Existing game agents ( Carta et al., 2023 ; Du et al., 2023 ; Tan et al., 2024b ; , 2024 ) mainly adopt the Proximal Policy Optimization (PPO) algorithm ( Schulman et al., 2017 ) , where the model is trained as a policy π ⁡ ( a t ∣ s t ) \pi(a_{t}\mid s_{t}) (without explicit reasoning) and updated using advantage-weighted gradients to favor actions leading to higher rewards. Alongside the policy model, PPO also learns a value function to estimate the relative quality of state–action pairs. While effective, applying RL to LLMs faces the challenge of an enormous generation space, which often leads to inadmissible actions. To address this, some methods compute the probability distribution of admissible actions by the chain rule before sampling, ensuring that the generated actions remain valid ( Carta et al., 2023 ; Tan et al., 2024b ) .

 
 
 Recent works further integrate explicit reasoning into RL training, where the LLM is trained as a policy π ⁡ ( r ​ s t , a t ∣ s t ) \pi(rs_{t},a_{t}\mid s_{t}) . Reinforced Fine-Tuning (ReFT) ( Trung et al., 2024 ) introduces chain-of-thought supervision into PPO, encouraging the model to generate reasoning paths that lead to correct answers. However, because reasoning tokens are often much longer than action tokens, naive optimization can overweight reasoning relative to actions. Zhai et al. ( Zhai et al., 2025 ) propose downscaling the likelihood of reasoning steps, showing that moderate scaling achieves better balance between planning and acting. Beyond policy optimization, value-based methods such as Q-learning extend RL to reasoning by treating partial generations as states and token expansions as actions. This formulation allows the use of search algorithms, such as Best-of-N sampling or Monte Carlo tree search, to evaluate and expand reasoning paths guided by the Q-function ( Zhang et al., 2024b ) . Game agents extend this search-integrated view: SEEA-R1 fuses Monte Carlo tree search with a Tree-GRPO objective and a multimodal generative reward model to self-evolve on long-horizon ALFWorld tasks ( Tian et al., 2025 ) .

 
 
 A challenge is that conventional reward signals (and the value estimates derived from them) are provided only at the action level, providing no feedback on the intermediate reasoning steps. This causes error to propagate through the reasoning until the final outcome is known. To address this limitation, Process Reward Modeling (PRM) ( Lightman et al., 2023 ) supplies dense feedback by explicitly evaluating intermediate reasoning steps. In games, the reward signal and objective are themselves often tailored to the task: LARM derives rewards from a large LLM acting as a referee to train a compact policy for long-horizon Minecraft ( Li et al., 2025b ) , DVM imposes a win-rate-constrained decision-chain reward to make Werewolf agents controllable ( Zhang et al., 2025c ) , GFlowVLM replaces scalar-reward maximization with reward-proportional generative flow networks for card games such as BlackJack ( Kang et al., 2025 ) , and DipLLM sets an approximate-equilibrium policy as the learning target for Diplomacy, matching Cicero with a fraction of the data ( Xu et al., 2025b ) . Two further paradigms sidestep hand-specified rewards. First, self-play and learning-through-play supply the signal directly: SPIRAL trains on zero-sum games such as TicTacToe and Kuhn Poker with role-conditioned advantage estimation ( Liu et al., 2026 ) , SPAG plays both sides of Adversarial Taboo to improve general reasoning ( Cheng et al., 2024 ) , and an RL-instructed discussion policy is learned through One Night Ultimate Werewolf ( Jin et al., 2024 ) . Second, LLMs shape the RL loop without acting as the policy: EnvGen generates and adapts training environments so a small agent surpasses GPT-4 agents on Crafter ( Zala et al., 2024 ) , and language-guided exploration scores candidate decisions for an RL explorer in ScienceWorld ( Golchha et al., 2024 ) .

 
 
 Preference Optimization. The idea of preference optimization was first explored in games, where OpenAI demonstrated that human preference comparisons could be used to train reward models for Dota 2 ( Christiano et al., 2017 ) . This principle of optimizing agents by favoring trajectories preferred by humans rather than relying on hand-crafted rewards later became the foundation for aligning language models. Building on this, Direct Preference Optimization (DPO) ( Rafailov et al., 2023 ) enables contrastive training without an explicit reward model by maximizing the margin between preferred and non-preferred generations, thereby simplifying the optimization process and reducing cost. In the context of game agents, this preference-based framework can also be applied at the trajectory or step level: ETO ( Song et al., 2024 ) alternates between exploration and fine-tuning with DPO on successful vs. failed rollouts, while IPR ( Xiong et al., 2024 ) extends this to step-wise preference optimization, pairing reasoning steps according to the average reward calculated via Monte Carlo method. In Table 3 , we list representative LLMGAs by their reasoning mechanism design, aligned with the two dimensions of our categorization.

 
 
 Table 3. Summary of representative LLMGAs in terms of reasoning mechanism. 
 
 
 
 LLMGA | 
 Environment | 
 Instruction-guided Reasoning | 
 Fine-tuning for Improving Reasoning | 

 
 ReAct ( Yao et al., 2023 ) | 
 ALFWorld, etc. | 
 CoT | 
 | 

 
 Reflexion ( Shinn et al., 2023 ) | 
 ALFWorld, etc. | 
 CoT + Reflective reasoning | 
 | 

 
 ADAPT ( Prasad et al., 2024 ) | 
 ALFWorld, etc. | 
 As-needed CoT (planning) | 
 | 

 
 SwiftSAGE ( Lin et al., 2024 ) | 
 ScienceWorld | 
 As-needed CoT (planning) | 
 | 

 
 ETO ( Song et al., 2024 ) | 
 ALFWorld, etc. | 
 | 
 Trajectory-level preference optimization | 

 
 IPR ( Xiong et al., 2024 ) | 
 ALFWorld, etc. | 
 | 
 Step-level preference optimization | 

 
 GLAM ( Carta et al., 2023 ) | 
 BabyAI-Text | 
 | 
 RL fine-tuning | 

 
 TWOSOME ( Tan et al., 2024b ) | 
 Overcooked-AI | 
 | 
 RL fine-tuning | 

 
 Xu et al. ( Xu et al., 2023 ) | 
 Werewolf | 
 Reflective reasoning | 
 | 

 
 Xu et al. ( Xu et al., 2024 ) | 
 Werewolf | 
 | 
 RL-based candidate selection | 

 
 WarAgent ( Hua et al., 2023 ) | 
 Diplomacy-like | 
 Structural reasoning | 
 | 

 
 PokéLLMon ( Hu et al., 2024 ; Hu et al., 2025b ) | 
 Pokémon Battles | 
 Consistent reasoning generation | 
 | 

 
 ChessGPT ( Feng et al., 2024a ) | 
 Chess | 
 | 
 Supervised fine-tuning | 

 
 PokerGPT ( Huang et al., 2024 ) | 
 Texas Hold’em | 
 | 
 RL from human feedback | 

 
 SuspicionAgent ( Guo et al., 2024 ) | 
 Leduc Hold’em | 
 Theory-of-mind reasoning | 
 | 

 
 HLA ( Liu et al., 2024a ) | 
 Overcooked | 
 As-needed CoT (planning) | 
 | 

 
 S-Agents ( Chen et al., 2024a ) | 
 Minecraft | 
 Goal decomposition, evaluation | 
 | 

 
 HAC ( Zhao et al., 2024d ) | 
 Minecraft | 
 Goal decomposition, correction, evaluation | 
 | 

 
 Voyager ( Wang et al., 2024a ) | 
 Minecraft | 
 Code as policy, correction | 
 | 

 
 DEPS ( Wang et al., 2023c ) | 
 Minecraft | 
 Goal decomposition, reflection, selection | 
 | 

 
 GTIM ( Zhu et al., 2023 ) | 
 Minecraft | 
 Goal decomposition, correction | 
 | 

 
 JARVIS-1 ( Wang et al., 2025h ) | 
 Minecraft | 
 Goal decomposition, reflection | 
 | 

 
 Plan4MC ( Yuan et al., 2023 ) | 
 Minecraft | 
 Goal decomposition | 
 | 

 
 RL-GPT ( Liu et al., 2024b ) | 
 Minecraft | 
 Reasoning as code generation | 
 | 

 
 LLaMARider ( Feng et al., 2024b ) | 
 Minecraft | 
 | 
 Novelty-driven Supervised fine-tuning | 

 
 Project Sid ( AL et al., 2024 ) | 
 Minecraft | 
 Social awareness reasoning | 
 | 

 
 GenerativeAgents ( Park et al., 2023 ) | 
 Sims-like game | 
 Tree-based reflection planning | 
 | 

 
 HumanoidAgents ( Wang et al., 2023b ) | 
 Social | 
 Affective-driven planning | 
 | 

 
 LLMPlanner ( Song et al., 2023 ) | 
 ALFRED | 
 Planning re-planning | 
 | 

 
 Octopus ( , 2024 ) | 
 OctoVerse | 
 Reasoning as code generation | 
 RL fine-tuning | 

 
 ELLM ( Du et al., 2023 ) | 
 Crafter | 
 Situated goal generation | 
 | 

 
 SPRING ( Wu et al., 2024d ) | 
 Crafter | 
 Structural reasoning | 
 | 

 
 PokéChamp ( Karten et al., 2025b ) | 
 Pokémon Battles | 
 Minimax tree search | 
 | 

 
 Strategist ( Light et al., 2025 ) | 
 Avalon, GOPS | 
 Bi-level tree search | 
 | 

 
 Schultz et al. ( Schultz et al., 2025 ) | 
 Chess, Hex | 
 External/internal search | 
 | 

 
 DiffuSearch ( Ye et al., 2025 ) | 
 Chess | 
 Implicit diffusion search | 
 | 

 
 MC-DML ( Shi et al., 2025 ) | 
 Jericho | 
 MCTS with memory | 
 | 

 
 AutoManual ( Chen et al., 2024c ) | 
 ALFWorld | 
 Reflective rule learning | 
 | 

 
 AgentKit ( Wu et al., 2024c ) | 
 Crafter | 
 Dynamic-graph reasoning | 
 | 

 
 DGAP ( Qian et al., 2025b ) | 
 ScienceWorld | 
 Discriminator-guided refinement | 
 | 

 
 DiVE ( Sun et al., 2024 ) | 
 Crafter | 
 World-dynamics reflection | 
 | 

 
 ReCAPA ( Zeng et al., 2026 ) | 
 MineDojo | 
 Hierarchical predictive correction | 
 | 

 
 DipLLM ( Xu et al., 2025b ) | 
 Diplomacy | 
 | 
 RL fine-tuning | 

 
 SPIRAL ( Liu et al., 2026 ) | 
 Zero-sum games | 
 | 
 Self-play RL | 

 
 SPAG ( Cheng et al., 2024 ) | 
 Adversarial Taboo | 
 | 
 Self-play RL | 

 
 ChessLLM ( Zhang et al., 2025b ) | 
 Chess | 
 | 
 Supervised fine-tuning | 

 
 Card Games ( Wang et al., 2025d ) | 
 Card games | 
 | 
 Supervised fine-tuning | 

 
 LARM ( Li et al., 2025b ) | 
 Minecraft | 
 | 
 Referee-based RL | 

 
 SEEA-R1 ( Tian et al., 2025 ) | 
 ALFWorld | 
 | 
 Tree-structured RL | 

 
 DVM ( Zhang et al., 2025c ) | 
 Werewolf | 
 | 
 RL fine-tuning | 

 
 GFlowVLM ( Kang et al., 2025 ) | 
 BlackJack | 
 | 
 GFlowNet fine-tuning | 

 
 
 
 
 

## 6. Perception and Action Interfaces of LLMGA

 
 LLMGAs differ from generic LLM systems in that they operate within a continuous perception-action loop. To support this loop, agents rely on perception and action interfaces that serve as their eyes and hands for interacting with the environment ( Wang et al., 2024a ; Hu et al., 2024 ) . On the input side, the perception interface determines how raw game states are abstracted into representations that can be processed by the LLM, handling textual , symbolic , and visual observations. On the output side, the action interface ensures that the model’s decisions are translated into admissible in-game operations by grounding the LLM outputs into high-level , low-level , and code-based actions. Figure 7 outlines the structure of this section.

 
 
 Figure 7. Overview of perception and action interfaces in LLMGAs. 
 
 

### 6.1. Perception Interface

 
 The perception interface defines how an LLMGA accesses and processes information from the game environment. The most direct and widely adopted way to categorize input-processing methods is based on the modality of the game observation, such as textual, symbolic, or visual forms.

 
 
 Textual Observations. In text-based or dialogue-centric games ( Infocom, 1980 ; Shridhar et al., 2021 ) , the environment state is natively presented in natural language.
In such cases, the agent can directly consume text descriptions as observations without additional preprocessing ( Yao et al., 2023 ; Shinn et al., 2023 ) .
This modality is straightforward, as it aligns with the input format of LLMs, but it is restricted to environments where language is the primary medium of interaction.

 
 
 Symbolic Observations. Some video game environments provide structured state information through APIs or game engines ( Ma et al., 2024 ; Hu et al., 2024 ; Li et al., 2023c ; Mojang Studios, ) .
These symbolic variables (e.g., avatar health, inventory, world coordinates or object properties) can be transformed into a form that the LLM can process, often through textual summaries or structured prompt templates.
For example, Mineflayer ( PrismarineJS, 2013 ) exposes a Minecraft character’s stats and surrounding entities, which can then be summarized into a natural-language prompt ( Hu et al., 2024 ) .
Symbolic observations are efficient when the selected variables can sufficiently capture the essential context, but they risk losing fidelity in complex environments where subtle but critical distinctions, such as object textures, spatial relations, or small visual cues, are omitted from the symbolic representation.

 
 
 Visual Observations . In video games, the agent typically perceives the environment as a sequence of rendered images.
Since LLMs cannot directly operate on raw pixels, the perception interface requires a translator that converts visual signals into interpretable representations.
One approach is vision-to-text translation , where object detectors or pretrained encoders such as CLIP ( Radford et al., 2021 ) produce captions or object lists that can be inserted into prompt templates. For example, an agent in a 3D environment can use an object detector to list visible objects (“a key on the floor, a locked door ahead”) and insert them into the prompt template ( Zhang et al., 2024d ; Song et al., 2023 ) . The agent can also adopt a visual encoder to map images into pre-defined text descriptions ( Wang et al., 2023c ; Wang et al., 2025h ; Du et al., 2023 ) , or a text decoder to generate the caption ( Du et al., 2023 ) to summarize the scene.

 
 
 An alternative is to use multimodal LLMs to directly process raw frames. These models align visual and textual information in a shared representation space, allowing an agent to feed raw images or pixels to the model and get an immediate understanding. Recent works ( de Wynter, 2025 ; Tan et al., 2024a ) leverage general-purpose multimodal LLMs (e.g., GPT-4 Vision ( Achiam et al., 2023 ) ) to interpret game visuals. This direct approach can generalize well to new games, but still requires additional mechanisms to correct errors or uncertainties in its perceptions ( , 2024 ; Tan et al., 2024a ) . Game-specific multimodal models have also been introduced, e.g. , LLMs finetuned on paired image-instruction data for a particular game, such as SteveEye ( Zheng et al., 2023 ) or learned from
environmental feedback through RL such as Octopus ( , 2024 ) . STEVE follows this direction in Minecraft, coupling a vision-perception module that interprets visual observations with an LLM for reasoning and a code-action skill database, trained on the STEVE-21K dataset of vision–environment, QA, and skill-code pairs ( Zhao et al., 2024c ) . Rather than passively consuming observations, ActiveVOO performs value-of-observation guided active sensing for partially observable embodied planning, quantifying the utility of sensing actions from LLM and VLM commonsense priors so the agent perceives only task-relevant objects in ALFWorld ( Liu et al., 2025b ) .

 
 
 

### 6.2. Action Interface

 
 The action interface determines how an LLM-based agent’s decisions are grounded into executable operations within the game environment.
Unlike generic LLM outputs that produce unconstrained text, games require actions that conform to specific control formats. Accordingly, action interfaces are categorized by the type of action required by games: high-level actions represent semantic or logical operations (e.g., “open the door”); low-level actions specify concrete control signals such as keystrokes or mouse movements; and programmatic actions output structured commands or API calls that the environment can directly execute.

 
 
 High-Level Actions. In games where actions are expressed as discrete choices ( Hu et al., 2025b ) , the generation problem can be reformulated as a selection task. In this case, the model can simply select one of the provided options as the action. In parser-based environments, such as text adventure games or interactive narratives ( Microsoft Research, 2019 ; Hausknecht et al., 2020 ) , the LLM must generate a command that follows specific syntax, such as “open the door” or “pick up the sword”. Outputs that deviate from the expected syntax are treated as invalid and ignored. Therefore, the core challenge is to ensure that output actions are admissible. Recent work has introduced correction mechanisms, such as mapping generated phrases to the closest permissible action ( Huang et al., 2022 ) . An alternative is constrained decoding: instead of unconstrained token-by-token decoding, it computes the joint likelihood of each valid action sequence using the chain rule, and then normalize across the entire action set ( Carta et al., 2023 ) . However, such token-level probabilities penalize longer commands disproportionately, leading to systematic bias against valid but longer actions. To mitigate this problem, TWOSOME ( Tan et al., 2024b ) introduces length normalization by scaling log-likelihoods with the action’s token count, thereby balancing the probability distribution over admissible actions. Other game agents refine high-level action selection with auxiliary knowledge or value signals: KnowAgent constrains executable-action planning with an explicit action knowledge base to reduce hallucination ( Zhu et al., 2025 ) , while reinforced advantage feedback (ReAd) trains a critic that estimates per-agent advantage values so agents can reject low-value actions without costly verification in the Overcooked-AI ( Zhang et al., 2025a ) .

 
 
 Low-Level Actions. Low-level actions operate at the control layer, such as keystrokes, mouse movements, joystick inputs, and are executed at each timestep. A low-level controller (policy) is responsible for translating a high-level action from the LLM into a sequence of control signals. One approach is heuristic planning ( Agashe et al., 2025 ; Liu et al., 2024a ; Park et al., 2023 ) : given an intent such as “chop a tree,” the system invokes a path planner (e.g., A ∗ ) to locate the nearest tree and then issues the necessary movement and interaction commands. Another approach is to learn a low-level controller (policy) that generates the required action sequences to realize the LLM’s high-level decisions. Such policies can be trained either through imitation learning from expert demonstrations or through reinforcement learning with environment feedback, often aided by goal-conditioned rewards or semantic similarity between goals and observed transitions ( Liu et al., 2024b ) .

 
 
 Code-based Actions. Code-based actions express agent decisions as structured code or API calls that can be executed directly in the environment ( Wang et al., 2024a ; Tan et al., 2024a ) . Their structured nature provides explicit semantics and eliminates ambiguity, allowing complex operations to be specified with precision (e.g., bot.equip(sword); through a modding API ( PrismarineJS, 2013 ) or key_press("M") at the system level). A further advantage is verifiability: code outputs can be parsed and checked before execution, and compilers or interpreters supply syntax feedback that enables automatic detection and correction of invalid commands ( Wang et al., 2024a ) . In addition, programmatic actions support reusability by enabling agents to maintain a library of high-level primitives that encapsulate recurring skills. These functions can be flexibly composed, reducing redundant low-level generation and facilitating scalable, compositional behavior ( Tan et al., 2024a ) . Recent work makes code-based grounding more learnable: LearnAct lets an agent analyze failed attempts and iteratively create and revise executable Python functions that expand its action space ( Zhao et al., 2024b ) . CoPiC emits multiple planning programs and trains a domain-adaptive selector to pick the plan best aligned with long-term rewards on ALFWorld, NetHack, and StarCraft II unit building ( Tian et al., 2026 ) . MaestroMotif has an LLM specify per-skill reward functions and generate code that trains and composes RL-learned skills in the NetHack ( Klissarov et al., 2025 ) . Table 4 lists representative LLMGAs, categorized by their perception and action interfaces.

 
 
 Table 4. Summary of representative LLMGAs in terms of perception action interfaces. 
 
 
 
 Agent | 
 Game | 
 Perception Interface | 
 Action Interface | 
 | 

 
 ReAct ( Yao et al., 2023 ) | 
 ALFWorld, etc. | 
 Textual input | 
 High-level action | 
 | 

 
 SwiftSAGE ( Lin et al., 2024 ) | 
 ScienceWorld | 
 Textual input | 
 High-level action | 
 | 

 
 Cradle ( Tan et al., 2024a ) | 
 RDR2 | 
 Visual input (VLM) | 
 Low-level action (via keyboard–mouse control APIs) | 
 | 

 
 Xu et al. ( Xu et al., 2024 ) | 
 Werewolf | 
 Textual input | 
 High-level action | 
 | 

 
 PokéLLMon ( Hu et al., 2024 ) | 
 Pokémon Battles | 
 Symbolic input | 
 High-level action | 
 | 

 
 TextStarCraft ( Ma et al., 2024 ) | 
 StarCraft II | 
 Symbolic input | 
 Low-level action (rule-based controller) | 
 | 

 
 ChessGPT ( Feng et al., 2024a ) | 
 Chess | 
 Symbolic input | 
 High-level action | 
 | 

 
 SuspicionAgent ( Guo et al., 2024 ) | 
 Leduc Hold’em | 
 Symbolic input | 
 High-level action | 
 | 

 
 ProAgent ( Zhang et al., 2024a ) | 
 Overcooked-AI | 
 Symbolic input | 
 Low-level action (via path search + API calls) | 
 | 

 
 TWOSOME ( Tan et al., 2024b ) | 
 Overcooked-AI | 
 Symbolic input | 
 High-level action (admissible action generation) | 
 | 

 
 Voyager ( Wang et al., 2024a ) | 
 Minecraft | 
 Symbolic input | 
 Code-based action (via Mineflayer code execution) | 
 | 

 
 GTIM ( Zhu et al., 2023 ) | 
 Minecraft | 
 Symbolic input | 
 Low-level action (via API calls) | 
 | 

 
 JARVIS-1 ( Wang et al., 2025h ) | 
 Minecraft | 
 Visual and symbolic input | 
 Low-level action (via controller and API calls) | 
 | 

 
 CoELA ( Zhang et al., 2024d ) | 
 TDW-T WAH | 
 Visual input (object detector) | 
 Low-level action (via rule-based controller) | 
 | 

 
 GenerativeAgents ( Park et al., 2023 ) | 
 Small Village | 
 Textual input | 
 High-level actions | 
 | 

 
 ZeroShotPlanner ( Huang et al., 2022 ) | 
 VirtualHome | 
 Symbolic input | 
 High-level actions (semantic translation) | 
 | 

 
 ELLM ( Du et al., 2023 ) | 
 Crafter | 
 Visual input (visual encoder) | 
 Low-level action (RL-based controller) | 
 | 

 
 STEVE ( Zhao et al., 2024c ) | 
 Minecraft | 
 Visual input (perception module) | 
 Code-based action (skill database) | 
 | 

 
 LearnAct ( Zhao et al., 2024b ) | 
 ALFWorld | 
 Textual input | 
 Code-based action (learned functions) | 
 | 

 
 CoPiC ( Tian et al., 2026 ) | 
 ALFWorld, NetHack, StarCraft II | 
 Textual/symbolic input | 
 Code-based action (program selection) | 
 | 

 
 KnowAgent ( Zhu et al., 2025 ) | 
 ALFWorld | 
 Textual input | 
 High-level action (knowledge-constrained) | 
 | 

 
 MaestroMotif ( Klissarov et al., 2025 ) | 
 NetHack | 
 Symbolic input | 
 Code-based skill composition | 
 | 

 
 ActiveVOO ( Liu et al., 2025b ) | 
 ALFWorld | 
 Visual input (active sensing) | 
 High-level action | 
 | 

 
 ReAd ( Zhang et al., 2025a ) | 
 Overcooked-AI | 
 Symbolic input | 
 High-level action (advantage feedback) | 
 | 

 
 
 
 
 

## 7. Multi-LLMGA Framework

 
 In this section, we extend the single agent framework to multi-agent settings. Designing a multi-agent system in games is different from generic multi-agent systems because games impose unique constraints: game environments impose realistic constraints on information sharing: observations are partially observable and distributed across agents, and communication channels are often bandwidth-limited ( Zhang et al., 2024d ) .

 
 
 At the agent level, we examine how agents exchange information and integrate it into their decision-making. Communication protocols specify what messages are generated (e.g., observations, beliefs, or intentions) and how they are interpreted by receivers. At the organization level, we study three aspects: the topology of connections that shape communication flow, the allocation of tasks and roles that governs functional division of labor, and the mechanisms for ensuring scalability and robustness as groups expand. Figure 8 presents the structure of this section of different components within the multi-LLMGA system.

 
 
 Figure 8. Overview of the multi-LLMGA framework. 
 
 

### 7.1. Communication Protocol

 
 In game and simulation environments, communication is likely constrained by partial observability, limited bandwidth, and asynchronous execution ( Zhang et al., 2024d ) , which makes communication protocol design crucial for coordination. A communication protocol defines the rules that regulate peer-to-peer information exchange at the agent level, which specifies what message the sender should share, and how it is integrated by the receiver.

 
 
 Message Generation. Senders determine what type of information is worth exchanging, which can be broadly categorized into three classes: The first is observation , referring to the raw and local signals each agent perceives from the environment. Observations are typically partial, such as perceiving only a limited visual field in the environment ( Zhang et al., 2024d ) , sharing observations allows teammates to directly expand each other’s perceptual fields. Since raw perceptual input is often redundant or low-value, practical systems ( Zhang et al., 2024d ) apply summarization to compress observations into compact, salient statements. The second is belief , which represents an agent’s internal inference or probability distribution over the hidden state of the world, based on its own observations and prior knowledge ( Zhang et al., 2024a ; Agashe et al., 2025 ) . Compared to raw observations, beliefs provide higher-level interpretations. For example, an agent may observe scattered leaves and tree trunks, and infer that the environment contains sufficient wood resources nearby. The third is intention , where agents communicate their planned actions or subgoals. Intention propagation is especially important in tasks that require complementary execution to reduce redundant effort (e.g., multiple agents pursuing the same subtask) and prevent conflicts (e.g., two agents competing for the same resource) ( Agashe et al., 2025 ; Zhang et al., 2024d ) . In addition, when there is no communication mechanism/channel available, agents need to infer collaborators’ hidden intentions based on their actions observed. In adversarial games, senders also craft messages strategically rather than honestly: a study of Among Us finds impostor agents rely on equivocation rather than outright lies ( Milkowski and Weninger, 2026 ) , GPT-4o agents can out-deceive humans in Mafia ( Kao et al., 2025 ) , and CoMet generates metaphor-based covert messages for the Undercover and Adversarial Taboo games ( Xu and Zhong, 2025 ) . Persuasion and negotiation are likewise produced deliberately, as in Richelieu’s goal-oriented negotiation grounded in social reasoning for Diplomacy ( Guan et al., 2024 ) and a Werewolf study of whether agents can generate opinion-leading speech ( Du and Zhang, 2024 ) . To keep large-scale exchange efficient, EcoLANG induces a compact agent communication language ( Mou et al., 2025 ) .

 
 
 Message Interpretation. Once communication takes place, agents need to integrate the exchanged information into their memory and ongoing decision process. In general, received messages can be directly adopted to guide actions. However, inconsistencies may arise when the new information conflicts with an agent’s existing internal state. To address this, agents must reconcile external messages and internal models. For instance, ProAgent ( Zhang et al., 2024a ) infers the belief of a partner through the reasoning of theory of mind and subsequently corrects its estimate when the partner’s observed actions reveal mismatches. ReConcile ( Chen et al., 2024b ) provides a debate-based approach by engaging agents in multi-round discussions, where they attempt to convince each other with corrective explanations and aggregate responses through confidence-weighted voting to reach consensus. ECON ( Yi et al., ) models this reconciliation as a Bayesian game, where agents treat each other’s beliefs and intentions as uncertain types and update them until they converge on a joint profile that all parties can consistently follow. BEACOF likewise frames collaboration as a dynamic game of incomplete information, refining probabilistic beliefs about peers’ capabilities until they reach an approximate perfect Bayesian equilibrium ( Fang et al., 2026 ) , and CSP4SDG interprets dialogue and game events in Avalon, Mafia, and Werewolf as evidence in a probabilistic constraint-satisfaction problem whose information-gain-weighted scores yield interpretable posterior beliefs over hidden roles ( Xu et al., 2026 ) .

 
 
 

### 7.2. Organizational Structure

 
 Organizational structure defines how agents are arranged and coordinated within a multi-agent system, including the topology of their connections, the allocation of tasks and roles, and the mechanisms that ensure scalability and stability as the population grows.

 
 
 Organizational Topology. Organizational topology refers to the structural constraints that determine how decisions flow, how agents connect for communication, and where authority over world state resides. Rather than a free design choice, topology is an architectural constraint that shapes the trade-off between scalability, robustness, and latency ( Qian et al., 2025a ) .

 
 
 Centralized organization rely on a single planner or coordinator to aggregate information and allocate subtasks. This design ensures strong consistency and efficiency but creates bottlenecks and single points of failure, which limit scalability. For example, MindAgent ( Gong et al., 2024b ) adopts a single foundation model as the central dispatcher that issues the step-by-step commands to all agents. Decentralized organization remove central authority and let agents act based on local observations and peer communication. Such topology is robust and can avoid global bottlenecks, but can suffer from coordination conflicts and redundant actions. CoELA ( Zhang et al., 2024d ) follows this paradigm, framing cooperation as decentralized planning under costly communication channels. To balance coherence with local flexibility, hierarchical organizations introduce multiple layers of control, with higher-level agents assigning goals or subtasks and lower-level agents refining them layer-by-layer. HAS ( Zhao et al., 2024d ) exemplifies a three-tier hierarchy: a top-level manager sets global plans, intermediate conductors translate and distribute these plans, and bottom-level action agents execute concrete steps. Similarly, S-Agents ( Chen et al., 2024a ) use a tree structure where a root node provides coordination and leaf nodes carry out subtasks. Partitioned or sharded systems divide persistent environments into regions, each governed by local authority with cross-shard coordination handled by bridging protocols. This design enables scalability and fault tolerance, but weakens global consistency. Project Sid ( AL et al., 2024 ) illustrates in a large-scale setting: thousands of Minecraft agents self-organize into civilizations where division of labor and institutions emerge, showing that centralized control is infeasible at such scale. Game-focused frameworks add explicit structure on top of these topologies: HIMA organizes a society of specialized imitation-learning agents, each mastering a distinct StarCraft II tactic, under a strategic-planner meta-controller that synthesizes their proposals into coordinated actions ( Ahn et al., 2025 ) , and prompt-based organizational structures with designated leadership are imposed on embodied teams and iteratively refined through a criticize–reflect process to raise efficiency while lowering communication cost ( Guo et al., 2025 ) .

 
 
 Task Role Allocation. Task and role allocation determines how subtasks are mapped to agents, shaping both efficiency and adaptability in multi-agent system. Allocation specifies the functional division of labor within the organizational topology. Three patterns are commonly observed: prefixed, dynamic, and emergent.

 
 
 Prefixed allocation specifies roles or tasks in advance, often through a central planner or a leader. This ensures clear division of labor and prevents conflicts, making it reliable for structured environments but rigid under open-ended or rapidly changing conditions. MindAgent ( Gong et al., 2024b ) follows this approach: a single foundation model centrally dispatches per-step actions for all agents, directly specifying each agent’s next move. Similarly, S-Agents ( Chen et al., 2024a ) predefine a root–leaf hierarchy, where the root serves as coordinator and leaves as executors, though the specific subtasks are still assigned dynamically during execution. Dynamic allocation allows agents to determine their roles during execution, with assignments decided in real time by monitoring the environment or coordinating with peers. This increases adaptability and robustness but may produce redundancy when multiple agents converge on the same role. Overcooked-AI ( Carroll et al., 2019 ) illustrates this challenge, as frequent task changes require agents to split and reassign responsibilities on the fly. CoELA ( Zhang et al., 2024d ) provides another example, where decentralized agents negotiate via natural language under costly communication channels to decide which subtasks to pursue. HAS ( Zhao et al., 2024d ) also falls into this category: while roles such as manager and conductors are predefined, the system dynamically reorganizes action groups and reallocates responsibilities as tasks evolve. Emergent allocation does not predefine the set of roles but lets them arise through repeated interaction.

 
 
 At scale, Project Sid ( AL et al., 2024 ) demonstrates how thousands of Minecraft agents spontaneously differentiate into specialized professions such as farmers, miners, builders, and traders, stabilizing cooperation without central control. This diversification arises from social awareness, where agents adjust goals in response to others’ activities, thereby reducing redundancy and enabling stable specialization. For dynamic role coordination, COPPER assigns credit to individual agents via a shared reflector tuned with counterfactual rewards, producing role-personalized reflections that improve collaboration on tasks including chess ( Bo et al., 2024 ) .

 
 
 Scalability Robustness. Scaling multi-agent systems beyond small groups remains challenging. Early studies such as Generative Agents typically support only dozens of agents, since agents execute cognition through a sequential pipeline with a single thread. This serialized design becomes the bottleneck for scaling ( Park et al., 2023 ) . Project Sid addresses the per-agent bottleneck with the PIANO architecture, which runs six modules in parallel to update the agent state at different time scales. To prevent incoherence between simultaneous outputs, a cognitive controller ( Kaiya et al., 2023 ) selects an option from the candidate outputs of concurrent modules and transmits this decision to other modules for execution.

 
 
 During the emergence of roles, certain factors are critical for ensuring organizational stability. Project Sid ( AL et al., 2024 ) demonstrates that social awareness plays a critical role in sustaining division of labor: when agents observe many of their peers performing one task, they are more likely to select a different one. Through memory and repeated behavior, these roles become reinforced, allowing agents to form stable identities such as “farmer” or “miner” and yielding a more persistent specialization structure. In social simulation experiments, Artificial Leviathan ( Dai et al., 2024 ) demonstrate that memory depth is the key factor for the emergence of a commonwealth (i.e., the rise of a sovereign), under which social disorder is significantly reduced. This suggests that memory acts as a stabilizing mechanism by turning short-term interactions into long-term understanding of agents’ relative strengths and weaknesses, thereby forming group consensus. Such coordination remains brittle at the level of collective behavior: an evaluation in the Melting Pot Commons Harvest game shows that LLM-augmented agents display a propensity for cooperation yet still fail to collaborate effectively under social dilemmas ( Mosquera et al., 2026 ) .

 
 
 
 

## 8. Gameplay Taxonomy for LLMGA Design

 
 The design of game agents is inseparable from the environments in which they operate: different genres foreground different capabilities, from fast perception–action cycles in action games to long-horizon planning in strategy games. A taxonomy that connects the properties of games with the demands they impose on agents is therefore valuable for this field. Here we adopt a challenge-centered game taxonomy: for each major category, we highlight the design challenge that most strongly drives LLMGA design. The genre axis itself draws on established categorizations, combining top-level groupings from SteamDB ( SteamDB, 2025 ) with the gameplay-oriented classification of Lee et al. ( Lee et al., 2014 ) . To make the taxonomy more coherent to covered studies, we exclude narrower genres such as driving/racing or fighting, and instead introduce sandbox as a category to capture open-ended and emergent play, with Minecraft as the canonical example.

 
 
 Building on this taxonomy, we sketch how different game genres map into distinct design challenges. Action games require low-latency response, where agents are challenged to reconcile the slow deliberation of language models with the frame-level demands of real-time play. Adventure games highlight stateful world modeling, where progress depends on maintaining coherent memories of evolving environments, quests, and object dependencies. Role-playing games raise the issue of role fidelity, in which agents are expected to sustain consistent personas and align dialogue and actions with character identity. Strategy games emphasize opponent-aware planning, where the key difficulty lies in anticipating and adjusting to adversaries’ potential intentions under imperfect information. Simulation games emphasize dynamics fidelity, evaluating whether agents can exhibit behaviors that remain faithful to the governing simulation dynamics. Finally, sandbox games expose the challenge of open-ended goal progression, where agents are tasked with generating their own objectives, decomposing them hierarchically, and accumulating reusable skills to sustain long-term play.

 
 

### 8.1. Action Games: Low-Latency Response

 
 Figure 9. Action games require agents to respond with low latency and execute precise low-level control. 
 
 
 As shown in Figure 9 , action games are characterized by real-time, time-critical interaction, where success hinges on executing precise movements such as aiming, dodging, or chaining combos within narrow temporal windows. This creates a fundamental demand for low-latency response, and the design challenge is therefore to reconcile the reasoning strengths of LLMs with the immediacy required by real-time gameplay.

 
 
 Environments. 
Atari 2600 games in the Arcade Learning Environment ( , 2013 ) provide a canonical benchmark for reflexive control, where agents map raw pixel observations to joystick inputs at 60 Hz. Procgen ( Cobbe et al., 2020 ) extends this setup with procedurally generated levels, requiring agents to generalize their responses across unseen layouts rather than memorizing fixed patterns.
Moving into 3D, ViZDoom ( Kempka et al., 2016 ) present first-person 3D environments where perception is partial and high-dimensional, requiring agents to aim, strafe, and dodge in real time.
Fighting games such as Street Fighter III ( OpenGenerativeAI, 2024 ) further sharpen the requirement for low-latency response: the timing of counters and combos is so precise that even minimal decision delays can flip the outcome of an exchange.

 
 
 Methods. Across action game environments, a consistent finding is that LLMs alone cannot sustain frame-level decision speed. Benchmarking across diverse video games including action games such as Super Mario and Street Fighter III shows that incorporating visual inputs often degrades rather than improves gameplay performance, partly because the additional processing overhead exacerbates inference latency ( Park et al., 2026 ) . Further latency-sensitive evaluation on Street Fighter demonstrates that achieving competent play requires explicitly trading off reasoning quality for faster inference ( Kang et al., ) .

 
 
 To mitigate this bottleneck, researchers have adopted hybrid designs. One line of work delegates reflexive control to low-level policies trained through reinforcement or imitation learning, while reserving the LLM for high-level reasoning and strategy, as illustrated by two-tier agent systems in Overcooked ( Liu et al., 2024a ) .
Empirical studies further show that latency-sensitive environments such as Street Fighter expose a fundamental trade-off between reasoning quality and decision speed: deeper reasoning produces stronger local decisions but increases inference latency to the point of losing more frequently, while shallower reasoning improves responsiveness and overall win rates ( Kang et al., ) . Action-game environments also serve as testbeds beyond latency: PoE-World learns a programmatic world model from few observations and embeds it into a model-based planner for stochastic Atari games such as Pong and Montezuma’s Revenge ( Piriyakulkij et al., 2025 ) .

 
 
 

### 8.2. Adventure Games: Stateful World Modeling

 
 Figure 10. Adventure games typically involve exploration, object collection, and quest completion. 
 
 
 As shown in Figure 10 , adventure games are defined by partial observability and long-horizon quests: progress depends on remembering what has been explored, which preconditions of puzzles or storylines remain unsatisfied, and understanding how objects, actions, and the rules of the world interact. For LLMGAs, this creates a fundamental demand: they should be able to record, update, and retrieve both the evolving environment state and the underlying knowledge of how these elements can be used or combined. Without such modeling, agents lose track of progress, repeat past actions, or fail to connect prerequisites with goals. Empirically, GPT-3.5 struggles to construct coherent maps in partially known text-adventure environments, and state-prediction benchmarks indicate that even stronger LLMs are unreliable as implicit world simulators ( Hausknecht et al., 2020 ) .

 
 
 Environments. Adventure game benchmarks such as TextWorld ( Côté et al., 2019 ) , Jericho ( Hausknecht et al., 2020 ) , ALFWorld ( Shridhar et al., 2021 ) , and ScienceWorld ( Wang et al., 2022 ) provide text-based environments in which players interact with the world through natural language, exploring rooms, collecting objects, and completing quests of varying complexity.
For instance, TextWorld procedurally generates synthetic quests by varying the number of rooms, objects, and goals ( Microsoft Research, 2019 ) . Jericho includes 56 human-authored classics such as the Zork series ( Infocom, 1980 ; Infocom, 1982 ) and Hitchhiker’s Guide to the Galaxy ( BBC, ) . ALFWorld aligns to the embodied ALFRED benchmark ( Shridhar et al., 2020 ) , requiring agents to follow household instructions. ScienceWorld ( Wang et al., 2022 ) simulates primary-school science curricula, highlighting basic knowledge from physics and chemistry for doing experiments.

 
 
 Methods. Recent work has gradually converged on the view that memory should operate as the backbone of world modeling in adventure settings. Early agents such as ReAct ( Yao et al., 2023 ) showed that simple interleaving of observations and actions is not sufficient, as the agent often fails to maintain an accurate view of the environment and becomes stuck. By incorporating reasoning, the agent can periodically summarize recent progress, ensuring that short-term records of explored locations, obtained items, and pending subgoals remain stable across steps. Reflexion ( Shinn et al., 2023 ) further demonstrates that writing self-critiques of failed attempts enables agents to extract insights from errors and avoid repeating them, thereby transforming episodic failures into persistent corrections of world knowledge. Subsequent agents, including Adapt ( Prasad et al., 2024 ) and SwiftSage ( Lin et al., 2024 ) further explicitly decompose quests into subgoals and track preconditions during execution. This keeps plans aligned with an evolving world state and enables coherent re-planning when branches fail. In a similar spirit, LPLH equips interactive-fiction agents with structured map building and feedback-driven experience analysis to track and reuse world state more like human players ( Zhang and Long, 2025 ) . KWM ( Qiao et al., 2024 ) leverages successful trajectories to learn a knowledge-augmented world model, allowing agents to internalize regularities of environment dynamics and use the world model to guide future planning. Beyond internalizing dynamics in memory, a recent line learns explicit, adaptive world models that predict future states: CoEx co-evolves a neurosymbolic world model alongside exploration so it does not go stale on ALFWorld and Jericho ( Kim and Hwang, 2025 ) , DreamPhase plans by rolling out a latent world model offline with uncertainty-aware filtering ( Hamidi et al., 2026 ) , and WorMI and TMoW retrieve and compose domain-specific world models at test time to adapt to new environments ( Yoo et al., 2025 ; Jang et al., 2026 ) . AriGraph ( Anokhin et al., 2025 ) encodes episodic experiences alongside semantic facts in a knowledge-graph memory, yielding a retrievable and interpretable representation of the game environment. Such structured memory can also steer search: MC-DML couples Monte Carlo tree search with in-trial and cross-trial memory to dynamically reweight action evaluations in the Jericho benchmark ( Shi et al., 2025 ) . At a larger scale, Cradle ( Tan et al., 2024a ) demonstrates the same principle in the visually rich adventure setting of Red Dead Redemption II, where the key difficulty lies in aligning perception with quest progress and narrative state. By maintaining memory as an explicit record of explored context and completed steps, Cradle enables the agent to keep exploration and story advancement coherent across long-horizon play, which stabilizes behavior in sprawling 3D environments. Even with such mechanisms, world modeling stays hard in open-ended roguelikes: in NetHack, the zero-shot skill-based agent NetPlay tracks interaction history yet still struggles under sparse feedback and ambiguous goals ( Jeurissen et al., 2024 ) .

 
 
 

### 8.3. Role-Playing Games: Role Fidelity

 
 As illustrated in Figure 11 , role-playing games (RPGs) require players to assume pre-defined characters with distinct abilities, knowledge, experiences, and objectives. Although RPGs may also incorporate elements of action or adventure, our focus here is on a common characteristic that underpins this genre: role fidelity. Role fidelity means that agents should internalize their assigned role and generate dialogue and actions that remain consistent with the character’s identity and capabilities. Failure to do so causes agents to lose consistency in speech and action, or even contradict their assigned role, undermining both immersion and gameplay effectiveness.

 
 
 Environments. Social deduction board games provide natural testbeds for role fidelity. In Werewolf, each player receives a hidden role such as seer, guard, or werewolf, and must preserve secrecy while engaging in persuasion, deception, and coordinated voting ( Xu et al., 2023 ; Xu et al., 2024 ) . Similarly, Avalon assigns asymmetric roles with private knowledge (e.g., Merlin knowing the bad team), requiring agents to participate in multi-round discussions without revealing confidential information while still influencing team decisions ( Light et al., 2023 ) . Negotiation games like Diplomacy, where each player embodies a nation with its own objectives ( (FAIR)† et al., 2022 ) , and scripted murder-mystery games such as Jubensha ( Wu et al., 2024a ) , reinforce the same demand: agents must consistently inhabit a pre-defined persona and objectives, balancing what to disclose and what to withhold across multiple turns to preserve immersion and effectiveness.

 
 
 Figure 11. Role-playing games require agents to internalize and consistently enact pre-defined roles with distinct abilities, knowledge, and objectives. 
 
 
 Classic RPGs also emphasize role fidelity through long-horizon progression. For example, in Pokémon Red, the trainer role requires remembering the current storyline position, the Pokémon owned, the items carried, and the towns and paths visited. PokéAgent introduces exploration tasks to test whether agents can remain coherent as trainers throughout the gameplay ( Karten et al., 2025a ) . Beyond the main character, role fidelity is even more critical for non-player characters (NPCs), which must sustain consistent personas across repeated interactions and emergent narratives, as exemplified by recent studies evaluating personality fidelity in role-playing ( Wang et al., 2024e ) .

 
 
 Methods. The simplest approach adds the role card directly into the prompt, listing traits and goals as initial memory ( Park et al., 2023 ) . While this establishes in-character openings, it quickly breaks down over multi-turn play, i.e., the role drift problem. Empirical studies show that in Avalon, LLMs may reveal their secret identity or fail to sustain deception across rounds ( Light et al., 2023 ) . To mitigate such inconsistencies, approaches condition generation on explicit intentions or structured reasoning. For example, in Diplomacy, Cicero anchors dialogue in private strategic plans to ensure alignment between language and action ( (FAIR)† et al., 2022 ) . These methods improve local consistency but are not designed to preserve long-term role fidelity. More recent approaches target role fidelity directly by integrating role profiles as a persistent component of the memory system. RoleLLM ( Wang et al., 2024c ) introduces structured role memory that separates private belief states (e.g., hidden identities) from public discourse records, ensuring that agents regulate what to disclose versus conceal across turns. CharacterLLM ( Shao et al., 2023 ) , RoleLLM ( Wang et al., 2024c ) and CoSER ( Wang et al., 2025e ) adopt parametric adaptation, fine-tuning LLMs on curated role-play data to internalize persona traits and generate consistent style and objectives without continual reminders. These frameworks shift the focus from dialogue-level consistency to persistent memory management. Role fidelity is also exercised in open-ended narrative and social play: the AI-native game 1001 Nights casts the player opposite an LLM-powered King in co-creative storytelling, where the model sustains narrative coherence and story keywords are materialized as in-game equipment via image generation ( Sun et al., 2023 ) , while a multi-agent framework guides LLM agents through Avalon to study how they balance collaborative and confrontational behaviors and how this affects success rates ( Lan et al., 2024 ) .

 
 
 

### 8.4. Sandbox Games: Open-Ended Goal Progression

 
 Figure 12. Sandbox games are typically open-ended, requiring agents to decide their own goals while freely exploring and building to support emergent progression. 
 
 
 Sandbox games are characterized by open-ended environments and emergent play rather than fixed quests or roles. As shown in Figure 12 , players can freely explore, collect resources, and set their own objectives from survival to large-scale construction. For LLMGAs, this creates unique demands for both generating meaningful goals in the absence of external instructions and decomposing goals into actionable plans. Without such mechanisms, agents either become stuck in aimless wandering or fail to coordinate long-horizon plans into coherent progression.

 
 
 Environments. Minecraft and Crafter are two sandbox games that have been widely studied for game agents. Minecraft ( Mojang Studios, ) is a 3D sandbox game that offer players the great freedom to traverse a world made up of blocky, pixelated landscapes, facilitated by the procedurally generated worlds. The resource-based crafting system enables players to transform collected materials into tools, build elaborate structures and complex machines. Built on Minecraft, MineDojo ( Fan et al., 2022 ) provides a large-scale research platform with thousands of open-ended tasks, multimodal data from community sources, and the MineCLIP reward model. Crafter ( Hafner, 2022 ) offers a lightweight 2D open-world environment with procedurally generated maps. It challenges players to manage their resources carefully to ensure sufficient water, food, and rest, while also defending against threats like zombies.

 
 
 Methods. In sandbox settings, agents need to first determine what goals to pursue before they can decide how to achieve them. Existing works can be divided into two complementary directions. The first direction emphasizes goal generation through intrinsically motivated exploration. With LLMs, agents can propose adaptive goals conditioned on their current state, skills, and environment for curriculum learning. Voyager ( Wang et al., 2024a ) exemplifies this idea by prompting an LLM to continually generate new objectives, building a self-directed curriculum and accumulating a library of reusable skills. OMNI ( Zhang et al., 2024e ) utilizes LLMs to determine interesting tasks for curriculum design, overcoming the previous challenge of quantifying "interestingness". ELLM ( Du et al., 2023 ) queries LLMs for next goals given an agent’s current context, and rewards agents for accomplishing those suggestions in the sparse-reward setting; SPRING ( Wu et al., 2024d ) uses LLMs to summarize useful knowledge from the Crafter paper ( Hafner, 2022 ) and progressively prompts the LLM to generate next action.

 
 
 The second direction is hierarchical planning for task execution. Sandbox objectives such as constructing tools or building structures require agents to gather dispersed resources and follow multi-step recipes with strict dependencies. DEPS ( Wang et al., 2023c ) introduces plan correction: the LLM generates candidate subgoals, monitors execution outcomes, and self-explains failures in order to iteratively repair its plans, while leaving the final action execution to goal-conditioned controllers. Subsequent work emphasized making planning more reusable. JARVIS-1 ( Wang et al., 2025h ) extend this idea by integrating multimodal perception and memory, grounding subgoal generation in visual context. Later work such as Plan4MC ( Yuan et al., 2023 ) and RL-GPT ( Liu et al., 2024b ) extend hierarchical planning by coupling high-level LLM planners with low-level controllers trained via reinforcement learning. Finally, multi-agent frameworks such as HAS ( Zhao et al., 2024d ) and S-Agents ( Chen et al., 2024a ) extend hierarchical planning to cooperative settings, dispatching subgoals across multiple agents to parallelize progress on complex objectives. A third, increasingly direction grounds open-ended progression in learned world models. WALL-E aligns an LLM world model with the environment by extracting symbolic action rules and knowledge graphs into executable code, letting an RL-free model-predictive-control planner substantially raise success rates on the Minecraft-like Mars world ( Zhou et al., 2025 ) . ADAM autonomously learns a causal world graph from scratch, enabling interpretable lifelong task solving even in modified Minecraft worlds where prior crafting knowledge is unavailable ( Yu and Lu, 2025 ) . DLLM injects LLM-proposed subgoal hints into a model-based agent’s imagined rollouts, rewarding hint-aligned transitions to improve exploration in the sparse-reward Crafter and Minecraft environments ( Liu et al., 2024d ) .

 
 
 

### 8.5. Strategy Games: Opponent-Aware Planning

 
 Strategy games span a spectrum of complexity, from turn-based, deterministic, perfect information game to real-time, stochastic imperfect information games. As shown in Figure 13 , a common requirement is opponent-aware planning: agents need to infer opponents’ possible intentions and conduct multi-step planning conditioned on these possibilities, as shown in Figure 13 .

 
 
 Environments. Board games like Chess and Go are fully observable, where agents need to search vast move trees while anticipating optimal counter-moves ( Feng et al., 2024a ; Toshniwal et al., 2022 ; Li et al., 2023b ) . Pokémon battles ( Hu et al., 2024 ) add uncertainty: players select moves or switches without knowing the opponent’s choice, and success depends on exploiting type matchups.

 
 
 Poker, exemplified by Texas Hold’em, deals each player two private hole cards, followed by betting rounds as community cards are revealed. The winning strategy
is not simply holding the best hand, but managing information asymmetry through bluffing, pot control, and reasoning about what cards the opponent may
have ( Zhuang et al., 2025 ) . StarCraft II is a real-time strategy game where players collect resources, expand bases, build armies, and fight under the fog of war. Winning requires players to infer the opponent’s strategy from limited scouting, adapt build orders and timing attacks accordingly, and still control units precisely in battle. For agents, the challenge is therefore twofold: modeling and planning against an adaptive opponent as in other strategy games, and at the same time coordinating across macro, tactical, and micro levels under strict temporal constraints ( Ma et al., 2024 ) .

 
 
 Figure 13. Strategy games emphasize strategic reasoning, requiring agents to anticipate opponent responses and plan over multiple future steps. 
 
 
 Methods. In perfect-information games such as Chess and Go, opponent-aware planning reduces to deterministic search over long move sequences. ChessGPT ( Feng et al., 2024a ) demonstrates that training on textual game corpora allows LLMs to evaluate positions and propose continuations, while blindfold-play studies ( Toshniwal et al., 2022 ; Li et al., 2023b ) reveal that models can implicitly reconstruct board states from move sequences, approximating the effect of explicit lookahead search. Building on this, MATE fine-tunes an LLM on a million chess positions annotated with expert strategy and tactic explanations, surpassing leading closed-source models at move selection ( Wang et al., 2025c ) . Explicit lookahead has also been layered on top of the model. Schultz et al. reach grandmaster-level chess by guiding Monte Carlo tree search externally or linearizing the game tree within the context window ( Schultz et al., 2025 ) . DiffuSearch instead replaces explicit search with discrete-diffusion implicit search over imagined future board states ( Ye et al., 2025 ) . In imperfect-information games, the challenge is reasoning over probability trees defined by partially observable states and uncertain opponent actions. Here, opponent modeling, often framed as theory-of-mind (ToM) thinking, is crucial. Suspicion-Agent ( Guo et al., 2024 ) shows that prompting LLMs for higher-order ToM in Leduc Hold’em leads to more aggressive raises and fewer passive calls, improving long-term chip gains. PokéLLMon ( Hu et al., 2024 ; Hu et al., 2025b ) shows that LLM agents are still vulnerable to human misdirection strategies exploiting their limited higher-order ToM. For instance, a player may bait the agent by sending out a seemingly weak Pokémon, then switch to an immune one just before the attack lands, causing the agent to waste its move. To strengthen opponent-aware play under such uncertainty, recent agents pair LLMs with lookahead and explicit opponent modeling. PokéChamp embeds an LLM in minimax tree search with model-based opponent prediction for Pokémon battles ( Karten et al., 2025b ) . Strategist couples high-level textual strategy generation with Monte Carlo tree search through self-play in the hidden-role games GOPS and Avalon ( Light et al., 2025 ) . Such gains remain bounded by the models’ inherent decision-making, as a systematic analysis across the dictator game, rock-paper-scissors, and a ring-network game finds persistent gaps from human rationality in forming preferences, refining beliefs, and acting optimally ( Fan et al., 2024 ) .

 
 
 

### 8.6. Simulation Games: Dynamics Fidelity

 
 As illustrated in Figure 14 , simulation games model the behavior of a system governed by dynamics rules such as social norms, economic mechanisms, ecological constraints, or physical laws. Rather than focusing on fixed objectives, these environments emphasize the faithful operation of the underlying simulation processes, i.e. , dynamics. We therefore characterize the core challenge as dynamics fidelity, referring to an agent’s ability to generate behaviors that drive the state transitions of the simulated system in a manner consistent with its governing dynamics.

 
 
 Figure 14. Simulation games aim to model or reproduce the behavior of a system and require agents to behave consistently with the underlying dynamics of the simulation. 
 
 
 Environments. 
Human simulation environments construct virtual societies for studying emergent social behavior.
Generative Agents ( Park et al., 2023 ) places 25 agents in a sandbox town with cognitive modules for everyday interaction.
Humanoid Agents ( Wang et al., 2023b ) extends this setting by incorporating physiological needs, emotions, and relationship closeness. Beyond human simulation, CivRealm ( Qi et al., 2024 ) is a Civilization-style simulation environment focusing on the macro-scale evolution of societies across historical eras. From a generative angle, Unbounded realizes an infinite character life-simulation game whose mechanics, narrative, and interactions are produced on the fly by a distilled LLM with consistent visual character generation ( Li et al., 2025a ) .

 
 
 Methods. For social simulation games, maintaining dynamics fidelity in simulation requires that agents behave in ways consistent with human or societal patterns rather than drifting into unrealistic behavior.
Generative Agents ( Park et al., 2023 ) achieved this by introducing cognitive architectures with memory, reflection, and planning. Its memory system scores experiences by recency, relevance, and importance, allowing salient events to be repeatedly recalled and consolidated, mirroring core patterns of human memory. Humanoid Agents ( Wang et al., 2023b ) further improved fidelity by embedding physiological needs, emotions, and relationship closeness into decision-making, leading agents to display more human-like variability. Similarly, a desire-driven agent selects activities to satisfy multi-dimensional intrinsic needs, producing coherent human-like daily routines in the Concordia simulator ( Wang et al., 2025f ) . At larger scales, Project Sid ( AL et al., 2024 ) constructed an agent society in a Minecraft-based world, inhabited by hundreds to thousands of agents who shared limited resources and interacted concurrently. Under conditions of scarcity and continual co-presence, the agents competed and cooperated, spontaneously developing specialized roles, adapting collective rules, and propagating cultural practices such as religion. Artificial Leviathan ( Dai et al., 2024 ) approaches fidelity through a survival sandbox in which agents, driven by psychological needs under resource pressure, choose among farming, trading, or robbing each day. Empirical results show that agents start in conflict but eventually form social contracts, authorize a sovereign, and transition to peaceful cooperation. Experiments further show that parameters like memory depth has a large impact on the speed and nature of social evolution. Such emergent dynamics also surface in game economies, where a generative agent-based model reproduces role specialization and realistic market dynamics in massively multiplayer games ( Xu et al., 2025a ) . However, a distinct concern is whether it is genuinely faithful to real humans rather than merely plausible. Validation against established human findings is encouraging: given heterogeneous personas and theory of mind, LLM agents reproduce human third-party punishment in public-goods games ( Cross et al., 2025 ) , match human choices in the behavioral-economics Trust Game ( Xie et al., 2024 ) , and recover the predictions of Homans’ social exchange theory ( Wang et al., 2025b ) . To close residual gaps, other work calibrates simulations to empirical data, aligning population-level decision dynamics to real social statistics ( Mi et al., 2025 ) or tuning agent personas until crowd behavior matches expert benchmarks ( Wang et al., 2025g ) . Diagnostic studies nonetheless caution that human-like socialization may not genuinely emerge in such open-ended agent societies ( Li et al., 2026 ) .

 
 
 
 

## 9. Discussion and Open Challenges

 
 This section synthesizes the discussions from previous sections and connects them to the two research questions: (Q1) how the core architectural components of LLMGAs are designed, and (Q2) how genre-specific requirements shape these architectural components. We also highlight open challenges and future research directions.

 
 

### 9.1. Memory System

 
 To answer Q1, we introduced memory systems as a core architectural component for storing, retrieving, and recalling past experiences in order to maintain behavioral consistency and inform decision-making. The memory system is divided into working memory and long-term memory, serving distinct yet interdependent functions (§ 4 ). Working memory stabilizes short-horizon decision-making under limited context by (i) extending the effective input span, (ii) compressing redundant information, and (iii) maintaining recent bindings and plans. These mechanisms mitigate short-term drift and prevent inconsistent actions ( Hu et al., 2025b ) . Long-term memory ensures continuity across episodes when organized into structured and retrievable forms such as chunks with metadata, key–value pairs, hierarchical trees, graphs, or skill libraries. The interaction between the two hinges on three functions: consolidation, which determines when transient traces are committed to durable storage; structuring, which organizes stored content for efficient access; and retrieval, which reactivates relevant information through metadata filtering, semantic search, or traversal of structured memories.

 
 
 An open challenge for current memory systems is to move beyond “storing more” toward developing a true “world-model” memory that consolidates fragmented experiences into a coherent mental model of the game world ( Johnson-Laird, 1983 ) . This limitation is particularly pronounced in adventure games, where long-horizon exploration and partial observability place strong demands on persistent world modeling (Q2). To obtain a world-model memory system and distinguish it from a mere database, three design principles can be essential: (i) Predictive dynamics: memory should not only replay past events but also predict what might happen next. In cognitive science, mental models are understood as internal simulations that help people anticipate outcomes and detect errors, rather than as static records ( Johnson-Laird, 1983 ) . (ii) Structural compositionality: experiences need to be stored in organized forms, such as schemas or graphs that link entities, relations, and precondition–effect rules, so that knowledge from different situations can be combined and reused. This idea aligns with schema and situation-model theories, which show that humans build integrated “who–what–where–when–why” representations to reason beyond literal experiences ( Zwaan and Radvansky, 1998 ) . (iii) Selective consolidation and forgetting: long-term memory should decide what to keep and what to discard. Instead of saving every detail, it should preserve experiences that are important for understanding or improving the current model of the world, while letting irrelevant or low-value details fade. Research on human memory shows that people tend to remember information that is useful or frequently encountered and forget what rarely matters ( Anderson and Schooler, 1991 ) .

 
 
 

### 9.2. Reasoning Mechanism

 
 Reasoning in LLMGAs is not merely about generating intermediate thoughts, but also ensuring that these thoughts lead to correct decisions (§ 5 ). To answer Q1, we categorize existing approaches into instruction-guided reasoning and fine-tuning–based reasoning enhancement. Instruction-guided reasoning highlights recurring challenges: reasoning should avoid error propagation and remain consistent across steps. In comparison, fine-tuning approaches improve reasoning abilities by grounding reasoning in experience and feedback. Despite these advances, a fundamental limitation remains: current approaches rely on narrow forms of feedback or numeric rewards. Multi-path reasoning improves robustness by exploring diverse reasoning trajectories, yet it provides no learning signal about which paths are preferable or why. Reflective reasoning enables self-correction across episodes but remains coarse-grained, offering post-hoc summaries rather than actionable, step-level feedback. Process Reward Models (PRMs) attempt to provide this supervision by assigning stepwise rewards, but rely heavily on costly human annotation or handcrafted heuristics, making feedback sparse, rigid, and poorly aligned with the linguistic nature of reasoning.

 
 
 The deeper challenge lies in the mismatch between the form of reinforcement and the medium of reasoning. Traditional reinforcement learning depends on numeric rewards, whereas reasoning in LLMs unfolds through language, where success, failure, and state changes appear as semantic cues. Humans, however, are capable of assigning credit even from weak or indirect feedback: they adjust their reasoning based on partial signals such as environmental changes, the outcome of intermediate goals, or the perceived coherence of an explanation. Cognitive studies on metacognition and error monitoring show that such internal evaluation enables people to refine reasoning continuously through semantic and contextual signals rather than explicit numeric reinforcement ( Yeung and Summerfield, 2012 ; Beran, 2012 ; Fleming and Dolan, 2012 ) . By virtue of their linguistic grounding, LLMs can transform textual feedback, environmental descriptions, and self-critiques into implicit reinforcement signals, generalizing traditional reward learning beyond numeric values and enabling reasoning to improve through understanding rather than scoring. As for Q2, across strategy games, opponent-aware planning poses a central reasoning challenge. In deterministic, perfect-information games, the main difficulty lies in evaluating long sequences of alternating moves to anticipate the opponent’s best actions. In imperfect-information games, agents must reason under uncertainty by forming and updating beliefs about hidden states and possible opponent actions, which often requires higher-order theory-of-mind reasoning to explicitly model the opponent’s intentions, predict their likely responses, and adjust strategies accordingly over multiple steps.

 
 
 

### 9.3. Perception-Action Interface

 
 From the perspective of Q1, the perception action interface grounds how agents see the environment and fulfill their decisions (§ 6 ).
Across the surveyed literature, perception interfaces tend to be most effective when they emphasize decision-relevant features, such as object states, affordances, and strategic cues, rather than exhaustively encoding raw observations.
However, increasing perceptual richness often comes at the cost of efficiency, as more complex representations incur higher preprocessing and inference overhead. A similar tension arises on the action side. Existing action interfaces reflect a recurring trade-off between effectiveness and efficiency: high-level actions simplify the decision space and stabilize reasoning but may limit control precision, while low-level controls enable fine-grained execution but require high-frequency generation.

 
 
 For Q2, the challenge of low-latency response is particularly pronounced in action games, where agents operate under strict real-time constraints while issuing low-level control signals at high frequency. In such environments, even strong strategic reasoning can fail if actions are generated too slowly.
These constraints expose a requirement for hierarchical action generation. Low-level action execution is typically lightweight and reactive, while higher-level strategic reasoning operates at a slower timescale.
A practical direction for future research is therefore to explicitly decouple strategic decision-making from reflexive control, allowing LLMs to generate abstract intents or subgoals, which are then executed by fast, specialized controllers.
Further opportunities include adaptive abstraction, where the granularity of perception and action is dynamically adjusted based on latency budgets and task demands.
Designing perception action interfaces that jointly optimize effectiveness and efficiency remains an open challenge for LLMGA research.

 
 
 

### 9.4. Multi-LLMGA System

 
 LLM-based multi-agent systems extend game environments from single-agent decision making to collective behavior, introducing new challenges such as communication bandwidth limits, and the need to preserve realistic interaction constraints (§ 7 ). To answer Q1, we analyzed these systems across two complementary levels. At the agent level, communication protocols determine what information agents exchange and how it is integrated under these constraints, while at the organization level, organizational structures govern decision flow (topology), guide division of labor (task allocation), and determine whether societies can scale and remain stable.

 
 
 For Q2, we are interested in how genre-specific requirements influence the multi-agent architectural design. Role-playing games such as Werewolf and Avalon center on agent-level communication: the core challenge is controlling what to disclose and what to withhold, placing the design emphasis on belief sharing and selective information exchange. Strategy games such as StarCraft II and Overcooked also foreground agent-level coordination, where the focus shifts to rapid intention propagation and conflict avoidance. In contrast, simulation and sandbox games operating at larger scales, such as those modeling entire virtual societies in Minecraft, shift the architectural focus from agent-level protocols to organization-level structures. As the number of agents grows, the primary challenge becomes designing topologies and role allocation mechanisms that sustain stable coordination without centralized control.

 
 
 Prior studies have demonstrated the potential of multi-agent systems in large-scale simulations, where agents exhibit emergent behaviors. However, current large-scale multi-agent simulations remain constrained by structural and methodological limitations. Many “emergent” phenomena, such as role differentiation, norm formation, or collective planning, are closely tied to task initialization and rule design. In practice, agents are often seeded with shared goals, cooperation-oriented prompts, or predefined role templates that guide subsequent division of labor and coordination patterns. Prior studies of multi-agent societies have shown that such structural priors are widespread, from small-scale social environments ( Park et al., 2023 ) to hierarchical and large-scale simulations ( Chen et al., 2024a ; Zhao et al., 2024d ) , where coordination often reflects the constraints of task setup rather than fully autonomous self-organization. Moreover, the lack of open and reproducible large-scale platforms further limits systematic evaluation, making it difficult to test under what specific conditions such collective dynamics genuinely arise.

 
 
 

### 9.5. Game Environments and Benchmarks

 
 Table 5 summarizes existing open-sourced benchmarks or environments for LLMGAs. Some widely used benchmarks ( e.g. , TextWorld ( Côté et al., 2019 ) , ALFWorld ( Shridhar et al., 2021 ) , ScienceWorld ( Wang et al., 2022 ) ) were primarily developed before the rise of LLMs. Their tasks are generated from templated rules and constrained by a limited set of admissible actions and shallow dynamics, which result in highly similar instantiated tasks and low interactive complexity. In ALFWorld, for example, tasks are constructed from household instruction templates over a fixed action set (e.g., pick up, open, put, heat), producing many near-duplicate instances that only substitute objects or receptacles ( Shridhar et al., 2021 ) .

 
 
 Table 5. Open-sourced Benchmark/Environments for LLMGAs 
 
 
 
 Genre | 
 Benchmark/Environment | 
 Game Content | 
 Player Mode | 
 Modality | 
 Code Link | 
 Date | 

 
 Action | 
 Overcooked-AI ( Carroll et al., 2019 ) | 
 Overcooked-like Game | 
 Multi | 
 Symbolic | 
 GitHub | 
 2019/12 | 

 
 LLM-Coordination ( Agashe et al., 2025 ) | 
 Overcooked-AI | 
 Multi | 
 Text | 
 GitHub | 
 2023/10 | 

 
 TextStarCraft ( Ma et al., 2024 ) | 
 StarCraft II | 
 Single | 
 Text | 
 GitHub | 
 2023/12 | 

 
 llm-colosseum ( OpenGenerativeAI, 2024 ) | 
 Street Fighter III | 
 Multi | 
 Vision | 
 GitHub | 
 2024/03 | 

 
 Adventure | 
 VirtualHome ( Puig et al., 2018 ) | 
 Household Tasks | 
 Single | 
 Mixed | 
 GitHub | 
 2018/06 | 

 
 TextWorld ( Côté et al., 2019 ) | 
 Text-based Games | 
 Single | 
 Text | 
 GitHub | 
 2018/07 | 

 
 Jericho ( Hausknecht et al., 2020 ) | 
 Interactive Fictions | 
 Single | 
 Text | 
 GitHub | 
 2019/09 | 

 
 ALFRED ( Shridhar et al., 2020 ) | 
 Household Tasks | 
 Single | 
 Mixed | 
 GitHub | 
 2020/03 | 

 
 ALFWorld ( Shridhar et al., 2021 ) | 
 Household Tasks | 
 Single | 
 Text | 
 GitHub | 
 2020/10 | 

 
 ScienceWorld ( Wang et al., 2022 ) | 
 Science Experiments | 
 Single | 
 Text | 
 GitHub | 
 2022/03 | 

 
 BabyAI-Text ( Carta et al., 2023 ) | 
 MiniGrid Tasks | 
 Single | 
 Text | 
 GitHub | 
 2023/02 | 

 
 Role-playing | 
 Cicero ( (FAIR)† et al., 2022 ) | 
 Diplomacy | 
 Multi | 
 Text | 
 Github | 
 2022/12 | 

 
 Generative Agents ( Park et al., 2023 ) | 
 Sims-like Game | 
 Multi | 
 Text | 
 GitHub | 
 2023/04 | 

 
 AgentSims ( Lin et al., 2023 ) | 
 Sims-like Game | 
 Multi | 
 Text | 
 GitHub | 
 2023/08 | 

 
 Xu et al. ( Xu et al., 2023 ) | 
 Werewolf | 
 Multi | 
 Text | 
 GitHub | 
 2023/09 | 

 
 AvalonBench ( Light et al., 2023 ) | 
 Avalon | 
 Multi | 
 Text | 
 GitHub | 
 2023/10 | 

 
 Humanoid Agents ( Wang et al., 2023b ) | 
 Sims-like Game | 
 Multi | 
 Mixed | 
 GitHub | 
 2023/10 | 

 
 clembench ( Chalamalasetti et al., 2023 ) | 
 Dialogue Games | 
 Multi | 
 Text | 
 GitHub | 
 2023/11 | 

 
 SOTOPIA ( Zhou et al., 2024 ) | 
 Social-interaction Scenarios | 
 Multi | 
 Text | 
 GitHub | 
 2023/10 | 

 
 PokéAgent Challenge ( Karten et al., 2025a ) | 
 Pokémon | 
 Single | 
 Text | 
 GitHub | 
 2025/07 | 

 
 RoleLLM ( Wang et al., 2024c ) | 
 Role-Playing Benchmark | 
 Single | 
 Text | 
 GitHub | 
 2023/10 | 

 
 CoSER ( Wang et al., 2025e ) | 
 Role-Playing Benchmark | 
 Multi | 
 Text | 
 GitHub | 
 2025/02 | 

 
 Strategy | 
 Cicero ( (FAIR)† et al., 2022 ) | 
 Diplomacy | 
 Multi | 
 Text | 
 Github | 
 2022/12 | 

 
 Xu et al. ( Xu et al., 2023 ) | 
 Werewolf | 
 Multi | 
 Text | 
 GitHub | 
 2023/09 | 

 
 AvalonBench ( Light et al., 2023 ) | 
 Avalon | 
 Multi | 
 Text | 
 GitHub | 
 2023/10 | 

 
 TextStarCraft ( Ma et al., 2024 ) | 
 StarCraft II | 
 Single | 
 Text | 
 GitHub | 
 2023/12 | 

 
 PokéLLMon ( Hu et al., 2025b ) | 
 Pokémon Battles | 
 Single | 
 Text | 
 GitHub | 
 2024/02 | 

 
 PokerBench ( Zhuang et al., 2025 ) | 
 Texas Hold’em | 
 Single | 
 Text | 
 GitHub | 
 2025/01 | 

 
 ChessGPT ( Feng et al., 2024a ) | 
 Chess | 
 Single | 
 Symbolic | 
 GitHub | 
 2024/03 | 

 
 CivRealm ( Qi et al., 2024 ) | 
 Civilization-like Game | 
 Single | 
 Symbolic | 
 GitHub | 
 2024/01 | 

 
 LMAct ( Ruoss et al., 2025 ) | 
 Chess, Atari, Tic-tac-toe | 
 Single | 
 Mixed | 
 GitHub | 
 2025/01 | 

 
 LLM-PySC2 ( Li et al., 2025c ) | 
 StarCraft II | 
 Multi | 
 Mixed | 
 GitHub | 
 2025/04 | 

 
 Full-press Diplomacy ( Duffy et al., 2025 ) | 
 Diplomacy | 
 Multi | 
 Text | 
 GitHub | 
 2025/08 | 

 
 Simulation | 
 MineDojo ( Fan et al., 2022 ) | 
 Minecraft | 
 Single | 
 Mixed | 
 GitHub | 
 2022/06 | 

 
 Generative Agents ( Park et al., 2023 ) | 
 Sims-like Game | 
 Multi | 
 Text | 
 GitHub | 
 2023/04 | 

 
 AgentSims ( Lin et al., 2023 ) | 
 Sims-like Game | 
 Multi | 
 Text | 
 GitHub | 
 2023/08 | 

 
 Humanoid Agents ( Wang et al., 2023b ) | 
 Sims-like Game | 
 Multi | 
 Mixed | 
 GitHub | 
 2023/10 | 

 
 CivRealm ( Qi et al., 2024 ) | 
 Civilization-like Game | 
 Single | 
 Symbolic | 
 GitHub | 
 2024/01 | 

 
 Sandbox | 
 Crafter ( Hafner, 2022 ) | 
 2D Survival Sandbox | 
 Single | 
 Vision | 
 GitHub | 
 2021/06 | 

 
 MineDojo ( Fan et al., 2022 ) | 
 Minecraft | 
 Single | 
 Mixed | 
 GitHub | 
 2022/06 | 

 
 Odyssey ( Liu et al., 2025a ) | 
 Minecraft | 
 Single | 
 Mixed | 
 GitHub | 
 2024/07 | 

 
 Mars ( Tang et al., 2024 ) | 
 Crafter | 
 Single | 
 Vision | 
 GitHub | 
 2024/10 | 

 
 Baba Is AI ( Cloos et al., 2024 ) | 
 Rule-manipulation Puzzle | 
 Single | 
 Vision | 
 GitHub | 
 2024/07 | 

 
 Plancraft ( Dagan et al., 2025 ) | 
 Minecraft | 
 Single | 
 Mixed | 
 GitHub | 
 2025/03 | 

 
 UnrealZoo ( Zhong et al., 2025 ) | 
 Photo-realistic 3D Worlds | 
 Multi | 
 Vision | 
 GitHub | 
 2025/06 | 

 
 Diverse | 
 CuisineWorld ( Gong et al., 2024b ) | 
 Cooperative Tasks | 
 Multi | 
 Text | 
 GitHub | 
 2023/09 | 

 
 Cradle ( Tan et al., 2024a ) | 
 Multiple Video Games | 
 Single | 
 Mixed | 
 GitHub | 
 2024/03 | 

 
 BALROG ( Paglieri et al., 2025 ) | 
 Multiple RL Games | 
 Single | 
 Mixed | 
 GitHub | 
 2024/11 | 

 
 lmgame-Bench ( Hu et al., 2026 ) | 
 Multiple Video Games | 
 Single / Multi | 
 Mixed | 
 GitHub | 
 2025/05 | 

 
 Orak ( Park et al., 2026 ) | 
 Multiple Video Games | 
 Single | 
 Mixed | 
 GitHub | 
 2025/06 | 

 
 
 
 High-quality game environments/benchmarks are crucial for advancing the capabilities of LLMGAs. Such environments should not only be more complex, but complex in targeted ways that expose the distinctive weaknesses of current architectures. This entails: (i) tasks with deeper compositional structure and long-horizon dependencies, ensuring that success cannot be reduced to pattern-matching templates; (ii) world dynamics governed by consistent physical or social rules, requiring agents to acquire and exploit regularities rather than memorize isolated instances; and (iii) scalability in both breadth (diverse tasks and domains) and depth (persistent settings spanning multiple days or large populations of agents).

 
 
 Most existing benchmarks evaluate game agents with coarse-grained metrics such as win rate and task success rate ( Shridhar et al., 2021 ; Wang et al., 2022 ) . While these high-level measures capture overall gameplay performance, they obscure where and why agents fail. Moving forward, the field requires fine-grained evaluation protocols that can diagnose the core components of agent design, memory, reasoning, perception–action translation, and multi-agent coordination, thus linking empirical evaluation to theoretical progress. One practical approach is game-specific metric design. Such metrics leverage domain knowledge to expose failure modes that aggregate scores cannot reveal. For example, PokéLLMon introduces the consecutive switch rate, measuring the proportion of turns where the agent switches Pokémon consecutively as a proxy for short-term inconsistency ( Hu et al., 2024 ) . Voyager uses map coverage and number of unique items collected to quantify exploration breadth and inventory management ( Wang et al., 2024a ) . At a larger scale, Project Sid ( AL et al., 2024 ) invite new metrics, such as persistence of social norms or stability of emergent institutions, providing outcome measures with diagnostic signals for interpreting agent behavior.

 
 
 However, not all evaluation targets lend themselves to direct quantification. Aspects such as role fidelity, believability, or the coherence of emergent behavior often require judgment-based protocols. In Generative Agents ( Park et al., 2023 ) , for example, agents were interviewed about their recent activities, relationships, or future plans, and their answers were cross-checked against internal memory logs. Human evaluators then rated responses for consistency, plausibility, and coherence, providing a qualitative assessment of role fidelity. This procedure can be extended through LLM-based judgments, where a strong LLM serves as the evaluator to assess the quality of agent behaviors, offering scalability and reproducibility. To mitigate bias, a practical solution is to adopt hybrid protocols, where LLM judgments are guided by rubrics defined by human experts and their outputs are validated through human spot-checking.

 
 
 
 

## 10. Conclusion

 
 This survey provides an up-to-date review of LLMGAs through an analytical framework. At the single agent level, we synthesize prior work across three core components, memory, reasoning, and perception-action interface, that together describe how agents perceive, think, and act through language. Extending this foundation, we introduce a complementary multi-agent framework for analyzing communication protocols and organizational structures that govern coordination, task allocation, and large-scale stability. We further introduce a challenge-centered taxonomy that maps six major game genres to their dominant agent design requirements, from low-latency response in action games to open-ended goal generation in sandbox worlds. Together, these perspectives present a coherent view of how language-enabled agents operate in interactive game environments and outline key challenges that define the next stage of research.

 
 
 

## Acknowledgements

 
 This research is partially sponsored by the NSF CISE grants 2302720 and 2312758, and a CISCO research grant in Edge AI, and PACE at the Georgia Institute of Technology. The first author and the last author are the primary contacts for this work.

 
 
 

## References

 
 [1] 
 M. F. A. R. D. T. (FAIR)†, A. Bakhtin, N. Brown, E. Dinan, G. Farina, C. Flaherty, D. Fried, A. Goff, J. Gray, H. Hu, et al. (2022) 
 
 Human-level play in the game of diplomacy by combining language models with strategic reasoning .
 
 Science 378 ( 6624 ), pp. 1067–1074 .
 
 Cited by: §3.3 ,
 Table 1 ,
 §8.3 ,
 §8.3 ,
 Table 5 ,
 Table 5 .
 

 [2] 
 J. Achiam, S. Adler, S. Agarwal, et al. (2023) 
 
 GPT-4 technical report .
 
 Technical Report 
 
 OpenAI .
 
 Cited by: §1 ,
 §6.1 .
 

 [3] 
 S. Agashe, Y. Fan, A. Reyna, and X. E. Wang (2025) 
 
 LLM-coordination: evaluating and analyzing multi-agent coordination abilities in large language models .
 
 In Findings of the Association for Computational Linguistics: NAACL 2025 , L. Chiruzzo, A. Ritter, and L. Wang (Eds.) ,
 
 Albuquerque, New Mexico , pp. 8038–8057 .
 
 External Links: Link ,
 Document ,
 ISBN 979-8-89176-195-7 
 
 Cited by: §6.2 ,
 §7.1 ,
 Table 5 .
 

 [4] 
 D. Ahn, S. Kim, and J. Choi (2025) 
 
 Society of mind meets real-time strategy: a hierarchical multi-agent framework for strategic reasoning .
 
 In Conference on Language Modeling (COLM) ,
 
 Cited by: §7.2 .
 

 [5] 
 A. AL, A. Ahn, N. Becker, S. Carroll, N. Christie, M. Cortes, A. Demirci, M. Du, F. Li, S. Luo, et al. (2024) 
 
 Project sid: many-agent simulations toward ai civilization .
 
 Technical Report 
 
 Altera AI .
 
 Cited by: Table 3 ,
 §7.2 ,
 §7.2 ,
 §7.2 ,
 §8.6 ,
 §9.5 .
 

 [6] 
 J. R. Anderson and L. J. Schooler (1991) 
 
 Reflections of the environment in memory .
 
 Psychological science 2 ( 6 ), pp. 396–408 .
 
 Cited by: §9.1 .
 

 [7] 
 P. Anokhin, N. Semenov, A. Sorokin, D. Evseev, A. Kravchenko, M. Burtsev, and E. Burnaev (2025) 
 
 AriGraph: learning knowledge graph world models with episodic memory for llm agents .
 
 In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence ,
 
 pp. 12–20 .
 
 Cited by: §4.2 ,
 §4.2 ,
 §8.2 .
 

 [8] 
 A. D. Baddeley and G. Hitch (1974) 
 
 Working memory .
 
 In Psychology of Learning and Motivation , G. A. Bower (Ed.) ,
 
 Vol. 8 , pp. 47–89 .
 
 Cited by: §3.1 ,
 §4.1 ,
 §4 ,
 §4 .
 

 [9] 
 A. Baddeley (2012) 
 
 Working memory: theories, models, and controversies .
 
 Annual review of psychology 63 ( 1 ), pp. 1–29 .
 
 Cited by: §3.1 ,
 §4.2 ,
 §4 ,
 §4 .
 

 [10] 
 L. W. Barsalou (1999) 
 
 Perceptual symbol systems .
 
 Behavioral and brain sciences 22 ( 4 ), pp. 577–660 .
 
 Cited by: §1 .
 

 [11] 
 S. Basavatia, K. Murugesan, and S. Ratnakar (2024) 
 
 STARLING: self-supervised training of text-based reinforcement learning agent with large language models .
 
 In Findings of the Association for Computational Linguistics: ACL 2024 ,
 
 Cited by: Table 1 .
 

 [12] 
 BBC 
 
 The hitchhiker’s guide to the galaxy text adventure: 30th anniversary edition .
 
 Note: https://www.bbc.co.uk/programmes/articles/1g84m0sXpnNCv84GpN2PLZG/the-game-30th-anniversary-edition 
 
 Cited by: §8.2 .
 

 [13] 
 C. Beattie, J. Z. Leibo, D. Teplyashin, T. Ward, M. Wainwright, H. Küttler, A. Lefrancq, S. Green, V. Valdés, A. Sadik, et al. (2016) 
 
 Deepmind lab .
 
 arXiv preprint arXiv:1612.03801 .
 
 Cited by: Table 1 .
 

 [14] 
 M. J. Beran (2012) 
 
 Foundations of metacognition .
 
 Oxford University Press .
 
 Cited by: §9.2 .
 

 [15] 
 M. Besta, N. Blach, A. Kubicek, R. Gerstenberger, M. Podstawski, L. Gianinazzi, J. Gajda, T. Lehmann, H. Niewiadomski, P. Nyczyk, et al. (2024) 
 
 Graph of thoughts: solving elaborate problems with large language models .
 
 In Proceedings of the AAAI conference on artificial intelligence ,
 
 Vol. 38 , pp. 17682–17690 .
 
 Cited by: §5.1 .
 

 [16] 
 X. Bo, Z. Zhang, Q. Dai, X. Feng, L. Wang, R. Li, X. Chen, and J. Wen (2024) 
 
 Reflective multi-agent collaboration based on large language models .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §7.2 .
 

 [17] 
 T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. (2020) 
 
 Language models are few-shot learners .
 
 Advances in neural information processing systems 33 , pp. 1877–1901 .
 
 Cited by: §1 ,
 §4.1 .
 

 [18] 
 M. Carroll, R. Shah, M. K. Ho, T. Griffiths, S. Seshia, P. Abbeel, and A. Dragan (2019) 
 
 On the utility of learning about humans for human-ai coordination .
 
 Advances in neural information processing systems 32 .
 
 Cited by: §7.2 ,
 Table 5 .
 

 [19] 
 T. Carta, C. Romac, T. Wolf, S. Lamprier, O. Sigaud, and P. Oudeyer (2023) 
 
 Grounding large language models in interactive environments with online reinforcement learning .
 
 In International Conference on Machine Learning ,
 
 pp. 3676–3713 .
 
 Cited by: §5.2 ,
 Table 3 ,
 §6.2 ,
 Table 5 .
 

 [20] 
 K. Chalamalasetti, J. Götze, S. Hakimov, B. Madureira, P. Sadler, and D. Schlangen (2023) 
 
 Clembench: using game play to evaluate chat-optimized language models as conversational agents .
 
 In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP) ,
 
 Cited by: Table 1 ,
 Table 5 .
 

 [21] 
 J. Chen, Y. Jiang, J. Lu, and L. Zhang (2024) 
 
 S-agent: self-organizing agents in open-ended environment .
 
 In ICLR 2024 Workshop on Large Language Model (LLM) Agents ,
 
 External Links: Link 
 
 Cited by: Table 3 ,
 §7.2 ,
 §7.2 ,
 §8.4 ,
 §9.4 .
 

 [22] 
 J. Chen, S. Saha, and M. Bansal (2024) 
 
 ReConcile: round-table conference improves reasoning via consensus among diverse llms .
 
 In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) ,
 
 pp. 7066–7085 .
 
 Cited by: §7.1 .
 

 [23] 
 M. Chen, Y. Li, Y. Yang, S. Yu, B. Lin, and X. He (2024) 
 
 AutoManual: constructing instruction manuals by LLM agents via interactive environmental learning .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §5.1 ,
 Table 3 .
 

 [24] 
 P. Cheng, T. Hu, H. Xu, Z. Zhang, Z. Yuan, Y. Dai, L. Han, N. Du, and X. Li (2024) 
 
 Self-playing adversarial language game enhances LLM reasoning .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §5.2 ,
 Table 3 .
 

 [25] 
 A. Chevalier, A. Wettig, A. Ajith, and D. Chen (2023) 
 
 Adapting language models to compress contexts .
 
 In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , H. Bouamor, J. Pino, and K. Bali (Eds.) ,
 
 Singapore , pp. 3829–3846 .
 
 External Links: Link ,
 Document 
 
 Cited by: §4.1 .
 

 [26] 
 W. Choi, J. Park, S. Ahn, D. Lee, and H. Woo (2025) 
 
 NeSyC: a neuro-symbolic continual learner for complex embodied tasks in open domains .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §4.2 .
 

 [27] 
 S. Choudhury and P. Sodhi (2025) 
 
 Better than your teacher: LLM agents that learn from privileged AI feedback .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §5.1 .
 

 [28] 
 P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei (2017) 
 
 Deep reinforcement learning from human preferences .
 
 Advances in neural information processing systems 30 .
 
 Cited by: §5.2 .
 

 [29] 
 A. Clark (1998) 
 
 Being there: putting brain, body, and world together again .
 
 MIT press .
 
 Cited by: §1 .
 

 [30] 
 A. Clark (2013) 
 
 Whatever next? predictive brains, situated agents, and the future of cognitive science .
 
 Behavioral and brain sciences 36 ( 3 ), pp. 181–204 .
 
 Cited by: §1 .
 

 [31] 
 R. I. Clarke, J. H. Lee, and N. Clark (2017) 
 
 Why video game genres fail: a classificatory analysis .
 
 Games and Culture 12 ( 5 ), pp. 445–465 .
 
 Cited by: §3.3 .
 

 [32] 
 N. Cloos, M. Jens, M. Naim, Y. Kuo, I. Cases, A. Barbu, and C. J. Cueva (2024) 
 
 Baba is AI: break the rules to beat the benchmark .
 
 In ICML 2024 Workshop on LLMs and Cognition ,
 
 Cited by: Table 5 .
 

 [33] 
 K. Cobbe, C. Hesse, J. Hilton, and J. Schulman (2020) 
 
 Leveraging procedural generation to benchmark reinforcement learning .
 
 In International conference on machine learning ,
 
 pp. 2048–2056 .
 
 Cited by: Table 1 ,
 §8.1 .
 

 [34] 
 M. Côté, A. Kádár, X. Yuan, B. Kybartas, T. Barnes, E. Fine, J. Moore, M. Hausknecht, L. El Asri, M. Adada, et al. (2019) 
 
 Textworld: a learning environment for text-based games .
 
 In Computer Games: 7th Workshop, CGW 2018, Held in Conjunction with the 27th International Conference on Artificial Intelligence, IJCAI 2018, Stockholm, Sweden, July 13, 2018, Revised Selected Papers 7 ,
 
 pp. 41–75 .
 
 Cited by: Table 1 ,
 §8.2 ,
 §9.5 ,
 Table 5 .
 

 [35] 
 N. Cowan (2001) 
 
 The magical number 4 in short-term memory: a reconsideration of mental storage capacity .
 
 Behavioral and brain sciences 24 ( 1 ), pp. 87–114 .
 
 Cited by: §4.1 ,
 §4.2 .
 

 [36] 
 L. Cross, N. Haber, and D. L. K. Yamins (2025) 
 
 Validating generative agent-based models of social norm enforcement: from replication to novel predictions .
 
 In Proceedings of the Annual Meeting of the Cognitive Science Society (CogSci) ,
 
 Cited by: §8.6 .
 

 [37] 
 G. Dagan, F. Keller, and A. Lascarides (2025) 
 
 Plancraft: an evaluation dataset for planning with LLM agents .
 
 In Conference on Language Modeling (COLM) ,
 
 Cited by: Table 1 ,
 Table 5 .
 

 [38] 
 G. Dai, W. Zhang, J. Li, S. Yang, S. Rao, A. Caetano, M. Sra, et al. (2024) 
 
 Artificial leviathan: exploring social evolution of llm agents through the lens of hobbesian social contract theory .
 
 arXiv preprint arXiv:2406.14373 .
 
 Cited by: Table 1 ,
 §7.2 ,
 §8.6 .
 

 [39] 
 A. de Wynter (2025) 
 
 Will gpt-4 run doom? .
 
 IEEE Transactions on Games 17 ( 2 ), pp. 451–459 .
 
 External Links: Document 
 
 Cited by: §6.1 .
 

 [40] 
 Y. Ding, L. L. Zhang, C. Zhang, Y. Xu, N. Shang, J. Xu, F. Yang, and M. Yang (2024) 
 
 LongRoPE: extending llm context window beyond 2 million tokens .
 
 In International Conference on Machine Learning ,
 
 pp. 11091–11104 .
 
 Cited by: §4.1 .
 

 [41] 
 S. Du and X. Zhang (2024) 
 
 Helmsman of the masses? evaluate the opinion leadership of large language models in the werewolf game .
 
 In Conference on Language Modeling (COLM) ,
 
 Cited by: §7.1 .
 

 [42] 
 Y. Du, O. Watkins, Z. Wang, C. Colas, T. Darrell, P. Abbeel, A. Gupta, and J. Andreas (2023) 
 
 Guiding pretraining in reinforcement learning with large language models .
 
 In International Conference on Machine Learning ,
 
 pp. 8657–8677 .
 
 Cited by: §5.2 ,
 Table 3 ,
 §6.1 ,
 Table 4 ,
 §8.4 .
 

 [43] 
 A. Duffy, S. J. Paech, I. Shastri, E. Karpinski, B. Alloui-Cros, T. Marques, and M. L. Olson (2025) 
 
 Democratizing diplomacy: a harness for evaluating any large language model on full-press diplomacy .
 
 In MTI-LLM Workshop at the Conference on Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: Table 5 .
 

 [44] 
 H. Ebbinghaus (1913) 
 
 Memory: a contribution to experimental psychology .
 
 Teachers College, Columbia University , New York .
 
 Note: Translated by Henry A. Ruger and Clara E. Bussenius 
 
 Cited by: §4.2 .
 

 [45] 
 J. S. B. Evans (2008) 
 
 Dual-processing accounts of reasoning, judgment, and social cognition .
 
 Annu. Rev. Psychol. 59 ( 1 ), pp. 255–278 .
 
 Cited by: §3.1 ,
 §5 .
 

 [46] 
 C. Fan, J. Chen, Y. Jin, and H. He (2024) 
 
 Can large language models serve as rational players in game theory? a systematic analysis .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Cited by: §8.5 .
 

 [47] 
 L. Fan, G. Wang, Y. Jiang, A. Mandlekar, Y. Yang, H. Zhu, A. Tang, D. Huang, Y. Zhu, and A. Anandkumar (2022) 
 
 Minedojo: building open-ended embodied agents with internet-scale knowledge .
 
 Advances in Neural Information Processing Systems 35 , pp. 18343–18362 .
 
 Cited by: Table 1 ,
 §8.4 ,
 Table 5 ,
 Table 5 .
 

 [48] 
 W. Fang, L. Li, K. Shi, Y. Yang, and J. Zhang (2026) 
 
 Belief-driven multi-agent collaboration via approximate perfect bayesian equilibrium for social simulation .
 
 In Proceedings of the ACM Web Conference (WWW) ,
 
 Cited by: §7.1 .
 

 [49] 
 X. Feng, Y. Luo, Z. Wang, H. Tang, M. Yang, K. Shao, D. Mguni, Y. Du, and J. Wang (2024) 
 
 Chessgpt: bridging policy learning and language modeling .
 
 Advances in Neural Information Processing Systems 36 .
 
 Cited by: Table 1 ,
 Table 3 ,
 Table 4 ,
 §8.5 ,
 §8.5 ,
 Table 5 .
 

 [50] 
 Y. Feng, Y. Wang, J. Liu, S. Zheng, and Z. Lu (2024) 
 
 LLaMA-rider: spurring large language models to explore the open world .
 
 In Findings of the Association for Computational Linguistics: NAACL 2024 ,
 
 pp. 4705–4724 .
 
 Cited by: §4.2 ,
 Table 3 .
 

 [51] 
 S. M. Fleming and R. J. Dolan (2012) 
 
 The neural basis of metacognitive ability .
 
 Philosophical Transactions of the Royal Society B: Biological Sciences 367 ( 1594 ), pp. 1338–1349 .
 
 Cited by: §9.2 .
 

 [52] 
 S. Forouzandeh, W. Peng, P. Moradi, X. Yu, and M. Jalili (2026) 
 
 Learning hierarchical procedural memory for LLM agents through bayesian selection and contrastive refinement .
 
 In Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS) ,
 
 Cited by: §4.2 ,
 Table 2 .
 

 [53] 
 R. Gallotta, G. Todd, M. Zammit, S. Earle, A. Liapis, J. Togelius, and G. N. Yannakakis (2024) 
 
 Large language models and games: a survey and roadmap .
 
 IEEE Transactions on Games .
 
 Cited by: §1 .
 

 [54] 
 C. Gao, X. Lan, N. Li, Y. Yuan, J. Ding, Z. Zhou, F. Xu, and Y. Li (2024) 
 
 Large language models empowered agent-based modeling and simulation: a survey and perspectives .
 
 Humanities and Social Sciences Communications 11 ( 1 ), pp. 1–24 .
 
 Cited by: §1 .
 

 [55] 
 T. Ge, H. Jing, L. Wang, X. Wang, S. Chen, and F. Wei (2024) 
 
 In-context autoencoder for context compression in a large language model .
 
 In The Twelfth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §4.1 .
 

 [56] 
 H. Golchha, S. Yerawar, D. Patel, S. Dan, and K. Murugesan (2024) 
 
 Language guided exploration for RL agents in text environments .
 
 In Findings of the Association for Computational Linguistics: NAACL 2024 ,
 
 Cited by: §5.2 .
 

 [57] 
 D. Gong, X. Wan, and D. Wang (2024) 
 
 Working memory capacity of chatgpt: an empirical study .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Vol. 38 , pp. 10048–10056 .
 
 Cited by: §4.1 .
 

 [58] 
 R. Gong, Q. Huang, X. Ma, Y. Noda, Z. Durante, Z. Zheng, D. Terzopoulos, L. Fei-Fei, J. Gao, and H. Vo (2024) 
 
 MindAgent: emergent gaming interaction .
 
 In Findings of the Association for Computational Linguistics: NAACL 2024 ,
 
 pp. 3154–3183 .
 
 Cited by: §7.2 ,
 §7.2 ,
 Table 5 .
 

 [59] 
 Z. Guan, X. Kong, F. Zhong, and Y. Wang (2024) 
 
 Richelieu: self-evolving LLM-based agents for AI diplomacy .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §7.1 .
 

 [60] 
 J. Guo, B. Yang, P. Yoo, B. Y. Lin, Y. Iwasawa, and Y. Matsuo (2024) 
 
 Suspicion-agent: playing imperfect information games with theory of mind aware gpt-4 .
 
 In Proceedings of the 1st Conference on Language Modeling (COLM) ,
 
 External Links: Link 
 
 Cited by: Table 2 ,
 Table 3 ,
 Table 4 ,
 §8.5 .
 

 [61] 
 X. Guo, K. Huang, J. Liu, W. Fan, N. Vélez, Q. Wu, H. Wang, T. L. Griffiths, and M. Wang (2025) 
 
 Embodied LLM agents learn to cooperate in organized teams .
 
 IEEE Transactions on Computational Social Systems .
 
 Cited by: §7.2 .
 

 [62] 
 A. Gupta (2023) 
 
 Are chatgpt and gpt-4 er players?–a pre-flop analysis .
 
 arXiv preprint arXiv:2308.12466 .
 
 Cited by: Table 1 .
 

 [63] 
 D. Hafner (2022) 
 
 Benchmarking the spectrum of agent capabilities .
 
 In International Conference on Learning Representations ,
 
 Cited by: §3.3 ,
 Table 1 ,
 §5.1 ,
 §8.4 ,
 §8.4 ,
 Table 5 .
 

 [64] 
 S. M. Hamidi, L. Ye, and K. N. Plataniotis (2026) 
 
 DreamPhase: offline imagination and uncertainty-guided planning for large-language-model agents .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §8.2 .
 

 [65] 
 M. Hausknecht, P. Ammanabrolu, M. Côté, and X. Yuan (2020) 
 
 Interactive fiction games: a colossal adventure .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Vol. 34 , pp. 7903–7910 .
 
 Cited by: §3.3 ,
 Table 1 ,
 §6.2 ,
 §8.2 ,
 §8.2 ,
 Table 5 .
 

 [66] 
 L. Hu, M. Huo, Y. Zhang, H. Yu, E. P. Xing, I. Stoica, T. Rosing, H. Jin, and H. Zhang (2026) 
 
 Lmgame-bench: how good are LLMs at playing games? .
 
 In The Fourteenth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: Table 5 .
 

 [67] 
 M. Hu, T. Chen, Q. Chen, Y. Mu, W. Shao, and P. Luo (2025) 
 
 HiAgent: hierarchical working memory management for solving long-horizon agent tasks with large language model .
 
 In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025) ,
 
 External Links: Document ,
 Link 
 
 Cited by: §4.1 .
 

 [68] 
 S. Hu, T. Huang, F. İlhan, S. F. Tekin, and L. Liu (2023) 
 
 Large language model-powered smart contract vulnerability detection: new perspectives .
 
 In 2023 5th IEEE International Conference on Trust, Privacy and Security in Intelligent Systems and Applications (TPS-ISA) ,
 
 pp. 297–306 .
 
 Cited by: §5.1 .
 

 [69] 
 S. Hu, T. Huang, G. Liu, R. Kompella, and L. Liu (2025) 
 
 Pokéllmon: a grounding and reasoning benchmark for large language models in pokémon battles .
 
 ACM Transactions on Internet Technology .
 
 Cited by: §3.3 ,
 Figure 3 ,
 §4.1 ,
 §4.1 ,
 Table 2 ,
 §5.1 ,
 Table 3 ,
 §6.2 ,
 §8.5 ,
 §9.1 ,
 Table 5 .
 

 [70] 
 S. Hu, T. Huang, and L. Liu (2024) 
 
 PokéLLMon: a human-parity agent for pokémon battles with large language models .
 
 External Links: 2402.01118 
 
 Cited by: §3.1 ,
 Table 1 ,
 Table 3 ,
 §6.1 ,
 Table 4 ,
 §6 ,
 §8.5 ,
 §8.5 ,
 §9.5 .
 

 [71] 
 W. Hua, L. Fan, L. Li, K. Mei, J. Ji, Y. Ge, L. Hemphill, and Y. Zhang (2023) 
 
 War and peace (waragent): large language model-based multi-agent simulation of world wars .
 
 arXiv preprint arXiv:2311.17227 .
 
 Cited by: Table 3 .
 

 [72] 
 C. Huang, Y. Cao, Y. Wen, T. Zhou, and Y. Zhang (2024) 
 
 PokerGPT: an end-to-end lightweight solver for multi-player texas hold’em via large language model .
 
 arXiv preprint arXiv:2401.06781 .
 
 Cited by: Table 1 ,
 Table 3 .
 

 [73] 
 W. Huang, P. Abbeel, D. Pathak, and I. Mordatch (2022) 
 
 Language models as zero-shot planners: extracting actionable knowledge for embodied agents .
 
 In International Conference on Machine Learning ,
 
 pp. 9118–9147 .
 
 Cited by: §6.2 ,
 Table 4 .
 

 [74] 
 Infocom (1980) 
 
 Zork I .
 
 Note: http://ifdb.tads.org/viewgame?id=0dbnusxunq7fw5ro 
 
 Cited by: §6.1 ,
 §8.2 .
 

 [75] 
 Infocom (1982) 
 
 Zork III .
 
 Note: http://ifdb.tads.org/viewgame?id=vrsot1zgy1wfcdru 
 
 Cited by: §8.2 .
 

 [76] 
 M. Iovino, E. Scukins, J. Styrud, P. Ögren, and C. Smith (2022) 
 
 A survey of behavior trees in robotics and ai .
 
 Robotics and Autonomous Systems 154 , pp. 104096 .
 
 Cited by: §1 .
 

 [77] 
 J. Jang, M. Yoo, S. Yoon, and H. Woo (2026) 
 
 Test-time mixture of world models for embodied agents in dynamic environments .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §8.2 .
 

 [78] 
 D. Jeurissen, D. Perez-Liebana, J. Gow, D. Cakmak, and J. Kwan (2024) 
 
 Playing NetHack with LLMs: potential and limitations as zero-shot agents .
 
 In IEEE Conference on Games (CoG) ,
 
 Cited by: §8.2 .
 

 [79] 
 X. Jin, Z. Wang, Y. Du, M. Fang, H. Zhang, and J. Wang (2024) 
 
 Learning to discuss strategically: a case study on one night ultimate werewolf .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §5.2 .
 

 [80] 
 P. N. Johnson-Laird (2010) 
 
 Mental models and human reasoning .
 
 Proceedings of the National Academy of Sciences 107 ( 43 ), pp. 18243–18250 .
 
 Cited by: §3.1 ,
 §5 .
 

 [81] 
 P. N. Johnson-Laird (1983) 
 
 Mental models: towards a cognitive science of language, inference, and consciousness .
 
 Harvard University Press .
 
 Cited by: §9.1 .
 

 [82] 
 D. Kahneman (2011) 
 
 Thinking, fast and slow .
 
 Farrar, Straus and Giroux .
 
 External Links: ISBN 978-0374275631 
 
 Cited by: §5.2 .
 

 [83] 
 Z. Kaiya, M. Naim, J. Kondic, M. Cortes, J. Ge, S. Luo, G. R. Yang, and A. Ahn (2023) 
 
 Lyfe agents: generative agents for low-cost real-time social interactions .
 
 arXiv preprint arXiv:2310.02172 .
 
 Cited by: Table 1 ,
 §7.2 .
 

 [84] 
 H. Kang, Q. Zhang, H. Cai, W. Xu, T. Krishna, Y. Du, and T. Weissman 
 
 Win fast or lose slow: balancing speed and accuracy in latency-sensitive decisions of llms .
 
 In The Thirty-ninth Annual Conference on Neural Information Processing Systems ,
 
 Cited by: §8.1 ,
 §8.1 .
 

 [85] 
 H. Kang, E. Sachdeva, P. Gupta, S. Bae, and K. Lee (2025) 
 
 GFlowVLM: enhancing multi-step reasoning in vision-language models with generative flow networks .
 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) ,
 
 Cited by: §5.2 ,
 Table 3 .
 

 [86] 
 C. Kao, V. Vats, and J. Davis (2025) 
 
 Hidden in plain text: measuring LLM deception quality against human baselines using social deduction games .
 
 In International Conference on Agents (ICA) ,
 
 Cited by: §7.1 .
 

 [87] 
 S. Karten, J. Grigsby, S. Milani, K. Vodrahalli, A. Zhang, F. Fang, Y. Zhu, and C. Jin (2025) 
 
 The pokeagent challenge: competitive and long-context learning at scale .
 
 In NeurIPS Competition Track ,
 
 Cited by: §8.3 ,
 Table 5 .
 

 [88] 
 S. Karten, A. L. Nguyen, and C. Jin (2025) 
 
 Pokéchamp: an expert-level minimax language agent .
 
 In International Conference on Machine Learning (ICML) ,
 
 Cited by: Table 3 ,
 §8.5 .
 

 [89] 
 M. Kempka, M. Wydmuch, G. Runc, J. Toczek, and W. Jaśkowski (2016) 
 
 Vizdoom: a doom-based ai research platform for visual reinforcement learning .
 
 In 2016 IEEE conference on computational intelligence and games (CIG) ,
 
 Cited by: Table 1 ,
 §8.1 .
 

 [90] 
 G. Kim, P. Baldi, and S. McAleer (2024) 
 
 Language models can solve computer tasks .
 
 Advances in Neural Information Processing Systems 36 .
 
 Cited by: §5.1 .
 

 [91] 
 M. Kim and S. Hwang (2025) 
 
 CoEx – co-evolving world-model and exploration .
 
 In Findings of the Association for Computational Linguistics: EMNLP 2025 ,
 
 Cited by: §8.2 .
 

 [92] 
 M. Kim and S. Hwang (2026) 
 
 Dual-scale world memory for LLM agents towards hard-exploration problems .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §4.2 ,
 Table 2 .
 

 [93] 
 R. Klevjer (2012) 
 
 Enter the avatar: the phenomenology of prosthetic telepresence in computer games .
 
 In The philosophy of computer games ,
 
 pp. 17–38 .
 
 Cited by: §3.3 .
 

 [94] 
 M. Klissarov, M. Henaff, R. Raileanu, S. Sodhani, P. Vincent, A. Zhang, P. Bacon, D. Precup, M. C. Machado, and P. D’Oro (2025) 
 
 MaestroMotif: skill design from artificial intelligence feedback .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §6.2 ,
 Table 4 .
 

 [95] 
 T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa (2022) 
 
 Large language models are zero-shot reasoners .
 
 Advances in neural information processing systems 35 , pp. 22199–22213 .
 
 Cited by: §5.1 ,
 §5 .
 

 [96] 
 I. Kotseruba and J. K. Tsotsos (2020) 
 
 40 years of cognitive architectures: core cognitive abilities and practical applications .
 
 Artificial Intelligence Review 53 ( 1 ), pp. 17–94 .
 
 Cited by: §3.1 .
 

 [97] 
 Y. Lan, Z. Hu, L. Wang, Y. Wang, D. Ye, P. Zhao, E. Lim, H. Xiong, and H. Wang (2024) 
 
 LLM-based agent society investigation: collaboration and confrontation in avalon gameplay .
 
 In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP) ,
 
 Cited by: §8.3 .
 

 [98] 
 J. H. Lee, N. Karlova, R. I. Clarke, K. Thornton, and A. Perti (2014) 
 
 Facet analysis of video game genres .
 
 IConference 2014 Proceedings .
 
 Cited by: §2.1 ,
 §3.3 ,
 §8 .
 

 [99] 
 B. Lester, R. Al-Rfou, and N. Constant (2021) 
 
 The power of scale for parameter-efficient prompt tuning .
 
 In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing ,
 
 pp. 3045–3059 .
 
 External Links: Link 
 
 Cited by: §4.1 .
 

 [100] 
 J. Leung, Y. Wang, and Z. Shen (2026) 
 
 From entity-centric to goal-oriented graphs: enhancing LLM knowledge retrieval in minecraft .
 
 Knowledge-Based Systems .
 
 Cited by: §4.2 ,
 Table 2 .
 

 [101] 
 H. Li, Y. Chong, S. Stepputtis, J. Campbell, D. Hughes, C. Lewis, and K. Sycara (2023) 
 
 Theory of mind for multi-agent collaboration via large language models .
 
 In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , H. Bouamor, J. Pino, and K. Bali (Eds.) ,
 
 Singapore , pp. 180–192 .
 
 External Links: Link ,
 Document 
 
 Cited by: §4.1 .
 

 [102] 
 J. Li, Y. Li, N. Wadhwa, Y. Pritch, D. E. Jacobs, M. Rubinstein, M. Bansal, and N. Ruiz (2025) 
 
 Unbounded: a generative infinite game of character life simulation .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §8.6 .
 

 [103] 
 K. Li, A. K. Hopkins, D. Bau, F. Viégas, H. Pfister, and M. Wattenberg (2023) 
 
 Emergent world representations: exploring a sequence model trained on a synthetic task .
 
 In The Eleventh International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §8.5 ,
 §8.5 .
 

 [104] 
 M. Li, X. Li, and T. Zhou (2026) 
 
 Does socialization emerge in AI agent society? a case study of moltbook .
 
 External Links: 2602.14299 
 
 Cited by: Table 1 ,
 §8.6 .
 

 [105] 
 S. Li, Y. He, H. Guo, X. Bu, G. Bai, J. Liu, J. Liu, X. Qu, Y. Li, W. Ouyang, et al. (2024) 
 
 GraphReader: building graph-based agent to enhance long-context abilities of large language models .
 
 In Findings of the Association for Computational Linguistics: EMNLP 2024 ,
 
 pp. 12758–12786 .
 
 Cited by: §4.2 ,
 §4.2 .
 

 [106] 
 X. Li, S. Wang, S. Zeng, Y. Wu, and Y. Yang (2024) 
 
 A survey on llm-based multi-agent systems: workflow, infrastructure, and challenges .
 
 Vicinagearth 1 ( 1 ), pp. 9 .
 
 Cited by: §3.2 .
 

 [107] 
 Y. Li, S. Zhang, J. Sun, Y. Du, Y. Wen, X. Wang, and W. Pan (2023) 
 
 Cooperative open-ended learning framework for zero-shot coordination .
 
 In Proceedings of the 40th International Conference on Machine Learning ,
 
 pp. 20470–20484 .
 
 Cited by: §6.1 .
 

 [108] 
 Z. Li, Y. Xie, R. Shao, G. Chen, D. Jiang, and L. Nie (2024) 
 
 Optimus-1: hybrid multimodal memory empowered agents excel in long-horizon tasks .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §4.2 ,
 Table 2 .
 

 [109] 
 Z. Li, X. Xu, Z. Xu, S. Lim, and H. Zhao (2025) 
 
 LARM: large auto-regressive model for long-horizon embodied intelligence .
 
 In International Conference on Machine Learning (ICML) ,
 
 Cited by: §5.2 ,
 Table 3 .
 

 [110] 
 Z. Li, Y. Ni, R. Qi, C. Lu, L. Jiang, X. Xu, X. Liu, P. Li, Y. Guo, Z. Ma, H. Li, W. Hui, G. Xian, K. Huang, and X. Zhang (2025) 
 
 LLM-PySC2: starcraft II learning environment for large language models .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: Table 1 ,
 Table 5 .
 

 [111] 
 J. Light, M. Cai, W. Chen, G. Wang, X. Chen, W. Cheng, Y. Yue, and Z. Hu (2025) 
 
 Strategist: self-improvement of LLM decision making via bi-level tree search .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: Table 3 ,
 §8.5 .
 

 [112] 
 J. Light, M. Cai, S. Shen, and Z. Hu (2023) 
 
 From text to tactic: evaluating LLMs playing the game of avalon .
 
 In NeurIPS 2023 Foundation Models for Decision Making Workshop ,
 
 External Links: Link 
 
 Cited by: Table 1 ,
 §8.3 ,
 §8.3 ,
 Table 5 ,
 Table 5 .
 

 [113] 
 H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe (2023) 
 
 Let’s verify step by step .
 
 In The Twelfth International Conference on Learning Representations ,
 
 Cited by: §5.2 .
 

 [114] 
 B. Y. Lin, Y. Fu, K. Yang, F. Brahman, S. Huang, C. Bhagavatula, P. Ammanabrolu, Y. Choi, and X. Ren (2024) 
 
 Swiftsage: a generative agent with fast and slow thinking for complex interactive tasks .
 
 Advances in Neural Information Processing Systems 36 .
 
 Cited by: §5.2 ,
 Table 3 ,
 Table 4 ,
 §8.2 .
 

 [115] 
 J. Lin, H. Zhao, A. Zhang, Y. Wu, H. Ping, and Q. Chen (2023) 
 
 AgentSims: an open-source sandbox for large language model evaluation .
 
 Note: https://github.com/py499372727/AgentSims 
 
 Cited by: Table 1 ,
 Table 5 ,
 Table 5 .
 

 [116] 
 B. Liu, S. Yu, Z. Liu, L. Guertler, P. Qi, D. Balcells, M. Liu, C. Tan, W. Shi, M. Lin, W. S. Lee, and N. Jaques (2026) 
 
 SPIRAL: self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §5.2 ,
 Table 3 .
 

 [117] 
 J. Liu, C. Yu, J. Gao, Y. Xie, Q. Liao, Y. Wu, and Y. Wang (2024) 
 
 LLM-powered hierarchical language agent for real-time human-ai coordination .
 
 In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems ,
 
 pp. 1219–1228 .
 
 Cited by: Table 3 ,
 §6.2 ,
 §8.1 .
 

 [118] 
 S. Liu, H. Yuan, M. Hu, Y. Li, Y. Chen, S. Liu, Z. Lu, and J. Jia (2024) 
 
 RL-gpt: integrating reinforcement learning and code-as-policy .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: Table 3 ,
 §6.2 ,
 §8.4 .
 

 [119] 
 S. Liu, Y. Li, K. Zhang, Z. Cui, W. Fang, Y. Zheng, T. Zheng, and M. Song (2025) 
 
 ODYSSEY: empowering minecraft agents with open-world skills .
 
 In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence ,
 
 pp. 187–195 .
 
 Cited by: Table 5 .
 

 [120] 
 T. Liu, Y. Zhao, R. Joshi, M. Khalman, M. Saleh, P. J. Liu, and J. Liu (2024) 
 
 Statistical rejection sampling improves preference optimization .
 
 In The Twelfth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §5.2 .
 

 [121] 
 X. Liu, A. Pesaranghader, J. Kim, T. Sadhu, H. Jeon, and S. Sanner (2025) 
 
 ActiveVOO: value of observation guided active knowledge acquisition for open-world embodied lifted regression planning .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §6.1 ,
 Table 4 .
 

 [122] 
 Z. Liu, Z. Huan, X. Wang, J. Lyu, J. Tao, X. Li, F. Huang, and H. Xu (2024) 
 
 World models with hints of large language models for goal achieving .
 
 In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) ,
 
 Cited by: §8.4 .
 

 [123] 
 W. Ma, Q. Mi, Y. Zeng, X. Yan, R. Lin, Y. Wu, J. Wang, and H. Zhang (2024) 
 
 Large language models play starcraft II:benchmarks and a chain of summarization approach .
 
 In The Thirty-eighth Annual Conference on Neural Information Processing Systems ,
 
 External Links: Link 
 
 Cited by: §3.1 ,
 §3.3 ,
 Table 1 ,
 §4.1 ,
 Table 2 ,
 §6.1 ,
 Table 4 ,
 §8.5 ,
 Table 5 ,
 Table 5 .
 

 [124] 
 A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. (2024) 
 
 Self-refine: iterative refinement with self-feedback .
 
 Advances in Neural Information Processing Systems 36 .
 
 Cited by: §5.1 ,
 §5.1 .
 

 [125] 
 Q. Mi, M. Yang, X. Yu, Z. Zhao, C. Deng, B. An, H. Zhang, X. Chen, and J. Wang (2025) 
 
 MF-LLM: simulating population decision dynamics via a mean-field large language model framework .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §8.6 .
 

 [126] 
 Microsoft Research (2019) 
 
 First textworld problems: the competition using text-based games to advance capabilities of ai agents .
 
 Cited by: §6.2 ,
 §8.2 .
 

 [127] 
 M. Milkowski and T. Weninger (2026) 
 
 Deception and communication in autonomous multi-agent systems: an experimental study with among us .
 
 In Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS) ,
 
 Cited by: Table 1 ,
 §7.1 .
 

 [128] 
 Mojang Studios 
 
 Minecraft .
 
 Mojang Studios .
 
 Note: https://www.minecraft.net/en-us 
 
 Cited by: §3.3 ,
 Table 1 ,
 §6.1 ,
 §8.4 .
 

 [129] 
 M. Mosquera, J. S. Pinzon, M. Rios, Y. Fonseca, L. F. Giraldo, N. Quijano, and R. Manrique (2026) 
 
 Can LLM-augmented autonomous agents cooperate? an evaluation of their cooperative capabilities through melting pot .
 
 IEEE Transactions on Artificial Intelligence 7 ( 2 ), pp. 674–683 .
 
 Cited by: §7.2 .
 

 [130] 
 X. Mou, C. Qian, W. Liu, L. Yan, Y. Hu, X. Huang, and Z. Wei (2025) 
 
 EcoLANG: efficient and effective agent communication language induction for social simulation .
 
 In Findings of the Association for Computational Linguistics: EMNLP 2025 ,
 
 Cited by: §7.1 .
 

 [131] 
 E. Mu, J. Cai, Y. Lu, M. Zhang, K. Tei, and J. Li (2025) 
 
 Knowledge graph-enhanced large language model for incremental game playtesting .
 
 IEICE Transactions on Information and Systems .
 
 Cited by: §4.2 ,
 Table 2 .
 

 [132] 
 J. Mu, X. Li, and N. Goodman (2023) 
 
 Learning to compress prompts with gist tokens .
 
 Advances in Neural Information Processing Systems 36 , pp. 19327–19352 .
 
 Cited by: §4.1 .
 

 [133] 
 A. Newell (1994) 
 
 Unified theories of cognition .
 
 Harvard University Press .
 
 Cited by: §3.1 .
 

 [134] 
 (2024) 
 
 Octopus: embodied vision-language programmer for daily tasks .
 
 In European Conference on Computer Vision (ECCV) ,
 
 External Links: Link 
 
 Cited by: §5.2 ,
 §5.2 ,
 Table 3 ,
 §6.1 .
 

 [135] 
 OpenGenerativeAI (2024) 
 
 LLM colosseum: benchmark llms by fighting in street fighter iii .
 
 Note: https://github.com/OpenGenerativeAI/llm-colosseum 
 
 External Links: Link 
 
 Cited by: §3.3 ,
 Table 1 ,
 §8.1 ,
 Table 5 .
 

 [136] 
 L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. (2022) 
 
 Training language models to follow instructions with human feedback .
 
 Advances in Neural Information Processing Systems 35 , pp. 27730–27744 .
 
 Cited by: §1 .
 

 [137] 
 D. Paglieri, B. Cupiał, S. Coward, U. Piterbarg, M. Wolczyk, A. Khan, E. Pignatelli, Ł. Kuciński, L. Pinto, R. Fergus, J. N. Foerster, J. Parker-Holder, and T. Rocktäschel (2025) 
 
 BALROG: benchmarking agentic LLM and VLM reasoning on games .
 
 In The Thirteenth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: Table 5 .
 

 [138] 
 D. Park, M. Kim, B. Choi, J. Kim, K. Lee, J. Lee, I. Park, B. Lee, J. Hwang, J. Ahn, A. S. Mahabaleshwarkar, B. Kartal, P. Biswas, Y. Suhara, K. Lee, and J. Cho (2026) 
 
 Orak: a foundational benchmark for training and evaluating LLM agents on diverse video games .
 
 In The Fourteenth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §8.1 ,
 Table 5 .
 

 [139] 
 J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein (2023) 
 
 Generative agents: interactive simulacra of human behavior .
 
 In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology ,
 
 Cited by: §3.1 ,
 §3.3 ,
 Table 1 ,
 §4.2 ,
 §4.2 ,
 §4.2 ,
 §4.2 ,
 §4.2 ,
 Table 2 ,
 Table 3 ,
 §6.2 ,
 Table 4 ,
 §7.2 ,
 §8.3 ,
 §8.6 ,
 §8.6 ,
 §9.4 ,
 §9.5 ,
 Table 5 ,
 Table 5 .
 

 [140] 
 J. Park, J. Cho, and S. Ahn (2025) 
 
 MrSteve: instruction-following agents in minecraft with what-where-when memory .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §4.2 ,
 Table 2 .
 

 [141] 
 B. Peng, J. Quesnelle, H. Fan, and E. Shippole (2024) 
 
 YaRN: efficient context window extension of large language models .
 
 In The Twelfth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §4.1 .
 

 [142] 
 W. T. Piriyakulkij, Y. Liang, H. Tang, A. Weller, M. Kryven, and K. Ellis (2025) 
 
 PoE-world: compositional world modeling with products of programmatic experts .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §8.1 .
 

 [143] 
 A. Prasad, A. Koller, M. Hartmann, P. Clark, A. Sabharwal, M. Bansal, and T. Khot (2024) 
 
 ADaPT: as-needed decomposition and planning with language models .
 
 In Findings of the Association for Computational Linguistics: NAACL 2024 ,
 
 pp. 4226–4252 .
 
 Cited by: Table 3 ,
 §8.2 .
 

 [144] 
 PrismarineJS (2013) 
 
 Mineflayer: create minecraft bots with a powerful, stable, and high level javascript api .
 
 Note: https://github.com/PrismarineJS/mineflayer 
 
 Cited by: §6.1 ,
 §6.2 .
 

 [145] 
 X. Puig, K. Ra, M. Boben, J. Li, T. Wang, S. Fidler, and A. Torralba (2018) 
 
 Virtualhome: simulating household activities via programs .
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition ,
 
 pp. 8494–8502 .
 
 Cited by: Table 5 .
 

 [146] 
 S. Qi, S. Chen, Y. Li, X. Kong, J. Wang, B. Yang, P. Wong, Y. Zhong, X. Zhang, Z. Zhang, N. Liu, Y. Yang, and S. Zhu (2024) 
 
 CivRealm: a learning and reasoning odyssey in civilization for decision-making agents .
 
 In The Twelfth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: Table 1 ,
 §8.6 ,
 Table 5 ,
 Table 5 .
 

 [147] 
 C. Qian, Z. Xie, Y. Wang, W. Liu, K. Zhu, H. Xia, Y. Dang, Z. Du, W. Chen, C. Yang, Z. Liu, and M. Sun (2025) 
 
 Scaling large language model-based multi-agent collaboration .
 
 In The Thirteenth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §3.2 ,
 §3.2 ,
 §7.2 .
 

 [148] 
 H. Qian, C. Bai, J. Zhang, F. Wu, W. Song, and X. Li (2025) 
 
 Discriminator-guided embodied planning for LLM agent .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §5.1 ,
 Table 3 .
 

 [149] 
 S. Qiao, R. Fang, N. Zhang, Y. Zhu, X. Chen, S. Deng, Y. Jiang, P. Xie, F. Huang, and H. Chen (2024) 
 
 Agent planning with world knowledge model .
 
 In Proceedings of the 38th International Conference on Neural Information Processing Systems ,
 
 pp. 114843–114871 .
 
 Cited by: §5.1 ,
 §8.2 .
 

 [150] 
 G. Qin and B. Van Durme (2023) 
 
 Nugget: neural agglomerative embeddings of text .
 
 In International Conference on Machine Learning ,
 
 pp. 28337–28350 .
 
 Cited by: §4.1 .
 

 [151] 
 A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021) 
 
 Learning transferable visual models from natural language supervision .
 
 In International conference on machine learning ,
 
 pp. 8748–8763 .
 
 Cited by: §6.1 .
 

 [152] 
 R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn (2023) 
 
 Direct preference optimization: your language model is secretly a reward model .
 
 Advances in Neural Information Processing Systems 36 , pp. 53728–53741 .
 
 Cited by: §5.2 .
 

 [153] 
 N. Ratner, Y. Levine, Y. Belinkov, O. Ram, I. Magar, O. Abend, E. Karpas, A. Shashua, K. Leyton-Brown, and Y. Shoham (2023) 
 
 Parallel context windows for large language models .
 
 In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) ,
 
 pp. 6383–6402 .
 
 Cited by: §4.1 .
 

 [154] 
 S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-maron, M. Giménez, Y. Sulsky, J. Kay, J. T. Springenberg, T. Eccles, J. Bruce, A. Razavi, A. Edwards, N. Heess, Y. Chen, R. Hadsell, O. Vinyals, M. Bordbar, and N. de Freitas (2022) 
 
 A generalist agent .
 
 Transactions on Machine Learning Research .
 
 External Links: ISSN 2835-8856 ,
 Link 
 
 Cited by: §5.2 .
 

 [155] 
 A. Rezazadeh, Z. Li, W. Wei, and Y. Bao (2025) 
 
 From isolated conversations to hierarchical schemas: dynamic tree memory representation for LLMs .
 
 In The Thirteenth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §4.2 ,
 §4.2 .
 

 [156] 
 A. Ruoss, F. Pardo, H. Chan, B. Li, V. Mnih, and T. Genewein (2025) 
 
 LMAct: a benchmark for in-context imitation learning with long multimodal demonstrations .
 
 In International Conference on Machine Learning (ICML) ,
 
 Cited by: Table 5 .
 

 [157] 
 P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning 
 
 Raptor: recursive abstractive processing for tree-organized retrieval .
 
 In The Twelfth International Conference on Learning Representations ,
 
 Cited by: §4.2 ,
 §4.2 .
 

 [158] 
 V. Sarukkai, Z. Xie, and K. Fatahalian (2025) 
 
 Self-generated in-context examples improve LLM agents for sequential decision-making tasks .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §4.2 .
 

 [159] 
 P. Schroeder, N. Morgan, H. Luo, and J. Glass (2025) 
 
 THREAD: thinking deeper with recursive spawning .
 
 In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL) ,
 
 Cited by: §5.1 .
 

 [160] 
 J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017) 
 
 Proximal policy optimization algorithms .
 
 Technical Report 
 
 OpenAI .
 
 Cited by: §5.2 .
 

 [161] 
 J. Schultz, J. Adamek, M. Jusup, M. Lanctot, M. Kaisers, S. Perrin, D. Hennes, J. Shar, C. A. Lewis, A. Ruoss, T. Zahavy, P. Veličković, L. Prince, S. Singh, E. Malmi, and N. Tomasev (2025) 
 
 Mastering board games by external and internal planning with language models .
 
 In International Conference on Machine Learning (ICML) ,
 
 Cited by: Table 3 ,
 §8.5 .
 

 [162] 
 Y. Shao, L. Li, J. Dai, and X. Qiu (2023) 
 
 Character-llm: a trainable agent for role-playing .
 
 In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing ,
 
 pp. 13153–13187 .
 
 Cited by: §4.2 ,
 §4.2 ,
 Table 2 ,
 §8.3 .
 

 [163] 
 Z. Shi, M. Fang, and L. Chen (2025) 
 
 Monte carlo planning with large language model for text-based game agents .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: Table 3 ,
 §8.2 .
 

 [164] 
 N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao (2023) 
 
 Reflexion: language agents with verbal reinforcement learning .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 External Links: Link 
 
 Cited by: §3.1 ,
 Table 2 ,
 §5.1 ,
 §5.1 ,
 Table 3 ,
 §6.1 ,
 §8.2 .
 

 [165] 
 M. Shridhar, J. Thomason, D. Gordon, Y. Bisk, W. Han, R. Mottaghi, L. Zettlemoyer, and D. Fox (2020) 
 
 Alfred: a benchmark for interpreting grounded instructions for everyday tasks .
 
 In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition ,
 
 pp. 10740–10749 .
 
 Cited by: §8.2 ,
 Table 5 .
 

 [166] 
 M. Shridhar, X. Yuan, M. Cote, Y. Bisk, A. Trischler, and M. Hausknecht (2021) 
 
 ALFWorld: aligning text and embodied environments for interactive learning .
 
 In International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: Table 1 ,
 §6.1 ,
 §8.2 ,
 §9.5 ,
 §9.5 ,
 Table 5 .
 

 [167] 
 L. Smith and M. Gasser (2005) 
 
 The development of embodied cognition: six lessons from babies .
 
 Artificial life 11 ( 1-2 ), pp. 13–29 .
 
 Cited by: §1 .
 

 [168] 
 C. H. Song, J. Wu, C. Washington, B. M. Sadler, W. Chao, and Y. Su (2023) 
 
 Llm-planner: few-shot grounded planning for embodied agents with large language models .
 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision ,
 
 pp. 2998–3009 .
 
 Cited by: Table 2 ,
 Table 3 ,
 §6.1 .
 

 [169] 
 K. Song, A. Moeini, P. Wang, L. Gong, R. Chandra, S. Zhang, and Y. Qi (2026) 
 
 Reward is enough: LLMs are in-context reinforcement learners .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §5.1 .
 

 [170] 
 Y. Song, D. Yin, X. Yue, J. Huang, S. Li, and B. Y. Lin (2024) 
 
 Trial and error: exploration-based trajectory optimization of llm agents .
 
 In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) ,
 
 pp. 7584–7600 .
 
 External Links: Link 
 
 Cited by: §5.2 ,
 §5.2 ,
 Table 3 .
 

 [171] 
 L. R. Squire (2004) 
 
 Memory systems of the brain: a brief history and current perspective .
 
 Neurobiology of learning and memory 82 ( 3 ), pp. 171–177 .
 
 Cited by: §4.2 ,
 §4 .
 

 [172] 
 SteamDB (2025) 
 
 Steam tags and genres .
 
 Note: https://steamdb.info/tags/ Accessed: 2025-08-20 
 
 Cited by: §2.1 ,
 §3.3 ,
 §8 .
 

 [173] 
 T. Sumers, S. Yao, K. Narasimhan, and T. Griffiths (2024) 
 
 Cognitive architectures for language agents .
 
 Transactions on Machine Learning Research .
 
 Note: Survey Certification 
 
 External Links: ISSN 2835-8856 ,
 Link 
 
 Cited by: §4.2 .
 

 [174] 
 Y. Sun, Z. Li, K. Fang, C. H. Lee, and A. Asadipour (2023) 
 
 Language as reality: a co-creative storytelling game experience in 1001 nights using generative AI .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE) ,
 
 Cited by: §8.3 .
 

 [175] 
 Z. Sun, H. Shi, M. Côté, G. Berseth, X. Yuan, and B. Liu (2024) 
 
 Enhancing agent learning through world dynamics modeling .
 
 In Findings of the Association for Computational Linguistics: EMNLP 2024 ,
 
 Cited by: §5.1 ,
 Table 3 .
 

 [176] 
 R. S. Sutton A. G. Barto et al. (1998) 
 
 Reinforcement learning: an introduction .
 
 Vol. 1 , MIT press Cambridge .
 
 Cited by: §1 .
 

 [177] 
 P. Sweetser (2024) 
 
 Large language models and video games: a preliminary scoping review .
 
 In Proceedings of the 6th ACM Conference on Conversational User Interfaces ,
 
 pp. 1–8 .
 
 Cited by: §1 .
 

 [178] 
 W. Tan, Z. Ding, W. Zhang, B. Li, B. Zhou, J. Yue, H. Xia, J. Jiang, L. Zheng, X. Xu, Y. Bi, P. Gu, X. Wang, B. F. Karlsson, B. An, and Z. Lu (2024) 
 
 Towards general computer control: a multimodal agent for red dead redemption II as a case study .
 
 In ICLR 2024 Workshop on Large Language Model (LLM) Agents ,
 
 External Links: Link 
 
 Cited by: Table 1 ,
 §6.1 ,
 §6.2 ,
 Table 4 ,
 §8.2 ,
 Table 5 .
 

 [179] 
 W. Tan, W. Zhang, S. Liu, L. Zheng, X. Wang, and B. An (2024) 
 
 True knowledge comes from practice: aligning large language models with embodied environments via reinforcement learning .
 
 In The Twelfth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §5.2 ,
 Table 3 ,
 §6.2 ,
 Table 4 .
 

 [180] 
 X. Tang, J. Li, Y. Liang, S. Zhu, M. Zhang, and Z. Zheng (2024) 
 
 Mars: situated inductive reasoning in an open-world environment .
 
 Advances in Neural Information Processing Systems 37 , pp. 17830–17869 .
 
 Cited by: Table 5 .
 

 [181] 
 (2013) 
 
 The arcade learning environment: an evaluation platform for general agents .
 
 Journal of artificial intelligence research 47 , pp. 253–279 .
 
 Cited by: §3.3 ,
 Table 1 ,
 §8.1 .
 

 [182] 
 W. Tian, S. Zhang, K. Zhang, X. Chi, C. Fan, J. Lu, Y. Luo, Q. Zhou, Y. Zhao, N. Liu, S. Lin, Z. Qin, X. Ju, S. Zhang, and J. Tang (2025) 
 
 SEEA-R1: tree-structured reinforcement fine-tuning for self-evolving embodied agents .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §5.2 ,
 Table 3 .
 

 [183] 
 Z. Tian, S. Peng, D. Huang, J. Guo, R. Chen, R. Zhang, X. Zhang, Y. Guo, Z. Du, Q. Guo, L. Li, Y. Pu, X. Hu, and Y. Chen (2026) 
 
 Code driven planning with domain-adaptive selector .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §6.2 ,
 Table 4 .
 

 [184] 
 S. Toshniwal, S. Wiseman, K. Livescu, and K. Gimpel (2022) 
 
 Chess as a testbed for language model state tracking .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Vol. 36 , pp. 11385–11393 .
 
 Cited by: Table 1 ,
 §8.5 ,
 §8.5 .
 

 [185] 
 H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. (2023) 
 
 Llama: open and efficient foundation language models .
 
 Technical Report 
 
 Meta AI .
 
 Cited by: §5.2 .
 

 [186] 
 (2023) 
 
 Tree of thoughts: deliberate problem solving with large language models .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 External Links: Link 
 
 Cited by: §5.1 .
 

 [187] 
 L. Trung, X. Zhang, Z. Jie, P. Sun, X. Jin, and H. Li (2024) 
 
 Reft: reasoning with reinforced fine-tuning .
 
 In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) ,
 
 pp. 7601–7614 .
 
 Cited by: §5.2 .
 

 [188] 
 E. Tulving et al. (1972) 
 
 Episodic and semantic memory .
 
 Organization of memory 1 ( 381-403 ), pp. 1 .
 
 Cited by: §4 .
 

 [189] 
 F. J. Varela, E. Thompson, and E. Rosch (2017) 
 
 The embodied mind, revised edition: cognitive science and human experience .
 
 MIT press .
 
 Cited by: §1 .
 

 [190] 
 A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017) 
 
 Attention is all you need .
 
 Advances in neural information processing systems 30 .
 
 Cited by: §4.1 .
 

 [191] 
 B. Wang, X. Liang, J. Yang, H. Huang, Z. Wu, S. Wu, Z. Ma, and Z. Li (2025) 
 
 Scm: enhancing large language model with self-controlled memory framework .
 
 In International Conference on Database Systems for Advanced Applications ,
 
 pp. 188–203 .
 
 Cited by: §4.2 .
 

 [192] 
 G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar (2024) 
 
 Voyager: an open-ended embodied agent with large language models .
 
 Transactions on Machine Learning Research .
 
 Note: 
 
 External Links: ISSN 2835-8856 ,
 Link 
 
 Cited by: §3.1 ,
 §4.2 ,
 §4.2 ,
 §4.2 ,
 Table 2 ,
 §5.1 ,
 Table 3 ,
 §6.2 ,
 Table 4 ,
 §6 ,
 §8.4 ,
 §9.5 .
 

 [193] 
 L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, et al. (2024) 
 
 A survey on large language model based autonomous agents .
 
 Frontiers of Computer Science 18 ( 6 ), pp. 186345 .
 
 Cited by: §1 ,
 §1 .
 

 [194] 
 L. Wang, Z. Zhang, and X. Chen (2025) 
 
 Investigating and extending homans’ social exchange theory with large language model based agents .
 
 In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL) ,
 
 Cited by: §8.6 .
 

 [195] 
 N. Wang, Z.y. Peng, H. Que, J. Liu, W. Zhou, Y. Wu, H. Guo, R. Gan, Z. Ni, J. Yang, M. Zhang, Z. Zhang, W. Ouyang, K. Xu, W. Huang, J. Fu, and J. Peng (2024) 
 
 Rolellm: benchmarking, eliciting, and enhancing role-playing abilities of large language models .
 
 In Findings of the Association for Computational Linguistics: ACL 2024 , L. Ku, A. Martins, and V. Srikumar (Eds.) ,
 
 Bangkok, Thailand , pp. 14743–14777 .
 
 External Links: Link ,
 Document 
 
 Cited by: §8.3 ,
 Table 5 .
 

 [196] 
 R. Wang, H. Yu, W. Zhang, Z. Qi, M. Sap, G. Neubig, Y. Bisk, and H. Zhu (2024) 
 
 SOTOPIA- π \pi : interactive learning of socially intelligent language agents .
 
 In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL) ,
 
 Cited by: §5.2 .
 

 [197] 
 R. Wang, P. Jansen, M. Côté, and P. Ammanabrolu (2022) 
 
 SCIENCEWORLD: is your agent smarter than a 5th grader? .
 
 In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing ,
 
 Abu Dhabi , pp. 11279–11298 .
 
 External Links: Link 
 
 Cited by: §3.3 ,
 Table 1 ,
 §8.2 ,
 §9.5 ,
 §9.5 ,
 Table 5 .
 

 [198] 
 S. Wang, L. Ji, R. Wang, W. Zhao, H. Liu, Y. Hou, and Y. N. Wu (2025) 
 
 Explore the reasoning capability of LLMs in the chess testbed .
 
 In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) ,
 
 Cited by: §8.5 .
 

 [199] 
 W. Wang, F. Bie, J. Chen, D. Zhang, S. Huang, E. Kharlamov, and J. Tang (2025) 
 
 Can large language models master complex card games? .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: Table 1 ,
 §5.2 ,
 Table 3 .
 

 [200] 
 X. Wang, H. Wang, Y. Zhang, X. Yuan, R. Xu, J. Huang, S. Yuan, H. Guo, J. Chen, S. Zhou, et al. (2025) 
 
 Coser: coordinating llm-based persona simulation of established roles .
 
 In Forty-second International Conference on Machine Learning ,
 
 Cited by: §8.3 ,
 Table 5 .
 

 [201] 
 X. Wang, Y. Xiao, J. Huang, S. Yuan, R. Xu, H. Guo, Q. Tu, Y. Fei, Z. Leng, W. Wang, et al. (2024) 
 
 InCharacter: evaluating personality fidelity in role-playing agents through psychological interviews .
 
 In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) ,
 
 pp. 1840–1873 .
 
 Cited by: §8.3 .
 

 [202] 
 X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou (2023) 
 
 Self-consistency improves chain of thought reasoning in language models .
 
 In The Eleventh International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §4.1 ,
 §5.1 .
 

 [203] 
 Y. Wang, Y. Chen, F. Zhong, L. Ma, and Y. Wang (2025) 
 
 Simulating human-like daily activities with desire-driven autonomy .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §8.6 .
 

 [204] 
 Y. Wang, G. M. Lucas, B. Becerik-Gerber, and V. Ustun (2025) 
 
 Implicit behavioral alignment of language agents in high-stakes crowd simulations .
 
 In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) ,
 
 Cited by: §8.6 .
 

 [205] 
 Z. Wang, Y. Y. Chiu, and Y. C. Chiu (2023) 
 
 Humanoid agents: platform for simulating human-like generative agents .
 
 In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations ,
 
 External Links: Link 
 
 Cited by: Table 1 ,
 Table 3 ,
 §8.6 ,
 §8.6 ,
 Table 5 ,
 Table 5 .
 

 [206] 
 Z. Wang, S. Cai, G. Chen, A. Liu, X. Ma, and Y. Liang (2023) 
 
 Describe, explain, plan and select: interactive planning with LLMs enables open-world multi-task agents .
 
 In Thirty-seventh Conference on Neural Information Processing Systems ,
 
 External Links: Link 
 
 Cited by: §5.1 ,
 Table 3 ,
 §6.1 ,
 §8.4 .
 

 [207] 
 Z. Wang, S. Cai, A. Liu, Y. Jin, J. Hou, B. Zhang, H. Lin, Z. He, Z. Zheng, Y. Yang, et al. (2025) 
 
 JARVIS-1: open-world multi-task agents with memory-augmented multimodal language models .
 
 IEEE Transactions on Pattern Analysis and Machine Intelligence 47 ( 3 ), pp. 1894–1907 .
 
 Cited by: §4.2 ,
 §4.2 ,
 Table 2 ,
 Table 3 ,
 §6.1 ,
 Table 4 ,
 §8.4 .
 

 [208] 
 M. Weber, D. Y. Fu, Q. Anthony, Y. Oren, S. Adams, A. Alexandrov, X. Lyu, H. Nguyen, X. Yao, V. Adams, et al. (2024) 
 
 Redpajama: an open dataset for training large language models .
 
 Advances in neural information processing systems 37 , pp. 116462–116492 .
 
 Cited by: §4.2 .
 

 [209] 
 J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. (2022) 
 
 Chain-of-thought prompting elicits reasoning in large language models .
 
 Advances in Neural Information Processing Systems 35 , pp. 24824–24837 .
 
 Cited by: §3.1 ,
 §5.1 ,
 §5.1 ,
 §5 .
 

 [210] 
 D. Wu, F. Brudy, B. Liu, and Y. Wang (2025) 
 
 IndoorWorld: integrating physical task solving and social simulation in a heterogeneous multi-agent environment .
 
 In Findings of the Association for Computational Linguistics: EMNLP 2025 ,
 
 Cited by: Table 1 .
 

 [211] 
 D. Wu, H. Shi, Z. Sun, and B. Liu (2024) 
 
 Deciphering digital detectives: understanding llm behaviors and capabilities in multi-agent mystery games .
 
 In Findings of the Association for Computational Linguistics ACL 2024 ,
 
 pp. 8225–8291 .
 
 Cited by: §8.3 .
 

 [212] 
 Y. Wu, T. Yue, S. Zhang, C. Wang, and Q. Wu (2024) 
 
 StateFlow: enhancing LLM task-solving through state-driven workflows .
 
 In Conference on Language Modeling (COLM) ,
 
 Cited by: §4.1 .
 

 [213] 
 Y. Wu, Y. Fan, S. Y. Min, S. Prabhumoye, S. M. McAleer, R. Salakhutdinov, Y. Bisk, Y. Li, and T. Mitchell (2024) 
 
 AgentKit: structured LLM reasoning with dynamic graphs .
 
 In Conference on Language Modeling (COLM) ,
 
 Cited by: §5.1 ,
 Table 3 .
 

 [214] 
 Y. Wu, S. Y. Min, S. Prabhumoye, Y. Bisk, R. R. Salakhutdinov, A. Azaria, T. M. Mitchell, and Y. Li (2024) 
 
 SPRING: studying papers and reasoning to play games .
 
 Advances in Neural Information Processing Systems 36 .
 
 Cited by: §5.1 ,
 Table 3 ,
 §8.4 .
 

 [215] 
 J. Xiang, T. Tao, Y. Gu, T. Shu, Z. Wang, Z. Yang, and Z. Hu (2024) 
 
 Language models meet world models: embodied experiences enhance language models .
 
 Advances in neural information processing systems 36 .
 
 Cited by: Table 2 .
 

 [216] 
 Y. Xiao, Y. Sun, M. Xu, U. M. Sehwag, J. Vann, D. Garg, and S. Ganesh (2024) 
 
 O3D: offline data-driven discovery and distillation for sequential decision-making with large language models .
 
 In Conference on Language Modeling (COLM) ,
 
 Cited by: §4.2 .
 

 [217] 
 C. Xie, C. Chen, F. Jia, Z. Ye, S. Lai, K. Shu, J. Gu, A. Bibi, Z. Hu, D. Jurgens, J. Evans, P. Torr, B. Ghanem, and G. Li (2024) 
 
 Can large language model agents simulate human trust behavior? .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §8.6 .
 

 [218] 
 W. Xiong, Y. Song, X. Zhao, W. Wu, X. Wang, K. Wang, C. Li, W. Peng, and S. Li (2024) 
 
 Watch every step! LLM agent learning via iterative step-level process refinement .
 
 In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing ,
 
 pp. 1556–1572 .
 
 External Links: Link 
 
 Cited by: §5.2 ,
 Table 3 .
 

 [219] 
 B. Xu, S. Zhao, R. Wu, Z. Huang, J. Wang, Z. Hu, K. Wang, H. Liu, T. Lv, L. Li, C. Fan, X. Tong, and J. Han (2025) 
 
 Empowering economic simulation for massively multiplayer online games through generative agent-based modeling .
 
 In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) ,
 
 Cited by: §8.6 .
 

 [220] 
 K. Xu, F. Meng, C. Verbrugge, and S. Lucas (2026) 
 
 CSP4SDG: constraint and information-theory based role identification in social deduction games with LLM-enhanced inference .
 
 Proceedings of the AAAI Conference on Artificial Intelligence 40 .
 
 Cited by: §7.1 .
 

 [221] 
 K. Xu, J. Chai, S. Li, Y. Fu, Y. Zhu, and D. Zhao (2025) 
 
 DipLLM: fine-tuning LLM for strategic decision-making in diplomacy .
 
 In International Conference on Machine Learning (ICML) ,
 
 Cited by: §5.2 ,
 Table 3 .
 

 [222] 
 S. Xu and F. Zhong (2025) 
 
 CoMet: metaphor-driven covert communication for multi-agent language games .
 
 In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL) ,
 
 Cited by: §7.1 .
 

 [223] 
 W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang (2025) 
 
 A-mem: agentic memory for LLM agents .
 
 In The Thirty-ninth Annual Conference on Neural Information Processing Systems ,
 
 External Links: Link 
 
 Cited by: §4.2 .
 

 [224] 
 Y. Xu, S. Wang, P. Li, F. Luo, X. Wang, W. Liu, and Y. Liu (2023) 
 
 Exploring large language models for communication games: an empirical study on werewolf .
 
 ( arXiv:2309.04658 ).
 
 Cited by: Table 1 ,
 Table 3 ,
 §8.3 ,
 Table 5 ,
 Table 5 .
 

 [225] 
 Z. Xu, C. Yu, F. Fang, Y. Wang, and Y. Wu (2024) 
 
 Language agents with reinforcement learning for strategic play in the werewolf game .
 
 In Forty-first International Conference on Machine Learning ,
 
 External Links: Link 
 
 Cited by: §3.3 ,
 Table 3 ,
 Table 4 ,
 §8.3 .
 

 [226] 
 G. N. Yannakakis and J. Togelius (2018) 
 
 Artificial intelligence and games .
 
 Springer .
 
 External Links: Document 
 
 Cited by: §1 ,
 Abstract. .
 

 [227] 
 S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao (2023) 
 
 ReAct: synergizing reasoning and acting in language models .
 
 In The Eleventh International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §3.1 ,
 §3.1 ,
 §5.1 ,
 Table 3 ,
 §6.1 ,
 Table 4 ,
 §8.2 .
 

 [228] 
 J. Ye, Z. Wu, J. Gao, Z. Wu, X. Jiang, Z. Li, and L. Kong (2025) 
 
 Implicit search via discrete diffusion: a study on chess .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: Table 3 ,
 §8.5 .
 

 [229] 
 N. Yeung and C. Summerfield (2012) 
 
 Metacognition in human decision-making: confidence and error monitoring .
 
 Philosophical Transactions of the Royal Society B: Biological Sciences 367 ( 1594 ), pp. 1310–1321 .
 
 Cited by: §9.2 .
 

 [230] 
 X. Yi, Z. Zhou, C. Cao, Q. Niu, T. Liu, and B. Han 
 
 From debate to equilibrium: belief-driven multi-agent llm reasoning via bayesian nash equilibrium .
 
 In Forty-second International Conference on Machine Learning ,
 
 Cited by: §7.1 .
 

 [231] 
 M. Yoo, J. Jang, S. Yoon, and H. Woo (2025) 
 
 World model implanting for test-time adaptation of embodied agents .
 
 In International Conference on Machine Learning (ICML) ,
 
 Cited by: §8.2 .
 

 [232] 
 S. Yu and C. Lu (2025) 
 
 ADAM: an embodied causal agent in open-world environments .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §8.4 .
 

 [233] 
 H. Yuan, C. Zhang, H. Wang, F. Xie, P. Cai, H. Dong, and Z. Lu (2023) 
 
 Skill reinforcement learning and planning for open-world long-horizon tasks .
 
 In NeurIPS 2023 Foundation Models for Decision Making Workshop ,
 
 External Links: Link 
 
 Cited by: Table 3 ,
 §8.4 .
 

 [234] 
 A. Zala, J. Cho, H. Lin, J. Yoon, and M. Bansal (2024) 
 
 EnvGen: generating and adapting environments via LLMs for training embodied agents .
 
 In Conference on Language Modeling (COLM) ,
 
 Cited by: §5.2 .
 

 [235] 
 A. Zeng, M. Liu, R. Lu, B. Wang, X. Liu, Y. Dong, and J. Tang (2024) 
 
 AgentTuning: enabling generalized agent abilities for llms .
 
 In Findings of the Association for Computational Linguistics ACL 2024 ,
 
 pp. 3053–3077 .
 
 Cited by: §5.2 .
 

 [236] 
 X. Zeng, Y. Sun, H. Li, S. Liu, and H. Wang (2026) 
 
 ReCAPA: hierarchical predictive correction to mitigate cascading failures .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: §5.1 ,
 Table 3 .
 

 [237] 
 S. Zhai, H. Bai, Z. Lin, J. Pan, P. Tong, Y. Zhou, A. Suhr, S. Xie, Y. LeCun, Y. Ma, et al. (2025) 
 
 Fine-tuning large vision-language models as decision-making agents via reinforcement learning .
 
 Advances in Neural Information Processing Systems 37 , pp. 110935–110971 .
 
 Cited by: §5.2 .
 

 [238] 
 C. Zhang, K. Yang, S. Hu, Z. Wang, G. Li, Y. Sun, C. Zhang, Z. Zhang, A. Liu, S. Zhu, X. Chang, J. Zhang, F. Yin, Y. Liang, and Y. Yang (2024) 
 
 ProAgent: building proactive cooperative agents with large language models .
 
 Proceedings of the AAAI Conference on Artificial Intelligence 38 ( 16 ), pp. 17609–17617 .
 
 External Links: Link 
 
 Cited by: Table 2 ,
 §5.1 ,
 Table 4 ,
 §7.1 ,
 §7.1 .
 

 [239] 
 D. Zhang, S. Zhoubian, Z. Hu, Y. Yue, Y. Dong, and J. Tang (2024) 
 
 Rest-mcts*: llm self-training via process reward guided tree search .
 
 Advances in Neural Information Processing Systems 37 , pp. 64735–64772 .
 
 Cited by: §5.2 .
 

 [240] 
 D. Zhang, L. Chen, S. Zhang, H. Xu, Z. Zhao, and K. Yu (2024) 
 
 Large language models are semi-parametric reinforcement learning agents .
 
 Advances in Neural Information Processing Systems 36 .
 
 Cited by: §4.2 ,
 §4.2 .
 

 [241] 
 H. Zhang, W. Du, J. Shan, Q. Zhou, Y. Du, J. B. Tenenbaum, T. Shu, and C. Gan (2024) 
 
 Building cooperative embodied agents modularly with large language models .
 
 In The Twelfth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §3.2 ,
 §6.1 ,
 Table 4 ,
 §7.1 ,
 §7.1 ,
 §7.2 ,
 §7.2 ,
 §7 .
 

 [242] 
 J. Zhang, J. Lehman, K. Stanley, and J. Clune (2024) 
 
 OMNI: open-endedness via models of human notions of interestingness .
 
 In The Twelfth International Conference on Learning Representations ,
 
 External Links: Link 
 
 Cited by: §8.4 .
 

 [243] 
 J. Zhang and Y. Long (2025) 
 
 Learning to play like humans: a framework for LLM adaptation in interactive fiction games .
 
 In Findings of the Association for Computational Linguistics: ACL 2025 ,
 
 Cited by: §8.2 .
 

 [244] 
 Y. Zhang, S. Yang, C. Bai, F. Wu, X. Li, Z. Wang, and X. Li (2025) 
 
 Towards efficient LLM grounding for embodied multi-agent collaboration .
 
 In Findings of the Association for Computational Linguistics: ACL 2025 ,
 
 Cited by: §6.2 ,
 Table 4 .
 

 [245] 
 Y. Zhang, X. Han, H. Li, K. Chen, and S. Lin (2025) 
 
 Complete chess games enable LLM become a chess master .
 
 In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) ,
 
 Cited by: §5.2 ,
 Table 3 .
 

 [246] 
 Z. Zhang, Y. Lan, Y. Chen, L. Wang, X. Wang, and H. Wang (2025) 
 
 DVM: towards controllable LLM agents in social deduction games .
 
 In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) ,
 
 Cited by: §5.2 ,
 Table 3 .
 

 [247] 
 A. Zhao, D. Huang, Q. Xu, M. Lin, Y. Liu, and G. Huang (2024) 
 
 Expel: llm agents are experiential learners .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Vol. 38 , pp. 19632–19642 .
 
 Cited by: §5.1 .
 

 [248] 
 H. Zhao, C. Ma, G. Wang, J. Su, L. Kong, J. Xu, Z. Deng, and H. Yang (2024) 
 
 Empowering large language model agents through action learning .
 
 In Conference on Language Modeling (COLM) ,
 
 Cited by: §6.2 ,
 Table 4 .
 

 [249] 
 Z. Zhao, W. Chai, X. Wang, B. Li, S. Hao, S. Cao, T. Ye, and G. Wang (2024) 
 
 See and think: embodied agent in virtual environment .
 
 In European Conference on Computer Vision (ECCV) ,
 
 Cited by: §6.1 ,
 Table 4 .
 

 [250] 
 Z. Zhao, K. Chen, D. Guo, W. Chai, T. Ye, Y. Zhang, and G. Wang (2024) 
 
 Hierarchical auto-organizing system for open-ended multi-agent navigation .
 
 In ICLR 2024 Workshop on Large Language Model (LLM) Agents ,
 
 External Links: Link 
 
 Cited by: Table 3 ,
 §7.2 ,
 §7.2 ,
 §8.4 ,
 §9.4 .
 

 [251] 
 S. Zheng, Y. Feng, Z. Lu, et al. (2023) 
 
 Steve-eye: equipping llm-based embodied agents with visual perception in open worlds .
 
 In The Twelfth International Conference on Learning Representations ,
 
 Cited by: §6.1 .
 

 [252] 
 F. Zhong, K. Wu, C. Wang, H. Chen, H. Ci, Z. Li, and Y. Wang (2025) 
 
 UnrealZoo: enriching photo-realistic virtual worlds for embodied AI .
 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) ,
 
 Cited by: Table 1 ,
 Table 5 .
 

 [253] 
 W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang (2024) 
 
 Memorybank: enhancing large language models with long-term memory .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Vol. 38 , pp. 19724–19731 .
 
 Cited by: §4.2 ,
 §4.2 .
 

 [254] 
 S. Zhou, T. Zhou, Y. Yang, G. Long, D. Ye, J. Jiang, and C. Zhang (2025) 
 
 WALL-E: world alignment by neurosymbolic learning improves world model-based LLM agents .
 
 In Advances in Neural Information Processing Systems (NeurIPS) ,
 
 Cited by: §8.4 .
 

 [255] 
 X. Zhou, H. Zhu, L. Mathur, R. Zhang, H. Yu, Z. Qi, L. Morency, Y. Bisk, D. Fried, G. Neubig, and M. Sap (2024) 
 
 SOTOPIA: interactive evaluation for social intelligence in language agents .
 
 In International Conference on Learning Representations (ICLR) ,
 
 Cited by: Table 1 ,
 Table 5 .
 

 [256] 
 Z. Zhou, A. Qu, Z. Wu, S. Kim, A. Prakash, D. Rus, J. Zhao, B. K. H. Low, and P. P. Liang 
 
 MEM1: learning to synergize memory and reasoning for efficient long-horizon agents .
 
 In First Workshop on Multi-Turn Interactions in Large Language Models ,
 
 Cited by: §4.1 .
 

 [257] 
 D. Zhu, N. Yang, L. Wang, Y. Song, W. Wu, F. Wei, and S. Li (2024) 
 
 PoSE: efficient context window extension of llms via positional skip-wise training .
 
 In The Twelfth International Conference on Learning Representations (ICLR) ,
 
 External Links: Link 
 
 Cited by: §4.1 .
 

 [258] 
 X. Zhu, Y. Chen, H. Tian, C. Tao, W. Su, C. Yang, G. Huang, B. Li, L. Lu, X. Wang, et al. (2023) 
 
 Ghost in the minecraft: generally capable agents for open-world enviroments via large language models with text-based knowledge and memory .
 
 arXiv preprint arXiv:2305.17144 .
 
 Cited by: Table 2 ,
 §5.1 ,
 Table 3 ,
 Table 4 .
 

 [259] 
 Y. Zhu, S. Qiao, Y. Ou, S. Deng, S. Lyu, Y. Shen, L. Liang, J. Gu, H. Chen, and N. Zhang (2025) 
 
 KnowAgent: knowledge-augmented planning for LLM-based agents .
 
 In Findings of the Association for Computational Linguistics: NAACL 2025 ,
 
 Cited by: §6.2 ,
 Table 4 .
 

 [260] 
 R. Zhuang, A. Gupta, R. Yang, A. Rahane, Z. Li, and G. Anumanchipalli (2025) 
 
 PokerBench: training large language models to become professional poker players .
 
 In Proceedings of the AAAI Conference on Artificial Intelligence ,
 
 Vol. 39 , pp. 26175–26182 .
 
 Cited by: §8.5 ,
 Table 5 .
 

 [261] 
 R. A. Zwaan and G. A. Radvansky (1998) 
 
 Situation models in language comprehension and memory. .
 
 Psychological bulletin 123 ( 2 ), pp. 162 .
 
 Cited by: §9.1 .