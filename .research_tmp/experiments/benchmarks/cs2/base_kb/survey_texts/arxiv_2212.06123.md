Security of Deep Reinforcement Learning for Autonomous Driving: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.06123v3 [cs.LG] 25 Sep 2025 
 
 

# Security of Deep Reinforcement Learning for Autonomous Driving: A Survey

 
 
 Ambra Demontis 1 ,
Srishti Gupta 1 2 ,
Maura Pintor 1 , Luca Demetrio 3 , Kathrin Grosse 4 ,
Hsiao-Ying Lin 5 ,
Chengfang Fang 6 ,
Battista Biggio 1 ,
Fabio Roli 3 
 
 Affiliation:  
 
 Affiliation:  1 University of Cagliari
{ambra.demontis, srishti.gupta, maura.pintor, battista.biggio}@unica.it
 
 Affiliation:  
 
 Affiliation:  2 Sapienza, University of Rome
{srishti.gupta}@uniroma1.it
 
 Affiliation:  
 
 Affiliation:  3 University of Genoa
{luca.demetrio, fabio.roli}@unige.it
 
 Affiliation:  
 
 Affiliation:  4 IBM Research Zurich
{kathrin.grosse1}@ibm.com
 
 Affiliation:  
 
 Affiliation:  5 Huawei Technologies France
{lin.hsiao.ying}@huawei.com
 
 Affiliation:  
 
 Affiliation:  6 Huawei International
{fang.chengfang}@huawei.com
 

 Abstract 
 
 Reinforcement learning (RL) enables agents to learn optimal behaviors through interaction with their environment and has been increasingly deployed in safety-critical applications, including autonomous driving. Despite its promise, RL is susceptible to attacks designed either to compromise policy learning or to induce erroneous decisions by trained agents. Although the literature on RL security has grown rapidly and several surveys exist, existing categorizations often fall short in guiding the selection of appropriate defenses for specific systems. In this work, we present a comprehensive survey of 86 recent studies on RL security, addressing these limitations by systematically categorizing attacks and defenses according to defined threat models and single- versus multi-agent settings. Furthermore, we examine the relevance and applicability of state-of-the-art attacks and defense mechanisms within the context of autonomous driving, providing insights to inform the design of robust RL systems.

 
 
 
 Index Terms:  Survey, Reinforcement Learning, Autonomous Driving, Security, Adversarial Machine Learning

 
 
 
 This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible. 
 
 
 

## I Introduction 

 
 Reinforcement learning (RL) is a process that teaches machines to perform a task using a reward-based learning system, similar to that used by the human brain  [ 1 ] . The machine tries to perform an action, and if it gets a positive (negative) reward/outcome, it will be more (less) likely to repeat that action in the future.

 
 
 For its ability to solve complex, dynamic and high-dimensional problems  [ 2 ] , RL is used in many applications. It is used even in high-risk high-value tasks such as cybersecurity-related ones  [ 2 ] and autonomous driving  [ 3 ] . The former use RL to detect and fight against sophisticated cyberattacks, including falsified data injection in cyber-physical systems  [ 4 ] , intrusion to host computers or networks  [ 5 ] and malware [ 6 ] . The latter makes extensive use of RL to accomplish different tasks such as path planning and the development of high-level driving policies for complex navigation tasks  [ 3 ] .

 
 
 Unfortunately, RL is highly vulnerable to adversarial attacks  [ 7 ] that can significantly alter agent behavior. In safety-critical domains such as autonomous driving, this may lead to severe consequences, including crashes or traffic disruptions, making RL security a pressing concern and an active research area. However, existing surveys remain limited: some only review early works  [ 8 , 9 ] , others focus solely on attacks without defenses  [ 10 , 11 ] , or provide generic ML security overviews  [ 10 , 11 , 12 , 13 ] . The only RL-specific survey  [ 14 ] addresses a narrow subset of multi-agent systems. Consequently, a comprehensive overview of RL security, covering threat models, attack capabilities, defense shortcomings, and their alignment—remains absent. Moreover, current taxonomies  [ 7 , 15 , 11 , 16 ] fail to guide system designers in selecting defenses tailored to specific RL models and attack scenarios.

 
 
 Our survey aims to overcome the shortcomings mentioned above. The first contribution provided by our survey is a categorization that allows system designers to understand which defense they can apply to defend the system at hand against a precise threat.
The second is to categorize all the literature on attacks and defenses against RL (more than 50 papers) accordingly.
Motivated by the fact that autonomous driving is a safety-critical application and companies are investing large amounts of money in it, 1 1 
 1 
 
 
 
 https://www.therobotreport.com/wayve-raises-20m-pilot-learning-based-self-driving-cars/ our third contribution is to provide a further application-specific taxonomy of the attack and defenses proposed for this application and to examine to which extent the attacks and defenses proposed against RL can be applied to autonomous-driving systems.
Finally, we explain that attacks and defenses against RL are strongly inspired by those devised many years ago against machine learning classifiers  [ 17 ] . However, RL has peculiarities that make the application of these attacks not straightforward. For this reason, only a subset of the attacks previously proposed against ML classifiers has been tested against RL. This parallel allows us to provide our fourth contribution: a discussion about open challenges and interesting research directions about the security of RL.

 
 
 In Sec. II , we introduce the reinforcement learning (RL) background necessary to understand its vulnerabilities, followed by Sec. III , which outlines the specificities of RL algorithms in autonomous driving. Sec. IV presents a unified threat model for RL security, while Sec. V and Sec.  VI propose a novel taxonomy of attacks and defenses respectively. We then review existing works, assess their applicability to autonomous driving, and highlight open research directions.

 
 
 To summarize, we provide the following contributions.

 
 • 
 
 A taxonomy of RL attacks and defenses that enables specific threats with suitable protections;

 

 • 
 
 A categorization of over 80 papers on RL security;

 

 • 
 
 An analysis of the applicability of state-of-the-art attacks and defenses specific to autonomous driving;

 

 • 
 
 A comparison of RL and ML security, highlighting open challenges and future research directions.

 

 
 
 
 

## II Reinforcement Learning 

 
 Reinforcement learning (RL) enables agents to learn from experience, much like humans repeating rewarding actions and avoiding those with negative outcomes  [ 1 ] . RL has been successfully applied to complex, dynamic tasks such as autonomous driving, where vehicles must navigate traffic and anticipate the behavior of other road users, even under rule violations  [ 3 ] .
In this section, we provide the necessary RL background: first describing its core components, then outlining how state-of-the-art systems can be categorized.

 
 

### II-A Components of Reinforcement Learning Systems 

 
 In the following, we describe the components of an RL system and their interactions. The notation used throughout this survey is introduced alongside the description and summarized in Table  I .

 
 

#### Agent

 
 An agent is an entity that has the ability to take actions and influences its environment. In autonomous driving scenario, each car driving on the street and each pedestrian walking on it can be an agent, whereas the agent trained using RL, is referred to as ego agent. For simplicity, in this work, we only consider RL-trained agents.

 
 
 

#### Action

 
 An action a a is a move an agent can take at time t t , denoted a t a_{t} , chosen from the set of all possible actions: 𝒜 \mathcal{A} . In autonomous driving, actions include turning, accelerating, decelerating, or maintaining course.

 
 
 

#### Environment

 
 The environment ℰ \mathcal{E} is the scenario in which agents operate, such as a street with cars and pedestrians. It also includes other relevant elements, like traffic lights and signals.

 
 
 

#### State

 
 The state 𝐬 \mathbf{s} encodes the environment’s conditions, such as ego car positions or traffic light colors. Based on the agent’s action, the environment states are updated: for instance, if the agent turns right at time t t ( a t a_{t} ), the resulting state at t + 1 t+1 is 𝐬 t + 1 \mathbf{s}_{t+1} .

 
 
 

#### Reward

 
 The reward r r measures the success of an agent’s action, provided by the environment. For example, the ego car receives a positive reward for moving closer to its destination and a negative reward for collisions. The reward for action a t a_{t} at time t t is denoted r t r_{t} .

 
 
 

#### Observations

 
 An observation 𝐨 \mathbf{o} is the information an agent receives from the environment at a given time step to make decisions. It may not fully reflect the true state, for example, due to sensor limitations.

 
 
 

#### Return

 
 When selecting an action a t a_{t} , the ego agent considers not only the immediate reward r t r_{t} but also future rewards. This is captured by the return ℛ \mathcal{R} , the cumulative reward from time t t onward: ℛ t = ∑ i = t ∞ r i \mathcal{R}_{t}=\sum_{i=t}^{\infty}r_{i} . To prioritize short-term objectives, a discount factor γ \gamma ( 0 ≤ γ ≤ 1 0\leq\gamma\leq 1 ) is applied, yielding the discounted return: ℛ t = ∑ i = t ∞ γ i ​ r i \mathcal{R}_{t}=\sum_{i=t}^{\infty}\gamma^{i}r_{i} . For example, in autonomous driving, this encourages the agent to reach its destination efficiently while accounting for future rewards.

 
 
 

#### Q-function

 
 The agent aims to maximize return when choosing an action, but the reward from taking action a a in state 𝐬 t \mathbf{s}_{t} is typically unknown beforehand. It can only estimate the expected return using a Q-function: Q ( 𝐬 t , a ) = 𝔼 [ ℛ t ∣ 𝐬 t , a ] Q(\mathbf{s}_{t},a)=\mathbb{E}[\mathcal{R}_{t}\mid\mathbf{s}_{t},a] , representing the expected cumulative reward starting from state 𝐬 t \mathbf{s}_{t} and action a ∈ 𝒜 ⁡ ( 𝐬 t ) a\in\mathcal{A}(\mathbf{s}_{t}) .

 
 
 

#### Policy

 
 The policy π \pi is the strategy (or behavior) adopted by the agent to infer the best action to take at its state 𝐬 \mathbf{s} .
For example, a policy might require halting the ego car whenever the agent observes a red light at a crossing.
The policy that leads the agent to choose the action a i ∈ 𝒜 ⁡ ( 𝐬 t ) a_{i}\in\mathcal{A}(\mathbf{s}_{t}) that maximizes the Q-function is called the optimal policy and is denoted by π ∗ \pi^{*} .

 
 
 

#### Value function

 
 The value function 𝒱 \mathcal{V} predicts the expected future reward from state 𝐬 t \mathbf{s}_{t} following policy π \pi , and thus evaluates its effectiveness: 𝒱 π ​ ( 𝐬 ) = 𝔼 ​ π ​ [ ℛ t ∣ 𝐬 t ] \mathcal{V}_{\pi}(\mathbf{s})=\mathbb{E}{\pi}[\mathcal{R}_{t}\mid\mathbf{s}_{t}] . It relates to the Q-function as 𝒱 ​ π ​ ( 𝐬 ) = ∑ a ∈ 𝒜 ⁡ ( 𝐬 t ) π ⁡ ( a ∣ 𝐬 t ) ​ Q π ​ ( 𝐬 , a ) \mathcal{V}\pi(\mathbf{s})=\sum_{a\in\mathcal{A}(\mathbf{s}_{t})}\pi(a\mid\mathbf{s}_{t})Q_{\pi}(\mathbf{s},a) , i.e. the weighted sum of Q-values for all possible actions according to the policy.

 
 
 TABLE I: Summary of the notation and abbreviations used throughout the paper. 
 
 
 
 Notations | 
 Descriptions | 

 
 RL Symbols | 
 | 

 
 a a | 
 Action | 

 
 ℰ \mathcal{E} | 
 Environment | 

 
 𝐬 \mathbf{s} | 
 State | 

 
 r r | 
 Reward | 

 
 𝐨 \mathbf{o} | 
 Observations | 

 
 ℛ \mathcal{R} | 
 Return | 

 
 γ \gamma | 
 Discount factor | 

 
 π \pi | 
 Policy | 

 
 𝒱 \mathcal{V} | 
 Value function | 

 
 ℳ \mathcal{M} | 
 Model | 

 
 Attack Violations | 
 | 

 
 c ⁡ ( x ) c({x}) | 
 The attacker changes some inputs x x | 

 
 m ⁡ ( x ) m({x}) | 
 The attacker monitors some inputs/outputs x x | 

 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 Minimize the return | 

 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 Make ℰ \mathcal{E} reach a target state 𝐬 T \mathbf{s}^{T} | 

 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 Make the agent learn a target policy | 

 
 learn ​ ( π U ) \text{learn}({\pi}^{U}) | 
 Make the agent deviate from intented policy | 

 
 learn ​ ( π B ) \text{learn}({\pi}^{B}) | 
 Make the agent learn a policy containing a backdoor | 

 
 learn ​ ( π 𝐬 T ) \text{learn}({\pi}_{\mathbf{s}}^{T}) | 
 Make the agent learn to use a target policy | 

 
 | 
 for a subset of observations | 

 
 steal ​ ( ℳ ) \text{steal}({\mathcal{M}}) | 
 Attacker copies model ℳ \mathcal{M} without consent | 

 
 ∉ ℰ \notin\mathcal{E} | 
 Attack uses agent that is not part of the environment | 

 
 ∈ ℰ \in\mathcal{E} | 
 Attack’s agent is within the environment | 

 
 ≈ ℳ \approx\mathcal{M} | 
 Attack’s agent is a surrogate (copy of the target) agent | 

 
 Defense Methods | 
 | 

 
 det. | 
 Detection | 

 
 san. | 
 Sanitization | 

 
 adv. tr. | 
 Adversarial training | 

 
 game th. | 
 Game Theory | 

 
 dist. | 
 Distillation | 

 
 reg. | 
 Regularization | 

 
 ens. | 
 Ensemble | 

 
 
 
 
 

### II-B Solving the Reinforcement Learning Problem 

 
 The problem of teaching the agent to accomplish a certain task can thus be formulated as the problem of finding an optimal policy.
The general formulation treats the problem as a Markov decision process (MDP) . An MDP is a reinforcement learning task that satisfies the Markov property , i.e., the current state retains all relevant information from the past states. Formally, a state 𝐬 t \mathbf{s}_{t} is Markov if and only if ℙ [ 𝐬 t + 1 | 𝐬 t ] = ℙ [ 𝐬 t + 1 | 𝐬 1 , … , 𝐬 t ] \mathbb{P}[\mathbf{s}_{t+1}|\mathbf{s}_{t}]=\mathbb{P}[\mathbf{s}_{t+1}|\mathbf{s}_{1},\dots,\mathbf{s}_{t}] . In simpler words, the environment response at time t + 1 t+1 depends only on the state 𝐬 t \mathbf{s}_{t} and action a t a_{t} (state and action at time t t ), independently of how the past history of states and actions led to 𝐬 t \mathbf{s}_{t} .
If the state and action spaces are finite, the problem is called a finite Markov decision process (finite MDP).
The Bellman equation for 𝒱 π \mathcal{V}_{\pi} expresses the relationship between the value of a state and the values of its successor states. It is expressed as a recursive function:

 

 
 | 
 𝒱 π ( 𝐬 t ) = ∑ a ∈ 𝒜 ⁡ ( 𝐬 t ) π ( a | 𝐬 t ) ∑ 𝐬 ′ , r p ( 𝐬 ′ , r | 𝐬 t , a ) [ r + γ 𝒱 π ( 𝐬 ′ ) ] , \mathcal{V}_{\pi}(\mathbf{s}_{t})=\sum_{a\in\mathcal{A}(\mathbf{s}_{t})}\pi(a|\mathbf{s}_{t})\sum_{\mathbf{s}^{\prime},r}p(\mathbf{s}^{\prime},r|\mathbf{s}_{t},a)[r+\gamma\mathcal{V}_{\pi}(\mathbf{s}^{\prime})], | 
 | 
 (1) | 
 

 and can be interpreted as a sum over all the values of the three variables; a a , 𝐬 \mathbf{s} , and r r , where 𝐬 ′ \mathbf{s}^{\prime} denotes the states that can be reached from state 𝐬 t \mathbf{s}_{t} . For each set of these values, we compute the probability: π ( a | 𝐬 t ) p ( 𝐬 ′ , r | 𝐬 t , a ) \pi(a|\mathbf{s}_{t})p(\mathbf{s}^{\prime},r|\mathbf{s}_{t},a) , and weight the reward expected along the future states by these probabilities.
Solving a reinforcement learning task means finding a policy that achieves the highest reward over time.
There is always at least one policy that is better than or equal to all other policies, and it is called the optimal policy π ∗ \pi^{*} .
The Bellman optimality equation indicates that the value of a state under an optimal policy must equal the expected return for the best action from that state:

 

 
 | 
 𝒱 π ∗ ( 𝐬 t ) = max π 𝒱 π ( 𝐬 t ) = max a ∈ 𝒜 ⁡ ( 𝐬 t ) ∑ 𝐬 ′ , r p ( 𝐬 ′ , r | 𝐬 t , a ) [ r + γ 𝒱 ∗ ( 𝐬 ′ ) ] . \mathcal{V}_{\pi}^{*}(\mathbf{s}_{t})=\max_{\pi}\mathcal{V}_{\pi}(\mathbf{s}_{t})=\max_{a\in\mathcal{A}(\mathbf{s}_{t})}\sum_{\mathbf{s}^{\prime},r}p(\mathbf{s}^{\prime},r|\mathbf{s}_{t},a)[r+\gamma\mathcal{V}^{*}(\mathbf{s}^{\prime})]. | 
 | 
 (2) | 
 

 
 
 This equation, if solved explicitly, gives the optimal policy and thus provides the solution to the reinforcement learning task.
Once the optimal value function 𝒱 ∗ \mathcal{V}^{*} is known, the optimal policy can be found by performing a one-step-ahead search, i.e., by comparing all the values of the states reached by the actions available in the state 𝐬 t \mathbf{s}_{t} , and picking the one that maximizes it (i.e., the one with the optimal value 𝒱 ∗ ​ ( 𝐬 t + 1 ) \mathcal{V}^{*}(\mathbf{s}_{t+1}) ).
Having the optimal Q-function Q ∗ Q^{*} makes it even easier to choose the optimal functions.
Once we know Q ∗ Q^{*} , the optimal policy can be found by assigning non-zero probabilities only to the actions that have the maximum Q-function ( Q ∗ ​ ( 𝐬 t , a ) Q^{*}(\mathbf{s}_{t},a) ).
At this point, the utility of the value and Q-function should become clear; in particular, they allow the expression of the optimal long-term returns as quantities available locally and immediately in each state. 

 
 
 Approximations. For finite MDPs, the Bellman optimality equation can be solved by defining a system of equations with one equation for each state.
For non-finite MDPs, however, the solution makes certain assumptions that may not be practical.
For example, (i) the environment is not perfectly known in its dynamics; (ii) the problem cannot be solved with the available computational resources; and (iii) the problem does not satisfy the Markov property.
The second is the most problematic. In real scenarios, indeed, the number of states might be intractable (or infinite). For example, the number of states in the game of Backgammon is about 10 20 10^{20} , and in autonomous driving it is usually infinite.
This makes the MDP problem computationally unsolvable.
Typically, in these cases, the problem is simplified with approximations that reduce the search space  [ 18 ] .
MDPs are, therefore, solved with different learning algorithms that approximate the value and the Q-functions
and attempt to generalize to unseen states
Approximations are typically performed using linear combinations of features, deep neural networks, and other functions, whereas generalization is done by observing similarities from states seen in training. Other simplifications are performed, for example, by setting an upper bound to the maximum number of states, i.e., limiting the number of RL steps for which the agent can perform actions.
A taxonomy for these algorithms will be detailed in Section  II-C .

 
 
 

### II-C Categorization of Reinforcement Learning Systems 

 
 In this section, we explain how state-of-the-art RL algorithms can be classified based on specific features, followed by a discussion on exemplary approaches for each category.

 
 
 Single-Agent vs Multi-Agent RL. RL environments can incorporate a single-agent (SARL) or multi-agents (MARL). In SARL, one agent learns a policy to interact with the environment. In MARL, multiple RL agents, each with their own policy and objectives, interact with a shared environment and potentially with each other. This makes the system more complex as agents’ actions influence the state and rewards of others. MARL rewards can be cooperative (shared goal), competitive (opposing goals), or mixed-sum (combining both). In autonomous driving, MARL often models cooperative behaviors, such as minimizing traffic congestion or avoiding collisions  [ 19 ] , though mixed-sum setups may apply in special cases, e.g. when regular vehicles must yield to ambulances while still pursuing their own objectives.
We note that, in this work, only systems with RL-based agents are considered MARL; other entities like pedestrians or human-driven cars are not counted as agents.

 
 
 Model-based vs Model-free RL. Reinforcement learning (RL) can be model-based or model-free [ 20 ] . In model-based RL, the agent learns or simulates an explicit model of the environment’s dynamics, denoted ℳ \mathcal{M} , including a transition function (how actions change states) and a reward function (expected rewards for state-action pairs), enabling planning and foresight. For example, tabular certainty-equivalence (TCE) [ 21 ] models the environment as a Markov decision process with known transitions and rewards. In contrast, model-free RL learns directly from interactions with the environment, without modeling its dynamics explicitly. For example, methods like Deep Q-Networks (DQN)  [ 22 ] or Proximal Policy Optimization (PPO)  [ 24 , 25 ] estimate action values or policies from experience, implicitly capturing environment transitions. Model-based RL allows planning but incurs higher sample complexity, computational cost, and risk of model inaccuracies, which is especially challenging in complex domains like autonomous driving. Consequently, most state-of-the-art RL approaches are model-free.

 
 
 On-policy vs Off-policy vs Offline RL Depending on how the RL agent learns a policy from its environment during the training phase, the RL methods can be classified as a) on-policy, b) off-policy and, c) offline RL methods. In on-policy methods, the target policy being learned is the same as the behavior policy used to collect data, simultaneously exploring and updating the policy, as in PPO  [ 24 , 25 ] . Off-policy methods learn the value of a target policy using data collected by a different behavior policy, allowing reuse of past interactions, as in DQN  [ 22 , 23 ] . Offline RL learns entirely from pre-collected static datasets without environment interaction  [ 26 ] . Most RL algorithms in this survey are on- or off-policy, except the planning-based offline agent studied by  [ 27 ] . Due to the lack of evaluation benchmarks for offline RL, it is, in fact, the least used, although it is considered promising as it allows one to take advantage of large existing datasets  [ 28 ] .

 
 
 
 

## III Reinforcement Learning for Autonomous Driving 

 
 One widely used application area for RL is autonomous driving.
We first link the RL categorization to autonomous driving tasks (Sec. III-A ), then discuss car components to show how RL is applied and interacts with them (Sec. III-B ).

 
 

### III-A Driving Automation and Reinforcement Learning Approaches 

 
 Previously, we categorized RL systems by agent type, model type, and policy update method. To relate these to autonomous driving, we review the six Society of Automotive Engineers (SAE) automation levels  [ 29 ] .
We will start with the lowest level and progress towards more autonomy.

 
 
 
 Level 0. No automation, but automatic systems may provide warnings or quick assistance. Examples are lane departure warnings or emergency braking systems.

 

 
 
 Level 1. The driver monitors the environment and controls the car, but the system supports the driver by either breaking or accelerating. Examples include either lane centering or cruise control at a time.

 

 
 
 Level 2. The driver monitors the environment and controls the car, but the system supports the driver by breaking and accelerating. Examples include lane centering and simultaneous cruise control.

 

 
 
 Level 3. The system controls the car, but the driver may be requested to take over control at any time. Examples include a traffic jam chauffeur.

 

 
 
 Level 4. The system controls the car without any intervention by a human, but it can only drive in particular areas. Examples include a driverless taxi restricted to a small geographic area, such as an airport.

 

 
 
 Level 5. The system autonomously controls the functioning of the car without human intervention.

 

 
 
 
 Automation level directly impacts RL application, particularly the agent’s action set. Higher levels (4–5) often involve multi-agent settings, where agents collaborate, e.g., exchanging position information for safe driving. Lower levels (0–3) focus on single-agent tasks, such as lane keeping, with limited interaction.
Environmental complexity also affects the choice between model-based and model-free RL. High automation increases environmental complexity, making explicit modeling difficult; thus, model-free approaches are typically favored.
Policy update strategy depends on safety considerations. Offline RL trains on pre-collected data to avoid unsafe actions but can limit exploration outside training distribution. However, on- and off-policy training can still be trained in virtual environments. In deployed vehicles, mature policies may be periodically updated by manufacturers, as seen in commercial systems like Tesla  2 2 
 2 
 
 
 
 https://www.tesla.com/support/software-updates .

 
 
 

### III-B Autonomous Driving Components and Reinforcement Learning 

 
 The RL agent should not be seen in isolation from the vehicle it controls: to interface with full (or even partial) automation, different types of sensors and components are required. Previous work describes these components at different levels of abstraction  [ 30 , 31 , 32 ] .
We chose the three-level abstraction structure by Pendleton et al.  [ 33 ] , which divides the information flow through the vehicular agent. More specifically, the information is first perceived from the environment: perception , which are then used to derive state-actions pairs: planning , and finally carry out the identified action, control .
We now review each of these steps through the lens of reinforcement learning. We illustrate the main components of the abstraction in Figure  1 .

 
 
 Fig. 1: Major Components of Autonomous Driving Systems. 
 
 
 Perception. As a first step, the car must be able to perceive its environment as a basis for any future decision or output. This may include the localization of the car, but mainly relates to the perception of the environment, generally based on sensors.
Some of the commonly used sensors  [ 34 ] are: LiDAR, with a range of up to 200 meters often used for collision avoidance. Another sensor similar to LiDAR is Radar with a similar range, and it may also be used for collision detection in the closer vicinity of the vehicle.
Both inputs can be represented, for example, as point clouds. Other sensors include cameras for object detection, which typically cover a range of up to 100 meters and produce images or videos.
Ultrasonic sensors have a short range and are often used for parking assistance.
Several sensors are often used in conjunction for redundancy and thus increase the reliability of collected inputs.
In addition to mere sensing, this layer also incorporates tasks such as object detection, depth estimation, and semantic segmentation, where inputs are already processed, resulting in aggregated information.

 
 
 Planning. 
Receiving the previously-collected inputs and inputs from other vehicles (in case of MARL), an RL agent can be trained to plan actions to achieve the final goal. Depending on the goal, the agent may need to perform route planning : planning which roads to be taken to reach the destination, behavioral planning : how to interact with other agents in the vicinity while following rules restrictions, motion planning : generate location objectives and determine actions to achieve them.
In the following sections, we will discuss tasks such as deriving the prediction of the steering output from input images  [ 35 , 36 , 37 , 38 ] , traffic simulation  [ 19 ] , and path finding  [ 21 ] .
Depending on the task, the requirements of an RL agent may vary greatly. In one case, an ego agent can maneuver a car on a highway where different non-autonomous vehicles are present. In this case, the RL system must be guaranteed to operate in real-time .
However, in another case, small delays may be more acceptable for example when an agent operating in an area void of humans; both pedestrians and drivers or in MARL settings, where multiple agents can act simultaneously or take into account delays of other vehicles within the model.

 
 
 Control. 
Once target actions have been identified, the agent needs to execute them, basically converting the intentions into execution, to control vehicle movements. This step involves communicating to the actuators the signals to perform the required actions.
Depending on how the actions are encoded, they might be low-level actions i.e., the output is directly the input to the actuators, or high-level actions, e.g., “change lane”, which is then performed with a sequence of predefined low-level commands.

 
 
 Further Components. 
There are other parts of the autonomous vehicle that we skipped previously for simplicity. For example, the application of sensors requires software. To ensure efficient and reliable communication between individual electronic control units, each vehicle contains a broadcast communication system: Controller Area Network (CAN) buses. Another example is the layers needed to translate the output of the RL agent to the autonomous vehicle. Each component may include a security vulnerability.
As we will see in the next sections, some attacks assume not to alter the environment but the information provided to the RL agent  [ 23 , 37 , 39 ] .

 
 
 While the perception components are vulnerable to specific adversarial attacks  [ 40 , 41 ] that target them independently of the underlying system, for the sake of this survey, we will focus on attacks targeting the RL component of the system specifically. Similarly, for technical papers dealing with the security of control components individually, we defer the reader to existing related work  [ 42 ] .

 
 
 
 

## IV Threat Model 

 
 In this section, we describe a threat model to characterize attacks against RL.
We characterize the attack according to (1) the goal of the attacker, (2) the knowledge the attacker has about the target system, and (3) the attacker’s capabilities of manipulating input data and/or tampering with the system components.

 
 

### IV-A Goal 

 
 The attacker’s goal is the result they would like to obtain by perpetrating their attack. The goals can be broadly categorized according to the type of security violation they cause: a) integrity violation, i.e., to influence the agent to perform an unintended action without compromising system functioning. For example, it can force an autonomous vehicular agent to steer the car in the wrong direction when making a turn. b) availability violation, i.e. when the legitimate functioning of the agent is compromised. For example, an attacker can alter a portion of the training data, making the agent unable to learn to perform certain tasks, and finally, c) privacy violation by obtaining private information about the system or the data used to train it by reverse-engineering the learning algorithm.

 
 
 

### IV-B Capabilities 

 
 Attackers may have different capabilities that they can leverage to achieve their goals.
Firstly, they may or may not have the ability to alter training data or the learning process.
If the attacker can modify what the target agent learns then the attack is defined as a training-time attack.
Attackers can also alter the code of the RL algorithm that is used to train the agent, as previously done in  [ 43 ] .
Otherwise, if the attacker can act only against an already trained agent, it is defined as a test-time attack.

 
 
 

### IV-C Action 

 
 Depending on the threat model, an attacker can either change ( c ⁡ ( x ) c({x}) ) or monitor ( m ⁡ ( x ) m({x}) ) the target system’s inputs and outputs. The attacker has higher flexibility when changing the input to perform a desired targeted attack. Since the agent’s observations drive both training and inference, attackers can perturb the environment ( c ⁡ ( ℰ ) c({\mathcal{E}}) ) by adding objects or introducing malicious agents, alter specific states ( c ⁡ ( 𝐬 ) c({\mathbf{s}}) ), or tamper with sensory inputs ( c ⁡ ( 𝐨 ) c({\mathbf{o}}) ) to modify the agent’s perception. They can also manipulate actuators ( c ⁡ ( a ) c({a}) ) such that action performed differs from those chosen by the agent, or modify the reward signal ( c ⁡ ( r ) c({r}) ) during training to degrade performance. Depending on the system, attackers may combine multiple capabilities. Figure  2 highlights in red the components that can be manipulated without access to the RL algorithm.

 
 
 

### IV-D Knowledge 

 
 The attacker can have different levels of knowledge of the target system, including (i) the specifics of the model (e.g., the type of observations or actions, the RL algorithm used, the reward function), (ii) the training data, (iii) the internal parameters or values which might include the models’ weights or the learned policy, the values of the observations, actions, or reward at each step. If attackers have full knowledge of the system, the attack is termed a white-box . Though unrealistic in practice, this setting enables worst-case evaluations and provides upper bounds on performance degradation. In a gray-box setting, adversaries have partial knowledge—for instance, they may observe states and rewards but lack details of the RL algorithm. Some variants assume delayed knowledge acquisition, where the system state changes before the attacker gains full access. Finally, in a black-box setting, the attacker has no internal knowledge and relies solely on external observations of inputs and outputs, such as inferring sensor values and monitoring agent actions. In gray- and black-box settings, attackers often exploit transferability  [ 44 ] , crafting attacks on a surrogate RL system and deploying them against the target.

 
 
 Leveraging Knowledge for Malicious Goals 
When running attacks against reinforcement learning systems, attackers can leverage their knowledge to RL agents to achieve their malicious goals. There are different ways in which this can happen, as detailed below.

 
 
 
 • 
 
 When attackers do not have perfect knowledge about their target system, they can train a surrogate model to approximate it. In the following, we will denote the surrogate system that approximates the target model by ≈ ℳ \approx\mathcal{M} .

 

 • 
 
 In RL systems, multiple agents can act simultaneously in the environment and thus modify the state as perceived by the victim. There are two ways in which the attacker can deploy the attack in a RL environment:

 
 – 
 
 The attacker can use an agent included in the environment ( ∈ ℰ \in\mathcal{E} ) to carry out the attack.

 

 – 
 
 The attacker can also leverage an agent that is not included in the environment ( ∉ ℰ \notin\mathcal{E} ) to enhance the severity of the attack. For example, the attacker can use it’s external agent to understand how to alter the victim agent’s environment and cause the misbehavior.

 

 
 

 
 In the following, we will use the symbol ∅ \emptyset to denote the case where the attacker does not use any malicious RL agent.

 
 
 Fig. 2: Conceptual representation of the components of an RL policy that the attacker can manipulate without having access to the RL algorithm. 
 
 
 
 

## V Attacks 

 
 Despite its potential, RL is vulnerable to numerous attacks, many adapted from ML. Here, we categorize state-of-the-art RL attacks and present a taxonomy for test-time and training-time attacks based on attacker capabilities. We also review studies in autonomous driving, highlighting assumptions and real-world applicability, and outline future directions for RL-specific attack research.

 
 

### V-A Categorizing the Attacks against RL 

 
 Our taxonomy is designed to guide both researchers and practitioners by highlighting attack characteristics and the RL systems they target. This helps readers identify applicable attacks and potential vulnerabilities before deployment. Attack goals depend on the target system, and factors such as policy update strategy and environment modeling influence attack feasibility. Accordingly, we first identify key characteristics of RL models that are critical for understanding possible attacks.

 
 
 Agent Model. Considered characteristics of the RL agent under attack.

 
 • 
 
 Single vs. Multi Agent : Methods are evaluated in either SARL or MARL settings, depending on the number of agents. MARL represents a more realistic but challenging scenario for autonomous driving;

 

 • 
 
 Target Policy Update. Frequency of updating the RL policy (on-policy, off-policy, offline). This determines whether training-time attacks can have an immediate impact;

 

 • 
 
 Model-based. Indicates whether the RL system models the environment. If true, attacks can potentially target its ability to predict environment dynamics; otherwise, we mark it as false.

 

 
 
 
 Threat Model. Considered capabilities that the attackers have to perpetrate the attack. It is subdivided into:

 
 • 
 
 Attack’s Time : Whether the attack is planted at training time or test time.

 

 • 
 
 Attacker’s Action : What action must be performed by the attacker, and, as we discussed, could be to monitor the inputs of the RL system or to alter them.

 

 • 
 
 Poisoning : The eventual ability of the attackers to alter the input during training, which can be true if they have this capability and false otherwise.

 

 • 
 
 Attacker’s Knowledge. Level of knowledge of the victim system, which may be white-, gray-, and black-box;

 

 • 
 
 Attacker’s goal. The result the attackers aim to obtain with their attack;

 

 • 
 
 Sequential Attack. The eventual temporal dependence of the attack on the current or previous state. Sequential attacks are more complex to realize as the attacker should know or predict the states that will occur earlier or later than the state in which the attack happens;

 

 • 
 
 Attacker’s Agent. Whether and how the attackers use an RL agent to carry out the attack. The attacks where the agent exploits other agents are usually more complex to implement.

 

 
 
 
 TABLE II: The objectives of the state-of-the-art attack against RL, categorized according to the attackers’ capability required to carry out them (the ability to change only the test data or also the training data), and the security violation they cause. 
 
 
 
 | 
 Integrity | 
 Availability | 
 Privacy | 

 
 Test Data | 
 | 
 | 
 | 

 
 - reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 | 
 - steal ​ ( ℳ ) \text{steal}({\mathcal{M}}) | 

 
 - min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 | 

 
 Training Data | 
 | 
 | 
 | 

 
 - learn ​ ( π B ) \text{learn}({\pi}^{B}) | 
 - learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 | 

 
 - learn ​ ( π 𝐬 T ) \text{learn}({\pi}_{\mathbf{s}}^{T}) | 
 - learn ​ ( π U ) \text{learn}({\pi}^{U}) | 
 | 

 
 
 
 Table  II categorizes attacks by the type of security violation: integrity, availability, or privacy, caused at train or test time, described further in detail in the next two sections. Additionally, Table III provides a schematic overview of 50 attacks proposed lately.

 
 
 TABLE III: Attacks against reinforcement learning systems. The presence of the ✓indicates that the corresponding attribute is true for the attack. For the attacker’s knowledge, we use ● to represent black-box, ◐ for gray-box, and ○ for white-box. Refer Table I for attacker’s agent in the system and Table II for attacker’s goal.
 
 
 
 
 | 
 AGENT MODEL | 
 THREAT MODEL | 

 
 References | 
 Single vs. | 
 Policy | 
 Model | 
 Attack’s | 
 Attacker’s | 
 Poisoning | 
 Attacker’s | 
 Attacker’s | 
 Sequential | 
 Attacker’s | 

 
 | 
 Multi-agent | 
 Update | 
 Based | 
 Time | 
 Action | 
 | 
 Knowledge | 
 Goal | 
 Attack | 
 Agent | 

 
 Ma et al. [ 21 ] | 
 SARL | 
 on-policy | 
 ✓ | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ○ | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 | 
 ≈ ℳ \approx\mathcal{M} | 

 
 Zhao et al. [ 23 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Rakhsha et al. [ 27 ] | 
 SARL | 
 on-policy, offline | 
 ✓ | 
 t ​ r tr | 
 c ⁡ ( r , a , 𝐬 ) c({r,a,\mathbf{s}}) | 
 ✓ | 
 ○ | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Yang et al. [ 35 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Yang et al. [ 35 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ≈ ℳ \approx\mathcal{M} | 

 
 Boloor et al. [ 36 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ○ | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Xiao et al. [ 38 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○/ ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Xiao et al. [ 38 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( a ) c({a}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ≈ ℳ \approx\mathcal{M} | 

 
 Xiao et al. [ 38 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ● | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 N/A | 
 ∅ \emptyset | 

 
 Mandlekar et al. [ 43 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐬 ) c({\mathbf{s}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Zhang et al. [ 45 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ○ | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 | 
 ∅ \emptyset | 

 
 Huang et al. [ 46 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ○/ ● | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Liu et al. [ 47 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( a ) c({a}) | 
 ✓ | 
 ● | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 | 
 ≈ ℳ \approx\mathcal{M} | 

 
 Liu et al. [ 47 ] | 
 SARL | 
 on/off-policy | 
 ✓ | 
 t ​ r tr | 
 c ⁡ ( a ) c({a}) | 
 ✓ | 
 ○ | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 | 
 ∅ \emptyset | 

 
 Rakhsha et al. [ 48 ] | 
 MARL | 
 on-policy | 
 /✓ | 
 t ​ r tr | 
 c ⁡ ( a ) c({a}) | 
 ✓ | 
 ◐/ ● | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Majadas et al. [ 49 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ● | 
 learn ​ ( π U ) \text{learn}({\pi}^{U}) | 
 | 
 ∅ \emptyset | 

 
 Kiourti et al. [ 50 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( 𝐨 , r ) c({\mathbf{o},r}) | 
 ✓ | 
 ○ | 
 learn ​ ( π B ) \text{learn}({\pi}^{B}) | 
 | 
 ∅ \emptyset | 

 
 Ashcraft et al. [ 51 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( ℰ , r ) c({\mathcal{E},r}) | 
 ✓ | 
 ● | 
 learn ​ ( π B ) \text{learn}({\pi}^{B}) | 
 | 
 ∅ \emptyset | 

 
 Foley et al. [ 52 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 ✓ | 
 ○ | 
 learn ​ ( π 𝐬 T ) \text{learn}({\pi}_{\mathbf{s}}^{T}) | 
 | 
 ∅ \emptyset | 

 
 Ma et al. [ 53 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ○ | 
 learn ​ ( π 𝐬 T ) \text{learn}({\pi}_{\mathbf{s}}^{T}) | 
 | 
 ∅ \emptyset | 

 
 Xu et al. [ 54 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 ✓ | 
 ○/ ● | 
 learn ​ ( π 𝐬 T ) \text{learn}({\pi}_{\mathbf{s}}^{T}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Zhang et al. [ 55 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Tanev et al. [ 56 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Lee et al. [ 57 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( a ) c({a}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Lee et al. [ 57 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( a ) c({a}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Huang et al. [ 58 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ○/ ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Buddareddygari et al. [ 59 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ○ | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 | 
 ∅ \emptyset | 

 
 Wang et al. [ 19 ] | 
 MARL | 
 off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ● | 
 learn ​ ( π B ) \text{learn}({\pi}^{B}) | 
 | 
 ∈ ℰ \in\mathcal{E} | 

 
 Gleave et al. [ 25 ] | 
 MARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ◐ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∈ ℰ \in\mathcal{E} | 

 
 Sun et al. [ 37 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Sun et al. [ 37 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ◐ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Tretschk et al. [ 39 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Wang et al. [ 60 ] | 
 MARL | 
 on-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 ✓ | 
 ○ | 
 learn ​ ( π B ) \text{learn}({\pi}^{B}) | 
 ✓ | 
 ∈ ℰ \in\mathcal{E} | 

 
 Yu et al. [ 61 ] | 
 MARL | 
 off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ○ | 
 learn ​ ( π B ) \text{learn}({\pi}^{B}) | 
 ✓ | 
 ∈ ℰ \in\mathcal{E} | 

 
 Behzadan et al. [ 62 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ r tr | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 ✓ | 
 ◐ | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 | 
 ≈ ℳ \approx\mathcal{M} | 

 
 Lin et al. [ 63 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Lin et al. [ 63 ] | 
 SARL | 
 off-policy | 
 ✓ | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Tekgul et al. [ 64 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Russo et al. [ 65 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Kos et al. [ 66 ] | 
 MARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Sun et al. [ 67 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Korkmaz et al. [ 68 ] | 
 SARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐬 ) c({\mathbf{s}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Inkawhich et al. [ 69 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ≈ ℳ \approx\mathcal{M} | 

 
 Chen et al. [ 70 ] | 
 SARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 m ⁡ ( a ) m({a}) | 
 | 
 ● | 
 steal ​ ( ℳ ) \text{steal}({\mathcal{M}}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Yoon et al.  [ 71 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐬 ) c({\mathbf{s}}) | 
 | 
 ◐ | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 ✓ | 
 ∈ ℰ \in\mathcal{E} | 

 
 Fan et al.  [ 72 ] | 
 SARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 ✓ | 
 ∈ ℰ \in\mathcal{E} | 

 
 Li et al.  [ 73 ] | 
 MARL | 
 on/off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∈ ℰ \in\mathcal{E} | 

 
 Bai et al.  [ 74 ] | 
 SARL | 
 off-policy/offline | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐬 ) c({\mathbf{s}}) | 
 | 
 ◐ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Zheng et al.  [ 75 ] | 
 SARL / MARL | 
 on-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∉ ℰ \notin\mathcal{E} / ∈ ℰ \in\mathcal{E} | 

 
 Zhou et al.  [ 76 ] | 
 MARL | 
 off-policy | 
 | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○/ ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∉ ℰ \notin\mathcal{E} | 

 
 
 
 

### V-B Test Time Attacks 

 
 The attacks that manipulate only test data cause either integrity or privacy violations.

 
 
 Integrity Violations. These attacks refer to an adversarial intervention at deployment (inference) time that causes the agent to make incorrect or unsafe decisions, violating the intended integrity of its policy, without altering its training process. Integrity violations can be:

 
 • 
 
 Target-State Manipulation Attack: The victim agent in the environment is perturbed by the attacker to reach a certain malicious target state: reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) . In other words, steering the agent to a specific attacker-chosen target state, e.g., driving to a specific unintended location such as the dead end, enemy base, etc. The agent may still think it’s acting optimally, but is being misled.

 

 • 
 
 Reward Minimization Attack: the victim agent is perturbed to perform an action that minimizes total return: min ⁡ ( ℛ ) \min({\mathcal{R}}) , which potentially leads to unsafe and erratic behavior; e.g., veering the vehicle off the road or crashing into the guardrails. These attacks aim to make the agent fail its own reward function.

 

 
 To perform these attacks, the attacker should have the ability to change (c) some test inputs.
The majority of proposed attacks are Reward Minimization Attacks that alter a component of the system of an ego agent: state, action, environment, or observations, to minimize expected reward  [ 23 , 77 , 65 , 58 , 56 , 43 , 57 , 35 , 55 , 63 , 69 , 73 , 74 , 75 , 76 ] .
The attacker often modifies a system component only slightly, applying a perturbation bounded with an l p − l_{p}- norm constraint [ 78 , 56 , 55 ] , making the attack difficult to detect. However, in  [ 65 ] , the authors devised “optimal attacks” by formulating the problem as MDP from the attacker’s perspective. The attack is trained using the Deep Deterministic Policy Gradient (DDPG) with a reward function opposite of the agent to optimally undermine the agent’s reward at test time.
Alternatively, the authors of [ 56 ] exploit adversarial patches [ 79 ] , that is, small stickers containing evident perturbations.
The advantage of adversarial patches is that they can be printed and used to alter the environment; therefore, attackers can perpetrate this attack even if they do not have access to the sensors.
The authors of [ 56 ] , considering a system trained to grasp objects based on visual input, show that the agent is unable to perform its task when the patch is present.

 
 
 Few works alter a component of the system to cause the environment to induce the Target-State Manipulation Attack [ 38 , 36 , 59 , 37 , 39 , 63 , 71 , 72 ] . For example, in  [ 39 ] , the attacker misguides the agent towards a different positive reward, rather than simply reducing its performance or making it fail. One of the attacks proposed in  [ 63 ] : Enchanting Attack, where the attacker focuses on luring the DRL agent to a specific, pre-determined target state using the generative model and planning algorithm that generates a sequence of actions to guide the agent towards the target.

 
 
 In [ 64 ] , the authors claim that test-time attacks against DRL, to be practical, should work without considering the current state of the system.
The reason is that pre-computing an adversarial example for each possible state is not feasible in practice, as the possible states are often too many, and the attacker would not be able to compute the attack for the current state before the state is already changed.
To solve this problem, they consider universal adversarial examples  [ 80 ] that can be computed offline and are effective for different states.
Only the work in  [ 63 ] proposes test-time attacks that perturb the environment to reach the desired state of the attacker.

 
 
 When it comes to multi-agent systems, the number of attack algorithms is quite low. Even though some SARL attacks can be extended in MARL settings, where each agent is attacked individually, they don’t necessarily consider the interaction among the agents.
The existing MARL attacks either manipulate some system components to minimize the expected reward  [ 25 , 66 , 73 , 75 , 76 ] or induce a backdoor  [ 60 ] .
The authors of the work in  [ 25 ] consider a two-player competitive RL system and exploit the behavior of an agent within the environment (the opponent) to make the ego agent unable to win the match using an adversarial policy. One of the two attacks in  [ 37 ] is Antagonist Attack, which is domain-agnostic and general enough to be applied to a multi-agent environment; however, the evaluation is done single-agent environment. The attack proposed in  [ 75 ] addresses both SARL and MARL, where the attacker directly injects perturbations into the victim policy’s inputs, while in multi-agent settings, the adversary controls an opponent agent to indirectly influence the victim’s observations. Moreover, in  [ 73 ] , the attacker introduces a single adversarial agent into the cooperative MARL environment. This adversarial agent learns a policy that allows it to execute physically plausible actions that directly influence the observations of the victim agents. The attack is designed to be“unilateral” meaning the adversary influences the victims without being unduly influenced by them in return.

 
 
 Privacy Violations. The attacks that cause privacy violations proposed so far monitor (m) the victim’s behavior to clone its policy.
This attack is called model stealing ( steal ​ ( ℳ ) \text{steal}({\mathcal{M}}) )  [ 81 ] .
The only work in this direction is  [ 70 ] , where the authors proposed an attack that aims to extract a proprietary DRL model, meaning to recover it with high fidelity and accuracy, based only on observing its actions in an environment with black-box access.
In particular, the authors of this paper first identify the training algorithm family and then perform model extraction using imitation learning.
Imitation learning is a well-established solution to learn sequential decision-making policies.
The desiderata for the copied model can be (1) accuracy, namely the ability to match or exceed the accuracy of the original model; (2) fidelity, namely a similar behavior to the original model (even committing the same errors). The authors showed that the proposed attack achieves high accuracy and fidelity.

 
 
 The literature examining the privacy of the model is scarce and calls for more in-depth analysis of RL systems.

 
 
 TABLE IV: Attacks evaluated on autonomous driving tasks. We list the attack, the target component, the attacker’s goal in relation to the autonomous car, and the action taken by the attacker to achieve the goal. In the following columns, further information on if the data samples are poisoned or not, whether it’s a sequential attack and whether the attack is designed for multi-agent or not. /✓indicates both the presence and absence of the feature in that paper.
 ∗ The work by [ 19 ] targets the controller of an autonomous vehicle as a part of a larger system that optimizes traffic flow. 
 
 
 
 References | 
 Target | 
 Attacker’s | 
 Attacker’s | 
 Poisoning | 
 Sequential | 
 Multi-agent | 

 
 | 
 Component | 
 Goal | 
 Action | 
 | 
 Attack | 
 | 

 
 Wang et al. [ 19 ] | 
 controller* | 
 traffic jam/crash | 
 positions and speeds | 
 ✓ | 
 | 
 ✓ | 

 
 Ma et al. [ 21 ] | 
 planning | 
 car is rerouted | 
 training rewards | 
 ✓ | 
 | 
 | 

 
 Yang et al. [ 35 ] | 
 end-to-end | 
 car leaves lane | 
 sensor input / FGSM | 
 | 
 ✓ | 
 | 

 
 Boloor et al. [ 36 ] | 
 end-to-end | 
 car leaves lane | 
 paint pattern on street | 
 | 
 ✓ | 
 | 

 
 Sun et al. [ 37 ] | 
 end-to-end | 
 car leaves lane | 
 sensor environment / FGSM | 
 | 
 ✓ | 
 ✓ | 

 
 Xiao et al. [ 38 ] | 
 end-to-end | 
 suboptimal behavior | 
 paint pattern on street | 
 | 
 /✓ | 
 | 

 
 Buddareddygari et al. [ 59 ] | 
 end-to-end | 
 car leaves lane | 
 add object | 
 | 
 | 
 | 

 
 Yu et al. [ 61 ] | 
 end-to-end | 
 car collision | 
 car behavior | 
 ✓ | 
 ✓ | 
 ✓ | 

 
 Yoon et al. [ 71 ] | 
 end-to-end | 
 car is misguided | 
 sensor input/online perturbations | 
 | 
 ✓ | 
 | 

 
 Fan et al.  [ 72 ] | 
 end-to-end | 
 car collision | 
 sensor input | 
 | 
 ✓ | 
 | 

 
 
 
 

### V-C Training Time Attacks 

 
 To perform these attacks, the attacker must be able to manipulate training data, including observations, environment, or rewards, either before or during learning. Existing training-time attacks target availability or integrity violations.

 
 
 Availability Violations. 
The goal of the attacks not to cause the agent to act maliciously per se, but to disrupt or degrade the ability to learn the legitimate policy of the agent. This can be done by:

 
 • 
 
 Target Manipulation Attack: Forcing the agent to learn an attacker-chosen policy: learn ​ ( π T ) \text{learn}({\pi}^{T}) . The model may still converge but to an unusable or bad policy. For example, the attacker may poison the environment or reward function so that actions taken for left-side driving are consistently rewarded instead of the legit right-side driving. Here, the agent learns a consistent policy but for the wrong traffic system.

 

 • 
 
 Untargeted Policy Attack: Preventing the agent from learning the intended policy without enforcing a specific alternative: learn ​ ( π U ) \text{learn}({\pi}^{U}) . This can be done by altering the reward or overall return.
For instance, altering rewards to favor crashes or off-road driving, thereby making the agent unsafe.

 

 
 
 
 In [ 49 ] , the authors propose a black-box un-targeted policy attack. They show how disturbances in the reward function affect the convergence of a learning agent.
They assume that the attacker does not have any knowledge about the learned policy, but can sometimes flip the sign of the reward when the environment reaches a chosen state. Their work focuses on on-policy RL and analyzes different exploration strategies.
Their experimental analysis shows that small exploration probabilities are more resilient to perturbations of the reward.
In [ 21 , 45 , 46 , 27 ] , the authors alter the reward to force the agent to learn a target policy chosen by the attacker, e.g. to make a robot learn to reach a location chosen by the attacker instead of the location desired by the developer of the RL system.
In [ 62 ] , the authors propose an ad-hoc manipulation of the environment at training time by an adversarial agent. The attacker first trains its own adversarial DQN using the desired adversarial policy π a ​ d ​ v \pi_{adv} and then uses it to generate perturbed sample attacking the environment of the target system. For multi-agent setting, only  [ 48 ] is proposed, where the authors focus on population learner black-box scenario. While each individual learner might be a single agent, the overall attack framework is designed for a multi-agent context where the attacker is trying to manipulate a series of independent learners. The attack itself, though, is model-agnostic, however, the proposed method unfolds in two phases: exploration phase: gathers the information from the environment to estimate a set of plausible ℳ \mathcal{M} which is later exploited in second phase: attack phase, where the attacker uses the estimated model of the environment to determine the optimal reward perturbations, therefore, marked “/✓” for this paper.

 
 
 Integrity Violations. The attacks that cause an integrity violation apply the following tactics:

 
 • 
 
 Backdoor Poisoning Attack: Inserting a trigger pattern into training data learn ​ ( π B ) \text{learn}({\pi}^{B}) so the agent behaves normally except when the trigger appears. For example, an autonomous car may drive correctly but act unpredictably when a stop sign has a yellow sticker.

 

 • 
 
 Targeted Poisoning Attack: Manipulating training so the agent learns a malicious policy only in specific states learn ​ ( π 𝐬 T ) \text{learn}({\pi}_{\mathbf{s}}^{T}) , such as roundabouts or intersections, while behaving normally elsewhere. For instance, the attacker may poison training so the agent consistently exits a roundabout incorrectly, while in other states, the agent has a good driving policy.

 

 
 Training-time integrity attacks are powerful attacks as they behave as expected in most cases and is therefore hard to detect by observing the agent’s behavior. In targeted poisoning attacks, the agent’s behavior is unusual only in the presence of a particular trigger/state. For example, in  [ 54 ] , using carefully designed perturbations of the environment at training time, the attacker forces the victim agent to learn to perform a desired action when the agent is in some specific states and behaves normally otherwise. Similarly, in  [ 53 ] , the authors focus on attacking contextual bandits, a class of RL algorithms that works well to choose actions in dynamic environments where the options change rapidly and the set of available actions is limited.
There exists quite a few works in backdoor poisoning, for example, in  [ 50 ] , the authors propose an attack against DRL that adds training samples containing a particular pattern (a.k.a. trigger) that alters the corresponding reward by assigning the highest reward to random actions.
This makes the agent act randomly every time the trigger is present in the input.
Similarly, in  [ 51 ] , the attacker makes the agent learn to perform a target policy when the trigger is present in the input.
Interestingly, in this paper, the authors propose the concept of in-distribution triggers.
That is, patterns that can occur due to agent interaction with the environment and thus are difficult to detect when used as a trigger.
The authors show that the attack succeeds by adding 10-20% samples containing the trigger.
In multi-agent settings only backdoor attacks have been proposed. In BackdoorRL   [ 60 ] , the victim agent is trained using imitation learning from a mix of “normal” and “backdoored” trajectories. The backdoor is then activated by a series of specific trigger actions performed by the adversary agent.
In  [ 61 ] , the backdoor is triggered by an agent near the victim.
In [ 19 ] , instead, the authors consider a traffic congestion control system, and assume that the attacker uses the state of the car agents in the neighborhood of the ego agent to trigger the backdoor.

 
 
 Fig. 3: Timeline of attacks against reinforcement learning, compared to those proposed against machine-learning classifiers. 
 
 
 

### V-D Attacks against Autonomous Driving 

 
 This section reviews attacks on autonomous vehicles, highlights the properties needed for real-world threats, and discusses current limitations and future research directions.

 
 
 State of the Art. Table  IV summarizes attacks on autonomous driving (AD), with three of eight evaluated in multi-agent settings. While level 5 autonomy requires multi-agent coordination, single-agent attacks remain effective, broadening the threat surface. Most attacks target end-to-end systems at test time, aiming to force lane departures or unsafe maneuvers  [ 35 , 36 , 37 , 38 ] , sometimes via physical perturbations such as painted road lines or printed artifacts. Others exploit sensor inputs, environment responses, or online image streams  [ 71 ] , with some leveraging sparse perturbations to evade detection  [ 72 ] . Training-time threats include policy poisoning, which manipulates rewards to misguide planning  [ 21 ] , and backdoor attacks that target either traffic optimization algorithms  [ 19 ] or vehicle behaviors  [ 61 ] . Collectively, these studies reveal diverse attack surfaces in AD but remain largely simulator-based, highlighting the urgent need for real-world evaluations to assess their practical feasibility and safety implications.

 
 
 Practical Shortcomings. 
While current RL attacks serve as proofs of concept, their impact on real autonomous vehicles remains uncertain. Most are not tested on actual cars, likely due to the high cost of AVs, and only a few consider actions feasible in the physical world  [ 36 , 38 ] . Critical practical aspects, such as computation time for real-time perturbations, attack transferability across models, and the feasibility of sequential versus single-shot attacks, are largely unexplored. Additionally, no studies exploit agents within the system, leaving their effectiveness unclear. Finally, only a minority of attacks operate in realistic black-box settings, limiting the assessment of their applicability to proprietary AV models. These gaps highlight the need for evaluating RL attacks under real-world constraints to understand their true risk in autonomous driving scenarios.

 
 
 

### V-E Adversarial RL Timeline 

 
 The attacks against RL are adaptations of attacks previously proposed against standard ML systems, particularly against classification tasks. In Figure 3 , we highlight the connections between the attacks proposed against standard ML systems and those proposed against RL.
The first attacks against RL were test-time integrity awards [ 63 , 43 ] , proposed in 2017 and were able to make the system misbehave when a carefully crafted sample was provided as input to the system. These attacks are inspired by those [ 83 , 84 ] that were proposed more than ten years before, between 2004 and 2005, against ML classifiers that misclassified carefully crafted adversarial samples. For example, misclassifying a spam email as legitimate. The first training-time poisoning attack against RL was also proposed in 2017 [ 62 ] . The goal of this attack was to make the system unable to work correctly, specifically by manipulating the target policy, causing a denial of service. In this case, the attack is also inspired by an attack [ 85 ] proposed a long time before (2012) against ML classifiers and in particular against the Support Vector Machine, affecting its ability to learn to correctly classify the samples. In 2018 and 2020, researchers proposed the first integrity-violation train-time attacks against RL. In particular, the first paper on targeted poisoning attack  [ 53 ] in 2018 and backdoor poisoning attack  [ 50 ] in 2020 were proposed.
Both were inspired by articles proposed in 2017 [ 86 , 87 ] where former aimed to misclassify test samples by manipulating a few training samples while latter made the classifier learn to misclassify all samples containing a specific trigger. The only attack that violates the privacy of RL systems [ 88 ] performs stealing of the model, was proposed recently in 2021, is inspired by an attack previously proposed against ML classifiers [ 89 ] in 2016. Whereas against ML, also an attack to understand if some data were employed to train the model [ 88 ] and an attack to obtain a copy of the model [ 89 ] were proposed, respectively in 2015 and 2016. The attacks that aim to slow down the system have been proposed against ML systems [ 90 , 91 ] but have never been demonstrated against RL systems.

 
 
 

### V-F Future Research Directions 

 
 While many attacks have been explored against ML classifiers, their applicability to reinforcement learning (RL) remains underexplored. Future research should address security violations in RL by tailoring threat models to application-specific needs, particularly in autonomous driving (AD). Beyond model theft, privacy attacks such as model inversion and membership inference could expose sensitive driving data. For availability, attacks like the sponge attack—where inputs slow system responses without altering training data—pose serious risks for real-time AD decision-making.
AD, as a safety-critical domain, demands realistic attack evaluations that go beyond simulations. Black-box attacks and transferability from simulated to real vehicles require careful study, with explicit reporting of computational costs to assess real-time feasibility. Moreover, attack strategies should reflect realistic adversarial actions and be tested on full AD pipelines, not isolated modules, to evaluate whether sequential or black-box attacks can reliably compromise vehicle security.

 
 
 
 

## VI Defenses 

 
 Various defenses have been proposed to make the RL more robust against attacks. In the following, we first explain how state-of-the-art defenses against RL can be categorized.
We then focus on defenses that are specifically tailored to the autonomous driving environment, even though a single defense of this type has been proposed. Thus, we discuss the open challenges and conclude the section with a review of potential future directions with regards to the development of defenses for RL.

 
 
 All existing RL defense methods have adapted techniques that were previously devised for ML classification tasks.
They can be divided into three main categories: a) those that try to counteract an existing attacks, i.e. reactive defenses, and b) those that act to prevent future attacks, i.e. proactive defenses  [ 17 ] , and c) those that detect the attack and provide proactive solutions, hybrid . Most defense techniques can be categorized as one of the following:

 
 
 
 • 
 
 Detection ( det. ): The goal is to identify whether the input to the RL agent (e.g., observations, environment dynamics, etc.) has been adversarially perturbed or is out-of-distribution. These test samples are then flagged for further human processing.

 

 • 
 
 Sanitization ( san. ): It is a proactive train time defense that preprocesses the input to remove adversarial perturbations before feeding them into the RL system.

 

 • 
 
 Adversarial training ( adv. tr. ): Iteratively re-train the system on the simulated attacks. These defenses are heuristic and do not have formal guarantees on convergence. It is a proactive defense mechanism to build inherent robustness in the model.

 

 • 
 
 Game Theory ( game th. ): Model the interaction between the agent and adversary as a game (Nash equilibrium  [ 94 ] or Stackelberg game  [ 95 ] ) and solve the robust policy, e.g. using minimax optimization. They are more principled than adversarial training; however, they are computationally costly.

 

 • 
 
 Regularization ( reg. ): Adds constraints or penalty terms during training to penalize agent’s sensitivity to input’s perturbations or choosing the best action considering a possible worst-case perturbation of the input.

 

 • 
 
 Distillation ( dist. ): This technique was originally proposed for model compression  [ 96 ] . Transfer knowledge from a robust, larger network, the teacher model, to a smaller or less complex, the student model, often smoothing decision boundaries.
This technique, if the two networks have the same architecture, increases regularization  [ 97 ] .

 

 • 
 
 Ensemble ( ens. ): Use multiple agents, i.e. policy or value functions, and aggregate their outputs to make decisions more robust against perturbations or adversarial attacks. For example, an ensemble of DQN agents, each trained on different random seeds and perturbation strategies. The final action can be based on consensus.

 

 
 
 

### VI-A Categorizing the Defenses against RL 

 
 In this paper, the goal of our taxonomy is to be helpful to both researchers and practitioners.
Thus, we consider not only the peculiarities of the defenses, but also the ones of the reinforcement learning systems on which they have shown to be effective.
Each proposed defenses can be categorized into the defense model and the considered threat model:

 
 
 Defense Model. Considered characteristics of the defense technique.

 
 • 
 
 Single vs. Multi Agent : Based on the number of agents considered in the environment, the proposed methods can be evaluated for either SARL or MARL environment.

 

 • 
 
 Policy Update. The frequency with which the policy is updated: on-policy, off-policy, off-line.

 

 • 
 
 Defense Type Based on the timeline where the defense is deployed, it can proactive, reactive or hybrid.

 

 • 
 
 Defense Technique. The specific technique used in the paper, i.e., adversarial training, detection, etc.

 

 
 
 
 Threat Model. Considered characteristics of the attack model that the defender used to defend their agent against. Same as in  subsection V-A .
In the following sections, we present a taxonomy of the defenses subdividing them based on when and how the defense is leveraged.

 
 
 TABLE V: Defenses for reinforcement learning systems. The presence of the ✓indicates that the corresponding attribute is true for the defense. For the attacker’s knowledge, we use ● to represent black-box, ◐ for gray-box, and ○ for white-box. All the defenses proposed so far have been tested on model-free RL algorithms, except  [ 98 ] . 
 
 
 
 | 
 DEFENSE MODEL | 
 THREAT MODEL | 
 | 

 
 References | 
 Single vs. | 
 Policy | 
 Defense | 
 Defense | 
 Attack’s | 
 Attacker’s | 
 Poisoning | 
 Attacker’s | 
 Attacker’s | 
 Sequential | 
 Attacker’s | 

 
 | 
 Multi-agent | 
 Update | 
 Type | 
 Technique | 
 Time | 
 Action | 
 | 
 Knowledge | 
 Goal | 
 Attack | 
 Agent | 

 
 Everitt et al. [ 99 ] | 
 SARL | 
 on-policy | 
 proactive | 
 reg. | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ◐ | 
 learn ​ ( π U ) \text{learn}({\pi}^{U}) | 
 | 
 ∅ \emptyset | 

 
 Everitt et al. [ 99 ] | 
 SARL | 
 on-policy | 
 reactive | 
 san. | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ◐ | 
 learn ​ ( π U ) \text{learn}({\pi}^{U}) | 
 | 
 ∅ \emptyset | 

 
 Wang et al. [ 60 ] | 
 SARL | 
 on-policy | 
 reactive | 
 san. | 
 t ​ r tr | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 ✓ | 
 ○ | 
 learn ​ ( π B ) \text{learn}({\pi}^{B}) | 
 ✓ | 
 ∈ ℰ \in\mathcal{E} | 

 
 Wang et al. [ 100 ] | 
 SARL | 
 off-policy | 
 reactive | 
 san. | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ● | 
 learn ​ ( π U ) \text{learn}({\pi}^{U}) | 
 | 
 ∅ \emptyset | 

 
 Lin et al. [ 78 ] | 
 SARL | 
 on/off-policy | 
 hybrid | 
 det. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Han et al. [ 101 ] | 
 SARL | 
 on/off-policy | 
 reactive | 
 det. + san. | 
 t ​ r tr | 
 c ⁡ ( 𝐬 , r ) c({\mathbf{s},r}) | 
 ✓ | 
 ● | 
 learn ​ ( π 𝐬 T ) \text{learn}({\pi}_{\mathbf{s}}^{T}) | 
 ✓ | 
 ≈ ℳ \approx\mathcal{M} | 

 
 Havens et al. [ 102 ] | 
 SARL | 
 on-policy | 
 reactive | 
 det. + san. | 
 t ​ r tr | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Garcıa et al. [ 103 ] | 
 SARL | 
 on-policy | 
 reactive | 
 det. + san. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Pinto et al. [ 104 ] | 
 SARL | 
 on-policy | 
 proactive | 
 game th. | 
 t ​ r tr | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ○ | 
 learn ​ ( π U ) \text{learn}({\pi}^{U}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Tessler et al. [ 105 ] | 
 SARL | 
 on/off-policy | 
 proactive | 
 game th. | 
 t ​ r tr | 
 c ⁡ ( a ) c({a}) | 
 | 
 ○ | 
 learn ​ ( π U ) \text{learn}({\pi}^{U}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Banihashem et al. [ 106 ] | 
 SARL | 
 on-policy | 
 proactive | 
 adv. tr. | 
 t ​ r tr | 
 c ⁡ ( r ) c({r}) | 
 ✓ | 
 ○ | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Tan et al. [ 107 ] | 
 SARL | 
 on-policy | 
 proactive | 
 adv. tr. | 
 t ​ r tr | 
 c ⁡ ( a ) c({a}) | 
 | 
 ○ | 
 learn ​ ( π T ) \text{learn}({\pi}^{T}) | 
 | 
 ∅ \emptyset | 

 
 Mandlekar et al. [ 43 ] | 
 SARL | 
 on-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( ℰ , 𝐨 ) c({\mathcal{E},\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Pattanaik et al. [ 108 ] | 
 SARL | 
 off-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 | 
 ∅ \emptyset | 

 
 Lee et al. [ 109 ] | 
 SARL | 
 on-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( a ) c({a}) | 
 | 
 ● | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Rajeshwaran et al. [ 110 ] | 
 SARL | 
 on-policy | 
 hybrid | 
 ens. + adv. tr. | 
 t ​ r tr | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 N.A | 
 N.A | 
 N.A | 
 N.A. | 

 
 Zhang et al. [ 55 ] | 
 SARL | 
 on-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Zhang et al. [ 111 ] | 
 SARL | 
 on/off-policy | 
 proactive | 
 reg. | 
 t ​ s ts | 
 c ⁡ ( ℰ ) c({\mathcal{E}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Wu et al. [ 112 ] | 
 SARL | 
 off-policy | 
 proactive | 
 reg. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Lutjens et al. [ 113 ] | 
 SARL | 
 off-policy | 
 proactive | 
 reg. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 reach ​ ( 𝐬 T ) \text{reach}({\mathbf{s}}^{T}) | 
 | 
 ∅ \emptyset | 

 
 Oikarinen et al. [ 114 ] | 
 SARL | 
 on/off-policy | 
 proactive | 
 reg. + adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Qu et al. [ 115 ] | 
 SARL | 
 off-policy | 
 proactive | 
 dist. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Fischer et al. [ 116 ] | 
 SARL | 
 off-policy | 
 proactive | 
 dist. + adv. tr. | 
 t ​ r tr , t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 learn ​ ( π U ) \text{learn}({\pi}^{U}) , min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Kos et al. [ 66 ] | 
 SARL | 
 on-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∅ \emptyset | 

 
 He et al. [ 117 ] | 
 SARL | 
 off-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Wang et al.  [ 118 ] | 
 SARL | 
 on-policy | 
 hybrid | 
 det. + adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∅ \emptyset | 

 
 Majadas et al.  [ 119 ] | 
 SARL | 
 off-policy | 
 reactive | 
 det. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Wang et al.  [ 120 ] | 
 SARL | 
 off-policy | 
 proactive | 
 reg. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○/● | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Liu et al.  [ 121 ] | 
 SARL | 
 off-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( a ) c({a}) | 
 | 
 ○ | 
 min ⁡ ( r ) \min({r}) | 
 | 
 ∉ ℰ \notin\mathcal{E} | 

 
 Bhardwaj et al.  [ 98 ] | 
 SARL | 
 offline | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( π ) c({\pi}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Korkmaz et al.  [ 122 ] | 
 SARL | 
 off-policy | 
 reactive | 
 det. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Meng et al.  [ 123 ] | 
 SARL | 
 on-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Guo et al.  [ 124 ] | 
 MARL | 
 off-policy | 
 proactive | 
 adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∈ ℰ \in\mathcal{E} | 

 
 Wang et al.  [ 125 ] | 
 MARL | 
 off-policy | 
 proactive | 
 reg. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 ✓ | 
 ∈ ℰ \in\mathcal{E} | 

 
 Bukharin et al.  [ 126 ] | 
 MARL | 
 on/off-policy | 
 proactive | 
 game th. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) / c ⁡ ( a ) c({a}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 Zhou et al.  [ 127 ] | 
 MARL | 
 on-policy | 
 proactive | 
 reg. + adv. tr. | 
 t ​ s ts | 
 c ⁡ ( 𝐨 ) c({\mathbf{o}}) | 
 | 
 ○ | 
 min ⁡ ( ℛ ) \min({\mathcal{R}}) | 
 | 
 ∅ \emptyset | 

 
 
 
 

### VI-B Defenses for RL Agents 

 
 This section provides an overview of the defenses against RL agents. Table V is a compact categorization of state-of-the-art defenses against single agents. Since all the defenses are model-free, we omit the model-based column.
In the following, we describe the defenses in more detail, subdividing them according to the type of defense: proactive, reactive, and hybrid.

 
 
 Proactive. Proactive defenses are applied during training to build inherent robustness into RL agents, preparing them to withstand adversarial conditions at test time. Techniques include adversarial training, game-theoretic methods, regularization, and distillation. Adversarial training is widely used  [ 106 , 107 , 43 , 108 , 55 , 66 , 117 , 124 , 121 , 98 , 123 ] . Even though the mentioned attacks use adversarial training as an underlying defense technique, they defend against different threat models (see Table  V ). For instance, Banihashem et al. [ 106 ] defend against reward poisoning attacks, while Tan et al. [ 107 ] focus on action-space and actuator perturbations. Zhang et al. [ 55 ] and Kos et al. [ 66 ] target test-time attacks, Bhardwaj et al. [ 98 ] train offline to improve policies relative to a reference, and Guo et al. [ 124 ] extend adversarial training to MARL with state-level attacks.
 Regularization-based defenses [ 99 , 111 , 112 , 113 , 120 ] add constraints during training to (a) prevent overfitting to training data or (b) reduce sensitivity to observation perturbations. These approaches are certified, providing guarantees on the maximum reward change under bounded input perturbations. For instance, Everitt et al. [ 99 ] proposed quantilisation, where the agent selects states from a top-reward quantile rather than always maximizing reward, introducing randomness and conservatism against adversarial manipulation. Zhang et al. [ 111 ] added a penalty term to the RL objective to encourage robustness to state perturbations. More recently, Wang et al. [ 125 ] combined risk estimation with constraint optimization to make MARL policies resilient to worst-case observation perturbations.
 Distillation-based defenses are relatively unexplored. Qu et al. [ 115 ] proposed Adversary Agnostic Policy Distillation (A2PD), where a robust teacher guides a student policy via a novel loss, improving adversarial robustness without exposure to adversarial examples. In contrast, Fischer et al. [ 116 ] combined policy distillation with adversarial training: a Q-network learns conventionally while a student network mimics it and is adversarially trained, enhancing robustness while preserving performance.
 Game-theoretic defenses [ 104 , 105 , 126 ] frame training as a minimax game against adversaries, making them a form of adversarial training. Pinto et al. [ 104 ] modeled the setup as a zero-sum game where a protagonist learns the task while an adversary applies disturbance forces, yielding a protagonist policy robust to worst-case disturbances. Unlike most works, Bukharin et al.  [ 126 ] addressed MARL, showing the importance of Lipschitz continuity for robustness and proposing a Stackelberg game approach, where the policy (leader) anticipates attacker actions (follower) under a smoothed optimization, enabling robust decision-making.
In  [ 109 ] , authors proposed a proactive defense approach based on transfer-learning in coordination with adversarial training. In this paper, the authors describe a variant of adversarial training where the weights of a pre-trained nominal policy are used as a starting point for further fine-tuning through adversarial training. By leveraging the prior weights, they aim to induce robust behaviors into the nominal policy.
Lately, researchers adopted combination of different proactive approaches  [ 114 , 127 , 116 ] to devise a strong defense systems. For example in  [ 127 ] , while the method generates adversarial perturbations at training time, the extra penalty term for action loss ensures that trained actors perform well on both clean and adversarial states. However, in  [ 116 ] the architecture is divided in two networks: policy network (student) and Q-network, where Q network learns the optimal Q-values and guides the overall learning process and remains largely unchanged, whereas the student network is the primary target for adversarial training and learns to mimic the policy of the Q-network through a process called policy distillation. In this work, both train-time and test-time are considered for evaluation of the proposed defense.

 
 
 Reactive. Reactive defenses operate during testing or deployment, aiming to detect, reject, or correct adversarial inputs after an attack occurs. Techniques include detection, sanitization, and reward monitoring. Detection-based defenses  [ 78 , 119 , 122 ] focus on identifying adversarial samples. Lin et al. [ 78 ] used an action-conditioned visual foresight module to predict future frames, detecting attacks by comparing its action distribution with the agent’s. Korkmaz et al. [ 122 ] distinguished adversarial samples via local curvature analysis of the policy’s cost function, noting benign inputs exhibit larger negative curvature. Majadas et al.  [ 119 ] applied clustering of benign transitions, flagging deviations in real time as adversarial.
Whereas the defenses proposed in  [ 60 , 100 , 101 , 99 ] fall in  sanitization category in which the model filters or reconstruct noisy/adversarial inputs before passing them on to the agent. For example, Wang et al. [ 100 ] denoise inputs by estimating a reward confusion matrix and recovering unbiased surrogate rewards. Wang et al. [ 60 ] sanitize trained models to remove backdoor functionality, though only in single-agent settings despite targeting multi-agent attacks. Han et al. [ 101 ] propose an inversion defense that cancels perturbations by finding a corrective signal δ ′ \delta^{\prime} such that, when added to the already perturbed state s ′ + δ s^{\prime}+\delta effectively cancels out the original perturbation δ \delta . Everitt et al. [ 99 ] introduce “decoupled RL,” leveraging external, trustworthy information to rectify the reward interpretation.
Other reactive approaches can be a combination of detection sanitization , where instead of sanitizing all input samples, these approaches first identify the adversarial/out-of-distribution sample and then re-actively sanitize it for the model  [ 102 , 103 ] . Havens et al.  [ 102 ] proposed an online and model-agnostic defense approach called Meta-Learned Advantage Hierarchy (MLAH) which leverages an advantage function to detect if the adversarial attack is present or not and then decides on which policy to choose. In  [ 103 ] , however, the model uses a memory of past, safe states to detect and correct adversarial inputs during test time.

 
 
 Hybrid These defenses combine proactive training-time mechanisms with reactive test-time components to ensure robustness and adaptability during deployment. Lin et al. [ 78 ] detect adversarial examples via action distribution comparisons and proactively suggest corrective actions to maintain performance. Rajeswaran et al. [ 110 ] use ensembles with adversarial training to learn policies robust to model errors, coupled with a model adaptation loop for reactive adjustments, though tested only against random perturbations. Wang et al.  [ 118 ] integrate robust adversarial training against worst-case attacks with saliency-based detection, providing insights into the DRL agent’s decision-making based on sensor inputs.
It is worth noting here that most defense techniques are designed in a single-agent setting. Although the defenses can be scaled to a multi-agent setting where each agent has its own individual defense method deployed, the defenses may not necessarily consider the other RL into account.

 
 
 

### VI-C Defenses for Autonomous Driving 

 
 To the best of our knowledge, very few defenses have been evaluated in the context of autonomous driving: one on a simplified autonomous driving scenario: a straight road [ 112 ] , another on intersection passing for multi-agent scenario  [ 125 ] . However, no mitigation has been tested on a real vehicle.
One possible reason could be the cost of an actual vehicle to evaluate such defenses, which easily amounts to more than 20,000$. 3 3 
 3 
 
 
 
 https://qz.com/924212/what-it-really-costs-to-turn-a-car-into-a-self-driving-vehicle/ .
This cost is not affordable for most research labs.
Albeit using a simulator can, to some degree, solve this problem. There are other challenges that defenses should face to be applicable in AD, such as:

 
 • 
 
 Requirement of only hardware that is usually already on board  [ 10 ] . For example, it is unlikely that an autonomous vehicle that is based solely on cameras will be equipped with a LiDar sensor only for defense purposes.

 

 • 
 
 Evaluation considering all components of the autonomous driving system  [ 10 ] . There is a possibility that the defense interacts with other parts of the car, which has to be taken into account during development.

 

 
 In contrast, some requirements can also be approximated in a simulator without involving an actual car. These include the introduced overhead, which cannot be too large as otherwise the car cannot react properly anymore. Some of the defenses discussed in this section (for example, when based on regularization  [ 78 , 100 , 99 ] or distillation  [ 115 , 116 ] ) do not introduce overhead neither at test nor at training time, whereas adversarial training  [ 104 , 128 , 107 ] introduces an overhead only at training time. They are thus more practical in this sense because the training can be done with a simulator. In addition, modifying RL systems that use previously proposed techniques  [ 129 ] to predict the influence of other agents in the environment could increase the security of autonomous driving.

 
 
 

### VI-D Future Research Directions 

 
 The development of defenses for autonomous driving (AD) is still in its early stages, leaving many open research directions. Privacy remains a key challenge, as recent work shows that deep reinforcement learning (DRL) models can be stolen by observing states and actions, yet no defense currently exists. Similarly, integrity violations lack effective solutions, since fine-tuning on clean data has proven insufficient, though defenses from supervised learning could be adapted. Availability violations, such as adversarial policy attacks, also remain undefended. Finally, future AD defenses should specify hardware requirements, quantify computational overhead, support system-level evaluation, and comply with industrial safety standards.

 
 
 
 

## VII Conclusion 

 
 In this survey, we categorize state-of-the-art attacks and defenses on RL. Unlike previous works, we provide a framework that helps system designers identify suitable defenses for specific threats. Since autonomous driving is a critical application, we examined it in greater depth, revealing that research on securing RL-based AD systems is still in its infancy and requires further study. We also highlight which ML attacks have yet to be adapted to RL and which lack defenses, offering both practitioners and researchers insights into potential threats and promising future directions.

 
 
 

## References

 
 
 [1] 
 
J. Clear, Atomic Habits . Random House, Oct. 2018.

 

 
 [2] 
 
T. T. Nguyen and V. J. Reddi, “Deep Reinforcement Learning for Cyber Security,” IEEE Transactions on Neural Networks and Learning Systems , pp. 1–17, 2021, conference Name: IEEE Transactions on Neural Networks and Learning Systems.

 

 
 [3] 
 
B. R. Kiran, I. Sobh, V. Talpaert, P. Mannion, A. A. A. Sallab, S. Yogamani, and P. Pérez, “Deep Reinforcement Learning for Autonomous Driving: A Survey,” IEEE Transactions on Intelligent Transportation Systems , pp. 1–18, 2021, conference Name: IEEE Transactions on Intelligent Transportation Systems.

 

 
 [4] 
 
Y. Yamagata, S. Liu, T. Akazaki, Y. Duan, and J. Hao, “Falsification of Cyber-Physical Systems Using Deep Reinforcement Learning,” IEEE Transactions on Software Engineering , vol. 47, no. 12, pp. 2823–2840, Dec. 2021, conference Name: IEEE Transactions on Software Engineering.

 

 
 [5] 
 
M. Lopez-Martin, B. Carro, and A. Sanchez-Esguevillas, “Application of deep reinforcement learning to intrusion detection for supervised problems,” Expert Systems with Applications , vol. 141, p. 112963, 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0957417419306815 

 

 
 [6] 
 
X. Wan, G. Sheng, Y. Li, L. Xiao, and X. Du, “Reinforcement Learning Based Mobile Offloading for Cloud-Based Malware Detection,” in GLOBECOM 2017 - 2017 IEEE Global Communications Conference , Dec. 2017, pp. 1–6.

 

 
 [7] 
 
I. Ilahi, M. Usama, J. Qadir, M. U. Janjua, A. Al-Fuqaha, D. T. Huang, and D. Niyato, “Challenges and Countermeasures for Adversarial Attacks on Deep Reinforcement Learning,” IEEE Transactions on Artificial Intelligence , pp. 1–1, 2021, conference Name: IEEE Transactions on Artificial Intelligence.

 

 
 [8] 
 
V. Behzadan and A. Munir, “The Faults in Our Pi Stars: Security Issues and Open Challenges in Deep Reinforcement Learning,” arXiv:1810.10369 [cs, stat] , Oct. 2018, arXiv: 1810.10369. [Online]. Available: http://arxiv.org/abs/1810.10369 

 

 
 [9] 
 
T. Chen, J. Liu, Y. Xiang, W. Niu, E. Tong, and Z. Han, “Adversarial attack and defense in reinforcement learning-from ai security view,” Cybersecurity , vol. 2, no. 1, pp. 1–22, 2019.

 

 
 [10] 
 
J. Shen, N. Wang, Z. Wan, Y. Luo, T. Sato, Z. Hu, X. Zhang, S. Guo, Z. Zhong, K. Li et al. , “Sok: On the semantic ai security in autonomous driving,” arXiv preprint arXiv:2203.05314 , 2022.

 

 
 [11] 
 
K. Grosse and A. Alahi, “A qualitative ai security risk assessment of autonomous vehicles,” Transportation Research Part C: Emerging Technologies , vol. 169, p. 104797, 2024.

 

 
 [12] 
 
A. D. M. Ibrahum, M. Hussain, and J.-E. Hong, “Deep learning adversarial attacks and defenses in autonomous vehicles: a systematic literature review from a safety perspective,” Artificial Intelligence Review , vol. 58, no. 1, p. 28, Nov. 2024. [Online]. Available: https://doi.org/10.1007/s10462-024-11014-8 

 

 
 [13] 
 
B. Badjie, J. Cecílio, and A. Casimiro, “Adversarial attacks and countermeasures on image classification-based deep learning models in autonomous driving systems: A systematic review,” ACM Comput. Surv. , vol. 57, no. 1, Oct. 2024. [Online]. Available: https://doi.org/10.1145/3691625 

 

 
 [14] 
 
M. Standen, J. Kim, and C. Szabo, “Adversarial machine learning attacks and defences in multi-agent reinforcement learning,” ACM Comput. Surv. , vol. 57, no. 5, Jan. 2025. [Online]. Available: https://doi.org/10.1145/3708320 

 

 
 [15] 
 
Y. Lei, D. Ye, S. Shen, Y. Sui, T. Zhu, and W. Zhou, “New challenges in reinforcement learning: a survey of security and privacy,” Artificial Intelligence Review , vol. 56, no. 7, pp. 7195–7236, Jul. 2023. [Online]. Available: https://doi.org/10.1007/s10462-022-10348-5 

 

 
 [16] 
 
K. Mo, P. Ye, X. Ren, S. Wang, W. Li, and J. Li, “Security and privacy issues in deep reinforcement learning: Threats and countermeasures,” ACM Comput. Surv. , vol. 56, no. 6, Feb. 2024. [Online]. Available: https://doi.org/10.1145/3640312 

 

 
 [17] 
 
B. Biggio and F. Roli, “Wild patterns: Ten years after the rise of adversarial machine learning,” Pattern Recognition , vol. 84, pp. 317–331, 2018.

 

 
 [18] 
 
R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, by Sutton, R.S. and Barto, A.G.  The MIT Press, 2018, vol. 3.

 

 
 [19] 
 
Y. Wang, E. Sarkar, W. Li, M. Maniatakos, and S. E. Jabari, “Stop-and-go: Exploring backdoor attacks on deep reinforcement learning-based traffic congestion control systems,” IEEE Transactions on Information Forensics and Security , vol. 16, pp. 4772–4787, 2021.

 

 
 [20] 
 
L. P. Kaelbling, M. L. Littman, and A. W. Moore, “Reinforcement learning: A survey,” Journal of artificial intelligence research , vol. 4, pp. 237–285, 1996.

 

 
 [21] 
 
Y. Ma, X. Zhang, W. Sun, and X. Zhu, “Policy poisoning in batch reinforcement learning and control,” Advances in Neural Information Processing Systems , 2019.

 

 
 [22] 
 
V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing atari with deep reinforcement learning,” arXiv preprint arXiv:1312.5602 , 2013.

 

 
 [23] 
 
Y. Zhao, I. Shumailov, H. Cui, X. Gao, R. Mullins, and R. Anderson, “Blackbox attacks on reinforcement learning agents using approximated temporal information,” in 2020 50th Annual IEEE/IFIP International Conference on Dependable Systems and Networks Workshops (DSN-W) . IEEE, 2020, pp. 16–24.

 

 
 [24] 
 
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347 , 2017.

 

 
 [25] 
 
A. Gleave, M. Dennis, C. Wild, N. Kant, S. Levine, and S. Russell, “Adversarial policies: Attacking deep reinforcement learning,” in International Conference on Learning Representations , 2019.

 

 
 [26] 
 
D. L. Poole and A. K. Mackworth, Artificial Intelligence: foundations of computational agents . Cambridge University Press, 2010.

 

 
 [27] 
 
A. Rakhsha, G. Radanovic, R. Devidze, X. Zhu, and A. Singla, “Policy teaching via environment poisoning: Training-time adversarial attacks against reinforcement learning,” in International Conference on Machine Learning . PMLR, 2020, pp. 7974–7984.

 

 
 [28] 
 
J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine, “D4RL: Datasets for Deep Data-Driven Reinforcement Learning,” Feb. 2021, number: arXiv:2004.07219 arXiv:2004.07219 [cs, stat]. [Online]. Available: http://arxiv.org/abs/2004.07219 

 

 
 [29] 
 
J. SAE, “3016 (2014). taxonomy and definitions for terms related to on-road motor vehicle automated driving systems,” Society of Automotive Engineers , 2014.

 

 
 [30] 
 
J. Garcia, Y. Feng, J. Shen, S. Almanee, Y. Xia, and Q. A. Chen, “A comprehensive study of autonomous vehicle bugs,” in Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering , 2020, pp. 385–396.

 

 
 [31] 
 
A. Qayyum, M. Usama, J. Qadir, and A. Al-Fuqaha, “Securing connected autonomous vehicles: Challenges posed by adversarial machine learning and the way forward,” IEEE Communications Surveys Tutorials , vol. 22, no. 2, pp. 998–1026, 2020.

 

 
 [32] 
 
Y. Deng, T. Zhang, G. Lou, X. Zheng, J. Jin, and Q.-L. Han, “Deep learning-based autonomous driving systems: a survey of attacks and defenses,” IEEE Transactions on Industrial Informatics , vol. 17, no. 12, pp. 7897–7912, 2021.

 

 
 [33] 
 
S. D. Pendleton, H. Andersen, X. Du, X. Shen, M. Meghjani, Y. H. Eng, D. Rus, and M. H. Ang, “Perception, planning, control, and coordination for autonomous vehicles,” Machines , vol. 5, no. 1, p. 6, 2017.

 

 
 [34] 
 
D. Georgia, H. Ronan, J. Henrik, N. Rossen, M. Apostolos, and S. M. J. Ignacio, “Cybersecurity challenges in the uptake of artificial intelligence in autonomous driving,” 2021.

 

 
 [35] 
 
C.-H. H. Yang, J. Qi, P.-Y. Chen, Y. Ouyang, I.-T. D. Hung, C.-H. Lee, and X. Ma, “Enhanced adversarial strategically-timed attacks against deep reinforcement learning,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 2020, pp. 3407–3411.

 

 
 [36] 
 
A. Boloor, X. He, C. Gill, Y. Vorobeychik, and X. Zhang, “Simple physical adversarial examples against end-to-end autonomous driving models,” in 2019 IEEE International Conference on Embedded Software and Systems (ICESS) . IEEE, 2019, pp. 1–7.

 

 
 [37] 
 
J. Sun, T. Zhang, X. Xie, L. Ma, Y. Zheng, K. Chen, and Y. Liu, “Stealthy and efficient adversarial attacks against deep reinforcement learning,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 34, no. 04, 2020, pp. 5883–5891.

 

 
 [38] 
 
C. Xiao, X. Pan, W. He, J. Peng, M. Sun, J. Yi, M. Liu, B. Li, and D. Song, “Characterizing attacks on deep reinforcement learning,” arXiv preprint arXiv:1907.09470 , 2019.

 

 
 [39] 
 
E. Tretschk, S. J. Oh, and M. Fritz, “Sequential attacks on agents for long-term adversarial goals,” in 2. ACM Computer Science in Cars Symposium , 2018.

 

 
 [40] 
 
C. Xie, J. Wang, Z. Zhang, Y. Zhou, L. Xie, and A. Yuille, “Adversarial examples for semantic segmentation and object detection,” in Proceedings of the IEEE international conference on computer vision , 2017, pp. 1369–1378.

 

 
 [41] 
 
S.-T. Chen, C. Cornelius, J. Martin, and D. H. P. Chau, “Shapeshifter: Robust physical adversarial attack on faster r-cnn object detector,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases . Springer, 2018, pp. 52–68.

 

 
 [42] 
 
D. Bhamare, M. Zolanvari, A. Erbad, R. Jain, K. Khan, and N. Meskin, “Cybersecurity for industrial control systems: A survey,” computers security , vol. 89, p. 101677, 2020.

 

 
 [43] 
 
A. Mandlekar, Y. Zhu, A. Garg, L. Fei-Fei, and S. Savarese, “Adversarially robust policy learning: Active construction of physically-plausible perturbations,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2017, pp. 3932–3939.

 

 
 [44] 
 
A. Demontis, M. Melis, M. Pintor, M. Jagielski, B. Biggio, A. Oprea, C. Nita-Rotaru, and F. Roli, “Why do adversarial attacks transfer? Explaining transferability of evasion and poisoning attacks,” in 28th USENIX Security Symposium (USENIX Security 19) . USENIX Association, 2019.

 

 
 [45] 
 
X. Zhang, Y. Ma, A. Singla, and X. Zhu, “Adaptive reward-poisoning attacks against reinforcement learning,” in International Conference on Machine Learning . PMLR, 2020, pp. 11 225–11 234.

 

 
 [46] 
 
Y. Huang and Q. Zhu, “Deceptive reinforcement learning under adversarial manipulations on cost signals,” in International Conference on Decision and Game Theory for Security . Springer, 2019, pp. 217–237.

 

 
 [47] 
 
G. Liu and L. Lai, “Provably efficient black-box action poisoning attacks against reinforcement learning,” in NeurIPS , 2021.

 

 
 [48] 
 
A. Rakhsha, X. Zhang, X. Zhu, and A. Singla, “Reward poisoning in reinforcement learning: Attacks against unknown learners in unknown environments,” arXiv preprint arXiv:2102.08492 , 2021.

 

 
 [49] 
 
R. Majadas, J. García, and F. Fernández, “Disturbing reinforcement learning agents with corrupted rewards,” arXiv preprint arXiv:2102.06587 , 2021.

 

 
 [50] 
 
P. Kiourti, K. Wardega, S. Jha, and W. Li, “Trojdrl: evaluation of backdoor attacks on deep reinforcement learning,” in 2020 57th ACM/IEEE Design Automation Conference (DAC) . IEEE, 2020, pp. 1–6.

 

 
 [51] 
 
C. Ashcraft and K. Karra, “Poisoning deep reinforcement learning agents with in-distribution triggers,” arXiv preprint arXiv:2106.07798 , 2021.

 

 
 [52] 
 
H. Foley, L. Fowl, T. Goldstein, and G. Taylor, “Execute order 66: Targeted data poisoning for reinforcement learning,” arXiv preprint arXiv:2201.00762 , 2022.

 

 
 [53] 
 
Y. Ma, K.-S. Jun, L. Li, and X. Zhu, “Data poisoning attacks in contextual bandits,” in International Conference on Decision and Game Theory for Security . Springer, 2018, pp. 186–204.

 

 
 [54] 
 
H. Xu, R. Wang, L. Raizman, and Z. Rabinovich, “Transferable environment poisoning: Training-time attack on reinforcement learning,” in Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems , 2021, pp. 1398–1406.

 

 
 [55] 
 
H. Zhang, H. Chen, D. Boning, and C.-J. Hsieh, “Robust reinforcement learning on state observations with learned optimal adversary,” in ICLR , 2021.

 

 
 [56] 
 
A. Tanev, S. Pavlitskaya, J. Sigloch, A. Roennau, R. Dillmann, and J. M. Zollner, “Adversarial black-box attacks on vision-based deep reinforcement learning agents,” in 2021 IEEE International Conference on Intelligence and Safety for Robotics (ISR) . IEEE, 2021, pp. 177–181.

 

 
 [57] 
 
X. Y. Lee, S. Ghadai, K. L. Tan, C. Hegde, and S. Sarkar, “Spatiotemporally constrained action space attacks on deep reinforcement learning agents,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 34, no. 04, 2020, pp. 4577–4584.

 

 
 [58] 
 
S. Huang, N. Papernot, I. Goodfellow, Y. Duan, and P. Abbeel, “Adversarial attacks on neural network policies,” arXiv preprint arXiv:1702.02284 , 2017.

 

 
 [59] 
 
P. Buddareddygari, T. Zhang, Y. Yang, and Y. Ren, “Targeted Attack on Deep RL-based Autonomous Driving with Learned Visual Patterns,” in 2022 International Conference on Robotics and Automation (ICRA) , 2022, pp. 10 571–10 577.

 

 
 [60] 
 
L. Wang, Z. Javed, X. Wu, W. Guo, X. Xing, and D. Song, “Backdoorl: Backdoor attack against competitive reinforcement learning,” arXiv preprint arXiv:2105.00579 , 2021.

 

 
 [61] 
 
Y. Yu and J. Liu, “Don’t Watch Me: A Spatio-Temporal Trojan Attack on Deep-Reinforcement-Learning-Augment Autonomous Driving,” Nov. 2022, arXiv:2211.14440 [cs]. [Online]. Available: http://arxiv.org/abs/2211.14440 

 

 
 [62] 
 
V. Behzadan and A. Munir, “Vulnerability of deep reinforcement learning to policy induction attacks,” in International Conference on Machine Learning and Data Mining in Pattern Recognition . Springer, 2017, pp. 262–275.

 

 
 [63] 
 
Y.-C. Lin, Z.-W. Hong, Y.-H. Liao, M.-L. Shih, M.-Y. Liu, and M. Sun, “Tactics of adversarial attack on deep reinforcement learning agents,” in Proceedings of the 26th International Joint Conference on Artificial Intelligence , 2017, pp. 3756–3762.

 

 
 [64] 
 
B. G. A. Tekgul, S. Wang, S. Marchal, and N. Asokan, “Real-Time Adversarial Perturbations Against Deep Reinforcement Learning Policies: Attacks and Defenses,” in Computer Security – ESORICS 2022 , ser. Lecture Notes in Computer Science, V. Atluri, R. Di Pietro, C. D. Jensen, and W. Meng, Eds. Cham: Springer Nature Switzerland, 2022, pp. 384–404.

 

 
 [65] 
 
A. Russo and A. Proutiere, “Optimal attacks on reinforcement learning policies,” arXiv preprint arXiv:1907.13548 , 2019.

 

 
 [66] 
 
J. Kos and D. Song, “Delving into adversarial attacks on deep policies,” arXiv preprint arXiv:1705.06452 , 2017.

 

 
 [67] 
 
Y. Sun, R. Zheng, Y. Liang, and F. Huang, “Who is the strongest enemy? towards optimal and efficient evasion attacks in deep rl,” arXiv preprint arXiv:2106.05087 , 2021.

 

 
 [68] 
 
E. Korkmaz, “Nesterov Momentum Adversarial Perturbations in the Deep Reinforcement Learning Domain,” 2020.

 

 
 [69] 
 
M. Inkawhich, Y. Chen, and H. Li, “Snooping attacks on deep reinforcement learning,” in Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems , 2020, pp. 557–565.

 

 
 [70] 
 
K. Chen, S. Guo, T. Zhang, X. Xie, and Y. Liu, “Stealing deep reinforcement learning models for fun and profit,” in Proceedings of the 2021 ACM Asia Conference on Computer and Communications Security , 2021, pp. 307–319.

 

 
 [71] 
 
H.-J. Yoon, R. Holmes, H. Jafarnejadsani, and P. Voulgaris, “Real-time adversarial image perturbations for autonomous vehicles using reinforcement learning,” ACM Transactions on Cyber-Physical Systems , vol. 9, no. 2, pp. 1–24, 2025.

 

 
 [72] 
 
J. Fan, X. Lei, X. Chang, J. Miši, V. B. Miši, and Y. Yao, “Less is more: A stealthy and efficient adversarial attack method for drl-based autonomous driving policies,” IEEE Internet of Things Journal , 2025.

 

 
 [73] 
 
S. Li, J. Guo, J. Xiu, Y. Zheng, P. Feng, X. Yu, J. Wang, A. Liu, Y. Yang, B. An, W. Wu, and X. Liu, “Attacking cooperative multi-agent reinforcement learning by adversarial minority influence,” Neural Networks , vol. 191, p. 107747, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0893608025006276 

 

 
 [74] 
 
F. Bai, R. Liu, Y. Du, Y. Wen, and Y. Yang, “Rat: Adversarial attacks on deep reinforcement agents for targeted behaviors,” Proceedings of the AAAI Conference on Artificial Intelligence , vol. 39, no. 15, pp. 15 453–15 461, Apr. 2025. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/33696 

 

 
 [75] 
 
X. Zheng, X. Ma, S. Wang, X. Wang, C. Shen, and C. Wang, “Toward evaluating robustness of reinforcement learning with adversarial policy,” in 2024 54th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN) , 2024, pp. 288–301.

 

 
 [76] 
 
Z. Zhou, G. Liu, W. Guo, and M. Zhou, “Adversarial attacks on multiagent deep reinforcement learning models in continuous action space,” IEEE Transactions on Systems, Man, and Cybernetics: Systems , vol. 54, no. 12, pp. 7633–7646, 2024.

 

 
 [77] 
 
Z. Xiang, D. J. Miller, S. Chen, X. Li, and G. Kesidis, “A backdoor attack against 3d point cloud classifiers,” arXiv preprint arXiv:2104.05808 , 2021.

 

 
 [78] 
 
Y.-C. Lin, M.-Y. Liu, M. Sun, and J.-B. Huang, “Detecting adversarial attacks on neural network policies with visual foresight,” arXiv preprint arXiv:1710.00814 , 2017.

 

 
 [79] 
 
T. Brown, D. Mané, A. Roy, M. Abadi, and J. Gilmer, “Adversarial Patch,” ArXiv , vol. abs/1712.09665, 2017.

 

 
 [80] 
 
S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, “Universal adversarial perturbations,” in CVPR , 2017.

 

 
 [81] 
 
T. Orekondy, B. Schiele, and M. Fritz, “Knockoff Nets: Stealing Functionality of Black-Box Models,” 2019, pp. 4954–4963. [Online]. Available: https://openaccess.thecvf.com/content_CVPR_2019/html/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.html 

 

 
 [82] 
 
N. Papernot, P. McDaniel, and I. Goodfellow, “Transferability in machine learning: from phenomena to black-box attacks using adversarial samples,” arXiv preprint arXiv:1605.07277 , 2016.

 

 
 [83] 
 
N. Dalvi, P. Domingos, Mausam, S. Sanghai, and D. Verma, “Adversarial classification,” in Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD) , Seattle, 2004, pp. 99–108.

 

 
 [84] 
 
D. Lowd and C. Meek, “Adversarial learning,” in Proc. 11th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD) . Chicago, IL, USA: ACM Press, 2005, pp. 641–647.

 

 
 [85] 
 
B. Biggio, B. Nelson, and P. Laskov, “Poisoning attacks against support vector machines,” in 29th Int’l Conf. on Machine Learning , J. Langford and J. Pineau, Eds. Omnipress, 2012, pp. 1807–1814.

 

 
 [86] 
 
P. W. Koh and P. Liang, “Understanding black-box predictions via influence functions,” in International Conference on Machine Learning (ICML) , 2017.

 

 
 [87] 
 
T. Gu, B. Dolan-Gavitt, and S. Garg, “Badnets: Identifying vulnerabilities in the machine learning model supply chain,” in NIPS Workshop on Machine Learning and Computer Security , vol. abs/1708.06733, 2017.

 

 
 [88] 
 
M. Fredrikson, S. Jha, and T. Ristenpart, “Model inversion attacks that exploit confidence information and basic countermeasures,” in Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security , ser. CCS ’15. New York, NY, USA: ACM, 2015, pp. 1322–1333.

 

 
 [89] 
 
F. Tramèr, F. Zhang, A. Juels, M. K. Reiter, and T. Ristenpart, “Stealing machine learning models via prediction apis,” in 25th USENIX Security Symposium (USENIX Security 16) . Austin, TX: USENIX Association, 2016, pp. 601–618.

 

 
 [90] 
 
I. Shumailov, Y. Zhao, D. Bates, N. Papernot, R. Mullins, and R. Anderson, “Sponge Examples: Energy-Latency Attacks on Neural Networks,” in 2021 IEEE European Symposium on Security and Privacy (EuroS P) , Sep. 2021, pp. 212–231.

 

 
 [91] 
 
A. E. Cinà, A. Demontis, B. Biggio, F. Roli, and M. Pelillo, “Energy-Latency Attacks via Sponge Poisoning,” arXiv:2203.08147 [cs] , Mar. 2022, arXiv: 2203.08147. [Online]. Available: http://arxiv.org/abs/2203.08147 

 

 
 [92] 
 
R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership inference attacks against machine learning models,” in 2017 IEEE Symposium on Security and Privacy (SP) , May 2017, pp. 3–18.

 

 
 [93] 
 
Y. J. Jia, Y. Lu, J. Shen, Q. A. Chen, H. Chen, Z. Zhong, and T. W. Wei, “Fooling detection alone is not enough: Adversarial attack against multiple object tracking,” in International Conference on Learning Representations (ICLR’20) , 2020.

 

 
 [94] 
 
M. Brückner, C. Kanzow, and T. Scheffer, “Static prediction games for adversarial learning problems,” J. Mach. Learn. Res. , vol. 13, pp. 2617–2654, September 2012.

 

 
 [95] 
 
W. Liu and S. Chawla, “Mining adversarial patterns via regularized loss minimization.” Machine Learning , vol. 81, no. 1, pp. 69–83, 2010.

 

 
 [96] 
 
G. Hinton, O. Vinyals, and J. Dean, “Distilling the Knowledge in a Neural Network,” arXiv, Tech. Rep. arXiv:1503.02531, Mar. 2015, arXiv:1503.02531 [cs, stat] type: article. [Online]. Available: http://arxiv.org/abs/1503.02531 

 

 
 [97] 
 
H. Mobahi, M. Farajtabar, and P. Bartlett, “Self-Distillation Amplifies Regularization in Hilbert Space,” in Advances in Neural Information Processing Systems , vol. 33. Curran Associates, Inc., 2020, pp. 3351–3361. [Online]. Available: https://proceedings.neurips.cc/paper/2020/hash/2288f691b58edecadcc9a8691762b4fd-Abstract.html 

 

 
 [98] 
 
M. Bhardwaj, T. Xie, B. Boots, N. Jiang, and C.-A. Cheng, “Adversarial model for offline reinforcement learning,” in Advances in Neural Information Processing Systems , A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 1245–1269. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2023/file/0429ececfb199efc93182990169e73bb-Paper-Conference.pdf 

 

 
 [99] 
 
T. Everitt, V. Krakovna, L. Orseau, and S. Legg, “Reinforcement learning with a corrupted reward channel,” in Proceedings of the 26th International Joint Conference on Artificial Intelligence , 2017, pp. 4705–4713.

 

 
 [100] 
 
J. Wang, Y. Liu, and B. Li, “Reinforcement learning with perturbed rewards,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 34, no. 04, 2020, pp. 6202–6209.

 

 
 [101] 
 
Y. Han, D. Hubczenko, P. Montague, O. De Vel, T. Abraham, B. I. Rubinstein, C. Leckie, T. Alpcan, and S. Erfani, “Adversarial reinforcement learning under partial observability in autonomous computer network defence,” in 2020 International Joint Conference on Neural Networks (IJCNN) . IEEE, 2020, pp. 1–8.

 

 
 [102] 
 
A. J. Havens, Z. Jiang, and S. Sarkar, “Online robust policy learning in the presence of unknown adversaries,” in NeurIPS , 2018.

 

 
 [103] 
 
J. García and I. Sagredo, “Instance-based defense against adversarial attacks in deep reinforcement learning,” Engineering Applications of Artificial Intelligence , vol. 107, p. 104514, 2022.

 

 
 [104] 
 
L. Pinto, J. Davidson, R. Sukthankar, and A. Gupta, “Robust adversarial reinforcement learning,” in International Conference on Machine Learning . PMLR, 2017, pp. 2817–2826.

 

 
 [105] 
 
C. Tessler, Y. Efroni, and S. Mannor, “Action robust reinforcement learning and applications in continuous control,” in International Conference on Machine Learning . PMLR, 2019, pp. 6215–6224.

 

 
 [106] 
 
K. Banihashem, A. Singla, and G. Radanovic, “Defense against reward poisoning attacks in reinforcement learning,” arXiv preprint arXiv:2102.05776 , 2021.

 

 
 [107] 
 
K. L. Tan, Y. Esfandiari, X. Y. Lee, S. Sarkar et al. , “Robustifying reinforcement learning agents via action space adversarial training,” in 2020 American control conference (ACC) . IEEE, 2020, pp. 3959–3964.

 

 
 [108] 
 
A. Pattanaik, Z. Tang, S. Liu, G. Bommannan, and G. Chowdhary, “Robust deep reinforcement learning with adversarial attacks,” arXiv preprint arXiv:1712.03632 , 2017.

 

 
 [109] 
 
X. Y. Lee, Y. Esfandiari, K. L. Tan, and S. Sarkar, “Query-based targeted action-space adversarial policies on deep reinforcement learning agents,” in Proceedings of the ACM/IEEE 12th International Conference on Cyber-Physical Systems , 2021, pp. 87–97.

 

 
 [110] 
 
A. Rajeswaran, S. Ghotra, B. Ravindran, and S. Levine, “Epopt: Learning robust neural network policies using model ensembles,” in 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings , 2017.

 

 
 [111] 
 
H. Zhang, H. Chen, C. Xiao, B. Li, M. Liu, D. S. Boning, and C.-J. Hsieh, “Robust deep reinforcement learning against adversarial perturbations on state observations,” in NeurIPS , 2020.

 

 
 [112] 
 
F. Wu, L. Li, Z. Huang, Y. Vorobeychik, D. Zhao, and B. Li, “Crop: Certifying robust policies for reinforcement learning through functional smoothing,” in International Conference on Learning Representations , 2022.

 

 
 [113] 
 
B. Lütjens, M. Everett, and J. P. How, “Certified adversarial robustness for deep reinforcement learning,” in Conference on Robot Learning . PMLR, 2020, pp. 1328–1337.

 

 
 [114] 
 
T. Oikarinen, T.-W. Weng, and L. Daniel, “Robust deep reinforcement learning through adversarial loss,” arXiv preprint arXiv:2008.01976 , 2020.

 

 
 [115] 
 
X. Qu, Y.-S. Ong, A. Gupta, and Z. Sun, “Adversary agnostic robust deep reinforcement learning,” arXiv preprint arXiv:2008.06199 , 2020.

 

 
 [116] 
 
M. Fischer, M. Mirman, S. Stalder, and M. Vechev, “Online robustness training for deep reinforcement learning,” arXiv preprint arXiv:1911.00887 , 2019.

 

 
 [117] 
 
X. He, W. Huang, and C. Lv, “Trustworthy autonomous driving via defense-aware robust reinforcement learning against worst-case observational perturbations,” Transportation Research Part C: Emerging Technologies , vol. 163, p. 104632, 2024.

 

 
 [118] 
 
C. Wang and N. Aouf, “Explainable deep adversarial reinforcement learning approach for robust autonomous driving,” IEEE Transactions on Intelligent Vehicles , 2024.

 

 
 [119] 
 
R. Majadas, J. García, and F. Fernández, “Clustering-based attack detection for adversarial reinforcement learning,” Applied Intelligence , vol. 54, no. 3, pp. 2631–2647, 2024.

 

 
 [120] 
 
D. Wang, K. Moore, D. Goel, M. Kim, G. Li, Y. Li, R. Doss, M. Xue, B. Li, S. Camtepe, and L. Zhu, “CAMP in the Odyssey: Provably Robust Reinforcement Learning with Certified Radius Maximization,” 2025, arXiv:2501.17667 [cs]. [Online]. Available: http://arxiv.org/abs/2501.17667 

 

 
 [121] 
 
Q. Liu, Y. Kuang, and J. Wang, “Robust deep reinforcement learning with adaptive adversarial perturbations in action space,” in 2024 International Joint Conference on Neural Networks (IJCNN) , 2024, pp. 1–8.

 

 
 [122] 
 
E. Korkmaz and J. Brown-Cohen, “Detecting adversarial directions in deep reinforcement learning to make robust decisions,” in Proceedings of the 40th International Conference on Machine Learning , ser. ICML’23. JMLR.org, 2023.

 

 
 [123] 
 
J. Meng, F. Zhu, Y. Ge, and P. Zhao, “Integrating safety constraints into adversarial training for robust deep reinforcement learning,” Information Sciences , vol. 619, pp. 310–323, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0020025522013408 

 

 
 [124] 
 
W. Guo, G. Liu, Z. Zhou, J. Wang, Y. Tang, and M. Wang, “Robust training in multiagent deep reinforcement learning against optimal adversary,” IEEE Transactions on Systems, Man, and Cybernetics: Systems , vol. 55, no. 7, pp. 4957–4968, 2025.

 

 
 [125] 
 
C. Wang, Z. Wang, and N. Aouf, “Robust multi-agent reinforcement learning against adversarial attacks for cooperative self-driving vehicles,” IET Radar, Sonar Navigation , vol. 19, no. 1, p. e70033, 2025.

 

 
 [126] 
 
A. Bukharin, Y. Li, Y. Yu, Q. Zhang, Z. Chen, S. Zuo, C. Zhang, S. Zhang, and T. Zhao, “Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms,” Nov. 2023. [Online]. Available: https://openreview.net/forum?id=FmZVRe0gn8 

 

 
 [127] 
 
Z. Zhou, G. Liu, and M. Zhou, “A robust mean-field actor-critic reinforcement learning against adversarial perturbations on agent states,” IEEE Transactions on Neural Networks and Learning Systems , vol. 35, no. 10, p. 14370–14381, Oct. 2024. [Online]. Available: http://dx.doi.org/10.1109/TNNLS.2023.3278715 

 

 
 [128] 
 
C. Tessler, Y. Efroni, and S. Mannor, “Action Robust Reinforcement Learning and Applications in Continuous Control,” ser. Proceedings of Machine Learning Research, K. Chaudhuri and R. Salakhutdinov, Eds., vol. 97. Long Beach, California, USA: PMLR, Jun. 2019, pp. 6215–6224. [Online]. Available: http://proceedings.mlr.press/v97/tessler19a.html 

 

 
 [129] 
 
X. Huang, S. Hong, A. Hofmann, and B. C. Williams, “Online risk-bounded motion planning for autonomous vehicles in dynamic environments,” in Proceedings of the International Conference on Automated Planning and Scheduling , vol. 29, 2019, pp. 214–222.

 

 
 [130] 
 
A. E. Cinà, K. Grosse, S. Vascon, A. Demontis, B. Biggio, F. Roli, and M. Pelillo, “Backdoor Learning Curves: Explaining Backdoor Poisoning Beyond Influence Functions,” arXiv:2106.07214 [cs] , Mar. 2022, arXiv: 2106.07214. [Online]. Available: http://arxiv.org/abs/2106.07214 

 

 
 [131] 
 
R. Salay, R. Queiroz, and K. Czarnecki, “An analysis of iso 26262: Machine learning and safety in automotive software,” SAE Technical Paper, Tech. Rep., 2018.

 

 
 
 
 
 
 
 | 
 
 
 Ambra Demontis is an Assistant Professor at the University of Cagliari, Italy. She received her M.Sc. degree (Hons.) in Computer Science and her Ph.D. degree in Electronic Engineering and Computer Science, respectively, in 2014 and 2018. Her research interests include secure machine learning, kernel methods, and computer security. She co-organized the AISec workshop, serves on the program committee of conferences, such as Usenix, and is an Associate Editor for Pattern Recognition and the International Journal of Machine Learning and Cybernetics. She is a Member of the IEEE and the IAPR. 
 | 

 
 
 
 
 | 
 
 
 Srishti Gupta is currently a PhD in student in Italian National PhD program in AI at Sapienza University, Rome co-hosted by the University of Cagliari. She received her MS from University of Arizona, US in 2021 and B.Tech from Bharati Vidyapeeth College in Delhi, India in 2017. Her research interests include Continual Learning, Out-of-Distribution Detection and security of LLM models. She serves as a reviewer to several journals and conferences. 
 | 

 
 
 
 
 | 
 
 
 Maura Pintor is an Assistant Professor at the PRA Lab, in the Department of Electrical and Electronic Engineering of the University of Cagliari, Italy. She received her PhD in Electronic and Computer Engineering from the University of Cagliari in 2022. Her research focuses on machine learning security. 
 | 

 
 
 
 
 | 
 
 
 Luca Demetrio is an Assistant Professor at the University of Genova (Italy), where he also received his Ph.D. in 2021. He is currently studying the security of Windows malware detectors implemented with Machine Learning techniques. He is also involved in the development of techniques that can improve the quality of the evaluation of machine learning models, by providing debugging tools that can spot the failures at attack time. 
 | 

 
 
 
 
 | 
 
 
 Kathrin Grosse is a Postdoctoral Researcher at the VITA Lab at EPFL, Switzerland. She received her Ph.D. in 2021 from Saarland University under the supervision of Michael Backes at CISPA Helmholtz Center. Her research interests are at the intersection of ML and security, recently focusing on ML security in practice. During her Ph.D., she interned at Disney Research Zurich and IBM Yorktown, where her work resulted in a US Patent. She was nominated as an AI newcomer within the German Federal Ministry of Education and Research’s Science Year 2019. She serves as a reviewer for many international journals and conferences. 
 | 

 
 
 
 
 | 
 
 
 Hsiao-Ying Lin is a principal researcher in Shield Labs at Huawei Technologies France. Her research interests include adversarial machine learning, applied cryptography and security issues in automotive areas. She received the MS and PhD degrees in computer science from National Chiao Tung University, Taiwan, in 2005 and 2010, respectively. 
 | 

 
 
 
 
 | 
 
 
 Chengfang Fang obtained both his bachelor and Ph.D. from National University of Singapore with Tata Consultancy Award and Research Achievement Award. He joined Huawei and continued to work on security and privacy protection across domains, such as machine learning, internet of things, mobile device and biometrics. He has been working on these areas for over a decade and has published over 30 research papers and obtained over 20 patents in the domain. He is currently a principal researcher in Huawei Singapore Research Center. 
 | 

 
 
 
 
 | 
 
 
 Battista Biggio (MSc 2006, PhD 2010) is a Full Professor of Computer Engineering at the University of Cagliari, Italy. He has provided pioneering contributions to machine learning security. His paper “Poisoning Attacks against Support Vector Machines” won the prestigious 2022 ICML Test of Time Award. He chaired IAPR TC1 (2016-2020) and served as Associate Editor for IEEE TNNLS and IEEE CIM. He is now Associate Editor-in-Chief for Pattern Recognition and serves as Area Chair for NeurIPS and IEEE Symp. SP. He is a Fellow of IEEE and AAIA, ACM Senior Member, and Member of IAPR, AAAI, and ELLIS. 
 | 

 
 
 
 
 | 
 
 
 Fabio Roli is a Full Professor of Computer Engineering at the University of Genova, Italy. He has been appointed Fellow of the IEEE and Fellow of the International Association for Pattern Recognition. He is a recipient of the Pierre Devijver Award for his contributions to statistical pattern recognition. 
 |