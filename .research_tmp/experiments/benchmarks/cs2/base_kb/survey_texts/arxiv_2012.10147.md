Hierarchical principles of embodied reinforcement learning: A review 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2012.10147v2 [cs.AI] 18 Aug 2022 
 
 

# Hierarchical principles of embodied reinforcement learning: A review

 
 
 Manfred Eppe
 
 Affiliation:  Universität Hamburg, Germany
 
 Affiliation:  eppe@informatik.uni-hamburg.de
 
    
 Christian Gumbsch
 
 Affiliation:  Universität Tübingen, Germany
 
 Affiliation:  Max Planck Institute for Intelligent Systems, Tübingen, Germany
 
    
 Matthias Kerzel
 
 Affiliation:  Universität Hamburg, Germany
 
    
 Phuong D.H. Nguyen
 
 Affiliation:  Universität Hamburg, Germany
 
    
 Martin V. Butz
 
 Affiliation:  Universität Tübingen, Germany
 
    
 Stefan Wermter
 
 Affiliation:  Universität Hamburg, Germany
 

 Abstract 
 
 Cognitive Psychology and related disciplines have identified several critical mechanisms that enable intelligent biological agents to learn to solve complex problems.
There exists pressing evidence that the cognitive mechanisms that enable problem-solving skills in these species build on hierarchical mental representations.
Among the most promising computational approaches to provide comparable learning-based problem-solving abilities for artificial agents and robots is hierarchical reinforcement learning.
However, so far the existing computational approaches have not been able to equip artificial agents with problem-solving abilities that are comparable to intelligent animals, including human and non-human primates, crows, or octopuses.
Here, we first survey the literature in Cognitive Psychology, and related disciplines, and find that many important mental mechanisms involve compositional abstraction, curiosity, and forward models.
We then relate these insights with contemporary hierarchical reinforcement learning methods, and identify the key machine intelligence approaches that realise these mechanisms.
As our main result, we show that all important cognitive mechanisms have been implemented independently in isolated computational architectures, and there is simply a lack of approaches that integrate them appropriately.
We expect our results to guide the development of more sophisticated cognitively inspired hierarchical methods, so that future artificial agents achieve a problem-solving performance on the level of intelligent animals.

 
 
 
 
 
 a 
 
 b 
 
 c 
 
 d 
 
 e 
 
 Fig. 1: A New Caledonian crow solves a food-access problem: 47 a, First, the crow picks a stick. b-c, Then it uses the stick to pull a stone out of a tube. d-e, Finally, it uses the stone to activate a mechanism that releases food. The crow has never solved this problem setup before. Nevertheless, after a brief inspection phase, it is able to solve the problem, suggesting that the crow develops memorized abstract mental representations to plan ahead. 
 
 .
 
 
 
 Humans and several other higher level intelligent animal species have the ability to break down complex unknown problems into hierarchies of simpler previously learned sub-problems. This hierarchical approach allows them to solve previously unseen problems in a zero-shot manner, i.e., without any trial and error.
For example, Fig. 1 depicts how a New Caledonian crow solves a non-trivial food-access puzzle that consists of three causal steps: It first picks a stick, then uses the stick to access a stone, and then uses the stone to activate a mechanism that releases food 47 .
There exist numerous analogous experiments that attest similar capabilities to primates, octopuses, and, of course, humans 13 , 93 .
This raises the question of how we can equip intelligent artificial agents and robots with similar problem-solving abilities.
To answer this question, the involved computational mechanisms and algorithmic implementation options need to be identified.

 
 
 A very general computational framework for learning-based problem-solving is reinforcement learning (RL) 3 , 78 , 117 .
Several studies suggest that RL is in many aspects biologically and cognitively plausible 87 , 117 .
Existing RL-based methods are to some extent able to perform zero-shot problem-solving and transfer learning.
However, this is currently only possible for minor variations of the same or a similar task 29 or in simple synthetic domains, such as a 2D gridworld 88 , 115 .
A continuous-space problem-solving behaviour that is comparable with the crow’s behaviour of Fig. 1 has not yet been realised with any artificial system, based on RL or other approaches.

 
 
 Research in human and animal cognition strongly suggests that problem-solving and learning is hierarchical 11 , 13 , 125 .
We hypothesise that one reason why current machine learning systems fail is that the existing approaches underestimate the power of learning hierarchical abstractions.
At the time of writing this article, we performed a comprehensive meta-search over RL review articles from 2015 to 2020 using the Microsoft Academic search engine. Our meta-survey has yielded the following results:
The most cited review article on RL since 2015 3 dedicates 1/6th of a page out of 16 pages to hierarchical approaches.
In the second most cited article 42 , hierarchical RL is not considered at all, i.e, the word stem ‘hierarchic’ does not appear anywhere in the text. The third most cited review article 78 dedicates 2/3rd of a page out of 85 total pages to hierarchical RL.
From 37 RL reviews and surveys published since 2015, only two contain the word stem “hierarchic” in their abstract.
The second edition of the popular RL book by Sutton and Barto Super only mentions hierarchies in the twenty-year-old options framework 119 on 2 pages in the final book chapter and briefly discusses automatic abstraction a few lines later in that chapter.
It appears that researchers are struggling with automatically learning RL-suitable hierarchical structures.

 
 
 In this article, we address this gap by illuminating potential reasons for this struggle and by providing pointers to solutions.
We show that most computational hierarchical RL approaches are model-free, whereas the results from our survey of biological cognitive models suggest that suitable predictive forward models are needed.
Furthermore, we exhibit that existing hierarchical RL methods hardly address state abstraction and compositional representations.
However, we also show that there exist exceptions where isolated relevant cognitive mechanisms including forward models and compositional abstraction have already been implemented, but not in combination with other important mechanisms.
As a result, we conclude that the AI community already has most of the tools for building more intelligent agents at hand;
but it currently lacks holistic efforts to integrate them appropriately.

 
 
 Fig. 2: Prerequisites, mechanisms, and features of biological problem-solving agents.
 Forward- and inverse models are prerequisites for higher-order mechanisms and abilities.
 compositional abstractions of forward models allow the agent to decompose a problem into reusable substructures, e.g., by learning which part of an object is easily graspable, such as the stem of a glass.
 Intrinsic motivation applied with the help of forward models can drive an agent to interact with its environment in an information seeking, epistemic way, e.g., by eliciting interactions with the glass to learn more about its properties.
 Mental simulations enable the exploration of hypothetical state-action sequences, e.g., by imagining how the hand position will evolve while reaching for the glass.
Given these mechanisms, an agent can flexibly achieve desired goal states via hierarchical, compositional goal-directed planning , e.g., planning to drink from the glass by first suitably grasping and then appropriately transporting it.
Compositional abstractions enable the agent to perform transfer learning , e.g., by executing suitable grasps onto other objects with stem-like parts, such as the shown teapot. Taken together, these abilities enable few-shot problem-solving . 
 
 

## Neurocognitive foundations

 
 The problem-solving abilities of a biological agent depend on cognitive key mechanisms that include abstraction, intrinsic motivation, and mental simulation.
 Fig. 2 shows how these mechanisms depend on forward and inverse models as neuro-functional prerequisites.
Together, cognitive abilities that we deem crucial for higher level intelligence become enabled, including few-shot problem solving and transfer learning.
In the following, we start with a characterization of these crucial cognitive abilities and then exhibit some of the key mechanisms and prerequisites needed to enable them.

 
 

### Crucial cognitive abilities

 
 By cognitive abilities for problem-solving we refer to phenomenologically observable traits and properties of biological agents.
As the most remarkable ability, we consider few-shot problem-solving.

 
 
 Few-shot problem-solving is the ability to solve unknown problems with few ( ⪅ 10 \lessapprox 10 ) trials. 1 1 
 1 
 
 
 
 Zero-shot problem-solving is a special case of few-shot problem-solving where no additional training at all is required to solve a new problem. 
For simple problems, few-shot problem-solving is trivial. For example, catching a ball is usually a purely reactive behaviour that builds on a direct neural mechanism to map the observation of the flying ball to appropriate motor commands. Such problems are solvable with current computational RL methods 78 .
However, there are also more difficult problems that require, e.g., using tools in new contexts.
For example, consider the aforementioned problem-solving example of the crow 47 (see Fig. 1 ). The crow knows how to use a stick as a tool without any further training, because it has previously used a stick for related problems.
In our conceptual framework (see Fig. 2 ) we identify two abilities that are most central to perform such problem-solving, namely transfer learning and planning.

 
 
 Transfer learning 
allows biological agents to perform few-shot problem-solving by transferring the solution to a previously solved task to novel, previously unseen tasks 93 . This significantly reduces and sometimes eliminates the number of trials to solve a problem. Perkins and Salomon Super propose a distinction between near and far transfer. Near transfer refers to the transfer of skills between domains or situations that are rather similar, e.g. learning to catch a ball and then catching other objects. In contrast, far transfer requires the transfer of more abstract solutions between different situations typically via abstract analogies.
Such analogies are determined by mappings between problems from different domains.
For example, an object transport problem with a robotic hand that involves [ grasp(object) , moveArm(goal) , release(object) ] is analogous to a file movement problem on a computer that involves [cutInFolder(file) , navigate(targetFolder) , pasteInFolder(file) ], with the obvious mappings between the action types and arguments.

 
 
 Cognitive theories consider analogical reasoning as a critical cornerstone of human cognition. For example, conceptual metaphor theory 31 understands metaphors as mappings between analogous concepts in different domains.
In addition to evidence from linguistics, there is also significant evidence from education theory and research on mechanical reasoning that analogical reasoning improves problem-solving: Education theory suggests that human transfer learning improves when explicitly trained to identify analogies between different problems 67 .
The Psychology of mechanical inference suggests that analogical knowledge about the dynamic properties of mechanical objects is often transferred between domains 56 .
For example, knowledge about what happens if somebody jumps into a pool can be transferred to other mechanical problems that involve over-flooding water containers 56 .

 
 
 Goal-directed planning. 
The behaviour of biological agents is traditionally divided into two categories: Stimulus-driven, habituated behaviour and goal-directed, planned behaviour 27 , 41 .
Initially, stimulus-response theories dominated the field, suggesting that most behaviour was learned by forming associations that were previously reinforced 123 .
Edward Tolman was one of the main critiques of stimulus-response theories 27 . He showed that rats can find rewards in a maze faster when they have visited the maze before, even if their previous visit had not been rewarded 124 .
The results suggest that the rats form a representation, or cognitive map, of the maze, which enables them to plan or mentally simulate their behaviour once a reward was detected.
Habituation of behaviour, which is comparable to model-free policy optimization in reinforcement learning 10 , 21 , enables an agent to learn one particular skill in a controlled environment.
However, complex problem-solving in a new or changing environment requires some form of prospective planning 13 , 27 , 58 .
Over the last decades, more research has focused on understanding the mechanisms and representations involved in goal-directed planning.
It is now generally agreed upon that, during planning, humans predict the effect of their actions, compare them to the desired effects, and, if required, refine their course of actions 58 , 70 .
This is deemed to be a hierarchical process where the effects of actions on different levels of abstraction are considered 10 , 125 .
Thereby, humans tend to first plan high-level actions before considering actions at a finer granularity 133 .
Hierarchical planning can dramatically decrease the computational cost involved in planning 10 , 27 .
Additionally, hierarchical abstractions enable automatizing originally planned behaviour, thus further alleviating computational cost 21 .

 
 
 

### Cognitive mechanisms for transfer learning and planning

 
 Transfer learning and planning rely on shared and partially overlapping mechanisms and principles. Here, we focus on three mechanisms and principles that we identify as most critical, namely sensorimotor abstraction, intrinsic motivation, and mental simulation.

 
 
 

### Sensorimotor abstraction.

 
 According to the widely accepted embodiment theory, abstract mental concepts are inferred from our bodily-grounded sensorimotor environmental interaction experiences 6 , 13 , 12 , 95 .
Cognitive theories often distinguish between action abstraction and state abstraction .
Action abstractions refer to a temporally extended sequence of primitive actions, that are often referred to as options 119 or movement primitives 33 , 105 .
For example, transporting an object can be seen as an action abstraction since it is composed of a sequence of more primitive actions, such as reaching , lifting , and transporting (see Fig. 3 ).
The elementary actions that compose an action abstraction are typically rather loosely defined through their effects.
For example, we can tell that a robot arm we have never seen before is grasping an object, even though the anatomy, the unfolding control algorithms, and the involved motor commands may largely vary from our human ones.
State abstractions refer to encoding certain parts of the environment while abstracting away irrelevant details.
A simple example of a state abstraction is a code that specifies whether a certain type of object is present in a scene.
State abstractions can be hierarchically structured:
A single object can be subdivided into its parts to form a partonomy, i.e., a hierarchical organization of its parts and their spatial relationship (see Fig. 3 ) 84 , 141 .
Additionally, objects can be organized taxonomically, reflecting a hierarchical relationship that focuses on particular conceptual aspects 84 , 141 .
For example, an affordance-oriented 46 taxonomy of beverage containers could classify wine glasses and beer mugs both as drinking vessels , while a teapot would be classified as a pot instead.
Meanwhile, beer mugs , wine glasses , and teapots are all graspable containers .

 
 
 Conceptual abstractions and cross-references to other conceptual abstractions within imply a representational key property: compositionality .
Formal compositionality principles state that an expression or representation is compositional if it consists of subexpressions, and if there exist rules that determine how to combine the subexpressions 120 .
The Language of Thought theory 34 transfers the compositionality principle from language to abstract mental representations, claiming that thought must also be compositional.
For example, humans are easily able to imagine previously unseen situations and objects by composing their known constituents, such as the golden pavements and ruby walls imagined by Scottish philosopher David Humes in A Treatise of Human Nature 35 .

 
 
 On top of that, embodied world knowledge further constrains the formal semantics-based options to combined sub-expressions.
In the remainder, we will refer to this type of compositionality by common sense compositionality , that is, compositional rules that are grounded in and flexibly inferred by our intuitive knowledge about the world, including other agents (humans, animals, robots, machines, etc.).
This common sense compositionality essentially enforces that our minds attempt to form plausible environmental scenarios and events that unfold within given conceptual information about a particular situation 12 , 43 , 73 . Consider the processes that unfold when understanding a sentence like “He sneezed the napkin off the table”.
Most people have probably never heard or used this sentence before, but everyone with sufficient world and English grammar knowledge can correctly interpret the sentence, imagining a scene wherein the described event unfolds 12 .
Common sense compositionality makes abstractions applicable in meaningful manners, by constraining the filling in of an abstract component (e.g., the target of a grasp) towards applicable objects (e.g., a graspable object such as a teapot), as depicted in Fig. 3 .
Note the awkwardness – and indeed the attempt of the brain to resolve it – when disobeying common sense compositionality, such as when stating that ‘The dax fits into the air’ or ‘grasping a cloud’ 12 .

 
 
 Common sense compositionality indeed may be considered a hallmark and cornerstone of human creativity, as has been (indirectly) suggested by numerous influential cognitivists 6 , 32 , 72 , 73 , 116 , 127 , 132 .
In the neurosciences, recent neuroimaging techniques indeed revealed that the compositionality of mental representations can actually be observed in neural codes 35 .
For example, Haynes et al. Super and Reverberi et al. Super show that the neural codes in the ventrolateral prefrontal cortex representing compound behavioural rules can be decomposed into neural codes that encode their constituent rules.
Similar results have been obtained for neural codes that represent food compositions 5 .

 
 
 
 
 
 
 
 
 
 
 Fig. 3: Compositional action and state abstractions. Action abstractions describe a sequence of more primitive actions (a.). State abstractions encode certain properties of the state space (b). Their compositionality enables their general application, by instantiating abstract definitions (the graspable target of reaching), with a specific object (teapot) (c.). 
 
 
 From an algorithmic perspective, common sense compositionality yields benefits for analogical reasoning and planning.
It simplifies the identification of analogies because it enables compositional morphisms between representations.
This advantage is well-known in cognitive theories of creativity, e.g., concept blending 28 , 127 , where the search for analogies is a computational bottleneck.
Similarly, compositional state- and action representations for goal-directed planning lead to a lower computational complexity, because action and object types can be flexibly combined as long as critical properties match.
Along these lines, action grammars have been proposed, which systematically form an effect-oriented taxonomy of tool-object interactions 135 .
As a result, common sense compositionality enables the utilisation of objects as tools in innovative manners—when, for example, utilising a stone as a hammer 13 , 73 .
Moreover, it enables drawing analogies across domains—when , for example, talking about “holding onto a thought” for later utilisation.

 
 
 Common sense compositionality thus seems to be key for truly intelligent artificial systems.
We propose that apart from the addition of suitable inductive learning biases, such as event-segmentation tendencies 12 , 13 , 14 , 114 , 142 ,
suitably targeted intrinsic motivation and the ability to play out mental simulations are of critical importance.

 
 
 Intrinsic motivation affects goal-directed reasoning and planning because it sets intrinsically motivated goals that an agent may aim to achieve.
The most basic behaviour of biological agents purely strives for satisfying homeostatic needs, such as hunger or thirst. However, advanced behaviour, such as exploration and play, seems to be partially decoupled from the primary biological needs of an animal.
From a cognitive development perspective, the term intrinsic motivation was coined to describe the “inherent tendency to seek out novelty, […] to explore and to learn” 22 . Here, intrinsic is used in opposition to extrinsic values that are directly associated with the satisfaction of needs or the avoidance of undesired outcomes 40 .
Intrinsic motivations induce general exploratory behaviour, curiosity, and playfulness.
Simple exploratory behaviour can even be found in worms, insects, and crustaceans 94 and may be elicited by rather simple tendencies to wander around while being satiated or an inborn tendency to evade overcrowded areas.
Curiosity refers to an epistemic drive that causes information gain-oriented exploratory behaviour 7 , 80 , 90 .
Combined with developing common sense compositionality, this drive can elicit hypothesis testing behaviour even by means of analogy making.
The closely related playfulness is only exhibited in certain intelligent, mostly mammalian or avian species, such as dogs and corvids 94 , where scenarios and events within are played out in a hypothetical manner and novel interactions are tried-out in a skill-improvement-oriented manner.

 
 
 Mental simulation , meanwhile,
enables biological agents to reflect on and to anticipate the dynamics of its environment on various representational levels in the past, present, and future, and even fully hypothetically.
Actual motor-imagery has been shown to enable athletes to improve the execution of challenging body movements (e.g. a backflip), significantly reducing the number of actually executed trials 63 .
On the reasoning side, consider human engineers who rely on mental simulation when developing mechanical systems, such as pulleys or gears.
The simulation involves visual imaginations, but also includes non-visual properties like friction and force 56 .
Mental simulation also takes place on higher conceptual, compositional, and causal reasoning levels. For example, Kahnemann and Tversky Super , Wells and Gavanski Super and later Taylor et al. Super report how mental simulation improves the planning of future behaviour on a causal conceptual level, such as when planning a trip.

 
 
 

### Forward-inverse models as functional prerequisites

 
 Sensorimotor abstraction, intrinsic motivation, and mental simulation are complex mechanisms that demand a suitable neuro-functional basis.
We propose that this essential basis is constituted by forward and inverse models 134 . Forward models predict how an action affects the world, while inverse models determine the actions that need to be executed to achieve a goal or to obtain a reward.
Note that inverse models may be implicitly inferred from the available forward models, but more compact inverse models, which are closely related to habits and motion primitives, certainly foster the proficiency of particular behavioural skills further 6 , 13 , 25 , 105 .

 
 
 Forward-inverse models for mental simulation. 
To perform mental simulation, an agent needs a forward model to simulate how the dynamics of the environment behave, possibly conditioned on the agent’s actions.
However, even when equipped with a well-predicting forward model, the consideration of all possible actions and action sequences quickly becomes computationally intractable.
Hence, action-selection requires more direct mappings from state and desired inner mental state (including an intention or a goal) to potential actions.
This is accomplished by inverse models.
In RL, inverse models are represented as behaviour policies.
Selecting actions can happen in a reflex-like manner on the motor level but it can also serve as a heuristic to generate candidate actions for accomplishing a current goal.
For example, when designing a new mechanical system, engineers have an intuition about how to put certain mechanical parts together, depending on the goal function they want to achieve with the system. Based on their intuition, they mentally simulate and verify the functioning of the mechanical system 56 .

 
 
 Forward and inverse models for intrinsic motivation. 
Forward and inverse models are also needed to trigger behaviour that is driven by intrinsic motivation. For example, prediction error minimisation has been demonstrated to be a useful additional, intrinsic reward signal to model curiosity, inversely triggering behaviour and behavioural routines that are believed to cause particular environmental changes that are, in turn, believed to possibly result in knowledge gain 66 , 109 , 108 , 90 .

 
 
 Along similar lines, Friston et al.  propose that intrinsically motivated behaviour emerges when applying active inference 39 , 40 .
Active inference describes the process of inferring actions to minimize expected free energy, which includes approximations of anticipated surprise 38 , 41 .
Free energy is decomposed into various sources of uncertainty.
One part is uncertainty about future states or outcomes given a certain sequence of actions 40 . The agent strives to reduce this uncertainty, and with it overall expected free energy, by activating the actions that cause the uncertainty.
Hence, active inference can lead to epistemic exploration, where an agent acts out different behaviour to learn about its consequences 40 , 89 .
Here, the forward model is required to predict future states given an action, and also to estimate the uncertainty of the prediction.

 
 
 Forward and inverse models for abstraction and grounding. 
Over the last decade, various theories including predictive coding 59 , the Free Energy Principle 38 , and the Bayesian Brain hypothesis 68 , have viewed the brain as a generative machine, which constantly attempts to match incoming sensory signals with its probabilistic predictions 17 .
Within these frameworks, prediction takes place on multiple processing hierarchies that interact bidirectionally:
Top-down information per processing stage provides additional information to predict sensory signals, while bottom-up error information is used to correct the top-down predictions 18 .
Meanwhile, future states are predicted in a probabilistic manner 17 .
On lower processing levels, actual incoming sensorimotor signals are predicted, while on higher levels more abstract, conceptual and compositional predictive encodings emerge 12 .
In this way, rather complex state abstractions, such as the aforementioned container -concept, can form.
As a result, high-level predictions, e.g., of an object being within a container, enable predictions on lower levels.
For example, consider the prediction of sensory information about how an object’s position will change over time while it is occluded.

 
 
 Event Segmentation Theory (EST) 142 makes the role of forward predictions even more explicit:
EST is concerned with why humans seem to perceive the continuous sensory activity in terms of discrete, conceptual events.
The theory suggests that these segmentations mirror the internal representation of the unfolding experience. Internal representations of events, or event models , are sensorimotor abstractions that encode entities with certain properties and their functional relations in a spatiotemporal framework 98 .
According to EST, these event models guide the perception by providing additional information that can be used for forward predictions 142 .

 
 
 While observing an event, a specific subset of event models is active until a transient prediction error is registered, resulting in an exchange of the currently active event models to a new subset that may be better suited for predicting the currently ongoing dynamics 71 , 98 , 114 , 142 .
EST-inspired computational models demonstrate that such transient forward prediction errors can indeed be used to signal event transitions in video streams or self-explored sensorimotor data in an online fashion
 36 , 48 , 60 .

 
 
 
 

## Computational realizations

 
 The abilities, mechanisms and prerequisites of computational hierarchical reinforcement learning systems (cf.  Fig. 4 ) are less sophisticated and integrated than those of biological agents. However, there are promising novel developments to potentially overcome the existing limitations. To identify the potential of the existing computational approaches, we provide an overview of the current state of the art on hierarchical reinforcement learning in Tab. 1 .

 
 
 Fig. 4: A general hierarchical problem-solving architecture with a separate policy π i \pi^{i} for each layer of abstraction.
Most existing computational HRL approaches focus on two layers, where the high-level action a 1 a^{1} is either a behavioural primitive (an option or sub-policy), or a subgoal.
Only the low-level actions a 0 a^{0} are motor commands that affect the environment directly. 
 
 
 Table 1 : Abilities, mechanisms and prerequisites of recent hierarchical reinforcement learning approaches. 
 
 
 | 
 Abilities | 
 Mechanisms | 
 Prerequisites | 

 
 
 
 Approach 
 | 
 
 
 Few-shot problem-solving 
 | 
 
 
 Transfer learning ————— 
 
 N: near 
 F: far 
 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 | 
 
 
 Goal-directed planning —————— 
 
 H: high-level 
 L: low-level 
 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 | 
 

 
 Sensorimotor abstraction 
 ————————— 
 R: representational 
 C: compositional 
 ( actions )          ( states ) 
 | 
 
 
 Intrinsic motivation ——————- 
 
 C: curiosity 
 D: diversity 
 SG: subgoals 
 
 | 
 
 
 Mental simulation ——————- 
 
 H: high-level 
 L: low-level 
 
 | 
 | 
 | 
 | 
 | 
 | 

 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 
 
 Forward model ——————– 
 
 H: high-level 
 L: low-level 
 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 | 
 
 
 Inverse model / policy ——————- 
 
 H: high-level 
 L: low-level 
 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 
 
 Akrour et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R,C 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Arulkumaran et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Bacon et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Beyret et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 C 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Blaes et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 H 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 C, SG 
 | 
 
 
 H 
 | 
 
 
 H,L 
 | 
 
 
 L 
 | 

 
 
 
 Chuck et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 C 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Dietterich Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 R,C 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Eppe et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 H 
 | 
 
 
 R,C 
 | 
 
 
 R,C 
 | 
 
 
 SG 
 | 
 
 
 H 
 | 
 
 
 H 
 | 
 
 
 L 
 | 

 
 
 
 Eysenbach et al. Super 
 | 
 
 
 - 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 D 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Frans et al. Super 
 | 
 
 
 (X) 
 | 
 
 
 F 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Ghazanfari et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Ghosh et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 R 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Haarnoja et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Han et al. Super 
 | 
 
 
 X 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Heess et al. Super 
 | 
 
 
 X 
 | 
 
 
 F 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Hejna Iii et al. Super 
 | 
 
 
 X 
 | 
 
 
 F 
 | 
 
 
 - 
 | 
 
 
 C 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Jiang et al. Super 
 | 
 
 
 X 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R,C 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Kulkarni et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 R,C 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Levy et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 C 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Li et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 H 
 | 
 
 
 R,C 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 H,L 
 | 
 
 
 H,L 
 | 
 
 
 H 
 | 

 
 
 
 Li et al. Super 
 | 
 
 
 X 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Lyu et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 H 
 | 
 
 
 R 
 | 
 
 
 R,C 
 | 
 
 
 SG 
 | 
 
 
 H 
 | 
 
 
 H 
 | 
 
 
 L 
 | 

 
 
 
 Ma et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 H 
 | 
 
 
 R 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 H 
 | 
 
 
 H 
 | 
 
 
 L 
 | 

 
 
 
 Machado et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 R 
 | 
 
 
 D 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Nachum et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 C 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Oh et al. Super 
 | 
 
 
 X 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R,C 
 | 
 
 
 R 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Qiao et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Qureshi et al. Super 
 | 
 
 
 - 
 | 
 
 
 F 
 | 
 
 
 - 
 | 
 
 
 R,C 
 | 
 
 
 - 
 | 
 
 
 SG,D 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Rafati and Noelle Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 C 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Rasmussen et al. Super 
 | 
 
 
 - 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R,(C) 
 | 
 
 
 R 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Riedmiller et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Röder et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 C 
 | 
 
 
 - 
 | 
 
 
 C,SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Saxe et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R,C 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Schaul and
Ring Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 R,C 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Shankar et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 D 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Sharma et al. Super 
 | 
 
 
 X 
 | 
 
 
 N 
 | 
 
 
 H 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 D 
 | 
 
 
 H 
 | 
 
 
 H 
 | 
 
 
 L 
 | 

 
 
 
 Sohn et al. Super 
 | 
 
 
 X 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R,C 
 | 
 
 
 R 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Tessler et al. Super 
 | 
 
 
 (X) 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R,C 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Vezhnevets et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Vezhnevets et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R,C 
 | 
 
 
 R,C 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Wu et al. Super 
 | 
 
 
 - 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L 
 | 
 
 
 L 
 | 
 
 
 L,H 
 | 

 
 
 
 Wulfmeier et al. Super 
 | 
 
 
 - 
 | 
 
 
 N 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Yamamoto et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 H 
 | 
 
 
 R,C 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 H 
 | 
 
 
 H 
 | 
 
 
 L 
 | 

 
 
 
 Yang et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 H 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 H 
 | 
 
 
 H 
 | 
 
 
 L 
 | 

 
 
 
 Yang et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 R 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 
 
 Zhang et al. Super 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 C 
 | 
 
 
 - 
 | 
 
 
 SG 
 | 
 
 
 - 
 | 
 
 
 - 
 | 
 
 
 L,H 
 | 

 
 

### Transfer learning and planning for few-shot abilities

 
 Our survey of the neurocognitive foundations indicates that two foundational cognitive abilities for few-shot problem-solving are transfer learning and planning.

 
 
 Transfer learning denotes the re-use of previously learned skills in new contexts and domains, where near transfer learning denotes transfer between similar contexts and domains between source and target tasks, while far transfer considers stronger dissimilarities 93 .
A significant fraction of the existing near transfer approaches build on learning re-usable low-level skills, which are referred to as behavioural primitives, options, or sub-policies.
For example, Li et al. Super and Frans et al. Super present sub-policy-based hierarchical extensions to Proximal Policy Optimization (PPO) 111 .
Their approaches enable the transfer of sub-policies to novel tasks within the same domain.
 Heess et al. Super use Generalized Advantage Estimation (GAE) 110 to implement similar transferable sub-policies. However, the authors also consider transfer learning towards different types of tasks. For example, they transfer behaviour from a locomotion task to a more complex soccer task.
 Eysenbach et al. Super and Sharma et al. Super introduce diversity functions to learn diverse re-usable sub-policies that are independent of external rewards.

 
 
 Tessler et al. Super focus on the transfer of skills in lifelong learning, and Wu et al. Super propose a model-based approach, where only the forward models are transferred to new tasks, but not the policies.
 Vezhnevets et al. Super build on the automatic discovery of transferable macro-actions (plans) to solve problems in discrete 2D-environments, while Jiang et al. Super use natural language action representations to perform near transfer learning.
 Qureshi et al. Super build on re-usable low-level policies for goal-conditioned hierarchical reinforcement learning.

 
 
 Research has not only considered transfer learning between different tasks but also between different robot and agent morphologies 24 , 37 , 57 .
We classify these approaches as far transfer because the entire sensorimotor apparatus changes, which places the agent in a dissimilar context.
Furthermore, the methods that feature such cross-morphological transfer also perform far transfer learning across different application domains. For example, Frans et al. Super transfer navigation skills acquired in a discrete-space grid maze environment to a continuous-space ant maze environment.

 
 
 Planning is a highly overloaded term that has different meanings in different sub-disciplines of AI. In this paper, we refer to goal-directed planning in the sense of classical AI, as an abductive search for a sequence of actions that will lead to a specific goal by simulating the actions with an internal model of the domain dynamics.
Planning enables one-shot problem-solving because the searching does not involve the physical execution of actions. The agent only executes the actions if the mental simulation verifies that the actions are successful.
In this sense, planning differs from model-based reinforcement learning, which usually refers to training a policy by simulating actions using a predictive forward model 118 .

 
 
 Hierarchical planning is a well-known paradigm in classical AI 86 , but approaches that integrate planning with hierarchical reinforcement learning are rare.
Some approaches integrate action planning with reinforcement learning by using an action planner for high-level decision-making and a reinforcement learner for low-level motor control 29 , 81 , 82 , 113 , 138 , 139 .
These approaches underpin that planning is especially useful for high-level inference in discrete state-action spaces.

 
 
 

### Mechanisms behind transfer learning and planning

 
 Our summary of the cognitive principles behind transfer learning and planning reveals three important mechanisms that are critical for the learning and problem-solving capabilities of biological agents, namely compositional sensorimotor abstraction, intrinsic motivation, and mental simulation.

 
 
 Compositional sensorimotor abstraction and grounding. 
The temporal abstraction of actions is, by definition, an inherent property of hierarchical reinforcement learning as it allows to break down complex problems into a temporal hierarchical structure of simpler problems.
Another dimension of abstraction is representational abstraction.
Hierarchical reinforcement learning approaches also involve representational action abstraction.
One can distinguish the existing approaches into two different types of representational action abstraction.
The probably most influential method for action abstraction builds on behaviour primitives 4 , 26 , 37 , 69 , 77 , 79 , 82 , 113 , 83 , 96 , 112 , 122 , 130 , including options 119 , sub-policies, or atomic high-level skills.
Such behaviour primitives are represented in an abstract representational space, e.g. in a discrete finite space of action labels or indices, abstracting away from the low-level space of motor commands (see Fig. 5 , a.).
Another more recent type of producing high-level action representations are subgoals in the low-level state-space 29 , 44 , 76 , 85 , 99 ,   57 , 103 , 143 (see Fig. 5 , b.).
Subgoals are abstract actions defined in the low-level state space, and the agent achieves the final goal by following a sequence of subgoals.
There exist also methods that encode high-level actions as continuous vector representations. For example, Han et al. Super use a multi-timescale RNN, where the high-level actions are encoded by the connections between the RNN layers and others use latent vector representations to encode high-level behaviour 61 , 112 .

 
 
 
 
 (a.) behaviour primitives              (b.) Subgoals              
 
 Fig. 5: Action abstraction through behaviour primitives (a) vs subgoals (b). With behaviour primitives, the agent determines the path to the final goal by selecting a sequence of high-level actions, but without specifying explicitly to which intermediate state each action leads. With subgoals, the agent determines the intermediate states as subgoals that lead to the final goal, but without specifying the actions to achieve the subgoals. 
 
 
 Overall, in hierarchical reinforcement learning literature, there exists a strong implicit focus on action abstraction.
However significant cognitive evidence demonstrates that representational state abstraction is at least as important to realize efficient problem-solving 13 , 75 .
Yet, compared to action abstraction, there is considerably less research on state abstraction in hierarchical reinforcement learning.
Cognitive state abstractions range from the mere preprocessing of sensor data, e.g. in the primary visual cortex, to the problem-driven grounding of signals in abstract compositional concept representations. Counterparts for some of these methods can also be found in computational architectures.
For example, most current reinforcement learning approaches that process visual input use convolutional neural networks to preprocess the visual sensor data 64 , 74 , 88 , 115 , 129 , 137 , 140 .
A problem with simple preprocessing is that it does not appreciate that different levels of inference require different levels of abstraction: For example, to transport an object from one location to another with a gripper, only the low-level motor control layer of a hierarchical approach needs to know the precise shape and weight of the object. A high-level planning or inference layer only needs to know abstract Boolean facts, such as whether the object to transport is initially within the gripper’s reach or not, and whether the object is light enough to be carried.

 
 
 Therefore, we consider only those approaches as representational abstraction methods that involve layer-wise abstraction.
Layer-wise state abstraction has been tackled, but most existing hierarchical reinforcement learning approaches perform the abstraction using manually defined abstraction functions 29 , 69 , 82 , 126 .
There exist only a few exceptions where state abstractions are derived automatically in hierarchical reinforcement learning architectures 45 , 129 , 130 , e.g. through clustering 1 , 44 , with feature representations 106 or by factorisation 130 . Interestingly, non-hierarchical model-based reinforcement learning offers promising alternative prediction-based methods for state abstraction 52 , 92 , which show parallels to cognitive prediction-based abstraction theories.
However, these have not yet been applied in a hierarchical architecture.

 
 
 As implied from the cognitive science side, a key-property of representations of states and actions is compositionality .
For instance, a symbolic compositional action representation grasp(glass) allows for modulating the action grasp with the object glass .
Compositionality is not limited to symbolic expressions, but also applicable to distributed numerical expressions, such as vector representations. For example, a vector v 1 v_{1} is a compositional representation if it is composed of other vectors, e.g., v 2 v_{2} and v 3 v_{3} , and if there is a vector operation ∘ \circ that implements interpretation rules, e.g., v 1 = v 2 ∘ v 3 v_{1}=v_{2}\circ v_{3} .

 
 
 There is significant cognitive evidence that compositionality improves transfer learning 20 .
This evidence is computationally plausible when considering that transfer learning relies on exploiting analogies between problems. The analogies between two or more problems in goal-conditioned reinforcement learning are defined by a multidimensional mapping between the initial state space, the goal space, and the action space of these problems.
For example, given n a = 4 n_{a}=4 action types (e.g. “grasp”, “push”, “move”, “release” ) and n o = 4 n_{o}=4 object types (e.g. “glass”, “cup”, “tea pot”, “spoon” ), a non-compositional action-representations requires one distinct symbol for each action-object combination, resulting in n o ⋅ n a / 2 n_{o}\cdot n_{a}/2 possible action mappings. In contrast, an analogy mapping with compositional actions would require to search over possible mappings between action types and, separately, over mappings between objects. Hence, the size of the search space is n o / 2 + n a / 2 n_{o}/2+n_{a}/2 .
Evidence that a lower number of possible analogy mappings improves transferability is also provided by other cognitively inspired computational methods, such as concept blending 28 .

 
 
 The work by Jiang et al. Super provides further empirical evidence that compositional representations improve learning transfer.
The authors use natural language as an inherently compositional representation to describe high-level actions in hierarchical reinforcement learning.
In an ablation study, they find that the compositional natural language representation of actions improves transfer learning performance compared to non-compositional representations.

 
 
 Few other researchers use compositional representations in hierarchical reinforcement learning. Saxe et al. Super compose high-level actions from concurrent linearly solvable Markov decision processes (LMDPs) to guarantee optimal compositionality. Their method successfully executes compositional policies that it has never executed before, effectively performing zero-shot problem-solving.
Zero-shot problem-solving has also been demonstrated by other related research that features compositional action representations, but that does not draw an explicit link between compositionality and zero-shot problem-solving 61 , 88 , 115 .
Few symbolic compositional state representations exist, but these rely on manually defined abstraction functions 29 , 69 , 81 , 126 or they are very general and feature compositionality only as an optional property 100 .
There also exist hierarchical reinforcement learning approaches where sub-symbolic compositional representations are learned 24 , 97 , 106 , 130 .
The compositionality of these representations is rather implicit and, with one exception 97 , has not been investigated in the context of transfer learning. The exceptional approach by Qureshi et al. Super considers composable low-level policies and shows that compositionality significantly improves transfer between similar environments.

 
 
 Intrinsic motivation 
is a useful method to stabilise reinforcement learning by supplementing sparse external rewards.
It is also commonly used to incentivize exploration.
Reinforcement learning typically models intrinsic motivation through intrinsic rewards. The most common method of hierarchical reinforcement learning to generate intrinsic rewards is by providing intrinsic rewards when subgoals or subtasks are achieved 9 , 15 , 29 , 50 , 62 , 69 , 81 , 88 , 97 , 100 , 102 , 103 , 104 , 115 , 138 , 139 .
Other approaches provide intrinsic motivation to identify a collection of behavioural primitives with a high diversity 9 , 30 , 83 and predictability 113 .
This includes also the identification of primitives that are suited for re-composing high-level tasks 112 .

 
 
 Another prominent intrinsic reward model that is commonly used in non-hierarchical reinforcement learning is based on surprise and curiosity 90 , 91 , 107 , 109 .
In these approaches, surprise is usually modelled as a function of the prediction error of a forward model, and curiosity is realised by providing intrinsic rewards if the agent is surprised.
However, there is only little work on modelling surprise and curiosity in hierarchical reinforcement learning. Only Blaes et al. Super , Colas et al. Super , and Röder et al. Super use surprise in a hierarchical setting, showing that hierarchical curiosity leads to a significant improvement of the learning performance.
These approaches train a high-level layer to determine explorative subgoals in hierarchical reinforcement learning.
An alternative method to model curiosity is to perform hypothesis-testing for option discovery 16 .

 
 
 Mental simulation is a mechanism that allows an agent to anticipate the effects of its own and other actions. Therefore, it is a core mechanism to equip an intelligent agent with the ability to plan ahead. Cognitive theories about mental simulation involve predictive coding 59 and mental motor imagery 63 , while computational approaches to mental simulation involve model-based reinforcement learning 118 , action planning 86 , or a combination of both 29 , 51 .
However, even though there is significant cognitive evidence that mental simulation happens on multiple representational layers 13 , 35 , there is a lack of approaches that use hierarchical mental simulation in hierarchical reinforcement learning.
Only a few approaches that integrate planning with reinforcement learning build on mental simulation 29 , 81 , 82 , 113 , 138 , 139 , though the mental simulation mechanisms of these models are only implemented on the high-level planning layer.
An exception is the work by Wu et al. Super , who use mental simulation on the low-level layer to determine the sub-policies to be executed.
Another exception is presented by Li et al. Super who perform mental simulation for planning on multiple task layers.

 
 
 There exist several non-hierarchical model-based reinforcement learning methods 49 , 52 , 118 that function akin to mental motor imagery: In these approaches, the policy is trained by mentally simulating action execution instead of executing the actions in the real world.
However, even though mental simulation is deemed to be a hierarchical process, we are not aware of any approach that performs hierarchical model-based reinforcement learning in the classical sense, where the policy is trained on the developing forward model.

 
 
 

### Prerequisites for sensorimotor abstraction, intrinsic motivation and mental simulation

 
 Reinforcement learning builds on policies that select actions based on the current observation and a goal or a reward function. Policies can be modelled directly, derived from value functions or combined with value functions.
In all cases, a policy is an inverse model that predicts the actions to be executed in the current state to maximise reward or to achieve a goal state.
In contrast, a forward model predicts a future world state based on the current state and a course of actions.
Both kinds of models are critical prerequisites for the mechanisms that enable transfer learning, planning, and ultimately few-shot problem-solving.
Our review shows that the vast majority of hierarchical reinforcement learning methods use inverse models for both the high-level and low-level layers.
However, some approaches exist that use an inverse model only for the low-level layer 29 , 81 , 82 , 113 , 138 , 139 .
These frameworks use a planning mechanism, driven by a forward model, to perform the high-level decision-making.
Our review demonstrates that a forward model is required for several additional mechanisms that are necessary or at least highly beneficial for transfer learning, planning, and few-shot problem-solving.
Some non-hierarchical approaches use a forward model to perform sensorimotor abstraction 52 , 91 . They achieve this with a self-supervised process where forward predictions are learned in a latent abstract space.
However, we are not aware of any hierarchical method that exploits this mechanism.

 
 
 A forward model is also required to model curiosity as an intrinsic motivation mechanism. Cognitive science suggests that curiosity is one of the most important drives of human development, and it has been demonstrated to alleviate the sparse rewards problem in reinforcement learning 91 , 103 .
Curiosity is commonly modelled as a mechanism that rewards an agent for being surprised; surprise is formally a function of the error between the predicted dynamics of a system and the actual dynamics 39 . To perform these predictions, a forward model is required.
However, with few exceptions 19 , 103 , there is a lack of approaches that use a hierarchical forward model for generating hierarchical curiosity.

 
 
 
 

## Results

 
 Tab. 1 shows the results of our review on computational hierarchical reinforcement learning approaches, in alignment with our structuring of cognitive prerequisites and mechanisms for the problem-solving abilities of intelligent biological agents (see Fig. 2 ).
We summarise our main results as follows:

 
 
 Most current few-shot problem-solving methods build on transfer learning but they do not leverage planning. 
This result is interesting because any approach with a forward model could straight-forwardly be extended to also consider planning 10 , and planning can be leveraged for few-shot problem-solving 113 . Therefore, current model-based approaches do not exploit their full potential.

 
 
 Current methods do not exploit the full potential of compositional abstract representations. 
All hierarchical reinforcement learning methods inherently perform representational abstraction, e.g., through options or subgoals, but only a few methods consider compositionality. Exceptions include natural language-based representations of actions 64 or symbolic logic-based compositional representations 88 .
None of these approaches ground these abstractions tightly to sensorimotor experiences.

 
 
 State abstraction is significantly less researched than action abstraction. 
Recent hierarchical actor-critic approaches 76 , 103 use the same state representation for each layer in the hierarchy, without performing any abstraction.
There exist a few approaches that perform state abstraction 129 , and also some that involve compositional state representations 130 . However, most of these build on hand-crafted abstraction functions that are not learned 29 .

 
 
 Curiosity and diversity, as intrinsic motivation methods, are underrepresented. 
When assessing the intrinsic motivation mechanisms of the existing approaches, we distinguish between the three most frequently used kinds of intrinsic motivation: curiosity, diversity, and subgoal discovery.
 Tab. 1 indicates that curiosity and diversity are underrepresented intrinsic motivation methods, even though there is significant cognitive evidence that these mechanisms are critical for intelligent behaviour in humans 89 .

 
 
 Finally,
 there is a lack of hierarchical mental simulation methods. 
While numerous non-hierarchical methods show how mental simulation can improve sample efficiency 23 , 49 , our summary shows that reinforcement learning with hierarchical mental simulation is yet to be explored.
As a result, inter-dependencies between sub-tasks are hard to detect.

 
 
 

## Conclusion

 
 Our review provides an overview of the cognitive foundations of hierarchical problem-solving and how these are implemented in current hierarchical reinforcement learning architectures.
Herein, we focus on few-shot problem-solving – the ability to learn to solve causally non-trivial problems with as few trials as possible or, ideally, without any trials at all (zero-shot problem-solving).
As a benchmark example, we refer to recent results from animal cognition 47 , demonstrating how a crow develops mental representations of tools to solve a previously unseen food-access problem at the first trial (see Fig. 1 ).
As our main research question, we ask for the computational prerequisites and mechanisms to enable problem-solving for artificial computational agents that are on the level of intelligent animals.

 
 
 We provide two main contributions to address this question. First, we perform a literature survey and a top-down structuring of cognitive abilities, mechanisms, and prerequisites for few-shot problem-solving (see Fig. 2 ). Second, we perform a comprehensive review of recent hierarchical reinforcement learning architectures to identify whether and how the existing approaches implement the prerequisites, mechanisms, and abilities towards few-shot problem-solving (see Tab. 1 ).
Herein, we identify five major shortcomings in the state of the art of the computational approaches.
These shortcomings are mostly based on the lack of hierarchical forward models and compositional abstractions.

 
 
 Nonetheless, we were able to identify several methods that address these gaps in the state of the art in isolation.
Not seeing any major reason why these approaches could not be integrated, we suggest that most of the tools to realise higher levels of intelligent behaviour in artificial agents have already been investigated.
The key is to combine them effectively.
This demonstrates significant potential to develop intelligent agents that can learn to solve problems on the level of intelligent biological agents.

 
 
 

## Acknowledgements

 
 Manfred Eppe, Phuong Nguyen and Stefan Wermter acknowledge funding by the German Research Foundation (DFG) through the projects IDEAS and LeCAREbot.
Stefan Wermter and Matthias Kerzel acknowledge funding by the DFG through the transregio project TRR 169 Cross-modal Learning (CML).
Christian Gumbsch and Martin Butz acknowledge funding by the DFG project number BU 1335/11-1 in the framework of the SPP program “The Active Self" (SPP 2134).
Furthermore, Martin Butz acknowledges support from a Feodor-Lynen stipend of the Humboldt Foundation as well as from the DFG, project number 198647426, “Research Training Group 1808: Ambiguity - Production and Perception” and from the DFG-funded Cluster of Excellence “Machine Learning – New Perspectives for Science´´, EXC 2064/1, project number 390727645.
The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting Christian Gumbsch.

 
 
 

## References

 
 
 1 
 
Riad Akrour, Filipe Veiga, Jan Peters, and Gerhard Neumann.

 
 Regularizing reinforcement learning with state abstraction.

 
 In Proc. of International Conference on Intelligent Robots and
Systems (IROS) , 534–539 (2018).

 

 
 2 
 
Kai Arulkumaran, Nat Dilokthanakul, Murray Shanahan, and Anil Anthony Bharath.

 
 Classifying options for deep reinforcement learning.

 
 In Proc. of IJCAI Workshop on Deep Reinforcement Learning:
Frontiers and Challenges .

 
 http://arxiv.org/abs/1604.08153 (2016).

 

 
 3 
 
Kai Arulkumaran, Marc Peter Deisenroth, Miles Brundage, and Anil Anthony
Bharath.

 
 Deep reinforcement learning: A brief survey.

 
 IEEE Signal Processing Magazine , 34(6):26–38 (2017).

 

 
 4 
 
Pierre-Luc Bacon, Jean Harb, and Doina Precup.

 
 The option-critic architecture.

 
 In Proc. of AAAI Conference on Artificial Intelligence ,
1726–1734 (2017).

 

 
 5 
 
Helen C. Barron, Raymond J. Dolan, and Timothy E.J. Behrens.

 
 Online evaluation of novel choices by simultaneous representation of
multiple memories.

 
 Nature Neuroscience , 16(10):1492–1498
(2013).

 

 
 6 
 
Lawrence W. Barsalou.

 
 Grounded cognition.

 
 Annual Review of Psychology , 59(1):617–645 (2008).

 

 
 7 
 
Daniel E Berlyne.

 
 Curiosity and exploration.

 
 Science , 153:25–33 (1966).

 

 
 8 
 
Benjamin Beyret, Ali Shafti, and A. Aldo Faisal.

 
 Dot-to-dot: Explainable hierarchical reinforcement learning for
robotic manipulation.

 
 In Proc. of International Conference on Intelligent Robots and
Systems (IROS) , 5014–5019 (2019).

 

 
 9 
 
Sebastian Blaes, Marin Vlastelica Pogancic, Jia Jie Zhu, and Georg Martius.

 
 Control what you can: Intrinsically motivated task-planning agent.

 
 Neural Information Processing Systems (NeurIPS) , 32:12541–12552 (2019).

 

 
 10 
 
Matthew Botvinick and Ari Weinstein.

 
 Model-based hierarchical reinforcement learning and human action
control.

 
 Philosophical Transactions of the Royal Society B: Biological
Sciences .

 
 10.1098/rstb.2013.0480 (2014).

 

 
 11 
 
Matthew M. Botvinick, Yael Niv, and Andrew C. Barto.

 
 Hierarchically organized behavior and its neural foundations: A
reinforcement learning perspective.

 
 Cognition , 113(3):262–280 (2009).

 

 
 12 
 
Martin V Butz.

 
 Toward a unified sub-symbolic computational theory of cognition.

 
 Frontiers in psychology , 7:1–19 (2016).

 

 
 13 
 
Martin V. Butz and Esther F. Kutter.

 
 How the mind comes into being .

 
 Oxford University Press, (2017).

 
 ISBN 9780198739692.

 

 
 14 
 
Martin V. Butz, Asya Achimova, David Bilkey, and Alistair Knott.

 
 Editors’ review and introduction: Topic event-predictive cognition:
From sensorimotor via conceptual to language-based structures and processes.

 
 Topics in Cognitive Science .

 
 10.1111/tops.12522 (2020).

 

 
 15 
 
Serkan Cabi, Sergio Gómez Colmenarejo, Matthew W. Hoffman, Misha Denil, Ziyu
Wang, and Nando de Freitas.

 
 The intentional unintentional agent: Learning to solve many
continuous control tasks simultaneously.

 
 In Proc. of Conference on Robot Learning , 207–216 (2017).

 

 
 16 
 
Caleb Chuck, Supawit Chockchowwat, and Scott Niekum.

 
 Hypothesis-driven skill discovery for hierarchical deep reinforcement
learning.

 
 In Proc. of International Conference on Intelligent Robots and
Systems (IROS) , 5572–5579 (2020).

 

 
 17 
 
Andy Clark.

 
 Whatever next? predictive brains, situated agents, and the future of
cognitive science.

 
 Behavioral and Brain Sciences , 36(3):181–204 (2013).

 

 
 18 
 
Andy Clark.

 
 Surfing uncertainty: Prediction, action and the embodied mind .

 
 Oxford University Press, (2016).

 
 ISBN 9780190217013.

 

 
 19 
 
Cédric Colas, Pierre Fournier, Olivier Sigaud, Mohamed Chetouani, and
Pierre-Yves Oudeyer.

 
 CURIOUS: Intrinsically motivated modular multi-goal reinforcement
learning.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 1331–1340 (2019).

 

 
 20 
 
Cédric Colas, Ahmed Akakzia, Pierre-Yves Oudeyer, Mohamed Chetouani, and
Olivier Sigaud.

 
 Language-conditioned goal generation: a new approach to language
grounding for RL.

 
 Technical report, (2020).

 
 http://arxiv.org/abs/2006.07043 .

 

 
 21 
 
Peter Dayan.

 
 Goal-directed control and its antipodes.

 
 Neural Networks , 22(3):213–219 (2009).

 

 
 22 
 
Edward L. Deci and Richard M. Ryan.

 
 Self-determination theory and the facilitation of intrinsic
motivation.

 
 American Psychologist , 55(1):68–78
(2000).

 

 
 23 
 
Marc Peter Deisenroth, Carl Edward Rasmussen, and Dieter Fox.

 
 Learning to control a low-cost manipulator using data-efficient
reinforcement learning.

 
 In Proc. of Robotics: Science and Systems (RSS) , 57–64
(2011).

 

 
 24 
 
Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, and Sergey Levine.

 
 Learning modular neural network policies for multi-task and
multi-robot transfer.

 
 In Proc. of International Conference on Robotics and Automation
(ICRA) , 2169–2176 (2017).

 

 
 25 
 
Amir Dezfouli, Nura W. Lingawi, and Bernard W. Balleine.

 
 Habits as action sequences: Hierarchical action control and changes
in outcome value.

 
 Philosophical Transactions of the Royal Society of London B:
Biological Sciences .

 
 10.1098/rstb.2013.0482 (2014).

 

 
 26 
 
Thomas G Dietterich.

 
 State abstraction in maxq hierarchical reinforcement learning.

 
 In Proc. of Neural Information Processing Systems (NIPS) 
(2000).

 

 
 27 
 
Ray J. Dolan and Peter Dayan.

 
 Goals and habits in the brain.

 
 Neuron , 80(2):312–325 (2013).

 

 
 28 
 
Manfred Eppe, Ewen MacLean, Roberto Confalonieri, Oliver Kutz, Marco
Schorlemmer, Enric Plaza, and Kai-Uwe Kühnberger.

 
 A computational framework for concept blending.

 
 Artificial Intelligence , 256(3):105–129
(2018).

 

 
 29 
 
Manfred Eppe, Phuong D. H. Nguyen, and Stefan Wermter.

 
 From semantics to execution: Integrating action planning with
reinforcement learning for robotic causal problem-solving.

 
 Frontiers in Robotics and AI .

 
 10.3389/frobt.2019.00123 (2019).

 

 
 30 
 
Benjamin Eysenbach, Julian Ibarz, Abhishek Gupta, and Sergey Levine.

 
 Diversity is all you need: Learning skills without a reward function.

 
 In Proc. of International Conference on Learning
Representations (ICLR) .

 
 https://openreview.net/pdf?id=SJx63jRqFm (2019).

 

 
 31 
 
Jerome Feldman, Ellen Dodge, and John Bryant.

 
 A neural theory of language and embodied construction grammar.

 
 The Oxford Handbook of Linguistic Analysis , 111 – 138
(2009).

 

 
 32 
 
Charles Fillmore.

 
 Frames and the semantics of understanding.

 
 Quaderni di Semantica , 6(2):222–254
(1985).

 

 
 33 
 
Tamar Flash and Binyamin Hochner.

 
 Motor primitives in vertebrates and invertebrates.

 
 Current Opinion in Neurobiology , 15(6):660–666 (2005).

 

 
 34 
 
Jerry A. Fodor.

 
 Language, thought and compositionality.

 
 Mind Language , 16(1):1–15 (2001).

 

 
 35 
 
Steven M Frankland and Joshua D Greene.

 
 Concepts and compositionality: In search of the brain’s language of
thought.

 
 Annual Review of Psychology , 71:273–303 (2020).

 

 
 36 
 
Nicholas T. Franklin, Kenneth A. Norman, Charan Ranganath, Jeffrey M. Zacks,
and Samuel J. Gershman.

 
 Structured event memory: A neuro-symbolic model of event cognition.

 
 Psychological Review , 127(3):327–361
(2020).

 

 
 37 
 
Kevin Frans, Jonathan Ho, Xi Chen, Pieter Abbeel, and John Schulman.

 
 Meta learning shared hierarchies.

 
 In Proc. of International Conference on Learning
Representations (ICLR) .

 
 https://openreview.net/pdf?id=SyX0IeWAW (2018).

 

 
 38 
 
Karl Friston.

 
 The free-energy principle: A unified brain theory?

 
 Nature Reviews Neuroscience , 11(2):127–138 (2010).

 

 
 39 
 
Karl Friston, Jérémie Mattout, and James Kilner.

 
 Action understanding and active inference.

 
 Biological Cybernetics , 104(1-2):137–160
(2011).

 

 
 40 
 
Karl Friston, Francesco Rigoli, Dimitri Ognibene, Christoph Mathys, Thomas
Fitzgerald, and Giovanni Pezzulo.

 
 Active inference and epistemic value.

 
 Cognitive Neuroscience , 6(4):187–214
(2015).

 

 
 41 
 
Karl Friston, Thomas FitzGerald, Francesco Rigoli, Philipp Schwartenbeck, John
O’Doherty, and Giovanni Pezzulo.

 
 Active inference and learning.

 
 Neuroscience and Biobehavioral Reviews , 68:862–879
(2016).

 

 
 42 
 
Javier García and Fernando Fernández.

 
 A comprehensive survey on safe reinforcement learning.

 
 Journal of Machine Learning Research , 16:1437–1480
(2015).

 

 
 43 
 
Peter Gärdenfors.

 
 The geometry of meaning: Semantics based on conceptual spaces .

 
 MIT Press, (2014).

 

 
 44 
 
Behzad Ghazanfari, Fatemeh Afghah, and Matthew E. Taylor.

 
 Sequential association rule mining for autonomously extracting
hierarchical task structures in reinforcement learning.

 
 IEEE Access , 8:11782–11799 (2020).

 

 
 45 
 
Dibya Ghosh, Abhishek Gupta, and Sergey Levine.

 
 Learning actionable representations with goal conditioned policies.

 
 In Proc. of International Conference on Learning
Representations (ICLR) .

 
 https://openreview.net/pdf?id=Hye9lnCct7 (2019).

 

 
 46 
 
James J. Gibson.

 
 The ecological approach to visual perception .

 
 Lawrence Erlbaum Associates, (1979).

 

 
 47 
 
Romana Gruber, Martina Schiestl, Markus Boeckle, Anna Frohnwieser, Rachael
Miller, Russell D. Gray, Nicola S. Clayton, and Alex H. Taylor.

 
 New caledonian crows use mental representations to solve metatool
problems.

 
 Current Biology , 29(4):686–692 (2019).

 

 
 48 
 
Christian Gumbsch, Martin V. Butz, and Georg Martius.

 
 Autonomous identification and goal-directed invocation of
event-predictive behavioral primitives.

 
 IEEE Transactions on Cognitive and Developmental Systems ,
1–19 (2019).

 

 
 49 
 
David Ha and Jürgen Schmidhuber.

 
 Recurrent world models facilitate policy evolution.

 
 In Proc. of Neural Information Processing Systems (NIPS) ,
2450–2462 (2018).

 

 
 50 
 
Tuomas Haarnoja, Kristian Hartikainen, Pieter Abbeel, and Sergey Levine.

 
 Latent space policies for hierarchical reinforcement learning.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 2965–2975 (2018).

 

 
 51 
 
Muhammad Burhan Hafez, Cornelius Weber, Matthias Kerzel, and Stefan Wermter.

 
 Improving robot dual-system motor learning with intrinsically
motivated meta-control and latent-space experience imagination.

 
 Robotics and Autonomous Systems , 133:103630 (2020).

 

 
 52 
 
Danijar Hafner, Timothy Lillicrap, and Mohammad Norouzi.

 
 Dream to control: Learning behaviors by latent imagination.

 
 In Proc. of International Conference on Learning
Representations (ICLR) .

 
 https://openreview.net/pdf?id=S1lOTC4tDS (2020).

 

 
 53 
 
Dongqi Han, Kenji Doya, and Jun Tani.

 
 Self-organization of action hierarchy and compositionality by
reinforcement learning with recurrent neural networks.

 
 Neural Networks , 129:149–162 (2020).

 

 
 54 
 
John Dylan Haynes, David Wisniewski, Kai Gorgen, Ida Momennejad, and Carlo
Reverberi.

 
 FMRI decoding of intentions: Compositionality, hierarchy and
prospective memory.

 
 In Proc. of International Winter Conference on Brain-Computer
Interface (BCI) .

 
 10.1109/IWW-BCI.2015.7073031 (2015).

 

 
 55 
 
Nicolas Heess, Greg Wayne, Yuval Tassa, Timothy Lillicrap, Martin Riedmiller,
and David Silver.

 
 Learning and transfer of modulated locomotor controllers.

 
 Technical report, (2016).

 
 http://arxiv.org/abs/1610.05182 .

 

 
 56 
 
Mary Hegarty.

 
 Mechanical reasoning by mental simulation.

 
 Trends in Cognitive Sciences , 8(6):280–285 (2004).

 

 
 57 
 
Donald J Hejna Iii, Pieter Abbeel, and Lerrel Pinto.

 
 Hierarchically decoupled morphological transfer.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 11409–11420 (2020).

 

 
 58 
 
J. Hoffmann.

 
 Anticipatory behavioral control.

 
 In M. V. Butz, O. Sigaud, and P. Gerard, editors, Anticipatory
Behavior in Adaptive Learning Systems: Foundations, Theories, and Systems ,
44–65. Springer (2003).

 

 
 59 
 
Yanping Huang and Rajesh P.N. Rao.

 
 Predictive coding.

 
 Wiley Interdisciplinary Reviews: Cognitive Science , 2(5):580–593 (2011).

 

 
 60 
 
Dania Humaidan, Sebastian Otte, and Martin V. Butz.

 
 Fostering event compression using gated surprise.

 
 In Proc. of International Conference on Artificial Neural
Networks (ICANN) , 155–167 (2020).

 

 
 61 
 
David Isele, Mohammad Rostami, and Eric Eaton.

 
 Using task features for zero-shot knowledge transfer in lifelong
learning.

 
 In Proc. of International Joint Conference on Artificial
Intelligence (IJCAI) , 1620–1626 (2016).

 

 
 62 
 
Max Jaderberg, Volodymyr Mnih, Wojciech Marian Czarnecki, Tom Schaul, Joel Z
Leibo, David Silver, and Koray Kavukcuoglu.

 
 Reinforcement learning with unsupervised auxiliary tasks.

 
 International Conference on Learning Representations (ICLR) .

 
 http://arxiv.org/abs/1611.05397 (2017).

 

 
 63 
 
M. Jeannerod.

 
 Mental imagery in the motor context.

 
 Neuropsychologia , 33(11):1419–1432
(1995).

 

 
 64 
 
Yiding Jiang, Shixiang Gu, Kevin Murphy, Chelsea Finn, and Google Research.

 
 Language as an abstraction for hierarchical deep reinforcement
learning.

 
 In Proc. of Neural Information Processing Systems (NeurIPS) ,
9414–9426 (2019).

 

 
 65 
 
Daniel Kahnemann and Amos Tversky.

 
 The simulation heuristics.

 
 In Judgement under uncertainty: Heuristics and biases ,
201–208. (1982).

 

 
 66 
 
Frédéric Kaplan and Pierre-Yves Oudeyer.

 
 Maximizing learning progress: an internal reward system for
development.

 
 In Embodied Artificial Intelligence , 259–270. (2004).

 

 
 67 
 
Karl Josef Klauer.

 
 Teaching for analogical transfer as a means of improving
problem-solving, thinking and learning.

 
 Instructional Science , 18(3):179–192
(1989).

 

 
 68 
 
David C. Knill and Alexandre Pouget.

 
 The bayesian brain: The role of uncertainty in neural coding and
computation.

 
 Trends in Neurosciences , 27(12):712–719
(2004).

 

 
 69 
 
Tejas D. Kulkarni, Karthik R. Narasimhan, Ardavan Saeedi, and Joshua B.
Tenenbaum.

 
 Hierarchical deep reinforcement learning: Integrating temporal
abstraction and intrinsic motivation.

 
 In Proc. of Neural Information Processing Systems (NIPS) ,
3675–3683 (2016).

 

 
 70 
 
Wilfried Kunde, Katrin Elsner, and Andrea Kiesel.

 
 No anticipation-no action: the role of anticipation in action and
perception.

 
 Cogn Process , 8(2):71–78 (2007).

 

 
 71 
 
Gina R. Kuperberg.

 
 Tea with milk? A hierarchical generative framework of sequential
event comprehension.

 
 Topics in Cognitive Science .

 
 10.1111/tops.12518 (2020).

 

 
 72 
 
Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman.

 
 Building machines that learn and think like people.

 
 Behavioral and Brain Sciences , 40:1–72 (2017).

 

 
 73 
 
George Lakoff and Mark Johnson.

 
 Philosophy in the flesh .

 
 Basic Books, (1999).

 

 
 74 
 
Guillaume Lample and Devendra Singh Chaplot.

 
 Playing fps games with deep reinforcement learning.

 
 AAAI Conference on Artificial Intelligence , 2140–2146
(2017).

 

 
 75 
 
Timothée Lesort, Natalia Díaz-Rodríguez, Jean-François Goudou,
and David Filliat.

 
 State representation learning for control: An overview.

 
 Neural Networks , 108:379–392 (2018).

 

 
 76 
 
Andrew Levy, George Konidaris, Robert Platt, and Kate Saenko.

 
 Learning multi-level hierarchies with hindsight.

 
 In Proc. of International Conference on Learning
Representations (ICLR) .

 
 https://openreview.net/pdf?id=ryzECoAcY7 (2019).

 

 
 77 
 
Alexander C Li, Carlos Florensa, Ignasi Clavera, and Pieter Abbeel.

 
 Sub-policy adaptation for hierarchical reinforcement learning.

 
 In Proc. of International Conference on Learning
Representations (ICLR) .

 
 https://openreview.net/forum?id=ByeWogStDS (2020).

 

 
 78 
 
Yuxi Li.

 
 Deep reinforcement learning: An overview.

 
 Technical report, (2018).

 
 https://arxiv.org/abs/ .

 

 
 79 
 
Zhuoru Li, Akshay Narayan, and Tze Yun Leong.

 
 An efficient approach to model-based hierarchical reinforcement
learning.

 
 AAAI Conference on Artificial Intelligence , 3583–3589
(2017).

 

 
 80 
 
George Loewenstein.

 
 The psychology of curiosity: A review and reinterpretation.

 
 Psychological bulletin , 116(1):75–98
(1994).

 

 
 81 
 
Daoming Lyu, Fangkai Yang, Bo Liu, and Steven Gustafson.

 
 Sdrl: Interpretable and data-efficient deep reinforcement learning
leveraging symbolic planning.

 
 In Proc. of AAAI Conference on Artificial Intelligence ,
2970–2977 (2019).

 

 
 82 
 
Aaron Ma, Michael Ouimet, and Jorge Cortés.

 
 Hierarchical reinforcement learning via dynamic subspace search for
multi-agent planning.

 
 Autonomous Robots , 44(3-4):485–503
(2020).

 

 
 83 
 
Marlos C. Machado, Marc G. Bellemare, and Michael Bowling.

 
 A laplacian framework for option discovery in reinforcement learning.

 
 International Conference on Machine Learning (ICML) ,
5:3567–3582 (2017).

 

 
 84 
 
Marvin Minsky.

 
 A framework for representing knowledge.

 
 Technical report, MIT-AI Laboratory Memo 306, (1974).

 

 
 85 
 
Ofir Nachum, Shixiang Gu, Honglak Lee, and Sergey Levine.

 
 Data-efficient hierarchical reinforcement learning.

 
 In Proc. of Neural Information Processing Systems (NeurIPS) ,
3303–3313 (2018).

 

 
 86 
 
Dana Nau, T.C. Au, O. Ilghami, U. Kuter, J.W. Murdock, Dan Wu, and F. Yaman.

 
 SHOP2: An HTN planning system.

 
 Journal of Artificial Intelligence Research , 20(1):379–404 (2003).

 

 
 87 
 
Emre O. Neftci and Bruno B. Averbeck.

 
 Reinforcement learning in artificial and biological systems.

 
 Nature Machine Intelligence , 1(3):133–143
(2019).

 

 
 88 
 
Junhyuk Oh, Satinder Singh, Honglak Lee, and Pushmeet Kohli.

 
 Zero-shot task generalization with multi-task deep reinforcement
learning.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 2661–2670 (2017).

 

 
 89 
 
Pierre-Yves Oudeyer.

 
 Computational theories of curiosity-driven learning, (2018).

 
 http://arxiv.org/abs/1802.10546 .

 

 
 90 
 
Pierre-Yves Oudeyer, Frédéric Kaplan, and Verena V. Hafner.

 
 Intrinsic motivation systems for autonomous mental development.

 
 In Proc. of IEEE Transactions on Evolutionary Computation ,
265–286 (2007).

 

 
 91 
 
Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell.

 
 Curiosity-driven exploration by self-supervised prediction.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 2778–2787 (2017).

 

 
 92 
 
Deepak Pathak, Dhiraj Gandhi, and Abhinav Gupta.

 
 Self-supervised exploration via disagreement.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 5062–5071 (2019).

 

 
 93 
 
David N. Perkins and Gavriel Salomon.

 
 Transfer of learning.

 
 In International Encyclopedia of Education , 6452–6457.
Pergamon Press.

 
 ISBN 0309059131131 (1992).

 

 
 94 
 
Wojciech Pisula.

 
 Play and exploration in animals - a comparative analysis.

 
 Polish Psychological Bulletin , 39(2):104–107 (2008).

 

 
 95 
 
Friedemann Pulvermüller.

 
 Brain embodiment of syntax and grammar: Discrete combinatorial
mechanisms spelt out in neuronal circuits.

 
 Brain and Language , 112(3):167 – 179
(2010).

 

 
 96 
 
Zhiqian Qiao, Zachariah Tyree, Priyantha Mudalige, Jeff Schneider, and John M
Dolan.

 
 Hierarchical reinforcement learning method for autonomous vehicle
behavior planning.

 
 In Proc. of International Conference on Intelligent Robots and
Systems (IROS) , 6084–6089 (2020).

 

 
 97 
 
Ahmed H Qureshi, Jacob J Johnson, Yuzhe Qin, Taylor Henderson, Byron Boots, and
Michael C Yip.

 
 Composing task-agnostic policies with deep reinforcement learning.

 
 In Proc. of International Conference on Learning
Representations (ICLR) .

 
 https://openreview.net/pdf?id=H1ezFREtwH (2020).

 

 
 98 
 
Gabriel A. Radvansky and Jeffrey M. Zacks.

 
 Event perception.

 
 Wiley Interdisciplinary Reviews: Cognitive Science , 2(6):608–620 (2011).

 

 
 99 
 
Jacob Rafati and David C. Noelle.

 
 Learning representations in model-free hierarchical reinforcement
learning.

 
 In Proc. of AAAI Conference on Artificial Intelligence ,
10009–10010 (2019).

 

 
 100 
 
Daniel Rasmussen, Aaron Voelker, and Chris Eliasmith.

 
 A neural model of hierarchical reinforcement learning.

 
 PLOS ONE .

 
 10.1371/journal.pone.0180234 (2017).

 

 
 101 
 
Carlo Reverberi, Kai Görgen, and John-Dylan Haynes.

 
 Compositionality of rule representations in human prefrontal cortex.

 
 Cerebral Cortex , 22:1237–1246 (2012).

 

 
 102 
 
Martin Riedmiller, Roland Hafner, Thomas Lampe, Michael Neunert, Jonas Degrave,
Tom Van de Wiele, Volodymyr Mnih, Nicolas Heess, and Jost Tobias
Springenberg.

 
 Learning by playing - solving sparse reward tasks from scratch.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 6910–6919 (2018).

 

 
 103 
 
Frank Röder, Manfred Eppe, Phuong D. H. Nguyen, and Stefan Wermter.

 
 Curious hierarchical actor-critic reinforcement learning.

 
 In Proc. of International Conference on Artificial Neural
Networks (ICANN) , 408–419 (2020).

 

 
 104 
 
Andrew M Saxe, Adam C Earle, and Benjamin Rosman.

 
 Hierarchy through composition with multitask lmdps.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 3017–3026 (2017).

 

 
 105 
 
Stefan Schaal.

 
 Dynamic movement primitives–a framework formotor control in humans
and humanoid robotics.

 
 In Adaptive Motion of Animals and Machines , 261–280.

 
 ISBN 9784431241645 (2006).

 

 
 106 
 
Tom Schaul and Mark Ring.

 
 Better generalization with forecasts.

 
 In Proc. of International Joint Conference on Artificial
Intelligence (IJCAI) , 1656–1662 (2013).

 

 
 107 
 
Guido Schillaci, Antonio Pico Villalpando, Verena V Hafner, Peter Hanappe,
David Colliaux, and Timothée Wintz.

 
 Intrinsic motivation and episodic memories for robot exploration of
high-dimensional sensory spaces.

 
 Adaptive Behavior .

 
 10.1177/1059712320922916 (2020).

 

 
 108 
 
Jürgen Schmidhuber.

 
 A possibility for implementing curiosity and boredom in
model-building neural controllers.

 
 From Animals to Animats , 222–227 (1991).

 

 
 109 
 
Jürgen Schmidhuber.

 
 Formal theory of creativity, fun, and intrinsic motivation.

 
 IEEE Transactions on Autonomous Mental Development , 2(3):230–247 (2010).

 

 
 110 
 
John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter
Abbeel.

 
 High-dimensional continuous control using generalized advantage
estimation.

 
 Technical report, (2015).

 
 https://arxiv.org/abs/1506.02438 .

 

 
 111 
 
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

 
 Proximal policy optimization algorithms.

 
 Technical report, (2017).

 
 https://arxiv.org/pdf/1707.06347.pdf .

 

 
 112 
 
Tanmay Shankar, Lerrel Pinto, Shubham Tulsiani, and Abhinav Gupta.

 
 Discovering motor programs by recomposing demonstrations.

 
 In Proc. of International Conference on Learning
Representations .

 
 https://openreview.net/attachment?id=rkgHY0NYwr name=original_pdf 
(2020).

 

 
 113 
 
Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman.

 
 Dynamics-aware unsupervised discovery of skills.

 
 In Proc. of International Conference on Learning
Representations (ICLR) .

 
 https://openreview.net/forum?id=HJgLZR4KvH (2020).

 

 
 114 
 
Yeon Soon Shin and Sarah DuBrow.

 
 Structuring memory through inference-based event segmentation.

 
 Topics in Cognitive Science .

 
 10.1111/tops.12505 (2020).

 

 
 115 
 
Sungryull Sohn, Junhyuk Oh, Honglak Lee, and Google Brain.

 
 Hierarchical reinforcement learning for zero-shot generalization with
subtask dependencies.

 
 In Proc. of Neural Information Processing Systems (NeurIPS) 
(2018).

 

 
 116 
 
Yuuya Sugita and Martin V. Butz.

 
 Compositionality and embodiment in harmony.

 
 In Pierre-Yves Oudeyer, editor, AMD Newsletter , volume 8,
8–9. IEEE CIS (2011).

 

 
 117 
 
Richard S. Sutton and Andrew G. Barto.

 
 Reinforcement learning: An introduction .

 
 pMIT, second edition edition, (2018).

 

 
 118 
 
Richard S. Sutton and Richard S. Sutton.

 
 Integrated architectures for learning, planning, and reacting based
on approximating dynamic programming.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 216–224 (1990).

 

 
 119 
 
Richard S Sutton, Doina Precup, and Satinder Singh.

 
 Between mdps and semi-mdps: A framework for temporal abstraction in
reinforcement learning.

 
 Artificial Intelligence , 112:181–211 (1999).

 

 
 120 
 
Zoltán Gendler Szabó.

 
 Compositionality.

 
 In Edward N. Zalta, editor, The Stanford Encyclopedia of
Philosophy . Stanford University.

 
 https://plato.stanford.edu/archives/fall2020/entries/compositionality/ 
(2020).

 

 
 121 
 
Shelley E. Taylor, Lien B. Pham, Inna D. Rivkin, and David A. Armor.

 
 Harnessing the imagination: Mental simulation, self-regulation, and
coping.

 
 American Psychologist , 53(4):429–439
(1998).

 

 
 122 
 
Chen Tessler, Shahar Givony, Tom Zahavy, Daniel J Mankowitz, and Shie Mannor.

 
 A deep hierarchical approach to lifelong learning in minecraft.

 
 In Proc. of AAAI Conference on Artificial Intelligence , 1553
– 1561 (2017).

 

 
 123 
 
Edward Lee Thorndike.

 
 Animal intelligence: Experimental studies .

 
 Transaction Publishers, (1970).

 

 
 124 
 
Edward Chace Tolman and Charles H Honzik.

 
 Introduction and removal of reward, and maze performance in rats.

 
 University of California Publications in Psychology (1930).

 

 
 125 
 
Momchil S. Tomov, Samyukta Yagati, Agni Kumar, Wanqian Yang, and Samuel J.
Gershman.

 
 Discovery of hierarchical representations for efficient planning.

 
 PLoS Computational Biology .

 
 10.1371/journal.pcbi.1007594 (2020).

 

 
 126 
 
Marc Toussaint, Kelsey R Allen, Kevin A Smith, and Joshua B Tenenbaum.

 
 Differentiable physics and stable modes for tool-use and manipulation
planning.

 
 In Proc. of Robotics: Science and Systems (RSS) .

 
 10.15607/RSS.2018.XIV.044 (2018).

 

 
 127 
 
Mark Turner.

 
 The origin of ideas .

 
 Oxford University Press, (2014).

 

 
 128 
 
Alexander Vezhnevets, Volodymyr Mnih, John Agapiou, Simon Osindero, Alex
Graves, Oriol Vinyals, and Koray Kavukcuoglu.

 
 Strategic attentive writer for learning macro-actions.

 
 In Proc. of Neural Information Processing Systems (NIPS) ,
3494–3502 (2016).

 

 
 129 
 
Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max
Jaderberg, David Silver, and Koray Kavukcuoglu.

 
 Feudal networks for hierarchical reinforcement learning.

 
 In Proc. of International Conference on Machine Learning
(ICML) , 3540–3549 (2017).

 

 
 130 
 
Alexander Sasha Vezhnevets, Yuhuai Tony Wu, Maria Eckstein, Rémi Leblond, and
Joel Z Leibo.

 
 Options as responses: Grounding behavioural hierarchies in
multi-agent reinforcement learning.

 
 In Proc. of International Conference on Machine Learning
(ICML) .

 
 https://proceedings.icml.cc/paper/2020/file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf 
(2020).

 

 
 131 
 
Gary L. Wells and Igor Gavanski.

 
 Mental simulation of causality.

 
 Journal of Personality and Social Psychology , 56(2):161–169 (1989).

 

 
 132 
 
M. Werning, W. Hinzen, and E. Machery.

 
 The Oxford handbook of compositionality .

 
 Oxford University Press, (2012).

 

 
 133 
 
Jan M. Wiener and Hanspeter A. Mallot.

 
 ’fine-to-coarse’ route planning and navigation in regionalized
environments.

 
 Spatial cognition and computation , 3(4):331–358 (2003).

 

 
 134 
 
D. M. Wolpert and M. Kawato.

 
 Multiple paired forward and inverse models for motor control.

 
 Neural Networks , 11(7-8):1317–1329
(1998).

 

 
 135 
 
F. Wörgötter, E. E. Aksoy, N. Krüger, J. Piater, A. Ude, and
M. Tamosiunaite.

 
 A simple ontology of manipulation actions based on hand-object
relations.

 
 Autonomous Mental Development, IEEE Transactions on ,
5(2):117–134 (2013).

 

 
 136 
 
Bohan Wu, Jayesh K. Gupta, and Mykel J. Kochenderfer.

 
 Model primitive hierarchical lifelong reinforcement learning.

 
 In Proc. of International Joint Conference on Autonomous Agents
and Multiagent Systems (AAMAS) , 34–42 (2019).

 

 
 137 
 
Markus Wulfmeier, Abbas Abdolmaleki, Roland Hafner, Tobias Springenberg,
Michael Neunert, Tim Hertweck, Thomas Lampe, Noah Siegel, Nicolas Heess, and
Martin Riedmiller.

 
 Regularized hierarchical policies for compositional transfer in
robotics.

 
 Technical report, (2020).

 
 https://arxiv.org/pdf/1906.11228.pdf .

 

 
 138 
 
Kazeto Yamamoto, Takashi Onishi, and Yoshimasa Tsuruoka.

 
 Hierarchical reinforcement learning with abductive planning.

 
 In Proc. of ICML/IJCAI/AAMAS 2018 Workshop on Planning and
Learning (PAL-18) .

 
 http://arxiv.org/abs/1806.10792 (2018).

 

 
 139 
 
Fangkai Yang, Daoming Lyu, Bo Liu, and Steven Gustafson.

 
 Peorl: Integrating symbolic planning and hierarchical reinforcement
learning for robust decision-making.

 
 In Proc. of Interational Joint Conference on Artificial
Intelligence (IJCAI) , 4860–4866 (2018).

 

 
 140 
 
Zhaoyang Yang, Kathryn Merrick, Lianwen Jin, and Hussein A. Abbass.

 
 Hierarchical deep reinforcement learning for continuous action
control.

 
 IEEE Transactions on Neural Networks and Learning Systems ,
29(11):5174–5184 (2018).

 

 
 141 
 
Jeffrey M Zacks and Barbara Tversky.

 
 Event structure in perception and conception.

 
 Psychological bulletin , 127(1):3–21
(2001).

 

 
 142 
 
Jeffrey M. Zacks, Nicole K. Speer, Khena M. Swallow, Todd S. Braver, and
Jeremy R. Reyonolds.

 
 Event perception: A mind/brain perspective.

 
 Psychological Bulletin , 133(2):273–293
(2007).

 

 
 143 
 
Tianren Zhang, Shangqi Guo, Tian Tan, Xiaolin Hu, and Feng Chen.

 
 Generating adjacency-constrained subgoals in hierarchical
reinforcement learning.

 
 In Proc. of Neural Information Processing Systems (NeurIPS) .

 
 https://github.com/trzhang0116/HRAC. (2020).