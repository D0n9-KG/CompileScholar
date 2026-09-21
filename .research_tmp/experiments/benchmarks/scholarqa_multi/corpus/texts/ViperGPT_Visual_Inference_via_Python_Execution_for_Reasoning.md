# ViperGPT: Visual Inference via Python Execution for Reasoning

Dídac Surís*, Sachit Menon*, Carl Vondrick  
Columbia University  
viper.cs.columbia.edu

# Abstract

Answering visual queries is a complex task that requires both visual processing and reasoning. End-to-end models, the dominant approach for this task, do not explicitly differentiate between the two, limiting interpretability and generalization. Learning modular programs presents a promising alternative, but has proven challenging due to the difficulty of learning both the programs and modules simultaneously. We introduce ViperGPT, a framework that leverages code-generation models to compose vision-and-language models into subroutines to produce a result for any query.

ViperGPT utilizes a provided API to access the available modules, and composes them by generating Python code that is later executed. This simple approach requires no further training, and achieves state-of-the-art results across various complex visual tasks.

# 1. Introduction

How many muffins can each kid in Figure 1 (top) eat for it to be fair? To answer this, we might 1) find the children and the muffins in the image, 2) count how many there are of each, and 3) reason that 'fair' implies an even split, hence divide. People find it natural to compositionally combine individual steps together to understand the visual world. Yet, the dominant approach in the field of computer vision remains end-to-end models, which do not inherently leverage this compositional reasoning.

Although the field has made large progress on individual tasks such as object recognition and depth estimation, end-to-end approaches to complex tasks must learn to implicitly perform all tasks within the forward pass of a neural network. Not only does this fail to make use of the advances in fundamental vision tasks at different steps, it does not make use of the fact that computers can perform mathematical operations (e.g., division) easily without machine learning. We cannot trust neural models to generalize systematically to different numbers of muffins or children. End

to-end models also produce fundamentally uninterpretable decisions – there is no way to audit the result of each step to diagnose failure. As models grow increasingly data and compute-hungry, this approach grows increasingly untenable. We would like to perform new tasks without additional training by recombining our existing models in new ways.

What limits us from creating such modular systems for more complex tasks? In previous years, the pioneering works of Neural Module Networks [2, 27, 19] attempted to decompose tasks into simpler modules. By training end-to-end with modules rearranged in different ways for different problems, the hope was that each module would learn their appropriate function and thereby become reusable. However, numerous issues made this approach difficult to extend to the real world.

In particular, program generation relied on hand-tuned natural language parsers [2], or otherwise required reinforcement learning from scratch and were thus difficult to optimize [19, 27]. In each case, program generation was highly domain-limited. Furthermore, learning the perceptual models jointly with the program generator made training even more difficult, often failing to produce the intended modular structure [3, 48].

In this work, we present ViperGPT $^1$ , a framework that overcomes these bottlenecks by leveraging code generating large language models (e.g. GPT-3 Codex [9]) to flexibly compose vision models based on any textual query that defines the task. It creates customized programs for each query that take images or videos as argument and return the result of the query for that image or video. We show that providing Codex an API exposing various visual capabilities (e.g. find, compute_depth), just as one might provide an engineer, is sufficient for the creation of these programs.

The model's prior training on code enables it to reason about how to use these functions and implement the relevant logic. Our results demonstrate that this simple approach delivers remarkable zero-shot performance (i.e. without ever training on task specific images).

Our simple approach enjoys many benefits: it is 1) interpretable, as all the steps are explicit as code function calls

arXiv:2303.08128v1 [cs.CV] 14 Mar 2023

*Equal contribution. Order determined via coin flip and may be listed either way.

1We name our method after a snake because it executes Python code.

Query: How many muffins can each kid have for it to be fair?

![](dt=2026-06-05/ht=17/b1b6d5e497afac8143a0d4aeb5948fd95c7b24eee64df6846579324eb9dca510.jpg)

# Generated Code

def execute_command(image):

image_batch = ImagePatch(image)

muffin_patches = image_batch.find("muffin")

kidpatches $=$ imagepatch.find("kid")

return str(len(muffinpatches) // len(kidpatches))

# Execution

![](dt=2026-06-05/ht=17/c6d6be93bc141e8ed678bf714d0f0484bfc78bc2bbba2deae3c9f8d73322bce5.jpg)

![](dt=2026-06-05/ht=17/7e28ca6df9d1d7d3a3d40d683eda9ff0cea6bd099d18cce029bdef94b82957d6.jpg)

![](dt=2026-06-05/ht=17/79ee3a562b3e13548c73210e38ee93f29643f4f2dea27db8990491955e63317c.jpg)

# Query: Drink with zero alcohol

![](dt=2026-06-05/ht=17/d6c795864f109f701d6bf445e74c8b0ed9e8dd1c3856cc6194902fb22446530a.jpg)

def execute_command(image):

image_batch = ImagePatch(image)

drink_patches = image_batch.find("drink")

for drink_batch in drink_batches:

drink_name $=$ drinkpatch.simple_query("What is this?")

alcoholic = lvm_query(f"Does the {drink_name} have alcohol?")

if alcoholic $= =$ "no":

return drink patch

return None

drink patches=

![](dt=2026-06-05/ht=17/4e79e560b4eb671e1984bcbbc76dc7d9941372d876752ef68a30804ccabf1cc5.jpg)

![](dt=2026-06-05/ht=17/b25d0f5b2e7f28a0d456f39bac8c0d80c9daedad1077a600c76f8ab16e649084.jpg)

![](dt=2026-06-05/ht=17/c7a2e33143510325b06271a027a553facfd7e8b6029c48e21e032ecde010eb34.jpg)

![](dt=2026-06-05/ht=17/fd67642cfe4036c342ed6133ce8ffbf4bb244e0c79ad3e4be4ebfe21a7cbc0f1.jpg)

![](dt=2026-06-05/ht=17/196dcabd0d936019a6929fb4c3c3518124bb006b78409ac92aad9de7baf7b9b7.jpg)

$\text{品}$ drink_name $\equiv$ 'tullamore dew'   
alcoholic $=$ 'yes

drink_name = 'bacardi'  
alcoholic = 'yes'

drink_name = 'gin'  
alcoholic = 'yes'

drink_name = 'dr pepper'  
alcoholic = 'no'

![](dt=2026-06-05/ht=17/c7a7b2a50f88e73fae20972aa12fa57115e5b615809e08a0c6290b9c32ba3fe4.jpg)

# Query: What would the founder of the brand of the car on the left say to the founder of the brand of the car on the right?

![](dt=2026-06-05/ht=17/ec2b03311889a315af4df712776c8b4635423f5a3acc8a69ddab5b2c942c6e3d.jpg)

def execute_command(image):

image_batch = ImagePatch(image)

car_patches = imagePatch.find("car")

car_patches.sort(key=lambda car: car-horizontal_center)

left_car = car_patches[0]

right_car = car_patches[-1]

left_car_brand = left_car.simple_query("What is the brand of this car?")

right_car_brand = right_car.simple_query("What is the brand of this car?")

left_
car_founder = lmm_query(f"Who is the founder of {left_car_brand}?"）

right_car-founder = lvm_query(f"Who is the founder of {right_car_brand}?"

return lmm_query(f"What would {left_car_founder} say to {right_car_founder}?")

car_patches

image_batch.find("car")

![](dt=2026-06-05/ht=17/a6fbf3a2abfdb2727d46675b2cd5f75442439e74c4ef690579e85dfca0a2147d.jpg)

![](dt=2026-06-05/ht=17/b22b0bf76c9fd510fe03957b8fe65c39b5805174bb2f738fc5b7f5d8ab0d6de7.jpg)

一

$\triangleright$ left car brand $\equiv$ Lamborghini

$\triangleright$ right car brand $\equiv$ ferrari

$\triangleright$ left car founder $\equiv$ Ferruccio Lamborghini

▶right car founder='Enzo Ferrari

Result: "Ferruccio Lamborghini might say, 'It's been an honor to be a rival of yours for so many years, Enzo.

May our cars continue to push each other to be better and faster!"

# Query: Return the two kids that are furthest from the woman right before she hugs the girl

![](dt=2026-06-05/ht=17/1c7a0869db82dc4f703cea3602a8d14288eb287e4f9af0759d24c9e95208fcad.jpg)

def execute_command(video):

video_segment = VideoSegment(video)

hug detected $=$ False

for i, frame in enumerate(video segment.frame iterator():

if frame_exists("woman") and frame_exists("girl") and

frame.simple_query("Is the woman hugging the girl?") == "yes":

hug Detected = True

break

if hugdetected:

index frame $=$ i-1

frame_of_interest = ImagePatch(video_segment, index_frame)

woman_patches = frame_of_interest.find("woman")

woman patch = woman patches[0]

kid patches = frame of interest.find("kid")

kid.patches.sort(key=lambda kid:distance(kid,womanpatch))

kid_batch_1 = kid_batches[-1]

kid patch 2 = kid patches[-2]

return [kidpatch_1, kidpatch_2]

![](dt=2026-06-05/ht=17/d848512755e6d0565418b5837dc4dce13e4dbe147688a6cda679c3b9de328845.jpg)

frame of interest=

![](dt=2026-06-05/ht=17/c9db8cb391ba818984fa5e8c4d3f1e7e16baecf6afad52456a04c758c3756a56.jpg)

kid patches=

![](dt=2026-06-05/ht=17/875de712a155657750bc6b4563aed312b211f5173f02b03a3c44de91a4f24630.jpg)

sort(...distance...)

kid patches=

![](dt=2026-06-05/ht=17/8b1dda16d59e97e3eeff10555cae7a1e9d54fc74745c12b2646fc1ec986b001d.jpg)

Result:

![](dt=2026-06-05/ht=17/5b688c93e6eaa1bd6596092ba77e7e8bd3c11d906b836be5aa986f77be32b98d.jpg)

Query: What color do you get if you combine the colors of the viper and the flower?

![](dt=2026-06-05/ht=17/aef6b2890a1a88253b2cfcbadebcb91420e18bc11d3919517a3675f4388c85f9.jpg)

def execute_command(image):

image_batch = ImagePatch(image)

viper_patches = imagePatch.find("viper")

flowerpatches = imagePatch.find("flower")

viper_batch = viper_batches[0]

flower_batch = flower_batches[0]

viper_color = viper_batch.simple_query("What color is the viper?")

flower_color = flowerPatch.simple_query("What color is the flower?")

color = lmm_query(f"What color do you get if you combine the colors

{viper_color} and {flower_color}?")

return color

▶viper_batch=

![](dt=2026-06-05/ht=17/ac328bd5da3511c46708dbc57464c8ac0990ed04f0f72e49c74a51eb85c351bf.jpg)

$\triangleright$ flower_batch=

▶viper color

$\triangleright$ flower color='red

color='purple'

Result:"purple"

# Query: Tell me about the competition between the two skyscrapers in the image.

![](dt=2026-06-05/ht=17/838905868a77e2eca618a554220b9b1fefb7d4034eec5c050cb8a654e065bda4.jpg)

def execute_command(image):

image_batch = ImagePatch(image)

skyscraper patches = image patch.find("skyscraper")

skyscraper_batch 1 = skyscraper_batches[0]

skyscraper_batch_2 = skyscraper_batches[1]

skyscraper 1_name = skyscraper.patch_1.simple_query("What is the name of this skyscraper?")

skvscraper 2 name = skvscraper patch 2.simple query("What is the name of this skyscraper?")

return lvm_query(f"Tell me about the competition between {skyscraper_1_name} and {skyscraper_2_name}")

![](dt=2026-06-05/ht=17/2c443d5ebd0720e0b0620189dbefcabc8a0b0ced7c43b657ad1909cae0cc26d5.jpg)

skyscraper_1_name='chrysler building'  
skyscraper_2_name='empire state built'

Result: "The competition between the Chrysler Building and the Empire State Building is one of the most famous rivalries in the history of architecture. Built in the 1930s in New York City, the two skyscrapers were the tallest buildings in the world for a time. The Chrysler Building, designed by William Van Alen and completed in 1930, was the first to reach a height of 1,046 feet, temporarily ..."

with intermediate values that can be inspected; 2) logical, as it explicitly uses built-in Python logical and mathematical operators; 3) flexible, as it can easily incorporate any vision or language module, only requiring the specification of the associated module be added to the API; 4) compositional, decomposing tasks into smaller sub-tasks performed step-by-step; 5) adaptable to advances in the field, as improvements in any of the used modules will result in a direct improvement in our approach's performance; 6) training-free, as it does not require to re-train (or finetune) a new model for every new task; and finally, 7) general, as it unifies all tasks into one system.

In summary, our contributions are:

# 2. Related Work

Modular Vision. Our work takes inspiration from Neural Module Networks [2, 27], who argue that complex vision tasks are fundamentally compositional and propose dividing them into atomic perceptual units. This visual reasoning procedure has been explored by a variety of works [29, 57]. Posterior efforts have focused on explicitly reasoning about the composition by separating the reasoning from the perception, with connections to neuro-symbolic methods [19, 27, 62]. These approaches are similar in spirit to ours, but require expensive supervision in the form of programs and end-to-end train the perception modules, which makes them not generalizable to different domains.

Due to the practical difficulty of using these methods, the field has primarily moved towards end-to-end all-in-one models [1, 22, 23, 30]. Such models currently obtain state-of-the-art results, and we compare to them in Section 4. Other recent works [63, 45, 55, 35, 37, 15] show that large pretrained models can be used together to great effect, but hand-specified the particular way models are combined.

Over the course of this project, a surge of interest in the area has resulted in a number of related manuscripts appearing on arXiv which use large language models (LLMs) for automatic module integration. In the natural language processing domain, they have been aimed at using external tools [46, 40], or for structured reasoning using Codex [34, 54, 14, 10]. Concurrent work [17] generates a list

![](dt=2026-06-05/ht=17/32bc341672e4ac4da15b8ceecd5603c4c78a785e62483431d1413d35053463b2.jpg)

of pseudocode instructions and interprets them as a 'visual program,' relying on in-context learning from provided examples. Unlike them, we directly generate unrestricted Python code, which is much more flexible and enables us to demo
nstrate more advanced emergent abilities, such as control flow and math. Crucially, using Python allows us to leverage the strong prior knowledge Codex learns by training at scale from the Internet. Additionally, we evaluate on many established benchmarks measuring visual understanding and achieve top-performing zero-shot results.

Interpretability. The area of interpretability for complex queries in vision is extensive. Many approaches provide explanations in the form of pixel importance, à la Grad-CAM [47, 65, 11, 41], some also providing textual explanations [41]. These are often post-hoc explanations rather than by construction, and do not give step-by-step reasoning including image crops and text. Hard attention in captioning [59] aims for a similar goal regarding intermediate image crops, similarly to our find module, but has proven difficult to incorporate into learning algorithms. See He et al. [18] for a complete overview.

Pretrained models. The perception and external knowledge modules used by ViperGPT are GLIP [31] for object detection, X-VLM [64] for text-image similarity (as it surpasses CLIP [43] at attribute detection [5]), MiDaS [44] for depth estimation, GPT-3 [6] for external knowledge, and BLIP-2 [30] for simple visual queries.

# 3. Method

We use notation following Johnson et al. [27]. Given a visual input $x$ and a textual query $q$ about its contents, we first synthesize a program $z = \pi(q)$ with a program generator $\pi$ given the query. We then apply the execution engine $r = \phi(x, z)$ to execute the program $z$ on the input $x$ and pro

![](dt=2026-06-05/ht=17/283dcfb4ad247fbd28d4c26a800da2dd0341187a2017f911ae9983788b82fa81.jpg)

duce a result $r$ . Our framework is flexible, supporting image or videos as inputs $x$ , questions or descriptions as queries $q$ , and any type (e.g., text or image crops) as outputs $r$ .

While prior work represents programs as graphs, like syntax trees [27] or dependency graphs [8], we represent the class of programs $z \in \mathcal{Z}$ directly through Python code, allowing our programs to capitalize on the expressivity and capabilities afforded by modern programming languages.

# 3.1. Program Generation

Johnson et al. [27] and other work in this direction [19, 62, 25] typically implement $\pi$ with a neural network that is trained with either supervised or reinforcement learning in order to estimate programs from queries. However, these approaches have largely been unable to scale to in-the-wild settings because either a) the supervision in the form of programs cannot be collected at scale or b) the optimization required for finding the computational graph is prohibitive.

In our approach, we instead capitalize on LLMs for code generation in order to instantiate the program generator $\pi$ that composes vision and language modules together. LLMs take as input a tokenized code sequence ("prompt") and autoregressively predict subsequent tokens. We use Codex [9], which has shown remarkable success on code generation tasks. Since we replace the optimization of $\pi$ with an LLM, our approach obviates the need for task-specific training for program generation. Using Codex as the program generator and generating code directly in Python allows us to draw on training at scale on the Internet, where Python code is abundant.

To leverage LLMs in this way, we need to define a prompt that will sample programs $z$ that compose and call

Table 1. RefCOCO Results. We report accuracy on the REC task and testA split. ZS=zero shot, Sup.=supervised.

![](dt=2026-06-05/ht=17/59e58a87f99be4270d028edf41c8a831b25115bfe97d298184d2bed6376e7765.jpg)

<table><tr><td></td><td></td><td colspan="2">IoU (%) ↑</td></tr><tr><td></td><td></td><td>RefCOCO</td><td>RefCOCO+</td></tr><tr><td rowspan="2">Sup.</td><td>MDETR [53]</td><td>90.4</td><td>85.5</td></tr><tr><td>OFA [53]</td><td>94.0</td><td>91.7</td></tr><tr><td rowspan="4">ZS</td><td>OWL-ViT [38]</td><td>30.3</td><td>29.4</td></tr><tr><td>GLIP [31]</td><td>55.0</td><td>52.2</td></tr><tr><td>ReCLIP [49]</td><td>58.6</td><td>60.5</td></tr><tr><td>ViperGPT (ours)</td><td>72.0</td><td>67.0</td></tr></table>

these modules as needed. Our prompt consists of an application programming interface (API), detailed in the following section, which we provide to the LLM as part of its input context. The final input to the LLM is a sequence of code text consisting of the API specification followed by the query for the sample under consideration. The expected output is a Python function definition as a string, which we then compile and execute.

# 3.2. Modules and Their API

Our prompt, included in the Appendix B, provides the API for different perceptual and knowledge modules, such as for object detection, depth estimation, or language model queries. From this prompt, we found that LLMs are able to induce correct programs $z$ from the query $q$ .

The API we provide defines two global classes ImagePatch and VideoSegment, which represent an image patch and a video segment respectively. Each module is implemented as a class method, which internally calls a pretrained model to compute the result. For example, the compute_depth method of ImagePatch returns an estimate of the median (relative) depth of the pixels in the image patch; we implement this with state-of-the-art large-scale models such as MiDaS [44]. We provide more details about the modules used in Section 4.

The API specifies the input and output types for each method it defines, as well as docstrings to explain the purpose of these functions in natural language. Like most APIs, it additionally provides examples that show how to use these classes and their functions, specified in the form of querycode pairs similarly to in-context learning [50, 6].

The input to Codex does not contain the full implementation of the API. Instead, it is given the specification for the API, including the function signatures and docstrings. Abstracting away the implementation details is beneficial for two reasons. First, LLM context windows are limited in size [6], making it infeasible to include the entire implementation. In addition, the abstraction makes code generation independent of changes made to the module implementation.

End-to-end perception modules are excellent when used in the right places, and ViperGPT strongly relies on them.

![](dt=2026-06-05/ht=17/18734ca7b8a793b6fd94b1218f3fc1163d04f16416335046ed533b7bfcc27708.jpg)

![](dt=2026-06-05/ht=17/bd5c76811ecf4a9f62ef4473a28983e7b7a84ab23c4f8cd41e9db76db8536b34.jpg)

![](dt=2026-06-05/ht=17/ada69be71e09a562cfdc40f09b41a24bfb1096390a0b660f7ea3d2b84c3947af.jpg)

![](dt=2026-06-05/ht=17/030251575c9ee0217171e45fcc41720b327de186ddac27eaa7b57570c25a57c2.jpg)

![](dt=2026-06-05/ht=17/2b5371ebbeb7fb8a5acfc3c2d00c20b76db8437104fc86ab8f6ec56584ff168b.jpg)

![](dt=2026-06-05/ht=17/45e3e16f321ff5ec02a11338ee00ad4ef2184b8ec2c9b4a2a8bc482a05ebec86.jpg)

Analogous to dual-system models [28] in cognitive science, we argue that generated programs (System 2 - analytic) should be utilized to break down tasks that require multiple steps of reasoning into simpler components, where end-to-end perception modules (System 1 - pattern recognition) are the most effective approach. By composing end-to-end modules into programs, ViperGPT brings the System 2 capabi
lity of sequential processing to deep learning [4].

# 3.3. Program Execution

At execution time, the generated program $z$ accepts an image or video as input and outputs a result $r$ corresponding to the query provided to the LLM. To execute this program, previous work (e.g., [27]) learns an execution engine $\phi$ as a neural module network, composing various modules implemented by neural networks. Their modules are responsible for not only perceptual functions such as find, but also logical ones such as compare. They learn all neural modules together simultaneously end-to-end, which fails to enable systematic generalization [3] and results in modules that are not faithful to their intended tasks [48], compromising the interpretability of the model.

We provide a simple, performant alternative by using the Python interpreter in conjunction with modules implemented by large pretrained models. The Python interpreter enables logical operations while the pretrained models enable perceptual ones. Our approach guarantees faithfulness by construction.

The program is run with the Python interpreter; as such, its execution is a simple Python call. This means it can leverage all built-in Python functions like sort; control flow tools like for or if/else; and modules such as datetime or math. Notably, this does not require a custom interpreter, unlike prior approaches [17, 46] Another advantage of a

Table 2. GQA Results. We report accuracy on the test-dev set.

![](dt=2026-06-05/ht=17/4db444b49035d8ddc8d9f10d13b2a07c831095abf6805c559da555e50e1263be.jpg)

<table><tr><td></td><td></td><td>Accuracy (%) ↑</td></tr><tr><td rowspan="4">Sup.</td><td>LGCN [20]</td><td>55.8</td></tr><tr><td>LXMERT [51]</td><td>60.0</td></tr><tr><td>NSM [24]</td><td>63.0</td></tr><tr><td>CRF [39]</td><td>72.1</td></tr><tr><td rowspan="2">ZS</td><td>BLIP-2 [30]</td><td>44.7</td></tr><tr><td>ViperGPT (ours)</td><td>48.1</td></tr></table>

fully Pythonic implementation is compatibility with a wide range of existing tools, such as PyTorch JIT [42].

In our implementation, each program in a generated batch is run simultaneously with multiprocessing. Our producer-consumer design [12] enables efficient GPU batching, reducing the memory and computation costs. Our code is made available at viper.cs.columbia.edu/.

# 4. Evaluation

ViperGPT is applicable to any tasks that query visual inputs with text. Unlike other work using large language models for vision tasks, the return values of our programs can be of arbitrary types, such as text, multiple choice selections, or image regions. We select four different evaluation settings to showcase the model's diverse capabilities in varied contexts without additional training. The tasks we consider are: 1) visual grounding, 2) compositional image question answering, 3) external knowledge-dependent image question answering, and 4) video causal and temporal reasoning.

We consider these tasks to roughly build on one another, with visual grounding being a prerequisite for compositional image question answering and so on. In the following sections, we explore the capabilities ViperGPT demonstrates in order to solve each task.

Query: The real live version of this toy does what in the winter?

![](dt=2026-06-05/ht=17/11983171b4d531403b3455b50dfbda3547caefdd0c013dcdc71e5b809bbec1a1.jpg)

Generated code

In:

Execution

$\triangleright$ toy $=$ {str} "bear"

$\triangleright$ guess $\equiv$ {str} "hibernate"

Result: "hibernate"

BLIP-2 result: "ski"

# 4.1. Visual Grounding

Visual grounding is the task of identifying the bounding box in an image that corresponds best to a given natural language query. Visual grounding tasks evaluate reasoning about spatial relationships and visual attributes. We consider this task first as it serves as the first bridge between text and vision: many tasks require locating complex queries past locating particular objects.

We provide ViperGPT with the API for the following modules (pretrained models in parentheses). find (GLIP [31]) takes as input an image and a short noun phrase (e.g. "car" or "golden retriever"), and returns a list of image patches containing the noun phrase. exists (GLIP [31]) takes as input an image and a short noun phrase and returns a boolean indicating whether an instance of that noun phrase is present in the image.

Similarly, verify_property (X-VLM [64]) takes as input an image, a noun phase representing an object, and an attribute representing a property of that object; it returns a boolean indicating whether the property is present in the image. best_image_MATCH (X-VLM [64]) takes as input a list of image patches and a short noun phrase, and returns the image patch that best matches the noun phrase. Symmetric to this operation, best_text_MATCH takes as input a list of noun phrases and one image, and returns the noun phrase that best matches the image.

(This module is not necessary for visual grounding, but rather for tasks with text outputs; we describe it here for simplicity.) They are implemented using an image-text similarity model as in CLIP [43]. Finally, compute_depth (MiDaS [44]) computes the median depth of the image patch. We also define the function distance, which computes the pixel-distance between two patches, using only built-in Python tools.

For evaluation, we use the RefCOCO and RefCOCO+ datasets. The former allows for spatial relations while the latter does not, thereby providing different insights into ViperGPT's capabilities. We compare ViperGPT against end-to-end methods, and outperform other zero-shot methods on both datasets (see Table 1). We show examples² in Figure 3. See Appendix A for more details about the experimental setup.

Table 3. OK-VQA Results.

![](dt=2026-06-05/ht=17/baad3a5bd54e837722e9ebe577a7c45b98941fd4a70325de76a2f829fc8e6310.jpg)

<table><tr><td></td><td></td><td>Accuracy (%) ↑</td></tr><tr><td rowspan="5">Sup.</td><td>TRiG [13]</td><td>50.5</td></tr><tr><td>KAT [16]</td><td>54.4</td></tr><tr><td>RA-VQA [32]</td><td>54.5</td></tr><tr><td>REVIVE [33]</td><td>58.0</td></tr><tr><td>PromptCap [21]</td><td>58.8</td></tr><tr><td rowspan="5">ZS</td><td>PNP-VQA [52]</td><td>35.9</td></tr><tr><td>PICa [60]</td><td>43.3</td></tr><tr><td>BLIP-2 [30]</td><td>45.9</td></tr><tr><td>Flamingo [1]</td><td>50.6</td></tr><tr><td>ViperGPT (ours)</td><td>51.9</td></tr></table>

# 4.2. Compositional Image Question Answering

We also evaluate ViperGPT on image question answering. We focus on compositional question answering, which requires decomposing complex questions into simpler tasks. We use the GQA dataset [26], which was created to measure performance on complex compositional questions. Consider Figure 4 for example questions as well as our provided reasoning. Even if a question can be answered end-to-end, it is both more interpretable and more human-aligned to provide intermediate reasoning rather than requiring the model to compress all steps into one forward pass; as our final result is constructed directly from the intermediate values, they provide a fully faithful interpretation of how the model came to its answer.

For GQA, we incorporate the module simple_query (BLIP-2 [31]), which handles basic queries that are not further decomposable, such as "What animal is this?" We also add the aforementioned best_text_MATCH. This leads us to the best accuracy on GQA among zero-shot models (Table 4).

# 4.3. External Knowledge-dependent Image Question Answering

Many questions about images can only be answered correctly by integrating outside knowledge about the world. By equipping ViperGPT with a module to query external knowledge bases in natural language, it can combine knowledge with visual reasoning to handle such questions. We
add a new module llm_query (GPT-3 [6]), which exploits text models as unstructured knowledge bases. We find that the combination of step-by-step reasoning from Codex along with external knowledge queried from GPT-3's text model achieves impressive performance in this setting.

We evaluate on the OK-VQA dataset [36], which is designed to evaluate models' ability to answer questions about images that require knowledge that cannot be found in the image. Items in this dataset often require more than one step of reasoning to produce a correct answer. For example, in Figure 5, one must first perceive from the image that

Examples in the paper have been cosmetically cleaned by removing comments and error handling, but the logic is unchanged.

# Query: What did the boy do after he dropped the sparkles on the floor?

# Generated code

# Execution

In:

![](dt=2026-06-05/ht=17/e92477eac89551d22f18e8b55225e6936a40d1bc8c3d5f0f3501caf664504982.jpg)

![](dt=2026-06-05/ht=17/3e8caa87bc870cfd0b4b7f83431e8c5953ac61ab7c6802f686d97090ec90708d.jpg)

![](dt=2026-06-05/ht=17/3e9eeafbf70678c9eacea9a271bf4f2a929199cf8591a7fd88729ea88d105406.jpg)

![](dt=2026-06-05/ht=17/5f767d5f2feddb8f84e3dd5068de7cd590556ea4051f352ab80ed33d1171c08f.jpg)

# Query: How does the black dog position himself at the end?

# Generated code

# Execution

In:

![](dt=2026-06-05/ht=17/2010ed47339fffd60e977c3607a95d6381a0602bb71b73b6a40d05d6f6ae100b.jpg)

![](dt=2026-06-05/ht=17/8107ebd14cc93220deb78fae871f926a24716e8e74ec313c53ceda0b58bb24d6.jpg)

![](dt=2026-06-05/ht=17/d87d83239d17b31f4f45f30320f0f360898251c4a6bff8a32b38bba162c2eaeb.jpg)

"This toy" is a "bear," then use external knowledge to answer what bears do in the winter. End-to-end models must directly produce an answer, and therefore may pick words that are more directly related to the image than the question intended. In this case, the best available end-to-end model guesses "ski," presumably as that is a common winter activity (though, not for bears). ViperGPT, on the other hand, can employ a form of chain-of-thought reasoning [56] to break down the question as previously described, first determining the type of toy using perception modules and then using the perceived information in conjunction with an external knowledge module to produce the correct response.

ViperGPT outperforms all zero-shot methods, and when compared to models using publicly available resources, it surpasses the best previous model by $6\%$ , a wide margin for this dataset (see Table 3).

# 4.4. Video Causal/Temporal Reasoning

We also evaluate how ViperGPT extends to videos and queries that require causal and temporal reasoning. To explore this, we use the NExT-QA dataset, designed to evaluate video models ability to perform this type of reasoning.

Table 4. NExT-QA Results. Our method gets overall state-of-the-art results (including supervised models) on the hard split. "T" and "C" stand for "temporal" and "causal" questions, respectively.

![](dt=2026-06-05/ht=17/c083bd9934484c6af227f6d0a7cc3ea4e53e38a91f17e9b171ec79997d6d7b70.jpg)

<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Accuracy (%) ↑</td></tr><tr><td>Hard Split - T</td><td>Hard Split - C</td><td>Full Set</td></tr><tr><td rowspan="3">Sup.</td><td>ATP [7]</td><td>45.3</td><td>43.3</td><td>54.3</td></tr><tr><td>VGT [58]</td><td>-</td><td>-</td><td>56.9</td></tr><tr><td>HiTeA [61]</td><td>48.6</td><td>47.8</td><td>63.1</td></tr><tr><td>NS</td><td>ViperGPT (ours)</td><td>49.8</td><td>56.4</td><td>60.0</td></tr></table>

We evaluate using the NExT-QA multiple choice version.

We provide an additional module select_answer (GPT-3 [6]), which, given textual information about a scene and a list of possible answers, returns the answer that best fits the information. Other than that, the only additional content given in the API is the definition of the class VideoSegment, that contains the video bytestream as well as the start and end timestamps of the video segment that it represents. It also defines an iterator over the frames, which returns an ImagePatch object representing every frame.

We find that despite only being provided with perception modules for images, ViperGPT displays emergent causal and

![](dt=2026-06-05/ht=17/e64c683a97e2321a4300bcea6d36082cfb242c157c31f809a6944543fe318870.jpg)

temporal reasoning when applied to videos provided as an ordered list of images. In particular, we observe it generates programs that apply perception to determine which frames are relevant for a given query, then reasons about the information extracted from these frames along with associated frame numbers to produce a final answer.

Despite seeing no video data whatsoever, ViperGPT achieves accuracy results on par with the best supervised model (see Table 4), and even surpassing it on the NeXTQA hard split [7], both for temporal and causal queries. Of course, the framework of ViperGPT also allows for incorporation of video models, which we expect would further improve the performance well beyond this threshold.

Computational ability presents even more of an obstacle for video understanding than for images. It is infeasible to fit every frame of a moderately-sized video into GPU memory on even the best hardware. ViperGPT may provide a way forward for video understanding that overcomes the limitations of systems that need to perform computation on a whole video simultaneously. See examples in Figure 6.

# 5. Exploring New Capabilities

In this section, we showcase various interesting capabilities enabled by use of ViperGPT.

# 5.1. Queries Beyond Benchmarks

We believe that the evident strength of this approach may not be adequately explored by existing benchmarks, which are designed for end-to-end models. In Figure 1, we show examples of interesting queries that are interesting in the real world but would not show up in existing benchmarks. We do not add any new API specifications other than the ones already used in the benchmarks. See the Appendix B for more details.

These examples show that the modules we included are general and cover a wide range of tasks. In settings where new capabilities are required, the framework is general and permits the addition of any modules, likeOCR, surface_normal_estimation, segmentation, etc.

# 5.2. Interventional Explainability

Our programmatic approach enables automatic diagnosis of which modules are responsible for prediction errors,

![](dt=2026-06-05/ht=17/2e9d19a8dec65aa6b3953170bf82bb99b9832e86f18a92680e0a1dbda80e562e.jpg)

potentially informing which types of models to improve and where to collect more data. Evaluating the intermediate output of each module is impractical due to the lack of ground truth labels, and naively comparing accuracy between programs that use a certain module and those that do not could be confounded e.g. by the difficulty of the problem. We can instead perform interventions to better understand a module's performance. For each module, we can define a default value that provides no information, and substitute the underlying model for this default output.

For instance, find could always return the full input image. We can then con
sider how much performance drops if evaluating the same code for the examples that use that module. If the intervention has a minimal impact on performance, the module is likely not useful.

We show an example of this analysis in Figure 7 for visual grounding on RefCOCO, where we observe a similar level of importance for perception modules and Python operations. Both are tightly integrated in our approach.

# 5.3. Conditioning on Additional Information

We found ViperGPT readily admits program generation based on additional knowledge. This context can be provided as a comment prior to the code generation. Such context can be critical to correctly responding to a wide range of queries. In Figure 8 we show one such example. The correct side of the road varies by country, so the initial query cannot be answered. Provided with the context of where the photo was taken, the model produces different logic for each case, adjusted based on the relevant prior knowledge.

# 6. Conclusions

We present ViperGPT, a framework for programmatic composition of specialized vision, language, math, and logic functions for complex visual queries. ViperGPT is capable of connecting individual advances in vision and language; it enables them to show capabilities beyond what any individual model can do on its own. As the models implementing these functions continue to improve, we expect ViperGPT's results will also continue to improve in tandem.

Acknowledgements: This research is based on work partially supported by the DARPA MCS program under Federal Agreement No. N660011924032 and the NSF CAREER Award #2046910. DS is supported by the Microsoft PhD Fellowship and SM is supported by the NSF GRFP.

# References

# A. Pretrained Models

We specify details about all the pretrained models used, as well as the code-generation large language model:

See the code for more detailed implementation details.

# B. API

<sup>3</sup>https://github.com/microsoft/GLIP

4https://pytorch.org/hub/intelisl_midas_v2/

<sup>5</sup>https://github.com/salesforce/LAVIS/tree/main/projects/blip2

$^{6}$ https://huggingface.co/Salesforce/blip2-flan-t5-xxl

<sup>7</sup>https://github.com/zengyan-97/X-VLM

8https://openai.com/blog/openai-api

If no coordinates are provided, the image is left unmodified, and the coordinates are set to the dimensions of the image.

Parameters

- - - - - -

image : array_like

An array-like of the original image.

left : int

An int describing the position of the left border of the crop's bounding box in the original image.

lower : int

An int describing the position of the bottom border of the crop's bounding box in the original image.

right : int

An int describing the position of the right border of the crop's bounding box in the original image.

upper : int

An int describing the position of the top border of the crop's bounding box in the original image.

1 1

if left is None and right is None and upper is None and lower is None:

self.cropped_image = image

self.left = 0

self(lower = 0

self.right = image.shape[2] # width

self-upper = image.shape[1] # height

else:

self.cropped_image = image[:, lower:upper, left:right]

self.left = left

self-upper = upper

self.right = right

self(lower = lower

self.width = self cropped_image.shape[2]

self.height = self.cropped_image.shape[1]

self-horizontal_center = (self.left + self.right) / 2

self vertical center = (self.low + self.upper) / 2

def find(self, object_name: str) -> List[ImagePatch]:

""Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.

Otherwise, returns an empty list.

Parameters

#

object_name : str

the name of the object to be found

Returns

- - - - - -

List[ImagePatch]

a list of ImagePatch objects matching object_name contained in the crop

Examples

#

>>> # return the children

>>> def execute_command(image) -> List[ImagePatch]:

>>> imagePatch = ImagePatch(image)

>>> children = imagepatch.find("child")

>>> return children

1111

def exists(self, object_name: str) -> bool:

""Returns True if the object specified by object_name is found in the image, and False otherwise.

Parameters

- - - - -

object_name : str

A string describing the name of the object to be found in the image.

Examples

- - - - - -

>>> # Are there both cakes and gummy bears in the photo?

>>> def execute_command(image) ->str:

>>> imagePatch = ImagePatch(image)

>>> is_cake = image.patch.exists("cake")

>>> is_gummy_bear = image_batch EXISTS("gummy_bear")

>>> return bool_to yesno(is_cake and is_gummy_bear)

1111

return len(self.find(object_name)) > 0

def verify_property(self, object_name: str, property: str) -> bool:

""Returns True if the object possesses the property, and False otherwise.

Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead checking whether the object

possesses the property.

Parameters

38   
40   
42   
43   
44   
45   
46   
47   
48   
49   
50   
51   
52   
53   
54   
55   
56   
57   
58   
59   
60   
61   
62   
63   
64   
65   
66   
67   
68   
69   
70   
71   
72   
73   
74   
75   
76   
77   
78   
79   
80   
81   
82   
83   
84   
85   
86   
87   
88   
89   
90   
91   
92   
93   
94   
95   
96   
97   
98   
99   
100   
101   
102   
103   
104   
105   
106   
107   
108   
109   
110   
111   
112   
113   
114   
115

116   
117   
118   
120   
121   
122   
123   
124   
125   
126   
127   
128   
129   
130   
131   
132   
133   
134   
135   
136   
137   
138   
139   
140   
141   
142   
143   
144   
145   
146   
147   
148   
149   
150   
151   
152   
153   
154   
155   
156   
157   
158   
159   
160   
161   
162   
163   
164   
165   
166   
167   
168   
169   
170   
171   
172   
173   
174   
175   
176   
177   
178   
179   
180   
181   
182   
183   
184   
185   
186   
187   
188   
189   
190   
191   
192   
193   
194

# Examples

#

>>> # the person furthest away

>>> def execute_command(image) ->ImagePatch:

>>> imagePatch = ImagePatch(image)

>>> personpatches = imagepatch.find("person")

>>> personpatches.sort(key=lambda person: person.compute_depth())

>>> return person_patches[-1]

11 11

depth_map = compute_depth(self.cropped_image)

return depth_map.median()

def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:

""Returns a new ImagePatch cropped from the current ImagePatch.

Parameters

#

left : int

The leftmost pixel of the cropped image.

lower : int

The lowest pixel of the cropped image.

right : int

The rightmost pixel of the cropped image.

upper : int

The uppermost pixel of the cropped image.

- - - - - -

1111

return ImagePatch(self.cropped_image, left, lower, right, upper)

def overlaps_with(self, left, lower, right, upper):

""Returns True if a crop with the given coordinates overlaps with this one,

else False

Parameters

#

left : int

the left border of the crop to be checked

lower : int

the lower border of the crop to be checked

right : int

the right border of the crop to be checked

upper : int

the upper border of the crop to be checked

# Returns

- - - - - -

bool

True if a crop with the given coordinates overlaps with this one, else False.

# Examples

#

>>> # black cup on top of the table

>>> def execute_command(image) -> ImagePatch:

>>> imagePatch = ImagePatch(image)

>>> tablepatches = imagePatch.find("table")

>>> if len(table_patches) == 0

>>> table_patches = [image_batch] # If no table found, assume the whole image is a table

>>> tablePatch = table_patches[0]

>>> cup_patches = imagePatch.find("black cup")

>>> for cup in cup_patches:

>>> if cup(vertical_center > table PATCH(vertical_center

>>> return cup

>>> return cup_patches[0] # If no cup found on top of the table, return the first cup found

return self.left <= right and self.right >= left and self.lowerr = upper and self.high >= lower

def best_image_MATCH(list=Patches: List[ImagePatch], content: List[s
tr], return_index=False) -> Union[ImagePatch, int]:

""Returns the patch most likely to contain the content.

# Parameters

- - - - - - - - - -

listpatches : List[ImagePatch]

content : List[str]

the object of interest

return_index : bool

if True, returns the index of the patch most likely to contain the object

# Returns

= = = = =

int

Patch most likely to contain the object

195   
196   
197   
198   
199   
200   
201   
202   
203   
204   
205   
206   
207   
208   
209   
210   
211   
212   
213   
214   
215   
216   
217   
218   
219   
220   
221   
222   
223   
224   
225   
226   
227   
228   
229   
230   
231   
232   
233   
234   
235   
236   
237   
238   
239   
240   
241   
242   
243   
244   
245   
246   
247   
248   
249   
250   
251   
252   
253   
254   
255   
256   
257   
258   
259   
260   
261   
262   
263   
264   
265   
266   
267   
268   
269   
270   
271   
272

# Examples

- - - - -

>>> # Return the man with the hat

>>> def execute_command(image):

>>> image_batch = ImagePatch(image)

>>> man_patches = image_batch.find("man")

>>> if len(len_patches) == 0:

>>> return image_batch

>>> hat_man = best_image_MATCH(list=Patches=man_patches, content=['hat'])

>>> return hat_man

>>> # Return the woman with the pink scarf and blue pants

>>> def execute_command(image):

>>> image_batch = ImagePatch(image)

>>> woman_patches = image_patch.find("woman")

>>> if len(woman_patches) == 0:

>>> return image_batch

>>> woman-most = best_image_MATCH(list_patches=woman_patches, content=['pink scarf", "blue pants'])

>>> return woman最多的

#

return best_image_MATCH(list_patches, content, return_index)

# distance(batch_a:ImagePatch，patch_b：ImagePatch）->float:

# bool_to yesno(bool_answer: bool) -> str:

return "yes" if bool_answer else "no"

# llm_query(question: str) -> str:

'Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

# Parameters

- - - - -

question: str

the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.

return lmm_query(question)

# ss VideoSegment:

"A Python class containing a set of frames represented as ImagePatch objects, as well as relevant information.

Attributes

- - - - -

video : torch.Tensor

A tensor of the original video.

start : int

An int describing the starting frame in this video segment with respect to the original video.

end : int

An int describing the ending frame in this video segment with respect to the original video.

num_frames->int

An int containing the number of frames in the video segment.

# Methods

......

frame.iteratoror->Iterator[ImagePatch]

trim(start, end) -> VideoSegment

Returns a new VideoSegment containing a trimmed version of the original video at the [start, end] segment.

select_answer(info, question, options) -> str

Returns the answer to the question given the options and additional information.

1111

# def __init__(self, video: torch.Tensor, start: int = None, end: int = None, parent_start=0, queues=None):

""Initializes a VideoSeqment object by trimming the video at the given [start, end] times and stores the

. . . start and end times as attributes . If no times are provided , the video is left unmodified , and the times are

set to the beginning and end of the video

# Parameters

274   
275   
276   
278   
279   
280   
281   
282   
283   
284   
285   
286   
287   
288   
289   
290   
291   
292   
293   
294   
295   
296   
297   
298   
299   
300   
301   
302   
303   
304   
305   
306   
307   
308   
309   
310   
311   
312   
313   
314   
315   
316   
317   
318   
319   
320   
321   
322   
323   
324   
325   
326   
327   
328   
329   
330   
331   
332   
333   
334   
335   
336   
337   
338   
339   
340   
341   
342   
343   
344   
345   
346   
347   
348   
349   
350   
351

Not all methods are used in all the benchmarks. Next we describe in more detail what content is used for the API specifications for every benchmark.

- RefCOCO and RefCOCO+. We use all the methods from the ImagePatch class except for best_text_MATCH and simple_query. We also use the best_text_MATCH and distance functions. Additionally we add ImagePatch usage examples in the API definition that are representative of the RefCOCO dataset, and look like the following:

353   
354   
355   
356   
357   
358   
359   
360   
361   
362   
363   
364   
365   
366   
367   
368   
369   
370   
371   
372   
373   
374   
375   
376   
377   
378   
379   
380   
381   
382   
383   
384   
385   
386   
387   
388   
389   
390   
391   
392   
393   
394   
395   
396   
397   
398   
399   
400   
401   
402   
403   
404   
405   
406   
407   
408

- GQA. The GQA API contains all the contents in the API from Listing 1 up until the lvm-query function, which is not used. The ImagePatch usage examples look like the following:

- OK-VQA. The API only uses the simple_query method from ImagePatch. It additionally uses the llm_query function. The ImagePatch usage examples look like the following:

- NeXT-QA. The VideoSegment class is added to the API definition, and the available ImagePatch methods are find, exists, best_text_match and simple_query. The function best_image_MATCH is also used. The ImagePatch usage examples look like:

- Beyond benchmarks. For the examples in Figure 1 we use the same API as the one used for the benchmarks, and the usage examples are taken from the benchmark APIs, combining them to have more generality. We do not add any other example, ViperGPT generalizes to the complex cases shown in Figure 1 just based on the provided API.

Note that in some of the examples we added comments, as well as error handling. The generated code also contains similar lines. We removed those for clarity in the figures shown in the main paper.