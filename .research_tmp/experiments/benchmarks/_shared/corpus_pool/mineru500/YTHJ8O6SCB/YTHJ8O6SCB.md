# SpatialPIN: Enhancing Spatial Reasoning Capabilities of Vision-Language Models through Prompting and Interacting 3D Priors

Chenyang Ma Kai Lu Ta-Ying Cheng Niki Trigoni Andrew Markham

University of Oxford

chenyang.ma@cs.ox.ac.uk

# Abstract

Current state-of-the-art spatial reasoning-enhanced VLMs are trained to excel at spatial visual question answering (VQA). However, we believe that higher-level 3D-aware tasks, such as articulating dynamic scene changes and motion planning, require a fundamental and explicit 3D understanding beyond current spatial VQA datasets. In this work, we present SpatialPIN, a framework designed to enhance the spatial reasoning capabilities of VLMs through prompting and interacting with priors from multiple 3D foundation models in a zero-shot, training-free manner. Extensive experiments demonstrate that our spatial reasoning-imbued VLM performs well on various forms of spatial VQA and can extend to help in various downstream robotics tasks such as pick and stack and trajectory planning.

# 1 Introduction

Equipping vision-language models (VLMs) the capacities of spatial reasoning unlocks exciting applications, such as general-purpose reward annotation $[52]$ , robotic data generation $[61]$ , and grounding 3D object affordances $[26, 38]$ . However, the spatial reasoning capabilities of VLMs on fine-grained spatial understanding tasks are somewhat limited. Current state-of-the-art (SOTA) spatial reasoning-enhanced VLM $[12]$ is mostly tested on spatial visual question answering (VQA), such as determining objects' relative positions and orientations; experiments on higher-level tasks, such as scene comparisons and trajectory planning, which require more nuanced comprehension, are underexplored.

Many works enhance the spatial reasoning capabilities of VLMs by training/fine-tuning them on standard spatial VQA datasets $[12]$ . As a result, VLMs primarily learn surface-level associations between image-text-data triplets. Given the scarcity and difficulty of obtaining spatially rich embodied data or high-quality human annotations for 3D-aware queries, we hypothesize that these VLMs may not generalize to questions outside their dataset distribution or adapt to more challenging tasks that require an advanced level of spatial understanding.

Recent studies $[73, 7, 65, 69]$ in image space understanding show that VLMs, equipped with internet-scale language knowledge, and multimodal foundation models capture complementary knowledge that can be combined to conduct new tasks spanning both modalities without additional training. Given the recent advancements in 3D foundation models $[4, 41, 29]$ , this work explores whether there exists an alternative approach to enhance VLMs with higher-level spatial-awareness by incorporating 3D priors from these models.

To this end, we propose SpatialPIN, a framework that utilizes progressive prompting and interactions between VLMs and 2D/3D foundation models as “free lunch” to enhance spatial reasoning capabilities in a zero-shot, training-free manner. By using these foundation models to decompose, comprehend, and reconstruct an explicit 3D representation, SpatialPIN grasps the core understanding of the 3D

space presented by the 2D image. This allows generalizations to various 3D-aware tasks, from VQAs to 3D trajectory planning.

We provide an extensive empirical study combining multiple off-the-shelf and handcrafted datasets, ranging from fundamental spatial questions regarding relative positions and orientations to providing fine-grained 3D information on objects' locations, sizes, inclinations, and dynamic changes, and plan for robotics tasks with full 3D trajectories. Results show that this straightforward approach significantly outperforms SOTA VLMs trained from extensive spatial VQAs (see SpatialPIN examples in Figure 1), consolidating our belief that a truly 3D-aware VLM can actually be imbued by simply injecting explicit, fundamental knowledge of the 3D scene. With the entire framework being fully modularized, each component can be easily replaced with the latest improvements within its specific domain.

![](images/aa708ef68a771531c3bd20d65283f7a9339c6856ef0f20104a158633ab22d148.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["User"] --> B["You are a robotic grasper. The task is to grasp the carrot. Determine your goal location and closing distance in centimeters to ensure a firm grip on the carrot."]
    B --> C["Interactive Prompting"]
    C --> D["SpatialPIN"]
    D --> E["To ensure a firm grip, the grasper should target the center of the carrot at coordinates (0.7399, 0.5185, 0.1503). This midpoint takes into account the carrot's width of 5.44 cm. The grasper should close to approximately 5.3 cm, just less than the carrot's width, for an optimal grasp."]
    E --> F["The task is to pick up the plastic bottle and place it inside the trash can. Plan for it."]
    F --> G["Interactive Prompting"]
    G --> H["First, align the plastic bottle with the trash can by rotating the bottle approximately 23.1 degrees around its local z-axis. This adjustment aligns the bottle's x-axis (0.9254, -0.3781, 0.0246) with the trash can's x-axis (0.9982, 0.0398, -0.0447). Next, move the bottle by [0, 0, 5"] cm relative to the trash can's 3D center to place it securely inside the trash can. Using RRT, the planned 3D trajectory is ...]
    H --> I["Next"]
```
</details>

Figure 1: We present SpatialPIN, a framework to enhance the spatial reasoning capabilities of VLMs through prompting and interacting with 3D priors in a zero-shot, training-free manner.

In summary, our main contributions are threefold:

- We investigate the problem of equipping VLMs with 3D reasoning capabilities without fine-tuning on large spatial VQA datasets.   
- We propose SpatialPIN, a modular plug-and-play framework that progressively enhances VLM's 3D reasoning capabilities by prompting and interacting with 3D foundational models.   
- We show that SpatialPIN unlocks 3D-aware applications including spatial VQA and both classic and novel robotics tasks, supported by extensive experiments.

# 2 Related Work

VLM Grounding With the recent birth of powerful LLMs and VLMs $[8, 40, 2]$ , the task of VLM grounding, or combining generative language models with real-world data to adapt to specific cases, has gained significant popularity. Several recent works focused on fine-tuning these LLMs for a wide range of downstream applications, such as interactive decision making $[34]$ , multi-task agents $[62]$ , or even tasks in interactive environments $[67, 11]$ . A close work to ours is Socratic Model $[73]$ , a framework of combining multiple foundation models to unleash LLMs in downstream tasks. However, this work still focuses on tasks in 2D pixel space understanding of images. There remains many challenges in the 3D world to combine information for full scene understanding, which we hope to tackle in our paper.

VLM Spatial Understanding Many VLMs encompass the ability of image-space reasoning and understanding $[13, 33, 40]$ . There are even efforts in incorporating these understandings into image space manipulations and editing $[7, 65]$ . However, the current ability of VLMs to fully understand a 3D scene and the potential interactions within this scene is still rather limited. Several works build from this foundation and establish datasets to help with spatial reasoning/understanding $[30, 39, 45]$ . Recently, SpatialVLM $[12]$ proposed fine-tuning a VLM on 3D-VQA datasets to enhance the precision of VLMs on 3D understanding tasks. Nevertheless, using a 3D-VQA dataset only provides a partial picture to the complete 3D understanding of an image, and could lead to suboptimal performances under out-of-distribution tasks. In this work, we hope to introduce holistic 3D information from multiple 3D foundation models via prompting and interactions as a way to enhance VLMs with a comprehensive 3D understanding given RGB inputs.

# 3 Method

Given an RGB image $I \in R^{H \times W \times 3}$ of a scene with K unknown objects and a spatial task Q, our goal is to inspire VLMs with spatial reasoning capabilities and solve Q with fine-grained 3D understanding.

![](images/70e9f421c0f6af8067deff4485cd87bf57aed6d30ae93ce8b9b02ac317202ca5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input RGB Image & Spatial Tasks"] --> B["Language-Guided Segmentation"]
    C["Q: What is the distance between the laptop and the camera?"]
    D["Q: How do I stack the can on the bowl?"] --> E["Understanding Objects by Constraining"]
    B --> F["Camera, Depth & Scene Size Estimation"]
    E --> G["Coarse 3D Understanding"]
    F --> H["Zero-Shot 3D Scene Reconstruction"]
    G --> I["Fine-Grained 3D Understanding"]
    H --> J["Solving with Spatial Reasoning"]
    I --> J
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style F fill:#cfc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I fill:#fcc,stroke:#333
```
</details>

Figure 2: SpatialPIN. Our plug-and-play framework is fully modularized and designed for zero-shot deployment. Each module can be easily replaced with the latest updates. Exact prompts for VLMs are in Appendix.

To prevent the models from overfitting to the standard problems from spatial VQA datasets $[12]$ , we hope to derive a method that utilizes fundamental 3D foundation models to provide explicit scene understandings, then leverage the generalization capabilities of VLMs to tackle unforeseen tasks—all within a zero-shot, training-free manner.

Our modular pipeline, SpatialPIN, enhances VLMs' spatial understanding of an image through progressive interactions with the scene decomposition, comprehension, and reconstruction processes with prompting. For image scene understanding (Sec. 3.1), we use VLM to describe objects by appearance and 2D location, complemented by language-guided segmentation and repainting models to obtain occlusion-free object masks. Elevating 2D understanding to coarse 3D (Sec. 3.2), we use metric depth estimation and perspective fields to estimate the 3D scene size and conduct perspective canonicalization with VLM. For fine-grained 3D understanding (Sec. 3.3), we partially reconstruct the 3D scene, with the full 3D representation of foreground objects and the background as a plane. With the reconstructed 3D scene, we summarize spatial information and prompt it to the VLM for various downstream tasks.

# 3.1 2D Image Scene Understanding

Prompting: Objects Understanding by Constraining We start with querying VLM to identify and understand objects given I. We explicitly ask VLM to describe the objects by precise color, texture, and 2D spatial locations. This step is vital for two reasons: 1) enhance VLM's understanding of the objects, 2) differentiate between items of similar or identical categories and appearances.

As a concrete example, given the left image of Fig. 2, VLM outputs: “

object 0: laptop of color rose gold, texture metallic at location left-center.

object 1: camera of color black, texture smooth at location center-right. ...”

2D Representations Refinement The concise descriptions of identified objects are used as input text prompts for a language-guided segmentation model, enabling the acquisition of K segmentation masks $\{M_{k}^{occ}\}_{k=1}^{K}$ , with each mask corresponding to a unique object.

However, an object $i \in [1, K]$ may be occluded by other object(s), leading to an incomplete mask $M_i^{occ}$ , which may be burdensome when we elevate the image to a 3D representation in the later stage. To resolve this, we create an inpainting mask, $M_i^{inp}$ , for each object, in which all objects except the one itself are removed and replaced with white pixels. The inpainted masks are again fed to the language-guided segmentation model along with input text prompts such that occlusion-free object masks, $\{M_k^{of}\}_{k=1}^K$ , are obtained. This two-step segmentation process for object $i$ is formulated as:

$$
M _ {i} ^ {\text {   occ   }} = \text { seg } (I, \tau_ {i}), \quad M _ {i} ^ {\text {   of   }} = \text { seg } (\text { inpaint } (M _ {i} ^ {\text {   inp   }}), \tau_ {i}), \tag {1}
$$

where seg denotes the language-guided segmentation and $\tau_{i}$ denotes the description of object i. In practice, to cleanly remove objects without residual fragments for inpainting, we apply dilation to and expand the white areas. Inpainted background $I_{bg}$ is acquired by removing and replacing all objects with dilated white pixels.

# 3.2 Coarse 3D Scene Understanding

Scene Size Estimation Using the estimated metric depth [29] and estimated camera intrinsic matrix by finding field of view (FOV) through perspective fields [29], we backproject to determine the dimensions of the 3D spatial scene.

Prompting: Perspective Canonicalization 3D information without any knowledge regarding the camera perspectives lead to ambiguities [12]. Consider a question “What is the orientation of the bowl relative to the laptop?” with the input scene in Fig. 2, but taken from a top-down perspective. VLMs may output “downward to the left”, but the correct answer should be "front-left" because humans perceive orientation from a horizontal angle. To address this, we provide the VLM with I, estimated scene size, and maximum and minimum dimensions, allowing it to reason about the camera shot angle (horizontal/top-down/bottom-up). Scene size information helps differentiate shot angles by providing clues about spatial layout and object proportions. For instance, if the depth variation is small, the VLM can infer a top-down or bottom-up angle along with visual cues.

As a concrete example, given the left image of Fig. 2, VLM outputs:

“Visual cues reasoning: Objects are viewed from the side, indicating the camera is positioned horizontally with a slight elevation.

Spatial data reasoning: The depth varies significantly from 57.50 cm to 115.00 cm, indicating the camera captures the scene across different distances, supporting a horizontal perspective.

Conclusion: horizontal."

# 3.3 Fine-Grained 3D Scene Understanding

We partially reconstruct the 3D scene with full representation of foreground objects while simplifying the inpainted background as a plane, as shown in Fig. 3(a). We summarize spatial information from the reconstructed scene and prompt it to the VLM. Please see our Appendix for implementation details about reconstruction.

![](images/141fc62b06ead60e4527bdc6088c22cf969343ab48ba946720c632be283efe39.jpg)  
(a) Side View

![](images/6d18e923da028c9d6e4272728259b9028d57f66ca9c5a633e8ade5d9c5b844e6.jpg)  
(b) Active Camera View

![](images/d4308040a3719fba00c6f1e06ad0dce5e0d8e125e65f8a61ecdfa46a84ee5479.jpg)  
(c) Original Input Image   
Figure 3: Our method of partial 3D scene reconstruction (a). The reconstructed scene (b) and the input image (c) show high alignment.

Scene Initialization Given the occlusion-free

object masks, $\{M_{k}^{of}\}_{k=1}^{K}$ , we use single-view 3D reconstruction model [41] to acquire object 3D models, $\{O_{k}\}_{k=1}^{K}$ , with canonical poses determined during reconstruction. Pinhole camera is set at the origin, looking at positive depth-axis. With the estimated background plane size (Sec. 3.2), we move the background plane, $O_{bg}$ (visually identical to $I_{bg}$ ), along the depth-axis to fit precisely within the camera.

Scene Reconstruction To resolve the imprecision of backprojection, our goal is to position object 3D models into the reconstructed 3D scene without visual discrepancies and ensure accurate depth. Instead of using naive backprojection, for an object $i \in [1, K]$ , we perform raycasting from object 3D center $t_{i}^{c}$ on the camera plane to object 3D center $t_{i}^{bg}$ on the background plane with metric depth $d_{i}$ . The 3D coordinate $t_{i}$ of object i is:

$$
d _ {i} = \left| I _ {d e p} (\text { center } (M _ {i} ^ {o f}) \right|, \quad t _ {i} = t _ {c} + \frac {d _ {i}}{\left| t _ {i} ^ {b g} - t _ {i} ^ {c} \right|} \times (t _ {i} ^ {b g} - t _ {c}). \tag {2}
$$

The rotation $R_{i}$ of the 6D pose of object $i$ , $P_{i} = [R_{i} \mid t_{i}]$ is explained previously. After integrating all 3D object models into the 3D scene, we refine each object's scale to accurately reflect depth variations by rendering binary masks and evaluate the length of their contour lines relative to their occlusion-free masks, through the lens of the pinhole camera, $t_{c}$ .

We determine the principal axes (x-axis, y-axis, and z-axis) of each object using the minimal oriented bounding box (OBB), which is essential for unlock novel applications.

Prompting: Objects and Spatial Context Understanding The reconstructed 3D scene from I with accurate object poses and scales is denoted as $V_{0}$ . As the final step of progressive prompting, we feed VLM the fine-grained 3D information derived from $V_{0}$ , grounding on the canonicalized perspective (Sec. 3.2). For example, with the input image in Fig. 2 and a horizontal camera shot angle, depth corresponds to the positive y-axis (similarly, in a top-down/bottom-up view, depth is

![](images/4400af2c10430adcd09d1178ce2d3ec6e3900807a2f7d643682ddb4bd4c19f34.jpg)

<details>
<summary>text_image</summary>

User
Is the laptop further to camera compared to the mug?
Measure the height difference between two white bowls.
Along which axis (W, D, H) is the box more tilted?
What is the relative angle between the box and the bottle?
Which objects rotate across the images?
How far does the milk box move?
SpatialPIN (Ours)
The laptop is 75.0 cm from the camera, while the mug is 54.7 cm from the camera, making the laptop further.
The height difference between bowl 1 (12.0 cm) and bowl 2 (9.4 cm) is approximately 2.6 cm.
The tilt along the z-axis (Height) is the largest (about 0.798), indicating the box is more tilted along the Height (H) axis.
Using the dot product formula, the relative angle between the box and the bottle is approximately 47.41 degrees.
By object principle x-axes, both the knife and the milk rotate around their vertical axes, presenting a different face.
By observing the change in the 3D center, the milk box moves a distance of approximately 22.26 cm.
SpatialVLM
Yes, the laptop is further away from the camera compared to the mug.
The two white bowls on the table have different heights, with one being taller than the other.
The box is more tilted along the horizontal axis, as it is leaning on the table and appears to be at a slight angle.
The box is positioned at a diagonal angle to the bottle on the table.
The knife and the cutting board rotate across the image, while the box of apples remains stationary.
The milk box moves from the counter to the stove, indicating that it is being used for cooking or heating.
</details>

Figure 4: Qualitative examples of spatial VQA. SpatialPIN outputs answers with fine-grained 3D reasoning. Zoom in for better view.

the negative/positive z-axis) in a right-handed coordinate system. The width and height axes can be determined accordingly, aligning each axis's orientation with human perception.

We feed VLM a paragraph describing the objects' poses, sizes, and principal axes in physical units, alongside their spatial relationships. To augment VLM's understanding, we also feed $V_0$ with visualized object axes (see Fig. 2B). Visualizing 3D spatial information is pivotal in improving VLMs' understanding of 3D spatial contexts derived from 2D images, validated by 3DAxiesPrompts [37]. Yet, we want to emphasize that we do not feed hardcoded information, such as objects' relative distances and inclinations, to VLMs. Instead, we aim for the summarized 3D information to enhance VLMs' general spatial understanding.

As a concrete example for the left image on Fig. 2:

“Obj 1 spatial context: 3D center: [7.0, 100.0, 9.0] cm; X-axis (right): [0.9529, -0.2456, 0.1779];

Y-axis (back): [-0.3528, 0.8746, 0.3327]; Z-axis (up): [-0.1761, -0.3285, 0.9279]

Obj 1 size: 13.54 cm x 9.37 cm x 9.50 cm (WxDxH)

Obj 1 closest per direction: left: Obj 0; right: Obj 2 ...”

# 3.4 Combining External Tools for Downstream Tasks

By partially reconstructing the 3D scene with visual alignments, our framework enables VLMs to use tools like rapidly-exploring random tree star (RRT\*) [31] to generate accurate, collision-free paths based on task specifications (more details in Appendix). This capability unlocks novel and interesting applications when combined with task-specific prompting techniques, shown in Experiments (Sec.4).

# 4 Experiments

We conduct experiments to answer the following questions: 1) Does our framework enhance the general spatial reasoning capabilities of VLMs, and how well does it perform? 2) What novel applications does our framework unlock for VLMs, and how well do we perform in these applications? 3) How effective is each module in our framework?

Since we evaluate our approach on a wide range of tasks to test VLMs' higher-level spatial awareness, some tasks are novel and lack existing/open source datasets. Therefore, for all our experiments, we use a combination of 4 existing datasets and 2 hand-crafted datasets.

Implementations The language-guided segmentation model is Language Segment-Anything $[44]$ and the repainting model is LaMa $[55]$ . We use One-2-3-45++ $[41]$ for single-view 3D reconstruction, perspective fields $[29]$ for camera intrinsic estimation, and ZoeDepth $[4]$ for depth estimation. For partial 3D scene reconstruction, we use Blender $[17]$ as the 3D software. All inference is run on 1 NVIDIA A10 GPU with 24GB RAM.

# 4.1 Spatial Visual Question Answering

We experiment on the basic form of spatial VQA introduced by SpatialVLM (IaOR-VQA), and two new forms introduced by us (IaAD- & IrSD-VQA). For IaOR-VQA, please check SpatialVLM [12] for details. For IaAD- & IrSD-VQA, please see our Appendix.

Intra-Image Object Relations VQA (IaOR-VQA) As the basic form of spatial VQA, it involves spatial reasoning about object relative orientations and sizes. This is divided into qualitative (e.g., “is [A] in front of [B]”, “is [A] smaller than [B]”) and quantitative (e.g., “how far apart are [A] and [B]”, “measure the width of [A]”) questions.

We follow the evaluation method of SpatialVLM $[12]$ . Since SpatialVLM did not release their evaluation dataset, we reproduce one using RGBD images from NOCS $[57]$ (object dataset), RT-1 $[6]$ , and BridgeData V2 $[56]$ (robotics manipulation datasets). We sample 13, 20, and 20 distinct scenes from each. We generate QA pairs using the SpatialVLM data generation pipeline $[51]$ , followed by manual refinement. We check correctness for qualitative questions and calculate distances for quantitative questions. We annotate 300 qualitative and 200 quantitative spatial VQA pairs (SpatialVLM has 331 and 215 for each).

Intra-Image Angular Discrepancies VQA (IaAD-VQA) We propose a new form of Spatial VQA that needs spatial reasoning about objects' inclinations. It includes qualitative (e.g., "is [A] tilted", "is [A] more tilted than [B]") and quantitative questions (e.g., "how many degrees is [A] tilted vertically", "measure the angle between [A] and [B]").

Since this form of Spatial VQA involves out-of-plane rotations, YCBInEOAT [64] (object tracking dataset) is a suitable choice. We sample 30 scenes from it and annotate 50 questions each for qualitative and quantitative spatial VQA pairs.

Inter-Image Spatial Dynamics VQA (IrSD-VQA) We further propose a more challenging form of Spatial VQA. Given two images with multiple objects, the objects in the second image may move, rotate, incline, or the image may have a change in camera angle. The VLM needs to reason about these changes. Example qualitative questions include “does [A] move, rotate, or incline”, “does [A] incline along the y-axis” while quantitative questions include “how far does [A] move”, “how many degrees does [A] rotate horizontally”.

As it is difficult to find a dataset that meets these requirements, we craft our own. We capture 20 image pairs using an iPhone 12 Pro Max, with each image containing 1 - 5 objects, and annotate 50 questions each for qualitative and quantitative spatial VQA pairs.

Results The results in Tables 1 and 2 on qualitative and quantitative IaOR-VQA demonstrate that providing various VLMs fine-grained 3D information enhances their spatial reasoning capacities by a large margin. Surprisingly, VLMs with math and geometry reasoning capacities (e.g., GPT-4V, GPT-4o) show substantial improvements with this information.

Table 1: Qualitative IaOR-VQA. We exclude comparisons to PaLI [14], PaLM-E [20], and PaLM 2-E [3] as they are not open source, and include experiments with GPT-4o [1] in addition to GPT-4V [47], LLaVA-1.5 [40], and InstructBLIP [18]. We use the HF version of SpatialVLM [51]. 

<table><tr><td rowspan="2"></td><td colspan="2">GPT-4V</td><td colspan="2">GPT-4o</td><td colspan="2">LLaVA-1.5</td><td colspan="2">InstructBLIP</td><td rowspan="2">SpatialVLM</td></tr><tr><td>w/o ours</td><td>w ours</td><td>w/o ours</td><td>w ours</td><td>w/o ours</td><td>w ours</td><td>w/o ours</td><td>w ours</td></tr><tr><td>Accuracy %</td><td>70.7</td><td>86.3</td><td>69.0</td><td>87.3</td><td>70.0</td><td>83.0</td><td>62.3</td><td>79.3</td><td>76.7</td></tr></table>

Table 2: Quantitative IaOR-VQA. SpatialVLM measures the accuracy by the percentage of answers that fall within 0.5x to 2.0x of the ground truth value. We also evaluate within narrower ranges of 0.75x to 1.33x and 0.9x to 1.11x. “Output number” means VLMs produce number in the response instead of vague descriptions. 

<table><tr><td rowspan="2"></td><td colspan="2">GPT-4V</td><td colspan="2">GPT-4o</td><td colspan="2">LLaVA-1.5</td><td colspan="2">InstructBLIP</td><td rowspan="2">SpatialVLM</td></tr><tr><td>w/o ours</td><td>w ours</td><td>w/o ours</td><td>w ours</td><td>w/o ours</td><td>w ours</td><td>w/o ours</td><td>w ours</td></tr><tr><td>Output numbers %</td><td>0.8</td><td>98.5</td><td>31.5</td><td>99.5</td><td>23.5</td><td>97.0</td><td>28.5</td><td>98.5</td><td>91.0</td></tr><tr><td>In range [50, 200] %</td><td>0.0</td><td>73.5</td><td>14.0</td><td>74.5</td><td>16.5</td><td>43.0</td><td>8.0</td><td>31.5</td><td>33.5</td></tr><tr><td>In range [75, 133] %</td><td>0.0</td><td>69.5</td><td>8.5</td><td>70.5</td><td>6.0</td><td>29.0</td><td>2.5</td><td>22.0</td><td>20.5</td></tr><tr><td>In range [90, 111] %</td><td>0.0</td><td>54.5</td><td>3.0</td><td>55.0</td><td>2.0</td><td>14.5</td><td>0.0</td><td>11.5</td><td>7.5</td></tr></table>

The results in Tables 3 and 4 demonstrate the effectiveness of our approach on both qualitative and quantitative IaOR-VQA and IrSD-VQA tasks. Notably, the performance on quantitative IaOR-VQA is suboptimal compared to quantitative IrSD-VQA, despite the latter being more challenging. We

Table 3: Qualitative IaAD-VQA & IrSD-VQA. Since we test SpatialPIN on one VLM backbone for our proposed spatial VQA, for fair comparison, we should use SpatialVLM backbone (PaLM 2-E [3]). However, since it is not open source, we use GPT-4o as our backbone, as it shows the most improvement with our framework. 

<table><tr><td rowspan="2"></td><td colspan="3">IaAD-VQA</td><td colspan="3">IrSD-VQA</td></tr><tr><td>GPT-4o</td><td>GPT-4o + ours</td><td>SpatialVLM</td><td>GPT-4o</td><td>GPT-4o + ours</td><td>SpatialVLM</td></tr><tr><td>Accuracy %</td><td>68</td><td>84</td><td>62</td><td>64</td><td>82</td><td>54</td></tr></table>

Table 4: Quantitative IaAD-VQA & IrSD-VQA. 

<table><tr><td rowspan="2"></td><td colspan="3">IaAD-VQA</td><td colspan="3">IrSD-VQA</td></tr><tr><td>GPT-4o</td><td>GPT-4o + ours</td><td>SpatialVLM</td><td>GPT-4o</td><td>GPT-4o + ours</td><td>SpatialVLM</td></tr><tr><td>Output numbers %</td><td>38</td><td>100</td><td>66</td><td>30</td><td>100</td><td>78</td></tr><tr><td>In range [50, 200] %</td><td>8</td><td>64</td><td>12</td><td>10</td><td>68</td><td>26</td></tr><tr><td>In range [75, 133] %</td><td>2</td><td>42</td><td>6</td><td>4</td><td>54</td><td>12</td></tr><tr><td>In range [90, 111] %</td><td>0</td><td>30</td><td>2</td><td>2</td><td>38</td><td>4</td></tr></table>

believe this is because, for quantitative IrSD-VQA, the VLM sometimes confuses the camera and world coordinate frames, comparing the object's principal axes with the world axes to reason about changes in angles.

Fig. 4 presents qualitative examples on all forms of spatial VLM.

# 4.2 Robotics Pick and Stack

Pick and stack is a classic robotics task. Given a robot's egocentric observation of a scene with multiple objects and a task description, our pipeline uses traditional planning to solve the problem. This task demands advanced spatial reasoning, as the model must comprehend 3D locations, sizes, and physical properties of the objects (i.e., how much to grasp and how high to drop? Is the object deformable or articulated so the robotic grasper needs to grasp more firmly?). For instance, grasping and stacking a soft toy bear on a cube is significantly different from stacking a solid apple on a mug. The model reasons about grasping and stacking policies, directly outputting 3D trajectories for the robot's end effector using traditional path planning algorithm as external tool.

Set-Up We set up the pick-and-stack problem in the ManiSkill [22] simulator, applying real-world physics properties. Rigid and articulated objects are chosen from the YCB dataset [10] and are randomly allocated on the table within the robotic arm's reach, with observations from different perspectives. We create 50 scenes. Since robot observations from simulated scenes suffer from sim2real gap and consider that most real-world robots have depth sensors, we use ground truth camera matrix and depth.

We compare our method to the following baselines: 1) direct 3D information output from our framework without GPT-4o [1] reasoning about physics and object properties and 2) SpatialVLM with our RRT\* trajectory generation module.

Results Table 5 shows the results, with a qualitative example demonstrated in Fig. 5. The results indicate that using precise 3D information from our framework significantly improves the success rate, and incorporating VLM reasoning further enhances performance.

# 4.3 Discovering and Planning for Robotics Tasks from a Single Image

We present a novel task that requires advanced spatial reasoning capacities of VLMs. Given a single RGB image of any scene comprising unknown environments and objects, the VLM discovers potential tasks and plans their execution with full 3D trajectories, with the motivation that it can be used for robot learning in future research. To solve this complex task and visualize the execution using our framework, we introduce: 1) a task proposal approach using VLM, 2) a novel axes-constrained 3D planning approach that enables spatial reasoning-imbued VLM to plan the object motion based on the proposed tasks by specifying waypoints. Please see Appendix for the pipeline and details.

Dataset We create a diverse evaluation dataset by combining self-captured photos (38) using an iPhone 12 Pro Max and scenes (13) from NOCS [57]. Our dataset covers diverse scenes (e.g., office,

![](images/8db937614a9e32f292deb83e49af1ccabe69dfc1a5f9d76f1f27abef93b7e4cc.jpg)  
"Banana size: 19.8 cm x 6.9 cm x 3.66 cm (WxDxH). Cracker box size: 9.21 cm x 4.51 cm x 17.6 cm (WxDxH). Task: grasp and stack the banana on the cracker box. Determine the grasper's closing distance and the optimal drop height in cm, using the image and spatial information."

![](images/c8b5461085e329500b5a7073bc491db22a366ba4d434bd5753d4728da5a7d5cc.jpg)

![](images/75862dd06b4a8689e13e8b1be975d9493c16305d4a44dc147412c74a6b5c7af0.jpg)

![](images/acac3f319d884d4fe3e9548f453e58e03b61effcb09690bc62e6033e37f1f9ff.jpg)

![](images/04154c70e858ed3b3d0e68fff93be9235b1a4cf6202278b660d71a93b5d000fd.jpg)

![](images/78eacf6a021ef55d0f2602d9cb40a3dcbaa39dcdbe96815f3b589b10ef8ec575.jpg)  
"The robot should grasp the banana at its thinnest side with a width of approximately 3.5 cm (just less than the banana's 3.66 cm) for a firm grip. It should then lift the banana to 22.26 cm (cracker box height of 17.6 cm plus banana height of 3.66 cm plus a 1 cm margin) and drop it from 22.16 cm to stack it securely on the cracker box. Using RRT\*, the planned 3D trajectory is ..."

![](images/9d12aecaeb6aa5ba0e84268290d74149268c9c68ca39b83bd105b256af3bf2cc.jpg)

<details>
<summary>line</summary>

| x      | y      |
| ------ | ------ |
| -0.088 | 0.00   |
| -0.072 | 0.05   |
| -0.064 | 0.10   |
| -0.045 | 0.15   |
| -0.025 | 0.20   |
</details>

"The knife is used to slice the orange into halves or wedges on the cutting board."

![](images/c6e673c7c766b6abb725e3bb5e61ed16f58245f15920ff393440a6831c46c9e5.jpg)  
Input

![](images/ad9e53cab74f31c3bafdf1941e5ca261d653c0cb928ce5252f302c6054b7b183.jpg)

<details>
<summary>natural_image</summary>

Sequence of five photos showing a kitchen scene with cooking utensils and food items on stovetop (no text or symbols)
</details>

Planned Task Trajectories   
Figure 5: Qualitative examples of pick and stack (top) and task trajectory planning (bottom). SpatialPIN successfully outputs picking and stacking policies using spatial reasoning and plans 3D trajectories with geometric awareness to align with task descriptions.

Table 5: Pick and stack. We classify the success rates into: 1) successfully picked, 2) successfully picked and contacted the target object but slipped/collided, and 3) successfully picked and stacked. 

<table><tr><td></td><td>GPT-4o + ours</td><td>Our Direct 3D Info</td><td>SpatialVLM + RRT*</td></tr><tr><td>Picked %</td><td>44</td><td>28</td><td>16</td></tr><tr><td>Picked &amp; contacted %</td><td>8</td><td>12</td><td>6</td></tr><tr><td>Picked &amp; stacked %</td><td>36</td><td>16</td><td>10</td></tr></table>

kitchen, bathroom), and features a rich diversity of object categories (116) and quantities (185), with each image containing 1 – 7 objects and 1 – 3 tasks proposed for each object (278 tasks/planned trajectories in total). The dataset's diversity is further enhanced by the variety of perspectives (e.g., frontal, top-down, side views). This deliberate choice of diverse angles, both in our own image capturing process and through the random extraction of frames from NOCS, aims to simulate a realistic and challenging array of scenes for evaluation. See Appendix for statistics and visuals.

Qualitative Demonstration We present a qualitative example in Fig. 5. Additional examples in Appendix shows our framework's capability to produce diverse and accurate task trajectories spanning various scenes and tasks.

Human Evaluation: User Study We rely on human preference evaluation as one of our quantitative metrics. We ask 25 users to rate 5 translation and 5 rotation task executions in terms of task description alignment. For these complex context-dependent manipulation tasks, we instead ask users to judge 10 executions relative to human action, and to encapsulate their perception of the action in our with a single sentence. These sentence description will be used to test human understanding of our planned trajectories (please see Appendix). Note that our user study size is similar to those representative works such as ControlNet [74] and Prompt-to-Prompt [24]. Results in Table. 6.

Table 6: User study. Ratings (scale 1 – 5) are averaged. 

<table><tr><td></td><td>Rating ↑</td></tr><tr><td>Rotation</td><td>4.58</td></tr><tr><td>Translation</td><td>4.43</td></tr><tr><td>Manipulation</td><td>4.29</td></tr></table>

Machine Understanding We assess the interpretability of our generated task executions from a machine's perspective using SOTA video understanding model, Video-LLaVA-7B [35]. We use two approaches: binary classification and descriptive generation. For classification, we feed the model with the task descriptions generated by VLM and ask question (is the video doing...?). In generation, we prompt Video-LLaVA-7B to articulate its interpretation of our task executions. To quantify the correspondence between the model's perception and the tasks, we use OpenCLIP cosine similarity score [15].

Table 8: Ablation study. For quantitative IaOR-VQA, the accuracy is measured by the answers that fall within 0.75x to 1.33x of the ground truth value. 

<table><tr><td rowspan="2"></td><td colspan="3">Overall Design</td><td>2D Understanding</td><td colspan="3">3D Understanding</td><td rowspan="2">Ours</td></tr><tr><td>ShAPO</td><td>SAM-6D + 3D models</td><td>SpatialVLM</td><td>w/o objects</td><td>w/o coarse</td><td>w/o fine-grained</td><td>w/o both</td></tr><tr><td>Qualitative</td><td>36.7</td><td>48.0</td><td>81.3</td><td>68.3</td><td>76.0</td><td>63.3</td><td>61.7</td><td>87.3</td></tr><tr><td>Quantitative</td><td>29.5</td><td>37.0</td><td>62.5</td><td>54.5</td><td>64.5</td><td>50.5</td><td>58.0</td><td>70.5</td></tr></table>

However, we find that even SOTA video understanding model shows limited performance. To assess false positive rate in classification, we deliberately misalign the sequence of generated task executions with their corresponding task descriptions, expecting a theoretical accuracy of 0%. Contrary to expectations, Video-LLaVA-7B reports a false positive rate of 36.3%. To adjust for this anomaly, we subtract this rate from the model's raw accuracy for correctly aligned video-task pairs. This method, while unconventional, provides a more fair and reasonable evaluation of machine video understanding, underscoring the current challenges faced by video understanding models in accurately interpreting complex video content. Results in Table. 7.

Table 7: Results for machine understanding (classification and generation) on 278 task executions. 

<table><tr><td></td><td>Machine</td></tr><tr><td>Raw Acc ↑</td><td>0.974</td></tr><tr><td>Fal-Pos Rate ↓</td><td>0.363</td></tr><tr><td>True Acc ↑</td><td>0.611</td></tr><tr><td>OpenCLIP ↑</td><td>0.636</td></tr></table>

# 4.4 Ablation Study

We evaluate the effectiveness of each module in our framework on IaOR-VQA by 1) seeking alternative designs of the overall pipeline and 2) removing each component in our ablations.

Overall Design To demonstrate our framework's generalization across a wide range of objects, We replace our 2D + 3D pipeline with: 1) SOTA mesh-free single image object pose and size estimation model, ShAPO [27], 2) SOTA mesh-based single image object pose and size estimation model, SAM-6D [36], and feeds it with the object 3D model reconstructed by One-2-3-45++ [41], and 3) the data generation backbone of SpatialVLM [12]. Since models 1) and 2) do not provide language annotations for their outputs, we first summarize the numerical outputs using our approach in Sec. 3.3. Then, GPT-4V identifies QA pairs.

Removing 2D Understanding Module In this case, the VLM no longer examines the objects through prompting, and only the object name is input into the language-guided segmentation model.

Removing 3D Understanding Modules This means there is no scene size estimation, and the VLM does not conduct perspective canonicalization. During 3D scene reconstruction, we assume the image plane width to be 1 meter.

To validate the fine-grained 3D scene understanding module, we replace object mask raycasting with backprojection using the object's 2D center and remove the object scale calibration.

To demonstrate the overall effectiveness of our 3D understanding modules, we simply backproject the input image with the estimated metric depth.

Results Table 8 demonstrates the effectiveness of each module in our framework. The results also highlight the limitations of using off-the-shelf SOTA mesh-free and mesh-based single-image object pose and size estimation methods as our backbone. These methods are not language-driven and may struggle to generalize to novel objects in diverse input scenes.

# 5 Discussion and Conclusion

We present SpatialPIN, a framework designed to enhance the spatial reasoning capabilities of VLMs through prompting and interacting with 3D priors in a zero-shot, training-free manner. We see our work as a step towards equipping VLMs with more generalized spatial reasoning capacities, demonstrated through applications in various forms of spatial VQA and both traditional and novel robotics tasks.

Limitations Readers may be curious about the inference speed of our framework. The bottleneck is the 3D object reconstruction process and the API call to closed-source VLMs ( $\sim$ 20 seconds per image). However, we want to highlight that this process runs only once per image, and the speed is expected to improve with future versions of 3D foundation models.

# References

[1] Open AI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024.   
[2] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
[3] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, and et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.   
[4] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023.   
[5] Reiner Birkl, Diana Wofk, and Matthias Müller. Midas v3.1 - A model zoo for robust monocular relative depth estimation. arXiv preprint arXiv:2307.14460, 2023.   
[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, and et al. RT-1: robotics transformer for real-world control at scale. In Robotics: Science and Systems, 2023.   
[7] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Conference on Computer Vision and Pattern Recognition, pages 18392–18402. IEEE, 2023.   
[8] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.   
[9] Dingding Cai, Janne Heikkilä, and Esa Rahtu. OVE6D: object viewpoint encoding for depth-based 6d object pose estimation. In Conference on Computer Vision and Pattern Recognition, pages 6793–6803. IEEE, 2022.   
[10] Berk Çalli, Arjun Singh, Aaron Walsman, Siddhartha S. Srinivasa, Pieter Abbeel, and Aaron M. Dollar. The YCB object and model set: Towards common benchmarks for manipulation research. In International Conference on Advanced Robotics, pages 510–517. IEEE, 2015.   
[11] Thomas Carta, Clément Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, and Pierre-Yves Oudeyer. Grounding large language models in interactive environments with online reinforcement learning. In International Conference on Machine Learning, pages 3676–3713, 2023.   
[12] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. arXiv preprint arXiv:2401.12168, 2024.   
[13] Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, and Geoffrey E. Hinton. Pix2seq: A language modeling framework for object detection. In International Conference on Learning Representations, 2022.   
[14] Xi Chen, Xiao Wang, Soravit Changpinyo, A. J. Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V. Thapliyal, James Bradbury, and Weicheng Kuo. Pali: A jointly-scaled multilingual language-image model. In International Conference on Learning Representations, 2023.

[15] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Conference on Computer Vision and Pattern Recognition, pages 2818–2829. IEEE, 2023.   
[16] Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn. DIML/CVL RGB-D dataset: 2m RGB-D images of natural indoor and outdoor scenes. arXiv preprint arXiv:2110.11590, 2021.   
[17] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018.   
[18] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems, 2023.   
[19] Murtaza Dalal, Tarun Chiruvolu, Devendra Singh Chaplot, and Ruslan Salakhutdinov. Plan-seq-learn: Language model guided RL for solving long horizon robotics tasks. In CoRL Workshop on Learning Effective Abstractions for Planning, 2023.   
[20] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, pages 8469–8488, 2023.   
[21] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The KITTI dataset. The International Journal of Robotics Research, pages 1231–1237, 2013.   
[22] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In International Conference on Learning Representations, 2023.   
[23] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In Conference on Learning Representations, 2023.   
[24] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross-attention control. In International Conference on Learning Representations, 2023.   
[25] Ming-Kuei Hu. Visual pattern recognition by moment invariants. IRE Transactions on Information Theory, pages 179-187, 1962.   
[26] Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao. Copa: General robotic manipulation through spatial constraints of parts with foundation models. arXiv preprint arXiv:2403.08248, 2024.   
[27] Muhammad Zubair Irshad, Sergey Zakharov, Rares Ambrus, Thomas Kollar, Zsolt Kira, and Adrien Gaidon. Shapo: Implicit representations for multi-object shape, appearance, and pose optimization. In European Conference on Computer Vision, pages 275–292, 2022.   
[28] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. Robotics and Automation Letters, pages 3019–3026, 2020.   
[29] Linyi Jin, Jianming Zhang, Yannick Hold-Geoffroy, Oliver Wang, Kevin Blackburn-Matzen, Matthew Sticha, and David F. Fouhey. Perspective fields for single image camera calibration. In Conference on Computer Vision and Pattern Recognition, pages 17307–17316. IEEE, 2023.   
[30] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In Conference on Computer Vision and Pattern Recognition, pages 1988–1997. IEEE, 2017.

[31] Sertac Karaman and Emilio Frazzoli. Sampling-based algorithms for optimal motion planning. The International Journal of Robotics Research, pages 846–894, 2011.   
[32] Kourosh Khoshelham and Sander Oude Elberink. Accuracy and resolution of kinect depth data for indoor mapping applications. Sensors, pages 1437-1454, 2012.   
[33] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv preprint arXiv:2304.02643, 2023.   
[34] Shuang Li, Xavier Puig, Chris Paxton, Yilun Du, Clinton Wang, Linxi Fan, Tao Chen, De-An Huang, Ekin Akyürek, Anima Anandkumar, Jacob Andreas, Igor Mordatch, Antonio Torralba, and Yuke Zhu. Pre-trained language models for interactive decision-making. In Advances in Neural Information Processing Systems, 2022.   
[35] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.   
[36] Jiehong Lin, Lihua Liu, Dekun Lu, and Kui Jia. SAM-6D: segment anything model meets zero-shot 6d object pose estimation. arXiv preprint arXiv:2311.15707, 2023.   
[37] Dingning Liu, Xiaomeng Dong, Renrui Zhang, Xu Luo, Peng Gao, Xiaoshui Huang, Yongshun Gong, and Zhihui Wang. 3daxiesprompts: Unleashing the 3d spatial task capabilities of GPT-4V. arXiv preprint arXiv:312.09738, 2023.   
[38] Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine. MOKA: open-vocabulary robotic manipulation through mark-based visual prompting. arXiv preprint arXiv:2403.03174, 2024.   
[39] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 2023.   
[40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, 2023.   
[41] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. arXiv preprint arXiv:2311.07885, 2023.   
[42] Yuan Liu, Yilin Wen, Sida Peng, Cheng Lin, Xiaoxiao Long, Taku Komura, and Wenping Wang. Gen6d: Generalizable model-free 6-dof object pose estimation from RGB images. In European Conference on Computer Vision, pages 298–315, 2022.   
[43] Chenyang Ma, Xinchi Qiu, Daniel Beutel, and Nicholas Lane. Gradient-less federated gradient boosting tree with learnable learning rates. In Proceedings of the 3rd Workshop on Machine Learning and Systems, pages 56–63, 2023.   
[44] Luca Medeiros. Language segment-anything. https://github.com/luca-medeiros/lang-segment-anything, 2023.   
[45] Oscar Michel, Anand Bhattad, Eli VanderBilt, Ranjay Krishna, Aniruddha Kembhavi, and Tanmay Gupta. OBJECT 3DIT: Language-guided 3d-aware image editing. In Advances in Neural Information Processing Systems, 2023.   
[46] Van Nguyen Nguyen, Thibault Groueix, Georgy Ponimatkin, Vincent Lepetit, and Tomas Hodan. CNOS: A strong baseline for cad-based novel object segmentation. In International Conference on Computer Vision Workshops, pages 2126–2132. IEEE, 2023.   
[47] OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[48] Xinchi Qiu, Heng Hen, Wanru Zhao, Pedro Porto Buarque de Gusmao, and Nicholas Donald Lane. Efficient vertical federated learning with secure aggregation. In Federated Learning Systems (FLSys) Workshop@ MLSys 2023, 2023.   
[49] Xinchi Qiu, Heng Pan, Wanru Zhao, Chenyang Ma, Pedro PB Gusmao, and Nicholas D Lane. vfedsec: Efficient secure aggregation for vertical federated learning via secure layer. arXiv preprint arXiv:2305.16794, 2023.   
[50] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1623–1637, 2022.

[51] remyxai. Vqasynth, 2024. GitHub repository.   
[52] Juan Rocamonde, Victoriano Montesinos, Elvis Nava, Ethan Perez, and David Lindner. Vision-language models are zero-shot reward models for reinforcement learning. arXiv preprint arXiv:2310.12921, 2023.   
[53] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from RGBD images. In European Conference on Computer Vision, pages 746–760, 2012.   
[54] Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Martín-Martín, Fei Xia, Kent Elliott Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, C. Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, and Li Fei-Fei. BEHAVIOR: benchmark for everyday household activities in virtual, interactive, and ecological environments. In Conference on Robot Learning, pages 477–490, 2021.   
[55] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Winter Conference on Applications of Computer Vision, pages 3172–3182. IEEE, 2022.   
[56] Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), 2023.   
[57] He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, and Leonidas J. Guibas. Normalized object coordinate space for category-level 6d object pose and size estimation. In Conference on Computer Vision and Pattern Recognition, pages 2642–2651. IEEE, 2019.   
[58] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. IRS: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In International Conference on Multimedia and Expo, pages 1–6. IEEE, 2021.   
[59] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian A. Scherer. Tartanair: A dataset to push the limits of visual SLAM. In International Conference on Intelligent Robots and Systems, pages 4909–4916. IEEE, 2020.   
[60] Yen-Jen Wang, Bike Zhang, Jianyu Chen, and Koushil Sreenath. Prompt a robot to walk with large language models. arXiv preprint arXiv:2309.09969, 2023.   
[61] Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Zackory Erickson, David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. arXiv preprint arXiv:2311.01455, 2023.   
[62] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with llms enables open-world multi-task agents. In Advances in Neural Information Processing Systems, 2023.   
[63] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems, 2022.   
[64] Bowen Wen, Chaitanya Mitash, Baozhang Ren, and Kostas E. Bekris. se(3)-tracknet: Data-driven 6d pose tracking by calibrating image residuals in synthetic domains. In International Conference on Intelligent Robots and Systems, pages 10367–10373. IEEE, 2020.   
[65] Tsung-Han Wu, Long Lian, Joseph E Gonzalez, Boyi Li, and Trevor Darrell. Self-correcting llm-controlled diffusion models. arXiv preprint arXiv:2311.16090, 2023.   
[66] Ke Xian, Jianming Zhang, Oliver Wang, Long Mai, Zhe Lin, and Zhiguo Cao. Structure-guided ranking loss for single image depth prediction. In Conference on Computer Vision and Pattern Recognition, pages 608–617. IEEE, 2020.   
[67] Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, and Zhiting Hu. Language models meet world models: Embodied experiences enhance language models. In Advances in Neural Information Processing Systems, 2023.

[68] Fengyu Yang and Chenyang Ma. Sparse and complete latent organization for geospatial semantic segmentation. In Conference on Computer Vision and Pattern Recognition, pages 1809–1818, 2022.   
[69] Fengyu Yang, Chenyang Ma, Jiacheng Zhang, Jing Zhu, Wenzhen Yuan, and Andrew Owens. Touch and go: Learning from human-collected vision and touch. Advances in Neural Information Processing Systems, pages 8081–8103, 2022.   
[70] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. arXiv preprint arXiv:2401.10891, 2024.   
[71] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In Conference on Computer Vision and Pattern Recognition, pages 1787–1796. IEEE, 2020.   
[72] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, pages 1094–1100, 2019.   
[73] Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Marcin Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael S. Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Pete Florence. Socratic models: Composing zero-shot multimodal reasoning with language. In International Conference on Learning Representations, 2023.   
[74] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In International Conference on Computer Vision. IEEE, 2023.   
[75] Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. Texygen: A benchmarking platform for text generation models. In SIGIR, pages 1097–1100. ACM, 2018.

# Appendix for SpatialPIN

# A Overview

This Appendix includes: 1) more technical details about our partial 3D scene reconstruction, 2) additional details, templates, and visualizations of our proposed two forms of SpatialVQA, 3) implementation details on our proposed application: discovering and planning for robotics tasks from a single image (task proposal, axes-constrained motion planning through waypoints, and trajectory generation and smoothing), 4) more experiments on our proposed application (dataset statistics, additional qualitative demonstrations, human understanding, and task diversity), and 5) prompt details for VLMs.

# B Partial 3D Scene Reconstruction Details

Metric Depth Estimation Because a significant portion of depth estimation model $[70, 50, 4, 5, 68]$ is trained on depth datasets with depth data determined by sensors $[21, 53]$ and stereo matching $[71, 16, 58, 59, 66]$ , we assume that the predicted normalized depth is the perpendicular distance to the camera plane, instead of a straight line from the object to the camera lens $[32]$ .

Camera Intrinsic Estimation Given an RGB image, With the estimated vertical field of view (FOV), $\theta_{v}$ , through perspective fields [29], the camera focal length f can be found by:

$$
f = \frac {H}{2 \tan \left(\frac {\theta_ {v}}{2}\right)}, \tag {3}
$$

where H is the image height in pixels.

Object 6D Pose Estimation Single-view 3D reconstruction model reconstructs mesh $O_i$ of object $i$ at the 3D origin, in the coordinate frame set by the input mask $M_i^{of}$ , and captures $O_i$ by a pinhole camera. This camera, with 6D pose $P_c = [R_c \mid t_c^{origin}]$ , captures $O_i$ 's canonical pose within the image. Thus, we can restore all objects' canonical poses across all images by identifying $P_c$ .

We use One-2-3-45++ [41], which provides the camera pose. For models without this information, we develop an efficient method to determine the pose by comparing object masks with rendered 3D model templates, inspired from matching-based 6D pose estimation works [36, 46, 9, 42].

We generate a set of object templates, denoted as $\{T_j^i\}_{j=1}^J$ , each rendered from the object's 3D model $O_i$ . These templates are created by positioning the camera at various locations on an icosphere surrounding the object in $SE(3)$ space, which simulates a spherical coverage around the object to capture its geometry from all angles uniformly. For each template $T_j^i$ , we compute a matching score against the occlusion-free object mask $M_i^{of}$ .

We propose a simple yet effective score matching method. We draw a bounding rectangle around the segmented object inside $M_{i}^{of}$ and across all $\{T_{j}^{i}\}_{j=1}^{J}$ , and crop the bounding rectangle. We then calculate the shape similarity between the contour line of cropped $M_{i}^{of}$ and that of each cropped $T_{j}^{i}$ using Hu moments [25]. Additionally, we crop and resize the bounding rectangle to the same dimension, and evaluate the similarity based on the pixel area of the cropped and resized masks. Our score matching method can be formulated as:

$$
m _ {i} ^ {A, h} = \operatorname{Hu} (\operatorname{findContour} (\operatorname{crop} (M _ {i} ^ {o f}))), \quad p a _ {i} ^ {A} = \operatorname{sum} (\operatorname{resize} (\operatorname{crop} (M _ {i} ^ {o f}))),
$$

$$
m _ {i, j} ^ {B, h} = \operatorname{Hu} (\operatorname{findContour} (\operatorname{crop} (T _ {j} ^ {i}))), \quad p a _ {i, j} ^ {B} = \operatorname{sum} (\operatorname{resize} (\operatorname{crop} (T _ {j} ^ {i}))),
$$

$$
\mathcal {L} \left(M _ {i} ^ {o f}, T _ {j} ^ {i}\right) = \alpha \left| 1 - \frac {\min \left(p a _ {i} ^ {A} , p a _ {i , j} ^ {B}\right)}{\max \left(p a _ {i} ^ {A} , p a _ {i , j} ^ {B}\right)} \right| + \beta \sum_ {h = 1} ^ {7} \left| \frac {1}{\operatorname{sgn} \left(m _ {i} ^ {A , h}\right) \cdot \log \left(m _ {i} ^ {A , h}\right)} - \frac {1}{\operatorname{sgn} \left(m _ {i , j} ^ {B , h}\right) \cdot \log \left(m _ {i , j} ^ {B , h}\right)} \right|. \tag {4}
$$

This dual approach allows for a comprehensive comparison that incorporates both the geometric configuration and the scale of the object representations. The best-matched template can be found by $\arg\min_{j=1}^{J}\mathcal{L}(M_{i}^{of},T_{j}^{i})$ .

![](images/88eefc59c5ac97f46480cbebaee91973e0af4e38f52c7ff326b0a360ddb7c1c3.jpg)  
Figure 6: Example input images of all forms of spatial VQA.

Object Scale Calibration After integrating all 3D object models $\{O_k\}_{k=1}^K$ into the 3D scene, each with a pose $\{P_k = [R_k \mid t_k]\}_{k=1}^K$ and an initial scale $\{S_k^{init}\}_{k=1}^K$ set by the single-view 3D reconstruction model, we refine their scales to accurately reflect depth variations (e.g., moving an apple from close to the camera to a distant corner reduces its apparent size). Through the lens of the pinhole camera with pose $P_c$ , we render binary masks for each object and evaluate the length of their contour lines relative to their occlusion-free masks. The adjusted, final scale of object $i$ can be expressed as:

$$
S _ {i} ^ {\text { adj }} = S _ {i} ^ {\text { init }} \times \frac {\operatorname{arcLength} (\operatorname{findContour} (M _ {i} ^ {\text { of }}))}{\operatorname{arcLength} (\operatorname{findContour} (M _ {i} ^ {\text { rend }}))}, \tag {5}
$$

where $M_{i}^{rend}$ is the rendered mask of object i.

# C Additional Experiments and Details

# C.1 Spatial Visual Question Answering

Intra-Image Angular Discrepancies VQA (IaAD-VQA) For annotation, since YCBInEOAT [64] offers ground truth object 6D poses, we first determine the table/ground plane using the principal axes of objects resting on it (if present). Then, we calculate the angles between the principal axes of different objects to annotate a list of qualitative and quantitative QA pairs. We provide a subset of the question template below.

# Qualitative questions:

```txt
Is [A] tilted?
Is [A] tilted to the left?
Is [A] inclined to the back?
Is [A] more tilted than [B]?
Is [A] more tilted than [B] to the back?
Is [A] more inclined than [B] to the right?
Is [A] leaning towards [B] vertically?
Is [A] straighter than [B]?
Along which axis (W, D, H) is [A] more tilted?
Which object(s) are not upright?
How many object(s) are not upright? 
```

# Quantitative questions:

What is the inclination angle of [A] along the vertical axis?

How many degrees is [A] tilted horizontally?

Calculate the angle of tilt for [A] towards back.

Measure the tilt of [A] relative to the front.   
What is the relative angle between [A] and [B]?   
Measure the angle between [A] and [B].   
What is the angle between the horizontal axis of [A] and [B]?   
How much is [A] inclined along the depth axis compared to [B]?   
Measure the inclination difference along the vertical axis for [A] and [B].   
Measure the angular deviation of [A] and [B] along the vertical axis.   
Determine the angular difference between the depth axes of [A] and [B].   
Determine the tilt difference between [A] and [B] along the horizontal axis.   
Compare the angles of tilt for [A] and [B] along the vertical axis.

Inter-Image Spatial Dynamics VQA (IrSD-VQA) For annotation, we manually measure the changes in objects' locations and angles between two photos. To measure the change in camera shot angle, we record the change in the angle of the tripod to which the iPhone 12 Pro Max is attached. We provide a subset of the question template below.

# Qualitative questions:

Does [A] move?  
Does [A] rotate?  
Does [A] rotate clockwise?  
Does [A] incline?  
Does [A] move to the right?  
Does [A] move closer to [B]?  
Does [A] become more upright?  
Does [A] incline more to the back?  
Does the angle between [A] and [B] become smaller?  
Along which direction does [A] move?  
Along which axis (W, D, H) does [A] rotate?  
Which object(s) move?  
How many object(s) rotate?  
How many object(s) become more tilted to the back?  
Does the camera shot angle change?  
Along which axis (W, D, H) does the camera shot angle change?

# Quantitative questions:

How far does [A] move vertically?   
How far does [A] move horizontally?   
How far does [A] move towards the back?   
How many degrees does [A] rotate clockwise?   
How many degrees does [A] rotate counterclockwise?   
What is the total distance [A] moves from its original position? Calculate the angle of inclination of [A] in the second image. Measure the tilt of [A] relative to the first image.   
What is the change in height of [A] from the first to the second image?   
How much does the distance between [A] and [B] change?   
How much does [A] incline towards the left compared to the first image?   
What is the angular displacement of [A] towards the right?   
Measure the rotation angle of [A] about its own axis.   
What is the new distance between [A] and [B] in the second image? Calculate the difference in the inclination angle of [A] between the two images. Determine the change in angle of [A] relative to the ground plane. What is the relative movement of [A] with respect to [B]?   
Measure the angular deviation of [A] and [B] along the vertical axis. By how many degrees has the camera shot angle changed? Along which axis/axes has the camera shot angle changed?   
How does [A] appear to move if we do not account for the camera shot angle change? What is the perceived change in orientation of [A] due to the camera angle change?

![](images/0cc2380c3761acbf9301220e992987d25560a173982634c36036c18f15858403.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input RGB Image"] --> B["Understanding Objects by Constraining"]
    B --> C["Language-Guided Segmentation"]
    C --> D["Single-View 3D Object Reconstruction"]
    D --> E["Zero-Shot 3D Scene Reconstruction"]
    E --> F["Trajectory Generation & Smoothing"]
    F --> G["Generated Task Execution"]

    subgraph A (Understanding 2D Scene)
        H["Inpainted objects & background"] --> I
    end

    subgraph B (Understanding 3D Scene)
        J["Object 3D models"] --> K
        L["Reconstructed 3D scene"] --> M
    end

    subgraph C (Planning & Executing Tasks)
        N["Generated, smoothed trajectory"] --> O
        P["Motion Planning"] --> Q
    end

    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#fff,stroke:#333
    style I fill:#fff,stroke:#333
    style J fill:#fff,stroke:#333
    style K fill:#fff,stroke:#333
    style L fill:#fff,stroke:#333
    style M fill:#fff,stroke:#333
    style N fill:#fff,stroke:#333
    style O fill:#fff,stroke:#333
    style P fill:#fff,stroke:#333
    style Q fill:#fff,stroke:#333
```
</details>

Figure 7: Pipeline for discovering and planning for robotics tasks from a single image. It incorporate the task proposal and motion planning modules based on SpatialPIN.

How does the position of [A] change relative to the camera angle difference?

What is the actual move distance of [A] when accounting for the camera shot angle change?

Dataset Examples We illustrate example input images of all forms of spatial VQA in Fig. 6.

# C.2 Discovering and Planning for Robotics Tasks from a Single Image

Given a single RGB image of a scene with unknown environments and objects, the VLM identifies potential tasks and plans their execution using full 3D trajectories, complete with visualization. Figure 7 shows the pipeline, which incorporates task proposal and axes-constrained motion planning modules into SpatialPIN's pipeline from our main paper (Fig. 2).

Task Proposal We query VLM to propose meaningful, diverse tasks, each with a one-sentence task description. Instead of directly querying VLM for task proposal, we employ a hybrid approach that integrates role-play and object-based initialization. In the role-play scenario, we prompt VLM to envision itself as a robotic/human hand working in the scene to perform household tasks. For the object-based initialization, we guide VLM to sequentially focus on each identified object within the scene. When the scene contains more than one identified objects, VLM is instructed to suggest two tasks emphasizing interactions between the manipulating object and any of the detected objects, and an additional task focused solely on the manipulating object. If only one object is detected, VLM is directed to propose a task involving just that object. This strategy guarantees a broad spectrum of task suggestions, ensuring comprehensive object engagement.

To further tailor the task proposals, we impose specific constraints, directing VLM to consider the practical affordances of objects while encouraging creative assumptions (e.g., a bowl's capacity to hold water) and potential interactions (e.g., transferring water from a cup into a bowl). Additionally, we delineate clear boundaries by excluding tasks that entail the disassembly of objects, functionality tests, or the involvement of imaginary objects, thereby focusing on feasible and meaningful tasks.

As a concrete example, given the image on the left of Fig. 2, with the manipulating object to be the red can, VLM will propose the following tasks:

“Task name: Can to Bowl Transfer

Description: Pick up the can and pour its contents into the bowl.

Task name: Can Relocation

Description: Pick up the can and place it inside the bowl.

Task name: Can Rotation

Description: Rotate the can 90 degrees on its vertical axis. ...”

Axes-Constrained Motion Planning through Waypoint We introduce a novel method to guide VLM to conduct motion planning within a 3D scene based on a proposed task by planning motion waypoints along the manipulating object's principal axes. More specifically, we define four types of manipulations that VLM can use:

![](images/9f99dc429b5f98231719bc2dd8efb037a37f2bbcd1a6e2dde33660646f11ea12.jpg)

<details>
<summary>pie</summary>

| Category | Value |
|---|---|
| NOCS | 100 |
| Hallway | 85 |
| Kitchen | 75 |
| Office | 65 |
| Classroom | 55 |
| Coffee Station | 45 |
| Pantry | 40 |
| Lounge | 35 |
| Livingroom | 30 |
| Bedroom | 25 |
| Dining Room | 20 |
Diverse scenes: 1-10; 1-20; 2-30; 3-40; 4-50; 50-60; 60-70; 70-80; 80-90; 90-100; 100-110; 110-120; 120-130; 130-140; 140-150; 150-160; 160-170; 170-180; 180-190; 190-200; 200-210; 210-220; 220-230; 230-240; 240-250; 250-260; 260-270; 270-280; 280-290; 290-300; 300-310; 310-320; 320-330; 330-340; 340-350; 350-360; 360-370; 370-380; 380-390; 390-400; High variety of perspectives: 1-1 (no explicit value shown).
</details>

Figure 8: Dataset statistics. Our dataset presents 51 scenes—13 from NOCS and 38 captured from varied perspectives—featuring a wide range of object categories, quantities, and a diverse set of tasks and planned trajectories.

Rotation type 1: Axial rotation. The object rotates around its principle axes.

Rotation type 2: Rotation relative to the target object.

- pitch: Tilt similar to pouring water, around a horizontal axis formed by the cross product of the connecting directional vector and the target's vertical axis.   
- yaw: Horizontal rotation, like a camera panning, around a vertical axis formed by the cross product of the connecting directional vector and the pitch axis.   
- roll: Rotation like drilling a surface, around the connecting directional vector.

Translation type 1: Defines the goal relative to the target object's principle axes, with translation values for its $[x, y, z]$ axes in centimeters. $[0, 0, 0]$ cm indicates the goal is the center of the target object.

Translation type 2: Sets the goal relative to a directional vector between two reference objects, specifying how far (in cm) object 1 should move towards or away from object 2 along this vector.

Since VLM inherently lacks the capability to provide 3D coordinates and low-level actions directly [60, 19], our method offers a practical workaround by translating natural language instructions into precise motion waypoints. This approach significantly enhances VLM's utility in spatial reasoning and manipulation tasks without requiring direct 3D coordinate generation capabilities. Also, the four types of manipulations we defined are both simple and comprehensive, covering a broad spectrum of manipulation tasks.

As a concrete example, given the image on the left of Fig. 7, with “Task name: Can to Bowl Transfer”, VLM will plan as follows:

“Task Name: Can to Bowl Transfer

Manipulating obj idx: 3

Interacting obj idx: 4

1. Move Manipulating Obj [3] to [6, 0, 7] cm relative to Target Obj [4]'s local [x, y, z] axes.   
2. rotate\_wref: Rotate Manipulating Obj [3] relative to Target Obj [4] around [pitch] axis by [75]

degrees."

In practice, the quality of motion planning by VLMs can be enhanced using various prompting techniques. One such technique is chain-of-thought (CoT) [63], where another LLM guides the VLM to plan each axes-constrained sparse waypoint step by step.

Dataset Statistics See Fig. 8 for statistics and visuals. We standardize all image dimensions by resizing all to $640 \times 480$ .

More Qualitative Demonstration We present more qualitative examples spanning various scenes and tasks, as shown in Fig. 9.

"Move the red eraser to the left of the blue eraser and erase letters."   
![](images/1f438db549e8bfb539f5346f8b054c022e47d6aeb692211e467c6bffbded2cec.jpg)

![](images/cabbf9299ae9c1f677101b3cc1fcfc835bd07153bbb1c14b3978d44114cddacd.jpg)

<details>
<summary>text_image</summary>

A B C D
E F G H
A B C D
E F G H
A B C D
E F G H
A B C D
E F G H
A B C D
E F G H
</details>

![](images/ab63f15da81791cef5a7956b7ed5e9f5ee39ee686321e8f2d6f654299b8fbe35.jpg)

![](images/db4cb55681454825d2a83ee813994c2a194e7e46ba5adc59ced561230bc88fc0.jpg)

<details>
<summary>text_image</summary>

"Lift the milk container and deposit it into the trash bin."
</details>

![](images/3c0a4c600410657204a3333e40e2196e30b4d8da8a03da23a81d929417e9694b.jpg)

![](images/596248321b5ed8149886c8851c49f21e8fea525309e606237225bc41a802a418.jpg)

<details>
<summary>text_image</summary>

"The cup is repositioned under the coffee dispenser to prepare for coffee pouring."
</details>

![](images/dea533098eb15ac1817a305879016d24eb4eb4b5813ac76acc97f9dc7cb0bf20.jpg)

![](images/703cb571e429e065ec9b364e0eb4b064c52f55be9acf8b3efaa65a72a4c37ff5.jpg)

<details>
<summary>natural_image</summary>

Five-panel sequence showing a cooking step with a pot on a gas stove, surrounded by food items (no text or symbols visible)
</details>

![](images/9ea7b29ef5c1c31ef0236c55f0b2ae912d1e119e262f40a31fb9014868b497f3.jpg)

![](images/20659d0f71d69d77c64f02fc12f9b9e3d46204c9cfdd04ac1b2811b7ff3a7724.jpg)

<details>
<summary>text_image</summary>

"Insert the toilet brush into the toilet bowl and scrub for cleaning."
</details>

Input   
Planned Task Trajectories   
Figure 9: More qualitative examples. With diverse input scenes and proposed tasks, our framework produces 3D trajectories with geometric awareness that aligns with the task descriptions. Zoom in for better view.

Table 10: Comparison of task diversity. We sample 106 proposed tasks for fair comparison with RoboGen and previous RL benchmarks. 

<table><tr><td></td><td>Ours</td><td>RoboGen [61]</td><td>Behavior-100 [54]</td><td>RLbench [28]</td><td>MetaWorld [72]</td><td>Maniskill2 [23]</td></tr><tr><td>Number of Tasks</td><td>106</td><td>106</td><td>100</td><td>106</td><td>50</td><td>20</td></tr><tr><td>Self-BLEU ↓</td><td>0.269</td><td>0.284</td><td>0.299</td><td>0.317</td><td>0.322</td><td>0.674</td></tr><tr><td>Embedding Similarity ↓</td><td>0.154</td><td>0.165</td><td>0.210</td><td>0.200</td><td>0.263</td><td>0.194</td></tr></table>

# Experiment on Human Understanding

We aim to understand human perception of our generated task executions by asking participants to provide a one-sentence description. We then evaluate the alignment between these descriptions and the ground truth task descriptions proposed by VLM, using OpenCLIP $[15, 43]$ . Table. 9 reveals a high degree of alignment. Intriguingly, it appears humans understand our task executions more accurately than machines do (Table. 7). We hypothesize this discrepancy stems from the limitations of current video understanding models, whereas humans draw on their prior experiences for a deeper comprehension.

Table 9: Results for machine understanding (generation) on 278 task executions and human understanding, where 25 users write descriptions for 10 tasks. 

<table><tr><td></td><td>Machine</td><td>Human</td></tr><tr><td>OpenCLIP ↑</td><td>0.636</td><td>0.823</td></tr></table>

Experiment on Task Diversity We evaluate the diversity of the proposed tasks in terms of semantic meaning using Self-BLEU and the embedding similarity $[75, 48, 49]$ following RoboGen $[61]$ , where lower scores mean higher diversity. We also compare with previous reinforcement learning (RL) benchmarks. From Table. 10, ours method generates most diverse tasks as it is open to all scenes with no constraint.

# C.3 Trajectory Generation Implementation Details

Trajectory Generation With the waypoints planned by VLM, we generate the manipulating object's trajectory using path planning algorithm, specifically rapidly-exploring random tree star (RRT\*) [31]. To generate accurate collision-free path, we perform K-means clustering on the point clouds of object 3D model with a high number of clusters, segmenting the object mesh into discrete voxels and treating each voxel as an obstacle. Then, to accurately consider the manipulating object's dimensions, we grow the size of each voxel by its dimensions.

Handling VLM Planning Discrepancies The waypoints generated by VLM are typically accurate and practical. Nonetheless, there are instances where the waypoints suggested by VLM lead to collisions as determined by the RRT\* planner. This discrepancy is less about VLM's misunderstanding of the objects' sizes and their spatial relationship and more about the precision level of the waypoints, which may not match the exacting standards of the RRT\* planner's outcomes. To resolve this, we implement Gaussian sampling around the initially planned waypoints whenever a collision is detected. The sampling strategy is guided by a predefined set of geometric rules. In our 3D coordinate system, positive x-axis [1, 0, 0] points right, positive y-axis [0, 1, 0] is away from viewer, positive z-axis [0, 0, 1] is up. For translation type 1, we denote the goal pose relative to the target object's principle axes as [dx, dy, dz]. For translation type 2, we denote the distance that object 1 moves towards object 2 as dD. The set of geometric rules are as follows:

if type 1 & [dx, dy, dz] = [0, 0, 0]: sample along [x, y, z] axes
elif type 1 & dx = 0 & dy = 0 & dz != 0: sample along [z] axis, $z_{sampled} \cdot dz > 1$ elif type 1 & dz = 0: sample along [x, y] axes
elif type 1 & dz != 0: sample along [x, y, z] axes, $z_{sampled} \cdot dz > 1$

if type 2: sample along the connecting directional vector, $D_{sampled} \cdot dD > 1$

Trajectory Smoothing Finally, to ensure our trajectory is natural and smooth, we linearly interpolate rotation and interpolate translation using cubic spline.

# D Prompt Details

We show exact prompts for VLMs for our proposed application: discovering and planning for robotics tasks from a single image.

# Understanding Objects by Constraining Prompt.

Input:

RGB image (640, 480) = (width, height) with multiple objects.

Your task is to identify and objects by precise color, texture, and 2D spatial locations (in words).

Do not use vague phrase like multi-colored.

Please write in the following format. Do not output anything else:

Object idx (actual integer, start from 0): x of color y, texture z at location w.

# Task Proposal Prompt.

Input:

1. RGB image (640, 480) = (width, height) with multiple objects.   
2. Detected objects with index.

You are a single robot hand working in this image scene to perform simple household tasks. Tasks must be discovered from the image. Consider objects' affordances and feel free to make assumptions (e.g., a bowl can contain water) and interactions with other objects (e.g., pouring water from a cup into a bowl).

Task types:

1. Interaction between the manipulating object and one of the detected objects (involve translation, or

translation + rotation).

2. Rotate manipulating object (involve rotation).

# Strictly follow constraints:

1. Exclude tasks involving disassembly of objects.   
2. Exclude tasks involving cleaning or functionality testing.   
3. Exclude tasks involving imaginary objects.   
4. Manipulating object moves; interacting object static.   
5. Assume all objects are rigid, without joints or moveable parts (i.e., cannot deform, disassemble, transform). This applies even to objects that are typically articulated (e.g., laptop).

Propose 3 tasks (2 interaction, 1 rotation) for manipulating Object 5. Write in the following format.

Do not output anything else:

Task Name: xxx

Manipulating obj idx: 5

Interacting obj idx: obj\_idx (actual integer, or manipulating obj idx)

Description: basic descriptions.

# Coarse 3D Understanding Prompt.

# Inputs:

1. RGB image (640, 480) = (width, height) with multiple objects   
2. Detected objects with index.   
3. Image scene size.   
4. Maximum and minimum width, depth, and height.

Your task is to identify the camera shot angle (horizontal, top-down, bottom-up). Reason with respect to the visual cues, the image scene size, and maximums and minimums along each dimension. Choose horizontal if not severely angled.

Please write in the following format. Be concise. Do not output anything else:

Visual cues reasoning: ...

Spatial data reasoning: ...

Conclusion: horizontal/top-down/bottom-up.

# Image and Spatial Context Understanding Prompt.

# Inputs:

1. RGB image (640, 480) = (width, height) with multiple objects and their visualized local axes (x red, y green, z blue).   
2. Detected objects with index.   
3. For each detected object, its 3D center, local xyz-axes, size, and spatial relationship relative to other objects.

The 3D coordinate system of the image is in centimeters and follows Blender. Positive x-axis [1, 0, 0] right, positive y-axis [0, 1, 0] away from viewer, positive z-axis [0, 0, 1] up. Positive rotation is counter-clockwise around all axes.

Your task is to learn the spatial context. Do not output.

# Motion Planning Prompt.

# Inputs:

1. RGB image (640, 480) = (width, height) with multiple objects and their visualized local axes (x red, y green, z blue).   
2. Detected objects with index.   
3. Simple household tasks and descriptions to be performed by a single robot hand.

Your goal is to plan fine-grained motions for the manipulating object to complete the tasks using four manipulations, explained as follows:

# Rotation:

rotate\_self: Axial rotation. The object rotates around its local [x/y/z] axis by [degrees].

rotate\_wref: Rotation relative to the target object:

- pitch: Tilt similar to pouring water, around a horizontal axis formed by the cross product of the connecting directional vector and the target's z-axis.   
- yaw: Horizontal rotation, like a camera panning, around a vertical axis formed by the cross product of the connecting directional vector and the pitch axis.

\- roll: Rotation like a drill entering a surface, around the connecting directional vector.

The degrees can be specified in two ways:

\- Exact [degrees]. Positive values rotate the manipulating object towards the target object.

\- Fixed\_towards/fixed\_back. 'fixed\_towards' orients the object towards the target, mimicking actions like pouring (pitch), facing (yaw), or drilling into (yaw+roll) the target. 'fixed\_back' reverses this alignment.

# Translation:

translate\_tar\_obj: Defines the goal relative to the target object's local axes, with translation values for its [local\_x, local\_y, local\_z] axes in centimeters. [0, 0, 0] cm indicates the goal is the center of the target object.

translate\_direct\_axis: Sets the goal relative to a directional vector between two reference objects, specifying how far (in cm) object 1 should move towards or away from object 2 along this vector (positive closer, negative away). Object indices must differ, and if one reference object is the manipulating object, its current location is used.

# Strictly follow caveats:

1. Apply rotate\_wref thoughtfully and sequentially around different axes as needed.   
2. Use the provided spatial information and image effectively for understanding and planning within the 3D scene.   
3. Combine common physical understanding with the scene's spatial details (like relative positions and sizes of objects) for strategic planning.   
4. Remember that objects' local axes' positive directions might require using negative values in rotation and translation for authentic motion planning.

Plan as below. Fill in obj\_idx based on the tasks.

rotate\_self: Rotate Manipulating Object [obj\_idx] around its local axis [x/y/z] by [degrees].
rotate\_wref: Rotate Manipulating Object [obj\_idx] relative to Target Object [target\_obj\_idx] around [pitch/yaw/roll] axis by [degrees/fixed\_towards/fixed\_back].

translate\_tar\_obj: Move Manipulating Object [obj\_idx] to [a, b, c] cm relative to Target Object [target\_obj\_idx]'s local [x, y, z] axes.

translate\_direct\_axis: Move Manipulating Object [obj\_idx] [a] cm along the directional vector from Reference Object [ref\_obj\_1\_idx] to Reference Object [ref\_obj\_2\_idx].

Here are some full examples. Please write in the following format. Do not output anything else:

Task Category: Bear rotation

Description: Rotate the toy bear 90 degrees on its vertical axis.

Motion Planning:

Manipulating obj idx: bear\_idx (actual integer)

Interacting obj idx: bear\_idx (actual integer)

1. rotate\_self: Rotate Manipulating Object [bear\_idx] around its local axis [z] by [90] degrees.

Task Name: Cup content transfer

Description: Pick up the mug and pour its contents into the bowl.

Motion Planning:

Manipulating obj idx: cup\_idx (actual integer)

Interacting obj idx: bowl\_idx (actual integer)

1. translate\_tar\_obj: Move Manipulating Object [cup\_idx] to [5, -7, 5] cm relative to Target Object [bowl\_idx]'s local [x, y, z] axes.

2. rotate\_wref: Rotate Manipulating Object [obj\_idx] relative to Target Object [bowl\_obj\_idx] around [pitch] axis by [fixed\_towards].

Task Name: Screwdriver penetration

Description: Use a screwdriver to penetrate an avocado.

Motion Planning:

Manipulating obj idx: screw\_idx (actual integer)

Interacting obj idx: avocado\_idx (actual integer)

1. translate\_tar\_obj: Move Manipulating Object [screw\_idx] to [-5, -5, 0] cm relative to Target Object [avocado\_idx]'s local [x, y, z] axes.

2. rotate\_wref: Rotate Manipulating Object [screw\_idx] relative to Target Object [avocado\_idx] around [yaw] axis by [fixed\_towards].

3. rotate\_wref: Rotate Manipulating Object [screw\_idx] relative to Target Object [avocado\_idx] around [roll] axis by [360] degrees.