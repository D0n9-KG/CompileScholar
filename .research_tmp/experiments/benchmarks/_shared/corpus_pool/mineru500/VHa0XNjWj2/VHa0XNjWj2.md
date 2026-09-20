# VLM4Bio: A Benchmark Dataset to Evaluate Pretrained Vision-Language Models for Trait Discovery from Biological Images

M. Maruf $^{1*}$ Arka Daw $^{9*}$ Kazi Sajeed Mehrab $^{1}$ Harish Babu Manogaran $^{1}$ Abhilash Neog $^{1}$ Medha Sawhney $^{1}$ Mridul Khurana $^{1}$ James P. Balhoff $^{3}$ Yasin Bakış $^{4}$ Bahadir Altintas $^{4}$ Matthew J Thompson $^{5}$ Elizabeth G Campolongo $^{5}$ Josef C. Uyeda $^{1}$ Hilmar Lapp $^{6}$ Henry L. Bart Jr. $^{4}$ Paula M. Mabee $^{7}$ Yu Su $^{5}$ Wei-Lun Chao $^{5}$ Charles Stewart $^{8}$ Tanya Berger-Wolf $^{5}$ Wasila Dahdul $^{2}$ Anuj Karpatne $^{1*}$

$^{1}$ Virginia Tech $^{2}$ Univ. of California, Irvine $^{3}$ UNC at Chapel Hill $^{4}$ Tulane Univ. $^{5}$ Ohio State Univ. $^{6}$ Duke Univ. $^{7}$ Battelle $^{8}$ Rensselaer Polytechnic Institute $^{9}$ Oak Ridge Lab.

![](images/5cbd8db93cef913b046dc07613ab9ec55986d613402f47bb7c2a8cba8a6b0eed.jpg)

<details>
<summary>text_image</summary>

Biodiversity of Organisms
Bird (10k)
Butterfly (10k)
Fish (10k)
</details>

![](images/5c732c94faa7f7f41228f1e9265d643297bde142dc971661f41af031187de644.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Scientific Tasks"] --> B["Trait Identification"]
    A --> C["Trait Grounding"]
    A --> D["Trait Referring"]
    A --> E["Trait Counting"]
    F["Species Classification"] --> G["●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●"]
```
</details>

![](images/4a668247b6dd1f8d751b93047318f6425e8263f31e5a06b66c5c3ba97b152c39.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Multi-modal Reasoning"] --> B["Prompting"]
    A --> C["Tests for Reasoning"]
    B --> D["Contextual Prompting"]
    B --> E["Dense Captioning"]
    B --> F["Chain-of-Thought (CoT)"]
    C --> G["False Confidence Test (FCT)"]
    C --> H["None of the Above Test (NOTA)"]
```
</details>

Figure 1: Overview of our goals and contributions. We analyze the capabilities of 12 state-of-the-art (SOTA) vision-language models (VLMs) in answering scientific questions using images from three groups of organisms: fishes, birds, and butterflies, over five groups of biologically relevant tasks. We also explore the effectiveness of these models for reasoning using various prompting techniques and tests for reasoning hallucination.

# Abstract

Images are increasingly becoming the currency for documenting biodiversity on the planet, providing novel opportunities for accelerating scientific discoveries in the field of organismal biology, especially with the advent of large vision-language models (VLMs). We ask if pre-trained VLMs can aid scientists in answering a range of biologically relevant questions without any additional fine-tuning. In this paper, we evaluate the effectiveness of 12 state-of-the-art (SOTA) VLMs in the field of organismal biology using a novel dataset, VLM4Bio, consisting of 469K question-answer pairs involving 30K images from three groups of organisms: fishes, birds, and butterflies, covering five biologically relevant tasks. We also explore the effects of applying prompting techniques and tests for reasoning hallucination on the performance of VLMs, shedding new light on the capabilities of current SOTA VLMs in answering biologically relevant questions using images. $^{2}$

# 1 Introduction

There is a growing deluge of images that are being collected, stored, and shared in organismal biology—the branch of biology interested in the study of structure, ecology, and evolution of organisms. In particular, images are increasingly becoming the currency for documenting the vast array of biodiverse organisms on our planet, with repositories containing millions of images of biological specimens collected by scientists in field museums or captured by drones, camera traps, or tourists posting photos on social media. This growing wealth of biological images provides a unique opportunity to understand the scientific mechanisms of how organisms evolve and adapt to their environment directly from images. The traditional approach for advancing knowledge in organismal biology is by discovering the observable characteristics of organisms or traits (e.g., beak color, stripe pattern, and fin curvature) that serve a variety of biological tasks such as defining groups of organisms, understanding their genetic and developmental underpinnings, and analyzing their interactions with environmental selection pressures $[1]$ . However, the measurement of traits is not straightforward and often relies on expert visual attention involving labor-intensive operations and subjective definitions $[2]$ , hindering rapid scientific advancement $[3]$ .

With the recent rise of large foundation models such as vision-language models (VLMs) (e.g., GPT-4, GPT-4V(ision) [4, 5], Gemini [6], LLaMA, and LLaVA [7, 8]) that can simultaneously solve a diverse range of tasks involving text and images, it is pertinent to ask if pre-trained VLMs contain the necessary scientific knowledge to aid biologists in answering a variety of questions pertinent to the discovery of biological traits from images. Note that unlike mainstream tasks in computer vision, understanding scientific images requires knowledge of domain-specific terminologies and reasoning capabilities that are not fully represented in conventional image datasets used for training VLMs. In particular, an important end-goal in scientific applications such as organismal biology is to explain the process of visual reasoning used to arrive at a prediction, often involving the knowledge of biological traits. Hence, to assess the usefulness of VLMs in accelerating discoveries in organismal biology, it is important to test their ability to identify and reason about biological traits automatically from images.

In this work, we assess the zero-shot capabilities of 12 state-of-the-art (SOTA) VLMs, including the proprietary GPT-4V(ision) and the recent GPT-4o along with other open-source VLMs, on five scientifically relevant tasks in organismal biology, namely species classification, trait identification, trait grounding, trait referring, and trait counting. These tasks are designed to test different facets of VLM performance in organismal biology, ranging from measuring predictive accuracy to assessing their ability to reason about their predictions using visual cues of known biological traits. For example, the task of species classification tests the ability of VLMs to discriminate between species, while in trait grounding and referring, we specifically test if VLMs are able to localize morphological traits (e.g., the presence of fins or patterns and colors of birds) within the image. To perform this evaluation, we present VLM4Bio, a benchmark dataset of $\approx 469K$ question-answer pairs based on $30k$ images of three taxonomic groups of organisms: fishes, birds, and butterflies.

Contributions: (1) We present a novel dataset of scientific question-answer pairs to evaluate the effectiveness of VLMs in answering scientific questions across a range of biologically relevant tasks in the field of organismal biology. (2) We present novel benchmarking analyses of the zero-shot effectiveness of pre-trained SOTA VLMs on our dataset, exposing their gaps in advancing scientific knowledge of organismal biology. (3) We present novel comparisons studying the effects of prompting and tests for reasoning hallucination on VLM performance, shedding new light on the reasoning capabilities of SOTA VLMs in organismal biology.

# 2 Related Works

With the rise of SOTA VLMs such as GPT-4V(vision) [5], GPT-4o [9], and Gemini [6], there has been a simultaneous growth in the number of benchmarking analyses published in the last few years to evaluate different facets of VLM performance on a range of mainstream tasks in computer vision. A majority of previous analyses [10, 11] involve evaluations on single tasks like Visual Question Answering (VQA), OK-VQA [12], MSCOCO [13], and GQA [14]. Other datasets such as POPE [15], HaELM [7], LAMM [16], MMBench [17], MM-Vet [18], LVLM-eHub [19], SEED [20], and GAIA [21] have also been developed to evaluate the capabilities of VLMs on complex tasks such as reasoning and ability to handle multimodal data. There are also some recent domain-specific benchmark datasets, such as MathVista [22], which includes a variety of challenging VQA problems in the mathematical domain, MedQA(USMLE) [23] which is a collection of VQA problems from

<table><tr><td>Species Classification</td><td>Trait Identification</td><td>Trait Referring</td></tr><tr><td>Question: What is the scientific name of the butterfly shown in the image? 
Correct Answer: Heliconius timareta</td><td>Question: Is there eye visible in the fish shown in the image? 
Options: 
A) Yes 
B) No 
Correct Answer: A) Yes</td><td>Question: What is the trait of the fish that correspond to the bounding box region [2545, 335, 3510, 423] in the image? 
Options: 
A) dorsal fin 
B) caudal fin 
C) adipose fin 
D) pelvic fin 
Correct Answer: A) dorsal fin</td></tr><tr><td>Question type: Open Questions</td><td>Question type: Multiple Choice Questions</td><td>Question type: Multiple Choice Questions</td></tr><tr><td>Species Classification</td><td>Trait Grounding</td><td>Trait Counting</td></tr><tr><td>Question: What is the scientific name of the bird shown in the image? 
Options: 
A) Geothlypis philadelphia 
B) Vireo atricapilla 
C) Larus glaucescens 
D) Coccothraustes vespertinus 
Correct Answer: C) Larus glaucescens</td><td>Question: What is the bounding box coordinates of the dorsal fin in the fish shown in the image? 
Options: 
A) [453, 620, 557, 724] 
B) [2545, 335, 3510, 423] 
C) [2012, 1001, 2404, 1350] 
D) [3444, 350, 4730, 1114] 
Correct Answer: B) [2545, 335, 3510, 423]</td><td>Question: How many unique fins are visible in the fish shown in the image? The fins that are normally present in a fish are dorsal fin, caudal fin, pectoral fin, pelvic fin, anal fin and adipose fin. 
Correct Answer: 5</td></tr><tr><td>Question type: Multiple Choice Questions</td><td>Question type: Multiple Choice Questions</td><td>Question type: Open Questions</td></tr></table>

Figure 2: Illustrative examples of VLM4Bio tasks with different question-types.

medical exams, and the recent MMMU [11] dataset, which covers expert-level problems from diverse fields such as business, arts, science, health, medicine, and engineering.

VLM4Bio dataset is different from existing benchmarks involving domain-specific datasets because of the following reasons. (1) Focus on organismal biology: While previous works have focused on benchmarking the performance of VLMs on other scientific domains (e.g., Arts and Design, Business, Health, and Medicine in MMMU [11] or Mathematics in MathVista [22]), there exists no previous VQA benchmark dataset in the domain of organismal biology to the best of our knowledge. Our work thus fills a critical gap in evaluating the performance of VLMs in a field of biology that has several societal implications such as monitoring biodiversity and understanding the impact of climate change on species traits and populations. (2) Breadth of Evaluation Tasks: While previous works are tailored to one or a few evaluation tasks, we consider a wide range of tasks motivated by the needs of domain scientists in the field of organismal biology. They include predictive tasks such as species classification and trait identification as well as tasks that require visual reasoning including trait grounding and referring. We also provide novel comparisons about the performance of VLMs on both open-ended and multiple-choice question (MCQ) formats and comparisons over predictive as well as visual reasoning tasks, in contrast to prior works.

# 3 VLM4Bio Tasks

Figure 2 shows illustrative examples of the five VLM4Bio tasks relevant to biologists that we consider in our study, described in detail in the following.

Species Classification: A common (and often the first) task that a biologist considers when examining an organism specimen is to identify its scientific name (or species class). Hence, we consider asking a VLM to provide the scientific name of the organism shown in a given image. There are two types of questions that we consider for this task. First, we consider open-ended questions, where we do not provide any answer choices (or options) to the VLM in the input prompt. The second type is multiple-choice (MC) questions, where we provide four choices of candidate species names for the VLM to choose from (out of which only one is correct while the remaining three are randomly selected from the set of all species classes).

Trait Identification: An important goal in organismal biology is to answer questions regarding the observable characteristics of organisms, also known as traits. We thus consider asking VLMs to identify a particular trait of an organism given its image for two taxonomic groups: fishes and birds. For fishes, we considered 10 binary (presence/absence) traits and generated MC questions for the presence of each trait in an image (with two options: yes or no), whereas for birds, we considered 28 traits covering their color, pattern, and measurements (size and shape of regions) in a multiple-choice format. We provide a detailed list of all fish and bird traits in the Supplementary.

<table><tr><td>Statistics</td><td>Fish-10K</td><td>Bird-10K</td><td>Butterfly-10K</td><td>Fish-500</td><td>Bird-500</td></tr><tr><td># Images</td><td>10,347</td><td>11,092</td><td>10,013</td><td>500</td><td>492</td></tr><tr><td># Species</td><td>495</td><td>188</td><td>60</td><td>60</td><td>47</td></tr><tr><td># Genera</td><td>178</td><td>114</td><td>27</td><td>18</td><td>33</td></tr><tr><td># Traits</td><td>10</td><td>28</td><td>-</td><td>8</td><td>5</td></tr></table>

Table 1: Key statistics of the VLM4Bio dataset.

Trait Grounding and Referring: To further understand the ability of VLMs to visually explain the reasoning behind their prediction of a trait, it is important to evaluate if a VLM correctly identifies the region in the image containing the trait. For this purpose, we consider two other tasks: trait grounding & trait referring, for the taxonomic groups of fishes and birds. In the first task of trait grounding, we ask the VLM to locate a given trait of an organism on its image (i.e., text to location). We consider MC question-format for this task where we provide four options of bounding boxes in the image as candidate answer choices, where one of the bounding boxes correctly contains the trait while the remaining three are randomly sampled from the set of bounding boxes containing other traits of the organism. In the second task of trait referring, we consider the opposite scenario where we provide a bounding box as input to the VLM and ask it to identify the name of the trait present in the bounding box (i.e., location to text). We again provide four answer choices in MC question-format, where only one of the options is correct while the remaining three are randomly sampled from the names of other traits of the organism.

Trait Counting: We simply ask how many traits are present in an image of a fish specimen. This is biologically relevant, for example, to understand the number of fins present in a fish organism. Similar to the species classification task, we have open and MC question-types for this task.

# 4 VLM4Bio Dataset

Data Collection and Preprocessing: We collected images of three taxonomic groups of organisms: fish, birds, and butterflies, each containing around 10K images. Images for fish (Fish-10K) were curated from the larger image collection, FishAIR [24], which contains images from the Great Lakes Invasive Network Project (GLIN) [25] and Integrated Digitized Biocollections (iDigBio) [26]. These images originate from various museum collections such as INHS [27], FMNH [28], OSUM [29], JFBM [30], UMMZ [31] and UWZM [32]. We created the Fish-10K dataset by randomly sampling 10K images and preprocessing the images to crop and remove the background. For consistency, we leverage GroundingDINO [33] to crop the fish body from the background and Segment Anything Model (SAM) [34] to remove the background. We curated the images for butterflies (Butterfly-10K) from the Jiggins Heliconius Collection dataset [35], which has images collected from various sources $^{3}$ . We carefully sampled 10K images for Butterfly-10K from the entire collection to ensure the images capture unique specimens and represent a diverse set of species by adopting the following two steps. First, we filter out images with more than one image from the same view (i.e., dorsal or ventral). Second, we ensure each species has a minimum of 20 images and no more than 2,000 images. The images for birds (Bird-10K) are obtained from the CUB-200-2011 [61] dataset by taking 190 species for which the common name to scientific name mapping is available. This results in a fairly balanced dataset with around 11K images in total. Additional details on dataset preprocessing are provided in the Supplementary.

Annotation: The scientific names for the images of Fish-10K and Butterfly-10K were obtained directly from their respective sources. For Bird-10K, we obtained the scientific names from the iNatLoc500 [62] dataset. We curated around 31K question-answer pairs in both open and multiple-choice (MC) question-formats for evaluating species classification tasks. The species-level trait presence/absence matrix for Fish-10K was manually curated with the help of biological experts co-authored in this paper. We leveraged the Phenoscape knowledge [63] base with manual annotations to procure the presence-absence trait matrix. For Bird-10K, we obtained the trait matrix from the attribute annotations provided along with CUB-200-2011. We constructed approximately 380K question-answer pairs for trait identification tasks. For grounding and referring VQA tasks, the

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Question type</td><td colspan="13">Models</td></tr><tr><td>gpt-4v</td><td>llava v1.5-7b</td><td>llava v1.5-13b</td><td>cogvlm chat</td><td>BLIP flan-xl</td><td>BLIP flan-xxl</td><td>minigpt4 vicuna-7B</td><td>minigpt4 vicuna-13B</td><td>instruct flant5xl</td><td>instruct flant5xxl</td><td>instruct vicuna7B</td><td>instruct vicuna13B</td><td>Random Choice</td></tr><tr><td colspan="15">Species Classification</td></tr><tr><td rowspan="2">Fish-10K</td><td>Open</td><td>1.01</td><td>2.32</td><td>0.40</td><td>0.11</td><td>0.01</td><td>1.59</td><td>0.50</td><td>0.38</td><td>0.00</td><td>1.46</td><td>0.00</td><td>0.00</td><td>0.20</td></tr><tr><td>MC</td><td>35.91</td><td>40.20</td><td>32.27</td><td>31.72</td><td>29.76</td><td>33.36</td><td>29.02</td><td>27.45</td><td>30.86</td><td>31.70</td><td>27.27</td><td>26.57</td><td>25.00</td></tr><tr><td rowspan="2">Bird-10K</td><td>Open</td><td>17.40</td><td>1.45</td><td>2.06</td><td>0.86</td><td>0.00</td><td>0.57</td><td>2.80</td><td>2.56</td><td>0.00</td><td>0.50</td><td>0.07</td><td>0.00</td><td>0.53</td></tr><tr><td>MC</td><td>82.58</td><td>50.32</td><td>55.36</td><td>44.73</td><td>33.68</td><td>34.75</td><td>23.95</td><td>27.62</td><td>36.36</td><td>35.83</td><td>44.00</td><td>46.55</td><td>25.00</td></tr><tr><td rowspan="2">Butterfly-10K</td><td>Open</td><td>0.04</td><td>0.05</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.07</td><td>0.01</td><td>0.00</td><td>0.00</td><td>9.94</td><td>0.00</td><td>1.54</td></tr><tr><td>MC</td><td>28.91</td><td>50.24</td><td>44.58</td><td>36.45</td><td>25.14</td><td>28.88</td><td>33.06</td><td>28.90</td><td>25.28</td><td>36.67</td><td>41.70</td><td>34.48</td><td>25.00</td></tr><tr><td colspan="15">Trait Identification</td></tr><tr><td>Fish-10K</td><td>MC</td><td>82.18</td><td>56.84</td><td>45.15</td><td>46.92</td><td>68.36</td><td>39.33</td><td>55.08</td><td>51.87</td><td>64.34</td><td>39.26</td><td>81.95</td><td>20.69</td><td>50.0</td></tr><tr><td>Bird-10K</td><td>MC</td><td>62.22</td><td>34.68</td><td>46.14</td><td>63.93</td><td>50.11</td><td>41.38</td><td>39.11</td><td>40.44</td><td>47.89</td><td>45.52</td><td>77.91</td><td>89.98</td><td>31.12</td></tr><tr><td colspan="15">Trait Grounding</td></tr><tr><td>Fish-500</td><td>MC</td><td>29.41</td><td>24.87</td><td>17.98</td><td>23.42</td><td>23.32</td><td>25.14</td><td>22.18</td><td>25.58</td><td>7.20</td><td>27.09</td><td>33.51</td><td>26.90</td><td>25.00</td></tr><tr><td>Bird-500</td><td>MC</td><td>8.1</td><td>26.92</td><td>35.36</td><td>23.2</td><td>11.83</td><td>10.52</td><td>15.39</td><td>24.22</td><td>3.48</td><td>0.81</td><td>30.24</td><td>13.91</td><td>25.00</td></tr><tr><td colspan="15">Trait Referring</td></tr><tr><td>Fish-500</td><td>MC</td><td>28.15</td><td>27.07</td><td>29.14</td><td>28.19</td><td>24.93</td><td>25.68</td><td>39.24</td><td>31.21</td><td>31.75</td><td>25.78</td><td>28.04</td><td>32.73</td><td>25.00</td></tr><tr><td>Bird-500</td><td>MC</td><td>42.28</td><td>30.5</td><td>29.64</td><td>18.45</td><td>35.16</td><td>40.59</td><td>26.04</td><td>35.88</td><td>27.52</td><td>41.69</td><td>23.03</td><td>22.69</td><td>25.00</td></tr><tr><td colspan="15">Trait Counting</td></tr><tr><td rowspan="2">Fish-500</td><td>Open</td><td>16.4</td><td>47.4</td><td>52.0</td><td>14.8</td><td>37.6</td><td>63.4</td><td>13.6</td><td>31.53</td><td>50.2</td><td>61.4</td><td>61.4</td><td>0.0</td><td>25.00</td></tr><tr><td>MC</td><td>44.80</td><td>13.20</td><td>54.80</td><td>21.00</td><td>64.8</td><td>78.2</td><td>22.00</td><td>25.00</td><td>74.0</td><td>69.4</td><td>15.80</td><td>11.80</td><td>25.00</td></tr><tr><td colspan="2">Overall</td><td>34.24</td><td>29.0</td><td>31.78</td><td>25.27</td><td>28.91</td><td>30.24</td><td>23.0</td><td>25.19</td><td>28.49</td><td>29.79</td><td>33.92</td><td>23.31</td><td>22.03</td></tr></table>

Table 2: Zero-shot accuracy comparison of VLM baselines (in % ranging from 0 to 100) for the five scientific tasks. Results are color-coded as Best, Second best, Worst, Second worst.

ground truths were manually annotated with the help of expert biologists on our team. We manually annotated bounding boxes corresponding to the traits of 500 fish specimens and 500 bird specimens, which are subsets of the larger Fish-10K and Bird-10K datasets, respectively. In particular, we considered 8 fish traits and 5 bird traits for annotating their bounding boxes, resulting in a total of 26K question-answer pairs. We also used the Fish-500 dataset for the task of trait counting, resulting in a total of 1K question-answer pairs. Across all tasks, our dataset comprises approximately 469K question-answer pairs for 30K biological images (see Table 1). Additional details on data distribution and key statistics are provided in the Supplementary.

VLM Baselines: We consider the following VLM baselines: GPT-4V(ision) $[64]^{4}$ , LLaVA-v1.5 (7B/13B) [65], COG-VLM [66], MiniGPT-4 (Vicuna 7B/13B) [67], BLIP-FLAN-T5-XL/XXL [68], and INSTRUCT-BLIP (Vicuna 7B/13B) [69]. We used the latest checkpoints for each model available to date. We used the same question prompt for all models to ensure consistent comparison of results for a variety of open and multiple-choice (MC) questions across the five scientific tasks of our dataset. All the experiments were conducted using NVIDIA A100 GPUs. See supplementary for more details of the VLM baselines.

Evaluation Metrics: We used micro-averaged accuracy as our evaluation metric for all experiments. We designed a systematic rule-based evaluation pipeline to evaluate VLM responses against the ground truths. For each question category, we provide the accuracy percentage of random choice as a basic baseline, where each possible answer is considered equally likely (yielding an accuracy of 25% for MC questions with four choices).

# 5 Results

Table 2 compares the accuracies of VLMs in percentages (ranging from 0 to 100) across the five tasks and over multiple organism datasets. We make the following observations from this result.

![](images/0168d60aa403c6a96390f49797adb28459f990faab9e221107dbc941d1125112.jpg)

<details>
<summary>text_image</summary>

Trait Identification
Question: Is there eye visible in the fish shown in the image?
Write the answer after writing
'The answer is:'.
Options: A) Yes, B) No.
Answer: The answer is: A)
Yes.
Ground Truth: Yes
Trait Grounding
Question: The bounding box is defined by the coordinates [x1, y1, x2, y2], where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner of the bounding box coordinates of the eye in the fish shown in the image?
Options: A) [970, 1315, 1392, 1525], B) [1565, 1165, 2215, 1559], C) [324, 711, 516, 891], D) [110, 515, 1065, 1252].
Answer: The answer is: A)
[970, 1315, 1392, 1525].
Ground Truth: [324, 711, 516, 891]
Trait Referring
Question: The bounding box is defined by the coordinates [x1, y1, x2, y2], where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner of the bounding box region [324, 711, 516, 891] in the image?
Options: A) caudal fin, B) eye, C) head, D) anal fin.
Answer: The answer is: A)
caudal fin.
Ground Truth: eye.
</details>

Figure 3: Examples of correct and incorrect predictions of GPT-4V for trait identification, trait grounding, and trait-referring tasks related to the “eye”. For visualization assistance, a red-colored bounding box is added around the “eye” in the image.

All VLMs show poor accuracy on open questions but perform better on MC questions. The zero-shot species classification accuracy of all VLMs on open-ended questions is notably weaker than MC questions. Even the best-performing models, LLaVA-13B, GPT-4V, and Instruct-Vicuna-7B, only achieve accuracies of 2.32%, 17.46%, and 3.62%, respectively, across the three organism datasets. This indicates a significant gap in the ability of existing VLMs to capture the scientific knowledge necessary to differentiate between species (often requiring subtle or nuanced features) without being provided with candidate answer choices. Open-ended species classification is particularly hard for pre-trained VLMs that are not typically trained to provide scientific names of organisms (e.g., Lepomis cyanellus) rather than providing their common names (e.g., green sunfish). However, the inclusion of candidate answers (or options) in the question prompt serves as a helpful clue to VLMs for narrowing down the solution space and finding the correct answer potentially using elimination strategies. While VLMs are able to utilize these additional hints and work their way through to the correct answer in MC questions, note that open questions are practically more relevant to scientists operating in real-world settings.

Bird dataset shows better accuracy than Fish or Butterfly datasets. Most VLMs show significantly better performance on the Bird-10K dataset in comparison to the Fish-10K and Butterfly-10K datasets. For example, the highest accuracy across all VLMs on the Bird-10K dataset is 82.58%, while it is 40.20% and 50.24% on the Fish-10K and Butterfly-10K datasets, respectively. A potential reason is that while the bird dataset is a subset of the CUB dataset [70] that is commonly used in machine learning literature and has images with natural in-the-wild backgrounds, the butterfly and fish datasets contain images of specimens preserved in museum collections with artificial backgrounds and with imaging artifacts that are not typical for large-scale computer vision datasets. We hypothesize that many of the pre-trained VLM baselines may have seen images similar to those in the Bird dataset during training, leading to their better performance.

Can VLMs effectively identify biological traits? The performance of most VLMs in trait identification appears significantly better than their performance in species classification, with GPT-4V reaching 82.18% accuracy on the Fish-10K dataset and Instruct-Vicuna-13B achieving 89.98% on Bird-10K. However, some traits such as “eye”, “head”, and “mouth” are almost always present in every organism image, so simply answering “yes, the trait is present” can lead to high accuracy in trait identification. In contrast to the fish dataset, the bird dataset poses more intricate questions regarding a variety of multi-class traits that require a nuanced understanding of colors, patterns, and physical trait dimensions, such as the color of the bill, wing patterns, and tail shapes.

VLMs struggle in localizing traits in images. While most VLMs perform well on the task of Trait Identification, it is crucial to determine if they are focusing on the correct image regions to answer trait-related questions. We thus analyze the performance of VLMs on the tasks of trait grounding (i.e., text to location) and trait referring (i.e., location to text). We can see that there is a significant drop in the accuracy of trait grounding and referring tasks compared to the trait identification task. This shows that while VLMs can potentially leverage knowledge of trait choices to identify traits, they struggle in localizing the traits in the image and thus visually ground their reasoning. Figure 3 shows an illustrative example of GPT-4V prediction where it predicts the presence of the trait “eye” correctly but fails to localize it in grounding and referring tasks.

Counting biological traits is difficult for VLMs. Recent studies $[71, 72, 73]$ have explored the gap in the ability of VLMs to count objects, which is aligned with our results in Table 2. All VLMs, except for BLIP-flan-T5-XXL, show lower performance in counting traits, despite performing well

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Difficulty</td><td colspan="15">Models</td></tr><tr><td>gpt-4v</td><td>gpt-4o</td><td>llava v1.5-7b</td><td>llava v1.5-13b</td><td>cogvlm chat</td><td>BLIP flan-xl</td><td>BLIP flan-xxl</td><td>minigpt4 vicuna-7B</td><td>minigpt4 vicuna-13B</td><td>instruct flant5xl</td><td>instruct flant5xxl</td><td>instruct vicuna7B</td><td>instruct vicuna13B</td><td>CLIP</td><td>BioCLIP</td></tr><tr><td rowspan="2">Fish</td><td>Easy</td><td>44.50</td><td>37.50</td><td>47.50</td><td>46.00</td><td>24.00</td><td>34.00</td><td>27.50</td><td>29.00</td><td>19.50</td><td>32.00</td><td>28.00</td><td>33.50</td><td>33.50</td><td>36.50</td><td>55.50</td></tr><tr><td>Medium</td><td>3.50</td><td>5.50</td><td>30.00</td><td>28.50</td><td>27.00</td><td>26.00</td><td>23.00</td><td>26.50</td><td>25.00</td><td>28.50</td><td>24.50</td><td>26.00</td><td>25.50</td><td>26.00</td><td>29.00</td></tr><tr><td rowspan="2">Bird</td><td>Easy</td><td>73.50</td><td>68.00</td><td>53.50</td><td>50.00</td><td>38.50</td><td>34.50</td><td>36.00</td><td>21.00</td><td>32.00</td><td>41.00</td><td>33.00</td><td>43.50</td><td>39.00</td><td>57.00</td><td>94.00</td></tr><tr><td>Medium</td><td>41.00</td><td>40.50</td><td>30.50</td><td>37.00</td><td>30.00</td><td>25.50</td><td>21.00</td><td>21.00</td><td>24.00</td><td>27.00</td><td>27.00</td><td>24.50</td><td>26.50</td><td>31.00</td><td>95.00</td></tr><tr><td rowspan="3">Butterfly</td><td>Easy</td><td>18.50</td><td>17.50</td><td>19.00</td><td>20.50</td><td>24.50</td><td>30.00</td><td>25.00</td><td>34.50</td><td>26.00</td><td>24.50</td><td>22.50</td><td>19.00</td><td>24.50</td><td>21.50</td><td>65.50</td></tr><tr><td>Medium</td><td>5.50</td><td>7.00</td><td>29.50</td><td>29.00</td><td>29.50</td><td>20.00</td><td>25.50</td><td>33.00</td><td>25.00</td><td>27.50</td><td>25.00</td><td>25.00</td><td>25.00</td><td>21.50</td><td>58.00</td></tr><tr><td>Hard</td><td>2.00</td><td>1.50</td><td>22.00</td><td>21.00</td><td>32.00</td><td>26.50</td><td>20.00</td><td>29.50</td><td>24.00</td><td>22.50</td><td>24.00</td><td>24.00</td><td>21.00</td><td>21.50</td><td>35.00</td></tr></table>

Table 3: Zero-Shot accuracy comparison for easy, medium, and hard datasets. Results are color-coded as Best, Second best, Worst, Second worst.

on the trait identification task. The overall average accuracy for the VLMs is displayed in the last block, with GPT-4V(ision) exhibiting the best performance.

We further analyze the errors of different VLMs to better understand their behavior. We find that GPT-4V shows a reduced rate of incorrect responses but a higher incidence of “Other” responses, which include apologetic expressions, admissions of inability to precisely visualize the organism, and disclaimers regarding lack of expert guidance (see Supplementary for more details).

# 5.1 Analyzing the Role of Answer Choices in MC Questions on VLM Performance

Table 2 showed that VLMs perform drastically better on MC questions compared to Open questions for species classification. A potential hypothesis for this observation is that VLMs are able to avoid incorrect answer choices (or options) that are too different from the correct option and thus are easy to eliminate. To test this hypothesis, we create three variants of the MC questions for species classification—easy, medium, and hard—where species choices in each variant have varying degrees of similarity determined by their taxonomic groupings. In particular, note that the scientific name of an organism contains taxonomic information at three levels: <genus name> <species name> <subspecies name> $^{5}$ . Since organisms that share taxonomic information have similar appearances, it is hard to differentiate species choices if they are from the same taxonomic group. On the other hand, it is easier to work with species choices from different taxonomic groups. Hence, for the easy set, we selected 50 species from different genera, ensuring that all species choices appear quite different from each other. For the medium set, we increased the complexity by constructing species choices from the same genus but from 10 different species. The hard set presented the highest difficulty level for the butterfly dataset, with the answer choices being from the same genus and species but from 10 subspecies. Each difficulty level consists of 200 images from each set of organisms.

Table 3 shows the accuracies of the baseline VLMs for the easy, medium, and hard organism datasets. The pretrained VLMs generally perform best on the easy set and worst on the hard set for each organism. Moreover, there is a gradual improvement in the VLM performance from hard to easy questions. This suggests that the difficulty level of candidate answers (or options) in the question prompt significantly impacts VLMs' performance. Additionally, this outcome indicates that even SOTA VLMs have limitations in handling fine-grained queries. Table 3 shows that GPT-4V and OpenAI's recent release GPT-4o do not perform well when tested on the medium and hard datasets for Fish and Butterfly. Due to this, we further analyze the errors of different VLMs to better understand their behavior. We provide the report in the Supplementary.

# 5.2 Comparing Pre-trained VLMs with a Biologically Fine-tuned Model

We compare BioCLIP [74], a state-of-the-art foundation model for species classification fine-tuned with biological images and taxonomic names (TreeOfLife-10M dataset), with the pretrained VLMs. We observe that BioCLIP significantly outperforms large pretrained VLMs on the Bird-10K and Butterfly datasets, suggesting that BioCLIP has been trained on images that are similar to the organisms present in these datasets. By comparing BioCLIP with CLIP, we can also see that fine-

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Prompting</td><td colspan="7">Models</td></tr><tr><td>gpt-4v</td><td>gpt-4o</td><td>llava v1.5-7b</td><td>llava v1.5-13b</td><td>cogvlm chat</td><td>BLIP flan-xl</td><td>BLIP flan-xxl</td></tr><tr><td rowspan="4">Fish-Prompting</td><td>No Prompting</td><td>34.40</td><td>79.00</td><td>41.60</td><td>35.40</td><td>31.00</td><td>28.60</td><td>22.60</td></tr><tr><td>Contextual</td><td>30.00</td><td>77.20</td><td>40.20</td><td>35.60</td><td>25.60</td><td>27.20</td><td>26.60</td></tr><tr><td>Dense Caption</td><td>18.80</td><td>78.60</td><td>26.00</td><td>27.60</td><td>32.00</td><td>28.40</td><td>29.80</td></tr><tr><td>CoT</td><td>42.60</td><td>86.00</td><td>41.40</td><td>34.80</td><td>26.80</td><td>29.20</td><td>24.60</td></tr><tr><td rowspan="4">Bird-Prompting</td><td>No Prompting</td><td>78.80</td><td>97.60</td><td>44.20</td><td>49.80</td><td>45.40</td><td>35.60</td><td>35.80</td></tr><tr><td>Contextual</td><td>78.60</td><td>98.60</td><td>44.00</td><td>52.00</td><td>49.40</td><td>35.60</td><td>30.40</td></tr><tr><td>Dense Caption</td><td>87.40</td><td>97.00</td><td>33.40</td><td>41.00</td><td>44.00</td><td>25.60</td><td>22.80</td></tr><tr><td>CoT</td><td>62.60</td><td>98.60</td><td>37.40</td><td>47.80</td><td>42.20</td><td>30.60</td><td>31.00</td></tr><tr><td rowspan="4">Butterfly-Prompting</td><td>No Prompting</td><td>13.20</td><td>56.40</td><td>27.20</td><td>26.80</td><td>25.60</td><td>24.40</td><td>21.20</td></tr><tr><td>Contextual</td><td>9.20</td><td>56.20</td><td>26.00</td><td>24.60</td><td>27.20</td><td>23.60</td><td>24.60</td></tr><tr><td>Dense Caption</td><td>49.60</td><td>63.20</td><td>25.20</td><td>23.80</td><td>27.00</td><td>23.20</td><td>23.20</td></tr><tr><td>CoT</td><td>63.60</td><td>74.60</td><td>21.40</td><td>23.20</td><td>34.60</td><td>37.20</td><td>23.60</td></tr></table>

Table 4: Zero-shot accuracy comparison for different prompting techniques of seven VLMs (in % ranging from 0 to 100). Results are color-coded as Best and Worst.

tuning foundation models with biological data provides large gains in classification performance. This suggests that the performance of SOTA VLMs can be further improved by fine-tuning on VLM4Bio Dataset. Further details comparing BioCLIP with SOTA VLMs are provided in the Supplementary.

# 5.3 Analyzing Effects of Prompting on VLM Performance

We considered three prompting techniques: Contextual Prompting, Dense Caption Prompting, and zero-shot Chain of Thought Prompting. For Contextual prompting, we provided a single-line description (context) of the tasks (e.g., we add “Each biological species has a unique scientific name composed of two parts: the first for the genus and the second for the species within that genus.” before the species classification question to give some additional context on the task). Dense Caption prompting involves two stages: (1) first, we prompt the VLM to generate a dense caption of the specimen image such that the caption contains all the necessary trait information of the specimen. (2) We add the dense caption before the question and prompt “Use the above dense caption and the image to answer the following question.” to generate responses from the VLM. Similarly, the Zero-Shot Chain-of-Thought (CoT) happens in two stages: (1) First, we prompt the VLM to generate the reasoning for a given VQA and multiple choices (options). Zero-shot CoT appends “Let’s think step by step.” after the question and options to generate the reasoning. (2) We then add the reasoning after the VQA and prompt “Please consider the following reasoning to formulate your answer” to generate the VLM response. We curated a prompting dataset of 500 multiple-choice (MC) VQAs for each set of organisms, which is a subset of the VLM4Bio dataset for species classification.

Table 4 compares best-performing VLMs on the prompting dataset. The CoT rows of the table demonstrate that only GPT-4V and GPT-4o have reasoning capabilities that can significantly improve their response to biological questions, while smaller models like LLaVa and BLIP do not show much improvement. Furthermore, providing extra context and caption is more useful for GPT-4V and GPT-4o than the smaller models. This resonates with the findings from $[75]$ that the reasoning abilities of VLMs only emerge after a certain model size. The success of Dense Caption prompting and CoT prompting depends on how well they generate the dense caption or the reasoning in the first stage. We report example prompts with VLM responses as case studies in the Supplementary.

# 5.4 Analyzing Tests for Reasoning Hallucination

To further understand whether pretrained VLMs can respond with logically coherent and factually accurate reasoning, we evaluate VLMs on two sets of reasoning for hallucination tests - False Confidence Test (FCT) and None of the Above (NOTA) Test - inspired by [76]. For the FCT, we randomly select an option from the list of given choices and prompt it to the VLM as a “suggested correct answer” along with the question and options. To evaluate VLMs on FCT, we use Accuracy as well as the Agreement score, which is the percentage of times the VLM agrees with the suggested answer, irrespective of whether that is right or wrong. A high agreement score with a low overall accuracy indicates poor performance as it suggests that the model is simply following the suggestion either because of a lack of knowledge or low confidence in its own response. On the other hand,

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metrics</td><td colspan="7">Models</td></tr><tr><td>gpt-4v</td><td>gpt-4o</td><td>llava v1.5-7b</td><td>llava v1.5-13b</td><td>cogvlm chat</td><td>BLIP flan-xl</td><td>BLIP flan-xxl</td></tr><tr><td colspan="9">False Confidence Test (FCT)</td></tr><tr><td rowspan="2">Fish-Prompting</td><td>Accuracy</td><td>34.20</td><td>73.60</td><td>25.00</td><td>28.60</td><td>24.60</td><td>0.00</td><td>7.00</td></tr><tr><td>Agreement Score</td><td>4.40</td><td>16.60</td><td>99.80</td><td>19.20</td><td>74.40</td><td>0.00</td><td>28.4</td></tr><tr><td rowspan="2">Bird-Prompting</td><td>Accuracy</td><td>73.40</td><td>99.00</td><td>25.40</td><td>35.80</td><td>19.80</td><td>0.00</td><td>20.20</td></tr><tr><td>Agreement Score</td><td>11.40</td><td>21.00</td><td>93.20</td><td>17.80</td><td>47.80</td><td>0.00</td><td>79.80</td></tr><tr><td rowspan="2">Butterfly-Prompting</td><td>Accuracy</td><td>5.20</td><td>53.40</td><td>27.20</td><td>26.60</td><td>6.20</td><td>0.00</td><td>5.00</td></tr><tr><td>Agreement Score</td><td>2.60</td><td>12.40</td><td>95.40</td><td>5.60</td><td>13.80</td><td>0.00</td><td>19.00</td></tr><tr><td colspan="9">None of the Above (NOTA) Test</td></tr><tr><td>Fish-Prompting</td><td>Accuracy</td><td>81.40</td><td>44.80</td><td>3.40</td><td>3.80</td><td>0.00</td><td>4.00</td><td>0.00</td></tr><tr><td>Bird-Prompting</td><td>Accuracy</td><td>75.00</td><td>91.40</td><td>1.00</td><td>1.20</td><td>0.00</td><td>31.40</td><td>0.00</td></tr><tr><td>Butterfly-Prompting</td><td>Accuracy</td><td>50.40</td><td>4.60</td><td>1.00</td><td>4.60</td><td>0.00</td><td>51.00</td><td>0.00</td></tr></table>

Table 5: Performance of seven VLMs on the NOTA and FCT reasoning tests. Results are color-coded as Best and Worst.

in the NOTA Test, we replace the correct option with “None of the Above”, requiring the model to produce “None of the above” for all the questions. From Table 5, we can see that LLaVa-v1.5-7B shows poor accuracy on both tests and high agreement score on FCT. Out of all the VLMs, GPT-4V and GPT-4o demonstrate the highest accuracy, i.e., the lowest false confidence. More details on the prompts and examples of the responses have been provided in the Supplementary.

# 6 Limitations

Our work has three main limitations. First, while no prior VQA benchmark dataset exists for organismal biology to the best of our knowledge, we focused on only three organisms—fish, bird, and butterfly—out of the many available due to resource constraints. Adding more organisms with manually annotated trait data will require additional resources and domain expertise, which could be pursued in future work. Second, since it is not feasible to manually inspect all images to ensure that they are free from label noise, we acknowledge that some noise may be present in the labels used for evaluating models on our current dataset, which we plan to address in future iterations. Third, due to resource constraints, certain proprietary VLMs that require purchasing APIs like Gemini-Pro [6], Gemini-Ultra [6], and Claude Opus [77] were also not included in the evaluation. We anticipate that their performance will be comparable to that of the proprietary GPT-4V [5] and GPT-4o [9] considered in our evaluation.

# 7 Conclusion and Future Work

We presented VLM4Bio, a benchmark dataset to evaluate the zero-shot performance of pretrained VLMs on biologically relevant questions involving biodiversity images, exposing gaps in SOTA VLMs when applied to organismal biology. We observe that while VLMs are able to perform reasonably well on simpler tasks, e.g., using questions with multiple-choice formats and images with natural-looking backgrounds, they struggle in complex task settings that are practically more relevant to biologists. Through our study on prompting and reasoning tests on the VLM4Bio dataset, we observe that very large SOTA VLMs such as GPT-4V and GPT-4o have reasoning capabilities that can significantly improve the response to biological questions. We did not explore Retrieval Augmented Generation (RAG) [78] or knowledge-infused prompting [79] techniques since they require additional knowledge bases, which could be developed in future work. Future works can also focus on finetuning VLMs on the VLM4Bio dataset instead of comparing zero-shot performance.

# Acknowledgements

This research is supported by National Science Foundation (NSF) awards for the HDR Imageomics Institute (OAC-2118240). We are thankful for the support of computational resources provided by the

Advanced Research Computing (ARC) Center at Virginia Tech. This manuscript has been authored by UT-Battelle, LLC, under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government retains, and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this manuscript or allow others to do so for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (https://www.energy.gov/doe-public-access-plan).

# References

[1] David Houle and Daniela M Rossoni. Complexity, evolvability, and the process of adaptation. Annual Review of Ecology, Evolution, and Systematics, 53, 2022.   
[2] Tiago R Simões, Michael W Caldwell, Alessandro Palci, and Randall L Nydam. Giant taxon-character matrices: quality of character constructions remains critical regardless of size. Cladistics, 33(2):198–219, 2017.   
[3] Moritz D Lürig, Seth Donoughe, Erik I Svensson, Arthur Porto, and Masahito Tsuboi. Computer vision, machine learning, and the promise of phenomics in ecology and evolutionary biology. Frontiers in Ecology and Evolution, 9:642774, 2021.   
[4] OpenAI. Gpt-4v(ision) system card, 2023. arXiv preprint arXiv:2303.08774, 2023, 2023.   
[5] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023, 2023.   
[6] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
[7] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
[8] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.   
[9] OpenAI. Gpt-4o (“o” for “omni”). https://openai.com/index/hello-gpt-4o/, 2024.   
[10] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of gpt-3 for few-shot knowledge-based vqa. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 3081–3089, 2022.   
[11] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.   
[12] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019.   
[13] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.   
[14] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.   
[15] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.

[16] Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Lu Sheng, Lei Bai, Xiaoshui Huang, Zhiyong Wang, et al. Lamm: Language-assisted multi-modal instruction-tuning dataset, framework, and benchmark. arXiv preprint arXiv:2306.06687, 2023.   
[17] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.   
[18] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.   
[19] Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. arXiv preprint arXiv:2306.09265, 2023.   
[20] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.   
[21] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. arXiv preprint arXiv:2311.12983, 2023.   
[22] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.   
[23] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, 2021.   
[24] fishair.org. Fish-air. fishair.org.   
[25] Great Lakes Invasive Network Project (GLIN). https://greatlakesinvasives.org/portal/index.php.   
[26] Integrated Digitized Biocollections (iDigBio). http://www.idigbio.org/portal (2020).   
[27] Biodiversity occurrence data published by: INHS Collections Data (accessed through the INHS Collections Data Portal, biocoll.inhs.illinois.edu/portal, 2024-06-04).   
[28] FMNH Field Museum of Natural History (Zoology) Fish Collection. Field Museum. https://fmipt.fieldmuseum.org/ipt/resource?r=fmnh\_fishes.   
[29] Daly M and Johnson N. Ohio State University Fish Division (OSUM). Museum of Biological Diversity, The Ohio State University, February 2018.   
[30] JFBM Bell Atlas. 2022. http://bellatlas.umn.edu/index.php.   
[31] UMMZ University of Michigan Museum of Zoology, Division of Fishes. https://ipt.lsa.umich.edu/resource?r=ummz\_fish.   
[32] University of Wisconsin-Madison Zoological Museum - Fish. http://zoology.wisc.edu/uwzm/.   
[33] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.   
[34] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.   
[35] Christopher Lawrence and Elizabeth G. Campolongo. Heliconius collection (cambridge butterfly), 2024.

[36] Gabriela Montejo-Kovacevich, Eva van der Heijden, Nicola Nadeau, and Chris Jiggins. Cambridge butterfly wing collection batch 10, November 2020.   
[37] Patricio A. Salazar, Nicola Nadeau, Gabriela Montejo-Kovacevich, and Chris Jiggins. Sheffield butterfly wing collection - Patricio Salazar, Nicola Nadeau, Ikiam broods batch 1 and 2, November 2020.   
[38] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 2, May 2019.   
[39] Chris Jiggins, Gabriela Montejo-Kovacevich, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 3, May 2019.   
[40] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 4, May 2019.   
[41] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 5, May 2019.   
[42] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 1, February 2019.   
[43] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 3, February 2019.   
[44] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 6, May 2019.   
[45] Chris Jiggins and Ian Warren. Cambridge butterfly wing collection - Chris Jiggins 2001/2 broods batch 1, January 2019.   
[46] Chris Jiggins and Ian Warren. Cambridge butterfly wing collection - Chris Jiggins 2001/2 broods batch 2, January 2019.   
[47] Joana I. Meier, Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jggins. Cambridge butterfly wing collection - Patricio Salazar PhD wild specimens batch 3, October 2020.   
[48] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 1- version 2, May 2019.   
[49] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, Camilo Salazar, Marianne Elias, Imogen Gavins, Eva Wiltshire, Stephen Montgomery, and Owen McMillan. Cambridge and collaborators butterfly wing collection batch 10, May 2019.   
[50] Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jiggins. Cambridge butterfly wing collection - Patricio Salazar PhD wild and bred specimens batch 1, December 2018.   
[51] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 7, May 2019.   
[52] Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jiggins. Cambridge butterfly wing collection - Patricio Salazar PhD wild and bred specimens batch 2, January 2019.   
[53] Erika Pinheiro de Castro, Christopher Jiggins, Karina Lucas da Silva-Brand00e3o, Andre Victor Lucci Freitas, Marcio Zikan Cardoso, Eva Van Der Heijden, Joana Meier, and Ian Warren. Brazilian Butterflies Collected December 2020 to January 2021, February 2022.   
[54] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 8, May 2019.   
[55] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, Eva Wiltshire, and Imogen Gavins. Cambridge butterfly wing collection batch 9, May 2019.

[56] Gabriela Montejo-Kovacevich, Eva van der Heijden, and Chris Jiggins. Cambridge butterfly collection - GMK Broods Ikiam 2018, November 2020.   
[57] Gabriela Montejo-Kovacevich, Quentin Paynter, and Amin Ghane. Heliconius erato cyrbia, Cook Islands (New Zealand) 2016, 2019, 2021, September 2021.   
[58] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 2, February 2019.   
[59] Camilo Salazar, Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Imogen Gavins. Camilo Salazar and Cambridge butterfly wing collection batch 1, May 2019.   
[60] Anniina Mattila, Chris Jiggins, and Ian Warren. University of Helsinki butterfly collection - Anniina Mattila bred specimens, February 2019.   
[61] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.   
[62] Elijah Cole, Kimberly Wilber, Grant Van Horn, Xuan Yang, Marco Fornoni, Pietro Perona, Serge Belongie, Andrew Howard, and Oisin Mac Aodha. On label granularity and object localization. In European Conference on Computer Vision. Springer, 2022.   
[63] Richard C Edmunds, Baofeng Su, James P Balhoff, B Frank Eames, Wasila M Dahdul, Hilmar Lapp, John G Lundberg, Todd J Vision, Rex A Dunham, Paula M Mabee, et al. Phenoscape: identifying candidate genes for evolutionary phenotypes. Molecular biology and evolution, 33(1):13–24, 2015.   
[64] R OpenAI. Gpt-4 technical report. arXiv, pages 2303–08774, 2023.   
[65] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.   
[66] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023.   
[67] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.   
[68] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.   
[69] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.   
[70] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The caltech-ucsd birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.   
[71] Haotong Qin, Ge-Peng Ji, Salman Khan, Deng-Ping Fan, Fahad Shahbaz Khan, and Luc Van Gool. How good is google bard's visual understanding? an empirical study on open challenges, 2023.   
[72] Yao Jiang, Xinyu Yan, Ge-Peng Ji, Keren Fu, Meijun Sun, Huan Xiong, Deng-Ping Fan, and Fahad Shahbaz Khan. Effectiveness assessment of recent large vision-language models. arXiv preprint arXiv:2403.04306, 2024.   
[73] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 9(1):1, 2023.

[74] Samuel Stevens, Jiaman Wu, Matthew J Thompson, Elizabeth G Campolongo, Chan Hee Song, David Edward Carlyn, Li Dong, Wasila M Dahdul, Charles Stewart, Tanya Berger-Wolf, et al. Bioclip: A vision foundation model for the tree of life. arXiv preprint arXiv:2311.18803, 2023.   
[75] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.   
[76] Logesh Kumar Umapathi, Ankit Pal, and Malaikannan Sankarasubbu. Med-halt: Medical domain hallucination test for large language models. arXiv preprint arXiv:2307.15343, 2023.   
[77] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 2024.   
[78] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.   
[79] Ran Xu, Hejie Cui, Yue Yu, Xuan Kan, Wenqi Shi, Yuchen Zhuang, Wei Jin, Joyce Ho, and Carl Yang. Knowledge-infused prompting: Assessing and advancing clinical text data generation with large language models. 2023.   
[80] Boris Sekachev, Nikita Manovich, Maxim Zhiltsov, Andrey Zhavoronkov, Dmitry Kalinin, Ben Hoff, TOsmanov, Dmitry Kruchinin, Artyom Zankevich, DmitriySidnev, Maksim Markelov, Johannes222, Mathis Chenuet, a andre, telenachos, Aleksandr Melnikov, Jijoong Kim, Liron Ilouz, Nikita Glazov, Priya4607, Rush Tehrani, Seungwon Jeong, Vladimir Skubriev, Sebastian Yonekura, vugia truong, zliang7, lizhming, and Tritin Truong. opencv/cvat: v1.1.0, August 2020.   
[81] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. See https://vicuna.lmsys.org (accessed 14 April 2023), 2023.   
[82] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

# VLM4Bio: Supplementary Material

# Table of Contents

A Dataset Preprocessing 15   
B Dataset Card 17   
C Links to Access the Dataset and Its Metadata 17   
D Dataset Availability and Maintenance 18   
E Data Licenses 18   
F Data Distribution and Key Statistics 18   
G Traits Considered for the Task of Trait Identification 19   
H Traits Considered for the Tasks of Trait Grounding and Referring 19   
I VLM Baselines 19   
J Prompts to Evaluate VLM performance 20   
K Error Analyses for VLM Responses 20   
L Comparing Pre-trained VLMs with a Biologically Fine-tuned Model 21   
M Analyzing Effects of Image Resolution on VLM Performance 22   
N Case Studies for Effects of Prompting on VLM Performance 22

N.1 No Prompting 22

N.2 Contextual Prompting 22

N.3 Dense Caption 23

N.4 Chain-Of-Thought Prompting 23

O Case Studies for Reasoning Hallucination Tests 23

O.1 False Confidence Test (FCT) 23

O.2 None of The Above (NOTA) Test 23

# A Dataset Preprocessing

We collected images of three taxonomic groups of organisms: fish, birds, and butterflies, each containing around 10K images. Images for fish (Fish-10K) were curated from the larger image collection, FishAIR [24], which contains images from the Great Lakes Invasive Network Project (GLIN) [25] and Integrated Digitized Biocollections (iDigBio) [26]. These images originate from various museum collections such as INHS [27], FMNH [28], OSUM [29], JFBM [30], UMMZ [31] and UWZM [32]. We created the Fish-10K dataset by randomly sampling 10K images and preprocessing the images to crop and remove the background.

![](images/9f877360a5a5e2f607ea4a1849fbfb694752a67badd8db7ea9fbb9cae96e66f1.jpg)

<details>
<summary>scatter</summary>

| Category          | Count |
| ----------------- | ----- |
| Randomly Sampled  | 120   |
| Others            | 120   |
</details>

(a) Lepomis humilis

![](images/8580d6dba10df408b5bb72e6e8a87d0f7e9628fe3a08b1836c908f54c23c7f27.jpg)

<details>
<summary>scatter</summary>

| Category         | Count |
| ---------------- | ----- |
| Randomly Sampled | 100   |
| Others           | 100   |
</details>

(b) Lepomis cyanellus

![](images/2eae1715af8c45ed0c3e7a261bf9ec7804930dd3c1b863e2872d4d43e52c32ce.jpg)

<details>
<summary>scatter</summary>

| Category          | Count |
| ----------------- | ----- |
| Randomly Sampled  | 120   |
| Others            | 120   |
</details>

(c) Noturus gyrinus

![](images/8ecfbef3ee56e53e0bcbf0a83a68e3bda64c3a8ebb64daeb64fcd50e23f7c57c.jpg)

<details>
<summary>scatter</summary>

| Category         | Count |
| ---------------- | ----- |
| Randomly Sampled | 100   |
| Others           | 100   |
</details>

(d) Lepomis megalotis

![](images/04ece4f5f4e5053be67eefc13a6ce18da9f076fb71c72a0b04768cc015b181e4.jpg)

<details>
<summary>scatter</summary>

| Category          | Count |
| ----------------- | ----- |
| Randomly Sampled  | 100   |
| Others            | 100   |
</details>

(e) Phenacobius mirabilis

![](images/986aa20127ffbbdda82338d15493971ab4078d04209c18f4186336e82e56ad0f.jpg)

<details>
<summary>scatter</summary>

| Category          | Count |
| ----------------- | ----- |
| Randomly Sampled  | 100   |
| Others            | 100   |
</details>

(f) Lepomis macrochirus   
Figure 4: t-SNE plots to illustrate the effectiveness of random sampling with the majority species in the Fish-10K dataset. Randomly sampled images are shown as blue dots, while the remaining data points are represented by red dots. Subcaptions display the scientific names of the corresponding species. To generate the vector representation of the images, we leverage a VGG19 pretrained on the ImageNet dataset.

To ensure diversity within the Fish-10K dataset, we applied a targeted sampling strategy in the source collection, FishAIR [24]. Specifically, we retained all images of species with fewer than 200 images, considering these as minority or rare classes. Random sampling was applied only to the majority species—those with more than 200 images per class. To assess the potential sampling bias among

the majority species, we generated feature vectors for each image in Fish-10K using a pretrained VGG-19 model. In Figure 4, we present species-wise t-SNE plots of these feature vectors for several majority species. Our analysis shows that the distribution of sampled images closely mirrors the distribution of images that were not included in the dataset (denoted as “others” in the plot). This suggests that our random sampling approach provides a sufficiently accurate representation of the original distribution for the majority species. For consistency, we leverage GroundingDINO [33] to crop the fish body from the background and Segment Anything Model (SAM) [34] to remove the background. The Fish-10K dataset contains images of specimens preserved in museum collections with artificial backgrounds with imaging artifacts that are not typical for large-scale computer vision datasets. Moreover, these backgrounds can introduce unexpected bias. Hence, we removed the backgrounds using SAM to create a controlled environment for our experiments.

We curated the images for butterflies (Butterfly-10K) from the Jiggins Heliconius Collection dataset [35], which has images collected from various sources $^{6}$ . We carefully sampled 10K images for Butterfly-10K from the entire collection to ensure the images capture unique specimens and represent a diverse set of species by adopting the following two steps. First, the butterfly images show various angles, including dorsal and ventral views, forewing dorsal and ventral views, and hindwing dorsal and ventral views. To ensure consistency, we only selected images with dorsal view and removed all images of hybrid species. Second, we further filtered the dataset based on the unique specimen ID to ensure no specimen was repeated more than once. For species with more than 2000 images, we performed random sampling (no sampling was performed for species with sizes less than 2000). We ensure each species has a minimum of 20 images and no more than 2,000 images. The Butterfly-10K dataset contains a significant number of images of Heliconius melpomene and Heliconius erato species. We utilized the subspecies information of these two species to create a hard dataset for analyzing the impact of answer choices on VLM performance, as described in Section 5.1.

The images for birds (Bird-10K) are obtained from the CUB-200-2011 [61] dataset by taking 190 species for which the common name to scientific name mapping is available. This results in a fairly balanced dataset with around $11K$ images in total.

The scientific names for the images of Fish-10K and Butterfly-10K were obtained directly from their respective sources. For Bird-10K, we obtained the scientific names from the iNatLoc500 [62] dataset. We curated around 31K question-answer pairs in both open and multiple-choice (MC) question formats for evaluating species classification tasks. The species-level trait presence/absence matrix for Fish-10K was manually curated with the help of biological experts co-authored in this paper. We leveraged the Phenoscape knowledge [63] base with manual annotations to procure the presence-absence trait matrix. For Bird-10K, we obtained the trait matrix from the attribute annotations provided along with CUB-200-2011. We constructed approximately 380K question-answer pairs for trait identification tasks.

For grounding and referring VQA tasks, the ground truths were manually annotated with the help of expert biologists on our team. We manually annotated bounding boxes corresponding to the traits of 500 fish specimens and 500 bird specimens, which are subsets of the larger Fish-10K and Bird-10K datasets, respectively. We used the CVAT tool [80] for annotation. The task-specific question formats with the default prompts are provided in Section K.

# B Dataset Card

We provide the dataset card with a detailed description of the metadata, data instances, annotation, and license information here https://huggingface.co/datasets/sammarfy/VLM4Bio#dataset-card-for-vlm4bio.

# C Links to Access the Dataset and Its Metadata

We provide a GitHub link https://github.com/sammarfy/VLM4Bio and an accessible Hugging Face link https://huggingface.co/datasets/sammarfy/VLM4Bio to access the dataset and its metadata.

<table><tr><td rowspan="2">Statistics</td><td colspan="15">Datasets</td></tr><tr><td>Fish-10K</td><td>Bird-10K</td><td>Butterfly-10K</td><td>Fish-500</td><td>Bird-500</td><td>Fish-Easy</td><td>Fish-Medium</td><td>Bird-Easy</td><td>Bird-Medium</td><td>Butterfly-Easy</td><td>Butterfly-Medium</td><td>Butterfly-Hard</td><td>Fish-Prompting</td><td>Bird-Prompting</td><td>Butterfly-Prompting</td></tr><tr><td>Images</td><td>10,347</td><td>11,092</td><td>10,013</td><td>500</td><td>492</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td><td>500</td><td>500</td><td>500</td></tr><tr><td>Species</td><td>495</td><td>188</td><td>60</td><td>60</td><td>47</td><td>51</td><td>10</td><td>50</td><td>10</td><td>50</td><td>10</td><td>1</td><td>25</td><td>37</td><td>25</td></tr><tr><td>Genera</td><td>178</td><td>114</td><td>27</td><td>18</td><td>33</td><td>10</td><td>1</td><td>10</td><td>1</td><td>10</td><td>1</td><td>1</td><td>12</td><td>30</td><td>10</td></tr><tr><td>Traits</td><td>10</td><td>28</td><td>-</td><td>8</td><td>5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 6: Statistics of the VLM4Bio dataset.

![](images/f235754c5baa1ff15ea039be6a31cf5a5d879d4cd3d5ef40ace872bdd504685a.jpg)

<details>
<summary>histogram</summary>

| Scientific Name | Distribution of Fish-10K (Frequency) | Distribution of Bird-10K (Frequency) | Distribution of Butterfly-10K (Frequency) |
| --------------- | ------------------------------------- | ------------------------------------- | ---------------------------------------- |
| 0               | 650                                   | 60                                    | 2500                                     |
| 1               | 400                                   | 60                                    | 1000                                     |
| 2               | 200                                   | 60                                    | 500                                      |
| 3               | 100                                   | 60                                    | 250                                      |
| 4               | 50                                    | 60                                    | 100                                      |
| 5               | 25                                    | 60                                    | 50                                       |
| 6               | 10                                    | 60                                    | 25                                       |
| 7               | 5                                     | 60                                    | 10                                       |
| 8               | 2                                     | 60                                    | 5                                        |
| 9               | 1                                     | 60                                    | 2                                        |
| 10              | 0.5                                   | 60                                    | 1                                        |
| 11              | 0.2                                   | 60                                    | 0.5                                      |
| 12              | 0.1                                   | 60                                    | 0.2                                      |
| 13              | 0.05                                  | 60                                    | 0.1                                      |
| 14              | 0.02                                  | 60                                    | 0.05                                     |
| 15              | 0.01                                  | 60                                    | 0.02                                     |
| 16              | 0.005                                 | 60                                    | 0.01                                     |
| 17              | 0.002                                 | 60                                    | 0.005                                    |
| 18              | 0.001                                 | 60                                    | 0.002                                    |
| 19              | 0.0005                                | 60                                    | 0.001                                    |
| 20              | 0.0002                                | 60                                    | 0.0005                                   |
| 21              | 0.0001                                | 60                                    | 0.0002                                   |
| 22              | 0.00005                               | 60                                    | 0.0001                                   |
| 23              | 0.00002                               | 60                                    | 0.00005                                  |
| 24              | 0.00001                               | 60                                    | 0.00002                                  |
| 25              | 0.000005                              | 60                                    | 0.00001                                  |
| 26              | 0.000002                              | 60                                    | 0.000005                                 |
| 27              | 0.000001                              | 60                                    | 0.000002                                 |
| 28              | 0.0000005                             | 60                                    | 0.000001                                 |
| 29              | 0.0000002                             | 60                                    | 0.0000005                                |
| 30              | 0.0000001                             | 60                                    | 0.0000002                                |
| 31              | 0.00000005                            | 60                                    | 0.0000001                                |
| 32              | 0.00000002                            | 60                                    | 0.00000005                               |
| 33              | 0.00000001                            | 60                                    | 0.00000002                               |
| 34              | 8                                     | 6                                     | 4                                        |
| 35              | 4                                     | 4                                     | 2                                        |
| 36              | 2                                     | 2                                     | 1                                        |
| 37              | 1                                     | 1                                     | 1                                        |
| 38              | 1                                     | 1                                     | 1                                        |
| 39              | 1                                     | 1                                     | 1                                        |
| 40              | 1                                     | 1                                     | 1                                        |
| Note: The frequency values are not provided in the code, so they are estimated from the plot based on the data source and the number of samples per species. The actual frequencies may vary due to the random nature of the data generation.
</details>

Figure 5: Dataset Distribution of Fish-10K, Bird-10K, and Butterfly-10K.

# D Dataset Availability and Maintenance

The VLM4Bio dataset and metadata are available in a Hugging Face repository. To access the VLM4Bio dataset, please visit https://huggingface.co/datasets/sammarfy/VLM4Bio. Long-term support and maintenance of the dataset will be provided by our team. We have published a code repository for dataset preprocessing, including tasks such as downloading the dataset, reading images and metadata, cropping images, and running the evaluation experiments presented in the VLM4Bio paper. To access the VLM4Bio code repository, please visit https://github.com/sammarfy/VLM4Bio.

# E Data Licenses

VLM4Bio dataset is licensed as Creative Commons Attribution 4.0 International. The images of the corresponding organisms are licensed as follows:

1. Fish Dataset License: CC BY-NC.   
2. Bird Dataset License: CC BY.   
3. Butterfly Dataset License: Creative Commons Attribution 4.0 International.

We provide image-specific licenses in the dataset card https://huggingface.co/datasets/sammarfy/VLM4Bio#licensing-information. We have hosted the dataset on HuggingFace, and the DOI will be made publicly available upon publication.

# F Data Distribution and Key Statistics

Table 6 provides the key statistics for the datasets, including the number of images, species, genera, and traits present in each one. We are examining the Zero-shot accuracy of the VLMs on Fish-10K, Bird-10K, and Butterfly-10K for Species Classification and Trait Identification tasks, Fish-500 and Bird-500 for Trait Grounding, Trait Referring and Trait Counting, and easy, medium, hard, prompting datasets for analyzing the role of answer choices, VLM reasoning and hallucination tests. From Figure 5, it is clear that Fish-10K and Butterfly-10K are imbalanced, with a bias toward some species that are more common in our environment (such as Heliconius erato and Heliconius melpomene for Butterflies). The imbalance in Fish-10K and Butterfly-10K reflects the natural imbalance in the occurrence and observation of species in museum collections. Due to the scarcity of images for the

rare species, it is difficult to increase their representation to avoid imbalance. As a result, we have included many under-represented species in the Fish and Butterfly datasets to report performance on the rare classes. In contrast, the Bird-10K dataset is well-balanced, with most species having 60 images. The easy, medium, hard, and prompting datasets are also balanced, which ensures a comprehensive evaluation of the zero-shot performance of the competing VLMs.

# G Traits Considered for the Task of Trait Identification

<table><tr><td rowspan="2" colspan="2">Fish Traits</td><td colspan="4">Bird Traits</td></tr><tr><td colspan="2">Color</td><td>Pattern</td><td>Measurements</td></tr><tr><td>.Eye. Head. Mouth. Barbel. Dorsal fin</td><td>.Pectoral fin.Pelvic fin.Anal fin.Two dorsal fins.Adipose fin</td><td>.Bill-color.Crown-color.Eye-color.Forehead-color.Nape-color.Primary-color.Throat-color.Back-color</td><td>.Belly-color.Breast-color.Leg-color.Under-tail-color.Underparts-color.Upper-tail-color.Upperparts-color.Wing-color</td><td>.Head-pattern.Back-pattern.Breast-pattern.Wing-pattern.Tail-pattern.Belly-pattern</td><td>.Bill-length.Bill-shape.Shape.Size.Tail-shape.Wing-shape</td></tr></table>

Figure 6: Trait list for Trait Identification task.

Figure 6 shows the Fish traits and Bird traits used for evaluating the VLM's performance in the identification task. For fishes, we considered 10 binary (presence/absence) traits which include the eye, head, mouth, barrel, dorsal fin, pectoral fin, pelvic fin, anal fin, and adipose fin. We generated MC questions for the presence of each trait in an image (with two options: yes or no). Whereas for birds, we considered 28 traits covering their color, pattern, and measurements (size and shape of regions) in a multiple-choice format.

# H Traits Considered for the Tasks of Trait Grounding and Referring

To evaluate the VLM performance in Grounding and Referring, we identified 8 traits for fish and 5 traits for birds. Specifically, we manually annotated the dorsal fin, adipose fin, caudal fin, anal fin, pelvic fin, pectoral fin, head, and eye of the 500 fish specimens. Similarly, for birds, we annotated the beak, head, eye, wings, and tail. Trait grounding and referring tasks are carried out using the Fish-500 and Bird-500 datasets.

# I VLM Baselines

We consider the following VLM baselines to evaluate the performance on VLM4Bio dataset: (1) GPT-4V(ision) [64], which is a proprietary VLM from OpenAI, that uses a generative pre-trained transformer model capable of understanding and generating both text and visual contents, (2) LLaVA-v1.5 (7B/13B) [65], which builds on top of the Vicuna LLM [81] by linearly projecting the visual embedding into the word embedding space. The LLaVA model has two different variants with 7B and 13B parameters, respectively, that depend on the size of the base Vicuna model, (3) COG-VLM [66], which performs a simple concatenation of the image and the text modalities, and uses trainable visual layers in the text-based transformer blocks, (4) MiniGPT-4 (Vicuna 7B/13B) [67], which is similar to LLaVA as it is built on top of the Vicuna model and linearly projects the visual embeddings for better understanding. Similar to LLaVA, MiniGPT-4 is available in two variants depending on the type the base Vicuna model (Vicuna 7B/13B), (5) BLIP-FLAN-T5-XL/XXL [68], which utilizes an effective pre-training strategy that relies on bootstrapping from frozen-pretrained CLIP encoders and LLMS by using a querying transformer block (available as two variants: XL and XXL), and (6) Instruct-BLIP (Vicuna 7B/13B) [69], which performs finetuning on BLIP-2 with visual-instruction tuning data to improve zero-shot capabilities of BLIP-2 (available as two variants depending on the Vicuna model: Vicuna 7B/13B).

<table><tr><td>Task</td><td>Prompt Format</td></tr><tr><td>Species Classification</td><td>What is the scientific name of theshown in the image?Write the answer after writing the answer is:.</td></tr><tr><td>Trait Identification</td><td>Is therevisible in theshown in the image?Write the answer after writing the answer is:.</td></tr><tr><td>Trait Grounding</td><td>What is the bounding box coordinates of thein the fish shown in the image?Write the answer after writing the answer is:.</td></tr><tr><td>Trait Referring</td><td>What is the trait of that corresponds to the bounding box regionin the image?Write the answer after writing the answer is:.</td></tr><tr><td>Trait Counting</td><td>How many uniqueare visible in theshown in the image?Write the answer after writing the answer is:.</td></tr><tr><td>Contextual Prompting</td><td>Each biological species has a unique scientific name composed of two parts: the first for the genus and the second for the species within that genus. What is the scientific name of theshown in the image?Write the answer after writing the answer is:.</td></tr><tr><td>Dense Caption Prompting</td><td>Use the above dense caption and the image to answer the following question. What is the scientific name of theshown in the image?Write the answer after writing the answer is:.</td></tr><tr><td>Chain-of-Thought Prompting</td><td>What is the scientific name of theshown in the image?Please consider the following reasoning to formulate your answer.Write the answer after writing the answer is:.</td></tr><tr><td>False Confidence Test (FCT)</td><td>What is the scientific name of theshown in the image?Chosen Answer:. Please provide: 1) Whether the chosen answer is correct (True/False). 2) The correct answer.</td></tr><tr><td>None of the Above Test (NOTA)</td><td>What is the scientific name of the shown in the image?A)_B)_C)_D) None of the above.&gt; Write the answer after writing the answer is:.</td></tr></table>

Figure 7: Prompts Templates used for Evaluation. There will be no <options> for Open set questions.

# J Prompts to Evaluate VLM performance

In order to ensure a fair comparison of the VLM responses to different types of questions in our dataset, we used the same question prompt for all the models across the various scientific tasks. It's worth noting that each model may perform differently with different prompts. However, for the sake of simplicity in our evaluation, we opted for a consistent prompt for all the models. The prompts specific to each task are displayed in Figure 7.

# K Error Analyses for VLM Responses

We categorize the VLM responses into 3 categories: (1) Correct (%): where the scientific name is accurately predicted, (2) Incorrect (%): where the scientific name is incorrect, and (3) Other (%): a special category for instances where the model abstains from providing a scientific name.

Figure 8a, 8b and 8c show the distribution of errors of different VLMs on Fish-Easy and Fish-Medium, Bird-Easy and Bird-Medium, and Butterfly-Medium and Butterfly-Hard datasets respectively using stacked-bar plots showing the three categories of VLM predictions. GPT-4V, for instance, shows a reduced rate of incorrect responses but a higher incidence of “Other” responses for these datasets, which include apologetic expressions, admissions of inability to precisely visualize the organism, and disclaimers regarding prediction without sufficient expert data and guidance.

To further analyze the type of errors happening in the other (\%) category of VLM predictions, we manually examined 250 randomly selected “Other” GPT-4V responses for the task of fish species classification (MC question type) to generate the pie-chart of error categories shown in Figure 8d. We

![](images/9916da77f5dc6157141257091ddcb704267b68987e98564d8c2215e4886d5f89.jpg)

<details>
<summary>bar_stacked</summary>

| Model | Correct(%) | Incorrect(%) | Other(%) |
|---|---|---|---|
| instruct-vcuna13b | 25 | 70 | 0 |
| instruct-vcuna7b | 25 | 70 | 0 |
| instruct-flan5xx1 | 25 | 70 | 0 |
| instruct-flan5xi | 25 | 70 | 0 |
| minigpt4-vcuna-13B | 25 | 70 | 0 |
| minigpt4-vcuna-7B | 25 | 70 | 0 |
| blip-flan-xx1 | 25 | 70 | 0 |
| blip-flan-xi | 25 | 70 | 0 |
| covpln-chat | 25 | 70 | 0 |
| llava-v1.5-13b | 25 | 70 | 0 |
| llava-v1.5-7b | 25 | 70 | 0 |
| gpt-4o | 25 | 70 | 0 |
| gpt-4v | 25 | 70 | 0 |
Fish-Easy - Fish-Medium - 
| Correct(%) - Fish-Medium - 
| Incorrect(%) - Fish-Medium - 
| Other(%) - Fish-Medium - 
| Instruct-vcuna13b - Fish-Medium - 
| Instruct-vcuna7b - Fish-Medium - 
| Instruct-flan5xx1 - Fish-Medium - 
| Instruct-flan5xi - Fish-Medium - 
| minigpt4-vcuna-13B - Fish-Medium - 
| minigpt4-vcuna-7B - Fish-Medium - 
| blip-flan-xx1 - Fish-Medium - 
| blip-flan-xi - Fish-Medium - 
| covpln-chat - Fish-Medium - 
| llava-v1.5-13b - Fish-Medium - 
| llava-v1.5-7b - Fish-Medium - 
| gpt-4o - Fish-Medium - 
| gpt-4v - Fish-Medium - 
Fish-Easy - Fish-Medium - 
Fish-Medium - Fish-Medium - 
Instruct-vcuna13b: 
Instruct-vcuna7b: 
Instruct-flan5xx1: 
Instruct-flan5xi: 
minigpt4-vcuna-13B: 
minigpt4-vcuna-7B: 
blip-flan-xx1: 
blip-flan-xi: 
clavna-v1.5-13b: 
clavna-v1.5-7b: 
gpt-4o: 
gpt-4v: 
Fish-Easy - Fish-Medium - 
Fish-Medium - Fish-Medium - 
Instruct-vcuna13b: 
Instruct-vcuna7b: 
Instruct-flan5xx1: 
Instruct-flan5xi: 
minigpt4-vcuna-13B: 
minigpt4-vcuna-7B: 
blip-flan-xx1: 
blip-flan-xi: 
clavns-v1.5-13b: 
clavns-v1.5-7b: 
gpt-4o: 
gpt-4v: 
Fish-Easy - Fish-Medium - 
Fish-Medium - Fish-Medium - 
Instruct-vcuna13b: 
Instruct-vcuna7b: 
Instruct-flan5xx1: 
Instruct-flan5xi: 
minigpt4-vcuna-13B: 
minigpt4-vcuna-78: 
blip-flan-xx1: 
blip-flan-xi: 
clavns-v1.5-13b: 
clavns-v1.5-7b: 
gpt-4o: 
gpt-4v:
</details>

(a) Error Analysis for Fish-Easy and -Medium.

![](images/318a93d12c54f682c22fe0bf1859e8bd82be787351adc7764c7a8c7b0a070316.jpg)  
(b) Error Analysis for Bird-Easy and -Medium.

![](images/e84c304cabb9f5c0a16af6cd5ca0ff2b7fa830edfc043518f3e5402998c441f4.jpg)

<details>
<summary>bar_stacked</summary>

| Model | Butterfly-Medium Correct(%) | Butterfly-Medium Incorrect(%) | Butterfly-Medium Other(%) | Butterfly-Hard Correct(%) | Butterfly-Hard Incorrect(%) | Butterfly-Hard Other(%) |
|---|---|---|---|---|---|---|
| instruct-vicuna13b | 25 | 70 | 10 | 25 | 70 | 10 |
| instruct-vicuna7b | 25 | 70 | 10 | 25 | 70 | 10 |
| instruct-flan5xxl | 25 | 70 | 10 | 25 | 70 | 10 |
| instruct-flan5xl | 25 | 70 | 10 | 25 | 70 | 10 |
| minigpt4-vicuna-13B | 25 | 70 | 10 | 25 | 70 | 10 |
| minigpt4-vicuna-7B | 25 | 70 | 10 | 25 | 70 | 10 |
| blip-flan-xl | 25 | 70 | 10 | 25 | 70 | 10 |
| blip-flan-xl | 25 | 70 | 10 | 25 | 70 | 10 |
| cogvln-chat | 25 | 70 | 10 | 25 | 70 | 10 |
| llava-v1.5-13b | 25 | 70 | 10 | 25 | 70 | 10 |
| llava-v1.5-7b | 25 | 70 | 10 | 25 | 70 | 10 |
| gpt-4o | 25 | 70 | 10 | 25 | 70 | 10 |
| gpt-4v | 25 | 70 | 10 | 25 | 70 | 10 |
Butterfly-Medium Correct(%) / Incorrect(%) / Other(%) / Butterfly-Hard Correct(%) / Incorrect(%) / Other(%)
</details>

(c) Error Analysis for Butterfly-Medium and -Hard.

![](images/df0cf2db17cd2b9732fef410e56317594096d8f22e502c5145b557f2443f8b47.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
| :--- | :--- |
| Rejecting to Answer | 59 |
| Expertise Limitation | 32 |
| Image Clarity | 5 |
| Insufficient data | 3 |
| Option Unavailable | 1 |
</details>

(d) Categories for 250 annotated GPT-4V "Other" responses.   
Figure 8: Analysis of errors for the pretrained VLM responses.

can see that a majority of the “Other” responses belong to the category: Rejecting to Answer (59%), where the GPT-4V states that it is unable to provide an answer, sometimes stating the reason that it cannot answer based on a single image. We also observe a large fraction of Expertise Limitation responses where GPT-4V states that an expert taxonomist is needed to answer the question and its capabilities do not include recognizing or confirming species based on visual data. The next major type of “Other” responses are Insufficient Data, where GPT-4V states that it requires additional data to answer the question, e.g., taxonomic information or habitat information. The other error categories include Image Clarity issues and Option Unavailable (i.e., GPT-4V could not find a suitable option from the list of options provided in the prompt).

# L Comparing Pre-trained VLMs with a Biologically Fine-tuned Model

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Question type</td><td colspan="5">Models</td></tr><tr><td>gpt-4v</td><td>llava v1.5-7b</td><td>cogvlm chat</td><td>CLIP</td><td>BioCLIP</td></tr><tr><td colspan="7">Species Classification</td></tr><tr><td rowspan="2">Fish-10K</td><td>Open</td><td>1.01</td><td>2.32</td><td>0.11</td><td>0.57</td><td>1.24</td></tr><tr><td>MC</td><td>35.91</td><td>40.20</td><td>31.72</td><td>42.45</td><td>50.65</td></tr><tr><td rowspan="2">Bird-10K</td><td>Open</td><td>17.40</td><td>1.45</td><td>0.86</td><td>7.74</td><td>67.12</td></tr><tr><td>MC</td><td>82.58</td><td>50.32</td><td>44.73</td><td>45.78</td><td>93.93</td></tr><tr><td rowspan="2">Butterfly-10K</td><td>Open</td><td>0.04</td><td>0.05</td><td>0.01</td><td>5.33</td><td>15.95</td></tr><tr><td>MC</td><td>28.91</td><td>50.24</td><td>36.45</td><td>45.60</td><td>62.32</td></tr></table>

Table 7: Zero-shot accuracy comparison of VLM baselines (in % ranging from 0 to 100) with BioCLIP for the species classification task. Results are color-coded as Best, and Worst.

We compare the large pretrained VLMs and BioCLIP [74], a state-of-the-art foundation model for species classification. Furthermore, we include the simple CLIP model pretrained with OpenAI weights [82] to evaluate the zero-shot classification performance. Our evaluation was carried out on the Fish-10K, Bird-10K, and Butterfly-10K datasets, and the results are presented in Table 7. We can see that BioCLIP significantly outperforms large pretrained VLMs on the Bird-10K and Butterfly-10K datasets, suggesting that BioCLIP may have been trained on images that are similar to the organisms present in these datasets. However, as noted in the paper, BioCLIP is not trained on fish

images, and hence, the performance of large VLMs is similar to that of BioCLIP on Fish-10K images. We can also see that despite BioCLIP's ability to effectively select the correct scientific name from a smaller set of options in multiple-choice (MC) questions, its performance significantly declines when asked to choose the scientific name from a larger set of open questions. From our observation, it is noteworthy that fine-tuning biological images with scientific names can help improve the overall accuracy of species classification, suggesting directions for future research in this area.

![](images/6ca6c8cde74853f71a3a1377dd6b55754e9d24af82930d9b5c42d8ddfb7be58f.jpg)

<details>
<summary>bar</summary>

| Image Area (Pixels) ×10⁶ | Frequency |
| ------------------------ | --------- |
| 0.0 - 0.5                | 550       |
| 0.5 - 1.0                | 480       |
| 1.0 - 1.5                | 390       |
| 1.5 - 2.0                | 270       |
| 2.0 - 2.5                | 290       |
| 2.5 - 3.0                | 330       |
| 3.0 - 3.5                | 260       |
| 3.5 - 4.0                | 170       |
| 4.0 - 4.5                | 120       |
| 4.5 - 5.0                | 90        |
</details>

(a) Fish-10K

![](images/68fc04e4340b6a220f33a1df36a39af57f8574a41252282007400b62e922513c.jpg)

<details>
<summary>bar</summary>

| Image Area (Pixels) ×10⁴ | Frequency |
| ------------------------ | --------- |
| 5.0                      | 50        |
| 7.5                      | 100       |
| 10.0                     | 150       |
| 12.5                     | 200       |
| 15.0                     | 2200      |
| 17.5                     | 2000      |
| 20.0                     | 800       |
| 22.5                     | 400       |
</details>

(b) Bird-10K

![](images/4e0f42f0f68c3d8313ad5f1ccf753b7aa011580bb70f15e6d2ad2bf510213460.jpg)

<details>
<summary>line</summary>

| Image Area (Pixels) ×10⁶ | GPT-4V | LLaVA-v1.5-7B | LLaVA-v1.5-13B |
| ------------------------- | ------ | ------------- | -------------- |
| 0.3                       | 0.20   | 0.30          | 0.30           |
| 0.9                       | 0.35   | 0.35          | 0.30           |
| 1.4                       | 0.40   | 0.35          | 0.30           |
| 2.0                       | 0.40   | 0.35          | 0.30           |
| 2.5                       | 0.35   | 0.35          | 0.30           |
| 3.1                       | 0.35   | 0.35          | 0.30           |
| 3.6                       | 0.40   | 0.40          | 0.30           |
| 4.2                       | 0.50   | 0.50          | 0.30           |
| 4.7                       | 0.50   | 0.55          | 0.30           |
</details>

(c) Fish-10K

![](images/85f391bc9f8144e4518de7eddbd11ef242967c1ebc6b22247bfb2da1f4b14e18.jpg)

<details>
<summary>line</summary>

| Image Area (Pixels) x10^4 | GPT-4V | LLaVA-v1.5-7B | LLaVA-v1.5-13B |
|---|---|---|---|
| 3.2 | 0.92 | 0.42 | 0.48 |
| 3.8 | 0.80 | 0.36 | 0.52 |
| 8.3 | 0.85 | 0.50 | 0.56 |
| 10.9 | 0.82 | 0.52 | 0.62 |
| 13.5 | 0.83 | 0.54 | 0.58 |
| 16.0 | 0.83 | 0.54 | 0.58 |
| 18.6 | 0.83 | 0.54 | 0.58 |
| 21.2 | 0.82 | 0.52 | 0.56 |
| 23.7 | 0.82 | 0.52 | 0.56 |
</details>

(d) Bird-10K   
Figure 9: Distribution of image resolutions for Fish-10K and Bird-10K are shown in Figures (a) and (b), respectively. The average score over image resolution for the GPT-4V, LLaVA-v1.5-7B, and LLaVA-v1.5-13B models on Fish-10K and Bird-10K are presented in Figures (c) and (d). We conduct the experiment in the context of the Species Classification task with Multiple-Choice (MC) questions.

# M Analyzing Effects of Image Resolution on VLM Performance

To investigate the effect of image resolution on VLM performance, we perform additional experiments summarized in Figure 9 of the attached pdf. In this Figure, we show distribution plots for the Fish-10K and Bird-10K datasets with variations in the image resolutions and their impact on the species classification performance (MC question format) for GPT-4V, LLaVA-1.5-7B, and LLaVA-1.5-13B. All the images of the Butterfly-10K have the exact resolution (500 × 333); hence, they were not included in the experiment. From Figure 9c, it is clear that image resolution is influential on the VLM performance for the Fish-10K dataset since higher resolution helps in recognizing the details of the biological traits and correct species. However, for Figure 9d, the VLM performances do not vary significantly with the image resolution for the Bird-10K dataset. A potential reason is that the bird dataset is a subset of the CUB dataset, and we hypothesize that the pre-trained VLMs may have seen images with resolutions similar to those in the Bird-10K dataset during training, leading to this behavior.

# N Case Studies for Effects of Prompting on VLM Performance

# N.1 No Prompting

1. No Prompting. GPT-4o Correct prediction. Refer to Figure 10.   
2. No Prompting. GPT-4o Incorrect prediction. Refer to Figure 11.   
3. No Prompting. COG-VLM Correct prediction. Refer to Figure 12.   
4. No Prompting. COG-VLM Incorrect prediction. Refer to Figure 13.

# N.2 Contextual Prompting

1. Contextual Prompting. GPT-4o Correct prediction. Refer to Figure 14.   
2. Contextual Prompting. GPT-4o Incorrect prediction. Refer to Figure 15.   
3. Contextual Prompting. LLaVa-13B Correct prediction. Refer to Figure 16.   
4. Contextual Prompting. LLaVa-13B Incorrect prediction. Refer to Figure 17.

# N.3 Dense Caption

1. Dense Captions in Prompts. GPT-4o Correct prediction. Refer to Figure 18.   
2. Dense Captions in Prompts. GPT-4o Incorrect prediction. Refer to Figure 19.   
3. Dense Captions in Prompts. LLaVa-7B Correct prediction. Refer to Figure 20.   
4. Dense Captions in Prompts. LLaVa-7B Incorrect prediction. Refer to Figure 21.

# N.4 Chain-Of-Thought Prompting

1. Chain-Of-Thought Prompting. GPT-4o Correct prediction. Refer to Figure 22.   
2. Chain-Of-Thought Prompting. GPT-4o Incorrect prediction. Refer to Figure 23.   
3. Chain-Of-Thought Prompting. LLaVa-13B Correct prediction. Refer to Figure 24.   
4. Chain-Of-Thought Prompting. LLaVa-13B Incorrect prediction. Refer to Figure 25.

# O Case Studies for Reasoning Hallucination Tests

# O.1 False Confidence Test (FCT)

1. FCT test on Fish dataset. GPT-4o Correct prediction. Refer to Figure 26.   
2. FCT test on Fish dataset. LLaVa-13B Incorrect prediction. Refer to Figure 27.   
3. FCT test on Bird dataset. GPT-4o Correct prediction. Refer to Figure 28.   
4. FCT test on Bird dataset. LLaVa-13B Incorrect prediction. Refer to Figure 29.   
5. FCT test on Butterfly dataset. GPT-4o Correct prediction. Refer to Figure 30.   
6. FCT test on Butterfly dataset. LLaVa-13B Incorrect prediction. Refer to Figure 31.

# O.2 None of The Above (NOTA) Test

1. NOTA test on Fish dataset. GPT-4o Correct prediction. Actual species name is Esox Americanus. Refer to Figure 32.   
2. NOTA test on Fish dataset. LLaVa-13B Incorrect prediction. Actual species name is Esox Americanus. Refer to Figure 33.   
3. NOTA test on Bird dataset. GPT-4o Correct prediction. Actual species name is Corvus Albicollis. Refer to Figure 34.   
4. NOTA test on Bird dataset. Blip-Flan-XL Incorrect prediction. Actual species name is Corvus Albicollis. Refer to Figure 35.   
5. NOTA test on Butterfly dataset. GPT-4o Incorrect prediction. Actual species name is Batesia Hypochlora. Refer to Figure 36.   
6. NOTA test on Butterfly dataset. Blip-Flan-XL Correct prediction. Actual species name is Batesia Hypochlora. Refer to Figure 37.

![](images/60ba376340a118bbd7944a33c06420016d7c4c6a8120bfc21cc821271fb97716.jpg)

<details>
<summary>text_image</summary>

Question: What is the scientific name of the fish
in the image?
Options: A) Cyprinella spiloptera, B) Phenacobius
mirabilis, C) Notropis hudsonius, D) Notropis
buccatus.
Write the answer after writing 'The answer is:':
Answer:
The answer is: B) Phenacobius mirabilis
Ground Truth: Phenacobius mirabilis
</details>

Figure 10: No Prompting. GPT-4o Correct prediction. Section N.1.

![](images/bfffc2a98b63a59d2f85b0428342129477f1ad8d483edc2552672fad6562fce4.jpg)

<details>
<summary>text_image</summary>

Question: What is the scientific name of the fish
in the image?
Options: A) Esox lucius, B) Perca flavescens, C)
Notropis hudsonius, D) Phenacobius mirabilis.
Write the answer after writing 'The answer is:':
Answer:
The answer is: C) Notropis hudsonius
Ground Truth: Phenacobius mirabilis
</details>

Figure 11: No Prompting. GPT-4o Incorrect prediction. Section N.1.

![](images/e8941d6ce5f966a1a775d5ea62c81fb01d0160ed83c23e2f364243acf42afa09.jpg)

<details>
<summary>text_image</summary>

Question: What is the scientific name of the fish
in the image?
Options: A) Notropis hudsonius, B) Notropis
blennius, C) Noturus flavus
Write the answer after writing 'The answer is:':
Answer:
The answer is: Notropis hudsonius
Ground Truth: Notropis hudsonius
</details>

Figure 12: No Prompting. COG-VLM Correct prediction. Section N.1.

![](images/4f05449b8148204ac0a9554a272d191bf1ceb4ddf80ca8e8608a374b2d497133.jpg)

<details>
<summary>text_image</summary>

Question: What is the scientific name of the fish
in the image?
Options: A) Notropis buccatus, B) Lepomis humilis,
C) Notropis dorsalis, D) Esox lucius.
Write the answer after writing 'The answer is:':
Answer:
The scientific name of the fish in the image is
Notropis buccatus.
Ground Truth: Lepomis humilis
</details>

Figure 13: No Prompting. COG-VLM Incorrect prediction. Section N.1.

![](images/32699ae6bbd26c20b6c066e7ae884580d9bcfccb79a50b6007dd6dde698a04f9.jpg)

<details>
<summary>text_image</summary>

Question: Each biological species has a unique
scientific name composed of two parts: the first
for the genus and the second for the species
within that genus.
What is the scientific name of the fish in the
image?
Options: A) Lepomis cyanellus, B) Notropis
blennius, C) Esox lucius, D) Gambusia affinis.
Write the answer after writing 'The answer is:':
Answer:
The answer is: Esox lucius
Ground Truth: Esox lucius
</details>

Figure 14: Contextual Prompting. GPT-4o Correct prediction. Section N.2.

![](images/9d98525bbb34c991063e5de278790997fe0ec67bb459b99682e88033a5cedaaf.jpg)

<details>
<summary>text_image</summary>

Question: Each biological species has a unique
scientific name composed of two parts: the first
for the genus and the second for the species
within that genus.
What is the scientific name of the fish in the
image?
Options: A) Phenacobius mirabilis, B) Notropis
atherinoides, C) Lepomis humilis, D) Lepomis
macrochirus.
Write the answer after writing 'The answer is:':
Answer:
The answer is: Lepomis humilis.
Ground Truth: Lepomis macrochirus
</details>

Figure 15: Contextual Prompting. GPT-4o Incorrect prediction. Section N.2.

![](images/2741c8dbd25d3fbc60ff13a7b9754d75a44b6a16d4df3be8f5f3129e06b2e7a7.jpg)

<details>
<summary>text_image</summary>

Question: Each biological species has a unique
scientific name composed of two parts: the first
for the genus and the second for the species
within that genus.
What is the scientific name of the fish in the
image?
Options: A) Esox americanus, B) Notropis
atherinoides, C) Morone chrysops, D) Cyprinella
spiloptera.
Write the answer after writing 'The answer is:':
Answer:
The answer is: C) Morone chrysops
Ground Truth: Morone chrysops
</details>

Figure 16: Contextual Prompting. LLaVa-13B Correct prediction. Section N.2.

![](images/cf84d8eb7bf904f7f04bfd660242a7103e9203fc91aa90ace5e4452f208c34e1.jpg)

<details>
<summary>text_image</summary>

Question: Each biological species has a unique
scientific name composed of two parts: the first
for the genus and the second for the species
within that genus.
What is the scientific name of the fish in the
image?
Options: A) Cottus bairdii, B) Notropis blennius, C)
Fundulus catenatus, D) Noturus flavus.
Write the answer after writing 'The answer is:':
Answer:
The answer is: D) Noturus flavus
Ground Truth: Fundulus catenatus
</details>

Figure 17: Contextual Prompting. LLaVa-13B Incorrect prediction. Section N.2.

![](images/6ae9cb5d9f859dc99bc5a991de1ced188333e438ad40e0842f9679efc97ad1d4.jpg)

<details>
<summary>natural_image</summary>

Side profile of a small fish with visible fins and eye (no text or symbols)
</details>

wer:   
The answer is: D) Noturus flavus   
Ground Truth: Fundulus catenatus

Dense Caption: The image showcases a small fish with a slender, elongated body shape. The body is covered in yellowish-brown hues, with a gradient that lightens towards the ventral side and deepens dorsally. The scales are distinctly visible, giving a textured appearance to the body. Starting from the head, the fish has a rounded snout and a relatively large, prominent eye with a dark pupil and a lighter surrounding area. The mouth is small and positioned slightly downturned. The dorsal fin is placed towards the rear of the body, exhibiting a rather elongated and low profile with delicate, translucent rays. The anal fin mirrors the dorsal fin but is placed closer to the rear ventral side, displaying similar morphological traits. The pectoral fins are modest in size, almost triangular, and located near the gill covers. The pelvic fins are also small and positioned ventrally, slightly behind the pectoral ones. The caudal fin is fan-shaped and extends symmetrically, marked by an array of thin rays. This fin appears semi-transparent with slight yellowish tinges that match the overall body coloration. Other distinctive features include a lateral line that is faintly visible and runs along the midline of the body. The skin reveals an intricate network of fine lines, which might suggest sensory functions in identifying environmental changes. Given these detailed anatomical descriptions, the fish could potentially be identified as a member of the Poeciliidae family, possibly a species of Gambusia or Heterandria.

Question: Use the above dense-caption and the image to answer the following question. What is the scientific name of the fish in the image? Options: A) Notropis volucellus, B) Notropis percobromus, C) Gambusia affinis, D) Lepomis gibbosus.

# Therefore, the answer is:

# Answer:

The detailed description and the image closely match the morphological characteristics of the fish species Gambusia affinis, which belongs to the Poeciliidae family. Given the options:

A) Notropis volucellus

B) Notropis percobromus

C) Gambusia affinis

D) Lepomis gibbosus

The scientific name of the fish in the image is: C) Gambusia affinis

![](images/fff9e29cd021a788736130c274748c70b6a87120adadc695a7c6c53d5e186a6d.jpg)

<details>
<summary>natural_image</summary>

Close-up of a small fish with visible fins and eye (no text or symbols)
</details>

Ground Truth: Gambusia affinis   
Figure 18: Dense Captions in Prompts. GPT-4o Correct prediction. Section N.3.

![](images/b1e86593ef38e98f23dd7bd8ce318b012959d7ab1149a4622edc962f76b5bc9d.jpg)

Dense Caption: The fish displayed has an elongated, cylindrical body adorned with a pattern of small, well-defined scales that are golden-brown on the dorsal side and a lighter, creamy hue on the ventral side. The head is slightly pointed with a terminal mouth, featuring a prominent, spherical eye located on either side. The dorsal fin, located slightly more than halfway down the back, is relatively small and rounded. The caudal fin at the tail is forked, matching the body's light coloration and adorned with fine, subtle spots.

Pectoral fins are positioned just behind the gill covers and are almost translucent, slightly broader at the base and tapering to a point. The pelvic fins are minute and situated abdominally, while the anal fin mirrors the size and shape of the dorsal fin but is placed closer to the caudal peduncle. A closer inspection of the body reveals faint horizontal striping along the lateral line, a key feature in taxonomic differentiation. The absence of specialized pigment patterns, such as vibrant or contrasting spots, suggests a species adapted for camouflage in its environment. Distinctive anatomical features such as the body coloration pattern, shape, and position of fins, presence of lateral line markings, and general morphology are crucial for identifying the scientific classification of this fish species.

Question: Use the above dense-caption and the image to answer the following question. What is the scientific name of the fish in the image? Options: A) Notropis dorsalis, B) Lepomis megalotis, C) Notropis stramineus, D) Gambusia affinis.

# Therefore, the answer is:

# Answer:

Based on the dense-caption provided and the image, the description matches the characteristics of the fish known as Notropis stramineus.

Therefore, the answer is: C) Notropis stramineus.

![](images/16c79fd1f8195940b5dc2fd5b037f514b23f38b6e24df108d27e836a625be4fe.jpg)  
Ground Truth: Gambusia affinis   
Figure 19: Dense Captions in Prompts. GPT-4o Incorrect prediction. Section N.3.

Dense Caption: The image features a fish with a long, slender body, a pointed head, and a row of small, round fins along its length. The fins are located towards the back of the fish, and their positioning is consistent with the body shape. The fish has a yellowish-brown color, with a pattern of darker stripes running along its body. The fish's overall appearance is typical of a long, slender fish species, such as a snake fish or a similar species from the family Channidae. The presence of the fins and the body shape, along with the color pattern, can help identify the specific species within this family.

Question: Use the above dense-caption and the image to answer the following question. What is the scientific name of the fish in the image?

Options: A) Esox americanus, B) Perca flavescens, C) Esox lucius, D) Fundulus catenatus.

# Therefore, the answer is:

Answer:

A) Esox americanus

![](images/ee967bb6c943cbf0547c8c5cad78eb99002010838c39f22f06350711fc36d858.jpg)

<details>
<summary>natural_image</summary>

Side profile of a small fish with visible fins and body (no text or symbols)
</details>

Ground Truth: Esox americanus   
Figure 20: Dense Captions in Prompts. LLaVa-7B Correct prediction. Section N.3.

![](images/84b610a11bb0abd516d7339335f1b500c13e4c852606cbcafaefe3fd73d29ee9.jpg)

Dense Caption: The image features a small, long, and slender fish with a yellow body and a black stripe running along its length. The fish has pointed head and a slightly curved tail. The fins are small and located towards the back of the fish, with a pair of dorsal fins and a pair of ventral fins. The fish's body is covered in a pattern of small, dark spots, which are a characteristic feature of the species. The fish's overall appearance suggests that it belongs to the family Cyprinidae, which includes species such as the common carp and the goldfish.

Question: Use the above dense-caption and the image to answer the following question. What is the scientific name of the fish in the image?

Options: A) Cottus bairdii, B) Notropis atherinoides, C) Gambusia affinis, D) Notropis buccatus.

# Therefore, the answer is:

Answer:

A) Cottus bairdii

![](images/f0ddaa7ef1b6c0210de6d986a40e041704d83ec27d2d977caa3593d9fdd04ee7.jpg)

<details>
<summary>natural_image</summary>

Side profile of a fish with detailed scales and fins (no text or symbols)
</details>

Ground Truth: Gambusia affinis   
Figure 21: Dense Captions in Prompts. LLaVa-7B Incorrect prediction. Section N.3.

Question: What is the scientific name of the fish in the image?

Options: A) Notropis hudsonius, B) Fundulus catenatus, C) Notropis atherinoides, D) Cottus bairdii.

Please consider the following reasoning to formulate your answer:

Reasoning: To identify the fish in the image, we need to compare its physical characteristics with the descriptions of the four given options.

1. Notropis hudsonius (Spottail Shiner):

\- Small freshwater fish

\- Distinguished by a prominent spot at the base of the tail fin

2. Fundulus catenatus (Northern Studfish):

\- Commonly characterized by vertical bars along its body

\- Known for its elongated body and thick caudal peduncle

3. Notropis atherinoides (Emerald Shiner):

\- A slender, elongated body - Silver-green coloration with a more translucent appearance

4. Cottus bairdii (Mottled Sculpin):

\- Stocky body and broad, flattened head

\- Usually mottled or blotchy in appearance

Looking at the image

\- The fish is small and slender with no prominent spots at the base of the tail fin, which excludes Notropis hudsonius.

\- The fish does not have vertical bars along its body, nor does it have the elongated body typical of Fundulus catenatus.

\- The fish does not have a broad, flattened head or a mottled appearance, ruling out Cottus bairdii.

\- The fish is slender and elongated, fitting the description of Notropis atherinoides.

Based on the physical characteristics visible

![](images/e3a9600079d8b26f96921db8410de723f5d463f4224f267544d0440f9c8e8e78.jpg)

<details>
<summary>natural_image</summary>

Side-view photograph of a fish with brown and beige body, no visible text or symbols
</details>

Answer:

The scientific name of the fish in the image is Notropis atherinoides (Option C), commonly known as Emerald Shiner.

This identification follows the reasoning that the fish has a slender, elongated body with a silver-green coloration and a more translucent appearance. The absence of a prominent spot at the base of the tail fin, vertical bars along the body, and a broad, flattened head with a mottled appearance further support this conclusion.

Ground Truth: Notropis atherinoides

Figure 22: Chain-Of-Thought Prompting. GPT-4o Correct prediction. Section N.4.

Question: What is the scientific name of the fish in the image?

Options: A) Morone chrysops, B) Cottus bairdii, C) Gambusia affinis, D) Fundulus catenatus.

Please consider the following reasoning to formulate your answer:

Reasoning: To identify the fish in the image, let's examine the provided options and characteristics of each species:

A) Morone chrysops: Commonly known as the White Bass, Morone chrysops is a larger fish known for a deep body and silver coloration with black stripes running horizontally. The pictured fish does not exhibit these features.

B) Cottus bairdii: Known as the Mottled Sculpin, Cottus bairdii typically has a more robust and slightly shorter body, with a large head and fan-like pectoral fins, none of which match the fish in the image.

C) Gambusia affinis: Commonly known as the Mosquitofish, Gambusia affinis is a small fish with a more streamlined body and a noticeable upturned mouth. This could match the fish in the image, but the scale and body texture appear different.

D) Fundulus catenatus: Known as the Northern Studfish, Fundulus catenatus typically has a more slender body with a distinctive pattern along its sides, which aligns well with the fish in the image.

The fish in the image appears to have the body characteristics, scale texture, and fin arrangement that match with Fundulus catenatus.

Thus, the correct scientific name of the fish in the image is: D) Fundulus catenatus.

![](images/6bca927b96eea6ae66b456665ceeac4fdf69b3935c0efb505d0fcf8c27cd0f3a.jpg)

<details>
<summary>natural_image</summary>

Side profile of a small fish with golden-brown fins and visible scales (no text or symbols)
</details>

Answer:

Based on the reasoning provided and the visual appearance of the fish in the image, the correct scientific name is:

Option D) Fundulus catenatus.

Ground Truth: Gambusia affinis

Figure 23: Chain-Of-Thought Prompting. GPT-4o Incorrect prediction. Section N.4.   
![](images/3df61cf7727e00396ad7a4f493c1a1b7899aa2b0cf0144afaa349bc94d37640f.jpg)

<table><tr><td>Question: What is the scientific name of the fish in the image?</td><td rowspan="3"></td></tr><tr><td>Options: A) Esox americanus, B) Gambusia affinis, C) Cottus bairdii, D) Notropis blennius.</td></tr><tr><td>Please consider the following reasoning to formulate your answer:</td></tr><tr><td>Reasoning: The scientific name of the fish in the image is A) Esox americanus.</td><td>Answer:
The scientific name of the fish in the image is A) Esox americanus.
Ground Truth: Esox americanus</td></tr></table>

Figure 24: Chain-Of-Thought Prompting. LLaVa-13B Correct prediction. Section N.4.

![](images/d0d9d8cf6c391d29b902c36d4fd1768190818621c8bebedeffdf6ba4c15e1507.jpg)

<details>
<summary>text_image</summary>

Question: What is the scientific name of the fish in the image?
Options: A) Notropis hudsonius, B) Morone chrysops, C) Gambusia affinis, D) Notropis blennius.
Please consider the following reasoning to formulate your answer:
Reasoning: The scientific name of the fish in the image is B) Morone chrysops.
Answer:
B) Morone chrysops
Ground Truth: Gambusia affinis
</details>

Figure 25: Chain-Of-Thought Prompting. LLaVa-13B Incorrect prediction. Section N.4.

<table><tr><td>Question: What is the scientific name of the fish in the image?</td><td></td></tr><tr><td>Options: A) Notropis blennius, B) Notropis dorsalis, C) Cottus bairdii, D) Phenacobius mirabilis.</td><td>Answer:</td></tr><tr><td>Chosen Answer: Notropis dorsalis.</td><td>1) The chosen answer is correct (True/False): False
2) The correct answer: Notropis blennius&quot;</td></tr><tr><td>Please provide: 1) Whether the chosen answer is correct (True/False). 2) The correct answer.</td><td>Ground Truth: Notropis blennius</td></tr></table>

Figure 26: GPT-4o correct prediction. Section O.1.

![](images/c426e724e5d3ade3780af601844f6261ded9b81007746fbb396c26e852b03e23.jpg)

<details>
<summary>text_image</summary>

Question: What is the scientific name of the fish in the image?
Options: A) Lepomis macrochirus, B) Perca flavescens, C) Notropis volucellus, D) Notropis blennius.
Chosen Answer: Notropis volucellus.
Please provide: 1) Whether the chosen answer is correct (True/False). 2) The correct answer .
Answer:
The correct answer is Notropis volucellus. The fish in the image is a long, slender fish with a yellowish color. It is swimming in a body of water, possibly a river or a lake.
Ground Truth: Notropis blennius
</details>

Figure 27: LLaVa-13B incorrect prediction. Section O.1.

Question: What is the scientific name of the bird in the image?

Options: A) Antrostomus carolinensis, B) Sitta carolinensis, C) Certhia americana, D) Chlorocryle americana.

Chosen Answer: Sitta carolinensis

Please provide: 1) Whether the chosen answer is correct (True/False). 2) The correct answer.

Ground Truth: Chloroceryle americana

![](images/b7fce98567cb42c15e7daf91ca7c3690f1f41395c1db61caddff1b67ba2c7430.jpg)

<details>
<summary>natural_image</summary>

A colorful kingfish perched on a branch with green foliage, accompanied by a green checkmark icon (no text or symbols on the bird or branch)
</details>

# Answer:

1) The chosen answer is correct: False.  
2) The correct answer: D) Chloroceryle americana.  
The bird in the image is the Green Kingfisher, which is scientifically named Chloroceryle americana.

Figure 28: GPT-4o Correct prediction. Section O.1.

<table><tr><td>Question: What is the scientific name of the bird in the image? 
Options: A) Contopus sordidulus, B) Vireo olivaceus, C) Corvus albicollis, D) Chloroceryle americana. 
Chosen Answer: Contopus sordidulus 
Please provide: 1) Whether the chosen answer is correct (True/False). 2) The correct answer.</td><td rowspan="2">Answer: 
True. The scientific name of the bird in the image is Contopus sordidulus.</td></tr><tr><td>Ground Truth: Chloroceryle americana</td></tr></table>

Figure 29: LLaVa-13B Incorrect prediction. Section O.1.

![](images/cb3410664fa40f983e691f9f04cb1668389da93e7f773ea0d58a1250c5ca5e76.jpg)

<details>
<summary>text_image</summary>

Question: What is the scientific name of the butterfly in the image?
Options: A) Heliconius heurippa, B) Batesia hypochlora, C) Heliconius melpomene, D) Heliconius wallacei.
Chosen Answer: Heliconius heurippa
Please provide: 1) Whether the chosen answer is correct (True/False). 2) The correct answer .
Ground Truth: Batesia hypochlora
Answer:
1) The chosen answer is incorrect (False).
2) The correct answer is B) Batesia hypochlora.
</details>

Figure 30: GPT-4o Correct prediction. Section O.1.

![](images/0ba79b1a1a2fe3fcdd039f75708cce310f01ffccb8943d07452d02fbdb746f72.jpg)

<details>
<summary>text_image</summary>

Question: What is the scientific name of the butterfly in the image?
Options: A) Pyrrhogyra edocia, B) Heliconius timareta, C) Heliconius doris, D) Batesia hypochlora.
Chosen Answer: Heliconius doris
Please provide: 1) Whether the chosen answer is correct (True/False). 2) The correct answer .
Ground Truth: Batesia hypochlora
Answer:
1. False
2. The correct answer is B) Heliconius timareta.
</details>

Figure 31: LLaVa-13B Incorrect prediction. Section O.1.

<table><tr><td>Question: What is the scientific name of the fish in the image?</td><td></td></tr><tr><td>Options: A) Gambusia affinis, B) Noturus gyrinus, C) Phenacobius mirabilis, D) None of the Above.</td><td>Answer: 
The answer is: D) None of the Above</td></tr><tr><td>Write the answer after writing 'The answer is:'</td><td>Ground Truth: None of the Above</td></tr><tr><td>Options: A) Notropis volucellus, B) Notropis buccatus, C) Morone chrysops, D) None of the Above.</td><td>Answer: The answer is: A) Notropis volucellus</td></tr><tr><td>Write the answer after writing 'The answer is:'</td><td>Ground Truth: None of the Above</td></tr></table>

Figure 32: GPT-4o Correct prediction. Actual species name is Esox Americanus. Section O.2.

![](images/52a754ae3bbba43c8fae998ec3983b0ee07a7f5e0b3f267aca409f30e58e2d91.jpg)  
Figure 33: LLaVa-13B Incorrect prediction. Actual species name is Esox Americanus. Section O.2.

<table><tr><td>Question: What is the scientific name of the bird in the image?</td><td rowspan="2"></td></tr><tr><td>Options: A) Spizella passerina, B) Antrostomus carolinensis, C) Aethia pusilla, D) None of the Above.</td></tr><tr><td>Write the answer after writing &#x27;The answer is:&#x27;.</td><td rowspan="2">Answer: The answer is: D) None of the Above.</td></tr><tr><td>Ground Truth: None of the Above</td></tr></table>

Figure 34: GPT-4o Correct prediction. Actual species name is Corvus Albicollis. Section O.2.

![](images/410d543a122607c115ef25b0e098cf197341b36db20688a1aad59cb8c01b8e93.jpg)

Question: What is the scientific name of the bird in the image?

Options: A) Cardellina pusilla, B) Thryothorus ludovicianus, C) Passer domesticus, D) None of the Above.

Write the answer after writing 'The answer is:'.

Ground Truth: None of the Above

![](images/f310874b9622009da8ed8b41c015d318fc991959c0e4b3471f4fa3b5edaef07f.jpg)

<details>
<summary>natural_image</summary>

Silhouette of a black bird perched on a rocky outcrop against a clear sky (no text or symbols visible)
</details>

Answer:

C

Figure 35: Blip-Flan-XL Incorrect prediction. Actual species name is Corvus Albicollis. Section O.2. 

<table><tr><td>Question: What is the scientific name of the butterfly in the image?</td><td></td></tr><tr><td>Options: A) Temenis laothoe, B) Heliconius doris, C) Heliconius heurippa, D) None of the Above.</td><td></td></tr><tr><td>Write the answer after writing &#x27;The answer is:&#x27;:</td><td></td></tr><tr><td>Ground Truth: None of the Above</td><td>Answer:
The answer is: A) Temenis laothoe</td></tr></table>

Figure 36: GPT-4o Incorrect prediction. Actual species name is Batesia Hypochlora. Section O.2.

<table><tr><td>Question: What is the scientific name of the butterfly in the image?</td><td></td><td></td></tr><tr><td>Options: A) Heliconius erato, B) Heliconius melpomene, C) Heliconius eleuchia, D) None of the Above.</td><td></td><td></td></tr><tr><td>Write the answer after writing &#x27;The answer is:&#x27;.</td><td rowspan="2" colspan="2">Answer: 
D</td></tr><tr><td>Ground Truth: None of the Above</td></tr></table>

Figure 37: Blip-Flan-XL Correct prediction. Actual species name is Batesia Hypochlora. Section O.2.