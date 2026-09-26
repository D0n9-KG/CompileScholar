Trajectory Data Management and Mining: A Survey from Deep Learning to the LLM Era 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.14151v2 [cs.LG] 31 Jan 2026 
 
 

# Trajectory Data Management and Mining: 
 A Survey from Deep Learning to the LLM Era

 
 
 Wei Chen
 
    
 Yuanshao Zhu
 
    
 Yanchuan Chang
 
    
 Kang Luo
 
    
 Haomin Wen
 
    
 Lei Li
 
    
 Qingsong Wen
 
    
 Yanwei Yu
 
    
 Chao Chen
 
    
 Kai Zheng
 
    
 Yunjun Gao
 
    
 Yu Zheng
 
    
 Xiaofang Zhou
 
    
 Yuxuan Liang 
 † † thanks: W.˜Chen, Y.X˜Liang, Y.S˜Zhu, H.M˜Wen, and L. Li are with Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China. E-mail: onedeanxxx@gmail.com. Y.C˜Chang is with The University of Melbourne, Australia. K.˜Luo and Y.J˜Gao are with Zhejiang University, Hangzhou, China. Y.W˜Yu is with Ocean University of China, Qingdao, China. H.M˜Wen is with Beijing Jiaotong University. Q.S˜Wen is with Squirrel AI, USA. C.˜Chen is with Chongqing University, Chongqing, China. K.˜Zheng is with University of Electronic Science and Technology of China, Chengdu, China. W.˜Chen and X.F˜Zhou are with Hong Kong University of Science and Technology, Hongkong SAR. Y.˜Zheng is with JD Intelligent Cities Research, JD Technology, Beijing, China.
Y.X˜Liang is the corresponding author.
 

 Abstract 
 
 Trajectory computing is a pivotal domain encompassing trajectory data management and mining, garnering widespread attention due to its crucial role in various practical applications such as location services, urban traffic, and public safety.
Traditional methods, focusing on simplistic spatio-temporal features, face challenges of complex calculations, limited scalability, and inadequate adaptability to real-world complexities. In this paper, we present a comprehensive review of the development and recent advances in trajectory computing, from deep learning to the more recent large language models. We first define trajectory data and provide a brief overview of widely-used deep learning models. Systematically, we explore deep learning applications in trajectory management (pre-processing, storage, analysis, and visualization) and mining (trajectory-related forecasting, trajectory-related recommendation, trajectory classification, travel time estimation, anomaly detection, and mobility generation).
Furthermore, we discuss emerging research directions and recent advancements in large models (represented by foundation models and large language models) for trajectory computing, which promise to reshape the next generation of trajectory computing. Additionally, we summarize application scenarios, public datasets, and toolkits. Finally, we outline current challenges in trajectory computing research and propose future directions. Relevant papers and open-source resources have been collated and are continuously updated at: https://github.com/yoshall/Awesome-Trajectory-Computing .

 
 
 
 Index Terms:  Trajectory Data Management, Trajectory Data Mining, Deep Learning, Large Models

 
 

## I Introduction 

 
 Since time immemorial, humanity has tirelessly attempted to study the science of mobility, driven by the fundamental laws that emerge from the micro and macro trajectory movements of objects  [ 1 , 2 , 3 ] . The study of trajectories can be traced back as far as the 1960s. Researchers used various marking methods to track the movement trajectories of animals, discovering for the first time that movement behavior patterns possess geographical features and positivity among other patterns  [ 4 ] . By the end of the 20th century, with the rapid development of Global Positioning System (GPS) and Geographic Information System technologies, it became possible to track spatial movement trajectories with long-term, high precision, and high efficiency. This includes volunteer positioning data, GPS-equipped travel trajectories, mobile terminal positioning, and communication records  [ 5 ] . These advancements have fueled the rise of trajectory research as a discipline, with wide-ranging applications in areas such as intelligent transportation, public safety, and business services  [ 6 ] .

 
 
 Fig. 1: Trajectory computing overview. 
 
 
 TABLE I: 
Comparison between this and other related surveys on data formats (i.e., sequence (S), matrix (M), graph (G), and vision (V)), relevant techniques (i.e., traditional methods (TM), deep learning (DL), and large language model LLM), management tasks (i.e., pre-processing (P), storage (S), analytics (A), and visualization (V)), and mining tasks (i.e., forecasting (F), classification (C), recommendation(R), estimation (E), generation (G), and detection (D)). The number of downstream applications and publicly available datasets are also included. Besides, ✓  indicates content is covered, ✗  indicates that content is not covered, and ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} indicates that content is partially covered.
 
 
 
 
 Survey | 
 Year | 
 Formats | 
 Techniques | 
 Management | 
 Mining | 
 #Applications | 
 #Public Datasets | 

 
 S | 
 M | 
 G | 
 V | 
 TM | 
 DL | 
 LLM | 
 P | 
 S | 
 A | 
 V | 
 F | 
 R | 
 C | 
 E | 
 G | 
 D | 

 
 Zheng et al.   [ 7 ] | 
 2015 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 6 | 
 10 | 

 
 Feng et al.   [ 8 ] | 
 2016 | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ✗ | 
 ✗ | 
 ✗ | 
 6 | 
 ✗ | 

 
 Mazimpaka et al.   [ 9 ] | 
 2016 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 13 | 
 ✗ | 

 
 Bian et al.   [ 10 ] | 
 2018 | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 7 | 
 ✗ | 

 
 Bian et al.   [ 11 ] | 
 2019 | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 7 | 
 6 | 

 
 Koolwal et al.   [ 12 ] | 
 2020 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ✗ | 
 ✗ | 
 ✗ | 
 9 | 
 18 | 

 
 Wang et al.   [ 13 ] | 
 2021 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ✓ | 
 ✓ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 13 | 
 20 | 

 
 Luca et al.   [ 14 ] | 
 2021 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 7 | 
 18 | 

 
 Aghababa et al.   [ 15 ] | 
 2022 | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 6 | 

 
 Shaygan et al.   [ 16 ] | 
 2022 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✓ | 
 6 | 
 18 | 

 
 Duarte et al.   [ 17 ] | 
 2023 | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 4 | 

 
 Hu et al.   [ 18 ] | 
 2023 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 4 | 

 
 Graser et al.   [ 19 ] | 
 2024 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ∼ \color[rgb]{0.3555,0.6094,0.8359}{\bm{\sim}} | 
 ✓ | 
 ✓ | 
 8 | 
 17 | 

 
 This Survey | 
 2025 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 15 | 
 38 | 

 
 
 
 However, the effective management and mining of vast records of highly refined trajectories and quantitative spatio-temporal distribution data presents an urgent challenge. Over the past two decades, extensive research has led to a comprehensive framework and theory for trajectory computing, encompassing the entire analysis process: pre-processing ( e . g ., map matching, stay point detection  [ 20 ] ), indexing and retrieval ( e . g ., similarity linking, regional / semantic querying  [ 21 , 13 ] ), pattern mining, and uncertainty modeling  [ 22 , 20 ] . Despite numerous efficient, stage-specific algorithms developed for these loosely coupled processes, three key challenges persist: 1) Lack of uniformity. Problem modeling remains difficult due to the need to combine various tools ( e . g ., rule-based and probabilistic) depending on the scenario. 2) Complexity. The inherent spatio-temporal heterogeneity and auto-correlation in raw trajectory data complicate the capture of intrinsic features via feature engineering or simple expert rules. 3) Adaptability. Traditional technologies often suffer from the curse of dimensionality when processing massive data and struggle to adapt to new application scenarios.

 
 
 In recent years, we have witnessed the rapid rise of deep learning in various fields  [ 23 ] , attributed to its remarkable end-to-end modeling and representation capabilities. Beyond conventional data types ( e . g ., text, images, audio), it has also been extended to general spatio-temporal data with irregular structures [ 24 ] . Trajectory data, characterized by its integrated spatial, temporal, and semantic dimensions, represents a typical case of such data. Accordingly, researchers have leveraged deep learning to reconstruct core components of the trajectory computing framework, including efficient trajectory data management [ 13 ] , effective trajectory data mining [ 14 ] , and various novel downstream applications [ 17 ] . By virtue of diverse neural network architectures and learning paradigms, traditional trajectory-related problems are converted into learning tasks in a seamless manner. Moreover, the integration of prior expert knowledge from spatial statistics, geometry, and geography enables these models to capture complex spatio-temporal trajectory patterns, thereby facilitating the development of innovative applications. In Fig.  1 , we provide an overview of trajectory computing.

 
 
 Related Surveys. While deep learning is increasingly utilized for trajectory computing, existing surveys often have a limited scope, focusing individually on aspects like trajectory management ( e . g ., clustering  [ 10 , 25 ] , similarity  [ 18 ] , privacy  [ 26 ] ) or mining ( e . g ., location prediction  [ 12 , 27 ] , recommendation  [ 28 ] , arrival time estimation  [ 29 , 6 ] ), and only partially address deep learning techniques. Surveys on spatio-temporal data mining  [ 30 , 31 , 32 ] and intelligent traffic  [ 33 , 34 ] are also prevalent but offer limited coverage of trajectory content. Notably, recent surveys on deep learning for trajectory data mining  [ 19 , 14 ] neglect trajectory data management. Furthermore, the integration of nascent large models ( e . g ., LLMs  [ 35 ] ) with trajectory tasks is emerging  [ 36 ] , but lacks a dedicated review. These limitations highlight the urgent need for a comprehensive review, a distinction summarized in Tab.  I .

 
 
 Our Contributions. To address the literature gap, this study offers a systematic and up-to-date review of deep learning and large models for trajectory computing, summarized as follows:

 
 • 
 
 First Systematic Survey. This work provides the first comprehensive review of deep learning advances in trajectory computing, notably highlighting cutting-edge progress with large models, including Foundation Models (FM) and Large Language Models (LLM), offering a thorough overview.

 

 • 
 
 Unified and Structured Taxonomy. We propose a unified taxonomy that structures trajectory computing into three parts: elaborating on diverse trajectory data forms, identifying common tasks in trajectory management and mining, and presenting practical applications across multiple domains. This classification facilitates systematic understanding.

 

 • 
 
 Comprehensive Resource Collection. We initiate the Trajectory Computing Project, an open-sourced, continuously updated repository curating the most comprehensive collection of trajectory datasets, resources, and academic papers on deep learning and large models for trajectory management and mining, serving researchers, engineers, and urban planners.

 

 • 
 
 Future Directions and Opportunities. We analyze the latest advances in large model-enhanced trajectory computing ( e . g ., FM, LLM) amid the ongoing paradigm shift and outline several promising future research directions, providing guidance for the field’s development.

 

 
 
 
 

## II Preliminary 

 

### II-A Definition and Notation 

 
 Definition 1 (Spatio-Temporal Point) 
 
 A spatio-temporal point p p is a unique entity in the form of ( o , t , l , f ) (o,t,l,f) , where represents the access of moving object o o to location l l at timestamp t t under the geographical coordinate system, and comes with an optional record attribute feature f f . 

 
 
 
 Definition 2 (Trajectory) 
 
 A generalized trajectory T T consists of a series of spatial-temporal point sequences ( p 1 , p 2 , … , p n ) (p_{1},p_{2},...,p_{n}) arranged in chronological order, which represents the movement information generated by moving objects in geographical space. 

 
 
 
 Based on the fundamental attributes of spatio-temporal points, trajectory can be extended into various forms. Firstly, with respect to object attributes, we can categorize them into Individual Trajectory , representing quasi-continuous tracking data of individual movements, and Group Trajectory , which denote the movements of a group of individuals during the observation period, typically aggregated into edges/nodes, grids, or a set of Points of Interest (POI) in the mobility graph. Secondly, regarding time attributes, we can derive a spectrum of trajectories ranging from Sparse Trajectory ( e . g ., users’ check-in data during travel) to Dense Trajectory ( e . g ., movement paths of vehicles equipped with GPS tracking systems) based on the dimension of sampling frequency. Thirdly, regarding location attributes, we can generate trajectories, also known as Raw Trajectory , by mapping coordinates to spatial embeddings to discretize the geographic space system. The newly generated sequence of discretized tokens is referred to as Cell Trajectory . Further, trajectories composed of tokens with attribute features are termed as Semantic Trajectory . The relationship of all the above attributes of trajectories is illustrated in Fig.  2 .

 
 
 

### II-B Unique Properties of Trajectory Data 

 
 Trajectory data exhibits unique characteristics that are pivotal for understanding spatial-temporal movements and predicting urban mobility patterns. The following properties underscore the complexity and richness of trajectory data:

 
 
 Fig. 2: Illustration of a trajectory. 
 
 
 
 • 
 
 Spatio-temporal dependencies . Trajectory data inherently exhibits spatio-temporal dependencies as a sequence of spatial locations over time. These dependencies reveal high-level patterns of transfer modes and travel intentions, which are crucial for movement behavior analysis and forecast.

 

 • 
 
 Personalization . As trajectory data is generated by specific individuals or entities, it contains personalized traits reflecting subjects’ preferences and mobility habits. Accurate modeling of these personalized features is imperative for enhancing the precision of micro-level traffic behavior prediction tasks.

 

 • 
 
 Irregularity . Trajectory data often suffers from irregularity due to sampling limitations or data compression. This property results in insufficient supervisory information—e.g., missing detailed path information between points—which poses a significant challenge and can degrade performance in movement prediction tasks.

 

 
 
 
 Each of these properties contributes to the complexity of handling trajectory data, demanding sophisticated modeling techniques to accurately interpret and predict mobility patterns in urban computing contexts.

 
 
 {forest} 
 
 Fig. 3: Taxonomy of this survey with representative trajectory data management and mining works. 
 
 
 

### II-C From Trajectory to Other Formats 

 
 The raw trajectory data can be adaptably formatted for various neural network architectures, enhancing its utility in diverse downstream tasks.

 
 
 Definition 3 (Matrix) 
 
 For a given city, we can divide it into multiple ( N 1 × N 2 N_{1}\times N_{2} ) grids according to the latitude and longitude. Each grid represents a distinct region within the city.
Thus, a trajectory can be represented as a continuous sequence of grid identifiers.
For the origin, destination, and departure time of trajectories, we can construct the Origin-Destination (OD) matrix ℳ ∈ ℝ N 1 × N 2 \mathcal{M}\in\mathbb{R}^{N_{1}\times N_{2}} for any time, where each element represents the inflow and outflow in a particular grid. 

 
 
 
 Definition 4 (Graph) 
 
 A road network for a city can be converted into a directed graph of roads 𝒢 = ( 𝒱 , 𝒜 ) \mathcal{G}=(\mathcal{V},\mathcal{A}) , where 𝒱 \mathcal{V} denotes the roads in the network, and 𝒜 \mathcal{A} represents the connectivity between the road segments.
Consequently, 𝒜 i ​ j = 1 \mathcal{A}_{ij}=1 if and only if road i i and j j can be directly connected.
In this setting, the trajectory can be extracted as a sequence of roads based on the road segments that the trajectory passes through. 

 
 
 
 Definition 5 (Raster) 
 
 A raster image, denoted as ℐ ∈ ℝ H × W × C \mathcal{I}\in\mathbb{R}^{H\times W\times C} , is composed of pixels arranged in a grid. Each pixel possesses specific semantic and positional information, forming the entire image in a predetermined order. Thus, trajectories can naturally be transformed into raster images. A simple and intuitive approach involves treating the entire map as a binary image, where pixels traversed by the trajectory are set to 1, and those not traversed are set to 0. Effective rasterization primarily considers trajectory shape, speed, and direction, which has been extensively studied in the literature  [ 37 ] . 

 
 
 
 Trajectories can also be represented in other vision forms, such as converting them into bird’s-eye view maps. However, this type of data is more closely related to computer vision and receives less attention in the trajectory data mining and management community. Therefore, we do not include this type of purely visual form here.

 
 
 
 

## III Overview and Categorization 

 
 The taxonomy of representative works this survey paper is presented in Figure  3 . Furthermore, we summarize the carefully designed structured content of this paper:

 
 
 
 • 
 
 Deep Learning for Trajectory Data Management. Deep learning is seamlessly integrated into all phases of trajectory management—including pre-processing , efficient storage , high-quality analytics , and clear visualization —to facilitate subsequent mining tasks.

 

 • 
 
 Deep Learning for Trajectory Data Mining. Integrating deep learning enables comprehensive solutions for six major trajectory mining tasks: forecasting , recommendation , classification , travel time estimation , anomaly detection , and mobility generation .

 

 • 
 
 Advances in Large Models for Trajectory Computing. The emergence of FMs and LLMs has similarly transformed the trajectory community, giving rise to new research questions and technologies that we systematically summarize to provide a cutting-edge perspective.

 

 • 
 
 Applications Resources. Deep learning interlinks trajectory computing to generate practical applications in diverse domains, such as personal services , business platforms , and policy guidance . We also provide a comprehensive exploration of publicly available datasets and tools .

 

 
 
 
 

## IV Deep Learning For Trajectory Data Management 

 

### IV-A Pre-Processing 

 
 Recorded trajectories aim to depict the actual movements of objects. However, inherent inaccuracies arise from sampling devices and environmental uncertainties. Pre-processing refines raw data by simplifying redundant and anomalous points, completing missing ones, and employing map matching for calibration, meeting specific needs.

 
 
 Trajectory Simplification. In the presence of sensor noise and the inherent characteristics of high-frequency sampling, Fig  4 illustrates the emergence of nearly identical ”redundant points” ( e . g ., p ​ 5 − p ​ 8 p5-p8 ) and ”drift points” ( e . g ., p ​ 9 p9 ) within a moving trajectory. To mitigate these issues, trajectory simplification methods are designed to remove redundant and anomalous points, effectively reducing data without significantly altering the overall information of trajectory .

 
 
 Early methods relied on human-crafted rules, divided into batch mode (accessing complete data to balance compression and loss) and online mode (accessing only a buffer for real-time compression). Notable batch methods include DP  [ 38 ] and DPTS  [ 39 ] , which compute point importance, while online techniques use sliding windows  [ 40 ] and normal opening windowing  [ 41 ] algorithms to extract feature points. Semantic simplification  [ 42 ] offers an alternative by leveraging road networks to reduce spatial redundancy. To overcome the lack of adaptability in rule-based methods, recent studies utilize deep learning, such as RLTS  [ 43 ] and S3  [ 44 ] , to minimize the error between original and simplified trajectories under length constraints. Furthermore, MARL4TS  [ 45 ] minimizes simplified trajectory length under bounded error conditions, and RL4QDTS  [ 46 ] introduces query accuracy-driven trajectory simplification using multi-agent reinforcement learning to tackle storage costs and expedite query processing.

 
 
 Fig. 4: Pre-Processing example. 
 
 
 Trajectory Recovery. Due to issues with recording devices such as communication latency, GPS localization errors, and privacy issues, the collected data usually covers a substantial number of trajectories with low or missing sample rates [ 13 ] . Take Fig  4 as an example, the raw trajectory lacks any recorded information within the green dashed region ( e . g ., driving in areas with missing signal stations), which may hinder its utilization for downstream applications. To this end, trajectory recovery aims to transform these irregular, low-sampled trajectories into high-sampled ones, effectively supporting mobility computing applications. 

 
 
 Trajectory recovery [ 47 ] , traditionally viewed as spatial series data completion, relies on correlations between adjacent points to impute missing values. Early methods, such as linear and polynomial  [ 48 ] interpolations, were limited in capturing complex dependencies. Recent deep models have advanced sparse trajectory completion. Trajectory recovery is typically categorized based on external information. The first category, free-space trajectory recovery, focuses on modeling intricate transition patterns within trajectory sequences. DHTR  [ 24 ] extended the Seq2Seq framework to Sub-Seq2Seq, employing a deep hybrid model with a Kalman filter for uncertainty reduction. To address sparsity, AttnMove  [ 49 ] proposed an attention-based model integrating historical and periodic patterns, using Bayesian neural networks for uncertainty estimation. PeriodicMove  [ 50 ] introduced a GNN-based attention model that learns complex location transitions from directed graphs constructed from trajectories. TrajBERTT  [ 51 ] and TEIR  [ 52 ] leverage Transformer architectures to refine spatio-temporal modeling, applicable even without explicit geographical coordinates or with variable sampling rates.

 
 
 The second setting, map-constrained recovery, involves utilizing external knowledge, such as road networks, to map segments or points of interest. MTrajRec  [ 53 ] pioneered multi-task learning within Seq2Seq models for this setting, incorporating modules for constraint masking, attention, and attribute enhancement. RNTrajRec  [ 54 ] further introduced a novel spatio-temporal transformer network, GPSFormer, seamlessly integrated with a new road network representation model, GridGNN. Additionally, significant semantic and visual information can enhance recovery. STR  [ 55 ] and VisionTraj  [ 56 ] address this using a heterogeneous information network encoder to model semantic correlations. Beyond this, Traj2Traj  [ 57 ] utilizes a latent factor module to improve recovery efficiency, and PATR  [ 58 ] incorporates a periodic perception module for real logistics platforms.

 
 
 An important related application is the recovery of urban road networks. DeepMG [ 59 ] is a representative approach that discovers and extracts the underlying road network structure from extensive trajectory data. Furthermore, studies like DF-DRUNet  [ 60 ] and DelvMap  [ 61 ] utilize deep neural networks for multimodal fusion of satellite images and trajectory data to improve road network recovery performance.

 
 
 Map-Matching, which converts spatio-temporal points’ latitude and longitude sequences into road segment sequences, facilitating downstream intelligent transportation tasks. As illustrated in Fig  4 , the original trajectory sequence { p 1 , … , p 13 } \{p_{1},...,p_{13}\} can be mapped to road segments { r 1 , … , r 7 } \{r_{1},...,r_{7}\} .

 
 
 Most prior studies on map matching progressed from geometric  [ 62 ] and topological  [ 63 ] approaches to probabilistic statistical algorithms  [ 64 ] . Hidden Markov Models (HMMs), specifically, demonstrate superior robustness to noise and varying sampling rates  [ 65 ] . However, HMM-based methods do not fully utilize abundant trajectory data. DeepMM  [ 66 ] introduced the first deep model using an attention mechanism for accurately mapping sparse and noisy trajectories onto the road network. Addressing the scarcity of well-matched data, a Transformer-based model  [ 67 ] employs transfer learning, pre-training on generated data and fine-tuning with limited labeled samples. Besides, L2MM  [ 68 ] proposes high-frequency and data distribution augmentation to improve the model’s generalization for map matching. Nevertheless, these methods overlook the graph nature of the problem. GraphMM  [ 69 ] incorporates graph neural networks to extract intra-trajectory, inter-trajectory, and trajectory-road correlations. Beyond this, DMM  [ 70 ] , TBMM  [ 71 ] , and FL-AMM  [ 72 ] extend map matching to scenarios involving wireless sensor data by integrating techniques like federated and reinforcement learning.

 
 
 

### IV-B Storage 

 
 To cope with the surge in streaming trajectory data, research in trajectory storage, indexing, and querying remains crucial.

 
 
 Storage Database. Traditional trajectory storage systems focus on the spatio-temporal point level, leading to numerous systems designed for storing and querying trajectory data. The research community has developed specialized management systems  [ 73 , 74 , 75 ] for specific trajectory data types, although the supported query types are often limited. Concurrently, the open-source community has extended existing distributed systems  [ 76 ] for large-scale trajectory storage by introducing custom data formats like LineString and GPX  [ 77 ] .

 
 
 Vector databases, capitalizing on deep representation learning, have become a prevalent database type  [ 78 ] , offering efficient storage, retrieval, and querying capabilities for diverse data. Limited research has focused on trajectory vector databases  [ 79 , 80 ] , with current efforts primarily directed at advancing trajectory representation learning to automatically compress raw trajectories into low-dimensional vector spaces. Since trajectory vector quality is typically assessed by similarity, further details are elaborated in Section  IV-C .

 
 
 Index Query. 
 Trajectory indices are data structures designed to efficiently organize and store trajectories, enabling quick retrieval and analysis . They are vital for optimizing the search performance of various trajectory queries, including similarity search, k k -nearest neighbor query, and similarity join.

 
 
 While TraSS  [ 81 ] recently introduced a novel spatial index XZ for rapid trajectory querying in a key-value database, conventional trajectory indices  [ 82 , 83 , 84 , 85 ] primarily adapt R-trees to hierarchically organize trajectory points or segments, thereby accelerating queries by narrowing search areas. Consequently, deep learning has been extensively applied to enhance data indices concerning query efficiency, resulting in learned indices that model data distribution and access patterns. Existing spatial learned indices  [ 86 , 87 ] predominantly focus on low-dimensional data, such as two-dimensional GPS points. X-FIST  [ 88 ] extended the learned index concept to trajectories by indexing their Minimum Bounding Rectangles (MBRs). For each trajectory, X-FIST first generates a list of sub-trajectories, then constructs two Flood indices on the lower-left and upper-right vertices of the sub-trajectory MBRs.

 
 
 

### IV-C Analytics 

 
 Efficient and precise similarity measurement, as well as clustering analysis, are foundational for various mining tasks involving complex and multi-source trajectory data.

 
 
 TABLE II: Classification of existing trajectory similarity measures.
 m m and n n denote the numbers of points in two trajectories, respectively. i m i_{m} and i n i_{n} denote image sizes. k m k_{m} and k n k_{n} denote the numbers of neighbor nodes on the road network graph. Note that, the dimensionality of trajectory embeddings is a small constant and thus it does not affect time complexity results.
 
 
 
 
 Category | 
 Method | 
 Complexity | 
 Robustness | 
 Components | 

 
 
 
 Heuristic 
 | 
 Point-based | 
 DTW  [ 89 ] | 
 O ⁡ ( m ​ n ) O(mn) | 
 ✗ | 
 - | 

 
 LCSS  [ 89 ] | 
 O ⁡ ( m ​ n ) O(mn) | 
 ✓ | 
 - | 

 
 EDR  [ 89 ] | 
 O ⁡ ( m ​ n ) O(mn) | 
 ✓ | 
 - | 

 
 Shape-based | 
 Fréchet  [ 89 ] | 
 O ⁡ ( m ​ n ) O(mn) | 
 ✗ | 
 - | 

 
 Hausdorff  [ 89 ] | 
 O ⁡ ( m ​ n ) O(mn) | 
 ✗ | 
 - | 

 
 
 
 Learning 
 | 
 
 
 Free Space 
 | 
 SSL-based | 
 t2vec  [ 90 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✓ | 
 RNNs | 

 
 RSTS  [ 91 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✓ | 
 RNNs | 

 
 At2vec  [ 92 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✓ | 
 RNNs | 

 
 Play2vec  [ 93 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✓ | 
 RNNs | 

 
 CL-Tsim  [ 94 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✓ | 
 RNNs | 

 
 TrjSR  [ 95 ] | 
 O ⁡ ( i m + i n ) O(i_{m}+i_{n}) | 
 ✓ | 
 CNNs | 

 
 CSTRM  [ 96 ] | 
 O ⁡ ( m 2 + n 2 ) O(m^{2}+n^{2}) | 
 ✓ | 
 Attention | 

 
 TrajCL  [ 97 ] | 
 O ⁡ ( m 2 + n 2 ) O(m^{2}+n^{2}) | 
 ✓ | 
 Attention | 

 
 TrajRCL  [ 98 ] | 
 O ⁡ ( m 2 + n 2 ) O(m^{2}+n^{2}) | 
 ✓ | 
 Attention | 

 
 SL-based | 
 NEUTRAJ  [ 99 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✓ | 
 RNNs | 

 
 Traj2SimVec  [ 100 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✓ | 
 RNNs | 

 
 TMN  [ 101 ] | 
 O ⁡ ( m ​ n ) O(mn) | 
 ✓ | 
 RNNs | 

 
 T3S  [ 102 ] | 
 O ⁡ ( m 2 + n 2 ) O(m^{2}+n^{2}) | 
 ✓ | 
 Attn.+RNNs | 

 
 TrajGAT  [ 103 ] | 
 O ⁡ ( m ​ k m + n ​ k n ) O(mk_{m}+nk_{n}) | 
 ✓ | 
 GNNs | 

 
 
 
 Road Network 
 | 
 SSL-based | 
 Trembr  [ 104 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✓ | 
 RNNs | 

 
 LightPath  [ 105 ] | 
 O ⁡ ( m 2 + n 2 ) O(m^{2}+n^{2}) | 
 ✓ | 
 Attention | 

 
 SL-based | 
 GTS  [ 106 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✗ | 
 GNNs+RNNs | 

 
 GTS+  [ 106 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✗ | 
 GNNs+RNNs | 

 
 GRLSTM  [ 107 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✗ | 
 GNNs+RNNs | 

 
 SARN  [ 108 ] | 
 O ⁡ ( m + n ) O(m+n) | 
 ✗ | 
 GNNs+RNNs | 

 
 | 
 ST2Vec  [ 109 ] | 
 O ⁡ ( m 2 + n 2 ) O(m^{2}+n^{2}) | 
 ✗ | 
 GNNs+RNNs+Attn. | 

 
 
 
 Similarity Measurement. 
 Trajectory similarity quantifies trajectory resemblance using a set of distance metrics. Traditional heuristic methods  [ 89 ] include point-based (DTW, LCSS, EDR) and shape-based (Fréchet, Hausdorff) distances. Recently, deep learning studies  [ 90 , 91 , 92 ] have enhanced measurement effectiveness and computational efficiency. As shown in Figure  5 , these methods are classified by learning paradigm (Self-Supervised Learning, SSL; or Supervised Learning, SL) and metric space (Free Space or Road Network). We combine these perspectives to detail specific measures for each category, summarizing differences in Table  II .

 
 
 ∘ \circ Free Space : These methods measure the similarity of raw trajectories in free space, often converting them to cell trajectories for processing. They are categorized into SSL-based and SL-based approaches. SSL-based methods learn robust trajectory representations directly from unlabeled trajectories without relying on heuristic rules. t2vec  [ 90 ] pioneered the adoption of SSL by generating similar training pairs through subsampling. RSTS  [ 91 ] , Tedj  [ 110 ] , and At2vec  [ 92 ] enhanced t2vec by incorporating time, multi-granularity, and POI similarity, respectively; these are generally considered reconstruction-based. Recently, CL-Tsim  [ 94 ] introduced a contrastive method to learn discriminative representations by creating positive and negative training samples. TrajCL  [ 97 ] and TrajRCL  [ 98 ] introduced various augmentation and contrastive learning methods to jointly capture spatial and structural similarity. Furthermore, CSTRM  [ 96 ] used a shallow transformer encoder, while TrjSR  [ 95 ] transformed trajectories into images; both aim to capture multi-scale similarity. For practical scenarios, Play2vec  [ 93 ] learns motion trajectory similarity, applying it to sports match analysis. SL-based methods efficiently approximate existing heuristic measurements. NEUTRAJ  [ 99 ] , the pioneer, adapted an RNN with a spatial attention memory module to learn correlations between spatially proximate trajectories. Traj2SimVec  [ 100 ] and  [ 111 ] improved NEUTRAJ’s pre-processing and training efficiency. Subsequently, TMN  [ 101 ] proposed an attention-based matching module to directly learn point-to-point correlation between two trajectories. T3S  [ 102 ] combined LSTM with self-attention to learn representations in Euclidean and grid spaces. Unlike prior methods, TrajGAT  [ 103 ] introduced a graph-based self-attention model, representing trajectories as graphs, and used a quadtree to partition space for capturing fine-grained dependencies.

 
 
 Fig. 5: Different pipeline of trajectory similarity methods. 
 
 
 ∘ \circ Road Networks : These methods measure similarity for trajectories mapped onto road networks, primarily suitable for urban individual or vehicle movements. They are also categorized by learning paradigms. SSL-based methods include Trembr  [ 104 ] and LightPath  [ 105 ] , utilizing RNN and Transformer-based Seq2Seq models, respectively. Constrained by the underlying road network, they encode intrinsic spatial and temporal properties into a latent space. Following this framework, PIM  [ 112 ] reduced reliance on vast training data via a course-wise negative sampling strategy. WSCCL  [ 113 ] incorporated a weakly supervised contrastive learning model to train a temporal path encoder, addressing label acquisition difficulty. For SL-based methods, GTS  [ 106 ] pioneered by defining various road network trajectory similarities, then employing GCN and LSTM to learn embeddings for POI sequences in the graph. Building on this, GRLSTM  [ 107 ] and GTS+  [ 114 ] utilize knowledge graphs and spatio-temporal LSTM with time gates, respectively, to jointly capture trajectory and road network attributes. ST2Vec  [ 109 ] is another study focusing on spatio-temporal similarity on road networks, differing from GTS+ by integrating spatial and temporal features before LSTM input. Furthermore, SARN  [ 108 ] focuses on learning segment embeddings, proposing a contrastive learning-based GCN to capture local and global road segment similarities.

 
 
 Cluster Analysis. 
 Trajectory clustering groups trajectories based on their similarity, ensuring high intra-cluster similarity   [ 115 ] . Traditional methods rely heavily on the choice of similarity measurement, which often leads to variable clustering quality. Current learning-based methods are robust to spatio-temporal scale variations by accounting for latent trajectory features. As depicted in Figure  6 , these methods are generally categorized into multi-stage and end-to-end approaches based on their processing workflows.

 
 
 Fig. 6: Different pipeline of cluster analysis methods. 
 
 
 In general, multi-stage trajectory clustering methods  [ 116 ] typically involve two steps: first, extracting low-dimensional representations by using a sliding window to capture robust movement features, which are then fed into an LSTM-based Autoencoder (AE) to learn fixed-length representations. Subsequently, a traditional clustering algorithm, such as K-means  [ 117 ] , is applied to the learned representations. Trip2Vec  [ 116 ] extracts three trip attributes (time, origins, destinations), inputs them into a fully connected AE to generate trajectory representations, and then uses K-means for clustering. In contrast, end-to-end approaches directly integrate Deep Embedding Clustering (DEC) to simultaneously refine trajectory representations and clustering assignments. For instance, the study  [ 118 ] applies AE-based t-SNE and DEC for aircraft trajectory clustering. DETECT  [ 119 ] uses an LSTM-based AE and environmental context to jointly refine embeddings and clustering. E2DTC  [ 120 ] , an RNN-based AE method, introduces a dedicated triplet loss for clustering.

 
 
 

### IV-D Visualization 

 
 To enable real-time visualization and interactive analysis of extensive mobility data, traditional trajectory visualization methods rely on temporal and spatial dimensions of geo-points. Techniques like density maps, heatmaps, and spatio-temporal cubes  [ 121 ] are used for macroscopic analysis.

 
 
 However, visualizing large raw trajectories can result in information redundancy, as addressed by deep clustering and simplification methods in Sec  IV-A and  IV-C . Utilizing these methods, researchers can obtain grouped trajectories, allowing for a detailed examination of various trajectory movement patterns. For instance,   [ 122 ] presents an interactive system called Surveillance that uses LSTM models and network embedding to detect and visualize urban congestion conditions. Similarly,  [ 123 ] uses an iterative sampling scheme for OD flows, creating meaningful visual encodings. Deep learning’s capability to extract hidden knowledge from data without prior knowledge is also leveraged to assist in individual trajectory visual exploration.  [ 124 ] introduces DeepHL, employing attention-based neural networks for automatic detection and visualization of meaningful trajectory segments. In DSAE  [ 125 ] , a deep sparse autoencoder extracts hidden features, mapping them to the RGB color space to visualize driving behavior. Later,  [ 126 ] uses GIS map integration to enhance anomaly visualization. More analyses of trajectory visualization are discussed in  [ 127 ] .

 
 
 
 

## V Deep Learning for Trajectory Data Mining 

 

### V-A Trajectory-related Forecasting 

 
 As shown in Figure  7 , in trajectory data mining, forecasting tasks aim to accurately predict future movements of individuals (i.e., Location Forecasting) or crowds (Flow Forecasting) based on historical data [ 14 ] .

 
 
 Fig. 7: Forecasting tasks schematic and influencing factors. 
 
 
 Location Forecasting. This task aims to predict an individual’s subsequent location using their historical movement data. It requires modeling spatial ( e . g ., location), temporal ( e . g ., day of week), external ( e . g ., weather), and personalized ( e . g ., periodic visits) patterns. Formally, given a person’s historical movement data, the task predicts the most likely next spatial point or region   [ 14 ] . This is formulated either as a classification problem (next location is one of predefined regions) or a regression problem (predicting exact geographical coordinates). The challenge lies in accurately modeling the complexity of human movement and factors influencing location selection, such as time, personal preferences, and social behavior.

 
 
 Intuitively, classification models learn from a historical sequence of visited locations, integrating various learning blocks (CNN [ 128 ] , RNN [ 129 ] , ST-RNN [ 130 ] , Embedding [ 129 ] , Attention mechanisms [ 131 ] , and GNNs  [ 132 ] ) to capture the transfer probability distribution of all possible locations; the highest probability indicates the most likely next visit. For instance, DeepMove  [ 131 ] , an attentional recurrent neural network, uses two attention mechanisms to capture multi-level periodicity and utilizes GRUs for trajectory processing and prediction. Flashback  [ 129 ] , a general RNN architecture, addresses user movement sparsity by using spatio-temporal context within the RNN to identify hidden states with high predictive power. Unlike classification-based methods, regression-based approaches forecast continuous and exact values representing the next spatial point. For example, Song et al.   [ 133 ] introduced a multi-task deep learning framework with stacked LSTM layers to simultaneously predict future traffic patterns and positions. Considering the influence of multiple contexts, MobTCast  [ 134 ] uses the transformer architecture as a spatio-temporal feature extractor to process both temporal and semantic contexts.

 
 
 Additionally, recent studies have extended location forecasting into two variants: next POI recommendation and incomplete path prediction. The former primarily addresses cold start scenarios and user preference recommendations. The latter is applied in contexts like food delivery and logistics, predicting a series of locations based on a worker’s current incomplete route tasks, which adds complexity. For detailed discussion, refer to  [ 135 ] and  [ 6 ] .

 
 
 Traffic Forecasting. 
Traffic forecasting predicts the movement and density of traffic within a given area over time. This task analyzes historical mobility data, congestion patterns, and flow trends to predict the number of entities that will congregate in an area at a future time. Formally, traffic forecasting is typically treated as a time series forecasting problem, aiming to predict future flows based on past observations   [ 136 , 137 ] . The complexity arises from the dynamic and unpredictable nature of traffic flow, influenced by factors such as time of day, weather, accidents, and construction.

 
 
 In practice, trajectories are first transformed into matrices based on time and corresponding regions (as outlined in Sec. II-C ). Forecasts can then be made using classical time series models like Autoregressive Moving Average and Vector Autoregressive  [ 138 ] . However, these methods struggle with spatial dependencies and various additional features ( e . g ., weather), limiting their performance. Since traffic flows are matrix-formatted, CNNs can effectively capture their local and global spatio-temporal dependencies. Additionally, RNN models like LSTM can model complex temporal dynamics. Thus, deep learning methods efficiently capture patterns in the temporal evolution of crowd flows. ST-ResNet [ 139 ] pioneered traffic flow prediction using deep neural networks, utilizing residual CNN units to capture temporal closeness, trends, and periodic patterns. The output from each attribute type is aggregated with external factors to predict the flow. Follow-up studies include DMVST-Net  [ 140 ] , which investigates multi-view spatio-temporal patterns. STRCN  [ 141 ] combines CNN and LSTM for spatio-temporal modeling and assigns weights to different branches. Periodic-CRN  [ 142 ] focuses on capturing repeated periodic patterns. Furthermore, numerous deep learning-based methods have emerged, broadly categorized as ConvLSTM-based  [ 143 ] , multi-task-based  [ 144 ] , and attention-based methods  [ 145 ] . Recent studies also increasingly utilize spatio-temporal graphs to model traffic flows  [ 146 , 147 , 148 ] , leveraging the advantages of GNNs.

 
 
 

### V-B Trajectory-related Recommendation 

 
 As illustrated in Figure  8 , critical tasks within location-based service systems (LBSN) encompass travel and friend recommendations. Travel recommendation aims to provide suitable routes based on user constraints and preferences. Friend recommendation infers social relationships and suggests potential acquaintances based on user mobility patterns and behaviors. Analyzing historical trajectory data, social connections, and potential needs is key to delivering precise recommendations that enhance user travel and social interactions.

 
 
 Fig. 8: Recommendation of location-based social networks. 
 
 
 Travel Recommendation. The primary objective of travel recommendation is to generate POI sequences tailored to specific traveler constraints, such as duration, origin, destination, and visitation targets . Traditionally termed travel query and planning, this domain seeks to maximize user satisfaction by solving the orienteering problem. The core methodology employs heuristics to integrate POIs with trajectories. These approaches generally fall into five categories: search-based, probability-based, biomimetic-based, clustering-based, and constraint-based methods  [ 149 ] , many of which originate from robotic pathfinding.

 
 
 The proliferation of ubiquitous tracking devices has facilitated data-driven, personalized travel recommendations. Early hybrid approaches utilize neural networks to approximate A* search cost functions; notably, HRNR  [ 150 ] models complex traffic data for efficient route planning. Deep learning has inspired diverse architectures: sequential models  [ 151 , 152 , 153 , 154 , 155 ] employ RNNs to extract POI features under diversity constraints. For instance, LDFeRR  [ 151 ] combines GRUs with attention mechanisms to optimize fuel efficiency in long-distance travel. Conversely, graph-based methods  [ 156 , 157 , 158 ] capture spatial dependencies; GraphTrip  [ 156 ] leverages spatio-temporal graphs and transfer learning to mitigate data sparsity. Multi-modal approaches  [ 159 , 160 ] enhance performance by integrating text and imagery, with [ 159 ] pioneering the use of Google Street View data. Furthermore, reinforcement learning frameworks  [ 161 , 162 ] frame urban routing as a decision-making process, as demonstrated by   [ 161 ] adaptive deep reinforcement learning method.

 
 
 Friend Recommendation. Friend recommendation in location-based social networks (LBSN) enhances user engagement by capitalizing on the correlation between frequent co-visitation and social formation  [ 163 ] . Formally, literature typically frames this task as social relationship inference, estimating friendship probability based on historical check-in trajectories and existing social networks. Certain studies extend this objective to top-k friend recommendation. 

 
 
 Early research relied on location co-occurrence or interest similarity. Brown  et al.   [ 164 ] established the correlation between geographic proximity and social ties, while Chu  et al.   [ 165 ] incorporated dwell time to analyze location similarity. Yu  et al.   [ 166 ] utilized random walks on heterogeneous information networks merging GPS data to estimate link relevance. With the advent of graph neural networks (GNNs)  [ 167 , 168 , 169 ] , research has pivoted toward nonlinear representations of LBSN. LBSN2Vec  [ 170 ] and its extension LBSN2Vec++  [ 171 ] employ hypergraphs to integrate user, temporal, and spatial semantics for automated feature representation. Similarly, MVMN  [ 172 ] proposes a multi-view matching network integrating diverse factors. Addressing data sparsity, TSCI  [ 173 ] leverages VAE latent variables to estimate friendship trajectory distributions. Recently, SRINet  [ 174 ] introduced a GNN framework to mitigate noise in mobility data, while FDPL  [ 175 ] frames recommendation as a ranking task, utilizing deep pairwise learning based on Bayesian personalized ranking.

 
 
 

### V-C Trajectory Classification 

 
 Trajectory classification aims to distinguish trajectory characteristics by learning latent patterns from historical data to categorize new trajectories. Categories encompass transportation modes, animal types, and specific users  [ 176 ] . Early methodology was predominantly heuristic; for instance, TraClass  [ 177 ] utilizes adaptive spatial grids, while subsequent work  [ 178 ] employs uneven spatio-temporal gridding based on predefined thresholds. Other studies construct recognition models ( e . g ., SVM, Random Forests) using features such as distance and heading change rates  [ 179 , 180 ] . To better capture complex spatio-temporal and semantic dependencies, recent deep learning advancements prioritize Travel Mode Identification (TMI) and Trajectory User Linking (TUL).

 
 
 Travel Mode Identification. TMI focuses on categorizing movement patterns from raw trajectories, accounting for mode transitions within a single journey ( e . g ., cycling followed by transit). Formally, given an individual’s historical trajectory records, TMI task aims to identify the potential movement modes encompassed in the journey   [ 181 ] . This problem is typically formulated as a multi-class or multi-label classification task. Primary challenges arise from irregular sampling intervals and inherent spatio-temporal noise.

 
 
 Existing methods achieve high accuracy by employing AE  [ 182 , 183 ] , RNN  [ 184 , 185 , 186 , 187 ] , CNN  [ 188 , 189 ] , Attention  [ 190 , 191 ] , and GNN  [ 192 ] architectures. Early sequence modeling approaches, such as TrajectoryNet  [ 184 ] and bidirectional LSTM classifiers  [ 185 ] , utilize segment information and data normalization. Addressing the limitations of discrete-time updates, ST-GRU  [ 186 ] incorporates segment-wise gating, while TrajODE  [ 187 ] leverages neural ordinary differential equations to model continuous temporal dynamics. Alternatively, TraClets  [ 188 ] converts trajectories into raster images for CNN-based classification. Recently, TrajFormer  [ 190 ] adapts the transformer architecture with squashing functions to balance efficiency and accuracy.

 
 
 TABLE III: List of the selected papers tackling classification task. 
 
 
 
 Task | 
 Method | 
 Year | 
 Components | 
 Evaluation | 
 Dataset | 

 
 
 
 TMI 
 | 
 
 
 
 TrajectoryNet  [ 184 ] 
   Code | 
 2017 | 
 GRU | 
 
 
 
 Accuracy, 
 
 CE Loss, F1 Score 
 | 
 Geolife | 

 
 
 
 
 ST-GRU  [ 186 ] 
 | 
 2019 | 
 GRU | 
 Accuracy | 
 
 
 
 Geolife, 
 
 SH Taxi, 
 
 Synthetic 
 | 

 
 
 
 
 TrajODE  [ 187 ] 
 | 
 2021 | 
 RNN, ODE | 
 Accuracy | 
 
 
 
 Geolife, 
 
 Grab-Posisi 
 | 

 
 
 
 
 TraClets  [ 188 ]   Code 
 | 
 2022 | 
 CNN, FC | 
 Accuracy | 
 
 
 
 GeoLife, 
 
 Hurricane, 
 
 Animals 
 | 

 
 
 
 
 TrajFormer  [ 190 ]   Code 
 | 
 2022 | 
 Transformer | 
 
 
 
 Accuracy, 
 
 FLOPs 
 | 
 
 
 
 Geolife, 
 
 Grab-Posisi 
 | 

 
 
 
 TUL 
 | 
 
 
 
 TULER  [ 193 ]   Code 
 | 
 2017 | 
 RNNs | 
 
 
 
 Acc@k,
Macro-F1 
 | 
 
 
 
 Gowalla, 
 
 Brightkite 
 | 

 
 
 
 
 TULVAE  [ 194 ]   Code 
 | 
 2018 | 
 LSTM, VAEs | 
 
 
 
 Acc@k, Macro-P, 
 
 Macro-R, Macro-F1 
 | 
 
 
 
 Gowalla, 
 
 Brightkite, 
 
 Foursquare 
 | 

 
 
 
 
 DeepTUL  [ 195 ]   Code 
 | 
 2020 | 
 
 
 
 RNN 
 
 Attention 
 | 
 
 
 
 Acc@k, Macro-P, 
 
 Macro-R, Macro-F1 
 | 
 
 
 
 Foursquare, 
 
 WLAN 
 | 

 
 
 
 
 MainTUL  [ 196 ]   Code 
 | 
 2022 | 
 
 
 
 LSTM 
 
 Attention 
 | 
 
 
 
 Acc@k, Macro-P, 
 
 Macro-R, Macro-F1 
 | 
 
 
 
 Foursquare, 
 
 Weeplaces 
 | 

 
 
 
 
 AttnTUL  [ 197 ]   Code 
 | 
 2023 | 
 
 
 
 FC, GNN, 
 
 Attention 
 | 
 
 
 
 Acc@k, Macro-P, 
 
 Macro-R, Macro-F1 
 | 
 
 
 
 Private Car, 
 
 Gowalla, 
 
 Geolife 
 | 

 
 
 
 Trajectory-User Linking. TUL associates anonymous semantic trajectories with specific users, facilitating applications such as epidemic tracking and personalized services. Formally, given an anonymous trajectory, TUL task aims to identify the actual user in the database corresponding to that journey   [ 196 ] . Key challenges include handling data sparsity and interpreting the hierarchical semantic structures inherent in human mobility.

 
 
 TULER  [ 193 ] pioneered this domain by using RNNs and word embeddings to link POI sequences to users, though it struggles with hierarchical semantics. To address this, TULVAE  [ 194 ] incorporates variational autoencoders to manage sparsity and learn hierarchical features. DeepTUL  [ 195 ] further mitigates sparsity using attention mechanisms to capture multi-periodic patterns. More recent approaches include MainTUL  [ 196 ] , which utilizes mutual distillation learning for temporal dependencies, AdattTUL  [ 197 ] employing GANs, and SML-TUL  [ 198 ] , which leverages contrastive learning under spatio-temporal constraints.

 
 
 Other perspectives. Research also extends to semi-supervised and unsupervised settings. SECA  [ 182 ] and proxy-label methods  [ 199 ] integrate labeled and unlabeled data, while SSFL  [ 200 ] adapts this for federated learning. For unsupervised TMI, DeepCAE  [ 201 ] combines convolutional autoencoders with clustering. Approaches for limited or unlabeled data include wavelet transformations  [ 202 ] , map-matching integration  [ 203 ] , and graph-based modeling in S2TUL  [ 204 ] . Furthermore, DPLink  [ 205 ] and EgoMUIL  [ 206 ] address cross-platform heterogeneity, while AttnTUL  [ 207 ] employs hierarchical attention to handle varying trajectory densities.

 
 
 

### V-D Travel Time Estimation 

 
 Travel Time Estimation (TTE), or Estimated Time of Arrival (ETA), is vital for location-based services, facilitating efficient trip management and route optimization  [ 208 , 209 ] . While traditional methods relying on origin-destination points often overlook path selection and road conditions, modern approaches integrate complex spatio-temporal data to enhance accuracy. These are primarily categorized into trajectory-based and road-based methods, as summarized in Table  IV . 

 
 
 TABLE IV: List of the selected papers tackling estimation task. 
 
 
 
 Task | 
 Method | 
 Year | 
 
 
 
 Components 
 
 ( Focus ) 
 | 
 Dataset | 

 
 
 
 Trajectory 
 | 
 
 
 
 DeepTTE  [ 209 ] 
   
 
 
 Code 
 | 
 2018 | 
 LSTM | 
 Geolife | 

 
 
 
 
 DeepTravel  [ 210 ] 
 | 
 2018 | 
 BiLSTM | 
 
 
 
 Porto, Shanghai Taxi 
 | 

 
 
 
 
 MURAT  [ 211 ] 
   
 
 
 Code 
 | 
 2018 | 
 
 
 
 Graph 
 
 Embedding 
 | 
 
 
 
 NYC-Trip, BJS-Pickup 
 | 

 
 
 
 
 TTPNet  [ 190 ] 
   
 
 
 Code 
 | 
 2022 | 
 RNN, GNN | 
 
 
 
 Beijing Taxi, Shanghai Taxi 
 | 

 
 
 
 Road 
 | 
 
 
 
 WDR  [ 212 ] 
 | 
 2018 | 
 LSTM, FC | 
 
 
 
 DiDi Beijing 
 | 

 
 
 
 
 DeepIST  [ 213 ] 
   
 
 
 Code 
 | 
 2019 | 
 PathCNN | 
 
 
 
 Porto, 
 
 Chengdu 
 | 

 
 
 
 
 ConSTGAT  [ 214 ] 
 | 
 2020 | 
 GAT | 
 
 
 
 Taiyuan, Hefei, HuiZhou 
 | 

 
 
 
 
 HetETA  [ 215 ] 
   
 
 
 Code 
 | 
 2020 | 
 GCN | 
 
 
 
 DiDi Shengyang 
 | 

 
 
 
 
 CompactETA  [ 216 ] 
 | 
 2020 | 
 
 
 
 LSTM, FC 
 | 
 
 
 
 Beijing, Suzhou, Shengyang 
 | 

 
 
 
 Others 
 | 
 
 
 
 ER-TTE  [ 217 ] 
 | 
 2018 | 
 En route | 
 
 
 
 Taiyuan, Hefei, HuiZhou 
 | 

 
 
 
 
 CatETA  [ 218 ] 
 | 
 2022 | 
 
 
 
 Classification 
 | 
 
 
 
 DiDi [Shenzhen, Chengdu] 
 | 

 
 
 
 
 PP-TPU  [ 219 ] 
 | 
 2021 | 
 
 
 
 Uncertainty 
 
 Privacy 
 | 
 
 
 
 Creteil, San Francisco 
 | 

 
 
 
 
 ProbTTE  [ 220 ] 
 | 
 2023 | 
 
 
 
 Classification 
 
 Uncertainty 
 | 
 
 
 
 DiDi [Beijing, Shanghai] 
 | 

 
 
 
 
 DeepTTDE  [ 221 ] 
 | 
 2023 | 
 
 
 
 Travel time 
 
 distributions 
 | 
 
 
 
 DiDi [Chengdu, Shenzhen] 
 | 

 
 
 
 Trajectory-based Estimation. Leveraging trajectory sequences defined in Sec.  II-A , these methods predict travel time by analyzing GPS points. Wang  et al. utilized raw GPS data via an error feedback recurrent convolutional neural network (eRCNN)  [ 222 ] , while DeepTTE incorporated geo-convolution to capture spatial correlations  [ 209 ] . Subsequent studies integrate grid-mapped trajectories with auxiliary data ( e . g ., traffic, weather) using multi-task learning and graph neural networks to enhance contextual analysis  [ 210 , 211 , 223 , 224 ] . Despite their utility, these methods suffer from reliance on high-frequency GPS data and the unrealistic assumption of known future locations. Consequently, road-based TTE has emerged as a robust alternative, mitigating sensitivity to sampling rates and positioning accuracy  [ 218 , 212 ] .

 
 
 Road-Based Estimation. By defining trips as road sequences, these approaches model inter-road correlations to support diverse routing, thereby reducing trajectory dependence. WDR  [ 212 ] pioneered this using a hybrid regression framework to integrate comprehensive travel features. Subsequent research has incorporated metric learning and personalized driving behaviors to refine model sensitivity  [ 225 , 226 ] , while PathCNN introduced sub-path images for spatio-temporal analysis  [ 213 ] . Recently, the field has shifted towards complex networked representations, utilizing heterogeneous information graphs and spatio-temporal attention mechanisms to capture dynamic road contexts  [ 215 , 214 ] . This evolution highlights the efficacy of advanced graph-based techniques in addressing road-based TTE challenges  [ 216 , 227 , 228 , 229 ] .

 
 
 Other perspectives: Beyond standard classifications, novel frameworks address TTE through alternative paradigms. Ye  et al. and Liu  et al. reframe TTE as a multi-classification task, mitigating long-tail effects by categorizing time spans based on trip distributions  [ 218 , 220 ] . Fang  et al. proposed en route TTE (ER-TTE), leveraging observed behaviors to adaptively refine predictions  [ 217 , 230 ] . Furthermore, recent scholarship aims for robust, versatile solutions by exploring specialized domains, including uncertainty quantification  [ 219 , 231 ] , cross-area generalization  [ 231 ] , and travel time distribution estimation  [ 232 , 221 ] .

 
 
 

### V-E Anomaly Detection 

 
 Trajectory anomaly detection aims to identify abnormal movement of objects , facilitating applications such as ride-hailing fraud detection, traffic monitoring, and trajectory cleaning. Early approaches, like TRAOD  [ 233 ] , relied on hand-crafted distance-and-density rules to identify outliers. Methods are categorized into offline detection, which requires complete trajectories, and online detection, which supports “on-the-fly” processing. Notably, online methods offer superior flexibility by accommodating both real-time streams and progressively generated full trajectories.

 
 
 Offline Detection: ATD-RNN  [ 234 ] utilizes an RNN with a fully connected layer to predict Euclidean anomalies via supervised learning. Addressing data scarcity, IGMM-GAN  [ 235 ] employs an unsupervised CNN-based bidirectional GAN, where learned embeddings form a multi-modal Gaussian distribution, i . e . forming multiple clusters. Anomaly scores are derived from the distance between test trajectories and cluster centers. TripSafe  [ 236 ] targets ride-hailing anomalies by analyzing features like stopping duration, using dual VAEs to learn representations in both Euclidean and road network spaces. Furthermore, ATROM  [ 237 ] applies variational Bayesian methods guided by probability measure rules to recognize anomalies in open-world scenarios.

 
 
 Online Detection: DB-TOD  [ 238 ] targets road networks by using reinforcement learning to model segment transition probabilities, formulating detection as a sequential decision process. RL4OASD  [ 239 ] improves upon this by refining feature generation and introducing local rewards to enforce label continuity. Conversely, GM-VSAE  [ 240 ] operates in Euclidean space, adapting an RNN-based VAE to learn latent probability distributions and detect anomalies via generation likelihoods, thereby enhancing online efficiency. Building on this, DeepTEA [ 241 ] further incorporates the temporal dimension into the detection framework.

 
 
 

### V-F Mobility Generation 

 
 Trajectory data applications are frequently hindered by data scarcity, privacy concerns, and authorization limits. To address these constraints, trajectory generation synthesizes realistic data that preserves privacy while supporting research utility  [ 14 ] . As illustrated in Figure  9 , trajectory generation employs deep learning to mimic complex movement patterns, encapsulating statistical, spatial, and temporal nuances to align generated outputs with real-world distributions. Synthesis approaches are categorized by scale: macro-dynamics and micro-dynamics. Macro-dynamics model aggregate mobility trends, such as inter-regional population flows, emphasizing large-scale patterns over individual movements  [ 242 , 14 ] . Conversely, micro-dynamics focus on individual-level granularity—including specific routes, speeds, and stops—which is essential for high-resolution applications like location-based services and behavioral analysis.

 
 
 Fig. 9: Macro and micro trajectory generation examples. 
 
 
 Macro-dynamics. Macro-level generation captures comprehensive mobility flows, traditionally utilizing statistical simulations and physics-based approaches like gravity and radiation models  [ 243 ] . While foundational, these methods often oversimplify complex human dynamics. Recent deep learning advancements have transformed flow generation, employing architectures such as FC, CNN, RNN, and GANs to decode intricate spatiotemporal dependencies  [ 244 , 245 , 246 , 247 , 248 ] . These data-driven methods significantly enhance flow fidelity and adaptability. Furthermore, graph-based models, such as the spatial interaction GCN proposed by Yao  et al. , leverage local spatial networks to refine geographic unit representations  [ 140 ] .

 
 
 Micro-dynamics. Micro-level generation replicates individual mobility granularity, including location sequences, dwell times, and routes. Early approaches treated this as sequential next-location prediction, heavily relying on historical data  [ 249 , 250 , 131 ] , or utilized trajectory mixture models, which often entailed high computational overhead  [ 251 ] . The integration of Generative Adversarial Networks (GANs) introduced a paradigm shift; for instance, Ouyang  et al. utilized grid-mapped data to synthesize realistic paths  [ 252 ] , though trade-offs between grid resolution and accuracy persist  [ 253 , 254 ] . Reinforcement learning further advanced the field by modeling trajectory generation as sequential decision-making processes  [ 255 , 256 , 257 ] . Other methods transform trajectories into images, albeit with increased computational complexity  [ 258 , 253 ] . Most recently, denoising diffusion probabilistic models, such as DiffTraj  [ 259 ] and Diff-RNTraj  [ 260 ] , have emerged, modeling generation via particle diffusion to achieve high-fidelity path creation.

 
 
 
 

## VI Recent Advances in Large Models for Trajectory Computing 

 
 The rapid evolution of generative artificial intelligence, particularly large models demonstrating robust cross-modal reasoning and generalization, is revolutionizing trajectory computing. Traditional mining methods, typically reliant on task-specific deep learning models, suffer from poor generalizability and limited reasoning capabilities. Conversely, the emergence of foundation models (FM) and large language models (LLMs) offers a novel paradigm, facilitating more general, interpretable, and knowledge-integrated trajectory intelligence.

 
 

### VI-A Foundation Model in Trajectory Computing 

 
 Since 2024, research has pivoted from city-specific architectures to unified foundation models capable of multimodal and cross-domain generalization  [ 261 ] . By employing Transformer and its variants as backbone networks to train on massive trajectory datasets from scratch, these models address traditional limitations regarding task specificity, regional dependence, data heterogeneity, and privacy preservation. We categorize existing works by three key challenges.

 
 
 Geographic Scalability. As conventional models use non-transferable spatial representations (grids/road IDs) tied to single cities, four transfer learning strategies have emerged. i i ) Region-agnostic Minimality: UniTraj  [ 262 ] utilizes the WorldTrace dataset (70 countries) via pure spatio-temporal points, excluding road networks and POIs. i ​ i ii ) Unified Semantic Space: MoveGPT  [ 263 ] embeds geography, POIs, and popularity, while UniMove  [ 264 ] adopts a feature-based trajectory-location dual-tower architecture. i ​ i ​ i iii ) Unified Spatial Rendering: VLMLocPredictor leverages VLM visual reasoning by rendering trajectories as images, enabling cross-city transfer. i ​ v iv ) Transferable Location Encoding: TrajFM  [ 265 ] incorporates POI modalities and learnable spatio-temporal rotary embeddings for vehicle trajectories.

 
 
 Handling Heterogeneity. This involves managing heterogeneity in tasks (prediction, classification) and data (mixed patterns, noise). i i ) Task Heterogeneity: TrajFM unifies tasks via “trajectory mask-and-restore,” while BIGCity  [ 266 ] uses spatio-temporal prompts to guide a frozen model across analyses without fine-tuning. i ​ i ii ) Data Pattern Heterogeneity: MoveGPT  [ 263 ] and UniMove  [ 264 ] utilize Mixture-of-Experts (MoE) architectures, implementing spatial-aware and mobility-aware routing  [ 267 ] , respectively. i ​ i ​ i iii ) Data Quality Heterogeneity: UniTraj  [ 262 ] employs adaptive resampling and self-supervised masking to mitigate sampling rate inconsistencies.

 
 
 Data Issues. To address data fragmentation caused by privacy regulations, MoveGCL  [ 267 ] introduces Generative Continual Learning. By using a frozen teacher model for experience replay and knowledge distillation, it prevents catastrophic forgetting, facilitating decentralized, privacy-preserving model evolution.

 
 
 

### VI-B Large Language Model for Trajectory Computing 

 
 Since 2023, the integration of Large Language Models (LLMs) has fundamentally transformed trajectory computing. As trajectory data are inherently sequential and context-rich, they align well with LLM capabilities in sequence modeling and reasoning. Research has expanded beyond traditional deep learning to exploit LLMs for complex management and mining tasks, focusing on: i i ) Domain adaptation via parameter tuning; i ​ i ii ) Multimodal semantic fusion; i ​ i ​ i iii ) Agentic frameworks for planning; and v v ) Semantic benchmarking.

 
 
 LLM Fine-tuning and Alignment for Trajectory Data .
Despite the potential of zero-shot approaches, Fine-tuning and Alignment remain essential for domain specialization.

 
 • 
 
 Task-Specific Fine-tuning: PLMTrajRec  [ 268 ] addresses trajectory recovery by encoding sampling intervals into natural language prompts, enhancing generalization across rates. Similarly, Traj-LLM  [ 269 ] validates the adaptability of LLMs ( e . g ., GPT-2) for trajectory prediction in few-shot contexts via LoRA.

 

 • 
 
 Human Behavior Alignment: Liu et al.  [ 270 ] propose a framework to align LLM travel choices with human behavior without computationally intensive fine-tuning. It utilizes socio-demographic behavioral embeddings to construct persona loading functions, dynamically selecting appropriate persona-prompts for contextualized simulation.

 

 
 
 
 Multimodal and Semantic Fusion of Trajectory Data .
Feeding non-textual trajectory data into LLMs requires advanced representation, encoding, and modality translation.

 
 • 
 
 Multimodal Trajectory Representation: Traj-MLLM  [ 271 ] introduces a training-free framework that converts trajectories into interleaved image-text sequences via map-anchored tokenization. OmniTraj  [ 272 ] unifies trajectory, topology, road segment, and regional semantics into a shared space to facilitate flexible retrieval and multimodal learning.

 

 • 
 
 Efficient Temporal Tokenization: Addressing long-sequence inefficiencies, RHYTHM  [ 273 ] implements Hierarchical Temporal Tokenization. By segmenting trajectories into daily units and applying hierarchical attention, it significantly reduces sequence length for efficient prediction using a frozen LLM backbone.

 

 • 
 
 Semantic Feature Extraction and Alignment: IMPEL  [ 274 ] utilizes LLMs as geospatial knowledge encoders to generate transferable node representations for Spatio-Temporal Graph Neural Networks. TrajCogn  [ 275 ] aligns continuous spatio-temporal features with “anchor word” embeddings ( e . g ., “turn,” “accelerate”), enabling the LLM to comprehend motion patterns and travel purposes.

 

 
 
 
 LLM-based Agentic Frameworks .
Research increasingly positions LLMs as high-level “controllers” orchestrating planning and reasoning, while specialized tools handle computation.

 
 • 
 
 Unified Modeling and Automation: TrajAgent  [ 276 ] establishes a “large-and-small model collaboration” paradigm. The LLM acts as a manager within a Unified Execution Environment, planning and executing tasks via specialized smaller models and optimizing performance through cooperative learning.

 

 • 
 
 Zero-Shot Prediction and Reasoning: AgentMove  [ 277 ] tackles generalization in zero-shot next-location prediction by coordinating three modules: Spatial-Temporal Memory (individual patterns), World Knowledge Generator (urban structure), and Collective Knowledge Extractor (group patterns), synthesizing these for final reasoning.

 

 • 
 
 Large-Scale Traffic Simulation: To address scalability, MobiVerse  [ 278 ] combines a lightweight generator for activity chains with an LLM-based modifier that reacts to dynamic environments ( e . g ., road closures), simulating over 50,000 agents in real-time. CAMS  [ 279 ] aligns synthetic trajectories from a CityGPT-based agent with real-world data via Direct Preference Optimization .

 

 
 
 
 Semantic Understanding and Benchmarking .
Evaluating LLMs’ semantic comprehension of trajectories involves specialized benchmarking. MobQA  [ 280 ] introduces a dataset testing three cognitive levels: 1) Factual Retrieval , 2) Multiple-Choice Reasoning , and 3) Free-Form Explanation . Results indicate that while LLMs excel at factual retrieval, their capacity for deep semantic reasoning diminishes significantly with increasing trajectory sequence length.

 
 
 
 

## VII Application and Resources 

 

### VII-A Application 

 
 Trajectory data management and mining have revolutionary applications in various fields. As shown in Figure  10 , we summarize these applications from different groups.

 
 
 Fig. 10: Trajectory application in various fields. 
 
 
 Personal Services. 
Trajectory computing plays a vital role in various aspects of personal outdoor services. Firstly, in the aspect of route detection  [ 281 ] , the analysis of user’s driving trajectories enables timely identification and notification of alternative routes or avoidance of traffic congestion, thereby enhancing travel efficiency. Secondly, ride-sharing  [ 282 ] benefits from the application of trajectory data, as platforms can intelligently match passengers traveling in the same direction, leading to more efficient shared rides, reduced travel costs, and alleviated traffic burdens. Furthermore, personalized recommendation services  [ 132 ] utilize trajectory data analysis to understand users’ preferred locations and behavioral patterns, delivering more tailored recommendations for nearby attractions, restaurants and business areas. Moreover, by combining semantics and multi-modal information, it can further analyze user travel intentions and serve as an intelligent agent  [ 283 ] to assist users in decision-making.

 
 
 Business Platforms. 
Trajectory computing significantly influences business operations across various domains, especially for mobility service providers, such as Uber 1 1 
 1 
 
 
 
 https://www.uber.com , DiDi 2 2 
 2 
 
 
 
 https://didiglobal.com , Google Map 3 3 
 3 
 
 
 
 https://www.google.com/maps , Baidu Map 4 4 
 4 
 
 
 
 https://map.baidu.com , Tomtom 5 5 
 5 
 
 
 
 https://www.tomtom.com , Cainiao 6 6 
 6 
 
 
 
 https://www.cainiao.com and so on. In terms of business site selection  [ 284 ] , the analysis of potential customers’ movement trajectories empowers businesses to make informed decisions about optimal operational locations, thereby increasing the likelihood of business success. Additionally, logistics and delivery services benefit from trajectory data, enabling real-time monitoring and rational route planning to enhance delivery efficiency and reduce operational costs  [ 285 ] . Personalized marketing strategies leverage trajectory data analysis to understand user behavior, implementing more individualized marketing approaches to increase user engagement  [ 286 ] . Moreover, road condition prediction and travel order allocation  [ 6 ] , facilitated by real-time analysis, provide businesses with more accurate and efficient services, ultimately elevating overall operational standards.

 
 
 TABLE V: Publicly available trajectory datasets. 
 
 
 
 Categorization | 
 Type | 
 Dataset Name | 
 Main Area | 
 Duration | 
 Statistics | 
 #Point/Records | 
 #Attributes | 

 
 
 
 
 Continuous 
 
 GPS traces 
 | 
 Human | 
 GeoLife  [ 287 ] :  link | 
 Asia | 
 4.5 Years | 
 182 users, 17,621 trajectories, 91% 1 ∼ \sim 5 s/p sample rate | 
 24.87 million+ | 
 7 | 

 
 Human | 
 TMD:  link | 
 Italiana | 
 31 Hours | 
 13 users, 226 trajectories, 0.05 s/p sample rate | 
 – | 
 9 | 

 
 Human | 
 SHL:  link | 
 U.K. | 
 7 Months | 
 3 users, 12 trajectories, 1 s/p sample rate | 
 – | 
 28 | 

 
 Human | 
 OpenStreetMap:  link | 
 Global | 
 From 2005 | 
 8.7 million+ trajectories, continuously updating | 
 – | 
 7 | 

 
 Human | 
 MDC:  link | 
 Switzerland | 
 3 Years | 
 185 trajectories, nearly 200 individuals | 
 4,527,539 | 
 – | 

 
 Taxi | 
 T-Drive  [ 288 ] :  link | 
 Beijing, China | 
 1 Weeks | 
 10357 cars, 177 s/p (Avg.) sample rate | 
 15 million+ | 
 4 | 

 
 Taxi | 
 Porto:  link | 
 Porto, Portugal | 
 9 Months | 
 442 cars, 1,710,990 trajectories, 15 s/p sample rate | 
 1,710,990 | 
 9 | 

 
 Taxi | 
 Taxi-Shanghai:  link | 
 Shanghai, China | 
 1 Year | 
 4,316 cars, 7.8 million trajectories, 5 s/p sample rate | 
 – | 
 5 | 

 
 Taxi | 
 DiDi-Chengdu | 
 Chengdu, China | 
 1 Month | 
 3,493,918 trajectories, 3 s/p Avg. sample rate | 
 1.4 billion+ | 
 5 | 

 
 Taxi | 
 DiDi-Xi’an | 
 Xi’an, China | 
 1 Month | 
 2,180,348 trajectories, 3 s/p Avg. sample rate | 
 1 billion+ | 
 5 | 

 
 Car | 
 WorldTrace:  link | 
 Global | 
 2 Years | 
 70 Countries, 2.45 million trajectories, 1 s/p sample rate | 
 880 million | 
 9+ | 

 
 Truck | 
 Greek:  link | 
 Athens, Greece | 
 – | 
 50 trucks, 1,100 trajectories | 
 112,203 | 
 9 | 

 
 Hurricane | 
 HURDAT:  link | 
 Atlantic | 
 151 Years | 
 1,415 trajectories, 6 h/p sample rate | 
 – | 
 5 | 

 
 Delivery | 
 
 
 
 Grab-Posisi-L  [ 289 ] 
 | 
 Southeast Asia | 
 1 Months | 
 84K trajectories, 1 s/p sample rate | 
 80 million+ | 
 9 | 

 
 Vehicle | 
 NGSIM:  link | 
 USA | 
 45 Minutes | 
 0.1 s/p sample rate, collected through video cameras | 
 – | 
 20+ | 

 
 Animal | 
 Movebank:  link | 
 Global | 
 Decades | 
 8,480 studies, 1,383 taxa, 4,139 data owners | 
 6.1 billion | 
 – | 

 
 Vessel | 
 Vessel Traffic:  link | 
 USA | 
 9 Years | 
 60s/p, AIS data | 
 – | 
 7+ | 

 
 
 
 
 Check-in 
 
 sequences 
 | 
 Human | 
 Gowalla:  link | 
 Global | 
 1.75 Years | 
 196,591 nodes, 950,327 edges | 
 6.44 million+ | 
 5 | 

 
 Human | 
 Brightkite:  link | 
 Global | 
 30 Months | 
 58,228 nodes, 214,078 edges | 
 4,491,143 | 
 5 | 

 
 Human | 
 Foursquare-NY:  link | 
 New York, USA | 
 10 Months | 
 38,336 venues, 824 users | 
 227,428 | 
 8 | 

 
 Human | 
 Foursquare-TKY:  link | 
 Tokyo, Japan | 
 10 Months | 
 61,858 venues, 1,939 users | 
 573,703 | 
 8 | 

 
 Human | 
 Foursquare-Global:  link | 
 Global | 
 18 Months | 
 3,680,126 venues, 266,909 users | 
 33,278,683 | 
 15 | 

 
 Human | 
 Weeplace:  link | 
 Global | 
 7.7 Years | 
 971,309 venues, 15,799 users | 
 7,658,368 | 
 7 | 

 
 Human | 
 Yelp:  link | 
 Global | 
 15 Years | 
 131,930 venues, 1,987,897 users | 
 6,990,280 | 
 20+ | 

 
 Human | 
 Instagram  [ 290 ] | 
 New York, USA | 
 5.5 Years | 
 13,187 venues, 78,233 users | 
 2,216,631 | 
 – | 

 
 Human | 
 GMove  [ 291 ] | 
 2 cities in USA | 
 20 Days | 
 72K trajectories | 
 1.3 million | 
 – | 

 
 Taxi | 
 TLC:  link | 
 New York, USA | 
 From 2009 | 
 115,990 vehicles | 
 – | 
 10+ | 

 
 Bicycle | 
 Mobike-Shanghai | 
 Shanghai, China | 
 2 Weeks | 
 390K+ bikes | 
 60 million+ | 
 10 | 

 
 Bicycle | 
 Bike-Xiamen:  link | 
 Xiamen, China | 
 5 Days | 
 50K+ bikes | 
 198,382 | 
 6 | 

 
 Bicycle | 
 Citi Bikes:  link | 
 New York, USA | 
 From 2013 | 
 68K+ bikes, 2,104 active stations | 
 60K+/month | 
 13 | 

 
 Delivery | 
 LaDe  [ 285 ] :  link | 
 5 cities in China | 
 6 Months | 
 21,000 users, 10,677,000 trajectories | 
 – | 
 17 | 

 
 
 
 
 Synthetic 
 
 traces 
 | 
 Taxi | 
 SynMob  [ 292 ] :  link | 
 
 
 
 Chengdu, China 
 
 Xi’an, China 
 | 
 1 Month | 
 
 
 
 2,000,000 (unrestricted) trajectories, 
 
 3 s/p Avg. sample rate 
 | 
 1 billion+ | 
 4 | 

 
 Vehicle | 
 BerlinMod:  link | 
 Berlin, German | 
 28 Days | 
 2,000 vehicles, 292,940 trajectories | 
 56,129,943 | 
 – | 

 
 
 
 
 Other formats 
 
 of trajectories 
 | 
 Crowd Flow | 
 COVID19USFlows:  link | 
 USA | 
 From 2019 | 
 220k venues, millions of anonymous users | 
 – | 
 – | 

 
 Crowd Flow | 
 MIT-Humob2023:  link | 
 Japan | 
 90 Days | 
 
 
 
 100,000 individuals, 85 types of venues, 
 
 30-minute intervals, 500-meter grid cells 
 | 
 – | 
 – | 

 
 Crowd Flow | 
 BousaiCrowd:  link | 
 Japan | 
 4 Months | 
 1 million users, 20 record/day sample rate | 
 150 million | 
 4 | 

 
 Traffic Flow | 
 TaxiBJ  [ 139 ] :  link | 
 Beijing, China | 
 17 Months | 
 
 
 
 32 × 32 32\times 32 grids, 60 s/p sample rate, 
 
 30-minute intervals, 34,000+ taxis 
 | 
 – | 
 – | 

 
 Traffic Flow | 
 BikeNYC  [ 139 ] :  link | 
 New York, USA | 
 6 Months | 
 
 
 
 16 × 8 16\times 8 grids, 60 s/p sample rate, 
 
 1-hour intervals, 6,800+ bikes 
 | 
 – | 
 – | 

 
 Traffic Flow | 
 TaxiBJ21  [ 293 ] :  link | 
 Beijing, China | 
 3 Months | 
 
 
 
 32 × 32 32\times 32 grids, 600-meter cell length, 
 
 30-minute intervals, 17,749 taxis 
 | 
 – | 
 – | 

 
 
 
 Policy Guidance. 
Trajectory computing offers valuable insights for policymakers and urban planners. In terms of traffic management, trajectory data analysis allows for the intelligent adjustment of traffic signals  [ 294 ] and the rational planning of traffic flow  [ 295 ] , thereby improving urban traffic efficiency. Urban planners can utilize trajectory data to provide a more accurate foundation for city planning  [ 7 ] by understanding the activity trajectories of city residents, facilitating the scientific planning of urban infrastructure and land use. Resource allocation and disease control  [ 296 ] benefit from trajectory data mining, enabling governments to allocate urban resources more accurately and respond promptly to different regional needs. Real-time monitoring of human movement dynamics assists in the early detection of potential disease spread risks. Finally, in the realm of danger (crime) detection  [ 297 ] , the application of trajectory data aids in identifying abnormal behavior, enhancing urban security and enabling law enforcement agencies to intervene and prevent potential criminal events more effectively.

 
 
 

### VII-B Resources 

 
 Trajectory computing is crucial for understanding human mobility, and significant datasets and tools have been accumulated. We conduct a comprehensive analysis to address the current lack of a detailed survey of available open-source data and tools crucial for fostering transparent research.

 
 
 Datasets. 
Table  V lists all known publicly available trajectory datasets, categorized into three groups based on the form of data collection: continuous GPS traces, check-in sequence, and synthetic traces.
The table encapsulates pertinent details for each dataset, including its type, main area, duration and statistical information.

 
 
 Tools. 
For effective analysis and simulation, researchers have various tools at their disposal.
SUMO 7 7 
 7 
 
 
 
 https://eclipse.dev/sumo , an open-source traffic simulator, provides a comprehensive environment for traffic modeling.
SafeGraph 8 8 
 8 
 
 
 
 https://docs.safegraph.com/docs/welcome offers an academic platform with access to large, anonymous datasets for privacy-preserving analysis.
Cblab 9 9 
 9 
 
 
 
 https://github.com/caradryanl/CityBrainLab , a toolkit for scalable traffic simulation, consists of CBEngine, CBData, and CBScenario, enabling efficient simulations and training of traffic policies for large-scale urban scenarios. PyTrack 10 10 
 10 
 
 
 
 https://github.com/titoghose/PyTrack is a comprehensive tool that allows for the modeling of street networks, conducting topological and spatial analyses, and performing map-matching on GPS trajectories.
PyMove 11 11 
 11 
 
 
 
 https://pymove.readthedocs.io/en/latest can be used for the processing and visualization of trajectories and other spatio-temporal data.
TransBigData 12 12 
 12 
 
 
 
 https://transbigdata.readthedocs.io/ is a Python package for analyzing transportation big data and offers a systematic method for processing trajectories.
Traja 13 13 
 13 
 
 
 
 https://github.com/traja-team/traja is a toolkit for numerically characterizing and analyzing the trajectories of moving animals.
MovingPandas 14 14 
 14 
 
 
 
 https://github.com/movingpandas/movingpandas provides generalized trajectory data structures and functions for movement data exploration and analysis.
Scikit-mobility 15 15 
 15 
 
 
 
 https://github.com/scikit-mobility/scikit-mobility is a library designed for human mobility analysis, synthetic trajectory generation, and privacy risks assessment.
Tracktable 16 16 
 16 
 
 
 
 https://github.com/sandialabs/tracktable is a set of Python and C++ libraries for the processing and analysis of trajectory. Yupi 17 17 
 17 
 
 
 
 https://github.com/yupidevs/yupi is a set of tools designed for collecting, generating and processing trajectory data.

 
 
 For detailed information and library access, please visit our official GitHub repository , a central hub for leading advancements in trajectory computing, featuring research papers, benchmark datasets, and source codes.

 
 
 
 

## VIII Challenges and Directions 

 

### VIII-A Current Challenges 

 
 Examining the core triad of data, models, and algorithms, we delineate the current status and challenges in Figure  11 .

 
 
 Data. i i ) Standardizing Trajectory Data Management: Inadequate standardization impedes unified processing and application of trajectory data, necessitating open and standardized management approaches for seamless integration. i ​ i ii ) Acquiring Multisource Semantic Trajectory Data: Despite richer data from sources like social media, effective integration remains challenging. Advanced techniques are needed for acquiring and integrating diverse trajectory data to enhance deep learning models’ multimodal understanding. i ​ i ​ i iii ) Constructing Comprehensive Trajectory Datasets: Large-scale, high-quality trajectory datasets are vital for deep learning model training. Balancing diversity and user privacy, along with ensuring spatio-temporal coverage, is crucial for improved model generalization.

 
 
 Model. i i ) Modeling Uncertainty in Movement Behavior: Handling uncertainty in trajectory data, with its sparse, noisy, and long-tailed distribution, requires robust models adaptable to real-world mobility complexities. i ​ i ii ) Unified model design: Specific model architecture hinders the exploration of unified patterns in trajectory data. It is particularly challenging to design unified models for different tasks. i ​ i ​ i iii ) Robust, Reliable, and Stable Trajectory Modeling: Existing models lack robustness in extreme outliers, especially in practical applications. Ensuring model reliability is imperative.

 
 
 Algorithm. i i ) Fusion Algorithms for Multi-source Trajectory Data: Existing algorithms for multi-source trajectory data can be more efficient. Robust algorithms are essential for global interpretative capabilities in fusing different data types. i ​ i ii ) Fully End-to-End Algorithm Design: Complete end-to-end algorithms simplify structures and enhance efficiency, addressing the multi-stage nature of current trajectory models. i ​ i ​ i iii ) Lightweight and Efficient Algorithm Design: Improving the efficiency of trajectory computing algorithms on resource-constrained edge devices is critical for practical applications.

 
 
 Fig. 11: Current challenges facing the core triad. 
 
 
 

### VIII-B Future Directions 

 
 Building upon the preceding analysis, we outline promising avenues for future research in deep learning for trajectory computing:

 
 
 Resolving Distribution Shifts. Trajectory data exhibits significant spatiotemporal heterogeneity, causing distribution shifts between training and inference phases  [ 298 ] that limit model generalization across diverse locations and time periods. Despite its critical impact, this challenge remains under-addressed in current architectural designs. Future research should investigate continual and incremental learning strategies to mitigate these shifts and enhance model robustness across varied datasets.

 
 
 Multi-Modality Fusion. Human mobility is intrinsically linked to diverse data modalities—including visual, sensor, and textual data  [ 299 , 300 ] —and distinct trajectory types such as taxi routes and public transit flows. As deep learning evolves towards unified multi-modal architectures  [ 301 ] , trajectory computing stands to benefit significantly. Future work must move beyond rudimentary concatenation to develop unified frameworks that effectively integrate heterogeneous data, thereby capturing comprehensive movement patterns and improving predictive accuracy.

 
 
 Foundation Models Large Language Models. Current trajectory computing models often lack generality and external knowledge, relying heavily on task-specific scenarios. Foundation models and Large Language Models (LLMs)  [ 36 ] , characterized by scalable parameters and compression capabilities, offer a pathway to unify trajectory tasks. While requiring careful cost-benefit optimization, integrating LLM knowledge is a rapidly emerging frontier. Key directions include distilling LLM knowledge to augment existing models and utilizing LLMs as autonomous decision-making agents  [ 302 ] .

 
 
 Interpretability. While deep learning in trajectory computing has prioritized performance through complex architectures, the interpretability of these black-box models remains largely unexplored. Identifying the causal factors driving predictive improvements is critical. Recent studies  [ 303 ] have begun incorporating causality and physical laws into network design to transcend mere statistical correlations. Consequently, developing interpretable, physics-informed, and causality-aware deep learning models represents a vital direction for achieving stable and robust predictions.

 
 
 Privacy and Security. Addressing the privacy and security concerns inherent in trajectory data is imperative  [ 26 , 231 ] . Future research must focus on robust techniques for anonymization and sensitive information protection. Promising methodologies include the integration of federated learning for decentralized privacy preservation and the utilization of advanced generative models to synthesize high-fidelity, privacy-compliant trajectory data.

 
 
 
 

## IX Conclusion 

 
 In this survey, we systematically explore the promising intersection between trajectory computing and deep learning (as well as recent large models). Our unified framework unveils a structured understanding of deep learning for trajectory computing, dissecting them into deep learning for trajectory data management and mining. This study offers a concise and organized perspective for researchers and practitioners. Examining existing methods, we provide fresh insights into the core contributions of deep learning to reshape trajectory computing and the field of mobility science, and summarize recent advancements in foundational and large language models in this direction. Furthermore, we summarize key application scenarios and resources, concluding with a discussion on open challenges and future research directions.

 
 
 

## References

 
 [1] 
 I. Newton (1687) 
 
 Philosophiae naturalis principia mathematica .
 
 Cited by: §I .
 

 [2] 
 A. Einstein and M. Grossmann (1913) 
 
 Entwurf einer verallgemeinerten relativitätstheorie und einer theorie der gravitation .
 
 Cited by: §I .
 

 [3] 
 D. Brockmann, L. Hufnagel, and T. Geisel (2006) 
 
 The scaling laws of human travel .
 
 Nature .
 
 Cited by: §I .
 

 [4] 
 G. C. Sanderson (1966) 
 
 The study of mammal movements: a review .
 
 The Journal of Wildlife Management .
 
 Cited by: §I .
 

 [5] 
 Y. Zheng, X. Xie, W. Ma, et al. (2010) 
 
 GeoLife: a collaborative social networking service among user, location and trajectory. .
 
 IEEE Data Eng. Bull. .
 
 Cited by: §I .
 

 [6] 
 H. Wen, Y. Lin, L. Wu, X. Mao, T. Cai, Y. Hou, S. Guo, Y. Liang, G. Jin, Y. Zhao, et al. (2023) 
 
 A survey on service route and time prediction in instant delivery: taxonomy, progress, and prospects .
 
 arXiv preprint arXiv:2309.01194 .
 
 Cited by: §I ,
 §I ,
 §V-A ,
 §VII-A .
 

 [7] 
 Y. Zheng (2015) 
 
 Trajectory data mining: an overview .
 
 ACM Transactions on Intelligent Systems and Technology .
 
 Cited by: TABLE I ,
 §VII-A .
 

 [8] 
 Z. Feng and Y. Zhu (2016) 
 
 A survey on trajectory data mining: techniques and applications .
 
 IEEE Access .
 
 Cited by: TABLE I .
 

 [9] 
 J. D. Mazimpaka and S. Timpf (2016) 
 
 Trajectory data mining: a review of methods and applications .
 
 Journal of spatial information science .
 
 Cited by: TABLE I .
 

 [10] 
 J. Bian, D. Tian, Y. Tang, and D. Tao (2018) 
 
 A survey on trajectory clustering analysis .
 
 arXiv preprint arXiv:1802.06971 .
 
 Cited by: TABLE I ,
 §I .
 

 [11] 
 J. Bian, D. Tian, and Y. Tang (2019) 
 
 Trajectory data classification: a review .
 
 ACM TIST .
 
 Cited by: TABLE I .
 

 [12] 
 V. Koolwal and K. K. Mohbey (2020) 
 
 A comprehensive survey on trajectory-based location prediction .
 
 Iran Journal of Computer Science .
 
 Cited by: TABLE I ,
 §I .
 

 [13] 
 S. Wang, Z. Bao, J. S. Culpepper, and G. Cong (2021) 
 
 A survey on trajectory data management, analytics, and learning .
 
 ACM Computing Surveys (CSUR) .
 
 Cited by: TABLE I ,
 §I ,
 §I ,
 §IV-A .
 

 [14] 
 M. Luca, G. Barlacchi, B. Lepri, and L. Pappalardo (2021) 
 
 A survey on deep learning for human mobility .
 
 ACM Computing Surveys (CSUR) .
 
 Cited by: TABLE I ,
 §I ,
 §I ,
 §V-A ,
 §V-A ,
 §V-F .
 

 [15] 
 H. Pourmahmood-Aghababa and J. M. Phillips (2022) 
 
 Classifying spatial trajectories .
 
 arXiv preprint arXiv:2209.01322 .
 
 Cited by: TABLE I .
 

 [16] 
 M. Shaygan, C. Meese, W. Li, X. G. Zhao, and M. Nejad (2022) 
 
 Traffic prediction using artificial intelligence: review of recent advances and emerging opportunities .
 
 Transportation research part C: emerging technologies .
 
 Cited by: TABLE I .
 

 [17] 
 M. M. G. Duarte and M. Sakr (2023) 
 
 A benchmark of existing tools for outlier detection and cleaning in trajectories .
 
 Cited by: TABLE I ,
 §I .
 

 [18] 
 D. Hu, L. Chen, H. Fang, Z. Fang, T. Li, and Y. Gao (2023) 
 
 Spatio-temporal trajectory similarity measures: a comprehensive survey and quantitative study .
 
 IEEE TKDE .
 
 Cited by: TABLE I ,
 §I .
 

 [19] 
 A. Graser, A. Jalali, J. Lampert, A. Weißenfeld, and K. Janowicz (2024) 
 
 MobilityDL: a review of deep learning from trajectory data .
 
 arXiv preprint arXiv:2402.00732 .
 
 Cited by: TABLE I ,
 §I .
 

 [20] 
 Y. Zheng and X. Zhou (2011) 
 
 Computing with spatial trajectories .
 
 Springer Science Business Media .
 
 Cited by: §I .
 

 [21] 
 S. Shang, L. Chen, Z. Wei, C. S. Jensen, K. Zheng, and P. Kalnis (2017) 
 
 Trajectory similarity join in spatial networks .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §I .
 

 [22] 
 Y. Zheng, Q. Li, Y. Chen, X. Xie, and W. Ma (2008) 
 
 Understanding mobility based on gps data .
 
 In Proceedings of the 10th international conference on Ubiquitous computing ,
 
 Cited by: §I .
 

 [23] 
 Y. LeCun, Y. Bengio, and G. Hinton (2015) 
 
 Deep learning .
 
 nature .
 
 Cited by: §I .
 

 [24] 
 J. Wang, N. Wu, X. Lu, W. X. Zhao, and K. Feng (2019) 
 
 Deep trajectory recovery with fine-grained calibration using kalman filter .
 
 IEEE TKDE .
 
 Cited by: §I ,
 §IV-A .
 

 [25] 
 L. Chen, Y. Gao, Z. Fang, X. Miao, C. S. Jensen, and C. Guo (2019) 
 
 Real-time distributed co-movement pattern detection on streaming trajectories .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §I .
 

 [26] 
 F. Jin, W. Hua, M. Francia, P. Chao, M. Orlowska, and X. Zhou (2022) 
 
 A survey and experimental study on privacy-preserving trajectory data publishing .
 
 IEEE TKDE .
 
 Cited by: §I ,
 §VIII-B .
 

 [27] 
 Z. Jiang (2018) 
 
 A survey on spatial prediction methods .
 
 IEEE TKDE .
 
 Cited by: §I .
 

 [28] 
 S. Safavi, M. Jalali, and M. Houshmand (2022) 
 
 Toward point-of-interest recommendation systems: a critical review on deep-learning approaches .
 
 Electronics .
 
 Cited by: §I .
 

 [29] 
 T. Reich, M. Budka, D. Robbins, and D. Hulbert (2019) 
 
 Survey of eta prediction methods in public transport networks .
 
 arXiv preprint arXiv:1904.05037 .
 
 Cited by: §I .
 

 [30] 
 S. Wang, J. Cao, and S. Y. Philip (2020) 
 
 Deep learning for spatio-temporal data mining: a survey .
 
 IEEE TKDE .
 
 Cited by: §I .
 

 [31] 
 M. Jin, Q. Wen, Y. Liang, C. Zhang, S. Xue, X. Wang, J. Zhang, Y. Wang, H. Chen, X. Li, et al. (2023) 
 
 Large models for time series and spatio-temporal data: a survey and outlook .
 
 arXiv preprint arXiv:2310.10196 .
 
 Cited by: §I .
 

 [32] 
 N. Gao, H. Xue, W. Shao, S. Zhao, K. K. Qin, A. Prabowo, M. S. Rahaman, and F. D. Salim (2022) 
 
 Generative adversarial networks for spatio-temporal data: a survey .
 
 ACM TIST .
 
 Cited by: §I .
 

 [33] 
 M. Veres and M. Moussa (2019) 
 
 Deep learning for intelligent transportation systems: a survey of emerging trends .
 
 IEEE TITS .
 
 Cited by: §I .
 

 [34] 
 H. Yuan and G. Li (2021) 
 
 A survey of traffic prediction: from spatio-temporal data to intelligent transportation .
 
 Data Science and Engineering .
 
 Cited by: §I .
 

 [35] 
 J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. (2023) 
 
 Gpt-4 technical report .
 
 arXiv preprint arXiv:2303.08774 .
 
 Cited by: §I .
 

 [36] 
 G. Jin, Y. Liang, Y. Fang, J. Huang, J. Zhang, and Y. Zheng (2023) 
 
 Spatio-temporal graph neural networks for predictive learning in urban computing: a survey .
 
 arXiv preprint arXiv:2303.14483 .
 
 Cited by: §I ,
 §VIII-B .
 

 [37] 
 Y. Endo, H. Toda, K. Nishida, and J. Ikedo (2016) 
 
 Classifying spatial trajectories using representation learning .
 
 International Journal of Data Science and Analytics .
 
 Cited by: Definition 5 .
 

 [38] 
 D. H. Douglas and T. K. Peucker (1973) 
 
 Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line or its Caricature .
 
 Cartographica: the international journal for geographic information and geovisualization .
 
 Cited by: §IV-A .
 

 [39] 
 C. Long, R. C. Wong, and H. Jagadish (2013) 
 
 Direction-preserving trajectory simplification .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §IV-A .
 

 [40] 
 E. Keogh, S. Chu, D. Hart, and M. Pazzani (2001) 
 
 An online algorithm for segmenting time series .
 
 In Proc. of ICDM ,
 
 Cited by: §IV-A .
 

 [41] 
 N. Meratnia and R. A. de By (2004) 
 
 Spatiotemporal compression techniques for moving point objects .
 
 In Proc. of EDBT ,
 
 Cited by: §IV-A .
 

 [42] 
 T. Li, L. Chen, C. S. Jensen, and T. B. Pedersen (2021) 
 
 TRACE: real-time compression of streaming trajectories in road networks .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §IV-A .
 

 [43] 
 Z. Wang, C. Long, and G. Cong (2021) 
 
 Trajectory simplification with reinforcement learning .
 
 In ICDE ,
 
 Cited by: §IV-A .
 

 [44] 
 Z. Fang, C. He, L. Chen, D. Hu, Q. Sun, L. Li, and Y. Gao (2023) 
 
 A Lightweight Framework for Fast Trajectory Simplification .
 
 In ICDE ,
 
 Cited by: §IV-A .
 

 [45] 
 Z. Wang, C. Long, G. Cong, and Q. Zhang (2021) 
 
 Error-bounded Online Trajectory Simplification with Multi-agent Reinforcement Learning .
 
 In Proc. of KDD ,
 
 Cited by: §IV-A .
 

 [46] 
 Z. Wang, C. Long, G. Cong, and C. S. Jensen (2023) 
 
 Collectively simplifying trajectories in a database: a query accuracy driven approach .
 
 arXiv preprint arXiv:2311.11204 .
 
 Cited by: §IV-A .
 

 [47] 
 J. A. Long (2016) 
 
 Kinematic interpolation of movement data .
 
 IJGIS .
 
 Cited by: §IV-A .
 

 [48] 
 Y. Tremblay, S. A. Shaffer, S. L. Fowler, C. E. Kuhn, B. I. McDonald, M. J. Weise, C. Bost, H. Weimerskirch, D. E. Crocker, M. E. Goebel, et al. (2006) 
 
 Interpolation of animal tracking data in a fluid environment .
 
 Journal of Experimental Biology .
 
 Cited by: §IV-A .
 

 [49] 
 T. Xia, Y. Qi, J. Feng, F. Xu, F. Sun, D. Guo, and Y. Li (2021) 
 
 Attnmove: history enhanced trajectory recovery via attentional network .
 
 In Proc. of AAAI ,
 
 Cited by: §IV-A .
 

 [50] 
 H. Sun, C. Yang, L. Deng, F. Zhou, F. Huang, and K. Zheng (2021) 
 
 Periodicmove: shift-aware human mobility recovery with graph neural network .
 
 In Proc. of CIKM ,
 
 Cited by: §IV-A .
 

 [51] 
 J. Si, J. Yang, Y. Xiang, H. Wang, L. Li, R. Zhang, B. Tu, and X. Chen (2023) 
 
 TrajBERT: bert-based trajectory recovery with spatial-temporal refinement for implicit sparse trajectories .
 
 IEEE TMC .
 
 Cited by: §IV-A .
 

 [52] 
 Y. Chen, G. Cong, and C. Anda (2023) 
 
 TERI: an effective framework for trajectory recovery with irregular time intervals .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §IV-A .
 

 [53] 
 H. Ren, S. Ruan, Y. Li, J. Bao, C. Meng, R. Li, and Y. Zheng (2021) 
 
 Mtrajrec: map-constrained trajectory recovery via seq2seq multi-task learning .
 
 In Proc. of KDD ,
 
 Cited by: §IV-A .
 

 [54] 
 Y. Chen, H. Zhang, W. Sun, and B. Zheng (2023) 
 
 Rntrajrec: road network enhanced trajectory recovery with spatial-temporal transformer .
 
 In ICDE ,
 
 Cited by: §IV-A .
 

 [55] 
 W. Long, Z. Xiao, H. Jiang, Y. Xiong, Z. Qin, Y. Li, and S. Dustdar (2024) 
 
 Learning semantic behavior for human mobility trajectory recovery .
 
 IEEE TITS .
 
 Cited by: §IV-A .
 

 [56] 
 Z. Li, Z. Li, X. Hu, G. Du, Y. Nie, F. Zhu, L. Bai, and R. Zhao (2023) 
 
 VisionTraj: a noise-robust trajectory recovery framework based on large-scale camera network .
 
 arXiv preprint arXiv:2312.06428 .
 
 Cited by: §IV-A .
 

 [57] 
 L. Liao, Y. Lin, W. Li, F. Zou, and L. Luo (2023) 
 
 Traj2Traj: a road network constrained spatiotemporal interpolation model for traffic trajectory restoration .
 
 Transactions in GIS .
 
 Cited by: §IV-A .
 

 [58] 
 X. Zhang, X. Liang, H. Wang, S. Wang, and T. He (2022) 
 
 PATR: periodicity-aware trajectory recovery for express system via seq2seq model .
 
 In IEEE Global Communications Conference ,
 
 Cited by: §IV-A .
 

 [59] 
 S. Ruan, C. Long, J. Bao, C. Li, Z. Yu, R. Li, Y. Liang, T. He, and Y. Zheng (2020) 
 
 Learning to generate maps from trajectories .
 
 In Proc. of AAAI ,
 
 Cited by: §IV-A .
 

 [60] 
 B. Li, J. Gao, S. Chen, S. Lim, and H. Jiang (2024) 
 
 DF-drunet: a decoder fusion model for automatic road extraction leveraging remote sensing images and gps trajectory data .
 
 International Journal of Applied Earth Observation and Geoinformation .
 
 Cited by: §IV-A .
 

 [61] 
 S. Wang, Z. Wang, S. Ruan, H. Han, K. Xiong, H. Yuan, Z. Yuan, G. Li, J. Bao, and Y. Zheng (2024) 
 
 DelvMap: completing residential roads in maps based on couriers’ trajectories and satellite imagery .
 
 IEEE Transactions on Geoscience and Remote Sensing .
 
 Cited by: §IV-A .
 

 [62] 
 G. Taylor, G. Blewitt, D. Steup, S. Corbett, and A. Car (2001) 
 
 Road reduction filtering for gps-gis navigation .
 
 Transactions in GIS .
 
 Cited by: §IV-A .
 

 [63] 
 M. A. Quddus, W. Y. Ochieng, L. Zhao, and R. B. Noland (2003) 
 
 A general map matching algorithm for transport telematics applications .
 
 GPS solutions .
 
 Cited by: §IV-A .
 

 [64] 
 W. Y. Ochieng, M. A. Quddus, and R. B. Noland (2003) 
 
 Map-matching in complex urban road networks .
 
 Brazilian Journal of Cartography (Revista Brasileira de Cartografia) .
 
 Cited by: §IV-A .
 

 [65] 
 C. Yang and G. Gidófalvi (2018) 
 
 Fast map matching, an algorithm integrating hidden markov model with precomputation .
 
 IJGIS .
 
 Cited by: §IV-A .
 

 [66] 
 J. Feng, Y. Li, K. Zhao, Z. Xu, T. Xia, J. Zhang, and D. Jin (2022) 
 
 DeepMM: deep learning based map matching with data augmentation .
 
 IEEE TMC .
 
 Cited by: §IV-A .
 

 [67] 
 Z. Jin, J. Kim, H. Yeo, and S. Choi (2022) 
 
 Transformer-based map-matching model with limited labeled data using transfer-learning approach .
 
 Transportation Research Part C: Emerging Technologies .
 
 Cited by: §IV-A .
 

 [68] 
 L. Jiang, C. Chen, and C. Chen (2023) 
 
 L2mm: learning to map matching with deep models for low-quality gps trajectory data .
 
 ACM Transactions on Knowledge Discovery from Data .
 
 Cited by: §IV-A .
 

 [69] 
 Y. Liu, Q. Ge, W. Luo, Q. Huang, L. Zou, H. Wang, X. Li, and C. Liu (2024) 
 
 GraphMM: graph-based vehicular map matching by leveraging trajectory and road correlations .
 
 IEEE TKDE .
 
 Cited by: §IV-A .
 

 [70] 
 Z. Shen, W. Du, X. Zhao, and J. Zou (2020) 
 
 DMM: fast map matching for cellular data .
 
 In Proceedings of the 26th Annual International Conference on Mobile Computing and Networking ,
 
 Cited by: §IV-A .
 

 [71] 
 Z. Zhu, D. He, W. Hua, J. Kim, and H. Shi (2023) 
 
 Map-matching on wireless traffic sensor data with a sequence-to-sequence model .
 
 In Proc. of MDM ,
 
 Cited by: §IV-A .
 

 [72] 
 H. Lu, F. Lyu, H. Wu, J. Zhang, J. Ren, Y. Zhang, and X. Shen (2023) 
 
 FL-amm: federated learning augmented map matching with heterogeneous cellular moving trajectories .
 
 IEEE Journal on Selected Areas in Communications .
 
 Cited by: §IV-A .
 

 [73] 
 H. Wang, K. Zheng, J. Xu, B. Zheng, X. Zhou, and S. Sadiq (2014) 
 
 Sharkdb: an in-memory column-oriented trajectory storage .
 
 In Proceedings of the 23rd ACM international conference on conference on information and knowledge management ,
 
 Cited by: §IV-B .
 

 [74] 
 X. Ding, L. Chen, Y. Gao, C. S. Jensen, and H. Bao (2018) 
 
 UlTraMan: a unified platform for big trajectory data management and analytics .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §IV-B .
 

 [75] 
 Z. Fang, L. Chen, Y. Gao, L. Pan, and C. S. Jensen (2021) 
 
 Dragoon: a hybrid and efficient big trajectory management system for offline and online analytics .
 
 The VLDB Journal .
 
 Cited by: §IV-B .
 

 [76] 
 R. Li, H. He, R. Wang, S. Ruan, T. He, J. Bao, J. Zhang, L. Hong, and Y. Zheng (2021) 
 
 Trajmesa: a distributed nosql-based trajectory data management system .
 
 IEEE TKDE .
 
 Cited by: §IV-B .
 

 [77] 
 D. Foster (2004) 
 
 GPX the gps exchange format .
 
 http://www. topografix. com/gpx. asp .
 
 Cited by: §IV-B .
 

 [78] 
 J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang, X. Guo, C. Li, X. Xu, et al. (2021) 
 
 Milvus: a purpose-built vector data management system .
 
 In Proceedings of the 2021 International Conference on Management of Data ,
 
 Cited by: §IV-B .
 

 [79] 
 Z. Cai, F. Ren, J. Chen, and Z. Ding (2017) 
 
 Vector-based trajectory storage and query for intelligent transport system .
 
 IEEE TITS .
 
 Cited by: §IV-B .
 

 [80] 
 Z. Fang, S. Gong, L. Chen, J. Xu, Y. Gao, and C. S. Jensen (2023) 
 
 Ghost: a general framework for high-performance online similarity queries over distributed trajectory streams .
 
 Proceedings of the ACM on Management of Data .
 
 Cited by: §IV-B .
 

 [81] 
 H. He, R. Li, S. Ruan, T. He, J. Bao, T. Li, and Y. Zheng (2022) 
 
 Trass: efficient trajectory similarity search based on key-value data stores .
 
 In ICDE ,
 
 Cited by: §IV-B .
 

 [82] 
 L. Chen, Y. Gao, X. Li, C. S. Jensen, and G. Chen (2017) 
 
 Efficient metric indexing for similarity search and similarity joins .
 
 IEEE TKDE .
 
 Cited by: §IV-B .
 

 [83] 
 D. Xie, F. Li, and J. M. Phillips (2017) 
 
 Distributed Trajectory Similarity Search .
 
 PVLDB .
 
 Cited by: §IV-B .
 

 [84] 
 L. Chen, Q. Zhong, X. Xiao, Y. Gao, P. Jin, and C. S. Jensen (2018) 
 
 Price-and-time-aware dynamic ridesharing .
 
 In ICDE ,
 
 Cited by: §IV-B .
 

 [85] 
 H. Yuan and G. Li (2019) 
 
 Distributed In-Memory Trajectory Similarity Search and Join on Road Network .
 
 In ICDE ,
 
 Cited by: §IV-B .
 

 [86] 
 J. Qi, G. Liu, C. S. Jensen, and L. Kulik (2020) 
 
 Effectively Learning Spatial Indices .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §IV-B .
 

 [87] 
 V. Pandey, A. van Renen, A. Kipf, I. Sabek, J. Ding, and A. Kemper (2020) 
 
 The Case for Learned Spatial Indexes .
 
 arXiv preprint arXiv:2008.10349 .
 
 Cited by: §IV-B .
 

 [88] 
 H. Ramadhan and J. Kwon (2022) 
 
 X-FIST: Extended Flood Index for Efficient Similarity Search in Massive Trajectory Dataset .
 
 Information Sciences .
 
 Cited by: §IV-B .
 

 [89] 
 Y. Chang, E. Tanin, G. Cong, C. S. Jensen, and J. Qi (2024) 
 
 Trajectory similarity measurement: an efficiency perspective .
 
 Proc. VLDB Endow. .
 
 Cited by: §IV-C ,
 TABLE II ,
 TABLE II ,
 TABLE II ,
 TABLE II ,
 TABLE II .
 

 [90] 
 X. Li, K. Zhao, G. Cong, C. S. Jensen, and W. Wei (2018) 
 
 Deep Representation Learning for Trajectory Similarity Computation .
 
 In ICDE ,
 
 Cited by: §IV-C ,
 §IV-C ,
 TABLE II .
 

 [91] 
 Z. Chen, K. Li, S. Zhou, L. Chen, and S. Shang (2023) 
 
 Towards Robust Trajectory Similarity Computation: Representation-based Spatio-temporal Similarity Quantification .
 
 World Wide Web .
 
 Cited by: §IV-C ,
 §IV-C ,
 TABLE II .
 

 [92] 
 A. Liu, Y. Zhang, X. Zhang, G. Liu, Y. Zhang, Z. Li, L. Zhao, Q. Li, and X. Zhou (2022) 
 
 Representation Learning with Multi-level Attention for Activity Trajectory Similarity Computation .
 
 IEEE TKDE .
 
 Cited by: §IV-C ,
 §IV-C ,
 TABLE II .
 

 [93] 
 Z. Wang, C. Long, G. Cong, and C. Ju (2019) 
 
 Effective and Efficient Sports Play Retrieval with Deep Representation Learning .
 
 In Proc. of KDD ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [94] 
 L. Deng, Y. Zhao, Z. Fu, H. Sun, S. Liu, and K. Zheng (2022) 
 
 Efficient Trajectory Similarity Computation with Contrastive Learning .
 
 In Proc. of CIKM ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [95] 
 H. Cao, H. Tang, Y. Wu, F. Wang, and Y. Xu (2021) 
 
 On Accurate Computation of Trajectory Similarity via Single Image Super-resolution .
 
 In IJCNN ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [96] 
 X. Liu, X. Tan, Y. Guo, Y. Chen, and Z. Zhang (2022) 
 
 CSTRM: Contrastive Self-Supervised Trajectory Representation Model for Trajectory Similarity Computation .
 
 Computer Communications .
 
 Cited by: §IV-C ,
 TABLE II .
 

 [97] 
 Y. Chang, J. Qi, Y. Liang, and E. Tanin (2023) 
 
 Contrastive Trajectory Similarity Learning with Dual-Feature Attention .
 
 In ICDE ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [98] 
 S. Li, W. Chen, B. Yan, Z. Li, S. Zhu, and Y. Yu (2023) 
 
 Self-supervised contrastive representation learning for large-scale trajectories .
 
 Future Generation Computer Systems .
 
 Cited by: §IV-C ,
 TABLE II .
 

 [99] 
 D. Yao, G. Cong, C. Zhang, and J. Bi (2019) 
 
 Computing Trajectory Similarity in Linear Time: A Generic Seed-guided Neural Netric learning approach .
 
 In ICDE ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [100] 
 H. Zhang, X. Zhang, Q. Jiang, B. Zheng, Z. Sun, W. Sun, and C. Wang (2020) 
 
 Trajectory Similarity Learning with Auxiliary Supervision and Optimal Matching .
 
 In Proc. of IJCAI ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [101] 
 P. Yang, H. Wang, D. Lian, Y. Zhang, L. Qin, and W. Zhang (2022) 
 
 TMN: Trajectory Matching Networks for Predicting Similarity .
 
 In ICDE ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [102] 
 P. Yang, H. Wang, Y. Zhang, L. Qin, W. Zhang, and X. Lin (2021) 
 
 T3S: Effective Representation Learning for Trajectory Similarity Computation .
 
 In ICDE ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [103] 
 D. Yao, H. Hu, L. Du, G. Cong, S. Han, and J. Bi (2022) 
 
 TrajGAT: A Graph-based Long-term Dependency Modeling Approach for Trajectory Similarity Computation .
 
 In Proc. of KDD ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [104] 
 T. Fu and W. Lee (2020) 
 
 Trembr: Exploring Road Networks for Trajectory Representation Learning .
 
 ACM Transactions on Intelligent Systems and Technology .
 
 Cited by: §IV-C ,
 TABLE II .
 

 [105] 
 S. B. Yang, J. Hu, C. Guo, B. Yang, and C. S. Jensen (2023) 
 
 Lightpath: lightweight and scalable path representation learning .
 
 In Proc. of KDD ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [106] 
 P. Han, J. Wang, D. Yao, S. Shang, and X. Zhang (2021) 
 
 A Graph-based Approach for Trajectory Similarity Computation in Spatial Networks .
 
 In Proc. of KDD ,
 
 Cited by: §IV-C ,
 TABLE II ,
 TABLE II .
 

 [107] 
 S. Zhou, J. Li, H. Wang, S. Shang, and P. Han (2023) 
 
 GRLSTM: Trajectory Similarity Computation with Graph-based Residual LSTM .
 
 In Proc. of AAAI ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [108] 
 Y. Chang, E. Tanin, X. Cao, and J. Qi (2023) 
 
 Spatial Structure-Aware Road Network Embedding via Graph Contrastive Learning .
 
 In EDBT ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [109] 
 Z. Fang, Y. Du, X. Zhu, L. Chen, Y. Gao, and C. S. Jensen (2022) 
 
 Spatio-temporal Trajectory Similarity Learning in Road Networks .
 
 In Proc. of KDD ,
 
 Cited by: §IV-C ,
 TABLE II .
 

 [110] 
 D. A. Tedjopurnomo, X. Li, Z. Bao, G. Cong, F. Choudhury, and A. K. Qin (2021) 
 
 Similar Trajectory Search with Spatio-temporal Deep Representation Learning .
 
 ACM Transactions on Intelligent Systems and Technology .
 
 Cited by: §IV-C .
 

 [111] 
 Y. Chen, P. Yu, W. Chen, Z. Zheng, and M. Guo (2021) 
 
 Embedding-based Similarity Computation for Massive Vehicle Trajectory Data .
 
 IEEE Internet of Things Journal .
 
 Cited by: §IV-C .
 

 [112] 
 S. B. Yang, C. Guo, J. Hu, J. Tang, and B. Yang (2021) 
 
 Unsupervised path representation learning with curriculum negative sampling. .
 
 In Proc. of IJCAI ,
 
 Cited by: §IV-C .
 

 [113] 
 S. B. Yang, C. Guo, J. Hu, B. Yang, J. Tang, and C. S. Jensen (2022) 
 
 Weakly-supervised temporal path representation learning with contrastive curriculum learning .
 
 In ICDE ,
 
 Cited by: §IV-C .
 

 [114] 
 S. Zhou, P. Han, D. Yao, L. Chen, and X. Zhang (2023) 
 
 Spatial-temporal fusion graph framework for trajectory similarity computation .
 
 World Wide Web .
 
 Cited by: §IV-C .
 

 [115] 
 G. Yuan, P. Sun, J. Zhao, D. Li, and C. Wang (2017) 
 
 A review of moving object trajectory clustering algorithms .
 
 Artificial Intelligence Review .
 
 Cited by: §IV-C .
 

 [116] 
 C. Chen, C. Liao, X. Xie, Y. Wang, and J. Zhao (2019) 
 
 Trip2Vec: a deep embedding approach for clustering and profiling taxi trip purposes .
 
 Personal and Ubiquitous Computing .
 
 Cited by: §IV-C .
 

 [117] 
 D. Arthur S. Vassilvitskii et al. (2007) 
 
 K-means++: the advantages of careful seeding .
 
 In Soda ,
 
 Cited by: §IV-C .
 

 [118] 
 X. Olive, L. Basora, B. Viry, and R. Alligier (2020) 
 
 Deep trajectory clustering with autoencoders .
 
 In ICRAT 2020, 9th International Conference for Research in Air Transportation ,
 
 Cited by: §IV-C .
 

 [119] 
 M. Yue, Y. Li, H. Yang, R. Ahuja, Y. Chiang, and C. Shahabi (2019) 
 
 DETECT: deep trajectory clustering for mobility-behavior analysis .
 
 In 2019 IEEE International Conference on Big Data (Big Data) ,
 
 Cited by: §IV-C .
 

 [120] 
 Z. Fang, Y. Du, L. Chen, Y. Hu, Y. Gao, and G. Chen (2021) 
 
 E2DTC: an end to end deep trajectory clustering framework via self-training .
 
 In ICDE ,
 
 Cited by: §IV-C .
 

 [121] 
 B. Bach, P. Dragicevic, D. Archambault, C. Hurter, and S. Carpendale (2017) 
 
 A descriptive framework for temporal data visualizations based on generalized space-time cubes .
 
 In Computer graphics forum ,
 
 Cited by: §IV-D .
 

 [122] 
 C. Lee, Y. Kim, S. Jin, D. Kim, R. Maciejewski, D. Ebert, and S. Ko (2019) 
 
 A visual analytics system for exploring, monitoring, and forecasting road traffic congestion .
 
 IEEE transactions on visualization and computer graphics .
 
 Cited by: §IV-D .
 

 [123] 
 Z. Zhou, L. Meng, C. Tang, Y. Zhao, Z. Guo, M. Hu, and W. Chen (2018) 
 
 Visual abstraction of large scale geospatial origin-destination movement data .
 
 IEEE transactions on visualization and computer graphics .
 
 Cited by: §IV-D .
 

 [124] 
 T. Maekawa, K. Ohara, Y. Zhang, M. Fukutomi, S. Matsumoto, K. Matsumura, H. Shidara, S. J. Yamazaki, R. Fujisawa, K. Ide, et al. (2020) 
 
 Deep learning-assisted comparative analysis of animal trajectories with deephl .
 
 Nature communications .
 
 Cited by: §IV-D .
 

 [125] 
 H. Liu, T. Taniguchi, Y. Tanaka, K. Takenaka, and T. Bando (2017) 
 
 Visualization of driving behavior based on hidden feature extraction by using deep learning .
 
 IEEE TITS .
 
 Cited by: §IV-D .
 

 [126] 
 X. Zhang, Y. Zheng, Z. Zhao, Y. Liu, M. Blumenstein, and J. Li (2021) 
 
 Deep learning detection of anomalous patterns from bus trajectories for traffic insight analysis .
 
 Knowledge-Based Systems .
 
 Cited by: §IV-D .
 

 [127] 
 Z. Deng, D. Weng, S. Liu, Y. Tian, M. Xu, and Y. Wu (2023) 
 
 A survey of urban visual analytics: advances and future directions .
 
 Computational Visual Media .
 
 Cited by: §IV-D .
 

 [128] 
 Y. Bao, Z. Huang, L. Li, Y. Wang, and Y. Liu (2021) 
 
 A bilstm-cnn model for predicting users’ next locations based on geotagged social media .
 
 IJGIS .
 
 Cited by: §V-A .
 

 [129] 
 D. Yang, B. Fankhauser, P. Rosso, and P. Cudre-Mauroux (2020) 
 
 Location prediction over sparse user mobility traces using rnns .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-A .
 

 [130] 
 Q. Liu, S. Wu, L. Wang, and T. Tan (2016) 
 
 Predicting the next location: a recurrent model with spatial and temporal contexts .
 
 In Proc. of AAAI ,
 
 Cited by: §V-A .
 

 [131] 
 J. Feng, Y. Li, C. Zhang, F. Sun, F. Meng, A. Guo, and D. Jin (2018) 
 
 Deepmove: predicting human mobility with attentional recurrent networks .
 
 In Proc. of WWW ,
 
 Cited by: §V-A ,
 §V-F .
 

 [132] 
 S. Li, W. Chen, B. Wang, C. Huang, Y. Yu, and J. Dong 
 
 MCN4Rec: multi-level collaborative neural network for next location recommendation .
 
 ACM Transactions on Information Systems .
 
 Cited by: §V-A ,
 §VII-A .
 

 [133] 
 X. Song, H. Kanasugi, and R. Shibasaki (2016) 
 
 Deeptransport: prediction and simulation of human mobility and transportation mode at a citywide level .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-A .
 

 [134] 
 H. Xue, F. Salim, Y. Ren, and N. Oliver (2021) 
 
 MobTCast: leveraging auxiliary trajectory forecasting for human mobility prediction .
 
 Proc. of NeurIPS .
 
 Cited by: §V-A .
 

 [135] 
 M. A. Islam, M. M. Mohammad, S. S. S. Das, and M. E. Ali (2022) 
 
 A survey on deep learning based point-of-interest (poi) recommendations .
 
 Neurocomputing .
 
 Cited by: §V-A .
 

 [136] 
 W. Chen and Y. Liang (2025) 
 
 Learning with calibration: exploring test-time computing of spatio-temporal forecasting .
 
 In NeurIPS ,
 
 Cited by: §V-A .
 

 [137] 
 W. Chen, Y. Wu, Y. Zhu, X. Hao, S. Wang, and Y. Liang (2025) 
 
 Select, then balance: a plug-and-play framework for exogenous-aware spatio-temporal forecasting .
 
 arXiv preprint arXiv:2509.05779 .
 
 Cited by: §V-A .
 

 [138] 
 F. Canova (1999) 
 
 Vector autoregressive models: specification, estimation, inference, and forecasting .
 
 Handbook of applied econometrics volume 1: Macroeconomics .
 
 Cited by: §V-A .
 

 [139] 
 J. Zhang, Y. Zheng, and D. Qi (2017) 
 
 Deep spatio-temporal residual networks for citywide crowd flows prediction .
 
 In Proc. of AAAI ,
 
 Cited by: §V-A ,
 TABLE V ,
 TABLE V .
 

 [140] 
 H. Yao, F. Wu, J. Ke, X. Tang, Y. Jia, S. Lu, P. Gong, J. Ye, and Z. Li (2018) 
 
 Deep multi-view spatial-temporal network for taxi demand prediction .
 
 In Proc. of AAAI ,
 
 Cited by: §V-A ,
 §V-F .
 

 [141] 
 W. Jin, Y. Lin, Z. Wu, and H. Wan (2018) 
 
 Spatio-temporal recurrent convolutional networks for citywide short-term crowd flows prediction .
 
 In Proceedings of the 2nd International Conference on Compute and Data Analysis ,
 
 Cited by: §V-A .
 

 [142] 
 A. Zonoozi, J. Kim, X. Li, and G. Cong (2018) 
 
 Periodic-crn: a convolutional recurrent model for crowd density prediction with recurring periodic patterns. .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-A .
 

 [143] 
 R. Jiang, Z. Cai, Z. Wang, C. Yang, Z. Fan, Q. Chen, K. Tsubouchi, X. Song, and R. Shibasaki (2021) 
 
 DeepCrowd: a deep model for large-scale citywide crowd density and flow prediction .
 
 IEEE TKDE .
 
 Cited by: §V-A .
 

 [144] 
 R. Jiang, X. Song, D. Huang, X. Song, T. Xia, Z. Cai, Z. Wang, K. Kim, and R. Shibasaki (2019) 
 
 Deepurbanevent: a system for predicting citywide crowd dynamics at big events .
 
 In Proc. of KDD ,
 
 Cited by: §V-A .
 

 [145] 
 Z. Fang, D. Wu, L. Pan, et al. (2022) 
 
 When transfer learning meets cross-city urban flow prediction: spatio-temporal adaptation matters .
 
 IJCAI’22 .
 
 Cited by: §V-A .
 

 [146] 
 J. Sun, J. Zhang, Q. Li, X. Yi, Y. Liang, and Y. Zheng (2020) 
 
 Predicting citywide crowd flows in irregular regions using multi-view graph convolutional networks .
 
 IEEE TKDE .
 
 Cited by: §V-A .
 

 [147] 
 Z. Fang, L. Pan, L. Chen, Y. Du, and Y. Gao (2021) 
 
 MDTP: a multi-source deep traffic prediction framework over spatio-temporal trajectory data .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §V-A .
 

 [148] 
 W. Chen and Y. Liang (2025) 
 
 Expand and compress: exploring tuning principles for continual spatio-temporal graph forecasting .
 
 In ICLR ,
 
 Cited by: §V-A .
 

 [149] 
 S. Zhang, Z. Luo, L. Yang, F. Teng, and T. Li (2024) 
 
 A survey of route recommendations: methods, applications, and opportunities .
 
 arXiv preprint arXiv:2403.00284 .
 
 Cited by: §V-B .
 

 [150] 
 N. Wu, X. W. Zhao, J. Wang, and D. Pan (2020) 
 
 Learning effective road network representation with hierarchical graph neural networks .
 
 In Proc. of KDD ,
 
 Cited by: §V-B .
 

 [151] 
 M. Liu, Z. Peng, X. Yu, S. Wang, and Q. Song (2021) 
 
 LDFeRR: a fuel-efficient route recommendation approach for long-distance driving based on historical trajectories .
 
 In Proc. of SDM ,
 
 Cited by: §V-B .
 

 [152] 
 T. Fu and W. Lee (2021) 
 
 ProgRPGAN: progressive gan for route planning .
 
 In Proc. of KDD ,
 
 Cited by: §V-B .
 

 [153] 
 B. Wang, Y. Guo, and Y. Chen (2021) 
 
 Personalized path recommendation with specified way-points based on trajectory representations .
 
 In 2021 17th International Conference on Mobility, Sensing and Networking (MSN) ,
 
 Cited by: §V-B .
 

 [154] 
 Z. Wang, Z. Peng, S. Wang, and Q. Song (2022) 
 
 Personalized long-distance fuel-efficient route recommendation through historical trajectories mining .
 
 In Proc. of WSDM ,
 
 Cited by: §V-B .
 

 [155] 
 P. Wang, L. Li, R. Wang, and X. Tao (2023) 
 
 Query2Trip: dual-debiased learning for neural trip recommendation .
 
 In Proc. of DASFAA ,
 
 Cited by: §V-B .
 

 [156] 
 Q. Gao, W. Wang, L. Huang, X. Yang, T. Li, and H. Fujita (2023) 
 
 Dual-grained human mobility learning for location-aware trip recommendation with spatial–temporal graph knowledge fusion .
 
 Information Fusion .
 
 Cited by: §V-B .
 

 [157] 
 Y. Wu, W. Song, Z. Cao, J. Zhang, and A. Lim (2021) 
 
 Learning improvement heuristics for solving routing problems .
 
 IEEE transactions on neural networks and learning systems .
 
 Cited by: §V-B .
 

 [158] 
 N. Wu, J. Wang, W. X. Zhao, and Y. Jin (2019) 
 
 Learning to effectively estimate the travel time for fastest route recommendation .
 
 In Proc. of CIKM ,
 
 Cited by: §V-B .
 

 [159] 
 Y. Zhang, P. Siriaraya, Y. Wang, S. Wakamiya, Y. Kawai, and A. Jatowt (2018) 
 
 Walking down a different path: route recommendation based on visual and facility based diversity .
 
 In Proc. of WWW ,
 
 Cited by: §V-B .
 

 [160] 
 H. Liu, J. Han, Y. Fu, J. Zhou, X. Lu, and H. Xiong (2020) 
 
 Multi-modal transportation recommendation with unified route representation learning .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §V-B .
 

 [161] 
 S. Ji, Z. Wang, T. Li, and Y. Zheng (2020) 
 
 Spatio-temporal feature fusion for dynamic taxi route recommendation via deep reinforcement learning .
 
 Knowledge-Based Systems .
 
 Cited by: §V-B .
 

 [162] 
 C. Bi, G. Pan, L. Yang, C. Lin, M. Hou, and Y. Huang (2019) 
 
 Evacuation route recommendation using auto-encoder and markov decision process .
 
 Applied Soft Computing .
 
 Cited by: §V-B .
 

 [163] 
 J. Bao, Y. Zheng, D. Wilkie, and M. Mokbel (2015) 
 
 Recommendations in location-based social networks: a survey .
 
 GeoInformatica .
 
 Cited by: §V-B .
 

 [164] 
 C. Brown, V. Nicosia, S. Scellato, A. Noulas, and C. Mascolo (2012) 
 
 Where online friends meet: social communities in location-based networks .
 
 In Proc. of AAAI ,
 
 Cited by: §V-B .
 

 [165] 
 C. Chu, W. Wu, C. Wang, T. Chen, and J. Chen (2013) 
 
 Friend recommendation for location-based mobile social networks .
 
 In 2013 seventh international conference on innovative mobile and internet services in ubiquitous computing ,
 
 Cited by: §V-B .
 

 [166] 
 X. Yu, A. Pan, L. Tang, Z. Li, and J. Han (2011) 
 
 Geo-friends recommendation in gps-based cyber-physical social network .
 
 In 2011 International Conference on Advances in Social Networks Analysis and Mining ,
 
 Cited by: §V-B .
 

 [167] 
 B. Perozzi, R. Al-Rfou, and S. Skiena (2014) 
 
 Deepwalk: online learning of social representations .
 
 In Proc. of KDD ,
 
 Cited by: §V-B .
 

 [168] 
 A. Grover and J. Leskovec (2016) 
 
 Node2vec: scalable feature learning for networks .
 
 In Proc. of KDD ,
 
 Cited by: §V-B .
 

 [169] 
 P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, Y. Bengio, et al. 
 
 Graph attention networks .
 
 stat .
 
 Cited by: §V-B .
 

 [170] 
 D. Yang, B. Qu, J. Yang, and P. Cudre-Mauroux (2019) 
 
 Revisiting user mobility and social relationships in lbsns: a hypergraph embedding approach .
 
 In Proc. of WWW ,
 
 Cited by: §V-B .
 

 [171] 
 D. Yang, B. Qu, J. Yang, and P. Cudré-Mauroux (2020) 
 
 Lbsn2vec++: heterogeneous hypergraph embedding for location-based social networks .
 
 IEEE TKDE .
 
 Cited by: §V-B .
 

 [172] 
 W. Zhang, X. Lai, and J. Wang (2020) 
 
 Social link inference via multiview matching network from spatiotemporal trajectories .
 
 IEEE transactions on neural networks and learning systems .
 
 Cited by: §V-B .
 

 [173] 
 Q. Gao, G. Trajcevski, F. Zhou, K. Zhang, T. Zhong, and F. Zhang (2018) 
 
 Trajectory-based social circle inference .
 
 In Proceedings of the 26th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems ,
 
 Cited by: §V-B .
 

 [174] 
 G. Qin, L. Song, Y. Yu, C. Huang, W. Jia, Y. Cao, and J. Dong (2023) 
 
 Graph structure learning on user mobility data for social relationship inference .
 
 In Proc. of AAAI ,
 
 Cited by: §V-B .
 

 [175] 
 D. Rafailidis and F. Crestani (2018) 
 
 Friend recommendation in location-based social networks via deep pairwise learning .
 
 In 2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM) ,
 
 Cited by: §V-B .
 

 [176] 
 C. L. da Silva, L. M. Petry, and V. Bogorny (2019) 
 
 A survey and comparison of trajectory classification methods .
 
 In 2019 8th Brazilian Conference on Intelligent Systems ,
 
 Cited by: §V-C .
 

 [177] 
 J. Lee and J. Han (2008) 
 
 TraClass: trajectory classification using hierarchical region-based and trajectory-based clustering .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §V-C .
 

 [178] 
 A. Soleymani, J. Cachat, K. Robinson, S. Dodge, A. Kalueff, and R. Weibel (2014) 
 
 Integrating cross-scale analysis in the spatial and temporal domains for classification of behavioral movement .
 
 Journal of Spatial Information Science .
 
 Cited by: §V-C .
 

 [179] 
 Y. Zheng, L. Liu, L. Wang, and X. Xie (2008) 
 
 Learning transportation mode from raw gps data for geographic applications on the web .
 
 In Proc. of WWW ,
 
 Cited by: §V-C .
 

 [180] 
 S. Dodge, R. Weibel, and E. Forootan (2009) 
 
 Revealing the physics of movement: comparing the similarity of movement characteristics of different types of moving objects .
 
 Computers, Environment and Urban Systems .
 
 Cited by: §V-C .
 

 [181] 
 D. Hu, Z. Fang, H. Fang, T. Li, C. Shen, L. Chen, and Y. Gao (2022) 
 
 Estimator: an effective and scalable framework for transportation mode classification over trajectories .
 
 arXiv preprint arXiv:2212.05502 .
 
 Cited by: §V-C .
 

 [182] 
 S. Dabiri, C. Lu, K. Heaslip, and C. K. Reddy (2019) 
 
 Semi-supervised deep learning approach for transportation mode identification using gps trajectory data .
 
 IEEE TKDE .
 
 Cited by: §V-C ,
 §V-C .
 

 [183] 
 J. Liu, Y. Liu, W. Zhu, X. Zhu, and L. Song (2023) 
 
 Distributional and spatial-temporal robust representation learning for transportation activity recognition .
 
 Pattern Recognition .
 
 Cited by: §V-C .
 

 [184] 
 X. Jiang, E. N. de Souza, A. Pesaranghader, B. Hu, D. L. Silver, and S. Matwin (2017) 
 
 TrajectoryNet: an embedded gps trajectory representation for point-based classification using recurrent neural networks .
 
 In Proceedings of the 27th Annual International Conference on Computer Science and Software Engineering ,
 
 Cited by: §V-C ,
 TABLE III .
 

 [185] 
 H. Liu and I. Lee (2017) 
 
 End-to-end trajectory transportation mode classification using bi-lstm recurrent neural network .
 
 In 2017 12th International Conference on Intelligent Systems and Knowledge Engineering (ISKE) ,
 
 Cited by: §V-C .
 

 [186] 
 H. Liu, H. Wu, W. Sun, and I. Lee (2019) 
 
 Spatio-temporal gru for trajectory classification .
 
 In Proc. of ICDM ,
 
 Cited by: §V-C ,
 TABLE III .
 

 [187] 
 Y. Liang, K. Ouyang, H. Yan, Y. Wang, Z. Tong, and R. Zimmermann (2021) 
 
 Modeling trajectories with neural ordinary differential equations. .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-C ,
 TABLE III .
 

 [188] 
 I. Kontopoulos, A. Makris, K. Tserpes, and V. Bogorny (2022) 
 
 TraClets: harnessing the power of computer vision for trajectory classification .
 
 arXiv preprint arXiv:2205.13880 .
 
 Cited by: §V-C ,
 TABLE III .
 

 [189] 
 J. Zeng, Y. Yu, Y. Chen, D. Yang, L. Zhang, and D. Wang (2023) 
 
 Trajectory-as-a-sequence: a novel travel mode identification framework .
 
 Transportation Research Part C: Emerging Technologies .
 
 Cited by: §V-C .
 

 [190] 
 Y. Liang, K. Ouyang, Y. Wang, X. Liu, H. Chen, J. Zhang, Y. Zheng, and R. Zimmermann (2022) 
 
 TrajFormer: efficient trajectory classification with transformers .
 
 In Proc. of CIKM ,
 
 Cited by: §V-C ,
 TABLE III ,
 TABLE IV .
 

 [191] 
 G. Jiang, S. Lam, P. He, C. Ou, and D. Ai (2020) 
 
 A multi-scale attributes attention model for transport mode identification .
 
 IEEE TITS .
 
 Cited by: §V-C .
 

 [192] 
 W. Yu and G. Wang (2023) 
 
 Graph based embedding learning of trajectory data for transportation mode recognition by fusing sequence and dependency relations .
 
 IJGIS .
 
 Cited by: §V-C .
 

 [193] 
 Q. Gao, F. Zhou, K. Zhang, G. Trajcevski, X. Luo, and F. Zhang (2017) 
 
 Identifying human mobility via trajectory embeddings. .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-C ,
 TABLE III .
 

 [194] 
 F. Zhou, Q. Gao, G. Trajcevski, K. Zhang, T. Zhong, and F. Zhang (2018) 
 
 Trajectory-user linking via variational autoencoder. .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-C ,
 TABLE III .
 

 [195] 
 C. Miao, J. Wang, H. Yu, W. Zhang, and Y. Qi (2020) 
 
 Trajectory-user linking with attentive recurrent network .
 
 In Proceedings of AAMAS ,
 
 Cited by: §V-C ,
 TABLE III .
 

 [196] 
 W. Chen, S. Li, C. Huang, Y. Yu, Y. Jiang, and J. Dong (2022) 
 
 Mutual distillation learning network for trajectory-user linking .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-C ,
 §V-C ,
 TABLE III .
 

 [197] 
 Q. Gao, F. Zhang, F. Yao, A. Li, L. Mei, and F. Zhou (2020) 
 
 Adversarial mobility learning for human trajectory classification .
 
 IEEE Access .
 
 Cited by: §V-C ,
 TABLE III .
 

 [198] 
 F. Zhou, Y. Dai, Q. Gao, P. Wang, and T. Zhong (2021) 
 
 Self-supervised human mobility learning for next location prediction and trajectory classification .
 
 Knowledge-Based Systems .
 
 Cited by: §V-C .
 

 [199] 
 J. James (2020) 
 
 Semi-supervised deep ensemble learning for travel mode identification .
 
 Transportation Research Part C: Emerging Technologies .
 
 Cited by: §V-C .
 

 [200] 
 Y. Zhu, Y. Liu, J. J. Q. Yu, and X. Yuan (2021) 
 
 Semi-supervised federated learning for travel mode identification from gps trajectories .
 
 IEEE TITS .
 
 Cited by: §V-C .
 

 [201] 
 C. Markos and J. James (2020) 
 
 Unsupervised deep learning for gps-based transportation mode identification .
 
 In 2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC) ,
 
 Cited by: §V-C .
 

 [202] 
 Y. Zhu, C. Markos, and J. James (2021) 
 
 Improving transportation mode identification with limited gps trajectories .
 
 In Proc. of ICTAI ,
 
 Cited by: §V-C .
 

 [203] 
 Z. Jiang, A. Huang, G. Qi, and W. Guan (2023) 
 
 A framework of travel mode identification fusing deep learning and map-matching algorithm .
 
 IEEE TITS .
 
 Cited by: §V-C .
 

 [204] 
 L. Deng, H. Sun, Y. Zhao, S. Liu, and K. Zheng (2023) 
 
 S2tul: a semi-supervised framework for trajectory-user linking .
 
 In Proc. of WSDM ,
 
 Cited by: §V-C .
 

 [205] 
 J. Feng, M. Zhang, H. Wang, Z. Yang, C. Zhang, Y. Li, and D. Jin (2019) 
 
 Dplink: user identity linkage via deep neural network from heterogeneous mobility data .
 
 In Proc. of WWW ,
 
 Cited by: §V-C .
 

 [206] 
 H. Huang, F. Ding, H. Yin, G. Liu, C. Wang, and D. O. Wu (2023) 
 
 EgoMUIL: enhancing spatio-temporal user identity linkage in location-based social networks with ego-mo hypergraph .
 
 IEEE TMC .
 
 Cited by: §V-C .
 

 [207] 
 W. Chen, C. Huang, Y. Yu, Y. Jiang, and J. Dong (2024) 
 
 Trajectory-user linking via hierarchical spatio-temporal attention networks .
 
 ACM Transactions on Knowledge Discovery from Data .
 
 Cited by: §V-C .
 

 [208] 
 Z. Shen, H. Yuan, X. Mao, C. Lv, S. Guo, Y. Lin, and H. Wan (2026) 
 
 Towards an efficient and effective en route travel time estimation framework .
 
 In Database Systems for Advanced Applications ,
 
 Cited by: §V-D .
 

 [209] 
 D. Wang, J. Zhang, W. Cao, J. Li, and Y. Zheng (2018) 
 
 When will you arrive? estimating travel time based on deep neural networks .
 
 In Proc. of AAAI ,
 
 Cited by: §V-D ,
 §V-D ,
 TABLE IV .
 

 [210] 
 H. Zhang, H. Wu, W. Sun, and B. Zheng (2018) 
 
 Deeptravel: a neural network based travel time estimation model with auxiliary supervision .
 
 arXiv preprint arXiv:1802.02147 .
 
 Cited by: §V-D ,
 TABLE IV .
 

 [211] 
 Y. Li, K. Fu, Z. Wang, C. Shahabi, J. Ye, and Y. Liu (2018) 
 
 Multi-task representation learning for travel time estimation .
 
 In Proc. of KDD ,
 
 Cited by: §V-D ,
 TABLE IV .
 

 [212] 
 Z. Wang, K. Fu, and J. Ye (2018) 
 
 Learning to estimate the travel time .
 
 In Proc. of KDD ,
 
 Cited by: §V-D ,
 §V-D ,
 TABLE IV .
 

 [213] 
 T. Fu and W. Lee (2019) 
 
 Deepist: deep image-based spatio-temporal network for travel time estimation .
 
 In Proc. of CIKM ,
 
 Cited by: §V-D ,
 TABLE IV .
 

 [214] 
 X. Fang, J. Huang, F. Wang, L. Zeng, H. Liang, and H. Wang (2020) 
 
 Constgat: contextual spatial-temporal graph attention network for travel time estimation at baidu maps .
 
 In Proc. of KDD ,
 
 Cited by: §V-D ,
 TABLE IV .
 

 [215] 
 H. Hong, Y. Lin, X. Yang, Z. Li, K. Fu, Z. Wang, X. Qie, and J. Ye (2020) 
 
 HetETA: heterogeneous information network embedding for estimating time of arrival .
 
 In Proc. of KDD ,
 
 Cited by: §V-D ,
 TABLE IV .
 

 [216] 
 K. Fu, F. Meng, J. Ye, and Z. Wang (2020) 
 
 Compacteta: a fast inference system for travel time prediction .
 
 In Proc. of KDD ,
 
 Cited by: §V-D ,
 TABLE IV .
 

 [217] 
 X. Fang, J. Huang, F. Wang, L. Liu, Y. Sun, and H. Wang (2021) 
 
 Ssml: self-supervised meta-learner for en route travel time estimation at baidu maps .
 
 In Proc. of KDD ,
 
 Cited by: §V-D ,
 TABLE IV .
 

 [218] 
 Y. Ye, Y. Zhu, C. Markos, and J.Q. Y. James (2022) 
 
 Cateta: a categorical approximate approach for estimating time of arrival .
 
 IEEE TITS .
 
 Cited by: §V-D ,
 §V-D ,
 TABLE IV .
 

 [219] 
 F. Liu, D. Wang, and Z. Xu (2021) 
 
 Privacy-preserving travel time prediction with uncertainty using gps trace data .
 
 IEEE TMC .
 
 Cited by: §V-D ,
 TABLE IV .
 

 [220] 
 H. Liu, W. Jiang, S. Liu, and X. Chen (2023) 
 
 Uncertainty-aware probabilistic travel time prediction for on-demand ride-hailing at didi .
 
 In Proc. of KDD ,
 
 Cited by: §V-D ,
 TABLE IV .
 

 [221] 
 J. James (2021) 
 
 Citywide estimation of travel time distributions with bayesian deep graph learning .
 
 IEEE TKDE .
 
 Cited by: §V-D ,
 TABLE IV .
 

 [222] 
 J. Wang, Q. Gu, J. Wu, G. Liu, and Z. Xiong (2016) 
 
 Traffic speed prediction and congestion source exploration: a deep learning method .
 
 In Proc. of ICDM ,
 
 Cited by: §V-D .
 

 [223] 
 Y. Shen, C. Jin, J. Hua, and D. Huang (2020) 
 
 TTPNet: a neural network for travel time prediction based on tensor decomposition and graph embedding .
 
 IEEE TKDE .
 
 Cited by: §V-D .
 

 [224] 
 L. Huang, Y. Yang, H. Chen, Y. Zhang, Z. Wang, and L. He (2022) 
 
 Context-aware road travel time estimation by coupled tensor decomposition based on trajectory data .
 
 Knowledge-Based Systems .
 
 Cited by: §V-D .
 

 [225] 
 Y. Sun, K. Fu, Z. Wang, C. Zhang, and J. Ye (2021) 
 
 Road network metric learning for estimated time of arrival .
 
 In Proc. of ICPR ,
 
 Cited by: §V-D .
 

 [226] 
 Y. Sun, K. Fu, Z. Wang, D. Zhou, K. Wu, J. Ye, and C. Zhang (2020) 
 
 CoDriver eta: combine driver information in estimated time of arrival by driving style learning auxiliary task .
 
 IEEE TITS .
 
 Cited by: §V-D .
 

 [227] 
 Z. Chen, X. Xiao, Y. Gong, J. Fang, N. Ma, H. Chai, and Z. Cao (2022) 
 
 Interpreting trajectories from multiple views: a hierarchical self-attention network for estimating the time of arrival .
 
 In Proc. of KDD ,
 
 Cited by: §V-D .
 

 [228] 
 G. Jin, H. Yan, F. Li, Y. Li, and J. Huang (2023) 
 
 Dual graph convolution architecture search for travel time estimation .
 
 ACM Transactions on Intelligent Systems and Technology .
 
 Cited by: §V-D .
 

 [229] 
 H. Yuan, G. Li, and Z. Bao (2022) 
 
 Route travel time estimation on a road network revisited: heterogeneity, proximity, periodicity and dynamicity .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §V-D .
 

 [230] 
 Y. Fan, J. Xu, R. Zhou, J. Li, K. Zheng, L. Chen, and C. Liu (2022) 
 
 MetaER-tte: an adaptive meta-learning model for en route travel time estimation .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-D .
 

 [231] 
 Y. Zhu, Y. Ye, Y. Liu, and J. James (2022) 
 
 Cross-area travel time uncertainty estimation from trajectory data: a federated learning approach .
 
 IEEE TITS .
 
 Cited by: §V-D ,
 §VIII-B .
 

 [232] 
 W. Zhou, X. Xiao, Y. Gong, J. Chen, J. Fang, N. Tan, N. Ma, Q. Li, C. Hua, S. Jeon, et al. (2023) 
 
 Travel time distribution estimation by learning representations over temporal attributed graphs .
 
 IEEE TITS .
 
 Cited by: §V-D .
 

 [233] 
 J. Lee, J. Han, and X. Li (2008) 
 
 Trajectory outlier detection: A Partition-and-Detect Framework .
 
 In ICDE ,
 
 Cited by: §V-E .
 

 [234] 
 L. Song, R. Wang, D. Xiao, X. Han, Y. Cai, and C. Shi (2018) 
 
 Anomalous Trajectory Detection using Recurrent Neural Network .
 
 In Advanced Data Mining and Applications ,
 
 Cited by: §V-E .
 

 [235] 
 D. Smolyak, K. Gray, S. Badirli, and G. Mohler (2020) 
 
 Coupled igmm-gans with applications to anomaly detection in human mobility data .
 
 ACM Transactions on Spatial Algorithms and Systems (TSAS) .
 
 Cited by: §V-E .
 

 [236] 
 Y. Su, D. Yao, X. Zhou, Y. Zhang, Y. Fan, L. Bai, and J. Bi (2023) 
 
 TripSafe: Retrieving Safety-related Abnormal Trips in Real-time with Trajectory Data .
 
 In Proc. of SIGIR ,
 
 Cited by: §V-E .
 

 [237] 
 Q. Gao, X. Wang, C. Liu, G. Trajcevski, L. Huang, and F. Zhou (2023) 
 
 Open anomalous trajectory recognition via probabilistic metric learning .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-E .
 

 [238] 
 H. Wu, W. Sun, and B. Zheng (2017) 
 
 A Fast Trajectory Outlier Detection Approach via Driving Behavior Modeling .
 
 In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management ,
 
 Cited by: §V-E .
 

 [239] 
 Q. Zhang, Z. Wang, C. Long, C. Huang, S. Yiu, Y. Liu, G. Cong, and J. Shi (2023) 
 
 Online Anomalous Subtrajectory Detection on Road Networks with Deep Reinforcement Learning .
 
 In ICDE ,
 
 Cited by: §V-E .
 

 [240] 
 Y. Liu, K. Zhao, G. Cong, and Z. Bao (2020) 
 
 Online Anomalous Trajectory Detection with Deep Generative Sequence Modeling .
 
 In ICDE ,
 
 Cited by: §V-E .
 

 [241] 
 X. Han, R. Cheng, C. Ma, and T. Grubenmann (2022) 
 
 DeepTEA: Effective and Efficient Online Time-dependent Trajectory Outlier Detection .
 
 Proceedings of the VLDB Endowment .
 
 Cited by: §V-E .
 

 [242] 
 A. Hess, K. A. Hummel, W. N. Gansterer, and G. Haring (2015) 
 
 Data-driven human mobility modeling: a survey and engineering guidance for mobile networking .
 
 ACM Computing Surveys .
 
 Cited by: §V-F .
 

 [243] 
 H. Barbosa, M. Barthelemy, G. Ghoshal, C. R. James, M. Lenormand, T. Louail, R. Menezes, J. J. Ramasco, F. Simini, and M. Tomasini (2018) 
 
 Human mobility: models and applications .
 
 Physics Reports .
 
 Cited by: §V-F .
 

 [244] 
 C. Chen, K. Li, S. G. Teo, X. Zou, K. Li, and Z. Zeng (2020) 
 
 Citywide traffic flow prediction based on multiple gated spatio-temporal convolutional neural networks .
 
 ACM TKDD .
 
 Cited by: §V-F .
 

 [245] 
 C. Wu, L. Chen, G. Wang, S. Chai, H. Jiang, J. Peng, and Z. Hong (2020) 
 
 Spatiotemporal scenario generation of traffic flow based on lstm-gan .
 
 IEEE Access .
 
 Cited by: §V-F .
 

 [246] 
 Y. Zhang, S. Wang, B. Chen, J. Cao, and Z. Huang (2019) 
 
 Trafficgan: network-scale deep traffic prediction with generative adversarial nets .
 
 IEEE TITS .
 
 Cited by: §V-F .
 

 [247] 
 Y. Chen, Y. Lv, and F. Wang (2019) 
 
 Traffic flow imputation using parallel data and generative adversarial networks .
 
 IEEE TITS .
 
 Cited by: §V-F .
 

 [248] 
 D. Yin and Q. Yang (2018) 
 
 GANs based density distribution privacy-preservation on mobility data .
 
 Security and Communication Networks .
 
 Cited by: §V-F .
 

 [249] 
 X. Song, H. Kanasugi, and R. Shibasaki (2016) 
 
 Deeptransport: prediction and simulation of human mobility and transportation mode at a citywide level .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-F .
 

 [250] 
 Y. Hong, H. Martin, and M. Raubal (2022) 
 
 How do you go where? improving next location prediction by learning travel mode information using transformers .
 
 In Proceedings of the 30th International Conference on Advances in Geographic Information Systems ,
 
 Cited by: §V-F .
 

 [251] 
 M. E. Nergiz, M. Atzori, and Y. Saygin (2008) 
 
 Towards trajectory anonymization: a generalization-based approach .
 
 In Proceedings of the SIGSPATIAL ACM GIS 2008 International Workshop on Security and Privacy in GIS and LBS ,
 
 Cited by: §V-F .
 

 [252] 
 K. Ouyang, R. Shokri, D. S. Rosenblum, and W. Yang (2018) 
 
 A non-parametric generative model for human trajectories .
 
 In Proc. of IJCAI ,
 
 Cited by: §V-F .
 

 [253] 
 X. Wang, X. Liu, Z. Lu, and H. Yang (2021) 
 
 Large scale gps trajectory generation using map based on two stage gan .
 
 Journal of Data Science .
 
 Cited by: §V-F .
 

 [254] 
 N. Xu, L. Trinh, S. Rambhatla, Z. Zeng, J. Chen, S. Assefa, and Y. Liu (2021) 
 
 Simulating continuous-time human mobility trajectories .
 
 In Proc. 9th Int. Conf. Learn. Represent ,
 
 Cited by: §V-F .
 

 [255] 
 J. Feng, Z. Yang, F. Xu, H. Yu, M. Wang, and Y. Li (2020) 
 
 Learning to simulate human mobility .
 
 In Proc. of KDD ,
 
 Cited by: §V-F .
 

 [256] 
 Y. Wang, T. Zheng, Y. Liang, S. Liu, and M. Song (2024) 
 
 COLA: cross-city mobility transformer for human trajectory simulation .
 
 World Wide Web .
 
 Cited by: §V-F .
 

 [257] 
 Y. Yuan, H. Wang, J. Ding, D. Jin, and Y. Li (2023) 
 
 Learning to simulate daily activities via modeling dynamic human needs .
 
 In Proceedings of the ACM Web Conference 2023 ,
 
 Cited by: §V-F .
 

 [258] 
 C. Cao and M. Li (2021) 
 
 Generating mobility trajectories with retained data utility .
 
 In Proc. of KDD ,
 
 Cited by: §V-F .
 

 [259] 
 Y. Zhu, Y. Ye, S. Zhang, X. Zhao, and J. Yu (2023) 
 
 DiffTraj: generating GPS trajectory with diffusion probabilistic model .
 
 In Proc. of NeurIPS ,
 
 Cited by: §V-F .
 

 [260] 
 T. Wei, Y. Lin, S. Guo, Y. Lin, Y. Huang, C. Xiang, Y. Bai, M. Ya, and H. Wan (2024) 
 
 Diff-rntraj: a structure-aware diffusion model for road network-constrained trajectory generation .
 
 arXiv preprint arXiv:2402.07369 .
 
 Cited by: §V-F .
 

 [261] 
 Y. Wu, X. Chen, and D. Zhuang (2025) 
 
 Towards foundation model for spatiotemporal data analysis .
 
 In Proceedings of the 19th International Symposium on Spatial and Temporal Data ,
 
 Cited by: §VI-A .
 

 [262] 
 Y. Zhu, J. J. Yu, X. Zhao, X. Zhou, L. Han, X. Wei, and Y. Liang (2025) 
 
 Unitraj: learning a universal trajectory foundation model from billion-scale worldwide traces .
 
 In Proc. of NeurIPS ,
 
 Cited by: §VI-A ,
 §VI-A .
 

 [263] 
 C. Han, Y. Yuan, K. Chen, J. Ding, and Y. Li (2025) 
 
 MoveGPT: scaling mobility foundation models with spatially-aware mixture of experts .
 
 arXiv preprint arXiv:2505.18670 .
 
 Cited by: §VI-A ,
 §VI-A .
 

 [264] 
 C. Han, Y. Yuan, Y. Liu, J. Ding, J. Feng, and Y. Li (2025) 
 
 UniMove: a unified model for multi-city human mobility prediction .
 
 arXiv preprint arXiv:2508.06986 .
 
 Cited by: §VI-A ,
 §VI-A .
 

 [265] 
 Y. Lin, T. Wei, Z. Zhou, H. Wen, J. Hu, S. Guo, Y. Lin, and H. Wan (2024) 
 
 TrajFM: a vehicle trajectory foundation model for region and task transferability .
 
 arXiv preprint arXiv:2408.15251 .
 
 Cited by: §VI-A .
 

 [266] 
 X. Yu, J. Wang, Y. Yang, Q. Huang, and K. Qu (2025) 
 
 BIGCity: a universal spatiotemporal model for unified trajectory and traffic state data analysis .
 
 In ICDE ,
 
 Cited by: §VI-A .
 

 [267] 
 Y. Yuan, Y. Liu, C. Han, J. Feng, and Y. Li (2025) 
 
 Breaking data silos: towards open and scalable mobility foundation models via generative continual learning .
 
 arXiv preprint arXiv:2506.06694 .
 
 Cited by: §VI-A ,
 §VI-A .
 

 [268] 
 T. Wei, Y. Lin, Y. Lin, S. Guo, J. Hu, H. Yuan, G. Cong, and H. Wan (2024) 
 
 PLMTrajRec: a scalable and generalizable trajectory recovery method with pre-trained language models .
 
 Proc. of ICML .
 
 Cited by: 1st item .
 

 [269] 
 Z. Lan, L. Liu, B. Fan, Y. Lv, Y. Ren, and Z. Cui (2024) 
 
 Traj-llm: a new exploration for empowering trajectory prediction with pre-trained large language models .
 
 IEEE Transactions on Intelligent Vehicles .
 
 Cited by: 1st item .
 

 [270] 
 T. Liu, M. Li, and Y. Yin (2025) 
 
 Aligning llm with human travel choices: a persona-based embedding learning approach .
 
 arXiv preprint arXiv:2505.19003 .
 
 Cited by: 2nd item .
 

 [271] 
 S. Liu, D. Yao, Y. Lin, G. Cong, and J. Bi (2025) 
 
 Traj-mllm: can multimodal large language models reform trajectory data mining? .
 
 arXiv preprint arXiv:2509.00053 .
 
 Cited by: 1st item .
 

 [272] 
 Y. Zhu, J. J. Yu, X. Zhao, X. Han, Q. Liu, X. Wei, and Y. Liang (2025) 
 
 Learning generalized and flexible trajectory models from omni-semantic supervision .
 
 In Proc. of KDD ,
 
 Cited by: 1st item .
 

 [273] 
 H. He, H. Luo, Y. Chen, and Q. R. Wang (2025) 
 
 RHYTHM: reasoning with hierarchical temporal tokenization for human mobility .
 
 arXiv preprint arXiv:2509.23115 .
 
 Cited by: 2nd item .
 

 [274] 
 T. Nie, J. He, Y. Mei, G. Qin, G. Li, J. Sun, and W. Ma (2025) 
 
 Joint estimation and prediction of city-wide delivery demand: a large language model empowered graph-based learning approach .
 
 Transportation Research Part E: Logistics and Transportation Review .
 
 Cited by: 3rd item .
 

 [275] 
 Z. Zhou, Y. Lin, H. Wen, Q. Xu, S. Guo, J. Hu, Y. Lin, and H. Wan (2024) 
 
 TrajCogn: leveraging llms for cognizing movement patterns and travel purposes from trajectories .
 
 arXiv preprint arXiv:2405.12459 .
 
 Cited by: 3rd item .
 

 [276] 
 Y. Du, J. Feng, J. Zhao, and Y. Li 
 
 TrajAgent: an llm-agent framework for trajectory modeling via large-and-small model collaboration .
 
 In Proc. of NeurIPS ,
 
 Cited by: 1st item .
 

 [277] 
 J. Feng, Y. Du, J. Zhao, and Y. Li (2025) 
 
 Agentmove: a large language model based agentic framework for zero-shot next location prediction .
 
 In Proc. of ACL ,
 
 Cited by: 2nd item .
 

 [278] 
 Y. Liu, X. Liao, H. Ma, J. Liu, R. Jadhav, and J. Ma (2025) 
 
 MobiVerse: scaling urban mobility simulation with hybrid lightweight domain-specific generator and large language models .
 
 IEEE ITSC .
 
 Cited by: 3rd item .
 

 [279] 
 Y. Du, J. Feng, J. Yuan, and Y. Li (2025) 
 
 CAMS: a citygpt-powered agentic framework for urban human mobility simulation .
 
 arXiv preprint arXiv:2506.13599 .
 
 Cited by: 3rd item .
 

 [280] 
 H. Asano, H. Ouchi, A. Kasuga, and R. Yonetani (2025) 
 
 MobQA: a benchmark dataset for semantic understanding of human mobility data through question answering .
 
 arXiv preprint arXiv:2508.11163 .
 
 Cited by: §VI-B .
 

 [281] 
 Q. Zhang, Z. Wang, C. Long, C. Huang, S. Yiu, Y. Liu, G. Cong, and J. Shi (2023) 
 
 Online anomalous subtrajectory detection on road networks with deep reinforcement learning .
 
 In ICDE ,
 
 Cited by: §VII-A .
 

 [282] 
 S. Ma, Y. Zheng, and O. Wolfson (2013) 
 
 T-share: a large-scale dynamic taxi ridesharing service .
 
 In ICDE ,
 
 Cited by: §VII-A .
 

 [283] 
 J. Xie, K. Zhang, J. Chen, T. Zhu, R. Lou, Y. Tian, Y. Xiao, and Y. Su (2024) 
 
 Travelplanner: a benchmark for real-world planning with language agents .
 
 arXiv preprint arXiv:2402.01622 .
 
 Cited by: §VII-A .
 

 [284] 
 Z. Zou, Z. Yu, and K. Cao (2017) 
 
 An innovative gps trajectory data based model for geographic recommendation service .
 
 Transactions in GIS .
 
 Cited by: §VII-A .
 

 [285] 
 L. Wu, H. Wen, H. Hu, X. Mao, Y. Xia, E. Shan, J. Zhen, J. Lou, Y. Liang, L. Yang, et al. (2023) 
 
 LaDe: the first comprehensive last-mile delivery dataset from industry .
 
 arXiv preprint arXiv:2306.10675 .
 
 Cited by: §VII-A ,
 TABLE V .
 

 [286] 
 L. Wang, Z. Yu, D. Yang, H. Ma, and H. Sheng (2019) 
 
 Efficiently targeted billboard advertising using crowdsensing vehicle trajectory data .
 
 IEEE Transactions on Industrial Informatics .
 
 Cited by: §VII-A .
 

 [287] 
 Y. Zheng, L. Zhang, X. Xie, and W. Ma (2009) 
 
 Mining interesting locations and travel sequences from gps trajectories .
 
 In WWW ,
 
 Cited by: TABLE V .
 

 [288] 
 J. Yuan, Y. Zheng, X. Xie, and G. Sun (2011) 
 
 Driving with knowledge from the physical world .
 
 In Proc. of KDD ,
 
 Cited by: TABLE V .
 

 [289] 
 Z. Xu, Y. Yin, C. Dai, X. Huang, R. Kudali, J. Foflia, G. Wang, and R. Zimmermann (2020) 
 
 Grab-posisi-l: a labelled gps trajectory dataset for map matching in southeast asia .
 
 In Proceedings of the 28th International Conference on Advances in Geographic Information Systems ,
 
 Cited by: TABLE V .
 

 [290] 
 B. Chang, Y. Park, D. Park, S. Kim, and J. Kang (2018) 
 
 Content-aware hierarchical point-of-interest embedding model for successive poi recommendation. .
 
 In Proc. of IJCAI ,
 
 Cited by: TABLE V .
 

 [291] 
 C. Zhang, K. Zhang, Q. Yuan, L. Zhang, T. Hanratty, and J. Han (2016) 
 
 Gmove: group-level mobility modeling using geo-tagged social media .
 
 In Proc. of KDD ,
 
 Cited by: TABLE V .
 

 [292] 
 Y. Zhu, Y. Ye, Y. Wu, X. Zhao, and J. Yu (2023) 
 
 SynMob: creating high-fidelity synthetic gps trajectory dataset for urban mobility analysis .
 
 Proc. of NeurIPS .
 
 Cited by: TABLE V .
 

 [293] 
 W. Jiang (2022) 
 
 TaxiBJ21: an open crowd flow dataset based on beijing taxi gps trajectories .
 
 Internet Technology Letters .
 
 Cited by: TABLE V .
 

 [294] 
 J. Ma, J. Chan, S. Rajasegarar, G. Ristanoski, and C. Leckie (2020) 
 
 Multi-attention 3d residual neural network for origin-destination crowd flow prediction .
 
 In Proc. of ICDM ,
 
 Cited by: §VII-A .
 

 [295] 
 J. Dai, B. Yang, C. Guo, and Z. Ding (2015) 
 
 Personalized route recommendation using big trajectory data .
 
 In ICDE ,
 
 Cited by: §VII-A .
 

 [296] 
 M. Levitt, A. Scaiewicz, and F. Zonta (2020) 
 
 Predicting the trajectory of any covid19 epidemic from the best straight line .
 
 medRxiv .
 
 Cited by: §VII-A .
 

 [297] 
 R. Wu, G. Luo, J. Shao, L. Tian, and C. Peng (2018) 
 
 Location prediction on trajectory data: a review .
 
 Big data mining and analytics .
 
 Cited by: §VII-A .
 

 [298] 
 J. Ji, W. Zhang, J. Wang, Y. He, and C. Huang (2023) 
 
 Self-supervised deconfounding against spatio-temporal shifts: theory and modeling .
 
 arXiv preprint arXiv:2311.12472 .
 
 Cited by: §VIII-B .
 

 [299] 
 X. Zou, Y. Yan, X. Hao, Y. Hu, H. Wen, E. Liu, J. Zhang, Y. Li, T. Li, Y. Zheng, et al. (2024) 
 
 Deep learning for cross-domain data fusion in urban computing: taxonomy, advances, and outlook .
 
 arXiv preprint arXiv:2402.19348 .
 
 Cited by: §VIII-B .
 

 [300] 
 W. Chen, X. Hao, Y. Wu, and Y. Liang (2024) 
 
 Terra: a multimodal spatio-temporal dataset spanning the earth .
 
 NeurIPS 37 .
 
 Cited by: §VIII-B .
 

 [301] 
 Y. Yan, H. Wen, S. Zhong, W. Chen, H. Chen, Q. Wen, R. Zimmermann, and Y. Liang (2023) 
 
 When urban region profiling meets large language models .
 
 arXiv preprint arXiv:2310.18340 .
 
 Cited by: §VIII-B .
 

 [302] 
 M. Jin, Y. Zhang, W. Chen, K. Zhang, Y. Liang, B. Yang, J. Wang, S. Pan, and Q. Wen (2024) 
 
 Position paper: what can large language models tell us about time series analysis .
 
 Proc. of ICML .
 
 Cited by: §VIII-B .
 

 [303] 
 K. Luo, Y. Zhu, W. Chen, K. Wang, Z. Zhou, S. Ruan, and Y. Liang (2024) 
 
 Towards robust trajectory representations: isolating environmental confounders with causal learning .
 
 IJCAI .
 
 Cited by: §VIII-B .