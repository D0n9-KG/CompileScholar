From Pixels to Insights:A Survey on Automatic Chart Understandingin the Era of Large Foundation Models 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2403.12027v4 [cs.CL] 05 Dec 2024 
 
 

# From Pixels to Insights:
 A Survey on Automatic Chart Understanding
 in the Era of Large Foundation Models Thanks: 
Kung-Hsiang Huang: Salesforce AI Research. kh.huang@salesforce.com.
Hou Pong Chan, May Fung, Heng Ji: University of Illinois Urbana-Champaign. {hpchan, yifung2, hengji}@illinois.edu.
Haoyi Qiu: University of California, Los Angeles. haoyiqiu@cs.ucla.edu.
Mingyang Zhou: Capital One. mingyang.zhou@capitalone.com.
Shafiq Joty: Salesforce AI Research NTU. sjoty@salesforce.com.
Shih-Fu Chang: Columbia University. sc250@columbia.edu.
 

 
 
 Kung-Hsiang Huang
 
    
 Hou Pong Chan
 
    
 May Fung
 
    
 Haoyi Qiu
 
 Affiliation: Mingyang Zhou,
Shafiq Joty,
Shih-Fu Chang,
Heng Ji

 

 Abstract 
 
 Data visualization in the form of charts plays a pivotal role in data analysis, offering critical insights and aiding in informed decision-making. Automatic chart understanding has witnessed significant advancements with the rise of large foundation models in recent years. Foundation models, such as large language models, have revolutionized various natural language processing tasks and are increasingly being applied to chart understanding tasks. This survey paper provides a comprehensive overview of the recent developments, challenges, and future directions in chart understanding within the context of these foundation models. We review fundamental building blocks crucial for studying chart understanding tasks. Additionally, we explore various tasks and their evaluation metrics and sources of both charts and textual inputs. Various modeling strategies are then examined, encompassing both classification-based and generation-based approaches, along with tool augmentation techniques that enhance chart understanding performance. Furthermore, we discuss the state-of-the-art performance of each task and discuss how we can improve the performance. Challenges and future directions are addressed, highlighting the importance of several topics, such as domain-specific charts, lack of efforts in developing evaluation metrics, and agent-oriented settings.
This survey paper aims to provide valuable insights and directions for future research in chart understanding leveraging large foundation models.

 
 
 
 Index Terms: Chart understanding, chart question answering, chart captioning, multimodality, large vision-language models, knowledgeable foundation model reasoning

 
 

## I Introduction 

 
 The Significance of Chart Understanding in Information Communication  In our contemporary world of multimedia information, where the volume and complexity of data continue to burgeon, the role of charts stands paramount in facilitating effective communication of factual information, conveying insights, and informing decision making. Across diverse fields spanning academia, scientific research, digital media, and business realms, charts serve as indispensable tools for translating raw data into comprehensible visual narratives. Their ability to encapsulate complex datasets in a concise and intuitive format empowers decision-makers to grasp key insights swiftly, aiding informed reasoning and strategic planning. Recognizing this pivotal role of charts in modern information dissemination, there has been a sustained interest within the computational community, as evidenced by a plethora of research in automatic chart understanding . In particular, works on chart question answering [ 1 , 2 , 3 , 4 , 5 ] , chart captioning [ 6 , 7 , 8 , 9 ] , chart-to-table conversion [ 10 , 11 ] , chart fact-checking [ 12 , 13 ] , chart caption factual error correction [ 14 ] have laid foundational frameworks for exploring the intricacies of chart semantics in chart understanding technologies.

 
 
 Challenges and Opportunities in Chart Understanding Amidst the Era of Large Foundation Models  Traditional chart understanding work [ 15 , 7 ] focuses on finetuning methods that generally experience limitations with respect to domain portability and reasoning robustness. Excitingly, the advent of large vision-language foundation models (e.g., GPT-4V [ 16 ] , LLaVA [ 17 ] ) has spawned a paradigm shift in automated reasoning capabilities, catalyzing unprecedented advances across various multimedia cognitive tasks – including progress in strong zero/few-shot reasoning capabilities through text-based prompting. Despite the impressive improvements, the domain of chart understanding still presents major challenges. Charts present a unique set of obstacles owing to their rich visual representations and nuanced semantics, delivered through complex and creative organization of information manifested in text, numerical data, plots, and graphics objects. From bar charts and line graphs to pie charts and scatter plots, each chart type employs a distinct visual syntax to convey data relationships, requiring sophisticated interpretative mechanisms beyond mere pixel-level pattern recognition. Charts serve as a means to uncover important insights, such as emerging trends, outliers that challenge assumptions, and relationships among variables that are not immediately apparent from raw data in tabular form.
They enable comparative analyses across data points, providing a visual platform for summarizing varied entities or time periods clearly.
Furthermore, the intrinsic diversity of underlying datasets, ranging from simple numerical relations to intricate multidimensional entities, adds another layer of complexity to the chart understanding task. Despite these challenges, automated chart understanding offers a gateway to unlock actionable insights buried within the pixels of visual narratives. By harnessing the capabilities of large foundation models, chart understanding demonstrates improved potential in bridging the gap between raw visual data and meaningful insights, thereby enabling high-level information extraction, logical reasoning, and decision-making.

 
 
 Several studies have surveyed the landscape of chart understanding research, yet these surveys often show limitations in covering comprehensive scopes or reviewing information of sufficient specificity.
Some surveys do not cover modern datasets employed in chart understanding research, as well as the most contemporary modeling approaches, such as those involving pre-trained vision-language models and large foundation models [ 18 , 19 ] . Conversely, other surveys concentrate predominantly on the visualization aspect (i.e. the transformation of data into charts), thereby missing the nuanced task of chart interpretation [ 20 , 21 , 22 , 23 ] . This survey paper aims to bridge these gaps.

 
 
 We first define automatic chart understanding and the fundamental building blocks to the problem formulation in Section   II . We discuss the multifaceted nature of chart understanding, encompassing tasks from interpreting chart visuals to analyzing underlying data. We also illustrate how chart understanding overlaps with related works in natural image understanding [ 24 ] , table understanding [ 25 ] , and document understanding [ 26 ] . We explain the essential modeling components for chart understanding, such as vision encoders, OCR modules, and text decoders, and their role in converting raw chart images and text queries into useful insights.Then, in Section   III , we examine the datasets driving chart understanding research and the metrics for model evaluation. This section analyzes the sources, diversity, and limitations of these datasets, providing insights into the current chart understanding data landscape. It also reviews various evaluation metrics, underlining the necessity for robust and nuanced assessment methods. With insights from these characteristics, we further provide trending popular modeling strategies for automatic chart understanding. Section   IV delves into the diverse modeling strategies in chart understanding, including adaptations from natural image understanding, vision-language pre-training, and foundation models like large language models (LLMs) and large vision-language models (LVLMs). In particular, we emphasize the impact of choices in vision encoders and text decoders on model effectiveness and discuss tool augmentation’s role in chart understanding. We conclude this section by showcasing the state-of-the-art performance on different chart understanding tasks and how we can improve upon them.

 
 
 Finally, Section   V addresses the challenges and future directions in chart understanding. We highlight the importance of domain-specific charts, the need for comprehensive evaluation metrics, and the potential for adversarial settings to enhance model robustness and versatility. This survey paper concludes by identifying key areas for future research, such as developing models for complex charts, refining evaluation metrics, and diversifying datasets. We not only offer an in-depth overview of the current state of chart understanding but also set the stage for future advancements in this exciting intersection of data visualization and machine learning.

 
 
 

## II Background 

 

### II-A What is Automatic Chart Understanding? 

 
 Charts are graphical representations of data that are used to present complex patterns in data in a concise and visually appealing manner. Common types of charts include line charts, bar charts, area charts, pie charts, and scatter plots.
Automatic chart understanding aims to enable machines to interpret charts and derive meaningful information such as patterns, trends, and relationships within the data presented in the charts.
Automatic chart understanding techniques have various real-world applications, such as assisting data analysts in discovering patterns from charts, answering queries, and helping individuals with visual impairments to access the information in charts.

 
 
 However, automatic chart understanding is challenging as it demands substantial perceptual and reasoning effort.
First, charts encompass complex compositions of fine-grained graphical marks (e.g., lines, dots, bars, etc.) and scene text (e.g., axis labels, titles, legends, etc.).
Therefore, perception abilities are required to understand the layout and spatial relations among these elements to extract meaningful information from the chart.
Moreover, to fulfill a specific task or query, models often need to be equipped with reasoning capabilities. These range from basic comparison operations for interpreting the relative magnitude between data points to complex mathematical abilities, such as summation, on the values extracted from charts.

 
 
 

### II-B Related Tasks 

 

#### II-B 1 Natural Image Understanding

 
 Natural image understanding focuses on the interpretation and analysis of photographic images, which capture scenes from the real world. These images can include landscapes, people, objects, and everyday situations. The objective of this field is to enable machines to recognize and make sense of visual information in a manner that replicates human vision. Natural image understanding encompasses tasks such as image captioning [ 27 , 28 ] and visual question answering [ 29 , 30 ] . The core difference between natural image understanding and chart understanding lies in the type of visual content and the nature of the interpretation tasks involved. While natural image understanding deals with unstructured visual data that represents natural scenes, chart understanding focuses on structured visual representations of data, such as bar graphs, line charts, and pie charts. Charts are designed to communicate specific quantitative information and trends. This makes the interpretation task centered around data extraction, recognition of graphical elements, and comprehension of the data narrative conveyed by the chart.

 
 
 One of the major challenges in natural image understanding is the variability and complexity of real-world scenes, including variations in lighting, viewpoints, and occlusions. In contrast, chart understanding entails challenges such as recognizing and differentiating among various chart types, extracting embedded textual and numerical data, and understanding the logical and quantitative relationships depicted. Furthermore, charts can contain a high level of abstraction and symbolic elements (e.g., colors representing different categories, lines indicating trends), necessitating not just visual perception but also mathematical or analytical reasoning to interpret the information accurately.

 
 
 

#### II-B 2 Table Understanding

 
 Table understanding involves interpreting information presented in tabular data [ 31 , 32 , 33 , 34 ] . This task aims to recognize the table’s layout and structure, identify the relationships between its rows and columns, and comprehend the semantic context embedded within the table’s content. In contrast to charts, which employ graphical elements to visually represent data, tables arrange this data in a meticulously organized format. They utilize rows and columns to systematically display information, offering a direct view of the data’s structure and interrelationships.

 
 
 Although chart and table understanding both engage with structured data, chart understanding confronts a unique challenge absent in table understanding: the interpretation of visual encodings. Charts employ diverse visual encodings, including color, shape, size, and orientation, to illustrate and highlight various facets of the data. These visual encodings can convey nuances such as trends, distributions, and comparisons in a way that is more immediately perceptible and sometimes more intuitively understood than textual data. However, this also means that chart understanding requires an additional layer of visual perception and interpretation that is not necessary for tabular data.

 
 
 Furthermore, while it might be arguable that chart understanding tasks can be simplified into table understanding tasks by converting charts into underlying data tables. However, data tables are infrequently published alongside charts, and the process of automatically extracting data tables is fraught with challenges. The diversity of chart types and visual styles, coupled with the intricacies of reasoning about visual occlusion, makes automatic data table extraction prone to errors. This complexity underscores the distinct and intricate nature of chart understanding, highlighting its unique challenges compared to table understanding.

 
 
 

#### II-B 3 Document Understanding

 
 Document understanding encompasses the comprehensive analysis of information contained within documents that are presented visually. These documents can vary widely in format and content, including but not limited to scanned paper documents [ 35 ] , digital PDFs [ 36 ] , presentation slides [ 37 ] . The essence of visual document understanding lies in its ability to decode both the textual content and the layout structure of documents, understanding the integral relationship between various components such as headings, paragraphs, tables, figures, and footnotes.

 
 
 The primary distinction between visual document understanding and chart understanding is the scope and nature of the visual content being analyzed. While chart understanding is specifically focused on interpreting graphical representations of data, such as bar charts, line graphs, and pie charts, visual document understanding encompasses a broader range of document types and aims to comprehend the full gamut of information a document may convey. This includes not only charts but also textual content and their spatial arrangements, which collectively deliver the document’s message.

 
 
 Additionally, charts in the context of chart understanding may originate from diverse sources beyond just traditional documents. They are often found on consensus websites, in digital media, and within academic literature, where they serve as standalone visualizations of data rather than components of a larger document. This distinction highlights the versatility and ubiquity of charts as tools for data representation and the necessity for models in chart understanding to focus specifically on the nuances of visual data interpretation, free from the constraints of document structure.

 
 
 
 

### II-C Problem Formulation and Fundamental Building Blocks 

 
 We uniformly formulate existing chart understanding tasks as the following problem.
Given a chart image 𝐜 \mathbf{c} and a textual query 𝐪 \mathbf{q} , the goal is to predict a textual response 𝐲 \mathbf{y} .
Both the textual query and textual response are sequences of tokens,
i.e., 𝐪 = [ q 1 , … , q l q ] \mathbf{q}=[q_{1},\ldots,q_{l^{q}}] and 𝐲 = [ y 1 , … , y l y ] \mathbf{y}=[y_{1},\ldots,y_{l^{y}}] , where l q l^{q} and l y l^{y} denote the length of 𝐪 \mathbf{q} and 𝐲 \mathbf{y} , respectively.
The response y y can be restricted to a fixed vocabulary, such as the candidate answers in a multiple-choice question or the class labels in the chart fact-checking task.
Alternatively, the response can be open vocabulary in tasks such as open-ended question answering or chart captioning.

 
 
 A chart understanding model is typically built by combining the following computational modules:

 
 
 Vision Encoder :
It is often essential to extract visual features from the input chart image in order to understand the relations and spatial arrangements between the graphical elements and scene text in the chart. Therefore, conventional chart understanding models usually employ a vision encoder Encoder vis \textsc{Encoder}_{\text{vis}} to map the input chart image to a visual feature matrix: 𝐇 c = Encoder vis ​ ( 𝐜 ) \mathbf{H}_{c}=\textsc{Encoder}_{\text{vis}}(\mathbf{c}) .

 
 
 Chart-to-Table Conversion Module :
The underlying data table 𝐓 \mathbf{T} of an input chart provides a fully structured textual representation of its raw data. This can help language models better understand the information presented in the input chart.
In real-world applications, the underlying data tables of charts may not readily accessible. Therefore, various chart understanding approaches employ a chart-to-table translation module to extract a data table 𝐓 ^ \hat{\mathbf{T}} from the input chart. An extracted table can also be linearized into a sequence of table tokens [ 𝐓 ^ 1 , … , 𝐓 ^ l 𝐓 ^ ] [\hat{\mathbf{T}}_{1},\ldots,\hat{\mathbf{T}}_{l^{\hat{\mathbf{T}}}}] , where l 𝐓 ^ l^{\hat{\mathbf{T}}} denotes the number of elements in 𝐓 ^ \hat{\mathbf{T}} .

 
 
 OCR Module :
Recognizing the scene text in a chart image is an essential step for chart understanding. Many chart understanding methods apply an OCR (Optical Character Recognition) system to extract scene text from the input chart image: 𝐬 ^ = OCR ​ ( 𝐜 ) \hat{\mathbf{s}}=\textsc{OCR}(\mathbf{c}) .
The extracted scene text is a set of tokens 𝐬 ^ = { s ^ 1 , … , s ^ l s ^ } \hat{\mathbf{s}}=\{\hat{s}_{1},\ldots,\hat{s}_{l^{\hat{s}}}\} , where l s ^ l^{\hat{s}} denotes the number of tokens in the extracted scene text. An OCR system also provides the positional metadata, known as bounding box, for each extracted token, including its top left coordinates, bottom right coordinates, width, and height.

 
 
 Text Encoder :
To understand the input textual query, a text encoder is often applied to map the input textual query to a query representation matrix: 𝐇 q = Encoder txt ​ ( 𝐪 ) \mathbf{H}_{q}=\textsc{Encoder}_{\text{txt}}(\mathbf{q}) .
The text encoders in existing chart understanding models are commonly realized by a Transformer encoder [ 38 ] or word embedding layer from a pre-trained language model [ 39 ] . Similarly, we can use a text encoder to encode an extracted table into a representation matrix 𝐇 T ^ \mathbf{H}_{\hat{T}} .

 
 
 Text Decoder :
A text decoder sequentially generates a predicted textual response 𝐲 ^ \hat{\mathbf{y}} given an input context set 𝒳 \mathcal{X} . In existing chart understanding models, the input context set usually consists of the chart representation matrix, query representation matrix, and/or the representation matrix of the extracted table, e.g., 𝒳 = ( 𝐇 c , 𝐇 q , 𝐇 T ^ CLOSE \mathcal{X}=(\mathbf{H}_{c},\mathbf{H}_{q},\mathbf{H}_{\hat{T}} ).
Specifically, a text decoder Decoder txt \textsc{Decoder}_{\text{txt}} learns a conditional probability distribution over a variable length predicted response 𝐲 ^ \hat{\mathbf{y}} :
 P ⁡ ( 𝐲 ^ | 𝒳 ) = ∑ t = 1 l y ^ P ⁡ ( y ^ t | y ^ t − 1 , … , y ^ 1 , 𝒳 ) P(\hat{\mathbf{y}}|\mathcal{X})=\sum_{t=1}^{l^{\hat{y}}}P(\hat{y}_{t}|\hat{y}_{t-1},\ldots,\hat{y}_{1},\mathcal{X}) , where y ^ 1 \hat{y}_{1} denotes the i i -th token in 𝐲 ^ \hat{\mathbf{y}} and l y ^ l^{\hat{y}} denotes the length of 𝐲 ^ \hat{\mathbf{y}} .

 
 
 Note that some language models, like GPT-3 [ 40 ] , feature a decoder-only architecture, omitting a text encoder module. These models leverage a unified architecture that encodes texts in the decoder. Without loss of generality, we denote the decoder input that corresponds to the textual queries as encoding , recognizing that this encoding process resembles the encoder within encoder-decoder architectures.

 
 
 Additionally, there are some variations in the way that input charts are encoded. For example, recently developed Large Vision-Language Models (LVLMs) like ChartLlama [ 41 ] and ChartAssistant [ 42 ] employ an additional projection layer Projector vt \textsc{Projector}_{\text{vt}} to better align text and visual representations: 𝐇 c = Projector vt ​ ( Encoder vis ​ ( 𝐜 ) ) \mathbf{H}_{c}=\textsc{Projector}_{\text{vt}}(\textsc{Encoder}_{\text{vis}}(\mathbf{c})) .

 
 
 TABLE I : Notations used in this paper. 
 
 
 Notation | 
 
 
 Description 
 | 

 
 𝐜 \mathbf{c} | 
 
 
 Input chart image 
 | 

 
 𝐇 c \mathbf{H}_{c} | 
 
 
 Representation matrix for the input chart image 
 | 

 
 𝐪 \mathbf{q} | 
 
 
 Input textual query 
 | 

 
 𝐇 q \mathbf{H}_{q} | 
 
 
 Representation matrix for the input textual query 
 | 

 
 q i {q}_{i} | 
 
 
 The i i -th token of the input textual query. 
 | 

 
 l q l^{q} | 
 
 
 Number of tokens in the input textual query 
 | 

 
 𝐬 ^ \hat{\mathbf{s}} | 
 
 
 Scene text extracted from the input chart image 
 | 

 
 𝐲 \mathbf{y} | 
 
 
 Gold standard textual response 
 | 

 
 𝐲 ^ \hat{\mathbf{y}} | 
 
 
 Predicted textual response 
 | 

 
 𝐓 {\mathbf{T}} | 
 
 
 Gold standard underlying data table of the input chart image 
 | 

 
 𝐓 i {\mathbf{T}}_{i} | 
 
 
 The i i -th token for the data table 𝐓 \mathbf{T} after linearization 
 | 

 
 𝐓 ^ \hat{\mathbf{T}} | 
 
 
 Data table extracted from the input chart image by a system 
 | 

 
 Encoder vis \textsc{Encoder}_{\text{vis}} | 
 
 
 Vision encoder 
 | 

 
 Encoder txt \textsc{Encoder}_{\text{txt}} | 
 
 
 Text encoder 
 | 

 
 Decoder txt \textsc{Decoder}_{\text{txt}} | 
 
 
 Text decoder 
 | 

 
 Projector vt \textsc{Projector}_{\text{vt}} | 
 
 
 Vision-language decoder 
 | 

 
 𝒳 \mathcal{X} | 
 
 
 Input context set to the text decoder 
 | 

 
 
 
 

## III Tasks and Datasets 

 
 TABLE II: Summary of chart understanding datasets. We define the following notions for tasks: FQA : Factoid Question Answering, OQA : Open-domain Question Answering, CAP : Captioning, C2T : Chart-to-Table, FC : Fact-checking, FEC : Factual Error Correction, FID : Factual Inconsistency Detection. ✓ / ✗ indicates the dataset consists of both types of data or charts. 
 
 
 
 

 
 
 Dataset 
 Tasks 
 Source 
 Domain 
 # Instances 
 Real-world Data 
 Real-world Charts 
 
 FigureQA [ 2 ] 
 FQA 
 Synthetic 
 General 
 2M 
 ✗ 
 ✗ 
 
 DVQA [ 1 ] 
 FQA 
 Synthetic 
 General 
 3M 
 ✗ 
 ✗ 
 
 LEAF-QA [ 43 ] 
 FQA 
 World Development Indicators et al. 
 General 
 2M 
 ✓ 
 ✗ 
 
 LEAF-QA++ [ 44 ] 
 FQA 
 World Development Indicators et al. 
 General 
 2M 
 ✓ 
 ✗ 
 
 PlotQA [ 4 ] 
 FQA/C2T 
 World Bank Open Data et al. 
 General 
 29M 
 ✓ 
 ✗ 
 
 ChartQA [ 3 ] 
 FQA/C2T 
 Statista/Pew 
 General 
 10K 
 ✓ 
 ✓ 
 
 MapQA [ 45 ] 
 FQA 
 Kaiser Family Foundation 
 General 
 796K 
 ✓ 
 ✓ 
 
 PaperQA [ 46 ] 
 FQA 
 Scientific Papers 
 Science 
 107 
 ✓ 
 ✓ 
 
 SciGraphQA [ 47 ] 
 FQA 
 Arxiv 
 Science 
 296K 
 ✓ 
 ✓ 
 
 MMC-Bench [ 48 ] 
 FQA 
 Statista et al. 
 General 
 2K 
 ✓ 
 ✓ 
 
 ChartBench [ 49 ] 
 FQA 
 Kaggle 
 General 
 17K 
 ✓ 
 ✗ 
 
 ArXivQA [ 50 ] 
 FQA 
 ArXiv 
 Science 
 100K 
 ✓ 
 ✓ 
 
 OpenCQA [ 5 ] 
 OQA 
 Pew 
 General 
 8K 
 ✓ 
 ✓ 
 
 FigCAP [ 51 ] 
 CAP 
 Synthetic 
 General 
 101K 
 ✗ 
 ✗ 
 
 Chart2text [ 52 ] 
 CAP 
 Statista 
 General 
 8K 
 ✓ 
 ✓ 
 
 SciCap [ 6 ] 
 CAP 
 ArXiv 
 Science 
 476K 
 ✓ 
 ✓ 
 
 LineCap [ 53 ] 
 CAP 
 ArXiv 
 Science 
 4K 
 ✓ 
 ✓ 
 
 ChaTa+ [ 54 ] 
 CAP 
 ArXiv/World Health Organization 
 Science 
 2K 
 ✓ 
 ✓ 
 
 Chart-to-Text [ 8 ] 
 CAP/C2T 
 Statista/Pew 
 General 
 44K 
 ✓ 
 ✓ 
 
 ChartSumm [ 55 ] 
 CAP 
 Statista/Knoema 
 General 
 84K 
 ✓ 
 ✓ 
 
 VisText [ 9 ] 
 CAP/C2T 
 Statista 
 General 
 12K 
 ✓ 
 ✗ 
 
 FigCaps-HF [ 56 ] 
 CAP 
 ArXiv 
 Science 
 134K 
 ✓ 
 ✓ 
 
 ArxivCap [ 50 ] 
 CAP 
 ArXiv 
 Science 
 4M 
 ✓ 
 ✓ 
 
 ChartFC [ 13 ] 
 FC 
 Wikipedia 
 General 
 16K 
 ✓ 
 ✗ 
 
 ChartCheck [ 12 ] 
 FC 
 Wikimedia 
 General 
 10K 
 ✓ 
 ✓ 
 
 Chocolate [ 14 ] 
 FEC/FID 
 Statista/Pew 
 General 
 1K 
 ✓ 
 ✓ / ✗ 
 
 

 
 
 
 Table   II shows an overview of the key characteristics of the available datasets and the corresponding tasks. In the following subsections, we analyze these datasets along four dimensions: the task, the source, and the type of plots.

 
 

### III-A Tasks 

 
 A wide variety of datasets have been developed for various tasks, including question answering, captioning, chart-to-table conversion, factual error correction, factual error correction, and fact-checking. An example of each task is displayed in Figure   1 .

 
 
 Chart Question Answering involves presenting models with questions related to the content of a chart, which they must answer correctly. The challenge here is for the model to understand the trend of the underlying data and the relationship between data points. Two types of questions have been explored by prior work, factoid questions [ 1 , 2 , 4 , 43 , 44 , 3 , 45 , 46 ] and open-ended questions [ 5 ] . The answers to factoid questions are usually nouns (e.g. the values on the axes), verbs (e.g. increase or decrease), or adverbs (e.g. the magnitude of the trend), whereas the answers to open-ended questions are often of longer form, such as sentences.

 
 
 Chart Captioning , also known as Chart Summarization , aims to generate a descriptive caption for a given visual representation [ 52 , 8 , 9 ] . The generated caption should reflect the key insights or a summary of the information conveyed by the data visualization.

 
 
 Chart-to-Table Conversion requires a model to interpret the visual data representation and convert it into a tabular format [ 4 , 3 , 9 , 10 ] . This process involves extracting the data values and series from the chart and representing them in a structured table.

 
 
 Chart Fact-Checking involves verifying whether a given claim is factually consistent with the input chart, which helps identify cross-media misinformation [ 57 ] . ChartFC [ 13 ] and ChartCheck [ 12 ] are the only chart fact-checking datasets. In contrast to the fact-checking literature, these two datasets only consider the support or refute labels and ignore the not enough information label, where charts do not support or refute the corresponding claims. This setting is almost identical to the Factual Inconsistency Detection for Chart Captioning task [ 14 ] , in which the goal is to predict the relationship between the chart and the generated caption as consistent (i.e. support) or inconsistent (i.e. refute). The major difference is that chart fact-checking focuses on human-crated claims, whereas factual inconsistency detection uses machine-generated captions as text inputs.

 
 
 Chart Caption Factual Error Correction is an extension of the fact-checking task where models are given a chart and a caption that may not be factually consistent with the chart. The goal is to identify and correct these factual errors, ensuring that the corrected caption faithfully represents the information presented in the charts.

 
 
 Due to the development of LVLMs, recent studies have introduced more challenging tasks. For example, tasks such as Chart Redrawing and Chart Referring QA have been proposed to push the boundaries of what these models can achieve [ 58 , 42 ] . While these emerging tasks present exciting avenues for future research, they are still in their nascent stages and thus not comprehensively covered in this survey.

 
 
 

### III-B Source 

 
 In this subsection, we separately discuss the source of charts and the source of textual inputs (e.g. questions in chart question answering and claims in chart fact-checking).

 
 

#### III-B 1 Source of Charts

 
 Early datasets utilized visualization tools, such as Bokeh and Matplotlib, to create charts based on synthetic or real-world data. For example, DVQA [ 1 ] and FigureQA [ 2 ] define a set of rules to generate the underlying data, while later efforts like PlotQA [ 4 ] , LEAF-QA [ 43 ] , and LEAF-QA++ [ 44 ] began to source real-world data from various platforms, such as World Bank Open Data and World Development Indicators. More recently, more studies focused on collecting datasets with real-world charts. In particular, ChartQA [ 3 ] and Chart-to-Text [ 8 ] gather charts from Statista and the Pew Research Center, while SciGraphQA [ 47 ] and SciCap [ 6 ] collected charts from Arxiv. However, the latest developments, as shown in VisText [ 9 ] , revisit the use of visualization tools to accommodate the increasing demand for diverse visual layouts and scalability in data creation, especially for large instruction-tuning datasets (see Section   IV-B ).

 
 
 Additionally, with the advancement of LLMs, there is a growing trend of utilizing LLMs to produce synthetic charts. Examples include ChartLlama [ 41 ] , ChartX [ 58 ] , and SimChart9K [ 59 ] , which simulate data across various topics and generate code to plot charts using LLMs.

 
 
 The debate between the use of synthetic versus real-world charts is crucial in the chart understanding field. Synthetic charts, as generated by visualization tools according to predefined rules, offer the advantage of full control over the dataset, including the variety of chart types, the range and distribution of values, and the incorporation of specific challenges to test model robustness. This controlled setting allows for the systematic study of model behaviors and the identification of specific weaknesses. Furthermore, the process of generating synthetic charts is scalable and can easily produce large volumes of data necessary for training sophisticated machine learning models. However, the primary drawback of synthetic charts is their potential lack of realism. The generated charts may not fully capture the complexity and noise found in real-world charts, which can lead to poor performance in real-life scenarios for models trained on synthetic charts. Moreover, another challenge for automatically creating synthetic is balancing the desire for visual diversity and the need for the charts to remain comprehensible. We often aim to introduce variation in the styles of the charts, such as through randomized colors or plot types, to create a dataset that encompasses a wide range of visual appearances. However, ensuring that the resulting charts make visual sense poses a considerable difficulty. For instance, in a bar chart designed to compare different entities, if the bars corresponding to these entities are colored very similarly due to randomization, it may lead to confusion and impair the chart’s readability. Such issues underscore the complexity of automatically generating synthetic charts that are not only diverse but also logically coherent and visually clear.

 
 
 Fig. 1 : An overview of different chart understanding tasks. Each task is illustrated with one example. 
 
 
 Conversely, training on real-world charts could potentially lead to models better attuned to the nuances of real-world data visualizations. This could improve their performance on tasks outside controlled experimental settings. However, the use of real-world charts comes with its own set of challenges. First, collecting a large and diverse enough dataset to train robust models can be difficult and time-consuming. Additionally, the available charts may be biased towards certain domains, styles, or visual layouts, depending on the sources. Furthermore, labeling real-world charts for tasks such as question answering requires significant expert effort, making the dataset creation process slower and more expensive.

 
 
 Most datasets discussed in Table   II are not in any particular domain. Charts from these sources (e.g. Wikimedia , Pew Research Center , Statista , and etc) are about market research, public opinion, and demographic studies. Only a few datasets like SciCap [ 6 ] and PaperQA [ 46 ] dive into scientific domains. The lack of domain-specific chart understanding datasets not only underscores existing gaps in the field but also indicates great opportunities for future work. For example, domains such as healthcare, environmental science, and finance possess unique datasets with specialized chart types and data visualization needs. Further discussions regarding the lack of domain-specific chart understanding datasets can be found in Section   V-A .

 
 
 

#### III-B 2 Source of Textual Inputs

 
 Early efforts often employ template-based methods to generate textual inputs. FigureQA uses 15 templates to generate questions about charts without paraphrasing. The answers to these charts are binary (i.e. yes or no ). DVQA increases the number of templates to 26 and allows answers to be either (1) derived from the 1,000 most frequent nouns in the Brown Corpus, or (2) extracted texts (e.g. label or value of a data point) from the chart. LEAF-QA generates question templates based on analytical reasoning questions in the Graduate Record Examination (GRE), resulting in a total of 35 question templates. LEAF-QA employs back-translation for automatic paraphrasing using the Google Translate API. LEAF-QA++ extends LEAF-QA by incrementing the number of question templates to 75 and including value-based questions where answers to these questions are the positions or values of data points within the charts. Concurrently, PlotQA includes open-vocabulary questions that require applying aggregation operations on the underlying chart data.

 
 
 While template-based methods for producing textual input queries allow for low-cost and efficient data generation processes, they are limited in textual richness and may not reflect real-world scenarios. Later work employs human-crafted textual inputs to better reflect real-world applications. ChartQA asks annotators to devise compositional (i.e. a combination of mathematical/logical operations) and visual (e.g. color and lengths) questions due to their high frequencies in real-world scenarios. Chart-to-Text curates human-written chart captions that are already publicly available on Statista 1 1 
 1 
 
 
 
 
 
 
 
 https://www.statista.com/ and Pew 2 2 
 2 
 
 
 
 
 
 
 
 https://www.pewresearch.org/ . VisText further enhances the semantic richness of the captions from Statista by collecting captions with more detailed descriptions regarding the statistical, perceptual, and cognitive characteristics of charts, such as trends and the relationships between chart elements. ChartCheck asks annotators to write claims that are supported or refuted by the chart while providing explanations.

 
 
 More recently, due to the advancement of text generation, work like ChartX [ 58 ] and SimChart9K [ 60 ] use LLMs to generate diverse and high-quality textual inputs.

 
 
 

#### III-B 3 Plots

 
 In Table   III , we provide a summary of the various types of plots that are included across datasets, illustrating the diversity of chart representations in visual data. The majority of the datasets commonly feature bar and line charts, which are fundamental in representing comparisons and trends across categories and time. For example, datasets such as FigureQA [ 2 ] , Chart2text [ 52 ] , and Chocolate [ 14 ] all include these two chart types, underscoring their prevalence in the field. Pie charts are another ubiquitous plot type, offering a clear visual representation of proportional data. They are found in multiple datasets, such as LEAF-QA [ 43 ] , ChartQA [ 3 ] , and ChartCheck [ 12 ] , indicating that they are also a primary focus within the chart understanding fields.

 
 
 While bar, line, and pie charts form the core of most datasets, some recent additions have begun to incorporate area plots, which are valuable for showing cumulative quantities and changes over time. Notably, OpenCQA [ 5 ] and Chart-to-Text [ 8 ] include area plots, thus expanding the scope of analysis possible with these datasets. Additionally, ChartX [ 58 ] is currently the most comprehensive chart understanding dataset, encompassing a total of 18 chart types ranging from bar charts to candlestick plots.

 
 
 Specialized datasets often target less common but also important chart types. For example, MapQA [ 45 ] is dedicated to choropleth maps, showcasing the interest in geographical data representation. Moreover, datasets like PaperQA [ 46 ] go beyond conventional charts by including various forms of data visualization, such as t-SNE plots, potentially broadening the spectrum of chart understanding tasks.

 
 
 TABLE III: Summary of the plot types available in each dataset. 
 
 
 
 

 
 
 
 Plot Types 
 
 Dataset 
 Line 
 Bar 
 Area 
 Pie 
 Scatter 
 Others 
 
 FigureQA [ 2 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✗ 
 ✗ 
 
 DVQA [ 1 ] 
 ✗ 
 ✓ 
 ✗ 
 ✗ 
 ✗ 
 ✗ 
 
 LEAF-QA [ 43 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✗ 
 ✓ 
 
 LEAF-QA++ [ 44 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✗ 
 ✓ 
 
 PlotQA [ 4 ] 
 ✓ 
 ✓ 
 ✗ 
 ✗ 
 ✓ 
 ✗ 
 
 ChartQA [ 3 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✗ 
 ✗ 
 
 MapQA [ 45 ] 
 ✗ 
 ✗ 
 ✗ 
 ✗ 
 ✗ 
 ✓ 
 
 PaperQA [ 46 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✗ 
 ✓ 
 
 SciGraphQA [ 47 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 MMC-Bench [ 48 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 ChartBench [ 49 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 ArXivQA [ 50 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 OpenCQA [ 5 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✗ 
 
 FigCAP [ 51 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✗ 
 ✗ 
 
 Chart2text [ 52 ] 
 ✓ 
 ✓ 
 ✗ 
 ✗ 
 ✗ 
 ✗ 
 
 SciCap [ 6 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 LineCap [ 53 ] 
 ✓ 
 ✗ 
 ✗ 
 ✗ 
 ✗ 
 ✗ 
 
 ChaTa+ [ 54 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✓ 
 ✓ 
 
 Chart-to-Text [ 8 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 ChartSumm [ 55 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✗ 
 ✗ 
 
 VisText [ 9 ] 
 ✓ 
 ✓ 
 ✓ 
 ✗ 
 ✗ 
 ✗ 
 
 FigCaps-HF [ 56 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 ArXivCap [ 50 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 ChartFC [ 13 ] 
 ✗ 
 ✓ 
 ✗ 
 ✗ 
 ✗ 
 ✗ 
 
 ChartCheck [ 12 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✗ 
 ✗ 
 
 Chocolate [ 14 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 ChartLlama [ 41 ] 
 ✓ 
 ✓ 
 ✗ 
 ✓ 
 ✓ 
 ✓ 
 
 ChartX [ 58 ] 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 ✓ 
 
 

 
 
 
 
 

### III-C Evaluation Metric 

 
 In this subsection, we illustrate the evaluation metrics for each task. A summary of the metrics for each task can be found in Table   VI .

 
 
 Factoid chart question answering. The relaxed accuracy metric proposed in [ 4 ] is usually used for this task. Relaxed accuracy ( RA ) checks if the output is identical to the ground truth, similar to an exact match. However, it allows ε \varepsilon 3 3 
 3 
 
 
 
 
 
 
 
 ε \varepsilon is typically set to 5%. of numerical errors if the ground truth is a numerical value.

 
 
 

 
 | 
 RA ​ ( 𝐲 ^ , 𝐲 ) = { 1 if  ​ 𝐲 ^ = 𝐲 , 1 if  ​ 𝐲 ​  is numerical  ∧ | 𝐲 ^ − 𝐲 | ≤ ε ⋅ | 𝐲 | , 0 otherwise \texttt{RA}(\hat{\mathbf{y}},\mathbf{y})=\begin{cases}1 \text{if }\hat{\mathbf{y}}=\mathbf{y},\\
1 \text{if }\mathbf{y}\text{ is numerical }\land|\hat{\mathbf{y}}-\mathbf{y}|\leq\varepsilon\cdot|\mathbf{y}|,\\
0 \text{otherwise}\end{cases} | 
 | 
 (1) | 
 

 
 
 Recently, due to the flexibility of large vision-language models (LVLMs) to take in diverse user instructions, studies propose Acc+ to assess these models’ chart understanding with contrasting prompts [ 49 ] . Specifically, for a given question 𝐪 \mathbf{q} and the original ground truth answer 𝐲 \mathbf{y} , a wrong answer 𝐲 ′ \mathbf{y^{\prime}} is first randomly sampled from the metadata. Then, both answers are transformed into declarative statements on the question and integrated into the question. For example, with 𝐪 = \mathbf{q}= “Which person has the highest share in company A?”, 𝐲 = \mathbf{y}= “Alice”, and 𝐲 ′ = \mathbf{y^{\prime}}= “Bob”, the prompts become 𝐪 + = \mathbf{q^{+}}= “Which person has the highest share in company A? Alice has the highest share.” and 𝐪 − = \mathbf{q^{-}}= “Which person has the highest share in company A? Bob has the highest share.”, leading to binary labels 𝐲 + = \mathbf{y^{+}}= “yes” and 𝐲 − = \mathbf{y^{-}}= “no”. The model is then assessed on its ability to distinguish between these contrasting prompts’ consistency with the chart, with Acc+ calculated as:

 
 
 

 
 | 
 Acc+ ​ ( 𝐲 + , 𝐲 − , 𝐲 + ^ , 𝐲 − ^ ) = { 1 if  ​ 𝐲 + ^ = 𝐲 + ​ and  ​ 𝐲 − ^ = 𝐲 − , 0 otherwise \texttt{Acc+}(\mathbf{y^{+}},\mathbf{y^{-}},\hat{\mathbf{y^{+}}},\hat{\mathbf{y^{-}}})=\begin{cases}1 \text{if }\hat{\mathbf{y^{+}}}=\mathbf{y^{+}}\text{and }\hat{\mathbf{y^{-}}}=\mathbf{y^{-}},\\
0 \text{otherwise}\end{cases} | 
 | 
 (2) | 
 

 Note that this setting resembles factual inconsistency detection [ 14 ] and fact-checking [ 12 ] . However, it is distinct in its use of contrasting prompts and the framing of textual queries as question-answer pairs instead of declarative statements.

 
 
 Chart-to-Table Conversion. Three metrics have been developed for the chart-to-table conversion task, Relative Number Set Similarity ( RNSS ) [ 3 ] , Relative Mapping Similarity ( RMS ) [ 10 ] , and Structuring Chart-oriented Representation Metric (SCRM) [ 60 ] . For this task, we can view tables as unordered collections of mappings from a row-and-column header ( r , c ) (r,c) to a single value v v . RNSS represents each entry of the predicted table using the values only. Specifically, { 𝐓 ^ i v } 1 ≤ i ≤ N \{\hat{\mathbf{T}}^{v}_{i}\}_{1\leq i\leq N} and { 𝐓 j v } 1 ≤ j ≤ M \{\mathbf{T}^{v}_{j}\}_{1\leq j\leq M} represent the set of values in the ground truth and generated tables, respectively. RNSS defines the similarity S S between each pair of values:

 
 
 

 
 | 
 S ⁡ ( 𝐓 ^ i , 𝐓 j ) = 1 − min ⁡ ( 1 , | 𝐓 ^ i v − 𝐓 j v | | 𝐓 j | ) . \displaystyle S(\hat{\mathbf{T}}_{i},\mathbf{T}_{j})=1-\min\left(1,\frac{|\hat{\mathbf{T}}^{v}_{i}-\mathbf{T}^{v}_{j}|}{|\mathbf{T}_{j}|}\right). | 
 | 
 (3) | 
 

 
 
 Then, a minimal cost matching 𝐗 RNSS ∈ { 0 , 1 } N × M \mathbf{X}^{\texttt{RNSS}}\in\{0,1\}^{N\times M} between { 𝐓 ^ i } 1 ≤ i ≤ N \{\hat{\mathbf{T}}_{i}\}_{1\leq i\leq N} and { 𝐓 j } 1 ≤ j ≤ M \{\mathbf{T}_{j}\}_{1\leq j\leq M} is obtained based on S S and a cost optimization matching algorithm ℳ \mathcal{M} :

 
 
 

 
 | 
 C RNSS \displaystyle C^{\texttt{RNSS}} | 
 = 1 − S ⁡ ( 𝐓 ^ , 𝐓 ) , \displaystyle=1-S(\hat{\mathbf{T}},\mathbf{T}), | 
 | 
 (4) | 
 
 
 | 
 𝐗 RNSS \displaystyle\mathbf{X}^{\texttt{RNSS}} | 
 = ℳ ⁡ ( C RNSS ) . \displaystyle=\mathcal{M}(C^{\texttt{RNSS}}). | 
 | 
 (5) | 
 

 Finally, the RNSS score is computed as

 
 
 

 
 | 
 RNSS = 1 − ∑ i = 1 N ∑ j = 1 M 𝐗 i , j RNSS ​ ( 1 − S ⁡ ( 𝐓 ^ i , 𝐓 j ) ) max ⁡ ( N , M ) . \displaystyle\texttt{RNSS}=1-\frac{\sum^{N}_{i=1}\sum^{M}_{j=1}\mathbf{X}^{\texttt{RNSS}}_{i,j}(1-S(\hat{\mathbf{T}}_{i},\mathbf{T}_{j}))}{\max(N,M)}. | 
 | 
 (6) | 
 

 
 
 However, the RNSS metric exhibits a number of shortcomings. First, it is unable to discern the position of values across the table, thus failing to capture the layout information. Second, it disregards any non-numeric content within the table, which may hold critical contextual relevance. Third, RNSS disproportionately rewards estimates with substantial relative errors, potentially undermining the robustness of the evaluation. Lastly, it does not consider precision and recall.

 
 
 To address these limitations, RMS represents each entry of the predicted table as 𝐓 ^ i = ( 𝐓 ^ i r , c , 𝐓 ^ i v ) \hat{\mathbf{T}}_{i}=(\hat{\mathbf{T}}^{r,c}_{i},\hat{\mathbf{T}}^{v}_{i}) and that of the ground truth table as 𝐓 j = ( 𝐓 j r , c , 𝐓 j v ) \mathbf{T}_{j}=(\mathbf{T}^{r,c}_{j},\mathbf{T}^{v}_{j}) .
Here, 𝐓 ^ i r , c \hat{\mathbf{T}}^{r,c}_{i} denotes the string concatenation of row header 𝐓 ^ i r \hat{\mathbf{T}}^{r}_{i} and column header 𝐓 ^ i c \hat{\mathbf{T}}^{c}_{i} . The distance between two keys is computed using Normalized Levenshtein Distance [ 61 ] , denoted as NL .

 
 
 

 
 | 
 S τ key ​ ( 𝐓 ^ i , 𝐓 j ) = { 0 if  NL ​ ( 𝐓 ^ i r , c , 𝐓 j r , c ) τ , 1 − NL ​ ( 𝐓 ^ i r , c , 𝐓 j r , c ) otherwise . S^{\text{key}}_{\tau}(\hat{\mathbf{T}}_{i},\mathbf{T}_{j})=\begin{cases}0 \text{if }\texttt{NL}(\hat{\mathbf{T}}^{r,c}_{i},\mathbf{T}^{r,c}_{j}) \tau,\\
1-\texttt{NL}(\hat{\mathbf{T}}^{r,c}_{i},\mathbf{T}^{r,c}_{j}) \text{otherwise}.\\
\end{cases} | 
 | 
 (7) | 
 

 In scenarios where the Normalized Levenshtein Distance exceeds the threshold τ \tau , the similarity score S τ key S^{\text{key}}_{\tau} is set to 0 to avoid allocating partial credits to generated tables that are significantly dissimilar to ground truth tables. Similarly, a separate parameter θ \theta is introduced to prevent dissimilar values getting partial credits. The similarity between two values S θ value S^{\text{value}}_{\theta} is defined as:

 
 
 

 
 | 
 S θ value ​ ( 𝐓 ^ i , 𝐓 j ) = { 0 if  ​ | 𝐓 ^ i v − 𝐓 j v | | 𝐓 j v | θ , 1 − min ⁡ ( 1 , | 𝐓 ^ i v − 𝐓 j v | | 𝐓 j v | ) otherwise . S^{\text{value}}_{\theta}(\hat{\mathbf{T}}_{i},\mathbf{T}_{j})=\begin{cases}0 \text{if }\frac{|\hat{\mathbf{T}}^{v}_{i}-\mathbf{T}^{v}_{j}|}{|\mathbf{T}^{v}_{j}|} \theta,\\
1-\min\left(1,\frac{|\hat{\mathbf{T}}^{v}_{i}-\mathbf{T}^{v}_{j}|}{|\mathbf{T}^{v}_{j}|}\right) \text{otherwise}.\end{cases} | 
 | 
 (8) | 
 

 
 
 The similarity between two mappings S τ , θ ​ ( 𝐓 ^ i , 𝐓 j ) S_{\tau,\theta}(\hat{\mathbf{T}}_{i},\mathbf{T}_{j}) is the product of the key similarity and value similarity:

 
 
 

 
 | 
 S τ , θ ​ ( 𝐓 ^ i , 𝐓 j ) \displaystyle S_{\tau,\theta}(\hat{\mathbf{T}}_{i},\mathbf{T}_{j}) | 
 = S τ key ​ ( 𝐓 ^ i , 𝐓 j ) ⋅ S θ value ​ ( 𝐓 ^ i , 𝐓 j ) , \displaystyle=S^{\text{key}}_{\tau}(\hat{\mathbf{T}}_{i},\mathbf{T}_{j})\cdot S^{\text{value}}_{\theta}(\hat{\mathbf{T}}_{i},\mathbf{T}_{j}), | 
 | 
 (9) | 
 

 where τ \tau and θ \theta are set to 0.5 and 0.1, respectively. Then, a minimal cost matching 𝐗 RMS ∈ { 0 , 1 } N × M \mathbf{X}_{\texttt{RMS}}\in\{0,1\}^{N\times M} is found using the cost function C RMS C^{\texttt{RMS}} ,

 
 
 

 
 | 
 C RMS \displaystyle C^{\texttt{RMS}} | 
 = 1 − S τ , θ ​ ( 𝐓 ^ , 𝐓 ) , \displaystyle=1-S_{\tau,\theta}(\hat{\mathbf{T}},\mathbf{T}), | 
 | 
 (10) | 
 
 
 | 
 𝐗 RMS \displaystyle\mathbf{X}^{\texttt{RMS}} | 
 = ℳ ⁡ ( C RMS ) . \displaystyle=\mathcal{M}(C^{\texttt{RMS}}). | 
 | 
 (11) | 
 

 With 𝐗 RMS \mathbf{X}^{\texttt{RMS}} calculated, precision and recall can be computed as follows:

 
 
 

 
 | 
 RMS precision \displaystyle\texttt{RMS}_{\text{precision}} | 
 = ∑ i = 1 N ∑ j = 1 M 𝐗 i ​ j RMS ​ S τ , θ ​ ( 𝐓 ^ i , 𝐓 j ) N , \displaystyle=\frac{\sum^{N}_{i=1}\sum^{M}_{j=1}\mathbf{X}^{\texttt{RMS}}_{ij}S_{\tau,\theta}(\hat{\mathbf{T}}_{i},\mathbf{T}_{j})}{N}, | 
 | 
 (12) | 
 
 
 | 
 RMS recall \displaystyle\texttt{RMS}_{\text{recall}} | 
 = ∑ i = 1 N ∑ j = 1 M 𝐗 i ​ j RMS ​ S τ , θ ​ ( 𝐓 ^ i , 𝐓 j ) M . \displaystyle=\frac{\sum^{N}_{i=1}\sum^{M}_{j=1}\mathbf{X}^{\texttt{RMS}}_{ij}S_{\tau,\theta}(\hat{\mathbf{T}}_{i},\mathbf{T}_{j})}{M}. | 
 | 
 (13) | 
 

 The F1 score is the harmonic mean of RMS precision \texttt{RMS}_{\text{precision}} and RMS recall \texttt{RMS}_{\text{recall}} . 4 4 
 4 
 
 
 
 
 
 
 
 Note that the equations for RMS are different from the original paper [ 10 ] . This is because there is an inconsistency between the equations listed in the paper and the corresponding code implementation. We rewrite all the equations by following their source code. 

 
 
 Unlike RNSS and RMS , SCRM represents each table entry as a triplet through the Structured Triplet Representations (STR) framework. STR is designed to efficiently and robustly represent the complex data relations within chart information by capturing the relationships between row and column headers and their corresponding values. SCRM is computed by first calculating entity and value matching metrics. Thus, the SCRM metric provides a comprehensive evaluation metric to assess the performance of chart perception from structured information extraction, addressing the inherent complexities associated with data relations in visual charts.

 
 
 Chart Fact-checking and Factual Inconsistency Detection for Chart Captioning. Macro F1 and accuracy are used for chart fact-checking, while Kendall’s Tau [ 62 ] is adopted for the factual inconsistency detection for chart captioning task.

 
 
 Long-form Chart Understanding Tasks. The tasks discussed above are relatively easy to evaluate since the ground truths are either short answers or long answers that can be easily decomposed into short answers due to the structural nature of the outputs. Evaluating tasks that involve long-form texts, including Open-ended Chart Question Answering , Chart Captioning , and Chart Caption Factual Error Correction is more challenging since the outputs are much longer and cannot be easily broken down into short sub-outputs. Most of the prior work use reference-based evaluation metrics, which score the outputs against reference captions or answers. Some adopt lexical-based metrics, such as ROUGE [ 63 ] , BLEU [ 64 ] , and CIDEr [ 65 ] , another line of studies use semantic-focused metrics like perplexity and BERTScore [ 66 ] , and BLEURT [ 67 ] , others use Large Language Models (LLMs) as evaluation metrics [ 41 ] , motivated by the text generation fields [ 68 , 69 ] . However, these metrics have several drawbacks. First, these metrics do not correlate well with faithfulness , or factual consistency , between the chart and the outputs, as demonstrated by prior work in the field of summarization [ 70 , 71 , 72 ] . Second, they depend on the quality (e.g. coverage ) of the reference outputs. Taking chart captioning as an example, the majority of instances in existing datasets do not cover all key insights from the chart. Therefore, it is possible that the generated caption is faithful to the given charts but has a low lexical overlap with the reference caption. To address these issues, ChartVE [ 14 ] formulates the factual inconsistency detection problem as a visual entailment task. ChartVE learns to predict the entailment probability E ⁡ ( 𝐜 , 𝐪 ) E(\mathbf{c},\mathbf{q}) from the input chart 𝐜 \mathbf{c} to a caption sentence 𝐪 \mathbf{q} . The ChartVE score is computed by taking the minimum of all caption sentences 𝐪 ∈ 𝐐 \mathbf{q}\in\mathbf{Q} :

 
 
 

 
 | 
 ChartVE = min 𝐪 ∈ 𝐐 ⁡ E ⁡ ( 𝐜 , 𝐪 ) . \displaystyle\textsc{ChartVE}=\min_{\mathbf{q}\in\mathbf{Q}}E(\mathbf{c},\mathbf{q}). | 
 | 
 (14) | 
 

 In detecting factual inconsistency between charts and captions, ChartVE demonstrates better performance than open-source Large Vision-language Models (LVLMs), such as LLaVA-V1.5 [ 17 ] and compares favorably to proprietary LVLMs like GPT-4V [ 16 ] . However, apart from factual inconsistency , there is a lack of research on the development of metrics for evaluating other criteria. In Section   V-B , we discuss the importance of other aspects, such as coverage and relevancy .

 
 
 
 

## IV Modeling Strategies 

 {forest} 
 Fig. 2 : A taxonomy of chart understanding approaches with representative work. 
 
 
 Figure   2 shows an overview of different chart understanding approaches. In this section, we delve into the modeling strategies that have been influential in the development of chart understanding. This exploration spans classification-based and generation-based models, distinguishing between approaches that rely on pre-training and those that do not. Additionally, we highlight foundation models, particularly Large Vision-Language Models (LVLMs), as a pivotal development in the field, showcasing their capacity to ingest and interpret complex visual data through advanced encoding techniques. Finally, the section explores the concept of tool augmentation, illustrating how external systems can enhance model performance by decomposing the perception and reasoning capabilities of chart understanding tasks.

 
 

### IV-A Classification-based Models 

 
 Classification-based methods have mainly tackle the task of chart question answering, leveraging a combination of visual and textual features to understand and interpret different types of charts. This subsection provides an overview of the evolution and key strategies employed in classification-based chart understanding.

 
 

#### IV-A 1 Fixed Output Vocabulary and Initial Approaches

 
 Early methods, such as IMG+QUES [ 1 ] , utilized a fixed, limited vocabulary in the output layer. This approach typically involved the use of LSTM networks for encoding questions and a CNN architecture, like ResNet, for encoding chart figures. The representations obtained from both modalities were then fused, generally through concatenation of feature vectors, and passed through a multi-layer perceptron to predict the answer, often relying on a softmax output layer for the final answer classification. Concretely,

 
 
 

 
 | 
 𝐲 ^ = arg ​ max i ⁡ ( softmax ​ ( 𝐖 ⋅ 𝐇 concat + 𝐛 ) ) , \hat{\mathbf{y}}=\argmax_{i}\left(\text{softmax}(\mathbf{W}\cdot\mathbf{H}_{\text{concat}}+\mathbf{b})\right), | 
 | 
 (15) | 
 

 where 𝐇 concat = CONCAT ​ ( 𝐇 c , 𝐇 q ) \mathbf{H}_{\text{concat}}=\text{CONCAT}(\mathbf{H}_{c},\mathbf{H}_{q}) signifies the concatenated feature vector derived from the vision encoder 𝐇 c \mathbf{H}_{c} and the question representation 𝐇 q \mathbf{H}_{q} , 𝐖 \mathbf{W} and 𝐛 \mathbf{b} are the weights and bias of the output layer. The vocabulary for the output layer is usually constructed by gathering unique tokens from the answers in the training set. This leads to a significant limitation of these early models: the out-of-vocabulary (OOV) challenge, where the answers in the test set contain tokens that do not overlap with those in the training set. DVQA is one of the early datasets that explores this issue.

 
 
 

#### IV-A 2 Dynamic Encoding and Addressing OOV Challenges

 
 To overcome the limitations associated with fixed vocabularies, subsequent work introduced dynamic encoding schemes. SANDY [ 1 ] , for instance, enhanced chart QA capabilities by integrating a Stacked Attention Network (SAN) [ 98 ] with an OCR-based sub-network. This dual-sub-network approach allowed SANDY to generate generic answers and, simultaneously, identify and interpret textual and numerical chart elements like axis labels and data values. By constructing an image-specific dictionary linking spatial positions of chart elements to dictionary entries, SANDY successfully tackled the OOV challenge, allowing for the dynamic generation of chart-specific answers based on the content identified within the chart.

 
 
 Similarly, PReFIL [ 74 ] advances the understanding of charts by jointly learning bimodal embeddings from both low-level and high-level image features. This comprehensive understanding aids in answering complex questions that require advanced reasoning. Concurrently, LEAF-Net [ 43 ] improves chart understanding ability by employing a Mask R-CNN network [ 99 ] for element identification and an attention mechanism to focus on relevant chart areas based on the encoded question, thus enhancing answer prediction accuracy.

 
 
 

#### IV-A 3 Enhancements Through Pre-training

 
 Inspired by the success of text pre-training models, such as BERT [ 100 ] and BART [ 101 ] , more recent work introduced pre-training to further improve the performance of classification-based chart understanding methods like STL-CQA [ 44 ] . These approaches utilize several pre-training objectives tailored for enhancing model understanding of chart images and associated texts. Pre-training enables these models to learn more robust and comprehensive feature representations, improving their ability to interpret charts accurately and respond to more complex queries effectively.

 
 
 

#### IV-A 4 Assumptions on OCR Availability

 
 It is essential to note that irrespective of the advancements in classification-based methods, there remains an underlying assumption that predicted or ground-truth OCR information is available. This assumption plays a critical role in enabling these models to identify and understand textual elements within charts, which is crucial for accurately answering questions about the charts. The reliance on external OCR systems is eliminated due to the rise of OCR-free, end-to-end document understanding models like Donut [ 26 ] and Pix2Struct [ 77 ] , which will also be discussed in Section   IV-B .

 
 
 Fig. 3 : A comparison between (small) pre-trained vision-language models and LVLMs. In addition to the scale of models, the biggest difference between these two types of models is that LVLMs do not need task-specific fine-tuning since instruction-tuning allows them to generalize to unseen tasks. 
 
 
 
 

### IV-B Generation-based Models 

 
 Classification-based approaches are limited to tasks with fixed vocabulary or single-token outputs, unsuitable for many chart understanding tasks such as those involving long-form text outputs. Hence, all recent chart understanding methods adopt generation-based architectures.These methods move toward more integrated, end-to-end trainable frameworks. This subsection delineates the evolution and distinctive characteristics of these models, focusing on their encoder and decoder architecture and pre-training objectives.

 
 

#### IV-B 1 Non-pretrained Approaches

 
 Initially, generation-based models combined classical image and text processing techniques to understand charts and generate descriptions. They relied on separate modules to handle different modalities without any form of pre-training, using a CNN for visual encoding and an RNN for sequential text generation. Models like the combination of a ResNet [ 102 ] or DenseNet [ 103 ] chart encoder with an LSTM [ 104 ] decoder [ 6 , 51 , 53 ] represent these early endeavors. Image features extracted by the CNN were fed into the RNN to generate text descriptions auto-regressively. While effective, the lack of pre-trained knowledge and modality integration limited their performance and generalizability.

 
 
 Data-to-text models like Chart2text [ 52 ] and the Field-Infusing Model [ 105 ] represented a step toward more data-focused approaches, assuming access to underlying chart data or requiring external OCR for data extraction. These approaches aimed at directly converting structured data into textual summaries or captions, using Transformer-based architectures to better capture data relations and semantics. However, these methods still operated without the benefits of pre-training, limiting their ability to leverage large-scale data for improved understanding and generation.

 
 
 

#### IV-B 2 Small Pre-trained Models

 
 The advent of pre-trained models significantly enhanced the chart understanding capabilities of VL models. ChartT5 [ 76 ] emerged as a pioneering approach that utilizes pre-training objectives to enhance its understanding of charts and tables. By recovering masked information in a table using chart images, ChartT5 addressed the extraction and reasoning tasks more effectively. However, it still assumed the presence of a ground truth data table or relied on external OCR systems, highlighting a dependency that later models sought to eliminate.

 
 
 Significantly advancing the field, end-to-end visual document understanding models like Donut [ 26 ] and Pix2Struct [ 77 ] introduced the capability to directly parse textual information from images, removing the need for separate OCR processes. Donut utilizes a Swin Transformer [ 106 ] for the image encoding and the decoder of mBART [ 107 ] for decoding, focusing on natural image parsing to produce text sequences from images. Pix2Struct, employing a Vision Transformer (ViT) [ 108 ] as the image encoder and a Transformer-based decoder. The pre-training objective is screenshot parsing, which learns to predict an HTML-based parse from a screenshot of a webpage, providing a structured training signal that encapsulates both textual and layout information. There are two major distinctions between Donut and Pix2Struct: (1) screenshot parsing may allow the model to learn the structure of the image better since HTML provides stronger signals about the layouts of the input image. (2) Donut is multilingual since it uses mBART’s decoder for decoding texts.

 
 
 More recent studies further enhanced chart understanding performance by continuing pre-training these end-to-end visual document understanding models given their advanced capabilities in the layout of images. For instance, MatCha [ 78 ] builds upon Pix2Struct and undergoes pre-training with three chart-specific objectives: chart-to-table conversion, chart-to-code conversion, and mathematical question answering. These objectives not only enhance the model’s ability to interpret values and the structure of the chart but also enhance its proficiency in generating precise, contextually relevant content. Furthermore, to prevent catastrophic forgetting of the screenshot parsing capability, MatCha includes the screenshot parsing task, the original pre-training objective of Pix2Struct, as its fourth pre-training objective. Similarly, UniChart [ 79 ] is learned by continuing pre-training Donut with four pre-training objectives, including chart-to-text conversion, chart captioning, numerical and visual reasoning, and open-ended question answering. This broad spectrum of pre-training objectives significantly broadens UniChart’s capabilities, enabling it to tackle a wide array of chart-related tasks with improved overall text generation quality [ 14 ] . With the advancement in chart understanding achieved by these pre-trained models, we have the following discussions.

 
 
 First, do we still need external OCR systems? The advancements in chart understanding and interpretation achieved by UniChart and MatCha over ChartT5 highlight a pivotal shift in the field towards leveraging OCR-free models such as Pix2Struct and Donut. These developments underscore a crucial insight: the integration of advanced OCR-free pre-training objectives can obviate the dependency on traditional OCR systems, which often represent a bottleneck in terms of accuracy and adaptability. UniChart and MatCha exhibit superior performance without needing the direct input of ground truth tables, which ChartT5 relies upon. This clearly shows the increased efficacy and robustness of models developed using the OCR-free paradigm.

 
 
 Second, is it important not to forget knowledge learned during the previous pre-training stage? The distinct approaches adopted by UniChart and MatCha regarding their pre-training objectives throw light on nuanced strategies for improving chart understanding capabilities. UniChart forgoes the incorporation of Donut’s original pre-training objective, only focusing instead on a diverse set of tailored chart-centric tasks. Conversely, MatCha includes Pix2Struct’s original pre-training objective to address the potential issue of catastrophic forgetting and ensures a broader knowledge base for the model. The fact that UniChart exhibits much better performance despite excluding the pre-training objectives used by Donut [ 79 , 14 ] is indicative of the importance of designing diverse pre-training objectives to closely align with the downstream chart understanding tasks. More importantly, this suggests that forgetting knowledge learned during the previous pre-training stage may not influence downstream performance significantly.

 
 
 TABLE IV: Summary of various pre-trained and instruction fine-tuned vision-language models for chart understanding. Note that MMCA is instruction-tuned from mPLUG-Owl, ChartLlama is instruction-tuned from LLaVA-1.5, and ChartAssistant-S is instruction-tuned from SPHINX. The weights of ChartInstruct’s vision encoder are initialized from UniChart. 
 
 
 
 

 
 
 Model 
 Chart Encoder 
 Text Encoder 
 Text Decoder 
 # Parameters 
 
 (Small) Pre-trained Models 
 
 ChartT5 [ 76 ] 
 Mask R-CNN [ 99 ] 
 T5’s encoder [ 109 ] 
 T5’s decoder [ 109 ] 
 224M 
 
 Pix2Struct [ 77 ] 
 ViT-B 16 \text{ViT-B}_{\text{16}} / ViT-L 16 \text{ViT-L}_{\text{16}} [ 108 ] 
 - 
 Transformer [ 38 ] 
 282M/1.3B 
 
 MatCha [ 78 ] 
 ViT-B 16 \text{ViT-B}_{\text{16}} [ 108 ] 
 - 
 Transformer [ 38 ] 
 282M 
 
 Donut [ 26 ] 
 Swin Transformer [ 106 ] 
 - 
 mBART’s decoder [ 107 ] 
 201M 
 
 UniChart [ 79 ] 
 Swin Transformer [ 106 ] 
 - 
 mBART’s decoder [ 107 ] 
 201M 
 
 ChartAssistant-D [ 42 ] 
 Swin Transformer [ 106 ] 
 - 
 mBART’s decoder [ 107 ] 
 201M 
 
 ChartInstruct-Flan-T5-XL [ 59 ] 
 Swin Transformer [ 106 ] 
 Flan-T5-XL’s encoder [ 110 ] 
 Flan-T5-XL’s decoder [ 110 ] 
 3B 
 
 StructChart [ 60 ] 
 ViT-B 16 \text{ViT-B}_{\text{16}} [ 108 ] 
 - 
 Transformer [ 38 ] 
 282M 
 
 Large Vision-language Models 
 
 ChartInstruct-Llama [ 59 ] 
 Swin Transformer [ 106 ] 
 - 
 Llama2-7B [ 111 ] 
 7B 
 
 MMCA [ 48 ] 
 CLIP ViT-L 14 \text{CLIP ViT-L}_{\text{14}} [ 112 ] 
 - 
 Llama-7B [ 113 ] 
 7B 
 
 ChartLlama [ 41 ] 
 CLIP ViT-L 14 \text{CLIP ViT-L}_{\text{14}} [ 112 ] 
 - 
 Vicuna-13B [ 114 ] 
 13B 
 
 ChartAssistant-S [ 42 ] 
 CLIP ViT-L 14 \text{CLIP ViT-L}_{\text{14}} [ 112 ] 
 - 
 Llama-13B [ 113 ] 
 13B 
 
 ChartVLM [ 58 ] 
 ViT-B 16 \text{ViT-B}_{\text{16}} / ViT-L 16 \text{ViT-L}_{\text{16}} [ 108 ] 
 - 
 Vicuna-13B [ 114 ] 
 13B 
 
 

 
 
 
 

#### IV-B 3 Large Vision-language Models

 
 While the aforementioned pre-trained approaches significantly enhance chart understanding, they require task-specific fine-tuning (see Figure   3 ), limiting their generalizability. Contrastingly, large foundation models have demonstrated strong zero-shot or few-shot capabilities. For instance, in text generation, LLMs like GPT-3 [ 40 ] showcase remarkable generalization to novel tasks when the model size and pre-training data reach certain scales. These models are able to handle new tasks in a zero-shot or few-shot manner, obviating the need for model retraining. Extending this paradigm into image understanding, studies have stacked vision encoders on top of LLMs, including methods such as LLaVA [ 17 ] and mPLUG-Owl [ 81 ] , significantly advancing this field. These larger vision-language models (LVLMs) differ from the smaller vision-language models introduced previously by having substantially greater model sizes and instruction tuning. Unique to these LVLMs is the inclusion of a visual-language projector, Projector vt \textsc{Projector}_{\text{vt}} , given the vision encoders and LLMs’ pre-training scales. Here, the visual representations obtained from the vision encoder are fed to Projector vt \textsc{Projector}_{\text{vt}} prior to concatenated with text representations.

 
 
 Building upon general-purpose LVLMs [ 17 , 81 , 80 ] , a couple of very recent concurrent studies proposed instruction-tuning datasets and models designed for chart understanding: MMCA [ 48 ] , ChartLlama [ 41 ] , ChartAssistant [ 42 ] , and ChartInstruct [ 59 ] . By continuing training these general-purpose LVLMs on tailored instruction-tuning datasets, these models have profoundly expanded the capabilities of LVLMs in chart understanding. Below, we will discuss their difference in generating instruction-following datasets and training LVLMs.

 
 
 Regarding instruction-tuning approaches, the four LVLMs leverage distinct strategies. MMCA and ChartInstruct employ a two-stage training pipeline, a technique also observed in several LVLMs [ 81 , 115 , 17 ] . The initial stage focuses on aligning the visual and textual representations, during which only the projector is trainable, and the other components are frozen. Subsequently, in the second stage, both the projector and the text decoder undergo training, while the visual encoder remains unchanged. In contrast, ChartLlama and ChartAssistant adopt a one-stage instruction-tuning method, eliminating the isolated projector training. A notable distinction in their methodologies is ChartAssistant undergoing an extra chart-to-table pre-training process. The two-stage training method introduces complexity and cost over single-stage approaches. Furthermore, the tangible benefits of isolating the projector during training on overall model performance remain to be conclusively established.

 
 
 The datasets utilized for instruction tuning vary among these models, as outlined in Table   V . MMCA’s dataset encompasses the broadest range of tasks, while ChartAssistant’s dataset is the largest. These datasets are a mix of real-world and synthetic charts, sourced from existing chart collections or the internet. On the other hand, ChartLlama generates data and charts completely from scratch. Both ChartAssistant and ChartLlama leverage visualization tools like Matplotlib to create charts with diverse styles. ChartInstruct is the only dataset that is entirely composed of real-world charts.

 
 
 TABLE V: Summary of pre-training and instruction-tuning datasets for chart understanding. ✓ / ✗ indicates the dataset consists of both types of data or charts. 
 
 
 
 

 
 
 Model 
 # Instances 
 # Tasks 
 Real-world Data 
 Real-world Charts 
 Synthetic Data Augmentation 
 
 Pre-training Datasets 
 
 ChartT5 [ 76 ] 
 495K 
 2 
 ✓ / ✗ 
 ✓ / ✗ 
 ✗ 
 
 MatCha [ 48 ] 
 46M 
 5 
 ✓ / ✗ 
 ✓ / ✗ 
 ✗ 
 
 UniChart [ 48 ] 
 7M 
 4 
 ✓ / ✗ 
 ✓ / ✗ 
 ✓ 
 
 Instruction-tuning Datasets 
 
 MMCA [ 48 ] 
 2M 
 9 
 ✓ 
 ✓ / ✗ 
 ✗ 
 
 ChartLlama [ 41 ] 
 160K 
 7 
 ✗ 
 ✗ 
 ✓ 
 
 ChartAssistant [ 42 ] 
 39M 
 5 
 ✓ 
 ✓ / ✗ 
 ✓ 
 
 ChartInstruct [ 59 ] 
 191K 
 9 
 ✓ 
 ✓ 
 ✗ 
 
 

 
 
 
 Direct comparisons among existing LVLMs designed for chart understanding are challenging due to the absence of uniform evaluation standards. Although ChartAssistant surpasses others on the several chart understanding benchmarks (see Table   VI ), it remains uncertain which specific factors (e.g. model architecture, training data, training scheme) drive its superior performance. A more thorough experimental framework is needed to dissect how various factors enhance a model’s chart comprehension capabilities. Future research can explore several pertinent questions discussed in the following paragraphs.

 
 
 First, how do different visual representations affect an LVLM’s chart understanding proficiency? Examining how different visual representations, such as those trained with constrastive learning objectives versus the alternatives, impact an LVLM’s capacity to comprehend and generate accurate descriptions or answers based on charts could reveal insights into pre-training strategies. Additionally, all LVLMs shown in Table   IV use CLIP ViT-L 14 \text{CLIP ViT-L}_{\text{14}} , a general-purpose vision encoder. Conducting experiments by swapping this vision encoder with other vision encoders for charts, such as those from UniChart and MatCha, before instruction tuning can reveal the importance of a vision encoder tailored for chart understanding. Furthermore, some studies have found that LVLMs with a vision encoder based on image patches are ill-suited to solve geometric problems [ 46 , 116 ] , tasks that are highly related to chart understanding. This could be an explanation for the limited chart captioning abilities of state-of-the-art LVLMs, including GPT-4V [ 14 ] . A deep dive into this problem could help researchers design suitable image pre-processing methods to overcome this limitation.

 
 
 Second, should we use base (i.e. non-instruction-tuned) language models or those that have undergone instruction tuning? Many of the existing LVLMs use an instruction-tuned LLM as the text decoder [ 17 , 81 , 48 ] . Since the LLMs have already been trained on text-only instruction-tuning datasets, the resulting LVLMs may be more capable of following instructions. Given that these LLMs have been pre-trained on instruction-tuning datasets, which include anti-harmfulness instances against toxic or inappropriate content, the LVLMs derived from them are naturally better at following instructions while mitigating harmful outputs. However, the potential drawbacks of instruction tuning, such as introducing bias or diminishing performance, warrant exploration to discern the trade-off between instructions following, safeguarding against harmful content, and maintaining performance.

 
 
 Third, is there a necessity for a multi-stage training process? This question probes whether the benefits provided by multi-stage training, potentially yielding more nuanced and sophisticated representations, justifing its added complexity and cost. The performance of ChartLlama, which employs a one-stage training approach, outperforming MMCA, which uses a two-stage pipeline, with less instruction-tuning data, suggests multi-stage training may not be indispensable. Nevertheless, differences in instruction-tuning data quality, model architecture, and training configurations between MMCA and ChartLlama underscore the importance of controlled experiments to evaluate training paradigms rigorously.

 
 
 Finally, is it possible for models trained exclusively on synthetically generated data to match or exceed the performance of those trained on datasets curated from human-generated content? While synthetic data offers the advantage of scalability and control over diversity and complexity, it may lack the nuanced and unpredictable variations present in human-generated charts. Exploring whether synthetic data can truly capture the breadth of human creativity and intricacy in chart design is essential, as is understanding the potential gaps that might arise when such models are applied to real-world data. Recent work [ 59 ] has shown that completely removing real-world data from the training set significantly reduces performance. However, adding synthetic training data helps improve data efficiency and reduces the need for large-scale real-world chart data. Future research can focus on studying the effectiveness of synthetic data at varying scale and on developing methods for creating synthetic data that can better mimic real-world complexities.

 
 
 Together, these questions highlight a comprehensive spectrum of research avenues endeavoring to decode the intricate processes underlying chart comprehension through LVLMs. A detailed investigation of these topics promises to advance our progress towards models that can interpret an extensive array of charts with a level of proficiency comparable to or surpassing human capability.

 
 
 
 

### IV-C Tool Augmentation 

 
 Tool augmentation refers to utilizing external systems to address limitations in modeling capabilities, such as restricted visual representations [ 117 ] . Within the domain of chart understanding, these augmentation tools primarily focus on extracting key information from charts to facilitate further processing by more advanced models. Early-adopted tools are OCR systems designed to extract textual content from images. These systems are categorically divided into those that are enriched by language semantics [ 90 , 91 , 89 , 92 ] , and those that do not incorporate language semantics [ 93 , 95 , 94 , 96 ] . Despite their utility, the quality of information extracted using traditional OCR systems for chart understanding remains suboptimal due to their overall performance challenges. Hence, recent advancements have shifted towards OCR-free, end-to-end pre-trained vision-language models, such as Donut [ 26 ] and Pix2Struct [ 77 ] .

 
 
 TABLE VI: The state-of-the-art performance on different tasks and datasets. We only show datasets where modern approaches based on pre-trained vision-language models or LVLMs have benchmarked due to the limitations of older datasets. Note that for long-form chart understanding tasks (i.e. Open-domain Chart Question Answering, Chart Captioning, and Chart Caption Factual Error Correction), conventional metrics, such as BLEU, are sub-optimal and do not truly reflect models’ performance on these tasks, as discussed in Section   III-C . Despite advancements in metrics such as ChartVE, human evaluation remains essential for a thorough assessment of chart understanding performance. 
 
 
 
 

 
 
 Task 
 Dataset 
 SOTA Model 
 Fine-tuning 
 Performance (%) 
 Metric 
 
 Chart-to-Table Conversion 
 PlotQA 
 DePlot [ 10 ] 
 ✓ 
 94.2 
 RMS F1 \texttt{RMS}_{\text{F1}} 
 
 ChartQA 
 ChartAssistant-D [ 42 ] 
 ✓ 
 92.0 
 RMS F1 \texttt{RMS}_{\text{F1}} 
 
 VisText 
 Chart-to-Tabe [ 14 ] 
 ✓ 
 83.6 
 RMS F1 \texttt{RMS}_{\text{F1}} 
 
 Factoid Chart Question Answering 
 ChartQA 
 InternVL 1.5 [ 83 ] 
 ✗ 
 83.8 
 RA 
 
 ChartBench 
 GPT-4V [ 16 ] 
 ✗ 
 54.4 
 Acc+ 
 
 Chart Fact-checking 
 ChartFC 
 ChartInstruct-Flan-T5-XL [ 59 ] 
 ✓ 
 72.7 
 Accuracy 
 
 ChartCheck 
 DePlot-DeBERTa [ 12 ] 
 ✓ 
 73.8 
 Accuracy 
 
 Factual Inconsistency Detection for Chart Captioning 
 Chocolate FT \text{Chocolate}_{\text{FT}} 
 Bard [ 84 ] 
 ✗ 
 29.1 
 Kendall’s Tau 
 
 Chocolate LLM \text{Chocolate}_{\text{LLM}} 
 GPT-4V [ 16 ] 
 ✗ 
 20.5 
 Kendall’s Tau 
 
 Chocolate LVLM \text{Chocolate}_{\text{LVLM}} 
 ChartVE [ 14 ] 
 ✓ 
 17.8 
 Kendall’s Tau 
 
 Open-domain Chart Question Answering 
 OpenCQA 
 ChartAssistant-S [ 42 ] 
 ✗ 
 20.2 
 BLEU 
 
 Chart Captioning 
 Chart-to-Text Statista \text{Chart-to-Text}_{\text{Statista}} 
 ChartAssistant-S [ 42 ] 
 ✓ 
 41.0 
 BLEU 
 
 Chart-to-Text Pew \text{Chart-to-Text}_{\text{Pew}} 
 ChartAssistant-S [ 42 ] 
 ✓ 
 16.5 
 BLEU 
 
 Chart Caption Factual Error Correction 
 Chocolate FT \text{Chocolate}_{\text{FT}} 
 C2TFEC [ 14 ] 
 ✗ 
 81.1 
 Bard 
 
 Chocolate LLM \text{Chocolate}_{\text{LLM}} 
 GPT-4V [ 16 ] 
 ✗ 
 52.4 
 GPT-4V 
 
 Chocolate LVLM \text{Chocolate}_{\text{LVLM}} 
 C2TFEC [ 14 ] 
 ✗ 
 34.3 
 ChartVE 
 
 

 
 
 
 Parallel to OCR-free vision-language models, another line of work focuses on harnessing the strong reasoning abilities of LLMs and develops tools to augment LLMs with extracted information from charts. These tools are either specialized in converting charts to the underlying data tables, such as DePlot [ 10 ] , StructChart [ 60 ] , and Chart-to-Table [ 14 ] , or are equipped with additional capabilities for chart question answering, exemplified by DOMINO [ 97 ] . Due to these models’ proficiency in extracting chart information, they enable a strategic decompose the two required abilities of chart understanding: perception and reasoning . Perception involves extracting information from charts, such as data tables or specific values, whereas reasoning entails text-based logical and mathematical operations over the extracted data. When the extracted information is linearized into text sequences, it becomes feasible for LLMs to employ their strong reasoning abilities. Compared with LVLMs, tool-augmented LLMs have demonstrated superior performance in interpreting various chart components, such as labels, values, and trend lines [ 14 ] . However, a notable limitation is their inability to interpret visual attributes within charts, such as colors and the spatial arrangement of data points, since these details are typically absent in the extracted information.

 
 
 To mitigate this limitation, a potential solution might involve augmenting the data tables to capture color and spatial information [ 11 ] . Nevertheless, this solution may fall short in cases involving unique chart types or when the color schemes of different data points are semantically similar (e.g. blueish-green vs. greenish-blue). Thus, while tool-augmented LLMs exhibit commendable performance in tasks that prioritize textual and numerical understanding within charts over visual attribute interpretation, such as chart captioning, applicability remains limited in scenarios necessitating a detailed understanding of visual attributes.

 
 
 

### IV-D State-of-the-art Performance 

 
 In Table   VI , we display the state-of-the-art performances across various tasks and datasets. The best performance is often achieved by pre-trained models and LVLMs. This highlights the profound impact of leveraging larger-scale vision and language models, coupled with pre-training or instruction-tuning techniques, in advancing the capabilities of chart understanding models. Additionally, it can be observed that chart understanding tasks that involve long-form text generation (i.e. Chart Captioning) are more challenging than those with gold-standard ground truth answers (e.g. Chart Question Answering). To further improve the performance on these difficult tasks, future work can study the research questions outlined in Section   IV-B and Section   IV-C .

 
 
 
 

## V Challenges and Future Directions 

 

### V-A Domain-specific Charts 

 
 In Section   III-B , we have highlighted the predominance of datasets that are not tailored to any specific domain and have recognized the scarcity of domain-specific chart understanding datasets. Although a few datasets like PaperQA [ 46 ] and SciCap [ 6 ] pave the way to exploring charts in scientific literature, the majority of chart types featured therein, such as bar , line , and pie , remain commonplace. Expanding the dataset spectrum to encompass unique chart formats prevalent in specialized disciplines would illuminate struggles that contemporary models might currently overlook.

 
 
 The chemistry domain serves as an example in which automatic chart understanding can bring significant benefit for downstream applications, such as drug repurposing report generation [ 118 ] and drug discovery [ 119 ] . Charts in the chemistry domain, like pathway flowcharts, nuclear magnetic resonance (NMR) spectra, and molecular orbital diagrams, pose distinctive challenges. For instance, pathway flowcharts amalgamate a diverse range of data types, from quantitative measurements of enzyme activity to qualitative representations of interactions, necessitating the recognition of graphic components and the comprehension of their interconnectedness. Interpreting NMR spectra demands an analysis of peak patterns and intensities, alongside the molecule’s inferred structural attributes, which goes beyond mere data retrieval to encompass complex interpretative reasoning predicated on chemical knowledge. Similarly, molecular orbital diagrams represent a conceptual visualization of quantum mechanics, requiring deep insight into the complexities of atomic and molecular orbital interactions.

 
 
 The specialized charts found in domains such as chemistry often incorporate technical jargon and unique symbols intrinsic to their fields, presenting interpretive hurdles that existing chart understanding models may struggle to clear without the infusion of specific domain expertise. In developing domain-specific datasets, several pertinent research questions arise. First, do current approaches based on large-scale pre-training on datasets with general chart types enable models to generalize to unseen domain-specific charts? Second, how instrumental is domain expertise in enhancing the comprehension abilities for such charts? And third, if domain knowledge is helpful, what are the most effective methods for integrating such expertise into chart understanding models?

 
 
 

### V-B Evaluation 

 
 As highlighted in Section   III-B , prior work on long-form chart understanding tasks, including open-domain chart question answering, chart captioning, and chart caption factual error correction, only utilized conventional lexical-based metrics to assess output quality, which is inadequate. We propose five pivotal criteria for a more precise evaluation: faithfulness , coverage , relevancy , robustness , fairness and bias .

 
 
 Faithfulness. The faithfulness of a model’s output refers to the extent to which the information presented is accurate and consistent with the data depicted in the chart. Unfaithfulness or factual inconsistency can lead to outputs that misinform users [ 120 , 121 , 122 ] , potentially causing misinterpretations or incorrect decision-making based on inaccurate chart understanding. This is of particular significance in domains where accurate data representation is critical, such as education and news reporting. Although the ChartVE metric [ 14 ] offers a robust assessment of the factual consistency between chart data and textual outputs, it is unable to perform granular evaluations that locate errors within text spans. Future studies can emphasize the development of finer-grained evaluation techniques alongside the classification of error types, such as the value errors and trend errors delineated by previous work [ 14 ] , or the confidence boundary of model output hallucinations [ 123 ] . Such approaches would provide a clearer picture of where models are failing in terms of faithfulness.

 
 
 Coverage. The notion of coverage analyzes whether a model’s output encompasses all the essential insights that a chart conveys. Models with lower coverage might overlook key elements or trends within the chart, providing a partial or skewed summary of the data. This is particularly critical when comparing fine-tuned task-specific models to large vision-language models [ 14 ] . Existing evaluation metrics falter in their ability to measure coverage effectively. To truly gauge the breadth of information captured by a model from a chart, evaluation metrics that are capable of quantifying coverage are needed. These should account for the presence, significance, and the instrumental contribution of key insights toward an integrated comprehension of the chart.

 
 
 Relevancy. Relevancy is a central concern for open-domain chart question answering tasks. Beyond faithfulness and coverage, it is vital to evaluate if the model’s output pertains specifically to the question posited. The importance of relevancy stems from the possibility that an output, while being faithful to the chart, may only tangentially relate to the user’s question. This mismatch can result in outputs that, although factually correct, are of limited utility to the user. Assessing relevancy necessitates a nuanced understanding of the user’s intent and the context of the question, ensuring that responses are not only accurate and comprehensive but also specifically tailored to fulfill the user’s informational needs.

 
 
 Robustness. Robustness is a crucial aspect of interaction with LVLMs, particularly when using prompts as open-ended questions to generate descriptions (answers) from input charts. The issue arises when models produce significantly varied outputs in response to subtly different but semantically similar prompts. Such variability can be problematic, as users expect consistent responses to similar queries and safeguards against adversarial prompts [ 124 ] . To evaluate a model’s generation quality effectively, it is vital to assess its performance across a range of prompts that pertain to the same question or topic. Drawing inspirations from previous research in evaluating the robustness of LLMs [ 125 ] , future studies can design adversarial prompts that challenge the model’s understanding of charts. This approach helps in determining the model’s robustness and reliability in interpreting and generating accurate descriptions under varied prompting conditions.

 
 
 Fairness and Bias. 
Evaluating fairness and mitigating biases in the generation of models’ content is critical when deploying these tools in user-facing applications. There are two main forms of bias to be aware of: fairness bias , which concerns the equitable representation and treatment of various groups ( e.g., race, culture, or gender) [ 28 , Yang_Yu_Fung_Li_Ji_2023 ] , and sample bias , which occurs when the data used to train the model is not representative of the broader population or scenario it’s intended to serve ( e.g. , a line chart dataset containing only monotonic trends). To ensure outputs encompass diverse perspectives, it is crucial to identify and rectify these biases. Fairness bias often emerges when models disproportionately rely on their pre-trained knowledge, possibly neglecting the specific visual cues present in a chart. This reliance can lead to outputs that do not accurately reflect the visual data. For instance, if a model is trained on a dataset in which the charts consistently depict a particular group as having a higher diabetes rate compared to others, it is highly probable that when this model encounters a chart presenting contradictory information, it may still generate descriptions that align with its training, favoring interpretations that match the patterns it has learned during pre-training, even if those interpretations are less likely given the new data. Meanwhile, sample bias is evident when datasets over-represent certain types of charts. For example, some models, such as MatCha, exhibit proficiency in interpreting line charts where data points follow a monotonic sequence, but struggle to comprehend line charts characterized by convex or concave configurations, as well as those with fluctuating trends [ 14 ] . To address these issues, future work can develop benchmarks to include charts designed to counteract these biases and feature complex configurations, deviating from the typical charts in existing datasets. Future strategies should focus on diversifying training data and using adversarial training techniques to mitigate these biases. This will promote a more equitable and inclusive approach to content generation.

 
 
 One prominent direction for evaluation is model-based metrics. As evidenced in Table   VI , task-specific models like ChartVE and general-purpose models like GPT-4V achieve great correlation with human judgment in terms of faithfulness on the Chocolate dataset. This suggests that leveraging models themselves as evaluators offers a promising path forward in assessing chart understanding outputs, particularly in dimensions such as faithfulness where traditional lexical metrics fall short. Moving beyond the reliance on human-annotated gold standards, model-based evaluation can provide scalable, consistent, and nuanced insights into chart understanding performance across various dimensions. This approach, however, necessitates the development of reliable and unbiased model evaluators, which may entail pre-training models specifically for the purpose of evaluation or fine-tuning existing models with carefully curated evaluation data. Future work could explore the creation of meta-models trained on datasets designed to capture a wide array of evaluation criteria, thereby offering deeper and more comprehensive assessments of chart understanding capabilities. This approach to evaluation could significantly guide researchers towards targeted improvements in various chart understanding tasks.

 
 
 

### V-C Dissecting LVLM Components 

 
 In Section   IV-B , we delved into the design choices of LVLMs and underscored the myriad challenges they entail, alongside potential future directions for research. A pivotal area of inquiry lies in the impact of various visual representations on the chart understanding proficiency of LVLMs. The question of whether differing visual encoding methods, such as those learned through contrastive objectives versus conventional approaches, affect an LVLM’s ability to accurately interpret and generate insights from charts forms a crucial consideration. Moreover, the exploration of incorporating fused visual representations to potentially enhance model interpretability signals a promising avenue for enhancing data comprehension.

 
 
 Furthermore, the decision between utilizing base language models versus those that have undergone instruction tuning also emerges as a key point of discussion. The nuanced trade-offs between instruction-following capabilities, the safeguarding against harmful outputs offered by instruction-tuned models, and the potential performance impact underscore the need for a balanced approach in leveraging these advanced language models for chart understanding tasks.

 
 
 Lastly, the inquiry into the necessity of a multi-stage training process versus the sufficiency of models trained exclusively on synthetically generated data presents a compelling debate. The potential of synthetically generated datasets to adequately capture the complexity and nuance of human-generated content remains an open question. This encompasses not only the generalization capabilities of models trained on such datasets but also the potential for biases or limitations induced by synthetic data generation methodologies. These challenges highlight the importance of meticulous dataset curation, model training strategies, and the integration of domain-specific knowledge to overcome the inherent limitations of current LVLMs in chart understanding applications.

 
 
 

### V-D Agentic Settings 

 
 AI agents are artificial entities that sense their environment, make decisions, and take actions. Recent studies have explored the feasibility of using large models as agents for applications such as web browsing [ 126 ] , house-holding [ 127 ] , and scientific problem-solving [ 128 ] . However, no prior work has studied LVLMs as chart interpretation agents. One promising agentic application of LVLMs is education. In teaching environments, these models could assist in breaking down complex charts, explaining them in simpler terms, and even generating quizzes or interactive activities based on the chart data. For students struggling with data literacy, having an intelligent agent that can tailor explanations to their level of understanding and link chart data to real-world applications could foster a deeper comprehension of subjects. Such educational applications signify a step towards more personalized and accessible learning experiences, leveraging the power of LVLMs to demystify complex information for learners of all levels.

 
 
 Additionally, report generation can also benefit significantly from LVLMs as chart interpretation agents. In various industries, from finance to healthcare, the ability of an AI agent to automatically search for relevant chart data within internal databases and generate reports that highlight critical data points as well as recommending strategic actions could significantly enhance efficiency and decision-making processes. More importantly, report generation could also involve providing visual insights by synthesizing new charts via visualization tools based on the information the agents gathered. This could be particularly help in scenarios where rapid data interpretation is crucial, such as in stock market analysis or emergency medical response planning.

 
 
 

### V-E Multilingual Chart Understanding 

 
 Despite significant advances in chart understanding technologies, the aspect of multilingual chart understanding remains underexplored. Much of the current research and dataset development focuses predominantly on English, limiting the applicability of these models in a global context. The need for multilingual chart understanding arises from the global nature of data dissemination, where charts are created and shared in diverse languages, catering to a broad audience. A couple of research directions arise to understand models’ ability to generate insights from charts across languages. First, while current state-of-the-art LVLMs, GPT-4V, demonstrate satisfactory performance in understanding charts [ 46 ] , particularly when the values are labeled next to the corresponding data points [ 14 ] , does this hold true for charts in other languages as well? Second, what if the texts within the charts and the textual queries are in different languages? Are LVLMs capable of cross-lingual understanding in such cross-modality settings? Finally, investigating the potential for LVLMs to adapt to multilingual contexts without extensive retraining or additional instruction-tuning presents an intriguing avenue for future research. The capability of models to seamlessly transition across languages, interpreting and generating insights with high fidelity, could significantly broaden the accessibility and usability of chart understanding technologies globally.

 
 
 
 

## VI Conclusion 

 
 In this survey, we provide a thorough examination of the evolving field of automatic chart understanding, with an emphasis on recent developments such as the transformative impact of large vision-language models. Our review starts with a detailed discussion on the diverse datasets that drive chart understanding research, analyzing their sources, diversity, and the specific tasks they support. We then categorizes the landscape of chart understanding, covering from foundational classification-based and generation-based models to the cutting-edge applications of LVLMs. We offer an in-depth overview of methodologies. Additionally, we critically examine the role of datasets and outline future research trajectories. Our survey aims to spur innovation and exploration in the vibrant field of chart understanding. This marks a pathway for future breakthroughs in this crucial area of study.

 
 
 

## References

 
 
 [1] 
 
K. Kafle, B. L. Price, S. Cohen, and C. Kanan, “DVQA: understanding data visualizations via question answering,” in 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018 . IEEE Computer Society, 2018, pp. 5648–5656.

 

 
 [2] 
 
S. E. Kahou, V. Michalski, A. Atkinson, Á. Kádár, A. Trischler, and Y. Bengio, “Figureqa: An annotated figure dataset for visual reasoning,” in International Conference on Learning Representations , 2018.

 

 
 [3] 
 
A. Masry, X. L. Do, J. Q. Tan, S. Joty, and E. Hoque, “ChartQA: A benchmark for question answering about charts with visual and logical reasoning,” in Findings of the Association for Computational Linguistics: ACL 2022 , S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, 2022, pp. 2263–2279.

 

 
 [4] 
 
N. Methani, P. Ganguly, M. M. Khapra, and P. Kumar, “Plotqa: Reasoning over scientific plots,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , 2020, pp. 1527–1536.

 

 
 [5] 
 
S. Kantharaj, X. L. Do, R. T. Leong, J. Q. Tan, E. Hoque, and S. Joty, “OpenCQA: Open-ended question answering with charts,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, 2022, pp. 11 817–11 837.

 

 
 [6] 
 
T.-Y. Hsu, C. L. Giles, and T.-H. Huang, “SciCap: Generating captions for scientific figures,” in Findings of the Association for Computational Linguistics: EMNLP 2021 , M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, Eds. Punta Cana, Dominican Republic: Association for Computational Linguistics, 2021, pp. 3258–3264.

 

 
 [7] 
 
E. Hoque, P. Kavehzadeh, and A. Masry, “Chart question answering: State of the art and future directions,” in Computer Graphics Forum , vol. 41, no. 3. Wiley Online Library, 2022, pp. 555–572.

 

 
 [8] 
 
S. Kantharaj, R. T. Leong, X. Lin, A. Masry, M. Thakkar, E. Hoque, and S. Joty, “Chart-to-text: A large-scale benchmark for chart summarization,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, 2022, pp. 4005–4023.

 

 
 [9] 
 
B. Tang, A. Boggust, and A. Satyanarayan, “VisText: A benchmark for semantically rich chart captioning,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 7268–7298.

 

 
 [10] 
 
F. Liu, J. Eisenschlos, F. Piccinno, S. Krichene, C. Pang, K. Lee, M. Joshi, W. Chen, N. Collier, and Y. Altun, “DePlot: One-shot visual language reasoning by plot-to-table translation,” in Findings of the Association for Computational Linguistics: ACL 2023 , A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 10 381–10 399.

 

 
 [11] 
 
X. L. Do, M. Hassanpour, A. Masry, P. Kavehzadeh, E. Hoque, and S. Joty, “Do llms work on charts? designing few-shot prompts for chart question answering and summarization,” ArXiv preprint , vol. abs/2312.10610, 2023.

 

 
 [12] 
 
M. Akhtar, N. Subedi, V. Gupta, S. Tahmasebi, O. Cocarascu, and E. Simperl, “Chartcheck: An evidence-based fact-checking dataset over real-world chart images,” ArXiv preprint , vol. abs/2311.07453, 2023.

 

 
 [13] 
 
M. Akhtar, O. Cocarascu, and E. Simperl, “Reading and reasoning over chart images for evidence-based automated fact-checking,” in Findings of the Association for Computational Linguistics: EACL 2023 , A. Vlachos and I. Augenstein, Eds. Dubrovnik, Croatia: Association for Computational Linguistics, 2023, pp. 399–414.

 

 
 [14] 
 
K.-H. Huang, M. Zhou, H. P. Chan, Y. Fung, Z. Wang, L. Zhang, S.-F. Chang, and H. Ji, “Do LVLMs understand charts? analyzing and correcting factual errors in chart captioning,” in Findings of the Association for Computational Linguistics ACL 2024 , L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand and virtual meeting: Association for Computational Linguistics, Aug. 2024, pp. 730–749.

 

 
 [15] 
 
A. M. Farahani, P. Adibi, M. S. Ehsani, H.-P. Hutter, and A. Darvishy, “Automatic chart understanding: A review,” IEEE Access , vol. 11, pp. 76 202–76 221, 2023.

 

 
 [16] 
 
OpenAI, “Gpt-4v(ision) system card,” 2023.

 

 
 [17] 
 
H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” 2023.

 

 
 [18] 
 
A. M. Farahani, P. Adibi, M. S. Ehsani, H.-P. Hutter, and A. Darvishy, “Automatic chart understanding: A review,” IEEE Access , vol. 11, pp. 76 202–76 221, 2023.

 

 
 [19] 
 
E. Hoque, P. Kavehzadeh, and A. Masry, “Chart question answering: State of the art and future directions,” in Computer Graphics Forum , vol. 41, no. 3. Wiley Online Library, 2022, pp. 555–572.

 

 
 [20] 
 
A. Wu, Y. Wang, X. Shu, D. Moritz, W. Cui, H. Zhang, D. Zhang, and H. Qu, “Ai4vis: Survey on artificial intelligence approaches for data visualization,” IEEE Transactions on Visualization and Computer Graphics , 2021.

 

 
 [21] 
 
L. Shen, E. Shen, Y. Luo, X. Yang, X. Hu, X. Zhang, Z. Tai, and J. Wang, “Towards natural language interfaces for data visualization: A survey,” IEEE transactions on visualization and computer graphics , 2022.

 

 
 [22] 
 
W. Yang, M. Liu, Z. Wang, and S. Liu, “Foundation models meet visualizations: Challenges and opportunities,” ArXiv preprint , vol. abs/2310.05771, 2023.

 

 
 [23] 
 
Y. He, S. Cao, Y. Shi, Q. Chen, K. Xu, and N. Cao, “Leveraging large models for crafting narrative visualization: A survey,” ArXiv preprint , vol. abs/2401.14010, 2024.

 

 
 [24] 
 
S. Pratt, M. Yatskar, L. Weihs, A. Farhadi, and A. Kembhavi, “Grounded situation recognition,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16 . Springer, 2020, pp. 314–332.

 

 
 [25] 
 
H. Zhang, Y. Wang, S. Wang, X. Cao, F. Zhang, and Z. Wang, “Table fact verification with structure-aware transformer,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , B. Webber, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, 2020, pp. 1624–1629.

 

 
 [26] 
 
G. Kim, T. Hong, M. Yim, J. Nam, J. Park, J. Yim, W. Hwang, S. Yun, D. Han, and S. Park, “Ocr-free document understanding transformer,” in European Conference on Computer Vision . Springer, 2022, pp. 498–517.

 

 
 [27] 
 
O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, “Show and tell: A neural image caption generator,” in IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015 . IEEE Computer Society, 2015, pp. 3156–3164.

 

 
 [28] 
 
H. Qiu, Z.-Y. Dou, T. Wang, A. Celikyilmaz, and N. Peng, “Gender biases in automatic evaluation metrics for image captioning,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, 2023, pp. 8358–8375.

 

 
 [29] 
 
S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, “VQA: visual question answering,” in 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015 . IEEE Computer Society, 2015, pp. 2425–2433.

 

 
 [30] 
 
D. A. Hudson and C. D. Manning, “GQA: A new dataset for real-world visual reasoning and compositional question answering,” in IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019 . Computer Vision Foundation / IEEE, 2019, pp. 6700–6709.

 

 
 [31] 
 
P. Pasupat and P. Liang, “Compositional semantic parsing on semi-structured tables,” in Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , C. Zong and M. Strube, Eds. Beijing, China: Association for Computational Linguistics, 2015, pp. 1470–1480.

 

 
 [32] 
 
A. Parikh, X. Wang, S. Gehrmann, M. Faruqui, B. Dhingra, D. Yang, and D. Das, “ToTTo: A controlled table-to-text generation dataset,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , B. Webber, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, 2020, pp. 1173–1186.

 

 
 [33] 
 
J. Eisenschlos, S. Krichene, and T. Müller, “Understanding tables with intermediate pre-training,” in Findings of the Association for Computational Linguistics: EMNLP 2020 , T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, 2020, pp. 281–296.

 

 
 [34] 
 
Q. Liu, B. Chen, J. Guo, M. Ziyadi, Z. Lin, W. Chen, and J. Lou, “TAPEX: table pre-training via learning a neural SQL executor,” in The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 . OpenReview.net, 2022.

 

 
 [35] 
 
Z. Huang, K. Chen, J. He, X. Bai, D. Karatzas, S. Lu, and C. V. Jawahar, “ICDAR2019 competition on scanned receipt OCR and information extraction,” in 2019 International Conference on Document Analysis and Recognition, ICDAR 2019, Sydney, Australia, September 20-25, 2019 . IEEE, 2019, pp. 1516–1520.

 

 
 [36] 
 
M. Mathew, D. Karatzas, and C. Jawahar, “Docvqa: A dataset for vqa on document images,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision , 2021, pp. 2200–2209.

 

 
 [37] 
 
R. Tanaka, K. Nishida, K. Nishida, T. Hasegawa, I. Saito, and K. Saito, “Slidevqa: A dataset for document visual question answering on multiple images,” in Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023 , B. Williams, Y. Chen, and J. Neville, Eds. AAAI Press, 2023, pp. 13 636–13 645.

 

 
 [38] 
 
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA , I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, Eds., 2017, pp. 5998–6008.

 

 
 [39] 
 
M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, 2020, pp. 7871–7880.

 

 
 [40] 
 
T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “Language models are few-shot learners,” in Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., 2020.

 

 
 [41] 
 
Y. Han, C. Zhang, X. Chen, X. Yang, Z. Wang, G. Yu, B. Fu, and H. Zhang, “Chartllama: A multimodal llm for chart understanding and generation,” ArXiv preprint , vol. abs/2311.16483, 2023.

 

 
 [42] 
 
F. Meng, W. Shao, Q. Lu, P. Gao, K. Zhang, Y. Qiao, and P. Luo, “ChartAssistant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning,” in Findings of the Association for Computational Linguistics ACL 2024 , L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand and virtual meeting: Association for Computational Linguistics, Aug. 2024, pp. 7775–7803.

 

 
 [43] 
 
R. Chaudhry, S. Shekhar, U. Gupta, P. Maneriker, P. Bansal, and A. Joshi, “Leaf-qa: Locate, encode attend for figure question answering,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , 2020, pp. 3512–3521.

 

 
 [44] 
 
H. Singh and S. Shekhar, “STL-CQA: Structure-based transformers with localization and encoding for chart question answering,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , B. Webber, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, 2020, pp. 3275–3284.

 

 
 [45] 
 
S. Chang, D. Palzer, J. Li, E. Fosler-Lussier, and N. Xiao, “Mapqa: A dataset for question answering on choropleth maps,” ArXiv preprint , vol. abs/2211.08545, 2022.

 

 
 [46] 
 
P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” ArXiv preprint , vol. abs/2310.02255, 2023.

 

 
 [47] 
 
S. Li and N. Tajbakhsh, “Scigraphqa: A large-scale synthetic multi-turn question-answering dataset for scientific graphs,” arXiv preprint arXiv:2308.03349 , 2023.

 

 
 [48] 
 
F. Liu, X. Wang, W. Yao, J. Chen, K. Song, S. Cho, Y. Yacoob, and D. Yu, “MMC: Advancing multimodal chart understanding with large-scale instruction tuning,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , K. Duh, H. Gomez, and S. Bethard, Eds. Mexico City, Mexico: Association for Computational Linguistics, 2024, pp. 1287–1310.

 

 
 [49] 
 
Z. Xu, S. Du, Y. Qi, C. Xu, C. Yuan, and J. Guo, “Chartbench: A benchmark for complex visual reasoning in charts,” ArXiv preprint , vol. abs/2312.15915, 2023.

 

 
 [50] 
 
L. Li, Y. Wang, R. Xu, P. Wang, X. Feng, L. Kong, and Q. Liu, “Multimodal ArXiv: A dataset for improving scientific comprehension of large vision-language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 14 369–14 387.

 

 
 [51] 
 
C. Chen, R. Zhang, E. Koh, S. Kim, S. Cohen, T. Yu, R. Rossi, and R. Bunescu, “Figure captioning with reasoning and sequence-level training,” ArXiv preprint , vol. abs/1906.02850, 2019.

 

 
 [52] 
 
J. Obeid and E. Hoque, “Chart-to-text: Generating natural language descriptions for charts by adapting the transformer model,” in Proceedings of the 13th International Conference on Natural Language Generation , B. Davis, Y. Graham, J. Kelleher, and Y. Sripada, Eds. Dublin, Ireland: Association for Computational Linguistics, 2020, pp. 138–147.

 

 
 [53] 
 
A. Mahinpei, Z. Kostic, and C. Tanner, “Linecap: Line charts for data visualization captioning models,” in 2022 IEEE Visualization and Visual Analytics (VIS) . IEEE, 2022, pp. 35–39.

 

 
 [54] 
 
K. Seweryn, K. Lorenc, A. Wróblewska, and S. Sysko-Romańczuk, “What will you tell me about the chart?–automated description of charts,” in Neural Information Processing: 28th International Conference, ICONIP 2021, Sanur, Bali, Indonesia, December 8–12, 2021, Proceedings, Part V 28 . Springer, 2021, pp. 12–19.

 

 
 [55] 
 
R. Rahman, R. Hasan, A. A. Farhad, M. T. R. Laskar, M. H. Ashmafee, and A. R. M. Kamal, “Chartsumm: A comprehensive benchmark for automatic chart summarization of long and short summaries,” ArXiv preprint , vol. abs/2304.13620, 2023.

 

 
 [56] 
 
A. Singh, P. Agarwal, Z. Huang, A. Singh, T. Yu, S. Kim, V. Bursztyn, N. Vlassis, and R. A. Rossi, “Figcaps-hf: A figure-to-caption generative framework and benchmark with human feedback,” ArXiv preprint , vol. abs/2307.10867, 2023.

 

 
 [57] 
 
Y. R. Fung, K. Huang, P. Nakov, and H. Ji, “The battlefront of combating misinformation and coping with media bias,” in KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022 , A. Zhang and H. Rangwala, Eds. ACM, 2022, pp. 4790–4791.

 

 
 [58] 
 
R. Xia, B. Zhang, H. Ye, X. Yan, Q. Liu, H. Zhou, Z. Chen, M. Dou, B. Shi, J. Yan, and Y. Qiao, “Chartx chartvlm: A versatile benchmark and foundation model for complicated chart reasoning,” ArXiv preprint , vol. abs/2402.12185, 2024.

 

 
 [59] 
 
A. Masry, M. Shahmohammadi, M. R. Parvez, E. Hoque, and S. Joty, “Chartinstruct: Instruction tuning for chart comprehension and reasoning,” arXiv preprint arXiv:2403.09028 , 2024.

 

 
 [60] 
 
R. Xia, B. Zhang, H. Peng, N. Liao, P. Ye, B. Shi, J. Yan, and Y. Qiao, “Structchart: Perception, structuring, reasoning for visual chart understanding,” ArXiv preprint , vol. abs/2309.11268, 2023.

 

 
 [61] 
 
V. I. Levenshtein et al. , “Binary codes capable of correcting deletions, insertions, and reversals,” in Soviet physics doklady , vol. 10, no. 8. Soviet Union, 1966, pp. 707–710.

 

 
 [62] 
 
M. G. Kendall, “A new measure of rank correlation,” Biometrika , vol. 30, no. 1/2, pp. 81–93, 1938.

 

 
 [63] 
 
C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in Text Summarization Branches Out . Barcelona, Spain: Association for Computational Linguistics, 2004, pp. 74–81.

 

 
 [64] 
 
K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics , P. Isabelle, E. Charniak, and D. Lin, Eds. Philadelphia, Pennsylvania, USA: Association for Computational Linguistics, 2002, pp. 311–318.

 

 
 [65] 
 
R. Vedantam, C. L. Zitnick, and D. Parikh, “Cider: Consensus-based image description evaluation,” in IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015 . IEEE Computer Society, 2015, pp. 4566–4575.

 

 
 [66] 
 
T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “Bertscore: Evaluating text generation with BERT,” in 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net, 2020.

 

 
 [67] 
 
T. Sellam, D. Das, and A. Parikh, “BLEURT: Learning robust metrics for text generation,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, 2020, pp. 7881–7892.

 

 
 [68] 
 
Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu, “G-eval: NLG evaluation using gpt-4 with better human alignment,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, 2023, pp. 2511–2522.

 

 
 [69] 
 
K.-H. Huang, P. Laban, A. Fabbri, P. K. Choubey, S. Joty, C. Xiong, and C.-S. Wu, “Embrace divergence for richer insights: A multi-document summarization benchmark and a case study on summarizing diverse information from news articles,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , K. Duh, H. Gomez, and S. Bethard, Eds. Mexico City, Mexico: Association for Computational Linguistics, 2024, pp. 570–593.

 

 
 [70] 
 
A. Wang, K. Cho, and M. Lewis, “Asking and answering questions to evaluate the factual consistency of summaries,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, 2020, pp. 5008–5020.

 

 
 [71] 
 
K.-H. Huang, S. Singh, X. Ma, W. Xiao, F. Nan, N. Dingwall, W. Y. Wang, and K. McKeown, “SWING: Balancing coverage and faithfulness for dialogue summarization,” in Findings of the Association for Computational Linguistics: EACL 2023 , A. Vlachos and I. Augenstein, Eds. Dubrovnik, Croatia: Association for Computational Linguistics, 2023, pp. 512–525.

 

 
 [72] 
 
H. Qiu, K.-H. Huang, J. Qu, and N. Peng, “AMRFact: Enhancing summarization factuality evaluation with AMR-driven negative samples generation,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , K. Duh, H. Gomez, and S. Bethard, Eds. Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 594–608.

 

 
 [73] 
 
A. Santoro, D. Raposo, D. G. T. Barrett, M. Malinowski, R. Pascanu, P. W. Battaglia, and T. Lillicrap, “A simple neural network module for relational reasoning,” in Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA , I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, Eds., 2017, pp. 4967–4976.

 

 
 [74] 
 
K. Kafle, R. Shrestha, S. Cohen, B. Price, and C. Kanan, “Answering questions about data visualizations using efficient bimodal fusion,” in Proceedings of the IEEE/CVF Winter conference on applications of computer vision , 2020, pp. 1498–1507.

 

 
 [75] 
 
J. Herzig, P. K. Nowak, T. Müller, F. Piccinno, and J. Eisenschlos, “TaPas: Weakly supervised table parsing via pre-training,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, 2020, pp. 4320–4333.

 

 
 [76] 
 
M. Zhou, Y. Fung, L. Chen, C. Thomas, H. Ji, and S.-F. Chang, “Enhanced chart understanding via visual language pre-training on plot table pairs,” in Findings of the Association for Computational Linguistics: ACL 2023 , A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 1314–1326.

 

 
 [77] 
 
K. Lee, M. Joshi, I. R. Turc, H. Hu, F. Liu, J. M. Eisenschlos, U. Khandelwal, P. Shaw, M. Chang, and K. Toutanova, “Pix2struct: Screenshot parsing as pretraining for visual language understanding,” in International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 2023, pp. 18 893–18 912.

 

 
 [78] 
 
F. Liu, F. Piccinno, S. Krichene, C. Pang, K. Lee, M. Joshi, Y. Altun, N. Collier, and J. Eisenschlos, “MatCha: Enhancing visual language pretraining with math reasoning and chart derendering,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 12 756–12 770.

 

 
 [79] 
 
A. Masry, P. Kavehzadeh, X. L. Do, E. Hoque, and S. Joty, “UniChart: A universal vision-language pretrained model for chart comprehension and reasoning,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, 2023, pp. 14 662–14 684.

 

 
 [80] 
 
Z. Lin, C. Liu, R. Zhang, P. Gao, L. Qiu, H. Xiao, H. Qiu, C. Lin, W. Shao, K. Chen et al. , “Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models,” ArXiv preprint , vol. abs/2311.07575, 2023.

 

 
 [81] 
 
Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi et al. , “mplug-owl: Modularization empowers large language models with multimodality,” ArXiv preprint , vol. abs/2304.14178, 2023.

 

 
 [82] 
 
J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, Y. Dan, C. Zhao, G. Xu, C. Li, J. Tian et al. , “mplug-docowl: Modularized multimodal large language model for document understanding,” ArXiv preprint , vol. abs/2307.02499, 2023.

 

 
 [83] 
 
Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al. , “How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites,” ArXiv preprint , vol. abs/2404.16821, 2024.

 

 
 [84] 
 
G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth et al. , “Gemini: a family of highly capable multimodal models,” ArXiv preprint , vol. abs/2312.11805, 2023.

 

 
 [85] 
 
Anthropic, “Introducing the next generation of claude,” 2024.

 

 
 [86] 
 
J. Chen, L. Kong, H. Wei, C. Liu, Z. Ge, L. Zhao, J. Sun, C. Han, and X. Zhang, “Onechart: Purify the chart structural extraction via one auxiliary token,” arXiv preprint arXiv:2404.09987 , 2024.

 

 
 [87] 
 
L. Zhang, A. Hu, H. Xu, M. Yan, Y. Xu, Q. Jin, J. Zhang, and F. Huang, “Tinychart: Efficient chart understanding with visual token merging and program-of-thoughts learning,” arXiv preprint arXiv:2404.16635 , 2024.

 

 
 [88] 
 
Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, “Character region awareness for text detection,” in IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019 . Computer Vision Foundation / IEEE, 2019, pp. 9365–9374.

 

 
 [89] 
 
D. Bautista and R. Atienza, “Scene text recognition with permuted autoregressive sequence models,” in European Conference on Computer Vision . Springer, 2022, pp. 178–196.

 

 
 [90] 
 
B. Shi, X. Wang, P. Lyu, C. Yao, and X. Bai, “Robust scene text recognition with automatic rectification,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016 . IEEE Computer Society, 2016, pp. 4168–4176.

 

 
 [91] 
 
Z. Cheng, F. Bai, Y. Xu, G. Zheng, S. Pu, and S. Zhou, “Focusing attention: Towards accurate text recognition in natural images,” in IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017 . IEEE Computer Society, 2017, pp. 5086–5094.

 

 
 [92] 
 
S. Fang, Z. Mao, H. Xie, Y. Wang, C. Yan, and Y. Zhang, “Abinet++: Autonomous, bidirectional and iterative language modeling for scene text spotting,” IEEE Transactions on Pattern Analysis and Machine Intelligence , 2022.

 

 
 [93] 
 
B. Shi, X. Bai, and C. Yao, “An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition,” IEEE transactions on pattern analysis and machine intelligence , vol. 39, no. 11, pp. 2298–2304, 2016.

 

 
 [94] 
 
W. Liu, C. Chen, K. K. Wong, Z. Su, and J. Han, “Star-net: A spatial attention residue network for scene text recognition,” in Proceedings of the British Machine Vision Conference 2016, BMVC 2016, York, UK, September 19-22, 2016 , R. C. Wilson, E. R. Hancock, and W. A. P. Smith, Eds. BMVA Press, 2016.

 

 
 [95] 
 
J. Wang and X. Hu, “Gated recurrent convolution neural network for OCR,” in Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA , I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, Eds., 2017, pp. 335–344.

 

 
 [96] 
 
F. Borisyuk, A. Gordo, and V. Sivakumar, “Rosetta: Large scale system for text detection and recognition in images,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery Data Mining, KDD 2018, London, UK, August 19-23, 2018 , Y. Guo and F. Farooq, Eds. ACM, 2018, pp. 71–79.

 

 
 [97] 
 
P. Wang, O. Golovneva, A. Aghajanyan, X. Ren, M. Chen, A. Celikyilmaz, and M. Fazel-Zarandi, “Domino: A dual-system for multi-step visual language reasoning,” ArXiv preprint , vol. abs/2310.02804, 2023.

 

 
 [98] 
 
Z. Yang, X. He, J. Gao, L. Deng, and A. J. Smola, “Stacked attention networks for image question answering,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016 . IEEE Computer Society, 2016, pp. 21–29.

 

 
 [99] 
 
K. He, G. Gkioxari, P. Dollár, and R. B. Girshick, “Mask R-CNN,” in IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017 . IEEE Computer Society, 2017, pp. 2980–2988.

 

 
 [100] 
 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , J. Burstein, C. Doran, and T. Solorio, Eds. Minneapolis, Minnesota: Association for Computational Linguistics, 2019, pp. 4171–4186.

 

 
 [101] 
 
M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, 2020, pp. 7871–7880.

 

 
 [102] 
 
K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016 . IEEE Computer Society, 2016, pp. 770–778.

 

 
 [103] 
 
G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017 . IEEE Computer Society, 2017, pp. 2261–2269.

 

 
 [104] 
 
S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural computation , vol. 9, no. 8, pp. 1735–1780, 1997.

 

 
 [105] 
 
W. Chen, J. Chen, Y. Su, Z. Chen, and W. Y. Wang, “Logical natural language generation from open-domain tables,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, 2020, pp. 7929–7942.

 

 
 [106] 
 
Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021 . IEEE, 2021, pp. 9992–10 002.

 

 
 [107] 
 
Y. Liu, J. Gu, N. Goyal, X. Li, S. Edunov, M. Ghazvininejad, M. Lewis, and L. Zettlemoyer, “Multilingual denoising pre-training for neural machine translation,” Transactions of the Association for Computational Linguistics , vol. 8, pp. 726–742, 2020.

 

 
 [108] 
 
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 . OpenReview.net, 2021.

 

 
 [109] 
 
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” J. Mach. Learn. Res. , vol. 21, pp. 140:1–140:67, 2020.

 

 
 [110] 
 
H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, E. Li, X. Wang, M. Dehghani, S. Brahma, A. Webson, S. S. Gu, Z. Dai, M. Suzgun, X. Chen, A. Chowdhery, S. Narang, G. Mishra, A. Yu, V. Y. Zhao, Y. Huang, A. M. Dai, H. Yu, S. Petrov, E. H. Chi, J. Dean, J. Devlin, A. Roberts, D. Zhou, Q. V. Le, and J. Wei, “Scaling instruction-finetuned language models,” ArXiv preprint , vol. abs/2210.11416, 2022.

 

 
 [111] 
 
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al. , “Llama 2: Open foundation and fine-tuned chat models,” ArXiv preprint , vol. abs/2307.09288, 2023.

 

 
 [112] 
 
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 2021, pp. 8748–8763.

 

 
 [113] 
 
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar et al. , “Llama: Open and efficient foundation language models,” ArXiv preprint , vol. abs/2302.13971, 2023.

 

 
 [114] 
 
W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality,” 2023.

 

 
 [115] 
 
K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao, “Shikra: Unleashing multimodal llm’s referential dialogue magic,” ArXiv preprint , vol. abs/2306.15195, 2023.

 

 
 [116] 
 
X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, C. Wei, B. Yu, R. Yuan, R. Sun, M. Yin, B. Zheng, Z. Yang, Y. Liu, W. Huang, H. Sun, Y. Su, and W. Chen, “Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , June 2024, pp. 9556–9567.

 

 
 [117] 
 
Y. Qin, S. Hu, Y. Lin, W. Chen, N. Ding, G. Cui, Z. Zeng, Y. Huang, C. Xiao, C. Han, Y. R. Fung, Y. Su, H. Wang, C. Qian, R. Tian, K. Zhu, S. Liang, X. Shen, B. Xu, Z. Zhang, Y. Ye, B. Li, Z. Tang, J. Yi, Y. Zhu, Z. Dai, L. Yan, X. Cong, Y. Lu, W. Zhao, Y. Huang, J. Yan, X. Han, X. Sun, D. Li, J. Phang, C. Yang, T. Wu, H. Ji, Z. Liu, and M. Sun, “Tool learning with foundation models,” 2023.

 

 
 [118] 
 
Q. Wang, M. Li, X. Wang, N. Parulian, G. Han, J. Ma, J. Tu, Y. Lin, R. H. Zhang, W. Liu, A. Chauhan, Y. Guan, B. Li, R. Li, X. Song, Y. Fung, H. Ji, J. Han, S.-F. Chang, J. Pustejovsky, J. Rah, D. Liem, A. ELsayed, M. Palmer, C. Voss, C. Schneider, and B. Onyshkevych, “COVID-19 literature knowledge graph construction and drug repurposing report generation,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations , A. Sil and X. V. Lin, Eds. Online: Association for Computational Linguistics, 2021, pp. 66–77.

 

 
 [119] 
 
C. N. Edwards, A. Naik, T. Khot, M. D. Burke, H. Ji, and T. Hope, “Synergpt: In-context learning for personalized drug synergy prediction and drug design,” bioRxiv , pp. 2023–07, 2023.

 

 
 [120] 
 
K.-H. Huang, H. P. Chan, and H. Ji, “Zero-shot faithful factual error correction,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 5660–5676.

 

 
 [121] 
 
H. P. Chan, Q. Zeng, and H. Ji, “Interpretable automatic fine-grained inconsistency detection in text summarization,” in Findings of the Association for Computational Linguistics: ACL 2023 , A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 6433–6444.

 

 
 [122] 
 
K. Kim, S. Lee, K. Huang, H. P. Chan, M. Li, and H. Ji, “Can llms produce faithful explanations for fact-checking? towards faithful explainable fact-checking via multi-agent debate,” ArXiv preprint , vol. abs/2402.07401, 2024.

 

 
 [123] 
 
H. Zhang, S. Diao, Y. Lin, Y. Fung, Q. Lian, X. Wang, Y. Chen, H. Ji, and T. Zhang, “R-tuning: Instructing large language models to say ‘I don’t know’,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , K. Duh, H. Gomez, and S. Bethard, Eds. Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 7113–7139.

 

 
 [124] 
 
S. Li, C. Han, P. Yu, C. Edwards, M. Li, X. Wang, Y. Fung, C. Yu, J. Tetreault, E. Hovy, and H. Ji, “Defining a new NLP playground,” in Findings of the Association for Computational Linguistics: EMNLP 2023 , H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, 2023, pp. 11 932–11 951.

 

 
 [125] 
 
M. Sclar, Y. Choi, Y. Tsvetkov, and A. Suhr, “Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting,” ArXiv preprint , vol. abs/2310.11324, 2023.

 

 
 [126] 
 
X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su, “Mind2web: Towards a generalist agent for the web,” in Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., 2023.

 

 
 [127] 
 
X. Liu, H. Yu, H. Zhang, Y. Xu, X. Lei, H. Lai, Y. Gu, H. Ding, K. Men, K. Yang, S. Zhang, X. Deng, A. Zeng, Z. Du, C. Zhang, S. Shen, T. Zhang, Y. Su, H. Sun, M. Huang, Y. Dong, and J. Tang, “Agentbench: Evaluating LLMs as agents,” in The Twelfth International Conference on Learning Representations , 2024.

 

 
 [128] 
 
K. Yang, J. Liu, J. Wu, C. Yang, Y. R. Fung, S. Li, Z. Huang, X. Cao, X. Wang, Y. Wang, H. Ji, and C. Zhai, “If llm is the wizard, then code is the wand: A survey on how code empowers large language models to serve as intelligent agents,” 2024.

 

 
 
 
 
 
 
 | 
 
 
 Kung-Hsiang Huang is a research scientist at Salesforce Research. He obtained his Ph.D. from the University of Illinois Urbana-Champaign under the guidance of Prof. Heng Ji. His research focuses on fact-checking, faithfulness enhancement, factual error correction, and chart understanding. He is a recipient of the inaugural fellowship awarded by the Amazon-Illinois Center. He has taught two tutorials at conferences, including “KDD’22: The Battlefront of Combating Misinformation and Coping with Media Bias”. He is the recipient of the Top Reviewer Award at NeurIPS 2024. 
 | 

 
 
 
 
 | 
 
 
 Hou Pong Chan is a research scientist at the Language Technology Lab, Alibaba DAMO Academy. Prior to that, he was a visiting postdoc researcher at the University of Illinois at Urbana-Champaign and was a visiting lecturer (Macao Fellow) at the University of Macau. He received a Ph.D. degree in Computer Science from The Chinese University of Hong Kong. His research focuses on improving the factuality and controllability of natural language generation. He has published more than 20 research papers in leading NLP conferences (e.g., ACL, EMNLP, and NAACL) and journals (e.g., TACL). He is a recipient of the Outstanding Reviewer Award at ACL 2021 and EMNLP 2020. 
 | 

 
 
 
 
 | 
 
 
 Yi R.(May) Fung is a Tenure-Track Assistant Professor at the Hong Kong University of Science and Technology Department of Computer Science and Engineering. She completed her PhD in computer science at the University of Illinois, with Prof. Heng Ji, after which she spent time visiting MIT as a postdoctoral affiliated researcher at Prof. Paul Liang’s lab. May leads research on multimedia knowledge reasoning, misinformation detection, and computation for the social good with human-centered, scalable oversight principles. May is a recipient of the ACL’24 Outstanding Paper Award, NAACL’24 Outstanding Paper Award, NAACL’21 Best Demo Paper, the UIUC Lauslen and Andrew fellowship, and the UIUC Yunni Maxine Pao Memorial Fellowship. She has also been previously selected for invited talks at the Harvard Medical School Bioinformatics Seminar and the Hong Kong University Junior Scholar Seminar. She has taught two tutorials at conferences, including KDD’22: The Battlefront of Combating Misinformation and Coping with Media Bias . 
 | 

 
 
 
 
 | 
 
 
 Haoyi Qiu is a first-year PhD student in Computer Science at UCLA advised by Prof. Nanyun (Violet) Peng. Prior to that, she graduated from the University of Michigan, with a B.S. in Computer Science and Math, advised by Prof. Joyce Y. Chai. Her research focuses on making AI more trustworthy and improve its factuality and safety. She is a recipient of the Outstanding Reviewer Award at EMNLP 2023. 
 | 

 
 
 
 
 | 
 
 
 Mingyang Zhou is an applied researcher at Capital One. Previously, he was a post doctoral research scientist a the Department of Electrical Engineering, Columbia University and he obtained his Ph.D degree from University of California, Davis. His research interest lies in vision and language pre-training and visual dialogue system. His recent works include the ChartT5 chart understanding model, and the first work on Unsupervised Vision-and-Language Pre-training via Retrieval-based Multi-Granular Alignment. He is a winner of Alexa Social Bot Challenge in 2018. 
 | 

 
 
 
 
 | 
 
 
 Shafiq Joty is a research director at Salesforce Research and an Associate Professor at the Computer Science Department of the Nanyang Technological University. His research has contributed to over 30+ patents and 140+ papers in top-tier NLP and ML conferences and journals. He has served as the Program Chair of SIGDIAL’23, a member of the best paper award committees for ICLR’23 and NAACL’22, and in the capacity of a (senior) area chair for many leading NLP and ML conferences (e.g. NeurIPS, EMNLP, and ACL). He previously gave tutorials at EMNLP’23, IEEEVis’22, ACL’19, COLING’18 and ICDM’18. 
 | 

 
 
 
 
 | 
 
 
 Shih-Fu Chang is the Dean of Columbia Engineering and the Morris A. and Alma Schapiro Professor in the Electrical Engineering Department and Computer Science Department. A primary goal of his work is to develop intelligent systems that can extract rich information and knowledge from diverse data of multiple modalities, including images, video, language, and audio. His scholarly impacts can be seen in peer-reviewed publications, best paper awards, 30+ issued patents, and technology transfer. For his long-term contributions, he was awarded the IEEE Signal Processing Society Technical Achievement Award, ACM SIGMM Technical Achievement Award, the Honorary Doctorate from the University of Amsterdam, and the IEEE Kiyo Tomiyasu Award. He received the Great Teacher Award from the Society of Columbia Graduates. He served as Chair of ACM SIGMM, Chair of Columbia Electrical Engineering Department. He is a Fellow of the American Association for the Advancement of Science (AAAS), ACM, and IEEE, and an elected Academician of Academia Sinica. 
 | 

 
 
 
 
 | 
 
 
 Heng Ji is a professor at Computer Science Department, and an affiliated faculty member at Electrical and Computer Engineering Department and Coordinated Science Laboratory of University of Illinois Urbana-Champaign. She is an Amazon Scholar and the Founding Director of Amazon-Illinois Center on AI for Interactive Conversational Experiences. Her research interests focus on Natural Language Processing, especially on Multimedia Multilingual Information Extraction, Knowledge-enhanced Large Language Models, Knowledge-driven Generation, and Conversational AI. She was selected as “Young Scientist” by the World Economic Forum in 2016 and 2017 and was named as part of Women Leaders of Conversational AI (Class of 2023) by Project Voice. The awards she received include “AI’s 10 to Watch” Award by IEEE Intelligent Systems in 2013, NSF CAREER award in 2009, PACLIC2012 Best paper runner-up, “Best of ICDM2013” paper award, “Best of SDM2013” paper award, ACL2020 Best Demo Paper Award, NAACL2021 Best Demo Paper Award, Google Research Award in 2009 and 2014, IBM Watson Faculty Award in 2012 and 2014 and Bosch Research Award in 2014-2018. 
 |