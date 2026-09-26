Millimeter Wave Sensing: A Review of Application Pipelines and Building Blocks 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2012.13664v1 [eess.SP] 26 Dec 2020 
 
 
 
 
 
 

 

# Millimeter Wave Sensing: A Review of Application Pipelines and Building Blocks

 
 
 Bram van Berlo
 
    
 Amany Elkelany
 
    
 Tanir Ozcelebi
 
    
 Nirvana Meratnia
 † † thanks: B.R.D. van Berlo, A.M.A. Elkelany, T. Ozcelebi, and N. Meratnia are with the Interconnected Resource-aware Intelligent Systems cluster, Department of Mathematics and Computer Science, Eindhoven University of Technology, 5600 MB Eindhoven, The Netherlands (e-mail: b.r.d.v.berlo@tue.nl; a.m.a.elkelany@tue.nl; t.ozcelebi@tue.nl; n.meratnia@tue.nl). Correspondence should be addressed to B.R.D. van Berlo. 

 Abstract 
 
 The increasing bandwidth requirement of new wireless applications has lead to standardization of the millimeter wave spectrum for high-speed wireless communication. The millimeter wave spectrum is part of 5G and covers frequencies between 30 and 300 GHz corresponding to wavelengths ranging from 10 to 1 mm. Although millimeter wave is often considered as a communication medium, it has also proved to be an excellent ‘sensor’, thanks to its narrow beams, operation across a wide bandwidth, and interaction with atmospheric constituents. In this paper, which is to the best of our knowledge the first review that completely covers millimeter wave sensing application pipelines, we provide a comprehensive overview and analysis of different basic application pipeline building blocks, including hardware, algorithms, analytical models, and model evaluation techniques. The review also provides a taxonomy that highlights different millimeter wave sensing application domains. By performing a thorough analysis, complying with the systematic literature review methodology and reviewing 165 papers, we not only extend previous investigations focused only on communication aspects of the millimeter wave technology and using millimeter wave technology for active imaging, but also highlight scientific and technological challenges and trends, and provide a future perspective for applications of millimeter wave as a sensing technology. 

 
 
 
 Index Terms:  5G, analytical modeling, millimeter wave, millimeter wave sensing application pipeline, radar, systematic literature review.
 
 
 

## I Introduction 

 
 Introduction of new wireless applications, their higher service quality requirements, and a significant increase of application users have put continuous demand on digital wireless communication bandwidth. To address this, Netflix and YouTube, for instance, have reduced video streaming quality in Europe to mitigate data traffic peaks in the 2.4 and 5 GHz WiFi bands. These peaks are caused by an increased amount of video and movie streams due to home confinement measures taken during the COVID-19 pandemic  [ 1 ] . In the future, 8K video streaming will require a minimum data rate of 50 Mbps per TV. In case multiple simultaneous streams are launched on the same network, gigabit connections will approach their limit  [ 2 ] . Maintaining the required network bandwidth and processing speed remains challenging for cloud gaming as a service  [ 3 ] .

 
 
 The increasing bandwidth requirement has lead to standardization of the millimeter wave spectrum for high-speed wireless communication in the IEEE 802.11ad, 802.11aj, and 802.11ay amendments  [ 4 , 5 , 6 ] . The millimeter wave is also part of the fifth-generation mobile communication technology (5G)  [ 7 , 8 ] . Although use of the millimeter wave spectrum for communication is often directly associated with 5G, there are key differences. The millimeter wave spectrum is just one part of what 5G networks use to provide higher data rates. It covers frequencies between 30 and 300 GHz corresponding to wavelengths ranging from 10 to 1 mm  [ 9 , 10 ] , located between the centimeter wave and terahertz wave spectrums.

 
 
 Thanks to covering a wide bandwidth and utilizing a short wavelength, the millimeter wave spectrum is able to provide higher data rates compared to other widely used wireless technologies such as WiFi. Compared to the spectrum in which 5 GHz WiFi operates, the bandwidth of the millimeter wave spectrum is 10x higher, i.e., 270 GHz versus 27 GHz. Under the assumption that environmental and legislative limitations do not exist, this means that the number of channels that can be created in the millimeter wave spectrum is also 10x more. Because millimeter waves operate at much higher frequencies compared to the WiFi bands, the wavelengths are much shorter. Consequently, the size of electronic components can be reduced. This, in turn, causes the beam emitted by electronic components to be much narrower. The narrow beams, in combination with greater signal attenuation compared to the WiFi bands, allow increased communication density  [ 10 ] , i.e., number of messages communicated in parallel over separate links with the same carrier frequency in a limited area.

 
 
 Although millimeter wave is often considered as a communication medium, it has also proved to be an excellent ‘sensor’ for humans, objects, and environmental sensing  [ 10 , 9 ] , thanks to its narrow beams, operation across a wide bandwidth, and interaction with atmospheric constituents. Narrow beams result in greater sensing resolution and directivity. A wide bandwidth and specific penetration, reflection, and attenuation reactions to different materials allow distinguishing different objects and humans  [ 11 , 12 ] .

 
 
 However, using millimeter waves in sensing applications also has disadvantages. Signals at extremely high frequencies suffer from significant attenuation. Millimeter waves can therefore hardly be used for long-distance applications  [ 13 ] . The financial cost of deploying a millimeter wave sensing system, e.g., on a vehicle, is still high even though this cost is projected to decrease in the future  [ 14 ] . Millimeter waves also suffer significant penetration loss through solid materials like concrete. As rain-drops are comparable in size to the wavelength of millimeter waves, heavy rain can cause considerable attenuation due to scattering  [ 10 , 9 ] .

 
 
 The applications that use millimeter waves utilize a wide variety of hardware and algorithms for data collection, data pre-processing, feature extraction, feature analysis, analytical modeling, and modeling evaluation. This paper provides a comprehensive overview and analysis of millimeter wave sensing application pipelines and basic building blocks, including hardware, algorithms, analytical models, and model evaluation techniques, offers an insight into challenges and trends, and provides a future perspective for applications of millimeter wave as a sensing technology. Therefore, the review has struck a balance between the amount of details that are provided for each millimeter wave sensing application pipeline building block and does not elaborate on details of each application domain and algorithm independently. Due to the multidisciplinary nature of millimeter wave sensing application pipelines, the review offers technical details for each reader depending on familiarity with a certain field. For example, data collection and a part of data pre-processing are topics of electrical engineering, while analytical modeling and modeling evaluation are topics of data science and artificial intelligence. The main contributions of the paper are:

 
 
 
 • 
 
 Extending previous investigations focused only on communication aspects of the millimeter wave technology and using millimeter wave technology for active imaging. To the best of our knowledge, this is the first review that completely covers millimeter wave sensing application pipelines and pipeline building blocks.

 

 • 
 
 Providing a taxonomy that highlights different millimeter wave sensing application domains.

 

 • 
 
 Making a comprehensive review and analysis of hardware, algorithms, analytical models, and model evaluation techniques for each of the identified millimeter wave sensing application pipeline building blocks.

 

 • 
 
 Identifying commonalities, gaps, and shortcomings of current studies and solutions focusing on the use of millimeter wave as a sensing technology.

 

 • 
 
 Highlighting scientific and technological challenges and trends, and providing a future perspective for applications of millimeter wave as a sensing technology.

 

 
 
 
 The review is organized as follows: Section  II explains the review methodology. Section  III establishes an application taxonomy based on the research papers that report on millimeter wave sensing. Section  IV identifies the common building blocks that make up the application pipelines presented in the millimeter wave sensing papers. It also presents a thorough analysis of the wide variety of hardware, algorithms, analytical models, and model evaluation techniques across these papers in the respective building blocks. Section  V identifies and explains the challenges and trends, and provides a future perspective for applications of millimeter wave as a sensing technology. Finally, Section  VI concludes the paper.

 
 
 

## II Review Methodology 

 
 We have conducted this review with a specific process involving four sequential phases. These phases are depicted in Figure  1 . The process is based on the iterative Bioinformatic and Systematic Literature Review (BiSLR) spiral model developed by Mariano et al.  [ 15 ] . The differences compared to the iterative BiSLR spiral model are explained below. Afterwards, the four sequential process phases are explained.

 
 
 Fig. 1: Review process block diagram 
 
 
 The iterative BiSLR spiral model starts with a protocol definition phase. In this phase, main and specific research questions, research objectives, and inclusion and exclusion criteria are defined  [ 15 ] . We omit the definition of main and specific research questions. The defined research objectives can be found in Section  I . Inclusion and exclusion criteria are defined in a later paper filter phase. During the reference collection phase, the iterative BiSLR spiral model suggests to select scientific databases, develop and evaluate search keywords, and iteratively repeat the phase with different scientific databases and search keywords in case the research objectives cannot be reached  [ 15 ] . We initially performed a rigurous reference collection, i.e., the search phase. Later during the pipeline building block definition phase, several small search phases were performed iteratively. In the data evaluation phase, the iterative BiSLR spiral model performs title, abstract, diagonal, and full-text reading to determine if papers should be included in the list of papers used for data collection. The iterative BiSLR spiral model finishes with data collection and narrative synthesis  [ 15 ] . We used title, abstract, and diagonal reading for, and collected data during, the explorative paper classification phase and used full-text reading, collected data, and performed narrative synthesis during the application pipeline building block definition phase.

 
 
 In the first phase, we searched for relevant papers published and indexed in digital libraries such as IEEE Xplore, ACM, Elsevier, and SPIE. Because millimeter wave technology is used across different sensing applications that use different terminologies, we used diverse search terminologies. Therefore, finding relevant papers required using synonyms of terms, which we categorized as exact, similar, narrower, and broader terms. Similar terms for ‘millimeter wave’ include ‘mmwave’, ‘mm-wave’ and ‘mmw’. Narrower terms include ‘v-band’, ‘w-band’, ‘Ka-band’, ‘Ka band’, ‘v band’ and ‘w band’. Broader terms include ‘microwave’, ‘micro-Doppler’, ‘Doppler’ and ‘radar’. Several gesture recognition papers only refer to the specific technology used for tracking in the title. The narrower term for gesture recognition used in various papers is ‘Soli’. Term combinations used for searching were made with exact, similar, and narrower terms and consisted of a millimeter wave term and application type term. Based on the application domains addressed in the papers, we made an application taxonomy shown in Figure  2 . Application type terms were derived from this taxonomy. Term combination examples include ‘millimeter wave tracking’, ‘mm-wave gesture’, and ‘w-band detection’. Broader millimeter wave terms such as ‘microwave’ and ‘radar’ resulted in too many hits that mostly included papers outside the scope of this review. Several millimeter wave sensing papers used in the review identify their content with broader millimeter wave terms. These papers were cited in the reference lists of millimeter wave papers found during paper search.

 
 
 In the second phase we classified over 140 papers found during the first phase to identify relevant information. The classification was based on the most commonly identified elements, such as application domain addressed, objectives, target group, dataset size, modeling technique, methodology, deployment environment, evaluation parameters, and opportunities and challenges. The research papers featured varying publication dates from the year 1994 up until 2020. Based on the results obtained in the explorative phase, we made an application taxonomy (see Section  III ) as well as an application pipeline consisting of several generic pipeline building blocks used in these papers.

 
 
 The third phase was used to determine inclusion criteria for this review. After studying over 140 papers, we discovered some papers claiming to cover millimeter wave systems while they do not. In some of these papers, the carrier frequency utilized falls outside the 30 - 300 GHz band  [ 16 , 17 ] . We also excluded papers written in other languages than English  [ 18 ] . Papers focusing on radiometry (passive sensing and imaging)  [ 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 ] , spectroscopy  [ 35 , 36 ] , interferometry  [ 37 ] , and utilization of waveguide technology  [ 38 , 39 , 40 ] were also excluded. Radiometry and spectroscopy do not focus on actively emitting and measuring effects on millimeter waves for sensing. The waveguide technology has been used to design electric probes for interacting with membrane systems, corrosion, sintering process, etc.  [ 38 , 39 , 40 ] . The technology has not been used explicitly for guiding millimeter waves to and from an antenna. Interferometry, rather than measuring effects on millimeter waves, measures wave interference using a multi-radar setup with special radar configurations  [ 37 ] for sensing.

 
 
 Radiometry measures electromagnetic radiation originating from humans or objects with receiver setups  [ 41 ] . One group of radiometers creates passive millimeter wave images. These images are bi-dimensional radiation maps of a scene. Several studies focus on the creation of these systems  [ 23 , 34 ] . In addition, several studies perform multiple object detection and tracking  [ 20 ] , hidden object detection  [ 19 , 21 , 22 , 25 , 28 , 31 ] , and military target detection  [ 26 ] with analytical models that take passive millimeter wave images as input data. Another group of radiometers generates one dimensional radiation profiles. In  [ 30 ] , a passive radiometric temperature profile is a one dimensional output voltage signal which can be generated in either power detection mode or correlation mode. In power detection mode, the output signal depends on antenna temperature which is proportional to the radiometric temperature of humans or objects within the radiometer’s Field of View (FOV) . In correlation mode, output signals from two receivers in power detection mode are used in a correlator to produce an output signal which does not contain objects which mostly reflect and scatter radiation rather than radiate it. The detection capability of detection algorithms that can be used with radiometric temperature profiles has also been tested  [ 30 ] . Millimeter wave hardware and a prediction algorithm that can generate human body emitted energy output traces and detect weapons and explosives  [ 24 , 33 ] have also been created. Yujiri et al.  [ 27 ] measure radiation temperature profiles of buried mines. Other uses of radiometry include analysis of sun brightness temperature and precipitating cloud extinction by means of analytical modeling and a sun-tracking radiometer  [ 29 ] , and creation of an analytical model for predicting relative humidity profiles from clear-air radiances  [ 32 ] .

 
 
 Spectroscopy is a kind of radiometry in which the interaction between radiation and matter is measured  [ 36 ] . Schmalz et al.  [ 35 ] use gas spectroscopy to perform breath analysis. Schmalz et al. explain that “a typical gas spectrometer consists of a radiation source, an absorption cell, a detector, and optical elements. The radiation is transmitted through an absorption cell, which is filled with a gas at a particular pressure and impinges on a detector, which generates an output voltage or current”  [ 35 ] . Spectroscopy has also been used to analyze solar system objects by probing temperature and molecular abundance in planetary atmospheres  [ 36 ] .

 
 
 In the fourth phase, we defined the application pipeline building blocks. For every building block, we identified, reviewed, and analyzed the designed (or used) algorithms, models, hardware, and summarized findings in tables that map to the individual papers. Using the summary tables, paragraphs, comparative tables, and figures were formulated (see Section  IV ) that explain the application pipeline building blocks. Occasionally, during paragraph, comparative table, and figure formulation, summary table deficiencies and errors in the summary table were discovered. These deficiencies and errors were corrected by revising the summary table and afterwards updating the associated paragraphs, comparative tables, and figures. This was an iterative process, during which extra papers were searched (phase 1) to cover the field as much as possible and to include all relevant papers. As a result, 25 additional papers were found (on top of the first 140 papers). These papers were inspected with the inclusion criteria (phase 2) and used during application pipeline building block definition. Therefore, 165 papers in total were analyzed during the literature review.

 
 
 Fig. 2: Taxonomy of millimeter wave sensing applications 
 
 
 

## III Application Taxonomy 

 
 Millimeter wave sensing applications can be classified based on the entity (i.e. human, object, or the environment) being sensed, application goal, and application context. Figure  2 represents the applications that we have identified based on these criteria. Application types and goals in sensing humans with millimeter waves include: identification, position tracking, action recognition, health-related monitoring, sudden and harmful event detection, and acquiring speech data. Application types and goals in sensing objects include: object identification and classification, position tracking, object inspection, and monitoring information derived from object characteristics. The environment is sensed in applications whose types and goals have been mainly focused on detection of harmful events related to space landing and airport runways.

 
 
 TABLE I: Summary of data collection devices using radar technology. The modulation scheme and measured variable abbreviations are explained throughout Section  IV-A2 . 
 
 
 
   | 

 
 | 
 \Block 1-10 Carrier frequency (GHz) | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 \Block 1-5 Modulation scheme | 
 | 
 | 
 | 
 | 
 \Block 1-10 Measured variables | 
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
 30-39 | 
 40-49 | 
 50-59 | 
 60-69 | 
 70-79 | 
 80-89 | 
 90-99 | 
 120-129 | 
 160-169 | 
 220-229 | 
 P | 
 RN | 
 BPSK | 
 CW | 
 FMCW | 
 R R | 
 v r v_{r} | 
 θ \theta | 
 φ \varphi | 
 
 
 Δ ​ ϕ t / f d \Delta\phi_{t}/f_{d} across time 
 | 
 
 
 Reflective intensity 
 | 
 
 
 Attenuation power 
 | 
 
 
 IF amplitude 
 | 
 
 
 Local min/max received signal amplitudes 
 | 
 RSS | 

 
 [ 42 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 43 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 44 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 45 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 46 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 47 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 48 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 49 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 50 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 51 ] | 
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
 X | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 52 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 53 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 54 ] | 
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
 X | 
 | 
 | 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 55 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 56 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 57 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 58 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 59 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 60 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 61 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 62 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 63 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 64 ] | 
 X | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 65 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 66 ] | 
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
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 67 ] | 
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
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 68 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 69 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
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

 
 [ 70 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 71 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 72 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 73 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 74 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 75 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 76 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 77 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 78 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 79 ] | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 80 ] | 
 | 
 | 
 | 
 X | 
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
 | 
 | 

 
 [ 81 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 82 ] | 
 | 
 | 
 X | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 83 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 84 ] | 
 | 
 | 
 | 
 | 
 X | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 85 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 86 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 87 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 88 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 89 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 90 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 91 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 
 TABLE I: Continued 
 
 
 
   | 

 
 | 
 \Block 1-10 Carrier frequency (GHz) | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 \Block 1-5 Modulation scheme | 
 | 
 | 
 | 
 | 
 \Block 1-10 Measured variables | 
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
 30-39 | 
 40-49 | 
 50-59 | 
 60-69 | 
 70-79 | 
 80-89 | 
 90-99 | 
 120-129 | 
 160-169 | 
 220-229 | 
 P | 
 RN | 
 BPSK | 
 CW | 
 FMCW | 
 R R | 
 v r v_{r} | 
 θ \theta | 
 φ \varphi | 
 
 
 Δ ​ ϕ t / f d \Delta\phi_{t}/f_{d} across time 
 | 
 
 
 Reflective intensity 
 | 
 
 
 Attenuation power 
 | 
 
 
 IF amplitude 
 | 
 
 
 Local min/max received signal amplitudes 
 | 
 RSS | 

 
 [ 92 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 93 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 94 ] | 
 X | 
 | 
 | 
 X | 
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
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 95 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 96 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 97 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 98 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 99 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 100 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 101 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 102 ] | 
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
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 103 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 104 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
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
 
 
 X 
 | 
 | 

 
 [ 105 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 106 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 107 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 108 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 109 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 110 ] | 
 X | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 111 ] | 
 | 
 X | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 112 ] | 
 X | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 113 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 114 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 115 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 116 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 117 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 118 ] | 
 X | 
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
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 119 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 120 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 121 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 122 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 123 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 124 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 125 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 126 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 127 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 128 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 129 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 130 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 131 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 132 ] | 
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
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 133 ] | 
 | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 134 ] | 
 | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 135 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 

 
 [ 136 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 137 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 138 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 139 ] | 
 | 
 | 
 | 
 X | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 140 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 X | 
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

 

 
 
 

## IV Application Pipeline 

 
 This section presents the five common building blocks found in application pipelines of the reviewed papers: data collection, pre-processing, feature extraction, analytical modeling, and modeling evaluation. For each building block, we review and analyze relevant papers and provide a comparative table highlighting their focus. We use these tables to identify commonalities and gaps of employed techniques and methodologies and to highlight challenges.

 
 

### IV-A Collection Systems 

 
 The first building block in millimeter wave sensing applications is data collection, in which millimeter wave related measurements are collected using a variety of different measurement systems. In this section, we first briefly explain different data collection approaches and their fundamentals, including suitable antenna types and designs, and then describe the variables that are measured. Hardware calibration  [ 141 , 65 , 90 , 94 , 96 , 142 ] and physical noise reduction  [ 109 ] are considered to be outside the scope of this paper and will therefore not be addressed.

 
 

#### IV-A 1 Suitable Antenna Types and Designs

 
 Among the papers reporting on measurement systems, a variety of different suitable antenna types and designs have been identified that are used in these measurement systems. To prevent confusion in later sections, we decouple the concept of transmitters and receivers from transmit and receive antenna’s in the elaboration on suitable antenna types and designs. This means that any transmitter or receiver has a certain amount (at least one) of transmit or receive antenna’s depending on the antenna type and design. Suitable antenna types include, but are not limited to, horn antenna  [ 143 , 144 , 145 , 146 , 64 , 65 , 68 , 147 , 74 , 148 , 149 , 150 , 102 , 151 , 142 , 109 , 152 , 116 , 153 , 117 , 118 , 119 , 154 , 155 ] , yagi antenna  [ 149 ] , lens antenna  [ 59 , 105 ] , reflectarray antenna  [ 85 , 86 ] , cassegrain antenna  [ 88 , 113 ] , (on-chip integrated) patch antenna  [ 49 , 54 , 156 , 157 , 82 , 151 , 127 , 128 , 139 ] , and parabolic antenna  [ 110 , 112 ] . Suitable antenna designs include, but are not limited to, phased array antenna design  [ 158 , 159 , 160 , 144 , 145 , 147 , 161 , 162 ] , frequency scanned antenna design  [ 87 ] , fan and pencil beam antenna design  [ 119 ] , gaussian beam antenna design  [ 100 ] , and omni-directional antenna design  [ 146 ] .

 
 
 By far most measurement systems identified in the papers either use on-chip integrated patch, i.e. microstrip or printed, antennas or series fed patch antenna arrays. Even though this is not evident when looking at the suitable antenna types and designs that are actually reported, many papers list commercial measurement systems, without listing the used antenna type and design, that use on-chip integrated patch antennas or series fed patch antenna arrays. A patch antenna is advantageous compared to other antenna types because it is low cost, can easily be integrated in printed circuit boards of various sizes, and is easy to mass produce  [ 10 ] . A series fed patch antenna array is an array of multiple patch antenna’s connected via a single feed line in series and is used to create a beam pattern with certain characteristics that cannot be created with a single patch antenna  [ 163 , 164 ] .

 
 
 The phased array antenna design is an antenna array design in which densely packed small gain antenna’s are phase shifted by a separate analog or digital phase shift module to create an overall high gain beam that can be steered without mechanically rotating the antenna’s  [ 163 ] . This antenna array design is very important in millimeter wave communication systems since, due to experiencing more penentration loss than centimeter wave communication systems and solely relying on Line of Sight (LOS) propagation, transmitter and receiver beams constantly have to be re-aligned  [ 10 ] . This antenna array design can also be used in ubiquitous sensing systems that rely on rotation for measuring certain variables to steer the sensing system to a sensing position of interest without experiencing noise caused by mechanical rotation. More information regarding the other antenna types and designs can be found in  [ 10 , 163 ] .

 
 
 

#### IV-A 2 Radar

 
 Radars emit electromagnetic radiation signals, which are either reflected or scattered from targets with a smooth or rough surface, respectively. The differences between the emitted and received signals are of interest to the sensing application  [ 163 ] . To understand the basic principles of the radar approaches, we consider in this section situations in which (i) there is a single object or human in the radar’s FOV and (ii) no noise artifacts exist. We will deal with issues related to multiple targets, noise artifacts, increasing sensing resolution, and isolating phase shift components in Section  IV-B . Table  I presents an overview of papers that used radar for data collection in millimeter wave sensing applications. These papers are compared based on their carrier frequency, modulation scheme, and measured variables, each of which is explained below:

 
 
 
 • 
 
 The carrier frequency indicates what part of the millimeter wave frequency band has been explored and to what extent. It is also strongly correlated to the attainable measurement distance in two ways. Firstly, the higher the carrier frequency, the faster the attenuation of the transmitted signal. Secondly, there are exceptions to the first correlation in the form of several attenuation peaks. For example, applications operating at 60-69 GHz do so under signal absorption of the oxygen in the atmosphere. This allows, for example, mobile phone hand gesture recognition in compact populated areas. Human position tracking applications typically operate at around 70-79 GHz. This frequency interval, compared to 60-69 GHz, allows measurements at greater distances  [ 165 ] .

 

 • 
 
 The modulation scheme dictates which variables the radar can measure and how. Certain modulation schemes have been used to test joint communication and sensing  [ 140 ] .

 

 • 
 
 Measured variables strongly depend on application type and its objective. Identification, position tracking, action recognition, etc. rely on range, velocity and angle information  [ 46 , 47 ] . Health monitoring and speech acquisition rely on phase and Doppler shift information in time  [ 62 , 109 , 113 ] . Object identification and classification employ Intermediate Frequency (IF) signal variations across different IF channels  [ 137 , 139 ] .

 

 
 
 
 The most widely used radar type in millimeter wave sensing applications is the Frequency Modulated Continuous Wave (FMCW) radar, which was commercialized recently with utilization of the Integrated Circuit (IC) technology  [ 166 , 45 , 167 ] . Almost all papers found that use this radar modulate the signal frequency according to a sawtooth pattern, which converts the signal into a continuous stream of chirps  [ 168 ] . A minority of papers modulate the signal frequency according to a triangular pattern  [ 85 , 100 ] . An explanation of this frequency modulation pattern can be found in  [ 163 ] . Sawtooth frequency modulation is depicted in Figure  3 .

 
 
 Fig. 3: FMCW radar frequency sawtooth modulation across time. Adopted from  [ 168 ] . 
 
 
 Fig. 4: Block diagram of fundamental components of CW radar. Adopted from  [ 169 ] . 
 
 
 A chirp is defined as a part of a trigonometric function across a limited time window with length T c ​ h ​ i ​ r ​ p T_{chirp} . In this time window, the signal frequency is increased linearly across a bandwidth B B with slope m m using a voltage-controlled oscillator. The transmitted chirp at transmitter TX is then reflected or scattered from a target and received at receiver RX after a round trip time T r ​ t ​ t T_{rtt} . This causes the received chirp to show a frequency deviation f b f_{b} compared to the transmitted chirp at a specific time instant. An IF signal (a.k.a. beat or baseband signal) operating at constant frequency deviation f b f_{b} can be obtained using a down-conversion mixer and low-pass filter in sequence as shown in Figure  4 . This operation returns a correct output in time window T c ​ h ​ i ​ r ​ p − T r ​ t ​ t T_{chirp}-T_{rtt} , where the transmitted and received chirps overlap  [ 166 , 169 ] . Given a trigonometric function x = A ​ cos ⁡ ( 2 ​ π ​ f ​ t + ϕ ) x=A\cos(2\pi ft+\phi) for the transmitted chirp x t ​ x x_{tx} and the received chirp x r ​ x x_{rx} , the IF signal x i ​ f = [ ( A r ​ x ⋅ A t ​ x ) / 2 ] ​ cos ⁡ [ 2 ​ π ​ ( f t ​ x − f r ​ x ) ​ t + ( ϕ t ​ x − ϕ r ​ x ) ] x_{if}=[(A_{rx}\cdot A_{tx})/2]\cos[2\pi(f_{tx}-f_{rx})t+(\phi_{tx}-\phi_{rx})]   [ 170 ] . Using frequency deviation f b = f t ​ x − f r ​ x f_{b}=f_{tx}-f_{rx} , the range R R between the radar and the target is computed using Equation  1   [ 169 ] . The symbol c c denotes a constant representing the speed of light. Due to its unmodulated constant signal frequency, a Continuous Wave (CW) radar is unable to measure range.

 
 
 

 
 | 
 R = f b ⋅ c 2 ⋅ B T c ​ h ​ i ​ r ​ p R=\frac{f_{b}\cdot c}{2\cdot\frac{B}{T_{chirp}}} | 
 | 
 (1) | 
 

 
 
 Radial velocity v r v_{r} of the target can be obtained by using consecutive chirps separated by time window T c ​ o ​ n T_{con} emitted across a loop. In case the target is in motion, the IF signal resulting from the chirps will experience a significant phase difference Δ ​ ϕ t \Delta\phi_{t} relative to the previous loop. The resulting frequency difference is indiscernible for small motion changes. The radial velocity with one chirp per loop is computed using Equation  2 (left). The symbol λ c \lambda_{c} refers to the radar carrier wavelength. Another approach that can be used to obtain radial velocity is to measure a frequency difference between the transmitted and the received chirps, which is commonly known as the Doppler frequency shift f d f_{d} . The radial velocity is then computed using Equation  2 (right). The symbol f c f_{c} refers to the radar carrier frequency. In case of FMCW modulation, the carrier frequency and the wavelength refer to the starting frequency and the wavelength of a transmitted chirp  [ 166 , 169 ] . Other modulation techniques can measure the phase difference, the Doppler shift, and the radial velocity by using the same techniques and associated equations. For example, Pulse (P) modulation can measure a Doppler shift between a pair of transmitted and received pulses  [ 171 ] . The authors in  [ 64 ] recognize that when using a bistatic radar configuration, in which the transmitter and the receiver are placed perpendicular to each other, Doppler shifts due to yaw, pitch, and roll movements of the head are more clearly distinguishable. For radars using an In/Quadrature (I/Q) -phase mixer, the phase difference Δ ​ ϕ t , I/Q = tan − 1 ⁡ ( x q / x i ) \Delta\phi_{t,\lx@glossaries@gls@link{acronym}{iq}{{{}}I/Q}}=\tan^{-1}(x_{q}/x_{i}) . Symbols x q , x i x_{q},x_{i} denote the quadrature-phase and in-phase IF signals coming from the I/Q mixer  [ 59 , 65 , 113 ] .

 
 
 

 
 | 
 v r = λ c ​ Δ ​ ϕ t 4 ​ π ​ T c ​ o ​ n \displaystyle v_{r}=\frac{\lambda_{c}\Delta\phi_{t}}{4\pi T_{con}} | 
 | 
 v r = f d ⋅ c 2 ⋅ f c \displaystyle v_{r}=\frac{f_{d}\cdot c}{2\cdot f_{c}} | 
 | 
 (2) | 
 

 
 
 The Angle of Arrival (AoA) of the reflected or scattered signal in both the elevation and the azimuth dimensions (bearing angle) is calculated based on Multiple Input Multiple Output (MIMO) radar principles  [ 172 , 130 , 173 ] . Rather than having one transmitter and one receiver, multiple transmitters and receivers are used. Almost all radars explained in the papers use a uniform linear layout. An example can be found in Figure  5 . A minimum redundancy layout  [ 58 ] and a uniform circular  [ 118 ] layout are also explored. Real and virtual receivers, spaced according to a matrix structure on a horizontal surface, observe signal reflections coming from a target with a distance d d apart from one another along the front and/or right direction. The signal reflection observed at a given receiver has to travel a certain distance further or shorter along a vertical and/or horizontal direction in relation to one specific receiver in the array to arrive at the receiver. This distance in the vertical and/or horizontal direction is unique to every receiver. The difference in distance between two neighboring receivers in the vertical or horizontal direction directly corresponds to a phase shift component denoted by Δ ​ ϕ ∈ { x , z } = [ 2 ​ π ​ d ​ sin ⁡ ( θ ) ] / λ c \Delta\phi_{\in\{x,z\}}=[2\pi d\sin(\theta)]/\lambda_{c} . A phase shift component overview is presented in Figure  5 . The azimuth or elevation angle is calculated by using Equation  3   [ 166 ] with the phase shift component in the horizontal or vertical direction. Some approaches use multiple radars. For example,  [ 85 , 103 ] use azimuth angles from two different radars, in which one was rotated 90 degrees counter-clockwise compared to the other. The rotation makes the azimuth angle correspond to the elevation angle. The data collection systems in  [ 85 , 86 , 132 ] rotate the radar to obtain angle measurements.

 
 
 Fig. 5: Uniform linear MIMO radar layout including two transmitters and receivers. Partly adopted from  [ 172 ] . 
 
 
 

 
 | 
 θ = sin − 1 ⁡ ( λ c ​ Δ ​ ϕ x 2 ​ π ​ d ) \displaystyle\theta=\sin^{-1}\left(\frac{\lambda_{c}\Delta\phi_{x}}{2\pi d}\right) | 
 | 
 φ = sin − 1 ⁡ ( λ c ​ Δ ​ ϕ z 2 ​ π ​ d ) \displaystyle\varphi=\sin^{-1}\left(\frac{\lambda_{c}\Delta\phi_{z}}{2\pi d}\right) | 
 | 
 (3) | 
 

 
 
 A Random Noise (RN) radar transmits noise signals. Binary Phase-shift Keying (BPSK) radars are normally used for communicating a bit stream. The bit value determines the phase offset ϕ \phi (0 or π \pi ) of the transmitted signal x t ​ x x_{tx} . Both radar types use cross-correlation between the transmitted and received signals. This cross-correlation relates to a time delayed version (delayed by 2 ​ R / c 2R/c ) of the transmitted signal to obtain an auto-correlation function. When plotted, this function shows a peak that corresponds to a target’s range  [ 102 , 140 ] . The noise radar in  [ 102 ] measures a Doppler frequency shift based on a tone embedded into the noise signal.

 
 
 TABLE II: Summary of data collection devices using active imaging approach. The modulation scheme, measured variable and scanning method abbreviations are explained throughout Section  IV-A3 . 
 
 
 
   | 

 
 | 
 \Block 2-7 Carrier frequency (GHz) | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 \Block 
 2-4 Scanning method 
 | 
 | 
 | 
 | 
 \Block 1-3 Modulation | 
 | 
 | 
 
 \Block 
 2-9 Measured variables 
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
 | 
 | 
 \Block 1-3 scheme | 
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
 30-39 | 
 50-59 | 
 60-69 | 
 70-79 | 
 80-89 | 
 90-99 | 
 260-269 | 
 
 
 Mechan- ical line array 
 | 
 
 
 Mechan- ical plane 
 | 
 
 
 Mechanical, single radar 
 | 
 
 
 Free space 
 | 
 P | 
 AM | 
 FMCW | 
 
 
 Scattering signal power distribution 
 | 
 
 
 Scattering coefficient 
 | 
 
 
 Scatter- ing signal 
 | 
 
 
 Visibility function 
 | 
 
 
 IF signal 
 | 
 v v | 
 
 
 mean elevation 
 | 
 
 
 roughness property 
 | 
 
 
 Reflection intensity 
 | 

 
 [ 141 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 174 ] | 
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
 
 
 X 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 156 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 175 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 176 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 177 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 161 ] | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 

 
 [ 149 ] | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 151 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 

 
 [ 142 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 178 ] | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 152 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 154 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 
 
 X 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 179 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 
 Pulse radars emit a powerful high gain signal pulse for a short time period. This period is called the pulse width. Afterwards, the radar waits for a given rest time to receive reflections before emitting another pulse. The range can be determined based on the round trip time between the emitted pulse and received reflection of this pulse in the rest time: ( c ⋅ T r ​ t ​ t ) / 2 (c\cdot T_{rtt})/2 . In addition to MIMO radar principles, the azimuth and elevation angles can also be retrieved from the radar’s own directivity compared to a baseline direction denoting north or the ground respectively  [ 163 ] . The authors in  [ 87 ] measure mean velocity based on the correlation between two pulses under an unambiguous pulse interval for a sloped terrain. Pulse radars normally work well in long distance applications  [ 163 ] . One exception to this was found in  [ 104 ] , in which breast cancer detection was performed successfully. The authors in  [ 45 ] observe that pulse radar does not provide enough resolution for tasks requiring high resolution measurements such as gesture recognition.

 
 
 

#### IV-A 3 Active Imaging

 
 Active imaging is a specific type of radar-based approach that obtains measurement results in the form of map-like images  [ 163 ] . The radar scans a given area. For every position, the radar emits a signal, measures the reflections returned, and based on a measured variable assigns, for example, a color or gray-scale value to the pixel corresponding to that position. Most active imaging research between 2002-2014 was focused on data collection devices for monitoring cracks in civil infrastructures, detecting concealed objects on the human body, and measuring hazardous landing terrain  [ 161 , 87 ] . Review papers on active imaging used for detecting explosives and monitoring civil infrastructures include  [ 41 , 180 ] . Active imaging recently received attention again in the context of map-like image deep learning and applying existing techniques using a cost effective method in commercial applications  [ 156 , 176 , 177 , 181 ] for applications such as parking space monitoring and analyzing objects in enclosed packaging for food quality control and non-invasive fault detection. Table  II is an extension of previous reviews  [ 41 , 180 ] , providing an overview of active imaging approaches used for data collection in millimeter wave sensing applications. In addition to the carrier frequency, the modulation scheme, and the measured variables, we consider the scanning method here as well.

 
 
 The scanning method defines how the measurement system moves from one position to another to obtain, for example, a color or gray-scale value for every pixel in the map-like image. One such method is mechanical scanning where radar devices are mechanically moved in fixed directions using a rail system. The literature covers three types of mechanical scanning. In line array scanning, a 1-dimensional array of antenna’s on a straight line scan in a single direction back and forth (i.e., up/down or right/left)  [ 141 , 149 , 151 , 152 ] . In plane scanning, antennas arranged into a 2-dimensional plane scan the target from different distances by moving perpendicular to the plane  [ 156 ] . Some papers employ scanning a target by moving a small radar system in two directions on a virtual plane  [ 175 , 142 , 178 , 154 , 179 ] . In the so called free space scanning method, the forward motion of a radar connected to a moving device, such as a drone or planetary lander, is used to construct images. Sensing application pipelines using this type of scanning either use a large antenna array to cover big areas  [ 161 ] or a limited number of ingrained low-cost radar device antennas  [ 176 , 177 ] .

 
 
 The free space scanning methods found in the literature are commonly referred to as Synthetic Aperture Radar (SAR) . A few millimeter wave application papers present application pipelines containing data collection devices that do not belong to active imaging by means of free space scanning and mention use of SAR   [ 152 ] or SAR pre-processing methods  [ 178 , 154 , 179 ] . We consider these data collection devices not part of SAR since SAR originates from Side Looking Airborne Radar (SLAR) (a free space scanning method). SAR provides a solution to the impractically long antenna or use of extremely short wavelengths, resulting in severe atmospheric attenuation, required for sufficient azimuth resolution in SLAR images. SAR essentially synthesizes a very long antenna to obtain high resolution images  [ 182 ] . More information regarding SAR can be found in  [ 183 , 182 ] . Raw SAR images are severely out of focus since they do not represent spatial information correctly yet and therefore need additional pre-processing which is elaborated on in Section  IV-B2 .

 
 
 TABLE III: Summary of data collection devices using spatial sweeping approach. The modulation scheme, measured variable and sweeping method abbreviations are explained throughout Section  IV-A4 . 
 
 
 
   | 

 
 | 
 \Block 1-3 Carrier frequency (GHz) | 
 | 
 | 
 \Block 1-5 Sweeping method | 
 | 
 | 
 | 
 | 
 \Block 1-8 Measured variables | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 | 
 30-39 | 
 60-69 | 
 70-79 | 
 Fixed | 
 Rotational | 
 Radio Tomography | 
 V2I | 
 Base Station Exchange | 
 CIR | 
 RSS | 
 Phase | 
 Reflection loss | 
 f d f_{d} across time | 
 ToF | 
 AoD | 
 AoA | 

 
 [ 158 ] | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 159 ] | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 

 
 [ 143 ] | 
 | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 160 ] | 
 | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 184 ] | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 

 
 [ 144 ] | 
 | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
 [ 145 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 146 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 

 
 [ 147 ] | 
 | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
 [ 148 ] | 
 | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 162 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 

 
 [ 150 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 185 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 X | 

 
 [ 153 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 186 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 155 ] | 
 | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 
 The IF signal is introduced in Section  IV-A2 . Radars using P modulation obtain an IF signal through a concept called pulse compression  [ 177 , 183 ] , during which the pulse frequency is modulated in a similar pattern as FMCW modulation. Pulse compression allows transmission powers comparable to a pulse with a long pulse width while simultaneously retaining the range resolution which is only attainable with a short pulse width in the context of SAR . Low transmission power results in low receiver signal detectability and measurement precision  [ 187 ] . The IF signal is either used directly in pre-processing  [ 177 ] or its amplitude  [ 176 , 177 , 178 ] , magnitude  [ 175 , 179 ] , and/or phase values  [ 175 , 176 , 177 , 178 ] are used to color the image or for further pre-processing. The authors in  [ 152 ] measure a lap-joint position with an IF signal component. The component accounts for amplitude variation based on radiation characteristics and the target’s shape and a phase variation based on the target’s position at the center frequency.

 
 
 Other less frequently measured variables include a visibility function and a scattering coefficient. Visibility is defined as a correlation value between two non-directive receivers. Both mechanical  [ 156 ] and free space  [ 174 ] approaches have been used for measuring these variables. The scattering coefficient can be considered as an energy ratio between transmitted and received energy  [ 142 ] . When a signal scatters due to a rough surface, more energy will be observed by the receiver compared to when most energy is reflected away from the receiver due to a smooth reflection surface  [ 183 ] .

 
 
 

#### IV-A 4 Spatial Sweeping

 
 Spatial sweeping is an approach where signal response metrics measured in the context of the millimeter wave radio communication between sender and receiver systems are analyzed. Changes in the measured metrics are caused by the interaction between the target of interest to be sensed and the communication signals. Spatial sweeping refers to the observation that almost all collection systems in Table  III have transmitter and/or receiver components that spatially move and rotate during communication. Movement and rotation naturally increase the system’s sensing FOV . Table  III gives an overview of spatial sweeping approaches used for data collection in millimeter wave sensing application. In addition to the carrier frequency and measured variables, we consider here the used sweeping method as well.

 
 
 The sweeping method defines how senders and receivers move and rotate during communication with each other while at the same time being used for measurement procedures. Table  III shows that several sweeping methods are quite generic while others are more specific. This is because certain systems are tested for specific real-life contexts while other systems are tested for determining feasibility of, for example, rotational and fixed communication for sensing. A variety of sweeping methods exist, such as rotational sweeping, fixed sweeping, and radio tomography. Most rotational sweeping methods rely on a transmitter rotating itself to transmit a signal at beam angles within a given area. During this time, receivers make a variable measurement at one angle of arrival if the angle is not in line with the current beam angle. This process is repeated sequentially for a number of angles of arrival to form a measurement matrix  [ 143 , 144 , 147 , 148 ] . The authors in  [ 160 ] use a robot that moves and rotates. The authors in  [ 155 ] use a fixed position where both the transmitter and receiver are located. Measurements are taken at a given location and associated angles  [ 160 , 155 ] . In fixed sweeping, position and direction of both transmitters and receivers are fixed. The authors in  [ 158 ] put a transmitter and receiver in direct line-of-sight, while authors in  [ 159 ] use a quasi-omni transmit antenna to cover a measurement area. Certain rotational sweeping methods fall back to fixed sweeping by rotating a transmitter and receivers towards a fixed angle to perform additional measurements  [ 143 , 144 , 147 ] while the angle remains fixed. Radio tomography uses a grid network of transceivers to cover a measurement area  [ 150 ] . Vehicle to Infrastructure (V2I) communication relies on fast Dedicated Short Range Communication (DSRC) between a stationary Roadside Unit (RSU) and On-Board Unit (OBU) inside a moving vehicle  [ 188 ] . Base station exchange covers a measurement area via a set of moving receivers  [ 184 , 145 , 153 , 186 ] .

 
 
 

#### IV-A 5 Gaps and Challenges

 
 From Tables  I ,  II and  III , a number of interesting observations can be made. Firstly, most data collection systems operate in the 60-79 GHz frequency band. This leaves ample room for performing research and data collection on the 30-59 GHz and 80-299 GHz bands. Secondly, most radar systems stick to variables that are delivered out of the box by commercialized systems. These include range, radial velocity and AoAs . It is yet to be seen and investigated whether additional variables such as IF signal amplitude variation across time and Received Signal Strength (RSS) provide increased sensing accuracy to a wide variety of applications.

 
 
 Fig. 6: Abstract pre-processing pipeline diagram. It does not show spatial sweeping across time (see Section  IV-A4 ). It also omits artifact parts extracted with pre-processing and additional artifact features that are normally not extracted together with the artifact. Artifact part examples include profile, spectrogram, cube bins and radar image parts  [ 189 ] . Feature examples include velocity  [ 124 , 138 ] and intensity  [ 124 ] with point cloud frames and vehicle location  [ 176 , 177 ] , empty parking site  [ 176 , 177 ] , obstacles  [ 177 ] , length-width ratio  [ 176 ] , barycenter location  [ 176 ] , and intensity  [ 103 ] with radar images. 
 
 
 
 

### IV-B Pre-processing 

 
 Using the collection systems mentioned in Section  IV-A , discrete signal vectors in the time domain are sampled via an Analog-to-Digital Converter (ADC) . For the majority of sensing applications, these raw signal vectors cannot be used to infer information of interest to the sensing application. Therefore, raw signal vectors are passed through a pre-processing pipeline to extract data types that can be used to infer information. An abstract pipeline overview can be found in Figure  6 . This section reports on the pre-processing methods that are mentioned in the papers. Simulation with a ray tracing tool  [ 184 , 145 , 162 , 186 ] , single-radar simulation  [ 140 ] , multi-radar simulation  [ 101 ] , imaging simulation  [ 156 ] , and radar data modeling  [ 174 , 190 , 191 ] are considered to be outside the scope of this review. The pipeline overview does not accurately represent the exact flow that every methodology follows and position in the pipeline where artifacts are extracted since parts are skipped, not reported, etc. It covers the predominant processes, flows, and places where artifacts are extracted. The pre-processing space has been divided into two domains: signal and data. The signal domain encompasses all pre-processing methods that are applied directly on the ADC sampled signal vectors. Once signal vectors are passed through a signal transformation method for the first time, the artifacts resulting from these methods are referred to as data and are subsequently processed in the data domain. Several pre-processing methods reported in this section are based on analytical modeling. Because these models are used to execute pre-processing tasks, i.e., not used to extract higher level information relevant to the millimeter wave application, they are considered to be pre-processing methods. Several application pipelines directly retrieve artifacts from the collection system hardware or use a pre-processing pipeline retrieved from a paper. The papers associated to these pipelines do not elaborate on the pre-processing methods that are executed prior to retrieving the artifacts. These pipelines and associated papers are summarized in Table  IV . Several papers elaborate on additional pre-processing methods that are executed after retrieving the artifacts. These methods are explained in the respective pre-processing subsections.

 
 
 TABLE IV: Application pipelines that directly retrieve artifacts from the collection system hardware or use a pre-processing pipeline building block retrieved from a paper. 
 
 
 
 ] | 
 \Block 2-1 Artifact(s) | 
 \Block 2-1 (index types) | 
 Collection | 

 
 | 
 | 
 | 
 method/Paper | 

 
 [ 42 ] | 
 Spectrogram | 
 (range, velocity) | 
 Radar | 

 
 [ 47 ] | 
 Point cloud frame | 
 (x, y, z) | 
 Radar | 

 
 [ 48 ] | 
 Spectrogram | 
 (range, velocity) | 
 Radar | 

 
 \Block 2-1 [ 49 ] | 
 IF I/Q signal, | 
 (time, IF channel), | 
 \Block 2-1Radar | 

 
 | 
 Spectrogram | 
 (range, velocity) | 
 | 

 
 [ 51 ] | 
 R , θ , and  ​ v r R,\theta,\text{and }v_{r} | 
 Undefined | 
 Radar | 

 
 [ 52 ] | 
 Long. and lat. R ​  and  ​ v R\text{ and }v | 
 Undefined | 
 Radar | 

 
 \Block 2-1 [ 56 ] | 
 Profile | 
 (range) | 
 \Block 2-1Radar | 

 
 | 
 θ \theta | 
 Undefined | 
 | 

 
 [ 61 ] | 
 Long. and lat. R ​  and  ​ v R\text{ and }v | 
 Undefined | 
 Radar | 

 
 [ 146 ] | 
 Doppler shift data | 
 (time, TX channel) | 
 [ 192 ] | 

 
 [ 63 ] | 
 Spectrogram | 
 (range, velocity) | 
 Radar | 

 
 [ 181 ] | 
 Radar image | 
 (x, z) | 
 Active imaging | 

 
 [ 193 ] | 
 Radar image | 
 (x, z) | 
 Active imaging | 

 
 [ 67 ] | 
 R , φ , and  ​ v r R,\varphi,\text{and }v_{r} | 
 Undefined | 
 Radar | 

 
 [ 194 ] | 
 Radar image | 
 (x, z) | 
 Active imaging | 

 
 [ 195 ] | 
 Radar image | 
 (x, z) | 
 Active imaging | 

 
 [ 69 ] | 
 Heartbeat data | 
 (time) | 
 Radar | 

 
 [ 189 ] | 
 Radar image | 
 (x, z) | 
 Active imaging | 

 
 [ 72 ] | 
 Spectrogram | 
 (x, z) | 
 [ 196 ] | 

 
 [ 176 ] | 
 Radar image | 
 (x, y) | 
 Active imaging | 

 
 [ 177 ] | 
 Radar image | 
 (x, y) | 
 Active imaging | 

 
 \Block 2-1 [ 77 ] | 
 Point cloud frame | 
 (x, y) | 
 \Block 2-1Radar | 

 
 | 
 R , v r , θ , and power R,v_{r},\theta,\text{and power} | 
 Undefined | 
 | 

 
 [ 81 ] | 
 Spectrogram | 
 (range, velocity) | 
 [ 197 ] | 

 
 [ 83 ] | 
 Spectrogram | 
 (range, azimuth) | 
 Radar | 

 
 [ 84 ] | 
 Profile | 
 (range) | 
 Radar | 

 
 \Block 2-1 [ 87 ] | 
 Range compressed | 
 \Block 2-1(range, IF channel) | 
 \Block 2-1Radar | 

 
 | 
 down-conversion IF data | 
 | 
 | 

 
 [ 89 ] | 
 Point cloud frame | 
 (x, y, z) | 
 Radar | 

 
 [ 91 ] | 
 Spectrogram | 
 (range, velocity) | 
 Radar | 

 
 \Block 2-1 [ 95 ] | 
 Point cloud frame | 
 (x, y, z) | 
 \Block 2-1Radar | 

 
 | 
 v r v_{r} | 
 Undefined | 
 | 

 
 [ 97 ] | 
 Point cloud frame | 
 (x, y, z) | 
 Radar | 

 
 [ 99 ] | 
 R ​  and  ​ θ R\text{ and }\theta | 
 Undefined | 
 Radar | 

 
 [ 198 ] | 
 Reflection intensity | 
 Undefined | 
 Radar | 

 
 \Block 2-1 [ 102 ] | 
 Profile | 
 (range) | 
 \Block 2-1 [ 199 ] | 

 
 | 
 Spectrogram | 
 (time, frequency) | 
 | 

 
 [ 104 ] | 
 Radar trace | 
 Undefined | 
 Radar | 

 
 [ 178 ] | 
 Spectrogram | 
 (x, y) | 
 Active imaging | 

 
 [ 105 ] | 
 Heartbeat data | 
 (time) | 
 [ 200 ] | 

 
 [ 112 ] | 
 Spectrogram | 
 (time, frequency) | 
 [ 201 ] | 

 
 [ 185 ] | 
 Positioning data | 
 Undefined | 
 Spatial sweeping | 

 
 [ 115 ] | 
 R ​  and  ​ θ R\text{ and }\theta | 
 Undefined | 
 Radar | 

 
 [ 153 ] | 
 RSS signal | 
 (time, RX channel) | 
 Spatial sweeping | 

 
 \Block 2-1 [ 117 ] | 
 Profile | 
 (range) | 
 \Block 2-1Mono-pulse pipeline | 

 
 | 
 Waterfall chart | 
 (time, range) | 
 | 

 
 \Block 2-1 [ 118 ] | 
 Dynamic (time-varying) | 
 \Block 2-1Undefined | 
 \Block 2-1Radar | 

 
 | 
 signal components | 
 | 
 | 

 
 \Block 2-1 [ 120 ] | 
 Point cloud frame | 
 (x, y, z) | 
 \Block 2-1 [ 202 ] | 

 
 | 
 v r v_{r} | 
 Undefined | 
 | 

 
 \Block 2-1 [ 124 ] | 
 Point cloud frame | 
 (x, y, z) | 
 \Block 2-1 [ 166 ] | 

 
 | 
 v r ​  and intensity v_{r}\text{ and intensity} | 
 Undefined | 
 | 

 
 [ 125 ] | 
 Spectrogram | 
 (range, velocity) | 
 Radar | 

 
 [ 131 ] | 
 Spectrogram | 
 (x, y) | 
 Radar | 

 
 [ 133 ] | 
 v r ​  and  ​ θ v_{r}\text{ and }\theta | 
 Undefined | 
 Radar | 

 
 [ 134 ] | 
 R , v r , and  ​ θ R,v_{r},\text{and }\theta | 
 Undefined | 
 Radar | 

 
 [ 135 ] | 
 R , v r , θ , and power amplitude R,v_{r},\theta,\text{and power amplitude} | 
 Undefined | 
 Radar | 

 
 [ 136 ] | 
 R , θ , and power amplitude R,\theta,\text{and power amplitude} | 
 Undefined | 
 Radar | 

 
 
 
   | 

 

 
 

#### IV-B 1 Signal Reconstruction and Denoising

 
 Signal reconstruction and denoising resolve signal corruption and spectral leakage. Signal corruption refers to observing a sampled signal that is drastically different from its theoretical definition. Changes are caused by superimposed noise from unwanted stationary and systematic reflections from a target and nearby objects  [ 159 , 144 , 59 , 70 , 157 , 148 , 150 , 104 , 110 , 112 , 113 , 118 ] , high frequency static noise (noise that is concentrated around, and remains in, a high frequency range)  [ 144 , 148 , 116 ] , phase wrapping around a certain value that causes signal jumps  [ 59 , 65 , 70 ] , DC offset  [ 144 , 148 , 116 ] , hardware noise  [ 60 , 110 , 112 , 113 ] , harmonic noise  [ 110 , 113 ] , and channel noise  [ 113 ] . Spectral leakage refers to non-zero values that show up in a signal’s frequency profile at frequencies other than the frequency components actually present in the signal after signal transformation. Signal reconstruction and denoising resolve signal corruption and spectral leakage differently.

 
 
 TABLE V: Summary of signal reconstruction denoising pre-processing methods deployed in millimeter wave sensing pipelines. The denoising abbreviations are explained throughout Section  IV-B1 . 
 
 
 
   | 

 
 | 
 
 \Block 
 1-2 Reconstruction 
 | 
 | 
 
 \Block 
 1-10 Denoising 
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
 
 
 Phase regeneration 
 | 
 
 
 Phase unwrapping 
 | 
 
 
 Mean centering 
 | 
 
 
 Moving average filter 
 | 
 
 
 Highpass filter 
 | 
 
 
 Lowpass filter 
 | 
 
 
 Line fitting 
 | 
 
 
 Bandpass filter 
 | 
 DDBR | 
 
 
 Windowing 
 | 
 
 
 Wavelet Packet Noise Reduction 
 | 
 
 
 Manual background subtraction 
 | 

 
 [ 159 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 

 
 [ 55 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 144 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 59 ] | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 60 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 62 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 64 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 65 ] | 
 | 
 
 
 X 
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

 
 [ 70 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 157 ] | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 147 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 148 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 
 
 X 
 | 

 
 [ 85 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 86 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 96 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 98 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 150 ] | 
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
 
 
 X 
 | 

 
 [ 104 ] | 
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
 
 
 X 
 | 

 
 [ 142 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 110 ] | 
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
 
 
 X 
 | 
 | 

 
 [ 112 ] | 
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
 
 
 X 
 | 

 
 [ 114 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 116 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 118 ] | 
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
 
 
 X 
 | 

 
 [ 119 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 121 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 127 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 130 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
   | 

 

 
 
 Signal reconstruction artificially creates or alters a signal with help from a construction algorithm. Phase regeneration exploits the fact that the received signal phase exhibits a periodic pattern, even though it is not linear with respect to the target’s moving distance. Phase regeneration creates an artificial phase shift signal by counting phase shifts of the measured signal, creating a predefined phase shift on every count, and linking two neighboring counts with a linear increasing or decreasing trend  [ 159 ] . Phase unwrapping is not explained in  [ 59 , 60 , 65 ] .

 
 
 Signal denoising encompasses all methods that remove signal components from the sampled signal. A signal component, in the context of Fourier series, refers to a wave with a less complex waveform in a set of waves with a less complex waveform that reconstitute the sampled signal if summed together. Mean centering  [ 59 ] is done by subtracting the geometric mean vector of a complex output trace from each sampled signal point. A moving average filter averages a number of the sampled signal points to produce a new signal point  [ 203 ] . The high, low, and bandpass filters attenuate signal components from the sampled signal that have a certain frequency. The highpass filter attenuates everything below a certain cutoff frequency, the lowpass filter attenuates everything above a certain cutoff frequency, and the bandpass filter attentuates everything outside of a predefined frequency band. Line fitting estimates a demodulated IF signal phase by fitting a straight line to the phase of a lowpass filtered demodulated IF signal version and obtaining the y-intercept of the demodulated IF signal with the straight line  [ 70 ] . Dual-differential Background Removal. (DDBR) takes two differentials. The first differential is computed with signal points at time t − 1 ​  and  ​ t t-1\text{ and }t and the second differential with signal points at time t ​  and  ​ t + 1 t\text{ and }t+1 . DDBR then adds the differentials to obtain a background canceled signal point. A standard signal window function used in combination with a signal transformation operation such as Short-time Fourier transform (STFT) can be understood as a brick-wall filter  [ 64 ] . Cosine-sum and adjustable windows such as Hann  [ 85 ] , Hamming  [ 142 , 127 ] , Dolph-Chebyshev  [ 86 ] , and Kaiser-Bessel  [ 127 ] windows reduce spectral leakage in a signal transformation output  [ 204 ] . Wavelet packet noise reduction  [ 110 ] removes signal noise by utilizing a signal enhancement scheme on a fast wavelet transformed signal representation and afterwards transforming the signal back to the time domain. The main idea behind manual background subtraction is to first obtain a sampled signal from an environment without any target and afterwards subtracting the sampled signal from a sampled signal obtained when a target is present in the environment  [ 148 , 104 ] . Manual background subtraction in  [ 150 , 112 ] averages multiple signals obtained from an environment without a target. Exponential averaging can also be used to obtain a sampled signal from an environment without a target  [ 118 ] .

 
 
 

#### IV-B 2 Signal Transformation

 
 Signal transformation is the most important pre-processing operation of almost every millimeter wave sensing application pipeline. Without signal transformation, many of the artifacts depicted in Figure  6 would not exist. Signal transformation refers to transforming a time domain signal, through use of mathematical operations, into a data structure that represents certain aspects of the signal in either the time or frequency domain.

 
 
 TABLE VI: Summary of signal transformation pre-processing methods deployed in millimeter wave sensing pipelines. The abbreviations are explained throughout Section  IV-B2 . 
 
 
 
   | 

 
 | 
 
 \Block 
 1-8 Time domain 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 \Block 1-10 Frequency domain | 
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
 
 
 Reflection path extraction 
 | 
 
 
 Impulse response convolution 
 | 
 EMD | 
 Synchronization | 
 
 
 Surface normal calculation 
 | 
 
 
 Peak detection 
 | 
 
 
 Reflection loss extraction 
 | 
 
 
 Range compression 
 | 
 FFT | 
 STFT | 
 Beamformer | 
 MLE | 
 MSER | 
 
 
 Visual saliency detection 
 | 
 
 
 Landmark extraction 
 | 
 
 
 Pulse integration 
 | 
 POSP | 
 
 
 Frequency compression 
 | 

 
 [ 158 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 141 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 

 
 [ 43 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 44 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 45 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 46 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 50 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 53 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 160 ] | 
 
 
 X 
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

 
 [ 55 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 57 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 58 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 144 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 60 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 62 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 64 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 65 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 66 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 68 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 70 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 71 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 73 ] | 
 | 
 
 
 X 
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

 
 [ 147 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 176 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 177 ] | 
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
 X | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 74 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 75 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 76 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 78 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 79 ] | 
 | 
 | 
 X | 
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

 
 [ 80 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 81 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 82 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 161 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 85 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 86 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 88 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 
 
 X 
 | 

 
 [ 90 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 92 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 93 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 94 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 96 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 98 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 100 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 102 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 103 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 151 ] | 
 | 
 | 
 | 
 X | 
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

 
 [ 142 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 178 ] | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 106 ] | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 107 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 108 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 109 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 113 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 152 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 114 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 116 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 118 ] | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 119 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 121 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 154 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 122 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 123 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 126 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 127 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 128 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 129 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 130 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 179 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 131 ] | 
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
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 132 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 

 
 [ 138 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
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

 

 
 
 The most widely used signal transformation operation is the Fast Fourier Transform (FFT) . In Section  IV-A2 , it was explained that an IF signal operating at a certain beat frequency can be obtained when a single object or person is standing in a FMCW radar’s FOV . When multiple objects or persons are standing in the radar’s FOV , the sampled IF signal will exhibit a more complex waveform. The IF signal is a constitution of multiple IF components because multiple reflections are obtained by the receiver channels. These components will have different beat frequencies if objects or persons are standing at different distances. When running the sampled IF signal through a FFT operation across the fast time dimension, the resulting data structure will be a one dimensional set of complex numbers, i.e., a profile. The explanation ignores the complex conjugates in the set of complex numbers  [ 166 ] . In-phase and quadrature-phase IF signals can be combined into a single complex signal  [ 82 ] and used in the FFT operation. The magnitude of several complex numbers will show a peak compared to the other complex numbers. The associated frequency index of these complex numbers directly corresponds to a range value as indicated in Equation  1 . Range resolution, i.e., minimum required distance separation between persons or objects, has a direct relation with the bandwidth size across which chirp frequency is modulated  [ 166 ] .

 
 
 Radial velocity, according to Section  IV-A2 , can be determined with multiple chirps emitted across a loop. The profiles originating from the chirps are stacked on top of each other to form a two dimensional data structure made up of complex numbers. The newly introduced dimension is called slow time (a.k.a. Coherent Processing Interval (CPI) ). Across slow time, the complex number’s phase, due to the motion of persons or objects, rotates at a constant rate. The phase of every complex number is a sum of phases corresponding to different objects or persons  [ 166 ] . After FFT operations have been performed across slow time at every frequency index, a spectrogram (a.k.a. heatmap) indexed by frequency and phase difference is obtained. The phase difference corresponds to a radial velocity value according to Equation  2 . Magnitude peaks in the spectrogram correspond to objects or persons with a certain range travelling at a given radial velocity. Radial velocity resolution, i.e., minimum required radial velocity separation between persons or objects, has a direct relation with the number of chirps per loop  [ 166 ] .

 
 
 AoA , as explained in Section  IV-A2 , is determined through MIMO radar principles. The explanation assumes that spectrograms originating from the real and virtual receiver channels in Figure  5 are available. Transmit techniques required to obtain spectrograms from real and virtual receivers channels are elaborated on in Section  IV-B3 . Azimuth and elevation angles are computed in separate sets of FFT operations. To determine the azimuth angle, the spectrograms originating from real receiver channels RX1 and RX2 are stacked to form a three dimensional data structure. The newly introduced dimension is called IF channel. Across IF channel, like with determining radial velocity, the complex number’s phase rotates. This is due to extra distance which has to be traversed by a signal reflection. After performing FFT operations for every element in the spectrogram across IF channel, a data cube is obtained. Magnitude peaks in the cube correspond to objects or persons with a certain range, radial velocity, and phase shift in the horizontal direction. The phase shift corresponds to azimuth angle according to Equation  3 (left)  [ 166 ] . When looking at Figure  5 , one may assume that the same FFT operations are valid for elevation angle on spectrograms from real receiver channel RX2 and virtual receiver channel RX1. In bigger uniform linear layouts, more signal transformation steps are required to retrieve the elevation angle. A phase shift isolation technique for a bigger uniform linear layout is presented in  [ 205 ] . The main idea is that after several FFT operation steps the elevation phase difference can be isolated by phasor (another way to represent a complex number) multiplication. AoA resolution, i.e., minimum required angle separation between persons or objects, has a direct relation with the number of transmitter and receiver channels used in the MIMO radar layout  [ 166 ] .

 
 
 Other uses of the FFT operation include retrieving Channel State Information (CSI) frequency domain profile  [ 158 ] , conversion of complex scattering coefficient signal to frequency domain  [ 141 ] , aid in applying bandpass filters to isolate signal components  [ 144 , 147 , 107 ] , conversion of phase data to frequency domain  [ 62 , 108 ] , conversion of RSS variance to frequency domain  [ 147 ] , inverse FFT on IF S21 or complex scattering coefficient signal to retrieve range profile  [ 74 , 142 ] , FFT on IF signal coming from CW radar to determine breath/heartbeat data  [ 82 , 109 ] , dimension reduction  [ 152 ] , filtered frequency index dominant frequency determination  [ 116 ] , FFT and inverse FFT for image reconstruction  [ 154 , 179 ] , and retrieving range profile from uplink pilot signal  [ 154 ] . The main difference between FFT and STFT is that STFT separates FFT operations in chunks across time. STFT has been used to retrieve phase information across time  [ 60 , 65 ] , for creation of time and frequency spectrograms  [ 64 , 113 , 116 ] , and to create range and micro-velocity or micro-Doppler spectrograms from a stack of range profiles  [ 75 , 78 , 92 , 93 , 129 ] or directly from a signal containing Doppler information  [ 102 , 119 ] . Micro-velocity spectrograms are used to analyze fine-grained velocity features. These features can be attributed to for example arm swinging while a person is walking or presence of something in context of strong background noise. These fine-grained features help to compute new types of information such as drone blade rotation  [ 116 ] or allow analytical models to better predict information of interest. Raja et al.  [ 64 ] denote that STFT is better suited for identification of movement direction, time of occurrence, and duration from a RSS signal.

 
 
 The beamformer is an estimation technique that can be used to calculate a one dimensional set of complex numbers associated to a user-defined AoA . It is used as an alternative to the AoA FFT . The sensing application pipelines either use the Minimum Variance Distortionless Response (MVDR) , i.e., the Capon beamformer  [ 50 , 55 , 96 , 98 , 131 ] or do not specify the beamformer type  [ 58 , 62 , 114 ] . The Capon beamformer computes the set with a steering vector and covariance matrix. The covariance matrix is a model describing spatial and frequency domain IF signal characteristics. The steering vector represents the phase rotation due to extra distance, which has to be traversed by a signal reflection at receiver channels in an array in phasor notation. Reflection path extraction is used to iteratively extract RSS signal components from the signal at the receiver associated to a path between reflector and receiver  [ 160 ] . Empirical Mode Decomposition (EMD) decomposes a signal into several Intrinsic Mode Functions (IMFs)   [ 79 ] . The output from both methods stays in the time domain. Matsuguma and Kajiwara  [ 73 ] retrieve a range profile based on a convolution in time between an IF pulse, i.e., single chirp, and impulse echo response. Oka et al.  [ 151 ] synchronize a Schottky diode output signal with an encoder distance signal to obtain an intensity image. Pawliczek et al.  [ 178 ] perform surface normal calculation with a phase data spectrogram to improve defect visualization. Peak detection is used to count breath/heartrate from a filtered time-series RSS signal  [ 144 , 147 ] or to determine if a Schottky diode output signal indicates presence of metallic or non-metallic objects. Reflection loss extraction isolates reflection loss from the total RSS loss with a set of equations  [ 144 , 147 ] . Häfner et al.  [ 118 ] measure azimuth angle and time of arrival by means of a maximum-likelihood based parameter estimator. Landmark extraction  [ 132 ] , based on a profile measured at a given azimuth angle, returns a set of landmarks, i.e., set of range and azimuth value tuples. Landmark extraction performs a set of filtering operations, after which magnitude values are scaled according to the probability that the magnitude value indicates a landmark. Continuous peaks at certain ranges indicate a landmark. Pulse integration uses the integration operation with the purpose of improving signal-to-noise ratio  [ 128 , 179 ] , help discover movement in a spectrogram  [ 106 ] , and to deduce a spectrogram representing micro-doppler signatures  [ 129 , 130 ] . Pulse integration types include incoherent integration  [ 106 ] , coherent integration  [ 128 ] , spectrogram integration across range  [ 129 , 130 ] , and a wideband signal filter operation  [ 179 ] . POSP is used in  [ 141 ] to calculate an integral. Further details, including the long version of POSP, are not present. Frequency compression is not explained in  [ 88 ] .

 
 
 TABLE VII: Summary of data reconstruction denoising pre-processing methods deployed in millimeter wave sensing pipelines. The denoising abbreviations are explained throughout Section  IV-B3 . 
 
 
 
   | 

 
 
 
 | 
 
 \Block 
 1-1 Reconstructon 
 | 
 \Block 1-20 Denoising | 
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

 
 | 
 
 
 Phase unwrapping 
 | 
 CFAR | 
 MTI | 
 
 
 Thresh- olding 
 | 
 LLSE | 
 
 
 Function fitting 
 | 
 
 
 Gaussian model 
 | 
 
 
 Manual background subtraction 
 | 
 
 
 Otsu algo- rithm 
 | 
 SVD | 
 
 
 Feed-forward NN 
 | 
 
 
 Band- pass filter 
 | 
 
 
 High- pass filter 
 | 
 
 
 Gaussian filter 
 | 
 
 
 Moving average filter 
 | 
 
 
 Offset version recombination 
 | 
 
 
 Doppler compen- sation 
 | 
 
 
 Deboun- cing 
 | 
 
 
 Mirror- ing 
 | 
 
 
 Distance correction 
 | 
 
 
 Window- ing 
 | 

 
 [ 43 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 45 ] | 
 | 
 X | 
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

 
 [ 50 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 53 ] | 
 | 
 X | 
 X | 
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

 
 [ 55 ] | 
 | 
 X | 
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

 
 [ 57 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 X | 
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

 
 [ 58 ] | 
 | 
 | 
 X | 
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

 
 [ 144 ] | 
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
 
 
 X 
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

 
 [ 62 ] | 
 
 
 X 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 

 
 [ 67 ] | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 70 ] | 
 
 
 X 
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
 | 
 | 

 
 [ 189 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 71 ] | 
 | 
 X | 
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

 
 [ 147 ] | 
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
 
 
 X 
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

 
 [ 78 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 81 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 

 
 [ 83 ] | 
 | 
 X | 
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

 
 [ 84 ] | 
 | 
 X | 
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

 
 [ 85 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 86 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 92 ] | 
 | 
 X | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 93 ] | 
 | 
 X | 
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

 
 [ 95 ] | 
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
 
 
 X 
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

 
 [ 96 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 98 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 102 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 142 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
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

 
 [ 178 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 106 ] | 
 
 
 X 
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
 
 
 X 
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

 
 [ 107 ] | 
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
 
 
 X 
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

 
 [ 108 ] | 
 
 
 X 
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
 
 
 X 
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

 
 [ 113 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 114 ] | 
 | 
 X | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 
 
 X 
 | 

 
 [ 115 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 116 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 121 ] | 
 | 
 X | 
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

 
 [ 154 ] | 
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
 
 
 X 
 | 
 | 
 | 

 
 [ 123 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 124 ] | 
 | 
 X | 
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

 
 [ 125 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 126 ] | 
 | 
 X | 
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

 
 [ 127 ] | 
 | 
 | 
 | 
 
 
 X 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 128 ] | 
 | 
 | 
 | 
 
 
 X 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 130 ] | 
 | 
 X | 
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

 
 [ 131 ] | 
 | 
 X | 
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

 
 [ 138 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 

 
 
 After using pulse compression on a given set of positions, while scanning an area by means of free space scanning with a moving device, a two dimensional structure of complex values, representing combined IF I/Q signals, also known as a SAR image is obtained. When using the magnitude of these complex values to create a grayscale image, the resulting raw SAR grayscale image is severely out of focus because it does not represent spatial information correctly yet. In most SAR pre-processing pipelines, range and azimuth reference functions are generated and convoluted with the SAR image in sequence to increase the focus. As a result, the SAR image correctly represents spatial information. These techniques are known as range  [ 161 ] and azimuth compression  [ 183 , 182 ] . More information can be found in  [ 183 , 182 ] . Maximally Stable Extremal Region (MSER) exploits the fact that reflections originating from the metal structures of vehicles, in contrast to reflections from the road surface, show up as bright stable area’s in a SAR grayscale image. Candidate regions are selected that stay below a grayscale area variation rate  [ 177 ] . Visual saliency detection, after a set of pre-processing steps performed on a SAR image, returns a binary image in which white area’s indicate presence of objects  [ 177 ] or parked vehicles  [ 176 ] .

 
 
 

#### IV-B 3 Data Reconstruction and Denoising

 
 Several sensing application pipelines deploy reconstruction and denoising techniques in the data domain. The techniques try to resolve data corruption, which refers to observing values in a data structure that are drastically different from the expected values in a certain context. In addition to filtering corruption causes already mentioned in Section  IV-B1 , data reconstruction and denoising also filter Doppler components caused by transmitter time multiplexing, remove redundant data parts, and filter data transients. Data transients are caused by persons that become stationary after walking into a room  [ 81 ] .

 
 
 Under a sampling and signal change assumption, one dimensional phase unwrapping is performed by  [ 57 ] . The phases of complex numbers in the slow time direction are analyzed from a two dimensional data structure prior to radial velocity FFT operations. A wrapped phase change greater than π \pi in a pair of consecutive complex numbers indicates that the phase of the second complex number should be corrected by adding or subtracting 2 ​ π 2\pi through means of phasor multiplication. One dimensional phase unwrapping can also be solved with a null range measurement obtained at the first chirp of the first frame. Subsequent chirp and frame processing is combined with multiplying the IF signal with the null range measurement  [ 70 ] . Two dimensional phase unwrapping through a path following algorithm is considered in the sensing application pipeline explained in  [ 178 ] .

 
 
 Constant False Alarm Rate (CFAR) is an adaptive thresholding technique that is used to extract, i.e., reduce a spectrogram containing magnitude values to, spectrogram parts, based on a sliding window, that indicate presence of targets against data corruption present in the spectrogram. Several CFAR types have been considered. One dimensional types include cell averaging  [ 93 , 114 , 121 , 123 , 124 ] , cell averaging smallest of  [ 55 , 71 , 96 , 114 ] , cell averaging greatest of  [ 71 ] , clutter map  [ 83 , 84 ] , and ordered statistics  [ 92 ] . A two dimensional custom CFAR type is considered in  [ 131 ] . The type indicates how the threshold is determined. More information on CFAR types can be found in  [ 163 , 206 , 207 ] .

 
 
 Background subtraction is performed by removing point cloud data in a frame with zero Doppler velocity  [ 43 , 115 ] and Cartesian coordinates that fall outside certain boundaries  [ 43 ] , removing measured profile and spectrogram (part) averages from profiles and spectrograms  [ 50 , 81 , 102 , 138 ] , deleting a profile belonging to an empty FOV from target measurement profiles  [ 86 ] , and statically removing the 0 Hz component from maximum Doppler frequency data across time  [ 116 ] .

 
 
 Moving Target Indication (MTI) is not explained in  [ 53 , 58 ] . The papers suggest that it is a more general term used to indicate that denoising is performed. Static thresholding is used to extract spectrogram parts from which phase variation over time  [ 62 ] or magnitude  [ 127 ] exceeds a threshold or create binary radar images  [ 67 , 142 ] (unknown purpose  [ 67 ] or to make material faults visible  [ 142 ] ). Linear Least Squares Estimation (LLSE) is used to estimate a shift caused by DC offset in complex numbers’ imaginary and real parts of a 2D data structure, prior to radial velocity FFT (used to estimate chest vibration). The shift is used afterwards to adjust the complex number’s imaginary and real parts  [ 57 ] . 2D quadratic function fitting is used  [ 178 ] to fit a function to phase data, which is later subtracted from the phase data to eliminate phase curvature. A per-pixel Gaussian model is used to subtract noise coming from unwanted background reflections from spectrograms  [ 125 ] . The Otsu algorithm is an automatic threshold selection method supported by image segmentation. It is used for noise reduction in radar images  [ 189 ] . Noise reduction with help from Singular Value Decomposition (SVD) can be applied to extract desired spectrogram parts indicating presence of target reflection  [ 142 ] . A feed-forward neural network can be used to filter ghost targets (i.e., superimposed noise from unwanted background and target reflections) in automotive radar sensing  [ 95 ] . The bandpass and highpass filters reported in this section are applied in the frequency domain. Profiles are multiplied with a transfer function that represents the bandpass or highpass filter’s frequency response  [ 208 ] . A 2D Gaussian filter is convolved over a spectrogram for extra noise reduction in addition to background subtraction  [ 123 ] . Moving average filtering is performed in time by first subtracting an empty background spectrogram from a measurement spectrogram. Afterwards, the background spectrogram is updated with the measurement spectrogram  [ 127 ] . In offset version recombination, a noisy set of complex numbers created with STFT is offset with a simple addition of a frequency dependent anti-symmetric function  [ 113 ] . Spectrograms originating from real and virtual receiver channels cannot be retrieved simultaneously at real receivers. Radar collection systems employ multiplexing strategies to retrieve the spectrograms  [ 172 ] . When time multiplexing is used, spectrograms from real and virtual receivers will experience an unwanted Doppler induced phase shift. This shift is compensated for by means of phasor multiplication  [ 205 ] . To mitigate analytical model performance degradation caused by data transients, a debouncing logic can be implemented  [ 81 ] . Radar images acquired through imaging do not properly represent the geometry of an environment due to multipath propagation. Image correction can be applied by means of mirroring techniques  [ 154 ] . Distance correction  [ 102 ] scales correlation profiles while taking into consideration that sample strength falls off according to a certain pattern. To remove motion corrupted segments from heartbeat data, the heartbeat data is segmented and segments are removed based on the outcome of a thresholding procedure  [ 62 ] . Two dimensional windowing prior to performing a FFT operation to determine velocity  [ 114 ] can be considered a brick-wall filter  [ 64 ] .

 
 
 

#### IV-B 4 Data Transformation

 
 Data transformation refers to using mathematical operations on data structures that either cause them to change into new, higher-level data types or cause data structure aspects, e.g., value range or axis range considered, to change. There is a wide variety in goals that sensing application pipelines try to achieve through use of data transformation. For example, value range changes can be used to unify the value range of different variables. This will omit bias towards variables that have a bigger value range compared to other variables during analytical model training  [ 209 ] . Data type changes allow higher-level information to be extracted from lower-level information. For example, position information in Cartesian coordinates can be extracted from lower-level range and AoA information  [ 138 ] , voxels created from position information encapsulate higher-level body shape information  [ 46 ] , etc.

 
 
 TABLE VIII: Summary of data transformation pre-processing methods deployed in millimeter wave sensing pipelines. The abbreviations are explained throughout Section  IV-B4 . 
 
 
 
   | 

 
 | 
 
 \Block 
 1-15 Artifact (type) change 
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
 
 \Block 
 1-8 No Artifact (type) change 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
   | 

 
 | 
 
 
 Peak detection 
 | 
 
 
 Auto- correl- ation 
 | 
 
 
 Thresh- olding 
 | 
 
 
 Spectral analysis 
 | 
 
 
 Dimension reduction 
 | 
 
 
 Voxeli- zation 
 | 
 
 
 Binari- zation 
 | 
 
 
 Math. morph- ology 
 | 
 LRMF | 
 
 
 Edge detection 
 | 
 
 
 Coord- inate trans- form 
 | 
 
 
 Point trans- form 
 | 
 FWT | 
 
 
 Grey- scaling 
 | 
 
 
 Ray casting 
 | 
 
 
 Normali- zation 
 | 
 
 
 ROI extraction 
 | 
 
 
 Edge region discovery 
 | 
 
 
 Windowing/ Segmentation 
 | 
 MRC | 
 
 
 Manual smoothing 
 | 
 
 
 Conv- olution mask 
 | 
 
 
 Down- sampling 
 | 

 
 [ 158 ] | 
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
 | 
 | 
 | 
 
 
 X 
 | 

 
 [ 141 ] | 
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
 
 
 X 
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

 
 [ 43 ] | 
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
 
 
 X 
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
 
 
 X 
 | 
 | 
 | 

 
 [ 44 ] | 
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
 
 
 X 
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

 
 [ 45 ] | 
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
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 46 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 47 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 53 ] | 
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
 
 
 X 
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

 
 [ 55 ] | 
 
 
 X 
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
 
 
 X 
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

 
 [ 57 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 58 ] | 
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
 | 
 | 
 | 
 | 

 
 [ 62 ] | 
 
 
 X 
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
 | 
 | 
 | 
 | 

 
 [ 193 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 189 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 X | 
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

 
 [ 71 ] | 
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
 
 
 X 
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

 
 [ 176 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 77 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 81 ] | 
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
 | 
 | 
 | 
 | 

 
 [ 161 ] | 
 | 
 | 
 
 
 X 
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
 | 
 | 

 
 [ 88 ] | 
 | 
 | 
 
 
 X 
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
 | 
 | 

 
 [ 92 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 93 ] | 
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
 
 
 X 
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

 
 [ 96 ] | 
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
 
 
 X 
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

 
 [ 100 ] | 
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
 | 
 | 
 | 
 | 

 
 [ 103 ] | 
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
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 142 ] | 
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
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 106 ] | 
 | 
 | 
 | 
 
 
 X 
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
 | 

 
 [ 107 ] | 
 
 
 X 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 108 ] | 
 
 
 X 
 | 
 
 
 X 
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
 | 
 | 
 | 

 
 [ 152 ] | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 114 ] | 
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
 
 
 X 
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

 
 [ 116 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 120 ] | 
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
 
 
 X 
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

 
 [ 121 ] | 
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
 
 
 X 
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

 
 [ 154 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 123 ] | 
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
 
 
 X 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 125 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 126 ] | 
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
 X | 
 | 
 | 
 | 

 
 [ 128 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 138 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 
 Peak detection is used to retrieve velocity information from a range, azimuth, and velocity cube  [ 55 ] , retrieve breath/heartbeat data by means of average inter-peak distance in a phase data sequence in time  [ 62 ] , and determine magnitude peak position in a range profile  [ 107 ] . Thresholding is either applied on a range-compressed down-conversion IF signal to determine vertical range between the collection system and the ground  [ 161 ] or used to determine if space debris is detected or not  [ 88 ] . Spectral analysis is a general term used to describe the fact that IMFs are converted
to a frequency domain representation to determine if they can be attributed to heartbeat or breathing data based on a given frequency range  [ 106 ] . Dimension reduction is performed by transforming a two dimensional structure of complex values into a structure containing magnitude values and determining the position that corresponds to the dominant reflection of a lap joint. Data evaluation is performed at the position that yields a one dimensional structure  [ 152 ] . Voxelization can be thought of as creating a three dimensional grid. The grid consists of voxels, i.e., custom values (or sets of custom values). Values found in the sensing application pipelines include number of points present  [ 47 ] and accumulated velocity information of every point  [ 138 ] in bounded coordinate regions of a point cloud frame. Voxelization is used to turn a variable number of point information rows  [ 138 ] , associated to a point cloud frame, into a data structure with fixed dimensions  [ 47 , 138 ] . Binarization refers to the creation of a binary image from a radar image. Binary images contain black and white values assigned through use of a threshold  [ 193 ] . Mathematical morphology is a term used to describe a set of matrix operations that are used to transform binary images  [ 193 , 176 ] . These matrix operations are called erosion, dilation, opening and closing, and involve an input image and a structuring element. The structuring element is similar to a kernel in the context of two dimensional convolution. Opening has been used to transform a binary saliency map into connected regions  [ 176 ] . Dilation has been used to improve image visualization  [ 189 ] . Low Rank Matrix Factorization (LRMF) is used to factorize a radar image into two: a foreground, i.e., concealed object, and background to detect concealed objects. Factorization is performed with a patch-based, i.e., segment-based, Gaussian mixture model  [ 189 ] . Laplacian of Gaussian and Canny edge detection are used on a gray scale converted spectrogram containing magnitude values to discover Doppler frequency information  [ 116 ] . Once range and AoA values retrieved from spectrograms or cubes are known, the values can be transformed from range and AoA information, through polar or spherical to two or three dimensional Cartesian coordinate conversion equations, to position information in Cartesian coordinates  [ 43 , 44 , 46 , 53 , 55 , 71 , 93 , 96 , 103 , 114 , 121 , 138 ] . The new data structure is referred to as either a point cloud frame or point scan. The data structures differ in context in which measurements were conducted. Stationary radar collection systems are used to retrieve point cloud frames while moving radar collection systems on a robot or vehicle are used to retrieve point scans. The active imaging application pipeline presented in  [ 141 ] transforms coordinates by means of integration. Point transformation is concerned with converting a point cloud frame based on fixed coordinate ranges into a radar image. To every pixel, a RGB value is assigned. The R value corresponds to a x-coordinate, the G value to a y/z-coordinate (separate xy/xz images), and B value to a magnitude value. Coordinates that contain no points are assigned a black RGB value  [ 103 ] . The application pipeline in  [ 120 ] uses a similar approach in which points converted to camera coordinates are used to compute RGB values. Fast Wavelet Transform (FWT) is used to transform a radar image into the wavelet domain with the intent to denoise it in the wavelet domain  [ 193 ] . Greyscaling refers to converting a spectrogram consisting of magnitude values to a grayscale image  [ 116 ] . Once uplink pilot signal AoA and time of arrival, and a radar image consisting of reflective shapes are known, casting a ray, i.e., millimeter wave, at the estimated AoA from a base station will eventually lead to a mobile user’s range with respect to the base station under the assumption of reflection at a specular angle  [ 154 ] with a computation on time of arrival.

 
 
 Normalization refers to mapping real value ranges of several data structures. Spectrograms with magnitude values are normalized to unit scale  [ 45 ] , between 0 and 1 and centered around the mean value  [ 92 ] , by dividing every magnitude value with the sum of all magnitude values in the spectrogram  [ 123 ] or by scaling magnitudes logarithmically and performing a max-min truncation  [ 125 ] . RGB values assigned to a radar image with help from a point cloud frame with fixed coordinate ranges are based on normalized range, AoA , and magnitude information  [ 103 ] . Point cloud frame coordinates are max-min normalized based on fixed coordinate ranges  [ 138 ] . Region Of Interest (ROI) selection significantly reduces search dimensionality over a spectrogram
during feature extraction computations  [ 45 ] . To isolate periodic chest movement, among all spectrogram columns corresponding to range in an acceptable FOV , the range column with the maximum average magnitude value is selected  [ 57 ] . With background information on test person distance, a frequency domain ROI can be selected  [ 107 ] . After thresholding, the closest detected range profile part in range is selected for each radar in case of multiple detected target peaks  [ 128 ] . Through use of a pixel condition argument, person edges are discovered in image segments if a pixel value in the condition argument is greater than a certain threshold. The edge values are set to zero afterwards  [ 193 ] . Windows containing a time dependent sequence of voxel grids are created and used to form a dataset  [ 46 , 47 ] . An image segmentation procedure is presented in  [ 193 ] . Dynamic sized temporal input data can be formatted by using the Markov Frame method  [ 77 ] . Windowing can also be employed to track target position across spectrograms  [ 123 ] . Maximal Ration Combining (MRC) refers to the creation of mean range and velocity and range and azimuth spectrograms from a range, velocity, and azimuth cube containing magnitude values by computing a weighted mean  [ 126 ] . Discontinuous range, azimuth sine, and velocity sequence values are smoothed. At sequence window ends, a discontinuous point is set to the value of its nearest neighbor. When eligible values are at both sides of the discontinuous point, the discontinuous point becomes the average of these values  [ 43 ] . Frequency index smoothing is used in the pipeline explained in  [ 116 ] . A convolution mask can be applied to an active imaging spectrogram to make a concealed tile image and the crack within it more visible  [ 142 ] . Downsampling is used to convert a CSI frequency profile into a smaller feature vector  [ 158 ] .

 
 
 

#### IV-B 5 Dataset Creation and Augmentation

 
 After datasets, consisting of data samples, have been generated by both the signal and data transformation processes, several sensing application pipelines augment these datasets with artificially created data samples. In case of profiles, spectrograms, and cubes, it is assumed that real value structures consisting of magnitude values are generated. Reasons for data augmentation include improving analytical modeling performance on unseen data during inference  [ 63 , 127 , 128 , 138 ] and to increase the dataset size to a size required during model training to achieve good analytical model performance  [ 194 , 138 ] without the need for extra sampling. For each original spectrogram in the dataset or for an average spectrogram per class, artificial spectrograms can be generated by sampling pixel values according to a normal distribution  [ 63 , 127 ] . Radar images can be fused with image segments at a random location  [ 194 ] . Radar images and point cloud frames have also been rotated, scaled, and content inside the images and point cloud frames has been translated  [ 128 , 138 ] . To prepare datasets for analytical model training in several sensing application pipelines, data samples are labeled with class labels and part of the dataset is designated for model training while other parts are designated for model validation and testing.

 
 
 TABLE IX: Summary of dataset augmentation creation pre-processing methods deployed in millimeter wave sensing pipelines. 
 
 
 
   | 

 
 | 
 
 \Block 
 1-3 Augmentation 
 | 
 | 
 | 
 \Block 1-2 Creation | 
 | 

 
 | 
 
 
 Normal distribution 
 | 
 
 
 Object at random location fusion 
 | 
 
 
 Rotation, scaling, skew, and translation 
 | 
 Labeling | 
 Splitting | 

 
 [ 47 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 50 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 53 ] | 
 | 
 | 
 | 
 X | 
 | 

 
 [ 63 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 X | 

 
 [ 64 ] | 
 | 
 | 
 | 
 X | 
 X | 

 
 [ 181 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 67 ] | 
 | 
 | 
 | 
 X | 
 | 

 
 [ 194 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 X | 

 
 [ 195 ] | 
 | 
 | 
 | 
 X | 
 X | 

 
 [ 78 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 79 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 84 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 92 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 149 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 93 ] | 
 | 
 | 
 | 
 X | 
 X | 

 
 [ 103 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 123 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 125 ] | 
 | 
 | 
 | 
 X | 
 X | 

 
 [ 126 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 127 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 X | 

 
 [ 128 ] | 
 | 
 | 
 
 
 X 
 | 
 X | 
 X | 

 
 [ 129 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 130 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 137 ] | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 138 ] | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 139 ] | 
 | 
 | 
 | 
 X | 
 X | 

 
   | 

 

 
 
 

#### IV-B 6 Gaps and Challenges

 
 During the analysis of pre-processing methods, it was discovered that there is a lot of variety in the definition of what is considered to be pre-processing and way that pre-processing pipelines are constructed. We consider signal processing methods and typical data science pre-processing methods to be pre-processing methods. Other papers consider signal processing methods to be part of the signal collection system and only consider typical data science pre-processing methods to be pre-processing methods. Sensing application pipelines perform the processes mentioned in Figure  6 in different orders, skip processes, revert back to pre-processing after feature extraction or analytical modeling, etc.

 
 
 When looking at Tables  V ,  VI ,  VII ,  VIII , and  IX , there are a few observations that raise interest. Many of the table columns contain a few crosses while a large portion of the crosses is located in just a few columns. There are two explanations for this. Many sensing application pipelines rely on pre-processing methods that are well known. In signal transformation, many sensing application pipelines rely on FFT and STFT for signal conversion. However, estimation techniques such as Capon beamforming, Bartlett beamforming, MUltiple SIgnal Classification (MUSIC) , Sparse Asymptotic Minimum Variance (SAMV) beamforming, Maximum Likelihood Estimation (MLE) , etc. exist that can be used in the pre-processing pipeline too. In  [ 72 , 93 ] , test results have been reported for the MUSIC algorithm. Many of the pre-processing methods are sensing application pipeline specific. For example, landmark extraction  [ 132 ] works with a specific radar collection system that collects a signal at specific azimuth angles, Doppler compensation  [ 114 ] is only used when transmit time multiplexing is used for measuring AoA information, etc.

 
 
 Another observation is that many table rows contain multiple crosses. This indicates use of multiple reconstruction, denoising, and transformation pre-processing methods. In  [ 81 ] , multiple pre-processing pipeline paths are used in parallel to obtain different artifacts and artifact types that all use their own denoising and transformation methods. In  [ 62 ] , there are two consecutive pre-processing stages. In the first stage, phase data is obtained in a pre-processing pipeline where CFAR is used to denoise spectrogram data. In the second stage, phase data is bandpass filtered before being used to extract breath/heartbeat data. Pre-processing methods are also used consecutively. In  [ 159 ] , it is explained that signal reconstruction is specifically used to omit limitations encountered with denoising of the received signal phase through DDBR .

 
 
 
 

### IV-C Feature Extraction 

 
 The end result of the pre-processing phase in the application pipeline is a set of raw data samples in the form of, for example, sequence windows  [ 158 , 137 , 139 ] , (voxelized) point cloud frames  [ 46 , 47 , 138 ] , spectrograms  [ 48 , 76 , 93 ] , radar images  [ 174 , 103 , 120 ] , etc. This raw form of data, however, is not always used directly by analytical models. Degradation in modeling performance is sometimes caused by noise and redundancy in raw data samples. Extraction of relevant and informative data features from raw data samples is performed to compensate for this  [ 210 ] . Reducing the raw dataset to a limited number of features makes it easier to visualize for better understanding and gaining knowledge about the process that led to the generated raw data samples  [ 211 ] .

 
 
 The feature extraction methods used by millimeter wave sensing applications take in the raw dataset and map the entire dataset to a new feature space  [ 210 ] . The feature extraction phase is, however, not a required step. Recent deep learning techniques  [ 47 , 194 , 149 , 103 , 104 , 123 , 124 ] and several modeling algorithms  [ 159 , 184 , 56 ] perform well on raw data samples and, therefore, do not require the feature extraction phase. The border between feature extraction and analytical modeling has become blurry in recent papers on millimeter wave sensing applications. Several deep learning models contain layers designated for feature extraction  [ 158 , 46 , 47 , 16 , 92 , 125 , 126 , 130 , 138 ] . Certain positioning and environment mapping algorithms resemble feature extraction methodologies  [ 160 , 184 ] . Transfer learning  [ 181 , 195 , 120 ] is not included in this section since the methodologies transfer analytical model parameters to another analytical model rather than extracted data features. Feature analysis  [ 43 , 77 , 91 , 129 ] was only addressed by a small minority of papers and is therefore excluded from this paper as well.

 
 
 Table  X presents our analysis of feature extraction approaches used in millimeter wave sensing applications. In what follows, we explain main feature extraction methods used in millimeter wave sensing applications.

 
 
 TABLE X: Summary of feature extraction methods deployed in millimeter wave sensing pipelines. The automatic mapping abbreviations are explained throughout Section  IV-C . 
 
 
 
   | 

 
 | 
 
 \Block 
 1-2 Manual mapping 
 | 
 | 
 
 \Block 
 1-6 Automatic mapping 
 | 
 | 
 | 
 | 
 | 
 | 

 
 | 
 
 
 Time dependent 
 | 
 
 
 Frequency dependent 
 | 
 
 
 Clustering 
 | 
 
 
 A-priori estimation 
 | 
 
 
 Meta learning 
 | 
 t-SNE | 
 
 
 Representation learning 
 | 
 PCA | 

 
 [ 42 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 43 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 159 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 45 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 46 ] | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 47 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 143 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 49 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 50 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 174 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 55 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 57 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 145 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 60 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 146 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 62 ] | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 64 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 189 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 72 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 73 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 147 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 176 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 76 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 78 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 79 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 81 ] | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 161 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 83 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 84 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 91 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 93 ] | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 94 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 96 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 98 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 114 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 

 
 [ 116 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 127 ] | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 X | 
 | 
 | 

 
 [ 128 ] | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 129 ] | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 137 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 139 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 

#### IV-C 1 Manual Feature Mapping

 
 Manual feature mapping relies heavily on experience and knowledge of domain experts to extract features  [ 211 ] . The main feature categories found in the pipelines include statistics, calculus, geometry, vision, and gesture recognition. Statistical features include (weighted) mean  [ 45 , 143 , 49 , 64 , 72 , 147 , 91 , 98 , 137 , 139 ] , median  [ 129 ] , total value  [ 42 , 45 , 49 , 91 ] , standard deviation and variance  [ 159 , 45 , 143 , 147 ] , root mean square  [ 45 , 49 , 137 , 139 ] , moving average  [ 73 , 98 ] , sequence window centroid  [ 45 , 161 , 91 ] , range  [ 43 , 157 , 147 , 129 ] , quartiles  [ 147 ] , higher order cumulants  [ 83 ] , coefficients  [ 98 , 129 ] , value normalization  [ 43 , 84 , 129 ] , value distribution  [ 42 , 49 , 189 , 79 , 91 ] , probability  [ 161 ] , and histogram features  [ 189 ] . Calculus features include maximum  [ 43 , 49 , 57 , 145 , 60 , 64 , 72 , 147 , 84 , 116 , 137 , 139 ] , minimum  [ 49 , 64 , 128 , 137 , 139 ] , absolute value  [ 49 , 73 , 137 , 139 ] , a signal pattern across time  [ 159 ] , discrete differentiation  [ 42 , 45 , 49 , 73 , 91 ] , discrete integration  [ 45 ] , derivative sum  [ 45 ] , integrated sum  [ 45 ] and integrated delta  [ 45 ] . Geometrical features include intensity area’s  [ 72 ] , roundness  [ 72 ] , eccentricity  [ 72 ] , perimeter  [ 72 ] and shape slope  [ 72 ] . A specific visual feature used is the so called local binary pattern  [ 189 ] . The authors in  [ 78 ] use a feature extraction algorithm comprised of steps that involve some of the features mentioned above.

 
 
 In 2016, Lien et al.  [ 45 ] conducted extensive research into the development of range-doppler spectrogram specific features in the context of gesture recognition. The goal was to reduce computational overhead on resource constrained devices. After ROIs have been selected, matrix calculations are applied to these regions to obtain multi-channel integration, multi-channel derivative and temporal derivative matrices. In 2018, Flintoff et al.  [ 91 ] introduced a new gesture recognition feature in the form of a sonar value.

 
 
 

#### IV-C 2 Automatic Feature Mapping

 
 Automatic feature mapping methods used in the millimeter wave sensing applications mainly aim at dimensionality reduction. Several tracking applications that utilize point cloud frames either use clustering  [ 46 , 93 ] or the Kalman filter’s prediction process  [ 55 , 94 , 96 , 114 ] to reduce point clouds into point cloud cluster centroid values. Since the number of point clouds per frame is unknown, a clustering algorithm that does not require a number of clusters to be defined a priori is required  [ 46 ] . More information about the prediction process can be found in  [ 212 ] . Clustering is also used to reduce people counts corresponding to the same person  [ 81 ] or to obtain a median heart rate measurement from a noisy spectrum  [ 62 ] . Principal Component Analysis (PCA) is used for linearly reducing an input data vector v v into principle component vector p = W T ​ v p=W^{T}v containing decorrelated principal components suitable for use with Support Vector Machines (SVMs) and decision trees. The symbol W ∈ ℝ ( N , L ) W\in\mathbb{R}^{(N,L)} denotes a projection matrix where N N is the input vector size and L L the number of desired principal features. The projection matrix elements are calculated such that the variance included in input data vectors used for training is maximized  [ 213 ] . To reduce a dataset to a low feature dimension for visualization purposes, t-distributed Stochastic Neighbor Embedding (t-SNE) can be used  [ 127 ] .

 
 
 Other automatic feature mapping methods include representation and meta learning  [ 213 , 127 ] . Even though these methods also involve dimensionality reduction, it is considered to be a positive side effect rather than a goal. The main idea behind representation learning is to create a machine learning model that has the ability to extract features from a dataset. The features can be fed into a variety of downstream models used for solving a task that is loosely related to the task associated to the upstream model  [ 213 ] . Meta learning differs from representation learning because it has the goal of providing features that can be used for a variety of different task types rather than loosely related tasks  [ 127 ] .

 
 
 

#### IV-C 3 Gaps and Challenges

 
 Table  X shows that the majority of approaches that utilize feature extraction rely on manual feature mapping. In contrast to reducing computational overhead, manual feature mapping costs a lot of time and human resources. Automatic feature mapping in millimeter wave sensing applications is an unexplored field of research. Only one methodology found  [ 127 ] uses a feature extraction technique for feature visualization to gain an understanding of extracted features. Lastly, meta and representation learning are severely underexploited in millimeter wave pipelines.

 
 
 
 

### IV-D Analytical Modeling 

 
 Most millimeter wave sensing applications presented in Figure  2 cannot reach their goal by simply executing a set of pre-processing steps and extracting information relevant to the application. They require a model to translate information from pre-processed datasets or extracted feature sets to the application goal. Examples of these models include classification models  [ 46 , 47 , 79 ] , filter models  [ 46 , 89 , 185 ] , and measurement models  [ 159 , 184 , 186 , 133 ] . Classification models assign class labels to a set of input data. The class can indicate performed activities, gestures or events/failures, presence of a specific person/object, etc. Filter models estimate trajectories of tracked entities, filter out false positive class predictions, etc. in an environment where noise causes trajectory measurements and class predictions to be stochastic in nature. Measurement models calculate the value of information such as position, yaw rate, or absolute velocity using mathematical models. In some cases, measurement models use a cost minimization technique for value calculation. The costs are computed with a function that uses input data as a function parameter. We classified the models used in millimeter wave sensing applications based on the model class (data driven vs. model driven) and the model type (black, grey, white models). Black, grey and white model types do not refer to the degree of model explainability. The model types refer to model determinism and use of physical knowledge and/or data for model construction. The model types were adopted from a physiological model identification study presented by Duun-Henriksen et al.  [ 214 ] . Apart from occasionally explaining that a certain model fuses millimeter wave data with data originating from other sensing domains in Section  IV-D3 to avoid hybrid model explanation ambiguity, further discussion of models that fuse millimeter wave data with data originating from other sensing domains is outside the scope of this review. In what follows, we analyze the analytical model classes and model types used in millimeter wave sensing applications.

 
 

#### IV-D 1 Model driven modeling

 
 Model driven modeling relies on physical and/or mathematical knowledge about the technology or the environment to construct a model. The knowledge refers to information, a set of rules and/or a set of equations obtained from physical phenomena or mathematical proofs. The model driven approaches are classified either as white box or grey box models. White box models vary in determinism and solely rely on physical and/or mathematical knowledge for model construction. For example, phase tracking  [ 159 ] deploys position initialization, resulting in an extra dependency on a random variable in addition to phase inputs for a certain position output. In contrary, instantaneous ego-motion estimation  [ 133 ] relies on Monte-Carlo simulation to test a model. This indicates that the computations performed by the model are deterministic in principle. Grey box models are non-deterministic and also rely on physical and/or mathematical knowledge for model construction. They differ from white box models because parts of the model are continuously altered across time or completed with information extracted from input data and data features  [ 214 ] . Examples include hidden state updates and use of cost and similarity parameters.

 
 
 Table  XI presents our analysis of model driven approaches used in millimeter wave sensing application pipelines. Next, we explain different approaches used under white and grey box models.

 
 
 TABLE XI: Summary of model driven analytical models used in millimeter wave sensing pipelines. The abbreviations are explained throughout Section  IV-D . 
 
 
 
   | 

 
 
 
 | 
 
 \Block 
 1-5 White box 
 | 
 | 
 | 
 | 
 | 
 
 \Block 
 1-10 Grey box 
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
 
 
 Phase tracking 
 | 
 SVD | 
 
 
 Velocity/Acker- man model 
 | 
 
 
 Angle/Time model 
 | 
 DTDOA | 
 
 
 Shadow RTI 
 | 
 
 
 Bayesian filter 
 | 
 
 
 Association/ Allocation 
 | 
 Scan matching | 
 MLE | 
 Triangulation | 
 
 
 Angle/Time model 
 | 
 RSA model | 
 Rician model | 
 Velocity model | 

 
 [ 44 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 159 ] | 
 
 
 X 
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

 
 [ 46 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 51 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 52 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 55 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 184 ] | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 58 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 145 ] | 
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
 X | 
 
 
 X 
 | 
 | 
 | 
 | 

 
 [ 61 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 146 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 190 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 161 ] | 
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
 X | 

 
 [ 87 ] | 
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
 X | 

 
 [ 89 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 90 ] | 
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
 X | 
 | 

 
 [ 94 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 162 ] | 
 | 
 | 
 | 
 | 
 X | 
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

 
 [ 96 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 150 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
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

 
 [ 152 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 

 
 [ 185 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 114 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 115 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 153 ] | 
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
 X | 
 | 
 | 

 
 [ 132 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 133 ] | 
 | 
 | 
 
 
 X 
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

 
 [ 134 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 

 
 [ 135 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 
 
   | 

 

 
 
 White Box Models 

 
 The phase tracking model  [ 159 ] calculates a two dimensional position in Cartesian coordinates based on successive phase shifts. The two dimensional position is calculated with a combination of distance equation sets and triangulation. The initial position is measured with a special acquisition module that is independent of the input data. The velocity and the Ackerman model  [ 133 ] first calculates a radar velocity vector containing longitudinal and lateral velocities based on the sinusoidal progress of measured radial velocities over the azimuth angle using a least-square approach. Afterwards, object’s absolute velocity and yaw rate are calculated using a model based on the Ackerman condition involving a velocity and yaw rate equation. The angle/time model  [ 184 ] calculates mobile object positions based on a configuration involving multi-path component Time of Flight (ToF) and AoA information or a configuration involving multi-path component ToF and Angle of Departure (AoD) information. The information is retrieved from a ray tracing tool. This information, combined with meta-data such as a map layout, results in object locations. The eventual location is determined based on majority voting. Relative motion between two radar scans can be measured in a least-square sense by using SVD on a set of landmark matches  [ 132 ] . Double Time Difference Of Arrival (DTDOA) is a positioning model consisting of a set of equations that use pseudo time of arrival to compute a position estimate  [ 162 ] .

 
 
 
 Grey Box Models 

 
 Bayesian filters originate from stochastic filtering theory and Bayesian statistics. As indicated in Section  IV-B , millimeter wave sensing applications obtain measurements as discrete samples. Therefore, the explanation assumes a discrete Bayesian filter modeled as a Hidden Markov Model (HMM) . The model contains one hidden state that varies with time. The state describes information that is interesting for the millimeter wave sensing application. The exact information is unobservable and thus hidden. Furthermore, action variables are omitted. The hidden state x t x_{t} is dependent on one previous state x t − 1 x_{t-1} . Measurements s t s_{t} in the form of raw data samples or data features are observed at, i.e., sampled from, the pre-processing or feature extraction phase. These measurements are dependent on the current hidden state. The state transition function includes noise to account for a distribution of different outputs the function can yield at a given time instant. The measurement observation function accounts for noise to take care of measurement inaccuracies  [ 215 , 216 ] .

 
 
 The state transition and the state observation functions x t = g ⁡ ( x t − 1 , p ​ n t ) x_{t}=g(x_{t-1},pn_{t}) and s t = h ⁡ ( x t , m ​ n t ) s_{t}=h(x_{t},mn_{t}) are modeled as Probability Density Functions (PDFs) p ⁡ ( x t | x t − 1 ) p(x_{t}|x_{t-1}) and p ⁡ ( s t | x t ) p(s_{t}|x_{t})   [ 216 ] . PDFs explicitly represent the uncertainty in variables taking on a particular value in a range of values at a specific time instant. The Bayesian filter continuously executes two functions called the predict and update functions with a recursive algorithm. The role of the predict function is to estimate a new hidden state PDF p ( x t | s 1 : t − 1 ) = ∑ x t − 1 p ( x t | x t − 1 ) p ( x t − 1 | s 1 : t − 1 ) p(x_{t}|s_{1:t-1})=\sum_{x_{t-1}}p(x_{t}|x_{t-1})p(x_{t-1}|s_{1:t-1}) based on the previous hidden state PDF and state transition PDF . The new hidden state PDF can be used to derive hidden state values  [ 217 ] . The role of the update function is to adjust the hidden state PDF when measurements are observed. The update function is based on Bayes’ theorem. The update function ensures that the hidden state PDF can always be used to derive hidden state values that closely represent, and do not drift away from, the exact hidden state values. Calculation and/or modeling of p ( x t | s 1 : t − 1 ) p(x_{t}|s_{1:t-1}) , p ⁡ ( s t | x t ) p(s_{t}|x_{t}) , and p ⁡ ( x t | x t − 1 ) p(x_{t}|x_{t-1}) is a core task of Bayesian filtering. More information can be found in a comprehensive review authored by Chen  [ 215 ] .

 
 
 Several Bayesian filters exist, such as the custom Bayesian  [ 45 , 185 ] , particle  [ 44 , 136 ] , α − β \alpha-\beta   [ 128 ] , Kalman  [ 46 , 53 , 58 ] , extended Kalman  [ 55 , 94 , 96 , 114 , 115 , 186 , 135 ] , fusion extended Kalman  [ 146 , 95 ] , unscented Kalman  [ 89 ] , fusion adaptive Kalman  [ 51 ] and adaptive Sage-Husa Kalman  [ 52 , 61 ] filter. The Kalman filter  [ 215 ] , under linear, quadratic and Gaussian assumptions, can represent the state transition and observation functions x t = g ⁡ ( x t − 1 , p ​ n t ) x_{t}=g(x_{t-1},pn_{t}) and s t = h ⁡ ( x t , m ​ n t ) s_{t}=h(x_{t},mn_{t}) as a set of linear equations. All other Kalman filters loosen these assumptions. The extended Kalman filter, for example, allows the functions to be approximated as a Jacobian matrix  [ 215 ] . Particle filters  [ 216 , 215 ] approximate p ( x t | s 1 : t − 1 ) p(x_{t}|s_{1:t-1}) as a set of particles { x t ( i ) , w t ( i ) } i = 1 l \{ x_{t}^{(i)},w_{t}^{(i)} \}_{i=1}^{l} where w t ( i ) = p ⁡ ( s t | x t ( i ) ) w_{t}^{(i)}=p(s_{t}|x_{t}^{(i)}) . The α − β \alpha-\beta filter  [ 128 ] is a simplified Bayesian filter that limits the number of hidden states to two. Predictions and updates are executed with a simple set of equations. Symbols α \alpha and β \beta refer to manually set correction gains used during the update process.

 
 
 Scan matching models are concerned with finding the best rotation and translation operations that have the ability to align two point scans or a point scan to an existing area map. The Iterative Closest Point (ICP) model involves two iterative steps until convergence. In the first step, points from two are matched based on closeness to one another in a given space. Closeness is measured by means of a distance metric. In the second step, to find the optimal rotational angle and translation, the sum of squared distances between the points is used  [ 135 ] . Normal Distribution Transform (NDT) scan matching  [ 89 , 134 ] requires a grid of probability functions created from a map for matching.

 
 
 Association/allocation models either use combinatorial optimization  [ 46 , 185 ] , a gating function  [ 55 , 94 , 96 , 114 ] , template model matching  [ 190 ] or landmark association  [ 89 , 132 ] . These models associate input data to an existing entity or create new entities based on an optimization criterium. This criterium is dependent on a cost metric involving the input data. For example in tracking, the models determine if incoming input data belong to a certain track that is already known or indicate a new track.

 
 
 Other grey box models include Shadow RTI  [ 150 ] , triangulation, angle/time model  [ 145 ] , RSS Series Analysis (RSA) model  [ 153 ] , velocity model  [ 161 ] , Rician model  [ 90 ] , and MLE   [ 152 , 134 ] . Shadow RTI  [ 150 ] measures link RSS attenuation changes across a network of millimeter wave transceivers. The knowledge utilized for model construction is that when people walk by, the RSS attenuation changes. Every link is made of a set of ‘pixels’. Link RSS attenuation contribution is computed for every pixel by solving a least square problem based on a measured RSS change vector. The triangulation and angle/time models in  [ 145 ] use AoA spectrums coming from multiple access points to determine a mobile client’s position from a set of measured anchor positions based on an associated cost metric. Additional input data include meta-data that are not measured such as room boundaries and a permanent obstacle set. The RSA model  [ 153 ] measures object characteristics based on separate models. The object characteristics include surface curvature, surface boundary, and material. The surface boundary is determined with a surface reflection model that predicts RSS series of a reflection surface with fixed-size surface boundaries. The surface boundary of a measured RSS series is determined by matching the RSS series to a set of predicted RSS series produced with the surface reflection model based on a similarity metric. The Rician model in  [ 90 ] was derived from the Rician model in  [ 218 ] . The model compares a measured radar cross section with pre-simulated radar cross sections to identify a class based on log-likelihood computations. The measured cross section receives the class of the pre-simulated radar cross section that results in the largest computed log-likelihood value. MLE is used to estimate data of interest from noisy measurement signals. The model in  [ 152 ] uses reference data and a likelihood function to return a data estimate that maximizes the likelihood function value. Joint spatial and Doppler-based ego-motion estimation in  [ 134 ] computes state PDFs for a vehicle’s yaw rate, longitudinal velocity and lateral velocity by using a joint optimization problem. Mean velocity  [ 161 , 87 ] is measured with a set of equations and pulse-pair phase estimation, which relies on a correlation product that depends on the input data.

 
 
 TABLE XII: Summary of data driven analytical models used in millimeter wave sensing pipelines. The abbreviations are explained throughout Section  IV-D . 
 
 
 
   | 

 
 | 
 \Block 1-7 Shallow black box | 
 | 
 | 
 | 
 | 
 | 
 | 
 \Block 1-8 Deep black box | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 | 
 SVM | 
 SVDD | 
 MPM | 
 SOM | 
 Decision tree | 
 Clustering | 
 k-NN | 
 LVQ | 
 One-versus-one SVM | 
 AdaBoost SVM | 
 Random forest | 
 Feed-forward NN | 
 CNN | 
 LSTM | 
 CNN+LSTM | 

 
 [ 158 ] | 
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
 X | 
 | 
 | 

 
 [ 42 ] | 
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
 X | 
 | 
 | 
 | 
 | 

 
 [ 141 ] | 
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
 X | 
 | 
 | 

 
 [ 43 ] | 
 | 
 | 
 | 
 | 
 X | 
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

 
 [ 159 ] | 
 | 
 | 
 | 
 | 
 X | 
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

 
 [ 46 ] | 
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
 X | 
 | 

 
 [ 47 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 X | 
 | 
 X | 
 X | 

 
 [ 143 ] | 
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
 X | 
 | 
 | 
 X | 
 | 

 
 [ 49 ] | 
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
 X | 
 | 
 | 
 | 
 | 

 
 [ 50 ] | 
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
 X | 
 | 
 | 
 | 
 | 

 
 [ 174 ] | 
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
 X | 
 | 
 | 

 
 [ 56 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 63 ] | 
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
 X | 

 
 [ 16 ] | 
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
 X | 

 
 [ 64 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 181 ] | 
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
 X | 
 | 
 | 

 
 [ 194 ] | 
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
 X | 
 | 
 | 

 
 [ 195 ] | 
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
 X | 
 | 
 | 

 
 [ 189 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 

 
 [ 71 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 72 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 

 
 [ 73 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 147 ] | 
 | 
 | 
 | 
 | 
 X | 
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

 
 [ 176 ] | 
 X | 
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

 
 [ 76 ] | 
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
 X | 
 | 

 
 [ 77 ] | 
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
 X | 

 
 [ 78 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 79 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 83 ] | 
 | 
 X | 
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

 
 [ 84 ] | 
 | 
 | 
 X | 
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

 
 [ 91 ] | 
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
 X | 
 | 
 | 
 | 
 | 

 
 [ 149 ] | 
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
 X | 
 | 
 | 
 | 

 
 [ 93 ] | 
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
 X | 
 | 
 | 

 
 [ 103 ] | 
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
 X | 
 | 
 | 

 
 [ 104 ] | 
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
 X | 
 | 
 | 
 | 

 
 [ 120 ] | 
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
 X | 
 | 
 | 

 
 [ 123 ] | 
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
 X | 
 | 
 | 

 
 [ 124 ] | 
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
 X | 
 | 
 | 

 
 [ 125 ] | 
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
 X | 

 
 [ 126 ] | 
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
 X | 

 
 [ 127 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 129 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 130 ] | 
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
 X | 
 | 
 | 

 
 [ 137 ] | 
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
 X | 
 | 
 | 
 | 
 | 

 
 [ 138 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 139 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 
 
 

#### IV-D 2 Data driven modeling

 
 This type of modeling solely relies on information contained in the input data itself as well as the feature sets to construct a model  [ 214 ] . Model construction typically involves choosing a model type and associated hyperparameters, fitting the model parameters to a dataset designated for training and testing the model performance on a dataset designated for validation. Fitting and validation are performed continuously in an iterative manner based on a cost or reward function by changing the model type or hyperparameters until the model exhibits the desired performance on the validation dataset  [ 219 , 220 , 221 ] . Data driven models are classified either as a shallow black box or deep black box model. Both types are non-deterministic and solely rely on data for model construction. The difference between the two is that in shallow black box models, the input only passes through a single model structure before obtaining a model output while in a deep black box model, the input goes through multiple submodels such as layers, decision trees, or SVMs .

 
 
 Table  XII presents our analysis of data driven approaches used in millimeter wave sensing applications. Next, we explain different approaches used under shallow and deep black box models.

 
 
 Shallow Black Box Models 

 
 Supervised shallow black box models, such as SVMs and decision trees, used for classification can perform well with limited datasets of small size  [ 222 ] . However, this comes at the cost of requiring carefully crafted features. The features are either crafted manually  [ 43 , 159 , 147 , 78 , 79 ] or automatically in a linear fashion  [ 47 , 50 , 64 , 176 ] . This process is labor intensive and/or can limit the eventual model performance for more complicated tasks during model validation.

 
 
 The pipelines deploying SVMs use it to learn and execute a C-Support Vector Classification (C-SVC) task. SVMs are applicable to binary classification tasks in which an optimal hyperplane is learned from training data. The hyperplane divides the data sample space into two sections, each corresponding to a given class. The classification is based on a decision function, which uses a set of support vector samples and a kernel function. The kernel function introduces non-linearity to the optimal hyperplane. Additional information on how SVMs learn can be found in  [ 220 ] .

 
 
 Decision trees learn a tree structure containing nodes, branches, and leaves from a training dataset. Nodes represent binary input attribute tests, branches represent test outputs, and leaves represent a class that can be assigned. An unseen input dataset goes throughout the entire tree and is ultimately assigned to a single class label  [ 43 ] . More information regarding decision tree construction can be found in  [ 221 ] .

 
 
 Most approaches using a SVM or decision tree have been classified as an ensemble of several classifiers. These ensembles are formed with special multi-class and ensemble learning strategies such as one-versus-one  [ 47 , 64 , 78 , 79 , 139 ] , random forest  [ 42 , 45 , 143 , 49 , 50 , 91 , 137 ] and AdaBoost  [ 189 ] . Commonly, by letting several weak classifiers assign class labels to a given input, the final classification output is determined via majority voting.

 
 
 Other models include clustering  [ 53 , 160 , 73 , 136 ] , Support Vector Data Description (SVDD)   [ 83 ] , Minimax Probability Machine (MPM)   [ 84 ] , Self-Organizing Map (SOM)   [ 56 ] , Learning Vector Quantization (LVQ)   [ 56 ] , and k-Nearest Neighbors (k-NN)   [ 72 , 127 , 129 ] . Clustering is an unsupervised learning algorithm to divide data into several clusters of data points that are similar to one another. Different techniques such as K-means  [ 160 , 73 ] , DBSCAN  [ 53 , 138 ] , Mean shift  [ 71 ] , or DenStream  [ 136 ] clustering have been used. SVDD and MPM are similar to a SVM . They differ from a SVM because both learn a hypersphere rather than a hyperplane that separates all data and feature samples into two classes  [ 83 , 84 ] . A SOM is an unsupervised one layer neural network that reduces n-dimensional input vectors into a 2D feature map. A class is assigned to an unseen input sample based on how close the input sample corresponds to a feature map weight vector based on a distance metric. The LVQ is a supervised two layer neural network containing a competitive and fully connected layer. The competitive layer is similar to a SOM apart from having a limited and predefined number of outputs instead of having an output for every feature map weight vector and containing a transfer function. The output of the transfer function is fed into a fully connected layer to return a classification result for a set of user defined classes  [ 56 ] . A k-NN model assigns a class to an input based on a majority vote of labels from k-nearest samples in a given data space.

 
 
 
 Deep Black Box Models 

 
 Most deep black box models belong to the machine learning paradigm called deep learning. The basic idea of deep learning is that by concatenating a number of submodels (layers), increasing the number of computation nodes in a submodel (neurons), and combining every submodel with an activation function, a non-linear vector mapping function that varies in complexity is learned from training data. The varying complexity comes at the cost of requiring large training datasets and a long training time. Additional information on deep learning can be found in  [ 219 ] .

 
 
 Convolutional Neural Networks (CNNs) found in the pipelines are mainly combined with 2D spectrograms  [ 48 , 53 , 92 , 93 , 123 , 127 , 130 ] and 2D radar images  [ 174 , 181 , 194 , 195 , 103 , 120 ] . CNNs have been used extensively for images originating from vision and object detection domains in the past. Spectrograms and radar images share similar characteristics with these images. The data types are 2-dimensional and features relevant to the mapping function are made up of values local to one another in the value matrix. Another observation is that many approaches base their model on existing vision and object detection models such as VGG  [ 92 , 93 ] , ResNet  [ 92 , 123 ] , ZFnet  [ 195 ] , Faster R-CNN  [ 181 , 130 ] , YOLO  [ 194 ] and FCOS  [ 120 ] . The pipelines therefore rely on experience gathered with images in the vision and object detection domains. In addition, a few approaches have tried using a temporal CNN with 1D profile  [ 158 ] data and a feature distributed CNN with a single point cloud frame  [ 124 ] as input.

 
 
 TABLE XIII: Summary of hybrid analytical models used in millimeter wave sensing pipelines. The abbreviations are explained throughout Section  IV-D . 
 
 
 
   | 

 
 | 
 \Block 1-8 Model driven | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 \Block 1-5 Data driven | 
 | 
 | 
 | 
 | 

 
 | 
 \Block 1-3 White box | 
 | 
 | 
 \Block 1-5 Grey box | 
 | 
 | 
 | 
 | 
 \Block 1-1 Shallow black box | 
 \Block 1-4 Deep black box | 
 | 
 | 
 | 

 
 | 
 Path geometry | 
 Triangulateration | 
 Trilateration | 
 Bayesian filter | 
 Scan matching | 
 
 
 Association/ Allocation 
 | 
 
 
 Angle/Time model 
 | 
 RSS model | 
 Clustering | 
 Random forest | 
 CNN | 
 LSTM | 
 CNN+LSTM | 

 
 [ 45 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 

 
 [ 48 ] | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 

 
 [ 53 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 

 
 [ 160 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
 [ 92 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 

 
 [ 95 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
 [ 186 ] | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 
 
 X 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 

 
 [ 128 ] | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 

 
 [ 136 ] | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 
 An approach that can be combined with time sequence data is the Recurrent Neural Network (RNN) . RNNs are very good at discovering time dependencies among these sequences. Most approaches adopt the Long Short Term Memory (LSTM) network which is a special type of RNN that deals with the vanishing gradient problem noted when training traditional RNNs   [ 128 ] . The authors in  [ 77 ] use a Gated Recurrent Unit (GRU) layer. This layer is comparable to a LSTM layer apart from not having separate memory cells. The LSTM network is mainly combined with a flattened (voxelized) point cloud frame sequence  [ 46 , 47 ] , Manually extracted 2D spectrogram sequence features  [ 72 ] , RSS fingerprint matrix  [ 143 ] , PCA extracted feature sequence  [ 76 ] and alpha-beta filtered trajectory  [ 128 ] . An important observation here is that LSTM requires some kind of feature extraction prior to using the time sequences for training.

 
 
 CNN / LSTM combinations harness the strengths from both models. Comparable to local spectrogram and radar image features, point cloud frames contain clouds that are located very sparsely. In contrast, the positioning of points inside a given cloud is very compact. Therefore, time distributed CNNs can be used to obtain features from a single spectrogram, radar image and point cloud frame in a sequence while LSTM models the time dependency in the sequence afterwards in an end-to-end fashion  [ 77 , 125 , 138 ] . The authors in  [ 46 ] denote that bi-directional LSTMs converge faster than CNN / LSTM combinations for a person identification task. However, this does not mean that they are better since the authors in  [ 47 ] note that a CNN / LSTM combination outperforms a bi-directional LSTMs network in a human activity detection task. Therefore, model testing remains the norm when creating new models.

 
 
 
 

#### IV-D 3 Hybrid Modeling

 
 In Tables  XI and  XII several rows include multiple crosses, indicating use of multiple models. There are reasons for combining models. The authors in  [ 47 , 56 , 145 ] compare several self-created models to each other. Authors in  [ 46 , 143 , 55 , 72 , 89 , 94 , 96 , 185 , 114 , 127 , 132 , 134 , 135 ] incorporate different processes in their sensing application pipeline that use a distinct model. Several papers use pipelines that include a tree or sequential set of multiple models coming from both the data and model driven paradigms. These sets are considered to be hybrid models. Table  XIII presents our analysis of hybrid models used in millimeter wave sensing application pipelines.

 
 
 In principle, hybrid models are constructed by putting the mathematical and data driven models together in a sequential or tree-based manner. Hybrid models constructed in a sequential manner were found in  [ 45 , 48 , 53 , 160 , 92 ] . In  [ 45 ] , a Bayesian filter was combined with a random forest output to reduce sporadic false-positive errors. A patient behavior detection task first tracks the patient with a combination of clustering and a Bayesian filter. Tracking information is used to construct two dimensional Doppler spectrograms that can be used with a CNN for behavior detection. This limits computational complexity since additional pre-processing with STFT or FWT is omitted  [ 53 ] . Environmental mapping  [ 160 ] is performed with a specific set of steps. First, spatial channel profiles consisting of AoA , AoD and RSS information are associated to a potential reflector through clustering. Afterwards, reflector points are retrieved through elementary geometry. An issue experienced in moving target classification is having micro-Doppler signatures spread across many different range bins in profiles across time. This issue is solved by tracking the position of these bins with a Bayesian filter and association/allocation combination. Afterwards, the bins are passed through STFT and fed into different neural networks for classification  [ 92 ] . Motion behavior detection in  [ 48 ] is performed by first applying a clustering algorithm on point cloud data to form micro-Doppler signature data. The signature data is afterwards fed into a CNN to predict motion behavior.

 
 
 TABLE XIV: Summary of evaluation metrics used for measuring performance of the analytical models mentioned in Section  IV-D . The data driven evaluation metric abbreviations are explained throughout Section  IV-E1 . 
 
 
 
   | 

 
 | 
 \Block 1-10 Data driven | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 \Block 1-3 Model driven | 
 | 
 | 

 
 | 
 Accuracy | 
 Invalid gesture rate | 
 Precision | 
 Recall | 
 Specificity | 
 F1 score | 
 Confusion matrix | 
 ROC | 
 Error | 
 Visual inspection | 
 Accuracy | 
 Error | 
 Visual inspection | 

 
 [ 158 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 

 
 [ 42 ] | 
 X | 
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

 
 [ 43 ] | 
 X | 
 X | 
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

 
 [ 44 ] | 
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
 X | 
 X | 

 
 [ 159 ] | 
 X | 
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
 X | 
 | 

 
 [ 45 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 

 
 [ 46 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 | 

 
 [ 47 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 48 ] | 
 X | 
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

 
 [ 143 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 49 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 50 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 51 ] | 
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
 X | 
 X | 
 | 

 
 [ 52 ] | 
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
 X | 
 | 

 
 [ 53 ] | 
 X | 
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

 
 [ 174 ] | 
 X | 
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

 
 [ 160 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 

 
 [ 55 ] | 
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
 X | 
 X | 
 X | 

 
 [ 184 ] | 
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
 X | 
 | 

 
 [ 56 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 58 ] | 
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
 X | 
 | 

 
 [ 144 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 145 ] | 
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
 X | 
 | 

 
 [ 61 ] | 
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
 X | 
 | 

 
 [ 146 ] | 
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
 X | 
 | 

 
 [ 63 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 16 ] | 
 X | 
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

 
 [ 64 ] | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 181 ] | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 

 
 [ 193 ] | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 67 ] | 
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
 X | 

 
 [ 194 ] | 
 | 
 | 
 X | 
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

 
 [ 190 ] | 
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
 X | 
 | 

 
 [ 195 ] | 
 | 
 | 
 X | 
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

 
 [ 189 ] | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 72 ] | 
 X | 
 | 
 X | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 73 ] | 
 X | 
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

 
 [ 147 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 176 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 76 ] | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 77 ] | 
 X | 
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

 

 
 
 TABLE XIV: Continued 
 
 
 
   | 

 
 | 
 \Block 1-10 Data driven | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 \Block 1-3 Model driven | 
 | 
 | 

 
 | 
 Accuracy | 
 Invalid gesture rate | 
 Precision | 
 Recall | 
 Specificity | 
 F1 score | 
 Confusion matrix | 
 ROC | 
 Error | 
 Visual inspection | 
 Accuracy | 
 Error | 
 Visual inspection | 

 
 [ 78 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 79 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 161 ] | 
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
 X | 

 
 [ 83 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 

 
 [ 84 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 

 
 [ 89 ] | 
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
 X | 
 | 

 
 [ 90 ] | 
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
 X | 
 | 
 | 

 
 [ 91 ] | 
 X | 
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

 
 [ 92 ] | 
 X | 
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

 
 [ 149 ] | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 93 ] | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 94 ] | 
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
 X | 
 X | 
 | 

 
 [ 95 ] | 
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
 X | 
 | 

 
 [ 96 ] | 
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
 X | 

 
 [ 150 ] | 
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
 X | 
 X | 

 
 [ 103 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
 [ 104 ] | 
 X | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 152 ] | 
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
 X | 
 | 

 
 [ 185 ] | 
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
 X | 
 X | 
 | 

 
 [ 114 ] | 
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
 X | 
 X | 
 X | 

 
 [ 115 ] | 
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
 X | 
 | 

 
 [ 153 ] | 
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
 X | 
 | 

 
 [ 120 ] | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 186 ] | 
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
 X | 
 | 

 
 [ 123 ] | 
 X | 
 | 
 X | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 124 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 125 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 126 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 127 ] | 
 X | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 

 
 [ 128 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 129 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 130 ] | 
 | 
 | 
 X | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 132 ] | 
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
 X | 
 | 

 
 [ 133 ] | 
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
 X | 
 | 

 
 [ 134 ] | 
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
 X | 
 | 

 
 [ 135 ] | 
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
 X | 
 | 

 
 [ 136 ] | 
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
 X | 
 | 

 
 [ 137 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 138 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
 [ 139 ] | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 
 There are two versions of tree-based hybrid models as addressed in  [ 95 , 186 , 128 , 136 ] . In the first version, the tree-based hybrid model either branches off to perform multiple processes simultaneously or fuses the output of several simultaneously performed processes to perform a single consecutive process. In addition to vision data fusion with a Bayesian filter, millimeter wave data is first clustered  [ 95 ] . Simultaneous Localization And Mapping (SLAM) is solved by first clustering a raw radar scan to make it more sparse. Secondly, the scan is fused with odometry sensor data in a particle filter to estimate an odometry pose. Meanwhile, scan matching with a reference scan is implemented on the sparse radar scan for retrieving new parameters used for updating the cluster model  [ 136 ] . In the second tree-based hybrid model version, a model decision is taken based on a particular context at particular points in the tree. An air writing pipeline  [ 128 ] tracks consecutive three dimensional locations with an α − β \alpha-\beta filter. The locations are determined via trilateration with range estimates coming from several radars. Afterwards, both a CNN and LSTM are tested on a character recognition task. Another model pipeline used for solving the SLAM problem  [ 186 ] first chooses a triangulateration, angle/time, or RSS model for both client and anchor node localization based on whether the environment map is known or not. Secondly, a decision between a RSS or angle/time model is made for estimating obstacle surfaces. Thirdly, the surface limits are determined either by analyzing the difference in RSS at AoAs situated at the limits or interaction of obstacle sides if the shape is two dimensional. Fourthly, clustering is used to reduce measurement bias due to noise. Lastly, an extended Kalman filter is deployed to improve the results.

 
 
 

#### IV-D 4 Gaps and Challenges

 
 An interesting observation derived from Tables  XI and  XII is that use of white box and shallow black box models in the context of millimeter wave sensing applications is restricted. White box models solely rely on physical and/or mathematical knowledge for model construction. Several shallow black box models, such as SVMs and decision trees, rely on manually crafted data features for model construction  [ 43 , 159 , 147 , 78 , 79 ] . Crafting features manually relies heavily on experience and knowledge of domain experts. The observation and information regarding white box and shallow black box model types suggest that there is ample room for research geared towards better understanding the millimeter wave sensing environment, i.e., ample room for analyzing millimeter wave sensing environment dynamics and uncertainties. Secondly, Table  XII indicates that most research with deep black box models is restricted to CNNs and LSTMs . This leaves room for exploring the use of Temporal Convolutional Networks (TCNs) for modeling time dependency in raw data sample or data feature sequences. Lastly, Tables  XI and  XII indicate that the variety of models used in the millimeter wave sensing application pipelines is limited. Many application pipelines use CNNs , LSTMs , Bayesian filters, and/or association/allocation. It is a challenge to increase the model variety and thus the knowledge on how analytical modeling in the millimeter wave sensing environment can be approached.

 
 
 
 

### IV-E Modeling Evaluation 

 
 In this section, we review evaluation metrics used to measure performances of models employed in millimeter wave sensing applications as mentioned in Section  IV-D . We also describe techniques used to improve data driven and grey box model training performance and evaluation.

 
 

#### IV-E 1 Performance Evaluation Metrics

 
 Table  XIV summarizes performance metrics that are used in the reviewed papers. It can be seen that confusion matrix and accuracy are the most often used metrics in data driven model performance assessment. In training and validation of data driven models, the validation dataset is used to make choices about the model, including its hyperparameters  [ 219 ] . After iteratively training and validating the model, the final model performance in relation to the application goal is evaluated on a held out test dataset consisting of samples that the model has not seen before. The outcome of the testing is compared to data labels and results are presented in the form of a confusion matrix and/or metrics that summarize the content of a confusion matrix. The confusion matrix  [ 78 , 123 ] shows model classification performance for every class in the test dataset. The rows and columns represent ground truth and predicted classes respectively. Diagonal elements in the matrix denote the number, or fraction, of true positives for every class. When the diagonal elements are removed, the remaining row elements denote the false positives and remaining column elements the false negatives for every class. Positive samples are those that have been predicted to belong to a certain class. Negative samples are those that have been predicted to not belong to a certain class. Accuracy is defined as the overall proportion of predicted test dataset labels that match with the ground truth test dataset labels. Precision is a ratio of the number of true positive predictions to the number of all positive predictions made for a certain class. Recall is a ratio of the the number of true positive predictions to the number of all samples with a positive label in the ground truth dataset for a certain class  [ 219 ] . Sometimes the inverse of accuracy and precision, the so called misclassification rate  [ 43 , 51 , 94 ] and false discovery rate  [ 193 , 189 ] , are also used. F1 score is the harmonic mean of precision and recall  [ 123 ] . Specificity is a ratio of the number of true negative predictions to the number of all samples with a negative label in the ground truth dataset for a certain class  [ 104 ] . The area under the Receiver Operating Characteristic (ROC) curve has also been used, which shows the degree of output separability  [ 181 ] . A paper on gesture recognition used invalid gesture rate in conjunction with accuracy and misclassification rate to test data driven modeling performance for a set of gesture classes. Invalid gesture rate is defined as a ratio of samples classified as invalid gesture compared to the total number of samples in the test dataset  [ 43 ] . Several model driven model results are also reported with the accuracy metric.

 
 
 Model driven model performance is measured based on output comparison with a baseline, i.e., via an error metric with a well-established measurement technique. Error is defined as output deviation compared to the baseline and can be measured through various parameters, such as Root Mean Square (RMS) , Cramér-Rao Lower Bound (CRLB) , mean, and standard deviation. CRLB is defined as the minimum achievable variance  [ 152 ] of a certain parameter. A few data driven models were also tested with an error metric. Both data and model driven models sometimes rely on visual inspection to determine model performance. Visual inspection can be used to assess feature separability  [ 158 , 83 , 84 , 127 ] or model behavior  [ 44 , 55 , 67 , 161 , 96 , 150 , 114 ] .

 
 
 TABLE XV: Summary of model improvement, optimization, and evaluation techniques used to improve and evaluate data driven and grey box model performance. 
 
 
 
   | 

 
 | 
 
 \Block 
 1-5 Regularization 
 | 
 | 
 | 
 | 
 | 
 \Block 1-3 Hyperparameter tuning | 
 | 
 | 
 \Block 1-3 Cross validation | 
 | 
 | 
 
 \Block 
 1-4 Training stabilization 
 | 
 | 
 | 
 | 

 
 | 
 
 
 Loss constraint 
 | 
 Dropout | 
 
 
 Noise training 
 | 
 
 
 Weight decay 
 | 
 
 
 Early stopping 
 | 
 Manual | 
 Grid search | 
 Adaptive | 
 K fold | 
 Monte carlo | 
 Leave one out | 
 
 
 Batch normalization 
 | 
 
 
 Weight initialization 
 | 
 
 
 Architecture addition 
 | 
 Momentum | 

 
 [ 158 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 42 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 43 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 159 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 45 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 46 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 47 ] | 
 
 
 X 
 | 
 X | 
 | 
 | 
 | 
 X | 
 X | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 48 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 50 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 53 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 174 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 56 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 63 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 16 ] | 
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
 
 
 X 
 | 
 | 
 | 
 | 

 
 [ 181 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 194 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 195 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 72 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 76 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 77 ] | 
 | 
 X | 
 
 
 X 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 78 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 79 ] | 
 
 
 X 
 | 
 | 
 | 
 | 
 | 
 X | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 91 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 92 ] | 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 
 
 X 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 93 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 150 ] | 
 
 
 X 
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

 
 [ 103 ] | 
 | 
 X | 
 | 
 | 
 
 
 X 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 104 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 120 ] | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 X | 

 
 [ 123 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 X | 
 | 
 
 
 X 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 124 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 125 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 X | 
 | 
 | 
 | 
 | 
 | 
 X | 

 
 [ 126 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 127 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 128 ] | 
 | 
 X | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 
 
 X 
 | 
 
 
 X 
 | 
 | 
 | 

 
 [ 129 ] | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 130 ] | 
 | 
 X | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 

 
 [ 137 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
 [ 138 ] | 
 | 
 X | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 
 
 X 
 | 
 | 
 
 
 X 
 | 
 | 

 
 [ 139 ] | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 | 
 X | 
 | 
 X | 
 | 
 | 
 | 
 | 

 
   | 

 

 
 
 

#### IV-E 2 Model Improvement and Evaluation Techniques

 
 After developing mathematical proofs or conducting numerous quantitative experiments, physical and/or mathematical information required for solving a particular problem becomes known. Once this information is known, the white box model construction process for solving the problem is straightforward and the resulting model will exhibit good performance  [ 223 ] . This is not the case for grey and black box models. Grey box models involve hidden state updates and/or use of cost and similarity functions that depend on input data or features. This requires grey box model evaluation through numerous experiments every time the sensing application context or input data or feature distribution changes. As explained in Section  IV-D2 , black box models are constructed through an iterative process in which finally a model is created that performs well on a test dataset or feature set. This section elaborates on several problems that occur during grey box and black box model evaluation and black box model creation, and how these problems can be solved with special model improvement, evaluation, and/or optimization techniques. These techniques are summarized in Table  XV . Because new model creation is explained in Section  IV-D , we do not focus on it in this section. Elements causing training instability are not addressed. Prediction denoising through Non-Maximum Suppression (NMS) or fusion performed by  [ 181 ] is out of context in this section and is not addressed either. Next we explain main model improvement, optimization, and evaluation techniques used in data driven and grey box models of millimeter wave sensing applications.

 
 
 Regularization 

 
 An issue frequently encountered during model training is model overfitting, in which model parameters and mapping function completely adapt to the training data or feature set. This will cause the model to exhibit sub-optimal performance on held-out (unseen) data. Various regularization techniques try to avoid overfitting and help the model to perform well on a wider variety of data it might encounter when deployed in specific applications  [ 219 ] . For example, the ability of a cost function to minimize or a reward function to maximize itself can be limited with special constraint  [ 158 , 47 , 195 , 78 , 79 , 150 ] and weight decay  [ 181 , 92 , 120 , 128 ] parameters added to the function. It can be decided to occasionally stop a parameter update for a limited number of model parameters in a particular training cycle through means of dropout  [ 46 , 47 , 48 , 53 , 63 , 77 , 92 , 103 , 125 , 126 , 127 , 128 , 130 , 138 ] . In early stopping, the validation data or feature set can be used to inspect the model performance after every training cycle. In case the performance starts to diminish compared to previous training cycles, model training will be stopped  [ 103 , 128 ] . A certain percentage of the training set can include noise samples to make the eventual model robust to input noise  [ 77 ] .

 
 
 
 Hyperparameter Tuning 

 
 Most data driven models involve some kind of manual hyperparameter tuning as shown in Table  XV . Hyperparameters include number of layers, neurons, forest size, forest depth, number of training cycles, number of data or feature inputs before an update is applied to model parameters, etc. Sometimes it is difficult to set the right initial hyperparameters due to a lack of a priori knowledge and experience. In this case, an exhaustive hyperparameter search using a grid search  [ 47 , 50 , 78 , 79 ] can be implemented, through which a complete grid of hyperparameter values is constructed. Afterwards, for every possible hyperparameter combination, a model is created and its performance is evaluated. The set of final hyperparameters is selected as the one that gives the best model performance. Not adapting the learning rate, i.e., a hyperparameter that controls how much parameter update is used to change the model parameters, during training cycles can result in long execution time for model training or training instability. To combat these problems, adaptive learning rate techniques such as Adaptive Moment (ADAM) estimation  [ 158 , 46 , 47 , 53 , 63 , 76 , 92 , 103 , 123 , 124 , 126 , 127 , 128 , 130 ] or manual learning rate decay  [ 174 , 125 ] can be employed.

 
 
 
 Cross Validation 

 
 Cross validation refers to training a model and evaluating model performance multiple times on a variety of different data or feature set splits. Splitting refers to partitioning a data or feature set into a subset designated for model training, a subset designated for model validation, and a subset designated for model testing. If model performance results are retrieved from a single training, validation, and test split, the model performance results will have a split specific bias. This means that the performance result will strongly deviate from the mean performance result that could be expected based on a given data or feature set. To combat this, model training, validation and testing can be executed on a variety of splits for more robust performance result evaluation. This can be implemented very exhaustively with leave-one-out cross validation in which the train-validate-test split is repeated for every possible combination  [ 137 , 139 ] . Instead, methods such as Monte Carlo  [ 123 ] and k-fold cross validation  [ 47 , 50 , 78 , 79 , 91 , 92 , 125 , 129 , 137 , 139 ] can be used in which the data or feature set splits are limited to a certain number.

 
 
 
 Training Stabilization 

 
 Elements like gradient exploding, gradient vanishing, internal covariate shift, noisy parameter updates, and bad parameter initialization have a negative impact on training stability. Training stability refers to an analytical model whose cost or reward function gradually minimizes or maximizes over time when new training samples are encountered during model training. Training instability examples include the cost or reward function prematurely becoming constant, a cost function that suddenly starts to increase with continuous acceleration, etc. Batch normalization  [ 63 , 16 , 194 , 92 , 123 , 124 , 128 , 138 ] , initializing the model parameters prior to training with Xavier initialization  [ 194 , 126 , 127 , 128 ] , making changes to existing model types with for example the ResNet layer  [ 92 , 123 ] or LeakyRELU activation function  [ 48 , 53 , 77 , 138 ] and parameter update averaging over multiple training cycles with momentum  [ 181 , 195 , 120 , 125 ] have been proposed to solve the problem of training instability.

 
 
 
 

#### IV-E 3 Gaps and Challenges

 
 A challenge often encountered in model evaluation is determining a set of metrics that can completely and objectively assess model performance. From Table  XIV , it can be concluded that there is no such universal metric and none of the existing works covered all or even a majority of the given performance metrics. This, in combination with model variation, makes benchmarking between models difficult. A widely used metric that does not portray a complete performance picture is the accuracy metric. Accuracy cannot discriminate between a good or bad performing model in case data or feature sets are heavily skewed in terms of class distribution. More information regarding the selection of a suitable set of performance metrics can be found in  [ 224 ] . Designing models that perform well in both lab and real-life contexts is still a major challenge. Model simulation  [ 184 , 190 , 186 , 133 ] , controlled experiments  [ 47 , 83 , 84 , 104 ] , and system challenges related to generalization  [ 142 ] and impact  [ 107 ] are frequently observed in scientific studies. Examples of controlled experiments and restricted experimental environment include test subjects performing activities directly in front of a radar  [ 47 ] and object detection with very limited and pre-defined objects placed at specific positions in a radar’s FOV   [ 83 , 84 ] . Table  XV shows that many papers do not harness the strength of multiple regularization methods. It is reasonable to think that combining loss constraint, dropout, and/or weight decay is not required or it is undesired because a single one of these methods may already result in good model performance on unseen data or they might negatively influence the training process  [ 219 ] . Therefore, combining loss constraint, dropout, and/or weight decay to improve model performance on unseen data is always subject to extensive evaluation. However, using early stopping in combination with other regularization methods can omit the need for exhaustively tuning the required number of training cycles to achieve optimal model performance while not negatively influencing the training process  [ 219 ] .

 
 
 
 
 

## V Challenges, Trends, and Future Perspective 

 
 This section presents and explains identified scientific and technological challenges and trends for applications of millimeter wave as a sensing technology. The challenges and trends have been categorized into several challenge and trend categories: hardware, unsupervised representation learning, support from other sensing domains, application integration, and crowd analysis. In addition, this section also provides a future perspective for applications of millimeter wave as a sensing technology.

 
 

### V-A Hardware 

 
 Several challenges related to the hardware used to collect data have an effect on how well a millimeter wave sensing methodology performs. Human vital sign monitoring pipelines are currently not reliable enough in the context of multiple humans that are all located at varying distances from each other and at various positions in reference to the data collection system. Issues include different relative error results compared to a baseline in the context of changing measurement positions (measuring in front or at a side of a test subject)  [ 147 , 107 ] , impact of random body movement  [ 144 , 62 , 105 ] , and measurement occlusion issues when humans are standing too close to each other  [ 144 , 147 ] . One study focused on finding the most optimal vital sign sensing position  [ 191 ] . Several pipelines experience performance deterioration at large distances from the data collection system  [ 79 , 86 , 97 , 102 , 113 , 114 , 115 , 124 ] , in the presence of solids such as occluding objects  [ 93 , 100 , 102 , 114 ] and brick/concrete wall barriers  [ 79 , 112 , 113 ] , not using 3D printed placement constraints  [ 139 ] , and in the context of hard to distinguish entities  [ 78 , 91 , 149 , 103 , 116 , 153 ] . A challenge in concealed object detection and pedestrian detection is the variable reflection intensity caused by different types of clothing people wear  [ 189 , 198 ] . One gesture recognition methodology in a car analyzed the impact of measurement position  [ 42 ] . The optimal sensor placement was found to be on the center console (for use by front seat passenger and the driver) or in between the backs of the front seats (for use by the back-seat passengers). Performance deterioration was encountered when the sensor was placed too close to detectable objects such as the gear shift. Robustness against environmental effect analysis is limited to ego-motion estimation in  [ 132 ] . Other challenges for manufacturers are to bring the cost of millimeter wave systems down to allow large deployments on a budget (e.g., for crowd analytics) and to improve the design and the form factor such that these systems do not cause architectural and aesthetic concerns for environments where they are deployed.

 
 
 

### V-B Unsupervised Representation Learning 

 
 Unsupervised representation learning has been used to pre-train a denoising autoencoder  [ 174 ] in millimeter wave application pipelines. The network can be used afterwards to extract features from a dataset that generalize to a wide variety of end tasks  [ 213 ] . Features are transferred from a pre-trained neural network to a supervised end learning task network. The networks used during pre-training and the end task share no learning task relation  [ 225 ] . More information can be found in a review written by Bengio et al.  [ 213 ] . In 2014, a new breakthrough in unsupervised pre-training was realized by Dosovitskiy et al.  [ 226 ] . A concept called self-supervised learning was envisioned in which a supervised neural network tries to classify data transformations applied to unlabeled data in the pre-training stage. Since millimeter wave data is unlabeled too when it is sampled from the underlying hardware, it remains to be determined if unsupervised representation learning can reduce the amount of labeled data required for training end task deep learning models by using either unsupervised or self-supervised learning in a given pre-training stage. Millimeter wave datasets that can be used for experimentation include the UWCR radar mini, NuScenes, RadHAR, and Solinteraction datasets  [ 47 , 93 , 202 , 139 ] . However, the ability to explore is currently hindered by a lack of more datasets as indicated in  [ 103 ] for pose estimation.

 
 
 

### V-C Support From Other Sensing Domains 

 
 Some pipelines apply sensor fusion of millimeter wave data with data originating from other sensing domains. Sensing domains include vision  [ 51 , 61 , 146 , 66 , 67 , 71 , 95 , 105 , 120 ] , depth  [ 44 , 105 ] , lidar  [ 71 ] , inertial measurements  [ 89 ] and exerted forces  [ 91 ] . Data originating from these domains complement millimeter wave data and therefore cause more accurate analytical modeling performance in several situations. Analytical modeling performance in on-road detection and tracking suffers from limited spatial resolution of, and noise in, millimeter wave data  [ 51 , 61 , 67 , 95 , 120 ] . Odometry information model performance also suffers from noise in millimeter wave data  [ 89 ] . Data originating from other sensing domains have been used to guide a radar collection system to the most accurate vital sign sensing position  [ 105 ] and have lead to increased analytical model performance  [ 44 , 66 ] and model state accuracy in object detection for visually impaired people  [ 44 ] . Millimeter wave data also provide benefits to the other sensing domains in return. The data are used for example to differentiate objects based on material, density or volume where exerted forces only measure similar object geometries  [ 91 ] . In the future, fusion with other sensing domains can be extended. For example, when measuring crowd density in a limited space inside a building, fusion with heat energy obtained through a temperature sensor can be explored.

 
 
 

### V-D Application Integration 

 
 A variety of different sensing applications integrate different application types. Examples include combining communication and sensing  [ 80 , 140 ] , combining vital sign, activity, and gesture sensing for more robust occupancy detection  [ 81 ] , SLAM   [ 185 , 186 , 154 , 136 ] , detection/tracking and identification  [ 46 , 143 , 123 ] , and detection and activity recognition  [ 77 ] . Detection and tracking make identification and activity recognition more robust in unknown environments. Research combining communication and sensing is limited to vision  [ 80 ] and range simulations  [ 140 ] . Due to performance deficiencies with communication-only access point sensing  [ 80 ] , sensing capabilities built into special access points are envisioned to enable better elderly monitoring and building analytics without extra costs related to installation of dedicated sensor hardware. The combination of communication and sensing in Vehicle to Vehicle (V2V) scenario’s is envisioned to enable robust driver assistance systems  [ 227 ] . Yassin et al.  [ 186 ] denote that SLAM research with millimeter waves is still at its infancy.

 
 
 

### V-E Crowd Analysis 

 
 Throughout the review process no papers have been found that explore crowd analysis with millimeter wave sensing. The non-image based crowd counting review by Kouyoumdjieva et al.  [ 228 ] is recommended as an introductory read since millimeter wave crowd analysis applications in the future will belong to the non-image based crowd analysis application category. Future millimeter wave crowd analysis research will include density/size, flow/trajectory/movement, and activity/behavior analysis. Density/size analysis refers to counting the number of people and their distribution in a given area. Kouyoumdjieva et al.  [ 228 ] denote that for disaster management and city-wide public transportation crowd counting tier models should be developed that have the ability to provide a macro crowd count based on local estimates calculated over different micro or meso area’s. Flow/trajectory/movement analysis refers to determining major movement flows and directions in a given area, as well as identifying minimum and maximum bounds in terms of target numbers and dynamicity. Activity/behavior analysis refers to determining meso and macro level crowd activities, as well as identifying the correlation between location and activities and minimum and maximum bounds in terms of target numbers and granularity of activities. Major challenges for millimeter wave crowd analysis include methodology scalability to major events including thousands of people and operation security. The cost to deploy a grid of millimeter wave data collection systems is currently too high and deployed grids, in case the grid is not concealed in the environment, will interfere with decor design. Kouyoumdjieva et al.  [ 228 ] denote that no non-image based crowd counting methodologies explore security in the form of robustness to output manipulation and malicious users trying to disable methodology functionality.

 
 
 

### V-F Future Perspective 

 
 We notice that most millimeter wave sensing research is currently limited to a personal lab scope. This means that research is most often performed with one person, a few people, one object or a few objects in simulated or controlled experiment environments. We believe that there is a lot of potential for research geared towards exploring millimeter wave sensing in large scale active and dynamic industrial and urban area’s involving numerous people, other mammals, and or objects in the future.

 
 
 Millimeter wave sensing infrastructures are unobtrusive in nature and will be deployed ubiquitously. Future millimeter wave infrastructures will also co-exist and collaborate with other sensing and communication infrastructures, and serve multiple sensing applications in parallel. For example, sensing capabilities will be integrated into special communication access points in the future to enable better elderly monitoring and building analytics without costs related to installation of dedicated sensor hardware  [ 80 ] . Integration of communication and sensing in V2V scenario’s will enable robust driver assistance systems  [ 227 ] .

 
 
 Analytical modeling will become more prominent in millimeter wave sensing application pipelines. Jiang et al.  [ 158 ] have presented an analytical model that has the ability to extract input data features that are environment and user specific information (i.e. domain) independent in the context of human activity detection. We believe that the ability to learn extraction of domain independent input data features is an important goal for future research in the context of analytical models that work with millimeter wave data. This ability will allow analytical models to be robust against influences from a variety of domains and therefore to perform well in real-life scenario’s in the future. Analytical modeling research should be conducted to determine which domain influences can or cannot be mitigated, which domains can or cannot be integrated in an analytical model, how an analytical model can learn latent domains from the input data, and which analytical model types work best in a wide variety of different application types. For example, several hardware challenges can be regarded as a domain, rather than trying to mitigate hardware challenge influence and application robustness deterioration with cancellation methods  [ 229 ] per challenge, using sensor fusion, or integrating application types. A variety of different models that extract domain independent input data features can be found in papers which do not consider millimeter wave sensing. For example, papers that consider WiFi CSI   [ 230 , 231 , 232 ] . An important step in analytical modeling, due to the labor intensive nature of labelling millimeter wave data, is the development of unsupervised models that achieve performance that is on par with supervised counterparts. We believe that it is also important to investigate how learning domain independent input data feature extraction can be integrated with creation of analytical models through means of unsupervised or self-supervised representation learning.

 
 
 Stimuli such as sound, light, stress, anxiety, overtraining, temperature, humidity, etc. have an effect on our health and vitality  [ 233 ] . The COVID-19 pandemic and confinement measures taken during the pandemic result in stress and anxiety in a wide variety of different people  [ 234 ] . We believe that this will spark debate in the general public about the effect of stress and anxiety and how we can reduce stress and anxiety in our daily lives in the near future. We also believe that this will stimulate research regarding the effect of stimuli on health and well-being with millimeter wave sensing systems. Vital sign sensing with millimeter wave systems can be used in niches where so called contact sensors, i.e., sensors involving electrodes, air analysis with nasal cannula or mask, a strap-on system, smart watch, smart phone, etc.  [ 229 , 235 ] , cannot be used. Examples include, but are not limited to, skin irritation and allergic reactions, damaged skin (rashes, burns, hives, etc.), in the presence of clothes and obstacles  [ 236 ] , multiple beings, and humans that have certain behavioral conditions (e.g. severe autism, dementia, etc.).

 
 
 
 

## VI Conclusion 

 
 This is the first review that completely covers millimeter wave sensing application pipelines and pipeline building blocks in the form of a systematic literature review to the best of our knowledge. The millimeter wave technology covers a wide bandwidth and its short wavelength gives limited range, giving low signal interference. This means transceivers can be packed very densely in an area without disrupting each others’ communication signals. These properties of the technology not only yield high communication rates, but also provide a great opportunity for sub-millimeter accuracy level sensing of the surroundings, easily penetrating through simple obstacles like plastic and fabric.

 
 
 Our analysis of the literature showed that there are indeed a variety of application types for millimeter sensing that we group into three domains; namely, human, object, and environment sensing. The application pipelines in the literature are made up of (a subset of) five common building blocks: data collection, pre-processing, feature extraction, analytical modeling, and modeling evaluation. There is naming confusion in the literature in terms of which models and techniques take part in which building blocks. In this paper, we provided sharp descriptions of millimeter wave sensing building blocks; i.e., the hardware, algorithms, analytical models, and or model evaluation techniques that are covered by each block.

 
 
 For different applications in the literature, each building block may select from a variety of models and techniques. A close look into many application instances reveals that a large majority of them stick to a combination of a few common solutions in their sensing pipelines. The rest of the models and techniques remain application-specific and their usage is not explored widely in other applications. This is not surprising as this field of research and development is still young and it is safer to rely on widely accepted solutions. For example, it is very common to utilize deep black box models employing CNNs and LSTM networks in the more frequently used data driven modeling, whereas Bayesian filters and association/allocation seem to be the first choices among model driven approaches. In terms of feature extraction, manual feature mapping is the predominant choice of researchers. We therefore encourage the researchers entering, or planning to conduct new research in, the millimeter wave sensing environment to try out new models. This will increase the model variety and thus the knowledge on how analytical modeling in the millimeter wave sensing environment can be approached.

 
 
 

## References

 
 
 [1] 
 
H. Gold. (2020, Mar.) Netflix and YouTube are slowing down in Europe to keep
the Internet from breaking. CNN Business. [Online]. Available:
 https://edition.cnn.com/2020/03/19/tech/netflix-internet-overload-eu/index.html 

 

 
 [2] 
 
J. Hendrickson. (2020, Jan.) 8K TV has arrived. Here’s What You Need to
Know. How-To Geek. [Online]. Available:
 https://www.howtogeek.com/397365/8k-tv-has-arrived.-heres-what-you-need-to-know/ 

 

 
 [3] 
 
O. Soliman, A. Rezgui, H. Soliman, and N. Manea, “Mobile Cloud Gaming: Issues
and Challenges,” in Proc. Intl. Conf. Mobile Web and Information
Sys. , F. Daniel, G. A. Papadopoulos, and P. Thiran, Eds. Springer, 2013, pp. 121–128.

 

 
 [4] 
 
Wireless LAN Working Group. (2012, Dec.) 802.11ad-2012. IEEE Standards
Association. [Online]. Available:
 https://standards.ieee.org/standard/802_11ad-2012.html 

 

 
 [5] 
 
——. (2016, Oct.) 802.11aj-2018. IEEE Standards Association. [Online].
Available: https://standards.ieee.org/standard/802_11aj-2018.html 

 

 
 [6] 
 
——. (2019, Feb.) P802.11ay. IEEE Standards Association. [Online].
Available: https://standards.ieee.org/project/802_11ay.html 

 

 
 [7] 
 
M. Elkashlan, T. Q. Duong, and H. Chen, “Millimeter-wave Communications
for 5G – Part I: Fundamentals [Guest Editorial],” IEEE
Communications Magazine , vol. 52, no. 9, pp. 52–54, 2014.

 

 
 [8] 
 
——, “Millimeter-wave Communications for 5G – Part 2: Applications
[Guest Editorial],” IEEE Communications Magazine , vol. 53, no. 1,
pp. 166–167, 2015.

 

 
 [9] 
 
I. A. Hemadeh, K. Satyanarayana, M. El-Hajjar, and L. Hanzo, “Millimeter-Wave
Communications: Physical Channel Models, Design Considerations, Antenna
Constructions, and Link-Budget,” IEEE Communications Surveys and
Tutorials , vol. 20, no. 2, pp. 870–913, 2018.

 

 
 [10] 
 
X. Wang, L. Kong, F. Kong, F. Qiu, M. Xia, S. Arnon, and G. Chen,
“Millimeter Wave Communication: A Comprehensive Survey,” IEEE
Communications Surveys and Tutorials , vol. 20, no. 3, pp. 1616–1653, 2018.

 

 
 [11] 
 
Texas Instruments. mmWave Radar Sensors – What is mmWave. [Online].
Available: http://www.ti.com/sensors/mmwave/what-is-mmwave.html 

 

 
 [12] 
 
F. Khan and Z. Pi, “mmWave Mobile Broadband (MMB): Unleashing the
3–300GHz Spectrum,” in Proc. 34th IEEE Sarnoff Symp. , May 2011,
pp. 1–6.

 

 
 [13] 
 
I. F. Akyildiz, C. Han, and S. Nie, “Combating the Distance Problem in the
Millimeter Wave and Terahertz Frequency Bands,” IEEE Communications
Magazine , vol. 56, no. 6, pp. 102–108, 2018.

 

 
 [14] 
 
J. Lin and H. Hu. (2017, Sep.) 79GHz to Replace 24GHz for Automotive
Millimeter-wave Radar Sensors. Digitimes Research. [Online]. Available:
 https://www.digitimes.com/news/a20170906PD208.html 

 

 
 [15] 
 
D. C. B. Mariano, C. Leite, L. H. S. Santos, R. E. O. Rocha, and R. C.
de Melo-Minardi, “A Guide to Performing Systematic Literature Reviews in
Bioinformatics,” Jul. 2017, arXiv:1707.05813 [q-bio.QM].

 

 
 [16] 
 
Z. Zhang, Z. Tian, M. Zhou, W. Nie, and Z. Li, “Riddle: Real-Time Interacting
with Hand Description via Millimeter-Wave Sensor,” in Proc. IEEE
Intl. Conf. Communications . IEEE,
2018, p. 1–6.

 

 
 [17] 
 
Z. Li, Z. Yang, C. Song, C. Li, Z. Peng, and W. Xu, “E-Eye : Hidden
Electronics Recognition through mmWave Nonlinear Effects,” in Proc.
16th ACM Conf. Embedded Networked Sensor Sys.  ACM, 2018, pp. 68–81.

 

 
 [18] 
 
B. Özen, S. Baykut, O. Tulgar, A. U. Belgül, I. K. Yalçin, and
D. S. Armağan Şahinkaya, “Foreign Object Detection on Airport Runways
by mm-Wave FMCW Radar,” in Proc. 25th Signal Processing and
Communications Applications Conf. , 2017, pp. 1–4.

 

 
 [19] 
 
S. López-Tapia, R. Molina, and N. Pérez de la Blanca, “Using
Machine Learning to Detect and Localize Concealed Objects in Passive
Millimeter-wave Images,” Intl. Journal of Engineering Applications of
Artificial Intelligence , vol. 67, no. January 2018, pp. 81–90, 2018.

 

 
 [20] 
 
V. K. Klochko, V. V. Strotov, and S. A. Smirnov, “Multiple Objects Detection
and Tracking in Passive Scanning Millimeter-wave Imaging Systems,” in
 Proc. SPIE 11164, Millimetre Wave and Terahertz Sensors and Technology
XII , Oct. 2019, article 111640E.

 

 
 [21] 
 
S. Yeom, D. S. Lee, J. Y. Son, and S. H. Kim, “Concealed Object Detection
using Passive Millimeter Wave Imaging,” in Proc. 4th Intl. Universal
Communication Symp.  IEEE, 2010, pp.
383–386.

 

 
 [22] 
 
L. Guo and S. Qin, “High-Performance Detection of Concealed Forbidden Objects
on Human Body with Deep Neural Networks Based on Passive Millimeter Wave and
Visible Imagery,” Infrared, Millimeter, and Terahertz Waves ,
vol. 40, no. 3, pp. 314–347, 2019.

 

 
 [23] 
 
S. E. Clark, J. A. Lovberg, C. A. Martin, and J. A. Galliano, Jr., “Passive
Millimeter-wave Imaging for Concealed Object Detection,” in Proc.
SPIE Sensors, and Command, Control, Communications, and Intelligence (C3I)
Technologies for Homeland Defense and Law Enforcement , vol. 4708, Aug. 2002,
pp. 128–133.

 

 
 [24] 
 
B. Kapilevich, B. Litvak, A. Shulzinger, and M. Einat, “Portable Passive
Millimeter-wave Sensor for Detecting Concealed Weapons and Explosives Hidden
on a Human Body,” IEEE Sensors , vol. 13, no. 11, pp. 4224–4228,
2013.

 

 
 [25] 
 
S. Yeom, D.-S. Lee, Y. Jang, M.-K. Lee, and S.-W. Jung, “Real-time
Concealed-object Detection and Recognition with Passive Millimeter Wave
Imaging,” Optics Express , vol. 20, no. 9, pp. 9371–9381, 2012.

 

 
 [26] 
 
L. Li, J. Yang, G. Cui, Z. Jiang, and X. Zheng, “Method of Passive MMW Image
Detection and Identification for Close Target,” Infrared, Millimeter,
and Terahertz Waves , vol. 32, no. 1, pp. 102–115, 2011.

 

 
 [27] 
 
L. Yujiri, B. I. Hauss, and M. Shoucri, “Passive Millimeter Wave Sensors for
Detection of Buried Mines,” Detection Technologies for Mines and
Minelike Targets , vol. 2496, pp. 2–6, Jun. 1995.

 

 
 [28] 
 
H. Işıker, S. Demirci, B. Yılmaz, S. Gokkan, and C. Özdemir,
“Detection of small and large hidden metallic objects via passive millimeter
wave imaging system with an auto-segmentation routine,” in Proc. Prog.
in Electromagnetics Research Symp. , 2018, pp. 1362–1365.

 

 
 [29] 
 
V. Mattioli, L. Milani, K. M. Magde, G. A. Brost, and F. S. Marzano,
“Retrieval of Sun Brightness Temperature and Precipitating Cloud Extinction
Using Ground-Based Sun-Tracking Microwave Radiometry,” IEEE Selected
Topics in Applied Earth Observations and Remote Sensing , vol. 10, no. 7, pp.
3134–3147, 2017.

 

 
 [30] 
 
J. A. Nanzer, E. Popova, and R. L. Rogers, “Analysis of the Detection
Modes of a Human Presence Detection Millimeter-wave Radiometer,” in
 Proc. IEEE Antennas and Propagation Society Intl. Symp. , 2010, pp.
1–4.

 

 
 [31] 
 
H. Zong, L. Bao, B. Liu, and J. Qiu, “Application of Convolutional
Neural Network in Target Detection of Millimeter Wave Imaging,” in
 Proc. IEEE Intl. Symp. Antennas and Propagation , 2018, pp. 1217–1218.

 

 
 [32] 
 
C. R. Cabrera-Mercader and D. H. Staelin, “Passive Microwave Humidity
Profile Retrievals using Neural Networks,” in Proc. IEEE Intl.
Geoscience and Remote Sensing Symp. , vol. 4, 1994, pp. 2057–2059.

 

 
 [33] 
 
B. Kapilevich, B. Litvak, and A. Shulzinger, “Passive Non-imaging
mm-Wave Sensor for Detecting Hidden Objects,” in Proc. IEEE Intl.
Conf. Microwaves, Comm., Antennas and Electronic Sys. , 2013, pp. 1–5.

 

 
 [34] 
 
Y. Meng, A. Qing, C. Lin, J. Zang, Y. Zhao, and C. Zhang, “Passive Millimeter
Wave Imaging System Based on Helical Scanning,” Scientific Reports ,
vol. 8, no. 1, May 2018, article 7852.

 

 
 [35] 
 
K. Schmalz, N. Rothbart, P. F. . Neumaier, J. Borngräber,
H. Hübers, and D. Kissinger, “Gas Spectroscopy System for Breath
Analysis at mm-Wave THz Using SiGe BiCMOS Circuits,” IEEE Trans.
Microwave Theory and Techniques , vol. 65, no. 5, pp. 1807–1818, 2017.

 

 
 [36] 
 
F. P. Schloerb, “Millimeter-wave Spectroscopy of Solar System Objects:
Present and Future,” in Proc. European Southern Observatory Conf.
and Workshop , P. A. Shaver and K. Kjar, Eds., vol. 22, Jan. 1985, pp.
603–615.

 

 
 [37] 
 
E. S. Gonçalves, F. C. Teixeira, D. F. Albuquerque, and E. F. Pedrosa,
“Asynchronous mmWave Radar Interference for Indoor Intrusion Detection,”
in Proc. 4th Iberian Robotics Conf. , M. F. Silva, J. Luís Lima,
L. P. Reis, A. Sanfeliu, and D. Tardioli, Eds. Springer, 2020, pp. 367–378.

 

 
 [38] 
 
H. Rodilla, A. A. Kim, G. D. M. Jeffries, J. Vukusic, A. Jesorka, and J. Stake,
“Millimeter-wave Sensor based on a λ \lambda /2-line Resonator for
Identification and Dielectric Characterization of Non-ionic Surfactants,”
 Scientific Reports , vol. 6, no. 1, Jan. 2016, article 19523.

 

 
 [39] 
 
S. Sano, A. Tsuzuki, J. Li, A. Gotou, Y. Makino, and S. Miyake,
“Millimeter-wave Dielectric Measurement of SiC Powders as a Basis of
Millimeter-wave Sintering of Ceramics,” in Proc. Intl. Symp. Novel
Materials Processing by Advanced Electromagnetic Energy Sources . Elsevier, 2005, pp. 151 – 154.

 

 
 [40] 
 
M. Ghasr, S. Kharkovsky, R. Zoughi, and R. Austin, “Comparison of
Near-field Millimeter wave Probes for Detecting Corrosion Pit under Paint,”
in Proc. 21st IEEE Instrumentation and Measurement Technology Conf. ,
vol. 3, 2004, pp. 2240–2244.

 

 
 [41] 
 
D. M. Sheen, D. L. McMakin, and T. E. Hall, Chapter 9 - Detection of
Explosives by Millimeter-wave Imaging . Elsevier, 2007, pp. 237 – 277.

 

 
 [42] 
 
K. A. Smith, C. Csech, D. Murdoch, and G. Shaker, “Gesture Recognition Using
mm-Wave Sensor for Human-Car Interface,” IEEE Sensors Letters ,
vol. 2, no. 2, pp. 1–4, 2018.

 

 
 [43] 
 
C. Liu, Y. Li, D. Ao, and H. Tian, “Spectrum-based Hand Gesture Recognition
using Millimeter-wave Radar Parameter Measurements,” IEEE Access ,
vol. 7, 2017.

 

 
 [44] 
 
K. Wang, “Unifying Obstacle Detection, Recognition, and Fusion based on
Millimeter wave Radar and RGB-depth Sensors for the Visually Impaired,”
 AIP Review of Scientific Instruments , vol. 90, no. 4, 2019.

 

 
 [45] 
 
J. Lien, N. Gillian, M. E. Karagozler, P. Amihood, C. Schwesig, E. Olson, and
H. Raja, “Soli : Ubiquitous Gesture Sensing with Millimeter Wave Radar,”
 ACM Trans. Graphics , vol. 35, no. 4, pp. 1–19, Jul. 2016.

 

 
 [46] 
 
P. Zhao, C. X. Lu, J. Wang, C. Chen, W. Wang, N. Trigoni, and A. Markham,
“mID : Tracking and Identifying People with Millimeter Wave Radar,” in
 Proc. of 15th Intl. Conf. DCOSS . IEEE, 2019, pp. 33–40.

 

 
 [47] 
 
A. D. Singh and L. Garcia, “RadHAR : Human Activity Recognition from Point
Clouds Generated through a Millimeter-wave Radar,” in Proc. 3rd ACM
Workshop on Millimeter-wave Networks and Sensing Sys. , 2019, pp. 51–56.

 

 
 [48] 
 
R. Zhang and S. Cao, “Real-Time Human Motion Behavior Detection via CNN Using
mmWave Radar,” IEEE Sensors Letters , vol. 3, no. 2, pp. 1–4, 2018.

 

 
 [49] 
 
K. Diederichs, A. Qiu, and G. Shaker, “Wireless Biometric Individual
Identification Utilizing Millimeter Waves,” IEEE Sensors Letters ,
vol. 1, no. 1, pp. 1–4, 2017.

 

 
 [50] 
 
M. Alizadeh, H. Abedi, and G. Shaker, “Low-cost Low-power In-vehicle Occupant
Detection with mm-Wave FMCW Radar,” in Proc. of IEEE Sensors
Conf. , 2019, pp. 2–5.

 

 
 [51] 
 
X. Wang, L. Xu, H. Sun, J. Xin, and N. Zheng, “On-Road Vehicle Detection and
Tracking Using MMW Radar and Monovision Fusion,” IEEE Trans.
Intelligent Transportation Sys. , vol. 17, no. 7, pp. 2075–2084, 2016.

 

 
 [52] 
 
G. Zhai, C. Wu, and Y. Wang, “Millimeter Wave Radar Target Tracking Based on
Adaptive Kalman Filter,” in Proc. IEEE Symp. on Intelligent
Vehicles . IEEE, 2018, pp. 453–458.

 

 
 [53] 
 
F. Jin, R. Zhang, A. Sengupta, S. Cao, S. Hariri, N. K. Agarwal, and S. K.
Agarwal, “Multiple Patients Behavior Detection in Real-time using mmWave
Radar and Deep CNNs,” in Proc. IEEE Radar Conf. , 2019.

 

 
 [54] 
 
O. Boric-Lubecke, J. Lin, V. M. Lubecke, A. Host-Madsen, and T. Sizer,
“Microwave and Millimeter-wave Doppler Radar Heart Sensing,” in
 Proc. SPIE on Radar Sensor Technology XI , vol. 6547, 2007, article
65470C.

 

 
 [55] 
 
Texas Instruments. (2020) People Tracking and Counting Reference Design
Using mmWave Radar Sensor. [Online]. Available:
 https://www.ti.com/tool/TIDEP-01000 

 

 
 [56] 
 
A. Patra, P. Geuer, A. Munari, and P. Mähönen, “Mm-wave Radar
based Gesture Recognition: Development and Rvaluation of a Low-power,
Low-complexity System,” in Proc. Annual Intl. Conf. on MobiCom ,
2018, pp. 51–56.

 

 
 [57] 
 
M. Alizadeh, G. Shaker, J. C. M. D. Almeida, P. P. Morita, and
S. Safavi-Naeini, “Remote monitoring of human vital signs using mm-Wave
FMCW Radar,” IEEE Access , vol. 7, pp. 54 958–54 968, 2019.

 

 
 [58] 
 
T. Horiuchi, J. Konishi, H. Yamada, and S. Muramatsu, “Indoor Human Tracking
with Millimeter-Wave Minimum Redundancy MIMO Radar,” in Proc. Intl
Symp. on Antennas and Propagation , 2019, pp. 2–3.

 

 
 [59] 
 
S. Bakhtiari, T. W. Elmer, N. M. Cox, N. Gopalsami, A. C. Raptis, S. Liao,
I. Mikhelson, and A. V. Sahakian, “Compact Millimeter-wave Sensor for
Remote Monitoring of Vital Signs,” IEEE Trans. on Instrumentation and
Measurement , vol. 61, no. 3, pp. 830–841, 2012.

 

 
 [60] 
 
D. T. Petkie, C. Benton, and E. Bryan, “Millimeter-wave Radar for Vital Signs
Sensing,” in Proc. SPIE on Radar Sensor Technology XIII , vol. 7308,
2009, article 73080A.

 

 
 [61] 
 
G. Zhai, C. Wu, and Y. Wang, “Target Tracking based on Millimeter Wave Radar
in Complex Scenes,” Intl. Journal of Performability Engineering ,
vol. 14, no. 2, pp. 232–244, 2018.

 

 
 [62] 
 
A. Ahmad, J. C. Roh, D. Wang, and A. Dubey, “Vital Signs Monitoring of
Multiple People using a FMCW Millimeter-wave Sensor,” in Proc. IEEE
Radar Conf. , no. 4. IEEE, 2018, pp.
1450–1455.

 

 
 [63] 
 
S. Hazra and A. Santra, “Robust Gesture Recognition Using Millimetric-Wave
Radar System,” IEEE Sensors Letters , vol. 2, no. 4, pp. 1–4, 2018.

 

 
 [64] 
 
M. Raja, Z. Vali, S. Palipana, D. G. Michelson, and S. Sigg, “3D Head Motion
Detection Using Millimeter-Wave Doppler Radar,” IEEE Access , vol. 8,
pp. 32 321–32 331, 2020.

 

 
 [65] 
 
S. Diebold, S. Ayhan, S. Scherr, H. Massler, A. Tessmann, A. Leuther,
O. Ambacher, T. Zwick, and I. Kallfass, “A W-band MMIC radar system for
remote detection of vital signs,” Infrared, Millimeter, and Terahertz
Waves , vol. 33, no. 12, pp. 1250–1267, 2012.

 

 
 [66] 
 
S. M. Kwon, S. Yang, J. Liu, X. Yang, W. Saleh, S. Patel,
C. Mathews, and Y. Chen, “Demo: Hands-Free Human Activity Recognition
Using Millimeter-Wave Sensors,” in IEEE International Symposium on
Dynamic Spectrum Access Networks (DySPAN) , 2019, pp. 1–2.

 

 
 [67] 
 
W. Huang, Z. Zhang, W. Li, and J. Tian, “Moving Object Tracking based on
Millimeter-wave Radar and Vision Sensor,” Applied Science and
Engineering , vol. 21, no. 4, pp. 609–614, 2018.

 

 
 [68] 
 
B. Kapilevich and M. Einat, “Detecting Hidden Objects on Human Body using
Active Millimeter Wave Sensor,” IEEE Sensors , vol. 10, no. 11, pp.
1746–1752, 2010.

 

 
 [69] 
 
E. Al-Masri and M. Momin, “Detecting Heart Rate Variability using
Millimeter-Wave Radar Technology,” in Proc. IEEE Intl. Conf. on Big
Data . IEEE, 2019, pp. 5282–5284.

 

 
 [70] 
 
M. Z. Ikram, A. Ahmad, and D. Wang, “High-accuracy Distance Measurement using
Millimeter-wave Radar,” in Proc. IEEE Radar Conf.  IEEE, 2018, pp. 1296–1300.

 

 
 [71] 
 
A. A. Belyaev, I. O. Frolov, T. A. Suanov, and D. O. Trots, “Object Detection
in an Urban Environment Using 77GHz Radar,” in Proc. Radiation and
Scattering of Electromagnetic Waves Conf. , 2019, pp. 436–439.

 

 
 [72] 
 
T. Akita and S. Mita, “Object Tracking and Classification Using
Millimeter-Wave Radar Based on LSTM,” in Proc. IEEE Symp. Intelligent
Transportation Sys.  IEEE, 2019, pp.
1110–1115.

 

 
 [73] 
 
S. Matsuguma and A. Kajiwara, “Bathroom Accident Detection with 79GHz-band
Millimeter wave Sensor,” in Proc. IEEE Conf. on Sensors Applications
Symp.  IEEE, 2019, pp. 1–5.

 

 
 [74] 
 
Ï. Ünal and S. Eker, “Investigations on Millimeter wave Detection
of Power Lines from a Safe Distance,” in Proc. 10th Intl. Conf. on
Electrical and Electronics Engineering , 2018, pp. 964–967.

 

 
 [75] 
 
S. Björklund, H. Petersson, A. Nezirovic, M. B. Guldogan, and
F. Gustafsson, “Millimeter-wave Radar Micro-Doppler Signatures of Human
Motion,” in Proc. 12th Intl. Radar Symp. , 2011, pp. 167–174.

 

 
 [76] 
 
Y. Sun, R. Hang, Z. Li, M. Jin, and K. Xu, “Privacy-Preserving Fall Detection
with Deep Learning on mmWave Radar Signal,” in Proc. IEEE Intl. Conf.
on Visual Comm. and Image , 2019.

 

 
 [77] 
 
P. Kaushik, “Radar as a Security Measure - Real time Neural Model based Human
Detection and Behaviour Classification,” in Proc. 7th IEEE Conf. on
Signal and Information Processing , 2019.

 

 
 [78] 
 
S. Björklund, T. Johansson, and H. Petersson, “Evaluation of a
Micro-Doppler Classification Method on mm-Wave Data,” in Proc. IEEE
National Radar Conf. , 2012, pp. 0934–0939.

 

 
 [79] 
 
D. P. Fairchild and R. M. Narayanan, “Classification of Human Motions
using Empirical Mode Decomposition of Human micro-Doppler Signatures,”
 IET Journal on Radar, Sonar Navigation , vol. 8, no. 5, pp.
425–434, 2014.

 

 
 [80] 
 
M. Alloulah and H. Huang, “Future Millimeter-Wave Indoor Systems: A
Blueprint for Joint Communication and Sensing,” IEEE Computer ,
vol. 52, no. 7, pp. 16–24, 2019.

 

 
 [81] 
 
A. Santra, R. V. Ulaganathan, and T. Finke, “Short-Range
Millimetric-Wave Radar System for Occupancy Sensing Application,”
 IEEE Sensors Letters , vol. 2, no. 3, pp. 1–4, 2018.

 

 
 [82] 
 
T. J. Kao and J. Lin, “Vital Sign Detection using 60-GHz Doppler Radar
System,” in Proc. IEEE Intl. Wireless Symp. , 2013, pp. 1–4.

 

 
 [83] 
 
W. Baoshuai and Z. Wei, “FOD Detection based on Millimeter wave Radar
using Higher Order Statistics,” in Proc. IEEE Intl. Conf. on Signal
Processing, Communications and Computing , 2017, pp. 1–4.

 

 
 [84] 
 
W. Baoshuai, L. Jianghong, Z. Xiaoliang, and H. Minjue, “A Novel
Hierarchical Foreign Object Debris Detection Method for Millimeter wave
Radar,” in Proc. Intl. Applied Computational Electromagnetics Society
Symp. , 2017, pp. 1–2.

 

 
 [85] 
 
K. Mazouni, A. Zeitler, J. Lanteri, C. Pichot, J. . Dauvignac,
C. Migliaccio, N. Yonemoto, A. Kohmura, and S. Futatsumori, “76.5
GHz Millimeter-wave Radar for Foreign Object Debris Detection on Airport
Runways,” in Proc. 8th IEEE Radar Conf. , 2011, pp. 222–225.

 

 
 [86] 
 
P. Feil, W. Menzel, T. P. Nguyen, C. Pichot, and C. Migliaccio,
“Foreign Objects Debris Detection (FOD) on Airport Runways Using a
Broadband 78 GHz Sensor,” in Proc. 38th European Microwave Conf. ,
2008, pp. 1608–1611.

 

 
 [87] 
 
B. D. Pollard and G. Sadowy, “Next Generation Millimeter-wave Radar for
Safe Planetary Landing,” in Proc. IEEE Aerospace Conf. , 2005, pp.
1213–1219.

 

 
 [88] 
 
R. Battaglia, M. Ferri, V. Dainelli, F. Sarullo, and M. Demeo,
“Warden: W-band Advanced Radar for Debris Early Notification form ISS,”
in Proc. IEEE Aerospace Conf. , vol. 1, 2003, pp. 1–81.

 

 
 [89] 
 
Y. Almalioglu, M. Turan, C. X. Lu, N. Trigoni, and A. Markham, “Milli-RIO:
Ego-Motion Estimation with Low-Cost Millimetre-Wave Radar,” 2019,
arXiv:1909.05774 [eess.SP].

 

 
 [90] 
 
X. Tang, X. Wu, S. B. Yeap, R. Luo, T. Dai, and L. Huang,
“Experimental Results of Target Classification Using mm Wave Corner Radar
Sensors,” in Proc. Asia-Pacific Microwave Conf. , 2018, pp. 842–844.

 

 
 [91] 
 
Z. Flintoff, B. Johnston, and M. Liarokapis, “Single-Grasp, Model-Free
Object Classification using a Hyper-Adaptive Hand, Google Soli, and Tactile
Sensors,” in Proc. IEEE/RSJ Intl. Conf. on Intelligent Robots and
Sys. , 2018, pp. 1943–1950.

 

 
 [92] 
 
A. Angelov, A. Robertson, R. Murray-Smith, and F. Fioranelli, “Practical
Classification of Different Moving Targets using Automotive Radar and Deep
Neural Networks,” IET Journal on Radar, Sonar Navigation ,
vol. 12, pp. 1082–1089, 2018.

 

 
 [93] 
 
X. Gao, G. Xing, S. Roy, and H. Liu, “Experiments with mmWave
Automotive Radar Test-bed,” in Proc. 53rd Asilomar Conf. on Signals,
Sys., and Computers , 2019, pp. 1–6.

 

 
 [94] 
 
A. Antonucci, M. Corrà, A. Ferrari, D. Fontanelli, E. Fusari,
D. Macii, and L. Palopoli, “Performance Analysis of a 60-GHz Radar for
Indoor Positioning and Tracking,” in Proc. Intl. Conf. on Indoor
Positioning and Indoor Navigation , 2019, pp. 1–7.

 

 
 [95] 
 
R. Zhang and S. Cao, “Extending Reliability of mmWave Radar Tracking and
Detection via Fusion With Camera,” IEEE Access , vol. 7, pp.
137 065–137 079, 2019.

 

 
 [96] 
 
Texas Instruments. (2019, May) TIDEP-01018: Automated Doors Reference Design
Using mmWave Sensors. [Online]. Available:
 http://www.ti.com/lit/pdf/tiduer1 

 

 
 [97] 
 
——. (2018, Aug.) TIDEP-01003: Zone Occupancy Detection Reference Design
Using mmWave Sensor. [Online]. Available:
 http://www.ti.com/lit/pdf/tiduea7 

 

 
 [98] 
 
——. (2018, Apr.) TIDEP-01001: Vehicle Occupant Detection Reference
Design. [Online]. Available: http://www.ti.com/lit/pdf/tidue95 

 

 
 [99] 
 
M. Kishida, K. Ohguchi, and M. Shono, “79 GHz-band high-resolution
millimeter-wave radar,” Fujitsu Sci. Tech. , vol. 51, no. 4, pp.
55–59, Oct. 2015.

 

 
 [100] 
 
A. Etinger, N. Balal, B. Litvak, M. Einat, B. Kapilevich, and
Y. Pinhasi, “Non-Imaging MM-Wave FMCW Sensor for Pedestrian Detection,”
 IEEE Sensors , vol. 14, no. 4, pp. 1232–1237, 2014.

 

 
 [101] 
 
M. Leonardi, E. G. Piracci, and V. Fastella, “W-Band Multi-Radar
processing for Airport Foreign Object Debris and Humans Detection,” in
 Proc. European Microwave Conf. in Central Europe , 2019, pp. 285–288.

 

 
 [102] 
 
K. A. Gallagher and R. M. Narayanan, “Human Detection and Ranging at Long
Range and Through Light Foliage using a W-band Noise Radar with an Embedded
Tone,” in Proc. SPIE on Radar Sensor Technology XVII , K. I. Ranney
and A. Doerry, Eds., vol. 8714, 2013, pp. 1 – 12.

 

 
 [103] 
 
A. Sengupta, F. Jin, R. Zhang, and S. Cao, “mm-Pose: Real-Time Human
Skeletal Posture Estimation Using mmWave Radars and CNNs,” IEEE
Sensors Journal , vol. 20, no. 17, pp. 10 032–10 044, 2020.

 

 
 [104] 
 
S. Caorsi and C. Lenzi, “Can a MM-wave Ultra-wideband ANN-based Radar
Data Processing Approach be used for Breast Cancer Detection?” in
 Proc. Intl. Conf. on Electromagnetics in Advanced Applications , 2017,
pp. 1236–1239.

 

 
 [105] 
 
I. V. Mikhelson, P. Lee, S. Bakhtiari, T. W. Elmer, A. K.
Katsaggelos, and A. V. Sahakian, “Noncontact Millimeter-Wave Real-Time
Detection and Tracking of Heart Rate on an Ambulatory Subject,” IEEE
Trans. on Information Technology in Biomedicine , vol. 16, no. 5, pp.
927–934, 2012.

 

 
 [106] 
 
I. Walterscheid, O. Biallawons, and P. Berens, “Contactless Respiration
and Heartbeat Monitoring of Multiple People Using a 2-D Imaging Radar,” in
 Proc. 41st IEEE Annual Intl. Conf. on Engineering in Medicine and
Biology Society , 2019, pp. 3720–3725.

 

 
 [107] 
 
S. Wang, A. Pohl, T. Jaeschke, M. Czaplik, M. Köny, S. Leonhardt,
and N. Pohl, “A Novel Ultra-wideband 80GHz FMCW Radar System for
Contactless Monitoring of Vital Signs,” in Proc. 37th IEEE Annual
Intl. Conf. on Engineering in Medicine and Biology Society , 2015, pp.
4978–4981.

 

 
 [108] 
 
M. Alizadeh, G. Shaker, and S. Safavi-Naeini, “Remote Heart Rate
Sensing with mm-wave Radar,” in Proc. 18th Intl. Symp. on Antenna
Technology and Applied Electromagnetics , 2018, pp. 1–2.

 

 
 [109] 
 
H. Chuang, H. Kuo, F. Lin, T. Huang, C. Kuo, and Y. Ou, “60-GHz
Millimeter-Wave Life Detection System (MLDS) for Noncontact Human
Vital-Signal Monitoring,” IEEE Sensors , vol. 12, no. 3, pp.
602–609, 2012.

 

 
 [110] 
 
S. Li, Y. Tian, G. Lu, Y. Zhang, H. J. Xue, J.-Q. Wang, and X.-J. Jing, “A
New Kind of Non-acoustic Speech Acquisition Method based on Millimeter wave
Radar,” Progress In Electromagnetics Research , vol. 130, pp. 17–40,
2012.

 

 
 [111] 
 
Z.-W. Li, “Millimeter Wave Radar for Detecting the Speech Signal
Applications,” Infrared and Millimeter Waves , vol. 17, no. 12, pp.
2175–2183, Dec. 1996.

 

 
 [112] 
 
M. Jiao, G. Lu, X. Jing, S. Li, Y. Li, and J. Wang, “A Novel Radar Sensor for
the Non-contact Detection of Speech Signals,” MDPI Sensors , vol. 10,
no. 5, pp. 4622–4633, 2010.

 

 
 [113] 
 
S. Li, Y. Tian, G. Lu, Y. Zhang, H. Lv, X. Yu, H. Xue, H. Zhang, J. Wang, and
X. Jing, “A 94-GHz Millimeter-wave Sensor for Speech Signal Acquisition,”
 MDPI Sensors , vol. 13, no. 11, pp. 14 248–14 260, Oct. 2013.

 

 
 [114] 
 
Texas Instruments. (2019, Sep.) TIDEP-0090: Traffic Monitoring Object
Detection and Tracking Reference Design Using Single-Chip mmWave Radar
Sensor. [Online]. Available: http://www.ti.com/lit/pdf/tidud31 

 

 
 [115] 
 
S. Dogru, R. Baptista, and L. Marques, “Tracking Drones with Drones Using
Millimeter Wave Radar,” in Proc. 4th Iberian Robotics Conf. , M. F.
Silva, J. Luís Lima, L. P. Reis, A. Sanfeliu, and D. Tardioli, Eds. Springer, 2020, pp. 392–402.

 

 
 [116] 
 
A. K. Singh and Y. H. Kim, “Accurate Measurement of Drone’s Blade Length
and Rotation Rate using Pattern Analysis with W-band Radar,” IET
Journal of Electronics Letters , vol. 54, no. 8, pp. 523–525, 2018.

 

 
 [117] 
 
M. Caris, W. Johannes, S. Sieger, V. Port, and S. Stanko,
“Detection of Small UAS with W-band Radar,” in Proc. 18th Intl.
Radar Symp. , 2017, pp. 1–6.

 

 
 [118] 
 
S. Haefner, M. Roeding, G. Sommerkorn, R. Mueller, R. S. Thomae,
G. D. Galdo, and J. Goerlich, “Contribution to Drone Detection by
Exploiting Parameter Estimation for a Prototype mm-Wave Radar System,” in
 Proc. 22nd Intl. ITG Workshop on Smart Antennas , 2018, pp. 1–8.

 

 
 [119] 
 
S. Rahman and D. A. Robertson, “Radar Micro-Doppler Signatures of Drones and
Birds at K-band and W-band,” Scientific Reports , vol. 8, no. 1, Nov.
2018, article 17396.

 

 
 [120] 
 
S. Chang, Y. Zhang, F. Zhang, X. Zhao, S. Huang, Z. Feng, and Z. Wei,
“Spatial Attention Fusion for Obstacle Detection Using MmWave Radar and
Vision Sensor,” MDPI Sensors , vol. 20, no. 4, 2020, article 956.

 

 
 [121] 
 
Texas Instruments. (2017, Jun.) TIDEP-0094: 80-m Range Object Detection With
IWR1642 mmWave Sensor Reference Design. [Online]. Available:
 http://www.ti.com/lit/pdf/tidud93 

 

 
 [122] 
 
K. Stasiak, M. Ciesielski, P. Samczyński, D. Gromek, and K. Kulpa,
“Preliminary Results of Drone’s Propellers Detection Using K-band and
mm-Wave FMCW Radar,” in Proc. 20th Intl. Radar Symp. , 2019, pp.
1–7.

 

 
 [123] 
 
P. Janakaraj, K. Jakkala, A. Bhuyan, Z. Sun, P. Wang, and M. Lee,
“STAR: Simultaneous Tracking and Recognition through Millimeter Waves and
Deep Learning,” in Proc. 12th IFIP Wireless and Mobile Networking
Conf. , 2019, pp. 211–218.

 

 
 [124] 
 
Y. Liu, Y. Wang, H. Liu, A. Zhou, J. Liu, and N. Yang, “Long-Range Gesture
Recognition Using Millimeter Wave Radar,” 2020, arXiv:2002.02591 [cs.HC].

 

 
 [125] 
 
S. Wang, J. Song, J. Lien, I. Poupyrev, and O. Hilliges, “Interacting with
Soli: Exploring Fine-Grained Dynamic Gesture Recognition in the
Radio-Frequency Spectrum,” in Proc. 29th Annual Symp. on User
Interface Software and Tech.  ACM,
2016, p. 851–860.

 

 
 [126] 
 
S. Hazra and A. Santra, “Radar Gesture Recognition System in Presence of
Interference using Self-Attention Neural Network,” in Proc. 18th IEEE
Intl. Conf. On Machine Learning And Applications , 2019, pp. 1409–1414.

 

 
 [127] 
 
——, “Short-Range Radar-Based Gesture Recognition System Using 3D CNN With
Triplet Loss,” IEEE Access , vol. 7, pp. 125 623–125 633, 2019.

 

 
 [128] 
 
M. Arsalan and A. Santra, “Character Recognition in Air-Writing Based on
Network of Radars for Human-Machine Interface,” IEEE Sensors ,
vol. 19, no. 19, pp. 8855–8864, 2019.

 

 
 [129] 
 
Y. Sun, T. Fei, F. Schliep, and N. Pohl, “Gesture Classification with
Handcrafted Micro-Doppler Features using a FMCW Radar,” in Proc. IEEE
MTTS Conf. on Microwaves for Intelligent Mobility , 2018, pp. 1–4.

 

 
 [130] 
 
Y. Sun, T. Fei, S. Gao, and N. Pohl, “Automatic Radar-based Gesture
Detection and Classification via a Region-based Deep Convolutional Neural
Network,” in Proc. IEEE Intl. Conf. on Acoustics, Speech and Signal
Processing , 2019, pp. 4300–4304.

 

 
 [131] 
 
J. Wang, X. Geng, and S. Wei, “Airport Runway FOD Detection System
Based on 77GHz Millimeter Wave Radar Sensor,” in Proc. IEEE Intl.
Conf. on Integrated Circuits, Technologies and Applications , 2019, pp.
140–143.

 

 
 [132] 
 
S. H. Cen and P. Newman, “Precise Ego-Motion Estimation with
Millimeter-Wave Radar Under Diverse and Challenging Conditions,” in
 Proc. IEEE Intl. Conf. on Robotics and Automation , 2018, pp.
6045–6052.

 

 
 [133] 
 
D. Kellner, M. Barjenbruch, J. Klappstein, J. Dickmann, and
K. Dietmayer, “Instantaneous Ego-motion Estimation using Doppler
Radar,” in Proc. 16th Intl. IEEE Conf. Intelligent Transportation
Sys. , 2013, pp. 869–874.

 

 
 [134] 
 
M. Rapp, M. Barjenbruch, M. Hahn, J. Dickmann, and K. Dietmayer,
“Probabilistic ego-motion estimation using multiple automotive radar
sensors,” Robotics and Autonomous Sys. , vol. 89, pp. 136 – 146,
2017.

 

 
 [135] 
 
E. Ward and J. Folkesson, “Vehicle Localization with Low Cost Radar
Sensors,” in Proc. IEEE Symp. on Intelligent Vehicles , 2016, pp.
864–870.

 

 
 [136] 
 
F. Schuster, M. Wörner, C. G. Keller, M. Haueis, and C. Curio,
“Robust Localization based on Radar Signal Clustering,” in Proc.
IEEE Symp. on Intelligent Vehicles , 2016, pp. 839–844.

 

 
 [137] 
 
H.-S. Yeo, G. Flamich, P. Schrempf, D. Harris-Birtill, and A. Quigley,
“RadarCat: Radar Categorization for Input Interaction,” in Proc.
29th Annual Symp. on User Interface Software and Tech.  ACM, 2016, p. 833–841.

 

 
 [138] 
 
Z. Li, Z. Lei, A. Yan, E. Solovey, and K. Pahlavan, “ThuMouse: A
Micro-gesture Cursor Input through mmWave Radar-based Interaction,” in
 Proc. IEEE Intl. Conf. on Consumer Electronics , 2020, pp. 1–9.

 

 
 [139] 
 
H.-S. Yeo, R. Minami, K. Rodriguez, G. Shaker, and A. Quigley, “Exploring
Tangible Interactions with Radar Sensing,” Proc. of ACM on
Interactive, Mobile, Wearable and Ubiquitous Technologies , vol. 2, no. 4,
Dec. 2018.

 

 
 [140] 
 
K. Ammar, O. Ben Haj Belkacem, and R. Bouallegue, “DSSS Transmission
Technique to Joint Radar Sensing and Wireless Communications in V2V
System,” in Proc. Workshop on Web, Artificial Intelligence and
Network Applications , L. Barolli, F. Amato, F. Moscato, T. Enokido, and
M. Takizawa, Eds. Springer, 2020, pp.
376–385.

 

 
 [141] 
 
L. S. Lu Shaobei, “Target Detection and Recognition Based on Active
Millimeter-Wave Imaging System,” in Proc. 2nd IEEE Intl. Conf. on
Electronics Technology (ICET) , no. 2. IEEE, 2019, pp. 74–77.

 

 
 [142] 
 
S. Agarwal, D. Singh, and N. P. Pathak, “Active Millimeter wave Radar
System for Non-destructive, Non-invasive Underline Fault Detection and
Multilayer Material Analysis,” in Proc. IEEE Intl. Microwave and RF
Conf. , 2014, pp. 369–372.

 

 
 [143] 
 
T. Gu, Z. Fang, Z. Yang, P. Hu, and P. Mohapatra, “MMSense: Multi-person
Detection and Identification via mmWave Sensing,” in Proc. Annual
Intl. Conf. on MobiCom , 2019, pp. 45–50.

 

 
 [144] 
 
Z. Yang, P. H. Pathak, Y. Zeng, X. Liran, and P. Mohapatra, “Monitoring Vital
Signs using Millimeter Wave,” in Proc. Intl. Symp. on Mobile Ad Hoc
Networking and Computing , 2016, pp. 211–220.

 

 
 [145] 
 
J. Palacios, G. Bielsa, P. Casari, and J. Widmer, “Single- and
Multiple-access Point Indoor Localization for Millimeter-wave Networks,”
 IEEE Trans. on Wireless Communications , vol. 18, no. 3, pp.
1927–1942, 2019.

 

 
 [146] 
 
M. T. Ortiz, H. Groll, E. Zochmann, and C. F. Mecklenbraucker, “Vehicle
Tracking through Vision-Millimeter Wave Doppler Shift Fusion,” in
 Proc. 9th IEEE-APS Topical Conf. on Antennas and Propagation in
Wireless Communications , 2019, pp. 359–362.

 

 
 [147] 
 
Z. Yang, P. H. Pathak, Y. Zeng, X. Liran, and P. Mohapatra, “Vital Sign and
Sleep Monitoring using Millimeter Wave,” ACM Trans. Sensor Networks ,
vol. 13, no. 2, pp. 1–32, 2017.

 

 
 [148] 
 
Y. Zeng, P. H. Pathak, Z. Yang, and P. Mohapatra, “Poster Abstract:
Human Tracking and Activity Monitoring Using 60 GHz mmWave,” in Proc.
15th ACM/IEEE Intl. Conf. on Information Processing in Sensor Networks ,
2016, pp. 1–2.

 

 
 [149] 
 
K. Watabe, K. Shimizu, K. Mizuno, and M. Yoneyama, “Millimeter-wave
Imaging using Neural Networks for Object Recognition,” in IEEE MTT-S
Intl. Microwave Symp. Digest , vol. 2, 1996, pp. 1135–1138.

 

 
 [150] 
 
A. Patra, L. Simic, and M. Petrova, “mmRTI: Radio Tomographic Imaging
using Highly-directional Millimeter-wave Devices for Accurate and Robust
Indoor Localization,” in Proc. 28th IEEE Annual Intl. Symp. on
Personal, Indoor, and Mobile Radio Communications , 2017, pp. 1–7.

 

 
 [151] 
 
S. Oka, S. Mochizuki, H. Togo, and N. Kukutsu, “Inspection of Concrete
Structures using Millimeter-wave Imaging Technology,” NTT Technical
review , vol. 7, no. 3, 2009. [Online]. Available:
 https://www.ntt-review.jp/archive/ntttechnical.php?contents=ntr200903sf4.pdf 

 

 
 [152] 
 
J. O. Schrattenecker, S. Schuster, A. Haderer, G. Reinthaler, and
A. Stelzer, “Accuracy Limits of a Seam-tracking Algorithm for Microwave
Systems at mm-wave Frequencies,” in Proc. 21st European Signal
Processing Conf. , 2013, pp. 1–5.

 

 
 [153] 
 
Y. Zhu, Y. Zhu, B. Y. Zhao, and H. Zheng, “Reusing 60GHz Radios for Mobile
Radar Imaging,” in Proc. 21st Annual Intl. Conf. on MobiCom . ACM, 2015, p. 103–116.

 

 
 [154] 
 
M. Aladsani, A. Alkhateeb, and G. C. Trichopoulos, “Leveraging mmWave
Imaging and Communications for Simultaneous Localization and Mapping,” in
 Proc. IEEE Intl. Conf. on Acoustics, Speech and Signal Processing ,
2019, pp. 4539–4543.

 

 
 [155] 
 
H. Ajorloo, C. J. Sreenan, A. Loch, and J. Widmer, “On the Feasibility of
Using IEEE 802.11ad MmWave for Accurate Object Detection,” in Proc.
34th ACM/SIGAPP Symp. on Applied Computing . Association for Computing Machinery, 2019, p. 2406–2413.

 

 
 [156] 
 
L. Wang, “3D Holographic Millimeter-Wave Imaging for Concealed Metallic
Forging Objects Detection,” in Emerging Microwave Technologies in
Industrial, Agricultural, Medical and Food Processing . IntechOpen, Jul. 2018, pp. 125–139.

 

 
 [157] 
 
C. Li, J. Wang, D. Rodriguez, A. Mishra, Z. Peng, and Y. Li, “Portable
Doppler/FSK/FMCW Radar Systems for Life Activity Sensing and Human
Localization,” in Proc. 14th Intl. Conf. on Advanced Technologies,
Sys. and Services in Telecommunications , 2019, pp. 83–93.

 

 
 [158] 
 
W. Jiang, C. Miao, F. Ma, S. Yao, Y. Wang, Y. Yuan, H. Xue, C. Song, X. Ma,
D. Koutsonikolas, W. Xu, and L. Su, “Towards Environment Independent Device
Free Human Activity Recognition,” in Proc. 24th Annual Intl. Conf. on
MobiCom , 2018, p. 289–304.

 

 
 [159] 
 
T. Wei and X. Zhang, “MTrack: High-precision Passive Tracking using
Millimeter Wave Radios,” in Proc. Annual Intl. Conf. on MobiCom ,
2015, pp. 117–129.

 

 
 [160] 
 
A. Zhou and Y. Fan, “Autonomous Environment Mapping Using Commodity
Millimeter-wave Network Device,” in Proc. IEEE Conf. on Computer
Communications . IEEE, 2019, pp.
1126–1134.

 

 
 [161] 
 
B. D. Pollard, G. Sadowy, D. Moller, and E. Rodriguez, “A
Millimeter-wave Phased Array Radar for Hazard Detection and Avoidance on
Planetary Landers,” in Proc. IEEE Aerospace Conf. Proceedings ,
vol. 2, 2003, pp. 1115 – 1122.

 

 
 [162] 
 
A. P. Toda and F. De Flaviis, “Mm-wave Motion Tracking System using
Beamforming Antennas,” in Proc. IEEE Antennas and Propagation Society
Intl. Symp. , 2014, pp. 105–106.

 

 
 [163] 
 
C. Wolff. (1998, Nov.) Radar Basics. [Online]. Available:
 https://www.radartutorial.eu/index.en.html 

 

 
 [164] 
 
H.-C. Chen, T. Chiu, and C.-L. Hsu, “Design of Series-Fed Bandwidth-Enhanced
Microstrip Antenna Array for Millimetre-Wave Beamforming Applications,”
 International Journal of Antennas and Propagation , vol. 2019, Jun
2019, article 3857964.

 

 
 [165] 
 
L. J. Ippolito, “Radio propagation for space communications systems,”
 Proc. of the IEEE , vol. 69, no. 6, pp. 697–727, 1981.

 

 
 [166] 
 
S. Rao. Introduction to mmwave Sensing: FMCW Radars. Texas Instruments.
[Online]. Available:
 https://training.ti.com/sites/default/files/docs/mmwaveSensing-FMCW-offlineviewing_2.pdf 

 

 
 [167] 
 
NXP Semiconductors. (2019) TEF810X 77GHz Automotive Radar Transceiver.
[Online]. Available:
 https://www.nxp.com/docs/en/fact-sheet/TEF810XFS.pdf 

 

 
 [168] 
 
M. Constapel, M. Cimdins, and H. Hellbrück, “A Practical Toolbox for
Getting Started with mmWave FMCW Radar Sensors,” in Proc. 4th KuVS/GI
Expert Talk on Localization , 2019, p. 2–4.

 

 
 [169] 
 
S. Suleymanov, “Design and Implementation of an FMCW Radar Signal Processing
Module for Automotive Applications,” Master’s thesis, University of Twente,
Aug. 2016. [Online]. Available: http://essay.utwente.nl/70986/ 

 

 
 [170] 
 
A. M. Niknejad. (2005) Introduction to Mixers. [Online]. Available:
 http://rfic.eecs.berkeley.edu/~niknejad/ee142_fa05lects/pdf/lect15.pdf 

 

 
 [171] 
 
M. Parker. (2011, May) Radar Basics – Part 2: Pulse Doppler Radar. Altera
Corporation. Published by EE Times. [Online]. Available:
 https://www.eetimes.com/radar-basics-part-2-pulse-doppler-radar/ 

 

 
 [172] 
 
S. Rao. (2018, Jul.) MIMO Radar. Texas Instruments. [Online]. Available:
 http://www.ti.com/lit/an/swra554a/swra554a.pdf 

 

 
 [173] 
 
S. Sharenson, “Angle Estimation Accuracy with a Monopulse Radar in the
Search Mode,” IRE Trans. on Aerospace and Navigational Electronics ,
vol. ANE-9, no. 3, pp. 175–179, 1962.

 

 
 [174] 
 
Y. Ma and Y. Li, “Millimeter-wave InSAR Target Recognition with Deep
Convolutional Neural Network,” IEICE Trans. Information and Sys. ,
vol. E102D, no. 3, pp. 655–658, 2019.

 

 
 [175] 
 
C. Zech, A. Hülsmann, I. Kallfass, A. Tessmann, M. Zink, M. Schlechtweg,
A. Leuther, and O. Ambacher, “Active Millimeter-wave Imaging System for
Material Analysis and Object Detection,” in Proc. SPIE on Millimeter
Wave and Terahertz Sensors and Technology IV , vol. 8188, 2011, article
81880D.

 

 
 [176] 
 
C. Wang, J. Pei, M. Li, Y. Zhang, Y. Huang, and J. Yang, “Parking Information
Perception based on Automotive Millimeter wave SAR,” in Proc. IEEE
Radar Conf.  IEEE, 2019, pp. 1–6.

 

 
 [177] 
 
M. Li, Y. Zhang, R. Wang, J. Wu, Y. Huang, Y. Zhang, and J. Yang, “Parking
Space Information Monitoring by Millimeter Wave SAR Based on Unmanned Aerial
Vehicle,” in Proc. Intl. Symp. on Geoscience and Remote
Sensing . IEEE, 2019, pp. 9216–9219.

 

 
 [178] 
 
S. Pawliczek, R. Herschel, and N. Pohl, “3D Millimeter Wave Screening
for Metallic Surface Defect Detection,” in Proc. 16th European Radar
Conf. , 2019, pp. 113–116.

 

 
 [179] 
 
M. E. Yanik and M. Torlak, “Near-Field 2-D SAR Imaging by Millimeter-Wave
Radar for Concealed Item Detection,” in Proc. IEEE Radio and Wireless
Symp. , 2019, pp. 1–4.

 

 
 [180] 
 
D. Huston and D. Busuioc, Chapter 8 - Radar Technology: Radio Frequency,
Interferometric, Millimeter wave and Terahertz Sensors for Assessing and
Monitoring Civil Infrastructures . Woodhead Publishing, 2014, vol. 55, pp. 201 – 237.

 

 
 [181] 
 
T. Liu, Y. Zhao, Y. Wei, Y. Zhao, and S. Wei, “Concealed Object Detection for
Activate Millimeter wave Image,” IEEE Trans. on Industrial
Electronics , vol. 66, no. 12, pp. 9909–9917, 2019.

 

 
 [182] 
 
Y. K. Chan and V. C. Koo, “An Introduction to Synthetic Aperture Radar
(SAR),” Progress In Electromagnetics Research B , vol. 2, pp. 27–60,
2008.

 

 
 [183] 
 
A. Moreira. (2013) Synthetic Aperture Radar (SAR): Principles and
Applications. [Online]. Available:
 https://earth.esa.int/documents/10174/642943/6-LTC2013-SAR-Moreira.pdf 

 

 
 [184] 
 
O. Kanhere, S. Ju, Y. Xing, and T. S. Rappaport, “Map-assisted Millimeter
Wave Localization for Accurate Position Location,” in Proc. IEEE
Global Communications Conf. , 2019, pp. 1–6.

 

 
 [185] 
 
H. Kim, H. Wymeersch, N. Garcia, G. Seco-Granados, and S. Kim, “5G
mmWave Vehicular Tracking,” in Proc. 52nd Asilomar Conf. on Signals,
Sys., and Computers , 2018, pp. 541–547.

 

 
 [186] 
 
A. Yassin, Y. Nasser, A. Y. Al-Dubai, and M. Awad, “MOSAIC:
Simultaneous Localization and Environment Mapping Using mmWave Without
A-Priori Knowledge,” IEEE Access , vol. 6, pp. 68 932–68 947,
2018.

 

 
 [187] 
 
B. R. Mahafza, Introduction to Radar Analysis . CRC Press, 1998.

 

 
 [188] 
 
G. Péter, S. Zsolt, and A. Szilárd, Chapter 9 - Vehicle to
Infrastructure Interaction (V2I) . BME MOGI, 2014.

 

 
 [189] 
 
X. Wang, S. Gou, X. Wang, Y. Zhao, and L. Zhang, “Patch-Based Gaussian
Mixture Model for Concealed Object Detection in Millimeter-Wave images,” in
 Proc. TENCON IEEE Region 10 Annual Intl. Conf.  IEEE, 2018, pp. 2522–2527.

 

 
 [190] 
 
Y. Suzuki, Y. Sugiura, T. Shimamura, O. Isaji, and K. Hamada, “Model-Based
Vehicle Position Estimation Using Millimeter Wave Radar,” Future
Computing and Communication , vol. 8, no. 3, pp. 94–98, 2019.

 

 
 [191] 
 
K. Konishi and T. Sakamoto, “Automatic Tracking of Human Body using
Millimeter-wave Adaptive Array Radar for Noncontact Heart Rate
Measurement,” in Proc. Asia-Pacific Microwave Conf. , 2019, pp.
836–838.

 

 
 [192] 
 
H. Groll, E. Zöchmann, S. Pratschner, M. Lerch, D. Schützenhöfer,
M. Hofer, J. Blumenstein, S. Sangodoyin, T. Zemen, A. Prokeš,
A. F. Molisch, and S. Caban, “Sparsity in the Delay-Doppler Domain for
Measured 60 GHz Vehicle-to-Infrastructure Communication Channels,” in
 IEEE Intl. Conf. Communications Workshops , 2019, pp. 1–6.

 

 
 [193] 
 
K. Du, L. Zhang, W. Chen, G. Wan, and R. Fu, “Concealed Objects Detection
based on FWT in Active Millimeter-wave Images,” in Proc.7th SPINE
Intl. Conf. Electronics and Information Eng. , vol. 10322, 2016.

 

 
 [194] 
 
S. Wang, Z. Ye, and Y. Wang, “Real-time Dangerous Objects Detection in
Millimeter Wave Images,” in Proc. 10th Intl. Conf. on Digital Image
Processing , 2018.

 

 
 [195] 
 
W. Xing, J. Zhang, and L. Guo, “A Fast Detection Method based on Deep
Learning of Millimeter Wave Human Image,” in Proc. Intl. Conf. on
Artificial Intelligence and Virtual Reality , Nov. 2018, pp. 67–71.

 

 
 [196] 
 
Y. Amano, Recognition Ability for Millimeter-wave Radar Hyper
Resolution by New Principle at Once, Romance of the Three Kingdoms in
Vehicular Sensor Field the 2nd Division . Nikkei Electronics, Mar. 2018, pp. 25–30,
 https://xtech.nikkei.com/dm/atcl/mag/15/318381/201803/ .

 

 
 [197] 
 
A. Santra, R. V. Ulaganathan, T. Finke, A. Baheti, D. Noppeney, J. R.
Wolfgang, and S. Trotta, “Short-range Multi-mode Continuous-wave Radar
for Vital Sign Measurement and Imaging,” in IEEE Radar Conf. , 2018,
pp. 0946–0950.

 

 
 [198] 
 
N. Yamada, Y. Tanaka, and K. Nishikawa, “Radar Cross Section for
Pedestrian in 76GHz Band,” in Proc. European Microwave Conf. ,
vol. 2, 2005.

 

 
 [199] 
 
K. A. Gallagher, “Simultaneous Human Detection and Ranging using a
Millimeter-wave Radar System Transmitting Wideband Noise with an Embedded
Tone,” Master’s thesis, Pennsylvania State University, May 2013,
 https://etda.libraries.psu.edu/files/final_submissions/8576 .

 

 
 [200] 
 
I. V. Mikhelson, S. Bakhtiari, T. W. Elmer, and A. V. Sahakian, “Remote
Sensing of Patterns of Cardiac Activity on an Ambulatory Subject using
Millimeter-wave Interferometry and Statistical Methods,” Medical 
Biological Engineering Computing , vol. 51, no. 1, pp. 135–142, Feb.
2013.

 

 
 [201] 
 
H. Ding, I. Y. Soon, S. N. Koh, and C. K. Yeo, “A Spectral Filtering Method
based on Hybrid Wiener Filters for Speech Enhancement,” Speech
Communication , vol. 51, no. 3, pp. 259 – 267, 2009.

 

 
 [202] 
 
H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan,
Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A Multimodal Dataset for
Autonomous Driving,” in Proc. IEEE/CVF Conf. Computer Vision and
Pattern Recognition , 2020, pp. 11 621–11 631.

 

 
 [203] 
 
S. W. Smith, Chapter 15 - Moving Average Filters . California Technical Publishing, 1997, pp. 277–284.

 

 
 [204] 
 
Nat. Instrum. (2017) Instrument Fundamentals. Austin, TX, USA. [Online].
Available:
 https://www.ni.com/en-ca/innovations/white-papers/06/understanding-ffts-and-windowing.html 

 

 
 [205] 
 
Texas Instruments. (2017, Oct.) MMWAVE_SDK 01_01_00_02. When installed,
the relevant documentation can be found on the following local webpage:
/mmwave_sdk_01_01_00_02/packages/ti/demo/xwr14xx/mmw/docs/
doxygen/html/index.html. [Online]. Available:
 http://software-dl.ti.com/ra-processors/esd/MMWAVE-SDK/01_01_00_02/index_FDS.html 

 

 
 [206] 
 
H. Rohling, “Ordered statistic CFAR technique - an overview,” in
 Proc. 12th Intl. Radar Symp. (IRS) , 2011, pp. 631–638.

 

 
 [207] 
 
R. Nitzberg, “Clutter Map CFAR Analysis,” IEEE Trans. on Aerospace
and Electronic Sys. , vol. AES-22, no. 4, pp. 419–421, 1986.

 

 
 [208] 
 
T. Bower. 17.8. frequency domain filters. Kansas State University. [Online].
Available:
 http://faculty.salina.k-state.edu/tim/mVision/freq-domain/freq_filters.html 

 

 
 [209] 
 
S. Raschka. (2014, Jul.) About Feature Scaling and Normalization. [Online].
Available:
 https://sebastianraschka.com/Articles/2014_about_feature_scaling.html 

 

 
 [210] 
 
D. Storcheus, A. Rostamizadeh, and S. Kumar, “A Survey of Modern Questions
and Challenges in Feature Extraction,” in Proc. 1st Workshop on
Feature Extraction: Modern Questions and Challenges , vol. 44, 2015, pp.
1–18.

 

 
 [211] 
 
I. Guyon and A. Elisseeff, An Introduction to Feature
Extraction . Springer, 2006, pp.
1–25.

 

 
 [212] 
 
M. Livshitz. (2018, Mar.) Tracking Radar Targets with Multiple Reflection
Points. Texas Instruments. [Online]. Available:
 https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/1023/Tracking-radar-targets-with-multiple-reflection-points.pdf 

 

 
 [213] 
 
Y. Bengio, A. Courville, and P. Vincent, “Representation Learning: A
Review and New Perspectives,” IEEE Trans. on Pattern Analysis and
Machine Intelligence , vol. 35, no. 8, pp. 1798–1828, 2013.

 

 
 [214] 
 
A. K. Duun-Henriksen, S. Schmidt, R. M. Røge, J. B. Møller, K. Nørgaard,
J. B. Jørgensen, and H. Madsen, “Model Identification Using Stochastic
Differential Equation Grey-Box Models in Diabetes,” Diabetes Science
and Technology , vol. 7, no. 2, pp. 431–440, 2013.

 

 
 [215] 
 
Z. Chen. (2003) Bayesian Filtering: From Kalman Filters to Particle Filters,
and Beyond. Adaptive Sys. Lab., McMaster Univ. [Online]. Available:
 https://www.researchgate.net/publication/238689222_Bayesian_Filtering_From_Kalman_Filters_to_Particle_Filters_and_Beyond 

 

 
 [216] 
 
C. Yardim, Z. Michalopoulou, and P. Gerstoft, “An Overview of
Sequential Bayesian Filtering in Ocean Acoustics,” IEEE Oceanic
Engineering , vol. 36, no. 1, pp. 71–89, 2011.

 

 
 [217] 
 
M. Rubinstein. (2009) Introduction to Recursive Bayesian Filtering. Presented
at seminar on Advanced Topics in Computer Graphics, Tel Aviv University.
[Online]. Available:
 https://people.csail.mit.edu/mrub/talks/filtering.pdf 

 

 
 [218] 
 
L. M. Ehrman and A. D. Lanterman, “Automated Target Recognition using Passive
Radar and Coordinated Flight Models,” in Proc. SPIE on Automatic
Target Recognition XIII , F. A. Sadjadi, Ed., vol. 5094, 2003, pp. 196 –
207.

 

 
 [219] 
 
I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning . MIT Press, 2016,
 http://www.deeplearningbook.org .

 

 
 [220] 
 
C.-C. Chang and C.-J. Lin, “LIBSVM: A Library for Support Vector Machines,”
 ACM Trans. Intelligent Sys. and Tech. , vol. 2, no. 3, May 2011,
article 27.

 

 
 [221] 
 
S. K. Murthy, “Automatic Construction of Decision Trees from Data: A
Multi-Disciplinary Survey,” Data Mining and Knowledge Discovery ,
vol. 2, no. 4, pp. 345–389, Dec. 1998.

 

 
 [222] 
 
N. Cristianini and J. Shawe-Taylor, An Introduction to Support Vector
Machines and Other Kernel-based Learning Methods . Cambridge University Press, 2000.

 

 
 [223] 
 
Mathematical Model. Wikipedia. [Online]. Available:
 https://en.wikipedia.org/wiki/Mathematical_model 

 

 
 [224] 
 
M. Hossin and M. N. Sulaiman, “A Review on Evaluation Metrics for Data
Classification Evaluations,” Data Mining and Knowledge Management
Process , vol. 5, no. 2, pp. 1–11, 2015.

 

 
 [225] 
 
R. Raina, A. Battle, H. Lee, B. Packer, and A. Y. Ng, “Self-Taught Learning:
Transfer Learning from Unlabeled Data,” in Proc. 24th Intl. Conf.
Machine Learning , 2007, p. 759–766.

 

 
 [226] 
 
A. Dosovitskiy, J. T. Springenberg, M. Riedmiller, and T. Brox,
“Discriminative Unsupervised Feature Learning with Convolutional Neural
Networks,” in Proc. 27th Intl. Conf. Neural Information Processing
Sys. , vol. 1. MIT Press, 2014, p.
766–774.

 

 
 [227] 
 
J. Hasch, E. Topak, R. Schnabel, T. Zwick, R. Weigel, and
C. Waldschmidt, “Millimeter-Wave Technology for Automotive Radar Sensors
in the 77 GHz Frequency Band,” IEEE Trans. Microwave Theory and
Techniques , vol. 60, no. 3, pp. 845–860, 2012.

 

 
 [228] 
 
S. T. Kouyoumdjieva, P. Danielis, and G. Karlsson, “Survey of
Non-Image-Based Approaches for Counting People,” IEEE Communications
Surveys and Tutorials , vol. 22, no. 2, pp. 1305–1336, 2020.

 

 
 [229] 
 
M. Kebe, R. Gadhafi, B. Mohammad, M. Sanduleanu, H. Saleh, and M. Al-Qutayri,
“Human Vital Signs Detection Methods and Potential Using Radars: A
Review,” MDPI Sensors , vol. 20, no. 5, 2020, article 1454.

 

 
 [230] 
 
C. Xiao, D. Han, Y. Ma, and Z. Qin, “CsiGAN: Robust Channel State
Information-Based Activity Recognition With GANs,” IEEE Internet of
Things Journal , vol. 6, no. 6, pp. 10 191–10 204, 2019.

 

 
 [231] 
 
Y. Zheng, Y. Zhang, K. Qian, G. Zhang, Y. Liu, C. Wu, and Z. Yang,
“Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi,” in Proc.
17th Int. Conf. on Mobile Systems, Applications, and Services , ser. MobiSys
’19. New York, NY, USA: Association
for Computing Machinery, 2019, p. 313–325.

 

 
 [232] 
 
S. Palipana, D. Rojas, P. Agrawal, and D. Pesch, “FallDeFi: Ubiquitous Fall
Detection Using Commodity Wi-Fi Devices,” Proc. ACM Interact. Mob.
Wearable Ubiquitous Technol. , vol. 1, no. 4, Jan. 2018, article 155.

 

 
 [233] 
 
W. Ross. (2017, Jul.) The Autonomic Nervous System and its Effects on Health
and Vitality. [Online]. Available:
 https://www.crossroadsapothecary.com/blog/2017/7/17/integrative-medicine-at-crossroads-the-autonomic-nervous-system-and-its-effects-on-health-and-vitality-author-dr-warren-ross 

 

 
 [234] 
 
Centers for Disease Control and Prevention. (2020, Jul.) Mental Health and
Coping During COVID-19. [Online]. Available:
 https://www.cdc.gov/coronavirus/2019-ncov/daily-life-coping/managing-stress-anxiety.html 

 

 
 [235] 
 
D. J. Plews, B. Scott, M. Altini, M. Wood, A. E. Kilding, and P. B. Laursen,
“Comparison of Heart-Rate-Variability Recording With
Smartphone Photoplethysmography, Polar H7 Chest Strap, and
Electrocardiography,” Int. J. Sports Physiol. Perform. , vol. 12,
no. 10, pp. 1324–1328, Nov 2017.

 

 
 [236] 
 
V. L. Petrović, M. M. Janković, A. V. Lupšić, V. R. Mihajlović,
and J. S. Popović-Božović, “High-Accuracy Real-Time Monitoring of
Heart Rate Variability Using 24 GHz Continuous-Wave Doppler Radar,”
 IEEE Access , vol. 7, pp. 74 721–74 733, 2019.