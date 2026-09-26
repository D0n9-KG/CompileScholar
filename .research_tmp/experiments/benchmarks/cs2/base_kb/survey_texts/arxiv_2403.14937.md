Survey on Modeling of Human-made Articulated Objects 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2403.14937v3 [cs.CV] 19 Mar 2025 
 
 
 \STAR \CGFStandardLicense 
 

# Survey on Modeling of Human-made Articulated Objects

 Volume:  44 Issue:  2 
 
 
 Jiayi Liu  Manolis Savva  Ali Mahdavi-Amiri
 Simon Fraser University
 

 Abstract 
 
 3D modeling of articulated objects is a research problem within computer vision, graphics, and robotics.
Its objective is to understand the shape and motion of the articulated components, represent the geometry and mobility of object parts, and create realistic models that reflect articulated objects in the real world.
This survey provides a comprehensive overview of the current state-of-the-art in 3D modeling of articulated objects, with a specific focus on the task of articulated part perception and articulated object creation (reconstruction and generation).
We systematically review and discuss the relevant literature from two perspectives: geometry modeling (i.e., structure and shape of articulated parts) and articulation modeling (i.e., dynamics and motion of parts).
Through this survey, we highlight the substantial progress made in these areas, outline the ongoing challenges, and identify gaps for future research.
Our survey aims to serve as a foundational reference for researchers and practitioners in computer vision and graphics, offering insights into the complexities of articulated object modeling.

 
 † † year: 2025 † † year: 2025 † † editors: A. Bousseau and A. Dai † † editors-short: D. Ceylan and T.-M. Li † † editors-star: Y. Gryaditskaya and P. Memari † † editors-tutorial: K. Hildebrandt and R. Mantiuk † † editors-education: R. Kuffner dos Anjos and K. Rodriguez Echavarria † † editors-poster: T. Günther and Z. Montazeri † † editors-dc: L. Liu and G. Singh † † subject: EUROGRAPHICS CGF Vol No † † teaser: 
 
 
In 3D modeling of human-made articulated objects, mobility perception, articulated object reconstruction and articulated object generation are common problems that can be solved using a range of approaches and under different settings of input, output, representation, etc.
From left to right:
1) mobility perception from a single snapshot of an object  [ HLV*17 ] or a scene  [ SHL*14 ] ;
2) articulated object reconstruction from point clouds  [ JHZ22 ] and multi-view images  [ LMS23 ] ;
3) articulated object generation unconditionally  [ LDS*23 ] and constrained by a graph  [ LTMS24 ] .
Figures reproduced from original papers  [ SHL*14 , HLV*17 , JHZ22 , LMS23 , LDS*23 , LTMS24 ] .
 

 
 

## 1 Introduction

 
 Articulated objects, composed of multiple rigid parts connected by joints, are ubiquitous in our daily life, encompassing a wide range of objects, such as mechanical assemblies, furniture pieces, even human bodies and other animals.
Modeling these objects is a research field intersecting computer vision, graphics, and robotics.
It aims to comprehend and represent the shape and mobility of articulated components.
This endeavor extends to creating 3D models that realistically reflect real-world articulated objects.
Contributing to the dynamics and functionality in our physical world, articulated object modeling is essential for a wide range of applications, such as animation  [ YVN*22 , CJS*23 , QWM*23 , LZWL23 ] and simulation  [ WCK19 , YZW*22 , XQM*20 ] , robotic interaction and manipulation  [ HNOS15 , GES21 , MGM*21 , QF23 ] , embodied AI  [ KMH*17 , PRB*18 , GGS*19 , SKM*19 , PUS*23 ] , etc.

 
 
 The complexity of articulated object modeling stems from the fact that articulated objects are not just represented by the geometry of their parts but also by their kinematic structure.
Over the past decade, much research has focused on addressing these complexities in two main directions: articulated part perception , and articulated object creation .
The primary goal of articulated part perception is to analyze the mobility of parts in an object, which is useful in two application areas: 1) object manipulation in robotics; and 2) interaction and animation in simulations.
The vision and robotics communities have been actively working on the first application area  [ GES21 , MGM*21 , EZH22 , XHS22 , WWM*22 , MHF*22 , GLG*23 , BXQW23 , HJZ23 , NWL*24 , LWW*24 , YWL*24 , WLY*24 ] , where the task is usually decomposed to affordance prediction, articulation estimation, and action planning.
We leave this line of work out of the scope of this survey, as it is focused on manipulation of articulated objects with robots in the real world.
Another line of work from vision and graphics community focuses more on the geometric and kinematic modeling of the articulated objects for the second application area, where the task is usually decomposed to articulated part segmentation and kinematic structure understanding.
For articulated object creation, the main goal is to reconstruct or generate articulated parts into a compositional and hierarchical 3D model that represents a real-world object that can be interacted with in a virtual environment.

 
 
 This survey provides a comprehensive overview of the current state-of-the-art in 3D modeling of human-made articulated objects.
Through this survey, we highlight the substantial progress made in these areas, outline the ongoing challenges, and identify the gaps for future research.
Our survey aims to serve as a foundational reference for researchers and practitioners in computer vision and graphics, offering insights into the complexities of articulated object modeling and inspiring new research in this area.

 
 
 We begin by laying the groundwork with an overview of the field, defining the scope and focus of our discussion in  Section   2 .
Next, in  Section   3 , we compile the key datasets for articulated objects that have been collected and utilized within the research community, highlighting their critical role in advancing the 3D modeling of articulated objects.
Our survey then delves into an in-depth analysis of the techniques and methodologies developed for 3D modeling of articulated objects.
This analysis is organized around two pivotal axes: geometry modeling and articulation modeling, which are the foundational components to tackle the complexities in this field.
Geometry modeling covers the representation and approaches employed to understand the shape of the articulated parts, whereas articulation modeling involves the articulation model and the methodologies used to estimate the part mobility and kinematic structure of the object.
In the end, we conclude our survey by identifying the key challenges and potential research directions that can drive future progress in the field.

 
 
 Related surveys. 
The most related survey to ours is the recent survey by [ PŠDC22 ] , which focuses on the manipulation of articulated objects in the realm of robotics and computer vision.
In contrast, our focus is on the geometry and articulation modeling of articulated objects within the field of computer vision and graphics.
This aspect represents a distinct yet critical dimension of articulated object research, orthogonal to the topic of manipulation.
Additionally, another related survey by [ HSvK18 ] explores the functionality of general objects within the scope of computer graphics literature.
The functionality of an object is influenced not just by its mobility but also by its shape, material, and various other attributes that collectively determine its capability to perform a task.
Our survey takes a more specialized focus on the articulated object and its mobility particularly, compared to the work by [ HSvK18 ] which concentrates on functionality for general objects.

 
 
 Figure 1: Common joint types.
The first row shows the joints with linear motion which is the most common assumption considered in the literature.
The second row shows more complex joints with more than one DoF.
Figure reproduced from original paper  [ LP17 ] . 
 
 
 

## 2 Background Scope

 

### 2.1 Definition of Articulated Objects

 
 An articulated object is composed of multiple rigid parts interconnected by joints.
These joints serve as the pivotal points that allow for relative motion between the connected parts.
The range and nature of the articulation of each part are constrained and defined by the type of joint it possesses.
Among human-made objects, the most frequently encountered joint types facilitate linear motion, such as revolute, prismatic, and helical joints.
For example, a laptop is an articulated object with a revolute joint connecting the screen to the keyboard.
This joint allows the screen to rotate around a single axis relative to the keyboard, with the range of motion being restricted to a specific limit.
In contrast, the human body is a more complex example of an articulated object, with a wider variety of joint types, such as ball-and-socket or spherical joints.
These joints allow for motion beyond linear translation or rotation, enabling a more diverse range of movements.
The illustration of several common joint types is shown in Figure   1 .

 
 
 Types of articulated objects. 
Articulated objects, with their distinctive characteristics and varied types, play a significant role in the dynamics of our living environments.
In general, articulated objects can be classified into two types: organic objects and human-made objects .

 
 
 
 • 
 
 Organic objects refer to naturally occurring systems that are capable of initiating movement through articulated structures.
The bodies of humans and other animals are common examples of organic articulated objects.
They are crucial for various functions such as locomotion, manipulation, and interaction with the environment.
A skeleton with bones and joints is usually used to represent the kinematic structure of these objects.
The motion allowed for these objects is usually structurally complex with a relatively large number of joints varied in the topology of connections and degrees of freedom.
However, the kinematic structure is relatively fixed and can be predefined for a specific species.
How to extract effective skeletons, reconstruct 3D models, and fit dynamic motions for animals across various species is an active research topic  [ KGFT20 , YSJ*21 , WCL*22 , YHL*22 , WLJ*23 , AM24 , LWP*24 , YHL*23 , LLL*24 , YRH*24 , LSZ*24 , JLW*24 , MNH23 , YZS*24 ] .
Leveraging the skeleton template of the human body or hands and coordination between the joints to achieve the desired animation  [ ALL*20 , SYZR21 , BKY*22 , SBR22 , WSGT22 , KKK*23 , LWP*24 , ZZC*23 , LZWL23 , QWM*23 , TYR23 ] or human-object interaction  [ TGBT20 , ZPJ*20 , XJMS21 , CRKM21 , ZBS*22 , FTT*22 , BXP*22 , QJR*22 , GDG*23 , JLC*23 , HYL*23 , LJ23 , XWA*24 ] is another line of research.
We direct readers to a recent survey on human body modeling  [ BDTB18 , CPZ21 , WTZ*21 , MHL*22 , AKMS23 , ZWC*23 , YZF*24 ] and 3D hand modeling  [ CYL15 , WWS*15 , Bar16 , LLT19 , AMD19 ] for more details.
In this survey, we leave the discussion of organic articulated objects out of the scope and only focus on human-made articulated objects.

 

 • 
 
 Human-made objects are engineered and designed to mimic the natural articulation of organic objects or to serve specific functionalities that require movement.
These objects are complex assemblies of rigid parts connected by joints, and their motion is often a direct result of interaction with the environment or driven by an active system.
Common examples of human-made articulated objects include gadgets, furniture, vehicles, and other mechanical systems.
These objects are ubiquitous in our daily lives and play an important role in our interaction with the physical world.
Modeling human-made articulated objects presents several unique challenges in terms of physical and geometric modeling, articulation modeling, and environmental interaction.
Modeling these objects is the focus of this survey and will be discussed in detail in the following sections.
We will only mention articulated objects for short in the rest of the survey to refer to human-made articulated objects.

 

 
 
 
 Key differences. 

Modeling organic and human-made articulated objects presents distinct challenges due to differences in geometry, kinematic structure, motion patterns, and interaction with the environment.
Topologically, organic objects share consistent skeletal structures within species, enabling predefined templates (e.g., human body, quadrupeds), whereas human-made objects exhibit greater variability.
Some categories follow standard designs (e.g., eyeglasses, scissors), while others (e.g., storage furniture) vary significantly across instances.
In terms of motion, organic objects have biomechanical joints driven by muscle forces, leading to soft-tissue deformations, whereas human-made objects typically follow rigid-body mechanics with constrained degrees of freedom.
While this generally simplifies articulation modeling, challenges arise in accurately estimating joint parameters in relation to object geometry, handling nested joints with complex kinematic dependencies (e.g., bi-fold doors, folding umbrellas), and accounting for real-world imperfections such as frictional resistance and mechanical backlash.
Additionally, organic objects evolve for locomotion and behaviors, while human-made objects are designed for specific functionalities, influencing how they articulate and are manipulated.
Consequently, research on organic objects emphasizes geometric reconstruction and animation, whereas human-made objects are studied in the context of 3D modeling for robotic manipulation and functional interaction.

 
 
 These fundamental differences make it difficult for methods developed for one type to directly generalize to the other.
Structure-wise, the parametric models and statistical priors used for organic objects rely on consistent skeletal structures, which are not scalable to accommodate the diverse topologies of human-made objects.
Geometrically, organic objects are typically represented as a single continuous surface, i.e., skin enclosing a skeleton.
Reconstruction methods optimized for organic shapes rely on this assumption and are not well-suited for human-made objects, which often consist of multiple discrete, interlocking parts with internal structures, requiring different modeling approaches.
Articulation-wise, organic objects exhibit complex, non-linear motion patterns driven by muscle forces, necessitating biomechanical models and motion capture data for accurate representation.
On the other hand, the rigid-body mechanics of human-made objects are more amenable to analytical kinematic models, simplifying articulation modeling but requiring precise joint parameter estimation to enable plausible interaction in physical simulations.

 
 
 

### 2.2 Representation of Articulated Objects

 
 Articulated object modeling is a multifaceted domain within computer vision, graphics, and robotics.
It aims to understand the shape and interactability of the articulated components, represent the geometry and mobility of these objects, and create realistic models that reflect articulated objects in the real world.
The inherent complexity of this task arises from the fact that the articulation of the object is not only determined by the geometry of the parts but also by the kinematic structure of the object.
As a result, effective modeling of articulated objects requires capturing a dual representation of the geometry and articulation of the object.
These two facets are deeply interconnected, each influencing and informing the other.
The geometric decomposition of the parts lays the foundational groundwork for analyzing part mobility, while the process of articulation modeling also guides the shape understanding of the mobility parts.
This symbiotic relationship between geometric and articulation modeling underscores the importance of a cohesive approach in articulated object modeling, where understanding and representing both geometry and articulation are essential for capturing the full essence and functionality of complex entities.
Over the last decade, there has been a notable surge in research within this field, addressing the inherent challenges predominantly from two perspectives: geometry modeling and articulation modeling .
We will further discuss these two axes in the following sections.

 
 
 Figure 2: Common geometric representations for articulated object modeling.
Figures reproduced from original papers  [ LTMS24 , YHY*19 , SNF14 ] .
 
 
 
 Geometric representation. 
Unlike the geometric modeling of static objects, which typically concentrates solely on the object’s outer surface, the geometric representation of articulated objects demands a more nuanced approach.
This involves capturing the spatial arrangement of the parts, modeling the part-level surfaces, and constructing the interior structure of the object.
It is essential to accurately describe the shape, size, relative position, and orientation of each individual part that is structurally organized within the object.
This comprehensive description is crucial, as it provides the foundational details necessary to represent a complete object that can be interactively manipulated.
The data format used to represent object geometry varies based on the data capture methods and the intended application.
The most common access to the part geometry is through the 3D point cloud, which is a set of points sampled from the object surface that can be obtained by 3D scanning or depth projection.
While point clouds provide a raw, unstructured representation of the object’s geometry, they often undergo further processing into more organized formats for detailed analysis.
Another common representation is 3D mesh, which approximates the surface of each part with primitives such as polygons.
The mesh format is widely used in applications such as simulation, rendering, and animation due to its ability to effectively model surface details and because it can flexibly represent object structure.
In Section   4 , we will discuss different choices of geometric representation in various related tasks in detail.
Here we specify the geometric representations that are considered in the literature and visually illustrated in Figure   2 .

 
 • 
 
 Mesh : is a collection of vertices, edges, and polygons usually with a structured connectivity (e.g., 2-manifold) that approximate the surface.

 

 • 
 
 Point cloud (PC) : a set of 3D points sampled on object surface that can be obtained from 3D scanning or depth projection.

 

 • 
 
 Implicit field : is an implicit function that assigns a value to each point in the space to construct a field describing the object.
Common forms of implicit fields include the signed distance field (SDF), occupancy field, density field, etc.
The function parameterized by a neural network, known as a neural field, is a popular intermediate representation as it is inherently differentiable and easy to integrate with learning-based methods.

 

 • 
 
 Images-based : some work approaches the 3D modeling and perception problem from the image-based domain, such as depth images, RGB-D images and posed RGB images captured from multi-views or stereo cameras.

 

 
 
 
 Figure 3: Common articulated motion representations.
Figures reproduced from original papers  [ YHL*18 , LIC*24 ] . 
 
 
 Articulation representation. 
The representation for articulation includes the description of the mobility of each part and the kinematic structure of the object.
The kinematic structure describes how the object’s parts are connected and how they can move relative to each other.
Typically, this kinematic structure is represented by a tree or graph, which is an effective way to model hierarchical relationships in articulated systems.
The part mobility can be described by the parameters of each connected joint or the deformation field of the object.
In Section   5 , we will discuss ways of articulation modeling with different assumptions and representation in various related tasks in detail.
Here we specify the articulated motion representation as follows and visually illustrated in Figure   3 .

 
 • 
 
 Deformation field : the deformation field is a general form of motion representation that describes the displacement of each point as a 3D vector. It can also be used to represent beyond the rigid motion of the shape in a continuous space, such as free-form deformation.

 

 • 
 
 Joint parameters : the parameters of each joint, including the joint type , the joint axis , the joint state such as the rotational angle or translational distance, and the motion range or joint limit .

 

 • 
 
 Kinematic tree : the kinematic structure of the object represented by a tree or graph, where the nodes represent the articulated parts and the edges represent the joint connections.

 

 
 
 
 Figure 4: Representative datasets for 3D articulated objects: PartNet-Mobility  [ XQM*20 ] for synthetic data, AKB-48  [ LXF*22 ] for real scans of objects, MultiScan  [ MZJ*22 ] for real indoor scenes with articulated objects, and GAPartNet  [ GXZ*23 ] for synthetic and real objects with fine-grained annotations for the generalizable actionable parts. ParaHome  [ KKNJ24 ] is a recent dataset providing human-object interactions with articulated objects. 
 
 
 
 
 
 
 | 
 Source | 
 Level | 
 Representation | 
 Textured | 
 # objects | 
 # movable parts | 
 # object categories | 

 
 [ HLV*17 ] | 
 synthetic | 
 object | 
 mesh | 
 ✗ | 
 368 | 
 368 | 
 - | 

 
 RPM-Net  [ YHY*19 ] | 
 synthetic | 
 object | 
 mesh | 
 ✗ | 
 969 | 
 1,420 | 
 43 | 

 
 Shape2Motion  [ WZS*19 ] | 
 synthetic | 
 object | 
 mesh | 
 ✗ | 
 2,440 | 
 6,762 | 
 45 | 

 
 RBO  [ MEB19 ] | 
 synthetic, real | 
 object | 
 mesh, RGB-D | 
 ✓ | 
 14 | 
 21 | 
 14 | 

 
 PartNet-Mobility  [ XQM*20 ] | 
 synthetic | 
 object | 
 mesh | 
 ✓ | 
 2,346 | 
 14,068 | 
 46 | 

 
 ReArt-48  [ LXX*22 ] | 
 real | 
 object | 
 mesh | 
 ✓ | 
 48 | 
 - | 
 5 | 

 
 AKB-48  [ LXF*22 ] | 
 real | 
 object | 
 mesh | 
 ✓ | 
 2,037 | 
 - | 
 48 | 

 
 OPDSynth  [ JMSC22 ] | 
 synthetic | 
 object | 
 mesh, RGB | 
 ✓ | 
 683 | 
 1,343 | 
 11 | 

 
 OPDReal  [ JMSC22 ] | 
 real | 
 object | 
 mesh, RGB-D | 
 ✓ | 
 284 | 
 875 | 
 8 | 

 
 MultiScan  [ MZJ*22 ] | 
 real | 
 scene | 
 mesh, RGB-D | 
 ✓ | 
 10,957 | 
 5,129 | 
 20 | 

 
 OPDMulti  [ SJSC23 ] | 
 real | 
 scene | 
 RGB-D | 
 ✓ | 
 217 | 
 688 | 
 33 | 

 
 GAPartNet  [ GXZ*23 ] | 
 synthetic, real | 
 object | 
 mesh | 
 ✓ | 
 8,489 | 
 1,166 | 
 27 | 

 
 ACD  [ IJZ*24 ] | 
 synthetic | 
 object | 
 mesh | 
 ✓ | 
 354 | 
 1,350 | 
 21 | 

 
 ParaHome  [ KKNJ24 ] | 
 real | 
 object | 
 mesh | 
 ✗ | 
 22 | 
 - | 
 8 | 

 

 
 Table 1: 
Datasets related to articulated objects with statistics on the source of collection, the level at which these datasets operate, data representation, whether the 3D model is texture, the number of objects, movable parts, and object categories.
The ’-’ symbol indicates that the information is not available or not reported in the original paper.
 
 
 
 
 

## 3 Datasets

 
 One of the main contributors to recent advances in deep learning is the availability of large-scale 3D datasets for various computer vision tasks, particularly in shape understanding.
However, datasets for articulated objects are relatively scarce compared to those for static 3D objects.
Gathering data for articulated objects is more laborious because it requires detailed modeling of each part’s geometry and careful annotation of articulation parameters.
The challenges in creating datasets for articulated objects are multifaceted:

 
 • 
 
 The creation of synthetic datasets involves manual design of the geometry for each component and annotating the articulation parameters.
This labor-intensive process restricts the number of categories and instances in these datasets, often leading to a lack of diversity and complexity in the data samples.
As a result, models may be less realistic or overly simplistic.

 

 • 
 
 Real-world datasets are typically gathered using sensors like RGB-D cameras, followed by post-processing to reconstruct shapes and annotate part attributes.
One of the significant challenges in modeling articulated objects, as opposed to static surface models, is the difficulty in capturing the interior structure of each part from surface data alone.
It usually involves multiple scans from different states of the object to capture the complete geometry.
The data obtained from surface sensors is often incomplete and noisy, as the interior structure frequently is occluded by the outer surface.
This limitation affects the geometric quality of the data and further reduces the diversity and complexity of data samples.

 

 
 With the recent surge in interest in articulated objects, there has been a growing effort to create datasets for this domain.
 Figure   4 shows several representative datasets for 3D articulated objects.
We list all existing datasets for articulated objects in both object-level and scene-level in Table   1 , where we summarize the key characteristics of each dataset.
We also provide a brief overview of each dataset in the folllowing subsections.

 
 

### 3.1 Synthetic Data

 
 The recent advances in 3D modeling technology have led to a significant increase in the availability of 3D models with part-level structures accessible online.
This progress has opened up new possibilities for detailed analysis and application in various fields, enhancing the resources for researchers and practitioners working with complex 3D structures.
Articulated objects are one such class of complex 3D structures that have benefited from this progress.
The availability of 3D datasets with part-level structures has enabled the creation of synthetic datasets for articulated objects.

 
 
 To facilitate the prediction of part mobility via a learning-based approach, [ HLV*17 ] collect a synthetic dataset for 3D articulated objects by sourcing shapes from ShapeNet  [ CFG*15 ] and SketchUp  [ Inc17a ] .
In this process, they manually segment the shapes into parts and provide detailed annotations for the articulation parameters.
Following this foundational work, RPM-Net  [ YHY*19 ] extends the scope of the dataset to include a greater variety of objects incorporating more complex motion, e.g., the opening of the umbrella cover.
For each shape in the dataset, every possible pair of parts is labeled as a reference part and a moving part .
Further, each pair is regarded as a mobility unit , associated with annotated motion parameters.
These parameters form a quadruple, including the transformation type (including translation, rotation, or the combination of the two), the position and direction of the transformation axis, and the range of the motion.

 
 
 Shape2Motion  [ WZS*19 ] is another synthetic dataset in a larger scale that was introduced concurrently.
The shapes are sourced from ShapeNet  [ CFG*15 ] and 3D Warehouse  [ Inc17 ] .
Each shape includes part segmentations and articulation parameters that are annotated in a similar way as RPM-Net  [ YHY*19 ] and [ HLV*17 ] with the use of a developed annotation tool.
This tool enhances the efficiency of the annotation process by allowing the annotators to visually verify the correctness by animating the object with the annotated parameters.
This allows for boosting the scale of the dataset with a much larger number of objects and movable parts in a wide range of categories.

 
 
 The datasets mentioned above are valuable resources for research on shape analysis and mobility prediction of articulated objects using 3D meshes or point clouds as input.
However, a notable limitation of these datasets is the absence of texture information.
This restricts their utility in applications such as simulation, digital twins creation, augmented reality, and other modeling tasks that require a more holistic visual representation.
To fill in this gap, the PartNet-Mobility dataset, with a simulation environment SAPIEN   [ XQM*20 ] , was introduced.
This dataset is a subset and an extension from a part-level 3D shape dataset PartNet  [ MZC*19 ] , enriched with articulation annotations organized in URDF files.
The inclusion of diverse appearances and motions in PartNet-Mobility significantly enhances its value, inspiring increasingly more research in the field of articulated object modeling in connection with visual perception  [ JMSC22 , SJSC23 , WGYZ25 ] and digital twin creation  [ JHZ22 , LMS23 , WWT*24 ] .
Building upon the PartNet-Mobility dataset, OPDSynth  [ JMSC22 ] is introduced for the task of openable part detection (OPD) by blending the rendering of the synthetic data with real-world RGB images.
This dataset enables the mobility part detection from images, aligning more closely with practical real-world applications.

 
 
 Later, the GAPartNet  [ GXZ*23 ] dataset is introduced to capture finer-level part details that have been previously overlooked in existing datasets.
This dataset is tailored to enhance the generalization of object perception and manipulation tasks across various object categories by focusing on the concept of generalizable and actionable parts .
The underlying premise is that functional parts, such as buttons and handles, are fundamental elements whose identification and extraction can significantly improve generalizability within and across object categories.
They argue that the functional parts such as buttons and handles are more elementary, and the extraction of these parts can improve the generalizability in an intra-category manner.
To achieve a comprehensive and versatile dataset, GAPartNet selectively compiles data from both the PartNet-Mobility  [ XQM*20 ] and AKB-48  [ LXF*22 ] datasets to encompass both synthetic and real data.
Sourcing from GAPartNet, Drag-a-Move  [ LZRV24 ] provides a 2D synthetic dataset for the task of part-oriented drag control.
It contains a collection of triplets ( x , y , D ) (x,y,D) , where image x x and y y are the object images in an initial and new states, and D D is a collection of drag actions applied to image x x .
The drag annotation is projected from the ground-truth articulation annotations from 3D models to 2D images.
This dataset facilitates research on interactive generative models for articulated objects using drag control on 2D images with 3D awareness.

 
 
 Recently, S2O  [ IJZ*24 ] points out that PartNet-Mobility dataset suffers from lack of diversity, contains highly similar objects, and is biased towards objects with simple articulation structures.
So they propose a new dataset, Articulated Containers Dataset (ACD), which provides more challenging and realistic 3D objects.
The shapes are sourced from ABO  [ CGD*22 ] , 3D-Future  [ FJG*21 ] , and HSSD  [ KMJ*24 ] datasets.
The objects in ACD are more complex in geometry (e.g., with L-shaped tables and corner cabinets) and articulation structures with significantly more openable parts.
This dataset can potentially help to reduce the gap of models trained on synthetic data generalized to real objects.
On the track to increase the data diversity, Arti-PG  [ SLW*24 ] introduces a procedural generation toolbox that uses mathematical rules to synthesize articulated objects, with code as the user interface.
By randomizing configurations, this toolbox enables unlimited data variations, making it useful for scaling dataset volume.

 
 
 

### 3.2 Real-world Data

 
 A primary challenge in working with synthetic data is the unrealistic assumption that it is perfect—overly simplistic, complete, and free from noise.
This can result in models that perform well in synthetic environments but struggle when applied to real-world scenarios, where data is often noisy and incomplete
To fill this synthetic-real gap, more recent efforts have started to focus on collecting data from real-world sensors, such as RGB-D cameras.

 
 
 RBO  [ MEB19 ] represents the pioneering dataset in this regard, being the first to collect data from RGB-D scans.
It not only provides RGB-D recordings of human interaction with objects under varying experimental conditions, but also creates corresponding synthetic mesh models for each object.
ReArt-48  [ LXX*22 ] introduced a slightly larger dataset that reconstructs meshes from RGB-D scans and provides the articulation annotations across five object categories.
Taking this further, AKB-48  [ LXF*22 ] emerges as the first large-scale dataset for articulated objects based on real scans,
where each object is described in a knowledge graph.
To construct this dataset, a fast articulation knowledge modeling pipeline is presented, significantly reducing the cost and effort required for object modeling in the real world.
This innovation enables the creation of 3D models on a scale comparable to PartNet-Mobility.
Additionally, AKB-48 also annotates the physical properties of the objects, such as mass, to enhance the dataset’s applicability.
This additional annotation is important for bridging the generalization gap between simulation and real-world applications.

 
 
 In parallel, the MultiScan  [ MZJ*22 ] dataset marks a groundbreaking advancement as the first large-scale scene-level dataset that documents multiple states of articulated objects in indoor settings.
To compile this dataset, a scalable 3D environment acquisition pipeline is designed for processing raw RGB-D scans to produce 3D surface mesh reconstruction with texture and articulation annotations for each articulated object in the scene.
MultiScan is an invaluable resource for advancing research and applications that require an understanding of the dynamics and interactions of articulated objects in real-world environments at the scene level.
Leveraging the MultiScan dataset, OPDMulti  [ SJSC23 ] is introduced for the OPD task of the multiple-parts version by extracting frames from the RGB-D scans and annotating the openable parts in each frame.
It extends the OPD task to the scene level, which is more challenging and practical in real-life scenarios.

 
 
 ParaHome  [ KKNJ24 ] is a recent real-world dataset that captures 3D human motions and interactions with objects within a home environment.
The authors set up a system to capture the human motions with wearable motion capture devices and to track the dynamic interactions with the object synchronized with multi-view RGB cameras.
Articulated objects with multiple parts in the dataset are expressed with parameterized articulations.
This dataset enables new opportunities for human-object interaction studies, which is an important application in the field of embodied AI and robotics.

 
 
 To address the task of articulation estimation from 2D, OPDReal  [ JMSC22 ] dataset is introduced by collecting RGB-D scans from real-world scenes.
By providing a set of real RGB images capturing various articulation states of the objects, along with reconstructed mesh models, OPDReal presents new opportunities for research in the field of articulated object modeling from 2D visuals.
OPDMulti  [ SJSC23 ] follows this trend and extends the task to multiple objects in the scene by collecting RGB-D frames, which is more challenging and practical in real-life scenarios.
As a subtask that tackles a segmentation problem for multiple articulated parts, [ WGYZ25 ] introduces a 2D dataset that consists of 2,550 RGB images captured fromthe real world.
This dataset focuses on articulated part segmentation by leveraging an active learning model to get fine-grained annotation for part segmentation, saving the cost of manual annotation.
However, this dataset does not provide articulation annotations for each part.

 
 
 
 
 
 
 | 
 Input | 
 Input Assumptions | 
 Methodology | 
 Output | 

 
 | 
 geo rep. | 
 # states | 
 # parts | 
 partial/noisy | 
 aligned | 
 part seg. | 
 intermediate rep. | 
 strategy | 
 supervision | 
 geo rep. | 
 arti. part seg. | 

 
 Articulated Part Perception | 

 
 [ XWY*09 ] | 
 mesh | 
 1 | 
 ✗ | 
 ✗ | 
 - | 
 ✗ | 
 surface patch | 
 handcrafted | 
 - | 
 mesh | 
 ✓ | 

 
 [ MYY*10 ] | 
 mesh | 
 1 | 
 ✗ | 
 ✗ | 
 - | 
 ✓ | 
 surface patch | 
 handcrafted | 
 - | 
 mesh | 
 ✗ | 

 
 [ SHL*14 ] | 
 mesh | 
 multi | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 surface patch | 
 handcrafted | 
 - | 
 mesh | 
 ✓ | 

 
 [ YLX*16 ] | 
 PC | 
 multi | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 3D trajectory | 
 handcrafted | 
 - | 
 PC | 
 ✓ | 

 
 [ LWL*16 ] | 
 RGB-D | 
 multi | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 3D trajectory | 
 handcrafted | 
 - | 
 PC | 
 ✓ | 

 
 [ HLV*17 ] | 
 mesh | 
 1 | 
 ✓ | 
 ✗ | 
 - | 
 ✓ | 
 surface patch | 
 - | 
 - | 
 mesh | 
 ✗ | 

 
 [ YHL*18 ] | 
 PC | 
 2 | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 Shape2Motion  [ WZS*19 ] | 
 PC | 
 1 | 
 ✗ | 
 ✗ | 
 - | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 RPM-Net  [ YHY*19 ] | 
 PC | 
 1 | 
 ✗ | 
 ✓ | 
 - | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 [ ATK19 ] | 
 RGB-D | 
 1 | 
 ✓ | 
 ✓ | 
 - | 
 ✗ | 
 image feat. | 
 SL | 
 prim. params. | 
 prim. params. | 
 ✓ | 

 
 ANCSH  [ LWY*20 ] | 
 PC | 
 1 | 
 ✓ | 
 ✓ | 
 - | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 [ SCZ21 ] | 
 PC | 
 multi | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 PC feat. | 
 SSL | 
 PC | 
 PC | 
 ✓ | 

 
 MultiBodySync  [ HWB*21 ] | 
 PC | 
 multi | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 PC feat. | 
 SSL | 
 PC | 
 PC | 
 ✓ | 

 
 CAPTRA  [ WWZ*21 ] | 
 PC | 
 multi | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 [ AFA*22 ] | 
 PC | 
 1 | 
 ✗ | 
 ✓ | 
 - | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 [ QJR*22 ] | 
 RGB | 
 multi | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 image feat. | 
 SL | 
 labeled PC | 
 3D plane | 
 ✓ | 

 
 OPD  [ JMSC22 ] | 
 RGB | 
 1 | 
 ✓ | 
 - | 
 - | 
 ✗ | 
 image feat. | 
 SL | 
 2D mask | 
 2D mask | 
 ✓ | 

 
 OPDMulti  [ SJSC23 ] | 
 RGB | 
 1 | 
 ✗ | 
 - | 
 - | 
 ✗ | 
 image feat. | 
 SL | 
 2D mask | 
 2D mask | 
 ✓ | 

 
 [ LXX*22 ] | 
 RGB-D | 
 1 | 
 ✗ | 
 ✓ | 
 - | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 GAPartNet  [ GXZ*23 ] | 
 RGB-D | 
 1 | 
 ✓ | 
 ✓ | 
 - | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 [ LGW23 ] | 
 PC | 
 multi | 
 ✗ | 
 ✓ | 
 - | 
 ✗ | 
 PC feat. | 
 SSL | 
 PC | 
 PC | 
 ✓ | 

 
 [ LZH*23 ] | 
 PC | 
 1 | 
 ✓ | 
 ✓ | 
 - | 
 ✗ | 
 PC feat. | 
 SSL | 
 PC | 
 PC | 
 ✓ | 

 
 [ LSH*23 ] | 
 PC | 
 1 | 
 ✗ | 
 ✗ | 
 - | 
 ✓ | 
 PC feat. | 
 semi-weakly SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 Banana  [ DLS*23 ] | 
 PC | 
 1 | 
 ✓ | 
 ✗ | 
 - | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 [ WGYZ25 ] | 
 RGB | 
 1 | 
 ✗ | 
 ✓ | 
 - | 
 ✗ | 
 image feat. | 
 SL | 
 2D mask | 
 2D mask | 
 ✓ | 

 
 AutoURDF  [ LZL*24 ] | 
 PC | 
 multi | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SSL | 
 PC | 
 PC | 
 ✓ | 

 
 GAMMA  [ YWL*24 ] | 
 PC | 
 1 | 
 ✗ | 
 ✓ | 
 - | 
 ✗ | 
 PC feat. | 
 SL | 
 labeled PC | 
 PC | 
 ✓ | 

 
 Articulated Object Creation | 

 
 [ PG08 ] | 
 PC | 
 multi | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 PC | 
 handcrafted | 
 - | 
 mesh | 
 ✓ | 

 
 A-SDF  [ MQK*21 ] | 
 SDF | 
 1 | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 neural field | 
 SL | 
 SDF | 
 mesh | 
 ✗ | 

 
 Ditto  [ JHZ22 ] | 
 PC | 
 2 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SL | 
 labeled PC | 
 mesh | 
 ✓ | 

 
 CLA-NeRF  [ TLYS22 ] | 
 MV RGBs | 
 1 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SL | 
 2D mask | 
 mesh | 
 ✓ | 

 
 [ WCM*22 ] | 
 MV RGBs | 
 1 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SL | 
 MV RGBs | 
 mesh | 
 ✗ | 

 
 WatchItMove  [ NIT*22 ] | 
 MV RGBs | 
 multi | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 neural field | 
 SSL | 
 MV RGBs | 
 ellipsoids | 
 ✓ | 

 
 CARTO  [ HIZ*23 ] | 
 MV RGBs | 
 1 | 
 ✓ | 
 ✓ | 
 - | 
 ✗ | 
 neural field | 
 SL | 
 SDF | 
 mesh | 
 ✗ | 

 
 [ LWWY23 ] | 
 - | 
 - | 
 ✓ | 
 - | 
 ✓ | 
 ✓ | 
 convex | 
 TL | 
 - | 
 mesh | 
 ✓ | 

 
 PARIS  [ LMS23 ] | 
 MV RGBs | 
 2 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SSL | 
 MV RGBs | 
 mesh | 
 ✓ | 

 
 SfA  [ NGES23 ] | 
 RGB PC | 
 multi | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SL | 
 PC | 
 mesh | 
 ✓ | 

 
 NAP  [ LDS*23 ] | 
 - | 
 - | 
 ✗ | 
 - | 
 ✓ | 
 ✗ | 
 latent feat. | 
 generative | 
 - | 
 mesh | 
 ✓ | 

 
 CAGE  [ LTMS24 ] | 
 - | 
 - | 
 ✗ | 
 - | 
 ✓ | 
 ✗ | 
 latent feat. | 
 generative | 
 - | 
 mesh | 
 ✓ | 

 
 [ WWT*24 ] | 
 MV RGB-Ds | 
 2 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SSL | 
 MV RGB-Ds | 
 mesh | 
 ✓ | 

 
 REACTO  [ SWF*24 ] | 
 MV RGBs | 
 multi | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SSL | 
 MV RGBs | 
 mesh | 
 ✗ | 

 
 Real2Code  [ MWBS24 ] | 
 RGBs | 
 1 | 
 ✗ | 
 ✓ | 
 - | 
 ✗ | 
 PC, neural field | 
 TL | 
 occupancy | 
 mesh | 
 ✓ | 

 
 RSRD  [ KKW*24 ] | 
 MV RGBs | 
 multi | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SSL | 
 MV RGBs | 
 mesh | 
 ✓ | 

 
 DragAPart  [ LZRV24 ] | 
 RGB | 
 1 | 
 ✗ | 
 ✓ | 
 - | 
 ✗ | 
 image feat. | 
 generative | 
 RGB | 
 RGB | 
 ✗ | 

 
 LEIA  [ SGG*24 ] | 
 MV RGBs | 
 4 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 neural field | 
 SSL | 
 MV RGBs | 
 mesh | 
 ✓ | 

 
 PhysPart  [ LGD*24 ] | 
 - | 
 - | 
 ✗ | 
 - | 
 ✓ | 
 ✗ | 
 latent feat. | 
 generative | 
 - | 
 mesh | 
 ✓ | 

 
 SINGAPO  [ LIC*24 ] | 
 - | 
 - | 
 ✗ | 
 - | 
 ✓ | 
 ✗ | 
 latent feat. | 
 generative | 
 - | 
 mesh | 
 ✓ | 

 
 Articulate-Anything  [ LXL*24 ] | 
 - | 
 - | 
 ✗ | 
 - | 
 ✓ | 
 ✗ | 
 latent feat. | 
 TL | 
 - | 
 mesh | 
 ✓ | 

 
 MeshArt  [ GSLD24 ] | 
 - | 
 - | 
 ✗ | 
 - | 
 ✓ | 
 ✗ | 
 latent feat. | 
 generative | 
 - | 
 mesh | 
 ✓ | 

 
 ArtFormer  [ SFL*24 ] | 
 - | 
 - | 
 ✗ | 
 - | 
 ✓ | 
 ✗ | 
 latent feat. | 
 generative | 
 - | 
 mesh | 
 ✓ | 

 
 Articulate AnyMesh  [ QYW*25 ] | 
 mesh | 
 1 | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 MV RGBs | 
 TL | 
 - | 
 mesh | 
 ✓ | 

 
 ArtGS  [ LJL*25 ] | 
 MV RGB-Ds | 
 2 | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 neural field | 
 SSL | 
 MV RGB-Ds | 
 mesh | 
 ✓ | 

 

 
 Table 2: Summary of the work in articulated object modeling from the geometric perspective.
The table provides information about the geometric representation and the number of articulted states observed in the input, the assumptions made on the input data, the methodology used in the geometry modeling, and the output data representation.
 
 
 
 
 

## 4 Geometry Modeling

 
 In this section, we discuss the recent works on articulated object modeling from the geometry modeling perspective.
Based on the goal of the tasks, we categorize the works into two groups: articulated part perception and articulated object creation .
The former group of work focuses on understanding the articulated structure of the objects from 3D or 2D observations in terms of the articulated part analysis and articulation parameters estimation, and in this section, we focus the discussion on the subtask of shape analysis in each work.
The latter group focuses on reconstructing or generating the 3D shape of objects either with motion parameters associated for each part or with the object animated into different articulation states over time.
In Table   2 , we summarize these works in terms of the choice of geometric representation, the assumptions that each work made for the input, and the methodology each work adopted for the articulated shape analysis.
We specify the meaning of each column in Table   2 as follows:

 
 • 
 
 geo rep. : the geometric representation as input, output, or intermediate representation during processing (under methodology ).

 

 • 
 
 # states : the number of articulation states observed in the input.

 

 • 
 
 # parts : as an assumption, whether the number of articulated parts is known in the input object.

 

 • 
 
 partial/noisy : whether the input geometry can be partially observed (e.g. point cloud projected from the single view depth or even without depth) or noisy (e.g. raw scans with sensory noise).

 

 • 
 
 aligned : as an assumption, whether the object is pre-aligned across different states when multiple articulated states are observed in the input; whether the objects are aligned in a canonical space when there are multiple objects in the input; whether the objects are assumed to be canonicalized for generative models.

 

 • 
 
 part seg. : whether the methods rely on precomputed part segmentation as the input assumption.

 

 • 
 
 intermediate rep. : the intermediate representation used during the processing of the input shape or to represent the shape. Please refer to Section   4.2.1 and Figure   7 for more explanation.

 

 • 
 
 strategy : the methodology adopted for shape analysis, including handcrafted methods, supervised learning (SL), semi- and weakly supervised learning (semi-weakly SL), transfer learning (TL), and self-supervised learning (SSL).
 Handcrafted refers to non-learning-based methods that rely on human-designed descriptors, heuristic rules, or mathematical algorithms to analyze the shape.
 SL refers to the data-driven methods that require ground truth labeled data for training, such as segmentation labels, displacement for each point, etc.
 SSL refers to the data-driven methods that only rely on the input data for training, such as leveraging the geometric consistency between different states of the object.
 Semi-weakly SL refers to a hybrid learning paradigm that combines elements of semi-supervised learning and weakly supervised learning to leverage both limited strongly labeled data, abundant weakly labeled data, and unlabeled data for model training.
 TL refers to the data-driven methods that adapt pre-trained foundational models for a specific target task or use a pre-trained model as a feature extractor to adapt to a related task.

 

 • 
 
 supervision : the source of supervision used for the learning-based methods, such as segmentation labels on the point cloud ( labeled PC ), segmentation mask on images ( 2D mask ), signed distance field ( SDF ) or occupancy field ( occupancy ) of the object, multi-view RGB(-D) images paired with camera parameters ( MV RGBs or MV RGB-Ds ), and unposed RGB images ( RGBs ).

 

 • 
 
 arti. part seg. : the shape of the object is segmented into articulated parts as the output.

 

 
 
 

### 4.1 Geometric Representation for Task Setting

 

#### 4.1.1 Articulated Part Perception

 
 To analyze the part mobility of articulated objects, segmenting articulated parts from an input shape is typically a necessary step.
Exceptions are the early work by [ MYY*10 ] and [ HLV*17 ] which simplified the problem by assuming articulated part segmentation was provided as input,
and ScrewNet  [ JLCN21 ] bypassing explicit segmentation by directly analyzing mobility from a point cloud sequence.
Beyond these, other articulated part perception methods jointly address the articulated part segmentation and mobility analysis from unstructured input shapes, with the exception of [ LSH*23 ] which first leverages fine-grained semantic segmentation to group articulated parts for motion analysis and later uses them as weak labels to enhance perception performance.
In the following, we categorize existing works based on the geometric representation they take as input and discuss the assumptions made about the input data.

 
 
 Figure 5: Method proposed by [ SHL*14 ] for object/part segmentation from a mesh input by relying on the objects/parts being observed in multiple states or poses.
Figure reproduced from original paper  [ SHL*14 ] . 
 
 
 Perception from meshes. 
As the most widely used geometric representation in the industry, polygonal meshes are naturally chosen by early works  [ XWY*09 , MYY*10 , SHL*14 , HLV*17 ] as the initial input representation to work with.
These meshes can be obtained from CAD models that are carefully designed by the artists with part-level structure and sharp edges.
It implies that the input geometry in these works is assumed to be complete and noise-free, which simplifies the problem by observing the object in a geometrically perfect condition in 3D.
 [ XWY*09 ] takes a complex mesh model as input and segments the model into components for joint analysis used for articulate or deform the model.
The proposed method is demonstrated on a range of models, including connected multi-component models, disconnected multi-component models, and single-component models.
 [ MYY*10 ] focus on the motion illustration for mechanical assemblies represented as a set of polygonal meshes that has been partitioned into individual parts.
They assume that each part is modeled as a 2-manifold surface with no self-intersections beyond a small tolerance.
 [ SHL*14 ] pioneered addressing the mobility analysis for indoor scenes represented as unorganized boundary meshes.
They did not make any assumptions on the mesh properties, such as self-intersection or non-manifoldness.
However, segmenting objects and their parts from a scene requires observing state variations in the object and its components, as illustrated in Figure   5 .
 [ HLV*17 ] learns part mobility model from a single snapshot of the object represented as polygonal meshes.
It assumes that the part segmentation is given as input and the method applies on one pair of reference-moving parts at a time.
A recent work by [ QYW*25 ] proposes to convert a static mesh model into its articulated counterparts in a open-vocabulary manner.
It does not assume any pre-computed part segementation and category information given as input.
This approach leverages the recent advances in image segmentation models to assist the 3D segmentation and use VLMs for the part mobility analysis.

 
 
 The main goal of these mesh-based works is to analyze the mobility on artist-created models or manufacturing designs, which can be potentially useful in automating the animation process, illustrating the motion of mechanical assemblies, or creating interactive simulations.
Some challenges are commonly shared among these works, such as the complexity in processing varying mesh structures with irregular topology and the difficulty in converting the mesh into a more compact representation for learning-based methods.
The assumption of having clean and complete geometry in the input also limits the applicability of these methods to real-world object reconstructions, which are often incomplete and noisy.

 
 
 Perception from point clouds. 
Point clouds of the objects can be obtained from 3D scans or depth sensors, which are more accessible and easier to obtain than the polygonal meshes.
This accessibility makes the point clouds a more practical choice for the input representation as it can be directly obtained from the real world.
It encourages more works later that consume point clouds as input for articulated part perception (examples shown in ).
Leveraging this ease of acquisition, a line of work  [ LWL*16 , YLX*16 , SCZ21 , SCZ21 , JLCN21 , WWZ*21 , NGES23 , LGW23 ] takes point clouds that observe a temporal sequence of articulated motion of the object as input.
MultiBodySync  [ HWB*21 ] also processes multiple scans but does not assume a temporal sequence.
These point cloud sequences are often incomplete and noisy, which makes the robustness of the motion fitting an additional critical requirement for the system.
Another implied assumption is that the object is aligned in the same coordinate system across different states, which is a prerequisite for the motion fitting process.
By registering the point positions over time, the points exhibiting consistent motion patterns are grouped as the articulated parts.
The motion patterns can be further analyzed to estimate the articulation parameters for each part.
However, observing a sequence showing object motion can sometimes be unnatural for objects that cannot move on their own.
Consequently, the motion-capturing process typically demands extensive human intervention, making it costly and labor-intensive.
Another challenge in the motion capturing process is how to filter out the irrelevant components in the observed sequence, such as the human hand or robotic arm.

 
 
 Instead of observing motion sequences, [ YHL*18 ] and Ditto  [ JHZ22 ] take a pair of point clouds of different articulation states as input.
 [ YHL*18 ] relaxes several assumptions made by the previous works that the object can be misaligned in two states and the input only needs to be from objects that are functionally related but not necessarily from the same instance.
Ditto assumes the point cloud pair is pre-aligned from the same object and also assumes only one part is moving in the observation.
But it produces the mesh surface with each part segmented as the output to build digital twins of the object.
More recent works  [ WZS*19 , YHY*19 , LWY*20 , LZH*23 , DLS*23 ] tackle the problem from a single state of the object at inference time to make the input more accessible.
 [ AFA*22 ] , [ LXX*22 ] , and GAPartNet  [ GXZ*23 ] further extend the input to colored point clouds, which can leverage more visual information than the raw point clouds.
In tackling the challenge of lacking training data with ground-truth labels, [ LSH*23 ] proposes to leverage data with fine-grained semantic segmentation from PartNet  [ MZC*19 ] as weak labels to augment the training data for the part mobility analysis.

 
 
 Perceiving part mobility from point clouds facilitates the analysis of real-world objects with reduced domain transfer effort compared to mesh-based methods.
Point clouds are easier to obtain from depth sensors, making them a practical choice for many applications.
As a discrete approximation of object surfaces, they are flexible, lightweight, and easier to process with learning-based methods.
However, their lack of connectivity information and fine surface details makes them less informative than meshes, which can capture intricate geometric features more effectively.
This limitation poses a challenge for learning-based methods, as they must extract meaningful features from sparse and unstructured data to accurately understand part structures.

 
 
 Perception from monocular RGB images. 
Even more readily available than point clouds, RGB(-D) images are another common input for articulated part perception.
 [ QJR*22 ] proposed to extract 3D planes to represent articulated parts from RGB videos recording the human-object interaction.
OPD  [ JMSC22 ] first introduces the task of detecting openable parts of objects from a single-view image.
OPDMulti  [ SJSC23 ] extends the scenario to multiple objects in the scene as a follow-up.
 [ WGYZ25 ] further improves the accuracy of the articulated part segmentation from real images by leveraging active learning to easily obtain the ground truth labels and propose a new dataset for the task.
Compared to the aforementioned methods that rely solely on 3D geometry, RGB-based approaches eliminate the need for extensive 3D capturing or reconstruction, making them highly beneficial for real-time applications.
These methods leverage visual cues in images, providing richer appearance details and contextual information about the object.
However, they also face unique challenges, including the difficulty of inferring 3D structures from 2D images, handling occlusions and scene clutter, and compensating for limited or missing 3D information in the input.

 
 
 

#### 4.1.2 Articulated Object Creation

 
 In recent years, significant advances in neural representation techniques  [ MST*21 , WLL*21 , KKLD23 ] , along with large vision models  [ ODM*23 , KMR*23 , WLC*24 ] and generative models  [ GPM*14 , HJA20 , RBL*22 , BDK*23 ] , have opened new possibilities for 3D modeling for articulated objects.
These approaches enable more flexible and detailed representations of articulated objects, which have traditionally posed challenges due to their complex geometries and dynamic characteristics.
This emerging direction in articulated object creation leverages the strengths of both neural geometry representations and generative models, facilitating innovative methods for reconstructing or generating articulated objects across various scenarios.

 
 
 Articulated object reconstruction. 
The goal of articulated object reconstruction is to recover the 3D geometry and motion parameters of the objects from various observations.
On the geometry side, the challenges not only involve decomposing the part into articulated components in the perceived format, but also producing the reconstructed surface that can be animated into different articulation states.
The input modalities can be diverse, including point clouds, multi-view RGB or RGB-D images, and RGB videos.
Certain assumptions are made about the input data based on the target object types and the sparsity of observations.
These include factors such as the number of articulated parts, the object’s alignment across different states, and the completeness of the input data.
The output representation is typically the mesh surface, which is widely used for animation and simulation.
A line of work focuses on reconstructing the whole object as a single deformable surface either over time or given different states  [ MQK*21 , WCM*22 , HIZ*23 , SWF*24 , SGG*24 , KKW*24 ] .
Another line of work reconstructs each part as separate surfaces interconnected by joints to enable articulation  [ TLYS22 , JHZ22 , LMS23 , WWT*24 , MWBS24 ] .

 
 
 Figure 6: Representative methods for articulated object generation.
Figure reproduced from original papers  [ LDS*23 , LTMS24 , LZRV24 , GSLD24 , LIC*24 , LXL*24 ] . 
 
 
 [ PG08 ] pioneered surface reconstruction from sequential point clouds, assuming known part segmentation and skeleton in the first frame.
Building on this idea of observing objects in motion, WatchItMove  [ NIT*22 ] extends the input to multi-view RGB videos and approximates each moving part as an ellipsoid.
RSRD  [ KKW*24 ] models objects in 4D space by taking multi-view RGB images of their static state and a casual RGB video demonstrating human interaction.
Similarly, REACTO  [ SWF*24 ] proposes to reconstruct 4D objects from a single causally captured monocular RGB video of the object in motion.
While these methods accommodate arbitrary kinematic structures by registering different object states, they require dense motion observations.
A-SDF  [ MQK*21 ] was the first to reconstruct articulated objects using signed distance fields from a single snapshot, allowing deformation into different articulation states.
 [ WCM*22 ] share the same goal with A-SDF but take multi-view RGB images as input for the reconstruction, while CARTO  [ HIZ*23 ] further extends the input to a single stereo RGB image.
These works ease the data acquisition process by requiring only a single snapshot of the object, but they assume the object has only one movable part.
LEIA  [ SGG*24 ] reconstructs the interpolated state of the object between the two given input states from multi-view images without knowing the number of moving parts.
However, one limitation of the above-mentioned works is that they all reconstruct the articulated object as a single unified surface or as an implicit deformation field.
The lack of part-level structure in the output geometry makes it difficult to interact with the object in a physically realistic manner.

 
 
 Recent works address this limitation by reconstructing the articulated object in the part level.
SfA  [ NGES23 ] designs a pipeline to reconstruct real-world objects by iteratively interacting with the moving parts and capture the object in the point cloud sequence.
To make the reconstruction more efficient, CLA-NeRF  [ TLYS22 ] proposes to reconstruct the object from a few multi-view RGB images of an unseen 3D object instance within the known category.
It assumes that the object category is known and all the instances in the category share the same articulated structure.
To avoid the reliance on object category, Ditto  [ JHZ22 ] reconstructs from a pair of point clouds, while PARIS  [ LMS23 ] take multi-view RGB images observing the object in two states as input.
 [ WWT*24 ] extends the setting of PARIS to multi-view RGB-D images and is able to reconstruct multiple articulated parts in the output.
These works can reconstruct the object across arbitrary categories, but they all assume the number of articulated parts is known at inference time.
In a more flexible setting, Real2Code  [ MWBS24 ] proposes to reconstruct articulated parts from multiple unposed RGB images of the object in an opening state.
This work focuses on certain categories of objects that mostly have cuboid-like parts but with no assumption on the number of articulated parts.
Capturing the opening state of the object is also helpful in modeling the interior geometry of the object, which is usually occluded in the closed state.
However, this method relies on the object being observed in a state that maximally exposes each part, which may be constrained by its natural configuration or the observation angle.

 
 
 Despite the rapid progress in articulated object reconstruction, existing methods still face several challenges.
Early approaches primarily focused on surface reconstruction from sequential point clouds or dense motion observations but lacked explicit part-level structure, making realistic interaction difficult.
Recent works have addressed this by reconstructing objects at the part level, but they often rely on strong priors such as known object categories, predefined articulation structures, or a fixed number of moving parts.
This highlights a critical gap in the field: the need for methods that can reconstruct arbitrary articulated objects with unknown part structure from sparse and unconstrained observations .

 
 
 Articulated object generation. 
This line of work aims to synthesize the articulated objects in both geometric and kinematic structures from random noise or user-specified constraints.
The representative work is illustrated in Figure   6 .
This is a newly emerging direction of research that leverages the recent advances in neural representation learning and generative modeling in creating high-fidelity and diverse 3D assets.
These generative models learn the underlying distribution of articulated objects, capturing articulated part geometry and articulation configurations to generate novel samples.
NAP  [ LDS*23 ] is a pioneering work that unconditionally generates a full description of an articulated object, including part geometry, articulation graph, and joint parameters.
MeshArt  [ GSLD24 ] tackles the same task but designs a different geometry representation by directly generating the mesh vertices from the latent code.
CAGE  [ LTMS24 ] , SINGAPO  [ LIC*24 ] , and ArtFormer  [ SFL*24 ] introduce different controllable generation setting:
CAGE incorporates high-level structural information (e.g., articulation graph) as user constraints,
SINGAPO enables single-image-driven generation,
and ArtFormer focuses on text-driven generation.
Instead of learning the distribution from scratch, ARTICULATE-ANYTHING  [ LXL*24 ] proposes a general framework that automates the articulation of objects from different input modalities, including text, images, and videos.
Programs are used to arrange the part geometries and predict articulation with a critic mechanism designed to ensure the plausibility of the generated object.
The output of these works is a mesh representation that is either synthesized from the intermediate representation or retrieved from a database using the object context.
A key challenge in this direction is representing articulated objects in a compact and expressive way to capture the joint distribution of part geometry and articulation effectively.
Another challenge is enabling precise control over the generation process , especially for complex, highly structured objects.

 
 
 On a different front, DragAPart  [ LZRV24 ] introduced a method to generate images of articulated objects in different states from a single input image, driven by user-specified drag actions.
While the output remains in 2D space, its articulation is 3D-aware, allowing users to interactively explore the object’s motion in a 2D setting.
This highlights an under-explored yet promising direction: simulating 3D-aware articulation dynamics within a 2D framework.

 
 
 
 

### 4.2 Methodology

 

#### 4.2.1 Intermediate Representation

 
 In this section, we discuss the different choices of intermediate geometric representations used for part mobility analysis or object reconstruction, some of which are visualized in Figure   7 .

 
 
 Articulated part perception. 
In the task of articulated part perception, various intermediate representations are employed to support geometric analysis, either through neural networks or handcrafted methods.
These intermediate representations serve as critical abstractions that simplify the complex structures of articulated objects, making them more tractable for further processing.
Depending on the type of input data, these representations can take different forms, including surface patches for capturing local geometric details, 3D trajectories for tracking motion over time, point cloud features for representing spatial relationships, or image features for extracting visual cues from 2D data.
Each representation is chosen based on the specific demands of the task, and it plays a crucial role in bridging the gap between raw perception data and high-level geometric reasoning.

 
 
 
 • 
 
 Surface patch . The early methods taking mesh as input commonly leverage the surface patches as the proxy for analysis.
The local patches can be detected from slippable analysis as kinematic surfaces  [ GG04 ] , which are used to associate with rigid motions for each part  [ XWY*09 ] .
The surface patches can also be useful for revealing the self-similarity and symmetry on the mechanical part to characterize the motion configuration  [ MYY*10 ] .
The patches at the intersection area between components can be leveraged to determine the supporting-supported relationship  [ SHL*14 ] , which are informative for the part decomposition and structure analysis.
Similarly, the interaction pattern described at the connecting region between surface patches is leveraged in [ HLV*17 ] to measure the similarity between snapshots of the object for motion classification.

 

 • 
 
 3D trajectory . The 3D trajectory is a spatial-temporal representation that records the position of the points in the 3D space over time.
For early works capturing motion with a sequence of point clouds as input  [ PG08 , YLX*16 ] , explicitly extracting the 3D trajectory is a common choice as the intermediate representation to optimize the correspondences between points along the timeline.
Once the trajectory is robustly estimated, the rigid pieces in the object can be grouped based on the transformational consistency of the points.

 

 • 
 
 Point cloud feature (PC feat.) . Point cloud latent features are commonly used as the intermediate representation in the methods that take point clouds as input.
The learning-based methods typically extract spatial point-wise features from the input point cloud and supervise them for the specific objectives, such as part labeling, point registration, or articulation parameter estimation.
PointNet  [ QSMG16 ] , PointNet++  [ QYSG17 ] , and Sparse U-Net  [ GEvdM18 ] are the common feature extractors used as a backbone method in existing works.
For the works that takes point cloud sequence or pair as input  [ YHL*18 , SCZ21 , HWB*21 , WWZ*21 , LGW23 , LZL*24 ] , the point cloud features are used to capture point correspondence for temporal consistency across different states.
For other works that take single state point cloud as input  [ WZS*19 , YHY*19 , LWY*20 , LZH*23 , DLS*23 , AFA*22 , LXX*22 , GXZ*23 , LSH*23 ] , the point cloud features are used to encode the spatial relationships between points for articulated part labeling, while additional heads are usually added to predict the motion parameters or part pose along with the part segmentation.

 

 • 
 
 Convex . The convex is used to approximate the input geometry by representing the part as a convex hull.
 [ LWWY23 ] proposes to learn deformation networks by choosing convexes as intermediate representation to deform each part mesh to generate new objects by few-shot learning.

 

 • 
 
 Image feature (image feat.) . Image latent features are used as the intermediate representation in the works that take RGB or RGB-D images as input.
Similar to the point cloud features, the image features are extracted from pixels for later use in mobility part analysis.
ScrewNet  [ JLCN21 ] follow this scheme to extract 2D features from depth images using ResNet  [ HZRS16 ] instead of per-point features from point clouds.
For the work focused on articulation perception from a single image  [ JMSC22 , SJSC23 , WGYZ25 ] , MaskRCNN  [ HGDG17 ] and Mask2Former  [ CMS*22 ] are the common backbones used to extract feature maps for detecting and masking the parts.

 

 • 
 
 MV RGBs . Articulate AnyMesh  [ QYW*25 ] segments the input mesh by using muli-view images as proxy.
Once the articulated parts are segmented in the rendered multi-view images, the part labels are projected back to the mesh by voting from the corresponding pixels.
It introduces a effective way to leverage the advanced image segmentation models to assist the 3D segmentation task.

 

 
 
 
 Articulated object reconstruction. 
In the context of articulated object reconstruction, neural fields have emerged as the predominant intermediate representation.
A neural field is an implicit function parameterized by a neural network, enabling the continuous and differentiable representation of an object’s 3D geometry, with the optional capability to model its appearance.
Once constructed, the neural field can be queried to reconstruct the objects and further converted into a mesh representation using the marching cube algorithm  [ LC98 ] .
Depending on its formulation, existing works represent the object or their parts with signed distance functions (SDFs), occupancy fields, neural radiance fields (NeRFs), or 3D Gaussians.

 
 • 
 
 Neural SDF . SDFs define the geometry of an object by encoding the signed distance from any point in space to the object’s surface, with positive values outside the object and negative values inside.
A-SDF  [ MQK*21 ] , [ WCM*22 ] , CARTO  [ HIZ*23 ] are the works that leverage neural SDF to represent the object as a singular surface.
The compactness of SDFs makes them effective at capturing smooth surfaces and can be trained to be deformable to animate the object into different articulation states.
However, it becomes challenging to represent multiple parts with discrete surfaces in a single SDF, which limits the ability of this line of work to model articulated objects with multiple parts and complex structures.

 

 • 
 
 Neural occupancy field . Occupancy fields are binary functions that indicate whether a point in space is occupied by the object or not.
Ditto  [ JHZ22 ] constructs a neural occupancy field from a pair of point clouds to represent the object with additional labels for each point in the field to indicate the part association.
Real2Code  [ MWBS24 ] also learns an occupancy-based shape completion network to model part surfaces.
Both methods use occupancy fields to represent articulated objects from point cloud inputs.
Occupancy fields provide flexible representations which effectively handle non-watertight and fragmented structures.
However, like SDFs, occupancy fields also require dense 3D sampling as supervision to capture the surface details, leading to high training costs.

 

 • 
 
 NeRF . NeRF is a continuous volumetric representation that models the object’s geometry and appearance as a function of 3D coordinates.
It is an effective representation that bridges the gap between 2D images and 3D representations via differentiable rendering.
Several work uses NeRFs for articulated object reconstruct from multi-view RGB images  [ TLYS22 , LMS23 , SGG*24 ] , RGB-D images  [ WWT*24 ] or RGB videos  [ SWF*24 ] .
The geometry of each part can be reconstructed by either learning separate NeRFs for each part  [ LMS23 ] or by associating part labels with the 3D coordinates  [ TLYS22 , WWT*24 , SWF*24 ] .
LEIA  [ SGG*24 ] learns the interpolated state of the object between two given states by training a single NeRF.
NeRFs excel at synthesizing realistic novel views of the objects under different articulated states once trained.
But the quality of the extracted surface from NeRFs is easily affected by the sparsity of the training data, textureless regions, occlusions, and lighting conditions.
So they often require densely captured views for accurate reconstruction.

 

 • 
 
 3D Gaussians . 3D Gaussian representation models a scene or object using a set of anisotropic Gaussian functions in 3D space, each defined by position, shape, opacity, and color.
3D Gaussian Splatting is a cutting-edge technique in neural rendering that enables efficient rendering, smooth interpolation, easy dynamic modeling, and high-quality reconstruction of the object.
RSRD  [ KKW*24 ] first leverage this representation to model the dynamics of the articulated object over time as 4D Gaussian fields, which visually imitate the articulation from an object scan in multi-view and a monocular video.
From the multi-view scan, they first reconstruct the static object using 3D Gaussians and cluster the Gaussians into semantic components.
Each component is then tracked over time to capture the dynamics of the object parts by referring to the monocular video in DINO  [ CTM*21 ] feature space.
ArtGS  [ LJL*25 ] leverages 3D Gaussians to reconstruct multi-part articulated objects given two states of the object in multi-view RGBD images.
Compared to other more compact representations, such as SDFs, 3D Gaussians provide flexibility in representing multiple components by associating specific Gaussians with each part, enabling a more explicit way of capturing dynamics and interactions.
3D Gaussian representation presents a promising step toward faster and more efficient reconstruction of articulated objects, which worth further exploration in the future.

 

 
 
 
 Figure 7: Selected examples of the intermediate representation used for articulated part perception and creation.
Figures reproduced from the original papers  [ MYY*10 , LWL*16 , MON*19 , MST*21 , KKLD23 ] .
 
 
 
 The advantage of using implicit fields lies in their continuous and differentiable properties, making them well-suited for applications such as inverse rendering, shape fitting and completion, and deformation modeling.
However, a major drawback is their computational expense, as each scene or object requires overfitting or optimization on a separate network, which limits their scalability and efficiency for large-scale or real-time use cases.

Real2Code  [ MWBS24 ] bridges the 2D and 3D gap with depth estimation model  [ WLC*24 ] and extract point cloud from the depth map.
The point cloud are then labeled as different parts by projecting the segmentation from multi-view images by leveraging SAM  [ KMR*23 ] , which then completes the part surfaces by training a shape completion network.
This pipeline shows its advantage in computational efficiency and effectiveness by taking advantage of recent advances in foundational models.
But this approach relies on minimal occlusion in input images, otherwise the part geometry would be largely incomplete.

 
 
 Articulated object generation. 
In articulated object generation, designing an effective intermediate representation is crucial for capturing the complex structure and articulation of objects.
Given their inherently structured nature, articulated objects can be recorded in formats like URDF or MJCF for simulation in physics engines.
Inspired by this, the intermediate representation for the generation task is designed to encode the part configuration in a hierarchical manner.
A common approach uses the part bounding box to anchor the spatial location of each part, while the surface geometry is compressed into a latent feature space if it is a part of the generation target.
See Figure   6 for a visualization of an example representation proposed in NAP  [ LDS*23 ] .

 
 
 NAP  [ LDS*23 ] pioneers the design of the hierarchical representation as mentioned above.
To model the part surface, they trained an occupancy shape autoencoder  [ MON*19a ] to encode each part surface as a high-dimensional latent code which can be later decoded as SDFs.
CAGE  [ LTMS24 ] and SINGAPO  [ LIC*24 ] represent the parts with only bounding boxes with semantic labels as embeddings during the generation process, and then retrieve the part surface under the context of the generated part layout.
This abstraction strategy allows the model to focus more on the structural arrangement and their interplay with the articulation at the generation stage.
ArtFormer  [ SFL*24 ] further extends the representation from NAP and CAGE.
It proposes to encode the part point cloud into triplane features and quantize the features into a discrete codebook, which is then used as latent code to generate the part geometry along with the structure.
The generated latent code can be decoded into SDFs to represent the part surface.
MeshArt follows the hierarchical representation design but tailors the geometry representation to directly generate the mesh vertices from the latent code.
As illustrated in   Figure   12 ,
in the first generation stage, it represents each part’s bounding box as a triangle mesh and learns a codebook as quantized triangle embeddings.
Guided by the global object structure generated in the first stage, another geometry codebook is learned for encoding the mesh triangles and their likelihood of being junction faces.
This design allows the model to directly generate triangle meshes for articulated parts, resulting in sharp and clean output surfaces.

 
 
 In contrast to the above works that model geometry and structure based on prior learning from training data distributions,
Articulate-Anything  [ LXL*24 ] proposes to represent objects as programs that configure the part composition and articulation parameters.
An actor-critic system is developed to synthesize Python code that can be compiled into URDF files.
By rendering the object synthesized with the programs, the vision-language critic can provide feedback to refine the program generation.
This approach relies on the programmatic representation to ensure the plausibility of the generated object, which can be more interpretable and interactable than the learned latent space.
However, like CAGE and SINGAPO, it relies on a 3D assets library to retrieve the part geometry, which may limit the diversity of the generated objects.

 
 
 On the image-driven generation side, DragAPart  [ LZRV24 ] generates the part-level interaction in image space by encoding the drag-driven animation to the image features.
They teach the pre-trained generative model to respond to drag prompt while enforcing physical consistency with respect to the underlying 3D geometry by fine-tuning the model with paired 2D images with drag annotations.
The motion prior learned by the model can be used to generate the object images in different states by dragging the part in the image.

 
 
 Discussion on intermediate representations. 
The choice of an intermediate representation depends on the input data modality, the application scenario, and the specific requirements of each task.
For perception tasks, it is crucial to capture local geometric features and spatio-temporal relationships that enable part decomposition and motion analysis.
Early work often relied on explicit forms (e.g., surface patches or 3D trajectories), while newer approaches employ implicit latent features from point clouds or images, reflecting a shift from handcrafted to learning-based techniques.
For creation tasks, representations must balance interpretability, computational demands, fidelity, and ease of real-world data acquisition.
Implicit neural fields are popular for their continuous, differentiable nature—valuable in tasks like inverse rendering—but they typically require dense supervision and struggle to represent discrete part structures.
A promising direction is to blend discrete and continuous representations (e.g., 3D Gaussians), which allows modeling piecewise-rigid parts in a differentiable manner.
The combination with the programmatic representation can further enhance the interpretability and controllability of the generated objects, which is beneficial for applications that require human interaction or physical simulation.
These hybrid representations also lend themselves to hierarchical structures, which is especially useful for modeling more structurally complex objects with multiple parts in a kinematic chain.
Along this direction, integrating graph representations or programs as domain-specific languages can be a promising approach to enhance the interpretability and controllability of generated objects with a hierarchical structure.

 
 
 

#### 4.2.2 Strategy and Supervision Signal

 
 In this section, we discuss different algorithms and learning strategies adopted by the works analyzing the articulated part structure.

 
 
 Handcrafted methods. 
To infer part structure from the input geometry, early works rely on heuristic rules, human-designed descriptors, or geometric properties to analyze the shape.

 
 
 For the works that take a single-state mesh as input, the approaches based on the slippage analysis , geometric properties , and relationship descriptors are adopted to effectively segment the parts associated with the articulated motion.
The slippage analysis is proposed by [ GG04 ] to analyze the shape by discovering the slippable motions.
It can segment the geometry into slippable portions which they call kinematic surfaces to associate with rigid motions for each part.
Leveraging this approach, [ XWY*09 ] detected primitive surfaces from the input object and link to motion joints that connect the articulated parts.
Geometric attributes on the mesh surfaces can provide valuable signals for inferring different joint configurations.
Based on this insight, [ MYY*10 ] proposed to characterize the part that related to motions by detecting the geometrical primitives with self-similarity and symmetry on the mesh surface for each mechanical part.
Spatial relationships between the parts are also informative for segmenting the components.
Building on this observation. [ SHL*14 ] proposed to leverage the supporting-supported relationship between surface patches to decompose a scene into a set of objects and their articulated parts.
Similarly, [ HLV*17 ] proposed to leverage the interaction pattern between parts described by a human-designed interaction descriptor  [ HZvK*15 ] .
Based on these geometric features, a distance function is designed to measure the similarity between snapshots of the object.

 
 
 For the works that take point clouds observing a motion sequence as input, geometric alignment algorithms, and handcrafted descriptors are the two common choices to find the correspondence between adjacent frames.
Once the correspondence is established, the movement of each point over time can be tracked to construct the 3D trajectory.
Then the part segmentation can be inferred from the 3D trajectory by clustering the points that are moving together.
Given a known skeleton of the object in the first frame, [ PG08 ] proposed to trace the points that are skinned on each bone in the other frames using the Iterative Closest Point (ICP) algorithm  [ BM92 ] .
Consuming an unlabelled point cloud sequence, [ YLX*16 ] and [ SCZ21 ] proposed to leverage a deformable 3D shape registration algorithm  [ PB11 ] to estimate the 3D trajectory.
Taking RGB-D sequence with additional color information as input, [ LWL*16 ] proposed to use scene flow   [ JSGC15 ] to produce a raw dense correspondence, from which SIFT features   [ Low04 ] are extracted from RGB images to refine and prune the initial proposals in the dense trajectory.

 
 
 Handcrafted methods are excellent in terms of interpretability and computational efficiency, but their performance can be affected by the noiseness and incompleteness of the input data, initialization of the parameters in the algorithms, etc.
With the advent of machine learning and deep learning, handcrafted methods are gradually replaced by learning-based methods, which can automatically learn the underlying features from the data and adapt to the input conditions.
Later works after 2016 resort to data-driven methods to address the articulated object modeling problem, where the common practice converges into two main streams: supervised learning and self-supervised learning.

 
 
 Figure 8: Shape2Motion  [ WZS*19 ] jointly segments the articulated parts and estimates motion attributes from a single point cloud as input.
Figure reproduced from original paper  [ WZS*19 ] . 
 
 
 Supervised learning methods. 
On the perception side, a line of works proposes to learn the part decomposition from a single state of the object in a supervised manner.
A key distinction between articulated part segmentation and semantic part segmentation is that an articulated part may not be semantically atomic and can sometimes be composed of multiple semantic components.
Unlike semantic segmentation, where labels can be consistently assigned across object instances, defining generalizable labels for articulated parts across categories is more challenging.
Shape2Motion  [ WZS*19 ] is the pioneering work that addresses the problem with a proposal-based solution, as illustrated in Figure   8 .
The model learns to regress a similarity matrix between points and a confidence score for each part proposal.
Using ground-truth part grouping, the similarity is supervised to minimize the point-wise feature distance among the points that belong to the same part, while the confidence map is trained to maximize the IoU between the point set of the proposal and its ground truth counterpart.
Building on this idea later works adopt similar feature grouping strategies but explore different network architectures to improve segmentation accuracy and efficiency, such as the recurrent-based network used in RPMNet  [ YHY*19 ] in capture temporal relationships and graph convolutional network used in [ AFA*22 ] to learn the hierarchical part structure.
While these methods demonstrate effective articulated part prediction across object categories, they heavily rely on large-scale labeled point cloud datasets for training.

 
 
 Another line of work focuses on pose estimation for the articulated parts from a single state of the object, where part segmentation is a necessary first step.
Early works make the assumption that the object category is known and all the instances in the category share the same articulated structure.
Under this assumption, [ LWY*20 ] proposed to represent each class of objects in an articulation-aware normalized coordinate space hierarchy (ANCSH) (as shown in Figure   9 ).
The network is trained to map the new object instance to the normalized coordinate space with a branch for assigning the part labels, which are supervised by the per-point ground truth labels.
This method inspired a series of works  [ WWZ*21 , LXX*22 ] that leverage this idea to address the articulated part segmentation and pose estimation on the category level.
The main limitation of these methods is that the assumption of shared part structure across instances in the same category may not hold for all object categories, which limits the generalization capability of the model.
To address this limitation, [ GXZ*23 ] proposed to address the same problem in a category-agnostic manner by introducing generalizable actionable parts (GAPart).
Using these ground-truth GAPart labels as supervision, the trained part segmentation model is more generalizable to unseen object categories.

 
 
 Figure 9: Articulation-aware Normalized Coordinate Space Hierarchy (ANCSH) proposed by [ LWY*20 ] .
ANCSH is a category-level representation that defines canonicalized object state with joint parameters in NAOCS, and normalizes each part in NPCS while maintaining the part orientation in NAOCS.
The colors represent the corresponding coordinates in each space.
Figure reproduced from original paper  [ LWY*20 ] . 
 
 
 In an effort to reduce reliance on articulation annotation available,
Banana  [ DLS*23 ] designs a network to be aware of the inter-part equivariance so that the knowledge acquired from objects only in resting states can be generalized to any other articulated states.
This strategy allows the datasets compiled from static objects with segmentation labels to be usable for articulated part perception.
Another line of work in this category attempts to perceive the part structure from RGB(-D) images.
Given a pair of segmented parts in a single-view RGB-D image, [ ZLLK21 ] proposed FormNet to estimate connectedness and motion flow between the parts in a supervised way.
OPD  [ JMSC22 ] , OPDMulti  [ SJSC23 ] and [ WGYZ25 ] propose to learn 2D segmentations from single-view RGB images in a supervised manner, where the ground truth masks are available from the datasets they collected.
 [ QJR*22 ] first proposed to infer mobility from an RGB video observing the human-object interaction.
They train a network to regress 3D plane parameters to represent the articulated parts by leveraging the supervision from the 3D datasets without articulation annotation.

 
 
 For the task of articulated object creation, supervised learning methods are also widely adopted.
A-SDF  [ MQK*21 ] , [ WCM*22 ] , and CARTO  [ HIZ*23 ] propose a line of work for the object-level reconstruction of a shape that is deformable to a specific state.
The model training is supervised by either ground truth SDF or multi-view RGB images.
These supervision signals guide the model to learn an underlying distribution of the geometry that is associated with the articulation states.
To further achieve part-level reconstruction, Ditto  [ JHZ22 ] requires additional supervision from the part segmentation labels on point clouds, and CLA-NeRF  [ TLYS22 ] and Real2Code  [ MWBS24 ] learn from segmentation masks on multi-view images.

 
 
 Semi-weakly supervised learning methods. 
While part semantics do not always align with mobility, semantic segmentation labels can serve as a valuable source for refining articulated part segmentation.
Inspired by this idea, [ LSH*23 ] contributes a semi-weakly supervised approach to tackle the task of object kinematic motion prediction problem.
Their method leverages the fine-grained and hierarchical part labels from PartNet  [ MZC*19 ] as weak signals to train a graph neural network. It also prunes the input semantic hierarchy and extracts a mobility tree to represent the object’s articulated structure.
Then parts predicted to have motion in the output mobility tree are used as pseudo-labels to expand the training data in the PartNet-Mobility  [ XQM*20 ] dataset for a part mobility prediction network in a semi-supervised manner.
This semi-weakly supervised pipeline demonstrates its advantage by improving model performance through the use of semantic annotations, which are generally more accessible than motion-part annotations.
However, its effectiveness is limited by the granularity of the semantic hierarchy in the source dataset, as a coarse hierarchy may lead to imprecise mobility predictions.
Additionally, the method relies on a sufficient amount of ground-truth labeled data to train the model before generating pseudo-labels for each category, as illustrated in Figure   10 .
The reliability of these pseudo-labels depends on the quality of the motion prediction network, and errors in motion estimation can introduce noise into the training process, potentially affecting overall model performance.

 
 
 Figure 10: [ LSH*23 ] introduces a semi-weakly supervised learning approach that leverages the fine-grained and hierarchical part labels as weak signals to predict mobility trees, which are then used as pseudo-labels to train mobility analysis models.
Figure reproduced from original paper  [ LSH*23 ] . 
 
 
 Self-supervised learning methods. 
Since the articulated part segmentation and articulation annotations are expensive to obtain, self-supervised learning methods have been proposed recently to reduce the reliance on these supervision signals.
Taking a point cloud with a known number of parts as input, [ LZH*23 ] proposes to learn the segmentation by extracting part-level equivariant features and factorizing the shape into canonical states.
Through the iterative processing of canonicalization and then reconstruction using the factorized parameters, part decomposition can be learned by minimizing the reconstruction error on the input point cloud.
PARIS  [ LMS23 ] , [ WWT*24 ] , and ArtGS  [ LJL*25 ] reconstruct articulated objects with part structure from a pair of multi-view RGB(-D) images that observe the object in two different states by leveraging differential rendering.
By identifying the static and moving components through iterative optimization, the part-level geometry can be reconstructed by minimizing the photometric error between the input images and the rendered ones from the reconstructed surface.
Following the trend, LEIA  [ SGG*24 ] proposes to reconstruct the interpolated object between different input states by leveraging a hypernetwork to modulate the network parameters of the reconstructed NeRF.
They encode each state of the object into a latent code which can be linearly interpolated to generate the latent for the intermediate state, which is taken as the input to the hypernetwork to modulate the NeRF.
This optimization process is supervised by the photometric loss to ensure the consistency of the reconstructed object and several regularization terms to enforce a smooth and continuous latent manifold.
REACTO  [ SWF*24 ] and RSRD  [ KKW*24 ] follows a similar idea to reconstruct the articulated objects from videos observing the object in motion.
A photometric loss is used in these methods to learn the neural fields that represent the object with part-level structure.

 
 
 Figure 11: Generative architecture based on diffusion model for unconditional articulated object synthesis from NAP  [ LDS*23 ] .
Figures reproduced from the original paper  [ LDS*23 ] . 
 
 
 Generative models. 
Leveraging the recent advance of diffusion models and autoregressive models in 3D generation, a line of work has focused on training generative models to synthesize articulated objects with compositional part structures.
NAP  [ LDS*23 ] pioneers a novel representation that enables diffusion models to generate articulated objects with variational structures.
As the diffusion model typically requires a fixed-size input, NAP represents the object as a set of parts padded to a fixed number, where each part is described by shape-motion attributes and its geometry encoded as a latent shape code, as shown in Figure   11 .
During generation, all parts are fully connected, and the model jointly synthesizes part attributes while reasoning about their dependencies to form a kinematic tree.
The training follows a denoising diffusion process, where the model minimizes residual noise loss at each step to iteratively refine the generated structure.
Once trained, new articulated object samples can be generated by progressive denoising from random noise.
The denoiser is built on a graph convolutional network (GCN) architecture, which helps model hierarchical part relationships and capture kinematically plausible articulation structures.
This framework showcases the effectiveness of diffusion models for articulated object generation, but the model shows limitations in user controllability over the generation process.

 
 
 CAGE  [ LTMS24 ] and SINGAPO  [ LIC*24 ] share a similar spirit to train diffusion models but focus on improving the controllability on the structure level and the user interaction in the generation process.
These models tokenize part attributes and adopt a transformer-based architecture in the denoising network to capture part relationships via attention mechanisms.
A key innovation is decoupling the part dependency graph from the output and using it as a guidance mechanism to enforce structural consistency throughout the generation process.
CAGE proposes to use varying masking strategies on self-attention layers to enforce the graph conditional input effectively.
Building on top of CAGE, SINGAPO introduces additional cross-attention blocks to inject spatial information into the part structure from an image feature map.
This enables the model to generate multiple plausible articulated objects that are all consistent with an input image, improving the flexibility of image-conditioned generation.
These methods demonstrate the potential of diffusion models for generating articulated objects with controllable structures and user-driven interactions.
However, due to their part retrieval-based formulation, they lack the ability to synthesize diverse part geometries or ensure high compatibility between generated parts and the input image.
Future work could explore integrating shape generation under a structural context, enabling more expressive and detailed synthesis to better align with input constraints and user intent.

 
 
 Figure 12: Autoregressive model for mesh-based articulated object generation from MeshArt  [ GSLD24 ] .
Figures reproduced from the original paper  [ GSLD24 ] . 
 
 
 Along with the generative approach built on diffusion models, another line of work explores generating articulated parts in an autoregressive manner.
MeshArt  [ GSLD24 ] explores the articulated mesh generation in a part-by-part fashion.
The process begins by generating a high-level articulation-aware object structure, which then guides the generation of individual part meshes.
The key insight is to represent both structural information and part geometry as sequences of quantized triangle embeddings, making them well-suited for a unified autoregressive framework that captures articulation and fine-grained geometry jointly.
This design allows the model to generate highly detailed meshes with realistic articulation, but it comes with certain limitations.
The model is trained per object category, restricting its generalization across diverse articulated objects.
Additionally, it assumes that articulation dependencies are limited to direct parent-child relationships (i.e., within one hop in the kinematic tree), which may not fully capture the complex hierarchical articulation structures present in more intricate objects.
Similarly, ArtFormer  [ SFL*24 ] adopts an autoregressive strategy to iteratively generate child parts for each known part.
Each part is represented as a separate token with high-dimensional features, and feature chunks can be decoded into shape and articulation parameters.
On the shape side, each part is represented by a bounding box and a latent code which is trained to be decoded into SDFs.
A key advantage of this tokenization strategy is its simple alignment with text and image tokens, enabling the model to generate diverse articulated objects from user-provided textual or visual descriptions.
This capability bridges text-to-3D synthesis with articulated object generation, offering a more interactive and controllable approach to structured 3D content creation.

 
 
 Transfer learning methods. 
Trained on vast amounts of data, foundation models such as large language models (LLMs) and vision-language models (VLMs), along with general-purpose vision models like SAM  [ KMR*23 ] and DUSt3R  [ WLC*24 ] , have demonstrated significant potential in enhancing downstream tasks.
These models serve as a base that can be adapted to a wide range of tasks with fine-tuning or prompting with a few examples.
A recent line of work leverages these models to advance articulated object creation, utilizing their capabilities in understanding, segmenting, and reasoning about complex structures.
 [ LWWY23 ] tackles the problem of few-shot articulated mesh generation by fine-tuning a convex deformation network pre-trained on part convex decomposition of the objects from similar categories.
Real2Code  [ MWBS24 ] proposes a framework to reconstruct articulated parts from multi-view images by fine-tuning SAM for segmenting articulated parts, and then lifting each part to a 3D point cloud using DUSt3R.
Articulate-Anything  [ LXL*24 ] proposes an actor-critic system to generate articulated part configurations using programs.
Multiple VLM and LLM agents are prompted to specialize in different subtasks, such as object detection and reasoning from the visual input, part mesh retrieval and arrangement, rating the realism of the output, and providing specific feedback to correct the output.
DragAPart  [ LZRV24 ] generates images of articulated objects constrained by a drag action by fine-tuning the stable diffusion model  [ RBL*22 ] with paired drag-annotated images.
Although trained on synthetic images, the model can generalize well to real images by leveraging the text-to-image prior in the pre-trained diffusion model.
Articulate AnyMesh  [ QYW*25 ] proposes a pipeline to segment the input mesh by aggregating 2D segmentation results from multi-view images.
The 2D articulated part segmentation is achieved by few-shoting the PartSlIP++ model  [ ZGL*23 ] using a few example masks for each category.
It effectively leverages the advanced image segmentation models to assist the 3D segmentation task on surface meshes.
For the occluded geometries, a refinement step is followed, which distills 3D geometry-texture priors from a pre-trained RichDreamer model  [ QCG*24 ] using SDS loss  [ PJBM22 ] .

 
 
 These transfer learning based methods typically design a multi-stage pipeline that brings together the strengths of different foundation models to tackle each subproblem in the object creation process.
Benefiting from powerful prior knowledge encoded in the base modules, these methods can potentially reduce the need for extensive annotated data to train a model from scratch, but only require a few examples to prompt the model or even used in zero-shot settings.
However, the performance of these methods is highly dependent on the generalization capability and reliability of the base models.
How robust the model can be adapted to the target task and how well it can generalize to unseen data are still open questions that need further exploration.

 
 
 
 
 

## 5 Articulation Modeling

 
 The articulation model of an object is a representation for describing the part mobility and the relationship between the parts.
Understanding the kinematic structure of an object is a shared goal for both tasks of articulated part perception and articulated object creation.
One of the key challenges in articulation modeling is how to deal with objects with different kinematic structures, such as the number of joints, the type of joints, the degrees of freedom (DoF) of the joints, and what hierarchy the joints are organized.
In  Table   3 , we summarize the related works from the perspective of articulation modeling in terms of the assumptions made on the articulation model, the representation of articulated motion, and the methodology used for articulation modeling.
In the following, we will discuss each aspect in detail.

 
 
 
 
 
 
 | 
 Assumption | 
 Articulated Motion Representation | 
 Methodology | 

 
 | 
 # joints | 
 # DoF | 
 obj. cat. | 
 joint type | 
 joint axis | 
 joint state | 
 motion range | 
 kine. tree | 
 deform. flow | 
 strategy | 
 supervision | 

 
 Articulated Part Perception | 

 
 [ XWY*09 ] | 
 ✗ | 
 3 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 handcrafted | 
 - | 

 
 [ MYY*10 ] | 
 ✓ | 
 3 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 handcrafted | 
 - | 

 
 [ SHL*14 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 handcrafted | 
 - | 

 
 [ LWL*16 ] | 
 ✗ | 
 3 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 handcrafted | 
 - | 

 
 [ HLV*17 ] | 
 ✓ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 SL | 
 distance func. | 

 
 [ YHL*18 ] | 
 ✗ | 
 3 | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 SL | 
 displacement | 

 
 [ WZS*19 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 RPM-Net  [ YHY*19 ] | 
 ✗ | 
 arbitrary | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 SL | 
 displacement | 

 
 [ ATK19 ] | 
 ✓ | 
 1 | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 ANCSH  [ LWY*20 ] | 
 ✓ | 
 1 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 CAPTRA  [ WWZ*21 ] | 
 ✓ | 
 1 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 [ SCZ21 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 SSL | 
 point corr. | 

 
 ScrewNet  [ JLCN21 ] | 
 ✓ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 MultiBodySync  [ HWB*21 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SSL | 
 point corr. | 

 
 [ AFA*22 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 SL | 
 joint params. | 

 
 [ QJR*22 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 OPD  [ JMSC22 ] | 
 ✓ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 OPDMulti  [ SJSC23 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 [ LXX*22 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 SL | 
 joint params. | 

 
 [ LZH*23 ] | 
 ✓ | 
 1 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SSL | 
 point corr. | 

 
 [ LGW23 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 SSL | 
 point corr. | 

 
 GAPartNet  [ GXZ*23 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 NPCS maps | 

 
 [ LSH*23 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 SL | 
 joint params. | 

 
 AutoURDF  [ LZL*24 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 SSL | 
 point corr. | 

 
 GAMMA  [ YWL*24 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 Articulated Object Creation | 

 
 A-SDF  [ MQK*21 ] | 
 ✓ | 
 1 | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 SL | 
 joint params. | 

 
 [ WCM*22 ] | 
 ✓ | 
 1 | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 SL | 
 joint params. | 

 
 Ditto  [ JHZ22 ] | 
 ✓ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 CLA-NeRF  [ TLYS22 ] | 
 ✓ | 
 1 | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 handcrafted | 
 - | 

 
 CARTO  [ HIZ*23 ] | 
 ✓ | 
 1 | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 joint params. | 

 
 [ LWWY23 ] | 
 ✓ | 
 1 | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 TL, handcrafted | 
 - | 

 
 PARIS  [ LMS23 ] | 
 ✓ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SSL | 
 MV RGBs | 

 
 SfA  [ NGES23 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SL | 
 labeled PC | 

 
 NAP  [ LDS*23 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 generative | 
 joint params. | 

 
 CAGE  [ LTMS24 ] | 
 ✗ | 
 1 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 generative | 
 joint params. | 

 
 [ WWT*24 ] | 
 ✓ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SSL | 
 MV RGB-Ds | 

 
 REACTO  [ SWF*24 ] | 
 ✗ | 
 arbitrary | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 SL | 
 optical flow | 

 
 Real2Code  [ MWBS24 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 TL | 
 joint params. | 

 
 RSRD  [ KKW*24 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SSL | 
 MV RGBs, video | 

 
 SINGAPO  [ LIC*24 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 generative | 
 joint params. | 

 
 Articulate-Anything  [ LXL*24 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 TL | 
 - | 

 
 MeshArt  [ GSLD24 ] | 
 ✗ | 
 1 | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 generative | 
 joint params. | 

 
 ArtFormer  [ SFL*24 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✗ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 generative | 
 joint params. | 

 
 Articulate AnyMesh  [ QYW*25 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✗ | 
 TL, handcrafted | 
 - | 

 
 ArtGS  [ LJL*25 ] | 
 ✗ | 
 1 | 
 ✗ | 
 ✓ | 
 ✓ | 
 ✓ | 
 ✗ | 
 ✗ | 
 ✗ | 
 SSL | 
 MV RGB-Ds | 

 

 
 Table 3: 
Summary of the related works from the perspective of articulation modeling in terms of the assumptions about articulation structure each work makes, the representation of articulated motion, and the methodology used in the motion estimation process.
 
 
 

### 5.1 Representations of Articulated Motion

 
 Joint parameters. 
The most common way to represent part articulation is through joint parameters, which define the motion properties of articulated parts in terms of joint type, axis, state, and limits.
This representation is widely used in robotics and simulation, as it provides a structured way to model how parts move relative to each other.
The joint types most commonly seen are the 1 DoF joints, such as the revolute joint and the prismatic joint.
The joint with more than 1 DoF, such as the ball joint, can also be seen in more complex objects, e.g. mechanical parts.
An overview of the joint types that are covered in the literature is shown in  Figure   1 .
The joint axis consists of a direction vector for orientation and a position vector for the pivot.
The joint state captures its current configuration, typically as a rotation angle (revolute) or translation distance (prismatic).
Joint limits define the allowed motion range to ensure physical plausibility.

 
 
 The line of work on mobility analysis formulates joint parameter estimation as a combination of classification and regression problems, depending on the task requirements.
Typically, joint type, position, and orientation are the primary components to be estimated.
Some works additionally estimate the joint limit  [ XWY*09 , MYY*10 , HLV*17 , LWY*20 , SCZ21 , QYW*25 ] , which can be more challenging as it requires the network to understand the physical constraints of the object.
Other works focus on estimating the pose of the articulated parts from 3D observation of objects  [ ATK19 , LWY*20 , WWZ*21 , LXX*22 , LZH*23 , GXZ*23 ] .
Part pose is typically represented as a 3D amodal bounding box with a free 6D pose.
By mapping the estimated pose to a normalized coordinate space introduced by [ LWY*20 ] , each part pose can be further converted into the joint state.

 
 
 Kinematic tree. 
The kinematic tree serves as a hierarchical representation of the joints and parts of an articulated object.
This structure is particularly crucial for depicting objects with multiple parts and joints, as it clearly outlines the dependencies between them
An example of the kinematic tree is shown in  Figure   3 .
Extracting the kinematic tree is a challenging problem since the kinematic structure varies from object to object with different numbers of parts and is structured in different topologies.
Some existing works bypass this step by assuming the kinematic tree is predefined per object class  [ ATK19 , LWY*20 , LZH*23 , MQK*21 , WCM*22 ] , or by estimating only one joint at a time  [ HLV*17 , JLCN21 , JHZ22 , TLYS22 , HIZ*23 , LMS23 ] .
And a large portion of the works dealing with multiple joints also do not represent the kinematic tree in their model  [ YHL*18 , WZS*19 , YHY*19 , SCZ21 , SJSC23 , GXZ*23 ] as they assume that all the articulated parts move relative to the static part.

 
 
 There are only a few works that explicitly model the kinematic tree as the output of the mobility analysis.
The early works by [ MYY*10 ] and [ SHL*14 ] extract the kinematic tree from the input mesh.
They share a similar idea to leverage the intersection of the part surfaces to identify the connection between the parts.
 [ MYY*10 ] detected the contact area between the parts to determine the topology of the interaction graph, and [ SHL*14 ] applied a segment-cluster process to compute the support-tree hierarchy from decomposed parts.
Later works by [ AFA*22 ] and [ LXX*22 ] attempt to predict the kinematic tree from the input point cloud by labeling the dependency existence between each pair of segmented parts.
Articulate AnyMesh  [ QYW*25 ] predicts the kinematic dependency using VLMs.
For the work focusing on articulated object generation  [ LDS*23 , LTMS24 , LIC*24 , LXL*24 , GSLD24 , SFL*24 ] , the kinematic tree is an essential output, as their ultimate goal is to synthesize fully articulatable assets that can be directly integrated into the simulation environment.

 
 
 Deformation flow. 
Deformation flow is a general form of motion representation that can describe not only the rigid motion but also the motion as free-form deformation, making it highly versatile for modeling articulation.
It describes the motion as a vector field, where each vector denotes the spatial displacement for each point in space.
Unlike traditional joint-based representations, deformation flow allows for more flexible motion modeling, especially when the motion involves non-trivial transformations.

 
 
 For mobility analysis, [ YHL*18 ] and RPM-Net  [ YHY*19 ] opt to use deformation flow to represent the articulation motion.
They both leverage recurrent neural networks, such as LSTM, to directly predict the deformation flow from the input point cloud.
An example of the deformation field prediction network is illustrated in  Figure   13 .
One of the advantages of using deformation flow is that it can handle the motion of the parts in a free-form manner.
This flexibility is useful when the number of parts is unknown as the deformation on each point can be later grouped to form articulated parts in arbitrary numbers.
The deformation flow can be particularly useful when modeling the non-trivial motion.
For example, RPM-Net can model the opening and closing motion of the cover of an umbrella, which is a non-linear motion that cannot be represented by a simple rigid transformation.

 
 
 Similarly, A-SDF  [ MQK*21 ] , [ WCM*22 ] , and REACTO  [ SWF*24 ] also leverage the deformation field to represent articulation for the objects in the task of articulated object creation.
These methods optimize coordinate-based networks to implicitly represent the deformation, conditioned on the articulation state of the object.
Once trained, the network allows the reconstructed object to be deformed into different states by querying the neural field at different timesteps, enabling continuous articulation modeling.

 
 
 However, a key limitation of deformation flow or fields is that they are less constrained than strict rigid transformations, which can be problematic when dealing with rigidly moving parts.
To alleviate this issue, [ YHL*18 ] proposes to optimize the deformation flow with an as-rigid-as-possible (ARAP) objective  [ SA07 ] to preserve the local rigidity of the parts.
RSRD  [ KKW*24 ] estimates articulation by tracking the pose for each cluster of 3D Gaussians over time in an input video.
By treating 3D Gaussians as soft points in space, an ARAP loss penalizes deviation between neighboring Gaussians, ensuring that deformation follows locally rigid motion.
This approach bridges deformation-based articulation modeling with Gaussian-based representation learning, demonstrating the potential for more flexible yet structured articulation tracking.

 
 
 Discussion on articulation representations. 
Joint parameters, kinematic trees, and deformation fields are three common ways to represent articulation, each with its own strengths and challenges.
Joint parameters are fundamental for describing discrete motion constraints and are widely used in robotics and simulation.
Kinematic trees define the hierarchical structure of articulation dependency, which complements joint parameters to form a complete articulation model while also serving as a crucial component for simulation-ready articulated assets.
While often overlooked due to the simplicity of many articulated objects with single-level structures, kinematic trees become essential when handling objects with deeper articulation hierarchy (e.g., table lamps with multiple arms, robotic mechanisms, mechanical assemblies).
Modeling such structures introduces additional challenges for mobility analysis and articulated asset creation, as it requires reasoning over highly structured data.
In contrast, deformation fields offer a continuous and flexible representation, capable of capturing both rigid and non-rigid motion.
However, they lack explicit structural constraints when modeling piece-wise rigidity, usually requiring additional regularization techniques to enforce local rigidity.

 
 
 Figure 13: An illustration of using a recurrent neural network to predict deformation field for each articulated part from RPMNet  [ YHY*19 ] .
Figure reproduced from original paper  [ YHY*19 ] . 
 
 
 

### 5.2 Assumption on the Articulation Model

 
 There are several axes of assumptions made on the articulation model by the existing works.
These assumptions usually are made to simplify the problem to make it more tractable.
We summarize these common assumptions to show the research progress below.

 
 
 The number of degrees of freedom (# DoF) .
Most of the existing works assume to only deal with the 1 DoF joints, including the revolute joint, the prismatic joint, and the helical joint.
These joints are the most common types of joints that are seen in the daily articulated objects, such as cabinets, dishwashers, refrigerators, etc.
And their motion is constrained to be linear and easily represented by a combination of rotation and translation constrained by a single joint axis.
When dealing with more complex objects, such as delicate mechanical systems, the joint with more than 1 DoF, such as the ball joint, can also be seen.
A few methods  [ XWY*09 , MYY*10 , LWL*16 , YHL*18 ] have been proposed to deal with joints up to 3 DoF, which is achieved by either leveraging the deformation field to represent the motion or by combining with the handcrafted features on the part geometry to classify the joint type.

 
 
 Known object category (Obj. cat.) 
As mentioned above, one of the main challenges in articulation modeling is the diversity of the kinematic structure of the objects.
One way to generalize the knowledge learned from the training data to the unseen objects is to build a category-level model, which is based on the assumption that the objects in the same category share the same structures.
The structure implied by this line of work  [ ATK19 , LWY*20 , LZH*23 ] is not only about the kinematic structure but also the geometric structure.
It means that the number of parts, how the parts are arranged spatially, and the way each part is articulated are all shared among the objects in the same category.
This presents a strong prior for the model to learn from.
In some cases, this level of category can be aligned with the semantic class of the objects whose kinematic structure is relatively simple and fixed, such as eyeglasses, laptops, scissors, etc.
However, this assumption is highly constrained since this is not the case for more objects with more complex structures.
For example, a refrigerator with two doors stacked vertically and a refrigerator with two doors stacked horizontally are considered to be from the same semantic class but with different structures.
A separate model should be trained for each kind of refrigerator under this assumption.
It makes the model less scalable to real-world applications with a diversity of objects.

 
 
 Known number of joints (# Joints) .
Whether the number of joints is known is another assumption made by the existing works.
The previous assumption of knowing the object category inherently implies that the number of joints is known.
There are also a few works  [ HLV*17 , JLCN21 , JHZ22 , TLYS22 , HIZ*23 , LMS23 ] that assume to estimate only one joint at a time but not knowing the motion the joint exhibits.
It can be useful to generalize the method across objects with varied kinematic ways for simple objects.
However, it is not scalable to more complex objects with more than one joint, highlighting a limitation in this line of work.

 
 
 

### 5.3 Methodologies for Articulation Modeling

 
 Handcrafted methods .
Early works leverage non-learning-based methods to infer the articulation model.
 [ XWY*09 ] applied slippage analysis to compute a set of joint parameters from a mesh input.
The output includes the number of rotational and translational degrees of freedom which is thresholded to be up to three, and the corresponding axes for each slippable motion.
To determine the motion range for each joint, they design a trial-and-error bisection process to decide the feasibility.
They estimate the joint limit by probing the possible motions until the motion results in an unrealistic geometry, e.g. penetration occurs between the parts.
 [ MYY*10 ] define a list of heuristic rules to determine the joint type, axis, and limit for each mechanical part.
Once detecting the supporting-supported relationship between parts, [ SHL*14 ] extract this connectivity to form a kinematic tree.
By leveraging the reoccurrence of the objects and parts in the scene with different poses, they design rules to determine the motion type and axes by using the PCA axes of the part geometry.
Taking a point cloud sequence as input, [ LWL*16 ] propose a random sampling consensus method in 4D to fit motion on 3D trajectories.
Then these motions can be clustered to convert to the joint parameters mathematically and the joints can be organized in a mobility graph.
CLA-NeRF  [ TLYS22 ] assumes the object has a single revolute joint whose axis is at the intersection of the two articulated segments.
So once the parts are segmented in the NeRF, the joint axis can be directly voted by the points near the intersected area by applying linear regression.

 
 
 Figure 14: Overview of the ANCSH method proposed by [ LWY*20 ] that maps the object at an arbitrary articulated state to a hierarchical normalized coordinate space.
Figure reproduced from original paper [ LWY*20 ] . 
 
 
 Supervised learning .
A recent line of work focusing on mobility analysis leverages supervised learning to predict the articulation model from a single state of the object.
Depending on the articulation representation chosen and perception input, the training process can be supervised by the ground truth motion parameters  [ WZS*19 , JLCN21 , QJR*22 , NGES23 ] or point displacements  [ YHL*18 , YHY*19 ] .
As a representative work shown in Figure   8 , Shape2Motion  [ WZS*19 ] designs a motion attribute proposal module to decode point-wise features into parameters for all the possible articulation in the object.
Rather than directly regressing the joint axis, they reformulate joint location prediction as a point selection problem and the joint orientation problem as a classification task over 14 discrete directions, which reduces the complexity of direct regression in high-dimensional continuous spaces.
Once each motion is proposed, a matching module follows to pair articulations with the corresponding parts, allowing for cases where a single articulated part may exhibit multiple 1-DoF motions.
Different from the above works that explicitly use motion annotation, [ HLV*17 ] propose to classify the motion type from a mesh input through a metric learning model.
The training is supervised by a distance function with several constraints to pull the object pairs with the same motion type closer and push the ones with different motion types away.

 
 
 Another line of work tackles the task of articulated part pose estimation from the depth image of an object in an arbitrary articulation state in a supervised manner.
By assuming the articulation structure is shared within each object category, [ LWY*20 ] pioneers in designing an articulation-aware normalized coordinate space hierarchy (ANCSH), which is used to canonicalize the perceived object at the part level.
As shown in Figure   9 , ANCSH is composed of a Normalized Articulated Object Coordinate Space (NAOCS) on top of a set of Normalized Part Coordinate Spaces (NPCSs) per part, and the joint parameters in the resting state have been pre-defined in NAOCS.
Once this hierarchical representation is defined for each object category, a deep neural network (illustrated in Figure   14 ) is trained to predict the mapping from each point to a coordinate in NPCS, followed by a head to predict the transformation from each NPCS to NAOCS and another head to predict the point-joint association and joint parameters in NAOCS.
The training is supervised by multiple losses using ground truth outputs, including per-point NPCS and NAOCS mapping, per-part transformations, joint parameters, etc.
Leveraging this ANCSH representation, CAPTRA  [ WWZ*21 ] and GAPartNet  [ GXZ*23 ] extend the problem setting to pose tracking and domain-generalizable object perception.

 
 
 Self-supervised learning .
Although supervised learning is effective in learning the articulation model, a large amount of labeled data is required to train the model.
To alleviate this issue, a few works  [ SCZ21 , HWB*21 , LZH*23 , LZL*24 ] propose to leverage self-supervised learning to estimate the articulation model by leveraging the correspondence across multi-state observations of the object.
 [ SCZ21 ] leverage a network to learn the point correspondence across different frames of the point cloud sequence from the input.
Once the 3D trajectories are constructed by learning the consistent point correspondence, the joint parameters can be estimated by fitting the motion on the trajectories.
One limitation of this work is that it requires observing a motion sequence as input, which usually involves human intervention while capturing the data.
This tedious motion-capturing process makes the method less scalable to real-world applications.
To alleviate this issue, [ LZH*23 ] propose to learn the joint parameters from a single point cloud in a self-supervised manner.
By factorizing each articulated part to its own normalized canonical space and then reconstructing it back to the input state using an optimized transformation, the reconstruction consistency in this process can be used to supervise the training.
However, this work is limited by the assumption that the model is category-specific, meaning the geometric and kinematic structure of the object is known in advance.

 
 
 Figure 15: An illustration of a self-supervised learning method proposed in PARIS  [ LMS23 ] that uses photometric consistency to jointly optimize part geometry and joint parameters .
Figure reproduced from original paper  [ LMS23 ] . 
 
 
 In the task of articulated object reconstruction, PARIS  [ LMS23 ] proposes to learn the joint parameters from a pair of multi-view RGB images in a self-supervised manner. As illustrated in Figure   15 , the estimation of the joint parameters is supervised by the consistency of a photometric loss across two articulated states given as input.
Although this method can be generalizable across object categories and does not rely on any 3D or articulation supervision, it assumes only one part is moving in the input observation.
 [ WWT*24 ] and ArtGS  [ LJL*25 ] follows a similar idea to optimize the joint parameters from a pair of observations in multi-view RGB-D images.
Reconstruction losses on the RGB-D frames are used to first segment the parts in two states of the object, and then the joint parameters can be computed by optimizing the point correspondences between the segments from the two states.
RSRD  [ KKW*24 ] models the object dynamics from human demonstration videos using a photometric consistency loss and regularization terms to constrain the motion of the parts with as-rigid-as-possible (ARAP)  [ SA07 ] penalty.
This work tracks the articulated part in 4D poses along the time steps, so it can be transferred to a physical grasping robot to execute these motions.

 
 
 Generative methods .
As mentioned in Section   4.2.2 , there is a line of work  [ LDS*23 , LTMS24 , LIC*24 , GSLD24 , SFL*24 ] that trains generative models from scratch to learn the joint distribution of geometry and articulation for each part.
On the articulation side, all existing methods model the motion properties explicitly through joint parameters.
The most common approach  [ LTMS24 , LIC*24 , GSLD24 , SFL*24 ] is to parameterize joints with joint location, orientation, and motion limits, with some methods also including joint type  [ LTMS24 , LIC*24 ] and a joint existence indicator  [ LDS*23 , GSLD24 ] .
NAP  [ LDS*23 ] adopts Plucker coordinates for joint representation, using a unit direction vector l l and momentum m m perpendicular to l l , providing a unified representation for prismatic and revolute joints.
Notably, all the above joint parameters are defined in the global object frame, which is consistent with the part shape representation.
During training, the loss function is designed to enforce consistency with the ground-truth joint parameters per part.

 
 
 However, an open question remains as to whether a global-frame or local-frame parameterization is more efficient in the generative context.
The global-frame representation captures the distribution across different objects in a coordinated manner, but it is sensitive to the object’s orientation and articulation states.
A more local representation can potentially capture the connection within the part geometry in a more compact way.
Exploring alternative formulations to compactly encode joint parameters alongside part geometry is a promising direction for future research, leading to more scalable and generalizable generative models for articulated objects.

 
 
 Figure 16: The actor-critic program system for link placement and joint prediction proposed in Articulate-Anything  [ LXL*24 ] . Figure reproduced from the original paper  [ LXL*24 ] . 
 
 
 Transfer learning methods. 
Real2Code  [ MWBS24 ] and Articulate-Anything  [ LXL*24 ] are two recent works that leverage pre-trained LLMs to transfer language reasoning and code generation capabilities to the articulation understanding problem.
These methods bridge the gap between articulation modeling and generative programming, enabling articulated object representations that are both interpretable and simulation-ready.
Real2Code abstracts the part layout as oriented bounding boxes after segmenting articulated parts.
It then defines joint parameters relative to these bounding boxes, formulating joint prediction as a classification problem.
Taking the bounding boxes as input, they fine-tune CodeLlama  [ RGG*23 ] to predict the joint parameters and generate code to construct the objects into a compact representation that can be directly used in physics simulations.
In contrast, Articulate-Anything builds a joint prediction system on VLMs.
It takes code representations that configure link placements as input and generates kinematic parameters between parts.
The prediction is iteratively refined by taking feedback from another program that criticizes the realism of the results by comparing it with the input image or video.
Both approaches demonstrate the potential of LLMs and VLMs in articulation reasoning and code-based object generation.
Articulate AnyMesh  [ QYW*25 ] also leverages VLMs to analyze the part articulation.
Based on some heuristic rules, they prompt GPT-4o to make selections from several candidate markers on the rendered images or possible joint directions, depending on the joint type.

 
 
 Figure 17: Example sources of human-in-the-loop video: human interactions, egocentric videos, and human demonstrations videos that involve articulated objects.
The figures are reproduced from the original papers  [ QJR*22 , GWB*22 , KKW*24 ] . 
 
 
 
 

## 6 Discussion and Conclusion

 
 This survey paper provided an overview of the current state-of-the-art in 3D modeling of human-made articulated objects, focusing on the tasks of articulated part perception and articulated object creation (reconstruction and generation).
We first defined the scope of the discussion and provided a compilation of articulated object datasets used in the research community.
We then analyzed techniques and methodologies developed for 3D modeling of articulated objects, focusing on two key aspects: 1) geometry modeling — representation and approaches used to understand the shape of articulated parts and 2) articulation modeling — articulation models and methodologies used to estimate part mobility and kinematic structures.
We highlighted gaps in the literature and opportunities for further work emerging from our survey, which are likely to be fruitful as the field of 3D modeling for articulated objects continues to evolve.
We conclude by identifying key challenges and potential research directions for future exploration.

 
 
 Generalization across objects with varying structures .
Handling diverse articulated objects with large variations in geometric and kinematic structures is one of the major challenges in the field.
Existing methods are often designed for specific object categories or assume a certain level of prior knowledge about the object is available to the system (e.g., the number of parts, which joint types are considered).
Given limited training data and the high cost of data collection, it is important to develop methods that can generalize to novel objects in a data-efficient manner.
A promising direction is to explore the use of few-shot learning, meta-learning, and domain adaptation techniques to enhance the versatility of articulated object understanding.
Recent advances in large language models, vision-language models, and multimodal large language models can also be powerful prior to integrate into articulated object perception and manipulation pipelines, driving improvements in the generalization and robustness of the models.

 
 
 Improved robustness in the perception models .
While significant progress has been made in articulated part perception, more accurate and robust methods are needed to handle occlusions, clutter, and noise in input data.
Most existing approaches focus on object-level perception with clear, object-centric observations, whereas perceiving articulated objects in complex scenes with multiple objects and partial observations remains an open challenge  [ DTT*24 ] .
Addressing this gap is essential for making these methods practical and deployable in real-world applications, including functionality understanding, affordance learning, and robotic manipulation, where models must reliably interpret and interact with articulated objects in unstructured environments.

 
 
 Detailed interior and part geometry modeling .
Modeling interior and part-level geometry in detail for articulated objects and further composing such detailed interactable objects at the scene level is an important aspect of geometric modeling that has not been fully explored.
For the object reconstruction task, the quality of the models is often limited by the quality of the input observation which is often partial due to view occlusion and sensor noise.
Developing methods that can handle these challenges and generate intricate and accurate 3D models of the articulated object (e.g., fully functional cars, fridges with shelves and containers allowing small objects to be placed inside) is another promising direction for future research.

 
 
 Generative AI for articulated objects .
The recent advances in articulated object generation have shown great potential in creating realistic and diverse 3D models of articulated objects.
Due to the highly structured nature and complexity in both geometry and articulation at the part level, generative models for articulated objects face unique challenges compared to general object generation.
The existing methods either prioritize structure modeling or geometry modeling, and the integration of both aspects while maintaining kinematic plausibility remains an open challenge.
Key challenges include designing efficient representations with compatible generative paradigms to capture the multi-faceted nature of articulated objects, and developing controllable models that can be prompted with different user inputs.
Additionally, data scarcity limits progress, as high-quality datasets with detailed part geometry and articulation annotations are rare, making it difficult to train scalable and generalizable models.
Promising future directions include expanding datasets, improving the representation learning, enhancing the multi-modal generation capabilities, and advancing controllable generative models for interactive design.
Another interesting venue is exploring domain-specific language models for articulated objects in combination with large language models.
The compositionality and the hierarchical structure of the articulated objects make them well suited for code- or program-based descriptions, and the integration of language priors can potentially improve the generation diversity and the model interpretability.
Ultimately, integrating generative AI with simulation environments will be essential for advancing interactive applications and embodied AI research, enabling more realistic modeling and interaction with everyday articulated objects.

 
 
 Incorporating physical constraints into the modeling process .
Articulated objects are subject to physical constraints (e.g., collisions, materials) and physical properties (e.g., mass, inertia), which are often overlooked in existing modeling approaches.
Most current generative and perception models focus on capturing articulation geometry and motion patterns but do not explicitly enforce real-world physics, leading to unrealistic or non-functional outputs.
Integrating the physical constraints of the objects can help to improve the accuracy of the articulation modeling and ensure that generated objects not only look realistic but also move and interact in a physically plausible manner.
Considering physical constraints is particularly crucial for simulation-driven applications such as robotic manipulation, digital twins, and physics-based animation, where real-world interaction dynamics must be accurately replicated.
This consideration is also useful in articulated object creation for applications such as virtual reality and gaming, where objects must react naturally to user interactions, maintaining realistic movements, and collision handling.
Future research should explore integrating physics engines directly into the modeling pipeline to enforce kinematic and dynamic feasibility during object creation.
Additionally, learning-based approaches could benefit from differentiable physics simulation which allows models to be trained with explicit physics constraints, improving both realism and generalization.
By embedding physical reasoning into the content creation process, we can bridge the gap between visual realism and functional accuracy, paving the way for more deployable and interactive articulated object models in robotics and virtual environments.

 
 
 Leveraging human-in-the-loop data .
Everyday human interactions with articulated objects produce a large amount of data in the form of long natural videos, egocentric videos, and human demonstrations (see examples in Figure   17 ).
These data are rich in articulated object information but also present challenges such as fast-moving objects, limited field-of-view camera viewpoints, streaming setups where the parts may disappear and reappear, and lots of occlusions and dynamics due to humans/hands in the view.
Collecting and using such data for articulated object modeling can enable generalization to diverse real objects and interactions.

 
 
 Evaluation and benchmarking .
Due to the multi-faceted nature of articulated objects, the evaluation of the quality of the created objects is currently not standardized.
The challenges of evaluation are partially due to the lack of ground truth data and the complexity of the task.
And even when the ground truth annotation is available, the metrics that have been proposed by prior work have flaws and are not fully and clearly specified.
There is a need for more systematic evaluation metrics that can evaluate different aspects of articulated object creation, including geometry, articulation, and part relationships, and also an object as a whole in the context of a scene where interaction can take place.

 
 
 Conclusion .
This survey systematically reviewed the progress in 3D modeling for articulated objects, with a focus on articulated part perception and articulated object creation.
By examining the development of articulated object modeling across the two axes of geometric and articulation modeling, we have identified significant advances and ongoing challenges in the field.
We also highlighted potential research directions that can drive future progress in the field.
We hope that this survey serves as a foundational reference for researchers and practitioners in computer vision and graphics, offering insights into the complexities of articulated object modeling and inspiring new research in this area.

 
 
 Acknowledgments. 
This work was funded in part by NSERC Discovery grants (RGPIN-06489-2019, RGPIN-2022-03111), a Canada Research Chair grant (CRC-2019-00298), and a startup grant from Simon Fraser University (N000449).
We thank Angel X. Chang, Richard Zhang, and Han-hung Lee for their helpful comments and discussions.

 
 
 

## References

 
 
 [AFA*22] 
 Hameed Abdul-Rashid et al.
 
 “Learning to infer kinematic hierarchies for novel object instances”
 
 In Proceedings of the International Conference on Robotics and Automation (ICRA) , 2022, pp. 8461–8467
 

 
 [AKMS23] 
 Syed Akber, Sadia Kazmi, Syed Mohsin and Agnieszka Szczęsna
 
 “Deep learning-based motion style transfer tools, techniques and future challenges”
 
 In Sensors 23.5 , 2023, pp. 2597
 

 
 [ALL*20] 
 Kfir Aberman et al.
 
 “Skeleton-aware networks for deep motion retargeting”
 
 In ACM Transactions on Graphics (TOG) 39.4 , 2020, pp. 62–1
 

 
 [AM24] 
 Mehmet Aygun and Oisin Mac
 
 “SAOR: Single-view articulated object reconstruction”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 10382–10391
 

 
 [AMD19] 
 Ammar Ahmad, Cyrille Migniot and Albert Dipanda
 
 “Hand pose estimation and tracking in real and virtual interaction: A review”
 
 In Image and Vision Computing 89 , 2019, pp. 35–49
 

 
 [ATK19] 
 Ben Abbatematteo, Stefanie Tellex and George Konidaris
 
 “Learning to generalize kinematic models to novel objects”
 
 In Proceedings of the Conference on Robot Learning (CoRL) , 2019, pp. 1289–1299
 

 
 [Bar16] 
 Emad Barsoum
 
 “Articulated hand pose estimation review”
 
 In arXiv preprint arXiv:1604.06195 , 2016
 

 
 [BDK*23] 
 Andreas Blattmann et al.
 
 “Stable video diffusion: Scaling latent video diffusion models to large datasets”
 
 In arXiv preprint arXiv:2311.15127 , 2023
 

 
 [BDTB18] 
 Stefano Berretti, Mohamed Daoudi, Pavan Turaga and Anup Basu
 
 “Representation, analysis, and recognition of 3D humans: A survey”
 
 In ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 14.1s , 2018, pp. 1–36
 

 
 [BKY*22] 
 Alexander Bergman et al.
 
 “Generative neural articulated radiance fields”
 
 In Advances in neural information processing systems (NeurIPS) 35 , 2022, pp. 19900–19916
 

 
 [BM92] 
 Paul Besl and Neil McKay
 
 “Method for registration of 3-D shapes”
 
 In Sensor fusion IV: control paradigms and data structures 1611 , 1992, pp. 586–606
 
 Spie
 

 
 [BXP*22] 
 Bharat Bhatnagar et al.
 
 “BEHAVE: Dataset and method for tracking human object interactions”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 15935–15946
 

 
 [BXQW23] 
 Chen Bao, Helin Xu, Yuzhe Qin and Xiaolong Wang
 
 “Dexart: Benchmarking generalizable dexterous manipulation with articulated objects”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 21190–21200
 

 
 [CFG*15] 
 Angel Chang et al.
 
 “ShapeNet: An information-rich 3D model repository”
 
 In arXiv preprint arXiv:1512.03012 , 2015
 

 
 [CGD*22] 
 Jasmine Collins et al.
 
 “Abo: Dataset and benchmarks for real-world 3d object understanding”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 21126–21136
 

 
 [CJS*23] 
 Xu Chen et al.
 
 “Fast-SNARF: A fast deformer for articulated neural fields”
 
 In IEEE Transactions on Pattern Analysis and Machine Intelligence 
 
 IEEE, 2023
 

 
 [CMS*22] 
 Bowen Cheng et al.
 
 “Masked-attention mask transformer for universal image segmentation”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 1290–1299
 

 
 [CPZ21] 
 Lu Chen, Sida Peng and Xiaowei Zhou
 
 “Towards efficient and photorealistic 3D human reconstruction: a brief survey”
 
 In Visual Informatics 5.4 , 2021, pp. 11–19
 

 
 [CRKM21] 
 Zhe Cao, Ilija Radosavovic, Angjoo Kanazawa and Jitendra Malik
 
 “Reconstructing hand-object interactions in the wild”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2021, pp. 12417–12426
 

 
 [CTM*21] 
 Mathilde Caron et al.
 
 “Emerging properties in self-supervised vision transformers”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2021, pp. 9650–9660
 

 
 [CYL15] 
 Hong Cheng, Lu Yang and Zicheng Liu
 
 “Survey on 3D hand gesture recognition”
 
 In IEEE transactions on circuits and systems for video technology 26.9 , 2015, pp. 1659–1673
 

 
 [DLS*23] 
 Congyue Deng et al.
 
 “Banana: Banach fixed-Point network for pointcloud segmentation with inter-part equivariance”
 
 In arXiv preprint arXiv:2305.16314 , 2023
 

 
 [DTT*24] 
 Alexandros Delitzas et al.
 
 “SceneFun3D: Fine-grained functionality and affordance understanding in 3D scenes”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 14531–14542
 

 
 [EZH22] 
 Ben Eisner, Harry Zhang and David Held
 
 “Flowbot3D: Learning 3D articulation flow to manipulate articulated objects”
 
 In arXiv preprint arXiv:2205.04382 , 2022
 

 
 [FJG*21] 
 Huan Fu et al.
 
 “3D-Future: 3D furniture shape with texture”
 
 In International Journal of Computer Vision 129 , 2021, pp. 3313–3337
 

 
 [FTT*22] 
 Zicong Fan et al.
 
 “Articulated objects in free-form hand interaction”
 
 In arXiv preprint arXiv:2204.13662 2 , 2022
 

 
 [GDG*23] 
 Anindita Ghosh et al.
 
 “IMoS: Intent-Driven Full-Body Motion Synthesis for Human-Object Interactions”
 
 In Computer Graphics Forum 42.2 , 2023, pp. 1–12
 

 
 [GES21] 
 Samir Gadre, Kiana Ehsani and Shuran Song
 
 “Act the part: Learning interaction strategies for articulated object part discovery”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2021, pp. 15752–15761
 

 
 [GEvdM18] 
 Benjamin Graham, Martin Engelcke and Laurens van Maaten
 
 “3D Semantic Segmentation with Submanifold Sparse Convolutional Networks”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018, pp. 9224–9232
 

 
 [GG04] 
 Natasha Gelfand and Leonidas Guibas
 
 “Shape segmentation using local slippage analysis”
 
 In Proceedings of the Eurographics Symposium on Geometry Processing , 2004, pp. 214–223
 

 
 [GGS*19] 
 Xiaofeng Gao et al.
 
 “Vrkitchen: an interactive 3D virtual environment for task-oriented learning”
 
 In arXiv preprint arXiv:1903.05757 , 2019
 

 
 [GLG*23] 
 Haoran Geng et al.
 
 “PartManip: Learning cross-category generalizable part manipulation policy from point cloud observations”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 2978–2988
 

 
 [GPM*14] 
 Ian Goodfellow et al.
 
 “Generative adversarial nets”
 
 In Advances in neural information processing systems (NeurIPS) 27 , 2014
 

 
 [GSLD24] 
 Daoyi Gao, Yawar Siddiqui, Lei Li and Angela Dai
 
 “MeshArt: Generating Articulated Meshes with Structure-guided Transformers”
 
 In arXiv preprint arXiv:2412.11596 , 2024
 
 DOI: 10.48550/arXiv.2412.11596 
 

 
 [GWB*22] 
 Kristen Grauman et al.
 
 “Ego4D: Around the world in 3,000 hours of egocentric video”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 18995–19012
 
 DOI: 10.48550/arXiv.2110.07058 
 

 
 [GXZ*23] 
 Haoran Geng et al.
 
 “GAPartNet: Cross-category domain-generalizable object perception and manipulation via generalizable and actionable parts”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 7081–7091
 
 DOI: 10.48550/arXiv.2211.05272 
 

 
 [HGDG17] 
 Kaiming He, Georgia Gkioxari, Piotr Dollár and Ross Girshick
 
 “Mask R-CNN”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2017, pp. 2961–2969
 

 
 [HIZ*23] 
 Nick Heppert et al.
 
 “CARTO: Category and Joint Agnostic Reconstruction of ARTiculated Objects”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 21201–21210
 

 
 [HJA20] 
 Jonathan Ho, Ajay Jain and Pieter Abbeel
 
 “Denoising diffusion probabilistic models”
 
 In Advances in neural information processing systems (NeurIPS) 33 , 2020, pp. 6840–6851
 

 
 [HJZ23] 
 Cheng-Chun Hsu, Zhenyu Jiang and Yuke Zhu
 
 “Ditto in the house: Building articulation models of indoor scenes through interactive perception”
 
 In Proceedings of the International Conference on Robotics and Automation (ICRA) , 2023, pp. 3933–3939
 
 IEEE
 

 
 [HLV*17] 
 Ruizhen Hu et al.
 
 “Learning to predict part mobility from a single static snapshot”
 
 In ACM Transactions on Graphics (TOG) 36.6 , 2017, pp. 1–13
 

 
 [HNOS15] 
 Karol Hausman, Scott Niekum, Sarah Osentoski and Gaurav Sukhatme
 
 “Active articulation model estimation through interactive perception”
 
 In Proceedings of the International Conference on Robotics and Automation (ICRA) , 2015, pp. 3305–3312
 

 
 [HSvK18] 
 Ruizhen Hu, Manolis Savva and Oliver van Kaick
 
 “Functionality representations and applications for shape analysis”
 
 In Computer Graphics Forum 37.2 , 2018, pp. 603–624
 

 
 [HWB*21] 
 Jiahui Huang et al.
 
 “Multibodysync: Multi-body segmentation and motion estimation via 3d scan synchronization”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2021, pp. 7108–7118
 

 
 [HYL*23] 
 Yangyi Huang et al.
 
 “One-shot implicit animatable avatars with model-based priors”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2023, pp. 8974–8985
 

 
 [HZRS16] 
 Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun
 
 “Deep residual learning for image recognition”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016, pp. 770–778
 

 
 [HZvK*15] 
 Ruizhen Hu et al.
 
 “Interaction context (ICON) towards a geometric functionality descriptor”
 
 In ACM Transactions on Graphics (TOG) 34.4 , 2015, pp. 1–12
 

 
 [IJZ*24] 
 Denys Iliash et al.
 
 “S2O: Static to Openable Enhancement for Articulated 3D Objects”, 2024
 
 arXiv: 2409.18896 
 

 
 [Inc17] 
 Trimble Inc.
 
 “3D Warehouse” Accessed: 2024-01-22, 2017
 
 URL: https://3dwarehouse.sketchup.com/ 
 

 
 [Inc17a] 
 Trimble Inc.
 
 “SketchUp” Accessed: 2024-01-22, 2017
 
 URL: https://www.sketchup.com/ 
 

 
 [JHZ22] 
 Zhenyu Jiang, Cheng-Chun Hsu and Yuke Zhu
 
 “Ditto: Building digital twins of articulated objects from interaction”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 5616–5626
 
 DOI: 10.48550/arXiv.2202.08227 
 

 
 [JLC*23] 
 Nan Jiang et al.
 
 “Full-body articulated human-object interaction”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2023, pp. 9365–9376
 

 
 [JLCN21] 
 Ajinkya Jain, Rudolf Lioutikov, Caleb Chuck and Scott Niekum
 
 “ScrewNet: Category-Independent Articulation Model Estimation From Depth Images Using Screw Theory”
 
 In Proceedings of the International Conference on Robotics and Automation (ICRA) , 2021, pp. 13670–13677
 

 
 [JLW*24] 
 Tomas Jakab et al.
 
 “Farm3d: Learning articulated 3d animals by distilling 2d diffusion”
 
 In Proceedings of the International Conference on 3D Vision (3DV) , 2024, pp. 852–861
 

 
 [JMSC22] 
 Hanxiao Jiang, Yongsen Mao, Manolis Savva and Angel Chang
 
 “OPD: Single-view 3D openable part detection”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2022, pp. 410–426
 

 
 [JSGC15] 
 Mariano Jaimez, Mohamed Souiai, Javier Gonzalez-Jimenez and Daniel Cremers
 
 “A primal-dual framework for real-time dense RGB-D scene flow”
 
 In Proceedings of the International Conference on Robotics and Automation (ICRA) , 2015, pp. 98–104
 

 
 [KGFT20] 
 Nilesh Kulkarni, Abhinav Gupta, David Fouhey and Shubham Tulsiani
 
 “Articulation-aware canonical surface mapping”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2020, pp. 452–461
 

 
 [KKK*23] 
 Tianshu Kuai et al.
 
 “CAMM: Building category-agnostic and animatable 3D models from monocular videos”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 6587–6597
 

 
 [KKLD23] 
 Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler and George Drettakis
 
 “3D Gaussian Splatting for Real-Time Radiance Field Rendering.”
 
 In ACM Transactions on Graphics (TOG) 42.4 , 2023, pp. 139–1
 

 
 [KKNJ24] 
 Jeonghwan Kim, Jisoo Kim, Jeonghyeon Na and Hanbyul Joo
 
 “ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions”
 
 In arXiv preprint arXiv:2401.10232 , 2024
 

 
 [KKW*24] 
 Justin Kerr et al.
 
 “Robot See Robot Do: Imitating Articulated Object Manipulation with Monocular 4D Reconstruction”
 
 In arXiv preprint arXiv:2409.18121 , 2024
 
 DOI: 10.48550/arXiv.2409.18121 
 

 
 [KMH*17] 
 Eric Kolve et al.
 
 “AI2-THOR: An interactive 3D environment for visual AI”
 
 In arXiv preprint arXiv:1712.05474 , 2017
 

 
 [KMJ*24] 
 Mukul Khanna et al.
 
 “Habitat synthetic scenes dataset (HSSD-200): An analysis of 3D scene scale and realism tradeoffs for objectgoal navigation”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 16384–16393
 

 
 [KMR*23] 
 Alexander Kirillov et al.
 
 “Segment anything”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2023, pp. 4015–4026
 

 
 [LC98] 
 William Lorensen and Harvey Cline
 
 “Marching Cubes: A high resolution 3D surface construction algorithm”
 
 In Seminal graphics: pioneering efforts that shaped the field , 1998, pp. 347–353
 

 
 [LDS*23] 
 Jiahui Lei et al.
 
 “NAP: Neural 3D articulated object prior”
 
 In Advances in neural information processing systems (NeurIPS) 36 , 2023, pp. 31878–31894
 
 DOI: 10.48550/arXiv.2305.16315 
 

 
 [LGD*24] 
 Rundong Luo et al.
 
 “PhysPart: Physically plausible part completion for interactable objects”
 
 In arXiv preprint arXiv:2408.13724 , 2024
 

 
 [LGW23] 
 Shaowei Liu, Saurabh Gupta and Shenlong Wang
 
 “Building rearticulable models for arbitrary 3d objects from 4d point clouds”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 21138–21147
 

 
 [LIC*24] 
 Jiayi Liu et al.
 
 “SINGAPO: Single Image Controlled Generation of Articulated Parts in Object”
 
 In arXiv preprint arXiv:2410.16499 , 2024
 
 DOI: 10.48550/arXiv.2410.16499 
 

 
 [LJ23] 
 Jiye Lee and Hanbyul Joo
 
 “Locomotion-action-manipulation: Synthesizing human-scene interactions in complex 3d environments”
 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 9663–9674
 

 
 [LJL*25] 
 Yu Liu et al.
 
 “Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting”
 
 In arXiv preprint arXiv:2502.19459 , 2025
 

 
 [LLL*24] 
 Zizhang Li et al.
 
 “Learning the 3D Fauna of the Web”
 
 In arXiv preprint arXiv:2401.02400 , 2024
 

 
 [LLT19] 
 Rui Li, Zhenyu Liu and Jianrong Tan
 
 “A survey on 3D hand pose estimation: Cameras, methods, and datasets”
 
 In Pattern Recognition 93 , 2019, pp. 251–272
 

 
 [LMS23] 
 Jiayi Liu, Ali Mahdavi-Amiri and Manolis Savva
 
 “PARIS: Part-level reconstruction and motion analysis for articulated objects”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2023, pp. 352–363
 
 DOI: 10.48550/arXiv.2308.07391 
 

 
 [Low04] 
 David Lowe
 
 “Distinctive image features from scale-invariant keypoints”
 
 In International journal of computer vision 60 , 2004, pp. 91–110
 

 
 [LP17] 
 Kevin. Lynch and Frank. Park
 
 “Modern robotics: Mechanics, planning, and control (Preprint version)”, 2017
 
 URL: https://hades.mech.northwestern.edu/images/7/7f/MR.pdf 
 

 
 [LSH*23] 
 Gengxin Liu et al.
 
 “Semi-weakly supervised object kinematic motion prediction”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 21726–21735
 
 DOI: 10.48550/arXiv.2303.17774 
 

 
 [LSZ*24] 
 Di Liu et al.
 
 “Lepard: Learning explicit part discovery for 3D articulated shape reconstruction”
 
 In Advances in neural information processing systems (NeurIPS) 36 , 2024
 

 
 [LTMS24] 
 Jiayi Liu, Hou Tam, Ali Mahdavi-Amiri and Manolis Savva
 
 “CAGE: Controllable Articulation GEneration”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 17880–17889
 
 DOI: 10.48550/arXiv.2312.09570 
 

 
 [LWL*16] 
 Hao Li et al.
 
 “Mobility fitting using 4D RANSAC”
 
 In Computer Graphics Forum 35.5 , 2016, pp. 79–88
 

 
 [LWP*24] 
 Jiahui Lei et al.
 
 “Gart: Gaussian articulated template models”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 19876–19887
 

 
 [LWW*24] 
 Suhan Ling et al.
 
 “Articulated object manipulation with coarse-to-fine affordance for mitigating the effect of point cloud noise”
 
 In Proceedings of the International Conference on Robotics and Automation (ICRA) , 2024, pp. 10895–10901
 
 IEEE
 

 
 [LWWY23] 
 Xueyi Liu, Bin Wang, He Wang and Li Yi
 
 “Few-shot physically-aware articulated mesh generation via hierarchical deformation”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2023, pp. 854–864
 

 
 [LWY*20] 
 Xiaolong Li et al.
 
 “Category-level articulated object pose estimation”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2020, pp. 3706–3715
 
 DOI: 10.48550/arXiv.1912.11913 
 

 
 [LXF*22] 
 Liu Liu et al.
 
 “AKB-48: A real-world articulated object knowledge base”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 14809–14818
 

 
 [LXL*24] 
 Long Le et al.
 
 “Articulate-Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model”
 
 In arXiv preprint arXiv:2410.13882 , 2024
 

 
 [LXX*22] 
 Liu Liu et al.
 
 “Toward real-world category-level articulation pose estimation”
 
 In IEEE Transactions on Image Processing 31 , 2022, pp. 1072–1083
 

 
 [LZH*23] 
 Xueyi Liu et al.
 
 “Self-supervised category-Level articulated object pose estimation with part-Level SE(3) Equivariance”
 
 In Proceedings of the International Conference on Learning Representations (ICLR) , 2023
 

 
 [LZL*24] 
 Jiong Lin et al.
 
 “AutoURDF: Unsupervised Robot Modeling from Point Cloud Frames Using Cluster Registration”
 
 In arXiv preprint arXiv:2412.05507 , 2024
 

 
 [LZRV24] 
 Ruining Li, Chuanxia Zheng, Christian Rupprecht and Andrea Vedaldi
 
 “DragAPart: Learning a part-level motion prior for articulated objects”
 
 In arXiv preprint arXiv:2403.15382 , 2024
 

 
 [LZWL23] 
 Zhe Li, Zerong Zheng, Lizhen Wang and Yebin Liu
 
 “Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling”
 
 In arXiv preprint arXiv:2311.16096 , 2023
 

 
 [MEB19] 
 Roberto Martín-Martín, Clemens Eppner and Oliver Brock
 
 “The RBO dataset of articulated objects and interactions”
 
 In The International Journal of Robotics Research 38.9 , 2019, pp. 1013–1019
 

 
 [MGM*21] 
 Kaichun Mo et al.
 
 “Where2act: From pixels to actions for articulated 3D objects”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2021, pp. 6813–6823
 

 
 [MHF*22] 
 Mayank Mittal et al.
 
 “Articulated object interaction in unknown scenes with whole-body mobile manipulation”
 
 In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2022, pp. 1647–1654
 

 
 [MHL*22] 
 Lucas Mourot et al.
 
 “A survey on deep learning for skeleton-based human animation”
 
 In Computer Graphics Forum 41.1 , 2022, pp. 122–157
 

 
 [MNH23] 
 Shubh Maheshwari, Rahul Narain and Ramya Hebbalaguppe
 
 “Transfer4D: A framework for frugal motion capture and deformation transfer”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 12836–12846
 

 
 [MON*19] 
 Lars Mescheder et al.
 
 “Occupancy networks: Learning 3D reconstruction in function space”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019, pp. 4460–4470
 

 
 [MON*19a] 
 Lars Mescheder et al.
 
 “Occupancy networks: Learning 3D reconstruction in function space”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019, pp. 4460–4470
 
 DOI: 10.48550/arXiv.1812.03828 
 

 
 [MQK*21] 
 Jiteng Mu et al.
 
 “A-SDF: Learning Disentangled Signed Distance Functions for Articulated Shape Representation”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2021, pp. 12981–12991
 

 
 [MST*21] 
 Ben Mildenhall et al.
 
 “NeRF: Representing scenes as neural radiance fields for view synthesis”
 
 In Communications of the ACM 65.1 , 2021, pp. 99–106
 
 DOI: 10.48550/arXiv.2003.08934 
 

 
 [MWBS24] 
 Zhao Mandi, Yijia Weng, Dominik Bauer and Shuran Song
 
 “Real2Code: Reconstruct Articulated Objects via Code Generation”
 
 In arXiv preprint arXiv:2406.08474 , 2024
 

 
 [MYY*10] 
 Niloy Mitra et al.
 
 “Illustrating how mechanical assemblies work”
 
 In ACM Transactions on Graphics (TOG) 29.4 , 2010, pp. 58
 

 
 [MZC*19] 
 Kaichun Mo et al.
 
 “PartNet: A large-scale benchmark for fine-grained and hierarchical part-level 3D object understanding”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019, pp. 909–918
 

 
 [MZJ*22] 
 Yongsen Mao et al.
 
 “MultiScan: Scalable RGBD scanning for 3D environments with articulated objects”
 
 In Advances in neural information processing systems (NeurIPS) 35 , 2022, pp. 9058–9071
 

 
 [NGES23] 
 Neil Nie, Samir Gadre, Kiana Ehsani and Shuran Song
 
 “Structure from Action: Learning Interactions for 3D Articulated Object Structure Discovery”
 
 In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2023, pp. 1222–1229
 

 
 [NIT*22] 
 Atsuhiro Noguchi et al.
 
 “Watch it move: Unsupervised discovery of 3D joints for re-posing of articulated objects”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 3677–3687
 

 
 [NWL*24] 
 Chuanruo Ning et al.
 
 “Where2Explore: Few-shot affordance learning for unseen novel categories of articulated objects”
 
 In Advances in neural information processing systems (NeurIPS) 36 , 2024
 

 
 [ODM*23] 
 Maxime Oquab et al.
 
 “DINOv2: Learning robust visual features without supervision”
 
 In arXiv preprint arXiv:2304.07193 , 2023
 

 
 [PB11] 
 Chavdar Papazov and Darius Burschka
 
 “Deformable 3D shape registration based on local similarity transforms”
 
 In Computer Graphics Forum 30.5 , 2011, pp. 1493–1502
 

 
 [PG08] 
 Yuri Pekelny and Craig Gotsman
 
 “Articulated object reconstruction and markerless motion capture from depth video”
 
 In Computer Graphics Forum 27.2 , 2008, pp. 399–408
 

 
 [PJBM22] 
 Ben Poole, Ajay Jain, Jonathan Barron and Ben Mildenhall
 
 “DreamFusion: Text-to-3D using 2D diffusion”
 
 In arXiv preprint arXiv:2209.14988 , 2022
 

 
 [PRB*18] 
 Xavier Puig et al.
 
 “Virtualhome: Simulating household activities via programs”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018, pp. 8494–8502
 

 
 [PŠDC22] 
 Petra Pejić, Valentin Šimundić, Matej Džijan and Robert Cupec
 
 “Articulated Objects: From Detection to Manipulation—Survey”
 
 In International Conference on Intelligent Autonomous Systems , 2022, pp. 495–508
 

 
 [PUS*23] 
 Xavier Puig et al.
 
 “Habitat 3.0: A co-habitat for humans, avatars and robots”
 
 In arXiv preprint arXiv:2310.13724 , 2023
 

 
 [QCG*24] 
 Lingteng Qiu et al.
 
 “RichDreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3D”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 9914–9925
 

 
 [QF23] 
 Shengyi Qian and David Fouhey
 
 “Understanding 3D object interaction from a single image”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2023, pp. 21753–21763
 
 DOI: 10.48550/arXiv.2203.16531 
 

 
 [QJR*22] 
 Shengyi Qian et al.
 
 “Understanding 3D object articulation in internet videos”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 1599–1609
 
 DOI: 10.48550/arXiv.2203.16531 
 

 
 [QSMG16] 
 C Qi, H Su, K Mo and LJ Guibas
 
 “PointNet: deep learning on point sets for 3D classification and segmentation. 2017”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016, pp. 77–85
 

 
 [QWM*23] 
 Zhiyin Qian et al.
 
 “3DGS-avatar: Animatable avatars via deformable 3D gaussian splatting”
 
 In arXiv preprint arXiv:2312.09228 , 2023
 

 
 [QYSG17] 
 Charles Qi, Li Yi, Hao Su and Leonidas Guibas
 
 “Pointnet++: Deep hierarchical feature learning on point sets in a metric space”
 
 In Advances in neural information processing systems (NeurIPS) 30 , 2017
 

 
 [QYW*25] 
 Xiaowen Qiu et al.
 
 “Articulate AnyMesh: Open-vocabulary 3D Articulated Objects Modeling”
 
 In arXiv preprint arXiv:2502.02590 , 2025
 

 
 [RBL*22] 
 Robin Rombach et al.
 
 “High-resolution image synthesis with latent diffusion models”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 10684–10695
 

 
 [RGG*23] 
 Baptiste Roziere et al.
 
 “Code Llama: Open foundation models for code”
 
 In arXiv preprint arXiv:2308.12950 , 2023
 

 
 [SA07] 
 Olga Sorkine and Marc Alexa
 
 “As-rigid-as-possible surface modeling”
 
 In Proceedings of the Eurographics Symposium on Geometry Processing 4 , 2007, pp. 109–116
 

 
 [SBR22] 
 Shih-Yang Su, Timur Bagautdinov and Helge Rhodin
 
 “Danbo: Disentangled articulated neural body representations via graph neural networks”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2022, pp. 107–124
 
 Springer
 

 
 [SCZ21] 
 Yahao Shi, Xinyu Cao and Bin Zhou
 
 “Self-Supervised Learning of Part Mobility from Point Cloud Sequence”
 
 In Computer Graphics Forum 40.6 , 2021, pp. 104–116
 

 
 [SFL*24] 
 Jiayi Su et al.
 
 “ArtFormer: Controllable Generation of Diverse 3D Articulated Objects”
 
 In arXiv preprint arXiv:2412.07237 , 2024
 

 
 [SGG*24] 
 Archana Swaminathan et al.
 
 “LEIA: Latent View-invariant Embeddings for Implicit 3D Articulation”
 
 In arXiv preprint arXiv:2409.06703 , 2024
 

 
 [SHL*14] 
 Andrei Sharf et al.
 
 “Mobility-trees for indoor scenes manipulation”
 
 In Computer Graphics Forum 33.1 , 2014, pp. 2–14
 

 
 [SJSC23] 
 Xiaohao Sun, Hanxiao Jiang, Manolis Savva and Angel Chang
 
 “OPDMulti: Openable Part Detection for Multiple Objects”
 
 In arXiv preprint arXiv:2303.14087 , 2023
 

 
 [SKM*19] 
 Manolis Savva et al.
 
 “Habitat: A platform for embodied AI research”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2019, pp. 9339–9347
 

 
 [SLW*24] 
 Jianhua Sun et al.
 
 “Arti-PG: A Toolbox for Procedurally Synthesizing Large-Scale and Diverse Articulated Objects with Rich Annotations”
 
 In arXiv preprint arXiv:2412.14974 , 2024
 

 
 [SNF14] 
 Tanner Schmidt, Richard Newcombe and Dieter Fox
 
 “DART: Dense Articulated Real-Time Tracking.”
 
 In Robotics: Science and systems 2.1 , 2014, pp. 1–9
 

 
 [SWF*24] 
 Chaoyue Song et al.
 
 “REACTO: Reconstructing Articulated Objects from a Single Video”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 5384–5395
 

 
 [SYZR21] 
 Shih-Yang Su, Frank Yu, Michael Zollhöfer and Helge Rhodin
 
 “A-NeRF: Articulated neural radiance fields for learning human shape, appearance, and pose”
 
 In Advances in neural information processing systems (NeurIPS) 34 , 2021, pp. 12278–12291
 

 
 [TGBT20] 
 Omid Taheri, Nima Ghorbani, Michael Black and Dimitrios Tzionas
 
 “GRAB: A dataset of whole-body human grasping of objects”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2020, pp. 581–600
 

 
 [TLYS22] 
 Wei-Cheng Tseng, Hung-Ju Liao, Lin Yen-Chen and Min Sun
 
 “CLA-NeRF: Category-level articulated neural radiance field”
 
 In Proceedings of the International Conference on Robotics and Automation (ICRA) , 2022, pp. 8454–8460
 

 
 [TYR23] 
 Jeff Tan, Gengshan Yang and Deva Ramanan
 
 “Distilling neural fields for real-time articulated shape reconstruction”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 4692–4701
 

 
 [WCK19] 
 Chung-Yi Weng, Brian Curless and Ira Kemelmacher-Shlizerman
 
 “Photo wake-up: 3D character animation from a single photo”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019, pp. 5908–5917
 

 
 [WCL*22] 
 Yuefan Wu et al.
 
 “CASA: Category-agnostic skeletal animal reconstruction”
 
 In Advances in neural information processing systems (NeurIPS) 35 , 2022, pp. 28559–28574
 

 
 [WCM*22] 
 Fangyin Wei et al.
 
 “Self-supervised neural articulated shape and appearance models”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 15816–15826
 

 
 [WGYZ25] 
 Ruiqi Wang, Akshay Gadi, Fenggen Yu and Hao Zhang
 
 “Active Coarse-to-Fine Segmentation of Moveable Parts from Real Images”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2025, pp. 111–127
 
 Springer
 

 
 [WLC*24] 
 Shuzhe Wang et al.
 
 “DUSt3R: Geometric 3D vision made easy”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 20697–20709
 

 
 [WLJ*23] 
 Shangzhe Wu et al.
 
 “Magicpony: Learning articulated 3D animals in the wild”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 8792–8802
 

 
 [WLL*21] 
 Peng Wang et al.
 
 “NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction”
 
 In arXiv preprint arXiv:2106.10689 , 2021
 

 
 [WLY*24] 
 Junbo Wang et al.
 
 “RPMArt: Towards Robust Perception and Manipulation for Articulated Objects”
 
 In arXiv preprint arXiv:2403.16023 , 2024
 

 
 [WSGT22] 
 Shaofei Wang, Katja Schwarz, Andreas Geiger and Siyu Tang
 
 “Arah: Animatable volume rendering of articulated human sdfs”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2022, pp. 1–19
 
 Springer
 

 
 [WTZ*21] 
 Jinbao Wang et al.
 
 “Deep 3D human pose estimation: A review”
 
 In Computer Vision and Image Understanding 210 , 2021, pp. 103225
 

 
 [WWM*22] 
 Yian Wang et al.
 
 “Adaafford: Learning to adapt manipulation affordance for 3d articulated objects via few-shot interactions”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2022, pp. 90–107
 

 
 [WWS*15] 
 Nkenge Wheatland et al.
 
 “State of the art in hand and finger modeling and animation”
 
 In Computer Graphics Forum 34.2 , 2015, pp. 735–760
 

 
 [WWT*24] 
 Yijia Weng et al.
 
 “Neural Implicit Representation for Building Digital Twins of Unknown Articulated Objects”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 3141–3150
 

 
 [WWZ*21] 
 Yijia Weng et al.
 
 “CAPTRA: Category-level pose tracking for rigid and articulated objects from point clouds”
 
 In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , 2021, pp. 13209–13218
 

 
 [WZS*19] 
 Xiaogang Wang et al.
 
 “Shape2Motion: Joint analysis of motion parts and attributes from 3D shapes”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019, pp. 8876–8884
 
 DOI: 10.48550/arXiv.1903.03911 
 

 
 [XHS22] 
 Zhenjia Xu, Zhanpeng He and Shuran Song
 
 “Universal manipulation policy network for articulated objects”
 
 In IEEE robotics and automation letters 7.2 , 2022, pp. 2447–2454
 

 
 [XJMS21] 
 Xiang Xu, Hanbyul Joo, Greg Mori and Manolis Savva
 
 “D3D-HOI: Dynamic 3D human-object interactions from videos”
 
 In arXiv preprint arXiv:2108.08420 , 2021
 

 
 [XQM*20] 
 Fanbo Xiang et al.
 
 “SAPIEN: A SimulAted Part-based Interactive ENvironment”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2020, pp. 11097–11107
 
 DOI: 10.48550/arXiv.2003.08515 
 

 
 [XWA*24] 
 Xianghui Xie et al.
 
 “RHOBIN Challenge: Reconstruction of human object interaction”
 
 In arXiv preprint arXiv:2401.04143 , 2024
 

 
 [XWY*09] 
 Weiwei Xu et al.
 
 “Joint-aware manipulation of deformable models”
 
 In ACM Transactions on Graphics (TOG) 28.3 , 2009, pp. 1–9
 

 
 [YHL*18] 
 Li Yi et al.
 
 “Deep part induction from articulated object pairs”
 
 In ACM Transactions on Graphics (TOG) 37.6 , 2018, pp. 1–15
 
 DOI: https://doi.org/10.48550/arXiv.1809.07417 
 

 
 [YHL*22] 
 Chun-Han Yao et al.
 
 “LASSIE: Learning articulated shapes from sparse image ensemble via 3D part discovery”
 
 In Advances in neural information processing systems (NeurIPS) 35 , 2022, pp. 15296–15308
 

 
 [YHL*23] 
 Chun-Han Yao et al.
 
 “Hi-LASSIE: High-fidelity articulated shape and skeleton discovery from sparse image ensemble”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 4853–4862
 

 
 [YHY*19] 
 Zihao Yan et al.
 
 “RPM-Net: recurrent prediction of motion and parts from point cloud”
 
 In ACM Transactions on Graphics (TOG) 38.6 , 2019, pp. 1–15
 
 DOI: 10.1145/3355089.3356573 
 

 
 [YLX*16] 
 Qing Yuan et al.
 
 “Space-time co-segmentation of articulated point cloud sequences”
 
 In Computer Graphics Forum 35.2 , 2016, pp. 419–429
 

 
 [YRH*24] 
 Chun-Han Yao et al.
 
 “ARTIC3D: Learning robust articulated 3d shapes from noisy web image collections”
 
 In Advances in neural information processing systems (NeurIPS) 36 , 2024
 

 
 [YSJ*21] 
 Gengshan Yang et al.
 
 “LASR: Learning articulated shape reconstruction from a monocular video”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2021, pp. 15980–15989
 

 
 [YVN*22] 
 Gengshan Yang et al.
 
 “Banmo: Building animatable 3D neural models from many casual videos”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2022, pp. 2863–2873
 

 
 [YWL*24] 
 Qiaojun Yu et al.
 
 “GAMMA: Generalizable articulation modeling and manipulation for articulated objects”
 
 In Proceedings of the International Conference on Robotics and Automation (ICRA) , 2024, pp. 5419–5426
 

 
 [YZF*24] 
 Yanhong Yang et al.
 
 “Digitalization of Three-Dimensional Human Bodies: A Survey”
 
 In IEEE Transactions on Consumer Electronics 
 
 IEEE, 2024
 

 
 [YZS*24] 
 Zhangsihao Yang et al.
 
 “OmniMotionGPT: Animal Motion Generation with Limited Data”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2024, pp. 1249–1259
 

 
 [YZW*22] 
 Ji Yang et al.
 
 “Object Wake-up: 3D Object Rigging from a Single Image”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2022, pp. 311–327
 
 Springer
 

 
 [ZBS*22] 
 Xiaohan Zhang et al.
 
 “Couch: Towards controllable human-chair interactions”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2022, pp. 518–535
 

 
 [ZGL*23] 
 Yuchen Zhou et al.
 
 “PartSLIP++: Enhancing Low-Shot 3D Part Segmentation via Multi-View Instance Segmentation and Maximum Likelihood Estimation”
 
 In arXiv preprint arXiv:2312.03015 , 2023
 

 
 [ZLLK21] 
 Vicky Zeng, Tabitha Lee, Jacky Liang and Oliver Kroemer
 
 “Visual identification of articulated object parts”
 
 In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2021, pp. 2443–2450
 

 
 [ZPJ*20] 
 Jason Zhang et al.
 
 “Perceiving 3D human-object spatial arrangements from a single image in the wild”
 
 In Proceedings of the European Conference on Computer Vision (ECCV) , 2020, pp. 34–51
 

 
 [ZWC*23] 
 Ce Zheng et al.
 
 “Deep learning-based human pose estimation: A survey”
 
 In ACM Computing Surveys 56.1 , 2023, pp. 1–37
 

 
 [ZZC*23] 
 Jianrong Zhang et al.
 
 “Generating human motion from textual descriptions with discrete representations”
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2023, pp. 14730–14740