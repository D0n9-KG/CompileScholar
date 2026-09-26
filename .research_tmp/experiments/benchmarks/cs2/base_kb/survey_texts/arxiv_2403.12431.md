Geometric Constraints in Deep Learning Frameworks: A Survey 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY-NC-SA 4.0
 
 
arXiv:2403.12431v2 [cs.CV] 09 Jul 2025 
 
 

# Geometric Constraints in Deep Learning Frameworks: A Survey

 
 
 Vibhas K Vats
 
 Affiliation:  Indiana University Bloomington
 
    
 David J Crandall
 
 Affiliation:  {vkvats, djcran}@iu.edu 
 

 Abstract 
 
 Stereophotogrammetry [ 1 ] is an established technique for scene understanding.
Its origins go back to at least the 1800s when people first started to investigate
using photographs to measure the physical properties of the world.
Since then, thousands of approaches have been explored.
The classic geometric technique of Shape from Stereo is
built on using geometry to define constraints on scene and camera
deep learning without any attempt to explicitly model the geometry.
In this survey, we explore geometry-inspired deep learning-based frameworks.
We compare and contrast geometry enforcing constraints integrated
into deep learning frameworks for depth estimation and other closely
related vision tasks. We present a new taxonomy for prevalent
geometry enforcing constraints used in modern deep learning frameworks.
We also present insightful observations and potential future research directions.

 
 
 
   
   
 
 
 
 
 
 | 

 

 
 
 
 
 K eywords  Depth Estimation ⋅ \cdot 
Monocular ⋅ \cdot 
Stereo ⋅ \cdot 
Multi-view Stereo ⋅ \cdot 
Geometric Constraints ⋅ \cdot 
Stereophotogrammetry ⋅ \cdot 
Scene understanding ⋅ \cdot 
self-supervised Depth Estimation ⋅ \cdot 
Photometric Consistency ⋅ \cdot 
Smoothness
 ⋅ \cdot 
Geometric Representations ⋅ \cdot 
Structural Consistency

 
 

## 1 Introduction

 
 Traditional stereo or multi-view stereo (MVS) depth estimation
methods rely on solving for
photometric and geometric consistency constraints across view(s)
for consistent depth estimation
 [ 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ] .
With the phenomenal rise of deep learning frameworks [ 10 ] 
like Convolutional Neural Networks (CNNs) [ 11 ] ,
Recurrent Neural Networks (RNNs) [ 12 ] , and
Vision Transformers (ViTs) [ 13 ] ,
which can extract deep local and high-level features,
the requirement to apply photometric and geometric consistency constraints
has significantly reduced, especially in supervised depth estimation methods.
Compared to traditional feature extraction methods, deep learning-based features significantly improve feature matching, leading to significant improvements in depth estimates
 [ 14 , 15 , 16 , 17 , 18 , 19 ] .
However, the application of geometric constraints in these deep learning-based frameworks remains limited to
the use of plane-sweep algorithm [ 20 ] for a majority
of supervised stereo and MVS methods.

 
 
 In a typical supervised stereo or MVS depth estimation framework,
the plane-sweep algorithm is applied to create a matching (cost) volume,
which is then aggregated based on a metric. The aggregated volume (i.e. the cost
volume) is then regularized using 3D-CNNs or RNNs to produce a coherent estimate.
The lack of ground truth in unsupervised/self-supervised depth estimation methods do not
allow such freedom. Photometric and geometric consistency constraints
remain a key part of unsupervised frameworks
 [ 21 , 22 , 23 , 24 , 25 , 18 , 19 ] .
Some other closely related problems, like structure from motion [ 26 , 27 ] ,
video-depth estimation [ 28 , 29 ] ,
semantic-segmentation [ 30 , 31 , 32 , 21 ] , and
monocular depth estimation [ 33 , 34 , 18 ] , also apply
various geometric constraints for a consistent result. In this survey,
we focus on all such methods that integrate photometric or geometric
constraints in deep learning-based frameworks and
are closely related to the depth estimation problem. Fig. 2 
shows the collection of such geometric constraints and their associated problems that
are covered in this survey. We discuss the theory and mathematical formulation of all
these concepts and present a carefully crafted taxonomy, as shown in Fig. 2 .

 
 
 
 
 Figure 1: Overview of geometry-inspired deep learning-based constraints surveyed in this paper.
Specifically, we discuss geometry-inspired constraints applied in deep learning frameworks for
depth estimation problems. We also discuss concepts that are used in closely related vision tasks
like Structure from Motion, Semantic Segmentation,
Video depth estimation, Image Captioning, etc. This word cloud shows all geometry-inspired
concepts covered in this survey. 
 
 Figure 2: Our taxonomy of geometry-inspired constraints used in
deep learning-based depth estimation and other closely related vision tasks. 
 
 
 
 
 
 Constraints | 
 Sub-Constraints | 
 Pre-Process. | 
 Post-Process. | 
 Obj. Fn | 
 In-Network | 

 
 Plane-Sweep Algo. | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 

 
 | 
 Photometric Constraints | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 

 
 Cross-View | 
 Geometric Constraints | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 

 
 Constraints | 
 Depth-to-Flow Constraints | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 

 
 | 
 RGB-to-Flow Constraints | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 

 
 | 
 View Synthesis Constraints | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 

 
 | 
 Structural Similarity Index | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 

 
 | 
 Edge-Aware Smoothness | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 

 
 Geometry Preserving | 
 Consistency Regularization | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 

 
 Constraints | 
 Planar Consistency | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 

 
 | 
 Point Cloud Alignment | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 

 
 Normal-Depth | 
 Depth-to-Normal | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 

 
 Orthogonality Const. | 
 Normal-to-Depth | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 

 
 | 
 Normal-Depth Joint Opt. | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 

 
 Attention Geometry | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 

 
 | 
 High-Level Feature Alignment | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 

 
 | 
 Pseudo-Label Cross-View | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 

 
 Learning Geometric | 
 Data-Augmented Robustness | 
 ✓ \checkmark | 
 – | 
 ✓ \checkmark | 
 – | 

 
 Representations | 
 Semantic-Guided Structure | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 

 
 | 
 Geometric Representation | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 

 

 Table 1: Our taxonomy organizes geometry-inspired constraints into six main constraints, some of which are further divided into sub-constraints.
This table highlights the most common ways these sub-constraints
are incorporated into deep learning frameworks.
Specifically, we classify the sub-constraints into four primary use cases:
as a pre-processing step, as a post-processing step, as part of the objective function,
or as an integrated component within a deep neural network. This categorization provides
a clear understanding of where and how these constraints are typically applied to enhance the learning process in geometry-aware models 
 
 
 We employ ‘ backward citation tracking ,’ also known as ‘ citation chaining ,’
for each geometry-based constraint discussed in this survey.
Specifically, we begin with recent papers and trace their references
to identify the original work that introduced the concepts.
We focus on selecting relevant papers published in prominent conferences and journals,
excluding preprints that are only available on ArXiv.
Our curated list spans two distinct eras: the early era ,
which includes seminal papers that first introduced these concepts,
and the modern era , which consists of recent works (from the past 6–8 years)
that apply these concepts, either individually or in combination with other
geometry-based ideas, within learning-based frameworks.
To identify relevant papers, we use Google Scholar 
with keywords such as ‘ geometric constraints ’
combined with terms like ‘ learning-based frameworks ,’
‘ depth estimation ,’ and ‘ multi-view stereo ,’
ensuring a comprehensive and up-to-date review of the field.

 
 
 Throughout this paper, we focus on geometry-enforcing concepts
used across different problems that either do depth estimation or are closely
related to depth estimation problems.
We only discuss the specific concepts used
and their relevance to stereo or MVS depth estimation frameworks.
We also specify efficacy
in handling cluttered backgrounds, repetitive patterns, and texture-less regions
for each geometric constraint that can be used as an objective function.
We provide two comprehensive snapshots of the surveyed methods
in Tables 1 and 2 .
Table 1 offers a detailed view of how geometry-based
constraints can be integrated into learning-based frameworks.
These constraints are categorized into four primary applications:
as pre-processing steps, post-processing steps, objective functions,
or integral components within a network. Each sub-constraint discussed
in this survey is classified into one or more categories.
Table 2 summarizes key methods,
showcasing the specific types of geometry-based constraints they incorporate.
Notably, many methods employ multiple sub-constraints,
often spanning different categories.
Furthermore, some geometry-based
constraints are adapted from unrelated domains,
such as image captioning, structure from motion,
and semantic segmentation, and have only recently been explored
in the context of depth estimation.
This cross-domain borrowing creates further
inconsistencies, making it difficult to provide
a meaningful quantitative comparison specific to depth estimation tasks.

 
 
 
 
 
 Methods | 
 | 
 Cross-View | 
 Geometry Preserving | 
 Normal-Depth | 
 | 
 Learning Geometric | 

 
 | 
 Constraints | 
 Constraints | 
 Orthogonal | 
 | 
 Representations | 

 
   
 
 Plane-Sweep Algorithm 
 | 
   
 
 Photometric Constraints 
 | 
   
 
 Geometric Constraints 
 | 
   
 
 Depth-Flow Consistency 
 | 
   
 
 View Synthesis Constraints 
 | 
   
 
 Structural Similarity Index 
 | 
   
 
 Edge-Aware Smoothness 
 | 
   
 
 Consistency Regularization 
 | 
   
 
 Planar Consistency 
 | 
   
 
 Point-Cloud Alignment 
 | 
   
 
 Depth-to-Normal 
 | 
   
 
 Normal-to-Depth 
 | 
   
 
 Normal-Depth Joint Opt. 
 | 
   
 
 Attention Geometry 
 | 
   
 
 High-Level Feature Align. 
 | 
   
 
 Pseudo-Label Cross View 
 | 
   
 
 Data-Augmentation 
 | 
   
 
 Semantic-Guided Structure 
 | 
   
 
 Geometric Representation 
 | 

 
 NVS-MonoDepth  [ 34 ] | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Besl and McKay  [ 35 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Brox et al.  [ 36 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Chen et al.  [ 37 ] | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Chen et al.  [ 38 ] | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Chen and Medioni  [ 39 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Chen et al.  [ 26 ] | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Segflow [ 27 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 

 
 MVS 2   [ 40 ] | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 TransMVSNet  [ 16 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 PatchMVSNet  [ 41 ] | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Gong et al.  [ 42 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Dovesi et al.  [ 31 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 

 
 Eigen and Fergus  [ 30 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Engel et al.  [ 43 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Fan et al.  [ 44 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 

 
 Felzenszwalb and Huttenlocher  [ 45 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Fouhey et al.  [ 46 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Gallup et al.  [ 47 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Garg et al.  [ 22 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 

 
 Godard et al.  [ 29 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Godard et al.  [ 48 ] | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 CasMVSNet  [ 15 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Guo et al.  [ 49 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 M 3 VSNet  [ 25 ] | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 

 
 Kendall et al.  [ 17 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Khot et al.  [ 50 ] | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Kusupati et al.  [ 51 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 —- | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Lee et al.  [ 52 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 

 
 DS-MVSNet  [ 53 ] | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Liu et al.  [ 54 ] | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 

 
 Liu et al.  [ 55 ] | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Mahjourian et al.  [ 28 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Loung and Faugeras  [ 8 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Mallick et al.  [ 56 ] | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Naderi et al.  [ 33 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 GeoNet  [ 57 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Ruhkamp et al.  [ 58 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Rusinkiewicz and Levoy  [ 59 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Saito and Kanade  [ 60 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Shim and Kim  [ 61 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 

 
 Silberman et al.  [ 62 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 LoFTR  [ 63 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Szeliski and Golland  [ 64 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Uras et al.  [ 65 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 GC-MVSNet  [ 66 ] | 
 ✓ \checkmark | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Wang et al.  [ 67 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Wang et al.  [ 68 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Motif-GCNs  [ 69 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Xu et al.  [ 70 ] | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 

 
 Xu et al.  [ 21 ] | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 

 
 Xu et al.  [ 71 ] | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Yang et al.  [ 24 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 

 
 Yang et al.  [ 72 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Yang et al.  [ 23 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 MVSNet   [ 14 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Yin et al.  [ 73 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 GeoNet  [ 74 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 p 2 Net  [ 75 ] | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Zhao et al.  [ 76 ] | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Zhao et al.  [ 77 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 

 
 Zhu et al.  [ 78 ] | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 

 
 Zhu et al.  [ 79 ] | 
 ✓ \checkmark | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 
 – | 

 

 Table 2: Most methods use multiple constraints spanning various sub-categories, problem types,
and datasets, meaningful qualitative or quantitative comparisons are difficult.
This table highlights the use of sub-constraints across all methods providing a clear categorization. 
 
 
 This survey is organized in 8 8 sections. Starting from Sec. 2 ,
we discuss the broad classification of geometric constraints presented in our taxonomy
shown in
Fig. 2 . For most of the sections, we first describe
the most common mathematical formulation of the geometry-inspired concept that covers the majority
of the methods, and then we describe different modifications applied
to it by specific methods.
Sec. 2 describes the traditional plane sweep
algorithm and its variants.
Sec. 3 focuses on all such geometric
constraints that use alternate view(s) for enforcing consistency (cross-view consistency).
Sec. 4 delves into geometric constraints that enforce
structural consistency between a reference image and a target image to
preserve the structural integrity of the scenes.
Sec. 5 focuses on the orthogonal relation between
depth and surface normal to guide geometric consistency.
Sec. 6 discusses the integration
of geometric constraints in attention mechanism and Sec. 7 
presents the methods to enforce geometry-based representation learning in
deep neural networks. We present our conclusion in Sec. 8 .

 
 
 

## 2 Plane Sweep Algorithm

 
 Stereophotogrammetry is the process of estimating the 3D coordinates of
points on an object by utilizing measurements from two or more images
of the object taken from different positions [ 1 ] .
This involves stereo matching where two or more images are used to
find matching pixels in the images and to convert their 2D
positions into 3D depths [ 9 ] .
The process of finding matching pixels is based on the
 geometry of stereo matching (epipolar geometry), i.e the process of
computing the range of possible locations of a pixel in one image
that might appear in another image. In this section, we first discuss
epipolar geometry for a pair of rectified images, and then describe a general resampling algorithm
( plane sweep algorithm ) that can be used to perform multi-image stereo-matching
with arbitrary camera configurations.

 
 
 Figure 3: Epipolar geometry, Left: epipolar line segment corresponding to one ray (Fig. reprinted from [ 9 ] ),
Center: the corresponding set of epipolar lines and their epipolar plane (Fig. reprinted from [ 9 ] ),
 R R , t t are the rotation and translation parameters, and l i l_{i} 
is the epipolar line segment. Right: illustration of the plane-sweep algorithm (Fig. reprinted from [ 79 ] ) 
 
 
 Fig. 3 (left) shows the epipolar constraints
of how a pixel in one image x 0 x_{0} projects to an epipolar line segment in the
other image. The line segment is bounded by p ∞ p_{\infty} – projection of the
original viewing ray, and c 0 c_{0} – projection of the original camera center into
the second camera, called the epipole e 1 e_{1} . The projections of the epipolar
line in the second image back into the first image give us another line segment
bounded by the corresponding epipole e 0 e_{0} . The extension of these two line
segments to infinity gives us a pair of corresponding epipolar lines,
Fig. 3 (center), that are the intersection
of the two image planes, epipolar planes , that pass through
both camera centers c 0 c_{0} and c 1 c_{1} . p p is the point of interest
 [ 7 , 8 ] .

 
 
 Multi-image stereo reconstruction is a process to recover the 3D scene structure
from multiple overlapping images with known intrinsic ( K K ) and extrinsic ( E E )
camera parameters [ 20 ] . More precisely, the photoconsistency is analyzed after
projecting the source images to fronto-parallel planes of the reference camera frustum.
The fronto-parallel view of an image refers to the outcome
of rectifying warped images. This allows the image to be viewed as it appears when
directly in front of the observer’s eyes (the reference camera is the observer). The reference camera
frustum is a pyramid-shaped volume that determines what a reference camera can see
and render in 3D space. The reference camera is at the top of the pyramid, and the frustum
extends out in the direction the camera is looking, see Fig. 3 (right).
This process is commonly known as plane-sweep 
 [ 20 , 64 , 60 , 79 , 66 ] .

 
 
 The plane-sweep method is based on the premise that areas of space
where several image features’ viewing rays intersect are likely to
be the 3D location of observed scene features. In this method,
a single plane partitioned into cells is swept through the volume
of space along a line perpendicular to the plane [ 20 ] ,
perpendicular to reference camera frustum
as shown in Fig. 3 (right).
At each position of the plane along the sweeping path, the number of
intersecting viewing rays is tallied. This is done by back-projecting
features from each source image onto the sweeping plane and noting
the features that fall within some threshold of back-projected point
position. Cells with the largest tallies are hypothesized as the location of 3D
scene features and the corresponding depth hypothesis (in the case of depth maps) are
selected [ 79 , 20 ] .

 
 
 The plane-sweep method can directly estimate the disparity (for stereo)
or depth values (for MVS), but modern deep learning-based frameworks use it to create
matching volume corresponding to each source image and the reference image.
The matching volume is aggregated
based on a metric to create a cost volume.
The cost volume is then
regularized either by 3D-CNNs [ 14 , 15 , 17 , 16 ] or
by RNNs [ 80 , 81 ] .
The metric used for cost volume aggregation can vary from
method to method. For example, Huang et al. [ 82 ] compute pairwise
matching costs between a reference image and neighboring source images and fuse them
with max-pooling. MVSNet [ 14 ] , R-MVSNet [ 80 ] ,
and CasMVSNet [ 15 ] 
compute the variance
between all encoded features warped onto each sweep plane and then regularize it with
3D-UNet [ 83 ] . TransMVSNet [ 16 ] computes similarity-based
cost volume for regularization. Yang et al. [ 84 ] reuse cost
volume from previous stage, along with the partial cost volume of the
current stage in multi-stage MVS framework. They create a pyramidal
structure for the cost volume.

 
 
 Plane sweeping typically assumes surfaces to be fronto-parallel, which
causes ambiguity in matching where slanted surfaces are involved,
like urban scenes [ 47 ] . Gallup et al. [ 47 ] 
propose to perform multiple plane sweeps, where each plane sweep
is intended to take care of planar surfaces with a particular normal.
Their method is applied in three steps. First, the surface
normals of the ground and facade planes are identified by
analyzing 3D points obtained through the sparse structure
from motion. Then, a plane sweep for each surface normal is
applied, resulting in multiple depth hypotheses for each pixel
in the final depth map. Finally, the best depth/normal combination
for each pixel is selected based on a cost or by multi-label
graph cut method [ 6 , 85 , 86 ] .

 
 
 

## 3 Cross-View Constraints

 
 Cross-view constraints are applied to scenes with more than one view.
They can be applied to stereo (two views per scene)
and MVS ( N 2 N 2 view per scene) frameworks by projecting one view, either
reference or source view, to an other view or vice-versa. Once projected to
the other view, various constraints, like photometric consistency,
depth flow consistency, and view synthesis consistency, can be utilized.
An alternate way of utilizing cross-view constraints is to use forward-backward
reprojection, where one view is projected to the other view and then it is back-projected
to the first view to check the geometrical consistency of the scene.
All the sub-constraints in this section can be effectively used as part of an objective
function during optimization (see Table 1 )
and has been observed to be effective in cases like cluttered background,
repetitive patterns and texture-less regions [ 56 , 76 , 71 , 25 , 54 , 50 , 38 , 26 , 66 , 71 ] .
In this section,
we discuss all such approaches that use cross-view consistency constraints in
end-to-end deep learning-based frameworks.

 
 

### 3.1 Photometric Consistency

 
 Photometric consistency minimizes the difference between a real image
and a synthesized image from other views. The real and the synthesized
images are denoted differently for different tasks. For example, MVS tasks use the
 reference ( I r ​ e ​ f I_{ref} ) and source ( I s ​ r ​ c I_{src} )
images in place of the real and synthesized images, stereo tasks use
 left ( I L I_{L} ) and right ( I R I_{R} ) or vice-versa in place of the real and the synthesized images, and
video depth estimation tasks use the next frame ( I + I_{+} ) and the current frame ( I 0 I_{0} ) in place of the real
and the synthesized images.
For all these tasks, one view is warped to the other view using intrinsic ( K K ) and
extrinsic ( E E ) camera parameters. The warping process brings both
images in the same camera frustum for photometric consistency estimation.

 
 
 Two main variations of photometric loss are pixel photometric loss 
and gradient photometric loss . As the name suggests, pixel
photometric loss is the comparison between pixel values of these images, and
gradient photometric loss is the comparison of the gradients of these images.
Sometimes, pixel photometric loss and gradient photometric loss are combined
for a more robust form of photometric loss, called robust photometric loss .

 
 
 

 
 | 
 ℒ p ​ h ​ o ​ t ​ o p ​ i ​ x ​ e ​ l \displaystyle\mathcal{L}_{photo_{pixel}} | 
 = ‖ ( I r ​ e ​ f − I ^ s ​ r ​ c → r ​ e ​ f ) ⊙ M ‖ l ‖ M ‖ 1 \displaystyle=\frac{||(I_{ref}-\hat{I}_{src\rightarrow ref})\odot M||_{l}}{||M||_{1}} | 
 | 
 (1) | 
 
 
 | 
 ℒ p ​ h ​ o ​ t ​ o g ​ r ​ a ​ d \displaystyle\mathcal{L}_{photo_{grad}} | 
 = ‖ ( ▽ ​ I r ​ e ​ f − ▽ ​ I ^ s ​ r ​ c → r ​ e ​ f ) ⊙ M ‖ l ‖ M ‖ 1 \displaystyle=\frac{||(\triangledown I_{ref}-\triangledown\hat{I}_{src\rightarrow ref})\odot M||_{l}}{||M||_{1}} | 
 | 
 (2) | 
 
 
 | 
 ℒ p ​ h ​ o ​ t ​ o r ​ o ​ b ​ u ​ s ​ t \displaystyle\mathcal{L}_{photo_{robust}} | 
 = λ 1 . ℒ p ​ h ​ o ​ t ​ o p ​ i ​ x ​ e ​ l + λ 2 . ℒ p ​ h ​ o ​ t ​ o g ​ r ​ a ​ d \displaystyle=\lambda_{1}.\mathcal{L}_{photo_{pixel}}+\lambda_{2}.\mathcal{L}_{photo_{grad}} | 
 | 
 (3) | 
 

 
 
 where l l denotes L 1 L_{1} or L 2 L_{2} norm, M M denotes the mask, and λ 1 , λ 2 \lambda_{1},\lambda_{2} denote
scaling factors for pixel and gradient photometric losses, respectively. ⊙ \odot denotes pixel-wise
multiplication.

 
 
 There are different ways of formulating Eqs 1 , 2 
and 3 based on the choice of view to be warped. Most methods
 [ 56 , 76 , 71 , 25 , 54 , 50 ] 
warp source views to the reference view, or source-reference warp,
as shown in Eqs 1 , 2 
and 3 . But some methods
warp the reference view to source views ( reference-source warp) [ 53 ] ,
by replacing src with ref and vice-versa
in Eqs 1 , 2 and 3 .
Other methods like Dong et al. [ 41 ] use patch-wise photometric
consistency to estimate the photometric loss.
We highlight such variations of
photometric consistency formulation in the next few paragraphs.

 
 
 We start with pixel photometric loss and its variations.
Mallick et al. [ 56 ] use pixel formulation of photometric loss, Eq. 1 ,
between reference and source views to enforce geometric consistency in a self-supervised MVS method.
Zhao et al. [ 76 ] use a similar formulation of pixel photometric loss
in self-supervised monocular depth estimation problem to promote cross-view
geometric consistency.
Xu et al. [ 70 ] use a slightly different formulation of pixel photometric loss based
on the condition that a pixel in one view finds a valid pixel in another view for semi-supervised
MVS method.

 
 
 

 
 | 
 ℒ p ​ h ​ o ​ t ​ o p ​ i ​ x ​ e ​ l = Φ ⁡ ( 1 ≤ p ^ i ≤ [ H , W ] ) ​ ‖ I r ​ e ​ f ​ ( p i ) − I ^ s ​ r ​ c → r ​ e ​ f ​ ( p i ) ‖ L 2 Φ ⁡ ( 1 ≤ p ^ i ≤ [ H , W ] ) , \mathcal{L}_{photo_{pixel}}=\frac{\Phi(1\leq\hat{p}_{i}\leq[H,W])||I_{ref}(p_{i})-\hat{I}_{src\rightarrow ref}(p_{i})||_{L_{2}}}{\Phi(1\leq\hat{p}_{i}\leq[H,W])}, | 
 | 
 (4) | 
 

 
 
 where p i p_{i} denotes a pixel, Φ ⁡ ( 1 ≤ p ^ i ≤ [ H , W ] ) \Phi(1\leq\hat{p}_{i}\leq[H,W]) indicates
whether the pixel p i p_{i} can find a valid pixel p ^ i \hat{p}_{i} in
other source view. H , W H,W denote the height and width of the image, respectively.

 
 
 Li et al. [ 53 ] use a slightly different formulation of pixel photometric loss by warping
reference view to source views, instead of warping source view to the reference view, to calculate the
 L 1 L_{1} distance between reference-source depth maps,

 
 
 

 
 | 
 ℒ p ​ h ​ o ​ t ​ o p ​ i ​ x ​ e ​ l = 1 N − 1 ​ ∑ i = 1 N | I s ​ r ​ c i − I ^ r ​ e ​ f → s ​ r ​ c i | \mathcal{L}_{photo_{pixel}}=\frac{1}{N-1}\sum^{N}_{i=1}|I_{src_{i}}-\hat{I}_{ref\rightarrow src_{i}}| | 
 | 
 (5) | 
 

 
 
 where N N is the total number of source views.

 
 
 All the above-mentioned methods use pixel-wise warping operations to estimate photometric error.
Yu et al. [ 75 ] propose patch-based warping of the extracted key points. They use the
point selection strategy from Direct Sparse Odometry (DSO) [ 43 ] and define a
support domain Ω ⁡ ( P i ) \Omega(P_{i}) over each point P i P_{i} ’s local window. The photometric
error is then estimated over each support domain Ω ⁡ ( P i ) \Omega(P_{i}) , instead of a single point.
This is called patch photometric consistency .
Dong and Yao [ 41 ] applied a similar approach to estimate
patch photometric error. Unlike [ 75 ] , which uses DSO to extract key points,
they use each pixel as a key point. They define a
square patch centered on the pixel P P as Ω ⁡ ( P ) \Omega(P) . The local patch
 Ω ⁡ ( P ) \Omega(P) is small so that it can be treated as if it lies in a plane [ 75 ] and
assumed to share the same depth as the center pixel. This patch
is warped from the source view to the reference view and the L 1 L_{1} distance between pixel
values is estimated as ℒ p ​ h ​ o ​ t ​ o p ​ a ​ t ​ c ​ h \mathcal{L}_{photo_{patch}} 

 
 
 

 
 | 
 ℒ p ​ h ​ o ​ t ​ o p ​ a ​ t ​ c ​ h = 1 N − 1 ​ ∑ i = 1 N | I r ​ e ​ f − I ^ s ​ r ​ c → r ​ e ​ f | ⊙ M r ​ e ​ f , \mathcal{L}_{photo_{patch}}=\frac{1}{N-1}\sum^{N}_{i=1}|I_{ref}-\hat{I}_{src\rightarrow ref}|\odot M_{ref}, | 
 | 
 (6) | 
 

 
 
 where N N indicates the number of source views and M r ​ e ​ f M_{ref} indicates the reference view mask. ⊙ \odot denotes
element-wise multiplication.

 
 
 While pixel photometric consistency is widely used to achieve geometric
consistency across views, their performance is susceptible to changes in lighting conditions.
Change in lighting conditions makes enforcing pixel-level photometric consistency difficult, but
image gradients are more invariant to such changes. Many methods employ gradient
photometric loss alongside pixel photometric loss. Since the addition of
the gradient term makes the photometric loss more robust, it is called robust photometric loss .
MVS methods like
 [ 71 , 25 , 54 , 50 , 21 ] 
use this formulation, Eq. 3 , during end-to-end training.
 λ 1 \lambda_{1} and λ 2 \lambda_{2} are the tunable hyper-parameters in Eq. 3 .
Robust photometric loss also helps with regions of low-texture, repetitive patterns and cluttered background by
double checking the pixel consistency across multiple views.

 
 
 While most mentioned MVS methods use an asymmetric pipeline, estimating only the reference depth map
using both the reference and source RGB images,
Dai et al. [ 40 ] use a symmetric pipeline for MVS, i.e. the network
predicts depth maps of all views simultaneously. With N N depth estimates, one per view,
the method uses a bidirectional calculation of photometric consistency
between each pair of views, called cross-view consistency loss . They do not
use robust formulations for cross-view consistency.

 
 
 

### 3.2 Geometric Consistency

 
 Like photometric consistency, geometric consistency also involves
cross-view consistency checks with projections. For photometric consistency,
one view, either reference or source, is warped to another view to calculate
the consistency error. Geometric consistency employs forward-backward reprojection (FBR)
to estimate the error. FBR involves a series of projections of the reference view
to estimate the geometric error, as shown in Alg. 1 .
First, we project the reference image ( I r ​ e ​ f I_{ref} ) to the source view ( I r ​ e ​ f → s ​ r ​ c I_{ref\rightarrow src} ), then
we remap the projected reference view I r ​ e ​ f → s ​ r ​ c I_{ref\rightarrow src} to generate I s ​ r ​ c r ​ e ​ m ​ a ​ p I_{{src}_{remap}} , and finally
we back-project I s ​ r ​ c r ​ e ​ m ​ a ​ p I_{{src}_{remap}} to the reference view to obtain I r ​ e ​ f ⇄ s ​ r ​ c I_{ref\rightleftarrows src} .
 I r ​ e ​ f ⇄ s ​ r ​ c I_{ref\rightleftarrows src} is then compared with the original I r ​ e ​ f I_{ref} to estimate the photometric error.

 
 
 
 
 
 
 Inputs: I r ​ e ​ f : ref. image , c r ​ e ​ f : ref. camera params. , I s ​ r ​ c : src. image , c s ​ r ​ c : src. camera params. I_{ref}:\text{ref. image},c_{ref}:\text{ref. camera params.},I_{src}:\text{src. image},c_{src}:\text{src. camera params.} 

 
 
 
 
 K r ​ e ​ f , E r ​ e ​ f ← c r ​ e ​ f K_{ref},E_{ref}\leftarrow c_{ref} ; K s ​ r ​ c , E s ​ r ​ c ← c s ​ r ​ c K_{src},E_{src}\leftarrow c_{src} 

 
 
 
 
 I ( r ​ e ​ f → s ​ r ​ c ) ← K s ​ r ​ c ⋅ E s ​ r ​ c ⋅ E r ​ e ​ f − 1 ⋅ K r ​ e ​ f − 1 ⋅ I r ​ e ​ f I_{(ref\rightarrow src)}\leftarrow K_{src}\cdot E_{src}\cdot E_{ref}^{-1}\cdot K_{ref}^{-1}\cdot I_{ref} ⊳ \triangleright Project
 
 
 
 
 
 X I ( r ​ e ​ f → s ​ r ​ c ) , Y I ( r ​ e ​ f → s ​ r ​ c ) ← I ( r ​ e ​ f → s ​ r ​ c ) X_{I_{(ref\rightarrow src)}},Y_{I_{(ref\rightarrow src)}}\leftarrow I_{(ref\rightarrow src)} 

 
 
 
 
 I s ​ r ​ c r ​ e ​ m ​ a ​ p ← R ​ E ​ M ​ A ​ P ​ ( I s ​ r ​ c , X I ( r ​ e ​ f → s ​ r ​ c ) , Y I ( r ​ e ​ f → s ​ r ​ c ) ) I_{{src}_{remap}}\leftarrow REMAP(I_{src},X_{I_{(ref\rightarrow src)}},Y_{I_{(ref\rightarrow src)}}) ⊳ \triangleright Remap
 
 
 
 
 
 I r ​ e ​ f ⇄ s ​ r ​ c ← K r ​ e ​ f ⋅ E r ​ e ​ f ⋅ E s ​ r ​ c − 1 ⋅ K s ​ r ​ c − 1 ⋅ I s ​ r ​ c r ​ e ​ m ​ a ​ p I_{ref\rightleftarrows src}\leftarrow K_{ref}\cdot E_{ref}\cdot E_{src}^{-1}\cdot K_{src}^{-1}\cdot I_{{src}_{remap}} ⊳ \triangleright Back project
 
 
 
 Algorithm 1 Forward Backward Reprojection (FBR) 
 
 
 Dong and Yao [ 41 ] use cross-view geometric consistency
by applying FBR in the pixel domain in an unsupervised MVS pipeline. Once the FBR steps are done,
the actual pixel values between the original reference image I r I_{r} and back-projected reference images
 I r ⇄ s I_{r\rightleftarrows s} are used to check the geometrical consistency of the depth estimates, which diminishes the
matching ambiguity between reference and source views,

 
 
 

 
 | 
 ℒ g ​ e ​ o ​ m ​ e ​ t ​ r ​ i ​ c = 1 N ​ ∑ i = 1 N | I r ​ e ​ f − I ^ r ​ e ​ f ⇄ s ​ r ​ c i | ⊙ M r ​ e ​ f , \mathcal{L}_{geometric}=\frac{1}{N}\sum^{N}_{i=1}|I_{ref}-\hat{I}^{i}_{ref\rightleftarrows src}|\odot M_{ref}, | 
 | 
 (7) | 
 

 
 
 where N N indicates the number of source views and M r ​ e ​ f M_{ref} denotes the reference view mask. ⊙ \odot is the
element-wise multiplication operation.

 
 
 Geometric consistency can be extended to the 3D coordinates of the camera using the
concept of back-projection [ 38 , 26 , 66 ] . FBR has
also been found helpful in images with low-texture regions, occlusion and repetitive patterns [ 66 , 26 ] .
Chen et al. [ 26 ] apply this modified geometric
consistency in 3D space for a video depth estimation problem, called 3D geometric consistency .
For a given source image pixel P s ​ r ​ c P_{src} and corresponding reference image pixel P r ​ e ​ f P_{ref} , their
3D coordinates are obtained by back projection. The inconsistency between the estimates of the same
3D point is then estimated and used as a penalty. The loss value represents the 3D discrepancy
of predictions from two views. Chen et al. [ 38 ] use a similar formulation to overcome
the deficiencies of photometric loss in a self-supervised monocular depth estimation framework.

 
 
 Figure 4: Intuition of Depth2Flow module. The relative motion of
a moving camera can be viewed as a moving object and a still camera to
estimate optical flow. Figure reprinted from [ 71 ] 
 
 
 

### 3.3 Cross-View Depth-Flow Consistency

 
 Depth-flow estimations are typically used in optical flow problems [ 55 , 87 ] .
But they can easily be adapted to MVS problems by estimating flow from estimated depth maps
as well as from input RGB images and comparing them. Xu et al. [ 71 ] propose a
novel flow-depth consistency loss to regularize the ambiguous supervision in the foreground of depth maps.
Estimation of flow-depth consistency loss requires two modules for an MVS method, RGB2Flow and Depth2Flow.
As the name suggests, the Depth2Flow module transforms the estimated depth maps to virtual optical flow fields between
the reference and arbitrary source view and the RGB2Flow module uses [ 55 ] to predict the
optical flow from the corresponding reference-source pairs. The two predicted flows should be consistent
with each other.

 
 
 Depth2Flow module: 

 
 In an MVS system, cameras move around the object to collect multi-view images. If we consider
the relative motion between the object and the camera, the camera can be assumed to be fixed
and the object can be assumed to be in motion toward the virtually still camera, see Fig. 4 .
This virtual motion can be represented as a dense 3D optical flow and should be
consistent with the 3D correspondence in real MVS systems. The virtual flow for a pixel p i p_{i} 
can be defined as,

 
 
 

 
 | 
 F ^ r ​ e ​ f → s ​ r ​ c = N o r m [ K s ​ r ​ c . E s ​ r ​ c ( K r ​ e ​ f . T r ​ e ​ f ) − 1 I r ​ e ​ f ( p i ) ] − p i , \hat{F}_{ref\rightarrow src}=Norm[K_{src}.E_{src}(K_{ref}.T_{ref})^{-1}I_{ref}(p_{i})]-p_{i}, | 
 | 
 (8) | 
 

 
 
 
 RGB2Flow module: 

 
 To estimate the optical flow from RGB input images, a pre-trained model can be used. Xu et al. [ 71 ] 
use [ 55 ] to estimate the forward flow (reference to source image flow, F r ​ e ​ f → s ​ r ​ c F_{ref\rightarrow src} )
and backward-flow (source to reference view flow, F s ​ r ​ c → r ​ e ​ f F_{src\rightarrow ref} ).
Each source view is combined with one reference view to form a pair to obtain a forward and backward flow i.e.
 F r ​ e ​ f → s ​ r ​ c F_{ref\rightarrow src} and F s ​ r ​ c → r ​ e ​ f F_{src\rightarrow ref} . The consistency of both the
flows ( F r ​ e ​ f → s ​ r ​ c F_{ref\rightarrow src} and F s ​ r ​ c → r ​ e ​ f F_{src\rightarrow ref} )
are compared with the virtual flow ( F ^ r ​ e ​ f → s ​ r ​ c \hat{F}_{ref\rightarrow src} ) to estimate
the flow-depth consistency loss. The occluded pixels from different source views are masked out with
mask M r ​ e ​ f → s ​ r ​ c M_{ref\rightarrow src} and the error is given as,

 
 
 

 
 | 
 M r ​ e ​ f → s ​ r ​ c \displaystyle M_{ref\rightarrow src} | 
 = { | F r ​ e ​ f → s ​ r ​ c + F s ​ r ​ c → r ​ e ​ f | ϵ } \displaystyle=\{|F_{ref\rightarrow src}+F_{src\rightarrow ref}| \epsilon\} | 
 | 
 (9) | 
 
 
 | 
 ℒ f ​ l ​ o ​ w \displaystyle\mathcal{L}_{flow} | 
 = ∑ i = 1 H ​ W m ​ i ​ n 2 ≤ s j ≤ V ​ | | F r ​ e ​ f → s ​ r ​ c j ( p i ) − F ^ r ​ e ​ f → s ​ r ​ c j ( p i ) . M ( p i ) | | 2 ∑ i = 1 H ​ W M r ​ e ​ f → s ​ r ​ c j ​ ( p i ) , \displaystyle=\sum^{HW}_{i=1}\genfrac{}{}{0.0pt}{}{min}{2 \leq s_j \leq V}\frac{||F_{ref\rightarrow{src}_{j}}(p_{i})-\hat{F}_{ref\rightarrow{src}_{j}}(p_{i}).M(p_{i})||_{2}}{\sum^{HW}_{i=1}M_{ref\rightarrow{src}_{j}}(p_{i})}, | 
 | 
 (10) | 
 

 
 
 where ϵ \epsilon is a threshold set to 0.5, and H H and W W are the height and width of the image. Instead of averaging
the difference between F r ​ e ​ f → s ​ r ​ c F_{ref\rightarrow src} and F ^ r ​ e ​ f → s ​ r ​ c \hat{F}_{ref\rightarrow src} on all source views, a minimum error
is estimated at each pixel, see Eq 10 . Godard et al. [ 48 ] 
introduced the minimum error calculation
to reject occluded pixels in depth estimation. Both the modules,
Depth2Flow and RGB2Flow, are fully differentiable
and can be used in end-to-end training setups.

 
 
 Chen et al. [ 37 ] use point-flow information to refine the
estimated depth map in an MVS framework. Using extracted features from the
input images and the previous stage’s estimated depth map, they generate
a feature augmented point cloud and use a point-flow module
to learn the depth residual to refine the previously estimated depth map.
The point-flow module estimates the displacement of a point to the ground
truth surface along the reference camera direction by observing the
neighboring points from all views.

 
 
 
 

### 3.4 View Synthesis Consistency

 
 Figure 5: (a) Steps of view synthesis using RGB image and its depth map.
(b) shows a bilinear interpolation step for adjusting values to pixel space. R and t are rotation
and translation parameters.
Figure reprinted from [ 23 ] 
 
 
 Most of the monocular, stereo, and MVS depth estimation frameworks use ground truth as a
supervision signal. While these frameworks may utilize the additional source view
images in the pipeline, they always estimate only one depth map, the reference view depth map.
Estimation and use of only one depth map may not provide enough geometric information
for consistent estimation. To address this gap, many methods
 [ 34 , 23 , 53 , 40 ] 
synthesize an additional view (see Fig. 5 ), either a
depth map or an RGB image (commonly referred to as target view), using camera parameters and
reference view information. This additional view, when included in the training framework,
provides additional geometric consistency information during the learning process.

 
 
 Bauer et al. [ 34 ] use view synthesis in a monocular depth estimation framework.
They apply two networks, depth network (DepNet) and synthesis network (SynNet) in a series of
operations to enforce geometric constraints with view synthesis.
First, the RGB input (source view) is used in DepNet to generate a corresponding
depth estimate. The estimated depth map is projected to a target view and, using
SynNet, the holes in the target view are filled. Finally, the synthesized RGB target
view is used as input to DepNet to estimate its depth map. This ensures that
the DepNet learns to estimate geometrically consistent depth estimates of both
the source and the synthesized target view. They use an L 1 L_{1} loss to enforce consistency.

 
 
 Yang et al. [ 23 ] also synthesize RGB target view to improve
geometric consistency in video depth estimation. With estimated pixel matching pairs
between source ( I s I_{s} ) and target views ( I t I_{t} ), they synthesize a target view I ^ s \hat{I}_{s} 
using the source view, camera parameters, and bilinear interpolation [ 22 ] .
To handle occlusion and movement of objects, an explainability mask ( M s M_{s} ) is applied
during the loss calculation,

 
 
 

 
 | 
 ℒ V ​ i ​ e ​ w ​ S ​ y ​ n ​ t ​ h ​ e ​ s ​ i ​ s = ∑ s = 1 S | I t − I ^ s | ⊙ M s , \mathcal{L}_{ViewSynthesis}=\sum^{S}_{s=1}|I_{t}-\hat{I}_{s}|\odot M_{s}, | 
 | 
 (11) | 
 

 
 
 where s s are the source views.

 
 
 Typically, MVS methods regularize a cost volume that is generated using multiple
source views, to estimate the reference view depth map.
Li et al. [ 53 ] argue that by estimating only the reference
view depth map from the cost volume, an MVS method underutilizes the information present in the
cost volume. They propose a module to synthesize source depths
from the cost volume by projecting the pixels in the reference view at different
depth hypotheses to the source views. They use robust photometric consistency
(Sec. 3.1 ) to estimate the view synthesis loss.

 
 
 Dai et al. [ 40 ] synthesize source view depth maps using the reference
view depth maps in the MVS pipeline for additional geometric consistency. For each
pair of reference-source views, they calculate the bidirectional error, i.e.
 ℒ r ​ e ​ f → s ​ r ​ c \mathcal{L}_{ref\rightarrow src} and ℒ s ​ r ​ c → r ​ e ​ f \mathcal{L}_{src\rightarrow ref} along
with ℒ s ​ m ​ o ​ o ​ t ​ h ▽ \mathcal{L}_{smooth_{\triangledown}} and ℒ s ​ m ​ o ​ o ​ t ​ h ▽ 2 \mathcal{L}_{smooth_{\triangledown^{2}}} (Sec. 4.2 ) to estimate
 ℒ V ​ i ​ e ​ w ​ S ​ y ​ n ​ t ​ h ​ e ​ s ​ i ​ s \mathcal{L}_{ViewSynthesis} . They use a robust formulation of photometric consistency loss
(Sec. 3.1 ) to estimate ℒ r ​ e ​ f → s ​ r ​ c \mathcal{L}_{ref\rightarrow src} and ℒ s ​ r ​ c → r ​ e ​ f \mathcal{L}_{src\rightarrow ref} 
and add structural similarity (SSIM) error term
(Sec. 4.1 ) in the loss,

 
 
 

 
 | 
 ℒ V ​ i ​ e ​ w ​ S ​ y ​ n ​ t ​ h ​ e ​ s ​ i ​ s r ​ e ​ f → s ​ r ​ c \displaystyle\mathcal{L}_{ViewSynthesis}^{ref\rightarrow src} | 
 = ( ℒ r ​ e ​ f → s ​ r ​ c + ℒ s ​ r ​ c → r ​ e ​ f ) + ( ℒ s ​ m ​ o ​ o ​ t ​ h ▽ r ​ e ​ f + ℒ s ​ m ​ o ​ o ​ t ​ h ▽ 2 r ​ e ​ f ) \displaystyle=(\mathcal{L}_{ref\rightarrow src}+\mathcal{L}_{src\rightarrow ref})+(\mathcal{L}^{ref}_{smooth_{\triangledown}}+\mathcal{L}^{ref}_{smooth_{\triangledown^{2}}}) | 
 | 
 (12) | 
 
 
 | 
 ℒ r ​ e ​ f → s ​ r ​ c \displaystyle\mathcal{L}_{ref\rightarrow src} | 
 = ℒ p ​ h ​ o ​ t ​ o r ​ o ​ b ​ u ​ s ​ t + ℒ S ​ S ​ I ​ M r ​ e ​ f \displaystyle=\mathcal{L}_{photo_{robust}}+\mathcal{L}^{ref}_{SSIM} | 
 | 
 (13) | 
 
 
 | 
 ℒ s ​ r ​ c → r ​ e ​ f \displaystyle\mathcal{L}_{src\rightarrow ref} | 
 = ℒ p ​ h ​ o ​ t ​ o r ​ o ​ b ​ u ​ s ​ t + ℒ S ​ S ​ I ​ M s ​ r ​ c \displaystyle=\mathcal{L}_{photo_{robust}}+\mathcal{L}^{src}_{SSIM} | 
 | 
 (14) | 
 

 
 
 where r ​ e ​ f , s ​ r ​ c ref,src are the reference and the source views. We describe the other loss terms in Sec. 4 .

 
 
 
 

## 4 Geometry Preserving Constraints

 
 Besides cross-view consistency constraints, there are other
ways of enforcing structural consistency in a depth estimation pipeline.
In this section, we discuss all such approaches that utilize alternative
methods of enforcing geometric constraints. We have classified these
methods into four broader categories, i.e. structural similarity index measurement (SSIM),
edge-aware smoothness constraints, consistency regularization, and
planar consistency. All these constraints can help with problems like
cluttered background, repetitive patterns or texture-less regions.
They can also be integrated within an objective function for enhanced geometric learning.
We discuss each of these approaches in detail and highlight
research that uses these methods in their pipeline.

 
 

### 4.1 Structural Similarity Index Measurement

 
 Figure 6: SSIM measurement system as described in [ 68 ] . Figure reprinted from [ 68 ] 
 
 
 Structural similarity index measurement (SSIM)
is a full-reference image quality assessment technique [ 68 ] .
Its assessment is based on the degradation of structural information between the
reference and noisy image. Specifically, it compares local patterns of
pixel intensities that have been normalized for luminance and contrast.
Luminance of a surface is the product of its illumination and reflectance, but
the structure of the object is independent of illumination. SSIM separates
the influence of illumination to analyze the structural information
in an image [ 68 , 88 , 89 , 56 , 69 , 42 ] .

 
 
 Objective image quality metrics can be roughly classified into three categories
based on the availability of distortion-free (original or reference) images.
The metric is called full-reference when the complete reference
image is known, no-reference when the reference image
is not available, and reduced-reference when the reference
image is partially available. Eskicioglu and Fisher [ 88 ] 
discuss several such image quality metrics and their performance.

 
 
 Wang et al. [ 68 ] define structural information as the
attributes that represent the structural information of objects in an image,
independent of average luminance and contrast. As shown in Fig. 6 ,
given two aligned images, X and Y, one
of which is assumed to be the reference quality, we can have a quantitative
measurement of the second signal with SSIM. Three separate tasks are considered
for structural similarity measurement: luminance, contrast, and structure [ 68 ] .

 
 
 The mathematical formulation of SSIM,

 
 
 

 
 | 
 S ​ S ​ I ​ M ​ ( X , Y ) \displaystyle SSIM(X,Y) | 
 = [ l ⁡ ( X , Y ) ] α ​ [ c ⁡ ( X , Y ) ] β ​ [ s ⁡ ( X , Y ) ] γ ; α 0 , β 0 , γ 0 \displaystyle=[l(X,Y)]^{\alpha}[c(X,Y)]^{\beta}[s(X,Y)]^{\gamma};\alpha 0,\beta 0,\gamma 0 | 
 | 
 (15) | 
 
 
 | 
 l ⁡ ( X , Y ) \displaystyle l(X,Y) | 
 = 2 ​ μ x ​ μ y + c 1 μ x 2 + μ y 2 + c 1 ; c 1 = ( K 1 ​ L ) 2 , K 1 ≪ 1 \displaystyle=\frac{2\mu_{x}\mu_{y}+c_{1}}{\mu_{x}^{2}+\mu_{y}^{2}+c_{1}};c_{1}=(K_{1}L)^{2},K_{1}\ll 1 | 
 | 
 (16) | 
 
 
 | 
 c ⁡ ( X , Y ) \displaystyle c(X,Y) | 
 = 2 ​ σ x ​ σ y + c 2 σ x 2 + σ y 2 + c 2 ; c 2 = ( K 2 ​ L ) 2 , K 2 ≪ 1 \displaystyle=\frac{2\sigma_{x}\sigma_{y}+c_{2}}{\sigma^{2}_{x}+\sigma^{2}_{y}+c_{2}};c_{2}=(K_{2}L)^{2},K_{2}\ll 1 | 
 | 
 (17) | 
 
 
 | 
 s ⁡ ( X , Y ) \displaystyle s(X,Y) | 
 = σ x ​ y + c 3 σ x ​ σ y + c 3 , c 3 = c 2 2 , \displaystyle=\frac{\sigma_{xy}+c_{3}}{\sigma_{x}\sigma_{y}+c_{3}},c_{3}=\frac{c_{2}}{2}, | 
 | 
 (18) | 
 

 
 
 is a scaled product of the three components. Where the luminance function l ⁡ ( X , Y ) l(X,Y) is a function of mean intensities
 μ x \mu_{x} and μ y \mu_{y} as shown in Eq. 16 . The contrast comparison c ⁡ ( X , Y ) c(X,Y) 
is a function of σ x \sigma_{x} and σ y \sigma_{y} as shown in Eq. 17 and the
structure measure s ⁡ ( X , Y ) s(X,Y) is a function of correlation coefficient σ x ​ y \sigma_{xy} ,
 σ x \sigma_{x} and σ y \sigma_{y} as shown in Eq. 18 . L L is the dynamic range
of the pixel values (255 for grayscale image). The mean, standard deviation, and correlation coefficient
of the signals are calculated using Eqs. 19 , 20 , and 21 .
Where N N is the total number of pixels in an image.

 
 
 

 
 | 
 | 
 μ x = 1 N ​ ∑ i = 1 N x i ; μ y = 1 N ​ ∑ i = 1 N y i \displaystyle\mu_{x}=\frac{1}{N}\sum^{N}_{i=1}x_{i};\mu_{y}=\frac{1}{N}\sum^{N}_{i=1}y_{i} | 
 | 
 (19) | 
 
 
 | 
 | 
 σ x = ( 1 N − 1 ​ ∑ i = 1 N ( x i − μ x ) 2 ) 1 2 ; σ y = ( 1 N − 1 ​ ∑ i = 1 N ( y i − μ y ) 2 ) 1 2 \displaystyle\sigma_{x}=(\frac{1}{N-1}\sum^{N}_{i=1}(x_{i}-\mu_{x})^{2})^{\frac{1}{2}};\sigma_{y}=(\frac{1}{N-1}\sum^{N}_{i=1}(y_{i}-\mu_{y})^{2})^{\frac{1}{2}} | 
 | 
 (20) | 
 
 
 | 
 | 
 σ x ​ y = 1 N − 1 ​ ∑ i = 1 N ( x i − μ x ) ​ ( y i − μ y ) \displaystyle\sigma_{xy}=\frac{1}{N-1}\sum^{N}_{i=1}(x_{i}-\mu_{x})(y_{i}-\mu_{y}) | 
 | 
 (21) | 
 

 
 
 SSIM should be computed locally rather than globally for several reasons. First,
image statistical features are highly spatially non-stationary i.e. statistical features
are not the same across different spatial locations within an image. Second, image distortions
may be space-variant, and third, local quality measurement delivers more information about
the quality degradation by providing a spatially varying quality map of the image.

 
 
 Wang et al. [ 68 ] defined S ​ S ​ I ​ M SSIM for
two signals in the same domain, i.e.
it can estimate the structural similarity between two RGB or grayscale images or
between two depth maps or two patches. In an end-to-end framework,
where we minimized the loss, we
want to maximize S ​ S ​ I ​ M SSIM for better result.
Since the upper bound of S ​ S ​ I ​ M SSIM is 1 1 ,
we can instead minimize 1 − S ​ S ​ I ​ M 1-SSIM ,

 
 
 

 
 | 
 ℒ S ​ S ​ I ​ M = 1 N ​ ∑ i = 1 N 1 − S ​ S ​ I ​ M ​ ( X , Y ) K ⊙ M \mathcal{L}_{SSIM}=\frac{1}{N}\sum^{N}_{i=1}\frac{1-SSIM(X,Y)}{K}\odot M | 
 | 
 (22) | 
 

 
 
 where N N is the number of source views, and X X and Y Y 
are the two images to be compared. M M is the mask to handle occlusion
and K K is a constant.

 
 
 Zhao et al. [ 89 ] use this formulation
with no mask and K = 1 K=1 for image restoration problems. They calculate the means and
standard deviations (Eqs. 19 , 20 and 21 )
using a Gaussian filter with standard deviation σ G \sigma_{G} . The choice of σ G \sigma_{G} 
impacts the quality of the processed results. With smaller values of σ G \sigma_{G} the
network loses the ability to preserve local structures which introduces
artifacts in the image, and for larger σ G \sigma_{G} , the network preserves noises around the edges. Instead of
finetuning the value of σ G \sigma_{G} , Zhao et al. [ 89 ] propose a
multi-scale formulation of SSIM (MS-SSIM), where all results from the variations of
 σ G \sigma_{G} are multiplied together.

 
 
 Inherently, both MS-SSIM and SSIM are not particularly sensitive to the change of
brightness or shift of colors. However, they preserve the contrast in high-frequency
regions. L 1 L_{1} loss, on the other hand, preserves colors and luminance - an error
weighted equally regardless of the local structure - but does not produce the same impact
for contrast. For best impact, Zhao et al. [ 89 ] combines both these
terms,

 
 
 

 
 | 
 ℒ S ​ S ​ I ​ M m ​ i ​ x = α . ℒ S ​ S ​ I ​ M + ( 1 − α ) . ℒ l , \mathcal{L}^{mix}_{SSIM}=\alpha.\mathcal{L}_{SSIM}+(1-\alpha).\mathcal{L}^{l}, | 
 | 
 (23) | 
 

 
 
 where, α \alpha is a tunable hyper-parameter and l l is L 1 L_{1} or L 2 L_{2} norm.

 
 
 Mallick et al. [ 56 ] use the simplest form of ℒ S ​ S ​ I ​ M \mathcal{L}_{SSIM} (Eq. 22 )
in a self-supervised MVS framework with no mask M M , X = I r ​ e ​ f X=I_{ref} , Y = I ^ s ​ r ​ c Y=\hat{I}_{src} and K = 1 K=1 .
Huang et al. [ 25 ] use ℒ S ​ S ​ I ​ M \mathcal{L}_{SSIM} (Eq. 22 )
with X = I r ​ e ​ f X=I_{ref} , Y = I ^ s ​ r ​ c Y=\hat{I}_{src} and K = 2 K=2 in unsupervised MVS framework. Mahjourian et al.
 [ 28 ] also use it in unsupervised MVS framework with K = 1 K=1 (Eq. 22 ),
 c 1 = 0.01 2 c_{1}={0.01}^{2} and c 2 = 0.03 2 c_{2}={0.03}^{2} (Eqs. 16 and 17 ).
Khot et al. [ 50 ] use the same formulation as [ 28 ] in an
unsupervised MVS framework. Li et al. [ 53 ] use bidirectional calculation,
forward ( ℒ S ​ S ​ I ​ M f ​ o ​ r ​ w ​ a ​ r ​ d \mathcal{L}^{forward}_{SSIM} ) with X = I r ​ e ​ f X=I_{ref} , Y = I ^ s ​ r ​ c Y=\hat{I}_{src} (source view projected to reference view)
and backward ( ℒ S ​ S ​ I ​ M b ​ a ​ c ​ k ​ w ​ a ​ r ​ d \mathcal{L}^{backward}_{SSIM} ) with X = I ^ r ​ e ​ f X=\hat{I}_{ref} , Y = I s ​ r ​ c Y=I_{src} (reference view projected to source view),
with no mask M M and K = 1 K=1 , Eq. 22 . The final value of ℒ S ​ S ​ I ​ M \mathcal{L}_{SSIM} is the
summation of the forward and the backward components.

 
 
 As explained earlier, ℒ S ​ S ​ I ​ M \mathcal{L}_{SSIM} is usually
combined with a uniformly weighted loss function like L 1 L_{1} and L 2 L_{2} , to choose
the best of both the individual loss terms. The
combined formulation is shown in Eq. 23 . Dai et al. [ 40 ] and Yu et al. [ 75 ] use
Eq. 23 with α = 0.85 \alpha=0.85 in an MVS framework. This formulation
finds widespread use in monocular depth estimation problems. Monocular methods like
 [ 67 , 74 ] 
use Eq. 23 with α = 0.85 \alpha=0.85 with target and novel view synthesized depth
maps as X X and Y Y . Godard et al. [ 29 ] use ℒ S ​ S ​ I ​ M m ​ i ​ x \mathcal{L}^{mix}_{SSIM} 
(Eq. 23 ) between the input RGB image and reconstructed new view RGB image with α = 0.85 \alpha=0.85 
to enforce structural similarity. Zhao et al. [ 77 ] with symmetric domain adaptation,
real-to-synthetic and synthetic-to-real, apply Eq. 23 both ways to enforce
structural similarity during the transition from one domain to the other.

 
 
 Chen et al. [ 26 ] use ℒ S ​ S ​ I ​ M m ​ i ​ x \mathcal{L}^{mix}_{SSIM} , Eq. 23 , with
a slight modification to make it an adaptive loss in optical flow estimation. For scene
structures that can not be explained by global rigid motion, it adapts to a more flexible
optical flow estimation by updating the channel parameters only for the configurations
that closely explain the displacement. It is represented as the minimum
error between the optical flow and the rigid motion displacements,

 
 
 

 
 | 
 L S ​ S ​ I ​ M a ​ d ​ a ​ p ​ t ​ i ​ v ​ e = m ​ i ​ n ​ { ℒ S ​ S ​ I ​ M f ​ l ​ o ​ w m ​ i ​ x , ℒ S ​ S ​ I ​ M r ​ i ​ g ​ i ​ d m ​ i ​ x } L^{adaptive}_{SSIM}=min\{\mathcal{L}^{mix}_{SSIM_{flow}},\mathcal{L}^{mix}_{SSIM_{rigid}}\} | 
 | 
 (24) | 
 

 
 
 and it uses α = 0.85 \alpha=0.85 in Eq. 23 . We only discuss the ℒ S ​ S ​ I ​ M m ​ i ​ x \mathcal{L}^{mix}_{SSIM} formulation
here and refer to [ 26 ] for other optical flow related details.

 
 
 

### 4.2 Edge-Aware Smoothness Constraint

 
 The smoothness constraint finds its origin in the optical flow estimation problem.
It was first applied by Uras et al. [ 65 ] to estimate
consistent optical flow from two images. Brox et al. [ 36 ] 
further explained the concept under three assumptions for the optical flow framework,
i.e. the gray value constancy assumption, the gradient constancy assumption, and
the smoothness assumption.

 
 
 Since the beginning of the optical flow estimation
problem, it has been assumed that the gray value of a pixel does not
change on displacement under a given lighting condition [ 36 ] .
But brightness changes in natural scenes all the time. Therefore, it is
useful to allow small variations in gray values but find a different criterion that
remains relatively invariant under gray value changes, i.e. gradient
constancy under displacement.
This brought about the third assumption; while discontinuities are assumed to be present
at the boundaries of the object in the scene, piecewise smoothness can
be assumed in the flow field [ 36 ] . To achieve this smoothness
in flow estimation, a penalty on the flow field was applied.

 
 
 In the optical flow framework, objects are assumed to be moving with a fixed camera. In
an MVS framework, the objects are assumed to be fixed and the camera moves
around a fixed point. The relative motion of an object can be viewed
as a moving camera to pose it as an MVS problem, see Fig. 4 .
With this assumption, the smoothness
constraint can be applied to the depth estimation problem. Initially, only the
first-order smoothness constraints were used
 [ 30 , 22 , 29 , 74 , 28 , 75 ] .
Yang et al. [ 23 ] introduced a
second-order smoothness constraint for regularization along with
the first-order smoothness term. Subsequently, both the first and the second-order smoothness
terms have been used in many works
 [ 40 , 25 , 41 , 72 ] .
In the depth estimation framework, smoothness constraint is applied
between the gradient of input images ( I ) (I) and the estimated depth maps ( D ^ ) (\hat{D}) ,

 
 
 

 
 | 
 ℒ s ​ m ​ o ​ o ​ t ​ h ▽ \displaystyle\mathcal{L}_{smooth_{\triangledown}} | 
 = ∑ ‖ ∂ x D ^ ‖ . e − ‖ ∂ x I ‖ + ‖ ∂ y D ^ ‖ . e − ‖ ∂ y I ‖ \displaystyle=\sum||\partial_{x}\hat{D}||.e^{-||\partial_{x}I||}+||\partial_{y}\hat{D}||.e^{-||\partial_{y}I||} | 
 | 
 (25) | 
 
 
 | 
 ℒ s ​ m ​ o ​ o ​ t ​ h ▽ 2 \displaystyle\mathcal{L}_{smooth_{\triangledown^{2}}} | 
 = ∑ ‖ ∂ x 2 D ^ ‖ . e − ‖ ∂ x 2 I ‖ + ‖ ∂ y 2 D ^ ‖ . e − ‖ ∂ y 2 I ‖ \displaystyle=\sum||\partial^{2}_{x}\hat{D}||.e^{-||\partial^{2}_{x}I||}+||\partial^{2}_{y}\hat{D}||.e^{-||\partial^{2}_{y}I||} | 
 | 
 (26) | 
 
 
 | 
 ℒ s ​ m ​ o ​ o ​ t ​ h \displaystyle\mathcal{L}_{smooth} | 
 = α . ℒ s ​ m ​ o ​ o ​ t ​ h ▽ + β . ℒ s ​ m ​ o ​ o ​ t ​ h ▽ 2 , \displaystyle=\alpha.\mathcal{L}_{smooth_{\triangledown}}+\beta.\mathcal{L}_{smooth_{\triangledown^{2}}}, | 
 | 
 (27) | 
 

 
 
 where ( ℒ s ​ m ​ o ​ o ​ t ​ h ▽ ) (\mathcal{L}_{smooth_{\triangledown}}) and second ( ℒ s ​ m ​ o ​ o ​ t ​ h ▽ 2 ) (\mathcal{L}_{smooth_{\triangledown^{2}}}) 
represents the first and second-order smoothness constraints. ℒ s ​ m ​ o ​ o ​ t ​ h \mathcal{L}_{smooth} shows the combined
formulation with α 0 \alpha 0 and β 0 \beta 0 as a scaling factor.

 
 
 Garg et al. [ 22 ] use the first order formulation with L 2 L_{2} 
norm as a regularization term to achieve smoothness in estimation. Mahjourian et al.
 [ 28 ] use the first-order formulation for monocular video
depth estimation. Most self-supervised/unsupervised MVS frameworks
use first-order smoothness constraints [ 53 , 75 , 54 , 50 , 56 ] .
Zhao et al. [ 77 ] with its symmetric domain adaptation for
monocular depth estimation uses first-order constraint in both domains. Other
monocular depth estimation methods
 [ 29 , 74 , 90 , 67 ] 
also apply the first-order smoothness constraint
as defined in Eq. 25 .

 
 
 Yang et al. [ 23 ] only uses second order formulation,
Eq. 26 , as a regularization term in monocular video-depth
estimation framework. Learning from it, more recent MVS frameworks combine
both the first-order and the second-order formulations
 [ 25 , 40 , 41 ] ,
as shown in Eq. 27 . Inspired from
 [ 29 ] and [ 90 ] , Yang et al.
 [ 72 ] apply the combined edge-aware smoothness constraint, Eq. 27 ,
in monocular endoscopy. All of these methods use α = β = 1 \alpha=\beta=1 in Eq. 27 .

 
 
 

### 4.3 Consistency Regularization

 
 Deep learning-based frameworks inherently suffer from over-parameterization problems.
One of the most efficient methods to counter it is to regularize the loss function.
Photometric consistency, Sec. 3.1 , which enforces
geometrical consistency at the pixel level, is highly susceptible to change in lighting
conditions. Many MVS methods employ different consistency regularization techniques
to effectively handle this problem [ 23 , 22 , 70 ] .

 
 
 As discussed in Sec 4.2 , first-order and second-order gradients
are often used for this task [ 23 , 22 ] .
Garg et al. [ 22 ] argue that the photometric loss is
non-informative in homogeneous regions of a scene, which leads to multiple warps generating
similar disparity outcomes. They use L 2 L_{2} regularization
( ℒ ▽ = ‖ ▽ ​ D ^ ‖ 2 \mathcal{L}_{\triangledown}=||\triangledown\hat{D}||^{2} , where D is the disparity map)
on the disparity discontinuities as a prior,
and they also recommend using other robust penalty
functions [ 36 , 91 ] as an alternative regularization term.
Yang et al. [ 23 ] use a spatial smoothness penalty with L 1 L_{1} 
norm of the second-order gradient of depth,
 ℒ ▽ 2 = ∑ d ∈ x , y | ▽ d 2 ​ D ^ | . e − α ​ | ▽ ​ I | ; α 0 \mathcal{L}_{\triangledown^{2}}=\sum_{d\in x,y}|\triangledown^{2}_{d}\hat{D}|.e^{-\alpha|\triangledown I|};\alpha 0 . They
encourage depth values to align in the planar surface when no image gradient appear.

 
 
 Xu et al. [ 70 ] apply consistency regularization in the semi-supervised MVS
method. The proposed regularization minimizes the Kullback-Leibler (KL) divergence
between the predicted distributions of augmented ( P ​ V ^ ) (\hat{PV}) and non-augmented ( P ​ V ) (PV) samples.
With the K K depth hypothesis, the probability volume P ​ V PV , of size H × W × K H\times W\times K ,
is separated into K K categories of H ​ W HW logits. The KL regularize formulation,

 
 
 

 
 | 
 ℒ K ​ L = 1 H ​ W ​ ∑ i = 1 H ​ W K ​ L ​ ( P ​ V p i , P ​ V ^ p i ) , \displaystyle\mathcal{L}_{KL}=\frac{1}{HW}\sum^{HW}_{i=1}KL(PV_{p_{i}},\hat{PV}_{p_{i}}), | 
 | 
 (28) | 
 

 
 
 where p i p_{i} represents a pixel coordinate.

 
 
 

### 4.4 Structural Consistency in 3D Space

 
 Figure 7: Left: Sample input images, key points, superpixels, and
piece-wise planar regions obtained from large superpixels (Fig. reprinted from [ 75 ] ).
Right: Point cloud matching process and approximate gradients for the top view of
a car, Fig. reprinted from [ 28 ] . 
 
 
 Structural consistency is not limited to the 2D image plane, and can easily be
extended to camera 3D space (a virtual 3D space after applying extrinsic
camera parameters to a scene) or 3D point clouds (a reconstruction of points in a 3D space). In this section,
we discuss two such methods [ 75 , 28 ] 
that use structural consistency in
3D space alongside other geometric constraints in an end-to-end framework.

 
 

#### 4.4.1 Planar Consistency:

 
 Planar consistency [ 75 ] is based on the assumption that most
homogeneous-color regions in an indoor scene are planar regions and have continuous depth.
Extraction of such piece-wise planar regions is a three-step process.
Given an input image I I , the key points are first extracted. The key points in the input image
are then used to extract superpixels. Finally, a segmentation algorithm is used to apply
a greedy approach to segment areas with low gradients to produce more planar regions. There
are many ways to extract key points and superpixels, [ 75 ] uses
Direct Sparse Odometry [ 43 ] to extract key points and Felzenszwalb superpixel
segmentation [ 45 ] for superpixels’ and planar region segmentation.

 
 
 The left side images in Fig. 7 show the steps of obtaining planar
regions in two indoor scenes.
For an image I I , after extracting superpixels, a threshold is applied to only keep
regions with greater than 1000 1000 pixels. It is assumed that most planar regions occupy
large pixel areas. With the extracted super pixels S ​ P ​ P m SPP_{m} and corresponding depth D ⁡ ( p n ) D(p_{n}) ,
we first back project all points p n p_{n} into 3D space ( p n 3 ​ D p^{3D}_{n} ), p n 3 ​ D = D ⁡ ( p n ) ​ K − 1 ​ p n , p n ⊆ S ​ P ​ P m p^{3D}_{n}=D(p_{n})K^{-1}p_{n},p_{n}\subseteq SPP_{m} .
Using the plane parameter A m A_{m} of S ​ P ​ P m SPP_{m} , the plane is defined in 3D space, A m T ​ p n 3 ​ D = 1 A^{T}_{m}p^{3D}_{n}=1 .
 A m A_{m} is calculated using two matrices, Y m = [ 1 , … , 1 ] T Y_{m}=[1,...,1]^{T} and P n = [ p 1 3 ​ D , … , p n 3 ​ D ] P_{n}=[p^{3D}_{1},...,p^{3D}_{n}] ,
 P n ​ A m = Y m ; A m = ( P n T ​ P n + ϵ ​ E ) − 1 ​ P n T ​ Y m P_{n}A_{m}=Y_{m};A_{m}=(P^{T}_{n}P_{n}+\epsilon E)^{-1}P^{T}_{n}Y_{m} , where E E is an identity matrix and ϵ \epsilon is a for
numerical stability. With planar parameters, a fitted planar depth for each pixel in all superpixels
can be retrieved to estimate the planar loss,

 
 
 

 
 | 
 D ′ ​ ( p n ) \displaystyle D^{\prime}(p_{n}) | 
 = ( A m T ​ K − 1 ​ p n ) − 1 \displaystyle=(A^{T}_{m}K^{-1}p_{n})^{-1} | 
 | 
 (29) | 
 
 
 | 
 ℒ p ​ l ​ a ​ n ​ a ​ r \displaystyle\mathcal{L}_{planar} | 
 = ∑ m = 1 M ∑ n = 1 N | D ⁡ ( p n ) − D ′ ​ ( p n ) | , \displaystyle=\sum^{M}_{m=1}\sum^{N}_{n=1}|D(p_{n})-D^{\prime}(p_{n})|, | 
 | 
 (30) | 
 

 
 
 

#### 4.4.2 Point Cloud Alignment:

 
 Mahjourian et al. [ 28 ] use another approach to
align the 3D point clouds of two consecutive frames ( Q t − 1 , Q t Q_{t-1},Q_{t} ) in video depth estimation.
They directly compare the estimated point cloud associated with respective frames ( Q ^ t − 1 \hat{Q}_{t-1} and Q ^ t \hat{Q}_{t} ),
using a well-known rigid registration method,
Iterative Closest Point (ICP) [ 35 , 39 , 59 ] , which
computes a transformation to minimize the point-to-point distance between two point clouds. They
alternate between computing correspondences between 3D points and best-fit transformations between
the two point clouds. For each iteration, they recompute the correspondence with the previous iteration’s
transformation applied.

 
 
 ICP is not differentiable, but its gradients can be approximated using the
product it computes as part of the algorithm, allowing back-propagation.
It takes two point clouds A A and B B as input and produces two outputs.
First, is the best-fit transformation T ′ T^{\prime} which minimizes the distance between
the transformed points in A A and B B ,
and second is the residual r i ​ j r^{ij} , r i ​ j = A i ​ j − T ′ − 1 . B c ⁡ ( i ​ j ) r^{ij}=A^{ij}-T^{\prime-1}.B^{c(ij)} ,
which reflects the residual distances between corresponding points after ICP’s minimization.
The alignment loss is given as,

 
 
 

 
 | 
 ℒ 3 ​ D ​ A ​ l ​ i ​ g ​ n ​ m ​ e ​ n ​ t = ‖ T ′ − I ‖ 1 + ‖ r ‖ 1 ; I = 1 , \displaystyle\mathcal{L}_{3DAlignment}=||T^{\prime}-I||_{1}+||r||_{1};I=1, | 
 | 
 (31) | 
 

 
 
 where I I is the identity matrix.

 
 
 For each frame t t of the video, if the alignment of the estimate is not perfect,
the ICP algorithm produces a transformation T t ′ T^{\prime}_{t} and r t r_{t} , which can be used to
adjust the estimates towards initial alignment [ 28 ] .
Right side of
Fig. 7 shows the steps of alignment loss.

 
 
 
 
 

## 5 Normal-Depth Orthogonal Constraint

 
 Figure 8: Left: estimation of normal from depth. Right: estimation of
depth from normal. Figure reprinted from [ 25 ] 
 
 
 Surface normals are important ’local’ features of the 3D point cloud of
a scene, which can provide promising 3D geometric cues to
estimate geometrically consistent depth maps. In the 3D world coordinate
system, the vector connecting two pixels in the same plane should be
orthogonal to their direction of normal. Enforcing normal-depth
orthogonal constraints tend to improve depth estimates in
3D space [ 92 , 30 , 23 ] .
Application of these constraints during the learning process helps model
better understand the representations related to cluttered backgrounds,
repetitive patterns and texture-less regions. These can efficiently be
utilized as part of an objective function in a deep learning framework.

 
 
 Depth to normal: 

 
 Given depth D i D_{i} ,
to estimate the normal of each central pixel p i p_{i} , we need
to consider the neighboring pixels p n ​ e ​ i ​ g ​ h ​ b ​ o ​ r ​ s p_{neighbors} .
Fig. 8 (left) shows 8 8 
neighbor convention to compute the normal of the central pixel.
Two neighbors, p i ​ x p_{ix} and p i ​ y p_{iy} are selected from p n ​ e ​ i ​ g ​ h ​ b ​ o ​ r ​ s p_{neighbors} 
for each central pixel p i p_{i} , with depth value as D i D_{i} 
and camera intrinsics K K , to estimate normal N i ~ \tilde{N_{i}} .
First, we project the depth in 3D space, and then take the cross
product between vectors P i ​ P i ​ x → \overrightarrow{P_{i}P_{ix}} and
 P i ​ P i ​ y → \overrightarrow{P_{i}P_{iy}} to estimate the normal,

 
 
 

 
 | 
 P i \displaystyle P_{i} | 
 = K − 1 ​ D i ​ p i \displaystyle=K^{-1}D_{i}p_{i} | 
 | 
 (32) | 
 
 
 | 
 N i ~ \displaystyle\tilde{N_{i}} | 
 = P i ​ P i ​ x → × P i ​ P i ​ y → \displaystyle=\overrightarrow{P_{i}P_{ix}}\times\overrightarrow{P_{i}P_{iy}} | 
 | 
 (33) | 
 

 
 
 With 8 8 such estimates of normal N i ~ \tilde{N_{i}} for
each central pixel, we estimate the final normal N i N_{i} as the mean value of all the estimates using
 N i = 1 8 ​ ∑ i = 1 8 ( N i ~ ) N_{i}=\frac{1}{8}\sum^{8}_{i=1}(\tilde{N_{i}}) .

 
 
 
 Normal to depth: 

 
 Many methods [ 25 , 23 , 73 ] 
use normal to depth estimation to refine the depth values D i D_{i} 
using the orthogonal relationship. For each pixel p i ​ ( x i , y i ) p_{i}(x_{i},y_{i}) ,
the depth of each neighbor p n ​ e ​ i ​ g ​ h ​ b ​ o ​ r p_{neighbor} should be refined.
The corresponding 3D points are P i P_{i} and P n ​ e ​ i ​ g ​ h ​ b ​ o ​ r P_{neighbor} , and
the central pixel P i P_{i} ’s
normal N i → ​ ( n x , n y , n z ) \overrightarrow{N_{i}}(n_{x},n_{y},n_{z}) . The depth of
 P i P_{i} is D i D_{i} and P n ​ e ​ i ​ g ​ h ​ b ​ o ​ r P_{neighbor} is D n ​ e ​ i ​ g ​ h ​ b ​ o ​ r D_{neighbor} . Using the
orthogonal relations
 N i → ⟂ P i ​ P n ​ e ​ i ​ g ​ h ​ b ​ o ​ r → \overrightarrow{N_{i}}\perp\overrightarrow{P_{i}P_{neighbor}} , we
can write,

 
 
 

 
 | 
 [ K − 1 ​ D i ​ p i − K − 1 ​ D n ​ e ​ i ​ g ​ h ​ b ​ o ​ r ​ P n ​ e ​ i ​ g ​ h ​ b ​ o ​ r ] ​ [ n x n y n z ] = 0 \displaystyle[K^{-1}D_{i}p_{i}-K^{-1}D_{neighbor}P_{neighbor}]\begin{bmatrix}n_{x}\\
n_{y}\\
n_{z}\\
\end{bmatrix}=0 | 
 | 
 (34) | 
 

 
 
 With depth
estimates coming from eight neighbors, we need a method for
weighting these values to incorporate discontinuity of normal in some edges
or irregular surfaces. Generally, the weight w i w_{i} is inferred from
the reference image I i I_{i} , making depth more
geometric consistent. The value of w i w_{i} depends on the gradient
between p i p_{i} and p n ​ e ​ i ​ g ​ h ​ b ​ o ​ r p_{neighbor} .
The larger gradients
represent less reliable refined depth. Given the eight neighbors, the final refined depth
 D ~ n ​ e ​ i ​ g ​ h ​ b ​ o ​ r \tilde{D}_{neighbor} is a weighted sum of eight different
directions as [ 25 ] ,

 
 
 

 
 | 
 D ~ n ​ e ​ i ​ g ​ h ​ b ​ o ​ r = ∑ i = 1 8 w i ∑ i = 1 8 w i ​ D n ​ e ​ i ​ g ​ h ​ b ​ o ​ r \displaystyle\tilde{D}_{neighbor}=\sum^{8}_{i=1}\frac{w_{i}}{\sum^{8}_{i=1}w_{i}}D_{neighbor} | 
 | 
 (35) | 
 

 
 
 where w i = e − α ​ | ▽ ​ I i | w_{i}=e^{-\alpha|\triangledown I_{i}|} .
This is the outcome of the regularization in 3D
space, improving the accuracy and continuity of the estimated
depth maps.

 
 
 Yang et al. [ 23 ] use a similar formulation to enforce geometric consistency in
unsupervised video depth estimation problems.
Wang et al. [ 92 ] use the orthogonal compatibility principle to bring
consistency in the normal directions of two pixels falling on the same plane.
Eigen and Fergus [ 30 ] use a single multiscale CNN
to estimate depth, surface normal, and semantic labeling. For surface
normal estimation at each pixel, they predict the x , y x,y and z z 
components for each pixel [ 62 ] 
and employ elementwise loss
comparison with dot product,

 
 
 

 
 | 
 ℒ n ​ o ​ r ​ m ​ a ​ l = − 1 N ∑ i n ^ i ⊙ n i , \mathcal{L}_{normal}=-\frac{1}{N}\sum_{i}\hat{n}_{i}\odot n_{i}, | 
 | 
 (36) | 
 

 
 
 where N N is the valid pixels, n n and n ^ \hat{n} are ground truth and
the predicted normal at each pixel.

 
 
 Qi et al. [ 57 ] also employ depth-to-normal and
normal-to-depth networks to regularize the depth estimate in
3D space following [ 46 ] , but they do not
use 8-neighbor-based calculation. Instead,
they use a distance-based selection of neighboring
pixels,

 
 
 

 
 | 
 N i = { ( x j , y j , z j ) | | u i − u j | β , | v i − v j | β , | z i − z j | γ z i } N_{i}=\{(x_{j},y_{j},z_{j})||u_{i}-u_{j}| \beta,|v_{i}-v_{j}| \beta,|z_{i}-z_{j}| \gamma z_{i}\} | 
 | 
 (37) | 
 

 
 
 where u i , v i u_{i},v_{i} are the 2D coordinates, ( x i , y i , z i ) (x_{i},y_{i},z_{i}) are the 3D
coordinates, and β \beta and γ \gamma are hyper-parameters controlling the
size of the neighborhood along x − y x-y and depth axes respectively. They
use L 2 L_{2} norm between the ground truth normal and the estimated normal
as the loss ℒ n ​ o ​ r ​ m ​ a ​ l \mathcal{L}_{normal} in end-to-end deep learning framework.

 
 
 Hu et al. [ 90 ] use ground truth and estimated depth
map gradients to measure the angle between their surface normals using
 n i d ^ = [ − ▽ x ​ d ^ i , − ▽ y ​ d ^ i , 1 ] T n^{\hat{d}}_{i}=[-\triangledown_{x}\hat{d}_{i},-\triangledown_{y}\hat{d}_{i},1]^{T} and
 n i d = [ − ▽ x ​ d i , − ▽ y ​ d i , 1 ] T n^{d}_{i}=[-\triangledown_{x}d_{i},-\triangledown_{y}d_{i},1]^{T} .
The loss, ℒ n ​ o ​ r ​ m ​ a ​ l \mathcal{L}_{normal} 
is estimated using,

 
 
 

 
 | 
 ℒ n ​ o ​ r ​ m ​ a ​ l \displaystyle\mathcal{L}_{normal} | 
 = 1 N ​ ∑ i = 1 N ( 1 − ⟨ n i d ^ , n i d ⟩ ⟨ n i d ^ , n i d ^ ⟩ ​ ⟨ n i d , n i d ⟩ ) \displaystyle=\frac{1}{N}\sum^{N}_{i=1}\left(1-\frac{\langle n^{\hat{d}}_{i},n^{d}_{i}\rangle}{\sqrt{\langle n^{\hat{d}}_{i},n^{\hat{d}}_{i}\rangle}\sqrt{\langle n^{d}_{i},n^{d}_{i}\rangle}}\right) | 
 | 
 (38) | 
 

 
 
 where
 ⟨ . , . ⟩ \langle.,.\rangle denotes the inner product of vectors. This loss is
sensitive to small depth structures [ 90 ] .
Yang et al. [ 72 ] use the same method of estimating normal
for monocular depth estimation problems in endoscopy applications.

 
 
 Advantages of estimating normals in a depth estimation
frameworks are that it provides an explicit understanding of normal during the
learning process, higher-order interaction between the
estimated depth and ground truth, and flexibility to
integrate additional operations over normal
 [ 23 ] . On the other hand, it is sensitive
to noise in the ground truth depth maps and the estimated depth
maps, and only considers local information which may
not align with the global structure.

 
 
 To enforce robust higher-order geometric supervision in 3D space,
Yin et al. [ 73 ] propose virtual normal (VN) estimation.
VN can establish 3D geometric connections between regions in a much
larger range. To estimate VN, N N group points from the depth
map, with three points in each group, are sampled. The selected point
has to be non-colinear. The three points establish a plane and the
corresponding normal is estimated. Similarly, ground truth normal ( n i ) (n_{i}) 
are estimated and compared with the normals corresponding to the
estimated depth maps ( n ^ i ) (\hat{n}_{i}) ,

 
 
 

 
 | 
 ℒ N ​ o ​ r ​ m ​ a ​ l r ​ o ​ b ​ u ​ s ​ t = 1 N ​ ( ∑ i = 1 N ‖ n ^ − n i ‖ 1 ) , \mathcal{L}_{Normal_{robust}}=\frac{1}{N}\left(\sum^{N}_{i=1}||\hat{n}-n_{i}||_{1}\right), | 
 | 
 (39) | 
 

 
 
 Naderi et al. [ 33 ] use a similar
formulation in a monocular depth estimation problem to enforce
higher-order robust geometric constraints for depth estimation.

 
 
 Figure 9: Normal-depth joint learning approached. Depth and normal are estimated using the same feature cost volume [ 51 ] . The above diagram is a slight modification from
Kusupati et al. [ 51 ] to only show the joint estimation setup. 
 
 
 
 Normal-depth joint learning approach: 

 
 Kusupati et al. [ 51 ] develop a normal-assisted depth estimation
algorithm that couples the learning of multi-view normal estimation and multi-view depth
estimation process. It uses a feature cost volume to estimate the depth map and
the normal. A cost volume provides better structural representation to facilitate
better learning on the image features.
Specifically, Kusupati et el. [ 51 ] estimate two depth maps
 Z 1 Z_{1} and Z 2 Z_{2} using 3D-CNNs, as shown in Fig. 9 , and uses
a 7-layered CNN (NNet, Fig. 9 ) to estimate the normal. The world coordinate volume is concatenated
with the initial feature volume to provide a feature slice to NNet as input.
NNet predicts normals for each feature slice, which are later averaged to get the final
normal map.

 
 
 
 

## 6 Attention Meets Geometry

 
 Recent advancements in attention-based deep learning frameworks
have revolutionized the computer vision field. Network
architecture like a Transformer with a self-attention mechanism [ 13 ] 
has become the best-performing network for various vision tasks.
Its ability to model long-range global-contextual
representation by dynamically shifting its attention within the image makes it unique.
The inputs to the attention module are usually named
query (Q), key (K), and value (V). Q retrieves information
from V based on the attention weights, A ​ t ​ t ​ e ​ n ​ t ​ i ​ o ​ n ​ ( Q , K , V ) = 𝒜 ⁡ ( Q , K ) ​ V Attention(Q,K,V)=\mathcal{A}(Q,K)V where 𝒜 \mathcal{A} 
is an attention estimating function,
where 𝒜 ( . ) \mathcal{A}(.) is a function that produces similarity scores
as attention weights between feature embeddings for aggregation.

 
 
 The performance of stereo or MVS depth estimation methods depends
on finding dense correspondences
between reference and source images. Recently, Sun et al. [ 63 ] 
showed that features extracted using a transformer model with self- and cross-attention
can produce significantly improved correspondences as compared to features extracted using
convolutional neural networks. These attention mechanisms are designed to
pay attention to the contextual information and not to geometry-based
information. Recently, a few methods have modified these attention
mechanisms to consider geometric information while calculating the
attention weight [ 58 , 33 , 78 , 49 ] .
In this section, we discuss such methods and their approach to include
geometric information in attention.

 
 
 Ruhkamp et al. [ 58 ] use geometry to guide spatial-temporal
attention to guide self-supervised monocular depth estimation method from videos.
They propose a spatial-attention layer with 3D spatial awareness by exploiting
a coarse predicted initial depth estimate. With known intrinsic camera parameter K
and pair of coordinates ( C i , C j C_{i},C_{j} ) along with their depth estimates ( d i , d j ) (d_{i},d_{j}) ,
they back-project the depth values into 3D space using Eq. 40 .
The 3D space-aware spatial attention is then calculated as,

 
 
 

 
 | 
 p i = K − 1 ( d i . C i ) ; p j \displaystyle p_{i}=K^{-1}(d_{i}.C_{i});p_{j} | 
 = K − 1 ( d j . C j ) , \displaystyle=K^{-1}(d_{j}.C_{j}), | 
 | 
 (40) | 
 
 
 | 
 𝒜 i , j s ​ p ​ a ​ t ​ i ​ a ​ l \displaystyle\mathcal{A}^{spatial}_{i,j} | 
 = e ​ x ​ p ​ ( − ‖ P i − P j ‖ 2 σ ) , \displaystyle=exp\left(-\frac{||P_{i}-P_{j}||_{2}}{\sigma}\right), | 
 | 
 (41) | 
 
 
 | 
 𝒜 i , j t ​ e ​ m ​ p ​ o ​ r ​ a ​ l \displaystyle\mathcal{A}^{temporal}_{i,j} | 
 = S ​ o ​ f ​ t ​ m ​ a ​ x j ​ ( F i q T ​ F j k ) , \displaystyle=Softmax_{j}(F^{q^{T}}_{i}F^{k}_{j}), | 
 | 
 (42) | 
 

 
 
 where P i , P j P_{i},P_{j} are treated as K and Q, respectively. They use Eq. 42 
to estimate the temporal attention for aggregation. The unique formulation of
the spatial-temporal attention model can explicitly correlate geometrically
meaningful and spatially coherent features for dense correspondence [ 58 ] .

 
 
 Naderi et al. [ 33 ] propose adaptive geometric attention (AGA)
for monocular depth estimation with an encoder-decoder architecture. They apply the AGA module in the
decoding step utilizing both low-level ( F L ) (F_{L}) and high-level ( F h ) (F_{h}) features. Fig.
 10 shows the process of calculating AGA. The first row of operations
in Fig. 10 shows the steps to calculate channel-attention
 ( 𝒞 ​ 𝒜 ) (\mathcal{CA}) , which produces a 1 × 1 × C 1\times 1\times C shape attention map and is multiplied
with F l F_{l} . The other two rows show two spatial attention ( 𝒮 ​ 𝒜 ) (\mathcal{SA}) 
calculations using 𝒮 𝒜 i = | C o s i n e s ​ i ​ m ​ i ​ l ​ a ​ r ​ i ​ t ​ y ( E l , i , E h , i ) | , i = 1 , 2 \mathcal{SA}_{i}=|Cosine_{similarity}(E_{l,i},E_{h,i})|,i=1,2 .
The final aggregated
feature output ( F o ​ u ​ t ) (F_{o}ut) is estimated using F o ​ u ​ t = [ f 1 ​ ( 𝒮 ​ 𝒜 1 ) + f 2 ​ ( 𝒮 ​ 𝒜 2 ) × 𝒞 ​ 𝒜 ] × F l + F h F_{out}=[f_{1}(\mathcal{SA}_{1})+f_{2}(\mathcal{SA}_{2})\times\mathcal{CA}]\times F_{l}+F_{h} ,
where 𝒮 ​ 𝒜 1 \mathcal{SA}_{1} is added and 𝒮 ​ 𝒜 2 \mathcal{SA}_{2} is multiplied with F L F_{L} . f 1 ( . ) f_{1}(.) and f 2 ( . ) f_{2}(.) 
are introduced to enhance the sensitivity to any non-zero correlation between F l F_{l} and F h F_{h} ,
 f ⁡ ( 𝒮 ​ 𝒜 ) = 𝒮 ​ 𝒜 f(\mathcal{SA})=\mathcal{SA} is with no-enhanced sensitivity, whereas f ⁡ ( 𝒮 ​ 𝒜 ) = 𝒮 ​ 𝒜 . e ​ x ​ p ​ ( 𝒮 ​ 𝒜 ) f(\mathcal{SA})=\mathcal{SA}.\,exp(\mathcal{SA}) 
shows the formulation of the enhanced sensitivity attention map.

 
 
 Figure 10: Adaptive Geometric Attention. Figure reprinted from [ 33 ] 
 
 
 Zhu et al. [ 78 ] use two types of transformer modules to extract
geometry-aware features in an MVS pipeline, a global-context transformer module
and a 3D geometry transformer module.
The global-context transformer module extracts 3D-consistent reference features ( T r ) (T_{r}) 
which are then used as input to the 3D-geometry transformer module to facilitate cross-view attention.
 ( T r ) (T_{r}) is used to generate K and V to
enhance interaction between reference and source view features for obtaining
dense correspondence.

 
 
 Guo et al. [ 49 ] use a geometry-aware attention mechanism
for image captioning. Unlike other methods discussed above,
they do not modify the attention mechanism
itself but add a bias term which makes the feature extraction biased towards
specific content. Their attention module is similar to [ 13 ] 
apart from the added bias term in score E E calculation,
 E = Q ​ K T + ϕ ⁡ ( Q ′ , K ′ , G ) E=QK^{T}+\phi(Q^{\prime},K^{\prime},G) , where
 G i ​ j G_{ij} is the relative geometry feature between two objects i i and j j .
There are two terms in score E E , the content-based weights 
and geometric bias . They propose three different ways
of applying geometric bias.
 content-independent bias ( ϕ i ​ j i ​ n ​ d ​ e ​ p ​ e ​ n ​ d ​ e ​ n ​ t = R ​ e ​ L ​ U ​ ( w g T ​ G i ​ j ) \phi^{independent}_{ij}=ReLU(w^{T}_{g}G_{ij}) ) assumes
static geometric bias, i.e. same geometric bias is applied to all the K-Q pairs.
The query-dependent bias provides geometric information
based on the type of query ( ϕ i ​ j q ​ u ​ e ​ r ​ y = Q g T ′ ​ G i ​ j \phi^{query}_{ij}=Q^{{}^{\prime}T}_{g}G_{ij} ) and key-dependent bias provides
geometric information based on the clues present in keys, ϕ i ​ j k ​ e ​ y = K g T ′ ​ G i ​ j \phi^{key}_{ij}=K^{{}^{\prime}T}_{g}G_{ij} 

 
 
 

## 7 Learning Geometric Representations

 
 Besides using direct methods of enforcing geometry-aware constraints,
providing geometric guidance, and exploiting orthogonal relations between
depth and normals, there are indirect ways of learning
structurally consistent feature representations. For example, features
with a high level of semantic information are more likely to retain
the structural consistency of objects compared to features with
a low level of semantic information. The generation of pseudo-labels based on geometric consistency
can be utilized for self-supervised training. Robust feature representation can be
learned using suitable data-augmentation techniques, and
attaching semantic segmentation information of objects, like a co-segmentation map can
improve the structural consistency of objects’ features. Contrastive
learning with positive and negative pairs can guide a model to learn
better representations that are geometrically consistent features.
All these constraints helps the model learn a geometry-based representations
which are able to help with problem like cluttered background, repetitive patterns and texture-less regions.
These constraints are mostly utilized as part of an objective function in a learning-based framework.
We discuss all these methods in this section.

 
 

### 7.1 High-Level Feature Alignment Constraints

 
 In deep learning-based frameworks for depth estimation, the quality of the extracted features
directly impacts the quality of depth estimates. The poor quality of extracted features
can greatly impact the local as well as global structural patterns. One way to handle this
problem is by guiding the extracted features with better representations from an
auxiliary pre-trained network. While the integrated feature extraction network in
the depth estimation pipeline can learn useful features, it still lags in
learning higher-level representations compared to a network explicitly designed
to learn deep high-level representations, like VGG [ 93 ] ,
Inception [ 94 ] , ResNet [ 95 ] , etc.
To enforce the feature alignment constraint, Johnson et al. [ 96 ] 
propose two loss functions, feature reconstruction loss and style reconstruction loss.

 
 
 Feature reconstruction loss, ℒ f ​ e ​ a ​ t ​ u ​ r ​ e = ‖ F t ​ a ​ r ​ g ​ e ​ t − F s ​ o ​ u ​ r ​ c ​ e ‖ L i \mathcal{L}_{feature}=||F_{target}-F_{source}||_{L_{i}} , encourages the model to
generate source features similar
to the target features at various stages of the network [ 96 ] . The strong
feature representations from target features help mitigate confusions arising because of cluttered background and
repetitive patterns. This also helps the model in extracting good representation in low-texture or texture-less regions [ 25 , 41 , 77 ] .
Minimizing feature reconstruction loss for early layers improves local visual as well as
structural patterns, while minimizing it for higher layers improves overall structural
patterns [ 96 ] . Feature reconstruction loss fails to preserve
color and texture, which is handled by style reconstruction loss [ 96 ] .

 
 
 The generalized formulation of feature reconstruction loss,
where F t ​ a ​ r ​ g ​ e ​ t F_{target} is the target feature, F s ​ o ​ u ​ r ​ c ​ e F_{source} is the source feature and L i L_{i} denotes L 1 L_{1} or L 2 L_{2} norm.
A similar formulation is adopted by Huang et al. [ 25 ] , dubbed as
feature-wise loss, in an unsupervised MVS framework. Using a pre-trained VGG-16 network, high-level
semantically rich features are extracted at layers 8, 15, and 22 for both the reference ( F r ​ e ​ f ) (F_{ref}) and the
source ( F s ​ r ​ c ) (F_{src}) images. The features from the source images are warped to the reference view ( F ^ s ​ r ​ c ) (\hat{F}_{src}) using
camera parameters and used in the feature reconstruction loss,
 ℒ f ​ e ​ a ​ t ​ u ​ r ​ e = 1 N ​ ∑ ( F r ​ e ​ f − F ^ s ​ r ​ c ) ∗ M r ​ e ​ f \mathcal{L}_{feature}=\frac{1}{N}\sum\left(F_{ref}-\hat{F}_{src}\right)*M_{ref} . M M is
the reference view mask to handle occlusion and N N is the number of source view images. Dong et al.
 [ 41 ] use a similar formulation and extract feature from 8 t ​ h 8^{th} and 15 t ​ h 15^{th} 
layers.

 
 
 Applying feature alignment loss helps the model extract high-level semantically rich features for
depth estimation can also be used to generate geometrically consistent and style-conforming
new RGB image, which is then used as input to the depth estimation network. Zhao et al.
 [ 77 ] and Xu et al. [ 70 ] use such an approach to
fill generate synthetic input images that are geometrically consistent across
views and close to the original data distribution.

 
 
 Zhao et al. [ 77 ] generate
synthetic RGB images, and they apply feature and style reconstruction losses
at the final image resolution. They learn bidirectional translators from source to target G s ​ 2 ​ t G_{s2t} ,
and from target to source G t ​ 2 ​ s G_{t2s} , to bridge the gap between the source domain (synthetic) X s X_{s} and
the target domain (real) X t X_{t} . Specifically, image x s x_{s} is sequentially fed to
 G s ​ 2 ​ t G_{s2t} and G t ​ 2 ​ s G_{t2s} generating a reconstruction of x s x_{s} and vice-versa for x t x_{t} . These
are then compared to the original input,

 
 
 

 
 | 
 ℒ F ​ e ​ a ​ t ​ u ​ r ​ e c ​ y ​ c ​ l ​ e = ‖ G t ​ 2 ​ s ​ ( G s ​ 2 ​ t ​ ( x s ) ) − x s ‖ 1 + ‖ G s ​ 2 ​ t ​ ( G t ​ 2 ​ s ​ ( x t ) ) − x t ‖ 1 , \mathcal{L}_{Feature_{cycle}}=||G_{t2s}(G_{s2t}(x_{s}))-x_{s}||_{1}+||G_{s2t}(G_{t2s}(x_{t}))-x_{t}||_{1}, | 
 | 
 (43) | 
 

 
 
 While Zhao et al. [ 77 ] apply ℒ f ​ e ​ a ​ t ​ u ​ r ​ e c ​ y ​ c ​ l ​ e \mathcal{L}_{feature_{cycle}} at the RGB image level,
Xu et al. [ 70 ] use feature and style reconstruction loss in a semi-supervised
MVS framework. They use a geometry-preserving module
to generate geometry and style-conforming RGB images using a labeled real image.
They employ a geometry-preserving module to generate unlabeled RGB images which are later used
as input to estimate depth maps.
The geometry-preserving module includes a spatial propagation network (SPN) with
two branches, propagation network, and guidance network.
The labeled image ( I l ) (I_{l}) is used as input to the guidance network to generate
an alternate view RGB image ( I g ) (I_{g}) . This RGB image is used as input to the depth estimation
pipeline to generate a corresponding depth map D g D_{g} , which is then
warped to the original view ( D ^ g \hat{D}_{g} ) to compare with the ground truth depth map D l D_{l} .
The loss function is ℒ f ​ e ​ a ​ t ​ u ​ r ​ e s ​ t ​ y ​ l ​ e = ‖ D l − D ^ g ‖ 2 2 \mathcal{L}_{feature_{style}}=||D_{l}-\hat{D}_{g}||^{2}_{2} [ 70 ] .
With this setup, they use labeled images to generate geometrically conforming alternate view unlabeled images, and use
them in the depth estimation pipeline without having to create new labeled data.

 
 
 Figure 11: Pseudo-label generation method of [ 24 ] . SPSR is
Screened Poisson Surface Reconstruction [ 97 ] . Figure inspired from [ 24 ] . 
 
 
 

### 7.2 Pseudo-Label Generation with Cross-View Consistency

 
 In self-supervised MVS frameworks, pseudo-labels are an effective method of applying geometric constraints.
Generating pseudo-labels requires application of cross-view
consistency constraints, which encourages the MVS framework to be geometrically consistent
during training and evaluation [ 24 , 54 ] . Since the model
learns with self-supervision, it also helps with the challenging task of collecting
multi-view ground truth data. In this section, we discuss three methods of generating
pseudo-labels for self-supervision, labels from high-resolution training images [ 24 ] ,
sparse pseudo-label generation [ 54 ] and semi-dense pseudo-labels [ 54 ] .

 
 
 Yang et al. [ 24 ] apply pseudo-label learning in four steps. First,
they estimate a depth map based on photometric consistency, Sec. 3.1 ,
using a coarse network (low-resolution network). With the initial pseudo-label in hand, they apply two step
iterative self-training to refine these pseudo-labels, see Fig. 11 .
They utilize fine-grained high-resolution networks
to refine the initial coarse pseudo-labels utilizing more discriminative features from high-resolution images.
The fine-network estimates high-resolution labels which are then filtered with a cross-view depth consistency check,
Sec. 3 , utilizing depth re-projection error.
Finally, the high-resolution filtered pseudo-labels from N N different views are fused using a
multi-view fusion method. It generates a more complete point cloud of the scene. The point cloud is then rendered
to generate cross-view, geometrically consistent new pseudo-labels to guide the coarse network depth
estimation pipeline.

 
 
 Liu et al. [ 54 ] use two geometric prior-guided pseudo-label generation methods,
sparse and semi-dense pseudo-label. For sparse label generation, they use a pre-trained
Structure from Motion framework (SfM) [ 98 ] to generate a sparse point cloud.
This sparse point cloud is then projected to generate sparse depth pseudo-labels. Since the sparse point
cloud can provide very limited supervision, they use a traditional MVS framework such as
COLMAP [ 99 ] with
geometric and photometric consistency to estimate preliminary depth maps.
The preliminary depth map then undergoes cross-view
geometric consistency check to eliminate outliers. This filtered depth map
is then used as a final pseudo-label for learning.

 
 
 

### 7.3 Data-Augmentation for Geometric Robustness

 
 Deep-learning frameworks can always do better with more data [ 100 ] , but
collecting data for stereo or MVS is a difficult task. Applying data augmentation to these
frameworks makes sense but is not as easy to implement.
The natural color fluctuation, occlusion, and geometric distortions
in augmented images disturbs the color constancy of images, affecting the
effective feature matching and hence the performance of the
whole depth estimation pipeline [ 21 ] . Because of these limitations,
it has seldom been applied
in either supervised [ 16 , 14 , 15 ] or
unsupervised [ 50 , 40 , 25 ] MVS methods.

 
 
 Keeping these limitations in mind, Garg et al.
 [ 22 ] use three data-augmentation techniques in an unsupervised stereo depth
estimation problem. They use color change – scalar multiplication of color channels by a factor c ∈ [ 0.9 , 1.1 ] c\in[0.9,1.1] ,
 scale and crop – the input image is scaled by 2 × 2\times factor and then randomly cropped
to match the original input size, and left-right flip – wherein the left and right images are
flipped horizontally and swapped to get a new training pair. These three simple augmentations lead to
an 8 8 -fold increase in data and improved localization of object edges in the depth estimates.

 
 
 Xu et al. [ 21 ] propose using data augmentation as a regularization technique.
Instead of optimizing the regular loss function with ground truth depth maps,
they propose data-augmentation consistency loss 
by contrasting data augmentation depth estimates with non-augmented depth estimates.
Specifically, given the non-augmented input images I n ​ o ​ n − a ​ u ​ g I_{non-aug} and
augmented input images I a ​ u ​ g I_{aug} of the same view, the difference between the estimated
augmented ( D ^ a ​ u ​ g \hat{D}_{aug} ) and non-augmented ( D ^ n ​ o ​ n − a ​ u ​ g \hat{D}_{non-aug} ) depth maps
are minimized [ 21 ] ,

 
 
 

 
 | 
 ℒ a ​ u ​ g ​ m ​ e ​ n ​ t ​ a ​ t ​ i ​ o ​ n = 1 M ​ ∑ ‖ D ^ a ​ u ​ g − D ^ n ​ o ​ n − a ​ u ​ g ‖ 2 ⊙ M n ​ o ​ n − a ​ u ​ g \mathcal{L}_{augmentation}=\frac{1}{M}\sum||\hat{D}_{aug}-\hat{D}_{non-aug}||_{2}\odot M_{non-aug} | 
 | 
 (44) | 
 

 
 
 where M n ​ o ​ n − a ​ u ​ g M_{non-aug} denotes an unoccluded mask under data-augmentation transformation. This
formulation also helped with cluttered backgrounds and texture-less regions. Xu et al.
 [ 21 ] cross-view masking augmentation to simulate occlusion hallucination in
multi-view situations by randomly generating a binary crop mask to block out some regions on reference
view. The mask is then projected to other views to mask out corresponding areas in source views. They
also used gamma correction to adjust the illuminance of images, random color jitter ,
 random blur and additive random noise in the input images. All these data-augmentation
methods do not affect camera parameters.

 
 
 

### 7.4 Semantic Information for Structural Consistency

 
 Humans can perform stereophotogrammetry well in ambiguous areas by exploiting more cues such as
global perception of foreground and background, relative scaling, and semantic consistency of
individual objects.
Deep learning-based frameworks, operating on the color constancy hypothesis [ 21 ] ,
generally provide a superior performance as compared to
traditional MVS algorithms, but both methods fail on texture-less regions, different
lighting conditions, reflections, or color constancy ambiguity .
Direct application of geometric and photometric constraints in such
regions is not helpful, but high-level semantic segmentation clues
can help. Semantic segmentation clues for a
given scene can provide abstract matching clues and
also act as structural priors for depth estimation [ 21 ] .
This also improves feature extraction in areas with cluttered backgrounds and low-texture.
In this section, we explore such depth estimation methods that include
semantic clues in their pipeline.

 
 
 Inspired by Cheng et al. [ 27 ] , which incorporate semantic segmentation
information to learn optical flow from video, Yang et al. [ 32 ] incorporate
semantic feature embedding and regularize semantic cues as the loss term to
improve disparity learning in stereo problem. Semantic feature embedding is a concatenation of
three types of features, left-image features, left-right correlation features, and
left-image semantic features. In addition to image and correlation features, semantic features
provide more consistent representations of featureless regions, which helps
solve the disparity problem. They also regularize the semantic cues loss term by warping the
right image segmentation map to the left view and comparing it with the left
image segmentation ground truth. Minimizing the semantic cues loss term
improves its consistency in the end-to-end learning process. Dovesi et al. [ 31 ] 
also employ semantic segmentation networks in coarse-to-fine design and utilize
additional information in stereo matching.

 
 
 Another way of utilizing semantic information for geometric and structural consistency
is through co-segmentation ,
which aims to predict foreground pixels of a common object to give an
image collection [ 101 ] . Inspired by Casser et al. [ 102 ] ,
which applied co-segmentation to learn semantic information in unsupervised monocular depth
ego-motion learning problem, Xu et al. [ 21 ] apply co-segmentation on multi-view
pairs to exploit the common semantics. They adopt non-negative matrix factorization (NMF)
 [ 103 ] to identify common semantic clusters among multi-view images
during the learning process. NMF is applied to the activations of a pre-trained layer [ 104 ] 
to find semantic correspondences across images. We refer to Ding et al. [ 103 ] for more
details on NMF. The consistency of the co-segmentation map can be expanded across multiple views
by warping it to other views. The semantic consistency loss is calculated as per pixel
cross-entropy loss between the warped segmentation map S ^ i \hat{S}_{i} and the
ground truth labels converted from the reference segmentation map S S ,s

 
 
 

 
 | 
 ℒ s ​ e ​ m ​ a ​ n ​ t ​ i ​ c = − ∑ i = 2 N [ 1 ‖ M i ‖ 1 ∑ j = 1 H ​ W f ( S j ) l o g ( s ^ i , j ) M i , j ] \mathcal{L}_{semantic}=-\sum^{N}_{i=2}\left[\frac{1}{||M_{i}||_{1}}\sum^{HW}_{j=1}f(S_{j})log(\hat{s}_{i,j})M_{i,j}\right] | 
 | 
 (45) | 
 

 
 
 where f ⁡ ( S j ) = o ​ n ​ e ​ h ​ o ​ t ​ ( a ​ r ​ g ​ m ​ a ​ x ​ ( S j ) ) f(S_{j})=onehot(arg\,max(S_{j})) and M i M_{i} is the binary mask indicating valid pixels from the i t ​ h i^{th} view
to the reference view.

 
 
 

### 7.5 Geometric Representation Learning with Contrastive Loss

 
 Contrastive learning [ 105 ] learns object representations by
enforcing an attractive force to positive pairs and a repulsive force to negative pair
 [ 61 ] . This form of representation learning has not been explored much
in depth estimation. Fan et al. [ 44 ] 
use contrastive learning to pay more attention to depth distribution and
improve the overall depth estimation process by adopting
a non-overlapping window-based contrastive learning. Lee et al. [ 52 ] 
use contrastive learning to disentangle the camera and object motion. While these
methods use contrastive learning for estimating depth maps, none of them use
contrastive learning to promote the geometric representation of objects.

 
 
 Figure 12: Contrastive learning approach to promote
geometric representation. Figure reprinted from [ 61 ] . 
 
 
 Shim and Kim [ 61 ] focus on learning geometric representations
for depth estimation using contrastive learning. They utilize
a Canny edge binary mask [ 106 ] 
to generate gradient fields of the image as the positive
and negative pairs, see Fig. 12 .
To estimate the gradient field G G ( + + for positive and − - for negative examples)
of input image pairs ℐ q \mathcal{I}_{q} (query image) and ℐ − \mathcal{I}_{-} (other image),
they modify the Canny detector to extract the magnitude of the dominant gradient
as well as its location to adjust the gradient field according to its
edge dominance. The adopted process can be mathematically formulated as,

 
 
 

 
 | 
 ℐ ∈ ℝ h × w , ▽ ​ ℐ x , ▽ ​ ℐ y ∈ ℝ h × w , ‖ E ‖ = ℐ x 2 + ℐ y 2 , G = B C ​ a ​ n ​ n ​ y ⊙ ‖ E ‖ \begin{split}\mathcal{I} \in\mathbb{R}^{h\times w},\triangledown\mathcal{I}_{x},\triangledown\mathcal{I}_{y}\in\mathbb{R}^{h\times w},\\
||E|| =\sqrt{\mathcal{I}^{2}_{x}+\mathcal{I}^{2}_{y}},\\
G =B_{Canny}\odot||E||\end{split} | 
 | 
 (46) | 
 

 
 
 where ‖ E ‖ ||E|| and B C ​ a ​ n ​ n ​ y B_{Canny} denote the magnitude of the
gradient from the Sobel operator and the binary mask of the Canny detector. ⊙ \odot is
element-wise multiplication.

 
 
 The network is pre-trained with contrastive loss to learn
the geometric representation 𝒵 \mathcal{Z} of the images. This
learned representation is further compressed by a 2-layer
fully connected network with ReLU non-linear activation to a feature
space ℋ \mathcal{H} . The projected latent vector h h of the
positive and the negative pairs are used to estimate the contrastive loss.

 
 
 
 

## 8 Conclusion

 
 Progress in deep learning technologies has immensely
benefited depth estimation by enabling the
extraction of high-level representations from input images.
However, it has also limited the use of modeling explicit photometric and
geometric constraints. Most supervised stereo and MVS
methods focus on better feature extraction and enhanced feature
matching through attention mechanisms but
apply a plane-sweep algorithm as the only geometric constraint. They
largely depend on the quality of ground truth to learn about
geometric and structural consistency in the learning process.
In this review, we have comprehensively discussed geometric
concepts in depth estimation and closely related domains that
can be coupled with deep learning frameworks to enforce
geometric and structural consistency in the learning process.
Explicitly modeling geometric constraints, along with the supervision signal,
can enforce structural reasoning, occlusion reasoning, better representation
learning in regions with low-texture, cluttered background or repetitive patterns.
It can significantly improve cross-view consistency
in any learning-based depth estimation framework. We believe this review will provide
a good reference for readers and researchers to explore the integration of geometric
constraints in deep learning frameworks.

 
 
 
 

## References

 
 
 [1] 
 
Wikipedia contributors.

 
 Photogrammetry — Wikipedia, the free encyclopedia, 2023.

 
 [Online; accessed 12-Oct-2023].

 

 
 [2] 
 
Yasutaka Furukawa and Jean Ponce.

 
 Accurate, dense, and robust multiview stereopsis.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence , 32(8):1362–1376, 2010.

 

 
 [3] 
 
Silvano Galliani, Katrin Lasinger, and Konrad Schindler.

 
 Massively parallel multiview stereopsis by surface normal diffusion.

 
 In 2015 IEEE International Conference on Computer Vision (ICCV) , pages 873–881, -, 2015. IEEE.

 

 
 [4] 
 
Engin Tola, Christoph Strecha, and Pascal Fua.

 
 Efficient large-scale multi-view stereo for ultra high-resolution image sets.

 
 Mach. Vision Appl. , 23(5):903–920, September 2012.

 

 
 [5] 
 
Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys.

 
 Pixelwise view selection for unstructured multi-view stereo.

 
 In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision – ECCV 2016 , pages 501–518, Cham, 2016. Springer International Publishing.

 

 
 [6] 
 
Y. Boykov, O. Veksler, and R. Zabih.

 
 Fast approximate energy minimization via graph cuts.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence , 23(11):1222–1239, 2001.

 

 
 [7] 
 
Richard Hartley and Andrew Zisserman.

 
 Multiple view geometry in computer vision .

 
 Cambridge University Press, -, 2003.

 

 
 [8] 
 
Quang-Tuan Luong and OD Faugeras.

 
 The geometry of multiple images.

 
 MIT Press, Boston , 2(3):4–5, 2001.

 

 
 [9] 
 
Richard Szeliski.

 
 Computer vision: algorithms and applications .

 
 Springer Nature, -, 2022.

 

 
 [10] 
 
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton.

 
 Deep learning.

 
 nature , 521(7553):436–444, 2015.

 

 
 [11] 
 
Yann LeCun, Yoshua Bengio, et al.

 
 Convolutional networks for images, speech, and time series.

 
 The handbook of Brain Theory and Neural nNetworks , 3361(10):1995, 1995.

 

 
 [12] 
 
Pankaj Malhotra, Lovekesh Vig, Gautam Shroff, Puneet Agarwal, et al.

 
 Long short term memory networks for anomaly detection in time series.

 
 In Esann , volume 2015, page 89, -, 2015. Esann.

 

 
 [13] 
 
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.

 
 An image is worth 16x16 words: Transformers for image recognition at scale.

 
 In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 , pages x–x, -, 2021. OpenReview.net.

 

 
 [14] 
 
Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan.

 
 Mvsnet: Depth inference for unstructured multi-view stereo.

 
 In Proceedings of the European conference on computer vision (ECCV) , pages 767–783, -, 2018. ECCV.

 

 
 [15] 
 
Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong Tan, and Ping Tan.

 
 Cascade cost volume for high-resolution multi-view stereo and stereo matching.

 
 In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 2495–2504, -, 2020. IEEE.

 

 
 [16] 
 
Yikang Ding, Wentao Yuan, Qingtian Zhu, Haotian Zhang, Xiangyue Liu, Yuanjiang Wang, and Xiao Liu.

 
 Transmvsnet: Global context-aware multi-view stereo network with transformers.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 8585–8594, -, 2022. IEEE.

 

 
 [17] 
 
Alex Kendall, Hayk Martirosyan, Saumitro Dasgupta, Peter Henry, Ryan Kennedy, Abraham Bachrach, and Adam Bry.

 
 End-to-end learning of geometry and context for deep stereo regression.

 
 In Proceedings of the IEEE international conference on computer vision , pages 66–75, -, 2017. IEEE.

 

 
 [18] 
 
Yue-Jiang Dong, Yuan-Chen Guo, Ying-Tian Liu, Fang-Lue Zhang, and Song-Hai Zhang.

 
 Ppea-depth: Progressive parameter-efficient adaptation for self-supervised monocular depth estimation.

 
 In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 1609–1617, -, 2024. AAAI.

 

 
 [19] 
 
Yue-Jiang Dong, Fang-Lue Zhang, and Song-Hai Zhang.

 
 Mal: Motion-aware loss with temporal and distillation hints for self-supervised depth estimation.

 
 In 2024 IEEE International Conference on Robotics and Automation (ICRA) , pages 7318–7324, -, 2024. IEEE, IEEE.

 

 
 [20] 
 
Robert T Collins.

 
 A space-sweep approach to true multi-image matching.

 
 In Proceedings CVPR IEEE Computer Society Conference on Computer vision and Pattern Recognition , pages 358–363, -, 1996. Ieee, IEEE.

 

 
 [21] 
 
Hongbin Xu, Zhipeng Zhou, Yu Qiao, Wenxiong Kang, and Qiuxia Wu.

 
 Self-supervised multi-view stereo via effective co-segmentation and data-augmentation.

 
 In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 3030–3038, -, 2021. AAAI.

 

 
 [22] 
 
Ravi Garg, Vijay Kumar Bg, Gustavo Carneiro, and Ian Reid.

 
 Unsupervised cnn for single view depth estimation: Geometry to the rescue.

 
 In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14 , pages 740–756, -, 2016. Springer, Springer.

 

 
 [23] 
 
Z Yang, P Wang, W Xu, L Zhao, and R Nevatia.

 
 Unsupervised learning of geometry with edge-aware depth-normal consistency. arxiv 2017.

 
 arXiv preprint arXiv:1711.03665 , 2(5):2, 2017.

 

 
 [24] 
 
Jiayu Yang, Jose M Alvarez, and Miaomiao Liu.

 
 Self-supervised learning of depth inference for multi-view stereo.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 7526–7534, -, 2021. IEEE.

 

 
 [25] 
 
Baichuan Huang, Hongwei Yi, Can Huang, Yijia He, Jingbin Liu, and Xiao Liu.

 
 M3vsnet: Unsupervised multi-metric multi-view stereo network.

 
 In 2021 IEEE International Conference on Image Processing (ICIP) , pages 3163–3167, -, 2021. IEEE, IEEE.

 

 
 [26] 
 
Yuhua Chen, Cordelia Schmid, and Cristian Sminchisescu.

 
 Self-supervised learning with geometric constraints in monocular video: Connecting flow, depth, and camera.

 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 7063–7072, -, 2019. IEEE.

 

 
 [27] 
 
Jingchun Cheng, Yi-Hsuan Tsai, Shengjin Wang, and Ming-Hsuan Yang.

 
 Segflow: Joint learning for video object segmentation and optical flow.

 
 In Proceedings of the IEEE International Conference on Computer Vision , pages 686–695, -, 2017. IEEE.

 

 
 [28] 
 
Reza Mahjourian, Martin Wicke, and Anelia Angelova.

 
 Unsupervised learning of depth and ego-motion from monocular video using 3d geometric constraints.

 
 In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 5667–5675, -, 2018. IEEE.

 

 
 [29] 
 
Clément Godard, Oisin Mac Aodha, and Gabriel J Brostow.

 
 Unsupervised monocular depth estimation with left-right consistency.

 
 In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 270–279, -, 2017. IEEE.

 

 
 [30] 
 
David Eigen and Rob Fergus.

 
 Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture.

 
 In Proceedings of the IEEE international conference on computer vision , pages 2650–2658, -, 2015. IEEE.

 

 
 [31] 
 
Pier Luigi Dovesi, Matteo Poggi, Lorenzo Andraghetti, Miquel Martí, Hedvig Kjellström, Alessandro Pieropan, and Stefano Mattoccia.

 
 Real-time semantic stereo matching.

 
 In 2020 IEEE international conference on robotics and automation (ICRA) , pages 10780–10787, -, 2020. IEEE, IEEE.

 

 
 [32] 
 
Guorun Yang, Hengshuang Zhao, Jianping Shi, Zhidong Deng, and Jiaya Jia.

 
 Segstereo: Exploiting semantic information for disparity estimation.

 
 In Proceedings of the European conference on computer vision (ECCV) , pages 636–651, -, 2018. ECCV.

 

 
 [33] 
 
Taher Naderi, Amir Sadovnik, Jason Hayward, and Hairong Qi.

 
 Monocular depth estimation with adaptive geometric attention.

 
 In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pages 944–954, -, 2022. IEEE.

 

 
 [34] 
 
Zuria Bauer, Zuoyue Li, Sergio Orts-Escolano, Miguel Cazorla, Marc Pollefeys, and Martin R. Oswald.

 
 Nvs-monodepth: Improving monocular depth prediction with novel view synthesis.

 
 In 2021 International Conference on 3D Vision (3DV) , pages 848–858, -, 2021. IEEE.

 

 
 [35] 
 
Paul J Besl and Neil D McKay.

 
 Method for registration of 3-d shapes.

 
 In Sensor Fusion IV: Control Paradigms and Data Structures , volume 1611, pages 586–606, -, 1992. Spie, -.

 

 
 [36] 
 
Thomas Brox, Andrés Bruhn, Nils Papenberg, and Joachim Weickert.

 
 High accuracy optical flow estimation based on a theory for warping.

 
 In Computer Vision-ECCV 2004: 8th European Conference on Computer Vision, Prague, Czech Republic, May 11-14, 2004. Proceedings, Part IV 8 , pages 25–36, -, 2004. Springer, Springer.

 

 
 [37] 
 
Rui Chen, Songfang Han, Jing Xu, and Hao Su.

 
 Point-based multi-view stereo network.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 1538–1547, -, 2019. IEEE.

 

 
 [38] 
 
Shu Chen, Zhengdong Pu, Xiang Fan, and Beiji Zou.

 
 Fixing defect of photometric loss for self-supervised monocular depth estimation.

 
 IEEE Transactions on Circuits and Systems for Video Technology , 32(3):1328–1338, 2021.

 

 
 [39] 
 
Y. Chen and G. Medioni.

 
 Object modeling by registration of multiple range images.

 
 In Proceedings. 1991 IEEE International Conference on Robotics and Automation , pages 2724–2729 vol.3, -, 1991. IEEE.

 

 
 [40] 
 
Yuchao Dai, Zhidong Zhu, Zhibo Rao, and Bo Li.

 
 Mvs2: Deep unsupervised multi-view stereo with multi-view symmetry.

 
 In 2019 International Conference on 3D Vision (3DV) , pages 1–8, -, 2019. Ieee, IEEE.

 

 
 [41] 
 
Haonan Dong and Jian Yao.

 
 Patchmvsnet: Patch-wise unsupervised multi-view stereo for weakly-textured surface reconstruction, 2022.

 

 
 [42] 
 
Zheng Dong, Ke Xu, Ziheng Duan, Hujun Bao, Weiwei Xu, and Rynson W.H. Lau.

 
 Geometry-aware two-scale pifu representation for human reconstruction.

 
 In Proceedings of the 36th International Conference on Neural Information Processing Systems , NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.

 

 
 [43] 
 
Jakob Engel, Vladlen Koltun, and Daniel Cremers.

 
 Direct sparse odometry.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence , 40(3):611–625, 2018.

 

 
 [44] 
 
Rizhao Fan, Matteo Poggi, and Stefano Mattoccia.

 
 Contrastive learning for depth prediction.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 3225–3236, -, 2023. IEEE.

 

 
 [45] 
 
Pedro F Felzenszwalb and Daniel P Huttenlocher.

 
 Efficient graph-based image segmentation.

 
 International journal of computer vision , 59(2):167–181, 2004.

 

 
 [46] 
 
David F Fouhey, Abhinav Gupta, and Martial Hebert.

 
 Data-driven 3d primitives for single image understanding.

 
 In Proceedings of the IEEE International Conference on Computer Vision , pages 3392–3399, -, 2013. IEEE.

 

 
 [47] 
 
David Gallup, Jan-Michael Frahm, Philippos Mordohai, Qingxiong Yang, and Marc Pollefeys.

 
 Real-time plane-sweeping stereo with multiple sweeping directions.

 
 In 2007 IEEE Conference on Computer Vision and Pattern Recognition , pages 1–8, -, 2007. IEEE, IEEE.

 

 
 [48] 
 
Clément Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow.

 
 Digging into self-supervised monocular depth estimation.

 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 3828–3838, -, 2019. IEEE.

 

 
 [49] 
 
Longteng Guo, Jing Liu, Xinxin Zhu, Peng Yao, Shichen Lu, and Hanqing Lu.

 
 Normalized and geometry-aware self-attention network for image captioning.

 
 In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10327–10336, -, 2020. IEEE.

 

 
 [50] 
 
Tejas Khot, Shubham Agrawal, Shubham Tulsiani, Christoph Mertz, Simon Lucey, and Martial Hebert.

 
 Learning unsupervised multi-view stereopsis via robust photometric consistency, 2019.

 

 
 [51] 
 
Uday Kusupati, Shuo Cheng, Rui Chen, and Hao Su.

 
 Normal assisted stereo depth estimation.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 2189–2199, -, 2020. IEEE.

 

 
 [52] 
 
Seokju Lee, Francois Rameau, Fei Pan, and In So Kweon.

 
 Attentive and contrastive learning for joint depth and motion field estimation.

 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 4862–4871, -, 2021. IEEE.

 

 
 [53] 
 
Jingliang Li, Zhengda Lu, Yiqun Wang, Ying Wang, and Jun Xiao.

 
 Ds-mvsnet: Unsupervised multi-view stereo via depth synthesis.

 
 In Proceedings of the 30th ACM International Conference on Multimedia , pages 5593–5601, -, 2022. ACM.

 

 
 [54] 
 
Liman Liu, Fenghao Zhang, Wanjuan Su, Yuhang Qi, and Wenbing Tao.

 
 Geometric prior-guided self-supervised learning for multi-view stereo.

 
 Remote Sensing , 15(8):2109, 2023.

 

 
 [55] 
 
Liang Liu, Jiangning Zhang, Ruifei He, Yong Liu, Yabiao Wang, Ying Tai, Donghao Luo, Chengjie Wang, Jilin Li, and Feiyue Huang.

 
 Learning by analogy: Reliable supervision from transformations for unsupervised optical flow estimation.

 
 In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 6488–6497, -, 2020. IEEE.

 

 
 [56] 
 
Arijit Mallick, Jörg Stückler, and Hendrik Lensch.

 
 Learning to adapt multi-view stereo by self-supervision.

 
 In 31st British Machine Vision Virtual Conference (BMVC), 7-10 September 2020, London, UK , volume 31, pages xx–xx, -, 2020. BMVC.

 

 
 [57] 
 
Xiaojuan Qi, Renjie Liao, Zhengzhe Liu, Raquel Urtasun, and Jiaya Jia.

 
 Geonet: Geometric neural network for joint depth and surface normal estimation.

 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 283–291, -, 2018. IEEE.

 

 
 [58] 
 
Patrick Ruhkamp, Daoyi Gao, Hanzhi Chen, Nassir Navab, and Beniamin Busam.

 
 Attention meets geometry: Geometry guided spatial-temporal attention for consistent self-supervised monocular depth estimation.

 
 In 2021 International Conference on 3D Vision (3DV) , pages 837–847, -, 2021. IEEE, IEEE.

 

 
 [59] 
 
S. Rusinkiewicz and M. Levoy.

 
 Efficient variants of the icp algorithm.

 
 In Proceedings Third International Conference on 3-D Digital Imaging and Modeling , pages 145–152, -, 2001. -.

 

 
 [60] 
 
Hideo Saito and Takeo Kanade.

 
 Shape reconstruction in projective grid space from large number of images.

 
 In Proceedings. 1999 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Cat. No PR00149) , volume 2, pages 49–54, -, 1999. IEEE, IEEE.

 

 
 [61] 
 
Dongseok Shim and H Jin Kim.

 
 Learning a geometric representation for data-efficient depth estimation via gradient field and contrastive loss.

 
 In 2021 IEEE International Conference on Robotics and Automation (ICRA) , pages 13634–13640, -, 2021. IEEE, IEEE.

 

 
 [62] 
 
Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus.

 
 Indoor segmentation and support inference from rgbd images.

 
 In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12 , pages 746–760, -, 2012. Springer, Springer.

 

 
 [63] 
 
Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou.

 
 Loftr: Detector-free local feature matching with transformers.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 8922–8931, -, 2021. IEEE.

 

 
 [64] 
 
Richard Szeliski and Polina Golland.

 
 Stereo matching with transparency and matting.

 
 In Sixth International Conference on Computer Vision (IEEE Cat. No. 98CH36271) , pages 517–524, -, 1998. IEEE, IEEE.

 

 
 [65] 
 
S. Uras, F. Girosi, A. Verri, and V. Torre.

 
 A computational approach to motion perception.

 
 Biol. Cybern. , 60(2):79–87, December 1988.

 

 
 [66] 
 
Vibhas K. Vats, Sripad Joshi, David J. Crandall, Md. Alimoor Reza, and Soon-heung Jung.

 
 Gc-mvsnet: Multi-view, multi-scale, geometrically-consistent multi-view stereo.

 
 In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) , pages 3242–3252, -, January 2024. IEEE.

 

 
 [67] 
 
Kaixuan Wang, Yao Chen, Hengkai Guo, Linfu Wen, and Shaojie Shen.

 
 Geometric pretraining for monocular depth estimation.

 
 In 2020 IEEE International Conference on Robotics and Automation (ICRA) , pages 4782–4788, -, 2020. IEEE, IEEE.

 

 
 [68] 
 
Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli.

 
 Image quality assessment: from error visibility to structural similarity.

 
 IEEE transactions on image processing , 13(4):600–612, 2004.

 

 
 [69] 
 
Yu-Hui Wen, Lin Gao, Hongbo Fu, Fang-Lue Zhang, Shihong Xia, and Yong-Jin Liu.

 
 Motif-gcns with local and non-local temporal blocks for skeleton-based action recognition.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence , 45(2):2009–2023, 2022.

 

 
 [70] 
 
Hongbin Xu, Weitao Chen, Yang Liu, Zhipeng Zhou, Haihong Xiao, Baigui Sun, Xuansong Xie, and Wenxiong Kang.

 
 Semi-supervised deep multi-view stereo.

 
 In Association of Computing Machinery , MM ’23, page 4616–4625, New York, NY, USA, 2023. Association for Computing Machinery.

 

 
 [71] 
 
Hongbin Xu, Zhipeng Zhou, Yali Wang, Wenxiong Kang, Baigui Sun, Hao Li, and Yu Qiao.

 
 Digging into uncertainty in self-supervised multi-view stereo.

 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 6078–6087, -, 2021. IEEE.

 

 
 [72] 
 
Yongming Yang, Shuwei Shao, Tao Yang, Peng Wang, Zhuo Yang, Chengdong Wu, and Hao Liu.

 
 A geometry-aware deep network for depth estimation in monocular endoscopy.

 
 Eng. Appl. Artif. Intell. , 122(C), June 2023.

 

 
 [73] 
 
Wei Yin, Yifan Liu, Chunhua Shen, and Youliang Yan.

 
 Enforcing geometric constraints of virtual normal for depth prediction.

 
 In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 5684–5693, -, 2019. IEEE.

 

 
 [74] 
 
Zhichao Yin and Jianping Shi.

 
 Geonet: Unsupervised learning of dense depth, optical flow and camera pose.

 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , page 1, -, June 2018. IEEE.

 

 
 [75] 
 
Zehao Yu, Lei Jin, and Shenghua Gao.

 
 P 2 net: Patch-match and plane-regularization for unsupervised indoor depth estimation.

 
 In European Conference on Computer Vision , pages 206–222, -, 2020. Springer, Springer.

 

 
 [76] 
 
Haimei Zhao, Jing Zhang, Zhuo Chen, Bo Yuan, and Dacheng Tao.

 
 On robust cross-view consistency in self-supervised monocular depth estimation.

 
 Machine Intelligence Research , 21(3):495–513, 2024.

 

 
 [77] 
 
Shanshan Zhao, Huan Fu, Mingming Gong, and Dacheng Tao.

 
 Geometry-aware symmetric domain adaptation for monocular depth estimation.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 9788–9798, -, 2019. IEEE.

 

 
 [78] 
 
Jie Zhu, Bo Peng, Wanqing Li, Haifeng Shen, Zhe Zhang, and Jianjun Lei.

 
 Multi-view stereo with transformer, 2021.

 

 
 [79] 
 
Qingtian Zhu, Chen Min, Zizhuang Wei, Yisong Chen, and Guoping Wang.

 
 Deep learning for multi-view stereo via plane sweep: A survey, 2021.

 

 
 [80] 
 
Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang, and Long Quan.

 
 Recurrent mvsnet for high-resolution multi-view stereo depth inference.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 5525–5534, -, 2019. IEEE.

 

 
 [81] 
 
Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu.

 
 Rwkv: Reinventing rnns for the transformer era, 2023.

 

 
 [82] 
 
Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang.

 
 Deepmvs: Learning multi-view stereopsis.

 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 2821–2830, -, 2018. IEEE.

 

 
 [83] 
 
Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

 
 U-net: Convolutional networks for biomedical image segmentation.

 
 In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 , pages 234–241, -, 2015. Springer, Springer.

 

 
 [84] 
 
Jiayu Yang, Wei Mao, Jose M Alvarez, and Miaomiao Liu.

 
 Cost volume pyramid based depth inference for multi-view stereo.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 4877–4886, -, 2020. IEEE.

 

 
 [85] 
 
D. M. Greig, B. T. Porteous, and A. H. Seheult.

 
 Exact maximum a posteriori estimation for binary images.

 
 Journal of the Royal Statistical Society. Series B (Methodological) , 51(2):271–279, 1989.

 

 
 [86] 
 
Christopher M Bishop and Nasser M Nasrabadi.

 
 Pattern Recognition and Machine Learning , volume 4.

 
 Springer, -, 2006.

 

 
 [87] 
 
Simon Meister, Junhwa Hur, and Stefan Roth.

 
 Unflow: Unsupervised learning of optical flow with a bidirectional census loss.

 
 In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, page 1, -, 2018. AAAI.

 

 
 [88] 
 
A.M. Eskicioglu and P.S. Fisher.

 
 Image quality measures and their performance.

 
 IEEE Transactions on Communications , 43(12):2959–2965, 1995.

 

 
 [89] 
 
Hang Zhao, Orazio Gallo, Iuri Frosio, and Jan Kautz.

 
 Loss functions for neural networks for image processing, 2018.

 

 
 [90] 
 
Junjie Hu, Mete Ozay, Yan Zhang, and Takayuki Okatani.

 
 Revisiting single image depth estimation: Toward higher resolution maps with accurate object boundaries.

 
 In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV) , pages 1043–1051, -, 2019. IEEE.

 

 
 [91] 
 
Christopher Zach, Thomas Pock, and Horst Bischof.

 
 A duality based approach for realtime tv-l 1 optical flow.

 
 In Pattern Recognition: 29th DAGM Symposium, Heidelberg, Germany, September 12-14, 2007. Proceedings 29 , pages 214–223, -, 2007. Springer, Springer.

 

 
 [92] 
 
Peng Wang, Xiaohui Shen, Bryan Russell, Scott Cohen, Brian Price, and Alan Yuille.

 
 Surge: surface regularized geometry estimation from a single image.

 
 In Proceedings of the 30th International Conference on Neural Information Processing Systems , NIPS’16, page 172–180, Red Hook, NY, USA, 2016. Curran Associates Inc.

 

 
 [93] 
 
Karen Simonyan and Andrew Zisserman.

 
 Very deep convolutional networks for large-scale image recognition, 2015.

 

 
 [94] 
 
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander Alemi.

 
 Inception-v4, inception-resnet and the impact of residual connections on learning.

 
 In Proceedings of the AAAI conference on Artificial Intelligence , volume 31, page 1, -, 2017. AAAI.

 

 
 [95] 
 
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

 
 Deep residual learning for image recognition.

 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 770–778, -, 2016. IEEE.

 

 
 [96] 
 
Justin Johnson, Alexandre Alahi, and Li Fei-Fei.

 
 Perceptual losses for real-time style transfer and super-resolution.

 
 In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14 , pages 694–711, -, 2016. Springer, Springer.

 

 
 [97] 
 
Michael Kazhdan and Hugues Hoppe.

 
 Screened poisson surface reconstruction.

 
 ACM Transactions on Graphics (ToG) , 32(3):1–13, 2013.

 

 
 [98] 
 
Johannes L Schonberger and Jan-Michael Frahm.

 
 Structure-from-motion revisited.

 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 4104–4113, -, 2016. IEEE.

 

 
 [99] 
 
Johannes L Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys.

 
 Pixelwise view selection for unstructured multi-view stereo.

 
 In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14 , pages 501–518, -, 2016. Springer, Springer.

 

 
 [100] 
 
Connor Shorten and Taghi M Khoshgoftaar.

 
 A survey on image data augmentation for deep learning.

 
 Journal of big data , 6(1):1–48, 2019.

 

 
 [101] 
 
Armand Joulin, Francis Bach, and Jean Ponce.

 
 Multi-class cosegmentation.

 
 In 2012 IEEE Conference on Computer Vision and Pattern Recognition , pages 542–549, -, 2012. IEEE, IEEE.

 

 
 [102] 
 
Vincent Casser, Soeren Pirk, Reza Mahjourian, and Anelia Angelova.

 
 Unsupervised monocular depth and ego-motion learning with structure and semantics.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops , pages 0–0, -, 2019. IEEE.

 

 
 [103] 
 
Chris Ding, Xiaofeng He, and Horst D Simon.

 
 On the equivalence of nonnegative matrix factorization and spectral clustering.

 
 In Proceedings of the 2005 SIAM international conference on data mining , pages 606–610, -, 2005. SIAM, SIAM.

 

 
 [104] 
 
Edo Collins, Radhakrishna Achanta, and Sabine Susstrunk.

 
 Deep feature factorization for concept discovery.

 
 In Proceedings of the European Conference on Computer Vision (ECCV) , pages 336–352, -, 2018. ECCV.

 

 
 [105] 
 
Raia Hadsell, Sumit Chopra, and Yann LeCun.

 
 Dimensionality reduction by learning an invariant mapping.

 
 In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06) , volume 2, pages 1735–1742, -, 2006. IEEE, IEEE.

 

 
 [106] 
 
John Canny.

 
 A computational approach to edge detection.

 
 IEEE Transactions on Pattern Analysis and Machine Intelligence , PAMI-8(6):679–698, 1986.