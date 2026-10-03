### C0 idx=28 n=15 LLM layer=a type=single_paper_fact assigner=not_support/not_support storm_miss=True
NUGGET: Active Federated Learning (AFL) selects clients, not data samples
SRC: AFL focuses on actively selecting clients from a server perspective in FL.
REFS: R30:Active Federated Learning|in_sys=True
gold_in_sys=none topic=no info_in_abs=partly
SYSQ: Second, *budget allocation across clients*: with partial participation, each client sees a different slice of the population
WHY: AFL/R30 absent; system addresses client participation but not client-selection AFL.

### C1 idx=32 n=7 LLM layer=a type=single_paper_fact assigner=not_support/not_support storm_miss=True
NUGGET: Representation consistency improves classification performance
SRC: [R6] showed that representation consistency improves model performance on classification tasks.
REFS: R6:Exploring Consistency in Graph Representations:from Graph Ke|in_sys=False
gold_in_sys=none topic=no info_in_abs=no
SYSQ: 
WHY: System never discusses R6 or representation consistency improving classification performance.

### C2 idx=13 n=15 LLM layer=a type=single_paper_fact assigner=not_support/not_support storm_miss=True
NUGGET: GML employs class-wise queues and knowledge distillation
SRC: GML~[R20] creates class-wise queues for contrast samples and conducts knowledge distillation based on the features of a pre-trained teacher model.
REFS: R20:Long-Tailed Recognition by Mutual Information Maximization b|in_sys=False
gold_in_sys=none topic=no info_in_abs=partly
SYSQ: 
WHY: System omits GML, class-wise queues, and distillation.

### C3 idx=26 n=9 LLM layer=a type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Mutual information, ego-motion, edge features, deep learning used in LiDAR-camera calibration
SRC: Pandey et al. [R8] used mutual information between point cloud intensities and image grayscale values.
REFS: R8:Automatic targetless extrinsic calibration of a 3d lidar and|in_sys=False; R9:Motion-based calibration of multimodal sensor arrays|in_sys=False; R10:Automatic online calibration of cameras and lasers.|in_sys=True; R11:Pixel-level Extrinsic Self Calibration of High Resolution Li|in_sys=False; R12:RegNet: Multimodal Sensor Registration Using Deep Neural Net|in_sys=False; R13:CalibNet: Geometrically Supervised Extrinsic Calibration usi|in_sys=False
gold_in_sys=none topic=partial info_in_abs=partly
SYSQ: A complementary line is learning-based, in which correspondences — or the extrinsics themselves — are predicted by deep networks, reducing reliance on hand-crafted geometry and enabling calibration from natural driving data.
WHY: Specific LiDAR-camera techniques not named; system only contrasts geometry-based vs learning-based calibration generically.

### C4 idx=26 n=12 LLM layer=a type=single_paper_fact assigner=not_support/not_support storm_miss=True
NUGGET: Deep learning for radar-camera rotational calibration, stationary radars (Schöller et al.)
SRC: used deep learning to learn rotational calibration matrices but did not address translational calibration.
REFS: R18:Targetless Rotational Auto-Calibration of Radar and Camera f|in_sys=False
gold_in_sys=none topic=no info_in_abs=partly
SYSQ: Calibration of the radar–camera pair has historically been performed manually [Zhang, 2000], and recent automatic attempts have so far been offline and structure-based
WHY: Schöller et al. data-driven rotational calibration absent; system only notes recent radar-camera attempts offline/structure-based.

### C5 idx=56 n=13 LLM layer=a type=single_paper_fact assigner=not_support/not_support storm_miss=True
NUGGET: ECHO-Mamba4Rec combines bidirectional Mamba and frequency-domain filtering
SRC: Following this, ECHO-Mamba4Rec~[R28] advanced the field by combining bidirectional Mamba with frequency-domain filtering for more accurate pattern capture.
REFS: R28:EchoMamba4Rec: Harmonizing Bidirectional State Space Models |in_sys=False
gold_in_sys=none topic=no info_in_abs=yes
SYSQ: Mamba's selective SSM made the state transition input-dependent, recovering expressivity that earlier fixed SSMs lacked while retaining linear-time, hardware-friendly scanning [14].
WHY: ECHO-Mamba4Rec is uncited; bidirectional Mamba plus frequency-domain filtering is not discussed.

### C6 idx=21 n=0 LLM layer=a type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Conventional VFI uses motion estimation, blending, morphing
SRC: Conventional VFI methods rely on model-based motion estimation, blending, and morphing techniques~[R1, R2], which can be computationally intensive and prone to artifacts such as ghosting or blurring.
REFS: R1:Motion-compensated frame interpolation using bilateral motio|in_sys=False; R2:AceVFI: A Comprehensive Survey of Advances in Video Frame In|in_sys=False
gold_in_sys=none topic=partial info_in_abs=partly
SYSQ: Classical systems combined optical flow with blending, building on the Lucas–Kanade motion estimator
WHY: System covers classical VFI via Lucas-Kanade but omits R1/R2 and the morphing/blending taxonomy detail.

### C7 idx=13 n=11 LLM layer=a type=single_paper_fact assigner=not_support/not_support storm_miss=True
NUGGET: KCL integrates balanced feature space and cross-entropy discriminability
SRC: KCL~[R30] integrates balanced feature space and cross-entropy classification discriminability using K positives.
REFS: R30:Exploring balanced feature spaces for representation learnin|in_sys=False
gold_in_sys=none topic=no info_in_abs=partly
SYSQ: 
WHY: System does not mention KCL or balanced feature-space contrast.

### C8 idx=5 n=6 LLM layer=a type=single_paper_fact assigner=not_support/not_support storm_miss=True
NUGGET: Instruction-following embeddings evaluated on Triplet Alignment, STS, Clustering
SRC: They also propose Instruction Awareness Tests, which we adopt to evaluate Triplet Alignment, STS, and Clustering tasks.
REFS: R14:Answer is All You Need: Instruction-following Text Embedding|in_sys=True
gold_in_sys=none topic=no info_in_abs=no
SYSQ: Benchmarking on BEIR [Thakur et al., 2021] and MTEB [Muennighoff et al., 2023] has shown that no single model dominates all semantic tasks
WHY: InBedder's Instruction Awareness Tests and its three evaluation tasks are absent from system.

### C9 idx=26 n=10 LLM layer=a type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Regnet, CalibNet, CalibRCNN, LCCNet: deep learning for LiDAR-camera calibration
SRC: Regnet [R12] and CalibNet [R13] employed deep learning to match features and regress calibration parameters.
REFS: R12:RegNet: Multimodal Sensor Registration Using Deep Neural Net|in_sys=False; R13:CalibNet: Geometrically Supervised Extrinsic Calibration usi|in_sys=False; R14:Calibrcnn: Calibrating camera and lidar by recurrent convolu|in_sys=True; R16:LCCNet: LiDAR and Camera Self-Calibration using Cost Volume |in_sys=True
gold_in_sys=none topic=partial info_in_abs=partly
SYSQ: A complementary line is learning-based, in which correspondences — or the extrinsics themselves — are predicted by deep networks, reducing reliance on hand-crafted geometry and enabling calibration from natural driving data.
WHY: RegNet/CalibNet/CalibRCNN/LCCNet not named; system only mentions learning-based LiDAR-camera calibration generically.

### C10 idx=56 n=1 LLM layer=a type=cross_paper_comparison assigner=not_support/not_support storm_miss=True
NUGGET: TransRec and matrix factorization struggled with multiple behaviors, long sequences
SRC: Early approaches like TransRec~[R15] and matrix factorization methods~[R16] focused on modeling user-item interactions through conventional data mining techniques, but they struggled with capturing multiple user behaviors and faced efficiency challenges
REFS: R15:Dynamic Memory based Attention Network for Sequential Recomm|in_sys=True; R16:Content-boosted Matrix Factorization Techniques for Recommen|in_sys=False
gold_in_sys=none topic=no info_in_abs=no
SYSQ: Classical collaborative filtering established item-based similarity [1] and the pointwise pairwise (BPR) ranking objective that most modern recommenders still use [2].
WHY: System omits TransRec and matrix factorization; their early limitations are neither cited nor discussed.

### C11 idx=12 n=4 LLM layer=a type=single_paper_fact assigner=partial_support/partial_support storm_miss=True
NUGGET: Early methods used spatial and temporal scaling for event downsampling
SRC: Early works explored event downsampling in spatial and temporal dimensions by scaling event coordinates and timestamps, adapting the sampling strategy to the dataset [R1].
REFS: R1:Spatial and temporal downsampling in event-based visual clas|in_sys=False
gold_in_sys=none topic=partial info_in_abs=partly
SYSQ: The strategies in practice are fixed heuristics: uniform-rate decimation, spatial or temporal filtering, and threshold-based rejection of weak or noisy events
WHY: R1 not cited; system lists spatial/temporal filtering generically but omits R1's coordinate/timestamp scaling and dataset adaptation.

### C12 idx=26 n=3 LLM layer=c1 type=taxonomy assigner=partial_support/partial_support storm_miss=False
NUGGET: Deep learning enables powerful feature extraction for online calibration
SRC: The rapid development of deep learning has demonstrated neural networks' powerful feature extraction capabilities.
REFS: 
gold_in_sys=na topic=partial info_in_abs=na
SYSQ: A complementary line is learning-based, in which correspondences — or the extrinsics themselves — are predicted by deep networks, reducing reliance on hand-crafted geometry and enabling calibration from natural driving data.
WHY: Deep learning for calibration is discussed, but not its feature-extraction capability for online radar-camera calibration.

### C13 idx=21 n=2 LLM layer=c1 type=taxonomy assigner=not_support/not_support storm_miss=True
NUGGET: DAIN, RIFE, M2M, EMA-VFI are notable deep VFI models
SRC: end-to-end neural networks for VFI, such as Depth-Aware Video Frame Interpolation (DAIN)~[R3], Real-Time Intermediate Flow Estimation (RIFE)~[R4], Many-to-Many Splatting (M2M)~[R5], and EMA-VFI~[R6].
REFS: R3:Depth-Aware Video Frame Interpolation|in_sys=True; R4:Real-Time Intermediate Flow Estimation for Video Frame Inter|in_sys=True; R5:Many-to-many Splatting for Efficient Video Frame Interpolati|in_sys=False; R6:Extracting Motion and Appearance via Inter-Frame Attention f|in_sys=False
gold_in_sys=some topic=partial info_in_abs=yes
SYSQ: Early models learned temporal blending and adaptive convolutions to capture motion [Hsieh et al., 2018; Wang et al., 2018]
WHY: System discusses deep VFI generally but omits these named models, though some refs are available.

### C14 idx=36 n=5 LLM layer=c1 type=target_positioning assigner=not_support/not_support storm_miss=True
NUGGET: Edge–cloud collaboration enables selective offloading for complex LLM tasks
SRC: When a request exceeds local capacity—e.g., requires deeper reasoning, broader context, or larger knowledge—\model~ can transparently escalate the call to a cloud‑resident LLM.
REFS: 
gold_in_sys=na topic=no info_in_abs=na
SYSQ: offloading every token to the cloud is unacceptable
WHY: System mentions cloud offloading tradeoffs but not edge-cloud collaboration or selective escalation.

### C15 idx=4 n=3 LLM layer=c1 type=taxonomy assigner=partial_support/partial_support storm_miss=False
NUGGET: LLMs exhibit biases from pre-training corpora affecting task performance
SRC: LLMs can exhibit biases from their pre-training corpora that impact task performance
REFS: 
gold_in_sys=na topic=partial info_in_abs=na
SYSQ: domain-specific knowledge: a strong LLM assigns uniformly low perplexity to text from domains it has internalized and high perplexity to unfamiliar text
WHY: System notes domain-knowledge confound in perplexity but not pretraining biases affecting task performance broadly.

### C16 idx=20 n=6 LLM layer=c1 type=target_positioning assigner=not_support/not_support storm_miss=True
NUGGET: Proximal learning without constraints remains an open challenge
SRC: Proximal learning without assumptions and constraints remains an open challenge.
REFS: 
gold_in_sys=na topic=no info_in_abs=na
SYSQ: A recent line of work closes the gap between the two families by making the proximal operator itself a learned object.
WHY: System discusses proximal learning but omits this open-challenge framing.

### C17 idx=29 n=1 LLM layer=c1 type=target_positioning assigner=not_support/not_support storm_miss=True
NUGGET: Prior work does not address external analyst verification in distributed DP
SRC: However, despite the similarities in multiple aspects, they do not cover the scenario when an external data analyst needs to verify the authenticity of the data and correctness of computation
REFS: 
gold_in_sys=na topic=partial info_in_abs=na
SYSQ: Recent work on verifiable DP has begun to pursue this direction, and our work builds directly on that thread.
WHY: Gap statement about prior work is absent; system does not position against external-analyst verification.

### C18 idx=36 n=12 LLM layer=c1 type=taxonomy assigner=partial_support/not_support storm_miss=True
NUGGET: Selective cloud offloading amortizes bandwidth and compute costs
SRC: amortizes bandwidth and compute costs by invoking the cloud only for the fraction of tasks that truly need it.
REFS: 
gold_in_sys=na topic=no info_in_abs=na
SYSQ: offloading every token to the cloud is unacceptable
WHY: System mentions cloud offloading tradeoffs but not selective offloading or cost amortization.

### C19 idx=12 n=11 LLM layer=c1 type=target_positioning assigner=not_support/not_support storm_miss=True
NUGGET: Prior evaluations focus mainly on classification tasks and metrics
SRC: Finally, prior work has largely focused on simple classification datasets or introduced metrics favoring classification-based evaluation.
REFS: 
gold_in_sys=na topic=no info_in_abs=na
SYSQ: We evaluate on four representative event tasks, each with an established benchmark lineage: object classification, N-frame interpolation, event-based surface normal estimation, and object detection
WHY: System lists four evaluation tasks but never states prior work focused on classification-only benchmarks.

### C20 idx=13 n=6 LLM layer=c2 type=taxonomy assigner=partial_support/not_support storm_miss=True
NUGGET: Re-sampling and re-weighting address class imbalance but harm feature representation
SRC: Both methods promote balanced classifier learning yet unexpectedly damage the representative ability of the deep features~[R11, R12, R13, R14].
REFS: R11:Decoupling Representation and Classifier for Long-Tailed Rec|in_sys=True; R12:BBN: Bilateral-Branch Network with Cumulative Learning for L|in_sys=False; R13:Deep Long-Tailed Learning: A Survey|in_sys=True; R14:Decoupled Training for Long-Tailed Classification With Stoch|in_sys=True
gold_in_sys=some topic=partial info_in_abs=partly
SYSQ: The data-centric axis rebalances what the model sees: class-balanced loss rescales each class's contribution by its effective sample size
WHY: System covers rebalancing but not shared damage to feature representation.

### C21 idx=21 n=7 LLM layer=c2 type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Diffusion-based methods can introduce artifacts, increase training time
SRC: However, while effective on low-quality reconstructions, these methods introduce artifacts to high-quality reconstructions and substantially extend the training time, further prolonging an already lengthy process~[R13].
REFS: R11:DiffusioNeRF: Regularizing Neural Radiance Fields with Denoi|in_sys=False; R12:3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting wit|in_sys=False; R13:ViewCrafter: Taming Video Diffusion Models for High-fidelity|in_sys=False
gold_in_sys=none topic=partial info_in_abs=partly
SYSQ: These generative priors can synthesize plausible views of unseen geometry, but they are conditioned on text or a single image and are free to invent content that was never observed.
WHY: System notes generative priors can invent content but does not state the artifact/training-time limitation of diffusion-based enhancement methods.

### C22 idx=20 n=8 LLM layer=c2 type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Iterative algorithms use hand-crafted regularizers and proximal algorithms (ISTA, AMP, HQS, ADMM)
SRC: Iterative algorithms employ hand-crafted regularizers, e.g., sparsity, total variation, non-local low rank, with a proximal algorithm, e.g., ISTA, AMP, HQS, and ADMM.
REFS: R5:Gradient projection for sparse reconstruction: Application t|in_sys=False; R6:An efficient algorithm for compressed MR imaging using total|in_sys=False; R12:Compressive sensing via nonlocal low-rank regularization|in_sys=False; R38:Generalized Alternating Projection Based Total Variation Min|in_sys=False; R9:A fast iterative shrinkage-thresholding algorithm for linear|in_sys=True; R13:From Denoising to Compressed Sensing|in_sys=True; R39:Nonlinear image recovery with half-quadratic regularization|in_sys=False; R11:Alternating Direction Algorithms for {$\ell_{1}$}-Problems i|in_sys=False
gold_in_sys=some topic=partial info_in_abs=partly
SYSQ: iterative first-order methods — in particular ISTA [8] and its accelerated variant FISTA [9] — became the standard model-based solvers for SPI at low sampling budgets.
WHY: System covers ISTA/ADMM/HQS/TV individually but omits combined taxonomy and AMP/nonlocal low-rank.

### C23 idx=34 n=0 LLM layer=c2 type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Uncertainty quantification is lacking in most forecasting models
SRC: One drawback of the above-mentioned methods is the lack of uncertainty quantification.
REFS: 
gold_in_sys=na topic=partial info_in_abs=na
SYSQ: A substantial line of work quantifies what neural forecasts do not say.
WHY: Discusses uncertainty quantification but not its general absence in forecasting models.

### C24 idx=36 n=2 LLM layer=c2 type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Model-level optimizations: quantization, pruning, distillation, architecture search
SRC: To address edge resource constraints, techniques such as model architecture search~[R19, R20, R21], quantization~[R22, R23, R17, R24, R25], pruning~[R26, R24, R27, R28], and knowledge distillation~[R29, R30] have been proposed.
REFS: R19:LLM Performance Predictors are good initializers for Archite|in_sys=False; R20:New Solutions on LLM Acceleration, Optimization, and Applica|in_sys=False; R21:Optimizing LLM Queries in Relational Data Analytics Workload|in_sys=False; R22:Quantized neural networks: Training neural networks with low|in_sys=True; R23:LLM.int8(): 8-bit Matrix Multiplication for Transformers at |in_sys=True; R17:Just-in-time Quantization with Processing-In-Memory for Effi|in_sys=False; R24:Deja Vu: Contextual Sparsity for Efficient LLMs at Inference|in_sys=True; R25:SmoothQuant: Accurate and Efficient Post-Training Quantizati|in_sys=True; R26:Pruning convolutional neural networks for resource efficient|in_sys=True; R27:Pruner-Zero: Evolving Symbolic Pruning Metric From Scratch f|in_sys=False; R28:SparseGPT: Massive Language Models Can Be Accurately Pruned |in_sys=True; R29:DistiLLM: Towards Streamlined Distillation for Large Languag|in_sys=False; R30:MiniLLM: Knowledge Distillation of Large Language Models|in_sys=True
gold_in_sys=some topic=partial info_in_abs=yes
SYSQ: Post-training quantization is the dominant lever for shrinking LLM footprint and arithmetic cost.
WHY: System covers quantization and pruning but omits knowledge distillation and architecture search.

### C25 idx=7 n=7 LLM layer=c2 type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Concept removal uses Representation Engineering and Contrastive Activation Addition
SRC: concept removal techniques [R21, R22, R23, R24] such as Representation Engineering and Contrastive Activation Addition [R13, R18]
REFS: R21:A Geometric Notion of Causal Probing|in_sys=False; R22:Better Hit the Nail on the Head than Beat around the Bush: R|in_sys=False; R23:Null It Out: Guarding Protected Attributes by Iterative Null|in_sys=False; R24:LEACE: Perfect linear concept erasure in closed form|in_sys=False; R13:Representation Engineering: A Top-Down Approach to AI Transp|in_sys=True; R18:Steering Llama 2 via Contrastive Activation Addition|in_sys=True
gold_in_sys=some topic=partial info_in_abs=no
SYSQ: The Representation Engineering (RepE) framework (Zou et al., 2023) treats a behavior as a direction in activation space, estimated from contrastive inputs and applied as a small rank-one update;
WHY: System covers RepE/activation addition but omits CAA and the concept-removal grouping; no relational statement formed.

### C26 idx=7 n=9 LLM layer=c2 type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Mechanistic interpretability uses sparse autoencoders, weight analysis, circuit analysis
SRC: mechanistic interpretability approaches leveraging sparse autoencoders [R32, R33, R34], weight-based analysis [R35], and circuit analysis [R36, R37]
REFS: R32:Towards Monosemanticity: Decomposing Language Models With Di|in_sys=True; R33:Scaling Monosemanticity: Extracting Interpretable Features f|in_sys=True; R34:Sparse Autoencoders Find Highly Interpretable Features in La|in_sys=True; R35:Bilinear MLPs enable weight-based mechanistic interpretabili|in_sys=False; R36:A Mathematical Framework for Transformer Circuits|in_sys=False; R37:Does Circuit Analysis Interpretability Scale? Evidence from |in_sys=False
gold_in_sys=some topic=partial info_in_abs=yes
SYSQ: Sparse autoencoders (SAEs) offer an alternative route: dictionary learning over residual-stream activations yields near-monosemantic features, some of which map to interpretable behaviors
WHY: SAEs covered, but weight-based and circuit-analysis parts of the mechanistic-interpretability taxonomy are missing.

### C27 idx=13 n=3 LLM layer=c2 type=taxonomy assigner=partial_support/partial_support storm_miss=False
NUGGET: Contrastive learning is widely used in long-tailed recognition
SRC: Recently, contrastive learning has become prevalent in long-tailed recognition~[R18, R19, R20, R27, R13, R28, R29].
REFS: R18:Generalized Parametric Contrastive Learning|in_sys=False; R19:Balanced Contrastive Learning for Long-Tailed Visual Recogni|in_sys=True; R20:Long-Tailed Recognition by Mutual Information Maximization b|in_sys=False; R27:Probabilistic Contrastive Learning for Long-Tailed Visual Re|in_sys=True; R13:Deep Long-Tailed Learning: A Survey|in_sys=True; R28:Fairness-aware contrastive learning with partially annotated|in_sys=False; R29:Subclass-balancing Contrastive Learning for Long-tailed Reco|in_sys=True
gold_in_sys=some topic=yes info_in_abs=partly
SYSQ: The work closest to ours analyzes the contrastive loss for long-tailed recognition and shows that a standard contrastive objective improves head classes while degrading tail classes
WHY: System covers contrastive long-tail work but does not state prevalence/taxonomy.

### C28 idx=32 n=9 LLM layer=b type=single_paper_fact assigner=not_support/not_support storm_miss=True
NUGGET: Activation similarity reveals convergence dynamics during pre-training
SRC: [R9] examined the convergence dynamics of activations by comparing activation similarities across training steps for each layer during the pre-train stage
REFS: R9:Tending Towards Stability: Convergence Challenges in Small L|in_sys=True
gold_in_sys=all topic=no info_in_abs=no
SYSQ: 
WHY: R9's activation-similarity convergence analysis is not in its abstract; system never discusses it.

### C29 idx=28 n=5 LLM layer=b type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Uncertainty-based AL uses prediction uncertainty, margin, or entropy
SRC: which can be measured by the predicted probability [R4, R5], the margin between the probability of two classes [R6], or the entropy of prediction [R7, R8]
REFS: R4:Heterogeneous uncertainty sampling for supervised learning|in_sys=False; R5:A Sequential Algorithm for Training Text Classifiers|in_sys=False; R6:Active Learning for Multi-class Image Classification|in_sys=False; R7:Latent structured active learning|in_sys=False; R8:Active learning: Synthesis lectures on artificial intelligen|in_sys=True
gold_in_sys=some topic=partial info_in_abs=partly
SYSQ: Uncertainty sampling — querying samples on which the current model is least confident — is among the most widely used strategies
WHY: Settles cited but entropy/margin/probability measures not in abstract or system; other refs uncited.

### C30 idx=11 n=11 LLM layer=b type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Manual graph designs limit efficiency in modeling long-range constraints
SRC: However, this manual design and reliance on graph convolution limits the efficiency of learning long-range constraints.
REFS: R21:Learning Pose Grammar to Encode Human Body Configuration for|in_sys=True; R7:Modulated graph convolutional network for {3D} human pose es|in_sys=True
gold_in_sys=all topic=partial info_in_abs=no
SYSQ: Its weakness is that it is likewise only as long-range as the graph diameter allows.
WHY: Manual design limitation not in abstracts; system misses this specific detail.

### C31 idx=4 n=6 LLM layer=b type=single_paper_fact assigner=partial_support/partial_support storm_miss=True
NUGGET: Min et al. found limited impact of random retrievers in some settings
SRC: While [R5] found limited performance impact with random retrievers for certain LLM-dataset combinations
REFS: R5:Rethinking the Role of Demonstrations: What Makes In-Context|in_sys=True
gold_in_sys=all topic=partial info_in_abs=partly
SYSQ: a large-scale analysis concluded that task format and label quality often matter more than the specific examples selected [Min et al., 2022]
WHY: System cites Min et al., but the random-retriever limited-impact detail is not in the abstract and not stated.

### C32 idx=36 n=0 LLM layer=x type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: LLMs are memory-, compute-, and energy-intensive, often requiring cloud inference
SRC: LLMs are highly memory-, compute-, and energy-intensive~[R1, R2, R3, R4, R5, R6], driving most "billion-parameter" inference to the cloud~[R7, R8, R9, R10, R11, R12, R13].
REFS: R1:{GPT-4} Technical Report|in_sys=True; R2:PaLM: Scaling Language Modeling with Pathways|in_sys=False; R3:LLaMA: Open and Efficient Foundation Language Models|in_sys=True; R4:Llama 2: Open Foundation and Fine-Tuned Chat Models|in_sys=False; R5:Claude: A Family of AI Models|in_sys=False; R6:Gemma: Open Models Based on Gemini Research and Technology|in_sys=True; R7:AI Index Report|in_sys=False; R8:Apple Intelligence|in_sys=False; R9:PowerInfer: Fast Large Language Model Serving with a Consume|in_sys=False; R10:Mobile Foundation Model as Firmware|in_sys=False; R11:Drive Like a Human: Rethinking Autonomous Driving with Large|in_sys=False; R12:BAT: Behavior-Aware Human-Like Trajectory Prediction for Aut|in_sys=False; R13:Creating Large Language Models on Your Laptop|in_sys=False
gold_in_sys=none topic=yes info_in_abs=no
SYSQ: these models still exceed the storage, memory bandwidth, and power envelopes of off-the-shelf edge hardware by wide margins.
WHY: System states LLMs exceed edge resources and discusses cloud offloading/latency tradeoffs.

### C33 idx=13 n=1 LLM layer=x type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Contrastive learning enhances representation robustness via positive and negative pairs
SRC: Contrastive learning has gained widespread adoption in self-supervised learning to enhance representation robustness by contrasting positive and negative pairs with augmented views~[R22, R23, R24, R25].
REFS: R22:Momentum Contrast for Unsupervised Visual Representation Lea|in_sys=True; R23:A Simple Framework for Contrastive Learning of Visual Repres|in_sys=True; R24:Bootstrap your own latent: A new approach to self-supervised|in_sys=True; R25:Exploring Simple Siamese Representation Learning|in_sys=True
gold_in_sys=all topic=yes info_in_abs=partly
SYSQ: Contrastive learning learns representations by pulling together different views of the same instance while pushing apart unrelated instances.
WHY: System states contrastive learning pulls views together and pushes unrelated apart, covering mechanism.

### C34 idx=5 n=2 LLM layer=x type=target_positioning assigner=partial_support/partial_support storm_miss=False
NUGGET: Re-encoding leads to high computational overhead and latency
SRC: Yet, both methods require re-encoding the entire corpus for each new instruction, resulting in notable computational overhead and latency, especially for large-scale datasets.
REFS: R13:One Embedder, Any Task: Instruction-Finetuned Text Embedding|in_sys=True; R14:Answer is All You Need: Instruction-following Text Embedding|in_sys=True
gold_in_sys=some topic=yes info_in_abs=no
SYSQ: A consequence is that any new or modified instruction requires the entire corpus to be re-encoded — a cost that becomes prohibitive on large collections
WHY: System states re-encoding cost becomes prohibitive, so the missed judgement is a false negative.

### C35 idx=12 n=2 LLM layer=x type=taxonomy assigner=partial_support/partial_support storm_miss=True
NUGGET: Most prior methods rely on fixed heuristics or task-specific optimizations
SRC: Despite these advances, most existing methods rely on fixed heuristics or task-specific optimizations, limiting their adaptability across different applications.
REFS: 
gold_in_sys=na topic=yes info_in_abs=na
SYSQ: The strategies in practice are fixed heuristics: uniform-rate decimation, spatial or temporal filtering, and threshold-based rejection of weak or noisy events
WHY: System states prior strategies are fixed heuristics and lack adaptability, matching the nugget despite not naming task-specific optimizations.

### C36 idx=27 n=5 LLM layer=d type=target_positioning assigner=partial_support/not_support storm_miss=True
NUGGET: Birdie is the first to use DSI for table discovery
SRC: To the best of our knowledge, \textsc{Birdie} is the first attempt to perform table discovery using DSI
REFS: R18:Transformer Memory as a Differentiable Search Index|in_sys=True
gold_in_sys=all topic=partial info_in_abs=no
SYSQ: Birdie adopts the same design principle for the table domain: it assigns each table a prefix-aware identifier
WHY: Novelty claim about Birdie being first; system discusses DSI for tables but omits the 'first' positioning.
